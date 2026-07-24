from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from robata.adapters.sqlite_inference_evidence import (
    SQLiteInferenceEvidenceLedger,
    SQLiteInferenceEvidenceLedgerError,
)
from robata.contracts.hashing import exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.evidence import (
    InferenceEvidenceStore,
    InferenceEvidenceStoreError,
    InMemoryInferenceEvidenceStore,
)
from tests.unit.test_sqlite_inference_evidence import (
    _build_after_raw,
    _Evidence,
    _intent,
    _stable_uuid,
)


def _prepared_sqlite_lineage(
    tmp_path: Path,
) -> tuple[Path, SQLiteInferenceEvidenceLedger, _Evidence]:
    database = tmp_path / "inference-evidence.sqlite3"
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id=f"offline:{intent.request_id}",
        data=raw_data,
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    )
    ledger.append_terminal(evidence.terminal)
    ledger.append_selection(evidence.selection)
    return database, ledger, evidence


def _lineage_counts(database: Path) -> tuple[int, int, int]:
    connection = sqlite3.connect(database)
    try:
        return tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "parsed_provider_claims",
                "selected_attempt_outputs",
                "enriched_provider_outputs",
            )
        )  # type: ignore[return-value]
    finally:
        connection.close()


def test_in_memory_accepted_lineage_checks_all_conflicts_before_writing() -> None:
    fixture, intent, raw_data = _intent()
    raw_artifact_id = _stable_uuid(
        "raw-provider-response",
        intent.request_id,
        exact_bytes_sha256(raw_data),
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        raw_artifact_id,
        "offline:fixture",
    )
    store = InMemoryInferenceEvidenceStore()
    assert isinstance(store, InferenceEvidenceStore)
    conflicting_enriched = evidence.enriched.model_copy(
        update={"created_at": "2026-07-20T12:00:01Z"}
    )
    store.append_enriched_output(conflicting_enriched)

    with pytest.raises(
        InferenceEvidenceStoreError,
        match="enriched output identity has conflicting content",
    ):
        store.append_accepted_lineage(
            evidence.parsed,
            evidence.selected,
            evidence.enriched,
        )

    assert store.get_parsed_claim(evidence.parsed.artifact_id) is None
    assert store.get_selected_output(evidence.selected.selection_id) is None
    assert store.get_enriched_output(evidence.enriched.artifact_id) == conflicting_enriched


def test_sqlite_accepted_lineage_is_exactly_replayable(tmp_path: Path) -> None:
    database, ledger, evidence = _prepared_sqlite_lineage(tmp_path)

    expected = (evidence.parsed, evidence.selected, evidence.enriched)
    assert ledger.append_accepted_lineage(*expected) == expected
    assert ledger.append_accepted_lineage(*expected) == expected
    assert _lineage_counts(database) == (1, 1, 1)

    restarted = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    assert restarted.append_accepted_lineage(*expected) == expected
    assert _lineage_counts(database) == (1, 1, 1)


def test_sqlite_accepted_lineage_rolls_back_earlier_rows_on_late_failure(
    tmp_path: Path,
) -> None:
    database, ledger, evidence = _prepared_sqlite_lineage(tmp_path)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            CREATE TRIGGER reject_enriched_fixture
            BEFORE INSERT ON enriched_provider_outputs
            BEGIN
                SELECT RAISE(ABORT, 'injected enriched failure');
            END
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteInferenceEvidenceLedgerError):
        ledger.append_accepted_lineage(
            evidence.parsed,
            evidence.selected,
            evidence.enriched,
        )

    assert _lineage_counts(database) == (0, 0, 0)
    assert ledger.get_parsed_claim(evidence.parsed.artifact_id) is None
    assert ledger.get_selected_output(evidence.selected.selection_id) is None
    assert ledger.get_enriched_output(evidence.enriched.artifact_id) is None


def test_sqlite_selected_terminal_rolls_back_as_one_unit_on_selection_failure(
    tmp_path: Path,
) -> None:
    database = tmp_path / "inference-evidence.sqlite3"
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id=f"offline:{intent.request_id}",
        data=raw_data,
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            CREATE TRIGGER reject_atomic_selection
            BEFORE INSERT ON inference_attempt_selections
            BEGIN
                SELECT RAISE(ABORT, 'injected selection failure');
            END
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteInferenceEvidenceLedgerError):
        ledger.append_terminal_and_selection(
            evidence.terminal,
            evidence.selection,
        )

    connection = sqlite3.connect(database)
    try:
        counts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "model_inference_terminals",
                "raw_provider_artifacts",
                "inference_attempt_selections",
            )
        )
        dispatch_facts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("inference_intents", "raw_provider_responses")
        )
    finally:
        connection.close()
    assert counts == (0, 0, 0)
    assert dispatch_facts == (1, 1)
    assert ledger.get_terminal(evidence.terminal.inference_id) is None
    assert (
        ledger.get_selection(
            evidence.selection.logical_invocation_id,
            evidence.selection.policy_version,
        )
        is None
    )
