from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from robata.adapters.sqlite_inference_evidence import (
    SQLiteInferenceEvidenceLedger,
    SQLiteInferenceEvidenceLedgerError,
)
from robata.contracts.schema_registry import SchemaRegistry
from tests.unit.test_sqlite_inference_evidence import _intent


def test_raw_provider_bytes_can_use_filesystem_cas_without_sqlite_duplication(
    tmp_path: Path,
) -> None:
    database = tmp_path / "inference-evidence.sqlite3"
    cas_root = tmp_path / "raw-cas"
    ledger = SQLiteInferenceEvidenceLedger(
        database,
        SchemaRegistry(),
        raw_bytes_cas_root=cas_root,
    )
    _fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)

    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id=f"offline:{intent.request_id}",
        data=raw_data,
    )

    cas_path = cas_root / stored.exact_bytes_sha256[:2] / stored.exact_bytes_sha256
    assert cas_path.read_bytes() == raw_data
    connection = sqlite3.connect(database)
    try:
        sqlite_payload = connection.execute(
            "SELECT raw_bytes FROM raw_provider_responses WHERE artifact_id = ?",
            (stored.artifact_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert sqlite_payload != raw_data
    assert sqlite_payload == f"robata-cas-v1:{stored.exact_bytes_sha256}".encode()

    restarted = SQLiteInferenceEvidenceLedger(
        database,
        SchemaRegistry(),
        raw_bytes_cas_root=cas_root,
    )
    assert restarted.get(stored.artifact_id) == stored


def test_missing_or_tampered_raw_cas_object_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "inference-evidence.sqlite3"
    cas_root = tmp_path / "raw-cas"
    ledger = SQLiteInferenceEvidenceLedger(
        database,
        SchemaRegistry(),
        raw_bytes_cas_root=cas_root,
    )
    _fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id=f"offline:{intent.request_id}",
        data=raw_data,
    )
    cas_path = cas_root / stored.exact_bytes_sha256[:2] / stored.exact_bytes_sha256
    cas_path.write_bytes(b"tampered")

    with pytest.raises(
        SQLiteInferenceEvidenceLedgerError,
        match="CAS object failed exact digest verification",
    ):
        SQLiteInferenceEvidenceLedger(
            database,
            SchemaRegistry(),
            raw_bytes_cas_root=cas_root,
        )


def test_cas_backed_database_requires_the_same_cas_root_on_restart(tmp_path: Path) -> None:
    database = tmp_path / "inference-evidence.sqlite3"
    ledger = SQLiteInferenceEvidenceLedger(
        database,
        SchemaRegistry(),
        raw_bytes_cas_root=tmp_path / "raw-cas",
    )
    _fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    ledger.append(
        request_id=intent.request_id,
        provider_request_id=f"offline:{intent.request_id}",
        data=raw_data,
    )

    with pytest.raises(
        SQLiteInferenceEvidenceLedgerError,
        match="requires its configured CAS root",
    ):
        SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
