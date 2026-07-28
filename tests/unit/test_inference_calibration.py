from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from robata.adapters.sqlite_inference_evidence import (
    _APPLICATION_ID,
    _SCHEMA_VERSION,
    _V1_SCHEMA_STATEMENTS,
    SQLiteInferenceEvidenceLedger,
    SQLiteInferenceEvidenceLedgerError,
)
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.calibration import (
    AcceptedInferenceCalibrationBridge,
    CalibrationApplicability,
    CalibrationArtifact,
    CalibrationAssociation,
    CalibrationAssociationOutcome,
    CalibrationBinding,
    CalibrationFittingMethod,
    CalibrationGroupedSplitLineage,
    CalibrationPolicyDecision,
    CalibrationScoreSource,
    CalibrationTrainingPopulation,
    accepted_calibration_score_input,
    calibration_association_id,
)
from robata.inference.models import ModelInference
from tests.unit.test_sqlite_inference_evidence import (
    NOW,
    _build_after_raw,
    _database,
    _digest,
    _Evidence,
    _intent,
    _persist_chain,
)

_SCORE_FAMILY = "provider_self_report.v1"
_RUNTIME = "runtime-v1"
_PREPROCESS = "preprocess-v1"
_VALID_UNTIL = "2026-08-20T12:00:00Z"


def _population(*, labelled_member_count: int = 8) -> CalibrationTrainingPopulation:
    return CalibrationTrainingPopulation(
        population_artifact_id="calibration-population-v1",
        population_sha256=_digest(101),
        label_set_sha256=_digest(102),
        member_count=8,
        labelled_member_count=labelled_member_count,
    )


def _split() -> CalibrationGroupedSplitLineage:
    return CalibrationGroupedSplitLineage(
        split_artifact_id="grouped-split-v1",
        split_sha256=_digest(103),
        grouping_key="recording-camera-time",
        leakage_policy_version="split-policy-1",
        development_group_sha256=_digest(104),
        calibration_group_sha256=_digest(105),
        frozen_evaluation_group_sha256=_digest(106),
    )


def _artifact(inference: ModelInference) -> CalibrationArtifact:
    applicability = CalibrationApplicability.from_inference(
        inference,
        score_family=_SCORE_FAMILY,
        runtime_revision=_RUNTIME,
        preprocess_revision=_PREPROCESS,
    )
    return CalibrationArtifact.create(
        applicability=applicability,
        fitting_method=CalibrationFittingMethod.PLATT_LOGISTIC,
        fitting_parameters={"slope": 1.5, "intercept": -0.25},
        training_population=_population(),
        grouped_split_lineage=_split(),
        fitted_at=NOW,
        valid_from=NOW,
        valid_until=_VALID_UNTIL,
        created_at=NOW,
    )


def _association(
    evidence: _Evidence,
    artifact: CalibrationArtifact | None,
    *,
    score_family: str = _SCORE_FAMILY,
    runtime_revision: str = _RUNTIME,
    evaluated_at: str = NOW,
    source_claim_ordinal: int = 0,
    policy_input_kind: Literal["RAW_SCORE", "CALIBRATED_PROBABILITY"]
    | None = "CALIBRATED_PROBABILITY",
) -> CalibrationAssociation:
    selection = evidence.selection
    terminal = evidence.terminal
    raw_score, deterministic_inputs = accepted_calibration_score_input(
        score_source=CalibrationScoreSource.ENRICHED_CLAIM_REPORTED_CONFIDENCE,
        source_claim_ordinal=source_claim_ordinal,
        inference=terminal,
        selected_output=evidence.selected,
        enriched_output=evidence.enriched,
    )
    return CalibrationAssociation.create(
        selection_id=selection.selection_id,
        inference=terminal,
        score_family=score_family,
        runtime_revision=runtime_revision,
        preprocess_revision=_PREPROCESS,
        evaluated_at=evaluated_at,
        raw_score=raw_score,
        deterministic_inputs=deterministic_inputs,
        policy_decision=(
            None
            if policy_input_kind is None
            else CalibrationPolicyDecision(
                policy_version="calibration-policy-1",
                decision="REVIEW",
                input_kind=policy_input_kind,
                threshold=0.4,
            )
        ),
        calibration_artifact=artifact,
        created_at=evaluated_at,
    )


def test_calibration_artifact_is_content_addressed_and_requires_labels(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    evidence = _persist_chain(ledger)
    artifact = _artifact(evidence.terminal)

    assert artifact.artifact_id == f"calibration-artifact:{artifact.artifact_sha256}"
    assert ledger.append_calibration_artifact(artifact) == artifact
    stored = ledger.append_calibration_artifact(artifact)
    before_calibration = stored.calibrate(0.2)
    with pytest.raises(TypeError):
        stored.fitting_parameters["slope"] = 9.0  # type: ignore[index]
    assert stored.calibrate(0.2) == before_calibration
    reloaded = ledger.get_calibration_artifact(artifact.artifact_id)
    assert reloaded is not None
    assert reloaded.calibrate(0.2) == before_calibration

    with pytest.raises(ValidationError):
        _population(labelled_member_count=0)


def test_calibration_association_is_replayable_and_never_rewrites_terminal(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    evidence = _persist_chain(ledger)
    artifact = _artifact(evidence.terminal)
    association = _association(evidence, artifact)
    before_terminal = sqlite3.connect(database)
    try:
        before_payload = before_terminal.execute(
            "SELECT payload_json FROM model_inference_terminals WHERE inference_id = ?",
            (evidence.terminal.inference_id,),
        ).fetchone()[0]
    finally:
        before_terminal.close()

    assert ledger.append_calibration_artifact(artifact) == artifact
    assert ledger.append_calibration_association(association) == association
    assert ledger.append_calibration_association(association) == association
    assert association.outcome is CalibrationAssociationOutcome.APPLIED
    assert association.raw_score == 0.8
    assert association.calibrated_probability == pytest.approx(artifact.calibrate(0.8))

    after_terminal = sqlite3.connect(database)
    try:
        after_payload = after_terminal.execute(
            "SELECT payload_json FROM model_inference_terminals WHERE inference_id = ?",
            (evidence.terminal.inference_id,),
        ).fetchone()[0]
    finally:
        after_terminal.close()
    assert after_payload == before_payload

    ledger.close()
    restarted = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    assert restarted.get_calibration_artifact(artifact.artifact_id) == artifact
    assert (
        restarted.get_calibration_association(
            evidence.selection.selection_id,
            _SCORE_FAMILY,
        )
        == association
    )


def test_missing_or_inapplicable_calibration_fails_closed_to_raw_score(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    evidence = _persist_chain(ledger)
    artifact = _artifact(evidence.terminal)
    missing = _association(
        evidence,
        None,
        score_family="provider_self_report_missing.v1",
        policy_input_kind="RAW_SCORE",
    )
    inapplicable = _association(
        evidence,
        artifact,
        runtime_revision="stale-runtime-v0",
        policy_input_kind="RAW_SCORE",
    )
    unavailable = _association(
        evidence,
        None,
        score_family="provider_self_report_unavailable.v1",
        source_claim_ordinal=1,
        policy_input_kind=None,
    )
    expired = _association(
        evidence,
        artifact,
        evaluated_at="2026-08-21T12:00:00Z",
        policy_input_kind="RAW_SCORE",
    )

    with pytest.raises(ValidationError, match="raw-score fallback policy"):
        _association(
            evidence,
            None,
            score_family="provider_self_report_bad-policy.v1",
            policy_input_kind="CALIBRATED_PROBABILITY",
        )

    assert ledger.append_calibration_association(missing) == missing
    assert ledger.append_calibration_artifact(artifact) == artifact
    assert ledger.append_calibration_association(inapplicable) == inapplicable
    assert ledger.append_calibration_association(unavailable) == unavailable
    assert missing.outcome is CalibrationAssociationOutcome.RAW_FALLBACK_MISSING_ARTIFACT
    assert missing.calibrated_probability is None
    assert inapplicable.outcome is CalibrationAssociationOutcome.RAW_FALLBACK_INAPPLICABLE
    assert inapplicable.calibrated_probability is None
    assert "runtime_revision" in inapplicable.mismatch_reasons
    assert unavailable.outcome is CalibrationAssociationOutcome.RAW_FALLBACK_UNAVAILABLE_SCORE
    assert unavailable.raw_score is None
    assert expired.outcome is CalibrationAssociationOutcome.RAW_FALLBACK_INAPPLICABLE
    assert "validity_window" in expired.mismatch_reasons


def test_calibration_association_rejects_wrong_selection_binding_or_artifact_digest(
    tmp_path: Path,
) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    evidence = _persist_chain(ledger)
    artifact = _artifact(evidence.terminal)
    association = _association(evidence, artifact)
    assert ledger.append_calibration_artifact(artifact) == artifact

    wrong_inference = association.model_copy(
        update={"inference_id": "00000000-0000-0000-0000-000000000999"}
    )
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="same inference"):
        ledger.append_calibration_association(wrong_inference)

    missing_selection_id = "00000000-0000-0000-0000-000000000998"
    unaccepted_selection = CalibrationAssociation.model_validate(
        {
            **association.model_dump(mode="python"),
            "selection_id": missing_selection_id,
            "association_id": calibration_association_id(
                selection_id=missing_selection_id,
                score_family=association.score_family,
            ),
        },
        strict=True,
    )
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="accepted selection"):
        ledger.append_calibration_association(unaccepted_selection)

    wrong_digest = association.model_copy(update={"calibration_artifact_sha256": _digest(999)})
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="identity or digest"):
        ledger.append_calibration_association(wrong_digest)


def test_ledger_rejects_forged_calibration_score_or_nonselection_timestamp(
    tmp_path: Path,
) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    evidence = _persist_chain(ledger)
    artifact = _artifact(evidence.terminal)
    association = _association(evidence, artifact)
    assert ledger.append_calibration_artifact(artifact) == artifact

    forged_score = CalibrationAssociation.create(
        selection_id=evidence.selection.selection_id,
        inference=evidence.terminal,
        score_family=association.score_family,
        runtime_revision=association.runtime_revision,
        preprocess_revision=association.preprocess_revision,
        evaluated_at=evidence.selection.selected_at,
        raw_score=0.2,
        deterministic_inputs=association.deterministic_inputs,
        policy_decision=association.policy_decision,
        calibration_artifact=artifact,
        created_at=evidence.selection.selected_at,
    )
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="raw score does not match"):
        ledger.append_calibration_association(forged_score)

    later = _association(
        evidence,
        artifact,
        evaluated_at="2026-07-21T12:00:00Z",
    )
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="timestamps must equal"):
        ledger.append_calibration_association(later)


def test_ledger_requires_selected_and_enriched_output_before_calibration_append(
    tmp_path: Path,
) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
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
    ledger.append_parsed_claim(evidence.parsed)
    artifact = _artifact(evidence.terminal)
    association = _association(evidence, artifact)
    ledger.append_calibration_artifact(artifact)

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="persisted accepted selected"):
        ledger.append_calibration_association(association)

    ledger.append_selected_output(evidence.selected)
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="persisted accepted enriched"):
        ledger.append_calibration_association(association)


def test_v1_ledger_migrates_with_append_only_calibration_tables(tmp_path: Path) -> None:
    database = _database(tmp_path)
    connection = sqlite3.connect(database)
    try:
        for statement in _V1_SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    migrated = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    migrated.close()
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (_SCHEMA_VERSION,)
        assert (
            connection.execute(
                "SELECT sql FROM sqlite_schema WHERE name = 'calibration_artifacts_no_update'"
            ).fetchone()
            is not None
        )
        assert (
            connection.execute(
                "SELECT sql FROM sqlite_schema "
                "WHERE name = 'inference_calibration_associations_no_delete'"
            ).fetchone()
            is not None
        )
    finally:
        connection.close()

    SQLiteInferenceEvidenceLedger(database, SchemaRegistry()).close()


def test_tampered_calibration_association_fails_closed_on_reopen(tmp_path: Path) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    evidence = _persist_chain(ledger)
    artifact = _artifact(evidence.terminal)
    association = _association(evidence, artifact)
    ledger.append_calibration_artifact(artifact)
    ledger.append_calibration_association(association)
    ledger.close()

    connection = sqlite3.connect(database)
    try:
        trigger_sql = connection.execute(
            """
            SELECT sql FROM sqlite_schema
            WHERE name = 'inference_calibration_associations_no_update'
            """
        ).fetchone()[0]
        connection.execute("DROP TRIGGER inference_calibration_associations_no_update")
        connection.execute(
            "UPDATE inference_calibration_associations SET outcome = 'RAW_FALLBACK_INAPPLICABLE'"
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="indexed column outcome"):
        SQLiteInferenceEvidenceLedger(database, SchemaRegistry())


def test_accepted_bridge_records_one_explicit_claim_score_with_replay_stable_time(
    tmp_path: Path,
) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    evidence = _persist_chain(ledger)
    artifact = _artifact(evidence.terminal)
    bridge = AcceptedInferenceCalibrationBridge(
        store=ledger,
        bindings=(
            CalibrationBinding(
                task=evidence.terminal.stage,
                score_family=_SCORE_FAMILY,
                runtime_revision=_RUNTIME,
                preprocess_revision=_PREPROCESS,
                score_source=CalibrationScoreSource.ENRICHED_CLAIM_REPORTED_CONFIDENCE,
                source_claim_ordinal=0,
                calibration_artifact=artifact,
            ),
        ),
    )

    first = bridge.record_accepted(
        task=evidence.terminal.stage,
        inference=evidence.terminal,
        selection=evidence.selection,
        selected_output=evidence.selected,
        enriched_output=evidence.enriched,
    )
    replayed = bridge.record_accepted(
        task=evidence.terminal.stage,
        inference=evidence.terminal,
        selection=evidence.selection,
        selected_output=evidence.selected,
        enriched_output=evidence.enriched,
    )

    assert replayed == first
    assert len(first) == 1
    association = first[0]
    assert association.outcome is CalibrationAssociationOutcome.APPLIED
    assert association.raw_score == pytest.approx(0.8)
    assert association.calibrated_probability == pytest.approx(artifact.calibrate(0.8))
    assert association.evaluated_at == evidence.selection.selected_at
    assert association.created_at == evidence.selection.selected_at
    assert association.deterministic_inputs["source_claim_ordinal"] == 0
    assert association.deterministic_inputs["score_available"] is True
    assert ledger.get_calibration_artifact(artifact.artifact_id) == artifact
    assert (
        ledger.get_calibration_association(evidence.selection.selection_id, _SCORE_FAMILY)
        == association
    )
    assert evidence.terminal.calibrated_confidence is None


def test_accepted_bridge_rejects_competing_score_sources_and_records_raw_fallbacks(
    tmp_path: Path,
) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    evidence = _persist_chain(ledger)
    unavailable_artifact = _artifact(evidence.terminal)

    with pytest.raises(ValueError, match="same calibration score family"):
        AcceptedInferenceCalibrationBridge(
            store=ledger,
            bindings=(
                CalibrationBinding(
                    task=evidence.terminal.stage,
                    score_family="duplicate-score-family.v1",
                    runtime_revision=_RUNTIME,
                    preprocess_revision=_PREPROCESS,
                    score_source=CalibrationScoreSource.ENRICHED_CLAIM_REPORTED_CONFIDENCE,
                    source_claim_ordinal=0,
                ),
                CalibrationBinding(
                    task=evidence.terminal.stage,
                    score_family="duplicate-score-family.v1",
                    runtime_revision=_RUNTIME,
                    preprocess_revision=_PREPROCESS,
                    score_source=CalibrationScoreSource.ENRICHED_CLAIM_REPORTED_CONFIDENCE,
                    source_claim_ordinal=1,
                ),
            ),
        )

    missing_score_family = "provider_self_report_missing_artifact.v1"
    unavailable_score_family = "provider_self_report_unavailable_score.v1"
    bridge = AcceptedInferenceCalibrationBridge(
        store=ledger,
        bindings=(
            CalibrationBinding(
                task=evidence.terminal.stage,
                score_family=missing_score_family,
                runtime_revision=_RUNTIME,
                preprocess_revision=_PREPROCESS,
                score_source=CalibrationScoreSource.ENRICHED_CLAIM_REPORTED_CONFIDENCE,
                source_claim_ordinal=0,
            ),
            CalibrationBinding(
                task=evidence.terminal.stage,
                score_family=unavailable_score_family,
                runtime_revision=_RUNTIME,
                preprocess_revision=_PREPROCESS,
                score_source=CalibrationScoreSource.ENRICHED_CLAIM_REPORTED_CONFIDENCE,
                source_claim_ordinal=1,
                calibration_artifact=unavailable_artifact,
            ),
        ),
    )

    associations = bridge.record_accepted(
        task=evidence.terminal.stage,
        inference=evidence.terminal,
        selection=evidence.selection,
        selected_output=evidence.selected,
        enriched_output=evidence.enriched,
    )
    by_score_family = {item.score_family: item for item in associations}

    missing = by_score_family[missing_score_family]
    assert missing.outcome is CalibrationAssociationOutcome.RAW_FALLBACK_MISSING_ARTIFACT
    assert missing.raw_score == pytest.approx(0.8)
    assert missing.calibrated_probability is None
    unavailable = by_score_family[unavailable_score_family]
    assert unavailable.outcome is CalibrationAssociationOutcome.RAW_FALLBACK_UNAVAILABLE_SCORE
    assert unavailable.raw_score is None
    assert unavailable.calibrated_probability is None
    assert unavailable.calibration_artifact_id is None
    assert ledger.get_calibration_artifact(unavailable_artifact.artifact_id) is None
