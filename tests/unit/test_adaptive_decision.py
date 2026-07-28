from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.sampling.adaptive import AdaptiveCoveragePolicy, AdaptiveUpgradeReason
from robata.sampling.adaptive_decision import (
    AcceptedAdaptiveEvidenceBinding,
    AdaptiveDecisionBaseBinding,
    AdaptiveDecisionSourceBinding,
    AdaptiveNoAdditionalWorkProof,
    AdaptiveNoAdditionalWorkProofKind,
    AdaptiveSamplingDecision,
    AdaptiveSamplingDecisionOutcome,
    AdaptiveTriggerEvidenceKind,
    AdaptiveTriggerProvenance,
    accepted_selection_evidence_sha256,
    adaptive_sampling_decision_projection,
    build_adaptive_late_feedback_audit,
    build_adaptive_sampling_decision,
)
from robata.sampling.adaptive_decision_store import (
    AdaptiveDecisionStoreConflict,
    SQLiteAdaptiveDecisionStore,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _base() -> AdaptiveDecisionBaseBinding:
    return AdaptiveDecisionBaseBinding(
        sampling_plan_sha256=_digest(1),
        package_set_id="00000000-0000-4000-8000-000000000001",
        package_set_member_manifest_sha256=_digest(2),
        package_set_split_plan_sha256=_digest(3),
    )


def _accepted() -> AcceptedAdaptiveEvidenceBinding:
    return AcceptedAdaptiveEvidenceBinding(
        selection_id="00000000-0000-4000-8000-000000000002",
        selection_decision_logical_key="inference-attempt-selection:" + _digest(4),
        selected_output_sha256=_digest(5),
        enriched_output_artifact_id="00000000-0000-4000-8000-000000000003",
        enriched_output_semantic_sha256=_digest(6),
    )


def _source() -> AdaptiveDecisionSourceBinding:
    return AdaptiveDecisionSourceBinding(
        source_content_sha256=_digest(7),
        camera_mapping_semantic_sha256=_digest(8),
        alignment_semantic_sha256=_digest(9),
    )


def _interval() -> NanosecondInterval:
    return NanosecondInterval(start_ns=0, end_ns=2_000_000_000)


def _policy(
    *,
    max_targets_per_camera: int = 4,
    max_targets_total: int = 24,
) -> AdaptiveCoveragePolicy:
    return AdaptiveCoveragePolicy(
        version="adaptive-coverage-v1",
        base_rate_num=1,
        base_target_budget_per_camera=1,
        context_offsets_ns=(-100_000_000, 100_000_000),
        max_targets_per_camera=max_targets_per_camera,
        max_targets_total=max_targets_total,
    )


def _trigger(
    *,
    timestamp_ns: int = 500_000_000,
    locator: str = "claim/0",
    reason: AdaptiveUpgradeReason = AdaptiveUpgradeReason.COARSE_UNCERTAINTY,
) -> AdaptiveTriggerProvenance:
    return AdaptiveTriggerProvenance(
        camera_id=CameraId.CAM_01,
        trigger_timestamp_ns=timestamp_ns,
        reason=reason,
        evidence_kind=AdaptiveTriggerEvidenceKind.ENRICHED_OUTPUT,
        evidence_sha256=_accepted().enriched_output_semantic_sha256,
        evidence_locator=locator,
    )


def _decision(*triggers: AdaptiveTriggerProvenance):
    return build_adaptive_sampling_decision(
        base=_base(),
        accepted_evidence=_accepted(),
        source=_source(),
        effective_interval=_interval(),
        policy=_policy(),
        triggers=triggers,
    )


def test_decision_normalizes_duplicate_and_out_of_order_triggers() -> None:
    first = _trigger(locator="claim/a")
    second = _trigger(
        timestamp_ns=750_000_000,
        locator="claim/b",
        reason=AdaptiveUpgradeReason.CROSS_CAMERA_DISAGREEMENT,
    )

    decision = _decision(second, first, first)
    replay = _decision(first, second)

    assert decision == replay
    assert decision.outcome is AdaptiveSamplingDecisionOutcome.ADDITIONAL_TARGETS_SCHEDULED
    assert tuple(item.evidence_locator for item in decision.triggers) == ("claim/a", "claim/b")
    assert tuple(item.ordinal for item in decision.incremental_targets) == tuple(
        range(len(decision.incremental_targets))
    )
    assert all(
        not item.target_ns < decision.effective_interval.start_ns
        and item.target_ns < decision.effective_interval.end_ns
        for item in decision.incremental_targets
    )
    assert decision.coverage_accounting is not None
    assert decision.coverage_accounting.incremental_target_count == len(
        decision.incremental_targets
    )


def test_direct_model_validation_rejects_forged_content_addressed_targets() -> None:
    decision = _decision(_trigger())
    values = decision.model_dump(mode="python")
    forged_targets = list(values["incremental_targets"])
    forged_targets[0] = {
        **forged_targets[0],
        "target_ns": forged_targets[0]["target_ns"] + 1,
    }
    values["incremental_targets"] = tuple(forged_targets)
    draft = AdaptiveSamplingDecision.model_construct(
        **{
            **values,
            "decision_id": "pending",
            "semantic_sha256": "0" * 64,
        }
    )
    digest = semantic_sha256(adaptive_sampling_decision_projection(draft))
    values["decision_id"] = f"adaptive-sampling-decision:{digest}"
    values["semantic_sha256"] = digest

    with pytest.raises(ValueError, match="incremental targets do not match"):
        AdaptiveSamplingDecision.model_validate(values, strict=True)


def test_trigger_evidence_must_bind_the_accepted_branch() -> None:
    invalid = AdaptiveTriggerProvenance(
        camera_id=CameraId.CAM_01,
        trigger_timestamp_ns=500_000_000,
        reason=AdaptiveUpgradeReason.COARSE_UNCERTAINTY,
        evidence_kind=AdaptiveTriggerEvidenceKind.SELECTED_OUTPUT,
        evidence_sha256=_digest(999),
        evidence_locator="claim/invalid",
    )

    with pytest.raises(ValueError, match="accepted selected output"):
        _decision(invalid)

    accepted = _accepted()
    selection_trigger = AdaptiveTriggerProvenance(
        camera_id=CameraId.CAM_01,
        trigger_timestamp_ns=500_000_000,
        reason=AdaptiveUpgradeReason.EVENT_CANDIDATE,
        evidence_kind=AdaptiveTriggerEvidenceKind.ACCEPTED_SELECTION,
        evidence_sha256=accepted_selection_evidence_sha256(accepted),
        evidence_locator=accepted.selection_id,
    )
    assert _decision(selection_trigger).triggers == (selection_trigger,)


def test_explicit_non_trigger_outcomes_never_claim_no_events() -> None:
    abstained = build_adaptive_sampling_decision(
        base=_base(),
        accepted_evidence=_accepted(),
        source=_source(),
        effective_interval=_interval(),
        policy=_policy(),
        no_trigger_outcome=AdaptiveSamplingDecisionOutcome.UPSTREAM_ABSTAINED,
        outcome_detail="accepted model branch abstained from an upgrade signal",
    )
    proof = AdaptiveNoAdditionalWorkProof(
        proof_kind=AdaptiveNoAdditionalWorkProofKind.DENSE_QA_COARSE_COMPLETE,
        evidence_artifact_id="00000000-0000-4000-8000-000000000004",
        evidence_sha256=_digest(10),
        policy_version="dense-qa-completion-v1",
    )
    complete = build_adaptive_sampling_decision(
        base=_base(),
        accepted_evidence=_accepted(),
        source=_source(),
        effective_interval=_interval(),
        policy=_policy(),
        no_trigger_outcome=AdaptiveSamplingDecisionOutcome.DENSE_QA_ALREADY_COMPLETE,
        outcome_detail="coarse-complete dense QA proof permits no additional dense targets",
        no_additional_work_proof=proof,
    )

    assert abstained.incremental_targets == ()
    assert complete.no_additional_work_proof == proof
    assert "NO_EVENTS" not in {item.value for item in AdaptiveSamplingDecisionOutcome}
    with pytest.raises(ValueError, match="requires its domain proof"):
        build_adaptive_sampling_decision(
            base=_base(),
            accepted_evidence=_accepted(),
            source=_source(),
            effective_interval=_interval(),
            policy=_policy(),
            no_trigger_outcome=AdaptiveSamplingDecisionOutcome.DENSE_QA_ALREADY_COMPLETE,
            outcome_detail="missing proof",
        )


def test_budget_exhaustion_is_an_explicit_decision() -> None:
    decision = build_adaptive_sampling_decision(
        base=_base(),
        accepted_evidence=_accepted(),
        source=_source(),
        effective_interval=_interval(),
        policy=_policy(max_targets_per_camera=1, max_targets_total=6),
        triggers=(_trigger(),),
    )

    assert decision.outcome is AdaptiveSamplingDecisionOutcome.UPGRADE_BUDGET_EXHAUSTED
    assert decision.incremental_targets == ()
    assert decision.coverage_accounting is None


def test_store_reopens_exact_frozen_decision_and_rejects_slot_collision(tmp_path: Path) -> None:
    database_path = tmp_path / "adaptive.sqlite3"
    decision = _decision(_trigger())
    first = SQLiteAdaptiveDecisionStore(database_path).put_or_get(
        decision,
        sealed_at="2026-07-28T01:00:00Z",
    )
    replay = SQLiteAdaptiveDecisionStore(database_path).put_or_get(
        decision,
        sealed_at="2026-07-28T02:00:00Z",
    )

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.decision == decision
    assert replay.exact_bytes == canonical_json_bytes(decision)
    assert replay.sealed_at == "2026-07-28T01:00:00Z"
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT payload_json FROM adaptive_sampling_decisions").fetchone()
    assert row == (canonical_json_bytes(decision),)

    changed = _decision(_trigger(timestamp_ns=750_000_000, locator="claim/changed"))
    with pytest.raises(AdaptiveDecisionStoreConflict, match="different exact bytes"):
        SQLiteAdaptiveDecisionStore(database_path).put_or_get(changed)

    policy_changed = build_adaptive_sampling_decision(
        base=_base(),
        accepted_evidence=_accepted(),
        source=_source(),
        effective_interval=_interval(),
        policy=_policy(max_targets_per_camera=5, max_targets_total=30),
        triggers=(_trigger(),),
    )
    assert policy_changed.decision_scope_sha256 != decision.decision_scope_sha256
    assert (
        SQLiteAdaptiveDecisionStore(database_path)
        .put_or_get(
            policy_changed,
            sealed_at="2026-07-28T03:00:00Z",
        )
        .replayed
        is False
    )


def test_late_feedback_is_append_only_and_cannot_change_a_sealed_decision(tmp_path: Path) -> None:
    store = SQLiteAdaptiveDecisionStore(tmp_path / "adaptive.sqlite3")
    decision = _decision(_trigger())
    store.put_or_get(decision, sealed_at="2026-07-28T01:00:00Z")
    late = build_adaptive_late_feedback_audit(
        decision_scope_sha256=decision.decision_scope_sha256,
        arrival_id="broker-message-42",
        trigger=_trigger(timestamp_ns=900_000_000, locator="claim/late"),
        observed_at="2026-07-28T01:00:03Z",
    )

    recorded = store.record_late_feedback(late, recorded_at="2026-07-28T01:00:04Z")
    replay = SQLiteAdaptiveDecisionStore(store.database_path).record_late_feedback(
        late,
        recorded_at="2026-07-28T02:00:00Z",
    )
    reopened = SQLiteAdaptiveDecisionStore(store.database_path).get(decision.decision_scope_sha256)

    assert recorded.replayed is False
    assert replay.replayed is True
    assert reopened is not None and reopened.decision == decision
    assert (
        SQLiteAdaptiveDecisionStore(store.database_path)
        .list_late_feedback(decision.decision_scope_sha256)[0]
        .audit
        == late
    )

    for statement, parameters in (
        (
            "UPDATE adaptive_sampling_decisions SET sealed_at = ? WHERE decision_scope_sha256 = ?",
            ("2026-07-28T03:00:00Z", decision.decision_scope_sha256),
        ),
        (
            "DELETE FROM adaptive_sampling_decisions WHERE decision_scope_sha256 = ?",
            (decision.decision_scope_sha256,),
        ),
        (
            "UPDATE adaptive_sampling_late_feedback_audits SET recorded_at = ? WHERE audit_id = ?",
            ("2026-07-28T03:00:00Z", late.audit_id),
        ),
        (
            "DELETE FROM adaptive_sampling_late_feedback_audits WHERE audit_id = ?",
            (late.audit_id,),
        ),
    ):
        with (
            sqlite3.connect(store.database_path) as connection,
            pytest.raises(sqlite3.IntegrityError, match="immutable"),
        ):
            connection.execute(statement, parameters)

    assert (
        SQLiteAdaptiveDecisionStore(store.database_path).get(decision.decision_scope_sha256)
        is not None
    )
    assert (
        len(
            SQLiteAdaptiveDecisionStore(store.database_path).list_late_feedback(
                decision.decision_scope_sha256
            )
        )
        == 1
    )

    conflicting = build_adaptive_late_feedback_audit(
        decision_scope_sha256=decision.decision_scope_sha256,
        arrival_id="broker-message-42",
        trigger=_trigger(timestamp_ns=950_000_000, locator="claim/conflict"),
        observed_at="2026-07-28T01:00:03Z",
    )
    with pytest.raises(AdaptiveDecisionStoreConflict, match="different exact bytes"):
        store.record_late_feedback(conflicting)
