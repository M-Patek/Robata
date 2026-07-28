from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robata.application.canonical.adaptive_sampling import (
    AdaptiveSamplingExecutionConflict,
    AdaptiveSamplingExecutionStatus,
    AdaptiveSamplingExecutionStorageError,
    AdaptiveSamplingExecutionStore,
    AdaptiveSamplingExecutionTerminalKind,
    AdaptiveSamplingWorkReceipt,
    CanonicalAdaptiveSamplingAuthorityError,
    CanonicalAdaptiveSamplingBridge,
    build_adaptive_sampling_execution_intent,
)
from robata.application.canonical.models import CanonicalOfflineRunStatus
from robata.application.canonical.result_validation import CanonicalOfflineRunResult
from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes
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
    build_adaptive_sampling_decision,
)
from robata.sampling.adaptive_decision_store import SQLiteAdaptiveDecisionStore


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


def _policy() -> AdaptiveCoveragePolicy:
    return AdaptiveCoveragePolicy(
        version="adaptive-coverage-v1",
        base_rate_num=1,
        base_target_budget_per_camera=1,
        context_offsets_ns=(-100_000_000, 100_000_000),
        max_targets_per_camera=4,
        max_targets_total=24,
    )


def _scheduled_decision() -> AdaptiveSamplingDecision:
    accepted = _accepted()
    return build_adaptive_sampling_decision(
        base=_base(),
        accepted_evidence=accepted,
        source=_source(),
        effective_interval=NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        policy=_policy(),
        triggers=(
            AdaptiveTriggerProvenance(
                camera_id=CameraId.CAM_01,
                trigger_timestamp_ns=500_000_000,
                reason=AdaptiveUpgradeReason.COARSE_UNCERTAINTY,
                evidence_kind=AdaptiveTriggerEvidenceKind.ENRICHED_OUTPUT,
                evidence_sha256=accepted.enriched_output_semantic_sha256,
                evidence_locator="claim/0",
            ),
        ),
    )


def _clear_video_decision() -> AdaptiveSamplingDecision:
    return build_adaptive_sampling_decision(
        base=_base(),
        accepted_evidence=_accepted(),
        source=_source(),
        effective_interval=NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        policy=_policy(),
        no_trigger_outcome=AdaptiveSamplingDecisionOutcome.DENSE_QA_ALREADY_COMPLETE,
        outcome_detail="coarse-complete dense QA permits no extra dense coordinates",
        no_additional_work_proof=AdaptiveNoAdditionalWorkProof(
            proof_kind=AdaptiveNoAdditionalWorkProofKind.DENSE_QA_COARSE_COMPLETE,
            evidence_artifact_id="00000000-0000-4000-8000-000000000004",
            evidence_sha256=_digest(10),
            policy_version="dense-qa-completion-v1",
        ),
    )


def _canonical_result_for(decision: AdaptiveSamplingDecision) -> CanonicalOfflineRunResult:
    accepted = decision.accepted_evidence
    source = decision.source
    selection = SimpleNamespace(
        selection_id=accepted.selection_id,
        selection_decision_logical_key=accepted.selection_decision_logical_key,
    )
    selected = SimpleNamespace(
        selection_decision_logical_key=accepted.selection_decision_logical_key,
        output_sha256=accepted.selected_output_sha256,
    )
    enriched = SimpleNamespace(
        selected_attempt=selected,
        artifact_id=accepted.enriched_output_artifact_id,
        semantic_sha256=accepted.enriched_output_semantic_sha256,
    )
    return CanonicalOfflineRunResult.model_construct(
        status=CanonicalOfflineRunStatus.SUCCEEDED,
        window=SimpleNamespace(
            source_content_sha256=source.source_content_sha256,
            camera_mapping_semantic_sha256=source.camera_mapping_semantic_sha256,
            alignment_semantic_sha256=source.alignment_semantic_sha256,
            interval=decision.effective_interval,
        ),
        package_set=SimpleNamespace(
            package_set_id=decision.base.package_set_id,
            member_manifest_sha256=decision.base.package_set_member_manifest_sha256,
            split_plan_digest=decision.base.package_set_split_plan_sha256,
            lineage=SimpleNamespace(
                source_content_sha256=source.source_content_sha256,
                camera_mapping_semantic_sha256=source.camera_mapping_semantic_sha256,
                alignment_semantic_sha256=source.alignment_semantic_sha256,
                sampling_plan_sha256=decision.base.sampling_plan_sha256,
            ),
        ),
        part_results=(
            SimpleNamespace(
                selection=selection,
                selected_output=selected,
                enriched_output=enriched,
            ),
        ),
    )


def _bridge(
    tmp_path: Path,
) -> tuple[
    SQLiteAdaptiveDecisionStore,
    AdaptiveSamplingExecutionStore,
    CanonicalAdaptiveSamplingBridge,
]:
    decisions = SQLiteAdaptiveDecisionStore(tmp_path / "adaptive-decisions.sqlite3")
    executions = AdaptiveSamplingExecutionStore(tmp_path / "adaptive-executions")
    return decisions, executions, CanonicalAdaptiveSamplingBridge(decisions, executions)


def _work_receipt(intent_id: str) -> AdaptiveSamplingWorkReceipt:
    return AdaptiveSamplingWorkReceipt(
        work_product_sha256=_digest(11),
        work_product_locator=f"local://adaptive-extra/{intent_id}",
    )


def test_seals_decision_then_persists_intent_before_callback(tmp_path: Path) -> None:
    decisions, executions, bridge = _bridge(tmp_path)
    decision = _scheduled_decision()
    observed: list[tuple[str, int]] = []

    def callback(intent: object, targets: tuple[object, ...]) -> AdaptiveSamplingWorkReceipt:
        assert decisions.get(decision.decision_scope_sha256) is not None
        assert hasattr(intent, "execution_id")
        execution_id = intent.execution_id  # type: ignore[attr-defined]
        assert executions.get_intent(execution_id) is not None
        assert targets == tuple(decision.incremental_targets)
        observed.append((intent.idempotency_key, len(targets)))  # type: ignore[attr-defined]
        return _work_receipt(intent.idempotency_key)  # type: ignore[attr-defined]

    result = bridge.execute(decision, callback=callback)

    assert result.status is AdaptiveSamplingExecutionStatus.EXECUTED
    assert result.decision_replayed is False
    assert result.terminal_replayed is False
    assert result.terminal_receipt.terminal_kind is (
        AdaptiveSamplingExecutionTerminalKind.ADDITIONAL_TARGETS_COMPLETED
    )
    assert observed == [(result.intent.idempotency_key, len(decision.incremental_targets))]
    assert executions.get_terminal(result.intent) == result.terminal_receipt


def test_canonical_result_authority_precedes_durable_adaptive_execution(tmp_path: Path) -> None:
    decisions, executions, bridge = _bridge(tmp_path)
    decision = _scheduled_decision()
    result = _canonical_result_for(decision)
    observed: list[str] = []

    outcome = bridge.execute_for_canonical_result(
        decision,
        result=result,
        callback=lambda intent, _targets: (
            observed.append(intent.idempotency_key) or _work_receipt(intent.idempotency_key)
        ),
    )

    assert outcome.status is AdaptiveSamplingExecutionStatus.EXECUTED
    assert observed == [outcome.intent.idempotency_key]
    assert decisions.get(decision.decision_scope_sha256) is not None
    assert executions.get_intent(outcome.intent.execution_id) == outcome.intent


def test_canonical_result_authority_rejects_unbound_decision_before_persistence(
    tmp_path: Path,
) -> None:
    decisions, executions, bridge = _bridge(tmp_path)
    decision = _scheduled_decision()
    result = _canonical_result_for(decision).model_copy(update={"part_results": ()})

    with pytest.raises(CanonicalAdaptiveSamplingAuthorityError, match="exactly one accepted"):
        bridge.execute_for_canonical_result(
            decision,
            result=result,
            callback=lambda intent, _targets: _work_receipt(intent.idempotency_key),
        )

    assert decisions.get(decision.decision_scope_sha256) is None
    assert (
        executions.get_intent(build_adaptive_sampling_execution_intent(decision).execution_id)
        is None
    )


def test_restart_replays_terminal_without_calling_provider_callback(tmp_path: Path) -> None:
    _, _, bridge = _bridge(tmp_path)
    decision = _scheduled_decision()
    first = bridge.execute(
        decision,
        callback=lambda intent, _targets: _work_receipt(intent.idempotency_key),
    )

    decisions, executions, restarted = _bridge(tmp_path)

    def should_not_execute(
        _intent: object, _targets: tuple[object, ...]
    ) -> AdaptiveSamplingWorkReceipt:
        raise AssertionError("completed adaptive work must replay without a callback")

    replay = restarted.execute(decision, callback=should_not_execute)

    assert decisions.get(decision.decision_scope_sha256) is not None
    assert executions.get_intent(first.intent.execution_id) == first.intent
    assert replay.status is AdaptiveSamplingExecutionStatus.REPLAYED
    assert replay.intent.execution_id == first.intent.execution_id
    assert replay.terminal_receipt == first.terminal_receipt
    assert replay.decision_replayed is True
    assert replay.terminal_replayed is True


def test_failed_callback_recovers_with_the_same_stable_idempotency_key(tmp_path: Path) -> None:
    _, executions, bridge = _bridge(tmp_path)
    decision = _scheduled_decision()
    attempted: list[str] = []

    def fail_after_intent(
        intent: object, _targets: tuple[object, ...]
    ) -> AdaptiveSamplingWorkReceipt:
        attempted.append(intent.idempotency_key)  # type: ignore[attr-defined]
        raise RuntimeError("provider transport disconnected after accepting the idempotency key")

    with pytest.raises(RuntimeError, match="transport disconnected"):
        bridge.execute(decision, callback=fail_after_intent)

    expected_intent = build_adaptive_sampling_execution_intent(decision)
    assert attempted == [expected_intent.idempotency_key]
    assert executions.get_intent(expected_intent.execution_id) == expected_intent
    assert executions.get_terminal(expected_intent) is None

    recovered: list[str] = []

    def retry(intent: object, _targets: tuple[object, ...]) -> AdaptiveSamplingWorkReceipt:
        recovered.append(intent.idempotency_key)  # type: ignore[attr-defined]
        return _work_receipt(intent.idempotency_key)  # type: ignore[attr-defined]

    result = bridge.execute(decision, callback=retry)

    assert recovered == [expected_intent.idempotency_key]
    assert result.intent.execution_id == expected_intent.execution_id
    assert result.status is AdaptiveSamplingExecutionStatus.EXECUTED


def test_clear_video_no_work_is_explicit_and_event_neutral(tmp_path: Path) -> None:
    _, executions, bridge = _bridge(tmp_path)
    decision = _clear_video_decision()

    def should_not_execute(
        _intent: object, _targets: tuple[object, ...]
    ) -> AdaptiveSamplingWorkReceipt:
        raise AssertionError("proved no-extra-work must not call an additional-target callback")

    first = bridge.execute(decision, callback=should_not_execute)
    second = bridge.execute(decision, callback=should_not_execute)
    terminal_bytes = executions.terminal_path(first.intent.execution_id).read_bytes()

    assert first.status is AdaptiveSamplingExecutionStatus.NO_ADDITIONAL_WORK
    assert (
        first.terminal_receipt.terminal_kind
        is AdaptiveSamplingExecutionTerminalKind.NO_ADDITIONAL_WORK
    )
    assert first.terminal_receipt.no_additional_work_outcome is (
        AdaptiveSamplingDecisionOutcome.DENSE_QA_ALREADY_COMPLETE
    )
    assert first.terminal_receipt.work_receipt is None
    assert second.status is AdaptiveSamplingExecutionStatus.REPLAYED
    assert b"NO_EVENTS" not in terminal_bytes


def test_execution_store_fails_closed_for_conflicts_and_tampered_files(tmp_path: Path) -> None:
    _, executions, _ = _bridge(tmp_path)
    intent = build_adaptive_sampling_execution_intent(_scheduled_decision())
    stored, replayed = executions.put_or_get_intent(intent)

    assert stored == intent
    assert replayed is False
    changed = intent.model_copy(update={"target_count": intent.target_count + 1})
    with pytest.raises(ValueError, match="execution intent semantic digest"):
        executions.put_or_get_intent(changed)

    intent_path = executions.intent_path(intent.execution_id)
    intent_path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(
        AdaptiveSamplingExecutionConflict, match="different immutable canonical bytes"
    ):
        executions.put_or_get_intent(intent)

    intent_path.write_bytes(b"{}")
    with pytest.raises(AdaptiveSamplingExecutionStorageError, match="invalid execution intent"):
        executions.get_intent(intent.execution_id)


def test_terminal_receipt_conflicts_and_tampering_fail_closed(tmp_path: Path) -> None:
    _, executions, bridge = _bridge(tmp_path)
    decision = _scheduled_decision()
    result = bridge.execute(
        decision,
        callback=lambda intent, _targets: _work_receipt(intent.idempotency_key),
    )
    changed = result.terminal_receipt.model_copy(
        update={
            "work_receipt": AdaptiveSamplingWorkReceipt(
                work_product_sha256=_digest(12),
                work_product_locator="local://adaptive-extra/tampered",
            )
        }
    )
    with pytest.raises(ValueError, match="terminal receipt semantic digest"):
        executions.put_or_get_terminal(changed, intent=result.intent)

    terminal_path = executions.terminal_path(result.intent.execution_id)
    terminal_path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(
        AdaptiveSamplingExecutionConflict, match="different immutable canonical bytes"
    ):
        executions.put_or_get_terminal(result.terminal_receipt, intent=result.intent)

    terminal_path.write_bytes(b"{}")
    with pytest.raises(
        AdaptiveSamplingExecutionStorageError,
        match="invalid execution terminal receipt",
    ):
        executions.get_terminal(result.intent)
