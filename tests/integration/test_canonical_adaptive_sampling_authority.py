from __future__ import annotations

from pathlib import Path

from robata.application.canonical.adaptive_sampling import (
    AdaptiveSamplingExecutionStatus,
    AdaptiveSamplingExecutionStore,
    AdaptiveSamplingWorkReceipt,
    CanonicalAdaptiveSamplingBridge,
)
from robata.application.canonical.result_validation import CanonicalOfflineRunResult
from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.sampling.adaptive import AdaptiveCoveragePolicy, AdaptiveUpgradeReason
from robata.sampling.adaptive_decision import (
    AcceptedAdaptiveEvidenceBinding,
    AdaptiveDecisionBaseBinding,
    AdaptiveDecisionSourceBinding,
    AdaptiveTriggerEvidenceKind,
    AdaptiveTriggerProvenance,
    build_adaptive_sampling_decision,
)
from robata.sampling.adaptive_decision_store import SQLiteAdaptiveDecisionStore
from tests.integration.test_canonical_offline import _claim_bytes, _harness, _run


def _adaptive_decision_for_completed_result(
    result: CanonicalOfflineRunResult,
):
    package_set = result.package_set
    window = result.window
    assert package_set is not None
    assert window is not None
    part = next(
        item
        for item in result.part_results
        if (
            item.selection is not None
            and item.selected_output is not None
            and item.enriched_output is not None
        )
    )
    selection = part.selection
    selected = part.selected_output
    enriched = part.enriched_output
    assert selection is not None
    assert selected is not None
    assert enriched is not None
    interval = window.interval
    trigger_timestamp_ns = (interval.start_ns + interval.end_ns) // 2
    accepted = AcceptedAdaptiveEvidenceBinding(
        selection_id=selection.selection_id,
        selection_decision_logical_key=selection.selection_decision_logical_key,
        selected_output_sha256=selected.output_sha256,
        enriched_output_artifact_id=enriched.artifact_id,
        enriched_output_semantic_sha256=enriched.semantic_sha256,
    )
    return build_adaptive_sampling_decision(
        base=AdaptiveDecisionBaseBinding(
            sampling_plan_sha256=package_set.lineage.sampling_plan_sha256,
            package_set_id=package_set.package_set_id,
            package_set_member_manifest_sha256=package_set.member_manifest_sha256,
            package_set_split_plan_sha256=package_set.split_plan_digest,
        ),
        accepted_evidence=accepted,
        source=AdaptiveDecisionSourceBinding(
            source_content_sha256=window.source_content_sha256,
            camera_mapping_semantic_sha256=window.camera_mapping_semantic_sha256,
            alignment_semantic_sha256=window.alignment_semantic_sha256,
        ),
        effective_interval=NanosecondInterval(
            start_ns=interval.start_ns,
            end_ns=interval.end_ns,
        ),
        policy=AdaptiveCoveragePolicy(
            version="canonical-adaptive-authority-integration-v1",
            base_rate_num=1,
            base_target_budget_per_camera=1,
            context_offsets_ns=(-100_000_000, 100_000_000),
            max_targets_per_camera=4,
            max_targets_total=24,
        ),
        triggers=(
            AdaptiveTriggerProvenance(
                camera_id=CameraId.CAM_01,
                trigger_timestamp_ns=trigger_timestamp_ns,
                reason=AdaptiveUpgradeReason.COARSE_UNCERTAINTY,
                evidence_kind=AdaptiveTriggerEvidenceKind.ENRICHED_OUTPUT,
                evidence_sha256=accepted.enriched_output_semantic_sha256,
                evidence_locator="integration/final-fusion/part/0",
            ),
        ),
    )


def test_canonical_result_authorizes_adaptive_execution_only_from_real_retained_lineage(
    tmp_path: Path,
) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path / "logical-registry")
    result = _run(harness)
    decision = _adaptive_decision_for_completed_result(result)
    bridge = CanonicalAdaptiveSamplingBridge(
        SQLiteAdaptiveDecisionStore(tmp_path / "adaptive-decisions.sqlite3"),
        AdaptiveSamplingExecutionStore(tmp_path / "adaptive-executions"),
    )

    outcome = bridge.execute_for_canonical_result(
        decision,
        result=result,
        callback=lambda intent, _targets: AdaptiveSamplingWorkReceipt(
            work_product_sha256="f" * 64,
            work_product_locator=f"local://adaptive-integration/{intent.execution_id}",
        ),
    )

    assert outcome.status is AdaptiveSamplingExecutionStatus.EXECUTED
    assert outcome.terminal_receipt.work_receipt is not None
