"""Typed logical-node producers for the canonical offline pipeline."""

from __future__ import annotations

from typing import Final

from robata.application.canonical.models import (
    CanonicalOfflineConfigurationError,
    CanonicalRootWindow,
)
from robata.application.canonical.output_admission import CanonicalOutputAdmissionDecision
from robata.application.canonical.projections import (
    CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION,
    CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE,
)
from robata.application.canonical.reduction import CanonicalFusionReduction
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import LogicalNode, logical_node_from_semantic_digest
from robata.contracts.temporal import TemporalPackageSet
from robata.event_pipeline.identity_registry import (
    EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE,
    PlatformEnrichedEventHypothesis,
)
from robata.inference.call_barrier import InferenceCallReduction
from robata.inference.enrichment import (
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    SelectedAttemptOutput,
)
from robata.inference.input_plan import (
    CALL_BARRIER_LOGICAL_KEY_NAMESPACE,
    CALL_PART_LOGICAL_KEY_NAMESPACE,
    INPUT_PLAN_LOGICAL_KEY_NAMESPACE,
    InferenceCallPart,
    InferenceInputPlan,
)
from robata.inference.models import (
    InferenceAttemptSelection,
    inference_attempt_selection_digest,
)

CANONICAL_INPUT_PLAN_IDENTITY_POLICY_VERSION: Final = "canonical-input-plan-node-v2"
CANONICAL_CALL_PART_IDENTITY_POLICY_VERSION: Final = "canonical-input-call-part-node-v2"
CANONICAL_CALL_BARRIER_IDENTITY_POLICY_VERSION: Final = "canonical-input-barrier-node-v2"
CANONICAL_EVENT_HYPOTHESIS_IDENTITY_POLICY_VERSION: Final = "canonical-event-hypothesis-node-v2"


def _canonical_logical_node(
    *,
    node_type: str,
    key_namespace: str,
    semantic_digest: str,
    logical_key: str,
    identity_policy_version: str,
) -> LogicalNode:
    expected_key = f"{key_namespace}:{semantic_digest}"
    if logical_key != expected_key:
        raise CanonicalOfflineConfigurationError(
            f"{node_type} logical key does not match its typed semantic digest"
        )
    return logical_node_from_semantic_digest(
        node_type=node_type,
        key_namespace=key_namespace,
        semantic_sha256=semantic_digest,
        identity_policy_version=identity_policy_version,
    )


def canonical_root_window_logical_node(window: CanonicalRootWindow) -> LogicalNode:
    """Admit one validated root-window producer into the generic node registry."""

    checked = CanonicalRootWindow.model_validate(window.model_dump(mode="python"), strict=True)
    return _canonical_logical_node(
        node_type="TEMPORAL_WINDOW",
        key_namespace="temporal-window",
        semantic_digest=checked.semantic_sha256,
        logical_key=checked.window_logical_key,
        identity_policy_version="canonical-root-window-node-v1",
    )


def canonical_package_set_semantic_projection(
    package_set: TemporalPackageSet,
) -> dict[str, object]:
    """Return package-set identity without execution or storage locators."""

    checked = TemporalPackageSet.model_validate(package_set.model_dump(mode="python"), strict=True)
    return {
        "schema_version": checked.schema_version,
        "lineage": checked.lineage.model_dump(mode="json"),
        "requested_interval": {
            "start_ns": str(checked.requested_start_ns),
            "end_ns": str(checked.requested_end_ns),
        },
        "effective_interval": {
            "start_ns": str(checked.start_ns),
            "end_ns": str(checked.end_ns),
        },
        "split_reason": checked.split_reason.value,
        "split_policy_version": checked.split_policy_version,
        "split_plan_digest": checked.split_plan_digest,
        "members": [
            {
                "ordinal": member.ordinal,
                "part_count": member.part_count,
                "requested_start_ns": str(member.requested_start_ns),
                "requested_end_ns": str(member.requested_end_ns),
                "start_ns": str(member.start_ns),
                "end_ns": str(member.end_ns),
                "overlap_before_ns": str(member.overlap_before_ns),
                "overlap_after_ns": str(member.overlap_after_ns),
                "package_semantic_content_sha256": member.package_semantic_content_sha256,
            }
            for member in checked.members
        ],
        "member_manifest_sha256": checked.member_manifest_sha256,
        "reduction_policy_version": checked.reduction_policy_version,
    }


def canonical_package_set_logical_node(package_set: TemporalPackageSet) -> LogicalNode:
    checked = TemporalPackageSet.model_validate(package_set.model_dump(mode="python"), strict=True)
    digest = semantic_sha256(canonical_package_set_semantic_projection(checked))
    return _canonical_logical_node(
        node_type="TEMPORAL_PACKAGE_SET",
        key_namespace="temporal-package-set",
        semantic_digest=digest,
        logical_key=f"temporal-package-set:{digest}",
        identity_policy_version="canonical-package-set-node-v1",
    )


def canonical_input_plan_logical_node(input_plan: InferenceInputPlan) -> LogicalNode:
    checked = InferenceInputPlan.model_validate(input_plan.model_dump(mode="python"), strict=True)
    return _canonical_logical_node(
        node_type="INFERENCE_INPUT_PLAN",
        key_namespace=INPUT_PLAN_LOGICAL_KEY_NAMESPACE,
        semantic_digest=checked.semantic_sha256,
        logical_key=f"{INPUT_PLAN_LOGICAL_KEY_NAMESPACE}:{checked.semantic_sha256}",
        identity_policy_version=CANONICAL_INPUT_PLAN_IDENTITY_POLICY_VERSION,
    )


def canonical_call_part_logical_node(
    input_plan: InferenceInputPlan,
    part: InferenceCallPart,
) -> LogicalNode:
    checked_plan = InferenceInputPlan.model_validate(
        input_plan.model_dump(mode="python"), strict=True
    )
    checked = InferenceCallPart.model_validate(part.model_dump(mode="python"), strict=True)
    if (
        checked.ordinal >= len(checked_plan.call_plan.parts)
        or checked_plan.call_plan.parts[checked.ordinal] != checked
    ):
        raise CanonicalOfflineConfigurationError(
            "call part is not an exact member of the validated input plan"
        )
    return _canonical_logical_node(
        node_type="INFERENCE_CALL_PART",
        key_namespace=CALL_PART_LOGICAL_KEY_NAMESPACE,
        semantic_digest=checked.part_semantic_sha256,
        logical_key=checked.part_logical_key,
        identity_policy_version=CANONICAL_CALL_PART_IDENTITY_POLICY_VERSION,
    )


def canonical_call_barrier_logical_node(input_plan: InferenceInputPlan) -> LogicalNode:
    checked = InferenceInputPlan.model_validate(input_plan.model_dump(mode="python"), strict=True)
    barrier = checked.call_plan
    return _canonical_logical_node(
        node_type="INFERENCE_CALL_BARRIER",
        key_namespace=CALL_BARRIER_LOGICAL_KEY_NAMESPACE,
        semantic_digest=barrier.barrier_semantic_sha256,
        logical_key=barrier.barrier_logical_key,
        identity_policy_version=CANONICAL_CALL_BARRIER_IDENTITY_POLICY_VERSION,
    )


def canonical_selection_logical_node(
    selection: InferenceAttemptSelection,
) -> LogicalNode:
    checked = InferenceAttemptSelection.model_validate(
        selection.model_dump(mode="python"), strict=True
    )
    digest = inference_attempt_selection_digest(
        logical_invocation_id=checked.logical_invocation_id,
        policy_version=checked.policy_version,
    )
    return _canonical_logical_node(
        node_type="INFERENCE_ATTEMPT_SELECTION",
        key_namespace="inference-attempt-selection",
        semantic_digest=digest,
        logical_key=checked.selection_decision_logical_key,
        identity_policy_version="canonical-attempt-selection-node-v1",
    )


def canonical_parsed_claim_logical_node(
    parsed: ParsedProviderClaimArtifact,
) -> LogicalNode:
    checked = ParsedProviderClaimArtifact.model_validate(
        parsed.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="PARSED_PROVIDER_CLAIM",
        key_namespace="parsed-provider-claim",
        semantic_digest=checked.semantic_sha256,
        logical_key=f"parsed-provider-claim:{checked.semantic_sha256}",
        identity_policy_version="canonical-parsed-claim-node-v1",
    )


def canonical_selected_output_logical_node(
    selected: SelectedAttemptOutput,
) -> LogicalNode:
    checked = SelectedAttemptOutput.model_validate(selected.model_dump(mode="python"), strict=True)
    return _canonical_logical_node(
        node_type="SELECTED_ATTEMPT_OUTPUT",
        key_namespace="selected-attempt-output",
        semantic_digest=checked.output_sha256,
        logical_key=f"selected-attempt-output:{checked.output_sha256}",
        identity_policy_version="canonical-selected-output-node-v1",
    )


def canonical_enrichment_logical_node(
    output: OrchestratorEnrichedOutput,
) -> LogicalNode:
    checked = OrchestratorEnrichedOutput.model_validate(
        output.model_dump(mode="python"), strict=True
    )
    digest = checked.enrichment_logical_key.rsplit(":", maxsplit=1)[-1]
    return _canonical_logical_node(
        node_type="ORCHESTRATOR_ENRICHMENT",
        key_namespace="orchestrator-enrichment",
        semantic_digest=digest,
        logical_key=checked.enrichment_logical_key,
        identity_policy_version="canonical-enrichment-node-v1",
    )


def canonical_call_reduction_logical_node(
    reduction: InferenceCallReduction,
) -> LogicalNode:
    checked = InferenceCallReduction.model_validate(
        reduction.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="INFERENCE_CALL_REDUCTION",
        key_namespace="inference-call-reduction",
        semantic_digest=checked.reduction_semantic_sha256,
        logical_key=f"inference-call-reduction:{checked.reduction_semantic_sha256}",
        identity_policy_version="canonical-call-reduction-node-v1",
    )


def canonical_fusion_reduction_logical_node(
    reduction: CanonicalFusionReduction,
) -> LogicalNode:
    checked = CanonicalFusionReduction.model_validate(
        reduction.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="FUSION_REDUCTION",
        key_namespace="fusion-reduction",
        semantic_digest=checked.semantic_sha256,
        logical_key=checked.reduction_logical_key,
        identity_policy_version="canonical-fusion-reduction-node-v1",
    )


def canonical_output_decision_logical_node(
    decision: CanonicalOutputAdmissionDecision,
) -> LogicalNode:
    checked = CanonicalOutputAdmissionDecision.model_validate(
        decision.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="OUTPUT_ADMISSION_DECISION",
        key_namespace=CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE,
        semantic_digest=checked.semantic_sha256,
        logical_key=(
            f"{CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE}:{checked.semantic_sha256}"
        ),
        identity_policy_version=CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION,
    )


def canonical_event_hypothesis_logical_node(
    hypothesis: PlatformEnrichedEventHypothesis,
) -> LogicalNode:
    checked = PlatformEnrichedEventHypothesis.model_validate(
        hypothesis.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="EVENT_HYPOTHESIS",
        key_namespace=EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE,
        semantic_digest=checked.semantic_sha256,
        logical_key=checked.event_hypothesis_logical_key,
        identity_policy_version=CANONICAL_EVENT_HYPOTHESIS_IDENTITY_POLICY_VERSION,
    )


__all__ = [
    "CANONICAL_CALL_BARRIER_IDENTITY_POLICY_VERSION",
    "CANONICAL_CALL_PART_IDENTITY_POLICY_VERSION",
    "CANONICAL_EVENT_HYPOTHESIS_IDENTITY_POLICY_VERSION",
    "CANONICAL_INPUT_PLAN_IDENTITY_POLICY_VERSION",
    "canonical_call_barrier_logical_node",
    "canonical_call_part_logical_node",
    "canonical_call_reduction_logical_node",
    "canonical_enrichment_logical_node",
    "canonical_event_hypothesis_logical_node",
    "canonical_fusion_reduction_logical_node",
    "canonical_input_plan_logical_node",
    "canonical_output_decision_logical_node",
    "canonical_package_set_logical_node",
    "canonical_package_set_semantic_projection",
    "canonical_parsed_claim_logical_node",
    "canonical_root_window_logical_node",
    "canonical_selected_output_logical_node",
    "canonical_selection_logical_node",
]
