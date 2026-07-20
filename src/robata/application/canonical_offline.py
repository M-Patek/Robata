"""Canonical, transport-free vertical slice for the registered V2 path.

The service in this module is deliberately local and non-promotional.  It
connects the registered admission evidence through temporal materialization,
provider-specific planning, an all-part inference barrier, strict raw response
parsing, enrichment, deterministic fusion reduction, and immutable
recording-scoped local event hypotheses. It stops before event-identity
assignment or outbox publication.
No network-capable adapter is accepted by the coordinator.
"""

from __future__ import annotations

from robata.application.canonical.logical_nodes import (
    CANONICAL_CALL_BARRIER_IDENTITY_POLICY_VERSION as CANONICAL_CALL_BARRIER_IDENTITY_POLICY_VERSION,  # noqa: E501
)
from robata.application.canonical.logical_nodes import (
    CANONICAL_CALL_PART_IDENTITY_POLICY_VERSION as CANONICAL_CALL_PART_IDENTITY_POLICY_VERSION,
)
from robata.application.canonical.logical_nodes import (
    CANONICAL_EVENT_HYPOTHESIS_IDENTITY_POLICY_VERSION as CANONICAL_EVENT_HYPOTHESIS_IDENTITY_POLICY_VERSION,  # noqa: E501
)
from robata.application.canonical.logical_nodes import (
    CANONICAL_INPUT_PLAN_IDENTITY_POLICY_VERSION as CANONICAL_INPUT_PLAN_IDENTITY_POLICY_VERSION,
)
from robata.application.canonical.logical_nodes import (
    canonical_call_barrier_logical_node as canonical_call_barrier_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_call_part_logical_node as canonical_call_part_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_call_reduction_logical_node as canonical_call_reduction_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_enrichment_logical_node as canonical_enrichment_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_event_hypothesis_logical_node as canonical_event_hypothesis_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_fusion_reduction_logical_node as canonical_fusion_reduction_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_input_plan_logical_node as canonical_input_plan_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_output_decision_logical_node as canonical_output_decision_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_package_set_logical_node as canonical_package_set_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_package_set_semantic_projection as canonical_package_set_semantic_projection,
)
from robata.application.canonical.logical_nodes import (
    canonical_parsed_claim_logical_node as canonical_parsed_claim_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_root_window_logical_node as canonical_root_window_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_selected_output_logical_node as canonical_selected_output_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_selection_logical_node as canonical_selection_logical_node,
)
from robata.application.canonical.models import (
    CANONICAL_OFFLINE_PIPELINE_VERSION as CANONICAL_OFFLINE_PIPELINE_VERSION,
)
from robata.application.canonical.models import (
    CanonicalOfflineConfigurationError as CanonicalOfflineConfigurationError,
)
from robata.application.canonical.models import (
    CanonicalOfflineError as CanonicalOfflineError,
)
from robata.application.canonical.models import (
    CanonicalOfflineExecutionPolicy as CanonicalOfflineExecutionPolicy,
)
from robata.application.canonical.models import (
    CanonicalOfflinePartResult as CanonicalOfflinePartResult,
)
from robata.application.canonical.models import (
    CanonicalOfflinePartStatus as CanonicalOfflinePartStatus,
)
from robata.application.canonical.models import (
    CanonicalOfflineRunStatus as CanonicalOfflineRunStatus,
)
from robata.application.canonical.models import (
    CanonicalOfflineStage as CanonicalOfflineStage,
)
from robata.application.canonical.models import (
    CanonicalRootWindow as CanonicalRootWindow,
)
from robata.application.canonical.models import (
    canonical_lineage as canonical_lineage,
)
from robata.application.canonical.output_admission import (
    CanonicalOutputAdmissionDecision as CanonicalOutputAdmissionDecision,
)
from robata.application.canonical.output_admission import (
    FusionEventHypothesisProjector as FusionEventHypothesisProjector,
)
from robata.application.canonical.projections import (
    CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION as CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION,  # noqa: E501
)
from robata.application.canonical.projections import (
    CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE as CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE,  # noqa: E501
)
from robata.application.canonical.projections import (
    CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION as CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION,  # noqa: E501
)
from robata.application.canonical.projections import (
    CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE as CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE,
)
from robata.application.canonical.projections import (
    _canonical_output_decision_projection_values as _canonical_output_decision_projection_values,
)
from robata.application.canonical.projections import (
    _fusion_claim_reduction_digest as _fusion_claim_reduction_digest,
)
from robata.application.canonical.projections import (
    _stable_uuid as _stable_uuid,
)
from robata.application.canonical.projections import (
    canonical_execution_policy_projection as canonical_execution_policy_projection,
)
from robata.application.canonical.projections import (
    canonical_execution_policy_projection_values as canonical_execution_policy_projection_values,
)
from robata.application.canonical.projections import (
    canonical_fusion_reduction_projection as canonical_fusion_reduction_projection,
)
from robata.application.canonical.projections import (
    canonical_output_decision_projection as canonical_output_decision_projection,
)
from robata.application.canonical.projections import (
    canonical_root_window_projection_values as canonical_root_window_projection_values,
)
from robata.application.canonical.reduction import (
    CanonicalFusionClaimSource as CanonicalFusionClaimSource,
)
from robata.application.canonical.reduction import (
    CanonicalFusionPartSource as CanonicalFusionPartSource,
)
from robata.application.canonical.reduction import (
    CanonicalFusionReduction as CanonicalFusionReduction,
)
from robata.application.canonical.reduction import (
    CanonicalReducedFusionClaim as CanonicalReducedFusionClaim,
)
from robata.application.canonical.reduction import (
    _reduce_provider_claim_payloads as _reduce_provider_claim_payloads,
)
from robata.application.canonical.result_validation import (
    CanonicalOfflineRunResult as CanonicalOfflineRunResult,
)
from robata.application.canonical.runner import (
    CanonicalOfflinePipeline as CanonicalOfflinePipeline,
)

__all__ = [
    "CANONICAL_CALL_BARRIER_IDENTITY_POLICY_VERSION",
    "CANONICAL_CALL_PART_IDENTITY_POLICY_VERSION",
    "CANONICAL_EVENT_HYPOTHESIS_IDENTITY_POLICY_VERSION",
    "CANONICAL_INPUT_PLAN_IDENTITY_POLICY_VERSION",
    "CANONICAL_OFFLINE_PIPELINE_VERSION",
    "CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION",
    "CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE",
    "CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION",
    "CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE",
    "CanonicalFusionClaimSource",
    "CanonicalFusionPartSource",
    "CanonicalFusionReduction",
    "CanonicalOfflineConfigurationError",
    "CanonicalOfflineError",
    "CanonicalOfflineExecutionPolicy",
    "CanonicalOfflinePartResult",
    "CanonicalOfflinePartStatus",
    "CanonicalOfflinePipeline",
    "CanonicalOfflineRunResult",
    "CanonicalOfflineRunStatus",
    "CanonicalOfflineStage",
    "CanonicalOutputAdmissionDecision",
    "CanonicalReducedFusionClaim",
    "CanonicalRootWindow",
    "FusionEventHypothesisProjector",
    "canonical_call_barrier_logical_node",
    "canonical_call_part_logical_node",
    "canonical_call_reduction_logical_node",
    "canonical_enrichment_logical_node",
    "canonical_event_hypothesis_logical_node",
    "canonical_execution_policy_projection",
    "canonical_fusion_reduction_logical_node",
    "canonical_fusion_reduction_projection",
    "canonical_input_plan_logical_node",
    "canonical_lineage",
    "canonical_output_decision_logical_node",
    "canonical_output_decision_projection",
    "canonical_package_set_logical_node",
    "canonical_package_set_semantic_projection",
    "canonical_parsed_claim_logical_node",
    "canonical_root_window_logical_node",
    "canonical_root_window_projection_values",
    "canonical_selected_output_logical_node",
    "canonical_selection_logical_node",
]
