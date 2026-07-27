"""Canonical, transport-free vertical slice for the registered V2 path.

The service in this module is deliberately local and non-promotional.  It
connects the registered admission evidence through temporal materialization,
provider-specific planning, an all-part inference barrier, strict raw response
parsing, enrichment, deterministic fusion reduction, and immutable
recording-scoped local event hypotheses. It stops before event-identity
assignment or outbox publication.
No network-capable adapter is accepted by the coordinator.

A separate side-effect-free preparation API can derive local-conformance
ActionEvent genesis revision, selection, and current-projection commands after
identity preparation. The coordinator still does not publish those commands.
"""

from __future__ import annotations

from robata.application.canonical.action_event_revision import (
    CanonicalActionEventRevisionError as CanonicalActionEventRevisionError,
)
from robata.application.canonical.action_event_revision import (
    PreparedInitialActionEventRevision as PreparedInitialActionEventRevision,
)
from robata.application.canonical.action_event_revision import (
    PreparedInitialActionEventRevisionBatch as PreparedInitialActionEventRevisionBatch,
)
from robata.application.canonical.action_event_revision import (
    prepare_initial_action_event_publications as prepare_initial_action_event_publications,
)
from robata.application.canonical.boundary_windows import (
    CanonicalBoundaryRefinementWindow as CanonicalBoundaryRefinementWindow,
)
from robata.application.canonical.boundary_windows import (
    boundary_refinement_window_projection as boundary_refinement_window_projection,
)
from robata.application.canonical.boundary_windows import (
    canonical_boundary_refinement_lineage as canonical_boundary_refinement_lineage,
)
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
    canonical_action_evidence_result_logical_node as canonical_action_evidence_result_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_boundary_refinement_result_logical_node as canonical_boundary_refinement_result_logical_node,  # noqa: E501
)
from robata.application.canonical.logical_nodes import (
    canonical_boundary_role_result_logical_node as canonical_boundary_role_result_logical_node,
)
from robata.application.canonical.logical_nodes import (
    canonical_boundary_window_logical_node as canonical_boundary_window_logical_node,
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
    canonical_candidate_dense_window_logical_node as canonical_candidate_dense_window_logical_node,
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
    CanonicalCandidateDenseWindow as CanonicalCandidateDenseWindow,
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
    canonical_candidate_dense_lineage as canonical_candidate_dense_lineage,
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
from robata.application.canonical.primary_completion import (
    CanonicalPrimaryCompletionDetail as CanonicalPrimaryCompletionDetail,
)
from robata.application.canonical.primary_completion import (
    CommittedPrimaryCompletion as CommittedPrimaryCompletion,
)
from robata.application.canonical.primary_completion import (
    PrimaryCompletionCommand as PrimaryCompletionCommand,
)
from robata.application.canonical.primary_completion import (
    PrimaryCompletionCommitResult as PrimaryCompletionCommitResult,
)
from robata.application.canonical.primary_completion import (
    PrimaryCompletionError as PrimaryCompletionError,
)
from robata.application.canonical.primary_completion import (
    PrimaryCompletionErrorCode as PrimaryCompletionErrorCode,
)
from robata.application.canonical.primary_completion import (
    PrimaryCompletionEvidenceReference as PrimaryCompletionEvidenceReference,
)
from robata.application.canonical.primary_completion import (
    PrimaryCompletionEvidenceRole as PrimaryCompletionEvidenceRole,
)
from robata.application.canonical.primary_completion import (
    PrimaryCompletionRepository as PrimaryCompletionRepository,
)
from robata.application.canonical.primary_completion import (
    create_primary_completion_command as create_primary_completion_command,
)
from robata.application.canonical.projections import (
    CANONICAL_EVENT_INDEX_PROJECTION_VERSION as CANONICAL_EVENT_INDEX_PROJECTION_VERSION,
)
from robata.application.canonical.projections import (
    CANONICAL_EXECUTION_POLICY_SEMANTIC_PROJECTION_VERSION as CANONICAL_EXECUTION_POLICY_SEMANTIC_PROJECTION_VERSION,  # noqa: E501
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
    canonical_event_index_batch_projection as canonical_event_index_batch_projection,
)
from robata.application.canonical.projections import (
    canonical_event_index_projection as canonical_event_index_projection,
)
from robata.application.canonical.projections import (
    canonical_event_index_projection_batch as canonical_event_index_projection_batch,
)
from robata.application.canonical.projections import (
    canonical_event_index_projection_values as canonical_event_index_projection_values,
)
from robata.application.canonical.projections import (
    canonical_event_index_revision_projection as canonical_event_index_revision_projection,
)
from robata.application.canonical.projections import (
    canonical_event_index_row_projection as canonical_event_index_row_projection,
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
from robata.application.canonical.projections import (
    canonical_terminal_event_index_projection as canonical_terminal_event_index_projection,
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
    CanonicalActionEvidenceExecution as CanonicalActionEvidenceExecution,
)
from robata.application.canonical.result_validation import (
    CanonicalBoundaryRefinementExecution as CanonicalBoundaryRefinementExecution,
)
from robata.application.canonical.result_validation import (
    CanonicalBoundaryRefinementPassExecution as CanonicalBoundaryRefinementPassExecution,
)
from robata.application.canonical.result_validation import (
    CanonicalDenseQAExecution as CanonicalDenseQAExecution,
)
from robata.application.canonical.result_validation import (
    CanonicalOfflineRunResult as CanonicalOfflineRunResult,
)
from robata.application.canonical.result_validation import (
    canonical_action_evidence_execution_membership_lineage,
    canonical_dense_qa_execution_membership_lineage,
)
from robata.application.canonical.runner import (
    CanonicalOfflinePipeline as CanonicalOfflinePipeline,
)
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementOutcome as BoundaryRefinementOutcome,
)
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementPolicy as BoundaryRefinementPolicy,
)
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementResult as BoundaryRefinementResult,
)
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementRole as BoundaryRefinementRole,
)
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementRoleResult as BoundaryRefinementRoleResult,
)

__all__ = [
    "CANONICAL_CALL_BARRIER_IDENTITY_POLICY_VERSION",
    "CANONICAL_CALL_PART_IDENTITY_POLICY_VERSION",
    "CANONICAL_EVENT_HYPOTHESIS_IDENTITY_POLICY_VERSION",
    "CANONICAL_EVENT_INDEX_PROJECTION_VERSION",
    "CANONICAL_EXECUTION_POLICY_SEMANTIC_PROJECTION_VERSION",
    "CANONICAL_INPUT_PLAN_IDENTITY_POLICY_VERSION",
    "CANONICAL_OFFLINE_PIPELINE_VERSION",
    "CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION",
    "CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE",
    "CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION",
    "CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE",
    "BoundaryRefinementOutcome",
    "BoundaryRefinementPolicy",
    "BoundaryRefinementResult",
    "BoundaryRefinementRole",
    "BoundaryRefinementRoleResult",
    "CanonicalActionEventRevisionError",
    "CanonicalActionEvidenceExecution",
    "CanonicalBoundaryRefinementExecution",
    "CanonicalBoundaryRefinementPassExecution",
    "CanonicalBoundaryRefinementWindow",
    "CanonicalCandidateDenseWindow",
    "CanonicalDenseQAExecution",
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
    "CanonicalPrimaryCompletionDetail",
    "CanonicalReducedFusionClaim",
    "CanonicalRootWindow",
    "CommittedPrimaryCompletion",
    "FusionEventHypothesisProjector",
    "PreparedInitialActionEventRevision",
    "PreparedInitialActionEventRevisionBatch",
    "PrimaryCompletionCommand",
    "PrimaryCompletionCommitResult",
    "PrimaryCompletionError",
    "PrimaryCompletionErrorCode",
    "PrimaryCompletionEvidenceReference",
    "PrimaryCompletionEvidenceRole",
    "PrimaryCompletionRepository",
    "boundary_refinement_window_projection",
    "canonical_action_evidence_execution_membership_lineage",
    "canonical_action_evidence_result_logical_node",
    "canonical_boundary_refinement_lineage",
    "canonical_boundary_refinement_result_logical_node",
    "canonical_boundary_role_result_logical_node",
    "canonical_boundary_window_logical_node",
    "canonical_call_barrier_logical_node",
    "canonical_call_part_logical_node",
    "canonical_call_reduction_logical_node",
    "canonical_candidate_dense_lineage",
    "canonical_candidate_dense_window_logical_node",
    "canonical_dense_qa_execution_membership_lineage",
    "canonical_enrichment_logical_node",
    "canonical_event_hypothesis_logical_node",
    "canonical_event_index_batch_projection",
    "canonical_event_index_projection",
    "canonical_event_index_projection_batch",
    "canonical_event_index_projection_values",
    "canonical_event_index_revision_projection",
    "canonical_event_index_row_projection",
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
    "canonical_terminal_event_index_projection",
    "create_primary_completion_command",
    "prepare_initial_action_event_publications",
]
