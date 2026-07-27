"""Terminal result model and retained-lineage validation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ValidationInfo, model_validator

from robata.application.canonical.boundary_windows import CanonicalBoundaryRefinementWindow
from robata.application.canonical.logical_nodes import (
    canonical_action_evidence_result_logical_node,
    canonical_boundary_refinement_result_logical_node,
    canonical_boundary_role_result_logical_node,
    canonical_boundary_window_logical_node,
    canonical_call_barrier_logical_node,
    canonical_call_part_logical_node,
    canonical_call_reduction_logical_node,
    canonical_candidate_dense_window_logical_node,
    canonical_candidate_event_logical_node,
    canonical_candidate_reduction_logical_node,
    canonical_coarse_qa_logical_node,
    canonical_dense_qa_result_logical_node,
    canonical_enrichment_logical_node,
    canonical_event_hypothesis_logical_node,
    canonical_event_proposal_result_logical_node,
    canonical_fusion_reduction_logical_node,
    canonical_input_plan_logical_node,
    canonical_output_decision_logical_node,
    canonical_package_set_logical_node,
    canonical_parsed_claim_logical_node,
    canonical_provisional_fusion_result_logical_node,
    canonical_provisional_physical_action_logical_node,
    canonical_qa_completion_logical_node,
    canonical_root_window_logical_node,
    canonical_selected_output_logical_node,
    canonical_selection_logical_node,
)
from robata.application.canonical.models import (
    CANONICAL_OFFLINE_PIPELINE_VERSION,
    CanonicalCandidateDenseWindow,
    CanonicalOfflineError,
    CanonicalOfflinePartResult,
    CanonicalOfflinePartStatus,
    CanonicalOfflineRunStatus,
    CanonicalRootWindow,
    NonNegativeInt,
)
from robata.application.canonical.output_admission import (
    CanonicalFinalFusionContext,
    CanonicalOutputAdmissionDecision,
    validate_final_fusion_reduction,
)
from robata.application.canonical.product_qa import (
    CanonicalProductQAContext,
    CanonicalProductQAProjector,
)
from robata.application.canonical.reduction import CanonicalFusionReduction
from robata.application.canonical.runner_support import _rfc3339_datetime
from robata.application.canonical_run_membership import (
    CanonicalProcessingRunPrimaryStatus,
    CanonicalProcessingRunRecord,
    canonical_first_work_item_id,
)
from robata.contracts.admission_v2 import AlignmentManifestV2
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import NanosecondInterval, Sha256Digest, StrictModel
from robata.contracts.logical_nodes import (
    LogicalNode,
    OpaqueUuid,
    ProcessingRunNodeMembership,
    RunNodeDisposition,
    RunNodeRole,
)
from robata.contracts.pipeline import SamplingPurpose
from robata.contracts.temporal import TemporalPackageSet
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementOutcome,
    BoundaryRefinementProjector,
    BoundaryRefinementResult,
    BoundaryRefinementRole,
    BoundaryRefinementRoleResult,
)
from robata.event_pipeline.candidate import (
    CandidateReductionResult,
    CanonicalCandidateEvent,
)
from robata.event_pipeline.evidence import (
    ActionEvidenceOutcome,
    ActionEvidenceProjector,
    ActionEvidenceResult,
)
from robata.event_pipeline.identity_registry import (
    AdmissionEvidenceClass,
    EventIdentityBatchResult,
    PlatformEnrichedEventHypothesis,
    PlatformEnrichedOutputReference,
    ProductionAdmittedHypothesisFact,
)
from robata.event_pipeline.proposer import EventProposalOutcome, EventProposalResult
from robata.event_pipeline.provisional_fusion import (
    ProvisionalFusionResult,
    ProvisionalPhysicalAction,
    ProvisionalPhysicalActionFuser,
)
from robata.inference.call_barrier import InferenceCallReduction
from robata.inference.enrichment import (
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    ProviderReferenceCatalog,
    RawProviderResponseArtifact,
    SelectedAttemptOutput,
)
from robata.inference.input_plan import InferenceInputPlan
from robata.inference.models import InferenceAttemptSelection, ModelInference, VisionTask
from robata.qa_pipeline.coarse import CoarseQAResult
from robata.qa_pipeline.completion import (
    QACompletionProjector,
    QACompletionResult,
    QACompletionStatus,
)
from robata.qa_pipeline.dense import DenseQAUnitEvidence, DenseQAWorkUnit
from robata.qa_pipeline.product import ProductQACascadeResult


class CanonicalDenseQAExecution(StrictModel):
    """Exact successful local execution chain for one planned QA_DENSE unit."""

    work_unit: DenseQAWorkUnit
    window: CanonicalRootWindow
    package_set: TemporalPackageSet
    input_plan: InferenceInputPlan
    reference_catalog: ProviderReferenceCatalog
    part_results: tuple[CanonicalOfflinePartResult, ...]
    barrier_reduction: InferenceCallReduction
    unit_evidence: DenseQAUnitEvidence
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        unit = self.work_unit
        window = self.window
        package_set = self.package_set
        input_plan = self.input_plan
        call_plan = input_plan.call_plan
        package_ids = tuple(member.package_id for member in package_set.members)

        if (
            window.purpose is not SamplingPurpose.QA_DENSE
            or window.requested_interval != unit.effective_interval
            or window.interval != unit.effective_interval
            or package_set.window_id != window.window_id
            or package_set.mcap_id != window.mcap_id
            or package_set.camera_mapping_run_id != window.camera_mapping_run_id
            or package_set.alignment_id != window.alignment_id
            or package_set.requested_start_ns != window.requested_interval.start_ns
            or package_set.requested_end_ns != window.requested_interval.end_ns
            or package_set.start_ns != window.interval.start_ns
            or package_set.end_ns != window.interval.end_ns
            or package_set.lineage.source_content_sha256 != window.source_content_sha256
            or package_set.lineage.window_semantic_sha256 != window.semantic_sha256
            or package_set.lineage.camera_mapping_semantic_sha256
            != window.camera_mapping_semantic_sha256
            or package_set.lineage.alignment_semantic_sha256 != window.alignment_semantic_sha256
        ):
            raise ValueError("dense QA package lineage does not match its work-unit window")

        package_facts = tuple(
            (
                member.package_id,
                member.ordinal,
                member.package_semantic_content_sha256,
                member.package_manifest_sha256,
            )
            for member in package_set.members
        )
        subject_facts = tuple(
            (
                package.package_id,
                package.ordinal,
                package.semantic_content_sha256,
                package.manifest_bytes_sha256,
            )
            for package in input_plan.subject.packages
        )
        catalog_facts = tuple(
            (
                package.package_id,
                package.ordinal,
                package.semantic_content_sha256,
                package.manifest_bytes_sha256,
            )
            for package in input_plan.request_catalog.packages
        )
        if (
            input_plan.subject.task is not VisionTask.QA_DENSE
            or input_plan.request_catalog.task is not VisionTask.QA_DENSE
            or subject_facts != package_facts
            or catalog_facts != package_facts
            or any(
                tuple(camera.camera_id for camera in package.cameras) != CAMERA_IDS
                for package in input_plan.request_catalog.packages
            )
        ):
            raise ValueError("dense QA input plan does not bind the exact six-camera packages")

        reference_catalog = self.reference_catalog
        expected_entries = ProviderReferenceCatalog.derive_entries(
            request_catalog_sha256=input_plan.request_catalog.semantic_sha256,
            rendered_items=input_plan.rendered_items,
            token_policy_version=reference_catalog.token_policy_version,
        )
        if (
            reference_catalog.task is not VisionTask.QA_DENSE
            or reference_catalog.input_plan_id != input_plan.input_plan_id
            or reference_catalog.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or reference_catalog.request_catalog_id != input_plan.request_catalog.request_catalog_id
            or reference_catalog.request_catalog_sha256
            != input_plan.request_catalog.semantic_sha256
            or reference_catalog.entries != expected_entries
        ):
            raise ValueError("dense QA reference catalog does not match its input plan")

        if len(self.part_results) != len(call_plan.parts) or tuple(
            item.part_ordinal for item in self.part_results
        ) != tuple(range(len(call_plan.parts))):
            raise ValueError("dense QA part results must exactly cover the declared call plan")

        enriched_outputs: list[OrchestratorEnrichedOutput] = []
        for result, planned in zip(self.part_results, call_plan.parts, strict=True):
            terminal = result.terminal
            selection = result.selection
            completion = result.completion
            parsed = result.parsed_claims
            selected = result.selected_output
            enriched = result.enriched_output
            if (
                result.status is not CanonicalOfflinePartStatus.ENRICHED
                or selection is None
                or parsed is None
                or selected is None
                or enriched is None
                or result.part_count != len(call_plan.parts)
                or result.part_semantic_sha256 != planned.part_semantic_sha256
                or terminal.stage is not VisionTask.QA_DENSE
                or terminal.mcap_id != window.mcap_id
                or terminal.package_set_id != package_set.package_set_id
                or terminal.package_ids != package_ids
                or terminal.camera_mapping_run_id != window.camera_mapping_run_id
                or terminal.alignment_id != window.alignment_id
                or terminal.start_ns != window.interval.start_ns
                or terminal.end_ns != window.interval.end_ns
                or terminal.input_plan_id != input_plan.input_plan_id
                or terminal.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or terminal.input_plan_part_ordinal != planned.ordinal
                or terminal.input_plan_part_count != planned.part_count
                or terminal.input_plan_part_semantic_sha256 != planned.part_semantic_sha256
                or parsed.task is not VisionTask.QA_DENSE
                or parsed.payload.model_dump(mode="json") != terminal.normalized_output
                or enriched.task is not VisionTask.QA_DENSE
                or enriched.input_plan_id != input_plan.input_plan_id
                or enriched.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or enriched.request_catalog_id != input_plan.request_catalog.request_catalog_id
                or enriched.request_catalog_sha256 != input_plan.request_catalog.semantic_sha256
                or enriched.reference_catalog_id != reference_catalog.reference_catalog_id
                or enriched.reference_catalog_sha256 != reference_catalog.semantic_sha256
                or enriched.authority.recording_identity != window.recording_identity
                or enriched.authority.mcap_id != window.mcap_id
                or enriched.authority.camera_mapping_run_id != window.camera_mapping_run_id
                or enriched.authority.alignment_id != window.alignment_id
                or completion.barrier_semantic_sha256 != call_plan.barrier_semantic_sha256
                or completion.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or completion.call_plan_sha256 != call_plan.call_plan_sha256
                or completion.part_logical_key != planned.part_logical_key
                or completion.part_idempotency_key != planned.idempotency_key
                or completion.normalized_output != terminal.normalized_output
            ):
                raise ValueError("dense QA part result differs from its exact execution lineage")
            enriched_outputs.append(enriched)

        reduction = self.barrier_reduction
        if (
            reduction.barrier_semantic_sha256 != call_plan.barrier_semantic_sha256
            or reduction.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or reduction.call_plan_sha256 != call_plan.call_plan_sha256
            or reduction.reduction_policy != call_plan.reduction_policy
            or reduction.reduction_policy_version != call_plan.reduction_policy_version
            or reduction.ordered_completion_ids
            != tuple(item.completion.completion_id for item in self.part_results)
            or reduction.ordered_part_semantic_sha256s
            != tuple(item.part_semantic_sha256 for item in self.part_results)
            or reduction.ordered_normalized_output_sha256s
            != tuple(item.completion.normalized_output_sha256 for item in self.part_results)
            or reduction.ordered_selection_decision_logical_keys
            != tuple(
                item.selection.selection_decision_logical_key
                for item in self.part_results
                if item.selection is not None
            )
        ):
            raise ValueError("dense QA reduction does not bind the exact successful call set")

        evidence = self.unit_evidence
        evidence_package_facts = tuple(
            (
                package.package_id,
                package.ordinal,
                package.interval.start_ns,
                package.interval.end_ns,
                package.semantic_content_sha256,
                package.manifest_sha256,
            )
            for package in evidence.packages
        )
        member_evidence_facts = tuple(
            (
                member.package_id,
                member.ordinal,
                member.start_ns,
                member.end_ns,
                member.package_semantic_content_sha256,
                member.package_manifest_sha256,
            )
            for member in package_set.members
        )
        expected_output_facts = tuple(
            sorted(
                (
                    output.artifact_id,
                    output.semantic_sha256,
                    output.enrichment_logical_key,
                    output.selected_attempt.inference_id,
                    output.input_plan_id,
                    output.input_plan_semantic_sha256,
                    output.task.value,
                )
                for output in enriched_outputs
            )
        )
        evidence_output_facts = tuple(
            (
                output.artifact_id,
                output.semantic_sha256,
                output.enrichment_logical_key,
                output.inference_id,
                output.input_plan_id,
                output.input_plan_semantic_sha256,
                output.task,
            )
            for output in evidence.source_outputs
        )
        expected_coordinates = tuple(
            (member.ordinal, camera_id)
            for member in package_set.members
            for camera_id in CAMERA_IDS
        )
        actual_coordinates = tuple(
            (item.package_ordinal, item.camera_id) for item in evidence.package_camera_results
        )
        if (
            evidence.unit_id != unit.unit_id
            or evidence.unit_semantic_digest != unit.semantic_digest
            or evidence.mcap_id != package_set.mcap_id
            or evidence.camera_mapping_run_id != package_set.camera_mapping_run_id
            or evidence.camera_mapping_semantic_sha256
            != package_set.lineage.camera_mapping_semantic_sha256
            or evidence.alignment_id != package_set.alignment_id
            or evidence.alignment_semantic_sha256 != package_set.lineage.alignment_semantic_sha256
            or evidence.package_set_id != package_set.package_set_id
            or evidence.split_plan_digest != package_set.split_plan_digest
            or evidence.member_manifest_sha256 != package_set.member_manifest_sha256
            or evidence_package_facts != member_evidence_facts
            or evidence.input_plan.input_plan_id != input_plan.input_plan_id
            or evidence.input_plan.semantic_sha256 != input_plan.semantic_sha256
            or evidence.input_plan.task != VisionTask.QA_DENSE.value
            or evidence.input_plan.package_ids != package_ids
            or evidence_output_facts != expected_output_facts
            or actual_coordinates != expected_coordinates
        ):
            raise ValueError("dense QA unit evidence does not match the exact execution")
        return self


class CanonicalActionEvidenceExecution(StrictModel):
    """Exact successful ACTION_DENSE to ACTION_EVIDENCE chain for one candidate."""

    candidate: CanonicalCandidateEvent
    window: CanonicalCandidateDenseWindow
    package_set: TemporalPackageSet
    input_plan: InferenceInputPlan
    reference_catalog: ProviderReferenceCatalog
    part_results: tuple[CanonicalOfflinePartResult, ...]
    barrier_reduction: InferenceCallReduction
    evidence_result: ActionEvidenceResult
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        candidate = self.candidate
        window = self.window
        package_set = self.package_set
        input_plan = self.input_plan
        parts = input_plan.call_plan.parts
        package_facts = tuple(
            (
                item.package_id,
                item.ordinal,
                item.package_semantic_content_sha256,
                item.package_manifest_sha256,
            )
            for item in package_set.members
        )
        subject_facts = tuple(
            (
                item.package_id,
                item.ordinal,
                item.semantic_content_sha256,
                item.manifest_bytes_sha256,
            )
            for item in input_plan.subject.packages
        )
        if (
            window.purpose is not SamplingPurpose.ACTION_DENSE
            or window.candidate_event_id != candidate.candidate_event_id
            or window.candidate_logical_key != candidate.candidate_logical_key
            or window.candidate_effective_interval != candidate.effective_interval
            or window.requested_interval != candidate.requested_dense_interval
            or package_set.window_id != window.window_id
            or package_set.mcap_id != window.mcap_id
            or package_set.camera_mapping_run_id != window.camera_mapping_run_id
            or package_set.alignment_id != window.alignment_id
            or package_set.requested_start_ns != window.requested_interval.start_ns
            or package_set.requested_end_ns != window.requested_interval.end_ns
            or package_set.start_ns != window.interval.start_ns
            or package_set.end_ns != window.interval.end_ns
            or package_set.lineage.window_semantic_sha256 != window.semantic_sha256
            or input_plan.subject.task is not VisionTask.ACTION_EVIDENCE
            or input_plan.request_catalog.task is not VisionTask.ACTION_EVIDENCE
            or subject_facts != package_facts
        ):
            raise ValueError("action evidence execution has inconsistent candidate/package lineage")
        expected_entries = ProviderReferenceCatalog.derive_entries(
            request_catalog_sha256=input_plan.request_catalog.semantic_sha256,
            rendered_items=input_plan.rendered_items,
            token_policy_version=self.reference_catalog.token_policy_version,
        )
        if (
            self.reference_catalog.task is not VisionTask.ACTION_EVIDENCE
            or self.reference_catalog.input_plan_id != input_plan.input_plan_id
            or self.reference_catalog.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or self.reference_catalog.entries != expected_entries
        ):
            raise ValueError("action evidence reference catalog differs from its input plan")
        if len(self.part_results) != len(parts) or tuple(
            item.part_ordinal for item in self.part_results
        ) != tuple(range(len(parts))):
            raise ValueError("action evidence part results do not exactly cover the call plan")
        for result, planned in zip(self.part_results, parts, strict=True):
            enriched = result.enriched_output
            if (
                result.status is not CanonicalOfflinePartStatus.ENRICHED
                or result.selection is None
                or result.parsed_claims is None
                or result.selected_output is None
                or enriched is None
                or result.part_semantic_sha256 != planned.part_semantic_sha256
                or result.terminal.stage is not VisionTask.ACTION_EVIDENCE
                or result.terminal.package_set_id != package_set.package_set_id
                or result.terminal.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or result.parsed_claims.task is not VisionTask.ACTION_EVIDENCE
                or enriched.task is not VisionTask.ACTION_EVIDENCE
                or enriched.input_plan_semantic_sha256 != input_plan.semantic_sha256
            ):
                raise ValueError("action evidence part differs from its declared execution")
        reduction = self.barrier_reduction
        if (
            reduction.barrier_semantic_sha256 != input_plan.call_plan.barrier_semantic_sha256
            or reduction.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or reduction.ordered_completion_ids
            != tuple(item.completion.completion_id for item in self.part_results)
        ):
            raise ValueError("action evidence reduction does not bind the exact call set")
        expected_evidence = ActionEvidenceProjector().project(
            input_plan=input_plan,
            package_set=package_set,
            candidate=candidate,
            enriched_outputs=tuple(
                item.enriched_output
                for item in self.part_results
                if item.enriched_output is not None
            ),
        )
        if self.evidence_result != expected_evidence:
            raise ValueError("normalized action evidence differs from retained enriched claims")
        return self


class CanonicalBoundaryRefinementPassExecution(StrictModel):
    """Exact successful inference closure for one role-bound boundary pass."""

    window: CanonicalBoundaryRefinementWindow
    alignment_manifest: AlignmentManifestV2
    package_set: TemporalPackageSet
    input_plan: InferenceInputPlan
    reference_catalog: ProviderReferenceCatalog
    part_results: tuple[CanonicalOfflinePartResult, ...]
    barrier_reduction: InferenceCallReduction
    role_result: BoundaryRefinementRoleResult
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        window = self.window
        alignment = self.alignment_manifest
        package_set = self.package_set
        input_plan = self.input_plan
        call_plan = input_plan.call_plan
        role_result = self.role_result
        role = role_result.role
        package_ids = tuple(member.package_id for member in package_set.members)

        if (
            window.purpose is not SamplingPurpose.BOUNDARY_REFINEMENT
            or window.refinement_role is not role
            or window.provisional_action_logical_key != role_result.source_action_logical_key
            or window.provisional_action_semantic_sha256
            != role_result.source_action_semantic_sha256
            or window.coarse_interval != role_result.coarse_interval
            or window.boundary_anchor_ns != role_result.coarse_anchor_ns
            or window.padding_before_ns != role_result.policy.padding_before_ns
            or window.padding_after_ns != role_result.policy.padding_after_ns
            or window.production_eligible
            or role_result.used_fallback
            or role_result.production_eligible
        ):
            raise ValueError("boundary role result differs from its role-bound window")

        if (
            alignment.recording_identity != window.recording_identity
            or alignment.mcap_id != window.mcap_id
            or alignment.source_content_sha256 != window.source_content_sha256
            or alignment.camera_mapping_run_id != window.camera_mapping_run_id
            or alignment.camera_mapping_semantic_sha256 != window.camera_mapping_semantic_sha256
            or alignment.alignment_id != window.alignment_id
            or alignment.alignment_semantic_sha256 != window.alignment_semantic_sha256
            or alignment.reference_timebase != window.reference_timebase
        ):
            raise ValueError("boundary alignment manifest differs from its canonical window")

        if (
            package_set.window_id != window.window_id
            or package_set.mcap_id != window.mcap_id
            or package_set.camera_mapping_run_id != window.camera_mapping_run_id
            or package_set.alignment_id != window.alignment_id
            or package_set.requested_start_ns != window.requested_interval.start_ns
            or package_set.requested_end_ns != window.requested_interval.end_ns
            or package_set.start_ns != window.interval.start_ns
            or package_set.end_ns != window.interval.end_ns
            or package_set.lineage.source_content_sha256 != window.source_content_sha256
            or package_set.lineage.window_semantic_sha256 != window.semantic_sha256
            or package_set.lineage.camera_mapping_semantic_sha256
            != window.camera_mapping_semantic_sha256
            or package_set.lineage.alignment_semantic_sha256 != window.alignment_semantic_sha256
        ):
            raise ValueError("boundary package set differs from its role-bound window")

        package_facts = tuple(
            (
                member.package_id,
                member.ordinal,
                member.package_semantic_content_sha256,
                member.package_manifest_sha256,
            )
            for member in package_set.members
        )
        subject_facts = tuple(
            (
                package.package_id,
                package.ordinal,
                package.semantic_content_sha256,
                package.manifest_bytes_sha256,
            )
            for package in input_plan.subject.packages
        )
        catalog_facts = tuple(
            (
                package.package_id,
                package.ordinal,
                package.semantic_content_sha256,
                package.manifest_bytes_sha256,
            )
            for package in input_plan.request_catalog.packages
        )
        if (
            input_plan.subject.task is not VisionTask.BOUNDARY_REFINEMENT
            or input_plan.request_catalog.task is not VisionTask.BOUNDARY_REFINEMENT
            or subject_facts != package_facts
            or catalog_facts != package_facts
            or any(
                tuple(camera.camera_id for camera in package.cameras) != CAMERA_IDS
                for package in input_plan.request_catalog.packages
            )
        ):
            raise ValueError("boundary input plan does not bind the exact six-camera packages")

        reference_catalog = self.reference_catalog
        expected_entries = ProviderReferenceCatalog.derive_entries(
            request_catalog_sha256=input_plan.request_catalog.semantic_sha256,
            rendered_items=input_plan.rendered_items,
            token_policy_version=reference_catalog.token_policy_version,
        )
        if (
            reference_catalog.task is not VisionTask.BOUNDARY_REFINEMENT
            or reference_catalog.input_plan_id != input_plan.input_plan_id
            or reference_catalog.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or reference_catalog.request_catalog_id != input_plan.request_catalog.request_catalog_id
            or reference_catalog.request_catalog_sha256
            != input_plan.request_catalog.semantic_sha256
            or reference_catalog.entries != expected_entries
        ):
            raise ValueError("boundary reference catalog differs from its input plan")

        if len(self.part_results) != len(call_plan.parts) or tuple(
            item.part_ordinal for item in self.part_results
        ) != tuple(range(len(call_plan.parts))):
            raise ValueError("boundary part results must exactly cover the call plan")

        enriched_outputs: list[OrchestratorEnrichedOutput] = []
        for result, planned in zip(self.part_results, call_plan.parts, strict=True):
            terminal = result.terminal
            selection = result.selection
            parsed = result.parsed_claims
            selected = result.selected_output
            enriched = result.enriched_output
            completion = result.completion
            dependency = terminal.input_config
            if (
                result.status is not CanonicalOfflinePartStatus.ENRICHED
                or selection is None
                or parsed is None
                or selected is None
                or enriched is None
                or enriched.abstained
                or result.part_count != len(call_plan.parts)
                or result.part_semantic_sha256 != planned.part_semantic_sha256
                or terminal.stage is not VisionTask.BOUNDARY_REFINEMENT
                or terminal.mcap_id != window.mcap_id
                or terminal.package_set_id != package_set.package_set_id
                or terminal.package_ids != package_ids
                or terminal.camera_mapping_run_id != window.camera_mapping_run_id
                or terminal.alignment_id != window.alignment_id
                or terminal.start_ns != window.interval.start_ns
                or terminal.end_ns != window.interval.end_ns
                or terminal.input_plan_id != input_plan.input_plan_id
                or terminal.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or terminal.input_plan_part_ordinal != planned.ordinal
                or terminal.input_plan_part_count != planned.part_count
                or terminal.input_plan_part_semantic_sha256 != planned.part_semantic_sha256
                or terminal.output_schema_sha256
                != input_plan.prompt_output.provider_response_schema_sha256
                or parsed.task is not VisionTask.BOUNDARY_REFINEMENT
                or parsed.payload.model_dump(mode="json") != terminal.normalized_output
                or enriched.task is not VisionTask.BOUNDARY_REFINEMENT
                or enriched.input_plan_id != input_plan.input_plan_id
                or enriched.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or enriched.request_catalog_id != input_plan.request_catalog.request_catalog_id
                or enriched.request_catalog_sha256 != input_plan.request_catalog.semantic_sha256
                or enriched.reference_catalog_id != reference_catalog.reference_catalog_id
                or enriched.reference_catalog_sha256 != reference_catalog.semantic_sha256
                or enriched.authority.recording_identity != window.recording_identity
                or enriched.authority.mcap_id != window.mcap_id
                or enriched.authority.camera_mapping_run_id != window.camera_mapping_run_id
                or enriched.authority.alignment_id != window.alignment_id
                or completion.barrier_semantic_sha256 != call_plan.barrier_semantic_sha256
                or completion.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or completion.call_plan_sha256 != call_plan.call_plan_sha256
                or completion.part_logical_key != planned.part_logical_key
                or completion.part_idempotency_key != planned.idempotency_key
                or completion.normalized_output != terminal.normalized_output
                or dependency.get("provisional_fusion_semantic_sha256")
                != window.provisional_fusion_semantic_sha256
                or dependency.get("provisional_physical_action_logical_key")
                != window.provisional_action_logical_key
                or dependency.get("provisional_physical_action_semantic_sha256")
                != window.provisional_action_semantic_sha256
                or dependency.get("boundary_refinement_role") != role.value
                or dependency.get("boundary_anchor_ns") != window.boundary_anchor_ns
                or dependency.get("boundary_refinement_window_semantic_sha256")
                != window.semantic_sha256
                or dependency.get("boundary_refinement_policy_version")
                != role_result.policy.version
                or dependency.get("boundary_refinement_policy_semantic_sha256")
                != role_result.policy.semantic_sha256
            ):
                raise ValueError("boundary part differs from its exact execution lineage")
            enriched_outputs.append(enriched)

        reduction = self.barrier_reduction
        if (
            reduction.barrier_semantic_sha256 != call_plan.barrier_semantic_sha256
            or reduction.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or reduction.call_plan_sha256 != call_plan.call_plan_sha256
            or reduction.reduction_policy != call_plan.reduction_policy
            or reduction.reduction_policy_version != call_plan.reduction_policy_version
            or reduction.output_schema_sha256
            != input_plan.prompt_output.provider_response_schema_sha256
            or reduction.ordered_completion_ids
            != tuple(item.completion.completion_id for item in self.part_results)
            or reduction.ordered_part_semantic_sha256s
            != tuple(item.part_semantic_sha256 for item in self.part_results)
            or reduction.ordered_normalized_output_sha256s
            != tuple(item.completion.normalized_output_sha256 for item in self.part_results)
            or reduction.ordered_selection_decision_logical_keys
            != tuple(
                item.selection.selection_decision_logical_key
                for item in self.part_results
                if item.selection is not None
            )
            or any(item.completion.barrier_id != reduction.barrier_id for item in self.part_results)
        ):
            raise ValueError("boundary reduction does not bind the exact successful call set")

        expected_packages = tuple(
            (
                member.package_id,
                member.ordinal,
                member.start_ns,
                member.end_ns,
                member.package_semantic_content_sha256,
                member.package_manifest_sha256,
            )
            for member in package_set.members
        )
        actual_packages = tuple(
            (
                package.package_id,
                package.package_ordinal,
                package.interval.start_ns,
                package.interval.end_ns,
                package.semantic_content_sha256,
                package.manifest_sha256,
            )
            for package in role_result.packages
        )
        expected_outputs = tuple(
            (
                planned.ordinal,
                planned.part_semantic_sha256,
                output.authority.inference_id,
                output.artifact_id,
                output.selected_attempt.output_sha256,
                output.selected_attempt.selection_decision_logical_key,
                output.enrichment_logical_key,
            )
            for planned, output in zip(call_plan.parts, enriched_outputs, strict=True)
        )
        actual_outputs = tuple(
            (
                output.part_ordinal,
                output.part_semantic_sha256,
                output.source_inference_id,
                output.source_artifact_id,
                output.selected_output_sha256,
                output.selection_decision_logical_key,
                output.enrichment_logical_key,
            )
            for output in role_result.source_outputs
        )
        if (
            role_result.task is not VisionTask.BOUNDARY_REFINEMENT
            or role_result.recording_identity != window.recording_identity
            or role_result.mcap_id != window.mcap_id
            or role_result.source_content_sha256 != window.source_content_sha256
            or role_result.camera_mapping_run_id != window.camera_mapping_run_id
            or role_result.camera_mapping_semantic_sha256 != window.camera_mapping_semantic_sha256
            or role_result.alignment_id != window.alignment_id
            or role_result.alignment_semantic_sha256 != window.alignment_semantic_sha256
            or role_result.window_semantic_sha256 != window.semantic_sha256
            or role_result.requested_window_interval != window.requested_interval
            or role_result.window_interval != window.interval
            or role_result.package_set_id != package_set.package_set_id
            or role_result.split_plan_digest != package_set.split_plan_digest
            or role_result.member_manifest_sha256 != package_set.member_manifest_sha256
            or actual_packages != expected_packages
            or role_result.input_plan_id != input_plan.input_plan_id
            or role_result.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or actual_outputs != expected_outputs
        ):
            raise ValueError("boundary role result differs from its retained execution closure")
        return self


class CanonicalBoundaryRefinementExecution(StrictModel):
    """Exact dual-role reprojection and reduction for one provisional action."""

    action: ProvisionalPhysicalAction
    onset: CanonicalBoundaryRefinementPassExecution
    offset: CanonicalBoundaryRefinementPassExecution
    result: BoundaryRefinementResult
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        action = self.action
        onset = self.onset
        offset = self.offset
        result = self.result
        if (
            onset.role_result.role is not BoundaryRefinementRole.ONSET
            or offset.role_result.role is not BoundaryRefinementRole.OFFSET
            or onset.window.refinement_role is not BoundaryRefinementRole.ONSET
            or offset.window.refinement_role is not BoundaryRefinementRole.OFFSET
            or onset.alignment_manifest != offset.alignment_manifest
            or onset.window.parent_window_logical_key != offset.window.parent_window_logical_key
            or onset.window.provisional_fusion_semantic_sha256
            != offset.window.provisional_fusion_semantic_sha256
            or onset.window.semantic_sha256 == offset.window.semantic_sha256
            or onset.role_result.policy != result.policy
            or offset.role_result.policy != result.policy
        ):
            raise ValueError("boundary execution lacks one exact ONSET/OFFSET trust closure")

        for execution in (onset, offset):
            role_result = execution.role_result
            window = execution.window
            if (
                window.provisional_action_logical_key != action.logical_key
                or window.provisional_action_semantic_sha256 != action.semantic_sha256
                or window.coarse_interval != action.coarse_interval
                or role_result.source_action_logical_key != action.logical_key
                or role_result.source_action_semantic_sha256 != action.semantic_sha256
                or role_result.source_action_policy_semantic_sha256 != action.policy_semantic_sha256
                or role_result.source_action_ordinal != action.ordinal
                or role_result.action_label != action.label
                or role_result.coarse_interval != action.coarse_interval
                or role_result.mcap_id != action.mcap_id
                or role_result.source_content_sha256 != action.source_content_sha256
                or role_result.camera_mapping_semantic_sha256
                != action.camera_mapping_semantic_sha256
                or role_result.alignment_semantic_sha256 != action.alignment_semantic_sha256
            ):
                raise ValueError("boundary pass differs from its provisional action")

        projector = BoundaryRefinementProjector(result.policy)
        onset_outputs = tuple(
            item.enriched_output for item in onset.part_results if item.enriched_output is not None
        )
        offset_outputs = tuple(
            item.enriched_output for item in offset.part_results if item.enriched_output is not None
        )
        expected_onset = projector.project_role(
            action=action,
            role=BoundaryRefinementRole.ONSET,
            input_plan=onset.input_plan,
            package_set=onset.package_set,
            enriched_outputs=onset_outputs,
            alignment_manifest=onset.alignment_manifest,
        )
        expected_offset = projector.project_role(
            action=action,
            role=BoundaryRefinementRole.OFFSET,
            input_plan=offset.input_plan,
            package_set=offset.package_set,
            enriched_outputs=offset_outputs,
            alignment_manifest=offset.alignment_manifest,
        )
        if onset.role_result != expected_onset or offset.role_result != expected_offset:
            raise ValueError("boundary role result differs from exact enriched reprojection")
        expected_result = projector.reduce(
            action=action,
            onset=expected_onset,
            offset=expected_offset,
        )
        if result != expected_result:
            raise ValueError("boundary result differs from exact dual-role reduction")
        return self


class CanonicalOfflineRunResult(StrictModel):
    """Inspectable local result; this is not a registered persistence schema."""

    schema_version: Literal["1.0"]
    run_id: OpaqueUuid
    processing_run: CanonicalProcessingRunRecord
    run_memberships: tuple[ProcessingRunNodeMembership, ...]
    recording_identity: Sha256Digest
    mcap_id: OpaqueUuid
    execution_policy_sha256: Sha256Digest
    status: CanonicalOfflineRunStatus
    window: CanonicalRootWindow | None
    materialized_package_ids: tuple[OpaqueUuid, ...]
    package_set: TemporalPackageSet | None
    coarse_qa_result: CoarseQAResult | None
    dense_qa_executions: tuple[CanonicalDenseQAExecution, ...] = ()
    qa_completion_result: QACompletionResult | None
    product_qa_context: CanonicalProductQAContext | None = None
    product_qa_result: ProductQACascadeResult | None = None
    event_proposal_result: EventProposalResult | None = None
    candidate_reduction_result: CandidateReductionResult | None = None
    action_evidence_executions: tuple[CanonicalActionEvidenceExecution, ...] = ()
    provisional_fusion_result: ProvisionalFusionResult | None = None
    boundary_refinement_executions: tuple[CanonicalBoundaryRefinementExecution, ...] = ()
    final_fusion_context: CanonicalFinalFusionContext | None = None
    input_plan: InferenceInputPlan | None
    reference_catalog: ProviderReferenceCatalog | None
    part_results: tuple[CanonicalOfflinePartResult, ...]
    barrier_reduction: InferenceCallReduction | None
    fusion_reduction: CanonicalFusionReduction | None
    # Single-part compatibility fields. Multi-part truth lives only in part_results.
    terminal: ModelInference | None
    selection: InferenceAttemptSelection | None
    raw_response: RawProviderResponseArtifact | None
    parsed_claims: ParsedProviderClaimArtifact | None
    selected_output: SelectedAttemptOutput | None
    enriched_output: OrchestratorEnrichedOutput | None
    output_decision: CanonicalOutputAdmissionDecision | None
    hypotheses: tuple[PlatformEnrichedEventHypothesis, ...]
    identity_result: EventIdentityBatchResult | None
    attempt_count: NonNegativeInt
    adapter_infer_calls: NonNegativeInt
    error: CanonicalOfflineError | None

    @model_validator(mode="after")
    def validate_result(self, info: ValidationInfo) -> Self:
        # ``canonical-primary-completion-detail`` v4 predates the internal
        # product-QA context/result bridge. Its published bytes cannot be
        # changed in place, so completion-detail recovery validates the
        # persisted upstream closure with an explicit compatibility context.
        # Fresh canonical runs still use the default strict path below and must
        # retain the complete 21-class product projection before completion.
        validation_context = info.context if isinstance(info.context, dict) else {}
        allow_missing_product_qa = (
            validation_context.get("allow_missing_product_qa", False) is True
        )
        expected_run_status = CanonicalProcessingRunPrimaryStatus(self.status.value)
        if (
            self.processing_run.run_id != self.run_id
            or self.processing_run.recording_identity != self.recording_identity
            or self.processing_run.mcap_id != self.mcap_id
            or self.processing_run.pipeline_version != CANONICAL_OFFLINE_PIPELINE_VERSION
            or self.processing_run.config_sha256 != self.execution_policy_sha256
            or self.processing_run.primary_status is not expected_run_status
            or self.processing_run.completed_at is None
        ):
            raise ValueError("run result does not match its terminal processing-run record")
        if self.window is not None and self.mcap_id != self.window.mcap_id:
            raise ValueError("run result does not match the root window MCAP")
        membership_identities = tuple(item.identity for item in self.run_memberships)
        if len(set(membership_identities)) != len(membership_identities):
            raise ValueError("run result contains duplicate node memberships")
        if any(item.run_id != self.run_id for item in self.run_memberships):
            raise ValueError("run result contains a membership from another run")
        completed_at = _rfc3339_datetime(self.processing_run.completed_at)
        started_at = _rfc3339_datetime(self.processing_run.started_at)
        if any(
            not started_at <= _rfc3339_datetime(item.attached_at) <= completed_at
            for item in self.run_memberships
        ):
            raise ValueError("run membership timestamps lie outside the processing run")
        if self.window is not None and self.window.recording_identity != self.recording_identity:
            raise ValueError("run window crosses recording scope")
        if self.package_set is not None:
            if self.window is None or self.package_set.mcap_id != self.mcap_id:
                raise ValueError("run package set does not match its root MCAP lineage")
            expected_ids = tuple(member.package_id for member in self.package_set.members)
            if self.materialized_package_ids != expected_ids:
                raise ValueError("run package IDs do not match the package set")
        elif self.materialized_package_ids:
            raise ValueError("materialized package IDs require a package set")
        if self.coarse_qa_result is not None:
            coarse = self.coarse_qa_result
            if (
                self.package_set is None
                or coarse.package_set_id != self.package_set.package_set_id
                or coarse.mcap_id != self.mcap_id
                or coarse.package_ids != self.materialized_package_ids
                or coarse.production_eligible
            ):
                raise ValueError("coarse QA result does not match the run package lineage")
        if self.dense_qa_executions:
            if self.coarse_qa_result is None or self.qa_completion_result is None:
                raise ValueError("dense QA executions require retained coarse and completion facts")
            manifest_units = self.qa_completion_result.dense_work_manifest.units
            execution_units = tuple(item.work_unit for item in self.dense_qa_executions)
            if execution_units != manifest_units[: len(execution_units)] or any(
                item.window.recording_identity != self.recording_identity
                or item.window.mcap_id != self.mcap_id
                or any(
                    part.terminal.input_config.get("canonical_execution_policy_sha256")
                    != self.execution_policy_sha256
                    for part in item.part_results
                )
                for item in self.dense_qa_executions
            ):
                raise ValueError("dense QA executions do not match the run or manifest lineage")
        if self.qa_completion_result is not None:
            if self.coarse_qa_result is None:
                raise ValueError("QA completion requires its retained coarse QA result")
            completion_projector = QACompletionProjector(self.qa_completion_result.policy_version)
            recording_interval = (
                NanosecondInterval(
                    start_ns=0,
                    end_ns=self.window.recording_duration_ns,
                )
                if self.window is not None
                else None
            )
            dense_failure_code = self.qa_completion_result.dense_failure_code
            expected_completion = (
                completion_projector.block(
                    self.coarse_qa_result,
                    dense_failure_code,
                    recording_interval=recording_interval,
                )
                if dense_failure_code is not None
                else completion_projector.project(
                    self.coarse_qa_result,
                    self.qa_completion_result.dense_result,
                    recording_interval=recording_interval,
                )
            )
            if self.qa_completion_result != expected_completion:
                raise ValueError("QA completion does not exactly project retained coarse/dense QA")
            dense_result = self.qa_completion_result.dense_result
            if dense_result is None:
                if len(self.dense_qa_executions) > len(
                    self.qa_completion_result.dense_work_manifest.units
                ):
                    raise ValueError("dense QA executions exceed the planned work manifest")
            elif (
                tuple(item.work_unit.unit_id for item in self.dense_qa_executions)
                != tuple(item.work_unit_id for item in dense_result.units)
                or tuple(item.work_unit.semantic_digest for item in self.dense_qa_executions)
                != tuple(item.work_unit_semantic_digest for item in dense_result.units)
                or tuple(item.unit_evidence for item in self.dense_qa_executions)
                != tuple(item.evidence for item in dense_result.units)
            ):
                raise ValueError("dense QA result does not exactly cover retained executions")
        if self.event_proposal_result is not None:
            proposal = self.event_proposal_result
            if (
                proposal.task is not VisionTask.EVENT_PROPOSAL
                or proposal.production_eligible
                or self.package_set is None
            ):
                raise ValueError("event proposal result has invalid canonical binding")
        if self.candidate_reduction_result is not None:
            reduction = self.candidate_reduction_result
            if (
                reduction.production_eligible
                or self.event_proposal_result is None
                or reduction.source_event_proposal_result_semantic_sha256
                != self.event_proposal_result.semantic_sha256
                or any(item.production_eligible for item in reduction.candidates)
            ):
                raise ValueError("candidate reduction result has invalid canonical binding")
        elif self.event_proposal_result is not None:
            raise ValueError("event proposal result requires candidate reduction result")
        if self.action_evidence_executions:
            if self.candidate_reduction_result is None:
                raise ValueError("action evidence executions require candidate reduction")
            expected_candidates = self.candidate_reduction_result.candidates
            actual_candidates = tuple(item.candidate for item in self.action_evidence_executions)
            if actual_candidates != expected_candidates[: len(actual_candidates)] or any(
                item.window.recording_identity != self.recording_identity
                or item.window.mcap_id != self.mcap_id
                or item.production_eligible
                for item in self.action_evidence_executions
            ):
                raise ValueError("action evidence executions are not the ordered candidate prefix")
        if self.provisional_fusion_result is not None:
            if self.candidate_reduction_result is None:
                raise ValueError("provisional fusion requires candidate reduction")
            expected_fusion = ProvisionalPhysicalActionFuser(
                self.provisional_fusion_result.policy
            ).fuse(
                self.candidate_reduction_result,
                tuple(item.evidence_result for item in self.action_evidence_executions),
            )
            if self.provisional_fusion_result != expected_fusion:
                raise ValueError("provisional fusion differs from retained candidate evidence")
        if self.boundary_refinement_executions:
            if (
                self.provisional_fusion_result is None
                or self.window is None
                or self.qa_completion_result is None
                or self.candidate_reduction_result is None
            ):
                raise ValueError(
                    "boundary refinement executions require their complete upstream closure"
                )
            provisional_actions = self.provisional_fusion_result.actions
            actual_actions = tuple(item.action for item in self.boundary_refinement_executions)
            if actual_actions != provisional_actions[: len(actual_actions)]:
                raise ValueError("boundary refinement executions are not the ordered action prefix")
            action_evidence_digests = [
                item.evidence_result.semantic_sha256 for item in self.action_evidence_executions
            ]
            for execution in self.boundary_refinement_executions:
                if (
                    execution.production_eligible
                    or execution.action.mcap_id != self.mcap_id
                    or execution.onset.window.recording_identity != self.recording_identity
                    or execution.offset.window.recording_identity != self.recording_identity
                    or execution.onset.window.mcap_id != self.mcap_id
                    or execution.offset.window.mcap_id != self.mcap_id
                    or execution.onset.window.parent_window_logical_key
                    != self.window.window_logical_key
                    or execution.offset.window.parent_window_logical_key
                    != self.window.window_logical_key
                    or execution.onset.window.provisional_fusion_semantic_sha256
                    != self.provisional_fusion_result.semantic_sha256
                    or execution.offset.window.provisional_fusion_semantic_sha256
                    != self.provisional_fusion_result.semantic_sha256
                ):
                    raise ValueError(
                        "boundary refinement execution crosses the canonical run closure"
                    )
                for role_execution in (execution.onset, execution.offset):
                    for part in role_execution.part_results:
                        dependency = part.terminal.input_config
                        if (
                            dependency.get("canonical_execution_policy_sha256")
                            != self.execution_policy_sha256
                            or dependency.get("qa_completion_semantic_sha256")
                            != self.qa_completion_result.semantic_sha256
                            or dependency.get("candidate_reduction_semantic_sha256")
                            != self.candidate_reduction_result.semantic_sha256
                            or dependency.get("action_evidence_result_semantic_sha256s")
                            != action_evidence_digests
                        ):
                            raise ValueError(
                                "boundary calls do not bind the exact upstream run closure"
                            )
        product_context = self.product_qa_context
        product_result = self.product_qa_result
        if not allow_missing_product_qa and product_result is not None:
            if (
                product_context is None
                or self.coarse_qa_result is None
                or self.window is None
                or self.qa_completion_result is None
            ):
                raise ValueError(
                    "product QA result requires retained context, coarse QA, and completion"
                )
            expected_product_result = CanonicalProductQAProjector(
                product_result.policy_version
            ).project(
                recording_id=self.recording_identity,
                recording_duration_ns=self.window.recording_duration_ns,
                coarse_result=self.coarse_qa_result,
                qa_completion_result=self.qa_completion_result,
                candidate_reduction_result=self.candidate_reduction_result,
                action_evidence_results=tuple(
                    item.evidence_result for item in self.action_evidence_executions
                ),
                boundary_results=tuple(
                    item.result for item in self.boundary_refinement_executions
                ),
                context=product_context,
                pipeline_incomplete=self.status not in {
                    CanonicalOfflineRunStatus.SUCCEEDED,
                    CanonicalOfflineRunStatus.NO_EVENTS,
                    CanonicalOfflineRunStatus.ABSTAINED,
                },
                pipeline_abstained=self.status is CanonicalOfflineRunStatus.ABSTAINED,
            )
            if product_result != expected_product_result:
                raise ValueError("product QA result differs from retained canonical evidence")
        elif (
            not allow_missing_product_qa
            and product_context is not None
            and self.qa_completion_result is not None
        ):
            raise ValueError("retained product QA context requires a complete local projection")
        final_fusion_context = self.final_fusion_context
        if final_fusion_context is not None:
            if (
                self.candidate_reduction_result is None
                or len(self.action_evidence_executions)
                != len(self.candidate_reduction_result.candidates)
                or self.provisional_fusion_result is None
                or not self.provisional_fusion_result.actions
                or len(self.boundary_refinement_executions)
                != len(self.provisional_fusion_result.actions)
                or tuple(item.action for item in self.boundary_refinement_executions)
                != self.provisional_fusion_result.actions
                or any(
                    item.result.outcome is not BoundaryRefinementOutcome.REFINED
                    for item in self.boundary_refinement_executions
                )
                or self.qa_completion_result is None
                or self.event_proposal_result is None
            ):
                raise ValueError(
                    "final fusion context requires the complete refined action closure"
                )
            expected_final_context = CanonicalFinalFusionContext.from_boundary_results(
                results=tuple(item.result for item in self.boundary_refinement_executions),
                recording_identity=self.recording_identity,
                policy_version=final_fusion_context.policy_version,
            )
            if final_fusion_context != expected_final_context:
                raise ValueError("retained final fusion context differs from boundary results")
        elif self.input_plan is not None:
            raise ValueError("final fusion input plan requires its retained action context")

        if self.input_plan is not None:
            assert final_fusion_context is not None
            assert self.qa_completion_result is not None
            assert self.event_proposal_result is not None
            assert self.candidate_reduction_result is not None
            assert self.provisional_fusion_result is not None
            expected_context = final_fusion_context.model_dump(mode="json")
            evidence_digests = [
                item.evidence_result.semantic_sha256 for item in self.action_evidence_executions
            ]
            if any(
                item.terminal.input_config.get("qa_completion_semantic_sha256")
                != self.qa_completion_result.semantic_sha256
                or item.terminal.input_config.get("event_proposal_semantic_sha256")
                != self.event_proposal_result.semantic_sha256
                or item.terminal.input_config.get("candidate_reduction_semantic_sha256")
                != self.candidate_reduction_result.semantic_sha256
                or item.terminal.input_config.get("action_evidence_result_semantic_sha256s")
                != evidence_digests
                or item.terminal.input_config.get("provisional_fusion_semantic_sha256")
                != self.provisional_fusion_result.semantic_sha256
                or item.terminal.input_config.get("final_fusion_policy_version")
                != final_fusion_context.policy_version
                or item.terminal.input_config.get("final_fusion_context_semantic_sha256")
                != final_fusion_context.semantic_sha256
                or item.terminal.input_config.get("final_fusion_context") != expected_context
                for item in self.part_results
            ):
                raise ValueError("final fusion calls do not bind the refined action closure")
        if self.input_plan is not None and self.package_set is not None:
            plan_ids = tuple(item.package_id for item in self.input_plan.subject.packages)
            if plan_ids != self.materialized_package_ids:
                raise ValueError("run input plan does not match materialized packages")
        if self.reference_catalog is not None and (
            self.input_plan is None
            or self.reference_catalog.input_plan_id != self.input_plan.input_plan_id
            or self.reference_catalog.input_plan_semantic_sha256 != self.input_plan.semantic_sha256
        ):
            raise ValueError("run reference catalog does not match its input plan")
        if self.barrier_reduction is not None and (
            self.input_plan is None
            or self.barrier_reduction.input_plan_semantic_sha256 != self.input_plan.semantic_sha256
        ):
            raise ValueError("run reduction does not match its input plan")
        if self.part_results:
            if self.input_plan is None:
                raise ValueError("part results require an input plan")
            if tuple(item.part_ordinal for item in self.part_results) != tuple(
                range(len(self.part_results))
            ):
                raise ValueError("run part results must be an ordered prefix")
            for item in self.part_results:
                planned = self.input_plan.call_plan.parts[item.part_ordinal]
                if (
                    item.part_count != len(self.input_plan.call_plan.parts)
                    or item.part_semantic_sha256 != planned.part_semantic_sha256
                    or item.terminal.mcap_id != self.mcap_id
                    or item.terminal.input_config.get("canonical_execution_policy_sha256")
                    != self.execution_policy_sha256
                ):
                    raise ValueError("run part result differs from its input plan or run binding")
        if self.attempt_count != sum(
            item.orchestration_attempt_count for item in self.part_results
        ):
            raise ValueError("run attempt count must equal all per-part attempts")

        compatibility = (
            self.terminal,
            self.selection,
            self.raw_response,
            self.parsed_claims,
            self.selected_output,
            self.enriched_output,
        )
        if self.part_results and self.part_results[0].part_count == 1:
            part = self.part_results[0]
            if compatibility != (
                part.terminal,
                part.selection,
                part.raw_response,
                part.parsed_claims,
                part.selected_output,
                part.enriched_output,
            ):
                raise ValueError("single-part compatibility fields differ from part truth")
        elif any(item is not None for item in compatibility):
            raise ValueError("multi-part runs cannot expose ambiguous singular lineage")

        enriched_outputs = tuple(
            item.enriched_output for item in self.part_results if item.enriched_output is not None
        )
        enriched_refs = tuple(
            sorted(
                (PlatformEnrichedOutputReference.from_output(item) for item in enriched_outputs),
                key=lambda item: item.enrichment_logical_key,
            )
        )
        if self.fusion_reduction is not None and (
            self.barrier_reduction is None
            or self.input_plan is None
            or self.fusion_reduction.input_plan_semantic_sha256 != self.input_plan.semantic_sha256
            or self.fusion_reduction.barrier_reduction_id != self.barrier_reduction.reduction_id
            or self.fusion_reduction.barrier_reduction_semantic_sha256
            != self.barrier_reduction.reduction_semantic_sha256
            or tuple(
                sorted(
                    (item.enrichment for item in self.fusion_reduction.parts),
                    key=lambda item: item.enrichment_logical_key,
                )
            )
            != enriched_refs
        ):
            raise ValueError("run fusion reduction lineage is inconsistent")
        if self.fusion_reduction is not None:
            for source, result in zip(
                self.fusion_reduction.parts,
                self.part_results,
                strict=True,
            ):
                enriched = result.enriched_output
                selected = result.selected_output
                if (
                    enriched is None
                    or selected is None
                    or source.part_semantic_sha256 != result.part_semantic_sha256
                    or source.completion_id != result.completion.completion_id
                    or source.inference_id != result.terminal.inference_id
                    or source.selected_attempt_output_sha256 != selected.output_sha256
                    or source.enrichment != PlatformEnrichedOutputReference.from_output(enriched)
                    or source.abstained != enriched.abstained
                ):
                    raise ValueError("run fusion part sources differ from per-part truth")
        if self.output_decision is not None:
            if (
                self.output_decision.recording_identity != self.recording_identity
                or self.fusion_reduction is None
                or self.output_decision.source_enrichments != enriched_refs
                or self.output_decision.fusion_reduction_semantic_sha256
                != self.fusion_reduction.semantic_sha256
                or self.output_decision.fusion_reduction_logical_key
                != self.fusion_reduction.reduction_logical_key
            ):
                raise ValueError("run output decision lineage is inconsistent")
            if final_fusion_context is None:
                raise ValueError("output decision requires its final fusion context")
            validate_final_fusion_reduction(
                context=final_fusion_context,
                fusion_reduction=self.fusion_reduction,
            )
            if (
                self.output_decision.evidence_class is not AdmissionEvidenceClass.LOCAL_CONFORMANCE
                or self.output_decision.production_eligible
            ):
                raise ValueError("offline conformance decisions require LOCAL_CONFORMANCE evidence")
        if any(item.recording_identity != self.recording_identity for item in self.hypotheses):
            raise ValueError("run hypotheses cross recording scope")
        if self.identity_result is not None:
            raise ValueError(
                "offline conformance result cannot carry authoritative identity assignments"
            )
        if self.hypotheses:
            if (
                self.output_decision is None
                or self.output_decision.production_output_admission is None
            ):
                raise ValueError("run hypotheses require an admitted output decision proof")
            output_proof = self.output_decision.production_output_admission
            primary_proof = self.hypotheses[0].production_admission
            if any(
                item.source_enrichments != self.output_decision.source_enrichments
                or item.production_admission != primary_proof
                or item.production_output_admission != output_proof
                or item.production_admission.evidence_class
                is not AdmissionEvidenceClass.LOCAL_CONFORMANCE
                or item.production_admission.production_eligible
                or item.production_output_admission.evidence_class
                is not AdmissionEvidenceClass.LOCAL_CONFORMANCE
                or item.production_output_admission.production_eligible
                for item in self.hypotheses
            ):
                raise ValueError(
                    "offline conformance hypotheses require one exact local proof lineage"
                )
            admitted_facts = tuple(
                sorted(
                    (
                        ProductionAdmittedHypothesisFact(
                            fusion_output_ordinal=item.fusion_output_ordinal,
                            effective_interval=item.effective_interval,
                            semantic_fingerprint_sha256=item.semantic_fingerprint_sha256,
                            fusion_logical_key=item.fusion_logical_key,
                        )
                        for item in self.hypotheses
                    ),
                    key=lambda item: (
                        item.effective_interval.start_ns,
                        item.effective_interval.end_ns,
                        item.fusion_logical_key,
                        item.fusion_output_ordinal,
                        item.semantic_fingerprint_sha256,
                    ),
                )
            )
            if output_proof.admitted_hypothesis_facts != admitted_facts:
                raise ValueError("output admission proof must exactly cover the run hypotheses")

        completed = self.status in {
            CanonicalOfflineRunStatus.SUCCEEDED,
            CanonicalOfflineRunStatus.NO_EVENTS,
            CanonicalOfflineRunStatus.ABSTAINED,
        }
        if completed and (
            self.qa_completion_result is None
            or self.qa_completion_result.status is not QACompletionStatus.QA_COMPLETE
        ):
            raise ValueError("completed run requires a QA_COMPLETE prerequisite")
        if not allow_missing_product_qa and completed and (
            self.product_qa_context is None
            or self.product_qa_result is None
        ):
            raise ValueError(
                "completed run requires its complete 21-class product QA projection"
            )
        if self.input_plan is not None and (
            self.qa_completion_result is None
            or self.qa_completion_result.status is not QACompletionStatus.QA_COMPLETE
        ):
            raise ValueError("downstream inference requires a QA_COMPLETE prerequisite")
        expected_memberships = _canonical_result_membership_lineage(self)
        actual_membership_lineage = tuple(
            (item.node_type, item.node_logical_key, item.role) for item in self.run_memberships
        )
        expected_membership_lineage = tuple(
            (node.node_type, node.node_logical_key, role) for node, role in expected_memberships
        )
        if (
            actual_membership_lineage
            != expected_membership_lineage[: len(actual_membership_lineage)]
        ):
            raise ValueError("run memberships are not the exact ordered lineage prefix")
        for membership, (node, role) in zip(
            self.run_memberships,
            expected_memberships[: len(self.run_memberships)],
            strict=True,
        ):
            if (
                membership.first_work_item_id
                != canonical_first_work_item_id(
                    run_id=self.run_id,
                    node=node,
                    role=role,
                )
                or membership.attached_at != self.processing_run.started_at
                or membership.disposition
                not in {RunNodeDisposition.CREATED, RunNodeDisposition.REUSED}
            ):
                raise ValueError("run membership facts do not match the canonical attachment")
        if completed and not self.run_memberships:
            raise ValueError("completed run requires its complete nonempty membership lineage")
        if self.status is CanonicalOfflineRunStatus.RUN_MEMBERSHIP_FAILED:
            if self.identity_result is not None:
                raise ValueError("membership-failed run cannot carry a published identity result")
            if len(self.run_memberships) >= len(expected_memberships):
                raise ValueError(
                    "membership-failed run requires a strict unpublished lineage suffix"
                )
        elif len(self.run_memberships) != len(expected_memberships):
            raise ValueError("run requires its complete retained membership lineage")
        if completed and self.error is not None:
            raise ValueError("completed run cannot contain an error")
        if not completed and self.error is None:
            raise ValueError("non-completed run requires a structured error")
        if self.status is CanonicalOfflineRunStatus.SUCCEEDED:
            if (
                self.input_plan is None
                or len(self.part_results) != len(self.input_plan.call_plan.parts)
                or any(
                    item.status is not CanonicalOfflinePartStatus.ENRICHED
                    for item in self.part_results
                )
                or self.barrier_reduction is None
                or self.fusion_reduction is None
                or self.fusion_reduction.outcome != "CLAIMS"
                or self.output_decision is None
                or self.output_decision.decision != "ADMITTED"
                or not self.hypotheses
            ):
                raise ValueError("successful run is missing admitted hypothesis lineage")
        elif self.status is CanonicalOfflineRunStatus.ABSTAINED:
            if (
                self.input_plan is None
                or len(self.part_results) != len(self.input_plan.call_plan.parts)
                or any(
                    item.status is not CanonicalOfflinePartStatus.ENRICHED
                    or item.enriched_output is None
                    or not item.enriched_output.abstained
                    for item in self.part_results
                )
                or self.barrier_reduction is None
                or self.fusion_reduction is None
                or self.fusion_reduction.outcome != "ALL_PARTS_ABSTAINED"
                or self.output_decision is None
                or self.output_decision.decision != "ABSTAINED"
                or self.hypotheses
                or self.identity_result is not None
            ):
                raise ValueError("ABSTAINED run has an inconsistent terminal shape")
        elif self.status is CanonicalOfflineRunStatus.NO_EVENTS:
            proposal_no_events = (
                self.event_proposal_result is not None
                and self.event_proposal_result.outcome is EventProposalOutcome.NO_EVENTS
                and self.candidate_reduction_result is not None
                and self.candidate_reduction_result.no_events
                and not self.action_evidence_executions
                and self.provisional_fusion_result is None
                and not self.boundary_refinement_executions
                and self.final_fusion_context is None
                and self.input_plan is None
                and not self.part_results
                and self.barrier_reduction is None
                and self.fusion_reduction is None
                and self.output_decision is None
                and not self.hypotheses
                and self.identity_result is None
            )
            action_no_events = (
                self.candidate_reduction_result is not None
                and bool(self.candidate_reduction_result.candidates)
                and len(self.action_evidence_executions)
                == len(self.candidate_reduction_result.candidates)
                and all(
                    item.evidence_result.outcome is ActionEvidenceOutcome.NO_ACTION
                    for item in self.action_evidence_executions
                )
                and self.provisional_fusion_result is not None
                and self.provisional_fusion_result.no_actions
                and not self.boundary_refinement_executions
                and self.final_fusion_context is None
                and self.input_plan is None
                and not self.part_results
                and self.barrier_reduction is None
                and self.fusion_reduction is None
                and self.output_decision is None
                and not self.hypotheses
                and self.identity_result is None
            )
            fusion_no_events = (
                self.provisional_fusion_result is not None
                and bool(self.provisional_fusion_result.actions)
                and len(self.boundary_refinement_executions)
                == len(self.provisional_fusion_result.actions)
                and tuple(item.action for item in self.boundary_refinement_executions)
                == self.provisional_fusion_result.actions
                and all(
                    item.result.outcome is BoundaryRefinementOutcome.REFINED
                    for item in self.boundary_refinement_executions
                )
                and self.final_fusion_context is not None
                and self.input_plan is not None
                and len(self.part_results) == len(self.input_plan.call_plan.parts)
                and all(
                    item.status is CanonicalOfflinePartStatus.ENRICHED for item in self.part_results
                )
                and self.barrier_reduction is not None
                and self.fusion_reduction is not None
                and self.fusion_reduction.outcome == "NO_SURVIVING_EVENTS"
                and self.output_decision is not None
                and self.output_decision.decision == "NO_EVENTS"
                and not self.hypotheses
                and self.identity_result is None
            )
            if not (proposal_no_events or action_no_events or fusion_no_events):
                raise ValueError("NO_EVENTS run has an inconsistent terminal shape")
        elif self.status is CanonicalOfflineRunStatus.INCOMPLETE:
            qa_stop = (
                self.qa_completion_result is not None
                and self.qa_completion_result.status
                in {QACompletionStatus.DENSE_REQUIRED, QACompletionStatus.QA_INCOMPLETE}
                and not self.action_evidence_executions
                and self.provisional_fusion_result is None
                and not self.boundary_refinement_executions
                and self.final_fusion_context is None
                and self.input_plan is None
                and self.reference_catalog is None
                and not self.part_results
                and self.barrier_reduction is None
                and self.fusion_reduction is None
                and self.output_decision is None
                and not self.hypotheses
                and self.identity_result is None
            )
            action_stop = (
                self.qa_completion_result is not None
                and self.qa_completion_result.status is QACompletionStatus.QA_COMPLETE
                and bool(self.action_evidence_executions)
                and self.provisional_fusion_result is None
                and not self.boundary_refinement_executions
                and self.final_fusion_context is None
                and self.input_plan is None
                and not self.part_results
                and self.barrier_reduction is None
                and self.fusion_reduction is None
                and self.output_decision is None
                and not self.hypotheses
                and self.identity_result is None
            )
            boundary_indeterminate_stop = (
                self.error is not None
                and self.error.code == "BOUNDARY_REFINEMENT_INDETERMINATE"
                and self.provisional_fusion_result is not None
                and bool(self.provisional_fusion_result.actions)
                and tuple(item.action for item in self.boundary_refinement_executions)
                == self.provisional_fusion_result.actions
                and any(
                    item.result.outcome is BoundaryRefinementOutcome.INDETERMINATE
                    for item in self.boundary_refinement_executions
                )
                and self.final_fusion_context is None
                and self.input_plan is None
                and not self.part_results
                and self.barrier_reduction is None
                and self.fusion_reduction is None
                and self.output_decision is None
                and not self.hypotheses
                and self.identity_result is None
            )

            inference_stop = (
                self.input_plan is not None
                and len(self.part_results) == len(self.input_plan.call_plan.parts)
                and self.barrier_reduction is None
                and self.fusion_reduction is None
                and self.output_decision is None
                and not self.hypotheses
                and self.identity_result is None
            )
            if not any(
                (
                    qa_stop,
                    action_stop,
                    boundary_indeterminate_stop,
                    inference_stop,
                )
            ):
                raise ValueError("INCOMPLETE run has an inconsistent stopped-stage shape")
        return self


def canonical_dense_qa_execution_membership_lineage(
    execution: CanonicalDenseQAExecution,
) -> tuple[tuple[LogicalNode, RunNodeRole], ...]:
    """Return the sole ordered membership chain for one successful dense unit."""

    checked = CanonicalDenseQAExecution.model_validate(
        execution.model_dump(mode="python"),
        strict=True,
    )
    entries: list[tuple[LogicalNode, RunNodeRole]] = [
        (canonical_root_window_logical_node(checked.window), "DENSE_QA_WINDOW"),
        (
            canonical_package_set_logical_node(checked.package_set),
            "DENSE_QA_PACKAGE_SET",
        ),
        (canonical_input_plan_logical_node(checked.input_plan), "DENSE_QA_INPUT_PLAN"),
    ]
    entries.extend(
        (
            canonical_call_part_logical_node(checked.input_plan, part),
            "DENSE_QA_CALL_PART",
        )
        for part in checked.input_plan.call_plan.parts
    )
    entries.append(
        (
            canonical_call_barrier_logical_node(checked.input_plan),
            "DENSE_QA_CALL_BARRIER",
        )
    )
    for part_result in checked.part_results:
        assert part_result.selection is not None
        assert part_result.parsed_claims is not None
        assert part_result.selected_output is not None
        assert part_result.enriched_output is not None
        entries.extend(
            (
                (
                    canonical_selection_logical_node(part_result.selection),
                    "DENSE_QA_ATTEMPT_SELECTION",
                ),
                (
                    canonical_parsed_claim_logical_node(part_result.parsed_claims),
                    "DENSE_QA_PARSED_CLAIM",
                ),
                (
                    canonical_selected_output_logical_node(part_result.selected_output),
                    "DENSE_QA_SELECTED_OUTPUT",
                ),
                (
                    canonical_enrichment_logical_node(part_result.enriched_output),
                    "DENSE_QA_ENRICHED_OUTPUT",
                ),
            )
        )
    entries.append(
        (
            canonical_call_reduction_logical_node(checked.barrier_reduction),
            "DENSE_QA_CALL_REDUCTION",
        )
    )
    return tuple(entries)


def canonical_action_evidence_execution_membership_lineage(
    execution: CanonicalActionEvidenceExecution,
) -> tuple[tuple[LogicalNode, RunNodeRole], ...]:
    """Return the ordered reusable node chain for one candidate evidence execution."""

    checked = CanonicalActionEvidenceExecution.model_validate(
        execution.model_dump(mode="python"), strict=True
    )
    entries: list[tuple[LogicalNode, RunNodeRole]] = [
        (
            canonical_candidate_dense_window_logical_node(checked.window),
            "ACTION_EVIDENCE_WINDOW",
        ),
        (canonical_package_set_logical_node(checked.package_set), "ACTION_EVIDENCE_PACKAGE_SET"),
        (canonical_input_plan_logical_node(checked.input_plan), "ACTION_EVIDENCE_INPUT_PLAN"),
    ]
    entries.extend(
        (
            canonical_call_part_logical_node(checked.input_plan, part),
            "ACTION_EVIDENCE_CALL_PART",
        )
        for part in checked.input_plan.call_plan.parts
    )
    entries.append(
        (
            canonical_call_barrier_logical_node(checked.input_plan),
            "ACTION_EVIDENCE_CALL_BARRIER",
        )
    )
    for part_result in checked.part_results:
        assert part_result.selection is not None
        assert part_result.parsed_claims is not None
        assert part_result.selected_output is not None
        assert part_result.enriched_output is not None
        entries.extend(
            (
                (
                    canonical_selection_logical_node(part_result.selection),
                    "ACTION_EVIDENCE_ATTEMPT_SELECTION",
                ),
                (
                    canonical_parsed_claim_logical_node(part_result.parsed_claims),
                    "ACTION_EVIDENCE_PARSED_CLAIM",
                ),
                (
                    canonical_selected_output_logical_node(part_result.selected_output),
                    "ACTION_EVIDENCE_SELECTED_OUTPUT",
                ),
                (
                    canonical_enrichment_logical_node(part_result.enriched_output),
                    "ACTION_EVIDENCE_ENRICHED_OUTPUT",
                ),
            )
        )
    entries.extend(
        (
            (
                canonical_call_reduction_logical_node(checked.barrier_reduction),
                "ACTION_EVIDENCE_CALL_REDUCTION",
            ),
            (
                canonical_action_evidence_result_logical_node(checked.evidence_result),
                "ACTION_EVIDENCE_RESULT",
            ),
        )
    )
    return tuple(entries)


def canonical_boundary_refinement_pass_membership_lineage(
    execution: CanonicalBoundaryRefinementPassExecution,
) -> tuple[tuple[LogicalNode, RunNodeRole], ...]:
    """Return one role pass's exact reusable trust and reduction chain."""

    checked = CanonicalBoundaryRefinementPassExecution.model_validate(
        execution.model_dump(mode="python"), strict=True
    )
    role_prefix = f"BOUNDARY_{checked.role_result.role.value}"
    entries: list[tuple[LogicalNode, RunNodeRole]] = [
        (
            canonical_boundary_window_logical_node(checked.window),
            f"{role_prefix}_WINDOW",
        ),
        (
            canonical_package_set_logical_node(checked.package_set),
            f"{role_prefix}_PACKAGE_SET",
        ),
        (
            canonical_input_plan_logical_node(checked.input_plan),
            f"{role_prefix}_INPUT_PLAN",
        ),
    ]
    entries.extend(
        (
            canonical_call_part_logical_node(checked.input_plan, part),
            f"{role_prefix}_CALL_PART",
        )
        for part in checked.input_plan.call_plan.parts
    )
    entries.append(
        (
            canonical_call_barrier_logical_node(checked.input_plan),
            f"{role_prefix}_CALL_BARRIER",
        )
    )
    for part_result in checked.part_results:
        assert part_result.selection is not None
        assert part_result.parsed_claims is not None
        assert part_result.selected_output is not None
        assert part_result.enriched_output is not None
        entries.extend(
            (
                (
                    canonical_selection_logical_node(part_result.selection),
                    f"{role_prefix}_ATTEMPT_SELECTION",
                ),
                (
                    canonical_parsed_claim_logical_node(part_result.parsed_claims),
                    f"{role_prefix}_PARSED_CLAIM",
                ),
                (
                    canonical_selected_output_logical_node(part_result.selected_output),
                    f"{role_prefix}_SELECTED_OUTPUT",
                ),
                (
                    canonical_enrichment_logical_node(part_result.enriched_output),
                    f"{role_prefix}_ENRICHED_OUTPUT",
                ),
            )
        )
    entries.extend(
        (
            (
                canonical_call_reduction_logical_node(checked.barrier_reduction),
                f"{role_prefix}_CALL_REDUCTION",
            ),
            (
                canonical_boundary_role_result_logical_node(checked.role_result),
                f"{role_prefix}_RESULT",
            ),
        )
    )
    return tuple(entries)


def canonical_boundary_refinement_execution_membership_lineage(
    execution: CanonicalBoundaryRefinementExecution,
) -> tuple[tuple[LogicalNode, RunNodeRole], ...]:
    """Return both role chains and their action-level deterministic result."""

    checked = CanonicalBoundaryRefinementExecution.model_validate(
        execution.model_dump(mode="python"), strict=True
    )
    return (
        *canonical_boundary_refinement_pass_membership_lineage(checked.onset),
        *canonical_boundary_refinement_pass_membership_lineage(checked.offset),
        (
            canonical_boundary_refinement_result_logical_node(checked.result),
            "BOUNDARY_REFINEMENT_RESULT",
        ),
    )


def _canonical_result_membership_lineage(
    result: CanonicalOfflineRunResult,
) -> tuple[tuple[LogicalNode, RunNodeRole], ...]:
    """Rebuild the canonical attachment order solely from retained result lineage."""

    entries: list[tuple[LogicalNode, RunNodeRole]] = []
    if result.window is not None:
        entries.append((canonical_root_window_logical_node(result.window), "ROOT_WINDOW"))
    if result.package_set is not None:
        entries.append((canonical_package_set_logical_node(result.package_set), "PACKAGE_SET"))
    if result.coarse_qa_result is not None:
        entries.append(
            (
                canonical_coarse_qa_logical_node(result.coarse_qa_result),
                "COARSE_QA",
            )
        )
    for execution in result.dense_qa_executions:
        entries.extend(canonical_dense_qa_execution_membership_lineage(execution))
    if (
        result.qa_completion_result is not None
        and result.qa_completion_result.dense_result is not None
    ):
        entries.append(
            (
                canonical_dense_qa_result_logical_node(result.qa_completion_result.dense_result),
                "DENSE_QA_RESULT",
            )
        )
    if result.qa_completion_result is not None:
        entries.append(
            (
                canonical_qa_completion_logical_node(result.qa_completion_result),
                "QA_COMPLETION",
            )
        )
    if result.event_proposal_result is not None:
        entries.append(
            (
                canonical_event_proposal_result_logical_node(result.event_proposal_result),
                "EVENT_PROPOSAL_RESULT",
            )
        )
    if result.candidate_reduction_result is not None:
        entries.append(
            (
                canonical_candidate_reduction_logical_node(result.candidate_reduction_result),
                "CANDIDATE_REDUCTION",
            )
        )
        entries.extend(
            (canonical_candidate_event_logical_node(item), "CANDIDATE_EVENT")
            for item in result.candidate_reduction_result.candidates
        )
    for action_execution in result.action_evidence_executions:
        entries.extend(canonical_action_evidence_execution_membership_lineage(action_execution))
    if result.provisional_fusion_result is not None:
        entries.append(
            (
                canonical_provisional_fusion_result_logical_node(result.provisional_fusion_result),
                "PROVISIONAL_ACTION_FUSION_RESULT",
            )
        )
        entries.extend(
            (
                canonical_provisional_physical_action_logical_node(action),
                "PROVISIONAL_PHYSICAL_ACTION",
            )
            for action in result.provisional_fusion_result.actions
        )
    for boundary_execution in result.boundary_refinement_executions:
        entries.extend(
            canonical_boundary_refinement_execution_membership_lineage(boundary_execution)
        )
    if result.input_plan is not None:
        entries.append((canonical_input_plan_logical_node(result.input_plan), "INPUT_PLAN"))
        entries.extend(
            (canonical_call_part_logical_node(result.input_plan, part), "CALL_PART")
            for part in result.input_plan.call_plan.parts
        )
        entries.append((canonical_call_barrier_logical_node(result.input_plan), "CALL_BARRIER"))
    for part_result in result.part_results:
        if part_result.selection is not None:
            entries.append(
                (canonical_selection_logical_node(part_result.selection), "ATTEMPT_SELECTION")
            )
        if part_result.parsed_claims is not None:
            entries.append(
                (canonical_parsed_claim_logical_node(part_result.parsed_claims), "PARSED_CLAIM")
            )
        if part_result.selected_output is not None:
            entries.append(
                (
                    canonical_selected_output_logical_node(part_result.selected_output),
                    "SELECTED_OUTPUT",
                )
            )
        if part_result.enriched_output is not None:
            entries.append(
                (
                    canonical_enrichment_logical_node(part_result.enriched_output),
                    "ENRICHED_OUTPUT",
                )
            )
    if result.barrier_reduction is not None:
        entries.append(
            (
                canonical_call_reduction_logical_node(result.barrier_reduction),
                "CALL_REDUCTION",
            )
        )
    if result.fusion_reduction is not None:
        entries.append(
            (
                canonical_fusion_reduction_logical_node(result.fusion_reduction),
                "FUSION_REDUCTION",
            )
        )
    if result.output_decision is not None:
        entries.append(
            (
                canonical_output_decision_logical_node(result.output_decision),
                "OUTPUT_DECISION",
            )
        )
    entries.extend(
        (canonical_event_hypothesis_logical_node(item), "EVENT_HYPOTHESIS")
        for item in result.hypotheses
    )
    unique_entries: list[tuple[LogicalNode, RunNodeRole]] = []
    seen: set[tuple[str, str, str]] = set()
    for node, role in entries:
        identity = (node.node_type, node.node_logical_key, role)
        if identity not in seen:
            seen.add(identity)
            unique_entries.append((node, role))
    return tuple(unique_entries)


__all__ = [
    "CanonicalActionEvidenceExecution",
    "CanonicalBoundaryRefinementExecution",
    "CanonicalBoundaryRefinementPassExecution",
    "CanonicalDenseQAExecution",
    "CanonicalOfflineRunResult",
    "canonical_action_evidence_execution_membership_lineage",
    "canonical_boundary_refinement_execution_membership_lineage",
    "canonical_boundary_refinement_pass_membership_lineage",
    "canonical_dense_qa_execution_membership_lineage",
]
