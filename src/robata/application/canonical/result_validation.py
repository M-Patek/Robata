"""Terminal result model and retained-lineage validation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from robata.application.canonical.logical_nodes import (
    canonical_call_barrier_logical_node,
    canonical_call_part_logical_node,
    canonical_call_reduction_logical_node,
    canonical_enrichment_logical_node,
    canonical_event_hypothesis_logical_node,
    canonical_fusion_reduction_logical_node,
    canonical_input_plan_logical_node,
    canonical_output_decision_logical_node,
    canonical_package_set_logical_node,
    canonical_parsed_claim_logical_node,
    canonical_root_window_logical_node,
    canonical_selected_output_logical_node,
    canonical_selection_logical_node,
)
from robata.application.canonical.models import (
    CANONICAL_OFFLINE_PIPELINE_VERSION,
    CanonicalOfflineError,
    CanonicalOfflinePartResult,
    CanonicalOfflinePartStatus,
    CanonicalOfflineRunStatus,
    CanonicalRootWindow,
    NonNegativeInt,
)
from robata.application.canonical.output_admission import CanonicalOutputAdmissionDecision
from robata.application.canonical.reduction import CanonicalFusionReduction
from robata.application.canonical.runner_support import _rfc3339_datetime
from robata.application.canonical_run_membership import (
    CanonicalProcessingRunPrimaryStatus,
    CanonicalProcessingRunRecord,
    canonical_first_work_item_id,
)
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.logical_nodes import (
    LogicalNode,
    OpaqueUuid,
    ProcessingRunNodeMembership,
    RunNodeDisposition,
    RunNodeRole,
)
from robata.contracts.temporal import TemporalPackageSet
from robata.event_pipeline.identity_registry import (
    AdmissionEvidenceClass,
    EventIdentityBatchResult,
    PlatformEnrichedEventHypothesis,
    PlatformEnrichedOutputReference,
    ProductionAdmittedHypothesisFact,
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
from robata.inference.models import InferenceAttemptSelection, ModelInference


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
    network_call_count: Literal[0]
    error: CanonicalOfflineError | None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
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
            if (
                self.input_plan is None
                or len(self.part_results) != len(self.input_plan.call_plan.parts)
                or any(
                    item.status is not CanonicalOfflinePartStatus.ENRICHED
                    for item in self.part_results
                )
                or self.barrier_reduction is None
                or self.fusion_reduction is None
                or self.fusion_reduction.outcome != "NO_SURVIVING_EVENTS"
                or self.output_decision is None
                or self.output_decision.decision != "NO_EVENTS"
                or self.hypotheses
                or self.identity_result is not None
            ):
                raise ValueError("NO_EVENTS run has an inconsistent terminal shape")
        elif self.status is CanonicalOfflineRunStatus.INCOMPLETE and (
            self.input_plan is None
            or len(self.part_results) != len(self.input_plan.call_plan.parts)
            or self.barrier_reduction is not None
            or self.fusion_reduction is not None
            or self.output_decision is not None
            or self.hypotheses
            or self.identity_result is not None
        ):
            raise ValueError("INCOMPLETE run cannot publish reduced output")
        return self


def _canonical_result_membership_lineage(
    result: CanonicalOfflineRunResult,
) -> tuple[tuple[LogicalNode, RunNodeRole], ...]:
    """Rebuild the canonical attachment order solely from retained result lineage."""

    entries: list[tuple[LogicalNode, RunNodeRole]] = []
    if result.window is not None:
        entries.append((canonical_root_window_logical_node(result.window), "ROOT_WINDOW"))
    if result.package_set is not None:
        entries.append((canonical_package_set_logical_node(result.package_set), "PACKAGE_SET"))
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


__all__ = ["CanonicalOfflineRunResult"]
