"""Prepare one local-conformance authoritative primary completion command.

The models in this module are the application boundary between the pure
canonical runner and the aggregate completion repository.  They intentionally
cover only successful event-producing runs and explicit no-event runs.  The
repository is responsible for applying the command atomically; this module
performs no writes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import model_validator

from robata.application.canonical.action_event_revision import (
    PreparedInitialActionEventRevisionBatch,
)
from robata.application.canonical.models import (
    CanonicalOfflinePartResult,
    CanonicalOfflineRunStatus,
    CanonicalRootWindow,
)
from robata.application.canonical.output_admission import (
    CanonicalFinalFusionContext,
    CanonicalOutputAdmissionDecision,
)
from robata.application.canonical.reduction import CanonicalFusionReduction
from robata.application.canonical.result_validation import (
    CanonicalActionEvidenceExecution,
    CanonicalBoundaryRefinementExecution,
    CanonicalDenseQAExecution,
    CanonicalOfflineRunResult,
)
from robata.application.canonical_run_membership import (
    CanonicalProcessingRunContext,
    CanonicalProcessingRunRecord,
)
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import (
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)
from robata.contracts.logical_nodes import OpaqueUuid, ProcessingRunNodeMembership
from robata.contracts.primary_completion import (
    PRIMARY_COMPLETION_RECORD_SCHEMA_ID,
    PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION,
    DetailedResultArtifactReference,
    PrimaryCompletionOutcome,
    PrimaryCompletionRecord,
    PrimaryCompletionTerminalStage,
    create_primary_completion_record,
    validate_registered_primary_completion_record,
)
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry, default_schema_registry
from robata.contracts.temporal import TemporalPackageSet
from robata.event_pipeline.candidate import CandidateReductionResult
from robata.event_pipeline.identity_registry import (
    EventIdentityBatchResult,
    EventIdentityOutboxRecord,
    EventRegistrySnapshot,
    PlatformEnrichedEventHypothesis,
    PreparedEventIdentityBatch,
)
from robata.event_pipeline.proposer import EventProposalResult
from robata.event_pipeline.provisional_fusion import ProvisionalFusionResult
from robata.inference.call_barrier import InferenceCallReduction
from robata.inference.enrichment import ProviderReferenceCatalog
from robata.inference.input_plan import InferenceInputPlan
from robata.qa_pipeline.coarse import CoarseQAResult
from robata.qa_pipeline.completion import QACompletionResult

CANONICAL_PRIMARY_COMPLETION_DETAIL_SCHEMA_ID: Final = (
    "https://schemas.robata.dev/canonical-primary-completion-detail"
)
CANONICAL_PRIMARY_COMPLETION_DETAIL_SCHEMA_VERSION: Final = "4.0.0"
CANONICAL_PRIMARY_COMPLETION_DETAIL_WIRE_VERSION: Final = "4.0"
CANONICAL_PRIMARY_COMPLETION_DETAIL_PROJECTION_VERSION: Final = (
    "canonical-primary-completion-detail-semantic-v4"
)
CANONICAL_PRIMARY_COMPLETION_COMMAND_PROJECTION_VERSION: Final = (
    "canonical-primary-completion-command-v1"
)
CANONICAL_PRIMARY_COLLECTION_ROOT_PROJECTION_VERSION: Final = "canonical-primary-collection-root-v1"


class PrimaryCompletionErrorCode(StrEnum):
    """Stable error categories exposed by the local aggregate boundary."""

    INVALID_COMMAND = "INVALID_COMMAND"
    RUN_CONFLICT = "RUN_CONFLICT"
    STALE_RUN = "STALE_RUN"
    STALE_IDENTITY = "STALE_IDENTITY"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"


class PrimaryCompletionError(RuntimeError):
    """A command cannot be prepared, committed, or recovered truthfully."""

    def __init__(self, code: PrimaryCompletionErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class CanonicalPrimaryCompletionDetail(StrictModel):
    """Schema-governed terminal semantic closure referenced by the compact completion."""

    schema_version: Literal["4.0"]
    schema_ref: SchemaRef
    semantic_projection_version: Literal["canonical-primary-completion-detail-semantic-v4"]
    semantic_sha256: Sha256Digest
    evidence_class: Literal["LOCAL_CONFORMANCE"]
    production_eligible: Literal[False]

    run_id: OpaqueUuid
    recording_identity: Sha256Digest
    mcap_id: OpaqueUuid
    execution_policy_sha256: Sha256Digest
    status: Literal["SUCCEEDED", "NO_EVENTS"]
    processing_run: CanonicalProcessingRunRecord
    run_memberships: tuple[ProcessingRunNodeMembership, ...]
    window: CanonicalRootWindow
    package_set: TemporalPackageSet
    coarse_qa_result: CoarseQAResult
    qa_completion_result: QACompletionResult
    dense_qa_executions: tuple[CanonicalDenseQAExecution, ...]
    event_proposal_result: EventProposalResult
    candidate_reduction_result: CandidateReductionResult
    action_evidence_executions: tuple[CanonicalActionEvidenceExecution, ...]
    provisional_fusion_result: ProvisionalFusionResult | None
    boundary_refinement_executions: tuple[CanonicalBoundaryRefinementExecution, ...]
    final_fusion_context: CanonicalFinalFusionContext | None
    input_plan: InferenceInputPlan | None
    reference_catalog: ProviderReferenceCatalog | None
    part_results: tuple[CanonicalOfflinePartResult, ...]
    barrier_reduction: InferenceCallReduction | None
    fusion_reduction: CanonicalFusionReduction | None
    output_decision: CanonicalOutputAdmissionDecision | None
    hypotheses: tuple[PlatformEnrichedEventHypothesis, ...]
    prepared_identities: PreparedEventIdentityBatch | None
    action_event_publications: PreparedInitialActionEventRevisionBatch

    @model_validator(mode="after")
    def validate_detail(self) -> Self:
        if (
            self.schema_ref.schema_id != CANONICAL_PRIMARY_COMPLETION_DETAIL_SCHEMA_ID
            or self.schema_ref.version != CANONICAL_PRIMARY_COMPLETION_DETAIL_SCHEMA_VERSION
        ):
            raise ValueError("schema_ref does not identify the detailed completion schema")
        if (
            self.processing_run.run_id != self.run_id
            or self.processing_run.recording_identity != self.recording_identity
            or self.processing_run.mcap_id != self.mcap_id
            or self.processing_run.config_sha256 != self.execution_policy_sha256
            or self.processing_run.primary_status.value != self.status
            or self.processing_run.pipeline_version != "canonical-offline-v5"
        ):
            raise ValueError("detailed completion does not match its processing run")

        compatibility: tuple[object | None, ...]
        if len(self.part_results) == 1:
            part = self.part_results[0]
            compatibility = (
                part.terminal,
                part.selection,
                part.raw_response,
                part.parsed_claims,
                part.selected_output,
                part.enriched_output,
            )
        else:
            compatibility = (None, None, None, None, None, None)
        CanonicalOfflineRunResult(
            schema_version="1.0",
            run_id=self.run_id,
            processing_run=self.processing_run,
            run_memberships=self.run_memberships,
            recording_identity=self.recording_identity,
            mcap_id=self.mcap_id,
            execution_policy_sha256=self.execution_policy_sha256,
            status=CanonicalOfflineRunStatus(self.status),
            window=self.window,
            materialized_package_ids=tuple(item.package_id for item in self.package_set.members),
            package_set=self.package_set,
            coarse_qa_result=self.coarse_qa_result,
            dense_qa_executions=self.dense_qa_executions,
            qa_completion_result=self.qa_completion_result,
            event_proposal_result=self.event_proposal_result,
            candidate_reduction_result=self.candidate_reduction_result,
            action_evidence_executions=self.action_evidence_executions,
            provisional_fusion_result=self.provisional_fusion_result,
            boundary_refinement_executions=self.boundary_refinement_executions,
            final_fusion_context=self.final_fusion_context,
            input_plan=self.input_plan,
            reference_catalog=self.reference_catalog,
            part_results=self.part_results,
            barrier_reduction=self.barrier_reduction,
            fusion_reduction=self.fusion_reduction,
            terminal=compatibility[0],
            selection=compatibility[1],
            raw_response=compatibility[2],
            parsed_claims=compatibility[3],
            selected_output=compatibility[4],
            enriched_output=compatibility[5],
            output_decision=self.output_decision,
            hypotheses=self.hypotheses,
            identity_result=None,
            attempt_count=sum(item.orchestration_attempt_count for item in self.part_results),
            adapter_infer_calls=0,
            error=None,
        )

        if self.action_event_publications.recording_identity != self.recording_identity or any(
            item.payload.production_eligible for item in self.action_event_publications.publications
        ):
            raise ValueError("local detailed completion cannot be production eligible")
        if self.status == "SUCCEEDED":
            prepared = self.prepared_identities
            publications = self.action_event_publications
            if (
                not self.hypotheses
                or prepared is None
                or publications.outcome != "PREPARED"
                or len(prepared.assignments) != len(self.hypotheses)
                or len(publications.publications) != len(self.hypotheses)
            ):
                raise ValueError("successful completion lacks identity or revision facts")
            assert prepared is not None
            hypothesis_bindings = tuple(
                (item.event_hypothesis_logical_key, item.semantic_sha256)
                for item in sorted(
                    self.hypotheses,
                    key=lambda item: (
                        item.effective_start_ns,
                        item.effective_end_ns,
                        item.event_hypothesis_logical_key,
                    ),
                )
            )
            assignment_bindings = tuple(
                (
                    item.event_hypothesis_logical_key,
                    item.event_hypothesis_semantic_sha256,
                )
                for item in prepared.assignments
            )
            if (
                prepared.recording_identity != self.recording_identity
                or prepared.ordered_hypothesis_logical_keys
                != tuple(item[0] for item in hypothesis_bindings)
                or assignment_bindings != hypothesis_bindings
                or publications.expected_generation != prepared.expected_generation
                or publications.expected_fence != prepared.expected_fence
                or tuple(item.assignment for item in publications.publications)
                != prepared.assignments
            ):
                raise ValueError(
                    "successful completion does not bind its hypotheses, identities, and revisions"
                )
        elif (
            self.hypotheses
            or self.prepared_identities is not None
            or self.action_event_publications.outcome != "NO_EVENTS"
            or self.action_event_publications.publications
        ):
            raise ValueError("no-event completion must carry explicit empty event facts")

        expected = semantic_sha256(canonical_primary_completion_detail_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("detailed completion semantic_sha256 is inconsistent")
        return self


class PrimaryCompletionCommand(StrictModel):
    """Exact idempotent command consumed by the aggregate repository."""

    schema_version: Literal["1.0"]
    semantic_projection_version: Literal["canonical-primary-completion-command-v1"]
    command_sha256: Sha256Digest
    detail: CanonicalPrimaryCompletionDetail
    completion: PrimaryCompletionRecord

    @model_validator(mode="after")
    def validate_command(self) -> Self:
        detail_bytes = canonical_json_bytes(self.detail)
        detail_digest = exact_bytes_sha256(detail_bytes)
        reference = self.completion.detailed_result
        if (
            self.completion.run_id != self.detail.run_id
            or self.completion.recording_identity != self.detail.recording_identity
            or self.completion.mcap_id != self.detail.mcap_id
            or reference.exact_bytes_sha256 != detail_digest
            or reference.byte_count != len(detail_bytes)
            or reference.artifact_id != _detailed_result_artifact_id(detail_digest)
            or reference.schema_ref != self.detail.schema_ref
        ):
            raise ValueError("completion record does not bind the exact detailed result")
        expected_completion = _primary_completion_record_from_detail(
            self.detail,
            schema_ref=self.completion.schema_ref,
            detailed_result=reference,
        )
        if self.completion != expected_completion:
            raise ValueError("compact completion does not match the detailed completion facts")
        expected = semantic_sha256(primary_completion_command_projection(self))
        if self.command_sha256 != expected:
            raise ValueError("primary completion command_sha256 is inconsistent")
        return self


class CommittedPrimaryCompletion(StrictModel):
    """Authoritative recovery view returned after the aggregate transaction."""

    schema_version: Literal["1.0"]
    command_sha256: Sha256Digest
    processing_run: CanonicalProcessingRunRecord
    completion: PrimaryCompletionRecord
    detail: CanonicalPrimaryCompletionDetail
    identity_result: EventIdentityBatchResult | None
    action_event_publications: PreparedInitialActionEventRevisionBatch
    outbox: tuple[EventIdentityOutboxRecord, ...]

    @model_validator(mode="after")
    def validate_committed(self) -> Self:
        if (
            self.processing_run.run_id != self.completion.run_id
            or self.processing_run.run_id != self.detail.run_id
            or self.processing_run.primary_status.value != self.detail.status
            or self.action_event_publications != self.detail.action_event_publications
        ):
            raise ValueError("committed completion facts disagree")
        if self.detail.status == "SUCCEEDED":
            if self.identity_result is None:
                raise ValueError("successful completion requires committed identity facts")
            if (
                self.identity_result.assignments != self.detail.prepared_identities.assignments  # type: ignore[union-attr]
                or self.identity_result.outbox != self.outbox
            ):
                raise ValueError("committed identity result differs from its preparation")
        elif self.identity_result is not None or self.outbox:
            raise ValueError("no-event completion cannot publish identity outbox facts")
        return self


class PrimaryCompletionCommitResult(StrictModel):
    """Commit result distinguishes first application from exact replay."""

    committed: CommittedPrimaryCompletion
    replayed: bool


class PrimaryCompletionRepository(Protocol):
    """Small aggregate boundary used by canonical composition and recovery."""

    def begin_run(self, context: CanonicalProcessingRunContext) -> CanonicalProcessingRunRecord: ...

    def snapshot(self, recording_identity: str) -> EventRegistrySnapshot: ...

    def get(self, run_id: str) -> CommittedPrimaryCompletion | None: ...

    def commit(self, command: PrimaryCompletionCommand) -> PrimaryCompletionCommitResult: ...

    def list_outbox(self, recording_identity: str) -> tuple[EventIdentityOutboxRecord, ...]: ...


def canonical_primary_completion_detail_projection(
    detail: CanonicalPrimaryCompletionDetail,
) -> dict[str, object]:
    """Return the local detailed-result semantic projection."""

    return detail.model_dump(
        mode="json",
        exclude={"schema_ref", "semantic_sha256"},
    )


def canonical_collection_digest_root(
    collection: str,
    ordered_item_digests: tuple[str, ...],
) -> Sha256Digest:
    """Bind an ordered collection, including an explicit empty collection."""

    if not collection:
        raise ValueError("collection must be nonempty")
    return semantic_sha256(
        {
            "semantic_projection_version": (CANONICAL_PRIMARY_COLLECTION_ROOT_PROJECTION_VERSION),
            "collection": collection,
            "count": len(ordered_item_digests),
            "ordered_item_digests": list(ordered_item_digests),
        }
    )


def primary_completion_command_projection(
    command: PrimaryCompletionCommand,
) -> dict[str, object]:
    """Bind the exact detail bytes and compact completion identity."""

    return {
        "semantic_projection_version": command.semantic_projection_version,
        "run_id": command.detail.run_id,
        "detail_semantic_sha256": command.detail.semantic_sha256,
        "detailed_result_exact_bytes_sha256": (
            command.completion.detailed_result.exact_bytes_sha256
        ),
        "detailed_result_schema_ref": (
            command.completion.detailed_result.schema_ref.model_dump(mode="json")
        ),
        "completion_semantic_sha256": command.completion.semantic_sha256,
    }


def create_primary_completion_command(
    *,
    result: CanonicalOfflineRunResult,
    prepared_identities: PreparedEventIdentityBatch | None,
    action_event_publications: PreparedInitialActionEventRevisionBatch,
    registry: SchemaRegistry | None = None,
) -> PrimaryCompletionCommand:
    """Create and schema-validate one exact aggregate command without writes."""

    if not isinstance(result, CanonicalOfflineRunResult):
        raise TypeError("result must be CanonicalOfflineRunResult")
    if not isinstance(action_event_publications, PreparedInitialActionEventRevisionBatch):
        raise TypeError("action_event_publications must be PreparedInitialActionEventRevisionBatch")
    if result.status not in {
        CanonicalOfflineRunStatus.SUCCEEDED,
        CanonicalOfflineRunStatus.NO_EVENTS,
    }:
        raise PrimaryCompletionError(
            PrimaryCompletionErrorCode.INVALID_COMMAND,
            "primary completion supports only SUCCEEDED or NO_EVENTS",
        )
    required = (
        result.window,
        result.package_set,
        result.coarse_qa_result,
        result.qa_completion_result,
        result.event_proposal_result,
        result.candidate_reduction_result,
    )
    if any(item is None for item in required):
        raise PrimaryCompletionError(
            PrimaryCompletionErrorCode.INVALID_COMMAND,
            "terminal canonical result lacks required completion lineage",
        )
    assert result.window is not None
    assert result.package_set is not None
    assert result.coarse_qa_result is not None
    assert result.qa_completion_result is not None
    assert result.event_proposal_result is not None
    assert result.candidate_reduction_result is not None
    active_registry = registry or default_schema_registry()
    detail_schema_ref = active_registry.resolve_version(
        CANONICAL_PRIMARY_COMPLETION_DETAIL_SCHEMA_ID,
        CANONICAL_PRIMARY_COMPLETION_DETAIL_SCHEMA_VERSION,
    ).ref
    detail_fields: dict[str, object] = {
        "schema_version": CANONICAL_PRIMARY_COMPLETION_DETAIL_WIRE_VERSION,
        "schema_ref": detail_schema_ref,
        "semantic_projection_version": (CANONICAL_PRIMARY_COMPLETION_DETAIL_PROJECTION_VERSION),
        "semantic_sha256": "0" * 64,
        "evidence_class": "LOCAL_CONFORMANCE",
        "production_eligible": False,
        "run_id": result.run_id,
        "recording_identity": result.recording_identity,
        "mcap_id": result.mcap_id,
        "execution_policy_sha256": result.execution_policy_sha256,
        "status": result.status.value,
        "processing_run": result.processing_run,
        "run_memberships": result.run_memberships,
        "window": result.window,
        "package_set": result.package_set,
        "coarse_qa_result": result.coarse_qa_result,
        "qa_completion_result": result.qa_completion_result,
        "dense_qa_executions": result.dense_qa_executions,
        "event_proposal_result": result.event_proposal_result,
        "candidate_reduction_result": result.candidate_reduction_result,
        "action_evidence_executions": result.action_evidence_executions,
        "provisional_fusion_result": result.provisional_fusion_result,
        "boundary_refinement_executions": result.boundary_refinement_executions,
        "final_fusion_context": result.final_fusion_context,
        "input_plan": result.input_plan,
        "reference_catalog": result.reference_catalog,
        "part_results": result.part_results,
        "barrier_reduction": result.barrier_reduction,
        "fusion_reduction": result.fusion_reduction,
        "output_decision": result.output_decision,
        "hypotheses": result.hypotheses,
        "prepared_identities": prepared_identities,
        "action_event_publications": action_event_publications,
    }
    draft_detail = CanonicalPrimaryCompletionDetail.model_construct(
        **detail_fields  # type: ignore[arg-type]
    )
    detail_fields["semantic_sha256"] = semantic_sha256(
        canonical_primary_completion_detail_projection(draft_detail)
    )
    detail = CanonicalPrimaryCompletionDetail.model_validate(detail_fields, strict=True)
    active_registry.validate_pinned(detail.schema_ref, detail.model_dump(mode="json"))

    detail_bytes = canonical_json_bytes(detail)
    detail_exact_digest = exact_bytes_sha256(detail_bytes)
    detailed_reference = DetailedResultArtifactReference(
        artifact_id=_detailed_result_artifact_id(detail_exact_digest),
        exact_bytes_sha256=detail_exact_digest,
        byte_count=len(detail_bytes),
        media_type="application/json",
        schema_ref=detail.schema_ref,
    )
    completion_ref = active_registry.resolve_version(
        PRIMARY_COMPLETION_RECORD_SCHEMA_ID,
        PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION,
    ).ref
    completion = _primary_completion_record_from_detail(
        detail,
        schema_ref=completion_ref,
        detailed_result=detailed_reference,
    )
    validate_registered_primary_completion_record(completion, active_registry)
    command_fields: dict[str, object] = {
        "schema_version": "1.0",
        "semantic_projection_version": (CANONICAL_PRIMARY_COMPLETION_COMMAND_PROJECTION_VERSION),
        "command_sha256": "0" * 64,
        "detail": detail,
        "completion": completion,
    }
    draft_command = PrimaryCompletionCommand.model_construct(
        **command_fields  # type: ignore[arg-type]
    )
    command_fields["command_sha256"] = semantic_sha256(
        primary_completion_command_projection(draft_command)
    )
    return PrimaryCompletionCommand.model_validate(command_fields, strict=True)


def _primary_completion_record_from_detail(
    detail: CanonicalPrimaryCompletionDetail,
    *,
    schema_ref: SchemaRef,
    detailed_result: DetailedResultArtifactReference,
) -> PrimaryCompletionRecord:
    prepared = detail.prepared_identities
    mutation = prepared.mutation if prepared is not None else None
    assignments = prepared.assignments if prepared is not None else ()
    new_identities = mutation.identities if mutation is not None else ()
    relations = mutation.relations if mutation is not None else ()
    successor_outbox = mutation.outbox if mutation is not None else ()
    publications = detail.action_event_publications.publications
    completed_at = detail.processing_run.completed_at
    assert completed_at is not None
    terminal_stage, terminal_evidence_sha256 = _primary_terminal_evidence(detail)
    if terminal_stage is PrimaryCompletionTerminalStage.FINAL_FUSION:
        assert detail.input_plan is not None
        assert detail.barrier_reduction is not None
        assert detail.output_decision is not None
        barrier_definition_sha256: str | None = detail.input_plan.call_plan.barrier_semantic_sha256
        barrier_reduction_sha256: str | None = detail.barrier_reduction.reduction_semantic_sha256
        output_decision_sha256: str | None = detail.output_decision.semantic_sha256
        output_policy_version: str | None = detail.output_decision.policy_version
        output_policy_sha256: str | None = detail.output_decision.policy_sha256
    else:
        barrier_definition_sha256 = None
        barrier_reduction_sha256 = None
        output_decision_sha256 = None
        output_policy_version = None
        output_policy_sha256 = None
    return create_primary_completion_record(
        schema_ref=schema_ref,
        run_id=detail.run_id,
        recording_identity=detail.recording_identity,
        mcap_id=detail.mcap_id,
        pipeline_version=detail.processing_run.pipeline_version,
        config_sha256=detail.execution_policy_sha256,
        started_at=detail.processing_run.started_at,
        outcome=(
            PrimaryCompletionOutcome.PRIMARY_COMPLETE
            if detail.status == "SUCCEEDED"
            else PrimaryCompletionOutcome.PRIMARY_COMPLETE_NO_EVENTS
        ),
        terminal_stage=terminal_stage,
        terminal_evidence_semantic_sha256=terminal_evidence_sha256,
        barrier_definition_semantic_sha256=barrier_definition_sha256,
        barrier_reduction_semantic_sha256=barrier_reduction_sha256,
        output_decision_semantic_sha256=output_decision_sha256,
        output_admission_policy_version=output_policy_version,
        output_admission_policy_sha256=output_policy_sha256,
        run_membership_count=len(detail.run_memberships),
        run_membership_digest_root=_model_collection_root(
            "run-memberships", detail.run_memberships
        ),
        barrier_member_count=len(detail.part_results),
        barrier_member_digest_root=canonical_collection_digest_root(
            "barrier-members",
            tuple(item.completion.completion_semantic_sha256 for item in detail.part_results),
        ),
        hypothesis_count=len(detail.hypotheses),
        hypothesis_digest_root=canonical_collection_digest_root(
            "event-hypotheses", tuple(item.semantic_sha256 for item in detail.hypotheses)
        ),
        identity_assignment_count=len(assignments),
        identity_assignment_digest_root=canonical_collection_digest_root(
            "identity-assignments",
            tuple(item.assignment_semantic_sha256 for item in assignments),
        ),
        new_identity_count=len(new_identities),
        new_identity_digest_root=_model_collection_root("new-identities", new_identities),
        identity_relation_count=len(relations),
        identity_relation_digest_root=_model_collection_root("identity-relations", relations),
        revision_count=len(publications),
        revision_digest_root=canonical_collection_digest_root(
            "action-event-revisions",
            tuple(item.revision.semantic_sha256 for item in publications),
        ),
        selection_decision_count=len(publications),
        selection_decision_digest_root=canonical_collection_digest_root(
            "action-event-selections",
            tuple(item.selection.semantic_sha256 for item in publications),
        ),
        current_selection_count=len(publications),
        current_selection_digest_root=canonical_collection_digest_root(
            "action-event-current-selections",
            tuple(
                semantic_sha256(
                    {
                        "current": item.current.model_dump(mode="json"),
                        "event_current_revision": item.current_revision.model_dump(mode="json"),
                    }
                )
                for item in publications
            ),
        ),
        successor_outbox_count=len(successor_outbox),
        successor_outbox_digest_root=_model_collection_root(
            "primary-successor-outbox", successor_outbox
        ),
        skipped_work_item_count=0,
        skipped_work_item_digest_root=canonical_collection_digest_root("skipped-work-items", ()),
        detailed_result=detailed_result,
        completed_at=completed_at,
    )


def _primary_terminal_evidence(
    detail: CanonicalPrimaryCompletionDetail,
) -> tuple[PrimaryCompletionTerminalStage, Sha256Digest]:
    if detail.input_plan is not None:
        if detail.output_decision is None:
            raise ValueError("final-fusion completion lacks its output decision")
        return (
            PrimaryCompletionTerminalStage.FINAL_FUSION,
            detail.output_decision.semantic_sha256,
        )
    if detail.provisional_fusion_result is not None:
        return (
            PrimaryCompletionTerminalStage.PROVISIONAL_FUSION,
            detail.provisional_fusion_result.semantic_sha256,
        )
    return (
        PrimaryCompletionTerminalStage.EVENT_PROPOSAL,
        detail.candidate_reduction_result.semantic_sha256,
    )


def _model_collection_root(collection: str, items: tuple[object, ...]) -> Sha256Digest:
    return canonical_collection_digest_root(
        collection,
        tuple(semantic_sha256(item.model_dump(mode="json")) for item in items),  # type: ignore[attr-defined]
    )


def _detailed_result_artifact_id(exact_digest: str) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"robata:canonical-primary-completion-detail:{exact_digest}",
        )
    )


__all__ = [
    "CANONICAL_PRIMARY_COLLECTION_ROOT_PROJECTION_VERSION",
    "CANONICAL_PRIMARY_COMPLETION_COMMAND_PROJECTION_VERSION",
    "CANONICAL_PRIMARY_COMPLETION_DETAIL_PROJECTION_VERSION",
    "CANONICAL_PRIMARY_COMPLETION_DETAIL_SCHEMA_ID",
    "CANONICAL_PRIMARY_COMPLETION_DETAIL_SCHEMA_VERSION",
    "CANONICAL_PRIMARY_COMPLETION_DETAIL_WIRE_VERSION",
    "CanonicalPrimaryCompletionDetail",
    "CommittedPrimaryCompletion",
    "PrimaryCompletionCommand",
    "PrimaryCompletionCommitResult",
    "PrimaryCompletionError",
    "PrimaryCompletionErrorCode",
    "PrimaryCompletionRepository",
    "canonical_collection_digest_root",
    "canonical_primary_completion_detail_projection",
    "create_primary_completion_command",
    "primary_completion_command_projection",
]
