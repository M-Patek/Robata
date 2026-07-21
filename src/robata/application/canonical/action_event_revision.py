"""Prepare local-conformance ActionEvent genesis publications without writes.

This module deliberately emits an internal typed payload, not the complete
registered ActionEvent wire contract from Architecture Design section 14. It
retains only facts proved by the current canonical path and leaves persistence
to the future primary-completion aggregate transaction.
"""

from __future__ import annotations

from typing import Final, Literal, Self

from pydantic import model_validator

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.models import (
    CanonicalOfflineExecutionPolicy,
    CanonicalOfflineRunStatus,
)
from robata.application.canonical.output_admission import _fusion_event_fingerprint
from robata.application.canonical.projections import _stable_uuid
from robata.application.canonical.result_validation import CanonicalOfflineRunResult
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import (
    LogicalNode,
    NodeLogicalKey,
    OpaqueUuid,
    logical_node_from_semantic_digest,
)
from robata.contracts.revisions import (
    CurrentSelection,
    ImmutableNodeRevision,
    RevisionEligibility,
    SelectionDecision,
    create_immutable_node_revision,
    create_selection_decision,
)
from robata.event_pipeline.identity_registry import (
    AdmissionEvidenceClass,
    EventCurrentRevisionReference,
    EventIdentityAssignment,
    EventIdentityAssignmentDisposition,
    PlatformEnrichedEventHypothesis,
    PlatformEnrichedOutputReference,
    PreparedEventIdentityBatch,
)
from robata.inference.enrichment import (
    EnrichedEvidenceReference,
    EnrichedProviderClaim,
    ProviderClaimKind,
    ProviderObservation,
)
from robata.inference.input_plan import RequestCatalog

ACTION_EVENT_SUBJECT_PROJECTION_VERSION: Final = "canonical-action-event-subject-v1"
ACTION_EVENT_PAYLOAD_PROJECTION_VERSION: Final = "canonical-action-event-local-payload-v1"
ACTION_EVENT_LINEAGE_PROJECTION_VERSION: Final = "canonical-action-event-local-lineage-v1"
ACTION_EVENT_REVISION_POLICY_VERSION: Final = "canonical-action-event-local-revision-v1"
ACTION_EVENT_SELECTION_POLICY_VERSION: Final = "canonical-action-event-local-selection-v1"
ACTION_EVENT_CURRENT_PROJECTION_VERSION: Final = "current-selection-v1"


class CanonicalActionEventRevisionError(ValueError):
    """The supplied canonical facts cannot truthfully form an initial revision."""


class CanonicalActionEventCitedFrame(StrictModel):
    """One authoritative source frame cited by the reduced provider claim."""

    package_id: OpaqueUuid
    package_ordinal: int
    package_semantic_content_sha256: Sha256Digest
    package_manifest_sha256: Sha256Digest
    camera_id: CameraId
    camera_ordinal: int
    frame_id: OpaqueUuid
    frame_ordinal: int
    aligned_timestamp_ns: Nanoseconds
    source_timestamp_ns: Nanoseconds
    source_artifact_uri: str
    source_artifact_sha256: Sha256Digest


class CanonicalActionEventCameraSource(StrictModel):
    """Per-camera citations; NOT_CITED is never negative event evidence."""

    camera_id: CameraId
    camera_ordinal: int
    citation_status: Literal["CITED", "NOT_CITED"]
    cited_frames: tuple[CanonicalActionEventCitedFrame, ...]

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if not 0 <= self.camera_ordinal < len(CAMERA_IDS):
            raise ValueError("camera source ordinal is outside the six-camera domain")
        if CAMERA_IDS[self.camera_ordinal] is not self.camera_id:
            raise ValueError("camera source ordinal does not match camera identity")
        if any(
            item.camera_id is not self.camera_id or item.camera_ordinal != self.camera_ordinal
            for item in self.cited_frames
        ):
            raise ValueError("cited frame crosses its camera source")
        keys = tuple(
            (item.package_ordinal, item.frame_ordinal, item.frame_id) for item in self.cited_frames
        )
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("camera source citations must be unique and canonical")
        expected_status = "CITED" if self.cited_frames else "NOT_CITED"
        if self.citation_status != expected_status:
            raise ValueError("camera citation status does not match its cited frames")
        return self


class CanonicalActionEventRevisionPayload(StrictModel):
    """Known local event facts; this is not the production ActionEvent schema."""

    schema_version: Literal["1.0"]
    semantic_projection_version: Literal["canonical-action-event-local-payload-v1"]
    recording_identity: Sha256Digest
    event_id: OpaqueUuid
    mcap_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid
    alignment_id: OpaqueUuid
    effective_interval: NanosecondInterval
    action_label: str
    observation: ProviderObservation
    conflict_codes: tuple[str, ...]
    identity_disposition: EventIdentityAssignmentDisposition
    event_status: Literal["NEEDS_REVIEW", "AMBIGUOUS"]
    camera_sources: tuple[CanonicalActionEventCameraSource, ...]
    evidence_class: Literal[AdmissionEvidenceClass.LOCAL_CONFORMANCE]
    production_eligible: Literal[False]

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if not self.action_label.strip():
            raise ValueError("local ActionEvent revision requires a nonblank action label")
        if self.conflict_codes != tuple(sorted(set(self.conflict_codes))):
            raise ValueError("conflict codes must be unique and canonical")
        if tuple(item.camera_id for item in self.camera_sources) != CAMERA_IDS:
            raise ValueError("local ActionEvent revision requires six ordered camera sources")
        ambiguous = (
            self.identity_disposition is EventIdentityAssignmentDisposition.AMBIGUOUS
            or self.observation is ProviderObservation.CONFLICT
        )
        expected_status = "AMBIGUOUS" if ambiguous else "NEEDS_REVIEW"
        if self.event_status != expected_status:
            raise ValueError("event status does not match identity/provider ambiguity")
        return self


class CanonicalActionEventRevisionLineage(StrictModel):
    """Exact upstream facts retained separately from the event payload."""

    schema_version: Literal["1.0"]
    semantic_projection_version: Literal["canonical-action-event-local-lineage-v1"]
    admitted_context_semantic_sha256: Sha256Digest
    execution_policy_sha256: Sha256Digest
    request_catalog_id: OpaqueUuid
    request_catalog_semantic_sha256: Sha256Digest
    fusion_reduction_logical_key: NodeLogicalKey
    fusion_reduction_semantic_sha256: Sha256Digest
    output_decision_id: OpaqueUuid
    output_decision_semantic_sha256: Sha256Digest
    reduction_claim_semantic_sha256: Sha256Digest
    event_hypothesis_logical_key: NodeLogicalKey
    event_hypothesis_semantic_sha256: Sha256Digest
    identity_assignment_logical_key: NodeLogicalKey
    identity_assignment_semantic_sha256: Sha256Digest
    source_enrichments: tuple[PlatformEnrichedOutputReference, ...]

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        keys = tuple(item.enrichment_logical_key for item in self.source_enrichments)
        if not keys or keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("ActionEvent source enrichments must be nonempty and canonical")
        return self


class PreparedInitialActionEventRevision(StrictModel):
    """One complete, side-effect-free genesis publication command."""

    assignment: EventIdentityAssignment
    subject: LogicalNode
    payload: CanonicalActionEventRevisionPayload
    lineage: CanonicalActionEventRevisionLineage
    revision: ImmutableNodeRevision
    selection: SelectionDecision
    current: CurrentSelection
    current_revision: EventCurrentRevisionReference

    @model_validator(mode="after")
    def validate_publication(self) -> Self:
        subject_digest = semantic_sha256(
            canonical_action_event_subject_projection(
                recording_identity=self.payload.recording_identity,
                event_id=self.payload.event_id,
            )
        )
        if (
            self.assignment.recording_identity != self.payload.recording_identity
            or self.assignment.event_id != self.payload.event_id
            or self.assignment.assignment_logical_key
            != self.lineage.identity_assignment_logical_key
            or self.assignment.assignment_semantic_sha256
            != self.lineage.identity_assignment_semantic_sha256
        ):
            raise ValueError("ActionEvent publication differs from its identity assignment")
        if (
            self.subject.node_type != "ACTION_EVENT"
            or self.subject.semantic_sha256 != subject_digest
            or self.subject.node_logical_key != f"action-event:{subject_digest}"
        ):
            raise ValueError("ActionEvent publication has an invalid stable subject")
        payload_digest = semantic_sha256(canonical_action_event_payload_projection(self.payload))
        lineage_digest = semantic_sha256(canonical_action_event_lineage_projection(self.lineage))
        if (
            self.revision.subject_type != self.subject.node_type
            or self.revision.subject_id != self.subject.node_logical_key
            or self.revision.payload_sha256 != payload_digest
            or self.revision.lineage_sha256 != lineage_digest
            or self.revision.status_at_publication != "PUBLISHED_LOCAL_CONFORMANCE"
            or self.revision.eligibility_at_publication is not RevisionEligibility.ELIGIBLE
            or self.revision.supersedes_revision_id is not None
            or self.revision.supersedes_revision_logical_key is not None
            or self.revision.published_at != self.assignment.decided_at
        ):
            raise ValueError("ActionEvent genesis revision bindings are inconsistent")
        if (
            self.selection.subject_type != self.subject.node_type
            or self.selection.subject_id != self.subject.node_logical_key
            or self.selection.selected_revision_id != self.revision.revision_id
            or self.selection.selected_revision_logical_key != self.revision.revision_logical_key
            or self.selection.previous_selection_decision_id is not None
            or self.selection.previous_selection_decision_logical_key is not None
            or self.selection.selection_sequence != 1
            or self.selection.selected_at != self.assignment.decided_at
        ):
            raise ValueError("ActionEvent genesis selection bindings are inconsistent")
        if (
            self.current.subject_type != self.subject.node_type
            or self.current.subject_id != self.subject.node_logical_key
            or self.current.selected_revision_id != self.revision.revision_id
            or self.current.selection_decision_id != self.selection.selection_decision_id
            or self.current.selection_policy_version != ACTION_EVENT_SELECTION_POLICY_VERSION
            or self.current.projection_version != ACTION_EVENT_CURRENT_PROJECTION_VERSION
            or self.current.selected_at != self.assignment.decided_at
        ):
            raise ValueError("ActionEvent current-selection projection is inconsistent")
        if (
            self.current_revision.recording_identity != self.payload.recording_identity
            or self.current_revision.event_id != self.payload.event_id
            or self.current_revision.revision_logical_key != self.revision.revision_logical_key
            or self.current_revision.revision_semantic_sha256 != self.revision.semantic_sha256
            or self.current_revision.effective_interval != self.payload.effective_interval
        ):
            raise ValueError("identity current-revision reference is inconsistent")
        return self


class PreparedInitialActionEventRevisionBatch(StrictModel):
    """All genesis publications prepared for one canonical terminal result."""

    recording_identity: Sha256Digest
    outcome: Literal["PREPARED", "NO_EVENTS", "ABSTAINED"]
    expected_generation: int | None
    expected_fence: int | None
    publications: tuple[PreparedInitialActionEventRevision, ...]

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        if self.outcome == "PREPARED":
            if (
                self.expected_generation is None
                or self.expected_fence is None
                or not self.publications
            ):
                raise ValueError("prepared ActionEvent batch requires fence-bound publications")
        elif (
            self.expected_generation is not None
            or self.expected_fence is not None
            or self.publications
        ):
            raise ValueError("empty terminal outcome cannot carry publication commands")
        if any(
            item.payload.recording_identity != self.recording_identity for item in self.publications
        ):
            raise ValueError("ActionEvent publication batch crosses recording scope")
        event_ids = tuple(item.payload.event_id for item in self.publications)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("ActionEvent publication batch contains duplicate event IDs")
        return self


def canonical_action_event_subject_projection(
    *,
    recording_identity: str,
    event_id: str,
) -> dict[str, str]:
    """Return the stable physical-event subject identity projection."""

    return {
        "semantic_projection_version": ACTION_EVENT_SUBJECT_PROJECTION_VERSION,
        "recording_identity": recording_identity,
        "event_id": event_id,
    }


def canonical_action_event_payload_projection(
    payload: CanonicalActionEventRevisionPayload,
) -> dict[str, object]:
    """Return the complete local payload projection."""

    return payload.model_dump(mode="json")


def canonical_action_event_lineage_projection(
    lineage: CanonicalActionEventRevisionLineage,
) -> dict[str, object]:
    """Return the exact upstream lineage projection."""

    return lineage.model_dump(mode="json")


def prepare_initial_action_event_publications(
    *,
    context: AdmittedRecordingContextV2,
    result: CanonicalOfflineRunResult,
    prepared_identities: PreparedEventIdentityBatch | None,
    execution_policy: CanonicalOfflineExecutionPolicy,
) -> PreparedInitialActionEventRevisionBatch:
    """Prepare genesis facts for one completed local canonical run.

    The function is side-effect free. A caller may safely retry it: all IDs and
    timestamps derive from already committed inputs. A REUSED assignment is
    rejected because the snapshot lacks the prior selection-chain information
    required to construct an honest successor.
    """

    admitted_context = context
    identity = prepared_identities
    if not isinstance(result, CanonicalOfflineRunResult):
        raise CanonicalActionEventRevisionError("result must be CanonicalOfflineRunResult")
    if not isinstance(admitted_context, AdmittedRecordingContextV2):
        raise CanonicalActionEventRevisionError(
            "admitted_context must be AdmittedRecordingContextV2"
        )
    if (
        not isinstance(execution_policy, CanonicalOfflineExecutionPolicy)
        or execution_policy.semantic_sha256 != result.execution_policy_sha256
    ):
        raise CanonicalActionEventRevisionError(
            "execution policy does not match the canonical result"
        )
    if result.recording_identity != admitted_context.recording_identity:
        raise CanonicalActionEventRevisionError(
            "result and admission context cross recording scope"
        )
    window = result.window
    if (
        result.mcap_id != admitted_context.ready_manifest.mcap_id
        or window is None
        or window.recording_identity != admitted_context.recording_identity
        or window.source_content_sha256 != admitted_context.source_content_sha256
        or window.mcap_id != admitted_context.ready_manifest.mcap_id
        or window.camera_mapping_run_id != admitted_context.ready_manifest.camera_mapping_run_id
        or window.alignment_id != admitted_context.alignment_manifest.alignment_id
        or window.camera_mapping_semantic_sha256 != admitted_context.camera_mapping_semantic_sha256
        or window.alignment_semantic_sha256 != admitted_context.alignment_semantic_sha256
        or window.source_lineage_sha256 != admitted_context.semantic_sha256
        or window.recording_duration_ns != admitted_context.ready_manifest.recording.duration_ns
    ):
        raise CanonicalActionEventRevisionError(
            "canonical result does not bind the supplied admission context"
        )
    reduction = result.fusion_reduction
    if reduction is not None and (
        reduction.reduction_policy != execution_policy.reduction_policy
        or reduction.reduction_policy_version != execution_policy.reduction_policy_version
    ):
        raise CanonicalActionEventRevisionError(
            "fusion reduction does not bind the execution policy"
        )
    output_decision = result.output_decision
    if output_decision is not None and (
        output_decision.policy_version != execution_policy.output_admission_policy.version
        or output_decision.policy_sha256 != execution_policy.output_admission_policy.semantic_sha256
    ):
        raise CanonicalActionEventRevisionError(
            "output decision does not bind the execution policy"
        )

    terminal_status = result.status
    if terminal_status in {
        CanonicalOfflineRunStatus.NO_EVENTS,
        CanonicalOfflineRunStatus.ABSTAINED,
    }:
        if result.hypotheses or identity is not None:
            raise CanonicalActionEventRevisionError(
                "empty terminal outcomes cannot carry ActionEvent identity preparation"
            )
        empty_outcome: Literal["NO_EVENTS", "ABSTAINED"] = (
            "NO_EVENTS" if terminal_status is CanonicalOfflineRunStatus.NO_EVENTS else "ABSTAINED"
        )
        return PreparedInitialActionEventRevisionBatch(
            recording_identity=result.recording_identity,
            outcome=empty_outcome,
            expected_generation=None,
            expected_fence=None,
            publications=(),
        )
    if terminal_status is not CanonicalOfflineRunStatus.SUCCEEDED:
        raise CanonicalActionEventRevisionError(
            "only a successful canonical run can prepare ActionEvent revisions"
        )
    if identity is None:
        raise CanonicalActionEventRevisionError(
            "successful ActionEvent publication requires prepared event identities"
        )
    if not isinstance(identity, PreparedEventIdentityBatch):
        raise CanonicalActionEventRevisionError("identity must be PreparedEventIdentityBatch")
    if identity.recording_identity != result.recording_identity:
        raise CanonicalActionEventRevisionError("identity preparation crosses recording scope")
    if any(
        assignment.disposition is EventIdentityAssignmentDisposition.REUSED
        for assignment in identity.assignments
    ):
        raise CanonicalActionEventRevisionError(
            "REUSED identity assignments require a prior selection chain"
        )
    if result.input_plan is None or result.fusion_reduction is None:
        raise CanonicalActionEventRevisionError(
            "successful result is missing input-plan or fusion-reduction lineage"
        )
    if result.output_decision is None or result.output_decision.decision != "ADMITTED":
        raise CanonicalActionEventRevisionError(
            "successful result is missing an admitted output decision"
        )
    if len(identity.assignments) != len(result.hypotheses):
        raise CanonicalActionEventRevisionError(
            "identity preparation does not cover every event hypothesis"
        )

    hypotheses = {item.event_hypothesis_logical_key: item for item in result.hypotheses}
    if len(hypotheses) != len(result.hypotheses):
        raise CanonicalActionEventRevisionError("result contains duplicate event hypotheses")
    expected_keys = tuple(item.event_hypothesis_logical_key for item in identity.assignments)
    if expected_keys != identity.ordered_hypothesis_logical_keys:
        raise CanonicalActionEventRevisionError(
            "identity preparation order does not match its assignments"
        )
    canonical_keys = tuple(
        item.event_hypothesis_logical_key
        for item in sorted(
            result.hypotheses,
            key=lambda item: (
                item.effective_start_ns,
                item.effective_end_ns,
                item.event_hypothesis_logical_key,
            ),
        )
    )
    if expected_keys != canonical_keys:
        raise CanonicalActionEventRevisionError(
            "identity preparation does not preserve canonical hypothesis order"
        )
    publications = tuple(
        _prepare_one_action_event_revision(
            context=admitted_context,
            result=result,
            catalog=result.input_plan.request_catalog,
            hypothesis=hypotheses.get(assignment.event_hypothesis_logical_key),
            assignment=assignment,
            projector_policy_version=execution_policy.projector_policy_version,
        )
        for assignment in identity.assignments
    )
    return PreparedInitialActionEventRevisionBatch(
        recording_identity=result.recording_identity,
        outcome="PREPARED",
        expected_generation=identity.expected_generation,
        expected_fence=identity.expected_fence,
        publications=publications,
    )


def _prepare_one_action_event_revision(
    *,
    context: AdmittedRecordingContextV2,
    result: CanonicalOfflineRunResult,
    catalog: RequestCatalog,
    hypothesis: PlatformEnrichedEventHypothesis | None,
    assignment: EventIdentityAssignment,
    projector_policy_version: str,
) -> PreparedInitialActionEventRevision:
    if hypothesis is None:
        raise CanonicalActionEventRevisionError(
            "identity assignment references an unknown event hypothesis"
        )
    if (
        assignment.recording_identity != result.recording_identity
        or hypothesis.recording_identity != result.recording_identity
        or assignment.event_hypothesis_semantic_sha256 != hypothesis.semantic_sha256
        or assignment.event_hypothesis_logical_key != hypothesis.event_hypothesis_logical_key
    ):
        raise CanonicalActionEventRevisionError(
            "identity assignment does not bind the exact event hypothesis"
        )
    reduction = result.fusion_reduction
    if reduction is None:
        raise CanonicalActionEventRevisionError("missing fusion reduction")
    if hypothesis.fusion_output_ordinal >= len(reduction.claims):
        raise CanonicalActionEventRevisionError(
            "hypothesis fusion ordinal is outside the reduction"
        )
    reduced_claim = reduction.claims[hypothesis.fusion_output_ordinal]
    claim = reduced_claim.representative
    if (
        reduced_claim.fusion_output_ordinal != hypothesis.fusion_output_ordinal
        or claim.kind is not ProviderClaimKind.FUSION_HYPOTHESIS
        or claim.interval is None
        or not claim.evidence
        or NanosecondInterval(
            start_ns=claim.interval.start_ns,
            end_ns=claim.interval.end_ns,
        )
        != hypothesis.effective_interval
        or claim.label is None
    ):
        raise CanonicalActionEventRevisionError(
            "reduced claim does not bind the event hypothesis interval or label"
        )
    expected_fingerprint = _fusion_event_fingerprint(
        recording_identity=result.recording_identity,
        claim=claim,
        projector_version=projector_policy_version,
    )
    expected_fusion_digest = semantic_sha256(
        {
            "semantic_fingerprint_sha256": expected_fingerprint,
            "fusion_reduction_semantic_sha256": reduction.semantic_sha256,
            "projector_version": projector_policy_version,
        }
    )
    if (
        hypothesis.semantic_fingerprint_sha256 != expected_fingerprint
        or hypothesis.fusion_logical_key != f"fusion:{expected_fusion_digest}"
    ):
        raise CanonicalActionEventRevisionError(
            "event hypothesis fingerprint differs from the reduced claim"
        )

    camera_sources = _camera_sources(catalog, claim)
    event_status: Literal["NEEDS_REVIEW", "AMBIGUOUS"] = (
        "AMBIGUOUS"
        if (
            assignment.disposition is EventIdentityAssignmentDisposition.AMBIGUOUS
            or claim.observation is ProviderObservation.CONFLICT
        )
        else "NEEDS_REVIEW"
    )
    payload = CanonicalActionEventRevisionPayload(
        schema_version="1.0",
        semantic_projection_version=ACTION_EVENT_PAYLOAD_PROJECTION_VERSION,
        recording_identity=result.recording_identity,
        event_id=assignment.event_id,
        mcap_id=context.ready_manifest.mcap_id,
        camera_mapping_run_id=context.ready_manifest.camera_mapping_run_id,
        alignment_id=context.alignment_manifest.alignment_id,
        effective_interval=hypothesis.effective_interval,
        action_label=claim.label,
        observation=claim.observation,
        conflict_codes=tuple(sorted(claim.conflict_codes)),
        identity_disposition=assignment.disposition,
        event_status=event_status,
        camera_sources=camera_sources,
        evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
        production_eligible=False,
    )
    output_decision = result.output_decision
    if output_decision is None:
        raise CanonicalActionEventRevisionError("missing admitted output decision")
    lineage = CanonicalActionEventRevisionLineage(
        schema_version="1.0",
        semantic_projection_version=ACTION_EVENT_LINEAGE_PROJECTION_VERSION,
        admitted_context_semantic_sha256=context.semantic_sha256,
        execution_policy_sha256=result.execution_policy_sha256,
        request_catalog_id=catalog.request_catalog_id,
        request_catalog_semantic_sha256=catalog.semantic_sha256,
        fusion_reduction_logical_key=reduction.reduction_logical_key,
        fusion_reduction_semantic_sha256=reduction.semantic_sha256,
        output_decision_id=output_decision.decision_id,
        output_decision_semantic_sha256=output_decision.semantic_sha256,
        reduction_claim_semantic_sha256=reduced_claim.claim_semantic_sha256,
        event_hypothesis_logical_key=hypothesis.event_hypothesis_logical_key,
        event_hypothesis_semantic_sha256=hypothesis.semantic_sha256,
        identity_assignment_logical_key=assignment.assignment_logical_key,
        identity_assignment_semantic_sha256=assignment.assignment_semantic_sha256,
        source_enrichments=hypothesis.source_enrichments,
    )
    subject_digest = semantic_sha256(
        canonical_action_event_subject_projection(
            recording_identity=result.recording_identity,
            event_id=assignment.event_id,
        )
    )
    subject = logical_node_from_semantic_digest(
        node_type="ACTION_EVENT",
        key_namespace="action-event",
        semantic_sha256=subject_digest,
        identity_policy_version=ACTION_EVENT_SUBJECT_PROJECTION_VERSION,
    )
    payload_digest = semantic_sha256(canonical_action_event_payload_projection(payload))
    lineage_digest = semantic_sha256(canonical_action_event_lineage_projection(lineage))
    revision = create_immutable_node_revision(
        revision_id=_stable_uuid(
            "canonical-action-event-revision",
            subject.node_logical_key,
            payload_digest,
            lineage_digest,
            "PUBLISHED_LOCAL_CONFORMANCE",
            RevisionEligibility.ELIGIBLE.value,
            ACTION_EVENT_REVISION_POLICY_VERSION,
        ),
        subject_type="ACTION_EVENT",
        subject_id=subject.node_logical_key,
        revision_key_namespace="canonical-action-event-revision",
        payload_sha256=payload_digest,
        lineage_sha256=lineage_digest,
        status_at_publication="PUBLISHED_LOCAL_CONFORMANCE",
        eligibility_at_publication=RevisionEligibility.ELIGIBLE,
        revision_policy_version=ACTION_EVENT_REVISION_POLICY_VERSION,
        supersedes_revision_id=None,
        supersedes_revision_logical_key=None,
        published_at=assignment.decided_at,
    )
    selection = create_selection_decision(
        selection_decision_id=_stable_uuid(
            "canonical-action-event-selection",
            subject.node_logical_key,
            revision.revision_logical_key,
            ACTION_EVENT_SELECTION_POLICY_VERSION,
            ACTION_EVENT_CURRENT_PROJECTION_VERSION,
        ),
        selection_key_namespace="canonical-action-event-selection",
        subject_type="ACTION_EVENT",
        subject_id=subject.node_logical_key,
        selected_revision_id=revision.revision_id,
        selected_revision_logical_key=revision.revision_logical_key,
        previous_selection_decision_id=None,
        previous_selection_decision_logical_key=None,
        selection_sequence=1,
        selection_policy_version=ACTION_EVENT_SELECTION_POLICY_VERSION,
        projection_version=ACTION_EVENT_CURRENT_PROJECTION_VERSION,
        selected_at=assignment.decided_at,
    )
    current = CurrentSelection(
        schema_version="1.0",
        subject_type="ACTION_EVENT",
        subject_id=subject.node_logical_key,
        selected_revision_id=revision.revision_id,
        selection_decision_id=selection.selection_decision_id,
        selection_policy_version=ACTION_EVENT_SELECTION_POLICY_VERSION,
        projection_version=ACTION_EVENT_CURRENT_PROJECTION_VERSION,
        selected_at=assignment.decided_at,
    )
    current_revision = EventCurrentRevisionReference(
        recording_identity=result.recording_identity,
        event_id=assignment.event_id,
        revision_logical_key=revision.revision_logical_key,
        revision_semantic_sha256=revision.semantic_sha256,
        effective_interval=hypothesis.effective_interval,
    )
    try:
        return PreparedInitialActionEventRevision(
            assignment=assignment,
            subject=subject,
            payload=payload,
            lineage=lineage,
            revision=revision,
            selection=selection,
            current=current,
            current_revision=current_revision,
        )
    except ValueError as exc:
        raise CanonicalActionEventRevisionError(
            "constructed ActionEvent publication failed its cross-fact checks"
        ) from exc


def _camera_sources(
    catalog: RequestCatalog,
    claim: EnrichedProviderClaim,
) -> tuple[CanonicalActionEventCameraSource, ...]:
    cited_by_camera: dict[CameraId, list[CanonicalActionEventCitedFrame]] = {
        camera_id: [] for camera_id in CAMERA_IDS
    }
    seen_frames: set[tuple[int, int, int]] = set()
    for evidence in claim.evidence:
        cited = _cited_frame(catalog, evidence)
        frame_key = (
            cited.package_ordinal,
            cited.camera_ordinal,
            cited.frame_ordinal,
        )
        if frame_key in seen_frames:
            raise CanonicalActionEventRevisionError(
                "one reduced claim cites the same authoritative frame more than once"
            )
        seen_frames.add(frame_key)
        cited_by_camera[cited.camera_id].append(cited)
    return tuple(
        CanonicalActionEventCameraSource(
            camera_id=camera_id,
            camera_ordinal=ordinal,
            citation_status="CITED" if cited_by_camera[camera_id] else "NOT_CITED",
            cited_frames=tuple(
                sorted(
                    cited_by_camera[camera_id],
                    key=lambda item: (
                        item.package_ordinal,
                        item.frame_ordinal,
                        item.frame_id,
                    ),
                )
            ),
        )
        for ordinal, camera_id in enumerate(CAMERA_IDS)
    )


def _cited_frame(
    catalog: RequestCatalog,
    evidence: EnrichedEvidenceReference,
) -> CanonicalActionEventCitedFrame:
    try:
        package = catalog.packages[evidence.package_ordinal]
        camera = package.cameras[evidence.camera_ordinal]
        frame = camera.frames[evidence.frame_ordinal]
    except IndexError as exc:
        raise CanonicalActionEventRevisionError(
            "claim evidence lies outside the authoritative request catalog"
        ) from exc
    if (
        package.ordinal != evidence.package_ordinal
        or package.package_id != evidence.package_id
        or package.semantic_content_sha256 != evidence.package_semantic_content_sha256
        or package.manifest_bytes_sha256 != evidence.package_manifest_sha256
        or camera.ordinal != evidence.camera_ordinal
        or camera.camera_id is not evidence.camera_id
        or frame.ordinal != evidence.frame_ordinal
        or frame.frame_id != evidence.frame_id
        or frame.aligned_timestamp_ns != evidence.aligned_timestamp_ns
        or frame.source_timestamp_ns != evidence.source_timestamp_ns
        or frame.source_artifact_uri != evidence.source_artifact_uri
        or frame.source_artifact_sha256 != evidence.source_artifact_sha256
    ):
        raise CanonicalActionEventRevisionError(
            "claim evidence differs from the authoritative request catalog"
        )
    return CanonicalActionEventCitedFrame(
        package_id=package.package_id,
        package_ordinal=package.ordinal,
        package_semantic_content_sha256=package.semantic_content_sha256,
        package_manifest_sha256=package.manifest_bytes_sha256,
        camera_id=camera.camera_id,
        camera_ordinal=camera.ordinal,
        frame_id=frame.frame_id,
        frame_ordinal=frame.ordinal,
        aligned_timestamp_ns=frame.aligned_timestamp_ns,
        source_timestamp_ns=frame.source_timestamp_ns,
        source_artifact_uri=frame.source_artifact_uri,
        source_artifact_sha256=frame.source_artifact_sha256,
    )


__all__ = [
    "ACTION_EVENT_CURRENT_PROJECTION_VERSION",
    "ACTION_EVENT_LINEAGE_PROJECTION_VERSION",
    "ACTION_EVENT_PAYLOAD_PROJECTION_VERSION",
    "ACTION_EVENT_REVISION_POLICY_VERSION",
    "ACTION_EVENT_SELECTION_POLICY_VERSION",
    "ACTION_EVENT_SUBJECT_PROJECTION_VERSION",
    "CanonicalActionEventCameraSource",
    "CanonicalActionEventCitedFrame",
    "CanonicalActionEventRevisionError",
    "CanonicalActionEventRevisionLineage",
    "CanonicalActionEventRevisionPayload",
    "PreparedInitialActionEventRevision",
    "PreparedInitialActionEventRevisionBatch",
    "canonical_action_event_lineage_projection",
    "canonical_action_event_payload_projection",
    "canonical_action_event_subject_projection",
    "prepare_initial_action_event_publications",
]
