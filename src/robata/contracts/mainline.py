"""Minimal contracts for the local six-camera mainline vertical slice."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.artifacts import ArtifactUri
from robata.contracts.cameras import CameraId, SixCameraMap
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
PositiveFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
UnitInterval = Annotated[
    float,
    Field(strict=True, ge=0, le=1, allow_inf_nan=False),
]
type FrozenTuple[T] = Annotated[tuple[T, ...], Field(strict=False)]
"""Accept a JSON array while retaining immutable, strictly validated elements."""


class SamplingPurpose(StrEnum):
    QA_COARSE = "QA_COARSE"
    QA_DENSE = "QA_DENSE"
    EVENT_PROPOSAL = "EVENT_PROPOSAL"
    ACTION_DENSE = "ACTION_DENSE"
    BOUNDARY_REFINEMENT = "BOUNDARY_REFINEMENT"


class SamplingStrategy(StrEnum):
    UNIFORM = "UNIFORM"
    DENSE = "DENSE"
    NOT_REQUESTED = "NOT_REQUESTED"


class CameraPackageStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    NO_FRAME = "NO_FRAME"
    UNAVAILABLE = "UNAVAILABLE"
    CORRUPT = "CORRUPT"
    NOT_REQUESTED = "NOT_REQUESTED"


class VisionTask(StrEnum):
    QA_COARSE = "QA_COARSE"
    QA_DENSE = "QA_DENSE"
    EVENT_PROPOSAL = "EVENT_PROPOSAL"
    ACTION_EVIDENCE = "ACTION_EVIDENCE"
    BOUNDARY_REFINEMENT = "BOUNDARY_REFINEMENT"
    FUSION_ADJUDICATION = "FUSION_ADJUDICATION"


class CameraQAStatus(StrEnum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"
    UNKNOWN = "UNKNOWN"
    INCOMPLETE = "INCOMPLETE"


class RecordingQAStatus(StrEnum):
    USABLE = "USABLE"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"
    INCOMPLETE = "INCOMPLETE"


class QAIssueSeverity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CameraEvidenceStatus(StrEnum):
    SUPPORTING = "SUPPORTING"
    PARTIAL = "PARTIAL"
    NO_EVENT = "NO_EVENT"
    OCCLUDED = "OCCLUDED"
    UNUSABLE = "UNUSABLE"
    MISSING = "MISSING"


class BoundaryStatus(StrEnum):
    OBSERVED = "OBSERVED"
    NO_BOUNDARY = "NO_BOUNDARY"
    OCCLUDED = "OCCLUDED"
    UNUSABLE = "UNUSABLE"
    MISSING = "MISSING"


class CandidateEventStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class ActionEventStatus(StrEnum):
    FINAL = "FINAL"
    AMBIGUOUS = "AMBIGUOUS"


class InferenceStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    INVALID_OUTPUT = "INVALID_OUTPUT"


class Retryability(StrEnum):
    RETRYABLE = "RETRYABLE"
    RATE_LIMITED = "RATE_LIMITED"
    PERMANENT = "PERMANENT"


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    PRIMARY_COMPLETE = "PRIMARY_COMPLETE"
    PRIMARY_COMPLETE_NO_EVENTS = "PRIMARY_COMPLETE_NO_EVENTS"
    PRIMARY_BLOCKED = "PRIMARY_BLOCKED"
    FAILED = "FAILED"


class MainlineStage(StrEnum):
    WINDOWING = "WINDOWING"
    SAMPLING = "SAMPLING"
    QA_INFERENCE = "QA_INFERENCE"
    QA_AGGREGATION = "QA_AGGREGATION"
    EVENT_PROPOSAL = "EVENT_PROPOSAL"
    ACTION_EVIDENCE = "ACTION_EVIDENCE"
    BOUNDARY_REFINEMENT = "BOUNDARY_REFINEMENT"
    FUSION = "FUSION"
    PUBLISH = "PUBLISH"


class StageStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"
    SKIPPED = "SKIPPED"


def _contains(outer: NanosecondInterval, inner: NanosecondInterval) -> bool:
    return outer.start_ns <= inner.start_ns and inner.end_ns <= outer.end_ns


def _intersects(left: NanosecondInterval, right: NanosecondInterval) -> bool:
    return left.start_ns < right.end_ns and right.start_ns < left.end_ns


def _require_sorted_unique(values: tuple[int, ...], name: str) -> None:
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} must be strictly increasing and unique")


class TemporalWindow(StrictModel):
    """One scheduling interval; frames live only in a derived package."""

    schema_version: Literal["1.0"]
    window_id: OpaqueUuid
    mcap_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid | None
    alignment_id: OpaqueUuid | None
    requested_interval: NanosecondInterval
    interval: NanosecondInterval
    purpose: SamplingPurpose
    parent_window_id: OpaqueUuid | None = None
    source_candidate_id: OpaqueUuid | None = None
    source_event_id: OpaqueUuid | None = None
    generation: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if not _contains(self.requested_interval, self.interval):
            raise ValueError("effective interval must be contained by requested_interval")
        if self.source_candidate_id is not None and self.source_event_id is not None:
            raise ValueError("a window may reference a candidate or event, not both")
        if self.generation == 0 and self.parent_window_id is not None:
            raise ValueError("generation zero windows cannot have a parent_window_id")
        if self.generation > 0 and self.parent_window_id is None:
            raise ValueError("derived windows require a parent_window_id")
        if (
            self.purpose
            in {
                SamplingPurpose.ACTION_DENSE,
                SamplingPurpose.BOUNDARY_REFINEMENT,
            }
            and self.source_candidate_id is None
            and self.source_event_id is None
        ):
            raise ValueError("dense and boundary windows require a source candidate or event")
        return self


class MaterializedFrame(StrictModel):
    """One immutable, locally materialized frame selected by timestamp."""

    frame_id: OpaqueUuid
    ordinal: NonNegativeInt
    source_frame_index: NonNegativeInt
    target_timestamp_ns: Nanoseconds
    aligned_timestamp_ns: Nanoseconds
    source_timestamp_ns: Nanoseconds
    delta_to_target_ns: Nanoseconds
    artifact_uri: ArtifactUri
    artifact_sha256: Sha256Digest
    width: PositiveInt
    height: PositiveInt
    quality_flags: FrozenTuple[NonEmptyString] = ()

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        if self.aligned_timestamp_ns - self.target_timestamp_ns != self.delta_to_target_ns:
            raise ValueError(
                "delta_to_target_ns must equal aligned_timestamp_ns minus target_timestamp_ns"
            )
        return self


class SamplingSummary(StrictModel):
    """Target and actual counts for one camera within one package."""

    strategy: SamplingStrategy
    target_fps: PositiveFiniteFloat | None
    actual_fps: NonNegativeFiniteFloat
    target_count: NonNegativeInt
    actual_count: NonNegativeInt
    missed_targets: NonNegativeInt

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.strategy is SamplingStrategy.NOT_REQUESTED:
            if (
                self.target_fps is not None
                or self.actual_fps != 0
                or self.target_count != 0
                or self.actual_count != 0
                or self.missed_targets != 0
            ):
                raise ValueError("NOT_REQUESTED sampling requires null FPS and zero counts")
            return self
        if self.target_fps is None:
            raise ValueError("requested sampling requires target_fps")
        if self.target_count != self.actual_count + self.missed_targets:
            raise ValueError("target_count must equal actual_count plus missed_targets")
        if self.actual_count == 0 and self.actual_fps != 0:
            raise ValueError("zero selected frames require actual_fps zero")
        if self.actual_count > 0 and self.actual_fps <= 0:
            raise ValueError("selected frames require positive actual_fps")
        return self


class CameraPackage(StrictModel):
    """One canonical camera slot in a temporal package."""

    camera_id: CameraId
    status: CameraPackageStatus
    source_video_uri: ArtifactUri
    frames: FrozenTuple[MaterializedFrame]
    sampling: SamplingSummary
    missing_reason: NonEmptyString | None

    @model_validator(mode="after")
    def validate_camera_package(self) -> Self:
        if self.sampling.actual_count != len(self.frames):
            raise ValueError("sampling actual_count must equal the number of frames")
        if self.status is CameraPackageStatus.AVAILABLE:
            if not self.frames:
                raise ValueError("AVAILABLE camera packages require at least one frame")
            if self.missing_reason is not None:
                raise ValueError("AVAILABLE camera packages cannot have missing_reason")
        else:
            if self.frames:
                raise ValueError("non-AVAILABLE camera packages cannot publish frames")
            if self.missing_reason is None:
                raise ValueError("non-AVAILABLE camera packages require missing_reason")
        if self.status is CameraPackageStatus.NOT_REQUESTED:
            if self.sampling.strategy is not SamplingStrategy.NOT_REQUESTED:
                raise ValueError("NOT_REQUESTED camera status requires NOT_REQUESTED sampling")
        elif self.sampling.strategy is SamplingStrategy.NOT_REQUESTED:
            raise ValueError("NOT_REQUESTED sampling requires NOT_REQUESTED camera status")

        ordinals = tuple(frame.ordinal for frame in self.frames)
        if ordinals != tuple(range(len(self.frames))):
            raise ValueError("frame ordinals must be contiguous from zero")
        timestamps = tuple(frame.aligned_timestamp_ns for frame in self.frames)
        if timestamps != tuple(sorted(timestamps)):
            raise ValueError("frame aligned timestamps must be nondecreasing")
        return self


class TemporalVisualPackage(StrictModel):
    """Immutable provider-neutral evidence for one temporal window."""

    schema_version: Literal["1.0"]
    package_id: OpaqueUuid
    content_sha256: Sha256Digest
    mcap_id: OpaqueUuid
    window_id: OpaqueUuid
    purpose: SamplingPurpose
    interval: NanosecondInterval
    cameras: SixCameraMap[CameraPackage]
    frame_count_total: PositiveInt
    producer_version: SchemaVersion
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_package(self) -> Self:
        actual_count = 0
        for camera_id, camera in self.cameras.items():
            if camera.camera_id is not camera_id:
                raise ValueError("camera map keys must match nested camera_id values")
            actual_count += len(camera.frames)
            for frame in camera.frames:
                if not self.interval.contains(frame.aligned_timestamp_ns):
                    raise ValueError("every selected frame must lie in the package interval")
        if self.frame_count_total != actual_count:
            raise ValueError("frame_count_total must equal all camera frame counts")
        return self


class QAIssueClaim(StrictModel):
    """Provider-authored QA claim without authoritative lineage identifiers."""

    code: NonEmptyString
    interval: NanosecondInterval
    severity: QAIssueSeverity
    reported_score: UnitInterval | None


class CameraQAClaim(StrictModel):
    """One provider-authored camera QA observation."""

    camera_id: CameraId
    observed_interval: NanosecondInterval
    status: CameraQAStatus
    issues: FrozenTuple[QAIssueClaim]
    reported_score: UnitInterval | None
    frame_ordinals: FrozenTuple[NonNegativeInt]

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        _require_sorted_unique(self.frame_ordinals, "frame_ordinals")
        for issue in self.issues:
            if not _contains(self.observed_interval, issue.interval):
                raise ValueError("QA issue intervals must lie in observed_interval")
        return self


class QAOutput(StrictModel):
    """Six-camera provider QA output; the inference envelope owns persisted IDs."""

    cameras: SixCameraMap[CameraQAClaim]

    @model_validator(mode="after")
    def validate_camera_keys(self) -> Self:
        for camera_id, claim in self.cameras.items():
            if claim.camera_id is not camera_id:
                raise ValueError("camera map keys must match QA claim camera_id values")
        return self


class CameraQAResult(StrictModel):
    """Orchestrator-enriched QA result with authoritative package and frame lineage."""

    qa_result_id: OpaqueUuid
    mcap_id: OpaqueUuid
    package_id: OpaqueUuid
    inference_id: OpaqueUuid
    camera_id: CameraId
    claim: CameraQAClaim
    evidence_frame_ids: FrozenTuple[OpaqueUuid]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.claim.camera_id is not self.camera_id:
            raise ValueError("claim camera_id must match result camera_id")
        if len(self.evidence_frame_ids) != len(self.claim.frame_ordinals):
            raise ValueError("evidence_frame_ids must resolve every provider frame ordinal")
        if len(set(self.evidence_frame_ids)) != len(self.evidence_frame_ids):
            raise ValueError("evidence_frame_ids must be unique")
        return self


class QAResultAggregate(StrictModel):
    """Recording-level QA over exactly six enriched camera results."""

    aggregate_id: OpaqueUuid
    mcap_id: OpaqueUuid
    scope: NanosecondInterval
    overall_status: RecordingQAStatus
    usable_camera_count: NonNegativeInt
    camera_results: SixCameraMap[CameraQAResult]
    policy_version: SchemaVersion

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        usable = 0
        for camera_id, result in self.camera_results.items():
            if result.camera_id is not camera_id:
                raise ValueError("camera map keys must match QA result camera_id values")
            if result.mcap_id != self.mcap_id:
                raise ValueError("all camera QA results must belong to aggregate mcap_id")
            if not _contains(self.scope, result.claim.observed_interval):
                raise ValueError("camera QA observations must lie in aggregate scope")
            if result.claim.status in {CameraQAStatus.GOOD, CameraQAStatus.DEGRADED}:
                usable += 1
        if self.usable_camera_count != usable:
            raise ValueError("usable_camera_count must match GOOD and DEGRADED camera results")
        if self.usable_camera_count > 6:
            raise ValueError("usable_camera_count cannot exceed six")
        return self


class ProposalCameraClaim(StrictModel):
    """Provider-local proposal coverage for one canonical camera."""

    camera_id: CameraId
    status: CameraEvidenceStatus
    frame_ordinals: FrozenTuple[NonNegativeInt]

    @model_validator(mode="after")
    def validate_ordinals(self) -> Self:
        _require_sorted_unique(self.frame_ordinals, "frame_ordinals")
        if (
            self.status in {CameraEvidenceStatus.SUPPORTING, CameraEvidenceStatus.PARTIAL}
            and not self.frame_ordinals
        ):
            raise ValueError("supporting proposal claims require frame ordinals")
        if self.status is CameraEvidenceStatus.MISSING and self.frame_ordinals:
            raise ValueError("MISSING proposal claims cannot contain frame ordinals")
        return self


class EventProposal(StrictModel):
    """One provider-local event proposal; ordinal is scoped to its response."""

    ordinal: NonNegativeInt
    interval: NanosecondInterval
    label_hint: NonEmptyString | None
    reported_score: UnitInterval | None
    cameras: SixCameraMap[ProposalCameraClaim]

    @model_validator(mode="after")
    def validate_camera_keys(self) -> Self:
        for camera_id, claim in self.cameras.items():
            if claim.camera_id is not camera_id:
                raise ValueError("camera map keys must match proposal camera_id values")
        return self


class EventProposalOutput(StrictModel):
    """Zero or more provider-local event proposals."""

    proposals: FrozenTuple[EventProposal]

    @model_validator(mode="after")
    def validate_ordinals(self) -> Self:
        ordinals = tuple(proposal.ordinal for proposal in self.proposals)
        if ordinals != tuple(range(len(self.proposals))):
            raise ValueError("proposal ordinals must be contiguous from zero")
        return self


class CandidateEvent(StrictModel):
    """Orchestrator-owned candidate enriched from one provider-local proposal."""

    candidate_event_id: OpaqueUuid
    mcap_id: OpaqueUuid
    source_package_id: OpaqueUuid
    source_inference_id: OpaqueUuid
    proposal: EventProposal
    dense_interval: NanosecondInterval
    ontology_version: SchemaVersion
    status: CandidateEventStatus

    @model_validator(mode="after")
    def validate_dense_interval(self) -> Self:
        if not _contains(self.dense_interval, self.proposal.interval):
            raise ValueError("dense_interval must contain the proposal interval")
        return self


class CameraActionClaim(StrictModel):
    """Provider-local action evidence for one camera."""

    camera_id: CameraId
    status: CameraEvidenceStatus
    event_interval: NanosecondInterval | None
    observed_interval: NanosecondInterval | None
    visibility: UnitInterval | None
    observed_frame_count: NonNegativeInt
    coverage_fraction: UnitInterval
    reported_score: UnitInterval | None
    frame_ordinals: FrozenTuple[NonNegativeInt]
    reason: NonEmptyString | None

    @model_validator(mode="after")
    def validate_evidence_semantics(self) -> Self:
        _require_sorted_unique(self.frame_ordinals, "frame_ordinals")
        if self.observed_frame_count != len(self.frame_ordinals):
            raise ValueError("observed_frame_count must equal resolved frame ordinals")

        if self.status in {CameraEvidenceStatus.SUPPORTING, CameraEvidenceStatus.PARTIAL}:
            if self.event_interval is None or self.observed_interval is None:
                raise ValueError("supporting action evidence requires event and observed intervals")
            if not _contains(self.observed_interval, self.event_interval):
                raise ValueError("event_interval must lie in observed_interval")
            if self.observed_frame_count == 0 or self.coverage_fraction <= 0:
                raise ValueError("supporting action evidence requires positive observed coverage")
            return self

        if self.status is CameraEvidenceStatus.NO_EVENT:
            if self.event_interval is not None or self.observed_interval is None:
                raise ValueError("NO_EVENT requires a null event and nonnull observed interval")
            if self.observed_frame_count == 0 or self.coverage_fraction <= 0:
                raise ValueError("NO_EVENT requires positive observed coverage")
            return self

        if self.event_interval is not None:
            raise ValueError("non-observing action states require a null event_interval")
        if self.reason is None:
            raise ValueError("non-observing action states require a reason")
        if self.status is CameraEvidenceStatus.MISSING:
            if (
                self.observed_interval is not None
                or self.observed_frame_count != 0
                or self.coverage_fraction != 0
                or self.frame_ordinals
            ):
                raise ValueError("MISSING action evidence cannot claim observation coverage")
        elif self.observed_interval is None:
            raise ValueError("only MISSING action evidence may omit observed_interval")
        return self


class CrossViewHypothesis(StrictModel):
    """Provider-local fused hint without a persisted event identity."""

    ordinal: NonNegativeInt
    interval: NanosecondInterval
    action_type: NonEmptyString
    reported_score: UnitInterval | None


class ActionEvidence(StrictModel):
    """Provider action-evidence output with exactly six camera claims."""

    cameras: SixCameraMap[CameraActionClaim]
    cross_view_hypotheses: FrozenTuple[CrossViewHypothesis]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        for camera_id, claim in self.cameras.items():
            if claim.camera_id is not camera_id:
                raise ValueError("camera map keys must match action claim camera_id values")
        ordinals = tuple(hypothesis.ordinal for hypothesis in self.cross_view_hypotheses)
        if ordinals != tuple(range(len(ordinals))):
            raise ValueError("cross-view hypothesis ordinals must be contiguous from zero")
        return self


class BoundaryCameraClaim(StrictModel):
    """Provider-local onset and offset evidence for one camera."""

    camera_id: CameraId
    status: BoundaryStatus
    observed_interval: NanosecondInterval | None
    onset_interval: NanosecondInterval | None
    offset_interval: NanosecondInterval | None
    reported_score: UnitInterval | None
    frame_ordinals: FrozenTuple[NonNegativeInt]
    reason: NonEmptyString | None

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        _require_sorted_unique(self.frame_ordinals, "frame_ordinals")
        if self.status is BoundaryStatus.OBSERVED:
            if (
                self.observed_interval is None
                or self.onset_interval is None
                or self.offset_interval is None
            ):
                raise ValueError("OBSERVED boundary evidence requires all intervals")
            if not _contains(self.observed_interval, self.onset_interval) or not _contains(
                self.observed_interval, self.offset_interval
            ):
                raise ValueError("boundary intervals must lie in observed_interval")
            if not self.frame_ordinals:
                raise ValueError("OBSERVED boundary evidence requires frame ordinals")
            return self

        if self.onset_interval is not None or self.offset_interval is not None:
            raise ValueError("non-observed boundary states cannot claim onset or offset")
        if self.reason is None:
            raise ValueError("non-observed boundary states require a reason")
        if self.status is BoundaryStatus.MISSING:
            if self.observed_interval is not None or self.frame_ordinals:
                raise ValueError("MISSING boundary evidence cannot claim observations")
        elif self.observed_interval is None:
            raise ValueError("only MISSING boundary evidence may omit observed_interval")
        return self


class BoundaryRefinement(StrictModel):
    """Provider boundary-refinement output with exactly six camera claims."""

    cameras: SixCameraMap[BoundaryCameraClaim]

    @model_validator(mode="after")
    def validate_camera_keys(self) -> Self:
        for camera_id, claim in self.cameras.items():
            if claim.camera_id is not camera_id:
                raise ValueError("camera map keys must match boundary claim camera_id values")
        return self


class FusionHypothesisClaim(StrictModel):
    ordinal: NonNegativeInt
    interval: NanosecondInterval
    action_type: NonEmptyString
    conflict_codes: FrozenTuple[NonEmptyString]
    reported_score: UnitInterval | None


class FusionAdjudicationOutput(StrictModel):
    """Optional provider adjudication claim; local deterministic fusion remains authoritative."""

    hypotheses: FrozenTuple[FusionHypothesisClaim]
    abstained: bool

    @model_validator(mode="after")
    def validate_hypotheses(self) -> Self:
        ordinals = tuple(hypothesis.ordinal for hypothesis in self.hypotheses)
        if ordinals != tuple(range(len(ordinals))):
            raise ValueError("fusion hypothesis ordinals must be contiguous from zero")
        if self.abstained and self.hypotheses:
            raise ValueError("an abstained adjudication cannot contain hypotheses")
        if not self.abstained and not self.hypotheses:
            raise ValueError("a non-abstained adjudication requires hypotheses")
        return self


class InferenceCameraInput(StrictModel):
    """Authoritative request-catalog entry; providers receive only its local order."""

    camera_id: CameraId
    frame_ids: FrozenTuple[OpaqueUuid]

    @model_validator(mode="after")
    def validate_frame_ids(self) -> Self:
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("request catalog frame_ids must be unique")
        return self


class VisionInferenceRequest(StrictModel):
    """Provider-neutral invocation intent accepted by fake and real adapters."""

    schema_version: Literal["1.0"]
    inference_id: OpaqueUuid
    request_id: OpaqueUuid
    mcap_id: OpaqueUuid
    package_id: OpaqueUuid
    package_content_sha256: Sha256Digest
    interval: NanosecondInterval
    subject_candidate_id: OpaqueUuid | None
    task: VisionTask
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    prompt_version: SchemaVersion
    output_contract_version: SchemaVersion
    camera_inputs: SixCameraMap[InferenceCameraInput]
    timeout_ms: PositiveInt

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        for camera_id, item in self.camera_inputs.items():
            if item.camera_id is not camera_id:
                raise ValueError("camera map keys must match request catalog camera_id values")
        candidate_tasks = {
            VisionTask.ACTION_EVIDENCE,
            VisionTask.BOUNDARY_REFINEMENT,
            VisionTask.FUSION_ADJUDICATION,
        }
        if self.task in candidate_tasks and self.subject_candidate_id is None:
            raise ValueError("candidate-scoped tasks require subject_candidate_id")
        if self.task not in candidate_tasks and self.subject_candidate_id is not None:
            raise ValueError("recording/window tasks cannot carry subject_candidate_id")
        return self


class VisionUsage(StrictModel):
    input_frames: NonNegativeInt
    input_images: NonNegativeInt
    input_tokens: NonNegativeInt | None
    output_tokens: NonNegativeInt | None
    cost: NonNegativeFiniteFloat | None
    currency: NonEmptyString | None

    @model_validator(mode="after")
    def validate_cost(self) -> Self:
        if (self.cost is None) != (self.currency is None):
            raise ValueError("cost and currency must both be null or both be present")
        return self


type VisionProviderOutput = (
    QAOutput | EventProposalOutput | ActionEvidence | BoundaryRefinement | FusionAdjudicationOutput
)


class VisionInferenceSuccess(StrictModel):
    """Validated provider claim wrapped in orchestrator-owned attempt identity."""

    schema_version: Literal["1.0"]
    inference_id: OpaqueUuid
    request_id: OpaqueUuid
    task: VisionTask
    status: Literal[InferenceStatus.SUCCEEDED]
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    output: VisionProviderOutput
    raw_output_sha256: Sha256Digest
    schema_valid: Literal[True]
    usage: VisionUsage
    latency_ms: NonNegativeInt

    @model_validator(mode="after")
    def validate_output_type(self) -> Self:
        expected: dict[VisionTask, type[StrictModel]] = {
            VisionTask.QA_COARSE: QAOutput,
            VisionTask.QA_DENSE: QAOutput,
            VisionTask.EVENT_PROPOSAL: EventProposalOutput,
            VisionTask.ACTION_EVIDENCE: ActionEvidence,
            VisionTask.BOUNDARY_REFINEMENT: BoundaryRefinement,
            VisionTask.FUSION_ADJUDICATION: FusionAdjudicationOutput,
        }
        if not isinstance(self.output, expected[self.task]):
            raise ValueError("inference output type must match task")
        return self


class InferenceFailureDetail(StrictModel):
    code: NonEmptyString
    detail: NonEmptyString
    retryability: Retryability


class VisionInferenceFailure(StrictModel):
    """Terminal non-success outcome; it can never carry normalized output."""

    schema_version: Literal["1.0"]
    inference_id: OpaqueUuid
    request_id: OpaqueUuid
    task: VisionTask
    status: Literal[
        InferenceStatus.FAILED,
        InferenceStatus.TIMEOUT,
        InferenceStatus.CANCELLED,
        InferenceStatus.INVALID_OUTPUT,
    ]
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    output: None
    raw_output_sha256: Sha256Digest | None
    schema_valid: Literal[False]
    usage: VisionUsage
    latency_ms: NonNegativeInt
    failure: InferenceFailureDetail


VisionInferenceOutcome = Annotated[
    VisionInferenceSuccess | VisionInferenceFailure,
    Field(discriminator="status"),
]


class CameraEventProvenance(StrictModel):
    """Authoritative resolution of one provider camera claim into persisted lineage."""

    camera_id: CameraId
    claim: CameraActionClaim
    package_id: OpaqueUuid
    inference_id: OpaqueUuid
    frame_ids: FrozenTuple[OpaqueUuid]
    qa_result_id: OpaqueUuid

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if self.claim.camera_id is not self.camera_id:
            raise ValueError("claim camera_id must match provenance camera_id")
        if len(self.frame_ids) != len(self.claim.frame_ordinals):
            raise ValueError("frame_ids must resolve every provider frame ordinal")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("event provenance frame_ids must be unique")
        return self


class FusedActionEvent(StrictModel):
    """One development-only physical action with six explicit evidence records."""

    schema_version: Literal["1.0"]
    event_id: OpaqueUuid
    mcap_id: OpaqueUuid
    candidate_event_ids: FrozenTuple[OpaqueUuid]
    interval: NanosecondInterval
    action_type: NonEmptyString
    boundary_start_uncertainty_ns: Nanoseconds
    boundary_end_uncertainty_ns: Nanoseconds
    camera_evidence: SixCameraMap[CameraEventProvenance]
    fusion_policy_version: SchemaVersion
    boundary_inference_id: OpaqueUuid | None
    producer_provider: NonEmptyString
    status: ActionEventStatus
    production_eligible: Literal[False]
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if not self.candidate_event_ids:
            raise ValueError("fused events require at least one candidate_event_id")
        if len(set(self.candidate_event_ids)) != len(self.candidate_event_ids):
            raise ValueError("candidate_event_ids must be unique")
        if self.boundary_start_uncertainty_ns < 0 or self.boundary_end_uncertainty_ns < 0:
            raise ValueError("boundary uncertainty must be nonnegative")
        for camera_id, provenance in self.camera_evidence.items():
            if provenance.camera_id is not camera_id:
                raise ValueError("camera map keys must match event provenance camera_id values")
            claim_interval = provenance.claim.event_interval
            if claim_interval is not None and not _intersects(self.interval, claim_interval):
                raise ValueError("supporting camera intervals must intersect the fused event")
        return self


class StageReport(StrictModel):
    """Conserved work counts for one local mainline stage."""

    stage: MainlineStage
    status: StageStatus
    planned: NonNegativeInt
    succeeded: NonNegativeInt
    failed: NonNegativeInt
    pending: NonNegativeInt
    skipped: NonNegativeInt
    duration_ms: NonNegativeInt
    error_codes: FrozenTuple[NonEmptyString] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.planned != self.succeeded + self.failed + self.pending + self.skipped:
            raise ValueError("stage planned count must reconcile all outcomes")
        if self.status is StageStatus.SUCCEEDED and (self.failed or self.pending):
            raise ValueError("SUCCEEDED stages cannot contain failed or pending work")
        if self.status is StageStatus.FAILED and self.failed == 0:
            raise ValueError("FAILED stages require failed work")
        if self.status is StageStatus.INCOMPLETE and self.pending == 0:
            raise ValueError("INCOMPLETE stages require pending work")
        if self.status is StageStatus.SKIPPED and (
            self.skipped != self.planned or self.succeeded or self.failed or self.pending
        ):
            raise ValueError("SKIPPED stages require every planned unit to be skipped")
        return self


class MainlineRunReport(StrictModel):
    """Machine-readable local-run completion and accounting report."""

    schema_version: Literal["1.0"]
    run_id: OpaqueUuid
    source_mcap_id: OpaqueUuid
    source_recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    video_manifest_artifact_id: OpaqueUuid
    video_manifest_sha256: Sha256Digest
    video_manifest_semantic_sha256: Sha256Digest
    pipeline_version: SchemaVersion
    config_sha256: Sha256Digest
    status: RunStatus
    started_at: Rfc3339Timestamp
    completed_at: Rfc3339Timestamp | None
    duration_ms: NonNegativeInt
    stages: FrozenTuple[StageReport]
    window_count: NonNegativeInt
    package_count: NonNegativeInt
    inference_attempt_count: NonNegativeInt
    inference_success_count: NonNegativeInt
    inference_failure_count: NonNegativeInt
    inference_invalid_output_count: NonNegativeInt
    candidate_count: NonNegativeInt
    event_count: NonNegativeInt
    fake_inference_attempt_count: NonNegativeInt
    real_provider_request_count: Literal[0]
    error_codes: FrozenTuple[NonEmptyString] = ()

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if len({stage.stage for stage in self.stages}) != len(self.stages):
            raise ValueError("run stages must be unique")
        if self.inference_attempt_count != (
            self.inference_success_count
            + self.inference_failure_count
            + self.inference_invalid_output_count
        ):
            raise ValueError("inference attempt counts must reconcile")
        if self.fake_inference_attempt_count != self.inference_attempt_count:
            raise ValueError("V0 requires every inference attempt to use the fake adapter")

        complete = self.status in {
            RunStatus.PRIMARY_COMPLETE,
            RunStatus.PRIMARY_COMPLETE_NO_EVENTS,
        }
        if self.status is RunStatus.RUNNING:
            if self.completed_at is not None:
                raise ValueError("RUNNING reports cannot have completed_at")
        elif self.completed_at is None:
            raise ValueError("terminal run reports require completed_at")
        if complete and any(
            stage.status in {StageStatus.FAILED, StageStatus.INCOMPLETE} for stage in self.stages
        ):
            raise ValueError("complete runs cannot contain failed or incomplete stages")
        if complete and {stage.stage for stage in self.stages} != set(MainlineStage):
            raise ValueError("complete runs must report every mainline stage")
        if self.status is RunStatus.PRIMARY_COMPLETE and self.event_count == 0:
            raise ValueError("PRIMARY_COMPLETE requires at least one event")
        if self.status is RunStatus.PRIMARY_COMPLETE_NO_EVENTS and self.event_count != 0:
            raise ValueError("PRIMARY_COMPLETE_NO_EVENTS requires zero events")
        return self


class MainlineBundle(StrictModel):
    """Connected output bundle for one local end-to-end run."""

    schema_version: Literal["1.0"]
    report: MainlineRunReport
    windows: FrozenTuple[TemporalWindow]
    packages: FrozenTuple[TemporalVisualPackage]
    inference_requests: FrozenTuple[VisionInferenceRequest]
    inference_outcomes: FrozenTuple[VisionInferenceOutcome]
    qa_aggregates: FrozenTuple[QAResultAggregate]
    candidates: FrozenTuple[CandidateEvent]
    events: FrozenTuple[FusedActionEvent]

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        window_by_id = {window.window_id: window for window in self.windows}
        package_by_id = {package.package_id: package for package in self.packages}
        request_by_inference = {
            request.inference_id: request for request in self.inference_requests
        }
        candidate_by_id = {candidate.candidate_event_id: candidate for candidate in self.candidates}
        aggregate_by_id = {aggregate.aggregate_id: aggregate for aggregate in self.qa_aggregates}
        if len(window_by_id) != len(self.windows):
            raise ValueError("window IDs must be unique")
        if len(package_by_id) != len(self.packages):
            raise ValueError("package IDs must be unique")
        if len(request_by_inference) != len(self.inference_requests):
            raise ValueError("inference IDs must be unique")
        if len(candidate_by_id) != len(self.candidates):
            raise ValueError("candidate IDs must be unique")
        if len(aggregate_by_id) != len(self.qa_aggregates):
            raise ValueError("QA aggregate IDs must be unique")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("event IDs must be unique")

        source_mcap_id = self.report.source_mcap_id
        for window in self.windows:
            if window.mcap_id != source_mcap_id:
                raise ValueError("every window must belong to the report source MCAP")
        for package in self.packages:
            referenced_window = window_by_id.get(package.window_id)
            if referenced_window is None:
                raise ValueError("every package must reference a bundled window")
            if package.mcap_id != source_mcap_id or package.mcap_id != referenced_window.mcap_id:
                raise ValueError("package MCAP lineage must match its window and run")
            if (
                package.purpose is not referenced_window.purpose
                or package.interval != referenced_window.interval
            ):
                raise ValueError("package purpose and interval must match its window")

        for request in self.inference_requests:
            referenced_package = package_by_id.get(request.package_id)
            if referenced_package is None:
                raise ValueError("every inference request must reference a bundled package")
            if request.mcap_id != source_mcap_id:
                raise ValueError("inference request MCAP must match the run")
            if request.package_content_sha256 != referenced_package.content_sha256:
                raise ValueError("inference request package digest must match the package")
            if request.interval != referenced_package.interval:
                raise ValueError("inference request interval must match the package")
            for camera_id, camera_input in request.camera_inputs.items():
                package_frames = tuple(
                    frame.frame_id for frame in referenced_package.cameras[camera_id].frames
                )
                if camera_input.frame_ids != package_frames:
                    raise ValueError("request catalog must preserve package frame order")

        if len(self.inference_outcomes) != len(self.inference_requests):
            raise ValueError("every inference request requires exactly one terminal outcome")
        outcome_ids: set[str] = set()
        outcome_by_inference: dict[str, VisionInferenceOutcome] = {}
        for outcome in self.inference_outcomes:
            if outcome.inference_id in outcome_ids:
                raise ValueError("inference outcomes must be unique by inference_id")
            outcome_ids.add(outcome.inference_id)
            outcome_by_inference[outcome.inference_id] = outcome
            referenced_request = request_by_inference.get(outcome.inference_id)
            if referenced_request is None:
                raise ValueError("every inference outcome must reference a bundled request")
            if (
                outcome.request_id != referenced_request.request_id
                or outcome.task is not referenced_request.task
                or outcome.provider != referenced_request.provider
                or outcome.model_name != referenced_request.model_name
                or outcome.model_version != referenced_request.model_version
            ):
                raise ValueError("inference outcome identity must match its request")

        for candidate in self.candidates:
            source_request = request_by_inference.get(candidate.source_inference_id)
            source_package = package_by_id.get(candidate.source_package_id)
            source_outcome = outcome_by_inference.get(candidate.source_inference_id)
            if candidate.mcap_id != source_mcap_id:
                raise ValueError("candidate MCAP must match the run")
            if source_package is None:
                raise ValueError("candidate must reference a bundled package")
            if (
                source_request is None
                or source_request.task is not VisionTask.EVENT_PROPOSAL
                or source_request.package_id != candidate.source_package_id
            ):
                raise ValueError("candidate must reference an EVENT_PROPOSAL request")
            if (
                not isinstance(source_outcome, VisionInferenceSuccess)
                or not isinstance(source_outcome.output, EventProposalOutput)
                or candidate.proposal not in source_outcome.output.proposals
            ):
                raise ValueError("candidate proposal must be present in its successful outcome")
            if not _contains(source_package.interval, candidate.proposal.interval):
                raise ValueError("candidate proposal must lie in its source package")
            dense_windows = tuple(
                window
                for window in self.windows
                if window.source_candidate_id == candidate.candidate_event_id
            )
            if (
                len(dense_windows) != 1
                or dense_windows[0].interval != candidate.dense_interval
                or dense_windows[0].purpose is not SamplingPurpose.ACTION_DENSE
            ):
                raise ValueError("candidate must own exactly one matching dense action window")

        for event in self.events:
            if event.mcap_id != source_mcap_id:
                raise ValueError("event MCAP must match the run")
            if any(
                candidate_id not in candidate_by_id for candidate_id in event.candidate_event_ids
            ):
                raise ValueError("event candidates must be present in the bundle")
            action_inference_ids = {
                provenance.inference_id for provenance in event.camera_evidence.values()
            }
            evidence_package_ids = {
                provenance.package_id for provenance in event.camera_evidence.values()
            }
            if len(action_inference_ids) != 1 or len(evidence_package_ids) != 1:
                raise ValueError("one event must use one action inference and evidence package")
            action_inference_id = next(iter(action_inference_ids))
            evidence_package_id = next(iter(evidence_package_ids))
            action_request = request_by_inference.get(action_inference_id)
            action_outcome = outcome_by_inference.get(action_inference_id)
            evidence_package = package_by_id.get(evidence_package_id)
            if evidence_package is None:
                raise ValueError("event provenance must reference a bundled package")
            if (
                action_request is None
                or action_request.task is not VisionTask.ACTION_EVIDENCE
                or action_request.package_id != evidence_package_id
                or action_request.subject_candidate_id not in event.candidate_event_ids
            ):
                raise ValueError("event provenance must reference its candidate action request")
            if (
                not isinstance(action_outcome, VisionInferenceSuccess)
                or not isinstance(action_outcome.output, ActionEvidence)
                or action_outcome.provider != event.producer_provider
            ):
                raise ValueError("event provenance requires a matching successful action outcome")
            if not _contains(evidence_package.interval, event.interval):
                raise ValueError("fused event must lie in its evidence package interval")
            for camera_id, provenance in event.camera_evidence.items():
                claim = action_outcome.output.cameras[camera_id]
                if provenance.claim != claim:
                    raise ValueError("event claims must match the action outcome")
                if claim.observed_interval is not None and not _contains(
                    evidence_package.interval, claim.observed_interval
                ):
                    raise ValueError("action observations must lie in the evidence package")
                frames = evidence_package.cameras[camera_id].frames
                try:
                    expected_frame_ids = tuple(
                        frames[ordinal].frame_id for ordinal in claim.frame_ordinals
                    )
                except IndexError as error:
                    raise ValueError(
                        "event action claim references an absent package frame ordinal"
                    ) from error
                if provenance.frame_ids != expected_frame_ids:
                    raise ValueError("event frame IDs must resolve action claim ordinals")

            boundary_id = event.boundary_inference_id
            if boundary_id is None:
                raise ValueError("complete fused events require boundary inference lineage")
            boundary_request = request_by_inference.get(boundary_id)
            boundary_outcome = outcome_by_inference.get(boundary_id)
            boundary_output = (
                boundary_outcome.output
                if isinstance(boundary_outcome, VisionInferenceSuccess)
                else None
            )
            if (
                boundary_request is None
                or boundary_request.task is not VisionTask.BOUNDARY_REFINEMENT
                or boundary_request.package_id != evidence_package_id
                or boundary_request.subject_candidate_id not in event.candidate_event_ids
                or not isinstance(boundary_output, BoundaryRefinement)
            ):
                raise ValueError("event must reference its successful boundary refinement")
            for camera_id, boundary_claim in boundary_output.cameras.items():
                for interval in (
                    boundary_claim.observed_interval,
                    boundary_claim.onset_interval,
                    boundary_claim.offset_interval,
                ):
                    if interval is not None and not _contains(evidence_package.interval, interval):
                        raise ValueError("boundary evidence must lie in the evidence package")
                frame_count = len(evidence_package.cameras[camera_id].frames)
                if any(ordinal >= frame_count for ordinal in boundary_claim.frame_ordinals):
                    raise ValueError("boundary claim references an absent package frame ordinal")

        aggregate_inference_ids: set[str] = set()
        qa_result_by_id: dict[str, CameraQAResult] = {}
        for aggregate in self.qa_aggregates:
            if aggregate.mcap_id != source_mcap_id:
                raise ValueError("QA aggregate MCAP must match the run")
            package_ids = {result.package_id for result in aggregate.camera_results.values()}
            inference_ids = {result.inference_id for result in aggregate.camera_results.values()}
            if len(package_ids) != 1 or len(inference_ids) != 1:
                raise ValueError("each QA aggregate must describe one package and inference")
            package_id = next(iter(package_ids))
            inference_id = next(iter(inference_ids))
            qa_package = package_by_id.get(package_id)
            qa_request = request_by_inference.get(inference_id)
            if qa_package is None:
                raise ValueError("QA aggregate must match a bundled package")
            if aggregate.scope != qa_package.interval:
                raise ValueError("QA aggregate must match a bundled package and its interval")
            if qa_request is None:
                raise ValueError("QA aggregate must reference its bundled QA request")
            if qa_request.package_id != package_id or qa_request.task not in {
                VisionTask.QA_COARSE,
                VisionTask.QA_DENSE,
            }:
                raise ValueError("QA aggregate must reference its bundled QA request")
            if inference_id in aggregate_inference_ids:
                raise ValueError("each bundled QA request requires exactly one QA aggregate")
            for camera_id, result in aggregate.camera_results.items():
                if result.qa_result_id in qa_result_by_id:
                    raise ValueError("QA result IDs must be unique across aggregates")
                qa_result_by_id[result.qa_result_id] = result
                frames = qa_package.cameras[camera_id].frames
                try:
                    expected_frame_ids = tuple(
                        frames[ordinal].frame_id for ordinal in result.claim.frame_ordinals
                    )
                except IndexError as error:
                    raise ValueError(
                        "QA aggregate references an absent package frame ordinal"
                    ) from error
                if result.evidence_frame_ids != expected_frame_ids:
                    raise ValueError("QA evidence IDs must resolve package frame ordinals")
            aggregate_inference_ids.add(inference_id)

        qa_request_ids = {
            request.inference_id
            for request in self.inference_requests
            if request.task in {VisionTask.QA_COARSE, VisionTask.QA_DENSE}
        }
        if aggregate_inference_ids != qa_request_ids:
            raise ValueError("every bundled QA request requires exactly one QA aggregate")

        for event in self.events:
            for camera_id, provenance in event.camera_evidence.items():
                qa_result = qa_result_by_id.get(provenance.qa_result_id)
                if (
                    qa_result is None
                    or qa_result.camera_id is not camera_id
                    or qa_result.package_id != provenance.package_id
                ):
                    raise ValueError("event provenance must reference matching bundled QA evidence")

        succeeded = sum(
            outcome.status is InferenceStatus.SUCCEEDED for outcome in self.inference_outcomes
        )
        invalid = sum(
            outcome.status is InferenceStatus.INVALID_OUTPUT for outcome in self.inference_outcomes
        )
        failed = len(self.inference_outcomes) - succeeded - invalid
        expected_counts = (
            len(self.windows),
            len(self.packages),
            len(self.inference_requests),
            succeeded,
            failed,
            invalid,
            len(self.candidates),
            len(self.events),
        )
        report_counts = (
            self.report.window_count,
            self.report.package_count,
            self.report.inference_attempt_count,
            self.report.inference_success_count,
            self.report.inference_failure_count,
            self.report.inference_invalid_output_count,
            self.report.candidate_count,
            self.report.event_count,
        )
        if report_counts != expected_counts:
            raise ValueError("run report counts must match bundled records")
        return self


__all__ = [
    "ActionEventStatus",
    "ActionEvidence",
    "BoundaryCameraClaim",
    "BoundaryRefinement",
    "BoundaryStatus",
    "CameraActionClaim",
    "CameraEventProvenance",
    "CameraEvidenceStatus",
    "CameraPackage",
    "CameraPackageStatus",
    "CameraQAClaim",
    "CameraQAResult",
    "CameraQAStatus",
    "CandidateEvent",
    "CandidateEventStatus",
    "CrossViewHypothesis",
    "EventProposal",
    "EventProposalOutput",
    "FusedActionEvent",
    "FusionAdjudicationOutput",
    "FusionHypothesisClaim",
    "InferenceCameraInput",
    "InferenceFailureDetail",
    "InferenceStatus",
    "MainlineBundle",
    "MainlineRunReport",
    "MainlineStage",
    "MaterializedFrame",
    "ProposalCameraClaim",
    "QAIssueClaim",
    "QAIssueSeverity",
    "QAOutput",
    "QAResultAggregate",
    "RecordingQAStatus",
    "Retryability",
    "RunStatus",
    "SamplingPurpose",
    "SamplingStrategy",
    "SamplingSummary",
    "StageReport",
    "StageStatus",
    "TemporalVisualPackage",
    "TemporalWindow",
    "VisionInferenceFailure",
    "VisionInferenceOutcome",
    "VisionInferenceRequest",
    "VisionInferenceSuccess",
    "VisionProviderOutput",
    "VisionTask",
    "VisionUsage",
]
