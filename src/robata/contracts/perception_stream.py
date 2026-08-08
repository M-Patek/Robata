"""Durable contracts for the stream-oriented Mage perception boundary.

This module is deliberately additive to the published window contract family.  It
separates immutable non-overlapping storage segments, the causal inference context,
and action intervals.  Mage recurrent/KV state is never represented here: it is an
ephemeral acceleration and can always be rebuilt from the context manifest.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import (
    NanosecondInterval,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid, Rfc3339Timestamp

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
CanonicalToken = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    ),
]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]

PERCEPTION_CONTEXT_PROJECTION_VERSION: Final = "perception-context-semantic-v1"
PERCEPTION_CONTEXT_IDENTITY_POLICY_VERSION: Final = "perception-context-identity-v1"
PERCEPTION_CONTEXT_KEY_NAMESPACE: Final = "perception-context-v1"
MAGE_OBSERVATION_SCHEMA_VERSION: Final = "1.0"
MAGE_OBSERVATION_PROJECTION_VERSION: Final = "mage-observation-semantic-v1"
MAGE_OBSERVATION_IDENTITY_POLICY_VERSION: Final = "mage-observation-identity-v1"
MAGE_OBSERVATION_KEY_NAMESPACE: Final = "mage-observation-v1"
MAGE_OBSERVATION_UUID_NAMESPACE: Final = "robata:mage-observation-v1"
REFINE_REQUEST_PROJECTION_VERSION: Final = "perception-refine-request-semantic-v1"
REFINE_REQUEST_IDENTITY_POLICY_VERSION: Final = "perception-refine-request-identity-v1"
REFINE_REQUEST_KEY_NAMESPACE: Final = "perception-refine-request-v1"
REFINE_REQUEST_UUID_NAMESPACE: Final = "robata:perception-refine-request-v1"


class CameraAbsenceReason(StrEnum):
    """Why one canonical camera cannot contribute to the context."""

    MISSING = "MISSING"
    LATE = "LATE"
    DECODE_ERROR = "DECODE_ERROR"
    TIMESTAMP_GAP = "TIMESTAMP_GAP"
    CORRUPT = "CORRUPT"
    UNAVAILABLE = "UNAVAILABLE"


class CameraEvidenceRelation(StrEnum):
    """A camera-local relation to an action observation."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    INDETERMINATE = "INDETERMINATE"
    NOT_OBSERVABLE = "NOT_OBSERVABLE"


class SemanticQaDisposition(StrEnum):
    """Mage semantic assessment, separate from deterministic media health."""

    USABLE = "USABLE"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"
    UNKNOWN = "UNKNOWN"


class RefineReason(StrEnum):
    """Narrow reasons that may trigger the exceptional perception path."""

    BOUNDARY = "BOUNDARY"
    LABEL = "LABEL"
    CONFLICT = "CONFLICT"
    QA = "QA"


class RefineTargetField(StrEnum):
    """Fields a refinement call is permitted to author."""

    START_BOUNDARY = "START_BOUNDARY"
    END_BOUNDARY = "END_BOUNDARY"
    ACTION_LABEL = "ACTION_LABEL"
    CAMERA_RELATION = "CAMERA_RELATION"
    SEMANTIC_QA = "SEMANTIC_QA"


class StorageSegmentReference(StrictModel):
    """Reference to one immutable, non-overlapping storage segment manifest."""

    segment_ordinal: NonNegativeInt
    segment_key: NodeLogicalKey
    segment_semantic_sha256: Sha256Digest
    interval: NanosecondInterval

    @model_validator(mode="after")
    def validate_key_binding(self) -> Self:
        if not self.segment_key.endswith(f":{self.segment_semantic_sha256}"):
            raise ValueError("segment_key must end with segment_semantic_sha256")
        return self


class CameraContextBinding(StrictModel):
    """Durable camera input facts; model features and hidden tensors are excluded."""

    camera_id: CameraId
    available: bool
    selected_for_inference: bool
    codec_stream_exact_sha256: Sha256Digest | None = None
    segment_semantic_sha256_values: tuple[Sha256Digest, ...] = ()
    absence_reason: CameraAbsenceReason | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.available:
            if self.absence_reason is not None:
                raise ValueError("available camera cannot declare an absence reason")
            if self.codec_stream_exact_sha256 is None:
                raise ValueError("available camera requires codec_stream_exact_sha256")
            if not self.segment_semantic_sha256_values:
                raise ValueError("available camera requires segment lineage")
        else:
            if self.selected_for_inference:
                raise ValueError("unavailable camera cannot be selected for inference")
            if self.codec_stream_exact_sha256 is not None:
                raise ValueError("unavailable camera cannot bind codec bytes")
            if self.segment_semantic_sha256_values:
                raise ValueError("unavailable camera cannot bind segment lineage")
            if self.absence_reason is None:
                raise ValueError("unavailable camera requires an absence reason")
        return self


class PerceptionContextManifest(StrictModel):
    """Replayable causal context for one stream-oriented perception call."""

    schema_version: Literal["1.0"] = "1.0"
    projection_version: Literal["perception-context-semantic-v1"] = (
        PERCEPTION_CONTEXT_PROJECTION_VERSION
    )
    identity_policy_version: Literal["perception-context-identity-v1"] = (
        PERCEPTION_CONTEXT_IDENTITY_POLICY_VERSION
    )
    source_recording_key: NonEmptyString
    source_recording_exact_sha256: Sha256Digest
    context_interval: NanosecondInterval
    ordered_segments: tuple[StorageSegmentReference, ...]
    focus_segment_ordinal: NonNegativeInt
    cameras: SixCameraMap[CameraContextBinding]
    codec_policy_version: SchemaVersion
    context_policy_version: SchemaVersion
    context_manifest_key: NodeLogicalKey
    context_manifest_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if not self.ordered_segments:
            raise ValueError("perception context requires at least one storage segment")
        previous: StorageSegmentReference | None = None
        ordinals: set[int] = set()
        segment_digests = tuple(item.segment_semantic_sha256 for item in self.ordered_segments)
        for segment in self.ordered_segments:
            if segment.segment_ordinal in ordinals:
                raise ValueError("storage segment ordinals must be unique")
            ordinals.add(segment.segment_ordinal)
            if previous is not None:
                if segment.segment_ordinal <= previous.segment_ordinal:
                    raise ValueError("storage segments must be ordered by ordinal")
                if segment.interval.start_ns < previous.interval.end_ns:
                    raise ValueError("storage segments must not overlap")
            if (
                segment.interval.start_ns < self.context_interval.start_ns
                or segment.interval.end_ns > self.context_interval.end_ns
            ):
                raise ValueError("storage segment must be inside the inference context")
            previous = segment

        if self.focus_segment_ordinal != self.ordered_segments[-1].segment_ordinal:
            raise ValueError("focus segment must be the newest causal context segment")

        selected = 0
        for camera_id in CAMERA_IDS:
            binding = self.cameras[camera_id]
            if binding.camera_id is not camera_id:
                raise ValueError("camera context key must match camera_id")
            if binding.selected_for_inference:
                selected += 1
            if binding.available and binding.segment_semantic_sha256_values != segment_digests:
                raise ValueError("available camera lineage must match the ordered segment manifest")
        if selected == 0:
            raise ValueError("perception context requires at least one selected camera")

        digest = perception_context_semantic_sha256(self)
        if (
            self.context_manifest_semantic_sha256 != digest
            or self.context_manifest_key != f"{PERCEPTION_CONTEXT_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("perception context identity does not match its semantic projection")
        return self


class CognitionGateSignal(StrictModel):
    """Shadow-only gate evidence; it cannot suppress perception in v1."""

    mode: Literal["SHADOW_ONLY"] = "SHADOW_ONLY"
    score: UnitInterval | None
    threshold: UnitInterval
    would_admit: bool | None
    gate_policy_version: SchemaVersion

    @model_validator(mode="after")
    def validate_shadow_decision(self) -> Self:
        if self.score is None:
            if self.would_admit is not None:
                raise ValueError("unscored gate signal cannot declare would_admit")
        elif self.would_admit is not (self.score >= self.threshold):
            raise ValueError("would_admit must be derived from score and threshold")
        return self


class SemanticQaIssue(StrictModel):
    """One normalized semantic QA issue emitted by Mage."""

    code: CanonicalToken
    detail: NonEmptyString | None = None


class SemanticCameraQa(StrictModel):
    """Camera-local semantic QA from the single perception generation."""

    camera_id: CameraId
    disposition: SemanticQaDisposition
    issues: tuple[SemanticQaIssue, ...] = ()
    confidence: UnitInterval | None = None

    @model_validator(mode="after")
    def validate_issues(self) -> Self:
        codes = tuple(issue.code for issue in self.issues)
        if codes != tuple(sorted(set(codes))):
            raise ValueError("semantic QA issues must be unique and ordered by code")
        if self.disposition is SemanticQaDisposition.USABLE and self.issues:
            raise ValueError("usable semantic QA cannot contain issues")
        return self


class ActorObservation(StrictModel):
    """Optional actor attributes reported by perception."""

    hand: CanonicalToken | None = None
    actor_type: CanonicalToken | None = None


class ObjectObservation(StrictModel):
    """Optional object attributes; identity hints remain hypotheses, not facts."""

    object_type: CanonicalToken
    identity_hint: NonEmptyString | None = None


class BoundaryAssessment(StrictModel):
    """Boundary certainty and continuation state for a clipped context observation."""

    start_confidence: UnitInterval
    end_confidence: UnitInterval
    started_before_context: bool = False
    continues_after_context: bool = False


class CameraObservationEvidence(StrictModel):
    """One explicit six-slot camera relation to an action observation."""

    camera_id: CameraId
    relation: CameraEvidenceRelation
    visibility: UnitInterval | None = None
    observed_interval: NanosecondInterval | None = None
    evidence_semantic_sha256_values: tuple[Sha256Digest, ...] = ()

    @model_validator(mode="after")
    def validate_relation(self) -> Self:
        if self.relation is CameraEvidenceRelation.NOT_OBSERVABLE:
            if self.visibility is not None or self.observed_interval is not None:
                raise ValueError("not-observable evidence cannot claim visibility or an interval")
            if self.evidence_semantic_sha256_values:
                raise ValueError("not-observable evidence cannot cite semantic evidence")
        elif self.observed_interval is None:
            raise ValueError("observable camera evidence requires an observed interval")
        if len(set(self.evidence_semantic_sha256_values)) != len(
            self.evidence_semantic_sha256_values
        ):
            raise ValueError("camera evidence digests must be unique")
        return self


class MageActionObservation(StrictModel):
    """What Mage observed, before Robata projects any business fact."""

    local_ref: NonEmptyString
    action: CanonicalToken
    interval: NanosecondInterval
    confidence: UnitInterval | None = None
    actor: ActorObservation | None = None
    object: ObjectObservation | None = None
    camera_evidence: SixCameraMap[CameraObservationEvidence]
    boundary: BoundaryAssessment

    @model_validator(mode="after")
    def validate_camera_ids(self) -> Self:
        for camera_id in CAMERA_IDS:
            if self.camera_evidence[camera_id].camera_id is not camera_id:
                raise ValueError("camera evidence key must match camera_id")
        return self


class MageObservation(StrictModel):
    """One durable, parsed Mage perception artifact for one causal context."""

    schema_version: Literal["1.0"] = MAGE_OBSERVATION_SCHEMA_VERSION
    projection_version: Literal["mage-observation-semantic-v1"] = (
        MAGE_OBSERVATION_PROJECTION_VERSION
    )
    identity_policy_version: Literal["mage-observation-identity-v1"] = (
        MAGE_OBSERVATION_IDENTITY_POLICY_VERSION
    )
    observation_schema_version: SchemaVersion
    observation_id: OpaqueUuid
    observation_logical_key: NodeLogicalKey
    observation_identity_sha256: Sha256Digest
    observation_semantic_sha256: Sha256Digest
    context: PerceptionContextManifest
    model_family: CanonicalToken
    model_revision: NonEmptyString
    model_artifact_manifest_sha256: Sha256Digest
    prompt_version: SchemaVersion
    inference_artifact_exact_sha256: Sha256Digest
    cognition_gate: CognitionGateSignal
    semantic_qa: SixCameraMap[SemanticCameraQa]
    observations: tuple[MageActionObservation, ...]
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        local_refs: set[str] = set()
        for camera_id in CAMERA_IDS:
            qa = self.semantic_qa[camera_id]
            context_camera = self.context.cameras[camera_id]
            if qa.camera_id is not camera_id:
                raise ValueError("semantic QA key must match camera_id")
            if (
                not context_camera.available or not context_camera.selected_for_inference
            ) and qa.disposition is not SemanticQaDisposition.UNKNOWN:
                raise ValueError("unavailable or unselected cameras must have UNKNOWN semantic QA")

        for observation in self.observations:
            if observation.local_ref in local_refs:
                raise ValueError("Mage observation local_ref values must be unique")
            local_refs.add(observation.local_ref)
            if (
                observation.interval.start_ns < self.context.context_interval.start_ns
                or observation.interval.end_ns > self.context.context_interval.end_ns
            ):
                raise ValueError("action observation must be inside the inference context")
            if (
                observation.boundary.started_before_context
                and observation.interval.start_ns != self.context.context_interval.start_ns
            ):
                raise ValueError("continued observation must start at the context boundary")
            if (
                observation.boundary.continues_after_context
                and observation.interval.end_ns != self.context.context_interval.end_ns
            ):
                raise ValueError("continuing observation must end at the context boundary")
            for camera_id in CAMERA_IDS:
                camera = self.context.cameras[camera_id]
                evidence = observation.camera_evidence[camera_id]
                if (
                    not camera.available or not camera.selected_for_inference
                ) and evidence.relation is not CameraEvidenceRelation.NOT_OBSERVABLE:
                    raise ValueError("unavailable or unselected camera must be NOT_OBSERVABLE")
                if evidence.observed_interval is not None and (
                    evidence.observed_interval.start_ns < self.context.context_interval.start_ns
                    or evidence.observed_interval.end_ns > self.context.context_interval.end_ns
                ):
                    raise ValueError("camera evidence interval must be inside the context")

        identity_digest = mage_observation_identity_sha256(self)
        semantic_digest = mage_observation_semantic_sha256(self)
        if (
            self.observation_identity_sha256 != identity_digest
            or self.observation_logical_key != f"{MAGE_OBSERVATION_KEY_NAMESPACE}:{identity_digest}"
            or self.observation_id
            != str(uuid5(NAMESPACE_URL, f"{MAGE_OBSERVATION_UUID_NAMESPACE}:{identity_digest}"))
        ):
            raise ValueError("Mage observation logical identity is inconsistent")
        if self.observation_semantic_sha256 != semantic_digest:
            raise ValueError("Mage observation semantic digest is inconsistent")
        return self


class PerceptionRefineRequest(StrictModel):
    """A bounded, single-purpose request for exceptional Mage refinement."""

    schema_version: Literal["1.0"] = "1.0"
    projection_version: Literal["perception-refine-request-semantic-v1"] = (
        REFINE_REQUEST_PROJECTION_VERSION
    )
    identity_policy_version: Literal["perception-refine-request-identity-v1"] = (
        REFINE_REQUEST_IDENTITY_POLICY_VERSION
    )
    refine_request_id: OpaqueUuid
    refine_request_key: NodeLogicalKey
    refine_request_semantic_sha256: Sha256Digest
    source_observation_logical_key: NodeLogicalKey
    source_observation_semantic_sha256: Sha256Digest
    target_hypothesis_logical_key: NodeLogicalKey
    target_hypothesis_semantic_sha256: Sha256Digest
    reason: RefineReason
    target_fields: tuple[RefineTargetField, ...]
    refine_interval: NanosecondInterval
    refine_policy_version: SchemaVersion
    prompt_version: SchemaVersion

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not self.target_fields:
            raise ValueError("refinement request requires at least one target field")
        if self.target_fields != tuple(
            sorted(set(self.target_fields), key=lambda item: item.value)
        ):
            raise ValueError("refinement target fields must be unique and canonically ordered")
        digest = perception_refine_request_semantic_sha256(self)
        if (
            self.refine_request_semantic_sha256 != digest
            or self.refine_request_key != f"{REFINE_REQUEST_KEY_NAMESPACE}:{digest}"
            or self.refine_request_id
            != str(uuid5(NAMESPACE_URL, f"{REFINE_REQUEST_UUID_NAMESPACE}:{digest}"))
        ):
            raise ValueError("refinement request identity is inconsistent")
        return self


def perception_context_semantic_projection(
    context: PerceptionContextManifest,
) -> dict[str, object]:
    return {
        "projection_version": context.projection_version,
        "identity_policy_version": context.identity_policy_version,
        "source_recording_key": context.source_recording_key,
        "source_recording_exact_sha256": context.source_recording_exact_sha256,
        "context_interval": context.context_interval.model_dump(mode="json"),
        "ordered_segments": [
            segment.model_dump(mode="json") for segment in context.ordered_segments
        ],
        "focus_segment_ordinal": context.focus_segment_ordinal,
        "cameras": context.cameras.model_dump(mode="json"),
        "codec_policy_version": context.codec_policy_version,
        "context_policy_version": context.context_policy_version,
    }


def perception_context_semantic_sha256(context: PerceptionContextManifest) -> Sha256Digest:
    return semantic_sha256(perception_context_semantic_projection(context))


def create_perception_context_manifest(
    *,
    source_recording_key: str,
    source_recording_exact_sha256: Sha256Digest,
    context_interval: NanosecondInterval,
    ordered_segments: tuple[StorageSegmentReference, ...],
    focus_segment_ordinal: int,
    cameras: SixCameraMap[CameraContextBinding],
    codec_policy_version: str,
    context_policy_version: str,
) -> PerceptionContextManifest:
    values = {
        "source_recording_key": source_recording_key,
        "source_recording_exact_sha256": source_recording_exact_sha256,
        "context_interval": context_interval,
        "ordered_segments": ordered_segments,
        "focus_segment_ordinal": focus_segment_ordinal,
        "cameras": cameras,
        "codec_policy_version": codec_policy_version,
        "context_policy_version": context_policy_version,
    }
    draft = PerceptionContextManifest.model_construct(
        context_manifest_key=f"{PERCEPTION_CONTEXT_KEY_NAMESPACE}:{'0' * 64}",
        context_manifest_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = perception_context_semantic_sha256(draft)
    return PerceptionContextManifest(
        context_manifest_key=f"{PERCEPTION_CONTEXT_KEY_NAMESPACE}:{digest}",
        context_manifest_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


def mage_observation_identity_projection(observation: MageObservation) -> dict[str, object]:
    """Logical request identity, independent from attempt/runtime timestamps."""

    return {
        "identity_policy_version": observation.identity_policy_version,
        "source_recording_key": observation.context.source_recording_key,
        "source_recording_exact_sha256": observation.context.source_recording_exact_sha256,
        "ordered_segment_semantic_sha256_values": [
            item.segment_semantic_sha256 for item in observation.context.ordered_segments
        ],
        "camera_context": observation.context.cameras.model_dump(mode="json"),
        "context_manifest_semantic_sha256": observation.context.context_manifest_semantic_sha256,
        "model_family": observation.model_family,
        "model_revision": observation.model_revision,
        "model_artifact_manifest_sha256": observation.model_artifact_manifest_sha256,
        "codec_policy_version": observation.context.codec_policy_version,
        "context_policy_version": observation.context.context_policy_version,
        "prompt_version": observation.prompt_version,
        "observation_schema_version": observation.observation_schema_version,
    }


def mage_observation_identity_sha256(observation: MageObservation) -> Sha256Digest:
    return semantic_sha256(mage_observation_identity_projection(observation))


def mage_observation_semantic_projection(observation: MageObservation) -> dict[str, object]:
    """Parsed artifact identity; artifact replay of this projection is byte-stable."""

    return {
        "projection_version": observation.projection_version,
        "observation_identity_sha256": observation.observation_identity_sha256,
        "inference_artifact_exact_sha256": observation.inference_artifact_exact_sha256,
        "cognition_gate": observation.cognition_gate.model_dump(mode="json"),
        "semantic_qa": observation.semantic_qa.model_dump(mode="json"),
        "observations": [item.model_dump(mode="json") for item in observation.observations],
    }


def mage_observation_semantic_sha256(observation: MageObservation) -> Sha256Digest:
    return semantic_sha256(mage_observation_semantic_projection(observation))


def create_mage_observation(
    *,
    observation_schema_version: str,
    context: PerceptionContextManifest,
    model_family: str,
    model_revision: str,
    model_artifact_manifest_sha256: Sha256Digest,
    prompt_version: str,
    inference_artifact_exact_sha256: Sha256Digest,
    cognition_gate: CognitionGateSignal,
    semantic_qa: SixCameraMap[SemanticCameraQa],
    observations: tuple[MageActionObservation, ...],
    created_at: str,
) -> MageObservation:
    values = {
        "observation_schema_version": observation_schema_version,
        "context": context,
        "model_family": model_family,
        "model_revision": model_revision,
        "model_artifact_manifest_sha256": model_artifact_manifest_sha256,
        "prompt_version": prompt_version,
        "inference_artifact_exact_sha256": inference_artifact_exact_sha256,
        "cognition_gate": cognition_gate,
        "semantic_qa": semantic_qa,
        "observations": observations,
        "created_at": created_at,
    }
    draft = MageObservation.model_construct(
        observation_id="00000000-0000-0000-0000-000000000000",
        observation_logical_key=f"{MAGE_OBSERVATION_KEY_NAMESPACE}:{'0' * 64}",
        observation_identity_sha256="0" * 64,
        observation_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    identity_digest = mage_observation_identity_sha256(draft)
    semantic_draft = draft.model_copy(update={"observation_identity_sha256": identity_digest})
    semantic_digest = mage_observation_semantic_sha256(semantic_draft)
    return MageObservation(
        observation_id=str(
            uuid5(NAMESPACE_URL, f"{MAGE_OBSERVATION_UUID_NAMESPACE}:{identity_digest}")
        ),
        observation_logical_key=f"{MAGE_OBSERVATION_KEY_NAMESPACE}:{identity_digest}",
        observation_identity_sha256=identity_digest,
        observation_semantic_sha256=semantic_digest,
        **cast(dict[str, Any], values),
    )


def perception_refine_request_semantic_projection(
    request: PerceptionRefineRequest,
) -> dict[str, object]:
    return {
        "projection_version": request.projection_version,
        "identity_policy_version": request.identity_policy_version,
        "source_observation_logical_key": request.source_observation_logical_key,
        "source_observation_semantic_sha256": request.source_observation_semantic_sha256,
        "target_hypothesis_logical_key": request.target_hypothesis_logical_key,
        "target_hypothesis_semantic_sha256": request.target_hypothesis_semantic_sha256,
        "reason": request.reason.value,
        "target_fields": [item.value for item in request.target_fields],
        "refine_interval": request.refine_interval.model_dump(mode="json"),
        "refine_policy_version": request.refine_policy_version,
        "prompt_version": request.prompt_version,
    }


def perception_refine_request_semantic_sha256(
    request: PerceptionRefineRequest,
) -> Sha256Digest:
    return semantic_sha256(perception_refine_request_semantic_projection(request))


def create_perception_refine_request(
    *,
    source_observation_logical_key: NodeLogicalKey,
    source_observation_semantic_sha256: Sha256Digest,
    target_hypothesis_logical_key: NodeLogicalKey,
    target_hypothesis_semantic_sha256: Sha256Digest,
    reason: RefineReason,
    target_fields: tuple[RefineTargetField, ...],
    refine_interval: NanosecondInterval,
    refine_policy_version: str,
    prompt_version: str,
) -> PerceptionRefineRequest:
    canonical_fields = tuple(sorted(set(target_fields), key=lambda item: item.value))
    values = {
        "source_observation_logical_key": source_observation_logical_key,
        "source_observation_semantic_sha256": source_observation_semantic_sha256,
        "target_hypothesis_logical_key": target_hypothesis_logical_key,
        "target_hypothesis_semantic_sha256": target_hypothesis_semantic_sha256,
        "reason": reason,
        "target_fields": canonical_fields,
        "refine_interval": refine_interval,
        "refine_policy_version": refine_policy_version,
        "prompt_version": prompt_version,
    }
    draft = PerceptionRefineRequest.model_construct(
        refine_request_id="00000000-0000-0000-0000-000000000000",
        refine_request_key=f"{REFINE_REQUEST_KEY_NAMESPACE}:{'0' * 64}",
        refine_request_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = perception_refine_request_semantic_sha256(draft)
    return PerceptionRefineRequest(
        refine_request_id=str(uuid5(NAMESPACE_URL, f"{REFINE_REQUEST_UUID_NAMESPACE}:{digest}")),
        refine_request_key=f"{REFINE_REQUEST_KEY_NAMESPACE}:{digest}",
        refine_request_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


__all__ = [
    "ActorObservation",
    "BoundaryAssessment",
    "CameraAbsenceReason",
    "CameraContextBinding",
    "CameraEvidenceRelation",
    "CameraObservationEvidence",
    "CognitionGateSignal",
    "MageActionObservation",
    "MageObservation",
    "ObjectObservation",
    "PerceptionContextManifest",
    "PerceptionRefineRequest",
    "RefineReason",
    "RefineTargetField",
    "SemanticCameraQa",
    "SemanticQaDisposition",
    "SemanticQaIssue",
    "StorageSegmentReference",
    "create_mage_observation",
    "create_perception_context_manifest",
    "create_perception_refine_request",
    "mage_observation_identity_projection",
    "mage_observation_identity_sha256",
    "mage_observation_semantic_projection",
    "mage_observation_semantic_sha256",
    "perception_context_semantic_projection",
    "perception_context_semantic_sha256",
    "perception_refine_request_semantic_projection",
    "perception_refine_request_semantic_sha256",
]
