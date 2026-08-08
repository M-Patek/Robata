"""Deterministic business projections from one durable Mage observation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final, Literal, Self, cast

from pydantic import model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey
from robata.contracts.perception_stream import (
    ActorObservation,
    CameraEvidenceRelation,
    MageActionObservation,
    MageObservation,
    NonEmptyString,
    ObjectObservation,
    SemanticQaDisposition,
    UnitInterval,
)

MEDIA_HEALTH_PROJECTION_VERSION: Final = "media-health-semantic-v1"
MEDIA_HEALTH_KEY_NAMESPACE: Final = "media-health-v1"
QA_PROJECTOR_VERSION: Final = "mage-observation-qa-projector-v1"
QA_PROJECTION_KEY_NAMESPACE: Final = "observation-qa-projection-v1"
EVENT_PROJECTOR_VERSION: Final = "mage-observation-event-projector-v1"
EVENT_HYPOTHESIS_KEY_NAMESPACE: Final = "event-hypothesis-v1"
EVENT_PROJECTION_KEY_NAMESPACE: Final = "observation-event-projection-v1"
EVIDENCE_PROJECTOR_VERSION: Final = "mage-observation-evidence-projector-v1"
EVIDENCE_FACT_KEY_NAMESPACE: Final = "camera-evidence-fact-v1"
EVIDENCE_PROJECTION_KEY_NAMESPACE: Final = "observation-evidence-projection-v1"


class MediaHealthDisposition(StrEnum):
    """Deterministic media status, computed without a VLM."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"
    UNAVAILABLE = "UNAVAILABLE"


class ProjectedQaDisposition(StrEnum):
    """Fused QA disposition used by downstream deterministic policy."""

    USABLE = "USABLE"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class MediaHealthCameraFact(StrictModel):
    """Cheap camera health facts such as decode, gap, black, or freeze issues."""

    camera_id: CameraId
    disposition: MediaHealthDisposition
    issue_codes: tuple[NonEmptyString, ...] = ()
    observed_interval: NanosecondInterval | None = None

    @model_validator(mode="after")
    def validate_fact(self) -> Self:
        if self.issue_codes != tuple(sorted(set(self.issue_codes))):
            raise ValueError("media health issue codes must be unique and ordered")
        if self.disposition is MediaHealthDisposition.HEALTHY and self.issue_codes:
            raise ValueError("healthy media cannot contain issue codes")
        if self.disposition is MediaHealthDisposition.UNAVAILABLE:
            if self.observed_interval is not None:
                raise ValueError("unavailable media cannot have an observed interval")
            if not self.issue_codes:
                raise ValueError("unavailable media requires an issue code")
        elif self.observed_interval is None:
            raise ValueError("available media health requires an observed interval")
        return self


class MediaHealthReport(StrictModel):
    """Deterministic six-camera report bound to one perception context."""

    schema_version: Literal["1.0"] = "1.0"
    projection_version: Literal["media-health-semantic-v1"] = MEDIA_HEALTH_PROJECTION_VERSION
    context_manifest_semantic_sha256: Sha256Digest
    policy_version: SchemaVersion
    cameras: SixCameraMap[MediaHealthCameraFact]
    media_health_key: NodeLogicalKey
    media_health_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        for camera_id in CAMERA_IDS:
            if self.cameras[camera_id].camera_id is not camera_id:
                raise ValueError("media health camera key must match camera_id")
        digest = media_health_semantic_sha256(self)
        if (
            self.media_health_semantic_sha256 != digest
            or self.media_health_key != f"{MEDIA_HEALTH_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("media health identity is inconsistent")
        return self


class ProjectedCameraQaFact(StrictModel):
    """One deterministic QA fact derived from media and semantic QA."""

    camera_id: CameraId
    disposition: ProjectedQaDisposition
    media_health_disposition: MediaHealthDisposition
    semantic_qa_disposition: SemanticQaDisposition
    issue_codes: tuple[NonEmptyString, ...]
    semantic_confidence: UnitInterval | None = None


class QaProjection(StrictModel):
    """The logical QA stage retained without a second VLM invocation."""

    schema_version: Literal["1.0"] = "1.0"
    projector_version: Literal["mage-observation-qa-projector-v1"] = QA_PROJECTOR_VERSION
    source_observation_logical_key: NodeLogicalKey
    source_observation_semantic_sha256: Sha256Digest
    source_media_health_semantic_sha256: Sha256Digest
    camera_facts: SixCameraMap[ProjectedCameraQaFact]
    qa_projection_key: NodeLogicalKey
    qa_projection_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        for camera_id in CAMERA_IDS:
            if self.camera_facts[camera_id].camera_id is not camera_id:
                raise ValueError("projected QA camera key must match camera_id")
        digest = qa_projection_semantic_sha256(self)
        if (
            self.qa_projection_semantic_sha256 != digest
            or self.qa_projection_key != f"{QA_PROJECTION_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("QA projection identity is inconsistent")
        return self


class EventHypothesis(StrictModel):
    """A projected action hypothesis; it is not yet a durable physical fact."""

    schema_version: Literal["1.0"] = "1.0"
    projector_version: Literal["mage-observation-event-projector-v1"] = EVENT_PROJECTOR_VERSION
    source_observation_logical_key: NodeLogicalKey
    source_observation_semantic_sha256: Sha256Digest
    source_local_ref: NonEmptyString
    hypothesis_logical_key: NodeLogicalKey
    hypothesis_semantic_sha256: Sha256Digest
    action: NonEmptyString
    interval: NanosecondInterval
    model_reported_confidence: UnitInterval | None = None
    actor: ActorObservation | None = None
    object: ObjectObservation | None = None
    start_confidence: UnitInterval
    end_confidence: UnitInterval
    started_before_context: bool
    continues_after_context: bool

    @model_validator(mode="after")
    def validate_hypothesis(self) -> Self:
        digest = event_hypothesis_semantic_sha256(self)
        if (
            self.hypothesis_semantic_sha256 != digest
            or self.hypothesis_logical_key != f"{EVENT_HYPOTHESIS_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("event hypothesis identity is inconsistent")
        return self


class EventProjection(StrictModel):
    """All hypotheses projected from one observation in canonical order."""

    schema_version: Literal["1.0"] = "1.0"
    projector_version: Literal["mage-observation-event-projector-v1"] = EVENT_PROJECTOR_VERSION
    source_observation_logical_key: NodeLogicalKey
    source_observation_semantic_sha256: Sha256Digest
    hypotheses: tuple[EventHypothesis, ...]
    event_projection_key: NodeLogicalKey
    event_projection_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        sort_keys = tuple(
            (item.interval.start_ns, item.interval.end_ns, item.source_local_ref)
            for item in self.hypotheses
        )
        if sort_keys != tuple(sorted(sort_keys)):
            raise ValueError("event hypotheses must be in canonical temporal order")
        digest = event_projection_semantic_sha256(self)
        if (
            self.event_projection_semantic_sha256 != digest
            or self.event_projection_key != f"{EVENT_PROJECTION_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("event projection identity is inconsistent")
        return self


class ProjectedCameraEvidenceFact(StrictModel):
    """One explicit camera fact projected without another model generation."""

    schema_version: Literal["1.0"] = "1.0"
    projector_version: Literal["mage-observation-evidence-projector-v1"] = (
        EVIDENCE_PROJECTOR_VERSION
    )
    source_observation_semantic_sha256: Sha256Digest
    hypothesis_logical_key: NodeLogicalKey
    hypothesis_semantic_sha256: Sha256Digest
    camera_id: CameraId
    selected_for_inference: bool
    relation: CameraEvidenceRelation
    visibility: UnitInterval | None = None
    observed_interval: NanosecondInterval | None = None
    source_evidence_semantic_sha256_values: tuple[Sha256Digest, ...] = ()
    evidence_fact_key: NodeLogicalKey
    evidence_fact_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_fact(self) -> Self:
        digest = camera_evidence_fact_semantic_sha256(self)
        if (
            self.evidence_fact_semantic_sha256 != digest
            or self.evidence_fact_key != f"{EVIDENCE_FACT_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("camera evidence fact identity is inconsistent")
        return self


class EventEvidenceProjection(StrictModel):
    """Six explicit evidence slots for one projected event hypothesis."""

    hypothesis_logical_key: NodeLogicalKey
    hypothesis_semantic_sha256: Sha256Digest
    cameras: SixCameraMap[ProjectedCameraEvidenceFact]

    @model_validator(mode="after")
    def validate_cameras(self) -> Self:
        for camera_id in CAMERA_IDS:
            fact = self.cameras[camera_id]
            if fact.camera_id is not camera_id:
                raise ValueError("evidence camera key must match camera_id")
            if fact.hypothesis_logical_key != self.hypothesis_logical_key:
                raise ValueError("evidence fact belongs to another hypothesis")
            if fact.hypothesis_semantic_sha256 != self.hypothesis_semantic_sha256:
                raise ValueError("evidence fact hypothesis digest mismatch")
        return self


class EvidenceProjection(StrictModel):
    """The logical evidence stage retained without a second VLM invocation."""

    schema_version: Literal["1.0"] = "1.0"
    projector_version: Literal["mage-observation-evidence-projector-v1"] = (
        EVIDENCE_PROJECTOR_VERSION
    )
    source_observation_logical_key: NodeLogicalKey
    source_observation_semantic_sha256: Sha256Digest
    event_evidence: tuple[EventEvidenceProjection, ...]
    evidence_projection_key: NodeLogicalKey
    evidence_projection_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        keys = tuple(item.hypothesis_logical_key for item in self.event_evidence)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("event evidence must be unique and ordered by hypothesis key")
        digest = evidence_projection_semantic_sha256(self)
        if (
            self.evidence_projection_semantic_sha256 != digest
            or self.evidence_projection_key != f"{EVIDENCE_PROJECTION_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("evidence projection identity is inconsistent")
        return self


def media_health_semantic_projection(report: MediaHealthReport) -> dict[str, object]:
    return {
        "projection_version": report.projection_version,
        "context_manifest_semantic_sha256": report.context_manifest_semantic_sha256,
        "policy_version": report.policy_version,
        "cameras": report.cameras.model_dump(mode="json"),
    }


def media_health_semantic_sha256(report: MediaHealthReport) -> Sha256Digest:
    return semantic_sha256(media_health_semantic_projection(report))


def create_media_health_report(
    *,
    context_manifest_semantic_sha256: Sha256Digest,
    policy_version: str,
    cameras: SixCameraMap[MediaHealthCameraFact],
) -> MediaHealthReport:
    values = {
        "context_manifest_semantic_sha256": context_manifest_semantic_sha256,
        "policy_version": policy_version,
        "cameras": cameras,
    }
    draft = MediaHealthReport.model_construct(
        media_health_key=f"{MEDIA_HEALTH_KEY_NAMESPACE}:{'0' * 64}",
        media_health_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = media_health_semantic_sha256(draft)
    return MediaHealthReport(
        media_health_key=f"{MEDIA_HEALTH_KEY_NAMESPACE}:{digest}",
        media_health_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


def qa_projection_semantic_projection(projection: QaProjection) -> dict[str, object]:
    return {
        "projector_version": projection.projector_version,
        "source_observation_logical_key": projection.source_observation_logical_key,
        "source_observation_semantic_sha256": projection.source_observation_semantic_sha256,
        "source_media_health_semantic_sha256": (projection.source_media_health_semantic_sha256),
        "camera_facts": projection.camera_facts.model_dump(mode="json"),
    }


def qa_projection_semantic_sha256(projection: QaProjection) -> Sha256Digest:
    return semantic_sha256(qa_projection_semantic_projection(projection))


def _projected_qa_disposition(
    media: MediaHealthDisposition,
    semantic: SemanticQaDisposition,
) -> ProjectedQaDisposition:
    if media is MediaHealthDisposition.UNAVAILABLE:
        return ProjectedQaDisposition.UNAVAILABLE
    if media is MediaHealthDisposition.UNUSABLE or semantic is SemanticQaDisposition.UNUSABLE:
        return ProjectedQaDisposition.UNUSABLE
    if media is MediaHealthDisposition.DEGRADED or semantic is SemanticQaDisposition.DEGRADED:
        return ProjectedQaDisposition.DEGRADED
    if semantic is SemanticQaDisposition.UNKNOWN:
        return ProjectedQaDisposition.UNKNOWN
    return ProjectedQaDisposition.USABLE


class QaProjector:
    """Fuse deterministic media health with Mage semantic QA."""

    def project(
        self, observation: MageObservation, media_health: MediaHealthReport
    ) -> QaProjection:
        if (
            media_health.context_manifest_semantic_sha256
            != observation.context.context_manifest_semantic_sha256
        ):
            raise ValueError("media health belongs to another perception context")
        facts: dict[CameraId, ProjectedCameraQaFact] = {}
        for camera_id in CAMERA_IDS:
            media = media_health.cameras[camera_id]
            semantic = observation.semantic_qa[camera_id]
            issue_codes = tuple(
                sorted(
                    set(media.issue_codes) | {f"semantic:{issue.code}" for issue in semantic.issues}
                )
            )
            facts[camera_id] = ProjectedCameraQaFact(
                camera_id=camera_id,
                disposition=_projected_qa_disposition(media.disposition, semantic.disposition),
                media_health_disposition=media.disposition,
                semantic_qa_disposition=semantic.disposition,
                issue_codes=issue_codes,
                semantic_confidence=semantic.confidence,
            )
        values = {
            "source_observation_logical_key": observation.observation_logical_key,
            "source_observation_semantic_sha256": observation.observation_semantic_sha256,
            "source_media_health_semantic_sha256": media_health.media_health_semantic_sha256,
            "camera_facts": SixCameraMap(facts),
        }
        draft = QaProjection.model_construct(
            qa_projection_key=f"{QA_PROJECTION_KEY_NAMESPACE}:{'0' * 64}",
            qa_projection_semantic_sha256="0" * 64,
            **cast(dict[str, Any], values),
        )
        digest = qa_projection_semantic_sha256(draft)
        return QaProjection(
            qa_projection_key=f"{QA_PROJECTION_KEY_NAMESPACE}:{digest}",
            qa_projection_semantic_sha256=digest,
            **cast(dict[str, Any], values),
        )


def event_hypothesis_semantic_projection(hypothesis: EventHypothesis) -> dict[str, object]:
    return {
        "projector_version": hypothesis.projector_version,
        "source_observation_logical_key": hypothesis.source_observation_logical_key,
        "source_observation_semantic_sha256": hypothesis.source_observation_semantic_sha256,
        "source_local_ref": hypothesis.source_local_ref,
        "action": hypothesis.action,
        "interval": hypothesis.interval.model_dump(mode="json"),
        "model_reported_confidence": hypothesis.model_reported_confidence,
        "actor": hypothesis.actor.model_dump(mode="json") if hypothesis.actor else None,
        "object": hypothesis.object.model_dump(mode="json") if hypothesis.object else None,
        "start_confidence": hypothesis.start_confidence,
        "end_confidence": hypothesis.end_confidence,
        "started_before_context": hypothesis.started_before_context,
        "continues_after_context": hypothesis.continues_after_context,
    }


def event_hypothesis_semantic_sha256(hypothesis: EventHypothesis) -> Sha256Digest:
    return semantic_sha256(event_hypothesis_semantic_projection(hypothesis))


def _create_event_hypothesis(
    observation: MageObservation,
    source: MageActionObservation,
) -> EventHypothesis:
    values = {
        "source_observation_logical_key": observation.observation_logical_key,
        "source_observation_semantic_sha256": observation.observation_semantic_sha256,
        "source_local_ref": source.local_ref,
        "action": source.action,
        "interval": source.interval,
        "model_reported_confidence": source.confidence,
        "actor": source.actor,
        "object": source.object,
        "start_confidence": source.boundary.start_confidence,
        "end_confidence": source.boundary.end_confidence,
        "started_before_context": source.boundary.started_before_context,
        "continues_after_context": source.boundary.continues_after_context,
    }
    draft = EventHypothesis.model_construct(
        hypothesis_logical_key=f"{EVENT_HYPOTHESIS_KEY_NAMESPACE}:{'0' * 64}",
        hypothesis_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = event_hypothesis_semantic_sha256(draft)
    return EventHypothesis(
        hypothesis_logical_key=f"{EVENT_HYPOTHESIS_KEY_NAMESPACE}:{digest}",
        hypothesis_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


def event_projection_semantic_projection(projection: EventProjection) -> dict[str, object]:
    return {
        "projector_version": projection.projector_version,
        "source_observation_logical_key": projection.source_observation_logical_key,
        "source_observation_semantic_sha256": projection.source_observation_semantic_sha256,
        "ordered_hypothesis_semantic_sha256_values": [
            item.hypothesis_semantic_sha256 for item in projection.hypotheses
        ],
    }


def event_projection_semantic_sha256(projection: EventProjection) -> Sha256Digest:
    return semantic_sha256(event_projection_semantic_projection(projection))


class EventProjector:
    """Project all action observations without any additional model call."""

    def project(self, observation: MageObservation) -> EventProjection:
        hypotheses = tuple(
            sorted(
                (_create_event_hypothesis(observation, item) for item in observation.observations),
                key=lambda item: (
                    item.interval.start_ns,
                    item.interval.end_ns,
                    item.source_local_ref,
                ),
            )
        )
        values = {
            "source_observation_logical_key": observation.observation_logical_key,
            "source_observation_semantic_sha256": observation.observation_semantic_sha256,
            "hypotheses": hypotheses,
        }
        draft = EventProjection.model_construct(
            event_projection_key=f"{EVENT_PROJECTION_KEY_NAMESPACE}:{'0' * 64}",
            event_projection_semantic_sha256="0" * 64,
            **cast(dict[str, Any], values),
        )
        digest = event_projection_semantic_sha256(draft)
        return EventProjection(
            event_projection_key=f"{EVENT_PROJECTION_KEY_NAMESPACE}:{digest}",
            event_projection_semantic_sha256=digest,
            **cast(dict[str, Any], values),
        )


def camera_evidence_fact_semantic_projection(
    fact: ProjectedCameraEvidenceFact,
) -> dict[str, object]:
    return {
        "projector_version": fact.projector_version,
        "source_observation_semantic_sha256": fact.source_observation_semantic_sha256,
        "hypothesis_logical_key": fact.hypothesis_logical_key,
        "hypothesis_semantic_sha256": fact.hypothesis_semantic_sha256,
        "camera_id": fact.camera_id.value,
        "selected_for_inference": fact.selected_for_inference,
        "relation": fact.relation.value,
        "visibility": fact.visibility,
        "observed_interval": (
            fact.observed_interval.model_dump(mode="json")
            if fact.observed_interval is not None
            else None
        ),
        "source_evidence_semantic_sha256_values": list(fact.source_evidence_semantic_sha256_values),
    }


def camera_evidence_fact_semantic_sha256(
    fact: ProjectedCameraEvidenceFact,
) -> Sha256Digest:
    return semantic_sha256(camera_evidence_fact_semantic_projection(fact))


def _create_camera_evidence_fact(
    observation: MageObservation,
    hypothesis: EventHypothesis,
    source: MageActionObservation,
    camera_id: CameraId,
) -> ProjectedCameraEvidenceFact:
    evidence = source.camera_evidence[camera_id]
    values = {
        "source_observation_semantic_sha256": observation.observation_semantic_sha256,
        "hypothesis_logical_key": hypothesis.hypothesis_logical_key,
        "hypothesis_semantic_sha256": hypothesis.hypothesis_semantic_sha256,
        "camera_id": camera_id,
        "selected_for_inference": observation.context.cameras[camera_id].selected_for_inference,
        "relation": evidence.relation,
        "visibility": evidence.visibility,
        "observed_interval": evidence.observed_interval,
        "source_evidence_semantic_sha256_values": evidence.evidence_semantic_sha256_values,
    }
    draft = ProjectedCameraEvidenceFact.model_construct(
        evidence_fact_key=f"{EVIDENCE_FACT_KEY_NAMESPACE}:{'0' * 64}",
        evidence_fact_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = camera_evidence_fact_semantic_sha256(draft)
    return ProjectedCameraEvidenceFact(
        evidence_fact_key=f"{EVIDENCE_FACT_KEY_NAMESPACE}:{digest}",
        evidence_fact_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


def evidence_projection_semantic_projection(
    projection: EvidenceProjection,
) -> dict[str, object]:
    return {
        "projector_version": projection.projector_version,
        "source_observation_logical_key": projection.source_observation_logical_key,
        "source_observation_semantic_sha256": projection.source_observation_semantic_sha256,
        "event_evidence": [
            {
                "hypothesis_logical_key": item.hypothesis_logical_key,
                "hypothesis_semantic_sha256": item.hypothesis_semantic_sha256,
                "ordered_camera_fact_semantic_sha256_values": [
                    item.cameras[camera_id].evidence_fact_semantic_sha256
                    for camera_id in CAMERA_IDS
                ],
            }
            for item in projection.event_evidence
        ],
    }


def evidence_projection_semantic_sha256(projection: EvidenceProjection) -> Sha256Digest:
    return semantic_sha256(evidence_projection_semantic_projection(projection))


class EvidenceProjector:
    """Project the explicit six-camera evidence universe from the same observation."""

    def project(
        self,
        observation: MageObservation,
        event_projection: EventProjection,
    ) -> EvidenceProjection:
        if (
            event_projection.source_observation_semantic_sha256
            != observation.observation_semantic_sha256
        ):
            raise ValueError("event projection belongs to another Mage observation")
        source_by_ref = {item.local_ref: item for item in observation.observations}
        bundles: list[EventEvidenceProjection] = []
        for hypothesis in event_projection.hypotheses:
            source = source_by_ref.get(hypothesis.source_local_ref)
            if source is None:
                raise ValueError("event hypothesis source local_ref is missing")
            camera_facts = SixCameraMap(
                {
                    camera_id: _create_camera_evidence_fact(
                        observation, hypothesis, source, camera_id
                    )
                    for camera_id in CAMERA_IDS
                }
            )
            bundles.append(
                EventEvidenceProjection(
                    hypothesis_logical_key=hypothesis.hypothesis_logical_key,
                    hypothesis_semantic_sha256=hypothesis.hypothesis_semantic_sha256,
                    cameras=camera_facts,
                )
            )
        ordered = tuple(sorted(bundles, key=lambda item: item.hypothesis_logical_key))
        values = {
            "source_observation_logical_key": observation.observation_logical_key,
            "source_observation_semantic_sha256": observation.observation_semantic_sha256,
            "event_evidence": ordered,
        }
        draft = EvidenceProjection.model_construct(
            evidence_projection_key=f"{EVIDENCE_PROJECTION_KEY_NAMESPACE}:{'0' * 64}",
            evidence_projection_semantic_sha256="0" * 64,
            **cast(dict[str, Any], values),
        )
        digest = evidence_projection_semantic_sha256(draft)
        return EvidenceProjection(
            evidence_projection_key=f"{EVIDENCE_PROJECTION_KEY_NAMESPACE}:{digest}",
            evidence_projection_semantic_sha256=digest,
            **cast(dict[str, Any], values),
        )


__all__ = [
    "EVENT_PROJECTOR_VERSION",
    "EVIDENCE_PROJECTOR_VERSION",
    "QA_PROJECTOR_VERSION",
    "EventEvidenceProjection",
    "EventHypothesis",
    "EventProjection",
    "EventProjector",
    "EvidenceProjection",
    "EvidenceProjector",
    "MediaHealthCameraFact",
    "MediaHealthDisposition",
    "MediaHealthReport",
    "ProjectedCameraEvidenceFact",
    "ProjectedCameraQaFact",
    "ProjectedQaDisposition",
    "QaProjection",
    "QaProjector",
    "camera_evidence_fact_semantic_projection",
    "camera_evidence_fact_semantic_sha256",
    "create_media_health_report",
    "event_hypothesis_semantic_projection",
    "event_hypothesis_semantic_sha256",
    "event_projection_semantic_projection",
    "event_projection_semantic_sha256",
    "evidence_projection_semantic_projection",
    "evidence_projection_semantic_sha256",
    "media_health_semantic_projection",
    "media_health_semantic_sha256",
    "qa_projection_semantic_projection",
    "qa_projection_semantic_sha256",
]
