"""Deterministic temporal reconciliation of segment observations into event tracks."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, model_validator

from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.contracts.perception_stream import (
    ActorObservation,
    CanonicalToken,
    NonEmptyString,
    ObjectObservation,
    UnitInterval,
)
from robata.perception.projectors import EventHypothesis, EventProjection

EVENT_TRACK_IDENTITY_POLICY_VERSION: Final = "event-track-identity-v1"
EVENT_TRACK_REVISION_PROJECTION_VERSION: Final = "event-track-revision-semantic-v1"
EVENT_TRACK_KEY_NAMESPACE: Final = "event-track-v1"
EVENT_TRACK_UUID_NAMESPACE: Final = "robata:event-track-v1"
TEMPORAL_RECONCILE_PROJECTION_VERSION: Final = "temporal-reconcile-semantic-v1"
TEMPORAL_RECONCILE_KEY_NAMESPACE: Final = "temporal-reconcile-v1"

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class EventTrackState(StrEnum):
    """Lifecycle of a cross-segment event hypothesis."""

    CANDIDATE = "CANDIDATE"
    OPEN = "OPEN"
    UPDATED = "UPDATED"
    CLOSED = "CLOSED"
    FINALIZED = "FINALIZED"


class EventHypothesisReference(StrictModel):
    """Minimal immutable lineage retained by an event-track revision."""

    hypothesis_logical_key: NodeLogicalKey
    hypothesis_semantic_sha256: Sha256Digest
    source_observation_logical_key: NodeLogicalKey
    source_observation_semantic_sha256: Sha256Digest
    source_local_ref: NonEmptyString
    interval: NanosecondInterval


class EventTrackPolicy(StrictModel):
    """Versioned deterministic association policy."""

    version: SchemaVersion
    max_merge_gap_ns: NonNegativeInt = 250_000_000
    require_continuation_signal_for_nonoverlap: bool = True


class EventTrackRevision(StrictModel):
    """One immutable revision of a stable cross-segment track."""

    schema_version: Literal["1.0"] = "1.0"
    projection_version: Literal["event-track-revision-semantic-v1"] = (
        EVENT_TRACK_REVISION_PROJECTION_VERSION
    )
    identity_policy_version: Literal["event-track-identity-v1"] = (
        EVENT_TRACK_IDENTITY_POLICY_VERSION
    )
    event_track_id: OpaqueUuid
    event_track_key: NodeLogicalKey
    event_track_identity_sha256: Sha256Digest
    identity_seed_hypothesis_logical_key: NodeLogicalKey
    identity_seed_hypothesis_semantic_sha256: Sha256Digest
    revision_index: NonNegativeInt
    parent_revision_semantic_sha256: Sha256Digest | None
    revision_semantic_sha256: Sha256Digest
    state: EventTrackState
    action: CanonicalToken
    interval: NanosecondInterval
    actor: ActorObservation | None = None
    object: ObjectObservation | None = None
    source_hypotheses: tuple[EventHypothesisReference, ...]
    model_reported_confidence_values: tuple[UnitInterval, ...] = ()
    start_confidence: UnitInterval
    end_confidence: UnitInterval
    continues_after_context: bool
    tracking_policy_version: SchemaVersion
    resolved_event_semantic_sha256: Sha256Digest | None = None
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        if not self.source_hypotheses:
            raise ValueError("event track requires at least one source hypothesis")
        keys = tuple(item.hypothesis_logical_key for item in self.source_hypotheses)
        if len(set(keys)) != len(keys):
            raise ValueError("event track source hypotheses must be unique")
        temporal_keys = tuple(
            (item.interval.start_ns, item.interval.end_ns, item.hypothesis_logical_key)
            for item in self.source_hypotheses
        )
        if temporal_keys != tuple(sorted(temporal_keys)):
            raise ValueError("event track source hypotheses must be in temporal order")
        if self.revision_index == 0 and self.parent_revision_semantic_sha256 is not None:
            raise ValueError("first event-track revision cannot have a parent")
        if self.revision_index > 0 and self.parent_revision_semantic_sha256 is None:
            raise ValueError("later event-track revision requires a parent digest")
        if self.state is EventTrackState.FINALIZED:
            if self.resolved_event_semantic_sha256 is None:
                raise ValueError("finalized track requires a resolved event digest")
            if self.continues_after_context:
                raise ValueError("finalized track cannot continue")
        elif self.resolved_event_semantic_sha256 is not None:
            raise ValueError("only finalized tracks may bind a resolved event")
        if self.state is EventTrackState.CLOSED and self.continues_after_context:
            raise ValueError("closed track cannot continue")

        if not any(
            item.hypothesis_logical_key == self.identity_seed_hypothesis_logical_key
            and item.hypothesis_semantic_sha256 == self.identity_seed_hypothesis_semantic_sha256
            for item in self.source_hypotheses
        ):
            raise ValueError("event-track identity seed must remain in source lineage")
        identity_digest = event_track_identity_sha256_from_values(
            first_hypothesis_logical_key=self.identity_seed_hypothesis_logical_key,
            first_hypothesis_semantic_sha256=(self.identity_seed_hypothesis_semantic_sha256),
            action=self.action,
            actor=self.actor,
            object=self.object,
            tracking_policy_version=self.tracking_policy_version,
        )
        revision_digest = event_track_revision_semantic_sha256(self)
        if (
            self.event_track_identity_sha256 != identity_digest
            or self.event_track_key != f"{EVENT_TRACK_KEY_NAMESPACE}:{identity_digest}"
            or self.event_track_id
            != str(uuid5(NAMESPACE_URL, f"{EVENT_TRACK_UUID_NAMESPACE}:{identity_digest}"))
        ):
            raise ValueError("event-track logical identity is inconsistent")
        if self.revision_semantic_sha256 != revision_digest:
            raise ValueError("event-track revision digest is inconsistent")
        return self


class TemporalReconcileResult(StrictModel):
    """Deterministic result of reconciling one event projection."""

    schema_version: Literal["1.0"] = "1.0"
    projection_version: Literal["temporal-reconcile-semantic-v1"] = (
        TEMPORAL_RECONCILE_PROJECTION_VERSION
    )
    tracking_policy_version: SchemaVersion
    source_event_projection_semantic_sha256: Sha256Digest
    prior_revision_semantic_sha256_values: tuple[Sha256Digest, ...]
    current_tracks: tuple[EventTrackRevision, ...]
    created_track_keys: tuple[NodeLogicalKey, ...]
    updated_track_keys: tuple[NodeLogicalKey, ...]
    closed_track_keys: tuple[NodeLogicalKey, ...]
    reconcile_key: NodeLogicalKey
    reconcile_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        track_keys = tuple(item.event_track_key for item in self.current_tracks)
        if track_keys != tuple(sorted(track_keys)) or len(set(track_keys)) != len(track_keys):
            raise ValueError("current tracks must be unique and ordered by key")
        for values in (
            self.created_track_keys,
            self.updated_track_keys,
            self.closed_track_keys,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("track transition key sets must be unique and ordered")
        digest = temporal_reconcile_semantic_sha256(self)
        if (
            self.reconcile_semantic_sha256 != digest
            or self.reconcile_key != f"{TEMPORAL_RECONCILE_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("temporal reconcile identity is inconsistent")
        return self


def _hypothesis_reference(hypothesis: EventHypothesis) -> EventHypothesisReference:
    return EventHypothesisReference(
        hypothesis_logical_key=hypothesis.hypothesis_logical_key,
        hypothesis_semantic_sha256=hypothesis.hypothesis_semantic_sha256,
        source_observation_logical_key=hypothesis.source_observation_logical_key,
        source_observation_semantic_sha256=hypothesis.source_observation_semantic_sha256,
        source_local_ref=hypothesis.source_local_ref,
        interval=hypothesis.interval,
    )


def event_track_identity_projection_from_values(
    *,
    first_hypothesis_logical_key: NodeLogicalKey,
    first_hypothesis_semantic_sha256: Sha256Digest,
    action: str,
    actor: ActorObservation | None,
    object: ObjectObservation | None,
    tracking_policy_version: str,
) -> dict[str, object]:
    return {
        "identity_policy_version": EVENT_TRACK_IDENTITY_POLICY_VERSION,
        "first_hypothesis_logical_key": first_hypothesis_logical_key,
        "first_hypothesis_semantic_sha256": first_hypothesis_semantic_sha256,
        "action": action,
        "actor": actor.model_dump(mode="json") if actor is not None else None,
        "object": object.model_dump(mode="json") if object is not None else None,
        "tracking_policy_version": tracking_policy_version,
    }


def event_track_identity_sha256_from_values(
    *,
    first_hypothesis_logical_key: NodeLogicalKey,
    first_hypothesis_semantic_sha256: Sha256Digest,
    action: str,
    actor: ActorObservation | None,
    object: ObjectObservation | None,
    tracking_policy_version: str,
) -> Sha256Digest:
    return semantic_sha256(
        event_track_identity_projection_from_values(
            first_hypothesis_logical_key=first_hypothesis_logical_key,
            first_hypothesis_semantic_sha256=first_hypothesis_semantic_sha256,
            action=action,
            actor=actor,
            object=object,
            tracking_policy_version=tracking_policy_version,
        )
    )


def event_track_revision_semantic_projection(
    revision: EventTrackRevision,
) -> dict[str, object]:
    return {
        "projection_version": revision.projection_version,
        "event_track_identity_sha256": revision.event_track_identity_sha256,
        "identity_seed_hypothesis_logical_key": (revision.identity_seed_hypothesis_logical_key),
        "identity_seed_hypothesis_semantic_sha256": (
            revision.identity_seed_hypothesis_semantic_sha256
        ),
        "revision_index": revision.revision_index,
        "parent_revision_semantic_sha256": revision.parent_revision_semantic_sha256,
        "state": revision.state.value,
        "action": revision.action,
        "interval": revision.interval.model_dump(mode="json"),
        "actor": revision.actor.model_dump(mode="json") if revision.actor is not None else None,
        "object": revision.object.model_dump(mode="json") if revision.object is not None else None,
        "source_hypotheses": [item.model_dump(mode="json") for item in revision.source_hypotheses],
        "model_reported_confidence_values": list(revision.model_reported_confidence_values),
        "start_confidence": revision.start_confidence,
        "end_confidence": revision.end_confidence,
        "continues_after_context": revision.continues_after_context,
        "tracking_policy_version": revision.tracking_policy_version,
        "resolved_event_semantic_sha256": revision.resolved_event_semantic_sha256,
    }


def event_track_revision_semantic_sha256(revision: EventTrackRevision) -> Sha256Digest:
    return semantic_sha256(event_track_revision_semantic_projection(revision))


def _build_track_revision(
    *,
    event_track_id: str,
    event_track_key: NodeLogicalKey,
    event_track_identity_sha256: Sha256Digest,
    identity_seed_hypothesis_logical_key: NodeLogicalKey,
    identity_seed_hypothesis_semantic_sha256: Sha256Digest,
    revision_index: int,
    parent_revision_semantic_sha256: Sha256Digest | None,
    state: EventTrackState,
    action: str,
    interval: NanosecondInterval,
    actor: ActorObservation | None,
    object: ObjectObservation | None,
    source_hypotheses: tuple[EventHypothesisReference, ...],
    model_reported_confidence_values: tuple[float, ...],
    start_confidence: float,
    end_confidence: float,
    continues_after_context: bool,
    tracking_policy_version: str,
    resolved_event_semantic_sha256: Sha256Digest | None = None,
) -> EventTrackRevision:
    values = {
        "event_track_id": event_track_id,
        "event_track_key": event_track_key,
        "event_track_identity_sha256": event_track_identity_sha256,
        "identity_seed_hypothesis_logical_key": identity_seed_hypothesis_logical_key,
        "identity_seed_hypothesis_semantic_sha256": (identity_seed_hypothesis_semantic_sha256),
        "revision_index": revision_index,
        "parent_revision_semantic_sha256": parent_revision_semantic_sha256,
        "state": state,
        "action": action,
        "interval": interval,
        "actor": actor,
        "object": object,
        "source_hypotheses": source_hypotheses,
        "model_reported_confidence_values": model_reported_confidence_values,
        "start_confidence": start_confidence,
        "end_confidence": end_confidence,
        "continues_after_context": continues_after_context,
        "tracking_policy_version": tracking_policy_version,
        "resolved_event_semantic_sha256": resolved_event_semantic_sha256,
    }
    draft = EventTrackRevision.model_construct(
        revision_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = event_track_revision_semantic_sha256(draft)
    return EventTrackRevision(revision_semantic_sha256=digest, **cast(dict[str, Any], values))


def create_candidate_track(
    hypothesis: EventHypothesis,
    policy: EventTrackPolicy,
) -> EventTrackRevision:
    """Create the stable track identity from the first hypothesis."""

    identity_digest = event_track_identity_sha256_from_values(
        first_hypothesis_logical_key=hypothesis.hypothesis_logical_key,
        first_hypothesis_semantic_sha256=hypothesis.hypothesis_semantic_sha256,
        action=hypothesis.action,
        actor=hypothesis.actor,
        object=hypothesis.object,
        tracking_policy_version=policy.version,
    )
    confidences = (
        (hypothesis.model_reported_confidence,)
        if hypothesis.model_reported_confidence is not None
        else ()
    )
    return _build_track_revision(
        event_track_id=str(uuid5(NAMESPACE_URL, f"{EVENT_TRACK_UUID_NAMESPACE}:{identity_digest}")),
        event_track_key=f"{EVENT_TRACK_KEY_NAMESPACE}:{identity_digest}",
        event_track_identity_sha256=identity_digest,
        identity_seed_hypothesis_logical_key=hypothesis.hypothesis_logical_key,
        identity_seed_hypothesis_semantic_sha256=(hypothesis.hypothesis_semantic_sha256),
        revision_index=0,
        parent_revision_semantic_sha256=None,
        state=EventTrackState.CANDIDATE,
        action=hypothesis.action,
        interval=hypothesis.interval,
        actor=hypothesis.actor,
        object=hypothesis.object,
        source_hypotheses=(_hypothesis_reference(hypothesis),),
        model_reported_confidence_values=confidences,
        start_confidence=hypothesis.start_confidence,
        end_confidence=hypothesis.end_confidence,
        continues_after_context=hypothesis.continues_after_context,
        tracking_policy_version=policy.version,
    )


def _transition_track(
    track: EventTrackRevision,
    *,
    state: EventTrackState,
    interval: NanosecondInterval | None = None,
    source_hypotheses: tuple[EventHypothesisReference, ...] | None = None,
    model_reported_confidence_values: tuple[float, ...] | None = None,
    end_confidence: float | None = None,
    continues_after_context: bool | None = None,
    resolved_event_semantic_sha256: Sha256Digest | None = None,
) -> EventTrackRevision:
    return _build_track_revision(
        event_track_id=track.event_track_id,
        event_track_key=track.event_track_key,
        event_track_identity_sha256=track.event_track_identity_sha256,
        identity_seed_hypothesis_logical_key=(track.identity_seed_hypothesis_logical_key),
        identity_seed_hypothesis_semantic_sha256=(track.identity_seed_hypothesis_semantic_sha256),
        revision_index=track.revision_index + 1,
        parent_revision_semantic_sha256=track.revision_semantic_sha256,
        state=state,
        action=track.action,
        interval=interval or track.interval,
        actor=track.actor,
        object=track.object,
        source_hypotheses=source_hypotheses or track.source_hypotheses,
        model_reported_confidence_values=(
            model_reported_confidence_values
            if model_reported_confidence_values is not None
            else track.model_reported_confidence_values
        ),
        start_confidence=track.start_confidence,
        end_confidence=end_confidence if end_confidence is not None else track.end_confidence,
        continues_after_context=(
            continues_after_context
            if continues_after_context is not None
            else track.continues_after_context
        ),
        tracking_policy_version=track.tracking_policy_version,
        resolved_event_semantic_sha256=resolved_event_semantic_sha256,
    )


def open_event_track(track: EventTrackRevision) -> EventTrackRevision:
    if track.state is not EventTrackState.CANDIDATE:
        raise ValueError("only a candidate track can be opened")
    return _transition_track(track, state=EventTrackState.OPEN)


def update_event_track(
    track: EventTrackRevision,
    hypothesis: EventHypothesis,
) -> EventTrackRevision:
    if track.state not in {
        EventTrackState.CANDIDATE,
        EventTrackState.OPEN,
        EventTrackState.UPDATED,
    }:
        raise ValueError("only an active track can be updated")
    if not _compatible_identity(track, hypothesis):
        raise ValueError("hypothesis is incompatible with the event track")
    references = tuple(
        sorted(
            (*track.source_hypotheses, _hypothesis_reference(hypothesis)),
            key=lambda item: (
                item.interval.start_ns,
                item.interval.end_ns,
                item.hypothesis_logical_key,
            ),
        )
    )
    if len({item.hypothesis_logical_key for item in references}) != len(references):
        raise ValueError("hypothesis is already present in the event track")
    confidences = track.model_reported_confidence_values
    if hypothesis.model_reported_confidence is not None:
        confidences = (*confidences, hypothesis.model_reported_confidence)
    interval = NanosecondInterval(
        start_ns=min(track.interval.start_ns, hypothesis.interval.start_ns),
        end_ns=max(track.interval.end_ns, hypothesis.interval.end_ns),
    )
    return _transition_track(
        track,
        state=EventTrackState.UPDATED,
        interval=interval,
        source_hypotheses=references,
        model_reported_confidence_values=confidences,
        end_confidence=hypothesis.end_confidence,
        continues_after_context=hypothesis.continues_after_context,
    )


def close_event_track(track: EventTrackRevision) -> EventTrackRevision:
    if track.state not in {
        EventTrackState.CANDIDATE,
        EventTrackState.OPEN,
        EventTrackState.UPDATED,
    }:
        raise ValueError("only an active track can be closed")
    return _transition_track(
        track,
        state=EventTrackState.CLOSED,
        continues_after_context=False,
    )


def finalize_event_track(
    track: EventTrackRevision,
    *,
    resolved_event_semantic_sha256: Sha256Digest,
) -> EventTrackRevision:
    if track.state is not EventTrackState.CLOSED:
        raise ValueError("only a closed track can be finalized")
    return _transition_track(
        track,
        state=EventTrackState.FINALIZED,
        continues_after_context=False,
        resolved_event_semantic_sha256=resolved_event_semantic_sha256,
    )


def _compatible_optional(left: object | None, right: object | None) -> bool:
    return left is None or right is None or left == right


def _compatible_identity(track: EventTrackRevision, hypothesis: EventHypothesis) -> bool:
    return (
        track.action == hypothesis.action
        and _compatible_optional(track.actor, hypothesis.actor)
        and _compatible_optional(track.object, hypothesis.object)
    )


def _interval_gap_ns(left: NanosecondInterval, right: NanosecondInterval) -> int:
    if left.end_ns < right.start_ns:
        return right.start_ns - left.end_ns
    if right.end_ns < left.start_ns:
        return left.start_ns - right.end_ns
    return 0


def _can_associate(
    track: EventTrackRevision,
    hypothesis: EventHypothesis,
    policy: EventTrackPolicy,
) -> bool:
    if track.state not in {
        EventTrackState.CANDIDATE,
        EventTrackState.OPEN,
        EventTrackState.UPDATED,
    }:
        return False
    if not _compatible_identity(track, hypothesis):
        return False
    gap = _interval_gap_ns(track.interval, hypothesis.interval)
    if gap > policy.max_merge_gap_ns:
        return False
    overlaps = (
        track.interval.start_ns < hypothesis.interval.end_ns
        and hypothesis.interval.start_ns < track.interval.end_ns
    )
    return not (
        policy.require_continuation_signal_for_nonoverlap
        and not overlaps
        and not (track.continues_after_context and hypothesis.started_before_context)
    )


def temporal_reconcile_semantic_projection(
    result: TemporalReconcileResult,
) -> dict[str, object]:
    return {
        "projection_version": result.projection_version,
        "tracking_policy_version": result.tracking_policy_version,
        "source_event_projection_semantic_sha256": (result.source_event_projection_semantic_sha256),
        "prior_revision_semantic_sha256_values": list(result.prior_revision_semantic_sha256_values),
        "current_revision_semantic_sha256_values": [
            item.revision_semantic_sha256 for item in result.current_tracks
        ],
        "created_track_keys": list(result.created_track_keys),
        "updated_track_keys": list(result.updated_track_keys),
        "closed_track_keys": list(result.closed_track_keys),
    }


def temporal_reconcile_semantic_sha256(result: TemporalReconcileResult) -> Sha256Digest:
    return semantic_sha256(temporal_reconcile_semantic_projection(result))


class EventTrackReconciler:
    """Associate one non-overlapping segment projection with active tracks."""

    def __init__(self, policy: EventTrackPolicy) -> None:
        self._policy = policy

    def reconcile(
        self,
        prior_tracks: tuple[EventTrackRevision, ...],
        event_projection: EventProjection,
    ) -> TemporalReconcileResult:
        prior_keys = tuple(item.event_track_key for item in prior_tracks)
        if len(set(prior_keys)) != len(prior_keys):
            raise ValueError("prior event tracks must be unique")
        if any(item.tracking_policy_version != self._policy.version for item in prior_tracks):
            raise ValueError("prior track policy version mismatch")

        current = {item.event_track_key: item for item in prior_tracks}
        used_track_keys: set[str] = set()
        created: set[str] = set()
        updated: set[str] = set()
        closed: set[str] = set()

        for hypothesis in event_projection.hypotheses:
            candidates = [
                track
                for track in current.values()
                if track.event_track_key not in used_track_keys
                and _can_associate(track, hypothesis, self._policy)
            ]
            if candidates:
                selected = min(
                    candidates,
                    key=lambda track: (
                        _interval_gap_ns(track.interval, hypothesis.interval),
                        track.event_track_key,
                    ),
                )
                next_track = update_event_track(selected, hypothesis)
                updated.add(selected.event_track_key)
                used_track_keys.add(selected.event_track_key)
            else:
                next_track = create_candidate_track(hypothesis, self._policy)
                created.add(next_track.event_track_key)
                used_track_keys.add(next_track.event_track_key)

            if next_track.continues_after_context:
                if next_track.state is EventTrackState.CANDIDATE:
                    next_track = open_event_track(next_track)
            else:
                next_track = close_event_track(next_track)
                closed.add(next_track.event_track_key)
            current[next_track.event_track_key] = next_track

        for track_key, track in tuple(current.items()):
            if track_key not in used_track_keys and track.state in {
                EventTrackState.CANDIDATE,
                EventTrackState.OPEN,
                EventTrackState.UPDATED,
            }:
                current[track_key] = close_event_track(track)
                closed.add(track_key)

        ordered_tracks = tuple(sorted(current.values(), key=lambda item: item.event_track_key))
        values = {
            "tracking_policy_version": self._policy.version,
            "source_event_projection_semantic_sha256": (
                event_projection.event_projection_semantic_sha256
            ),
            "prior_revision_semantic_sha256_values": tuple(
                item.revision_semantic_sha256
                for item in sorted(prior_tracks, key=lambda item: item.event_track_key)
            ),
            "current_tracks": ordered_tracks,
            "created_track_keys": tuple(sorted(created)),
            "updated_track_keys": tuple(sorted(updated)),
            "closed_track_keys": tuple(sorted(closed)),
        }
        draft = TemporalReconcileResult.model_construct(
            reconcile_key=f"{TEMPORAL_RECONCILE_KEY_NAMESPACE}:{'0' * 64}",
            reconcile_semantic_sha256="0" * 64,
            **cast(dict[str, Any], values),
        )
        digest = temporal_reconcile_semantic_sha256(draft)
        return TemporalReconcileResult(
            reconcile_key=f"{TEMPORAL_RECONCILE_KEY_NAMESPACE}:{digest}",
            reconcile_semantic_sha256=digest,
            **cast(dict[str, Any], values),
        )


__all__ = [
    "EventHypothesisReference",
    "EventTrackPolicy",
    "EventTrackReconciler",
    "EventTrackRevision",
    "EventTrackState",
    "TemporalReconcileResult",
    "close_event_track",
    "create_candidate_track",
    "event_track_identity_projection_from_values",
    "event_track_identity_sha256_from_values",
    "event_track_revision_semantic_projection",
    "event_track_revision_semantic_sha256",
    "finalize_event_track",
    "open_event_track",
    "temporal_reconcile_semantic_projection",
    "temporal_reconcile_semantic_sha256",
    "update_event_track",
]
