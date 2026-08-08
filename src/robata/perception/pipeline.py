"""Provider-neutral stream-oriented perception composition.

The composition has one normal model call per non-overlapping scan segment. QA,
event, and evidence remain separate logical products, but are deterministic
projections of the same Mage observation. Refinement is emitted only as a narrow,
explicit exception request.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Final, Protocol, TypeVar

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.perception_stream import (
    MageObservation,
    PerceptionContextManifest,
    PerceptionRefineRequest,
    RefineReason,
    RefineTargetField,
    create_perception_refine_request,
)
from robata.perception.fusion import (
    FusionAmbiguity,
    PerceptionFusionDecision,
    PerceptionFusionEngine,
)
from robata.perception.projectors import (
    EventProjection,
    EventProjector,
    EvidenceProjection,
    EvidenceProjector,
    MediaHealthReport,
    QaProjection,
    QaProjector,
)
from robata.perception.tracking import (
    EventTrackReconciler,
    EventTrackRevision,
    EventTrackState,
    TemporalReconcileResult,
    close_event_track,
)
from robata.runtime.observability import RuntimeObserver, runtime_span

PERCEPTION_PIPELINE_POLICY_VERSION: Final = "mage-stream-perception-pipeline-v1"
PERCEPTION_TERMINAL_MANIFEST_VERSION: Final = "perception-terminal-manifest-v1"
PERCEPTION_TERMINAL_MANIFEST_KEY_NAMESPACE: Final = "perception-terminal-manifest-v1"
LOCAL_PERCEPTION_ARTIFACT_INDEX_VERSION: Final = "local-perception-artifact-index-v1"
_T = TypeVar("_T")


class PerceptionStage(StrEnum):
    """Physical capability stages; provider names never appear here."""

    MEDIA_SCAN = "MEDIA_SCAN"
    PERCEPTION_OBSERVE = "PERCEPTION_OBSERVE"
    OBSERVATION_PROJECT = "OBSERVATION_PROJECT"
    TEMPORAL_RECONCILE = "TEMPORAL_RECONCILE"
    FUSION = "FUSION"
    PERCEPTION_REFINE = "PERCEPTION_REFINE"
    FINALIZE = "FINALIZE"


_PERCEPTION_STAGE_SPAN_NAMES: Final = {
    PerceptionStage.MEDIA_SCAN: "perception.media_scan",
    PerceptionStage.PERCEPTION_OBSERVE: "perception.observe",
    PerceptionStage.OBSERVATION_PROJECT: "perception.project",
    PerceptionStage.TEMPORAL_RECONCILE: "perception.temporal_reconcile",
    PerceptionStage.FUSION: "perception.fusion",
    PerceptionStage.PERCEPTION_REFINE: "perception.refine",
    PerceptionStage.FINALIZE: "perception.finalize",
}


class MageObservationProvider(Protocol):
    """One expensive normal observation call over a durable context manifest."""

    def observe(self, context: PerceptionContextManifest) -> MageObservation: ...


class PerceptionArtifactSink(Protocol):
    """Optional exact-byte sink used for local recovery and artifact replay."""

    def put(self, *, kind: str, logical_key: str, payload: bytes) -> str: ...


@dataclass(frozen=True, slots=True)
class PerceptionStageMeasurement:
    stage: PerceptionStage
    invocation_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PerceptionContextOutcome:
    context: PerceptionContextManifest
    observation: MageObservation
    media_health: MediaHealthReport
    qa_projection: QaProjection
    event_projection: EventProjection
    evidence_projection: EvidenceProjection
    temporal_reconcile: TemporalReconcileResult


@dataclass(frozen=True, slots=True)
class PerceptionProjectedContext:
    """Deterministic projection handoff before temporal state is reduced.

    This is an internal execution boundary, not a published wire object. It
    lets a durable worker commit ``OBSERVATION_PROJECT`` before claiming the
    causally dependent ``TEMPORAL_RECONCILE`` work item.
    """

    context: PerceptionContextManifest
    observation: MageObservation
    media_health: MediaHealthReport
    qa_projection: QaProjection
    event_projection: EventProjection
    evidence_projection: EvidenceProjection


@dataclass(frozen=True, slots=True)
class LocalPerceptionArtifactReference:
    """Stable logical lookup plus exact-byte identity for one local CAS object."""

    kind: str
    logical_key: str
    exact_sha256: str


@dataclass(frozen=True, slots=True)
class PerceptionTerminalArtifacts:
    """Exact local closure for tracks, fusion, refine handoff, and terminal lineage."""

    event_tracks: tuple[LocalPerceptionArtifactReference, ...]
    fusion_decisions: tuple[LocalPerceptionArtifactReference, ...]
    refine_requests: tuple[LocalPerceptionArtifactReference, ...]
    terminal_manifest: LocalPerceptionArtifactReference


@dataclass(frozen=True, slots=True)
class StreamPerceptionRunResult:
    contexts: tuple[PerceptionContextOutcome, ...]
    event_tracks: tuple[EventTrackRevision, ...]
    fusion_decisions: tuple[PerceptionFusionDecision, ...]
    refine_requests: tuple[PerceptionRefineRequest, ...]
    stage_measurements: tuple[PerceptionStageMeasurement, ...]
    normal_model_call_count: int
    refinement_model_call_count: int
    terminal_artifacts: PerceptionTerminalArtifacts | None = None

    @property
    def total_model_call_count(self) -> int:
        return self.normal_model_call_count + self.refinement_model_call_count


class LocalPerceptionArtifactStore:
    """Minimal exact-byte CAS; accepted artifacts are replayed, not recomputed."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, *, kind: str, logical_key: str, payload: bytes) -> str:
        if not kind or not logical_key or not payload:
            raise ValueError("artifact kind, logical key, and payload must be nonempty")
        digest = hashlib.sha256(payload).hexdigest()
        path = self._artifact_path(kind=kind, exact_sha256=digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as output:
                output.write(payload)
        except FileExistsError as error:
            if path.read_bytes() != payload:
                raise RuntimeError("perception CAS digest conflicts with existing bytes") from error
        self._put_logical_reference(
            LocalPerceptionArtifactReference(
                kind=kind,
                logical_key=logical_key,
                exact_sha256=digest,
            )
        )
        return digest

    def read(self, *, kind: str, logical_key: str) -> bytes:
        """Resolve a logical key to exact accepted bytes and verify both index and CAS."""

        reference = self.reference(kind=kind, logical_key=logical_key)
        path = self._artifact_path(kind=kind, exact_sha256=reference.exact_sha256)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise RuntimeError("perception CAS object is missing") from error
        if hashlib.sha256(payload).hexdigest() != reference.exact_sha256:
            raise RuntimeError("perception CAS object failed exact-byte verification")
        return payload

    def reference(self, *, kind: str, logical_key: str) -> LocalPerceptionArtifactReference:
        if not kind or not logical_key:
            raise ValueError("artifact kind and logical key must be nonempty")
        path = self._logical_reference_path(kind=kind, logical_key=logical_key)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise KeyError("perception artifact logical reference is unavailable") from error
        expected = {
            "index_version": LOCAL_PERCEPTION_ARTIFACT_INDEX_VERSION,
            "kind": kind,
            "logical_key": logical_key,
        }
        if not isinstance(document, dict) or any(
            document.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError("perception artifact logical reference is inconsistent")
        exact_sha256 = document.get("exact_sha256")
        if (
            not isinstance(exact_sha256, str)
            or len(exact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in exact_sha256)
        ):
            raise RuntimeError("perception artifact logical reference has an invalid digest")
        return LocalPerceptionArtifactReference(
            kind=kind,
            logical_key=logical_key,
            exact_sha256=exact_sha256,
        )

    def references(self, *, kind: str) -> tuple[LocalPerceptionArtifactReference, ...]:
        """Enumerate one local artifact family through verified logical index entries."""

        if not kind:
            raise ValueError("artifact kind must be nonempty")
        root = self._root / "_logical" / kind
        if not root.exists():
            return ()
        values: list[LocalPerceptionArtifactReference] = []
        for path in sorted(root.glob("*.ref")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError("perception artifact logical index is unreadable") from error
            logical_key = document.get("logical_key") if isinstance(document, dict) else None
            if not isinstance(logical_key, str):
                raise RuntimeError("perception artifact logical index lacks a logical key")
            values.append(self.reference(kind=kind, logical_key=logical_key))
        return tuple(sorted(values, key=lambda item: item.logical_key))

    def _put_logical_reference(self, reference: LocalPerceptionArtifactReference) -> None:
        path = self._logical_reference_path(
            kind=reference.kind,
            logical_key=reference.logical_key,
        )
        payload = canonical_json_bytes(
            {
                "index_version": LOCAL_PERCEPTION_ARTIFACT_INDEX_VERSION,
                "kind": reference.kind,
                "logical_key": reference.logical_key,
                "exact_sha256": reference.exact_sha256,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as output:
                output.write(payload)
        except FileExistsError as error:
            if path.read_bytes() != payload:
                raise RuntimeError(
                    "perception logical key already points to different exact bytes"
                ) from error

    def _artifact_path(self, *, kind: str, exact_sha256: str) -> Path:
        return self._root / kind / exact_sha256[:2] / f"{exact_sha256}.json"

    def _logical_reference_path(self, *, kind: str, logical_key: str) -> Path:
        key_digest = hashlib.sha256(logical_key.encode("utf-8")).hexdigest()
        return self._root / "_logical" / kind / f"{key_digest}.ref"


class StreamPerceptionSession:
    """Single-worker, backpressured consumer for an ordered perception stream.

    A session deliberately owns only one context at a time.  Callers materialize
    and scan a focus segment, call :meth:`consume`, then proceed to the next
    segment.  This keeps the normal model call on the critical path before a
    later segment can be prepared, while preserving the same deterministic
    projector, tracker, fusion, and artifact semantics as the batch convenience
    method.
    """

    def __init__(self, pipeline: StreamPerceptionPipeline) -> None:
        self._pipeline = pipeline
        self._elapsed_by_stage = {stage: 0.0 for stage in PerceptionStage}
        self._count_by_stage = {stage: 0 for stage in PerceptionStage}
        self._outcomes: list[PerceptionContextOutcome] = []
        self._tracks: tuple[EventTrackRevision, ...] = ()
        self._evidence_history: list[EvidenceProjection] = []
        self._qa_history: list[QaProjection] = []
        self._fused_by_track: dict[str, PerceptionFusionDecision] = {}
        self._prior_current_segment: tuple[int, int, str] | None = None
        self._source_recording_key: str | None = None
        self._source_recording_exact_sha256: str | None = None
        self._context_policy_version: str | None = None
        self._codec_policy_version: str | None = None
        self._pending_projection: PerceptionProjectedContext | None = None
        self._finalized = False

    def scan_media(self, operation: Callable[[], MediaHealthReport]) -> MediaHealthReport:
        """Measure a deterministic media scan before observing its context."""

        if self._finalized:
            raise RuntimeError("cannot scan media after perception session finalization")
        return self._pipeline._timed(
            PerceptionStage.MEDIA_SCAN,
            operation,
            self._elapsed_by_stage,
            self._count_by_stage,
        )

    def stage_measurements(self) -> tuple[PerceptionStageMeasurement, ...]:
        """Snapshot real elapsed/count values accumulated so far in this session."""

        return tuple(
            PerceptionStageMeasurement(
                stage=stage,
                invocation_count=self._count_by_stage[stage],
                elapsed_seconds=self._elapsed_by_stage[stage],
            )
            for stage in PerceptionStage
        )

    def consume(
        self,
        *,
        context: PerceptionContextManifest,
        media_health: MediaHealthReport,
    ) -> PerceptionContextOutcome:
        """Observe and consume one scanned non-overlapping focus context."""

        if self._finalized:
            raise RuntimeError("cannot consume after perception session finalization")
        self._validate_next_input(context, media_health)
        observation, observation_elapsed_seconds = self._pipeline.observe_context(context)
        projected = self._project_precomputed(
            context=context,
            media_health=media_health,
            observation=observation,
            observation_elapsed_seconds=observation_elapsed_seconds,
            validate_input=False,
        )
        return self.reconcile_projected(projected)

    def consume_precomputed(
        self,
        *,
        context: PerceptionContextManifest,
        media_health: MediaHealthReport,
        observation: MageObservation,
        observation_elapsed_seconds: float,
    ) -> PerceptionContextOutcome:
        """Consume an ordered observation generated by a bounded in-flight worker."""

        projected = self.project_precomputed(
            context=context,
            media_health=media_health,
            observation=observation,
            observation_elapsed_seconds=observation_elapsed_seconds,
        )
        return self.reconcile_projected(projected)

    def project_precomputed(
        self,
        *,
        context: PerceptionContextManifest,
        media_health: MediaHealthReport,
        observation: MageObservation,
        observation_elapsed_seconds: float,
    ) -> PerceptionProjectedContext:
        """Project one accepted observation without advancing temporal track state."""

        return self._project_precomputed(
            context=context,
            media_health=media_health,
            observation=observation,
            observation_elapsed_seconds=observation_elapsed_seconds,
            validate_input=True,
        )

    def _project_precomputed(
        self,
        *,
        context: PerceptionContextManifest,
        media_health: MediaHealthReport,
        observation: MageObservation,
        observation_elapsed_seconds: float,
        validate_input: bool,
    ) -> PerceptionProjectedContext:
        if self._finalized:
            raise RuntimeError("cannot project after perception session finalization")
        if self._pending_projection is not None:
            raise RuntimeError("cannot project another context before temporal reconciliation")
        if validate_input:
            self._validate_next_input(context, media_health)
        if (
            not isinstance(observation_elapsed_seconds, float)
            or not math.isfinite(observation_elapsed_seconds)
            or observation_elapsed_seconds < 0
        ):
            raise ValueError("observation_elapsed_seconds must be a nonnegative finite float")
        if observation.context.context_manifest_semantic_sha256 != (
            context.context_manifest_semantic_sha256
        ):
            raise ValueError("provider returned an observation for another context")
        self._elapsed_by_stage[PerceptionStage.PERCEPTION_OBSERVE] += observation_elapsed_seconds
        self._count_by_stage[PerceptionStage.PERCEPTION_OBSERVE] += 1

        def project_observation() -> tuple[QaProjection, EventProjection, EvidenceProjection]:
            qa_projection = self._pipeline._qa_projector.project(observation, media_health)
            event_projection = self._pipeline._event_projector.project(observation)
            evidence_projection = self._pipeline._evidence_projector.project(
                observation, event_projection
            )
            return qa_projection, event_projection, evidence_projection

        qa_projection, event_projection, evidence_projection = self._pipeline._timed(
            PerceptionStage.OBSERVATION_PROJECT,
            project_observation,
            self._elapsed_by_stage,
            self._count_by_stage,
        )
        projected = PerceptionProjectedContext(
            context=context,
            observation=observation,
            media_health=media_health,
            qa_projection=qa_projection,
            event_projection=event_projection,
            evidence_projection=evidence_projection,
        )
        self._pending_projection = projected
        return projected

    def reconcile_projected(
        self, projected: PerceptionProjectedContext
    ) -> PerceptionContextOutcome:
        """Causally reduce one previously projected context into event tracks."""

        if self._finalized:
            raise RuntimeError("cannot reconcile after perception session finalization")
        if not isinstance(projected, PerceptionProjectedContext):
            raise TypeError("projected must be PerceptionProjectedContext")
        if projected is not self._pending_projection:
            raise RuntimeError("temporal reconciliation requires the pending projection")
        self._qa_history.append(projected.qa_projection)
        self._evidence_history.append(projected.evidence_projection)

        def reconcile_context() -> TemporalReconcileResult:
            return self._pipeline._reconciler.reconcile(self._tracks, projected.event_projection)

        reconcile = self._pipeline._timed(
            PerceptionStage.TEMPORAL_RECONCILE,
            reconcile_context,
            self._elapsed_by_stage,
            self._count_by_stage,
        )
        self._tracks = reconcile.current_tracks
        for track_key in reconcile.closed_track_keys:
            track = next(item for item in self._tracks if item.event_track_key == track_key)
            self._fused_by_track[track_key] = self._fuse(track)

        outcome = PerceptionContextOutcome(
            context=projected.context,
            observation=projected.observation,
            media_health=projected.media_health,
            qa_projection=projected.qa_projection,
            event_projection=projected.event_projection,
            evidence_projection=projected.evidence_projection,
            temporal_reconcile=reconcile,
        )
        self._outcomes.append(outcome)
        self._pipeline._persist_outcome(outcome)
        self._pending_projection = None
        return outcome

    def finalize(self) -> StreamPerceptionRunResult:
        """Close any live track and return immutable measurements and artifacts."""

        if self._finalized:
            raise RuntimeError("perception session has already been finalized")
        if self._pending_projection is not None:
            raise RuntimeError("cannot finalize before pending temporal reconciliation")
        if not self._outcomes:
            raise ValueError("stream perception requires at least one context")
        self._finalized = True
        finalization_started = time.perf_counter()
        final_tracks: list[EventTrackRevision] = []
        with runtime_span(
            self._pipeline._runtime_observer,
            _PERCEPTION_STAGE_SPAN_NAMES[PerceptionStage.FINALIZE],
            {"pipeline_policy_version": PERCEPTION_PIPELINE_POLICY_VERSION},
        ):
            for track in self._tracks:
                if track.state in {
                    EventTrackState.CANDIDATE,
                    EventTrackState.OPEN,
                    EventTrackState.UPDATED,
                }:
                    track = close_event_track(track)
                final_tracks.append(track)
                if (
                    track.state is EventTrackState.CLOSED
                    and track.event_track_key not in self._fused_by_track
                ):
                    self._fused_by_track[track.event_track_key] = self._fuse(track)
        self._elapsed_by_stage[PerceptionStage.FINALIZE] += (
            time.perf_counter() - finalization_started
        )
        self._count_by_stage[PerceptionStage.FINALIZE] += 1

        decisions = tuple(
            sorted(self._fused_by_track.values(), key=lambda item: item.source_event_track_key)
        )
        final_tracks_tuple = tuple(sorted(final_tracks, key=lambda item: item.event_track_key))
        requests = tuple(
            request
            for decision in decisions
            for request in self._pipeline._refine_requests(decision, final_tracks_tuple)
        )
        # Refinement remains durable exception work.  This local consumer never
        # makes an implicit second model call merely because a decision is ambiguous.
        self._count_by_stage[PerceptionStage.PERCEPTION_REFINE] = 0
        measurements = tuple(
            PerceptionStageMeasurement(
                stage=stage,
                invocation_count=self._count_by_stage[stage],
                elapsed_seconds=self._elapsed_by_stage[stage],
            )
            for stage in PerceptionStage
        )
        terminal_artifacts = self._pipeline._persist_terminal_artifacts(
            contexts=tuple(self._outcomes),
            tracks=final_tracks_tuple,
            decisions=decisions,
            requests=requests,
            measurements=measurements,
        )
        return StreamPerceptionRunResult(
            contexts=tuple(self._outcomes),
            event_tracks=final_tracks_tuple,
            fusion_decisions=decisions,
            refine_requests=requests,
            stage_measurements=measurements,
            normal_model_call_count=len(self._outcomes),
            refinement_model_call_count=0,
            terminal_artifacts=terminal_artifacts,
        )

    def _fuse(self, track: EventTrackRevision) -> PerceptionFusionDecision:
        def fuse_closed_track() -> PerceptionFusionDecision:
            return self._pipeline._fusion_engine.fuse(
                track,
                evidence_projections=tuple(self._evidence_history),
                qa_projections=tuple(self._qa_history),
            )

        return self._pipeline._timed(
            PerceptionStage.FUSION,
            fuse_closed_track,
            self._elapsed_by_stage,
            self._count_by_stage,
        )

    def _validate_next_input(
        self,
        context: PerceptionContextManifest,
        media_health: MediaHealthReport,
    ) -> None:
        if not isinstance(context, PerceptionContextManifest):
            raise TypeError("context must be PerceptionContextManifest")
        if not isinstance(media_health, MediaHealthReport):
            raise TypeError("media_health must be MediaHealthReport")
        if media_health.context_manifest_semantic_sha256 != (
            context.context_manifest_semantic_sha256
        ):
            raise ValueError("media health belongs to another perception context")
        if self._source_recording_key is None:
            self._source_recording_key = context.source_recording_key
            self._source_recording_exact_sha256 = context.source_recording_exact_sha256
            self._context_policy_version = context.context_policy_version
            self._codec_policy_version = context.codec_policy_version
        elif (
            context.source_recording_key != self._source_recording_key
            or context.source_recording_exact_sha256 != self._source_recording_exact_sha256
            or context.context_policy_version != self._context_policy_version
            or context.codec_policy_version != self._codec_policy_version
        ):
            raise ValueError("perception contexts must belong to one recording and policy")
        current = context.ordered_segments[-1]
        current_segment = (
            current.interval.start_ns,
            current.interval.end_ns,
            current.segment_semantic_sha256,
        )
        prior = self._prior_current_segment
        if prior is not None:
            if current_segment[0] < prior[0]:
                raise ValueError("perception contexts must be ordered by current storage segment")
            if current_segment[2] == prior[2]:
                raise ValueError("one storage segment cannot trigger perception twice")
            if prior[1] > current_segment[0]:
                raise ValueError("current perception segments must not overlap")
        self._prior_current_segment = current_segment


class StreamPerceptionPipeline:
    """Run the vNext graph with exactly one normal model call per context."""

    def __init__(
        self,
        *,
        provider: MageObservationProvider,
        qa_projector: QaProjector,
        event_projector: EventProjector,
        evidence_projector: EvidenceProjector,
        reconciler: EventTrackReconciler,
        fusion_engine: PerceptionFusionEngine,
        refine_policy_version: str,
        refine_prompt_version: str,
        artifact_sink: PerceptionArtifactSink | None = None,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not refine_policy_version or not refine_prompt_version:
            raise ValueError("refinement policy and prompt versions must be nonempty")
        self._provider = provider
        self._qa_projector = qa_projector
        self._event_projector = event_projector
        self._evidence_projector = evidence_projector
        self._reconciler = reconciler
        self._fusion_engine = fusion_engine
        self._refine_policy_version = refine_policy_version
        self._refine_prompt_version = refine_prompt_version
        self._artifact_sink = artifact_sink
        self._runtime_observer = runtime_observer

    def observe_context(self, context: PerceptionContextManifest) -> tuple[MageObservation, float]:
        """Run one provider call with a real span, returning its isolated elapsed time."""

        if not isinstance(context, PerceptionContextManifest):
            raise TypeError("context must be PerceptionContextManifest")
        started = time.perf_counter()
        try:
            with runtime_span(
                self._runtime_observer,
                _PERCEPTION_STAGE_SPAN_NAMES[PerceptionStage.PERCEPTION_OBSERVE],
                {
                    "pipeline_policy_version": PERCEPTION_PIPELINE_POLICY_VERSION,
                    "stage": PerceptionStage.PERCEPTION_OBSERVE.value,
                    "context_manifest_semantic_sha256": (context.context_manifest_semantic_sha256),
                },
            ):
                observation = self._provider.observe(context)
        finally:
            elapsed_seconds = time.perf_counter() - started
        return observation, elapsed_seconds

    def open_session(self) -> StreamPerceptionSession:
        """Open a single-worker producer-consumer session with backpressure one."""

        return StreamPerceptionSession(self)

    def run(
        self,
        *,
        contexts: Sequence[PerceptionContextManifest],
        media_health: Sequence[MediaHealthReport],
    ) -> StreamPerceptionRunResult:
        """Batch convenience wrapper over the same incremental consumer."""

        ordered_contexts = tuple(contexts)
        ordered_health = tuple(media_health)
        self._validate_inputs(ordered_contexts, ordered_health)
        session = self.open_session()
        for context, health in zip(ordered_contexts, ordered_health, strict=True):
            # A caller that supplies already-computed health still gets a real
            # measured handoff span; the local execution path computes health
            # inside this span before each observation.
            def existing_health(health: MediaHealthReport = health) -> MediaHealthReport:
                return health

            scanned_health = session.scan_media(existing_health)
            session.consume(context=context, media_health=scanned_health)
        return session.finalize()

    @staticmethod
    def _validate_inputs(
        contexts: tuple[PerceptionContextManifest, ...],
        media_health: tuple[MediaHealthReport, ...],
    ) -> None:
        if not contexts:
            raise ValueError("stream perception requires at least one context")
        if len(contexts) != len(media_health):
            raise ValueError("every context requires one media-health report")
        current_segments: list[tuple[int, int, str]] = []
        recording_identity: tuple[str, str] | None = None
        policy_identity: tuple[str, str] | None = None
        for context, health in zip(contexts, media_health, strict=True):
            if health.context_manifest_semantic_sha256 != (
                context.context_manifest_semantic_sha256
            ):
                raise ValueError("media health belongs to another perception context")
            current_recording = (
                context.source_recording_key,
                context.source_recording_exact_sha256,
            )
            current_policy = (
                context.context_policy_version,
                context.codec_policy_version,
            )
            if recording_identity is None:
                recording_identity = current_recording
                policy_identity = current_policy
            elif current_recording != recording_identity or current_policy != policy_identity:
                raise ValueError("perception contexts must belong to one recording and policy")
            current = context.ordered_segments[-1]
            current_segments.append(
                (
                    current.interval.start_ns,
                    current.interval.end_ns,
                    current.segment_semantic_sha256,
                )
            )
        if current_segments != sorted(current_segments):
            raise ValueError("perception contexts must be ordered by current storage segment")
        if len({item[2] for item in current_segments}) != len(current_segments):
            raise ValueError("one storage segment cannot trigger perception twice")
        for left, right in pairwise(current_segments):
            if left[1] > right[0]:
                raise ValueError("current perception segments must not overlap")
            if left[1] != right[0]:
                raise ValueError("current perception segments must form a contiguous partition")

    def _timed(
        self,
        stage: PerceptionStage,
        operation: Callable[[], _T],
        elapsed: dict[PerceptionStage, float],
        counts: dict[PerceptionStage, int],
    ) -> _T:
        started = time.perf_counter()
        try:
            with runtime_span(
                self._runtime_observer,
                _PERCEPTION_STAGE_SPAN_NAMES[stage],
                {
                    "pipeline_policy_version": PERCEPTION_PIPELINE_POLICY_VERSION,
                    "stage": stage.value,
                },
            ):
                return operation()
        finally:
            elapsed[stage] += time.perf_counter() - started
            counts[stage] += 1

    def _persist_outcome(self, outcome: PerceptionContextOutcome) -> None:
        if self._artifact_sink is None:
            return
        artifacts = (
            ("context", outcome.context.context_manifest_key, outcome.context),
            ("observation", outcome.observation.observation_logical_key, outcome.observation),
            ("media-health", outcome.media_health.media_health_key, outcome.media_health),
            ("qa", outcome.qa_projection.qa_projection_key, outcome.qa_projection),
            ("event", outcome.event_projection.event_projection_key, outcome.event_projection),
            (
                "evidence",
                outcome.evidence_projection.evidence_projection_key,
                outcome.evidence_projection,
            ),
            (
                "temporal-reconcile",
                outcome.temporal_reconcile.reconcile_key,
                outcome.temporal_reconcile,
            ),
        )
        for kind, logical_key, model in artifacts:
            self._artifact_sink.put(
                kind=kind,
                logical_key=logical_key,
                payload=canonical_json_bytes(model),
            )

    def _persist_terminal_artifacts(
        self,
        *,
        contexts: tuple[PerceptionContextOutcome, ...],
        tracks: tuple[EventTrackRevision, ...],
        decisions: tuple[PerceptionFusionDecision, ...],
        requests: tuple[PerceptionRefineRequest, ...],
        measurements: tuple[PerceptionStageMeasurement, ...],
    ) -> PerceptionTerminalArtifacts | None:
        if self._artifact_sink is None:
            return None

        track_references = tuple(
            LocalPerceptionArtifactReference(
                kind="event-track",
                logical_key=track.event_track_key,
                exact_sha256=self._artifact_sink.put(
                    kind="event-track",
                    logical_key=track.event_track_key,
                    payload=canonical_json_bytes(track),
                ),
            )
            for track in tracks
        )
        fusion_references = tuple(
            LocalPerceptionArtifactReference(
                kind="fusion-decision",
                logical_key=decision.fusion_key,
                exact_sha256=self._artifact_sink.put(
                    kind="fusion-decision",
                    logical_key=decision.fusion_key,
                    payload=canonical_json_bytes(decision),
                ),
            )
            for decision in decisions
        )
        refine_references = tuple(
            LocalPerceptionArtifactReference(
                kind="refine-request",
                logical_key=request.refine_request_key,
                exact_sha256=self._artifact_sink.put(
                    kind="refine-request",
                    logical_key=request.refine_request_key,
                    payload=canonical_json_bytes(request),
                ),
            )
            for request in requests
        )
        manifest_projection: dict[str, object] = {
            "manifest_version": PERCEPTION_TERMINAL_MANIFEST_VERSION,
            "pipeline_policy_version": PERCEPTION_PIPELINE_POLICY_VERSION,
            "contexts": [
                {
                    "context_manifest_key": outcome.context.context_manifest_key,
                    "context_manifest_semantic_sha256": (
                        outcome.context.context_manifest_semantic_sha256
                    ),
                    "observation_logical_key": outcome.observation.observation_logical_key,
                    "observation_semantic_sha256": (
                        outcome.observation.observation_semantic_sha256
                    ),
                    "inference_artifact_exact_sha256": (
                        outcome.observation.inference_artifact_exact_sha256
                    ),
                    "qa_projection_key": outcome.qa_projection.qa_projection_key,
                    "event_projection_key": outcome.event_projection.event_projection_key,
                    "evidence_projection_key": (
                        outcome.evidence_projection.evidence_projection_key
                    ),
                    "temporal_reconcile_key": outcome.temporal_reconcile.reconcile_key,
                }
                for outcome in contexts
            ],
            "event_tracks": [
                {
                    "kind": reference.kind,
                    "logical_key": reference.logical_key,
                    "exact_sha256": reference.exact_sha256,
                }
                for reference in track_references
            ],
            "fusion_decisions": [
                {
                    "kind": reference.kind,
                    "logical_key": reference.logical_key,
                    "exact_sha256": reference.exact_sha256,
                }
                for reference in fusion_references
            ],
            "refine_requests": [
                {
                    "kind": reference.kind,
                    "logical_key": reference.logical_key,
                    "exact_sha256": reference.exact_sha256,
                }
                for reference in refine_references
            ],
            "stage_invocation_counts": [
                {
                    "stage": measurement.stage.value,
                    "invocation_count": measurement.invocation_count,
                }
                for measurement in measurements
            ],
            "normal_model_call_count": len(contexts),
            "refinement_model_call_count": 0,
        }
        manifest_semantic_sha256 = semantic_sha256(manifest_projection)
        manifest_logical_key = (
            f"{PERCEPTION_TERMINAL_MANIFEST_KEY_NAMESPACE}:{manifest_semantic_sha256}"
        )
        manifest_exact_sha256 = self._artifact_sink.put(
            kind="perception-terminal-manifest",
            logical_key=manifest_logical_key,
            payload=canonical_json_bytes(manifest_projection),
        )
        return PerceptionTerminalArtifacts(
            event_tracks=track_references,
            fusion_decisions=fusion_references,
            refine_requests=refine_references,
            terminal_manifest=LocalPerceptionArtifactReference(
                kind="perception-terminal-manifest",
                logical_key=manifest_logical_key,
                exact_sha256=manifest_exact_sha256,
            ),
        )

    def _refine_requests(
        self,
        decision: PerceptionFusionDecision,
        tracks: tuple[EventTrackRevision, ...],
    ) -> tuple[PerceptionRefineRequest, ...]:
        if not decision.requires_refinement:
            return ()
        track = next(
            item for item in tracks if item.event_track_key == decision.source_event_track_key
        )
        target = track.source_hypotheses[-1]
        requests: list[PerceptionRefineRequest] = []
        for reason in sorted(decision.refine_reasons, key=lambda item: item.value):
            fields: set[RefineTargetField] = set()
            if reason is RefineReason.BOUNDARY:
                if FusionAmbiguity.START_BOUNDARY_UNCERTAIN in decision.ambiguity_reasons:
                    fields.add(RefineTargetField.START_BOUNDARY)
                if FusionAmbiguity.END_BOUNDARY_UNCERTAIN in decision.ambiguity_reasons:
                    fields.add(RefineTargetField.END_BOUNDARY)
            elif reason is RefineReason.CONFLICT:
                fields.add(RefineTargetField.CAMERA_RELATION)
            elif reason is RefineReason.QA:
                fields.add(RefineTargetField.SEMANTIC_QA)
            elif reason is RefineReason.LABEL:
                fields.add(RefineTargetField.ACTION_LABEL)
            if not fields:
                continue
            requests.append(
                create_perception_refine_request(
                    source_observation_logical_key=target.source_observation_logical_key,
                    source_observation_semantic_sha256=target.source_observation_semantic_sha256,
                    target_hypothesis_logical_key=target.hypothesis_logical_key,
                    target_hypothesis_semantic_sha256=target.hypothesis_semantic_sha256,
                    reason=reason,
                    target_fields=tuple(sorted(fields, key=lambda item: item.value)),
                    refine_interval=track.interval,
                    refine_policy_version=self._refine_policy_version,
                    prompt_version=self._refine_prompt_version,
                )
            )
        return tuple(requests)


__all__ = [
    "LOCAL_PERCEPTION_ARTIFACT_INDEX_VERSION",
    "PERCEPTION_PIPELINE_POLICY_VERSION",
    "PERCEPTION_TERMINAL_MANIFEST_KEY_NAMESPACE",
    "PERCEPTION_TERMINAL_MANIFEST_VERSION",
    "LocalPerceptionArtifactReference",
    "LocalPerceptionArtifactStore",
    "MageObservationProvider",
    "PerceptionArtifactSink",
    "PerceptionContextOutcome",
    "PerceptionProjectedContext",
    "PerceptionStage",
    "PerceptionStageMeasurement",
    "PerceptionTerminalArtifacts",
    "StreamPerceptionPipeline",
    "StreamPerceptionRunResult",
    "StreamPerceptionSession",
]
