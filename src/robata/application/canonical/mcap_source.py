"""Real MCAP input bridge for the local canonical composition."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Final
from uuid import NAMESPACE_URL, uuid5

import av
from pydantic import ValidationError

from robata.adapters.local_artifact_registry import LocalArtifactRegistry
from robata.adapters.mcap_inspector import McapPreflight, OfficialMcapInspector
from robata.adapters.mcap_single_pass import H264PacketEnvelope, iter_h264_spool
from robata.adapters.pyav_frame_materializer import (
    _CameraLedger,
    _encode_png,
    _frame_pts_ns,
    _load_camera_ledger,
    _validate_publication,
)
from robata.adapters.pyav_mp4_exporter import (
    EXPORT_CONFIG,
    EXPORT_PROFILE_ID,
    EXPORT_PROFILE_VERSION,
    EXPORTER_NAME,
    EXPORTER_VERSION,
    PyAvH264Mp4Exporter,
    is_independent_h264_bootstrap,
)
from robata.adapters.sqlite_capture_authority import SQLiteLocalCaptureAuthority
from robata.adapters.sqlite_inference_evidence import MODEL_INFERENCE_SCHEMA_ID
from robata.adapters.sqlite_stream_delivery import SQLiteStreamDeliveryAuthority
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler
from robata.admission.context import AdmissionContextResolver, AdmittedRecordingContextV2
from robata.admission.ledger import (
    AlignmentAdmissionOutcome,
    PrimaryAdmissionEvaluation,
    PrimaryAdmissionPolicy,
    SourceAdmissionOutcome,
)
from robata.application.canonical.bounded_media import (
    BoundedMediaPolicy,
    PlannerEmission,
    PlannerFinish,
)
from robata.application.canonical.local_stream_finalization import (
    LOCAL_STREAM_WORK_RECEIPT_SCHEMA_ID,
    LOCAL_STREAM_WORK_RECEIPT_SCHEMA_VERSION,
    LocalConformanceStreamFinalizer,
    LocalStreamFinalizationSchemaRefs,
)
from robata.application.canonical.media_quality import (
    DEFAULT_MEDIA_QUALITY_POLICY,
    FrameQualityObservation,
    FrameTimingEvidence,
    LocalFrameQualityAnalyzer,
    LocalMediaQualityPolicy,
    LocalMediaQualityReport,
    build_local_media_quality_report,
    pyav_decoded_frame_view,
    registered_local_media_quality_report_document,
)
from robata.application.canonical.single_pass_video import (
    DurableSinglePassVideoProducer,
    H264SpoolSetFacts,
    read_sealed_mcap_inspection,
)
from robata.application.canonical.stream_recording_reduction import (
    LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
    LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION,
)
from robata.application.canonical.stream_scheduler import (
    DEFAULT_STREAM_BACKPRESSURE_CONFIG,
    STREAM_WINDOW_DAG_POLICY_VERSION,
    DurableStreamWindowScheduler,
    EosSealInputs,
    StreamSchedulerSchemaRefs,
)
from robata.application.registered_video_export import (
    PublishedRegisteredVideoExport,
    RegisteredSixCameraVideoExportService,
)
from robata.application.video_export import LocalVideoExportRequest, VideoExporterDescriptor
from robata.contracts.admission_v2 import (
    ADMISSION_EVIDENCE_V2_SCHEMA_VERSION,
    ALIGNMENT_MANIFEST_V2_SCHEMA_ID,
    MCAP_READY_MANIFEST_V2_SCHEMA_ID,
    MCAP_VALIDATION_REPORT_V2_SCHEMA_ID,
    AlignmentManifestV2,
    CameraAlignmentV2,
    DecoderProbeEvidenceV2,
    DecoderProbeOutcome,
    EvidenceComponent,
    MCAPReadyCameraV2,
    MCAPReadyManifestV2,
    MCAPReadySourceV2,
    MCAPValidationReportV2,
    MCAPValidationSourceV2,
    MCAPValidationVerdictV2,
    ProbedVideoStreamFactV2,
    SchemaSupportStatus,
    SemanticPolicyReference,
    SourceDurabilityEvidenceV2,
    StreamSchemaEvidenceV2,
    ValidationCameraMappingV2,
    ValidationCheckEvidenceV2,
    ValidationCheckOutcome,
    alignment_manifest_v2_semantic_projection,
    compute_camera_mapping_semantic_sha256_v2,
    compute_stream_semantic_sha256_v2,
    mcap_ready_manifest_v2_semantic_projection,
    mcap_validation_report_v2_semantic_projection,
)
from robata.contracts.alignment import (
    AlignmentMethod,
    AlignmentSegment,
    AlignmentStatus,
    CanonicalOrigin,
)
from robata.contracts.artifacts import ArtifactType
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import (
    canonical_json_bytes,
    exact_bytes_sha256,
    recording_identity,
    semantic_sha256,
)
from robata.contracts.local_stream_causal import (
    LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_ID,
    LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_VERSION,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_ID,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_VERSION,
)
from robata.contracts.mcap import MCAPReadyRecording
from robata.contracts.sampling_plan import FrameBudget, OverflowPolicy, SamplingPlan
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.contracts.stream_common import (
    AuthorityBinding,
    ChannelBinding,
    StreamPolicyBinding,
    StreamPurpose,
)
from robata.contracts.stream_finalization import (
    RECORDING_FINALIZATION_SCHEMA_ID,
    RECORDING_FINALIZATION_SCHEMA_VERSION,
    WINDOW_TERMINAL_CLOSURE_SCHEMA_ID,
    WINDOW_TERMINAL_CLOSURE_SCHEMA_VERSION,
    WINDOW_TERMINAL_MEMBER_SCHEMA_ID,
    WINDOW_TERMINAL_MEMBER_SCHEMA_VERSION,
)
from robata.contracts.stream_inference import (
    STREAM_ACCEPTED_CALL_SCHEMA_ID,
    STREAM_ACCEPTED_CALL_SCHEMA_VERSION,
    STREAM_INFERENCE_INTENT_SCHEMA_ID,
    STREAM_INFERENCE_INTENT_SCHEMA_VERSION,
    STREAM_INFERENCE_TERMINAL_SCHEMA_ID,
    STREAM_INFERENCE_TERMINAL_SCHEMA_VERSION,
    STREAM_WINDOW_RESULT_SCHEMA_ID,
    STREAM_WINDOW_RESULT_SCHEMA_VERSION,
)
from robata.contracts.stream_planning import (
    EXPECTED_WINDOW_DECLARATION_SCHEMA_ID,
    EXPECTED_WINDOW_DECLARATION_SCHEMA_VERSION,
    EXPECTED_WINDOW_PLAN_SCHEMA_ID,
    EXPECTED_WINDOW_PLAN_SCHEMA_VERSION,
    EXPECTED_WINDOW_SEAL_SCHEMA_ID,
    EXPECTED_WINDOW_SEAL_SCHEMA_VERSION,
    STREAM_WORK_PLAN_SCHEMA_ID,
    STREAM_WORK_PLAN_SCHEMA_VERSION,
    StreamWorkItemPlan,
    create_expected_window_plan,
)
from robata.contracts.stream_source import (
    PRE_EOS_CAPTURE_SCHEMA_ID,
    PRE_EOS_CAPTURE_SCHEMA_VERSION,
    PreEosCaptureSubject,
)
from robata.contracts.stream_window import (
    INCREMENTAL_WINDOW_SCHEMA_ID,
    INCREMENTAL_WINDOW_SCHEMA_VERSION,
    STREAM_INFERENCE_ATTEMPT_SCHEMA_ID,
    STREAM_INFERENCE_ATTEMPT_SCHEMA_VERSION,
    STREAM_INFERENCE_SCHEMA_ID,
    STREAM_INFERENCE_SCHEMA_VERSION,
)
from robata.contracts.video_export import VideoExporterMode
from robata.ingestion.mapping import ExactTopicMappingPolicy, TopicMappingProfile
from robata.ports.ingestion import ChannelInspection, McapInspection
from robata.queue.outbox import OutboxRetryPolicy
from robata.queue.stream_models import StreamTerminalEvidence
from robata.runtime.observability import RuntimeObserver, runtime_increment, runtime_span
from robata.sampling.grid import FrameCandidate, SamplingGrid, SamplingRate, SelectionStatus
from robata.sampling.materializer import (
    CameraSourceFrameIndex,
    CanonicalSixCameraFrameIndex,
    FrameAlignmentProjectionFact,
    IndexedSourceFrame,
    MaterializedArtifactManifest,
    MaterializedFrameArtifactFact,
)
from robata.tempfiles import make_temp_file

MCAP_SOURCE_POLICY_VERSION = "canonical-development-mcap-v1"
MCAP_SAMPLING_POLICY_VERSION = "canonical-development-sampling-v1"
MCAP_RECORDING_NAMESPACE = "robata-canonical-development-mcap-v1"
MCAP_PNG_EXTRACTOR_VERSION = "canonical-mcap-png-320-v1"
MCAP_PNG_MAX_WIDTH: Final = 320
MCAP_MEDIA_PROCESSING_POLICY_VERSION: Final = "canonical-mcap-media-processing-v1"
MCAP_MEDIA_EVIDENCE_ENCODING: Final = "png"
MCAP_BASE_SENTINEL_RATE_NUMERATOR: Final = 2
MCAP_BASE_SENTINEL_RATE_DENOMINATOR: Final = 1
MCAP_TARGET_SELECTION_TOLERANCE_NS: Final = 300_000_000
MCAP_PRE_EOS_POLICY_VERSION: Final = "canonical-mcap-pre-eos-v1"
MCAP_PRE_EOS_CLOCK_POLICY_VERSION: Final = "canonical-mcap-log-time-clock-v1"
MCAP_CAPTURE_AUTHORITY_ID: Final = "local-mcap-capture-authority"
MCAP_CAPTURE_ASSIGNMENT_POLICY_VERSION: Final = "local-mcap-capture-assignment-v1"
MCAP_EXPECTED_PLAN_PLANNER_VERSION: Final = "bounded-single-pass-media-planner-v1"
MCAP_WATERMARK_POLICY_VERSION: Final = "canonical-mcap-watermark-v1"
MCAP_LATENESS_POLICY_VERSION: Final = "canonical-mcap-lateness-v1"
MCAP_IDLE_SOURCE_POLICY_VERSION: Final = "canonical-mcap-idle-source-v1"
MCAP_SPOOL_EXPORT_WORKERS: Final = 6
MCAP_INCREMENTAL_DRAIN_BATCH_SIZE: Final = 256


@dataclass(slots=True)
class _IncrementalLocalStreamPlanningSink:
    """Persist planner output and keep provider-neutral window work bounded."""

    scheduler: DurableStreamWindowScheduler
    executor: LocalConformanceStreamFinalizer
    runtime_observer: RuntimeObserver | None

    def append_emission(self, emission: PlannerEmission) -> None:
        if not emission.windows:
            return
        # The planner owns packet/timeline cursors in memory. The durable stream
        # scheduler only owns emitted windows, so invoking it for every source
        # packet would turn an otherwise event-driven capture path back into a
        # message-proportional scheduling path.
        self.scheduler.append_emission(emission)
        completed = self.executor.drain_ready(max_items=max(1, len(emission.windows) * 5))
        runtime_increment(
            self.runtime_observer,
            "stream.incremental.executed_work",
            completed,
        )

    def seal(self, finish: PlannerFinish) -> None:
        self.scheduler.seal(finish)
        while completed := self.executor.drain_ready(max_items=MCAP_INCREMENTAL_DRAIN_BATCH_SIZE):
            runtime_increment(
                self.runtime_observer,
                "stream.incremental.executed_work",
                completed,
            )


MCAP_FRAME_MATERIALIZATION_WORKERS: Final = 6


class CanonicalMcapSourceError(ValueError):
    """A raw MCAP cannot produce complete local canonical inputs."""


@dataclass(frozen=True, slots=True)
class McapMediaProcessingPolicy:
    """Bounded local visual processing policy, kept outside the published wire report.

    The packet/sidecar timeline remains complete and authoritative. This policy only
    decides which decoded frames become inexpensive visual sentinels and which selected
    frames are rendered as retained evidence. Its complete projection is bound into the
    local source-run identity by ``local_composition``.
    """

    version: str = MCAP_MEDIA_PROCESSING_POLICY_VERSION
    semantic_rate_numerator: int = 2
    semantic_rate_denominator: int = 1
    sentinel_rate_numerator: int = MCAP_BASE_SENTINEL_RATE_NUMERATOR
    sentinel_rate_denominator: int = MCAP_BASE_SENTINEL_RATE_DENOMINATOR
    selection_tolerance_ns: int = MCAP_TARGET_SELECTION_TOLERANCE_NS
    sentinel_analysis_width: int = 64
    evidence_encoding: str = MCAP_MEDIA_EVIDENCE_ENCODING
    evidence_max_width: int = MCAP_PNG_MAX_WIDTH
    evidence_extractor_version: str = MCAP_PNG_EXTRACTOR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("media processing policy version must be a non-empty string")
        positive_fields = (
            self.semantic_rate_numerator,
            self.semantic_rate_denominator,
            self.sentinel_rate_numerator,
            self.sentinel_rate_denominator,
            self.selection_tolerance_ns,
            self.sentinel_analysis_width,
            self.evidence_max_width,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive_fields
        ):
            raise ValueError("media processing policy numeric fields must be positive integers")
        if self.evidence_encoding != MCAP_MEDIA_EVIDENCE_ENCODING:
            raise ValueError(f"unsupported media evidence encoding: {self.evidence_encoding!r}")
        if (
            not isinstance(self.evidence_extractor_version, str)
            or not self.evidence_extractor_version
        ):
            raise ValueError("evidence extractor version must be a non-empty string")


DEFAULT_MCAP_MEDIA_PROCESSING_POLICY: Final = McapMediaProcessingPolicy()


def mcap_media_processing_policy_projection(
    policy: McapMediaProcessingPolicy,
) -> dict[str, str | int]:
    """Return the complete deterministic policy projection used for source identity."""

    if not isinstance(policy, McapMediaProcessingPolicy):
        raise TypeError("policy must be a McapMediaProcessingPolicy")
    return {
        "version": policy.version,
        "semantic_rate_numerator": policy.semantic_rate_numerator,
        "semantic_rate_denominator": policy.semantic_rate_denominator,
        "sentinel_rate_numerator": policy.sentinel_rate_numerator,
        "sentinel_rate_denominator": policy.sentinel_rate_denominator,
        "selection_tolerance_ns": policy.selection_tolerance_ns,
        "sentinel_analysis_width": policy.sentinel_analysis_width,
        "evidence_encoding": policy.evidence_encoding,
        "evidence_max_width": policy.evidence_max_width,
        "evidence_extractor_version": policy.evidence_extractor_version,
    }


@dataclass(frozen=True, slots=True)
class AuthorizedMcapMapping:
    """A mapping profile authorized before any source bytes are accessed."""

    profile: TopicMappingProfile
    policy: ExactTopicMappingPolicy

    @property
    def semantic_sha256(self) -> str:
        return self.profile.semantic_digest


@dataclass(frozen=True, slots=True)
class CanonicalMcapSourceBundle:
    """Canonical runner inputs derived from one real MCAP."""

    source_content_sha256: str
    admitted_context: AdmittedRecordingContextV2
    requested_interval: NanosecondInterval
    sampling_plan: SamplingPlan
    frame_index: CanonicalSixCameraFrameIndex
    media_quality_report: LocalMediaQualityReport
    _artifacts: Mapping[tuple[CameraId, str], MaterializedFrameArtifactFact]
    _artifact_resolver: _VerifiedMcapArtifactResolver

    def resolve_artifact(
        self,
        camera_id: CameraId,
        frame: IndexedSourceFrame,
    ) -> MaterializedFrameArtifactFact | None:
        return self._artifact_resolver(camera_id, frame)


@dataclass(frozen=True, slots=True)
class _PreparedVideoPublication:
    publication: PublishedRegisteredVideoExport
    inspection: McapInspection
    channels: SixCameraMap[ChannelInspection]
    export_time_visual_results: _ExportTimeVisualResults | None


@dataclass(frozen=True, slots=True)
class _RenderedPngFact:
    path: Path
    sha256: str
    bytes: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _ExportTimeVisualObservation:
    timing: FrameTimingEvidence
    quality: FrameQualityObservation


@dataclass(frozen=True, slots=True)
class _ExportTimeVisualCameraResult:
    leading_access_unit_count: int
    semantic_source_orders: frozenset[int]
    sentinel_source_orders: frozenset[int]
    rendered: Mapping[int, _RenderedPngFact]
    observations: Mapping[int, _ExportTimeVisualObservation]


@dataclass(frozen=True, slots=True)
class _ExportTimeVisualResults:
    cameras: Mapping[CameraId, _ExportTimeVisualCameraResult]


@dataclass(slots=True)
class _CameraExportVisualCollector:
    camera_id: CameraId
    origin_ns: int
    leading_access_unit_count: int
    semantic_source_orders: frozenset[int]
    sentinel_source_orders: frozenset[int]
    output_root: Path
    quality_policy: LocalMediaQualityPolicy
    evidence_max_width: int
    evidence_extractor_version: str
    analyzer: LocalFrameQualityAnalyzer = field(init=False)
    rendered: dict[int, _RenderedPngFact] = field(default_factory=dict, init=False)
    observations: dict[int, _ExportTimeVisualObservation] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        self.analyzer = LocalFrameQualityAnalyzer(self.camera_id, policy=self.quality_policy)

    def observe(
        self,
        envelope: H264PacketEnvelope,
        decoded_frame: Any,
        exported_index: int,
    ) -> None:
        packet = envelope.packet
        if packet.camera_id is not self.camera_id:
            raise CanonicalMcapSourceError(
                f"{self.camera_id.value} export callback camera differs from its spool"
            )
        expected_exported_index = packet.source_order - self.leading_access_unit_count
        if expected_exported_index != exported_index:
            raise CanonicalMcapSourceError(
                f"{self.camera_id.value} export callback index differs from source order"
            )
        raw_order = packet.source_order
        if raw_order in self.semantic_source_orders:
            if raw_order in self.rendered:
                raise CanonicalMcapSourceError(
                    f"{self.camera_id.value} received a duplicate semantic frame callback"
                )
            png_bytes, width, height = _encode_png(
                decoded_frame,
                max_width=self.evidence_max_width,
            )
            digest = exact_bytes_sha256(png_bytes)
            path = _publish_png(self.output_root, digest, png_bytes)
            self.rendered[raw_order] = _RenderedPngFact(
                path=path,
                sha256=digest,
                bytes=len(png_bytes),
                width=width,
                height=height,
            )
        if raw_order in self.sentinel_source_orders:
            if raw_order in self.observations:
                raise CanonicalMcapSourceError(
                    f"{self.camera_id.value} received a duplicate sentinel frame callback"
                )
            timing = FrameTimingEvidence(
                camera_id=self.camera_id,
                packet_index=exported_index,
                aligned_timestamp_ns=packet.source_timestamp_ns - self.origin_ns,
                source_timestamp_ns=packet.source_timestamp_ns,
                source_sequence=packet.source_sequence,
            )
            quality = self.analyzer.observe(
                pyav_decoded_frame_view(
                    decoded_frame,
                    timestamp_ns=timing.aligned_timestamp_ns,
                    analysis_width=self.quality_policy.analysis_width,
                ),
                timing,
            )
            self.observations[raw_order] = _ExportTimeVisualObservation(
                timing=timing,
                quality=quality,
            )

    def finish(self) -> _ExportTimeVisualCameraResult:
        if set(self.rendered) != set(self.semantic_source_orders):
            raise CanonicalMcapSourceError(
                f"{self.camera_id.value} export did not render every selected semantic frame"
            )
        if set(self.observations) != set(self.sentinel_source_orders):
            raise CanonicalMcapSourceError(
                f"{self.camera_id.value} export did not observe every selected sentinel frame"
            )
        return _ExportTimeVisualCameraResult(
            leading_access_unit_count=self.leading_access_unit_count,
            semantic_source_orders=self.semantic_source_orders,
            sentinel_source_orders=self.sentinel_source_orders,
            rendered=MappingProxyType(dict(self.rendered)),
            observations=MappingProxyType(dict(self.observations)),
        )


@dataclass(slots=True)
class _ExportTimeVisualObserver:
    collectors: Mapping[CameraId, _CameraExportVisualCollector]

    def observe(
        self,
        envelope: H264PacketEnvelope,
        decoded_frame: Any,
        exported_index: int,
    ) -> None:
        collector = self.collectors.get(envelope.packet.camera_id)
        if collector is None:
            raise CanonicalMcapSourceError(
                "export callback camera is not part of the canonical camera set"
            )
        collector.observe(envelope, decoded_frame, exported_index)

    def finish(self) -> _ExportTimeVisualResults:
        return _ExportTimeVisualResults(
            cameras=MappingProxyType(
                {camera_id: self.collectors[camera_id].finish() for camera_id in CAMERA_IDS}
            )
        )


@dataclass(slots=True)
class _VerifiedMcapArtifactResolver:
    """Resolve selected frames against a previously verified MP4/sidecar ledger."""

    frame_index: CanonicalSixCameraFrameIndex
    ledgers: Mapping[CameraId, _CameraLedger]
    requested_interval: NanosecondInterval
    output_root: Path
    artifacts: dict[tuple[CameraId, str], MaterializedFrameArtifactFact]
    evidence_max_width: int
    evidence_extractor_version: str
    runtime_observer: RuntimeObserver | None = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _frames_by_id: dict[CameraId, dict[str, IndexedSourceFrame]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._frames_by_id = {
            camera_id: {
                frame.source_frame_id: frame for frame in self.frame_index.cameras[camera_id].frames
            }
            for camera_id in CAMERA_IDS
        }

    def __call__(
        self,
        camera_id: CameraId,
        frame: IndexedSourceFrame,
    ) -> MaterializedFrameArtifactFact | None:
        canonical = self._frames_by_id.get(camera_id, {}).get(frame.source_frame_id)
        if canonical is None:
            return None
        if canonical != frame:
            raise CanonicalMcapSourceError(
                f"{camera_id.value} artifact request differs from its canonical frame index"
            )
        aligned_ns = canonical.alignment_projection.aligned_timestamp_ns
        if not (self.requested_interval.start_ns <= aligned_ns < self.requested_interval.end_ns):
            return None

        key = (camera_id, canonical.source_frame_id)
        with self._lock:
            cached = self.artifacts.get(key)
            if cached is not None:
                runtime_increment(
                    self.runtime_observer,
                    "source.artifact_resolver.requests",
                    attributes={"cache": "HIT", "camera_id": camera_id.value},
                )
                return cached
            attributes = {"camera_id": camera_id.value}
            runtime_increment(
                self.runtime_observer,
                "source.artifact_resolver.requests",
                attributes={"cache": "MISS", **attributes},
            )
            with runtime_span(
                self.runtime_observer,
                "source.lazy_materialize",
                attributes,
            ):
                artifact = _materialize_verified_source_frame(
                    camera_id=camera_id,
                    ledger=self.ledgers[camera_id],
                    source_frame=canonical,
                    output_root=self.output_root,
                    evidence_max_width=self.evidence_max_width,
                    evidence_extractor_version=self.evidence_extractor_version,
                )
            self.artifacts[key] = artifact
            runtime_increment(
                self.runtime_observer,
                "source.lazy_materialized_artifacts",
                attributes=attributes,
            )
            return artifact


def authorize_mcap_mapping(
    mapping_config: Path,
    *,
    allow_unapproved_profile: bool,
) -> AuthorizedMcapMapping:
    """Authorize exact topic mapping without reading the MCAP source."""

    try:
        profile = TopicMappingProfile.load(Path(mapping_config))
        policy = ExactTopicMappingPolicy.from_profile(
            profile,
            allow_unapproved=allow_unapproved_profile,
        )
    except (OSError, TypeError, ValueError) as error:
        raise CanonicalMcapSourceError(f"invalid MCAP mapping authorization: {error}") from error
    return AuthorizedMcapMapping(profile=profile, policy=policy)


def load_canonical_mcap_source(
    source: Path,
    *,
    authorization: AuthorizedMcapMapping,
    state_dir: Path,
    expected_source_sha256: str,
    max_duration_ns: int | None = None,
    media_processing_policy: McapMediaProcessingPolicy | None = None,
    schema_registry: SchemaRegistry | None = None,
    clock: Callable[[], datetime] | None = None,
    runtime_observer: RuntimeObserver | None = None,
    execution_scheduler: SQLiteWorkScheduler | None = None,
    stream_run_id: str | None = None,
    stream_artifact_root: Path | None = None,
    stage_terminal_executor: (
        Callable[[StreamWorkItemPlan], StreamTerminalEvidence | None] | None
    ) = None,
    provider_terminal_required: bool = False,
) -> CanonicalMcapSourceBundle:
    """Inspect, export, admit, index, and materialize one real six-camera MCAP."""

    try:
        if max_duration_ns is not None:
            if isinstance(max_duration_ns, bool) or not isinstance(max_duration_ns, int):
                raise TypeError("max_duration_ns must be an integer or None")
            if max_duration_ns <= 0:
                raise ValueError("max_duration_ns must be positive")
        if (execution_scheduler is None) != (stream_run_id is None):
            raise ValueError("execution_scheduler and stream_run_id must be configured together")
        if stage_terminal_executor is not None and not callable(stage_terminal_executor):
            raise TypeError("stage_terminal_executor must be callable or None")
        if not isinstance(provider_terminal_required, bool):
            raise TypeError("provider_terminal_required must be bool")
        if stage_terminal_executor is not None and execution_scheduler is None:
            raise ValueError(
                "stage_terminal_executor requires execution_scheduler and stream_run_id"
            )
        resolved_media_processing_policy = (
            DEFAULT_MCAP_MEDIA_PROCESSING_POLICY
            if media_processing_policy is None
            else media_processing_policy
        )
        if not isinstance(resolved_media_processing_policy, McapMediaProcessingPolicy):
            raise TypeError("media_processing_policy must be a McapMediaProcessingPolicy or None")
        return _load_canonical_mcap_source(
            Path(source),
            authorization=authorization,
            state_dir=Path(state_dir),
            expected_source_sha256=expected_source_sha256,
            max_duration_ns=max_duration_ns,
            media_processing_policy=resolved_media_processing_policy,
            schema_registry=schema_registry or SchemaRegistry(),
            clock=clock or (lambda: datetime.now(tz=UTC)),
            runtime_observer=runtime_observer,
            execution_scheduler=execution_scheduler,
            stream_run_id=stream_run_id,
            stream_artifact_root=(
                None if stream_artifact_root is None else Path(stream_artifact_root)
            ),
            stage_terminal_executor=stage_terminal_executor,
            provider_terminal_required=provider_terminal_required,
        )
    except CanonicalMcapSourceError:
        raise
    except (OSError, RuntimeError, TypeError, ValidationError, ValueError) as error:
        raise CanonicalMcapSourceError(f"MCAP source preparation failed: {error}") from error


def _load_canonical_mcap_source(
    source: Path,
    *,
    authorization: AuthorizedMcapMapping,
    state_dir: Path,
    expected_source_sha256: str,
    max_duration_ns: int | None,
    media_processing_policy: McapMediaProcessingPolicy,
    schema_registry: SchemaRegistry,
    clock: Callable[[], datetime],
    runtime_observer: RuntimeObserver | None,
    execution_scheduler: SQLiteWorkScheduler | None,
    stream_run_id: str | None,
    stream_artifact_root: Path | None,
    stage_terminal_executor: (
        Callable[[StreamWorkItemPlan], StreamTerminalEvidence | None] | None
    ),
    provider_terminal_required: bool,
) -> CanonicalMcapSourceBundle:
    observed_at = _rfc3339(clock())
    stage_attributes = {"camera_count": len(CAMERA_IDS)}
    spool_directory = state_dir / "h264-spools"
    with runtime_span(runtime_observer, "source.inspect", stage_attributes):
        if spool_directory.exists():
            inspection = read_sealed_mcap_inspection(
                spool_directory,
                source=source,
                expected_source_sha256=expected_source_sha256,
            )
            preflight: McapPreflight | None = None
        else:
            preflight = OfficialMcapInspector().preflight(source)
            inspection = preflight.as_mapping_inspection(expected_source_sha256)
    if inspection.source_sha256 != expected_source_sha256:
        raise CanonicalMcapSourceError("source bytes changed after the run identity was derived")
    with runtime_span(runtime_observer, "source.mapping.resolve", stage_attributes):
        channels = authorization.policy.resolve(inspection)

    state_dir.mkdir(parents=True, exist_ok=True)
    with runtime_span(runtime_observer, "source.stream.capture_publish", stage_attributes):
        prepared_publication = _export_registered_videos(
            source=source,
            state_dir=state_dir,
            inspection=inspection,
            channels=channels,
            max_duration_ns=max_duration_ns,
            media_processing_policy=media_processing_policy,
            authorization=authorization,
            schema_registry=schema_registry,
            clock=clock,
            runtime_observer=runtime_observer,
            preflight=preflight,
            execution_scheduler=execution_scheduler,
            stream_run_id=stream_run_id,
            stream_artifact_root=stream_artifact_root,
            stage_terminal_executor=stage_terminal_executor,
            provider_terminal_required=provider_terminal_required,
        )
        publication = prepared_publication.publication
        inspection = prepared_publication.inspection
        channels = prepared_publication.channels
        export_time_visual_results = prepared_publication.export_time_visual_results
        fresh_media_handoff = not publication.derivation_reused
    if inspection.message_count > 0:
        runtime_increment(
            runtime_observer,
            "source.message_count",
            inspection.message_count,
            stage_attributes,
        )
    if inspection.first_message_time_ns is not None and inspection.last_message_time_ns is not None:
        source_span_duration_ns = inspection.last_message_time_ns - inspection.first_message_time_ns
        if source_span_duration_ns > 0:
            runtime_increment(
                runtime_observer,
                "source.span_duration_ns",
                source_span_duration_ns,
                stage_attributes,
            )
    with runtime_span(runtime_observer, "source.video.publication_validate", stage_attributes):
        view, manifest = _validate_publication(publication)
    with runtime_span(runtime_observer, "source.video.ledger_load", stage_attributes):
        ledgers = {
            record.camera_id: _load_camera_ledger(
                view,
                manifest,
                record,
                verify_video_digest=not fresh_media_handoff,
            )
            for record in manifest.cameras
        }
    with runtime_span(runtime_observer, "source.metadata.build", stage_attributes):
        mapping_policy = SemanticPolicyReference(
            version=authorization.profile.version,
            semantic_sha256=authorization.policy.semantic_digest,
        )
        stream_records = _build_stream_records(
            inspection=inspection,
            channels=channels,
            publication=publication,
        )
        camera_mappings = tuple(
            ValidationCameraMappingV2(
                camera_id=camera_id,
                role=camera_id.value,
                stream_id=stream_records[camera_id][0].stream_id,
                stream_semantic_sha256=stream_records[camera_id][0].stream_semantic_sha256,
            )
            for camera_id in CAMERA_IDS
        )
        mapping_digest = compute_camera_mapping_semantic_sha256_v2(
            source_content_sha256=inspection.source_sha256,
            mapping_policy=mapping_policy,
            camera_mappings=camera_mappings,
        )
        recording_id = recording_identity(MCAP_RECORDING_NAMESPACE, inspection.source_sha256)
        if manifest.recording_identity != recording_id:
            raise CanonicalMcapSourceError(
                "registered video export recording identity is inconsistent"
            )
        mcap_id = _stable_uuid("canonical-mcap-source", inspection.source_sha256)
        validation_report = _validation_report(
            schema_registry=schema_registry,
            inspection=inspection,
            recording_id=recording_id,
            mcap_id=mcap_id,
            mapping_policy=mapping_policy,
            mapping_digest=mapping_digest,
            stream_records=stream_records,
            camera_mappings=camera_mappings,
            observed_at=observed_at,
        )
        ready_manifest = _ready_manifest(
            schema_registry=schema_registry,
            inspection=inspection,
            validation_report=validation_report,
            mapping_policy=mapping_policy,
            publication=publication,
            channels=channels,
            ledgers=ledgers,
            observed_at=observed_at,
        )
        alignment_manifest = _alignment_manifest(
            schema_registry=schema_registry,
            ready_manifest=ready_manifest,
            ledgers=ledgers,
            observed_at=observed_at,
        )
        primary_policy = PrimaryAdmissionPolicy.create(
            version=MCAP_SOURCE_POLICY_VERSION,
            admissible_alignment_outcomes=(AlignmentAdmissionOutcome.VALID,),
        )
        evaluation = PrimaryAdmissionEvaluation(
            recording_identity=recording_id,
            ready_manifest_id=ready_manifest.ready_manifest_id,
            ready_manifest_semantic_sha256=ready_manifest.ready_manifest_semantic_sha256,
            source_outcome=SourceAdmissionOutcome.READY,
            alignment_outcome=AlignmentAdmissionOutcome.VALID,
            alignment_id=alignment_manifest.alignment_id,
            alignment_semantic_sha256=alignment_manifest.alignment_semantic_sha256,
            policy_version=primary_policy.version,
            policy_sha256=primary_policy.semantic_sha256,
            admissible=True,
            reason_code="ADMISSIBLE",
        )
        admitted_context = AdmissionContextResolver().resolve_v2(
            evaluation=evaluation,
            policy=primary_policy,
            validation_report=validation_report,
            ready_manifest=ready_manifest,
            alignment_manifest=alignment_manifest,
            registry=schema_registry,
        )
    runtime_increment(
        runtime_observer,
        "source.recording_duration_ns",
        ready_manifest.recording.duration_ns,
        stage_attributes,
    )
    with runtime_span(runtime_observer, "source.frame_index", stage_attributes):
        frame_index = _frame_index(
            context=admitted_context,
            stream_records=stream_records,
            ledgers=ledgers,
        )
    runtime_increment(
        runtime_observer,
        "source.frame_index.frames",
        sum(len(frame_index.cameras[camera_id].frames) for camera_id in CAMERA_IDS),
        stage_attributes,
    )
    with runtime_span(runtime_observer, "source.quality.timing", stage_attributes):
        quality_timings = _frame_timing_evidence(frame_index=frame_index, ledgers=ledgers)
    requested_end_ns = ready_manifest.recording.duration_ns
    if max_duration_ns is not None:
        requested_end_ns = min(requested_end_ns, max_duration_ns)
    requested_interval = NanosecondInterval(start_ns=0, end_ns=requested_end_ns)
    runtime_increment(
        runtime_observer,
        "source.requested_duration_ns",
        requested_interval.duration_ns,
        stage_attributes,
    )
    frame_output_root = state_dir / "frames"
    window_limited = requested_end_ns < ready_manifest.recording.duration_ns
    quality_policy = replace(
        DEFAULT_MEDIA_QUALITY_POLICY,
        analysis_width=media_processing_policy.sentinel_analysis_width,
    )
    materialize_attributes: dict[str, str | int | float | bool] = {
        **stage_attributes,
        "max_duration_limited": window_limited,
        "media_processing_policy_version": media_processing_policy.version,
        "semantic_target_rate": (
            f"{media_processing_policy.semantic_rate_numerator}/"
            f"{media_processing_policy.semantic_rate_denominator}"
        ),
        "sentinel_rate": (
            f"{media_processing_policy.sentinel_rate_numerator}/"
            f"{media_processing_policy.sentinel_rate_denominator}"
        ),
        "target_selection_tolerance_ns": media_processing_policy.selection_tolerance_ns,
        "sentinel_analysis_width": media_processing_policy.sentinel_analysis_width,
        "evidence_encoding": media_processing_policy.evidence_encoding,
        "evidence_max_width": media_processing_policy.evidence_max_width,
        "evidence_extractor_version": media_processing_policy.evidence_extractor_version,
    }
    with runtime_span(runtime_observer, "source.materialize", materialize_attributes):
        if export_time_visual_results is None:
            artifacts, quality_observations = _materialize_selected_frames(
                frame_index=frame_index,
                ledgers=ledgers,
                quality_timings=quality_timings,
                requested_interval=requested_interval,
                output_root=frame_output_root,
                media_processing_policy=media_processing_policy,
                quality_policy=quality_policy,
                stop_after_selected=window_limited,
            )
        else:
            artifacts, quality_observations = _bind_export_time_visual_results(
                frame_index=frame_index,
                ledgers=ledgers,
                quality_timings=quality_timings,
                requested_interval=requested_interval,
                output_root=frame_output_root,
                media_processing_policy=media_processing_policy,
                quality_policy=quality_policy,
                visual_results=export_time_visual_results,
            )
    if artifacts:
        runtime_increment(
            runtime_observer,
            "source.materialized_artifacts",
            len(artifacts),
            materialize_attributes,
        )
    frame_observation_count = sum(len(items) for items in quality_observations.values())
    if frame_observation_count > 0:
        runtime_increment(
            runtime_observer,
            "source.frame_observations",
            frame_observation_count,
            materialize_attributes,
        )
    with runtime_span(runtime_observer, "source.quality.report", materialize_attributes):
        media_quality_report = build_local_media_quality_report(
            requested_max_duration_ns=max_duration_ns,
            recording_duration_ns=ready_manifest.recording.duration_ns,
            requested_interval=requested_interval,
            timings=quality_timings,
            frame_observations=quality_observations,
            policy=quality_policy,
        )
    with runtime_span(runtime_observer, "source.quality.publish", materialize_attributes):
        _publish_exact_state_file(
            state_dir / "media-quality-report.json",
            canonical_json_bytes(
                registered_local_media_quality_report_document(
                    media_quality_report,
                    schema_registry,
                )
            ),
            label="media quality report",
        )
    artifact_resolver = _VerifiedMcapArtifactResolver(
        frame_index=frame_index,
        ledgers=MappingProxyType(dict(ledgers)),
        requested_interval=requested_interval,
        output_root=frame_output_root,
        artifacts=artifacts,
        evidence_max_width=media_processing_policy.evidence_max_width,
        evidence_extractor_version=media_processing_policy.evidence_extractor_version,
        runtime_observer=runtime_observer,
    )
    return CanonicalMcapSourceBundle(
        source_content_sha256=inspection.source_sha256,
        admitted_context=admitted_context,
        requested_interval=requested_interval,
        sampling_plan=_sampling_plan(),
        frame_index=frame_index,
        media_quality_report=media_quality_report,
        _artifacts=MappingProxyType(artifacts),
        _artifact_resolver=artifact_resolver,
    )


def _build_export_time_visual_observer(
    *,
    spool_set: H264SpoolSetFacts,
    final_channels: SixCameraMap[ChannelInspection],
    origin_ns: int,
    requested_interval: NanosecondInterval,
    output_root: Path,
    media_processing_policy: McapMediaProcessingPolicy,
    quality_policy: LocalMediaQualityPolicy,
) -> _ExportTimeVisualObserver:
    """Preselect source orders from sealed spools before the remux workers run."""

    if not isinstance(media_processing_policy, McapMediaProcessingPolicy):
        raise TypeError("media_processing_policy must be a McapMediaProcessingPolicy")
    if not isinstance(quality_policy, LocalMediaQualityPolicy):
        raise TypeError("quality_policy must be a LocalMediaQualityPolicy")
    if quality_policy.analysis_width != media_processing_policy.sentinel_analysis_width:
        raise ValueError("quality policy width must match the media sentinel analysis width")
    mapped_origin_ns, _ = _mapped_source_bounds(final_channels)
    if mapped_origin_ns != origin_ns:
        raise CanonicalMcapSourceError(
            "export visual origin differs from the final mapped channel origin"
        )
    collectors: dict[CameraId, _CameraExportVisualCollector] = {}
    semantic_rate = SamplingRate(
        media_processing_policy.semantic_rate_numerator,
        media_processing_policy.semantic_rate_denominator,
    )
    sentinel_rate = SamplingRate(
        media_processing_policy.sentinel_rate_numerator,
        media_processing_policy.sentinel_rate_denominator,
    )
    for camera_id in CAMERA_IDS:
        spool = spool_set.spools[camera_id]
        candidates: list[FrameCandidate] = []
        orders_by_locator: dict[bytes, int] = {}
        leading_count = 0
        bootstrap_found = False
        expected_source_order = 0
        for envelope in iter_h264_spool(spool.path):
            packet = envelope.packet
            if packet.camera_id is not camera_id or packet.source_order != expected_source_order:
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} spool source order is not contiguous"
                )
            expected_source_order += 1
            if not bootstrap_found:
                if not is_independent_h264_bootstrap(envelope.nal_types):
                    leading_count += 1
                    continue
                bootstrap_found = True
            locator = canonical_json_bytes(
                {
                    "camera_id": camera_id.value,
                    "source_order": packet.source_order,
                    "source_timestamp_ns": str(packet.source_timestamp_ns),
                }
            )
            if locator in orders_by_locator:
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} spool source locator is duplicated"
                )
            orders_by_locator[locator] = packet.source_order
            candidates.append(
                FrameCandidate(
                    aligned_timestamp_ns=packet.source_timestamp_ns - origin_ns,
                    source_timestamp_ns=packet.source_timestamp_ns,
                    source_locator_bytes=locator,
                    decodable=True,
                )
            )
        if expected_source_order != spool.packet_count:
            raise CanonicalMcapSourceError(
                f"{camera_id.value} spool packet count differs from its seal"
            )
        if not bootstrap_found:
            raise CanonicalMcapSourceError(
                f"{camera_id.value} spool has no independent H.264 bootstrap"
            )

        def selected_orders(
            rate: SamplingRate,
            *,
            candidates: list[FrameCandidate] = candidates,
            orders_by_locator: dict[bytes, int] = orders_by_locator,
        ) -> frozenset[int]:
            selected: set[int] = set()
            selections = SamplingGrid(grid_origin_ns=0, rate=rate).select_frames(
                candidates,
                requested_interval.start_ns,
                requested_interval.end_ns,
                media_processing_policy.selection_tolerance_ns,
            )
            for selection in selections:
                if selection.status is not SelectionStatus.SELECTED:
                    continue
                assert selection.frame is not None
                selected.add(orders_by_locator[selection.frame.source_locator_bytes])
            return frozenset(selected)

        collectors[camera_id] = _CameraExportVisualCollector(
            camera_id=camera_id,
            origin_ns=origin_ns,
            leading_access_unit_count=leading_count,
            semantic_source_orders=selected_orders(semantic_rate),
            sentinel_source_orders=selected_orders(sentinel_rate),
            output_root=output_root,
            quality_policy=quality_policy,
            evidence_max_width=media_processing_policy.evidence_max_width,
            evidence_extractor_version=media_processing_policy.evidence_extractor_version,
        )
    return _ExportTimeVisualObserver(collectors=MappingProxyType(collectors))


def _export_registered_videos(
    *,
    source: Path,
    state_dir: Path,
    inspection: McapInspection,
    channels: SixCameraMap[ChannelInspection],
    max_duration_ns: int | None,
    media_processing_policy: McapMediaProcessingPolicy,
    authorization: AuthorizedMcapMapping,
    schema_registry: SchemaRegistry,
    clock: Callable[[], datetime],
    runtime_observer: RuntimeObserver | None,
    preflight: McapPreflight | None,
    execution_scheduler: SQLiteWorkScheduler | None,
    stream_run_id: str | None,
    stream_artifact_root: Path | None,
    stage_terminal_executor: (
        Callable[[StreamWorkItemPlan], StreamTerminalEvidence | None] | None
    ),
    provider_terminal_required: bool,
) -> _PreparedVideoPublication:
    if authorization.profile.approved:
        raise CanonicalMcapSourceError(
            "the local registered exporter requires a development UNAPPROVED profile"
        )
    descriptor = VideoExporterDescriptor(
        name=EXPORTER_NAME,
        version=EXPORTER_VERSION,
        mode=VideoExporterMode.REMUX,
        export_profile_id=EXPORT_PROFILE_ID,
        profile_version=EXPORT_PROFILE_VERSION,
        canonical_config_sha256=semantic_sha256(EXPORT_CONFIG),
    )
    exporter = PyAvH264Mp4Exporter()
    service = RegisteredSixCameraVideoExportService(
        exporter,
        LocalArtifactRegistry(
            state_dir / "artifact-registry",
            runtime_observer=runtime_observer,
            hardlink_artifact_types=frozenset({ArtifactType.RAW_MCAP}),
        ),
        schema_registry,
        clock=clock,
    )
    planner_source_scope_digest = inspection.source_sha256
    stream_scheduler: DurableStreamWindowScheduler | None = None
    planning_sink: _IncrementalLocalStreamPlanningSink | None = None
    capture_subject: PreEosCaptureSubject | None = None
    if execution_scheduler is not None:
        if stream_run_id is None:
            raise CanonicalMcapSourceError("stream scheduler run identity is absent")
        capture_subject, stream_scheduler = _create_mcap_stream_scheduler(
            execution_scheduler=execution_scheduler,
            stream_run_id=stream_run_id,
            channels=channels,
            authorization=authorization,
            schema_registry=schema_registry,
            clock=clock,
        )
        planner_source_scope_digest = capture_subject.capture_scope_digest
        planning_sink = _IncrementalLocalStreamPlanningSink(
            scheduler=stream_scheduler,
            executor=LocalConformanceStreamFinalizer(
                scheduler=stream_scheduler,
                delivery_authority=SQLiteStreamDeliveryAuthority(
                    execution_scheduler,
                    retry_policy=OutboxRetryPolicy(
                        version="local-stream-delivery-retry-v1",
                        max_attempts=3,
                        base_delay_seconds=1,
                        max_delay_seconds=30,
                    ),
                    clock=clock,
                ),
                artifact_root=stream_artifact_root or state_dir / "stream-artifacts",
                schema_refs=LocalStreamFinalizationSchemaRefs(
                    local_work_receipt=schema_registry.resolve_version(
                        LOCAL_STREAM_WORK_RECEIPT_SCHEMA_ID,
                        LOCAL_STREAM_WORK_RECEIPT_SCHEMA_VERSION,
                    ).ref,
                    stream_window_result=schema_registry.resolve_version(
                        STREAM_WINDOW_RESULT_SCHEMA_ID,
                        STREAM_WINDOW_RESULT_SCHEMA_VERSION,
                    ).ref,
                    recording_finalization=schema_registry.resolve_version(
                        RECORDING_FINALIZATION_SCHEMA_ID,
                        RECORDING_FINALIZATION_SCHEMA_VERSION,
                    ).ref,
                    stream_recording_result=schema_registry.resolve_version(
                        LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
                        LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION,
                    ).ref,
                    window_inference_plan=schema_registry.resolve_version(
                        LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_ID,
                        LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_VERSION,
                    ).ref,
                    window_semantic_evidence_v2=schema_registry.resolve_version(
                        LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_ID,
                        LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_VERSION,
                    ).ref,
                    stream_inference_identity=schema_registry.resolve_version(
                        STREAM_INFERENCE_SCHEMA_ID,
                        STREAM_INFERENCE_SCHEMA_VERSION,
                    ).ref,
                    stream_inference_attempt=schema_registry.resolve_version(
                        STREAM_INFERENCE_ATTEMPT_SCHEMA_ID,
                        STREAM_INFERENCE_ATTEMPT_SCHEMA_VERSION,
                    ).ref,
                    stream_inference_intent=schema_registry.resolve_version(
                        STREAM_INFERENCE_INTENT_SCHEMA_ID,
                        STREAM_INFERENCE_INTENT_SCHEMA_VERSION,
                    ).ref,
                    stream_accepted_call=schema_registry.resolve_version(
                        STREAM_ACCEPTED_CALL_SCHEMA_ID,
                        STREAM_ACCEPTED_CALL_SCHEMA_VERSION,
                    ).ref,
                    stream_inference_terminal=schema_registry.resolve_version(
                        STREAM_INFERENCE_TERMINAL_SCHEMA_ID,
                        STREAM_INFERENCE_TERMINAL_SCHEMA_VERSION,
                    ).ref,
                    model_inference=(
                        schema_registry.resolve_version(
                            MODEL_INFERENCE_SCHEMA_ID,
                            "1.0.0",
                        ).ref
                        if stage_terminal_executor is not None
                        else None
                    ),
                    provider_terminal_required=provider_terminal_required,
                ),
                window_purpose=StreamPurpose.EVENT_PROPOSAL,
                recover_graph_before_execute=False,
                stage_terminal_executor=stage_terminal_executor,
                clock=clock,
            ),
            runtime_observer=runtime_observer,
        )
    planner_policy_factory: Callable[[int], BoundedMediaPolicy] | None = None
    if preflight is not None and not preflight.message_indexes_complete:
        planner_policy = None
        source_end_ns = None

        def planner_policy_factory(
            source_origin_ns: int,
        ) -> BoundedMediaPolicy:
            return _pre_eos_media_policy(
                source_scope_digest=planner_source_scope_digest,
                authorization=authorization,
                source_origin_ns=source_origin_ns,
            )

    else:
        source_origin_ns, source_end_ns = _mapped_source_bounds(channels)
        planner_policy = _pre_eos_media_policy(
            source_scope_digest=planner_source_scope_digest,
            authorization=authorization,
            source_origin_ns=source_origin_ns,
        )
    producer = DurableSinglePassVideoProducer(
        inspection=inspection,
        channels=channels,
        planner_policy=planner_policy,
        spool_directory=state_dir / "h264-spools",
        final_end_ns=source_end_ns,
        max_parallel_exports=MCAP_SPOOL_EXPORT_WORKERS,
        preflight=preflight,
        planner_policy_factory=planner_policy_factory,
        planner_source_scope_digest=planner_source_scope_digest,
        planning_sink=planning_sink,
        exporter=exporter,
    )
    spool_set = producer.prepare()
    runtime_increment(
        runtime_observer,
        "media.planning_mode",
        attributes={
            "message_indexes_complete": spool_set.preflight_message_indexes_complete,
            "mode": spool_set.planning_mode,
        },
    )
    final_inspection = producer.inspection
    final_channels = authorization.policy.resolve(final_inspection)
    if final_channels != producer.channels:
        raise CanonicalMcapSourceError(
            "final mapped channels differ from the single-pass accepted mapping"
        )
    source_origin_ns, source_end_ns = _mapped_source_bounds(final_channels)
    export_requested_end_ns = source_end_ns - source_origin_ns
    if max_duration_ns is not None:
        export_requested_end_ns = min(export_requested_end_ns, max_duration_ns)
    export_requested_interval = NanosecondInterval(
        start_ns=0,
        end_ns=export_requested_end_ns,
    )
    export_quality_policy = replace(
        DEFAULT_MEDIA_QUALITY_POLICY,
        analysis_width=media_processing_policy.sentinel_analysis_width,
    )
    export_time_visual_observer = _build_export_time_visual_observer(
        spool_set=spool_set,
        final_channels=final_channels,
        origin_ns=source_origin_ns,
        requested_interval=export_requested_interval,
        output_root=state_dir / "frames",
        media_processing_policy=media_processing_policy,
        quality_policy=export_quality_policy,
    )
    producer.set_decoded_frame_observer(export_time_visual_observer.observe)

    request = LocalVideoExportRequest(
        source=source,
        output_directory=state_dir / "video-view",
        namespace=MCAP_RECORDING_NAMESPACE,
        inspection=final_inspection,
        channels=final_channels,
        mapping_profile=authorization.profile,
        mapping_profile_digest=authorization.semantic_sha256,
        exporter=descriptor,
        verify_staged_file_digests=False,
    )
    # The sealed spools are the durable hand-off: export may advance independently
    # while EOS facts are sealed, and joins only at the fixed-six export barrier.
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="robata-registered-video-export",
    ) as export_executor:
        export_context = copy_context()
        publication_future = export_executor.submit(
            export_context.run,
            service.export_staged_local,
            request,
            producer,
        )
        if stream_scheduler is not None:
            assert capture_subject is not None
            stream_scheduler.finalize_eos(
                _mcap_eos_inputs(
                    capture_subject=capture_subject,
                    spool_set=spool_set,
                    planner_policy=producer.planner_policy,
                    planner_finish=producer.planner_finish,
                    final_channels=final_channels,
                    final_inspection=final_inspection,
                )
            )
        publication = publication_future.result()
        export_time_visual_results = (
            None if publication.derivation_reused else export_time_visual_observer.finish()
        )
    if stream_scheduler is not None:
        stream_scheduler.mark_export_barrier_complete(
            export_manifest_semantic_sha256=publication.manifest.semantic_content_sha256,
            completed_member_count=len(publication.manifest.cameras),
        )
    producer.release_exported_spool_set()
    return _PreparedVideoPublication(
        publication=publication,
        inspection=final_inspection,
        channels=final_channels,
        export_time_visual_results=export_time_visual_results,
    )


def _pre_eos_media_policy(
    *,
    source_scope_digest: str,
    authorization: AuthorizedMcapMapping,
    source_origin_ns: int,
) -> BoundedMediaPolicy:
    clock_policy_sha256 = _pre_eos_clock_policy_sha256()
    return BoundedMediaPolicy(
        source_scope_digest=source_scope_digest,
        mapping_semantic_sha256=authorization.semantic_sha256,
        alignment_semantic_sha256=clock_policy_sha256,
        source_origin_ns=source_origin_ns,
        segmentation_policy_version=f"{MCAP_PRE_EOS_POLICY_VERSION}-segmentation",
        window_policy_version=f"{MCAP_PRE_EOS_POLICY_VERSION}-window",
        quality_policy_version=f"{MCAP_PRE_EOS_POLICY_VERSION}-quality",
    )


def _create_mcap_stream_scheduler(
    *,
    execution_scheduler: SQLiteWorkScheduler,
    stream_run_id: str,
    channels: SixCameraMap[ChannelInspection],
    authorization: AuthorizedMcapMapping,
    schema_registry: SchemaRegistry,
    clock: Callable[[], datetime],
) -> tuple[PreEosCaptureSubject, DurableStreamWindowScheduler]:
    clock_policy_sha256 = _pre_eos_clock_policy_sha256()
    mapping_authority = AuthorityBinding(
        authority_id=f"local-mcap-mapping:{authorization.profile.profile_id}",
        authority_epoch=1,
        policy_version=authorization.profile.version,
        initial_binding_semantic_sha256=authorization.semantic_sha256,
    )
    clock_authority = AuthorityBinding(
        authority_id="local-mcap-log-time",
        authority_epoch=1,
        policy_version=MCAP_PRE_EOS_CLOCK_POLICY_VERSION,
        initial_binding_semantic_sha256=clock_policy_sha256,
    )
    capture = SQLiteLocalCaptureAuthority(
        execution_scheduler,
        capture_authority_id=MCAP_CAPTURE_AUTHORITY_ID,
        capture_authority_epoch=1,
        capture_assignment_policy_version=MCAP_CAPTURE_ASSIGNMENT_POLICY_VERSION,
    ).issue(
        stream_run_id,
        schema_registry.resolve_version(
            PRE_EOS_CAPTURE_SCHEMA_ID,
            PRE_EOS_CAPTURE_SCHEMA_VERSION,
        ).ref,
        _mcap_channel_bindings(
            channels=channels,
            mapping_semantic_sha256=authorization.semantic_sha256,
        ),
        mapping_authority,
        clock_authority,
    )
    policy = _pre_eos_media_policy(
        source_scope_digest=capture.capture_scope_digest,
        authorization=authorization,
        source_origin_ns=0,
    )
    expected_plan = create_expected_window_plan(
        schema_ref=schema_registry.resolve_version(
            EXPECTED_WINDOW_PLAN_SCHEMA_ID,
            EXPECTED_WINDOW_PLAN_SCHEMA_VERSION,
        ).ref,
        capture_scope_digest=capture.capture_scope_digest,
        segmentation_policy=_stream_policy_binding(
            policy.segmentation_policy_version,
            {
                "projection_version": "canonical-mcap-segmentation-policy-semantic-v1",
                "segment_duration_ns": policy.segment_duration_ns,
                "compressed_member_policy": "APPEND_ONLY_H264_SPOOL_RANGE",
            },
        ),
        window_policy=_stream_policy_binding(
            policy.window_policy_version,
            {
                "projection_version": "canonical-mcap-window-policy-semantic-v1",
                "window_width_ns": policy.window_width_ns,
                "window_hop_ns": policy.window_hop_ns,
                "window_purpose": policy.window_purpose.value,
                "quality_policy_version": policy.quality_policy_version,
                "quality_period_ns": policy.quality_period_ns,
                "quality_target_phase_ns": policy.quality_target_phase_ns,
                "quality_selection_tolerance_ns": policy.quality_selection_tolerance_ns,
            },
        ),
        watermark_policy=_stream_policy_binding(
            MCAP_WATERMARK_POLICY_VERSION,
            {
                "projection_version": "canonical-mcap-watermark-policy-semantic-v1",
                "allowed_lateness_ns": policy.allowed_lateness_ns,
                "source": "MINIMUM_SIX_CAMERA_LATEST_ALIGNED_TIMESTAMP",
                "eos_closes_partial_window": True,
            },
        ),
        lateness_policy=_stream_policy_binding(
            MCAP_LATENESS_POLICY_VERSION,
            {
                "projection_version": "canonical-mcap-lateness-policy-semantic-v1",
                "allowed_lateness_ns": policy.allowed_lateness_ns,
                "late_input_outcome": "EXPLICIT_LATE",
            },
        ),
        idle_source_policy=_stream_policy_binding(
            MCAP_IDLE_SOURCE_POLICY_VERSION,
            {
                "projection_version": "canonical-mcap-idle-source-policy-semantic-v1",
                "offline_source_policy": "EOS_CLOSES_ALL_SIX_SLOTS",
            },
        ),
        planner_version=MCAP_EXPECTED_PLAN_PLANNER_VERSION,
    )
    schema_refs = _stream_scheduler_schema_refs(schema_registry)
    dag_config_semantic_sha256 = semantic_sha256(
        {
            "projection_version": "canonical-mcap-stream-dag-config-semantic-v2",
            "stream_window_dag_policy_version": STREAM_WINDOW_DAG_POLICY_VERSION,
            "backpressure_config": DEFAULT_STREAM_BACKPRESSURE_CONFIG,
            "ordered_stages": [
                "WINDOW",
                "QA_COARSE",
                "QA_DENSE",
                "EVENT_PROPOSAL",
                "WINDOW_REDUCTION",
                "FINALIZATION",
            ],
        }
    )
    return capture, DurableStreamWindowScheduler(
        database_path=execution_scheduler.database_path,
        execution_scheduler=execution_scheduler,
        expected_plan=expected_plan,
        source_subject=capture.reference(),
        stream_run_id=stream_run_id,
        schema_refs=schema_refs,
        dag_config_semantic_sha256=dag_config_semantic_sha256,
        clock=clock,
    )


def _mcap_channel_bindings(
    *,
    channels: SixCameraMap[ChannelInspection],
    mapping_semantic_sha256: str,
) -> tuple[ChannelBinding, ...]:
    return tuple(
        ChannelBinding(
            camera_id=camera_id,
            source_channel_id=f"mcap-channel:{channels[camera_id].channel_id}",
            source_channel_epoch=1,
            channel_binding_semantic_sha256=semantic_sha256(
                {
                    "projection_version": "canonical-mcap-channel-binding-semantic-v1",
                    "camera_id": camera_id.value,
                    "channel_id": channels[camera_id].channel_id,
                    "topic": channels[camera_id].topic,
                    "schema_name": channels[camera_id].schema_name,
                    "schema_encoding": channels[camera_id].schema_encoding,
                    "schema_content_sha256": channels[camera_id].schema_content_sha256,
                    "message_encoding": channels[camera_id].message_encoding,
                    "mapping_semantic_sha256": mapping_semantic_sha256,
                }
            ),
        )
        for camera_id in CAMERA_IDS
    )


def _stream_policy_binding(
    version: str,
    semantic_projection: Mapping[str, object],
) -> StreamPolicyBinding:
    return StreamPolicyBinding(
        version=version,
        semantic_sha256=semantic_sha256(semantic_projection),
    )


def _stream_scheduler_schema_refs(
    schema_registry: SchemaRegistry,
) -> StreamSchedulerSchemaRefs:
    def resolve(schema_id: str, version: str) -> SchemaRef:
        return schema_registry.resolve_version(schema_id, version).ref

    return StreamSchedulerSchemaRefs(
        incremental_window=resolve(
            INCREMENTAL_WINDOW_SCHEMA_ID,
            INCREMENTAL_WINDOW_SCHEMA_VERSION,
        ),
        expected_declaration=resolve(
            EXPECTED_WINDOW_DECLARATION_SCHEMA_ID,
            EXPECTED_WINDOW_DECLARATION_SCHEMA_VERSION,
        ),
        expected_plan_seal=resolve(
            EXPECTED_WINDOW_SEAL_SCHEMA_ID,
            EXPECTED_WINDOW_SEAL_SCHEMA_VERSION,
        ),
        stream_work_plan=resolve(
            STREAM_WORK_PLAN_SCHEMA_ID,
            STREAM_WORK_PLAN_SCHEMA_VERSION,
        ),
        terminal_member=resolve(
            WINDOW_TERMINAL_MEMBER_SCHEMA_ID,
            WINDOW_TERMINAL_MEMBER_SCHEMA_VERSION,
        ),
        terminal_closure=resolve(
            WINDOW_TERMINAL_CLOSURE_SCHEMA_ID,
            WINDOW_TERMINAL_CLOSURE_SCHEMA_VERSION,
        ),
    )


def _mcap_eos_inputs(
    *,
    capture_subject: PreEosCaptureSubject,
    spool_set: H264SpoolSetFacts,
    planner_policy: BoundedMediaPolicy,
    planner_finish: PlannerFinish,
    final_channels: SixCameraMap[ChannelInspection],
    final_inspection: McapInspection,
) -> EosSealInputs:
    if (
        spool_set.source_sha256 != final_inspection.source_sha256
        or spool_set.source_size_bytes != final_inspection.source_size_bytes
        or spool_set.source_message_count != final_inspection.message_count
    ):
        raise CanonicalMcapSourceError(
            "EOS scheduler facts differ from the accepted source and spool set"
        )
    facts_by_camera = {fact.camera_id: fact for fact in planner_finish.facts}
    if set(facts_by_camera) != set(CAMERA_IDS):
        raise CanonicalMcapSourceError("planner EOS facts do not cover all six cameras")
    eos_source_receipt = semantic_sha256(
        {
            "projection_version": "canonical-mcap-eos-source-receipt-semantic-v1",
            "capture_scope_digest": capture_subject.capture_scope_digest,
            "source_sha256": spool_set.source_sha256,
            "source_size_bytes": str(spool_set.source_size_bytes),
            "source_message_count": str(spool_set.source_message_count),
            "ordered_spools": [
                {
                    "camera_id": camera_id.value,
                    "packet_count": str(spool_set.spools[camera_id].packet_count),
                    "size_bytes": str(spool_set.spools[camera_id].size_bytes),
                    "sha256": spool_set.spools[camera_id].sha256,
                }
                for camera_id in CAMERA_IDS
            ],
        }
    )
    final_timeline = semantic_sha256(
        {
            "projection_version": "canonical-mcap-final-timeline-semantic-v1",
            "capture_scope_digest": capture_subject.capture_scope_digest,
            "source_origin_ns": str(planner_policy.source_origin_ns),
            "final_end_ns": str(spool_set.final_end_ns),
            "ordered_camera_timelines": [
                {
                    "camera_id": camera_id.value,
                    "packet_count": str(facts_by_camera[camera_id].packet_count),
                    "first_timestamp_ns": (
                        None
                        if facts_by_camera[camera_id].first_timestamp_ns is None
                        else str(facts_by_camera[camera_id].first_timestamp_ns)
                    ),
                    "last_timestamp_ns": (
                        None
                        if facts_by_camera[camera_id].last_timestamp_ns is None
                        else str(facts_by_camera[camera_id].last_timestamp_ns)
                    ),
                    "first_sequence": (
                        None
                        if facts_by_camera[camera_id].first_sequence is None
                        else str(facts_by_camera[camera_id].first_sequence)
                    ),
                    "last_sequence": (
                        None
                        if facts_by_camera[camera_id].last_sequence is None
                        else str(facts_by_camera[camera_id].last_sequence)
                    ),
                }
                for camera_id in CAMERA_IDS
            ],
        }
    )
    channel_health = semantic_sha256(
        {
            "projection_version": "canonical-mcap-six-channel-health-semantic-v1",
            "ordered_channels": [
                {
                    "camera_id": camera_id.value,
                    "channel_id": str(final_channels[camera_id].channel_id),
                    "message_count": str(final_channels[camera_id].message_count),
                    "monotonic": final_channels[camera_id].monotonic,
                    "sequence_gap_count": str(facts_by_camera[camera_id].sequence_gap_count),
                    "payload_bytes": str(facts_by_camera[camera_id].payload_bytes),
                }
                for camera_id in CAMERA_IDS
            ],
        }
    )
    mapping_closure = semantic_sha256(
        {
            "projection_version": "canonical-mcap-final-mapping-closure-semantic-v1",
            "capture_scope_digest": capture_subject.capture_scope_digest,
            "mapping_semantic_sha256": planner_policy.mapping_semantic_sha256,
            "ordered_channel_bindings": [
                {
                    "camera_id": camera_id.value,
                    "channel_id": str(final_channels[camera_id].channel_id),
                    "topic": final_channels[camera_id].topic,
                    "schema_content_sha256": (final_channels[camera_id].schema_content_sha256),
                }
                for camera_id in CAMERA_IDS
            ],
        }
    )
    clock_closure = semantic_sha256(
        {
            "projection_version": "canonical-mcap-final-clock-closure-semantic-v1",
            "capture_scope_digest": capture_subject.capture_scope_digest,
            "clock_policy_semantic_sha256": planner_policy.alignment_semantic_sha256,
            "final_timeline_semantic_sha256": final_timeline,
        }
    )
    return EosSealInputs(
        eos_source_receipt_semantic_sha256=eos_source_receipt,
        final_source_timeline_semantic_sha256=final_timeline,
        final_duration_ns=spool_set.final_end_ns - planner_policy.source_origin_ns,
        ordered_six_channel_health_closure_sha256=channel_health,
        mapping_closure_semantic_sha256=mapping_closure,
        clock_or_alignment_closure_semantic_sha256=clock_closure,
    )


def _pre_eos_clock_policy_sha256() -> str:
    return semantic_sha256(
        {
            "clock_policy_version": MCAP_PRE_EOS_CLOCK_POLICY_VERSION,
            "source_clock": "MCAP_LOG_TIME_NS",
            "pre_alignment_transform": "IDENTITY",
            "source_origin_policy": "MINIMUM_MAPPED_FIRST_LOG_TIME",
        }
    )


def _mapped_source_bounds(
    channels: SixCameraMap[ChannelInspection],
) -> tuple[int, int]:
    first_times: list[int] = []
    last_times: list[int] = []
    for camera_id in CAMERA_IDS:
        channel = channels[camera_id]
        if (
            channel.message_count < 1
            or channel.first_message_time_ns is None
            or channel.last_message_time_ns is None
            or not channel.monotonic
            or channel.schema_encoding is None
            or channel.schema_content_sha256 is None
        ):
            raise CanonicalMcapSourceError(
                f"{camera_id.value} lacks complete monotonic source/schema facts"
            )
        first_times.append(channel.first_message_time_ns)
        last_times.append(channel.last_message_time_ns)
    return min(first_times), max(last_times) + 1


def _build_stream_records(
    *,
    inspection: McapInspection,
    channels: SixCameraMap[ChannelInspection],
    publication: PublishedRegisteredVideoExport,
) -> dict[CameraId, tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2]]:
    schema_policy = _policy("protobuf-compressed-image-schema")
    probe_component = _component("pyav-h264-remux-decode-validation")
    publication_records = {record.camera_id: record for record in publication.manifest.cameras}
    records: dict[CameraId, tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2]] = {}
    for camera_id in CAMERA_IDS:
        channel = channels[camera_id]
        result = publication_records[camera_id]
        assert channel.schema_name is not None
        assert channel.schema_encoding is not None
        assert channel.schema_content_sha256 is not None
        stream_id = _stable_uuid(
            "canonical-mcap-stream",
            f"{inspection.source_sha256}:{channel.channel_id}:{channel.topic}",
        )
        schema_fact = StreamSchemaEvidenceV2(
            stream_id=stream_id,
            stream_semantic_sha256="0" * 64,
            schema_name=channel.schema_name,
            schema_encoding=channel.schema_encoding,
            schema_content_sha256=channel.schema_content_sha256,
            support_status=SchemaSupportStatus.SUPPORTED,
            support_policy=schema_policy,
            diagnostic_ids=(),
        )
        probe_fact = ProbedVideoStreamFactV2(
            stream_id=stream_id,
            stream_semantic_sha256="0" * 64,
            topic=channel.topic,
            channel_id=channel.channel_id,
            message_encoding=channel.message_encoding,
            codec=result.source.codec,
            message_count=channel.message_count,
            first_timestamp_ns=channel.first_message_time_ns,
            last_timestamp_ns=channel.last_message_time_ns,
            decoder_probe=DecoderProbeEvidenceV2(
                probe=probe_component,
                outcome=DecoderProbeOutcome.PASSED,
                decoded_frame_count=result.exported_frame_count,
                decoded_width=result.width,
                decoded_height=result.height,
                diagnostic_ids=(),
            ),
        )
        stream_digest = compute_stream_semantic_sha256_v2(
            source_content_sha256=inspection.source_sha256,
            schema_evidence=schema_fact,
            probed_stream_fact=probe_fact,
        )
        records[camera_id] = (
            schema_fact.model_copy(update={"stream_semantic_sha256": stream_digest}),
            probe_fact.model_copy(update={"stream_semantic_sha256": stream_digest}),
        )
    return records


def _validation_report(
    *,
    schema_registry: SchemaRegistry,
    inspection: McapInspection,
    recording_id: str,
    mcap_id: str,
    mapping_policy: SemanticPolicyReference,
    mapping_digest: str,
    stream_records: Mapping[
        CameraId,
        tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2],
    ],
    camera_mappings: tuple[ValidationCameraMappingV2, ...],
    observed_at: str,
) -> MCAPValidationReportV2:
    fields: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": schema_registry.resolve_version(
            MCAP_VALIDATION_REPORT_V2_SCHEMA_ID,
            ADMISSION_EVIDENCE_V2_SCHEMA_VERSION,
        ).ref,
        "validation_report_id": _stable_uuid("canonical-mcap-validation-row", mcap_id),
        "validation_report_semantic_sha256": "0" * 64,
        "mcap_id": mcap_id,
        "recording_identity": recording_id,
        "source_content_sha256": inspection.source_sha256,
        "source": MCAPValidationSourceV2(
            uri=inspection.source.resolve().as_uri(),
            object_version=inspection.source_sha256,
            sha256=inspection.source_sha256,
            bytes=inspection.source_size_bytes,
        ),
        "mapping_policy": mapping_policy,
        "camera_mapping_semantic_sha256": mapping_digest,
        "validator": _component("official-mcap-v2-validator"),
        "checks": tuple(
            _pass_check(check_id, subject)
            for check_id, subject in (
                ("decoder-probe", "six mapped H264 streams"),
                ("exact-camera-mapping", "authorized six-camera topic mapping"),
                ("schema-bytes", "exact protobuf schema bytes"),
                ("source-log-time", "strictly monotonic mapped log time"),
            )
        ),
        "diagnostics": (),
        "schema_evidence": tuple(
            sorted(
                (record[0] for record in stream_records.values()),
                key=lambda item: (item.stream_semantic_sha256, item.schema_name),
            )
        ),
        "probed_stream_facts": tuple(
            sorted(
                (record[1] for record in stream_records.values()),
                key=lambda item: (
                    item.stream_semantic_sha256,
                    item.topic,
                    item.channel_id,
                ),
            )
        ),
        "camera_mappings": camera_mappings,
        "discovered_video_stream_count": 6,
        "mapped_camera_count": 6,
        "verdict": MCAPValidationVerdictV2.VALID,
        "validated_at": observed_at,
    }
    draft = MCAPValidationReportV2.model_construct(**fields)
    digest = semantic_sha256(mcap_validation_report_v2_semantic_projection(draft))
    fields["validation_report_semantic_sha256"] = digest
    fields["validation_report_id"] = _stable_uuid("canonical-mcap-validation", digest)
    return MCAPValidationReportV2.model_validate(fields, strict=True)


def _ready_manifest(
    *,
    schema_registry: SchemaRegistry,
    inspection: McapInspection,
    validation_report: MCAPValidationReportV2,
    mapping_policy: SemanticPolicyReference,
    publication: PublishedRegisteredVideoExport,
    channels: SixCameraMap[ChannelInspection],
    ledgers: Mapping[CameraId, _CameraLedger],
    observed_at: str,
) -> MCAPReadyManifestV2:
    cameras: list[MCAPReadyCameraV2] = []
    ready_by_camera = {item.camera_id: item for item in publication.manifest.cameras}
    for camera_id in CAMERA_IDS:
        channel = channels[camera_id]
        record = ready_by_camera[camera_id]
        ledger = ledgers[camera_id]
        assert channel.first_message_time_ns is not None
        assert channel.last_message_time_ns is not None
        elapsed_ns = ledger.rows[-1].source_log_time_ns - ledger.rows[0].source_log_time_ns
        nominal_fps = (
            1_000_000_000.0 / ledger.rows[0].duration_ns
            if len(ledger.rows) == 1
            else (len(ledger.rows) - 1) * 1_000_000_000.0 / elapsed_ns
        )
        stream = next(
            item
            for item in validation_report.probed_stream_facts
            if item.channel_id == channel.channel_id
        )
        cameras.append(
            MCAPReadyCameraV2(
                camera_id=camera_id,
                role=camera_id.value,
                stream_id=stream.stream_id,
                stream_semantic_sha256=stream.stream_semantic_sha256,
                topic=channel.topic,
                channel_id=channel.channel_id,
                codec=stream.codec,
                width=record.width,
                height=record.height,
                nominal_fps=nominal_fps,
                source_start_ns=channel.first_message_time_ns,
                source_end_ns=channel.last_message_time_ns + 1,
                frame_count=record.exported_frame_count,
            )
        )
    source_origin_ns = min(
        channel.first_message_time_ns
        for channel in channels.values()
        if channel.first_message_time_ns is not None
    )
    source_end_ns = (
        max(
            channel.last_message_time_ns
            for channel in channels.values()
            if channel.last_message_time_ns is not None
        )
        + 1
    )
    fields: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": schema_registry.resolve_version(
            MCAP_READY_MANIFEST_V2_SCHEMA_ID,
            ADMISSION_EVIDENCE_V2_SCHEMA_VERSION,
        ).ref,
        "ready_manifest_id": _stable_uuid(
            "canonical-mcap-ready-row",
            validation_report.validation_report_semantic_sha256,
        ),
        "ready_manifest_semantic_sha256": "0" * 64,
        "mcap_id": validation_report.mcap_id,
        "recording_identity": validation_report.recording_identity,
        "source_content_sha256": validation_report.source_content_sha256,
        "source": MCAPReadySourceV2(
            artifact_id=publication.manifest.source_artifact_id,
            uri=inspection.source.resolve().as_uri(),
            object_version=inspection.source_sha256,
            sha256=inspection.source_sha256,
            bytes=inspection.source_size_bytes,
        ),
        "source_durability": SourceDurabilityEvidenceV2(
            verifier=_component("local-artifact-registry-source-verifier"),
            outcome="PASS",
            verified_sha256=inspection.source_sha256,
            verified_bytes=inspection.source_size_bytes,
        ),
        "validation_report_id": validation_report.validation_report_id,
        "validation_report_semantic_sha256": (validation_report.validation_report_semantic_sha256),
        "validation_report_schema_ref": validation_report.schema_ref,
        "camera_mapping_run_id": _stable_uuid(
            "canonical-mcap-camera-mapping",
            validation_report.camera_mapping_semantic_sha256,
        ),
        "camera_mapping_semantic_sha256": (validation_report.camera_mapping_semantic_sha256),
        "mapping_policy": mapping_policy,
        "admission_policy": _policy("local-mcap-source-admission"),
        "recording": MCAPReadyRecording(
            start_utc=None,
            end_utc=None,
            duration_ns=source_end_ns - source_origin_ns,
            timebase="recording_relative_ns",
        ),
        "camera_count": 6,
        "cameras": tuple(cameras),
        "published_at": observed_at,
    }
    draft = MCAPReadyManifestV2.model_construct(**fields)
    digest = semantic_sha256(mcap_ready_manifest_v2_semantic_projection(draft))
    fields["ready_manifest_semantic_sha256"] = digest
    fields["ready_manifest_id"] = _stable_uuid("canonical-mcap-ready", digest)
    return MCAPReadyManifestV2.model_validate(fields, strict=True)


def _alignment_manifest(
    *,
    schema_registry: SchemaRegistry,
    ready_manifest: MCAPReadyManifestV2,
    ledgers: Mapping[CameraId, _CameraLedger],
    observed_at: str,
) -> AlignmentManifestV2:
    origin_ns = min(camera.source_start_ns for camera in ready_manifest.cameras)
    ready_by_camera = {camera.camera_id: camera for camera in ready_manifest.cameras}
    cameras: dict[str, CameraAlignmentV2] = {}
    for camera_id in CAMERA_IDS:
        ready_camera = ready_by_camera[camera_id]
        rows = ledgers[camera_id].rows
        segment = AlignmentSegment(
            segment_id=_stable_uuid(
                "canonical-mcap-alignment-segment",
                f"{ready_manifest.source_content_sha256}:{camera_id.value}",
            ),
            source_epoch_id="mcap-log-time-0",
            source_order_start=0,
            source_order_end=len(rows),
            source_start_ns=rows[0].source_log_time_ns,
            source_end_ns=rows[-1].source_log_time_ns + rows[-1].duration_ns,
            source_anchor_ns=origin_ns,
            canonical_anchor_ns=0,
            rate_numerator="1",
            rate_denominator="1",
            rounding="HALF_EVEN",
        )
        cameras[camera_id.value] = CameraAlignmentV2(
            source_clock_id="mcap-log-time",
            source_timestamp_unit="ns",
            derived_drift_ppm=0.0,
            residual_p95_ns=0,
            max_error_ns=0,
            coverage=1.0,
            segments=(segment,),
            status=AlignmentStatus.VALID,
            stream_id=ready_camera.stream_id,
            stream_semantic_sha256=ready_camera.stream_semantic_sha256,
        )
    fields: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": schema_registry.resolve_version(
            ALIGNMENT_MANIFEST_V2_SCHEMA_ID,
            ADMISSION_EVIDENCE_V2_SCHEMA_VERSION,
        ).ref,
        "alignment_id": _stable_uuid(
            "canonical-mcap-alignment-row",
            ready_manifest.ready_manifest_semantic_sha256,
        ),
        "alignment_semantic_sha256": "0" * 64,
        "mcap_id": ready_manifest.mcap_id,
        "recording_identity": ready_manifest.recording_identity,
        "source_content_sha256": ready_manifest.source_content_sha256,
        "ready_manifest_id": ready_manifest.ready_manifest_id,
        "ready_manifest_semantic_sha256": ready_manifest.ready_manifest_semantic_sha256,
        "ready_manifest_schema_ref": ready_manifest.schema_ref,
        "camera_mapping_run_id": ready_manifest.camera_mapping_run_id,
        "camera_mapping_semantic_sha256": ready_manifest.camera_mapping_semantic_sha256,
        "reference_timebase": "recording_relative_ns",
        "canonical_origin": CanonicalOrigin(
            source="minimum_mapped_mcap_log_time",
            reference_timestamp_ns=origin_ns,
            utc=None,
        ),
        "method": AlignmentMethod.MCAP_LOG_TIME,
        "algorithm": _component("identity-mcap-log-time-alignment"),
        "status": AlignmentStatus.VALID,
        "cameras": cameras,
        "policy": _policy("local-mcap-log-time-alignment"),
        "validator": _component("mcap-log-time-alignment-validator"),
        "checks": (_pass_check("identity-transform", "mapped MCAP log timestamps"),),
        "diagnostics": (),
        "created_at": observed_at,
    }
    draft = AlignmentManifestV2.model_construct(**fields)
    digest = semantic_sha256(alignment_manifest_v2_semantic_projection(draft))
    fields["alignment_semantic_sha256"] = digest
    fields["alignment_id"] = _stable_uuid("canonical-mcap-alignment", digest)
    return AlignmentManifestV2.model_validate(fields, strict=True)


def _frame_index(
    *,
    context: AdmittedRecordingContextV2,
    stream_records: Mapping[
        CameraId,
        tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2],
    ],
    ledgers: Mapping[CameraId, _CameraLedger],
) -> CanonicalSixCameraFrameIndex:
    origin_ns = context.alignment_manifest.canonical_origin.reference_timestamp_ns
    cameras: dict[CameraId, CameraSourceFrameIndex] = {}
    for camera_id in CAMERA_IDS:
        ledger = ledgers[camera_id]
        segment = context.alignment_manifest.cameras[camera_id.value].segments[0]
        frames: list[IndexedSourceFrame] = []
        for row in ledger.rows:
            identity = semantic_sha256(
                {
                    "policy_version": MCAP_SOURCE_POLICY_VERSION,
                    "source_content_sha256": context.source_content_sha256,
                    "camera_id": camera_id.value,
                    "packet_index": row.packet_index,
                    "source_sequence": row.source_sequence,
                    "source_log_time_ns": str(row.source_log_time_ns),
                    "video_sha256": ledger.record.video_artifact.sha256,
                }
            )
            frame_id = _stable_uuid("canonical-mcap-source-frame", identity)
            frames.append(
                IndexedSourceFrame(
                    source_frame_id=frame_id,
                    source_order=row.packet_index,
                    source_timestamp_ns=row.source_log_time_ns,
                    source_locator={
                        "camera_id": camera_id.value,
                        "packet_index": row.packet_index,
                        "source_log_time_ns": str(row.source_log_time_ns),
                        "timestamp_sidecar_sha256": ledger.sidecar_sha256,
                        "video_sha256": ledger.record.video_artifact.sha256,
                    },
                    decodable=True,
                    alignment_projection=FrameAlignmentProjectionFact(
                        projection_id=_stable_uuid(
                            "canonical-mcap-frame-alignment",
                            f"{context.alignment_semantic_sha256}:{identity}",
                        ),
                        alignment_id=context.alignment_manifest.alignment_id,
                        segment_id=segment.segment_id,
                        aligned_timestamp_ns=row.source_log_time_ns - origin_ns,
                    ),
                )
            )
        stream = stream_records[camera_id][0]
        cameras[camera_id] = CameraSourceFrameIndex(
            camera_id=camera_id,
            stream_id=stream.stream_id,
            stream_semantic_sha256=stream.stream_semantic_sha256,
            frames=tuple(frames),
        )
    return CanonicalSixCameraFrameIndex(
        mcap_id=context.ready_manifest.mcap_id,
        camera_mapping_run_id=context.ready_manifest.camera_mapping_run_id,
        alignment_id=context.alignment_manifest.alignment_id,
        source_content_sha256=context.source_content_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=context.alignment_semantic_sha256,
        cameras=SixCameraMap[CameraSourceFrameIndex](cameras),
    )


def _frame_timing_evidence(
    *,
    frame_index: CanonicalSixCameraFrameIndex,
    ledgers: Mapping[CameraId, _CameraLedger],
) -> dict[CameraId, tuple[FrameTimingEvidence, ...]]:
    timings: dict[CameraId, tuple[FrameTimingEvidence, ...]] = {}
    for camera_id in CAMERA_IDS:
        frames = frame_index.cameras[camera_id].frames
        rows = ledgers[camera_id].rows
        if len(frames) != len(rows):
            raise CanonicalMcapSourceError(
                f"{camera_id.value} frame index differs from its verified timestamp sidecar"
            )
        camera_timings: list[FrameTimingEvidence] = []
        for frame, row in zip(frames, rows, strict=True):
            if (
                frame.source_order != row.packet_index
                or frame.source_timestamp_ns != row.source_log_time_ns
            ):
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} frame timing differs from its verified timestamp sidecar"
                )
            camera_timings.append(
                FrameTimingEvidence(
                    camera_id=camera_id,
                    packet_index=row.packet_index,
                    aligned_timestamp_ns=frame.alignment_projection.aligned_timestamp_ns,
                    source_timestamp_ns=row.source_log_time_ns,
                    source_sequence=row.source_sequence,
                )
            )
        timings[camera_id] = tuple(camera_timings)
    return timings


def _materialize_selected_frames(
    *,
    frame_index: CanonicalSixCameraFrameIndex,
    ledgers: Mapping[CameraId, _CameraLedger],
    quality_timings: Mapping[CameraId, tuple[FrameTimingEvidence, ...]],
    requested_interval: NanosecondInterval,
    output_root: Path,
    media_processing_policy: McapMediaProcessingPolicy = DEFAULT_MCAP_MEDIA_PROCESSING_POLICY,
    quality_policy: LocalMediaQualityPolicy = DEFAULT_MEDIA_QUALITY_POLICY,
    stop_after_selected: bool = False,
) -> tuple[
    dict[tuple[CameraId, str], MaterializedFrameArtifactFact],
    dict[CameraId, tuple[FrameQualityObservation, ...]],
]:
    """Decode each camera once for selected evidence and bounded visual sentinels.

    Packet/timeline integrity is already captured in the verified sidecars and is reduced
    independently by the quality report. This pass only creates compact grayscale views
    at configured sentinel targets and PNG bytes at configured semantic targets; no raw
    decoded RGB representation is retained after each frame is handled.
    """

    if not isinstance(media_processing_policy, McapMediaProcessingPolicy):
        raise TypeError("media_processing_policy must be a McapMediaProcessingPolicy")
    if not isinstance(quality_policy, LocalMediaQualityPolicy):
        raise TypeError("quality_policy must be a LocalMediaQualityPolicy")
    if quality_policy.analysis_width != media_processing_policy.sentinel_analysis_width:
        raise ValueError("quality policy width must match the media sentinel analysis width")
    if not isinstance(stop_after_selected, bool):
        raise TypeError("stop_after_selected must be a boolean")

    output_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[tuple[CameraId, str], MaterializedFrameArtifactFact] = {}
    quality_observations: dict[CameraId, tuple[FrameQualityObservation, ...]] = {}
    selected_frames: dict[CameraId, dict[int, IndexedSourceFrame]] = {}
    sentinel_indexes: dict[CameraId, frozenset[int]] = {}
    for camera_id in CAMERA_IDS:
        source_frames = frame_index.cameras[camera_id].frames
        selected_frames[camera_id] = _select_media_target_frames(
            source_frames=source_frames,
            requested_interval=requested_interval,
            rate=SamplingRate(
                media_processing_policy.semantic_rate_numerator,
                media_processing_policy.semantic_rate_denominator,
            ),
            selection_tolerance_ns=media_processing_policy.selection_tolerance_ns,
        )
        sentinel_indexes[camera_id] = frozenset(
            _select_media_target_frames(
                source_frames=source_frames,
                requested_interval=requested_interval,
                rate=SamplingRate(
                    media_processing_policy.sentinel_rate_numerator,
                    media_processing_policy.sentinel_rate_denominator,
                ),
                selection_tolerance_ns=media_processing_policy.selection_tolerance_ns,
            )
        )
    with ThreadPoolExecutor(
        max_workers=min(MCAP_FRAME_MATERIALIZATION_WORKERS, len(CAMERA_IDS)),
        thread_name_prefix="robata-frame-materializer",
    ) as pool:
        futures = {
            camera_id: pool.submit(
                _decode_selected_camera_frames,
                camera_id=camera_id,
                ledger=ledgers[camera_id],
                quality_timings=quality_timings[camera_id],
                selected_by_index=selected_frames[camera_id],
                sentinel_indexes=sentinel_indexes[camera_id],
                output_root=output_root,
                quality_policy=quality_policy,
                evidence_max_width=media_processing_policy.evidence_max_width,
                evidence_extractor_version=media_processing_policy.evidence_extractor_version,
                stop_after_selected=stop_after_selected,
            )
            for camera_id in CAMERA_IDS
        }
        for camera_id in CAMERA_IDS:
            camera_artifacts, observations = futures[camera_id].result()
            artifacts.update(camera_artifacts)
            quality_observations[camera_id] = observations
    return artifacts, quality_observations


def _bind_export_time_visual_results(
    *,
    frame_index: CanonicalSixCameraFrameIndex,
    ledgers: Mapping[CameraId, _CameraLedger],
    quality_timings: Mapping[CameraId, tuple[FrameTimingEvidence, ...]],
    requested_interval: NanosecondInterval,
    output_root: Path,
    media_processing_policy: McapMediaProcessingPolicy,
    quality_policy: LocalMediaQualityPolicy,
    visual_results: _ExportTimeVisualResults,
) -> tuple[
    dict[tuple[CameraId, str], MaterializedFrameArtifactFact],
    dict[CameraId, tuple[FrameQualityObservation, ...]],
]:
    """Bind bounded export-time facts to the final verified frame index."""

    if not isinstance(media_processing_policy, McapMediaProcessingPolicy):
        raise TypeError("media_processing_policy must be a McapMediaProcessingPolicy")
    if not isinstance(quality_policy, LocalMediaQualityPolicy):
        raise TypeError("quality_policy must be a LocalMediaQualityPolicy")
    if quality_policy.analysis_width != media_processing_policy.sentinel_analysis_width:
        raise ValueError("quality policy width must match the media sentinel analysis width")
    artifacts: dict[tuple[CameraId, str], MaterializedFrameArtifactFact] = {}
    quality_observations: dict[CameraId, tuple[FrameQualityObservation, ...]] = {}
    semantic_rate = SamplingRate(
        media_processing_policy.semantic_rate_numerator,
        media_processing_policy.semantic_rate_denominator,
    )
    sentinel_rate = SamplingRate(
        media_processing_policy.sentinel_rate_numerator,
        media_processing_policy.sentinel_rate_denominator,
    )
    for camera_id in CAMERA_IDS:
        source_frames = frame_index.cameras[camera_id].frames
        timings = quality_timings[camera_id]
        if len(source_frames) != len(timings):
            raise CanonicalMcapSourceError(
                f"{camera_id.value} quality timing differs from its canonical frame index"
            )
        selected_frames = _select_media_target_frames(
            source_frames=source_frames,
            requested_interval=requested_interval,
            rate=semantic_rate,
            selection_tolerance_ns=media_processing_policy.selection_tolerance_ns,
        )
        sentinel_indexes = frozenset(
            _select_media_target_frames(
                source_frames=source_frames,
                requested_interval=requested_interval,
                rate=sentinel_rate,
                selection_tolerance_ns=media_processing_policy.selection_tolerance_ns,
            )
        )
        result = visual_results.cameras.get(camera_id)
        if result is None:
            raise CanonicalMcapSourceError(f"{camera_id.value} has no export-time visual result")
        if result.leading_access_unit_count != ledgers[camera_id].record.leading_drops.count:
            raise CanonicalMcapSourceError(
                f"{camera_id.value} export leading-drop count differs from its sidecar"
            )
        expected_semantic_orders = {
            result.leading_access_unit_count + index for index in selected_frames
        }
        expected_sentinel_orders = {
            result.leading_access_unit_count + index for index in sentinel_indexes
        }
        if result.semantic_source_orders != frozenset(expected_semantic_orders):
            raise CanonicalMcapSourceError(
                f"{camera_id.value} export semantic selection differs from its frame index"
            )
        if result.sentinel_source_orders != frozenset(expected_sentinel_orders):
            raise CanonicalMcapSourceError(
                f"{camera_id.value} export sentinel selection differs from its frame index"
            )
        frames_by_order = {frame.source_order: frame for frame in source_frames}
        if set(result.rendered) != expected_semantic_orders:
            raise CanonicalMcapSourceError(
                f"{camera_id.value} export rendered keys differ from its semantic selection"
            )
        for raw_order, rendered in result.rendered.items():
            exported_index = raw_order - result.leading_access_unit_count
            source_frame = frames_by_order.get(exported_index)
            if source_frame is None:
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} export rendered frame is absent from its frame index"
                )
            artifacts[(camera_id, source_frame.source_frame_id)] = (
                _materialized_frame_artifact_from_rendered(
                    source_frame=source_frame,
                    rendered=rendered,
                    evidence_extractor_version=media_processing_policy.evidence_extractor_version,
                )
            )
        if set(result.observations) != expected_sentinel_orders:
            raise CanonicalMcapSourceError(
                f"{camera_id.value} export observed keys differ from its sentinel selection"
            )
        observations_by_index: dict[int, FrameQualityObservation] = {}
        for raw_order, evidence in result.observations.items():
            exported_index = raw_order - result.leading_access_unit_count
            if not 0 <= exported_index < len(timings):
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} export observation index is outside its sidecar"
                )
            if evidence.timing != timings[exported_index]:
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} export timing differs from its sidecar"
                )
            quality = evidence.quality
            if (
                quality.camera_id is not camera_id
                or quality.packet_index != exported_index
                or quality.aligned_timestamp_ns != evidence.timing.aligned_timestamp_ns
                or quality.source_timestamp_ns != evidence.timing.source_timestamp_ns
            ):
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} export quality observation differs from its timing"
                )
            observations_by_index[exported_index] = quality
        if set(observations_by_index) != set(sentinel_indexes):
            raise CanonicalMcapSourceError(
                f"{camera_id.value} export did not retain every selected quality observation"
            )
        quality_observations[camera_id] = tuple(
            observations_by_index[index] for index in sorted(observations_by_index)
        )
    return artifacts, quality_observations


def _select_media_target_frames(
    *,
    source_frames: tuple[IndexedSourceFrame, ...],
    requested_interval: NanosecondInterval,
    rate: SamplingRate,
    selection_tolerance_ns: int,
) -> dict[int, IndexedSourceFrame]:
    """Select deterministic frame targets without decoding visual pixels."""

    if not isinstance(rate, SamplingRate):
        raise TypeError("rate must be a SamplingRate")
    if (
        isinstance(selection_tolerance_ns, bool)
        or not isinstance(selection_tolerance_ns, int)
        or selection_tolerance_ns <= 0
    ):
        raise ValueError("selection_tolerance_ns must be a positive integer")
    frames_by_locator = {
        canonical_json_bytes(frame.source_locator): frame for frame in source_frames
    }
    grid = SamplingGrid(grid_origin_ns=0, rate=rate)
    selections = grid.select_frames(
        (
            FrameCandidate(
                aligned_timestamp_ns=frame.alignment_projection.aligned_timestamp_ns,
                source_timestamp_ns=frame.source_timestamp_ns,
                source_locator_bytes=locator,
                decodable=frame.decodable,
            )
            for locator, frame in frames_by_locator.items()
        ),
        requested_interval.start_ns,
        requested_interval.end_ns,
        selection_tolerance_ns,
    )
    selected_by_index: dict[int, IndexedSourceFrame] = {}
    for selection in selections:
        if selection.status is not SelectionStatus.SELECTED:
            continue
        assert selection.frame is not None
        source_frame = frames_by_locator[selection.frame.source_locator_bytes]
        packet_index = source_frame.source_locator.get("packet_index")
        if isinstance(packet_index, bool) or not isinstance(packet_index, int):
            raise CanonicalMcapSourceError("canonical packet locator is not an integer")
        selected_by_index[packet_index] = source_frame
    return selected_by_index


def _decode_selected_camera_frames(
    *,
    camera_id: CameraId,
    ledger: _CameraLedger,
    quality_timings: tuple[FrameTimingEvidence, ...],
    selected_by_index: Mapping[int, IndexedSourceFrame],
    sentinel_indexes: frozenset[int],
    output_root: Path,
    quality_policy: LocalMediaQualityPolicy = DEFAULT_MEDIA_QUALITY_POLICY,
    evidence_max_width: int = MCAP_PNG_MAX_WIDTH,
    evidence_extractor_version: str = MCAP_PNG_EXTRACTOR_VERSION,
    stop_after_selected: bool = False,
) -> tuple[
    dict[tuple[CameraId, str], MaterializedFrameArtifactFact],
    tuple[FrameQualityObservation, ...],
]:
    """Walk one verified MP4 sequentially and service both bounded target sets."""

    if not isinstance(stop_after_selected, bool):
        raise TypeError("stop_after_selected must be a boolean")
    if not isinstance(sentinel_indexes, frozenset) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in sentinel_indexes
    ):
        raise TypeError("sentinel_indexes must be a frozen set of nonnegative integers")
    if not isinstance(quality_policy, LocalMediaQualityPolicy):
        raise TypeError("quality_policy must be a LocalMediaQualityPolicy")
    if isinstance(evidence_max_width, bool) or not isinstance(evidence_max_width, int):
        raise TypeError("evidence_max_width must be an integer")
    if evidence_max_width <= 0:
        raise ValueError("evidence_max_width must be positive")
    if not isinstance(evidence_extractor_version, str) or not evidence_extractor_version:
        raise ValueError("evidence_extractor_version must be a non-empty string")
    required_indexes = frozenset(selected_by_index) | sentinel_indexes
    if stop_after_selected and not required_indexes:
        return {}, ()
    if len(quality_timings) != len(ledger.rows):
        raise CanonicalMcapSourceError(
            f"{camera_id.value} quality timing differs from its verified timestamp sidecar"
        )

    decoded_count = 0
    rendered_indexes: set[int] = set()
    observed_indexes: set[int] = set()
    artifacts: dict[tuple[CameraId, str], MaterializedFrameArtifactFact] = {}
    observations: list[FrameQualityObservation] = []
    analyzer = LocalFrameQualityAnalyzer(camera_id, policy=quality_policy)
    last_required_index = max(required_indexes) if stop_after_selected else None
    with av.open(str(ledger.video_path), mode="r") as container:
        streams = tuple(container.streams.video)
        if len(streams) != 1 or bool(streams[0].codec_context.has_b_frames):
            raise CanonicalMcapSourceError(
                f"{camera_id.value} registered MP4 is not one non-reordered video stream"
            )
        for decoded_index, frame in enumerate(container.decode(streams[0])):
            if decoded_index >= len(ledger.rows):
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} decoded more frames than its sidecar"
                )
            row = ledger.rows[decoded_index]
            if (
                _frame_pts_ns(frame) != row.relative_pts_ns
                or frame.width != ledger.record.width
                or frame.height != ledger.record.height
            ):
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} decoded frame differs from registered evidence"
                )
            decoded_count += 1
            if decoded_index in sentinel_indexes:
                timing = quality_timings[decoded_index]
                observations.append(
                    analyzer.observe(
                        pyav_decoded_frame_view(
                            frame,
                            timestamp_ns=timing.aligned_timestamp_ns,
                            analysis_width=quality_policy.analysis_width,
                        ),
                        timing,
                    )
                )
                observed_indexes.add(decoded_index)
            source_frame = selected_by_index.get(decoded_index)
            if source_frame is not None:
                artifacts[(camera_id, source_frame.source_frame_id)] = _materialized_frame_artifact(
                    source_frame=source_frame,
                    decoded_frame=frame,
                    output_root=output_root,
                    evidence_max_width=evidence_max_width,
                    evidence_extractor_version=evidence_extractor_version,
                )
                rendered_indexes.add(decoded_index)
            if last_required_index is not None and decoded_index >= last_required_index:
                break
    if last_required_index is None and decoded_count != len(ledger.rows):
        raise CanonicalMcapSourceError(
            f"{camera_id.value} decoded frame count differs from its sidecar"
        )
    if last_required_index is not None and decoded_count != last_required_index + 1:
        raise CanonicalMcapSourceError(
            f"{camera_id.value} did not decode through its final selected source frame"
        )
    if rendered_indexes != set(selected_by_index):
        raise CanonicalMcapSourceError(
            f"{camera_id.value} did not materialize every selected source frame"
        )
    if observed_indexes != set(sentinel_indexes):
        raise CanonicalMcapSourceError(
            f"{camera_id.value} did not observe every selected sentinel frame"
        )
    return artifacts, tuple(observations)


def _materialize_verified_source_frame(
    *,
    camera_id: CameraId,
    ledger: _CameraLedger,
    source_frame: IndexedSourceFrame,
    output_root: Path,
    evidence_max_width: int = MCAP_PNG_MAX_WIDTH,
    evidence_extractor_version: str = MCAP_PNG_EXTRACTOR_VERSION,
) -> MaterializedFrameArtifactFact:
    """Seek from verified keyframe evidence and materialize one exact source frame."""

    packet_index = source_frame.source_locator.get("packet_index")
    expected_locator = {
        "camera_id": camera_id.value,
        "packet_index": source_frame.source_order,
        "source_log_time_ns": str(source_frame.source_timestamp_ns),
        "timestamp_sidecar_sha256": ledger.sidecar_sha256,
        "video_sha256": ledger.record.video_artifact.sha256,
    }
    if (
        isinstance(packet_index, bool)
        or not isinstance(packet_index, int)
        or packet_index != source_frame.source_order
        or packet_index >= len(ledger.rows)
        or source_frame.source_locator != expected_locator
    ):
        raise CanonicalMcapSourceError(
            f"{camera_id.value} artifact request has an invalid verified source locator"
        )
    row = ledger.rows[packet_index]
    if (
        row.packet_index != packet_index
        or row.camera_id is not camera_id
        or row.source_log_time_ns != source_frame.source_timestamp_ns
    ):
        raise CanonicalMcapSourceError(
            f"{camera_id.value} artifact request differs from its timestamp sidecar"
        )

    keyframe_indexes = tuple(
        candidate.packet_index
        for candidate in ledger.rows[: packet_index + 1]
        if candidate.is_keyframe
    )
    if not keyframe_indexes:
        raise CanonicalMcapSourceError(
            f"{camera_id.value} has no verified keyframe before source frame {packet_index}"
        )
    keyframe_row = ledger.rows[keyframe_indexes[-1]]
    row_by_pts = {candidate.relative_pts_ns: candidate for candidate in ledger.rows}
    try:
        with av.open(str(ledger.video_path), mode="r") as container:
            streams = tuple(container.streams.video)
            if len(streams) != 1 or bool(streams[0].codec_context.has_b_frames):
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} registered MP4 is not one non-reordered video stream"
                )
            stream = streams[0]
            if stream.time_base is None:
                raise CanonicalMcapSourceError(
                    f"{camera_id.value} registered MP4 stream has no time base"
                )
            seek_position = Fraction(keyframe_row.relative_pts_ns, 1_000_000_000) / Fraction(
                stream.time_base
            )
            container.seek(
                seek_position.numerator // seek_position.denominator,
                stream=stream,
                any_frame=False,
                backward=True,
            )
            for decoded_frame in container.decode(stream):
                actual_pts_ns = _frame_pts_ns(decoded_frame)
                decoded_row = row_by_pts.get(actual_pts_ns)
                if decoded_row is None:
                    raise CanonicalMcapSourceError(
                        f"{camera_id.value} sought frame PTS is absent from its sidecar"
                    )
                if (
                    decoded_frame.width != ledger.record.width
                    or decoded_frame.height != ledger.record.height
                ):
                    raise CanonicalMcapSourceError(
                        f"{camera_id.value} sought frame dimensions differ from registered evidence"
                    )
                if decoded_row.packet_index == packet_index:
                    return _materialized_frame_artifact(
                        source_frame=source_frame,
                        decoded_frame=decoded_frame,
                        output_root=output_root,
                        evidence_max_width=evidence_max_width,
                        evidence_extractor_version=evidence_extractor_version,
                    )
                if actual_pts_ns > row.relative_pts_ns:
                    break
    except CanonicalMcapSourceError:
        raise
    except Exception as error:
        raise CanonicalMcapSourceError(
            f"{camera_id.value} could not seek and decode source frame {packet_index}: {error}"
        ) from error
    raise CanonicalMcapSourceError(
        f"{camera_id.value} did not decode requested source frame {packet_index}"
    )


def _materialized_frame_artifact_from_rendered(
    *,
    source_frame: IndexedSourceFrame,
    rendered: _RenderedPngFact,
    evidence_extractor_version: str,
) -> MaterializedFrameArtifactFact:
    artifact_identity = semantic_sha256(
        {
            "extractor_version": evidence_extractor_version,
            "source_frame_id": source_frame.source_frame_id,
            "png_sha256": rendered.sha256,
        }
    )
    return MaterializedFrameArtifactFact(
        artifact=MaterializedArtifactManifest(
            artifact_id=_stable_uuid(
                "canonical-mcap-frame-artifact",
                artifact_identity,
            ),
            uri=rendered.path.as_uri(),
            sha256=rendered.sha256,
            bytes=rendered.bytes,
            media_type="image/png",
        ),
        width=rendered.width,
        height=rendered.height,
        quality_flags=(
            "LOCAL_CONFORMANCE",
            "REAL_MCAP_H264_DECODED",
        ),
    )


def _materialized_frame_artifact(
    *,
    source_frame: IndexedSourceFrame,
    decoded_frame: Any,
    output_root: Path,
    evidence_max_width: int = MCAP_PNG_MAX_WIDTH,
    evidence_extractor_version: str = MCAP_PNG_EXTRACTOR_VERSION,
) -> MaterializedFrameArtifactFact:
    png_bytes, width, height = _encode_png(
        decoded_frame,
        max_width=evidence_max_width,
    )
    digest = exact_bytes_sha256(png_bytes)
    path = _publish_png(output_root, digest, png_bytes)
    artifact_identity = semantic_sha256(
        {
            "extractor_version": evidence_extractor_version,
            "source_frame_id": source_frame.source_frame_id,
            "png_sha256": digest,
        }
    )
    return MaterializedFrameArtifactFact(
        artifact=MaterializedArtifactManifest(
            artifact_id=_stable_uuid(
                "canonical-mcap-frame-artifact",
                artifact_identity,
            ),
            uri=path.as_uri(),
            sha256=digest,
            bytes=len(png_bytes),
            media_type="image/png",
        ),
        width=width,
        height=height,
        quality_flags=(
            "LOCAL_CONFORMANCE",
            "REAL_MCAP_H264_DECODED",
        ),
    )


def _publish_png(root: Path, digest: str, contents: bytes) -> Path:
    directory = root / "sha256" / digest[:2]
    target = directory / f"{digest}.png"
    directory.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or exact_bytes_sha256(target.read_bytes()) != digest:
            raise CanonicalMcapSourceError(f"existing frame artifact is corrupt: {target}")
        return target.resolve()

    descriptor, temporary = make_temp_file(
        directory,
        prefix=f".{digest}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or exact_bytes_sha256(target.read_bytes()) != digest:
                raise CanonicalMcapSourceError(
                    f"concurrent frame artifact is corrupt: {target}"
                ) from None
        return target.resolve()
    finally:
        temporary.unlink(missing_ok=True)


def _publish_exact_state_file(
    target: Path,
    contents: bytes,
    *,
    label: str,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != contents:
            raise CanonicalMcapSourceError(f"existing {label} bytes are inconsistent")
        return target.resolve()

    descriptor, temporary = make_temp_file(
        target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != contents:
                raise CanonicalMcapSourceError(
                    f"concurrent {label} bytes are inconsistent"
                ) from None
        return target.resolve()
    finally:
        temporary.unlink(missing_ok=True)


def _sampling_plan() -> SamplingPlan:
    return SamplingPlan(
        sampling_plan_id=_stable_uuid(
            "canonical-mcap-sampling-plan",
            MCAP_SAMPLING_POLICY_VERSION,
        ),
        version=MCAP_SAMPLING_POLICY_VERSION,
        qa_sampling_rate_fps=1.0,
        event_sampling_rate_fps=2.0,
        dense_sampling_rate_fps=2.0,
        frame_budget=FrameBudget(
            max_frames_per_camera=1_000,
            max_frames_total=6_000,
            overflow_policy=OverflowPolicy.SPLIT_WINDOW,
        ),
    )


def _component(name: str) -> EvidenceComponent:
    return EvidenceComponent(
        name=name,
        version=MCAP_SOURCE_POLICY_VERSION,
        code_sha256=semantic_sha256(
            {"component": name, "implementation": MCAP_SOURCE_POLICY_VERSION}
        ),
        configuration_sha256=semantic_sha256(
            {"component": name, "configuration": MCAP_SOURCE_POLICY_VERSION}
        ),
    )


def _policy(name: str) -> SemanticPolicyReference:
    return SemanticPolicyReference(
        version=MCAP_SOURCE_POLICY_VERSION,
        semantic_sha256=semantic_sha256({"policy": name, "version": MCAP_SOURCE_POLICY_VERSION}),
    )


def _pass_check(check_id: str, subject: str) -> ValidationCheckEvidenceV2:
    return ValidationCheckEvidenceV2(
        check_id=check_id,
        check_version=MCAP_SOURCE_POLICY_VERSION,
        subject=subject,
        outcome=ValidationCheckOutcome.PASS,
        diagnostic_ids=(),
    )


def _stable_uuid(namespace: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{value}"))


def _rfc3339(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "MCAP_RECORDING_NAMESPACE",
    "MCAP_SAMPLING_POLICY_VERSION",
    "MCAP_SOURCE_POLICY_VERSION",
    "AuthorizedMcapMapping",
    "CanonicalMcapSourceBundle",
    "CanonicalMcapSourceError",
    "authorize_mcap_mapping",
    "load_canonical_mcap_source",
]
