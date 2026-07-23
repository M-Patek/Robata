"""Real MCAP input bridge for the local canonical composition."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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
from robata.adapters.mcap_inspector import OfficialMcapInspector
from robata.adapters.pyav_decoder import PyAvH264DecoderProbe
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
)
from robata.admission.context import AdmissionContextResolver, AdmittedRecordingContextV2
from robata.admission.ledger import (
    AlignmentAdmissionOutcome,
    PrimaryAdmissionEvaluation,
    PrimaryAdmissionPolicy,
    SourceAdmissionOutcome,
)
from robata.application.canonical.media_quality import (
    FrameQualityObservation,
    FrameTimingEvidence,
    LocalFrameQualityAnalyzer,
    LocalMediaQualityReport,
    build_local_media_quality_report,
    registered_local_media_quality_report_document,
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
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import (
    canonical_json_bytes,
    exact_bytes_sha256,
    recording_identity,
    semantic_sha256,
)
from robata.contracts.mcap import MCAPReadyRecording
from robata.contracts.sampling_plan import FrameBudget, OverflowPolicy, SamplingPlan
from robata.contracts.schema_registry import SchemaRegistry
from robata.contracts.video_export import VideoExporterMode
from robata.ingestion.mapping import ExactTopicMappingPolicy, TopicMappingProfile
from robata.ports.ingestion import ChannelInspection, DecoderProbeResult, McapInspection
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


class CanonicalMcapSourceError(ValueError):
    """A raw MCAP cannot produce complete local canonical inputs."""


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


@dataclass(slots=True)
class _VerifiedMcapArtifactResolver:
    """Resolve selected frames against a previously verified MP4/sidecar ledger."""

    frame_index: CanonicalSixCameraFrameIndex
    ledgers: Mapping[CameraId, _CameraLedger]
    requested_interval: NanosecondInterval
    output_root: Path
    artifacts: dict[tuple[CameraId, str], MaterializedFrameArtifactFact]
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
    schema_registry: SchemaRegistry | None = None,
    clock: Callable[[], datetime] | None = None,
    runtime_observer: RuntimeObserver | None = None,
) -> CanonicalMcapSourceBundle:
    """Inspect, export, admit, index, and materialize one real six-camera MCAP."""

    try:
        if max_duration_ns is not None:
            if isinstance(max_duration_ns, bool) or not isinstance(max_duration_ns, int):
                raise TypeError("max_duration_ns must be an integer or None")
            if max_duration_ns <= 0:
                raise ValueError("max_duration_ns must be positive")
        return _load_canonical_mcap_source(
            Path(source),
            authorization=authorization,
            state_dir=Path(state_dir),
            expected_source_sha256=expected_source_sha256,
            max_duration_ns=max_duration_ns,
            schema_registry=schema_registry or SchemaRegistry(),
            clock=clock or (lambda: datetime.now(tz=UTC)),
            runtime_observer=runtime_observer,
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
    schema_registry: SchemaRegistry,
    clock: Callable[[], datetime],
    runtime_observer: RuntimeObserver | None,
) -> CanonicalMcapSourceBundle:
    observed_at = _rfc3339(clock())
    stage_attributes = {"camera_count": len(CAMERA_IDS)}
    with runtime_span(runtime_observer, "source.inspect", stage_attributes):
        inspection = OfficialMcapInspector().inspect(source)
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
    if inspection.source_sha256 != expected_source_sha256:
        raise CanonicalMcapSourceError("source bytes changed after the run identity was derived")
    with runtime_span(runtime_observer, "source.mapping.resolve", stage_attributes):
        channels = authorization.policy.resolve(inspection)
    with runtime_span(runtime_observer, "source.demux.probe", stage_attributes):
        probes = _probe_channels(source, channels)

    state_dir.mkdir(parents=True, exist_ok=True)
    with runtime_span(runtime_observer, "source.export", stage_attributes):
        publication = _export_registered_videos(
            source=source,
            state_dir=state_dir,
            inspection=inspection,
            channels=channels,
            authorization=authorization,
            schema_registry=schema_registry,
            clock=clock,
            runtime_observer=runtime_observer,
        )
    with runtime_span(runtime_observer, "source.export.validate", stage_attributes):
        view, manifest = _validate_publication(publication)
    with runtime_span(runtime_observer, "source.decode.ledger_load", stage_attributes):
        ledgers = {
            record.camera_id: _load_camera_ledger(view, manifest, record)
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
            probes=probes,
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
    materialize_attributes = {
        **stage_attributes,
        "max_duration_limited": window_limited,
    }
    with runtime_span(runtime_observer, "source.materialize", materialize_attributes):
        artifacts, quality_observations = _materialize_selected_frames(
            frame_index=frame_index,
            ledgers=ledgers,
            quality_timings=quality_timings,
            requested_interval=requested_interval,
            output_root=frame_output_root,
            stop_after_selected=window_limited,
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


def _probe_channels(
    source: Path,
    channels: SixCameraMap[ChannelInspection],
) -> dict[CameraId, DecoderProbeResult]:
    probe = PyAvH264DecoderProbe()
    results: dict[CameraId, DecoderProbeResult] = {}
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
        result = probe.probe(source, channel)
        if (
            not result.success
            or result.decoded_frames < 1
            or result.width is None
            or result.height is None
        ):
            raise CanonicalMcapSourceError(
                f"{camera_id.value} did not pass the real H264 decoder probe"
            )
        results[camera_id] = result
    return results


def _export_registered_videos(
    *,
    source: Path,
    state_dir: Path,
    inspection: McapInspection,
    channels: SixCameraMap[ChannelInspection],
    authorization: AuthorizedMcapMapping,
    schema_registry: SchemaRegistry,
    clock: Callable[[], datetime],
    runtime_observer: RuntimeObserver | None,
) -> PublishedRegisteredVideoExport:
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
    service = RegisteredSixCameraVideoExportService(
        PyAvH264Mp4Exporter(),
        LocalArtifactRegistry(
            state_dir / "artifact-registry",
            runtime_observer=runtime_observer,
        ),
        schema_registry,
        clock=clock,
    )
    return service.export_local(
        LocalVideoExportRequest(
            source=source,
            output_directory=state_dir / "video-view",
            namespace=MCAP_RECORDING_NAMESPACE,
            inspection=inspection,
            channels=channels,
            mapping_profile=authorization.profile,
            mapping_profile_digest=authorization.semantic_sha256,
            exporter=descriptor,
        )
    )


def _build_stream_records(
    *,
    inspection: McapInspection,
    channels: SixCameraMap[ChannelInspection],
    probes: Mapping[CameraId, DecoderProbeResult],
) -> dict[CameraId, tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2]]:
    schema_policy = _policy("protobuf-compressed-image-schema")
    probe_component = _component("pyav-h264-decoder-probe")
    records: dict[CameraId, tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2]] = {}
    for camera_id in CAMERA_IDS:
        channel = channels[camera_id]
        result = probes[camera_id]
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
            codec=result.codec,
            message_count=channel.message_count,
            first_timestamp_ns=channel.first_message_time_ns,
            last_timestamp_ns=channel.last_message_time_ns,
            decoder_probe=DecoderProbeEvidenceV2(
                probe=probe_component,
                outcome=DecoderProbeOutcome.PASSED,
                decoded_frame_count=result.decoded_frames,
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
    stop_after_selected: bool = False,
) -> tuple[
    dict[tuple[CameraId, str], MaterializedFrameArtifactFact],
    dict[CameraId, tuple[FrameQualityObservation, ...]],
]:
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[tuple[CameraId, str], MaterializedFrameArtifactFact] = {}
    quality_observations: dict[CameraId, tuple[FrameQualityObservation, ...]] = {}
    grid = SamplingGrid(grid_origin_ns=0, rate=SamplingRate(2, 1))
    for camera_id in CAMERA_IDS:
        source_frames = frame_index.cameras[camera_id].frames
        frames_by_locator = {
            canonical_json_bytes(frame.source_locator): frame for frame in source_frames
        }
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
            300_000_000,
        )
        selected_by_index: dict[int, IndexedSourceFrame] = {}
        for selection in selections:
            if selection.status is not SelectionStatus.SELECTED:
                continue
            assert selection.frame is not None
            source_frame = frames_by_locator[selection.frame.source_locator_bytes]
            packet_index = source_frame.source_locator["packet_index"]
            if isinstance(packet_index, bool) or not isinstance(packet_index, int):
                raise CanonicalMcapSourceError("canonical packet locator is not an integer")
            selected_by_index[packet_index] = source_frame
        quality_observations[camera_id] = _decode_selected_camera_frames(
            camera_id=camera_id,
            ledger=ledgers[camera_id],
            quality_timings=quality_timings[camera_id],
            selected_by_index=selected_by_index,
            output_root=output_root,
            artifacts=artifacts,
            stop_after_selected=stop_after_selected,
        )
    return artifacts, quality_observations


def _decode_selected_camera_frames(
    *,
    camera_id: CameraId,
    ledger: _CameraLedger,
    quality_timings: tuple[FrameTimingEvidence, ...],
    selected_by_index: Mapping[int, IndexedSourceFrame],
    output_root: Path,
    artifacts: dict[tuple[CameraId, str], MaterializedFrameArtifactFact],
    stop_after_selected: bool = False,
) -> tuple[FrameQualityObservation, ...]:
    if not isinstance(stop_after_selected, bool):
        raise TypeError("stop_after_selected must be a boolean")
    if stop_after_selected and not selected_by_index:
        return ()
    if len(quality_timings) != len(ledger.rows):
        raise CanonicalMcapSourceError(
            f"{camera_id.value} quality timing differs from its verified timestamp sidecar"
        )

    decoded_count = 0
    rendered_indexes: set[int] = set()
    observations: list[FrameQualityObservation] = []
    analyzer = LocalFrameQualityAnalyzer(camera_id)
    last_required_index = max(selected_by_index) if stop_after_selected else None
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
            observations.append(analyzer.observe(frame, quality_timings[decoded_index]))
            source_frame = selected_by_index.get(decoded_index)
            if source_frame is None:
                continue
            artifacts[(camera_id, source_frame.source_frame_id)] = _materialized_frame_artifact(
                source_frame=source_frame,
                decoded_frame=frame,
                output_root=output_root,
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
    return tuple(observations)


def _materialize_verified_source_frame(
    *,
    camera_id: CameraId,
    ledger: _CameraLedger,
    source_frame: IndexedSourceFrame,
    output_root: Path,
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


def _materialized_frame_artifact(
    *,
    source_frame: IndexedSourceFrame,
    decoded_frame: Any,
    output_root: Path,
) -> MaterializedFrameArtifactFact:
    png_bytes, width, height = _encode_png(
        decoded_frame,
        max_width=MCAP_PNG_MAX_WIDTH,
    )
    digest = exact_bytes_sha256(png_bytes)
    path = _publish_png(output_root, digest, png_bytes)
    artifact_identity = semantic_sha256(
        {
            "extractor_version": MCAP_PNG_EXTRACTOR_VERSION,
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
