"""Atomic orchestration for a complete local six-camera video export."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from robata.alignment.rational_time import round_half_even
from robata.contracts import (
    CAMERA_IDS,
    CameraId,
    Sha256Digest,
    SixCameraMap,
    canonical_json_bytes,
    exact_bytes_sha256,
    recording_identity,
)
from robata.contracts.schema_registry import SchemaRef, default_schema_registry
from robata.contracts.video_export import (
    CameraVideoExportManifest,
    CameraVideoExportRecord,
    CameraVideoTimestampRow,
    DerivedArtifact,
    DroppedMessageProvenance,
    DroppedMessageReasonCode,
    MappingProfileReference,
    MediaTimeMapping,
    SourceVideoStream,
    TailDurationPolicy,
    TimestampSidecarArtifact,
    VideoExportAlignmentStatus,
    VideoExporterIdentity,
    VideoExporterMode,
    VideoExportExecutionMode,
)
from robata.ingestion.mapping import TopicMappingProfile
from robata.ports import COMPRESSED_IMAGE_SCHEMA, ChannelInspection, McapInspection
from robata.ports.video_export import CameraVideoExporter, ExportedCameraVideoFacts
from robata.tempfiles import make_staging_directory

MANIFEST_FILENAME = "camera-video-export-manifest.json"
MANIFEST_SCHEMA_NAME = "camera-video-export-manifest"
MANIFEST_SCHEMA_ID = "https://schemas.robata.dev/camera-video-export-manifest"
MANIFEST_SCHEMA_VERSION = "1.0.0"
TIMESTAMP_SCHEMA_ID = "https://schemas.robata.dev/camera-video-timestamp-row"
TIMESTAMP_SCHEMA_VERSION = "1.0.0"
TIMESTAMP_SIDECAR_MEDIA_TYPE = "application/x-ndjson"
VIDEO_MEDIA_TYPE = "video/mp4"


class VideoExportRunErrorCode(StrEnum):
    """Stable failures owned by the six-camera application service."""

    INVALID_REQUEST = "INVALID_REQUEST"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    EXISTING_OUTPUT_INVALID = "EXISTING_OUTPUT_INVALID"
    DERIVED_ARTIFACT_INVALID = "DERIVED_ARTIFACT_INVALID"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    OUTPUT_IO_ERROR = "OUTPUT_IO_ERROR"
    ATOMIC_PUBLISH_FAILED = "ATOMIC_PUBLISH_FAILED"
    ARTIFACT_REGISTRY_FAILED = "ARTIFACT_REGISTRY_FAILED"
    MATERIALIZED_VIEW_FAILED = "MATERIALIZED_VIEW_FAILED"


class VideoExportRunError(RuntimeError):
    """An orchestration failure with a machine-readable code."""

    def __init__(self, code: VideoExportRunErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class VideoExporterDescriptor:
    """Versioned exporter and canonical configuration identity."""

    name: str
    version: str
    mode: VideoExporterMode
    export_profile_id: str
    profile_version: str
    canonical_config_sha256: Sha256Digest

    def as_contract(self) -> VideoExporterIdentity:
        return VideoExporterIdentity(
            name=self.name,
            version=self.version,
            mode=self.mode,
            export_profile_id=self.export_profile_id,
            profile_version=self.profile_version,
            canonical_config_sha256=self.canonical_config_sha256,
        )


@dataclass(frozen=True, slots=True)
class LocalVideoExportRequest:
    """All verified inputs needed for a non-governed local export."""

    source: Path
    output_directory: Path
    namespace: str
    inspection: McapInspection
    channels: SixCameraMap[ChannelInspection]
    mapping_profile: TopicMappingProfile
    mapping_profile_digest: Sha256Digest
    exporter: VideoExporterDescriptor


@dataclass(frozen=True, slots=True)
class PublishedVideoExport:
    """A verified complete manifest and its immutable output directory."""

    output_directory: Path
    manifest: CameraVideoExportManifest
    manifest_sha256: Sha256Digest
    reused: bool


class SixCameraVideoExportService:
    """Export all six slots into a private staging directory, then publish once."""

    def __init__(self, exporter: CameraVideoExporter, *, max_parallel_exports: int = 1) -> None:
        if isinstance(max_parallel_exports, bool) or not isinstance(max_parallel_exports, int):
            raise TypeError("max_parallel_exports must be an integer")
        if max_parallel_exports <= 0:
            raise ValueError("max_parallel_exports must be positive")
        if max_parallel_exports > len(CAMERA_IDS):
            raise ValueError("max_parallel_exports cannot exceed the six camera slots")
        self._exporter = exporter
        self._max_parallel_exports = max_parallel_exports

    def export_local(self, request: LocalVideoExportRequest) -> PublishedVideoExport:
        self._validate_request(request)
        try:
            output_directory = request.output_directory.resolve()
        except OSError as exc:
            raise VideoExportRunError(
                VideoExportRunErrorCode.OUTPUT_IO_ERROR,
                f"could not resolve export destination: {exc}",
            ) from exc
        if not output_directory.name:
            raise VideoExportRunError(
                VideoExportRunErrorCode.INVALID_REQUEST,
                "export destination must name a child directory",
            )
        if output_directory.exists():
            return self._reuse_existing(request, output_directory)

        try:
            output_directory.parent.mkdir(parents=True, exist_ok=True)
            staging_directory = make_staging_directory(
                output_directory.parent,
                prefix=f".{output_directory.name}.partial-",
            )
        except OSError as exc:
            raise VideoExportRunError(
                VideoExportRunErrorCode.OUTPUT_IO_ERROR,
                f"could not create private export staging directory: {exc}",
            ) from exc
        published = False
        try:
            facts = self._export_all_cameras(request, staging_directory)
            self._verify_source_unchanged(request)
            manifest = self._build_manifest(request, facts)
            manifest_bytes = canonical_json_bytes(manifest)
            _validate_v1_manifest(manifest.model_dump(mode="json"))
            manifest_path = staging_directory / MANIFEST_FILENAME
            _write_new_file(manifest_path, manifest_bytes)
            _sync_directory(staging_directory)
            try:
                staging_directory.rename(output_directory)
            except FileExistsError as exc:
                raise VideoExportRunError(
                    VideoExportRunErrorCode.OUTPUT_EXISTS,
                    f"export output appeared during publication: {output_directory}",
                ) from exc
            except OSError as exc:
                raise VideoExportRunError(
                    VideoExportRunErrorCode.ATOMIC_PUBLISH_FAILED,
                    f"could not publish complete export directory: {exc}",
                ) from exc
            published = True
            _sync_directory(output_directory.parent)
            return PublishedVideoExport(
                output_directory=output_directory,
                manifest=manifest,
                manifest_sha256=exact_bytes_sha256(manifest_bytes),
                reused=False,
            )
        except (ValidationError, ValueError) as exc:
            raise VideoExportRunError(
                VideoExportRunErrorCode.MANIFEST_INVALID,
                f"camera-video export manifest is invalid: {exc}",
            ) from exc
        finally:
            if not published and staging_directory.exists():
                shutil.rmtree(staging_directory)

    def _export_all_cameras(
        self,
        request: LocalVideoExportRequest,
        staging_directory: Path,
    ) -> tuple[ExportedCameraVideoFacts, ...]:
        """Export camera slots, preserving canonical order regardless of completion order."""

        if self._max_parallel_exports == 1:
            return tuple(
                self._export_camera(request, camera_id, staging_directory)
                for camera_id in CAMERA_IDS
            )

        with ThreadPoolExecutor(
            max_workers=self._max_parallel_exports,
            thread_name_prefix="robata-export",
        ) as executor:
            futures = {
                camera_id: executor.submit(
                    self._export_camera,
                    request,
                    camera_id,
                    staging_directory,
                )
                for camera_id in CAMERA_IDS
            }
            # Calling ``result`` in canonical order makes the tuple deterministic while
            # still allowing independent camera work to overlap.
            return tuple(futures[camera_id].result() for camera_id in CAMERA_IDS)

    def _export_camera(
        self,
        request: LocalVideoExportRequest,
        camera_id: CameraId,
        staging_directory: Path,
    ) -> ExportedCameraVideoFacts:
        channel = request.channels[camera_id]
        video_path = staging_directory / f"{camera_id.value}.mp4"
        sidecar_path = staging_directory / f"{camera_id.value}.timestamps.jsonl"
        facts = self._exporter.export(
            request.source,
            camera_id,
            channel,
            video_path,
            sidecar_path,
        )
        try:
            if (
                facts.camera_id is not camera_id
                or facts.channel_id != channel.channel_id
                or facts.topic != channel.topic
                or facts.source_message_count != channel.message_count
                or facts.source_first_log_time_ns != channel.first_message_time_ns
                or facts.source_last_log_time_ns != channel.last_message_time_ns
                or facts.video_path.resolve() != video_path.resolve()
                or facts.sidecar_path.resolve() != sidecar_path.resolve()
                or facts.sidecar_row_count != facts.exported_packet_count
                or facts.time_base_numerator != 1
                or facts.time_base_denominator != 1_000_000_000
                or facts.first_pts_ns != 0
                or facts.last_pts_ns
                != facts.export_last_source_log_time_ns - facts.export_first_source_log_time_ns
                or facts.duration_ns != facts.last_pts_ns + facts.tail_duration_ns
            ):
                raise ValueError("exporter facts do not match the selected source channel")
            _verify_file_facts(video_path, facts.video_size_bytes, facts.video_sha256)
            _verify_file_facts(sidecar_path, facts.sidecar_size_bytes, facts.sidecar_sha256)
            _verify_sidecar_rows(
                sidecar_path,
                camera_id=camera_id,
                export_profile_id=request.exporter.export_profile_id,
                export_profile_version=request.exporter.profile_version,
                expected_count=facts.sidecar_row_count,
                expected_first_source_ns=facts.export_first_source_log_time_ns,
                expected_last_source_ns=facts.export_last_source_log_time_ns,
                expected_first_pts=facts.first_pts_ns,
                expected_last_pts=facts.last_pts_ns,
                expected_last_duration_ns=facts.tail_duration_ns,
                expected_keyframe_count=facts.keyframe_count,
            )
            _sync_file(video_path)
            _sync_file(sidecar_path)
        except (OSError, ValidationError, ValueError) as exc:
            raise VideoExportRunError(
                VideoExportRunErrorCode.DERIVED_ARTIFACT_INVALID,
                f"exported artifacts for {camera_id.value} failed verification: {exc}",
            ) from exc
        return facts

    @staticmethod
    def _validate_request(request: LocalVideoExportRequest) -> None:
        if not request.namespace:
            raise VideoExportRunError(
                VideoExportRunErrorCode.INVALID_REQUEST,
                "recording identity namespace must be nonempty",
            )
        if request.mapping_profile.approved:
            raise VideoExportRunError(
                VideoExportRunErrorCode.INVALID_REQUEST,
                "local development export requires an unapproved mapping profile",
            )
        if request.mapping_profile_digest != request.mapping_profile.semantic_digest:
            raise VideoExportRunError(
                VideoExportRunErrorCode.INVALID_REQUEST,
                "mapping profile digest does not match its parsed semantics",
            )
        try:
            request.exporter.as_contract()
        except ValidationError as exc:
            raise VideoExportRunError(
                VideoExportRunErrorCode.INVALID_REQUEST,
                f"exporter descriptor is invalid: {exc}",
            ) from exc
        if request.inspection.source.resolve() != request.source.resolve():
            raise VideoExportRunError(
                VideoExportRunErrorCode.INVALID_REQUEST,
                "inspection source does not match export source",
            )
        selected_channels = tuple(request.channels.values())
        if len({channel.topic for channel in selected_channels}) != len(CAMERA_IDS) or len(
            {channel.channel_id for channel in selected_channels}
        ) != len(CAMERA_IDS):
            raise VideoExportRunError(
                VideoExportRunErrorCode.INVALID_REQUEST,
                "selected camera channels must have unique topics and channel IDs",
            )
        for camera_id in CAMERA_IDS:
            channel = request.channels[camera_id]
            if request.mapping_profile.topics[camera_id] != channel.topic:
                raise VideoExportRunError(
                    VideoExportRunErrorCode.INVALID_REQUEST,
                    f"mapping profile topic differs from selected {camera_id.value} channel",
                )
            if (
                channel.message_count <= 0
                or channel.schema_name != COMPRESSED_IMAGE_SCHEMA
                or (channel.codec or "").strip().lower() != "h264"
                or not channel.monotonic
                or channel.first_message_time_ns is None
                or channel.last_message_time_ns is None
                or channel.first_message_time_ns > channel.last_message_time_ns
                or channel not in request.inspection.channels
            ):
                raise VideoExportRunError(
                    VideoExportRunErrorCode.INVALID_REQUEST,
                    f"mapped channel facts for {camera_id.value} are not exportable",
                )

    @staticmethod
    def _verify_source_unchanged(request: LocalVideoExportRequest) -> None:
        size_bytes, digest = _hash_file(request.source)
        if (
            size_bytes != request.inspection.source_size_bytes
            or digest != request.inspection.source_sha256
        ):
            raise VideoExportRunError(
                VideoExportRunErrorCode.SOURCE_CHANGED,
                "source bytes changed after inspection and before publication",
            )

    @staticmethod
    def _build_manifest(
        request: LocalVideoExportRequest,
        camera_facts: tuple[ExportedCameraVideoFacts, ...],
    ) -> CameraVideoExportManifest:
        records = tuple(_camera_record(facts) for facts in camera_facts)
        return CameraVideoExportManifest(
            schema_version="1.0",
            execution_mode=VideoExportExecutionMode.LOCAL_DEVELOPMENT_OVERRIDE,
            recording_identity=recording_identity(
                request.namespace,
                request.inspection.source_sha256,
            ),
            source_content_sha256=request.inspection.source_sha256,
            source_size_bytes=request.inspection.source_size_bytes,
            mapping_profile=MappingProfileReference(
                version=request.mapping_profile.version,
                digest=request.mapping_profile_digest,
                approved=False,
            ),
            ready_manifest_id=None,
            alignment_id=None,
            alignment_status=VideoExportAlignmentStatus.UNVERIFIED,
            exporter=request.exporter.as_contract(),
            cameras=records,
        )

    @staticmethod
    def _reuse_existing(
        request: LocalVideoExportRequest,
        output_directory: Path,
    ) -> PublishedVideoExport:
        if not output_directory.is_dir() or output_directory.is_symlink():
            raise VideoExportRunError(
                VideoExportRunErrorCode.OUTPUT_EXISTS,
                f"export destination already exists and is not a directory: {output_directory}",
            )
        manifest_path = output_directory / MANIFEST_FILENAME
        try:
            if not manifest_path.is_file() or manifest_path.is_symlink():
                raise ValueError("manifest is not a regular non-symlink file")
            manifest_bytes = manifest_path.read_bytes()
            payload = json.loads(manifest_bytes)
            _validate_v1_manifest(payload)
            manifest = CameraVideoExportManifest.model_validate_json(
                manifest_bytes,
                strict=True,
            )
            if canonical_json_bytes(manifest) != manifest_bytes:
                raise ValueError("manifest is not canonical JSON")
            _verify_existing_identity(request, manifest)
            _verify_existing_layout(output_directory)
            for record in manifest.cameras:
                _verify_artifact(
                    output_directory / f"{record.camera_id.value}.mp4",
                    record.video_artifact,
                )
                _verify_artifact(
                    output_directory / f"{record.camera_id.value}.timestamps.jsonl",
                    record.timestamp_sidecar_artifact,
                )
                _verify_sidecar_rows(
                    output_directory / f"{record.camera_id.value}.timestamps.jsonl",
                    camera_id=record.camera_id,
                    export_profile_id=manifest.exporter.export_profile_id,
                    export_profile_version=manifest.exporter.profile_version,
                    expected_count=record.timestamp_sidecar_artifact.row_count,
                    expected_first_source_ns=record.export_first_observed_source_message_ns,
                    expected_last_source_ns=record.export_last_observed_source_message_ns,
                    expected_first_pts=record.media_time_mapping.first_pts,
                    expected_last_pts=record.media_time_mapping.last_pts,
                    expected_last_duration_ns=record.media_time_mapping.last_duration,
                    expected_keyframe_count=record.keyframe_count,
                )
                _verify_existing_camera_source(request, record)
            SixCameraVideoExportService._verify_source_unchanged(request)
        except VideoExportRunError:
            raise
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise VideoExportRunError(
                VideoExportRunErrorCode.EXISTING_OUTPUT_INVALID,
                f"existing export cannot be safely reused: {exc}",
            ) from exc
        return PublishedVideoExport(
            output_directory=output_directory,
            manifest=manifest,
            manifest_sha256=exact_bytes_sha256(manifest_bytes),
            reused=True,
        )


def _camera_record(facts: ExportedCameraVideoFacts) -> CameraVideoExportRecord:
    leading = _drop_provenance(
        facts.leading_access_unit_count,
        DroppedMessageReasonCode.BEFORE_FIRST_DECODABLE_KEYFRAME,
        facts.leading_first_log_time_ns,
        facts.leading_last_log_time_ns,
    )
    trailing = _drop_provenance(
        facts.trailing_access_unit_count,
        DroppedMessageReasonCode.AFTER_LAST_COMPLETE_SAMPLE,
        facts.trailing_first_log_time_ns,
        facts.trailing_last_log_time_ns,
    )
    last_duration = round_half_even(
        facts.tail_duration_ns * facts.time_base_denominator,
        1_000_000_000 * facts.time_base_numerator,
    )
    return CameraVideoExportRecord(
        camera_id=facts.camera_id,
        source=SourceVideoStream(
            topic=facts.topic,
            channel_id=facts.channel_id,
            schema_name="foxglove.CompressedImage",
            codec="h264",
        ),
        input_message_count=facts.source_message_count,
        source_first_observed_message_ns=facts.source_first_log_time_ns,
        source_last_observed_message_ns=facts.source_last_log_time_ns,
        export_first_observed_source_message_ns=facts.export_first_source_log_time_ns,
        export_last_observed_source_message_ns=facts.export_last_source_log_time_ns,
        leading_drops=leading,
        trailing_drops=trailing,
        exported_packet_count=facts.exported_packet_count,
        exported_frame_count=facts.decoded_frame_count,
        keyframe_count=facts.keyframe_count,
        width=facts.width,
        height=facts.height,
        video_artifact=DerivedArtifact(
            uri=f"robata-local://video/{facts.camera_id.value}.mp4",
            sha256=facts.video_sha256,
            bytes=facts.video_size_bytes,
            media_type=VIDEO_MEDIA_TYPE,
        ),
        timestamp_sidecar_artifact=TimestampSidecarArtifact(
            uri=f"robata-local://timestamp-map/{facts.camera_id.value}.timestamps.jsonl",
            sha256=facts.sidecar_sha256,
            bytes=facts.sidecar_size_bytes,
            media_type=TIMESTAMP_SIDECAR_MEDIA_TYPE,
            row_count=facts.sidecar_row_count,
        ),
        media_time_mapping=MediaTimeMapping(
            zero_source_ns=facts.export_first_source_log_time_ns,
            time_base_numerator=facts.time_base_numerator,
            time_base_denominator=facts.time_base_denominator,
            first_pts=facts.first_pts_ns,
            last_pts=facts.last_pts_ns,
            last_duration=last_duration,
            tail_duration_policy=TailDurationPolicy(facts.tail_duration_policy),
            rounding="HALF_EVEN",
            max_rounding_error_ns=facts.max_timestamp_mapping_error_ns,
        ),
    )


def _validate_v1_manifest(payload: object) -> None:
    registry = default_schema_registry()
    registered = registry.resolve_version(MANIFEST_SCHEMA_ID, MANIFEST_SCHEMA_VERSION)
    registry.validate_pinned(registered.ref, payload)


def _drop_provenance(
    count: int,
    reason: DroppedMessageReasonCode,
    first_source_ns: int | None,
    last_source_ns: int | None,
) -> DroppedMessageProvenance:
    if count == 0:
        return DroppedMessageProvenance(
            count=0,
            reason_code=DroppedMessageReasonCode.NONE,
            first_source_ns=None,
            last_source_ns=None,
        )
    return DroppedMessageProvenance(
        count=count,
        reason_code=reason,
        first_source_ns=first_source_ns,
        last_source_ns=last_source_ns,
    )


def _verify_existing_identity(
    request: LocalVideoExportRequest,
    manifest: CameraVideoExportManifest,
) -> None:
    expected_recording_identity = recording_identity(
        request.namespace,
        request.inspection.source_sha256,
    )
    if (
        manifest.execution_mode is not VideoExportExecutionMode.LOCAL_DEVELOPMENT_OVERRIDE
        or manifest.recording_identity != expected_recording_identity
        or manifest.source_content_sha256 != request.inspection.source_sha256
        or manifest.source_size_bytes != request.inspection.source_size_bytes
        or manifest.mapping_profile.version != request.mapping_profile.version
        or manifest.mapping_profile.digest != request.mapping_profile_digest
        or manifest.mapping_profile.approved
        or manifest.exporter != request.exporter.as_contract()
    ):
        raise VideoExportRunError(
            VideoExportRunErrorCode.OUTPUT_EXISTS,
            "existing export belongs to different source, mapping, or exporter semantics",
        )


def _verify_existing_camera_source(
    request: LocalVideoExportRequest,
    record: CameraVideoExportRecord,
) -> None:
    channel = request.channels[record.camera_id]
    if (
        record.source.topic != channel.topic
        or record.source.channel_id != channel.channel_id
        or record.source.schema_name != channel.schema_name
        or record.source.codec != (channel.codec or "").strip().lower()
        or record.input_message_count != channel.message_count
        or record.source_first_observed_message_ns != channel.first_message_time_ns
        or record.source_last_observed_message_ns != channel.last_message_time_ns
        or record.video_artifact.uri != f"robata-local://video/{record.camera_id.value}.mp4"
        or record.timestamp_sidecar_artifact.uri
        != (f"robata-local://timestamp-map/{record.camera_id.value}.timestamps.jsonl")
    ):
        raise VideoExportRunError(
            VideoExportRunErrorCode.OUTPUT_EXISTS,
            f"existing export source facts differ for {record.camera_id.value}",
        )


def _verify_artifact(path: Path, artifact: DerivedArtifact) -> None:
    _verify_file_facts(path, artifact.bytes, artifact.sha256)


def _verify_file_facts(path: Path, expected_bytes: int, expected_digest: Sha256Digest) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"artifact is not a regular non-symlink file: {path.name}")
    size_bytes, digest = _hash_file(path)
    if size_bytes != expected_bytes or digest != expected_digest:
        raise ValueError(f"artifact bytes do not match declared facts: {path.name}")


def _verify_existing_layout(output_directory: Path) -> None:
    expected = {MANIFEST_FILENAME}
    for camera_id in CAMERA_IDS:
        expected.add(f"{camera_id.value}.mp4")
        expected.add(f"{camera_id.value}.timestamps.jsonl")
    actual = {child.name for child in output_directory.iterdir()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"export directory layout differs: missing={missing}, extra={extra}")


def _verify_sidecar_rows(
    path: Path,
    *,
    camera_id: CameraId,
    export_profile_id: str,
    export_profile_version: str,
    expected_count: int,
    expected_first_source_ns: int,
    expected_last_source_ns: int,
    expected_first_pts: int,
    expected_last_pts: int,
    expected_last_duration_ns: int,
    expected_keyframe_count: int,
    schema_ref: SchemaRef | None = None,
) -> None:
    if expected_count <= 0:
        raise ValueError("timestamp sidecar expected row count must be positive")
    contents = path.read_bytes()
    if not contents.endswith(b"\n"):
        raise ValueError("timestamp sidecar must end with a newline")
    raw_lines = contents.splitlines()
    if len(raw_lines) != expected_count:
        raise ValueError("timestamp sidecar physical row count differs from declared facts")

    registry = default_schema_registry()
    exact_schema_ref = (
        schema_ref
        or registry.resolve_version(
            TIMESTAMP_SCHEMA_ID,
            TIMESTAMP_SCHEMA_VERSION,
        ).ref
    )
    rows: list[CameraVideoTimestampRow] = []
    for index, raw_line in enumerate(raw_lines):
        row = CameraVideoTimestampRow.model_validate_json(raw_line, strict=True)
        payload = row.model_dump(mode="json")
        registry.validate_pinned(exact_schema_ref, payload)
        if canonical_json_bytes(row) != raw_line:
            raise ValueError(f"timestamp sidecar row {index} is not canonical JSON")
        if (
            row.camera_id is not camera_id
            or row.packet_index != index
            or row.export_profile_id != export_profile_id
            or row.export_profile_version != export_profile_version
            or row.relative_dts_ns != row.relative_pts_ns
            or row.relative_pts_ns != row.source_log_time_ns - expected_first_source_ns
            or row.time_base_numerator != 1
            or row.time_base_denominator != 1_000_000_000
            or row.duration_is_estimated != (index == expected_count - 1)
        ):
            raise ValueError(f"timestamp sidecar row {index} violates export provenance")
        if rows and row.source_log_time_ns <= rows[-1].source_log_time_ns:
            raise ValueError("timestamp sidecar source times must be strictly increasing")
        rows.append(row)

    if (
        rows[0].source_log_time_ns != expected_first_source_ns
        or rows[-1].source_log_time_ns != expected_last_source_ns
        or rows[0].relative_pts_ns != expected_first_pts
        or rows[-1].relative_pts_ns != expected_last_pts
        or rows[-1].duration_ns != expected_last_duration_ns
        or sum(int(row.is_keyframe) for row in rows) != expected_keyframe_count
    ):
        raise ValueError("timestamp sidecar aggregate facts do not match the export")


def _hash_file(path: Path) -> tuple[int, Sha256Digest]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, digest.hexdigest()


def _write_new_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "MANIFEST_FILENAME",
    "LocalVideoExportRequest",
    "PublishedVideoExport",
    "SixCameraVideoExportService",
    "VideoExportRunError",
    "VideoExportRunErrorCode",
    "VideoExporterDescriptor",
]
