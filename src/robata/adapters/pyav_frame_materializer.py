"""Verified PyAV frame materialization for registered six-camera video views."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Never
from uuid import NAMESPACE_URL, uuid5

import av
from pydantic import ValidationError

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.pipeline import (
    CameraPackage,
    CameraPackageStatus,
    MaterializedFrame,
    SamplingPurpose,
    SamplingStrategy,
    SamplingSummary,
    TemporalVisualPackage,
)
from robata.contracts.video_export import CameraVideoTimestampRow
from robata.contracts.video_export_v2 import (
    CameraVideoExportManifestV2,
    CameraVideoExportRecordV2,
)
from robata.ports.frame_materialization import (
    FrameMaterializationError,
    FrameMaterializationErrorCode,
    FrameMaterializationRequest,
)
from robata.sampling import FrameCandidate, SamplingGrid, SamplingRate, SelectionStatus
from robata.tempfiles import make_staging_directory

_MANIFEST_FILENAME = "camera-video-export-manifest.json"
_NANOSECONDS_PER_SECOND = 1_000_000_000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PRODUCER_VERSION = "0.1.0"
_QUALITY_FLAGS = ("LOCAL_UNVERIFIED_IDENTITY_ALIGNMENT",)


@dataclass(frozen=True, slots=True)
class _CameraLedger:
    record: CameraVideoExportRecordV2
    video_path: Path
    sidecar_sha256: str
    rows: tuple[CameraVideoTimestampRow, ...]


@dataclass(frozen=True, slots=True)
class _SelectedSourceFrame:
    ordinal: int
    packet_index: int
    target_timestamp_ns: int
    aligned_timestamp_ns: int
    source_timestamp_ns: int
    delta_to_target_ns: int


@dataclass(frozen=True, slots=True)
class _CameraPlan:
    ledger: _CameraLedger
    selected: tuple[_SelectedSourceFrame, ...]
    target_count: int


@dataclass(frozen=True, slots=True)
class _RenderedFrame:
    source: _SelectedSourceFrame
    filename: str
    sha256: str
    width: int
    height: int


def _fail(code: FrameMaterializationErrorCode, message: str) -> Never:
    raise FrameMaterializationError(code, message)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(
            FrameMaterializationErrorCode.INVALID_REQUEST,
            "materializer clock must return a timezone-aware datetime",
        )
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _expected_view_names() -> set[str]:
    names = {_MANIFEST_FILENAME}
    for camera_id in CAMERA_IDS:
        names.add(f"{camera_id.value}.mp4")
        names.add(f"{camera_id.value}.timestamps.jsonl")
    return names


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(
            FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
            f"video view member is not a regular file: {path.name}",
        )


def _read_regular_file(path: Path) -> bytes:
    _require_regular_file(path)
    try:
        return path.read_bytes()
    except OSError as error:
        _fail(
            FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
            f"cannot read video view member {path.name}: {error}",
        )


def _hash_regular_file(path: Path) -> tuple[int, str]:
    _require_regular_file(path)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        _fail(
            FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
            f"cannot hash video view member {path.name}: {error}",
        )
    return size, digest.hexdigest()


def _validate_publication(
    publication: object,
) -> tuple[Path, CameraVideoExportManifestV2]:
    output_directory = getattr(publication, "output_directory", None)
    supplied_manifest = getattr(publication, "manifest", None)
    supplied_manifest_sha256 = getattr(publication, "manifest_sha256", None)
    if not isinstance(output_directory, Path):
        _fail(
            FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
            "video export publication has no Path output_directory",
        )
    if not isinstance(supplied_manifest, CameraVideoExportManifestV2):
        _fail(
            FrameMaterializationErrorCode.INVALID_MANIFEST,
            "video export publication has no V2 manifest",
        )

    try:
        manifest = CameraVideoExportManifestV2.model_validate(
            supplied_manifest.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as error:
        _fail(
            FrameMaterializationErrorCode.INVALID_MANIFEST,
            f"published V2 manifest is invalid: {error}",
        )

    try:
        view = Path(os.path.abspath(output_directory))
        if view.is_symlink() or not view.is_dir():
            _fail(
                FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
                "published video view is not a regular directory",
            )
        actual_names = {child.name for child in view.iterdir()}
    except OSError as error:
        _fail(
            FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
            f"cannot inspect published video view: {error}",
        )
    expected_names = _expected_view_names()
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        _fail(
            FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
            f"V2 video view must contain exactly 13 files; missing={missing}, extra={extra}",
        )

    manifest_bytes = _read_regular_file(view / _MANIFEST_FILENAME)
    manifest_digest = exact_bytes_sha256(manifest_bytes)
    if manifest_digest != supplied_manifest_sha256:
        _fail(
            FrameMaterializationErrorCode.INVALID_MANIFEST,
            "manifest file digest differs from the publication digest",
        )
    try:
        parsed_manifest = CameraVideoExportManifestV2.model_validate_json(
            manifest_bytes,
            strict=True,
        )
    except ValidationError as error:
        _fail(
            FrameMaterializationErrorCode.INVALID_MANIFEST,
            f"manifest file is invalid: {error}",
        )
    if canonical_json_bytes(parsed_manifest) != manifest_bytes:
        _fail(
            FrameMaterializationErrorCode.INVALID_MANIFEST,
            "manifest file is not canonical JSON",
        )
    if parsed_manifest != manifest:
        _fail(
            FrameMaterializationErrorCode.INVALID_MANIFEST,
            "manifest file differs from the published manifest",
        )
    return view, manifest


def _load_camera_ledger(
    view: Path,
    manifest: CameraVideoExportManifestV2,
    record: CameraVideoExportRecordV2,
    *,
    verify_video_digest: bool = True,
) -> _CameraLedger:
    camera_id = record.camera_id
    video_path = view / f"{camera_id.value}.mp4"
    if not isinstance(verify_video_digest, bool):
        _fail(
            FrameMaterializationErrorCode.INVALID_REQUEST,
            "verify_video_digest must be a boolean",
        )
    if verify_video_digest:
        video_size, video_sha256 = _hash_regular_file(video_path)
        if (
            video_size != record.video_artifact.bytes
            or video_sha256 != record.video_artifact.sha256
        ):
            _fail(
                FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
                f"{camera_id.value} MP4 size or digest differs from its manifest summary",
            )
    else:
        try:
            video_stat = video_path.lstat()
        except OSError as error:
            _fail(
                FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
                f"{camera_id.value} MP4 cannot be inspected: {error}",
            )
        if (
            video_path.is_symlink()
            or not video_path.is_file()
            or video_stat.st_size != record.video_artifact.bytes
        ):
            _fail(
                FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
                f"{camera_id.value} MP4 shape differs from its manifest summary",
            )

    sidecar_path = view / f"{camera_id.value}.timestamps.jsonl"
    sidecar_bytes = _read_regular_file(sidecar_path)
    sidecar_artifact = record.timestamp_sidecar_artifact.artifact
    sidecar_sha256 = exact_bytes_sha256(sidecar_bytes)
    if len(sidecar_bytes) != sidecar_artifact.bytes or sidecar_sha256 != sidecar_artifact.sha256:
        _fail(
            FrameMaterializationErrorCode.INVALID_VIDEO_VIEW,
            f"{camera_id.value} timestamp sidecar differs from its manifest summary",
        )
    if not sidecar_bytes.endswith(b"\n"):
        _fail(
            FrameMaterializationErrorCode.INVALID_SIDECAR,
            f"{camera_id.value} timestamp sidecar must end with a newline",
        )
    raw_lines = sidecar_bytes.splitlines()
    expected_count = record.timestamp_sidecar_artifact.row_count
    if (
        len(raw_lines) != expected_count
        or expected_count != record.exported_packet_count
        or record.exported_frame_count != record.exported_packet_count
    ):
        _fail(
            FrameMaterializationErrorCode.INVALID_SIDECAR,
            f"{camera_id.value} packet, frame, and sidecar row counts disagree",
        )

    mapping = record.media_time_mapping
    rows: list[CameraVideoTimestampRow] = []
    for packet_index, raw_line in enumerate(raw_lines):
        try:
            row = CameraVideoTimestampRow.model_validate_json(raw_line, strict=True)
        except ValidationError as error:
            _fail(
                FrameMaterializationErrorCode.INVALID_SIDECAR,
                f"{camera_id.value} timestamp row {packet_index} is invalid: {error}",
            )
        if canonical_json_bytes(row) != raw_line:
            _fail(
                FrameMaterializationErrorCode.INVALID_SIDECAR,
                f"{camera_id.value} timestamp row {packet_index} is not canonical JSON",
            )
        if (
            row.camera_id is not camera_id
            or row.packet_index != packet_index
            or row.export_profile_id != manifest.exporter.export_profile_id
            or row.export_profile_version != manifest.exporter.profile_version
            or row.relative_pts_ns != row.source_log_time_ns - mapping.zero_source_ns
            or row.relative_dts_ns != row.relative_pts_ns
            or row.time_base_numerator != mapping.time_base_numerator
            or row.time_base_denominator != mapping.time_base_denominator
            or row.duration_is_estimated != (packet_index == expected_count - 1)
        ):
            _fail(
                FrameMaterializationErrorCode.INVALID_SIDECAR,
                f"{camera_id.value} timestamp row {packet_index} violates export provenance",
            )
        if rows and row.source_log_time_ns <= rows[-1].source_log_time_ns:
            _fail(
                FrameMaterializationErrorCode.INVALID_SIDECAR,
                f"{camera_id.value} source timestamps are not strictly increasing",
            )
        rows.append(row)

    if (
        rows[0].source_log_time_ns != record.export_first_observed_source_message_ns
        or rows[-1].source_log_time_ns != record.export_last_observed_source_message_ns
        or rows[0].relative_pts_ns != mapping.first_pts
        or rows[-1].relative_pts_ns != mapping.last_pts
        or rows[-1].duration_ns != mapping.last_duration
        or sum(int(row.is_keyframe) for row in rows) != record.keyframe_count
    ):
        _fail(
            FrameMaterializationErrorCode.INVALID_SIDECAR,
            f"{camera_id.value} timestamp aggregates differ from the manifest summary",
        )
    return _CameraLedger(
        record=record,
        video_path=video_path,
        sidecar_sha256=sidecar_sha256,
        rows=tuple(rows),
    )


def _build_camera_plan(
    ledger: _CameraLedger,
    *,
    recording_origin_ns: int,
    grid: SamplingGrid,
    request: FrameMaterializationRequest,
) -> _CameraPlan:
    locator_to_index: dict[bytes, int] = {}
    candidates: list[FrameCandidate] = []
    for row in ledger.rows:
        locator = canonical_json_bytes(
            {
                "camera_id": ledger.record.camera_id.value,
                "packet_index": row.packet_index,
                "video_sha256": ledger.record.video_artifact.sha256,
            }
        )
        locator_to_index[locator] = row.packet_index
        candidates.append(
            FrameCandidate(
                aligned_timestamp_ns=row.source_log_time_ns - recording_origin_ns,
                source_timestamp_ns=row.source_log_time_ns,
                source_locator_bytes=locator,
            )
        )

    interval = request.window.interval
    selections = grid.select_frames(
        candidates,
        interval.start_ns,
        interval.end_ns,
        request.selection_tolerance_ns,
    )
    selected: list[_SelectedSourceFrame] = []
    for selection in selections:
        if selection.status is not SelectionStatus.SELECTED:
            continue
        assert selection.frame is not None
        assert selection.delta_to_target_ns is not None
        selected.append(
            _SelectedSourceFrame(
                ordinal=len(selected),
                packet_index=locator_to_index[selection.frame.source_locator_bytes],
                target_timestamp_ns=selection.target_ns,
                aligned_timestamp_ns=selection.frame.aligned_timestamp_ns,
                source_timestamp_ns=selection.frame.source_timestamp_ns,
                delta_to_target_ns=selection.delta_to_target_ns,
            )
        )
    return _CameraPlan(ledger=ledger, selected=tuple(selected), target_count=len(selections))


def _frame_pts_ns(frame: Any) -> int:
    if frame.pts is None or frame.time_base is None:
        _fail(
            FrameMaterializationErrorCode.TIMESTAMP_MISMATCH,
            "decoded frame has no PTS or time base",
        )
    exact_ns = Fraction(frame.pts) * Fraction(frame.time_base) * _NANOSECONDS_PER_SECOND
    if exact_ns.denominator != 1:
        _fail(
            FrameMaterializationErrorCode.TIMESTAMP_MISMATCH,
            "decoded frame PTS cannot be represented as exact integer nanoseconds",
        )
    return exact_ns.numerator


def _encode_png(frame: Any, *, max_width: int | None) -> tuple[bytes, int, int]:
    try:
        output_width = frame.width if max_width is None else min(frame.width, max_width)
        output_height = max(
            1,
            (frame.height * output_width + frame.width // 2) // frame.width,
        )
        rgb_frame = frame.reformat(
            width=output_width,
            height=output_height,
            format="rgb24",
        )
        rgb_frame.pts = 0
        rgb_frame.time_base = Fraction(1, 1)
        encoder = av.CodecContext.create("png", "w")
        encoder.width = rgb_frame.width
        encoder.height = rgb_frame.height
        encoder.pix_fmt = "rgb24"
        encoder.time_base = Fraction(1, 1)
        packets = list(encoder.encode(rgb_frame))
        packets.extend(encoder.encode(None))
        png_bytes = b"".join(bytes(packet) for packet in packets)
    except Exception as error:
        _fail(
            FrameMaterializationErrorCode.PNG_ENCODE_FAILED,
            f"PyAV could not encode a selected frame as PNG: {error}",
        )
    if not png_bytes.startswith(_PNG_SIGNATURE):
        _fail(
            FrameMaterializationErrorCode.PNG_ENCODE_FAILED,
            "PyAV PNG encoder returned invalid output",
        )
    return png_bytes, rgb_frame.width, rgb_frame.height


def _write_new_file(path: Path, contents: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        _fail(
            FrameMaterializationErrorCode.OUTPUT_IO_ERROR,
            f"cannot write materialized frame {path.name}: {error}",
        )


def _decode_and_render(
    plan: _CameraPlan,
    staging: Path,
    *,
    max_width: int | None,
) -> tuple[_RenderedFrame, ...]:
    record = plan.ledger.record
    camera_id = record.camera_id
    selected_by_index = {item.packet_index: item for item in plan.selected}
    if len(selected_by_index) != len(plan.selected):
        _fail(
            FrameMaterializationErrorCode.INVALID_REQUEST,
            f"{camera_id.value} sampling selected one source frame more than once",
        )

    camera_directory = staging / camera_id.value
    if plan.selected:
        try:
            camera_directory.mkdir()
        except OSError as error:
            _fail(
                FrameMaterializationErrorCode.OUTPUT_IO_ERROR,
                f"cannot create {camera_id.value} frame directory: {error}",
            )

    rendered_by_index: dict[int, _RenderedFrame] = {}
    decoded_count = 0
    try:
        with av.open(str(plan.ledger.video_path), mode="r") as container:
            video_streams = tuple(container.streams.video)
            if len(video_streams) != 1:
                _fail(
                    FrameMaterializationErrorCode.DECODE_FAILED,
                    f"{camera_id.value} MP4 must contain exactly one video stream",
                )
            stream = video_streams[0]
            if bool(stream.codec_context.has_b_frames):
                _fail(
                    FrameMaterializationErrorCode.TIMESTAMP_MISMATCH,
                    f"{camera_id.value} MP4 unexpectedly contains reordered B-frames",
                )
            for decoded_index, frame in enumerate(container.decode(stream)):
                if decoded_index >= len(plan.ledger.rows):
                    _fail(
                        FrameMaterializationErrorCode.TIMESTAMP_MISMATCH,
                        f"{camera_id.value} decoded more frames than its timestamp sidecar",
                    )
                row = plan.ledger.rows[decoded_index]
                actual_pts_ns = _frame_pts_ns(frame)
                if actual_pts_ns != row.relative_pts_ns:
                    _fail(
                        FrameMaterializationErrorCode.TIMESTAMP_MISMATCH,
                        f"{camera_id.value} frame {decoded_index} PTS {actual_pts_ns} "
                        f"does not match sidecar PTS {row.relative_pts_ns}",
                    )
                if frame.width != record.width or frame.height != record.height:
                    _fail(
                        FrameMaterializationErrorCode.DECODE_FAILED,
                        f"{camera_id.value} frame {decoded_index} dimensions differ from manifest",
                    )
                decoded_count += 1
                selected = selected_by_index.get(decoded_index)
                if selected is None:
                    continue
                png_bytes, output_width, output_height = _encode_png(
                    frame,
                    max_width=max_width,
                )
                filename = f"{selected.ordinal:04d}-{selected.packet_index:08d}.png"
                _write_new_file(camera_directory / filename, png_bytes)
                rendered_by_index[decoded_index] = _RenderedFrame(
                    source=selected,
                    filename=filename,
                    sha256=exact_bytes_sha256(png_bytes),
                    width=output_width,
                    height=output_height,
                )
    except FrameMaterializationError:
        raise
    except Exception as error:
        _fail(
            FrameMaterializationErrorCode.DECODE_FAILED,
            f"PyAV could not decode {camera_id.value}: {error}",
        )

    if decoded_count != len(plan.ledger.rows) or decoded_count != record.exported_frame_count:
        _fail(
            FrameMaterializationErrorCode.TIMESTAMP_MISMATCH,
            f"{camera_id.value} decoded frame count differs from sidecar and manifest",
        )
    if len(rendered_by_index) != len(plan.selected):
        _fail(
            FrameMaterializationErrorCode.DECODE_FAILED,
            f"{camera_id.value} did not decode every selected source frame",
        )
    return tuple(rendered_by_index[item.packet_index] for item in plan.selected)


def _content_projection(
    manifest: CameraVideoExportManifestV2,
    request: FrameMaterializationRequest,
    plans: tuple[_CameraPlan, ...],
    rendered: dict[CameraId, tuple[_RenderedFrame, ...]],
) -> dict[str, Any]:
    cameras: list[dict[str, Any]] = []
    for plan in plans:
        camera_id = plan.ledger.record.camera_id
        cameras.append(
            {
                "camera_id": camera_id.value,
                "video_sha256": plan.ledger.record.video_artifact.sha256,
                "timestamp_sidecar_sha256": plan.ledger.sidecar_sha256,
                "target_count": plan.target_count,
                "frames": [
                    {
                        "source_frame_index": frame.source.packet_index,
                        "target_timestamp_ns": str(frame.source.target_timestamp_ns),
                        "aligned_timestamp_ns": str(frame.source.aligned_timestamp_ns),
                        "source_timestamp_ns": str(frame.source.source_timestamp_ns),
                        "delta_to_target_ns": str(frame.source.delta_to_target_ns),
                        "artifact_sha256": frame.sha256,
                        "width": frame.width,
                        "height": frame.height,
                        "quality_flags": _QUALITY_FLAGS,
                    }
                    for frame in rendered[camera_id]
                ],
            }
        )
    return {
        "schema_version": "1.0",
        "video_manifest_semantic_sha256": manifest.semantic_content_sha256,
        "recording_identity": manifest.recording_identity,
        "mcap_id": request.window.mcap_id,
        "window_id": request.window.window_id,
        "purpose": request.purpose.value,
        "interval": request.window.interval.model_dump(mode="json"),
        "sampling": {
            "rate_num": request.rate_num,
            "rate_den": request.rate_den,
            "selection_tolerance_ns": request.selection_tolerance_ns,
        },
        "cameras": cameras,
    }


def _sampling_strategy(purpose: SamplingPurpose) -> SamplingStrategy:
    if purpose in {
        SamplingPurpose.QA_DENSE,
        SamplingPurpose.ACTION_DENSE,
        SamplingPurpose.BOUNDARY_REFINEMENT,
    }:
        return SamplingStrategy.DENSE
    return SamplingStrategy.UNIFORM


def _build_package(
    *,
    package_id: str,
    content_sha256: str,
    request: FrameMaterializationRequest,
    plans: tuple[_CameraPlan, ...],
    rendered: dict[CameraId, tuple[_RenderedFrame, ...]],
    created_at: str,
) -> TemporalVisualPackage:
    plan_by_camera = {plan.ledger.record.camera_id: plan for plan in plans}
    camera_packages: dict[CameraId, CameraPackage] = {}
    frame_count_total = 0
    strategy = _sampling_strategy(request.purpose)
    target_fps = float(Fraction(request.rate_num, request.rate_den))
    duration_ns = request.window.interval.duration_ns
    for camera_id in CAMERA_IDS:
        plan = plan_by_camera[camera_id]
        rendered_frames = rendered[camera_id]
        frames = tuple(
            MaterializedFrame(
                frame_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"robata-frame/v1/{package_id}/{camera_id.value}/"
                        f"{item.source.packet_index}/{item.sha256}",
                    )
                ),
                ordinal=item.source.ordinal,
                source_frame_index=item.source.packet_index,
                target_timestamp_ns=item.source.target_timestamp_ns,
                aligned_timestamp_ns=item.source.aligned_timestamp_ns,
                source_timestamp_ns=item.source.source_timestamp_ns,
                delta_to_target_ns=item.source.delta_to_target_ns,
                artifact_uri=(
                    f"robata-run://frames/{package_id}/{camera_id.value}/{item.filename}"
                ),
                artifact_sha256=item.sha256,
                width=item.width,
                height=item.height,
                quality_flags=_QUALITY_FLAGS,
            )
            for item in rendered_frames
        )
        actual_count = len(frames)
        frame_count_total += actual_count
        sampling = SamplingSummary(
            strategy=strategy,
            target_fps=target_fps,
            actual_fps=(
                0.0
                if actual_count == 0
                else float(Fraction(actual_count * _NANOSECONDS_PER_SECOND, duration_ns))
            ),
            target_count=plan.target_count,
            actual_count=actual_count,
            missed_targets=plan.target_count - actual_count,
        )
        camera_packages[camera_id] = CameraPackage(
            camera_id=camera_id,
            status=(CameraPackageStatus.AVAILABLE if frames else CameraPackageStatus.NO_FRAME),
            source_video_uri=f"robata-video-view://{camera_id.value}.mp4",
            frames=frames,
            sampling=sampling,
            missing_reason=None if frames else "NO_FRAME_WITHIN_TOLERANCE",
        )
    if frame_count_total == 0:
        _fail(
            FrameMaterializationErrorCode.INVALID_REQUEST,
            "sampling selected no frames across the six-camera package",
        )
    return TemporalVisualPackage(
        schema_version="1.0",
        package_id=package_id,
        content_sha256=content_sha256,
        mcap_id=request.window.mcap_id,
        window_id=request.window.window_id,
        purpose=request.purpose,
        interval=request.window.interval,
        cameras=SixCameraMap[CameraPackage](camera_packages),
        frame_count_total=frame_count_total,
        producer_version=_PRODUCER_VERSION,
        created_at=created_at,
    )


class PyAvFrameMaterializer:
    """Create immutable PNG evidence from one verified registered-video publication."""

    def __init__(
        self,
        *,
        max_width: int | None = 320,
        clock: Callable[[], datetime] = _utc_now,
        max_parallel_cameras: int = 1,
    ) -> None:
        if max_width is not None:
            if isinstance(max_width, bool) or not isinstance(max_width, int):
                raise TypeError("max_width must be an integer or None")
            if max_width <= 0:
                raise ValueError("max_width must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if isinstance(max_parallel_cameras, bool) or not isinstance(max_parallel_cameras, int):
            raise TypeError("max_parallel_cameras must be an integer")
        if max_parallel_cameras <= 0:
            raise ValueError("max_parallel_cameras must be positive")
        if max_parallel_cameras > len(CAMERA_IDS):
            raise ValueError("max_parallel_cameras cannot exceed the six camera slots")
        self._max_width = max_width
        self._clock = clock
        self._max_parallel_cameras = max_parallel_cameras

    def _render_all_cameras(
        self,
        plans: tuple[_CameraPlan, ...],
        staging: Path,
    ) -> dict[CameraId, tuple[_RenderedFrame, ...]]:
        if self._max_parallel_cameras == 1:
            return {
                plan.ledger.record.camera_id: _decode_and_render(
                    plan,
                    staging,
                    max_width=self._max_width,
                )
                for plan in plans
            }

        with ThreadPoolExecutor(
            max_workers=self._max_parallel_cameras,
            thread_name_prefix="robata-frame",
        ) as executor:
            futures = {
                plan.ledger.record.camera_id: executor.submit(
                    _decode_and_render,
                    plan,
                    staging,
                    max_width=self._max_width,
                )
                for plan in plans
            }
            # Resolve in canonical camera order so content projections never depend on
            # completion order. Each camera owns a distinct staging subdirectory.
            return {
                camera_id: futures[camera_id].result()
                for camera_id in CAMERA_IDS
                if camera_id in futures
            }

    def materialize(self, request: FrameMaterializationRequest) -> TemporalVisualPackage:
        """Verify, sample, decode, and atomically materialize one visual package."""

        if not isinstance(request, FrameMaterializationRequest):
            _fail(
                FrameMaterializationErrorCode.INVALID_REQUEST,
                "request must be a FrameMaterializationRequest",
            )
        view, manifest = _validate_publication(request.video_export)
        ledgers = tuple(_load_camera_ledger(view, manifest, record) for record in manifest.cameras)
        recording_origin_ns = min(ledger.rows[0].source_log_time_ns for ledger in ledgers)
        grid = SamplingGrid(
            grid_origin_ns=0,
            rate=SamplingRate(request.rate_num, request.rate_den),
        )
        plans = tuple(
            _build_camera_plan(
                ledger,
                recording_origin_ns=recording_origin_ns,
                grid=grid,
                request=request,
            )
            for ledger in ledgers
        )

        try:
            output_directory = Path(os.path.abspath(request.output_directory))
            frames_root = output_directory / "frames"
            frames_root.mkdir(parents=True, exist_ok=True)
            staging = make_staging_directory(frames_root, prefix=".materialization.partial-")
        except OSError as error:
            _fail(
                FrameMaterializationErrorCode.OUTPUT_IO_ERROR,
                f"cannot create frame-materialization staging directory: {error}",
            )

        published = False
        try:
            rendered = self._render_all_cameras(plans, staging)
            content_sha256 = semantic_sha256(
                _content_projection(manifest, request, plans, rendered)
            )
            package_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"robata-temporal-visual-package/v1/{content_sha256}",
                )
            )
            package = _build_package(
                package_id=package_id,
                content_sha256=content_sha256,
                request=request,
                plans=plans,
                rendered=rendered,
                created_at=_rfc3339(self._clock()),
            )
            target = frames_root / package_id
            if target.exists() or target.is_symlink():
                _fail(
                    FrameMaterializationErrorCode.OUTPUT_EXISTS,
                    f"frame package already exists: {package_id}",
                )
            try:
                staging.rename(target)
            except OSError as error:
                code = (
                    FrameMaterializationErrorCode.OUTPUT_EXISTS
                    if target.exists() or target.is_symlink()
                    else FrameMaterializationErrorCode.OUTPUT_IO_ERROR
                )
                _fail(code, f"cannot publish frame package {package_id}: {error}")
            published = True
            return package
        finally:
            if not published:
                with suppress(OSError):
                    shutil.rmtree(staging)


__all__ = ["PyAvFrameMaterializer"]
