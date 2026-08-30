"""Bounded visual review surfaces for a production-shaped MCAP cohort.

This module is intentionally narrower than a model runner.  It reads the
source-bound cohort manifest, decodes only the six mapped H.264 camera topics,
and writes small JPEG thumbnails/contact sheets for each window/camera pair.
The resulting JSON is a local review aid: it contains source timestamps and
paths, but no model prediction, label, hash, or publication claim.

Machine-assisted annotation is represented as a *separate* contract slot.  A
future caller may attach a draft from a vision model, but that result must be
labelled ``MACHINE_ASSISTED_DRAFT`` with review state ``PROVISIONAL`` and can
never be accepted as gold by this module.  Surface generation itself does not
invoke a model and leaves every draft slot ``NOT_GENERATED``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .production_cohort import DEFAULT_CAMERA_TOPICS

PRODUCTION_REVIEW_SURFACES_VERSION: Final = "robata-production-review-surfaces-v1"
MACHINE_ASSISTED_REVIEW_CONTRACT_VERSION: Final = "robata-production-machine-assisted-review-v1"
LOCAL_NONPRODUCTION_AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
PRODUCTION_COHORT_MANIFEST_FORMAT: Final = "robata-production-shaped-cohort-v1"
CAMERA_IDS: Final = tuple(f"cam_{index:02d}" for index in range(1, 7))
REQUIRED_LABEL_FIELDS: Final = ("verb", "noun", "attributes", "location", "hand")


class ProductionReviewSurfacesError(ValueError):
    """Raised when a bounded review-surface request cannot be fulfilled."""


@dataclass(frozen=True, slots=True)
class _WindowSpec:
    ordinal: int
    window_id: str
    start_seconds: float
    end_seconds: float
    camera_topics: Mapping[str, str]
    start_timestamp_ns: int
    end_timestamp_ns: int
    target_timestamps_ns: tuple[int, ...]


@dataclass(slots=True)
class _TargetSlot:
    target_index: int
    target_timestamp_ns: int
    delta_ns: int | None = None
    source_timestamp_ns: int | None = None
    image_bytes: bytes | None = None
    decoded_width: int | None = None
    decoded_height: int | None = None
    thumbnail_width: int | None = None
    thumbnail_height: int | None = None


@dataclass(slots=True)
class _CameraState:
    camera_id: str
    topic: str
    messages_examined: int = 0
    decoded_frames: int = 0
    decode_failures: int = 0
    failures: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionReviewSurfacesError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionReviewSurfacesError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionReviewSurfacesError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionReviewSurfacesError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ProductionReviewSurfacesError(f"{field} must be a finite non-negative number")
    return number


def _integer_timestamp(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ProductionReviewSurfacesError(f"{field} must be an integer timestamp")
    if not isinstance(value, (str, int)):
        raise ProductionReviewSurfacesError(f"{field} must be an integer timestamp")
    try:
        timestamp = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionReviewSurfacesError(f"{field} must be an integer timestamp") from exc
    if timestamp < 0:
        raise ProductionReviewSurfacesError(f"{field} must be an integer timestamp")
    return timestamp


def _normalise_topics(value: object | None, *, field: str) -> dict[str, str]:
    """Return exactly the canonical six-camera mapping in stable order."""

    raw: object = DEFAULT_CAMERA_TOPICS if value is None else value
    if isinstance(raw, Mapping):
        if "topics" in raw:
            raw = raw["topics"]
        elif "camera_topics" in raw:
            raw = raw["camera_topics"]
        elif "cameras" in raw and isinstance(raw["cameras"], Sequence):
            raw = raw["cameras"]

    pairs: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for camera_id, topic in raw.items():
            camera = _text(camera_id, field=f"{field}.camera_id")
            pairs[camera] = _text(topic, field=f"{field}.{camera}")
    else:
        rows = _sequence(raw, field=field)
        for index, row_value in enumerate(rows):
            row = _mapping(row_value, field=f"{field}[{index}]")
            camera = _text(
                row.get("camera_id", row.get("id")),
                field=f"{field}[{index}].camera_id",
            )
            topic = _text(
                row.get("topic", row.get("camera_topic")),
                field=f"{field}[{index}].topic",
            )
            if camera in pairs:
                raise ProductionReviewSurfacesError(f"duplicate camera ID: {camera}")
            pairs[camera] = topic

    if tuple(pairs) != CAMERA_IDS:
        missing = sorted(set(CAMERA_IDS) - set(pairs))
        extra = sorted(set(pairs) - set(CAMERA_IDS))
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise ProductionReviewSurfacesError(
            "camera mapping must contain cam_01 through cam_06 in order"
            + (f" ({'; '.join(detail)})" if detail else "")
        )
    if len(set(pairs.values())) != len(pairs):
        raise ProductionReviewSurfacesError("camera topics must be unique")
    return {camera_id: pairs[camera_id] for camera_id in CAMERA_IDS}


def _source_camera_topics(source: Mapping[str, Any]) -> dict[str, str] | None:
    raw_cameras = source.get("cameras")
    if raw_cameras is None:
        return None
    rows = _sequence(raw_cameras, field="manifest.source.cameras")
    pairs: dict[str, str] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"manifest.source.cameras[{index}]")
        camera = _text(
            row.get("camera_id", row.get("id")),
            field=f"manifest.source.cameras[{index}].camera_id",
        )
        topic = _text(
            row.get("topic", row.get("camera_topic")),
            field=f"manifest.source.cameras[{index}].topic",
        )
        if camera in pairs:
            raise ProductionReviewSurfacesError(f"duplicate source camera ID: {camera}")
        pairs[camera] = topic
    return _normalise_topics(pairs, field="manifest.source.cameras")


def _manifest_specs(
    manifest: Mapping[str, Any],
    *,
    frames_per_camera: int,
) -> tuple[Path, int, tuple[_WindowSpec, ...], dict[str, str]]:
    if manifest.get("format") != PRODUCTION_COHORT_MANIFEST_FORMAT:
        raise ProductionReviewSurfacesError(
            "manifest.format must be robata-production-shaped-cohort-v1"
        )
    if manifest.get("authority") != LOCAL_NONPRODUCTION_AUTHORITY:
        raise ProductionReviewSurfacesError("manifest authority must be LOCAL_NONPRODUCTION_ONLY")
    source = _mapping(manifest.get("source"), field="manifest.source")
    source_path = Path(_text(source.get("path"), field="manifest.source.path"))
    source_path = source_path.expanduser().resolve()

    source_topics = _source_camera_topics(source)
    raw_windows = _sequence(manifest.get("windows"), field="manifest.windows")
    if not raw_windows:
        raise ProductionReviewSurfacesError("manifest.windows must not be empty")

    raw_origin = source.get("common_start_timestamp_ns")
    if raw_origin is None and source_topics is not None:
        # A few development manifests omit the intersection field.  Use the
        # latest first timestamp when it is available, which is the same
        # intersection origin used by the cohort builder.
        camera_rows = _sequence(source.get("cameras"), field="manifest.source.cameras")
        starts: list[int] = []
        for index, raw in enumerate(camera_rows):
            row = _mapping(raw, field=f"manifest.source.cameras[{index}]")
            if row.get("first_timestamp_ns") is not None:
                starts.append(
                    _integer_timestamp(
                        row.get("first_timestamp_ns"),
                        field=f"manifest.source.cameras[{index}].first_timestamp_ns",
                    )
                )
        origin_ns = max(starts) if starts else 0
    else:
        origin_ns = _integer_timestamp(
            raw_origin if raw_origin is not None else 0,
            field="manifest.source.common_start_timestamp_ns",
        )

    specs: list[_WindowSpec] = []
    expected_topics: dict[str, str] | None = source_topics
    previous_end = -math.inf
    seen_ids: set[str] = set()
    for index, raw_window in enumerate(raw_windows):
        window = _mapping(raw_window, field=f"manifest.windows[{index}]")
        ordinal_raw = window.get("ordinal", index)
        if isinstance(ordinal_raw, bool) or not isinstance(ordinal_raw, int) or ordinal_raw < 0:
            raise ProductionReviewSurfacesError(
                f"manifest.windows[{index}].ordinal must be a non-negative integer"
            )
        window_id = _text(
            window.get("window_id"),
            field=f"manifest.windows[{index}].window_id",
        )
        if window_id in seen_ids:
            raise ProductionReviewSurfacesError(f"duplicate window_id: {window_id}")
        seen_ids.add(window_id)
        start = _finite_nonnegative(
            window.get("start_seconds"),
            field=f"manifest.windows[{index}].start_seconds",
        )
        end = _finite_nonnegative(
            window.get("end_seconds"),
            field=f"manifest.windows[{index}].end_seconds",
        )
        if end <= start:
            raise ProductionReviewSurfacesError(f"manifest.windows[{index}] end must exceed start")
        if start < previous_end:
            raise ProductionReviewSurfacesError(
                "manifest windows must be ordered and non-overlapping"
            )
        previous_end = end
        camera_ids = _sequence(
            window.get("camera_ids"),
            field=f"manifest.windows[{index}].camera_ids",
        )
        if (
            tuple(
                _text(value, field=f"manifest.windows[{index}].camera_ids[{camera_index}]")
                for camera_index, value in enumerate(camera_ids)
            )
            != CAMERA_IDS
        ):
            raise ProductionReviewSurfacesError(
                f"manifest.windows[{index}].camera_ids must be cam_01 through cam_06"
            )
        topics = _normalise_topics(
            window.get("camera_topics", expected_topics),
            field=f"manifest.windows[{index}].camera_topics",
        )
        if expected_topics is None:
            expected_topics = topics
        elif topics != expected_topics:
            raise ProductionReviewSurfacesError(
                f"manifest.windows[{index}].camera_topics does not bind source mapping"
            )
        start_ns = origin_ns + round(start * 1_000_000_000)
        end_ns = origin_ns + round(end * 1_000_000_000)
        denominator = max(1, frames_per_camera - 1)
        targets = tuple(
            start_ns + ((end_ns - start_ns) * target_index) // denominator
            for target_index in range(frames_per_camera)
        )
        specs.append(
            _WindowSpec(
                ordinal=ordinal_raw,
                window_id=window_id,
                start_seconds=start,
                end_seconds=end,
                camera_topics=topics,
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                target_timestamps_ns=targets,
            )
        )

    if expected_topics is None:
        expected_topics = _normalise_topics(None, field="camera_topics")
    return source_path, origin_ns, tuple(specs), expected_topics


def _frame_timestamp_ns(frame: Any, fallback_ns: int | None) -> int | None:
    if fallback_ns is None:
        return None
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    if pts is None or time_base is None:
        return fallback_ns
    try:
        return int(Fraction(pts) * Fraction(time_base) * Fraction(1_000_000_000))
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return fallback_ns


def _schema_name(schema: Any) -> str | None:
    if isinstance(schema, str):
        return schema
    name = getattr(schema, "name", None)
    return name if isinstance(name, str) else None


def _failure(
    state: _CameraState,
    *,
    code: str,
    timestamp_ns: int | None,
    message: str,
) -> None:
    state.decode_failures += 1
    # Keep the report bounded even when a source contains a long run of bad
    # packets.  The count remains exact; only verbose examples are capped.
    if state.failures is not None and len(state.failures) < 32:
        state.failures.append({"code": code, "timestamp_ns": timestamp_ns, "message": message})


def _thumbnail_bytes(
    frame: Any,
    *,
    max_side: int,
) -> tuple[bytes, int, int, int, int]:
    try:
        from PIL import Image  # type: ignore[import-not-found,unused-ignore]
    except Exception as exc:  # pragma: no cover - optional dependency boundary
        raise ProductionReviewSurfacesError("review-surface rendering requires Pillow") from exc
    try:
        image = frame.to_image().convert("RGB")
    except Exception as exc:
        raise ProductionReviewSurfacesError(
            f"decoded frame could not be converted to an image: {type(exc).__name__}: {exc}"
        ) from exc
    decoded_width, decoded_height = int(image.width), int(image.height)
    if max(image.size) > max_side:
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
        image.thumbnail((max_side, max_side), resampling)
    output = BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=False)
    return output.getvalue(), decoded_width, decoded_height, int(image.width), int(image.height)


def _select_frames(
    source_path: Path,
    windows: tuple[_WindowSpec, ...],
    camera_topics: Mapping[str, str],
    *,
    frames_per_camera: int,
    thumbnail_max_side: int,
    max_messages_per_camera: int,
    validate_crcs: bool,
) -> tuple[dict[tuple[str, str], list[_TargetSlot]], dict[str, _CameraState]]:
    try:
        import av
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ProductionReviewSurfacesError(
            "review-surface decoding requires PyAV, mcap, and mcap-protobuf"
        ) from exc

    slots: dict[tuple[str, str], list[_TargetSlot]] = {
        (window.window_id, camera_id): [
            _TargetSlot(target_index=index, target_timestamp_ns=target_timestamp)
            for index, target_timestamp in enumerate(window.target_timestamps_ns)
        ]
        for window in windows
        for camera_id in CAMERA_IDS
    }
    states = {
        camera_id: _CameraState(camera_id=camera_id, topic=camera_topics[camera_id])
        for camera_id in CAMERA_IDS
    }
    topic_to_camera = {topic: camera_id for camera_id, topic in camera_topics.items()}
    try:
        decoders = {camera_id: av.CodecContext.create("h264", "r") for camera_id in CAMERA_IDS}
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ProductionReviewSurfacesError(
            f"could not initialize H.264 decoder: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        with source_path.open("rb") as stream:
            reader = make_reader(
                stream,
                validate_crcs=validate_crcs,
                decoder_factories=[DecoderFactory()],
            )
            for schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=tuple(topic_to_camera),
                log_time_order=False,
            ):
                topic = getattr(channel, "topic", None)
                camera_id = topic_to_camera.get(topic) if isinstance(topic, str) else None
                if camera_id is None:
                    continue
                state = states[camera_id]
                if state.messages_examined >= max_messages_per_camera:
                    continue
                state.messages_examined += 1
                timestamp_raw = getattr(message, "log_time", None)
                try:
                    timestamp_ns = int(timestamp_raw) if timestamp_raw is not None else None
                except (TypeError, ValueError, OverflowError):
                    timestamp_ns = None
                if _schema_name(schema) != "foxglove.CompressedImage":
                    _failure(
                        state,
                        code="INVALID_COMPRESSED_IMAGE_SCHEMA",
                        timestamp_ns=timestamp_ns,
                        message="mapped topic did not decode as foxglove.CompressedImage",
                    )
                    continue
                payload = getattr(decoded, "data", None)
                image_format = str(getattr(decoded, "format", "")).strip().casefold()
                if image_format != "h264":
                    _failure(
                        state,
                        code="UNSUPPORTED_IMAGE_FORMAT",
                        timestamp_ns=timestamp_ns,
                        message=(
                            "mapped CompressedImage format is "
                            f"{image_format or '<empty>'!r}, expected 'h264'"
                        ),
                    )
                    continue
                if not isinstance(payload, bytes) or not payload:
                    _failure(
                        state,
                        code="INVALID_COMPRESSED_IMAGE_PAYLOAD",
                        timestamp_ns=timestamp_ns,
                        message="CompressedImage.data must be non-empty bytes",
                    )
                    continue
                # ``CodecContext.decode`` returns a list of decoded frames, but
                # malformed packets (or a decoder exception) produce no frames.
                # Keep the local variable typed as a read-only sequence so the
                # empty fallback does not narrow it to ``list[VideoFrame]`` and
                # trigger a strict-mypy tuple-assignment error.
                frames: Sequence[Any] = ()
                try:
                    packet = av.Packet(payload)
                    if timestamp_ns is not None:
                        packet.pts = timestamp_ns
                        packet.dts = timestamp_ns
                    packet.time_base = Fraction(1, 1_000_000_000)
                    frames = decoders[camera_id].decode(packet)
                except Exception as exc:
                    _failure(
                        state,
                        code="H264_DECODE_ERROR",
                        timestamp_ns=timestamp_ns,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                    frames = ()
                if frames is None:
                    frames = ()
                state.decoded_frames += len(frames)
                for frame in frames:
                    frame_timestamp_ns = _frame_timestamp_ns(frame, timestamp_ns)
                    if frame_timestamp_ns is None:
                        continue
                    for window in windows:
                        if not (
                            window.start_timestamp_ns
                            <= frame_timestamp_ns
                            < window.end_timestamp_ns
                        ):
                            continue
                        unfilled_targets = [
                            candidate
                            for candidate in range(frames_per_camera)
                            if slots[(window.window_id, camera_id)][candidate].image_bytes is None
                        ]
                        target_candidates = unfilled_targets or list(range(frames_per_camera))
                        target_index = min(
                            target_candidates,
                            key=lambda candidate: (
                                abs(window.target_timestamps_ns[candidate] - frame_timestamp_ns),
                                # Prefer the later target on an exact midpoint
                                # tie; this avoids starving the final slot when
                                # a tiny fixture has only two frames.
                                -candidate,
                            ),
                        )
                        slot = slots[(window.window_id, camera_id)][target_index]
                        delta_ns = abs(slot.target_timestamp_ns - frame_timestamp_ns)
                        previous = slot.delta_ns
                        if previous is not None and (
                            delta_ns > previous
                            or (
                                delta_ns == previous
                                and slot.source_timestamp_ns is not None
                                and frame_timestamp_ns >= slot.source_timestamp_ns
                            )
                        ):
                            continue
                        (
                            image_bytes,
                            decoded_width,
                            decoded_height,
                            thumbnail_width,
                            thumbnail_height,
                        ) = _thumbnail_bytes(frame, max_side=thumbnail_max_side)
                        slot.delta_ns = delta_ns
                        slot.source_timestamp_ns = frame_timestamp_ns
                        slot.image_bytes = image_bytes
                        slot.decoded_width = decoded_width
                        slot.decoded_height = decoded_height
                        slot.thumbnail_width = thumbnail_width
                        slot.thumbnail_height = thumbnail_height
    except OSError as exc:
        raise ProductionReviewSurfacesError(
            f"could not read MCAP source {source_path}: {exc}"
        ) from exc
    except ProductionReviewSurfacesError:
        raise
    except Exception as exc:
        raise ProductionReviewSurfacesError(
            f"bounded MCAP review-surface decode failed: {type(exc).__name__}: {exc}"
        ) from exc
    return slots, states


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "item"


def _load_image(data: bytes) -> Any:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency boundary
        raise ProductionReviewSurfacesError("review-surface rendering requires Pillow") from exc
    try:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise ProductionReviewSurfacesError(
            f"review thumbnail is not decodable: {type(exc).__name__}: {exc}"
        ) from exc


def _load_slot_image(slot: _TargetSlot) -> Any:
    data = slot.image_bytes
    if data is None:
        raise ProductionReviewSurfacesError("cannot render a missing review thumbnail")
    return _load_image(data)


def _draw_font() -> Any:
    try:
        from PIL import ImageFont

        return ImageFont.load_default()
    except Exception:  # pragma: no cover - Pillow boundary
        return None


def _save_jpeg(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.save(path, format="JPEG", quality=88, optimize=False)
    except OSError as exc:
        raise ProductionReviewSurfacesError(f"could not write review surface {path}") from exc


def _render_camera_sheet(
    slots: Sequence[_TargetSlot],
    *,
    title: str,
    window_start_timestamp_ns: int,
) -> Any:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - optional dependency boundary
        raise ProductionReviewSurfacesError("review-surface rendering requires Pillow") from exc
    images = [_load_image(slot.image_bytes) for slot in slots if slot.image_bytes is not None]
    max_width = max((int(image.width) for image in images), default=320)
    max_height = max((int(image.height) for image in images), default=240)
    cell_width = max(220, max_width)
    cell_height = max(180, max_height)
    gap = 8
    header_height = 28
    canvas = Image.new(
        "RGB",
        (gap + len(slots) * (cell_width + gap), header_height + gap + cell_height + 30),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = _draw_font()
    draw.text((gap, 6), title, fill=(20, 20, 20), font=font)
    for index, slot in enumerate(slots):
        left = gap + index * (cell_width + gap)
        top = header_height
        if slot.image_bytes is None:
            draw.rectangle((left, top, left + cell_width, top + cell_height), fill=(210, 210, 210))
            draw.text((left + 8, top + 8), "MISSING FRAME", fill=(120, 0, 0), font=font)
        else:
            image = _load_image(slot.image_bytes)
            canvas.paste(image, (left, top))
        delta_ms = "n/a" if slot.delta_ns is None else f"{slot.delta_ns / 1_000_000:.1f}ms"
        source_seconds = (
            "n/a"
            if slot.source_timestamp_ns is None
            else f"{(slot.source_timestamp_ns - window_start_timestamp_ns) / 1_000_000_000:.3f}s"
        )
        draw.text(
            (left, top + cell_height + 5),
            f"target {slot.target_index} | source {source_seconds} | delta {delta_ms}",
            fill=(25, 25, 25),
            font=font,
        )
    return canvas


def _render_overview(
    window: _WindowSpec,
    per_camera_slots: Mapping[str, Sequence[_TargetSlot]],
) -> Any:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - optional dependency boundary
        raise ProductionReviewSurfacesError("review-surface rendering requires Pillow") from exc
    rows = [
        [slot for slot in per_camera_slots[camera_id] if slot.image_bytes is not None]
        for camera_id in CAMERA_IDS
    ]
    image_rows = [[_load_slot_image(slot) for slot in row] for row in rows]
    cell_width = max(
        220,
        max((int(image.width) for row in image_rows for image in row), default=320),
    )
    cell_height = max(
        180,
        max((int(image.height) for row in image_rows for image in row), default=240),
    )
    gap = 8
    label_width = 64
    header_height = 28
    canvas = Image.new(
        "RGB",
        (
            label_width + gap + len(window.target_timestamps_ns) * (cell_width + gap),
            header_height + gap + len(CAMERA_IDS) * (cell_height + 24 + gap),
        ),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = _draw_font()
    draw.text(
        (gap, 6),
        f"{window.window_id} | [{window.start_seconds:.3f}, {window.end_seconds:.3f})s",
        fill=(20, 20, 20),
        font=font,
    )
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        row_top = header_height + gap + camera_index * (cell_height + 24 + gap)
        draw.text((gap, row_top + 4), camera_id, fill=(20, 20, 20), font=font)
        for target_index, slot in enumerate(per_camera_slots[camera_id]):
            left = label_width + gap + target_index * (cell_width + gap)
            if slot.image_bytes is None:
                draw.rectangle(
                    (left, row_top, left + cell_width, row_top + cell_height),
                    fill=(210, 210, 210),
                )
                draw.text((left + 5, row_top + 5), "MISSING", fill=(120, 0, 0), font=font)
            else:
                canvas.paste(_load_image(slot.image_bytes), (left, row_top))
            source_seconds = (
                "n/a"
                if slot.source_timestamp_ns is None
                else (
                    f"{(slot.source_timestamp_ns - window.start_timestamp_ns) / 1_000_000_000:.3f}s"
                )
            )
            draw.text(
                (left, row_top + cell_height + 4),
                source_seconds,
                fill=(25, 25, 25),
                font=font,
            )
    return canvas


def _relative_path(path: PurePosixPath) -> str:
    return path.as_posix()


def _draft_slot(*, window: _WindowSpec, surface_paths: Sequence[str]) -> dict[str, Any]:
    """Create the explicit no-inference draft placeholder for one window."""

    return {
        "contract_version": MACHINE_ASSISTED_REVIEW_CONTRACT_VERSION,
        "status": "NOT_GENERATED",
        "review_state": "NOT_REQUESTED",
        "provisional_status_if_generated": "MACHINE_ASSISTED_DRAFT",
        "provisional_review_state_if_generated": "PROVISIONAL",
        "accepted_as_gold": False,
        "review_required": True,
        "segments": [],
        "source_surface_paths": list(surface_paths),
        "window_bound_only": True,
        "note": (
            "No model was invoked while generating these surfaces. Any future "
            "machine-assisted labels must remain MACHINE_ASSISTED_DRAFT/PROVISIONAL."
        ),
        "window_id": window.window_id,
        "start_seconds": window.start_seconds,
        "end_seconds": window.end_seconds,
    }


def _surface_entry(
    *,
    window: _WindowSpec,
    camera_id: str,
    slots: Sequence[_TargetSlot],
    state: _CameraState,
    contact_sheet_path: str,
    frame_paths: Sequence[str | None],
) -> dict[str, Any]:
    selected = [slot for slot in slots if slot.image_bytes is not None]
    if not selected:
        status = "FAILED"
    elif len(selected) == len(slots):
        status = "READY"
    else:
        status = "PARTIAL"
    frames: list[dict[str, Any]] = []
    for slot, frame_path in zip(slots, frame_paths, strict=True):
        frames.append(
            {
                "target_index": slot.target_index,
                "target_timestamp_ns": str(slot.target_timestamp_ns),
                "source_timestamp_ns": (
                    None if slot.source_timestamp_ns is None else str(slot.source_timestamp_ns)
                ),
                "target_relative_seconds": (
                    (slot.target_timestamp_ns - window.start_timestamp_ns) / 1_000_000_000
                ),
                "source_relative_seconds": (
                    None
                    if slot.source_timestamp_ns is None
                    else (slot.source_timestamp_ns - window.start_timestamp_ns) / 1_000_000_000
                ),
                "delta_seconds": (None if slot.delta_ns is None else slot.delta_ns / 1_000_000_000),
                "selected": slot.image_bytes is not None,
                "frame_path": frame_path,
                "decoded_dimensions": (
                    None
                    if slot.decoded_width is None or slot.decoded_height is None
                    else [slot.decoded_width, slot.decoded_height]
                ),
                "thumbnail_dimensions": (
                    None
                    if slot.thumbnail_width is None or slot.thumbnail_height is None
                    else [slot.thumbnail_width, slot.thumbnail_height]
                ),
            }
        )
    return {
        "camera_id": camera_id,
        "topic": state.topic,
        "status": status,
        "contact_sheet_path": contact_sheet_path,
        "frame_count": len(selected),
        "requested_frame_count": len(slots),
        "frames": frames,
        "decode": {
            "messages_examined": state.messages_examined,
            "decoded_frames": state.decoded_frames,
            "decode_failures": state.decode_failures,
            "failures": list(state.failures or []),
        },
    }


def _write_bundle(
    *,
    output_root: Path,
    source_path: Path,
    origin_ns: int,
    windows: tuple[_WindowSpec, ...],
    camera_topics: Mapping[str, str],
    slots: Mapping[tuple[str, str], Sequence[_TargetSlot]],
    states: Mapping[str, _CameraState],
    frames_per_camera: int,
    thumbnail_max_side: int,
    max_messages_per_camera: int,
    validate_crcs: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    surface_root = PurePosixPath("surfaces")
    report_windows: list[dict[str, Any]] = []
    total_surfaces = 0
    ready_surfaces = 0
    for window in windows:
        window_component = _safe_component(window.window_id)
        window_relative = surface_root / window_component
        per_camera_slots: dict[str, Sequence[_TargetSlot]] = {}
        camera_entries: list[dict[str, Any]] = []
        draft_paths: list[str] = []
        for camera_id in CAMERA_IDS:
            camera_slots = slots[(window.window_id, camera_id)]
            per_camera_slots[camera_id] = camera_slots
            camera_component = _safe_component(camera_id)
            frame_paths: list[str | None] = []
            for slot in camera_slots:
                frame_relative = (
                    window_relative / camera_component / f"frame-{slot.target_index:02d}.jpg"
                )
                frame_path: str | None = None
                if slot.image_bytes is not None:
                    destination = output_root / Path(*frame_relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(slot.image_bytes)
                    frame_path = _relative_path(frame_relative)
                    draft_paths.append(frame_path)
                frame_paths.append(frame_path)
            contact_relative = window_relative / f"{camera_component}-contact-sheet.jpg"
            contact_image = _render_camera_sheet(
                camera_slots,
                title=(
                    f"{window.window_id} | {camera_id} | "
                    f"[{window.start_seconds:.3f}, {window.end_seconds:.3f})s"
                ),
                window_start_timestamp_ns=window.start_timestamp_ns,
            )
            _save_jpeg(output_root / Path(*contact_relative.parts), contact_image)
            contact_path = _relative_path(contact_relative)
            draft_paths.append(contact_path)
            entry = _surface_entry(
                window=window,
                camera_id=camera_id,
                slots=camera_slots,
                state=states[camera_id],
                contact_sheet_path=contact_path,
                frame_paths=frame_paths,
            )
            camera_entries.append(entry)
            total_surfaces += 1
            if entry["status"] == "READY":
                ready_surfaces += 1

        overview_relative = window_relative / "six-camera-overview.jpg"
        overview_image = _render_overview(window, per_camera_slots)
        _save_jpeg(output_root / Path(*overview_relative.parts), overview_image)
        overview_path = _relative_path(overview_relative)
        report_windows.append(
            {
                "ordinal": window.ordinal,
                "window_id": window.window_id,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "duration_seconds": window.end_seconds - window.start_seconds,
                "camera_count": len(CAMERA_IDS),
                "camera_surfaces": camera_entries,
                "window_contact_sheet_path": overview_path,
                "review_binding": {
                    "review_pack_format": "robata-production-human-review-pack-v1",
                    "qa_status": "PENDING",
                    "gold_status": "PENDING_HUMAN_REVIEW",
                    "human_gold_written": False,
                    "model_outputs_are_not_gold": True,
                },
                "status": (
                    "READY"
                    if all(entry["status"] == "READY" for entry in camera_entries)
                    else "PARTIAL"
                ),
                "machine_assisted_draft": _draft_slot(
                    window=window,
                    surface_paths=[overview_path, *draft_paths],
                ),
            }
        )

    return {
        "format": PRODUCTION_REVIEW_SURFACES_VERSION,
        "authority": LOCAL_NONPRODUCTION_AUTHORITY,
        "production_eligible": False,
        "counts": {
            "windows": len(windows),
            "camera_surfaces": total_surfaces,
            "ready_camera_surfaces": ready_surfaces,
            "selected_frames": sum(
                sum(
                    1
                    for slot in slots[(window.window_id, camera_id)]
                    if slot.image_bytes is not None
                )
                for window in windows
                for camera_id in CAMERA_IDS
            ),
        },
        "source": {
            "path": str(source_path),
            "source_bound": True,
            "common_start_timestamp_ns": str(origin_ns),
            "camera_count": len(CAMERA_IDS),
            "camera_topics": dict(camera_topics),
        },
        "bundle_root": str(output_root),
        "surface_root": surface_root.as_posix(),
        "surface_policy": {
            "selection": "nearest_to_even_temporal_targets",
            "frames_per_camera_per_window": frames_per_camera,
            "thumbnail_max_side": thumbnail_max_side,
            "image_format": "JPEG",
            "max_messages_per_camera": max_messages_per_camera,
            "validate_crcs": validate_crcs,
            "window_count": len(windows),
            "camera_count": len(CAMERA_IDS),
        },
        "machine_assisted_draft_contract": {
            "format": MACHINE_ASSISTED_REVIEW_CONTRACT_VERSION,
            "generated_status": "MACHINE_ASSISTED_DRAFT",
            "generated_review_state": "PROVISIONAL",
            "not_generated_status": "NOT_GENERATED",
            "accepted_as_gold": False,
            "gold_status": "PENDING_HUMAN_REVIEW",
            "independent_human_review_required": True,
            "required_label_fields": list(REQUIRED_LABEL_FIELDS),
            "window_boundaries_are_not_action_boundaries": True,
            "model_outputs_are_not_gold": True,
        },
        "windows": report_windows,
        "controls": {
            "model_invoked": False,
            "gpu_invoked": False,
            "labels_inferred": False,
            "machine_assisted_draft_generated": False,
            "human_gold_written": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "sha_or_digest_computed": False,
            "frames_decoded": any(state.decoded_frames > 0 for state in states.values()),
        },
    }


def build_production_review_surfaces(
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    frames_per_camera: int = 4,
    thumbnail_max_side: int = 320,
    max_messages_per_camera: int = 5000,
    validate_crcs: bool = True,
) -> dict[str, Any]:
    """Decode bounded camera windows and materialize visual review surfaces.

    No model is loaded and no digest is computed.  ``output_dir`` becomes the
    bundle root; all paths inside the returned report are POSIX-relative to
    that directory.
    """

    if isinstance(frames_per_camera, bool) or not isinstance(frames_per_camera, int):
        raise ProductionReviewSurfacesError("frames_per_camera must be a positive integer")
    if frames_per_camera < 1 or frames_per_camera > 32:
        raise ProductionReviewSurfacesError("frames_per_camera must be between 1 and 32")
    if isinstance(thumbnail_max_side, bool) or not isinstance(thumbnail_max_side, int):
        raise ProductionReviewSurfacesError("thumbnail_max_side must be a positive integer")
    if thumbnail_max_side < 32 or thumbnail_max_side > 2048:
        raise ProductionReviewSurfacesError("thumbnail_max_side must be between 32 and 2048")
    if isinstance(max_messages_per_camera, bool) or not isinstance(max_messages_per_camera, int):
        raise ProductionReviewSurfacesError("max_messages_per_camera must be a positive integer")
    if max_messages_per_camera < 1:
        raise ProductionReviewSurfacesError("max_messages_per_camera must be a positive integer")
    if not isinstance(validate_crcs, bool):
        raise ProductionReviewSurfacesError("validate_crcs must be boolean")

    manifest_mapping = _mapping(manifest, field="manifest")
    source_path, origin_ns, windows, camera_topics = _manifest_specs(
        manifest_mapping,
        frames_per_camera=frames_per_camera,
    )
    if not source_path.is_file():
        raise ProductionReviewSurfacesError(f"MCAP source is not a file: {source_path}")
    output_root = Path(output_dir).expanduser().resolve()
    slots, states = _select_frames(
        source_path,
        windows,
        camera_topics,
        frames_per_camera=frames_per_camera,
        thumbnail_max_side=thumbnail_max_side,
        max_messages_per_camera=max_messages_per_camera,
        validate_crcs=validate_crcs,
    )
    return _write_bundle(
        output_root=output_root,
        source_path=source_path,
        origin_ns=origin_ns,
        windows=windows,
        camera_topics=camera_topics,
        slots=slots,
        states=states,
        frames_per_camera=frames_per_camera,
        thumbnail_max_side=thumbnail_max_side,
        max_messages_per_camera=max_messages_per_camera,
        validate_crcs=validate_crcs,
    )


def write_production_review_surfaces(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """Write one JSON report for a previously materialized surface bundle."""

    import json

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(
            json.dumps(dict(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ProductionReviewSurfacesError(
            f"could not write review-surface report: {destination}"
        ) from exc


# Compatibility aliases make the seam easy to discover alongside the existing
# production cohort/review-pack builders.
build_review_surfaces = build_production_review_surfaces
write_review_surfaces = write_production_review_surfaces


__all__ = [
    "CAMERA_IDS",
    "LOCAL_NONPRODUCTION_AUTHORITY",
    "MACHINE_ASSISTED_REVIEW_CONTRACT_VERSION",
    "PRODUCTION_COHORT_MANIFEST_FORMAT",
    "PRODUCTION_REVIEW_SURFACES_VERSION",
    "ProductionReviewSurfacesError",
    "build_production_review_surfaces",
    "build_review_surfaces",
    "write_production_review_surfaces",
    "write_review_surfaces",
]
