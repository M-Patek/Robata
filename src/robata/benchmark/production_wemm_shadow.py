"""Label-blind WeMM shadow retrieval over a bounded six-camera MCAP cohort.

This is a benchmark-local execution seam, not the production retrieval service.
It decodes the selected camera topics once, feeds each bounded frame sequence to
WeMM's native video embedding path, retrieves candidates from an explicitly
supplied ontology, and fuses camera rankings deterministically.  The output is
always marked non-production and ``NOT_MEASURED`` for quality because the
production sample has no accepted action gold yet.

The module intentionally does not compute hashes/digests, invoke Qwen/Mage,
call the Mapper, or mutate the cohort review pack.  Its output can therefore be
passed to the machine-assisted review draft builder without contaminating gold.
"""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from importlib import import_module
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from .wemm_action_retrieval import (
    LabelVariant,
    RetrievedAction,
    build_joint_action_catalog,
    rank_joint_actions,
)
from .wemm_embedding_backend import WemmEmbeddingBackend
from .wemm_multiview_retrieval import fuse_camera_rankings

PRODUCTION_WEMM_SHADOW_VERSION: Final = "robata-production-wemm-shadow-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
CAMERA_IDS: Final = tuple(f"cam_{index:02d}" for index in range(1, 7))
_COMPRESSED_SCHEMA: Final = "foxglove.CompressedImage"


def _ontology_profile(provenance: Mapping[str, Any]) -> str:
    """Classify the supplied catalog without changing its labels.

    Historically this route always emitted an EPIC profile, even when a
    production-provisional pair catalog was supplied.  That made a valid
    production-vocabulary shadow look like an EPIC run and was the source of
    ``take cloth``/``put cloth`` confusion.  The profile is observational only:
    it is derived from declared catalog provenance and does not infer or
    promote semantic mappings.
    """

    format_value = str(provenance.get("format") or "").casefold()
    source_value = str(provenance.get("source") or "").casefold()
    if "production" in format_value or "production" in source_value:
        return "PRODUCTION_VOCABULARY_FOR_SHADOW_ONLY"
    return "PROVISIONAL_EPIC_ONTOLOGY_FOR_SHADOW_ONLY"


class ProductionWemmShadowError(RuntimeError):
    """Raised when the bounded WeMM shadow route cannot complete."""


@dataclass(frozen=True, slots=True)
class ProductionFrameGroup:
    camera_id: str
    window_id: str
    frames: tuple[Any, ...]
    selected_timestamps_ns: tuple[int, ...]
    messages_examined: int
    decoded_frames: int
    decode_failures: tuple[str, ...]
    width: int
    height: int
    fps: float
    start_seconds: float
    end_seconds: float

    def metadata(self) -> dict[str, Any]:
        return {
            "total_num_frames": len(self.frames),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration": self.end_seconds - self.start_seconds,
            "frames_indices": list(range(len(self.frames))),
            "video_backend": "mcap-h264-bounded",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "window_id": self.window_id,
            "frame_count": len(self.frames),
            "selected_timestamps_ns": list(self.selected_timestamps_ns),
            "messages_examined": self.messages_examined,
            "decoded_frames": self.decoded_frames,
            "decode_failures": list(self.decode_failures),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmShadowError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmShadowError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionWemmShadowError(f"{field} must be a non-empty string")
    return value.strip()


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionWemmShadowError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ProductionWemmShadowError(f"{field} must be finite")
    return number


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionWemmShadowError(f"{field} must be a positive integer")
    return value


def _load_manifest(
    manifest: Mapping[str, Any],
) -> tuple[Path, int, tuple[dict[str, Any], ...], tuple[tuple[str, str], ...]]:
    source = _mapping(manifest.get("source"), field="manifest.source")
    source_path = (
        Path(_text(source.get("path"), field="manifest.source.path")).expanduser().resolve()
    )
    if not source_path.is_file():
        raise ProductionWemmShadowError(f"MCAP source is not a file: {source_path}")
    try:
        common_start = int(
            _text(source.get("common_start_timestamp_ns"), field="source.common_start_timestamp_ns")
        )
    except ValueError as exc:
        raise ProductionWemmShadowError(
            "source.common_start_timestamp_ns must be an integer"
        ) from exc
    raw_cameras = _sequence(source.get("cameras"), field="manifest.source.cameras")
    camera_pairs: list[tuple[str, str]] = []
    for index, raw in enumerate(raw_cameras):
        camera = _mapping(raw, field=f"manifest.source.cameras[{index}]")
        camera_id = _text(camera.get("camera_id"), field=f"cameras[{index}].camera_id")
        topic = _text(camera.get("topic"), field=f"cameras[{index}].topic")
        camera_pairs.append((camera_id, topic))
    if tuple(camera for camera, _ in camera_pairs) != CAMERA_IDS:
        raise ProductionWemmShadowError("manifest must contain cam_01 through cam_06 in order")
    if len({topic for _, topic in camera_pairs}) != len(camera_pairs):
        raise ProductionWemmShadowError("camera topics must be unique")

    raw_windows = _sequence(manifest.get("windows"), field="manifest.windows")
    windows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_windows):
        window = _mapping(raw, field=f"manifest.windows[{index}]")
        window_id = _text(window.get("window_id"), field=f"windows[{index}].window_id")
        if window_id in seen:
            raise ProductionWemmShadowError(f"duplicate window_id: {window_id}")
        seen.add(window_id)
        start = _finite(window.get("start_seconds"), field=f"{window_id}.start_seconds")
        end = _finite(window.get("end_seconds"), field=f"{window_id}.end_seconds")
        if start < 0 or end <= start:
            raise ProductionWemmShadowError(f"invalid interval for {window_id}")
        windows.append(
            {
                "ordinal": int(window.get("ordinal", index)),
                "window_id": window_id,
                "start_seconds": start,
                "end_seconds": end,
            }
        )
    if not windows:
        raise ProductionWemmShadowError("manifest.windows must not be empty")
    return source_path, common_start, tuple(windows), tuple(camera_pairs)


def _decode_frame_timestamp(frame: Any, fallback_ns: int) -> int:
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    if pts is None or time_base is None:
        return fallback_ns
    try:
        return int(Fraction(pts) * time_base * 1_000_000_000)
    except (TypeError, ValueError, OverflowError):
        return fallback_ns


def _to_rgb_image(frame: Any) -> Any:
    """Convert a decoded frame to RGB while reusing an RGB image.

    ``VideoFrame.to_image()`` already returns RGB for the H.264 path used by
    this benchmark.  Avoiding a second full-frame ``convert`` copy reduces
    ingest CPU/memory work; non-RGB codecs retain the conversion fallback.
    """

    image = frame.to_image()
    if getattr(image, "mode", None) == "RGB":
        return image
    return image.convert("RGB")


def decode_production_windows(
    manifest: Mapping[str, Any],
    *,
    frame_count: int = 4,
    validate_crcs: bool = False,
) -> dict[str, dict[str, ProductionFrameGroup]]:
    """Decode a small, bounded frame set for every camera/window.

    Decoding is source-bound and keeps only the nearest decoded frame for each
    requested sample time.  The H.264 decoder still receives packets from the
    beginning of the recording so inter-frame dependencies are preserved.
    """

    frame_count = _positive_int(frame_count, field="frame_count")
    if frame_count < 2 or frame_count > 64:
        raise ProductionWemmShadowError("frame_count must be between 2 and 64")
    if not isinstance(validate_crcs, bool):
        raise ProductionWemmShadowError("validate_crcs must be boolean")
    source, common_start, windows, camera_pairs = _load_manifest(manifest)
    try:
        av = import_module("av")
        mcap_reader = import_module("mcap.reader")
        decoder_module = import_module("mcap_protobuf.decoder")
        import_module("PIL.Image")
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ProductionWemmShadowError(
            "production WeMM shadow requires av, mcap, mcap-protobuf, and Pillow"
        ) from exc

    topic_to_camera = {topic: camera for camera, topic in camera_pairs}
    camera_state: dict[str, dict[str, Any]] = {}
    for camera_id, _topic in camera_pairs:
        camera_state[camera_id] = {
            "decoder": av.CodecContext.create("h264", "r"),
            "examined": 0,
            "decoded": 0,
            "failures": [],
            "selected": {
                window["window_id"]: {
                    "targets": [
                        common_start
                        + round(
                            (
                                window["start_seconds"]
                                + (window["end_seconds"] - window["start_seconds"])
                                * index
                                / frame_count
                            )
                            * 1_000_000_000
                        )
                        for index in range(frame_count)
                    ],
                    "frames": {},
                }
                for window in windows
            },
        }

    max_end_ns = common_start + int(
        max(window["end_seconds"] for window in windows) * 1_000_000_000
    )
    try:
        with source.open("rb") as stream:
            reader = mcap_reader.make_reader(
                stream,
                validate_crcs=validate_crcs,
                decoder_factories=[decoder_module.DecoderFactory()],
            )
            for schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=tuple(topic_to_camera), log_time_order=False
            ):
                topic = getattr(channel, "topic", None)
                if not isinstance(topic, str):
                    continue
                observed_camera_id = topic_to_camera.get(topic)
                if observed_camera_id is None:
                    continue
                state = camera_state[observed_camera_id]
                timestamp = int(getattr(message, "log_time", 0) or 0)
                state["examined"] += 1
                if getattr(schema, "name", None) != _COMPRESSED_SCHEMA:
                    state["failures"].append("INVALID_COMPRESSED_IMAGE_SCHEMA")
                    continue
                payload = getattr(decoded, "data", None)
                if not isinstance(payload, bytes) or not payload:
                    state["failures"].append("INVALID_COMPRESSED_IMAGE_PAYLOAD")
                    continue
                try:
                    packet = av.Packet(payload)
                    packet.pts = timestamp
                    packet.dts = timestamp
                    packet.time_base = Fraction(1, 1_000_000_000)
                    frames = state["decoder"].decode(packet)
                except Exception as exc:  # decoder errors are retained, not fatal
                    state["failures"].append(f"H264_DECODE_ERROR:{type(exc).__name__}")
                    continue
                state["decoded"] += len(frames or ())
                for frame in frames or ():
                    frame_ts = _decode_frame_timestamp(frame, timestamp)
                    if frame_ts < common_start or frame_ts > max_end_ns:
                        continue
                    for window in windows:
                        window_id = window["window_id"]
                        start_ns = common_start + int(window["start_seconds"] * 1_000_000_000)
                        end_ns = common_start + int(window["end_seconds"] * 1_000_000_000)
                        if not start_ns <= frame_ts < end_ns:
                            continue
                        selected = state["selected"][window_id]
                        targets = selected["targets"]
                        # Choose the nearest target in this bounded window and
                        # retain the best observation if a later packet is closer.
                        index = min(
                            range(frame_count), key=lambda item: abs(frame_ts - targets[item])
                        )
                        previous = selected["frames"].get(index)
                        delta = abs(frame_ts - targets[index])
                        if previous is not None and previous["delta"] <= delta:
                            continue
                        image = _to_rgb_image(frame)
                        selected["frames"][index] = {
                            "delta": delta,
                            "timestamp_ns": frame_ts,
                            "image": image,
                        }
                # Once every target is populated for every camera, no more
                # source packets are needed for this bounded shadow run.
                if (
                    all(
                        len(state["selected"][window["window_id"]]["frames"]) == frame_count
                        for state in camera_state.values()
                        for window in windows
                    )
                    and timestamp >= max_end_ns
                ):
                    break
    except OSError as exc:
        raise ProductionWemmShadowError(f"could not read MCAP source: {exc}") from exc

    result: dict[str, dict[str, ProductionFrameGroup]] = {}
    fps_by_camera = {
        camera_id: max(
            1.0,
            float(
                next(
                    (
                        camera.get("frame_count", 0)
                        / max(float(camera.get("duration_seconds", 1.0)), 1e-6)
                        for camera in _sequence(
                            manifest["source"]["cameras"], field="source.cameras"
                        )
                        if isinstance(camera, Mapping) and camera.get("camera_id") == camera_id
                    ),
                    1.0,
                )
            ),
        )
        for camera_id, _topic in camera_pairs
    }
    for camera_id, _topic in camera_pairs:
        state = camera_state[camera_id]
        result[camera_id] = {}
        for window in windows:
            window_id = window["window_id"]
            selected = state["selected"][window_id]["frames"]
            if len(selected) != frame_count:
                raise ProductionWemmShadowError(
                    f"camera {camera_id} window {window_id} yielded "
                    f"{len(selected)}/{frame_count} frames"
                )
            ordered = [selected[index] for index in range(frame_count)]
            first_image = ordered[0]["image"]
            result[camera_id][window_id] = ProductionFrameGroup(
                camera_id=camera_id,
                window_id=window_id,
                frames=tuple(item["image"] for item in ordered),
                selected_timestamps_ns=tuple(int(item["timestamp_ns"]) for item in ordered),
                messages_examined=int(state["examined"]),
                decoded_frames=int(state["decoded"]),
                decode_failures=tuple(str(item) for item in state["failures"]),
                width=int(getattr(first_image, "width", 0) or 0),
                height=int(getattr(first_image, "height", 0) or 0),
                fps=fps_by_camera[camera_id],
                start_seconds=float(window["start_seconds"]),
                end_seconds=float(window["end_seconds"]),
            )
    return result


def iter_decode_production_window_chunks(
    manifest: Mapping[str, Any],
    *,
    frame_count: int = 4,
    validate_crcs: bool = False,
    window_chunk_size: int = 1,
) -> Iterator[dict[str, dict[str, ProductionFrameGroup]]]:
    """Yield decoded windows in bounded chunks while decoding the source once.

    Unlike :func:`decode_production_windows`, this iterator keeps one H.264
    ``CodecContext`` per camera for the lifetime of the source scan.  Windows
    are consumed in manifest order and only the selected PIL frames for the
    current chunk (plus a small boundary carry queue) are retained.  The
    yielded mapping has the same camera/window shape as the legacy decoder, so
    callers can process a chunk and release its images before requesting the
    next one.

    ``messages_examined``, ``decoded_frames`` and ``decode_failures`` in each
    yielded group are cumulative for that camera up to the emitted chunk, which
    matches the legacy decoder's cumulative metadata semantics.  Recoverable
    codec errors are retained as warnings and never terminate the iterator.
    """

    frame_count = _positive_int(frame_count, field="frame_count")
    if frame_count < 2 or frame_count > 64:
        raise ProductionWemmShadowError("frame_count must be between 2 and 64")
    if not isinstance(validate_crcs, bool):
        raise ProductionWemmShadowError("validate_crcs must be boolean")
    window_chunk_size = _positive_int(window_chunk_size, field="window_chunk_size")
    source, common_start, windows, camera_pairs = _load_manifest(manifest)
    try:
        av = import_module("av")
        mcap_reader = import_module("mcap.reader")
        decoder_module = import_module("mcap_protobuf.decoder")
        import_module("PIL.Image")
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ProductionWemmShadowError(
            "production WeMM shadow requires av, mcap, mcap-protobuf, and Pillow"
        ) from exc

    # Build absolute timestamp specs once.  The target calculation deliberately
    # mirrors the legacy decoder (rounding target timestamps and truncating
    # interval bounds) to preserve frame selection at boundaries.
    specs: list[dict[str, Any]] = []
    for window in windows:
        start_seconds = float(window["start_seconds"])
        end_seconds = float(window["end_seconds"])
        specs.append(
            {
                **window,
                "start_ns": common_start + int(start_seconds * 1_000_000_000),
                "end_ns": common_start + int(end_seconds * 1_000_000_000),
                "targets": [
                    common_start
                    + round(
                        (start_seconds + (end_seconds - start_seconds) * index / frame_count)
                        * 1_000_000_000
                    )
                    for index in range(frame_count)
                ],
            }
        )

    fps_by_camera = {
        camera_id: max(
            1.0,
            float(
                next(
                    (
                        camera.get("frame_count", 0)
                        / max(float(camera.get("duration_seconds", 1.0)), 1e-6)
                        for camera in _sequence(
                            manifest["source"]["cameras"], field="source.cameras"
                        )
                        if isinstance(camera, Mapping) and camera.get("camera_id") == camera_id
                    ),
                    1.0,
                )
            ),
        )
        for camera_id, _topic in camera_pairs
    }

    topic_to_camera = {topic: camera for camera, topic in camera_pairs}
    camera_state: dict[str, dict[str, Any]] = {
        camera_id: {
            "decoder": av.CodecContext.create("h264", "r"),
            "examined": 0,
            "decoded": 0,
            "failures": [],
            # Frames that arrive before a future overlapping context is
            # activated are retained as the nearest candidate for that
            # context's targets.  The map is target-indexed rather than a raw
            # frame queue, so memory is bounded by the number of active
            # overlapping contexts and ``frame_count`` even when MCAP channel
            # order is skewed.
            "carry": {},
        }
        for camera_id, _topic in camera_pairs
    }

    def _close_image(image: Any) -> None:
        close = getattr(image, "close", None)
        if callable(close):
            with suppress(Exception):
                close()

    def _new_selections(
        chunk_specs: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
        return {
            camera_id: {str(spec["window_id"]): {} for spec in chunk_specs}
            for camera_id, _topic in camera_pairs
        }

    def _future_index(
        specs: Sequence[Mapping[str, Any]],
    ) -> tuple[Sequence[Mapping[str, Any]], tuple[int, ...], tuple[int, ...], bool]:
        """Precompute temporal bounds used by per-frame future matching.

        Dense runs can contain thousands of windows.  Rebuilding integer bound
        tuples for every decoded frame turns an otherwise bounded carry lookup
        into an avoidable quadratic CPU cost, so this index is rebuilt only
        when the active chunk advances.
        """

        starts = tuple(int(spec["start_ns"]) for spec in specs)
        ends = tuple(int(spec["end_ns"]) for spec in specs)
        monotonic_bounds = all(left <= right for left, right in pairwise(starts)) and all(
            left <= right for left, right in pairwise(ends)
        )
        return specs, starts, ends, monotonic_bounds

    def _assign_frame(
        state: dict[str, Any],
        frame_ts: int,
        frame: Any,
        chunk_specs: Sequence[Mapping[str, Any]],
        future_index: tuple[Sequence[Mapping[str, Any]], tuple[int, ...], tuple[int, ...], bool],
    ) -> None:
        """Assign one decoded frame to every matching current/future context.

        Dense context windows overlap by design.  A decoded frame can therefore
        be the nearest sample for several windows (and for several chunks), not
        just the first matching window.  The old implementation returned after
        the first assignment and only carried frames into the immediately next
        chunk, which left dense windows such as ``[1, 5)`` with one selected
        frame out of four.  We now retain target-indexed candidates for *all*
        future windows that contain the frame and drain them when their chunk
        becomes current.

        Each destination receives its own PIL image.  Sharing one image object
        across windows would let the consumer closing one ``ProductionFrameGroup``
        invalidate a sibling group that still owns the same frame.
        """

        future_specs, future_starts, future_ends, monotonic_bounds = future_index
        if not chunk_specs and not future_specs:
            return

        # ``future_specs`` is sorted by source start in normal manifests.  Use
        # binary search to avoid scanning all later windows for every decoded
        # frame, while retaining a conservative linear fallback for an
        # externally supplied manifest whose starts or end times are not
        # monotonic.  ``bisect_right`` is valid only when both series are
        # ordered; checking ends alone could silently omit a future context.
        future_matches: Sequence[Mapping[str, Any]] = ()
        if future_specs:
            if monotonic_bounds:
                started = bisect_right(future_starts, frame_ts)
                expired = bisect_right(future_ends, frame_ts)
                future_matches = tuple(
                    spec
                    for spec in future_specs[expired:started]
                    if int(spec["start_ns"]) <= frame_ts < int(spec["end_ns"])
                )
            else:
                future_matches = tuple(
                    spec
                    for spec in future_specs
                    if int(spec["start_ns"]) <= frame_ts < int(spec["end_ns"])
                )

        base_image: Any | None = None
        base_transferred = False

        def _image_for_destination() -> Any:
            """Return an independently owned image for one window target."""

            nonlocal base_image, base_transferred
            if base_image is None:
                base_image = _to_rgb_image(frame)
            if not base_transferred:
                # Transfer the first conversion directly; subsequent targets
                # receive a copy (or a fresh conversion for lightweight test
                # doubles without ``copy``).
                base_transferred = True
                return base_image
            copy = getattr(base_image, "copy", None)
            if callable(copy):
                try:
                    copied = copy()
                    if copied is not base_image:
                        return copied
                except Exception:
                    pass
            return _to_rgb_image(frame)

        def _store(
            destination: dict[int, dict[str, Any]],
            spec: Mapping[str, Any],
        ) -> None:
            targets = spec["targets"]
            index = min(range(frame_count), key=lambda item: abs(frame_ts - targets[item]))
            delta = abs(frame_ts - targets[index])
            previous = destination.get(index)
            if previous is not None and previous["delta"] <= delta:
                return
            try:
                image = _image_for_destination()
            except Exception as exc:
                state["failures"].append(f"FRAME_CONVERSION_ERROR:{type(exc).__name__}")
                return
            if previous is not None:
                _close_image(previous["image"])
            destination[index] = {
                "delta": delta,
                "timestamp_ns": frame_ts,
                "image": image,
            }

        try:
            selected_by_window = state["selected"]
            for spec in chunk_specs:
                start_ns = int(spec["start_ns"])
                end_ns = int(spec["end_ns"])
                if start_ns <= frame_ts < end_ns:
                    window_id = str(spec["window_id"])
                    selected = selected_by_window.get(window_id)
                    if selected is not None:
                        _store(selected, spec)

            # Do not return after a current assignment: the same decoded frame
            # may also be needed by one or more overlapping future contexts.
            for spec in future_matches:
                window_id = str(spec["window_id"])
                carry_window = state["carry"].setdefault(window_id, {})
                _store(carry_window, spec)
        finally:
            # If the first conversion was never transferred (for example, all
            # candidate targets already had a closer frame), release it here.
            if base_image is not None and not base_transferred:
                _close_image(base_image)

    def _drain_carry(
        state: dict[str, Any],
        chunk_specs: Sequence[Mapping[str, Any]],
        remaining_specs: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Move candidates for the current chunk into selected windows.

        Carry for later overlapping chunks must survive this transition.  Only
        entries for the newly current windows are drained; entries whose window
        has already fallen outside the remaining plan are closed explicitly so
        a malformed/irregular manifest cannot leak PIL images.
        """

        carry = state.setdefault("carry", {})
        current_ids = {str(spec["window_id"]) for spec in chunk_specs}
        remaining_ids = {str(spec["window_id"]) for spec in remaining_specs}
        for window_id in current_ids:
            candidates = carry.pop(window_id, {})
            selected = state["selected"].get(window_id)
            if selected is None:
                for item in candidates.values():
                    _close_image(item["image"])
                continue
            for index, item in candidates.items():
                previous = selected.get(index)
                if previous is not None and previous["delta"] <= item["delta"]:
                    _close_image(item["image"])
                    continue
                if previous is not None:
                    _close_image(previous["image"])
                selected[index] = item

        valid_ids = current_ids | remaining_ids
        for window_id in tuple(carry):
            if window_id in valid_ids:
                continue
            stale = carry.pop(window_id)
            for item in stale.values():
                _close_image(item["image"])

    def _close_unconsumed_carry() -> None:
        """Release future-window images when the iterator stops early."""

        for state in state_by_camera.values():
            carry = state.get("carry", {})
            if not isinstance(carry, dict):
                continue
            for candidates in carry.values():
                if not isinstance(candidates, Mapping):
                    continue
                for item in candidates.values():
                    if isinstance(item, Mapping):
                        _close_image(item.get("image"))
            carry.clear()

    def _chunk_ready(
        state_by_camera: Mapping[str, Mapping[str, Any]],
        chunk_specs: Sequence[Mapping[str, Any]],
    ) -> bool:
        if not chunk_specs:
            return False
        for state in state_by_camera.values():
            if not state.get("past_end", False):
                return False
            for spec in chunk_specs:
                if len(state["selected"][str(spec["window_id"])]) != frame_count:
                    return False
        return True

    def _build_chunk(
        state_by_camera: Mapping[str, Mapping[str, Any]],
        chunk_specs: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, ProductionFrameGroup]]:
        result: dict[str, dict[str, ProductionFrameGroup]] = {}
        for camera_id, _topic in camera_pairs:
            state = state_by_camera[camera_id]
            result[camera_id] = {}
            for spec in chunk_specs:
                window_id = str(spec["window_id"])
                selected = state["selected"][window_id]
                if len(selected) != frame_count:
                    raise ProductionWemmShadowError(
                        f"camera {camera_id} window {window_id} yielded "
                        f"{len(selected)}/{frame_count} frames"
                    )
                ordered = [selected[index] for index in range(frame_count)]
                first_image = ordered[0]["image"]
                result[camera_id][window_id] = ProductionFrameGroup(
                    camera_id=camera_id,
                    window_id=window_id,
                    frames=tuple(item["image"] for item in ordered),
                    selected_timestamps_ns=tuple(int(item["timestamp_ns"]) for item in ordered),
                    messages_examined=int(state["examined"]),
                    decoded_frames=int(state["decoded"]),
                    decode_failures=tuple(str(item) for item in state["failures"]),
                    width=int(getattr(first_image, "width", 0) or 0),
                    height=int(getattr(first_image, "height", 0) or 0),
                    fps=fps_by_camera[camera_id],
                    start_seconds=float(spec["start_seconds"]),
                    end_seconds=float(spec["end_seconds"]),
                )
        return result

    chunk_specs_list = [
        specs[index : index + window_chunk_size]
        for index in range(0, len(specs), window_chunk_size)
    ]
    state_by_camera = camera_state
    chunk_index = 0
    current_specs: Sequence[Mapping[str, Any]] = ()
    future_specs: Sequence[Mapping[str, Any]] = ()
    future_index = _future_index(future_specs)
    try:
        with source.open("rb") as stream:
            reader = mcap_reader.make_reader(
                stream,
                validate_crcs=validate_crcs,
                decoder_factories=[decoder_module.DecoderFactory()],
            )
            current_specs = chunk_specs_list[0]
            future_specs = tuple(spec for chunk in chunk_specs_list[1:] for spec in chunk)
            future_index = _future_index(future_specs)
            state_by_camera = {
                camera_id: {**state, "selected": _new_selections(current_specs), "past_end": False}
                for camera_id, state in camera_state.items()
            }
            # ``_new_selections`` creates all cameras; retain only this camera's
            # branch in each state to keep the state shape simple.
            for camera_id in state_by_camera:
                state_by_camera[camera_id]["selected"] = {
                    str(spec["window_id"]): {} for spec in current_specs
                }
            for schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=tuple(topic_to_camera), log_time_order=False
            ):
                topic = getattr(channel, "topic", None)
                if not isinstance(topic, str):
                    continue
                mapped_camera_id = topic_to_camera.get(topic)
                if mapped_camera_id is None:
                    continue
                state = state_by_camera[mapped_camera_id]
                timestamp = int(getattr(message, "log_time", 0) or 0)
                state["examined"] += 1
                if getattr(schema, "name", None) != _COMPRESSED_SCHEMA:
                    state["failures"].append("INVALID_COMPRESSED_IMAGE_SCHEMA")
                    continue
                payload = getattr(decoded, "data", None)
                if not isinstance(payload, bytes) or not payload:
                    state["failures"].append("INVALID_COMPRESSED_IMAGE_PAYLOAD")
                    continue
                try:
                    packet = av.Packet(payload)
                    packet.pts = timestamp
                    packet.dts = timestamp
                    packet.time_base = Fraction(1, 1_000_000_000)
                    frames = state["decoder"].decode(packet)
                except Exception as exc:  # decoder errors are retained, not fatal
                    state["failures"].append(f"H264_DECODE_ERROR:{type(exc).__name__}")
                    frames = ()
                state["decoded"] += len(frames or ())
                for frame in frames or ():
                    frame_ts = _decode_frame_timestamp(frame, timestamp)
                    _assign_frame(state, frame_ts, frame, current_specs, future_index)
                if timestamp >= int(current_specs[-1]["end_ns"]):
                    state["past_end"] = True

                if _chunk_ready(state_by_camera, current_specs):
                    emitted = _build_chunk(state_by_camera, current_specs)
                    yield emitted
                    del emitted
                    chunk_index += 1
                    if chunk_index >= len(chunk_specs_list):
                        _close_unconsumed_carry()
                        return
                    current_specs = chunk_specs_list[chunk_index]
                    future_specs = tuple(
                        spec for chunk in chunk_specs_list[chunk_index + 1 :] for spec in chunk
                    )
                    future_index = _future_index(future_specs)
                    for camera_state_item in state_by_camera.values():
                        camera_state_item["selected"] = {
                            str(spec["window_id"]): {} for spec in current_specs
                        }
                        camera_state_item["past_end"] = False
                        _drain_carry(camera_state_item, current_specs, future_specs)
    except OSError as exc:
        _close_unconsumed_carry()
        raise ProductionWemmShadowError(f"could not read MCAP source: {exc}") from exc
    except BaseException:
        # Includes ``GeneratorExit`` when a consumer abandons a partial run.
        _close_unconsumed_carry()
        raise

    # EOF: permit the final chunk to be emitted if all requested frames were
    # selected.  Otherwise preserve the legacy diagnostic with the first
    # missing camera/window.
    for state in state_by_camera.values():
        state["past_end"] = True
    while current_specs and chunk_index < len(chunk_specs_list):
        if not _chunk_ready(state_by_camera, current_specs):
            for camera_id, _topic in camera_pairs:
                state = state_by_camera[camera_id]
                for spec in current_specs:
                    selected_count = len(state["selected"][str(spec["window_id"])])
                    if selected_count != frame_count:
                        raise ProductionWemmShadowError(
                            f"camera {camera_id} window {spec['window_id']} yielded "
                            f"{selected_count}/{frame_count} frames"
                        )
            raise ProductionWemmShadowError("decoder ended before all windows were selected")
        emitted = _build_chunk(state_by_camera, current_specs)
        yield emitted
        del emitted
        chunk_index += 1
        if chunk_index >= len(chunk_specs_list):
            break
        current_specs = chunk_specs_list[chunk_index]
        future_specs = tuple(
            spec for chunk in chunk_specs_list[chunk_index + 1 :] for spec in chunk
        )
        future_index = _future_index(future_specs)
        for camera_state_item in state_by_camera.values():
            camera_state_item["selected"] = {str(spec["window_id"]): {} for spec in current_specs}
            camera_state_item["past_end"] = True
            _drain_carry(camera_state_item, current_specs, future_specs)

    _close_unconsumed_carry()


def _read_classes(path: Path, *, field: str) -> dict[int, str]:
    if not path.is_file():
        raise ProductionWemmShadowError(f"{field} is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if (
                reader.fieldnames is None
                or "id" not in reader.fieldnames
                or "key" not in reader.fieldnames
            ):
                raise ProductionWemmShadowError(f"{field} must contain id,key columns")
            result: dict[int, str] = {}
            for row in reader:
                try:
                    class_id = int(str(row["id"]).strip())
                except (TypeError, ValueError) as exc:
                    raise ProductionWemmShadowError(f"invalid {field} class id") from exc
                key = _text(row.get("key"), field=f"{field}[{class_id}].key")
                result[class_id] = key
    except OSError as exc:
        raise ProductionWemmShadowError(f"could not read {field}: {exc}") from exc
    if not result:
        raise ProductionWemmShadowError(f"{field} is empty")
    return dict(sorted(result.items()))


def _read_pairs(
    path: Path,
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionWemmShadowError(f"could not read ontology pair catalog: {path}") from exc
    document = _mapping(payload, field="ontology pair catalog")
    raw_pairs = _sequence(
        document.get("action_pairs", document.get("pairs")), field="ontology.action_pairs"
    )
    provenance = _mapping(document.get("provenance"), field="ontology.provenance")
    if provenance.get("label_blind") is not True:
        raise ProductionWemmShadowError(
            "ontology pair catalog must declare provenance.label_blind=true"
        )
    pairs: list[tuple[int, int]] = []
    for index, raw in enumerate(raw_pairs):
        values = _sequence(raw, field=f"ontology.action_pairs[{index}]")
        if len(values) != 2:
            raise ProductionWemmShadowError("ontology action pairs must contain two IDs")
        try:
            pair = (int(values[0]), int(values[1]))
        except (TypeError, ValueError) as exc:
            raise ProductionWemmShadowError("ontology action pair IDs must be integers") from exc
        if pair[0] < 0 or pair[1] < 0:
            raise ProductionWemmShadowError("ontology action pair IDs must be non-negative")
        pairs.append(pair)
    if not pairs:
        raise ProductionWemmShadowError("ontology action pair catalog is empty")
    return tuple(dict.fromkeys(pairs)), {
        "path": str(path.resolve()),
        "format": document.get("format"),
        "source": provenance.get("source"),
        "split": provenance.get("split"),
        "label_blind": True,
    }


def _prediction_row(item: RetrievedAction, *, camera_id: str) -> dict[str, Any]:
    row: dict[str, Any] = dict(item.to_dict())
    row["camera_id"] = camera_id
    row["verb"] = item.label.verb_key
    row["noun"] = item.label.noun_key
    row["source"] = "wemm_visual_embedding"
    return row


def run_production_wemm_shadow(
    manifest: Mapping[str, Any],
    *,
    model_directory: str | Path,
    verb_classes: str | Path,
    noun_classes: str | Path,
    ontology_pairs: str | Path,
    frame_count: int = 4,
    top_k: int = 10,
    dimension: int = 2048,
    device: str = "cuda",
    label_variant: LabelVariant = "canonical",
    max_windows: int | None = None,
    validate_crcs: bool = False,
) -> dict[str, Any]:
    """Execute the label-blind WeMM production shadow route."""

    top_k = _positive_int(top_k, field="top_k")
    dimension = _positive_int(dimension, field="dimension")
    if not isinstance(device, str) or not device.strip():
        raise ProductionWemmShadowError("device must be non-empty")
    source_path, _common_start, windows, camera_pairs = _load_manifest(manifest)
    if max_windows is not None:
        max_windows = _positive_int(max_windows, field="max_windows")
        windows = windows[:max_windows]
    if not windows:
        raise ProductionWemmShadowError("no windows selected")
    verbs = _read_classes(Path(verb_classes).expanduser().resolve(), field="verb_classes")
    nouns = _read_classes(Path(noun_classes).expanduser().resolve(), field="noun_classes")
    pairs, provenance = _read_pairs(Path(ontology_pairs).expanduser().resolve())
    labels = build_joint_action_catalog(
        verb_table_or_entries=verbs,
        noun_table_or_entries=nouns,
        action_pairs=pairs,
    )
    try:
        label_texts = [label.text_for(label_variant) for label in labels]
    except Exception as exc:
        raise ProductionWemmShadowError(f"unsupported label_variant: {label_variant!r}") from exc

    groups = decode_production_windows(
        {
            **dict(manifest),
            "windows": list(windows),
        },
        frame_count=frame_count,
        validate_crcs=validate_crcs,
    )
    backend = WemmEmbeddingBackend(
        model_directory=model_directory,
        device=device,
        dimension=dimension,
    )
    try:
        label_vectors = backend.encode_texts(label_texts, batch_size=32)
        label_embeddings = {
            label.action_key: vector for label, vector in zip(labels, label_vectors, strict=True)
        }
        output_windows: list[dict[str, Any]] = []
        camera_order = [camera for camera, _topic in camera_pairs]
        for window in windows:
            window_id = window["window_id"]
            per_camera_payload: dict[str, dict[str, Any]] = {}
            per_camera_predictions: dict[str, list[dict[str, Any]]] = {}
            input_observations: list[dict[str, Any]] = []
            for camera_id in camera_order:
                group = groups[camera_id][window_id]
                query_vector = backend.encode_video_frames(
                    [group.frames], metadata_groups=[group.metadata()]
                )[0]
                ranked = rank_joint_actions(
                    labels=labels,
                    query_embedding=query_vector,
                    label_embeddings=label_embeddings,
                    label_variant=label_variant,
                    mode="visual",
                    top_k=top_k,
                )
                predictions = [_prediction_row(item, camera_id=camera_id) for item in ranked]
                per_camera_predictions[camera_id] = predictions
                per_camera_payload[camera_id] = {
                    "query_embedding": query_vector,
                    "candidates": [
                        {
                            "action_key": row["action_key"],
                            "rank": row["rank"],
                            "score": row["visual_score"],
                            "verb_key": row["verb_key"],
                            "noun_key": row["noun_key"],
                            "label_text": row["label_text"],
                            "camera_id": camera_id,
                        }
                        for row in predictions
                    ],
                }
                input_observations.append(
                    {
                        **group.to_dict(),
                        "model_observation": backend.observations[-1].to_dict(),
                    }
                )
            fused = fuse_camera_rankings(
                per_camera_payload,
                camera_order=camera_order,
                expected_cameras=camera_order,
                top_k=top_k,
                fusion="mean",
                score_normalization="unit",
                missing_score="omit",
                include_embeddings=False,
            )
            fused_predictions: list[dict[str, Any]] = []
            for row in fused.get("candidates", fused.get("ranking", [])):
                if not isinstance(row, Mapping):
                    continue
                action = row.get("action_key")
                if not isinstance(action, Sequence) or len(action) != 2:
                    continue
                try:
                    label = next(
                        item
                        for item in labels
                        if item.action_key == (int(action[0]), int(action[1]))
                    )
                except (StopIteration, TypeError, ValueError):
                    continue
                fused_predictions.append(
                    {
                        "rank": row.get("rank"),
                        "action_key": list(label.action_key),
                        "verb": label.verb_key,
                        "noun": label.noun_key,
                        "label_text": label.text_for(label_variant),
                        "score": row.get("fused_score", row.get("score")),
                        "camera_coverage": row.get("camera_coverage"),
                        "camera_coverage_fraction": row.get("camera_coverage_fraction"),
                        "source": "wemm_multiview_mean_fusion",
                    }
                )
            output_windows.append(
                {
                    "ordinal": window["ordinal"],
                    "window_id": window_id,
                    "start_seconds": window["start_seconds"],
                    "end_seconds": window["end_seconds"],
                    "model": {
                        "model": "wemm",
                        "native_route": "complete_bounded_video_embedding",
                        "status": "SUCCEEDED",
                        "predictions": fused_predictions,
                        "per_camera_predictions": per_camera_predictions,
                        "input_observations": input_observations,
                        "fusion": fused,
                    },
                }
            )
        return {
            "format": PRODUCTION_WEMM_SHADOW_VERSION,
            "authority": AUTHORITY,
            "production_eligible": False,
            "source": {
                "path": str(source_path),
                "manifest_format": manifest.get("format"),
                "window_count": len(output_windows),
                "camera_count": len(camera_order),
            },
            "model": {
                "identifier": "WeMM-Embedding-2B",
                "model_directory": str(Path(model_directory).expanduser().resolve()),
                "dimension": dimension,
                "label_variant": label_variant,
                "frame_count": frame_count,
            },
            "ontology": {
                **provenance,
                "pair_count": len(labels),
                "profile": _ontology_profile(provenance),
            },
            "windows": output_windows,
            "quality": {
                "measurement_status": "NOT_MEASURED",
                "reason": "production cohort has no accepted source-bound action gold",
                "values": {},
            },
            "controls": {
                "model_invoked": True,
                "gold_included": False,
                "predictions_are_gold": False,
                "existing_mapper_invoked": False,
                "ontology_modified": False,
                "mapper_modified": False,
                "training_invoked": False,
                "heldout_100_opened": False,
                "hash_or_sha_used": False,
                "ground_truth_used_in_encoder_input": False,
            },
            "backend_observations": backend.observation_payload(),
        }
    finally:
        backend.close()


__all__ = [
    "AUTHORITY",
    "CAMERA_IDS",
    "PRODUCTION_WEMM_SHADOW_VERSION",
    "ProductionFrameGroup",
    "ProductionWemmShadowError",
    "decode_production_windows",
    "iter_decode_production_window_chunks",
    "run_production_wemm_shadow",
]
