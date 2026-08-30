"""Bounded, benchmark-local H.264 frame probes for a six-camera MCAP.

The probe is deliberately narrower than the canonical ingestion adapters.  It
reads only the six caller-supplied ``foxglove.CompressedImage`` topics and
stops each camera after its first decoded frame or a fixed message bound.  It
records source timestamps and decoded dimensions, but does not publish media,
load a model, or alter ontology/mapping state.

This module is useful for the production-shaped readiness track: it proves that
the mapped camera payloads can reach PyAV without turning a decoder smoke into
a quality or production qualification result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any, Final

COMPRESSED_IMAGE_SCHEMA: Final = "foxglove.CompressedImage"
CAMERA_IDS: Final = tuple(f"cam_{index:02d}" for index in range(1, 7))
DEFAULT_CAMERA_TOPICS: Final[dict[str, str]] = {
    camera_id: f"/robot0/sensor/camera{index}/compressed"
    for index, camera_id in enumerate(CAMERA_IDS)
}
PRODUCTION_MEDIA_DECODE_PROBE_VERSION: Final = "robata-production-media-decode-probe-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"


class ProductionMediaDecodeProbeError(ValueError):
    """Raised when probe inputs or the bounded decoder setup are invalid."""


@dataclass(frozen=True, slots=True)
class CameraDecodeObservation:
    """Source-bound result for one mapped camera topic."""

    camera_id: str
    topic: str
    schema: str
    codec: str
    success: bool
    source_timestamp_ns: int | None
    first_decoded_timestamp_ns: int | None
    width: int | None
    height: int | None
    messages_examined: int
    decoded_frames: int
    failures: tuple[dict[str, Any], ...]

    @property
    def decode_failures(self) -> int:
        """Count payloads that did not decode during the bounded probe."""

        return len(self.failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "topic": self.topic,
            "schema": self.schema,
            "codec": self.codec,
            "success": self.success,
            "source_timestamp_ns": self.source_timestamp_ns,
            "first_decoded_timestamp_ns": self.first_decoded_timestamp_ns,
            # ``timestamp_ns`` is a compact compatibility alias for consumers
            # that do not distinguish the source and decoded clock fields.
            "timestamp_ns": self.first_decoded_timestamp_ns,
            "width": self.width,
            "height": self.height,
            "messages_examined": self.messages_examined,
            "decoded_frames": self.decoded_frames,
            "decode_failures": self.decode_failures,
            "failures": [dict(item) for item in self.failures],
        }


def _clean_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionMediaDecodeProbeError(f"{field} must be a non-empty string")
    return value.strip()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionMediaDecodeProbeError(f"{field} must be an object")
    return value


def _as_sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionMediaDecodeProbeError(f"{field} must be an array")
    return value


def normalize_camera_topics(value: object | None = None) -> tuple[tuple[str, str], ...]:
    """Normalize a six-camera topic mapping into canonical camera order.

    The helper accepts the ``topics`` object from an observed mapping profile,
    a production cohort's ``camera_topics`` object, a ``cameras`` list, or a
    direct ``{camera_id: topic}`` object.  It never treats a partial mapping as
    six-camera coverage.
    """

    raw: object = DEFAULT_CAMERA_TOPICS if value is None else value
    if isinstance(raw, Mapping):
        # Mapping/config wrappers used by existing local artifacts.
        for wrapper_key in ("topics", "camera_topics"):
            if wrapper_key in raw:
                raw = raw[wrapper_key]
                break
        else:
            if "cameras" in raw and isinstance(raw["cameras"], Sequence):
                raw = raw["cameras"]
    pairs: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for camera_id, topic in raw.items():
            camera = _clean_text(camera_id, field="camera_topics.camera_id")
            pairs[camera] = _clean_text(topic, field=f"camera_topics.{camera}")
    else:
        rows = _as_sequence(raw, field="camera_topics")
        for index, row_value in enumerate(rows):
            row = _mapping(row_value, field=f"camera_topics[{index}]")
            camera = _clean_text(
                row.get("camera_id", row.get("id")),
                field=f"camera_topics[{index}].camera_id",
            )
            topic = _clean_text(
                row.get("topic", row.get("camera_topic")),
                field=f"camera_topics[{index}].topic",
            )
            if camera in pairs:
                raise ProductionMediaDecodeProbeError(f"duplicate camera ID: {camera}")
            pairs[camera] = topic

    expected = set(CAMERA_IDS)
    if set(pairs) != expected:
        missing = sorted(expected - set(pairs))
        extra = sorted(set(pairs) - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ProductionMediaDecodeProbeError(
            "camera mapping must contain exactly cam_01 through cam_06"
            + (f" ({'; '.join(details)})" if details else "")
        )
    topics = tuple((camera_id, pairs[camera_id]) for camera_id in CAMERA_IDS)
    if len({topic for _, topic in topics}) != len(topics):
        raise ProductionMediaDecodeProbeError("camera topics must be unique")
    return topics


def _failure(code: str, timestamp_ns: int | None, message: str) -> dict[str, Any]:
    return {
        "code": code,
        "timestamp_ns": timestamp_ns,
        "message": message,
    }


def _frame_timestamp_ns(frame: Any, fallback_ns: int) -> int:
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    if pts is None or time_base is None:
        return fallback_ns
    try:
        return int(Fraction(pts) * time_base * 1_000_000_000)
    except (TypeError, ValueError, OverflowError):
        return fallback_ns


def _schema_name(schema: Any) -> str | None:
    """Return a decoder schema name across MCAP and lightweight test doubles."""

    if isinstance(schema, str):
        return schema
    name = getattr(schema, "name", None)
    return name if isinstance(name, str) else None


def _message_timestamp_ns(message: Any) -> int | None:
    """Read an MCAP log timestamp without turning malformed metadata into a crash."""

    raw = getattr(message, "log_time", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _source_timestamp_ns(decoded: Any, fallback_ns: int | None) -> int | None:
    """Prefer the CompressedImage header clock, falling back to MCAP log time."""

    header = getattr(decoded, "header", None)
    raw = getattr(header, "timestamp", None)
    try:
        return int(raw) if raw is not None else fallback_ns
    except (TypeError, ValueError, OverflowError):
        return fallback_ns


def probe_production_media(
    source: str | Path,
    *,
    camera_topics: object | None = None,
    max_messages_per_camera: int = 120,
    validate_crcs: bool = True,
) -> dict[str, Any]:
    """Boundedly decode the first H.264 frame for each mapped camera.

    The reader is filtered to the six selected topics, and each topic gets its
    own PyAV decoder context.  A malformed packet contributes one structured
    failure and does not consume an unbounded retry loop.  The returned object
    is JSON-compatible and contains no model or publication result.
    """

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ProductionMediaDecodeProbeError(f"MCAP source is not a file: {path}")
    if isinstance(max_messages_per_camera, bool) or not isinstance(max_messages_per_camera, int):
        raise ProductionMediaDecodeProbeError("max_messages_per_camera must be a positive integer")
    if max_messages_per_camera <= 0:
        raise ProductionMediaDecodeProbeError("max_messages_per_camera must be a positive integer")
    if not isinstance(validate_crcs, bool):
        raise ProductionMediaDecodeProbeError("validate_crcs must be boolean")
    topics = normalize_camera_topics(camera_topics)

    try:
        av_module = import_module("av")
        reader_module = import_module("mcap.reader")
        decoder_module = import_module("mcap_protobuf.decoder")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProductionMediaDecodeProbeError(
            "bounded MCAP decoding requires PyAV, mcap, and mcap-protobuf"
        ) from exc

    topic_to_camera = {topic: camera_id for camera_id, topic in topics}
    camera_state: dict[str, dict[str, Any]] = {
        camera_id: {
            "camera_id": camera_id,
            "topic": topic,
            "examined": 0,
            "decoded_frames": 0,
            "failures": [],
            "source_timestamp_ns": None,
            "first_decoded_timestamp_ns": None,
            "width": None,
            "height": None,
            "success": False,
        }
        for camera_id, topic in topics
    }
    decoders: dict[str, Any] = {}
    for camera_id, _topic in topics:
        try:
            decoders[camera_id] = av_module.CodecContext.create("h264", "r")
        except Exception as exc:
            camera_state[camera_id]["failures"].append(
                _failure("H264_DECODER_INIT_ERROR", None, f"{type(exc).__name__}: {exc}")
            )

    reader = None
    try:
        with path.open("rb") as stream:
            reader = reader_module.make_reader(
                stream,
                validate_crcs=validate_crcs,
                decoder_factories=[decoder_module.DecoderFactory()],
            )
            iterator = reader.iter_decoded_messages(
                topics=tuple(topic_to_camera),
                log_time_order=False,
            )
            for schema, channel, message, decoded in iterator:
                observed_topic = getattr(channel, "topic", None)
                if not isinstance(observed_topic, str):
                    continue
                observed_camera_id = topic_to_camera.get(observed_topic)
                if observed_camera_id is None:
                    continue
                state = camera_state[observed_camera_id]
                if state["success"] or state["examined"] >= max_messages_per_camera:
                    continue
                state["examined"] += 1
                timestamp_ns = _message_timestamp_ns(message)
                schema_name = _schema_name(schema)
                if schema_name != COMPRESSED_IMAGE_SCHEMA:
                    state["failures"].append(
                        _failure(
                            "INVALID_COMPRESSED_IMAGE_SCHEMA",
                            timestamp_ns,
                            "mapped topic did not decode as foxglove.CompressedImage",
                        )
                    )
                    continue
                payload = getattr(decoded, "data", None)
                format_name = str(getattr(decoded, "format", "")).strip().casefold()
                if format_name != "h264":
                    state["failures"].append(
                        _failure(
                            "UNSUPPORTED_IMAGE_FORMAT",
                            timestamp_ns,
                            (
                                "mapped CompressedImage format is "
                                f"{format_name or '<empty>'!r}, expected 'h264'"
                            ),
                        )
                    )
                    continue
                if not isinstance(payload, bytes) or not payload:
                    state["failures"].append(
                        _failure(
                            "INVALID_COMPRESSED_IMAGE_PAYLOAD",
                            timestamp_ns,
                            "CompressedImage.data must be non-empty bytes",
                        )
                    )
                    continue
                decoder = decoders.get(observed_camera_id)
                if decoder is None:
                    continue
                try:
                    packet = av_module.Packet(payload)
                    if timestamp_ns is not None:
                        packet.pts = timestamp_ns
                        packet.dts = timestamp_ns
                    packet.time_base = Fraction(1, 1_000_000_000)
                except Exception as exc:
                    state["failures"].append(
                        _failure("H264_PACKET_ERROR", timestamp_ns, f"{type(exc).__name__}: {exc}")
                    )
                    continue
                try:
                    frames = decoder.decode(packet)
                except Exception as exc:
                    state["failures"].append(
                        _failure("H264_DECODE_ERROR", timestamp_ns, f"{type(exc).__name__}: {exc}")
                    )
                    frames = ()
                if frames is None:
                    frames = ()
                state["decoded_frames"] += len(frames)
                if frames:
                    frame = frames[0]
                    width = int(getattr(frame, "width", 0) or 0)
                    height = int(getattr(frame, "height", 0) or 0)
                    if width <= 0 or height <= 0:
                        state["failures"].append(
                            _failure(
                                "INVALID_DECODED_DIMENSIONS",
                                timestamp_ns,
                                "decoded frame dimensions must be positive",
                            )
                        )
                        continue
                    state["success"] = True
                    state["source_timestamp_ns"] = _source_timestamp_ns(decoded, timestamp_ns)
                    state["first_decoded_timestamp_ns"] = _frame_timestamp_ns(
                        frame, timestamp_ns if timestamp_ns is not None else 0
                    )
                    state["width"] = width
                    state["height"] = height

                if all(
                    item["success"] or item["examined"] >= max_messages_per_camera
                    for item in camera_state.values()
                ):
                    break
    except OSError as exc:
        raise ProductionMediaDecodeProbeError(f"could not read MCAP source {path}: {exc}") from exc
    except ProductionMediaDecodeProbeError:
        raise
    except Exception as exc:
        # Keep a bounded, serialisable failure for each camera still in flight.
        for state in camera_state.values():
            if not state["success"] and not state["failures"]:
                state["failures"].append(
                    _failure("MCAP_READ_ERROR", None, f"{type(exc).__name__}: {exc}")
                )

    observations: list[CameraDecodeObservation] = []
    for camera_id, topic in topics:
        state = camera_state[camera_id]
        if state["examined"] == 0 and not state["success"]:
            state["failures"].append(
                _failure("CAMERA_TOPIC_NOT_OBSERVED", None, "mapped topic had no examined messages")
            )
        observations.append(
            CameraDecodeObservation(
                camera_id=camera_id,
                topic=topic,
                schema=COMPRESSED_IMAGE_SCHEMA,
                codec="h264",
                success=bool(state["success"]),
                source_timestamp_ns=state["source_timestamp_ns"],
                first_decoded_timestamp_ns=state["first_decoded_timestamp_ns"],
                width=state["width"],
                height=state["height"],
                messages_examined=int(state["examined"]),
                decoded_frames=int(state["decoded_frames"]),
                failures=tuple(dict(item) for item in state["failures"]),
            )
        )

    rows = [item.to_dict() for item in observations]
    successes = [item for item in observations if item.success]
    failure_count = sum(item.decode_failures for item in observations)
    return {
        "format": PRODUCTION_MEDIA_DECODE_PROBE_VERSION,
        "authority": AUTHORITY,
        "production_eligible": False,
        "probe_only": True,
        "source": {
            "path": str(path),
            "source_bound": True,
        },
        "camera_order": list(CAMERA_IDS),
        "camera_mapping": {camera_id: topic for camera_id, topic in topics},
        "schema": COMPRESSED_IMAGE_SCHEMA,
        "codec": "h264",
        "limits": {
            "max_messages_per_camera": max_messages_per_camera,
            "validate_crcs": validate_crcs,
        },
        "status": "SUCCEEDED" if len(successes) == len(observations) else "PARTIAL",
        "camera_count": len(observations),
        "decoded_camera_count": len(successes),
        "camera_coverage_fraction": len(successes) / len(observations),
        "messages_examined": sum(item.messages_examined for item in observations),
        "decode_failures": failure_count,
        "cameras": rows,
        "controls": {
            "model_invoked": False,
            "gpu_invoked": False,
            "frames_decoded": bool(successes),
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "quality_measured": False,
        },
        "quality_measurement_status": "NOT_MEASURED",
    }


__all__ = [
    "AUTHORITY",
    "CAMERA_IDS",
    "COMPRESSED_IMAGE_SCHEMA",
    "DEFAULT_CAMERA_TOPICS",
    "PRODUCTION_MEDIA_DECODE_PROBE_VERSION",
    "CameraDecodeObservation",
    "ProductionMediaDecodeProbeError",
    "normalize_camera_topics",
    "probe_production_media",
]
