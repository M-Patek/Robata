"""Lightweight MCAP-to-Qwen native-video input bridge.

The production Qwen verifier consumes a directory containing one complete MP4
per camera.  The canonical video-export service is intentionally registry and
digest aware; that is the right path for governed artifact publication, but it
is unnecessarily heavy for this benchmark-local input adapter.  This module
keeps the two concerns separate:

* MCAP inspection and topic mapping are factual and do not calculate a digest.
* The output manifest contains media/timeline facts only (never a SHA/hash).
* The legacy PyAV exporter is used through a tiny compatibility seam which
  suppresses its two legacy file-digest slots.  Those slots are not published
  by this bridge.  The seam is deliberately replaceable so a future exporter
  can implement a native no-digest result rather than relying on the legacy
  facts shape.

No model is loaded or invoked here.  A dry-run only inspects the MCAP and emits
the planned six-camera layout; materialization is an explicit separate call.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from robata.adapters.pyav_mp4_exporter import PyAvH264Mp4Exporter
from robata.contracts import CameraId
from robata.ingestion.mapping import TopicMappingProfile
from robata.ports.ingestion import COMPRESSED_IMAGE_SCHEMA, ChannelInspection
from robata.tempfiles import make_staging_directory

QWEN_NATIVE_VIDEO_BRIDGE_FORMAT: Final = "robata-qwen-native-video-bridge-v1"
QWEN_NATIVE_VIDEO_BRIDGE_MANIFEST_FILENAME: Final = "qwen-native-video-input-manifest.json"
QWEN_NATIVE_VIDEO_BRIDGE_VERSION: Final = "1"
QWEN_NATIVE_VIDEO_BRIDGE_AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
QWEN_NATIVE_VIDEO_BRIDGE_CAMERA_IDS: Final = tuple(f"cam_{index:02d}" for index in range(1, 7))
_DEFAULT_MAPPING_CONFIG = (
    Path(__file__).resolve().parents[3] / "config" / "genrobot-observed-v0.json"
)


class QwenNativeVideoBridgeError(RuntimeError):
    """A source, mapping, export, or manifest error at this bridge boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class QwenNativeVideoBridgeResult:
    """Result of a plan or materialization operation."""

    output_directory: Path
    manifest_path: Path
    manifest: dict[str, Any]
    dry_run: bool


class DigestFreeCameraVideoExporter(Protocol):
    """Minimal exporter surface needed by the bridge.

    The returned object is intentionally structural.  A caller may provide a
    future exporter that never exposes the legacy ``video_sha256`` fields.  The
    bundled compatibility exporter returns the existing facts object, while
    this bridge projects only the factual media fields needed by Qwen.
    """

    def export(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> Any:
        """Write one MP4 and timestamp sidecar without computing a digest."""


@dataclass(slots=True)
class _ChannelState:
    channel_id: int
    topic: str
    schema_name: str | None
    message_encoding: str
    schema_encoding: str | None
    message_count: int = 0
    first_message_time_ns: int | None = None
    last_message_time_ns: int | None = None
    previous_message_time_ns: int | None = None
    monotonic: bool = True
    codec: str | None = None
    frame_id: str | None = None
    metadata_decoded: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedBridge:
    source: Path
    source_size_bytes: int
    profile: TopicMappingProfile
    selected: dict[str, ChannelInspection]
    plan: dict[str, Any]


class NoDigestPyAvH264Mp4Exporter(PyAvH264Mp4Exporter):  # type: ignore[misc, unused-ignore]
    """Compatibility seam around the legacy PyAV exporter.

    ``PyAvH264RemuxSession.seal`` historically calls ``_hash_file`` solely to
    populate the two legacy fields on ``ExportedCameraVideoFacts``.  The bridge
    must not perform that operation, so this override reports file size and a
    ``None`` placeholder (the value is never serialized or used for identity).
    Existing governed callers continue to use ``PyAvH264Mp4Exporter`` unchanged.
    """

    @staticmethod
    def _hash_file(path: Path) -> tuple[int, Any]:
        # Keep the old private method signature for session compatibility.  A
        # stat is enough for the bridge and does not read/hash file contents.
        return path.stat().st_size, None


def inspect_mcap_without_digests(
    source: str | Path,
) -> tuple[Path, int, tuple[ChannelInspection, ...]]:
    """Scan MCAP channel/timestamp facts without hashing source or schemas.

    The official inspector intentionally computes source and schema digests for
    governed admission.  This adapter uses the lower-level reader directly so
    a dry-run cannot accidentally enter that identity path.
    """

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise QwenNativeVideoBridgeError("SOURCE_NOT_FOUND", f"MCAP source does not exist: {path}")
    if not path.is_file():
        raise QwenNativeVideoBridgeError("SOURCE_IO_ERROR", f"MCAP source is not a file: {path}")
    try:
        source_size_bytes = path.stat().st_size
    except OSError as exc:
        raise QwenNativeVideoBridgeError(
            "SOURCE_IO_ERROR", f"could not stat MCAP source {path}: {exc}"
        ) from exc

    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise QwenNativeVideoBridgeError(
            "MCAP_DEPENDENCY_UNAVAILABLE",
            "MCAP bridge requires mcap and mcap-protobuf support",
        ) from exc

    states: dict[int, _ChannelState] = {}
    try:
        with path.open("rb") as stream:
            reader = make_reader(
                stream,
                validate_crcs=True,
            )
            decoder_factory = DecoderFactory()
            # ``iter_decoded_messages`` eagerly decodes every schema in an MCAP.
            # Production recordings can carry unrelated malformed protobuf
            # descriptors, so use the raw iterator and decode only the camera
            # schema needed by this bridge.
            for schema, channel, message in reader.iter_messages(log_time_order=False):
                channel_id = _exact_int(getattr(channel, "id", None), "channel id")
                topic = _required_text(getattr(channel, "topic", None), "channel topic")
                schema_name = _schema_name(schema)
                schema_encoding = _schema_encoding(schema)
                message_encoding = _required_text(
                    getattr(channel, "message_encoding", None),
                    "message encoding",
                )
                state = states.get(channel_id)
                identity = (topic, schema_name, schema_encoding, message_encoding)
                if state is None:
                    state = _ChannelState(
                        channel_id=channel_id,
                        topic=topic,
                        schema_name=schema_name,
                        message_encoding=message_encoding,
                        schema_encoding=schema_encoding,
                    )
                    states[channel_id] = state
                elif identity != (
                    state.topic,
                    state.schema_name,
                    state.schema_encoding,
                    state.message_encoding,
                ):
                    raise QwenNativeVideoBridgeError(
                        "CORRUPT_MCAP",
                        f"channel {channel_id} changes identity within the MCAP",
                    )

                timestamp_ns = _exact_int(
                    getattr(message, "log_time", None),
                    "message log time",
                )
                state.message_count += 1
                if state.first_message_time_ns is None:
                    state.first_message_time_ns = timestamp_ns
                state.last_message_time_ns = timestamp_ns
                if (
                    state.previous_message_time_ns is not None
                    and timestamp_ns <= state.previous_message_time_ns
                ):
                    state.monotonic = False
                state.previous_message_time_ns = timestamp_ns

                if state.schema_name == COMPRESSED_IMAGE_SCHEMA and not state.metadata_decoded:
                    decoder = decoder_factory.decoder_for(message_encoding, schema)
                    if decoder is not None:
                        try:
                            decoded = decoder(message.data)
                        except Exception as exc:
                            raise QwenNativeVideoBridgeError(
                                "CORRUPT_MCAP",
                                f"camera message cannot be decoded: {type(exc).__name__}: {exc}",
                            ) from exc
                        codec = getattr(decoded, "format", None)
                        state.codec = codec.strip().lower() if isinstance(codec, str) else None
                        frame_id = getattr(decoded, "frame_id", None)
                        state.frame_id = frame_id if isinstance(frame_id, str) else None
                    state.metadata_decoded = True
    except QwenNativeVideoBridgeError:
        raise
    except OSError as exc:
        raise QwenNativeVideoBridgeError(
            "SOURCE_IO_ERROR", f"could not read MCAP source {path}: {exc}"
        ) from exc
    except Exception as exc:
        raise QwenNativeVideoBridgeError(
            "CORRUPT_MCAP", f"MCAP structure is unreadable: {type(exc).__name__}: {exc}"
        ) from exc

    channels = tuple(
        ChannelInspection(
            channel_id=state.channel_id,
            topic=state.topic,
            schema_name=state.schema_name,
            message_encoding=state.message_encoding,
            message_count=state.message_count,
            first_message_time_ns=state.first_message_time_ns,
            last_message_time_ns=state.last_message_time_ns,
            monotonic=state.monotonic,
            codec=state.codec,
            frame_id=state.frame_id,
            schema_encoding=state.schema_encoding,
            # Intentionally absent: schema/content digests are outside this
            # adapter's provenance domain.
            schema_content_sha256=None,
        )
        for state in sorted(states.values(), key=lambda value: value.channel_id)
    )
    return path, source_size_bytes, channels


def build_qwen_native_video_plan(
    source: str | Path,
    output_directory: str | Path,
    *,
    mapping_config: str | Path | None = None,
    mapping_profile: TopicMappingProfile | None = None,
    allow_unapproved_profile: bool = False,
) -> dict[str, Any]:
    """Build a digest-free six-camera output plan without writing media."""

    if type(allow_unapproved_profile) is not bool:
        raise QwenNativeVideoBridgeError(
            "INVALID_REQUEST", "allow_unapproved_profile must be a boolean"
        )
    return _prepare_bridge(
        source,
        output_directory,
        mapping_config=mapping_config,
        mapping_profile=mapping_profile,
        allow_unapproved_profile=allow_unapproved_profile,
    ).plan


def _prepare_bridge(
    source: str | Path,
    output_directory: str | Path,
    *,
    mapping_config: str | Path | None,
    mapping_profile: TopicMappingProfile | None,
    allow_unapproved_profile: bool,
) -> _PreparedBridge:
    if type(allow_unapproved_profile) is not bool:
        raise QwenNativeVideoBridgeError(
            "INVALID_REQUEST", "allow_unapproved_profile must be a boolean"
        )
    path, source_size_bytes, channels = inspect_mcap_without_digests(source)
    profile = _load_profile(mapping_config, mapping_profile)
    _authorize_profile(profile, allow_unapproved_profile)
    selected = _resolve_channels(channels, profile)
    output = Path(output_directory).expanduser().resolve()
    cameras = [
        _planned_camera_row(camera_id, selected[camera_id])
        for camera_id in QWEN_NATIVE_VIDEO_BRIDGE_CAMERA_IDS
    ]
    files = [filename for camera in cameras for filename in (camera["video"], camera["timestamps"])]
    files.append(QWEN_NATIVE_VIDEO_BRIDGE_MANIFEST_FILENAME)
    plan = {
        "format": QWEN_NATIVE_VIDEO_BRIDGE_FORMAT,
        "version": QWEN_NATIVE_VIDEO_BRIDGE_VERSION,
        "authority": QWEN_NATIVE_VIDEO_BRIDGE_AUTHORITY,
        "status": "DRY_RUN",
        "source": {
            "path": str(path),
            "media_type": "application/x-mcap",
            "size_bytes": source_size_bytes,
        },
        "output": {
            "directory": str(output),
            "video_root": str(output),
            "required_files": files,
            "camera_order": list(QWEN_NATIVE_VIDEO_BRIDGE_CAMERA_IDS),
        },
        "mapping": {
            "profile_id": profile.profile_id,
            "version": profile.version,
            "approval_status": profile.approval_status,
            "approved": profile.approved,
            "topics": {
                camera_id: selected[camera_id].topic
                for camera_id in QWEN_NATIVE_VIDEO_BRIDGE_CAMERA_IDS
            },
        },
        "cameras": cameras,
        "materialization": {
            "status": "NOT_STARTED",
            "backend": "no_digest_pyav_compatibility_seam",
            "complete_native_video": True,
        },
        "controls": {
            "model_invoked": False,
            "sha_or_digest_computed": False,
            "source_modified": False,
            "published_schema_modified": False,
        },
        "limitations": [
            "source_mutation_check_not_performed_without_content_identity",
            "legacy_exporter_digest_slots_are_isolated_and_not_published",
            "artifact_registry_publication_not_performed",
        ],
        "quality_status": "NOT_MEASURED",
    }
    return _PreparedBridge(
        source=path,
        source_size_bytes=source_size_bytes,
        profile=profile,
        selected=selected,
        plan=plan,
    )


def materialize_qwen_native_video_inputs(
    source: str | Path,
    output_directory: str | Path,
    *,
    mapping_config: str | Path | None = None,
    mapping_profile: TopicMappingProfile | None = None,
    allow_unapproved_profile: bool = False,
    exporter: DigestFreeCameraVideoExporter | None = None,
    dry_run: bool = False,
) -> QwenNativeVideoBridgeResult:
    """Materialize one complete six-camera Qwen video root.

    ``dry_run=True`` performs no writes other than stdout handled by a caller.
    A custom exporter is accepted for tests and future native no-digest backends;
    the default compatibility seam uses the existing PyAV remux implementation
    but suppresses its legacy file-digest computation.
    """

    if type(dry_run) is not bool:
        raise QwenNativeVideoBridgeError("INVALID_REQUEST", "dry_run must be a boolean")

    prepared = _prepare_bridge(
        source,
        output_directory,
        mapping_config=mapping_config,
        mapping_profile=mapping_profile,
        allow_unapproved_profile=allow_unapproved_profile,
    )
    plan = prepared.plan
    output = Path(output_directory).expanduser().resolve()
    manifest_path = output / QWEN_NATIVE_VIDEO_BRIDGE_MANIFEST_FILENAME
    if dry_run:
        return QwenNativeVideoBridgeResult(
            output_directory=output,
            manifest_path=manifest_path,
            manifest=plan,
            dry_run=True,
        )
    if os.path.lexists(output):
        raise QwenNativeVideoBridgeError(
            "OUTPUT_EXISTS",
            f"refusing to reuse or overwrite an existing output directory: {output}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    path = prepared.source
    selected = prepared.selected
    backend: DigestFreeCameraVideoExporter = (
        exporter if exporter is not None else NoDigestPyAvH264Mp4Exporter()
    )

    temporary = make_staging_directory(output.parent, prefix=f".{output.name}.qwen-")
    published = False
    try:
        rows: list[dict[str, Any]] = []
        for camera_id in QWEN_NATIVE_VIDEO_BRIDGE_CAMERA_IDS:
            camera = CameraId(camera_id)
            video_path = temporary / f"{camera_id}.mp4"
            timestamps_path = temporary / f"{camera_id}.timestamps.jsonl"
            try:
                facts = backend.export(
                    path,
                    camera,
                    selected[camera_id],
                    video_path,
                    timestamps_path,
                )
            except QwenNativeVideoBridgeError:
                raise
            except Exception as exc:
                raise QwenNativeVideoBridgeError(
                    "EXPORT_FAILED",
                    f"{camera_id} export failed: {type(exc).__name__}: {exc}",
                ) from exc
            _require_regular_file(video_path, f"{camera_id} MP4")
            _require_regular_file(timestamps_path, f"{camera_id} timestamp sidecar")
            rows.append(
                _materialized_camera_row(
                    camera_id,
                    selected[camera_id],
                    facts,
                    video_path,
                    timestamps_path,
                )
            )

        manifest = dict(plan)
        manifest["status"] = "MATERIALIZED"
        manifest["cameras"] = rows
        manifest["materialization"] = {
            "status": "COMPLETE",
            "backend": type(backend).__name__,
            "complete_native_video": True,
        }
        manifest_bytes = (
            json.dumps(
                manifest,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        manifest_path_temporary = temporary / QWEN_NATIVE_VIDEO_BRIDGE_MANIFEST_FILENAME
        manifest_path_temporary.write_bytes(manifest_bytes)
        _require_regular_file(manifest_path_temporary, "Qwen input manifest")
        try:
            temporary.rename(output)
        except FileExistsError as exc:
            raise QwenNativeVideoBridgeError(
                "OUTPUT_EXISTS", f"output appeared during publication: {output}"
            ) from exc
        published = True
        return QwenNativeVideoBridgeResult(
            output_directory=output,
            manifest_path=output / QWEN_NATIVE_VIDEO_BRIDGE_MANIFEST_FILENAME,
            manifest=manifest,
            dry_run=False,
        )
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _load_profile(
    mapping_config: str | Path | None,
    mapping_profile: TopicMappingProfile | None,
) -> TopicMappingProfile:
    if mapping_profile is not None:
        if not isinstance(mapping_profile, TopicMappingProfile):
            raise QwenNativeVideoBridgeError(
                "INVALID_CAMERA_MAPPING", "mapping_profile has an unsupported type"
            )
        return mapping_profile
    config = _DEFAULT_MAPPING_CONFIG if mapping_config is None else Path(mapping_config)
    try:
        return TopicMappingProfile.load(config)
    except Exception as exc:
        code = getattr(getattr(exc, "code", None), "value", "INVALID_CAMERA_MAPPING")
        raise QwenNativeVideoBridgeError(str(code), str(exc)) from exc


def _authorize_profile(profile: TopicMappingProfile, allow_unapproved: bool) -> None:
    if not profile.approved and not allow_unapproved:
        raise QwenNativeVideoBridgeError(
            "INVALID_CAMERA_MAPPING",
            f"mapping profile {profile.profile_id!r} is not approved; "
            "pass the explicit local override",
        )


def _resolve_channels(
    channels: Sequence[ChannelInspection],
    profile: TopicMappingProfile,
) -> dict[str, ChannelInspection]:
    configured_topics = tuple(
        profile.topics[CameraId(camera_id)] for camera_id in QWEN_NATIVE_VIDEO_BRIDGE_CAMERA_IDS
    )
    duplicate_topics = sorted(
        {topic for topic in configured_topics if configured_topics.count(topic) > 1}
    )
    if duplicate_topics:
        raise QwenNativeVideoBridgeError(
            "INVALID_CAMERA_MAPPING",
            f"mapping profile assigns topics more than once: {duplicate_topics!r}",
        )
    by_topic: dict[str, list[ChannelInspection]] = {}
    for channel in channels:
        by_topic.setdefault(channel.topic, []).append(channel)
    selected: dict[str, ChannelInspection] = {}
    errors: list[str] = []
    for camera_id in QWEN_NATIVE_VIDEO_BRIDGE_CAMERA_IDS:
        topic = profile.topics[CameraId(camera_id)]
        matches = by_topic.get(topic, [])
        if len(matches) != 1:
            errors.append(f"{camera_id}: expected one mapped topic {topic!r}, found {len(matches)}")
            continue
        channel = matches[0]
        if channel.schema_name != COMPRESSED_IMAGE_SCHEMA:
            errors.append(
                f"{camera_id}: topic {topic!r} has schema {channel.schema_name!r}, "
                f"expected {COMPRESSED_IMAGE_SCHEMA!r}"
            )
            continue
        if (channel.codec or "").strip().lower() != "h264":
            errors.append(f"{camera_id}: topic {topic!r} is not declared H.264")
            continue
        if channel.message_count <= 0:
            errors.append(f"{camera_id}: topic {topic!r} has no messages")
            continue
        if not channel.monotonic:
            errors.append(f"{camera_id}: topic {topic!r} has nonmonotonic log times")
            continue
        selected[camera_id] = channel
    if errors:
        raise QwenNativeVideoBridgeError("INVALID_CAMERA_MAPPING", "; ".join(errors))
    return selected


def _planned_camera_row(camera_id: str, channel: ChannelInspection) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "topic": channel.topic,
        "channel_id": channel.channel_id,
        "video": f"{camera_id}.mp4",
        "timestamps": f"{camera_id}.timestamps.jsonl",
        "input_message_count": channel.message_count,
        "source_first_log_time_ns": channel.first_message_time_ns,
        "source_last_log_time_ns": channel.last_message_time_ns,
        "schema": channel.schema_name,
        "codec": channel.codec,
    }


def _materialized_camera_row(
    camera_id: str,
    channel: ChannelInspection,
    facts: Any,
    video_path: Path,
    timestamps_path: Path,
) -> dict[str, Any]:
    row = _planned_camera_row(camera_id, channel)
    row.update(
        {
            "exported_packet_count": _optional_int(facts, "exported_packet_count"),
            "exported_frame_count": _optional_int(facts, "decoded_frame_count"),
            "keyframe_count": _optional_int(facts, "keyframe_count"),
            "leading_dropped_message_count": _optional_int(
                facts, "leading_access_unit_count", default=0
            ),
            "trailing_dropped_message_count": _optional_int(
                facts, "trailing_access_unit_count", default=0
            ),
            "width": _optional_int(facts, "width"),
            "height": _optional_int(facts, "height"),
            "export_first_source_log_time_ns": _optional_int(
                facts, "export_first_source_log_time_ns"
            ),
            "export_last_source_log_time_ns": _optional_int(
                facts, "export_last_source_log_time_ns"
            ),
            "first_pts_ns": _optional_int(facts, "first_pts_ns"),
            "last_pts_ns": _optional_int(facts, "last_pts_ns"),
            "duration_ns": _optional_int(facts, "duration_ns"),
            "time_base_numerator": _optional_int(facts, "time_base_numerator"),
            "time_base_denominator": _optional_int(facts, "time_base_denominator"),
            "tail_duration_ns": _optional_int(facts, "tail_duration_ns"),
            "sidecar_row_count": _optional_int(
                facts,
                "sidecar_row_count",
                default=_optional_int(facts, "exported_packet_count", default=0),
            ),
            "video": video_path.name,
            "timestamps": timestamps_path.name,
        }
    )
    return row


def _optional_int(value: Any, field: str, *, default: int | None = None) -> int | None:
    raw = (
        value.get(field, default) if isinstance(value, Mapping) else getattr(value, field, default)
    )
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise QwenNativeVideoBridgeError(
            "EXPORT_FACTS_INVALID", f"export facts field {field!r} must be an integer"
        )
    return raw


def _require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise QwenNativeVideoBridgeError(
            "EXPORT_FACTS_INVALID", f"{label} is not a regular non-symlink file: {path}"
        )
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise QwenNativeVideoBridgeError(
            "EXPORT_FACTS_INVALID", f"{label} cannot be stat'ed: {path}: {exc}"
        ) from exc
    if size_bytes <= 0:
        raise QwenNativeVideoBridgeError(
            "EXPORT_FACTS_INVALID", f"{label} must be non-empty: {path}"
        )


def _schema_name(schema: Any) -> str | None:
    name = getattr(schema, "name", None)
    return name if isinstance(name, str) else None


def _schema_encoding(schema: Any) -> str | None:
    encoding = getattr(schema, "encoding", None)
    return encoding if isinstance(encoding, str) else None


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QwenNativeVideoBridgeError("CORRUPT_MCAP", f"{field} must be non-empty text")
    return value


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QwenNativeVideoBridgeError("CORRUPT_MCAP", f"{field} must be an integer")
    return cast(int, value)


__all__ = [
    "QWEN_NATIVE_VIDEO_BRIDGE_AUTHORITY",
    "QWEN_NATIVE_VIDEO_BRIDGE_FORMAT",
    "QWEN_NATIVE_VIDEO_BRIDGE_MANIFEST_FILENAME",
    "QWEN_NATIVE_VIDEO_BRIDGE_VERSION",
    "DigestFreeCameraVideoExporter",
    "NoDigestPyAvH264Mp4Exporter",
    "QwenNativeVideoBridgeError",
    "QwenNativeVideoBridgeResult",
    "build_qwen_native_video_plan",
    "inspect_mcap_without_digests",
    "materialize_qwen_native_video_inputs",
]
