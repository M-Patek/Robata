"""One ordered MCAP traversal feeding bounded planning and six H.264 branches.

The existing offline exporter owns a source-reading API.  This adapter is the
pre-EOS boundary: it decodes each selected protobuf envelope once, writes the
complete access-unit facts to exactly one camera branch, and gives the same
payload object to the bounded planner.  Branches are synchronous by design, so a
caller can put a bounded queue behind the protocol without this layer retaining
an unbounded packet collection.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, Protocol, cast

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from robata.adapters.mcap_inspector import (
    McapInspectionAccumulator,
    McapPreflight,
    OfficialMcapInspector,
)
from robata.application.canonical.bounded_media import (
    ACCESS_UNIT_FRAMING_VERSION,
    BoundedSinglePassMediaPlanner,
    EncodedMediaPacket,
    PacketReference,
    PlannerFinish,
    SinglePassPlanningSink,
)
from robata.contracts import CAMERA_IDS, CameraId, SixCameraMap, canonical_json_bytes
from robata.ports import COMPRESSED_IMAGE_SCHEMA, ChannelInspection, McapInspection

_SPOOL_MAGIC = b"ROBATA-H264-SPOOL-V1\n"
_MAX_METADATA_BYTES = 64 * 1024
_MAX_BOOTSTRAP_ENVELOPES = 256
_MAX_BOOTSTRAP_PAYLOAD_BYTES = 32 * 1024 * 1024
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
PLANNING_MODE_LIVE_INDEXED: Final = "LIVE_INDEXED"
PLANNING_MODE_LIVE_BOOTSTRAP: Final = "LIVE_BOOTSTRAP"
PLANNING_MODE_EOS_SPOOL_REPLAY: Final = "EOS_SPOOL_REPLAY"


class SinglePassMcapError(RuntimeError):
    """The selected H.264 source cannot be traversed as an ordered stream."""


@dataclass(frozen=True, slots=True)
class H264PacketEnvelope:
    """Planner packet plus exact metadata needed by an MP4/sidecar branch."""

    packet: EncodedMediaPacket
    source_publish_time_ns: int
    embedded_header_time_ns: int
    nal_types: tuple[int, ...]

    @property
    def camera_id(self) -> CameraId:
        return self.packet.camera_id

    @property
    def is_keyframe(self) -> bool:
        return self.packet.is_keyframe


class H264PacketBranch(Protocol):
    """One camera's incremental export/spool input."""

    camera_id: CameraId

    def append_access_unit(
        self,
        envelope: H264PacketEnvelope,
        reference: PacketReference,
        *,
        framing_version: str,
    ) -> None:
        """Consume one access unit before its planning emission is persisted."""

    def seal(self) -> None:
        """Flush the branch after source EOS."""

    def abort(self) -> None:
        """Release branch resources after a failed traversal."""


@dataclass(frozen=True, slots=True)
class H264SpoolFacts:
    camera_id: CameraId
    path: Path
    packet_count: int
    size_bytes: int
    sha256: str


class AppendOnlyH264SpoolBranch:
    """Streaming, deterministic packet spool for one future MP4 branch."""

    def __init__(self, camera_id: CameraId, path: Path) -> None:
        self.camera_id = camera_id
        self.path = Path(path)
        self._stream = self.path.open("xb")
        self._digest = hashlib.sha256()
        self._size_bytes = 0
        self._packet_count = 0
        self._next_source_order = 0
        self._sealed = False
        self._facts: H264SpoolFacts | None = None
        self._write(_SPOOL_MAGIC)

    def append_access_unit(
        self,
        envelope: H264PacketEnvelope,
        reference: PacketReference,
        *,
        framing_version: str,
    ) -> None:
        if self._sealed:
            raise RuntimeError("cannot append to a sealed H.264 spool")
        packet = envelope.packet
        if packet.camera_id is not self.camera_id:
            raise SinglePassMcapError("packet was routed to the wrong camera branch")
        if packet.source_order != self._next_source_order:
            raise SinglePassMcapError("camera spool source_order is not contiguous")
        if reference != packet.reference():
            raise SinglePassMcapError("camera spool reference differs from the packet")
        if framing_version != ACCESS_UNIT_FRAMING_VERSION:
            raise SinglePassMcapError("camera spool received an unsupported framing version")

        metadata = canonical_json_bytes(_spool_metadata(envelope, framing_version))
        if len(metadata) > _MAX_METADATA_BYTES:
            raise SinglePassMcapError("camera spool metadata exceeds its fixed bound")
        self._write(len(metadata).to_bytes(4, "big"))
        self._write(len(packet.payload).to_bytes(8, "big"))
        self._write(metadata)
        self._write(packet.payload)
        self._packet_count += 1
        self._next_source_order += 1

    def seal(self) -> None:
        if self._sealed:
            return
        self._stream.flush()
        self._stream.close()
        self._sealed = True
        self._facts = H264SpoolFacts(
            camera_id=self.camera_id,
            path=self.path,
            packet_count=self._packet_count,
            size_bytes=self._size_bytes,
            sha256=self._digest.hexdigest(),
        )

    def abort(self) -> None:
        if not self._stream.closed:
            self._stream.close()
        self._sealed = True

    @property
    def facts(self) -> H264SpoolFacts:
        if self._facts is None:
            raise RuntimeError("H.264 spool facts are available only after seal")
        return self._facts

    def _write(self, value: bytes) -> None:
        self._stream.write(value)
        self._digest.update(value)
        self._size_bytes += len(value)


@dataclass(frozen=True, slots=True)
class SinglePassTraversalResult:
    source_size_bytes: int
    source_sha256: str
    source_message_count: int
    selected_packet_count: int
    camera_packet_counts: SixCameraMap[int]
    final_end_ns: int
    planner_finish: PlannerFinish | None
    inspection: McapInspection
    planning_mode: str


class McapSinglePassH264Tee:
    """Traverse physical MCAP order once and tee mapped H.264 access units."""

    def __init__(self, *, validate_crcs: bool = True) -> None:
        self._validate_crcs = validate_crcs

    def traverse(
        self,
        source: Path,
        channels: SixCameraMap[ChannelInspection],
        planner: BoundedSinglePassMediaPlanner | None,
        branches: Mapping[CameraId, H264PacketBranch],
        *,
        align_timestamp: Callable[[CameraId, int], int] | None = None,
        planning_sink: SinglePassPlanningSink | None = None,
        final_end_ns: int | None = None,
        preflight: McapPreflight | None = None,
        expected_source_sha256: str | None = None,
        planner_source_scope_digest: str | None = None,
        bootstrap_planner_factory: (Callable[[int], BoundedSinglePassMediaPlanner] | None) = None,
    ) -> SinglePassTraversalResult:
        source = Path(source)
        if not source.is_file():
            raise SinglePassMcapError(f"MCAP source is not a file: {source}")
        channel_map = _camera_by_channel(channels)
        branch_map = _validate_branches(branches)
        align = align_timestamp or _identity_alignment
        source_orders = {camera_id: 0 for camera_id in CAMERA_IDS}
        source_message_count = 0
        selected_packet_count = 0
        max_aligned_ns: int | None = None
        completed = False
        if planner is not None and bootstrap_planner_factory is not None:
            raise SinglePassMcapError(
                "live planner and bootstrap planner factory are mutually exclusive"
            )
        if planner is None and bootstrap_planner_factory is None and planning_sink is not None:
            raise SinglePassMcapError("planning sink requires a live planner during MCAP traversal")
        resolved_preflight = preflight or OfficialMcapInspector(
            validate_crcs=self._validate_crcs
        ).preflight(source)
        if resolved_preflight.source.resolve() != source.resolve():
            raise SinglePassMcapError("MCAP preflight source differs from the traversal source")
        if bootstrap_planner_factory is not None and resolved_preflight.message_indexes_complete:
            raise SinglePassMcapError(
                "bootstrap planning requires an incomplete-index MCAP preflight"
            )
        if expected_source_sha256 is None:
            raise SinglePassMcapError("capture-only traversal requires an expected source digest")
        resolved_expected_digest = expected_source_sha256
        resolved_planner_scope = planner_source_scope_digest
        if planner is not None:
            if resolved_planner_scope is None:
                resolved_planner_scope = planner.policy.source_scope_digest
            elif planner.policy.source_scope_digest != resolved_planner_scope:
                raise SinglePassMcapError(
                    "live planner policy differs from the declared planner source scope"
                )
        elif bootstrap_planner_factory is not None and resolved_planner_scope is None:
            raise SinglePassMcapError("bootstrap planning requires a planner source scope digest")
        inspection_accumulator = McapInspectionAccumulator(
            resolved_preflight,
            expected_source_sha256=resolved_expected_digest,
        )
        active_planner = planner
        planning_mode = (
            PLANNING_MODE_LIVE_INDEXED
            if active_planner is not None
            else PLANNING_MODE_EOS_SPOOL_REPLAY
        )
        bootstrap_envelopes: list[H264PacketEnvelope] = []
        bootstrap_payload_bytes = 0
        bootstrap_first_times: dict[CameraId, int] = {}
        bootstrap_last_order: dict[CameraId, tuple[int, int, int]] = {}
        bootstrap_fallback = False

        try:
            with source.open("rb") as raw_stream:
                stream = _HashingNonSeekingStream(raw_stream)
                reader = make_reader(
                    cast(BinaryIO, stream),
                    validate_crcs=self._validate_crcs,
                )
                decoder_factory = DecoderFactory()
                for schema, channel, message in reader.iter_messages(log_time_order=False):
                    traversal_index = source_message_count
                    source_message_count += 1
                    camera_id = channel_map.get(channel.id)
                    if camera_id is None:
                        inspection_accumulator.observe(schema, channel, message)
                        continue
                    expected = channels[camera_id]
                    if channel.topic != expected.topic or channel.id != expected.channel_id:
                        raise SinglePassMcapError(
                            "mapped channel identity changed during traversal"
                        )
                    decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
                    if decoder is None:
                        raise SinglePassMcapError("mapped camera channel has no protobuf decoder")
                    decoded = decoder(message.data)
                    inspection_accumulator.observe(
                        schema,
                        channel,
                        message,
                        decoded=decoded,
                    )
                    envelope = _envelope(
                        traversal_index=traversal_index,
                        camera_id=camera_id,
                        channel_id=expected.channel_id,
                        source_order=source_orders[camera_id],
                        schema=schema,
                        message=message,
                        decoded=decoded,
                        align_timestamp=align,
                    )
                    packet = envelope.packet
                    if active_planner is not None:
                        emission = active_planner.push(packet)
                        branch_map[camera_id].append_access_unit(
                            envelope,
                            packet.reference(),
                            framing_version=ACCESS_UNIT_FRAMING_VERSION,
                        )
                        if planning_sink is not None:
                            planning_sink.append_emission(emission)
                    else:
                        previous = bootstrap_last_order.get(camera_id)
                        current = (
                            packet.source_sequence,
                            packet.source_timestamp_ns,
                            packet.aligned_timestamp_ns,
                        )
                        if previous is not None and any(
                            current[index] <= previous[index] for index in range(3)
                        ):
                            raise SinglePassMcapError(
                                f"{camera_id.value} bootstrap packet order is not monotonic"
                            )
                        bootstrap_last_order[camera_id] = current
                        branch_map[camera_id].append_access_unit(
                            envelope,
                            packet.reference(),
                            framing_version=ACCESS_UNIT_FRAMING_VERSION,
                        )
                        if bootstrap_planner_factory is not None and not bootstrap_fallback:
                            bootstrap_envelopes.append(envelope)
                            bootstrap_payload_bytes += packet.payload_bytes
                            bootstrap_first_times.setdefault(
                                camera_id,
                                packet.source_timestamp_ns,
                            )
                            if len(bootstrap_first_times) == len(CAMERA_IDS):
                                source_origin_ns = min(bootstrap_first_times.values())
                                active_planner = bootstrap_planner_factory(source_origin_ns)
                                if (
                                    active_planner.policy.source_origin_ns != source_origin_ns
                                    or active_planner.policy.source_scope_digest
                                    != resolved_planner_scope
                                ):
                                    raise SinglePassMcapError(
                                        "bootstrap planner policy differs from planning identity"
                                    )
                                for buffered in bootstrap_envelopes:
                                    buffered_emission = active_planner.push(buffered.packet)
                                    if planning_sink is not None:
                                        planning_sink.append_emission(buffered_emission)
                                bootstrap_envelopes.clear()
                                bootstrap_payload_bytes = 0
                                planning_mode = PLANNING_MODE_LIVE_BOOTSTRAP
                            elif (
                                len(bootstrap_envelopes) > _MAX_BOOTSTRAP_ENVELOPES
                                or bootstrap_payload_bytes > _MAX_BOOTSTRAP_PAYLOAD_BYTES
                            ):
                                bootstrap_envelopes.clear()
                                bootstrap_payload_bytes = 0
                                bootstrap_fallback = True
                    source_orders[camera_id] += 1
                    selected_packet_count += 1
                    max_aligned_ns = (
                        packet.aligned_timestamp_ns
                        if max_aligned_ns is None
                        else max(max_aligned_ns, packet.aligned_timestamp_ns)
                    )

                # Cover any reader-specific footer tail with the same source handle.
                stream.drain()

            inspection = inspection_accumulator.finish(
                source_size_bytes=stream.size_bytes,
                source_sha256=stream.sha256,
            )

            if max_aligned_ns is None:
                raise SinglePassMcapError("MCAP contains no messages for the six mapped channels")
            resolved_end_ns = max_aligned_ns + 1 if final_end_ns is None else final_end_ns
            finish = active_planner.finish(resolved_end_ns) if active_planner is not None else None
            if planning_sink is not None and finish is not None:
                planning_sink.seal(finish)
            for camera_id in CAMERA_IDS:
                branch_map[camera_id].seal()
            completed = True
            return SinglePassTraversalResult(
                source_size_bytes=stream.size_bytes,
                source_sha256=stream.sha256,
                source_message_count=source_message_count,
                selected_packet_count=selected_packet_count,
                camera_packet_counts=SixCameraMap[int].model_validate(source_orders, strict=True),
                final_end_ns=resolved_end_ns,
                planner_finish=finish,
                inspection=inspection,
                planning_mode=planning_mode,
            )
        except SinglePassMcapError:
            raise
        except Exception as exc:
            raise SinglePassMcapError(
                f"single-pass MCAP traversal failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if not completed:
                for camera_id in CAMERA_IDS:
                    branch_map[camera_id].abort()


def iter_h264_spool(path: Path) -> Iterator[H264PacketEnvelope]:
    """Read one deterministic spool record at a time."""

    with Path(path).open("rb") as stream:
        if stream.read(len(_SPOOL_MAGIC)) != _SPOOL_MAGIC:
            raise SinglePassMcapError("H.264 spool magic is invalid")
        while first := stream.read(1):
            metadata_size = int.from_bytes(first + _read_exact(stream, 3), "big")
            if metadata_size <= 0 or metadata_size > _MAX_METADATA_BYTES:
                raise SinglePassMcapError("H.264 spool metadata size is invalid")
            payload_size = int.from_bytes(_read_exact(stream, 8), "big")
            if payload_size <= 0:
                raise SinglePassMcapError("H.264 spool payload size is invalid")
            metadata = json.loads(_read_exact(stream, metadata_size))
            payload = _read_exact(stream, payload_size)
            yield _envelope_from_spool(metadata, payload)


def _camera_by_channel(
    channels: SixCameraMap[ChannelInspection],
) -> dict[int, CameraId]:
    result: dict[int, CameraId] = {}
    for camera_id in CAMERA_IDS:
        channel = channels[camera_id]
        if channel.channel_id in result:
            raise SinglePassMcapError("mapped camera channel IDs must be unique")
        result[channel.channel_id] = camera_id
    return result


def _validate_branches(
    branches: Mapping[CameraId, H264PacketBranch],
) -> dict[CameraId, H264PacketBranch]:
    if frozenset(branches) != frozenset(CAMERA_IDS):
        raise SinglePassMcapError("exactly one branch is required for each canonical camera")
    result: dict[CameraId, H264PacketBranch] = {}
    for camera_id in CAMERA_IDS:
        branch = branches[camera_id]
        if branch.camera_id is not camera_id:
            raise SinglePassMcapError("camera branch identity differs from its map key")
        result[camera_id] = branch
    return result


def _identity_alignment(_camera_id: CameraId, timestamp_ns: int) -> int:
    return timestamp_ns


def _envelope(
    *,
    traversal_index: int,
    camera_id: CameraId,
    channel_id: int,
    source_order: int,
    schema: Any,
    message: Any,
    decoded: Any,
    align_timestamp: Callable[[CameraId, int], int],
) -> H264PacketEnvelope:
    if schema is None or schema.name != COMPRESSED_IMAGE_SCHEMA:
        raise SinglePassMcapError("mapped camera schema changed during traversal")
    codec = getattr(decoded, "format", None)
    if not isinstance(codec, str) or codec.strip().lower() != "h264":
        raise SinglePassMcapError("mapped camera payload is not declared as H.264")
    payload = getattr(decoded, "data", None)
    if not isinstance(payload, bytes) or not payload:
        raise SinglePassMcapError("mapped camera payload is not non-empty bytes")
    nal_types = _annex_b_nal_types(payload)
    if not nal_types:
        raise SinglePassMcapError("mapped H.264 payload is not Annex-B framed")
    log_time_ns = _exact_int(message.log_time, "source log time")
    publish_time_ns = _exact_int(message.publish_time, "source publish time")
    source_sequence = _exact_int(message.sequence, "source sequence")
    embedded_header_time_ns = _embedded_header_time_ns(decoded)
    aligned_timestamp_ns = align_timestamp(camera_id, log_time_ns)
    if isinstance(aligned_timestamp_ns, bool) or not isinstance(aligned_timestamp_ns, int):
        raise SinglePassMcapError("alignment callback must return an integer timestamp")
    packet = EncodedMediaPacket(
        traversal_index=traversal_index,
        camera_id=camera_id,
        source_order=source_order,
        source_sequence=source_sequence,
        source_timestamp_ns=log_time_ns,
        aligned_timestamp_ns=aligned_timestamp_ns,
        source_locator=f"mcap://channel/{channel_id}/packet/{source_order}",
        payload=payload,
        is_keyframe=5 in nal_types,
    )
    return H264PacketEnvelope(
        packet=packet,
        source_publish_time_ns=publish_time_ns,
        embedded_header_time_ns=embedded_header_time_ns,
        nal_types=nal_types,
    )


def _spool_metadata(envelope: H264PacketEnvelope, framing_version: str) -> dict[str, object]:
    packet = envelope.packet
    return {
        "aligned_timestamp_ns": str(packet.aligned_timestamp_ns),
        "camera_id": packet.camera_id.value,
        "embedded_header_time_ns": str(envelope.embedded_header_time_ns),
        "framing_version": framing_version,
        "is_keyframe": packet.is_keyframe,
        "nal_types": list(envelope.nal_types),
        "source_locator": packet.source_locator,
        "source_log_time_ns": str(packet.source_timestamp_ns),
        "source_order": packet.source_order,
        "source_publish_time_ns": str(envelope.source_publish_time_ns),
        "source_sequence": packet.source_sequence,
        "traversal_index": packet.traversal_index,
    }


def _envelope_from_spool(metadata: Any, payload: bytes) -> H264PacketEnvelope:
    if not isinstance(metadata, dict):
        raise SinglePassMcapError("H.264 spool metadata is not an object")
    try:
        if metadata["framing_version"] != ACCESS_UNIT_FRAMING_VERSION:
            raise SinglePassMcapError("H.264 spool framing version is unsupported")
        nal_types_raw = metadata["nal_types"]
        if not isinstance(nal_types_raw, list) or not all(
            type(value) is int for value in nal_types_raw
        ):
            raise SinglePassMcapError("H.264 spool NAL types are invalid")
        actual_nal_types = _annex_b_nal_types(payload)
        if not nal_types_raw or any(value < 0 or value > 31 for value in nal_types_raw):
            raise SinglePassMcapError("H.264 spool NAL types are out of range")
        if tuple(nal_types_raw) != actual_nal_types:
            raise SinglePassMcapError("H.264 spool NAL types differ from the access-unit payload")
        is_keyframe = metadata["is_keyframe"]
        if type(is_keyframe) is not bool or is_keyframe != (5 in actual_nal_types):
            raise SinglePassMcapError(
                "H.264 spool keyframe flag differs from the access-unit payload"
            )
        source_locator = metadata["source_locator"]
        if not isinstance(source_locator, str) or not source_locator:
            raise SinglePassMcapError("H.264 spool source locator is invalid")
        packet = EncodedMediaPacket(
            traversal_index=_json_int(metadata["traversal_index"]),
            camera_id=CameraId(metadata["camera_id"]),
            source_order=_json_int(metadata["source_order"]),
            source_sequence=_json_int(metadata["source_sequence"]),
            source_timestamp_ns=_json_int(metadata["source_log_time_ns"]),
            aligned_timestamp_ns=_json_int(metadata["aligned_timestamp_ns"]),
            source_locator=str(metadata["source_locator"]),
            payload=payload,
            is_keyframe=metadata["is_keyframe"] is True,
        )
        return H264PacketEnvelope(
            packet=packet,
            source_publish_time_ns=_json_int(metadata["source_publish_time_ns"]),
            embedded_header_time_ns=_json_int(metadata["embedded_header_time_ns"]),
            nal_types=tuple(nal_types_raw),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SinglePassMcapError(f"H.264 spool metadata is invalid: {exc}") from exc


def _json_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError("value is not an integer")


def _exact_int(value: object, field: str) -> int:
    if type(value) is not int or not _INT64_MIN <= value <= _INT64_MAX:
        raise SinglePassMcapError(f"{field} is not a signed 64-bit integer")
    return value


def _embedded_header_time_ns(decoded: Any) -> int:
    has_field = getattr(decoded, "HasField", None)
    header_present = True
    if callable(has_field):
        try:
            header_present = bool(has_field("header"))
        except ValueError:
            header_present = False
    header = getattr(decoded, "header", None)
    value = getattr(header, "timestamp", None) if header_present else None
    return _exact_int(value, "embedded header time")


def _annex_b_nal_types(payload: bytes) -> tuple[int, ...]:
    starts: list[tuple[int, int]] = []
    index = 0
    while index <= len(payload) - 3:
        if payload[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
        elif payload[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
        else:
            index += 1
    result: list[int] = []
    for position, (start, prefix_length) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(payload)
        header = start + prefix_length
        if header < end:
            result.append(payload[header] & 0x1F)
    return tuple(result)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise SinglePassMcapError("H.264 spool record is truncated")
    return value


class _HashingNonSeekingStream(io.BufferedIOBase):
    """Sequential file view that computes exact identity during MCAP parsing."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self._size_bytes = 0

    def read(self, size: int | None = -1) -> bytes:
        value = self._stream.read(-1 if size is None else size)
        self._digest.update(value)
        self._size_bytes += len(value)
        return value

    def drain(self) -> None:
        while self.read(1024 * 1024):
            pass

    @staticmethod
    def seekable() -> bool:
        return False

    @staticmethod
    def readable() -> bool:
        return True

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


__all__ = [
    "PLANNING_MODE_EOS_SPOOL_REPLAY",
    "PLANNING_MODE_LIVE_BOOTSTRAP",
    "PLANNING_MODE_LIVE_INDEXED",
    "AppendOnlyH264SpoolBranch",
    "H264PacketBranch",
    "H264PacketEnvelope",
    "H264SpoolFacts",
    "McapSinglePassH264Tee",
    "SinglePassMcapError",
    "SinglePassTraversalResult",
    "iter_h264_spool",
]
