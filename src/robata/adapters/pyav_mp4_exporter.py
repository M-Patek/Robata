"""Direct H.264 MCAP-to-MP4 remux with an exact timestamp ledger."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Final

import av
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from robata.adapters.mcap_single_pass import H264PacketEnvelope
from robata.application.canonical.bounded_media import (
    ACCESS_UNIT_FRAMING_VERSION,
    PacketReference,
)
from robata.contracts import CameraId, Sha256Digest, canonical_json_bytes
from robata.ports import (
    COMPRESSED_IMAGE_SCHEMA,
    ChannelInspection,
    ExportedCameraVideoFacts,
    VideoExportError,
    VideoExportErrorCode,
)
from robata.tempfiles import make_temp_file

DecodedFrameObserver = Callable[[H264PacketEnvelope, Any, int], None]

_NANOSECOND_TIME_BASE = Fraction(1, 1_000_000_000)
_TAIL_DURATION_POLICY = "MEDIAN_POSITIVE_INTERVAL"
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_INTERVAL_RECORD = struct.Struct("<q")
_INTERVAL_SCAN_SIZE = 64 * 1024

EXPORTER_NAME: Final = "robata.pyav_h264_mp4_exporter"
EXPORTER_VERSION: Final = "0.1.0"
EXPORT_PROFILE_ID: Final = "direct-h264-remux-no-reordering"
EXPORT_PROFILE_VERSION: Final = "1.0"
EXPORT_CONFIG: Final[Mapping[str, str | int | bool]] = MappingProxyType(
    {
        "bootstrap_policy": "SPS_PPS_BEFORE_IDR",
        "codec": "h264",
        "frame_reordering_policy": "REJECT",
        "mcap_crc_validation": True,
        "median_rounding": "HALF_EVEN",
        "mp4_time_base_denominator": 1_000_000_000,
        "mp4_time_base_numerator": 1,
        "schema": COMPRESSED_IMAGE_SCHEMA,
        "tail_duration_policy": _TAIL_DURATION_POLICY,
        "timestamp_origin": "FIRST_EXPORTED_SOURCE_LOG_TIME",
    }
)


@dataclass(frozen=True, slots=True)
class _AccessUnit:
    log_time_ns: int
    publish_time_ns: int
    embedded_header_time_ns: int
    source_sequence: int
    payload: bytes
    nal_types: tuple[int, ...]

    @property
    def is_keyframe(self) -> bool:
        return 5 in self.nal_types


@dataclass(frozen=True, slots=True)
class _TemporaryExportFacts:
    source_message_count: int
    leading_access_unit_count: int
    exported_packet_count: int
    decoded_frame_count: int
    keyframe_count: int
    width: int
    height: int
    source_first_log_time_ns: int
    source_last_log_time_ns: int
    leading_first_log_time_ns: int | None
    leading_last_log_time_ns: int | None
    export_first_source_log_time_ns: int
    export_last_source_log_time_ns: int
    first_pts_ns: int
    last_pts_ns: int
    duration_ns: int
    tail_duration_ns: int


@dataclass(frozen=True, slots=True)
class _SidecarExpectation:
    pts_ns: int
    duration_ns: int
    is_keyframe: bool


class _Int64IntervalSpool:
    """Fixed-width interval ledger with exact bounded-memory median selection."""

    __slots__ = ("_count", "_maximum", "_minimum", "_stream")

    def __init__(self, path: Path) -> None:
        self._stream: BinaryIO | None = path.open("r+b")
        self._count = 0
        self._minimum: int | None = None
        self._maximum: int | None = None

    @property
    def count(self) -> int:
        return self._count

    @property
    def minimum(self) -> int | None:
        return self._minimum

    @property
    def maximum(self) -> int | None:
        return self._maximum

    def append(self, value: int) -> None:
        if type(value) is not int or not 0 < value <= _INT64_MAX:
            raise ValueError("packet interval must be a positive signed int64")
        stream = self._require_open()
        if stream.write(_INTERVAL_RECORD.pack(value)) != _INTERVAL_RECORD.size:
            raise OSError("could not write the complete packet interval")
        self._count += 1
        self._minimum = value if self._minimum is None else min(self._minimum, value)
        self._maximum = value if self._maximum is None else max(self._maximum, value)

    def median_half_even(self) -> int:
        if self._count == 0 or self._minimum is None or self._maximum is None:
            raise ValueError("cannot compute a median without packet intervals")
        self._require_open().flush()
        upper_rank = self._count // 2
        upper = self._select_rank(upper_rank, self._minimum, self._maximum)
        if self._count % 2:
            return upper
        lower = self._select_rank(upper_rank - 1, self._minimum, upper)
        total = lower + upper
        quotient, remainder = divmod(total, 2)
        return quotient + int(remainder == 1 and quotient % 2 == 1)

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.close()

    def _select_rank(self, rank: int, lower: int, upper: int) -> int:
        while lower < upper:
            midpoint = lower + (upper - lower) // 2
            if self._count_less_equal(midpoint) > rank:
                upper = midpoint
            else:
                lower = midpoint + 1
        return lower

    def _count_less_equal(self, threshold: int) -> int:
        stream = self._require_open()
        stream.seek(0)
        seen = 0
        count = 0
        while chunk := stream.read(_INTERVAL_SCAN_SIZE):
            if len(chunk) % _INTERVAL_RECORD.size:
                raise OSError("packet interval spool contains a truncated record")
            seen += len(chunk) // _INTERVAL_RECORD.size
            count += sum(value <= threshold for (value,) in _INTERVAL_RECORD.iter_unpack(chunk))
        if seen != self._count:
            raise OSError("packet interval spool record count changed unexpectedly")
        return count

    def _require_open(self) -> BinaryIO:
        if self._stream is None:
            raise RuntimeError("packet interval spool is closed")
        return self._stream


def _annex_b_nal_types(payload: bytes) -> tuple[int, ...]:
    """Return H.264 NAL unit types from a three/four-byte Annex-B byte stream."""

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

    nal_types: list[int] = []
    for position, (start, prefix_length) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(payload)
        header = start + prefix_length
        if header < end:
            nal_types.append(payload[header] & 0x1F)
    return tuple(nal_types)


def is_independent_h264_bootstrap(nal_types: tuple[int, ...]) -> bool:
    try:
        idr_index = nal_types.index(5)
    except ValueError:
        return False
    prefix = nal_types[:idr_index]
    return 7 in prefix and 8 in prefix


def _canonical_sidecar_line(
    camera_id: CameraId,
    unit: _AccessUnit,
    relative_pts_ns: int,
    duration_ns: int,
    packet_index: int,
    *,
    duration_is_estimated: bool,
) -> bytes:
    row = {
        "camera_id": camera_id.value,
        "duration_is_estimated": duration_is_estimated,
        "duration_ns": str(duration_ns),
        "embedded_header_time_ns": str(unit.embedded_header_time_ns),
        "export_profile_id": EXPORT_PROFILE_ID,
        "export_profile_version": EXPORT_PROFILE_VERSION,
        "is_keyframe": unit.is_keyframe,
        "packet_index": packet_index,
        "relative_dts_ns": str(relative_pts_ns),
        "relative_pts_ns": str(relative_pts_ns),
        "schema_version": "1.0",
        "source_log_time_ns": str(unit.log_time_ns),
        "source_publish_time_ns": str(unit.publish_time_ns),
        "source_sequence": unit.source_sequence,
        "time_base_denominator": _NANOSECOND_TIME_BASE.denominator,
        "time_base_numerator": _NANOSECOND_TIME_BASE.numerator,
    }
    return canonical_json_bytes(row) + b"\n"


class PyAvH264Mp4Exporter:
    """Export one mapped Foxglove H.264 channel without transcoding."""

    def __init__(self) -> None:
        self._validate_crcs = True

    def export(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> ExportedCameraVideoFacts:
        source = Path(source)
        video_path = Path(video_path)
        sidecar_path = Path(sidecar_path)
        session: PyAvH264RemuxSession | None = None
        try:
            self._validate_source(source)
            session = self.begin_incremental(
                camera_id,
                channel,
                video_path,
                sidecar_path,
            )
            self._stream_source_into_session(source, channel, session)
            session.seal()
            return session.facts
        except VideoExportError:
            raise
        except OSError as exc:
            raise VideoExportError(
                VideoExportErrorCode.SOURCE_IO_ERROR,
                f"camera-video export I/O failed: {exc}",
            ) from exc
        except Exception as exc:
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                f"camera-video remux failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if session is not None:
                session.abort()

    def begin_incremental(
        self,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
        *,
        decoded_frame_observer: DecodedFrameObserver | None = None,
        validate_output: bool = True,
    ) -> PyAvH264RemuxSession:
        """Create one branch-compatible remux session for an ordered camera stream."""

        video_path = Path(video_path)
        sidecar_path = Path(sidecar_path)
        if decoded_frame_observer is not None and not callable(decoded_frame_observer):
            raise TypeError("decoded_frame_observer must be callable or None")
        if not isinstance(validate_output, bool):
            raise TypeError("validate_output must be a boolean")
        if not validate_output and decoded_frame_observer is None:
            raise ValueError("validate_output=False requires a decoded_frame_observer")
        self._validate_incremental_request(camera_id, channel, video_path, sidecar_path)
        video_temp: Path | None = None
        sidecar_temp: Path | None = None
        interval_temp: Path | None = None
        session: PyAvH264RemuxSession | None = None
        try:
            video_temp = self._make_sibling_temp(video_path)
            sidecar_temp = self._make_sibling_temp(sidecar_path)
            interval_temp = self._make_sibling_temp(sidecar_path)
            session = PyAvH264RemuxSession(
                exporter=self,
                camera_id=camera_id,
                channel=channel,
                video_path=video_path,
                sidecar_path=sidecar_path,
                video_temp=video_temp,
                sidecar_temp=sidecar_temp,
                interval_temp=interval_temp,
                decoded_frame_observer=decoded_frame_observer,
                validate_output=validate_output,
            )
            return session
        except VideoExportError:
            raise
        except OSError as exc:
            raise VideoExportError(
                VideoExportErrorCode.SOURCE_IO_ERROR,
                f"camera-video export I/O failed: {exc}",
            ) from exc
        except Exception as exc:
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                f"camera-video remux initialization failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if video_temp is not None and session is None:
                self._safe_unlink(video_temp)
            if sidecar_temp is not None and session is None:
                self._safe_unlink(sidecar_temp)
            if interval_temp is not None and session is None:
                self._safe_unlink(interval_temp)

    @staticmethod
    def _validate_source(source: Path) -> None:
        if not source.exists():
            raise VideoExportError(
                VideoExportErrorCode.SOURCE_NOT_FOUND,
                f"MCAP source does not exist: {source}",
            )
        if not source.is_file():
            raise VideoExportError(
                VideoExportErrorCode.SOURCE_IO_ERROR,
                f"MCAP source is not a file: {source}",
            )

    @staticmethod
    def _validate_incremental_request(
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> None:
        if not isinstance(camera_id, CameraId):
            raise VideoExportError(
                VideoExportErrorCode.INVALID_CHANNEL,
                "camera_id must be a canonical CameraId",
            )
        if channel.schema_name != COMPRESSED_IMAGE_SCHEMA:
            raise VideoExportError(
                VideoExportErrorCode.UNSUPPORTED_SCHEMA,
                f"topic {channel.topic!r} is not a {COMPRESSED_IMAGE_SCHEMA} channel",
            )
        if (channel.codec or "").strip().lower() != "h264":
            raise VideoExportError(
                VideoExportErrorCode.UNSUPPORTED_CODEC,
                f"topic {channel.topic!r} declares unsupported codec {channel.codec!r}",
            )
        if not channel.monotonic:
            raise VideoExportError(
                VideoExportErrorCode.NONMONOTONIC_LOG_TIME,
                f"topic {channel.topic!r} was inspected as nonmonotonic",
            )
        if os.path.abspath(video_path) == os.path.abspath(sidecar_path):
            raise VideoExportError(
                VideoExportErrorCode.INVALID_DESTINATION,
                "video and sidecar destinations must be distinct",
            )
        for destination in (video_path, sidecar_path):
            if os.path.lexists(destination):
                raise VideoExportError(
                    VideoExportErrorCode.DESTINATION_EXISTS,
                    f"destination already exists: {destination}",
                )
            if not destination.parent.is_dir():
                raise VideoExportError(
                    VideoExportErrorCode.INVALID_DESTINATION,
                    f"destination parent is not an existing directory: {destination.parent}",
                )

    def _stream_source_into_session(
        self,
        source: Path,
        channel: ChannelInspection,
        session: PyAvH264RemuxSession,
    ) -> None:
        try:
            with source.open("rb") as source_stream:
                reader = make_reader(
                    source_stream,
                    validate_crcs=self._validate_crcs,
                    decoder_factories=[DecoderFactory()],
                )
                messages = reader.iter_decoded_messages(
                    topics=[channel.topic],
                    log_time_order=False,
                )
                for schema, observed_channel, message, decoded in messages:
                    if observed_channel.id != channel.channel_id:
                        continue
                    if schema is None or schema.name != COMPRESSED_IMAGE_SCHEMA:
                        raise VideoExportError(
                            VideoExportErrorCode.UNSUPPORTED_SCHEMA,
                            "mapped camera schema changed while reading the MCAP",
                        )
                    session._append_unit(self._access_unit(decoded, message))
        except VideoExportError:
            raise
        except OSError:
            raise
        except Exception as exc:
            raise VideoExportError(
                VideoExportErrorCode.MCAP_READ_ERROR,
                f"could not read mapped camera messages: {type(exc).__name__}: {exc}",
            ) from exc

    @classmethod
    def _access_unit(cls, decoded: Any, message: Any) -> _AccessUnit:
        codec = getattr(decoded, "format", None)
        if not isinstance(codec, str) or codec.strip().lower() != "h264":
            raise VideoExportError(
                VideoExportErrorCode.UNSUPPORTED_CODEC,
                f"CompressedImage message declares unsupported format {codec!r}",
            )
        payload = getattr(decoded, "data", None)
        if not isinstance(payload, bytes):
            raise VideoExportError(
                VideoExportErrorCode.INVALID_ACCESS_UNIT,
                "CompressedImage.data is not bytes",
            )
        nal_types = _annex_b_nal_types(payload)
        if not nal_types:
            raise VideoExportError(
                VideoExportErrorCode.INVALID_ACCESS_UNIT,
                "H.264 payload is not an Annex-B access unit",
            )
        return _AccessUnit(
            log_time_ns=cls._exact_int(message.log_time, "source log time"),
            publish_time_ns=cls._exact_int(message.publish_time, "source publish time"),
            embedded_header_time_ns=cls._embedded_header_time_ns(decoded),
            source_sequence=cls._exact_int(message.sequence, "source sequence"),
            payload=payload,
            nal_types=nal_types,
        )

    @staticmethod
    def _embedded_header_time_ns(decoded: Any) -> int:
        has_field = getattr(decoded, "HasField", None)
        header_present = True
        if callable(has_field):
            try:
                header_present = bool(has_field("header"))
            except ValueError:
                header_present = False
        header = getattr(decoded, "header", None)
        if header_present and header is not None:
            value = getattr(header, "timestamp", None)
            if type(value) is int and _INT64_MIN <= value <= _INT64_MAX:
                return value
        raise VideoExportError(
            VideoExportErrorCode.INVALID_TIMESTAMP_METADATA,
            "CompressedImage.header has no valid authoritative integer timestamp",
        )

    @staticmethod
    def _decode_frame_without_reordering(
        decoder: Any,
        unit: _AccessUnit,
        relative_pts_ns: int,
    ) -> Any:
        packet = av.Packet(unit.payload)
        packet.pts = relative_pts_ns
        packet.dts = relative_pts_ns
        packet.time_base = _NANOSECOND_TIME_BASE
        try:
            frames = decoder.decode(packet)
        except Exception as exc:
            raise VideoExportError(
                VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                f"H.264 access unit cannot be decoded: {type(exc).__name__}: {exc}",
            ) from exc
        if decoder.has_b_frames or len(frames) != 1:
            raise VideoExportError(
                VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                "access unit did not produce exactly one immediate frame",
            )
        frame = frames[0]
        picture_type = getattr(frame.pict_type, "name", str(frame.pict_type))
        if picture_type == "B":
            raise VideoExportError(
                VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                "B-frames are not supported by capture-time PTS/DTS remux",
            )
        if frame.pts is None or frame.time_base is None:
            raise VideoExportError(
                VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                "decoded frame has no attributable timestamp",
            )
        decoded_time = Fraction(frame.pts) * frame.time_base * 1_000_000_000
        if decoded_time.denominator != 1 or decoded_time.numerator != relative_pts_ns:
            raise VideoExportError(
                VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                "decoded frame timestamp differs from its source access unit",
            )
        return frame

    @staticmethod
    def _mux_and_record(
        output: Any,
        output_stream: Any,
        sidecar_stream: BinaryIO,
        camera_id: CameraId,
        unit: _AccessUnit,
        bootstrap_time_ns: int,
        duration_ns: int,
        packet_index: int,
        *,
        duration_is_estimated: bool,
    ) -> None:
        relative_pts_ns = unit.log_time_ns - bootstrap_time_ns
        packet = av.Packet(unit.payload)
        packet.pts = relative_pts_ns
        packet.dts = relative_pts_ns
        packet.duration = duration_ns
        packet.time_base = _NANOSECOND_TIME_BASE
        packet.is_keyframe = unit.is_keyframe
        packet.stream = output_stream
        try:
            output.mux(packet)
        except Exception as exc:
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                f"MP4 mux rejected access unit {packet_index}: {type(exc).__name__}: {exc}",
            ) from exc

        sidecar_stream.write(
            _canonical_sidecar_line(
                camera_id,
                unit,
                relative_pts_ns,
                duration_ns,
                packet_index,
                duration_is_estimated=duration_is_estimated,
            )
        )

    @staticmethod
    def _validate_exported_mp4(
        path: Path,
        sidecar_path: Path,
        facts: _TemporaryExportFacts,
    ) -> int:
        try:
            with (
                av.open(str(path), mode="r") as container,
                sidecar_path.open("rb") as sidecar_stream,
            ):
                if len(container.streams.video) != 1:
                    raise VideoExportError(
                        VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                        "exported MP4 does not contain exactly one video stream",
                    )
                stream = container.streams.video[0]
                if stream.time_base != _NANOSECOND_TIME_BASE:
                    raise VideoExportError(
                        VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                        f"exported MP4 time base changed to {stream.time_base}",
                    )
                packet_count = 0
                for packet in container.demux(stream):
                    if packet.dts is None:
                        continue
                    if packet_count >= facts.exported_packet_count:
                        raise VideoExportError(
                            VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                            "exported MP4 contains more packets than the timestamp ledger",
                        )
                    expected = PyAvH264Mp4Exporter._read_sidecar_expectation(
                        sidecar_stream,
                        packet_count,
                    )
                    if packet.pts is None or packet.dts is None or packet.time_base is None:
                        raise VideoExportError(
                            VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                            "exported packet is missing PTS/DTS/time base",
                        )
                    pts_ns = Fraction(packet.pts) * packet.time_base * 1_000_000_000
                    dts_ns = Fraction(packet.dts) * packet.time_base * 1_000_000_000
                    if (
                        pts_ns.denominator != 1
                        or dts_ns.denominator != 1
                        or pts_ns.numerator != expected.pts_ns
                        or dts_ns.numerator != expected.pts_ns
                        or packet.duration != expected.duration_ns
                        or packet.is_keyframe != expected.is_keyframe
                    ):
                        raise VideoExportError(
                            VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                            f"exported packet {packet_count} differs from its timestamp ledger",
                        )
                    packet_count += 1
                if packet_count != facts.exported_packet_count or sidecar_stream.readline():
                    raise VideoExportError(
                        VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                        "exported MP4 packet count differs from the timestamp ledger",
                    )

            decoded_count = 0
            with (
                av.open(str(path), mode="r") as container,
                sidecar_path.open("rb") as sidecar_stream,
            ):
                stream = container.streams.video[0]
                if stream.codec_context.has_b_frames:
                    raise VideoExportError(
                        VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                        "exported MP4 declares reordered frames",
                    )
                for index, frame in enumerate(container.decode(stream)):
                    if index >= facts.exported_packet_count:
                        raise VideoExportError(
                            VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                            "exported MP4 decoded more frames than source access units",
                        )
                    expected = PyAvH264Mp4Exporter._read_sidecar_expectation(
                        sidecar_stream,
                        index,
                    )
                    picture_type = getattr(frame.pict_type, "name", str(frame.pict_type))
                    if picture_type == "B" or frame.pts is None or frame.time_base is None:
                        raise VideoExportError(
                            VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                            "exported MP4 contains unattributable or reordered frames",
                        )
                    pts_ns = Fraction(frame.pts) * frame.time_base * 1_000_000_000
                    if pts_ns.denominator != 1 or pts_ns.numerator != expected.pts_ns:
                        raise VideoExportError(
                            VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                            "exported frame order differs from source access-unit order",
                        )
                    if (frame.width, frame.height) != (facts.width, facts.height):
                        raise VideoExportError(
                            VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                            "exported MP4 dimensions differ from source decode",
                        )
                    decoded_count += 1
            return decoded_count
        except VideoExportError:
            raise
        except Exception as exc:
            raise VideoExportError(
                VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                f"exported MP4 cannot be independently decoded: {type(exc).__name__}: {exc}",
            ) from exc

    @staticmethod
    def _read_sidecar_expectation(
        sidecar_stream: BinaryIO,
        expected_index: int,
    ) -> _SidecarExpectation:
        line = sidecar_stream.readline()
        if not line:
            raise VideoExportError(
                VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                "timestamp ledger ended before the exported MP4",
            )
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("row is not an object")
            packet_index = row["packet_index"]
            pts_ns = int(row["relative_pts_ns"])
            dts_ns = int(row["relative_dts_ns"])
            duration_ns = int(row["duration_ns"])
            is_keyframe = row["is_keyframe"]
            if (
                type(packet_index) is not int
                or packet_index != expected_index
                or pts_ns != dts_ns
                or duration_ns <= 0
                or type(is_keyframe) is not bool
            ):
                raise ValueError("row fields are inconsistent")
            return _SidecarExpectation(
                pts_ns=pts_ns,
                duration_ns=duration_ns,
                is_keyframe=is_keyframe,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VideoExportError(
                VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                f"timestamp ledger row {expected_index} is invalid: {exc}",
            ) from exc

    @staticmethod
    def _median_half_even(values: list[int]) -> int:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        total = ordered[middle - 1] + ordered[middle]
        quotient, remainder = divmod(total, 2)
        return quotient + int(remainder == 1 and quotient % 2 == 1)

    @staticmethod
    def _exact_int(value: Any, field: str) -> int:
        if type(value) is not int:
            raise VideoExportError(
                VideoExportErrorCode.INVALID_TIMESTAMP_METADATA,
                f"{field} must be an exact integer",
            )
        return value

    @staticmethod
    def _make_sibling_temp(destination: Path) -> Path:
        descriptor, raw_path = make_temp_file(
            destination.parent,
            prefix=f".{destination.name}.robata-",
            suffix=".tmp",
        )
        os.close(descriptor)
        return Path(raw_path)

    @staticmethod
    def _hash_file(path: Path) -> tuple[int, Sha256Digest]:
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
        return size_bytes, digest.hexdigest()

    @classmethod
    def _publish_pair_non_overwriting(
        cls,
        pairs: tuple[tuple[Path, Path], tuple[Path, Path]],
    ) -> None:
        published: list[tuple[Path, Path]] = []
        try:
            for temporary, destination in pairs:
                try:
                    os.link(temporary, destination)
                except FileExistsError as exc:
                    raise VideoExportError(
                        VideoExportErrorCode.DESTINATION_EXISTS,
                        f"destination appeared during commit: {destination}",
                    ) from exc
                except OSError as exc:
                    raise VideoExportError(
                        VideoExportErrorCode.ATOMIC_COMMIT_FAILED,
                        f"could not publish destination {destination}: {exc}",
                    ) from exc
                published.append((temporary, destination))
        except Exception:
            for temporary, destination in reversed(published):
                try:
                    if destination.exists() and os.path.samefile(temporary, destination):
                        destination.unlink()
                except OSError:
                    pass
            raise
        for temporary, _ in pairs:
            cls._safe_unlink(temporary)

    @staticmethod
    def _safe_unlink(path: Path | None) -> None:
        if path is None:
            return
        with suppress(OSError):
            path.unlink(missing_ok=True)


class PyAvH264RemuxSession:
    """Incremental H.264 remux branch sharing the offline export state machine."""

    def __init__(
        self,
        *,
        exporter: PyAvH264Mp4Exporter,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
        video_temp: Path,
        sidecar_temp: Path,
        interval_temp: Path,
        decoded_frame_observer: DecodedFrameObserver | None,
        validate_output: bool,
    ) -> None:
        self._exporter = exporter
        self.camera_id = camera_id
        self._channel = channel
        self._video_path = video_path
        self._sidecar_path = sidecar_path
        self._video_temp = video_temp
        self._sidecar_temp = sidecar_temp
        self._interval_temp = interval_temp
        self._decoded_frame_observer = decoded_frame_observer
        self._validate_output = validate_output
        self._decoder = av.CodecContext.create("h264", "r")
        self._sidecar_stream: BinaryIO | None = sidecar_temp.open("wb")
        self._interval_spool: _Int64IntervalSpool | None = _Int64IntervalSpool(interval_temp)
        self._output: Any | None = None
        self._output_stream: Any | None = None
        self._previous: _AccessUnit | None = None
        self._bootstrap_time_ns: int | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._source_count = 0
        self._leading_count = 0
        self._keyframe_count = 0
        self._source_first_ns: int | None = None
        self._source_last_ns: int | None = None
        self._leading_first_ns: int | None = None
        self._leading_last_ns: int | None = None
        self._previous_source_time_ns: int | None = None
        self._exported_packet_count = 0
        self._decoded_frame_count = 0
        self._facts: ExportedCameraVideoFacts | None = None
        self._aborted = False

    def append_access_unit(
        self,
        envelope: H264PacketEnvelope,
        reference: PacketReference,
        *,
        framing_version: str,
    ) -> None:
        """Append one envelope from the single-pass MCAP tee."""

        try:
            packet = envelope.packet
            if framing_version != ACCESS_UNIT_FRAMING_VERSION:
                raise VideoExportError(
                    VideoExportErrorCode.INVALID_ACCESS_UNIT,
                    "incremental remux framing version is unsupported",
                )
            if packet.camera_id is not self.camera_id:
                raise VideoExportError(
                    VideoExportErrorCode.INVALID_CHANNEL,
                    "incremental remux packet belongs to another camera",
                )
            if packet.source_order != self._source_count:
                raise VideoExportError(
                    VideoExportErrorCode.INVALID_ACCESS_UNIT,
                    "incremental remux source_order is not contiguous",
                )
            if reference != packet.reference():
                raise VideoExportError(
                    VideoExportErrorCode.INVALID_ACCESS_UNIT,
                    "incremental remux packet reference differs from its envelope",
                )
            if packet.is_keyframe != (5 in envelope.nal_types):
                raise VideoExportError(
                    VideoExportErrorCode.INVALID_ACCESS_UNIT,
                    "incremental remux keyframe flag differs from its NAL units",
                )
            self._append_unit(
                _AccessUnit(
                    log_time_ns=packet.source_timestamp_ns,
                    publish_time_ns=envelope.source_publish_time_ns,
                    embedded_header_time_ns=envelope.embedded_header_time_ns,
                    source_sequence=packet.source_sequence,
                    payload=packet.payload,
                    nal_types=envelope.nal_types,
                ),
                envelope=envelope,
            )
        except VideoExportError:
            self.abort()
            raise
        except OSError as exc:
            self.abort()
            raise VideoExportError(
                VideoExportErrorCode.SOURCE_IO_ERROR,
                f"camera-video export I/O failed: {exc}",
            ) from exc
        except Exception as exc:
            self.abort()
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                f"incremental camera-video remux failed: {type(exc).__name__}: {exc}",
            ) from exc

    def _append_unit(
        self,
        unit: _AccessUnit,
        *,
        envelope: H264PacketEnvelope | None = None,
    ) -> None:
        self._ensure_open()
        self._source_count += 1
        log_time_ns = unit.log_time_ns
        if (
            self._previous_source_time_ns is not None
            and log_time_ns <= self._previous_source_time_ns
        ):
            raise VideoExportError(
                VideoExportErrorCode.NONMONOTONIC_LOG_TIME,
                "camera source log times must be strictly increasing",
            )
        self._previous_source_time_ns = log_time_ns
        self._source_first_ns = (
            self._source_first_ns if self._source_first_ns is not None else log_time_ns
        )
        self._source_last_ns = log_time_ns

        if self._bootstrap_time_ns is None:
            if not is_independent_h264_bootstrap(unit.nal_types):
                self._leading_count += 1
                self._leading_first_ns = (
                    self._leading_first_ns if self._leading_first_ns is not None else log_time_ns
                )
                self._leading_last_ns = log_time_ns
                return
            self._bootstrap_time_ns = log_time_ns

        relative_pts_ns = log_time_ns - self._bootstrap_time_ns
        frame = self._exporter._decode_frame_without_reordering(
            self._decoder,
            unit,
            relative_pts_ns,
        )
        frame_width, frame_height = frame.width, frame.height
        decoded_frame_index = self._decoded_frame_count
        self._decoded_frame_count += 1
        if self._width is None:
            self._width, self._height = frame_width, frame_height
        elif (frame_width, frame_height) != (self._width, self._height):
            raise VideoExportError(
                VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                "decoded camera dimensions change within one export",
            )
        if envelope is not None and self._decoded_frame_observer is not None:
            self._decoded_frame_observer(envelope, frame, decoded_frame_index)

        if self._previous is not None:
            duration_ns = unit.log_time_ns - self._previous.log_time_ns
            assert self._interval_spool is not None
            self._interval_spool.append(duration_ns)
            if self._output is None:
                assert self._width is not None
                assert self._height is not None
                self._output = av.open(str(self._video_temp), mode="w", format="mp4")
                self._output_stream = self._output.add_stream(
                    "h264",
                    rate=Fraction(1_000_000_000, duration_ns),
                )
                self._output_stream.width = self._width
                self._output_stream.height = self._height
                self._output_stream.time_base = _NANOSECOND_TIME_BASE
            assert self._output_stream is not None
            assert self._sidecar_stream is not None
            self._exporter._mux_and_record(
                self._output,
                self._output_stream,
                self._sidecar_stream,
                self.camera_id,
                self._previous,
                self._bootstrap_time_ns,
                duration_ns,
                self._exported_packet_count,
                duration_is_estimated=False,
            )
            self._exported_packet_count += 1
            self._keyframe_count += int(self._previous.is_keyframe)
        self._previous = unit

    def seal(self) -> None:
        """Finalize, verify, and atomically publish the MP4/sidecar pair."""

        if self._facts is not None:
            return
        try:
            temporary = self._finish_temporary_outputs()
            self._close_interval_spool()
            self._exporter._safe_unlink(self._interval_temp)
            self._close_sidecar()
            self._close_output(trailer_required=True)
            decoded_frames = (
                self._exporter._validate_exported_mp4(
                    self._video_temp,
                    self._sidecar_temp,
                    temporary,
                )
                if self._validate_output
                else temporary.decoded_frame_count
            )
            if decoded_frames != temporary.decoded_frame_count:
                raise VideoExportError(
                    VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                    "source and exported MP4 decoded-frame counts differ",
                )
            video_size_bytes, video_sha256 = self._exporter._hash_file(self._video_temp)
            sidecar_size_bytes, sidecar_sha256 = self._exporter._hash_file(self._sidecar_temp)
            facts = self._build_facts(
                temporary,
                decoded_frames=decoded_frames,
                video_size_bytes=video_size_bytes,
                video_sha256=video_sha256,
                sidecar_size_bytes=sidecar_size_bytes,
                sidecar_sha256=sidecar_sha256,
            )
            self._exporter._publish_pair_non_overwriting(
                (
                    (self._video_temp, self._video_path),
                    (self._sidecar_temp, self._sidecar_path),
                )
            )
            self._facts = facts
        except VideoExportError:
            self.abort()
            raise
        except OSError as exc:
            self.abort()
            raise VideoExportError(
                VideoExportErrorCode.SOURCE_IO_ERROR,
                f"camera-video export I/O failed: {exc}",
            ) from exc
        except Exception as exc:
            self.abort()
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                f"incremental camera-video remux finalization failed: {type(exc).__name__}: {exc}",
            ) from exc

    def abort(self) -> None:
        """Best-effort cleanup that never masks the initiating failure."""

        if self._facts is not None or self._aborted:
            return
        self._aborted = True
        with suppress(Exception):
            self._close_sidecar()
        with suppress(Exception):
            self._close_output(trailer_required=False)
        with suppress(Exception):
            self._close_interval_spool()
        self._exporter._safe_unlink(self._video_temp)
        self._exporter._safe_unlink(self._sidecar_temp)
        self._exporter._safe_unlink(self._interval_temp)

    @property
    def facts(self) -> ExportedCameraVideoFacts:
        if self._facts is None:
            raise RuntimeError("incremental remux facts are available only after seal")
        return self._facts

    def _ensure_open(self) -> None:
        if self._facts is not None:
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                "incremental remux session is already sealed",
            )
        if self._aborted:
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                "incremental remux session is aborted",
            )

    def _finish_temporary_outputs(self) -> _TemporaryExportFacts:
        self._ensure_open()
        if self._source_count == 0:
            raise VideoExportError(
                VideoExportErrorCode.NO_CAMERA_MESSAGES,
                f"no messages found for exact channel {self._channel.channel_id}",
            )
        if self._source_count != self._channel.message_count:
            raise VideoExportError(
                VideoExportErrorCode.INVALID_CHANNEL,
                "mapped channel message count differs from the inspected count",
            )
        if self._bootstrap_time_ns is None or self._previous is None:
            raise VideoExportError(
                VideoExportErrorCode.BOOTSTRAP_NOT_FOUND,
                "no Annex-B SPS+PPS+IDR bootstrap was found",
            )
        if (
            self._interval_spool is None
            or self._interval_spool.count == 0
            or self._output is None
            or self._output_stream is None
        ):
            raise VideoExportError(
                VideoExportErrorCode.INVALID_ACCESS_UNIT,
                "at least two exportable access units are required for packet duration",
            )
        delayed_frames = self._decoder.decode(None)
        if delayed_frames or self._decoder.has_b_frames:
            raise VideoExportError(
                VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                "decoder flush exposed delayed or reordered frames",
            )

        tail_duration_ns = self._interval_spool.median_half_even()
        assert self._sidecar_stream is not None
        self._exporter._mux_and_record(
            self._output,
            self._output_stream,
            self._sidecar_stream,
            self.camera_id,
            self._previous,
            self._bootstrap_time_ns,
            tail_duration_ns,
            self._exported_packet_count,
            duration_is_estimated=True,
        )
        self._exported_packet_count += 1
        self._keyframe_count += int(self._previous.is_keyframe)

        assert self._width is not None
        assert self._height is not None
        assert self._source_first_ns is not None
        assert self._source_last_ns is not None
        last_pts_ns = self._previous.log_time_ns - self._bootstrap_time_ns
        return _TemporaryExportFacts(
            source_message_count=self._source_count,
            leading_access_unit_count=self._leading_count,
            exported_packet_count=self._exported_packet_count,
            decoded_frame_count=self._decoded_frame_count,
            keyframe_count=self._keyframe_count,
            width=self._width,
            height=self._height,
            source_first_log_time_ns=self._source_first_ns,
            source_last_log_time_ns=self._source_last_ns,
            leading_first_log_time_ns=self._leading_first_ns,
            leading_last_log_time_ns=self._leading_last_ns,
            export_first_source_log_time_ns=self._bootstrap_time_ns,
            export_last_source_log_time_ns=self._previous.log_time_ns,
            first_pts_ns=0,
            last_pts_ns=last_pts_ns,
            duration_ns=last_pts_ns + tail_duration_ns,
            tail_duration_ns=tail_duration_ns,
        )

    def _close_sidecar(self) -> None:
        stream = self._sidecar_stream
        self._sidecar_stream = None
        if stream is not None:
            stream.close()

    def _close_interval_spool(self) -> None:
        spool = self._interval_spool
        self._interval_spool = None
        if spool is not None:
            spool.close()

    def _close_output(self, *, trailer_required: bool) -> None:
        output = self._output
        self._output = None
        self._output_stream = None
        if output is None:
            return
        try:
            output.close()
        except Exception as exc:
            if trailer_required:
                raise VideoExportError(
                    VideoExportErrorCode.REMUX_FAILED,
                    f"MP4 trailer write failed: {type(exc).__name__}: {exc}",
                ) from exc

    def _build_facts(
        self,
        temporary: _TemporaryExportFacts,
        *,
        decoded_frames: int,
        video_size_bytes: int,
        video_sha256: Sha256Digest,
        sidecar_size_bytes: int,
        sidecar_sha256: Sha256Digest,
    ) -> ExportedCameraVideoFacts:
        return ExportedCameraVideoFacts(
            camera_id=self.camera_id,
            channel_id=self._channel.channel_id,
            topic=self._channel.topic,
            video_path=self._video_path,
            sidecar_path=self._sidecar_path,
            source_message_count=temporary.source_message_count,
            leading_access_unit_count=temporary.leading_access_unit_count,
            trailing_access_unit_count=0,
            exported_packet_count=temporary.exported_packet_count,
            decoded_frame_count=decoded_frames,
            keyframe_count=temporary.keyframe_count,
            width=temporary.width,
            height=temporary.height,
            source_first_log_time_ns=temporary.source_first_log_time_ns,
            source_last_log_time_ns=temporary.source_last_log_time_ns,
            leading_first_log_time_ns=temporary.leading_first_log_time_ns,
            leading_last_log_time_ns=temporary.leading_last_log_time_ns,
            trailing_first_log_time_ns=None,
            trailing_last_log_time_ns=None,
            export_first_source_log_time_ns=temporary.export_first_source_log_time_ns,
            export_last_source_log_time_ns=temporary.export_last_source_log_time_ns,
            first_pts_ns=temporary.first_pts_ns,
            last_pts_ns=temporary.last_pts_ns,
            duration_ns=temporary.duration_ns,
            time_base_numerator=_NANOSECOND_TIME_BASE.numerator,
            time_base_denominator=_NANOSECOND_TIME_BASE.denominator,
            tail_duration_ns=temporary.tail_duration_ns,
            tail_duration_policy=_TAIL_DURATION_POLICY,
            max_timestamp_mapping_error_ns=0,
            video_size_bytes=video_size_bytes,
            video_sha256=video_sha256,
            sidecar_row_count=temporary.exported_packet_count,
            sidecar_size_bytes=sidecar_size_bytes,
            sidecar_sha256=sidecar_sha256,
        )
