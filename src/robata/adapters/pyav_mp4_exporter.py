"""Direct H.264 MCAP-to-MP4 remux with an exact timestamp ledger."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Final

import av
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from robata.contracts import CameraId, Sha256Digest, canonical_json_bytes
from robata.ports import (
    COMPRESSED_IMAGE_SCHEMA,
    ChannelInspection,
    ExportedCameraVideoFacts,
    VideoExportError,
    VideoExportErrorCode,
)
from robata.tempfiles import make_temp_file

_NANOSECOND_TIME_BASE = Fraction(1, 1_000_000_000)
_TAIL_DURATION_POLICY = "MEDIAN_POSITIVE_INTERVAL"
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

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
    expected_pts_ns: tuple[int, ...]
    expected_duration_ns: tuple[int, ...]
    expected_keyframes: tuple[bool, ...]


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


def _is_independent_bootstrap(nal_types: tuple[int, ...]) -> bool:
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
        self._validate_request(source, camera_id, channel, video_path, sidecar_path)

        video_temp: Path | None = None
        sidecar_temp: Path | None = None
        try:
            video_temp = self._make_sibling_temp(video_path)
            sidecar_temp = self._make_sibling_temp(sidecar_path)
            temporary = self._write_temporary_outputs(
                source,
                camera_id,
                channel,
                video_temp,
                sidecar_temp,
            )
            decoded_frames = self._validate_exported_mp4(video_temp, temporary)
            if decoded_frames != temporary.decoded_frame_count:
                raise VideoExportError(
                    VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                    "source and exported MP4 decoded-frame counts differ",
                )

            video_size_bytes, video_sha256 = self._hash_file(video_temp)
            sidecar_size_bytes, sidecar_sha256 = self._hash_file(sidecar_temp)
            self._publish_pair_non_overwriting(
                ((video_temp, video_path), (sidecar_temp, sidecar_path))
            )

            return ExportedCameraVideoFacts(
                camera_id=camera_id,
                channel_id=channel.channel_id,
                topic=channel.topic,
                video_path=video_path,
                sidecar_path=sidecar_path,
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
            self._safe_unlink(video_temp)
            self._safe_unlink(sidecar_temp)

    @staticmethod
    def _validate_request(
        source: Path,
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

    def _write_temporary_outputs(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_temp: Path,
        sidecar_temp: Path,
    ) -> _TemporaryExportFacts:
        decoder = av.CodecContext.create("h264", "r")
        output: Any | None = None
        output_stream: Any | None = None
        previous: _AccessUnit | None = None
        bootstrap_time_ns: int | None = None
        width: int | None = None
        height: int | None = None
        source_count = 0
        leading_count = 0
        keyframe_count = 0
        source_first_ns: int | None = None
        source_last_ns: int | None = None
        leading_first_ns: int | None = None
        leading_last_ns: int | None = None
        previous_source_time_ns: int | None = None
        intervals: list[int] = []
        expected_pts: list[int] = []
        expected_durations: list[int] = []
        expected_keyframes: list[bool] = []
        decoded_frame_count = 0
        write_completed = False

        try:
            with source.open("rb") as source_stream, sidecar_temp.open("wb") as sidecar_stream:
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
                    source_count += 1
                    log_time_ns = self._exact_int(message.log_time, "source log time")
                    if (
                        previous_source_time_ns is not None
                        and log_time_ns <= previous_source_time_ns
                    ):
                        raise VideoExportError(
                            VideoExportErrorCode.NONMONOTONIC_LOG_TIME,
                            "camera source log times must be strictly increasing",
                        )
                    previous_source_time_ns = log_time_ns
                    source_first_ns = (
                        source_first_ns if source_first_ns is not None else log_time_ns
                    )
                    source_last_ns = log_time_ns
                    unit = self._access_unit(decoded, message)

                    if bootstrap_time_ns is None:
                        if not _is_independent_bootstrap(unit.nal_types):
                            leading_count += 1
                            leading_first_ns = (
                                leading_first_ns if leading_first_ns is not None else log_time_ns
                            )
                            leading_last_ns = log_time_ns
                            continue
                        bootstrap_time_ns = log_time_ns

                    relative_pts_ns = log_time_ns - bootstrap_time_ns
                    frame_width, frame_height = self._decode_without_reordering(
                        decoder,
                        unit,
                        relative_pts_ns,
                    )
                    decoded_frame_count += 1
                    if width is None:
                        width, height = frame_width, frame_height
                    elif (frame_width, frame_height) != (width, height):
                        raise VideoExportError(
                            VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                            "decoded camera dimensions change within one export",
                        )

                    if previous is not None:
                        duration_ns = unit.log_time_ns - previous.log_time_ns
                        intervals.append(duration_ns)
                        if output is None:
                            if width is None or height is None:
                                raise VideoExportError(
                                    VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                                    "decoder did not establish video dimensions",
                                )
                            output = av.open(str(video_temp), mode="w", format="mp4")
                            output_stream = output.add_stream(
                                "h264",
                                rate=Fraction(1_000_000_000, duration_ns),
                            )
                            output_stream.width = width
                            output_stream.height = height
                            output_stream.time_base = _NANOSECOND_TIME_BASE
                        self._mux_and_record(
                            output,
                            output_stream,
                            sidecar_stream,
                            camera_id,
                            previous,
                            bootstrap_time_ns,
                            duration_ns,
                            len(expected_pts),
                            duration_is_estimated=False,
                        )
                        expected_pts.append(previous.log_time_ns - bootstrap_time_ns)
                        expected_durations.append(duration_ns)
                        expected_keyframes.append(previous.is_keyframe)
                        keyframe_count += int(previous.is_keyframe)
                    previous = unit

                if source_count == 0:
                    raise VideoExportError(
                        VideoExportErrorCode.NO_CAMERA_MESSAGES,
                        f"no messages found for exact channel {channel.channel_id}",
                    )
                if source_count != channel.message_count:
                    raise VideoExportError(
                        VideoExportErrorCode.INVALID_CHANNEL,
                        "mapped channel message count differs from the inspected count",
                    )
                if bootstrap_time_ns is None or previous is None:
                    raise VideoExportError(
                        VideoExportErrorCode.BOOTSTRAP_NOT_FOUND,
                        "no Annex-B SPS+PPS+IDR bootstrap was found",
                    )
                if not intervals or output is None or output_stream is None:
                    raise VideoExportError(
                        VideoExportErrorCode.INVALID_ACCESS_UNIT,
                        "at least two exportable access units are required for packet duration",
                    )
                delayed_frames = decoder.decode(None)
                if delayed_frames or decoder.has_b_frames:
                    raise VideoExportError(
                        VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                        "decoder flush exposed delayed or reordered frames",
                    )

                tail_duration_ns = self._median_half_even(intervals)
                self._mux_and_record(
                    output,
                    output_stream,
                    sidecar_stream,
                    camera_id,
                    previous,
                    bootstrap_time_ns,
                    tail_duration_ns,
                    len(expected_pts),
                    duration_is_estimated=True,
                )
                expected_pts.append(previous.log_time_ns - bootstrap_time_ns)
                expected_durations.append(tail_duration_ns)
                expected_keyframes.append(previous.is_keyframe)
                keyframe_count += int(previous.is_keyframe)
                write_completed = True
        except VideoExportError:
            raise
        except OSError:
            raise
        except Exception as exc:
            raise VideoExportError(
                VideoExportErrorCode.MCAP_READ_ERROR,
                f"could not read mapped camera messages: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if output is not None:
                try:
                    output.close()
                except Exception as exc:
                    if write_completed:
                        raise VideoExportError(
                            VideoExportErrorCode.REMUX_FAILED,
                            f"MP4 trailer write failed: {type(exc).__name__}: {exc}",
                        ) from exc

        assert width is not None
        assert height is not None
        assert source_first_ns is not None
        assert source_last_ns is not None
        assert bootstrap_time_ns is not None
        assert previous is not None
        return _TemporaryExportFacts(
            source_message_count=source_count,
            leading_access_unit_count=leading_count,
            exported_packet_count=len(expected_pts),
            decoded_frame_count=decoded_frame_count,
            keyframe_count=keyframe_count,
            width=width,
            height=height,
            source_first_log_time_ns=source_first_ns,
            source_last_log_time_ns=source_last_ns,
            leading_first_log_time_ns=leading_first_ns,
            leading_last_log_time_ns=leading_last_ns,
            export_first_source_log_time_ns=bootstrap_time_ns,
            export_last_source_log_time_ns=previous.log_time_ns,
            first_pts_ns=expected_pts[0],
            last_pts_ns=expected_pts[-1],
            duration_ns=expected_pts[-1] + expected_durations[-1],
            tail_duration_ns=expected_durations[-1],
            expected_pts_ns=tuple(expected_pts),
            expected_duration_ns=tuple(expected_durations),
            expected_keyframes=tuple(expected_keyframes),
        )

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
    def _decode_without_reordering(
        decoder: Any,
        unit: _AccessUnit,
        relative_pts_ns: int,
    ) -> tuple[int, int]:
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
        return frame.width, frame.height

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
    def _validate_exported_mp4(path: Path, facts: _TemporaryExportFacts) -> int:
        try:
            with av.open(str(path), mode="r") as container:
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
                packets = [packet for packet in container.demux(stream) if packet.dts is not None]
                if len(packets) != facts.exported_packet_count:
                    raise VideoExportError(
                        VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                        "exported MP4 packet count differs from the timestamp ledger",
                    )
                for index, packet in enumerate(packets):
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
                        or pts_ns.numerator != facts.expected_pts_ns[index]
                        or dts_ns.numerator != facts.expected_pts_ns[index]
                        or packet.duration != facts.expected_duration_ns[index]
                        or packet.is_keyframe != facts.expected_keyframes[index]
                    ):
                        raise VideoExportError(
                            VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                            f"exported packet {index} differs from its timestamp ledger",
                        )

            decoded_count = 0
            with av.open(str(path), mode="r") as container:
                stream = container.streams.video[0]
                if stream.codec_context.has_b_frames:
                    raise VideoExportError(
                        VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                        "exported MP4 declares reordered frames",
                    )
                for index, frame in enumerate(container.decode(stream)):
                    if index >= len(facts.expected_pts_ns):
                        raise VideoExportError(
                            VideoExportErrorCode.DECODE_VALIDATION_FAILED,
                            "exported MP4 decoded more frames than source access units",
                        )
                    picture_type = getattr(frame.pict_type, "name", str(frame.pict_type))
                    if picture_type == "B" or frame.pts is None or frame.time_base is None:
                        raise VideoExportError(
                            VideoExportErrorCode.FRAME_REORDERING_UNSUPPORTED,
                            "exported MP4 contains unattributable or reordered frames",
                        )
                    pts_ns = Fraction(frame.pts) * frame.time_base * 1_000_000_000
                    if pts_ns.denominator != 1 or pts_ns.numerator != facts.expected_pts_ns[index]:
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
