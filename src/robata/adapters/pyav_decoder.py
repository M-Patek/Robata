"""H.264 decoder probe backed by PyAV/FFmpeg."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import av
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from robata.ports import (
    COMPRESSED_IMAGE_SCHEMA,
    ChannelInspection,
    DecodeFailure,
    DecoderProbeResult,
    IngestionError,
    IngestionErrorCode,
)


class PyAvH264DecoderProbe:
    """Decode enough source packets to obtain the first real video frame."""

    def __init__(self, *, max_messages: int = 120, validate_crcs: bool = True) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        self._max_messages = max_messages
        self._validate_crcs = validate_crcs

    def probe(self, source: Path, channel: ChannelInspection) -> DecoderProbeResult:
        codec = (channel.codec or "").strip().lower()
        if codec != "h264":
            raise IngestionError(
                IngestionErrorCode.UNSUPPORTED_CODEC,
                f"topic {channel.topic!r} declares unsupported codec {channel.codec!r}",
            )
        if channel.schema_name != COMPRESSED_IMAGE_SCHEMA:
            raise IngestionError(
                IngestionErrorCode.INVALID_CAMERA_MAPPING,
                f"topic {channel.topic!r} is not a {COMPRESSED_IMAGE_SCHEMA} channel",
            )

        try:
            decoder = av.CodecContext.create("h264", "r")
        except Exception as exc:
            raise IngestionError(
                IngestionErrorCode.DECODER_PROBE_FAILED,
                f"could not initialize the PyAV H.264 decoder: {exc}",
            ) from exc

        failures: list[DecodeFailure] = []
        messages_examined = 0
        decoded_frames = 0
        try:
            with Path(source).open("rb") as stream:
                reader = make_reader(
                    stream,
                    validate_crcs=self._validate_crcs,
                    decoder_factories=[DecoderFactory()],
                )
                for schema, observed_channel, message, decoded in reader.iter_decoded_messages(
                    topics=channel.topic
                ):
                    if observed_channel.id != channel.channel_id:
                        continue
                    messages_examined += 1
                    if schema is None or schema.name != COMPRESSED_IMAGE_SCHEMA:
                        failures.append(
                            DecodeFailure(
                                code="INVALID_COMPRESSED_IMAGE_SCHEMA",
                                timestamp_ns=message.log_time,
                                message="message schema changed during decoder probe",
                            )
                        )
                        continue
                    payload = getattr(decoded, "data", None)
                    if not isinstance(payload, bytes):
                        failures.append(
                            DecodeFailure(
                                code="INVALID_COMPRESSED_IMAGE_PAYLOAD",
                                timestamp_ns=message.log_time,
                                message="CompressedImage.data is not bytes",
                            )
                        )
                        continue

                    packet = av.Packet(payload)
                    packet.pts = message.log_time
                    packet.dts = message.log_time
                    packet.time_base = Fraction(1, 1_000_000_000)
                    try:
                        frames = decoder.decode(packet)
                    except Exception as exc:
                        failures.append(
                            DecodeFailure(
                                code="H264_DECODE_ERROR",
                                timestamp_ns=message.log_time,
                                message=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        frames = []

                    decoded_frames += len(frames)
                    if frames:
                        frame = frames[0]
                        return DecoderProbeResult(
                            topic=channel.topic,
                            codec=codec,
                            success=True,
                            width=frame.width,
                            height=frame.height,
                            first_decoded_timestamp_ns=self._frame_timestamp_ns(
                                frame, message.log_time
                            ),
                            messages_examined=messages_examined,
                            decoded_frames=decoded_frames,
                            failures=tuple(failures),
                        )
                    if messages_examined >= self._max_messages:
                        break
        except IngestionError:
            raise
        except OSError as exc:
            raise IngestionError(
                IngestionErrorCode.SOURCE_IO_ERROR,
                f"could not read MCAP source {source}: {exc}",
            ) from exc
        except Exception as exc:
            failures.append(
                DecodeFailure(
                    code="MCAP_PAYLOAD_READ_ERROR",
                    timestamp_ns=None,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

        return DecoderProbeResult(
            topic=channel.topic,
            codec=codec,
            success=False,
            width=None,
            height=None,
            first_decoded_timestamp_ns=None,
            messages_examined=messages_examined,
            decoded_frames=decoded_frames,
            failures=tuple(failures),
        )

    @staticmethod
    def _frame_timestamp_ns(frame: Any, fallback_ns: int) -> int:
        if frame.pts is None or frame.time_base is None:
            return fallback_ns
        return int(Fraction(frame.pts) * frame.time_base * 1_000_000_000)
