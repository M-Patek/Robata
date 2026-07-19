"""MCAP inspection through the official Python reader."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap.records import Channel, Schema
from mcap_protobuf.decoder import DecoderFactory

from robata.ports import (
    COMPRESSED_IMAGE_SCHEMA,
    ChannelInspection,
    IngestionError,
    IngestionErrorCode,
    McapInspection,
)


@dataclass(slots=True)
class _ChannelAccumulator:
    channel_id: int
    topic: str
    schema_name: str | None
    message_encoding: str
    message_count: int = 0
    first_message_time_ns: int | None = None
    last_message_time_ns: int | None = None
    previous_time_ns: int | None = None
    monotonic: bool = True
    codec: str | None = None
    frame_id: str | None = None
    metadata_decoded: bool = False

    @classmethod
    def from_channel(cls, channel: Channel, schema: Schema | None) -> _ChannelAccumulator:
        return cls(
            channel_id=channel.id,
            topic=channel.topic,
            schema_name=schema.name if schema is not None else None,
            message_encoding=channel.message_encoding,
        )

    def observe_timestamp(self, timestamp_ns: int) -> None:
        self.message_count += 1
        if self.first_message_time_ns is None or timestamp_ns < self.first_message_time_ns:
            self.first_message_time_ns = timestamp_ns
        if self.last_message_time_ns is None or timestamp_ns > self.last_message_time_ns:
            self.last_message_time_ns = timestamp_ns
        if self.previous_time_ns is not None and timestamp_ns < self.previous_time_ns:
            self.monotonic = False
        self.previous_time_ns = timestamp_ns

    def freeze(self) -> ChannelInspection:
        return ChannelInspection(
            channel_id=self.channel_id,
            topic=self.topic,
            schema_name=self.schema_name,
            message_encoding=self.message_encoding,
            message_count=self.message_count,
            first_message_time_ns=self.first_message_time_ns,
            last_message_time_ns=self.last_message_time_ns,
            monotonic=self.monotonic,
            codec=self.codec,
            frame_id=self.frame_id,
        )


class OfficialMcapInspector:
    """Read header/summary and then scan source messages in physical file order."""

    def __init__(self, *, validate_crcs: bool = True) -> None:
        self._validate_crcs = validate_crcs

    def inspect(self, source: Path) -> McapInspection:
        source = Path(source)
        if not source.exists():
            raise IngestionError(
                IngestionErrorCode.SOURCE_NOT_FOUND,
                f"MCAP source does not exist: {source}",
            )
        if not source.is_file():
            raise IngestionError(
                IngestionErrorCode.SOURCE_IO_ERROR,
                f"MCAP source is not a file: {source}",
            )

        try:
            source_size_bytes, source_sha256 = self._hash_source(source)
            with source.open("rb") as stream:
                reader = make_reader(stream, validate_crcs=self._validate_crcs)
                header = reader.get_header()
                summary = reader.get_summary()
                accumulators = self._summary_channels(summary)
                decoder_factory = DecoderFactory()
                message_count = 0
                first_message_time_ns: int | None = None
                last_message_time_ns: int | None = None

                for schema, channel, message in reader.iter_messages(log_time_order=False):
                    accumulator = accumulators.get(channel.id)
                    if accumulator is None:
                        accumulator = _ChannelAccumulator.from_channel(channel, schema)
                        accumulators[channel.id] = accumulator
                    self._validate_channel_identity(accumulator, channel, schema)
                    accumulator.observe_timestamp(message.log_time)
                    message_count += 1
                    if first_message_time_ns is None or message.log_time < first_message_time_ns:
                        first_message_time_ns = message.log_time
                    if last_message_time_ns is None or message.log_time > last_message_time_ns:
                        last_message_time_ns = message.log_time
                    if (
                        accumulator.schema_name == COMPRESSED_IMAGE_SCHEMA
                        and not accumulator.metadata_decoded
                    ):
                        self._decode_image_metadata(
                            accumulator,
                            channel,
                            schema,
                            message.data,
                            decoder_factory,
                        )

                self._validate_summary_counts(summary, accumulators, message_count)
        except IngestionError:
            raise
        except OSError as exc:
            raise IngestionError(
                IngestionErrorCode.SOURCE_IO_ERROR,
                f"could not read MCAP source {source}: {exc}",
            ) from exc
        except Exception as exc:
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                f"MCAP structure is unreadable: {exc}",
            ) from exc

        channels = tuple(
            accumulator.freeze()
            for accumulator in sorted(accumulators.values(), key=lambda value: value.channel_id)
        )
        return McapInspection(
            source=source,
            source_size_bytes=source_size_bytes,
            source_sha256=source_sha256,
            header_profile=header.profile,
            header_library=header.library,
            summary_available=summary is not None,
            channel_count=len(channels),
            message_count=message_count,
            first_message_time_ns=first_message_time_ns,
            last_message_time_ns=last_message_time_ns,
            channels=channels,
        )

    @staticmethod
    def _hash_source(source: Path) -> tuple[int, str]:
        # This separate pass keeps the vertical slice simple; production ingestion can
        # combine hashing with its first durable source stream.
        digest = hashlib.sha256()
        size_bytes = 0
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
        return size_bytes, digest.hexdigest()

    @staticmethod
    def _summary_channels(summary: Any | None) -> dict[int, _ChannelAccumulator]:
        if summary is None:
            return {}
        result: dict[int, _ChannelAccumulator] = {}
        for channel_id, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id) if channel.schema_id != 0 else None
            result[channel_id] = _ChannelAccumulator.from_channel(channel, schema)
        return result

    @staticmethod
    def _validate_channel_identity(
        accumulator: _ChannelAccumulator,
        channel: Channel,
        schema: Schema | None,
    ) -> None:
        schema_name = schema.name if schema is not None else None
        if (
            accumulator.topic != channel.topic
            or accumulator.schema_name != schema_name
            or accumulator.message_encoding != channel.message_encoding
        ):
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                f"channel {channel.id} changes identity within the MCAP",
            )

    @staticmethod
    def _decode_image_metadata(
        accumulator: _ChannelAccumulator,
        channel: Channel,
        schema: Schema | None,
        data: bytes,
        decoder_factory: DecoderFactory,
    ) -> None:
        decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
        if decoder is None:
            accumulator.metadata_decoded = True
            return
        decoded = decoder(data)
        codec = getattr(decoded, "format", None)
        frame_id = getattr(decoded, "frame_id", None)
        accumulator.codec = codec.strip().lower() if isinstance(codec, str) and codec else None
        accumulator.frame_id = frame_id if isinstance(frame_id, str) and frame_id else None
        accumulator.metadata_decoded = True

    @staticmethod
    def _validate_summary_counts(
        summary: Any | None,
        accumulators: dict[int, _ChannelAccumulator],
        message_count: int,
    ) -> None:
        if summary is None or summary.statistics is None:
            return
        statistics = summary.statistics
        if statistics.channel_count != len(accumulators):
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                "MCAP summary channel count does not match the scanned channel inventory",
            )
        if statistics.message_count != message_count:
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                "MCAP summary message count does not match the scanned message count",
            )
        for channel_id, expected_count in statistics.channel_message_counts.items():
            accumulator = accumulators.get(channel_id)
            if accumulator is None or accumulator.message_count != expected_count:
                raise IngestionError(
                    IngestionErrorCode.CORRUPT_MCAP,
                    f"MCAP summary count does not match scanned channel {channel_id}",
                )
