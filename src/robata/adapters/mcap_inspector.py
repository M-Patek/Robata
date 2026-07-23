"""MCAP inspection through the official Python reader."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from mcap.reader import make_reader
from mcap.records import Channel, Message, MessageIndex, Schema
from mcap.stream_reader import StreamReader
from mcap.summary import Summary
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
    schema_encoding: str | None
    schema_content_sha256: str | None
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
            schema_encoding=None if schema is None else schema.encoding,
            schema_content_sha256=(
                None if schema is None else hashlib.sha256(schema.data).hexdigest()
            ),
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
            schema_encoding=self.schema_encoding,
            schema_content_sha256=self.schema_content_sha256,
        )


@dataclass(frozen=True, slots=True)
class McapPreflight:
    """Header, summary, and message-index facts read without source payloads."""

    source: Path
    source_size_bytes: int
    header_profile: str
    header_library: str
    channel_count: int
    message_count: int
    first_message_time_ns: int | None
    last_message_time_ns: int | None
    channels: tuple[ChannelInspection, ...]
    message_indexes_complete: bool

    def as_mapping_inspection(self, expected_source_sha256: str) -> McapInspection:
        """Expose only index-backed facts through the existing exact mapping policy."""

        return McapInspection(
            source=self.source,
            source_size_bytes=self.source_size_bytes,
            source_sha256=expected_source_sha256,
            header_profile=self.header_profile,
            header_library=self.header_library,
            summary_available=True,
            channel_count=self.channel_count,
            message_count=self.message_count,
            first_message_time_ns=self.first_message_time_ns,
            last_message_time_ns=self.last_message_time_ns,
            channels=self.channels,
        )


class McapInspectionAccumulator:
    """Build the accepted inspection while one non-seeking reader consumes payloads."""

    def __init__(
        self,
        preflight: McapPreflight,
        *,
        expected_source_sha256: str,
    ) -> None:
        self._preflight = preflight
        self._expected_source_sha256 = expected_source_sha256
        self._accumulators = {
            channel.channel_id: _ChannelAccumulator(
                channel_id=channel.channel_id,
                topic=channel.topic,
                schema_name=channel.schema_name,
                message_encoding=channel.message_encoding,
                schema_encoding=channel.schema_encoding,
                schema_content_sha256=channel.schema_content_sha256,
            )
            for channel in preflight.channels
        }
        self._identity_objects: dict[int, tuple[int, int | None]] = {}
        self._decoder_factory = DecoderFactory()
        self._message_count = 0
        self._first_message_time_ns: int | None = None
        self._last_message_time_ns: int | None = None

    def observe(
        self,
        schema: Schema | None,
        channel: Channel,
        message: Message,
        *,
        decoded: Any | None = None,
    ) -> None:
        accumulator = self._accumulators.get(channel.id)
        if accumulator is None:
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                f"channel {channel.id} is absent from the MCAP summary",
            )
        identity_objects = (id(channel), None if schema is None else id(schema))
        if self._identity_objects.get(channel.id) != identity_objects:
            OfficialMcapInspector._validate_channel_identity(accumulator, channel, schema)
            self._identity_objects[channel.id] = identity_objects

        accumulator.observe_timestamp(message.log_time)
        self._message_count += 1
        if self._first_message_time_ns is None or message.log_time < self._first_message_time_ns:
            self._first_message_time_ns = message.log_time
        if self._last_message_time_ns is None or message.log_time > self._last_message_time_ns:
            self._last_message_time_ns = message.log_time
        if accumulator.schema_name == COMPRESSED_IMAGE_SCHEMA and not accumulator.metadata_decoded:
            if decoded is None:
                OfficialMcapInspector._decode_image_metadata(
                    accumulator,
                    channel,
                    schema,
                    message.data,
                    self._decoder_factory,
                )
            else:
                _apply_image_metadata(accumulator, decoded)

    def finish(self, *, source_size_bytes: int, source_sha256: str) -> McapInspection:
        channels = tuple(
            accumulator.freeze()
            for accumulator in sorted(
                self._accumulators.values(),
                key=lambda value: value.channel_id,
            )
        )
        inspection = McapInspection(
            source=self._preflight.source,
            source_size_bytes=source_size_bytes,
            source_sha256=source_sha256,
            header_profile=self._preflight.header_profile,
            header_library=self._preflight.header_library,
            summary_available=True,
            channel_count=len(channels),
            message_count=self._message_count,
            first_message_time_ns=self._first_message_time_ns,
            last_message_time_ns=self._last_message_time_ns,
            channels=channels,
        )
        if source_size_bytes != self._preflight.source_size_bytes:
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                "single-pass source size differs from the bounded MCAP preflight",
            )
        if source_sha256 != self._expected_source_sha256:
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                "single-pass source digest differs from the expected source identity",
            )
        if (
            inspection.channel_count != self._preflight.channel_count
            or inspection.message_count != self._preflight.message_count
            or inspection.first_message_time_ns != self._preflight.first_message_time_ns
            or inspection.last_message_time_ns != self._preflight.last_message_time_ns
            or tuple(_identity_count_channel(channel) for channel in inspection.channels)
            != tuple(
                _identity_count_channel(channel) for channel in self._preflight.channels
            )
            or (
                self._preflight.message_indexes_complete
                and tuple(_indexed_channel(channel) for channel in inspection.channels)
                != tuple(_indexed_channel(channel) for channel in self._preflight.channels)
            )
        ):
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                "single-pass source facts differ from the MCAP summary/message indexes",
            )
        return inspection


class OfficialMcapInspector:
    """Read header/summary and then scan source messages in physical file order."""

    def __init__(self, *, validate_crcs: bool = True) -> None:
        self._validate_crcs = validate_crcs

    def preflight(self, source: Path) -> McapPreflight:
        """Read the bounded random-access facts required before the payload tee."""

        source = Path(source)
        self._validate_source_path(source)
        try:
            with source.open("rb") as stream:
                stream.seek(0, 2)
                source_size_bytes = stream.tell()
                stream.seek(0)
                reader = make_reader(stream, validate_crcs=self._validate_crcs)
                header = reader.get_header()
                summary = reader.get_summary()
                if summary is None or summary.statistics is None:
                    raise IngestionError(
                        IngestionErrorCode.CORRUPT_MCAP,
                        "canonical fast path requires an MCAP summary with statistics",
                    )
                channels, message_indexes_complete = self._summary_channel_facts(
                    stream,
                    summary,
                )
                statistics = summary.statistics
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
                f"MCAP bounded preflight is unreadable: {exc}",
            ) from exc

        return McapPreflight(
            source=source,
            source_size_bytes=source_size_bytes,
            header_profile=header.profile,
            header_library=header.library,
            channel_count=len(channels),
            message_count=statistics.message_count,
            first_message_time_ns=(
                statistics.message_start_time if statistics.message_count > 0 else None
            ),
            last_message_time_ns=(
                statistics.message_end_time if statistics.message_count > 0 else None
            ),
            channels=channels,
            message_indexes_complete=message_indexes_complete,
        )

    def inspect(self, source: Path) -> McapInspection:
        source = Path(source)
        self._validate_source_path(source)

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
    def _validate_source_path(source: Path) -> None:
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

    @classmethod
    def _summary_channel_facts(
        cls,
        stream: BinaryIO,
        summary: Summary,
    ) -> tuple[tuple[ChannelInspection, ...], bool]:
        statistics = summary.statistics
        if statistics is None:
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                "canonical fast path requires MCAP summary statistics",
            )
        accumulators = cls._summary_channels(summary)
        if statistics.channel_count != len(accumulators):
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                "MCAP summary channel count is inconsistent",
            )
        if set(statistics.channel_message_counts) != set(accumulators):
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                "MCAP summary channel counts are inconsistent",
            )
        for chunk in sorted(summary.chunk_indexes, key=lambda value: value.chunk_start_offset):
            for channel_id, offset in chunk.message_index_offsets.items():
                stream.seek(offset)
                record = next(StreamReader(stream, skip_magic=True).records)
                if not isinstance(record, MessageIndex) or record.channel_id != channel_id:
                    raise IngestionError(
                        IngestionErrorCode.CORRUPT_MCAP,
                        "MCAP chunk points to an invalid message index",
                    )
                accumulator = accumulators.get(channel_id)
                if accumulator is None:
                    raise IngestionError(
                        IngestionErrorCode.CORRUPT_MCAP,
                        f"message index references unknown channel {channel_id}",
                    )
                for timestamp_ns, _message_offset in record.records:
                    accumulator.observe_timestamp(timestamp_ns)

        indexed_count = sum(value.message_count for value in accumulators.values())
        indexes_complete = indexed_count == statistics.message_count and not any(
            accumulators[channel_id].message_count != expected_count
            for channel_id, expected_count in statistics.channel_message_counts.items()
        )
        if indexes_complete:
            channels = tuple(
                accumulator.freeze()
                for accumulator in sorted(
                    accumulators.values(),
                    key=lambda value: value.channel_id,
                )
            )
        else:
            channels = tuple(
                _summary_only_channel(
                    accumulator,
                    statistics.channel_message_counts.get(channel_id, 0),
                )
                for channel_id, accumulator in sorted(accumulators.items())
            )
        first_values = tuple(
            value
            for channel in channels
            for value in (channel.first_message_time_ns,)
            if value is not None
        )
        last_values = tuple(
            value
            for channel in channels
            for value in (channel.last_message_time_ns,)
            if value is not None
        )
        first = min(first_values, default=None)
        last = max(last_values, default=None)
        expected_first = statistics.message_start_time if statistics.message_count > 0 else None
        expected_last = statistics.message_end_time if statistics.message_count > 0 else None
        if indexes_complete and (first != expected_first or last != expected_last):
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                "MCAP message indexes disagree with summary timestamp bounds",
            )
        return channels, indexes_complete

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
        schema_encoding = schema.encoding if schema is not None else None
        schema_content_sha256 = (
            hashlib.sha256(schema.data).hexdigest() if schema is not None else None
        )
        if (
            accumulator.topic != channel.topic
            or accumulator.schema_name != schema_name
            or accumulator.message_encoding != channel.message_encoding
            or accumulator.schema_encoding != schema_encoding
            or accumulator.schema_content_sha256 != schema_content_sha256
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
        _apply_image_metadata(accumulator, decoded)

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


def _apply_image_metadata(accumulator: _ChannelAccumulator, decoded: Any) -> None:
    codec = getattr(decoded, "format", None)
    frame_id = getattr(decoded, "frame_id", None)
    accumulator.codec = codec.strip().lower() if isinstance(codec, str) and codec else None
    accumulator.frame_id = frame_id if isinstance(frame_id, str) and frame_id else None
    accumulator.metadata_decoded = True


def _summary_only_channel(
    accumulator: _ChannelAccumulator,
    message_count: int,
) -> ChannelInspection:
    channel = accumulator.freeze()
    return ChannelInspection(
        channel_id=channel.channel_id,
        topic=channel.topic,
        schema_name=channel.schema_name,
        message_encoding=channel.message_encoding,
        message_count=message_count,
        first_message_time_ns=None,
        last_message_time_ns=None,
        monotonic=False,
        codec=None,
        frame_id=None,
        schema_encoding=channel.schema_encoding,
        schema_content_sha256=channel.schema_content_sha256,
    )


def _identity_count_channel(channel: ChannelInspection) -> tuple[object, ...]:
    return (
        channel.channel_id,
        channel.topic,
        channel.schema_name,
        channel.message_encoding,
        channel.message_count,
        channel.schema_encoding,
        channel.schema_content_sha256,
    )


def _indexed_channel(channel: ChannelInspection) -> tuple[object, ...]:
    return (
        channel.first_message_time_ns,
        channel.last_message_time_ns,
        channel.monotonic,
    )
