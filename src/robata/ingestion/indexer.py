"""Deterministic stream indexing over already-observed MCAP facts.

The indexer never invents source-frame offsets or decoder facts. It can build
stream and mapping projections from an ``McapInspection`` plus successful
decoder probes; frame rows remain empty until a reader adapter supplies real
message locations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from robata.contracts import CAMERA_IDS, CameraId, Sha256Digest, semantic_sha256
from robata.ingestion.models import (
    CameraMapping,
    CameraMappingRun,
    IngestionResult,
    StreamIndex,
)
from robata.ports.ingestion import (
    CameraMappingPolicy,
    DecoderProbe,
    IngestionError,
    IngestionErrorCode,
    McapInspection,
    McapInspector,
)


class IndexingCapabilityError(RuntimeError):
    """Raised when indexing would require an unconfigured observation port."""


class StreamIndexer:
    """Build immutable six-stream indexes from observed source facts."""

    def __init__(
        self,
        *,
        mapping_policy: CameraMappingPolicy | None = None,
        decoder_probe: DecoderProbe | None = None,
        inspector: McapInspector | None = None,
        mapping_policy_version: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._mapping_policy = mapping_policy
        self._decoder_probe = decoder_probe
        self._inspector = inspector
        inferred_version = getattr(mapping_policy, "version", None)
        self._mapping_policy_version = mapping_policy_version or inferred_version
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def mapping_policy_version(self) -> str:
        """Return the exact mapping-policy version used by this indexer."""

        return self._require_mapping_policy_version()

    @property
    def mapping_policy_digest(self) -> Sha256Digest:
        """Return a policy-provided semantic digest or fail closed."""

        policy = self._require_mapping_policy()
        digest = getattr(policy, "semantic_digest", None)
        if not isinstance(digest, str) or len(digest) != 64:
            raise IndexingCapabilityError("mapping policy must expose a canonical semantic_digest")
        if any(character not in "0123456789abcdef" for character in digest):
            raise IndexingCapabilityError(
                "mapping policy semantic_digest must be lowercase hexadecimal"
            )
        return digest

    def index_streams(
        self,
        mcap_id: str,
        source: Any,
    ) -> IngestionResult:
        """Build six stream projections without fabricating frame locations.

        ``source`` may be an existing ``McapInspection`` or a local path when
        an inspector is configured. Missing policy/probe dependencies fail
        closed. Source-frame indexes are intentionally empty because the
        current inspection port exposes aggregate channel facts, not message
        offsets or sequences.
        """

        if not isinstance(mcap_id, str) or not mcap_id:
            raise ValueError("mcap_id must be a nonempty string")
        inspection = self._inspection_for(source)
        policy = self._require_mapping_policy()
        probe = self._require_decoder_probe()
        version = self._require_mapping_policy_version()

        try:
            resolved = policy.resolve(inspection)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                IngestionErrorCode.INVALID_CAMERA_MAPPING,
                f"camera mapping failed: {type(exc).__name__}: {exc}",
            ) from exc

        stream_indexes: list[StreamIndex] = []
        mappings: list[CameraMapping] = []
        source_content_sha256 = inspection.source_sha256
        for camera_id in CAMERA_IDS:
            channel = resolved[camera_id]
            self._validate_channel_observation(channel)
            try:
                probe_result = probe.probe(inspection.source, channel)
            except IngestionError:
                raise
            except Exception as exc:
                raise IngestionError(
                    IngestionErrorCode.SOURCE_IO_ERROR,
                    f"decoder probe failed for {channel.topic!r}: {type(exc).__name__}: {exc}",
                ) from exc
            if (
                not probe_result.success
                or probe_result.decoded_frames < 1
                or probe_result.width is None
                or probe_result.height is None
                or probe_result.width < 1
                or probe_result.height < 1
            ):
                raise IngestionError(
                    IngestionErrorCode.DECODER_PROBE_FAILED,
                    f"decoder probe produced no usable frame for {channel.topic!r}",
                )
            declared_codec = channel.codec
            if declared_codec is None or probe_result.codec.casefold() != declared_codec.casefold():
                raise IngestionError(
                    IngestionErrorCode.UNSUPPORTED_CODEC,
                    (
                        f"decoder codec {probe_result.codec!r} does not match declared "
                        f"codec {declared_codec!r} for {channel.topic!r}"
                    ),
                )

            first_ns = channel.first_message_time_ns
            last_ns = channel.last_message_time_ns
            assert first_ns is not None and last_ns is not None
            stream_id = _deterministic_id(
                "robata-stream-v1",
                {
                    "source_content_sha256": source_content_sha256,
                    "topic": channel.topic,
                    "channel_id": channel.channel_id,
                    "schema_name": channel.schema_name,
                },
            )
            index = StreamIndex(
                stream_id=stream_id,
                mcap_id=mcap_id,
                topic=channel.topic,
                channel_id=channel.channel_id,
                codec=declared_codec,
                width=probe_result.width,
                height=probe_result.height,
                nominal_fps=(channel.message_count - 1) * 1_000_000_000.0 / (last_ns - first_ns),
                source_start_ns=first_ns,
                source_end_ns=_exclusive_end(last_ns),
                frame_count=channel.message_count,
            )
            stream_indexes.append(index)
            mappings.append(
                CameraMapping(
                    camera_id=camera_id.value,
                    role=camera_id.value,
                    stream_id=stream_id,
                )
            )

        consistency = self.validate_stream_consistency(tuple(stream_indexes))
        if not consistency["valid"]:
            raise IngestionError(
                IngestionErrorCode.INVALID_CAMERA_MAPPING,
                "; ".join(consistency["issues"]),
            )

        mapping_projection = [mapping.model_dump(mode="json") for mapping in mappings]
        mapping_run = CameraMappingRun(
            mapping_run_id=_deterministic_id(
                "robata-camera-mapping-run-v1",
                {
                    "source_content_sha256": source_content_sha256,
                    "mapping_policy_version": version,
                    "mapping_policy_digest": self.mapping_policy_digest,
                    "cameras": mapping_projection,
                },
            ),
            mcap_id=mcap_id,
            mapping_policy_version=version,
            status="PUBLISHED",
            created_at=_rfc3339(self._clock()),
            cameras=tuple(mappings),
        )
        return IngestionResult(
            mcap_id=mcap_id,
            stream_index=tuple(stream_indexes),
            camera_mapping_run=mapping_run,
            frame_indexes=(),
            status="INDEXED",
            indexed_at=_rfc3339(self._clock()),
        )

    def resolve_camera_mapping(
        self,
        inspection: Any,
        mapping_run: CameraMappingRun,
    ) -> tuple[CameraMapping, ...]:
        """Validate and return the run's mapping in canonical camera order."""

        if not isinstance(inspection, McapInspection):
            raise TypeError("inspection must be an McapInspection")
        resolved = self._require_mapping_policy().resolve(inspection)
        by_camera: dict[CameraId, CameraMapping] = {}
        for contract_mapping in mapping_run.cameras:
            mapping = CameraMapping.model_validate(
                contract_mapping.model_dump(mode="python"), strict=True
            )
            try:
                camera_id = CameraId(mapping.camera_id)
            except ValueError as exc:
                raise IngestionError(
                    IngestionErrorCode.INVALID_CAMERA_MAPPING,
                    f"unknown camera slot {mapping.camera_id!r}",
                ) from exc
            if camera_id in by_camera:
                raise IngestionError(
                    IngestionErrorCode.INVALID_CAMERA_MAPPING,
                    f"duplicate camera slot {camera_id.value!r}",
                )
            by_camera[camera_id] = mapping
        if set(by_camera) != set(CAMERA_IDS):
            raise IngestionError(
                IngestionErrorCode.INVALID_CAMERA_MAPPING,
                "mapping run must contain exactly cam_01 through cam_06",
            )

        ordered: list[CameraMapping] = []
        seen_streams: set[str] = set()
        for camera_id in CAMERA_IDS:
            mapping = by_camera[camera_id]
            channel = resolved[camera_id]
            expected_stream_id = _deterministic_id(
                "robata-stream-v1",
                {
                    "source_content_sha256": inspection.source_sha256,
                    "topic": channel.topic,
                    "channel_id": channel.channel_id,
                    "schema_name": channel.schema_name,
                },
            )
            if mapping.stream_id != expected_stream_id:
                raise IngestionError(
                    IngestionErrorCode.INVALID_CAMERA_MAPPING,
                    f"stream identity mismatch for {camera_id.value}",
                )
            if mapping.stream_id in seen_streams:
                raise IngestionError(
                    IngestionErrorCode.INVALID_CAMERA_MAPPING,
                    f"stream {mapping.stream_id!r} fills more than one camera slot",
                )
            seen_streams.add(mapping.stream_id)
            ordered.append(mapping)
        return tuple(ordered)

    def validate_stream_consistency(
        self,
        stream_indexes: tuple[StreamIndex, ...],
    ) -> dict[str, Any]:
        """Return deterministic structural diagnostics for six stream indexes."""

        issues: list[str] = []
        if len(stream_indexes) != 6:
            issues.append(f"expected exactly six streams; got {len(stream_indexes)}")
        _append_duplicate_issue(issues, "stream IDs", (item.stream_id for item in stream_indexes))
        _append_duplicate_issue(issues, "topics", (item.topic for item in stream_indexes))
        _append_duplicate_issue(
            issues,
            "channel IDs",
            (str(item.channel_id) for item in stream_indexes),
        )
        mcap_ids = sorted({item.mcap_id for item in stream_indexes})
        if len(mcap_ids) > 1:
            issues.append(f"streams reference multiple MCAP IDs: {mcap_ids!r}")
        for item in stream_indexes:
            if item.source_end_ns <= item.source_start_ns:
                issues.append(f"stream {item.stream_id!r} has a non-positive timestamp range")
            if item.frame_count < 1:
                issues.append(f"stream {item.stream_id!r} has no frames")
            if not item.codec.strip():
                issues.append(f"stream {item.stream_id!r} has an empty codec")
            if item.nominal_fps <= 0:
                issues.append(f"stream {item.stream_id!r} has a non-positive nominal FPS")
        return {
            "valid": not issues,
            "issues": tuple(issues),
            "stream_count": len(stream_indexes),
            "mcap_ids": tuple(mcap_ids),
        }

    def _inspection_for(self, source: Any) -> McapInspection:
        if isinstance(source, McapInspection):
            return source
        if isinstance(source, (str, Path)):
            if self._inspector is None:
                raise IndexingCapabilityError("an MCAP inspector is required for path sources")
            try:
                return self._inspector.inspect(Path(source))
            except IngestionError:
                raise
            except Exception as exc:
                raise IngestionError(
                    IngestionErrorCode.SOURCE_IO_ERROR,
                    f"MCAP inspection failed: {type(exc).__name__}: {exc}",
                ) from exc
        raise TypeError("source must be an McapInspection or local path")

    def _require_mapping_policy(self) -> CameraMappingPolicy:
        if self._mapping_policy is None:
            raise IndexingCapabilityError("a versioned camera mapping policy is required")
        return self._mapping_policy

    def _require_decoder_probe(self) -> DecoderProbe:
        if self._decoder_probe is None:
            raise IndexingCapabilityError("a decoder probe is required")
        return self._decoder_probe

    def _require_mapping_policy_version(self) -> str:
        value = self._mapping_policy_version
        if not isinstance(value, str) or not value:
            raise IndexingCapabilityError("mapping_policy_version is required")
        return value

    @staticmethod
    def _validate_channel_observation(channel: Any) -> None:
        if channel.message_count < 2:
            raise IngestionError(
                IngestionErrorCode.MISSING_TIMESTAMPS,
                f"channel {channel.topic!r} needs at least two timestamped messages",
            )
        first_ns = channel.first_message_time_ns
        last_ns = channel.last_message_time_ns
        if (
            isinstance(first_ns, bool)
            or not isinstance(first_ns, int)
            or isinstance(last_ns, bool)
            or not isinstance(last_ns, int)
            or last_ns <= first_ns
        ):
            raise IngestionError(
                IngestionErrorCode.MISSING_TIMESTAMPS,
                f"channel {channel.topic!r} has no positive timestamp range",
            )
        if channel.channel_id < 0:
            raise IngestionError(
                IngestionErrorCode.CORRUPT_MCAP,
                f"channel {channel.topic!r} has a negative channel ID",
            )


def _deterministic_id(namespace: str, projection: object) -> str:
    digest = semantic_sha256({"namespace": namespace, "projection": projection})
    return str(uuid5(NAMESPACE_URL, f"{namespace}:{digest}"))


def _exclusive_end(last_ns: int) -> int:
    if last_ns >= 2**63 - 1:
        raise IngestionError(
            IngestionErrorCode.MISSING_TIMESTAMPS,
            "last timestamp cannot be represented as a half-open int64 interval",
        )
    return last_ns + 1


def _append_duplicate_issue(
    issues: list[str],
    label: str,
    values: Iterable[str],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        issues.append(f"duplicate {label}: {sorted(duplicates)!r}")


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "IndexingCapabilityError",
    "StreamIndexer",
]
