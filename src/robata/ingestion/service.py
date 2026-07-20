"""Fail-closed, dependency-injected MCAP ingestion orchestration."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import NAMESPACE_URL, uuid5

from robata.contracts import recording_identity as derive_recording_identity
from robata.contracts import semantic_sha256
from robata.contracts.cameras import CameraId
from robata.contracts.common import ValidationIssue
from robata.contracts.mcap import (
    CameraStream,
    MCAPManifest,
    MCAPMappingPolicyReference,
    MCAPReadyCamera,
    MCAPReadyRecording,
    MCAPReadySource,
    MCAPRecording,
    MCAPRecordingStatus,
    MCAPValidationError,
    MCAPValidationReport,
    MCAPValidationSource,
    MCAPValidationVerdict,
    TerminalErrorCode,
)
from robata.contracts.schema_registry import SchemaRegistry, default_schema_registry
from robata.ingestion.indexer import IndexingCapabilityError, StreamIndexer
from robata.ingestion.models import IngestionResult
from robata.ingestion.validator import MCAPValidator, ValidationResult, ValidationStatus
from robata.ports.ingestion import IngestionError, IngestionErrorCode, McapInspection, McapInspector

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MCAP_MANIFEST_SCHEMA_ID = "https://schemas.robata.dev/mcap-manifest"
_MCAP_VALIDATION_REPORT_SCHEMA_ID = "https://schemas.robata.dev/mcap-validation-report"
_SCHEMA_VERSION = "1.0.0"


class IngestionStateError(RuntimeError):
    """Raised when a command violates the ingestion state machine."""


class IngestionCapabilityError(RuntimeError):
    """Raised when a required source or durability adapter is absent."""


SourceHasher = Callable[[str], tuple[str, int]]
SourceResolver = Callable[[str], Path]
DurabilityCheck = Callable[[MCAPRecording], bool]


_ALLOWED_TRANSITIONS: dict[MCAPRecordingStatus, frozenset[MCAPRecordingStatus]] = {
    MCAPRecordingStatus.DISCOVERED: frozenset(
        {MCAPRecordingStatus.HASHING, MCAPRecordingStatus.RETRY_WAIT, MCAPRecordingStatus.FAILED}
    ),
    MCAPRecordingStatus.HASHING: frozenset(
        {MCAPRecordingStatus.INSPECTING, MCAPRecordingStatus.RETRY_WAIT, MCAPRecordingStatus.FAILED}
    ),
    MCAPRecordingStatus.INSPECTING: frozenset(
        {
            MCAPRecordingStatus.VALIDATING,
            MCAPRecordingStatus.INVALID,
            MCAPRecordingStatus.RETRY_WAIT,
            MCAPRecordingStatus.FAILED,
        }
    ),
    MCAPRecordingStatus.VALIDATING: frozenset(
        {
            MCAPRecordingStatus.READY,
            MCAPRecordingStatus.INVALID,
            MCAPRecordingStatus.RETRY_WAIT,
            MCAPRecordingStatus.FAILED,
        }
    ),
    MCAPRecordingStatus.READY: frozenset({MCAPRecordingStatus.ALIGNMENT_QUEUED}),
    MCAPRecordingStatus.RETRY_WAIT: frozenset(
        {
            MCAPRecordingStatus.HASHING,
            MCAPRecordingStatus.INSPECTING,
            MCAPRecordingStatus.VALIDATING,
            MCAPRecordingStatus.FAILED,
        }
    ),
    MCAPRecordingStatus.ALIGNMENT_QUEUED: frozenset(),
    MCAPRecordingStatus.INVALID: frozenset(),
    MCAPRecordingStatus.FAILED: frozenset(),
}


class MCAPIngestionService:
    """Orchestrate one MCAP through ``DISCOVERED`` to ``READY`` or quarantine.

    This service is deliberately storage-neutral. Every operation returns a new
    frozen contract snapshot. Durable persistence and source authorization stay
    behind injected ports; publication is unavailable when durability cannot be
    proven.
    """

    def __init__(
        self,
        *,
        source_hasher: SourceHasher | None = None,
        inspector: McapInspector | None = None,
        validator: MCAPValidator | None = None,
        indexer: StreamIndexer | None = None,
        source_resolver: SourceResolver | None = None,
        source_durable_check: DurabilityCheck | None = None,
        schema_registry: SchemaRegistry | None = None,
        max_retries: int = 3,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self._source_hasher = source_hasher
        self._inspector = inspector
        self._validator = validator
        self._indexer = indexer
        self._source_resolver = source_resolver
        self._source_durable_check = source_durable_check
        self._schema_registry = schema_registry or default_schema_registry()
        self._max_retries = max_retries
        self._clock = clock or (lambda: datetime.now(UTC))
        self._state: MCAPRecordingStatus | None = None
        self._active_mcap_id: str | None = None
        self._inspection: McapInspection | None = None
        self._index_result: IngestionResult | None = None
        self._validation_result: ValidationResult | None = None
        self._validation_checks: tuple[tuple[str, ValidationResult], ...] = ()
        self._validation_report: MCAPValidationReport | None = None
        self._ready_manifest: MCAPManifest | None = None
        self._ready_recording: MCAPRecording | None = None
        self._validated_at: str | None = None
        self._mapping_policy_version: str | None = None
        self._mapping_policy_digest: str | None = None
        self._retry_count = 0
        self._retry_resume_state: MCAPRecordingStatus | None = None
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        """Return the last retryable diagnostic without changing source status."""

        return self._last_error

    @property
    def validation_result(self) -> ValidationResult | None:
        """Return the latest immutable validation evidence."""

        return self._validation_result

    @property
    def index_result(self) -> IngestionResult | None:
        """Return the latest immutable stream-index result."""

        return self._index_result

    @property
    def validation_report(self) -> MCAPValidationReport | None:
        """Return the report published for the current validation pass, if any."""

        return self._validation_report

    @property
    def ready_recording(self) -> MCAPRecording | None:
        """Return the READY snapshot created atomically with manifest publication."""

        return self._ready_recording

    def transition_to(self, new_state: MCAPRecordingStatus) -> None:
        """Advance the in-memory state only along an authoritative edge."""

        if not isinstance(new_state, MCAPRecordingStatus):
            raise TypeError("new_state must be an MCAPRecordingStatus")
        if self._state is None:
            if new_state is not MCAPRecordingStatus.DISCOVERED:
                raise IngestionStateError("the first state must be DISCOVERED")
        elif new_state is not self._state and new_state not in _ALLOWED_TRANSITIONS[self._state]:
            raise IngestionStateError(
                f"invalid ingestion transition {self._state.value} -> {new_state.value}"
            )
        self._state = new_state

    def get_state(self) -> MCAPRecordingStatus | None:
        """Return the current ingestion state, or ``None`` before discovery."""

        return self._state

    def discover(
        self,
        source_uri: str,
        observed_size_bytes: int,
        *,
        source_version: str = "unversioned",
    ) -> MCAPRecording:
        """Create a deterministic ``DISCOVERED`` snapshot for one notification."""

        if self._state is not None:
            raise IngestionStateError("this service instance already has an active recording")
        if not isinstance(source_uri, str) or not source_uri:
            raise ValueError("source_uri must be a nonempty string")
        if not isinstance(source_version, str) or not source_version:
            raise ValueError("source_version must be a nonempty string")
        if (
            isinstance(observed_size_bytes, bool)
            or not isinstance(observed_size_bytes, int)
            or observed_size_bytes < 0
        ):
            raise ValueError("observed_size_bytes must be a nonnegative integer")
        alias_projection = {
            "source_uri": source_uri,
            "source_version": source_version,
        }
        mcap_id = _deterministic_id("robata-mcap-notification-v1", alias_projection)
        self.transition_to(MCAPRecordingStatus.DISCOVERED)
        self._active_mcap_id = mcap_id
        timestamp = _rfc3339(self._clock())
        return MCAPRecording(
            mcap_id=mcap_id,
            recording_identity=f"pending:{mcap_id}",
            source_artifact_id=_deterministic_id(
                "robata-pending-source-artifact-v1", alias_projection
            ),
            source_uri=source_uri,
            source_version=source_version,
            content_sha256="pending",
            observed_size_bytes=observed_size_bytes,
            duration_ns=0,
            timebase="unknown",
            camera_count=0,
            raw_video_stream_count=0,
            status=MCAPRecordingStatus.DISCOVERED,
            ingested_at=timestamp,
        )

    def hash_source(self, recording: MCAPRecording) -> MCAPRecording:
        """Hash through the configured port and verify notification size."""

        self._ensure_active(
            recording, {MCAPRecordingStatus.DISCOVERED, MCAPRecordingStatus.HASHING}
        )
        if self._source_hasher is None:
            raise IngestionCapabilityError("source_hasher is required before source access")
        if recording.status is MCAPRecordingStatus.DISCOVERED:
            self.transition_to(MCAPRecordingStatus.HASHING)
            recording = recording.model_copy(update={"status": MCAPRecordingStatus.HASHING})
        try:
            digest, actual_size = self._source_hasher(recording.source_uri)
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError("source hasher returned a non-canonical SHA-256 digest")
            if isinstance(actual_size, bool) or not isinstance(actual_size, int) or actual_size < 0:
                raise ValueError("source hasher returned an invalid byte size")
            if actual_size != recording.observed_size_bytes:
                raise OSError(
                    "source size changed: "
                    f"notified={recording.observed_size_bytes}, actual={actual_size}"
                )
        except Exception as exc:
            self._last_error = f"hashing failed: {type(exc).__name__}: {exc}"
            return self.retry_or_fail(recording)
        self._last_error = None
        return recording.model_copy(
            update={
                "recording_identity": derive_recording_identity("robata-recording-v1", digest),
                "source_artifact_id": _deterministic_id(
                    "robata-source-artifact-v1", {"content_sha256": digest}
                ),
                "content_sha256": digest,
                "observed_size_bytes": actual_size,
            }
        )

    def inspect(self, recording: MCAPRecording) -> MCAPRecording:
        """Inspect container facts and verify them against the hash stage."""

        self._ensure_active(recording, {MCAPRecordingStatus.HASHING})
        if recording.content_sha256 == "pending":
            raise IngestionStateError("source content must be hashed before inspection")
        if self._inspector is None:
            raise IngestionCapabilityError("an MCAP inspector is required")
        source = self._resolve_source(recording.source_uri)
        self.transition_to(MCAPRecordingStatus.INSPECTING)
        inspecting = recording.model_copy(update={"status": MCAPRecordingStatus.INSPECTING})
        try:
            inspection = self._inspector.inspect(source)
            if inspection.source_sha256 != recording.content_sha256:
                raise OSError("inspection digest differs from the verified source digest")
            if inspection.source_size_bytes != recording.observed_size_bytes:
                raise OSError("inspection byte size differs from the verified source size")
        except IngestionError as exc:
            self._last_error = str(exc)
            if exc.code is IngestionErrorCode.CORRUPT_MCAP:
                return self.quarantine(inspecting, TerminalErrorCode.CORRUPT_MCAP)
            return self.retry_or_fail(inspecting)
        except Exception as exc:
            self._last_error = f"inspection failed: {type(exc).__name__}: {exc}"
            return self.retry_or_fail(inspecting)
        self._inspection = inspection
        self._last_error = None
        first_ns = inspection.first_message_time_ns
        last_ns = inspection.last_message_time_ns
        duration_ns = (
            last_ns - first_ns
            if isinstance(first_ns, int) and isinstance(last_ns, int) and last_ns > first_ns
            else 0
        )
        raw_video_count = sum(
            channel.codec is not None or channel.schema_name == "foxglove.CompressedImage"
            for channel in inspection.channels
        )
        return inspecting.model_copy(
            update={
                "duration_ns": duration_ns,
                "timebase": "mcap_log_time",
                "raw_video_stream_count": raw_video_count,
            }
        )

    def validate_mcap(self, recording: MCAPRecording) -> MCAPRecording:
        """Validate observed structure, exact mapping, decoder paths, and timestamps."""

        self._ensure_active(
            recording,
            {MCAPRecordingStatus.INSPECTING, MCAPRecordingStatus.VALIDATING},
        )
        if self._inspection is None:
            raise IngestionStateError("inspection evidence is unavailable")
        if self._validator is None or self._indexer is None:
            raise IngestionCapabilityError("validator and stream indexer are required")
        if recording.status is MCAPRecordingStatus.INSPECTING:
            self.transition_to(MCAPRecordingStatus.VALIDATING)
            validating = recording.model_copy(update={"status": MCAPRecordingStatus.VALIDATING})
        else:
            validating = recording
        try:
            self._mapping_policy_version = self._indexer.mapping_policy_version
            self._mapping_policy_digest = self._indexer.mapping_policy_digest
            index_result = self._indexer.index_streams(recording.mcap_id, self._inspection)
            streams = tuple(
                CameraStream(
                    stream_id=item.stream_id,
                    topic=item.topic,
                    channel_id=item.channel_id,
                    codec=item.codec,
                    width=item.width,
                    height=item.height,
                    nominal_fps=item.nominal_fps,
                    source_start_ns=item.source_start_ns,
                    source_end_ns=item.source_end_ns,
                    frame_count=item.frame_count,
                )
                for item in index_result.stream_index
            )
            self._validator.set_observation(self._inspection, source=self._inspection.source)
            validation_checks = (
                ("camera_count", self._validator.validate_camera_count(streams)),
                ("timestamp_ranges", self._validator.validate_timestamp_ranges(streams)),
                ("decoder_paths", self._validator.validate_stream_decodability(streams)),
            )
            result = _aggregate_validation_checks(validation_checks)
            self._index_result = index_result
        except IndexingCapabilityError as exc:
            result = _exception_validation_result(
                "INCONCLUSIVE_INDEXING_CAPABILITY", str(exc), inconclusive=True
            )
            self._capture_validation(result, (("stream_indexing", result),))
            self._last_error = str(exc)
            return self.retry_or_fail(validating)
        except IngestionError as exc:
            inconclusive = exc.code in {
                IngestionErrorCode.SOURCE_NOT_FOUND,
                IngestionErrorCode.SOURCE_IO_ERROR,
            }
            result = _exception_validation_result(
                exc.code.value,
                str(exc),
                inconclusive=inconclusive,
            )
            self._capture_validation(result, (("stream_indexing", result),))
            self._last_error = str(exc)
            if inconclusive:
                return self.retry_or_fail(validating)
            terminal = _terminal_code_for_ingestion_error(exc.code)
            return self.quarantine(validating, terminal)
        except Exception as exc:
            self._last_error = f"validation failed: {type(exc).__name__}: {exc}"
            result = _exception_validation_result(
                "INCONCLUSIVE_VALIDATION_RUNTIME",
                self._last_error,
                inconclusive=True,
            )
            self._capture_validation(result, (("validation_runtime", result),))
            return self.retry_or_fail(validating)

        self._capture_validation(result, validation_checks)
        if not result.is_valid:
            if result.status is ValidationStatus.INCONCLUSIVE:
                self._last_error = "; ".join(issue.message for issue in result.issues)
                return self.retry_or_fail(validating)
            return self.quarantine(validating, _terminal_code_for_issues(result))
        if recording.duration_ns <= 0:
            result = _exception_validation_result(
                TerminalErrorCode.ZERO_DURATION.value,
                "recording has a non-positive duration",
                inconclusive=False,
            )
            checks = (*validation_checks, ("recording_duration", result))
            self._capture_validation(_aggregate_validation_checks(checks), checks)
            return self.quarantine(validating, TerminalErrorCode.ZERO_DURATION)
        self._last_error = None
        return validating.model_copy(
            update={
                "camera_count": 6,
                "status": MCAPRecordingStatus.VALIDATING,
                "error_code": None,
            }
        )

    def publish_validation_report(self, recording: MCAPRecording) -> MCAPValidationReport:
        """Publish immutable validation evidence under the exact registered schema."""

        if self._state is MCAPRecordingStatus.READY:
            if (
                self._validation_report is None
                or self._ready_recording is None
                or recording.mcap_id != self._ready_recording.mcap_id
                or recording.recording_identity != self._ready_recording.recording_identity
            ):
                raise IngestionStateError(
                    "validation report retry does not match READY publication"
                )
            return self._validation_report
        self._ensure_active(
            recording,
            {
                MCAPRecordingStatus.VALIDATING,
                MCAPRecordingStatus.READY,
                MCAPRecordingStatus.INVALID,
                MCAPRecordingStatus.RETRY_WAIT,
                MCAPRecordingStatus.FAILED,
            },
        )
        if self._validation_result is None or self._validated_at is None:
            raise IngestionStateError("validation evidence is unavailable")
        if self._validator is None:
            raise IngestionCapabilityError("validator metadata is unavailable")
        if self._mapping_policy_version is None or self._mapping_policy_digest is None:
            raise IngestionCapabilityError("mapping-policy identity is unavailable")
        registered = self._schema_registry.resolve_version(
            _MCAP_VALIDATION_REPORT_SCHEMA_ID,
            _SCHEMA_VERSION,
        )
        errors = tuple(
            _report_error(issue, self._index_result) for issue in self._validation_result.issues
        )
        verdict = {
            ValidationStatus.PASS: MCAPValidationVerdict.VALID,
            ValidationStatus.FAIL: MCAPValidationVerdict.INVALID,
            ValidationStatus.INCONCLUSIVE: MCAPValidationVerdict.INCONCLUSIVE,
        }[self._validation_result.status]
        report_source = MCAPValidationSource(
            uri=recording.source_uri,
            version=recording.source_version,
            sha256=recording.content_sha256,
            bytes=recording.observed_size_bytes,
        )
        mapping_policy = MCAPMappingPolicyReference(
            version=self._mapping_policy_version,
            digest=self._mapping_policy_digest,
        )
        mapped_camera_count = (
            len(self._index_result.camera_mapping_run.cameras)
            if self._index_result is not None
            else 0
        )
        report_projection = {
            "schema_version": "1.0",
            "mcap_id": recording.mcap_id,
            "recording_identity": recording.recording_identity,
            "source": report_source.model_dump(mode="json"),
            "mapping_policy": mapping_policy.model_dump(mode="json"),
            "verdict": verdict.value,
            "discovered_video_stream_count": recording.raw_video_stream_count,
            "mapped_camera_count": mapped_camera_count,
            "errors": [error.model_dump(mode="json") for error in errors],
            "validated_at": self._validated_at,
        }
        evidence_projection = {
            "report": report_projection,
            "validator_version": self._validator.validator_version,
            "schema_ref": registered.ref.model_dump(mode="json"),
            "checks": [
                {"name": name, "result": result.model_dump(mode="json")}
                for name, result in self._validation_checks
            ],
            "probed_stream_facts": _probe_fact_projection(self._validator),
        }
        report = MCAPValidationReport(
            schema_version="1.0",
            validation_report_id=_deterministic_id(
                "robata-mcap-validation-report-v1", evidence_projection
            ),
            mcap_id=recording.mcap_id,
            recording_identity=recording.recording_identity,
            source=report_source,
            mapping_policy=mapping_policy,
            verdict=verdict,
            discovered_video_stream_count=recording.raw_video_stream_count,
            mapped_camera_count=mapped_camera_count,
            errors=errors,
            validated_at=self._validated_at,
        )
        self._schema_registry.validate_pinned(
            registered.ref,
            report.model_dump(mode="json"),
        )
        if self._validation_report is not None and report != self._validation_report:
            raise IngestionStateError("validation report changed after publication")
        self._validation_report = report
        return report

    def publish_ready_manifest(
        self,
        recording: MCAPRecording,
        *,
        selected_validation_report: MCAPValidationReport,
    ) -> MCAPManifest:
        """Publish READY only from an explicitly selected VALID report."""

        if self._state is MCAPRecordingStatus.READY:
            if (
                self._ready_manifest is None
                or self._ready_recording is None
                or self._validation_report is None
                or selected_validation_report != self._validation_report
                or recording.mcap_id != self._ready_recording.mcap_id
                or recording.recording_identity != self._ready_recording.recording_identity
            ):
                raise IngestionStateError("READY publication retry does not match prior evidence")
            return self._ready_manifest
        self._ensure_active(recording, {MCAPRecordingStatus.VALIDATING})
        if self._index_result is None or self._validation_result is None:
            raise IngestionStateError("selected index and validation evidence are unavailable")
        if not self._validation_result.is_valid:
            raise IngestionStateError("invalid validation evidence cannot publish READY")
        if self._validation_report is None or selected_validation_report != self._validation_report:
            raise IngestionStateError("selected validation report was not published by this run")
        report_registration = self._schema_registry.resolve_version(
            _MCAP_VALIDATION_REPORT_SCHEMA_ID,
            _SCHEMA_VERSION,
        )
        self._schema_registry.validate_pinned(
            report_registration.ref,
            selected_validation_report.model_dump(mode="json"),
        )
        if selected_validation_report.verdict is not MCAPValidationVerdict.VALID:
            raise IngestionStateError("only a VALID report can publish READY")
        if (
            selected_validation_report.mcap_id != recording.mcap_id
            or selected_validation_report.recording_identity != recording.recording_identity
            or selected_validation_report.source.sha256 != recording.content_sha256
        ):
            raise IngestionStateError("selected report does not describe this source content")
        if self._source_durable_check is None:
            raise IngestionCapabilityError("source durability check is required for publication")
        try:
            durable = self._source_durable_check(recording)
        except Exception as exc:
            raise IngestionCapabilityError(
                f"source durability check failed: {type(exc).__name__}: {exc}"
            ) from exc
        if durable is not True:
            raise IngestionCapabilityError("source artifact durability was not proven")

        streams_by_id = {item.stream_id: item for item in self._index_result.stream_index}
        cameras: list[MCAPReadyCamera] = []
        for mapping in self._index_result.camera_mapping_run.cameras:
            item = streams_by_id.get(mapping.stream_id)
            if item is None:
                raise IngestionStateError("camera mapping references an unknown stream")
            cameras.append(
                MCAPReadyCamera(
                    camera_id=CameraId(mapping.camera_id),
                    role=mapping.role,
                    stream_id=item.stream_id,
                    topic=item.topic,
                    channel_id=item.channel_id,
                    codec=item.codec,
                    width=item.width,
                    height=item.height,
                    nominal_fps=item.nominal_fps,
                    source_start_ns=item.source_start_ns,
                    source_end_ns=item.source_end_ns,
                    frame_count=item.frame_count,
                )
            )
        if len(cameras) != 6:
            raise IngestionStateError("READY publication requires exactly six cameras")
        mapping_run = self._index_result.camera_mapping_run
        if mapping_run.status != "PUBLISHED":
            raise IngestionStateError("READY requires a published camera mapping revision")
        manifest = MCAPManifest(
            schema_version="1.0",
            mcap_id=recording.mcap_id,
            validation_report_id=selected_validation_report.validation_report_id,
            source=MCAPReadySource(
                uri=recording.source_uri,
                version=recording.source_version,
                sha256=recording.content_sha256,
                bytes=recording.observed_size_bytes,
            ),
            recording=MCAPReadyRecording(
                start_utc=recording.start_utc,
                end_utc=recording.end_utc,
                duration_ns=recording.duration_ns,
                timebase=recording.timebase,
            ),
            camera_count=6,
            camera_mapping_run_id=mapping_run.mapping_run_id,
            camera_mapping_version=mapping_run.mapping_policy_version,
            cameras=tuple(cameras),
            ingested_at=recording.ingested_at,
        )
        manifest_registration = self._schema_registry.resolve_version(
            _MCAP_MANIFEST_SCHEMA_ID,
            _SCHEMA_VERSION,
        )
        self._schema_registry.validate_pinned(
            manifest_registration.ref,
            manifest.model_dump(mode="json"),
        )
        ready_recording = recording.model_copy(
            update={
                "camera_count": 6,
                "status": MCAPRecordingStatus.READY,
                "error_code": None,
            }
        )
        self.transition_to(MCAPRecordingStatus.READY)
        self._ready_manifest = manifest
        self._ready_recording = ready_recording
        return manifest

    def _capture_validation(
        self,
        result: ValidationResult,
        checks: tuple[tuple[str, ValidationResult], ...],
    ) -> None:
        self._validation_result = result
        self._validation_checks = checks
        self._validation_report = None
        self._ready_manifest = None
        self._ready_recording = None
        self._validated_at = _rfc3339(self._clock())

    def quarantine(self, recording: MCAPRecording, error_code: TerminalErrorCode) -> MCAPRecording:
        """Mark invalid source data terminally without scheduling a retry."""

        self._ensure_active(
            recording,
            {MCAPRecordingStatus.INSPECTING, MCAPRecordingStatus.VALIDATING},
        )
        if not isinstance(error_code, TerminalErrorCode):
            raise TypeError("error_code must be a TerminalErrorCode")
        self.transition_to(MCAPRecordingStatus.INVALID)
        return recording.model_copy(
            update={"status": MCAPRecordingStatus.INVALID, "error_code": error_code}
        )

    def retry_or_fail(self, recording: MCAPRecording) -> MCAPRecording:
        """Move retryable infrastructure failures through ``RETRY_WAIT``."""

        self._ensure_active(
            recording,
            {
                MCAPRecordingStatus.DISCOVERED,
                MCAPRecordingStatus.HASHING,
                MCAPRecordingStatus.INSPECTING,
                MCAPRecordingStatus.VALIDATING,
                MCAPRecordingStatus.RETRY_WAIT,
            },
        )
        if recording.status is MCAPRecordingStatus.RETRY_WAIT:
            if self._retry_resume_state is None:
                raise IngestionStateError("retry resume state is unavailable")
            self.transition_to(self._retry_resume_state)
            return recording.model_copy(update={"status": self._retry_resume_state})

        if self._retry_count >= self._max_retries:
            self.transition_to(MCAPRecordingStatus.FAILED)
            return recording.model_copy(
                update={"status": MCAPRecordingStatus.FAILED, "error_code": None}
            )
        self._retry_count += 1
        self._retry_resume_state = recording.status
        self.transition_to(MCAPRecordingStatus.RETRY_WAIT)
        return recording.model_copy(
            update={"status": MCAPRecordingStatus.RETRY_WAIT, "error_code": None}
        )

    def _ensure_active(
        self,
        recording: MCAPRecording,
        allowed_states: set[MCAPRecordingStatus],
    ) -> None:
        if not isinstance(recording, MCAPRecording):
            raise TypeError("recording must be an MCAPRecording")
        if self._active_mcap_id != recording.mcap_id:
            raise IngestionStateError("recording does not belong to this service instance")
        if recording.status not in allowed_states:
            expected = ", ".join(sorted(state.value for state in allowed_states))
            raise IngestionStateError(
                f"recording state {recording.status.value} is not one of: {expected}"
            )
        if self._state is not recording.status:
            raise IngestionStateError("recording snapshot is stale for the current service state")

    def _resolve_source(self, source_uri: str) -> Path:
        if self._source_resolver is not None:
            path = self._source_resolver(source_uri)
            if not isinstance(path, Path):
                raise IngestionCapabilityError("source_resolver must return pathlib.Path")
            return path
        parsed = urlparse(source_uri)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path.lstrip("/")))
        if parsed.scheme:
            raise IngestionCapabilityError(
                f"source resolver is required for URI scheme {parsed.scheme!r}"
            )
        return Path(source_uri)


def _terminal_code_for_ingestion_error(code: IngestionErrorCode) -> TerminalErrorCode:
    direct = {
        IngestionErrorCode.CORRUPT_MCAP: TerminalErrorCode.CORRUPT_MCAP,
        IngestionErrorCode.INVALID_CAMERA_MAPPING: TerminalErrorCode.INVALID_CAMERA_MAPPING,
        IngestionErrorCode.UNSUPPORTED_CODEC: TerminalErrorCode.UNSUPPORTED_CODEC,
        IngestionErrorCode.MISSING_TIMESTAMPS: TerminalErrorCode.MISSING_TIMESTAMPS,
        IngestionErrorCode.DECODER_PROBE_FAILED: TerminalErrorCode.UNSUPPORTED_CODEC,
    }
    return direct.get(code, TerminalErrorCode.CORRUPT_MCAP)


def _aggregate_validation_checks(
    checks: tuple[tuple[str, ValidationResult], ...],
) -> ValidationResult:
    issues = tuple(issue for _, result in checks for issue in result.issues)
    if any(result.status is ValidationStatus.FAIL for _, result in checks):
        status = ValidationStatus.FAIL
    elif any(result.status is ValidationStatus.INCONCLUSIVE for _, result in checks):
        status = ValidationStatus.INCONCLUSIVE
    else:
        status = ValidationStatus.PASS
    return ValidationResult(status=status, issues=issues)


def _exception_validation_result(
    code: str,
    message: str,
    *,
    inconclusive: bool,
) -> ValidationResult:
    return ValidationResult(
        status=(ValidationStatus.INCONCLUSIVE if inconclusive else ValidationStatus.FAIL),
        issues=(ValidationIssue(code=code, message=message),),
    )


def _report_error(
    issue: ValidationIssue,
    index_result: IngestionResult | None,
) -> MCAPValidationError:
    stream_id: str | None = None
    camera_id: CameraId | None = None
    if index_result is not None:
        known_streams = {stream.stream_id for stream in index_result.stream_index}
        stream_id = next(
            (part for part in issue.path if isinstance(part, str) and part in known_streams),
            None,
        )
        if stream_id is not None:
            camera_by_stream = {
                mapping.stream_id: CameraId(mapping.camera_id)
                for mapping in index_result.camera_mapping_run.cameras
            }
            camera_id = camera_by_stream.get(stream_id)
    path = None
    if issue.path:
        pieces = [f"[{part}]" if isinstance(part, int) else f".{part}" for part in issue.path]
        path = "$" + "".join(pieces)
    return MCAPValidationError(
        code=issue.code,
        message=issue.message,
        path=path,
        camera_id=camera_id,
        stream_id=stream_id,
    )


def _probe_fact_projection(validator: MCAPValidator) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    for (topic, channel_id), result in sorted(validator.probe_results.items()):
        facts.append(
            {
                "topic": topic,
                "channel_id": channel_id,
                "codec": result.codec,
                "success": result.success,
                "width": result.width,
                "height": result.height,
                "first_decoded_timestamp_ns": result.first_decoded_timestamp_ns,
                "messages_examined": result.messages_examined,
                "decoded_frames": result.decoded_frames,
                "failures": [
                    {
                        "code": failure.code,
                        "timestamp_ns": failure.timestamp_ns,
                        "message": failure.message,
                    }
                    for failure in result.failures
                ],
            }
        )
    return facts


def _terminal_code_for_issues(result: ValidationResult) -> TerminalErrorCode:
    codes = {issue.code for issue in result.issues}
    for code in TerminalErrorCode:
        if code.value in codes:
            return code
    if "DECODER_PROBE_FAILED" in codes:
        return TerminalErrorCode.UNSUPPORTED_CODEC
    return TerminalErrorCode.CORRUPT_MCAP


def _deterministic_id(namespace: str, projection: object) -> str:
    digest = semantic_sha256({"namespace": namespace, "projection": projection})
    return str(uuid5(NAMESPACE_URL, f"{namespace}:{digest}"))


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "IngestionCapabilityError",
    "IngestionStateError",
    "MCAPIngestionService",
]
