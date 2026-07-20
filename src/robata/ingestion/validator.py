"""MCAP validation for ingestion.

The validator deliberately consumes observations from ports and never guesses
missing source facts. A readable container is therefore not sufficient for a
PASS result: a mapping and a successful decoder probe are required as well.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from robata.contracts.common import StrictModel, ValidationIssue, ValidationSeverity
from robata.contracts.mcap import (
    CameraStream,
    TerminalErrorCode,
)
from robata.ports.ingestion import (
    CameraMappingPolicy,
    DecoderProbe,
    DecoderProbeResult,
    IngestionError,
    McapInspection,
    McapInspector,
)


class ValidationStatus(StrEnum):
    """High-level result of a validation pass."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class ValidationResult(StrictModel):
    """Immutable result of a single validation pass."""

    status: ValidationStatus
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.PASS and not any(
            issue.severity.value == "ERROR" for issue in self.issues
        )


class MCAPValidator:
    """Validate an MCAP recording against ingestion policy."""

    def __init__(
        self,
        *,
        inspector: McapInspector | None = None,
        mapping_policy: CameraMappingPolicy | None = None,
        decoder_probe: DecoderProbe | None = None,
        container_check: Callable[[str], bool] | None = None,
        validator_version: str = "mcap-validator-v1",
    ) -> None:
        """Create a validator around source-observation ports."""

        if not isinstance(validator_version, str) or not validator_version:
            raise ValueError("validator_version must be a nonempty string")
        self._inspector = inspector
        self._mapping_policy = mapping_policy
        self._decoder_probe = decoder_probe
        self._container_check = container_check
        self._inspection: McapInspection | None = None
        self._source: Path | None = None
        self._probe_results: dict[tuple[str, int], DecoderProbeResult] = {}
        self._validator_version = validator_version

    @property
    def validator_version(self) -> str:
        """Return the immutable implementation/version input to report identity."""

        return self._validator_version

    @property
    def inspection(self) -> McapInspection | None:
        """Return the most recent inspection used by this validator."""

        return self._inspection

    @property
    def probe_results(self) -> dict[tuple[str, int], DecoderProbeResult]:
        """Return decoder evidence keyed by topic and channel."""

        return dict(self._probe_results)

    def set_observation(
        self,
        inspection: McapInspection,
        *,
        source: str | Path | None = None,
    ) -> None:
        """Bind already-observed source facts for a subsequent validation pass."""

        self._inspection = inspection
        self._source = Path(source) if source is not None else inspection.source
        self._probe_results.clear()

    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------
    def validate_container(self, mcap_path: str) -> ValidationResult:
        """Validate that the MCAP container parses without fatal structural error."""
        path = Path(mcap_path)
        if self._container_check is not None:
            try:
                passed = self._container_check(mcap_path)
            except Exception as exc:
                return _result(
                    "INCONCLUSIVE_CONTAINER",
                    f"container check failed: {type(exc).__name__}: {exc}",
                )
            if passed is not True:
                return _result(
                    TerminalErrorCode.CORRUPT_MCAP.value,
                    "container check reported an unreadable MCAP",
                )
            return ValidationResult(status=ValidationStatus.PASS)

        if self._inspector is None:
            return _result(
                "INCONCLUSIVE_CONTAINER_INSPECTOR_UNAVAILABLE",
                "no MCAP inspector is configured",
            )
        try:
            inspection = self._inspector.inspect(path)
        except IngestionError as exc:
            return _result(exc.code.value, str(exc))
        except Exception as exc:
            return _result(
                "INCONCLUSIVE_CONTAINER",
                f"MCAP inspection failed: {type(exc).__name__}: {exc}",
            )
        self.set_observation(inspection, source=path)
        return ValidationResult(status=ValidationStatus.PASS)

    # ------------------------------------------------------------------
    # Camera count validation
    # ------------------------------------------------------------------
    def validate_camera_count(self, streams: tuple[CameraStream, ...]) -> ValidationResult:
        """Validate that exactly six camera streams are present."""
        issues: list[ValidationIssue] = []
        if len(streams) != 6:
            issues.append(
                _issue(
                    TerminalErrorCode.INVALID_CAMERA_COUNT.value,
                    f"exactly six mapped camera streams are required; got {len(streams)}",
                )
            )
        duplicate_stream_ids = _duplicates([stream.stream_id for stream in streams])
        if duplicate_stream_ids:
            issues.append(
                _issue(
                    TerminalErrorCode.INVALID_CAMERA_MAPPING.value,
                    f"duplicate stream IDs: {duplicate_stream_ids!r}",
                )
            )
        duplicate_topics = _duplicates([stream.topic for stream in streams])
        if duplicate_topics:
            issues.append(
                _issue(
                    TerminalErrorCode.INVALID_CAMERA_MAPPING.value,
                    f"duplicate mapped topics: {duplicate_topics!r}",
                )
            )
        channel_ids = [stream.channel_id for stream in streams]
        if len(set(channel_ids)) != len(channel_ids):
            issues.append(
                _issue(
                    TerminalErrorCode.INVALID_CAMERA_MAPPING.value,
                    "duplicate mapped channel IDs",
                )
            )
        return _from_issues(issues)

    # ------------------------------------------------------------------
    # Codec / decodability validation
    # ------------------------------------------------------------------
    def validate_stream_decodability(self, streams: tuple[CameraStream, ...]) -> ValidationResult:
        """Validate that each mapped stream has a supported, successfully probed decoder path."""
        issues: list[ValidationIssue] = []
        if self._decoder_probe is None:
            return _result(
                "INCONCLUSIVE_DECODER_PROBE_UNAVAILABLE",
                "no decoder probe is configured; READY requires successful probes",
            )
        if self._inspection is None or self._source is None:
            return _result(
                "INCONCLUSIVE_DECODER_SOURCE_UNAVAILABLE",
                "decoder probing requires a bound MCAP inspection and source",
            )
        channels = {
            (channel.topic, channel.channel_id): channel for channel in self._inspection.channels
        }
        for stream in streams:
            channel = channels.get((stream.topic, stream.channel_id))
            if channel is None:
                issues.append(
                    _issue(
                        TerminalErrorCode.INVALID_CAMERA_MAPPING.value,
                        f"stream {stream.stream_id!r} is absent from the inspection",
                    )
                )
                continue
            try:
                result = self._decoder_probe.probe(self._source, channel)
            except IngestionError as exc:
                issues.append(_issue(exc.code.value, str(exc), path=(stream.stream_id,)))
                continue
            except Exception as exc:
                issues.append(
                    _issue(
                        "INCONCLUSIVE_DECODER_PROBE",
                        f"decoder probe failed: {type(exc).__name__}: {exc}",
                        path=(stream.stream_id,),
                    )
                )
                continue
            self._probe_results[(stream.topic, stream.channel_id)] = result
            if not result.success or result.decoded_frames < 1:
                detail = "; ".join(failure.message for failure in result.failures)
                issues.append(
                    _issue(
                        "DECODER_PROBE_FAILED",
                        (
                            f"stream {stream.stream_id!r} did not produce a decoded frame"
                            + (f": {detail}" if detail else "")
                        ),
                        path=(stream.stream_id,),
                    )
                )
            elif result.codec.strip().lower() != stream.codec.strip().lower():
                issues.append(
                    _issue(
                        TerminalErrorCode.UNSUPPORTED_CODEC.value,
                        f"decoder codec {result.codec!r} differs from declared {stream.codec!r}",
                        path=(stream.stream_id,),
                    )
                )
        return _from_issues(issues)

    # ------------------------------------------------------------------
    # Timestamp range validation
    # ------------------------------------------------------------------
    def validate_timestamp_ranges(self, streams: tuple[CameraStream, ...]) -> ValidationResult:
        """Validate that each stream exposes a usable, positive timestamp range."""
        issues: list[ValidationIssue] = []
        for stream in streams:
            if stream.source_end_ns <= stream.source_start_ns:
                issues.append(
                    _issue(
                        TerminalErrorCode.ZERO_DURATION.value,
                        f"stream {stream.stream_id!r} has a non-positive timestamp range",
                        path=(stream.stream_id,),
                    )
                )
            if stream.frame_count < 1:
                issues.append(
                    _issue(
                        TerminalErrorCode.MISSING_TIMESTAMPS.value,
                        f"stream {stream.stream_id!r} has no indexed frames",
                        path=(stream.stream_id,),
                    )
                )
        return _from_issues(issues)

    # ------------------------------------------------------------------
    # Aggregate validation
    # ------------------------------------------------------------------
    def validate(self, mcap_path: str, streams: tuple[CameraStream, ...]) -> ValidationResult:
        """Run all validation passes and return a composite result.

        Short-circuits on the first failure and accumulates issues.
        """
        container = self.validate_container(mcap_path)
        if not container.is_valid:
            return container
        results = [
            self.validate_camera_count(streams),
            self.validate_timestamp_ranges(streams),
        ]
        if self._mapping_policy is not None and self._inspection is not None:
            try:
                self._mapping_policy.resolve(self._inspection)
            except IngestionError as exc:
                results.append(_result(exc.code.value, str(exc)))
        elif self._mapping_policy is None:
            results.append(
                _result(
                    "INCONCLUSIVE_MAPPING_POLICY_UNAVAILABLE",
                    "no versioned camera mapping policy is configured",
                )
            )
        results.append(self.validate_stream_decodability(streams))
        issues = tuple(issue for result in results for issue in result.issues)
        return ValidationResult(
            status=_aggregate_status(results),
            issues=issues,
        )


def _issue(
    code: str,
    message: str,
    *,
    path: tuple[str | int, ...] = (),
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        path=path,
        severity=ValidationSeverity.ERROR,
    )


def _result(code: str, message: str) -> ValidationResult:
    return ValidationResult(
        status=(
            ValidationStatus.INCONCLUSIVE if _is_inconclusive_code(code) else ValidationStatus.FAIL
        ),
        issues=(_issue(code, message),),
    )


def _from_issues(issues: list[ValidationIssue]) -> ValidationResult:
    if not issues:
        return ValidationResult(status=ValidationStatus.PASS)
    statuses = tuple(
        ValidationStatus.INCONCLUSIVE
        if _is_inconclusive_code(issue.code)
        else ValidationStatus.FAIL
        for issue in issues
    )
    status = (
        ValidationStatus.FAIL
        if ValidationStatus.FAIL in statuses
        else ValidationStatus.INCONCLUSIVE
    )
    return ValidationResult(status=status, issues=tuple(issues))


def _aggregate_status(results: list[ValidationResult]) -> ValidationStatus:
    if any(result.status is ValidationStatus.FAIL for result in results):
        return ValidationStatus.FAIL
    if any(result.status is ValidationStatus.INCONCLUSIVE for result in results):
        return ValidationStatus.INCONCLUSIVE
    return ValidationStatus.PASS


def _is_inconclusive_code(code: str) -> bool:
    return code.startswith("INCONCLUSIVE_") or code in {
        "SOURCE_NOT_FOUND",
        "SOURCE_IO_ERROR",
    }


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate)


__all__ = [
    "MCAPValidator",
    "ValidationResult",
    "ValidationStatus",
]
