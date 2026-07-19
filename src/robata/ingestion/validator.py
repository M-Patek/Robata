"""MCAP validation for ingestion."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from robata.contracts.common import StrictModel, ValidationIssue, ValidationOutcome
from robata.contracts.mcap import (
    CameraStream,
    TerminalErrorCode,
)


class ValidationStatus(StrEnum):
    """High-level result of a validation pass."""

    PASS = "PASS"
    FAIL = "FAIL"


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

    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------
    def validate_container(self, mcap_path: str) -> ValidationResult:
        """Validate that the MCAP container parses without fatal structural error."""
        raise NotImplementedError(
            "validate_container() requires an MCAP parser / inspector adapter."
        )

    # ------------------------------------------------------------------
    # Camera count validation
    # ------------------------------------------------------------------
    def validate_camera_count(self, streams: tuple[CameraStream, ...]) -> ValidationResult:
        """Validate that exactly six camera streams are present."""
        raise NotImplementedError(
            "validate_camera_count() requires stream inventory from MCAP inspection."
        )

    # ------------------------------------------------------------------
    # Codec / decodability validation
    # ------------------------------------------------------------------
    def validate_stream_decodability(self, streams: tuple[CameraStream, ...]) -> ValidationResult:
        """Validate that each mapped stream has a supported, successfully probed decoder path."""
        raise NotImplementedError(
            "validate_stream_decodability() requires a DecoderProbe adapter."
        )

    # ------------------------------------------------------------------
    # Timestamp range validation
    # ------------------------------------------------------------------
    def validate_timestamp_ranges(self, streams: tuple[CameraStream, ...]) -> ValidationResult:
        """Validate that each stream exposes a usable, positive timestamp range."""
        raise NotImplementedError(
            "validate_timestamp_ranges() requires stream metadata from MCAP inspection."
        )

    # ------------------------------------------------------------------
    # Aggregate validation
    # ------------------------------------------------------------------
    def validate(self, mcap_path: str, streams: tuple[CameraStream, ...]) -> ValidationResult:
        """Run all validation passes and return a composite result.

        Short-circuits on the first failure and accumulates issues.
        """
        raise NotImplementedError(
            "validate() requires all sub-validations and a mapping policy to be wired."
        )


__all__ = [
    "MCAPValidator",
    "ValidationResult",
    "ValidationStatus",
]
