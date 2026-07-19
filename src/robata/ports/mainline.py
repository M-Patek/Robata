"""Provider-neutral boundaries for the local mainline vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from robata.contracts.common import INT64_MAX
from robata.contracts.mainline import (
    SamplingPurpose,
    TemporalVisualPackage,
    TemporalWindow,
    VisionInferenceOutcome,
    VisionInferenceRequest,
)

if TYPE_CHECKING:
    from robata.application.registered_video_export import PublishedRegisteredVideoExport


class FrameMaterializationErrorCode(StrEnum):
    """Stable failures exposed by a frame-materialization adapter."""

    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_VIDEO_VIEW = "INVALID_VIDEO_VIEW"
    INVALID_MANIFEST = "INVALID_MANIFEST"
    INVALID_SIDECAR = "INVALID_SIDECAR"
    DECODE_FAILED = "DECODE_FAILED"
    TIMESTAMP_MISMATCH = "TIMESTAMP_MISMATCH"
    PNG_ENCODE_FAILED = "PNG_ENCODE_FAILED"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    OUTPUT_IO_ERROR = "OUTPUT_IO_ERROR"


class FrameMaterializationError(RuntimeError):
    """A frame-materialization failure with a machine-readable code."""

    def __init__(self, code: FrameMaterializationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class FrameMaterializationRequest:
    """Inputs for one deterministic temporal-package materialization."""

    video_export: PublishedRegisteredVideoExport
    output_directory: Path
    window: TemporalWindow
    purpose: SamplingPurpose
    rate_num: int
    rate_den: int = 1
    selection_tolerance_ns: int = 100_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be a Path")
        if not isinstance(self.window, TemporalWindow):
            raise TypeError("window must be a TemporalWindow")
        if not isinstance(self.purpose, SamplingPurpose):
            raise TypeError("purpose must be a SamplingPurpose")
        if self.window.purpose is not self.purpose:
            raise ValueError("request purpose must match window purpose")
        for field, value in (("rate_num", self.rate_num), ("rate_den", self.rate_den)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if value <= 0:
                raise ValueError(f"{field} must be positive")
        tolerance = self.selection_tolerance_ns
        if isinstance(tolerance, bool) or not isinstance(tolerance, int):
            raise TypeError("selection_tolerance_ns must be an integer")
        if tolerance < 0 or tolerance > INT64_MAX:
            raise ValueError("selection_tolerance_ns must be a nonnegative int64")


class FrameMaterializer(Protocol):
    """Materialize selected frames for one window into a provider-neutral package."""

    def materialize(self, request: FrameMaterializationRequest) -> TemporalVisualPackage:
        """Read a verified video view and create one immutable temporal package."""


class VisionModelAdapter(Protocol):
    """Minimal swappable inference boundary used by the local orchestrator."""

    @property
    def provider(self) -> str:
        """Stable provider name used in inference request validation."""

    @property
    def model_name(self) -> str:
        """Provider model name recorded in normalized inference evidence."""

    @property
    def model_version(self) -> str:
        """Pinned provider model version recorded in inference evidence."""

    def infer(
        self,
        request: VisionInferenceRequest,
        package: TemporalVisualPackage,
        artifact_root: Path,
    ) -> VisionInferenceOutcome:
        """Return an outcome with access to package lineage and materialized media."""


__all__ = [
    "FrameMaterializationError",
    "FrameMaterializationErrorCode",
    "FrameMaterializationRequest",
    "FrameMaterializer",
    "VisionModelAdapter",
]
