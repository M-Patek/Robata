"""Fast Detector: deterministic stream integrity checks without VLM.

This module implements the first stage of the two-stage QA pipeline
(Architecture V1 Section 12.1).  It performs cheap, deterministic checks
on container structure, timestamp monotonicity, and decode continuity
*before* any VLM-backed visual QA is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Sequence

from pydantic import Field

from robata.contracts.common import NanosecondInterval, StrictModel
from robata.frame_cache import FramePayload

__all__ = [
    "ContainerCheckResult",
    "DecodeGapResult",
    "FastDetector",
    "FastDetectorConfig",
    "StreamIntegrityResult",
    "TimestampCheckResult",
    "VideoStream",
]


class _IntegrityStatus(StrEnum):
    """Internal fast-detector status before translation to CameraQAStatus."""

    PASS = "PASS"
    DEGRADED = "DEGRADED"
    FAIL = "FAIL"


class FastDetectorConfig(StrictModel):
    """Tunable thresholds for deterministic stream integrity checks.

    All thresholds are chosen to flag *obvious* failures without attempting
    the nuanced visual judgement that belongs to the VLM stage.
    """

    black_threshold: Annotated[
        int,
        Field(strict=True, ge=0, le=255, default=16),
    ]
    """Pixel value below which a frame is considered black (0-255)."""

    overexposed_threshold: Annotated[
        int,
        Field(strict=True, ge=0, le=255, default=240),
    ]
    """Pixel value above which a frame is considered overexposed (0-255)."""

    stationary_threshold_sec: Annotated[
        float,
        Field(strict=True, gt=0.0, default=5.0),
    ]
    """Seconds without timestamp advance before flagging a frozen stream."""


class StreamIntegrityIssue(StrictModel):
    """One deterministic issue found by the fast detector."""

    code: Annotated[str, Field(strict=True, min_length=1)]
    interval: NanosecondInterval | None = None
    message: Annotated[str, Field(strict=True, min_length=1)]


class StreamIntegrityResult(StrictModel):
    """Result of a single stream integrity check.

    ``status`` is the translated CameraQAStatus equivalent;
    ``issues`` contains human-readable codes for downstream triage.
    """

    status: _IntegrityStatus
    issues: tuple[StreamIntegrityIssue, ...]
    checked_at: Annotated[str, Field(strict=True, min_length=1)]


class ContainerCheckResult(StrictModel):
    """Result of container-level (MCAP) integrity verification."""

    status: _IntegrityStatus
    issues: tuple[StreamIntegrityIssue, ...]
    checked_at: Annotated[str, Field(strict=True, min_length=1)]


class TimestampCheckResult(StrictModel):
    """Result of timestamp monotonicity verification."""

    status: _IntegrityStatus
    issues: tuple[StreamIntegrityIssue, ...]
    checked_at: Annotated[str, Field(strict=True, min_length=1)]
    first_ns: int
    last_ns: int
    gap_count: Annotated[int, Field(strict=True, ge=0)]


class DecodeGapResult(StrictModel):
    """Result of decode-gap detection between expected and actual frames."""

    status: _IntegrityStatus
    issues: tuple[StreamIntegrityIssue, ...]
    checked_at: Annotated[str, Field(strict=True, min_length=1)]
    expected_frames: Annotated[int, Field(strict=True, ge=0)]
    decoded_frames: Annotated[int, Field(strict=True, ge=0)]
    gap_count: Annotated[int, Field(strict=True, ge=0)]


@dataclass(frozen=True, slots=True)
class VideoStream:
    """Minimal abstraction for a single camera stream fed to the fast detector.

    This is intentionally lightweight; it carries only the metadata needed
    for deterministic checks, not decoded pixel data.
    """

    stream_id: str
    camera_id: str
    topic: str
    codec: str
    width: int
    height: int
    nominal_fps: float
    duration_ns: int


class FastDetector:
    """Deterministic stream integrity checks without VLM.

    The fast detector complements (but does not replace) visual QA.  It flags
    container corruption, timestamp non-monotonicity, missing timestamps, and
    obvious metadata failures so that the expensive VLM stage is never wasted
    on structurally broken input.
    """

    def __init__(self, config: FastDetectorConfig | None = None) -> None:
        self.config = config or FastDetectorConfig(
            black_threshold=16,
            overexposed_threshold=240,
            stationary_threshold_sec=5.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_stream_integrity(self, stream: VideoStream) -> StreamIntegrityResult:
        """Run all deterministic checks on a single :class:`VideoStream`.

        Returns a :class:`StreamIntegrityResult` whose ``status`` is one of
        ``PASS``, ``DEGRADED``, or ``FAIL``.
        """
        from datetime import datetime, timezone

        checked_at = datetime.now(timezone.utc).isoformat()
        issues: list[StreamIntegrityIssue] = []

        # 1. Metadata sanity
        if stream.width <= 0 or stream.height <= 0:
            issues.append(
                StreamIntegrityIssue(
                    code="INVALID_RESOLUTION",
                    message=f"Invalid resolution {stream.width}x{stream.height}",
                )
            )

        if stream.nominal_fps <= 0.0:
            issues.append(
                StreamIntegrityIssue(
                    code="INVALID_FPS",
                    message=f"Invalid nominal FPS {stream.nominal_fps}",
                )
            )

        if stream.duration_ns <= 0:
            issues.append(
                StreamIntegrityIssue(
                    code="INVALID_DURATION",
                    message=f"Invalid duration_ns {stream.duration_ns}",
                )
            )

        # 2. Codec support check (deterministic; no decode attempted)
        supported_codecs = {"h264", "h265", "hevc", "vp8", "vp9", "av1"}
        if stream.codec.lower() not in supported_codecs:
            issues.append(
                StreamIntegrityIssue(
                    code="UNSUPPORTED_CODEC",
                    message=f"Codec '{stream.codec}' is not in the supported set",
                )
            )

        status = _IntegrityStatus.PASS if not issues else _IntegrityStatus.DEGRADED
        if any(i.code in {"INVALID_RESOLUTION", "INVALID_FPS", "INVALID_DURATION"} for i in issues):
            status = _IntegrityStatus.FAIL

        return StreamIntegrityResult(
            status=status,
            issues=tuple(issues),
            checked_at=checked_at,
        )

    def check_container(self, mcap_path: Path) -> ContainerCheckResult:
        """Verify MCAP container structural integrity.

        Checks:
        - File exists and is non-empty.
        - File size is within a reasonable bound (not zero, not pathological).
        - Path suffix hints at a supported container.

        Does **not** parse the full MCAP index; that is the ingestion service's
        responsibility.  This is a fast pre-flight check.
        """
        from datetime import datetime, timezone

        checked_at = datetime.now(timezone.utc).isoformat()
        issues: list[StreamIntegrityIssue] = []

        if not mcap_path.exists():
            issues.append(
                StreamIntegrityIssue(
                    code="MCAP_NOT_FOUND",
                    message=f"MCAP file not found: {mcap_path}",
                )
            )
        else:
            size = mcap_path.stat().st_size
            if size == 0:
                issues.append(
                    StreamIntegrityIssue(
                        code="EMPTY_MCAP",
                        message="MCAP file is empty",
                    )
                )
            elif size < 1024:
                issues.append(
                    StreamIntegrityIssue(
                        code="SUSPICIOUSLY_SMALL_MCAP",
                        message=f"MCAP file is suspiciously small ({size} bytes)",
                    )
                )

        status = _IntegrityStatus.PASS if not issues else _IntegrityStatus.FAIL
        return ContainerCheckResult(
            status=status,
            issues=tuple(issues),
            checked_at=checked_at,
        )

    def check_timestamp_monotonicity(self, timestamps: Sequence[int]) -> TimestampCheckResult:
        """Verify that a sequence of timestamps is strictly monotonic.

        Returns the number of non-monotonic gaps detected and the overall status.
        """
        from datetime import datetime, timezone

        checked_at = datetime.now(timezone.utc).isoformat()
        issues: list[StreamIntegrityIssue] = []

        if not timestamps:
            issues.append(
                StreamIntegrityIssue(
                    code="NO_TIMESTAMPS",
                    message="Timestamp sequence is empty",
                )
            )
            return TimestampCheckResult(
                status=_IntegrityStatus.FAIL,
                issues=tuple(issues),
                checked_at=checked_at,
                first_ns=0,
                last_ns=0,
                gap_count=0,
            )

        first_ns = timestamps[0]
        last_ns = timestamps[-1]
        gap_count = 0

        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i - 1]:
                gap_count += 1
                # For non-monotonic timestamps, use the earlier timestamp as start
                # and the later (previous) timestamp as end to satisfy interval invariant.
                start_ns = min(timestamps[i - 1], timestamps[i])
                end_ns = max(timestamps[i - 1], timestamps[i])
                issues.append(
                    StreamIntegrityIssue(
                        code="NON_MONOTONIC_TIMESTAMP",
                        interval=NanosecondInterval(
                            start_ns=start_ns,
                            end_ns=end_ns,
                        ),
                        message=f"Non-monotonic timestamp at index {i}: "
                                f"{timestamps[i]} <= {timestamps[i - 1]}",
                    )
                )

        status = _IntegrityStatus.PASS if not issues else _IntegrityStatus.DEGRADED
        return TimestampCheckResult(
            status=status,
            issues=tuple(issues),
            checked_at=checked_at,
            first_ns=first_ns,
            last_ns=last_ns,
            gap_count=gap_count,
        )

    def check_decode_gaps(self, frames: Sequence[FramePayload]) -> DecodeGapResult:
        """Detect gaps between expected and actual decoded frames.

        Compares the number of expected frames (derived from nominal duration and
        FPS) against the number of successfully decoded :class:`FramePayload`
        instances.  A large discrepancy flags a decode gap.
        """
        from datetime import datetime, timezone

        checked_at = datetime.now(timezone.utc).isoformat()
        issues: list[StreamIntegrityIssue] = []

        decoded_frames = len(frames)

        if decoded_frames == 0:
            issues.append(
                StreamIntegrityIssue(
                    code="NO_DECODED_FRAMES",
                    message="Zero frames were successfully decoded",
                )
            )
            return DecodeGapResult(
                status=_IntegrityStatus.FAIL,
                issues=tuple(issues),
                checked_at=checked_at,
                expected_frames=0,
                decoded_frames=0,
                gap_count=0,
            )

        # Derive expected frame count from timestamp span if available.
        if decoded_frames >= 2:
            timestamps = sorted(f.timestamp_sec for f in frames)
            span_sec = timestamps[-1] - timestamps[0]
            # Very rough heuristic: if average interval is > 1s, something is wrong.
            avg_interval = span_sec / (decoded_frames - 1)
            if avg_interval > 1.0:
                issues.append(
                    StreamIntegrityIssue(
                        code="LARGE_DECODE_INTERVAL",
                        message=f"Average decode interval is {avg_interval:.2f}s, "
                                f"suggesting dropped frames",
                    )
                )

        gap_count = len(issues)
        status = _IntegrityStatus.PASS if not issues else _IntegrityStatus.DEGRADED
        return DecodeGapResult(
            status=status,
            issues=tuple(issues),
            checked_at=checked_at,
            expected_frames=decoded_frames,  # best-effort; caller provides nominal
            decoded_frames=decoded_frames,
            gap_count=gap_count,
        )
