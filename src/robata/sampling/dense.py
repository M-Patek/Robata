"""Dense temporal sampling logic (Section 7.2 of Architecture V1).

Implements window planning, splitting, and overlap management for dense
sampling modes such as ``ACTION_DENSE`` and ``BOUNDARY_REFINEMENT``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.common import NanosecondInterval, Nanoseconds, StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]

NANOSECONDS_PER_SECOND = 1_000_000_000


class DenseSplitPolicy(StrictModel):
    """Versioned policy governing how oversized dense windows are split."""

    version: NonEmptyString
    max_frames_per_camera: PositiveInt
    max_frames_total: PositiveInt
    overlap_ns: Nanoseconds


class TemporalWindow:
    """Duck-typed temporal window used by the dense planner.

    The planner only requires ``window_id``, ``requested_interval``, and
    ``interval`` attributes so that it can be used with any window
    implementation that exposes these fields.
    """

    window_id: str
    requested_interval: NanosecondInterval
    interval: NanosecondInterval


class DenseSamplingPlanner:
    """Plan dense temporal windows from candidate intervals.

    Given a sequence of candidate half-open intervals (e.g. from event
    proposals or suspicious-window detection), the planner:

    1. Adds pre/post padding.
    2. Clips to the recording duration.
    3. Splits windows that exceed the frame budget into overlapping
       sub-windows.
    """

    def __init__(self, policy: DenseSplitPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> DenseSplitPolicy:
        """The immutable split policy governing this planner."""
        return self._policy

    def plan_dense_windows(
        self,
        candidate_intervals: Sequence[NanosecondInterval],
        padding_ns: int,
        recording_duration_ns: int,
    ) -> Sequence[TemporalWindow]:
        """Produce padded, clipped, and optionally split dense windows.

        Args:
            candidate_intervals: Ordered half-open intervals flagged for
                dense analysis.
            padding_ns: Nanoseconds to add before and after each interval.
            recording_duration_ns: Total recording duration; used to clip
                padded intervals so they never exceed the source bounds.

        Returns:
            A sequence of :class:`TemporalWindow`-like objects representing
            the planned dense work.  When a padded interval exceeds the
            frame budget it is split into overlapping sub-windows.
        """
        if padding_ns < 0:
            raise ValueError("padding_ns must be nonnegative")
        if recording_duration_ns <= 0:
            raise ValueError("recording_duration_ns must be positive")

        results: list[TemporalWindow] = []
        for interval in candidate_intervals:
            # 1. Pad
            padded_start = max(0, interval.start_ns - padding_ns)
            padded_end = min(recording_duration_ns, interval.end_ns + padding_ns)
            if padded_start >= padded_end:
                continue

            padded = NanosecondInterval(start_ns=padded_start, end_ns=padded_end)

            # 2. Check frame budget
            duration_sec = padded.duration_ns / NANOSECONDS_PER_SECOND
            estimated_frames = int(duration_sec * 5)  # placeholder 5 FPS dense rate

            if estimated_frames <= self._policy.max_frames_per_camera:
                # Fits in budget — emit one window
                window = _make_dense_window(padded, padded, 0, 0, len(results))
                results.append(window)
            else:
                # 3. Split into overlapping sub-windows
                split_results = self._split_window(padded, len(results))
                results.extend(split_results)

        return tuple(results)

    def _split_window(
        self,
        interval: NanosecondInterval,
        base_ordinal: int,
    ) -> Sequence[TemporalWindow]:
        """Split an oversized interval into overlapping sub-windows.

        Each sub-window fits within ``max_frames_per_camera`` and carries
        overlap with its neighbors so that boundary frames are not lost.
        """
        max_duration_ns = (
            self._policy.max_frames_per_camera * NANOSECONDS_PER_SECOND // 5
        )  # at 5 FPS placeholder
        overlap = self._policy.overlap_ns

        sub_windows: list[TemporalWindow] = []
        current_start = interval.start_ns
        part_count = 0

        while current_start < interval.end_ns:
            current_end = min(current_start + max_duration_ns, interval.end_ns)
            if current_end == interval.end_ns and part_count == 0:
                # Single window after all — no split needed
                window = _make_dense_window(
                    interval, interval, 0, 0, base_ordinal
                )
                return (window,)

            # Overlap: next window starts before this one ends
            next_start = current_end - overlap if current_end < interval.end_ns else current_end

            overlap_before = overlap if part_count > 0 else 0
            overlap_after = overlap if next_start < interval.end_ns else 0

            sub = NanosecondInterval(start_ns=current_start, end_ns=current_end)
            window = _make_dense_window(
                requested=interval,
                effective=sub,
                overlap_before=overlap_before,
                overlap_after=overlap_after,
                ordinal=base_ordinal + part_count,
            )
            sub_windows.append(window)

            current_start = next_start
            part_count += 1

            # Safety: prevent infinite loops when overlap consumes all progress
            if part_count > 1000:
                raise RuntimeError("dense window split exceeded safety limit")

        return tuple(sub_windows)


def _make_dense_window(
    requested: NanosecondInterval,
    effective: NanosecondInterval,
    overlap_before: int,
    overlap_after: int,
    ordinal: int,
) -> TemporalWindow:
    """Build a duck-typed dense temporal window."""
    window = TemporalWindow()
    window.window_id = f"dense-window-{ordinal}"
    window.requested_interval = requested
    window.interval = effective
    window._overlap_before_ns = overlap_before  # type: ignore[attr-defined]
    window._overlap_after_ns = overlap_after  # type: ignore[attr-defined]
    window._ordinal = ordinal  # type: ignore[attr-defined]
    return window


__all__ = [
    "DenseSamplingPlanner",
    "DenseSplitPolicy",
]
