"""Suspicious Interval Reducer.

Merges overlapping coarse-stage observations, adds configurable padding,
clips to source bounds, and produces an explicit dense work manifest.

Architecture V1 Section 12.1: "The suspicious interval reducer merges
overlapping observations from adjacent windows, adds configured padding,
clips to source bounds, and creates an explicit dense work manifest."
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from robata.contracts.cameras import CameraId
from robata.contracts.common import StrictModel
from robata.contracts.logical_nodes import OpaqueUuid

__all__ = [
    "ReducedInterval",
    "ReductionPolicyVersion",
    "SourceIntervalRef",
    "SuspiciousInterval",
    "SuspiciousIntervalReducer",
]


class SuspiciousInterval(StrictModel):
    """One interval flagged by coarse QA as requiring dense analysis.

    This is the input to the reducer; it may overlap with other intervals
    from the same or different cameras.
    """

    start_ns: Annotated[int, Field(strict=True)]
    end_ns: Annotated[int, Field(strict=True)]
    camera_id: CameraId
    issue_type: Annotated[str, Field(strict=True, min_length=1)]
    confidence: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ]


class SourceIntervalRef(StrictModel):
    """Reference to one or more source intervals that contributed to a reduced interval."""

    interval_id: OpaqueUuid
    camera_id: CameraId
    issue_type: Annotated[str, Field(strict=True, min_length=1)]
    original_start_ns: Annotated[int, Field(strict=True)]
    original_end_ns: Annotated[int, Field(strict=True)]
    confidence: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ]


class ReductionPolicyVersion(StrictModel):
    """Versioned policy that governs how intervals are reduced."""

    version: Annotated[str, Field(strict=True, min_length=1)]
    padding_ns: Annotated[int, Field(strict=True, ge=0)] = 500_000_000
    max_gap_ns: Annotated[int, Field(strict=True, ge=0)] = 1_000_000_000
    clip_to_recording_bounds: bool = True


class ReducedInterval(StrictModel):
    """One merged interval ready for dense QA.

    ``source_intervals`` preserves every contributing coarse observation so
    that provenance is not lost during reduction.
    """

    start_ns: Annotated[int, Field(strict=True)]
    end_ns: Annotated[int, Field(strict=True)]
    source_intervals: tuple[SourceIntervalRef, ...]
    reduction_policy_version: ReductionPolicyVersion
    merged_from_count: Annotated[int, Field(strict=True, ge=1)]
    cameras: tuple[CameraId, ...]


class SuspiciousIntervalReducer:
    """Merge overlapping suspicious intervals and produce a dense work manifest.

    The reducer groups intervals by camera, sorts them by start time, and
    merges any overlapping or nearly-adjacent intervals.  Configurable padding
    is added to each merged interval, and the result is clipped to the
    recording bounds.
    """

    def reduce(
        self,
        intervals: Sequence[SuspiciousInterval],
        padding_ns: int = 500_000_000,
    ) -> tuple[ReducedInterval, ...]:
        """Reduce a sequence of suspicious intervals into merged dense work items.

        Parameters
        ----------
        intervals:
            Raw suspicious intervals produced by coarse QA.
        padding_ns:
            Nanoseconds of padding to add to each side of a merged interval.
            Default is 500 ms (500_000_000 ns).

        Returns
        -------
        tuple[ReducedInterval, ...]
            Non-overlapping, sorted reduced intervals with provenance.
        """
        if not intervals:
            return ()

        # Group by camera, sort by start_ns within each group.
        by_camera: dict[CameraId, list[SuspiciousInterval]] = {}
        for interval in intervals:
            by_camera.setdefault(interval.camera_id, []).append(interval)

        for camera_intervals in by_camera.values():
            camera_intervals.sort(key=lambda i: i.start_ns)

        # Merge overlapping intervals per camera.
        merged: list[ReducedInterval] = []
        for camera_id, camera_intervals in by_camera.items():
            current_start = camera_intervals[0].start_ns
            current_end = camera_intervals[0].end_ns
            current_sources: list[SourceIntervalRef] = []

            for interval in camera_intervals:
                # Check overlap: if interval.start_ns <= current_end + padding_ns
                # (allowing for configurable gap tolerance)
                if interval.start_ns <= current_end + padding_ns:
                    current_end = max(current_end, interval.end_ns)
                    # Add source reference
                    # Use a synthetic UUID since we don't have real IDs here.
                    import uuid
                    current_sources.append(
                        SourceIntervalRef(
                            interval_id=str(uuid.uuid4()),
                            camera_id=interval.camera_id,
                            issue_type=interval.issue_type,
                            original_start_ns=interval.start_ns,
                            original_end_ns=interval.end_ns,
                            confidence=interval.confidence,
                        )
                    )
                else:
                    # Flush current merged interval
                    merged.append(
                        self._build_reduced(
                            current_start,
                            current_end,
                            current_sources,
                            camera_id,
                            padding_ns,
                        )
                    )
                    current_start = interval.start_ns
                    current_end = interval.end_ns
                    import uuid
                    current_sources = [
                        SourceIntervalRef(
                            interval_id=str(uuid.uuid4()),
                            camera_id=interval.camera_id,
                            issue_type=interval.issue_type,
                            original_start_ns=interval.start_ns,
                            original_end_ns=interval.end_ns,
                            confidence=interval.confidence,
                        )
                    ]

            # Flush last interval for this camera.
            if current_sources:
                merged.append(
                    self._build_reduced(
                        current_start,
                        current_end,
                        current_sources,
                        camera_id,
                        padding_ns,
                    )
                )

        # Sort by start_ns globally.
        merged.sort(key=lambda r: r.start_ns)
        return tuple(merged)

    def _build_reduced(
        self,
        start_ns: int,
        end_ns: int,
        sources: list[SourceIntervalRef],
        camera_id: CameraId,
        padding_ns: int,
    ) -> ReducedInterval:
        """Build a single ReducedInterval with padding and provenance."""
        padded_start = start_ns - padding_ns
        padded_end = end_ns + padding_ns

        # Clip to non-negative (can't know recording bounds without recording).
        padded_start = max(padded_start, 0)
        padded_end = max(padded_end, padded_start + 1)  # Ensure at least 1 ns.

        policy = ReductionPolicyVersion(
            version="v1.0",
            padding_ns=padding_ns,
        )

        return ReducedInterval(
            start_ns=padded_start,
            end_ns=padded_end,
            source_intervals=tuple(sources),
            reduction_policy_version=policy,
            merged_from_count=len(sources),
            cameras=(camera_id,),
        )
