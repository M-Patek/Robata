"""Suspicious Interval Reducer.

Merges overlapping coarse-stage observations, adds configurable padding,
clips to source bounds, and produces an explicit dense work manifest.

Architecture V1 Section 12.1: "The suspicious interval reducer merges
overlapping observations from adjacent windows, adds configured padding,
clips to source bounds, and creates an explicit dense work manifest."
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, model_validator

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

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.start_ns >= self.end_ns:
            raise ValueError("suspicious interval must be non-empty")
        return self


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

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.original_start_ns >= self.original_end_ns:
            raise ValueError("source interval must be non-empty")
        return self


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

    @staticmethod
    def _validate_interval(start_ns: int, end_ns: int) -> None:
        if start_ns >= end_ns:
            raise ValueError("reduced interval must be non-empty")

    @model_validator(mode="after")
    def validate_reduced(self) -> Self:
        self._validate_interval(self.start_ns, self.end_ns)
        if self.merged_from_count != len(self.source_intervals):
            raise ValueError("merged_from_count must match source_intervals")
        canonical_cameras = tuple(
            sorted(set(self.cameras), key=lambda item: item.value)
        )
        if not self.cameras or canonical_cameras != self.cameras:
            raise ValueError("cameras must be unique and in canonical order")
        return self


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
        *,
        max_gap_ns: int = 1_000_000_000,
        recording_duration_ns: int | None = None,
        policy_version: str = "v1.0",
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
        if padding_ns < 0 or max_gap_ns < 0:
            raise ValueError("padding_ns and max_gap_ns must be nonnegative")
        if recording_duration_ns is not None and recording_duration_ns <= 0:
            raise ValueError("recording_duration_ns must be positive")
        if not intervals:
            return ()

        ordered = sorted(
            intervals,
            key=lambda item: (
                item.start_ns,
                item.end_ns,
                item.camera_id.value,
                item.issue_type,
                item.confidence,
            ),
        )
        merged: list[ReducedInterval] = []
        current_start = ordered[0].start_ns
        current_end = ordered[0].end_ns
        current_items: list[SuspiciousInterval] = [ordered[0]]
        for interval in ordered[1:]:
            if interval.start_ns <= current_end + max_gap_ns:
                current_end = max(current_end, interval.end_ns)
                current_items.append(interval)
                continue
            merged.append(
                self._build_reduced(
                    current_start,
                    current_end,
                    current_items,
                    padding_ns,
                    recording_duration_ns,
                    policy_version,
                    max_gap_ns,
                )
            )
            current_start = interval.start_ns
            current_end = interval.end_ns
            current_items = [interval]
        merged.append(
            self._build_reduced(
                current_start,
                current_end,
                current_items,
                padding_ns,
                recording_duration_ns,
                policy_version,
                max_gap_ns,
            )
        )
        return tuple(sorted(merged, key=lambda item: (item.start_ns, item.end_ns)))

    def _build_reduced(
        self,
        start_ns: int,
        end_ns: int,
        source_items: list[SuspiciousInterval],
        padding_ns: int,
        recording_duration_ns: int | None,
        policy_version: str,
        max_gap_ns: int,
    ) -> ReducedInterval:
        """Build a single ReducedInterval with padding and provenance."""
        padded_start = max(start_ns - padding_ns, 0)
        padded_end = end_ns + padding_ns
        if recording_duration_ns is not None:
            padded_end = min(padded_end, recording_duration_ns)
        if padded_start >= padded_end:
            raise ValueError("padding/clipping produced an empty reduced interval")

        sources: list[SourceIntervalRef] = []
        for ordinal, interval in enumerate(
            sorted(
                source_items,
                key=lambda item: (
                    item.start_ns,
                    item.end_ns,
                    item.camera_id.value,
                    item.issue_type,
                    item.confidence,
                ),
            )
        ):
            seed = (
                f"{interval.camera_id.value}|{interval.issue_type}|"
                f"{interval.start_ns}|{interval.end_ns}|{interval.confidence:.17g}|{ordinal}"
            )
            source_id = str(uuid5(NAMESPACE_URL, f"robata:qa-source:{seed}"))
            sources.append(
                SourceIntervalRef(
                    interval_id=source_id,
                    camera_id=interval.camera_id,
                    issue_type=interval.issue_type,
                    original_start_ns=interval.start_ns,
                    original_end_ns=interval.end_ns,
                    confidence=interval.confidence,
                )
            )

        cameras = tuple(
            sorted({item.camera_id for item in source_items}, key=lambda item: item.value)
        )

        policy = ReductionPolicyVersion(
            version=policy_version,
            padding_ns=padding_ns,
            max_gap_ns=max_gap_ns,
            clip_to_recording_bounds=recording_duration_ns is not None,
        )

        return ReducedInterval(
            start_ns=padded_start,
            end_ns=padded_end,
            source_intervals=tuple(sources),
            reduction_policy_version=policy,
            merged_from_count=len(sources),
            cameras=cameras,
        )
