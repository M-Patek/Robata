"""Deterministic fixed-point timestamp alignment.

All calculations in this module use Python integers.  Python's unbounded integer
intermediates preserve epoch-scale source timestamps; only persisted timestamp inputs
and final aligned timestamps are constrained to signed int64 nanoseconds.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field

from robata.contracts.common import INT64_MAX, INT64_MIN


def _require_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _require_int64(name: str, value: int) -> None:
    _require_integer(name, value)
    if value < INT64_MIN or value > INT64_MAX:
        raise ValueError(f"{name} must fit in a signed 64-bit integer")


def round_half_even(numerator: int, denominator: int) -> int:
    """Round an exact rational to the nearest integer, resolving ties to even.

    ``denominator`` must be positive.  ``divmod`` deliberately uses Python's floor
    quotient: its nonnegative remainder makes the same three-way comparison correct
    for positive and negative numerators.
    """

    _require_integer("numerator", numerator)
    _require_integer("denominator", denominator)
    if denominator <= 0:
        raise ValueError("denominator must be positive")

    quotient, remainder = divmod(numerator, denominator)
    doubled_remainder = remainder * 2
    if doubled_remainder < denominator:
        return quotient
    if doubled_remainder > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


@dataclass(frozen=True, slots=True)
class RationalTransformSegment:
    """One source-order-bounded anchored rational timestamp transform.

    Both order and source timestamp bounds are half-open.  Source-order bounds select
    the segment; timestamp bounds then verify that the selected clock epoch is coherent.
    Numeric timestamp ranges may overlap between segments so clock resets are supported.
    """

    source_order_start: int
    source_order_end: int
    source_start_ns: int
    source_end_ns: int
    source_anchor_ns: int
    canonical_anchor_ns: int
    rate_numerator: int
    rate_denominator: int
    source_epoch_id: str = "default"
    segment_id: str | None = None

    def __post_init__(self) -> None:
        _require_integer("source_order_start", self.source_order_start)
        _require_integer("source_order_end", self.source_order_end)
        if self.source_order_start < 0:
            raise ValueError("source_order_start must be nonnegative")
        if self.source_order_start >= self.source_order_end:
            raise ValueError("source_order_start must be less than source_order_end")

        for name, value in (
            ("source_start_ns", self.source_start_ns),
            ("source_end_ns", self.source_end_ns),
            ("source_anchor_ns", self.source_anchor_ns),
            ("canonical_anchor_ns", self.canonical_anchor_ns),
        ):
            _require_int64(name, value)
        if self.source_start_ns >= self.source_end_ns:
            raise ValueError("source_start_ns must be less than source_end_ns")

        _require_integer("rate_numerator", self.rate_numerator)
        _require_integer("rate_denominator", self.rate_denominator)
        if self.rate_numerator <= 0 or self.rate_denominator <= 0:
            raise ValueError("rate numerator and denominator must be positive")
        if not isinstance(self.source_epoch_id, str) or not self.source_epoch_id:
            raise ValueError("source_epoch_id must be a nonempty string")
        if self.segment_id is not None and (
            not isinstance(self.segment_id, str) or not self.segment_id
        ):
            raise ValueError("segment_id must be a nonempty string when provided")

    def contains_order(self, source_order: int) -> bool:
        """Return whether ``source_order`` is in this segment's half-open order range."""

        _require_integer("source_order", source_order)
        return self.source_order_start <= source_order < self.source_order_end

    def contains_timestamp(self, source_timestamp_ns: int) -> bool:
        """Return whether a timestamp is in this segment's half-open source range."""

        _require_int64("source_timestamp_ns", source_timestamp_ns)
        return self.source_start_ns <= source_timestamp_ns < self.source_end_ns

    def apply(self, source_timestamp_ns: int, *, source_order: int | None = None) -> int:
        """Map one in-range source timestamp to signed-int64 canonical nanoseconds."""

        _require_int64("source_timestamp_ns", source_timestamp_ns)
        if not self.contains_timestamp(source_timestamp_ns):
            raise ValueError("source timestamp is outside the segment's half-open range")
        if source_order is not None and not self.contains_order(source_order):
            raise ValueError("source order is outside the segment's half-open range")

        delta_source_ns = source_timestamp_ns - self.source_anchor_ns
        scaled_delta_ns = round_half_even(
            delta_source_ns * self.rate_numerator,
            self.rate_denominator,
        )
        aligned_timestamp_ns = self.canonical_anchor_ns + scaled_delta_ns
        if aligned_timestamp_ns < INT64_MIN or aligned_timestamp_ns > INT64_MAX:
            raise OverflowError("aligned timestamp does not fit in signed int64 nanoseconds")
        return aligned_timestamp_ns


@dataclass(frozen=True, slots=True)
class PiecewiseAlignment:
    """An immutable ordered collection of non-overlapping transform segments."""

    segments: tuple[RationalTransformSegment, ...]
    _order_starts: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __init__(self, segments: Iterable[RationalTransformSegment]) -> None:
        materialized = tuple(segments)
        object.__setattr__(self, "segments", materialized)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("piecewise alignment requires at least one segment")
        for segment in self.segments:
            if not isinstance(segment, RationalTransformSegment):
                raise TypeError("segments must contain RationalTransformSegment values")

        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if current.source_order_start < previous.source_order_start:
                raise ValueError("alignment segments are out of source-order sequence")
            if current.source_order_start < previous.source_order_end:
                raise ValueError("alignment segment source-order ranges overlap")

        object.__setattr__(
            self,
            "_order_starts",
            tuple(segment.source_order_start for segment in self.segments),
        )

    def segment_for(
        self,
        source_order: int,
        source_timestamp_ns: int | None = None,
    ) -> RationalTransformSegment:
        """Select by source order and optionally validate the source clock timestamp."""

        _require_integer("source_order", source_order)
        candidate_index = bisect_right(self._order_starts, source_order) - 1
        if candidate_index < 0:
            raise ValueError("source order is not covered by an alignment segment")

        segment = self.segments[candidate_index]
        if not segment.contains_order(source_order):
            raise ValueError("source order is not covered by an alignment segment")
        if source_timestamp_ns is not None and not segment.contains_timestamp(source_timestamp_ns):
            raise ValueError("source timestamp is outside the source-order-selected segment")
        return segment

    def apply(self, source_order: int, source_timestamp_ns: int) -> int:
        """Select a segment by message order and apply its anchored transform."""

        segment = self.segment_for(source_order, source_timestamp_ns)
        return segment.apply(source_timestamp_ns, source_order=source_order)
