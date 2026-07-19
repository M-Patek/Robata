"""Exact rational sampling grids and deterministic nearest-frame selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import gcd

from robata.alignment.rational_time import round_half_even
from robata.contracts.common import INT64_MAX, INT64_MIN

NANOSECONDS_PER_SECOND = 1_000_000_000


def _require_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _require_int64(name: str, value: int) -> None:
    _require_integer(name, value)
    if value < INT64_MIN or value > INT64_MAX:
        raise ValueError(f"{name} must fit in a signed 64-bit integer")


@dataclass(frozen=True, slots=True)
class SamplingRate:
    """A positive frames-per-second rational in canonical reduced form."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        _require_integer("numerator", self.numerator)
        _require_integer("denominator", self.denominator)
        if self.numerator <= 0 or self.denominator <= 0:
            raise ValueError("sampling-rate numerator and denominator must be positive")

        divisor = gcd(self.numerator, self.denominator)
        object.__setattr__(self, "numerator", self.numerator // divisor)
        object.__setattr__(self, "denominator", self.denominator // divisor)

    @property
    def rate_num(self) -> int:
        return self.numerator

    @property
    def rate_den(self) -> int:
        return self.denominator


@dataclass(frozen=True, slots=True, order=True)
class SamplingTarget:
    """One rational-grid index and its rounded canonical timestamp."""

    k: int
    target_ns: int

    def __post_init__(self) -> None:
        _require_integer("k", self.k)
        _require_int64("target_ns", self.target_ns)

    @property
    def index(self) -> int:
        return self.k


@dataclass(frozen=True, slots=True)
class SamplingGrid:
    """A phase-stable rational grid anchored at a persisted canonical origin."""

    grid_origin_ns: int
    rate: SamplingRate

    def __post_init__(self) -> None:
        _require_int64("grid_origin_ns", self.grid_origin_ns)
        if not isinstance(self.rate, SamplingRate):
            raise TypeError("rate must be a SamplingRate")

    @property
    def period_num_ns(self) -> int:
        raw_numerator = NANOSECONDS_PER_SECOND * self.rate.denominator
        divisor = gcd(raw_numerator, self.rate.numerator)
        return raw_numerator // divisor

    @property
    def period_den(self) -> int:
        raw_numerator = NANOSECONDS_PER_SECOND * self.rate.denominator
        divisor = gcd(raw_numerator, self.rate.numerator)
        return self.rate.numerator // divisor

    def target_ns(self, k: int) -> int:
        """Return ``origin + round_half_even(k * exact_period)``."""

        _require_integer("k", k)
        target = self.grid_origin_ns + round_half_even(
            k * self.period_num_ns,
            self.period_den,
        )
        if target < INT64_MIN or target > INT64_MAX:
            raise OverflowError("sampling target does not fit in signed int64 nanoseconds")
        return target

    def _first_k_at_or_after(self, timestamp_ns: int) -> int:
        """Solve the HALF_EVEN boundary for the first target at/after a timestamp."""

        relative_target = timestamp_ns - self.grid_origin_ns
        # round(x) >= m changes at x = m - 1/2.  The exact boundary belongs to m
        # only when m is even; otherwise it still rounds to m - 1.
        boundary = (2 * relative_target - 1) * self.period_den
        scale = 2 * self.period_num_ns
        quotient, remainder = divmod(boundary, scale)
        if remainder == 0 and relative_target % 2 == 0:
            return quotient
        return quotient + 1

    def enumerate_targets(self, start_ns: int, end_ns: int) -> tuple[SamplingTarget, ...]:
        """Enumerate unique rounded targets in the half-open effective interval.

        The persisted origin is never changed, so clipping an interval cannot reset grid
        phase.  Iteration starts at the exact first covering index and therefore naturally
        supports negative ``k`` values.
        """

        _require_int64("start_ns", start_ns)
        _require_int64("end_ns", end_ns)
        if start_ns >= end_ns:
            raise ValueError("start_ns must be less than end_ns")

        first_k = self._first_k_at_or_after(start_ns)
        stop_k = self._first_k_at_or_after(end_ns)
        targets: list[SamplingTarget] = []
        previous_target_ns: int | None = None
        for k in range(first_k, stop_k):
            target_ns = self.target_ns(k)
            if target_ns == previous_target_ns:
                continue
            targets.append(SamplingTarget(k=k, target_ns=target_ns))
            previous_target_ns = target_ns
        return tuple(targets)

    def targets(self, start_ns: int, end_ns: int) -> tuple[SamplingTarget, ...]:
        """Alias for :meth:`enumerate_targets`."""

        return self.enumerate_targets(start_ns, end_ns)

    def select_frames(
        self,
        frames: Iterable[FrameCandidate],
        start_ns: int,
        end_ns: int,
        selection_tolerance_ns: int,
    ) -> tuple[TargetSelection, ...]:
        """Enumerate this grid and deterministically resolve its frame assignments."""

        return select_nearest_frames(
            self.enumerate_targets(start_ns, end_ns),
            frames,
            interval_start_ns=start_ns,
            interval_end_ns=end_ns,
            selection_tolerance_ns=selection_tolerance_ns,
        )


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    """A source frame projected into the selected alignment version."""

    aligned_timestamp_ns: int
    source_timestamp_ns: int
    source_locator_bytes: bytes
    decodable: bool = True

    def __post_init__(self) -> None:
        _require_int64("aligned_timestamp_ns", self.aligned_timestamp_ns)
        _require_int64("source_timestamp_ns", self.source_timestamp_ns)
        if not isinstance(self.source_locator_bytes, bytes):
            raise TypeError("source_locator_bytes must be immutable bytes")
        if not self.source_locator_bytes:
            raise ValueError("source_locator_bytes must not be empty")
        if not isinstance(self.decodable, bool):
            raise TypeError("decodable must be a boolean")


class SelectionStatus(StrEnum):
    DECODE_FAILED = "DECODE_FAILED"
    SELECTED = "SELECTED"
    NO_FRAME_WITHIN_TOLERANCE = "NO_FRAME_WITHIN_TOLERANCE"
    DEDUPLICATED_FRAME = "DEDUPLICATED_FRAME"


@dataclass(frozen=True, slots=True)
class TargetSelection:
    """The deterministic result for one unique rounded sampling target."""

    target: SamplingTarget
    status: SelectionStatus
    frame: FrameCandidate | None = None
    delta_to_target_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, SamplingTarget):
            raise TypeError("target must be a SamplingTarget")
        if not isinstance(self.status, SelectionStatus):
            raise TypeError("status must be a SelectionStatus")
        if self.status is SelectionStatus.NO_FRAME_WITHIN_TOLERANCE:
            if self.frame is not None or self.delta_to_target_ns is not None:
                raise ValueError("a missed target cannot contain an actual frame or delta")
            return
        if self.frame is None or self.delta_to_target_ns is None:
            raise ValueError("selected and deduplicated targets require a frame and delta")
        _require_int64("delta_to_target_ns", self.delta_to_target_ns)
        expected_delta = self.frame.aligned_timestamp_ns - self.target.target_ns
        if self.delta_to_target_ns != expected_delta:
            raise ValueError("delta_to_target_ns must equal actual aligned time minus target")

    @property
    def k(self) -> int:
        return self.target.k

    @property
    def target_ns(self) -> int:
        return self.target.target_ns

    @property
    def actual_timestamp_ns(self) -> int | None:
        return None if self.frame is None else self.frame.aligned_timestamp_ns


def _normalized_targets(targets: Iterable[SamplingTarget]) -> tuple[SamplingTarget, ...]:
    materialized = tuple(targets)
    for target in materialized:
        if not isinstance(target, SamplingTarget):
            raise TypeError("targets must contain SamplingTarget values")

    by_k: dict[int, int] = {}
    for target in materialized:
        previous = by_k.setdefault(target.k, target.target_ns)
        if previous != target.target_ns:
            raise ValueError("one grid index cannot identify different target timestamps")

    # Duplicate rounded timestamps retain the lowest k, independent of input order.
    by_timestamp: dict[int, SamplingTarget] = {}
    for target in sorted(materialized):
        by_timestamp.setdefault(target.target_ns, target)
    return tuple(sorted(by_timestamp.values(), key=lambda item: item.k))


def select_nearest_frames(
    targets: Iterable[SamplingTarget],
    frames: Iterable[FrameCandidate],
    *,
    interval_start_ns: int,
    interval_end_ns: int,
    selection_tolerance_ns: int,
) -> tuple[TargetSelection, ...]:
    """Select nearest decodable frames and then enforce one-use-per-source-frame.

    Nearest-frame ties use ``(abs delta, aligned time, source time, locator bytes)``.
    When several targets independently select one physical frame, the retained target is
    chosen by ``(abs delta, target time, k)``; all other assignments remain auditable as
    ``DEDUPLICATED_FRAME`` results rather than being silently dropped or reassigned.
    """

    _require_int64("interval_start_ns", interval_start_ns)
    _require_int64("interval_end_ns", interval_end_ns)
    if interval_start_ns >= interval_end_ns:
        raise ValueError("interval_start_ns must be less than interval_end_ns")
    _require_int64("selection_tolerance_ns", selection_tolerance_ns)
    if selection_tolerance_ns < 0:
        raise ValueError("selection_tolerance_ns must be nonnegative")

    normalized_targets = _normalized_targets(targets)
    materialized_frames = tuple(frames)
    for frame in materialized_frames:
        if not isinstance(frame, FrameCandidate):
            raise TypeError("frames must contain FrameCandidate values")

    interval_frames = tuple(
        frame
        for frame in materialized_frames
        if interval_start_ns <= frame.aligned_timestamp_ns < interval_end_ns
    )

    provisional: list[TargetSelection] = []
    for target in normalized_targets:
        within_tolerance = tuple(
            frame
            for frame in interval_frames
            if abs(frame.aligned_timestamp_ns - target.target_ns) <= selection_tolerance_ns
        )
        decodable_frames = (frame for frame in within_tolerance if frame.decodable)
        selected_frame = min(
            decodable_frames,
            key=lambda frame: (
                abs(frame.aligned_timestamp_ns - target.target_ns),
                frame.aligned_timestamp_ns,
                frame.source_timestamp_ns,
                frame.source_locator_bytes,
            ),
            default=None,
        )
        if selected_frame is None:
            failed_frame = min(
                within_tolerance,
                key=lambda frame: (
                    abs(frame.aligned_timestamp_ns - target.target_ns),
                    frame.aligned_timestamp_ns,
                    frame.source_timestamp_ns,
                    frame.source_locator_bytes,
                ),
                default=None,
            )
            if failed_frame is not None:
                provisional.append(
                    TargetSelection(
                        target=target,
                        status=SelectionStatus.DECODE_FAILED,
                        frame=failed_frame,
                        delta_to_target_ns=failed_frame.aligned_timestamp_ns - target.target_ns,
                    )
                )
                continue
            provisional.append(
                TargetSelection(
                    target=target,
                    status=SelectionStatus.NO_FRAME_WITHIN_TOLERANCE,
                )
            )
            continue

        provisional.append(
            TargetSelection(
                target=target,
                status=SelectionStatus.SELECTED,
                frame=selected_frame,
                delta_to_target_ns=selected_frame.aligned_timestamp_ns - target.target_ns,
            )
        )

    assignments_by_locator: dict[bytes, list[int]] = {}
    for index, selection in enumerate(provisional):
        if selection.status is SelectionStatus.SELECTED and selection.frame is not None:
            assignments_by_locator.setdefault(selection.frame.source_locator_bytes, []).append(
                index
            )

    results = list(provisional)
    for assignment_indexes in assignments_by_locator.values():
        if len(assignment_indexes) < 2:
            continue
        winner = min(
            assignment_indexes,
            key=lambda index: (
                abs(provisional[index].delta_to_target_ns or 0),
                provisional[index].target_ns,
                provisional[index].k,
            ),
        )
        for index in assignment_indexes:
            if index == winner:
                continue
            selection = provisional[index]
            results[index] = TargetSelection(
                target=selection.target,
                status=SelectionStatus.DEDUPLICATED_FRAME,
                frame=selection.frame,
                delta_to_target_ns=selection.delta_to_target_ns,
            )

    return tuple(results)


# Alternate domain wording kept as an import-level convenience.
CandidateFrame = FrameCandidate
FrameSelection = TargetSelection
