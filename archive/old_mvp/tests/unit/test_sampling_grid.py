from itertools import permutations

import pytest

from robata.sampling.grid import (
    FrameCandidate,
    SamplingGrid,
    SamplingRate,
    SamplingTarget,
    SelectionStatus,
    select_nearest_frames,
)


def test_sampling_rate_and_period_are_reduced() -> None:
    grid = SamplingGrid(grid_origin_ns=0, rate=SamplingRate(6, 4))

    assert grid.rate == SamplingRate(3, 2)
    assert (grid.period_num_ns, grid.period_den) == (2_000_000_000, 3)


def test_grid_uses_half_even_and_supports_negative_indexes() -> None:
    grid = SamplingGrid(grid_origin_ns=0, rate=SamplingRate(2_000_000_000))

    assert [(item.k, item.target_ns) for item in grid.enumerate_targets(-2, 3)] == [
        (-5, -2),
        (-2, -1),
        (-1, 0),
        (2, 1),
        (3, 2),
    ]


def test_clipping_does_not_reset_grid_phase() -> None:
    grid = SamplingGrid(grid_origin_ns=100, rate=SamplingRate(2))

    assert grid.enumerate_targets(200_000_000, 1_200_000_000) == (
        SamplingTarget(k=1, target_ns=500_000_100),
        SamplingTarget(k=2, target_ns=1_000_000_100),
    )


def test_grid_respects_half_open_bounds_at_epoch_scale() -> None:
    origin = 1_710_000_000_000_000_000
    grid = SamplingGrid(grid_origin_ns=origin, rate=SamplingRate(4))

    targets = grid.enumerate_targets(origin - 250_000_000, origin + 500_000_000)

    assert targets == (
        SamplingTarget(k=-1, target_ns=origin - 250_000_000),
        SamplingTarget(k=0, target_ns=origin),
        SamplingTarget(k=1, target_ns=origin + 250_000_000),
    )


def frame(aligned_ns: int, source_ns: int, locator: bytes) -> FrameCandidate:
    return FrameCandidate(
        aligned_timestamp_ns=aligned_ns,
        source_timestamp_ns=source_ns,
        source_locator_bytes=locator,
    )


def test_nearest_frame_tie_break_is_canonical_and_input_order_independent() -> None:
    target = SamplingTarget(k=0, target_ns=100)
    candidates = (
        frame(101, 1_001, b"z"),
        frame(99, 1_099, b"z"),
        frame(99, 1_001, b"z"),
        frame(99, 1_001, b"a"),
    )

    winners = {
        select_nearest_frames(
            [target],
            ordering,
            interval_start_ns=0,
            interval_end_ns=200,
            selection_tolerance_ns=1,
        )[0].frame
        for ordering in permutations(candidates)
    }

    assert winners == {frame(99, 1_001, b"a")}


def test_tolerance_interval_and_decodability_produce_explicit_miss() -> None:
    selections = select_nearest_frames(
        [SamplingTarget(k=0, target_ns=100)],
        [
            frame(102, 1_002, b"too-far"),
            frame(100, 1_000, b"outside"),
            FrameCandidate(100, 1_000, b"decode-failed", decodable=False),
        ],
        interval_start_ns=101,
        interval_end_ns=200,
        selection_tolerance_ns=1,
    )

    assert selections[0].status is SelectionStatus.NO_FRAME_WITHIN_TOLERANCE
    assert selections[0].frame is None
    assert selections[0].delta_to_target_ns is None


def test_decode_failure_is_distinct_from_no_frame_and_retains_nearest_audit_frame() -> None:
    selections = select_nearest_frames(
        [SamplingTarget(k=0, target_ns=100)],
        [
            FrameCandidate(99, 1_001, b"later-source", decodable=False),
            FrameCandidate(99, 999, b"earlier-source", decodable=False),
        ],
        interval_start_ns=0,
        interval_end_ns=200,
        selection_tolerance_ns=1,
    )

    assert selections[0].status is SelectionStatus.DECODE_FAILED
    assert selections[0].frame == FrameCandidate(99, 999, b"earlier-source", decodable=False)
    assert selections[0].delta_to_target_ns == -1


def test_decodable_frame_is_preferred_when_failed_frame_is_closer() -> None:
    selections = select_nearest_frames(
        [SamplingTarget(k=0, target_ns=100)],
        [
            FrameCandidate(100, 1_000, b"failed", decodable=False),
            FrameCandidate(102, 1_002, b"decoded"),
        ],
        interval_start_ns=0,
        interval_end_ns=200,
        selection_tolerance_ns=2,
    )

    assert selections[0].status is SelectionStatus.SELECTED
    assert selections[0].frame == FrameCandidate(102, 1_002, b"decoded")


def test_one_frame_is_selected_once_and_other_target_is_recorded_as_deduplicated() -> None:
    shared = frame(100, 1_000, b"shared")
    selections = select_nearest_frames(
        [SamplingTarget(k=0, target_ns=99), SamplingTarget(k=1, target_ns=101)],
        [shared],
        interval_start_ns=0,
        interval_end_ns=200,
        selection_tolerance_ns=2,
    )

    assert [selection.status for selection in selections] == [
        SelectionStatus.SELECTED,
        SelectionStatus.DEDUPLICATED_FRAME,
    ]
    assert [selection.delta_to_target_ns for selection in selections] == [1, -1]
    assert sum(selection.status is SelectionStatus.SELECTED for selection in selections) == 1


def test_dedupe_winner_prefers_absolute_delta_then_target_then_k() -> None:
    shared = frame(101, 1_000, b"shared")
    targets = [
        SamplingTarget(k=3, target_ns=100),
        SamplingTarget(k=2, target_ns=101),
        SamplingTarget(k=1, target_ns=102),
    ]

    selections = select_nearest_frames(
        targets,
        [shared],
        interval_start_ns=0,
        interval_end_ns=200,
        selection_tolerance_ns=1,
    )

    assert [
        selection.k for selection in selections if selection.status is SelectionStatus.SELECTED
    ] == [2]


def test_duplicate_rounded_targets_retain_lowest_k() -> None:
    selections = select_nearest_frames(
        [SamplingTarget(k=5, target_ns=10), SamplingTarget(k=3, target_ns=10)],
        [],
        interval_start_ns=0,
        interval_end_ns=20,
        selection_tolerance_ns=0,
    )

    assert len(selections) == 1
    assert selections[0].k == 3


@pytest.mark.parametrize("tolerance", [-1, 2**63])
def test_selection_rejects_invalid_tolerance(tolerance: int) -> None:
    with pytest.raises(ValueError):
        select_nearest_frames(
            [],
            [],
            interval_start_ns=0,
            interval_end_ns=1,
            selection_tolerance_ns=tolerance,
        )
