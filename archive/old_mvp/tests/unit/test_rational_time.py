import pytest

from robata.alignment.rational_time import (
    PiecewiseAlignment,
    RationalTransformSegment,
    round_half_even,
)
from robata.contracts.common import INT64_MAX, INT64_MIN


@pytest.mark.parametrize(
    "numerator,denominator,expected",
    [
        (1, 2, 0),
        (3, 2, 2),
        (5, 2, 2),
        (7, 2, 4),
        (-1, 2, 0),
        (-3, 2, -2),
        (-5, 2, -2),
        (-7, 2, -4),
        (8, 3, 3),
        (-8, 3, -3),
    ],
)
def test_round_half_even_is_exact_for_signs_and_ties(
    numerator: int,
    denominator: int,
    expected: int,
) -> None:
    assert round_half_even(numerator, denominator) == expected


@pytest.mark.parametrize("denominator", [0, -1])
def test_round_half_even_requires_positive_denominator(denominator: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        round_half_even(1, denominator)


def segment(**overrides: object) -> RationalTransformSegment:
    values: dict[str, object] = {
        "source_order_start": 0,
        "source_order_end": 10,
        "source_start_ns": 1_710_000_000_000_000_000,
        "source_end_ns": 1_710_000_000_000_000_100,
        "source_anchor_ns": 1_710_000_000_000_000_000,
        "canonical_anchor_ns": 0,
        "rate_numerator": 3,
        "rate_denominator": 2,
        "source_epoch_id": "epoch-0",
    }
    values.update(overrides)
    return RationalTransformSegment(**values)  # type: ignore[arg-type]


def test_anchored_transform_preserves_epoch_precision_and_negative_delta() -> None:
    transform = segment(
        source_start_ns=1_709_999_999_999_999_900,
        source_anchor_ns=1_710_000_000_000_000_000,
        canonical_anchor_ns=10,
    )

    assert transform.apply(1_710_000_000_000_000_003) == 14
    assert transform.apply(1_709_999_999_999_999_997) == 6


def test_transform_bounds_are_half_open_and_result_is_int64() -> None:
    transform = segment(rate_numerator=1, rate_denominator=1)

    assert transform.apply(transform.source_start_ns, source_order=0) == 0
    with pytest.raises(ValueError, match="half-open"):
        transform.apply(transform.source_end_ns)
    with pytest.raises(ValueError, match="source order"):
        transform.apply(transform.source_start_ns, source_order=10)

    overflowing = RationalTransformSegment(
        source_order_start=0,
        source_order_end=1,
        source_start_ns=0,
        source_end_ns=2,
        source_anchor_ns=0,
        canonical_anchor_ns=INT64_MAX,
        rate_numerator=1,
        rate_denominator=1,
    )
    with pytest.raises(OverflowError, match="int64"):
        overflowing.apply(1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_order_start": 1, "source_order_end": 1},
        {"source_start_ns": 2, "source_end_ns": 2},
        {"rate_numerator": 0},
        {"rate_denominator": -1},
        {"canonical_anchor_ns": INT64_MIN - 1},
    ],
)
def test_transform_rejects_invalid_ranges_and_rates(overrides: dict[str, int]) -> None:
    with pytest.raises((TypeError, ValueError)):
        segment(**overrides)


def test_piecewise_selection_uses_source_order_across_clock_reset() -> None:
    first = segment(
        source_order_start=0,
        source_order_end=2,
        source_start_ns=1_000,
        source_end_ns=1_100,
        source_anchor_ns=1_000,
        canonical_anchor_ns=0,
        rate_numerator=1,
        rate_denominator=1,
        source_epoch_id="epoch-0",
    )
    reset = segment(
        source_order_start=2,
        source_order_end=4,
        source_start_ns=100,
        source_end_ns=200,
        source_anchor_ns=100,
        canonical_anchor_ns=100,
        rate_numerator=1,
        rate_denominator=1,
        source_epoch_id="epoch-1",
    )
    alignment = PiecewiseAlignment([first, reset])

    assert alignment.apply(1, 1_050) == 50
    assert alignment.apply(2, 150) == 150
    assert alignment.segment_for(2).source_epoch_id == "epoch-1"
    with pytest.raises(ValueError, match="source-order-selected"):
        alignment.apply(2, 1_050)


def test_piecewise_alignment_rejects_overlap_and_out_of_order() -> None:
    first = segment(source_order_start=5, source_order_end=10)
    overlapping = segment(source_order_start=9, source_order_end=12)
    earlier = segment(source_order_start=0, source_order_end=1)

    with pytest.raises(ValueError, match="overlap"):
        PiecewiseAlignment([first, overlapping])
    with pytest.raises(ValueError, match="out of source-order"):
        PiecewiseAlignment([first, earlier])


def test_piecewise_alignment_detects_uncovered_order_gap() -> None:
    alignment = PiecewiseAlignment(
        [
            segment(source_order_start=0, source_order_end=2),
            segment(source_order_start=4, source_order_end=6),
        ]
    )

    with pytest.raises(ValueError, match="not covered"):
        alignment.segment_for(3)
