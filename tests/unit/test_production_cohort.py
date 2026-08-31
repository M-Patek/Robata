from __future__ import annotations

import pytest

from robata.benchmark.production_cohort import CameraSpan, build_windows, common_camera_span


def _spans(duration: float = 40.8335) -> tuple[CameraSpan, ...]:
    start = 1_000_000_000
    end = start + int(duration * 1_000_000_000)
    return tuple(
        CameraSpan(
            camera_id=f"cam_{index:02d}",
            topic=f"/camera/{index}",
            frame_count=1226,
            first_timestamp_ns=start + index,
            last_timestamp_ns=end - index,
        )
        for index in range(1, 7)
    )


def test_common_span_uses_intersection() -> None:
    spans = _spans()
    start, end = common_camera_span(spans)
    assert start == 1_000_000_006
    assert end == spans[-1].last_timestamp_ns


def test_default_policy_makes_five_full_windows_and_records_tail_separately() -> None:
    windows = build_windows(_spans(), window_seconds=8.0)
    assert len(windows) == 5
    assert windows[0].start_seconds == 0.0
    assert windows[-1].end_seconds == 40.0
    assert all(item.gold_status == "PENDING_HUMAN_REVIEW" for item in windows)


def test_include_tail_adds_short_window() -> None:
    windows = build_windows(_spans(), window_seconds=8.0, include_tail=True)
    assert len(windows) == 6
    assert windows[-1].window_id.endswith("tail")
    assert 0.8 < windows[-1].duration_seconds < 0.9


def test_dense_stride_creates_overlapping_context_windows_not_boundaries() -> None:
    windows = build_windows(
        _spans(6.0), window_seconds=4.0, window_stride_seconds=1.0, include_tail=True
    )
    # The final full context already reaches the source end.  Dense mode must
    # not emit redundant short contexts at starts 3, 4 and 5 merely because
    # ``include_tail`` is enabled.
    assert len(windows) == 3
    assert windows[0].start_seconds == 0.0
    assert windows[1].start_seconds == 1.0
    assert windows[-1].end_seconds == pytest.approx(6.0)
    assert all(item.end_seconds > item.start_seconds for item in windows)


def test_dense_stride_adds_at_most_one_short_tail() -> None:
    windows = build_windows(
        _spans(6.5), window_seconds=4.0, window_stride_seconds=1.0, include_tail=True
    )
    assert len(windows) == 4
    assert windows[-1].window_id.endswith("tail")
    assert windows[-1].start_seconds == pytest.approx(2.5)
    assert windows[-1].end_seconds == pytest.approx(6.5)


def test_stride_cannot_exceed_context_width() -> None:
    with pytest.raises(ValueError, match="window_stride_seconds"):
        build_windows(_spans(), window_seconds=4.0, window_stride_seconds=5.0)


@pytest.mark.parametrize("value", [[], {}, True, "not-a-number"])
def test_window_seconds_malformed_values_raise_cohort_error(value: object) -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        build_windows(_spans(), window_seconds=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[], {}, True, "not-a-number"])
def test_window_stride_malformed_values_raise_cohort_error(value: object) -> None:
    with pytest.raises(ValueError, match="window_stride_seconds"):
        build_windows(_spans(), window_stride_seconds=value)  # type: ignore[arg-type]
