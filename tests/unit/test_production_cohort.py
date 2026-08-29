from __future__ import annotations

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
