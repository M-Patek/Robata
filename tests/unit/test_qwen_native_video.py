from __future__ import annotations

from robata.benchmark.qwen_native_video import (
    QWEN_NATIVE_VIDEO_MAX_FRAMES,
    QwenNativeVideoInput,
    QwenNativeVideoInputError,
    QwenVideoFrame,
)


def _video(*frames: QwenVideoFrame) -> QwenNativeVideoInput:
    return QwenNativeVideoInput(
        frames=tuple(frames),
        source_fps=10.0,
        total_num_frames=100,
        width=640,
        height=480,
        duration_seconds=10.0,
        interval_start_seconds=2.0,
        interval_end_seconds=3.0,
        context_before_seconds=0.25,
        context_after_seconds=0.25,
        source_path="fixture.mp4",
    )


def test_native_video_metadata_preserves_source_timeline_and_evidence() -> None:
    video = _video(
        QwenVideoFrame(payload=b"frame-10", frame_index=10, timestamp_seconds=1.0),
        QwenVideoFrame(payload=b"frame-20", frame_index=20, timestamp_seconds=2.0),
    )

    assert video.frame_indices == (10, 20)
    assert video.frame_timestamps_seconds == (1.0, 2.0)
    assert video.processor_video_metadata() == {
        "total_num_frames": 100,
        "fps": 10.0,
        "width": 640,
        "height": 480,
        "duration": 10.0,
        "frames_indices": [10, 20],
    }
    evidence = video.evidence()
    assert evidence["adapter_version"] == "qwen-native-video-input-v1"
    assert evidence["source_window_start_seconds"] == 1.75
    assert evidence["source_window_end_seconds"] == 3.25
    assert [frame["frame_index"] for frame in evidence["frames"]] == [10, 20]
    assert all("sha256" not in frame for frame in evidence["frames"])


def test_native_video_input_rejects_non_monotonic_timeline() -> None:
    try:
        _video(
            QwenVideoFrame(payload=b"a", frame_index=20, timestamp_seconds=2.0),
            QwenVideoFrame(payload=b"b", frame_index=10, timestamp_seconds=1.0),
        )
    except QwenNativeVideoInputError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("expected non-monotonic frames to fail")


def test_native_video_input_rejects_unbounded_frame_count() -> None:
    frames = tuple(
        QwenVideoFrame(
            payload=f"frame-{index}".encode(),
            frame_index=index,
            timestamp_seconds=index / 10.0,
        )
        for index in range(QWEN_NATIVE_VIDEO_MAX_FRAMES + 1)
    )
    try:
        _video(*frames)
    except QwenNativeVideoInputError as error:
        assert "frame count" in str(error)
    else:
        raise AssertionError("expected frame count cap to fail")
