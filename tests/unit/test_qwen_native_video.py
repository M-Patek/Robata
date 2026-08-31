from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robata.benchmark import qwen_native_video
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


class _FakeImage:
    def __init__(self, frame_index: int) -> None:
        self.frame_index = frame_index
        self.closed = False

    def save(self, output: Any, *, format: str, quality: int) -> None:
        assert format == "JPEG"
        assert quality == 87
        output.write(f"jpeg-{self.frame_index}".encode())

    def close(self) -> None:
        self.closed = True


class _FakeFrame:
    def __init__(self, frame_index: int) -> None:
        self.frame_index = frame_index
        self.image = _FakeImage(frame_index)
        self.pts = frame_index
        self.time_base = 0.1

    def to_image(self) -> _FakeImage:
        return self.image


class _FakeContainer:
    def __init__(self, *, frame_count: int, frames_declared: int | None = None) -> None:
        stream = SimpleNamespace(
            average_rate=10.0,
            guessed_rate=None,
            base_rate=None,
            frames=frame_count if frames_declared is None else frames_declared,
            duration=frame_count,
            time_base=0.1,
            codec_context=SimpleNamespace(width=64, height=48),
        )
        self.streams = SimpleNamespace(video=[stream])
        self._frames = [_FakeFrame(index) for index in range(frame_count)]
        self.closed = False
        self.seek_calls: list[tuple[int, Any, bool, bool]] = []

    def decode(self, _stream: Any) -> list[_FakeFrame]:
        return self._frames

    def seek(self, position: int, *, stream: Any, any_frame: bool, backward: bool) -> None:
        self.seek_calls.append((position, stream, any_frame, backward))

    def close(self) -> None:
        self.closed = True


class _FakeAv:
    def __init__(self, container: _FakeContainer) -> None:
        self.container = container
        self.opened: list[tuple[str, str]] = []

    def open(self, path: str, *, mode: str) -> _FakeContainer:
        self.opened.append((path, mode))
        return self.container


def _patch_fake_av(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    container: _FakeContainer,
) -> _FakeAv:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"fixture")
    fake_av = _FakeAv(container)

    def import_module(name: str) -> Any:
        if name == "av":
            return fake_av
        raise AssertionError(f"unexpected optional import: {name}")

    monkeypatch.setattr(qwen_native_video.importlib, "import_module", import_module)
    return fake_av


def test_sampler_uses_pyav_sequential_decode_and_clamps_tail_interval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    container = _FakeContainer(frame_count=100)
    fake_av = _patch_fake_av(monkeypatch, tmp_path, container)

    video = qwen_native_video.sample_qwen_native_video(
        tmp_path / "fixture.mp4",
        start_seconds=8.0,
        end_seconds=12.0,
        frame_count=4,
        context_before_seconds=0.0,
        context_after_seconds=0.0,
        jpeg_quality=87,
    )

    assert fake_av.opened == [(str(tmp_path / "fixture.mp4"), "r")]
    assert container.closed is True
    assert container.seek_calls
    assert container.seek_calls[0][0] == 80
    assert container.seek_calls[0][2:] == (False, True)
    assert video.interval_start_seconds == 8.0
    assert video.interval_end_seconds == 10.0
    assert video.source_window_end_seconds == 10.0
    assert video.frame_indices[0] == 80
    assert video.frame_indices[-1] == 99
    assert video.frame_payloads[0] == b"jpeg-80"
    assert video.frame_payloads[-1] == b"jpeg-99"


def test_sampler_rejects_interval_after_physical_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    container = _FakeContainer(frame_count=100)
    _patch_fake_av(monkeypatch, tmp_path, container)

    with pytest.raises(QwenNativeVideoInputError, match="does not overlap"):
        qwen_native_video.sample_qwen_native_video(
            tmp_path / "fixture.mp4",
            start_seconds=10.0,
            end_seconds=11.0,
            context_before_seconds=0.0,
            context_after_seconds=0.0,
        )


def test_sampler_derives_missing_stream_frame_count_from_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    container = _FakeContainer(frame_count=30, frames_declared=0)
    _patch_fake_av(monkeypatch, tmp_path, container)

    video = qwen_native_video.sample_qwen_native_video(
        tmp_path / "fixture.mp4",
        start_seconds=1.0,
        end_seconds=2.0,
        frame_count=2,
        context_before_seconds=0.0,
        context_after_seconds=0.0,
        jpeg_quality=87,
    )

    assert video.total_num_frames == 30
    assert video.duration_seconds == 3.0


def test_sampler_accepts_single_frame_request_without_dividing_by_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    container = _FakeContainer(frame_count=30)
    _patch_fake_av(monkeypatch, tmp_path, container)

    video = qwen_native_video.sample_qwen_native_video(
        tmp_path / "fixture.mp4",
        start_seconds=1.0,
        end_seconds=2.0,
        frame_count=1,
        context_before_seconds=0.0,
        context_after_seconds=0.0,
        jpeg_quality=87,
    )

    assert video.frame_indices == (10,)
