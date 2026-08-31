from __future__ import annotations

from contextlib import AbstractContextManager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robata.inference.local_hf_runtime import (
    LocalHfLoadObservation,
    LocalHfVideoGenerationRequest,
    LocalHuggingFaceRuntimeError,
    LocalHuggingFaceVisionRuntime,
    _video_grid_rows,
)


class _Tensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def to(self, _device: Any) -> _Tensor:
        return self

    def __getitem__(self, key: object) -> _Tensor:
        if isinstance(key, tuple):
            row_key, col_key = key
            rows = self.rows[row_key] if isinstance(row_key, slice) else [self.rows[row_key]]
            return _Tensor([list(row[col_key]) for row in rows])
        if isinstance(key, int):
            return _Tensor([self.rows[key]])
        raise TypeError(key)

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def tolist(self) -> list[list[int]]:
        return self.rows


class _Cuda:
    bool = bool

    def reset_peak_memory_stats(self) -> None:
        pass

    def max_memory_allocated(self) -> int:
        return 123


class _Torch:
    cuda = _Cuda()
    bool = bool

    class _Inference(AbstractContextManager[None]):
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    def inference_mode(self) -> _Inference:
        return self._Inference()


class _Image:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.width = int(payload[0])
        self.height = 100
        self.closed = False

    def thumbnail(self, size: tuple[int, int]) -> None:
        self.width = min(self.width, size[0])
        self.height = min(self.height, size[1])

    def close(self) -> None:
        self.closed = True


class _Source(AbstractContextManager["_Source"]):
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def __enter__(self) -> _Source:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def convert(self, _mode: str) -> _Image:
        return _Image(self.payload)


class _Images:
    def __init__(self) -> None:
        self.sources: list[_Source] = []
        self.images: list[_Image] = []

    def open(self, stream: BytesIO) -> _Source:
        source = _Source(stream.read())
        self.sources.append(source)
        return source


class _Processor:
    def __init__(self) -> None:
        self.video_call: dict[str, Any] | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        assert messages[0]["content"][0]["type"] == "video"
        return "template"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.video_call = kwargs
        return {
            "input_ids": _Tensor([[1, 2, 3]]),
            "attention_mask": _Tensor([[1, 1, 1]]),
            "pixel_values_videos": _Tensor([[1, 2]]),
            "video_grid_thw": _Tensor([[2, 14, 16]]),
            "video_metadata": kwargs["video_metadata"],
        }

    def batch_decode(self, _values: _Tensor, **_kwargs: Any) -> list[str]:
        return ['{"ok":true}']


class _Model:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def parameters(self) -> Any:
        return iter([SimpleNamespace(device="cuda:0")])

    def generate(self, **kwargs: Any) -> _Tensor:
        self.calls.append(kwargs)
        return _Tensor([[1, 2, 3, 4, 5]])


def _runtime(tmp_path: Path) -> tuple[LocalHuggingFaceVisionRuntime, _Processor, _Model, _Images]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    runtime = LocalHuggingFaceVisionRuntime(
        model_directory=model_dir,
        offload_directory=tmp_path / "offload",
        max_image_side=64,
    )
    processor = _Processor()
    model = _Model()
    images = _Images()
    runtime._torch = _Torch()
    runtime._processor = processor
    runtime._model = model
    runtime._image_module = images
    runtime._load_observation = LocalHfLoadObservation(1.0, "fake", 1, 1, 1)
    return runtime, processor, model, images


def _request(**kwargs: Any) -> LocalHfVideoGenerationRequest:
    values: dict[str, Any] = {
        "video_payloads": (b"\x64a", b"\x65b"),
        "frame_indices": (8, 12),
        "frame_timestamps_seconds": (2.0, 3.0),
        "source_fps": 4.0,
        "total_num_frames": 40,
        "width": 640,
        "height": 480,
        "duration_seconds": 10.0,
        "source_window_start_seconds": 2.0,
        "source_window_end_seconds": 3.0,
        "prompt": "native",
        "max_new_tokens": 8,
        "stop_after_first_complete_json_object": True,
    }
    values.update(kwargs)
    return LocalHfVideoGenerationRequest(**values)


def test_generate_video_uses_complete_native_video_and_records_processor_evidence(
    tmp_path: Path,
) -> None:
    runtime, processor, model, images = _runtime(tmp_path)

    observation = runtime.generate_video(request=_request())

    assert processor.video_call is not None
    assert processor.video_call["do_sample_frames"] is False
    assert processor.video_call["video_metadata"][0]["frames_indices"] == [8, 12]
    assert model.calls[0]["pixel_values_videos"].shape == (1, 2)
    assert model.calls[0]["video_grid_thw"].shape == (1, 3)
    assert "video_metadata" not in model.calls[0]
    assert "stopping_criteria" in model.calls[0]
    assert observation.frame_sha256 == ()
    assert observation.source_window_start_seconds == 2.0
    assert observation.source_window_end_seconds == 3.0
    assert observation.visual_input is not None
    assert observation.visual_input.video_grid_thw == ((2, 14, 16),)
    assert observation.visual_input.processor_tensor_shapes[0][0] == "input_ids"
    assert all(source.closed for source in images.sources)


def test_generate_video_rejects_content_identifier_opt_in_before_loading(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    runtime = LocalHuggingFaceVisionRuntime(
        model_directory=model_dir,
        offload_directory=tmp_path / "offload",
    )
    with pytest.raises(LocalHuggingFaceRuntimeError, match="identifiers are disabled"):
        runtime.generate_video(request=_request(compute_frame_sha256=True))
    assert runtime.loaded is False


def test_video_grid_rows_require_qwen_thw_shape() -> None:
    with pytest.raises(LocalHuggingFaceRuntimeError, match="exactly \\[time, height, width\\]"):
        _video_grid_rows([[1, 2, 3, 4]])


def test_generate_video_rejects_uncovered_explicit_window(tmp_path: Path) -> None:
    runtime, _processor, _model, _images = _runtime(tmp_path)
    request = _request(
        source_window_start_seconds=1.0,
        source_window_end_seconds=4.0,
    )
    with pytest.raises(LocalHuggingFaceRuntimeError, match="cover the start"):
        runtime.generate_video(request=request)


def test_generate_video_accepts_complete_physical_tail_frame(tmp_path: Path) -> None:
    """The final frame covers the final frame-period of a complete video."""

    runtime, _processor, _model, _images = _runtime(tmp_path)
    request = _request(
        frame_indices=(8, 39),
        frame_timestamps_seconds=(2.0, 9.75),
        total_num_frames=40,
        source_fps=4.0,
        duration_seconds=10.0,
        source_window_start_seconds=2.0,
        source_window_end_seconds=10.0,
    )

    observation = runtime.generate_video(request=request)

    assert observation.source_window_end_seconds == 10.0


def test_generate_video_keeps_strict_end_coverage_before_physical_tail(tmp_path: Path) -> None:
    runtime, _processor, _model, _images = _runtime(tmp_path)
    request = _request(
        frame_indices=(8, 38),
        frame_timestamps_seconds=(2.0, 9.5),
        total_num_frames=40,
        source_fps=4.0,
        duration_seconds=10.0,
        source_window_start_seconds=2.0,
        source_window_end_seconds=10.0,
    )

    with pytest.raises(LocalHuggingFaceRuntimeError, match="cover the end"):
        runtime.generate_video(request=request)
