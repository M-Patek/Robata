from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from robata.benchmark.wemm_embedding_backend import (  # noqa: E402
    WEMM_VIDEO_KWARGS,
    WEMM_VIDEO_SIZE,
    WemmEmbeddingBackend,
)


class _Processor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages: list[Any] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        self.messages.append(messages)
        return "<embedding>"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        count = len(kwargs.get("videos", [])) or 1
        return {
            "input_ids": torch.ones((count, 2), dtype=torch.long),
            "attention_mask": torch.ones((count, 2), dtype=torch.long),
            "pixel_values_videos": torch.ones((count, 2, 3)),
            "video_grid_thw": torch.tensor([[2, 14, 16]] * count),
        }


class _Model:
    def embedding(self, **kwargs):
        count = int(kwargs["input_ids"].shape[0])
        return torch.ones((count, 2), dtype=torch.float32)


def _ready(*, video_min_pixels: int | None = None, video_max_pixels: int | None = None):
    processor = _Processor()
    backend = WemmEmbeddingBackend(
        ".",
        dimension=2,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
    )
    backend._processor = processor
    backend._model = _Model()
    backend._torch = torch
    backend._functional = torch.nn.functional
    backend._device = "cpu"
    backend._supported_dimensions = (2,)
    return backend, processor


def test_default_pixel_budget_preserves_existing_wire_shape() -> None:
    backend, processor = _ready()
    backend.encode_video_frames([["frame-0", "frame-1"]])

    assert backend.video_min_pixels == WEMM_VIDEO_KWARGS["min_pixels"]
    assert backend.video_max_pixels == WEMM_VIDEO_KWARGS["max_pixels"]
    assert processor.calls[-1]["videos_kwargs"]["size"] == WEMM_VIDEO_SIZE
    observation = backend.observation_payload()[-1]
    assert "video_min_pixels" not in observation
    assert "video_max_pixels" not in observation
    assert "video_size" not in observation


def test_custom_pixel_budget_reaches_singleton_processor_and_observation() -> None:
    backend, processor = _ready(video_max_pixels=524_288)
    backend.encode_video_frames([["frame-0", "frame-1"]])

    assert processor.calls[-1]["videos_kwargs"]["size"] == {
        "shortest_edge": WEMM_VIDEO_KWARGS["min_pixels"],
        "longest_edge": 524_288,
    }
    content = processor.messages[-1][0]["content"][0]
    assert content["max_pixels"] == 524_288
    observation = backend.observation_payload()[-1]
    assert observation["video_max_pixels"] == 524_288
    assert observation["video_size"]["longest_edge"] == 524_288
    assert observation["video_grid_thw"] == [[2, 14, 16]]


def test_custom_pixel_budget_reaches_batch_processor_and_observation() -> None:
    backend, processor = _ready(video_min_pixels=8_192, video_max_pixels=524_288)
    backend.encode_video_frames_batch(
        [["frame-0", "frame-1"], ["frame-2", "frame-3"]],
        batch_size=2,
    )

    assert processor.calls[-1]["videos_kwargs"]["size"] == {
        "shortest_edge": 8_192,
        "longest_edge": 524_288,
    }
    assert all(
        message[0]["content"][0]["max_pixels"] == 524_288 for message in processor.messages[-1]
    )
    observation = backend.observation_payload()[-1]
    assert observation["video_min_pixels"] == 8_192
    assert observation["video_max_pixels"] == 524_288
    assert observation["video_size"] == {
        "shortest_edge": 8_192,
        "longest_edge": 524_288,
    }


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_constructor_rejects_invalid_pixel_budget(value: object) -> None:
    with pytest.raises(ValueError, match="video_max_pixels"):
        WemmEmbeddingBackend(".", video_max_pixels=value)  # type: ignore[arg-type]


def test_constructor_rejects_minimum_above_maximum() -> None:
    with pytest.raises(ValueError, match="video_min_pixels"):
        WemmEmbeddingBackend(".", video_min_pixels=10_000, video_max_pixels=9_000)
