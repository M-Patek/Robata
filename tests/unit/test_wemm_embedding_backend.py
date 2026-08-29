from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import robata.benchmark.wemm_embedding_backend as backend_module  # noqa: E402
from robata.benchmark.wemm_embedding_backend import (  # noqa: E402
    WEMM_IMAGE_SIZE,
    WEMM_VIDEO_SIZE,
    WemmBackendUnavailable,
    WemmEmbeddingBackend,
    _tensor_to_rows,
)


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return "<embedding>"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "pixel_values_videos": torch.ones((2, 2)),
            "video_grid_thw": torch.tensor([[2, 1, 1]]),
            "video_metadata": [{"fps": 2.0}],
        }


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = types.SimpleNamespace(matryoshka_dimensions=[64, 128, 256, 512, 1024, 2048])

    def eval(self):
        return self

    def embedding(self, **kwargs):
        self.calls.append(kwargs)
        assert "video_metadata" not in kwargs
        return torch.tensor([[3.0, 4.0]])


class _BatchProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        assert isinstance(messages, list)
        return ["<embedding>"] * len(messages)

    def __call__(self, **kwargs):
        count = len(kwargs["text"])
        return {
            "input_ids": torch.ones((count, 2), dtype=torch.long),
            "attention_mask": torch.ones((count, 2), dtype=torch.long),
        }


class _BatchModel:
    def embedding(self, **kwargs):
        count = int(kwargs["input_ids"].shape[0])
        return torch.tensor([[3.0, 4.0]]).repeat(count, 1)


class _AnchorTokenizer:
    def get_vocab(self):
        return {"<embedding>": 9}

    def convert_tokens_to_ids(self, token):
        return 9 if token == "<embedding>" else 0


class _AnchorProcessor(_FakeProcessor):
    tokenizer = _AnchorTokenizer()

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        result["input_ids"] = torch.tensor([[1, 9]])
        result["attention_mask"] = torch.tensor([[1, 1]])
        return result


class _MissingAnchorProcessor(_AnchorProcessor):
    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        result["input_ids"] = torch.tensor([[1, 8]])
        return result


def _ready_backend(
    processor: _FakeProcessor | None = None,
    model: _FakeModel | None = None,
    *,
    dimension: int | None = 2,
) -> tuple[WemmEmbeddingBackend, _FakeProcessor, _FakeModel]:
    processor = processor or _FakeProcessor()
    model = model or _FakeModel()
    backend = WemmEmbeddingBackend(".", dimension=dimension)
    backend._processor = processor
    backend._model = model
    backend._torch = torch
    backend._functional = torch.nn.functional
    backend._device = "cpu"
    # A fake model has no local config file, so provide the dimensions that its
    # config advertises and exercise the same validation path as a real model.
    backend._supported_dimensions = (2,)
    return backend, processor, model


def test_encode_messages_drops_processor_metadata_before_embedding() -> None:
    backend, processor, model = _ready_backend()

    rows = backend._encode_messages(
        [
            {
                "role": "user",
                "content": [{"type": "video", "video": ["frame"]}],
            }
        ],
        modality="video",
        videos=[["frame"]],
        video_metadata=[{"fps": 2.0}],
    )

    assert rows[0] == pytest.approx((0.6, 0.8))
    assert processor.calls[0]["videos_kwargs"]["video_metadata"] == [{"fps": 2.0}]
    assert "video_metadata" not in model.calls[0]
    assert backend.observations[-1].video_grid_thw == ((2, 1, 1),)
    assert "video_metadata" not in backend.observations[-1].input_keys


def test_encode_messages_checks_final_embedding_anchor_when_tokenizer_exposes_it() -> None:
    backend, _, _ = _ready_backend(_AnchorProcessor())
    rows = backend._encode_messages(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        modality="text",
    )
    assert rows[0] == pytest.approx((0.6, 0.8))


def test_encode_messages_rejects_truncated_embedding_anchor() -> None:
    backend, _, _ = _ready_backend(_MissingAnchorProcessor())
    with pytest.raises(WemmBackendUnavailable, match=r"embedding.*anchor"):
        backend._encode_messages(
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            modality="text",
        )


def test_encode_images_uses_modality_scoped_resize_size() -> None:
    backend, processor, _ = _ready_backend()

    rows = backend.encode_images(["frame.jpg"])

    assert rows[0] == pytest.approx((0.6, 0.8))
    assert processor.calls[-1]["images_kwargs"] == {"size": WEMM_IMAGE_SIZE}
    assert "min_pixels" not in processor.calls[-1]
    assert "max_pixels" not in processor.calls[-1]


def test_encode_video_frames_uses_modality_scoped_resize_size() -> None:
    backend, processor, _ = _ready_backend()

    rows = backend.encode_video_frames(
        [["frame-0", "frame-1"]],
        metadata_groups=[{"fps": 5.0, "frames_indices": [8, 9]}],
    )

    assert rows[0] == pytest.approx((0.6, 0.8))
    assert processor.calls[-1]["videos_kwargs"]["size"] == WEMM_VIDEO_SIZE
    assert processor.calls[-1]["videos_kwargs"]["do_sample_frames"] is False
    assert "total_pixels" not in processor.calls[-1]


def test_encode_video_frames_filters_runner_only_metadata() -> None:
    backend, processor, _ = _ready_backend()

    backend.encode_video_frames(
        [["frame-0", "frame-1"]],
        metadata_groups=[
            {
                "fps": 5.0,
                "frames_indices": [8, 9],
                "total_num_frames": 20,
                "width": 1920,
                "height": 1080,
                "source_window_start_seconds": 1.25,
                "source_window_end_seconds": 2.0,
                "intervention": "normal",
            }
        ],
    )

    metadata = processor.calls[-1]["videos_kwargs"]["video_metadata"][0]
    assert metadata["fps"] == 5.0
    assert metadata["frames_indices"] == [8, 9]
    assert "source_window_start_seconds" not in metadata
    assert "source_window_end_seconds" not in metadata
    assert "intervention" not in metadata


def test_encode_video_frames_without_metadata_uses_bounded_synthetic_metadata() -> None:
    backend, processor, _ = _ready_backend()

    backend.encode_video_frames([["frame-0", "frame-1"]])

    scoped = processor.calls[-1]["videos_kwargs"]
    assert scoped["return_metadata"] is True
    assert scoped["video_metadata"][0]["total_num_frames"] == 2
    assert scoped["video_metadata"][0]["fps"] == 1.0


def test_encode_video_frames_consumes_groups_incrementally() -> None:
    backend, processor, _ = _ready_backend()
    yielded: list[int] = []

    def groups():
        for index in range(3):
            yielded.append(index)
            yield [f"frame-{index}-0", f"frame-{index}-1"]

    rows = backend.encode_video_frames(groups())

    assert len(rows) == 3
    assert len(processor.calls) == 3
    assert yielded == [0, 1, 2]


def test_encode_videos_merges_video_kwargs_without_duplicate_metadata(monkeypatch) -> None:
    processor = _FakeProcessor()
    model = _FakeModel()
    backend, processor, model = _ready_backend(processor, model)

    def process_vision_info(*args, **kwargs):
        assert kwargs["return_video_metadata"] is True
        return None, [(["frame"], {"fps": 5.0})], {"fps": 5.0}

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(process_vision_info=process_vision_info),
    )

    rows = backend.encode_videos([Path("clip.mp4")], frame_counts=[1])

    assert len(rows) == 1
    assert processor.calls[-1]["videos_kwargs"]["video_metadata"] == [{"fps": 5.0}]
    assert processor.calls[-1]["videos_kwargs"]["fps"] == 5.0
    assert "fps" not in processor.calls[-1]
    assert "video_metadata" not in model.calls[-1]


def test_encode_videos_rejects_frame_count_mismatch() -> None:
    backend, _, _ = _ready_backend()
    with pytest.raises(ValueError, match="frame_counts"):
        backend.encode_videos([Path("one.mp4")], frame_counts=[1, 2])


def test_constructor_rejects_non_integer_dimension() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        WemmEmbeddingBackend(".", dimension=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        WemmEmbeddingBackend(".", dimension=True)  # type: ignore[arg-type]


def test_dimension_uses_local_model_card_and_identity(monkeypatch) -> None:
    config = {"matryoshka_dimensions": [64, 128, 256, 512, 1024, 2048]}
    monkeypatch.setattr(backend_module, "_read_local_config", lambda _path: config)
    backend = WemmEmbeddingBackend("WeMM-Embedding-2B", dimension=300)
    assert "WeMM-Embedding-2B@" in backend.identity
    assert backend.variant == "2B"
    assert backend._supported_dimensions == (64, 128, 256, 512, 1024, 2048)
    assert backend.supported_dimensions == (64, 128, 256, 512, 1024, 2048)
    with pytest.raises(WemmBackendUnavailable, match="not supported"):
        _tensor_to_rows(
            torch.ones((1, 4)),
            torch=torch,
            functional=torch.nn.functional,
            dimension=300,
            supported_dimensions=backend._supported_dimensions,
        )


def test_tensor_dimension_rejects_unsupported_mrl_value() -> None:
    backend, _, _ = _ready_backend(dimension=300)
    with pytest.raises(WemmBackendUnavailable, match="not supported"):
        backend._encode_messages(
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            modality="text",
        )


def test_encode_texts_batches_and_preserves_order() -> None:
    backend = WemmEmbeddingBackend(".", dimension=2)
    backend._processor = _BatchProcessor()
    backend._model = _BatchModel()
    backend._torch = torch
    backend._functional = torch.nn.functional
    backend._device = "cpu"
    backend._supported_dimensions = (2,)

    rows = backend.encode_texts(["open door", "close door", "wash plate"], batch_size=2)

    assert len(rows) == 3
    assert rows[0] == pytest.approx((0.6, 0.8))
    assert [item.item_count for item in backend.observations] == [2, 1]
    with pytest.raises(ValueError, match="batch_size"):
        backend.encode_texts(["open door"], batch_size=0)


def test_encode_texts_cached_reuses_resident_text_prototypes() -> None:
    backend = WemmEmbeddingBackend(".", dimension=2)
    backend._processor = _BatchProcessor()
    backend._model = _BatchModel()
    backend._torch = torch
    backend._functional = torch.nn.functional
    backend._device = "cpu"
    backend._supported_dimensions = (2,)

    first = backend.encode_texts_cached(["open door", "close door"], batch_size=2)
    observations_after_first = len(backend.observations)
    second = backend.encode_texts_cached(["open door", "close door"], batch_size=2)

    assert second == first
    assert len(backend.observations) == observations_after_first
    assert backend.text_prototype_cache_stats() == {"entries": 1, "hits": 1, "misses": 1}
