from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from robata.benchmark.wemm_embedding_backend import WemmEmbeddingBackend  # noqa: E402


class _BatchProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return ["<embedding>"] * len(messages)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        videos = kwargs["videos"]
        count = len(videos)
        # Distinguish rows by the first frame token while keeping all tensors
        # bounded.  The fake model uses ``sample_ids`` to emit deterministic
        # row-ordered vectors.
        ids = torch.tensor(
            [[float(str(video[0]).split("-")[1])] for video in videos],
            dtype=torch.float32,
        )
        return {
            "input_ids": torch.ones((count, 2), dtype=torch.long),
            "attention_mask": torch.ones((count, 2), dtype=torch.long),
            "pixel_values_videos": torch.ones((count, 2, 3)),
            "video_grid_thw": torch.tensor([[2, 14 + index, 16] for index in range(count)]),
            "sample_ids": ids,
            "video_metadata": kwargs["videos_kwargs"]["video_metadata"],
        }


class _BatchModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def embedding(self, **kwargs):
        self.calls.append(kwargs)
        assert "video_metadata" not in kwargs
        ids = kwargs["sample_ids"]
        # Every row is distinct and already finite; backend normalization keeps
        # the public output deterministic.
        return torch.cat((ids + 1.0, ids + 2.0), dim=1)


def _backend() -> tuple[WemmEmbeddingBackend, _BatchProcessor, _BatchModel]:
    processor = _BatchProcessor()
    model = _BatchModel()
    backend = WemmEmbeddingBackend(".", dimension=2)
    backend._processor = processor
    backend._model = model
    backend._torch = torch
    backend._functional = torch.nn.functional
    backend._device = "cpu"
    backend._supported_dimensions = (2,)
    return backend, processor, model


def _groups(count: int) -> list[list[str]]:
    return [[f"frame-{index}-0", f"frame-{index}-1"] for index in range(count)]


def test_video_microbatch_preserves_order_and_pairs_metadata() -> None:
    backend, processor, model = _backend()
    groups = _groups(5)
    metadata = [
        {
            "total_num_frames": 2,
            "fps": 5.0,
            "frames_indices": [index, index + 1],
            "source_window_start_seconds": float(index),
        }
        for index in range(5)
    ]

    rows = backend.encode_video_frames_batch(
        groups,
        metadata_groups=metadata,
        batch_size=2,
    )

    assert len(rows) == 5
    assert len(processor.calls) == 3
    assert len(model.calls) == 3
    assert [len(call["videos"]) for call in processor.calls] == [2, 2, 1]
    assert [
        item["frames_indices"]
        for call in processor.calls
        for item in call["videos_kwargs"]["video_metadata"]
    ] == [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    # Rows are emitted in input order even though they are produced in chunks.
    assert rows[0] == pytest.approx((1 / (1**2 + 2**2) ** 0.5, 2 / 5**0.5))
    assert rows[-1] == pytest.approx((5 / (5**2 + 6**2) ** 0.5, 6 / 61**0.5))

    observations = backend.observation_payload()
    assert [item["batch_size"] for item in observations] == [2, 2, 1]
    assert [item["batch"] for item in observations] == [2, 2, 1]
    assert [item["batch_index"] for item in observations] == [0, 1, 2]
    assert observations[0]["frame_counts"] == [2, 2]
    assert observations[0]["video_grid_thw"] == [[2, 14, 16], [2, 15, 16]]
    assert observations[0]["processor_tensor_shapes"]["input_ids"] == [2, 2]
    assert set(observations[0]["phase_timings"]) >= {
        "prepare",
        "processor",
        "model",
        "postprocess",
        "total",
    }
    assert all(value >= 0 for value in observations[0]["phase_timings"].values())


def test_video_microbatch_consumes_generators_incrementally() -> None:
    backend, processor, _ = _backend()
    yielded: list[int] = []

    def groups():
        for index in range(5):
            yielded.append(index)
            yield [f"frame-{index}-0", f"frame-{index}-1"]

    rows = backend.encode_video_frames_batch(groups(), batch_size=4)

    assert len(rows) == 5
    assert yielded == [0, 1, 2, 3, 4]
    assert [len(call["videos"]) for call in processor.calls] == [4, 1]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_video_microbatch_rejects_invalid_batch_size(batch_size: object) -> None:
    backend, _, _ = _backend()
    with pytest.raises(ValueError, match="batch_size"):
        backend.encode_video_frames_batch([], batch_size=batch_size)  # type: ignore[arg-type]


def test_video_microbatch_rejects_metadata_length_mismatch() -> None:
    backend, _, _ = _backend()
    with pytest.raises(ValueError, match="metadata_groups"):
        backend.encode_video_frames_batch(
            _groups(2),
            metadata_groups=[{"fps": 1.0}],
            batch_size=2,
        )
    with pytest.raises(ValueError, match="metadata_groups"):
        backend.encode_video_frames_batch(
            _groups(1),
            metadata_groups=[{"fps": 1.0}, {"fps": 1.0}],
            batch_size=2,
        )


def test_serial_video_method_keeps_singleton_observation_shape() -> None:
    backend, processor, _ = _backend()
    backend.encode_video_frames(_groups(1))
    assert len(processor.calls) == 1
    observation = backend.observation_payload()[0]
    assert "batch_size" not in observation
    assert "phase_timings" not in observation
