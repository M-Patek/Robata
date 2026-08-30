from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robata.inference.local_hf_runtime import (
    LOCAL_HF_MAX_BATCH_REQUESTS,
    LOCAL_HF_MAX_VIDEO_FRAMES,
    LocalHfBatchGenerationRequest,
    LocalHfGenerationObservation,
    LocalHfLoadObservation,
    LocalHfVideoGenerationRequest,
    LocalHuggingFaceRuntimeError,
    LocalHuggingFaceVisionRuntime,
)


@dataclass
class _FakeScalar:
    value: int

    def item(self) -> int:
        return self.value


class _FakeRow:
    def __init__(self, values: list[int]) -> None:
        self.values = list(values)
        self.shape = (len(values),)

    def sum(self) -> _FakeScalar:
        return _FakeScalar(sum(self.values))

    def __ne__(self, other: object) -> _FakeRow:
        return _FakeRow([int(value != other) for value in self.values])


class _FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        if rows and any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("fake tensor rows must have equal width")
        self.rows = [list(row) for row in rows]
        self.shape = (len(rows), len(rows[0]) if rows else 0)
        self.moved_to: list[str] = []

    def to(self, device: str) -> _FakeTensor:
        self.moved_to.append(device)
        return self

    def __getitem__(self, key: object) -> _FakeTensor | _FakeRow:
        if isinstance(key, int):
            return _FakeRow(self.rows[key])
        if isinstance(key, tuple) and len(key) == 2:
            row_key, column_key = key
            selected_rows = (
                self.rows[row_key] if isinstance(row_key, slice) else [self.rows[row_key]]
            )
            return _FakeTensor([list(row[column_key]) for row in selected_rows])
        raise TypeError(f"unsupported fake tensor key: {key!r}")


class _FakeInferenceMode(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _FakeCuda:
    def __init__(self, *, peak_bytes: int = 123_456) -> None:
        self.peak_bytes = peak_bytes
        self.reset_calls = 0
        self.empty_cache_calls = 0

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def max_memory_allocated(self) -> int:
        return self.peak_bytes

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self, *, peak_bytes: int = 123_456) -> None:
        self.cuda = _FakeCuda(peak_bytes=peak_bytes)

    def inference_mode(self) -> _FakeInferenceMode:
        return _FakeInferenceMode()


class _FakeImage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.width = int(payload[0])
        self.height = 100
        self.closed = False
        self.thumbnail_calls: list[tuple[int, int]] = []

    def thumbnail(self, size: tuple[int, int]) -> None:
        self.thumbnail_calls.append(size)
        self.width = min(self.width, size[0])
        self.height = min(self.height, size[1])

    def close(self) -> None:
        self.closed = True


class _FakeSourceImage(AbstractContextManager["_FakeSourceImage"]):
    def __init__(self, payload: bytes, module: _FakeImageModule) -> None:
        self.payload = payload
        self.module = module
        self.closed = False

    def __enter__(self) -> _FakeSourceImage:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def convert(self, mode: str) -> _FakeImage:
        assert mode == "RGB"
        image = _FakeImage(self.payload)
        self.module.converted_images.append(image)
        return image


class _FakeImageModule:
    def __init__(self) -> None:
        self.sources: list[_FakeSourceImage] = []
        self.converted_images: list[_FakeImage] = []

    def open(self, stream: BytesIO) -> _FakeSourceImage:
        payload = stream.read()
        if payload == b"bad":
            raise OSError("not an image")
        source = _FakeSourceImage(payload, self)
        self.sources.append(source)
        return source


class _FakeProcessor:
    def __init__(
        self,
        *,
        input_rows: list[list[int]],
        attention_rows: list[list[int]],
        decoded: list[str],
        pad_token_id: int = 0,
    ) -> None:
        self.input_rows = input_rows
        self.attention_rows = attention_rows
        self.decoded = decoded
        self.tokenizer = SimpleNamespace(pad_token_id=pad_token_id, padding_side="right")
        self.template_members: list[tuple[str, tuple[bytes, ...]]] = []
        self.processor_calls: list[dict[str, object]] = []
        self.decoded_rows: list[list[list[int]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        assert isinstance(content, list)
        prompt = str(content[-1]["text"])
        image_payloads = tuple(item["image"].payload for item in content[:-1])
        self.template_members.append((prompt, image_payloads))
        return f"template:{prompt}"

    def __call__(
        self,
        *,
        text: list[str],
        images: list[_FakeImage],
        return_tensors: str,
        padding: bool | None = None,
    ) -> dict[str, _FakeTensor]:
        assert return_tensors == "pt"
        self.processor_calls.append(
            {
                "text": list(text),
                "images": [image.payload for image in images],
                "padding": padding,
            }
        )
        return {
            "input_ids": _FakeTensor(self.input_rows),
            "attention_mask": _FakeTensor(self.attention_rows),
        }

    def batch_decode(
        self,
        values: _FakeTensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        assert skip_special_tokens is True
        self.decoded_rows.append(values.rows)
        return list(self.decoded)


class _FakeNativeVideoProcessor(_FakeProcessor):
    def __init__(self) -> None:
        super().__init__(
            input_rows=[[1, 2, 3]],
            attention_rows=[[1, 1, 1]],
            decoded=['{"native":true}'],
        )
        self.video_template_seen = False
        self.video_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "video"
        assert content[0]["video"]
        assert content[-1] == {"type": "text", "text": "native prompt"}
        self.video_template_seen = True
        return "template:native"

    def __call__(
        self,
        *,
        text: list[str],
        videos: list[list[_FakeImage]],
        video_metadata: list[dict[str, object]],
        do_sample_frames: bool,
        return_metadata: bool,
        return_tensors: str,
    ) -> dict[str, object]:
        assert text == ["template:native"]
        assert len(videos) == 1
        assert len(videos[0]) == 2
        assert do_sample_frames is False
        assert return_metadata is True
        assert return_tensors == "pt"
        self.video_calls.append(
            {
                "videos": [[image.payload for image in video] for video in videos],
                "video_metadata": video_metadata,
                "do_sample_frames": do_sample_frames,
                "return_metadata": return_metadata,
            }
        )
        return {
            "input_ids": _FakeTensor([[1, 2, 3]]),
            "attention_mask": _FakeTensor([[1, 1, 1]]),
            "pixel_values_videos": _FakeTensor([[4]]),
            "video_grid_thw": _FakeTensor([[2, 1, 1]]),
            "video_metadata": video_metadata,
        }


class _FakeModel:
    def __init__(
        self,
        generated_rows: list[list[int]],
        *,
        failure: Exception | None = None,
    ) -> None:
        self.generated = _FakeTensor(generated_rows)
        self.failure = failure
        self.generate_calls: list[dict[str, object]] = []
        self.generation_config = SimpleNamespace(pad_token_id=None)
        self.config = SimpleNamespace(pad_token_id=None)

    def parameters(self) -> Any:
        return iter([SimpleNamespace(device="cuda:0")])

    def generate(self, **kwargs: object) -> _FakeTensor:
        self.generate_calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return self.generated


def _loaded_runtime(
    tmp_path: Path,
    *,
    processor: _FakeProcessor,
    model: _FakeModel,
    image_module: _FakeImageModule | None = None,
) -> tuple[LocalHuggingFaceVisionRuntime, _FakeTorch, _FakeImageModule]:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    runtime = LocalHuggingFaceVisionRuntime(
        model_directory=model_directory,
        offload_directory=tmp_path / "offload",
    )
    torch = _FakeTorch()
    images = image_module or _FakeImageModule()
    runtime._torch = torch
    runtime._processor = processor
    runtime._model = model
    runtime._image_module = images
    runtime._load_observation = LocalHfLoadObservation(
        load_seconds=1.0,
        gpu_name="fake-gpu",
        gpu_total_bytes=8_000,
        gpu_free_before_bytes=7_000,
        gpu_allocated_after_load_bytes=3_000,
    )
    return runtime, torch, images


def _request(
    *payloads: bytes,
    prompt: str = "describe",
    max_new_tokens: int = 12,
) -> LocalHfBatchGenerationRequest:
    return LocalHfBatchGenerationRequest(
        image_payloads=payloads,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


def test_generate_batch_uses_one_physical_call_and_preserves_member_order(
    tmp_path: Path,
) -> None:
    processor = _FakeProcessor(
        input_rows=[[1, 2, 0, 0], [3, 4, 5, 6]],
        attention_rows=[[1, 1, 0, 0], [1, 1, 1, 1]],
        decoded=['{"member":0}', '{"member":1}'],
    )
    model = _FakeModel(
        generated_rows=[
            [1, 2, 0, 0, 101, 102, 0],
            [3, 4, 5, 6, 201, 202, 203],
        ]
    )
    runtime, torch, image_module = _loaded_runtime(
        tmp_path,
        processor=processor,
        model=model,
    )

    observation = runtime.generate_batch(
        requests=(
            _request(b"a", b"b", prompt="short"),
            _request(b"c", prompt="longer prompt"),
        )
    )

    assert len(model.generate_calls) == 1
    physical_call = model.generate_calls[0]
    assert physical_call["max_new_tokens"] == 12
    assert physical_call["do_sample"] is False
    assert physical_call["use_cache"] is True
    assert processor.template_members == [
        ("short", (b"a", b"b")),
        ("longer prompt", (b"c",)),
    ]
    assert processor.processor_calls == [
        {
            "text": ["template:short", "template:longer prompt"],
            "images": [b"a", b"b", b"c"],
            "padding": True,
        }
    ]
    assert processor.decoded_rows == [[[101, 102, 0], [201, 202, 203]]]
    assert [member.rendered_image_sizes for member in observation.members] == [
        ((97, 100), (98, 100)),
        ((99, 100),),
    ]
    assert [member.prompt_tokens for member in observation.members] == [2, 4]
    assert [member.output_tokens for member in observation.members] == [2, 3]
    assert [member.output_text for member in observation.members] == [
        '{"member":0}',
        '{"member":1}',
    ]
    assert observation.physical_generation_seconds >= 0
    assert observation.physical_gpu_peak_allocated_bytes == 123_456
    assert torch.cuda.reset_calls == 1
    assert all(source.closed for source in image_module.sources)
    assert all(image.closed for image in image_module.converted_images)


@pytest.mark.parametrize(
    ("requests", "message"),
    [
        ((), "at least one batch request"),
        (
            tuple(_request(b"a") for _ in range(LOCAL_HF_MAX_BATCH_REQUESTS + 1)),
            "batch request count",
        ),
        ((_request(b"a", prompt="  "),), "prompt must be nonempty"),
        ((_request(),), "requires at least one image payload"),
        ((_request(*(b"a",) * 7),), "image count must not exceed 6"),
        ((_request(b""),), "must be nonempty bytes"),
        (
            (
                _request(b"a", max_new_tokens=12),
                _request(b"b", max_new_tokens=13),
            ),
            "same max_new_tokens",
        ),
        ((_request(b"a", max_new_tokens=0),), "must be a positive integer"),
    ],
)
def test_generate_batch_fails_closed_before_loading_for_invalid_requests(
    tmp_path: Path,
    requests: tuple[LocalHfBatchGenerationRequest, ...],
    message: str,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    runtime = LocalHuggingFaceVisionRuntime(
        model_directory=model_directory,
        offload_directory=tmp_path / "offload",
    )

    with pytest.raises(LocalHuggingFaceRuntimeError, match=message):
        runtime.generate_batch(requests=requests)

    assert runtime.loaded is False


def test_generate_batch_closes_every_image_when_physical_generation_fails(
    tmp_path: Path,
) -> None:
    processor = _FakeProcessor(
        input_rows=[[1, 0], [1, 2]],
        attention_rows=[[1, 0], [1, 1]],
        decoded=["unused", "unused"],
    )
    model = _FakeModel(
        generated_rows=[[1, 0, 2], [1, 2, 3]],
        failure=RuntimeError("physical failure"),
    )
    runtime, _, image_module = _loaded_runtime(
        tmp_path,
        processor=processor,
        model=model,
    )

    with pytest.raises(RuntimeError, match="physical failure"):
        runtime.generate_batch(
            requests=(
                _request(b"a", b"b"),
                _request(b"c"),
            )
        )

    assert len(model.generate_calls) == 1
    assert len(image_module.converted_images) == 3
    assert all(source.closed for source in image_module.sources)
    assert all(image.closed for image in image_module.converted_images)


def test_generate_batch_closes_prior_images_when_later_payload_cannot_decode(
    tmp_path: Path,
) -> None:
    processor = _FakeProcessor(
        input_rows=[[1]],
        attention_rows=[[1]],
        decoded=["unused"],
    )
    model = _FakeModel(generated_rows=[[1, 2]])
    runtime, _, image_module = _loaded_runtime(
        tmp_path,
        processor=processor,
        model=model,
    )

    with pytest.raises(LocalHuggingFaceRuntimeError, match="member 0 image payload 1"):
        runtime.generate_batch(requests=(_request(b"a", b"bad"),))

    assert model.generate_calls == []
    assert len(image_module.converted_images) == 1
    assert image_module.converted_images[0].closed is True
    assert all(source.closed for source in image_module.sources)


def test_existing_serial_generate_behavior_is_unchanged(tmp_path: Path) -> None:
    processor = _FakeProcessor(
        input_rows=[[1, 2, 3]],
        attention_rows=[[1, 1, 1]],
        decoded=['{"serial":true}'],
    )
    model = _FakeModel(generated_rows=[[1, 2, 3, 9, 10]])
    runtime, _, image_module = _loaded_runtime(
        tmp_path,
        processor=processor,
        model=model,
    )

    observation = runtime.generate(
        image_payloads=[b"a"],
        prompt="serial prompt",
        max_new_tokens=7,
    )

    assert observation == LocalHfGenerationObservation(
        rendered_image_sizes=((97, 100),),
        prompt_tokens=3,
        output_tokens=2,
        generation_seconds=observation.generation_seconds,
        gpu_peak_allocated_bytes=123_456,
        output_text='{"serial":true}',
    )
    assert len(model.generate_calls) == 1
    assert model.generate_calls[0]["max_new_tokens"] == 7
    assert processor.template_members == [("serial prompt", (b"a",))]
    assert processor.processor_calls == [
        {
            "text": ["template:serial prompt"],
            "images": [b"a"],
            "padding": None,
        }
    ]
    assert processor.decoded_rows == [[[9, 10]]]
    assert all(source.closed for source in image_module.sources)
    assert all(image.closed for image in image_module.converted_images)


def _video_request(*payloads: bytes) -> LocalHfVideoGenerationRequest:
    return LocalHfVideoGenerationRequest(
        video_payloads=payloads,
        frame_indices=tuple(range(len(payloads))),
        frame_timestamps_seconds=tuple(index / 4.0 for index in range(len(payloads))),
        source_fps=4.0,
        total_num_frames=16,
        width=640,
        height=480,
        duration_seconds=4.0,
        prompt="native prompt",
        max_new_tokens=8,
    )


def test_generate_video_uses_native_video_tokens_and_preserves_metadata(tmp_path: Path) -> None:
    processor = _FakeNativeVideoProcessor()
    model = _FakeModel(generated_rows=[[1, 2, 3, 9, 10]])
    runtime, _, image_module = _loaded_runtime(
        tmp_path,
        processor=processor,
        model=model,
    )

    observation = runtime.generate_video(request=_video_request(b"a", b"b"))

    assert processor.video_template_seen is True
    assert processor.video_calls == [
        {
            "videos": [[b"a", b"b"]],
            "video_metadata": [
                {
                    "total_num_frames": 16,
                    "fps": 4.0,
                    "width": 640,
                    "height": 480,
                    "frames_indices": [0, 1],
                    "duration": 4.0,
                }
            ],
            "do_sample_frames": False,
            "return_metadata": True,
        }
    ]
    assert len(model.generate_calls) == 1
    assert "video_metadata" not in model.generate_calls[0]
    assert "pixel_values_videos" in model.generate_calls[0]
    assert "video_grid_thw" in model.generate_calls[0]
    assert observation.input_mode == "native_video"
    assert observation.frame_indices == (0, 1)
    assert observation.frame_timestamps_seconds == (0.0, 0.25)
    assert observation.frame_sha256[0] == __import__("hashlib").sha256(b"a").hexdigest()
    assert observation.output_text == '{"native":true}'
    assert all(source.closed for source in image_module.sources)
    assert all(image.closed for image in image_module.converted_images)


@pytest.mark.parametrize(
    ("video_request", "message"),
    [
        (_video_request(), "at least one frame"),
        (
            LocalHfVideoGenerationRequest(
                video_payloads=(b"a",),
                frame_indices=(0, 1),
                frame_timestamps_seconds=(0.0,),
                source_fps=4.0,
                total_num_frames=16,
                width=640,
                height=480,
                duration_seconds=4.0,
                prompt="native prompt",
                max_new_tokens=8,
            ),
            "frame_indices count",
        ),
        (
            LocalHfVideoGenerationRequest(
                video_payloads=tuple(b"a" for _ in range(LOCAL_HF_MAX_VIDEO_FRAMES + 1)),
                frame_indices=tuple(range(LOCAL_HF_MAX_VIDEO_FRAMES + 1)),
                frame_timestamps_seconds=tuple(
                    index / 4.0 for index in range(LOCAL_HF_MAX_VIDEO_FRAMES + 1)
                ),
                source_fps=4.0,
                total_num_frames=64,
                width=640,
                height=480,
                duration_seconds=16.0,
                prompt="native prompt",
                max_new_tokens=8,
            ),
            "frame count",
        ),
        (
            LocalHfVideoGenerationRequest(
                video_payloads=(b"a", b"b"),
                frame_indices=(0, 1),
                frame_timestamps_seconds=(0.0, 0.5),
                source_fps=4.0,
                total_num_frames=16,
                width=640,
                height=480,
                duration_seconds=4.0,
                prompt="native prompt",
                max_new_tokens=8,
            ),
            "does not match source fps",
        ),
    ],
)
def test_generate_video_fails_closed_before_loading_for_invalid_request(
    tmp_path: Path,
    video_request: LocalHfVideoGenerationRequest,
    message: str,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    runtime = LocalHuggingFaceVisionRuntime(
        model_directory=model_directory,
        offload_directory=tmp_path / "offload",
    )

    with pytest.raises(LocalHuggingFaceRuntimeError, match=message):
        runtime.generate_video(request=video_request)

    assert runtime.loaded is False
