from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import robata.inference.mage_video_runtime as mage_video_runtime
from robata.inference.mage_video_runtime import (
    MageVideoCodecDependencyError,
    MageVideoLoadProfile,
    MageVideoRuntime,
    MageVideoRuntimeError,
    MageVideoRuntimeIdentity,
)


class _FakeTensor:
    def __init__(self, token_count: int) -> None:
        self.shape = (1, token_count)
        self.moves: list[object] = []

    def to(self, target: object) -> _FakeTensor:
        self.moves.append(target)
        return self

    def __getitem__(self, index: object) -> _FakeTensor:
        assert isinstance(index, tuple)
        token_slice = index[1]
        assert isinstance(token_slice, slice)
        start = int(token_slice.start or 0)
        return _FakeTensor(self.shape[1] - start)


class _FakeProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] | None = None
        self.call_kwargs: dict[str, object] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "native-video-prompt"

    def __call__(self, **kwargs: object) -> dict[str, _FakeTensor]:
        self.call_kwargs = kwargs
        return {
            "input_ids": _FakeTensor(5),
            "pixel_values": _FakeTensor(1),
            "patch_positions": _FakeTensor(1),
        }

    def batch_decode(self, _tokens: _FakeTensor, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens is True
        return ["  codec answer  "]


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)


class _FakeModel:
    device = "cuda:0"

    def __init__(
        self,
        *,
        dtype: object = "bfloat16",
        quantization_config: object | None = None,
        is_loaded_in_4bit: bool | None = None,
    ) -> None:
        self.dtype = dtype
        self.config = SimpleNamespace(quantization_config=quantization_config)
        self.is_loaded_in_4bit = (
            quantization_config is not None if is_loaded_in_4bit is None else is_loaded_in_4bit
        )
        self.generate_kwargs: dict[str, object] | None = None
        self.evaluated = False

    def eval(self) -> _FakeModel:
        self.evaluated = True
        return self

    def generate(self, **kwargs: object) -> _FakeTensor:
        self.generate_kwargs = kwargs
        return _FakeTensor(8)


class _FakeCuda:
    @staticmethod
    def empty_cache() -> None:
        return None

    @staticmethod
    def is_available() -> bool:
        return True


class _FakeTorch:
    bfloat16 = "bfloat16"
    cuda = _FakeCuda()

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


class _FakeTransformers:
    def __init__(self, processor: _FakeProcessor, model: _FakeModel) -> None:
        self._processor = processor
        self._model = model
        self.model_kwargs: dict[str, object] | None = None
        outer = self

        class _AutoProcessor:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> _FakeProcessor:
                assert args
                assert kwargs == {"local_files_only": True, "trust_remote_code": True}
                return outer._processor

        class _AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> _FakeModel:
                assert args
                outer.model_kwargs = dict(kwargs)
                return outer._model

        self.AutoProcessor = _AutoProcessor
        self.AutoModelForCausalLM = _AutoModelForCausalLM
        self.BitsAndBytesConfig = _FakeBitsAndBytesConfig


def _fake_importer(
    *,
    transformers: _FakeTransformers,
    imported: list[str],
    bitsandbytes_available: bool = True,
) -> Any:
    def fake_import_module(name: str) -> object:
        imported.append(name)
        if name == "torch":
            return _FakeTorch()
        if name == "transformers":
            return transformers
        if name == "bitsandbytes" and bitsandbytes_available:
            return object()
        if name == "bitsandbytes":
            raise ImportError("bitsandbytes is not installed")
        raise AssertionError(f"unexpected optional import: {name}")

    return fake_import_module


def test_runtime_construction_does_not_import_or_load_optional_model_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    imported: list[str] = []
    processor = _FakeProcessor()
    model = _FakeModel()
    transformers = _FakeTransformers(processor, model)
    monkeypatch.setattr(
        mage_video_runtime,
        "import_module",
        _fake_importer(transformers=transformers, imported=imported),
    )

    runtime = MageVideoRuntime(model_directory=model_directory)

    assert imported == []
    assert runtime.loaded is False
    assert runtime.load_profile is MageVideoLoadProfile.NATIVE_BF16
    assert runtime.runtime_identity == MageVideoRuntimeIdentity(
        load_profile=MageVideoLoadProfile.NATIVE_BF16
    )


def test_runtime_calls_native_codec_processor_path_with_mocked_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    video_path = tmp_path / "segments" / "cam-01.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video-segment")
    codec_binary = tmp_path / "cv-preinfer"
    codec_binary.write_bytes(b"stub")
    monkeypatch.setenv("CV_PREINFER_BIN", str(codec_binary))

    processor = _FakeProcessor()
    model = _FakeModel()
    transformers = _FakeTransformers(processor, model)
    imported: list[str] = []
    monkeypatch.setattr(
        mage_video_runtime,
        "import_module",
        _fake_importer(transformers=transformers, imported=imported),
    )
    runtime = MageVideoRuntime(model_directory=model_directory)

    generated = runtime.generate(
        video_paths=[video_path],
        prompt="Describe the scene.",
        max_new_tokens=32,
        codec_config={
            "engine": "hevc",
            "max_pixels": 150_000,
            "target_canvas": 8,
            "preprocess_device": "cpu",
        },
    )

    assert imported == ["torch", "transformers"]
    assert runtime.loaded is True
    assert runtime.load_observation.execution_device == "cuda:0"
    assert runtime.load_observation.runtime_identity == runtime.runtime_identity
    assert generated.input_video_count == 1
    assert generated.prompt_tokens == 5
    assert generated.output_tokens == 3
    assert generated.output_text == "codec answer"
    assert processor.messages == [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Describe the scene."},
            ],
        }
    ]
    assert processor.call_kwargs is not None
    assert processor.call_kwargs["videos"] == [str(video_path.resolve())]
    assert processor.call_kwargs["video_backend"] == "codec"
    assert processor.call_kwargs["codec_config"] == {
        "engine": "hevc",
        "max_pixels": 150_000,
        "target_canvas": 8,
    }
    assert processor.call_kwargs["padding"] is True
    assert model.evaluated is True
    assert transformers.model_kwargs == {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["max_new_tokens"] == 32
    assert model.generate_kwargs["use_cache"] is True
    assert model.generate_kwargs["do_sample"] is False


def test_runtime_loads_bitsandbytes_nf4_profile_with_official_transformers_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    actual_quantization = _FakeBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    processor = _FakeProcessor()
    model = _FakeModel(quantization_config=actual_quantization)
    transformers = _FakeTransformers(processor, model)
    imported: list[str] = []
    monkeypatch.setattr(
        mage_video_runtime,
        "import_module",
        _fake_importer(transformers=transformers, imported=imported),
    )

    runtime = MageVideoRuntime(
        model_directory=model_directory,
        load_profile=MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4,
    )
    loaded = runtime.load()

    assert imported == ["torch", "transformers", "bitsandbytes"]
    assert loaded.runtime_identity == MageVideoRuntimeIdentity(
        load_profile=MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4
    )
    assert transformers.model_kwargs is not None
    assert transformers.model_kwargs["torch_dtype"] == "bfloat16"
    quantization = transformers.model_kwargs["quantization_config"]
    assert isinstance(quantization, _FakeBitsAndBytesConfig)
    assert quantization.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    }


def test_runtime_fails_actionably_when_bitsandbytes_profile_dependency_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    processor = _FakeProcessor()
    model = _FakeModel()
    transformers = _FakeTransformers(processor, model)
    imported: list[str] = []
    monkeypatch.setattr(
        mage_video_runtime,
        "import_module",
        _fake_importer(
            transformers=transformers,
            imported=imported,
            bitsandbytes_available=False,
        ),
    )
    runtime = MageVideoRuntime(
        model_directory=model_directory,
        load_profile="bitsandbytes_4bit_nf4_v1",
    )

    with pytest.raises(MageVideoRuntimeError, match="working bitsandbytes installation"):
        runtime.load()

    assert imported == ["torch", "transformers", "bitsandbytes"]
    assert runtime.loaded is False


def test_runtime_fails_closed_when_loaded_execution_does_not_match_declared_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    mismatched_quantization = _FakeBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    processor = _FakeProcessor()
    model = _FakeModel(quantization_config=mismatched_quantization)
    transformers = _FakeTransformers(processor, model)
    imported: list[str] = []
    monkeypatch.setattr(
        mage_video_runtime,
        "import_module",
        _fake_importer(transformers=transformers, imported=imported),
    )
    runtime = MageVideoRuntime(
        model_directory=model_directory,
        load_profile=MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4,
    )

    with pytest.raises(MageVideoRuntimeError, match="does not match declared load profile"):
        runtime.load()

    assert runtime.loaded is False


def test_runtime_rejects_profile_that_conflicts_with_declared_runtime_identity(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()

    with pytest.raises(MageVideoRuntimeError, match="does not match the declared runtime identity"):
        MageVideoRuntime(
            model_directory=model_directory,
            load_profile=MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4,
            runtime_identity=MageVideoRuntimeIdentity(
                load_profile=MageVideoLoadProfile.NATIVE_BF16
            ),
        )


def test_runtime_fails_fast_when_traditional_codec_dependency_is_missing(tmp_path: Path) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    video_path = tmp_path / "segment.mp4"
    video_path.write_bytes(b"video-segment")
    runtime = MageVideoRuntime(model_directory=model_directory)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("CV_PREINFER_BIN", str(tmp_path / "not-installed"))
        with pytest.raises(MageVideoCodecDependencyError, match="codec-video-prep"):
            runtime.generate(
                video_paths=[video_path],
                prompt="Describe the scene.",
                max_new_tokens=8,
                codec_config={
                    "engine": "hevc",
                    "max_pixels": 150_000,
                    "preprocess_device": "cpu",
                },
            )

    assert runtime.loaded is False


def test_runtime_fails_fast_when_neural_codec_bundle_is_missing(tmp_path: Path) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    video_path = tmp_path / "segment.mp4"
    video_path.write_bytes(b"video-segment")
    runtime = MageVideoRuntime(model_directory=model_directory)

    with pytest.raises(MageVideoCodecDependencyError, match="dcvc_readiness_gen"):
        runtime.generate(
            video_paths=[video_path],
            prompt="Describe the scene.",
            max_new_tokens=8,
            codec_config={
                "engine": "dcvc-rt",
                "max_pixels": 150_000,
                "preprocess_device": "cpu",
                "dcvc": {},
            },
        )


def test_runtime_reserves_sequence_interface_but_rejects_multi_video_v1(tmp_path: Path) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    runtime = MageVideoRuntime(model_directory=model_directory)

    with pytest.raises(MageVideoRuntimeError, match="exactly one video path"):
        runtime.generate(
            video_paths=[first, second],
            prompt="Describe the scene.",
            max_new_tokens=8,
            codec_config={"engine": "hevc", "max_pixels": 150_000},
        )
