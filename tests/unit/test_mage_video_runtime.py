from __future__ import annotations

import hashlib
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import pytest

import robata.inference.mage_video_runtime as mage_video_runtime
from robata.contracts.hashing import semantic_sha256
from robata.inference.device_execution_guard import ExclusiveFileDeviceGuard
from robata.inference.mage_video_runtime import (
    MageVideoCodecCacheBinding,
    MageVideoCodecDependencyError,
    MageVideoExactCodecCacheAsset,
    MageVideoLoadProfile,
    MageVideoRuntime,
    MageVideoRuntimeError,
    MageVideoRuntimeIdentity,
    MageVideoTraditionalCodecCacheBinding,
    mage_video_codec_config_sha256,
)


def _hold_shared_device_guard(path: str, ready: Any, release: Any) -> None:
    with ExclusiveFileDeviceGuard(Path(path)).hold():
        ready.set()
        if not release.wait(timeout=10.0):
            raise RuntimeError("test guard holder timed out")


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
        self.decoded_tokens: _FakeTensor | None = None

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

    def batch_decode(self, tokens: _FakeTensor, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens is True
        self.decoded_tokens = tokens
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
        stopping_criteria = kwargs.get("stopping_criteria")
        assert isinstance(stopping_criteria, list)
        for criterion in stopping_criteria:
            assert callable(criterion)
            assert criterion(_FakeTensor(6), None) is False
        return _FakeTensor(8)


class _OverlapProcessor(_FakeProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.second_prepared = Event()
        self.first_decode_entered = Event()
        self.release_first_decode = Event()
        self._call_count = 0
        self._decode_count = 0
        self._counter_lock = Lock()

    def __call__(self, **kwargs: object) -> dict[str, _FakeTensor]:
        result = super().__call__(**kwargs)
        with self._counter_lock:
            self._call_count += 1
            call_count = self._call_count
        if call_count == 2:
            self.second_prepared.set()
        return result

    def batch_decode(self, tokens: _FakeTensor, *, skip_special_tokens: bool) -> list[str]:
        with self._counter_lock:
            self._decode_count += 1
            decode_count = self._decode_count
        if decode_count == 1:
            self.first_decode_entered.set()
            if not self.release_first_decode.wait(timeout=5.0):
                raise AssertionError("first decode was not released")
        return super().batch_decode(tokens, skip_special_tokens=skip_special_tokens)


class _OverlapModel(_FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.first_generation_entered = Event()
        self.release_first_generation = Event()
        self.second_generation_entered = Event()
        self.release_second_generation = Event()
        self._generate_count = 0
        self._generate_count_lock = Lock()

    def generate(self, **kwargs: object) -> _FakeTensor:
        self.generate_kwargs = kwargs
        stopping_criteria = kwargs.get("stopping_criteria")
        assert isinstance(stopping_criteria, list)
        for criterion in stopping_criteria:
            assert callable(criterion)
            assert criterion(_FakeTensor(6), None) is False
        with self._generate_count_lock:
            self._generate_count += 1
            generation_count = self._generate_count
        if generation_count == 1:
            self.first_generation_entered.set()
            if not self.release_first_generation.wait(timeout=5.0):
                raise AssertionError("first generation was not released")
        elif generation_count == 2:
            self.second_generation_entered.set()
            if not self.release_second_generation.wait(timeout=5.0):
                raise AssertionError("second generation was not released")
        else:
            raise AssertionError("unexpected extra generation")
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
    assert generated.telemetry is not None
    assert generated.telemetry.generate_seconds == generated.generation_seconds
    assert generated.telemetry.total_request_seconds >= generated.generation_seconds
    assert generated.telemetry.processor_seconds >= 0.0
    assert generated.telemetry.input_materialization_seconds >= 0.0
    assert generated.telemetry.telemetry_version == "mage-video-generation-telemetry-v3"
    assert (
        generated.telemetry.generation_started_monotonic_seconds
        <= generated.telemetry.first_output_token_monotonic_seconds
        <= generated.telemetry.generation_completed_monotonic_seconds
    )
    assert generated.telemetry.time_to_first_token_seconds is not None
    assert generated.telemetry.time_to_first_token_seconds >= 0.0
    assert generated.telemetry.output_tokens_per_second is not None
    assert generated.telemetry.output_tokens_per_second > 0.0
    assert generated.telemetry.generate_seconds == pytest.approx(
        generated.telemetry.generation_completed_monotonic_seconds
        - generated.telemetry.generation_started_monotonic_seconds
    )
    assert (
        generated.telemetry.request_started_monotonic_seconds
        <= generated.telemetry.processor_started_monotonic_seconds
        <= generated.telemetry.processor_completed_monotonic_seconds
        <= generated.telemetry.input_materialization_started_monotonic_seconds
        <= generated.telemetry.input_materialization_completed_monotonic_seconds
        <= generated.telemetry.generation_started_monotonic_seconds
        <= generated.telemetry.generation_completed_monotonic_seconds
        <= generated.telemetry.decode_started_monotonic_seconds
        <= generated.telemetry.decode_completed_monotonic_seconds
        <= generated.telemetry.request_completed_monotonic_seconds
    )
    assert generated.telemetry.processor_seconds == pytest.approx(
        generated.telemetry.processor_completed_monotonic_seconds
        - generated.telemetry.processor_started_monotonic_seconds
    )
    assert generated.telemetry.input_materialization_seconds == pytest.approx(
        generated.telemetry.input_materialization_completed_monotonic_seconds
        - generated.telemetry.input_materialization_started_monotonic_seconds
    )
    assert generated.telemetry.decode_seconds == pytest.approx(
        generated.telemetry.decode_completed_monotonic_seconds
        - generated.telemetry.decode_started_monotonic_seconds
    )
    assert generated.telemetry.total_request_seconds == pytest.approx(
        generated.telemetry.request_completed_monotonic_seconds
        - generated.telemetry.request_started_monotonic_seconds
    )
    assert processor.decoded_tokens is not None
    assert processor.decoded_tokens.moves == ["cpu"]
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
    assert len(model.generate_kwargs["stopping_criteria"]) == 1


def test_runtime_calls_fixed_frame_processor_without_codec_fallback(
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
        _fake_importer(transformers=transformers, imported=imported),
    )
    dependency_checks: list[object] = []
    runtime = MageVideoRuntime(
        model_directory=model_directory,
        codec_dependency_checker=lambda *args: dependency_checks.append(args),
    )
    frames = (object(), object(), object())

    generated = runtime.generate_fixed_frames(
        frames=frames,
        prompt="Describe only visible actions.",
        max_new_tokens=48,
    )

    assert generated.output_text == "codec answer"
    assert generated.input_video_count == 1
    assert dependency_checks == []
    assert processor.call_kwargs is not None
    assert processor.call_kwargs["video_backend"] == "frames"
    assert processor.call_kwargs["num_frames"] == 3
    assert processor.call_kwargs["max_frames"] == 3
    assert processor.call_kwargs["videos"] == [list(frames)]
    assert "codec_config" not in processor.call_kwargs
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["use_cache"] is True


def test_runtime_rejects_empty_fixed_frame_sequence(tmp_path: Path) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    runtime = MageVideoRuntime(model_directory=model_directory)

    with pytest.raises(MageVideoRuntimeError, match="at least one frame"):
        runtime.generate_fixed_frames(
            frames=(),
            prompt="Describe only visible actions.",
            max_new_tokens=48,
        )


def test_runtime_consumes_exact_admitted_provider_cache_without_running_dcvc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    video_path = tmp_path / "segments" / "cam-01.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video-segment")
    cache_root = tmp_path / "qualified-cache"
    provider_cache = cache_root / "provider-entry-from-manifest"
    provider_cache.mkdir(parents=True)
    (provider_cache / "meta.json").write_text("{}", encoding="utf-8")
    (provider_cache / "src_patch_position.npy").write_bytes(b"positions")

    calls: dict[str, object] = {"dcvc": 0, "loads": []}

    def run_dcvc(_video_url: str, _out_dir: Path, _config: object) -> object:
        calls["dcvc"] = int(calls["dcvc"]) + 1
        raise AssertionError("bound request must not invoke _run_dcvc_rt")

    def upstream_process(video_url: str, config: object) -> object:
        return run_dcvc(video_url, cache_root / "wrong-runtime-key", config)

    def load_codec_result(directory: Path) -> dict[str, object]:
        loads = calls["loads"]
        assert isinstance(loads, list)
        loads.append(directory)
        return {"out_dir": str(directory)}

    codec_module = SimpleNamespace(
        process_codec_video=upstream_process,
        _load_codec_result=load_codec_result,
        _run_dcvc_rt=run_dcvc,
    )

    class BoundCacheProcessor(_FakeProcessor):
        def __call__(self, **kwargs: object) -> dict[str, _FakeTensor]:
            videos = kwargs["videos"]
            codec_config = kwargs["codec_config"]
            assert isinstance(videos, list)
            assert isinstance(codec_config, dict)
            codec_module.process_codec_video(
                str(videos[0]),
                SimpleNamespace(engine=codec_config["engine"]),
            )
            return super().__call__(**kwargs)

    BoundCacheProcessor.__module__ = "qualified_mage.processing_mage_vl"
    processor = BoundCacheProcessor()
    model = _FakeModel()
    transformers = _FakeTransformers(processor, model)
    imported: list[str] = []
    base_importer = _fake_importer(transformers=transformers, imported=imported)

    def importer(name: str) -> object:
        if name == "qualified_mage.codec_video_processing_mage_vl":
            imported.append(name)
            return codec_module
        return base_importer(name)

    monkeypatch.setattr(mage_video_runtime, "import_module", importer)
    runtime = MageVideoRuntime(
        model_directory=model_directory,
        codec_cache_root=cache_root,
        codec_dependency_checker=lambda _config, _model_directory: None,
    )
    binding = MageVideoCodecCacheBinding(
        source_path=video_path,
        provider_cache_directory=provider_cache,
    )

    generated = runtime.generate(
        video_paths=[video_path],
        prompt="Describe the scene.",
        max_new_tokens=32,
        codec_config={
            "engine": "dcvc-rt",
            "max_pixels": 150_000,
            "preprocess_device": "cuda",
            "dcvc": {},
        },
        codec_cache_binding=binding,
    )

    assert generated.output_text == "codec answer"
    assert calls == {"dcvc": 0, "loads": [provider_cache.resolve()]}
    assert codec_module.process_codec_video is upstream_process
    assert "qualified_mage.codec_video_processing_mage_vl" in imported


def test_runtime_generation_uses_cross_process_shared_device_guard(
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
    guard_path = tmp_path / "device-guards" / "cuda-0.lock"
    runtime = MageVideoRuntime(
        model_directory=model_directory,
        shared_device_guard_file=guard_path,
    )
    assert runtime.shared_device_guard_file == guard_path.resolve()
    assert runtime.runtime_identity == MageVideoRuntimeIdentity(
        load_profile=MageVideoLoadProfile.NATIVE_BF16
    )

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_shared_device_guard,
        args=(str(guard_path), ready, release),
    )
    holder.start()
    try:
        assert ready.wait(timeout=10.0)
        with pytest.raises(MageVideoRuntimeError, match="busy with DCVC preparation"):
            runtime.generate(
                video_paths=[video_path],
                prompt="Describe the scene.",
                max_new_tokens=32,
                codec_config={
                    "engine": "hevc",
                    "max_pixels": 150_000,
                    "preprocess_device": "cpu",
                },
            )
        assert model.generate_kwargs is None
    finally:
        release.set()
        holder.join(timeout=10.0)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5.0)
    assert holder.exitcode == 0

    generated = runtime.generate(
        video_paths=[video_path],
        prompt="Describe the scene.",
        max_new_tokens=32,
        codec_config={
            "engine": "hevc",
            "max_pixels": 150_000,
            "preprocess_device": "cpu",
        },
    )
    assert generated.output_text == "codec answer"
    assert model.generate_kwargs is not None


def test_runtime_releases_generation_lane_before_cpu_decode(
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

    processor = _OverlapProcessor()
    model = _OverlapModel()
    transformers = _FakeTransformers(processor, model)
    imported: list[str] = []
    monkeypatch.setattr(
        mage_video_runtime,
        "import_module",
        _fake_importer(transformers=transformers, imported=imported),
    )
    runtime = MageVideoRuntime(model_directory=model_directory)
    generation_kwargs = {
        "video_paths": [video_path],
        "prompt": "Describe the scene.",
        "max_new_tokens": 32,
        "codec_config": {
            "engine": "hevc",
            "max_pixels": 150_000,
            "preprocess_device": "cpu",
        },
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(runtime.generate, **generation_kwargs)
        assert model.first_generation_entered.wait(timeout=5.0)
        second_future = executor.submit(runtime.generate, **generation_kwargs)
        assert processor.second_prepared.wait(timeout=5.0)
        model.release_first_generation.set()
        try:
            assert processor.first_decode_entered.wait(timeout=5.0)
            assert model.second_generation_entered.wait(timeout=5.0)
        finally:
            processor.release_first_decode.set()
            model.release_second_generation.set()
        first = first_future.result(timeout=5.0)
        second = second_future.result(timeout=5.0)

    assert first.telemetry is not None
    assert second.telemetry is not None
    assert (
        first.telemetry.generation_completed_monotonic_seconds
        <= second.telemetry.generation_started_monotonic_seconds
    )
    assert max(
        first.telemetry.decode_started_monotonic_seconds,
        second.telemetry.generation_started_monotonic_seconds,
    ) < min(
        first.telemetry.decode_completed_monotonic_seconds,
        second.telemetry.generation_completed_monotonic_seconds,
    )
    assert processor.decoded_tokens is not None
    assert processor.decoded_tokens.moves == ["cpu"]


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


def _traditional_runtime_binding(
    *,
    source_path: Path,
    provider_cache_directory: Path,
    codec_config: dict[str, object],
) -> MageVideoTraditionalCodecCacheBinding:
    assets = tuple(
        MageVideoExactCodecCacheAsset(
            relative_path=path.relative_to(provider_cache_directory).as_posix(),
            byte_count=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(provider_cache_directory.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    )
    asset_set_sha256 = semantic_sha256(
        [
            {
                "relative_path": asset.relative_path,
                "byte_count": asset.byte_count,
                "sha256": asset.sha256,
            }
            for asset in assets
        ]
    )
    return MageVideoTraditionalCodecCacheBinding(
        source_path=source_path,
        provider_cache_directory=provider_cache_directory,
        codec_engine="hevc",
        codec_config_sha256=mage_video_codec_config_sha256(codec_config),
        checkpoint_manifest_sha256="1" * 64,
        codec_policy_sha256="2" * 64,
        provider_identity_sha256="3" * 64,
        toolchain_identity_sha256="4" * 64,
        effective_config_sha256="5" * 64,
        entry_semantic_sha256="6" * 64,
        asset_set_sha256=asset_set_sha256,
        assets=assets,
    )


def test_runtime_replays_exact_traditional_cache_without_cv_preinfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    video_path = tmp_path / "segments" / "cam-01.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"h264-video-segment")
    cache_root = tmp_path / "traditional-cache"
    provider_cache = cache_root / "manifest-entry"
    provider_cache.mkdir(parents=True)
    (provider_cache / "canvas_000.jpg").write_bytes(b"canvas")
    (provider_cache / "meta.json").write_bytes(b'{"canvas_files":["canvas_000.jpg"]}')
    (provider_cache / "src_patch_position.npy").write_bytes(b"positions")
    codec_config: dict[str, object] = {
        "engine": "hevc",
        "max_pixels": 150_000,
        "target_canvas": 8,
        "preprocess_device": "cpu",
    }

    calls: dict[str, object] = {"external": 0, "loads": [], "dependency": 0}

    def run_external(_video_url: str, _out_dir: Path, _config: object) -> object:
        calls["external"] = int(calls["external"]) + 1
        raise AssertionError("bound replay must not execute cv-preinfer")

    def upstream_process(video_url: str, config: object) -> object:
        return run_external(video_url, cache_root / "wrong-key", config)

    def load_codec_result(directory: Path) -> dict[str, object]:
        loads = calls["loads"]
        assert isinstance(loads, list)
        loads.append(directory)
        return {"out_dir": str(directory)}

    codec_module = SimpleNamespace(
        process_codec_video=upstream_process,
        _load_codec_result=load_codec_result,
        _run_cv_preinfer=run_external,
    )

    class TraditionalCacheProcessor(_FakeProcessor):
        def __call__(self, **kwargs: object) -> dict[str, _FakeTensor]:
            videos = kwargs["videos"]
            native_config = kwargs["codec_config"]
            assert isinstance(videos, list)
            assert isinstance(native_config, dict)
            codec_module.process_codec_video(
                str(videos[0]),
                SimpleNamespace(engine=native_config["engine"]),
            )
            return super().__call__(**kwargs)

    TraditionalCacheProcessor.__module__ = "qualified_mage.processing_mage_vl"
    processor = TraditionalCacheProcessor()
    model = _FakeModel()
    transformers = _FakeTransformers(processor, model)
    imported: list[str] = []
    base_importer = _fake_importer(transformers=transformers, imported=imported)

    def importer(name: str) -> object:
        if name == "qualified_mage.codec_video_processing_mage_vl":
            imported.append(name)
            return codec_module
        return base_importer(name)

    def dependency_checker(_config: object, _model_directory: Path) -> None:
        calls["dependency"] = int(calls["dependency"]) + 1
        raise AssertionError("exact traditional replay must not require cv-preinfer")

    monkeypatch.setattr(mage_video_runtime, "import_module", importer)
    runtime = MageVideoRuntime(
        model_directory=model_directory,
        codec_cache_root=cache_root,
        codec_dependency_checker=dependency_checker,
    )
    binding = _traditional_runtime_binding(
        source_path=video_path,
        provider_cache_directory=provider_cache,
        codec_config=codec_config,
    )

    generated = runtime.generate(
        video_paths=[video_path],
        prompt="Describe the scene.",
        max_new_tokens=32,
        codec_config=codec_config,
        codec_cache_binding=binding,
    )

    assert generated.output_text == "codec answer"
    assert calls == {
        "external": 0,
        "loads": [provider_cache.resolve()],
        "dependency": 0,
    }
    assert codec_module.process_codec_video is upstream_process


def test_runtime_traditional_binding_fails_closed_for_config_or_asset_change(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    video_path = tmp_path / "cam-01.mp4"
    video_path.write_bytes(b"h264-video-segment")
    cache_root = tmp_path / "traditional-cache"
    provider_cache = cache_root / "manifest-entry"
    provider_cache.mkdir(parents=True)
    (provider_cache / "meta.json").write_bytes(b"{}")
    (provider_cache / "src_patch_position.npy").write_bytes(b"positions")
    codec_config: dict[str, object] = {
        "engine": "hevc",
        "max_pixels": 150_000,
        "preprocess_device": "cpu",
    }
    binding = _traditional_runtime_binding(
        source_path=video_path,
        provider_cache_directory=provider_cache,
        codec_config=codec_config,
    )
    runtime = MageVideoRuntime(
        model_directory=model_directory,
        codec_cache_root=cache_root,
        codec_dependency_checker=lambda _config, _model_directory: None,
    )

    with pytest.raises(MageVideoRuntimeError, match="configuration does not match"):
        runtime.generate(
            video_paths=[video_path],
            prompt="Describe the scene.",
            max_new_tokens=32,
            codec_config={**codec_config, "max_pixels": 200_000},
            codec_cache_binding=binding,
        )

    (provider_cache / "src_patch_position.npy").write_bytes(b"tampered")
    with pytest.raises(MageVideoRuntimeError, match="assets changed before replay"):
        runtime.generate(
            video_paths=[video_path],
            prompt="Describe the scene.",
            max_new_tokens=32,
            codec_config=codec_config,
            codec_cache_binding=binding,
        )
