from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "benchmark_local_mage_nf4_attention.py"
)


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "benchmark_local_mage_nf4_attention_test", SCRIPT_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _digest(seed: int) -> str:
    return f"{seed:064x}"


def test_attention_override_is_process_local_and_restored(monkeypatch) -> None:
    module = _module()

    def original(**kwargs):
        return {"sentinel": kwargs["sentinel"]}

    monkeypatch.setattr(module.mage_video_runtime, "_build_model_load_kwargs", original)

    with module._temporary_attention_backend("sdpa"):
        patched = module.mage_video_runtime._build_model_load_kwargs
        assert patched(sentinel="kept") == {
            "sentinel": "kept",
            "attn_implementation": "sdpa",
        }

    assert module.mage_video_runtime._build_model_load_kwargs is original


def test_main_writes_hash_only_non_authoritative_report(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    video = tmp_path / "segment.mp4"
    output = tmp_path / "sdpa-report.json"
    prompt_file = tmp_path / "prompt.txt"
    model_dir = tmp_path / "model"
    cache_root = tmp_path / "cache"
    model_dir.mkdir()
    cache_root.mkdir()
    video.write_bytes(b"video")
    prompt_file.write_text("classify this segment", encoding="utf-8")

    checkpoint = SimpleNamespace(manifest_sha256=_digest(1))
    cache_manifest = SimpleNamespace(
        qualified_cache_root=str(cache_root),
        manifest_semantic_sha256=_digest(2),
        namespace_identity=_digest(3),
        codec_policy_sha256=_digest(4),
    )
    video_inputs = (
        {
            "ordinal": 0,
            "source_path": str(video.resolve()),
            "source_content_sha256": _digest(5),
            "source_byte_count": 5,
            "logical_cache_identity": _digest(6),
        },
    )
    monkeypatch.setattr(
        module,
        "_validate_inputs",
        lambda _arguments: (
            checkpoint,
            cache_manifest,
            (video.resolve(),),
            "classify this segment",
            video_inputs,
        ),
    )

    class FakeGpuReport:
        def to_payload(self) -> dict[str, object]:
            return {
                "format_version": "robata-nvidia-smi-gpu-telemetry-v1",
                "measurement_status": "MEASURED",
                "started_wall_clock_unix_ns": 1_786_249_163_241_504_500,
                "stopped_wall_clock_unix_ns": 1_786_249_163_341_504_500,
                "monotonic_duration_ns": 100_000_000,
                "summary": [
                    {
                        "gpu_index": 0,
                        "gpu_name": "fake",
                        "memory_used_fraction_max": 0.64,
                    }
                ],
                "samples": [
                    {
                        "wall_clock_unix_ns": 1_786_249_163_241_504_500,
                        "monotonic_offset_ns": 0,
                        "query_duration_ns": 1_000_000,
                    }
                ],
                "errors": [],
            }

    class FakeSampler:
        def __init__(self, *, interval_seconds: float) -> None:
            assert interval_seconds == 0.25
            self.started = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> FakeGpuReport:
            assert self.started
            return FakeGpuReport()

    monkeypatch.setattr(module, "NvidiaSmiGpuSampler", FakeSampler)

    closed = []

    class FakeRuntime:
        def __init__(self, **kwargs) -> None:
            assert kwargs["codec_cache_root"] == cache_root.resolve()
            assert kwargs["load_profile"].value == module.NF4_LOAD_PROFILE
            self.runtime_identity = SimpleNamespace(
                identity_version="mage-video-runtime-v1",
                load_profile=SimpleNamespace(value=module.NF4_LOAD_PROFILE),
            )
            config = SimpleNamespace(
                _attn_implementation="sdpa",
                text_config=SimpleNamespace(_attn_implementation="sdpa"),
                vision_config=SimpleNamespace(_attn_implementation="sdpa"),
            )
            self._model = SimpleNamespace(config=config)
            self.calls = 0

        def load(self):
            return SimpleNamespace(load_seconds=1.25)

        def generate(self, **_kwargs):
            self.calls += 1
            is_warmup = self.calls == 1
            return SimpleNamespace(
                output_text="discarded-warm-output" if is_warmup else "timed-secret-output",
                prompt_tokens=20,
                output_tokens=8 if is_warmup else 12,
                generation_seconds=0.5 if is_warmup else 1.0,
                telemetry=SimpleNamespace(
                    total_request_seconds=1.1,
                    time_to_first_token_seconds=0.2,
                    output_tokens_per_second=12.0,
                ),
            )

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(module.mage_video_runtime, "MageVideoRuntime", FakeRuntime)

    def original_builder(**kwargs):
        return dict(kwargs)

    monkeypatch.setattr(
        module.mage_video_runtime,
        "_build_model_load_kwargs",
        original_builder,
    )

    exit_code = module.main(
        [
            "--model-dir",
            str(model_dir),
            "--checkpoint-manifest-path",
            str(tmp_path / "checkpoint.json"),
            "--checkpoint-manifest-sha256",
            _digest(1),
            "--codec-cache-manifest",
            str(tmp_path / "cache.json"),
            "--video",
            str(video),
            "--prompt-file",
            str(prompt_file),
            "--attention",
            "sdpa",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert closed == [True]
    assert module.mage_video_runtime._build_model_load_kwargs is original_builder
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["authority"] == "NON_AUTHORITATIVE_EXPERIMENT"
    assert payload["production_eligible"] is False
    assert payload["attention_resolution_verified"] is True
    assert payload["runtime_identity"]["attention_backend_bound"] is False
    assert payload["experimental_scope"]["production_launcher_modified"] is False
    assert payload["input_videos"] == list(video_inputs)
    assert payload["gpu_telemetry"]["started_wall_clock_unix_ns"] == ("1786249163241504500")
    assert payload["gpu_telemetry"]["samples"][0]["wall_clock_unix_ns"] == ("1786249163241504500")
    assert payload["results"][0]["output_text_sha256"] == _digest_of("timed-secret-output")
    serialized = output.read_text(encoding="utf-8")
    assert "timed-secret-output" not in serialized
    assert "discarded-warm-output" not in serialized


def _digest_of(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
