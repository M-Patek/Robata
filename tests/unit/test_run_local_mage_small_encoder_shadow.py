from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_local_mage_small_encoder_shadow as runner


def _segment_context(*, codec_sha256: str) -> dict[str, object]:
    return {
        "cameras": {"cam_01": {"codec_stream_exact_sha256": codec_sha256}},
        "ordered_segments": [{"segment_semantic_sha256": "1" * 64}],
    }


def test_segment_model_validates_actual_media_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "segment.mp4"
    media.write_bytes(b"media")
    digest = hashlib.sha256(b"media").hexdigest()
    monkeypatch.setattr(runner, "_typed_context", lambda value: value)

    segment, typed = runner._segment_model(
        _segment_context(codec_sha256=digest),
        {"durable_path": str(media)},
    )

    assert typed["cameras"]["cam_01"]["codec_stream_exact_sha256"] == digest
    assert segment.content_sha256 == digest
    assert segment.codec_stream_exact_sha256 == digest
    assert segment.byte_count == 5


def test_segment_model_rejects_media_replaced_after_context_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "segment.mp4"
    media.write_bytes(b"tampered")
    monkeypatch.setattr(runner, "_typed_context", lambda value: value)

    with pytest.raises(ValueError, match="content_sha256 must equal codec_stream_exact_sha256"):
        runner._segment_model(
            _segment_context(codec_sha256=hashlib.sha256(b"original").hexdigest()),
            {"durable_path": str(media)},
        )


def test_seal_report_is_canonical_and_single_use() -> None:
    report = {
        "report_version": runner.MAGE_SMALL_ENCODER_REPORT_VERSION,
        "authority": "MAGE_NATIVE",
    }
    sealed = runner._seal_report(report)
    assert "report_sha256" not in report
    without_hash = dict(sealed)
    embedded = without_hash.pop("report_sha256")
    assert embedded == runner._canonical_sha256(without_hash)
    with pytest.raises(ValueError, match="already sealed"):
        runner._seal_report(sealed)


def test_prepare_processor_inputs_binds_declared_max_pixels() -> None:
    captured: dict[str, object] = {}

    class FakeProcessor:
        @staticmethod
        def apply_chat_template(
            messages: object, *, tokenize: bool, add_generation_prompt: bool
        ) -> str:
            assert messages
            assert tokenize is False
            assert add_generation_prompt is True
            return "prompt"

        def __call__(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return dict(kwargs)

    result = runner._prepare_processor_inputs(
        processor=FakeProcessor(),
        segment=SimpleNamespace(durable_path="segment.mp4"),
        prompt="observe",
        codec_config={"max_pixels": 32768},
        max_pixels=32768,
    )

    assert captured["max_pixels"] == 32768
    assert captured["codec_config"] == {"max_pixels": 32768}
    assert result["videos"] == ["segment.mp4"]


def test_cuda_peak_helpers_are_path_scoped() -> None:
    calls: list[object] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            calls.append("sync")

        @staticmethod
        def reset_peak_memory_stats(device: object) -> None:
            calls.append(("reset", device))

        @staticmethod
        def max_memory_allocated(device: object) -> int:
            calls.append(("read", device))
            return 64 * 1024 * 1024

    torch_module = SimpleNamespace(cuda=FakeCuda())
    runner._reset_cuda_peak_memory(torch_module, "cuda:0")
    peak = runner._cuda_peak_memory_mib(torch_module, "cuda:0")

    assert peak == 64.0
    assert calls == ["sync", ("reset", "cuda:0"), "sync", ("read", "cuda:0")]
