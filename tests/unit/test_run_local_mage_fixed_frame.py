from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robata.benchmark.qwen_mage_common_projection import (
    COMMON_QWEN_PROMPT_VERSION,
    load_common_projection_fixture,
)
from robata.benchmark.qwen_r12_request_corpus import (
    QWEN_R12_20260806_EXPECTED,
    load_qwen_request_corpus,
)

ROOT = Path(__file__).resolve().parents[2]
DB = Path(
    r"D:\tmp\robata-qwen-run-20260806\canonical-qwen-full-r12-20260806"
    r"\inference-evidence.sqlite3"
)
ARTIFACTS = ROOT / ".tmp" / "temporal-ab-131k-control-r3" / "stream-artifacts"
LOCAL_EVIDENCE = DB.is_file() and ARTIFACTS.is_dir()


def _module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "run_local_mage_fixed_frame_test_module",
        ROOT / "scripts" / "run_local_mage_fixed_frame.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not LOCAL_EVIDENCE, reason="frozen local fixture is unavailable")
def test_prompt_modes_are_explicit_and_identity_bound() -> None:
    module = _module()
    corpus = load_qwen_request_corpus(DB, expected=QWEN_R12_20260806_EXPECTED)
    fixture = load_common_projection_fixture(corpus=corpus, mage_stream_artifact_root=ARTIFACTS)
    case = fixture.cases[0]

    native_version, native_prompt = module._prompt_for_case(case, "native_mage")
    common_version, common_prompt = module._prompt_for_case(case, "common_qwen")

    assert native_version == "mage-unified-observation-prompt-v6"
    assert native_prompt == case.binding.endpoint_request.decoder.prompt
    assert common_version == COMMON_QWEN_PROMPT_VERSION
    assert common_prompt != native_prompt

    native_identity = module.build_fixed_frame_input_identity(
        case=case,
        checkpoint_manifest_sha256="a" * 64,
        model_revision="local-test",
        load_profile="bitsandbytes_4bit_nf4_v1",
        max_new_tokens=256,
        prompt=native_prompt,
        prompt_version=native_version,
    )
    common_identity = module.build_fixed_frame_input_identity(
        case=case,
        checkpoint_manifest_sha256="a" * 64,
        model_revision="local-test",
        load_profile="bitsandbytes_4bit_nf4_v1",
        max_new_tokens=256,
        prompt=common_prompt,
        prompt_version=common_version,
    )
    assert native_identity.semantic_sha256 != common_identity.semantic_sha256


def test_capacity_rejects_nonpositive_recurring_time() -> None:
    module = _module()
    with pytest.raises(module.MageFixedFrameQualificationError):
        module._capacity(media_seconds=40.0, recurring_wall_seconds=0.0)
