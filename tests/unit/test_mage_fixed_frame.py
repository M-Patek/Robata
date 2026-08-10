from __future__ import annotations

from pathlib import Path

import pytest

import robata.benchmark.mage_fixed_frame as mage_fixed_frame
from robata.benchmark.mage_fixed_frame import (
    MAGE_FIXED_FRAME_MODEL_FAMILY,
    MAGE_FIXED_FRAME_POLICY_VERSION,
    build_fixed_frame_input_identity,
    close_fixed_frame_images,
    load_verified_fixed_frame_images,
    project_mage_fixed_frame_output,
)
from robata.benchmark.qwen_mage_common_projection import load_common_projection_fixture
from robata.benchmark.qwen_r12_request_corpus import (
    QWEN_R12_20260806_EXPECTED,
    load_qwen_request_corpus,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
R12_DATABASE = Path(
    r"D:\tmp\robata-qwen-run-20260806\canonical-qwen-full-r12-20260806"
    r"\inference-evidence.sqlite3"
)
MAGE_ARTIFACT_ROOT = REPOSITORY_ROOT / ".tmp" / "temporal-ab-131k-control-r3" / "stream-artifacts"
LOCAL_EVIDENCE = R12_DATABASE.is_file() and MAGE_ARTIFACT_ROOT.is_dir()


def _fixture():  # type: ignore[no-untyped-def]
    corpus = load_qwen_request_corpus(R12_DATABASE, expected=QWEN_R12_20260806_EXPECTED)
    return load_common_projection_fixture(
        corpus=corpus,
        mage_stream_artifact_root=MAGE_ARTIFACT_ROOT,
    )


@pytest.mark.skipif(not LOCAL_EVIDENCE, reason="frozen local r12/Mage evidence is unavailable")
def test_fixed_frame_identity_binds_exact_frames_prompt_and_runtime_profile() -> None:
    case = _fixture().cases[0]

    identity = build_fixed_frame_input_identity(
        case=case,
        checkpoint_manifest_sha256="7" * 64,
        model_revision="local-test",
        load_profile="bitsandbytes_4bit_nf4_v1",
        max_new_tokens=160,
    )
    replay = build_fixed_frame_input_identity(
        case=case,
        checkpoint_manifest_sha256="7" * 64,
        model_revision="local-test",
        load_profile="bitsandbytes_4bit_nf4_v1",
        max_new_tokens=160,
    )
    changed = build_fixed_frame_input_identity(
        case=case,
        checkpoint_manifest_sha256="7" * 64,
        model_revision="local-test",
        load_profile="bitsandbytes_4bit_nf4_v1",
        max_new_tokens=161,
    )

    assert identity.policy_version == MAGE_FIXED_FRAME_POLICY_VERSION
    assert identity.semantic_sha256 == replay.semantic_sha256
    assert identity.semantic_sha256 != changed.semantic_sha256
    assert len(identity.frame_references) == 6
    assert [item["sha256"] for item in identity.frame_references] == [
        frame.sha256 for frame in case.selected_frames
    ]


@pytest.mark.skipif(not LOCAL_EVIDENCE, reason="frozen local r12/Mage evidence is unavailable")
def test_fixed_frame_loader_verifies_exact_local_frame_bytes_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture().cases[0]
    expected_sizes = iter((frame.width, frame.height) for frame in case.selected_frames)

    class FakeImage:
        mode = "RGB"

        def __init__(self, size: tuple[int, int]) -> None:
            self.size = size
            self.closed = False

        def convert(self, mode: str) -> FakeImage:
            assert mode == "RGB"
            return FakeImage(self.size)

        def load(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class FakeImageModule:
        @staticmethod
        def open(_payload: object) -> FakeImage:
            return FakeImage(next(expected_sizes))

    monkeypatch.setattr(
        mage_fixed_frame,
        "import_module",
        lambda name: FakeImageModule if name == "PIL.Image" else None,
    )

    images = load_verified_fixed_frame_images(case)
    try:
        assert len(images) == 6
        assert [tuple(image.size) for image in images] == [
            (frame.width, frame.height) for frame in case.selected_frames
        ]
        assert all(image.mode == "RGB" for image in images)
    finally:
        close_fixed_frame_images(images)
    assert all(image.closed for image in images)


@pytest.mark.skipif(not LOCAL_EVIDENCE, reason="frozen local r12/Mage evidence is unavailable")
def test_fixed_frame_projection_is_strict_and_keeps_nonproduction_model_family() -> None:
    case = _fixture().cases[0]
    identity = build_fixed_frame_input_identity(
        case=case,
        checkpoint_manifest_sha256="7" * 64,
        model_revision="local-test",
        load_profile="bitsandbytes_4bit_nf4_v1",
        max_new_tokens=160,
    )

    projection = project_mage_fixed_frame_output(
        case=case,
        input_identity=identity,
        output_text=case.binding.endpoint_response.output_text,
        created_at="2026-08-10T00:00:00Z",
    )

    assert projection.observation.model_family == MAGE_FIXED_FRAME_MODEL_FAMILY
    assert projection.observation.model_revision == "local-test"
    assert projection.observation.context == case.context
    assert projection.inference_artifact_exact_sha256 == (
        projection.observation.inference_artifact_exact_sha256
    )
