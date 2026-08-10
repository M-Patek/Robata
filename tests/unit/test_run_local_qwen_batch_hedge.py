from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from robata.benchmark.qwen_r12_request_corpus import QwenSelectedImage
from robata.inference.models import VisionTask

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_local_qwen_batch_hedge.py"
R12_DATABASE = Path(
    r"D:\tmp\robata-qwen-run-20260806\canonical-qwen-full-r12-20260806\inference-evidence.sqlite3"
)
QWEN_MODEL = Path(r"D:\HuggingFace\Qwen3-VL-4B-Instruct")


def _module() -> ModuleType:
    name = f"run_local_qwen_batch_hedge_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepared(task: VisionTask, ordinal: int) -> dict[str, object]:
    case = SimpleNamespace(request=SimpleNamespace(task=task), ordinal=ordinal)
    return {"case": case, "claim_group_count": 1}


def test_chunks_never_cross_task_compatibility_boundary() -> None:
    module = _module()
    prepared = tuple(
        [
            *(_prepared(VisionTask.QA_COARSE, index) for index in range(5)),
            *(_prepared(VisionTask.QA_DENSE, index + 5) for index in range(3)),
        ]
    )

    chunks = module._chunks_without_crossing_task(
        prepared,
        batch_size=4,
        packing_policy="contiguous-task-v1",
        multi_claim_policy="batch-v1",
    )

    assert [len(chunk) for chunk in chunks] == [4, 1, 3]
    assert [chunk[0]["case"].request.task for chunk in chunks] == [
        VisionTask.QA_COARSE,
        VisionTask.QA_COARSE,
        VisionTask.QA_DENSE,
    ]


def test_claim_group_packing_separates_output_shapes_and_restores_case_identity() -> None:
    module = _module()
    prepared = (
        {**_prepared(VisionTask.QA_COARSE, 0), "claim_group_count": 1},
        {**_prepared(VisionTask.QA_COARSE, 1), "claim_group_count": 2},
        {**_prepared(VisionTask.QA_COARSE, 2), "claim_group_count": 1},
        {**_prepared(VisionTask.QA_COARSE, 3), "claim_group_count": 2},
    )

    chunks = module._chunks_without_crossing_task(
        prepared,
        batch_size=2,
        packing_policy="task-claim-group-v1",
        multi_claim_policy="batch-v1",
    )

    assert [[item["case"].ordinal for item in chunk] for chunk in chunks] == [[0, 2], [1, 3]]
    assert [chunk[0]["claim_group_count"] for chunk in chunks] == [1, 2]


def test_multi_claim_serial_policy_keeps_single_claim_batches_native() -> None:
    module = _module()
    prepared = (
        {**_prepared(VisionTask.QA_COARSE, 0), "claim_group_count": 1},
        {**_prepared(VisionTask.QA_COARSE, 1), "claim_group_count": 1},
        {**_prepared(VisionTask.QA_COARSE, 2), "claim_group_count": 2},
        {**_prepared(VisionTask.QA_COARSE, 3), "claim_group_count": 2},
    )

    chunks = module._chunks_without_crossing_task(
        prepared,
        batch_size=4,
        packing_policy="task-claim-group-v1",
        multi_claim_policy="serial-v1",
    )

    assert [[item["case"].ordinal for item in chunk] for chunk in chunks] == [
        [0, 1],
        [2],
        [3],
    ]


def test_quality_gate_requires_parse_and_normalized_parity_without_exhaustion() -> None:
    module = _module()
    execution = {
        "cases": [
            {
                "parse_error": None,
                "raw_exact_match": False,
                "normalized_exact_match": True,
                "output_exhausted": False,
            },
            {
                "parse_error": None,
                "raw_exact_match": True,
                "normalized_exact_match": True,
                "output_exhausted": False,
            },
        ]
    }

    quality = module._quality_projection(execution)

    assert quality["quality_gate_pass"] is True
    assert quality["parse_valid_count"] == 2
    assert quality["raw_exact_match_count"] == 1
    assert quality["normalized_exact_match_count"] == 2

    execution["cases"][1]["output_exhausted"] = True
    assert module._quality_projection(execution)["quality_gate_pass"] is False


def test_image_byte_cache_is_exact_digest_keyed_and_counts_reuse(tmp_path: Path) -> None:
    module = _module()
    payload = b"\x89PNG\r\n\x1a\nfixture"
    path = tmp_path / "image.png"
    path.write_bytes(payload)
    sha = module.exact_bytes_sha256(payload)
    image = QwenSelectedImage(
        selected_ordinal=0,
        provider_item_ordinal=0,
        camera_id="cam_01",
        uri=path.as_uri(),
        path=path,
        sha256=sha,
        byte_count=len(payload),
        media_type="image/png",
        encoding="png",
        width=1,
        height=1,
    )
    cases = (
        SimpleNamespace(selected_images=(image,)),
        SimpleNamespace(selected_images=(image,)),
    )

    cache, metrics = module._load_image_bytes(cases)

    assert cache == {sha: payload}
    assert metrics == {
        "references": 2,
        "unique_images": 1,
        "cache_hits": 1,
        "unique_bytes": len(payload),
    }

    path.write_bytes(payload + b"drift")
    with pytest.raises(module.QwenBatchHedgeBenchmarkError, match="changed"):
        module._load_image_bytes(cases)


def test_capacity_labels_six_camera_qa_scope_without_production_claim() -> None:
    module = _module()
    images = tuple(SimpleNamespace(camera_id=f"cam_0{index}") for index in range(1, 7))
    cases = (
        SimpleNamespace(
            intent=SimpleNamespace(start_ns=0, end_ns=40_000_000_000),
            selected_images=images,
        ),
    )

    capacity = module._capacity_projection(
        cases=cases, execution_wall_seconds=20.0, complete_corpus=True
    )

    assert capacity["recording_real_time_multiple"] == 2.0
    assert capacity["camera_real_time_multiple"] == 12.0
    assert capacity["local_equivalent_lanes_for_25x_camera_hours"] == 3
    assert capacity["production_eligible"] is False
    assert capacity["scope"] == "QWEN_R12_QA_ONLY_NOT_FULL_PIPELINE"


@pytest.mark.skipif(
    not R12_DATABASE.is_file() or not QWEN_MODEL.is_dir(),
    reason="real frozen r12 corpus is not present",
)
def test_verify_only_reconstructs_real_r12_without_loading_model(tmp_path: Path) -> None:
    module = _module()
    output_dir = tmp_path / "verify"

    exit_code = module.main(
        [
            "--model-dir",
            str(QWEN_MODEL),
            "--corpus-db",
            str(R12_DATABASE),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
            "--verify-only",
        ]
    )

    assert exit_code == 0
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "CORPUS_VERIFIED"
    assert report["corpus"]["selected_case_count"] == 51
    assert (
        report["corpus"]["semantic_sha256"]
        == "d4bd44f5e573b2abc13000cf9421134ac0e8d00fe92890fc6a7fa265c84425ed"
    )
    assert report["preparation"]["image_cache"]["cache_hits"] == 30
    assert report["load"] is None
    assert report["production_eligible"] is False
