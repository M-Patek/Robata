from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType

import pytest

from robata.benchmark.qwen_mage_common_projection import (
    CommonFrameReference,
    CommonProjectionError,
    action_token_f1,
    build_qwen_common_prompt,
    interval_iou,
    load_common_projection_fixture,
    project_qwen_compact_output,
    select_evenly_spaced_frames,
)
from robata.benchmark.qwen_r12_request_corpus import (
    QWEN_R12_20260806_EXPECTED,
    load_qwen_request_corpus,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_qwen_mage_common_projection.py"
R12_DATABASE = Path(
    r"D:\tmp\robata-qwen-run-20260806\canonical-qwen-full-r12-20260806"
    r"\inference-evidence.sqlite3"
)
MAGE_ARTIFACT_ROOT = REPOSITORY_ROOT / ".tmp" / "temporal-ab-131k-control-r3" / "stream-artifacts"


def _module() -> ModuleType:
    name = f"run_qwen_mage_common_projection_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _frame(
    ordinal: int, timestamp_ns: int, *, digest_suffix: int | None = None
) -> CommonFrameReference:
    suffix = ordinal if digest_suffix is None else digest_suffix
    return CommonFrameReference(
        ordinal=ordinal,
        aligned_timestamp_ns=timestamp_ns,
        path=Path(f"frame-{ordinal}.png"),
        sha256=f"{suffix:064x}",
        byte_count=10,
        width=320,
        height=260,
    )


def test_even_frame_selection_is_deterministic_and_spans_interval() -> None:
    frames = tuple(_frame(index, index * 1_000_000_000) for index in range(8))

    selected = select_evenly_spaced_frames(
        frames,
        start_ns=0,
        end_ns=8_000_000_000,
        count=6,
    )

    assert [item.ordinal for item in selected] == [0, 1, 3, 4, 6, 7]


def test_even_frame_selection_rejects_duplicate_content() -> None:
    frames = tuple(_frame(index, index * 1_000_000_000, digest_suffix=0) for index in range(8))

    with pytest.raises(CommonProjectionError, match="duplicate content"):
        select_evenly_spaced_frames(
            frames,
            start_ns=0,
            end_ns=8_000_000_000,
            count=6,
        )


def test_action_and_interval_agreement_are_explicit_not_ground_truth() -> None:
    assert action_token_f1("a person folds a green shirt", "folding green shirt") == pytest.approx(
        2 / 3
    )
    assert interval_iou(0, 10, 5, 15) == pytest.approx(1 / 3)
    assert interval_iou(0, 1, 2, 3) == 0.0


def test_runner_builds_serial_and_batch4_physical_chunks() -> None:
    module = _module()

    assert module._chunks(5, mode="serial") == ((0,), (1,), (2,), (3,), (4,))
    assert module._chunks(5, mode="batch4") == ((0, 1, 2, 3), (4,))


def test_raw_output_diagnostic_does_not_repair_top_level_array() -> None:
    module = _module()

    diagnostic = module._raw_output_diagnostic('[{"action":"fold shirt"}]')

    assert diagnostic == {
        "valid_json": True,
        "root_type": "list",
        "array_length": 1,
        "member_key_sets": [["action"]],
    }


def test_capacity_uses_recurring_wall_and_ceiling_lane_count() -> None:
    module = _module()

    projection = module._capacity_projection(
        duration_seconds=40.0,
        recurring_wall_seconds=20.0,
    )

    assert projection["camera_realtime_factor"] == 2.0
    assert projection["local_equivalent_lanes_for_25x"] == 13
    assert projection["production_qualification"] == "NOT_CLAIMED"


def test_unlabeled_common_agreement_is_not_decision_eligible() -> None:
    module = _module()

    gate = module._semantic_quality_gate(
        {
            "authority": "UNLABELED_MODEL_AGREEMENT_ONLY",
            "is_ground_truth_accuracy": False,
        }
    )

    assert gate == {
        "authority": "UNLABELED_MODEL_AGREEMENT_ONLY",
        "is_ground_truth_accuracy": False,
        "quality_qualified": False,
        "decision_eligible": False,
        "hold_reason": "HOLD_UNLABELED_MODEL_AGREEMENT_ONLY_V1",
    }


def test_missing_common_projection_is_rejected_before_decision() -> None:
    module = _module()

    gate = module._semantic_quality_gate(None)

    assert gate == {
        "authority": None,
        "is_ground_truth_accuracy": False,
        "quality_qualified": False,
        "decision_eligible": False,
        "hold_reason": "REJECT_CANDIDATE_OUTPUT_BEFORE_DOWNSTREAM_V1",
    }


def test_downstream_action_matching_records_unmatched_candidate() -> None:
    module = _module()
    mage = [{"action": "fold green shirt", "start_ns": "0", "end_ns": "10"}]
    qwen = [
        {"action": "folding green shirt", "start_ns": "0", "end_ns": "10"},
        {"action": "pick up bag", "start_ns": "20", "end_ns": "30"},
    ]

    agreement = module._compare_projected_actions(mage, qwen)

    assert agreement["mage_count"] == 1
    assert agreement["qwen_count"] == 2
    assert agreement["unmatched_qwen_count"] == 1
    assert agreement["mean_temporal_iou"] == 1.0
    assert agreement["unmatched_qwen_actions"] == ["pick up bag"]


@pytest.mark.skipif(
    not (R12_DATABASE.is_file() and MAGE_ARTIFACT_ROOT.is_dir()),
    reason="frozen local r12/Mage evidence is not available",
)
def test_frozen_common_fixture_and_strict_compact_projection() -> None:
    corpus = load_qwen_request_corpus(
        R12_DATABASE,
        expected=QWEN_R12_20260806_EXPECTED,
    )
    fixture = load_common_projection_fixture(
        corpus=corpus,
        mage_stream_artifact_root=MAGE_ARTIFACT_ROOT,
    )

    assert fixture.semantic_sha256 == (
        "2db13e7ec98199e89df65646e6592ed2aad0bfa193396b1b250528dc4da7c8e1"
    )
    assert fixture.duration_seconds == 40.0
    assert [len(case.selected_frames) for case in fixture.cases] == [6, 6, 6, 6, 6]
    prompt = build_qwen_common_prompt(fixture.cases[0])
    assert '"json_root":"object"' in prompt
    assert '"exact_root_keys":["selected_camera_qa","observations"]' in prompt
    assert '"protocol":"qwen-mage-common-projection-prompt-v2"' in prompt
    assert "Do not output schema metadata keys" in prompt
    projection = project_qwen_compact_output(
        case=fixture.cases[0],
        checkpoint_manifest_sha256=(
            "1f7293b2629473f0240c8675025e1402da4306f05cc9026adf4c801f20f99f10"
        ),
        output_text=fixture.cases[0].binding.endpoint_response.output_text,
        created_at="2026-08-09T00:00:00Z",
    )
    assert len(projection.observation.observations) == 2
    assert projection.observation.context == fixture.cases[0].context
