from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "mage-fixed-frame-native-prompt-qualification-2026-08-10.json"
)
EXPECTED_EXACT_SHA256 = "0a7433b59541a4fb01db5934826a646ec0f3bcbe737c2f575203254182c96320"


def test_fixed_frame_native_prompt_report_is_fail_closed_and_complete() -> None:
    raw = REPORT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_EXACT_SHA256
    payload = json.loads(raw)

    assert payload["report_version"] == "mage-fixed-frame-control-qualification-v2"
    assert payload["authority"] == "LOCAL_NONPRODUCTION_ONLY"
    assert payload["production_eligible"] is False
    assert payload["admission"]["state"] == "BASELINE_ONLY_NOT_ADMITTED"

    configuration = payload["configuration"]
    assert configuration["processor_backend"] == "frames"
    assert configuration["prompt_mode"] == "native_mage"
    assert configuration["prompt_equivalence"] == "EXACT_MAGE_BINDING_PROMPT_BYTES"
    assert configuration["codec_preparation"] == "DISABLED"
    assert configuration["stream_memory"] == "DISABLED"
    assert configuration["frame_count_per_segment"] == [6, 6, 6, 6, 6]
    assert configuration["effective_max_new_tokens_per_segment"] == [256] * 5

    execution = payload["execution"]
    assert execution["status"] == "SUCCEEDED"
    assert execution["strict_projection_count"] == 5
    assert execution["strict_projection_failure_count"] == 0
    assert len(execution["segments"]) == 5
    assert all(row["projection_status"] == "SUCCEEDED" for row in execution["segments"])
    assert execution["capacity"]["quality_qualified"] is False
    assert execution["capacity"]["decision_eligible"] is False
    assert execution["capacity"]["production_qualification"] == "NOT_CLAIMED"

    semantic = payload["semantic_evidence"]
    assert semantic["comparison_role_mapping"] == {
        "mage_fields": "fixed_frame_candidate",
        "qwen_fields": "frozen_native_mage_reference",
    }
    assert semantic["authority"] == "UNLABELED_MODEL_AGREEMENT_ONLY"
    assert semantic["is_ground_truth_accuracy"] is False
    assert semantic["quality_qualified"] is False
    assert semantic["decision_eligible"] is False
