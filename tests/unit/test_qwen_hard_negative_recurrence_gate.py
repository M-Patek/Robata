from __future__ import annotations

from robata.benchmark.qwen_hard_negative_recurrence_gate import audit


def _doc(rows, *, independent=True):
    return {
        "independence_certified": independent,
        "cases": rows,
    }


def _row(case_id, group, cues, *, grounded=True):
    return {
        "case_id": case_id,
        "source_group": group,
        "negative_cues": list(cues),
        "direct_hand_object_visible": True,
        "control_or_state_visible": grounded,
        "adjacent_activity_visible": True,
        "cue_confidence": 0.9,
    }


def test_gate_holds_on_partial_nonindependent_evidence() -> None:
    a = _doc([_row("a1", "g1", ["adjacent_substitution"])], independent=True)
    b = _doc(
        [{"case_id": "a1", "cue": "adjacent_substitution", "resolved": True}], independent=False
    )
    result = audit(reviewer_a=a, reviewer_b=b)
    assert result["decision"] == "HOLD_NO_VALIDATED_CUE"
    assert result["gates"]["independence_certified"] is False
    assert result["reviewer_independence"]["a"] is True
    assert result["reviewer_independence"]["b"] is False
    assert result["gates"]["same_cue_two_groups_each_two_rows"] is False


def test_gate_admits_only_recurrent_cross_group_consensus() -> None:
    rows_a = []
    rows_b = []
    ordinal = 0
    for group in ("g1", "g2"):
        for _ in range(2):
            ordinal += 1
            cid = f"c{ordinal}"
            rows_a.append(_row(cid, group, ["adjacent_substitution"], grounded=True))
            rows_b.append(
                {
                    "case_id": cid,
                    "video_group": group,
                    "cue": "adjacent_substitution",
                    "resolved": True,
                    "grounded_control": True,
                }
            )
    a = _doc(rows_a, independent=True)
    b = _doc(rows_b, independent=True)
    result = audit(reviewer_a=a, reviewer_b=b)
    assert result["decision"] == "ADMIT_CONTEXT_BOTTLENECK_DIAGNOSTIC"
    assert result["gates"]["same_cue_two_groups_each_two_rows"] is True
    assert result["agreement"]["agreement_rate"] == 1.0


def test_unresolved_rows_do_not_count_as_recurrence() -> None:
    a = _doc([_row("c1", "g1", ["unsupported_state_or_direction"])], independent=True)
    b = _doc(
        [
            {
                "case_id": "c1",
                "video_group": "g1",
                "cue": "unsupported_state_or_direction",
                "resolved": False,
                "grounded_control": False,
            }
        ],
        independent=True,
    )
    result = audit(reviewer_a=a, reviewer_b=b)
    assert result["recurrence"]["unsupported_state_or_direction"]["consensus_resolved_rows"] == 0
    assert result["decision"] == "HOLD_NO_VALIDATED_CUE"
