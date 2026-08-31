"""CPU-only recurrence gate for blind-review Qwen hard-negative diagnosis.

This module is benchmark preparation only. It never loads media or model
weights and never reads official labels. It accepts two reviewer documents
using either the diagnostic-pack or compact review schema, then admits a cue
only when independent reviewers agree that it recurs across source groups.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CUES = (
    "adjacent_substitution",
    "generic_scene_overwrite",
    "unsupported_state_or_direction",
    "invented_attribute",
    "control_object_alias",
    "no_surface_evidence",
    "none",
)
TARGET_CUES = ("adjacent_substitution", "unsupported_state_or_direction")
LAYER = {
    "no_surface_evidence": "L1",
    "unsupported_state_or_direction": "L3",
    "adjacent_substitution": "L4",
    "generic_scene_overwrite": "L4",
    "invented_attribute": "L2",
    "control_object_alias": "L2",
    "none": "none",
}


def _rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = document.get("cases") or document.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError("review document cases/rows must be a list")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue

        case_id = str(
            row.get("case_id")
            or row.get("uid")
            or row.get("review_order")
            or row.get("surface_ref")
            or ""
        ).strip()
        if not case_id:
            continue

        cues: list[str] = []
        raw_cues = row.get("negative_cues")
        if isinstance(raw_cues, list):
            cues.extend(str(value) for value in raw_cues)
        for key in ("primary_cue", "cue"):
            value = row.get(key)
            if value is not None:
                cues.append(str(value))
        secondary_cues = row.get("secondary_cues")
        if isinstance(secondary_cues, list):
            cues.extend(str(value) for value in secondary_cues)
        cue_values = sorted({cue for cue in cues if cue in CUES})

        # Unknown is unresolved rather than false. Compact H8 rows may carry
        # an explicit ``resolved`` field; diagnostic-pack rows carry the four
        # component review facts instead.
        resolved = row.get("resolved")
        if not isinstance(resolved, bool):
            required = (
                "direct_hand_object_visible",
                "control_or_state_visible",
                "adjacent_activity_visible",
                "cue_confidence",
            )
            resolved = all(key in row and row.get(key) is not None for key in required)
            if not resolved and row.get("interaction_visibility") in {"clear", "partial"}:
                resolved = True

        grounded = row.get("grounded_control")
        if not isinstance(grounded, bool):
            control_visible = row.get("control_or_state_visible")
            grounded = control_visible if isinstance(control_visible, bool) else None

        group = str(row.get("source_group") or row.get("video_group") or "UNKNOWN")
        normalized.append(
            {
                "case_id": case_id,
                "cues": tuple(cue_values or ("none",)),
                "resolved": bool(resolved),
                "grounded_control": grounded if isinstance(grounded, bool) else None,
                "group": group,
            }
        )
    return normalized


def _independent(document: Mapping[str, Any]) -> bool:
    return document.get("independence_certified") is True


def _has_cue(row: Mapping[str, Any], cue: str) -> bool:
    return cue in set(row["cues"]) - {"none"}


def audit(
    *,
    reviewer_a: Mapping[str, Any],
    reviewer_b: Mapping[str, Any],
    require_full_overlap: bool = True,
) -> dict[str, Any]:
    """Audit cue agreement and recurrence across two blind-review documents."""

    rows_a = {row["case_id"]: row for row in _rows(reviewer_a)}
    rows_b = {row["case_id"]: row for row in _rows(reviewer_b)}
    overlap = sorted(set(rows_a) & set(rows_b))
    missing_a = sorted(set(rows_b) - set(rows_a))
    missing_b = sorted(set(rows_a) - set(rows_b))

    comparisons: list[dict[str, Any]] = []
    for case_id in overlap:
        for cue in TARGET_CUES:
            present_a = _has_cue(rows_a[case_id], cue)
            present_b = _has_cue(rows_b[case_id], cue)
            comparisons.append(
                {
                    "case_id": case_id,
                    "cue": cue,
                    "a": present_a,
                    "b": present_b,
                    "agree": present_a == present_b,
                }
            )

    agreements = sum(item["agree"] for item in comparisons)
    agreement_rate = agreements / len(comparisons) if comparisons else 0.0
    cue_stats: dict[str, dict[str, Any]] = {}
    recurrence: dict[str, dict[str, Any]] = {}

    for cue in TARGET_CUES:
        cue_stats[cue] = {}
        for label, document in (("a", rows_a), ("b", rows_b)):
            matching = [row for row in document.values() if row["resolved"] and _has_cue(row, cue)]
            groups = Counter(row["group"] for row in matching)
            cue_stats[cue][label] = {
                "resolved_rows": len(matching),
                "groups": dict(groups),
            }

        consensus_rows: list[dict[str, Any]] = []
        for case_id in overlap:
            row_a = rows_a[case_id]
            row_b = rows_b[case_id]
            if not (
                row_a["resolved"]
                and row_b["resolved"]
                and _has_cue(row_a, cue)
                and _has_cue(row_b, cue)
            ):
                continue
            group = row_a["group"]
            if group == "UNKNOWN":
                group = row_b["group"]
            consensus_rows.append({**row_a, "group": group})

        groups = Counter(row["group"] for row in consensus_rows)
        qualifying = {group: count for group, count in groups.items() if count >= 2}
        recurrence[cue] = {
            "consensus_resolved_rows": len(consensus_rows),
            "groups": dict(groups),
            "groups_with_at_least_two_rows": qualifying,
            "passes_two_groups": len(qualifying) >= 2,
        }

    grounded_groups: set[str] = set()
    for case_id in overlap:
        row_a = rows_a[case_id]
        row_b = rows_b[case_id]
        if not (
            row_a["resolved"]
            and row_b["resolved"]
            and row_a["grounded_control"] is True
            and row_b["grounded_control"] is True
        ):
            continue
        group = row_a["group"] if row_a["group"] != "UNKNOWN" else row_b["group"]
        grounded_groups.add(group)

    independent_a = _independent(reviewer_a)
    independent_b = _independent(reviewer_b)
    independent_ok = independent_a and independent_b
    full_overlap_ok = not require_full_overlap or set(rows_a) == set(rows_b)
    recurrence_ok = any(item["passes_two_groups"] for item in recurrence.values())
    decision = (
        "ADMIT_CONTEXT_BOTTLENECK_DIAGNOSTIC"
        if (
            independent_ok
            and full_overlap_ok
            and agreement_rate >= 0.75
            and recurrence_ok
            and len(grounded_groups) >= 2
        )
        else "HOLD_NO_VALIDATED_CUE"
    )

    return {
        "protocol_version": "qwen-hard-negative-recurrence-gate-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "label_blind": True,
        "official_reference_opened": False,
        "model_invoked": False,
        "gpu_invoked": False,
        "training_authorized": False,
        "reviewer_independence": {
            "a": independent_a,
            "b": independent_b,
            "both_certified": independent_ok,
        },
        "row_counts": {
            "a": len(rows_a),
            "b": len(rows_b),
            "overlap": len(overlap),
            "missing_from_a": missing_a,
            "missing_from_b": missing_b,
        },
        "agreement": {
            "cue_field_comparisons": len(comparisons),
            "agreement_count": agreements,
            "agreement_rate": round(agreement_rate, 4),
        },
        "cue_stats": cue_stats,
        "recurrence": recurrence,
        "grounded_control_consensus_groups": sorted(grounded_groups),
        "gates": {
            "independence_certified": independent_ok,
            "full_overlap": full_overlap_ok,
            "agreement_at_least_75pct": agreement_rate >= 0.75,
            "same_cue_two_groups_each_two_rows": recurrence_ok,
            "at_least_two_grounded_control_groups": len(grounded_groups) >= 2,
        },
        "decision": decision,
        "layer_map": LAYER,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reviewer_a", type=Path)
    parser.add_argument("reviewer_b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reviewer_a = json.loads(args.reviewer_a.read_text(encoding="utf-8-sig"))
    reviewer_b = json.loads(args.reviewer_b.read_text(encoding="utf-8-sig"))
    result = audit(reviewer_a=reviewer_a, reviewer_b=reviewer_b)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "decision": result["decision"]}))


if __name__ == "__main__":
    main()
