from __future__ import annotations

from copy import deepcopy

import pytest

from robata.benchmark.wemm_label_prototype_fusion import (
    WemmLabelPrototypeFusionError,
    build_diagnostic,
    fuse_rankings,
    fuse_scores,
)

PROTOTYPES = ("canonical", "verb_noun", "natural")
MODES = ("visual", "text", "hybrid")


def _item(
    action: tuple[int, int], rank: int, mode: str, score: float | None = 0.5
) -> dict[str, object]:
    fields: dict[str, object] = {
        "rank": rank,
        "action_key": list(action),
        "verb_id": action[0],
        "noun_id": action[1],
        "verb_key": f"v{action[0]}",
        "noun_key": f"n{action[1]}",
        "label_text": f"v{action[0]} n{action[1]}",
        "visual_score": None,
        "text_score": None,
        "fused_score": None,
    }
    fields[{"visual": "visual_score", "text": "text_score", "hybrid": "fused_score"}[mode]] = score
    return fields


def _report(
    *,
    rows: list[tuple[str, tuple[int, int]]],
    rankings: dict[str, dict[str, list[dict[str, object]]]],
    catalog: list[tuple[int, int]],
    audit_keys: list[str] | None = None,
) -> dict[str, object]:
    case_deltas = [{"id": row_id, "ground_truth": list(target)} for row_id, target in rows]
    result: dict[str, object] = {}
    for prototype in PROTOTYPES:
        result[prototype] = {
            "rankings": {
                mode: {row_id: value for row_id, value in rows_for_mode.items()}
                for mode, rows_for_mode in rankings[prototype].items()
            },
            "metrics": {"metrics": {mode: {} for mode in MODES}},
        }
    result["canonical"]["metrics"]["case_deltas"] = case_deltas  # type: ignore[index]
    payload: dict[str, object] = {
        "report_version": "fixture",
        "input": {
            "catalog_size": len(catalog),
            "case_count": len(rows),
            "label_variants": list(PROTOTYPES),
        },
        "labels": [{"action_key": list(action)} for action in catalog],
        "results": result,
    }
    if audit_keys is not None:
        payload["input"]["row_input_audit"] = [  # type: ignore[index]
            {"row_key": key} for key in audit_keys
        ]
    return payload


def _rankings(
    row_id: str,
    per_prototype: dict[str, list[tuple[int, int]]],
    *,
    depth: int,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    return {
        prototype: {
            mode: {
                row_id: [
                    _item(action, index, mode)
                    for index, action in enumerate(actions[:depth], start=1)
                ]
            }
            for mode in MODES
        }
        for prototype, actions in per_prototype.items()
    }


def test_rrf_is_deterministic_and_tracks_prototype_support() -> None:
    rankings = {
        "canonical": [_item((0, 1), 1, "visual"), _item((0, 0), 2, "visual")],
        "verb_noun": [_item((0, 1), 1, "visual"), _item((1, 0), 2, "visual")],
        "natural": [_item((0, 0), 1, "visual"), _item((0, 1), 2, "visual")],
    }
    first = fuse_rankings(rankings, rrf_k=1)
    second = fuse_rankings({key: rankings[key] for key in reversed(tuple(rankings))}, rrf_k=1)

    assert first == second
    assert first[0]["action_key"] == [0, 1]
    assert first[0]["support_count"] == 3
    assert first[1]["action_key"] == [0, 0]


def test_complete_score_fusion_uses_equal_prototype_mean() -> None:
    rankings = {
        prototype: [
            {**_item((0, 0), 1, "visual", score), "score": score},
            {**_item((0, 1), 2, "visual", other), "score": other},
        ]
        for prototype, score, other in (
            ("canonical", 0.9, 0.8),
            ("verb_noun", 0.2, 0.8),
            ("natural", 0.4, 0.1),
        )
    }
    fused = fuse_scores(rankings)

    assert fused[0]["action_key"] == [0, 1]
    assert fused[0]["mean_score"] == pytest.approx((0.8 + 0.8 + 0.1) / 3)
    assert fused[1]["action_key"] == [0, 0]


def test_incomplete_sidecar_suppresses_score_fusion_and_preserves_input() -> None:
    rows = [("row-a", (0, 0))]
    by_prototype = {prototype: [(0, 0), (0, 1)] for prototype in PROTOTYPES}
    rankings = {
        prototype: {
            mode: {"row-a": [_item(action, index, mode) for index, action in enumerate(actions, 1)]}
            for mode in MODES
        }
        for prototype, actions in by_prototype.items()
    }
    source = _report(rows=rows, rankings=rankings, catalog=[(0, 0), (0, 1), (1, 0)])
    before = deepcopy(source)

    diagnostic = build_diagnostic(source)

    assert source == before
    assert diagnostic["experiment"]["score_fusion"]["available_all_modes"] is False  # type: ignore[index]
    assert diagnostic["score_fused_rankings"] == {mode: {} for mode in MODES}
    assert (
        "top-k truncated" in diagnostic["experiment"]["score_fusion"]["by_mode"]["visual"]["reason"]  # type: ignore[index]
    )
    assert diagnostic["fusion_metrics"]["visual"]["rank_rrf"]["scored_query_count"] == 1  # type: ignore[index]
    assert diagnostic["fusion_metrics"]["visual"]["rank_rrf"]["target_found_count"] == 1  # type: ignore[index]


def test_row_input_audit_key_precedes_legacy_case_delta_id() -> None:
    rows = [("legacy-id", (0, 0))]
    stable = "uid-a"
    rankings = {
        prototype: {mode: {stable: [_item((0, 0), 1, mode)]} for mode in MODES}
        for prototype in PROTOTYPES
    }
    source = _report(
        rows=rows,
        rankings=rankings,
        catalog=[(0, 0)],
        audit_keys=[stable],
    )

    diagnostic = build_diagnostic(source)

    assert diagnostic["case_targets"] == [{"id": stable, "ground_truth": [0, 0]}]
    assert list(diagnostic["fused_rankings"]["visual"]) == [stable]  # type: ignore[index]
    assert diagnostic["fusion_metrics"]["visual"]["rank_rrf"]["top1_accuracy"] == 1.0  # type: ignore[index]


def test_duplicate_action_in_one_prototype_fails_closed() -> None:
    rankings = {
        "canonical": [_item((0, 0), 1, "visual"), _item((0, 0), 2, "visual")],
        "verb_noun": [],
        "natural": [],
    }

    with pytest.raises(WemmLabelPrototypeFusionError, match="duplicate action"):
        fuse_rankings(rankings)


def test_score_method_requires_complete_scores() -> None:
    rows = [("row-a", (0, 0))]
    rankings = {
        prototype: {mode: {"row-a": [_item((0, 0), 1, mode)]} for mode in MODES}
        for prototype in PROTOTYPES
    }
    source = _report(rows=rows, rankings=rankings, catalog=[(0, 0), (0, 1)])

    with pytest.raises(WemmLabelPrototypeFusionError, match="score fusion requested"):
        build_diagnostic(source, method="score")
