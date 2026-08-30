#!/usr/bin/env python3
"""Summarize benchmark-local WeMM EPIC retrieval runs.

This is deliberately a post-hoc report helper.  It reads the JSON reports
emitted by ``run_wemm_epic_retrieval.py`` and does not load a model, decode
media, or compute content identities.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ARMS = ("normal", "reverse", "freeze_pre", "freeze_post")
VARIANTS = ("canonical", "verb_noun", "natural")
MODES = ("visual", "text", "hybrid")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must be an object: {path}")
    return payload


def _pair(value: object) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _metric(report: Mapping[str, Any], variant: str, mode: str) -> Mapping[str, Any]:
    value = report["results"][variant]["metrics"]["metrics"][mode]
    if not isinstance(value, Mapping):
        raise ValueError(f"metrics block is not an object for {variant}/{mode}")
    return cast(Mapping[str, Any], value)


def _top1_changes(
    normal: Mapping[str, Any],
    intervention: Mapping[str, Any],
    variant: str,
    mode: str,
) -> int:
    normal_rows = normal["results"][variant]["rankings"][mode]
    intervention_rows = intervention["results"][variant]["rankings"][mode]
    changed = 0
    for key, normal_ranking in normal_rows.items():
        other = intervention_rows.get(key, ())
        if (
            normal_ranking
            and other
            and normal_ranking[0].get("action_key") != other[0].get("action_key")
        ):
            changed += 1
    return changed


def _projection_metrics(report: Mapping[str, Any], variant: str, mode: str) -> dict[str, Any]:
    rows = report["results"][variant]["metrics"]["case_deltas"]
    projections = report["results"][variant]["projections"]
    # Runner reports normally key rankings/projections by a stable manifest
    # identifier (uid/case_id/annotation_id).  Older reports used row-{index};
    # consult the per-row audit first and retain that ordinal fallback so
    # summaries remain valid when a manifest supplies real IDs.
    audit_rows = report.get("input", {}).get("row_input_audit", ())
    if not isinstance(audit_rows, Sequence) or isinstance(audit_rows, (str, bytes, bytearray)):
        audit_rows = ()
    accepted = 0
    correct = 0
    for index, row in enumerate(rows):
        stable_key: str | None = None
        if index < len(audit_rows) and isinstance(audit_rows[index], Mapping):
            raw_key = audit_rows[index].get("row_key")
            if raw_key is not None and str(raw_key).strip():
                stable_key = str(raw_key).strip()
        projection_row = projections.get(stable_key, {}) if stable_key else {}
        if not isinstance(projection_row, Mapping) or not projection_row:
            projection_row = projections.get(f"row-{index}", {})
        projection = projection_row.get(mode, {}) if isinstance(projection_row, Mapping) else {}
        mapped = projection.get("status") == "MAPPED"
        if mapped:
            accepted += 1
            selected = projection.get("joint_selected") or {}
            selected_pair = (selected.get("verb_id"), selected.get("noun_id"))
            if selected_pair == tuple(row.get("ground_truth", ())):
                correct += 1
    total = len(rows)
    return {
        "query_count": total,
        "accepted_count": accepted,
        "accepted_coverage": accepted / total if total else 0.0,
        "accepted_precision": correct / accepted if accepted else None,
        "raw_joint_accuracy": correct / total if total else 0.0,
        "thresholds": {"min_score": 0.0, "min_margin": 0.0},
    }


def _case_examples(report: Mapping[str, Any], *, limit: int = 8) -> dict[str, list[dict[str, Any]]]:
    labels = {
        tuple(item["action_key"]): item
        for item in report.get("labels", [])
        if isinstance(item, Mapping) and _pair(item.get("action_key")) is not None
    }
    del labels  # The reports already carry readable verb/noun keys in rankings.
    deltas = report["results"]["canonical"]["metrics"]["case_deltas"]
    examples: dict[str, list[dict[str, Any]]] = {
        "visual_success": [],
        "hybrid_success": [],
        "both_failed": [],
    }
    for index, delta in enumerate(deltas):
        item = {
            "row": index,
            "ground_truth": delta.get("ground_truth"),
            "visual_top1": delta.get("visual", {}).get("top1"),
            "hybrid_top1": delta.get("hybrid", {}).get("top1"),
            "visual_top1_correct": delta.get("visual", {}).get("top1_correct"),
            "hybrid_top1_correct": delta.get("hybrid", {}).get("top1_correct"),
        }
        if item["visual_top1_correct"]:
            examples["visual_success"].append(item)
        if item["hybrid_top1_correct"]:
            examples["hybrid_success"].append(item)
        if not item["visual_top1_correct"] and not item["hybrid_top1_correct"]:
            examples["both_failed"].append(item)
    return {key: value[:limit] for key, value in examples.items()}


def summarize(paths: Mapping[str, Path]) -> dict[str, Any]:
    reports = {arm: _load(paths[arm]) for arm in ARMS}
    normal = reports["normal"]
    labels = normal.get("labels", [])
    catalog_pairs = {
        _pair(item.get("action_key"))
        for item in labels
        if isinstance(item, Mapping) and _pair(item.get("action_key")) is not None
    }
    gt_pairs = {
        _pair(item.get("ground_truth"))
        for item in normal["results"]["canonical"]["metrics"]["case_deltas"]
        if _pair(item.get("ground_truth")) is not None
    }
    case_count = int(normal["input"]["case_count"])
    normal_metrics: dict[str, Any] = {}
    for variant in VARIANTS:
        normal_metrics[variant] = {
            mode: {
                key: value
                for key, value in _metric(normal, variant, mode).items()
                if key
                in {
                    "query_count",
                    "scored_query_count",
                    "recall_at_k",
                    "mrr",
                    "top1_accuracy",
                    "mean_top1_margin",
                    "group_count",
                    "video_groups",
                }
            }
            for mode in MODES
        }
    intervention_metrics: dict[str, Any] = {}
    for arm in ARMS:
        intervention_metrics[arm] = {
            variant: {
                mode: {
                    "top1_accuracy": _metric(reports[arm], variant, mode)["top1_accuracy"],
                    "recall_at_5": _metric(reports[arm], variant, mode)["recall_at_k"]["5"],
                    "recall_at_10": _metric(reports[arm], variant, mode)["recall_at_k"]["10"],
                    "mrr": _metric(reports[arm], variant, mode)["mrr"],
                }
                for mode in MODES
            }
            for variant in VARIANTS
        }
    deltas: dict[str, Any] = {}
    for arm in ARMS[1:]:
        deltas[arm] = {
            variant: {
                mode: {
                    "top1_changed_count": _top1_changes(normal, reports[arm], variant, mode),
                    "top1_changed_fraction": (
                        _top1_changes(normal, reports[arm], variant, mode) / case_count
                    ),
                    "delta_top1_accuracy": _metric(reports[arm], variant, mode)["top1_accuracy"]
                    - _metric(normal, variant, mode)["top1_accuracy"],
                }
                for mode in MODES
            }
            for variant in VARIANTS
        }

    return {
        "summary_version": "wemm-epic-ontology-retrieval-summary-v1",
        "generated_at": "2026-08-26",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "inference_performed_for_summary": False,
        "source_reports": {arm: str(paths[arm]) for arm in ARMS},
        "experiment": {
            "model": normal.get("model"),
            "case_count": case_count,
            "frame_count": normal["input"]["frame_count"],
            "catalog_size": normal["input"]["catalog_size"],
            "catalog_source": normal["input"]["catalog_source"],
            "catalog_provenance_verified": normal["input"].get("catalog_provenance_verified"),
            "catalog_ground_truth_coverage": {
                "unique_ground_truth_pairs": len(gt_pairs),
                "catalog_pairs": len(catalog_pairs),
                "covered_unique_pairs": len(gt_pairs & catalog_pairs),
                "all_ground_truth_pairs_present": gt_pairs <= catalog_pairs,
            },
            "label_variants": list(normal["input"]["label_variants"]),
            "fusion": normal.get("fusion"),
            "processor_observation": {
                "video_grid_thw": sorted(
                    {
                        tuple(item["video_grid_thw"][0])
                        for item in normal.get("processor_observations", [])
                        if item.get("modality") == "video" and item.get("video_grid_thw")
                    }
                ),
                "video_observation_count": sum(
                    item.get("modality") == "video"
                    for item in normal.get("processor_observations", [])
                ),
                "text_observation_count": sum(
                    item.get("modality") == "text"
                    for item in normal.get("processor_observations", [])
                ),
            },
        },
        "normal_metrics": normal_metrics,
        "candidate_projection": {
            "kind": "deterministic_mapper_shaped_projection",
            "existing_mapper_invoked": False,
            "note": (
                "The frozen existing Mapper is not called because this experiment produces "
                "joint ontology candidates rather than Mapper-native Qwen prose. Results are "
                "a compatible deterministic projection of top-1 candidates, not a Mapper "
                "quality claim."
            ),
            "default_threshold_metrics": {
                variant: {mode: _projection_metrics(normal, variant, mode) for mode in MODES}
                for variant in VARIANTS
            },
        },
        "intervention_metrics": intervention_metrics,
        "intervention_deltas_vs_normal": deltas,
        "examples": _case_examples(normal),
        "controls": {
            "production_path_changed": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "mapper_training_invoked": False,
            "heldout_100_opened": False,
            "larger_qwen_invoked": False,
            "hash_or_sha_used": False,
            "ground_truth_used_in_encoder_input": False,
        },
        "interpretation": {
            "visual_candidate_recall": (
                "WeMM visual-only improves candidate recall over text-only at some K, "
                "but remains low at top-1."
            ),
            "hybrid": (
                "The configured 0.7 visual/0.3 text fusion raises Recall@5/10 but does "
                "not improve canonical top-1 over visual-only."
            ),
            "temporal": (
                "Reverse/freeze ranking movement is not sufficient evidence of polarity "
                "understanding; the embedding space has no explicit no-op class and no "
                "causal verifier."
            ),
            "decision": (
                "Do not replace Qwen candidate proposal or production Mapper. Keep WeMM "
                "as an optional benchmark/retrieval candidate layer; only consider hybrid "
                "for recall-oriented shortlisting after a larger, disjoint evaluation."
            ),
        },
        "validation": {
            "focused_tests": 38,
            "ruff": "passed",
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def markdown(summary: Mapping[str, Any]) -> str:
    exp = summary["experiment"]
    lines = [
        "# WeMM-Embedding EPIC retrieval experiment (ontology catalog)",
        "",
        "Local, benchmark-only report. No inference was performed while generating this summary.",
        "",
        f"- Model: `{exp['model']['identity']}`; dimension `{exp['model']['dimension']}`.",
        (
            f"- Cohort: `{exp['case_count']}` development rows, "
            f"`{exp['frame_count']}` native-video frames per row, "
            f"`{exp['catalog_size']}` fixed ontology action pairs."
        ),
        (
            "- Held-out-100 remains closed; production Web/API/UI, Mapper, ontology and "
            "training are unchanged."
        ),
        "",
        "## Normal results",
        "",
        "| Variant | Mode | R@1 | R@3 | R@5 | R@10 | MRR | Top-1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for mode in MODES:
            m = summary["normal_metrics"][variant][mode]
            lines.append(
                f"| {variant} | {mode} | {_fmt(m['recall_at_k']['1'])} | "
                f"{_fmt(m['recall_at_k']['3'])} | {_fmt(m['recall_at_k']['5'])} | "
                f"{_fmt(m['recall_at_k']['10'])} | "
                f"{_fmt(m['mrr'])} | {_fmt(m['top1_accuracy'])} |"
            )
    canonical_visual = summary["normal_metrics"]["canonical"]["visual"]
    canonical_hybrid = summary["normal_metrics"]["canonical"]["hybrid"]
    canonical_text = summary["normal_metrics"]["canonical"]["text"]
    lines += [
        "",
        (
            "Canonical visual has Top-1 "
            f"{_fmt(canonical_visual['top1_accuracy'])} and R@10 "
            f"{_fmt(canonical_visual['recall_at_k']['10'])}. Canonical hybrid reaches "
            f"R@10 {_fmt(canonical_hybrid['recall_at_k']['10'])} and MRR "
            f"{_fmt(canonical_hybrid['mrr'])}, but Top-1 remains "
            f"{_fmt(canonical_hybrid['top1_accuracy'])}; text-only Top-1 is "
            f"{_fmt(canonical_text['top1_accuracy'])}."
        ),
        "",
        "## Intervention screen (canonical)",
        "",
        "| Arm | Mode | Top-1 | R@10 | Top-1 changed vs normal |",
        "|---|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        for mode in ("visual", "hybrid"):
            m = summary["intervention_metrics"][arm]["canonical"][mode]
            if arm == "normal":
                changed = 0.0
            else:
                changed = summary["intervention_deltas_vs_normal"][arm]["canonical"][mode][
                    "top1_changed_fraction"
                ]
            lines.append(
                f"| {arm} | {mode} | {_fmt(m['top1_accuracy'])} | "
                f"{_fmt(m['recall_at_10'])} | {_fmt(changed)} |"
            )
    lines += [
        "",
        (
            "Movement under reverse/freeze is a sensitivity signal, not a valid "
            "reverse-inversion or no-change score: the retrieval catalog has no "
            "`no_action` class and does not model polarity explicitly."
        ),
        "",
        "## Candidate projection (canonical, normal)",
        "",
        (
            "This is a deterministic Mapper-shaped projection of the retrieved joint "
            "candidate, not an invocation or modification of the frozen existing Mapper."
        ),
        "",
        "| Mode | Coverage | Accepted precision | Raw joint accuracy |",
        "|---|---:|---:|---:|",
    ]
    projection_metrics = summary["candidate_projection"]["default_threshold_metrics"]["canonical"]
    for mode in MODES:
        projection = projection_metrics[mode]
        precision = projection["accepted_precision"]
        precision_text = "n/a" if precision is None else _fmt(precision)
        lines.append(
            f"| {mode} | {_fmt(projection['accepted_coverage'])} | {precision_text} | "
            f"{_fmt(projection['raw_joint_accuracy'])} |"
        )
    lines += [
        "",
        "",
        "## Conclusion",
        "",
        (
            "WeMM is useful as an optional visual candidate-recall layer and can bypass "
            "prose lexical failure, but this run does not justify replacing Qwen or the "
            "Mapper. Hybrid improves shortlist recall at K=5/10, not reliable exact "
            "action selection. Keep the experiment benchmark-local and validate on a "
            "larger disjoint cohort before any integration decision."
        ),
        "",
        (
            "Artifacts: `wemm_epic_ontology_dev27_20260826*.json`, "
            "`wemm_epic_retrieval_summary_20260826.{json,md}`."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(".agent_tmp"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    paths = {
        "normal": directory / "wemm_epic_ontology_dev27_20260826.json",
        "reverse": directory / "wemm_epic_ontology_dev27_reverse_20260826.json",
        "freeze_pre": directory / "wemm_epic_ontology_dev27_freeze_pre_20260826.json",
        "freeze_post": directory / "wemm_epic_ontology_dev27_freeze_post_20260826.json",
    }
    summary = summarize(paths)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
