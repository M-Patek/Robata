#!/usr/bin/env python3
"""Compare lightweight, post-hoc WeMM candidate-fusion policies.

This command consumes an existing WeMM production shadow and the explicitly
non-gold agent visual review queue.  It does not invoke a model, decode media,
change the EPIC ontology/Mapper, open held-out data, or compute a digest.  All
semantic rows are labelled ``SURROGATE_ONLY`` and are intended only to decide
whether a cheap re-ranking experiment is worth promoting to a future approved
cohort.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robata.benchmark.wemm_multiview_retrieval import fuse_camera_rankings  # noqa: E402

INFLECTIONS = {
    "adjusts": "adjust",
    "adjusting": "adjust",
    "arranges": "arrange",
    "arranging": "arrange",
    "flattens": "flatten",
    "flattening": "flatten",
    "folds": "fold",
    "folding": "fold",
    "picks": "pick",
    "picking": "pick",
    "places": "place",
    "placing": "place",
    "presses": "press",
    "pressing": "press",
    "smooths": "smooth",
    "smoothing": "smooth",
    "spreads": "spread",
    "spreading": "spread",
}
TEXTILE_NOUNS = {
    "cloth",
    "clothes",
    "clothing",
    "fabric",
    "garment",
    "pants",
    "sheets",
    "shirt",
    "shorts",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def norm(value: object) -> str:
    text = re.sub(r"[_/\\-]+", " ", str(value or "").casefold())
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(INFLECTIONS.get(token, token) for token in text.split())


def pair(value: Mapping[str, Any]) -> tuple[str, str] | None:
    raw_pair = value.get("pair")
    if (
        isinstance(raw_pair, Sequence)
        and not isinstance(raw_pair, (str, bytes))
        and len(raw_pair) == 2
    ):
        result = (norm(raw_pair[0]), norm(raw_pair[1]))
        return result if all(result) else None
    result = (
        norm(value.get("verb", value.get("verb_key"))),
        norm(value.get("noun", value.get("noun_key"))),
    )
    return result if all(result) else None


def family(noun: str) -> str:
    return "textile_generic" if noun in TEXTILE_NOUNS else noun


def refs(review: Mapping[str, Any]) -> dict[str, dict[str, list[tuple[str, str]]]]:
    if review.get("production_eligible") is True:
        raise ValueError("agent review cannot be production eligible")
    contract = review.get("review_contract", {})
    if not isinstance(contract, Mapping) or contract.get("accepted_as_gold") is not False:
        raise ValueError("review must explicitly remain non-gold")
    result: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for raw_item in review.get("items", []):
        if not isinstance(raw_item, Mapping):
            continue
        window_id = str(raw_item.get("window_id", ""))
        primary: list[tuple[str, str]] = []
        envelope: list[tuple[str, str]] = []
        for raw_segment in raw_item.get("segments", []):
            if not isinstance(raw_segment, Mapping):
                continue
            parsed = pair(raw_segment)
            if parsed is None:
                continue
            primary.append(parsed)
            envelope.append(parsed)
            for raw_alt in raw_segment.get("alternatives", []):
                if isinstance(raw_alt, Mapping):
                    alternative = pair(raw_alt)
                    if alternative is not None and alternative not in envelope:
                        envelope.append(alternative)
        if window_id and primary:
            result[window_id] = {"primary": primary, "envelope": envelope}
    return result


def camera_inputs(
    window: Mapping[str, Any],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[tuple[Any, ...], dict[str, Any]],
]:
    cameras: dict[str, list[dict[str, Any]]] = {}
    labels: dict[tuple[Any, ...], dict[str, Any]] = {}
    per_camera = window.get("model", {}).get("per_camera_predictions", {})
    if not isinstance(per_camera, Mapping):
        return cameras, labels
    for camera_id, raw_predictions in per_camera.items():
        if not isinstance(raw_predictions, Sequence):
            continue
        candidates: list[dict[str, Any]] = []
        for raw in raw_predictions:
            if not isinstance(raw, Mapping) or raw.get("action_key") is None:
                continue
            candidate = dict(raw)
            score = candidate.get(
                "fused_score", candidate.get("visual_score", candidate.get("score"))
            )
            if score is not None:
                candidate["score"] = score
            candidates.append(candidate)
            action_key = candidate.get("action_key")
            if isinstance(action_key, Sequence) and not isinstance(action_key, (str, bytes)):
                key = tuple(action_key)
                labels.setdefault(
                    key,
                    {
                        "label_text": candidate.get("label_text"),
                        "verb": candidate.get("verb", candidate.get("verb_key")),
                        "noun": candidate.get("noun", candidate.get("noun_key")),
                        "verb_key": candidate.get("verb_key"),
                        "noun_key": candidate.get("noun_key"),
                    },
                )
        if candidates:
            cameras[str(camera_id)] = candidates
    return cameras, labels


def recorded_candidates(window: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = window.get("model", {}).get("predictions", [])
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def action_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    value = row.get("action_key")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return tuple(value)
    return None


def ranking_variants(window: Mapping[str, Any], top_k: int) -> dict[str, list[dict[str, Any]]]:
    cameras, labels = camera_inputs(window)
    recorded = recorded_candidates(window)[:top_k]
    variants: dict[str, list[dict[str, Any]]] = {"recorded_mean": recorded}
    if not cameras:
        return variants
    for name, kwargs in (
        ("mean_recomputed", {"fusion": "mean", "missing_score": "omit"}),
        ("rank_mean", {"fusion": "rank_mean", "missing_score": "omit"}),
        ("rrf", {"fusion": "rrf", "missing_score": "omit"}),
        ("max", {"fusion": "max", "missing_score": "omit"}),
        ("mean_zero", {"fusion": "mean", "missing_score": "zero"}),
    ):
        fused = fuse_camera_rankings(
            cameras,
            camera_order=list(cameras),
            top_k=None,
            score_normalization="unit",
            include_embeddings=False,
            **kwargs,
        )
        rows: list[dict[str, Any]] = []
        for row in fused["candidates"][:top_k]:
            row = dict(row)
            key = action_key(row)
            if key is not None and key in labels:
                row.update({k: v for k, v in labels[key].items() if v is not None})
            rows.append(row)
        variants[name] = rows
    # A transparent coverage-first policy is useful as a sanity check.  It is
    # intentionally not a production recommendation: coverage can favour a
    # persistent distractor.
    base = variants["mean_recomputed"]
    variants["coverage_first"] = sorted(
        base,
        key=lambda row: (
            -float(row.get("camera_coverage_fraction", 0.0)),
            -float(row.get("score", row.get("fused_score", 0.0))),
            str(row.get("action_key")),
        ),
    )[:top_k]
    return variants


def rank_for(
    candidates: Sequence[Mapping[str, Any]], references: Sequence[tuple[str, str]], mode: str
) -> int | None:
    for index, candidate in enumerate(candidates, 1):
        parsed = pair(candidate)
        if parsed is None:
            continue
        matches = []
        for reference in references:
            if mode == "strict_pair":
                matches.append(parsed == reference)
            elif mode == "family_pair":
                matches.append(
                    parsed[0] == reference[0] and family(parsed[1]) == family(reference[1])
                )
            elif mode == "verb_only":
                matches.append(parsed[0] == reference[0])
            elif mode == "noun_family_only":
                matches.append(family(parsed[1]) == family(reference[1]))
        if any(matches):
            return index
    return None


def metrics(
    candidates_by_window: Mapping[str, Sequence[Mapping[str, Any]]],
    references: Mapping[str, Mapping[str, Sequence[tuple[str, str]]]],
    top_k: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for ref_name in ("primary", "envelope"):
        for mode in ("strict_pair", "family_pair", "verb_only", "noun_family_only"):
            ranks = [
                rank_for(candidates_by_window.get(wid, []), refs[ref_name], mode)
                for wid, refs in references.items()
            ]
            denominator = len(ranks)
            output[f"{ref_name}_{mode}"] = {
                "windows": denominator,
                "top1": round(sum(rank == 1 for rank in ranks) / denominator, 4)
                if denominator
                else 0.0,
                "at_k": round(
                    sum(rank is not None and rank <= top_k for rank in ranks) / denominator, 4
                )
                if denominator
                else 0.0,
                "mean_first_match_rank": round(
                    sum(rank for rank in ranks if rank is not None)
                    / sum(rank is not None for rank in ranks),
                    4,
                )
                if any(rank is not None for rank in ranks)
                else None,
            }
    return output


def build_report(
    wemm: Mapping[str, Any], review: Mapping[str, Any], *, top_k: int = 5
) -> tuple[dict[str, Any], str]:
    references = refs(review)
    windows = {
        str(row.get("window_id")): row
        for row in wemm.get("windows", [])
        if isinstance(row, Mapping)
    }
    variant_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for window_id in references:
        for variant, rows in ranking_variants(windows.get(window_id, {}), top_k).items():
            variant_rows.setdefault(variant, {})[window_id] = rows

    route_reports: dict[str, Any] = {}
    per_window: dict[str, Any] = {}
    baseline_top1: dict[str, str | None] = {}
    for variant, rows_by_window in variant_rows.items():
        rows_report: dict[str, Any] = {}
        for window_id, rows in rows_by_window.items():
            labels = [str(row.get("label_text")) for row in rows if row.get("label_text")]
            top1 = labels[0] if labels else None
            if variant == "recorded_mean":
                baseline_top1[window_id] = top1
            per_window.setdefault(window_id, {})[variant] = {
                "top1": top1,
                "top5": labels[:top_k],
                "top1_camera_coverage": rows[0].get("camera_coverage") if rows else None,
                "top1_score": rows[0].get("score", rows[0].get("fused_score")) if rows else None,
                "margin_top1_top2": (
                    float(rows[0].get("score", rows[0].get("fused_score", 0.0)))
                    - float(rows[1].get("score", rows[1].get("fused_score", 0.0)))
                    if len(rows) > 1
                    else None
                ),
            }
            rows_report[window_id] = [
                {
                    "label_text": row.get("label_text"),
                    "action_key": row.get("action_key"),
                    "score": row.get("score", row.get("fused_score")),
                    "camera_coverage": row.get("camera_coverage"),
                    "camera_coverage_fraction": row.get("camera_coverage_fraction"),
                }
                for row in rows
            ]
        route_reports[variant] = {
            "measurement_status": "SURROGATE_ONLY",
            "metrics": metrics(rows_by_window, references, top_k),
            "rows": rows_report,
        }

    for window_id, variants in per_window.items():
        baseline = baseline_top1.get(window_id)
        for variant, payload in variants.items():
            payload["top1_changed_vs_recorded"] = (
                variant != "recorded_mean" and payload["top1"] != baseline
            )

    # Margin-aware output stratification is intentionally diagnostic.  It can
    # route a low-margin answer to review but cannot establish correctness.
    tiers: dict[str, int] = Counter()
    for _window_id, variants in per_window.items():
        row = variants.get("recorded_mean", {})
        coverage = float(row.get("top1_camera_coverage") or 0.0) / 6.0
        margin = float(row.get("margin_top1_top2") or 0.0)
        tier = "HIGH_CONSENSUS" if coverage >= 0.8 and margin >= 0.01 else "REVIEW"
        if coverage < 0.5 and margin < 0.01:
            tier = "ABSTAIN_CANDIDATE"
        tiers[tier] += 1

    payload: dict[str, Any] = {
        "format": "robata-production-wemm-fusion-variant-comparison-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "AGENT_SURROGATE_MEASURED_NON_GOLD",
        "quality_claim": False,
        "official_quality_status": "NOT_MEASURED",
        "source": {
            "wemm_artifact": ".agent_tmp/production_wemm_shadow_4s_20260827.json",
            "review_artifact": ".agent_tmp/production_agent_reviewed_segments_4s_16f_20260827.json",
            "window_count": len(references),
            "top_k": top_k,
        },
        "variants": route_reports,
        "margin_stratification": {
            "baseline": "recorded_mean",
            "tiers": dict(tiers),
            "policy": (
                "coverage>=0.8 and margin>=0.01 => HIGH_CONSENSUS; "
                "coverage<0.5 and margin<0.01 => ABSTAIN_CANDIDATE; otherwise REVIEW"
            ),
            "measurement_status": "SURROGATE_ONLY",
        },
        "per_window": per_window,
        "decision": {
            "promotion": "NONE",
            "reason": (
                "single 10-window agent-surrogate cohort; no independent production "
                "gold and no cross-group replication"
            ),
            "observed_finding": (
                "rank-only/reciprocal-rank variants over-promote one-camera candidates; "
                "coverage penalties alter labels but do not recover strict or family action pairs"
            ),
        },
        "controls": {
            "model_invoked": False,
            "source_media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_digest_computed": False,
        },
    }
    lines = [
        "# WeMM production fusion-variant diagnostic",
        "",
        "> **AGENT_SURROGATE_MEASURED_NON_GOLD.** No production accuracy claim is made.",
        "",
        "| Variant | Strict @1 | Strict @K | Family @1 | Family @K | Verb @K | Noun-family @K |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, report in route_reports.items():
        m = report["metrics"]
        lines.append(
            f"| {variant} | {m['primary_strict_pair']['top1']:.1%} | "
            f"{m['primary_strict_pair']['at_k']:.1%} | "
            f"{m['primary_family_pair']['top1']:.1%} | {m['primary_family_pair']['at_k']:.1%} | "
            f"{m['primary_verb_only']['at_k']:.1%} | {m['primary_noun_family_only']['at_k']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `recorded_mean` is the frozen WeMM multiview ranking.",
            "- Rank-only/reciprocal-rank policies can promote a candidate seen at rank 1 "
            "in one camera even when other cameras disagree.",
            "- Coverage penalties/coverage-first policies change the shortlist but do not "
            "overcome the production vocabulary/action-boundary gap.",
            "- Margin tiers are routing signals only; they do not turn a candidate into gold.",
            "- No variant is promoted without approved labels and a second independent "
            "video group.",
            "",
            "## Baseline margin tiers",
            "",
            f"`{dict(tiers)}`",
            "",
        ]
    )
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wemm", type=Path, default=Path(".agent_tmp/production_wemm_shadow_4s_20260827.json")
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path(".agent_tmp/production_agent_reviewed_segments_4s_16f_20260827.json"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    payload, markdown = build_report(load(args.wemm), load(args.review), top_k=args.top_k)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "variants": list(payload["variants"]),
                "windows": payload["source"]["window_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
