#!/usr/bin/env python3
"""Summarize an agent-surrogate production comparison without upgrading it to gold.

The production recording has no independently adjudicated action labels.  This
command joins the ordinary frozen evaluator output with the audit-light
agent-reference comparison and emits a clearly non-gold report.  It performs
no model inference, media decoding, ontology mutation, resolver rerun, or
identity/hash calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _route_surrogate_metrics(
    comparator: dict[str, Any], route: str, reference_set: str
) -> dict[str, Any]:
    route_payload = comparator.get("routes", {}).get(route, {})
    metrics = route_payload.get("metrics", {})
    strict = metrics.get(f"{reference_set}_strict_pair", {})
    family = metrics.get(f"{reference_set}_textile_pair", {})
    verb = metrics.get(f"{reference_set}_verb_only", {})
    candidate_windows = sum(
        bool(values)
        for values in route_payload.get("candidate_windows", {}).values()
        if isinstance(values, list)
    )
    denominator = int(strict.get("windows", family.get("windows", 0)) or 0)
    return {
        "windows": denominator,
        "candidate_windows": candidate_windows,
        "candidate_coverage": _fraction(candidate_windows, denominator),
        "strict_pair": {
            "top1": strict.get("agreement_at_1_fraction"),
            "at5": strict.get("coverage_at_k_fraction"),
            "mean_first_match_rank": strict.get("mean_first_match_rank"),
        },
        "textile_family": {
            "top1": family.get("agreement_at_1_fraction"),
            "at5": family.get("coverage_at_k_fraction"),
            "mean_first_match_rank": family.get("mean_first_match_rank"),
        },
        "verb_only": {
            "top1": verb.get("agreement_at_1_fraction"),
            "at5": verb.get("coverage_at_k_fraction"),
            "mean_first_match_rank": verb.get("mean_first_match_rank"),
        },
    }


def build_report(
    *,
    surrogate_pack: dict[str, Any],
    strict_evaluation: dict[str, Any],
    comparator: dict[str, Any],
    strict_evaluation_artifact: str | None = None,
) -> tuple[dict[str, Any], str]:
    if surrogate_pack.get("reference_status") != "AGENT_SURROGATE_NON_GOLD":
        raise ValueError("surrogate pack is not explicitly marked AGENT_SURROGATE_NON_GOLD")
    if surrogate_pack.get("reviewer_type") != "AGENT_SURROGATE_VISUAL_REVIEW":
        raise ValueError("surrogate pack reviewer_type is not agent visual review")
    if surrogate_pack.get("official_gold_status") != "NOT_ESTABLISHED":
        raise ValueError("surrogate pack unexpectedly claims official gold")
    if comparator.get("status") != "NON_GOLD_EXPLORATORY":
        raise ValueError("comparator is not NON_GOLD_EXPLORATORY")

    strict_models: dict[str, Any] = {}
    for name in ("wemm", "qwen", "merged", "mage"):
        model = strict_evaluation.get("models", {}).get(name, {})
        strict_models[name] = {
            "measurement_status": model.get("measurement_status"),
            "top1_precision": model.get("top1_precision"),
            "top1_recall": model.get("top1_recall"),
            "candidate_recall_at_k": model.get("candidate_recall_at_k", {}),
            "mrr": model.get("mrr"),
            "coverage": model.get("coverage"),
            "abstention_rate": model.get("abstention_rate"),
            "structured_fields_measurement_status": model.get(
                "structured_fields_measurement_status"
            ),
            "boundary_status": model.get("boundary", {}).get("measurement_status"),
        }

    exploratory_routes = {
        route: {
            "primary": _route_surrogate_metrics(comparator, route, "primary"),
            "primary_plus_alternatives": _route_surrogate_metrics(comparator, route, "envelope"),
        }
        for route in ("wemm", "qwen", "hybrid")
    }
    payload: dict[str, Any] = {
        "format": "robata-production-surrogate-quality-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "AGENT_SURROGATE_MEASURED_NON_GOLD",
        "quality_claim": False,
        "official_quality_status": "NOT_MEASURED",
        "reason": (
            "all reference segments were accepted by an agent visual surrogate; "
            "no independent human adjudication or official production gold exists"
        ),
        "reference": {
            "artifact": ".agent_tmp/production_review_pack_agent_surrogate_4s_16f_20260827.json",
            "reviewer_type": surrogate_pack.get("reviewer_type"),
            "official_gold_status": surrogate_pack.get("official_gold_status"),
            "human_adjudication": surrogate_pack.get("human_adjudication"),
            "accepted_window_count": strict_evaluation.get("source", {}).get(
                "accepted_window_count"
            ),
            "accepted_segment_count": strict_evaluation.get("source", {}).get(
                "accepted_segment_count"
            ),
        },
        "strict_exact_evaluator": {
            "artifact": strict_evaluation_artifact
            or ".agent_tmp/production_quality_evaluation_agent_surrogate_4s_16f_20260827.json",
            "measurement_status": "SURROGATE_ONLY",
            "models": strict_models,
        },
        "exploratory_family_comparison": {
            "artifact": ".agent_tmp/production_agent_reference_provisional_20260827.json",
            "measurement_status": "SURROGATE_ONLY",
            "routes": exploratory_routes,
            "alias_policy": comparator.get("reference", {}).get("matching_modes", {}),
        },
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "official_evaluator_invoked": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_digest_computed": False,
        },
        "interpretation": [
            (
                "Strict exact scores are zero because the provisional production noun "
                "garment is not the EPIC ontology noun cloth/clothes and the routes "
                "were not rerun against a production-approved ontology."
            ),
            (
                "Under the explicitly documented textile-family aid, Qwen has stronger "
                "candidate overlap than WeMM on this tiny cohort; this is a routing "
                "signal, not production precision or recall."
            ),
            (
                "The fixed WeMM/Qwen hybrid remains WeMM-dominated and does not inherit "
                "Qwen's family overlap; no weight change is promoted from this sample."
            ),
            (
                "Mage is omitted from semantic comparison because the required native "
                "codec/cache parity is unavailable."
            ),
        ],
    }

    def pct(value: Any) -> str:
        return f"{float(value):.1%}" if isinstance(value, (int, float)) else "—"

    lines = [
        "# Production surrogate quality comparison — 2026-08-27",
        "",
        "> **AGENT_SURROGATE_MEASURED_NON_GOLD.** Numbers below are exploratory and do "
        "not establish production quality.",
        "",
        "- Official quality status: `NOT_MEASURED`",
        "- Reference: agent visual surrogate; official gold: `NOT_ESTABLISHED`",
        "- Cohort: 10 contiguous 4-second windows, six cameras",
        "",
        "## Strict exact projection (surrogate reference)",
        "",
        "| Route | Top-1 precision | Top-1 recall | R@1 | R@5 | MRR | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route in ("wemm", "qwen", "merged", "mage"):
        metric = strict_models[route]
        recall = metric["candidate_recall_at_k"]
        lines.append(
            f"| {route} | {pct(metric['top1_precision'])} | {pct(metric['top1_recall'])} | "
            f"{pct(recall.get('1'))} | {pct(recall.get('5'))} | {pct(metric['mrr'])} | "
            f"{pct(metric['coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## Explicit textile-family exploratory aid",
            "",
            "| Route | Primary family @1 | Primary family @5 | Primary+alternatives @5 | Verb @5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for route in ("wemm", "qwen", "hybrid"):
        primary = exploratory_routes[route]["primary"]
        envelope = exploratory_routes[route]["primary_plus_alternatives"]
        lines.append(
            f"| {route} | {pct(primary['textile_family']['top1'])} | "
            f"{pct(primary['textile_family']['at5'])} | "
            f"{pct(envelope['textile_family']['at5'])} | "
            f"{pct(envelope['verb_only']['at5'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Do not call the surrogate table production accuracy.",
            "- Keep WeMM EPIC baseline as the retrieval reference, but do not use its EPIC "
            "ontology as a production gold vocabulary.",
            "- Qwen is the stronger provisional semantic candidate route on this cohort; "
            "the hybrid weight is not promoted.",
            "- Obtain an approved production vocabulary/independent review before claiming "
            "precision, recall, boundaries, or training-set quality.",
            "",
        ]
    )
    return payload, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surrogate-pack", type=Path, required=True)
    parser.add_argument("--strict-evaluation", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload, markdown = build_report(
            surrogate_pack=_load(args.surrogate_pack),
            strict_evaluation=_load(args.strict_evaluation),
            comparator=_load(args.comparator),
            strict_evaluation_artifact=str(args.strict_evaluation),
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"production surrogate evaluation failed: {exc}")
        return 2
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "status": payload["status"],
                "official_quality_status": payload["official_quality_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
