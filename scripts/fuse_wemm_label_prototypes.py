#!/usr/bin/env python3
"""Run a benchmark-local WeMM label-prototype fusion diagnostic.

The command is intentionally post-hoc: it reads an existing retrieval JSON
sidecar and writes a separate diagnostic JSON/Markdown pair.  It never loads a
model, decodes media, opens a dataset manifest, invokes the Mapper, or computes
an identity/hash.  ``auto`` uses RRF when the source rankings are truncated and
only permits score fusion for a complete finite score matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robata.benchmark.wemm_label_prototype_fusion import (  # noqa: E402
    FUSION_VERSION,
    MODES,
    PROTOTYPES,
    WemmLabelPrototypeFusionError,
    build_diagnostic,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WemmLabelPrototypeFusionError(f"source report must be a JSON object: {path}")
    return payload


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        try:
            number = float(str(value))
        except (TypeError, ValueError):
            return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.3f}"


def _metric_cell(metric: dict[str, Any]) -> str:
    recall = metric.get("recall_at_k", {})
    return " / ".join(
        [
            _fmt(recall.get("1")),
            _fmt(recall.get("3")),
            _fmt(recall.get("5")),
            _fmt(recall.get("10")),
            _fmt(metric.get("mrr")),
            _fmt(metric.get("top1_accuracy")),
        ]
    )


def _target_fields(metric: dict[str, Any]) -> tuple[object, object, object]:
    return (
        metric.get("scored_query_count"),
        metric.get("target_found_count"),
        metric.get("target_coverage"),
    )


def _render_markdown(diagnostic: dict[str, Any]) -> str:
    experiment = diagnostic["experiment"]
    score = experiment["score_fusion"]
    lines = [
        "# WeMM EPIC label-prototype fusion diagnostic (exploratory)",
        "",
        "> **Status: `LOCAL_NONPRODUCTION_ONLY` / exploratory.** This is a pure",
        "> post-processing audit over an existing WeMM sidecar. It is not a model",
        "> replay, a production quality result, or a Mapper/ontology change.",
        "",
        "## Scope and input",
        "",
        f"- Source sidecar: `{diagnostic.get('source_report') or '(in-memory)'}`",
        (
            f"- Cohort: `{experiment['case_count']}` rows; catalog declaration "
            f"`{experiment['catalog_size']}` actions"
        ),
        f"- Prototype surfaces: `{', '.join(experiment['prototypes'])}`",
        f"- Modes: `{', '.join(experiment['modes'])}`",
        (
            "- Rank method: reciprocal-rank fusion (RRF), "
            f"`k={experiment['rank_fusion']['rrf_k']}`, equal prototype weights"
        ),
        "",
        "The source rankings retain only a top-k list. Fused metrics below are",
        "therefore **top-k-union diagnostics**; an action absent from every retained",
        "list has no inferred rank.",
        "",
        "## Score-fusion gate",
        "",
        "| Mode | Available? | Max candidates/row | Max scored/row | Decision |",
        "|---|:---:|---:|---:|---|",
    ]
    for mode in MODES:
        item = score["by_mode"][mode]
        lines.append(
            f"| {mode} | {item['available']} | {item['max_candidates_per_row']} | "
            f"{item['max_scored_candidates_per_row']} | {item['reason']} |"
        )
    lines.extend(
        [
            "",
            "Finite score fields are present in the current sidecar, but they do not",
            "cover all 1,259 actions. Accordingly this run admits rank fusion only; no",
            "score-fused quality number is reported.",
            "",
            "## Overall metrics",
            "",
            "Values in each cell are `R@1 / R@3 / R@5 / R@10 / MRR / Top-1`.",
            "Source prototype values are copied from the baseline sidecar; the `RRF`",
            "row is newly computed from retained rankings.",
            "`scored_query_count` means a ranking row was retained; target presence is",
            "reported separately as `target_found_count`/`target_coverage` in JSON.",
            "",
            (
                "| Mode | Prototype / fusion | Scored rows | Target found | Target coverage | "
                "R@1 / R@3 / R@5 / R@10 / MRR / Top-1 |"
            ),
            "|---|---|---:|---:|---:|---|",
        ]
    )
    baseline = diagnostic["baseline_metrics_from_source"]
    reproduction = diagnostic["baseline_metric_reproduction"]
    fused = diagnostic["fusion_metrics"]
    for mode in MODES:
        for prototype in PROTOTYPES:
            source_metric = baseline[mode][prototype]
            recomputed = reproduction[mode][prototype]["recomputed_from_retained_rankings"]
            scored, target_found, target_coverage = _target_fields(recomputed)
            lines.append(
                f"| {mode} | `{prototype}` (source) | "
                f"{scored} | {target_found} | {_fmt(target_coverage)} | "
                f"{_metric_cell(source_metric)} |"
            )
        rrf_metric = fused[mode]["rank_rrf"]
        scored, target_found, target_coverage = _target_fields(rrf_metric)
        lines.append(
            f"| {mode} | **RRF** (top-k union) | {scored} | {target_found} | "
            f"{_fmt(target_coverage)} | {_metric_cell(rrf_metric)} |"
        )

    lines.extend(
        [
            "",
            "## Structural hard-negative comparison",
            "",
            "A hard negative is a retained candidate sharing the target verb, the target",
            "noun, or either. These are label-structural comparisons only, not evidence",
            "that a competing action is visually present.",
            "",
            (
                "| Mode | Kind | Prototype / fusion | Eligible | Scored | Target found | "
                "Target coverage | R@5 | MRR | Top-1 |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    hard = diagnostic["hard_negative_comparison"]["by_mode"]
    for mode in MODES:
        for kind in ("same_verb", "same_noun", "either"):
            for prototype in (*PROTOTYPES, "rank_rrf"):
                metric = hard[mode][kind][prototype]
                name = "RRF" if prototype == "rank_rrf" else prototype
                lines.append(
                    f"| {mode} | {kind} | `{name}` | {metric['eligible_query_count']} | "
                    f"{metric['scored_query_count']} | {metric['target_found_count']} | "
                    f"{_fmt(metric['target_coverage'])} | "
                    f"{_fmt(metric['recall_at_k'].get('5'))} | {_fmt(metric['mrr'])} | "
                    f"{_fmt(metric['top1_accuracy'])} |"
                )

    targets = {
        str(item["id"]): tuple(item["ground_truth"]) for item in diagnostic.get("case_targets", [])
    }
    lines.extend(["", "## Representative RRF structural confusions", ""])
    lines.append("| Mode | Row | Ground truth | RRF top-1 | Relationship | Target rank |")
    lines.append("|---|---|---|---|---|---:|")
    example_count = 0
    for mode in MODES:
        for row_id, ranking in diagnostic["fused_rankings"][mode].items():
            if not ranking:
                continue
            gt = targets.get(row_id)
            if gt is None:
                continue
            top = ranking[0]
            top_pair = tuple(top["action_key"])
            if top_pair == gt:
                continue
            same_verb = top_pair[0] == gt[0]
            same_noun = top_pair[1] == gt[1]
            if not (same_verb or same_noun):
                continue
            if same_verb and not same_noun:
                relationship = "same_verb"
            elif same_noun and not same_verb:
                relationship = "same_noun"
            else:
                relationship = "same_verb+same_noun"
            target_rank = next(
                (int(item["rank"]) for item in ranking if tuple(item["action_key"]) == gt),
                None,
            )
            lines.append(
                f"| {mode} | {row_id} | `{list(gt)}` | `{top['action_key']}` | "
                f"{relationship} | {_fmt(target_rank)} |"
            )
            example_count += 1
            if example_count >= 12:
                break
        if example_count >= 12:
            break
    if example_count == 0:
        lines.append("| - | - | - | no retained structural confusion | - | - |")

    lines.extend(
        [
            "",
            "## Controls and interpretation",
            "",
            "- `posthoc_inference_performed=false`; no model or media was touched by this command.",
            "- `baseline_report_mutated=false`; source metric blocks are preserved verbatim.",
            "- `heldout_100_opened_for_fusion=false`; no held-out cohort was read.",
            "- Ontology, Mapper, production API/UI paths, and training were not modified.",
            "- No SHA/hash/digest was computed.",
            "",
            "### Limitations",
            "",
            "1. The development cohort has 27 rows across five videos; it is not held-out-100.",
            (
                "2. Top-k truncation makes RRF metrics exploratory and potentially "
                "unstable if an action is outside all retained lists."
            ),
            "3. Structural hard negatives are not visibility-grounded negatives.",
            (
                "4. This artifact is not production quality and must not drive model "
                "promotion or training."
            ),
            "",
            f"Artifact version: `{FUSION_VERSION}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / ".agent_tmp" / "wemm_p1_ontology_dev27_normal_repro_20260826.json",
        help="existing WeMM retrieval sidecar (read-only)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / ".agent_tmp" / "wemm_epic_label_prototype_fusion_20260827.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / ".agent_tmp" / "wemm_epic_label_prototype_fusion_20260827.md",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="positive RRF constant")
    parser.add_argument("--top-k", type=int, default=10, help="retained source depth")
    parser.add_argument(
        "--method",
        choices=("auto", "rank", "score"),
        default="auto",
        help="auto/rank computes RRF; score requires complete finite source scores",
    )
    return parser


def _reject_forbidden_input(path: Path) -> None:
    # This helper is intentionally conservative: the task boundary forbids
    # opening the held-out-100 cohort.  A name-level check prevents an
    # accidental CLI invocation against a plainly labelled held-out artifact.
    name = path.name.casefold().replace("-", "_")
    if "heldout_100" in name or "held_out_100" in name or "exact_100" in name:
        raise WemmLabelPrototypeFusionError(
            "held-out-100 input is forbidden for this benchmark-local diagnostic"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.input.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_md = args.output_md.expanduser().resolve()
    _reject_forbidden_input(source)
    if source in {output_json, output_md}:
        raise WemmLabelPrototypeFusionError("output path must differ from read-only source report")
    report = _load_json(source)
    diagnostic = build_diagnostic(
        report,
        source_report=str(source),
        rrf_k=args.rrf_k,
        top_k=args.top_k,
        method=args.method,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(_render_markdown(diagnostic), encoding="utf-8")
    print(
        f"wrote {output_json} and {output_md}; "
        f"version={FUSION_VERSION} score_fusion_all_modes="
        f"{diagnostic['experiment']['score_fusion']['available_all_modes']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        raise SystemExit(main())
    except WemmLabelPrototypeFusionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
