#!/usr/bin/env python3
"""Check the named WeMM EPIC baseline coverage without running inference.

The EPIC WeMM baseline is intentionally a retrieval-only development screen.
This read-only checker makes the coverage contract explicit: all label-text
variants and retrieval modes must have Recall@K/MRR blocks, representative
nearest-neighbour cases must expose each mode, and the Mapper/resolver boundary
must remain fail-closed.  It reads an existing JSON sidecar only; it never
loads a model, decodes media, opens held-out data, or computes a digest.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

EXPECTED_VARIANTS = ("canonical", "verb_noun", "natural")
EXPECTED_MODES = ("visual", "text", "hybrid")
EXPECTED_KS = ("1", "3", "5", "10")
BOUNDARY_VERSION = "wemm-epic-baseline-completeness-v1"


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


def _finite_unit(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _metric_blocks(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the compact audit metric blocks or the raw runner blocks."""

    compact = _mapping(report.get("normal_metrics_27_ontology"))
    if compact is not None:
        return compact
    results = _mapping(report.get("results"))
    if results is None:
        return {}
    out: dict[str, Any] = {}
    for variant, result_value in results.items():
        result = _mapping(result_value)
        metrics = _mapping(result.get("metrics")) if result is not None else None
        out[variant] = _mapping(metrics.get("metrics")) if metrics is not None else {}
    return out


def _check_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    blocks = _metric_blocks(report)
    missing: list[str] = []
    invalid: list[str] = []
    for variant in EXPECTED_VARIANTS:
        variant_blocks = _mapping(blocks.get(variant))
        if variant_blocks is None:
            missing.extend(f"{variant}/{mode}" for mode in EXPECTED_MODES)
            continue
        for mode in EXPECTED_MODES:
            block = _mapping(variant_blocks.get(mode))
            if block is None:
                missing.append(f"{variant}/{mode}")
                continue
            recall = _mapping(block.get("recall_at_k"))
            if recall is None:
                # Compact audit reports use recall_at_1 ... recall_at_10.
                values = {k: block.get(f"recall_at_{k}") for k in EXPECTED_KS}
            else:
                values = {k: recall.get(k) for k in EXPECTED_KS}
            for key, value in values.items():
                if not _finite_unit(value):
                    invalid.append(f"{variant}/{mode}/Recall@{key}")
            if not _finite_unit(block.get("mrr")):
                invalid.append(f"{variant}/{mode}/MRR")
    return {
        "expected_variants": list(EXPECTED_VARIANTS),
        "expected_modes": list(EXPECTED_MODES),
        "required_recall_k": [f"Recall@{k}" for k in EXPECTED_KS],
        "block_count": len(EXPECTED_VARIANTS) * len(EXPECTED_MODES) - len(missing),
        "missing_blocks": missing,
        "invalid_values": invalid,
        "passed": not missing and not invalid,
    }


def _check_hard_negatives(report: Mapping[str, Any]) -> dict[str, Any]:
    raw = _sequence(report.get("nearest_neighbor_cases_canonical"))
    if raw is None:
        # A raw runner sidecar has rankings but not the curated examples.  The
        # checker deliberately reports that as a documentation gap rather than
        # manufacturing hard negatives from model output.
        return {
            "case_count": 0,
            "cases_with_all_modes": 0,
            "cases_with_hard_negative": 0,
            "source": "not_declared",
            "passed": False,
            "reason": "nearest_neighbor_cases_canonical is missing",
        }
    complete = 0
    hard_negative = 0
    malformed: list[int] = []
    for index, value in enumerate(raw):
        case = _mapping(value)
        modes = _mapping(case.get("modes")) if case is not None else None
        mode_rows: dict[str, Mapping[str, Any]] = {}
        if modes is not None:
            for mode in EXPECTED_MODES:
                mode_row = _mapping(modes.get(mode))
                top5 = _sequence(mode_row.get("top5")) if mode_row is not None else None
                if mode_row is None or not top5:
                    mode_rows = {}
                    break
                mode_rows[mode] = mode_row
        if case is None or modes is None or len(mode_rows) != len(EXPECTED_MODES):
            malformed.append(index)
        else:
            complete += 1
            # A curated case is a useful hard-negative example when at least
            # one mode either misses the target in its top ten or places a
            # different candidate at rank one.  Do not infer correctness from
            # the prose label; use only the retained rank fields.
            truth = case.get("ground_truth")
            has_hard_negative = False
            for mode in EXPECTED_MODES:
                mode_row = mode_rows[mode]
                if mode_row.get("target_rank_within_top10") is None:
                    has_hard_negative = True
                    break
                top5 = _sequence(mode_row.get("top5"))
                first = _mapping(top5[0]) if top5 else None
                if first is not None and first.get("action_key") != truth:
                    has_hard_negative = True
                    break
            if has_hard_negative:
                hard_negative += 1
    return {
        "case_count": len(raw),
        "cases_with_all_modes": complete,
        "cases_with_hard_negative": hard_negative,
        "malformed_case_indices": malformed,
        "source": "curated_nearest_neighbor_cases_canonical",
        "passed": bool(raw) and not malformed,
    }


def _check_boundaries(report: Mapping[str, Any]) -> dict[str, Any]:
    mapper = _mapping(report.get("mapper_boundary"))
    mapper_passed = bool(
        mapper is not None
        and mapper.get("retrieval_only") is True
        and mapper.get("existing_mapper_invoked") is False
    )

    # The baseline has no production sidecar and therefore must not invoke the
    # post-hoc resolver.  The explicit controls are recorded by this checker
    # even for older audit sidecars that predate a resolver_boundary field.
    scope = _mapping(report.get("scope")) or {}
    controls = _mapping(report.get("controls")) or {}
    resolver_declared = _mapping(report.get("resolver_boundary"))
    resolver_invoked = (
        resolver_declared.get("resolver_invoked")
        if resolver_declared is not None
        else controls.get(
            "resolver_invoked",
            controls.get("posthoc_resolver_invoked", False),
        )
    )
    production_paths_touched = scope.get("production_paths_touched")
    if production_paths_touched is None:
        production_paths_touched = controls.get("production_path_changed")
    quality_claim = bool(
        report.get("production_eligible") is True
        or controls.get("quality_claim") is True
        or (resolver_declared is not None and resolver_declared.get("quality_claim") is True)
    )
    resolver_passed = (
        resolver_invoked is False and production_paths_touched is False and not quality_claim
    )
    return {
        "mapper": {
            "retrieval_only": mapper.get("retrieval_only") if mapper is not None else None,
            "existing_mapper_invoked": (
                mapper.get("existing_mapper_invoked") if mapper is not None else None
            ),
            "passed": mapper_passed,
        },
        "resolver": {
            "resolver_invoked": resolver_invoked,
            "source_artifact_declared": resolver_declared is not None,
            "production_paths_touched": production_paths_touched,
            "quality_claim": quality_claim,
            "passed": resolver_passed,
            "note": "No post-hoc or production resolver is part of the EPIC retrieval baseline.",
        },
        "passed": mapper_passed and resolver_passed,
    }


def inspect(report: Mapping[str, Any], *, source: str | None = None) -> dict[str, Any]:
    """Build a JSON-native completeness result from one existing sidecar."""

    metrics = _check_metrics(report)
    hard_negatives = _check_hard_negatives(report)
    boundaries = _check_boundaries(report)
    controls = {
        "model_invoked_for_check": False,
        "media_decoded_for_check": False,
        "heldout_100_opened": False,
        "hash_or_sha_used": False,
        "production_path_changed": False,
    }
    return {
        "report_version": BOUNDARY_VERSION,
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source_artifact": source,
        "inference_performed_for_check": False,
        "coverage": {
            "label_variants": list(EXPECTED_VARIANTS),
            "retrieval_modes": list(EXPECTED_MODES),
            "metrics": metrics,
            "hard_negatives": hard_negatives,
            "boundaries": boundaries,
        },
        "complete": bool(metrics["passed"] and hard_negatives["passed"] and boundaries["passed"]),
        "controls": controls,
        "notes": [
            (
                "This is a read-only contract check over retained evidence; it is not a "
                "model evaluation."
            ),
            "Mapper-shaped projections remain compatibility diagnostics, not Mapper accuracy.",
            (
                "Resolver status is explicitly fail-closed: no post-hoc/production "
                "resolver was invoked."
            ),
        ],
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return cast(dict[str, Any], value)


def markdown(result: Mapping[str, Any]) -> str:
    coverage = _mapping(result["coverage"]) or {}
    metrics = _mapping(coverage.get("metrics")) or {}
    hard = _mapping(coverage.get("hard_negatives")) or {}
    boundaries = _mapping(coverage.get("boundaries")) or {}
    mapper = _mapping(boundaries.get("mapper")) or {}
    resolver = _mapping(boundaries.get("resolver")) or {}
    lines = [
        "# WeMM EPIC baseline completeness check",
        "",
        "Read-only check over retained benchmark evidence; no model/media/hash/heldout access.",
        "",
        f"- Overall: **{'COMPLETE' if result.get('complete') else 'INCOMPLETE'}**",
        f"- Variants: `{', '.join(str(x) for x in coverage.get('label_variants', ()))}`",
        f"- Modes: `{', '.join(str(x) for x in coverage.get('retrieval_modes', ()))}`",
        f"- Metric blocks: `{metrics.get('block_count', 0)}`; passed=`{metrics.get('passed')}`",
        f"- Hard-negative cases: `{hard.get('case_count', 0)}`; passed=`{hard.get('passed')}`",
        "",
        "## Boundaries",
        "",
        (
            f"- Mapper retrieval-only: `{mapper.get('retrieval_only')}`; existing "
            f"Mapper invoked: `{mapper.get('existing_mapper_invoked')}`."
        ),
        (
            f"- Resolver invoked: `{resolver.get('resolver_invoked')}`; production "
            f"quality claim: `{resolver.get('quality_claim')}`."
        ),
        f"- Boundary check passed: `{boundaries.get('passed')}`.",
        "",
        (
            "The projections are deterministic compatibility outputs. They must not be "
            "read as Mapper or resolver accuracy."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    result = inspect(_load(args.report), source=str(args.report.resolve()))
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"complete": result["complete"], "source": str(args.report)}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
