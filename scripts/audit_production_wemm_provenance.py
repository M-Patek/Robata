#!/usr/bin/env python3
"""Classify recorded WeMM artifacts without opening media or computing identities.

The repository has two intentionally separate WeMM routes:

* ``robata-production-wemm-vocabulary-shadow-v1`` uses the owner/Terra
  production vocabulary and is eligible for the production *review* queue.
* ``robata-production-wemm-shadow-v1`` uses the EPIC/provisional pair catalog
  and is an EPIC/legacy diagnostic only.

This command makes that boundary visible in one small JSON/Markdown report. It
does not invoke a model, decode media, read gold, modify Mapper/ontology, or
compute a hash/digest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEDICATED_FORMAT = "robata-production-wemm-vocabulary-shadow-v1"
LEGACY_FORMAT = "robata-production-wemm-shadow-v1"
COMPARISON_FORMAT = "robata-production-wemm-vocabulary-variant-comparison-v1"
OPEN_PREANNOTATION_FORMAT = "robata-production-wemm-preannotation-v1"
PRODUCTION_PROFILE = "PRODUCTION_OWNER_APPROVED_COARSE_VOCABULARY"
OPEN_PRODUCTION_CLASSIFICATION = "PRODUCTION_OPEN_PREANNOTATION"


def _load(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def classify(path: Path, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a metadata-only classification for one artifact."""

    if payload is None:
        return {"path": str(path), "classification": "UNREADABLE", "reasons": ["INVALID_JSON"]}
    fmt = str(payload.get("format") or "")
    reasons: list[str] = []
    if fmt == OPEN_PREANNOTATION_FORMAT:
        """Recognize the current open, review-only production envelope."""

        label_space = payload.get("label_space")
        label_space = label_space if isinstance(label_space, Mapping) else {}
        raw_model_output = payload.get("raw_model_output")
        raw_model_output = raw_model_output if isinstance(raw_model_output, Mapping) else {}
        catalog = raw_model_output.get("catalog")
        catalog = catalog if isinstance(catalog, Mapping) else {}
        if label_space.get("kind") != "OPEN_PROVISIONAL_PHRASES":
            reasons.append("UNEXPECTED_OPEN_LABEL_SPACE")
        if label_space.get("epic_ontology_used") is not False:
            reasons.append("EPIC_ONTOLOGY_NOT_FALSE")
        if label_space.get("mapper_used") is not False:
            reasons.append("MAPPER_USED_NOT_FALSE")
        if catalog.get("epic_ontology_used") is not False:
            reasons.append("CATALOG_EPIC_ONTOLOGY_NOT_FALSE")
        if catalog.get("mapper_used") is not False:
            reasons.append("CATALOG_MAPPER_USED_NOT_FALSE")
        if payload.get("production_eligible") is not False:
            reasons.append("OPEN_PREANNOTATION_MUST_REMAIN_REVIEW_ONLY")
        source = payload.get("source") if isinstance(payload.get("source"), Mapping) else {}
        model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
        return {
            "path": str(path),
            "format": fmt,
            "classification": OPEN_PRODUCTION_CLASSIFICATION if not reasons else "QUARANTINED",
            "reasons": reasons,
            "epic_ontology_used": label_space.get("epic_ontology_used"),
            "mapper_used": label_space.get("mapper_used"),
            "label_space_kind": label_space.get("kind"),
            "catalog_format": catalog.get("format"),
            "phrase_count": catalog.get("phrase_count"),
            "label_variant": model.get("label_variant"),
            "window_count": source.get("window_count", len(payload.get("windows", []))),
            "camera_count": source.get("camera_count"),
            "production_eligible": payload.get("production_eligible"),
        }
    if fmt == DEDICATED_FORMAT:
        vocabulary = payload.get("vocabulary")
        if not isinstance(vocabulary, Mapping):
            return {
                "path": str(path),
                "format": fmt,
                "classification": "QUARANTINED",
                "reasons": ["MISSING_VOCABULARY_PROVENANCE"],
            }
        epic = _bool(vocabulary.get("epic_ontology_used"))
        mapper = _bool(vocabulary.get("mapper_used"))
        profile = str(vocabulary.get("profile") or "")
        if epic is not False:
            reasons.append("EPIC_ONTOLOGY_NOT_FALSE")
        if mapper is not False:
            reasons.append("MAPPER_USED_NOT_FALSE")
        if profile != PRODUCTION_PROFILE:
            reasons.append("UNEXPECTED_PRODUCTION_PROFILE")
        classification = "PRODUCTION_TERRA_VOCABULARY" if not reasons else "QUARANTINED"
        model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
        source = payload.get("source") if isinstance(payload.get("source"), Mapping) else {}
        return {
            "path": str(path),
            "format": fmt,
            "classification": classification,
            "reasons": reasons,
            "epic_ontology_used": epic,
            "mapper_used": mapper,
            "vocabulary_profile": profile,
            "label_variant": model.get("label_variant"),
            "window_count": source.get("window_count", len(payload.get("windows", []))),
            "camera_count": source.get("camera_count"),
            "production_eligible": payload.get("production_eligible"),
        }
    if fmt == LEGACY_FORMAT:
        ontology = payload.get("ontology")
        profile = ontology.get("profile") if isinstance(ontology, Mapping) else None
        ontology_format = (
            str(ontology.get("format") or "").casefold() if isinstance(ontology, Mapping) else ""
        )
        ontology_source = (
            str(ontology.get("source") or "").casefold() if isinstance(ontology, Mapping) else ""
        )
        production_provisional = "production" in ontology_format or "production" in ontology_source
        epic = not production_provisional
        reasons.append("EPIC_LABEL_SPACE" if epic else "LEGACY_PRODUCTION_VOCABULARY_SHAPE")
        if isinstance(ontology, Mapping) and ontology.get("source"):
            reasons.append(str(ontology.get("source")))
        return {
            "path": str(path),
            "format": fmt,
            "classification": (
                "EPIC_DERIVED_QUARANTINED" if epic else "LEGACY_PRODUCTION_VOCABULARY_QUARANTINED"
            ),
            "reasons": reasons,
            "ontology_profile": profile,
            "epic_ontology_used": epic,
            "mapper_used": False,
            "production_eligible": payload.get("production_eligible"),
        }
    if fmt == COMPARISON_FORMAT:
        routes = payload.get("routes")
        route_count = len(routes) if isinstance(routes, Mapping) else 0
        provenance_ok = True
        for route in routes.values() if isinstance(routes, Mapping) else ():
            provenance = route.get("provenance") if isinstance(route, Mapping) else None
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("epic_ontology_used") is not False
                or provenance.get("mapper_used") is not False
            ):
                provenance_ok = False
        if not provenance_ok:
            reasons.append("ROUTE_PROVENANCE_NOT_EXPLICITLY_PRODUCTION_ONLY")
        return {
            "path": str(path),
            "format": fmt,
            "classification": "PRODUCTION_BASELINE_REPORT" if provenance_ok else "QUARANTINED",
            "reasons": reasons,
            "route_count": route_count,
            "reference_status": (payload.get("reference") or {}).get("status")
            if isinstance(payload.get("reference"), Mapping)
            else None,
            "production_eligible": payload.get("production_eligible"),
        }
    # EPIC benchmark artifacts are useful, but never production candidates.
    if "epic" in path.name.casefold() or "epic" in fmt.casefold():
        return {
            "path": str(path),
            "format": fmt,
            "classification": "EPIC_BENCHMARK",
            "reasons": ["EPIC_BENCHMARK_ONLY"],
            "production_eligible": payload.get("production_eligible"),
        }
    return {
        "path": str(path),
        "format": fmt,
        "classification": "OTHER_NONPRODUCTION",
        "reasons": ["NOT_A_PRODUCTION_VOCABULARY_SIDECAR"],
        "production_eligible": payload.get("production_eligible"),
    }


def audit(
    root: Path,
    *,
    pattern: str = "*wemm*.json",
    exclude: set[Path] | None = None,
) -> dict[str, Any]:
    excluded = {path.resolve() for path in (exclude or set())}
    rows = [
        classify(path, _load(path))
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.resolve() not in excluded
    ]
    counts = Counter(str(row["classification"]) for row in rows)
    return {
        "format": "robata-production-wemm-provenance-audit-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "DIAGNOSTIC_ONLY",
        "quality_claim": False,
        "official_quality_status": "NOT_MEASURED",
        "production_eligible": False,
        "root": str(root),
        "artifact_count": len(rows),
        "classification_counts": dict(sorted(counts.items())),
        "production_review_inputs": [
            row
            for row in rows
            if row["classification"]
            in {"PRODUCTION_TERRA_VOCABULARY", OPEN_PRODUCTION_CLASSIFICATION}
        ],
        "quarantined_inputs": [
            row
            for row in rows
            if row["classification"]
            in {
                "EPIC_DERIVED_QUARANTINED",
                "LEGACY_PRODUCTION_VOCABULARY_QUARANTINED",
                "LEGACY_QUARANTINED",
                "QUARANTINED",
            }
        ],
        "artifacts": rows,
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
        "routing_rule": (
            "Only explicit production-vocabulary or open review-only preannotation sidecars "
            "may enter production review; "
            "EPIC_DERIVED_QUARANTINED artifacts remain benchmark/legacy diagnostics."
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Production WeMM provenance audit",
        "",
        "> **DIAGNOSTIC_ONLY / NOT_MEASURED.** EPIC-derived sidecars are quarantined.",
        "",
        f"- Artifacts scanned: `{report.get('artifact_count', 0)}`",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    counts = report.get("classification_counts", {})
    if isinstance(counts, Mapping):
        for key, value in counts.items():
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Production review inputs", ""])
    for row in report.get("production_review_inputs", ()):
        if isinstance(row, Mapping):
            lines.append(f"- `{row.get('path')}` ({row.get('label_variant') or 'comparison'})")
    lines.extend(["", "## Quarantined inputs", ""])
    for row in report.get("quarantined_inputs", ()):
        if isinstance(row, Mapping):
            reasons = ", ".join(str(item) for item in row.get("reasons", ()))
            reasons = reasons or "provenance failure"
            lines.append(f"- `{row.get('path')}`: {reasons}")
    lines.extend(
        [
            "",
            "Only explicit Terra/owner production-vocabulary sidecars may feed "
            "the production review route.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".agent_tmp"))
    parser.add_argument("--pattern", default="*wemm*.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    report = audit(
        args.root,
        pattern=args.pattern,
        exclude={args.output_json, output_md},
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "artifact_count": report["artifact_count"],
                "classification_counts": report["classification_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
