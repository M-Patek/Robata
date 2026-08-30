#!/usr/bin/env python3
"""Assess three-model production-shaped cohort readiness without inference.

The command only reads a cohort manifest and optional review/model-output,
ontology, and mapping JSON files.  It never decodes media or loads WeMM, Qwen,
or Mage.  A zero exit code means the source-bound *invocation* contract is
ready; quality remains ``NOT_MEASURED`` until independent human gold and model
outputs are present.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_readiness import (  # noqa: E402
    ProductionReadinessError,
    assess_production_readiness,
)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ProductionReadinessError(f"could not read JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProductionReadinessError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProductionReadinessError(f"JSON root must be an object: {path}")
    return value


def run(
    *,
    manifest_path: Path,
    review_path: Path | None = None,
    sidecar_path: Path | None = None,
    ontology_path: Path | None = None,
    mapping_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    report = assess_production_readiness(
        _read(manifest_path),
        review_pack=None if review_path is None else _read(review_path),
        sidecar=None if sidecar_path is None else _read(sidecar_path),
        ontology=None if ontology_path is None else _read(ontology_path),
        mapping=None if mapping_path is None else _read(mapping_path),
    )
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            raise ProductionReadinessError(f"could not write report: {output_path}") from exc
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--ontology", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(
            manifest_path=args.manifest,
            review_path=args.review,
            sidecar_path=args.sidecar,
            ontology_path=args.ontology,
            mapping_path=args.mapping,
            output_path=args.output,
        )
    except ProductionReadinessError as exc:
        print(f"production readiness assessment failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "inference_readiness": report["inference_readiness"],
                "quality_measurement_status": report["quality_measurement_status"],
                "blocker_count": len(report["blockers"]),
                "output": None if args.output is None else str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["inference_readiness"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
