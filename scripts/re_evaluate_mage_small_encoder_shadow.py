"""Re-evaluate retained Mage small-encoder shadow reports without rerunning a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from robata.benchmark import mage_small_encoder as evaluator_module
from robata.benchmark.mage_small_encoder import (
    MAX_MATCHED_BOUNDARY_MAE_SECONDS,
    SMALL_ENCODER_EVALUATOR_VERSION,
    aggregate_small_encoder_shadow_run,
    evaluate_small_encoder_pair,
)

SMALL_ENCODER_ANALYSIS_VERSION = "mage-small-encoder-shadow-analysis-v3"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_report_payload(
    report: dict[str, Any], *, source_report_file_sha256: str | None = None
) -> dict[str, object]:
    """Apply the current evaluator to a retained report payload."""

    embedded = report.get("report_sha256")
    if not isinstance(embedded, str):
        raise ValueError("report is missing embedded report_sha256")
    report_without_hash = dict(report)
    report_without_hash.pop("report_sha256", None)
    embedded_valid = _canonical_sha256(report_without_hash) == embedded
    segments = report.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("report must contain a non-empty segments list")
    evaluations = tuple(
        evaluate_small_encoder_pair(
            native_output_text=str(segment["native"]["output_text"]),
            candidate_output_text=str(segment["small_encoder_shadow"]["output_text"]),
        )
        for segment in segments
    )
    summary = aggregate_small_encoder_shadow_run(
        evaluations=evaluations,
        native_generation_seconds=sum(
            float(segment["native"]["generation_seconds"]) for segment in segments
        ),
        candidate_generation_seconds=sum(
            float(segment["small_encoder_shadow"]["generation_seconds"]) for segment in segments
        ),
        candidate_preparation_seconds=sum(
            float(segment["small_encoder_shadow"].get("telemetry", {}).get("total_seconds", 0.0))
            for segment in segments
        ),
    )
    qualification = summary.as_projection()
    qualification_gates = dict(summary.gates)
    qualification_gates["source_report_identity_valid"] = embedded_valid
    qualification["gates"] = qualification_gates
    qualification["qualified"] = summary.qualified and embedded_valid
    evaluator_path = Path(evaluator_module.__file__).resolve()
    result: dict[str, object] = {
        "analysis_version": SMALL_ENCODER_ANALYSIS_VERSION,
        "evaluator": {
            "version": SMALL_ENCODER_EVALUATOR_VERSION,
            "source_sha256": _sha256_file(evaluator_path),
            "max_matched_boundary_mae_seconds": MAX_MATCHED_BOUNDARY_MAE_SECONDS,
        },
        "analysis_implementation": {
            "source_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "source_report_embedded_sha256": embedded,
        "source_report_embedded_sha256_valid": embedded_valid,
        "source_report_file_sha256": source_report_file_sha256,
        "candidate_policy_semantic_sha256": report.get("candidate", {}).get(
            "policy_semantic_sha256"
        ),
        "segment_evaluations": [evaluation.as_projection() for evaluation in evaluations],
        "qualification": qualification,
        "verdict": (
            "INVALID_SOURCE_REPORT_IDENTITY"
            if not embedded_valid
            else (
                "SHADOW_QUALIFIED_FOR_NEXT_CANARY_ONLY"
                if summary.qualified
                else "REJECTED_SHADOW_KEEP_MAGE_NATIVE_AUTHORITY"
            )
        ),
    }
    result["analysis_sha256"] = _canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report_path = arguments.report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = evaluate_report_payload(report, source_report_file_sha256=_sha256_file(report_path))
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
