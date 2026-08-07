"""Build a non-production external paired workload from frozen local evidence."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.external_paired_workload_builder import (  # noqa: E402
    ExternalPairedWorkloadBuilderError,
    build_external_paired_workload,
    write_external_paired_workload,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        metavar="PATH",
        help="frozen local-real-model-e2e-v1 report.json",
    )
    parser.add_argument(
        "--control-target",
        type=Path,
        required=True,
        metavar="PATH",
        help="explicit control target policy/input-plan config",
    )
    parser.add_argument(
        "--candidate-target",
        type=Path,
        required=True,
        metavar="PATH",
        help="explicit candidate target policy/input-plan config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="output robata-external-paired-workload-v1 manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _reject_output_input_collision(
            output=args.output,
            inputs=(args.report, args.control_target, args.candidate_target),
        )
        result = build_external_paired_workload(
            report_path=args.report,
            control_target_path=args.control_target,
            candidate_target_path=args.candidate_target,
        )
        workload_sha = write_external_paired_workload(result.workload, args.output)
        if workload_sha != result.workload_sha256:
            raise ExternalPairedWorkloadBuilderError(
                "workload digest changed between build and write"
            )
        print(
            canonical_json_bytes(
                {
                    "ok": True,
                    "format_version": result.workload.format_version,
                    "output": str(args.output.resolve()),
                    "workload_sha256": workload_sha,
                    "source_report_sha256": result.source_report_sha256,
                    "input_identity_sha256": result.input_identity_sha256,
                    "camera_count": len(result.camera_artifact_sha256),
                    "production_eligible": False,
                    "selection_eligible": False,
                }
            ).decode("utf-8")
        )
        return 0
    except (ExternalPairedWorkloadBuilderError, OSError, TypeError, ValueError) as error:
        print(
            canonical_json_bytes(
                {
                    "ok": False,
                    "code": "EXTERNAL_PAIRED_WORKLOAD_BUILD_FAILED",
                    "detail": str(error),
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2


def _reject_output_input_collision(*, output: Path, inputs: tuple[Path, ...]) -> None:
    try:
        resolved_output = output.resolve(strict=False)
        resolved_inputs = tuple(item.resolve(strict=False) for item in inputs)
    except OSError as error:
        raise ExternalPairedWorkloadBuilderError(
            "cannot resolve workload input/output paths"
        ) from error
    if resolved_output in resolved_inputs:
        raise ExternalPairedWorkloadBuilderError("--output must not overwrite a workload input")


if __name__ == "__main__":
    raise SystemExit(main())
