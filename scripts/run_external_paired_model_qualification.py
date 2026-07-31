"""Run one bounded, shadow-only external Qwen/Mage RunPod observation for P20."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.external_paired_qualification import (  # noqa: E402
    ExternalPairedQualificationError,
    TransportFactory,
    load_external_qualification_environment,
    redact_external_qualification_detail,
    run_external_paired_qualification,
    write_external_paired_qualification_report,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402
from robata.runtime.e2e_trace import (  # noqa: E402
    run_external_paired_qualification_with_trace,
    write_external_paired_e2e_trace,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        required=True,
        metavar="PATH",
        help="explicit RunPod control/candidate configuration and credential mapping",
    )
    parser.add_argument(
        "--capabilities-dir",
        type=Path,
        metavar="DIR",
        help="directory containing control.json and candidate.json capability snapshots",
    )
    parser.add_argument(
        "--control-capabilities",
        type=Path,
        metavar="PATH",
        help="full control ModelCapabilities JSON snapshot",
    )
    parser.add_argument(
        "--candidate-capabilities",
        type=Path,
        metavar="PATH",
        help="full candidate ModelCapabilities JSON snapshot",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        required=True,
        metavar="PATH",
        help="fully formed robata-external-paired-workload-v1 JSON manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="non-secret local JSON observation report path",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        metavar="PATH",
        help="optional noncanonical robata-e2e-trace-v1 sidecar path",
    )
    parser.add_argument(
        "--trace-id",
        metavar="UUID",
        help="optional UUID for the sidecar trace; generated when omitted",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        metavar="DIR",
        help="optional local directory for separate durable control/candidate SQLite evidence",
    )
    parser.add_argument(
        "--max-attempts-per-endpoint",
        type=int,
        default=1,
        metavar="N",
        help="bounded RunPod retries per side (1-5; defaults to one observation)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    transport_factory: TransportFactory | None = None,
) -> int:
    """Launch the observation and persist evidence even when a provider terminal fails.

    A provider failure is represented inside the returned report rather than
    converted into a fictional qualification pass. Configuration, manifest,
    and local persistence failures return a nonzero exit status.
    """

    args = _parser().parse_args(argv)
    base_values = os.environ if environment is None else environment
    values: Mapping[str, str] = base_values
    try:
        values = load_external_qualification_environment(base_values, args.env_file)
        control_capabilities, candidate_capabilities = _capability_paths(args)
        input_paths = (args.workload, control_capabilities, candidate_capabilities)
        _reject_output_input_collision(
            output=args.output,
            inputs=input_paths,
        )
        if args.trace_id is not None and args.trace_output is None:
            raise ExternalPairedQualificationError("--trace-id requires --trace-output")
        trace = None
        if args.trace_output is None:
            report = asyncio.run(
                run_external_paired_qualification(
                    environment=values,
                    control_capabilities_path=control_capabilities,
                    candidate_capabilities_path=candidate_capabilities,
                    workload_path=args.workload,
                    max_attempts_per_endpoint=args.max_attempts_per_endpoint,
                    evidence_directory=args.evidence_dir,
                    transport_factory=transport_factory,
                )
            )
        else:
            _reject_output_input_collision(
                output=args.trace_output,
                inputs=(*input_paths, args.output),
            )
            trace_execution = asyncio.run(
                run_external_paired_qualification_with_trace(
                    environment=values,
                    control_capabilities_path=control_capabilities,
                    candidate_capabilities_path=candidate_capabilities,
                    workload_path=args.workload,
                    trace_id=args.trace_id,
                    max_attempts_per_endpoint=args.max_attempts_per_endpoint,
                    evidence_directory=args.evidence_dir,
                    transport_factory=transport_factory,
                )
            )
            report = trace_execution.report
            trace = trace_execution.trace
        write_external_paired_qualification_report(report, args.output)
        if trace is not None:
            write_external_paired_e2e_trace(trace, args.trace_output)
        print(canonical_json_bytes(report.model_dump(mode="json")).decode("utf-8"))
        return 0
    except (
        ExternalPairedQualificationError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            canonical_json_bytes(
                {
                    "ok": False,
                    "code": "EXTERNAL_PAIRED_QUALIFICATION_FAILED",
                    "detail": redact_external_qualification_detail(str(error), values),
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2


def _capability_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    directory = args.capabilities_dir
    control = args.control_capabilities
    candidate = args.candidate_capabilities
    if directory is not None:
        if control is not None or candidate is not None:
            raise ExternalPairedQualificationError(
                "--capabilities-dir cannot be combined with explicit capability paths"
            )
        return directory / "control.json", directory / "candidate.json"
    if control is None or candidate is None:
        raise ExternalPairedQualificationError(
            "provide --capabilities-dir or both --control-capabilities and --candidate-capabilities"
        )
    return control, candidate


def _reject_output_input_collision(*, output: Path, inputs: tuple[Path, ...]) -> None:
    try:
        resolved_output = output.resolve(strict=False)
        resolved_inputs = tuple(item.resolve(strict=False) for item in inputs)
    except OSError as error:
        raise ExternalPairedQualificationError(
            "cannot resolve qualification input/output paths"
        ) from error
    if resolved_output in resolved_inputs:
        raise ExternalPairedQualificationError("--output must not overwrite a qualification input")


if __name__ == "__main__":
    raise SystemExit(main())
