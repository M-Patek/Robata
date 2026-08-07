"""Run one real local MCAP through one real local Hugging Face vision model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.local_real_model_e2e import (  # noqa: E402
    LocalRealModelE2EError,
    run_local_real_model_e2e,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="real local MCAP source")
    parser.add_argument("--mapping-config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=REPOSITORY_ROOT / ".local" / "real-model-e2e",
        help="local filesystem object store, SQLite, report, trace, and cache root",
    )
    parser.add_argument(
        "--allow-unapproved-profile",
        action="store_true",
        help="explicitly allow a development UNAPPROVED mapping profile",
    )
    parser.add_argument("--model-identifier", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--model-version", default="local")
    parser.add_argument("--max-image-side", type=_positive_int, default=448)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=64)
    parser.add_argument("--gpu-weight-memory-gib", type=_positive_int, default=4)
    parser.add_argument("--cpu-weight-memory-gib", type=_positive_int, default=1)
    parser.add_argument(
        "--endpoint-url",
        help="optional loopback model endpoint base URL, for example http://127.0.0.1:8101",
    )
    parser.add_argument(
        "--endpoint-timeout-seconds",
        type=_positive_float,
        default=300.0,
    )
    parser.add_argument("--prompt", help="optional explicit six-camera prompt")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional additional report copy; the run directory always receives report.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, report_path, trace_path, participation_path = run_local_real_model_e2e(
            source_path=args.source,
            mapping_config=args.mapping_config,
            model_directory=args.model_dir,
            state_directory=args.state_dir,
            allow_unapproved_profile=args.allow_unapproved_profile,
            model_identifier=args.model_identifier,
            model_version=args.model_version,
            max_image_side=args.max_image_side,
            max_new_tokens=args.max_new_tokens,
            gpu_weight_memory_gib=args.gpu_weight_memory_gib,
            cpu_weight_memory_gib=args.cpu_weight_memory_gib,
            endpoint_url=args.endpoint_url,
            endpoint_timeout_seconds=args.endpoint_timeout_seconds,
            prompt=args.prompt,
        )
        payload = (
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
    except (LocalRealModelE2EError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            canonical_json_bytes(
                {
                    "ok": False,
                    "code": "LOCAL_REAL_MODEL_E2E_FAILED",
                    "detail": str(error),
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "run_id": report.run_id,
                "report_path": str(report_path),
                "trace_path": str(trace_path),
                "participation_path": str(participation_path),
                "participation_coverage": report.participation_coverage.value,
                "sqlite_path": report.storage.sqlite_path,
                "model_output": report.model.output_text,
                "model_transport": report.model.model_transport,
                "production_eligible": report.production_eligible,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
