"""Launch one fail-closed Robata production real-sample attempt.

The command accepts a reviewed bootstrap and either a locally pinned MCAP or a
canonical R2 source manifest.  Without an explicit source-specific execution
driver it still performs bootstrap/runtime/source preparation, emits the audit
bundle, and exits non-zero rather than pretending that canonical completion
happened.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.application.canonical.production_real_sample_worker import (  # noqa: E402
    ProductionRealSampleWorkerConfig,
    run_production_real_sample,
)
from robata.runtime.e2e_participation import E2EParticipationDeclaration  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="bootstrap_config_path", type=Path, required=True)
    parser.add_argument("--mapping-config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", dest="source_path", type=Path)
    source.add_argument("--source-manifest", dest="source_manifest_path", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--source-byte-count", type=int)
    parser.add_argument("--source-media-type", default="application/x-mcap")
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-duration-ns", type=int)
    parser.add_argument("--allow-unapproved-profile", action="store_true")
    parser.add_argument(
        "--participation-plan",
        type=Path,
        help=(
            "optional JSON array declaring all seven E2E boundaries as "
            "PARTICIPATING/BYPASSED/NOT_CONFIGURED/FAILED"
        ),
    )
    return parser


def _load_participation_plan(
    path: Path | None,
) -> tuple[E2EParticipationDeclaration, ...]:
    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("participation plan must be a JSON array")
    return tuple(E2EParticipationDeclaration.model_validate(item, strict=True) for item in payload)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = ProductionRealSampleWorkerConfig(
            bootstrap_config_path=str(arguments.bootstrap_config_path),
            mapping_config_path=str(arguments.mapping_config),
            output_directory=str(arguments.output_directory),
            state_directory=str(arguments.state_directory),
            source_path=None if arguments.source_path is None else str(arguments.source_path),
            source_manifest_path=(
                None
                if arguments.source_manifest_path is None
                else str(arguments.source_manifest_path)
            ),
            source_sha256=arguments.source_sha256,
            source_byte_count=arguments.source_byte_count,
            source_media_type=arguments.source_media_type,
            allow_unapproved_profile=arguments.allow_unapproved_profile,
            max_duration_ns=arguments.max_duration_ns,
            run_id=arguments.run_id,
            e2e_participation=_load_participation_plan(arguments.participation_plan),
        )
        result = run_production_real_sample(config, environment=os.environ)
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": type(error).__name__,
                    "detail": str(error),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2

    print(
        json.dumps(
            {
                "ok": result.report.status == "SUCCEEDED",
                "run_id": result.report.run_id,
                "status": result.report.status,
                "failure_code": result.report.failure_code,
                "failure_detail": result.report.failure_detail,
                "report_path": str(result.report_path),
                "trace_path": str(result.trace_path),
                "participation_path": str(result.participation_path),
                "component_participation_path": str(result.component_participation_path),
                "participation_coverage": result.report.participation_coverage.value,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if result.report.status == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
