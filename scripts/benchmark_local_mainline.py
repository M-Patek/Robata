"""Measure the local fake-model mainline without making a capacity claim.

Each sample executes ``scripts/run_local_mainline.py`` in a fresh output directory and
records both required throughput units.  The command never selects a real provider and
fails closed if the child process reports provider requests or production eligibility.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Never

from robata.runtime.benchmark import ThroughputSample, summarize_samples

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts" / "run_local_mainline.py"
DEFAULT_MAPPING_CONFIG = REPOSITORY_ROOT / "config" / "genrobot-observed-v0.json"
MANIFEST_FILENAME = "camera-video-export-manifest.json"


class BenchmarkCliError(RuntimeError):
    """A benchmark invocation failure that can be serialized as JSON."""


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise BenchmarkCliError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(
        description=(
            "Run the deterministic local fake-model mainline repeatedly and report "
            "engineering-only throughput evidence."
        )
    )
    parser.add_argument("source", type=Path, help="local MCAP source")
    parser.add_argument("output_root", type=Path, help="new benchmark output root")
    parser.add_argument("--mapping-config", type=Path, default=DEFAULT_MAPPING_CONFIG)
    parser.add_argument("--namespace", default="robata")
    parser.add_argument("--iterations", type=int, default=1, metavar="N")
    parser.add_argument("--warmups", type=int, default=0, metavar="N")
    parser.add_argument("--allow-unapproved", action="store_true")
    parser.add_argument("--no-event", action="store_true")
    parser.add_argument("--parallel-independent-inference", action="store_true")
    parser.add_argument("--parallel-video-export", action="store_true")
    parser.add_argument("--max-video-export-workers", type=int, default=6, metavar="N")
    parser.add_argument("--parallel-frame-materialization", action="store_true")
    parser.add_argument("--max-frame-materialization-workers", type=int, default=6, metavar="N")
    parser.add_argument(
        "--shared-registry",
        action="store_true",
        help="reuse one local registry across iterations (warm-cache evidence)",
    )
    return parser


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    source = _absolute(args.source)
    output_root = _absolute(args.output_root)
    if not source.is_file() or source.is_symlink():
        raise BenchmarkCliError(f"source must be a regular file: {source}")
    if output_root.exists() or output_root.is_symlink():
        raise BenchmarkCliError(f"output root must be absent and not a symlink: {output_root}")
    if args.iterations <= 0:
        raise BenchmarkCliError("iterations must be positive")
    if args.warmups < 0:
        raise BenchmarkCliError("warmups must be nonnegative")
    if args.max_video_export_workers <= 0 or args.max_video_export_workers > 6:
        raise BenchmarkCliError("max-video-export-workers must be between 1 and 6")
    if args.max_frame_materialization_workers <= 0 or args.max_frame_materialization_workers > 6:
        raise BenchmarkCliError("max-frame-materialization-workers must be between 1 and 6")
    return source, output_root


def _recording_duration_ns(output_directory: Path) -> int:
    manifest_path = output_directory / "video" / MANIFEST_FILENAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        durations = [
            int(camera["media_time_mapping"]["last_pts"])
            + int(camera["media_time_mapping"]["last_duration"])
            for camera in payload["cameras"]
        ]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BenchmarkCliError(
            f"cannot derive recording duration from {manifest_path}: {error}"
        ) from error
    if not durations or any(duration <= 0 for duration in durations):
        raise BenchmarkCliError("video manifest contains no positive recording duration")
    return max(durations)


def _child_command(
    *,
    source: Path,
    output: Path,
    mapping_config: Path,
    namespace: str,
    allow_unapproved: bool,
    no_event: bool,
    parallel_independent_inference: bool,
    parallel_video_export: bool,
    max_video_export_workers: int,
    parallel_frame_materialization: bool,
    max_frame_materialization_workers: int,
    registry_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        str(source),
        str(output),
        "--mapping-config",
        str(mapping_config),
        "--namespace",
        namespace,
        "--registry-root",
        str(registry_root),
    ]
    if allow_unapproved:
        command.append("--allow-unapproved")
    if no_event:
        command.append("--no-event")
    if parallel_independent_inference:
        command.append("--parallel-independent-inference")
    if parallel_video_export:
        command.extend(
            ("--parallel-video-export", "--max-video-export-workers", str(max_video_export_workers))
        )
    return command


def _run_one(
    *,
    source: Path,
    output: Path,
    mapping_config: Path,
    namespace: str,
    allow_unapproved: bool,
    no_event: bool,
    parallel_independent_inference: bool,
    parallel_video_export: bool,
    max_video_export_workers: int,
    parallel_frame_materialization: bool,
    max_frame_materialization_workers: int,
    registry_root: Path,
) -> tuple[ThroughputSample, dict[str, Any]]:
    started = time.perf_counter()
    completed = subprocess.run(
        _child_command(
            source=source,
            output=output,
            mapping_config=mapping_config,
            namespace=namespace,
            allow_unapproved=allow_unapproved,
            no_event=no_event,
            parallel_independent_inference=parallel_independent_inference,
            parallel_video_export=parallel_video_export,
            max_video_export_workers=max_video_export_workers,
            parallel_frame_materialization=parallel_frame_materialization,
            max_frame_materialization_workers=max_frame_materialization_workers,
            registry_root=registry_root,
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_ms = max(1, round((time.perf_counter() - started) * 1_000))
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkCliError(f"local mainline failed ({completed.returncode}): {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkCliError(f"local mainline returned invalid JSON: {error}") from error
    if (
        payload.get("provider_requests") != 0
        or payload.get("execution_mode") != "LOCAL_DEVELOPMENT_FAKE_MODEL"
    ):
        raise BenchmarkCliError(
            "benchmark child violated the zero-provider local execution contract"
        )
    sample = ThroughputSample(
        elapsed_ms=elapsed_ms,
        recording_duration_ns=_recording_duration_ns(output),
    )
    return sample, payload


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source, output_root = _validate_args(args)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    registry_root = output_root / "registry" if args.shared_registry else None
    samples: list[ThroughputSample] = []
    runs: list[dict[str, Any]] = []
    total_runs = args.warmups + args.iterations
    for ordinal in range(total_runs):
        run_output = output_root / f"run-{ordinal + 1:03d}"
        run_registry = registry_root or (output_root / f"registry-{ordinal + 1:03d}")
        sample, payload = _run_one(
            source=source,
            output=run_output,
            mapping_config=_absolute(args.mapping_config),
            namespace=args.namespace,
            allow_unapproved=args.allow_unapproved,
            no_event=args.no_event,
            parallel_independent_inference=args.parallel_independent_inference,
            parallel_video_export=args.parallel_video_export,
            max_video_export_workers=args.max_video_export_workers,
            parallel_frame_materialization=args.parallel_frame_materialization,
            max_frame_materialization_workers=args.max_frame_materialization_workers,
            registry_root=run_registry,
        )
        if ordinal >= args.warmups:
            samples.append(sample)
            runs.append(
                {
                    "ordinal": ordinal + 1,
                    "output_directory": str(run_output),
                    "run_id": payload.get("analysis", {}).get("run_id"),
                    "bundle_sha256": payload.get("analysis", {}).get("bundle_sha256"),
                    "event_count": payload.get("event_count"),
                    "sample": sample.as_dict(),
                }
            )
    summary = summarize_samples("local-mainline-fake-model", samples)
    report = {
        "ok": True,
        "execution_mode": "LOCAL_DEVELOPMENT_FAKE_MODEL",
        "provider_requests": 0,
        "production_eligible": False,
        "source": str(source),
        "workload": {
            "workload_id": summary.workload_id,
            "iterations": args.iterations,
            "warmups": args.warmups,
            "cache_mode": "SHARED_REGISTRY" if args.shared_registry else "COLD_REGISTRY_PER_RUN",
            "parallel_independent_inference": args.parallel_independent_inference,
            "parallel_video_export": args.parallel_video_export,
            "max_video_export_workers": args.max_video_export_workers,
            "parallel_frame_materialization": args.parallel_frame_materialization,
            "max_frame_materialization_workers": args.max_frame_materialization_workers,
        },
        "summary": summary.as_dict(),
        "runs": runs,
    }
    (output_root / "benchmark-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        report = run_benchmark(_parser().parse_args(argv))
    except (BenchmarkCliError, OSError, ValueError, TypeError) as error:
        print(
            json.dumps(
                {"ok": False, "provider_requests": 0, "error": str(error)}, indent=2, sort_keys=True
            )
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
