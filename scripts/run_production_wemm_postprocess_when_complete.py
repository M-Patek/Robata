#!/usr/bin/env python3
"""Wait for the two production WeMM checkpoints, then finalize read-only artifacts.

This is an orchestration helper for the already-running WeMM batch.  It polls
checkpoint JSON only; it never starts/stops a model runner, opens MCAP/media,
reads gold, or computes hashes/digests.  Once B1 and the remaining checkpoint
are both COMPLETE it executes, exactly once, the independent review-pack
rebuild, aggregate, ambiguity-selection, and structural-verification steps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B1 = ROOT / ".agent_tmp" / "production_wemm_b1_stateful_chunk4_20260828" / "batch-run.json"
DEFAULT_REMAINING = (
    ROOT / ".agent_tmp" / "production_wemm_remaining_stateful_chunk4_20260828" / "batch-run.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / ".agent_tmp" / "production_wemm_full_postprocess_20260828"


class FinalizeError(RuntimeError):
    """Raised when a checkpoint or post-processing command is invalid."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise FinalizeError(f"{path} root must be an object")
    return value


def _checkpoint_state(path: Path) -> tuple[str, dict[str, Any]]:
    if not path.exists():
        return "MISSING", {}
    try:
        payload = _load(path)
    except (OSError, UnicodeError, json.JSONDecodeError, FinalizeError) as exc:
        return "UNREADABLE", {"error": str(exc)}
    if payload.get("format") != "robata-production-wemm-batch-run-v1":
        return "WRONG_FORMAT", {"format": payload.get("format")}
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return "MALFORMED", {"reason": "summary_not_object"}
    return str(payload.get("status", "UNKNOWN")), {
        "complete": int(summary.get("complete_count", 0) or 0),
        "running": int(summary.get("running_count", 0) or 0),
        "planned": int(summary.get("planned_count", 0) or 0),
        "failed": int(summary.get("failed_count", 0) or 0),
        "windows": int(summary.get("window_count", 0) or 0),
    }


def _run(command: list[str], *, log_path: Path) -> None:
    stamp = datetime.now(UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{stamp}] $ {' '.join(command)}\n")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        log.write(f"[{datetime.now(UTC).isoformat()}] exit={completed.returncode}\n")
    if completed.returncode != 0:
        raise FinalizeError(f"command failed with exit {completed.returncode}: {command[0]}")


def finalize(
    b1: Path,
    remaining: Path,
    output_root: Path,
    *,
    poll_seconds: float = 15.0,
    timeout_seconds: float = 6 * 3600.0,
) -> dict[str, Any]:
    started = time.monotonic()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "watcher.log"
    while True:
        b1_status, b1_info = _checkpoint_state(b1)
        rem_status, rem_info = _checkpoint_state(remaining)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"[{datetime.now(UTC).isoformat()}] b1={b1_status} {b1_info} "
                f"remaining={rem_status} {rem_info}\n"
            )
        if b1_status == "COMPLETE" and rem_status == "COMPLETE":
            break
        if b1_status in {"FAILED", "UNREADABLE", "WRONG_FORMAT", "MALFORMED"}:
            raise FinalizeError(f"B1 checkpoint cannot complete: {b1_status} {b1_info}")
        if rem_status in {"FAILED", "UNREADABLE", "WRONG_FORMAT", "MALFORMED"}:
            raise FinalizeError(f"remaining checkpoint cannot complete: {rem_status} {rem_info}")
        if time.monotonic() - started > timeout_seconds:
            raise FinalizeError("timed out waiting for COMPLETE checkpoints")
        time.sleep(poll_seconds)

    rebuild_dir = output_root / "review_rebuilt"
    aggregate_json = output_root / "review_aggregate.json"
    aggregate_md = output_root / "review_aggregate.md"
    selection_json = output_root / "ambiguity_selection.json"
    selection_md = output_root / "ambiguity_selection.md"
    verification_json = output_root / "verification.json"
    python = str(Path(sys.executable).resolve())
    _run(
        [
            python,
            "scripts/rebuild_production_wemm_review_packs.py",
            str(b1),
            str(remaining),
            "--output-dir",
            str(rebuild_dir),
            "--overwrite",
            "--pretty",
        ],
        log_path=log_path,
    )
    _run(
        [
            python,
            "scripts/aggregate_production_wemm_review_packs.py",
            str(rebuild_dir),
            "--json-output",
            str(aggregate_json),
            "--markdown-output",
            str(aggregate_md),
            "--pretty",
        ],
        log_path=log_path,
    )
    _run(
        [
            python,
            "scripts/select_production_wemm_ambiguities.py",
            str(aggregate_json),
            "--output-json",
            str(selection_json),
            "--output-md",
            str(selection_md),
        ],
        log_path=log_path,
    )
    _run(
        [
            python,
            "scripts/verify_production_wemm_full_postprocess.py",
            str(b1),
            str(remaining),
            "--rebuild-report",
            str(rebuild_dir / "rebuild-report.json"),
            "--aggregate",
            str(aggregate_json),
            "--selection",
            str(selection_json),
            "--expected-complete-per-checkpoint",
            "3",
            "33",
            "--expected-recordings",
            "36",
            "--expected-windows",
            "788",
            "--expected-cameras",
            "6",
            "--output-json",
            str(verification_json),
        ],
        log_path=log_path,
    )
    return {
        "status": "COMPLETE",
        "output_root": str(output_root),
        "rebuild_dir": str(rebuild_dir),
        "aggregate": str(aggregate_json),
        "selection": str(selection_json),
        "verification": str(verification_json),
        "log": str(log_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1", type=Path, default=DEFAULT_B1)
    parser.add_argument("--remaining", type=Path, default=DEFAULT_REMAINING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=float, default=6 * 3600.0)
    args = parser.parse_args(argv)
    try:
        report = finalize(
            args.b1,
            args.remaining,
            args.output_root,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, TypeError, ValueError, FinalizeError) as exc:
        print(f"production WeMM post-processing watcher failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
