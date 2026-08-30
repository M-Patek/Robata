#!/usr/bin/env python3
"""Verify the read-only full-corpus WeMM post-processing artifacts.

The verifier checks checkpoint completion, one rebuilt review pack per valid
recording, aggregate coverage, and preservation of source review fields in the
ambiguity queue.  It never loads a model, opens media, reads gold, or computes
an identity/hash/digest.  A non-zero exit means the artifacts are incomplete
or structurally inconsistent; it is not a quality claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class VerificationError(ValueError):
    """Raised when a post-processing artifact violates the expected contract."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise VerificationError(f"{path} root must be an object")
    return dict(value)


def _as_list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VerificationError(f"{field} must be an array")
    return list(value)


def _as_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationError(f"{field} must be an object")
    return value


def _check_checkpoint(path: Path, *, expected_complete: int | None) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("format") != "robata-production-wemm-batch-run-v1":
        raise VerificationError(f"{path}: unexpected checkpoint format")
    summary = _as_mapping(payload.get("summary", {}), field=f"{path}.summary")
    status = str(payload.get("status", ""))
    if status != "COMPLETE":
        raise VerificationError(f"{path}: checkpoint status is {status!r}, not COMPLETE")
    if int(summary.get("running_count", 0) or 0) != 0:
        raise VerificationError(f"{path}: running items remain")
    if int(summary.get("planned_count", 0) or 0) != 0:
        raise VerificationError(f"{path}: planned items remain")
    if int(summary.get("failed_count", 0) or 0) != 0:
        raise VerificationError(f"{path}: failed items remain")
    complete_count = int(summary.get("complete_count", 0) or 0)
    if expected_complete is not None and complete_count != expected_complete:
        raise VerificationError(
            f"{path}: complete_count={complete_count}, expected {expected_complete}"
        )
    items = _as_list(payload.get("items", []), field=f"{path}.items")
    complete_items = [
        item for item in items if isinstance(item, Mapping) and item.get("status") == "COMPLETE"
    ]
    if len(complete_items) != complete_count:
        raise VerificationError(f"{path}: summary complete_count disagrees with COMPLETE item rows")
    return {
        "path": str(path),
        "status": status,
        "complete_count": complete_count,
        "window_count": int(summary.get("window_count", 0) or 0),
    }


def _check_rebuild(
    path: Path,
    *,
    expected_recordings: int | None,
) -> tuple[dict[str, Any], set[str]]:
    payload = _load(path)
    if payload.get("format") != "robata-production-wemm-review-pack-rebuild-v1":
        raise VerificationError(f"{path}: unexpected rebuild-report format")
    summary = _as_mapping(payload.get("summary", {}), field=f"{path}.summary")
    if int(summary.get("invalid_count", 0) or 0) != 0:
        raise VerificationError(f"{path}: invalid source artifacts remain")
    if int(summary.get("unresolved_item_count", 0) or 0) != 0:
        raise VerificationError(f"{path}: unresolved checkpoint items remain")
    written = _as_list(payload.get("written", []), field=f"{path}.written")
    ids = {
        str(item.get("recording_id"))
        for item in written
        if isinstance(item, Mapping) and item.get("recording_id") is not None
    }
    written_count = int(summary.get("written_count", 0) or 0)
    if written_count != len(written) or written_count != len(ids):
        raise VerificationError(f"{path}: rebuilt recording rows are not unique/consistent")
    if expected_recordings is not None and written_count != expected_recordings:
        raise VerificationError(
            f"{path}: written_count={written_count}, expected {expected_recordings}"
        )
    return {
        "path": str(path),
        "status": str(payload.get("status", "")),
        "written_count": written_count,
        "recording_ids": sorted(ids),
    }, ids


def _check_aggregate(
    path: Path,
    *,
    expected_recordings: int | None,
    expected_windows: int | None,
    expected_cameras: int,
    rebuilt_ids: set[str],
) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("format") != "robata-production-wemm-review-pack-aggregate-v1":
        raise VerificationError(f"{path}: unexpected aggregate format")
    if payload.get("production_eligible") is not False:
        raise VerificationError(f"{path}: aggregate must remain non-production")
    if payload.get("official_gold_status") != "NOT_ESTABLISHED":
        raise VerificationError(f"{path}: gold status changed unexpectedly")
    summary = _as_mapping(payload.get("summary", {}), field=f"{path}.summary")
    recordings = _as_list(payload.get("recordings", []), field=f"{path}.recordings")
    items = _as_list(payload.get("items", []), field=f"{path}.items")
    recording_ids = {
        str(item.get("recording_id"))
        for item in recordings
        if isinstance(item, Mapping) and item.get("recording_id") is not None
    }
    aggregate_recording_count = int(summary.get("recording_count", 0) or 0)
    aggregate_window_count = int(summary.get("window_count", 0) or 0)
    if aggregate_recording_count != len(recording_ids):
        raise VerificationError(f"{path}: aggregate recording count is inconsistent")
    if aggregate_window_count != len(items):
        raise VerificationError(f"{path}: aggregate window count is inconsistent")
    if expected_recordings is not None and aggregate_recording_count != expected_recordings:
        raise VerificationError(
            f"{path}: recording_count={aggregate_recording_count}, expected {expected_recordings}"
        )
    if expected_windows is not None and aggregate_window_count != expected_windows:
        raise VerificationError(
            f"{path}: window_count={aggregate_window_count}, expected {expected_windows}"
        )
    if rebuilt_ids and recording_ids != rebuilt_ids:
        raise VerificationError(f"{path}: aggregate recording IDs differ from rebuilt packs")
    coverage = summary.get("camera_window_coverage")
    if coverage is None or abs(float(coverage) - 1.0) > 1e-9:
        raise VerificationError(f"{path}: camera-window coverage is not complete: {coverage!r}")
    expected_inputs = aggregate_window_count * expected_cameras
    if int(summary.get("expected_camera_window_input_count", -1) or -1) != expected_inputs:
        raise VerificationError(f"{path}: expected camera input count is inconsistent")
    return {
        "path": str(path),
        "recording_count": aggregate_recording_count,
        "window_count": aggregate_window_count,
        "camera_window_inputs": expected_inputs,
        "camera_window_coverage": float(coverage),
    }


def _check_selection(
    path: Path,
    *,
    expected_windows: int | None,
    expected_cameras: int,
) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("format") != "robata-production-wemm-ambiguity-selection-v1":
        raise VerificationError(f"{path}: unexpected ambiguity-selection format")
    if payload.get("production_eligible") is not False:
        raise VerificationError(f"{path}: selection must remain non-production")
    if payload.get("official_quality_status") != "NOT_MEASURED":
        raise VerificationError(f"{path}: quality status changed unexpectedly")
    summary = _as_mapping(payload.get("summary", {}), field=f"{path}.summary")
    windows = _as_list(payload.get("windows", []), field=f"{path}.windows")
    input_count = int(summary.get("input_window_count", 0) or 0)
    selected_count = int(summary.get("selected_window_count", 0) or 0)
    if selected_count != len(windows):
        raise VerificationError(f"{path}: selected count disagrees with queue rows")
    if expected_windows is not None and input_count != expected_windows:
        raise VerificationError(
            f"{path}: input_window_count={input_count}, expected {expected_windows}"
        )
    contracts = _as_list(payload.get("source_contracts", []), field=f"{path}.source_contracts")
    report_contract = payload.get("review_contract")
    if len(contracts) == 1:
        if not isinstance(report_contract, Mapping) or dict(report_contract) != contracts[0]:
            raise VerificationError(f"{path}: homogeneous source contract is not exposed")
    elif contracts and report_contract is not None:
        raise VerificationError(f"{path}: mixed source contracts must not be merged")
    for index, row in enumerate(windows):
        row_map = _as_mapping(row, field=f"{path}.windows[{index}]")
        for field in ("window_status", "window_decision", "raw_candidates", "review_contract"):
            if field not in row_map:
                raise VerificationError(f"{path}.windows[{index}] dropped {field}")
        if not isinstance(row_map["raw_candidates"], list):
            raise VerificationError(f"{path}.windows[{index}].raw_candidates must be an array")
        if not isinstance(row_map["review_contract"], Mapping):
            raise VerificationError(f"{path}.windows[{index}].review_contract must be an object")
        cameras = row_map.get("declared_camera_ids", [])
        if not isinstance(cameras, list) or len(cameras) != expected_cameras:
            raise VerificationError(
                f"{path}.windows[{index}] declared camera count is not {expected_cameras}"
            )
    return {
        "path": str(path),
        "input_window_count": input_count,
        "selected_window_count": selected_count,
        "source_contract_count": len(contracts),
    }


def verify(
    checkpoints: Sequence[Path],
    rebuild_report: Path,
    aggregate: Path,
    selection: Path,
    *,
    expected_complete_per_checkpoint: Sequence[int] | None = None,
    expected_recordings: int | None = None,
    expected_windows: int | None = None,
    expected_cameras: int = 6,
) -> dict[str, Any]:
    if expected_complete_per_checkpoint is not None and len(
        expected_complete_per_checkpoint
    ) != len(checkpoints):
        raise VerificationError("one expected-complete value is required per checkpoint")
    checkpoint_reports = [
        _check_checkpoint(
            path,
            expected_complete=(
                expected_complete_per_checkpoint[index]
                if expected_complete_per_checkpoint is not None
                else None
            ),
        )
        for index, path in enumerate(checkpoints)
    ]
    rebuild_report_value, rebuilt_ids = _check_rebuild(
        rebuild_report, expected_recordings=expected_recordings
    )
    aggregate_report = _check_aggregate(
        aggregate,
        expected_recordings=expected_recordings,
        expected_windows=expected_windows,
        expected_cameras=expected_cameras,
        rebuilt_ids=rebuilt_ids,
    )
    selection_report = _check_selection(
        selection,
        expected_windows=expected_windows,
        expected_cameras=expected_cameras,
    )
    return {
        "format": "robata-production-wemm-full-postprocess-verification-v1",
        "status": "STRUCTURALLY_VERIFIED",
        "quality_claim": False,
        "production_eligible": False,
        "checkpoints": checkpoint_reports,
        "rebuild": rebuild_report_value,
        "aggregate": aggregate_report,
        "selection": selection_report,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "qwen_read": False,
            "mage_read": False,
            "gold_read": False,
            "gold_written": False,
            "hash_or_digest_computed": False,
        },
        "limitations": [
            "Structural coverage is not visual QA, human gold, or accuracy.",
            "Processing windows remain context units, not inferred action boundaries.",
            "The malformed source recording is intentionally outside the valid corpus.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="completed batch-run checkpoints")
    parser.add_argument("--rebuild-report", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--expected-complete-per-checkpoint",
        type=int,
        nargs="+",
        help="optional expected COMPLETE count for each checkpoint, in argument order",
    )
    parser.add_argument("--expected-recordings", type=int)
    parser.add_argument("--expected-windows", type=int)
    parser.add_argument("--expected-cameras", type=int, default=6)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify(
            args.checkpoints,
            args.rebuild_report,
            args.aggregate,
            args.selection,
            expected_complete_per_checkpoint=args.expected_complete_per_checkpoint,
            expected_recordings=args.expected_recordings,
            expected_windows=args.expected_windows,
            expected_cameras=args.expected_cameras,
        )
    except (OSError, TypeError, ValueError, VerificationError) as exc:
        print(f"production WeMM post-process verification failed: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
