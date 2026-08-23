#!/usr/bin/env python3
"""Resolve the frozen Qwen atomic-event pools against local EPIC annotations/media.

This is a small benchmark preparation audit.  Pool membership is imported from
``qwen_atomic_event_pools`` and is never selected from model output.  Official
annotations are attached only after the fixed UIDs have been resolved.  The output
is ordinary local JSON; this script adds no digest, schema, or production contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.qwen_atomic_event_pools import (  # noqa: E402
    FrozenAtomicEventPoolCase,
    QwenAtomicEventPoolName,
    get_frozen_pool,
    resolve_pool_records,
    validate_pool_partition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        action="append",
        required=True,
        help="Repeat for every EPIC annotation split needed by the frozen pools.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pool",
        action="append",
        choices=tuple(item.value for item in QwenAtomicEventPoolName),
        help="Repeat to audit a subset. The default audits H8, D12, C24, and M9.",
    )
    return parser


def _timestamp_seconds(value: object) -> float:
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid EPIC timestamp: {text!r}")
    hours, minutes, seconds = parts
    result = (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)
    if result < 0:
        raise ValueError(f"negative EPIC timestamp: {text!r}")
    return result


def _read_annotations(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"annotation CSV is not a file: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                uid = str(row.get("narration_id", "")).strip()
                if not uid:
                    raise ValueError(f"annotation row without narration_id in {path}")
                if uid in seen:
                    raise ValueError(f"duplicate annotation UID across inputs: {uid}")
                seen.add(uid)
                rows.append(dict(row))
    return rows


def _action_text(row: Mapping[str, object]) -> str:
    verb = str(row.get("verb", "")).strip().replace("-", " ")
    noun = str(row.get("noun", "")).strip().replace(":", " ")
    return " ".join(part for part in (verb, noun) if part).casefold()


def _expected_action_matches(expected: str, row: Mapping[str, object]) -> bool:
    expected_tokens = expected.casefold().replace("-", " ").replace(":", " ").split()
    actual_tokens = _action_text(row).split()
    return all(token in actual_tokens for token in expected_tokens)


def _record_projection(
    *,
    row: Mapping[str, str],
    dataset_root: Path,
    pool_names: tuple[str, ...],
    stratum: str,
    historical: bool,
    frozen_posthoc_action: str,
) -> dict[str, Any]:
    uid = row["narration_id"]
    participant_id = row["participant_id"]
    video_id = row["video_id"]
    video_relpath = f"EPIC-KITCHENS/{participant_id}/videos/{video_id}.MP4"
    video_path = (dataset_root / Path(*video_relpath.split("/"))).resolve()
    try:
        video_path.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"video path escapes dataset root for {uid}") from error
    if not video_path.is_file():
        raise ValueError(f"downloaded video is missing for {uid}: {video_path}")
    start_seconds = _timestamp_seconds(row["start_timestamp"])
    end_seconds = _timestamp_seconds(row["stop_timestamp"])
    if end_seconds <= start_seconds:
        raise ValueError(f"non-positive annotation interval for {uid}")
    return {
        "uid": uid,
        "participant_id": participant_id,
        "video_id": video_id,
        "video_relpath": video_relpath,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": end_seconds - start_seconds,
        "pool_names": list(pool_names),
        "stratum": stratum,
        "historical_regression_anchor": historical,
        "frozen_posthoc_action": frozen_posthoc_action,
        "official_action_text": _action_text(row),
        "frozen_posthoc_action_tokens_match": _expected_action_matches(
            frozen_posthoc_action, row
        ),
        "official_reference": {
            "annotation_id": uid,
            "narration": row.get("narration", ""),
            "verb": row.get("verb", ""),
            "noun": row.get("noun", ""),
            "verb_class": row.get("verb_class", ""),
            "noun_class": row.get("noun_class", ""),
            "start_timestamp": row["start_timestamp"],
            "stop_timestamp": row["stop_timestamp"],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_pool_partition()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")

    selected_names = tuple(
        QwenAtomicEventPoolName(value)
        for value in (args.pool or [item.value for item in QwenAtomicEventPoolName])
    )
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("pool names must not be repeated")
    annotations = _read_annotations(args.annotation_csv)

    memberships: dict[
        str, list[tuple[QwenAtomicEventPoolName, FrozenAtomicEventPoolCase]]
    ] = {}
    pool_rows: dict[str, list[str]] = {}
    for pool_name in selected_names:
        cases = get_frozen_pool(pool_name)
        resolved = resolve_pool_records(
            pool_name,
            annotations,
            uid_getter=lambda row: row["narration_id"],
        )
        pool_rows[pool_name.value] = []
        for case, row in zip(cases, resolved, strict=True):
            if row["participant_id"] != case.participant_id or row["video_id"] != case.video_id:
                raise ValueError(f"annotation identity differs from frozen pool for {case.uid}")
            memberships.setdefault(case.uid, []).append((pool_name, case))
            pool_rows[pool_name.value].append(case.uid)

    by_uid = {row["narration_id"]: row for row in annotations}
    records: list[dict[str, Any]] = []
    for uid, members in memberships.items():
        first_case = members[0][1]
        records.append(
            _record_projection(
                row=by_uid[uid],
                dataset_root=dataset_root,
                pool_names=tuple(pool_name.value for pool_name, _case in members),
                stratum=first_case.stratum.value,
                historical=first_case.historical,
                frozen_posthoc_action=first_case.posthoc_official_action,
            )
        )

    report: dict[str, Any] = {
        "report_version": "qwen-atomic-event-pool-audit-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "official_references_provided_to_model": False,
        "selection_uses_model_output": False,
        "dataset_root": str(dataset_root),
        "annotation_csvs": [str(path.expanduser().resolve()) for path in args.annotation_csv],
        "pool_order": [name.value for name in selected_names],
        "pools": pool_rows,
        "unique_record_count": len(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pools": pool_rows, "unique_record_count": len(records)}, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
