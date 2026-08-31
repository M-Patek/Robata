"""Small, source-bound Qwen pass for the production WeMM ambiguity queue.

This module is intentionally an orchestration seam rather than another model
runner.  The WeMM ambiguity selector has already chosen a bounded set of
processing windows; this helper groups those rows by recording, stages one
MCAP at a time, materializes the complete six-camera native-video inputs, and
invokes the existing Qwen candidate verifier.  It never infers action
boundaries and it never reads EPIC/gold/Mapper artifacts.

The default native execution keeps one Qwen model resident for the batch while
still staging and materializing one recording at a time.  All output is
non-production review evidence and has ``quality_claim=False``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from .production_wemm_batch_runner import build_recording_manifest, stage_zip_member
from .qwen_native_video_bridge import (
    build_qwen_native_video_plan,
    materialize_qwen_native_video_inputs,
)

AMBIGUITY_SELECTION_FORMAT: Final = "robata-production-wemm-ambiguity-selection-v1"
BATCH_FORMAT: Final = "robata-production-wemm-qwen-ambiguity-batch-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
DEFAULT_FRAME_COUNT: Final = 8
DEFAULT_MAX_IMAGE_SIDE: Final = 320
DEFAULT_MAX_NEW_TOKENS: Final = 256
DEFAULT_GPU_WEIGHT_MEMORY_GIB: Final = 5
DEFAULT_CPU_WEIGHT_MEMORY_GIB: Final = 16
DEFAULT_JPEG_QUALITY: Final = 92
DEFAULT_WINDOW_SECONDS: Final = 8.0
DEFAULT_EXPECTED_CAMERA_IDS: Final = (
    "cam_01",
    "cam_02",
    "cam_03",
    "cam_04",
    "cam_05",
    "cam_06",
)

# A recording is considered durable enough to reuse only after the verifier
# route has written its per-recording report.  ``COMPLETE`` is accepted as a
# compatibility spelling for hand-authored/early checkpoints; the current
# runner writes ``SUCCEEDED``.
_RESUME_SUCCESS_STATUSES: Final = frozenset({"SUCCEEDED", "COMPLETE"})
_RESUME_COMPAT_CONFIG_KEYS: Final = (
    "window_seconds",
    "include_tail",
    "frame_count",
    "max_image_side",
    "max_new_tokens",
    "jpeg_quality",
    "proposal_index",
    "verdict_scope",
    "include_optional_fields",
    "dry_run",
    "mapping_config",
)


class ProductionQwenAmbiguityBatchError(RuntimeError):
    """Raised when an ambiguity batch cannot be planned or executed."""


@dataclass(frozen=True, slots=True)
class _RecordingSourcePlan:
    """One archive-member or direct-MCAP source selected for a recording.

    The direct-MCAP branch is deliberately diagnostic-only.  It lets a local
    WeMM sidecar that was produced directly from one MCAP exercise the same
    bounded Qwen path without fabricating a ZIP member.  Archive-backed
    selections continue to use the existing single-member staging lifecycle.
    """

    recording_id: str
    source_mode: str
    source_path: Path | None
    archive_path: Path | None
    archive_member: str | None
    source_preflight_status: str
    qa_status: str
    slug: str


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionQwenAmbiguityBatchError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionQwenAmbiguityBatchError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if value is None:
        if allow_empty:
            return ""
        raise ProductionQwenAmbiguityBatchError(f"{field} must be non-empty text")
    result = str(value).strip()
    if not result and not allow_empty:
        raise ProductionQwenAmbiguityBatchError(f"{field} must be non-empty text")
    return result


def _copy_json(value: object, *, field: str = "value") -> Any:
    """Detach bounded JSON metadata without deriving an identity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionQwenAmbiguityBatchError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionQwenAmbiguityBatchError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[]") for child in value]
    raise ProductionQwenAmbiguityBatchError(f"{field} must be JSON-compatible")


def _load_json(value: Mapping[str, Any] | str | Path, *, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionQwenAmbiguityBatchError(f"could not read {field} {path}: {exc}") from exc
    return dict(_mapping(payload, field=field))


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return token[:100] or "recording"


def _resolve_path(value: object, *, base: Path | None = None) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Repository commands conventionally pass paths relative to the working
    # directory, while sidecars may be colocated with a selection artifact.
    # Prefer an existing cwd-relative source and then the sidecar-relative
    # spelling; retain cwd-relative as the diagnostic fallback.
    candidates = [Path.cwd() / path]
    if base is not None:
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def _source_snapshot(row: Mapping[str, Any]) -> Mapping[str, Any]:
    source_ref = row.get("source_ref")
    if not isinstance(source_ref, Mapping):
        return {}
    source = source_ref.get("source")
    return source if isinstance(source, Mapping) else source_ref


def _row_archive_member_candidates(row: Mapping[str, Any]) -> tuple[object, ...]:
    source_ref = row.get("source_ref")
    source = _source_snapshot(row)
    return (
        row.get("archive_member"),
        source_ref.get("archive_member") if isinstance(source_ref, Mapping) else None,
        source.get("archive_member"),
    )


def _row_archive_member(row: Mapping[str, Any]) -> str:
    for candidate in _row_archive_member_candidates(row):
        if isinstance(candidate, str) and candidate.strip():
            member = candidate.strip().replace("\\", "/")
            parts = [part for part in member.split("/") if part]
            if (
                member.startswith("/")
                or "://" in member
                or any(part in {".", ".."} for part in parts)
            ):
                raise ProductionQwenAmbiguityBatchError(
                    f"unsafe archive member for {row.get('window_id', '<window>')!r}"
                )
            if not member.casefold().endswith(".mcap"):
                raise ProductionQwenAmbiguityBatchError(f"archive member is not an MCAP: {member}")
            return "/".join(parts)
    raise ProductionQwenAmbiguityBatchError(
        f"selection row {row.get('window_id', '<window>')!r} lacks source_ref.archive_member"
    )


def _row_direct_mcap_path(row: Mapping[str, Any], *, selection_base: Path | None) -> Path | None:
    """Resolve a local direct-MCAP reference when no archive member exists.

    Some bounded diagnostic cohorts are deliberately run from a local MCAP,
    rather than from the archive used by the serial production batch runner.
    Do not infer or manufacture an archive member for that case: accept only
    an explicitly declared ``*.mcap`` path and let the caller use a no-op
    staging context.  Archive-backed rows are selected first by the caller,
    so this helper cannot alter their behavior.
    """

    source_ref = row.get("source_ref")
    source = _source_snapshot(row)
    candidates: list[object] = [
        row.get("source_path"),
        source_ref.get("source_path") if isinstance(source_ref, Mapping) else None,
        source_ref.get("path") if isinstance(source_ref, Mapping) else None,
        source.get("source_path"),
        source.get("path"),
    ]
    first: Path | None = None
    for candidate in candidates:
        if not isinstance(candidate, (str, Path)) or not str(candidate).strip():
            continue
        declared = Path(candidate).expanduser()
        if declared.suffix.casefold() != ".mcap":
            continue
        resolved = _resolve_path(candidate, base=selection_base)
        if resolved is None:
            continue
        if first is None:
            first = resolved
        if resolved.is_file():
            return resolved
    # Keep an explicitly declared but currently missing path so the execution
    # path can report the useful source-file error rather than treating the
    # row as an archive selection.
    return first


def _selection_source_contract(
    row: Mapping[str, Any], *, selection_base: Path | None = None
) -> dict[str, Any]:
    """Snapshot the selector-declared source without deriving an identity."""

    has_archive_declaration = any(
        isinstance(candidate, str) and candidate.strip()
        for candidate in _row_archive_member_candidates(row)
    )
    if has_archive_declaration:
        return {
            "source_mode": "ARCHIVE_MEMBER",
            "archive_member": _row_archive_member(row),
            "source_path": None,
        }
    direct_path = _row_direct_mcap_path(row, selection_base=selection_base)
    if direct_path is not None:
        return {
            "source_mode": "DIRECT_MCAP",
            "archive_member": None,
            "source_path": str(direct_path),
        }
    raise ProductionQwenAmbiguityBatchError(
        f"selection row {row.get('window_id', '<window>')!r} lacks source_ref.archive_member "
        "or a direct MCAP source path"
    )


def _row_archive_path(
    row: Mapping[str, Any],
    *,
    override: str | Path | None,
    source_manifest: Mapping[str, Any] | None,
    selection_base: Path | None,
) -> Path | None:
    if override is not None:
        return _resolve_path(override, base=selection_base)
    source_ref = row.get("source_ref")
    source = _source_snapshot(row)
    candidates: list[object] = [
        source_ref.get("archive_path") if isinstance(source_ref, Mapping) else None,
        source.get("archive_path"),
    ]
    if source_manifest is not None:
        manifest_source = source_manifest.get("source")
        if isinstance(manifest_source, Mapping):
            candidates.append(manifest_source.get("archive_path"))
        candidates.append(source_manifest.get("archive_path"))
    first: Path | None = None
    for candidate in candidates:
        path = _resolve_path(candidate, base=selection_base)
        if path is not None:
            if first is None:
                first = path
            if path.is_file():
                return path
    # Keep the first path for a useful error in non-dry runs; callers may
    # intentionally use a not-yet-mounted source during planning.
    return first


def _source_status(
    row: Mapping[str, Any],
    *,
    archive_member: str | None,
    source_manifest: Mapping[str, Any] | None,
) -> tuple[str, str]:
    source = _source_snapshot(row)
    status = row.get("source_preflight_status") or source.get("source_preflight_status")
    qa = row.get("qa_status") or source.get("qa_status")
    if source_manifest is not None:
        raw_items = source_manifest.get("items", ())
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
            raw_items = ()
        if archive_member is not None:
            for raw in raw_items:
                if not isinstance(raw, Mapping):
                    continue
                member = raw.get("name", raw.get("archive_member"))
                if isinstance(member, str) and member.replace("\\", "/") == archive_member:
                    status = status or raw.get("source_preflight_status", raw.get("status"))
                    qa = qa or raw.get("qa_status")
                    break
    return str(status or "PASS").upper(), str(qa or "PENDING").upper()


def _camera_ids(row: Mapping[str, Any]) -> list[str]:
    for key in ("declared_camera_ids", "camera_ids"):
        value = row.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            result = [str(item).strip() for item in value if str(item).strip()]
            if result:
                return result
    source = _source_snapshot(row)
    value = source.get("camera_ids")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = [str(item).strip() for item in value if str(item).strip()]
        if result:
            return result
    return list(DEFAULT_EXPECTED_CAMERA_IDS)


def _selection_rows(
    selection: Mapping[str, Any] | str | Path,
    *,
    recording_ids: Sequence[str] | None,
    limit: int | None,
) -> tuple[dict[str, Any], str | None]:
    document = _load_json(selection, field="ambiguity_selection")
    if document.get("format") != AMBIGUITY_SELECTION_FORMAT:
        raise ProductionQwenAmbiguityBatchError(
            f"ambiguity_selection has unsupported format; expected {AMBIGUITY_SELECTION_FORMAT}"
        )
    raw_rows = _sequence(document.get("windows"), field="ambiguity_selection.windows")
    requested = None
    if recording_ids:
        requested = {str(value).strip() for value in recording_ids if str(value).strip()}
        if not requested:
            raise ProductionQwenAmbiguityBatchError("recording_ids must contain non-empty values")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row = dict(_mapping(raw, field=f"ambiguity_selection.windows[{index}]"))
        recording_id = _text(row.get("recording_id"), field=f"windows[{index}].recording_id")
        window_id = _text(row.get("window_id"), field=f"windows[{index}].window_id")
        if requested is not None and recording_id not in requested:
            continue
        # The queue contract explicitly says these are context windows.  A
        # false value is safe to add when an older selector omitted the field;
        # never derive an action interval from it.
        row["source_context_is_action_boundary"] = False
        row["recording_id"] = recording_id
        row["window_id"] = window_id
        rows.append(row)
    if requested is not None:
        found = {str(row["recording_id"]) for row in rows}
        unknown = sorted(requested - found)
        if unknown:
            raise ProductionQwenAmbiguityBatchError(
                "requested recording IDs were not found: " + ", ".join(unknown)
            )
    rows.sort(
        key=lambda row: (
            str(row["recording_id"]),
            int(row.get("ordinal", 0)),
            str(row["window_id"]),
        )
    )
    grouped_ids: list[str] = []
    for row in rows:
        rid = str(row["recording_id"])
        if rid not in grouped_ids:
            grouped_ids.append(rid)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ProductionQwenAmbiguityBatchError("limit must be a positive recording count")
        allowed = set(grouped_ids[:limit])
        rows = [row for row in rows if str(row["recording_id"]) in allowed]
    if not rows:
        raise ProductionQwenAmbiguityBatchError(
            "ambiguity selection contains no rows after filtering"
        )
    return {"document": document, "rows": rows}, (
        str(selection) if isinstance(selection, (str, Path)) else None
    )


def _source_manifest_items(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    document = _load_json(value, field="source_manifest")
    items = document.get("items")
    if items is not None and not isinstance(items, Sequence):
        raise ProductionQwenAmbiguityBatchError("source_manifest.items must be an array")
    return document


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _selection_resume_contract(
    rows: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    selection_base: Path | None = None,
) -> dict[str, Any]:
    """Return explicit, hash-free selection metadata for checkpoint recovery.

    The contract deliberately contains only the identifiers that the selector
    already supplied.  It does not derive an action boundary or an identity
    digest.  Keeping the ordered window/camera declarations lets a resumed run
    reject a different selection before staging media or loading a model.
    """

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("recording_id", "")),
            int(row.get("ordinal", 0) or 0),
            str(row.get("window_id", "")),
        ),
    )
    windows: list[dict[str, Any]] = []
    for row in ordered_rows:
        windows.append(
            {
                "recording_id": str(row.get("recording_id", "")),
                "window_id": str(row.get("window_id", "")),
                "ordinal": _copy_json(row.get("ordinal"), field="resume.ordinal"),
                **_selection_source_contract(row, selection_base=selection_base),
                "source_interval": _copy_json(
                    row.get("source_interval"), field="resume.source_interval"
                ),
                "camera_ids": list(_camera_ids(row)),
            }
        )
    return {
        "version": 1,
        "recording_ids": sorted(str(key) for key in grouped),
        "window_count": len(ordered_rows),
        "windows": windows,
    }


def _resume_config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only execution knobs whose change would invalidate a checkpoint."""

    return {
        key: _copy_json(config.get(key), field=f"resume.config.{key}")
        for key in _RESUME_COMPAT_CONFIG_KEYS
    }


def _validate_partial_resume(
    partial: Mapping[str, Any],
    *,
    partial_path: Path,
    selection_contract: Mapping[str, Any],
    archive_override: Path | None,
) -> None:
    """Fail clearly when a partial report cannot safely be resumed.

    No content digest is used.  We compare the selector's explicit recording,
    window and camera declarations, plus an explicitly supplied archive path.
    Older reports without this metadata are recovered from their explicit
    selector sidecar when available; otherwise the operator can rerun without
    ``--resume`` to start fresh.
    """

    if partial.get("format") != BATCH_FORMAT:
        raise ProductionQwenAmbiguityBatchError(
            f"partial report {partial_path} has unsupported format; expected {BATCH_FORMAT}"
        )
    contract = partial.get("resume_contract")
    if not isinstance(contract, Mapping):
        source = partial.get("source")
        if isinstance(source, Mapping):
            contract = source.get("selection_contract")
    if not isinstance(contract, Mapping):
        raise ProductionQwenAmbiguityBatchError(
            f"partial report {partial_path} lacks resume selection metadata; "
            "rerun without --resume to start a new batch"
        )
    prior_selection = contract.get("selection") if isinstance(contract, Mapping) else None
    if not isinstance(prior_selection, Mapping):
        # Accept the compact shape used by the first checkpoint writer.
        prior_selection = contract
    if dict(prior_selection) != dict(selection_contract):
        raise ProductionQwenAmbiguityBatchError(
            "partial report selection differs from requested ambiguity selection; "
            "use a new output directory or rerun without --resume"
        )
    prior_source = partial.get("source")
    prior_archive = None
    if isinstance(prior_source, Mapping):
        prior_archive = prior_source.get("archive_override")
    if prior_archive and archive_override is not None:
        prior_path = _resolve_path(prior_archive)
        if prior_path is not None and prior_path != archive_override:
            raise ProductionQwenAmbiguityBatchError(
                "partial report archive differs from requested source archive; "
                "use a new output directory or rerun without --resume"
            )
    recordings = partial.get("recordings")
    windows = partial.get("windows")
    if not isinstance(recordings, Sequence) or isinstance(recordings, (str, bytes, bytearray)):
        raise ProductionQwenAmbiguityBatchError(
            f"partial report {partial_path} has invalid recordings array"
        )
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes, bytearray)):
        raise ProductionQwenAmbiguityBatchError(
            f"partial report {partial_path} has invalid windows array"
        )
    expected_recordings = set(str(item) for item in selection_contract["recording_ids"])
    for index, raw in enumerate(recordings):
        item = _mapping(raw, field=f"partial.recordings[{index}]")
        recording_id = str(item.get("recording_id", ""))
        if recording_id and recording_id not in expected_recordings:
            raise ProductionQwenAmbiguityBatchError(
                "partial report contains a recording outside the requested selection: "
                f"{recording_id}"
            )


def _recover_legacy_selection_contract(
    partial: Mapping[str, Any], *, partial_path: Path
) -> dict[str, Any] | None:
    """Recover selection metadata from pre-resume checkpoints when possible.

    Early checkpoints did not write an explicit contract, but they did retain
    the selector path in ``source.selection``.  Reading that already-declared
    selection is safe and keeps an interrupted pilot resumable.  If the sidecar
    is unavailable, callers fail closed with a clear message.
    """

    source = partial.get("source")
    if not isinstance(source, Mapping):
        return None
    selection_ref = source.get("selection")
    selection_path = _resolve_path(selection_ref, base=partial_path.parent)
    if selection_path is None or not selection_path.is_file():
        return None
    try:
        selected, _ = _selection_rows(selection_path, recording_ids=None, limit=None)
    except ProductionQwenAmbiguityBatchError:
        return None
    raw_rows = selected.get("rows", [])
    if not isinstance(raw_rows, Sequence):
        return None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if isinstance(row, Mapping):
            grouped[str(row.get("recording_id", ""))].append(dict(row))
    # A legacy invocation may have used ``--limit``/``--recording-id``.  In
    # that case the durable recording list is the only explicit declaration of
    # the subset; narrow the recovered selection to those IDs when available.
    prior_recordings = partial.get("recordings")
    summary = partial.get("summary")
    selected_count = summary.get("selected_window_count") if isinstance(summary, Mapping) else None
    # Narrow only when the old report explicitly says it selected fewer rows
    # than the referenced selector artifact.  A full-batch interruption has a
    # short ``recordings`` prefix too, but must retain the selector's full set.
    if (
        isinstance(selected_count, int)
        and selected_count != len(raw_rows)
        and isinstance(prior_recordings, Sequence)
        and not isinstance(prior_recordings, (str, bytes, bytearray))
    ):
        prior_ids = {
            str(item.get("recording_id"))
            for item in prior_recordings
            if isinstance(item, Mapping) and str(item.get("recording_id", "")).strip()
        }
        narrowed = {rid: values for rid, values in grouped.items() if rid in prior_ids}
        narrowed_rows = [row for rid in sorted(narrowed) for row in narrowed[rid]]
        if narrowed and len(narrowed_rows) == selected_count:
            grouped = narrowed
            raw_rows = narrowed_rows
    if not grouped:
        return None
    return _selection_resume_contract(raw_rows, grouped, selection_base=partial_path.parent)


def _validate_partial_config(
    partial: Mapping[str, Any], *, current_config: Mapping[str, Any], partial_path: Path
) -> None:
    """Reject changes to processing semantics while allowing resource paths."""

    prior_config = partial.get("config")
    if not isinstance(prior_config, Mapping):
        raise ProductionQwenAmbiguityBatchError(
            f"partial report {partial_path} lacks execution configuration; "
            "rerun without --resume to start a new batch"
        )
    prior_projection = _resume_config_projection(prior_config)
    current_projection = _resume_config_projection(current_config)
    if prior_projection != current_projection:
        changed = [
            key
            for key in _RESUME_COMPAT_CONFIG_KEYS
            if prior_projection.get(key) != current_projection.get(key)
        ]
        raise ProductionQwenAmbiguityBatchError(
            "partial report execution configuration differs from requested run "
            f"({', '.join(changed)}); use a new output directory or rerun without --resume"
        )


def _refresh_ambiguity_summary(
    report: dict[str, Any], *, selected_recording_count: int, selected_window_count: int
) -> None:
    """Recompute counts from durable arrays after a fresh or resumed attempt."""

    recordings = report.get("recordings", [])
    windows = report.get("windows", [])
    if not isinstance(recordings, Sequence) or isinstance(recordings, (str, bytes, bytearray)):
        recordings = []
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes, bytearray)):
        windows = []
    succeeded = 0
    failed = 0
    planned = 0
    skipped = 0
    for raw in recordings:
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status", "")).upper()
        if status in _RESUME_SUCCESS_STATUSES:
            succeeded += 1
        elif status in {"FAILED", "PARTIAL"}:
            failed += 1
        elif status == "PLANNED":
            planned += 1
        if bool(raw.get("resume_skipped", False)):
            skipped += 1
    report["summary"] = {
        **dict(_mapping(report.get("summary", {}), field="report.summary")),
        "selected_recording_count": selected_recording_count,
        "selected_window_count": selected_window_count,
        "succeeded_recording_count": succeeded,
        "failed_recording_count": failed,
        "planned_recording_count": planned,
        "skipped_recording_count": skipped,
        "verifier_row_count": len(windows),
        "failed_row_count": sum(
            1
            for row in windows
            if isinstance(row, Mapping) and str(row.get("status", "")).upper() == "FAILED"
        ),
    }


def _native_verifier_run() -> Callable[[argparse.Namespace], dict[str, Any]]:
    """Load the existing script lazily so dry-runs do not import torch."""

    # When this CLI is launched as ``python scripts/<entrypoint>.py``, Python
    # puts ``scripts/`` (rather than the repository root) on ``sys.path``.
    # ``scripts`` is intentionally not a package, so a normal dotted import
    # fails under the Anaconda runtime used for Qwen.  Load the existing runner
    # by its file path instead; this keeps the model import lazy and avoids
    # duplicating verifier logic.
    repository_root = Path(__file__).resolve().parents[3]
    script_path = (
        repository_root / "scripts" / "run_production_wemm_qwen_candidate_verifier_native.py"
    )
    if not script_path.is_file():
        raise ProductionQwenAmbiguityBatchError(f"native verifier script is missing: {script_path}")
    module_name = "_robata_production_wemm_qwen_candidate_verifier_native"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ProductionQwenAmbiguityBatchError(
            f"could not load native verifier script: {script_path}"
        )
    module = importlib.util.module_from_spec(spec)
    # Register before execution just like a normal import.  This matters if a
    # future verifier helper introspects ``sys.modules`` while importing.
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    runner_value = getattr(module, "run", None)
    if not callable(runner_value):
        raise ProductionQwenAmbiguityBatchError(
            f"native verifier script has no callable run(): {script_path}"
        )
    runner = cast(Callable[[argparse.Namespace], dict[str, Any]], runner_value)
    # Keep the legacy callable shape for injected/test runners while exposing
    # the optional resident-runtime seam to the production batch below.  A
    # function object is intentionally used as the carrier so callers that
    # monkeypatch ``_native_verifier_run`` remain fully compatible.
    resident_runner = getattr(module, "run_with_runtime", None)
    if callable(resident_runner):
        with suppress(AttributeError, TypeError):  # pragma: no cover - exotic callables
            runner.run_with_runtime = resident_runner  # type: ignore[attr-defined]
    return runner


def _verifier_args(
    *,
    candidates: Path,
    manifest: Path,
    video_root: Path,
    model_dir: Path,
    offload_dir: Path,
    window_ids: Sequence[str],
    frame_count: int,
    max_image_side: int,
    max_new_tokens: int,
    gpu_weight_memory_gib: int,
    cpu_weight_memory_gib: int,
    jpeg_quality: int,
    proposal_index: int | None,
    verdict_scope: str,
    include_optional_fields: bool,
    output: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        candidates=candidates,
        manifest=manifest,
        video_root=video_root,
        model_dir=model_dir,
        offload_dir=offload_dir,
        limit=None,
        window_id=list(window_ids),
        camera_id=None,
        frame_count=frame_count,
        max_image_side=max_image_side,
        max_new_tokens=max_new_tokens,
        gpu_weight_memory_gib=gpu_weight_memory_gib,
        cpu_weight_memory_gib=cpu_weight_memory_gib,
        jpeg_quality=jpeg_quality,
        proposal_index=proposal_index,
        verdict_scope=verdict_scope,
        include_optional_fields=include_optional_fields,
        output=output,
    )


def _failure_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    error: str,
    recording_id: str,
    native_video_complete: bool = False,
    status: str = "FAILED",
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        interval = _copy_json(row.get("source_interval"), field="failure.source_interval")
        for camera_id in _camera_ids(row):
            result.append(
                {
                    "recording_id": recording_id,
                    "window_id": str(row.get("window_id")),
                    "ordinal": row.get("ordinal"),
                    "interval": interval,
                    "camera_id": camera_id,
                    "input_mode": "native_video",
                    "native_video_complete": native_video_complete,
                    "status": status,
                    "error": error if status == "FAILED" else None,
                    "provenance": {
                        "frame_sha256_computed": False,
                        "source_context_is_action_boundary": False,
                    },
                }
            )
    return result


def _candidate_document(
    rows: Sequence[Mapping[str, Any]],
    *,
    recording_id: str,
    archive_path: Path | None,
    archive_member: str | None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    source = _source_snapshot(rows[0]) if rows else {}
    source_out = dict(_copy_json(source, field="candidate.source") or {})
    source_out.update(
        {
            "recording_id": recording_id,
            "archive_path": (
                str(archive_path) if archive_path is not None else source_out.get("archive_path")
            ),
            "archive_member": archive_member,
            "path_lifecycle": (
                "DIRECT_MCAP_READ_ONLY"
                if source_path is not None and archive_member is None
                else "SOURCE_ARCHIVE_MEMBER_STAGED_PER_RECORDING"
            ),
        }
    )
    if source_path is not None:
        source_out["source_path"] = str(source_path)
        source_out["path"] = str(source_path)
    windows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(_copy_json(row, field="candidate.window"))
        item["source_context_is_action_boundary"] = False
        windows.append(item)
    return {
        "format": AMBIGUITY_SELECTION_FORMAT,
        "authority": AUTHORITY,
        "status": "QWEN_REVIEW_CANDIDATES",
        "production_eligible": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "official_quality_status": "NOT_MEASURED",
        "quality_claim": False,
        "source": source_out,
        "windows": windows,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "epic_ontology_read": False,
            "mapper_read": False,
            "hash_or_digest_computed": False,
        },
        "windows_are_action_segments": False,
    }


def _recording_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    archive_path: Path | None,
    source_manifest: Mapping[str, Any] | None,
    selection_base: Path | None,
) -> _RecordingSourcePlan:
    recording_id = _text(rows[0].get("recording_id"), field="recording_id")
    # Preserve the archive-backed path exactly as before.  A direct MCAP is
    # considered only when no archive member is declared (and no explicit
    # archive override was supplied), so a mixed/ambiguous source fails rather
    # than silently changing media provenance.
    archive_members: set[str] = set()
    declared_archive_rows = [
        row
        for row in rows
        if any(
            isinstance(candidate, str) and candidate.strip()
            for candidate in _row_archive_member_candidates(row)
        )
    ]
    has_archive_declaration = bool(declared_archive_rows)
    if has_archive_declaration:
        for row in rows:
            archive_members.add(_row_archive_member(row))
    if has_archive_declaration and len(declared_archive_rows) != len(rows):
        raise ProductionQwenAmbiguityBatchError(
            f"recording {recording_id} mixes archive and direct MCAP source declarations"
        )
    if archive_members or archive_path is not None:
        if not archive_members:
            raise ProductionQwenAmbiguityBatchError(
                f"recording {recording_id} lacks an archive member for the supplied archive"
            )
        if len(archive_members) != 1:
            raise ProductionQwenAmbiguityBatchError(
                f"recording {recording_id} has multiple archive members: "
                f"{sorted(archive_members)!r}"
            )
        member: str | None = next(iter(archive_members))
        resolved_archive = archive_path or _row_archive_path(
            rows[0],
            override=None,
            source_manifest=source_manifest,
            selection_base=selection_base,
        )
        source_path = None
        source_mode = "ARCHIVE_MEMBER"
    else:
        direct_paths = {_row_direct_mcap_path(row, selection_base=selection_base) for row in rows}
        direct_paths.discard(None)
        if not direct_paths:
            raise ProductionQwenAmbiguityBatchError(
                f"recording {recording_id} lacks an archive member or direct MCAP path"
            )
        if len(direct_paths) != 1:
            raise ProductionQwenAmbiguityBatchError(
                f"recording {recording_id} has multiple direct MCAP paths: "
                f"{sorted(str(path) for path in direct_paths)!r}"
            )
        member = None
        resolved_archive = None
        source_path = next(iter(direct_paths))
        assert source_path is not None  # narrowed by discard above
        source_mode = "DIRECT_MCAP"
    status, qa = _source_status(rows[0], archive_member=member, source_manifest=source_manifest)
    return _RecordingSourcePlan(
        recording_id=recording_id,
        source_mode=source_mode,
        source_path=source_path,
        archive_path=resolved_archive,
        archive_member=member,
        source_preflight_status=status,
        qa_status=qa,
        slug=_slug(recording_id),
    )


def run_production_qwen_ambiguity_batch(
    selection: Mapping[str, Any] | str | Path,
    *,
    output_directory: str | Path,
    archive_path: str | Path | None = None,
    source_zip: str | Path | None = None,
    source_manifest: Mapping[str, Any] | str | Path | None = None,
    source_preflight: Mapping[str, Any] | str | Path | None = None,
    model_directory: str | Path | None = None,
    offload_directory: str | Path | None = None,
    recording_ids: Sequence[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    keep_staging: bool = False,
    resume: bool = False,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    include_tail: bool = True,
    frame_count: int = DEFAULT_FRAME_COUNT,
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    gpu_weight_memory_gib: int = DEFAULT_GPU_WEIGHT_MEMORY_GIB,
    cpu_weight_memory_gib: int = DEFAULT_CPU_WEIGHT_MEMORY_GIB,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    proposal_index: int | None = None,
    verdict_scope: str = "selected_only",
    include_optional_fields: bool = False,
    mapping_config: str | Path | None = None,
    allow_unapproved_profile: bool = False,
    verifier_runner: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a bounded Qwen pass over selected WeMM ambiguity windows.

    ``limit`` counts recordings, not windows.  Every selected window remains a
    processing context; this function never projects it into an action span.
    """

    if not isinstance(dry_run, bool) or not isinstance(keep_staging, bool):
        raise ProductionQwenAmbiguityBatchError("dry_run and keep_staging must be boolean")
    if not isinstance(resume, bool):
        raise ProductionQwenAmbiguityBatchError("resume must be boolean")
    if not isinstance(include_tail, bool) or not isinstance(include_optional_fields, bool):
        raise ProductionQwenAmbiguityBatchError(
            "include_tail and include_optional_fields must be boolean"
        )
    if not math.isfinite(float(window_seconds)) or float(window_seconds) <= 0:
        raise ProductionQwenAmbiguityBatchError("window_seconds must be positive and finite")
    for value, name in (
        (frame_count, "frame_count"),
        (max_image_side, "max_image_side"),
        (max_new_tokens, "max_new_tokens"),
        (gpu_weight_memory_gib, "gpu_weight_memory_gib"),
        (cpu_weight_memory_gib, "cpu_weight_memory_gib"),
        (jpeg_quality, "jpeg_quality"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ProductionQwenAmbiguityBatchError(f"{name} must be a positive integer")
    if verdict_scope not in {"selected_only", "all_candidates", "pairwise"}:
        raise ProductionQwenAmbiguityBatchError("verdict_scope is unsupported")
    if include_optional_fields and verdict_scope != "selected_only":
        raise ProductionQwenAmbiguityBatchError(
            "include_optional_fields requires verdict_scope=selected_only"
        )
    if not isinstance(allow_unapproved_profile, bool):
        raise ProductionQwenAmbiguityBatchError("allow_unapproved_profile must be boolean")
    if archive_path is not None and source_zip is not None:
        raise ProductionQwenAmbiguityBatchError("pass only one of archive_path or source_zip")
    if source_manifest is not None and source_preflight is not None:
        raise ProductionQwenAmbiguityBatchError(
            "pass only one of source_manifest or source_preflight"
        )
    archive_path = archive_path if archive_path is not None else source_zip
    source_manifest = source_manifest if source_manifest is not None else source_preflight

    selected, selection_path = _selection_rows(selection, recording_ids=recording_ids, limit=limit)
    document = selected["document"]
    rows = selected["rows"]
    if not dry_run and model_directory is None:
        raise ProductionQwenAmbiguityBatchError("model_directory is required for a non-dry run")
    source_doc = _source_manifest_items(source_manifest)
    selection_base = (
        Path(selection_path).expanduser().resolve().parent if selection_path else Path.cwd()
    )
    archive_override = _resolve_path(archive_path, base=selection_base)
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_root = output_dir / "staging"
    manifests_dir = output_dir / "manifests"
    candidates_dir = output_dir / "candidates"
    bridge_dir = output_dir / "bridge"
    verifier_dir = output_dir / "verifier"
    video_dir = output_dir / "video"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recording_id"])].append(row)

    # Resume is deliberately opt-in.  Read and validate the lightweight
    # checkpoint before importing/loading any Qwen runtime so a mismatched
    # selection fails quickly and cannot mix outputs from another batch.
    partial_path = output_dir / "batch-report.partial.json"
    selection_contract = _selection_resume_contract(rows, grouped, selection_base=selection_base)
    prior_report: dict[str, Any] | None = None
    if resume:
        if not partial_path.is_file():
            raise ProductionQwenAmbiguityBatchError(
                f"resume requested but partial batch report is missing: {partial_path}"
            )
        prior_report = _load_json(partial_path, field="partial batch report")
        if not isinstance(prior_report.get("resume_contract"), Mapping):
            recovered = _recover_legacy_selection_contract(prior_report, partial_path=partial_path)
            if recovered is not None:
                # Keep the recovered declaration in memory and persist it only
                # after the current report has been validated/merged.
                prior_report["resume_contract"] = recovered
        _validate_partial_resume(
            prior_report,
            partial_path=partial_path,
            selection_contract=selection_contract,
            archive_override=archive_override,
        )
        _validate_partial_config(
            prior_report,
            current_config={
                "window_seconds": float(window_seconds),
                "include_tail": include_tail,
                "frame_count": frame_count,
                "max_image_side": max_image_side,
                "max_new_tokens": max_new_tokens,
                "jpeg_quality": jpeg_quality,
                "proposal_index": proposal_index,
                "verdict_scope": verdict_scope,
                "include_optional_fields": include_optional_fields,
                "dry_run": dry_run,
                "mapping_config": str(mapping_config) if mapping_config is not None else None,
            },
            partial_path=partial_path,
        )

    # The default native verifier can expose a resident-runtime entry point.
    # Keep custom injected runners untouched (they are used by unit tests and
    # offline adapters), while the real Qwen path loads one model for the
    # entire recording batch.
    verifier = verifier_runner
    resident_runtime: Any | None = None
    resident_load_observation: Any | None = None
    resident_load_error: BaseException | None = None
    resident_runner: Callable[..., dict[str, Any]] | None = None
    runtime_scope = "dry_run" if dry_run else ("injected_runner" if verifier else "per_recording")
    if not dry_run and verifier is None:
        verifier = _native_verifier_run()
        candidate_resident_runner = getattr(verifier, "run_with_runtime", None)
        if callable(candidate_resident_runner):
            resident_runner = candidate_resident_runner
            # Importing this module is lazy and does not import torch.  The
            # model itself is loaded only for a non-dry run.
            from robata.inference.local_hf_runtime import LocalHuggingFaceVisionRuntime

            if model_directory is None:  # narrowed for callers/type checkers
                raise ProductionQwenAmbiguityBatchError(
                    "model_directory is required for a non-dry run"
                )
            model_dir = Path(model_directory).expanduser().resolve()
            resident_offload_dir = (
                Path(offload_directory).expanduser().resolve()
                if offload_directory is not None
                else output_dir / "qwen-offload" / "resident"
            )
            resident_runtime = LocalHuggingFaceVisionRuntime(
                model_directory=model_dir,
                offload_directory=resident_offload_dir,
                max_image_side=max_image_side,
                gpu_weight_memory_gib=gpu_weight_memory_gib,
                cpu_weight_memory_gib=cpu_weight_memory_gib,
            )
            runtime_scope = "resident_batch"

    config: dict[str, Any] = {
        "runtime_scope": runtime_scope,
        "window_seconds": float(window_seconds),
        "include_tail": include_tail,
        "frame_count": frame_count,
        "max_image_side": max_image_side,
        "max_new_tokens": max_new_tokens,
        "gpu_weight_memory_gib": gpu_weight_memory_gib,
        "cpu_weight_memory_gib": cpu_weight_memory_gib,
        "jpeg_quality": jpeg_quality,
        "proposal_index": proposal_index,
        "verdict_scope": verdict_scope,
        "include_optional_fields": include_optional_fields,
        "dry_run": dry_run,
        "resume": resume,
        "keep_staging": keep_staging,
        "model_directory": str(model_directory) if model_directory is not None else None,
        "offload_directory": str(offload_directory) if offload_directory is not None else None,
        "mapping_config": str(mapping_config) if mapping_config is not None else None,
    }
    report: dict[str, Any] = {
        "format": BATCH_FORMAT,
        "authority": AUTHORITY,
        "status": "RUNNING",
        "production_eligible": False,
        "quality_claim": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "official_quality_status": "NOT_MEASURED",
        "runtime_scope": runtime_scope,
        "runtime": {
            "scope": runtime_scope,
            "resident": runtime_scope == "resident_batch",
            "model_load_count": 0,
            "model_load_seconds": None,
        },
        "source": {
            "selection": selection_path,
            "selection_format": document.get("format"),
            "selection_summary": _copy_json(document.get("summary", {}), field="selection.summary"),
            "archive_override": str(archive_override) if archive_override else None,
            "source_manifest": (
                str(source_manifest) if isinstance(source_manifest, (str, Path)) else None
            ),
            "windows_are_action_segments": False,
            "selection_contract": selection_contract,
        },
        "resume_contract": selection_contract,
        "config": config,
        "recordings": [],
        "windows": [],
        "resume_history": [],
        "summary": {
            "selected_recording_count": len(grouped),
            "selected_window_count": len(rows),
            "succeeded_recording_count": 0,
            "failed_recording_count": 0,
            "planned_recording_count": 0,
            "verifier_row_count": 0,
            "failed_row_count": 0,
        },
        "controls": {
            "model_invoked": False,
            "qwen_invoked": False,
            "qwen_read": False,
            "model_invocation_scope": runtime_scope,
            "source_media_staged": False,
            "source_media_direct": False,
            "source_media_decoded": False,
            "native_video_materialized": False,
            "gold_read": False,
            "gold_written": False,
            "epic_ontology_read": False,
            "mapper_read": False,
            "mage_read": False,
            "hash_or_digest_computed": False,
            "heldout_100_opened": False,
        },
        "limitations": [
            "This is a non-production ambiguity review pass, not an annotation or quality claim.",
            (
                "Selected source intervals are processing context only; action "
                "boundaries are not inferred."
            ),
            "Qwen is constrained to the WeMM candidate set; it cannot create a new action label.",
            "A complete native six-camera input is materialized per recording before verification.",
        ],
    }

    if prior_report is not None:
        # Keep durable completed reports/rows intact.  Non-completed attempts
        # are replaced when retried below, with their prior payload retained in
        # a small history sidecar so an interruption never silently erases
        # evidence.
        prior_recordings = [
            dict(_copy_json(item, field="partial.recording"))
            for item in _sequence(prior_report.get("recordings", []), field="partial.recordings")
        ]
        prior_windows = [
            dict(_copy_json(item, field="partial.window"))
            for item in _sequence(prior_report.get("windows", []), field="partial.windows")
            if isinstance(item, Mapping)
        ]
        report["recordings"] = prior_recordings
        report["windows"] = prior_windows
        prior_history = prior_report.get("resume_history", [])
        if isinstance(prior_history, Sequence) and not isinstance(
            prior_history, (str, bytes, bytearray)
        ):
            report["resume_history"] = [
                _copy_json(item, field="partial.resume_history") for item in prior_history
            ]
        report["resume"] = {
            "requested": True,
            "resumed_from_partial": True,
            "partial_path": str(partial_path),
            "previous_status": prior_report.get("status"),
            "skipped_recording_ids": [],
        }
        # Preserve prior control observations while the current invocation
        # starts with its own runtime/resource flags.
        prior_controls = prior_report.get("controls")
        if isinstance(prior_controls, Mapping):
            for key, value in prior_controls.items():
                if key in report["controls"] and isinstance(value, bool):
                    report["controls"][key] = bool(report["controls"][key] or value)
    else:
        report["resume"] = {
            "requested": bool(resume),
            "resumed_from_partial": False,
            "partial_path": str(partial_path) if resume else None,
            "previous_status": None,
            "skipped_recording_ids": [],
        }

    _refresh_ambiguity_summary(
        report,
        selected_recording_count=len(grouped),
        selected_window_count=len(rows),
    )
    # Write an initial checkpoint before staging/model work.  A process that is
    # interrupted before the first recording still leaves a resumable,
    # selection-bound report.
    _write_json(partial_path, report)

    resume_skip_ids: set[str] = set()
    resume_prior_by_id: dict[str, dict[str, Any]] = {}
    if prior_report is not None:
        for raw in report["recordings"]:
            item = _mapping(raw, field="partial.recordings[]")
            rid = str(item.get("recording_id", ""))
            if rid:
                resume_prior_by_id[rid] = dict(item)
        # Rebuild the active arrays from the completed durable recordings only.
        # Failed/planned attempts are retained in ``resume_history`` when they
        # are retried, but their stale rows must not be counted twice.
        preserved_recordings: list[dict[str, Any]] = []
        preserved_windows: list[dict[str, Any]] = []
        for recording_id in sorted(grouped):
            old = resume_prior_by_id.get(recording_id)
            if old is None or str(old.get("status", "")).upper() not in _RESUME_SUCCESS_STATUSES:
                if old is not None:
                    report["resume_history"].append(
                        {
                            "recording_id": recording_id,
                            "status": old.get("status"),
                            "recording": old,
                        }
                    )
                continue
            expected_ids = [str(row["window_id"]) for row in grouped[recording_id]]
            prior_ids = [str(value) for value in old.get("selected_window_ids", [])]
            if prior_ids and prior_ids != expected_ids:
                raise ProductionQwenAmbiguityBatchError(
                    "partial report recording window selection differs for "
                    f"{recording_id}; use a new output directory or rerun without --resume"
                )
            old["resume_skipped"] = True
            preserved_recordings.append(old)
            resume_skip_ids.add(recording_id)
            old_rows = old.get("rows", [])
            if isinstance(old_rows, Sequence) and not isinstance(old_rows, (str, bytes, bytearray)):
                preserved_windows.extend(
                    dict(_copy_json(row, field="partial.completed_window"))
                    for row in old_rows
                    if isinstance(row, Mapping)
                )
            else:
                # Older checkpoints may only have the flattened rows array.
                preserved_windows.extend(
                    dict(_copy_json(row, field="partial.completed_window"))
                    for row in report["windows"]
                    if isinstance(row, Mapping) and str(row.get("recording_id", "")) == recording_id
                )
        report["recordings"] = preserved_recordings
        report["windows"] = preserved_windows
        report["resume"]["skipped_recording_ids"] = sorted(resume_skip_ids)
        _refresh_ambiguity_summary(
            report,
            selected_recording_count=len(grouped),
            selected_window_count=len(rows),
        )
        _write_json(partial_path, report)

    def _invoke_verifier(verifier_args: argparse.Namespace) -> dict[str, Any]:
        """Dispatch one shard, reusing the resident runtime when available."""

        nonlocal resident_load_observation, resident_load_error
        if resident_runner is None or resident_runtime is None:
            if verifier is None:  # pragma: no cover - guarded by setup above
                raise ProductionQwenAmbiguityBatchError("verifier runner is unavailable")
            return verifier(verifier_args)
        if resident_load_error is not None:
            raise ProductionQwenAmbiguityBatchError(
                f"resident Qwen runtime failed to load: {resident_load_error}"
            ) from resident_load_error
        if resident_load_observation is None:
            try:
                resident_load_observation = resident_runtime.load()
                report["runtime"]["model_load_count"] = 1
                report["runtime"]["model_load_seconds"] = getattr(
                    resident_load_observation, "load_seconds", None
                )
            except Exception as exc:
                resident_load_error = exc
                raise
        return resident_runner(
            verifier_args,
            resident_runtime,
            load_observation=resident_load_observation,
        )

    def _close_resident_runtime() -> None:
        """Release the one batch-owned runtime after all recordings finish."""

        nonlocal resident_runtime
        if resident_runtime is not None:
            resident_runtime.close()
            resident_runtime = None

    try:
        for recording_index, recording_id in enumerate(sorted(grouped)):
            recording_rows = grouped[recording_id]
            if recording_id in resume_skip_ids:
                # The complete recording and all of its verifier rows were
                # loaded from the validated checkpoint above; do not stage
                # media or invoke Qwen again.
                continue
            started = time.perf_counter()
            rec: dict[str, Any] = {
                "recording_id": recording_id,
                "status": "PLANNED",
                "source_mode": None,
                "selected_window_ids": [str(row["window_id"]) for row in recording_rows],
                "selected_window_count": len(recording_rows),
                "manifest_path": None,
                "candidate_path": None,
                "bridge_plan_path": None,
                "verifier_path": None,
                "video_root": None,
                "archive_path": None,
                "archive_member": None,
                "source_path": None,
                "rows": [],
                "error": None,
            }
            previous_attempt = resume_prior_by_id.get(recording_id)
            if previous_attempt is not None:
                previous_status = str(previous_attempt.get("status", "")).upper()
                if previous_status not in _RESUME_SUCCESS_STATUSES:
                    rec["previous_attempt"] = previous_attempt
            try:
                source_plan = _recording_plan(
                    recording_rows,
                    archive_path=archive_override,
                    source_manifest=source_doc,
                    selection_base=selection_base,
                )
                rid = source_plan.recording_id
                member = source_plan.archive_member
                preflight_status = source_plan.source_preflight_status
                archive = source_plan.archive_path
                qa_status = source_plan.qa_status
                slug = source_plan.slug
                rec.update(
                    {
                        "recording_id": rid,
                        "source_mode": source_plan.source_mode,
                        "archive_path": str(archive) if archive is not None else None,
                        "archive_member": member,
                        "source_path": (
                            str(source_plan.source_path)
                            if source_plan.source_path is not None
                            else None
                        ),
                        "source_preflight_status": preflight_status,
                        "qa_status": qa_status,
                    }
                )
                if source_plan.source_mode == "ARCHIVE_MEMBER":
                    if archive is None:
                        raise ProductionQwenAmbiguityBatchError(
                            f"no source archive path for recording {recording_id}"
                        )
                    if not archive.is_file():
                        raise ProductionQwenAmbiguityBatchError(
                            f"source archive does not exist: {archive}"
                        )
                else:
                    direct_source = source_plan.source_path
                    if direct_source is None or not direct_source.is_file():
                        raise ProductionQwenAmbiguityBatchError(
                            f"direct MCAP source does not exist: {direct_source}"
                        )
                if preflight_status not in {"PASS", "WARNING"}:
                    raise ProductionQwenAmbiguityBatchError(
                        f"source preflight status {preflight_status!r} is not runnable"
                    )

                candidate_path = candidates_dir / f"{slug}.json"
                manifest_path = manifests_dir / f"{slug}.json"
                bridge_plan_path = bridge_dir / f"{slug}.json"
                verifier_path = verifier_dir / f"{slug}.json"
                native_root = video_dir / slug
                rec.update(
                    {
                        "candidate_path": str(candidate_path),
                        "manifest_path": str(manifest_path),
                        "bridge_plan_path": str(bridge_plan_path),
                        "verifier_path": str(verifier_path),
                        "video_root": str(native_root),
                    }
                )
                if source_plan.source_mode == "ARCHIVE_MEMBER":
                    stage_context = stage_zip_member(
                        archive,
                        member,
                        staging_root,
                        ordinal=recording_index,
                        keep_staged=keep_staging,
                    )
                else:
                    # A direct local MCAP is already at its source path.  The
                    # no-op context preserves the existing lifecycle while
                    # avoiding a fabricated ZIP member or a second copy.
                    stage_context = nullcontext(source_plan.source_path)
                with stage_context as staged:
                    if source_plan.source_mode == "ARCHIVE_MEMBER":
                        report["controls"]["source_media_staged"] = True
                    else:
                        report["controls"]["source_media_direct"] = True
                    direct_source_for_manifest = source_plan.source_path
                    if member is None and direct_source_for_manifest is None:
                        raise ProductionQwenAmbiguityBatchError(
                            f"recording {rid} has no direct MCAP source path"
                        )
                    manifest_member = member
                    if manifest_member is None:
                        assert direct_source_for_manifest is not None
                        manifest_member = f"direct/{direct_source_for_manifest.name}"
                    manifest = build_recording_manifest(
                        staged,
                        recording_id=rid,
                        # The legacy manifest builder requires a syntactically
                        # valid member name for its internal source projection.
                        # For direct MCAP we immediately replace that field
                        # with null below; no archive identity is fabricated.
                        archive_member=manifest_member,
                        source_preflight_status=preflight_status,
                        qa_status=qa_status,
                        window_seconds=window_seconds,
                        include_tail=include_tail,
                    )
                    source = dict(_mapping(manifest.get("source"), field="manifest.source"))
                    source.update(
                        {
                            "archive_path": str(archive) if archive is not None else None,
                            "archive_member": member,
                            "source_path": (
                                str(source_plan.source_path)
                                if source_plan.source_path is not None
                                else None
                            ),
                            "path_lifecycle": (
                                "DIRECT_MCAP_READ_ONLY"
                                if source_plan.source_mode == "DIRECT_MCAP"
                                else "STAGED_PATH_REMOVED_AFTER_RECORDING"
                            ),
                        }
                    )
                    manifest["source"] = source
                    # Keep every selected window's interval as context.  The
                    # generated manifest's action_boundary flags are already false.
                    _write_json(manifest_path, manifest)
                    candidate = _candidate_document(
                        recording_rows,
                        recording_id=rid,
                        archive_path=archive,
                        archive_member=member,
                        source_path=source_plan.source_path,
                    )
                    _write_json(candidate_path, candidate)

                    if dry_run:
                        bridge = build_qwen_native_video_plan(
                            staged,
                            native_root,
                            mapping_config=mapping_config,
                            allow_unapproved_profile=allow_unapproved_profile,
                        )
                        _write_json(bridge_plan_path, bridge)
                        rec["status"] = "PLANNED"
                        rec["rows"] = _failure_rows(
                            recording_rows,
                            recording_id=rid,
                            error="DRY_RUN_NOT_INVOKED",
                            status="PLANNED",
                        )
                        report["summary"]["planned_recording_count"] += 1
                    else:
                        native_result = materialize_qwen_native_video_inputs(
                            staged,
                            native_root,
                            mapping_config=mapping_config,
                            allow_unapproved_profile=allow_unapproved_profile,
                            dry_run=False,
                        )
                        report["controls"]["native_video_materialized"] = True
                        _write_json(bridge_plan_path, native_result.manifest)
                        if model_directory is None:  # narrowed for callers/type checkers
                            raise ProductionQwenAmbiguityBatchError(
                                "model_directory is required for a non-dry run"
                            )
                        model_dir = Path(model_directory).expanduser().resolve()
                        offload_dir = (
                            Path(offload_directory).expanduser().resolve()
                            if offload_directory is not None
                            else output_dir / "qwen-offload" / slug
                        )
                        args = _verifier_args(
                            candidates=candidate_path,
                            manifest=manifest_path,
                            video_root=native_root,
                            model_dir=model_dir,
                            offload_dir=offload_dir,
                            window_ids=[str(row["window_id"]) for row in recording_rows],
                            frame_count=frame_count,
                            max_image_side=max_image_side,
                            max_new_tokens=max_new_tokens,
                            gpu_weight_memory_gib=gpu_weight_memory_gib,
                            cpu_weight_memory_gib=cpu_weight_memory_gib,
                            jpeg_quality=jpeg_quality,
                            proposal_index=proposal_index,
                            verdict_scope=verdict_scope,
                            include_optional_fields=include_optional_fields,
                            output=verifier_path,
                        )
                        verifier_report = _invoke_verifier(args)
                        if not isinstance(verifier_report, Mapping):
                            raise ProductionQwenAmbiguityBatchError(
                                "native verifier returned a non-object report"
                            )
                        _write_json(verifier_path, verifier_report)
                        verifier_rows = verifier_report.get("windows", [])
                        if not isinstance(verifier_rows, Sequence) or isinstance(
                            verifier_rows, (str, bytes, bytearray)
                        ):
                            raise ProductionQwenAmbiguityBatchError(
                                "native verifier returned a non-array windows field"
                            )
                        rec["rows"] = []
                        for raw_row in verifier_rows:
                            if not isinstance(raw_row, Mapping):
                                continue
                            verifier_row = dict(_copy_json(raw_row, field="verifier.window"))
                            verifier_row.setdefault("recording_id", rid)
                            verifier_row["source_context_is_action_boundary"] = False
                            rec["rows"].append(verifier_row)
                        # A verifier may return a partial camera set after a
                        # decode/runtime error.  Make every omitted selected
                        # window/camera visible instead of silently dropping it.
                        observed_keys = {
                            (str(raw.get("window_id")), str(raw.get("camera_id")))
                            for raw in rec["rows"]
                            if isinstance(raw, Mapping)
                        }
                        missing_rows: list[dict[str, Any]] = []
                        for selected_row in recording_rows:
                            window_id = str(selected_row.get("window_id"))
                            for camera_id in _camera_ids(selected_row):
                                if (window_id, camera_id) in observed_keys:
                                    continue
                                missing_rows.extend(
                                    item
                                    for item in _failure_rows(
                                        [selected_row],
                                        recording_id=rid,
                                        error="VERIFIER_ROW_MISSING",
                                        native_video_complete=True,
                                    )
                                    if str(item.get("camera_id")) == camera_id
                                )
                        rec["rows"].extend(missing_rows)
                        report["controls"]["model_invoked"] = True
                        report["controls"]["qwen_invoked"] = True
                        report["controls"]["qwen_read"] = True
                        report["controls"]["source_media_decoded"] = True
                        # A recording is resumable-successful only when every
                        # selected window/camera row completed.  The verifier
                        # can legitimately return a partial set after a
                        # decode/runtime error; those synthesized FAILED rows
                        # must keep the recording retryable instead of being
                        # hidden behind a top-level COMPLETE status.
                        row_failures = sum(
                            1
                            for row in rec["rows"]
                            if isinstance(row, Mapping)
                            and str(row.get("status", "")).upper() == "FAILED"
                        )
                        if row_failures:
                            rec["status"] = "PARTIAL"
                            report["summary"]["failed_recording_count"] += 1
                        else:
                            rec["status"] = "SUCCEEDED"
                            report["summary"]["succeeded_recording_count"] += 1
            except Exception as exc:
                rec["status"] = "FAILED"
                rec["error"] = {"type": type(exc).__name__, "detail": str(exc)}
                rec["rows"] = _failure_rows(
                    recording_rows,
                    recording_id=recording_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                report["summary"]["failed_recording_count"] += 1
            rec["elapsed_seconds"] = time.perf_counter() - started
            report["recordings"].append(rec)
            report["windows"].extend(rec["rows"])

            # Persist a lightweight progress checkpoint after every recording.
            # The source MCAP is staged one-at-a-time while the resident Qwen
            # runtime remains owned by this batch, so an interruption must not
            # discard already completed rows.  This is a progress sidecar only (no
            # identity/digest computation and no change to the production wire
            # contract); the final report below remains the authoritative batch
            # summary.
            _refresh_ambiguity_summary(
                report,
                selected_recording_count=len(grouped),
                selected_window_count=len(rows),
            )
            report["status"] = "RUNNING"
            _write_json(partial_path, report)

    finally:
        _close_resident_runtime()

    _refresh_ambiguity_summary(
        report,
        selected_recording_count=len(grouped),
        selected_window_count=len(rows),
    )
    if dry_run and report["summary"]["failed_recording_count"] == 0:
        report["status"] = "DRY_RUN"
    elif report["summary"]["failed_recording_count"] == 0:
        report["status"] = "COMPLETE"
    else:
        report["status"] = "PARTIAL"
    # Keep the progress sidecar in sync with the final report.  It remains a
    # valid resume source (a subsequent explicit ``--resume`` simply skips all
    # successful recordings) and avoids leaving a misleading RUNNING status.
    _write_json(partial_path, report)
    # Keep one obvious durable artifact in the caller's output root.  This is
    # a report sidecar only; no image payloads or derived identities are added.
    _write_json(output_dir / "batch-report.json", report)
    return report


__all__ = [
    "AMBIGUITY_SELECTION_FORMAT",
    "AUTHORITY",
    "BATCH_FORMAT",
    "DEFAULT_EXPECTED_CAMERA_IDS",
    "ProductionQwenAmbiguityBatchError",
    "run_production_qwen_ambiguity_batch",
]
