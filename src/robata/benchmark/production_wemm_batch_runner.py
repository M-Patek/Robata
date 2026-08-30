"""Serial, resumable WeMM pre-annotation runner for the production archive.

The source archive is intentionally kept separate from the EPIC benchmark.  A
preflight artifact identifies structurally readable MCAP members; this module
selects only those members, extracts one member at a time to a bounded staging
directory, builds a label-neutral production cohort manifest, and invokes the
open-phrase WeMM runner.  Outputs are review-only envelopes and are never
treated as gold or as an approved production vocabulary.

The runner is serial by default.  This keeps peak disk use to one MCAP, allows a
single resident WeMM backend to be reused across recordings, and leaves a
checkpoint after every recording so a long run can be resumed without
re-running completed items.  An explicit ``include_pipeline`` option can
overlap bounded decode and model work *within* one recording; recording order
and checkpoint semantics remain serial.  No archive hashes, digests, EPIC
identities, Mapper calls, Qwen/Mage calls, or pixels are persisted by this
module.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Final

from .production_cohort import ProductionCohortError, build_manifest
from .production_wemm_open_runner import (
    DEFAULT_QUEUE_CAPACITY,
    run_production_wemm_open,
)
from .production_wemm_preannotation import build_review_pack
from .production_wemm_temporal import (
    DEFAULT_SCORE_POLICY,
    MODE_DENSE_SCORE,
    MODE_NONE,
    SCORE_POLICIES,
)
from .wemm_embedding_backend import WemmEmbeddingBackend

BATCH_RUN_FORMAT: Final = "robata-production-wemm-batch-run-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
RUN_STATUS_RUNNING: Final = "RUNNING"
RUN_STATUS_COMPLETE: Final = "COMPLETE"
RUN_STATUS_PARTIAL: Final = "PARTIAL"
RUN_STATUS_DRY_RUN: Final = "DRY_RUN"
RUN_STATUS_INTERRUPTED: Final = "INTERRUPTED"
DEFAULT_WINDOW_SECONDS: Final = 8.0
DEFAULT_FRAME_COUNT: Final = 4
DEFAULT_TOP_K: Final = 10
DEFAULT_DIMENSION: Final = 256
DEFAULT_WINDOW_CHUNK_SIZE: Final = 1
# Keep the production runner's historical singleton path by default. Batch2
# and Batch4 are opt-in scheduling choices that reuse the resident backend
# without changing the review envelope or source-bound recording lifecycle.
DEFAULT_INFERENCE_BATCH_SIZE: Final = 1
DEFAULT_COPY_CHUNK_BYTES: Final = 8 * 1024 * 1024
DEFAULT_TEMPORAL_STRIDE_DIVISOR: Final = 4


class ProductionWemmBatchRunnerError(RuntimeError):
    """Raised when a serial production run cannot be prepared."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmBatchRunnerError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmBatchRunnerError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionWemmBatchRunnerError(f"{field} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ProductionWemmBatchRunnerError(f"{field} must be non-empty")
    return result


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ProductionWemmBatchRunnerError(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionWemmBatchRunnerError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ProductionWemmBatchRunnerError(f"{field} must be finite")
    return result


def _json_copy(value: object, *, field: str = "value") -> Any:
    """Copy bounded JSON metadata without deriving an identity or digest."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmBatchRunnerError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmBatchRunnerError(f"{field} keys must be strings")
            result[key] = _json_copy(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_copy(child, field=f"{field}[]") for child in value]
    raise ProductionWemmBatchRunnerError(f"{field} must be JSON-compatible")


def _load_json(value: Mapping[str, Any] | Sequence[Any] | str | Path) -> Any:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductionWemmBatchRunnerError(f"could not read JSON {path}: {exc}") from exc
    return value


def _normalise_member(value: object, *, field: str = "archive_member") -> str:
    member = _text(value, field=field).replace("\\", "/")
    if member.startswith("/") or "://" in member:
        raise ProductionWemmBatchRunnerError(f"{field} must be archive-relative")
    parts = [part for part in member.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ProductionWemmBatchRunnerError(f"{field} contains an unsafe path")
    normalised = "/".join(parts)
    if not normalised.casefold().endswith(".mcap"):
        raise ProductionWemmBatchRunnerError(f"{field} must point to an .mcap member")
    return normalised


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return token[:80] or "recording"


def _resolve_path(value: str | Path, *, root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.resolve()


def load_source_preflight(
    value: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Load a source-preflight artifact and validate its bounded metadata.

    The returned document contains only fields required by the batch runner;
    arbitrary sidecar content is not copied into outputs.  ``source_preflight``
    status is distinct from clip-level visual QA and remains visible in every
    item result.
    """

    document = _load_json(value)
    body = dict(_mapping(document, field="source_preflight"))
    source = _mapping(body.get("source"), field="source_preflight.source")
    archive_path = _text(source.get("archive_path"), field="source_preflight.source.archive_path")
    raw_items = _sequence(body.get("items"), field="source_preflight.items")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        row = _mapping(raw, field=f"source_preflight.items[{index}]")
        raw_member = row.get("name", row.get("archive_member"))
        member = _normalise_member(raw_member, field=f"items[{index}].name")
        if member in seen:
            raise ProductionWemmBatchRunnerError(
                f"source_preflight contains duplicate archive member: {member}"
            )
        seen.add(member)
        status = str(row.get("source_preflight_status", row.get("status", "PENDING"))).upper()
        if status not in {"PASS", "FAIL", "WARNING", "PENDING"}:
            raise ProductionWemmBatchRunnerError(
                f"items[{index}].source_preflight_status has unknown value {status!r}"
            )
        qa_status = str(row.get("qa_status", "PENDING")).upper()
        if qa_status not in {"PASS", "WARNING", "FAIL", "PENDING"}:
            raise ProductionWemmBatchRunnerError(
                f"items[{index}].qa_status has unknown value {qa_status!r}"
            )
        ordinal_raw = row.get("ordinal", index)
        try:
            ordinal = int(ordinal_raw)
        except (TypeError, ValueError) as exc:
            raise ProductionWemmBatchRunnerError(
                f"items[{index}].ordinal must be an integer"
            ) from exc
        if ordinal < 0:
            raise ProductionWemmBatchRunnerError(f"items[{index}].ordinal must be non-negative")
        size = row.get("size_bytes")
        if size is not None:
            try:
                size = int(size)
            except (TypeError, ValueError) as exc:
                raise ProductionWemmBatchRunnerError(
                    f"items[{index}].size_bytes must be an integer"
                ) from exc
            if size < 0:
                raise ProductionWemmBatchRunnerError(
                    f"items[{index}].size_bytes must be non-negative"
                )
        duration = row.get("duration_seconds")
        if duration is not None:
            duration = _finite(duration, field=f"items[{index}].duration_seconds")
            if duration < 0:
                raise ProductionWemmBatchRunnerError(
                    f"items[{index}].duration_seconds must be non-negative"
                )
        item: dict[str, Any] = {
            "ordinal": ordinal,
            "archive_member": member,
            "size_bytes": size,
            "source_preflight_status": status,
            "qa_status": qa_status,
            "batch": row.get("batch"),
            "source_preflight_reason": row.get("source_preflight_reason"),
            "duration_seconds": duration,
            "camera_count": row.get("camera_count"),
            "camera_frames_total": row.get("camera_frames_total"),
        }
        items.append(item)
    items.sort(key=lambda row: (int(row["ordinal"]), str(row["archive_member"])))
    return {
        "format": str(body.get("format", "robata-production-source-preflight-v1")),
        "archive_path": archive_path,
        "items": items,
        "counts": _json_copy(body.get("counts", {}), field="source_preflight.counts"),
        "status": str(body.get("status", "UNKNOWN")),
    }


def select_preflight_items(
    preflight: Mapping[str, Any] | str | Path,
    *,
    status: str = "PASS",
    batch: str | None = None,
    ordinals: Sequence[int] | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Select source-preflight items without treating them as action QA."""

    # ``run_production_wemm_batch`` passes the already-normalised loader
    # result; callers may also pass the original artifact/path.
    if isinstance(preflight, Mapping) and "archive_path" in preflight and "items" in preflight:
        document = dict(preflight)
    else:
        document = load_source_preflight(preflight)
    wanted_status = _text(status, field="status").upper()
    if wanted_status not in {"PASS", "WARNING"}:
        raise ProductionWemmBatchRunnerError("status must be PASS or WARNING")
    wanted_ordinals: set[int] | None = None
    if ordinals is not None:
        wanted_ordinals = set()
        for ordinal in ordinals:
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
                raise ProductionWemmBatchRunnerError("ordinals must be non-negative integers")
            wanted_ordinals.add(ordinal)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ProductionWemmBatchRunnerError("limit must be a positive integer")
    selected: list[dict[str, Any]] = []
    for raw in document["items"]:
        item = dict(_mapping(raw, field="preflight.items[]"))
        if item["source_preflight_status"] != wanted_status:
            continue
        # Source preflight PASS is only a structural gate.  A separately
        # supplied clip-level QA FAIL must still be excluded; PENDING remains
        # runnable as a review-only, non-gold pre-annotation.
        if item.get("qa_status") == "FAIL":
            continue
        if batch is not None and item.get("batch") != batch:
            continue
        if wanted_ordinals is not None and int(item["ordinal"]) not in wanted_ordinals:
            continue
        selected.append(item)
    if wanted_ordinals is not None:
        found = {int(item["ordinal"]) for item in selected}
        missing = sorted(wanted_ordinals - found)
        if missing:
            raise ProductionWemmBatchRunnerError(
                "requested ordinals are not eligible source-preflight items: "
                + ", ".join(str(item) for item in missing)
            )
    if limit is not None:
        selected = selected[:limit]
    return tuple(selected)


def _safe_destination(root: Path, member: str, ordinal: int) -> Path:
    root_resolved = root.resolve()
    target_dir = (root_resolved / f"recording-{ordinal:04d}").resolve()
    if not target_dir.is_relative_to(root_resolved):
        raise ProductionWemmBatchRunnerError("staging directory escaped its root")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = (target_dir / Path(member).name).resolve()
    if not target.is_relative_to(target_dir):
        raise ProductionWemmBatchRunnerError("staging destination escaped its recording directory")
    return target


@contextmanager
def stage_zip_member(
    archive_path: str | Path,
    archive_member: str,
    staging_root: str | Path,
    *,
    ordinal: int = 0,
    chunk_bytes: int = DEFAULT_COPY_CHUNK_BYTES,
    keep_staged: bool = False,
) -> Iterator[Path]:
    """Stream one MCAP member into a safe temporary staging directory.

    ``ZipFile.extract`` is intentionally not used.  The member is validated as
    a normal archive-relative ``.mcap`` path and copied through a temporary
    ``.part`` file before being made visible to the decoder.  Only one member
    is materialised at a time.
    """

    member = _normalise_member(archive_member)
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ProductionWemmBatchRunnerError("chunk_bytes must be positive")
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise ProductionWemmBatchRunnerError(f"source archive does not exist: {archive}")
    root = Path(staging_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = _safe_destination(root, member, ordinal)
    partial = target.with_name(target.name + ".part")
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            try:
                info = zf.getinfo(member)
            except KeyError as exc:
                raise ProductionWemmBatchRunnerError(
                    f"archive member is missing: {member}"
                ) from exc
            if info.is_dir():
                raise ProductionWemmBatchRunnerError(f"archive member is a directory: {member}")
            with zf.open(info, "r") as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=chunk_bytes)
        partial.replace(target)
        yield target
    finally:
        with suppress(OSError):
            partial.unlink(missing_ok=True)
        if not keep_staged:
            # Remove only this recording directory, never an arbitrary caller
            # path.  The root itself is retained for the caller's lifecycle.
            recording_dir = target.parent
            with suppress(OSError):
                if recording_dir.exists() and recording_dir.is_relative_to(root):
                    shutil.rmtree(recording_dir, ignore_errors=True)


def build_recording_manifest(
    source_path: str | Path,
    *,
    recording_id: str,
    archive_member: str,
    source_preflight_status: str,
    qa_status: str = "PENDING",
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    include_tail: bool = True,
    window_stride_seconds: float | None = None,
) -> dict[str, Any]:
    """Build and annotate a source-bound processing-window manifest."""

    recording_id = _text(recording_id, field="recording_id")
    member = _normalise_member(archive_member)
    status = _text(source_preflight_status, field="source_preflight_status").upper()
    if status not in {"PASS", "WARNING"}:
        raise ProductionWemmBatchRunnerError(
            "source_preflight_status must be PASS or WARNING for execution"
        )
    qa_status = _text(qa_status, field="qa_status").upper()
    if qa_status not in {"PASS", "WARNING", "PENDING"}:
        raise ProductionWemmBatchRunnerError(
            "qa_status must be PASS, WARNING or PENDING for execution"
        )
    try:
        manifest_kwargs: dict[str, Any] = {
            "window_seconds": window_seconds,
            "include_tail": include_tail,
        }
        if window_stride_seconds is not None:
            manifest_kwargs["window_stride_seconds"] = window_stride_seconds
        raw_manifest = build_manifest(source_path, **manifest_kwargs)
    except (ProductionCohortError, OSError) as exc:
        raise ProductionWemmBatchRunnerError(
            f"could not build manifest for {recording_id}: {exc}"
        ) from exc
    manifest = dict(_mapping(raw_manifest, field="manifest"))
    source = dict(_mapping(manifest.get("source"), field="manifest.source"))
    source.update(
        {
            "recording_id": recording_id,
            "archive_member": member,
            "source_preflight_status": status,
            "qa_status": qa_status,
            "staging_lifecycle": "temporary_single_recording",
        }
    )
    manifest["source"] = source
    manifest["recording_id"] = recording_id
    manifest["window_policy"] = {
        **dict(_mapping(manifest.get("window_policy"), field="manifest.window_policy")),
        "window_semantics": "PROCESSING_WINDOW_NOT_ACTION_BOUNDARY",
        "action_boundaries_inferred": False,
    }
    windows = _sequence(manifest.get("windows"), field="manifest.windows")
    renamed: list[dict[str, Any]] = []
    for index, raw in enumerate(windows):
        window = dict(_mapping(raw, field=f"manifest.windows[{index}]"))
        window["window_id"] = f"{recording_id}-w{index:04d}"
        window["processing_window"] = True
        window["action_boundary"] = False
        renamed.append(window)
    manifest["windows"] = renamed
    controls = dict(_mapping(manifest.get("controls"), field="manifest.controls"))
    controls.update(
        {
            "frames_decoded": False,
            "model_invoked": False,
            "gold_read": False,
            "gold_written": False,
            "epic_ontology_used": False,
            "mapper_used": False,
            "hash_or_sha_used": False,
        }
    )
    manifest["controls"] = controls
    return manifest


def _recording_id(item: Mapping[str, Any]) -> str:
    ordinal = int(item["ordinal"])
    member = str(item["archive_member"])
    stem = Path(member).stem
    return f"production-{ordinal:03d}-{_slug(stem)}"


def _estimate_windows(
    duration_seconds: object,
    window_seconds: float,
    include_tail: bool,
    window_stride_seconds: float | None = None,
) -> int:
    """Estimate processing-window count from source preflight duration."""

    if duration_seconds is None:
        return 0
    try:
        if not isinstance(duration_seconds, (int, float, str)):
            return 0
        duration = float(duration_seconds)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(duration) or duration <= 0:
        return 0
    stride = window_seconds if window_stride_seconds is None else float(window_stride_seconds)
    if abs(stride - window_seconds) <= 1e-9:
        full = int(duration // window_seconds)
        if include_tail and duration - full * window_seconds > 1e-6:
            full += 1
        return full
    epsilon = 1e-9
    max_full_start = max(0.0, duration - window_seconds)
    count = 0
    last_full_start: float | None = None
    left = 0.0
    while left <= max_full_start + epsilon:
        if duration - left >= window_seconds - epsilon:
            count += 1
            last_full_start = left
        left += stride
    if include_tail:
        if last_full_start is None:
            # Match build_windows: one short tail for a source shorter than a
            # full context, rather than one tail per stride.
            count = 1
        else:
            tail_start = max(0.0, duration - window_seconds)
            if tail_start > last_full_start + epsilon:
                count += 1
    return count


def _temporal_resume_signature(
    config: Mapping[str, Any], *, default_mode: str | None = None
) -> tuple[object, ...]:
    """Return the explicit temporal settings that affect a resumable run.

    Older checkpoints predate temporal mode and therefore have none of these
    fields.  Treat that shape as the historical ``none`` mode, while refusing
    to reuse completed items when a dense run's stride or resolver parameters
    differ.  This is a direct field comparison; no identity hash or digest is
    introduced into the checkpoint contract.
    """

    mode_value = config.get("temporal_mode", default_mode)
    if mode_value is None:
        mode_value = (
            MODE_DENSE_SCORE
            if any(
                key in config
                for key in (
                    "temporal_start_threshold",
                    "temporal_stop_threshold",
                    "temporal_merge_gap_seconds",
                    "temporal_min_duration_seconds",
                    "temporal_min_camera_support",
                    "temporal_boundary_mode",
                    "temporal_score_policy",
                )
            )
            else MODE_NONE
        )

    def _numeric(key: str) -> object:
        value = config.get(key)
        if value is None:
            return None
        if isinstance(value, bool):
            return ("invalid", value)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return ("invalid", str(value))
        return number if math.isfinite(number) else ("invalid", str(value))

    stride = _numeric("window_stride_seconds")
    if mode_value == MODE_NONE:
        return (MODE_NONE, stride)
    if mode_value != MODE_DENSE_SCORE:
        return ("invalid", str(mode_value))
    return (
        MODE_DENSE_SCORE,
        stride,
        _numeric("temporal_start_threshold"),
        _numeric("temporal_stop_threshold"),
        _numeric("temporal_merge_gap_seconds"),
        _numeric("temporal_min_duration_seconds"),
        _numeric("temporal_min_camera_support"),
        config.get("temporal_boundary_mode"),
        config.get("temporal_score_policy"),
    )


def _empty_run(
    *,
    preflight: Mapping[str, Any],
    archive_path: Path,
    selected: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    configured_stride = config.get("window_stride_seconds")
    # ``config`` is assembled by ``run_production_wemm_batch`` after validating
    # the stride.  Keep this small type guard for resumed/externally composed
    # callers so a malformed checkpoint cannot turn the dry-run estimate into a
    # surprising exception or silently fall back to non-overlapping windows.
    window_seconds = float(config.get("window_seconds", DEFAULT_WINDOW_SECONDS))
    window_stride_seconds: float | None = None
    if (
        isinstance(configured_stride, (int, float))
        and not isinstance(configured_stride, bool)
        and math.isfinite(float(configured_stride))
        and 0.0 < float(configured_stride) <= window_seconds
    ):
        window_stride_seconds = float(configured_stride)
    estimated_windows = sum(
        _estimate_windows(
            raw.get("duration_seconds"),
            window_seconds,
            bool(config.get("include_tail", True)),
            window_stride_seconds,
        )
        for raw in selected
    )
    return {
        "format": BATCH_RUN_FORMAT,
        "authority": AUTHORITY,
        "status": RUN_STATUS_RUNNING,
        "production_eligible": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "source": {
            "archive_path": str(archive_path),
            "preflight_format": preflight.get("format"),
            "preflight_status": preflight.get("status"),
            "preflight_pass_selected": len(selected),
            "source_preflight_is_not_visual_qa": True,
            "qa_status": "PENDING_OR_UNESTABLISHED",
        },
        "config": dict(config),
        "controls": {
            "epic_ontology_used": False,
            "mapper_used": False,
            "qwen_used": False,
            "mage_used": False,
            "training_invoked": False,
            "gold_read": False,
            "gold_written": False,
            "hash_or_sha_used": False,
            "serial_source_bound": True,
            "backend_reused": bool(config.get("reuse_backend", True)),
        },
        "checkpoint_path": str(checkpoint_path),
        "summary": {
            "selected_count": len(selected),
            "planned_count": 0,
            "running_count": 0,
            "complete_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "window_count": 0,
            "estimated_window_count": estimated_windows,
            "estimated_camera_window_inputs": estimated_windows * 6,
            "quality_status": "NOT_MEASURED",
        },
        "items": [],
        "limitations": [
            "Source preflight PASS means structural readability only, not visual QA or gold.",
            "Processing windows are bounded compute/review units, not inferred action boundaries.",
            "All envelopes remain review-only until source-bound human gold exists.",
        ],
    }


def _write_checkpoint(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    with suppress(OSError):
        path.with_suffix(".md").write_text(render_batch_run_markdown(report), encoding="utf-8")


def render_batch_run_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact progress report alongside the JSON checkpoint."""

    source = _mapping(report.get("source"), field="run.source")
    summary = _mapping(report.get("summary"), field="run.summary")
    config = _mapping(report.get("config"), field="run.config")
    inference_batch_size = config.get("inference_batch_size", DEFAULT_INFERENCE_BATCH_SIZE)
    lines = [
        "# Production WeMM source-bound batch run",
        "",
        f"- Status: **{report.get('status', 'UNKNOWN')}**",
        f"- Archive: `{source.get('archive_path', '')}`",
        f"- Selected source-preflight PASS recordings: **{summary.get('selected_count', 0)}**",
        f"- Inference microbatch: **{inference_batch_size}**",
        (
            f"- Complete: {summary.get('complete_count', 0)}; "
            f"failed: {summary.get('failed_count', 0)}; "
            f"skipped: {summary.get('skipped_count', 0)}"
        ),
        (
            f"- Windows completed: {summary.get('window_count', 0)}; "
            f"estimated: {summary.get('estimated_window_count', 0)}"
        ),
        "- Quality: **NOT_MEASURED** (source-preflight PASS is not visual QA or gold)",
        "- Production eligible: **false**",
        "",
        "| Ordinal | Source member | Status | Windows | Elapsed (s) | Output |",
        "|---:|---|---|---:|---:|---|",
    ]
    if config.get("include_pipeline") is True:
        lines.insert(
            5,
            "- Producer/consumer pipeline: **enabled** "
            f"(queue capacity {config.get('queue_capacity', DEFAULT_QUEUE_CAPACITY)})",
        )
    for raw in _sequence(report.get("items", []), field="run.items"):
        item = _mapping(raw, field="run.items[]")
        member = str(item.get("archive_member", "")).replace("|", "\\|")
        output = str(item.get("preannotation_path") or "")
        elapsed = item.get("elapsed_seconds")
        elapsed_text = "" if elapsed is None else f"{float(elapsed):.1f}"
        lines.append(
            f"| {item.get('ordinal', '')} | `{member}` | {item.get('status', '')} | "
            f"{item.get('window_count', 0)} | {elapsed_text} | `{output}` |"
        )
    lines.extend(
        [
            "",
            (
                "All outputs are review-only open-phrase envelopes; fixed processing "
                "windows are not action boundaries."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionWemmBatchRunnerError(f"could not read checkpoint {path}: {exc}") from exc
    return dict(_mapping(value, field="checkpoint"))


def _summary(report: dict[str, Any]) -> None:
    items = [_mapping(item, field="run.items[]") for item in report["items"]]
    counts = {
        "planned_count": sum(item.get("status") == "PLANNED" for item in items),
        "running_count": sum(item.get("status") == "RUNNING" for item in items),
        "complete_count": sum(item.get("status") == "COMPLETE" for item in items),
        "failed_count": sum(item.get("status") == "FAILED" for item in items),
        "skipped_count": sum(bool(item.get("resume_skipped", False)) for item in items),
        "window_count": sum(int(item.get("window_count", 0) or 0) for item in items),
    }
    report["summary"] = {**dict(_mapping(report.get("summary"), field="run.summary")), **counts}


def _output_paths(output_dir: Path, item: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    ordinal = int(item["ordinal"])
    recording_id = _recording_id(item)
    prefix = f"{ordinal:03d}-{recording_id}"
    return (
        output_dir / "manifests" / f"{prefix}.json",
        output_dir / "preannotations" / f"{prefix}.json",
        output_dir / "review" / f"{prefix}.json",
    )


def run_production_wemm_batch(
    preflight: Mapping[str, Any] | str | Path,
    *,
    phrase_catalog: Mapping[str, Any] | Sequence[Any] | str | Path,
    model_directory: str | Path | None,
    output_directory: str | Path,
    archive_path: str | Path | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    include_tail: bool = True,
    frame_count: int = DEFAULT_FRAME_COUNT,
    top_k: int = DEFAULT_TOP_K,
    dimension: int = DEFAULT_DIMENSION,
    device: str = "cuda",
    label_variant: str = "canonical",
    fusion: str = "mean",
    score_normalization: str = "none",
    window_chunk_size: int = DEFAULT_WINDOW_CHUNK_SIZE,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    include_pipeline: bool = False,
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
    max_windows: int | None = None,
    ordinals: Sequence[int] | None = None,
    batch: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    resume: bool = True,
    reuse_backend: bool = True,
    staging_directory: str | Path | None = None,
    keep_staging: bool = False,
    checkpoint_path: str | Path | None = None,
    window_stride_seconds: float | None = None,
    temporal_mode: str = MODE_NONE,
    temporal_start_threshold: float = 0.65,
    temporal_stop_threshold: float = 0.50,
    temporal_merge_gap_seconds: float = 0.25,
    temporal_min_duration_seconds: float = 0.10,
    temporal_min_camera_support: int = 1,
    temporal_boundary_mode: str = "midpoint",
    temporal_score_policy: str = DEFAULT_SCORE_POLICY,
) -> dict[str, Any]:
    """Run selected source-preflight recordings serially and resumably.

    ``window_chunk_size`` is forwarded to the open runner to cap in-memory
    decoded frames. Keep it at the default of one on constrained hosts.

    ``inference_batch_size`` is an opt-in native video microbatch width. The
    default of one preserves the historical singleton path; values greater
    than one reuse the same resident backend and bounded decode chunk while
    changing only model scheduling. The output contract remains per-window
    and review-only.

    ``include_pipeline`` is an opt-in producer/consumer decode+inference
    schedule forwarded to :func:`run_production_wemm_open`; serial processing
    remains the default.  ``queue_capacity`` bounds that pipeline's pending
    decoded chunks.  The two fields are omitted from the default checkpoint
    configuration so older dry-run/resume consumers retain their shape.
    """

    if not math.isfinite(float(window_seconds)) or float(window_seconds) <= 0:
        raise ProductionWemmBatchRunnerError("window_seconds must be positive and finite")
    effective_window_stride = window_stride_seconds
    if temporal_mode == MODE_DENSE_SCORE and effective_window_stride is None:
        # A dense temporal pass needs overlapping contexts.  Derive a
        # documented default (4 probes per context) rather than silently
        # reusing the non-overlapping production compatibility stride.
        effective_window_stride = float(window_seconds) / DEFAULT_TEMPORAL_STRIDE_DIVISOR
    if effective_window_stride is not None and (
        not math.isfinite(float(effective_window_stride))
        or float(effective_window_stride) <= 0
        or float(effective_window_stride) > float(window_seconds)
    ):
        raise ProductionWemmBatchRunnerError(
            "window_stride_seconds must be positive and <= window_seconds"
        )
    if temporal_mode not in {MODE_NONE, MODE_DENSE_SCORE}:
        raise ProductionWemmBatchRunnerError(
            f"temporal_mode must be one of {MODE_NONE!r}, {MODE_DENSE_SCORE!r}"
        )
    if temporal_score_policy not in SCORE_POLICIES:
        raise ProductionWemmBatchRunnerError(
            f"temporal_score_policy must be one of {SCORE_POLICIES!r}"
        )
    if not isinstance(include_tail, bool):
        raise ProductionWemmBatchRunnerError("include_tail must be boolean")
    if (
        isinstance(window_chunk_size, bool)
        or not isinstance(window_chunk_size, int)
        or window_chunk_size <= 0
    ):
        raise ProductionWemmBatchRunnerError("window_chunk_size must be a positive integer")
    if (
        isinstance(inference_batch_size, bool)
        or not isinstance(inference_batch_size, int)
        or inference_batch_size <= 0
        or inference_batch_size > 64
    ):
        raise ProductionWemmBatchRunnerError(
            "inference_batch_size must be an integer between 1 and 64"
        )
    if not isinstance(include_pipeline, bool):
        raise ProductionWemmBatchRunnerError("include_pipeline must be boolean")
    if (
        isinstance(queue_capacity, bool)
        or not isinstance(queue_capacity, int)
        or queue_capacity <= 0
    ):
        raise ProductionWemmBatchRunnerError("queue_capacity must be a positive integer")
    if max_windows is not None and (
        isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows <= 0
    ):
        raise ProductionWemmBatchRunnerError("max_windows must be a positive integer")
    if not isinstance(reuse_backend, bool) or not isinstance(resume, bool):
        raise ProductionWemmBatchRunnerError("resume and reuse_backend must be boolean")
    preflight_doc = load_source_preflight(preflight)
    preflight_root: Path | None = None
    if isinstance(preflight, (str, Path)):
        preflight_root = Path(preflight).expanduser().resolve().parent
    raw_archive = Path(
        archive_path if archive_path is not None else str(preflight_doc["archive_path"])
    ).expanduser()
    if raw_archive.is_absolute():
        archive = raw_archive.resolve()
    else:
        # Preflight artifacts generated in this repository store paths relative
        # to the repository root, while callers may also provide a sidecar
        # relative to its own directory.  Prefer an existing candidate and
        # otherwise retain the normal cwd-relative resolution for diagnostics.
        candidates: list[Path] = [Path.cwd() / raw_archive]
        if preflight_root is not None:
            candidates.extend([preflight_root / raw_archive, preflight_root.parent / raw_archive])
        archive = next(
            (candidate.resolve() for candidate in candidates if candidate.is_file()),
            candidates[0].resolve(),
        )
    selected = select_preflight_items(
        preflight_doc,
        status="PASS",
        batch=batch,
        ordinals=ordinals,
        limit=limit,
    )
    if not selected:
        raise ProductionWemmBatchRunnerError("no PASS source-preflight recordings selected")
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else output_dir / "batch-run.json"
    )
    config: dict[str, Any] = {
        "window_seconds": float(window_seconds),
        "include_tail": bool(include_tail),
        "frame_count": frame_count,
        "top_k": top_k,
        "dimension": dimension,
        "device": device,
        "label_variant": label_variant,
        "fusion": fusion,
        "score_normalization": score_normalization,
        "window_chunk_size": window_chunk_size,
        "inference_batch_size": inference_batch_size,
        "max_windows": max_windows,
        "batch": batch,
        "limit": limit,
        "dry_run": dry_run,
        "resume": resume,
        "reuse_backend": reuse_backend,
        "phrase_catalog": (
            str(phrase_catalog) if isinstance(phrase_catalog, Path) else phrase_catalog
        ),
        "model_directory": str(model_directory) if model_directory is not None else None,
    }
    if effective_window_stride is not None:
        config["window_stride_seconds"] = float(effective_window_stride)
    if temporal_mode == MODE_DENSE_SCORE:
        config.update(
            {
                "temporal_mode": temporal_mode,
                "temporal_start_threshold": temporal_start_threshold,
                "temporal_stop_threshold": temporal_stop_threshold,
                "temporal_merge_gap_seconds": temporal_merge_gap_seconds,
                "temporal_min_duration_seconds": temporal_min_duration_seconds,
                "temporal_min_camera_support": temporal_min_camera_support,
                "temporal_boundary_mode": temporal_boundary_mode,
                "temporal_score_policy": temporal_score_policy,
            }
        )
    if include_pipeline:
        config["include_pipeline"] = True
        config["queue_capacity"] = queue_capacity
    report = _load_checkpoint(checkpoint) if resume else None
    if report is None:
        report = _empty_run(
            preflight=preflight_doc,
            archive_path=archive,
            selected=selected,
            config=config,
            checkpoint_path=checkpoint,
        )
        report["items"] = [
            {
                **dict(item),
                "recording_id": _recording_id(item),
                "status": "PLANNED",
                "manifest_path": None,
                "preannotation_path": None,
                "review_path": None,
                "window_count": 0,
                "estimated_window_count": _estimate_windows(
                    item.get("duration_seconds"),
                    window_seconds,
                    include_tail,
                    effective_window_stride,
                ),
                "elapsed_seconds": None,
                "error": None,
            }
            for item in selected
        ]
    else:
        # A checkpoint can only be resumed for the same source members.  This
        # prevents accidentally mixing outputs from two archive selections.
        prior = {
            str(_mapping(item, field="checkpoint.items[]").get("archive_member"))
            for item in _sequence(report.get("items"), field="checkpoint.items")
        }
        current = {str(item["archive_member"]) for item in selected}
        if prior and prior != current:
            raise ProductionWemmBatchRunnerError(
                "checkpoint selection differs from requested source-preflight members"
            )
        prior_config = _mapping(report.get("config", {}), field="checkpoint.config")
        requested_temporal = _temporal_resume_signature(config, default_mode=temporal_mode)
        prior_temporal = _temporal_resume_signature(prior_config)
        if prior_temporal != requested_temporal:
            raise ProductionWemmBatchRunnerError(
                "checkpoint temporal configuration differs from requested run; "
                "start a fresh checkpoint or keep temporal mode, stride and resolver "
                "parameters unchanged"
            )
        report["status"] = RUN_STATUS_RUNNING
        report["config"] = config
    _summary(report)
    _write_checkpoint(checkpoint, report)

    if dry_run:
        for raw in report["items"]:
            item = dict(_mapping(raw, field="run.items[]"))
            if item.get("status") in {"COMPLETE", "SKIPPED"}:
                continue
            item["status"] = "PLANNED"
            raw.clear()
            raw.update(item)
        report["status"] = RUN_STATUS_DRY_RUN
        _summary(report)
        _write_checkpoint(checkpoint, report)
        return report

    if model_directory is None:
        raise ProductionWemmBatchRunnerError("model_directory is required for a non-dry run")
    staging_root: Path
    own_staging_root = False
    if staging_directory is None:
        # ``TemporaryDirectory`` on this Windows workspace creates a restrictive
        # ACL which can make its own cleanup fail under the managed runner.
        # Use a normal repository-owned directory instead; per-recording
        # children are removed by ``stage_zip_member`` and the root is best
        # effort cleaned at the end.
        staging_root = output_dir / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        own_staging_root = True
    else:
        staging_root = Path(staging_directory).expanduser().resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
    backend: WemmEmbeddingBackend | None = None
    if reuse_backend:
        try:
            backend = WemmEmbeddingBackend(
                model_directory=model_directory, device=device, dimension=dimension
            )
        except Exception as exc:
            for raw in report["items"]:
                item = dict(_mapping(raw, field="run.items[]"))
                if item.get("status") == "COMPLETE":
                    continue
                item["status"] = "FAILED"
                item["error"] = {"type": type(exc).__name__, "detail": str(exc)}
                raw.clear()
                raw.update(item)
            report["status"] = RUN_STATUS_PARTIAL
            _summary(report)
            _write_checkpoint(checkpoint, report)
            raise ProductionWemmBatchRunnerError(
                f"could not initialise resident WeMM backend: {exc}"
            ) from exc
    try:
        for raw in report["items"]:
            item = dict(_mapping(raw, field="run.items[]"))
            if resume and item.get("status") == "COMPLETE":
                # Preserve the durable completion state; ``resume_skipped``
                # records that this invocation reused the existing artifact.
                item["resume_skipped"] = True
                raw.clear()
                raw.update(item)
                _summary(report)
                _write_checkpoint(checkpoint, report)
                continue
            started = time.perf_counter()
            item["status"] = "RUNNING"
            item["error"] = None
            raw.clear()
            raw.update(item)
            _summary(report)
            _write_checkpoint(checkpoint, report)
            manifest_path, envelope_path, review_path = _output_paths(output_dir, item)
            item["manifest_path"] = str(manifest_path)
            item["preannotation_path"] = str(envelope_path)
            item["review_path"] = str(review_path)
            try:
                with stage_zip_member(
                    archive,
                    str(item["archive_member"]),
                    staging_root,
                    ordinal=int(item["ordinal"]),
                    keep_staged=keep_staging,
                ) as staged:
                    manifest = build_recording_manifest(
                        staged,
                        recording_id=str(item["recording_id"]),
                        archive_member=str(item["archive_member"]),
                        source_preflight_status=str(item["source_preflight_status"]),
                        qa_status=str(item.get("qa_status", "PENDING")),
                        window_seconds=window_seconds,
                        include_tail=include_tail,
                        **(
                            {"window_stride_seconds": effective_window_stride}
                            if effective_window_stride is not None
                            else {}
                        ),
                    )
                    manifest_source = dict(
                        _mapping(manifest.get("source"), field="manifest.source")
                    )
                    manifest_source.update(
                        {
                            "archive_path": str(archive),
                            "archive_member": str(item["archive_member"]),
                            "staged_path": str(staged),
                            "path_lifecycle": "STAGED_PATH_REMOVED_AFTER_RECORDING",
                        }
                    )
                    manifest["source"] = manifest_source
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(
                        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    open_kwargs: dict[str, Any] = {
                        "phrase_catalog": phrase_catalog,
                        "model_directory": model_directory,
                        "frame_count": frame_count,
                        "top_k": top_k,
                        "dimension": dimension,
                        "device": device,
                        "label_variant": label_variant,
                        "max_windows": max_windows,
                        "fusion": fusion,
                        "score_normalization": score_normalization,
                        "backend": backend,
                        "window_chunk_size": window_chunk_size,
                        "inference_batch_size": inference_batch_size,
                    }
                    if temporal_mode == MODE_DENSE_SCORE:
                        open_kwargs.update(
                            {
                                "temporal_mode": temporal_mode,
                                "temporal_start_threshold": temporal_start_threshold,
                                "temporal_stop_threshold": temporal_stop_threshold,
                                "temporal_merge_gap_seconds": temporal_merge_gap_seconds,
                                "temporal_min_duration_seconds": temporal_min_duration_seconds,
                                "temporal_min_camera_support": temporal_min_camera_support,
                                "temporal_boundary_mode": temporal_boundary_mode,
                                "temporal_score_policy": temporal_score_policy,
                            }
                        )
                    # Do not add new keyword arguments to the historical
                    # serial call.  Besides preserving the default checkpoint
                    # shape, this keeps older injected/open-runner adapters
                    # source-compatible when pipeline mode is disabled.
                    if include_pipeline:
                        open_kwargs.update(
                            {
                                "include_pipeline": True,
                                "queue_capacity": queue_capacity,
                            }
                        )
                    envelope = run_production_wemm_open(manifest, **open_kwargs)
                    envelope_path.parent.mkdir(parents=True, exist_ok=True)
                    envelope_path.write_text(
                        json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    review = build_review_pack(envelope)
                    review_path.parent.mkdir(parents=True, exist_ok=True)
                    review_path.write_text(
                        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    item["window_count"] = len(envelope.get("windows", []))
                    if temporal_mode == MODE_DENSE_SCORE:
                        temporal = envelope.get("temporal_resolution", {})
                        if isinstance(temporal, Mapping):
                            segments = temporal.get("segments", [])
                            item["temporal_segment_count"] = (
                                len(segments) if isinstance(segments, Sequence) else 0
                            )
                    item["status"] = "COMPLETE"
                    item["production_eligible"] = bool(envelope.get("production_eligible", False))
            except Exception as exc:  # keep serial batch moving; checkpoint the failure
                item["status"] = "FAILED"
                item["error"] = {
                    "type": type(exc).__name__,
                    "detail": str(exc),
                }
            item["elapsed_seconds"] = time.perf_counter() - started
            raw.clear()
            raw.update(item)
            _summary(report)
            _write_checkpoint(checkpoint, report)
    except KeyboardInterrupt:
        report["status"] = RUN_STATUS_INTERRUPTED
        _summary(report)
        _write_checkpoint(checkpoint, report)
        raise
    finally:
        if backend is not None:
            backend.close()
        if own_staging_root and not keep_staging:
            # ``ignore_errors`` keeps a completed checkpoint usable even when
            # a managed Windows filesystem briefly denies directory removal.
            shutil.rmtree(staging_root, ignore_errors=True)
    summary = _mapping(report.get("summary"), field="run.summary")
    report["status"] = (
        RUN_STATUS_COMPLETE if int(summary.get("failed_count", 0)) == 0 else RUN_STATUS_PARTIAL
    )
    _write_checkpoint(checkpoint, report)
    return report


__all__ = [
    "AUTHORITY",
    "BATCH_RUN_FORMAT",
    "DEFAULT_DIMENSION",
    "DEFAULT_FRAME_COUNT",
    "DEFAULT_INFERENCE_BATCH_SIZE",
    "DEFAULT_QUEUE_CAPACITY",
    "DEFAULT_TOP_K",
    "DEFAULT_WINDOW_CHUNK_SIZE",
    "DEFAULT_WINDOW_SECONDS",
    "ProductionWemmBatchRunnerError",
    "build_recording_manifest",
    "load_source_preflight",
    "render_batch_run_markdown",
    "run_production_wemm_batch",
    "select_preflight_items",
    "stage_zip_member",
]
