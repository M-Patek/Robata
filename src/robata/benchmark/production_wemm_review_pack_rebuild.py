"""Rebuild WeMM review packs from completed pre-annotation sidecars.

The production runner writes a pre-annotation sidecar and a review pack while
processing each recording.  This helper is a bounded, read-only *model/media*
operation: it reads only ``preannotation_path`` entries whose batch status is
``COMPLETE``, calls the inference-free :func:`build_review_pack`, and writes the
result to a separate directory.  It is useful when an older runner emitted a
review pack without the compact model/catalog provenance now required by the
review queue.

Running/planned/failed checkpoint entries are never opened.  Existing runner
outputs are never overwritten unless the caller explicitly chooses an output
file that already exists and passes ``overwrite=True``; the CLI defaults to a
new, separate output directory and leaves existing files untouched.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .production_wemm_preannotation import (
    FORMAT as PREANNOTATION_FORMAT,
)
from .production_wemm_preannotation import (
    ProductionWemmPreannotationError,
    build_review_pack,
    validate_preannotation_envelope,
)

BATCH_RUN_FORMAT: Final = "robata-production-wemm-batch-run-v1"
REBUILD_FORMAT: Final = "robata-production-wemm-review-pack-rebuild-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"


class ProductionWemmReviewPackRebuildError(ValueError):
    """Raised for an invalid rebuild request (not one bad checkpoint row)."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmReviewPackRebuildError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmReviewPackRebuildError(f"{field} must be an array")
    return value


def _text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _copy_json(value: object, *, field: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmReviewPackRebuildError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(key): _copy_json(child, field=f"{field}.{key}") for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionWemmReviewPackRebuildError(f"{field} must be JSON-compatible")


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"could not read JSON: {exc}"
    if not isinstance(payload, Mapping):
        return None, "JSON root must be an object"
    return dict(payload), None


def _resolve(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.resolve()


def _safe_stem(value: object, *, fallback: str) -> str:
    raw = _text(value) or fallback
    # Keep the original recording name recognisable while avoiding path
    # separators and characters that are invalid on Windows.
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-")
    return stem or fallback


def _recording_filename(source: Mapping[str, Any], *, recording_id: str) -> str:
    for key in ("archive_member", "path", "source_path", "staged_path"):
        value = _text(source.get(key))
        if value:
            return Path(value.replace("\\", "/")).name or recording_id
    return recording_id


def _checkpoint_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists() or not path.is_dir():
        return []
    # Batch roots conventionally have one checkpoint at the root.  Recursive
    # discovery also supports a parent directory containing B1/remaining roots.
    return sorted(candidate for candidate in path.rglob("batch-run*.json") if candidate.is_file())


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output_location(output_dir: Path, input_paths: Sequence[Path]) -> None:
    """Prevent an output directory from being the runner's own output tree."""

    for input_path in input_paths:
        resolved = input_path.resolve()
        if output_dir == resolved:
            raise ProductionWemmReviewPackRebuildError(
                "output directory must be separate from an input path"
            )
        # A directory named review/preannotations is an active runner output
        # convention.  Refuse those exact locations *and descendants* while
        # allowing a sibling such as ``rebuilt-review`` under the batch root.
        if resolved.is_dir() and resolved.name.casefold() in {"review", "preannotations"}:
            runner_root = resolved.parent
        else:
            runner_root = resolved if resolved.is_dir() else resolved.parent
        runner_dirs = (runner_root / "review", runner_root / "preannotations")
        if any(
            output_dir == runner_dir or _is_within(output_dir, runner_dir)
            for runner_dir in runner_dirs
        ):
            raise ProductionWemmReviewPackRebuildError(
                "output directory must not be the runner review/preannotations directory"
            )


def _item_output_path(
    output_dir: Path,
    *,
    source: Mapping[str, Any],
    recording_id: str,
    ordinal: object,
    used_names: set[str],
) -> Path:
    filename = _recording_filename(source, recording_id=recording_id)
    stem = Path(filename).stem or recording_id
    prefix = _safe_stem(ordinal, fallback="0000")
    fallback_stem = _safe_stem(recording_id, fallback="recording")
    base = f"{prefix}-{_safe_stem(stem, fallback=fallback_stem)}-review"
    candidate_name = f"{base}.json"
    suffix = 2
    while candidate_name.casefold() in used_names:
        candidate_name = f"{base}-{suffix:02d}.json"
        suffix += 1
    used_names.add(candidate_name.casefold())
    return output_dir / candidate_name


def _checkpoint_rows(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    *,
    rejected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if checkpoint.get("format") != BATCH_RUN_FORMAT:
        rejected.append({"path": str(checkpoint_path), "reason": "ROOT_NOT_BATCH_CHECKPOINT"})
        return [], {
            "path": str(checkpoint_path),
            "status": "NOT_BATCH_CHECKPOINT",
            "item_count": 0,
            "complete_item_count": 0,
            "unresolved_item_count": 0,
        }

    raw_items = checkpoint.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
        rejected.append(
            {"path": f"{checkpoint_path}#items", "reason": "CHECKPOINT_ITEMS_NOT_ARRAY"}
        )
        return [], {
            "path": str(checkpoint_path),
            "status": _text(checkpoint.get("status")) or "UNKNOWN",
            "item_count": 0,
            "complete_item_count": 0,
            "unresolved_item_count": 0,
        }

    rows: list[dict[str, Any]] = []
    complete_count = 0
    unresolved_count = 0
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            unresolved_count += 1
            rejected.append(
                {
                    "path": f"{checkpoint_path}#items[{index}]",
                    "reason": "CHECKPOINT_ITEM_NOT_OBJECT",
                }
            )
            continue
        status = (_text(raw_item.get("status")) or "UNKNOWN").upper()
        if status != "COMPLETE":
            unresolved_count += 1
            rejected.append(
                {
                    "path": f"{checkpoint_path}#items[{index}]",
                    "reason": f"STATUS_NOT_COMPLETE:{status}",
                    "status": status,
                }
            )
            continue
        raw_sidecar = raw_item.get("preannotation_path")
        if not isinstance(raw_sidecar, (str, Path)) or not str(raw_sidecar).strip():
            unresolved_count += 1
            rejected.append(
                {
                    "path": f"{checkpoint_path}#items[{index}]",
                    "reason": "COMPLETE_ITEM_MISSING_PREANNOTATION_PATH",
                }
            )
            continue
        sidecar_path = _resolve(str(raw_sidecar), base=checkpoint_path.parent)
        rows.append(
            {
                "checkpoint_path": checkpoint_path,
                "checkpoint_item_index": index,
                "checkpoint_item": dict(raw_item),
                "preannotation_path": sidecar_path,
            }
        )
        complete_count += 1

    return rows, {
        "path": str(checkpoint_path),
        "status": _text(checkpoint.get("status")) or "UNKNOWN",
        "item_count": len(raw_items),
        "complete_item_count": complete_count,
        "unresolved_item_count": unresolved_count,
    }


def rebuild_production_wemm_review_packs(
    inputs: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rebuild review packs for COMPLETE checkpoint items only.

    The returned report is JSON-compatible and intentionally non-gold.  A bad
    sidecar is recorded in ``invalid_inputs`` while other completed entries are
    still rebuilt.
    """

    if isinstance(inputs, (str, Path)):
        raw_inputs: list[str | Path] = [inputs]
    elif isinstance(inputs, Sequence) and not isinstance(inputs, (bytes, bytearray)):
        raw_inputs = list(inputs)
    else:
        raise ProductionWemmReviewPackRebuildError("inputs must be a path or sequence of paths")
    if not raw_inputs:
        raise ProductionWemmReviewPackRebuildError("at least one input path is required")
    if not isinstance(overwrite, bool):
        raise ProductionWemmReviewPackRebuildError("overwrite must be boolean")

    input_paths = [_resolve(path) for path in raw_inputs]
    output = _resolve(output_dir)
    _validate_output_location(output, input_paths)
    output.mkdir(parents=True, exist_ok=True)

    rejected: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    checkpoint_sources: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    discovered_checkpoints: set[Path] = set()
    for input_path in input_paths:
        candidates = _checkpoint_paths(input_path)
        if not candidates:
            rejected.append({"path": str(input_path), "reason": "NO_BATCH_CHECKPOINT_FOUND"})
            continue
        for checkpoint_path in candidates:
            if checkpoint_path in discovered_checkpoints:
                rejected.append(
                    {"path": str(checkpoint_path), "reason": "DUPLICATE_CHECKPOINT_PATH"}
                )
                continue
            discovered_checkpoints.add(checkpoint_path)
            payload, error = _load_json(checkpoint_path)
            if error is not None or payload is None:
                invalid.append({"path": str(checkpoint_path), "reason": error or "INVALID_JSON"})
                continue
            rows, checkpoint_source = _checkpoint_rows(
                checkpoint_path,
                payload,
                rejected=rejected,
            )
            checkpoint_sources.append(checkpoint_source)
            checkpoint_rows.extend(rows)

    # A sidecar can be reachable from two checkpoints (for example after a
    # resumed run).  Process it once and make the duplicate explicit.
    seen_sidecars: set[Path] = set()
    used_names: set[str] = set()
    written: list[dict[str, Any]] = []
    skipped_existing: list[dict[str, Any]] = []
    overwritten_count = 0
    source_statuses: Counter[str] = Counter()
    for row in sorted(
        checkpoint_rows,
        key=lambda item: (
            str(item["preannotation_path"]),
            str(item["checkpoint_path"]),
            int(item["checkpoint_item_index"]),
        ),
    ):
        sidecar_path = Path(row["preannotation_path"])
        if sidecar_path in seen_sidecars:
            rejected.append({"path": str(sidecar_path), "reason": "DUPLICATE_PREANNOTATION_PATH"})
            continue
        seen_sidecars.add(sidecar_path)
        payload, error = _load_json(sidecar_path)
        if error is not None or payload is None:
            invalid.append({"path": str(sidecar_path), "reason": error or "INVALID_JSON"})
            continue
        if payload.get("format") != PREANNOTATION_FORMAT:
            invalid.append(
                {
                    "path": str(sidecar_path),
                    "reason": f"ROOT_NOT_PREANNOTATION:{payload.get('format')}",
                }
            )
            continue
        try:
            envelope = validate_preannotation_envelope(payload)
            review = build_review_pack(envelope)
        except (ProductionWemmPreannotationError, TypeError, ValueError) as exc:
            invalid.append({"path": str(sidecar_path), "reason": str(exc)})
            continue

        source = _mapping(envelope.get("source"), field=f"{sidecar_path}.source")
        recording_id = _text(source.get("recording_id") or source.get("source_id"))
        if recording_id is None:
            recording_id = _text(source.get("path"))
        if recording_id is None:
            invalid.append({"path": str(sidecar_path), "reason": "SOURCE_RECORDING_ID_MISSING"})
            continue
        source_statuses["COMPLETE"] += 1
        item_field = f"{row['checkpoint_path']}#items[{row['checkpoint_item_index']}]"
        checkpoint_item = _mapping(row["checkpoint_item"], field=item_field)
        ordinal = checkpoint_item.get("ordinal", row["checkpoint_item_index"])
        output_path = _item_output_path(
            output,
            source=source,
            recording_id=recording_id,
            ordinal=ordinal,
            used_names=used_names,
        )
        if output_path.exists() and not overwrite:
            skipped_existing.append(
                {
                    "path": str(output_path),
                    "recording_id": recording_id,
                    "preannotation_path": str(sidecar_path),
                    "reason": "OUTPUT_EXISTS",
                }
            )
            continue
        if output_path.exists():
            overwritten_count += 1
        review = _copy_json(review, field="review_pack")
        if not isinstance(review, dict):  # pragma: no cover - build_review_pack invariant
            raise ProductionWemmReviewPackRebuildError("review pack must be an object")
        review["rebuild_provenance"] = {
            "format": REBUILD_FORMAT,
            "authority": AUTHORITY,
            "source_preannotation_path": str(sidecar_path),
            "source_checkpoint_path": str(row["checkpoint_path"]),
            "source_checkpoint_item_index": row["checkpoint_item_index"],
            "source_status": "COMPLETE",
            "recording_id": recording_id,
            "recording_filename": _recording_filename(source, recording_id=recording_id),
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
        }
        output_path.write_text(
            json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(
            {
                "path": str(output_path),
                "recording_id": recording_id,
                "recording_filename": _recording_filename(source, recording_id=recording_id),
                "preannotation_path": str(sidecar_path),
                "checkpoint_path": str(row["checkpoint_path"]),
                "window_count": len(review.get("items", [])),
            }
        )

    unresolved = sum(int(item.get("unresolved_item_count", 0) or 0) for item in checkpoint_sources)
    complete_candidates = sum(
        int(item.get("complete_item_count", 0) or 0) for item in checkpoint_sources
    )
    if written or skipped_existing:
        status = "PARTIAL" if unresolved or invalid or skipped_existing else "COMPLETE"
    else:
        status = "NO_COMPLETE_ITEMS"
    return {
        "format": REBUILD_FORMAT,
        "authority": AUTHORITY,
        "status": status,
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "output_dir": str(output),
        "inputs": [str(path) for path in input_paths],
        "checkpoints": checkpoint_sources,
        "summary": {
            "checkpoint_count": len(checkpoint_sources),
            "complete_item_count": complete_candidates,
            "written_count": len(written),
            "skipped_existing_count": len(skipped_existing),
            "invalid_count": len(invalid),
            "rejected_count": len(rejected),
            "unresolved_item_count": unresolved,
            "source_status_counts": dict(sorted(source_statuses.items())),
        },
        "written": written,
        "skipped_existing": skipped_existing,
        "invalid_inputs": invalid,
        "rejected_inputs": rejected,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "qwen_read": False,
            "mage_read": False,
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "hash_or_sha_used": False,
            "runner_outputs_overwritten": False,
            "existing_rebuild_outputs_overwritten": overwritten_count > 0,
        },
        "limitations": [
            "Only COMPLETE checkpoint items and their preannotation sidecars are processed.",
            "Running, planned and failed items are rejected and never opened.",
            "The rebuilt packs remain review-only; no gold or quality metric is produced.",
            "Processing windows remain context units, not inferred action boundaries.",
            "Qwen, Mage and media decoders are not invoked.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping(report.get("summary", {}), field="report.summary")
    lines = [
        "# Production WeMM review-pack rebuild",
        "",
        f"- Status: **{report.get('status', 'UNKNOWN')}**",
        f"- Output directory: `{report.get('output_dir', '')}`",
        f"- Checkpoints: {summary.get('checkpoint_count', 0)}",
        f"- COMPLETE items: {summary.get('complete_item_count', 0)}",
        f"- Written: {summary.get('written_count', 0)}",
        f"- Skipped existing: {summary.get('skipped_existing_count', 0)}",
        f"- Rejected/unresolved: {summary.get('rejected_count', 0)}",
        f"- Invalid: {summary.get('invalid_count', 0)}",
        "",
        "## Limitations",
        "",
    ]
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


__all__ = [
    "AUTHORITY",
    "BATCH_RUN_FORMAT",
    "PREANNOTATION_FORMAT",
    "REBUILD_FORMAT",
    "ProductionWemmReviewPackRebuildError",
    "rebuild_production_wemm_review_packs",
    "render_markdown",
]
