"""Plan bounded human-review batches from a production WeMM draft.

This module is deliberately inference-free.  It chunks an existing editable
WeMM draft into small review files and keeps model proposals separate from the
reviewer decision.  The fixed processing interval remains context only; the
reviewer receives explicit, initially-unmeasured action-boundary fields.

No media is opened, no model is invoked, no ontology/mapper is changed, and no
hash or digest is calculated.
"""

# The queue's Markdown rendering intentionally keeps compact human-readable
# lines; line wrapping is not part of the artifact contract.
# ruff: noqa: E501

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

FORMAT: Final = "robata-production-review-queue-batches-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "PLANNED_NONPRODUCTION"
DECISION_OPTIONS: Final = ("accept", "edit", "split", "reject", "abstain")
REQUIRED_PROVENANCE_FIELDS: Final = (
    "recording_id",
    "qa_status",
    "source_preflight_status",
    "review_pack_path",
    "archive_member",
    "source_path",
)
REQUIRED_TOP_K = 6


class ProductionReviewQueueBatchError(ValueError):
    """Raised when a production draft cannot be chunked safely."""


def _copy_json(value: object, *, field: str = "value") -> Any:
    """Deep-copy JSON-shaped values while rejecting non-finite numbers."""

    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionReviewQueueBatchError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionReviewQueueBatchError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[]") for child in value]
    raise ProductionReviewQueueBatchError(f"{field} is not JSON-compatible")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionReviewQueueBatchError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionReviewQueueBatchError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionReviewQueueBatchError(f"{field} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ProductionReviewQueueBatchError(f"{field} must be non-empty")
    return result


def _finite(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionReviewQueueBatchError(f"{field} must be a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ProductionReviewQueueBatchError(f"{field} must be finite")
    return number


def _compact_provenance(window: Mapping[str, Any]) -> dict[str, Any]:
    raw = window.get("source_provenance")
    provenance = _mapping(raw, field="window.source_provenance")
    recording_id = _text(window.get("recording_id"), field="window.recording_id")
    result: dict[str, Any] = {"recording_id": recording_id}
    # Keep only stable source/QA locators in every queue item.  The full input
    # snapshots remain in the canonical draft and are not duplicated 789 times.
    for field in REQUIRED_PROVENANCE_FIELDS[1:]:
        value = provenance.get(field)
        if value is None:
            # The aggregate may retain archive/path values under source_ref.
            source_ref = window.get("source_ref")
            if isinstance(source_ref, Mapping):
                value = source_ref.get(field)
        if value is None and field in {"qa_status", "source_preflight_status"}:
            value = window.get(field)
        if value is None:
            raise ProductionReviewQueueBatchError(
                f"window {recording_id} is missing provenance field {field!r}"
            )
        result[field] = _copy_json(value, field=f"window.provenance.{field}")
    for field in (
        "archive_path",
        "manifest_format",
        "media_type",
        "camera_count",
        "common_duration_seconds",
        "path_lifecycle",
        "staging_lifecycle",
    ):
        value = provenance.get(field)
        if value is not None:
            result[field] = _copy_json(value, field=f"window.provenance.{field}")
    result["window_context_only"] = True
    return result


def _segment_for_window(window: Mapping[str, Any]) -> Mapping[str, Any] | None:
    draft = _mapping(window.get("annotation_draft"), field="window.annotation_draft")
    segments = _sequence(draft.get("segments", []), field="window.annotation_draft.segments")
    if not segments:
        return None
    return _mapping(segments[0], field="window.annotation_draft.segments[0]")


def _qwen_placeholder(result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if result is None:
        return {
            "status": "NOT_RUN",
            "result": None,
            "artifact_reference": None,
            "selected_rank": None,
            "decision": None,
            "segments": [],
        }
    return {
        "status": "AVAILABLE",
        "result": _copy_json(result, field="qwen.result"),
        "artifact_reference": result.get("artifact_reference"),
        "selected_rank": result.get("selected_rank"),
        "decision": result.get("decision"),
        "segments": _copy_json(result.get("segments", []), field="qwen.segments"),
    }


def _review_placeholder(*, window_id: str, segment: Mapping[str, Any] | None) -> dict[str, Any]:
    split_hint = bool(segment.get("split_hint", False)) if segment else False
    # These are the fields a reviewer must fill.  They are intentionally not
    # copied from the model proposal, even when WeMM has a provisional label.
    return {
        "status": "PENDING",
        "decision": "pending",
        "decision_options": list(DECISION_OPTIONS),
        "reviewer_id": None,
        "reviewed_at": None,
        "notes": None,
        "split_hint": split_hint,
        "annotation": {
            "start_seconds": None,
            "end_seconds": None,
            "boundary_status": "NOT_MEASURED",
            "timestamp_basis": None,
            "verb": None,
            "noun": None,
            "attributes": None,
            "location": None,
            "hand": None,
            "confidence": None,
            "evidence": [],
        },
        "segments": [
            {
                "segment_id": f"{window_id}-review-s01",
                "start_seconds": None,
                "end_seconds": None,
                "boundary_status": "NOT_MEASURED",
                "timestamp_basis": None,
                "verb": None,
                "noun": None,
                "attributes": None,
                "location": None,
                "hand": None,
                "confidence": None,
                "evidence": [],
                "status": "PENDING_HUMAN_REVIEW",
            }
        ],
        "context_is_action_boundary": False,
        "human_measurement_required": True,
    }


def _item(
    window: Mapping[str, Any],
    *,
    queue_index: int,
    batch_id: str,
    qwen_by_window: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    window_id = _text(window.get("window_id"), field="window.window_id")
    segment = _segment_for_window(window)
    if segment is None:
        raise ProductionReviewQueueBatchError(
            f"window {window_id} has no provisional segment; queue requires explicit review boundary fields"
        )
    top_k = _sequence(segment.get("top_k", []), field=f"{window_id}.top_k")
    if len(top_k) != REQUIRED_TOP_K:
        raise ProductionReviewQueueBatchError(
            f"window {window_id}.top_k must contain exactly {REQUIRED_TOP_K} entries"
        )
    context = _mapping(window.get("source_interval"), field=f"{window_id}.source_interval")
    context_start = _finite(
        context.get("start_seconds"), field=f"{window_id}.source_interval.start_seconds"
    )
    context_end = _finite(
        context.get("end_seconds"), field=f"{window_id}.source_interval.end_seconds"
    )
    context_status = _text(
        context.get("status", "WINDOW_CONTEXT_ONLY"),
        field=f"{window_id}.source_interval.status",
    )
    provenance = _compact_provenance(window)
    proposal_boundary = {
        "start_seconds": _finite(
            segment.get("start_seconds"), field=f"{window_id}.proposal.start_seconds"
        ),
        "end_seconds": _finite(
            segment.get("end_seconds"), field=f"{window_id}.proposal.end_seconds"
        ),
        "status": _text(
            segment.get("boundary_status", "NOT_MEASURED"),
            field=f"{window_id}.proposal.boundary_status",
        ),
        "timestamp_basis": segment.get("timestamp_basis"),
        "is_action_boundary": segment.get("window_context", {}).get("is_action_boundary", False)
        if isinstance(segment.get("window_context"), Mapping)
        else False,
    }
    wemm = {
        "status": "AVAILABLE",
        "proposal_id": segment.get("segment_id"),
        "label_text": segment.get("label_text"),
        "structured_labels": _copy_json(
            segment.get("structured_labels", {}), field=f"{window_id}.wemm.structured_labels"
        ),
        "confidence": _finite(segment.get("confidence"), field=f"{window_id}.wemm.confidence"),
        "evidence": _copy_json(segment.get("evidence", []), field=f"{window_id}.wemm.evidence"),
        "camera_support": _copy_json(
            segment.get("camera_support", []), field=f"{window_id}.wemm.camera_support"
        ),
        "top_k": _copy_json(top_k, field=f"{window_id}.wemm.top_k"),
        "margin": _finite(segment.get("margin"), field=f"{window_id}.wemm.margin"),
        "proposal_status": segment.get("proposal_status"),
        "boundary": proposal_boundary,
    }
    return {
        "queue_item_id": f"{batch_id}-I{queue_index:04d}",
        "queue_index": queue_index,
        "batch_id": batch_id,
        "window_id": window_id,
        "recording_id": provenance["recording_id"],
        "ordinal": window.get("ordinal", queue_index),
        "context": {
            "start_seconds": context_start,
            "end_seconds": context_end,
            "status": context_status,
            "is_action_boundary": False,
        },
        "review_boundary": {
            "start_seconds": None,
            "end_seconds": None,
            "boundary_status": "NOT_MEASURED",
            "timestamp_basis": None,
        },
        "provenance": provenance,
        "wemm": wemm,
        "qwen": _qwen_placeholder(qwen_by_window.get(window_id)),
        "review": _review_placeholder(window_id=window_id, segment=segment),
    }


def _qwen_index(value: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if value is None:
        return {}
    rows: list[Mapping[str, Any]] = []
    raw_items = value.get("items", value.get("windows"))
    if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes, bytearray)):
        rows.extend(_mapping(row, field="qwen.items[]") for row in raw_items)
    recordings = value.get("recordings")
    if isinstance(recordings, Sequence) and not isinstance(recordings, (str, bytes, bytearray)):
        for recording_index, recording in enumerate(recordings):
            rec = _mapping(recording, field=f"qwen.recordings[{recording_index}]")
            raw_rows = rec.get("rows", [])
            if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes, bytearray)):
                rows.extend(
                    _mapping(row, field=f"qwen.recordings[{recording_index}].rows[]")
                    for row in raw_rows
                )
    result: dict[str, Mapping[str, Any]] = {}
    for _index, row in enumerate(rows):
        window_id = row.get("window_id")
        if isinstance(window_id, str) and window_id.strip():
            result.setdefault(window_id.strip(), row)
    return result


def _validate_item(item: Mapping[str, Any], *, field: str) -> None:
    for required in (
        "queue_item_id",
        "batch_id",
        "window_id",
        "recording_id",
        "context",
        "review_boundary",
        "provenance",
        "wemm",
        "qwen",
        "review",
    ):
        if required not in item:
            raise ProductionReviewQueueBatchError(f"{field} missing {required!r}")
    context = _mapping(item["context"], field=f"{field}.context")
    if context.get("is_action_boundary") is not False:
        raise ProductionReviewQueueBatchError(f"{field}.context must remain non-action-boundary")
    boundary = _mapping(item["review_boundary"], field=f"{field}.review_boundary")
    if (
        boundary.get("start_seconds") is not None
        or boundary.get("end_seconds") is not None
        or boundary.get("boundary_status") != "NOT_MEASURED"
    ):
        raise ProductionReviewQueueBatchError(
            f"{field}.review_boundary must remain NOT_MEASURED with null times"
        )
    provenance = _mapping(item["provenance"], field=f"{field}.provenance")
    for required in REQUIRED_PROVENANCE_FIELDS:
        if provenance.get(required) in (None, ""):
            raise ProductionReviewQueueBatchError(f"{field}.provenance missing {required!r}")
    wemm = _mapping(item["wemm"], field=f"{field}.wemm")
    if len(_sequence(wemm.get("top_k"), field=f"{field}.wemm.top_k")) != REQUIRED_TOP_K:
        raise ProductionReviewQueueBatchError(f"{field}.wemm.top_k cardinality is not 6")
    review = _mapping(item["review"], field=f"{field}.review")
    if review.get("status") != "PENDING" or review.get("decision") != "pending":
        raise ProductionReviewQueueBatchError(f"{field}.review must remain pending")
    options = list(
        _sequence(review.get("decision_options"), field=f"{field}.review.decision_options")
    )
    if options != list(DECISION_OPTIONS):
        raise ProductionReviewQueueBatchError(f"{field}.review.decision_options are incomplete")
    annotation = _mapping(review.get("annotation"), field=f"{field}.review.annotation")
    if annotation.get("start_seconds") is not None or annotation.get("end_seconds") is not None:
        raise ProductionReviewQueueBatchError(f"{field}.review boundary must remain unmeasured")
    if annotation.get("boundary_status") != "NOT_MEASURED":
        raise ProductionReviewQueueBatchError(
            f"{field}.review boundary status must be NOT_MEASURED"
        )
    for segment in _sequence(review.get("segments"), field=f"{field}.review.segments"):
        row = _mapping(segment, field=f"{field}.review.segments[]")
        if row.get("start_seconds") is not None or row.get("end_seconds") is not None:
            raise ProductionReviewQueueBatchError(
                f"{field}.review segment boundary must remain unmeasured"
            )
        if row.get("boundary_status") != "NOT_MEASURED":
            raise ProductionReviewQueueBatchError(
                f"{field}.review segment status must be NOT_MEASURED"
            )


def validate_review_queue_batches(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a generated queue plan and return compact validation metrics."""

    if plan.get("format") != FORMAT:
        raise ProductionReviewQueueBatchError(f"plan.format must be {FORMAT}")
    if plan.get("authority") != AUTHORITY:
        raise ProductionReviewQueueBatchError("plan authority must be LOCAL_NONPRODUCTION_ONLY")
    batches = _sequence(plan.get("batches"), field="plan.batches")
    seen: set[str] = set()
    item_count = 0
    boundary_unmeasured = 0
    top_k_complete = 0
    for batch_index, raw_batch in enumerate(batches):
        batch = _mapping(raw_batch, field=f"plan.batches[{batch_index}]")
        batch_items = _sequence(batch.get("items"), field=f"plan.batches[{batch_index}].items")
        for item_index, raw_item in enumerate(batch_items):
            item = _mapping(raw_item, field=f"plan.batches[{batch_index}].items[{item_index}]")
            _validate_item(item, field=f"plan.batches[{batch_index}].items[{item_index}]")
            window_id = _text(item.get("window_id"), field="queue item.window_id")
            if window_id in seen:
                raise ProductionReviewQueueBatchError(f"duplicate window_id: {window_id}")
            seen.add(window_id)
            item_count += 1
            boundary_unmeasured += int(
                item["review"]["annotation"]["boundary_status"] == "NOT_MEASURED"
            )
            top_k_complete += int(len(item["wemm"]["top_k"]) == REQUIRED_TOP_K)
    expected = plan.get("summary", {}).get("window_count")
    if isinstance(expected, int) and expected != item_count:
        raise ProductionReviewQueueBatchError(
            f"plan summary window_count={expected} does not match {item_count} items"
        )
    return {
        "status": "VALID",
        "batch_count": len(batches),
        "window_count": item_count,
        "unique_window_count": len(seen),
        "top_k6_count": top_k_complete,
        "review_boundary_unmeasured_count": boundary_unmeasured,
        "all_decisions_pending": True,
        "all_qwen_slots_present": all(
            isinstance(batch.get("items"), Sequence)
            for batch in batches
            if isinstance(batch, Mapping)
        ),
    }


def build_review_queue_batches(
    draft: Mapping[str, Any],
    *,
    batch_size: int = 10,
    draft_path: str | None = None,
    qwen_results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Chunk a validated draft into review batches of ``batch_size`` windows."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ProductionReviewQueueBatchError("batch_size must be a positive integer")
    if draft.get("format") != "robata-production-wemm-annotation-draft-v1":
        raise ProductionReviewQueueBatchError(
            "draft.format is not a production WeMM annotation draft"
        )
    if draft.get("production_eligible") is not False:
        raise ProductionReviewQueueBatchError("draft must remain non-production eligible")
    raw_windows = _sequence(draft.get("windows"), field="draft.windows")
    if not raw_windows:
        raise ProductionReviewQueueBatchError("draft.windows must not be empty")
    qwen_by_window = _qwen_index(qwen_results)
    windows = [
        _mapping(row, field=f"draft.windows[{index}]") for index, row in enumerate(raw_windows)
    ]
    batches: list[dict[str, Any]] = []
    for batch_start in range(0, len(windows), batch_size):
        batch_index = len(batches)
        batch_id = f"RQB{batch_index:03d}"
        chunk = windows[batch_start : batch_start + batch_size]
        items = [
            _item(
                window,
                queue_index=batch_start + item_index,
                batch_id=batch_id,
                qwen_by_window=qwen_by_window,
            )
            for item_index, window in enumerate(chunk)
        ]
        batches.append(
            {
                "batch_id": batch_id,
                "batch_index": batch_index,
                "start_queue_index": batch_start,
                "end_queue_index": batch_start + len(items) - 1,
                "item_count": len(items),
                "recording_ids": sorted({item["recording_id"] for item in items}),
                "window_ids": [item["window_id"] for item in items],
                "status": STATUS,
                "items": items,
            }
        )
    plan: dict[str, Any] = {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": STATUS,
        "source": {
            "draft_path": draft_path,
            "draft_format": draft.get("format"),
            "label_space": _copy_json(draft.get("label_space", {}), field="draft.label_space"),
            "epic_ontology_used": False,
            "mapper_used": False,
        },
        "policy": {
            "batch_size_windows": batch_size,
            "ordering": "canonical draft window order",
            "fixed_windows_are_context_only": True,
            "one_review_item_per_window": True,
            "qwen_slot_optional": True,
            "top_k_required": REQUIRED_TOP_K,
            "review_decisions": list(DECISION_OPTIONS),
        },
        "summary": {
            "recording_count": len(
                {item["recording_id"] for batch in batches for item in batch["items"]}
            ),
            "window_count": len(windows),
            "camera_window_input_count": int(
                draft.get("metrics", {}).get("camera_window_input_count", 0)
            ),
            "batch_count": len(batches),
            "batch_size_windows": batch_size,
            "last_batch_size": len(batches[-1]["items"]),
            "top_k6_count": len(windows),
            "review_boundary_status": {"NOT_MEASURED": len(windows)},
            "qwen_placeholder_count": len(windows) - len(qwen_by_window),
            "official_gold_status": draft.get("official_gold_status"),
            "official_quality_status": draft.get("official_quality_status"),
        },
        "review_contract": {
            "required_provenance_fields": list(REQUIRED_PROVENANCE_FIELDS),
            "decision_options": list(DECISION_OPTIONS),
            "review_boundary_fields": [
                "start_seconds",
                "end_seconds",
                "boundary_status",
                "timestamp_basis",
            ],
            "fixed_window_is_not_action_boundary": True,
            "model_outputs_are_not_gold": True,
            "qwen_result_is_optional": True,
        },
        "batches": batches,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "labels_inferred": False,
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
        },
    }
    validation = validate_review_queue_batches(plan)
    plan["validation"] = validation
    return plan


def render_markdown(
    plan: Mapping[str, Any], *, batch_paths: Mapping[str, str] | None = None
) -> str:
    summary = _mapping(plan.get("summary"), field="plan.summary")
    validation = _mapping(plan.get("validation"), field="plan.validation")
    lines = [
        "# Production review queue batches",
        "",
        f"- Status: **{plan.get('status')}**",
        f"- Coverage: **{summary.get('recording_count')} recordings / {summary.get('window_count')} windows / {summary.get('camera_window_input_count')} camera-window inputs**",
        f"- Batch size: **{summary.get('batch_size_windows')} windows**; batches: **{summary.get('batch_count')}**; last batch: **{summary.get('last_batch_size')}**",
        f"- Top-K=6: **{summary.get('top_k6_count')}/{summary.get('window_count')}**; review boundaries NOT_MEASURED: **{summary.get('review_boundary_status', {}).get('NOT_MEASURED', 0)}/{summary.get('window_count')}**",
        f"- Qwen placeholders: **{summary.get('qwen_placeholder_count')}**; no Qwen/model invocation by this planner",
        "",
        "## Contract",
        "",
        "- Every item carries source provenance, WeMM Top-K6 and margin, an optional Qwen slot, and pending accept/edit/split/reject/abstain review fields.",
        "- Reviewer action boundaries start/end are null with `NOT_MEASURED`; fixed context windows are never treated as action boundaries.",
        "- This is a planning/review artifact, not gold and not production-eligible.",
        "",
        "## Validation",
        "",
        f"- **{validation.get('status')}**: {validation.get('window_count')} unique windows, {validation.get('top_k6_count')} Top-K6, {validation.get('review_boundary_unmeasured_count')} unmeasured review boundaries",
        "",
        "## Batch files",
        "",
    ]
    for batch in plan.get("batches", []):
        if not isinstance(batch, Mapping):
            continue
        batch_id = str(batch.get("batch_id"))
        path = batch_paths.get(batch_id, "") if batch_paths else ""
        lines.append(
            f"- `{batch_id}`: {batch.get('item_count')} windows, recordings={len(batch.get('recording_ids', []))}, {path}".rstrip()
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "DECISION_OPTIONS",
    "FORMAT",
    "ProductionReviewQueueBatchError",
    "build_review_queue_batches",
    "render_markdown",
    "validate_review_queue_batches",
]
