"""Plan production WeMM pre-annotation work without loading a model.

This module is deliberately separate from the EPIC benchmark runners and from
the annotation/review envelope builders.  It turns a ZIP central-directory
inventory (or an already materialised inventory mapping) into a deterministic
work plan for the production recordings.  The plan contains *source media
references only*: it does not extract an archive, decode frames, invoke WeMM,
or invent an action vocabulary.

The resulting artifact is useful before a model run because it makes the batch
boundary and the QA admission rule explicit.  ``PASS`` and ``WARNING`` records
are schedulable; ``FAIL`` records are retained as excluded evidence; records
whose QA status is still ``PENDING`` remain unscheduled unless the caller
explicitly asks for a dry-run plan that includes them.  A later runner can
consume the ``batches`` list one recording at a time and write a separate
pre-annotation envelope.

No EPIC class IDs, Mapper decisions, model outputs, identities, hashes, or
digests are produced here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .production_corpus_audit import ProductionCorpusInventory, audit_zip_archive

PRODUCTION_WEMM_BATCH_PLAN_VERSION: Final = "robata-production-wemm-batch-plan-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
PLAN_STATUS: Final = "PLANNED_NONPRODUCTION"
LABEL_SPACE: Final = "OPEN_PROVISIONAL_PHRASES"
ELIGIBLE_QA_STATUSES: Final = frozenset({"PASS", "WARNING"})
KNOWN_QA_STATUSES: Final = frozenset({"PASS", "WARNING", "FAIL", "PENDING"})
DEFAULT_MAX_BATCH_BYTES: Final = 2 * 1024**3
DEFAULT_MAX_ITEMS_PER_BATCH: Final = 1


class ProductionWemmBatchPlanError(ValueError):
    """Raised when a production batch-plan input is malformed."""


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionWemmBatchPlanError(f"{field} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ProductionWemmBatchPlanError(f"{field} must be non-empty")
    return result


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionWemmBatchPlanError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionWemmBatchPlanError(f"{field} must be a non-negative integer")
    return value


def _qa_status(value: object, *, field: str) -> str:
    if value is None:
        return "PENDING"
    result = _text(value, field=field).upper().replace("_", " ")
    # Accept the short spelling used by a few QA sidecars while emitting one
    # canonical form.  Do not use a substring replacement (``WARNING`` must
    # not become ``WARNINGING``).
    if result == "WARN":
        result = "WARNING"
    if result not in KNOWN_QA_STATUSES:
        raise ProductionWemmBatchPlanError(
            f"{field} must be one of {', '.join(sorted(KNOWN_QA_STATUSES))}"
        )
    return result


def _normalise_member(value: object, *, field: str) -> str:
    member = _text(value, field=field).replace("\\", "/")
    if member.startswith("/") or "://" in member:
        raise ProductionWemmBatchPlanError(f"{field} must be an archive-relative member")
    parts = [part for part in member.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ProductionWemmBatchPlanError(f"{field} must be a normal archive member")
    normalised = "/".join(parts)
    if not normalised.casefold().endswith(".mcap"):
        raise ProductionWemmBatchPlanError(f"{field} must refer to an .mcap member")
    return normalised


def _copy_json(value: object, *, field: str) -> Any:
    """Copy bounded JSON-shaped metadata without deriving an identity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmBatchPlanError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(key): _copy_json(child, field=f"{field}.{key}") for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[]") for child in value]
    raise ProductionWemmBatchPlanError(f"{field} contains a non-JSON value")


@dataclass(frozen=True, slots=True)
class ProductionWemmSourceItem:
    """One archive-relative production recording and its admission state."""

    ordinal: int
    archive_member: str
    size_bytes: int
    compressed_size_bytes: int | None = None
    qa_status: str = "PENDING"
    batch_hint: str | None = None

    @property
    def schedulable_by_default(self) -> bool:
        return self.qa_status in ELIGIBLE_QA_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "archive_member": self.archive_member,
            "size_bytes": self.size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "qa_status": self.qa_status,
            "schedulable_by_default": self.schedulable_by_default,
            "batch_hint": self.batch_hint,
            "preannotation_status": "NOT_RUN",
            "gold_status": "NOT_ESTABLISHED",
        }


@dataclass(frozen=True, slots=True)
class ProductionWemmBatch:
    """A bounded group of source recordings for a future WeMM worker."""

    batch_id: str
    item_ordinals: tuple[int, ...]
    archive_members: tuple[str, ...]
    total_size_bytes: int
    largest_item_bytes: int
    oversize_item: bool
    batch_hint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "item_ordinals": list(self.item_ordinals),
            "archive_members": list(self.archive_members),
            "item_count": len(self.item_ordinals),
            "total_size_bytes": self.total_size_bytes,
            "largest_item_bytes": self.largest_item_bytes,
            "oversize_item": self.oversize_item,
            "batch_hint": self.batch_hint,
            "route": "wemm_video_preannotation",
            "execution": "serial_source_bound",
        }


def _entry_rows(value: object) -> Sequence[Mapping[str, Any]]:
    """Extract MCAP entry rows from supported inventory representations."""

    if isinstance(value, ProductionCorpusInventory):
        return tuple(item.to_dict() for item in value.mcap_entries)
    if not isinstance(value, Mapping):
        raise ProductionWemmBatchPlanError(
            "inventory must be a mapping or ProductionCorpusInventory"
        )

    # ``production_corpus_audit`` and ``inventory_production_training_corpus``
    # intentionally use different wrappers.  Support both without treating
    # either as a label source.
    candidates: object = value.get("mcap_entries")
    if candidates is None:
        archive = value.get("archive")
        if isinstance(archive, Mapping):
            candidates = [
                row
                for row in archive.get("entries", ())
                if isinstance(row, Mapping)
                and (
                    str(row.get("name", "")).casefold().endswith(".mcap")
                    or row.get("kind") == "media_mcap"
                )
            ]
    if candidates is None:
        candidates = value.get("entries")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
        raise ProductionWemmBatchPlanError(
            "inventory must contain an mcap_entries or archive.entries array"
        )
    rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise ProductionWemmBatchPlanError(f"inventory entry {index} must be an object")
        rows.append(raw)
    return tuple(rows)


def _source_items(
    inventory: Mapping[str, Any] | ProductionCorpusInventory,
    *,
    qa_status_by_member: Mapping[str, object] | None,
    batch_hints: Mapping[str, object] | None,
) -> tuple[ProductionWemmSourceItem, ...]:
    rows = _entry_rows(inventory)
    qa_lookup: dict[str, object] = {}
    if qa_status_by_member is not None:
        if not isinstance(qa_status_by_member, Mapping):
            raise ProductionWemmBatchPlanError("qa_status_by_member must be an object")
        for raw_name, status in qa_status_by_member.items():
            member = _normalise_member(raw_name, field="qa_status_by_member key")
            qa_lookup[member] = status
    hint_lookup: dict[str, object] = {}
    if batch_hints is not None:
        if not isinstance(batch_hints, Mapping):
            raise ProductionWemmBatchPlanError("batch_hints must be an object")
        for raw_name, hint in batch_hints.items():
            member = _normalise_member(raw_name, field="batch_hints key")
            hint_lookup[member] = hint

    items: list[ProductionWemmSourceItem] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        name = raw.get("archive_member", raw.get("name", raw.get("path")))
        member = _normalise_member(name, field=f"inventory[{index}].name")
        if member in seen:
            raise ProductionWemmBatchPlanError(f"duplicate MCAP member: {member}")
        seen.add(member)
        size_raw = raw.get("size_bytes", raw.get("size"))
        size = _non_negative_int(size_raw, field=f"inventory[{index}].size_bytes")
        ordinal_raw = raw.get("ordinal", index)
        ordinal = _non_negative_int(ordinal_raw, field=f"inventory[{index}].ordinal")
        compressed_raw = raw.get("compressed_size_bytes", raw.get("compressed_size"))
        compressed = (
            None
            if compressed_raw is None
            else _non_negative_int(
                compressed_raw, field=f"inventory[{index}].compressed_size_bytes"
            )
        )
        qa_value = qa_lookup.get(member, raw.get("qa_status", raw.get("qa", "PENDING")))
        hint_value = hint_lookup.get(member, raw.get("batch_hint"))
        hint = (
            None
            if hint_value is None
            else _text(hint_value, field=f"inventory[{index}].batch_hint")
        )
        items.append(
            ProductionWemmSourceItem(
                ordinal=ordinal,
                archive_member=member,
                size_bytes=size,
                compressed_size_bytes=compressed,
                qa_status=_qa_status(qa_value, field=f"inventory[{index}].qa_status"),
                batch_hint=hint,
            )
        )
    if not items:
        raise ProductionWemmBatchPlanError("inventory contains no MCAP recordings")
    items.sort(key=lambda item: (item.ordinal, item.archive_member))
    return tuple(items)


def _ordered_items(
    items: Sequence[ProductionWemmSourceItem],
    *,
    priority_ordinals: Sequence[int] | None,
) -> tuple[ProductionWemmSourceItem, ...]:
    if priority_ordinals is None:
        return tuple(items)
    if not isinstance(priority_ordinals, Sequence) or isinstance(
        priority_ordinals, (str, bytes, bytearray)
    ):
        raise ProductionWemmBatchPlanError("priority_ordinals must be an array")
    priorities: list[int] = []
    for index, raw in enumerate(priority_ordinals):
        priorities.append(_non_negative_int(raw, field=f"priority_ordinals[{index}]"))
    if len(set(priorities)) != len(priorities):
        raise ProductionWemmBatchPlanError("priority_ordinals must not contain duplicates")
    by_ordinal = {item.ordinal: item for item in items}
    unknown = sorted(set(priorities) - set(by_ordinal))
    if unknown:
        raise ProductionWemmBatchPlanError(
            "priority_ordinals contain unknown recordings: " + ", ".join(map(str, unknown))
        )
    chosen = [by_ordinal[ordinal] for ordinal in priorities]
    chosen_set = set(priorities)
    chosen.extend(item for item in items if item.ordinal not in chosen_set)
    return tuple(chosen)


def _build_batches(
    items: Sequence[ProductionWemmSourceItem],
    *,
    max_batch_bytes: int,
    max_items_per_batch: int,
    include_pending_qa: bool,
) -> tuple[ProductionWemmBatch, ...]:
    eligible = [
        item
        for item in items
        if item.schedulable_by_default or (include_pending_qa and item.qa_status == "PENDING")
    ]
    batches: list[ProductionWemmBatch] = []
    current: list[ProductionWemmSourceItem] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if not current:
            return
        batches.append(
            ProductionWemmBatch(
                batch_id=f"B{len(batches):03d}",
                item_ordinals=tuple(item.ordinal for item in current),
                archive_members=tuple(item.archive_member for item in current),
                total_size_bytes=current_size,
                largest_item_bytes=max(item.size_bytes for item in current),
                oversize_item=any(item.size_bytes > max_batch_bytes for item in current),
                batch_hint=current[0].batch_hint or "scheduled",
            )
        )
        current = []
        current_size = 0

    for item in eligible:
        would_overflow = current and (
            len(current) >= max_items_per_batch or current_size + item.size_bytes > max_batch_bytes
        )
        if would_overflow:
            flush()
        current.append(item)
        current_size += item.size_bytes
        # An item larger than the limit is intentionally emitted as a singleton;
        # rejecting it here would make the full-corpus plan impossible to run.
        if item.size_bytes > max_batch_bytes or len(current) >= max_items_per_batch:
            flush()
    flush()
    return tuple(batches)


def build_production_wemm_batch_plan(
    inventory: Mapping[str, Any] | ProductionCorpusInventory | str | Path,
    *,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    max_items_per_batch: int = DEFAULT_MAX_ITEMS_PER_BATCH,
    priority_ordinals: Sequence[int] | None = None,
    qa_status_by_member: Mapping[str, object] | None = None,
    batch_hints: Mapping[str, object] | None = None,
    include_pending_qa: bool = False,
) -> dict[str, Any]:
    """Build a deterministic, label-neutral production WeMM work plan.

    ``inventory`` may be a ZIP archive path, the output of
    :func:`audit_zip_archive`, or a :class:`ProductionCorpusInventory`.  When a
    path is supplied, only the archive central directory is inspected.  The
    caller may provide QA statuses from a separate sidecar; no status is
    inferred from a filename.
    """

    max_batch_bytes = _positive_int(max_batch_bytes, field="max_batch_bytes")
    max_items_per_batch = _positive_int(max_items_per_batch, field="max_items_per_batch")
    if not isinstance(include_pending_qa, bool):
        raise ProductionWemmBatchPlanError("include_pending_qa must be boolean")

    source_archive: str | None = None
    if isinstance(inventory, (str, Path)):
        archive = Path(inventory).expanduser().resolve()
        audited = audit_zip_archive(archive)
        source_archive = str(archive)
        source_inventory: Mapping[str, Any] | ProductionCorpusInventory = audited
    else:
        source_inventory = inventory
        if isinstance(inventory, Mapping):
            source = inventory.get("archive_path")
            if source is None and isinstance(inventory.get("archive"), Mapping):
                source = inventory["archive"].get("path")
            if isinstance(source, str) and source.strip():
                source_archive = source.strip()
        elif isinstance(inventory, ProductionCorpusInventory):
            source_archive = inventory.archive_path

    items = _source_items(
        source_inventory,
        qa_status_by_member=qa_status_by_member,
        batch_hints=batch_hints,
    )
    ordered = _ordered_items(items, priority_ordinals=priority_ordinals)
    batches = _build_batches(
        ordered,
        max_batch_bytes=max_batch_bytes,
        max_items_per_batch=max_items_per_batch,
        include_pending_qa=include_pending_qa,
    )
    scheduled_ordinals = {ordinal for batch in batches for ordinal in batch.item_ordinals}
    pending = [item for item in items if item.qa_status == "PENDING"]
    failed = [item for item in items if item.qa_status == "FAIL"]
    warnings = [item for item in items if item.qa_status == "WARNING"]
    passed = [item for item in items if item.qa_status == "PASS"]
    total_bytes = sum(item.size_bytes for item in items)
    scheduled_bytes = sum(item.size_bytes for item in items if item.ordinal in scheduled_ordinals)
    return {
        "format": PRODUCTION_WEMM_BATCH_PLAN_VERSION,
        "authority": AUTHORITY,
        "status": PLAN_STATUS,
        "source": {
            "archive_path": source_archive,
            "media_kind": "mcap",
            "inspection_mode": "central_directory_only" if source_archive else "inventory_mapping",
            "extracted": False,
            "frames_decoded": False,
        },
        "label_space": {
            "kind": LABEL_SPACE,
            "epic_ontology_used": False,
            "mapper_used": False,
            "vocabulary_status": "OPEN_UNREVIEWED",
            "unknown_allowed": True,
            "abstain_allowed": True,
            "split_allowed": True,
        },
        "policy": {
            "qa_pass_and_warning_enter_preannotation": True,
            "qa_fail_excluded": True,
            "pending_qa_included": include_pending_qa,
            "max_batch_bytes": max_batch_bytes,
            "max_items_per_batch": max_items_per_batch,
            "execution": "serial_source_bound",
            "raw_output_preserved": True,
        },
        "summary": {
            "recording_count": len(items),
            "scheduled_recording_count": len(scheduled_ordinals),
            "pending_qa_count": len(pending),
            "pass_count": len(passed),
            "warning_count": len(warnings),
            "fail_count": len(failed),
            "total_size_bytes": total_bytes,
            "scheduled_size_bytes": scheduled_bytes,
            "batch_count": len(batches),
            "gold_status": "NOT_ESTABLISHED",
            "quality_status": "NOT_MEASURED",
        },
        "items": [item.to_dict() for item in ordered],
        "batches": [batch.to_dict() for batch in batches],
        "controls": {
            "model_invoked": False,
            "wemm_invoked": False,
            "qwen_invoked": False,
            "mage_invoked": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "gold_read": False,
            "gold_written": False,
            "raw_candidates_overwritten": False,
            "identity_computation": "none",
        },
        "limitations": [
            "The plan is scheduling evidence, not an annotation or quality result.",
            "QA status is pending unless supplied by a separate source-bound QA sidecar.",
            "Action vocabulary is open/provisional; no EPIC class is a production label.",
            "Human review remains required after WeMM pre-annotation.",
        ],
    }


def load_qa_statuses(value: Mapping[str, Any] | str | Path) -> dict[str, object]:
    """Load a simple ``member -> QA status`` mapping for the CLI/helper."""

    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductionWemmBatchPlanError(f"could not read QA status file: {path}") from exc
    else:
        payload = value
    if not isinstance(payload, Mapping):
        raise ProductionWemmBatchPlanError("QA status file must contain an object")
    # Permit a wrapper with ``statuses`` while keeping the accepted format
    # intentionally tiny and easy to edit by hand.
    statuses = payload.get("statuses", payload)
    if not isinstance(statuses, Mapping):
        raise ProductionWemmBatchPlanError("QA status payload must contain a mapping")
    return {str(member): status for member, status in statuses.items()}


def render_markdown(plan: Mapping[str, Any]) -> str:
    """Render a compact human-readable scheduling report."""

    summary = plan.get("summary", {})
    if not isinstance(summary, Mapping):
        raise ProductionWemmBatchPlanError("plan.summary must be an object")
    lines = [
        "# Production WeMM batch plan",
        "",
        "> **PLANNED_NONPRODUCTION.** This is a source schedule, not annotation gold.",
        "",
        f"- Recordings: `{summary.get('recording_count', 0)}`",
        f"- Scheduled: `{summary.get('scheduled_recording_count', 0)}`",
        f"- Pending QA: `{summary.get('pending_qa_count', 0)}`",
        f"- Excluded fail: `{summary.get('fail_count', 0)}`",
        f"- Batches: `{summary.get('batch_count', 0)}`",
        "",
        "| Batch | Items | Size bytes | Oversize | Route |",
        "|---|---:|---:|---|---|",
    ]
    batches = plan.get("batches", [])
    if isinstance(batches, Sequence) and not isinstance(batches, (str, bytes, bytearray)):
        for raw in batches:
            if not isinstance(raw, Mapping):
                continue
            lines.append(
                f"| `{raw.get('batch_id', '')}` | {raw.get('item_count', 0)} | "
                f"{raw.get('total_size_bytes', 0)} | "
                f"{str(bool(raw.get('oversize_item'))).lower()} | "
                f"{raw.get('route', 'wemm_video_preannotation')} |"
            )
    lines.extend(
        [
            "",
            "The plan keeps WeMM output open/provisional and permits unknown, abstain, and split.",
            (
                "Human review must define the production vocabulary; no EPIC ontology "
                "or Mapper is used."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AUTHORITY",
    "DEFAULT_MAX_BATCH_BYTES",
    "DEFAULT_MAX_ITEMS_PER_BATCH",
    "ELIGIBLE_QA_STATUSES",
    "LABEL_SPACE",
    "PLAN_STATUS",
    "PRODUCTION_WEMM_BATCH_PLAN_VERSION",
    "ProductionWemmBatch",
    "ProductionWemmBatchPlanError",
    "ProductionWemmSourceItem",
    "build_production_wemm_batch_plan",
    "load_qa_statuses",
    "render_markdown",
]
