"""Aggregate per-recording WeMM review packs into one review-only queue.

The WeMM runner writes one ``robata-production-wemm-preannotation-review-pack-v1``
sidecar per recording.  Review tools often need a single queue spanning several
recordings, but simply concatenating the ``items`` arrays would lose the source
lineage (and makes duplicate window IDs hard to diagnose).  This module performs
that small, read-only join.

No media is opened and no model, Qwen/Mage result, gold annotation, ontology
mapping, or evaluator denominator is read.  Source metadata is retained under
``recordings`` and each flattened item carries a ``source_ref`` back to its
recording and input pack.  Processing-window intervals remain context only; the
aggregator never treats them as action boundaries.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .production_wemm_interval_proposal import (
    BOUNDARY_SOURCE as TEMPORAL_BOUNDARY_SOURCE,
)
from .production_wemm_interval_proposal import (
    BOUNDARY_STATUS as TEMPORAL_BOUNDARY_STATUS,
)
from .production_wemm_temporal import FORMAT as TEMPORAL_RESOLUTION_FORMAT
from .production_wemm_temporal_refinement import (
    APPLIED_FORMAT as TEMPORAL_REFINEMENT_APPLIED_FORMAT,
)
from .production_wemm_temporal_refinement import (
    FORMAT as TEMPORAL_REFINEMENT_PLAN_FORMAT,
)
from .production_wemm_temporal_score_refinement import (
    FORMAT as TEMPORAL_SCORE_REFINEMENT_FORMAT,
)
from .production_wemm_temporal_score_refinement import (
    RESULT_FORMAT as TEMPORAL_SCORE_RESULT_FORMAT,
)

REVIEW_PACK_FORMAT: Final = "robata-production-wemm-preannotation-review-pack-v1"
# Kept local instead of importing the batch runner so this read-only module
# cannot pull in optional media/model dependencies merely to resolve a
# checkpoint.  This is the wire-format value emitted by the runner.
BATCH_RUN_FORMAT: Final = "robata-production-wemm-batch-run-v1"
AGGREGATE_FORMAT: Final = "robata-production-wemm-review-pack-aggregate-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
DECISION_OPTIONS: Final = ("accept", "edit", "split", "reject", "abstain")
REFINED_BOUNDARY_STATUSES: Final = ("MODEL_REFINED", "MODEL_REFINEMENT_PENDING")
REFINED_BOUNDARY_STATUS: Final = "MODEL_REFINED"
TEMPORAL_BOUNDARY_METHODS: Final = frozenset(
    {"observed_probe_span", "probe_center_midpoint", "mixed_probe_boundary"}
)
TEMPORAL_REFINEMENT_BOUNDARY_SOURCE: Final = "wemm_short_refinement"
TEMPORAL_REFINEMENT_BOUNDARY_METHOD: Final = "short_probe_model"
_GOLD_KEYS: Final = frozenset(
    {
        "gold",
        "groundtruth",
        "officialgold",
        "officialreference",
        "officiallabel",
        "humanannotation",
        "adjudicatedlabel",
    }
)


class ProductionWemmReviewPackAggregateError(ValueError):
    """Raised for an invalid aggregate request or review-pack contract."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmReviewPackAggregateError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmReviewPackAggregateError(f"{field} must be an array")
    return value


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _copy_json(value: object, *, field: str = "value") -> Any:
    """Return a detached JSON-compatible copy without semantic projection."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmReviewPackAggregateError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmReviewPackAggregateError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionWemmReviewPackAggregateError(f"{field} must be JSON-compatible")


def _key_token(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _assert_no_gold(value: object, *, field: str) -> None:
    """Reject explicit gold containers; this is a field-separation check only."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProductionWemmReviewPackAggregateError(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmReviewPackAggregateError(f"{field} keys must be strings")
            token = _key_token(key)
            if token in _GOLD_KEYS or any(
                fragment in token for fragment in ("groundtruth", "humanannotation", "adjudicated")
            ):
                raise ProductionWemmReviewPackAggregateError(
                    f"{field}.{key} contains gold/review data"
                )
            _assert_no_gold(child, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_gold(child, field=f"{field}[{index}]")
        return
    raise ProductionWemmReviewPackAggregateError(f"{field} must be JSON-compatible")


def _load_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
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


def _discover_paths(path: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Find only review-pack JSON files, never preannotations or model outputs."""

    rejected: list[dict[str, str]] = []
    if path.is_file():
        return [path], rejected
    if not path.exists():
        return [], [{"path": str(path), "reason": "PATH_NOT_FOUND"}]
    if not path.is_dir():
        return [], [{"path": str(path), "reason": "PATH_NOT_FILE_OR_DIRECTORY"}]

    # A caller may point directly at a ``review`` directory, a batch root, or
    # the separate rebuild directory.  Restrict recursive discovery to review
    # directories and direct ``*review*.json`` files; preannotations,
    # checkpoints, aggregate reports and Qwen/Mage outputs are not selected.
    candidates: set[Path] = {
        candidate.resolve()
        for candidate in path.rglob("*.json")
        if candidate.is_file()
        and any(parent.name.casefold() == "review" for parent in candidate.parents)
    }
    candidates.update(
        candidate.resolve() for candidate in path.glob("*review*.json") if candidate.is_file()
    )
    sorted_candidates = sorted(candidates)
    if not sorted_candidates:
        rejected.append({"path": str(path), "reason": "NO_REVIEW_PACK_JSON_FOUND"})
    return sorted_candidates, rejected


def _checkpoint_rows(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    rejected: list[dict[str, str]],
) -> tuple[list[tuple[str, dict[str, Any] | None, str | None]], dict[str, Any]]:
    """Resolve completed review paths from one batch checkpoint.

    A batch checkpoint is a progress/control artifact, not a review pack.  We
    only follow ``review_path`` for items whose runner status is ``COMPLETE``;
    this prevents a unified queue from accidentally reading a file while an
    active runner is still writing it.  Missing/planned/running items remain
    visible in ``rejected_inputs`` and in the checkpoint provenance summary.
    """

    rows: list[tuple[str, dict[str, Any] | None, str | None]] = []
    raw_items = checkpoint.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
        rejected.append(
            {
                "path": f"{checkpoint_path}#items",
                "reason": "CHECKPOINT_ITEMS_NOT_ARRAY",
            }
        )
        return rows, {
            "path": str(checkpoint_path),
            "kind": "batch_checkpoint",
            "format": checkpoint.get("format"),
            "status": _text(checkpoint.get("status")) or "UNKNOWN",
            "item_count": 0,
            "completed_item_count": 0,
            "unresolved_item_count": 0,
        }

    completed_count = 0
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
        raw_review_path = raw_item.get("review_path")
        if status != "COMPLETE":
            unresolved_count += 1
            rejected.append(
                {
                    "path": f"{checkpoint_path}#items[{index}]",
                    "reason": f"CHECKPOINT_ITEM_NOT_COMPLETE:{status}",
                }
            )
            continue
        if not isinstance(raw_review_path, (str, Path)) or not str(raw_review_path).strip():
            unresolved_count += 1
            rejected.append(
                {
                    "path": f"{checkpoint_path}#items[{index}]",
                    "reason": "CHECKPOINT_ITEM_MISSING_REVIEW_PATH",
                }
            )
            continue
        sidecar_path = _resolve(str(raw_review_path), base=checkpoint_path.parent)
        payload, error = _load_file(sidecar_path)
        rows.append((str(sidecar_path), payload, error))
        completed_count += 1

    checkpoint_summary: Any = None
    raw_summary = checkpoint.get("summary")
    if isinstance(raw_summary, Mapping):
        try:
            checkpoint_summary = _copy_json(raw_summary, field=f"{checkpoint_path}.summary")
        except ProductionWemmReviewPackAggregateError as exc:
            rejected.append(
                {
                    "path": f"{checkpoint_path}.summary",
                    "reason": f"CHECKPOINT_SUMMARY_INVALID:{exc}",
                }
            )

    checkpoint_config: Any = None
    raw_config = checkpoint.get("config")
    if isinstance(raw_config, Mapping):
        try:
            checkpoint_config = _copy_json(raw_config, field=f"{checkpoint_path}.config")
        except ProductionWemmReviewPackAggregateError as exc:
            rejected.append(
                {
                    "path": f"{checkpoint_path}.config",
                    "reason": f"CHECKPOINT_CONFIG_INVALID:{exc}",
                }
            )

    return rows, {
        "path": str(checkpoint_path),
        "kind": "batch_checkpoint",
        "format": checkpoint.get("format"),
        "status": _text(checkpoint.get("status")) or "UNKNOWN",
        "item_count": len(raw_items),
        "completed_item_count": completed_count,
        "unresolved_item_count": unresolved_count,
        "summary": checkpoint_summary,
        "config": checkpoint_config,
    }


def _collect_path_rows(
    raw_path: str | Path,
    *,
    rejected: list[dict[str, str]],
    input_sources: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any] | None, str | None]]:
    """Collect review-pack rows from a file, directory, or batch checkpoint."""

    path = _resolve(raw_path)
    if path.is_dir():
        discovered, discovery_rejected = _discover_paths(path)
        rejected.extend(discovery_rejected)
        input_sources.append(
            {
                "path": str(path),
                "kind": "directory",
                "discovery": "review_directories_only",
                "discovered_pack_count": len(discovered),
            }
        )
        rows: list[tuple[str, dict[str, Any] | None, str | None]] = []
        for candidate in discovered:
            payload, error = _load_file(candidate)
            rows.append((str(candidate), payload, error))
        return rows

    if not path.exists():
        input_sources.append({"path": str(path), "kind": "path", "status": "NOT_FOUND"})
        rejected.append({"path": str(path), "reason": "PATH_NOT_FOUND"})
        return []
    if not path.is_file():
        input_sources.append({"path": str(path), "kind": "path", "status": "NOT_FILE"})
        rejected.append({"path": str(path), "reason": "PATH_NOT_FILE_OR_DIRECTORY"})
        return []

    payload, error = _load_file(path)
    if error is not None or payload is None:
        input_sources.append({"path": str(path), "kind": "file", "status": "READ_ERROR"})
        return [(str(path), payload, error)]
    if payload.get("format") == BATCH_RUN_FORMAT:
        checkpoint_rows, checkpoint_source = _checkpoint_rows(payload, path, rejected)
        input_sources.append(checkpoint_source)
        return checkpoint_rows
    input_sources.append(
        {
            "path": str(path),
            "kind": "review_pack_file",
            "format": payload.get("format"),
        }
    )
    return [(str(path), payload, None)]


def _validate_pack(payload: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    if payload.get("format") != REVIEW_PACK_FORMAT:
        raise ProductionWemmReviewPackAggregateError(
            f"{path}: format must be {REVIEW_PACK_FORMAT!r}"
        )
    if payload.get("authority") != AUTHORITY:
        raise ProductionWemmReviewPackAggregateError(f"{path}: authority must remain {AUTHORITY!r}")
    if payload.get("production_eligible") is not False:
        raise ProductionWemmReviewPackAggregateError(f"{path}: production_eligible must be false")
    if payload.get("official_gold_status") not in {
        "NOT_ESTABLISHED",
        "PENDING_HUMAN_REVIEW",
    }:
        raise ProductionWemmReviewPackAggregateError(
            f"{path}: official_gold_status must remain unestablished"
        )
    source = _mapping(payload.get("source"), field=f"{path}.source")
    recording_id = _text(source.get("recording_id") or source.get("source_id"))
    if recording_id is None:
        recording_id = _text(source.get("path"))
    if recording_id is None:
        raise ProductionWemmReviewPackAggregateError(
            f"{path}.source.recording_id (or source.path) is required"
        )
    label_space = _mapping(payload.get("label_space"), field=f"{path}.label_space")
    if label_space.get("epic_ontology_used") is True or label_space.get("mapper_used") is True:
        raise ProductionWemmReviewPackAggregateError(
            f"{path}: EPIC ontology/Mapper cannot enter production review"
        )
    items = _sequence(payload.get("items"), field=f"{path}.items")
    # Review-pack items are still model proposals, even though they are later
    # flattened into an aggregate.  Validate explicit routing flags here so a
    # malformed item cannot become an apparently eligible row downstream.
    for item_index, raw_item in enumerate(items):
        item = _mapping(raw_item, field=f"{path}.items[{item_index}]")
        if item.get("review_required", True) is not True:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.items[{item_index}].review_required must be true"
            )
        if "automatic_eligible" in item and item.get("automatic_eligible") is not False:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.items[{item_index}].automatic_eligible must be false"
            )
        window_id = _text(item.get("window_id"))
        if window_id is None:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.items[{item_index}].window_id must be non-empty"
            )
        proposals = _sequence(
            item.get("proposals", []),
            field=f"{path}.items[{item_index}].proposals",
        )
        source_interval = item.get("source_interval")
        if source_interval is not None:
            source_interval_map = _mapping(
                source_interval,
                field=f"{path}.items[{item_index}].source_interval",
            )
            status = _text(source_interval_map.get("status"))
            if status not in {"WINDOW_CONTEXT_ONLY", "NOT_MEASURED"}:
                raise ProductionWemmReviewPackAggregateError(
                    f"{path}.items[{item_index}].source_interval must remain context-only"
                )
        for proposal_index, raw_proposal in enumerate(proposals):
            proposal = _mapping(
                raw_proposal,
                field=f"{path}.items[{item_index}].proposals[{proposal_index}]",
            )
            proposal_field = f"{path}.items[{item_index}].proposals[{proposal_index}]"
            if proposal.get("review_required", True) is not True:
                raise ProductionWemmReviewPackAggregateError(
                    f"{proposal_field}.review_required must be true"
                )
            if proposal.get("automatic_eligible", False) is not False:
                raise ProductionWemmReviewPackAggregateError(
                    f"{proposal_field}.automatic_eligible must be false"
                )
            decision = proposal.get("decision", "pending")
            if decision not in DECISION_OPTIONS and decision != "pending":
                raise ProductionWemmReviewPackAggregateError(
                    f"{proposal_field}.decision is invalid"
                )
            _assert_no_gold(proposal, field=proposal_field)
    # Temporal interval estimates are optional and additive.  Validate their
    # review-only invariants here so an aggregate cannot accidentally promote
    # a model boundary to an automatic/gold result.
    _temporal_sidecar_parts(payload, path=path)
    _temporal_refinement_sidecar_parts(payload, path=path)
    controls = payload.get("controls")
    if controls is not None:
        controls_map = _mapping(controls, field=f"{path}.controls")
        for key in ("gold_read", "gold_written", "predictions_copied_to_gold"):
            if controls_map.get(key) is True:
                raise ProductionWemmReviewPackAggregateError(
                    f"{path}: controls.{key} must be false"
                )
    _assert_no_gold(payload, field=path)
    detached = _copy_json(payload, field=path)
    if not isinstance(detached, dict):  # pragma: no cover - _copy_json invariant
        raise ProductionWemmReviewPackAggregateError(f"{path}: pack must be an object")
    detached["_recording_id"] = recording_id
    detached["_item_count"] = len(items)
    return detached


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
    }


def _recording_source_ref(
    source: Mapping[str, Any], *, path: str, recording_id: str
) -> dict[str, Any]:
    """Keep stable, human-useful source lineage without rewriting source fields."""

    result = {
        "recording_id": recording_id,
        "review_pack_path": path,
        "source_path": source.get("path"),
        "archive_member": source.get("archive_member"),
        "archive_path": source.get("archive_path"),
        "source_preflight_status": source.get("source_preflight_status"),
        "qa_status": source.get("qa_status"),
        "manifest_format": source.get("manifest_format"),
    }
    # Retain the complete source snapshot as a sibling so no provenance is
    # lost when an upstream source adds a field.
    result["source"] = _copy_json(source, field=f"{path}.source")
    return result


def _merge_lineage(
    value: object,
    additions: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    """Add canonical aggregate lineage while retaining upstream extensions.

    Aggregate-owned keys must describe the pack currently being flattened.  An
    upstream sidecar may carry stale values for those keys, so ``setdefault``
    would make the flattened record internally inconsistent.  Preserve any
    conflicting upstream values under ``upstream_lineage`` instead of allowing
    them to shadow the canonical projection.
    """

    copied = _copy_json(value if isinstance(value, Mapping) else {}, field=field)
    if not isinstance(copied, dict):  # pragma: no cover - _copy_json invariant
        copied = {}
    conflicts: dict[str, Any] = {}
    for key, child in additions.items():
        canonical = _copy_json(child, field=f"{field}.{key}")
        if key in copied and copied[key] != canonical:
            conflicts[key] = copied[key]
        copied[key] = canonical
    if conflicts:
        existing_upstream = copied.get("upstream_lineage")
        if isinstance(existing_upstream, Mapping):
            upstream = _copy_json(existing_upstream, field=f"{field}.upstream_lineage")
        else:
            upstream = {}
            if existing_upstream is not None:
                # Preserve malformed-but-observable upstream lineage instead
                # of dropping it while repairing a canonical conflict.
                upstream["value"] = _copy_json(
                    existing_upstream,
                    field=f"{field}.upstream_lineage",
                )
        if not isinstance(upstream, dict):  # pragma: no cover - helper invariant
            upstream = {}
        for key, previous in conflicts.items():
            # Prefer the most immediate upstream value when nested lineage
            # layers disagree, so the value overridden by this aggregate is
            # not silently hidden behind an older snapshot.
            upstream[key] = previous
        copied["upstream_lineage"] = upstream
    return copied


def _temporal_sidecar_parts(
    payload: Mapping[str, Any], *, path: str
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Validate and detach the optional model-driven temporal sidecar.

    Temporal interval estimates are deliberately additive to the historical
    window review-pack shape.  Accept either the full ``temporal_resolution``
    object emitted by the runner, its compact ``temporal_segments`` alias, or
    both.  When both are present they must describe the same ordered segment
    list; this catches stale sidecar copies without changing the window queue.
    """

    raw_resolution = payload.get("temporal_resolution")
    raw_alias = payload.get("temporal_segments")
    resolution_copy: dict[str, Any] | None = None
    resolution_segments: list[dict[str, Any]] = []

    def _validate_segment(raw_segment: Mapping[str, Any], *, field: str) -> dict[str, Any]:
        """Validate one model-bound interval using the preannotation contract.

        The aggregate is a read-only join, but it is also a trust boundary for
        review clients.  Keep these checks identical to the producer-side
        validator: a sidecar segment must carry explicit review-only flags,
        provenance, and a finite non-degenerate source-relative interval.
        Marker fields introduced after the first sidecars are defaulted when
        absent; explicit contradictory values are still rejected.
        """

        if raw_segment.get("review_required") is not True:
            raise ProductionWemmReviewPackAggregateError(f"{field}.review_required must be true")
        if raw_segment.get("automatic_eligible") is not False:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.automatic_eligible must be false"
            )
        segment_id = _text(raw_segment.get("segment_id"))
        if segment_id is None:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.segment_id must be a non-empty string"
            )
        if raw_segment.get("boundary_status") != TEMPORAL_BOUNDARY_STATUS:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.boundary_status must be {TEMPORAL_BOUNDARY_STATUS}"
            )
        # These markers were added after the first temporal sidecars were
        # generated.  Missing values are a known legacy shape and are
        # canonicalised below; an explicit contradictory value is still an
        # invalid boundary claim.
        for key in ("context_only", "window_context_only"):
            if key in raw_segment and raw_segment[key] is not True:
                raise ProductionWemmReviewPackAggregateError(f"{field}.{key} must be true")
        for key in ("is_action_boundary", "action_boundary"):
            if key in raw_segment and raw_segment[key] is not False:
                raise ProductionWemmReviewPackAggregateError(f"{field}.{key} must be false")
        start = _finite(raw_segment.get("start_seconds"))
        end = _finite(raw_segment.get("end_seconds"))
        if start is None or end is None or start < 0.0 or end <= start:
            raise ProductionWemmReviewPackAggregateError(
                f"{field} interval must satisfy 0 <= start < end"
            )
        if raw_segment.get("boundary_source") != TEMPORAL_BOUNDARY_SOURCE:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.boundary_source must be {TEMPORAL_BOUNDARY_SOURCE!r}"
            )
        boundary_method = raw_segment.get("boundary_method")
        if _text(boundary_method) is None:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.boundary_method must be a non-empty string"
            )
        if str(boundary_method).strip() not in TEMPORAL_BOUNDARY_METHODS:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.boundary_method must be one of "
                f"{', '.join(sorted(TEMPORAL_BOUNDARY_METHODS))}"
            )
        checked = dict(raw_segment)
        checked.setdefault("context_only", True)
        checked.setdefault("window_context_only", True)
        checked.setdefault("is_action_boundary", False)
        checked.setdefault("action_boundary", False)
        return checked

    if raw_resolution is not None:
        resolution = _mapping(raw_resolution, field=f"{path}.temporal_resolution")
        if resolution.get("format") != TEMPORAL_RESOLUTION_FORMAT:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}: temporal_resolution.format must be {TEMPORAL_RESOLUTION_FORMAT!r}"
            )
        if resolution.get("authority") != AUTHORITY:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}: temporal_resolution.authority must remain {AUTHORITY!r}"
            )
        if resolution.get("status") != "PROPOSALS_ONLY":
            raise ProductionWemmReviewPackAggregateError(
                f"{path}: temporal_resolution.status must be PROPOSALS_ONLY"
            )
        if resolution.get("production_eligible") is not False:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}: temporal_resolution.production_eligible must be false"
            )
        context = _mapping(
            resolution.get("context_interval"),
            field=f"{path}.temporal_resolution.context_interval",
        )
        if "context_only" in context and context.get("context_only") is not True:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.temporal_resolution.context_interval.context_only must be true"
            )
        if "is_action_boundary" in context and context.get("is_action_boundary") is not False:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.temporal_resolution.context_interval.is_action_boundary must be false"
            )
        if "action_boundary" in context and context.get("action_boundary") is not False:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.temporal_resolution.context_interval.action_boundary must be false"
            )
        copied = _copy_json(resolution, field=f"{path}.temporal_resolution")
        if not isinstance(copied, dict):  # pragma: no cover - _copy_json invariant
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.temporal_resolution must be an object"
            )
        context_copy = copied.get("context_interval")
        if isinstance(context_copy, dict):
            context_copy.setdefault("context_only", True)
            context_copy.setdefault("is_action_boundary", False)
            context_copy.setdefault("action_boundary", False)
        raw_segments = copied.get("segments", [])
        if not isinstance(raw_segments, list):
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.temporal_resolution.segments must be an array"
            )
        seen_ids: set[str] = set()
        validated_segments: list[dict[str, Any]] = []
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, Mapping):
                raise ProductionWemmReviewPackAggregateError(
                    f"{path}.temporal_resolution.segments[{index}] must be an object"
                )
            checked = _validate_segment(
                raw_segment,
                field=f"{path}.temporal_resolution.segments[{index}]",
            )
            segment_id = str(checked["segment_id"])
            if segment_id in seen_ids:
                raise ProductionWemmReviewPackAggregateError(
                    f"{path}.temporal_resolution.segments[{index}].segment_id must be unique"
                )
            seen_ids.add(segment_id)
            validated_segments.append(checked)
        resolution_copy = copied
        detached_segments = _copy_json(
            validated_segments, field=f"{path}.temporal_resolution.segments"
        )
        if not isinstance(detached_segments, list):  # pragma: no cover - _copy_json invariant
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.temporal_resolution.segments must be an array"
            )
        copied["segments"] = detached_segments
        resolution_segments = detached_segments

    alias_segments: list[dict[str, Any]] = []
    if raw_alias is not None:
        alias = _sequence(raw_alias, field=f"{path}.temporal_segments")
        copied_alias = _copy_json(alias, field=f"{path}.temporal_segments")
        if not isinstance(copied_alias, list):  # pragma: no cover - _copy_json invariant
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.temporal_segments must be an array"
            )
        seen_ids = set()
        validated_alias: list[dict[str, Any]] = []
        for index, raw_segment in enumerate(copied_alias):
            if not isinstance(raw_segment, Mapping):
                raise ProductionWemmReviewPackAggregateError(
                    f"{path}.temporal_segments[{index}] must be an object"
                )
            checked = _validate_segment(raw_segment, field=f"{path}.temporal_segments[{index}]")
            segment_id = str(checked["segment_id"])
            if segment_id in seen_ids:
                raise ProductionWemmReviewPackAggregateError(
                    f"{path}.temporal_segments[{index}].segment_id must be unique"
                )
            seen_ids.add(segment_id)
            validated_alias.append(checked)
        alias_segments = validated_alias

    if (
        resolution_copy is not None
        and raw_alias is not None
        and resolution_segments != alias_segments
    ):
        raise ProductionWemmReviewPackAggregateError(
            f"{path}: temporal_segments does not match temporal_resolution.segments"
        )
    return resolution_copy, resolution_segments if resolution_copy is not None else alias_segments


def _temporal_refinement_sidecar_parts(payload: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    """Validate and detach adaptive temporal sidecars from one review pack.

    Adaptive fields are deliberately independent from the historical coarse
    temporal sidecar.  The planner/fine-score schemas are owned by their
    producer modules, so this aggregate only checks JSON safety, lineage
    carriers, and the review-only boundary invariants.  All aliases must
    describe the same ordered refined rows when supplied together.
    """

    sidecars: dict[str, dict[str, Any]] = {}
    expected_formats = {
        "temporal_refinement_plan": TEMPORAL_REFINEMENT_PLAN_FORMAT,
        "temporal_refinement_fine_plan": TEMPORAL_SCORE_REFINEMENT_FORMAT,
        "temporal_refinement_score_resolution": TEMPORAL_SCORE_RESULT_FORMAT,
        "temporal_refinement": TEMPORAL_REFINEMENT_APPLIED_FORMAT,
    }
    for key in (
        "temporal_refinement_plan",
        "temporal_refinement_fine_plan",
        "temporal_refinement_score_resolution",
        "temporal_refinement",
    ):
        if key not in payload:
            continue
        raw = _mapping(payload.get(key), field=f"{path}.{key}")
        expected_format = expected_formats[key]
        if raw.get("format") != expected_format:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.{key}.format must be {expected_format!r}"
            )
        if raw.get("authority") != AUTHORITY:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.{key}.authority must remain {AUTHORITY!r}"
            )
        _assert_no_gold(raw, field=f"{path}.{key}")
        copied = _copy_json(raw, field=f"{path}.{key}")
        if not isinstance(copied, dict):  # pragma: no cover - helper invariant
            raise ProductionWemmReviewPackAggregateError(f"{path}.{key} must be an object")
        if copied.get("production_eligible") is not False:
            raise ProductionWemmReviewPackAggregateError(
                f"{path}.{key}.production_eligible must be false"
            )
        sidecars[key] = copied

    candidates: list[tuple[str, Sequence[Any]]] = []
    for key in (
        "refined_segments",
        "refined_temporal_segments",
        "temporal_refinement_segments",
    ):
        if key in payload:
            candidates.append((key, _sequence(payload.get(key), field=f"{path}.{key}")))
    applied = sidecars.get("temporal_refinement")
    if applied is not None:
        for key in ("refined_segments", "temporal_refinement_segments", "segments"):
            if key in applied:
                candidates.append(
                    (
                        f"temporal_refinement.{key}",
                        _sequence(applied.get(key), field=f"{path}.temporal_refinement.{key}"),
                    )
                )

    def _validate_refined(raw: object, *, field: str) -> dict[str, Any]:
        row = _mapping(raw, field=field)
        row_id = _text(row.get("segment_id"))
        if row_id is None:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.segment_id must be a non-empty string"
            )
        for key in ("context_only", "window_context_only"):
            if key in row and row.get(key) is not True:
                raise ProductionWemmReviewPackAggregateError(f"{field}.{key} must be true")
        for key in ("is_action_boundary", "action_boundary"):
            if key in row and row.get(key) is not False:
                raise ProductionWemmReviewPackAggregateError(f"{field}.{key} must be false")
        if row.get("boundary_source") != TEMPORAL_REFINEMENT_BOUNDARY_SOURCE:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.boundary_source must be {TEMPORAL_REFINEMENT_BOUNDARY_SOURCE!r}"
            )
        if row.get("boundary_method") != TEMPORAL_REFINEMENT_BOUNDARY_METHOD:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.boundary_method must be {TEMPORAL_REFINEMENT_BOUNDARY_METHOD!r}"
            )
        if row.get("review_required") is not True:
            raise ProductionWemmReviewPackAggregateError(f"{field}.review_required must be true")
        if row.get("automatic_eligible") is not False:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.automatic_eligible must be false"
            )
        boundary_status = _text(row.get("boundary_status"))
        if boundary_status is not None:
            boundary_status = boundary_status.upper()
        if boundary_status not in REFINED_BOUNDARY_STATUSES:
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.boundary_status must be one of {', '.join(REFINED_BOUNDARY_STATUSES)}"
            )
        coarse = _mapping(row.get("coarse_interval"), field=f"{field}.coarse_interval")
        coarse_start = _finite(coarse.get("start_seconds"))
        coarse_end = _finite(coarse.get("end_seconds"))
        if (
            coarse_start is None
            or coarse_end is None
            or coarse_start < 0.0
            or coarse_end <= coarse_start
        ):
            raise ProductionWemmReviewPackAggregateError(
                f"{field}.coarse_interval must satisfy 0 <= start < end"
            )
        start = _finite(row.get("start_seconds"))
        end = _finite(row.get("end_seconds"))
        if boundary_status == REFINED_BOUNDARY_STATUS:
            if start is None or end is None or start < 0.0 or end <= start:
                raise ProductionWemmReviewPackAggregateError(
                    f"{field} interval must satisfy 0 <= start < end"
                )
        elif start is not None or end is not None:
            raise ProductionWemmReviewPackAggregateError(
                f"{field} pending rows must omit start/end"
            )
        _assert_no_gold(row, field=field)
        copied = _copy_json(row, field=field)
        if not isinstance(copied, dict):  # pragma: no cover - helper invariant
            raise ProductionWemmReviewPackAggregateError(f"{field} must be an object")
        copied.setdefault("context_only", True)
        copied.setdefault("window_context_only", True)
        copied.setdefault("is_action_boundary", False)
        copied.setdefault("action_boundary", False)
        copied["boundary_status"] = boundary_status
        return copied

    normalised: list[tuple[str, list[dict[str, Any]]]] = []
    for key, rows in candidates:
        seen_ids: set[str] = set()
        checked_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            checked = _validate_refined(row, field=f"{path}.{key}[{index}]")
            row_id = str(checked["segment_id"])
            if row_id in seen_ids:
                raise ProductionWemmReviewPackAggregateError(
                    f"{path}.{key}[{index}].segment_id must be unique"
                )
            seen_ids.add(row_id)
            checked_rows.append(checked)
        normalised.append(
            (
                key,
                checked_rows,
            )
        )
    if normalised:
        first_key, refined_rows = normalised[0]
        for key, rows in normalised[1:]:
            if rows != refined_rows:
                raise ProductionWemmReviewPackAggregateError(
                    f"{path}.{key} does not match {path}.{first_key}"
                )
    else:
        refined_rows = []

    # Ensure the nested applied sidecar carries the same row snapshot when it
    # exposes one.  This check catches a stale top-level alias before a batch
    # aggregate can flatten divergent evidence.
    if applied is not None and "refined_segments" in applied:
        applied["refined_segments"] = _copy_json(
            refined_rows, field=f"{path}.temporal_refinement.refined_segments"
        )
    return {
        "plan": sidecars.get("temporal_refinement_plan"),
        "fine_plan": sidecars.get("temporal_refinement_fine_plan"),
        "score_resolution": sidecars.get("temporal_refinement_score_resolution"),
        "applied": applied,
        "segments": refined_rows,
        "present": bool(sidecars or candidates),
    }


def aggregate_production_wemm_review_packs(
    inputs: Mapping[str, Any] | Sequence[Any] | str | Path,
    *,
    expected_camera_count: int = 6,
) -> dict[str, Any]:
    """Flatten completed per-recording review packs into one review-only queue.

    ``inputs`` may be a review-pack object, a sequence of objects/paths,
    one path, a directory, or a batch checkpoint.  A sequence can contain
    multiple directories/checkpoints (for example B1 and ``remaining``),
    which are flattened into one queue.  Invalid/non-review files are
    reported and skipped; no media or model route is touched.
    """

    if (
        isinstance(expected_camera_count, bool)
        or not isinstance(expected_camera_count, int)
        or expected_camera_count <= 0
    ):
        raise ProductionWemmReviewPackAggregateError(
            "expected_camera_count must be a positive integer"
        )

    rows: list[tuple[str, Mapping[str, Any] | None, str | None]] = []
    rejected_inputs: list[dict[str, str]] = []
    input_sources: list[dict[str, Any]] = []

    if isinstance(inputs, Mapping):
        if inputs.get("format") == BATCH_RUN_FORMAT:
            checkpoint_rows, checkpoint_source = _checkpoint_rows(
                inputs, Path("<mapping-checkpoint>"), rejected_inputs
            )
            input_sources.append(checkpoint_source)
            rows.extend(checkpoint_rows)
        else:
            input_sources.append(
                {"path": "<mapping>", "kind": "review_pack_object", "format": inputs.get("format")}
            )
            rows = [("<mapping>", inputs, None)]
    elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes, bytearray, Path)):
        for index, raw in enumerate(inputs):
            if isinstance(raw, Mapping):
                if raw.get("format") == BATCH_RUN_FORMAT:
                    checkpoint_rows, checkpoint_source = _checkpoint_rows(
                        raw, Path(f"<sequence>[{index}]-checkpoint"), rejected_inputs
                    )
                    input_sources.append(checkpoint_source)
                    rows.extend(checkpoint_rows)
                else:
                    input_sources.append(
                        {
                            "path": f"<sequence>[{index}]",
                            "kind": "review_pack_object",
                            "format": raw.get("format"),
                        }
                    )
                    rows.append((f"<sequence>[{index}]", raw, None))
            elif isinstance(raw, (str, Path)):
                rows.extend(
                    _collect_path_rows(
                        raw,
                        rejected=rejected_inputs,
                        input_sources=input_sources,
                    )
                )
            else:
                rejected_inputs.append(
                    {"path": f"<sequence>[{index}]", "reason": "INPUT_NOT_OBJECT_OR_PATH"}
                )
    else:
        rows.extend(
            _collect_path_rows(
                inputs,  # type: ignore[arg-type]
                rejected=rejected_inputs,
                input_sources=input_sources,
            )
        )

    valid: list[tuple[str, dict[str, Any]]] = []
    invalid_inputs: list[dict[str, str]] = []
    for path, payload, error in rows:
        if error is not None or payload is None:
            invalid_inputs.append({"path": path, "reason": error or "INVALID_JSON"})
            continue
        if payload.get("format") != REVIEW_PACK_FORMAT:
            rejected_inputs.append({"path": path, "reason": "ROOT_NOT_REVIEW_PACK"})
            continue
        try:
            checked = _validate_pack(payload, path=path)
        except (ProductionWemmReviewPackAggregateError, TypeError, ValueError) as exc:
            invalid_inputs.append({"path": path, "reason": str(exc)})
            continue
        valid.append((path, checked))

    # Stable ordering makes a generated queue reproducible while preserving
    # the source path in every item.  Explicit in-memory inputs retain their
    # sequence order through the path key.
    valid.sort(key=lambda pair: (str(pair[1].get("_recording_id", "")), pair[0]))

    recordings: list[dict[str, Any]] = []
    flattened_items: list[dict[str, Any]] = []
    recording_ids: Counter[str] = Counter()
    window_keys: Counter[str] = Counter()
    review_statuses: Counter[str] = Counter()
    window_statuses: Counter[str] = Counter()
    window_decisions: Counter[str] = Counter()
    proposal_decisions: Counter[str] = Counter()
    source_interval_statuses: Counter[str] = Counter()
    topk_cardinality: Counter[str] = Counter()
    option_counts: Counter[str] = Counter()
    margins: list[float] = []
    proposal_count = 0
    camera_window_count = 0
    observed_camera_ids: set[str] = set()
    model_names: Counter[str] = Counter()
    model_routes: Counter[str] = Counter()
    catalog_formats: Counter[str] = Counter()
    temporal_boundary_statuses: Counter[str] = Counter()
    temporal_sidecar_recordings: list[dict[str, Any]] = []
    flattened_temporal_segments: list[dict[str, Any]] = []
    temporal_refinement_recordings: list[dict[str, Any]] = []
    flattened_temporal_refinement_segments: list[dict[str, Any]] = []
    flattened_refined_temporal_interval_proposals: list[dict[str, Any]] = []
    refined_temporal_boundary_statuses: Counter[str] = Counter()
    duplicate_recordings: list[str] = []
    duplicate_window_keys: list[str] = []

    for recording_index, (pack_path, pack) in enumerate(valid):
        recording_id = str(pack.pop("_recording_id"))
        item_count = int(pack.pop("_item_count", 0))
        source = _mapping(pack.get("source"), field=f"{pack_path}.source")
        recording_ids[recording_id] += 1
        source_ref = _recording_source_ref(source, path=pack_path, recording_id=recording_id)
        model_artifact = _copy_json(pack.get("model_artifact"), field=f"{pack_path}.model_artifact")
        label_space = _copy_json(pack.get("label_space"), field=f"{pack_path}.label_space")
        if isinstance(model_artifact, Mapping):
            catalog = model_artifact.get("catalog")
            if isinstance(catalog, Mapping):
                catalog_format = _text(catalog.get("format"))
                if catalog_format:
                    catalog_formats[catalog_format] += 1
        model = pack.get("model")
        if isinstance(model, Mapping):
            if _text(model.get("name")):
                model_names[str(model["name"])] += 1
            if _text(model.get("route")):
                model_routes[str(model["route"])] += 1
        temporal_resolution, temporal_segments = _temporal_sidecar_parts(pack, path=pack_path)
        if temporal_resolution is not None or "temporal_segments" in pack:
            temporal_sidecar_recordings.append(
                {
                    "recording_id": recording_id,
                    "review_pack_path": pack_path,
                    "format": (
                        temporal_resolution.get("format")
                        if temporal_resolution is not None
                        else None
                    ),
                    "mode": (
                        temporal_resolution.get("mode") if temporal_resolution is not None else None
                    ),
                    "status": (
                        temporal_resolution.get("status")
                        if temporal_resolution is not None
                        else "PROPOSALS_ONLY"
                    ),
                    "segment_count": len(temporal_segments),
                    "parameters": _copy_json(
                        temporal_resolution.get("parameters", {}),
                        field=f"{pack_path}.temporal_resolution.parameters",
                    )
                    if temporal_resolution is not None
                    else {},
                    "diagnostics": _copy_json(
                        temporal_resolution.get("diagnostics", {}),
                        field=f"{pack_path}.temporal_resolution.diagnostics",
                    )
                    if temporal_resolution is not None
                    else {},
                    # Keep the complete detached sidecar available to audit
                    # consumers; the compact fields above remain convenient
                    # summary projections.
                    "sidecar": _copy_json(
                        temporal_resolution,
                        field=f"{pack_path}.temporal_resolution",
                    ),
                }
            )
            for segment_index, raw_segment in enumerate(temporal_segments):
                segment = _copy_json(
                    raw_segment,
                    field=f"{pack_path}.temporal_segments[{segment_index}]",
                )
                if not isinstance(segment, dict):  # pragma: no cover - helper invariant
                    raise ProductionWemmReviewPackAggregateError(
                        f"{pack_path}.temporal_segments[{segment_index}] must be an object"
                    )
                segment_id = _text(segment.get("segment_id"))
                if segment_id is None:  # pragma: no cover - validated upstream
                    raise ProductionWemmReviewPackAggregateError(
                        f"{pack_path}.temporal_segments[{segment_index}].segment_id is required"
                    )
                temporal_key = f"{recording_id}::temporal::{segment_id}"
                boundary_status = _text(segment.get("boundary_status"))
                if boundary_status:
                    temporal_boundary_statuses[boundary_status.upper()] += 1
                segment["recording_id"] = recording_id
                segment["recording_index"] = recording_index
                segment["temporal_segment_key"] = temporal_key
                segment["source_ref"] = _merge_lineage(
                    segment.get("source_ref"),
                    {
                        "recording_id": recording_id,
                        "review_pack_path": pack_path,
                        "source_path": source.get("path"),
                        "archive_member": source.get("archive_member"),
                    },
                    field=f"{pack_path}.temporal_segments[{segment_index}].source_ref",
                )
                segment["provenance"] = _merge_lineage(
                    segment.get("provenance"),
                    {
                        "review_pack_path": pack_path,
                        "window_context_only": True,
                        "model_boundary_sidecar": True,
                    },
                    field=f"{pack_path}.temporal_segments[{segment_index}].provenance",
                )
                flattened_temporal_segments.append(segment)
        temporal_refinement = _temporal_refinement_sidecar_parts(pack, path=pack_path)
        refined_rows = temporal_refinement["segments"]
        if temporal_refinement.get("present"):
            refined_measured_rows: list[dict[str, Any]] = []
            for segment_index, raw_segment in enumerate(refined_rows):
                segment = _copy_json(
                    raw_segment,
                    field=f"{pack_path}.temporal_refinement_segments[{segment_index}]",
                )
                if not isinstance(segment, dict):  # pragma: no cover - helper invariant
                    raise ProductionWemmReviewPackAggregateError(
                        f"{pack_path}.temporal_refinement_segments[{segment_index}] "
                        "must be an object"
                    )
                segment_id = _text(segment.get("segment_id"))
                if segment_id is None:  # pragma: no cover - validated upstream
                    raise ProductionWemmReviewPackAggregateError(
                        f"{pack_path}.temporal_refinement_segments[{segment_index}].segment_id "
                        "is required"
                    )
                temporal_key = f"{recording_id}::temporal-refined::{segment_id}"
                boundary_status = _text(segment.get("boundary_status"))
                if boundary_status:
                    refined_temporal_boundary_statuses[boundary_status.upper()] += 1
                segment["recording_id"] = recording_id
                segment["recording_index"] = recording_index
                segment["refined_temporal_segment_key"] = temporal_key
                segment["source_ref"] = _merge_lineage(
                    segment.get("source_ref"),
                    {
                        "recording_id": recording_id,
                        "review_pack_path": pack_path,
                        "source_path": source.get("path"),
                        "archive_member": source.get("archive_member"),
                    },
                    field=(f"{pack_path}.temporal_refinement_segments[{segment_index}].source_ref"),
                )
                segment["provenance"] = _merge_lineage(
                    segment.get("provenance"),
                    {
                        "review_pack_path": pack_path,
                        "window_context_only": True,
                        "model_boundary_sidecar": False,
                        "model_refinement_sidecar": True,
                    },
                    field=(f"{pack_path}.temporal_refinement_segments[{segment_index}].provenance"),
                )
                flattened_temporal_refinement_segments.append(segment)
                if boundary_status == REFINED_BOUNDARY_STATUS:
                    refined_measured_rows.append(segment)
            flattened_refined_temporal_interval_proposals.extend(refined_measured_rows)
            temporal_refinement_recordings.append(
                {
                    "recording_id": recording_id,
                    "review_pack_path": pack_path,
                    "plan": _copy_json(
                        temporal_refinement.get("plan"),
                        field=f"{pack_path}.temporal_refinement_plan",
                    ),
                    "fine_plan": _copy_json(
                        temporal_refinement.get("fine_plan"),
                        field=f"{pack_path}.temporal_refinement_fine_plan",
                    ),
                    "score_resolution": _copy_json(
                        temporal_refinement.get("score_resolution"),
                        field=f"{pack_path}.temporal_refinement_score_resolution",
                    ),
                    "applied": _copy_json(
                        temporal_refinement.get("applied"),
                        field=f"{pack_path}.temporal_refinement",
                    ),
                    "refined_segment_count": len(refined_rows),
                    "measured_segment_count": len(refined_measured_rows),
                    "pending_segment_count": len(refined_rows) - len(refined_measured_rows),
                }
            )
        provenance = {
            "review_pack_path": pack_path,
            "recording_id": recording_id,
            "source": source_ref,
            "label_space": label_space,
            "model_artifact": model_artifact,
            "item_count": item_count,
            "temporal_resolution": (
                {
                    "format": temporal_resolution.get("format"),
                    "mode": temporal_resolution.get("mode"),
                    "status": temporal_resolution.get("status"),
                    "segment_count": len(temporal_segments),
                }
                if temporal_resolution is not None
                else (
                    {"segment_count": len(temporal_segments)}
                    if "temporal_segments" in pack
                    else None
                )
            ),
            "temporal_refinement": (
                {
                    "segment_count": len(refined_rows),
                    "measured_segment_count": sum(
                        1
                        for row in refined_rows
                        if _text(row.get("boundary_status")) == REFINED_BOUNDARY_STATUS
                    ),
                    "present": bool(temporal_refinement.get("present")),
                }
                if temporal_refinement.get("present")
                else None
            ),
        }
        recordings.append(provenance)

        items = _sequence(pack.get("items"), field=f"{pack_path}.items")
        for item_index, raw_item in enumerate(items):
            item = _mapping(raw_item, field=f"{pack_path}.items[{item_index}]")
            window_id = _text(item.get("window_id"))
            if window_id is None:
                invalid_inputs.append(
                    {
                        "path": f"{pack_path}.items[{item_index}]",
                        "reason": "window_id is required",
                    }
                )
                continue
            item_key = f"{recording_id}::{window_id}"
            window_keys[item_key] += 1
            camera_ids = [
                str(camera)
                for camera in _sequence(item.get("camera_ids", []), field=f"{item_key}.camera_ids")
                if _text(camera)
            ]
            observed_camera_ids.update(camera_ids)
            camera_window_count += len(camera_ids)
            review_status = str(item.get("review_status") or "PENDING").upper()
            review_statuses[review_status] += 1
            window_status = _text(item.get("window_status"))
            if window_status:
                window_statuses[window_status.upper()] += 1
            window_decision = _text(item.get("window_decision"))
            if window_decision:
                window_decisions[window_decision.casefold()] += 1
            options = _sequence(
                item.get("decision_options", DECISION_OPTIONS),
                field=f"{item_key}.decision_options",
            )
            for option in options:
                option_text = _text(option)
                if option_text:
                    option_counts[option_text.casefold()] += 1
            source_interval = _mapping(
                item.get("source_interval", {}), field=f"{item_key}.source_interval"
            )
            source_interval_statuses[str(source_interval.get("status") or "MISSING").upper()] += 1
            proposals = _sequence(item.get("proposals", []), field=f"{item_key}.proposals")
            proposal_count += len(proposals)
            for proposal_index, raw_proposal in enumerate(proposals):
                proposal = _mapping(
                    raw_proposal,
                    field=f"{item_key}.proposals[{proposal_index}]",
                )
                decision = _text(proposal.get("decision"))
                if decision:
                    proposal_decisions[decision.casefold()] += 1
                top_k = _sequence(
                    proposal.get("top_k", []),
                    field=f"{item_key}.proposals[{proposal_index}].top_k",
                )
                topk_cardinality[str(len(top_k))] += 1
                margin = _finite(proposal.get("margin"))
                if margin is not None:
                    margins.append(margin)

            copied_item = _copy_json(raw_item, field=f"{item_key}.item")
            if not isinstance(copied_item, dict):  # pragma: no cover - invariant
                raise ProductionWemmReviewPackAggregateError(f"{item_key} must be an object")
            copied_item.update(
                {
                    "recording_id": recording_id,
                    "recording_index": recording_index,
                    "item_key": item_key,
                    "source_ref": _merge_lineage(
                        copied_item.get("source_ref"),
                        {
                            "recording_id": recording_id,
                            "review_pack_path": pack_path,
                            "source_path": source.get("path"),
                            "archive_member": source.get("archive_member"),
                        },
                        field=f"{pack_path}.items[{item_index}].source_ref",
                    ),
                    "provenance": _merge_lineage(
                        copied_item.get("provenance"),
                        {
                            "review_pack_path": pack_path,
                            "source_preflight_status": source.get("source_preflight_status"),
                            "qa_status": source.get("qa_status"),
                            "window_context_only": True,
                        },
                        field=f"{pack_path}.items[{item_index}].provenance",
                    ),
                }
            )
            flattened_items.append(copied_item)

    duplicate_recordings = sorted(
        recording_id for recording_id, count in recording_ids.items() if count > 1
    )
    duplicate_window_keys = sorted(key for key, count in window_keys.items() if count > 1)
    checkpoint_sources = [
        source for source in input_sources if source.get("kind") == "batch_checkpoint"
    ]
    unresolved_checkpoint_items = sum(
        int(source.get("unresolved_item_count", 0) or 0) for source in checkpoint_sources
    )
    summary = {
        "recording_count": len(recordings),
        "window_count": len(flattened_items),
        "proposal_count": proposal_count,
        "camera_window_input_count": camera_window_count,
        "expected_camera_window_input_count": len(flattened_items) * expected_camera_count,
        "camera_window_coverage": (
            camera_window_count / (len(flattened_items) * expected_camera_count)
            if flattened_items
            else None
        ),
        "review_status_counts": dict(sorted(review_statuses.items())),
        "window_status_counts": dict(sorted(window_statuses.items())),
        "window_decision_counts": dict(sorted(window_decisions.items())),
        "proposal_decision_counts": dict(sorted(proposal_decisions.items())),
        "decision_option_counts": dict(sorted(option_counts.items())),
        "source_interval_status_counts": dict(sorted(source_interval_statuses.items())),
        "top_k_cardinality": dict(sorted(topk_cardinality.items(), key=lambda pair: int(pair[0]))),
        "margin_summary": _summary(margins),
        "temporal_sidecar_recording_count": len(temporal_sidecar_recordings),
        "temporal_segment_count": len(flattened_temporal_segments),
        "temporal_boundary_status_counts": dict(sorted(temporal_boundary_statuses.items())),
        "temporal_refinement_recording_count": len(temporal_refinement_recordings),
        "refined_temporal_segment_count": len(flattened_temporal_refinement_segments),
        "refined_temporal_interval_proposal_count": len(
            flattened_refined_temporal_interval_proposals
        ),
        "refined_temporal_boundary_status_counts": dict(
            sorted(refined_temporal_boundary_statuses.items())
        ),
        "observed_camera_ids": sorted(observed_camera_ids),
        "input_source_count": len(input_sources),
        "checkpoint_count": len(checkpoint_sources),
        "unresolved_checkpoint_item_count": unresolved_checkpoint_items,
        "all_checkpoint_items_complete": unresolved_checkpoint_items == 0,
    }
    return {
        "format": AGGREGATE_FORMAT,
        "authority": AUTHORITY,
        "status": "PENDING_REVIEW" if recordings else "NO_VALID_REVIEW_PACKS",
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "review_contract": {
            "decision_options": list(DECISION_OPTIONS),
            "required_fields": [
                "start_seconds",
                "end_seconds",
                "verb",
                "noun",
                "attributes",
                "location",
                "hand",
                "confidence",
                "evidence",
                "camera_support",
                "top_k",
                "margin",
            ],
            "window_context_fields": ["start_seconds", "end_seconds"],
            "model_predictions_are_not_gold": True,
            "human_review_required": True,
            "fixed_windows_are_action_boundaries": False,
            "source_interval_is_context_only": True,
            "status_fields": [
                "window_status",
                "window_decision",
                "proposal_status",
                "decision",
                "split_hint",
            ],
            "provenance_fields": [
                "camera_ids",
                "camera_support",
                "top_k",
                "margin",
            ],
            "temporal_sidecar_fields": [
                "temporal_resolution",
                "temporal_segments",
                "temporal_refinement_plan",
                "temporal_refinement_fine_plan",
                "temporal_refinement_score_resolution",
                "temporal_refinement",
                "refined_segments",
                "refined_temporal_segments",
                "temporal_refinement_segments",
                "refined_temporal_interval_proposals",
            ],
            "temporal_segments_review_only": True,
            "temporal_segments_are_action_boundary_proposals": True,
            "temporal_refinement_review_only": True,
            "refined_segments_review_only": True,
            "refined_segments_are_action_boundary_proposals": True,
        },
        "source": {
            "kind": "multi_recording",
            "recording_count": len(recordings),
            "camera_count": expected_camera_count,
            "recording_ids": [entry["recording_id"] for entry in recordings],
        },
        "recordings": recordings,
        "items": flattened_items,
        "temporal_resolution": {
            "recordings": temporal_sidecar_recordings,
            "segments": _copy_json(
                flattened_temporal_segments, field="aggregate.temporal_resolution.segments"
            ),
            "segment_count": len(flattened_temporal_segments),
            "boundary_status_counts": dict(sorted(temporal_boundary_statuses.items())),
            "review_only": True,
        },
        # Compact alias for consumers that only need proposed intervals and
        # not per-recording temporal parameters/diagnostics.
        "temporal_segments": _copy_json(
            flattened_temporal_segments, field="aggregate.temporal_segments"
        ),
        "temporal_refinement": {
            "recordings": temporal_refinement_recordings,
            "segments": _copy_json(
                flattened_temporal_refinement_segments,
                field="aggregate.temporal_refinement.segments",
            ),
            "refined_temporal_interval_proposals": _copy_json(
                flattened_refined_temporal_interval_proposals,
                field="aggregate.temporal_refinement.refined_temporal_interval_proposals",
            ),
            "segment_count": len(flattened_temporal_refinement_segments),
            "measured_segment_count": len(flattened_refined_temporal_interval_proposals),
            "boundary_status_counts": dict(sorted(refined_temporal_boundary_statuses.items())),
            "review_only": True,
        },
        "temporal_refinement_segments": _copy_json(
            flattened_temporal_refinement_segments,
            field="aggregate.temporal_refinement_segments",
        ),
        "refined_temporal_segments": _copy_json(
            flattened_temporal_refinement_segments,
            field="aggregate.refined_temporal_segments",
        ),
        "refined_temporal_interval_proposals": _copy_json(
            flattened_refined_temporal_interval_proposals,
            field="aggregate.refined_temporal_interval_proposals",
        ),
        "summary": summary,
        "provenance": {
            "input_pack_count": len(valid),
            "input_paths": [path for path, _ in valid],
            "input_sources": input_sources,
            "partial_input": unresolved_checkpoint_items > 0,
            "model_names": dict(sorted(model_names.items())),
            "model_routes": dict(sorted(model_routes.items())),
            "catalog_formats": dict(sorted(catalog_formats.items())),
        },
        "duplicates": {
            "recording_ids": duplicate_recordings,
            "item_keys": duplicate_window_keys,
            "duplicate_recording_count": len(duplicate_recordings),
            "duplicate_item_count": len(duplicate_window_keys),
        },
        "invalid_inputs": invalid_inputs,
        "rejected_inputs": rejected_inputs,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "qwen_read": False,
            "mage_read": False,
            "gold_read": False,
            "gold_written": False,
            "predictions_copied_to_gold": False,
            "window_boundaries_as_actions": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "hash_or_sha_used": False,
        },
        "limitations": [
            "This is a review queue, not a gold annotation or quality measurement.",
            "Processing-window intervals remain context only; action boundaries are not inferred.",
            "Temporal segments are model-derived review proposals and are not "
            "measured/gold boundaries.",
            "Adaptive refined segments are additive review proposals; pending rows "
            "remain unresolved.",
            "WeMM Top-K and margins are preserved as retrieval evidence, not probabilities.",
            "Qwen and Mage artifacts are intentionally not read or merged.",
            "Duplicate recording/item keys are reported and never silently collapsed.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact review-queue summary without embedding model output."""

    summary = _mapping(report.get("summary", {}), field="report.summary")
    provenance = _mapping(report.get("provenance", {}), field="report.provenance")
    status_json = json.dumps(summary.get("review_status_counts", {}), sort_keys=True)
    options_json = json.dumps(summary.get("decision_option_counts", {}), sort_keys=True)
    topk_json = json.dumps(summary.get("top_k_cardinality", {}), sort_keys=True)
    margin_json = json.dumps(summary.get("margin_summary", {}), sort_keys=True)
    temporal_status_json = json.dumps(
        summary.get("temporal_boundary_status_counts", {}), sort_keys=True
    )
    refined_status_json = json.dumps(
        summary.get("refined_temporal_boundary_status_counts", {}), sort_keys=True
    )
    lines = [
        "# Production WeMM review-pack aggregate",
        "",
        f"- Status: **{report.get('status', 'UNKNOWN')}**",
        f"- Official quality: **{report.get('official_quality_status', 'NOT_MEASURED')}**",
        f"- Official gold: **{report.get('official_gold_status', 'NOT_ESTABLISHED')}**",
        f"- Input packs: {provenance.get('input_pack_count', 0)}",
        f"- Input sources: {summary.get('input_source_count', 0)}; checkpoints: "
        f"{summary.get('checkpoint_count', 0)}",
        f"- Unresolved checkpoint items: {summary.get('unresolved_checkpoint_item_count', 0)}",
        "",
        "## Coverage",
        "",
        f"- Recordings: {summary.get('recording_count', 0)}",
        f"- Review items/windows (not action spans): {summary.get('window_count', 0)}",
        f"- Proposals: {summary.get('proposal_count', 0)}",
        f"- Camera-window inputs: {summary.get('camera_window_input_count', 0)} / "
        f"{summary.get('expected_camera_window_input_count', 0)} expected",
        "",
        "## Review decisions",
        "",
        f"- Statuses: `{status_json}`",
        f"- Allowed options: `{options_json}`",
        "",
        "## Retrieval evidence",
        "",
        f"- Top-K cardinality: `{topk_json}`",
        f"- Margin summary: `{margin_json}`",
        "",
        "## Temporal interval sidecar",
        "",
        "- Recordings with temporal sidecars: "
        f"{summary.get('temporal_sidecar_recording_count', 0)}",
        "- Model-derived interval proposals (review only): "
        f"{summary.get('temporal_segment_count', 0)}",
        f"- Boundary statuses: `{temporal_status_json}`",
        f"- Adaptive refined recordings: {summary.get('temporal_refinement_recording_count', 0)}",
        "- Adaptive refined rows (review only): "
        f"{summary.get('refined_temporal_segment_count', 0)}",
        "- Measured refined interval proposals: "
        f"{summary.get('refined_temporal_interval_proposal_count', 0)}",
        f"- Refined boundary statuses: `{refined_status_json}`",
        "",
        "## Limitations",
        "",
    ]
    for limitation in summary.get("limitations", []):
        lines.append(f"- {limitation}")
    # The limitations live at report level; retain this fallback for callers
    # that pass a compact report object to the renderer.
    if not lines[-1].startswith("- "):
        for limitation in report.get("limitations", []):
            lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


__all__ = [
    "AGGREGATE_FORMAT",
    "AUTHORITY",
    "BATCH_RUN_FORMAT",
    "DECISION_OPTIONS",
    "REVIEW_PACK_FORMAT",
    "ProductionWemmReviewPackAggregateError",
    "aggregate_production_wemm_review_packs",
    "render_markdown",
]
