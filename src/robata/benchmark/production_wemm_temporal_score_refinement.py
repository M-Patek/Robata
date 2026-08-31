"""Fine WeMM score probes for model-derived temporal boundaries.

The open-vocabulary WeMM runner emits a ranking for a *bounded* video
context.  A context edge is not an action edge, and a single short context
around a coarse transition cannot establish an onset or an offset.  This
module adds a small, deterministic score-refinement seam:

* :func:`plan_wemm_score_refinement_grid` expands each coarse onset/offset
  request into a source-relative before/after probe grid at one or more
  resolutions;
* a caller runs those probes through WeMM and supplies the resulting scores;
* :func:`resolve_wemm_score_refinement` finds a threshold crossing between
  adjacent probe centres and emits a measured result row keyed by the
  *parent* request ID, ready for
  :func:`production_wemm_temporal_refinement.apply_refined_boundaries`.

The score resolver never copies a probe edge into an action boundary.  A
boundary is measured only when the supplied model scores bracket the declared
threshold *and* the crossing straddles the coarse anchor in the role's
expected direction (before->after for onset, before->after through the offset
anchor for offset).  Same-side crossings are retained as review diagnostics
but remain ``UNCERTAIN``.  A threshold crossing that touches a frame-padded
fine probe is also always ``UNCERTAIN`` because duplicated edge frames cannot
localize an action boundary.  Missing, edge-clipped, non-monotone, or
unsupported evidence remains ``UNCERTAIN``.  All output is review-only and
contains no gold, ontology mutation, hash, or digest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final

from .production_wemm_temporal import DEFAULT_RELATIVE_MARGIN_SCALE
from .production_wemm_temporal_refinement import (
    REQUEST_TIMESTAMP_BASIS,
    plan_wemm_temporal_refinement,
)

FORMAT: Final = "robata-production-wemm-temporal-score-refinement-v1"
RESULT_FORMAT: Final = "robata-production-wemm-temporal-score-result-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "FINE_SCORE_PROBE_REQUESTS_ONLY"
RESULT_STATUS: Final = "FINE_SCORE_BOUNDARIES_REVIEW_ONLY"
# ``absolute`` and ``top1`` remain the historical policies.  The relative
# policies below are deliberately additive: they are only consumed by the
# fine-score resolver and never change the dense coarse resolver's defaults.
SCORE_POLICY_RELATIVE_MARGIN: Final = "relative_margin"
# Descriptive aliases accepted at the fine resolver boundary.  Keep the
# canonical spelling aligned with the coarse temporal resolver so a report can
# be passed between the two stages without a policy rename.
SCORE_POLICY_CANDIDATE_RELATIVE: Final = "candidate_relative"
SCORE_POLICY_RELATIVE: Final = "relative"
SCORE_POLICY_CONTRAST: Final = "contrast"
SCORE_POLICIES: Final = (
    "top1",
    "absolute",
    SCORE_POLICY_RELATIVE_MARGIN,
    SCORE_POLICY_CANDIDATE_RELATIVE,
    SCORE_POLICY_RELATIVE,
    SCORE_POLICY_CONTRAST,
)
RELATIVE_SCORE_POLICIES: Final = frozenset(
    {
        SCORE_POLICY_RELATIVE_MARGIN,
        SCORE_POLICY_CANDIDATE_RELATIVE,
        SCORE_POLICY_RELATIVE,
        SCORE_POLICY_CONTRAST,
    }
)
DEFAULT_START_MARGIN_THRESHOLD: Final = 0.0
DEFAULT_STOP_MARGIN_THRESHOLD: Final = 0.0
DEFAULT_MIN_MARGIN_PERSISTENCE: Final = 1
ROLES: Final = ("onset", "offset")
SIDES: Final = ("before", "after")
DEFAULT_PROBE_SPAN_SECONDS: Final = 0.50
DEFAULT_POINTS_PER_SIDE: Final = 2
DEFAULT_LEVELS: Final = 2
DEFAULT_MAX_REQUESTS: Final = 256
DEFAULT_MIN_PROBE_SPAN_SECONDS: Final = 0.05
DEFAULT_MIN_BOUNDARY_RESOLUTION_SECONDS: Final = 0.05


class ProductionWemmTemporalScoreRefinementError(ValueError):
    """Raised when a score-refinement grid or result is malformed."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be non-empty")
    return result


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be finite")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be finite")
    return result


def _positive(value: object, *, field: str) -> float:
    result = _finite(value, field=field)
    if result <= 0.0:
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be positive")
    return result


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionWemmTemporalScoreRefinementError(f"{field} must be a positive integer")
    return value


def _copy_json(value: object, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmTemporalScoreRefinementError(
                f"{field} contains a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmTemporalScoreRefinementError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[]") for child in value]
    raise ProductionWemmTemporalScoreRefinementError(f"{field} must be JSON-compatible")


def _source_bounds(report: Mapping[str, Any]) -> tuple[float, float]:
    for key in ("context_interval", "window"):
        raw = report.get(key)
        if isinstance(raw, Mapping) and raw.get("start_seconds") is not None:
            start = _finite(raw.get("start_seconds"), field=f"{key}.start_seconds")
            end = _finite(raw.get("end_seconds"), field=f"{key}.end_seconds")
            if start < 0.0 or end <= start:
                raise ProductionWemmTemporalScoreRefinementError(
                    f"{key} must satisfy 0 <= start < end"
                )
            return start, end
    starts: list[float] = []
    ends: list[float] = []
    raw_segments = report.get("segments", ())
    if isinstance(raw_segments, Sequence) and not isinstance(raw_segments, (str, bytes, bytearray)):
        for index, raw in enumerate(raw_segments):
            if not isinstance(raw, Mapping):
                continue
            start_raw, end_raw = raw.get("start_seconds"), raw.get("end_seconds")
            if start_raw is None or end_raw is None:
                continue
            start = _finite(start_raw, field=f"segments[{index}].start_seconds")
            end = _finite(end_raw, field=f"segments[{index}].end_seconds")
            if start >= 0.0 and end > start:
                starts.append(start)
                ends.append(end)
    if starts:
        return min(starts), max(ends)
    raise ProductionWemmTemporalScoreRefinementError(
        "coarse report needs context_interval, window, or bounded segments"
    )


def _parent_requests(
    coarse_report: Mapping[str, Any], parent_plan: Mapping[str, Any] | None
) -> tuple[dict[str, Any], ...]:
    plan = parent_plan
    if plan is None:
        plan = plan_wemm_temporal_refinement(coarse_report)
    if not isinstance(plan, Mapping):
        raise ProductionWemmTemporalScoreRefinementError("parent_plan must be an object")
    raw = _sequence(plan.get("requests", ()), field="parent_plan.requests")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        row = dict(_mapping(value, field=f"parent_plan.requests[{index}]"))
        request_id = _text(row.get("request_id"), field=f"parent_plan.requests[{index}].request_id")
        if request_id in seen:
            raise ProductionWemmTemporalScoreRefinementError(
                f"duplicate parent request: {request_id}"
            )
        seen.add(request_id)
        action = _text(row.get("action_key"), field=f"parent_plan.requests[{index}].action_key")
        role = _text(row.get("role"), field=f"parent_plan.requests[{index}].role").casefold()
        if role not in ROLES:
            raise ProductionWemmTemporalScoreRefinementError(
                f"parent_plan.requests[{index}].role must be one of {', '.join(ROLES)}"
            )
        start = _finite(
            row.get("start_seconds"), field=f"parent_plan.requests[{index}].start_seconds"
        )
        end = _finite(row.get("end_seconds"), field=f"parent_plan.requests[{index}].end_seconds")
        if start < 0.0 or end <= start:
            raise ProductionWemmTemporalScoreRefinementError(
                f"parent request {request_id} must satisfy 0 <= start < end"
            )
        row["request_id"] = request_id
        row["action_key"] = action
        row["role"] = role
        row["start_seconds"] = start
        row["end_seconds"] = end
        result.append(row)
    return tuple(result)


def _clip_interval(
    start: float, end: float, *, source_start: float, source_end: float
) -> tuple[float, float, bool] | None:
    clipped_start = max(source_start, min(start, source_end))
    clipped_end = max(source_start, min(end, source_end))
    if clipped_end <= clipped_start + 1e-9:
        return None
    return (
        clipped_start,
        clipped_end,
        (abs(clipped_start - start) > 1e-9 or abs(clipped_end - end) > 1e-9),
    )


def _round_robin_request_truncation(
    requests: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Select a bounded request set fairly across parent transitions.

    The planner's global time sort is useful for execution order, but taking
    its first ``limit`` rows can consume all onset/offset probes for one
    parent before later segments receive any evidence.  Group rows by parent,
    take one row per parent per round, and restore the original deterministic
    ordering for the emitted list.  A single parent may still receive more
    rows when the limit is smaller than the number of parents; no allocation
    can include every parent in that case.
    """

    if limit <= 0:
        return []
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in requests:
        parent_id = str(row.get("parent_request_id", ""))
        groups.setdefault(parent_id, []).append(row)
    parent_ids = sorted(groups)
    selected: list[Mapping[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for parent_id in parent_ids:
            rows = groups[parent_id]
            if offset >= len(rows):
                continue
            selected.append(rows[offset])
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
        offset += 1

    def _sort_key(row: Mapping[str, Any]) -> tuple[float, float, str, int, str, int]:
        return (
            float(row["start_seconds"]),
            float(row["end_seconds"]),
            str(row["parent_request_id"]),
            int(row["level"]),
            str(row["probe_side"]),
            int(row["offset_index"]),
        )

    return [dict(row) for row in sorted(selected, key=_sort_key)]


def plan_wemm_score_refinement_grid(
    coarse_report: Mapping[str, Any],
    *,
    parent_plan: Mapping[str, Any] | None = None,
    probe_span_seconds: float = DEFAULT_PROBE_SPAN_SECONDS,
    points_per_side: int = DEFAULT_POINTS_PER_SIDE,
    levels: int = DEFAULT_LEVELS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    min_probe_span_seconds: float = DEFAULT_MIN_PROBE_SPAN_SECONDS,
) -> dict[str, Any]:
    """Plan a multi-resolution before/after WeMM score grid.

    Each parent onset/offset request receives contiguous probes ending at and
    beginning at the coarse anchor.  Level zero covers ``points_per_side``
    probe widths on either side; subsequent levels add nested, shorter probes.
    The returned probe edges are input context edges only and are explicitly
    forbidden from being copied as action boundaries.
    """

    report = _mapping(coarse_report, field="coarse_report")
    source_start, source_end = _source_bounds(report)
    span = _positive(probe_span_seconds, field="probe_span_seconds")
    minimum = _positive(min_probe_span_seconds, field="min_probe_span_seconds")
    if span < minimum:
        raise ProductionWemmTemporalScoreRefinementError(
            "probe_span_seconds must be >= min_probe_span_seconds"
        )
    points = _positive_int(points_per_side, field="points_per_side")
    depth = _positive_int(levels, field="levels")
    limit = _positive_int(max_requests, field="max_requests")
    parents = _parent_requests(report, parent_plan)
    requests: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, float, float]] = set()
    edge_count = 0
    clipped_count = 0
    for parent in parents:
        parent_id = str(parent["request_id"])
        action = str(parent["action_key"])
        role = str(parent["role"])
        anchor_raw = parent.get("coarse_anchor_seconds")
        anchor = (
            _finite(anchor_raw, field=f"parent[{parent_id}].coarse_anchor_seconds")
            if anchor_raw is not None
            else (float(parent["start_seconds"]) + float(parent["end_seconds"])) / 2.0
        )
        anchor = max(source_start, min(anchor, source_end))
        for level in range(depth):
            width = span / (2.0**level)
            if width < minimum - 1e-9:
                continue
            # At finer levels one nested interval per side is sufficient; the
            # outer level supplies the wider bracket and the dense model pass
            # remains bounded for long recordings.
            side_points = points if level == 0 else 1
            for side in SIDES:
                for offset_index in range(side_points):
                    if side == "before":
                        end = anchor - offset_index * width
                        start = end - width
                    else:
                        start = anchor + offset_index * width
                        end = start + width
                    clipped = _clip_interval(
                        start,
                        end,
                        source_start=source_start,
                        source_end=source_end,
                    )
                    if clipped is None:
                        edge_count += 1
                        continue
                    clipped_start, clipped_end, was_clipped = clipped
                    clipped_count += int(was_clipped)
                    if was_clipped:
                        edge_count += 1
                    if clipped_end - clipped_start < minimum - 1e-9:
                        continue
                    key = (
                        parent_id,
                        level,
                        side,
                        round(clipped_start, 6),
                        round(clipped_end, 6),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    probe_id = (f"{parent_id}::score::{level}::{side}::{offset_index}").replace(
                        ".", "p"
                    )
                    requests.append(
                        {
                            "request_id": probe_id,
                            "parent_request_id": parent_id,
                            "action_key": action,
                            "role": role,
                            "probe_side": side,
                            "level": level,
                            "offset_index": offset_index,
                            "anchor_seconds": round(anchor, 6),
                            "start_seconds": round(clipped_start, 6),
                            "end_seconds": round(clipped_end, 6),
                            "edge_clipped": bool(was_clipped),
                            "timestamp_basis": "source_relative_seconds",
                            "request_timestamp_basis": REQUEST_TIMESTAMP_BASIS,
                            "interval_status": "FINE_SCORE_CONTEXT_REQUEST",
                            "boundary_status": "PENDING_MODEL_RECOMPUTE",
                            "requires_model_recompute": True,
                            "request_edges_are_not_boundaries": True,
                            "input_semantics": "FINE_BEFORE_AFTER_SCORE_PROBE",
                            "source_segment_ids": list(parent.get("source_segment_ids", [])),
                            "source_window_ids": list(parent.get("source_window_ids", [])),
                            "status": "PENDING",
                        }
                    )
    requests.sort(
        key=lambda row: (
            float(row["start_seconds"]),
            float(row["end_seconds"]),
            str(row["parent_request_id"]),
            int(row["level"]),
            str(row["probe_side"]),
            int(row["offset_index"]),
        )
    )
    planned_request_count = len(requests)
    planned_by_parent: dict[str, int] = {str(parent["request_id"]): 0 for parent in parents}
    for request in requests:
        parent_id = str(request["parent_request_id"])
        planned_by_parent[parent_id] = planned_by_parent.get(parent_id, 0) + 1
    truncated = planned_request_count > limit
    if truncated:
        requests = _round_robin_request_truncation(requests, limit)
    emitted_by_parent: dict[str, int] = {parent_id: 0 for parent_id in sorted(planned_by_parent)}
    for request in requests:
        parent_id = str(request["parent_request_id"])
        emitted_by_parent[parent_id] = emitted_by_parent.get(parent_id, 0) + 1
    return {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": STATUS,
        "production_eligible": False,
        "timestamp_basis": "source_relative_seconds",
        "request_timestamp_basis": REQUEST_TIMESTAMP_BASIS,
        "source": {
            "context_interval": {
                "start_seconds": source_start,
                "end_seconds": source_end,
                "context_only": True,
                "is_action_boundary": False,
                "action_boundary": False,
            },
            "parent_plan_format": (
                parent_plan.get("format") if isinstance(parent_plan, Mapping) else None
            ),
        },
        "parents": [_copy_json(parent, field="parents[]") for parent in parents],
        "requests": requests,
        "parameters": {
            "probe_span_seconds": span,
            "points_per_side": points,
            "levels": depth,
            "max_requests": limit,
            "min_probe_span_seconds": minimum,
        },
        "diagnostics": {
            "parent_request_count": len(parents),
            "planned_probe_request_count": planned_request_count,
            "emitted_probe_request_count": len(requests),
            "probe_request_count": len(requests),
            "parent_probe_request_allocation": {
                parent_id: {
                    "planned_count": planned_by_parent[parent_id],
                    "emitted_count": emitted_by_parent.get(parent_id, 0),
                }
                for parent_id in sorted(planned_by_parent)
            },
            "edge_or_clipped_probe_count": edge_count,
            "clipped_probe_count": clipped_count,
            "truncated": truncated,
            "fine_grid_before_after": True,
            "request_edges_are_not_boundaries": True,
        },
        "controls": {
            "media_decoded": False,
            "model_invoked": False,
            "runner_recompute_required": bool(requests),
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
        },
        "limitations": [
            "Fine probes are bounded model contexts, not action boundaries.",
            "A score crossing is review-only and requires independent review.",
            "Missing or edge-clipped before/after evidence remains unresolved.",
        ],
    }


_CANDIDATE_FIELD_FALLBACKS: Final = (
    "provisional_id",
    "action_key",
    "score",
    "fused_score",
    "visual_score",
    "similarity",
    "cosine",
    "confidence",
    "rank",
    "camera_support",
    "camera_support_count",
    "camera_coverage",
    "camera_ids",
    "camera_evidence",
    "evidence",
)

_CAMERA_SUPPORT_FIELDS: Final = (
    "camera_support",
    "camera_support_count",
    "camera_coverage",
    "camera_ids",
    "camera_evidence",
    "evidence",
)


def _effective_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the normalized candidate's raw provenance for score lookup.

    ``build_preannotation_envelope`` intentionally keeps the producer row in
    ``candidate.raw`` and exposes only contract fields at the top level.  In
    particular, an open-runner candidate's ``provisional_id`` and fused
    ``camera_coverage`` live in that raw row.  Fine temporal resolution must
    read both shapes; otherwise every normalized probe looks like a missing,
    zero-support candidate and no boundary can ever be measured.
    """

    result = dict(candidate)
    raw = candidate.get("raw")
    if isinstance(raw, Mapping):
        for key in _CANDIDATE_FIELD_FALLBACKS:
            value = result.get(key)
            if value is None and key in raw:
                result[key] = raw[key]
        # Fused retrieval uses ``camera_evidence`` while the normalized
        # contract uses ``evidence``.  Preserve it as evidence when needed so
        # camera support can be derived without inventing views.
        if not result.get("evidence") and isinstance(raw.get("camera_evidence"), Sequence):
            result["evidence"] = raw["camera_evidence"]
    return result


def _candidate_action(candidate: Mapping[str, Any]) -> object:
    """Return the opaque action identity from either candidate layer."""

    effective = _effective_candidate(candidate)
    return effective.get("provisional_id", effective.get("action_key"))


_RELATIVE_ACTION_FIELDS: Final = (
    # The first names are used by current and planned callers.  The aliases
    # make the resolver tolerant of small producer-specific vocabulary
    # differences without changing the wire contract.
    "relative_action_key",
    "comparison_action_key",
    "competing_action_key",
    "competitor_action_key",
    "adjacent_action_key",
    "neighbor_action_key",
    "coarse_neighbor_action_key",
    "margin_action_key",
)


def _relative_action_hint(*rows: Mapping[str, Any] | None) -> str | None:
    """Return an explicit comparison action from the first row that has one."""

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for field in _RELATIVE_ACTION_FIELDS:
            raw = row.get(field)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        # A few raw sidecars put the hint under a small nested comparison
        # object.  It is still only a hint; the candidate rows remain the
        # authoritative score/evidence source.
        comparison = row.get("comparison")
        if isinstance(comparison, Mapping):
            for field in _RELATIVE_ACTION_FIELDS:
                raw = comparison.get(field)
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
    return None


def _candidate_score(candidate: Mapping[str, Any]) -> float | None:
    """Read one candidate's retrieval score without inventing missing values."""

    effective = _effective_candidate(candidate)
    for field in (
        "score",
        "fused_score",
        "visual_score",
        "similarity",
        "cosine",
        "confidence",
    ):
        raw = effective.get(field)
        if raw is None:
            continue
        try:
            score = _finite(raw, field=f"candidate.{field}")
        except ProductionWemmTemporalScoreRefinementError:
            return None
        # WeMM similarity artifacts are expected to be unit scores.  Keep the
        # same conservative clipping used by the absolute resolver so a
        # malformed provider row cannot create an unbounded margin.
        return max(0.0, min(1.0, score))
    return None


def _candidate_support(candidate: Mapping[str, Any]) -> int:
    """Count camera support in the several producer-side candidate shapes."""

    effective = _effective_candidate(candidate)
    # ``dict.get(key, fallback)`` does not use the fallback when a producer
    # explicitly serializes a field as ``None``.  Several normalized/legacy
    # envelopes do exactly that while retaining a numeric support count, so
    # select the first non-null representation instead.
    support_raw: object = 0
    for field in _CAMERA_SUPPORT_FIELDS:
        value = effective.get(field)
        if value is not None:
            support_raw = value
            break
    if isinstance(support_raw, Mapping):
        return len([key for key in support_raw if isinstance(key, str) and key.strip()])
    if isinstance(support_raw, Sequence) and not isinstance(support_raw, (str, bytes, bytearray)):
        support_ids = {
            item.strip() for item in support_raw if isinstance(item, str) and item.strip()
        }
        if not support_ids:
            support_ids = {
                str(item.get("camera_id")).strip()
                for item in support_raw
                if isinstance(item, Mapping)
                and isinstance(item.get("camera_id"), str)
                and item.get("camera_id", "").strip()
            }
        return len(support_ids)
    if not isinstance(support_raw, (int, float, str, bytes, bytearray)):
        return 0
    try:
        return max(0, int(support_raw))
    except (TypeError, ValueError, OverflowError):
        return 0


def _candidate_camera_ids(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Return explicit camera IDs without treating counts as camera names."""

    effective = _effective_candidate(candidate)
    raw = effective.get("camera_ids")
    if raw is None:
        raw = effective.get("camera_support")
    if raw is None:
        raw = effective.get("camera_evidence", effective.get("evidence"))
    values: set[str] = set()
    if isinstance(raw, Mapping):
        values.update(str(key).strip() for key in raw if isinstance(key, str) and key.strip())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                values.add(item.strip())
            elif isinstance(item, Mapping):
                camera_id = item.get("camera_id")
                if isinstance(camera_id, str) and camera_id.strip():
                    values.add(camera_id.strip())
    return tuple(sorted(values))


def _candidate_rank(candidate: Mapping[str, Any]) -> int:
    effective = _effective_candidate(candidate)
    raw = effective.get("rank")
    try:
        rank = int(raw) if raw is not None else 1
    except (TypeError, ValueError, OverflowError):
        rank = 1
    return max(1, rank)


def _candidate_rows_from_window(
    window: Mapping[str, Any],
    *,
    _visited: set[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Collect a deduplicated action->candidate map from one fine result row."""

    visited = _visited if _visited is not None else set()
    if id(window) in visited:
        return {}
    visited.add(id(window))

    candidates: list[Mapping[str, Any]] = []

    def _append(value: object) -> None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return
        candidates.extend(item for item in value if isinstance(item, Mapping))

    _append(window.get("top_k"))
    _append(window.get("candidates"))
    fused = window.get("fused")
    if isinstance(fused, Mapping):
        _append(fused.get("candidates"))
    proposals = window.get("proposals")
    if isinstance(proposals, Sequence) and not isinstance(proposals, (str, bytes, bytearray)):
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                continue
            _append(proposal.get("top_k"))
            _append(proposal.get("candidates"))
            # A proposal itself can be the candidate in a normalized envelope.
            if _candidate_action(proposal) is not None and _candidate_score(proposal) is not None:
                candidates.append(proposal)

    # Raw open-runner envelopes occasionally wrap the candidate list under a
    # model/raw sidecar.  Merge those rows recursively while guarding against
    # a self-referential mapping supplied by a caller.
    nested_candidates: dict[str, dict[str, Any]] = {}
    for nested_key in ("raw", "model", "raw_model_output"):
        nested = window.get(nested_key)
        if isinstance(nested, Mapping):
            for action, candidate in _candidate_rows_from_window(nested, _visited=visited).items():
                nested_candidates[action] = candidate

    result: dict[str, dict[str, Any]] = {}
    result.update(nested_candidates)
    for raw_candidate in candidates:
        candidate = _effective_candidate(raw_candidate)
        action_value = _candidate_action(candidate)
        if not isinstance(action_value, str) or not action_value.strip():
            continue
        action = action_value.strip()
        score = _candidate_score(candidate)
        if score is None:
            continue
        candidate["score"] = score
        candidate["rank"] = _candidate_rank(candidate)
        # Keep the producer's original ``camera_support`` value (often a list
        # of IDs) intact for provenance; use a separate normalized count for
        # sorting and later support checks.
        candidate["camera_support_count"] = _candidate_support(candidate)
        previous = result.get(action)
        if previous is None:
            result[action] = candidate
            continue
        # Raw and normalized envelopes can expose the same candidate twice.
        # Keep the strongest supported observation, then the better rank, but
        # never combine scores from different rows into a synthetic candidate.
        previous_score = float(previous.get("score", 0.0))
        # ``camera_support`` is often the producer's list of camera IDs, not
        # an integer.  Compare the normalized count captured above instead of
        # attempting ``int(list)`` when raw and normalized envelopes expose
        # the same action twice.
        previous_support_raw = previous.get("camera_support_count")
        previous_support = (
            int(previous_support_raw)
            if isinstance(previous_support_raw, int) and not isinstance(previous_support_raw, bool)
            else _candidate_support(previous)
        )
        candidate_key = (
            int(candidate["camera_support_count"]),
            float(candidate["score"]),
        )
        previous_key = (previous_support, previous_score)
        if candidate_key > previous_key:
            result[action] = candidate
    return result


def _canonical_score_policy(value: str) -> str:
    """Normalize relative-policy aliases while preserving requested spelling."""

    if value in RELATIVE_SCORE_POLICIES:
        return SCORE_POLICY_RELATIVE_MARGIN
    return value


def _candidate_from_window(window: Mapping[str, Any], action: str) -> dict[str, Any] | None:
    """Find one action score in a runner window or score row."""

    candidates: list[Mapping[str, Any]] = []
    for key in ("top_k", "candidates"):
        raw = window.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            candidates.extend(item for item in raw if isinstance(item, Mapping))
    # The raw open-runner camera sidecar stores the fused ranking under
    # ``fused.candidates`` rather than the normalized ``proposals.top_k``
    # shape.  Accept it so callers can resolve directly from raw provenance.
    fused = window.get("fused")
    if isinstance(fused, Mapping):
        raw_fused = fused.get("candidates")
        if isinstance(raw_fused, Sequence) and not isinstance(raw_fused, (str, bytes, bytearray)):
            candidates.extend(item for item in raw_fused if isinstance(item, Mapping))
    proposals = window.get("proposals")
    if isinstance(proposals, Sequence) and not isinstance(proposals, (str, bytes, bytearray)):
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                continue
            raw_top = proposal.get("top_k")
            if isinstance(raw_top, Sequence) and not isinstance(raw_top, (str, bytes, bytearray)):
                candidates.extend(item for item in raw_top if isinstance(item, Mapping))
            candidate = _candidate_action(proposal)
            if candidate == action:
                return _effective_candidate(proposal)
    for candidate in candidates:
        candidate_id = _candidate_action(candidate)
        if candidate_id == action:
            return _effective_candidate(candidate)
    return None


def _score_from_window(
    window: Mapping[str, Any],
    *,
    action: str,
    score_policy: str,
    min_camera_support: int,
) -> tuple[float, dict[str, Any]] | None:
    candidate = _candidate_from_window(window, action)
    if candidate is None:
        return None
    raw_score = candidate.get(
        "score",
        candidate.get(
            "fused_score",
            candidate.get("visual_score", candidate.get("similarity", candidate.get("confidence"))),
        ),
    )
    if raw_score is None:
        return None
    score = _finite(raw_score, field="fine_score.score")
    if not 0.0 <= score <= 1.0:
        score = max(0.0, min(1.0, score))
    support = _candidate_support(candidate)
    if support < min_camera_support:
        score = 0.0
    rank_raw = candidate.get("rank")
    try:
        rank = int(rank_raw) if rank_raw is not None else 1
    except (TypeError, ValueError, OverflowError):
        rank = 1
    if score_policy == "top1" and rank != 1:
        score = 0.0
    return score, {
        "score": score,
        "raw_score": float(raw_score),
        "rank": rank,
        "camera_support": support,
        "candidate": _copy_json(candidate, field="fine_score.candidate"),
    }


def _padding_indices(value: object) -> tuple[int, ...]:
    """Normalize non-negative frame-padding indices from one metadata row."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(
        sorted(
            {
                item
                for item in value
                if isinstance(item, int) and not isinstance(item, bool) and item >= 0
            }
        )
    )


def _input_observations(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Find decoder observation rows across raw and lightly wrapped shapes."""

    found: list[Mapping[str, Any]] = []
    seen_ids: set[int] = set()

    def _collect(container: object) -> None:
        if not isinstance(container, Mapping):
            return
        raw = container.get("input_observations")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            for item in raw:
                if isinstance(item, Mapping) and id(item) not in seen_ids:
                    seen_ids.add(id(item))
                    found.append(item)

    _collect(value)
    for key in ("model", "raw", "raw_model_output"):
        nested = value.get(key)
        _collect(nested)
    return tuple(found)


def _padding_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Extract compact frame-padding provenance from one fine result row.

    Fine requests may be represented by normalized envelope rows, raw runner
    rows, or a small keyed score row.  Keep the provenance shape independent
    of the score/candidate parser so a padded probe cannot be mistaken for a
    normal visual observation merely because its candidate is present.
    """

    observations = _input_observations(value)
    indices: set[int] = set()
    padded_camera_ids: set[str] = set()
    padded_observation_count = 0
    provenance_available = bool(observations)

    for observation in observations:
        observation_indices = set(_padding_indices(observation.get("frame_padding_indices")))
        if not observation_indices:
            observation_indices.update(_padding_indices(observation.get("padding_indices")))
        explicit = any(
            key in observation
            for key in (
                "frame_padding_used",
                "frame_padding_indices",
                "padding_used",
                "padding_indices",
            )
        )
        provenance_available = provenance_available or explicit
        used = (
            observation.get("frame_padding_used") is True or observation.get("padding_used") is True
        )
        used = used or bool(observation_indices)
        if not used:
            continue
        padded_observation_count += 1
        indices.update(observation_indices)
        camera_id = observation.get("camera_id")
        if isinstance(camera_id, str) and camera_id.strip():
            padded_camera_ids.add(camera_id.strip())

    direct_indices = _padding_indices(value.get("frame_padding_indices"))
    if not direct_indices:
        direct_indices = _padding_indices(value.get("padding_indices"))
    direct_used = value.get("frame_padding_used") is True or value.get("padding_used") is True
    direct_used = direct_used or bool(direct_indices)
    direct_explicit = any(
        key in value
        for key in (
            "frame_padding_used",
            "frame_padding_indices",
            "padding_used",
            "padding_indices",
        )
    )
    provenance_available = provenance_available or direct_explicit
    if direct_used and not observations:
        padded_observation_count = max(padded_observation_count, 1)
    indices.update(direct_indices)

    normalized_indices = sorted(indices)
    normalized_camera_ids = sorted(padded_camera_ids)
    return {
        "padding_provenance_available": provenance_available,
        "padding_used": bool(padded_observation_count or direct_used),
        "padding_indices": normalized_indices,
        "padding_camera_ids": normalized_camera_ids,
        "padding_observation_count": padded_observation_count,
        # Mirror decoder field names for callers that pass rows through
        # without inspecting this internal normalized alias.
        "frame_padding_used": bool(padded_observation_count or direct_used),
        "frame_padding_indices": normalized_indices,
        "frame_padding_camera_ids": normalized_camera_ids,
    }


def _merge_padding_provenance(
    *rows: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge normalized and raw-sidecar padding metadata without double count."""

    available = False
    used = False
    indices: set[int] = set()
    camera_ids: set[str] = set()
    observation_count = 0
    for row in rows:
        provenance = _padding_provenance(row)
        available = available or provenance["padding_provenance_available"] is True
        used = used or provenance["padding_used"] is True
        indices.update(provenance["padding_indices"])
        camera_ids.update(provenance["padding_camera_ids"])
        observation_count = max(observation_count, int(provenance["padding_observation_count"]))
    normalized_indices = sorted(indices)
    normalized_camera_ids = sorted(camera_ids)
    return {
        "padding_provenance_available": available,
        "padding_used": used,
        "padding_indices": normalized_indices,
        "padding_camera_ids": normalized_camera_ids,
        "padding_observation_count": observation_count,
        "frame_padding_used": used,
        "frame_padding_indices": normalized_indices,
        "frame_padding_camera_ids": normalized_camera_ids,
    }


def _fine_request_map(fine_plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = _sequence(fine_plan.get("requests", ()), field="fine_plan.requests")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(raw):
        row = _mapping(value, field=f"fine_plan.requests[{index}]")
        request_id = _text(row.get("request_id"), field=f"fine_plan.requests[{index}].request_id")
        if request_id in result:
            raise ProductionWemmTemporalScoreRefinementError(
                f"duplicate fine request: {request_id}"
            )
        result[request_id] = row
    return result


def _window_id_to_request_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("temporal-refinement::"):
        return text.split("::", 1)[1]
    return text or None


def _request_id_from_window(window: Mapping[str, Any]) -> str | None:
    """Resolve the fine-request identity used by normalized and raw rows."""

    request_id = _window_id_to_request_id(
        window.get(
            "request_id",
            window.get("refinement_request_id", window.get("window_id")),
        )
    )
    return request_id


def _rows_from_result_container(
    value: object,
    *,
    field: str,
    allow_keyed: bool = False,
) -> list[Mapping[str, Any]]:
    """Extract validated row mappings from one runner/result container.

    A persisted pre-annotation envelope has two intentionally different row
    views: normalized ``windows`` and the raw runner sidecar's
    ``raw_model_output.windows``.  Keep extraction of either view in one place
    so callers can merge them without accidentally preferring the normalized
    projection when it has stripped opaque candidate IDs.
    """

    if isinstance(value, Mapping):
        for key in ("windows", "results"):
            payload = value.get(key)
            if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
                rows = _sequence(payload, field=f"{field}.{key}")
                return [
                    _mapping(row, field=f"{field}.{key}[{index}]") for index, row in enumerate(rows)
                ]
        if not allow_keyed:
            return []
        keyed_rows: list[Mapping[str, Any]] = []
        for request_id, raw in value.items():
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            row.setdefault("request_id", request_id)
            keyed_rows.append(row)
        return keyed_rows
    rows = _sequence(value, field=field)
    return [_mapping(row, field=f"{field}[{index}]") for index, row in enumerate(rows)]


def _fine_result_row_views(
    fine_results: Mapping[str, Any] | Sequence[Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Return normalized and raw fine-window rows separately.

    The open runner stores candidate IDs in the raw sidecar because the
    review-only envelope intentionally exposes only contract fields at the
    normalized layer.  Returning both views lets score extraction use the raw
    candidate identity while retaining normalized timing/provenance fields.
    """

    if not isinstance(fine_results, Mapping):
        return _rows_from_result_container(fine_results, field="fine_results"), []

    normalized_rows = _rows_from_result_container(
        fine_results,
        field="fine_results",
        allow_keyed=False,
    )
    raw_model_output = fine_results.get("raw_model_output")
    raw_rows = (
        _rows_from_result_container(
            raw_model_output,
            field="fine_results.raw_model_output",
            allow_keyed=False,
        )
        if isinstance(raw_model_output, Mapping)
        else []
    )
    if not normalized_rows and not raw_rows:
        # Preserve the historical compact keyed-row input shape.  Do this only
        # after checking both explicit window views so ``raw_model_output`` is
        # never misinterpreted as one score row.
        normalized_rows = _rows_from_result_container(
            fine_results,
            field="fine_results",
            allow_keyed=True,
        )
    return normalized_rows, raw_rows


def _index_fine_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[Mapping[str, Any]]], list[str]]:
    """Group fine rows by request ID while preserving first-seen order."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        request_id = _request_id_from_window(row)
        if request_id is None:
            continue
        if request_id not in grouped:
            grouped[request_id] = []
            order.append(request_id)
        grouped[request_id].append(row)
    return grouped, order


def _merge_candidate_views(
    preferred: Mapping[str, Mapping[str, Any]],
    fallback: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge candidate maps, preferring raw identity/score evidence.

    Raw rows are passed as ``preferred``.  A normalized candidate can still
    contribute fields that the raw side lacks (for example a retained camera
    evidence list), but it never replaces a raw score, rank, or identity.
    """

    merged: dict[str, dict[str, Any]] = {}
    for action, candidate in preferred.items():
        merged[action] = dict(candidate)
    for action, candidate in fallback.items():
        existing = merged.get(action)
        if existing is None:
            merged[action] = dict(candidate)
            continue
        for key, value in candidate.items():
            if key not in existing or existing[key] is None or existing[key] in ([], {}):
                existing[key] = value
    return list(merged.values())


def _merge_score_window_rows(
    normalized_window: Mapping[str, Any] | None,
    raw_window: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a scoreable raw-preferred view of one normalized/raw pair.

    ``build_preannotation_envelope`` removes ``provisional_id`` from the
    top-level normalized candidate and retains the producer row under
    ``candidate.raw``.  Older persisted envelopes may omit that nested field as
    well, while ``raw_model_output.windows`` still has the authoritative fused
    candidate IDs.  Merge candidate maps explicitly instead of relying on the
    normalized list's shape or ordering.
    """

    if normalized_window is None and raw_window is None:
        return {}
    if normalized_window is None:
        return dict(raw_window or {})
    if raw_window is None:
        return dict(normalized_window)

    merged = dict(normalized_window)
    # Keep normalized timing/request metadata, but let raw scalar score fields
    # win when both views expose a direct score row.
    for key, value in raw_window.items():
        if key in {"top_k", "candidates", "proposals", "fused"}:
            continue
        if key not in merged or merged[key] is None:
            merged[key] = value
    for key in (
        "score",
        "fused_score",
        "visual_score",
        "similarity",
        "cosine",
        "confidence",
    ):
        if raw_window.get(key) is not None:
            merged[key] = raw_window[key]

    raw_candidates = _candidate_rows_from_window(raw_window)
    normalized_candidates = _candidate_rows_from_window(normalized_window)
    candidates = _merge_candidate_views(raw_candidates, normalized_candidates)
    if candidates:
        # ``_score_from_window`` checks this field first, so the raw-preferred
        # candidate map is authoritative even when normalized proposals omit
        # opaque IDs from their top-k rows.
        merged["top_k"] = candidates
    return merged


def _normalise_score_rows(
    fine_plan: Mapping[str, Any],
    fine_results: Mapping[str, Any] | Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """Extract target-action scores from open-runner windows or simple rows."""

    requests = _fine_request_map(fine_plan)
    normalized_rows, raw_rows = _fine_result_row_views(fine_results)
    normalized_by_request, normalized_order = _index_fine_rows(normalized_rows)
    raw_by_request, raw_order = _index_fine_rows(raw_rows)
    ordered_request_ids = list(normalized_order)
    ordered_request_ids.extend(
        request_id for request_id in raw_order if request_id not in normalized_by_request
    )
    output: dict[str, dict[str, Any]] = {}
    for request_id in ordered_request_ids:
        if request_id is None or request_id not in requests:
            continue
        normalized_group = normalized_by_request.get(request_id, [])
        raw_group = raw_by_request.get(request_id, [])
        normalized_window: Mapping[str, Any] | None = None
        for row in normalized_group:
            normalized_window = _merge_score_window_rows(normalized_window, row)
        raw_window: Mapping[str, Any] | None = None
        for row in raw_group:
            raw_window = _merge_score_window_rows(raw_window, row)
        window = _merge_score_window_rows(normalized_window, raw_window)
        request = requests[request_id]
        action = _text(request.get("action_key"), field=f"fine_plan[{request_id}].action_key")
        score_row = _score_from_window(
            window,
            action=action,
            score_policy=str(fine_plan.get("score_policy", "absolute")),
            min_camera_support=int(fine_plan.get("min_camera_support", 1)),
        )
        # Simple score rows may put the score directly on the row rather than
        # under a top-k proposal.
        if score_row is None and window.get("score") is not None:
            score = _finite(window.get("score"), field=f"fine_results[{request_id}].score")
            score = max(0.0, min(1.0, score))
            # Compact keyed rows have no candidate object for
            # ``_score_from_window`` to inspect, but they may still carry the
            # producer rank.  Preserve the top1 policy for this shape too.
            rank = _candidate_rank(window)
            if str(fine_plan.get("score_policy", "absolute")) == "top1" and rank != 1:
                score = 0.0
            score_row = (score, {"score": score, "raw_score": score, "rank": rank})
        if score_row is None:
            continue
        score, evidence = score_row
        padding = _merge_padding_provenance(
            *(normalized_group or ({},)),
            *(raw_group or ({},)),
            request,
        )
        output[request_id] = {
            "request_id": request_id,
            "score": score,
            "start_seconds": _finite(
                request.get("start_seconds"), field=f"fine_plan[{request_id}].start_seconds"
            ),
            "end_seconds": _finite(
                request.get("end_seconds"), field=f"fine_plan[{request_id}].end_seconds"
            ),
            "parent_request_id": request.get("parent_request_id"),
            "role": request.get("role"),
            "probe_side": request.get("probe_side"),
            "level": request.get("level", 0),
            "anchor_seconds": request.get("anchor_seconds"),
            "edge_clipped": bool(request.get("edge_clipped", False)),
            "evidence": evidence,
            **padding,
        }
    return output


def _relative_competitor_hints(
    fine_plan: Mapping[str, Any],
    requests: Mapping[str, Mapping[str, Any]],
    override: Mapping[str, str] | None,
) -> dict[str, dict[str, Any]]:
    """Collect explicit target-vs-neighbour hints for relative scoring.

    A fine plan may be produced by an older caller that knows nothing about
    relative scoring.  Hints are therefore optional.  When absent, the
    normalizer selects a stable runner-up from the observed Top-K rows rather
    than treating a missing competitor as a zero score.
    """

    result: dict[str, dict[str, Any]] = {}
    parents = fine_plan.get("parents")
    if isinstance(parents, Sequence) and not isinstance(parents, (str, bytes, bytearray)):
        for raw_parent in parents:
            if not isinstance(raw_parent, Mapping):
                continue
            parent_id = raw_parent.get("request_id")
            if not isinstance(parent_id, str) or not parent_id.strip():
                continue
            hint = _relative_action_hint(raw_parent)
            if hint is not None:
                result[parent_id] = {
                    "action_key": hint,
                    "source": "fine_plan.parent",
                }

    # Accept small keyed maps from orchestration code without making them part
    # of the required plan contract.
    for map_key in (
        "relative_action_by_parent",
        "comparison_action_by_parent",
        "competitor_action_by_parent",
        "adjacent_action_by_parent",
    ):
        raw_map = fine_plan.get(map_key)
        if not isinstance(raw_map, Mapping):
            continue
        for parent_id, raw_action in raw_map.items():
            if isinstance(parent_id, str) and isinstance(raw_action, str) and raw_action.strip():
                result[parent_id] = {
                    "action_key": raw_action.strip(),
                    "source": f"fine_plan.{map_key}",
                }

    for request_id, request in requests.items():
        hint = _relative_action_hint(request)
        if hint is not None:
            parent_id = request.get("parent_request_id")
            key = parent_id if isinstance(parent_id, str) and parent_id else request_id
            result[key] = {"action_key": hint, "source": "fine_plan.request"}

    if override is not None:
        for parent_id, action in override.items():
            if isinstance(parent_id, str) and isinstance(action, str) and action.strip():
                result[parent_id] = {
                    "action_key": action.strip(),
                    "source": "resolver.override",
                }
    return result


def _relative_payload(
    fine_results: Mapping[str, Any] | Sequence[Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Extract raw-preferred rows for relative score resolution.

    Relative margins need the same candidate identity recovery as absolute
    scores.  In particular, a normalized pre-annotation envelope may retain
    only ``label_text``/``score`` in ``windows[].proposals[].top_k`` while the
    raw sidecar keeps the opaque ``action_key``.  Merge both views before
    selecting the target and competitor, and retain the raw rows separately so
    padding provenance remains explicit to the caller.
    """

    normalized_rows, raw_rows = _fine_result_row_views(fine_results)
    normalized_by_request, normalized_order = _index_fine_rows(normalized_rows)
    raw_by_request, raw_order = _index_fine_rows(raw_rows)
    ordered_request_ids = list(normalized_order)
    ordered_request_ids.extend(
        request_id for request_id in raw_order if request_id not in normalized_by_request
    )

    merged_rows: list[Mapping[str, Any]] = []
    raw_padding_by_request: dict[str, Mapping[str, Any]] = {}
    for request_id in ordered_request_ids:
        normalized_window: Mapping[str, Any] | None = None
        for row in normalized_by_request.get(request_id, []):
            normalized_window = _merge_score_window_rows(normalized_window, row)
        raw_window: Mapping[str, Any] | None = None
        for row in raw_by_request.get(request_id, []):
            raw_window = _merge_score_window_rows(raw_window, row)
        merged_window = _merge_score_window_rows(normalized_window, raw_window)
        if not merged_window:
            continue
        merged_rows.append(merged_window)
        if raw_window is not None:
            raw_padding_by_request[request_id] = raw_window
    return merged_rows, raw_padding_by_request


def _normalise_relative_score_rows(
    fine_plan: Mapping[str, Any],
    fine_results: Mapping[str, Any] | Sequence[Any],
    *,
    min_camera_support: int,
    min_target_score: float,
    competitor_action_by_parent: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Normalize target-versus-neighbour margins for fine score probes.

    Relative scoring is intentionally separate from ``_normalise_score_rows``:
    an absolute target score and a signed target-minus-competitor margin have
    different missing-data semantics.  In particular, an absent competitor is
    *unknown*, never an observed zero.
    """

    requests = _fine_request_map(fine_plan)
    hints = _relative_competitor_hints(fine_plan, requests, competitor_action_by_parent)
    raw_rows, raw_padding_by_request = _relative_payload(fine_results)
    parsed: list[dict[str, Any]] = []
    skipped_unknown_request = 0
    skipped_missing_target = 0
    skipped_target_floor = 0
    for window in raw_rows:
        request_id_raw = window.get("request_id", window.get("refinement_request_id"))
        request_id = _window_id_to_request_id(request_id_raw)
        if request_id is None:
            request_id = _window_id_to_request_id(window.get("window_id"))
        if request_id is None or request_id not in requests:
            skipped_unknown_request += 1
            continue
        request = requests[request_id]
        action = _text(request.get("action_key"), field=f"fine_plan[{request_id}].action_key")
        candidates = _candidate_rows_from_window(window)
        target = candidates.get(action)
        if target is None:
            skipped_missing_target += 1
            continue
        target_score = _candidate_score(target)
        if target_score is None or target_score < min_target_score - 1e-9:
            skipped_target_floor += 1
            continue
        row_hint = _relative_action_hint(window)
        parent_id = request.get("parent_request_id")
        parent_hint = hints.get(parent_id) if isinstance(parent_id, str) else None
        explicit_hint = row_hint or (
            parent_hint.get("action_key") if isinstance(parent_hint, Mapping) else None
        )
        runner_up: Mapping[str, Any] | None = None
        if explicit_hint is not None and explicit_hint != action:
            hinted = candidates.get(explicit_hint)
            # A relative margin is only meaningful when both sides meet the
            # requested camera-support floor.  Keep an explicit hint when it
            # is eligible; an unsupported hint is not allowed to suppress a
            # different, supported competitor.
            if hinted is not None and _candidate_support(hinted) >= min_camera_support:
                runner_up = hinted
        if runner_up is None:
            alternatives = [
                candidate
                for candidate_action, candidate in candidates.items()
                if candidate_action != action
                and _candidate_support(candidate) >= min_camera_support
            ]
            alternatives.sort(
                key=lambda candidate: (
                    -float(candidate.get("score", 0.0)),
                    _candidate_rank(candidate),
                    str(_candidate_action(candidate)),
                )
            )
            runner_up = alternatives[0] if alternatives else None
        # Keep the parsed candidate list so a stable competitor can be chosen
        # across probes after all rows have been inspected.
        parsed.append(
            {
                "request_id": request_id,
                "request": request,
                "window": window,
                "candidates": candidates,
                "target_action_key": action,
                "target": target,
                "target_score": target_score,
                "explicit_competitor_action_key": explicit_hint,
                "row_runner_up": runner_up,
                "raw_padding_row": raw_padding_by_request.get(request_id),
            }
        )

    # Resolve one competitor per parent.  A fixed competitor prevents a
    # changing runner-up from manufacturing a margin crossing unrelated to the
    # target action.  Explicit hints always win; otherwise choose the most
    # frequently observed runner-up, with mean score and lexical order as
    # deterministic tie-breakers.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in parsed:
        parent_id = item["request"].get("parent_request_id")
        if isinstance(parent_id, str) and parent_id.strip():
            grouped.setdefault(parent_id, []).append(item)
    selected_competitors: dict[str, dict[str, Any]] = {}
    for parent_id, items in grouped.items():
        explicit_actions = {
            str(item["explicit_competitor_action_key"])
            for item in items
            if isinstance(item.get("explicit_competitor_action_key"), str)
            and item["explicit_competitor_action_key"] != item["target_action_key"]
            and isinstance(item.get("row_runner_up"), Mapping)
            and _candidate_action(item["row_runner_up"]) == item["explicit_competitor_action_key"]
        }
        if len(explicit_actions) == 1:
            action = next(iter(explicit_actions))
            selected_competitors[parent_id] = {
                "action_key": action,
                "source": "explicit_hint",
            }
            continue
        observations: dict[str, list[float]] = {}
        for item in items:
            runner_up = item.get("row_runner_up")
            if not isinstance(runner_up, Mapping):
                continue
            action_value = _candidate_action(runner_up)
            score = _candidate_score(runner_up)
            if (
                isinstance(action_value, str)
                and action_value != item["target_action_key"]
                and score is not None
            ):
                observations.setdefault(action_value, []).append(score)
        if observations:
            action = sorted(
                observations,
                key=lambda candidate_action: (
                    -len(observations[candidate_action]),
                    -sum(observations[candidate_action]) / len(observations[candidate_action]),
                    candidate_action,
                ),
            )[0]
            selected_competitors[parent_id] = {
                "action_key": action,
                "source": "runner_up_consensus",
                "observed_action_count": len(observations[action]),
                "observed_row_count": len(items),
            }

    output: dict[str, dict[str, Any]] = {}
    skipped_missing_competitor = 0
    skipped_unsupported = 0
    skipped_camera_mismatch = 0
    competitor_action_by_request: dict[str, str] = {}
    competitor_selection_source_by_parent: dict[str, str] = {}
    for item in parsed:
        request = item["request"]
        request_id = str(item["request_id"])
        parent_id = request.get("parent_request_id")
        parent_key = parent_id if isinstance(parent_id, str) else request_id
        selection = selected_competitors.get(parent_key)
        competitor_action = selection.get("action_key") if isinstance(selection, Mapping) else None
        if not isinstance(competitor_action, str) or not competitor_action:
            # A request-level explicit hint may still be usable when its parent
            # has no other observed rows.
            explicit = item.get("explicit_competitor_action_key")
            competitor_action = explicit if isinstance(explicit, str) else None
            if competitor_action:
                selection = {"action_key": competitor_action, "source": "row_hint"}
        if not competitor_action or competitor_action == item["target_action_key"]:
            # An explicit hint can be retained for diagnostics even when the
            # hinted row was not camera-support eligible.  Never resurrect it
            # here: doing so would bypass the eligibility filter above.
            skipped_missing_competitor += 1
            continue
        competitor = item["candidates"].get(competitor_action)
        if competitor is None:
            skipped_missing_competitor += 1
            continue
        target_value = item.get("target")
        if not isinstance(target_value, Mapping):
            skipped_missing_target += 1
            continue
        target_candidate = target_value
        target_support = _candidate_support(target_candidate)
        competitor_support = _candidate_support(competitor)
        if target_support < min_camera_support or competitor_support < min_camera_support:
            skipped_unsupported += 1
            continue
        target_score = _candidate_score(target_candidate)
        competitor_score = _candidate_score(competitor)
        if target_score is None or competitor_score is None:
            skipped_missing_competitor += 1
            continue
        target_camera_ids = _candidate_camera_ids(target_candidate)
        competitor_camera_ids = _candidate_camera_ids(competitor)
        camera_overlap_ids = sorted(set(target_camera_ids) & set(competitor_camera_ids))
        camera_sets_compatible = (
            not target_camera_ids or not competitor_camera_ids or bool(camera_overlap_ids)
        )
        if not camera_sets_compatible:
            skipped_camera_mismatch += 1
            continue
        margin = target_score - competitor_score
        raw_padding_row = item.get("raw_padding_row")
        padding = _merge_padding_provenance(
            item["window"],
            raw_padding_row if isinstance(raw_padding_row, Mapping) else {},
            request,
        )
        selection_source = (
            selection.get("source", "unknown") if isinstance(selection, Mapping) else "unknown"
        )
        competitor_action_by_request[request_id] = competitor_action
        competitor_selection_source_by_parent[parent_key] = str(selection_source)
        output[request_id] = {
            "request_id": request_id,
            "score": margin,
            "raw_score": target_score,
            "target_score": target_score,
            "competitor_score": competitor_score,
            "margin": margin,
            "relative_margin": margin,
            "target_action_key": item["target_action_key"],
            "competitor_action_key": competitor_action,
            "relative_action_key": competitor_action,
            "competitor_selection_source": selection_source,
            "target_rank": _candidate_rank(target_candidate),
            "competitor_rank": _candidate_rank(competitor),
            "target_camera_support": target_support,
            "competitor_camera_support": competitor_support,
            "target_camera_ids": list(target_camera_ids),
            "competitor_camera_ids": list(competitor_camera_ids),
            "camera_overlap_ids": camera_overlap_ids,
            "camera_sets_compatible": camera_sets_compatible,
            "target_score_floor": min_target_score,
            "target_score_floor_passed": target_score >= min_target_score - 1e-9,
            "start_seconds": _finite(
                request.get("start_seconds"), field=f"fine_plan[{request_id}].start_seconds"
            ),
            "end_seconds": _finite(
                request.get("end_seconds"), field=f"fine_plan[{request_id}].end_seconds"
            ),
            "parent_request_id": request.get("parent_request_id"),
            "role": request.get("role"),
            "probe_side": request.get("probe_side"),
            "level": request.get("level", 0),
            "anchor_seconds": request.get("anchor_seconds"),
            "edge_clipped": bool(request.get("edge_clipped", False)),
            "evidence": {
                "score": margin,
                "raw_score": target_score,
                "target_score": target_score,
                "competitor_score": competitor_score,
                "margin": margin,
                "target_action_key": item["target_action_key"],
                "competitor_action_key": competitor_action,
                "relative_action_key": competitor_action,
                "competitor_selection_source": selection_source,
                "target_rank": _candidate_rank(target_candidate),
                "competitor_rank": _candidate_rank(competitor),
                "target_camera_support": target_support,
                "competitor_camera_support": competitor_support,
                "target_camera_ids": list(target_camera_ids),
                "competitor_camera_ids": list(competitor_camera_ids),
                "camera_overlap_ids": camera_overlap_ids,
                "camera_sets_compatible": camera_sets_compatible,
                "target_score_floor": min_target_score,
                "target_score_floor_passed": target_score >= min_target_score - 1e-9,
                "target_candidate": _copy_json(
                    target_candidate, field="fine_score.target_candidate"
                ),
                "competitor_candidate": _copy_json(
                    competitor, field="fine_score.competitor_candidate"
                ),
            },
            **padding,
        }

    diagnostics = {
        "relative_parent_competitors": {
            parent_id: dict(selection)
            for parent_id, selection in sorted(selected_competitors.items())
        },
        "relative_competitor_action_by_request": dict(sorted(competitor_action_by_request.items())),
        "relative_competitor_selection_source_by_parent": dict(
            sorted(competitor_selection_source_by_parent.items())
        ),
        "relative_candidate_row_count": len(parsed),
        "relative_score_row_count": len(output),
        "relative_skipped_unknown_request_count": skipped_unknown_request,
        "relative_skipped_missing_target_count": skipped_missing_target,
        "relative_skipped_missing_competitor_count": skipped_missing_competitor,
        "relative_skipped_unsupported_count": skipped_unsupported,
        "relative_skipped_target_floor_count": skipped_target_floor,
        "relative_skipped_camera_mismatch_count": skipped_camera_mismatch,
        "relative_margin_min_target_score": min_target_score,
        "relative_competitor_is_not_zero_filled": True,
    }
    return output, diagnostics


def _interpolated_boundary(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    threshold: float,
) -> tuple[float, float]:
    x0 = (float(left["start_seconds"]) + float(left["end_seconds"])) / 2.0
    x1 = (float(right["start_seconds"]) + float(right["end_seconds"])) / 2.0
    s0, s1 = float(left["score"]), float(right["score"])
    if x1 <= x0:
        raise ProductionWemmTemporalScoreRefinementError("fine probe centres must increase")
    denominator = s1 - s0
    alpha = 0.5 if abs(denominator) <= 1e-9 else (threshold - s0) / denominator
    alpha = max(0.0, min(1.0, alpha))
    boundary = x0 + alpha * (x1 - x0)
    confidence = max(0.0, min(1.0, abs(s1 - s0)))
    return boundary, confidence


def _result_interval(
    *,
    boundary: float,
    parent: Mapping[str, Any],
    resolution: float,
) -> tuple[float, float] | None:
    start = float(parent["start_seconds"])
    end = float(parent["end_seconds"])
    half = max(resolution / 2.0, 1e-3)
    left = max(start, boundary - half)
    right = min(end, boundary + half)
    if right <= left + 1e-9:
        return None
    return left - start, right - start


def _relative_margin_persistence(
    rows: Sequence[Mapping[str, Any]],
    *,
    left_index: int,
    right_index: int,
    role: str,
    threshold: float,
) -> int:
    """Count contiguous post-crossing probes on the expected margin side."""

    count = 0

    def _expected(value: float) -> bool:
        return value >= threshold if role == "onset" else value < threshold

    for row in rows[right_index:]:
        if not _expected(float(row["score"])):
            break
        count += 1
    return count


def resolve_wemm_score_refinement(
    parent_plan: Mapping[str, Any],
    fine_plan: Mapping[str, Any],
    fine_results: Mapping[str, Any] | Sequence[Any],
    *,
    start_threshold: float = 0.65,
    stop_threshold: float = 0.50,
    score_policy: str = "top1",
    min_camera_support: int = 1,
    min_boundary_resolution_seconds: float = DEFAULT_MIN_BOUNDARY_RESOLUTION_SECONDS,
    start_margin_threshold: float = DEFAULT_START_MARGIN_THRESHOLD,
    stop_margin_threshold: float = DEFAULT_STOP_MARGIN_THRESHOLD,
    margin_threshold: float | None = None,
    min_margin_persistence: int = DEFAULT_MIN_MARGIN_PERSISTENCE,
    relative_margin_scale: float = DEFAULT_RELATIVE_MARGIN_SCALE,
    relative_margin_min_target_score: float = 0.60,
    competitor_action_by_parent: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve fine probe scores into parent-request ``MEASURED`` rows.

    The returned ``results`` are intentionally keyed by the original short
    request IDs generated by ``plan_wemm_temporal_refinement``.  This makes
    the result directly consumable by ``apply_refined_boundaries``.  For each
    role, only a threshold crossing on the expected before/after sides of the
    coarse anchor can become ``MEASURED``; all other cases are ``UNCERTAIN``.

    ``score_policy='relative_margin'`` (with ``'candidate_relative'``,
    ``'relative'``, and ``'contrast'`` accepted as aliases) is an additive
    review-only mode.  It compares the
    target action's fine-probe score with one stable neighbouring action and
    resolves a signed margin crossing around zero (or the supplied margin
    threshold).  The historical ``absolute`` and ``top1`` policies retain
    their existing behavior.
    """

    fine = _mapping(fine_plan, field="fine_plan")
    fine_source = fine.get("source", {})
    fine_context = (
        fine_source.get("context_interval", {}) if isinstance(fine_source, Mapping) else {}
    )
    parent_map = {
        str(row["request_id"]): row
        for row in _parent_requests(
            {"context_interval": fine_context},
            parent_plan,
        )
    }
    requested_score_policy = _text(score_policy, field="score_policy")
    if requested_score_policy not in SCORE_POLICIES:
        raise ProductionWemmTemporalScoreRefinementError(
            f"score_policy must be one of {', '.join(SCORE_POLICIES)}"
        )
    effective_score_policy = _canonical_score_policy(requested_score_policy)
    relative_mode = effective_score_policy == SCORE_POLICY_RELATIVE_MARGIN
    start_threshold = _finite(start_threshold, field="start_threshold")
    stop_threshold = _finite(stop_threshold, field="stop_threshold")
    if not 0.0 <= start_threshold <= 1.0:
        raise ProductionWemmTemporalScoreRefinementError("start_threshold must be between 0 and 1")
    if not 0.0 <= stop_threshold <= 1.0:
        raise ProductionWemmTemporalScoreRefinementError("stop_threshold must be between 0 and 1")
    if stop_threshold > start_threshold:
        raise ProductionWemmTemporalScoreRefinementError(
            "stop_threshold must be <= start_threshold"
        )
    if isinstance(relative_margin_scale, bool) or not isinstance(
        relative_margin_scale, (int, float)
    ):
        raise ProductionWemmTemporalScoreRefinementError(
            "relative_margin_scale must be positive and finite"
        )
    relative_margin_scale = float(relative_margin_scale)
    if not math.isfinite(relative_margin_scale) or relative_margin_scale <= 0.0:
        raise ProductionWemmTemporalScoreRefinementError(
            "relative_margin_scale must be positive and finite"
        )
    if margin_threshold is not None and relative_mode:
        shared_margin = _finite(margin_threshold, field="margin_threshold")
        start_margin_threshold = shared_margin
        stop_margin_threshold = shared_margin
    if relative_mode:
        start_margin_threshold = _finite(start_margin_threshold, field="start_margin_threshold")
        stop_margin_threshold = _finite(stop_margin_threshold, field="stop_margin_threshold")
        if not -1.0 <= start_margin_threshold <= 1.0:
            raise ProductionWemmTemporalScoreRefinementError(
                "start_margin_threshold must be between -1 and 1"
            )
        if not -1.0 <= stop_margin_threshold <= 1.0:
            raise ProductionWemmTemporalScoreRefinementError(
                "stop_margin_threshold must be between -1 and 1"
            )
        if stop_margin_threshold > start_margin_threshold:
            raise ProductionWemmTemporalScoreRefinementError(
                "stop_margin_threshold must be <= start_margin_threshold"
            )
    if (
        isinstance(min_margin_persistence, bool)
        or not isinstance(min_margin_persistence, int)
        or min_margin_persistence <= 0
    ):
        raise ProductionWemmTemporalScoreRefinementError(
            "min_margin_persistence must be a positive integer"
        )
    relative_margin_min_target_score = _finite(
        relative_margin_min_target_score,
        field="relative_margin_min_target_score",
    )
    if not 0.0 <= relative_margin_min_target_score <= 1.0:
        raise ProductionWemmTemporalScoreRefinementError(
            "relative_margin_min_target_score must be between 0 and 1"
        )
    support_min = _positive_int(min_camera_support, field="min_camera_support")
    min_resolution = _positive(
        min_boundary_resolution_seconds, field="min_boundary_resolution_seconds"
    )
    # Avoid depending on optional metadata in the plan; annotate rows before
    # extracting scores so a caller can pass a plain open-runner envelope.
    fine_plan_copy = dict(fine_plan)
    fine_plan_copy["score_policy"] = effective_score_policy
    fine_plan_copy["min_camera_support"] = support_min
    fine_plan_copy["relative_margin_scale"] = relative_margin_scale
    relative_diagnostics: dict[str, Any] = {}
    if effective_score_policy == SCORE_POLICY_RELATIVE_MARGIN:
        scores, relative_diagnostics = _normalise_relative_score_rows(
            fine_plan_copy,
            fine_results,
            min_camera_support=support_min,
            min_target_score=relative_margin_min_target_score,
            competitor_action_by_parent=competitor_action_by_parent,
        )
        relative_diagnostics["relative_margin_scale"] = relative_margin_scale
    else:
        scores = _normalise_score_rows(fine_plan_copy, fine_results)
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in scores.values():
        parent_id = row.get("parent_request_id")
        if isinstance(parent_id, str) and parent_id in parent_map:
            by_parent.setdefault(parent_id, []).append(row)

    output_rows: list[dict[str, Any]] = []
    measured_count = 0
    unresolved_count = 0
    padded_score_row_count = sum(int(row.get("padding_used") is True) for row in scores.values())
    padded_crossing_count = 0
    padded_boundary_rejection_count = 0
    non_persistent_crossing_count = 0
    for parent_id, parent in parent_map.items():
        role = str(parent["role"])
        anchor_raw = parent.get("coarse_anchor_seconds")
        anchor = (
            _finite(anchor_raw, field=f"parent[{parent_id}].coarse_anchor_seconds")
            if anchor_raw is not None
            else (float(parent["start_seconds"]) + float(parent["end_seconds"])) / 2.0
        )
        rows = by_parent.get(parent_id, [])
        rows.sort(
            key=lambda row: (
                (float(row["start_seconds"]) + float(row["end_seconds"])) / 2.0,
                int(row.get("level", 0)),
            )
        )
        # Keep all score crossings separate from the crossings that actually
        # straddle the coarse anchor.  A valid threshold crossing on one side
        # of the anchor is not evidence for the requested onset/offset: it
        # can be caused by a non-monotone score trajectory elsewhere in the
        # context.  Earlier versions accepted such a same-side crossing as a
        # fallback, which could fabricate a precise-looking boundary from an
        # unrelated score fluctuation.
        crossings: list[tuple[dict[str, Any], dict[str, Any], float, bool, bool, int]] = []
        for pair_index, (left, right) in enumerate(pairwise(rows)):
            left_center = (float(left["start_seconds"]) + float(left["end_seconds"])) / 2.0
            right_center = (float(right["start_seconds"]) + float(right["end_seconds"])) / 2.0
            if right_center <= left_center + 1e-9:
                continue
            # A clipped side does not prove that the action was absent/present
            # outside the source.  Keep the parent unresolved rather than
            # converting an edge artifact into a measured boundary.
            if bool(left.get("edge_clipped")) or bool(right.get("edge_clipped")):
                continue
            threshold = float(
                (start_margin_threshold if role == "onset" else stop_margin_threshold)
                if relative_mode
                else (start_threshold if role == "onset" else stop_threshold)
            )
            valid = (
                float(left["score"]) < threshold <= float(right["score"])
                if role == "onset"
                else float(left["score"]) >= threshold > float(right["score"])
            )
            if valid:
                distance = right_center - left_center
                # The role determines which side may include the anchor.  The
                # strict inequality on the opposite side prevents a probe
                # pair entirely after an onset (or entirely before an offset)
                # from being treated as a temporal boundary.  A small numeric
                # tolerance is used only for floating-point coordinates.
                if role == "onset":
                    brackets_anchor = left_center < anchor - 1e-9 and anchor <= right_center + 1e-9
                else:
                    brackets_anchor = left_center <= anchor + 1e-9 and anchor < right_center - 1e-9
                padding_involved = bool(left.get("padding_used")) or bool(right.get("padding_used"))
                persistence = (
                    _relative_margin_persistence(
                        rows,
                        left_index=pair_index,
                        right_index=pair_index + 1,
                        role=role,
                        threshold=threshold,
                    )
                    if relative_mode
                    else 1
                )
                crossings.append(
                    (left, right, distance, brackets_anchor, padding_involved, persistence)
                )
        padded_crossings = [item for item in crossings if item[4]]
        padded_crossing_count += len(padded_crossings)
        if padded_crossings:
            # A padded frame may duplicate a boundary-adjacent image and is
            # therefore not valid evidence for localizing an action edge.
            # Reject the parent request even when another unpadded crossing is
            # available: otherwise selection order could turn the same probe
            # set into a measured result or an uncertainty.
            padded_boundary_rejection_count += 1
            unresolved_count += 1
            output_rows.append(
                {
                    "request_id": parent_id,
                    "status": "UNCERTAIN",
                    "confidence": 0.0,
                    "evidence": {
                        "reason": "PADDED_FINE_SCORE_PROBE",
                        "score_policy": requested_score_policy,
                        "role": role,
                        "anchor_seconds": anchor,
                        "observed_probe_count": len(rows),
                        "threshold_crossing_count": len(crossings),
                        "padded_crossing_count": len(padded_crossings),
                        "padded_probe_request_ids": sorted(
                            {
                                request_id
                                for item in padded_crossings
                                for probe in item[:2]
                                for request_id in [probe.get("request_id")]
                                if isinstance(request_id, str)
                            }
                        ),
                        "rejected_padded_crossings": [
                            {
                                "left_probe": _copy_json(item[0], field="evidence.left_probe"),
                                "right_probe": _copy_json(item[1], field="evidence.right_probe"),
                            }
                            for item in padded_crossings
                        ],
                        "request_edges_are_not_boundaries": True,
                    },
                }
            )
            continue
        pairs = [
            item
            for item in crossings
            if item[3] and (not relative_mode or item[5] >= min_margin_persistence)
        ]
        non_persistent = [
            item
            for item in crossings
            if item[3] and relative_mode and item[5] < min_margin_persistence
        ]
        non_persistent_crossing_count += len(non_persistent)
        if pairs:
            # Prefer the tightest bracket, then the finest level, then the
            # largest score jump.  This is deterministic across camera/order.
            left, right, distance, _brackets_anchor, _padding_involved, persistence = sorted(
                pairs,
                key=lambda item: (
                    abs(
                        (
                            (float(item[0]["start_seconds"]) + float(item[0]["end_seconds"])) / 2.0
                            + (float(item[1]["start_seconds"]) + float(item[1]["end_seconds"]))
                            / 2.0
                        )
                        / 2.0
                        - anchor
                    ),
                    item[2],
                    max(int(item[0].get("level", 0)), int(item[1].get("level", 0))),
                    -abs(float(item[1]["score"]) - float(item[0]["score"])),
                    str(item[0]["request_id"]),
                ),
            )[0]
            threshold = (
                (start_margin_threshold if role == "onset" else stop_margin_threshold)
                if relative_mode
                else (start_threshold if role == "onset" else stop_threshold)
            )
            boundary, confidence = _interpolated_boundary(left, right, threshold=threshold)
            parent_start = float(parent["start_seconds"])
            parent_end = float(parent["end_seconds"])
            if boundary < parent_start - 1e-9 or boundary > parent_end + 1e-9:
                # ``apply_refined_boundaries`` uses the parent request as its
                # relative coordinate frame.  A crossing outside that frame
                # cannot be represented without clamping (which would turn a
                # context edge into a fabricated boundary), so leave it
                # unresolved and retain the diagnostic evidence below.
                unresolved_count += 1
                output_rows.append(
                    {
                        "request_id": parent_id,
                        "status": "UNCERTAIN",
                        "confidence": 0.0,
                        "evidence": {
                            "reason": (
                                "FINE_SCORE_MARGIN_CROSSING_OUTSIDE_PARENT_REQUEST"
                                if relative_mode
                                else "FINE_SCORE_CROSSING_OUTSIDE_PARENT_REQUEST"
                            ),
                            "score_policy": requested_score_policy,
                            "role": role,
                            "anchor_seconds": anchor,
                            "source_boundary_seconds": round(boundary, 6),
                            "parent_interval": {
                                "start_seconds": parent_start,
                                "end_seconds": parent_end,
                            },
                            "left_probe": _copy_json(left, field="evidence.left_probe"),
                            "right_probe": _copy_json(right, field="evidence.right_probe"),
                            "persistence_count": persistence,
                            "request_edges_are_not_boundaries": True,
                        },
                    }
                )
                continue
            interval = _result_interval(
                boundary=boundary,
                parent=parent,
                resolution=max(min_resolution, distance / 2.0),
            )
            if interval is not None:
                rel_start, rel_end = interval
                measured_count += 1
                output_rows.append(
                    {
                        "request_id": parent_id,
                        "status": "MEASURED",
                        "timestamp_basis": REQUEST_TIMESTAMP_BASIS,
                        "start_seconds": round(rel_start, 6),
                        "end_seconds": round(rel_end, 6),
                        "confidence": round(confidence, 6),
                        "evidence": {
                            "reason": (
                                "FINE_SCORE_MARGIN_CROSSING"
                                if relative_mode
                                else "FINE_SCORE_THRESHOLD_CROSSING"
                            ),
                            "score_policy": requested_score_policy,
                            "role": role,
                            "threshold": threshold,
                            "source_boundary_seconds": round(boundary, 6),
                            "boundary_resolution_seconds": round(distance / 2.0, 6),
                            "persistence_count": persistence,
                            "left_score": float(left["score"]),
                            "right_score": float(right["score"]),
                            "score_delta": float(right["score"]) - float(left["score"]),
                            **(
                                {
                                    "target_action_key": left.get("target_action_key"),
                                    "competitor_action_key": left.get("competitor_action_key"),
                                    "relative_action_key": left.get("relative_action_key"),
                                    "competitor_selection_source": left.get(
                                        "competitor_selection_source"
                                    ),
                                    "target_score_before": left.get("target_score"),
                                    "target_score_after": right.get("target_score"),
                                    "competitor_score_before": left.get("competitor_score"),
                                    "competitor_score_after": right.get("competitor_score"),
                                    "margin_before": left.get("margin", left.get("score")),
                                    "margin_after": right.get("margin", right.get("score")),
                                }
                                if relative_mode
                                else {}
                            ),
                            "left_probe": _copy_json(left, field="evidence.left_probe"),
                            "right_probe": _copy_json(right, field="evidence.right_probe"),
                            "request_edges_are_not_boundaries": True,
                        },
                    }
                )
                continue
        unresolved_count += 1
        if crossings:
            # Preserve enough provenance for review to distinguish "no score
            # change" from "score changed, but not across the coarse anchor".
            # The probe rows are already bounded and JSON-compatible, so this
            # remains a small, useful diagnostic rather than a second model
            # output or a heavy audit artifact.
            rejected = [item for item in crossings if not item[3]]
            if relative_mode and non_persistent and not rejected:
                reason = "NON_PERSISTENT_FINE_SCORE_MARGIN_CROSSING"
            elif relative_mode:
                reason = "NO_ANCHOR_BRACKETED_FINE_SCORE_MARGIN_CROSSING"
            else:
                reason = "NO_ANCHOR_BRACKETED_FINE_SCORE_CROSSING"
            evidence: dict[str, Any] = {
                "reason": reason,
                "score_policy": requested_score_policy,
                "role": role,
                "anchor_seconds": anchor,
                "observed_probe_count": len(rows),
                "threshold_crossing_count": len(crossings),
                "same_side_crossing_count": len(rejected),
                "non_persistent_crossing_count": len(non_persistent),
                "min_margin_persistence": min_margin_persistence,
                "request_edges_are_not_boundaries": True,
            }
            if rejected:
                evidence["rejected_same_side_crossings"] = [
                    {
                        "left_probe": _copy_json(item[0], field="evidence.left_probe"),
                        "right_probe": _copy_json(item[1], field="evidence.right_probe"),
                    }
                    for item in rejected
                ]
        else:
            evidence = {
                "reason": (
                    "NO_FINE_SCORE_MARGIN_CROSSING"
                    if relative_mode
                    else "NO_FINE_SCORE_THRESHOLD_CROSSING"
                ),
                "score_policy": requested_score_policy,
                "role": role,
                "anchor_seconds": anchor,
                "observed_probe_count": len(rows),
                "min_margin_persistence": min_margin_persistence,
                "request_edges_are_not_boundaries": True,
            }
        output_rows.append(
            {
                "request_id": parent_id,
                "status": "UNCERTAIN",
                "confidence": 0.0,
                "evidence": evidence,
            }
        )
    output_rows.sort(key=lambda row: str(row["request_id"]))
    return {
        "format": RESULT_FORMAT,
        "authority": AUTHORITY,
        "status": RESULT_STATUS,
        "production_eligible": False,
        "score_policy": requested_score_policy,
        "effective_score_policy": effective_score_policy,
        "parameters": {
            "start_threshold": float(start_threshold),
            "stop_threshold": float(stop_threshold),
            "start_margin_threshold": float(start_margin_threshold),
            "stop_margin_threshold": float(stop_margin_threshold),
            "margin_threshold": (float(margin_threshold) if margin_threshold is not None else None),
            "min_margin_persistence": min_margin_persistence,
            "relative_margin_scale": float(relative_margin_scale),
            "relative_margin_min_target_score": float(relative_margin_min_target_score),
            "min_camera_support": support_min,
            "min_boundary_resolution_seconds": min_resolution,
        },
        "results": output_rows,
        "diagnostics": {
            "parent_request_count": len(parent_map),
            "fine_score_row_count": len(scores),
            "padded_score_row_count": padded_score_row_count,
            "padded_crossing_count": padded_crossing_count,
            "padded_boundary_rejection_count": padded_boundary_rejection_count,
            "non_persistent_crossing_count": non_persistent_crossing_count,
            "measured_result_count": measured_count,
            "unresolved_result_count": unresolved_count,
            "request_edges_used_as_boundaries": False,
            **relative_diagnostics,
        },
        "controls": {
            "media_decoded": False,
            "model_invoked": False,
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
        },
        "limitations": [
            "Measured rows are score-derived review evidence, not gold.",
            "A missing or non-bracketed score remains UNCERTAIN.",
            "Any threshold crossing involving a frame-padded fine probe remains UNCERTAIN.",
            "The parent request interval is used only as a coordinate frame.",
            *(
                [
                    (
                        "Candidate-relative margins compare a target action with one stable "
                        "observed neighbour."
                    ),
                    "An absent or camera-unsupported competitor is unknown, not a zero score.",
                    (
                        "Relative score boundaries are review-only and do not alter coarse "
                        "top1/absolute output."
                    ),
                ]
                if relative_mode
                else []
            ),
        ],
    }


# Descriptive aliases for orchestration callers.
plan_temporal_score_refinement_grid = plan_wemm_score_refinement_grid
resolve_temporal_score_refinement = resolve_wemm_score_refinement


def resolve_wemm_candidate_relative_score_refinement(
    parent_plan: Mapping[str, Any],
    fine_plan: Mapping[str, Any],
    fine_results: Mapping[str, Any] | Sequence[Any],
    *,
    start_margin_threshold: float = DEFAULT_START_MARGIN_THRESHOLD,
    stop_margin_threshold: float = DEFAULT_STOP_MARGIN_THRESHOLD,
    margin_threshold: float | None = None,
    min_margin_persistence: int = DEFAULT_MIN_MARGIN_PERSISTENCE,
    relative_margin_scale: float = DEFAULT_RELATIVE_MARGIN_SCALE,
    relative_margin_min_target_score: float = 0.60,
    min_camera_support: int = 1,
    min_boundary_resolution_seconds: float = DEFAULT_MIN_BOUNDARY_RESOLUTION_SECONDS,
    competitor_action_by_parent: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve a target-versus-neighbour signed margin crossing.

    This is a named convenience wrapper around :func:`resolve_wemm_score_refinement`.
    It is intentionally review-only and leaves the historical absolute/top1
    routes untouched.
    """

    return resolve_wemm_score_refinement(
        parent_plan,
        fine_plan,
        fine_results,
        score_policy=SCORE_POLICY_RELATIVE_MARGIN,
        start_margin_threshold=start_margin_threshold,
        stop_margin_threshold=stop_margin_threshold,
        margin_threshold=margin_threshold,
        min_margin_persistence=min_margin_persistence,
        relative_margin_scale=relative_margin_scale,
        relative_margin_min_target_score=relative_margin_min_target_score,
        min_camera_support=min_camera_support,
        min_boundary_resolution_seconds=min_boundary_resolution_seconds,
        competitor_action_by_parent=competitor_action_by_parent,
    )


resolve_wemm_relative_score_refinement = resolve_wemm_candidate_relative_score_refinement
resolve_wemm_margin_refinement = resolve_wemm_candidate_relative_score_refinement


__all__ = [
    "AUTHORITY",
    "DEFAULT_LEVELS",
    "DEFAULT_MAX_REQUESTS",
    "DEFAULT_MIN_BOUNDARY_RESOLUTION_SECONDS",
    "DEFAULT_MIN_MARGIN_PERSISTENCE",
    "DEFAULT_MIN_PROBE_SPAN_SECONDS",
    "DEFAULT_POINTS_PER_SIDE",
    "DEFAULT_PROBE_SPAN_SECONDS",
    "DEFAULT_RELATIVE_MARGIN_SCALE",
    "DEFAULT_START_MARGIN_THRESHOLD",
    "DEFAULT_STOP_MARGIN_THRESHOLD",
    "FORMAT",
    "RELATIVE_SCORE_POLICIES",
    "RESULT_FORMAT",
    "RESULT_STATUS",
    "ROLES",
    "SCORE_POLICIES",
    "SCORE_POLICY_CANDIDATE_RELATIVE",
    "SCORE_POLICY_CONTRAST",
    "SCORE_POLICY_RELATIVE",
    "SCORE_POLICY_RELATIVE_MARGIN",
    "SIDES",
    "STATUS",
    "ProductionWemmTemporalScoreRefinementError",
    "plan_temporal_score_refinement_grid",
    "plan_wemm_score_refinement_grid",
    "resolve_temporal_score_refinement",
    "resolve_wemm_candidate_relative_score_refinement",
    "resolve_wemm_margin_refinement",
    "resolve_wemm_relative_score_refinement",
    "resolve_wemm_score_refinement",
]
