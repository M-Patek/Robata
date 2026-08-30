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
threshold on the expected side of the coarse anchor.  Missing, edge-clipped,
non-monotone, or unsupported evidence remains ``UNCERTAIN``.  All output is
review-only and contains no gold, ontology mutation, hash, or digest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final

from .production_wemm_temporal_refinement import (
    REQUEST_TIMESTAMP_BASIS,
    plan_wemm_temporal_refinement,
)

FORMAT: Final = "robata-production-wemm-temporal-score-refinement-v1"
RESULT_FORMAT: Final = "robata-production-wemm-temporal-score-result-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "FINE_SCORE_PROBE_REQUESTS_ONLY"
RESULT_STATUS: Final = "FINE_SCORE_BOUNDARIES_REVIEW_ONLY"
SCORE_POLICIES: Final = ("top1", "absolute")
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
    truncated = len(requests) > limit
    if truncated:
        requests = requests[:limit]
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
            "probe_request_count": len(requests),
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


def _candidate_from_window(window: Mapping[str, Any], action: str) -> dict[str, Any] | None:
    """Find one action score in a runner window or score row."""

    candidates: list[Mapping[str, Any]] = []
    for key in ("top_k", "candidates"):
        raw = window.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            candidates.extend(item for item in raw if isinstance(item, Mapping))
    proposals = window.get("proposals")
    if isinstance(proposals, Sequence) and not isinstance(proposals, (str, bytes, bytearray)):
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                continue
            raw_top = proposal.get("top_k")
            if isinstance(raw_top, Sequence) and not isinstance(raw_top, (str, bytes, bytearray)):
                candidates.extend(item for item in raw_top if isinstance(item, Mapping))
            candidate = proposal.get("provisional_id", proposal.get("action_key"))
            if candidate == action:
                return dict(proposal)
    for candidate in candidates:
        candidate_id = candidate.get("provisional_id", candidate.get("action_key"))
        if candidate_id == action:
            return dict(candidate)
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
    raw_score = candidate.get("score", candidate.get("confidence"))
    if raw_score is None:
        return None
    score = _finite(raw_score, field="fine_score.score")
    if not 0.0 <= score <= 1.0:
        score = max(0.0, min(1.0, score))
    support_raw = candidate.get("camera_support", candidate.get("camera_support_count", 0))
    if isinstance(support_raw, Sequence) and not isinstance(support_raw, (str, bytes, bytearray)):
        support = len([item for item in support_raw if isinstance(item, str) and item.strip()])
    else:
        try:
            support = int(support_raw)
        except (TypeError, ValueError, OverflowError):
            support = 0
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


def _normalise_score_rows(
    fine_plan: Mapping[str, Any],
    fine_results: Mapping[str, Any] | Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """Extract target-action scores from open-runner windows or simple rows."""

    requests = _fine_request_map(fine_plan)
    payload: object = fine_results
    if isinstance(payload, Mapping):
        if isinstance(payload.get("windows"), Sequence) and not isinstance(
            payload.get("windows"), (str, bytes, bytearray)
        ):
            payload = payload["windows"]
        elif isinstance(payload.get("results"), Sequence) and not isinstance(
            payload.get("results"), (str, bytes, bytearray)
        ):
            payload = payload["results"]
        else:
            # Mapping keyed by request ID.
            keyed_rows: list[dict[str, Any]] = []
            for request_id, raw in payload.items():
                if isinstance(raw, Mapping):
                    row = dict(raw)
                    row.setdefault("request_id", request_id)
                    keyed_rows.append(row)
            payload = keyed_rows
    result_rows = _sequence(payload, field="fine_results")
    output: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(result_rows):
        window = _mapping(raw, field=f"fine_results[{index}]")
        request_id_raw = window.get("request_id", window.get("refinement_request_id"))
        request_id = _window_id_to_request_id(request_id_raw)
        if request_id is None:
            request_id = _window_id_to_request_id(window.get("window_id"))
        if request_id is None or request_id not in requests:
            continue
        request = requests[request_id]
        action = _text(request.get("action_key"), field=f"fine_plan[{request_id}].action_key")
        score_row = _score_from_window(
            window,
            action=action,
            score_policy=str(fine_plan.get("score_policy", "top1")),
            min_camera_support=int(fine_plan.get("min_camera_support", 1)),
        )
        # Simple score rows may put the score directly on the row rather than
        # under a top-k proposal.
        if score_row is None and window.get("score") is not None:
            score = _finite(window.get("score"), field=f"fine_results[{index}].score")
            score = max(0.0, min(1.0, score))
            score_row = (score, {"score": score, "raw_score": score})
        if score_row is None:
            continue
        score, evidence = score_row
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
        }
    return output


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
) -> dict[str, Any]:
    """Resolve fine probe scores into parent-request ``MEASURED`` rows.

    The returned ``results`` are intentionally keyed by the original short
    request IDs generated by ``plan_wemm_temporal_refinement``.  This makes
    the result directly consumable by ``apply_refined_boundaries``.  For each
    role, only a threshold crossing on the expected before/after sides of the
    coarse anchor can become ``MEASURED``; all other cases are ``UNCERTAIN``.
    """

    parent_map = {
        str(row["request_id"]): row
        for row in _parent_requests(
            {"context_interval": fine_plan.get("source", {}).get("context_interval", {})},
            parent_plan,
        )
    }
    if not 0.0 <= float(start_threshold) <= 1.0:
        raise ProductionWemmTemporalScoreRefinementError("start_threshold must be between 0 and 1")
    if not 0.0 <= float(stop_threshold) <= 1.0:
        raise ProductionWemmTemporalScoreRefinementError("stop_threshold must be between 0 and 1")
    if float(stop_threshold) > float(start_threshold):
        raise ProductionWemmTemporalScoreRefinementError(
            "stop_threshold must be <= start_threshold"
        )
    if score_policy not in SCORE_POLICIES:
        raise ProductionWemmTemporalScoreRefinementError(
            f"score_policy must be one of {', '.join(SCORE_POLICIES)}"
        )
    support_min = _positive_int(min_camera_support, field="min_camera_support")
    min_resolution = _positive(
        min_boundary_resolution_seconds, field="min_boundary_resolution_seconds"
    )
    # Avoid depending on optional metadata in the plan; annotate rows before
    # extracting scores so a caller can pass a plain open-runner envelope.
    fine_plan_copy = dict(fine_plan)
    fine_plan_copy["score_policy"] = score_policy
    fine_plan_copy["min_camera_support"] = support_min
    scores = _normalise_score_rows(fine_plan_copy, fine_results)
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in scores.values():
        parent_id = row.get("parent_request_id")
        if isinstance(parent_id, str) and parent_id in parent_map:
            by_parent.setdefault(parent_id, []).append(row)

    output_rows: list[dict[str, Any]] = []
    measured_count = 0
    unresolved_count = 0
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
        pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
        for left, right in pairwise(rows):
            left_center = (float(left["start_seconds"]) + float(left["end_seconds"])) / 2.0
            right_center = (float(right["start_seconds"]) + float(right["end_seconds"])) / 2.0
            if right_center <= left_center + 1e-9:
                continue
            # A clipped side does not prove that the action was absent/present
            # outside the source.  Keep the parent unresolved rather than
            # converting an edge artifact into a measured boundary.
            if bool(left.get("edge_clipped")) or bool(right.get("edge_clipped")):
                continue
            threshold = float(start_threshold if role == "onset" else stop_threshold)
            valid = (
                float(left["score"]) < threshold <= float(right["score"])
                if role == "onset"
                else float(left["score"]) >= threshold > float(right["score"])
            )
            if valid:
                distance = right_center - left_center
                pairs.append((left, right, distance))
        if pairs:
            # Prefer the tightest bracket, then the finest level, then the
            # largest score jump.  This is deterministic across camera/order.
            left, right, distance = sorted(
                pairs,
                key=lambda item: (
                    # Prefer a crossing nearest the coarse anchor.  A
                    # transition can legitimately occur between two probes
                    # on the same side of that anchor (for example, when the
                    # coarse estimate is early/late), so requiring a literal
                    # before/after pair would discard valid evidence.
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
            threshold = float(start_threshold if role == "onset" else stop_threshold)
            boundary, confidence = _interpolated_boundary(left, right, threshold=threshold)
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
                            "reason": "FINE_SCORE_THRESHOLD_CROSSING",
                            "role": role,
                            "threshold": threshold,
                            "source_boundary_seconds": round(boundary, 6),
                            "boundary_resolution_seconds": round(distance / 2.0, 6),
                            "left_probe": _copy_json(left, field="evidence.left_probe"),
                            "right_probe": _copy_json(right, field="evidence.right_probe"),
                            "request_edges_are_not_boundaries": True,
                        },
                    }
                )
                continue
        unresolved_count += 1
        output_rows.append(
            {
                "request_id": parent_id,
                "status": "UNCERTAIN",
                "confidence": 0.0,
                "evidence": {
                    "reason": "NO_FINE_SCORE_THRESHOLD_CROSSING",
                    "role": role,
                    "anchor_seconds": anchor,
                    "observed_probe_count": len(rows),
                    "request_edges_are_not_boundaries": True,
                },
            }
        )
    output_rows.sort(key=lambda row: str(row["request_id"]))
    return {
        "format": RESULT_FORMAT,
        "authority": AUTHORITY,
        "status": RESULT_STATUS,
        "production_eligible": False,
        "score_policy": score_policy,
        "parameters": {
            "start_threshold": float(start_threshold),
            "stop_threshold": float(stop_threshold),
            "min_camera_support": support_min,
            "min_boundary_resolution_seconds": min_resolution,
        },
        "results": output_rows,
        "diagnostics": {
            "parent_request_count": len(parent_map),
            "fine_score_row_count": len(scores),
            "measured_result_count": measured_count,
            "unresolved_result_count": unresolved_count,
            "request_edges_used_as_boundaries": False,
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
            "The parent request interval is used only as a coordinate frame.",
        ],
    }


# Descriptive aliases for orchestration callers.
plan_temporal_score_refinement_grid = plan_wemm_score_refinement_grid
resolve_temporal_score_refinement = resolve_wemm_score_refinement


__all__ = [
    "AUTHORITY",
    "DEFAULT_LEVELS",
    "DEFAULT_MAX_REQUESTS",
    "DEFAULT_MIN_BOUNDARY_RESOLUTION_SECONDS",
    "DEFAULT_MIN_PROBE_SPAN_SECONDS",
    "DEFAULT_POINTS_PER_SIDE",
    "DEFAULT_PROBE_SPAN_SECONDS",
    "FORMAT",
    "RESULT_FORMAT",
    "RESULT_STATUS",
    "ROLES",
    "SCORE_POLICIES",
    "SIDES",
    "STATUS",
    "ProductionWemmTemporalScoreRefinementError",
    "plan_temporal_score_refinement_grid",
    "plan_wemm_score_refinement_grid",
    "resolve_temporal_score_refinement",
    "resolve_wemm_score_refinement",
]
