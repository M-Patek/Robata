"""Plan short, model-driven temporal refinement requests for WeMM proposals.

The dense WeMM temporal resolver intentionally works on bounded context
windows.  Its ``start_seconds``/``end_seconds`` values are therefore a
coarse *hypothesis*, not an action boundary.  This module is the next
inference seam: it turns the coarse transition diagnostics into much shorter
source-relative spans that a runner can submit for a second WeMM (or other
boundary-capable) pass.

This module does not decode media, invoke a model, or alter an annotation
envelope.  The returned object is a request plan only.  In particular,
``requires_model_recompute`` is always explicit and a short request's edges
must never be interpreted as the final onset/offset.  Keeping this seam
separate lets callers benchmark the planner and the eventual refinement model
without changing the historical four-second context route.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

FORMAT: Final = "robata-production-wemm-temporal-refinement-plan-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "REFINEMENT_REQUESTS_ONLY"
# Result projection is deliberately a separate format.  The original coarse
# report remains represented under ``segments`` when a completed second pass
# is applied.
APPLIED_FORMAT: Final = "robata-production-wemm-temporal-refinement-review-v1"
APPLIED_STATUS: Final = "REFINEMENT_REVIEW_ONLY"
PURPOSE: Final = "BOUNDARY_REFINEMENT"
# ``adaptive_score`` names the two-pass route in orchestration metadata.  The
# coarse resolver remains ``dense_score`` for compatibility; this module is
# the optional second pass that turns coarse transitions into short probes.
MODE_ADAPTIVE_SCORE: Final = "adaptive_score"
TIMESTAMP_BASIS: Final = "source_relative_seconds"
REQUEST_TIMESTAMP_BASIS: Final = "request_relative_seconds"
DEFAULT_REFINEMENT_SPAN_SECONDS: Final = 1.0
DEFAULT_MAX_REQUESTS: Final = 128
DEFAULT_MIN_REQUEST_SPAN_SECONDS: Final = 0.10
_REQUEST_EDGE_EPSILON: Final = 1e-9
ROLES: Final = ("onset", "offset")
RESULT_STATUSES: Final = ("MEASURED", "UNCERTAIN", "NONE_VISIBLE", "ABSTAIN", "INVALID")


class ProductionWemmTemporalRefinementError(ValueError):
    """Raised when a coarse temporal report cannot form a refinement plan."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmTemporalRefinementError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmTemporalRefinementError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionWemmTemporalRefinementError(f"{field} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ProductionWemmTemporalRefinementError(f"{field} must be non-empty")
    return result


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ProductionWemmTemporalRefinementError(f"{field} must be finite")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionWemmTemporalRefinementError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ProductionWemmTemporalRefinementError(f"{field} must be finite")
    return result


def _positive(value: object, *, field: str) -> float:
    result = _finite(value, field=field)
    if result <= 0.0:
        raise ProductionWemmTemporalRefinementError(f"{field} must be positive")
    return result


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionWemmTemporalRefinementError(f"{field} must be a positive integer")
    return value


def _copy_json(value: object, *, field: str) -> Any:
    """Copy small JSON-shaped diagnostics without deriving an identity/hash."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmTemporalRefinementError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(key): _copy_json(child, field=f"{field}.{key}") for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[]") for child in value]
    raise ProductionWemmTemporalRefinementError(f"{field} must be JSON-compatible")


def _bounds_from_report(report: Mapping[str, Any]) -> tuple[float, float]:
    """Find the source-relative interval that bounds all requests."""

    for field in ("context_interval", "window"):
        raw = report.get(field)
        if isinstance(raw, Mapping):
            start_raw = raw.get("start_seconds")
            end_raw = raw.get("end_seconds")
            if start_raw is not None and end_raw is not None:
                start = _finite(start_raw, field=f"{field}.start_seconds")
                end = _finite(end_raw, field=f"{field}.end_seconds")
                if start < 0.0 or end <= start:
                    raise ProductionWemmTemporalRefinementError(
                        f"{field} must satisfy 0 <= start < end"
                    )
                return start, end

    # A report produced by an older caller may omit the top-level interval.
    # Recover it from the model-estimated segments, but never from a fixed
    # context width or an assumed recording duration.
    starts: list[float] = []
    ends: list[float] = []
    segments_value = report.get("segments", ())
    if isinstance(segments_value, Sequence) and not isinstance(
        segments_value, (str, bytes, bytearray)
    ):
        for index, raw_segment in enumerate(segments_value):
            if not isinstance(raw_segment, Mapping):
                continue
            start_raw = raw_segment.get("start_seconds")
            end_raw = raw_segment.get("end_seconds")
            if start_raw is None or end_raw is None:
                continue
            start = _finite(start_raw, field=f"segments[{index}].start_seconds")
            end = _finite(end_raw, field=f"segments[{index}].end_seconds")
            if start >= 0.0 and end > start:
                starts.append(start)
                ends.append(end)
    if starts and ends:
        return min(starts), max(ends)
    raise ProductionWemmTemporalRefinementError(
        "coarse temporal report needs context_interval or a bounded segment interval"
    )


def _coarse_context_width(report: Mapping[str, Any]) -> float | None:
    """Return a recorded coarse context width when the report exposes one."""

    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    for key in ("context_grid", "probe_grid"):
        grid = diagnostics.get(key)
        if not isinstance(grid, Mapping):
            continue
        for width_key in ("context_width_seconds", "probe_width_seconds"):
            raw = grid.get(width_key)
            if raw is None:
                continue
            width = _finite(raw, field=f"diagnostics.{key}.{width_key}")
            if width > 0.0:
                return width
    return None


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return token[:64] or "action"


def _segment_action(segment: Mapping[str, Any], *, field: str) -> str:
    value = segment.get("action_key", segment.get("provisional_id"))
    return _text(value, field=field)


def _segment_id(segment: Mapping[str, Any], *, index: int) -> str:
    value = segment.get("segment_id", segment.get("proposal_id"))
    if value is None:
        return f"segment-{index:04d}"
    return _text(value, field=f"segments[{index}].segment_id")


def _transition(
    segment: Mapping[str, Any],
    *,
    role: str,
    segment_index: int,
) -> tuple[float, str, bool, str, float | None, dict[str, Any] | None]:
    """Extract one coarse transition while retaining only review metadata."""

    if role not in ROLES:
        raise ProductionWemmTemporalRefinementError(f"unsupported refinement role: {role}")
    start = _finite(segment.get("start_seconds"), field=f"segments[{segment_index}].start_seconds")
    end = _finite(segment.get("end_seconds"), field=f"segments[{segment_index}].end_seconds")
    if start < 0.0 or end <= start:
        raise ProductionWemmTemporalRefinementError(
            f"segments[{segment_index}] interval must satisfy 0 <= start < end"
        )
    fallback_anchor = start if role == "onset" else end
    diagnostics_value = segment.get("transition_diagnostics", {})
    diagnostics = diagnostics_value if isinstance(diagnostics_value, Mapping) else {}
    raw = diagnostics.get(role)
    transition = raw if isinstance(raw, Mapping) else {}
    anchor_raw = transition.get("boundary_seconds", fallback_anchor)
    anchor = _finite(anchor_raw, field=f"segments[{segment_index}].{role}.boundary_seconds")
    method_raw = transition.get("boundary_method", "coarse_segment_edge")
    method = _text(method_raw, field=f"segments[{segment_index}].{role}.boundary_method")
    crossed = transition.get("crossed_threshold") is True
    reason_raw = transition.get("reason", "COARSE_BOUNDARY")
    reason = _text(reason_raw, field=f"segments[{segment_index}].{role}.reason")
    confidence_raw = transition.get("confidence")
    confidence = (
        None
        if confidence_raw is None
        else _finite(
            confidence_raw,
            field=f"segments[{segment_index}].{role}.confidence",
        )
    )
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ProductionWemmTemporalRefinementError(
            f"segments[{segment_index}].{role}.confidence must be between 0 and 1"
        )
    copied = _copy_json(transition, field=f"segments[{segment_index}].{role}")
    if not isinstance(copied, dict):  # pragma: no cover - _copy_json invariant
        copied = None
    return anchor, method, crossed, reason, confidence, copied


def _short_span(
    *,
    anchor: float,
    source_start: float,
    source_end: float,
    span_seconds: float,
) -> tuple[float, float]:
    """Center a short span at ``anchor`` and shift it inward at source edges."""

    source_duration = source_end - source_start
    width = min(span_seconds, source_duration)
    half = width / 2.0
    # First clamp the anchor, then shift the interval so its width is stable
    # whenever the source is long enough.  This avoids edge requests that are
    # accidentally shorter merely because an onset is near t=0.
    center = max(source_start, min(anchor, source_end))
    start = center - half
    end = center + half
    if start < source_start:
        end += source_start - start
        start = source_start
    if end > source_end:
        start -= end - source_end
        end = source_end
    start = max(source_start, start)
    end = min(source_end, end)
    if end <= start:
        raise ProductionWemmTemporalRefinementError("refinement span collapsed at source edge")
    return float(start), float(end)


def plan_wemm_temporal_refinement(
    coarse_report: Mapping[str, Any],
    *,
    refinement_span_seconds: float = DEFAULT_REFINEMENT_SPAN_SECONDS,
    min_request_span_seconds: float = DEFAULT_MIN_REQUEST_SPAN_SECONDS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> dict[str, Any]:
    """Build a model-free short-probe plan from a dense temporal report.

    ``coarse_report`` is the output of
    :func:`robata.benchmark.production_wemm_temporal.resolve_wemm_temporal_segments`.
    One ONSET and one OFFSET request is emitted for every coarse segment.  A
    request is centered on the resolver's transition estimate and clipped to
    the report's source interval, but its edges are explicitly *not* action
    boundaries.  Duplicate action/role/span requests are coalesced while all
    contributing segment IDs remain visible.

    No model or media operation occurs here.  A downstream runner must use the
    returned spans as new inputs, then replace the pending request with a
    separately parsed boundary result.
    """

    report = _mapping(coarse_report, field="coarse_report")
    source_start, source_end = _bounds_from_report(report)
    span = _positive(refinement_span_seconds, field="refinement_span_seconds")
    minimum = _positive(min_request_span_seconds, field="min_request_span_seconds")
    if minimum > span:
        raise ProductionWemmTemporalRefinementError(
            "min_request_span_seconds must be <= refinement_span_seconds"
        )
    limit = _positive_int(max_requests, field="max_requests")
    context_width = _coarse_context_width(report)
    if context_width is not None and span >= context_width - 1e-9:
        raise ProductionWemmTemporalRefinementError(
            "refinement_span_seconds must be shorter than the recorded coarse context width"
        )

    raw_segments = _sequence(report.get("segments", ()), field="coarse_report.segments")
    requests_by_key: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    candidate_count = 0
    edge_count = 0
    unbracketed_count = 0
    for segment_index, raw_segment in enumerate(raw_segments):
        segment = _mapping(raw_segment, field=f"coarse_report.segments[{segment_index}]")
        action = _segment_action(segment, field=f"segments[{segment_index}].action_key")
        segment_id = _segment_id(segment, index=segment_index)
        source_window_ids_raw = segment.get("supporting_window_ids", ())
        source_window_ids = [
            _text(value, field=f"segments[{segment_index}].supporting_window_ids[]")
            for value in _sequence(
                source_window_ids_raw,
                field=f"segments[{segment_index}].supporting_window_ids",
            )
        ]
        # Validate the segment interval even though the transition helper
        # reads it again for its role-specific fallback anchor.
        _finite(segment.get("start_seconds"), field=f"segments[{segment_index}].start_seconds")
        _finite(segment.get("end_seconds"), field=f"segments[{segment_index}].end_seconds")
        segment_confidence_raw = segment.get("boundary_confidence")
        segment_confidence = (
            None
            if segment_confidence_raw is None
            else _finite(
                segment_confidence_raw,
                field=f"segments[{segment_index}].boundary_confidence",
            )
        )
        if segment_confidence is not None and not 0.0 <= segment_confidence <= 1.0:
            raise ProductionWemmTemporalRefinementError(
                f"segments[{segment_index}].boundary_confidence must be between 0 and 1"
            )

        for role in ROLES:
            candidate_count += 1
            anchor, method, crossed, reason, confidence, transition = _transition(
                segment,
                role=role,
                segment_index=segment_index,
            )
            request_start, request_end = _short_span(
                anchor=anchor,
                source_start=source_start,
                source_end=source_end,
                span_seconds=span,
            )
            if request_end - request_start < minimum - 1e-9:
                # This can happen only when the whole source is shorter than
                # the requested span.  Keep the request visible but mark it
                # ineligible rather than fabricating a positive interval.
                continue
            edge = reason.startswith("NO_") or method == "coarse_segment_edge"
            edge_count += edge
            unbracketed_count += not crossed
            # Rounded coordinates make deduplication stable across JSON
            # serialisation while retaining six decimals in the emitted span.
            key = (action, role, round(request_start, 6), round(request_end, 6))
            existing = requests_by_key.get(key)
            if existing is not None:
                existing["source_segment_ids"].append(segment_id)
                existing["source_window_ids"] = sorted(
                    set(existing["source_window_ids"]) | set(source_window_ids)
                )
                existing["coarse_transition_diagnostics"].append(transition)
                existing["coarse_boundary_crossed"] = bool(
                    existing["coarse_boundary_crossed"] or crossed
                )
                continue
            request_id = (f"{_slug(action)}-{role}-{request_start:.6f}-{request_end:.6f}").replace(
                ".", "p"
            )
            requests_by_key[key] = {
                "request_id": request_id,
                "action_key": action,
                "role": role,
                "purpose": PURPOSE,
                "start_seconds": round(request_start, 6),
                "end_seconds": round(request_end, 6),
                "timestamp_basis": TIMESTAMP_BASIS,
                "request_timestamp_basis": REQUEST_TIMESTAMP_BASIS,
                "interval_status": "SHORT_CONTEXT_REQUEST",
                "boundary_status": "PENDING_MODEL_RECOMPUTE",
                "requires_model_recompute": True,
                "coarse_anchor_seconds": anchor,
                "coarse_boundary_method": method,
                "coarse_boundary_crossed": crossed,
                "coarse_boundary_reason": reason,
                "coarse_boundary_confidence": confidence,
                "coarse_segment_boundary_confidence": segment_confidence,
                "source_segment_ids": [segment_id],
                "source_window_ids": sorted(set(source_window_ids)),
                "coarse_transition_diagnostics": [transition],
                "input_semantics": "SHORT_CONTEXT_AROUND_COARSE_TRANSITION",
                "model_output_contract": {
                    "coordinate_mode": REQUEST_TIMESTAMP_BASIS,
                    "must_not_copy_request_edges": True,
                    "report_smallest_visible_action_interval": True,
                    "allow_uncertain_or_no_boundary": True,
                },
                "status": "PENDING",
            }

    ordered = sorted(
        requests_by_key.values(),
        key=lambda row: (
            float(row["start_seconds"]),
            float(row["end_seconds"]),
            str(row["action_key"]),
            str(row["role"]),
        ),
    )
    truncated_count = max(0, len(ordered) - limit)
    requests = ordered[:limit]
    for row in requests:
        row["source_segment_ids"] = sorted(set(row["source_segment_ids"]))
        row["coarse_transition_diagnostics"] = [
            _copy_json(value, field="request.coarse_transition_diagnostics")
            for value in row["coarse_transition_diagnostics"]
        ]

    return {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": STATUS,
        "production_eligible": False,
        "source": {
            "coarse_temporal_format": report.get("format"),
            "coarse_temporal_status": report.get("status"),
            "timestamp_basis": TIMESTAMP_BASIS,
            "context_interval": {
                "start_seconds": source_start,
                "end_seconds": source_end,
                "context_only": True,
            },
        },
        "parameters": {
            "refinement_span_seconds": span,
            "min_request_span_seconds": minimum,
            "max_requests": limit,
            "coarse_context_width_seconds": context_width,
        },
        "requests": requests,
        "diagnostics": {
            "coarse_segment_count": len(raw_segments),
            "candidate_boundary_count": candidate_count,
            "request_count": len(requests),
            "deduplicated_request_count": max(0, candidate_count - len(requests_by_key)),
            "truncated_request_count": truncated_count,
            "edge_boundary_count": edge_count,
            "unbracketed_boundary_count": unbracketed_count,
            "shorter_than_coarse_context": (context_width is None or span < context_width - 1e-9),
            "coarse_report_unchanged": True,
        },
        "controls": {
            "media_decoded": False,
            "model_invoked": False,
            "runner_recompute_required": bool(requests),
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "qwen_invoked": False,
            "mage_invoked": False,
            "production_eligible": False,
        },
        "limitations": [
            "Requests are short model-input spans, not action boundaries.",
            "A downstream runner must decode and re-score each request before projection.",
            "The planner does not infer an onset/offset when coarse evidence is absent.",
            "Edge or unbracketed coarse transitions remain review-required.",
        ],
    }


def plan_wemm_temporal_refinement_from_windows(
    windows: Sequence[Mapping[str, Any]],
    *,
    refinement_span_seconds: float = DEFAULT_REFINEMENT_SPAN_SECONDS,
    min_request_span_seconds: float = DEFAULT_MIN_REQUEST_SPAN_SECONDS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    **resolver_kwargs: Any,
) -> dict[str, Any]:
    """Resolve coarse windows, then build a short-probe plan.

    This convenience wrapper still performs no media/model work.  It is kept
    explicit so callers can see that the refinement plan is derived from the
    existing dense score trajectory and does not alter the default runner.
    """

    from .production_wemm_temporal import resolve_wemm_temporal_segments

    coarse = resolve_wemm_temporal_segments(windows, **resolver_kwargs)
    return plan_wemm_temporal_refinement(
        coarse,
        refinement_span_seconds=refinement_span_seconds,
        min_request_span_seconds=min_request_span_seconds,
        max_requests=max_requests,
    )


def _normalise_result_rows(
    value: Mapping[str, Any] | Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    """Normalize keyed or row-oriented refinement results."""

    if isinstance(value, Mapping):
        nested = value.get("results")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
            value = nested
        else:
            rows: list[dict[str, Any]] = []
            for request_id, raw_result in value.items():
                result = _mapping(raw_result, field=f"results[{request_id}]")
                row = dict(result)
                row.setdefault("request_id", request_id)
                rows.append(row)
            return tuple(rows)
    rows_value = _sequence(value, field="results")
    return tuple(
        dict(_mapping(raw_result, field=f"results[{index}]"))
        for index, raw_result in enumerate(rows_value)
    )


def _normalise_result(
    raw_result: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one model result and map a request-relative interval to source time."""

    request_id = _text(raw_result.get("request_id"), field="result.request_id")
    if request_id != request["request_id"]:
        raise ProductionWemmTemporalRefinementError(
            f"result request_id {request_id!r} does not match requested {request['request_id']!r}"
        )
    status_raw = raw_result.get("status", raw_result.get("boundary_status"))
    status = _text(status_raw, field=f"results[{request_id}].status").upper()
    # The Qwen boundary parser uses PARSED as an envelope status and stores
    # MEASURED/UNCERTAIN under boundary_status.  Accept that shape without
    # allowing it to hide an unresolved result.
    if status == "PARSED":
        status = _text(
            raw_result.get("boundary_status"),
            field=f"results[{request_id}].boundary_status",
        ).upper()
    if status not in RESULT_STATUSES:
        raise ProductionWemmTemporalRefinementError(
            f"results[{request_id}].status must be one of {', '.join(RESULT_STATUSES)}"
        )
    confidence_raw = raw_result.get("confidence")
    confidence = (
        None
        if confidence_raw is None
        else _finite(
            confidence_raw,
            field=f"results[{request_id}].confidence",
        )
    )
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ProductionWemmTemporalRefinementError(
            f"results[{request_id}].confidence must be between 0 and 1"
        )
    evidence = _copy_json(raw_result.get("evidence", ""), field=f"results[{request_id}].evidence")
    start_raw = raw_result.get("start_seconds", raw_result.get("start_time_sec"))
    end_raw = raw_result.get("end_seconds", raw_result.get("end_time_sec"))
    if (start_raw is None) != (end_raw is None):
        raise ProductionWemmTemporalRefinementError(
            f"results[{request_id}] must provide both relative boundaries or neither"
        )
    if status == "MEASURED" and (start_raw is None or end_raw is None):
        raise ProductionWemmTemporalRefinementError(
            f"results[{request_id}] MEASURED status requires relative boundaries"
        )
    if status != "MEASURED" and (start_raw is not None or end_raw is not None):
        raise ProductionWemmTemporalRefinementError(
            f"results[{request_id}] unresolved status cannot carry boundaries"
        )

    relative_start: float | None = None
    relative_end: float | None = None
    source_start: float | None = None
    source_end: float | None = None
    request_edge_rejected = False
    if start_raw is not None and end_raw is not None:
        basis_raw = raw_result.get("timestamp_basis", raw_result.get("coordinate_mode"))
        if basis_raw is None:
            raise ProductionWemmTemporalRefinementError(
                f"results[{request_id}] must declare {REQUEST_TIMESTAMP_BASIS}"
            )
        basis = _text(basis_raw, field=f"results[{request_id}].timestamp_basis")
        if basis != REQUEST_TIMESTAMP_BASIS:
            raise ProductionWemmTemporalRefinementError(
                f"results[{request_id}] must use {REQUEST_TIMESTAMP_BASIS}"
            )
        relative_start = _finite(start_raw, field=f"results[{request_id}].start_seconds")
        relative_end = _finite(end_raw, field=f"results[{request_id}].end_seconds")
        request_start = _finite(
            request.get("start_seconds"), field=f"request[{request_id}].start_seconds"
        )
        request_end = _finite(
            request.get("end_seconds"), field=f"request[{request_id}].end_seconds"
        )
        request_duration = request_end - request_start
        if relative_start < 0.0 or relative_end <= relative_start:
            raise ProductionWemmTemporalRefinementError(
                f"results[{request_id}] interval must satisfy 0 <= start < end"
            )
        if relative_end > request_duration + 1e-9:
            raise ProductionWemmTemporalRefinementError(
                f"results[{request_id}] interval exceeds its request span"
            )
        source_start = request_start + relative_start
        source_end = request_start + relative_end

        # A refinement request is a bounded visual context, never an action
        # interval.  ``apply_refined_boundaries`` uses the leading edge of an
        # ONSET result and the trailing edge of an OFFSET result as the
        # proposed timestamps.  Do not let a model turn an unobserved request
        # edge into either timestamp merely by returning a full/clipped
        # context interval.  The non-relevant side may touch an edge because
        # it can be an uncertainty envelope; only the role's projected edge
        # is disallowed here.
        role = _text(request.get("role"), field=f"request[{request_id}].role").casefold()
        request_edge_rejected = (role == "onset" and relative_start <= _REQUEST_EDGE_EPSILON) or (
            role == "offset" and relative_end >= request_duration - _REQUEST_EDGE_EPSILON
        )
        if request_edge_rejected:
            projected_edge = "start_seconds" if role == "onset" else "end_seconds"
            evidence = {
                "reason": "REQUEST_EDGE_NOT_MODEL_SELECTED",
                "role": role,
                "projected_edge": projected_edge,
                "request_interval": {
                    "start_seconds": request_start,
                    "end_seconds": request_end,
                },
                "reported_request_relative_interval": {
                    "start_seconds": relative_start,
                    "end_seconds": relative_end,
                },
                "model_evidence": evidence,
                "must_not_copy_request_edges": True,
            }
            # Preserve the raw result/evidence, but make its normalized status
            # unresolved so it cannot be projected as a model-selected action
            # boundary.  This is intentionally not a hard run failure: a
            # review pack should retain the failed localization evidence.
            status = "UNCERTAIN"
            relative_start = None
            relative_end = None
            source_start = None
            source_end = None

    return {
        "request_id": request_id,
        "status": status,
        "timestamp_basis": REQUEST_TIMESTAMP_BASIS if relative_start is not None else None,
        "relative_start_seconds": relative_start,
        "relative_end_seconds": relative_end,
        "source_start_seconds": source_start,
        "source_end_seconds": source_end,
        "confidence": confidence,
        "evidence": evidence,
        "request_edge_rejected": request_edge_rejected,
        "raw": _copy_json(raw_result, field=f"results[{request_id}].raw"),
    }


def _best_boundary_result(
    candidates: Sequence[tuple[Mapping[str, Any], Mapping[str, Any] | None]],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None] | None:
    """Choose one deterministic candidate, preferring measured confidence."""

    if not candidates:
        return None

    def key(item: tuple[Mapping[str, Any], Mapping[str, Any] | None]) -> tuple[int, float, str]:
        request, result = item
        measured = int(isinstance(result, Mapping) and result.get("status") == "MEASURED")
        confidence = -1.0
        if isinstance(result, Mapping):
            raw_confidence = result.get("confidence")
            if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
                confidence = float(raw_confidence)
        return (-measured, -confidence, str(request["request_id"]))

    return sorted(candidates, key=key)[0]


def apply_refined_boundaries(
    coarse_report: Mapping[str, Any],
    refinement_plan: Mapping[str, Any],
    refinement_results: Mapping[str, Any] | Sequence[Any],
) -> dict[str, Any]:
    """Project completed short-probe results without replacing coarse evidence.

    ``refinement_results`` may be a mapping keyed by ``request_id`` or an
    array of rows.  Measured boundaries are interpreted in the explicit
    ``request_relative_seconds`` clock and converted to source-relative time
    using the request span.  For an ONSET request the projected boundary is
    the measured interval's start; for an OFFSET request it is its end.

    The returned copy retains the original coarse ``segments`` unchanged and
    adds ``temporal_refinement`` plus ``refined_segments``.  A complete pair
    is still review-only (``MODEL_REFINED`` and ``automatic_eligible=false``);
    unresolved, partial, and inverted pairs never fall back silently to a
    fabricated refined interval.
    """

    coarse = _mapping(coarse_report, field="coarse_report")
    plan = _mapping(refinement_plan, field="refinement_plan")
    if plan.get("format") != FORMAT:
        raise ProductionWemmTemporalRefinementError(f"refinement_plan.format must be {FORMAT!r}")
    source_start, source_end = _bounds_from_report(coarse)
    plan_requests_raw = _sequence(plan.get("requests", ()), field="refinement_plan.requests")
    request_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_request in enumerate(plan_requests_raw):
        request = dict(_mapping(raw_request, field=f"refinement_plan.requests[{index}]"))
        request_id = _text(request.get("request_id"), field=f"requests[{index}].request_id")
        if request_id in request_by_id:
            raise ProductionWemmTemporalRefinementError(
                f"duplicate refinement request: {request_id}"
            )
        role = _text(request.get("role"), field=f"requests[{index}].role").casefold()
        if role not in ROLES:
            raise ProductionWemmTemporalRefinementError(
                f"requests[{index}].role must be one of {', '.join(ROLES)}"
            )
        start = _finite(request.get("start_seconds"), field=f"requests[{index}].start_seconds")
        end = _finite(request.get("end_seconds"), field=f"requests[{index}].end_seconds")
        if start < source_start or end > source_end or end <= start:
            raise ProductionWemmTemporalRefinementError(
                f"requests[{index}] interval is outside the coarse source bounds"
            )
        request["role"] = role
        request["action_key"] = _text(
            request.get("action_key"), field=f"requests[{index}].action_key"
        )
        request["source_segment_ids"] = [
            _text(value, field=f"requests[{index}].source_segment_ids[]")
            for value in _sequence(
                request.get("source_segment_ids", ()),
                field=f"requests[{index}].source_segment_ids",
            )
        ]
        request_by_id[request_id] = request

    result_rows = _normalise_result_rows(refinement_results)
    results_by_id: dict[str, dict[str, Any]] = {}
    for raw_result in result_rows:
        request_id = _text(raw_result.get("request_id"), field="result.request_id")
        if request_id not in request_by_id:
            raise ProductionWemmTemporalRefinementError(
                f"refinement result references unknown request_id {request_id!r}"
            )
        if request_id in results_by_id:
            raise ProductionWemmTemporalRefinementError(
                f"duplicate refinement result: {request_id}"
            )
        results_by_id[request_id] = _normalise_result(
            raw_result,
            request=request_by_id[request_id],
        )

    segment_rows = _sequence(coarse.get("segments", ()), field="coarse_report.segments")
    refined_segments: list[dict[str, Any]] = []
    complete_count = partial_count = unresolved_count = invalid_pair_count = 0
    for segment_index, raw_segment in enumerate(segment_rows):
        segment = dict(_mapping(raw_segment, field=f"coarse_report.segments[{segment_index}]"))
        segment_id = _segment_id(segment, index=segment_index)
        action = _segment_action(segment, field=f"segments[{segment_index}].action_key")
        segment_start = _finite(
            segment.get("start_seconds"), field=f"segments[{segment_index}].start_seconds"
        )
        segment_end = _finite(
            segment.get("end_seconds"), field=f"segments[{segment_index}].end_seconds"
        )
        candidates_by_role: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]] = {
            role: [] for role in ROLES
        }
        for request_id, request in request_by_id.items():
            if request["action_key"] != action or segment_id not in request["source_segment_ids"]:
                continue
            candidates_by_role[request["role"]].append((request, results_by_id.get(request_id)))
        chosen: dict[str, tuple[Mapping[str, Any], Mapping[str, Any] | None] | None] = {
            role: _best_boundary_result(candidates_by_role[role]) for role in ROLES
        }

        onset_choice = chosen["onset"]
        offset_choice = chosen["offset"]
        onset_result = onset_choice[1] if onset_choice is not None else None
        offset_result = offset_choice[1] if offset_choice is not None else None
        onset_boundary = (
            float(onset_result["source_start_seconds"])
            if isinstance(onset_result, Mapping)
            and onset_result.get("status") == "MEASURED"
            and isinstance(onset_result.get("source_start_seconds"), (int, float))
            else None
        )
        offset_boundary = (
            float(offset_result["source_end_seconds"])
            if isinstance(offset_result, Mapping)
            and offset_result.get("status") == "MEASURED"
            and isinstance(offset_result.get("source_end_seconds"), (int, float))
            else None
        )
        refined_interval: dict[str, Any] | None = None
        if onset_boundary is not None and offset_boundary is not None:
            if offset_boundary > onset_boundary:
                refined_interval = {
                    "start_seconds": onset_boundary,
                    "end_seconds": offset_boundary,
                    "status": "MEASURED",
                }
                refinement_status = "REFINED"
                complete_count += 1
            else:
                refinement_status = "INVALID_PAIR"
                invalid_pair_count += 1
        elif onset_boundary is not None or offset_boundary is not None:
            refinement_status = "PARTIAL"
            partial_count += 1
        else:
            refinement_status = "UNRESOLVED"
            unresolved_count += 1

        refined_value = _copy_json(segment, field=f"segments[{segment_index}]")
        if not isinstance(refined_value, dict):  # pragma: no cover - _copy_json invariant
            raise ProductionWemmTemporalRefinementError("segment copy must be an object")
        refined = refined_value
        refined.update(
            {
                "coarse_interval": {
                    "start_seconds": segment_start,
                    "end_seconds": segment_end,
                    "status": "MODEL_PROBE_BOUND",
                    "context_only": True,
                    "is_action_boundary": False,
                    "action_boundary": False,
                },
                "start_seconds": onset_boundary if refined_interval is not None else None,
                "end_seconds": offset_boundary if refined_interval is not None else None,
                "boundary_status": (
                    "MODEL_REFINED" if refined_interval is not None else "MODEL_REFINEMENT_PENDING"
                ),
                "boundary_source": "wemm_short_refinement",
                "boundary_method": "short_probe_model",
                "production_eligible": False,
                # A refined row is still model evidence for review.  These
                # explicit markers prevent a short-probe span from being
                # promoted as an action boundary by a generic reader.
                "context_only": True,
                "window_context_only": True,
                "is_action_boundary": False,
                "action_boundary": False,
                "refined_interval": refined_interval,
                "refinement_status": refinement_status,
                "onset_request_id": (
                    onset_choice[0]["request_id"] if onset_choice is not None else None
                ),
                "offset_request_id": (
                    offset_choice[0]["request_id"] if offset_choice is not None else None
                ),
                "onset_result": _copy_json(onset_result, field="refined.onset_result"),
                "offset_result": _copy_json(offset_result, field="refined.offset_result"),
                "review_required": True,
                "automatic_eligible": False,
                "decision": "pending",
            }
        )
        refined_segments.append(refined)

    normalised_results = [results_by_id[key] for key in sorted(results_by_id)]
    request_edge_rejected_count = sum(
        result.get("request_edge_rejected") is True for result in normalised_results
    )
    projection = {
        "format": APPLIED_FORMAT,
        "authority": AUTHORITY,
        "status": APPLIED_STATUS,
        "production_eligible": False,
        "timestamp_basis": TIMESTAMP_BASIS,
        "request_timestamp_basis": REQUEST_TIMESTAMP_BASIS,
        "source": {
            "coarse_temporal_format": coarse.get("format"),
            "refinement_plan_format": plan.get("format"),
            "context_interval": {
                "start_seconds": source_start,
                "end_seconds": source_end,
                "context_only": True,
                "is_action_boundary": False,
                "action_boundary": False,
            },
        },
        "requests": [
            _copy_json(request, field="projection.requests[]") for request in request_by_id.values()
        ],
        "results": normalised_results,
        "refined_segments": refined_segments,
        "diagnostics": {
            "request_count": len(request_by_id),
            "result_count": len(results_by_id),
            "missing_result_count": len(request_by_id) - len(results_by_id),
            "refined_segment_count": complete_count,
            "partial_segment_count": partial_count,
            "unresolved_segment_count": unresolved_count,
            "invalid_pair_count": invalid_pair_count,
            "request_edge_rejected_result_count": request_edge_rejected_count,
            "coarse_segments_preserved": True,
        },
        "controls": {
            "media_decoded": False,
            "model_invoked": False,
            "refinement_results_supplied": bool(results_by_id),
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "production_eligible": False,
        },
        "limitations": [
            "Refined intervals remain model evidence for review, not production gold.",
            "The coarse segments are retained and are never overwritten by this helper.",
            "Missing or unresolved role results cannot produce a refined interval.",
        ],
    }
    output_value = _copy_json(coarse, field="coarse_report")
    if not isinstance(output_value, dict):  # pragma: no cover - _copy_json invariant
        raise ProductionWemmTemporalRefinementError("coarse report copy must be an object")
    output_value["temporal_refinement"] = projection
    output_value["refined_segments"] = _copy_json(refined_segments, field="refined_segments")
    return output_value


# Descriptive aliases used by orchestration code.  Keep the longer function
# name as the implementation-level API so existing callers remain explicit.
plan_temporal_refinement = plan_wemm_temporal_refinement
project_refined_boundaries = apply_refined_boundaries


__all__ = [
    "APPLIED_FORMAT",
    "APPLIED_STATUS",
    "AUTHORITY",
    "DEFAULT_MAX_REQUESTS",
    "DEFAULT_MIN_REQUEST_SPAN_SECONDS",
    "DEFAULT_REFINEMENT_SPAN_SECONDS",
    "FORMAT",
    "MODE_ADAPTIVE_SCORE",
    "PURPOSE",
    "REQUEST_TIMESTAMP_BASIS",
    "RESULT_STATUSES",
    "ROLES",
    "STATUS",
    "TIMESTAMP_BASIS",
    "ProductionWemmTemporalRefinementError",
    "apply_refined_boundaries",
    "plan_temporal_refinement",
    "plan_wemm_temporal_refinement",
    "plan_wemm_temporal_refinement_from_windows",
    "project_refined_boundaries",
]
