"""Resolve dense WeMM context rankings into review-only action intervals.

WeMM receives bounded video contexts because the encoder needs finite visual
input.  Those contexts are not action annotations.  This adapter converts the
per-context Top-K ranking stream into a score trajectory for each provisional
production phrase, then delegates onset/offset selection to the small temporal
resolver.  It neither invents a production ontology nor asks Qwen/Mage to
describe the clip.

The adapter intentionally retains the original contexts, Top-K rows and score
trajectory.  A produced segment is ``MODEL_PROBE_BOUND`` and review-only: it
is a model estimate, never a copied context boundary or human/gold interval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from statistics import fmean, median
from typing import Any, Final

from .production_wemm_interval_proposal import (
    AUTHORITY,
    BOUNDARY_SOURCE,
    BOUNDARY_STATUS,
    DEFAULT_MERGE_GAP_SECONDS,
    DEFAULT_MIN_DURATION_SECONDS,
    DEFAULT_START_THRESHOLD,
    DEFAULT_STOP_THRESHOLD,
    ProductionWemmIntervalProposalError,
    propose_model_intervals,
)

FORMAT: Final = "robata-production-wemm-temporal-resolver-v1"
STATUS: Final = "PROPOSALS_ONLY"
MODE_NONE: Final = "none"
MODE_DENSE_SCORE: Final = "dense_score"
# ``adaptive_score`` is a two-pass orchestration mode.  Its coarse resolver
# remains the same dense score trajectory used by ``dense_score``; the runner
# adds a short-context refinement pass as an additive review sidecar.  Keeping
# the mode token here lets callers validate configuration without importing the
# refinement planner (which intentionally imports this module lazily to avoid a
# cycle).
MODE_ADAPTIVE_SCORE: Final = "adaptive_score"
TEMPORAL_MODES: Final = (MODE_NONE, MODE_DENSE_SCORE, MODE_ADAPTIVE_SCORE)
SCORE_POLICY_ABSOLUTE: Final = "absolute"
SCORE_POLICY_TOP1: Final = "top1"
SCORE_POLICY_WINNER_STABLE: Final = "winner_stable"
SCORE_POLICY_RELATIVE_MARGIN: Final = "relative_margin"
SCORE_POLICIES: Final = (
    SCORE_POLICY_ABSOLUTE,
    SCORE_POLICY_TOP1,
    SCORE_POLICY_WINNER_STABLE,
    SCORE_POLICY_RELATIVE_MARGIN,
)
DEFAULT_SCORE_POLICY: Final = SCORE_POLICY_TOP1
# Public callers may use descriptive spellings for the score policy.  Keep
# these aliases at the temporal contract boundary so the open and batch
# runners, the dense resolver, and any persisted checkpoint all resolve to
# one canonical wire value.  ``margin`` is intentionally *not* an alias: it
# is ambiguous between absolute and candidate-relative scores and must remain
# rejected rather than silently selecting a policy.
SCORE_POLICY_ALIASES: Final = {
    "raw": SCORE_POLICY_ABSOLUTE,
    "winner": SCORE_POLICY_TOP1,
    "stable": SCORE_POLICY_WINNER_STABLE,
    "winner_stability": SCORE_POLICY_WINNER_STABLE,
    "candidate_relative": SCORE_POLICY_RELATIVE_MARGIN,
    "relative": SCORE_POLICY_RELATIVE_MARGIN,
    "contrast": SCORE_POLICY_RELATIVE_MARGIN,
}
# WeMM cosine similarities in the production artifacts are tightly clustered.
# A signed candidate-vs-runner-up margin is therefore converted to a bounded
# confidence with a small logistic scale instead of applying the unusable raw
# 0.65/0.50 similarity thresholds directly.  The value is an explicit,
# reproducible experiment parameter and can be overridden by callers.
DEFAULT_RELATIVE_MARGIN_SCALE: Final = 0.02


class ProductionWemmTemporalError(ValueError):
    """Raised when a WeMM context ranking cannot form a temporal track."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmTemporalError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmTemporalError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionWemmTemporalError(f"{field} must be a non-empty string")
    return value.strip()


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionWemmTemporalError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ProductionWemmTemporalError(f"{field} must be finite")
    return result


def _unit_score(value: object, *, field: str) -> float:
    score = _finite(value, field=field)
    if not 0.0 <= score <= 1.0:
        raise ProductionWemmTemporalError(f"{field} must be between 0 and 1")
    return score


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionWemmTemporalError(f"{field} must be a positive integer")
    return value


def normalize_score_policy(value: object, *, field: str = "score_policy") -> str:
    """Return the canonical temporal score policy for a caller value.

    Normalization is intentionally shared by the public runners and the
    resolver.  It accepts surrounding whitespace, case differences, and
    hyphenated spellings, then maps only the documented descriptive aliases.
    Unknown values (including ``"margin"``) raise the temporal contract
    error instead of falling through to a policy-specific implementation.
    """

    if not isinstance(value, str):
        raise ProductionWemmTemporalError(f"{field} must be one of {', '.join(SCORE_POLICIES)}")
    policy = value.strip().casefold().replace("-", "_")
    policy = SCORE_POLICY_ALIASES.get(policy, policy)
    if policy not in SCORE_POLICIES:
        raise ProductionWemmTemporalError(f"{field} must be one of {', '.join(SCORE_POLICIES)}")
    return policy


def _score_policy(value: object, *, field: str = "score_policy") -> str:
    """Backward-compatible private wrapper for the shared policy normalizer."""

    return normalize_score_policy(value, field=field)


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionWemmTemporalError(f"{field} must be a non-negative integer")
    return value


def _json_copy(value: object, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmTemporalError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_copy(child, field=f"{field}.{key}") for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_copy(child, field=f"{field}[]") for child in value]
    raise ProductionWemmTemporalError(f"{field} must be JSON-compatible")


def _camera_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    # Normalized pre-annotation candidates keep producer-only fields under
    # ``raw``.  Flatten that sidecar for temporal resolution while retaining
    # explicit top-level values when present.
    effective = _effective_candidate(row)
    evidence = effective.get("evidence", [])
    result: set[str] = set()
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        for value in evidence:
            if isinstance(value, Mapping):
                camera = value.get("camera_id")
                if isinstance(camera, str) and camera.strip():
                    result.add(camera.strip())
    camera_id = effective.get("camera_id")
    if isinstance(camera_id, str) and camera_id.strip():
        result.add(camera_id.strip())
    camera_ids = effective.get("camera_ids")
    if isinstance(camera_ids, Sequence) and not isinstance(camera_ids, (str, bytes, bytearray)):
        for value in camera_ids:
            if isinstance(value, str) and value.strip():
                result.add(value.strip())
    support = effective.get("camera_support")
    if isinstance(support, Sequence) and not isinstance(support, (str, bytes, bytearray)):
        for value in support:
            if isinstance(value, str) and value.strip():
                result.add(value.strip())
    return tuple(sorted(result))


_CANDIDATE_FALLBACK_FIELDS: Final = (
    "provisional_id",
    "action_key",
    "label_text",
    "label_variant",
    "structured_labels",
    "rank",
    "score",
    "visual_score",
    "camera_id",
    "camera_support",
    "camera_coverage",
    "camera_ids",
    "camera_evidence",
    "evidence",
)


def _effective_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Expose producer fields retained under a normalized candidate's raw sidecar.

    ``build_preannotation_envelope`` deliberately keeps the opaque action
    identity and model-specific provenance in ``candidate.raw``.  Temporal
    resolution can also be called on that normalized envelope, so use the
    sidecar as a fallback without mutating the input or changing its wire
    shape.
    """

    result = dict(row)
    raw = row.get("raw")
    if isinstance(raw, Mapping):
        for key in _CANDIDATE_FALLBACK_FIELDS:
            if (key not in result or result[key] is None) and key in raw:
                result[key] = raw[key]
        if not result.get("evidence") and isinstance(raw.get("camera_evidence"), Sequence):
            result["evidence"] = raw["camera_evidence"]
    return result


def _camera_support_count(row: Mapping[str, Any]) -> int:
    effective = _effective_candidate(row)
    ids = _camera_ids(effective)
    if ids:
        return len(ids)
    raw = effective.get("camera_support")
    if raw is None:
        raw = effective.get("camera_coverage")
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int) and raw >= 0:
        return raw
    if isinstance(raw, float) and math.isfinite(raw) and raw >= 0 and raw.is_integer():
        return int(raw)
    # A fused ranking with no explicit coverage data is still a genuine fused
    # candidate.  Treat it as one supporting view rather than manufacturing a
    # six-camera observation.
    return 1


def _candidate_action(row: Mapping[str, Any], *, field: str) -> str:
    effective = _effective_candidate(row)
    return _text(effective.get("provisional_id", effective.get("action_key")), field=field)


def _candidate_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    row = _effective_candidate(row)
    return {
        key: _json_copy(row[key], field=f"candidate.{key}")
        for key in (
            "provisional_id",
            "label_text",
            "label_variant",
            "structured_labels",
            "rank",
            "score",
            "camera_support",
            "evidence",
            "raw",
        )
        if key in row
    }


def _candidate_rank(row: Mapping[str, Any], *, field: str) -> int:
    raw_rank = row.get("rank", 1)
    return _positive_int(raw_rank, field=f"{field}.rank")


def _context_grid_metadata(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe the temporal resolution available to the score trajectory.

    A WeMM score is produced for a bounded context, not for an instantaneous
    frame.  Keeping the context width, centre reference, and probe spacing in
    the sidecar prevents reviewers from mistaking a midpoint estimate for a
    frame-accurate timestamp.  ``None`` is used when a spacing cannot be
    inferred from a single context or an irregular grid.
    """

    spans = sorted(
        {
            (
                float(context["start_seconds"]),
                float(context["end_seconds"]),
            )
            for context in contexts
        }
    )
    widths = [end - start for start, end in spans if end > start]
    starts = [start for start, _end in spans]
    centres = [(start + end) / 2.0 for start, end in spans]
    start_deltas = [right - left for left, right in pairwise(starts) if right > left]
    centre_deltas = [right - left for left, right in pairwise(centres) if right > left]

    def _summary(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "max": None, "median": None}
        return {
            "min": float(min(values)),
            "max": float(max(values)),
            "median": float(median(values)),
        }

    width_summary = _summary(widths)
    centre_summary = _summary(centre_deltas)
    start_summary = _summary(start_deltas)
    return {
        "context_count": len(spans),
        "context_width_seconds": (
            float(widths[0])
            if widths and all(abs(value - widths[0]) <= 1e-9 for value in widths)
            else None
        ),
        "context_width_seconds_summary": width_summary,
        "probe_start_spacing_seconds": (
            float(start_deltas[0])
            if start_deltas and all(abs(value - start_deltas[0]) <= 1e-9 for value in start_deltas)
            else None
        ),
        "probe_start_spacing_seconds_summary": start_summary,
        "probe_center_spacing_seconds": (
            float(centre_deltas[0])
            if centre_deltas
            and all(abs(value - centre_deltas[0]) <= 1e-9 for value in centre_deltas)
            else None
        ),
        "probe_center_spacing_seconds_summary": centre_summary,
        "score_reference": "context_center",
        "context_center_latency_seconds": (float(fmean(widths) / 2.0) if widths else None),
        "estimated_boundary_resolution_seconds": (
            float(median(centre_deltas)) if centre_deltas else None
        ),
        "edge_boundary_policy": "observed_probe_span_when_neighbour_missing",
    }


def _winner_action(
    candidates: Mapping[str, Mapping[str, Any]], *, min_camera_support: int
) -> str | None:
    """Select the best camera-supported winner from a context's ranked Top-K.

    A lower-ranked candidate with sufficient multi-camera evidence must not be
    suppressed merely because an unsupported rank-1 row exists.  Filtering the
    winner before applying the ``top1`` temporal policy keeps camera-support
    gating and winner gating consistent.
    """

    eligible = {
        action: candidate
        for action, candidate in candidates.items()
        if _non_negative_int(
            candidate.get("camera_support_count", 0),
            field=f"candidate[{action}].camera_support_count",
        )
        >= min_camera_support
    }
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda action: (
            _candidate_rank(eligible[action], field=f"candidate[{action}]"),
            -float(eligible[action]["score"]),
            action,
        ),
    )


def _stabilize_winner_sequence(
    winners: Sequence[str | None],
) -> tuple[str | None, ...]:
    """Suppress one-context winner glitches without inventing persistence.

    A dense ranking stream can briefly select a neighbouring action (or have
    no eligible winner) even though the surrounding contexts agree.  The
    bounded ``winner_stable`` policy repairs only an interior singleton
    ``A, B, A``/``A, None, A`` run.  The pass is deliberately non-cascading:
    decisions use the *raw* neighbours, so a two-context excursion such as
    ``A, B, B, A`` remains visible.  Sequence edges are never filled because
    there is no evidence on both sides.
    """

    raw = tuple(winners)
    if len(raw) < 3:
        return raw
    stabilized = list(raw)
    for index in range(1, len(raw) - 1):
        left = raw[index - 1]
        right = raw[index + 1]
        if left is not None and left == right and raw[index] != left:
            stabilized[index] = left
    return tuple(stabilized)


def _eligible_candidate_score(
    context: Mapping[str, Any],
    action: str,
    *,
    min_camera_support: int,
) -> tuple[float, tuple[str, ...]] | None:
    """Return one action's eligible score and camera provenance, if present."""

    candidates = _mapping(context["candidates"], field="context.candidates")
    raw_candidate = candidates.get(action)
    if not isinstance(raw_candidate, Mapping):
        return None
    support = _non_negative_int(
        raw_candidate.get("camera_support_count", 0),
        field=f"candidate[{action}].camera_support_count",
    )
    if support < min_camera_support:
        return None
    score = _unit_score(raw_candidate.get("score"), field=f"candidate[{action}].score")
    camera_ids = tuple(
        value
        for value in raw_candidate.get("camera_ids", ())
        if isinstance(value, str) and value.strip()
    )
    return score, camera_ids


def _relative_margin_observation(
    candidates: Mapping[str, Mapping[str, Any]],
    action: str,
    *,
    min_camera_support: int,
    scale: float,
    min_target_score: float,
) -> dict[str, Any] | None:
    """Return a candidate-relative confidence observation.

    The production WeMM similarities are often high for several neighbouring
    phrases at once.  Comparing an action with the strongest *eligible*
    competitor removes that shared scene/verb component and leaves a signed
    margin that is useful for temporal transitions.  Missing or unsupported
    candidates are not treated as visual negatives: no relative observation is
    emitted unless both the target and a competitor have camera-supported
    scores.

    ``confidence`` is a logistic projection of the signed margin.  It is only
    a bounded temporal support score, not a calibrated probability.  The raw
    target/runner-up scores and signed margin are returned for review.
    """

    target = candidates.get(action)
    if not isinstance(target, Mapping):
        return None
    target_support = _non_negative_int(
        target.get("camera_support_count", 0),
        field=f"candidate[{action}].camera_support_count",
    )
    if target_support < min_camera_support:
        return None
    target_score = _unit_score(target.get("score"), field=f"candidate[{action}].score")
    competitors: list[tuple[str, Mapping[str, Any], float]] = []
    for competitor_action, raw in candidates.items():
        if competitor_action == action or not isinstance(raw, Mapping):
            continue
        support = _non_negative_int(
            raw.get("camera_support_count", 0),
            field=f"candidate[{competitor_action}].camera_support_count",
        )
        if support < min_camera_support:
            continue
        score = _unit_score(raw.get("score"), field=f"candidate[{competitor_action}].score")
        competitors.append((str(competitor_action), raw, score))
    if not competitors:
        return None
    # Scores are already ranked by the producer, but use score/rank/action
    # explicitly so normalized envelopes and lightweight fixtures behave
    # deterministically even when list order is not preserved.
    runner_action, runner, runner_score = sorted(
        competitors,
        key=lambda item: (
            -item[2],
            int(item[1].get("rank", 1))
            if isinstance(item[1].get("rank", 1), int)
            and not isinstance(item[1].get("rank", 1), bool)
            else 1,
            item[0],
        ),
    )[0]
    margin = float(target_score - runner_score)
    # Avoid overflow in exp for malformed/extreme scales while preserving
    # exact 0.5 at a zero margin.
    z = max(-60.0, min(60.0, margin / scale))
    confidence = 1.0 / (1.0 + math.exp(-z))
    target_camera_ids = tuple(
        value for value in target.get("camera_ids", ()) if isinstance(value, str) and value.strip()
    )
    runner_camera_ids = tuple(
        value for value in runner.get("camera_ids", ()) if isinstance(value, str) and value.strip()
    )
    camera_overlap_count = len(set(target_camera_ids) & set(runner_camera_ids))
    # ``camera_sets_compatible`` intentionally remains permissive for the
    # score stream: older fused artifacts may retain only a numeric support
    # count and no camera IDs.  That is enough to expose a relative score for
    # review, but it is not enough to certify that a rank switch is the sole
    # explanation for a temporal transition.  Keep this separate provenance
    # bit so the adaptive suppression guard can fail closed without changing
    # legacy score-policy behavior.
    camera_provenance_known = bool(target_camera_ids) and bool(runner_camera_ids)
    camera_sets_compatible = (
        not target_camera_ids or not runner_camera_ids or camera_overlap_count > 0
    )
    return {
        "target_action": action,
        "target_score": float(target_score),
        "target_rank": target.get("rank"),
        "target_camera_support_count": target_support,
        "target_camera_ids": list(target_camera_ids),
        "runner_up_action": runner_action,
        "runner_up_score": float(runner_score),
        "runner_up_rank": runner.get("rank"),
        "runner_up_camera_support_count": _non_negative_int(
            runner.get("camera_support_count", 0),
            field=f"candidate[{runner_action}].camera_support_count",
        ),
        "runner_up_camera_ids": list(runner_camera_ids),
        "camera_overlap_count": camera_overlap_count,
        "camera_provenance_known": camera_provenance_known,
        # Compatibility alias used by older review consumers.  Keep both
        # spellings in the observation so persisted sidecars can be compared
        # without a schema migration.
        "camera_sets_known": camera_provenance_known,
        "camera_sets_compatible": camera_sets_compatible,
        "signed_margin": margin,
        "relative_margin_scale": float(scale),
        "target_score_floor": float(min_target_score),
        "target_score_floor_passed": bool(target_score >= min_target_score - 1e-9),
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _neighbor_carried_score(
    contexts: Sequence[Mapping[str, Any]],
    index: int,
    action: str,
    *,
    min_camera_support: int,
) -> tuple[float, tuple[str, ...]] | None:
    """Carry a conservative score across one stabilized missing context.

    Both immediate neighbours must contain an eligible candidate.  The lower
    neighbour score is used so stability cannot raise confidence above the
    visual evidence actually observed.  Camera IDs are intentionally not
    returned for the missing context; callers must keep provenance attached to
    the real neighbouring rows only.
    """

    if index <= 0 or index >= len(contexts) - 1:
        return None
    previous = _eligible_candidate_score(
        contexts[index - 1], action, min_camera_support=min_camera_support
    )
    following = _eligible_candidate_score(
        contexts[index + 1], action, min_camera_support=min_camera_support
    )
    if previous is None or following is None:
        return None
    return min(previous[0], following[0]), (
        str(contexts[index - 1]["window_id"]),
        str(contexts[index + 1]["window_id"]),
    )


def _relative_margin_switch_boundary_evidence(
    *,
    action: str,
    side: str,
    transition: Mapping[str, Any],
    probe_metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    min_camera_support: int,
    start_threshold: float,
    stop_threshold: float,
) -> dict[str, Any] | None:
    """Return conservative evidence for a relative-margin rank switch.

    A relative margin is useful for routing, but it is not an action-vs-
    background measurement.  If the target's raw similarity remains above the
    absolute resolver threshold on both sides and only the ranked winner (and
    therefore the margin confidence) changes, the transition must not be
    promoted to a model-bound interval.  Unknown camera provenance, target
    floor state, or missing margin fields deliberately fail closed and leave
    the normal proposal untouched for human review.
    """

    if side not in {"onset", "offset"} or transition.get("crossed_threshold") is not True:
        return None
    active_probe = transition.get("active_probe")
    neighbouring_probe = transition.get("neighbouring_probe")
    if not isinstance(active_probe, Mapping) or not isinstance(neighbouring_probe, Mapping):
        return None

    def _window_ids(probe: Mapping[str, Any]) -> tuple[str, ...]:
        raw_ids = probe.get("window_ids", probe.get("window_id"))
        if isinstance(raw_ids, str) and raw_ids.strip():
            return (raw_ids.strip(),)
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
            return tuple(
                value.strip() for value in raw_ids if isinstance(value, str) and value.strip()
            )
        return ()

    active_window_ids = _window_ids(active_probe)
    neighbouring_window_ids = _window_ids(neighbouring_probe)
    if not active_window_ids or not neighbouring_window_ids:
        return None

    def _rows(window_ids: Sequence[str]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        result: list[tuple[str, Mapping[str, Any]]] = []
        for window_id in window_ids:
            row = probe_metadata.get((action, window_id))
            if not isinstance(row, Mapping):
                return ()
            result.append((window_id, row))
        return tuple(result)

    active_rows = _rows(active_window_ids)
    neighbouring_rows = _rows(neighbouring_window_ids)
    if not active_rows or not neighbouring_rows:
        return None

    threshold = start_threshold if side == "onset" else stop_threshold

    def _number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    def _valid(row: Mapping[str, Any]) -> bool:
        raw_score = _number(row.get("raw_score"))
        effective_score = _number(row.get("effective_score"))
        margin = _number(row.get("signed_margin"))
        support = row.get("camera_support_count")
        camera_sets_known = row.get("relative_margin_camera_sets_known")
        if camera_sets_known is None:
            camera_sets_known = row.get("relative_margin_camera_provenance_known")
        return (
            row.get("score_source") == "relative_margin"
            and isinstance(support, int)
            and not isinstance(support, bool)
            and support >= min_camera_support
            and row.get("relative_margin_target_floor_passed") is True
            and camera_sets_known is True
            and row.get("relative_margin_camera_sets_compatible") is True
            and raw_score is not None
            and raw_score >= threshold - 1e-9
            and effective_score is not None
            and margin is not None
            and isinstance(row.get("raw_winner_action"), str)
            and bool(row.get("raw_winner_action"))
            and isinstance(row.get("runner_up_action"), str)
            and bool(row.get("runner_up_action"))
            and row.get("runner_up_action") != action
        )

    if not all(_valid(row) for _window_id, row in active_rows):
        return None
    if not all(_valid(row) for _window_id, row in neighbouring_rows):
        return None

    active_scores = [_number(row.get("effective_score")) for _window_id, row in active_rows]
    neighbouring_scores = [
        _number(row.get("effective_score")) for _window_id, row in neighbouring_rows
    ]
    if any(value is None or value < threshold - 1e-9 for value in active_scores):
        return None
    if any(value is None or value >= threshold - 1e-9 for value in neighbouring_scores):
        return None

    active_winners = [row.get("raw_winner_action") for _window_id, row in active_rows]
    neighbouring_winners = [row.get("raw_winner_action") for _window_id, row in neighbouring_rows]
    if any(value != action for value in active_winners):
        return None
    if any(value == action for value in neighbouring_winners):
        return None

    active_margins = [_number(row.get("signed_margin")) for _window_id, row in active_rows]
    neighbouring_margins = [
        _number(row.get("signed_margin")) for _window_id, row in neighbouring_rows
    ]
    if any(value is None for value in active_margins + neighbouring_margins):
        return None
    # The guard above establishes that every margin is numeric, but mypy
    # cannot narrow values held in two separately-built lists.  Materialize
    # narrowed lists before comparing them so the runtime contract and the
    # static type both make that invariant explicit.
    active_margin_values = [value for value in active_margins if value is not None]
    neighbouring_margin_values = [value for value in neighbouring_margins if value is not None]
    # The target-vs-runner margin must move in the direction that caused the
    # effective confidence drop.  This avoids suppressing a malformed or
    # non-monotone transition that merely happens to straddle the threshold.
    if min(active_margin_values) <= max(neighbouring_margin_values):
        return None

    def _evidence(window_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "window_id": window_id,
            "raw_score": _number(row.get("raw_score")),
            "effective_score": _number(row.get("effective_score")),
            "signed_margin": _number(row.get("signed_margin")),
            "runner_up_action": row.get("runner_up_action"),
            "runner_up_score": _number(row.get("runner_up_score")),
            "runner_up_camera_support_count": row.get("runner_up_camera_support_count"),
            "runner_up_camera_ids": list(row.get("runner_up_camera_ids", ())),
            "raw_winner_action": row.get("raw_winner_action"),
            "camera_support_count": row.get("camera_support_count"),
            "camera_overlap_count": row.get("relative_margin_camera_overlap_count"),
            "camera_provenance_known": row.get("relative_margin_camera_provenance_known"),
            "camera_sets_known": row.get(
                "relative_margin_camera_sets_known",
                row.get("relative_margin_camera_provenance_known"),
            ),
            "camera_sets_compatible": row.get("relative_margin_camera_sets_compatible"),
            "score_source": row.get("score_source"),
        }

    return {
        "side": side,
        "action_key": action,
        "reason": "RANKING_SWITCH_ONLY",
        "boundary_status": "UNRESOLVED",
        "threshold": float(threshold),
        "active_window_ids": list(active_window_ids),
        "neighbouring_window_ids": list(neighbouring_window_ids),
        "active_evidence": [_evidence(window_id, row) for window_id, row in active_rows],
        "neighbouring_evidence": [
            _evidence(window_id, row) for window_id, row in neighbouring_rows
        ],
        "transition": _json_copy(transition, field="ranking_switch.transition"),
    }


def _ranking_switch_boundary_evidence(
    *,
    action: str,
    side: str,
    transition: Mapping[str, Any],
    probe_metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    policy: str,
    min_camera_support: int,
    start_threshold: float,
    stop_threshold: float,
) -> dict[str, Any] | None:
    """Return evidence for a boundary caused only by a winner switch.

    ``top1`` intentionally gates non-winning candidates to zero.  That is a
    useful compatibility policy for the dense route, but a rank change alone
    must not become an adaptive action boundary.  This helper is deliberately
    conservative: it only reports a switch when both sides contain an
    eligible candidate whose *raw* score remains above the relevant hysteresis
    threshold, while the effective score crossed solely because the winner
    changed.  Missing or camera-unsupported candidates are therefore left
    unresolved by the normal trajectory rather than being labelled a ranking
    switch.
    """

    if policy == SCORE_POLICY_RELATIVE_MARGIN:
        return _relative_margin_switch_boundary_evidence(
            action=action,
            side=side,
            transition=transition,
            probe_metadata=probe_metadata,
            min_camera_support=min_camera_support,
            start_threshold=start_threshold,
            stop_threshold=stop_threshold,
        )
    if policy not in {SCORE_POLICY_TOP1, SCORE_POLICY_WINNER_STABLE}:
        return None
    if side not in {"onset", "offset"} or transition.get("crossed_threshold") is not True:
        return None

    active_probe = transition.get("active_probe")
    neighbouring_probe = transition.get("neighbouring_probe")
    if not isinstance(active_probe, Mapping) or not isinstance(neighbouring_probe, Mapping):
        # A sequence edge cannot establish a winner switch because one side of
        # the transition has no observed probe.
        return None

    def _window_ids(probe: Mapping[str, Any]) -> tuple[str, ...]:
        raw_ids = probe.get("window_ids", probe.get("window_id"))
        if isinstance(raw_ids, str) and raw_ids.strip():
            return (raw_ids.strip(),)
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
            return tuple(
                str(value).strip() for value in raw_ids if isinstance(value, str) and value.strip()
            )
        return ()

    active_window_ids = _window_ids(active_probe)
    neighbouring_window_ids = _window_ids(neighbouring_probe)
    if not active_window_ids or not neighbouring_window_ids:
        return None

    def _metadata_rows(window_ids: Sequence[str]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        rows: list[tuple[str, Mapping[str, Any]]] = []
        for window_id in window_ids:
            row = probe_metadata.get((action, window_id))
            if not isinstance(row, Mapping):
                return ()
            rows.append((window_id, row))
        return tuple(rows)

    active_rows = _metadata_rows(active_window_ids)
    neighbouring_rows = _metadata_rows(neighbouring_window_ids)
    if not active_rows or not neighbouring_rows:
        return None

    threshold = start_threshold if side == "onset" else stop_threshold

    def _score(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    def _winner(row: Mapping[str, Any]) -> str | None:
        # ``stabilized_winner_action`` is the effective winner for the
        # winner_stable policy and equals the raw winner for top1.  Fall back
        # to the raw field for reports produced before that metadata existed.
        value = row.get("stabilized_winner_action")
        if not isinstance(value, str) or not value.strip():
            value = row.get("raw_winner_action")
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _eligible(row: Mapping[str, Any]) -> bool:
        support = row.get("camera_support_count")
        rank = row.get("rank")
        raw_score = _score(row.get("raw_score"))
        effective_score = _score(row.get("effective_score"))
        return (
            isinstance(support, int)
            and not isinstance(support, bool)
            and support >= min_camera_support
            and isinstance(rank, int)
            and not isinstance(rank, bool)
            and rank > 0
            and raw_score is not None
            and effective_score is not None
            and raw_score >= threshold - 1e-9
        )

    # The active side must be supported and above threshold in both raw and
    # effective streams.  The neighbouring side must remain raw-high but be
    # effective-low, which is precisely the top1 rank gate we are auditing.
    if not all(_eligible(row) for _window_id, row in active_rows):
        return None
    if not all(_eligible(row) for _window_id, row in neighbouring_rows):
        return None
    active_scores = [_score(row.get("effective_score")) for _window_id, row in active_rows]
    neighbouring_scores = [
        _score(row.get("effective_score")) for _window_id, row in neighbouring_rows
    ]
    if any(value is None or value < threshold - 1e-9 for value in active_scores):
        return None
    if any(value is None or value >= threshold - 1e-9 for value in neighbouring_scores):
        return None

    active_winners = [_winner(row) for _window_id, row in active_rows]
    neighbouring_winners = [_winner(row) for _window_id, row in neighbouring_rows]
    if any(value != action for value in active_winners):
        return None
    if any(value is None or value == action for value in neighbouring_winners):
        return None

    def _evidence(window_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "window_id": window_id,
            "raw_score": _score(row.get("raw_score")),
            "effective_score": _score(row.get("effective_score")),
            "rank": row.get("rank"),
            "camera_support_count": row.get("camera_support_count"),
            "raw_winner_action": row.get("raw_winner_action"),
            "stabilized_winner_action": row.get("stabilized_winner_action"),
            "score_source": row.get("score_source"),
        }

    return {
        "side": side,
        "action_key": action,
        "reason": "RANKING_SWITCH_ONLY",
        "boundary_status": "UNRESOLVED",
        "threshold": float(threshold),
        "active_window_ids": list(active_window_ids),
        "neighbouring_window_ids": list(neighbouring_window_ids),
        "active_evidence": [_evidence(window_id, row) for window_id, row in active_rows],
        "neighbouring_evidence": [
            _evidence(window_id, row) for window_id, row in neighbouring_rows
        ],
        "transition": _json_copy(transition, field="ranking_switch.transition"),
    }


def _context_rows(windows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_window in enumerate(windows):
        window = _mapping(raw_window, field=f"windows[{index}]")
        window_id = _text(window.get("window_id"), field=f"windows[{index}].window_id")
        if window_id in seen:
            raise ProductionWemmTemporalError(f"duplicate window_id: {window_id}")
        seen.add(window_id)
        # Persisted pre-annotation windows store their source context under
        # ``source_interval``; raw open-runner windows expose the same values
        # at the top level.  Accept both shapes without treating either as an
        # action boundary.
        interval = window.get("source_interval")
        interval_mapping = interval if isinstance(interval, Mapping) else {}
        start_value = window.get("start_seconds")
        end_value = window.get("end_seconds")
        if start_value is None:
            start_value = interval_mapping.get("start_seconds")
        if end_value is None:
            end_value = interval_mapping.get("end_seconds")
        start = _finite(start_value, field=f"{window_id}.start_seconds")
        end = _finite(end_value, field=f"{window_id}.end_seconds")
        if start < 0.0 or end <= start:
            raise ProductionWemmTemporalError(f"invalid context interval for {window_id}")
        proposals = _sequence(window.get("proposals", []), field=f"{window_id}.proposals")
        if len(proposals) > 1:
            raise ProductionWemmTemporalError(
                f"{window_id} must contain at most one fused retrieval proposal"
            )
        top_k: Sequence[Any] = ()
        if proposals:
            proposal = _mapping(proposals[0], field=f"{window_id}.proposals[0]")
            top_k = _sequence(proposal.get("top_k", []), field=f"{window_id}.top_k")
        candidates: dict[str, dict[str, Any]] = {}
        top_k_snapshots: list[dict[str, Any]] = []
        for candidate_index, raw_candidate in enumerate(top_k):
            candidate = _mapping(raw_candidate, field=f"{window_id}.top_k[{candidate_index}]")
            effective_candidate = _effective_candidate(candidate)
            action = _candidate_action(candidate, field=f"{window_id}.top_k[{candidate_index}]")
            if action in candidates:
                raise ProductionWemmTemporalError(
                    f"duplicate provisional candidate {action!r} in {window_id}"
                )
            rank = _candidate_rank(
                effective_candidate,
                field=f"{window_id}.top_k[{candidate_index}]",
            )
            snapshot = _candidate_snapshot(effective_candidate)
            # Rank is inferred from list position when a producer omitted it;
            # preserve the effective value in the temporal evidence either way.
            snapshot["rank"] = rank
            top_k_snapshots.append(snapshot)
            candidates[action] = {
                "rank": rank,
                "score": _unit_score(
                    effective_candidate.get("score"),
                    field=f"{window_id}.{action}.score",
                ),
                "camera_support_count": _camera_support_count(effective_candidate),
                "camera_ids": _camera_ids(effective_candidate),
                "snapshot": snapshot,
            }
        contexts.append(
            {
                "window_id": window_id,
                "start_seconds": start,
                "end_seconds": end,
                "candidates": candidates,
                "top_k": top_k_snapshots,
            }
        )
    if not contexts:
        raise ProductionWemmTemporalError("at least one context window is required")
    contexts.sort(
        key=lambda row: (
            float(row["start_seconds"]),
            float(row["end_seconds"]),
            row["window_id"],
        )
    )
    return tuple(contexts)


def resolve_wemm_temporal_segments(
    windows: Sequence[Mapping[str, Any]],
    *,
    start_threshold: float = DEFAULT_START_THRESHOLD,
    stop_threshold: float = DEFAULT_STOP_THRESHOLD,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS,
    min_camera_support: int = 1,
    boundary_mode: str = "midpoint",
    score_policy: str = DEFAULT_SCORE_POLICY,
    suppress_ranking_switch_boundaries: bool = False,
    relative_margin_scale: float = DEFAULT_RELATIVE_MARGIN_SCALE,
    relative_margin_min_target_score: float = 0.60,
) -> dict[str, Any]:
    """Resolve Top-K context rankings into action interval proposals.

    Every action seen in any context receives a score for every selected
    context.  ``score_policy='top1'`` (the dense-mode default) lets only the
    deterministic Top-K winner contribute temporal support; this prevents the
    globally high, tightly packed WeMM similarities from opening one track
    per phrase.  ``score_policy='absolute'`` retains the historical behavior
    for controlled comparisons.  ``score_policy='winner_stable'`` is an
    opt-in, one-context de-chatter pass over the winner sequence; it can carry
    a conservative neighbour-minimum score across one missing middle row but
    never invents camera evidence.  A missing Top-K entry is recorded as zero
    *ranking support*, not as visual proof that the action is impossible.

    ``relative_margin`` compares each candidate with the strongest
    camera-supported competitor and maps the signed margin through a logistic
    projection.  This policy is intended for the tightly clustered WeMM
    similarities seen in production artifacts; ``relative_margin_scale`` is
    recorded as an experiment parameter and is not a probability calibration.

    ``suppress_ranking_switch_boundaries`` is an opt-in guard for the adaptive
    route.  When enabled, a proposal whose onset or offset is explained only
    by a Top-K winner switch (the raw candidate remains above the relevant
    hysteresis threshold) is omitted from ``segments`` and retained as an
    unresolved diagnostic.  The default is ``False`` so legacy dense/top1
    output remains unchanged.
    """

    support_min = _positive_int(min_camera_support, field="min_camera_support")
    policy = _score_policy(score_policy)
    if not isinstance(suppress_ranking_switch_boundaries, bool):
        raise ProductionWemmTemporalError("suppress_ranking_switch_boundaries must be boolean")
    if isinstance(relative_margin_scale, bool) or not isinstance(
        relative_margin_scale, (int, float)
    ):
        raise ProductionWemmTemporalError("relative_margin_scale must be positive and finite")
    relative_margin_scale = float(relative_margin_scale)
    if not math.isfinite(relative_margin_scale) or relative_margin_scale <= 0.0:
        raise ProductionWemmTemporalError("relative_margin_scale must be positive and finite")
    if isinstance(relative_margin_min_target_score, bool) or not isinstance(
        relative_margin_min_target_score, (int, float)
    ):
        raise ProductionWemmTemporalError(
            "relative_margin_min_target_score must be between 0 and 1"
        )
    relative_margin_min_target_score = float(relative_margin_min_target_score)
    if not math.isfinite(relative_margin_min_target_score) or not (
        0.0 <= relative_margin_min_target_score <= 1.0
    ):
        raise ProductionWemmTemporalError(
            "relative_margin_min_target_score must be between 0 and 1"
        )
    ranking_switch_suppression_active = suppress_ranking_switch_boundaries and policy in {
        SCORE_POLICY_TOP1,
        SCORE_POLICY_WINNER_STABLE,
        SCORE_POLICY_RELATIVE_MARGIN,
    }
    contexts = _context_rows(windows)
    source_start = min(float(context["start_seconds"]) for context in contexts)
    source_end = max(float(context["end_seconds"]) for context in contexts)
    action_metadata: dict[str, dict[str, Any]] = {}
    action_count_by_context: list[int] = []
    all_actions: set[str] = set()
    for context in contexts:
        candidates = _mapping(context["candidates"], field="context.candidates")
        action_count_by_context.append(len(candidates))
        for action, raw_candidate in candidates.items():
            candidate = _mapping(raw_candidate, field="context.candidate")
            all_actions.add(action)
            action_metadata.setdefault(
                action,
                dict(_mapping(candidate["snapshot"], field="snapshot")),
            )

    # Compute the context winner stream once, before expanding action rows.
    # Keeping the raw and effective sequences separate makes the opt-in
    # stability policy auditable and leaves the historical ``top1`` route
    # byte-for-byte equivalent in its score decisions.
    raw_winners = tuple(
        _winner_action(
            _mapping(context["candidates"], field="context.candidates"),
            min_camera_support=support_min,
        )
        for context in contexts
    )
    stabilized_winners = (
        _stabilize_winner_sequence(raw_winners)
        if policy == SCORE_POLICY_WINNER_STABLE
        else raw_winners
    )
    winner_sequence = [
        {
            "window_id": str(context["window_id"]),
            "raw_winner": raw_winners[index],
            "stabilized_winner": stabilized_winners[index],
            "stabilized": raw_winners[index] != stabilized_winners[index],
        }
        for index, context in enumerate(contexts)
    ]
    raw_winner_switch_count = sum(int(left != right) for left, right in pairwise(raw_winners))
    stabilized_winner_switch_count = sum(
        int(left != right) for left, right in pairwise(stabilized_winners)
    )
    winner_stabilization_count = sum(
        int(raw != stabilized)
        for raw, stabilized in zip(raw_winners, stabilized_winners, strict=True)
    )
    if not all_actions:
        return {
            "format": FORMAT,
            "authority": AUTHORITY,
            "status": STATUS,
            "production_eligible": False,
            "official_quality_status": "NOT_MEASURED",
            "official_gold_status": "NOT_ESTABLISHED",
            "mode": MODE_DENSE_SCORE,
            "context_interval": {
                "start_seconds": source_start,
                "end_seconds": source_end,
                "context_only": True,
                "is_action_boundary": False,
                "action_boundary": False,
            },
            "segments": [],
            "score_trajectories": [],
            "parameters": {
                "start_threshold": start_threshold,
                "stop_threshold": stop_threshold,
                "merge_gap_seconds": merge_gap_seconds,
                "min_duration_seconds": min_duration_seconds,
                "min_camera_support": support_min,
                "boundary_mode": boundary_mode,
                "score_policy": policy,
                "relative_margin_scale": relative_margin_scale,
                "relative_margin_min_target_score": relative_margin_min_target_score,
                "suppress_ranking_switch_boundaries": suppress_ranking_switch_boundaries,
                "ranking_switch_suppression_active": ranking_switch_suppression_active,
            },
            "diagnostics": {
                "context_window_count": len(contexts),
                "candidate_action_count": 0,
                "temporal_probe_count": 0,
                "segment_count": 0,
                "context_grid": _context_grid_metadata(contexts),
                "fused_trajectory_camera_provenance": (
                    "source_camera_ids_from_top_k_candidate_evidence"
                ),
                "raw_winner_sequence": list(raw_winners),
                "stabilized_winner_sequence": list(stabilized_winners),
                "winner_sequence": winner_sequence,
                "raw_winner_switch_count": raw_winner_switch_count,
                "stabilized_winner_switch_count": stabilized_winner_switch_count,
                "winner_stabilization_count": winner_stabilization_count,
                "score_carry_count": 0,
                "score_imputation_count": 0,
                "ranking_switch_suppression_enabled": suppress_ranking_switch_boundaries,
                "ranking_switch_suppression_active": ranking_switch_suppression_active,
                "ranking_switch_unresolved_count": 0,
                "ranking_switch_unresolved_segments": [],
                "ranking_switch_unresolved_proposals": [],
            },
            "controls": {
                "model_invoked": False,
                "gold_read": False,
                "gold_written": False,
                "top_k_retained": True,
                "missing_top_k_recorded_as_zero": True,
                "winner_stability_applied": policy == SCORE_POLICY_WINNER_STABLE,
                "relative_margin_applied": policy == SCORE_POLICY_RELATIVE_MARGIN,
                "ranking_switch_suppression_applied": ranking_switch_suppression_active,
            },
        }

    probes: list[dict[str, Any]] = []
    candidate_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    probe_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for context_index, context in enumerate(contexts):
        window_id = _text(context["window_id"], field="context.window_id")
        candidates = _mapping(context["candidates"], field=f"{window_id}.candidates")
        raw_winner = raw_winners[context_index]
        winner = stabilized_winners[context_index]
        for action in sorted(all_actions):
            candidate_value = candidates.get(action)
            candidate_for_action = candidate_value if isinstance(candidate_value, Mapping) else None
            score_carried = False
            score_imputed = False
            carry_source_window_ids: tuple[str, ...] = ()
            score_source = "missing"
            relative_observation: dict[str, Any] | None = None
            if candidate_for_action is None:
                raw_score = 0.0
                rank: int | None = None
                support = 0
                effective_score = 0.0
                camera_ids: tuple[str, ...] = ()
                snapshot: Mapping[str, Any] | None = None
                if (
                    policy == SCORE_POLICY_WINNER_STABLE
                    and action == winner
                    and action != raw_winner
                ):
                    carried = _neighbor_carried_score(
                        contexts,
                        context_index,
                        action,
                        min_camera_support=support_min,
                    )
                    if carried is not None:
                        effective_score = carried[0]
                        carry_source_window_ids = carried[1]
                        score_carried = True
                        score_imputed = True
                        score_source = "neighbor_min"
            else:
                support = _non_negative_int(
                    candidate_for_action["camera_support_count"],
                    field=f"{window_id}.{action}.support",
                )
                raw_score = float(candidate_for_action["score"])
                rank = _positive_int(
                    candidate_for_action.get("rank", 1),
                    field=f"{window_id}.{action}.rank",
                )
                camera_ids = tuple(
                    value
                    for value in candidate_for_action.get("camera_ids", ())
                    if isinstance(value, str) and value.strip()
                )
                snapshot_value = candidate_for_action.get("snapshot")
                snapshot = snapshot_value if isinstance(snapshot_value, Mapping) else None
                if support < support_min:
                    effective_score = 0.0
                elif policy == SCORE_POLICY_TOP1:
                    # Similarity scores are globally high and tightly packed
                    # in the current WeMM artifacts.  Let only the ranked
                    # winner for a context contribute temporal support; all
                    # other candidates stay available in Top-K for review.
                    effective_score = raw_score if action == winner else 0.0
                    score_source = "candidate" if effective_score else "gated"
                elif policy == SCORE_POLICY_WINNER_STABLE:
                    effective_score = raw_score if action == winner else 0.0
                    score_source = "candidate" if effective_score else "gated"
                elif policy == SCORE_POLICY_RELATIVE_MARGIN:
                    relative_observation = _relative_margin_observation(
                        candidates,
                        action,
                        min_camera_support=support_min,
                        scale=relative_margin_scale,
                        min_target_score=relative_margin_min_target_score,
                    )
                    if relative_observation is None:
                        effective_score = 0.0
                        score_source = "relative_missing"
                    elif not relative_observation["camera_sets_compatible"]:
                        effective_score = 0.0
                        score_source = "relative_camera_mismatch"
                    elif not relative_observation["target_score_floor_passed"]:
                        effective_score = 0.0
                        score_source = "relative_target_below_floor"
                    else:
                        effective_score = float(relative_observation["confidence"])
                        score_source = "relative_margin"
                else:
                    effective_score = raw_score
                    score_source = "candidate"
            if candidate_for_action is not None and support >= support_min:
                candidate_rows[(action, window_id)] = candidate_for_action
            probe_metadata[(action, window_id)] = {
                "raw_score": raw_score,
                "effective_score": effective_score,
                "rank": rank,
                # ``winner`` remains the effective winner flag.  Explicit raw
                # and stabilized fields make any opt-in correction auditable.
                "winner": action == winner,
                "raw_winner": action == raw_winner,
                "stabilized_winner": action == winner,
                "raw_winner_action": raw_winner,
                "stabilized_winner_action": winner,
                "winner_stabilized": raw_winner != winner,
                "camera_support_count": support,
                "camera_ids": list(camera_ids),
                "snapshot": snapshot,
                "score_carried": score_carried,
                "score_imputed": score_imputed,
                "score_source": score_source,
                "carry_source_window_ids": list(carry_source_window_ids),
                "runner_up_action": (
                    relative_observation.get("runner_up_action")
                    if relative_observation is not None
                    else None
                ),
                "runner_up_score": (
                    relative_observation.get("runner_up_score")
                    if relative_observation is not None
                    else None
                ),
                "runner_up_camera_support_count": (
                    relative_observation.get("runner_up_camera_support_count")
                    if relative_observation is not None
                    else None
                ),
                "runner_up_camera_ids": (
                    list(relative_observation.get("runner_up_camera_ids", ()))
                    if relative_observation is not None
                    else []
                ),
                "signed_margin": (
                    relative_observation.get("signed_margin")
                    if relative_observation is not None
                    else None
                ),
                "relative_margin_confidence": (
                    relative_observation.get("confidence")
                    if relative_observation is not None
                    else None
                ),
                "relative_margin_scale": relative_margin_scale,
                "relative_margin_min_target_score": relative_margin_min_target_score,
                "relative_margin_target_floor_passed": (
                    relative_observation.get("target_score_floor_passed")
                    if relative_observation is not None
                    else None
                ),
                "relative_margin_camera_overlap_count": (
                    relative_observation.get("camera_overlap_count")
                    if relative_observation is not None
                    else None
                ),
                "relative_margin_camera_provenance_known": (
                    relative_observation.get("camera_provenance_known")
                    if relative_observation is not None
                    else None
                ),
                "relative_margin_camera_sets_known": (
                    relative_observation.get("camera_sets_known")
                    if relative_observation is not None
                    else None
                ),
                "relative_margin_camera_sets_compatible": (
                    relative_observation.get("camera_sets_compatible")
                    if relative_observation is not None
                    else None
                ),
            }
            probes.append(
                {
                    "action_key": action,
                    "camera_id": "__fused__",
                    "window_id": window_id,
                    "start_seconds": context["start_seconds"],
                    "end_seconds": context["end_seconds"],
                    "score": effective_score,
                }
            )

    score_carry_count = sum(
        int(metadata.get("score_carried") is True) for metadata in probe_metadata.values()
    )
    score_imputation_count = sum(
        int(metadata.get("score_imputed") is True) for metadata in probe_metadata.values()
    )

    try:
        resolved = propose_model_intervals(
            probes,
            window_start_seconds=source_start,
            window_end_seconds=source_end,
            start_threshold=start_threshold,
            stop_threshold=stop_threshold,
            merge_gap_seconds=merge_gap_seconds,
            min_duration_seconds=min_duration_seconds,
            min_camera_support=1,
            camera_fusion="mean",
            boundary_mode=boundary_mode,
        )
    except ProductionWemmIntervalProposalError as exc:
        raise ProductionWemmTemporalError(
            f"could not resolve temporal score tracks: {exc}"
        ) from exc

    # Add the untransformed retrieval score and rank metadata to each resolved
    # trajectory.  The resolver's ``score`` remains the effective temporal
    # score so threshold behavior is reproducible, while reviewers can still
    # inspect the raw WeMM similarity that produced it.
    resolved_trajectories = _json_copy(
        resolved.get("score_trajectories", []), field="score_trajectories"
    )
    if isinstance(resolved_trajectories, list):
        for trajectory in resolved_trajectories:
            if not isinstance(trajectory, dict):
                continue
            trajectory_action = trajectory.get("action_key")
            raw_probes = trajectory.get("probes", [])
            if not isinstance(trajectory_action, str) or not isinstance(raw_probes, list):
                continue
            for probe in raw_probes:
                if not isinstance(probe, dict):
                    continue
                raw_window_ids = probe.get("window_ids", probe.get("window_id"))
                if isinstance(raw_window_ids, list):
                    window_ids = [value for value in raw_window_ids if isinstance(value, str)]
                elif isinstance(raw_window_ids, str):
                    window_ids = [raw_window_ids]
                else:
                    window_ids = []
                metadata_rows = [
                    probe_metadata[(trajectory_action, window_id)]
                    for window_id in window_ids
                    if (trajectory_action, window_id) in probe_metadata
                ]
                if not metadata_rows:
                    continue
                metadata = metadata_rows[0]
                source_camera_ids = sorted(
                    {
                        camera_id
                        for metadata_row in metadata_rows
                        for camera_id in metadata_row.get("camera_ids", [])
                        if isinstance(camera_id, str) and camera_id.strip()
                    }
                )
                source_camera_support_count = max(
                    [
                        len(source_camera_ids),
                        *[
                            int(metadata_row.get("camera_support_count", 0))
                            for metadata_row in metadata_rows
                            if isinstance(metadata_row.get("camera_support_count", 0), int)
                            and not isinstance(metadata_row.get("camera_support_count", 0), bool)
                        ],
                    ]
                )
                carried_source_window_ids = sorted(
                    {
                        source_window_id
                        for metadata_row in metadata_rows
                        for source_window_id in metadata_row.get("carry_source_window_ids", [])
                        if isinstance(source_window_id, str) and source_window_id.strip()
                    }
                )
                score_carried = any(
                    metadata_row.get("score_carried") is True for metadata_row in metadata_rows
                )
                score_imputed = any(
                    metadata_row.get("score_imputed") is True for metadata_row in metadata_rows
                )
                probe.update(
                    {
                        "camera_id": "__fused__",
                        "raw_score": metadata["raw_score"],
                        "effective_score": metadata["effective_score"],
                        "score_policy": policy,
                        "rank": metadata["rank"],
                        "winner_for_context": metadata["winner"],
                        "raw_winner_for_context": metadata["raw_winner"],
                        "stabilized_winner_for_context": metadata["stabilized_winner"],
                        "raw_winner_action": metadata["raw_winner_action"],
                        "stabilized_winner_action": metadata["stabilized_winner_action"],
                        "winner_stabilized": metadata["winner_stabilized"],
                        "score_carried": score_carried,
                        "score_imputed": score_imputed,
                        "score_source": metadata["score_source"],
                        "carry_source_window_ids": carried_source_window_ids,
                        "runner_up_action": metadata.get("runner_up_action"),
                        "runner_up_score": metadata.get("runner_up_score"),
                        "runner_up_camera_support_count": metadata.get(
                            "runner_up_camera_support_count"
                        ),
                        "runner_up_camera_ids": list(metadata.get("runner_up_camera_ids", ())),
                        "signed_margin": metadata.get("signed_margin"),
                        "relative_margin_confidence": metadata.get("relative_margin_confidence"),
                        "relative_margin_scale": metadata.get(
                            "relative_margin_scale", relative_margin_scale
                        ),
                        "relative_margin_min_target_score": metadata.get(
                            "relative_margin_min_target_score", relative_margin_min_target_score
                        ),
                        "relative_margin_target_floor_passed": metadata.get(
                            "relative_margin_target_floor_passed"
                        ),
                        "relative_margin_camera_overlap_count": metadata.get(
                            "relative_margin_camera_overlap_count"
                        ),
                        "relative_margin_camera_provenance_known": metadata.get(
                            "relative_margin_camera_provenance_known"
                        ),
                        "relative_margin_camera_sets_known": metadata.get(
                            "relative_margin_camera_sets_known"
                        ),
                        "relative_margin_camera_sets_compatible": metadata.get(
                            "relative_margin_camera_sets_compatible"
                        ),
                        # ``camera_id`` remains ``__fused__`` to make the
                        # score stream's fusion stage explicit.  These fields
                        # preserve the real camera provenance used by the
                        # candidate evidence and avoid reporting a fused
                        # probe as one-camera support.
                        "source_camera_ids": source_camera_ids,
                        "source_camera_support_count": source_camera_support_count,
                        "source_camera_support_eligible": source_camera_support_count
                        >= support_min,
                        "camera_ids": source_camera_ids,
                        "camera_support_count": source_camera_support_count,
                        "camera_support_eligible": source_camera_support_count >= support_min,
                    }
                )

    segments: list[dict[str, Any]] = []
    ranking_switch_unresolved_segments: list[dict[str, Any]] = []
    for raw_proposal in _sequence(resolved["proposals"], field="resolved.proposals"):
        proposal = _mapping(raw_proposal, field="resolved.proposal")
        action = _text(proposal.get("action_key"), field="resolved.proposal.action_key")

        # In the adaptive route, a top1 winner change is not sufficient to
        # certify an action boundary.  Keep the full transition evidence in a
        # detached unresolved diagnostic and omit the proposal from the
        # ``MODEL_PROBE_BOUND`` segment collection so it cannot trigger a
        # short-context refinement request.
        unsupported_sides: list[str] = []
        ranking_switch_evidence: dict[str, Any] = {}
        transition_value = proposal.get("transition_diagnostics", {})
        transition_map = transition_value if isinstance(transition_value, Mapping) else {}
        if ranking_switch_suppression_active:
            for side in ("onset", "offset"):
                transition = transition_map.get(side)
                if not isinstance(transition, Mapping):
                    continue
                switch_evidence = _ranking_switch_boundary_evidence(
                    action=action,
                    side=side,
                    transition=transition,
                    probe_metadata=probe_metadata,
                    policy=policy,
                    min_camera_support=support_min,
                    start_threshold=start_threshold,
                    stop_threshold=stop_threshold,
                )
                if switch_evidence is not None:
                    unsupported_sides.append(side)
                    ranking_switch_evidence[side] = switch_evidence
        if unsupported_sides:
            ranking_switch_unresolved_segments.append(
                {
                    "proposal_id": _json_copy(
                        proposal.get("proposal_id"), field="ranking_switch.proposal_id"
                    ),
                    "action_key": action,
                    "provisional_id": action,
                    "start_seconds": _json_copy(
                        proposal.get("start_seconds"), field="ranking_switch.start_seconds"
                    ),
                    "end_seconds": _json_copy(
                        proposal.get("end_seconds"), field="ranking_switch.end_seconds"
                    ),
                    "boundary_status": "UNRESOLVED",
                    "reason": "RANKING_SWITCH_ONLY",
                    "unsupported_sides": unsupported_sides,
                    "supporting_window_ids": _json_copy(
                        proposal.get("supporting_window_ids", []),
                        field="ranking_switch.supporting_window_ids",
                    ),
                    "transition_diagnostics": _json_copy(
                        transition_map, field="ranking_switch.transition_diagnostics"
                    ),
                    "ranking_switch_evidence": ranking_switch_evidence,
                    "review_required": True,
                    "automatic_eligible": False,
                }
            )
            continue
        metadata = action_metadata[action]
        supporting_window_ids = [
            _text(value, field="supporting_window_ids[]")
            for value in _sequence(
                proposal.get("supporting_window_ids", []), field="supporting_window_ids"
            )
        ]
        supporting_candidates = [
            candidate_rows[(action, window_id)]
            for window_id in supporting_window_ids
            if (action, window_id) in candidate_rows
        ]
        top_k_by_window = [
            {
                "window_id": context["window_id"],
                "candidates": _json_copy(context["top_k"], field="context.top_k"),
            }
            for context in contexts
            if context["window_id"] in supporting_window_ids
        ]
        support_ids = tuple(
            sorted(
                {
                    camera_id
                    for candidate in supporting_candidates
                    for camera_id in _sequence(candidate.get("camera_ids", ()), field="camera_ids")
                    if isinstance(camera_id, str) and camera_id.strip()
                }
            )
        )
        support_count = max(
            [
                len(support_ids),
                *[
                    int(candidate.get("camera_support_count", 0))
                    for candidate in supporting_candidates
                    if isinstance(candidate.get("camera_support_count", 0), int)
                    and not isinstance(candidate.get("camera_support_count", 0), bool)
                ],
            ]
        )
        evidence: list[dict[str, Any]] = []
        for window_id in supporting_window_ids:
            metadata_row = probe_metadata[(action, window_id)]
            candidate_row = candidate_rows.get((action, window_id))
            evidence_row: dict[str, Any] = {
                "window_id": window_id,
                "score": float(metadata_row["effective_score"]),
                "raw_score": float(metadata_row["raw_score"]),
                "score_policy": policy,
                "rank": metadata_row["rank"],
                "winner_for_context": metadata_row["winner"],
                "raw_winner_for_context": metadata_row["raw_winner"],
                "stabilized_winner_for_context": metadata_row["stabilized_winner"],
                "raw_winner_action": metadata_row["raw_winner_action"],
                "stabilized_winner_action": metadata_row["stabilized_winner_action"],
                "winner_stabilized": metadata_row["winner_stabilized"],
                "score_carried": metadata_row["score_carried"],
                "score_imputed": metadata_row["score_imputed"],
                "score_source": metadata_row["score_source"],
                "carry_source_window_ids": list(metadata_row["carry_source_window_ids"]),
                "runner_up_action": metadata_row.get("runner_up_action"),
                "runner_up_score": metadata_row.get("runner_up_score"),
                "runner_up_camera_support_count": metadata_row.get(
                    "runner_up_camera_support_count"
                ),
                "runner_up_camera_ids": list(metadata_row.get("runner_up_camera_ids", ())),
                "signed_margin": metadata_row.get("signed_margin"),
                "relative_margin_confidence": metadata_row.get("relative_margin_confidence"),
                "relative_margin_scale": metadata_row.get(
                    "relative_margin_scale", relative_margin_scale
                ),
                "relative_margin_min_target_score": metadata_row.get(
                    "relative_margin_min_target_score", relative_margin_min_target_score
                ),
                "relative_margin_target_floor_passed": metadata_row.get(
                    "relative_margin_target_floor_passed"
                ),
                "relative_margin_camera_overlap_count": metadata_row.get(
                    "relative_margin_camera_overlap_count"
                ),
                "relative_margin_camera_provenance_known": metadata_row.get(
                    "relative_margin_camera_provenance_known"
                ),
                "relative_margin_camera_sets_known": metadata_row.get(
                    "relative_margin_camera_sets_known"
                ),
                "relative_margin_camera_sets_compatible": metadata_row.get(
                    "relative_margin_camera_sets_compatible"
                ),
                # An imputed middle score has no candidate row and therefore
                # deliberately carries no camera evidence.  Its surrounding
                # Top-K context remains available in ``top_k_by_window``.
                "camera_support": [],
                "top_k_candidate": None,
            }
            if candidate_row is not None:
                evidence_row["camera_support"] = list(
                    _sequence(candidate_row.get("camera_ids", ()), field="camera_ids")
                )
                evidence_row["top_k_candidate"] = _json_copy(
                    candidate_row["snapshot"], field="snapshot"
                )
            evidence.append(evidence_row)
        carried_count = sum(
            int(probe_metadata[(action, window_id)]["score_carried"])
            for window_id in supporting_window_ids
        )
        imputed_count = sum(
            int(probe_metadata[(action, window_id)]["score_imputed"])
            for window_id in supporting_window_ids
        )
        segment = {
            "segment_id": proposal["proposal_id"],
            "proposal_id": proposal["proposal_id"],
            "proposal_status": "PROPOSED",
            "provisional_id": action,
            "label_text": metadata.get("label_text"),
            "label_variant": metadata.get("label_variant"),
            "structured_labels": _json_copy(metadata.get("structured_labels", {}), field="labels"),
            "start_seconds": proposal["start_seconds"],
            "end_seconds": proposal["end_seconds"],
            "boundary_status": BOUNDARY_STATUS,
            "boundary_source": BOUNDARY_SOURCE,
            "boundary_method": proposal.get("boundary_method"),
            "production_eligible": False,
            # The score-derived interval is a review-only proposal.  Keep all
            # context/action-boundary markers explicit so downstream readers
            # cannot mistake it for a production annotation span.
            "context_only": True,
            "window_context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
            "boundary_confidence": proposal.get("boundary_confidence"),
            "confidence": proposal.get("confidence"),
            "mean_score": proposal.get("mean_score"),
            "peak_score": proposal.get("peak_score"),
            "camera_support": list(support_ids),
            "camera_support_count": support_count,
            "camera_support_ids_complete": (
                support_count > 0 and support_count == len(support_ids)
            ),
            "supporting_window_ids": supporting_window_ids,
            # Keep the complete retrieval context for human review.  Each
            # context retains its original ordered Top-K list; no candidate is
            # collapsed into a single language-generated label.
            "top_k": top_k_by_window,
            "top_k_by_window": top_k_by_window,
            "evidence": evidence,
            "score_carried_count": carried_count,
            "score_imputed_count": imputed_count,
            "transition_diagnostics": _json_copy(
                proposal.get("transition_diagnostics", {}), field="transition_diagnostics"
            ),
            "review_required": True,
            "automatic_eligible": False,
            "decision": "pending",
        }
        segments.append(segment)
    segments.sort(
        key=lambda row: (
            float(row["start_seconds"]),
            float(row["end_seconds"]),
            -float(row.get("peak_score") or 0.0),
            str(row["provisional_id"]),
        )
    )
    return {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": STATUS,
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "mode": MODE_DENSE_SCORE,
        "context_interval": {
            "start_seconds": source_start,
            "end_seconds": source_end,
            "context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
        },
        "parameters": {
            **dict(_mapping(resolved["parameters"], field="resolved.parameters")),
            "min_camera_support": support_min,
            "missing_top_k_score": 0.0,
            "missing_top_k_semantics": "ranking_absence_not_visual_negative",
            "score_policy": policy,
            "relative_margin_scale": relative_margin_scale,
            "relative_margin_min_target_score": relative_margin_min_target_score,
            "suppress_ranking_switch_boundaries": suppress_ranking_switch_boundaries,
            "ranking_switch_suppression_active": ranking_switch_suppression_active,
        },
        "segments": segments,
        "score_trajectories": resolved_trajectories,
        "diagnostics": {
            **dict(_mapping(resolved["diagnostics"], field="resolved.diagnostics")),
            "context_window_count": len(contexts),
            "candidate_action_count": len(all_actions),
            "temporal_probe_count": len(probes),
            "segment_count": len(segments),
            "mean_top_k_per_context": (
                sum(action_count_by_context) / len(action_count_by_context)
                if action_count_by_context
                else 0.0
            ),
            "context_grid": _context_grid_metadata(contexts),
            "fused_trajectory_camera_provenance": (
                "source_camera_ids_from_top_k_candidate_evidence"
            ),
            "score_policy": policy,
            "relative_margin_scale": relative_margin_scale,
            "winner_only_context_support": policy
            in {SCORE_POLICY_TOP1, SCORE_POLICY_WINNER_STABLE},
            "raw_winner_sequence": list(raw_winners),
            "stabilized_winner_sequence": list(stabilized_winners),
            "winner_sequence": winner_sequence,
            "raw_winner_switch_count": raw_winner_switch_count,
            "stabilized_winner_switch_count": stabilized_winner_switch_count,
            "winner_stabilization_count": winner_stabilization_count,
            "score_carry_count": score_carry_count,
            "score_imputation_count": score_imputation_count,
            "ranking_switch_suppression_enabled": suppress_ranking_switch_boundaries,
            "ranking_switch_suppression_active": ranking_switch_suppression_active,
            "ranking_switch_unresolved_count": len(ranking_switch_unresolved_segments),
            "ranking_switch_unresolved_segments": _json_copy(
                ranking_switch_unresolved_segments,
                field="diagnostics.ranking_switch_unresolved_segments",
            ),
            # Alias retained for callers that use proposal terminology rather
            # than segment terminology; both fields carry the same detached
            # unresolved diagnostics.
            "ranking_switch_unresolved_proposals": _json_copy(
                ranking_switch_unresolved_segments,
                field="diagnostics.ranking_switch_unresolved_proposals",
            ),
        },
        "controls": {
            "model_invoked": False,
            "gold_read": False,
            "gold_written": False,
            "top_k_retained": True,
            "missing_top_k_recorded_as_zero": True,
            "winner_stability_applied": policy == SCORE_POLICY_WINNER_STABLE,
            "ranking_switch_suppression_applied": ranking_switch_suppression_active,
            "ontology_modified": False,
            "mapper_modified": False,
            "qwen_invoked": False,
            "mage_invoked": False,
            "score_policy_applied": policy,
            "relative_margin_applied": policy == SCORE_POLICY_RELATIVE_MARGIN,
        },
        "limitations": [
            "Context windows remain WeMM input only and are not action boundaries.",
            "A missing Top-K row means absent retrieval support, not a visual negative label.",
            "The top1 score policy gates temporal support to the deterministic context winner; "
            "non-winning Top-K candidates remain available for human review.",
            "The winner_stable policy repairs only one-context interior winner glitches; "
            "carried scores are conservative review evidence and carry no camera provenance.",
            "The relative_margin policy compares a candidate with the strongest eligible "
            "competitor and applies a logistic projection; its confidence is not calibrated.",
            "A ranking-only winner switch is not an action boundary when adaptive suppression "
            "is enabled; the unresolved transition remains in diagnostics.",
            "Segments are model estimates and require human review.",
        ],
    }


__all__ = [
    "DEFAULT_RELATIVE_MARGIN_SCALE",
    "DEFAULT_SCORE_POLICY",
    "FORMAT",
    "MODE_ADAPTIVE_SCORE",
    "MODE_DENSE_SCORE",
    "MODE_NONE",
    "SCORE_POLICIES",
    "SCORE_POLICY_ABSOLUTE",
    "SCORE_POLICY_ALIASES",
    "SCORE_POLICY_RELATIVE_MARGIN",
    "SCORE_POLICY_TOP1",
    "SCORE_POLICY_WINNER_STABLE",
    "STATUS",
    "TEMPORAL_MODES",
    "ProductionWemmTemporalError",
    "_stabilize_winner_sequence",
    "normalize_score_policy",
    "resolve_wemm_temporal_segments",
]
