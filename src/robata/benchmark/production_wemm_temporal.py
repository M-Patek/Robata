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
TEMPORAL_MODES: Final = (MODE_NONE, MODE_DENSE_SCORE)
SCORE_POLICY_ABSOLUTE: Final = "absolute"
SCORE_POLICY_TOP1: Final = "top1"
SCORE_POLICIES: Final = (SCORE_POLICY_ABSOLUTE, SCORE_POLICY_TOP1)
DEFAULT_SCORE_POLICY: Final = SCORE_POLICY_TOP1


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


def _score_policy(value: object, *, field: str = "score_policy") -> str:
    if not isinstance(value, str):
        raise ProductionWemmTemporalError(f"{field} must be one of {', '.join(SCORE_POLICIES)}")
    policy = value.strip().casefold().replace("-", "_")
    aliases = {"raw": SCORE_POLICY_ABSOLUTE, "winner": SCORE_POLICY_TOP1}
    policy = aliases.get(policy, policy)
    if policy not in SCORE_POLICIES:
        raise ProductionWemmTemporalError(f"{field} must be one of {', '.join(SCORE_POLICIES)}")
    return policy


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
    evidence = row.get("evidence", [])
    result: set[str] = set()
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        for value in evidence:
            if isinstance(value, Mapping):
                camera = value.get("camera_id")
                if isinstance(camera, str) and camera.strip():
                    result.add(camera.strip())
    support = row.get("camera_support")
    if isinstance(support, Sequence) and not isinstance(support, (str, bytes, bytearray)):
        for value in support:
            if isinstance(value, str) and value.strip():
                result.add(value.strip())
    return tuple(sorted(result))


def _camera_support_count(row: Mapping[str, Any]) -> int:
    ids = _camera_ids(row)
    if ids:
        return len(ids)
    raw = row.get("camera_support")
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
    return _text(row.get("provisional_id", row.get("action_key")), field=field)


def _candidate_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
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


def _context_rows(windows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_window in enumerate(windows):
        window = _mapping(raw_window, field=f"windows[{index}]")
        window_id = _text(window.get("window_id"), field=f"windows[{index}].window_id")
        if window_id in seen:
            raise ProductionWemmTemporalError(f"duplicate window_id: {window_id}")
        seen.add(window_id)
        start = _finite(window.get("start_seconds"), field=f"{window_id}.start_seconds")
        end = _finite(window.get("end_seconds"), field=f"{window_id}.end_seconds")
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
            action = _candidate_action(candidate, field=f"{window_id}.top_k[{candidate_index}]")
            if action in candidates:
                raise ProductionWemmTemporalError(
                    f"duplicate provisional candidate {action!r} in {window_id}"
                )
            rank = _candidate_rank(candidate, field=f"{window_id}.top_k[{candidate_index}]")
            snapshot = _candidate_snapshot(candidate)
            # Rank is inferred from list position when a producer omitted it;
            # preserve the effective value in the temporal evidence either way.
            snapshot["rank"] = rank
            top_k_snapshots.append(snapshot)
            candidates[action] = {
                "rank": rank,
                "score": _unit_score(candidate.get("score"), field=f"{window_id}.{action}.score"),
                "camera_support_count": _camera_support_count(candidate),
                "camera_ids": _camera_ids(candidate),
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
) -> dict[str, Any]:
    """Resolve Top-K context rankings into action interval proposals.

    Every action seen in any context receives a score for every selected
    context.  ``score_policy='top1'`` (the dense-mode default) lets only the
    deterministic Top-K winner contribute temporal support; this prevents the
    globally high, tightly packed WeMM similarities from opening one track
    per phrase.  ``score_policy='absolute'`` retains the historical behavior
    for controlled comparisons.  A missing Top-K entry is recorded as zero
    *ranking support*, not as visual proof that the action is impossible.
    """

    support_min = _positive_int(min_camera_support, field="min_camera_support")
    policy = _score_policy(score_policy)
    contexts = _context_rows(windows)
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
    if not all_actions:
        return {
            "format": FORMAT,
            "authority": AUTHORITY,
            "status": STATUS,
            "production_eligible": False,
            "official_quality_status": "NOT_MEASURED",
            "official_gold_status": "NOT_ESTABLISHED",
            "mode": MODE_DENSE_SCORE,
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
            },
            "diagnostics": {
                "context_window_count": len(contexts),
                "candidate_action_count": 0,
                "temporal_probe_count": 0,
                "segment_count": 0,
            },
            "controls": {
                "model_invoked": False,
                "gold_read": False,
                "gold_written": False,
                "top_k_retained": True,
                "missing_top_k_recorded_as_zero": True,
            },
        }

    probes: list[dict[str, Any]] = []
    candidate_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    probe_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for context in contexts:
        window_id = _text(context["window_id"], field="context.window_id")
        candidates = _mapping(context["candidates"], field=f"{window_id}.candidates")
        winner = _winner_action(candidates, min_camera_support=support_min)
        for action in sorted(all_actions):
            candidate_value = candidates.get(action)
            candidate_for_action = candidate_value if isinstance(candidate_value, Mapping) else None
            if candidate_for_action is None:
                raw_score = 0.0
                rank: int | None = None
                support = 0
                effective_score = 0.0
                camera_ids: tuple[str, ...] = ()
                snapshot: Mapping[str, Any] | None = None
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
                else:
                    effective_score = raw_score
            if candidate_for_action is not None and support >= support_min:
                candidate_rows[(action, window_id)] = candidate_for_action
            probe_metadata[(action, window_id)] = {
                "raw_score": raw_score,
                "effective_score": effective_score,
                "rank": rank,
                "winner": action == winner,
                "camera_support_count": support,
                "camera_ids": list(camera_ids),
                "snapshot": snapshot,
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

    source_start = min(float(context["start_seconds"]) for context in contexts)
    source_end = max(float(context["end_seconds"]) for context in contexts)
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
                probe.update(
                    {
                        "raw_score": metadata["raw_score"],
                        "score_policy": policy,
                        "rank": metadata["rank"],
                        "winner_for_context": metadata["winner"],
                    }
                )

    segments: list[dict[str, Any]] = []
    for raw_proposal in _sequence(resolved["proposals"], field="resolved.proposals"):
        proposal = _mapping(raw_proposal, field="resolved.proposal")
        action = _text(proposal.get("action_key"), field="resolved.proposal.action_key")
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
        evidence = [
            {
                "window_id": window_id,
                "score": float(probe_metadata[(action, window_id)]["effective_score"]),
                "raw_score": float(probe_metadata[(action, window_id)]["raw_score"]),
                "score_policy": policy,
                "rank": probe_metadata[(action, window_id)]["rank"],
                "winner_for_context": probe_metadata[(action, window_id)]["winner"],
                "camera_support": list(
                    _sequence(
                        candidate_rows[(action, window_id)].get("camera_ids", ()),
                        field="camera_ids",
                    )
                ),
                "top_k_candidate": _json_copy(
                    candidate_rows[(action, window_id)]["snapshot"], field="snapshot"
                ),
            }
            for window_id in supporting_window_ids
            if (action, window_id) in candidate_rows
        ]
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
            "boundary_confidence": proposal.get("boundary_confidence"),
            "confidence": proposal.get("confidence"),
            "mean_score": proposal.get("mean_score"),
            "peak_score": proposal.get("peak_score"),
            "camera_support": list(support_ids),
            "supporting_window_ids": supporting_window_ids,
            # Keep the complete retrieval context for human review.  Each
            # context retains its original ordered Top-K list; no candidate is
            # collapsed into a single language-generated label.
            "top_k": top_k_by_window,
            "top_k_by_window": top_k_by_window,
            "evidence": evidence,
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
        },
        "parameters": {
            **dict(_mapping(resolved["parameters"], field="resolved.parameters")),
            "min_camera_support": support_min,
            "missing_top_k_score": 0.0,
            "missing_top_k_semantics": "ranking_absence_not_visual_negative",
            "score_policy": policy,
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
            "score_policy": policy,
            "winner_only_context_support": policy == SCORE_POLICY_TOP1,
        },
        "controls": {
            "model_invoked": False,
            "gold_read": False,
            "gold_written": False,
            "top_k_retained": True,
            "missing_top_k_recorded_as_zero": True,
            "ontology_modified": False,
            "mapper_modified": False,
            "qwen_invoked": False,
            "mage_invoked": False,
            "score_policy_applied": policy,
        },
        "limitations": [
            "Context windows remain WeMM input only and are not action boundaries.",
            "A missing Top-K row means absent retrieval support, not a visual negative label.",
            "The top1 score policy gates temporal support to the deterministic context winner; "
            "non-winning Top-K candidates remain available for human review.",
            "Segments are model estimates and require human review.",
        ],
    }


__all__ = [
    "DEFAULT_SCORE_POLICY",
    "FORMAT",
    "MODE_DENSE_SCORE",
    "MODE_NONE",
    "SCORE_POLICIES",
    "SCORE_POLICY_ABSOLUTE",
    "SCORE_POLICY_TOP1",
    "STATUS",
    "TEMPORAL_MODES",
    "ProductionWemmTemporalError",
    "resolve_wemm_temporal_segments",
]
