"""Project dense WeMM temporal probe scores into action-interval proposals.

The production WeMM runner deliberately processes bounded windows.  A bounded
window is a *context envelope*, not an action boundary, so a model score for a
window cannot be copied directly into an annotation interval.  This module is
the smallest deterministic seam for a later model-driven temporal pass:

* the caller supplies scores for synchronized, source-relative probe spans;
* scores are fused across cameras on the same probe grid;
* hysteresis (an activation and release threshold) suppresses score chatter;
* contiguous supported probes become interval proposals bounded by observed
  probe spans; and
* the result remains a review-only, non-gold artifact that can be copied into
  the existing pre-annotation proposal ``start_seconds``/``end_seconds``
  fields by an explicit caller.

No media is decoded, no model is loaded, and no ontology or Mapper decision is
made here.  The fixed processing window therefore remains visible as context
while the proposed interval is independently measurable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Final

FORMAT: Final = "robata-production-wemm-model-driven-interval-proposal-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "PROPOSALS_ONLY"
BOUNDARY_STATUS: Final = "MODEL_PROBE_BOUND"
BOUNDARY_SOURCE: Final = "wemm_temporal_score"
DEFAULT_START_THRESHOLD: Final = 0.65
DEFAULT_STOP_THRESHOLD: Final = 0.50
DEFAULT_MERGE_GAP_SECONDS: Final = 0.25
DEFAULT_MIN_DURATION_SECONDS: Final = 0.10
BOUNDARY_MODES: Final = ("observed_probe", "midpoint")


class ProductionWemmIntervalProposalError(ValueError):
    """Raised when temporal probe input or parameters are malformed."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmIntervalProposalError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmIntervalProposalError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionWemmIntervalProposalError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ProductionWemmIntervalProposalError(f"{field} must be finite")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionWemmIntervalProposalError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ProductionWemmIntervalProposalError(f"{field} must be finite")
    return result


def _threshold(value: object, *, field: str) -> float:
    result = _number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ProductionWemmIntervalProposalError(f"{field} must be between 0 and 1")
    return result


def _positive_or_zero(value: object, *, field: str) -> float:
    result = _number(value, field=field)
    if result < 0.0:
        raise ProductionWemmIntervalProposalError(f"{field} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class TemporalProbe:
    """One model score over one source-relative temporal probe span."""

    action_key: str
    start_seconds: float
    end_seconds: float
    score: float
    camera_id: str = "__aggregate__"
    window_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_key, str) or not self.action_key.strip():
            raise ProductionWemmIntervalProposalError("action_key must be non-empty")
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ProductionWemmIntervalProposalError("camera_id must be non-empty")
        if self.window_id is not None and (
            not isinstance(self.window_id, str) or not self.window_id.strip()
        ):
            raise ProductionWemmIntervalProposalError("window_id must be non-empty when supplied")
        if not math.isfinite(self.start_seconds) or not math.isfinite(self.end_seconds):
            raise ProductionWemmIntervalProposalError("probe bounds must be finite")
        if self.start_seconds < 0.0:
            raise ProductionWemmIntervalProposalError("probe start must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ProductionWemmIntervalProposalError("probe end must exceed start")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ProductionWemmIntervalProposalError("probe score must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action_key": self.action_key,
            "camera_id": self.camera_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "score": self.score,
        }
        if self.window_id is not None:
            result["window_id"] = self.window_id
        return result


@dataclass(frozen=True, slots=True)
class AggregatedProbe:
    """A synchronized probe after deterministic camera-score fusion."""

    action_key: str
    start_seconds: float
    end_seconds: float
    score: float
    camera_ids: tuple[str, ...]
    source_scores: tuple[float, ...]
    window_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action_key": self.action_key,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "score": self.score,
            "camera_ids": list(self.camera_ids),
            "source_scores": list(self.source_scores),
        }
        if self.window_ids:
            result["window_ids"] = list(self.window_ids)
        return result


def parse_temporal_probes(value: Sequence[Any]) -> tuple[TemporalProbe, ...]:
    """Validate JSON-shaped probe rows without invoking a model."""

    rows = _sequence(value, field="probes")
    parsed: list[TemporalProbe] = []
    seen: set[tuple[str, str, float, float]] = set()
    for index, raw in enumerate(rows):
        if isinstance(raw, TemporalProbe):
            probe = raw
            key = (probe.action_key, probe.camera_id, probe.start_seconds, probe.end_seconds)
            if key in seen:
                raise ProductionWemmIntervalProposalError(
                    f"duplicate probe for {probe.action_key!r}/{probe.camera_id!r} "
                    f"[{probe.start_seconds}, {probe.end_seconds})"
                )
            seen.add(key)
            parsed.append(probe)
            continue
        row = _mapping(raw, field=f"probes[{index}]")
        action_key = _text(row.get("action_key"), field=f"probes[{index}].action_key")
        camera_id = _text(row.get("camera_id", "__aggregate__"), field=f"probes[{index}].camera_id")
        window_id_raw = row.get("window_id")
        window_id = (
            _text(window_id_raw, field=f"probes[{index}].window_id")
            if window_id_raw is not None
            else None
        )
        start = _number(row.get("start_seconds"), field=f"probes[{index}].start_seconds")
        end = _number(row.get("end_seconds"), field=f"probes[{index}].end_seconds")
        score = _threshold(row.get("score"), field=f"probes[{index}].score")
        key = (action_key, camera_id, start, end)
        if key in seen:
            raise ProductionWemmIntervalProposalError(
                f"duplicate probe for {action_key!r}/{camera_id!r} [{start}, {end})"
            )
        seen.add(key)
        parsed.append(
            TemporalProbe(
                action_key=action_key,
                camera_id=camera_id,
                start_seconds=start,
                end_seconds=end,
                score=score,
                window_id=window_id,
            )
        )
    return tuple(parsed)


def aggregate_temporal_probes(
    probes: Sequence[TemporalProbe],
    *,
    method: str = "mean",
) -> tuple[AggregatedProbe, ...]:
    """Fuse camera scores on an identical probe grid.

    The production six-camera decoder uses synchronized source-relative spans.
    Requiring exact span equality here is intentional: silently interpolating
    camera clocks would manufacture temporal evidence.  Callers with already
    fused scores can use the default ``__aggregate__`` camera id.
    """

    if method not in {"mean", "max"}:
        raise ProductionWemmIntervalProposalError("method must be 'mean' or 'max'")
    grouped: dict[tuple[str, float, float], list[TemporalProbe]] = defaultdict(list)
    for probe in probes:
        grouped[(probe.action_key, probe.start_seconds, probe.end_seconds)].append(probe)
    result: list[AggregatedProbe] = []
    for (action_key, start, end), rows in grouped.items():
        ordered_rows = tuple(sorted(rows, key=lambda probe: probe.camera_id))
        camera_ids = tuple(probe.camera_id for probe in ordered_rows)
        if len(set(camera_ids)) != len(camera_ids):
            raise ProductionWemmIntervalProposalError(
                f"duplicate camera probe for {action_key!r} [{start}, {end})"
            )
        # Keep source_scores positionally aligned with camera_ids.  The input
        # order is deliberately not treated as semantic because six-camera
        # callers may enumerate topics in different orders.
        scores = tuple(float(probe.score) for probe in ordered_rows)
        score = max(scores) if method == "max" else fmean(scores)
        window_ids = tuple(
            sorted({probe.window_id for probe in ordered_rows if probe.window_id is not None})
        )
        result.append(
            AggregatedProbe(
                action_key=action_key,
                start_seconds=start,
                end_seconds=end,
                score=score,
                camera_ids=camera_ids,
                source_scores=scores,
                window_ids=window_ids,
            )
        )
    result.sort(key=lambda row: (row.action_key, row.start_seconds, row.end_seconds))
    return tuple(result)


def _validate_parameters(
    *,
    window_start_seconds: float,
    window_end_seconds: float,
    start_threshold: float,
    stop_threshold: float,
    merge_gap_seconds: float,
    min_duration_seconds: float,
    min_camera_support: int,
    boundary_mode: str,
) -> tuple[float, float, float, float, float, float, int]:
    window_start = _number(window_start_seconds, field="window_start_seconds")
    window_end = _number(window_end_seconds, field="window_end_seconds")
    if window_start < 0.0 or window_end <= window_start:
        raise ProductionWemmIntervalProposalError("window end must exceed non-negative start")
    start = _threshold(start_threshold, field="start_threshold")
    stop = _threshold(stop_threshold, field="stop_threshold")
    if stop > start:
        raise ProductionWemmIntervalProposalError(
            "stop_threshold must be <= start_threshold for hysteresis"
        )
    gap = _positive_or_zero(merge_gap_seconds, field="merge_gap_seconds")
    minimum = _positive_or_zero(min_duration_seconds, field="min_duration_seconds")
    if isinstance(min_camera_support, bool) or not isinstance(min_camera_support, int):
        raise ProductionWemmIntervalProposalError("min_camera_support must be an integer")
    if min_camera_support <= 0:
        raise ProductionWemmIntervalProposalError("min_camera_support must be positive")
    if boundary_mode not in BOUNDARY_MODES:
        raise ProductionWemmIntervalProposalError(
            "boundary_mode must be one of " + ", ".join(BOUNDARY_MODES)
        )
    return window_start, window_end, start, stop, gap, minimum, min_camera_support


def _clamp_unit(value: float) -> float:
    """Clamp a finite score-derived value to the contract's confidence range."""

    return max(0.0, min(1.0, float(value)))


def _transition_diagnostic(
    *,
    side: str,
    active_probe: AggregatedProbe,
    neighbouring_probe: AggregatedProbe | None,
    boundary_seconds: float,
    activation_threshold: float,
    release_threshold: float,
    boundary_mode: str,
) -> dict[str, Any]:
    """Describe the observed probe transition at one proposal boundary.

    The resolver intentionally does *not* interpolate a timestamp between two
    context windows.  ``boundary_seconds`` is therefore the observed active
    probe edge, while this diagnostic records whether an adjacent score crossed
    the hysteresis threshold.  A zero boundary confidence means that the
    active run touched a context edge or lacked a qualifying neighbour; it is
    not a claim that the action was absent.
    """

    if side not in {"onset", "offset"}:
        raise ProductionWemmIntervalProposalError("transition side must be onset or offset")
    gap: float | None = None
    if neighbouring_probe is None:
        crossed = False
        reason = "NO_PRECEDING_PROBE" if side == "onset" else "NO_FOLLOWING_PROBE"
        delta: float | None = None
        confidence = 0.0
    else:
        if side == "onset":
            crossed = (
                neighbouring_probe.score < activation_threshold
                and active_probe.score >= activation_threshold
            )
        else:
            crossed = (
                active_probe.score >= release_threshold
                and neighbouring_probe.score < release_threshold
            )
        delta = round(abs(active_probe.score - neighbouring_probe.score), 6)
        confidence = _clamp_unit(delta) if crossed else 0.0
        reason = "THRESHOLD_CROSSING" if crossed else "NO_THRESHOLD_CROSSING"
        gap = (
            active_probe.start_seconds - neighbouring_probe.end_seconds
            if side == "onset"
            else neighbouring_probe.start_seconds - active_probe.end_seconds
        )
    if neighbouring_probe is None or boundary_mode == "observed_probe":
        method = "observed_probe_span"
        interpolated = False
    else:
        # Dense context windows overlap.  The model score is attached to the
        # probe's centre, so a threshold crossing is best localized halfway
        # between neighbouring probe centres rather than by copying a 4 s
        # context edge into the action annotation.
        method = "probe_center_midpoint"
        interpolated = True
    return {
        "side": side,
        "boundary_seconds": float(boundary_seconds),
        "boundary_method": method,
        "transition_input_policy": "camera_support_eligible_probes_only",
        "interpolated": interpolated,
        "crossed_threshold": crossed,
        "score_delta": delta,
        "probe_gap_seconds": round(gap, 6) if gap is not None else None,
        "confidence": round(confidence, 6),
        "active_probe": active_probe.to_dict(),
        "neighbouring_probe": (
            neighbouring_probe.to_dict() if neighbouring_probe is not None else None
        ),
        "reason": reason,
    }


def _proposal_from_region(
    action_key: str,
    region: Sequence[AggregatedProbe],
    *,
    window_start_seconds: float,
    window_end_seconds: float,
    preceding_probe: AggregatedProbe | None = None,
    following_probe: AggregatedProbe | None = None,
    activation_threshold: float = DEFAULT_START_THRESHOLD,
    release_threshold: float = DEFAULT_STOP_THRESHOLD,
    boundary_mode: str = "observed_probe",
) -> dict[str, Any]:
    observed_start = min(row.start_seconds for row in region)
    observed_end = max(row.end_seconds for row in region)
    if boundary_mode == "midpoint" and preceding_probe is not None:
        preceding_center = (preceding_probe.start_seconds + preceding_probe.end_seconds) / 2.0
        active_center = (region[0].start_seconds + region[0].end_seconds) / 2.0
        start = (preceding_center + active_center) / 2.0
    else:
        start = observed_start
    if boundary_mode == "midpoint" and following_probe is not None:
        # For overlapping probes, use centre-to-centre localization instead
        # of an edge that can fall inside the active context span.
        active_center = (region[-1].start_seconds + region[-1].end_seconds) / 2.0
        following_center = (following_probe.start_seconds + following_probe.end_seconds) / 2.0
        end = (active_center + following_center) / 2.0
    else:
        end = observed_end
    start = max(window_start_seconds, min(start, window_end_seconds))
    end = max(window_start_seconds, min(end, window_end_seconds))
    if end <= start:
        # Keep malformed/degenerate midpoint crossings explicit; the caller's
        # minimum-duration gate will discard them rather than repairing them.
        end = observed_end
        start = observed_start
    camera_ids = tuple(sorted({camera for row in region for camera in row.camera_ids}))
    window_ids = tuple(sorted({window_id for row in region for window_id in row.window_ids}))
    scores = [row.score for row in region]
    onset = _transition_diagnostic(
        side="onset",
        active_probe=region[0],
        neighbouring_probe=preceding_probe,
        boundary_seconds=start,
        activation_threshold=activation_threshold,
        release_threshold=release_threshold,
        boundary_mode=boundary_mode,
    )
    offset = _transition_diagnostic(
        side="offset",
        active_probe=region[-1],
        neighbouring_probe=following_probe,
        boundary_seconds=end,
        activation_threshold=activation_threshold,
        release_threshold=release_threshold,
        boundary_mode=boundary_mode,
    )
    boundary_confidence = round(fmean([float(onset["confidence"]), float(offset["confidence"])]), 6)
    return {
        "proposal_id": f"{action_key}@{start:.6f}-{end:.6f}",
        "action_key": action_key,
        "start_seconds": start,
        "end_seconds": end,
        "boundary_status": BOUNDARY_STATUS,
        "boundary_source": BOUNDARY_SOURCE,
        "boundary_method": (
            "probe_center_midpoint" if boundary_mode == "midpoint" else "observed_probe_span"
        ),
        "boundary_confidence": boundary_confidence,
        "confidence": max(scores),
        "mean_score": fmean(scores),
        "peak_score": max(scores),
        "camera_support": list(camera_ids),
        "supporting_window_ids": list(window_ids),
        "probe_count": len(region),
        "evidence": [row.to_dict() for row in region],
        "transition_diagnostics": {"onset": onset, "offset": offset},
        "window_context_only": True,
        "review_required": True,
        "automatic_eligible": False,
    }


def propose_model_intervals(
    probes: Sequence[Any],
    *,
    window_start_seconds: float,
    window_end_seconds: float,
    start_threshold: float = DEFAULT_START_THRESHOLD,
    stop_threshold: float = DEFAULT_STOP_THRESHOLD,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS,
    min_camera_support: int = 1,
    camera_fusion: str = "mean",
    boundary_mode: str = "observed_probe",
) -> dict[str, Any]:
    """Turn temporal probe scores into review-only interval proposals.

    Probe spans must lie inside the supplied processing window.  The function
    never expands an interval beyond observed spans, never clips an invalid
    input silently, and preserves every contributing probe as evidence.
    """

    (
        window_start,
        window_end,
        activation,
        release,
        merge_gap,
        minimum,
        camera_support_min,
    ) = _validate_parameters(
        window_start_seconds=window_start_seconds,
        window_end_seconds=window_end_seconds,
        start_threshold=start_threshold,
        stop_threshold=stop_threshold,
        merge_gap_seconds=merge_gap_seconds,
        min_duration_seconds=min_duration_seconds,
        min_camera_support=min_camera_support,
        boundary_mode=boundary_mode,
    )
    parsed = parse_temporal_probes(probes)
    for parsed_probe in parsed:
        if parsed_probe.start_seconds < window_start or parsed_probe.end_seconds > window_end:
            raise ProductionWemmIntervalProposalError(
                f"probe [{parsed_probe.start_seconds}, "
                f"{parsed_probe.end_seconds}) is outside window"
            )
    aggregated = aggregate_temporal_probes(parsed, method=camera_fusion)
    by_action: dict[str, list[AggregatedProbe]] = defaultdict(list)
    trajectories: dict[str, list[AggregatedProbe]] = defaultdict(list)
    for aggregated_probe in aggregated:
        trajectories[aggregated_probe.action_key].append(aggregated_probe)
        if len(aggregated_probe.camera_ids) >= camera_support_min:
            by_action[aggregated_probe.action_key].append(aggregated_probe)

    proposals: list[dict[str, Any]] = []
    discarded_short: list[dict[str, Any]] = []
    for action_key, rows in sorted(by_action.items()):
        rows.sort(key=lambda row: (row.start_seconds, row.end_seconds, -row.score))
        region: list[AggregatedProbe] = []
        preceding_probe: AggregatedProbe | None = None
        for row_index, row in enumerate(rows):
            if not region:
                if row.score >= activation:
                    region = [row]
                    preceding_probe = rows[row_index - 1] if row_index else None
                continue
            previous = region[-1]
            gap = row.start_seconds - previous.end_seconds
            if gap <= merge_gap and row.score >= release:
                region.append(row)
                continue
            proposal = _proposal_from_region(
                action_key,
                region,
                window_start_seconds=window_start,
                window_end_seconds=window_end,
                preceding_probe=preceding_probe,
                following_probe=row,
                activation_threshold=activation,
                release_threshold=release,
                boundary_mode=boundary_mode,
            )
            if proposal["end_seconds"] - proposal["start_seconds"] >= minimum:
                proposals.append(proposal)
            else:
                discarded_short.append(proposal)
            if row.score >= activation:
                region = [row]
                preceding_probe = previous
            else:
                region = []
                preceding_probe = None
        if region:
            proposal = _proposal_from_region(
                action_key,
                region,
                window_start_seconds=window_start,
                window_end_seconds=window_end,
                preceding_probe=preceding_probe,
                following_probe=None,
                activation_threshold=activation,
                release_threshold=release,
                boundary_mode=boundary_mode,
            )
            if proposal["end_seconds"] - proposal["start_seconds"] >= minimum:
                proposals.append(proposal)
            else:
                discarded_short.append(proposal)

    proposals.sort(
        key=lambda row: (
            float(row["start_seconds"]),
            float(row["end_seconds"]),
            -float(row["peak_score"]),
            str(row["action_key"]),
        )
    )
    trajectory_rows: list[dict[str, Any]] = []
    for action_key, rows in sorted(trajectories.items()):
        trajectory_probes: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: (item.start_seconds, item.end_seconds)):
            item = row.to_dict()
            item["camera_support_count"] = len(row.camera_ids)
            item["camera_support_eligible"] = len(row.camera_ids) >= camera_support_min
            trajectory_probes.append(item)
        trajectory_rows.append(
            {
                "action_key": action_key,
                "probes": trajectory_probes,
                "camera_support_eligible": action_key in by_action,
            }
        )
    boundary_confidences = [float(row["boundary_confidence"]) for row in proposals]
    bracketed_count = sum(
        int(
            bool(row["transition_diagnostics"]["onset"]["crossed_threshold"])
            and bool(row["transition_diagnostics"]["offset"]["crossed_threshold"])
        )
        for row in proposals
    )
    return {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": STATUS,
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "window": {
            "start_seconds": window_start,
            "end_seconds": window_end,
            "context_only": True,
        },
        "parameters": {
            "start_threshold": activation,
            "stop_threshold": release,
            "merge_gap_seconds": merge_gap,
            "min_duration_seconds": minimum,
            "min_camera_support": camera_support_min,
            "camera_fusion": camera_fusion,
            "boundary_mode": boundary_mode,
        },
        "temporal_evidence_policy": {
            "transition_inputs": "camera_support_eligible_probes_only",
            "unsupported_probes_retained_in_score_trajectories": True,
        },
        "proposals": proposals,
        "discarded_short_proposals": discarded_short,
        "score_trajectories": trajectory_rows,
        "diagnostics": {
            "input_probe_count": len(parsed),
            "aggregated_probe_count": len(aggregated),
            "action_count": len(by_action),
            "proposal_count": len(proposals),
            "discarded_short_count": len(discarded_short),
            "trajectory_count": len(trajectory_rows),
            "boundary_bracketed_count": bracketed_count,
            "boundary_confidence_mean": (
                round(fmean(boundary_confidences), 6) if boundary_confidences else 0.0
            ),
            "transition_diagnostics_preserved": True,
            "boundary_mode": boundary_mode,
        },
        "controls": {
            "media_decoded": False,
            "model_invoked": False,
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "raw_probe_scores_preserved": True,
        },
        "limitations": [
            "Processing windows remain context envelopes, not action boundaries.",
            "Scores must come from synchronized source-relative probe spans.",
            "Probes below min_camera_support remain observable in score trajectories but do not "
            "supply proposal boundaries.",
            "No interpolation or clipping is performed for out-of-window probes.",
            "Every proposal requires human review and is not production gold.",
        ],
    }


# ``resolve_temporal_intervals`` is the descriptive name used by the runner
# integration plan.  Keep the original public function name as the stable
# low-level seam while exposing the alias so callers do not need to know that
# the implementation is still proposal-only.
resolve_temporal_intervals = propose_model_intervals


__all__ = [
    "AUTHORITY",
    "BOUNDARY_MODES",
    "BOUNDARY_SOURCE",
    "BOUNDARY_STATUS",
    "DEFAULT_MERGE_GAP_SECONDS",
    "DEFAULT_MIN_DURATION_SECONDS",
    "DEFAULT_START_THRESHOLD",
    "DEFAULT_STOP_THRESHOLD",
    "FORMAT",
    "AggregatedProbe",
    "ProductionWemmIntervalProposalError",
    "TemporalProbe",
    "aggregate_temporal_probes",
    "parse_temporal_probes",
    "propose_model_intervals",
    "resolve_temporal_intervals",
]
