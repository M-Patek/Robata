"""Post-hoc rank-distance analysis for production WeMM retrieval sidecars.

The analysis answers a narrow diagnostic question: when a Terra-surrogate action
is not ranked first, how far down the already-recorded WeMM list is it?  It is
deliberately read-only.  No model is loaded, media is decoded, ontology or
Mapper data is imported, and no content identity/hash material is computed.

The input is the production-only vocabulary comparison report emitted by
``production_wemm_vocabulary_comparison``.  Its Terra review is an independent
development surrogate, not official production gold; this distinction is
carried through every report produced here.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Final, cast

ANALYSIS_FORMAT: Final = "robata-production-wemm-rank-error-analysis-v1"
# This is an additive extension of the v1 report.  The wire format deliberately
# remains v1 so existing P17/P13 readers can continue to consume the report.
ANALYSIS_EXTENSION_FORMAT: Final = "robata-production-wemm-rank-error-analysis-extension-v1"
INPUT_FORMAT: Final = "robata-production-wemm-vocabulary-variant-comparison-v1"
DEFAULT_MARGIN_BINS: Final = (
    (0.0, 0.001, "[0,0.001)"),
    (0.001, 0.005, "[0.001,0.005)"),
    (0.005, 0.01, "[0.005,0.01)"),
    (0.01, 0.02, "[0.01,0.02)"),
    (0.02, math.inf, "[0.02,+inf)"),
)

# Keep these labels stable.  ``error_band`` predates this extension and retains
# its rank-2/3, rank-4/5, rank-6+ groupings for backwards compatibility; the
# additive ``rank_bucket`` fields below are the exact production diagnostic
# buckets requested by P13.
RANK_BUCKETS: Final = ("rank_1", "rank_2", "rank_3", "rank_4_plus", "not_in_top_k")
HARD_NEGATIVE_RELATIONS: Final = ("same_verb", "same_noun", "different_action")
# Camera agreement is a routing diagnostic, not semantic evidence.  Keep bins
# aligned with the default selective-routing threshold (0.5) while retaining a
# separate unanimous bucket for easy audit.
CAMERA_CONSENSUS_BINS: Final = (
    (0.0, 0.5, "[0,0.5)"),
    (0.5, 1.0, "[0.5,1)"),
    (1.0, math.inf, "[1,+inf)"),
)


class ProductionWemmRankErrorAnalysisError(ValueError):
    """Raised when a comparison report cannot support a rank analysis."""


def _contains_true_flag(value: object, *keys: str) -> str | None:
    """Return the first forbidden control key set to ``True``."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in keys and child is True:
                return str(key)
            found = _contains_true_flag(child, *keys)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = _contains_true_flag(child, *keys)
            if found is not None:
                return found
    return None


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmRankErrorAnalysisError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmRankErrorAnalysisError(f"{field} must be an array")
    return value


def _load(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductionWemmRankErrorAnalysisError(
                f"could not read comparison report {path}: {exc}"
            ) from exc
        return _mapping(payload, field=str(path))
    return _mapping(value, field="comparison report")


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _pair(value: object) -> tuple[str, str] | None:
    """Read a comparison-report pair while retaining its production spelling."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != 2:
        return None
    verb, noun = _normalise(value[0]), _normalise(value[1])
    if not verb or not noun:
        return None
    return verb, noun


def _pair_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]} {pair[1]}"


def _rank_bucket(rank: int | None) -> str:
    """Return the stable exact rank-distance bucket for one reference action.

    ``None`` means that the reference action was not present in the recorded
    candidate list.  This helper intentionally does not inspect candidate
    scores or invoke a model; it is purely a projection of recorded ordering.
    """

    if rank is None:
        return "not_in_top_k"
    if rank <= 0:
        # Ranks are validated before this helper is called.  Keep a defensive
        # fallback rather than producing a malformed report if a legacy route
        # is passed through the additive extension path.
        return "not_in_top_k"
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    return "rank_4_plus"


def _bucket_histogram(
    rows: Sequence[Mapping[str, Any]],
    *,
    rank_field: str,
) -> dict[str, int]:
    """Count exact rank buckets, retaining zero-valued buckets for comparability."""

    counts = Counter(
        _rank_bucket(int(row[rank_field]) if row.get(rank_field) is not None else None)
        for row in rows
    )
    return {bucket: int(counts.get(bucket, 0)) for bucket in RANK_BUCKETS}


def _bucket_rates(counts: Mapping[str, Any], *, denominator: int) -> dict[str, float]:
    return {
        bucket: (float(counts.get(bucket, 0)) / denominator if denominator else 0.0)
        for bucket in RANK_BUCKETS
    }


def _exact_rank_histogram(
    rows: Sequence[Mapping[str, Any]],
    *,
    rank_field: str,
    max_rank: int | None = None,
    candidate_count_field: str | None = None,
) -> dict[str, int]:
    """Build a dense rank histogram plus an explicit not-in-top-k bucket.

    The original report's ``rank_histogram`` is intentionally sparse to preserve
    its v1 shape.  Consumers that need exact rank comparisons should use this
    dense additive histogram: every rank from one through the observed maximum
    is present, including zero counts.
    """

    observed: list[int] = [
        int(row[rank_field])
        for row in rows
        if row.get(rank_field) is not None and int(row[rank_field]) > 0
    ]
    if max_rank is None:
        if candidate_count_field is not None:
            candidate_counts = [
                int(row[candidate_count_field])
                for row in rows
                if row.get(candidate_count_field) is not None
                and int(row[candidate_count_field]) > 0
            ]
            max_rank = max(candidate_counts, default=0)
        else:
            max_rank = max(observed, default=0)
    max_rank = max(0, int(max_rank))
    counts = Counter(f"rank_{rank}" for rank in observed)
    result = {f"rank_{rank}": int(counts.get(f"rank_{rank}", 0)) for rank in range(1, max_rank + 1)}
    result["not_in_top_k"] = int(sum(row.get(rank_field) is None for row in rows))
    return result


def _hard_negative_relation(
    candidate: tuple[str, str],
    references: Sequence[tuple[str, str]],
) -> str:
    """Classify a wrong candidate against the closest surrogate action.

    We only expose lexical relation labels (same verb, same noun, or different
    action).  This is deliberately not an ontology/Mapper operation and does
    not claim visual equivalence.
    """

    if not references:
        return "different_action"
    # Prefer a reference sharing the most lexical components.  Ties are broken
    # canonically by the normalised pair tuple, making reports reproducible.
    reference = max(
        references,
        key=lambda pair: (
            int(candidate[0] == pair[0]) + int(candidate[1] == pair[1]),
            tuple(reversed(pair)),
        ),
    )
    same_verb = candidate[0] == reference[0]
    same_noun = candidate[1] == reference[1]
    if same_verb and not same_noun:
        return "same_verb"
    if same_noun and not same_verb:
        return "same_noun"
    # Identical candidates are excluded by route validation.  Treat a defensive
    # duplicate as a different action rather than allowing a false negative.
    return "different_action"


def _float(value: object, *, field: str, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ProductionWemmRankErrorAnalysisError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ProductionWemmRankErrorAnalysisError(f"{field} must be finite")
    return result


def _int(value: object, *, field: str) -> int:
    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ProductionWemmRankErrorAnalysisError(f"{field} must be an integer") from exc
    if result <= 0:
        raise ProductionWemmRankErrorAnalysisError(f"{field} must be positive")
    return result


def _validate_input(report: Mapping[str, Any]) -> None:
    if report.get("format") != INPUT_FORMAT:
        raise ProductionWemmRankErrorAnalysisError(f"input format must be {INPUT_FORMAT!r}")
    if report.get("quality_claim") is not False:
        raise ProductionWemmRankErrorAnalysisError(
            "comparison report must explicitly have quality_claim=false"
        )
    if report.get("production_eligible") is not False:
        raise ProductionWemmRankErrorAnalysisError(
            "comparison report must remain production_eligible=false"
        )
    if report.get("official_quality_status") != "NOT_MEASURED":
        raise ProductionWemmRankErrorAnalysisError(
            "official quality status must remain NOT_MEASURED"
        )
    if report.get("official_gold_status") != "NOT_ESTABLISHED":
        raise ProductionWemmRankErrorAnalysisError(
            "official gold status must remain NOT_ESTABLISHED"
        )
    if report.get("status") != "SURROGATE_ONLY":
        raise ProductionWemmRankErrorAnalysisError("comparison report must remain SURROGATE_ONLY")
    forbidden = _contains_true_flag(
        report,
        "model_invoked",
        "media_decoded",
        "gold_read",
        "gold_written",
        "ontology_modified",
        "mapper_modified",
        "training_invoked",
        "heldout_100_opened",
        "hash_or_digest_computed",
        "hash_or_sha_used",
    )
    if forbidden is not None:
        raise ProductionWemmRankErrorAnalysisError(
            f"comparison report declares forbidden work: {forbidden}"
        )
    reference = report.get("reference")
    if not isinstance(reference, Mapping):
        raise ProductionWemmRankErrorAnalysisError("comparison report lacks reference metadata")
    if reference.get("status") != "INDEPENDENT_SURROGATE_REFERENCE":
        raise ProductionWemmRankErrorAnalysisError(
            "rank analysis requires the independent Terra surrogate reference"
        )
    binding = report.get("source_binding")
    if isinstance(binding, Mapping) and binding.get("status") == "CONFLICT":
        raise ProductionWemmRankErrorAnalysisError("source binding is conflicting")
    routes = _mapping(report.get("routes"), field="routes")
    if not routes:
        raise ProductionWemmRankErrorAnalysisError("routes must not be empty")
    for name, raw_route in routes.items():
        route = _mapping(raw_route, field=f"routes[{name}]")
        provenance = route.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ProductionWemmRankErrorAnalysisError(
                f"routes[{name}] lacks production provenance"
            )
        if provenance.get("epic_ontology_used") is not False:
            raise ProductionWemmRankErrorAnalysisError(
                f"routes[{name}] does not declare epic_ontology_used=false"
            )
        if provenance.get("mapper_used") is not False:
            raise ProductionWemmRankErrorAnalysisError(
                f"routes[{name}] does not declare mapper_used=false"
            )


def _ordered_candidates(route: Mapping[str, Any], *, window_id: str) -> list[dict[str, Any]]:
    raw_windows = _route_per_window_mapping(route, field="route.per_window")
    raw_window = raw_windows.get(window_id)
    if raw_window is None:
        return []
    window = _mapping(raw_window, field=f"{window_id}.per_window")
    raw_candidates = _sequence(window.get("candidates", []), field=f"{window_id}.candidates")
    candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_ranks: set[int] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _mapping(raw_candidate, field=f"{window_id}.candidates[{index}]")
        parsed_pair = _pair(candidate.get("pair"))
        if parsed_pair is None:
            raise ProductionWemmRankErrorAnalysisError(
                f"{window_id}.candidates[{index}].pair must be a verb/noun pair"
            )
        rank = _int(
            candidate.get("rank", candidate.get("recorded_rank", index + 1)),
            field=f"{window_id}.candidate.rank",
        )
        if rank in seen_ranks:
            raise ProductionWemmRankErrorAnalysisError(
                f"{window_id} contains duplicate candidate rank {rank}"
            )
        if parsed_pair in seen_pairs:
            raise ProductionWemmRankErrorAnalysisError(
                f"{window_id} contains duplicate candidate pair {_pair_key(parsed_pair)!r}"
            )
        seen_ranks.add(rank)
        seen_pairs.add(parsed_pair)
        score = _float(candidate.get("score"), field=f"{window_id}.candidate.score")
        candidates.append(
            {
                "recorded_rank": rank,
                "pair": list(parsed_pair),
                "action": _pair_key(parsed_pair),
                "label_id": candidate.get("label_id"),
                "label_text": candidate.get("label_text"),
                "score": score,
            }
        )
    candidates.sort(key=lambda row: int(row["recorded_rank"]))
    expected_ranks = list(range(1, len(candidates) + 1))
    actual_ranks = [int(candidate["recorded_rank"]) for candidate in candidates]
    if actual_ranks != expected_ranks:
        raise ProductionWemmRankErrorAnalysisError(
            f"{window_id} candidate ranks must be contiguous from 1"
        )
    return candidates


def _reference_pairs(route: Mapping[str, Any], *, window_id: str) -> list[tuple[str, str]]:
    raw_windows = _route_per_window_mapping(route, field="route.per_window")
    raw_window = raw_windows.get(window_id)
    if raw_window is None:
        return []
    window = _mapping(raw_window, field=f"{window_id}.per_window")
    raw_pairs = _sequence(window.get("reference_pairs", []), field=f"{window_id}.reference_pairs")
    result: list[tuple[str, str]] = []
    for index, value in enumerate(raw_pairs):
        parsed = _pair(value)
        if parsed is None:
            raise ProductionWemmRankErrorAnalysisError(
                f"{window_id}.reference_pairs[{index}] is not a pair"
            )
        if parsed not in result:
            result.append(parsed)
    return result


def _error_band(rank: int | None, *, candidate_count: int, top1_hit: bool) -> str:
    if top1_hit:
        return "rank_1"
    if rank is None:
        return "not_in_top_k"
    if rank <= 3:
        return "rank_2_3"
    if rank <= 5:
        return "rank_4_5"
    # Keep the label independent of the current six-label catalog.  A future
    # Terra catalog can therefore still use this report without renaming bins.
    del candidate_count
    return "rank_6_plus"


def _margin_bin(margin: float | None, bins: Sequence[tuple[float, float, str]]) -> str:
    if margin is None:
        return "unknown"
    for lower, upper, label in bins:
        if margin >= lower and margin < upper:
            return label
    # Negative margins are invalid for a sorted score list, but preserve an
    # explicit bucket if a legacy sidecar contains one rather than hiding it.
    if margin < bins[0][0]:
        return "negative"
    return "unknown"


def _camera_consensus_row(
    raw: object,
    *,
    references: Sequence[tuple[str, str]],
    fused_top1: tuple[str, str] | None,
) -> dict[str, Any]:
    """Normalize optional per-window camera consensus diagnostics.

    Vocabulary comparison reports emitted before camera projection simply lack
    this block.  Returning an explicit ``NOT_AVAILABLE`` row keeps old reports
    valid and prevents missing camera evidence from being counted as zero
    agreement.  When compact ``per_camera`` rows are present, summary values
    are recomputed from those rows rather than trusted as semantic labels.
    """

    unavailable = {
        "status": "NOT_AVAILABLE",
        "observed_camera_count": 0,
        "expected_camera_count": None,
        "coverage_fraction": None,
        "consensus_winner": None,
        "consensus_winning_votes": 0,
        "consensus_fraction": None,
        "strict_majority": False,
        "fused_top1_action": _pair_key(fused_top1) if fused_top1 is not None else None,
        "fused_top1_vote_count": 0,
        "fused_top1_vote_fraction": None,
        "winner_matches_reference": None,
        "fused_top1_matches_reference": None,
        "top1_margin_summary": {},
    }
    if not isinstance(raw, Mapping):
        return unavailable

    def optional_float(value: object, field: str) -> float | None:
        try:
            return _float(value, field=field)
        except ProductionWemmRankErrorAnalysisError:
            return None

    status_raw = raw.get("status")
    status = str(status_raw).strip().upper() if status_raw is not None else "AVAILABLE"
    if status not in {"AVAILABLE", "PARTIAL", "SUMMARY_ONLY"}:
        status = "NOT_AVAILABLE"

    # Prefer the compact per-camera rows emitted by the comparison helper.  A
    # summary-only sidecar remains useful for routing metrics, so all summary
    # fields are accepted as a fallback.
    per_camera_raw = raw.get("per_camera")
    per_camera = (
        [item for item in per_camera_raw if isinstance(item, Mapping)]
        if isinstance(per_camera_raw, Sequence)
        and not isinstance(per_camera_raw, (str, bytes, bytearray))
        else []
    )
    top1_actions: list[str] = []
    ranked_by_camera: list[list[str]] = []
    for item in per_camera:
        top1 = _normalise(item.get("top1_action"))
        if top1:
            top1_actions.append(top1)
        ranked_raw = item.get("ranked_actions")
        if isinstance(ranked_raw, Sequence) and not isinstance(ranked_raw, (str, bytes, bytearray)):
            ranked = [_normalise(value) for value in ranked_raw]
            ranked_by_camera.append([value for value in ranked if value])

    observed_count = len(per_camera)
    if observed_count == 0:
        observed_raw = raw.get("observed_camera_count")
        try:
            observed_count = int(observed_raw) if observed_raw is not None else 0
        except (TypeError, ValueError):
            observed_count = 0
    expected_raw = raw.get("expected_camera_count")
    try:
        expected_count = int(expected_raw) if expected_raw is not None else None
    except (TypeError, ValueError):
        expected_count = None
    if expected_count is not None and expected_count <= 0:
        expected_count = None
    coverage = optional_float(raw.get("coverage_fraction"), "camera.coverage_fraction")
    if coverage is None and expected_count:
        coverage = observed_count / expected_count

    vote_counts = Counter(top1_actions)
    ordered_votes = sorted(vote_counts.items(), key=lambda item: (-int(item[1]), item[0]))
    winner = (
        ordered_votes[0][0] if ordered_votes else _normalise(raw.get("consensus_winner")) or None
    )
    winning_votes = (
        int(ordered_votes[0][1])
        if ordered_votes
        else int(raw.get("consensus_winning_votes", 0) or 0)
    )
    consensus_fraction = (
        winning_votes / len(top1_actions)
        if top1_actions
        else optional_float(raw.get("consensus_fraction"), "camera.consensus_fraction")
    )
    strict_majority = (
        bool(winning_votes and observed_count and winning_votes * 2 > observed_count)
        if top1_actions
        else bool(raw.get("strict_majority", False))
    )
    fused_action = _normalise(raw.get("fused_top1_action")) or (
        _pair_key(fused_top1) if fused_top1 is not None else None
    )
    fused_votes = sum(action == fused_action for action in top1_actions) if fused_action else 0
    if not top1_actions:
        try:
            fused_votes = int(raw.get("fused_top1_vote_count", 0) or 0)
        except (TypeError, ValueError):
            fused_votes = 0
    fused_vote_fraction = (
        fused_votes / len(top1_actions)
        if top1_actions
        else optional_float(raw.get("fused_top1_vote_fraction"), "camera.fused_top1_vote_fraction")
    )
    reference_actions = {_pair_key(pair) for pair in references}
    winner_matches = bool(winner and winner in reference_actions) if winner else None
    fused_matches = (
        bool(fused_action and fused_action in reference_actions) if fused_action else None
    )

    # If a sidecar only provided per-camera rankings, derive a compact GT-rank
    # summary for the caller.  This is intentionally a rank projection, not a
    # visual-evidence assertion.
    reference_rank_summary: dict[str, dict[str, Any]] = {}
    for reference in references:
        action = _pair_key(reference)
        ranks: list[int] = []
        for ranked in ranked_by_camera:
            try:
                ranks.append(ranked.index(action) + 1)
            except ValueError:
                continue
        reference_rank_summary[action] = {
            "camera_count": len(ranked_by_camera),
            "found_count": len(ranks),
            "mean_rank": mean(ranks) if ranks else None,
            "median_rank": median(ranks) if ranks else None,
            "min_rank": min(ranks) if ranks else None,
            "top1_votes": sum(rank == 1 for rank in ranks),
            "top1_fraction": (
                sum(rank == 1 for rank in ranks) / len(ranked_by_camera)
                if ranked_by_camera
                else None
            ),
        }

    margin_summary = raw.get("top1_margin_summary")
    if not isinstance(margin_summary, Mapping):
        margin_summary = {}
    return {
        "status": status if observed_count or top1_actions or winner else "NOT_AVAILABLE",
        "observed_camera_count": observed_count,
        "expected_camera_count": expected_count,
        "coverage_fraction": coverage,
        "consensus_winner": winner,
        "consensus_winning_votes": winning_votes,
        "consensus_fraction": consensus_fraction,
        "strict_majority": strict_majority,
        "fused_top1_action": fused_action,
        "fused_top1_vote_count": fused_votes,
        "fused_top1_vote_fraction": fused_vote_fraction,
        "winner_matches_reference": winner_matches,
        "fused_top1_matches_reference": fused_matches,
        "top1_margin_summary": dict(margin_summary),
        "reference_rank_summary": reference_rank_summary,
    }


def _camera_consensus_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    for lower, upper, label in CAMERA_CONSENSUS_BINS:
        if value >= lower and value < upper:
            return label
    return "unknown"


def _camera_consensus_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate camera agreement and its relation to fused rank-1 results."""

    measured = [
        row
        for row in rows
        if isinstance(row.get("camera_consensus"), Mapping)
        and row["camera_consensus"].get("consensus_fraction") is not None
    ]
    if not measured:
        return {
            "status": "NOT_AVAILABLE",
            "unit": "window",
            "windows_total": len(rows),
            "windows_measured": 0,
            "coverage_fraction": {"mean": None, "median": None, "min": None, "max": None},
            "consensus_fraction": {"mean": None, "median": None, "min": None, "max": None},
            "strict_majority_count": 0,
            "strict_majority_rate": None,
            "winner_matches_reference_count": 0,
            "winner_matches_reference_rate": None,
            "fused_top1_agrees_with_consensus_count": 0,
            "fused_top1_agrees_with_consensus_rate": None,
            "bins": [],
        }

    def values(field: str) -> list[float]:
        result: list[float] = []
        for row in measured:
            camera = row["camera_consensus"]
            value = camera.get(field) if isinstance(camera, Mapping) else None
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                result.append(number)
        return result

    def summary(numbers: Sequence[float]) -> dict[str, float | None]:
        if not numbers:
            return {"mean": None, "median": None, "min": None, "max": None}
        ordered = sorted(numbers)
        middle = (
            ordered[len(ordered) // 2]
            if len(ordered) % 2
            else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
        )
        return {
            "mean": sum(ordered) / len(ordered),
            "median": middle,
            "min": ordered[0],
            "max": ordered[-1],
        }

    bins: list[dict[str, Any]] = []
    for _, _, label in CAMERA_CONSENSUS_BINS:
        members = [
            row
            for row in measured
            if _camera_consensus_bin(float(row["camera_consensus"]["consensus_fraction"])) == label
        ]
        rank_hist = Counter(
            f"rank_{row['matching_rank_min']}"
            if row.get("matching_rank_min") is not None
            else "not_in_top_k"
            for row in members
        )
        hits = sum(bool(row.get("top1_hit_any_reference")) for row in members)
        bins.append(
            {
                "label": label,
                "windows": len(members),
                "top1_hits_any_reference": hits,
                "top1_rate_any_reference": hits / len(members) if members else 0.0,
                "strict_majority_count": sum(
                    bool(row["camera_consensus"].get("strict_majority")) for row in members
                ),
                "rank_histogram_min_matching_rank": dict(
                    sorted(
                        rank_hist.items(),
                        key=lambda item: (
                            10**9 if item[0] == "not_in_top_k" else int(item[0].split("_", 1)[1])
                        ),
                    )
                ),
            }
        )
    fractions = values("consensus_fraction")
    coverage = values("coverage_fraction")
    strict_count = sum(bool(row["camera_consensus"].get("strict_majority")) for row in measured)
    winner_matches = sum(
        bool(row["camera_consensus"].get("winner_matches_reference")) for row in measured
    )
    agrees = sum(
        row["camera_consensus"].get("fused_top1_action")
        == row["camera_consensus"].get("consensus_winner")
        for row in measured
    )
    count = len(measured)
    return {
        "status": "AVAILABLE"
        if all(str(row["camera_consensus"].get("status")) == "AVAILABLE" for row in measured)
        else "PARTIAL",
        "unit": "window",
        "windows_total": len(rows),
        "windows_measured": count,
        "coverage_fraction": summary(coverage),
        "consensus_fraction": summary(fractions),
        "strict_majority_count": strict_count,
        "strict_majority_rate": strict_count / count if count else None,
        "winner_matches_reference_count": winner_matches,
        "winner_matches_reference_rate": winner_matches / count if count else None,
        "fused_top1_agrees_with_consensus_count": agrees,
        "fused_top1_agrees_with_consensus_rate": agrees / count if count else None,
        "bins": bins,
    }


def _margin_bin_record(
    label: str,
    lower: float | None,
    upper: float | None,
    rows: Sequence[Mapping[str, Any]],
    *,
    top1_key: str,
    rank_field: str,
) -> dict[str, Any]:
    count = len(rows)
    hits = sum(bool(row.get(top1_key)) for row in rows)
    rank_counts = Counter(
        f"rank_{row[rank_field]}" if row.get(rank_field) is not None else "not_in_top_k"
        for row in rows
    )
    rank_hist = dict(
        sorted(
            rank_counts.items(),
            key=lambda item: (
                10**9 if item[0] == "not_in_top_k" else int(item[0].split("_", 1)[1]),
            ),
        )
    )
    return {
        "label": label,
        "lower": lower,
        "upper": upper,
        "count": count,
        "top1_hits": hits,
        "top1_errors": count - hits,
        "top1_hit_rate": hits / count if count else 0.0,
        "rank_histogram": rank_hist,
    }


def _build_margin_bins(
    rows: Sequence[Mapping[str, Any]],
    bins: Sequence[tuple[float, float, str]],
    *,
    top1_key: str,
    rank_field: str,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("margin_bin", "unknown"))].append(row)
    result: list[dict[str, Any]] = []
    for lower, upper, label in bins:
        result.append(
            _margin_bin_record(
                label,
                lower,
                None if math.isinf(upper) else upper,
                buckets.get(label, ()),
                top1_key=top1_key,
                rank_field=rank_field,
            )
        )
    for label in ("unknown", "negative"):
        if label in buckets:
            result.append(
                _margin_bin_record(
                    label,
                    None,
                    None,
                    buckets[label],
                    top1_key=top1_key,
                    rank_field=rank_field,
                )
            )
    return result


def _top1(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return candidates[0] if candidates else None


def _second(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return candidates[1] if len(candidates) > 1 else None


def _hard_negatives_for_reference(
    candidates: Sequence[Mapping[str, Any]],
    *,
    reference: tuple[str, str],
    references: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Return recorded wrong candidates for one surrogate reference action.

    Every candidate that is not one of the reference actions is retained.  The
    ``outranks_reference`` flag identifies the candidates responsible for a
    rank error, while retaining lower-ranked negatives is useful for active
    learning and nearest-neighbour review.
    """

    matching_rank = next(
        (
            int(candidate["recorded_rank"])
            for candidate in candidates
            if tuple(candidate["pair"]) == reference
        ),
        None,
    )
    negatives: list[dict[str, Any]] = []
    reference_set = set(references)
    for candidate in candidates:
        pair = tuple(candidate["pair"])
        if pair in reference_set:
            continue
        rank = int(candidate["recorded_rank"])
        relation = _hard_negative_relation(pair, references)
        negatives.append(
            {
                "action": str(candidate["action"]),
                "pair": list(pair),
                "rank": rank,
                "score": candidate.get("score"),
                "relation_to_reference": relation,
                "hard_negative_type": relation,
                "is_top1": rank == 1,
                "reference_rank": matching_rank,
                "rank_distance_to_reference": (
                    rank - matching_rank if matching_rank is not None else None
                ),
                "outranks_reference": (matching_rank is None or rank < matching_rank),
            }
        )
    return negatives


def _attach_hard_negative_fields(
    row: dict[str, Any],
    *,
    candidates: Sequence[Mapping[str, Any]],
    references: Sequence[tuple[str, str]],
    reference: tuple[str, str],
) -> None:
    negatives = _hard_negatives_for_reference(
        candidates,
        reference=reference,
        references=references,
    )
    row["hard_negatives"] = negatives
    row["hard_negative_count"] = len(negatives)
    row["hard_negative_outranking_count"] = sum(
        bool(item["outranks_reference"]) for item in negatives
    )
    row["top1_hard_negative"] = next(
        (item for item in negatives if bool(item.get("is_top1"))),
        None,
    )
    row["top1_hard_negative_type"] = (
        row["top1_hard_negative"].get("hard_negative_type")
        if isinstance(row.get("top1_hard_negative"), Mapping)
        else None
    )


def _window_rows_for_route(
    route: Mapping[str, Any],
    *,
    window_ids: Sequence[str],
    margin_bins: Sequence[tuple[float, float, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for window_id in window_ids:
        references = _reference_pairs(route, window_id=window_id)
        candidates = _ordered_candidates(route, window_id=window_id)
        top1 = _top1(candidates)
        second = _second(candidates)
        raw_window_map = _route_per_window_mapping(route, field="route.per_window")
        raw_window = raw_window_map.get(window_id)
        raw_window_mapping = raw_window if isinstance(raw_window, Mapping) else {}
        top1_score = _float(top1.get("score"), field=f"{window_id}.top1.score") if top1 else None
        second_score = (
            _float(second.get("score"), field=f"{window_id}.top2.score") if second else None
        )
        margin = (
            top1_score - second_score
            if top1_score is not None and second_score is not None
            else None
        )
        margin_label = _margin_bin(margin, margin_bins)
        top1_pair = tuple(top1["pair"]) if top1 is not None else None
        top1_hit_any = top1_pair is not None and top1_pair in references
        camera_consensus = _camera_consensus_row(
            raw_window_mapping.get("camera_diagnostics"),
            references=references,
            fused_top1=top1_pair,
        )
        action_rows: list[dict[str, Any]] = []
        for action_index, reference in enumerate(references):
            matching = next(
                (candidate for candidate in candidates if tuple(candidate["pair"]) == reference),
                None,
            )
            rank = int(matching["recorded_rank"]) if matching is not None else None
            gt_score = matching.get("score") if matching is not None else None
            top1_is_gt = top1_pair == reference
            row = {
                "window_id": window_id,
                "action_index": action_index,
                "gt_action": _pair_key(reference),
                "gt_pair": list(reference),
                "reference_action_count": len(references),
                "split_reference": len(references) > 1,
                "gt_rank": rank,
                "rank_distance_from_top1": rank - 1 if rank is not None else None,
                "gt_score": gt_score,
                "top1_action": top1.get("action") if top1 else None,
                "top1_pair": list(top1_pair) if top1_pair is not None else None,
                "top1_score": top1_score,
                "top1_is_gt": top1_is_gt,
                "score_gap_top1_minus_gt": (
                    top1_score - float(gt_score)
                    if top1_score is not None and gt_score is not None
                    else None
                ),
                "top1_top2_margin": margin,
                "margin_bin": margin_label,
                "error_band": _error_band(
                    rank, candidate_count=len(candidates), top1_hit=top1_is_gt
                ),
                # Additive exact rank bucket.  Keep ``error_band`` above for
                # consumers of the original v1 report.
                "rank_bucket": _rank_bucket(rank),
                "rank_error_bucket": _rank_bucket(rank),
                "candidate_count": len(candidates),
                "camera_consensus_fraction": camera_consensus.get("consensus_fraction"),
                "camera_consensus_winner": camera_consensus.get("consensus_winner"),
                "camera_consensus_matches_gt": camera_consensus.get("consensus_winner")
                == _pair_key(reference)
                if camera_consensus.get("consensus_winner")
                else None,
                "camera_gt_rank_min": (
                    camera_consensus.get("reference_rank_summary", {})
                    .get(_pair_key(reference), {})
                    .get("min_rank")
                    if isinstance(camera_consensus.get("reference_rank_summary"), Mapping)
                    else None
                ),
                "camera_gt_rank_mean": (
                    camera_consensus.get("reference_rank_summary", {})
                    .get(_pair_key(reference), {})
                    .get("mean_rank")
                    if isinstance(camera_consensus.get("reference_rank_summary"), Mapping)
                    else None
                ),
                "camera_gt_rank_median": (
                    camera_consensus.get("reference_rank_summary", {})
                    .get(_pair_key(reference), {})
                    .get("median_rank")
                    if isinstance(camera_consensus.get("reference_rank_summary"), Mapping)
                    else None
                ),
            }
            _attach_hard_negative_fields(
                row,
                candidates=candidates,
                references=references,
                reference=reference,
            )
            action_rows.append(row)
            actions.append(row)
        min_rank = min(
            (int(row["gt_rank"]) for row in action_rows if row["gt_rank"] is not None),
            default=None,
        )
        windows.append(
            {
                "window_id": window_id,
                "reference_actions": [_pair_key(pair) for pair in references],
                "reference_pairs": [list(pair) for pair in references],
                "reference_action_count": len(references),
                "split_reference": len(references) > 1,
                "candidate_count": len(candidates),
                "top1_action": top1.get("action") if top1 else None,
                "top1_pair": list(top1_pair) if top1_pair is not None else None,
                "top1_score": top1_score,
                "top2_action": second.get("action") if second else None,
                "top2_score": second_score,
                "top1_top2_margin": margin,
                "margin_bin": margin_label,
                "top1_hit_any_reference": top1_hit_any,
                "matching_rank_min": min_rank,
                "error_band": _error_band(
                    min_rank, candidate_count=len(candidates), top1_hit=top1_hit_any
                ),
                "rank_bucket": _rank_bucket(min_rank),
                "rank_error_bucket": _rank_bucket(min_rank),
                "camera_consensus": camera_consensus,
                "action_rows": action_rows,
                "candidates": [
                    {
                        "rank": int(candidate["recorded_rank"]),
                        "action": candidate["action"],
                        "pair": candidate["pair"],
                        "label_id": candidate.get("label_id"),
                        "label_text": candidate.get("label_text"),
                        "score": candidate.get("score"),
                    }
                    for candidate in candidates
                ],
            }
        )
        # Window-level hard negatives are the union of negatives attached to
        # each reference action.  Preserve the first reference relation for a
        # candidate and expose all applicable surrogate actions separately.
        window = windows[-1]
        negatives_by_action: dict[str, dict[str, Any]] = {}
        for action_row in action_rows:
            for negative in action_row.get("hard_negatives", []):
                key = str(negative.get("action"))
                entry = negatives_by_action.get(key)
                if entry is None:
                    entry = dict(negative)
                    entry["reference_actions"] = [str(action_row["gt_action"])]
                    negatives_by_action[key] = entry
                elif str(action_row["gt_action"]) not in entry["reference_actions"]:
                    entry["reference_actions"].append(str(action_row["gt_action"]))
        window["hard_negatives"] = list(
            sorted(
                negatives_by_action.values(),
                key=lambda item: (int(item.get("rank", 10**9)), str(item.get("action"))),
            )
        )
        window["hard_negative_count"] = len(window["hard_negatives"])
        window["top1_hard_negative_type"] = (
            window["hard_negatives"][0].get("hard_negative_type")
            if window["hard_negatives"] and int(window["hard_negatives"][0].get("rank", 10**9)) == 1
            else None
        )
    return windows, actions


def _histogram(rows: Sequence[Mapping[str, Any]], *, field: str) -> dict[str, int]:
    counts = Counter(
        str(row[field]) if row.get(field) is not None else "not_in_top_k" for row in rows
    )
    order = ("rank_1", "rank_2_3", "rank_4_5", "rank_6_plus", "not_in_top_k")
    result = {key: int(counts.get(key, 0)) for key in order}
    result.update({key: int(value) for key, value in sorted(counts.items()) if key not in result})
    return result


def _rank_histogram(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        f"rank_{row['gt_rank']}" if row.get("gt_rank") is not None else "not_in_top_k"
        for row in rows
    )
    ranks = sorted(
        (key for key in counts if key.startswith("rank_")),
        key=lambda key: int(key.split("_", 1)[1]),
    )
    result = {key: int(counts[key]) for key in ranks}
    result["not_in_top_k"] = int(counts.get("not_in_top_k", 0))
    return result


def _confusions(rows: Sequence[Mapping[str, Any]], *, include_split: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row.get("top1_is_gt")):
            continue
        if not include_split and int(row.get("reference_action_count", 1) or 1) > 1:
            continue
        predicted = str(row.get("top1_action") or "<no-candidate>")
        target = str(row.get("gt_action") or "<missing>")
        grouped[(predicted, target)].append(row)
    result: list[dict[str, Any]] = []
    for (predicted, target), members in grouped.items():
        margins = [
            float(row["top1_top2_margin"])
            for row in members
            if row.get("top1_top2_margin") is not None
        ]
        ranks = [int(row["gt_rank"]) for row in members if row.get("gt_rank") is not None]
        result.append(
            {
                "predicted_top1": predicted,
                "ground_truth": target,
                "count": len(members),
                "window_ids": [str(row["window_id"]) for row in members],
                "mean_gt_rank": mean(ranks) if ranks else None,
                "mean_margin": mean(margins) if margins else None,
            }
        )
    result.sort(
        key=lambda row: (-int(row["count"]), str(row["predicted_top1"]), str(row["ground_truth"]))
    )
    return result


def _confusion_clusters(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_split: bool,
) -> list[dict[str, Any]]:
    """Group directed Top-1 confusions into deterministic lexical clusters.

    A cluster is a connected component of the undirected graph induced by
    ``predicted_top1 -> ground_truth`` edges.  Edges retain their direction and
    aggregate counts, so callers can inspect both recurring pair errors and the
    broader hard-negative family they belong to.
    """

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row.get("top1_is_gt")):
            continue
        if not include_split and int(row.get("reference_action_count", 1) or 1) > 1:
            continue
        predicted = str(row.get("top1_action") or "<no-candidate>")
        target = str(row.get("gt_action") or "<missing>")
        grouped[(predicted, target)].append(row)
    if not grouped:
        return []

    adjacency: dict[str, set[str]] = defaultdict(set)
    for predicted, target in grouped:
        adjacency[predicted].add(target)
        adjacency[target].add(predicted)

    components: list[tuple[str, ...]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            unseen.discard(node)
            stack.extend(sorted(adjacency.get(node, ()), reverse=True))
        components.append(tuple(sorted(seen)))
    components.sort(key=lambda nodes: (nodes[0], len(nodes), nodes))

    result: list[dict[str, Any]] = []
    for index, nodes in enumerate(components, 1):
        node_set = set(nodes)
        edge_rows: list[dict[str, Any]] = []
        member_rows: list[Mapping[str, Any]] = []
        for (predicted, target), members in sorted(grouped.items()):
            if predicted not in node_set or target not in node_set:
                continue
            member_rows.extend(members)
            margins = [
                float(row["top1_top2_margin"])
                for row in members
                if row.get("top1_top2_margin") is not None
            ]
            ranks = [int(row["gt_rank"]) for row in members if row.get("gt_rank") is not None]
            edge_rows.append(
                {
                    "predicted_top1": predicted,
                    "ground_truth": target,
                    "count": len(members),
                    "window_ids": sorted({str(row["window_id"]) for row in members}),
                    "rank_bucket_histogram": _bucket_histogram(members, rank_field="gt_rank"),
                    "mean_gt_rank": mean(ranks) if ranks else None,
                    "mean_margin": mean(margins) if margins else None,
                }
            )
        relation_counts = Counter(
            str(row.get("top1_hard_negative_type") or "unknown") for row in member_rows
        )
        result.append(
            {
                "cluster_id": f"confusion_cluster_{index}",
                "nodes": list(nodes),
                "actions": list(nodes),
                "edge_count": len(edge_rows),
                "pair_count": sum(int(edge["count"]) for edge in edge_rows),
                "window_count": len({str(row["window_id"]) for row in member_rows}),
                "window_ids": sorted({str(row["window_id"]) for row in member_rows}),
                "edges": edge_rows,
                "pairs": edge_rows,
                "rank_bucket_histogram": _bucket_histogram(member_rows, rank_field="gt_rank"),
                "hard_negative_relation_histogram": {
                    key: int(value) for key, value in sorted(relation_counts.items())
                },
            }
        )
    result.sort(
        key=lambda cluster: (
            -int(cluster["pair_count"]),
            str(cluster["nodes"]),
        )
    )
    # IDs are assigned after sorting so they remain stable when input route
    # insertion order changes.
    for index, cluster in enumerate(result, 1):
        cluster["cluster_id"] = f"confusion_cluster_{index}"
    return result


def _hard_negative_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate lexical hard-negative diagnostics from action rows."""

    negatives: list[Mapping[str, Any]] = []
    for row in rows:
        for item in row.get("hard_negatives", ()):
            if isinstance(item, Mapping):
                negatives.append(item)
    relation_counts = Counter(
        str(item.get("hard_negative_type") or "unknown") for item in negatives
    )
    outranking = [item for item in negatives if bool(item.get("outranks_reference"))]
    top1 = [item for item in negatives if bool(item.get("is_top1"))]
    by_bucket: dict[str, int] = {bucket: 0 for bucket in RANK_BUCKETS}
    for row in rows:
        bucket = str(row.get("rank_bucket", "not_in_top_k"))
        if bucket not in by_bucket:
            bucket = "not_in_top_k"
        by_bucket[bucket] += int(row.get("hard_negative_count", 0) or 0)
    examples: list[dict[str, Any]] = sorted(
        (
            {
                "window_id": str(row.get("window_id")),
                "ground_truth": str(row.get("gt_action")),
                "candidate": dict(item),
            }
            for row in rows
            for item in row.get("hard_negatives", ())
            if isinstance(item, Mapping)
        ),
        key=lambda item: (
            not bool(cast(Mapping[str, Any], item["candidate"]).get("outranks_reference")),
            int(cast(Mapping[str, Any], item["candidate"]).get("rank", 10**9)),
            str(item["window_id"]),
            str(cast(Mapping[str, Any], item["candidate"]).get("action")),
        ),
    )
    return {
        "unit": "wrong recorded candidate per surrogate action",
        "action_rows": len(rows),
        "rows_with_hard_negatives": sum(bool(row.get("hard_negatives")) for row in rows),
        "hard_negative_count": len(negatives),
        "outranking_hard_negative_count": len(outranking),
        "top1_hard_negative_count": len(top1),
        "relation_histogram": {
            relation: int(relation_counts.get(relation, 0))
            for relation in (*HARD_NEGATIVE_RELATIONS, "unknown")
            if relation_counts.get(relation, 0) or relation != "unknown"
        },
        "by_reference_rank_bucket": by_bucket,
        "examples": examples[:20],
    }


def _action_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["gt_action"])].append(row)
    result: list[dict[str, Any]] = []
    for action, members in sorted(grouped.items()):
        ranks = [int(row["gt_rank"]) for row in members if row.get("gt_rank") is not None]
        top1 = sum(bool(row.get("top1_is_gt")) for row in members)
        top3 = sum(row.get("gt_rank") is not None and int(row["gt_rank"]) <= 3 for row in members)
        top5 = sum(row.get("gt_rank") is not None and int(row["gt_rank"]) <= 5 for row in members)
        predictions = Counter(
            str(row.get("top1_action") or "<no-candidate>")
            for row in members
            if not bool(row.get("top1_is_gt"))
        )
        result.append(
            {
                "ground_truth": action,
                "instances": len(members),
                "top1_hits": top1,
                "top1_rate": top1 / len(members) if members else 0.0,
                "top3_hits": top3,
                "top5_hits": top5,
                "not_in_top_k": sum(row.get("gt_rank") is None for row in members),
                "mean_rank": mean(ranks) if ranks else None,
                "median_rank": median(ranks) if ranks else None,
                "rank_bucket_histogram": _bucket_histogram(members, rank_field="gt_rank"),
                "exact_rank_histogram": _exact_rank_histogram(
                    members,
                    rank_field="gt_rank",
                    candidate_count_field="candidate_count",
                ),
                "hard_negative_count": sum(
                    int(row.get("hard_negative_count", 0) or 0) for row in members
                ),
                "top1_hard_negative_types": dict(
                    Counter(
                        str(row.get("top1_hard_negative_type"))
                        for row in members
                        if row.get("top1_hard_negative_type") is not None
                    )
                ),
                "common_wrong_top1": [
                    {"action": prediction, "count": count}
                    for prediction, count in predictions.most_common()
                ],
            }
        )
    return result


def _metric_value(route: Mapping[str, Any], *path: str) -> float | None:
    value: object = route
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if value is None:
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_margin_bins(
    margin_bins: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    if (
        not isinstance(margin_bins, Sequence)
        or isinstance(margin_bins, (str, bytes, bytearray))
        or not margin_bins
    ):
        raise ProductionWemmRankErrorAnalysisError("margin_bins must be a sequence")
    parsed_bins: list[tuple[float, float, str]] = []
    previous_upper = 0.0
    for index, raw_bin in enumerate(margin_bins):
        if not isinstance(raw_bin, Sequence) or len(raw_bin) != 3:
            raise ProductionWemmRankErrorAnalysisError(
                f"margin_bins[{index}] must contain lower, upper, label"
            )
        try:
            lower = float(raw_bin[0])
            upper = float(raw_bin[1])
        except (TypeError, ValueError) as exc:
            raise ProductionWemmRankErrorAnalysisError(
                f"margin_bins[{index}] bounds are invalid"
            ) from exc
        label = str(raw_bin[2])
        if not label or lower < previous_upper or upper <= lower:
            raise ProductionWemmRankErrorAnalysisError(
                f"margin_bins[{index}] must be ordered, non-empty intervals"
            )
        if not math.isfinite(lower) or (not math.isfinite(upper) and not math.isinf(upper)):
            raise ProductionWemmRankErrorAnalysisError(f"margin_bins[{index}] bounds are invalid")
        parsed_bins.append((lower, upper, label))
        previous_upper = upper
    return parsed_bins


def _route_per_window_mapping(route: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    """Normalize input-comparison maps and v1 rank-report row lists."""

    raw = route.get("per_window")
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        result: dict[str, Any] = {}
        for index, item in enumerate(raw):
            row = _mapping(item, field=f"{field}[{index}]")
            window_id = row.get("window_id")
            if window_id is None:
                raise ProductionWemmRankErrorAnalysisError(f"{field}[{index}] lacks window_id")
            if str(window_id) in result:
                raise ProductionWemmRankErrorAnalysisError(
                    f"{field} contains duplicate window_id {window_id!r}"
                )
            result[str(window_id)] = row
        return result
    raise ProductionWemmRankErrorAnalysisError(f"{field} must be an object or row array")


def _compare_rank_values(left: object, right: object) -> str:
    """Compare ranks where a missing rank is worse than any observed rank."""

    left_rank = int(cast(Any, left)) if left is not None else None
    right_rank = int(cast(Any, right)) if right is not None else None
    if left_rank is None and right_rank is None:
        return "tie"
    if left_rank is None:
        return "right"
    if right_rank is None:
        return "left"
    if left_rank < right_rank:
        return "left"
    if right_rank < left_rank:
        return "right"
    return "tie"


def _variant_comparison(
    route_reports: Mapping[str, Mapping[str, Any]],
    *,
    variant_names: Sequence[str],
    window_ids: Sequence[str],
) -> dict[str, Any]:
    """Build paired per-window/per-action comparisons across prototype variants."""

    if not variant_names:
        return {
            "baseline_variant": None,
            "variants": [],
            "pairwise": {},
            "per_window": [],
            "per_action": [],
            "best_variant_by_metric": {},
        }
    baseline = "canonical" if "canonical" in variant_names else str(variant_names[0])
    per_window_by_variant: dict[str, dict[str, Mapping[str, Any]]] = {}
    per_action_by_variant: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {}
    for variant in variant_names:
        route = route_reports[variant]
        per_window_by_variant[variant] = {
            str(row.get("window_id")): row
            for row in route.get("per_window", ())
            if isinstance(row, Mapping)
        }
        per_action_by_variant[variant] = {
            (str(row.get("window_id")), int(row.get("action_index", 0))): row
            for row in route.get("per_action_instance", ())
            if isinstance(row, Mapping)
        }

    paired_windows: list[dict[str, Any]] = []
    for window_id in window_ids:
        values: dict[str, dict[str, Any]] = {}
        for variant in variant_names:
            row = per_window_by_variant[variant].get(str(window_id), {})
            values[variant] = {
                "matching_rank_min": row.get("matching_rank_min"),
                "rank_bucket": row.get("rank_bucket", _rank_bucket(row.get("matching_rank_min"))),
                "top1_action": row.get("top1_action"),
                "top1_hit_any_reference": bool(row.get("top1_hit_any_reference")),
                "top1_top2_margin": row.get("top1_top2_margin"),
                "candidate_count": row.get("candidate_count"),
            }
        best_rank = min(
            (
                int(value["matching_rank_min"])
                for value in values.values()
                if value.get("matching_rank_min") is not None
            ),
            default=None,
        )
        winners = [
            variant
            for variant, value in values.items()
            if value.get("matching_rank_min") == best_rank and best_rank is not None
        ]
        paired_windows.append(
            {
                "window_id": str(window_id),
                "variants": values,
                "best_rank": best_rank,
                "best_rank_variants": winners,
                "top1_hit_variants": [
                    variant for variant, value in values.items() if value["top1_hit_any_reference"]
                ],
            }
        )

    paired_actions: list[dict[str, Any]] = []
    action_keys = sorted(
        {key for variant in variant_names for key in per_action_by_variant[variant]}
    )
    for window_id, action_index in action_keys:
        action_values: dict[str, dict[str, Any]] = {}
        ground_truth: str | None = None
        for variant in variant_names:
            row = per_action_by_variant[variant].get((window_id, action_index), {})
            if ground_truth is None and row.get("gt_action") is not None:
                ground_truth = str(row.get("gt_action"))
            action_values[variant] = {
                "gt_rank": row.get("gt_rank"),
                "rank_distance_from_top1": row.get("rank_distance_from_top1"),
                "rank_bucket": row.get("rank_bucket", _rank_bucket(row.get("gt_rank"))),
                "top1_action": row.get("top1_action"),
                "top1_is_gt": bool(row.get("top1_is_gt")),
                "top1_top2_margin": row.get("top1_top2_margin"),
                "hard_negative_count": row.get("hard_negative_count", 0),
            }
        best_rank = min(
            (
                int(value["gt_rank"])
                for value in action_values.values()
                if value.get("gt_rank") is not None
            ),
            default=None,
        )
        winners = [
            variant
            for variant, value in action_values.items()
            if value.get("gt_rank") == best_rank and best_rank is not None
        ]
        paired_actions.append(
            {
                "window_id": window_id,
                "action_index": action_index,
                "ground_truth": ground_truth,
                "variants": action_values,
                "best_rank": best_rank,
                "best_rank_variants": winners,
                "top1_hit_variants": [
                    variant for variant, value in action_values.items() if value["top1_is_gt"]
                ],
            }
        )

    pairwise: dict[str, Any] = {}
    for left, right in combinations(variant_names, 2):
        window_left_wins = window_right_wins = window_ties = 0
        action_left_wins = action_right_wins = action_ties = 0
        top1_left_wins = top1_right_wins = top1_ties = 0
        bucket_delta = {bucket: 0 for bucket in RANK_BUCKETS}
        for row in paired_windows:
            comparison = _compare_rank_values(
                row["variants"][left].get("matching_rank_min"),
                row["variants"][right].get("matching_rank_min"),
            )
            if comparison == "left":
                window_left_wins += 1
            elif comparison == "right":
                window_right_wins += 1
            else:
                window_ties += 1
        for row in paired_actions:
            comparison = _compare_rank_values(
                row["variants"][left].get("gt_rank"),
                row["variants"][right].get("gt_rank"),
            )
            if comparison == "left":
                action_left_wins += 1
            elif comparison == "right":
                action_right_wins += 1
            else:
                action_ties += 1
            left_hit = bool(row["variants"][left].get("top1_is_gt"))
            right_hit = bool(row["variants"][right].get("top1_is_gt"))
            if left_hit and not right_hit:
                top1_left_wins += 1
            elif right_hit and not left_hit:
                top1_right_wins += 1
            else:
                top1_ties += 1
            left_bucket = str(row["variants"][left].get("rank_bucket"))
            right_bucket = str(row["variants"][right].get("rank_bucket"))
            if left_bucket in bucket_delta:
                bucket_delta[left_bucket] += 1
            if right_bucket in bucket_delta:
                bucket_delta[right_bucket] -= 1
        left_action_top1 = _metric_value(route_reports[left], "action_level", "top1_rate")
        right_action_top1 = _metric_value(route_reports[right], "action_level", "top1_rate")
        left_mrr = _metric_value(route_reports[left], "action_level", "mrr")
        right_mrr = _metric_value(route_reports[right], "action_level", "mrr")
        left_window_top1 = _metric_value(
            route_reports[left], "window_level", "top1_rate_any_reference"
        )
        right_window_top1 = _metric_value(
            route_reports[right], "window_level", "top1_rate_any_reference"
        )
        key = f"{left}_vs_{right}"
        pairwise[key] = {
            "left_variant": left,
            "right_variant": right,
            "window_count": len(paired_windows),
            "action_count": len(paired_actions),
            "window_rank_wins": {
                left: window_left_wins,
                right: window_right_wins,
                "ties": window_ties,
            },
            "action_rank_wins": {
                left: action_left_wins,
                right: action_right_wins,
                "ties": action_ties,
            },
            "action_top1_wins": {
                left: top1_left_wins,
                right: top1_right_wins,
                "ties": top1_ties,
            },
            "delta": {
                "window_top1_rate": (
                    right_window_top1 - left_window_top1
                    if left_window_top1 is not None and right_window_top1 is not None
                    else None
                ),
                "action_top1_rate": (
                    right_action_top1 - left_action_top1
                    if left_action_top1 is not None and right_action_top1 is not None
                    else None
                ),
                "mrr": (
                    right_mrr - left_mrr if left_mrr is not None and right_mrr is not None else None
                ),
            },
            "rank_bucket_delta_right_minus_left": bucket_delta,
            "per_window": [
                row for row in paired_windows if row["variants"][left] != row["variants"][right]
            ],
            "per_action": [
                row for row in paired_actions if row["variants"][left] != row["variants"][right]
            ],
        }
        # A second spelling keeps ad-hoc consumers that prefer a double
        # underscore key interoperable without changing the canonical key.
        pairwise[f"{left}__{right}"] = pairwise[key]

    def best(metric_path: tuple[str, ...]) -> str | None:
        values = {
            variant: _metric_value(route_reports[variant], *metric_path)
            for variant in variant_names
        }
        finite = {name: value for name, value in values.items() if value is not None}
        if not finite:
            return None
        return max(
            finite,
            key=lambda name: (
                float(finite[name]),
                -list(variant_names).index(name),
            ),
        )

    return {
        "baseline_variant": baseline,
        "variants": list(variant_names),
        "pairwise": pairwise,
        "per_window": paired_windows,
        "per_action": paired_actions,
        "best_variant_by_metric": {
            "window_top1_rate": best(("window_level", "top1_rate_any_reference")),
            "action_top1_rate": best(("action_level", "top1_rate")),
            "action_recall_at_3": best(("action_level", "recall_at_k", "3")),
            "action_recall_at_5": best(("action_level", "recall_at_k", "5")),
            "mrr": best(("action_level", "mrr")),
        },
    }


def _route_report(
    variant: str,
    route: Mapping[str, Any],
    *,
    window_ids: Sequence[str],
    margin_bins: Sequence[tuple[float, float, str]],
) -> dict[str, Any]:
    window_rows, action_rows = _window_rows_for_route(
        route, window_ids=window_ids, margin_bins=margin_bins
    )
    action_count = len(action_rows)
    window_count = len(window_rows)
    found = [int(row["gt_rank"]) for row in action_rows if row.get("gt_rank") is not None]

    def rate(rows: Sequence[Mapping[str, Any]], k: int, field: str = "gt_rank") -> float:
        return (
            sum(row.get(field) is not None and int(row[field]) <= k for row in rows) / len(rows)
            if rows
            else 0.0
        )

    action_top1_hits = sum(bool(row["top1_is_gt"]) for row in action_rows)
    window_top1_hits = sum(bool(row["top1_hit_any_reference"]) for row in window_rows)
    action_error_rows = [row for row in action_rows if not bool(row["top1_is_gt"])]
    rank_hist = _rank_histogram(action_rows)
    exact_rank_hist = _exact_rank_histogram(
        action_rows,
        rank_field="gt_rank",
        candidate_count_field="candidate_count",
    )
    window_exact_hist = _exact_rank_histogram(
        window_rows,
        rank_field="matching_rank_min",
        candidate_count_field="candidate_count",
    )
    rank_buckets = _bucket_histogram(action_rows, rank_field="gt_rank")
    window_bucket_hist = _bucket_histogram(window_rows, rank_field="matching_rank_min")
    error_band = _histogram(action_rows, field="error_band")
    margins = [
        float(row["top1_top2_margin"])
        for row in window_rows
        if row.get("top1_top2_margin") is not None
    ]
    confusion_all = _confusions(action_rows, include_split=True)
    confusion_single = _confusions(action_rows, include_split=False)
    confusion_clusters_all = _confusion_clusters(action_rows, include_split=True)
    confusion_clusters_single = _confusion_clusters(action_rows, include_split=False)
    # A split action row has no ``reference_action_count`` field itself; use the
    # window map to make the split exclusion explicit and deterministic.
    split_windows = {
        str(row["window_id"]) for row in window_rows if bool(row.get("split_reference"))
    }
    single_action_rows = [row for row in action_rows if str(row["window_id"]) not in split_windows]
    return {
        "variant": variant,
        "window_count": window_count,
        "action_instance_count": action_count,
        "split_window_count": len(split_windows),
        "window_level": {
            "top1_hits_any_reference": window_top1_hits,
            "top1_rate_any_reference": window_top1_hits / window_count if window_count else 0.0,
            "recall_at_k_any_reference": {
                str(k): sum(
                    row.get("matching_rank_min") is not None and int(row["matching_rank_min"]) <= k
                    for row in window_rows
                )
                / window_count
                if window_count
                else 0.0
                for k in (1, 3, 5, 10)
            },
            "rank_histogram_min_matching_rank": {
                f"rank_{k}": sum(row.get("matching_rank_min") == k for row in window_rows)
                for k in sorted(
                    {
                        int(row["matching_rank_min"])
                        for row in window_rows
                        if row.get("matching_rank_min") is not None
                    }
                )
            }
            | {"not_in_top_k": sum(row.get("matching_rank_min") is None for row in window_rows)},
            "exact_rank_histogram": window_exact_hist,
            "rank_histogram_exact": window_exact_hist,
            "rank_bucket_histogram": window_bucket_hist,
            "rank_bucket_rates": _bucket_rates(window_bucket_hist, denominator=window_count),
        },
        "action_level": {
            "top1_hits": action_top1_hits,
            "top1_rate": action_top1_hits / action_count if action_count else 0.0,
            "recall_at_k": {str(k): rate(action_rows, k) for k in (1, 3, 5, 10)},
            "mrr": sum(1.0 / rank for rank in found) / action_count if action_count else 0.0,
            "mean_rank": mean(found) if found else None,
            "median_rank": median(found) if found else None,
            "rank_histogram": rank_hist,
            "exact_rank_histogram": exact_rank_hist,
            "rank_histogram_exact": exact_rank_hist,
            "rank_distance_histogram": {
                str(distance): sum(
                    row.get("rank_distance_from_top1") == distance for row in action_rows
                )
                for distance in sorted(
                    {
                        int(row["rank_distance_from_top1"])
                        for row in action_rows
                        if row.get("rank_distance_from_top1") is not None
                    }
                )
            },
            "rank_bucket_histogram": rank_buckets,
            "rank_buckets": rank_buckets,
            "rank_bucket_rates": _bucket_rates(rank_buckets, denominator=action_count),
            "error_band_histogram": error_band,
            "top1_error_count": len(action_error_rows),
            "near_miss_rank_2_3_count": sum(
                row.get("error_band") == "rank_2_3" for row in action_error_rows
            ),
            "rank_4_plus_error_count": sum(
                row.get("error_band") in {"rank_4_5", "rank_6_plus"} for row in action_error_rows
            ),
            "not_in_top_k_count": sum(
                row.get("error_band") == "not_in_top_k" for row in action_error_rows
            ),
            "near_miss_fraction_of_errors": (
                sum(row.get("error_band") == "rank_2_3" for row in action_error_rows)
                / len(action_error_rows)
                if action_error_rows
                else 0.0
            ),
            "rank_4_plus_fraction_of_errors": (
                sum(
                    row.get("error_band") in {"rank_4_5", "rank_6_plus"}
                    for row in action_error_rows
                )
                / len(action_error_rows)
                if action_error_rows
                else 0.0
            ),
            "not_in_top_k_fraction_of_errors": (
                sum(row.get("error_band") == "not_in_top_k" for row in action_error_rows)
                / len(action_error_rows)
                if action_error_rows
                else 0.0
            ),
            "single_reference_top1_rate": (
                sum(bool(row["top1_is_gt"]) for row in single_action_rows) / len(single_action_rows)
                if single_action_rows
                else 0.0
            ),
        },
        "margin_bins": {
            "unit": "window",
            "bins": _build_margin_bins(
                window_rows,
                margin_bins,
                top1_key="top1_hit_any_reference",
                rank_field="matching_rank_min",
            ),
            "median_margin": median(margins) if margins else None,
            "mean_margin": mean(margins) if margins else None,
        },
        "action_margin_bins": {
            "unit": "action_instance",
            "bins": _build_margin_bins(
                action_rows,
                margin_bins,
                top1_key="top1_is_gt",
                rank_field="gt_rank",
            ),
        },
        # Camera consensus is retained as a routing diagnostic only.  It is
        # intentionally separate from rank/margin metrics and never upgrades a
        # surrogate result to production quality.
        "camera_consensus": _camera_consensus_summary(window_rows),
        "confusion_pairs": {
            "action_level_including_split": confusion_all,
            "single_reference_windows_only": confusion_single,
        },
        "confusion_clusters": {
            "action_level_including_split": confusion_clusters_all,
            "single_reference_windows_only": confusion_clusters_single,
        },
        "pairwise_confusion_clusters": {
            "action_level_including_split": confusion_clusters_all,
            "single_reference_windows_only": confusion_clusters_single,
        },
        "hard_negative_analysis": _hard_negative_summary(action_rows),
        "exact_rank_histogram": {
            "window_level": window_exact_hist,
            "action_level": exact_rank_hist,
        },
        "rank_histogram_exact": {
            "window_level": window_exact_hist,
            "action_level": exact_rank_hist,
        },
        "rank_bucket_histogram": {
            "window_level": window_bucket_hist,
            "action_level": rank_buckets,
        },
        "per_action": _action_summary(action_rows),
        "per_window": window_rows,
        "per_action_instance": action_rows,
        "source_route_metrics": route.get("metrics", {}),
    }


def analyze_production_wemm_rank_errors(
    comparison: Mapping[str, Any] | str | Path,
    *,
    margin_bins: Sequence[tuple[float, float, str]] = DEFAULT_MARGIN_BINS,
) -> dict[str, Any]:
    """Build a rank-distance report from an existing production comparison."""

    report = _load(comparison)
    if report.get("format") == ANALYSIS_FORMAT:
        return extend_production_wemm_rank_error_analysis(report, margin_bins=margin_bins)
    _validate_input(report)
    parsed_bins = _parse_margin_bins(margin_bins)
    routes = _mapping(report.get("routes"), field="routes")
    variant_names = [str(name) for name in routes]
    first_route = _mapping(routes[variant_names[0]], field=f"routes[{variant_names[0]}]")
    first_windows = _route_per_window_mapping(first_route, field="routes.per_window")
    window_ids = [str(window_id) for window_id in first_windows]
    if not window_ids:
        raise ProductionWemmRankErrorAnalysisError("comparison has no eligible windows")
    # All label variants must be aligned to the same surrogate windows.  A
    # missing row is a data issue, not a reason to silently change denominators.
    alignment: dict[str, Any] = {"window_ids": window_ids, "mismatches": []}
    for variant in variant_names:
        route = _mapping(routes[variant], field=f"routes[{variant}]")
        ids = list(_route_per_window_mapping(route, field="route.per_window"))
        if ids != window_ids:
            alignment["mismatches"].append(
                {"variant": variant, "expected": window_ids, "actual": ids}
            )
    if alignment["mismatches"]:
        raise ProductionWemmRankErrorAnalysisError(
            "routes are not aligned to the same eligible surrogate windows"
        )
    route_reports = {
        variant: _route_report(
            variant,
            _mapping(routes[variant], field=f"routes[{variant}]"),
            window_ids=window_ids,
            margin_bins=parsed_bins,
        )
        for variant in variant_names
    }
    prototype_comparison = _variant_comparison(
        route_reports,
        variant_names=variant_names,
        window_ids=window_ids,
    )
    reference = report.get("reference")
    reference_map = dict(reference) if isinstance(reference, Mapping) else {}
    controls = {
        "model_invoked": False,
        "media_decoded": False,
        "gold_read": False,
        "gold_written": False,
        "ontology_modified": False,
        "mapper_modified": False,
        "training_invoked": False,
        "heldout_100_opened": False,
        "hash_or_digest_computed": False,
    }
    return {
        "format": ANALYSIS_FORMAT,
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "SURROGATE_ONLY",
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "production_eligible": False,
        "input": {
            "format": report.get("format"),
            "source_report": str(comparison) if isinstance(comparison, (str, Path)) else None,
            "reference": reference_map,
            "source_binding": report.get("source_binding", {}),
            "eligible_window_ids": window_ids,
            "eligible_window_count": len(window_ids),
            "excluded_window_count": int(reference_map.get("excluded_window_count", 0) or 0),
            "excluded_windows_not_in_comparison": True,
        },
        "analysis": {
            "extension_format": ANALYSIS_EXTENSION_FORMAT,
            "rank_buckets": list(RANK_BUCKETS),
            "hard_negative_relations": list(HARD_NEGATIVE_RELATIONS),
            "camera_consensus_bins": [
                {
                    "lower": lower,
                    "upper": None if math.isinf(upper) else upper,
                    "label": label,
                }
                for lower, upper, label in CAMERA_CONSENSUS_BINS
            ],
            "rank_bucket_definitions": {
                "rank_1": "reference action is ranked first",
                "rank_2": "reference action is ranked second",
                "rank_3": "reference action is ranked third",
                "rank_4_plus": "reference action is ranked fourth or lower",
                "not_in_top_k": "reference action is absent from the recorded candidate list",
            },
            "unit_definitions": {
                "window_level": (
                    "one row per eligible window; split is a hit if any reference "
                    "action is in the rank list"
                ),
                "action_level": (
                    "one row per Terra-surrogate reference action; split windows "
                    "contribute one row per action"
                ),
                "rank_distance_from_top1": "gt_rank - 1; zero means the action is ranked first",
                "not_in_top_k": "reference action is absent from the recorded candidate list",
            },
            "margin_bins": [
                {"lower": lower, "upper": None if math.isinf(upper) else upper, "label": label}
                for lower, upper, label in parsed_bins
            ],
            "alignment": alignment,
        },
        "routes": route_reports,
        # Paired comparison is intentionally separate from per-route metrics so
        # prototype variants can be compared without changing their recorded
        # candidate lists or introducing a new model/ontology contract.
        "prototype_variant_comparison": prototype_comparison,
        "variant_comparison": prototype_comparison,
        "hard_negative_analysis": {
            "by_variant": {
                variant: route.get("hard_negative_analysis", {})
                for variant, route in route_reports.items()
            },
            "relations_are_lexical_only": True,
        },
        "controls": controls,
        "limitations": [
            (
                "All ranks and errors are measured against an independent Terra "
                "surrogate, not official gold."
            ),
            (
                "The comparison report contains eight eligible windows; abstain "
                "windows are excluded from scored ranks."
            ),
            (
                "The split window is reported per action and also as a window-level "
                "any-reference hit."
            ),
            "Margins are cosine-score differences, not calibrated probabilities.",
            (
                "Rank distance describes retrieval ordering and does not establish "
                "visual evidence correctness."
            ),
            (
                "Camera consensus is a post-hoc top-1 identity projection from the "
                "sidecar; it is not independent semantic or visual evidence."
            ),
        ],
    }


def extend_production_wemm_rank_error_analysis(
    rank_report: Mapping[str, Any] | str | Path,
    *,
    margin_bins: Sequence[tuple[float, float, str]] = DEFAULT_MARGIN_BINS,
) -> dict[str, Any]:
    """Add the P13 rank/hard-negative extension to an existing v1 report.

    P13 sidecars were emitted before the additive exact-bucket fields existed.
    This helper makes those artifacts forward-compatible without requiring the
    source comparison sidecar (and without rerunning a model or media).
    """

    source = _load(rank_report)
    if source.get("format") != ANALYSIS_FORMAT:
        raise ProductionWemmRankErrorAnalysisError(
            f"rank report format must be {ANALYSIS_FORMAT!r}"
        )
    for key, expected in (
        ("status", "SURROGATE_ONLY"),
        ("official_quality_status", "NOT_MEASURED"),
        ("official_gold_status", "NOT_ESTABLISHED"),
        ("quality_claim", False),
        ("production_eligible", False),
    ):
        if source.get(key) != expected:
            raise ProductionWemmRankErrorAnalysisError(
                f"rank report {key} must remain {expected!r}"
            )
    parsed_bins = _parse_margin_bins(margin_bins)
    routes = _mapping(source.get("routes"), field="rank_report.routes")
    if not routes:
        raise ProductionWemmRankErrorAnalysisError("rank report routes must not be empty")
    first_name = str(next(iter(routes)))
    first_route = _mapping(routes[first_name], field=f"rank_report.routes[{first_name}]")
    first_windows = _route_per_window_mapping(first_route, field="rank_report.routes.per_window")
    window_ids = [str(window_id) for window_id in first_windows]
    if not window_ids:
        raise ProductionWemmRankErrorAnalysisError("rank report has no eligible windows")
    route_reports = {
        str(name): _route_report(
            str(name),
            _mapping(raw_route, field=f"rank_report.routes[{name}]"),
            window_ids=window_ids,
            margin_bins=parsed_bins,
        )
        for name, raw_route in routes.items()
    }
    alignment = {"window_ids": window_ids, "mismatches": []}
    prototype_comparison = _variant_comparison(
        route_reports,
        variant_names=list(route_reports),
        window_ids=window_ids,
    )
    # Make a JSON-friendly copy without importing or hashing arbitrary source
    # material.  Nested values are replaced only at report-owned keys.
    result = dict(source)
    result["routes"] = route_reports
    result["prototype_variant_comparison"] = prototype_comparison
    result["variant_comparison"] = prototype_comparison
    analysis = dict(_mapping(source.get("analysis", {}), field="rank_report.analysis"))
    analysis.update(
        {
            "extension_format": ANALYSIS_EXTENSION_FORMAT,
            "rank_buckets": list(RANK_BUCKETS),
            "hard_negative_relations": list(HARD_NEGATIVE_RELATIONS),
            "camera_consensus_bins": [
                {
                    "lower": lower,
                    "upper": None if math.isinf(upper) else upper,
                    "label": label,
                }
                for lower, upper, label in CAMERA_CONSENSUS_BINS
            ],
            "rank_bucket_definitions": {
                "rank_1": "reference action is ranked first",
                "rank_2": "reference action is ranked second",
                "rank_3": "reference action is ranked third",
                "rank_4_plus": "reference action is ranked fourth or lower",
                "not_in_top_k": "reference action is absent from the recorded candidate list",
            },
            "margin_bins": [
                {"lower": lower, "upper": None if math.isinf(upper) else upper, "label": label}
                for lower, upper, label in parsed_bins
            ],
            "alignment": alignment,
        }
    )
    result["analysis"] = analysis
    controls = dict(_mapping(source.get("controls", {}), field="rank_report.controls"))
    controls.setdefault("model_invoked", False)
    controls.setdefault("media_decoded", False)
    controls.setdefault("gold_read", False)
    controls.setdefault("gold_written", False)
    controls.setdefault("ontology_modified", False)
    controls.setdefault("mapper_modified", False)
    controls.setdefault("training_invoked", False)
    controls.setdefault("heldout_100_opened", False)
    controls.setdefault("hash_or_digest_computed", False)
    result["controls"] = controls
    return result


def _percent(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(cast(Any, value)):.1%}"


def _number(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(cast(Any, value)):.3f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable rank-distance report."""

    lines = [
        "# WeMM production rank-1 error distance analysis",
        "",
        (
            "> **SURROGATE_ONLY.** Terra independent review is a development "
            "reference; official production quality remains `NOT_MEASURED`."
        ),
        "",
        (
            f"Eligible windows: **{report['input']['eligible_window_count']}**; "
            f"excluded windows: **{report['input']['excluded_window_count']}**."
        ),
        (
            "Ranks are computed from the already-recorded Terra-scoped WeMM candidate "
            "lists; no model or media was run while producing this report."
        ),
        (
            "The summary below reports action-level Top-K and MRR; window-level "
            "any-reference Top-K values are labeled separately. Candidate-list "
            "cardinality is shown below because a cutoff can equal the full "
            "recorded list."
        ),
        "",
        "## Variant summary",
        "",
        (
            "| Variant | Window Top-1 (any ref) | Action Top-1 | Action R@3 | "
            "Action R@5 | Action MRR | Median GT rank | Median margin |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    routes = report["routes"]
    for variant, route in routes.items():
        window = route["window_level"]
        action = route["action_level"]
        lines.append(
            f"| {variant} | {_percent(window['top1_rate_any_reference'])} | "
            f"{_percent(action['top1_rate'])} | {_percent(action['recall_at_k']['3'])} | "
            f"{_percent(action['recall_at_k']['5'])} | {_number(action['mrr'])} | "
            f"{_number(action['median_rank'])} | {_number(route['margin_bins']['median_margin'])} |"
        )
    lines += ["", "## Candidate-list cardinality", ""]
    lines.append(
        "| Variant | Min candidates/window | Max candidates/window | Unique counts | "
        "Full-list cutoffs |"
    )
    lines.append("|---|---:|---:|---|---|")
    for variant, route in routes.items():
        raw_rows = route.get("per_window", [])
        if isinstance(raw_rows, Mapping):
            raw_rows = raw_rows.values()
        counts = sorted(
            {
                int(row["candidate_count"])
                for row in raw_rows
                if isinstance(row, Mapping)
                and row.get("candidate_count") is not None
                and int(row["candidate_count"]) >= 0
            }
        )
        if counts:
            maximum = max(counts)
            cutoffs = ", ".join(f"K={k}" for k in (1, 3, 5, 10) if maximum <= k) or "none"
            unique_text = ", ".join(str(value) for value in counts)
            lines.append(f"| {variant} | {min(counts)} | {maximum} | {unique_text} | {cutoffs} |")
        else:
            lines.append(f"| {variant} | - | - | - | - |")
    lines.append(
        "R@K is full-list coverage whenever the maximum recorded candidate count "
        "is at most K; it is not recall over a larger action catalog."
    )
    lines += ["", "## Action-level rank histogram", ""]
    labels = sorted(
        {
            label
            for route in routes.values()
            for label in route["action_level"]["rank_histogram"]
            if label != "not_in_top_k"
        },
        key=lambda value: int(value.split("_", 1)[1]),
    )
    labels.append("not_in_top_k")
    lines.append("| Variant | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    for variant, route in routes.items():
        hist = route["action_level"]["rank_histogram"]
        values = " | ".join(str(hist.get(label, 0)) for label in labels)
        lines.append(f"| {variant} | {values} |")
    # The additive exact buckets make the rank distance decision auditable
    # without forcing consumers to reverse-engineer the legacy error bands.
    lines += ["", "## Exact rank buckets", ""]
    lines.append("| Variant | Rank 1 | Rank 2 | Rank 3 | Rank 4+ | Not in Top-K |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant, route in routes.items():
        buckets = route["action_level"].get("rank_bucket_histogram", {})
        lines.append(
            f"| {variant} | {buckets.get('rank_1', 0)} | {buckets.get('rank_2', 0)} | "
            f"{buckets.get('rank_3', 0)} | {buckets.get('rank_4_plus', 0)} | "
            f"{buckets.get('not_in_top_k', 0)} |"
        )
    lines += ["", "## Rank-1 error distance", ""]
    lines.append("| Variant | Top-1 errors | Rank 2-3 | Rank 4-5/6+ | Not in Top-K |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant, route in routes.items():
        action = route["action_level"]
        lines.append(
            f"| {variant} | {action['top1_error_count']} | {action['near_miss_rank_2_3_count']} | "
            f"{action['rank_4_plus_error_count']} | {action['not_in_top_k_count']} |"
        )
    lines += ["", "## Most frequent confusion pairs (single-reference windows)", ""]
    lines.append(
        "| Variant | Predicted Top-1 -> surrogate action | Count | Mean GT rank | "
        "Mean margin | Windows |"
    )
    lines.append("|---|---|---:|---:|---:|---|")
    for variant, route in routes.items():
        pairs = route["confusion_pairs"]["single_reference_windows_only"]
        if not pairs:
            lines.append(f"| {variant} | (none) | 0 | - | - | - |")
        for pair in pairs[:8]:
            lines.append(
                f"| {variant} | {pair['predicted_top1']} -> {pair['ground_truth']} | "
                f"{pair['count']} | {_number(pair['mean_gt_rank'])} | "
                f"{_number(pair['mean_margin'])} | {', '.join(pair['window_ids'])} |"
            )
    lines += ["", "## Margin bins (window-level)", ""]
    lines.append("| Variant | Bin | Windows | Top-1 hits | Top-1 hit rate | Rank histogram |")
    lines.append("|---|---|---:|---:|---:|---|")
    for variant, route in routes.items():
        for bucket in route["margin_bins"]["bins"]:
            lines.append(
                f"| {variant} | {bucket['label']} | {bucket['count']} | {bucket['top1_hits']} | "
                f"{_percent(bucket['top1_hit_rate'])} | {bucket['rank_histogram']} |"
            )
    lines += ["", "## Camera-consensus diagnostics", ""]
    lines.append(
        "Camera agreement is copied from per-camera WeMM rankings and is a "
        "routing signal, not independent semantic evidence."
    )
    lines.append(
        "| Variant | Measured windows | Mean consensus | Strict majority | "
        "Consensus winner matches reference | Fused Top-1 agrees | Status |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for variant, route in routes.items():
        camera = route.get("camera_consensus", {})
        if not isinstance(camera, Mapping):
            camera = {}
        consensus_block = camera.get("consensus_fraction")
        consensus_mean = (
            consensus_block.get("mean") if isinstance(consensus_block, Mapping) else None
        )
        lines.append(
            f"| {variant} | {camera.get('windows_measured', 0)} | "
            f"{_percent(consensus_mean)} | "
            f"{camera.get('strict_majority_count', 0)}/"
            f"{camera.get('windows_measured', 0)} | "
            f"{camera.get('winner_matches_reference_count', 0)}/"
            f"{camera.get('windows_measured', 0)} | "
            f"{camera.get('fused_top1_agrees_with_consensus_count', 0)}/"
            f"{camera.get('windows_measured', 0)} | {camera.get('status', 'NOT_AVAILABLE')} |"
        )
    for variant, route in routes.items():
        camera = route.get("camera_consensus", {})
        bins = camera.get("bins", []) if isinstance(camera, Mapping) else []
        if not bins:
            continue
        lines.append("")
        lines.append(f"**{variant} consensus bins:**")
        lines.append(
            ", ".join(
                f"{bucket.get('label')}: {bucket.get('windows')} window(s)"
                for bucket in bins
                if isinstance(bucket, Mapping)
            )
        )
    lines += ["", "## Per-window / per-action detail (canonical)", ""]
    canonical = routes.get("canonical")
    if canonical is not None:
        lines.append(
            "| Window | Surrogate action | GT rank | Distance | Top-1 | Margin | Error band |"
        )
        lines.append("|---|---|---:|---:|---|---:|---|")
        for row in canonical["per_action_instance"]:
            distance = row["rank_distance_from_top1"]
            distance_text = distance if distance is not None else "-"
            lines.append(
                f"| {row['window_id']} | {row['gt_action']} | {row['gt_rank'] or '-'} | "
                f"{distance_text} | "
                f"{row['top1_action'] or '-'} | "
                f"{_number(row['top1_top2_margin'])} | {row['error_band']} |"
            )
    lines += ["", "## Lexical hard-negative diagnostics", ""]
    lines.append(
        "These are structural same-verb/same-noun comparisons, not claims that a "
        "competing action is visually present."
    )
    lines.append(
        "| Variant | Action rows | Hard negatives | O outranking reference | "
        "Top-1 hard negatives | Relations |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for variant, route in routes.items():
        hard = route.get("hard_negative_analysis", {})
        relations = hard.get("relation_histogram", {})
        relation_text = ", ".join(
            f"{name}={relations.get(name, 0)}"
            for name in ("same_verb", "same_noun", "different_action")
        )
        lines.append(
            f"| {variant} | {hard.get('action_rows', 0)} | "
            f"{hard.get('hard_negative_count', 0)} | "
            f"{hard.get('outranking_hard_negative_count', 0)} | "
            f"{hard.get('top1_hard_negative_count', 0)} | {relation_text} |"
        )
    lines += ["", "## Confusion clusters (single-reference windows)", ""]
    lines.append(
        "Clusters connect recurring directed Top-1 confusions; they are ranking "
        "diagnostics, not visual ground truth."
    )
    for variant, route in routes.items():
        clusters = route.get("confusion_clusters", {}).get("single_reference_windows_only", [])
        if not clusters:
            lines.append(f"- **{variant}**: no recurring confusion cluster.")
            continue
        for cluster in clusters[:3]:
            nodes = ", ".join(str(node) for node in cluster.get("nodes", []))
            lines.append(
                f"- **{variant}** ({cluster.get('pair_count', 0)} pair(s), "
                f"{cluster.get('window_count', 0)} window(s)): {nodes}"
            )
    comparison = report.get("prototype_variant_comparison")
    if isinstance(comparison, Mapping):
        lines += ["", "## Prototype-variant paired comparison", ""]
        best = comparison.get("best_variant_by_metric", {})
        if isinstance(best, Mapping):
            lines.append(
                "Best recorded variant by metric (surrogate only): "
                + ", ".join(
                    f"{metric}={value}" for metric, value in best.items() if value is not None
                )
            )
        pairwise = comparison.get("pairwise", {})
        if isinstance(pairwise, Mapping):
            lines.append("| Pair | Action-rank wins (left/right/tie) | MRR delta (right-left) |")
            lines.append("|---|---:|---:|")
            emitted: set[str] = set()
            for key, pair in pairwise.items():
                if "__" in str(key) or not isinstance(pair, Mapping):
                    continue
                left = str(pair.get("left_variant", ""))
                right = str(pair.get("right_variant", ""))
                if not left or not right or f"{left}__{right}" in emitted:
                    continue
                wins = pair.get("action_rank_wins", {})
                if not isinstance(wins, Mapping):
                    wins = {}
                delta = pair.get("delta", {})
                if not isinstance(delta, Mapping):
                    delta = {}
                lines.append(
                    f"| {left} vs {right} | {wins.get(left, 0)}/{wins.get(right, 0)}/"
                    f"{wins.get('ties', 0)} | {_number(delta.get('mrr'))} |"
                )
                emitted.add(f"{left}__{right}")
    lines += ["", "## Interpretation", ""]
    for variant, route in routes.items():
        action = route["action_level"]
        errors = int(action["top1_error_count"])
        near = int(action["near_miss_rank_2_3_count"])
        far = int(action["rank_4_plus_error_count"])
        if errors:
            lines.append(
                f"- **{variant}**: {near}/{errors} Top-1 errors are rank 2-3 near misses "
                f"({near / errors:.1%}); {far}/{errors} are rank 4 or lower."
            )
        else:
            lines.append(f"- **{variant}**: no action-level Top-1 errors in this surrogate cohort.")
    lines += [
        "",
        (
            "This is a retrieval-order diagnostic, not a production accuracy claim. "
            "A high Top-K rank or small margin does not prove that the model observed "
            "the correct action evidence; obtain independent source-bound gold before "
            "P6 qualification."
        ),
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ANALYSIS_EXTENSION_FORMAT",
    "ANALYSIS_FORMAT",
    "CAMERA_CONSENSUS_BINS",
    "DEFAULT_MARGIN_BINS",
    "HARD_NEGATIVE_RELATIONS",
    "RANK_BUCKETS",
    "ProductionWemmRankErrorAnalysisError",
    "analyze_production_wemm_rank_errors",
    "extend_production_wemm_rank_error_analysis",
    "render_markdown",
]
