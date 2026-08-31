"""Post-hoc fusion diagnostics for WeMM EPIC label prototypes.

This module deliberately consumes an already-written WeMM retrieval sidecar.
It never loads a model, decodes media, reads an evaluation manifest, calls the
Mapper, or changes an ontology.  The three prototype surfaces (``canonical``,
``verb_noun`` and ``natural``) are treated as independent ranked lists over
the same action keys and are combined with reciprocal-rank fusion (RRF).

The current EPIC sidecars retain only their top-ten rows for each prototype.
Consequently, score fusion is enabled only when every prototype contains a
finite score for every catalog action.  On truncated sidecars the diagnostic
is intentionally rank-only and calls the result a *top-k-union* experiment;
an absent action is not silently assigned a model score.

Hard-negative summaries are structural label comparisons (same verb, same
noun, or either), not claims that a competing action is visually present.
Everything returned by this module is exploratory and non-production.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Final, Literal, TypeGuard

FUSION_VERSION: Final = "wemm-label-prototype-fusion-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
PROTOTYPES: Final = ("canonical", "verb_noun", "natural")
MODES: Final = ("visual", "text", "hybrid")
METRIC_KS: Final = (1, 3, 5, 10)
HARD_NEGATIVE_KINDS: Final = ("same_verb", "same_noun", "either")
RANK_SCORE_FIELDS: Final = {
    "visual": "visual_score",
    "text": "text_score",
    "hybrid": "fused_score",
}

FusionMethod = Literal["rank", "score", "auto"]


class WemmLabelPrototypeFusionError(ValueError):
    """Raised when a source sidecar cannot be audited safely."""


def _is_sequence(value: object) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _integer(value: object, *, field: str, non_negative: bool = False) -> int:
    if isinstance(value, bool):
        raise WemmLabelPrototypeFusionError(f"{field} must be an integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        result = int(value.strip())
    else:
        raise WemmLabelPrototypeFusionError(f"{field} must be an integer")
    if non_negative and result < 0:
        raise WemmLabelPrototypeFusionError(f"{field} must be non-negative")
    return result


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, (bool, str)) or value is None:
        raise WemmLabelPrototypeFusionError(f"{field} must be a finite number")
    if not isinstance(value, Real):
        # Decimal/numpy-like values may not register as Real; float() remains
        # safe here because this helper never imports or invokes a runtime.
        try:
            converted = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError) as exc:
            raise WemmLabelPrototypeFusionError(f"{field} must be a finite number") from exc
    else:
        converted = float(value)
    if not math.isfinite(converted):
        raise WemmLabelPrototypeFusionError(f"{field} must be a finite number")
    return converted


def _pair(value: object, *, field: str = "action_key") -> tuple[int, int]:
    """Parse one JSON action pair without accepting arbitrary identifiers."""

    if isinstance(value, Mapping):
        if "action_key" in value:
            return _pair(value["action_key"], field=field)
        if "joint_action" in value:
            return _pair(value["joint_action"], field=field)
        if "verb_id" in value and "noun_id" in value:
            return (
                _integer(value["verb_id"], field=f"{field}.verb_id", non_negative=True),
                _integer(value["noun_id"], field=f"{field}.noun_id", non_negative=True),
            )
        raise WemmLabelPrototypeFusionError(f"{field} is missing an action pair")
    if not _is_sequence(value) or len(value) != 2:
        raise WemmLabelPrototypeFusionError(f"{field} must contain exactly two IDs")
    return (
        _integer(value[0], field=f"{field}[0]", non_negative=True),
        _integer(value[1], field=f"{field}[1]", non_negative=True),
    )


def _json_pair(pair: tuple[int, int]) -> list[int]:
    return [pair[0], pair[1]]


def _action_sort_key(pair: tuple[int, int]) -> tuple[int, int]:
    return pair


def _row_id(delta: Mapping[str, Any], index: int) -> str:
    raw = delta.get("id")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    return f"row-{index}"


def _catalog_size(report: Mapping[str, Any]) -> int:
    input_block = report.get("input")
    if isinstance(input_block, Mapping):
        raw = input_block.get("catalog_size")
        if raw is not None:
            return _integer(raw, field="input.catalog_size", non_negative=True)
    labels = report.get("labels")
    if _is_sequence(labels):
        return len(labels)
    return 0


def _catalog_actions(report: Mapping[str, Any]) -> set[tuple[int, int]]:
    labels = report.get("labels")
    if labels is None:
        return set()
    if not _is_sequence(labels):
        raise WemmLabelPrototypeFusionError("labels must be an array")
    actions: set[tuple[int, int]] = set()
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise WemmLabelPrototypeFusionError(f"labels[{index}] must be an object")
        action = _pair(label.get("action_key"), field=f"labels[{index}].action_key")
        if action in actions:
            raise WemmLabelPrototypeFusionError(f"labels contain duplicate action {action}")
        actions.add(action)
    return actions


def _case_deltas(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        raw = report["results"]["canonical"]["metrics"]["case_deltas"]
    except (KeyError, TypeError) as exc:
        raise WemmLabelPrototypeFusionError(
            "source report is missing canonical metrics.case_deltas"
        ) from exc
    if not _is_sequence(raw):
        raise WemmLabelPrototypeFusionError("metrics.case_deltas must be an array")
    # Newer runner reports may place the canonical stable key in the ordinal
    # ``input.row_input_audit`` block.  Prefer that key exactly as the existing
    # summarizer does; older sidecars only carry ``case_deltas[*].id`` (or the
    # row-{index} fallback).
    input_block = report.get("input")
    audit_rows = input_block.get("row_input_audit") if isinstance(input_block, Mapping) else ()
    if not _is_sequence(audit_rows):
        audit_rows = ()
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise WemmLabelPrototypeFusionError(f"case_deltas[{index}] must be an object")
        gt = _pair(item.get("ground_truth"), field=f"case_deltas[{index}].ground_truth")
        stable_id: str | None = None
        if index < len(audit_rows) and isinstance(audit_rows[index], Mapping):
            raw_key = audit_rows[index].get("row_key")
            if raw_key is not None and str(raw_key).strip():
                stable_id = str(raw_key).strip()
        result.append({"id": stable_id or _row_id(item, index), "ground_truth": gt})
    if not result:
        raise WemmLabelPrototypeFusionError("source report has no evaluation rows")
    return result


def _ranking_map(report: Mapping[str, Any], prototype: str, mode: str) -> Mapping[str, Any]:
    try:
        rows = report["results"][prototype]["rankings"][mode]
    except (KeyError, TypeError) as exc:
        raise WemmLabelPrototypeFusionError(
            f"source report is missing rankings for {prototype}/{mode}"
        ) from exc
    if not isinstance(rows, Mapping):
        raise WemmLabelPrototypeFusionError(f"rankings for {prototype}/{mode} must be an object")
    return rows


def _find_row(ranking_rows: Mapping[str, Any], row_id: str, index: int) -> Sequence[Any]:
    value = ranking_rows.get(row_id)
    if value is None:
        value = ranking_rows.get(f"row-{index}")
    if value is None:
        return ()
    if not _is_sequence(value):
        raise WemmLabelPrototypeFusionError(f"ranking row {row_id} must be an array")
    return value


def _normalise_ranking(raw_rows: Sequence[Any], *, row_id: str, mode: str) -> list[dict[str, Any]]:
    """Copy the source top-k ranking into a small immutable-ish audit shape."""

    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise WemmLabelPrototypeFusionError(
                f"ranking row {row_id} item {index} must be an object"
            )
        action = _pair(raw.get("action_key"), field=f"{row_id}[{index}].action_key")
        if action in seen:
            raise WemmLabelPrototypeFusionError(
                f"ranking row {row_id} contains duplicate action {action}"
            )
        seen.add(action)
        rank = _integer(raw.get("rank", index + 1), field=f"{row_id}[{index}].rank")
        if rank <= 0:
            raise WemmLabelPrototypeFusionError(f"{row_id}[{index}].rank must be positive")
        score_field = RANK_SCORE_FIELDS[mode]
        raw_score = raw.get(score_field)
        score: float | None
        if raw_score is None:
            score = None
        else:
            score = _finite(raw_score, field=f"{row_id}[{index}].{score_field}")
        output.append(
            {
                "action_key": action,
                "rank": rank,
                "score": score,
                "verb_id": raw.get("verb_id"),
                "noun_id": raw.get("noun_id"),
                "verb_key": raw.get("verb_key"),
                "noun_key": raw.get("noun_key"),
                "label_text": raw.get("label_text"),
            }
        )
    return output


def _normalised_rows(
    report: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Return ``mode -> prototype -> row-id -> copied ranking``."""

    rows: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for mode in MODES:
        mode_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for prototype in PROTOTYPES:
            source_rows = _ranking_map(report, prototype, mode)
            prototype_rows: dict[str, list[dict[str, Any]]] = {}
            for index, case in enumerate(cases):
                row_id = str(case["id"])
                prototype_rows[row_id] = _normalise_ranking(
                    _find_row(source_rows, row_id, index), row_id=row_id, mode=mode
                )
            mode_rows[prototype] = prototype_rows
        rows[mode] = mode_rows
    return rows


def _ranking_depths(
    rows: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> dict[str, Any]:
    depths: list[int] = []
    per_mode: dict[str, dict[str, int]] = {}
    for mode, mode_rows in rows.items():
        per_mode[mode] = {}
        for prototype, prototype_rows in mode_rows.items():
            values = [len(value) for value in prototype_rows.values()]
            depth = max(values, default=0)
            per_mode[mode][prototype] = depth
            depths.extend(values)
    return {
        "min": min(depths, default=0),
        "max": max(depths, default=0),
        "mean": sum(depths) / len(depths) if depths else 0.0,
        "per_mode_prototype_max": per_mode,
    }


def _source_score_coverage(
    rows: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    *,
    catalog_size: int,
) -> dict[str, Any]:
    """Check whether score fusion is defensible for each retrieval mode."""

    by_mode: dict[str, Any] = {}
    all_available = True
    for mode, mode_rows in rows.items():
        per_prototype: dict[str, Any] = {}
        mode_available = True
        max_candidates = 0
        max_scored = 0
        score_fields_present = False
        for prototype, prototype_rows in mode_rows.items():
            counts: list[int] = []
            scored_counts: list[int] = []
            for ranking in prototype_rows.values():
                unique_count = len({item["action_key"] for item in ranking})
                scored_count = sum(item["score"] is not None for item in ranking)
                counts.append(unique_count)
                scored_counts.append(scored_count)
                max_candidates = max(max_candidates, unique_count)
                max_scored = max(max_scored, scored_count)
                score_fields_present = score_fields_present or scored_count > 0
            prototype_complete = bool(counts) and all(
                count == catalog_size and scored == catalog_size
                for count, scored in zip(counts, scored_counts, strict=True)
            )
            mode_available = mode_available and prototype_complete
            per_prototype[prototype] = {
                "rows": len(counts),
                "min_candidates": min(counts, default=0),
                "max_candidates": max(counts, default=0),
                "min_scored": min(scored_counts, default=0),
                "max_scored": max(scored_counts, default=0),
                "complete": prototype_complete,
            }
        if mode_available:
            reason = "finite mode-specific scores cover the full catalog for all prototypes"
        elif not score_fields_present:
            reason = "no finite mode-specific score fields are present"
        else:
            reason = (
                "score fields are present, but source rankings are top-k truncated; "
                f"at most {max_scored}/{catalog_size} scored candidates are retained per row"
            )
        by_mode[mode] = {
            "available": mode_available,
            "catalog_size": catalog_size,
            "max_candidates_per_row": max_candidates,
            "max_scored_candidates_per_row": max_scored,
            "score_field": RANK_SCORE_FIELDS[mode],
            "per_prototype": per_prototype,
            "reason": reason,
        }
        all_available = all_available and mode_available
    return {"available_all_modes": all_available, "by_mode": by_mode}


def fuse_rankings(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse prototype rankings with reciprocal-rank fusion.

    Missing actions contribute no term.  This is intentional for a truncated
    source sidecar and is surfaced in the caller's coverage metadata.
    """

    if rrf_k <= 0:
        raise WemmLabelPrototypeFusionError("rrf_k must be positive")
    aggregate: dict[tuple[int, int], dict[str, Any]] = {}
    for prototype in PROTOTYPES:
        ranking = rankings.get(prototype, ())
        for item in ranking:
            action = _pair(item.get("action_key"), field=f"{prototype}.action_key")
            rank = _integer(item.get("rank"), field=f"{prototype}.rank")
            if rank <= 0:
                raise WemmLabelPrototypeFusionError("ranking ranks must be positive")
            entry = aggregate.setdefault(
                action,
                {
                    "action_key": action,
                    "rrf_score": 0.0,
                    "support_count": 0,
                    "best_input_rank": rank,
                    "per_prototype_rank": {},
                    "per_prototype_score": {},
                    "label": {
                        "verb_id": item.get("verb_id"),
                        "noun_id": item.get("noun_id"),
                        "verb_key": item.get("verb_key"),
                        "noun_key": item.get("noun_key"),
                        "label_text": item.get("label_text"),
                    },
                },
            )
            if prototype in entry["per_prototype_rank"]:
                raise WemmLabelPrototypeFusionError(
                    f"duplicate action {action} in prototype {prototype}"
                )
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry["support_count"] += 1
            entry["best_input_rank"] = min(entry["best_input_rank"], rank)
            entry["per_prototype_rank"][prototype] = rank
            entry["per_prototype_score"][prototype] = item.get("score")
            # Prefer a readable label from canonical, then the first available
            # surface.  It is metadata only and never drives ordering.
            if prototype == "canonical" or not entry["label"].get("label_text"):
                entry["label"] = {
                    "verb_id": item.get("verb_id"),
                    "noun_id": item.get("noun_id"),
                    "verb_key": item.get("verb_key"),
                    "noun_key": item.get("noun_key"),
                    "label_text": item.get("label_text"),
                }

    ordered = sorted(
        aggregate.values(),
        key=lambda item: (
            -item["rrf_score"],
            -item["support_count"],
            item["best_input_rank"],
            _action_sort_key(item["action_key"]),
        ),
    )
    if top_k is not None:
        if top_k <= 0:
            raise WemmLabelPrototypeFusionError("top_k must be positive")
        ordered = ordered[:top_k]
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(ordered, start=1):
        label = item["label"]
        output.append(
            {
                "rank": rank,
                "action_key": _json_pair(item["action_key"]),
                "verb_id": label.get("verb_id"),
                "noun_id": label.get("noun_id"),
                "verb_key": label.get("verb_key"),
                "noun_key": label.get("noun_key"),
                "label_text": label.get("label_text"),
                "rrf_score": item["rrf_score"],
                "support_count": item["support_count"],
                "best_input_rank": item["best_input_rank"],
                "per_prototype_rank": dict(item["per_prototype_rank"]),
                "per_prototype_score": dict(item["per_prototype_score"]),
            }
        )
    return output


def fuse_scores(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse complete, finite score tables by an equal-weight mean.

    Callers should first use :func:`_source_score_coverage` (or the public
    wrapper below).  Missing scores are rejected rather than interpreted as
    zero evidence.
    """

    aggregate: dict[tuple[int, int], dict[str, Any]] = {}
    for prototype in PROTOTYPES:
        for item in rankings.get(prototype, ()):
            action = _pair(item.get("action_key"), field=f"{prototype}.action_key")
            score_raw = item.get("score")
            if score_raw is None:
                raise WemmLabelPrototypeFusionError(
                    f"score fusion requires a finite score for {prototype}/{action}"
                )
            score = _finite(score_raw, field=f"{prototype}/{action}.score")
            entry = aggregate.setdefault(
                action,
                {
                    "action_key": action,
                    "scores": {},
                    "label": {
                        "verb_id": item.get("verb_id"),
                        "noun_id": item.get("noun_id"),
                        "verb_key": item.get("verb_key"),
                        "noun_key": item.get("noun_key"),
                        "label_text": item.get("label_text"),
                    },
                },
            )
            if prototype in entry["scores"]:
                raise WemmLabelPrototypeFusionError(
                    f"duplicate action {action} in prototype {prototype}"
                )
            entry["scores"][prototype] = score
            if prototype == "canonical" or not entry["label"].get("label_text"):
                entry["label"] = {
                    "verb_id": item.get("verb_id"),
                    "noun_id": item.get("noun_id"),
                    "verb_key": item.get("verb_key"),
                    "noun_key": item.get("noun_key"),
                    "label_text": item.get("label_text"),
                }
    expected = set(aggregate)
    for action, item in aggregate.items():
        if set(item["scores"]) != set(PROTOTYPES):
            raise WemmLabelPrototypeFusionError(
                f"score fusion requires all prototypes for action {action}"
            )
    ordered = sorted(
        aggregate.values(),
        key=lambda item: (
            -sum(item["scores"].values()) / len(PROTOTYPES),
            _action_sort_key(item["action_key"]),
        ),
    )
    if top_k is not None:
        if top_k <= 0:
            raise WemmLabelPrototypeFusionError("top_k must be positive")
        ordered = ordered[:top_k]
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(ordered, start=1):
        label = item["label"]
        mean_score = sum(item["scores"].values()) / len(PROTOTYPES)
        output.append(
            {
                "rank": rank,
                "action_key": _json_pair(item["action_key"]),
                "verb_id": label.get("verb_id"),
                "noun_id": label.get("noun_id"),
                "verb_key": label.get("verb_key"),
                "noun_key": label.get("noun_key"),
                "label_text": label.get("label_text"),
                "mean_score": mean_score,
                "per_prototype_score": dict(item["scores"]),
            }
        )
    # Keep a local use of ``expected`` above explicit: it makes it clear that
    # no unseen catalog action is synthesized by score fusion.
    del expected
    return output


def _rank_of(ranking: Sequence[Mapping[str, Any]], target: tuple[int, int]) -> int | None:
    for index, item in enumerate(ranking, start=1):
        if _pair(item.get("action_key"), field="ranking.action_key") == target:
            return index
    return None


def _metrics(
    cases: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    eligible: Sequence[bool] | None = None,
) -> dict[str, Any]:
    if eligible is None:
        eligible = [True] * len(cases)
    if len(eligible) != len(cases):
        raise WemmLabelPrototypeFusionError("eligible mask length does not match cases")
    eligible_count = sum(bool(value) for value in eligible)
    ranks: list[int | None] = []
    for index, case in enumerate(cases):
        if not eligible[index]:
            continue
        row_id = str(case["id"])
        if row_id not in rankings:
            continue
        ranks.append(_rank_of(rankings[row_id], case["ground_truth"]))
    scored = len(ranks)
    target_found = sum(rank is not None for rank in ranks)
    denominator = scored
    recall = {
        str(k): sum(rank is not None and rank <= k for rank in ranks) / denominator
        if denominator
        else 0.0
        for k in METRIC_KS
    }
    mrr = (
        sum(1.0 / rank for rank in ranks if rank is not None) / denominator if denominator else 0.0
    )
    top1 = sum(rank == 1 for rank in ranks) / denominator if denominator else 0.0
    return {
        "query_count": len(cases),
        "eligible_query_count": eligible_count,
        "scored_query_count": scored,
        "target_found_count": target_found,
        "target_coverage": target_found / scored if scored else 0.0,
        # Compatibility spelling for callers that used the older exploratory
        # draft; this is target presence in retained rankings, not model score
        # coverage.
        "score_coverage": target_found / scored if scored else 0.0,
        "recall_at_k": recall,
        "mrr": mrr,
        "top1_accuracy": top1,
    }


def _hard_negative_mask(
    ranking: Sequence[Mapping[str, Any]],
    target: tuple[int, int],
    kind: str,
) -> tuple[list[dict[str, Any]], int]:
    if kind not in HARD_NEGATIVE_KINDS:
        raise WemmLabelPrototypeFusionError(f"unsupported hard-negative kind: {kind}")
    hard: list[dict[str, Any]] = []
    for item in ranking:
        action = _pair(item.get("action_key"), field="hard_negative.action_key")
        if action == target:
            continue
        same_verb = action[0] == target[0]
        same_noun = action[1] == target[1]
        if kind == "same_verb":
            include = same_verb
        elif kind == "same_noun":
            include = same_noun
        else:
            include = same_verb or same_noun
        if include:
            hard.append(dict(item))
    return hard, len(hard)


def hard_negative_metrics(
    cases: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    kind: str,
) -> dict[str, Any]:
    """Measure target-vs-structural-hard-negative ordering.

    The denominator includes rows with at least one observed hard negative;
    rows where the target is outside the supplied ranking remain misses.  This
    makes truncation visible through ``target_coverage``.
    """

    restricted: dict[str, Sequence[Mapping[str, Any]]] = {}
    eligible: list[bool] = []
    hard_counts: list[int] = []
    for case in cases:
        row_id = str(case["id"])
        ranking = rankings.get(row_id, ())
        hard, count = _hard_negative_mask(ranking, case["ground_truth"], kind)
        hard_counts.append(count)
        eligible.append(count > 0)
        # Preserve source order while retaining the target when it appears.
        restricted[row_id] = [
            dict(item)
            for item in ranking
            if _pair(item.get("action_key"), field="hard_negative.action_key")
            == case["ground_truth"]
            or item in hard
        ]
    result = _metrics(cases, restricted, eligible=eligible)
    result.update(
        {
            "hard_negative_kind": kind,
            "rows_with_observed_hard_negative": sum(eligible),
            "observed_hard_negative_count": sum(hard_counts),
            "mean_observed_hard_negative_count": (
                sum(hard_counts) / sum(eligible) if sum(eligible) else 0.0
            ),
        }
    )
    return result


def _serialise_ranking(ranking: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Every ranking produced by this module already has JSON-shaped values;
    # copying here ensures callers cannot mutate the report internals.
    return [
        {key: (_json_pair(value) if key == "action_key" else value) for key, value in item.items()}
        for item in ranking
    ]


def _source_metric_subset(report: Mapping[str, Any], prototype: str, mode: str) -> dict[str, Any]:
    try:
        metric = report["results"][prototype]["metrics"]["metrics"][mode]
    except (KeyError, TypeError) as exc:
        raise WemmLabelPrototypeFusionError(
            f"source report is missing metrics for {prototype}/{mode}"
        ) from exc
    if not isinstance(metric, Mapping):
        raise WemmLabelPrototypeFusionError(f"source metric {prototype}/{mode} must be an object")
    return {
        key: metric[key]
        for key in (
            "query_count",
            "scored_query_count",
            "recall_at_k",
            "mrr",
            "top1_accuracy",
        )
        if key in metric
    }


def build_diagnostic(
    report: Mapping[str, Any],
    *,
    source_report: str = "",
    rrf_k: int = 60,
    top_k: int = 10,
    method: FusionMethod = "auto",
) -> dict[str, Any]:
    """Build a JSON-native exploratory diagnostic from one source sidecar."""

    if not isinstance(report, Mapping):
        raise WemmLabelPrototypeFusionError("source report must be an object")
    if method not in {"rank", "score", "auto"}:
        raise WemmLabelPrototypeFusionError("method must be rank, score, or auto")
    if top_k <= 0:
        raise WemmLabelPrototypeFusionError("top_k must be positive")
    if rrf_k <= 0:
        raise WemmLabelPrototypeFusionError("rrf_k must be positive")
    cases = _case_deltas(report)
    catalog_actions = _catalog_actions(report)
    catalog_size = _catalog_size(report) or len(catalog_actions)
    if catalog_actions and catalog_size != len(catalog_actions):
        # A declared catalog size is a useful provenance check; do not silently
        # reinterpret a mismatch as a complete score table.
        raise WemmLabelPrototypeFusionError(
            f"catalog_size={catalog_size} disagrees with labels={len(catalog_actions)}"
        )
    rows = _normalised_rows(report, cases)
    score_coverage = _source_score_coverage(rows, catalog_size=catalog_size)
    if method == "score" and not score_coverage["available_all_modes"]:
        raise WemmLabelPrototypeFusionError(
            "score fusion requested, but at least one mode lacks complete finite scores"
        )
    score_enabled = method != "rank"

    baseline_metrics: dict[str, dict[str, Any]] = {}
    fused_metrics: dict[str, dict[str, Any]] = {}
    hard_comparison: dict[str, dict[str, Any]] = {}
    fused_rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    score_rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    metric_reproduction: dict[str, dict[str, Any]] = {}

    for mode in MODES:
        baseline_metrics[mode] = {
            prototype: _source_metric_subset(report, prototype, mode) for prototype in PROTOTYPES
        }
        fused_rankings[mode] = {}
        score_rankings[mode] = {}
        mode_rankings: dict[str, Mapping[str, Sequence[Mapping[str, Any]]]] = {
            prototype: rows[mode][prototype] for prototype in PROTOTYPES
        }
        # Recompute the source top-k metrics without touching the source block;
        # this is a useful integrity check for a post-hoc-only transformation.
        metric_reproduction[mode] = {}
        for prototype in PROTOTYPES:
            source_rows = mode_rankings[prototype]
            recomputed = _metrics(cases, source_rows)
            source_metric = baseline_metrics[mode][prototype]
            matches = (
                source_metric.get("query_count") == recomputed["query_count"]
                and source_metric.get("scored_query_count") == recomputed["scored_query_count"]
                and source_metric.get("recall_at_k") == recomputed["recall_at_k"]
                and math.isclose(
                    float(source_metric.get("mrr", 0.0)),
                    recomputed["mrr"],
                    rel_tol=1e-12,
                )
                and math.isclose(
                    float(source_metric.get("top1_accuracy", 0.0)),
                    recomputed["top1_accuracy"],
                    rel_tol=1e-12,
                )
            )
            metric_reproduction[mode][prototype] = {
                "matches_source": matches,
                "recomputed_from_retained_rankings": recomputed,
            }

        for index, case in enumerate(cases):
            row_id = str(case["id"])
            per_variant = {prototype: mode_rankings[prototype][row_id] for prototype in PROTOTYPES}
            fused = fuse_rankings(per_variant, rrf_k=rrf_k, top_k=None)
            fused_rankings[mode][row_id] = _serialise_ranking(fused[:top_k])

            if score_enabled and score_coverage["by_mode"][mode]["available"]:
                score_fused = fuse_scores(per_variant, top_k=None)
                score_rankings[mode][row_id] = _serialise_ranking(score_fused[:top_k])
            del index

        fused_rank_map = {row_id: ranking for row_id, ranking in fused_rankings[mode].items()}
        fused_metrics[mode] = {
            "rank_rrf": _metrics(cases, fused_rank_map),
            "selected_method": (
                "score_mean" if score_enabled and score_rankings[mode] else "rank_rrf"
            ),
            "rank_method": {
                "name": "reciprocal_rank_fusion",
                "rrf_k": rrf_k,
                "input_depth": "retained_source_top_k_union",
                "top_k": top_k,
            },
        }
        if score_rankings[mode]:
            fused_metrics[mode]["score_mean"] = _metrics(cases, score_rankings[mode])
        for kind in HARD_NEGATIVE_KINDS:
            hard_comparison.setdefault(mode, {})[kind] = {
                prototype: hard_negative_metrics(cases, mode_rankings[prototype], kind=kind)
                for prototype in PROTOTYPES
            }
            hard_comparison[mode][kind]["rank_rrf"] = hard_negative_metrics(
                cases, fused_rank_map, kind=kind
            )
            if score_rankings[mode]:
                hard_comparison[mode][kind]["score_mean"] = hard_negative_metrics(
                    cases, score_rankings[mode], kind=kind
                )

    source_controls = report.get("controls")
    source_input = report.get("input")
    return {
        "report_version": FUSION_VERSION,
        "generated_at": "2026-08-27",
        "authority": AUTHORITY,
        "production_eligible": False,
        "exploratory": True,
        "quality_status": "EXPLORATORY_NOT_PRODUCTION",
        "source_report": source_report,
        "source_report_version": report.get("report_version"),
        "experiment": {
            "kind": "cross_label_prototype_posthoc",
            "case_count": len(cases),
            "catalog_size": catalog_size,
            "catalog_action_count_observed": len(catalog_actions),
            "prototypes": list(PROTOTYPES),
            "modes": list(MODES),
            "rank_fusion": {
                "method": "reciprocal_rank_fusion",
                "rrf_k": rrf_k,
                "equal_prototype_weights": True,
                "retained_top_k_per_prototype": top_k,
                "requested_method": method,
            },
            "ranking_depths": _ranking_depths(rows),
            "score_fusion": score_coverage,
        },
        "controls": {
            "posthoc_inference_performed": False,
            "posthoc_media_decoded": False,
            "source_report_reused": True,
            "baseline_report_mutated": False,
            "ground_truth_used_only_for_posthoc_metrics": True,
            "heldout_100_opened_for_fusion": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "production_path_changed": False,
            "hash_or_sha_used": False,
            "source_controls_snapshot": dict(source_controls)
            if isinstance(source_controls, Mapping)
            else None,
            "source_input_snapshot": {
                key: source_input[key]
                for key in (
                    "manifest",
                    "case_count",
                    "catalog_size",
                    "catalog_source",
                    "label_variants",
                )
                if isinstance(source_input, Mapping) and key in source_input
            },
        },
        "baseline_metrics_from_source": baseline_metrics,
        "baseline_metric_reproduction": metric_reproduction,
        "case_targets": [
            {"id": str(case["id"]), "ground_truth": _json_pair(case["ground_truth"])}
            for case in cases
        ],
        "fusion_metrics": fused_metrics,
        "hard_negative_comparison": {
            "definition": {
                "same_verb": "candidate shares ground-truth verb but has another noun",
                "same_noun": "candidate shares ground-truth noun but has another verb",
                "either": "candidate shares either ground-truth verb or noun",
                "visual_grounding": False,
            },
            "by_mode": hard_comparison,
        },
        "fused_rankings": fused_rankings,
        "score_fused_rankings": score_rankings,
        "interpretation": [
            "RRF combines only the retained top-k rows from the three prototype surfaces.",
            (
                "The current sidecar has finite score fields but does not retain "
                "full-catalog scores; score fusion is therefore suppressed and rank "
                "fusion is the admitted result."
            ),
            (
                "Hard-negative rows are structural same-verb/same-noun label "
                "comparisons, not visibility-grounded negatives."
            ),
            (
                "This exploratory post-processing artifact does not alter or replace "
                "any baseline metric, Mapper result, ontology, production route, or "
                "UI/API behavior."
            ),
        ],
        "limitations": [
            "27-row development cohort across five videos; not held-out-100.",
            (
                "Source rankings are truncated (currently top ten), so fused "
                "Recall@K/MRR/Top-1 are top-k-union diagnostics rather than "
                "full-catalog guarantees."
            ),
            (
                "Missing candidates receive no RRF contribution; their unknown "
                "source ranks cannot be inferred."
            ),
            "Structural hard negatives are not proof of visual confusion or temporal adjacency.",
            "Exploratory only; not production quality and not a training or promotion signal.",
        ],
    }


__all__ = [
    "AUTHORITY",
    "FUSION_VERSION",
    "HARD_NEGATIVE_KINDS",
    "MODES",
    "PROTOTYPES",
    "WemmLabelPrototypeFusionError",
    "build_diagnostic",
    "fuse_rankings",
    "fuse_scores",
    "hard_negative_metrics",
]
