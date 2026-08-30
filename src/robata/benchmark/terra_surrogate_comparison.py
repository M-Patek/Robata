"""Compare two Terra production review artifacts as non-gold surrogates.

This module is intentionally a small, benchmark-only diagnostic.  It compares
the owner-confirmed and independent source-bound review artifacts without
invoking a model, opening media, reading or writing gold, or calculating an
identity/hash.  ``accept`` and ``edit`` are collapsed to a retained decision,
while split and abstain remain separate signals.

The result is a *surrogate consistency* report.  It is not an accuracy report:
the two inputs are review artifacts, neither is official gold, and their
intervals are frame-anchored estimates rather than independently adjudicated
action boundaries.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, TypeGuard

TERRA_SURROGATE_COMPARISON_VERSION: Final = "robata-terra-surrogate-comparison-v1"
VERSION: Final = TERRA_SURROGATE_COMPARISON_VERSION
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "NON_GOLD_SURROGATE_CONSISTENCY"
INTERVAL_TOLERANCE_SECONDS: Final = 1e-6

_DECISION_ALIASES: Final = {
    "accept": "RETAINED",
    "accepted": "RETAINED",
    "edit": "RETAINED",
    "retain": "RETAINED",
    "retained": "RETAINED",
    "split": "SPLIT",
    "reject": "REJECT",
    "rejected": "REJECT",
    "abstain": "ABSTAIN",
    "abstained": "ABSTAIN",
}
_LABEL_FIELDS: Final = ("verb", "noun", "attributes", "location", "hand")
_INTERVAL_START_KEYS: Final = ("start_seconds", "start_time_sec", "start")
_INTERVAL_END_KEYS: Final = ("end_seconds", "end_time_sec", "end")


class TerraSurrogateComparisonError(ValueError):
    """Raised when a review artifact cannot be compared safely."""


def _is_sequence(value: object) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TerraSurrogateComparisonError(f"{field} must be an object")
    return value


def _required_window_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TerraSurrogateComparisonError(f"{field} must be non-empty text")
    return value.strip()


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalise_component(value: object) -> str | None:
    """Normalize surface spelling without applying semantic synonyms."""

    text = _text(value)
    if text is None:
        return None
    # Underscores, hyphens, punctuation and repeated whitespace are only
    # surface formatting.  We deliberately do not map synonyms such as
    # ``pickup`` and ``pick up`` beyond this conservative punctuation rule.
    token = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    token = " ".join(token.split())
    return token or None


def _field_value(segment: Mapping[str, Any], field: str) -> object:
    """Read a label field from current and older sidecar spellings."""

    aliases: dict[str, tuple[str, ...]] = {
        "verb": ("verb", "primary_verb"),
        "noun": ("noun", "primary_noun"),
        "attributes": ("attributes", "attribute", "attrs"),
        "location": ("location", "place"),
        "hand": ("hand", "hands"),
    }
    for key in aliases[field]:
        if key in segment:
            return segment[key]

    nested = segment.get("structured_labels")
    if isinstance(nested, Mapping):
        for key in aliases[field]:
            if key not in nested:
                continue
            value = nested[key]
            if isinstance(value, Mapping) and "value" in value:
                return value.get("value")
            return value
    return None


def _interval_value(segment: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in segment:
            return _finite(segment[key])
    return None


def _normalise_segment(value: object, *, index: int, field: str) -> dict[str, Any]:
    segment = _mapping(value, field=field)
    labels = {label: _normalise_component(_field_value(segment, label)) for label in _LABEL_FIELDS}
    start = _interval_value(segment, _INTERVAL_START_KEYS)
    end = _interval_value(segment, _INTERVAL_END_KEYS)
    return {
        "index": index,
        "verb": labels["verb"],
        "noun": labels["noun"],
        "attributes": labels["attributes"],
        "location": labels["location"],
        "hand": labels["hand"],
        "start_seconds": start,
        "end_seconds": end,
        "label_tuple": tuple(labels[field] for field in _LABEL_FIELDS),
    }


def _extract_segments(row: Mapping[str, Any], *, field: str) -> list[dict[str, Any]]:
    value = row.get("segments", [])
    if value is None:
        return []
    if not _is_sequence(value):
        raise TerraSurrogateComparisonError(f"{field}.segments must be an array")
    return [
        _normalise_segment(item, index=index, field=f"{field}.segments[{index}]")
        for index, item in enumerate(value)
    ]


def _explicit_decision(row: Mapping[str, Any]) -> tuple[str | None, str]:
    """Return raw decision and where it came from.

    The independent artifact uses ``recommendation`` instead of ``decision``;
    both spellings are retained in the diagnostic metadata.
    """

    if "decision" in row:
        raw = _text(row.get("decision"))
        return raw, "decision" if raw is not None else "invalid_decision"
    if "recommendation" in row:
        raw = _text(row.get("recommendation"))
        return raw, "recommendation" if raw is not None else "invalid_recommendation"
    return None, "missing"


def _normalise_decision(
    raw: str | None,
    *,
    decision_source: str,
    segment_count: int,
) -> tuple[str, str]:
    if raw is not None:
        key = re.sub(r"[^a-z0-9]+", "_", raw.casefold()).strip("_")
        mapped = _DECISION_ALIASES.get(key)
        if mapped is not None:
            return mapped, decision_source
        return "UNKNOWN", "invalid_decision"

    # A missing recommendation is diagnosable rather than silently converted
    # to abstain.  Segment count provides a useful inferred signal when it is
    # available, but the report records that it was inferred.
    if segment_count > 1:
        return "SPLIT", "inferred_from_segments"
    if segment_count == 1:
        return "RETAINED", "inferred_from_segments"
    return "UNKNOWN", "missing"


def _normalise_row(row: object, *, index: int, side: str) -> dict[str, Any]:
    record = _mapping(row, field=f"{side}.rows[{index}]")
    window_id = _required_window_id(
        record.get("window_id"), field=f"{side}.rows[{index}].window_id"
    )
    segments = _extract_segments(record, field=f"{side}.rows[{index}]")
    raw_decision, decision_source = _explicit_decision(record)
    decision, decision_source = _normalise_decision(
        raw_decision,
        decision_source=decision_source,
        segment_count=len(segments),
    )

    # Explicit abstain/reject rows can retain a candidate segment for context;
    # it must not be scored as a retained action claim.
    retained = decision in {"RETAINED", "SPLIT"}
    claim_segments = segments if retained else []
    candidate_segments = segments if not retained else []
    split_signal = decision == "SPLIT" or (retained and len(segments) > 1)
    return {
        "window_id": window_id,
        "decision": decision,
        "raw_decision": raw_decision,
        "decision_source": decision_source,
        "segments": claim_segments,
        "candidate_segments": candidate_segments,
        "all_segment_count": len(segments),
        "claim_segment_count": len(claim_segments),
        "candidate_segment_count": len(candidate_segments),
        "split_signal": split_signal,
        "explicit_abstain": decision == "ABSTAIN",
    }


def _rows_for_payload(payload: Mapping[str, Any], *, side: str) -> list[dict[str, Any]]:
    key = "windows" if side == "confirmed" else "items"
    value = payload.get(key)
    if not _is_sequence(value):
        raise TerraSurrogateComparisonError(f"{side}.{key} must be an array")
    rows = [_normalise_row(item, index=index, side=side) for index, item in enumerate(value)]
    seen: set[str] = set()
    for row in rows:
        window_id = row["window_id"]
        if window_id in seen:
            raise TerraSurrogateComparisonError(
                f"{side} contains duplicate window_id {window_id!r}"
            )
        seen.add(window_id)
    return rows


def _validate_non_gold(payload: Mapping[str, Any], *, side: str) -> None:
    if payload.get("official_gold") is True:
        raise TerraSurrogateComparisonError(f"{side} explicitly claims official gold")
    status = payload.get("official_gold_status")
    if status is not None and str(status).strip().upper() not in {
        "NOT_ESTABLISHED",
        "NOT_MEASURED",
        "NON_GOLD",
    }:
        raise TerraSurrogateComparisonError(
            f"{side}.official_gold_status must remain non-gold, got {status!r}"
        )
    adjudication = payload.get("human_adjudication")
    if adjudication is not None and str(adjudication).strip().upper() not in {
        "NOT_PERFORMED",
        "NOT_ESTABLISHED",
        "NONE",
    }:
        raise TerraSurrogateComparisonError(
            f"{side}.human_adjudication unexpectedly claims adjudication: {adjudication!r}"
        )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _core_label_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_verb, left_noun = left.get("verb"), left.get("noun")
    right_verb, right_noun = right.get("verb"), right.get("noun")
    return (
        isinstance(left_verb, str)
        and isinstance(left_noun, str)
        and isinstance(right_verb, str)
        and isinstance(right_noun, str)
        and (left_verb, left_noun) == (right_verb, right_noun)
    )


def _structured_label_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not _core_label_exact(left, right):
        return False
    return tuple(left.get(field) for field in _LABEL_FIELDS) == tuple(
        right.get(field) for field in _LABEL_FIELDS
    )


def _interval_stats(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_start = left.get("start_seconds")
    left_end = left.get("end_seconds")
    right_start = right.get("start_seconds")
    right_end = right.get("end_seconds")
    left_start_f = _finite(left_start)
    left_end_f = _finite(left_end)
    right_start_f = _finite(right_start)
    right_end_f = _finite(right_end)
    if any(value is None for value in (left_start_f, left_end_f, right_start_f, right_end_f)):
        return {
            "exact": False,
            "positive_overlap": False,
            "overlap_seconds": None,
            "iou": None,
        }
    assert left_start_f is not None
    assert left_end_f is not None
    assert right_start_f is not None
    assert right_end_f is not None
    if left_end_f < left_start_f or right_end_f < right_start_f:
        return {
            "exact": False,
            "positive_overlap": False,
            "overlap_seconds": None,
            "iou": None,
        }
    intersection = max(0.0, min(left_end_f, right_end_f) - max(left_start_f, right_start_f))
    union = max(left_end_f, right_end_f) - min(left_start_f, right_start_f)
    if union > 0.0:
        iou: float | None = intersection / union
    else:
        iou = 1.0 if abs(left_start_f - right_start_f) <= INTERVAL_TOLERANCE_SECONDS else 0.0
    return {
        "exact": (
            abs(left_start_f - right_start_f) <= INTERVAL_TOLERANCE_SECONDS
            and abs(left_end_f - right_end_f) <= INTERVAL_TOLERANCE_SECONDS
        ),
        "positive_overlap": intersection > INTERVAL_TOLERANCE_SECONDS,
        "overlap_seconds": intersection,
        "iou": iou,
    }


def _pair_score(
    left: Mapping[str, Any], right: Mapping[str, Any], left_index: int, right_index: int
) -> tuple[int, int, int, float, int]:
    interval = _interval_stats(left, right)
    iou = interval["iou"] if isinstance(interval["iou"], float) else -1.0
    return (
        1 if _structured_label_exact(left, right) else 0,
        1 if _core_label_exact(left, right) else 0,
        1 if interval["positive_overlap"] else 0,
        iou,
        -abs(left_index - right_index),
    )


def _pair_segments(
    left_segments: Sequence[Mapping[str, Any]], right_segments: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]]:
    if len(left_segments) == len(right_segments):
        return [(left, right) for left, right in zip(left_segments, right_segments, strict=True)]

    # Different segment counts are uncommon, but a deterministic greedy match
    # keeps the diagnostic useful without pretending to solve alignment.
    remaining = set(range(len(right_segments)))
    pairs: list[tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]] = []
    for left_index, left in enumerate(left_segments):
        if not remaining:
            pairs.append((left, None))
            continue
        right_index = max(
            remaining,
            key=lambda candidate: _pair_score(
                left, right_segments[candidate], left_index, candidate
            ),
        )
        remaining.remove(right_index)
        pairs.append((left, right_segments[right_index]))
    pairs.extend((None, right_segments[index]) for index in sorted(remaining))
    return pairs


def _label_projection(segment: Mapping[str, Any] | None) -> dict[str, str | None] | None:
    if segment is None:
        return None
    return {field: segment.get(field) for field in _LABEL_FIELDS}


def _interval_projection(segment: Mapping[str, Any] | None) -> list[float | None] | None:
    if segment is None:
        return None
    start = segment.get("start_seconds")
    end = segment.get("end_seconds")
    return [start if isinstance(start, float) else None, end if isinstance(end, float) else None]


def _pair_projection(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> dict[str, Any]:
    interval = (
        _interval_stats(left or {}, right or {})
        if left is not None and right is not None
        else {
            "exact": False,
            "positive_overlap": False,
            "overlap_seconds": None,
            "iou": None,
        }
    )
    return {
        "confirmed_index": left.get("index") if left is not None else None,
        "independent_index": right.get("index") if right is not None else None,
        "confirmed_label": _label_projection(left),
        "independent_label": _label_projection(right),
        "confirmed_interval": _interval_projection(left),
        "independent_interval": _interval_projection(right),
        "exact_core_label": (
            _core_label_exact(left, right) if left is not None and right is not None else False
        ),
        "exact_structured_label": (
            _structured_label_exact(left, right)
            if left is not None and right is not None
            else False
        ),
        "exact_interval": interval["exact"],
        "positive_interval_overlap": interval["positive_overlap"],
        "overlap_seconds": interval["overlap_seconds"],
        "interval_iou": interval["iou"],
        "match_status": (
            "PAIRED"
            if left is not None and right is not None
            else "CONFIRMED_ONLY"
            if left is not None
            else "INDEPENDENT_ONLY"
        ),
    }


def _source_summary(payload: Mapping[str, Any], *, artifact: str | None) -> dict[str, Any]:
    source = payload.get("source")
    source_map = source if isinstance(source, Mapping) else {}
    return {
        "artifact": artifact,
        "format": payload.get("format"),
        "status": payload.get("status"),
        "review_state": payload.get("review_state"),
        "official_gold_status": payload.get("official_gold_status", "NOT_ESTABLISHED"),
        "human_adjudication": payload.get("human_adjudication", "NOT_PERFORMED"),
        "media_path": source_map.get("media_path"),
        "window_count_declared": source_map.get("window_count"),
    }


def compare_terra_surrogate_reviews(
    confirmed: Mapping[str, Any],
    independent: Mapping[str, Any],
    *,
    confirmed_artifact: str | None = None,
    independent_artifact: str | None = None,
) -> dict[str, Any]:
    """Compare two source-bound Terra review artifacts without gold claims."""

    confirmed_payload = _mapping(confirmed, field="confirmed")
    independent_payload = _mapping(independent, field="independent")
    _validate_non_gold(confirmed_payload, side="confirmed")
    _validate_non_gold(independent_payload, side="independent")
    confirmed_rows = _rows_for_payload(confirmed_payload, side="confirmed")
    independent_rows = _rows_for_payload(independent_payload, side="independent")
    confirmed_by_id = {row["window_id"]: row for row in confirmed_rows}
    independent_by_id = {row["window_id"]: row for row in independent_rows}
    ordered_ids = list(confirmed_by_id)
    ordered_ids.extend(
        window_id for window_id in independent_by_id if window_id not in confirmed_by_id
    )
    common_ids = [
        window_id
        for window_id in ordered_ids
        if window_id in confirmed_by_id and window_id in independent_by_id
    ]
    missing_from_independent = [
        window_id for window_id in confirmed_by_id if window_id not in independent_by_id
    ]
    missing_from_confirmed = [
        window_id for window_id in independent_by_id if window_id not in confirmed_by_id
    ]

    per_window: list[dict[str, Any]] = []
    compatible_count = 0
    decision_mismatch_count = 0
    decision_confusion: dict[str, int] = {}
    split_status_agreement_count = 0
    split_involved_count = 0
    split_segment_count_match = 0
    abstain_status_agreement_count = 0
    abstain_involved_count = 0
    paired_segment_count = 0
    unmatched_confirmed_count = 0
    unmatched_independent_count = 0
    core_label_matches = 0
    structured_label_matches = 0
    exact_interval_matches = 0
    positive_interval_overlaps = 0
    joint_matches = 0
    ious: list[float] = []

    for window_id in ordered_ids:
        confirmed_row = confirmed_by_id.get(window_id)
        independent_row = independent_by_id.get(window_id)
        if confirmed_row is None or independent_row is None:
            per_window.append(
                {
                    "window_id": window_id,
                    "confirmed": None
                    if confirmed_row is None
                    else _window_side_projection(confirmed_row),
                    "independent": None
                    if independent_row is None
                    else _window_side_projection(independent_row),
                    "decision_compatible": None,
                    "segment_pairs": [],
                }
            )
            continue

        confirmed_decision = confirmed_row["decision"]
        independent_decision = independent_row["decision"]
        decision_compatible = confirmed_decision == independent_decision
        if decision_compatible:
            compatible_count += 1
        else:
            decision_mismatch_count += 1
        confusion_key = f"{confirmed_decision}->{independent_decision}"
        decision_confusion[confusion_key] = decision_confusion.get(confusion_key, 0) + 1

        confirmed_split = bool(confirmed_row["split_signal"])
        independent_split = bool(independent_row["split_signal"])
        split_agree = confirmed_split == independent_split
        if split_agree:
            split_status_agreement_count += 1
        if confirmed_split or independent_split:
            split_involved_count += 1
        if (
            confirmed_split
            and independent_split
            and confirmed_row["claim_segment_count"] == independent_row["claim_segment_count"]
        ):
            split_segment_count_match += 1

        confirmed_abstain = confirmed_decision == "ABSTAIN"
        independent_abstain = independent_decision == "ABSTAIN"
        abstain_agree = confirmed_abstain == independent_abstain
        if abstain_agree:
            abstain_status_agreement_count += 1
        if confirmed_abstain or independent_abstain:
            abstain_involved_count += 1

        pairs = _pair_segments(confirmed_row["segments"], independent_row["segments"])
        pair_rows = [_pair_projection(left, right) for left, right in pairs]
        for pair in pair_rows:
            if pair["match_status"] == "CONFIRMED_ONLY":
                unmatched_confirmed_count += 1
                continue
            if pair["match_status"] == "INDEPENDENT_ONLY":
                unmatched_independent_count += 1
                continue
            paired_segment_count += 1
            if pair["exact_core_label"]:
                core_label_matches += 1
            if pair["exact_structured_label"]:
                structured_label_matches += 1
            if pair["exact_interval"]:
                exact_interval_matches += 1
            if pair["positive_interval_overlap"]:
                positive_interval_overlaps += 1
            if pair["exact_core_label"] and pair["exact_interval"]:
                joint_matches += 1
            if isinstance(pair["interval_iou"], (int, float)):
                ious.append(float(pair["interval_iou"]))

        per_window.append(
            {
                "window_id": window_id,
                "confirmed": _window_side_projection(confirmed_row),
                "independent": _window_side_projection(independent_row),
                "decision_compatible": decision_compatible,
                "split_status_agreement": split_agree,
                "abstain_status_agreement": abstain_agree,
                "segment_pairs": pair_rows,
            }
        )

    common_count = len(common_ids)
    split_both_count = sum(
        1
        for row in per_window
        if row.get("confirmed") is not None
        and row.get("independent") is not None
        and row["confirmed"].get("split_signal") is True
        and row["independent"].get("split_signal") is True
    )
    abstain_both_count = sum(
        1
        for row in per_window
        if row.get("confirmed") is not None
        and row.get("independent") is not None
        and row["confirmed"].get("decision") == "ABSTAIN"
        and row["independent"].get("decision") == "ABSTAIN"
    )
    result: dict[str, Any] = {
        "format": TERRA_SURROGATE_COMPARISON_VERSION,
        "authority": AUTHORITY,
        "status": STATUS,
        "quality_claim": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "human_adjudication": "NOT_PERFORMED",
        "method": {
            "decision_policy": (
                "accept/edit collapse to RETAINED; split, reject and abstain remain distinct"
            ),
            "missing_decision_policy": (
                "infer RETAINED/SPLIT only when segments are present; otherwise UNKNOWN"
            ),
            "label_normalization": (
                "case, whitespace, underscore, hyphen and punctuation only; no semantic synonyms"
            ),
            "structured_label_fields": list(_LABEL_FIELDS),
            "interval_tolerance_seconds": INTERVAL_TOLERANCE_SECONDS,
            "interval_overlap": (
                "continuous seconds intersection-over-union; window endpoints are not "
                "action boundaries"
            ),
            "pairing": (
                "index order when segment counts match; otherwise deterministic "
                "label/overlap greedy pairing"
            ),
            "candidate_policy": (
                "segments on explicit abstain/reject rows are candidate-only and excluded "
                "from retained-claim metrics"
            ),
        },
        "inputs": {
            "confirmed": _source_summary(confirmed_payload, artifact=confirmed_artifact),
            "independent": _source_summary(independent_payload, artifact=independent_artifact),
        },
        "windows": {
            "confirmed_count": len(confirmed_rows),
            "independent_count": len(independent_rows),
            "common_count": common_count,
            "missing_from_independent": missing_from_independent,
            "missing_from_confirmed": missing_from_confirmed,
            "common_ids": common_ids,
        },
        "decision_compatibility": {
            "common_windows": common_count,
            "compatible_count": compatible_count,
            "incompatible_count": decision_mismatch_count,
            "compatibility_rate": _ratio(compatible_count, common_count),
            "confusion": decision_confusion,
        },
        "label_interval_overlap": {
            "confirmed_retained_segments": sum(
                row["claim_segment_count"] for row in confirmed_rows
            ),
            "independent_retained_segments": sum(
                row["claim_segment_count"] for row in independent_rows
            ),
            "paired_segment_count": paired_segment_count,
            "unmatched_confirmed_count": unmatched_confirmed_count,
            "unmatched_independent_count": unmatched_independent_count,
            "core_label_exact_matches": core_label_matches,
            "core_label_exact_rate": _ratio(core_label_matches, paired_segment_count),
            "structured_label_exact_matches": structured_label_matches,
            "structured_label_exact_rate": _ratio(structured_label_matches, paired_segment_count),
            "exact_label_matches": structured_label_matches,
            "exact_label_match_rate": _ratio(structured_label_matches, paired_segment_count),
            "exact_interval_matches": exact_interval_matches,
            "exact_interval_match_rate": _ratio(exact_interval_matches, paired_segment_count),
            "positive_interval_overlap_count": positive_interval_overlaps,
            "positive_interval_overlap_rate": _ratio(
                positive_interval_overlaps, paired_segment_count
            ),
            "mean_interval_iou": sum(ious) / len(ious) if ious else None,
            "joint_core_label_and_exact_interval_matches": joint_matches,
            "joint_core_label_and_exact_interval_rate": _ratio(joint_matches, paired_segment_count),
        },
        "split_agreement": {
            "common_windows": common_count,
            "status_agreement_count": split_status_agreement_count,
            "status_agreement_rate": _ratio(split_status_agreement_count, common_count),
            "split_involved_windows": split_involved_count,
            "both_split_count": split_both_count,
            "confirmed_split_count": sum(1 for row in confirmed_rows if row["split_signal"]),
            "independent_split_count": sum(1 for row in independent_rows if row["split_signal"]),
            "split_segment_count_match": split_segment_count_match,
            "split_segment_count_match_rate": _ratio(split_segment_count_match, split_both_count),
        },
        "abstain_agreement": {
            "common_windows": common_count,
            "status_agreement_count": abstain_status_agreement_count,
            "status_agreement_rate": _ratio(abstain_status_agreement_count, common_count),
            "abstain_involved_windows": abstain_involved_count,
            "both_abstain_count": abstain_both_count,
            "confirmed_abstain_count": sum(
                1 for row in confirmed_rows if row["decision"] == "ABSTAIN"
            ),
            "independent_abstain_count": sum(
                1 for row in independent_rows if row["decision"] == "ABSTAIN"
            ),
        },
        "per_window": per_window,
        "limitations": [
            (
                "Both artifacts are explicitly non-gold review surrogates; no official annotation "
                "or human adjudication was read."
            ),
            (
                "Agreement is internal consistency, not precision, recall, correctness, or "
                "production readiness."
            ),
            (
                "The two reviews share the same ten-window source cohort and therefore are not "
                "independent media samples."
            ),
            (
                "The fixed four-second windows and frame-anchored intervals may clip action "
                "onset/completion; endpoints are not asserted boundaries."
            ),
            (
                "Structured-label exactness penalizes fields omitted by one artifact; core "
                "verb+noun agreement is reported separately."
            ),
            (
                "Explicit abstain/reject rows may carry candidate segments; those candidates are "
                "intentionally excluded from retained-claim overlap metrics."
            ),
            (
                "Missing decisions are surfaced as inferred or UNKNOWN diagnostics rather than "
                "silently treated as gold labels."
            ),
            (
                "No model, media decoder, mapper, ontology, training step, held-out set, or "
                "identity/hash operation was used."
            ),
        ],
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "source_media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "official_evaluator_invoked": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_digest_computed": False,
            "sha_or_digest_computed": False,
        },
    }
    if any(
        row["decision_source"] in {"missing", "invalid_decision", "inferred_from_segments"}
        for row in confirmed_rows + independent_rows
    ):
        result["limitations"].append(
            "At least one row lacked a usable explicit decision; its retained/split signal was "
            "inferred from segment presence or left UNKNOWN."
        )
    return result


def _window_side_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": row["decision"],
        "raw_decision": row["raw_decision"],
        "decision_source": row["decision_source"],
        "split_signal": row["split_signal"],
        "all_segment_count": row["all_segment_count"],
        "claim_segment_count": row["claim_segment_count"],
        "candidate_segment_count": row["candidate_segment_count"],
    }


def _format_rate(value: object) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{float(value):.1%}"


def _format_interval(value: object) -> str:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        return "-"
    start, end = value[0], value[1]
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return "-"
    return f"{float(start):.3f}-{float(end):.3f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable report without adding quality claims."""

    windows = report.get("windows", {})
    decisions = report.get("decision_compatibility", {})
    labels = report.get("label_interval_overlap", {})
    splits = report.get("split_agreement", {})
    abstains = report.get("abstain_agreement", {})
    decision_rate = _format_rate(decisions.get("compatibility_rate"))
    core_rate = _format_rate(labels.get("core_label_exact_rate"))
    structured_rate = _format_rate(labels.get("structured_label_exact_rate"))
    exact_interval_rate = _format_rate(labels.get("exact_interval_match_rate"))
    overlap_rate = _format_rate(labels.get("positive_interval_overlap_rate"))
    split_rate = _format_rate(splits.get("status_agreement_rate"))
    abstain_rate = _format_rate(abstains.get("status_agreement_rate"))
    mean_iou = (
        labels.get("mean_interval_iou")
        if isinstance(labels.get("mean_interval_iou"), (int, float))
        else "n/a"
    )
    lines = [
        "# Terra surrogate review consistency",
        "",
        (
            f"**Status:** `{report.get('status', STATUS)}`; **quality claim:** `false`; "
            "**official gold:** `NOT_ESTABLISHED`."
        ),
        "",
        (
            "This is a non-gold consistency comparison of two source-bound review artifacts. "
            "It is not model accuracy."
        ),
        "",
        "## Summary",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Common windows | {windows.get('common_count', 0)} |",
        (
            f"| Decision compatibility | {decisions.get('compatible_count', 0)}/"
            f"{decisions.get('common_windows', 0)} ({decision_rate}) |"
        ),
        (
            f"| Core verb+noun exact | {labels.get('core_label_exact_matches', 0)}/"
            f"{labels.get('paired_segment_count', 0)} ({core_rate}) |"
        ),
        (
            f"| Structured label exact | {labels.get('structured_label_exact_matches', 0)}/"
            f"{labels.get('paired_segment_count', 0)} ({structured_rate}) |"
        ),
        (
            f"| Exact interval | {labels.get('exact_interval_matches', 0)}/"
            f"{labels.get('paired_segment_count', 0)} ({exact_interval_rate}) |"
        ),
        (
            f"| Positive interval overlap | {labels.get('positive_interval_overlap_count', 0)}/"
            f"{labels.get('paired_segment_count', 0)} ({overlap_rate}) |"
        ),
        f"| Mean interval IoU | {mean_iou} |",
        (
            f"| Split status agreement | {splits.get('status_agreement_count', 0)}/"
            f"{splits.get('common_windows', 0)} ({split_rate}) |"
        ),
        (
            f"| Abstain status agreement | {abstains.get('status_agreement_count', 0)}/"
            f"{abstains.get('common_windows', 0)} ({abstain_rate}) |"
        ),
        "",
        "## Per-window decisions and retained claims",
        "",
        "| Window | Confirmed | Independent | Decision compatible | Segment pairs |",
        "|---|---|---|---:|---:|",
    ]
    per_window = report.get("per_window", [])
    if isinstance(per_window, Sequence) and not isinstance(per_window, (str, bytes, bytearray)):
        for row in per_window:
            if not isinstance(row, Mapping):
                continue
            confirmed = row.get("confirmed")
            independent = row.get("independent")
            confirmed_decision = (
                confirmed.get("decision", "-") if isinstance(confirmed, Mapping) else "-"
            )
            independent_decision = (
                independent.get("decision", "-") if isinstance(independent, Mapping) else "-"
            )
            pairs = row.get("segment_pairs", [])
            pair_count = (
                len(pairs)
                if isinstance(pairs, Sequence) and not isinstance(pairs, (str, bytes, bytearray))
                else 0
            )
            lines.append(
                f"| {row.get('window_id', '-')} | {confirmed_decision} | {independent_decision} | "
                f"{row.get('decision_compatible', 'n/a')} | {pair_count} |"
            )
    lines.extend(["", "## Limitations", ""])
    limitations = report.get("limitations", [])
    if isinstance(limitations, Sequence) and not isinstance(limitations, (str, bytes, bytearray)):
        lines.extend(f"- {item}" for item in limitations if isinstance(item, str))
    return "\n".join(lines) + "\n"


__all__ = [
    "AUTHORITY",
    "INTERVAL_TOLERANCE_SECONDS",
    "STATUS",
    "TERRA_SURROGATE_COMPARISON_VERSION",
    "VERSION",
    "TerraSurrogateComparisonError",
    "compare_terra_surrogate_reviews",
    "render_markdown",
]
