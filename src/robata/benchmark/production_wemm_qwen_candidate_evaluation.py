"""Evaluate the production WeMM Top-K -> Qwen verifier pilot.

The evaluator is deliberately post-hoc and non-gold.  It joins a Terra
source-bound surrogate review, a production-vocabulary candidate pack, and a
recorded Qwen verifier sidecar.  No media is decoded, no model is invoked, and
no label is written back as gold.  Retrieval and verifier scores are kept
separate so a compact verifier cannot hide a weak WeMM candidate recall (or
vice versa).
"""

from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

VERSION: Final = "robata-production-wemm-qwen-candidate-evaluation-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
OFFICIAL_GOLD_STATUS: Final = "NOT_ESTABLISHED"
_ELIGIBLE_DECISIONS: Final = frozenset({"accept", "edit", "split"})


class ProductionWemmQwenEvaluationError(ValueError):
    """Raised when an evaluation sidecar is malformed or mismatched."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmQwenEvaluationError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmQwenEvaluationError(f"{field} must be an array")
    return value


def load_json(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        decoded = json.loads(Path(value).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionWemmQwenEvaluationError(f"could not read {value}: {exc}") from exc
    return dict(_mapping(decoded, field=str(value)))


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


_VERB_FORMS: Final = {
    "pick up": "pick up",
    "pickup": "pick up",
    "picks up": "pick up",
    "picking up": "pick up",
    "spread": "spread",
    "spreads": "spread",
    "spreading": "spread",
    "flatten": "flatten",
    "flattens": "flatten",
    "flattening": "flatten",
    "adjust": "adjust",
    "adjusts": "adjust",
    "adjusting": "adjust",
    "smooth": "smooth",
    "smooths": "smooth",
    "smoothing": "smooth",
    "fold": "fold",
    "folds": "fold",
    "folding": "fold",
}
_GARMENT_NOUNS: Final = frozenset(
    {"cloth", "clothes", "clothing", "fabric", "garment", "shirt", "pants", "shorts", "sheets"}
)


def action_label(verb: object, noun: object) -> str | None:
    verb_text = _VERB_FORMS.get(_norm(verb), _norm(verb))
    noun_text = _norm(noun)
    if noun_text in _GARMENT_NOUNS:
        noun_text = "garment"
    if not verb_text or not noun_text:
        return None
    return f"{verb_text} {noun_text}"


def _reference_rows(reference: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = reference.get("items", reference.get("windows"))
    rows = _sequence(raw, field="reference.items")
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = _mapping(value, field=f"reference.items[{index}]")
        window_id = str(row.get("window_id") or "").strip()
        if not window_id:
            raise ProductionWemmQwenEvaluationError("reference row lacks window_id")
        segments = _sequence(row.get("segments", []), field=f"{window_id}.segments")
        expected: list[str] = []
        reference_segments: list[dict[str, Any]] = []
        for segment in segments:
            seg = _mapping(segment, field=f"{window_id}.segment")
            label = action_label(seg.get("verb", seg.get("verb_code")), seg.get("noun"))
            if label and label not in expected:
                expected.append(label)
            if label:
                reference_segments.append(
                    {
                        "label": label,
                        "interval": _interval(seg),
                    }
                )
        decision = str(row.get("recommendation", row.get("decision", ""))).strip().casefold()
        decision = {
            "accepted": "accept",
            "edited": "edit",
            "split": "split",
            "abstained": "abstain",
        }.get(decision, decision)
        window_start, window_end = _window_bounds(row)
        result[window_id] = {
            "decision": decision,
            "expected": expected,
            "segments": reference_segments,
            "window_start_seconds": window_start,
            "window_end_seconds": window_end,
            "reference_status": str(
                row.get("status", reference.get("status", "SURROGATE_REFERENCE"))
            ),
        }
    return result


def _pack_rows(pack: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = _sequence(pack.get("windows", []), field="candidate_pack.windows")
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw):
        row = _mapping(value, field=f"candidate_pack.windows[{index}]")
        window_id = str(row.get("window_id") or "").strip()
        if not window_id:
            raise ProductionWemmQwenEvaluationError("candidate pack row lacks window_id")
        context = row.get("model_context")
        context_map = _mapping(context, field=f"{window_id}.model_context")
        wemm = _mapping(context_map.get("wemm", {}), field=f"{window_id}.wemm")
        candidates = _sequence(
            wemm.get("top_k", wemm.get("predictions", [])),
            field=f"{window_id}.wemm.top_k",
        )
        top_k: list[dict[str, Any]] = []
        for candidate_index, candidate_value in enumerate(candidates):
            candidate = _mapping(
                candidate_value,
                field=f"{window_id}.wemm.top_k[{candidate_index}]",
            )
            label = action_label(candidate.get("verb"), candidate.get("noun"))
            if not label:
                continue
            try:
                rank = int(candidate.get("rank", candidate_index + 1))
            except (TypeError, ValueError) as exc:
                raise ProductionWemmQwenEvaluationError(
                    f"{window_id} candidate rank is invalid"
                ) from exc
            top_k.append(
                {
                    "rank": rank,
                    "label": label,
                    "raw_label": candidate.get("raw_label", candidate.get("label_text")),
                    "score": candidate.get("score"),
                    "camera_coverage": candidate.get("camera_coverage"),
                    "camera_coverage_fraction": candidate.get("camera_coverage_fraction"),
                    "source": candidate.get("source"),
                }
            )
        top_k.sort(key=lambda item: (int(item["rank"]), str(item["label"])))
        result[window_id] = {"top_k": top_k}
    return result


def _join_rows(joined: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = _sequence(joined.get("windows", []), field="joined.windows")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(raw):
        row = _mapping(value, field=f"joined.windows[{index}]")
        window_id = str(row.get("window_id") or "").strip()
        if window_id:
            if window_id in result:
                raise ProductionWemmQwenEvaluationError(
                    f"duplicate joined verifier window_id: {window_id}"
                )
            result[window_id] = row
    return result


def _rank_for(expected: Sequence[str], top_k: Sequence[Mapping[str, Any]]) -> int | None:
    ranks = [int(row["rank"]) for row in top_k if row.get("label") in expected]
    return min(ranks) if ranks else None


def _rate(hits: Sequence[bool], denominator: int) -> float:
    return sum(hits) / denominator if denominator else 0.0


def _finite_number(value: object) -> float | None:
    """Return a finite float without treating booleans as measurements."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _interval(value: object) -> tuple[float, float] | None:
    """Read either production or verifier interval spellings."""

    if not isinstance(value, Mapping):
        return None
    start = _finite_number(
        value.get("start_seconds", value.get("start_time_sec", value.get("start")))
    )
    end = _finite_number(value.get("end_seconds", value.get("end_time_sec", value.get("end"))))
    if start is None or end is None or end <= start:
        return None
    return start, end


def _window_bounds(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    start = _finite_number(row.get("window_start_seconds", row.get("start_seconds")))
    end = _finite_number(row.get("window_end_seconds", row.get("end_seconds")))
    if start is None or end is None or end <= start:
        return None, None
    return start, end


def _boundary_valid(
    boundary: object, *, duration_seconds: float | None
) -> tuple[bool, tuple[float, float] | None, str]:
    """Validate a verifier boundary structurally, without judging semantics."""

    if not isinstance(boundary, Mapping):
        return False, None, "missing"
    status = str(boundary.get("status") or "").strip().casefold()
    interval = _interval(boundary)
    if status != "measured" or interval is None:
        return False, None, status or "not_measured"
    if interval[0] < 0 or (duration_seconds is not None and interval[1] > duration_seconds):
        return False, interval, "out_of_window"
    return True, interval, "measured"


def _overlap_positive(left: tuple[float, float] | None, right: tuple[float, float] | None) -> bool:
    if left is None or right is None:
        return False
    return min(left[1], right[1]) > max(left[0], right[0])


def _selected_verdict(
    parsed: Mapping[str, Any], selected_rank: int | None
) -> Mapping[str, Any] | None:
    raw = parsed.get("candidate_verdicts", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    verdicts = [item for item in raw if isinstance(item, Mapping)]
    if selected_rank is not None:
        for item in verdicts:
            rank = item.get("rank")
            if isinstance(rank, int) and not isinstance(rank, bool) and rank == selected_rank:
                return item
    return verdicts[0] if len(verdicts) == 1 else None


def _metric_block(
    rows: Sequence[Mapping[str, Any]], *, denominator: int | None = None
) -> dict[str, Any]:
    """Small stratum summary used to prevent denominator ambiguity."""

    total = len(rows) if denominator is None else denominator
    accepted = sum(bool(row.get("accepted")) for row in rows)
    selected_match = sum(bool(row.get("selected_match")) for row in rows)
    accepted_match = sum(bool(row.get("accepted_and_match")) for row in rows)
    selected = sum(row.get("selected_rank") is not None for row in rows)
    return {
        "windows": total,
        "accepted_count": accepted,
        "selected_count": selected,
        "selected_match_count": selected_match,
        "accepted_match_count": accepted_match,
        "accepted_rate": accepted / total if total else 0.0,
        "selected_match_rate": selected_match / total if total else 0.0,
        "accepted_surrogate_agreement": accepted_match / accepted if accepted else None,
    }


def _decision(value: object) -> str:
    raw = str(value or "").strip().casefold()
    return {
        "accepted": "accept",
        "accepted_action": "accept",
        "edited": "edit",
        "abstained": "abstain",
        "rejected": "reject",
    }.get(raw, raw or "abstain")


def _qwen_camera_rows(verifier: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = verifier.get("camera_reports", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _qwen_native_complete(verifier: Mapping[str, Any]) -> bool:
    """Use the joined aggregate, falling back to camera-level provenance."""

    if "qwen_native_video_complete" in verifier:
        return bool(verifier.get("qwen_native_video_complete"))
    cameras = _qwen_camera_rows(verifier)
    return bool(cameras) and all(bool(row.get("native_video_complete")) for row in cameras)


def _qwen_parse_valid(parsed: Mapping[str, Any], verifier: Mapping[str, Any]) -> bool:
    status = str(parsed.get("parse_status") or "").strip().upper()
    if status:
        return status == "PARSED"
    cameras = _qwen_camera_rows(verifier)
    if cameras:
        statuses = [
            str(row.get("parsed_verification", {}).get("parse_status") or "").upper()
            for row in cameras
            if isinstance(row.get("parsed_verification"), Mapping)
        ]
        return bool(statuses) and all(value == "PARSED" for value in statuses)
    return False


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": round(statistics.mean(clean), 6),
        "median": round(statistics.median(clean), 6),
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
    }


def _recorded_cost_summary(
    qwen_sidecar: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize recorded Qwen runtime work without invoking model/media.

    This is deliberately a *recorded* cost block: values come only from the
    supplied native-runner sidecar.  Missing or malformed measurements remain
    ``None`` rather than being inferred as zero, while row counts still make
    the provenance surface auditable.
    """

    if qwen_sidecar is None:
        return {
            "status": "NOT_MEASURED",
            "model_load_seconds": None,
            "generation_seconds_total": None,
            "generation_seconds_mean": None,
            "generation_seconds_median": None,
            "output_tokens_total": None,
            "camera_rows": 0,
            "window_rows": 0,
            "elapsed_seconds": None,
        }

    model = qwen_sidecar.get("model")
    model_map = model if isinstance(model, Mapping) else {}
    raw_rows = qwen_sidecar.get("windows", [])
    rows = (
        [item for item in raw_rows if isinstance(item, Mapping)]
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes, bytearray))
        else []
    )
    generation_values = [
        value
        for item in rows
        for value in [_finite_number(item.get("generation_seconds"))]
        if value is not None
    ]
    token_values: list[int] = []
    for item in rows:
        value = item.get("output_tokens")
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            token_values.append(value)
        elif (
            isinstance(value, float) and math.isfinite(value) and value >= 0 and value.is_integer()
        ):
            token_values.append(int(value))
    window_ids = {
        str(item.get("window_id")).strip()
        for item in rows
        if str(item.get("window_id") or "").strip()
    }
    model_load = _finite_number(model_map.get("load_seconds"))
    elapsed = _finite_number(qwen_sidecar.get("elapsed_seconds"))
    measured = bool(
        generation_values or token_values or model_load is not None or elapsed is not None
    )
    return {
        "status": "RECORDED" if measured else "NOT_MEASURED",
        "model_load_seconds": model_load,
        "generation_seconds_total": round(sum(generation_values), 6) if generation_values else None,
        "generation_seconds_mean": (
            round(statistics.mean(generation_values), 6) if generation_values else None
        ),
        "generation_seconds_median": (
            round(statistics.median(generation_values), 6) if generation_values else None
        ),
        "output_tokens_total": sum(token_values) if token_values else None,
        "camera_rows": len(rows),
        "window_rows": len(window_ids),
        "elapsed_seconds": elapsed,
    }


def evaluate_wemm_qwen_candidate_verifier(
    reference: Mapping[str, Any] | str | Path,
    candidate_pack: Mapping[str, Any] | str | Path,
    joined_verifier: Mapping[str, Any] | str | Path,
    qwen_sidecar: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval and selected-only verifier outcomes separately.

    The independent Terra review is a *surrogate reference*.  Its labels are
    useful for development diagnostics only; no metric returned here is
    production accuracy or official precision/recall.  Abstain and split
    rows stay visible as separate behavior strata instead of disappearing
    from an eligible denominator.
    """

    reference_payload = load_json(reference)
    pack_payload = load_json(candidate_pack)
    joined_payload = load_json(joined_verifier)
    qwen_payload = load_json(qwen_sidecar) if qwen_sidecar is not None else None
    recorded_cost = _recorded_cost_summary(qwen_payload)
    refs = _reference_rows(reference_payload)
    packs = _pack_rows(pack_payload)
    joined = _join_rows(joined_payload)
    rows: list[dict[str, Any]] = []

    for window_id, ref in refs.items():
        expected = list(ref["expected"])
        expected_set = set(expected)
        top_k = packs.get(window_id, {}).get("top_k", [])
        verifier = joined.get(window_id, {})
        parsed = verifier.get("parsed_verification")
        parsed_map = (
            _mapping(parsed, field=f"{window_id}.parsed_verification")
            if isinstance(parsed, Mapping)
            else {}
        )
        raw_selected_rank = parsed_map.get("selected_rank")
        selected_rank = (
            raw_selected_rank
            if isinstance(raw_selected_rank, int) and not isinstance(raw_selected_rank, bool)
            else None
        )
        selected = next(
            (candidate for candidate in top_k if candidate.get("rank") == selected_rank), None
        )
        selected_label = selected.get("label") if selected else None
        selected_verdict = _selected_verdict(parsed_map, selected_rank)
        selected_support = (
            str(selected_verdict.get("support") or "").strip().casefold()
            if selected_verdict is not None
            else ""
        )
        decision = _decision(verifier.get("decision", parsed_map.get("decision", "abstain")))
        retrieval_rank = _rank_for(expected, top_k)
        candidate_labels = {str(candidate.get("label")) for candidate in top_k}
        matching_ranks = [
            int(candidate["rank"]) for candidate in top_k if candidate.get("label") in expected_set
        ]
        all_expected_in_top_k = bool(expected_set) and expected_set.issubset(candidate_labels)
        retrieval_all_rank = max(matching_ranks) if all_expected_in_top_k else None
        eligible = ref["decision"] in _ELIGIBLE_DECISIONS and bool(expected)
        selected_match = bool(selected_label and selected_label in expected_set)
        accepted = decision == "accept"

        native_complete = _qwen_native_complete(verifier)
        parse_valid = _qwen_parse_valid(parsed_map, verifier)
        camera_rows = _qwen_camera_rows(verifier)
        camera_native_count = sum(bool(item.get("native_video_complete")) for item in camera_rows)
        camera_parse_count = sum(
            isinstance(item.get("parsed_verification"), Mapping)
            and str(item["parsed_verification"].get("parse_status") or "").upper() == "PARSED"
            for item in camera_rows
        )

        raw_segments = parsed_map.get("segments", [])
        parsed_segments = (
            [item for item in raw_segments if isinstance(item, Mapping)]
            if isinstance(raw_segments, Sequence)
            and not isinstance(raw_segments, (str, bytes, bytearray))
            else []
        )
        supported_labels: list[str] = []
        raw_verdicts = parsed_map.get("candidate_verdicts", [])
        if isinstance(raw_verdicts, Sequence) and not isinstance(
            raw_verdicts, (str, bytes, bytearray)
        ):
            for raw_verdict in raw_verdicts:
                if not isinstance(raw_verdict, Mapping):
                    continue
                if str(raw_verdict.get("support") or "").strip().casefold() != "supported":
                    continue
                rank = raw_verdict.get("rank")
                candidate = next((item for item in top_k if item.get("rank") == rank), None)
                if candidate and candidate.get("label"):
                    label = str(candidate["label"])
                    if label not in supported_labels:
                        supported_labels.append(label)
        if (
            selected_label
            and selected_support == "supported"
            and selected_label not in supported_labels
        ):
            supported_labels.append(str(selected_label))

        emitted_labels: list[str] = []
        for segment in parsed_segments:
            rank = segment.get("candidate_rank", segment.get("rank"))
            candidate = next((item for item in top_k if item.get("rank") == rank), None)
            if candidate and candidate.get("label"):
                label = str(candidate["label"])
                if label not in emitted_labels:
                    emitted_labels.append(label)
        expected_supported_count = len(expected_set.intersection(supported_labels))
        expected_emitted_count = len(expected_set.intersection(emitted_labels))
        expected_segment_count = len(expected)
        split_undersegmented = ref["decision"] == "split" and (
            decision != "split" or expected_emitted_count < expected_segment_count
        )
        split_unsupported = ref["decision"] == "split" and expected_supported_count == 0
        split_exact = (
            ref["decision"] == "split" and not split_undersegmented and not split_unsupported
        )

        top1 = top_k[0] if top_k else None
        top1_label = top1.get("label") if top1 else None
        top1_match = bool(top1_label and top1_label in expected_set)
        score_values = [_finite_number(candidate.get("score")) for candidate in top_k[:2]]
        margin = (
            score_values[0] - score_values[1]
            if len(score_values) == 2
            and score_values[0] is not None
            and score_values[1] is not None
            else None
        )
        top1_camera_coverage = _finite_number(top1.get("camera_coverage") if top1 else None)
        top1_camera_coverage_fraction = _finite_number(
            top1.get("camera_coverage_fraction") if top1 else None
        )

        selected_boundary = selected_verdict.get("boundary") if selected_verdict else None
        window_start = ref.get("window_start_seconds")
        window_end = ref.get("window_end_seconds")
        duration = (
            window_end - window_start
            if isinstance(window_start, (int, float))
            and isinstance(window_end, (int, float))
            and window_end > window_start
            else None
        )
        boundary_ok, boundary_interval, boundary_status = _boundary_valid(
            selected_boundary, duration_seconds=duration
        )
        boundary_overlap = False
        if boundary_ok and boundary_interval is not None:
            for segment in ref.get("segments", []):
                interval = segment.get("interval") if isinstance(segment, Mapping) else None
                if interval is None:
                    continue
                # Verifier boundaries are window-relative; Terra review times
                # are source-absolute when the window bounds are known.
                if duration is not None and window_start is not None:
                    interval = (interval[0] - window_start, interval[1] - window_start)
                if _overlap_positive(boundary_interval, interval):
                    boundary_overlap = True
                    break

        verifier_reasons = verifier.get("reason_codes", [])
        if not isinstance(verifier_reasons, Sequence) or isinstance(
            verifier_reasons, (str, bytes, bytearray)
        ):
            verifier_reasons = []
        rows.append(
            {
                "window_id": window_id,
                "reference_decision": ref["decision"],
                "reference_stratum": (
                    "abstain"
                    if ref["decision"] == "abstain"
                    else "split"
                    if ref["decision"] == "split"
                    else "eligible"
                    if eligible
                    else "other"
                ),
                "expected_actions": expected,
                "expected_action_count": expected_segment_count,
                "eligible": eligible,
                "retrieval_rank": retrieval_rank,
                "retrieval_all_rank": retrieval_all_rank,
                "retrieval_match_count": len(expected_set.intersection(candidate_labels)),
                "retrieval_all_expected_in_top_k": all_expected_in_top_k,
                "top_k_count": len(top_k),
                "top1_action": top1_label,
                "top1_match": top1_match,
                "verifier_decision": decision,
                "selected_rank": selected_rank,
                "selected_action": selected_label,
                "selected_match": selected_match,
                "selected_supported": selected_support == "supported",
                "selected_in_top_k": selected is not None,
                "accepted": accepted,
                "accepted_and_match": accepted and selected_match,
                "native_video_complete": native_complete,
                "parse_valid": parse_valid,
                "parse_status": str(parsed_map.get("parse_status") or "MISSING").upper(),
                "camera_count": len(camera_rows),
                "camera_native_complete_count": camera_native_count,
                "camera_parse_valid_count": camera_parse_count,
                "supported_actions": supported_labels,
                "emitted_actions": emitted_labels,
                "split_expected_supported_count": expected_supported_count,
                "split_expected_emitted_count": expected_emitted_count,
                "split_undersegmented": split_undersegmented,
                "split_unsupported": split_unsupported,
                "split_exact": split_exact,
                "margin_top1_top2": margin,
                "top1_camera_coverage": top1_camera_coverage,
                "top1_camera_coverage_fraction": top1_camera_coverage_fraction,
                "boundary_valid": boundary_ok,
                "boundary_status": boundary_status,
                "boundary_overlaps_surrogate": boundary_overlap,
                "verifier_reasons": [str(item) for item in verifier_reasons],
            }
        )

    all_rows = list(rows)
    eligible_rows = [row for row in rows if row["eligible"]]
    abstain_rows = [row for row in rows if row["reference_stratum"] == "abstain"]
    split_rows = [row for row in rows if row["reference_stratum"] == "split"]
    denominator = len(eligible_rows)
    all_denominator = len(all_rows)
    ranks = [row["retrieval_rank"] for row in eligible_rows]

    retrieval: dict[str, Any] = {}
    for cutoff in (1, 3, 5, 10):
        retrieval[f"recall_at_{cutoff}"] = _rate(
            [rank is not None and rank <= cutoff for rank in ranks], denominator
        )
        retrieval[f"all_label_recall_at_{cutoff}"] = _rate(
            [
                row["retrieval_all_rank"] is not None and row["retrieval_all_rank"] <= cutoff
                for row in eligible_rows
            ],
            denominator,
        )
    retrieval["mrr"] = (
        sum(1.0 / rank for rank in ranks if rank is not None) / denominator if denominator else 0.0
    )
    retrieval["all_label_mrr"] = (
        sum(
            1.0 / row["retrieval_all_rank"]
            for row in eligible_rows
            if row["retrieval_all_rank"] is not None
        )
        / denominator
        if denominator
        else 0.0
    )
    retrieval["miss_count_at_10"] = sum(rank is None or rank > 10 for rank in ranks)
    retrieval["verifier_headroom_at_10"] = _rate(
        [rank is not None and rank <= 10 for rank in ranks], denominator
    )
    margins = [
        row["margin_top1_top2"] for row in eligible_rows if row["margin_top1_top2"] is not None
    ]
    coverages = [
        row["top1_camera_coverage"]
        for row in eligible_rows
        if row["top1_camera_coverage"] is not None
    ]
    coverage_fractions = [
        row["top1_camera_coverage_fraction"]
        for row in eligible_rows
        if row["top1_camera_coverage_fraction"] is not None
    ]
    retrieval["margin_top1_top2"] = _numeric_summary(margins)
    retrieval["top1_camera_coverage"] = _numeric_summary(coverages)
    retrieval["top1_camera_coverage_fraction"] = _numeric_summary(coverage_fractions)
    retrieval["rank_distribution"] = dict(sorted(Counter(str(rank) for rank in ranks).items()))
    retrieval["denominator_windows"] = denominator

    def _count(rows_value: Sequence[Mapping[str, Any]], key: str) -> int:
        return sum(bool(row.get(key)) for row in rows_value)

    accepted_count = _count(eligible_rows, "accepted")
    accepted_match_count = _count(eligible_rows, "accepted_and_match")
    selected_match_count = _count(eligible_rows, "selected_match")
    verifier_metrics: dict[str, Any] = {
        "accepted_count": accepted_count,
        "abstain_or_nonaccept_count": denominator - accepted_count,
        "accepted_match_count": accepted_match_count,
        "selected_match_count": selected_match_count,
        "accepted_coverage": _rate([bool(row["accepted"]) for row in eligible_rows], denominator),
        "selected_coverage": _rate(
            [bool(row["selected_match"]) for row in eligible_rows], denominator
        ),
        "accepted_precision": accepted_match_count / accepted_count if accepted_count else 0.0,
        "surrogate_accepted_precision": accepted_match_count / accepted_count
        if accepted_count
        else None,
        "strict_visual_joint": _rate(
            [bool(row["accepted_and_match"]) for row in eligible_rows], denominator
        ),
        "surrogate_strict_visual_joint": _rate(
            [bool(row["accepted_and_match"]) for row in eligible_rows], denominator
        ),
        "selected_in_top_k": {
            "hits": _count(eligible_rows, "selected_in_top_k"),
            "windows": denominator,
            "rate": _rate([bool(row["selected_in_top_k"]) for row in eligible_rows], denominator),
        },
        "selected_supported": {
            "hits": _count(eligible_rows, "selected_supported"),
            "windows": denominator,
            "rate": _rate([bool(row["selected_supported"]) for row in eligible_rows], denominator),
        },
        "native_video_complete": {
            "hits": _count(eligible_rows, "native_video_complete"),
            "windows": denominator,
            "rate": _rate(
                [bool(row["native_video_complete"]) for row in eligible_rows], denominator
            ),
        },
        "parse_valid": {
            "hits": _count(eligible_rows, "parse_valid"),
            "windows": denominator,
            "rate": _rate([bool(row["parse_valid"]) for row in eligible_rows], denominator),
        },
        "selected_rank_distribution": dict(
            sorted(Counter(str(row["selected_rank"]) for row in all_rows).items())
        ),
        "selected_rank1_prior_rate": _rate(
            [row["selected_rank"] == 1 for row in eligible_rows], denominator
        ),
        "selected_rank1_prior_rate_all": _rate(
            [row["selected_rank"] == 1 for row in all_rows], all_denominator
        ),
    }
    rescue_rows = [
        row
        for row in eligible_rows
        if not row["top1_match"]
        and row["selected_match"]
        and isinstance(row["selected_rank"], int)
        and row["selected_rank"] > 1
    ]
    regression_rows = [
        row for row in eligible_rows if row["top1_match"] and not row["selected_match"]
    ]
    verifier_metrics["rescue"] = {
        "count": len(rescue_rows),
        "eligible_windows": denominator,
        "window_ids": [row["window_id"] for row in rescue_rows],
    }
    verifier_metrics["regression"] = {
        "count": len(regression_rows),
        "eligible_windows": denominator,
        "window_ids": [row["window_id"] for row in regression_rows],
    }
    verifier_metrics["camera_provenance"] = {
        "camera_rows": sum(int(row["camera_count"]) for row in all_rows),
        "native_complete_camera_rows": sum(
            int(row["camera_native_complete_count"]) for row in all_rows
        ),
        "parse_valid_camera_rows": sum(int(row["camera_parse_valid_count"]) for row in all_rows),
    }

    abstain_overaccept_rows = [
        row
        for row in abstain_rows
        if row["verifier_decision"] in {"accept", "edit", "split"} or row["selected_supported"]
    ]
    verifier_metrics["abstain_invariant"] = {
        "expected": len(abstain_rows),
        "hits": sum(row["verifier_decision"] == "abstain" for row in abstain_rows),
        "rate": _rate(
            [row["verifier_decision"] == "abstain" for row in abstain_rows],
            len(abstain_rows),
        ),
        "overaccept_count": len(abstain_overaccept_rows),
        "overaccept_rate": _rate(
            [row in abstain_overaccept_rows for row in abstain_rows], len(abstain_rows)
        ),
        "supported_selection_count": sum(row["selected_supported"] for row in abstain_rows),
        "window_ids": [row["window_id"] for row in abstain_overaccept_rows],
    }
    verifier_metrics["split_invariant"] = {
        "expected": len(split_rows),
        "hits": sum(row["split_exact"] for row in split_rows),
        "rate": _rate([row["split_exact"] for row in split_rows], len(split_rows)),
        "decision_split_count": sum(row["verifier_decision"] == "split" for row in split_rows),
        "unsupported_count": sum(row["split_unsupported"] for row in split_rows),
        "undersegmented_count": sum(row["split_undersegmented"] for row in split_rows),
        "window_ids": [row["window_id"] for row in split_rows if row["split_undersegmented"]],
    }
    all_mechanical = {
        "windows": all_denominator,
        "selected_in_top_k": _rate(
            [bool(row["selected_in_top_k"]) for row in all_rows], all_denominator
        ),
        "selected_supported": _rate(
            [bool(row["selected_supported"]) for row in all_rows], all_denominator
        ),
        "native_video_complete": _rate(
            [bool(row["native_video_complete"]) for row in all_rows], all_denominator
        ),
        "parse_valid": _rate([bool(row["parse_valid"]) for row in all_rows], all_denominator),
    }
    metrics: dict[str, Any] = {
        # Keep denominator_windows as the historical eligible denominator;
        # expose the all-source population explicitly to prevent ambiguity.
        "denominator_windows": denominator,
        "eligible_denominator_windows": denominator,
        "all_source_windows": all_denominator,
        "all_source_denominator_windows": all_denominator,
        "strata": {
            "eligible": {"windows": denominator, "reference_decision_excluded": False},
            "abstain": {"windows": len(abstain_rows), "reference_decision_excluded": True},
            "split": {"windows": len(split_rows), "reference_decision_excluded": False},
        },
        "all_source_mechanical": all_mechanical,
        "cost": recorded_cost,
        "retrieval": retrieval,
        "verifier": verifier_metrics,
    }
    return {
        "format": VERSION,
        "authority": AUTHORITY,
        "status": "SURROGATE_ONLY",
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        # The reference is an independent surrogate review, not official
        # gold; no annotation accuracy is measured by this report.
        "accuracy_status": "NOT_MEASURED",
        "official_gold_status": OFFICIAL_GOLD_STATUS,
        "quality_claim": False,
        "production_eligible": False,
        "metrics": metrics,
        "windows": rows,
        "provenance": {
            "reference": str(reference) if isinstance(reference, (str, Path)) else "inline",
            "candidate_pack": str(candidate_pack)
            if isinstance(candidate_pack, (str, Path))
            else "inline",
            "joined_verifier": str(joined_verifier)
            if isinstance(joined_verifier, (str, Path))
            else "inline",
            "qwen_sidecar": str(qwen_sidecar)
            if isinstance(qwen_sidecar, (str, Path))
            else ("inline" if qwen_sidecar is not None else None),
            "reference_status": "INDEPENDENT_SURROGATE_REFERENCE",
            "official_gold_status": OFFICIAL_GOLD_STATUS,
            "epic_ontology_used": False,
            "mapper_used": False,
        },
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_digest_computed": False,
        },
        "cost": recorded_cost,
        "limitations": [
            "Terra review is an independent surrogate, not official production gold.",
            "All overlap/agreement values are surrogate diagnostics; they are not "
            "annotation accuracy or official precision/recall.",
            "Abstain windows are excluded from label-recall denominators but are "
            "reported as an explicit behavior stratum.",
            "Split windows require multi-label/multi-segment output; a selected-only "
            "verifier is expected to be undersegmented there.",
            "Boundary checks are structural and overlap-only; fixed review windows "
            "are not action boundaries.",
            "Camera availability, native provenance, and parser status are mechanical "
            "checks, not semantic correctness.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics", {}), field="report.metrics")
    retrieval = _mapping(metrics.get("retrieval", {}), field="report.metrics.retrieval")
    verifier = _mapping(metrics.get("verifier", {}), field="report.metrics.verifier")
    cost = _mapping(metrics.get("cost", report.get("cost", {})), field="report.metrics.cost")
    lines = [
        "# WeMM → Qwen candidate-verifier evaluation",
        "",
        f"Status: `{report.get('status', 'SURROGATE_ONLY')}`; official quality: `NOT_MEASURED`.",
        f"Accuracy status: `{report.get('accuracy_status', 'NOT_MEASURED')}`.",
        f"Denominator: `{metrics.get('denominator_windows', 0)}` eligible windows; "
        f"all source windows: `{metrics.get('all_source_windows', 0)}`.",
        "",
        "## Separate metrics",
        "",
        "| Route | R@1 / accepted coverage | R@3 / selected coverage | "
        "R@5 / accepted precision | R@10 / strict visual joint | MRR |",
        "|---|---:|---:|---:|---:|---:|",
        f"| WeMM retrieval | {float(retrieval.get('recall_at_1', 0.0)):.3f} | "
        f"{float(retrieval.get('recall_at_3', 0.0)):.3f} | "
        f"{float(retrieval.get('recall_at_5', 0.0)):.3f} | "
        f"{float(retrieval.get('recall_at_10', 0.0)):.3f} | "
        f"{float(retrieval.get('mrr', 0.0)):.3f} |",
        f"| Qwen selected-only verifier | {float(verifier.get('accepted_coverage', 0.0)):.3f} | "
        f"{float(verifier.get('selected_coverage', 0.0)):.3f} | "
        f"{float(verifier.get('accepted_precision', 0.0)):.3f} | "
        f"{float(verifier.get('strict_visual_joint', 0.0)):.3f} | — |",
        "",
        "## Recorded runtime cost (non-gold)",
        "",
        f"Status: `{cost.get('status', 'NOT_MEASURED')}`; model load seconds: "
        f"`{cost.get('model_load_seconds')}`; elapsed seconds: `{cost.get('elapsed_seconds')}`.",
        f"Generation total/mean/median seconds: `{cost.get('generation_seconds_total')}` / "
        f"`{cost.get('generation_seconds_mean')}` / `{cost.get('generation_seconds_median')}`; "
        f"output tokens total: `{cost.get('output_tokens_total')}`; camera rows: "
        f"`{cost.get('camera_rows')}`; window rows: `{cost.get('window_rows')}`.",
        "Values are copied from the optional recorded Qwen sidecar; no model or media "
        "work is performed by this evaluator.",
        "",
        "## Per-window outcomes",
        "",
        "| Window | Terra reference | WeMM rank | Qwen decision | Qwen selected | Match |",
        "|---|---|---:|---|---|---|",
    ]
    for row in report.get("windows", []):
        if not isinstance(row, Mapping):
            continue
        expected = ", ".join(str(x) for x in row.get("expected_actions", [])) or "—"
        rank = row.get("retrieval_rank") if row.get("retrieval_rank") is not None else "—"
        selected = row.get("selected_action") or "—"
        lines.append(
            f"| {row.get('window_id', '')} | {expected} "
            f"({row.get('reference_decision', '')}) | {rank} | "
            f"{row.get('verifier_decision', 'abstain')} | {selected} | "
            f"{bool(row.get('selected_match'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Retrieval recall measures whether the approved production action is present "
            "in WeMM Top-K.",
            "At R@10=1.0, the verifier has full candidate headroom on this cohort; "
            "selection errors are therefore downstream of retrieval.",
            "- Accepted precision/coverage measure whether Qwen selected a supported "
            "candidate; they are not official annotation accuracy.",
            "- `epic_ontology_used=false` and `mapper_used=false`; old EPIC/provisional "
            "sidecars remain quarantined.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "VERSION",
    "ProductionWemmQwenEvaluationError",
    "action_label",
    "evaluate_wemm_qwen_candidate_verifier",
    "load_json",
    "render_markdown",
]
