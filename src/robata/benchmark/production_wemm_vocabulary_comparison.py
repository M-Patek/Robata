"""Post-hoc comparison of production-vocabulary WeMM text variants.

This module reads completed production-vocabulary WeMM sidecars and an
owner-scoped review reference.  It performs no model invocation, media decode,
ontology/Mapper mutation, or identity/hash work.  The owner review is a
surrogate reference (official production gold is not established), therefore
all reported values are explicitly exploratory and non-gold.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

PRODUCTION_WEMM_VARIANT_COMPARISON_VERSION: Final = (
    "robata-production-wemm-vocabulary-variant-comparison-v1"
)
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "SURROGATE_ONLY"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
DEFAULT_KS: Final = (1, 3, 5, 10)
DEFAULT_OWNER_REVIEW = (
    # The independent source-bound review is the default surrogate reference.
    # Owner-confirmation artifacts remain useful for a separate replay, but
    # must not silently become the baseline for this comparison: they may have
    # been produced after seeing machine candidates.  Neither artifact is
    # official gold.
    ".agent_tmp/terra_independent_production_review_4s_16f_20260827.json"
)
DEFAULT_SIDECARS: Final = {
    "canonical": ".agent_tmp/production_wemm_production_vocab_4s_20260827.json",
    "verb_noun": ".agent_tmp/production_wemm_production_vocab_4s_verb_noun_20260827.json",
    "natural": ".agent_tmp/production_wemm_production_vocab_4s_natural_20260827.json",
}


class ProductionWemmVariantComparisonError(ValueError):
    """Raised when a sidecar/reference cannot be compared safely."""


def _finite_number(value: object) -> float | None:
    """Return a finite numeric value without making optional diagnostics fatal.

    Camera consensus is an additive diagnostic copied from a completed WeMM
    sidecar.  Older sidecars and partially written diagnostics may omit or
    contain non-numeric scores; those values should be represented as missing,
    not silently coerced or allowed to poison the comparison report.
    """

    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmVariantComparisonError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmVariantComparisonError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionWemmVariantComparisonError(f"{field} must be a string")
    result = unicodedata.normalize("NFKC", value).strip()
    if not result and not allow_empty:
        raise ProductionWemmVariantComparisonError(f"{field} must be non-empty")
    return result


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _verb(value: object) -> str:
    text = _normalise(value)
    if text in {"pick up", "pickup", "picks up", "picking up"}:
        return "pick up"
    forms = {
        "adjusting": "adjust",
        "adjusts": "adjust",
        "flattening": "flatten",
        "flattens": "flatten",
        "folding": "fold",
        "folds": "fold",
        "picking": "pick",
        "picks": "pick",
        "smoothing": "smooth",
        "smooths": "smooth",
        "spreading": "spread",
        "spreads": "spread",
    }
    tokens = text.split()
    if tokens:
        tokens[-1] = forms.get(tokens[-1], tokens[-1])
    return " ".join(tokens)


def _noun(value: object) -> str:
    noun = _normalise(value)
    return (
        "garment"
        if noun
        in {
            "cloth",
            "clothes",
            "clothing",
            "fabric",
            "garment",
            "pants",
            "shirt",
            "shorts",
            "sheets",
        }
        else noun
    )


def _pair(verb: object, noun: object) -> tuple[str, str] | None:
    result = (_verb(verb), _noun(noun))
    return result if result[0] and result[1] else None


def _load_json(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductionWemmVariantComparisonError(
                f"could not read comparison input {path}: {exc}"
            ) from exc
        return _mapping(payload, field=str(path))
    return value


def _source_refs(payload: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []

    def visit(value: object, *, context: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).casefold()
                child_context = context or key_text in {
                    "source",
                    "input",
                    "cohort",
                    "manifest_source",
                }
                if (
                    child_context
                    and key_text
                    in {
                        "path",
                        "media_path",
                        "source_path",
                        "mcap_path",
                        "video_path",
                        "manifest",
                    }
                    and isinstance(child, str)
                    and child.strip()
                ):
                    refs.append(child.strip())
                else:
                    visit(child, context=child_context)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child, context=context)

    visit(payload)
    return list(dict.fromkeys(refs))


def _source_token(value: str) -> str:
    return value.replace("\\", "/").casefold().rstrip("/")


def _source_binding(
    reference: Mapping[str, Any],
    sidecars: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_role = {"owner_review": _source_refs(reference)}
    for name, sidecar in sidecars.items():
        by_role[name] = _source_refs(sidecar)
    media_by_role = {
        role: [item for item in refs if _source_token(item).endswith((".mcap", ".mp4"))]
        for role, refs in by_role.items()
    }
    media = [item for refs in media_by_role.values() for item in refs]
    tokens = [_source_token(item) for item in media]
    status = "UNRESOLVED" if not media else "MATCHED"
    if len(tokens) > 1:
        for left in tokens:
            for right in tokens:
                if left == right or left.endswith("/" + right) or right.endswith("/" + left):
                    continue
                status = "CONFLICT"
    return {
        "status": status,
        "references_by_role": by_role,
        "media_references_by_role": media_by_role,
        "source_identity_derived": False,
    }


def _reference_windows(review: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = review.get("windows", review.get("items"))
    rows = _sequence(raw, field="owner_review.windows")
    result: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, field=f"owner_review.windows[{index}]")
        window_id = _text(row.get("window_id"), field=f"owner_review.windows[{index}].window_id")
        # Independent Terra review uses ``recommendation`` (EDIT/SPLIT/
        # ABSTAIN), while owner-confirmation envelopes use ``decision``.  Read
        # both explicitly; otherwise an ABSTAIN row with a diagnostic segment
        # would be counted as an eligible positive example.
        decision_value: object = row.get("decision")
        if decision_value is None:
            decision_value = row.get("recommendation", "accept")
        decision = str(decision_value).strip().casefold()
        decision = {
            "accepted": "accept",
            "edited": "edit",
            "split": "split",
            "rejected": "reject",
            "abstained": "abstain",
        }.get(decision, decision)
        segments_value = row.get("segments")
        if segments_value is None and isinstance(row.get("gold"), Mapping):
            segments_value = row["gold"].get("segments", [])
            decision_value = row["gold"].get("status", decision)
            decision = str(decision_value).strip().casefold()
            decision = {
                "accepted": "accept",
                "edited": "edit",
                "split": "split",
                "rejected": "reject",
                "abstained": "abstain",
            }.get(decision, decision)
        segments = _sequence(segments_value or (), field=f"{window_id}.segments")
        pairs: list[tuple[str, str]] = []
        for segment_index, raw_segment in enumerate(segments):
            segment = _mapping(raw_segment, field=f"{window_id}.segments[{segment_index}]")
            pair = _pair(segment.get("verb", segment.get("verb_code")), segment.get("noun"))
            if pair is not None:
                pairs.append(pair)
        if window_id in result:
            raise ProductionWemmVariantComparisonError(f"duplicate review window {window_id}")
        result[window_id] = {
            "window_id": window_id,
            "decision": decision,
            "pairs": list(dict.fromkeys(pairs)),
        }
    return result


def _reference_status(review: Mapping[str, Any]) -> str:
    """Describe the surrogate source used for metrics without calling it gold."""

    format_raw = review.get("format")
    state_raw = review.get("review_state")
    format_value = format_raw.casefold() if isinstance(format_raw, str) else ""
    review_state = state_raw.casefold() if isinstance(state_raw, str) else ""
    provenance = review.get("provenance")
    method = (
        provenance.get("method", "").casefold()
        if isinstance(provenance, Mapping) and isinstance(provenance.get("method"), str)
        else ""
    )
    if "independent" in format_value or "independent" in review_state or "label-blind" in method:
        return "INDEPENDENT_SURROGATE_REFERENCE"
    return "OWNER_SCOPED_SURROGATE_REFERENCE"


def _candidate_pair(candidate: Mapping[str, Any]) -> tuple[str, str] | None:
    return _pair(
        candidate.get("verb", candidate.get("verb_key")),
        candidate.get("noun", candidate.get("noun_key")),
    )


def _sidecar_candidates(sidecar: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw_windows = _sequence(sidecar.get("windows"), field="sidecar.windows")
    result: dict[str, list[dict[str, Any]]] = {}
    for index, raw_window in enumerate(raw_windows):
        window = _mapping(raw_window, field=f"sidecar.windows[{index}]")
        window_id = _text(window.get("window_id"), field=f"sidecar.windows[{index}].window_id")
        model = window.get("model")
        model_map = (
            _mapping(model, field=f"{window_id}.model") if isinstance(model, Mapping) else window
        )
        raw_predictions = model_map.get("predictions", model_map.get("candidates", []))
        predictions = _sequence(raw_predictions, field=f"{window_id}.predictions")
        ordered: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for candidate_index, raw_candidate in enumerate(predictions):
            candidate = _mapping(raw_candidate, field=f"{window_id}.predictions[{candidate_index}]")
            pair = _candidate_pair(candidate)
            if pair is None or pair in seen:
                continue
            seen.add(pair)
            rank_raw = candidate.get("rank", candidate_index + 1)
            try:
                rank = int(rank_raw)
            except (TypeError, ValueError) as exc:
                raise ProductionWemmVariantComparisonError(
                    f"{window_id}.predictions[{candidate_index}].rank must be an integer"
                ) from exc
            if rank <= 0:
                raise ProductionWemmVariantComparisonError(
                    f"{window_id}.predictions[{candidate_index}].rank must be positive"
                )
            ordered.append(
                {
                    "rank": rank,
                    "pair": list(pair),
                    "label_id": candidate.get("label_id"),
                    "label_text": candidate.get("label_text"),
                    "score": candidate.get("score", candidate.get("fused_score")),
                }
            )
        ordered.sort(
            key=lambda item: (
                int(item["rank"]),
                str(item.get("label_id") or item["pair"]),
            )
        )
        result[window_id] = [
            {**candidate, "rank": index + 1} for index, candidate in enumerate(ordered)
        ]
    return result


def _camera_diagnostics(
    model_map: Mapping[str, Any],
    fused_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project compact camera-consensus evidence from one WeMM window.

    Production WeMM sidecars already retain per-camera rankings, but the
    vocabulary comparison historically discarded them and therefore made it
    impossible to relate a rank error to camera agreement.  This helper is a
    read-only, additive projection: it keeps only camera top-1/top-2 summaries
    and the ranked action keys needed for post-hoc rank analysis.  It never
    invokes a model, decodes media, or treats camera agreement as semantic
    ground truth.

    ``NOT_AVAILABLE`` is intentional for legacy/synthetic sidecars without
    ``per_camera_predictions``.  A partial camera map remains visible as
    ``PARTIAL`` rather than being mistaken for full six-camera consensus.
    """

    raw_per_camera = model_map.get("per_camera_predictions")
    if not isinstance(raw_per_camera, Mapping) or not raw_per_camera:
        fused_action = None
        if fused_candidates:
            pair = fused_candidates[0].get("pair")
            if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes, bytearray)):
                fused_action = " ".join(str(item) for item in pair)
        return {
            "status": "NOT_AVAILABLE",
            "source": "sidecar.per_camera_predictions",
            "camera_ids": [],
            "observed_camera_count": 0,
            "expected_camera_count": None,
            "coverage_fraction": None,
            "top1_votes": [],
            "consensus_winner": None,
            "consensus_winning_votes": 0,
            "consensus_fraction": None,
            "strict_majority": False,
            "fused_top1_action": fused_action,
            "fused_top1_vote_count": 0,
            "fused_top1_vote_fraction": None,
            "per_camera": [],
            "top1_margin_summary": {
                "observed_count": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            },
        }

    # A camera order/coverage block is emitted by the multiview fusion seam.
    # It is optional so this projection remains compatible with earlier
    # sidecars that only recorded per-camera predictions.
    fusion = model_map.get("fusion")
    fusion_map = fusion if isinstance(fusion, Mapping) else {}
    coverage = fusion_map.get("camera_coverage")
    coverage_map = coverage if isinstance(coverage, Mapping) else {}
    expected_raw = coverage_map.get("expected_count")
    try:
        expected_count = int(expected_raw) if expected_raw is not None else None
    except (TypeError, ValueError):
        expected_count = None
    if expected_count is not None and expected_count <= 0:
        expected_count = None
    camera_order_raw = fusion_map.get("camera_order")
    camera_order = (
        [str(item) for item in camera_order_raw]
        if isinstance(camera_order_raw, Sequence)
        and not isinstance(camera_order_raw, (str, bytes, bytearray))
        else []
    )

    per_camera: list[dict[str, Any]] = []
    malformed_count = 0
    for raw_camera_id, raw_predictions in sorted(
        raw_per_camera.items(), key=lambda item: str(item[0])
    ):
        camera_id = str(raw_camera_id)
        if not isinstance(raw_predictions, Sequence) or isinstance(
            raw_predictions, (str, bytes, bytearray)
        ):
            malformed_count += 1
            continue
        parsed: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for index, raw_prediction in enumerate(raw_predictions):
            if not isinstance(raw_prediction, Mapping):
                malformed_count += 1
                continue
            pair = _candidate_pair(raw_prediction)
            if pair is None or pair in seen:
                malformed_count += 1
                continue
            seen.add(pair)
            rank_raw = raw_prediction.get("rank", index + 1)
            try:
                rank = int(rank_raw)
            except (TypeError, ValueError):
                malformed_count += 1
                continue
            if rank <= 0:
                malformed_count += 1
                continue
            score = _finite_number(
                raw_prediction.get(
                    "score",
                    raw_prediction.get(
                        "fused_score",
                        raw_prediction.get("visual_score", raw_prediction.get("visual_cosine")),
                    ),
                )
            )
            parsed.append({"rank": rank, "action": f"{pair[0]} {pair[1]}", "score": score})
        parsed.sort(key=lambda item: (int(item["rank"]), str(item["action"])))
        if not parsed:
            malformed_count += 1
            continue
        # Re-project to contiguous positions.  Sidecars should already be
        # contiguous; using the recorded order keeps this diagnostic useful for
        # legacy lists with sparse rank values without making a quality claim.
        top1 = parsed[0]
        top2 = parsed[1] if len(parsed) > 1 else None
        margin = (
            float(top1["score"]) - float(top2["score"])
            if top1.get("score") is not None and top2 is not None and top2.get("score") is not None
            else None
        )
        per_camera.append(
            {
                "camera_id": camera_id,
                "top1_action": top1["action"],
                "top1_score": top1.get("score"),
                "top2_action": top2["action"] if top2 is not None else None,
                "top2_score": top2.get("score") if top2 is not None else None,
                "top1_top2_margin": margin,
                "ranked_actions": [str(item["action"]) for item in parsed],
            }
        )

    observed_ids = [str(item["camera_id"]) for item in per_camera]
    observed_count = len(per_camera)
    if expected_count is None:
        expected_count = len(camera_order) or observed_count
    coverage_fraction = observed_count / expected_count if expected_count else None
    votes = Counter(str(item["top1_action"]) for item in per_camera)
    ordered_votes = sorted(votes.items(), key=lambda item: (-int(item[1]), item[0]))
    winner = ordered_votes[0][0] if ordered_votes else None
    winning_votes = int(ordered_votes[0][1]) if ordered_votes else 0
    consensus_fraction = winning_votes / observed_count if observed_count else None
    fused_top1 = None
    if fused_candidates:
        pair = fused_candidates[0].get("pair")
        if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes, bytearray)):
            fused_top1 = " ".join(str(item) for item in pair)
    fused_votes = int(votes.get(fused_top1, 0)) if fused_top1 else 0
    margins = [
        float(item["top1_top2_margin"])
        for item in per_camera
        if item.get("top1_top2_margin") is not None
    ]
    margins.sort()
    median_margin = (
        margins[len(margins) // 2]
        if len(margins) % 2
        else (margins[len(margins) // 2 - 1] + margins[len(margins) // 2]) / 2
        if margins
        else None
    )
    missing = [camera for camera in camera_order if camera not in observed_ids]
    status = "AVAILABLE"
    if malformed_count or (expected_count is not None and observed_count < expected_count):
        status = "PARTIAL"
    return {
        "status": status,
        "source": "sidecar.per_camera_predictions",
        "camera_ids": observed_ids,
        "observed_camera_count": observed_count,
        "expected_camera_count": expected_count,
        "coverage_fraction": coverage_fraction,
        "missing_camera_ids": missing,
        "malformed_entry_count": malformed_count,
        "top1_votes": [
            {"action": action, "votes": int(count), "fraction": count / observed_count}
            for action, count in ordered_votes
        ],
        "consensus_winner": winner,
        "consensus_winning_votes": winning_votes,
        "consensus_fraction": consensus_fraction,
        "strict_majority": bool(observed_count and winning_votes * 2 > observed_count),
        "fused_top1_action": fused_top1,
        "fused_top1_vote_count": fused_votes,
        "fused_top1_vote_fraction": fused_votes / observed_count if observed_count else None,
        "per_camera": per_camera,
        "top1_margin_summary": {
            "observed_count": len(margins),
            "mean": sum(margins) / len(margins) if margins else None,
            "median": median_margin,
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
    }


def _sidecar_camera_diagnostics(
    sidecar: Mapping[str, Any],
    candidates_by_window: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Build camera diagnostics keyed by sidecar window id.

    This deliberately mirrors :func:`_sidecar_candidates`'s permissive legacy
    handling: a missing model block is treated as a window-local unavailable
    diagnostic rather than invalidating an otherwise usable candidate list.
    """

    raw_windows = _sequence(sidecar.get("windows"), field="sidecar.windows")
    result: dict[str, dict[str, Any]] = {}
    for index, raw_window in enumerate(raw_windows):
        window = _mapping(raw_window, field=f"sidecar.windows[{index}]")
        window_id = _text(window.get("window_id"), field=f"sidecar.windows[{index}].window_id")
        model = window.get("model")
        model_map = (
            _mapping(model, field=f"{window_id}.model") if isinstance(model, Mapping) else window
        )
        result[window_id] = _camera_diagnostics(
            model_map,
            candidates_by_window.get(window_id, ()),
        )
    return result


def _validate_production_sidecar(sidecar: Mapping[str, Any], *, variant: str) -> None:
    """Reject accidental EPIC/candidate-space mixing before comparison."""

    expected_format = "robata-production-wemm-vocabulary-shadow-v1"
    if sidecar.get("format") != expected_format:
        raise ProductionWemmVariantComparisonError(
            f"{variant} sidecar format must be {expected_format!r}"
        )
    if sidecar.get("production_eligible") is not False:
        raise ProductionWemmVariantComparisonError(
            f"{variant} sidecar must remain production-ineligible"
        )
    vocabulary = sidecar.get("vocabulary")
    if not isinstance(vocabulary, Mapping):
        raise ProductionWemmVariantComparisonError(f"{variant} sidecar lacks vocabulary provenance")
    if vocabulary.get("epic_ontology_used") is not False:
        raise ProductionWemmVariantComparisonError(
            f"{variant} sidecar does not declare epic_ontology_used=false"
        )
    if vocabulary.get("mapper_used") is not False:
        raise ProductionWemmVariantComparisonError(
            f"{variant} sidecar does not declare mapper_used=false"
        )
    model = sidecar.get("model")
    if isinstance(model, Mapping):
        declared = model.get("label_variant")
        if declared is not None and str(declared) != variant:
            raise ProductionWemmVariantComparisonError(
                f"{variant} sidecar label_variant does not match its comparison key"
            )


def _metric_summary(ranks: Sequence[int | None], *, denominator: int, k: int) -> dict[str, Any]:
    found = [rank for rank in ranks if rank is not None]
    return {
        "windows": denominator,
        "hits": sum(rank is not None and rank <= k for rank in ranks),
        "rate": (
            sum(rank is not None and rank <= k for rank in ranks) / denominator
            if denominator
            else 0.0
        ),
        "k": k,
        "ranks": list(ranks),
        "mrr": sum(1.0 / rank for rank in found) / denominator if denominator else 0.0,
    }


def compare_production_wemm_vocabulary_variants(
    owner_review: Mapping[str, Any] | str | Path,
    sidecars: Mapping[str, Mapping[str, Any] | str | Path],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Compare canonical/verb-noun/natural sidecars against a surrogate review."""

    if not isinstance(ks, Sequence) or isinstance(ks, (str, bytes, bytearray)) or not ks:
        raise ProductionWemmVariantComparisonError("ks must be a non-empty sequence")
    parsed_ks: tuple[int, ...] = tuple(sorted({int(k) for k in ks}))
    if any(k <= 0 for k in parsed_ks):
        raise ProductionWemmVariantComparisonError("ks must contain positive integers")
    if not isinstance(sidecars, Mapping) or not sidecars:
        raise ProductionWemmVariantComparisonError("sidecars must be a non-empty mapping")
    review = _load_json(owner_review)
    parsed_sidecars = {str(name): _load_json(value) for name, value in sidecars.items()}
    source_binding = _source_binding(review, parsed_sidecars)
    if source_binding["status"] == "CONFLICT":
        raise ProductionWemmVariantComparisonError(
            "owner review and sidecars contain conflicting media references"
        )
    references = _reference_windows(review)
    eligible = {
        window_id: row
        for window_id, row in references.items()
        if row["decision"] in {"accept", "edit", "split"} and row["pairs"]
    }
    if not eligible:
        raise ProductionWemmVariantComparisonError("owner review has no eligible reference windows")
    routes: dict[str, Any] = {}
    for variant, sidecar in parsed_sidecars.items():
        _validate_production_sidecar(sidecar, variant=variant)
        candidates = _sidecar_candidates(sidecar)
        camera_diagnostics = _sidecar_camera_diagnostics(sidecar, candidates)
        ranks: list[int | None] = []
        per_window: dict[str, Any] = {}
        for window_id, reference in eligible.items():
            rows = candidates.get(window_id, [])
            rank = next(
                (
                    int(candidate["rank"])
                    for candidate in rows
                    if tuple(candidate["pair"]) in reference["pairs"]
                ),
                None,
            )
            ranks.append(rank)
            per_window[window_id] = {
                "reference_pairs": [list(pair) for pair in reference["pairs"]],
                "candidate_count": len(rows),
                "candidates": rows,
                "matching_rank": rank,
                # Additive, non-gold metadata.  Older sidecars produce an
                # explicit NOT_AVAILABLE block so downstream rank reports can
                # distinguish absent camera evidence from zero consensus.
                "camera_diagnostics": camera_diagnostics.get(
                    window_id,
                    {
                        "status": "NOT_AVAILABLE",
                        "source": "sidecar.per_camera_predictions",
                    },
                ),
            }
        metrics: dict[str, Any] = {}
        for k in parsed_ks:
            summary = _metric_summary(ranks, denominator=len(eligible), k=k)
            # Keep the requested public names while retaining a single rank
            # vector and MRR calculation for reproducibility.
            metrics[f"top{k}"] = {
                "windows": summary["windows"],
                "hits": summary["hits"],
                "rate": summary["rate"],
            }
            if k == parsed_ks[-1]:
                metrics["mrr"] = summary["mrr"]
        metrics["candidate_coverage"] = sum(
            bool(candidates.get(window_id)) for window_id in eligible
        ) / len(eligible)
        # Keep the cardinality of the recorded candidate lists explicit.  A
        # production-shaped vocabulary may be much smaller than the requested
        # retrieval cutoff (the current coarse vocabulary has six actions), in
        # which case R@10 is full-list coverage rather than recall over ten or
        # more independently ranked actions.  This is additive metadata so
        # existing consumers of ``metrics`` remain backward compatible.
        candidate_counts = [len(candidates.get(window_id, [])) for window_id in eligible]
        unique_candidate_counts = sorted(set(candidate_counts))
        metrics["candidate_list_cardinality"] = {
            "unit": "eligible_window",
            "min": min(candidate_counts) if candidate_counts else 0,
            "max": max(candidate_counts) if candidate_counts else 0,
            "unique": unique_candidate_counts,
            "all_windows_same": len(unique_candidate_counts) <= 1,
            "effective_max_k": max(candidate_counts) if candidate_counts else 0,
            "full_list_at_k": {
                str(k): bool(candidate_counts) and max(candidate_counts) <= k for k in parsed_ks
            },
        }
        model = sidecar.get("model")
        vocabulary = sidecar.get("vocabulary")
        source = sidecar.get("source")
        routes[variant] = {
            "label_variant": model.get("label_variant") if isinstance(model, Mapping) else variant,
            "sidecar_format": sidecar.get("format"),
            "status": sidecar.get("status"),
            "metrics": metrics,
            "ranks": ranks,
            "per_window": per_window,
            "provenance": {
                "source": dict(source) if isinstance(source, Mapping) else {},
                "vocabulary_profile": vocabulary.get("profile")
                if isinstance(vocabulary, Mapping)
                else None,
                "vocabulary_format": vocabulary.get("format")
                if isinstance(vocabulary, Mapping)
                else None,
                "epic_ontology_used": vocabulary.get("epic_ontology_used")
                if isinstance(vocabulary, Mapping)
                else None,
                "mapper_used": vocabulary.get("mapper_used")
                if isinstance(vocabulary, Mapping)
                else None,
            },
        }
    return {
        "format": PRODUCTION_WEMM_VARIANT_COMPARISON_VERSION,
        "authority": AUTHORITY,
        "status": STATUS,
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "production_eligible": False,
        "reference": {
            "status": _reference_status(review),
            "window_count": len(references),
            "eligible_window_count": len(eligible),
            "excluded_window_count": len(references) - len(eligible),
            "decision_counts": {
                decision: sum(row["decision"] == decision for row in references.values())
                for decision in ("accept", "edit", "split", "reject", "abstain")
            },
            "source": _source_refs(review),
        },
        "source_binding": source_binding,
        "ks": list(parsed_ks),
        "routes": routes,
        "metric_units": {
            "top_k": "window_level",
            "mrr": "window_level",
            "candidate_coverage": "window_level",
            "camera_consensus": "window_level",
        },
        "candidate_list_warning": (
            "Top-K and MRR are window-level metrics. When a recorded candidate "
            "list has at most K entries, R@K is full-list coverage, not recall "
            "over a larger catalog. Inspect routes[*].metrics.candidate_list_cardinality."
        ),
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_digest_computed": False,
            "source_sidecars_reused": True,
        },
        "limitations": [
            "Metrics are overlap against an owner-scoped non-gold review reference.",
            "Top-K retrieval does not score attributes, location, hand, or boundaries.",
            "Split windows are evaluated as a window-level hit when any reference pair is ranked.",
            "No EPIC ontology IDs or Mapper decisions are involved.",
        ],
    }


def _metric_rate(metrics: Mapping[str, Any], name: str) -> float:
    item = metrics.get(name, {})
    return float(item.get("rate", 0.0)) if isinstance(item, Mapping) else 0.0


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact variant comparison report."""

    lines = [
        "# Production WeMM vocabulary variant comparison",
        "",
        "> **SURROGATE_ONLY / NOT_MEASURED.** Terra review is not official gold.",
        "",
        f"- Reference: `{report.get('reference', {}).get('status', 'SURROGATE')}`",
        f"- Eligible windows: `{report.get('reference', {}).get('eligible_window_count', 0)}`",
        f"- Excluded windows: `{report.get('reference', {}).get('excluded_window_count', 0)}`",
        "",
        "| Variant | Window R@1 | Window R@3 | Window R@5 | Window R@10 | Window MRR | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, route in report.get("routes", {}).items():
        metrics = route.get("metrics", {}) if isinstance(route, Mapping) else {}

        lines.append(
            f"| {variant} | {_metric_rate(metrics, 'top1'):.1%} | "
            f"{_metric_rate(metrics, 'top3'):.1%} | "
            f"{_metric_rate(metrics, 'top5'):.1%} | "
            f"{_metric_rate(metrics, 'top10'):.1%} | "
            f"{float(metrics.get('mrr', 0.0)):.3f} | "
            f"{float(metrics.get('candidate_coverage', 0.0)):.1%} |"
        )
    lines.extend(["", "## Candidate-list cardinality", ""])
    lines.append(
        "| Variant | Min candidates/window | Max candidates/window | "
        "Unique counts | Full-list cutoffs |"
    )
    lines.append("|---|---:|---:|---|---|")
    for variant, route in report.get("routes", {}).items():
        metrics = route.get("metrics", {}) if isinstance(route, Mapping) else {}
        cardinality = metrics.get("candidate_list_cardinality", {})
        if not isinstance(cardinality, Mapping):
            cardinality = {}
        full_list = cardinality.get("full_list_at_k", {})
        if isinstance(full_list, Mapping):
            cutoffs = (
                ", ".join(f"K={k}" for k, is_full in full_list.items() if bool(is_full)) or "none"
            )
        else:
            cutoffs = "-"
        unique = cardinality.get("unique", [])
        unique_text = (
            ", ".join(str(value) for value in unique) if isinstance(unique, Sequence) else "-"
        )
        lines.append(
            f"| {variant} | {cardinality.get('min', '-')} | {cardinality.get('max', '-')} | "
            f"{unique_text} | {cutoffs} |"
        )
    lines.extend(
        [
            "",
            str(report.get("candidate_list_warning", "")),
            "",
            "Each route retains per-window candidate/rank details and sidecar provenance in JSON.",
            "No model, media, ontology, Mapper, or gold artifact was touched by this comparison.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AUTHORITY",
    "DEFAULT_KS",
    "DEFAULT_OWNER_REVIEW",
    "DEFAULT_SIDECARS",
    "OFFICIAL_QUALITY_STATUS",
    "PRODUCTION_WEMM_VARIANT_COMPARISON_VERSION",
    "ProductionWemmVariantComparisonError",
    "compare_production_wemm_vocabulary_variants",
    "render_markdown",
]
