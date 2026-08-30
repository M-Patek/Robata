"""Build a unified, review-only production candidate pack.

The production source cohort has no independently adjudicated action labels.
This module therefore joins *observations* from the already-recorded Qwen
identity sidecar, the source-bound Qwen boundary sidecar, and an optional WeMM
Top-K sidecar into one review surface.  It deliberately does not infer a
boundary, silently translate an EPIC label, invoke a model, decode media, or
write gold.  Every candidate remains pending an explicit reviewer decision.

The pack is intentionally small and descriptive.  It separates the useful
diagnostic dimensions (identity support, boundary availability, timestamp
provenance, optional-field coverage, and semantic review) so a missing time
claim cannot be confused with a semantic disagreement.
"""

from __future__ import annotations

import copy
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

FORMAT: Final = "robata-production-review-candidate-pack-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "REVIEW_ONLY"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
OFFICIAL_GOLD_STATUS: Final = "NOT_ESTABLISHED"
VOCABULARY_STATUS: Final = "PROVISIONAL_NON_GOLD"
DECISION_OPTIONS: Final = ("accept", "edit", "split", "reject", "abstain")
MODEL_NAMES: Final = ("wemm", "qwen", "mage")
PRODUCTION_ACTIONS: Final = (
    "pick up garment",
    "spread garment",
    "flatten garment",
    "adjust garment",
    "smooth garment",
    "fold garment",
)
_ACTION_VERB_ALIASES: Final = {
    "pick up": "pick up",
    "picks up": "pick up",
    "picking up": "pick up",
    "pickup": "pick up",
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
_NOUN_ALIASES: Final = {
    "garment": "garment",
}
_GOLD_KEYS: Final = frozenset(
    {
        "gold",
        "groundtruth",
        "officialreference",
        "officialgold",
        "humanannotation",
        "adjudicatedlabel",
    }
)


class ProductionReviewCandidatePackError(ValueError):
    """Raised when a model sidecar cannot be normalized safely."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionReviewCandidatePackError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionReviewCandidatePackError(f"{field} must be an array")
    return value


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _copy_json(value: object, *, field: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionReviewCandidatePackError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(key): _copy_json(child, field=f"{field}.{key}") for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionReviewCandidatePackError(f"{field} must be JSON-compatible")


def _load(value: Mapping[str, Any] | str | Path, *, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        decoded = json.loads(Path(value).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionReviewCandidatePackError(f"could not load {field}: {exc}") from exc
    return dict(_mapping(decoded, field=field))


def load_json(path: str | Path) -> dict[str, Any]:
    return _load(path, field=str(path))


def _assert_no_gold(value: object, *, field: str) -> None:
    """Reject obvious gold payloads in model-only input sidecars.

    This is a field-separation check, not an identity/hash or defensive
    mechanism.  Prose/evidence values are not inspected.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionReviewCandidatePackError(f"{field} keys must be strings")
            normalized = "".join(ch for ch in key.casefold() if ch.isalnum())
            if normalized in _GOLD_KEYS:
                raise ProductionReviewCandidatePackError(
                    f"{field}.{key} contains gold/official annotation data"
                )
            _assert_no_gold(child, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_gold(child, field=f"{field}[{index}]")
        return
    raise ProductionReviewCandidatePackError(f"{field} must be JSON-compatible")


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _action(value: object) -> str | None:
    """Normalize only the approved coarse production vocabulary."""

    text = _norm(value)
    if not text:
        return None
    parts = text.split()
    # The first two tokens may form the phrasal verb ``pick up``.
    verb_tokens = 2 if len(parts) >= 2 and " ".join(parts[:2]) in _ACTION_VERB_ALIASES else 1
    verb = _ACTION_VERB_ALIASES.get(" ".join(parts[:verb_tokens]))
    if verb is None:
        return None
    noun = _NOUN_ALIASES.get(" ".join(parts[verb_tokens:]))
    if noun is None:
        return None
    label = f"{verb} {noun}"
    return label if label in PRODUCTION_ACTIONS else None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _window_rows(payload: Mapping[str, Any], *, field: str) -> list[Mapping[str, Any]]:
    raw = payload.get("windows")
    if raw is None:
        raise ProductionReviewCandidatePackError(f"{field}.windows is required")
    rows: list[Mapping[str, Any]] = []
    for index, value in enumerate(_sequence(raw, field=f"{field}.windows")):
        row = _mapping(value, field=f"{field}.windows[{index}]")
        window_id = _text(row.get("window_id"))
        if not window_id:
            raise ProductionReviewCandidatePackError(
                f"{field}.windows[{index}].window_id is required"
            )
        rows.append(row)
    return rows


def _row_interval(row: Mapping[str, Any]) -> tuple[float, float] | None:
    for key in ("interval", "source_interval", "window_interval"):
        value = row.get(key)
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and len(value) == 2
        ):
            start, end = _finite(value[0]), _finite(value[1])
            if start is not None and end is not None and end > start:
                return start, end
    for first, second in (
        ("start_seconds", "end_seconds"),
        ("start_time_sec", "end_time_sec"),
        ("window_start_seconds", "window_end_seconds"),
    ):
        start, end = _finite(row.get(first)), _finite(row.get(second))
        if start is not None and end is not None and end > start:
            return start, end
    return None


def _camera_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("camera_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _identity_action(row: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    identity = row.get("parsed_identity")
    if not isinstance(identity, Mapping):
        identity = row.get("raw_identity")
    if not isinstance(identity, Mapping):
        identity = row
    reasons: list[str] = []
    if _text(identity.get("parse_status")).upper() not in {"PARSED", "SUCCESS", "SUCCEEDED"}:
        reasons.append("IDENTITY_PARSE_NOT_CLEAN")
    action = _action(identity.get("action"))
    if action is None:
        reasons.append("IDENTITY_ACTION_NOT_IN_PRODUCTION_VOCABULARY")
    return action, reasons


def _identity_evidence(row: Mapping[str, Any]) -> list[str]:
    identity = row.get("parsed_identity")
    if not isinstance(identity, Mapping):
        identity = row.get("raw_identity")
    if not isinstance(identity, Mapping):
        identity = row
    raw = identity.get("evidence", [])
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    return []


def _identity_confidence(row: Mapping[str, Any]) -> float | None:
    identity = row.get("parsed_identity")
    if not isinstance(identity, Mapping):
        identity = row.get("raw_identity")
    if not isinstance(identity, Mapping):
        identity = row
    value = _finite(identity.get("confidence"))
    return max(0.0, min(1.0, value)) if value is not None else None


def _boundary_claims(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = row.get("segments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
        raw = row.get("candidates")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
        parsed = row.get("parsed_boundary")
        if isinstance(parsed, Mapping):
            raw = [parsed]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _claim_labels(claim: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    labels = claim.get("structured_labels", claim.get("labels"))
    source: Mapping[str, Any] = labels if isinstance(labels, Mapping) else claim

    def value(key: str) -> Any:
        raw = source.get(key)
        return raw.get("value") if isinstance(raw, Mapping) else raw

    raw_action = " ".join(part for part in (_text(value("verb")), _text(value("noun"))) if part)
    fields = {key: value(key) for key in ("attributes", "location", "hand")}
    return _action(raw_action), fields


def _claim_interval(claim: Mapping[str, Any]) -> tuple[float, float] | None:
    for first, second in (
        ("start_time_sec", "end_time_sec"),
        ("start_seconds", "end_seconds"),
        ("mapped_start_time_sec", "mapped_end_time_sec"),
    ):
        start, end = _finite(claim.get(first)), _finite(claim.get(second))
        if start is not None and end is not None and end > start:
            return start, end
    return None


def _frame_ordinal_projection(
    row: Mapping[str, Any],
    claim: Mapping[str, Any],
    interval: tuple[float, float],
) -> tuple[dict[str, Any] | None, str | None]:
    """Project an explicitly mapped sampled-frame claim to source seconds.

    The frame-ordinal runner records ``start_time_sec``/``end_time_sec`` in
    the bounded window clock.  We only apply the window offset when *both*
    the coordinate mode and mapping status explicitly identify that runner
    contract.  Ordinary relative claims are intentionally left untouched and
    therefore fail the source-bound check below.
    """

    status = _norm(claim.get("timestamp_mapping_status") or row.get("timestamp_mapping_status"))
    coordinate_mode = _norm(claim.get("coordinate_mode") or row.get("coordinate_mode"))
    if status != "mapped from frame ordinal" or coordinate_mode != "sampled frame ordinal":
        return None, None
    start = _finite(claim.get("start_time_sec", claim.get("start_seconds")))
    end = _finite(claim.get("end_time_sec", claim.get("end_seconds")))
    low, high = interval
    duration = high - low
    if start is None or end is None or end <= start:
        return None, "BOUNDARY_MISSING_OR_INVALID"
    if start < 0.0 or end > duration:
        return None, "BOUNDARY_OUT_OF_SOURCE"
    projected = dict(claim)
    projected["start_time_sec"] = low + start
    projected["end_time_sec"] = low + end
    projected["timestamp_basis"] = "source_absolute_seconds"
    projected["mapped_timestamp_basis"] = "source_absolute_seconds"
    projected["timestamp_mapping_status"] = "MAPPED_FROM_FRAME_ORDINAL"
    projected["coordinate_mode"] = "sampled_frame_ordinal"
    projected["frame_ordinal_projection"] = {
        "status": "MEASURED",
        "source_interval": [low, high],
        "relative_interval": [start, end],
        "source_interval_seconds": [low + start, low + end],
        "mapping_status": "MAPPED_FROM_FRAME_ORDINAL",
        "coordinate_mode": "sampled_frame_ordinal",
        "start_frame_ordinal": claim.get("start_frame_ordinal"),
        "end_frame_ordinal": claim.get("end_frame_ordinal"),
        "frame_indices": _copy_json(row.get("frame_indices", []), field="boundary.frame_indices"),
        "frame_timestamps_seconds": _copy_json(
            row.get("frame_timestamps_seconds", []),
            field="boundary.frame_timestamps_seconds",
        ),
        "raw_claim": _copy_json(claim, field="boundary.raw_frame_ordinal_claim"),
    }
    return projected, "FRAME_ORDINAL_MAPPED_TO_SOURCE"


def _claim_source_bound(
    row: Mapping[str, Any], claim: Mapping[str, Any], interval: tuple[float, float]
) -> tuple[bool, str, Mapping[str, Any]]:
    """Return (source-bound, reason) using explicit provenance only."""

    projected, projection_reason = _frame_ordinal_projection(row, claim, interval)
    checked_claim: Mapping[str, Any] = projected if projected is not None else claim
    if projection_reason == "FRAME_ORDINAL_MAPPED_TO_SOURCE":
        claim_interval = _claim_interval(checked_claim)
    else:
        claim_interval = _claim_interval(checked_claim)
    if claim_interval is None:
        return False, projection_reason or "BOUNDARY_MISSING_OR_INVALID", checked_claim
    low, high = interval
    if claim_interval[0] < low or claim_interval[1] > high:
        return False, "BOUNDARY_OUT_OF_SOURCE", checked_claim
    row_status = _norm(row.get("timestamp_mapping_status") or row.get("mapping_status"))
    claim_status = _norm(
        checked_claim.get("timestamp_mapping_status")
        or checked_claim.get("mapping_status")
        or checked_claim.get("timestamp_basis_status")
    )
    basis = _norm(checked_claim.get("timestamp_basis") or row.get("timestamp_basis"))
    explicit = {"mapped", "recorded", "measured", "source bound", "source absolute"}
    if projection_reason == "FRAME_ORDINAL_MAPPED_TO_SOURCE":
        return True, projection_reason, checked_claim
    if (
        row_status not in explicit
        and claim_status not in explicit
        and basis != "source absolute seconds"
    ):
        return False, "BOUNDARY_NOT_SOURCE_MEASURED", checked_claim
    if basis and basis != "source absolute seconds":
        return False, "BOUNDARY_TIMESTAMP_BASIS_NOT_SOURCE_ABSOLUTE", checked_claim
    return True, "SOURCE_BOUND", checked_claim


def _wemm_predictions(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    model = row.get("model")
    if isinstance(model, Mapping):
        raw = model.get("predictions", model.get("candidates", []))
    else:
        raw = row.get("predictions", row.get("candidates", []))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _wemm_action(prediction: Mapping[str, Any]) -> str | None:
    explicit_pair = " ".join(
        part for part in (_text(prediction.get("verb")), _text(prediction.get("noun"))) if part
    )
    if explicit_pair:
        mapped = _action(explicit_pair)
        if mapped is not None:
            return mapped
    raw = prediction.get("label_text")
    return _action(raw) if isinstance(raw, str) and raw.strip() else None


def _wemm_context(row: Mapping[str, Any], *, top_k: int) -> dict[str, Any]:
    predictions = _wemm_predictions(row)
    retained: list[dict[str, Any]] = []
    for index, raw in enumerate(predictions[:top_k]):
        rank = raw.get("rank", index + 1)
        try:
            rank_value = int(rank)
        except (TypeError, ValueError, OverflowError):
            rank_value = index + 1
        score = _finite(raw.get("score", raw.get("fused_score", raw.get("visual_score"))))
        mapped = _wemm_action(raw)
        retained.append(
            {
                "rank": rank_value,
                "score": score,
                "raw_label": raw.get("label_text")
                or " ".join(
                    part for part in (_text(raw.get("verb")), _text(raw.get("noun"))) if part
                ),
                "verb": raw.get("verb"),
                "noun": raw.get("noun"),
                "mapped_action": mapped,
                "mapping_status": "MAPPED" if mapped else "UNMAPPED_EPIC_OR_FOREIGN_LABEL",
                "camera_coverage": raw.get("camera_coverage"),
                "camera_coverage_fraction": raw.get("camera_coverage_fraction"),
                "source": raw.get("source"),
            }
        )
    return {
        "status": _text(row.get("status"))
        or _text(_mapping(row.get("model", {}), field="wemm.model").get("status"))
        or "NOT_RUN",
        "top_k": retained,
        "raw_prediction_count": len(predictions),
        "source_window": _copy_json(row, field="wemm.window"),
    }


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_text(row.get("window_id"))].append(row)
    return grouped


def _source_path(payload: Mapping[str, Any]) -> str | None:
    source = payload.get("source")
    if not isinstance(source, Mapping):
        return None
    for key in ("path", "media_path", "source_path", "manifest", "video_root"):
        value = _text(source.get(key))
        if value:
            return value
    return None


def _boundary_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str | None], Mapping[str, Any]]:
    result: dict[tuple[str, str | None], Mapping[str, Any]] = {}
    for row in rows:
        result[(_text(row.get("window_id")), _camera_id(row))] = row
    return result


def _identity_observation(
    identity_row: Mapping[str, Any],
    boundary_row: Mapping[str, Any] | None,
    *,
    interval: tuple[float, float],
    row_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    action, identity_reasons = _identity_action(identity_row)
    camera = _camera_id(identity_row)
    confidence = _identity_confidence(identity_row)
    identity_evidence = _identity_evidence(identity_row)
    boundary_options: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if boundary_row is not None:
        for claim_index, claim in enumerate(_boundary_claims(boundary_row)):
            claim_action, fields = _claim_labels(claim)
            # Frame-ordinal boundary probes carry the action in the explicit
            # identity context rather than duplicating labels in the boundary
            # object.  Use that context only for this recorded boundary arm;
            # ordinary relative claims still require their own labels.
            if (
                claim_action is None
                and _norm(
                    claim.get("timestamp_mapping_status")
                    or boundary_row.get("timestamp_mapping_status")
                )
                == "mapped from frame ordinal"
            ):
                context = boundary_row.get("identity_context")
                if isinstance(context, Mapping):
                    claim_action = _action(context.get("action"))
            source_bound, reason, checked_claim = _claim_source_bound(boundary_row, claim, interval)
            claim_interval = _claim_interval(checked_claim)
            if claim_action is None or action is None or claim_action != action:
                if claim_action is not None:
                    rejected.append(
                        {
                            "claim_index": claim_index,
                            "camera_id": camera,
                            "reason_codes": ["BOUNDARY_ACTION_MISMATCH"],
                            "action": claim_action,
                            "raw_claim": _copy_json(claim, field="boundary.claim"),
                        }
                    )
                continue
            if not source_bound or claim_interval is None:
                rejected.append(
                    {
                        "claim_index": claim_index,
                        "camera_id": camera,
                        "reason_codes": [reason],
                        "action": claim_action,
                        "raw_claim": _copy_json(claim, field="boundary.claim"),
                    }
                )
                continue
            claim_conf = _finite(claim.get("confidence"))
            boundary_options.append(
                {
                    "start_seconds": claim_interval[0],
                    "end_seconds": claim_interval[1],
                    "confidence": max(0.0, min(1.0, claim_conf))
                    if claim_conf is not None
                    else None,
                    "timestamp_basis": "source_absolute_seconds",
                    "projection": _copy_json(
                        checked_claim.get("frame_ordinal_projection"),
                        field="boundary.frame_ordinal_projection",
                    )
                    if isinstance(checked_claim.get("frame_ordinal_projection"), Mapping)
                    else None,
                    "fields": fields,
                    "evidence": _copy_json(claim.get("evidence", []), field="boundary.evidence"),
                    "raw_claim": _copy_json(claim, field="boundary.claim"),
                }
            )
    chosen = max(
        boundary_options,
        key=lambda item: (item.get("confidence") is not None, item.get("confidence") or 0.0),
        default=None,
    )
    observation: dict[str, Any] = {
        "observation_id": (
            f"{_text(identity_row.get('window_id'))}:{camera or 'unknown'}:{row_index}"
        ),
        "camera_id": camera,
        "action": action,
        "identity_status": "MEASURED" if action else "NOT_MEASURED",
        "identity_confidence": confidence,
        "identity_evidence": identity_evidence,
        "boundary_status": "MEASURED" if chosen is not None else "NOT_MEASURED",
        "boundary": chosen,
        "timestamp_status": "MEASURED" if chosen is not None else "NOT_MEASURED",
        "optional_fields": chosen.get("fields", {}) if chosen is not None else {},
        "reason_codes": list(
            dict.fromkeys(identity_reasons + ([] if chosen else ["BOUNDARY_UNRESOLVED"]))
        ),
        "raw_identity": _copy_json(identity_row, field="identity.row"),
        "raw_boundary": _copy_json(boundary_row, field="boundary.row")
        if boundary_row is not None
        else None,
        "rejected_boundary_claims": rejected,
    }
    return observation, rejected


def _window_pack(
    *,
    window_id: str,
    ordinal: int,
    interval: tuple[float, float],
    identity_rows: Sequence[Mapping[str, Any]],
    boundary_by_key: Mapping[tuple[str, str | None], Mapping[str, Any]],
    wemm_row: Mapping[str, Any] | None,
    top_k: int,
    expected_camera_count: int | None,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    rejected_claims: list[dict[str, Any]] = []
    for index, identity_row in enumerate(identity_rows):
        camera = _camera_id(identity_row)
        observation, rejected = _identity_observation(
            identity_row,
            boundary_by_key.get((window_id, camera)),
            interval=interval,
            row_index=index,
        )
        observations.append(observation)
        rejected_claims.extend(rejected)

    clean = [row for row in observations if row.get("action") in PRODUCTION_ACTIONS]
    action_counts = Counter(str(row["action"]) for row in clean)
    ranked_actions = sorted(
        action_counts,
        key=lambda action: (
            -action_counts[action],
            -statistics.fmean(
                [
                    float(row["identity_confidence"])
                    for row in clean
                    if row.get("action") == action
                    and _finite(row.get("identity_confidence")) is not None
                ]
            )
            if any(
                _finite(row.get("identity_confidence")) is not None
                for row in clean
                if row.get("action") == action
            )
            else 0.0,
            action,
        ),
    )
    wemm = (
        _wemm_context(wemm_row, top_k=top_k)
        if wemm_row is not None
        else {
            "status": "NOT_RUN",
            "top_k": [],
            "raw_prediction_count": 0,
            "source_window": None,
        }
    )

    candidate_actions = set(ranked_actions)
    for prediction in wemm["top_k"]:
        mapped = prediction.get("mapped_action")
        if isinstance(mapped, str) and mapped in PRODUCTION_ACTIONS:
            candidate_actions.add(mapped)
    candidates: list[dict[str, Any]] = []
    for action in candidate_actions:
        supports = [row for row in clean if row.get("action") == action]
        measured = [row for row in supports if row.get("boundary_status") == "MEASURED"]
        starts = [
            float(row["boundary"]["start_seconds"])
            for row in measured
            if isinstance(row.get("boundary"), Mapping)
        ]
        ends = [
            float(row["boundary"]["end_seconds"])
            for row in measured
            if isinstance(row.get("boundary"), Mapping)
        ]
        wemm_rows = [row for row in wemm["top_k"] if row.get("mapped_action") == action]
        best_wemm = min(wemm_rows, key=lambda row: int(row.get("rank", 10**6)), default=None)
        identity_support = len(supports)
        mean_confidence = (
            statistics.fmean(
                [
                    float(row["identity_confidence"])
                    for row in supports
                    if _finite(row.get("identity_confidence")) is not None
                ]
            )
            if any(_finite(row.get("identity_confidence")) is not None for row in supports)
            else None
        )
        optional: dict[str, Any] = {}
        optional_status: dict[str, str] = {}
        for field in ("attributes", "location", "hand"):
            values = [
                row.get("optional_fields", {}).get(field)
                for row in measured
                if isinstance(row.get("optional_fields"), Mapping)
            ]
            values = [value for value in values if value is not None and str(value).strip()]
            optional[field] = (
                Counter(str(value) for value in values).most_common(1)[0][0] if values else None
            )
            optional_status[field] = "MEASURED" if values else "NOT_MEASURED"
        reasons: list[str] = []
        if identity_support < 2:
            reasons.append("IDENTITY_SUPPORT_LT_2_CAMERAS")
        if len(measured) < 2:
            reasons.append("BOUNDARY_SUPPORT_LT_2_CAMERAS")
        if best_wemm is None:
            reasons.append("NO_WEMM_VOCABULARY_MATCH")
        elif len(wemm_rows) > 1:
            reasons.append("WEMM_MULTIPLE_RANKS_FOR_ACTION")
        verb, noun = action.rsplit(" ", 1)
        candidate_start = statistics.median(starts) if starts else None
        candidate_end = statistics.median(ends) if ends else None
        candidates.append(
            {
                "claim_id": f"{window_id}:production:{re.sub(r'[^a-z0-9]+', '-', action)}",
                "status": "PENDING_HUMAN_REVIEW",
                "automatic_eligible": False,
                "accepted": False,
                "review_required": True,
                "semantic_status": "NOT_CHECKED",
                "verb": verb,
                "noun": noun,
                "label_text": action,
                "start_seconds": candidate_start,
                "end_seconds": candidate_end,
                # Keep both names used by the production references and the
                # source-bound model sidecars.  They are aliases, not a
                # second inference path; unresolved values remain null.
                "start_time_sec": candidate_start,
                "end_time_sec": candidate_end,
                "boundary_status": "MEASURED" if starts and ends else "NOT_MEASURED",
                "timestamp_basis": "source_absolute_seconds" if starts and ends else None,
                "attributes": optional["attributes"],
                "location": optional["location"],
                "hand": optional["hand"],
                "field_status": {
                    "verb": "MEASURED",
                    "noun": "MEASURED",
                    **optional_status,
                },
                "confidence": mean_confidence,
                "evidence": list(
                    dict.fromkeys(
                        text
                        for row in supports
                        for text in row.get("identity_evidence", [])
                        if isinstance(text, str) and text.strip()
                    )
                )[:3],
                "sources": {
                    "qwen_identity": {
                        "camera_support": identity_support,
                        "camera_ids": sorted(
                            {str(row.get("camera_id")) for row in supports if row.get("camera_id")}
                        ),
                        "mean_confidence": mean_confidence,
                    },
                    "qwen_boundary": {
                        "measured_camera_support": len(measured),
                        "camera_ids": sorted(
                            {str(row.get("camera_id")) for row in measured if row.get("camera_id")}
                        ),
                    },
                    "wemm": {
                        "present": best_wemm is not None,
                        "rank": best_wemm.get("rank") if best_wemm else None,
                        "score": best_wemm.get("score") if best_wemm else None,
                        "mapping_status": best_wemm.get("mapping_status")
                        if best_wemm
                        else "UNMAPPED",
                    },
                },
                "reason_codes": reasons or ["CROSS_ROUTE_CANDIDATE"],
                "raw_identity_observations": [
                    _copy_json(row, field="candidate.identity") for row in supports
                ],
            }
        )
    candidates.sort(
        key=lambda row: (
            -int(row["sources"]["qwen_identity"]["camera_support"]),
            -int(row["sources"]["qwen_boundary"]["measured_camera_support"]),
            int(row["sources"]["wemm"]["rank"] or 10**6),
            row["label_text"],
        )
    )
    measured_count = sum(row.get("boundary_status") == "MEASURED" for row in observations)
    source_bound_claims = sum(
        len(
            [
                item
                for item in row.get("rejected_boundary_claims", [])
                if "BOUNDARY_" in " ".join(item.get("reason_codes", []))
            ]
        )
        == 0
        and row.get("boundary_status") == "MEASURED"
        for row in observations
    )
    expected = expected_camera_count or len(
        {row.get("camera_id") for row in observations if row.get("camera_id")}
    )
    dimensions = {
        "identity": {
            "status": "MEASURED" if clean else "NOT_MEASURED",
            "observed_camera_count": len(
                {row.get("camera_id") for row in clean if row.get("camera_id")}
            ),
            "expected_camera_count": expected,
            "action_support": dict(action_counts),
            "top_action": ranked_actions[0] if ranked_actions else None,
            "top_support": action_counts[ranked_actions[0]] if ranked_actions else 0,
            "agreement_fraction": action_counts[ranked_actions[0]] / len(clean)
            if ranked_actions and clean
            else None,
        },
        "boundary": {
            "status": "MEASURED" if measured_count else "NOT_MEASURED",
            "identity_observation_count": len(clean),
            "measured_observation_count": measured_count,
            "measurement_rate": measured_count / len(clean) if clean else None,
            "source_bound_claim_count": source_bound_claims,
        },
        "timestamp": {
            "status": "MEASURED" if measured_count else "NOT_MEASURED",
            "source_absolute_count": measured_count,
            "unresolved_count": len(clean) - measured_count,
            "basis": "source_absolute_seconds" if measured_count else None,
        },
        "optional_fields": {
            field: {
                "status": "MEASURED"
                if any(
                    row.get("optional_fields", {}).get(field) is not None
                    for row in observations
                    if isinstance(row.get("optional_fields"), Mapping)
                )
                else "NOT_MEASURED",
                "measured_count": sum(
                    row.get("optional_fields", {}).get(field) is not None
                    for row in observations
                    if isinstance(row.get("optional_fields"), Mapping)
                ),
            }
            for field in ("attributes", "location", "hand")
        },
        "semantic": {"status": "NOT_CHECKED", "human_review_required": True},
    }
    return {
        "window_id": window_id,
        "ordinal": ordinal,
        "source_interval": [interval[0], interval[1]],
        "status": "REVIEW_REQUIRED" if candidates else "ABSTAIN",
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": OFFICIAL_GOLD_STATUS,
        "quality_claim": False,
        "production_eligible": False,
        "automatic_eligible": False,
        "human_adjudication": "NOT_PERFORMED",
        "decision": "pending",
        "decision_options": list(DECISION_OPTIONS),
        "annotation_candidates": candidates,
        "camera_observations": observations,
        "rejected_boundary_claims": rejected_claims,
        "model_context": {
            "wemm": wemm,
            "qwen_identity": {
                "status": "SUCCEEDED" if identity_rows else "NOT_RUN",
                "row_count": len(identity_rows),
                "raw_rows": [_copy_json(row, field="qwen.identity") for row in identity_rows],
            },
            "qwen_boundary": {
                "status": "RECORDED"
                if any(boundary_by_key.get((window_id, _camera_id(row))) for row in identity_rows)
                else "NOT_RUN",
                "raw_rows": [
                    _copy_json(boundary_by_key[(window_id, _camera_id(row))], field="qwen.boundary")
                    for row in identity_rows
                    if (window_id, _camera_id(row)) in boundary_by_key
                ],
            },
            "mage": {
                "status": "BLOCKED",
                "reason": "source-bound native semantic replay unavailable",
            },
        },
        "dimensions": dimensions,
        "abstention": {
            "abstained": not bool(candidates),
            "reason_codes": list(
                dict.fromkeys(
                    (["NO_PRODUCTION_VOCABULARY_CANDIDATE"] if not candidates else [])
                    + (["BOUNDARY_UNRESOLVED"] if clean and not measured_count else [])
                )
            ),
        },
    }


def build_production_review_candidate_pack(
    identity_sidecar: Mapping[str, Any] | str | Path,
    boundary_sidecar: Mapping[str, Any] | str | Path,
    wemm_sidecar: Mapping[str, Any] | str | Path | None = None,
    *,
    mage_sidecar: Mapping[str, Any] | str | Path | None = None,
    top_k: int = 10,
    expected_camera_count: int | None = 6,
    identity_path: str | None = None,
    boundary_path: str | None = None,
    wemm_path: str | None = None,
) -> dict[str, Any]:
    """Join recorded model observations into one pending review pack."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ProductionReviewCandidatePackError("top_k must be a positive integer")
    identity = _load(identity_sidecar, field="identity_sidecar")
    boundary = _load(boundary_sidecar, field="boundary_sidecar")
    wemm = _load(wemm_sidecar, field="wemm_sidecar") if wemm_sidecar is not None else None
    mage = _load(mage_sidecar, field="mage_sidecar") if mage_sidecar is not None else None
    _assert_no_gold(identity, field="identity_sidecar")
    _assert_no_gold(boundary, field="boundary_sidecar")
    if wemm is not None:
        _assert_no_gold(wemm, field="wemm_sidecar")
    if mage is not None:
        _assert_no_gold(mage, field="mage_sidecar")

    identity_rows = _window_rows(identity, field="identity_sidecar")
    boundary_rows = _window_rows(boundary, field="boundary_sidecar")
    wemm_rows = _window_rows(wemm, field="wemm_sidecar") if wemm is not None else []
    identity_grouped = _group_rows(identity_rows)
    boundary_index = _boundary_index(boundary_rows)
    wemm_by_window: dict[str, Mapping[str, Any]] = {}
    for row in wemm_rows:
        wid = _text(row.get("window_id"))
        # Keep the first explicit fused row; camera-level rows remain in the
        # raw sidecar and are not silently substituted for a fused ranking.
        wemm_by_window.setdefault(wid, row)

    # Use the identity cohort as the canonical window order, falling back to
    # boundary/WeMM windows if a future run omits the identity arm.
    order: list[str] = []
    window_specs: dict[str, tuple[int, tuple[float, float]]] = {}
    for source_rows in (identity_rows, boundary_rows, wemm_rows):
        for row in source_rows:
            wid = _text(row.get("window_id"))
            if wid in window_specs:
                continue
            interval = _row_interval(row)
            if interval is None:
                continue
            order.append(wid)
            ordinal = row.get("ordinal")
            ordinal_value = (
                ordinal
                if isinstance(ordinal, int) and not isinstance(ordinal, bool)
                else len(order) - 1
            )
            window_specs[wid] = (ordinal_value, interval)
    windows = [
        _window_pack(
            window_id=wid,
            ordinal=window_specs[wid][0],
            interval=window_specs[wid][1],
            identity_rows=identity_grouped.get(wid, []),
            boundary_by_key=boundary_index,
            wemm_row=wemm_by_window.get(wid),
            top_k=top_k,
            expected_camera_count=expected_camera_count,
        )
        for wid in order
    ]
    action_counts = Counter(
        candidate["label_text"]
        for window in windows
        for candidate in window["annotation_candidates"]
    )
    metrics = {
        "window_count": len(windows),
        "windows_with_candidates": sum(bool(window["annotation_candidates"]) for window in windows),
        "windows_abstained": sum(not bool(window["annotation_candidates"]) for window in windows),
        "candidate_count": sum(len(window["annotation_candidates"]) for window in windows),
        "identity_row_count": len(identity_rows),
        "boundary_row_count": len(boundary_rows),
        "wemm_window_count": len(wemm_rows),
        "source_bound_camera_observation_count": sum(
            row["dimensions"]["boundary"]["source_bound_claim_count"] for row in windows
        ),
        "action_candidate_counts": dict(action_counts),
        "top_k_window_count": sum(
            bool(window["model_context"]["wemm"]["top_k"]) for window in windows
        ),
    }
    output = {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": STATUS,
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": OFFICIAL_GOLD_STATUS,
        "quality_claim": False,
        "production_eligible": False,
        "automatic_eligible": False,
        "quality": {
            "measurement_status": "NOT_MEASURED",
            "quality_claim": False,
            "reason": "no independently adjudicated source-bound production action gold exists",
        },
        "source": {
            "identity_sidecar": identity_path,
            "boundary_sidecar": boundary_path,
            "wemm_sidecar": wemm_path,
            "mage_sidecar": str(mage_sidecar)
            if mage_sidecar is not None and not isinstance(mage_sidecar, Mapping)
            else None,
            "identity_source": _source_path(identity),
            "boundary_source": _source_path(boundary),
            "wemm_source": _source_path(wemm) if wemm is not None else None,
            "window_count": len(windows),
            "camera_count": expected_camera_count,
        },
        "vocabulary": {
            "status": VOCABULARY_STATUS,
            "labels": list(PRODUCTION_ACTIONS),
            "source": "Terra coarse production review vocabulary; provisional/non-gold",
            "epic_catalog_used_for_projection": False,
            "semantic_aliases_approved": False,
        },
        "metrics": metrics,
        "windows": windows,
        "contract": {
            "model_names": list(MODEL_NAMES),
            "production_vocabulary_is_provisional": True,
            "qwen_identity_is_candidate_only": True,
            "qwen_boundary_reused_only_when_source_bound": True,
            "wemm_top_k_preserved": True,
            "unmapped_wemm_labels_preserved": True,
            "fixed_window_is_not_action_boundary": True,
            "optional_fields_not_inferred": True,
            "semantic_status_requires_review": True,
            "explicit_decision_required": True,
            "official_gold_status": OFFICIAL_GOLD_STATUS,
            "automatic_eligible_always_false": True,
        },
        "controls": {
            "model_invoked": False,
            "source_media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "sha_or_digest_computed": False,
            "raw_observations_preserved": True,
        },
        "raw_sidecars": {
            "identity": _copy_json(identity, field="raw.identity"),
            "boundary": _copy_json(boundary, field="raw.boundary"),
            "wemm": _copy_json(wemm, field="raw.wemm") if wemm is not None else None,
            "mage": _copy_json(mage, field="raw.mage") if mage is not None else None,
        },
        "limitations": [
            "This pack is a review queue, not official production annotation or gold.",
            "Qwen identity and WeMM scores are observations; no calibrated probability is claimed.",
            "Only explicitly source-bound boundary claims receive non-null times.",
            (
                "EPIC/foreign WeMM labels remain visible but are not semantically "
                "aliased into production vocabulary."
            ),
            "Mage is BLOCKED unless a source-bound native semantic sidecar is supplied.",
        ],
    }
    return copy.deepcopy(output)


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics", {}), field="report.metrics")
    lines = [
        "# Production review-only candidate pack",
        "",
        "> **NOT_MEASURED / NON-GOLD.** Every candidate remains pending explicit review.",
        "",
        f"- Windows: `{metrics.get('window_count', 0)}`",
        f"- Candidate windows: `{metrics.get('windows_with_candidates', 0)}`",
        f"- Candidate rows: `{metrics.get('candidate_count', 0)}`",
        (
            "- Source-bound camera observations: "
            f"`{metrics.get('source_bound_camera_observation_count', 0)}`"
        ),
        f"- Windows with WeMM Top-K: `{metrics.get('top_k_window_count', 0)}`",
        "",
        "| Window | Status | Top candidate | Identity agreement | Boundary | WeMM Top-K |",
        "|---|---|---|---:|---|---:|",
    ]
    windows = report.get("windows", [])
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes, bytearray)):
        for raw in windows:
            if not isinstance(raw, Mapping):
                continue
            candidates = raw.get("annotation_candidates", [])
            top = (
                candidates[0]
                if isinstance(candidates, Sequence)
                and candidates
                and isinstance(candidates[0], Mapping)
                else {}
            )
            dimensions = raw.get("dimensions", {})
            identity = dimensions.get("identity", {}) if isinstance(dimensions, Mapping) else {}
            boundary = dimensions.get("boundary", {}) if isinstance(dimensions, Mapping) else {}
            context = raw.get("model_context", {})
            wemm = context.get("wemm", {}) if isinstance(context, Mapping) else {}
            lines.append(
                f"| {raw.get('window_id', '')} | {raw.get('status', '')} | "
                f"{top.get('label_text', '—')} | {identity.get('agreement_fraction', '—')} | "
                f"{boundary.get('status', '—')} | "
                f"{len(wemm.get('top_k', [])) if isinstance(wemm, Mapping) else 0} |"
            )
    lines.extend(
        [
            "",
            "## Dimensions",
            "",
            (
                "Identity support, boundary availability, timestamp provenance, "
                "optional-field coverage, and semantic review status are reported independently."
            ),
            (
                "Unmapped EPIC/foreign WeMM labels remain in the JSON Top-K context "
                "and are not silently translated."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AUTHORITY",
    "DECISION_OPTIONS",
    "FORMAT",
    "MODEL_NAMES",
    "OFFICIAL_GOLD_STATUS",
    "OFFICIAL_QUALITY_STATUS",
    "PRODUCTION_ACTIONS",
    "ProductionReviewCandidatePackError",
    "build_production_review_candidate_pack",
    "load_json",
    "render_markdown",
]
