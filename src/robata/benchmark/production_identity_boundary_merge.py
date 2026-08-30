"""Merge clean production identity observations with recorded boundary claims.

This is a small, post-hoc benchmark adapter.  It does not invoke a model,
decode media, infer a label from the fixed window, or modify the production
schema/ontology.  The identity arm supplies the owner-scoped action wording;
the separate structured arm supplies source-bound times only when its mapping
is explicit and valid.  Missing or conflicting times remain null and are sent
to review.

The module exists to move the local experiment forward without paying for a
second full Qwen run.  Every output is review-only and keeps both raw
sidecars available for inspection.
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

FORMAT: Final = "robata-production-identity-boundary-merge-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
OFFICIAL_GOLD_STATUS: Final = "NOT_ESTABLISHED"
MIN_BOUNDARY_CONSENSUS_CAMERAS: Final = 2
IDENTITY_ACTIONS: Final = (
    "pick up garment",
    "spread garment",
    "flatten garment",
    "adjust garment",
    "smooth garment",
    "fold garment",
)
_ACTION_VERBS: Final = {
    "pick up": "pick up",
    "pickups": "pick up",
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
_FILLER_VERBS: Final = frozenset(
    {"reach", "reaches", "move", "moves", "moving", "arrange", "arranges", "arranging"}
)
_NOUN_ALIASES: Final = frozenset(
    {"garment", "garments", "clothing", "cloth", "clothes", "fabric", "shirt", "shorts", "pants"}
)


class ProductionIdentityBoundaryMergeError(ValueError):
    """Raised when one of the local sidecars violates the merge contract."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionIdentityBoundaryMergeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionIdentityBoundaryMergeError(f"{field} must be an array")
    return value


def _copy(value: object, *, field: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionIdentityBoundaryMergeError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(k): _copy(v, field=f"{field}.{k}") for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy(v, field=f"{field}[{i}]") for i, v in enumerate(value)]
    raise ProductionIdentityBoundaryMergeError(f"{field} must be JSON-compatible")


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^a-z0-9 ]+", " ", text.replace("_", " ").replace("-", " "))
    return " ".join(text.split())


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _load(value: Mapping[str, Any] | str | Path, *, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    try:
        decoded = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionIdentityBoundaryMergeError(f"could not load {field}: {exc}") from exc
    return dict(_mapping(decoded, field=field))


def load_json(path: str | Path) -> dict[str, Any]:
    return _load(path, field=str(path))


def _action_from_identity(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    identity = row.get("parsed_identity")
    if not isinstance(identity, Mapping):
        identity = row.get("raw_identity")
    if not isinstance(identity, Mapping):
        identity = row
    if str(identity.get("parse_status", "")).upper() != "PARSED":
        return None, "IDENTITY_PARSE_NOT_CLEAN"
    action = _norm(identity.get("action"))
    if action not in IDENTITY_ACTIONS:
        return None, "IDENTITY_ACTION_UNSUPPORTED"
    return action, None


def _label_field(labels: object, key: str) -> tuple[Any, str]:
    if not isinstance(labels, Mapping) or key not in labels:
        return None, "NOT_MEASURED"
    raw = labels.get(key)
    if isinstance(raw, Mapping) and ("value" in raw or "status" in raw):
        status = str(raw.get("status") or "").upper() or (
            "NOT_OBSERVABLE" if raw.get("value") is None else "MEASURED"
        )
        return raw.get("value"), status
    if raw is None:
        return None, "NOT_OBSERVABLE"
    return raw, "MEASURED"


def _segment_candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return structured boundary claims from either sidecar shape.

    The first boundary runner emitted ``segments`` directly.  The later
    post-hoc boundary-review artifact keeps the same claims under
    ``candidates`` (and records source-time provenance on each candidate).
    Accepting both shapes here is structural normalization only; it does not
    infer a boundary or promote a candidate to gold.
    """

    raw = row.get("segments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raw = row.get("candidates")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        parsed = row.get("parsed_structured")
        raw = parsed.get("segments", []) if isinstance(parsed, Mapping) else []
    return [item for item in raw if isinstance(item, Mapping)]


def _boundary_mapping_is_source_bound(row: Mapping[str, Any], segment: Mapping[str, Any]) -> bool:
    """Check explicit source-time provenance before reusing a boundary.

    ``RECORDED`` is used by the tolerant review sidecar when the runner has
    already mapped a window-relative pair to source-absolute seconds.  It is
    therefore equivalent to ``MAPPED`` for this post-hoc join, but only when
    the row and claim both retain the source clock and a positive measured
    interval.  Unmapped relative claims remain unresolved.
    """

    row_status = _norm(row.get("timestamp_mapping_status"))
    row_mapping = _norm(row.get("mapping_status"))
    row_basis = _norm(row.get("canonical_timestamp_basis"))
    seg_status = _norm(segment.get("timestamp_mapping_status"))
    seg_basis = _norm(segment.get("timestamp_basis"))
    boundary_status = _norm(segment.get("boundary_status"))
    source_bound = segment.get("source_bound_positive_interval")
    mapped_status = {"mapped", "recorded", "measured", "source bound", "source absolute"}
    if row_status not in mapped_status and row_mapping not in mapped_status:
        return False
    if row_basis and row_basis != "source absolute seconds":
        return False
    if seg_status and seg_status not in mapped_status:
        return False
    if seg_basis and seg_basis != "source absolute seconds":
        return False
    if boundary_status and boundary_status not in {
        "measured",
        "explicit",
        "source bound",
        "source absolute",
    }:
        return False
    return source_bound is None or source_bound is True


def _boundary_for_action(
    row: Mapping[str, Any], action: str, window_interval: tuple[float, float]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Choose a measured structured claim matching the identity verb."""

    target = action.rsplit(" ", 1)[0]
    # Boundary-only rows deliberately contain no action label.  They are
    # eligible only when the runner attached the same identity hypothesis to
    # this exact window/camera; this is a provenance check, not a semantic
    # inference.  Their parsed times are window-relative and are converted to
    # source time only after the explicit frame-ordinal mapping has succeeded.
    direct = row.get("parsed_boundary")
    if isinstance(direct, Mapping):
        context = row.get("identity_context")
        context_action = _norm(context.get("action")) if isinstance(context, Mapping) else ""
        if context_action != action:
            return None, ["BOUNDARY_IDENTITY_CONTEXT_MISMATCH"]
        if (
            str(direct.get("parse_status") or "").upper() == "PARSED"
            and str(direct.get("boundary_status") or "").upper() == "MEASURED"
            and str(direct.get("timestamp_mapping_status") or "").upper()
            == "MAPPED_FROM_FRAME_ORDINAL"
        ):
            start_rel = _finite(direct.get("start_time_sec"))
            end_rel = _finite(direct.get("end_time_sec"))
            if start_rel is not None and end_rel is not None and end_rel > start_rel:
                source_start = window_interval[0]
                duration = window_interval[1] - source_start
                # Frame-ordinal claims are relative to the bounded source
                # window.  Do not offset an out-of-window pair into a
                # seemingly valid source-absolute interval.
                if start_rel < 0.0 or end_rel > duration:
                    return None, ["BOUNDARY_OUT_OF_SOURCE"]
                return {
                    "segment": {
                        "start_time_sec": source_start + start_rel,
                        "end_time_sec": source_start + end_rel,
                        "structured_labels": {},
                        "confidence": _finite(direct.get("confidence")),
                        "evidence": direct.get("evidence", ""),
                        "coordinate_mode": direct.get("coordinate_mode"),
                        "raw_boundary": direct,
                    },
                    "start": source_start + start_rel,
                    "end": source_start + end_rel,
                }, []
        return None, ["BOUNDARY_ONLY_UNRESOLVED"]
    source_status = str(row.get("timestamp_mapping_status") or "").upper()
    source_bound = source_status in {
        "MAPPED",
        "RECORDED",
        "MEASURED",
        "SOURCE_BOUND",
        "SOURCE_ABSOLUTE",
    } or str(row.get("mapping_status") or "").upper() in {
        "MAPPED",
        "RECORDED",
        "MEASURED",
        "SOURCE_BOUND",
        "SOURCE_ABSOLUTE",
    }
    reasons: list[str] = []
    if not source_bound:
        reasons.append("BOUNDARY_MAPPING_NOT_MEASURED")
    candidates: list[tuple[int, float, Mapping[str, Any]]] = []
    for _index, segment in enumerate(_segment_candidates(row)):
        start = _finite(segment.get("start_time_sec", segment.get("start_seconds")))
        end = _finite(segment.get("end_time_sec", segment.get("end_seconds")))
        labels = segment.get("structured_labels", segment.get("labels"))
        verb, verb_status = _label_field(labels, "verb")
        noun, noun_status = _label_field(labels, "noun")
        normalized_verb = _ACTION_VERBS.get(_norm(verb), _norm(verb))
        if normalized_verb in _FILLER_VERBS:
            continue
        if normalized_verb != target:
            continue
        if verb_status != "MEASURED" or noun_status not in {"MEASURED", "NOT_OBSERVABLE"}:
            continue
        if start is None or end is None or end <= start:
            continue
        low, high = window_interval
        if start < low or end > high or not _boundary_mapping_is_source_bound(row, segment):
            continue
        confidence = _finite(segment.get("confidence")) or 0.0
        # Prefer semantic matches, then confidence, then earliest segment.
        noun_bonus = 1 if _norm(noun) in _NOUN_ALIASES else 0
        candidates.append((noun_bonus, confidence, segment))
    if not candidates:
        reasons.append("BOUNDARY_FOR_IDENTITY_NOT_FOUND")
        return None, reasons
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = candidates[0][2]
    start = _finite(chosen.get("start_time_sec", chosen.get("start_seconds")))
    end = _finite(chosen.get("end_time_sec", chosen.get("end_seconds")))
    assert start is not None and end is not None
    return {"segment": chosen, "start": start, "end": end}, reasons


def _identity_values(row: Mapping[str, Any]) -> tuple[str | None, float | None, list[str]]:
    identity = row.get("parsed_identity")
    if not isinstance(identity, Mapping):
        identity = row.get("raw_identity")
    if not isinstance(identity, Mapping):
        identity = row
    action, error = _action_from_identity(row)
    reasons = [error] if error else []
    confidence = _finite(identity.get("confidence"))
    evidence = identity.get("evidence", [])
    if isinstance(evidence, str):
        evidence_list = [evidence] if evidence.strip() else []
    elif isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        evidence_list = [str(x) for x in evidence if isinstance(x, str) and x.strip()]
    else:
        evidence_list = []
    if not evidence_list:
        reasons.append("IDENTITY_EVIDENCE_MISSING")
    return action, confidence, evidence_list


def _camera_candidate(
    identity_row: Mapping[str, Any],
    boundary_row: Mapping[str, Any] | None,
    *,
    window_interval: tuple[float, float],
    row_index: int,
) -> dict[str, Any]:
    action, confidence, identity_evidence = _identity_values(identity_row)
    camera_id = identity_row.get("camera_id")
    camera = str(camera_id).strip() if isinstance(camera_id, str) and camera_id.strip() else None
    reasons: list[str] = []
    if action is None:
        reasons.append("IDENTITY_NOT_AVAILABLE")
    boundary: dict[str, Any] | None = None
    if action is not None and boundary_row is not None:
        boundary, boundary_reasons = _boundary_for_action(boundary_row, action, window_interval)
        reasons.extend(boundary_reasons)
    elif action is not None:
        reasons.append("BOUNDARY_ROW_MISSING")
    verb, noun = action.rsplit(" ", 1) if action else (None, None)
    fields: dict[str, Any] = {"attributes": None, "location": None, "hand": None}
    field_status: dict[str, str] = {
        "verb": "MEASURED" if verb else "NOT_MEASURED",
        "noun": "MEASURED" if noun else "NOT_MEASURED",
        "attributes": "NOT_MEASURED",
        "location": "NOT_MEASURED",
        "hand": "NOT_MEASURED",
    }
    boundary_evidence: list[str] = []
    start = end = None
    boundary_status = "NOT_MEASURED"
    if boundary is not None:
        segment = boundary["segment"]
        start, end = boundary["start"], boundary["end"]
        boundary_status = "MEASURED"
        labels = segment.get("structured_labels", segment.get("labels"))
        for key in fields:
            value, status = _label_field(labels, key)
            fields[key] = value if status == "MEASURED" else None
            field_status[key] = status
        raw_ev = segment.get("evidence", [])
        if isinstance(raw_ev, str):
            boundary_evidence = [raw_ev] if raw_ev.strip() else []
        elif isinstance(raw_ev, Sequence) and not isinstance(raw_ev, (str, bytes, bytearray)):
            boundary_evidence = [str(x) for x in raw_ev if isinstance(x, str) and x.strip()]
    if boundary_status != "MEASURED":
        reasons.append("BOUNDARY_UNRESOLVED")
    evidence = list(dict.fromkeys(identity_evidence + boundary_evidence))
    if confidence is None:
        reasons.append("IDENTITY_CONFIDENCE_MISSING")
    return {
        "claim_id": (
            f"{identity_row.get('window_id')}:{camera or 'unknown'}:identity-boundary:{row_index}"
        ),
        "source_claim_id": f"{identity_row.get('window_id')}:{camera or 'unknown'}:{row_index}",
        "source_model": "qwen",
        "source_profile": "production_identity_disambiguated",
        "camera_id": camera,
        "status": "PENDING_HUMAN_REVIEW" if action else "ABSTAIN",
        "automatic_eligible": False,
        "semantic_status": "NOT_CHECKED",
        "start_seconds": start,
        "end_seconds": end,
        "start_time_sec": start,
        "end_time_sec": end,
        "boundary_status": boundary_status,
        "timestamp_basis": "source_absolute_seconds" if boundary_status == "MEASURED" else None,
        "verb": verb,
        "noun": noun,
        "attributes": fields["attributes"],
        "location": fields["location"],
        "hand": fields["hand"],
        "field_status": field_status,
        "structured_labels": {
            key: {"value": value, "status": field_status[key]}
            for key, value in (("verb", verb), ("noun", noun), *fields.items())
        },
        "label_text": action,
        "confidence": confidence,
        "evidence": evidence,
        "evidence_status": "MEASURED" if evidence else "NOT_MEASURED",
        "review_required": True,
        "accepted": False,
        "reason_codes": list(dict.fromkeys(reasons or ["BOUNDARY_MERGED"])),
        "raw_identity": _copy(identity_row, field="raw_identity"),
        "raw_boundary": _copy(boundary_row, field="raw_boundary")
        if boundary_row is not None
        else None,
    }


def _mode(values: Sequence[Any]) -> Any:
    clean = [v for v in values if v is not None and str(v).strip()]
    return Counter(clean).most_common(1)[0][0] if clean else None


def _distinct_camera_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return the non-empty camera ids represented by *rows*.

    A consensus is explicitly a cross-camera property.  Counting rows would
    allow duplicate observations from one camera to satisfy that requirement,
    so all consensus/support metrics use this normalized set instead.
    """

    return {
        camera.strip()
        for row in rows
        if isinstance(camera := row.get("camera_id"), str) and camera.strip()
    }


def _consensus_vote_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count camera votes without letting repeated camera rows inflate them.

    Production sidecars normally contain one row per camera.  Named camera
    rows are therefore deduplicated by camera id; rows without a camera id
    remain individual anonymous observations instead of disappearing from the
    denominator.
    """

    camera_ids = _distinct_camera_ids(rows)
    anonymous_count = sum(
        not (isinstance(row.get("camera_id"), str) and row.get("camera_id", "").strip())
        for row in rows
    )
    return len(camera_ids) + anonymous_count


def _window_candidate(
    action: str, rows: Sequence[Mapping[str, Any]], *, window_id: str
) -> dict[str, Any]:
    supports = [row for row in rows if row.get("label_text") == action]
    consensus_support_count = _consensus_vote_count(supports)
    consensus_observation_count = _consensus_vote_count(rows)
    confidence_values = [
        float(row["confidence"]) for row in supports if _finite(row.get("confidence")) is not None
    ]
    measured = [row for row in supports if row.get("boundary_status") == "MEASURED"]
    starts = [
        float(row["start_seconds"])
        for row in measured
        if _finite(row.get("start_seconds")) is not None
    ]
    ends = [
        float(row["end_seconds"]) for row in measured if _finite(row.get("end_seconds")) is not None
    ]
    reasons: list[str] = []
    if consensus_support_count < 2:
        reasons.append("CONSENSUS_WEAK")
    measured_camera_ids = _distinct_camera_ids(measured)
    if len(measured_camera_ids) < MIN_BOUNDARY_CONSENSUS_CAMERAS:
        reasons.append("BOUNDARY_SUPPORT_LT_2_CAMERAS")
    for row in supports:
        for reason in row.get("reason_codes", []):
            if isinstance(reason, str) and reason not in reasons:
                reasons.append(reason)
    start = statistics.median(starts) if starts else None
    end = statistics.median(ends) if ends else None
    if start is None or end is None:
        reasons.append("BOUNDARY_UNRESOLVED")
    elif end <= start:
        start = end = None
        reasons.append("BOUNDARY_MEDIAN_INVALID")
    confidence = statistics.fmean(confidence_values) if confidence_values else None
    evidence: list[str] = []
    for row in supports:
        for item in row.get("evidence", []):
            if item not in evidence:
                evidence.append(item)
    fields = {
        key: _mode([row.get(key) for row in supports]) for key in ("attributes", "location", "hand")
    }
    field_status = {
        "verb": "MEASURED",
        "noun": "MEASURED",
        **{
            key: "MEASURED" if value is not None else "NOT_MEASURED"
            for key, value in fields.items()
        },
    }
    verb, noun = action.rsplit(" ", 1)
    return {
        "claim_id": f"{window_id}:qwen:consensus:{_norm(action).replace(' ', '-')}",
        "source_model": "qwen",
        "source_profile": "production_identity_disambiguated+boundary_merge",
        "status": "PENDING_HUMAN_REVIEW",
        "automatic_eligible": False,
        "semantic_status": "NOT_CHECKED",
        "start_seconds": start,
        "end_seconds": end,
        "start_time_sec": start,
        "end_time_sec": end,
        "boundary_status": "MEASURED" if start is not None and end is not None else "NOT_MEASURED",
        "timestamp_basis": "source_absolute_seconds"
        if start is not None and end is not None
        else None,
        "verb": verb,
        "noun": noun,
        **fields,
        "field_status": field_status,
        "structured_labels": {
            key: {"value": value, "status": field_status[key]}
            for key, value in (("verb", verb), ("noun", noun), *fields.items())
        },
        "label_text": action,
        "confidence": confidence,
        "evidence": evidence[:3],
        "evidence_status": "MEASURED" if evidence else "NOT_MEASURED",
        "review_required": True,
        "accepted": False,
        "reason_codes": list(dict.fromkeys(reasons or ["IDENTITY_BOUNDARY_CONSENSUS"])),
        "supporting_camera_ids": [row.get("camera_id") for row in supports],
        "camera_support": len(supports),
        "camera_boundary_support": len(measured_camera_ids),
        "consensus_support_count": consensus_support_count,
        "consensus_observation_count": consensus_observation_count,
        "consensus_fraction": (
            consensus_support_count / consensus_observation_count
            if consensus_observation_count
            else None
        ),
        "raw_camera_claims": [_copy(row, field="raw_camera_claim") for row in supports],
    }


def merge_identity_and_boundaries(
    identity_sidecar: Mapping[str, Any] | str | Path,
    boundary_sidecar: Mapping[str, Any] | str | Path,
    *,
    identity_path: str | None = None,
    boundary_path: str | None = None,
) -> dict[str, Any]:
    identity = _load(identity_sidecar, field="identity_sidecar")
    boundary = _load(boundary_sidecar, field="boundary_sidecar")
    identity_rows = _sequence(identity.get("windows"), field="identity_sidecar.windows")
    boundary_rows = _sequence(boundary.get("windows"), field="boundary_sidecar.windows")
    boundary_by_key: dict[tuple[str, str | None], Mapping[str, Any]] = {}
    for index, raw in enumerate(boundary_rows):
        row = _mapping(raw, field=f"boundary_sidecar.windows[{index}]")
        wid = str(row.get("window_id") or "").strip()
        if not wid:
            raise ProductionIdentityBoundaryMergeError("boundary row missing window_id")
        cam = row.get("camera_id")
        cam_key = str(cam).strip() if isinstance(cam, str) and cam.strip() else None
        boundary_by_key[(wid, cam_key)] = row

    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    clean_identity_rows = 0
    measured_camera_candidates = 0
    camera_candidates = 0
    for index, raw in enumerate(identity_rows):
        row = _mapping(raw, field=f"identity_sidecar.windows[{index}]")
        wid = str(row.get("window_id") or "").strip()
        if not wid:
            raise ProductionIdentityBoundaryMergeError("identity row missing window_id")
        interval_raw = row.get("interval", [None, None])
        interval = (
            _sequence(interval_raw, field=f"{wid}.interval")
            if isinstance(interval_raw, Sequence)
            else []
        )
        low, high = (
            (_finite(interval[0]), _finite(interval[1])) if len(interval) == 2 else (None, None)
        )
        if low is None or high is None or high <= low:
            raise ProductionIdentityBoundaryMergeError(f"{wid} has invalid source interval")
        if wid not in grouped:
            grouped[wid] = {
                "window_id": wid,
                "ordinal": row.get("ordinal", len(order)),
                "source_interval": [low, high],
                "status": "ABSTAIN",
                "official_quality_status": OFFICIAL_QUALITY_STATUS,
                "official_gold_status": OFFICIAL_GOLD_STATUS,
                "quality_claim": False,
                "production_eligible": False,
                "automatic_eligible": False,
                "human_adjudication": "NOT_PERFORMED",
                "decision": "pending",
                "decision_options": ["accept", "edit", "split", "reject", "abstain"],
                "annotation_candidates": [],
                "camera_observations": [],
            }
            order.append(wid)
        cam = row.get("camera_id")
        cam_key = str(cam).strip() if isinstance(cam, str) and cam.strip() else None
        action, _, _ = _identity_values(row)
        if action is not None:
            clean_identity_rows += 1
        merged = _camera_candidate(
            row,
            boundary_by_key.get((wid, cam_key)),
            window_interval=(low, high),
            row_index=index,
        )
        grouped[wid]["camera_observations"].append(merged)
        if action is not None:
            camera_candidates += 1
            measured_camera_candidates += int(merged["boundary_status"] == "MEASURED")

    for wid in order:
        window = grouped[wid]
        observations = [
            row
            for row in window["camera_observations"]
            if row.get("label_text") in IDENTITY_ACTIONS
        ]
        counts = Counter(str(row["label_text"]) for row in observations)
        ranked = sorted(
            counts,
            key=lambda action: (
                counts[action],
                statistics.fmean(
                    [
                        float(r["confidence"])
                        for r in observations
                        if r.get("label_text") == action
                        and _finite(r.get("confidence")) is not None
                    ]
                )
                if any(
                    _finite(r.get("confidence")) is not None
                    for r in observations
                    if r.get("label_text") == action
                )
                else 0.0,
            ),
            reverse=True,
        )
        window["metrics"] = {
            "camera_observation_count": len(window["camera_observations"]),
            "clean_identity_count": len(observations),
            "action_support": dict(counts),
            "measured_boundary_observation_count": sum(
                row.get("boundary_status") == "MEASURED" for row in observations
            ),
        }
        for action in ranked[:3]:
            window["annotation_candidates"].append(
                # Keep all camera observations as the denominator for the
                # candidate consensus fraction.  ``observations`` above is
                # intentionally filtered for ranking and therefore excludes
                # abstentions/invalid identities.
                _window_candidate(action, window["camera_observations"], window_id=wid)
            )
        if window["annotation_candidates"]:
            window["status"] = "REVIEW_REQUIRED"
        else:
            window["status"] = "ABSTAIN"

    output = {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": "REVIEW_ONLY",
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": OFFICIAL_GOLD_STATUS,
        "quality_claim": False,
        "production_eligible": False,
        "automatic_eligible": False,
        "source": {
            "identity_sidecar": identity_path,
            "boundary_sidecar": boundary_path,
            "window_count": len(order),
            "camera_count": len(
                {
                    row.get("camera_id")
                    for row in identity_rows
                    if isinstance(row, Mapping) and row.get("camera_id")
                }
            ),
        },
        "metrics": {
            "window_count": len(order),
            "identity_row_count": len(identity_rows),
            "clean_identity_row_count": clean_identity_rows,
            "camera_candidate_count": camera_candidates,
            "measured_camera_boundary_count": measured_camera_candidates,
            "camera_boundary_measurement_rate": measured_camera_candidates / camera_candidates
            if camera_candidates
            else None,
            "windows_with_candidates": sum(
                bool(w["annotation_candidates"]) for w in grouped.values()
            ),
            # A measured timestamp on one camera is still useful review
            # evidence, but it is not a *consensus*.  Keep this metric strict:
            # the top candidate must have a measured boundary supported by at
            # least two distinct cameras.  The candidate itself remains
            # reviewable so single-camera evidence is not discarded.
            "windows_with_measured_consensus": sum(
                bool(w["annotation_candidates"])
                and w["annotation_candidates"][0]["boundary_status"] == "MEASURED"
                and w["annotation_candidates"][0].get("camera_boundary_support", 0)
                >= MIN_BOUNDARY_CONSENSUS_CAMERAS
                for w in grouped.values()
            ),
            "windows_requiring_review": sum(
                w["status"] == "REVIEW_REQUIRED" for w in grouped.values()
            ),
            "windows_abstained": sum(w["status"] == "ABSTAIN" for w in grouped.values()),
        },
        "contract": {
            "identity_supplies_action_wording": True,
            "boundary_supplies_times_only_when_source_mapped": True,
            "fixed_window_is_not_action_boundary": True,
            "optional_fields_are_not_inferred": True,
            "top_k_or_raw_sidecars_retained": True,
            "human_decision_required": True,
            "official_gold_status": OFFICIAL_GOLD_STATUS,
        },
        "windows": [grouped[wid] for wid in order],
        "raw_sidecars": {
            "identity": _copy(identity, field="raw_identity_sidecar"),
            "boundary": _copy(boundary, field="raw_boundary_sidecar"),
        },
    }
    return output


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics"), field="report.metrics")
    lines = [
        "# Production identity + boundary merge (review-only)",
        "",
        (
            f"- Windows: `{metrics.get('window_count')}`; candidate windows: "
            f"`{metrics.get('windows_with_candidates')}`"
        ),
        (
            f"- Clean identity rows: `{metrics.get('clean_identity_row_count')}`; "
            f"camera candidates: `{metrics.get('camera_candidate_count')}`"
        ),
        f"- Measured camera boundaries: `{metrics.get('measured_camera_boundary_count')}` "
        f"(rate `{metrics.get('camera_boundary_measurement_rate')}`)",
        "- Official quality: `NOT_MEASURED`; every candidate remains `PENDING_HUMAN_REVIEW`.",
        "",
        "| Window | Status | Top candidate | Camera support | Boundary | Reasons |",
        "|---|---|---|---:|---|---|",
    ]
    for raw in _sequence(report.get("windows"), field="report.windows"):
        window = _mapping(raw, field="report.window")
        candidates = _sequence(
            window.get("annotation_candidates", []), field="window.annotation_candidates"
        )
        top = _mapping(candidates[0], field="window.annotation_candidates[0]") if candidates else {}
        reasons = ", ".join(str(x) for x in top.get("reason_codes", []))
        lines.append(
            f"| {window.get('window_id')} | {window.get('status')} | "
            f"{top.get('label_text', '—')} | "
            f"{top.get('camera_support', 0)} | {top.get('boundary_status', '—')} | {reasons} |"
        )
    return "\n".join(lines) + "\n"
