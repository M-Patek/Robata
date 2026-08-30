"""Project recorded Qwen structured claims into review-only annotations.

The projection is intentionally benchmark-local and sidecar-only.  Qwen claims
are eligible for consensus only when they carry an explicit source-absolute
interval, a measured boundary status, a measured verb/noun pair, and a
non-empty evidence field.  Claims with local, missing, or out-of-window times
are retained as raw claims with reasons, but are never shifted into a window.

Eligible claims are de-duplicated by normalized verb/noun pair and temporal
overlap across camera rows.  A deterministic, chronological projection emits
``label_text`` in annotation-principal ``verb attributes noun location with
hand`` order (including only measured optional fields).  WeMM Top-K is
retained as context only; it does not change Qwen consensus.  The resulting
artifact is always ``SURROGATE_ONLY``/``NOT_MEASURED`` and cannot be used as
gold or as a production annotation file.
"""

from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

PRODUCTION_ANNOTATION_PROJECTION_VERSION: Final = "robata-production-annotation-projection-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "SURROGATE_ONLY"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
SOURCE_TIMESTAMP_BASIS: Final = "source_absolute_seconds"
DEFAULT_TOP_K: Final = 5
DEFAULT_MIN_CONSENSUS_CAMERAS: Final = 2
DEFAULT_MERGE_GAP_SECONDS: Final = 0.05
BOUNDARY_VALID_STATUSES: Final = frozenset(
    {"MEASURED", "EXPLICIT", "SOURCE_BOUND", "SOURCE_ABSOLUTE"}
)
BOUNDARY_INVALID_STATUSES: Final = frozenset(
    {"WINDOW_BOUND_ONLY", "FIXED_WINDOW", "UNRESOLVED", "NOT_MEASURED"}
)
_VERB_INFLECTIONS: Final = {
    "adjusts": "adjust",
    "adjusting": "adjust",
    "arranges": "arrange",
    "arranging": "arrange",
    "flattens": "flatten",
    "flattening": "flatten",
    "folds": "fold",
    "folding": "fold",
    "moves": "move",
    "moving": "move",
    "places": "place",
    "placing": "place",
    "reaches": "reach",
    "reaching": "reach",
    "smooths": "smooth",
    "smoothing": "smooth",
    "spreads": "spread",
    "spreading": "spread",
    "takes": "take",
    "taking": "take",
    "turns": "turn",
    "turning": "turn",
    "wipes": "wipe",
    "wiping": "wipe",
}
_MISSING = object()


class ProductionAnnotationProjectionError(ValueError):
    """Raised when a sidecar cannot be projected safely."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionAnnotationProjectionError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionAnnotationProjectionError(f"{field} must be an array")
    return value


def _copy_json(value: object, *, field: str) -> Any:
    """Deep-copy JSON data without calculating an identity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionAnnotationProjectionError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionAnnotationProjectionError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{i}]") for i, child in enumerate(value)]
    raise ProductionAnnotationProjectionError(f"{field} must be JSON-compatible")


def _text(value: object) -> str:
    return str(value or "").strip()


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text).split())


def _verb(value: object) -> str:
    text = _normalise(value)
    if text in {"pick up", "pickup", "picks up", "picking up"}:
        return "pick up"
    tokens = text.split()
    if tokens:
        tokens[-1] = _VERB_INFLECTIONS.get(tokens[-1], tokens[-1])
    return " ".join(tokens)


def _pair(verb: object, noun: object) -> tuple[str, str] | None:
    result = (_verb(verb), _normalise(noun))
    return result if result[0] and result[1] else None


def _field_value(record: Mapping[str, Any], field: str) -> tuple[bool, Any, str]:
    labels = record.get("structured_labels", record.get("labels"))
    source: Mapping[str, Any]
    if field in {"verb", "noun", "attributes", "location", "hand"} and isinstance(labels, Mapping):
        source = labels
    else:
        source = record
    if field not in source:
        return False, None, "NOT_MEASURED"
    raw = source[field]
    if isinstance(raw, Mapping) and "value" in raw:
        status = _text(raw.get("status")).upper()
        if not status:
            status = _text(record.get(f"{field}_status")).upper()
        if not status:
            status = "NOT_OBSERVABLE" if raw.get("value") is None else "MEASURED"
        return True, raw.get("value"), status
    if raw is None:
        return True, None, _text(record.get(f"{field}_status")).upper() or "NOT_OBSERVABLE"
    return True, raw, _text(record.get(f"{field}_status")).upper() or "MEASURED"


def _source_interval(row: Mapping[str, Any]) -> tuple[float, float] | None:
    raw = row.get("interval")
    if raw is not None:
        try:
            values = _sequence(raw, field="source.interval")
        except ProductionAnnotationProjectionError:
            return None
        if len(values) == 2:
            start, end = _finite(values[0]), _finite(values[1])
            if start is not None and end is not None and start >= 0 and end > start:
                return start, end
    for start_key, end_key in (
        ("start_seconds", "end_seconds"),
        ("window_start_seconds", "window_end_seconds"),
        ("start_time_sec", "end_time_sec"),
    ):
        if start_key not in row and end_key not in row:
            continue
        start, end = _finite(row.get(start_key)), _finite(row.get(end_key))
        if start is not None and end is not None and start >= 0 and end > start:
            return start, end
    return None


def _segment_interval(segment: Mapping[str, Any]) -> tuple[float, float] | None:
    for start_key, end_key in (
        ("start_time_sec", "end_time_sec"),
        ("start_seconds", "end_seconds"),
    ):
        if start_key not in segment and end_key not in segment:
            continue
        start, end = _finite(segment.get(start_key)), _finite(segment.get(end_key))
        if start is not None and end is not None and end > start:
            return start, end
        return None
    return None


def _evidence_values(segment: Mapping[str, Any]) -> tuple[bool, list[Any]]:
    raw = segment.get("evidence", _MISSING)
    if raw is _MISSING or raw is None:
        return False, []
    if isinstance(raw, str):
        return bool(raw.strip()), [raw] if raw.strip() else []
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        return False, []
    values = [item for item in raw if isinstance(item, str) and item.strip()]
    return bool(values), values


def _segment_status(segment: Mapping[str, Any]) -> str:
    """Read or conservatively derive the source segment status.

    Canonical structured sidecars emit ``status`` explicitly.  Older native
    rows may omit it; in that case mirror the structured adapter's narrow
    fallback (measured when at least one structured label is measured) rather
    than treating a missing marker as a successful model result.  This value
    is provenance only and never replaces the projection's consensus status.
    """

    explicit = _text(segment.get("status")).upper()
    if explicit:
        return explicit
    measured = any(
        _field_value(segment, field)[2] == "MEASURED"
        for field in ("verb", "noun", "attributes", "location", "hand")
    )
    return "MEASURED" if measured else "NOT_MEASURED"


def _evidence_status(segment: Mapping[str, Any], *, has_evidence: bool) -> str:
    """Return the explicit evidence status without inventing evidence."""

    explicit = _text(segment.get("evidence_status")).upper()
    if explicit:
        return explicit
    raw = segment.get("evidence", _MISSING)
    if raw is None:
        return "NOT_OBSERVABLE"
    return "MEASURED" if has_evidence else "NOT_MEASURED"


def _timestamp_basis(value: object) -> str | None:
    """Read a timestamp basis marker without rewriting source times."""

    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _parse_status(row: Mapping[str, Any], segments: Sequence[Any]) -> str:
    parsed = row.get("parsed_structured")
    if isinstance(parsed, Mapping) and _text(parsed.get("parse_status")):
        return _text(parsed.get("parse_status")).upper()
    if _text(row.get("parse_status")):
        return _text(row.get("parse_status")).upper()
    return "PARSED" if segments else "EMPTY"


def _decode_raw_segments(raw_text: object) -> list[Mapping[str, Any]]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []
    try:
        decoded = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            return []
        try:
            decoded = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(decoded, Mapping):
        return []
    raw_segments = decoded.get("segments", [])
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes, bytearray)):
        return []
    return [item for item in raw_segments if isinstance(item, Mapping)]


def _has_explicit_source_shape(segment: Mapping[str, Any]) -> bool:
    """Check shape only; semantic validity is evaluated later per claim."""

    interval = _segment_interval(segment)
    status = _text(segment.get("boundary_status")).upper()
    has_evidence, _ = _evidence_values(segment)
    return interval is not None and status in BOUNDARY_VALID_STATUSES and has_evidence


def _window_id(row: Mapping[str, Any], *, field: str) -> str:
    result = _text(row.get("window_id"))
    if not result:
        raise ProductionAnnotationProjectionError(f"{field}.window_id must be non-empty")
    return result


def _window_bounds_from_review(review: Mapping[str, Any] | None) -> dict[str, tuple[float, float]]:
    if review is None:
        return {}
    raw = review.get("windows", review.get("items", []))
    if raw is None:
        return {}
    result: dict[str, tuple[float, float]] = {}
    rows = _sequence(raw, field="review.windows")
    for index, item in enumerate(rows):
        row = _mapping(item, field=f"review[{index}]")
        identifier = _window_id(row, field=f"review[{index}]")
        interval = _source_interval(row)
        if interval is not None:
            result[identifier] = interval
    return result


def _qwen_rows(
    sidecar: Mapping[str, Any],
    *,
    review_bounds: Mapping[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    raw_windows = sidecar.get("windows")
    if raw_windows is None:
        raise ProductionAnnotationProjectionError("qwen sidecar must contain windows")
    sidecar_timestamp_basis = _timestamp_basis(sidecar.get("timestamp_basis"))
    rows: list[dict[str, Any]] = []
    for window_index, raw in enumerate(_sequence(raw_windows, field="qwen.windows")):
        window = _mapping(raw, field=f"qwen.windows[{window_index}]")
        identifier = _window_id(window, field=f"qwen.windows[{window_index}]")
        fallback = _source_interval(window) or review_bounds.get(identifier)
        window_timestamp_basis = _timestamp_basis(window.get("timestamp_basis"))
        if window_timestamp_basis is None:
            window_timestamp_basis = sidecar_timestamp_basis
        # Native sidecar: one row per camera.
        if (
            "camera_id" in window
            or "raw_text" in window
            or "parsed_structured" in window
            or "segments" in window
        ):
            segments = window.get("segments", _MISSING)
            if segments is _MISSING or (
                isinstance(segments, Sequence)
                and not isinstance(segments, (str, bytes, bytearray))
                and not segments
            ):
                parsed = window.get("parsed_structured")
                segments = parsed.get("segments", []) if isinstance(parsed, Mapping) else []
            rows.append(
                {
                    "window_id": identifier,
                    "camera_id": _text(window.get("camera_id")) or "__aggregate__",
                    "row_index": window_index,
                    "source_interval": fallback,
                    "timestamp_basis": window_timestamp_basis,
                    "parse_status": _parse_status(
                        window,
                        segments if isinstance(segments, Sequence) else [],
                    ),
                    "segments": [
                        _mapping(item, field=f"qwen.{identifier}.segments[{i}]")
                        for i, item in enumerate(
                            _sequence(segments, field=f"qwen.{identifier}.segments")
                        )
                    ],
                    "raw_row": _copy_json(window, field=f"qwen.{identifier}.raw_row"),
                }
            )
            continue
        # Structured envelope: recover camera rows from candidate_sources when
        # their raw JSON is available.  If not, retain the merged section as a
        # single aggregate source rather than inventing camera attribution.
        models = window.get("models")
        qwen = models.get("qwen") if isinstance(models, Mapping) else None
        qwen_map = _mapping(qwen, field=f"qwen.{identifier}.models.qwen")
        sources = qwen_map.get("candidate_sources", [])
        recovered = False
        recovered_rows: list[dict[str, Any]] = []
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
            for source_index, raw_source in enumerate(sources):
                source = _mapping(
                    raw_source,
                    field=f"qwen.{identifier}.candidate_sources[{source_index}]",
                )
                segments = _decode_raw_segments(source.get("raw_text"))
                if not segments:
                    continue
                recovered = True
                source_row = dict(source)
                source_row["segments"] = segments
                source_timestamp_basis = _timestamp_basis(source.get("timestamp_basis"))
                if source_timestamp_basis is None:
                    source_timestamp_basis = window_timestamp_basis
                recovered_rows.append(
                    {
                        "window_id": identifier,
                        "camera_id": _text(source.get("camera_id")) or f"camera_{source_index}",
                        "row_index": window_index,
                        "source_interval": fallback,
                        "timestamp_basis": source_timestamp_basis,
                        "parse_status": "PARSED",
                        "segments": segments,
                        "raw_row": _copy_json(
                            source_row,
                            field=f"qwen.{identifier}.camera_{source_index}.raw_row",
                        ),
                    }
                )
        # Candidate-source raw JSON in older envelopes predates the explicit
        # boundary/evidence fields.  Do not manufacture those fields; fall
        # back to the canonical merged section when no recovered row carries
        # the source-bound shape.
        if recovered and any(
            _has_explicit_source_shape(segment)
            for source_row in recovered_rows
            for segment in source_row["segments"]
        ):
            rows.extend(recovered_rows)
            continue
        segments = qwen_map.get("segments", [])
        rows.append(
            {
                "window_id": identifier,
                "camera_id": "__aggregate__",
                "row_index": window_index,
                "source_interval": fallback,
                "timestamp_basis": _timestamp_basis(qwen_map.get("timestamp_basis"))
                or window_timestamp_basis,
                "parse_status": _text(qwen_map.get("parse_status")).upper()
                or ("PARSED" if segments else "EMPTY"),
                "segments": [
                    _mapping(item, field=f"qwen.{identifier}.segments[{i}]")
                    for i, item in enumerate(
                        _sequence(segments, field=f"qwen.{identifier}.segments")
                    )
                ],
                "raw_row": _copy_json(window, field=f"qwen.{identifier}.raw_row"),
            }
        )
    return rows


def _wemm_rows(sidecar: Mapping[str, Any] | None, *, top_k: int) -> dict[str, dict[str, Any]]:
    if sidecar is None:
        return {}
    raw_windows = sidecar.get("windows")
    if raw_windows is None:
        raise ProductionAnnotationProjectionError("wemm sidecar must contain windows")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_sequence(raw_windows, field="wemm.windows")):
        row = _mapping(raw, field=f"wemm.windows[{index}]")
        identifier = _window_id(row, field=f"wemm.windows[{index}]")
        if identifier in result:
            raise ProductionAnnotationProjectionError(f"duplicate wemm window_id: {identifier}")
        model = row.get("model")
        if isinstance(model, Mapping):
            values = model.get("predictions", model.get("candidates", []))
            status = _text(model.get("status")).upper() or "UNKNOWN"
        else:
            values = row.get("predictions", row.get("candidates", []))
            status = _text(row.get("status")).upper() or "UNKNOWN"
        candidates = _sequence(values, field=f"wemm.{identifier}.predictions")
        ranked = sorted(
            enumerate(candidates),
            key=lambda pair: (
                _finite(_mapping(pair[1], field="wemm.prediction").get("rank")) or 10**9,
                pair[0],
            ),
        )
        result[identifier] = {
            "status": status,
            "top_k": [
                _copy_json(
                    _mapping(candidate, field=f"wemm.{identifier}.prediction"),
                    field=f"wemm.{identifier}.prediction",
                )
                for _, candidate in ranked[:top_k]
            ],
            "raw_row": _copy_json(row, field=f"wemm.{identifier}.raw_row"),
        }
    return result


def _claim(
    *,
    window_id: str,
    camera_id: str,
    row_index: int,
    segment_index: int,
    segment: Mapping[str, Any],
    source_interval: tuple[float, float] | None,
    parse_status: str,
    timestamp_basis: str | None,
) -> dict[str, Any]:
    claim_id = f"{window_id}:{camera_id}:{row_index}:{segment_index}"
    reasons: list[str] = []
    if timestamp_basis is not None and timestamp_basis != SOURCE_TIMESTAMP_BASIS:
        # Retain the claim and its original interval for audit, but never
        # reinterpret a relative/unknown clock as source-absolute seconds.
        reasons.append("TIMESTAMP_BASIS_UNSUPPORTED")
    interval = _segment_interval(segment)
    boundary_status = _text(segment.get("boundary_status")).upper()
    if parse_status != "PARSED":
        reasons.append("PARSE_INVALID")
    if interval is None:
        reasons.append("BOUNDARY_MISSING_OR_INVALID")
    elif source_interval is None:
        reasons.append("SOURCE_INTERVAL_MISSING")
    elif not (
        interval[0] >= source_interval[0]
        and interval[1] <= source_interval[1]
        and interval[1] > interval[0]
    ):
        reasons.append("BOUNDARY_OUT_OF_SOURCE")
    if not boundary_status:
        reasons.append("BOUNDARY_STATUS_MISSING")
    elif (
        boundary_status in BOUNDARY_INVALID_STATUSES
        or boundary_status not in BOUNDARY_VALID_STATUSES
    ):
        reasons.append("BOUNDARY_NOT_SOURCE_MEASURED")
    verb_present, raw_verb, verb_status = _field_value(segment, "verb")
    noun_present, raw_noun, noun_status = _field_value(segment, "noun")
    pair = _pair(raw_verb, raw_noun) if verb_present and noun_present else None
    if not verb_present or not raw_verb or verb_status in {"NOT_MEASURED", "NOT_OBSERVABLE"}:
        reasons.append("VERB_NOT_MEASURED")
    if not noun_present or not raw_noun or noun_status in {"NOT_MEASURED", "NOT_OBSERVABLE"}:
        reasons.append("NOUN_NOT_MEASURED")
    has_evidence, evidence = _evidence_values(segment)
    evidence_status = _evidence_status(segment, has_evidence=has_evidence)
    if not has_evidence:
        reasons.append("EVIDENCE_MISSING")
    elif evidence_status in {"NOT_MEASURED", "NOT_OBSERVABLE", "INVALID"}:
        reasons.append("EVIDENCE_NOT_MEASURED")
    confidence = _finite(segment.get("confidence")) or 0.0
    source_status = _segment_status(segment)
    field_statuses = {
        field: _field_value(segment, field)[2]
        for field in ("verb", "noun", "attributes", "location", "hand")
    }
    return {
        "claim_id": claim_id,
        "window_id": window_id,
        "camera_id": camera_id,
        "row_index": row_index,
        "segment_index": segment_index,
        "pair": list(pair) if pair else None,
        "interval": list(interval) if interval else None,
        "source_interval": list(source_interval) if source_interval else None,
        "timestamp_basis": timestamp_basis,
        "timestamp_basis_status": (
            "MEASURED"
            if timestamp_basis == SOURCE_TIMESTAMP_BASIS
            else "NOT_MEASURED"
            if timestamp_basis is None
            else "UNSUPPORTED"
        ),
        "boundary_status": boundary_status or None,
        "confidence": confidence,
        # ``source_status`` is the model segment status.  It is deliberately
        # distinct from the candidate-level ``status`` (CONSENSUS or
        # SINGLE_SOURCE), which is assigned only after cross-camera clustering.
        "source_status": source_status,
        "field_statuses": field_statuses,
        "evidence": _copy_json(evidence, field=f"{claim_id}.evidence"),
        "evidence_status": evidence_status,
        "parse_status": parse_status,
        "valid_source_bound": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "raw_claim": _copy_json(segment, field=f"{claim_id}.raw_claim"),
    }


def _duration(interval: Sequence[float]) -> float:
    return max(0.0, float(interval[1]) - float(interval[0]))


def _overlap_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = max(
        0.0, min(float(first[1]), float(second[1])) - max(float(first[0]), float(second[0]))
    )
    denominator = min(_duration(first), _duration(second))
    return intersection / denominator if denominator > 0 else 0.0


def _cluster_claims(
    claims: Sequence[Mapping[str, Any]], *, merge_gap_seconds: float
) -> list[list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for claim in claims:
        grouped[tuple(claim["pair"])].append(claim)
    clusters: list[list[Mapping[str, Any]]] = []
    for pair in sorted(grouped):
        ordered = sorted(
            grouped[pair],
            key=lambda item: (
                float(item["interval"][0]),
                float(item["interval"][1]),
                str(item["camera_id"]),
                int(item["row_index"]),
                int(item["segment_index"]),
            ),
        )
        pair_clusters: list[list[Mapping[str, Any]]] = []
        for claim in ordered:
            target: list[Mapping[str, Any]] | None = None
            for candidate_cluster in reversed(pair_clusters):
                cluster_end = max(float(item["interval"][1]) for item in candidate_cluster)
                cluster_start = min(float(item["interval"][0]) for item in candidate_cluster)
                interval = claim["interval"]
                if float(interval[0]) <= cluster_end + merge_gap_seconds and (
                    _overlap_ratio(interval, (cluster_start, cluster_end)) > 0
                    or float(interval[0]) <= cluster_end + merge_gap_seconds
                ):
                    target = candidate_cluster
                    break
            if target is None:
                target = []
                pair_clusters.append(target)
            target.append(claim)
        clusters.extend(pair_clusters)
    return clusters


def _principal_claim(cluster: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return sorted(
        cluster,
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            str(item["camera_id"]),
            int(item["row_index"]),
            int(item["segment_index"]),
        ),
    )[0]


def _value_signature(value: object) -> str:
    """Return a deterministic comparison key without deriving an identity."""

    if isinstance(value, str):
        return _normalise(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return json.dumps(
            sorted(_normalise(item) if isinstance(item, str) else repr(item) for item in value),
            ensure_ascii=False,
        )
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return repr(value)


def _field_projection(
    cluster: Sequence[Mapping[str, Any]],
    field: str,
    *,
    principal_claim: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose one optional field by camera support and retain alternatives."""

    observations: dict[str, dict[str, Any]] = {}
    any_explicit = False
    any_not_observable = False
    for claim in cluster:
        raw_claim = claim.get("raw_claim", {})
        if not isinstance(raw_claim, Mapping):
            continue
        present, value, status = _field_value(raw_claim, field)
        if not present:
            continue
        any_explicit = True
        if status in {"NOT_MEASURED", "NOT_OBSERVABLE"} or value is None:
            any_not_observable = True
            continue
        signature = _value_signature(value)
        item = observations.setdefault(
            signature,
            {
                "value": _copy_json(value, field=f"field.{field}.value"),
                "status": "MEASURED",
                "camera_ids": set(),
                "claim_ids": [],
                "confidence_sum": 0.0,
            },
        )
        camera_id = str(claim.get("camera_id"))
        if camera_id != "__aggregate__":
            item["camera_ids"].add(camera_id)
        item["claim_ids"].append(str(claim["claim_id"]))
        item["confidence_sum"] += float(claim.get("confidence", 0.0))
    if not observations:
        status = "NOT_OBSERVABLE" if any_explicit and any_not_observable else "NOT_MEASURED"
        return {"value": None, "status": status}, []
    alternatives: list[dict[str, Any]] = []
    for item in observations.values():
        cameras = sorted(item["camera_ids"])
        alternatives.append(
            {
                "value": item["value"],
                "status": item["status"],
                "support_count": len(cameras) or 1,
                "camera_ids": cameras,
                "claim_ids": list(item["claim_ids"]),
                "confidence_sum": round(float(item["confidence_sum"]), 6),
            }
        )
    principal_signature: str | None = None
    principal_raw = principal_claim.get("raw_claim", {})
    if isinstance(principal_raw, Mapping):
        present, value, status = _field_value(principal_raw, field)
        if present and status == "MEASURED" and value is not None:
            principal_signature = _value_signature(value)
    alternatives.sort(
        key=lambda item: (
            -int(item["support_count"]),
            -float(item["confidence_sum"]),
            0 if _value_signature(item["value"]) == principal_signature else 1,
            _value_signature(item["value"]),
        )
    )
    selected = alternatives[0]
    return {
        "value": _copy_json(selected["value"], field=f"field.{field}.selected"),
        "status": "MEASURED",
        "support_count": selected["support_count"],
        "camera_ids": selected["camera_ids"],
        "claim_ids": selected["claim_ids"],
    }, alternatives


def _display_value(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(_text(item) for item in value if _text(item))
    return _text(value)


def _annotation_label_text(
    *,
    verb: str,
    noun: str,
    fields: Mapping[str, Mapping[str, Any]],
) -> str:
    """Format measured fields as ``verb attributes noun location with hand``.

    Every measured value is copied as supplied.  In particular, a bare
    location such as ``table`` is *not* rewritten to ``at table``: adding a
    preposition would infer a spatial relation that the model did not
    explicitly observe.  Producers that need a relation should emit it in the
    location value itself (for example ``on table``).
    """

    parts = [verb]
    attributes = fields.get("attributes", {})
    if attributes.get("status") == "MEASURED":
        value = _display_value(attributes.get("value"))
        if value:
            parts.append(value)
    parts.append(noun)
    location = fields.get("location", {})
    if location.get("status") == "MEASURED":
        value = _display_value(location.get("value"))
        if value:
            parts.append(value)
    hand = fields.get("hand", {})
    if hand.get("status") == "MEASURED":
        value = _display_value(hand.get("value"))
        if value:
            parts.extend(["with", value.removeprefix("with ")])
    return " ".join(part for part in parts if part)


def _conflicts(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    ordered = sorted(claims, key=lambda item: str(item["claim_id"]))
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if first["pair"] == second["pair"]:
                continue
            if _overlap_ratio(first["interval"], second["interval"]) <= 0:
                continue
            conflicts.append(
                {
                    "claim_ids": [first["claim_id"], second["claim_id"]],
                    "pairs": [first["pair"], second["pair"]],
                    "reason": "OVERLAPPING_DISTINCT_PAIRS",
                }
            )
    return conflicts


def _candidate(
    cluster: Sequence[Mapping[str, Any]],
    *,
    min_consensus_cameras: int,
    conflict_claim_ids: set[str],
) -> dict[str, Any]:
    principal = _principal_claim(cluster)
    pair = list(principal["pair"])
    camera_ids = sorted(
        {str(item["camera_id"]) for item in cluster if item["camera_id"] != "__aggregate__"}
    )
    support_count = len(camera_ids) or 1
    status = "CONSENSUS" if support_count >= min_consensus_cameras else "SINGLE_SOURCE"
    supporting_ids = [str(item["claim_id"]) for item in cluster]
    review_required = any(claim_id in conflict_claim_ids for claim_id in supporting_ids)
    mean_confidence = sum(float(item.get("confidence", 0.0)) for item in cluster) / len(cluster)
    optional_fields = {
        field: _field_projection(cluster, field, principal_claim=principal)
        for field in ("attributes", "location", "hand")
    }
    field_values = {field: values[0] for field, values in optional_fields.items()}
    field_alternatives = {
        field: values[1] for field, values in optional_fields.items() if len(values[1]) > 1
    }
    field_conflicts = sorted(field_alternatives)
    review_required = review_required or bool(field_conflicts)
    source_statuses = sorted({str(item.get("source_status", "NOT_MEASURED")) for item in cluster})
    timestamp_bases = sorted(
        {
            str(item["timestamp_basis"])
            for item in cluster
            if item.get("timestamp_basis") is not None
        }
    )
    timestamp_basis = principal.get("timestamp_basis")
    evidence_statuses = sorted(
        {str(item.get("evidence_status", "NOT_MEASURED")) for item in cluster}
    )
    field_statuses = {
        field: str(field_values[field].get("status", "NOT_MEASURED"))
        for field in ("attributes", "location", "hand")
    }
    structured_labels = {
        "verb": {"value": pair[0], "status": "MEASURED"},
        "noun": {"value": pair[1], "status": "MEASURED"},
        **field_values,
    }
    label_text = _annotation_label_text(verb=pair[0], noun=pair[1], fields=field_values)
    return {
        "status": status,
        # Keep projection status separate from source/model status.  The alias
        # ``principal_status`` is retained for consumers that name the selected
        # source claim explicitly; both refer to the same principal segment.
        "source_status": principal.get("source_status", "NOT_MEASURED"),
        "principal_status": principal.get("source_status", "NOT_MEASURED"),
        "source_statuses": source_statuses,
        "review_required": review_required,
        "label_text": label_text,
        "label_text_order": "verb attributes noun location with hand",
        "verb": pair[0],
        "noun": pair[1],
        "structured_labels": structured_labels,
        "field_statuses": field_statuses,
        "attributes": field_values["attributes"]["value"],
        "location": field_values["location"]["value"],
        "hand": field_values["hand"]["value"],
        "field_conflicts": field_conflicts,
        "field_alternatives": field_alternatives,
        "timestamp_basis": timestamp_basis,
        "timestamp_bases": timestamp_bases,
        "timestamp_basis_status": (
            "MEASURED"
            if timestamp_basis == SOURCE_TIMESTAMP_BASIS
            else "NOT_MEASURED"
            if timestamp_basis is None
            else "UNSUPPORTED"
        ),
        "evidence_status": (
            "MEASURED"
            if "MEASURED" in evidence_statuses
            else "NOT_OBSERVABLE"
            if "NOT_OBSERVABLE" in evidence_statuses
            else "NOT_MEASURED"
        ),
        "evidence_statuses": evidence_statuses,
        "start_time_sec": principal["interval"][0],
        "end_time_sec": principal["interval"][1],
        "boundary_status": principal["boundary_status"],
        "camera_ids": camera_ids,
        "support_count": support_count,
        "claim_count": len(cluster),
        "supporting_claim_ids": supporting_ids,
        "confidence": round(mean_confidence, 6),
        "principal_claim_id": principal["claim_id"],
        "evidence": list(
            dict.fromkeys(evidence for item in cluster for evidence in item.get("evidence", []))
        ),
        "raw_claims": [copy.deepcopy(item["raw_claim"]) for item in cluster],
    }


def _window_projection(
    window_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_interval: tuple[float, float] | None,
    wemm: Mapping[str, Any] | None,
    min_consensus_cameras: int,
    merge_gap_seconds: float,
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    for row in rows:
        for segment_index, segment in enumerate(row.get("segments", [])):
            claims.append(
                _claim(
                    window_id=window_id,
                    camera_id=str(row["camera_id"]),
                    row_index=int(row["row_index"]),
                    segment_index=segment_index,
                    segment=_mapping(segment, field=f"qwen.{window_id}.segment"),
                    source_interval=row.get("source_interval") or source_interval,
                    parse_status=str(row.get("parse_status", "UNKNOWN")),
                    timestamp_basis=_timestamp_basis(row.get("timestamp_basis")),
                )
            )
    valid = [
        claim
        for claim in claims
        if claim["valid_source_bound"] and claim["pair"] is not None and claim["interval"]
    ]
    conflicts = _conflicts(valid)
    conflict_claim_ids = {claim_id for conflict in conflicts for claim_id in conflict["claim_ids"]}
    clusters = _cluster_claims(valid, merge_gap_seconds=merge_gap_seconds)
    candidates = [
        _candidate(
            cluster,
            min_consensus_cameras=min_consensus_cameras,
            conflict_claim_ids=conflict_claim_ids,
        )
        for cluster in clusters
    ]
    candidates.sort(
        key=lambda item: (
            float(item["start_time_sec"]),
            float(item["end_time_sec"]),
            -int(item["support_count"]),
            -float(item["confidence"]),
            str(item["label_text"]),
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["annotation_order"] = index
    row_timestamp_bases = sorted(
        {str(row["timestamp_basis"]) for row in rows if row.get("timestamp_basis") is not None}
    )
    window_timestamp_basis = row_timestamp_bases[0] if len(row_timestamp_bases) == 1 else None
    invalid_reasons = sorted(
        {
            reason
            for claim in claims
            if not claim["valid_source_bound"]
            for reason in claim["reasons"]
        }
    )
    abstained = not candidates
    abstention_reasons = invalid_reasons or (["NO_SOURCE_BOUND_CLAIMS"] if abstained else [])
    if abstained and not claims:
        abstention_reasons = ["NO_QWEN_STRUCTURED_CLAIMS"]
    if candidates and conflicts:
        window_status = "REVIEW_REQUIRED"
    elif candidates:
        window_status = "PROJECTED"
    else:
        window_status = "ABSTAIN"
    return {
        "window_id": window_id,
        "source_interval": list(source_interval) if source_interval else None,
        "timestamp_basis": window_timestamp_basis,
        "timestamp_bases": row_timestamp_bases,
        "timestamp_basis_status": (
            "UNSUPPORTED"
            if any(basis != SOURCE_TIMESTAMP_BASIS for basis in row_timestamp_bases)
            else "MEASURED"
            if window_timestamp_basis == SOURCE_TIMESTAMP_BASIS
            else "NOT_MEASURED"
            if not row_timestamp_bases
            else "CONFLICT"
        ),
        "annotation": {
            "status": window_status,
            "segments": candidates,
            "label_order": "verb_then_noun",
            "boundaries_are_source_observations": True,
        },
        "raw_claims": claims,
        "valid_claim_count": len(valid),
        "conflicts": conflicts,
        "abstention": {
            "abstained": abstained,
            "reason_codes": list(dict.fromkeys(abstention_reasons)),
        },
        "wemm": {
            "status": wemm.get("status") if wemm else "NOT_PROVIDED",
            "top_k": copy.deepcopy(wemm.get("top_k", [])) if wemm else [],
            "raw_provenance": copy.deepcopy(wemm.get("raw_row")) if wemm else None,
        },
    }


def project_production_annotations(
    qwen_sidecar: Mapping[str, Any],
    wemm_shadow: Mapping[str, Any] | None = None,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_consensus_cameras: int = DEFAULT_MIN_CONSENSUS_CAMERAS,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    review: Mapping[str, Any] | None = None,
    input_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, review-only annotation projection."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ProductionAnnotationProjectionError("top_k must be a positive integer")
    if (
        isinstance(min_consensus_cameras, bool)
        or not isinstance(min_consensus_cameras, int)
        or min_consensus_cameras <= 0
    ):
        raise ProductionAnnotationProjectionError(
            "min_consensus_cameras must be a positive integer"
        )
    gap = _finite(merge_gap_seconds)
    if gap is None or gap < 0:
        raise ProductionAnnotationProjectionError("merge_gap_seconds must be non-negative")
    qwen = _mapping(qwen_sidecar, field="qwen_sidecar")
    wemm = _mapping(wemm_shadow, field="wemm_shadow") if wemm_shadow is not None else None
    review_map = _mapping(review, field="review") if review is not None else None
    review_bounds = _window_bounds_from_review(review_map)
    qwen_rows = _qwen_rows(qwen, review_bounds=review_bounds)
    wemm_rows = _wemm_rows(wemm, top_k=top_k)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    intervals: dict[str, tuple[float, float] | None] = {}
    for row in qwen_rows:
        identifier = str(row["window_id"])
        grouped[identifier].append(row)
        intervals.setdefault(identifier, row.get("source_interval"))
    # Preserve Qwen source order and include windows present only in WeMM for an
    # explicit abstention row; this makes missing-Qwen evidence observable.
    window_ids = list(grouped)
    for identifier in wemm_rows:
        if identifier not in grouped:
            window_ids.append(identifier)
            grouped[identifier] = []
            intervals[identifier] = review_bounds.get(identifier)
    windows = [
        _window_projection(
            identifier,
            grouped[identifier],
            source_interval=intervals.get(identifier),
            wemm=wemm_rows.get(identifier),
            min_consensus_cameras=min_consensus_cameras,
            merge_gap_seconds=float(gap),
        )
        for identifier in window_ids
    ]
    projected = [row for row in windows if row["annotation"]["segments"]]
    abstained = [row for row in windows if row["abstention"]["abstained"]]
    candidates = [segment for row in windows for segment in row["annotation"]["segments"]]
    conflicts = [row for row in windows if row["conflicts"]]
    timestamp_bases = sorted(
        {
            str(basis)
            for row in windows
            for basis in row.get("timestamp_bases", [])
            if basis is not None
        }
    )
    timestamp_basis = timestamp_bases[0] if len(timestamp_bases) == 1 else None
    timestamp_basis_status = (
        "UNSUPPORTED"
        if any(basis != SOURCE_TIMESTAMP_BASIS for basis in timestamp_bases)
        else "MEASURED"
        if timestamp_basis == SOURCE_TIMESTAMP_BASIS
        else "CONFLICT"
        if len(timestamp_bases) > 1
        else "NOT_MEASURED"
    )
    return cast(
        dict[str, Any],
        _copy_json(
            {
                "format": PRODUCTION_ANNOTATION_PROJECTION_VERSION,
                "authority": AUTHORITY,
                "status": STATUS,
                "official_quality_status": OFFICIAL_QUALITY_STATUS,
                "official_gold_status": "NOT_ESTABLISHED",
                "timestamp_basis": timestamp_basis,
                "timestamp_bases": timestamp_bases,
                "timestamp_basis_status": timestamp_basis_status,
                "quality_claim": False,
                "production_eligible": False,
                "source": {
                    "qwen_format": qwen.get("format"),
                    "wemm_format": wemm.get("format") if wemm else None,
                    "review_geometry_used": review_map is not None,
                    "input_paths": dict(input_paths or {}),
                },
                "parameters": {
                    "top_k": top_k,
                    "min_consensus_cameras": min_consensus_cameras,
                    "merge_gap_seconds": float(gap),
                    "normalization": "case/punctuation/explicit inflection cleanup only",
                    "label_order": "verb_then_noun",
                    "structured_pair_order": "verb_then_noun",
                    "label_text_order": "verb attributes noun location with hand",
                },
                "windows": windows,
                "metrics": {
                    "window_count": len(windows),
                    "projected_window_count": len(projected),
                    "abstained_window_count": len(abstained),
                    "conflict_window_count": len(conflicts),
                    "candidate_count": len(candidates),
                    "consensus_candidate_count": sum(
                        segment["status"] == "CONSENSUS" for segment in candidates
                    ),
                    "single_source_candidate_count": sum(
                        segment["status"] == "SINGLE_SOURCE" for segment in candidates
                    ),
                    "review_conflict_candidate_count": sum(
                        bool(segment.get("review_required")) for segment in candidates
                    ),
                    "field_conflict_counts": {
                        field: sum(
                            field in segment.get("field_conflicts", []) for segment in candidates
                        )
                        for field in ("attributes", "location", "hand")
                    },
                    "raw_claim_count": sum(len(row["raw_claims"]) for row in windows),
                    "valid_source_bound_claim_count": sum(
                        int(row["valid_claim_count"]) for row in windows
                    ),
                    "invalid_claim_count": sum(
                        len(row["raw_claims"]) - int(row["valid_claim_count"]) for row in windows
                    ),
                },
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
                    "sha_or_digest_computed": False,
                    "hash_or_sha_used": False,
                    "raw_claims_preserved": True,
                    "wemm_used_as_annotation_evidence": False,
                },
                "limitations": [
                    "The projection is a deterministic candidate annotation, not official gold.",
                    (
                        "Only explicit source-bound Qwen intervals with measured evidence "
                        "are eligible."
                    ),
                    (
                        "Invalid or out-of-window boundaries are retained but never shifted "
                        "or clipped."
                    ),
                    "WeMM Top-K is retained as context and does not override Qwen evidence.",
                    "Semantic synonyms are not added; normalization is limited to morphology.",
                    (
                        "Unsupported timestamp bases are retained as raw provenance but "
                        "never projected."
                    ),
                ],
            },
            field="projection",
        ),
    )


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionAnnotationProjectionError(f"cannot load JSON {source}: {exc}") from exc
    return dict(_mapping(value, field=str(source)))


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics", {}), field="report.metrics")
    lines = [
        "# Production annotation projection",
        "",
        "> **SURROGATE_ONLY.** Official quality remains `NOT_MEASURED`; output is not gold.",
        "",
        (
            f"- Windows: `{metrics.get('window_count', 0)}`; projected: "
            f"`{metrics.get('projected_window_count', 0)}`; abstained: "
            f"`{metrics.get('abstained_window_count', 0)}`"
        ),
        (
            f"- Candidates: `{metrics.get('candidate_count', 0)}`; valid "
            f"source-bound claims: `{metrics.get('valid_source_bound_claim_count', 0)}`; "
            f"invalid retained claims: `{metrics.get('invalid_claim_count', 0)}`"
        ),
        "",
        "| Window | Status | Label text | Support | Conflicts | Abstain |",
        "|---|---|---|---:|---:|---:|",
    ]
    for window in report.get("windows", []):
        annotation = window.get("annotation", {})
        segments = annotation.get("segments", [])
        first = segments[0] if segments else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(window.get("window_id")),
                    str(annotation.get("status", "ABSTAIN")),
                    str(first.get("label_text", "—")),
                    str(first.get("support_count", 0)),
                    str(len(window.get("conflicts", []))),
                    "yes" if window.get("abstention", {}).get("abstained") else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Raw Qwen claims and WeMM Top-K provenance remain in the JSON report. "
            "No model, media, ontology, Mapper, training, gold, or hash operation was performed.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AUTHORITY",
    "BOUNDARY_INVALID_STATUSES",
    "BOUNDARY_VALID_STATUSES",
    "DEFAULT_MERGE_GAP_SECONDS",
    "DEFAULT_MIN_CONSENSUS_CAMERAS",
    "DEFAULT_TOP_K",
    "OFFICIAL_QUALITY_STATUS",
    "PRODUCTION_ANNOTATION_PROJECTION_VERSION",
    "SOURCE_TIMESTAMP_BASIS",
    "STATUS",
    "ProductionAnnotationProjectionError",
    "load_json",
    "project_production_annotations",
    "render_markdown",
]
