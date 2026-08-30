"""Verify recorded structured-annotation evidence without generation.

This adapter is intentionally benchmark-local and label-blind.  It consumes the
existing production structured annotation envelope (or a recorded Qwen
structured sidecar), checks only explicit provenance/shape observations, and
returns a review or abstention diagnostic.  It never opens media, invokes a
model, computes a digest, reads gold, or infers an action from prose or a
candidate list.

The verifier is deliberately *not* a second annotation projection.  A segment
is reviewable only when its explicit positive interval is inside the source
window, its verb and noun are measured, and its evidence is non-empty and
measured.  Filler verbs remain claims, but add a review warning rather than
being rewritten or rejected semantically.  Candidate/Top-K rows are retained
as ``CANDIDATE_ONLY`` context and can never become structured claims.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from robata.benchmark.production_structured_annotation import (
    MODEL_NAMES,
    STRUCTURED_ANNOTATION_ENVELOPE_VERSION,
    ProductionStructuredAnnotationError,
    build_structured_annotation_envelope,
    normalize_structured_annotation_envelope,
)

PRODUCTION_STRUCTURED_EVIDENCE_VERIFIER_VERSION: Final = (
    "robata-production-structured-evidence-verifier-v1"
)
# Short aliases follow the naming used by the other benchmark adapters.
STRUCTURED_EVIDENCE_VERIFIER_VERSION: Final = PRODUCTION_STRUCTURED_EVIDENCE_VERIFIER_VERSION
PRODUCTION_STRUCTURED_EVIDENCE_VERSION: Final = PRODUCTION_STRUCTURED_EVIDENCE_VERIFIER_VERSION
VERSION: Final = PRODUCTION_STRUCTURED_EVIDENCE_VERIFIER_VERSION
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
ARTIFACT_KIND: Final = "production_structured_evidence_verification"
SOURCE_TIMESTAMP_BASIS: Final = "source_absolute_seconds"

_FILLER_VERBS: Final = frozenset(
    {
        "reach",
        "reaches",
        "reaching",
        "move",
        "moves",
        "moving",
        "adjust",
        "adjusts",
        "adjusting",
    }
)
_FIELD_NAMES: Final = ("verb", "noun", "attributes", "location", "hand")
_CANDIDATE_STATE_KEYS: Final = (
    "candidate_state",
    "candidate_status",
    "candidate_measurement_status",
    "top_k_status",
)
_OBSERVED_EMPTY_STATES: Final = frozenset(
    {"OBSERVED_EMPTY", "EMPTY_OBSERVED", "MEASURED_EMPTY", "EXPLICIT_EMPTY"}
)
_OBSERVED_STATES: Final = frozenset({"OBSERVED", "MEASURED", "PRESENT", "AVAILABLE"})
_NOT_MEASURED_STATES: Final = frozenset(
    {"NOT_MEASURED", "ABSENT", "UNAVAILABLE", "NOT_RUN", "NOT_OBSERVED"}
)


def _normalise_pair_token(value: object) -> str:
    """Normalize one candidate/claim token for exact lexical binding.

    This is deliberately limited to case, punctuation, and whitespace.  In
    particular, it does not apply the exploratory ``cloth -> garment`` (or
    any other noun) aliases used by some review diagnostics.  Candidate
    binding must not become a hidden semantic mapper.
    """

    if not isinstance(value, str):
        return ""
    text = value.casefold().replace("_", " ").replace("-", " ")
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text).split())


def _candidate_scalar(value: object) -> object:
    """Unwrap a structured value/status field without interpreting it."""

    if isinstance(value, Mapping) and "value" in value:
        status = _text(value.get("status")).upper()
        if status and status != "MEASURED":
            return None
        return value.get("value")
    return value


def _candidate_pair(value: object, *, depth: int = 0) -> tuple[str, str] | None:
    """Extract an explicit verb/noun pair from one Top-K candidate.

    Candidate objects vary slightly across the recorded WeMM runners.  This
    helper accepts only explicit pair/label fields and never maps synonyms or
    ontology IDs.  A depth bound keeps malformed nested provenance harmless.
    """

    if depth > 4 or not isinstance(value, Mapping):
        return None
    labels = value.get("structured_labels", value.get("labels"))
    if isinstance(labels, Mapping):
        verb = _normalise_pair_token(_candidate_scalar(labels.get("verb")))
        noun = _normalise_pair_token(_candidate_scalar(labels.get("noun")))
        if verb and noun:
            return verb, noun
    verb = _normalise_pair_token(_candidate_scalar(value.get("verb")))
    noun = _normalise_pair_token(_candidate_scalar(value.get("noun")))
    if verb and noun:
        return verb, noun
    explicit_pair = value.get("pair")
    if (
        isinstance(explicit_pair, Sequence)
        and not isinstance(explicit_pair, (str, bytes, bytearray))
        and len(explicit_pair) == 2
    ):
        verb = _normalise_pair_token(_candidate_scalar(explicit_pair[0]))
        noun = _normalise_pair_token(_candidate_scalar(explicit_pair[1]))
        if verb and noun:
            return verb, noun
    for key in ("candidate", "action", "prediction"):
        nested = value.get(key)
        pair = _candidate_pair(nested, depth=depth + 1)
        if pair is not None:
            return pair
    label = value.get("label_text", value.get("label"))
    if isinstance(label, str):
        tokens = _normalise_pair_token(label).split()
        if len(tokens) >= 3 and tokens[:2] == ["pick", "up"]:
            pair = ("pick up", " ".join(tokens[2:]))
        elif len(tokens) >= 2:
            pair = (tokens[0], " ".join(tokens[1:]))
        else:
            pair = None
        if pair is not None and all(pair):
            return pair
    return None


def _available_wemm_pairs(section: Mapping[str, Any] | None) -> set[tuple[str, str]] | None:
    """Return explicit non-empty WeMM Top-K pairs, if such a list exists.

    ``None`` means that no usable Top-K pair is available, preserving the
    legacy Qwen-only behavior.  The canonical ``candidates`` slot is the
    source of truth, including when it is explicitly empty.  Only envelopes
    without that slot can fall back to a ``candidate_groups`` entry explicitly
    marked ``source_field=candidates``.  This prevents an audit-only alternate
    ``predictions`` source from becoming a hidden Top-K candidate set.
    """

    if not isinstance(section, Mapping):
        return None
    values: list[object] = []
    if "candidates" in section:
        raw = section.get("candidates")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            values.extend(raw)
    else:
        # This compatibility fallback is deliberately narrower than the
        # envelope builder's audit retention: a recorded alternate
        # ``predictions`` list is not the canonical Top-K delivered to Qwen.
        groups = section.get("candidate_groups")
        if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes, bytearray)):
            for group in groups:
                if not isinstance(group, Mapping):
                    continue
                if _text(group.get("source_field")) != "candidates":
                    continue
                group_values = group.get("candidates")
                if isinstance(group_values, Sequence) and not isinstance(
                    group_values, (str, bytes, bytearray)
                ):
                    values.extend(group_values)
    pairs = {_candidate_pair(value) for value in values}
    usable = {pair for pair in pairs if pair is not None}
    return usable or None


class StructuredEvidenceVerifierError(ValueError):
    """Raised when an input cannot be used as a structured sidecar."""


# Compatibility spellings make the small adapter easy to discover from phase
# and contract terminology without creating another implementation.
ProductionStructuredEvidenceVerifierError = StructuredEvidenceVerifierError
ProductionStructuredEvidenceVerificationError = StructuredEvidenceVerifierError


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StructuredEvidenceVerifierError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StructuredEvidenceVerifierError(f"{field} must be an array")
    return value


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _copy_json(value: object, *, field: str = "value") -> Any:
    """Deep-copy JSON values without deriving an identity or hash."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StructuredEvidenceVerifierError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise StructuredEvidenceVerifierError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise StructuredEvidenceVerifierError(f"{field} must be JSON-compatible")


def _normalise_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _assert_no_gold(value: object, *, path: str = "document") -> None:
    """Reject official/human labels while allowing model structured fields."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise StructuredEvidenceVerifierError(f"{path} keys must be strings")
            token = _normalise_key(key)
            # Recorded model runners repeat their non-gold disposition at the
            # top level (for example ``official_gold_status=NOT_ESTABLISHED``).
            # These are provenance controls, not labels.  Accept only the
            # narrow non-gold values; any other gold/review-bearing value still
            # fails closed below.
            safe_non_gold_values: dict[str, frozenset[object]] = {
                "officialgoldstatus": frozenset({"NOT_ESTABLISHED", "NOT_MEASURED"}),
                "officialqualitystatus": frozenset({"NOT_MEASURED"}),
                "qualityclaim": frozenset({False}),
                "productioneligible": frozenset({False}),
                "automaticeligible": frozenset({False}),
                "automaticqualification": frozenset({False}),
                "officialgold": frozenset({False}),
                "acceptedasgold": frozenset({False}),
            }
            if (
                token in safe_non_gold_values
                and isinstance(child, (str, bool, int, float, type(None)))
                and child in safe_non_gold_values[token]
            ):
                _assert_no_gold(child, path=f"{path}.{key}")
                continue
            safe_locator_keys = {
                "reviewartifact",
                "decisionartifact",
                "priorprovisionalvocabulary",
                "surfacebundle",
            }
            if token in safe_locator_keys and isinstance(child, str):
                # Locator-only provenance; the verifier never opens the
                # referenced artifact and therefore cannot import labels.
                continue
            safe_expected_metadata = {
                "expectedcount",
                "expectedcameracount",
                "expectedcameracoveragefraction",
                "expectedwindowcount",
                "expectedframecount",
                "expecteddurationseconds",
                "expectedsizebytes",
            }
            if token in {
                "gold",
                "goldstatus",
                "goldlabel",
                "goldlabels",
                "groundtruth",
                "groundtruthlabel",
                "officiallabel",
                "officiallabels",
                "humanlabel",
                "humanlabels",
                "review",
                "adjudication",
            } or (token.startswith("expected") and token not in safe_expected_metadata):
                raise StructuredEvidenceVerifierError(
                    f"{path}.{key} must not contain gold or review data"
                )
            _assert_no_gold(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_gold(child, path=f"{path}[{index}]")
        return
    raise StructuredEvidenceVerifierError(f"{path} must be JSON-compatible")


def _load_input(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise StructuredEvidenceVerifierError(f"could not load structured input: {exc}") from exc
    return _mapping(payload, field="structured_input")


def _source_path(value: Mapping[str, Any]) -> str | None:
    source = value.get("source")
    if isinstance(source, Mapping):
        # Keep this list aligned with the structured-envelope builder.  These
        # are locators only; no media is opened by the verifier.
        for key in (
            "path",
            "media_path",
            "source_path",
            "mcap_path",
            "video_path",
            "manifest",
            "video_root",
        ):
            candidate = source.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    # Some recorded model sidecars keep the locator at the root rather than
    # under ``source``.  Accept the same compatibility spellings as the
    # envelope builder when wrapping a direct sidecar.
    for key in ("source_path", "media_path", "mcap_path", "video_path"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _raw_candidate_markers(value: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """Retain explicit candidate-state markers lost by envelope normalization.

    The canonical envelope intentionally keeps the model's ``candidates``
    array but does not require a Top-K measurement marker.  A raw sidecar may
    opt into ``candidate_state=OBSERVED_EMPTY``; retain that narrow diagnostic
    when normalizing the sidecar.  An empty Qwen list without the marker stays
    ``NOT_MEASURED`` because Qwen emits ``candidates=[]`` by contract.
    """

    result: dict[tuple[str, str], str] = {}
    raw_windows = value.get("windows")
    if not isinstance(raw_windows, Sequence) or isinstance(raw_windows, (str, bytes, bytearray)):
        return result
    declared_format = value.get("format")
    is_envelope = declared_format == STRUCTURED_ANNOTATION_ENVELOPE_VERSION
    for raw_window in raw_windows:
        if not isinstance(raw_window, Mapping):
            continue
        window_id = _text(raw_window.get("window_id"))
        if not window_id:
            continue
        if is_envelope:
            raw_models = raw_window.get("models")
            if not isinstance(raw_models, Mapping):
                continue
            for model in MODEL_NAMES:
                section = raw_models.get(model)
                if not isinstance(section, Mapping):
                    continue
                for key in _CANDIDATE_STATE_KEYS:
                    marker = _text(section.get(key)).upper()
                    if marker:
                        result[(window_id, model)] = marker
                        break
            continue
        # Direct model sidecars have one row per camera/window.  Keep a marker
        # from either the row or its nested model object.  Do not infer Mage
        # as WeMM merely because both are non-Qwen formats: Mage candidate
        # state is model provenance and must remain attached to the Mage slot.
        format_text = str(declared_format).casefold()
        if "qwen" in format_text:
            model = "qwen"
        elif "mage" in format_text:
            model = "mage"
        else:
            model = "wemm"
        nested = raw_window.get("model")
        sources = [raw_window] + ([nested] if isinstance(nested, Mapping) else [])
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            for key in _CANDIDATE_STATE_KEYS:
                marker = _text(source.get(key)).upper()
                if marker:
                    result[(window_id, model)] = marker
                    break
            if (window_id, model) in result:
                break
    return result


def _coerce_envelope(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    """Normalize a canonical envelope or wrap a recorded Qwen sidecar."""

    raw_markers = _raw_candidate_markers(value)
    declared_format = value.get("format")
    try:
        if declared_format == STRUCTURED_ANNOTATION_ENVELOPE_VERSION:
            envelope = normalize_structured_annotation_envelope(value)
            return envelope, raw_markers

        model_key = {
            "robata-production-qwen-structured-native-shadow-v1": "qwen",
            "robata-production-qwen-shadow-v1": "qwen",
            "robata-production-wemm-shadow-v1": "wemm",
            "robata-production-wemm-vocabulary-shadow-v1": "wemm",
            "robata-production-mage-shadow-v1": "mage",
            "robata-production-mage-structured-native-shadow-v1": "mage",
        }.get(str(declared_format))
        if model_key is None:
            raise StructuredEvidenceVerifierError(
                "input must be a structured annotation envelope or supported model sidecar"
            )
        source_path = _source_path(value)
        if source_path is None:
            raise StructuredEvidenceVerifierError("model sidecar source path is required")
        source = value.get("source")
        camera_count: int | None = None
        if isinstance(source, Mapping):
            raw_count = source.get("camera_count")
            if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count >= 0:
                camera_count = raw_count
        envelope = build_structured_annotation_envelope(
            {model_key: value}, source_path=source_path, camera_count=camera_count
        )
        return envelope, raw_markers
    except ProductionStructuredAnnotationError as exc:
        raise StructuredEvidenceVerifierError(str(exc)) from exc


def _field_value(segment: Mapping[str, Any], field: str) -> tuple[Any, str, bool]:
    labels = segment.get("structured_labels", segment.get("labels"))
    source: Mapping[str, Any] = labels if isinstance(labels, Mapping) else segment
    if field not in source:
        return None, "NOT_MEASURED", False
    raw = source.get(field)
    if isinstance(raw, Mapping):
        status = _text(raw.get("status")).upper() or (
            "NOT_OBSERVABLE" if raw.get("value") is None else "MEASURED"
        )
        return raw.get("value"), status, True
    if raw is None:
        return None, "NOT_OBSERVABLE", True
    return raw, "MEASURED", True


def _measured_text(value: object, status: str) -> bool:
    return status == "MEASURED" and isinstance(value, str) and bool(value.strip())


def _evidence(segment: Mapping[str, Any]) -> tuple[list[Any], str, bool]:
    present = "evidence" in segment
    raw = segment.get("evidence")
    status = _text(segment.get("evidence_status")).upper()
    values: list[Any]
    if isinstance(raw, str):
        values = [raw] if raw.strip() else []
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = [item for item in raw if isinstance(item, str) and item.strip()]
    else:
        values = []
    if not status:
        status = "MEASURED" if values else "NOT_MEASURED" if not present else "NOT_OBSERVABLE"
    return values, status, present


def _interval(value: Mapping[str, Any]) -> tuple[float, float] | None:
    start = _finite(value.get("start_time_sec"))
    end = _finite(value.get("end_time_sec"))
    if start is None or end is None or start < 0 or end <= start:
        return None
    return start, end


def _window_interval(window: Mapping[str, Any]) -> tuple[float, float] | None:
    start = _finite(window.get("start_time_sec"))
    end = _finite(window.get("end_time_sec"))
    if start is None or end is None or start < 0 or end <= start:
        return None
    return start, end


def _candidate_state(
    section: Mapping[str, Any], *, model: str, explicit_marker: str | None = None
) -> tuple[str, str, list[Any]]:
    raw_candidates = section.get("candidates", [])
    candidates = (
        list(raw_candidates)
        if isinstance(raw_candidates, Sequence)
        and not isinstance(raw_candidates, (str, bytes, bytearray))
        else []
    )
    # ``production_structured_annotation`` may retain a native Qwen ``raw_text``
    # object in ``candidate_groups`` for provenance when no named Top-K list was
    # supplied.  That object is a structured response, not a candidate ranking;
    # never turn it into a candidate-only claim.
    groups = section.get("candidate_groups", [])
    if (
        model == "qwen"
        and isinstance(groups, Sequence)
        and not isinstance(groups, (str, bytes, bytearray))
        and groups
        and all(
            isinstance(group, Mapping) and _text(group.get("source_field")) == "raw_text"
            for group in groups
        )
    ):
        candidates = []
    if candidates:
        return "OBSERVED", "non-empty model candidate list", candidates

    marker = _text(explicit_marker).upper()
    if not marker:
        for key in _CANDIDATE_STATE_KEYS:
            marker = _text(section.get(key)).upper()
            if marker:
                break
    if marker in _OBSERVED_EMPTY_STATES:
        return "OBSERVED_EMPTY", "explicit empty ranking marker", candidates
    if marker in _OBSERVED_STATES:
        return "OBSERVED_EMPTY", "explicit observed ranking has no entries", candidates
    # In particular, Qwen's native structured sidecar intentionally emits an
    # empty ``candidates`` array.  It is an absent Top-K measurement, not an
    # empty semantic ranking and never a source for inferred claims.
    return (
        "NOT_MEASURED",
        ("QWEN_TOP_K_ABSENT_BY_CONTRACT" if model == "qwen" else "candidate ranking not measured"),
        candidates,
    )


def _parse_diagnostics(section: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    def diagnostic_values(value: object) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    observations = section.get("parse_observations", [])
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes, bytearray)):
        observations = []
    for index, raw in enumerate(observations):
        if not isinstance(raw, Mapping):
            continue
        parse_status = _text(raw.get("parse_status")).upper() or "UNKNOWN"
        errors = diagnostic_values(raw.get("errors", []))
        warnings = diagnostic_values(raw.get("warnings", []))
        generation_warnings = diagnostic_values(raw.get("generation_warnings", []))
        row_reasons: list[str] = []
        if parse_status != "PARSED":
            row_reasons.append("PARSE_INVALID")
            reasons.append("PARSE_INVALID")
        for error in errors:
            code = f"PARSE_ERROR:{error}"
            row_reasons.append(code)
            reasons.append(code)
        for warning in [*warnings, *generation_warnings]:
            code = f"PARSE_WARNING:{warning}"
            row_reasons.append(code)
            reasons.append(code)
        mapping_status = _text(
            raw.get("timestamp_mapping_status", raw.get("mapping_status"))
        ).upper()
        if mapping_status in {"FAILED", "ERROR", "INVALID"}:
            row_reasons.append("TIMESTAMP_MAPPING_FAILED")
            reasons.append("TIMESTAMP_MAPPING_FAILED")
        elif mapping_status == "UNSUPPORTED":
            row_reasons.append("TIMESTAMP_MAPPING_UNSUPPORTED")
            reasons.append("TIMESTAMP_MAPPING_UNSUPPORTED")
        mapping_error = _text(raw.get("timestamp_mapping_error", raw.get("mapping_error")))
        if mapping_error:
            code = (
                "TIMESTAMP_MAPPING_FAILED"
                if "TIMESTAMP_MAPPING" in mapping_error.upper()
                else f"PARSE_ERROR:{mapping_error}"
            )
            row_reasons.append(code)
            reasons.append(code)
        rows.append(
            {
                "index": index,
                "camera_id": raw.get("camera_id"),
                "parse_status": parse_status,
                "errors": errors,
                "warnings": warnings,
                "generation_warnings": generation_warnings,
                "timestamp_mapping_status": mapping_status or None,
                "timestamp_mapping_error": mapping_error or None,
                "reason_codes": list(dict.fromkeys(row_reasons)),
                "raw": _copy_json(raw, field=f"parse_observations[{index}]"),
            }
        )
    # Section-level warnings are retained separately; they can include filler
    # diagnostics generated by the canonical parser.
    section_warnings = section.get("warnings", [])
    if isinstance(section_warnings, Sequence) and not isinstance(
        section_warnings, (str, bytes, bytearray)
    ):
        for warning in section_warnings:
            if isinstance(warning, str) and warning.strip():
                reasons.append(f"PARSE_WARNING:{warning.strip()}")
    return rows, list(dict.fromkeys(reasons))


def _claim_check(
    *,
    window: Mapping[str, Any],
    section: Mapping[str, Any],
    segment: Mapping[str, Any],
    segment_index: int,
    model: str = "qwen",
    available_wemm_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    window_id = _text(window.get("window_id")) or "__unknown__"
    reasons: list[str] = []
    review_reasons: list[str] = []
    source_window = _window_interval(window)
    segment_interval = _interval(segment)
    boundary_status = _text(segment.get("boundary_status")).upper()
    has_segment_basis = "timestamp_basis" in segment
    timestamp_basis = _text(segment.get("timestamp_basis")).strip()
    if not has_segment_basis:
        timestamp_basis = _text(window.get("timestamp_basis"))
    if timestamp_basis == SOURCE_TIMESTAMP_BASIS:
        timestamp_basis_status = "MEASURED"
    elif has_segment_basis and not timestamp_basis:
        timestamp_basis_status = "INVALID"
        reasons.append("TIMESTAMP_BASIS_INVALID")
    elif not timestamp_basis:
        timestamp_basis_status = "NOT_MEASURED"
    else:
        timestamp_basis_status = "UNSUPPORTED"
    if timestamp_basis_status == "UNSUPPORTED":
        reasons.append("TIMESTAMP_BASIS_UNSUPPORTED")
    mapping_status = _text(segment.get("timestamp_mapping_status")).upper()
    if mapping_status in {"FAILED", "ERROR", "INVALID"}:
        reasons.append("TIMESTAMP_MAPPING_FAILED")
    elif mapping_status == "UNSUPPORTED":
        reasons.append("TIMESTAMP_MAPPING_UNSUPPORTED")
    if segment_interval is None:
        reasons.append("BOUNDARY_MISSING_OR_INVALID")
    elif source_window is None:
        reasons.append("SOURCE_INTERVAL_MISSING")
    elif not (
        segment_interval[0] >= source_window[0]
        and segment_interval[1] <= source_window[1]
        and segment_interval[1] > segment_interval[0]
    ):
        reasons.append("BOUNDARY_OUT_OF_SOURCE")
    if boundary_status != "MEASURED":
        reasons.append("BOUNDARY_NOT_SOURCE_MEASURED")
    # The structured-envelope normalizer intentionally nulls an explicit
    # out-of-window boundary while retaining this diagnostic marker.  Keep the
    # more useful source-bound reason in the verifier report rather than
    # collapsing every such row into a generic missing-boundary failure.
    if _text(segment.get("boundary_error")).upper() in {
        "SEGMENT_BOUNDARY_OUTSIDE_WINDOW",
        "BOUNDARY_OUT_OF_SOURCE",
    }:
        reasons.append("BOUNDARY_OUT_OF_SOURCE")
    source_bound_positive_interval = not any(
        code in reasons
        for code in (
            "BOUNDARY_MISSING_OR_INVALID",
            "SOURCE_INTERVAL_MISSING",
            "BOUNDARY_OUT_OF_SOURCE",
            "BOUNDARY_NOT_SOURCE_MEASURED",
            "TIMESTAMP_BASIS_INVALID",
            "TIMESTAMP_BASIS_UNSUPPORTED",
            "TIMESTAMP_MAPPING_FAILED",
            "TIMESTAMP_MAPPING_UNSUPPORTED",
        )
    )

    verb, verb_status, verb_present = _field_value(segment, "verb")
    noun, noun_status, noun_present = _field_value(segment, "noun")
    verb_measured = _measured_text(verb, verb_status)
    noun_measured = _measured_text(noun, noun_status)
    if not verb_measured:
        reasons.append("VERB_NOT_MEASURED")
    if not noun_measured:
        reasons.append("NOUN_NOT_MEASURED")

    # When the production WeMM route supplied explicit Top-K pairs, a
    # structured Qwen/Mage claim must echo one of those pairs.  This is a
    # lexical binding gate only: it deliberately does not normalize synonyms
    # (for example ``cloth`` to ``garment``) or consult an ontology.  If no
    # usable Top-K pair exists, retain the historical Qwen-only behavior and
    # leave binding unmeasured.
    claim_pair = (
        (
            _normalise_pair_token(verb),
            _normalise_pair_token(noun),
        )
        if verb_measured and noun_measured
        else None
    )
    candidate_binding_status = "NOT_MEASURED"
    if available_wemm_pairs and claim_pair is not None:
        if claim_pair not in available_wemm_pairs:
            reasons.append("CLAIM_PAIR_NOT_IN_WEMM_TOP_K")
            candidate_binding_status = "MISMATCH"
        else:
            candidate_binding_status = "MATCH"

    evidence_values, evidence_status, evidence_present = _evidence(segment)
    evidence_present_check = bool(evidence_values)
    if not evidence_present_check:
        reasons.append("EVIDENCE_MISSING")
    elif evidence_status != "MEASURED":
        reasons.append("EVIDENCE_NOT_MEASURED")

    filler_code: str | None = None
    if isinstance(verb, str) and verb.strip().casefold() in _FILLER_VERBS:
        filler_code = f"FILLER_VERB_PRESENT:{verb.strip().casefold()}"
        review_reasons.append(filler_code)

    segment_status = _text(segment.get("status")).upper() or "NOT_MEASURED"
    if segment_status in {"FAILED", "BLOCKED", "NOT_MEASURED", "NOT_OBSERVABLE"}:
        reasons.append(f"STRUCTURED_CLAIM_STATUS_{segment_status}")
    runtime_status = _text(section.get("status")).upper() or "NOT_RUN"
    if runtime_status != "SUCCEEDED":
        reasons.append(f"MODEL_RUNTIME_{runtime_status}")

    reasons = list(dict.fromkeys(reasons))
    structurally_reviewable = not reasons
    if structurally_reviewable:
        disposition = "REVIEW"
        review_reasons.insert(0, "INDEPENDENT_REVIEW_REQUIRED")
    else:
        disposition = "ABSTAIN"
    # Keep the model in the local claim key.  Qwen and Mage can both emit a
    # segment at index zero for the same bounded window; omitting the model
    # would make their provenance collide in a downstream review queue.
    claim_id = f"{window_id}:{model}:{segment_index}"
    return {
        "claim_id": claim_id,
        "claim_kind": "STRUCTURED",
        "claim_type": "structured",
        "model": model,
        "segment_index": segment_index,
        "runtime_status": runtime_status,
        "segment_status": segment_status,
        "disposition": disposition,
        "claim_status": disposition,
        "accepted": False,
        "eligible_for_review": structurally_reviewable,
        "review_required": structurally_reviewable,
        "candidate_only": False,
        "structured_claim": True,
        "semantic_status": "NOT_CHECKED",
        "source_bound_positive_interval": source_bound_positive_interval,
        "source_interval": list(source_window) if source_window else None,
        "interval": list(segment_interval) if segment_interval else None,
        "boundary_status": boundary_status or None,
        "timestamp_basis": timestamp_basis or None,
        "timestamp_basis_status": timestamp_basis_status,
        "verb": verb,
        "verb_status": verb_status,
        "verb_present": verb_present,
        "verb_measured": verb_measured,
        "noun": noun,
        "noun_status": noun_status,
        "noun_present": noun_present,
        "noun_measured": noun_measured,
        "claim_pair": list(claim_pair) if claim_pair is not None else None,
        "wemm_top_k_binding": candidate_binding_status,
        "evidence": _copy_json(evidence_values, field=f"{claim_id}.evidence"),
        "evidence_status": evidence_status,
        "evidence_present": evidence_present,
        "evidence_presence": evidence_present_check,
        "filler_verb": filler_code is not None,
        "filler_verb_warning": filler_code,
        "reason_codes": reasons,
        "reasons": reasons,
        "review_reason_codes": list(dict.fromkeys(review_reasons)),
        "checks": {
            "source_bound_positive_interval": source_bound_positive_interval,
            "measured_verb": verb_measured,
            "measured_noun": noun_measured,
            "evidence_present": evidence_present_check,
            "filler_verb_warning": filler_code is not None,
        },
        "raw_claim": _copy_json(segment, field=f"{claim_id}.raw_claim"),
    }


def _candidate_claims(
    *,
    window_id: str,
    section: Mapping[str, Any],
    candidates: Sequence[Any],
    model: str = "qwen",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runtime_status = _text(section.get("status")).upper() or "NOT_RUN"
    for index, candidate in enumerate(candidates):
        rows.append(
            {
                "claim_id": f"{window_id}:{model}:candidate:{index}",
                "claim_kind": "CANDIDATE_ONLY",
                "claim_type": "candidate_only",
                "model": model,
                "candidate_index": index,
                "runtime_status": runtime_status,
                "disposition": "ABSTAIN",
                "claim_status": "ABSTAIN",
                "accepted": False,
                "eligible_for_review": False,
                "review_required": True,
                "semantic_status": "NOT_CHECKED",
                "candidate_only": True,
                "structured_claim": False,
                "reason_codes": ["CANDIDATE_ONLY_NOT_STRUCTURED"],
                "reasons": ["CANDIDATE_ONLY_NOT_STRUCTURED"],
                "review_reason_codes": ["CANDIDATE_ONLY_CONTEXT_ONLY"],
                "candidate": _copy_json(candidate, field=f"{window_id}.candidate[{index}]"),
            }
        )
    return rows


def _model_claim_report(
    *,
    window: Mapping[str, Any],
    section: Mapping[str, Any],
    model: str,
    raw_marker: str | None,
    available_wemm_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Check one model's structured claims with one shared structural gate.

    Qwen is the historical primary route, while Mage can provide the same
    source-bound segment shape after native semantic replay.  Parameterizing
    the checker keeps boundary/field/evidence policy identical and preserves
    model-specific provenance for the review queue.
    """

    window_id = _text(window.get("window_id")) or "__unknown__"
    raw_segments = section.get("segments", [])
    segments = _sequence(raw_segments, field=f"window[{window_id}].models.{model}.segments")
    structured_claims: list[dict[str, Any]] = []
    for index, raw_segment in enumerate(segments):
        segment = _mapping(
            raw_segment,
            field=f"window[{window_id}].models.{model}.segments[{index}]",
        )
        structured_claims.append(
            _claim_check(
                window=window,
                section=section,
                segment=segment,
                segment_index=index,
                model=model,
                available_wemm_pairs=available_wemm_pairs,
            )
        )

    candidate_state, candidate_state_reason, candidates = _candidate_state(
        section, model=model, explicit_marker=raw_marker
    )
    candidate_claims = _candidate_claims(
        window_id=window_id,
        section=section,
        candidates=candidates,
        model=model,
    )
    parse_diagnostics, parse_reasons = _parse_diagnostics(section)
    reasons: list[str] = list(parse_reasons)
    for claim in structured_claims:
        reasons.extend(claim["reason_codes"])
        reasons.extend(claim["review_reason_codes"])
    if candidate_claims:
        reasons.append("CANDIDATE_ONLY_CONTEXT_ONLY")
    elif candidate_state == "OBSERVED_EMPTY":
        reasons.append("OBSERVED_EMPTY_CANDIDATE_RANKING")

    runtime_status = _text(section.get("status")).upper() or "NOT_RUN"
    valid_claims = [claim for claim in structured_claims if claim["eligible_for_review"]]
    invalid_claims = [claim for claim in structured_claims if not claim["eligible_for_review"]]
    if runtime_status != "SUCCEEDED":
        reasons.append(f"MODEL_RUNTIME_{runtime_status}")
    if not structured_claims:
        reasons.append("NO_STRUCTURED_CLAIMS")
        if candidate_claims:
            reasons.append("CANDIDATE_ONLY_NO_STRUCTURED_CLAIM")
    elif not valid_claims:
        reasons.append("NO_SOURCE_BOUND_STRUCTURED_CLAIMS")
    if valid_claims:
        decision = "review"
        status = "REVIEW_REQUIRED"
        abstained = False
        review_reason_codes = ["INDEPENDENT_REVIEW_REQUIRED"]
        review_reason_codes.extend(
            code for claim in valid_claims for code in claim["review_reason_codes"]
        )
    else:
        decision = "abstain"
        status = "ABSTAIN"
        abstained = True
        review_reason_codes = []

    if runtime_status == "SUCCEEDED":
        if valid_claims:
            semantic_status = "STRUCTURED_REVIEWABLE"
        elif structured_claims or "PARSE_INVALID" in reasons:
            semantic_status = "STRUCTURED_INVALID"
        else:
            semantic_status = "NO_STRUCTURED_CLAIMS"
    else:
        semantic_status = f"MODEL_RUNTIME_{runtime_status}"
    return {
        "model": model,
        "runtime_status": runtime_status,
        "runtime_classification": f"MODEL_{runtime_status}",
        "structured_status": semantic_status,
        "semantic_status": semantic_status,
        "semantic_classification": semantic_status,
        "status": status,
        "decision": decision,
        "abstained": abstained,
        "structured_claim_count": len(structured_claims),
        "reviewable_claim_count": len(valid_claims),
        "invalid_claim_count": len(invalid_claims),
        "candidate_only_claim_count": len(candidate_claims),
        "candidate_state": candidate_state,
        "candidate_state_reason": candidate_state_reason,
        "candidate_mode": candidate_state.casefold(),
        "candidate_top_k_observed": candidate_state == "OBSERVED",
        "candidate_top_k_measured": candidate_state in {"OBSERVED", "OBSERVED_EMPTY"},
        "candidate_only": bool(candidate_claims),
        "wemm_top_k_pairs": (
            [list(pair) for pair in sorted(available_wemm_pairs)] if available_wemm_pairs else []
        ),
        "wemm_candidate_binding_checked": bool(available_wemm_pairs),
        "claims": [*structured_claims, *candidate_claims],
        "structured_claims": structured_claims,
        "candidate_only_claims": candidate_claims,
        "parse_diagnostics": parse_diagnostics,
        "reason_codes": list(dict.fromkeys(reasons)),
        "reasons": list(dict.fromkeys(reasons)),
        "review_reason_codes": list(dict.fromkeys(review_reason_codes)),
        "abstention": {
            "abstained": abstained,
            "decision": decision,
            "reason_codes": list(dict.fromkeys(reasons)) if abstained else [],
        },
        "gold_status": "EXTERNAL_NOT_READ",
        "semantic_inference_performed": False,
    }


def _window_report(
    window: Mapping[str, Any], *, raw_markers: Mapping[tuple[str, str], str]
) -> dict[str, Any]:
    window_id = _text(window.get("window_id"))
    if not window_id:
        raise StructuredEvidenceVerifierError("envelope window_id must be non-empty")
    source_interval = _window_interval(window)
    models = _mapping(window.get("models"), field=f"window[{window_id}].models")

    # WeMM is retrieval-only, but its explicit Top-K is the candidate set that
    # the selective Qwen/Mage verifier is allowed to support.  A missing or
    # opaque list yields ``None`` so legacy Qwen-only fixtures remain valid.
    wemm_section = models.get("wemm")
    wemm_top_k_pairs = _available_wemm_pairs(
        wemm_section if isinstance(wemm_section, Mapping) else None
    )

    # Keep the Qwen-primary fields for compatibility with existing reports.
    qwen = _mapping(models.get("qwen"), field=f"window[{window_id}].models.qwen")
    qwen_report = _model_claim_report(
        window=window,
        section=qwen,
        model="qwen",
        raw_marker=raw_markers.get((window_id, "qwen")),
        available_wemm_pairs=wemm_top_k_pairs,
    )
    model_reports: dict[str, dict[str, Any]] = {"qwen": qwen_report}

    # Mage normally remains BLOCKED until source-bound native semantic replay is
    # available.  If a structured Mage section is recorded, check it through
    # the same non-generative gate and expose its claims to the handoff.
    mage = models.get("mage")
    if isinstance(mage, Mapping):
        model_reports["mage"] = _model_claim_report(
            window=window,
            section=mage,
            model="mage",
            raw_marker=raw_markers.get((window_id, "mage")),
            available_wemm_pairs=wemm_top_k_pairs,
        )

    structured_claims_all = [
        claim for report in model_reports.values() for claim in report["structured_claims"]
    ]
    valid_claims_all = [claim for claim in structured_claims_all if claim["eligible_for_review"]]
    invalid_claims_all = [
        claim for claim in structured_claims_all if not claim["eligible_for_review"]
    ]
    candidate_claims_all = [
        claim for report in model_reports.values() for claim in report["candidate_only_claims"]
    ]

    # Avoid changing old Qwen-only reason lists merely because Mage is
    # explicitly BLOCKED.  A real Mage claim/parser observation is always
    # surfaced in the aggregate diagnostics.
    reasons: list[str] = list(qwen_report["reason_codes"])
    for model, report in model_reports.items():
        if model == "qwen":
            continue
        if (
            report["structured_claims"]
            or report["parse_diagnostics"]
            or report["candidate_only_claims"]
        ):
            reasons.extend(report["reason_codes"])
    reasons = list(dict.fromkeys(reasons))

    valid_any = bool(valid_claims_all)
    status = "REVIEW_REQUIRED" if valid_any else "ABSTAIN"
    decision = "review" if valid_any else "abstain"
    abstained = not valid_any
    review_reason_codes = (
        ["INDEPENDENT_REVIEW_REQUIRED"]
        + [code for claim in valid_claims_all for code in claim["review_reason_codes"]]
        if valid_any
        else []
    )
    if valid_any:
        semantic_status = "STRUCTURED_REVIEWABLE"
    elif structured_claims_all or "PARSE_INVALID" in reasons:
        semantic_status = "STRUCTURED_INVALID"
    else:
        semantic_status = qwen_report["semantic_status"]

    # WeMM is retrieval-only in this verifier.  Preserve its Top-K verbatim so
    # the reviewer can use it as context, but never turn it into a claim.
    wemm = models.get("wemm")
    if isinstance(wemm, Mapping):
        raw_candidates = wemm.get("candidates", [])
        candidates = (
            _copy_json(
                raw_candidates,
                field=f"window[{window_id}].models.wemm.candidates",
            )
            if isinstance(raw_candidates, Sequence)
            and not isinstance(raw_candidates, (str, bytes, bytearray))
            else []
        )
        retrieval_context = {
            "model": "wemm",
            "status": _text(wemm.get("status")).upper() or "NOT_RUN",
            "measurement_status": _text(wemm.get("measurement_status")).upper() or "NOT_MEASURED",
            "top_k": candidates,
            "top_k_is_context_only": True,
            "candidate_sources": _copy_json(
                wemm.get("candidate_sources", []),
                field=f"window[{window_id}].models.wemm.candidate_sources",
            ),
        }
    else:
        retrieval_context = {
            "model": "wemm",
            "status": "NOT_RUN",
            "measurement_status": "NOT_MEASURED",
            "top_k": [],
            "top_k_is_context_only": True,
            "candidate_sources": [],
        }

    return {
        "ordinal": window.get("ordinal"),
        "window_id": window_id,
        "source_interval": list(source_interval) if source_interval is not None else None,
        # Qwen-primary compatibility fields.
        "runtime_status": qwen_report["runtime_status"],
        "runtime_classification": qwen_report["runtime_classification"],
        "structured_status": semantic_status,
        "semantic_status": semantic_status,
        "semantic_classification": semantic_status,
        "status": status,
        "decision": decision,
        "abstained": abstained,
        "structured_claim_count": qwen_report["structured_claim_count"],
        "reviewable_claim_count": qwen_report["reviewable_claim_count"],
        "invalid_claim_count": qwen_report["invalid_claim_count"],
        "candidate_only_claim_count": qwen_report["candidate_only_claim_count"],
        "candidate_state": qwen_report["candidate_state"],
        "candidate_state_reason": qwen_report["candidate_state_reason"],
        "candidate_mode": qwen_report["candidate_mode"],
        "candidate_top_k_observed": qwen_report["candidate_top_k_observed"],
        "candidate_top_k_measured": qwen_report["candidate_top_k_measured"],
        "candidate_only": qwen_report["candidate_only"],
        "claims": qwen_report["claims"],
        "structured_claims": qwen_report["structured_claims"],
        "candidate_only_claims": qwen_report["candidate_only_claims"],
        "parse_diagnostics": qwen_report["parse_diagnostics"],
        # Multi-model additions consumed by the review handoff.
        "structured_claims_all": structured_claims_all,
        "candidate_only_claims_all": candidate_claims_all,
        "structured_claim_count_all": len(structured_claims_all),
        "reviewable_claim_count_all": len(valid_claims_all),
        "invalid_claim_count_all": len(invalid_claims_all),
        "candidate_only_claim_count_all": len(candidate_claims_all),
        "models": model_reports,
        "parse_diagnostics_by_model": {
            model: report["parse_diagnostics"] for model, report in model_reports.items()
        },
        "retrieval_context": retrieval_context,
        "wemm_top_k_pairs": (
            [list(pair) for pair in sorted(wemm_top_k_pairs)] if wemm_top_k_pairs else []
        ),
        "wemm_candidate_binding_checked": bool(wemm_top_k_pairs),
        "reason_codes": reasons,
        "reasons": reasons,
        "review_reason_codes": list(dict.fromkeys(review_reason_codes)),
        "abstention": {
            "abstained": abstained,
            "decision": decision,
            "reason_codes": reasons if abstained else [],
        },
        "gold_status": "EXTERNAL_NOT_READ",
        "semantic_inference_performed": False,
    }


def verify_production_structured_evidence(
    value: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Verify a recorded envelope and return per-claim/window diagnostics."""

    payload = _load_input(value)
    _assert_no_gold(payload)
    envelope, raw_markers = _coerce_envelope(payload)
    _assert_no_gold(envelope)
    windows = _sequence(envelope.get("windows"), field="envelope.windows")
    reports = [
        _window_report(
            _mapping(raw_window, field=f"envelope.windows[{index}]"),
            raw_markers=raw_markers,
        )
        for index, raw_window in enumerate(windows)
    ]
    status = (
        "REVIEW_REQUIRED"
        if any(row["status"] == "REVIEW_REQUIRED" for row in reports)
        else "ABSTAIN"
    )
    metrics = {
        "window_count": len(reports),
        "review_required_window_count": sum(row["status"] == "REVIEW_REQUIRED" for row in reports),
        "abstained_window_count": sum(row["abstained"] for row in reports),
        "structured_claim_count": sum(row["structured_claim_count"] for row in reports),
        "reviewable_claim_count": sum(row["reviewable_claim_count"] for row in reports),
        "invalid_claim_count": sum(row["invalid_claim_count"] for row in reports),
        "candidate_only_claim_count": sum(row["candidate_only_claim_count"] for row in reports),
        "structured_claim_count_all": sum(row["structured_claim_count_all"] for row in reports),
        "reviewable_claim_count_all": sum(row["reviewable_claim_count_all"] for row in reports),
        "invalid_claim_count_all": sum(row["invalid_claim_count_all"] for row in reports),
        "candidate_only_claim_count_all": sum(
            row["candidate_only_claim_count_all"] for row in reports
        ),
        "model_counts": {
            model: {
                "structured_claim_count": sum(
                    report["structured_claim_count"]
                    for row in reports
                    for report in row.get("models", {}).values()
                    if report.get("model") == model
                ),
                "reviewable_claim_count": sum(
                    report["reviewable_claim_count"]
                    for row in reports
                    for report in row.get("models", {}).values()
                    if report.get("model") == model
                ),
                "invalid_claim_count": sum(
                    report["invalid_claim_count"]
                    for row in reports
                    for report in row.get("models", {}).values()
                    if report.get("model") == model
                ),
            }
            for model in ("qwen", "mage")
        },
        "filler_verb_warning_count": sum(
            claim.get("filler_verb_warning") is not None
            for row in reports
            for claim in row["structured_claims"]
        ),
        "candidate_state_counts": {
            state: sum(row["candidate_state"] == state for row in reports)
            for state in ("OBSERVED", "OBSERVED_EMPTY", "NOT_MEASURED")
        },
        "parse_invalid_window_count": sum(
            "PARSE_INVALID" in row["reason_codes"] for row in reports
        ),
    }
    source = _copy_json(envelope.get("source", {}), field="envelope.source")
    quality = {
        "measurement_status": "NOT_MEASURED",
        "quality_claim": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "production_eligible": False,
    }
    return {
        "format": PRODUCTION_STRUCTURED_EVIDENCE_VERIFIER_VERSION,
        "verifier_version": PRODUCTION_STRUCTURED_EVIDENCE_VERIFIER_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "authority": AUTHORITY,
        "valid": True,
        "status": status,
        "production_eligible": False,
        "quality": quality,
        "source": source,
        "windows": reports,
        "metrics": metrics,
        "contract": {
            "input_format": STRUCTURED_ANNOTATION_ENVELOPE_VERSION,
            "model_checked": "qwen",
            "structured_models_checked": ["qwen", "mage"],
            "mage_claims_checked_when_present": True,
            "wemm_retrieval_context_only": True,
            "candidate_claims_are_model_only": True,
            "empty_qwen_candidates_are_not_inferred": True,
            "structured_claims_bound_to_wemm_top_k_when_available": True,
            "fixed_window_is_not_action_boundary": True,
            "gold_is_external": True,
            "semantic_action_identity_checked": False,
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
            "sha_or_digest_computed": False,
        },
        "semantic_inference_performed": False,
        "gold_promoted": False,
        "errors": [],
        "limitations": [
            "Checks are structural/provenance diagnostics, not semantic action verification.",
            (
                "A filler verb is retained and only triggers independent review; "
                "it is never rewritten."
            ),
            (
                "Qwen candidates=[] is treated as NOT_MEASURED unless an explicit "
                "observed-empty marker exists."
            ),
            "No claim is accepted as gold or production annotation.",
        ],
    }


# Readable aliases used by nearby benchmark phases.
verify_structured_annotation_evidence = verify_production_structured_evidence
verify_structured_evidence = verify_production_structured_evidence
build_structured_evidence_verification = verify_production_structured_evidence
build_production_structured_evidence_report = verify_production_structured_evidence
verify_qwen_structured_evidence = verify_production_structured_evidence
verify_production_structured_annotation_evidence = verify_production_structured_evidence


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable diagnostic without adding semantics."""

    metrics = report.get("metrics", {})
    lines = [
        "# Production structured evidence verification",
        "",
        f"- Status: `{report.get('status', 'ABSTAIN')}`",
        (
            f"- Windows: `{metrics.get('window_count', 0)}`; "
            f"review-required: `{metrics.get('review_required_window_count', 0)}`; "
            f"abstained: `{metrics.get('abstained_window_count', 0)}`"
        ),
        (
            f"- Structured claims: `{metrics.get('structured_claim_count', 0)}`; "
            f"reviewable: `{metrics.get('reviewable_claim_count', 0)}`; "
            f"invalid: `{metrics.get('invalid_claim_count', 0)}`"
        ),
        (
            f"- Candidate-only claims: `{metrics.get('candidate_only_claim_count', 0)}`; "
            f"filler warnings: `{metrics.get('filler_verb_warning_count', 0)}`"
        ),
        "",
        "| Window | Runtime | Structured | Reviewable | Candidate state | Decision | Reasons |",
        "|---|---|---:|---:|---|---|---|",
    ]
    windows = report.get("windows", [])
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes, bytearray)):
        for row in windows:
            if not isinstance(row, Mapping):
                continue
            reasons = ", ".join(str(item) for item in row.get("reason_codes", []))
            lines.append(
                "| "
                f"{row.get('window_id', '')} | {row.get('runtime_status', '')} | "
                f"{row.get('structured_claim_count', 0)} | "
                f"{row.get('reviewable_claim_count', 0)} | "
                f"{row.get('candidate_state', '')} | {row.get('decision', '')} | "
                f"{reasons or '-'} |"
            )
    lines.extend(
        [
            "",
            (
                "This is a label-blind, non-generative diagnostic. No "
                "model/media/gold/hash operation was performed; no claim is promoted."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: str | Path) -> Mapping[str, Any]:
    return _load_input(Path(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structured_input", type=Path, help="structured envelope or Qwen sidecar JSON"
    )
    parser.add_argument("--output", type=Path, required=True, help="verification report JSON path")
    parser.add_argument("--output-md", type=Path, help="optional Markdown report path")
    args = parser.parse_args(argv)
    try:
        report = verify_production_structured_evidence(_load_json(args.structured_input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, UnicodeError, StructuredEvidenceVerifierError, ValueError) as exc:
        print(
            f"production structured evidence verification failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "windows": report["metrics"]["window_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


__all__ = [
    "ARTIFACT_KIND",
    "AUTHORITY",
    "PRODUCTION_STRUCTURED_EVIDENCE_VERIFIER_VERSION",
    "PRODUCTION_STRUCTURED_EVIDENCE_VERSION",
    "STRUCTURED_EVIDENCE_VERIFIER_VERSION",
    "VERSION",
    "ProductionStructuredEvidenceVerificationError",
    "ProductionStructuredEvidenceVerifierError",
    "StructuredEvidenceVerifierError",
    "build_production_structured_evidence_report",
    "build_structured_evidence_verification",
    "main",
    "render_markdown",
    "verify_production_structured_annotation_evidence",
    "verify_production_structured_evidence",
    "verify_qwen_structured_evidence",
    "verify_structured_annotation_evidence",
    "verify_structured_evidence",
]


if __name__ == "__main__":
    raise SystemExit(main())
