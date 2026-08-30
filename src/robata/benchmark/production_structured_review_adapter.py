"""Build a provisional review draft from the structured model sidecar.

This adapter is deliberately separate from the human review pack and from the
production Mapper.  It consumes the benchmark-local structured annotation
envelope (or a single model sidecar that can be wrapped into one), retains each
model's Top-K list verbatim, and exposes explicit structured claims for a later
reviewer.  A fixed source window is recorded as geometry only; it is never
promoted to an action boundary.

No model is called, no media is decoded, no gold/review payload is read, and no
hash/digest is calculated here.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from robata.benchmark.production_structured_annotation import (
    MODEL_NAMES,
    STRUCTURED_ANNOTATION_ENVELOPE_VERSION,
    ProductionStructuredAnnotationError,
    build_structured_annotation_envelope,
    normalize_structured_annotation_envelope,
)

STRUCTURED_REVIEW_DRAFT_VERSION: Final = "robata-production-structured-review-draft-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
WINDOW_BOUNDARY_NOTE: Final = (
    "The fixed source window is geometry for review only; it is not an action boundary."
)


class StructuredReviewAdapterError(ValueError):
    """Raised when a structured sidecar cannot be used for a review draft."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StructuredReviewAdapterError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StructuredReviewAdapterError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredReviewAdapterError(f"{field} must be non-empty text")
    return value.strip()


def _copy_json(value: object, *, field: str) -> Any:
    """Deep-copy JSON-compatible values without deriving an identity."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return copy.deepcopy(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise StructuredReviewAdapterError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise StructuredReviewAdapterError(f"{field} must be JSON-compatible")


def _normalise_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return token or None


def _candidate_pair(value: object) -> tuple[str, str] | None:
    """Extract an explicit pair solely for disagreement diagnostics."""

    if not isinstance(value, Mapping):
        return None
    verb = _normalise_token(value.get("verb"))
    noun = _normalise_token(value.get("noun"))
    if verb and noun:
        return verb, noun
    for key in ("candidate", "action", "prediction"):
        nested = value.get(key)
        pair = _candidate_pair(nested)
        if pair is not None:
            return pair
    label = value.get("label_text", value.get("label"))
    if isinstance(label, str):
        pieces = label.strip().split(None, 1)
        if len(pieces) == 2:
            left, right = _normalise_token(pieces[0]), _normalise_token(pieces[1])
            if left and right:
                return left, right
    return None


def _segment_pair(segment: Mapping[str, Any]) -> tuple[str, str] | None:
    labels = segment.get("structured_labels")
    if not isinstance(labels, Mapping):
        return None
    values: list[str] = []
    for key in ("verb", "noun"):
        record = labels.get(key)
        if not isinstance(record, Mapping) or record.get("status") != "MEASURED":
            return None
        value = _normalise_token(record.get("value"))
        if value is None:
            return None
        values.append(value)
    return values[0], values[1]


def _diagnostic_values(value: object) -> list[str]:
    """Return scalar/list diagnostic values without interpreting model prose.

    ``parse_observations`` is a deliberately loose provenance slot: older
    runners emitted one string while newer runners emit an array.  The review
    draft should expose either shape, but must never let a malformed value
    crash the diagnostic path or turn it into a semantic claim.
    """

    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _parse_observation_conflicts(section: Mapping[str, Any]) -> list[str]:
    """Project parser/mapping observations into review-only reason codes.

    The structured envelope intentionally retains invalid model output as raw
    provenance.  Without carrying its parser observations into the review
    draft, an invalid or timestamp-mapping-failed row looks indistinguishable
    from a clean no-action response (both have an empty ``segments`` array).
    These codes are diagnostics only: they do not promote a claim, read gold,
    or alter the provisional/abstain decision policy.
    """

    raw_observations = section.get("parse_observations", [])
    if not isinstance(raw_observations, Sequence) or isinstance(
        raw_observations, (str, bytes, bytearray)
    ):
        return []

    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason and reason not in reasons:
            reasons.append(reason)

    for raw_observation in raw_observations:
        if not isinstance(raw_observation, Mapping):
            continue

        parse_status = raw_observation.get("parse_status")
        if isinstance(parse_status, str):
            status = parse_status.strip().upper()
            if status and status not in {"PARSED", "EMPTY", "NOT_RUN", "NOT_APPLICABLE"}:
                add("PARSE_" + status)

        mapping_status = raw_observation.get("timestamp_mapping_status")
        if isinstance(mapping_status, str):
            status = mapping_status.strip().upper()
            if status == "FAILED":
                add("TIMESTAMP_MAPPING_FAILED")
            elif status == "UNSUPPORTED":
                add("TIMESTAMP_MAPPING_UNSUPPORTED")

        for error in _diagnostic_values(raw_observation.get("errors")):
            code = error.strip()
            upper_code = code.upper()
            # Mapping failures are important enough to have a stable,
            # unprefixed code for downstream review/telemetry.  Keep generic
            # parser messages namespaced so they cannot collide with the
            # review adapter's own conflict vocabulary.
            if upper_code in {
                "TIMESTAMP_MAPPING_FAILED",
                "TIMESTAMP_MAPPING_UNSUPPORTED",
            }:
                add(upper_code)
            else:
                add("PARSE_ERROR:" + code)

        for warning_key in ("warnings", "generation_warnings"):
            for warning in _diagnostic_values(raw_observation.get(warning_key)):
                add("PARSE_WARNING:" + warning)

        # Some runner revisions kept the mapping failure in a dedicated field
        # rather than in ``errors``.  Surface that field when present while
        # retaining its raw text as a parser error for auditability.
        mapping_error = raw_observation.get("timestamp_mapping_error")
        if isinstance(mapping_error, str) and mapping_error.strip():
            upper_error = mapping_error.strip().upper()
            if "TIMESTAMP_MAPPING" in upper_error:
                add("TIMESTAMP_MAPPING_FAILED")
            add("PARSE_ERROR:" + mapping_error.strip())

    return reasons


def _coerce_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an envelope, or wrap one raw model sidecar locally."""

    declared_format = value.get("format")
    if declared_format == STRUCTURED_ANNOTATION_ENVELOPE_VERSION:
        try:
            return normalize_structured_annotation_envelope(value)
        except ProductionStructuredAnnotationError as exc:
            raise StructuredReviewAdapterError(str(exc)) from exc

    # A direct Qwen/WeMM/Mage artifact is useful for a single-model review
    # smoke.  Its source reference is retained as the local source path; a
    # caller can still pass the combined envelope for a multi-model draft.
    model_key = {
        "robata-production-qwen-structured-native-shadow-v1": "qwen",
        "robata-production-wemm-shadow-v1": "wemm",
        "robata-production-wemm-vocabulary-shadow-v1": "wemm",
        "robata-production-mage-shadow-v1": "mage",
        "robata-production-mage-structured-native-shadow-v1": "mage",
    }.get(str(declared_format))
    if model_key is None:
        raise StructuredReviewAdapterError(
            "input must be a structured annotation envelope or a supported model sidecar"
        )
    source = value.get("source")
    source_path: str | None = None
    if isinstance(source, Mapping):
        for key in ("path", "manifest", "video_root"):
            candidate = source.get(key)
            if isinstance(candidate, str) and candidate.strip():
                source_path = candidate.strip()
                break
    if source_path is None:
        raise StructuredReviewAdapterError("model sidecar source path is required")
    source_mapping = source if isinstance(source, Mapping) else {}
    camera_count_value = source_mapping.get("camera_count")
    if not isinstance(camera_count_value, int) or isinstance(camera_count_value, bool):
        camera_count_value = None
    try:
        return build_structured_annotation_envelope(
            {model_key: value},
            source_path=source_path,
            camera_count=camera_count_value,
        )
    except (ProductionStructuredAnnotationError, ValueError, TypeError) as exc:
        raise StructuredReviewAdapterError(str(exc)) from exc


def _model_top_k(section: Mapping[str, Any], *, field: str) -> list[Any]:
    raw = section.get("candidates", [])
    copied = _copy_json(_sequence(raw, field=f"{field}.candidates"), field=f"{field}.candidates")
    if not isinstance(copied, list):
        raise StructuredReviewAdapterError(f"{field}.candidates must be an array")
    return copied


def _window_conflicts(window: Mapping[str, Any]) -> list[str]:
    models = _mapping(window.get("models"), field="window.models")
    conflicts: list[str] = []
    candidate_pairs: dict[str, tuple[str, str]] = {}
    segment_pairs: dict[str, set[tuple[str, str]]] = {}
    for model in MODEL_NAMES:
        section = _mapping(models.get(model), field=f"window.models.{model}")
        status = str(section.get("status", "NOT_RUN")).upper()
        if status in {"FAILED", "BLOCKED", "NOT_RUN"}:
            conflicts.append(f"MODEL_{status}:{model}")
        candidates = _sequence(
            section.get("candidates", []), field=f"window.models.{model}.candidates"
        )
        if candidates:
            pair = _candidate_pair(candidates[0])
            if pair is not None:
                candidate_pairs[model] = pair
        segments = _sequence(section.get("segments", []), field=f"window.models.{model}.segments")
        pairs = {
            pair
            for raw_segment in segments
            if isinstance(raw_segment, Mapping)
            for pair in [_segment_pair(raw_segment)]
            if pair is not None
        }
        if pairs:
            segment_pairs[model] = pairs

        # Parser/mapping observations are model/runtime provenance, not
        # semantic evidence.  Carry them into the per-window diagnostics so
        # an INVALID or timestamp-mapping-failed empty response cannot look
        # identical to a clean no-action observation.
        conflicts.extend(_parse_observation_conflicts(section))

    distinct_candidates = set(candidate_pairs.values())
    if len(distinct_candidates) > 1:
        conflicts.append("TOP1_MODEL_DISAGREEMENT")
    # A single model may legitimately emit multiple structured segments for
    # one bounded window.  Compare each model's *set of segment pairs* across
    # models rather than flattening all pairs into one set; otherwise a
    # multi-segment Qwen response is falsely reported as a cross-model
    # disagreement even when no other model supplied a structured claim.
    distinct_segment_sets = {frozenset(pairs) for pairs in segment_pairs.values()}
    if len(distinct_segment_sets) > 1:
        conflicts.append("STRUCTURED_SEGMENT_DISAGREEMENT")
    for model, pairs in segment_pairs.items():
        candidate = candidate_pairs.get(model)
        if candidate is not None and any(pair != candidate for pair in pairs):
            conflicts.append(f"STRUCTURED_VS_TOP1_DISAGREEMENT:{model}")
    return list(dict.fromkeys(conflicts))


def _window_draft(window: Mapping[str, Any]) -> dict[str, Any]:
    window_id = _text(window.get("window_id"), field="window.window_id")
    start = window.get("start_time_sec")
    end = window.get("end_time_sec")
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        raise StructuredReviewAdapterError(f"{window_id}.start_time_sec is invalid")
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        raise StructuredReviewAdapterError(f"{window_id}.end_time_sec is invalid")
    if float(end) <= float(start):
        raise StructuredReviewAdapterError(f"{window_id} window end must exceed start")
    models = _mapping(window.get("models"), field=f"window[{window_id}].models")

    model_top_k: dict[str, list[Any]] = {}
    claims: list[dict[str, Any]] = []
    source_models: list[str] = []
    for model in MODEL_NAMES:
        section = _mapping(models.get(model), field=f"window[{window_id}].models.{model}")
        model_top_k[model] = _model_top_k(section, field=f"window[{window_id}].models.{model}")
        raw_segments = _sequence(
            section.get("segments", []), field=f"window[{window_id}].models.{model}.segments"
        )
        for segment_index, raw_segment in enumerate(raw_segments):
            segment = _mapping(
                raw_segment,
                field=f"window[{window_id}].models.{model}.segments[{segment_index}]",
            )
            source_models.append(model)
            claims.append(
                {
                    "model": model,
                    "segment_index": segment_index,
                    "claim": _copy_json(segment, field=f"window[{window_id}].{model}.claim"),
                }
            )

    conflicts = _window_conflicts(window)
    if claims:
        draft_status = "PROVISIONAL"
        abstention_decision = "review"
        abstention_reasons = ["INDEPENDENT_REVIEW_REQUIRED"]
    else:
        draft_status = "ABSTAIN"
        abstention_decision = "abstain"
        abstention_reasons = ["NO_STRUCTURED_SEGMENTS"]
    if not any(model_top_k.values()):
        abstention_reasons.append("NO_MODEL_TOP_K")
    abstention_reasons.extend(
        conflict for conflict in conflicts if conflict not in abstention_reasons
    )
    return {
        "ordinal": window.get("ordinal"),
        "window_id": window_id,
        "interval": [float(start), float(end)],
        "annotation_draft": {
            "status": draft_status,
            "segments": claims,
            "source_models": sorted(set(source_models)),
            "window_boundary": {
                "start_time_sec": float(start),
                "end_time_sec": float(end),
                "is_action_boundary": False,
                "note": WINDOW_BOUNDARY_NOTE,
            },
        },
        "model_top_k": model_top_k,
        "conflicts": conflicts,
        "abstention": {
            "decision": abstention_decision,
            "reason_codes": list(dict.fromkeys(abstention_reasons)),
        },
    }


def build_structured_review_draft(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build a label-blind, sidecar-only review draft."""

    envelope = _coerce_envelope(_mapping(value, field="structured_input"))
    source = _mapping(envelope.get("source"), field="envelope.source")
    raw_windows = _sequence(envelope.get("windows"), field="envelope.windows")
    windows = [
        _window_draft(_mapping(raw, field=f"envelope.windows[{i}]"))
        for i, raw in enumerate(raw_windows)
    ]
    return {
        "format": STRUCTURED_REVIEW_DRAFT_VERSION,
        "authority": AUTHORITY,
        "production_eligible": False,
        "source": _copy_json(source, field="envelope.source"),
        "windows": windows,
        "review_contract": {
            "draft_status": "PROVISIONAL",
            "model_claims_are_not_gold": True,
            "window_boundaries_are_not_action_boundaries": True,
            "independent_review_required": True,
            "top_k_preserved_verbatim": True,
        },
        "quality": {
            "measurement_status": "NOT_MEASURED",
            "quality_claim": False,
            "reason": "machine-assisted claims require independent visual review",
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
    }


__all__ = [
    "AUTHORITY",
    "STRUCTURED_REVIEW_DRAFT_VERSION",
    "StructuredReviewAdapterError",
    "build_structured_review_draft",
]
