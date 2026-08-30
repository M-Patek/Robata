"""Normalize model claims into a label-blind structured annotation sidecar.

This module is deliberately a *benchmark-local adapter*.  It is not a
published schema and it does not change the production model-output contract,
the ontology, the Mapper, or any Web/API/UI surface.  Its only job is to put
already-recorded model observations from the WeMM and Qwen sidecars behind a
common, inspectable envelope so that a later evaluator can reason about
structured fields without mistaking missing evidence for a value.

The adapter has two important epistemic rules:

* A fixed model window is not an action boundary.  ``start_time_sec`` and
  ``end_time_sec`` are emitted only when a model explicitly supplied segment
  boundaries; otherwise a segment has null boundaries and
  ``boundary_status=NOT_MEASURED`` (and no segment is fabricated for a
  candidate-only model output).
* Every structured field is represented by a ``{"value": ..., "status":
  ...}`` record.  An absent field is ``NOT_MEASURED``; an explicit null is
  ``NOT_OBSERVABLE``.  No value is inferred from a candidate label, review
  artifact, frame metadata, or a fixed window.

Mage is represented as an explicit ``BLOCKED`` model section when no
source-bound native-codec observation is available.  This is a status, not a
prediction.  Candidate Top-K arrays are deep-copied verbatim under their
model-specific section and are never merged or rewritten here.

No media is decoded, no model is invoked, and no hash/digest is calculated.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

STRUCTURED_ANNOTATION_ENVELOPE_VERSION: Final = (
    "robata-production-structured-annotation-envelope-v1"
)
PRODUCTION_STRUCTURED_ANNOTATION_FORMAT: Final = STRUCTURED_ANNOTATION_ENVELOPE_VERSION
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
MODEL_NAMES: Final = ("wemm", "qwen", "mage")
STRUCTURED_FIELDS: Final = ("verb", "noun", "attributes", "location", "hand")
TIMESTAMP_BASIS: Final = "source_absolute_seconds"
WINDOW_RELATIVE_TIMESTAMP_BASIS: Final = "window_relative_seconds"
RELATIVE_TIMESTAMP_BASIS: Final = WINDOW_RELATIVE_TIMESTAMP_BASIS
TIMESTAMP_MAPPING_VERSION: Final = "window-relative-to-source-v1"
SUPPORTED_TIMESTAMP_BASES: Final = frozenset({TIMESTAMP_BASIS, WINDOW_RELATIVE_TIMESTAMP_BASIS})
QWEN_STRUCTURED_NATIVE_PROMPT_VERSION: Final = "qwen-native-structured-segments-v1"
QWEN_STRUCTURED_NATIVE_PROMPT: Final = (
    "Review the complete bounded native video interval once. Return ONLY one "
    "compact JSON object (not a top-level array, prose, or markdown). The no-"
    'action shape is {"segments":[]}; do not emit a placeholder segment. '
    'Each nonempty segment is shaped as {"start_time_sec":null,'
    '"end_time_sec":null,"structured_labels":{"verb":null,'
    '"noun":null,"attributes":null,"location":null,'
    '"hand":null},"confidence":0.0,"evidence":[]}. Return at most 3 '
    "segments. Use coarse visible object interactions, merge one continuous "
    "action, and omit reach/move/adjust or other transition filler unless it is "
    "the only visible action. Timestamps are source-absolute seconds from the "
    "original video; do not reset them to zero at the window start. Set boundaries "
    "only when directly visible on the source timeline; otherwise use null. Use "
    "null for an unobservable label. "
    "Include confidence in 0..1 (use 0.0 when uncertain) and at most one short "
    "evidence phrase per segment. Do not wrap structured_labels in an array. "
    "Do not infer intent, unseen objects, or taxonomy IDs, and never use the "
    "fixed window as an action boundary."
)
QWEN_STRUCTURED_NATIVE_RELATIVE_PROMPT_VERSION: Final = (
    "qwen-native-structured-segments-relative-v1"
)
QWEN_STRUCTURED_NATIVE_RELATIVE_PROMPT: Final = (
    "Review the complete bounded native video interval once. Return ONLY one "
    "compact JSON object (not a top-level array, prose, or markdown). The no-"
    'action shape is {"timestamp_basis":"window_relative_seconds",'
    '"segments":[]}; begin the object with the required timestamp_basis '
    "field and do not emit a placeholder segment. "
    'Each nonempty segment is shaped as {"start_time_sec":null,'
    '"end_time_sec":null,"structured_labels":{"verb":null,'
    '"noun":null,"attributes":null,"location":null,'
    '"hand":null},"confidence":0.0,"evidence":[]}. Return at most 3 '
    "segments. Use coarse visible object interactions, merge one continuous "
    "action, and omit reach/move/adjust or other transition filler unless it is "
    'the only visible action. Set the root field "timestamp_basis" to '
    '"window_relative_seconds" and express segment timestamps as offsets from '
    "the beginning of this bounded window (0 through the window duration). The "
    "runner maps these offsets to source-absolute seconds; do not add the window "
    "start yourself. Set boundaries only when directly visible; otherwise use "
    "null. Measured start_time_sec must be strictly less than end_time_sec; "
    "never emit equal timestamps, and use both null when a boundary is unclear. "
    "Prefer one compact segment over repeated guesses. Use null for an "
    "unobservable label. Include confidence in 0..1 "
    "(use 0.0 when uncertain) and at most one short evidence phrase per segment. "
    "Do not wrap structured_labels in an array. Do not infer intent, unseen "
    "objects, or taxonomy IDs, and never use the fixed window as an action "
    "boundary."
)
# Benchmark-only prompt variants for a stricter boundary policy.  The default
# prompts above remain byte-for-byte unchanged; callers must opt in through
# the runner's ``--strict-boundaries`` (or compatibility alias) flag.
QWEN_STRUCTURED_NATIVE_STRICT_BOUNDARIES_PROMPT_VERSION: Final = (
    "qwen-native-structured-segments-strict-boundaries-v1"
)
QWEN_STRUCTURED_NATIVE_STRICT_BOUNDARIES_PROMPT: Final = (
    QWEN_STRUCTURED_NATIVE_PROMPT
    + " Strict boundary mode: when either positive segment boundary is not "
    "directly visible, return exactly one atomic segment with both "
    "start_time_sec and end_time_sec set to null; if no atomic action is "
    "directly visible, return an empty segments array. Do not guess, split "
    "unclear boundaries, or use the fixed window as an action boundary."
)
QWEN_STRUCTURED_NATIVE_RELATIVE_STRICT_BOUNDARIES_PROMPT_VERSION: Final = (
    "qwen-native-structured-segments-relative-strict-boundaries-v1"
)
QWEN_STRUCTURED_NATIVE_RELATIVE_STRICT_BOUNDARIES_PROMPT: Final = (
    QWEN_STRUCTURED_NATIVE_RELATIVE_PROMPT
    + " Strict boundary mode: when either positive segment boundary is not "
    "directly visible, return exactly one atomic segment with both "
    "start_time_sec and end_time_sec set to null; if no atomic action is "
    "directly visible, return an empty segments array. Do not guess, split "
    "unclear boundaries, or use the fixed window as an action boundary."
)

# Benchmark-only production vocabulary arm.  The default Qwen prompt remains
# unchanged: this profile is an explicit experiment that tests whether a
# small, owner-scoped coarse vocabulary reduces free-prose drift on the local
# production-shaped textile cohort.  The vocabulary is *not* an ontology or
# gold source; its provenance is recorded by the runner and every output still
# goes through the ordinary review-only sidecar.
QWEN_PRODUCTION_COARSE_PROMPT_VERSION: Final = "qwen-production-coarse-vocabulary-relative-v1"
QWEN_PRODUCTION_COARSE_PROMPT: Final = (
    "Review the complete bounded native video interval once. Return ONLY one "
    "compact JSON object (not a top-level array, prose, or markdown) with "
    '{"timestamp_basis":"window_relative_seconds","segments":[]}. '
    "This is a production-shaped coarse annotation experiment, not EPIC "
    "classification. For visible textile/garment handling, use only these "
    "exact verb+noun labels: pick up garment, spread garment, flatten garment, "
    "adjust garment, smooth garment, or fold garment. Do not emit reach, move, "
    "arrange, take, put, point, or other filler/intent verbs. If none of the "
    "six labels is directly supported, return an empty segments array. Return "
    "at most two segments and split only when the visible action changes. Each "
    'segment is {"start_time_sec":null,"end_time_sec":null,'
    '"structured_labels":{"verb":"pick up","noun":"garment",'
    '"attributes":null,"location":null,"hand":null},'
    '"confidence":0.0,"evidence":[]}. Use window-relative seconds '
    "from 0 through the bounded interval duration; the runner maps them to "
    "source time, and never add the window start yourself. Set both boundaries "
    "to null when onset or completion is not directly visible; never use the "
    "fixed window as an action boundary. Use null for unobservable optional "
    "fields, confidence in 0..1, and at most one short evidence phrase. Do not "
    "infer intent, unseen objects, or taxonomy IDs."
)
QWEN_PRODUCTION_COARSE_STRICT_BOUNDARIES_PROMPT_VERSION: Final = (
    "qwen-production-coarse-vocabulary-relative-strict-boundaries-v1"
)
QWEN_PRODUCTION_COARSE_STRICT_BOUNDARIES_PROMPT: Final = (
    QWEN_PRODUCTION_COARSE_PROMPT
    + " Strict boundary mode: if either positive boundary is unclear, emit "
    "exactly one matching segment with both boundaries null; do not guess or "
    "split an unclear interval."
)

# A second, deliberately single-variable arm for the observed failure mode of
# the first coarse prompt.  That arm coupled action identity to boundary
# certainty and returned the empty shape for every production window.  This arm
# keeps the same six-label vocabulary but asks for one best label whenever
# garment handling is visible; uncertain onset/end are represented by null
# boundaries instead of suppressing the identity.  It remains benchmark-only
# and is never an ontology or gold source.
QWEN_PRODUCTION_COARSE_FORCED_PROMPT_VERSION: Final = (
    "qwen-production-coarse-vocabulary-relative-forced-candidate-v2"
)
QWEN_PRODUCTION_COARSE_FORCED_PROMPT: Final = (
    "Review the complete bounded native video once. Decide only the dominant "
    "garment-interaction identity; do not describe a sequence. Return exactly "
    "one compact JSON object and never more than one segment. If any hand to "
    "garment interaction is visible, MUST emit one segment even when timing is "
    "unclear. Choose exactly one bare verb value from: pick up, spread, flatten, "
    "adjust, smooth, fold. The verb value must never include the word garment. "
    "The noun must be exactly garment. Use an empty segments list "
    "only when no garment interaction is visible in any frame. Do not use reach, "
    "move, arrange, take, put, point, or intent verbs. The JSON object must have "
    "timestamp_basis, segments, and one segment with start_time_sec, end_time_sec, "
    "structured_labels (verb, noun, attributes, location, hand), confidence, and "
    "evidence. For a positive segment, Set both boundaries to null, use noun "
    "garment, one allowed verb, and one short evidence string. Within each segment include "
    "only these keys: start_time_sec, end_time_sec, structured_labels, confidence, "
    "and evidence; structured_labels contains verb, noun, attributes, location, "
    "and hand. Always keep both boundaries null in this identity pass; boundary "
    "uncertainty must never suppress the identity. Do not emit numeric timestamps, "
    "infer intent, unseen objects, or taxonomy IDs."
)
QWEN_PRODUCTION_COARSE_FORCED_STRICT_BOUNDARIES_PROMPT_VERSION: Final = (
    "qwen-production-coarse-vocabulary-relative-forced-candidate-strict-boundaries-v2"
)
QWEN_PRODUCTION_COARSE_FORCED_STRICT_BOUNDARIES_PROMPT: Final = (
    QWEN_PRODUCTION_COARSE_FORCED_PROMPT
    + " Strict boundary mode applies only to timing: keep the selected identity "
    "and both boundaries null."
)

# Identity-only diagnostic arm. It removes timestamp and segment-shape burden
# from the model so visual action identity can be measured independently from
# temporal grounding. Its output is never treated as a production annotation;
# it is a probe for deciding whether a later deterministic identity projection
# is warranted.
QWEN_PRODUCTION_IDENTITY_ONLY_PROMPT_VERSION: Final = "qwen-production-identity-only-v1"
QWEN_PRODUCTION_IDENTITY_ONLY_PROMPT: Final = (
    "Review the complete bounded native video once and decide only the dominant "
    "garment-interaction identity. Return exactly one compact JSON object with "
    "only these keys: action, confidence, evidence. The action value must be "
    "exactly one of: pick up garment, spread garment, flatten garment, adjust "
    "garment, smooth garment, fold garment, none visible, uncertain. If any "
    "hand-to-garment interaction is visible, choose the closest one of the six "
    "garment labels even when its onset or completion is unclear. Use none visible "
    "only when no garment interaction appears in any frame; use uncertain only "
    "when a garment interaction is visible but none of the six labels is "
    "distinguishable. Confidence must be a number from 0 to 1. Evidence must be "
    "one short directly visible fact. Do not output timestamps, segments, arrays, "
    "taxonomy IDs, intent, or prose outside the JSON object."
)
QWEN_PRODUCTION_IDENTITY_DISAMBIGUATED_PROMPT_VERSION: Final = (
    "qwen-production-identity-disambiguated-v1"
)
QWEN_PRODUCTION_IDENTITY_DISAMBIGUATED_PROMPT: Final = (
    "Review the complete bounded native video once and decide only the dominant "
    "garment-interaction identity. Return exactly one compact JSON object with "
    "only these keys: action, confidence, evidence. The action value must be "
    "exactly one of: pick up garment, spread garment, flatten garment, adjust "
    "garment, smooth garment, fold garment, none visible, uncertain. Use these "
    "visual definitions: pick up means the garment leaves its support surface or "
    "is held; spread means a bunched/folded garment is opened across a surface; "
    "flatten means pressing or pulling changes the garment to a flatter state; "
    "smooth means surface strokes remove wrinkles without materially changing its "
    "shape; fold means edges or halves are brought inward; adjust means only a "
    "minor repositioning when none of the other five applies. Choose the net "
    "state-changing interaction, not the most frequent hand motion. Do not use "
    "flatten as a catch-all. If the distinction is not visible, use uncertain. "
    "Use none visible only when no garment interaction appears in any frame. "
    "If any interaction is visible, do not return none visible. Confidence must "
    "be a number from 0 to 1 and evidence one short directly visible fact. Do not "
    "output timestamps, segments, arrays, taxonomy IDs, intent, or prose outside "
    "the JSON object."
)

_IDENTITY_ACTIONS: Final = {
    "pick up garment": ("pick up", "garment"),
    "spread garment": ("spread", "garment"),
    "flatten garment": ("flatten", "garment"),
    "adjust garment": ("adjust", "garment"),
    "smooth garment": ("smooth", "garment"),
    "fold garment": ("fold", "garment"),
}
FIELD_STATUSES: Final = ("MEASURED", "NOT_MEASURED", "NOT_OBSERVABLE")
SEGMENT_STATUSES: Final = (
    "MEASURED",
    "NOT_MEASURED",
    "NOT_OBSERVABLE",
    "BLOCKED",
    "FAILED",
)
MODEL_STATUSES: Final = (
    "NOT_RUN",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    "SKIPPED",
    "BLOCKED",
)
_MODEL_STATUS_SET = frozenset(MODEL_STATUSES)
_FIELD_STATUS_SET = frozenset(FIELD_STATUSES)
_SEGMENT_STATUS_SET = frozenset(SEGMENT_STATUSES)
_MISSING = object()
_COARSE_FILLER_VERBS = frozenset(
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


class ProductionStructuredAnnotationError(ValueError):
    """Raised when a structured annotation envelope cannot be normalized."""


# Compatibility aliases make the local adapter discoverable without creating
# another implementation or a second wire contract.
ProductionStructuredAnnotationContractError = ProductionStructuredAnnotationError


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionStructuredAnnotationError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionStructuredAnnotationError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionStructuredAnnotationError(f"{field} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ProductionStructuredAnnotationError(f"{field} must be non-empty")
    return result


def _finite(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionStructuredAnnotationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ProductionStructuredAnnotationError(f"{field} must be a finite number")
    return result


def _json_copy(value: object, *, field: str) -> Any:
    """Deep-copy JSON data and reject non-finite or runtime objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionStructuredAnnotationError(
                f"{field} must not contain a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionStructuredAnnotationError(f"{field} keys must be strings")
            result[key] = _json_copy(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_copy(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionStructuredAnnotationError(f"{field} must be JSON-compatible")


def _normalised_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


_GOLD_KEY_PARTS: Final = (
    "gold",
    "groundtruth",
    "officialreference",
    "officiallabel",
    "humanlabel",
    "humanannotation",
    "adjudication",
    "review",
    "annotation",
)


def _assert_no_gold_fields(value: object, *, field: str) -> None:
    """Fail closed on explicit gold/review keys.

    Model prose is allowed to contain words such as ``review``; only mapping
    keys are inspected.  This mirrors the existing label-blind output
    contract and prevents accidental promotion of a review artifact.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProductionStructuredAnnotationError(f"{field} must not contain non-finite data")
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ProductionStructuredAnnotationError(f"{field} keys must be strings")
            key = _normalised_key(raw_key)
            # Existing model sidecars carry safe boolean controls such as
            # gold_included=false.  Those are provenance facts, not labels.
            safe_control = key in {
                "goldincluded",
                "goldread",
                "goldwritten",
                "predictionsaregold",
                "predictionscopiedtogold",
                "goldfieldsincluded",
                "goldisexternal",
                "groundtruthused",
                "groundtruthusedinencoderinput",
                "heldout100opened",
            }
            safe_control_value = child is True if key == "goldisexternal" else child is False
            # Current model shadows also repeat a non-gold status/eligibility
            # envelope.  Permit only the explicit safe values; a producer
            # claiming an accepted/official label must still be rejected.
            safe_status_values: dict[str, frozenset[object]] = {
                "officialgoldstatus": frozenset({"NOT_ESTABLISHED", "NOT_MEASURED"}),
                "officialqualitystatus": frozenset({"NOT_MEASURED"}),
                "qualityclaim": frozenset({False}),
                "productioneligible": frozenset({False}),
                "automaticeligible": frozenset({False}),
                "automaticqualification": frozenset({False}),
                "officialgold": frozenset({False}),
                "acceptedasgold": frozenset({False}),
            }
            safe_status = (
                key in safe_status_values
                and isinstance(child, (str, bool, int, float, type(None)))
                and child in safe_status_values[key]
            )
            # Some retrieval shadows record paths to upstream review/surface
            # artifacts solely as provenance.  They are not read by this
            # adapter.  Permit those locator strings, but never accept an
            # embedded object under a review-bearing key.
            safe_locator_keys = {
                "reviewartifact",
                "decisionartifact",
                "priorprovisionalvocabulary",
                "surfacebundle",
            }
            safe_locator = key in safe_locator_keys and isinstance(child, str)
            if any(part in key for part in _GOLD_KEY_PARTS) and not (
                (safe_control and safe_control_value) or safe_status or safe_locator
            ):
                raise ProductionStructuredAnnotationError(
                    f"{field}.{raw_key} contains gold/review/annotation data"
                )
            _assert_no_gold_fields(child, field=f"{field}.{raw_key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_gold_fields(child, field=f"{field}[{index}]")
        return
    raise ProductionStructuredAnnotationError(f"{field} must be JSON-compatible")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON sidecar without deriving a content identity."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionStructuredAnnotationError(f"cannot load JSON sidecar {source}") from exc
    payload = _mapping(value, field=f"{source}")
    _assert_no_gold_fields(payload, field=str(source))
    return cast(dict[str, Any], _json_copy(payload, field=str(source)))


def _payload(value: object, *, field: str, allow_gold: bool = False) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        if allow_gold:
            source = Path(value)
            try:
                with source.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ProductionStructuredAnnotationError(
                    f"cannot load JSON sidecar {source}"
                ) from exc
            return _mapping(loaded, field=str(source))
        return load_json(value)
    payload = _mapping(value, field=field)
    if not allow_gold:
        _assert_no_gold_fields(payload, field=field)
    return payload


def _extract_source_path(payload: Mapping[str, Any]) -> str | None:
    """Return a useful source path while avoiding any content identity."""

    source = payload.get("source")
    if isinstance(source, Mapping):
        for key in ("path", "media_path", "source_path", "mcap_path", "video_path"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # Qwen's shadow stores a manifest and a video root.  The manifest is
        # the most stable source-bound reference when no explicit media path
        # is available.
        for key in ("manifest", "video_root"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("source_path", "media_path", "mcap_path", "video_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_camera_count(payload: Mapping[str, Any]) -> int | None:
    source = payload.get("source")
    if isinstance(source, Mapping):
        value = source.get("camera_count")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    for key in ("camera_count", "expected_camera_count"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    camera_ids = payload.get("camera_ids")
    if isinstance(camera_ids, Sequence) and not isinstance(camera_ids, (str, bytes, bytearray)):
        return len(camera_ids)
    return None


def _window_id(value: Mapping[str, Any], *, field: str) -> str:
    return _text(value.get("window_id"), field=f"{field}.window_id")


def _explicit_interval(value: Mapping[str, Any], *, field: str) -> tuple[float, float] | None:
    """Read a canonical interval, but not a Qwen observational ``interval``."""

    if "start_time_sec" in value or "end_time_sec" in value:
        start_key, end_key = "start_time_sec", "end_time_sec"
    elif "start_seconds" in value or "end_seconds" in value:
        start_key, end_key = "start_seconds", "end_seconds"
    else:
        return None
    if value.get(start_key) is None or value.get(end_key) is None:
        return None
    start = _finite(value.get(start_key), field=f"{field}.{start_key}", minimum=0.0)
    end = _finite(value.get(end_key), field=f"{field}.{end_key}", minimum=0.0)
    if end <= start:
        raise ProductionStructuredAnnotationError(f"{field} interval end must exceed start")
    return start, end


def _observational_interval(value: Mapping[str, Any], *, field: str) -> tuple[float, float] | None:
    """Read Qwen's ``interval`` only as provenance, never as action boundary."""

    raw = value.get("interval")
    if raw is None:
        return None
    values = _sequence(raw, field=f"{field}.interval")
    if len(values) != 2:
        raise ProductionStructuredAnnotationError(f"{field}.interval must contain two values")
    start = _finite(values[0], field=f"{field}.interval[0]", minimum=0.0)
    end = _finite(values[1], field=f"{field}.interval[1]", minimum=0.0)
    if end <= start:
        raise ProductionStructuredAnnotationError(f"{field}.interval end must exceed start")
    return start, end


def _status(value: object, *, field: str, default: str = "NOT_RUN") -> str:
    if value is None:
        return default
    result = _text(value, field=field).upper()
    if result not in _MODEL_STATUS_SET:
        # A few adapters use ``NOT_MEASURED`` as a model status.  Keep the
        # model section valid while making the epistemic state explicit.
        if result == "NOT_MEASURED":
            return "NOT_RUN"
        raise ProductionStructuredAnnotationError(
            f"{field} must be one of {', '.join(MODEL_STATUSES)}"
        )
    return result


def _source_window_rows(payload: Mapping[str, Any], *, model: str) -> list[Mapping[str, Any]]:
    raw_windows = payload.get("windows")
    if raw_windows is None:
        return []
    rows = _sequence(raw_windows, field=f"{model}.windows")
    result: list[Mapping[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"{model}.windows[{index}]")
        result.append(row)
    return result


def _candidate_list(row: Mapping[str, Any], *, model: str, field: str) -> list[Any]:
    """Extract and deep-copy the model's existing Top-K candidates verbatim."""

    groups = _candidate_groups(row, model=model, field=field)
    if groups:
        return cast(list[Any], groups[0]["candidates"])
    return []


def _candidate_groups(row: Mapping[str, Any], *, model: str, field: str) -> list[dict[str, Any]]:
    """Retain every candidate source without changing the legacy Top-K slot."""

    groups: list[dict[str, Any]] = []

    # Production model-output slots and resolver rows may expose both
    # ``candidates`` and ``predictions``.  Keep each list under its source key;
    # the legacy ``candidates`` field continues to use the first source below.
    for key in ("candidates", "predictions"):
        if key in row:
            values = _sequence(row.get(key), field=f"{field}.{key}")
            groups.append(
                {
                    "source_field": key,
                    "candidates": _json_copy(values, field=f"{field}.{key}"),
                }
            )
    nested = row.get("model")
    if isinstance(nested, Mapping):
        for key in ("candidates", "predictions"):
            if key in nested:
                values = _sequence(nested.get(key), field=f"{field}.model.{key}")
                groups.append(
                    {
                        "source_field": f"model.{key}",
                        "candidates": _json_copy(values, field=f"{field}.model.{key}"),
                    }
                )
    # Qwen shadow rows expose one JSON/prose result per camera rather than a
    # named Top-K list.  Retain the parsed object (or raw text) as one claim;
    # camera/runtime metadata remains in ``candidate_sources``.
    if model == "qwen" and "raw_text" in row and not groups:
        raw_text = row.get("raw_text")
        if isinstance(raw_text, str):
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                parsed = {"raw_text": raw_text}
            groups.append(
                {
                    "source_field": "raw_text",
                    "candidates": [_json_copy(parsed, field=f"{field}.raw_text")],
                }
            )
        else:
            groups.append(
                {
                    "source_field": "raw_text",
                    "candidates": [_json_copy(raw_text, field=f"{field}.raw_text")],
                }
            )
    return groups


def _candidate_sources(row: Mapping[str, Any], *, field: str) -> list[dict[str, Any]]:
    """Retain Qwen per-camera provenance without changing candidate claims."""

    keep = (
        "camera_id",
        "status",
        "raw_text",
        "frame_indices",
        "frame_timestamps_seconds",
        "input_mode",
    )
    result: dict[str, Any] = {}
    for key in keep:
        if key in row:
            result[key] = _json_copy(row[key], field=f"{field}.{key}")
    return [result] if result else []


def _field_record(value: object, *, present: bool, field: str) -> dict[str, Any]:
    if not present:
        return {"value": None, "status": "NOT_MEASURED"}
    if isinstance(value, Mapping) and "status" in value and "value" in value:
        # The Qwen parser may already have produced the canonical value/status
        # representation.  Preserve it rather than wrapping it a second time.
        status = _text(value.get("status"), field=f"{field}.status").upper()
        if status not in _FIELD_STATUS_SET:
            raise ProductionStructuredAnnotationError(
                f"{field}.status must be one of {', '.join(FIELD_STATUSES)}"
            )
        inner = value.get("value")
        if status == "MEASURED" and inner is None:
            raise ProductionStructuredAnnotationError(f"{field}.value cannot be null when measured")
        if status != "MEASURED" and inner not in (None, "UNSPECIFIED"):
            raise ProductionStructuredAnnotationError(f"{field}.value must be null for {status}")
        return {"value": _json_copy(inner, field=f"{field}.value"), "status": status}
    if isinstance(value, str):
        marker = value.strip().upper()
        if marker in {"NOT_MEASURED", "NOT_OBSERVABLE"}:
            return {"value": None, "status": marker}
        if marker == "UNSPECIFIED":
            # UNSPECIFIED is an explicit non-observation, not a measured
            # lexical value.  Keep the marker for auditability.
            return {"value": "UNSPECIFIED", "status": "NOT_OBSERVABLE"}
    if value is None:
        return {"value": None, "status": "NOT_OBSERVABLE"}
    return {"value": _json_copy(value, field=field), "status": "MEASURED"}


def _labels_from_segment(segment: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    nested = segment.get("structured_labels")
    if nested is None:
        nested = segment.get("labels")
    labels = _mapping(nested, field=f"{field}.structured_labels") if nested is not None else segment
    return {
        name: _field_record(labels.get(name), present=name in labels, field=f"{field}.{name}")
        for name in STRUCTURED_FIELDS
    }


def _alias_values_equal(left: object, right: object) -> bool:
    """Compare two compatibility aliases without deriving an identity."""

    def canonical(value: object) -> object:
        # A plain label (``"open"``) and the parser's canonical
        # ``{"value":"open","status":"MEASURED"}`` are the same explicit
        # observation.  Treat those as equal aliases while keeping
        # non-measured status markers distinct.
        if isinstance(value, Mapping) and ("status" in value or "value" in value):
            status = str(value.get("status") or "").strip().upper()
            if status == "MEASURED":
                return canonical(value.get("value"))
            if status in _FIELD_STATUS_SET:
                return (status, canonical(value.get("value")))
        if isinstance(value, Mapping):
            return tuple(sorted((str(key), canonical(child)) for key, child in value.items()))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return tuple(canonical(child) for child in value)
        if isinstance(value, str):
            return value.strip()
        return value

    return canonical(left) == canonical(right)


def _assert_segment_alias_consistency(segment: Mapping[str, Any], *, field: str) -> None:
    """Reject conflicting legacy/canonical aliases instead of choosing one.

    The sidecar adapter accepts both timestamp spellings and both structured
    label container spellings for compatibility.  If a producer emits both,
    silently preferring one would make the observation non-reproducible; equal
    aliases remain valid, while disagreements are explicit parse failures.
    """

    for primary, alias in (
        ("start_time_sec", "start_seconds"),
        ("end_time_sec", "end_seconds"),
    ):
        if (
            primary in segment
            and alias in segment
            and not _alias_values_equal(segment.get(primary), segment.get(alias))
        ):
            raise ProductionStructuredAnnotationError(
                f"{field} ALIAS_CONFLICT: {primary} and {alias} disagree"
            )
    if "structured_labels" in segment and "labels" in segment:
        primary_label = segment.get("structured_labels")
        alias_label = segment.get("labels")
        if not isinstance(primary_label, Mapping) or not isinstance(alias_label, Mapping):
            if not _alias_values_equal(primary_label, alias_label):
                raise ProductionStructuredAnnotationError(
                    f"{field} ALIAS_CONFLICT: structured_labels and labels disagree"
                )
        else:
            if set(primary_label) != set(alias_label):
                raise ProductionStructuredAnnotationError(
                    f"{field} ALIAS_CONFLICT: structured_labels and labels disagree"
                )
            for name in set(primary_label) & set(alias_label):
                if not _alias_values_equal(primary_label.get(name), alias_label.get(name)):
                    raise ProductionStructuredAnnotationError(
                        f"{field} ALIAS_CONFLICT: structured_labels and labels disagree"
                    )


def _normalise_segment(segment: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    # Explicit segment records only.  In particular, a source window's
    # ``start_seconds``/``end_seconds`` are never used as this segment's
    # boundaries.
    _assert_segment_alias_consistency(segment, field=field)
    has_start = "start_time_sec" in segment or "start_seconds" in segment
    has_end = "end_time_sec" in segment or "end_seconds" in segment
    if has_start != has_end:
        raise ProductionStructuredAnnotationError(
            f"{field} PARTIAL_BOUNDARY requires both start and end"
        )
    if "start_time_sec" in segment:
        start_raw = segment.get("start_time_sec")
    else:
        start_raw = segment.get("start_seconds")
    if "end_time_sec" in segment:
        end_raw = segment.get("end_time_sec")
    else:
        end_raw = segment.get("end_seconds")
    if (start_raw is None) != (end_raw is None):
        raise ProductionStructuredAnnotationError(
            f"{field} PARTIAL_BOUNDARY requires both start and end"
        )
    if start_raw is None and end_raw is None:
        start, end = None, None
        boundary_status = "NOT_MEASURED"
    else:
        start = _finite(start_raw, field=f"{field}.start_time_sec", minimum=0.0)
        end = _finite(end_raw, field=f"{field}.end_time_sec", minimum=0.0)
        if end <= start:
            raise ProductionStructuredAnnotationError(
                f"{field} end_time_sec must exceed start_time_sec"
            )
        boundary_status = "MEASURED"

    labels = _labels_from_segment(segment, field=field)

    confidence_present = "confidence" in segment
    confidence_raw = segment.get("confidence")
    if not confidence_present:
        confidence, confidence_status = None, "NOT_MEASURED"
    elif confidence_raw is None:
        confidence, confidence_status = None, "NOT_OBSERVABLE"
    else:
        confidence = _finite(confidence_raw, field=f"{field}.confidence", minimum=0.0)
        if confidence > 1.0:
            raise ProductionStructuredAnnotationError(f"{field}.confidence must be between 0 and 1")
        confidence_status = "MEASURED"

    evidence_present = "evidence" in segment
    evidence_raw = segment.get("evidence")
    evidence: Any
    if not evidence_present:
        evidence, evidence_status = [], "NOT_MEASURED"
    elif evidence_raw is None:
        evidence, evidence_status = [], "NOT_OBSERVABLE"
    else:
        evidence = _json_copy(evidence_raw, field=f"{field}.evidence")
        evidence_status = "MEASURED"

    explicit_status = segment.get("status")
    if explicit_status is not None:
        segment_status = _text(explicit_status, field=f"{field}.status").upper()
        if segment_status not in _SEGMENT_STATUS_SET:
            raise ProductionStructuredAnnotationError(
                f"{field}.status must be one of {', '.join(SEGMENT_STATUSES)}"
            )
    else:
        measured_labels = any(item["status"] == "MEASURED" for item in labels.values())
        segment_status = "MEASURED" if measured_labels else "NOT_MEASURED"

    result: dict[str, Any] = {
        "start_time_sec": start,
        "end_time_sec": end,
        "boundary_status": boundary_status,
        "structured_labels": labels,
        "confidence": confidence,
        "confidence_status": confidence_status,
        "evidence": evidence,
        "evidence_status": evidence_status,
        "status": segment_status,
    }
    # Relative-time mapping is performed after parsing.  Keep its explicit
    # provenance when a mapped segment is passed through the envelope builder;
    # historical segments without these keys remain unchanged.
    for key in (
        "raw_start_time_sec",
        "raw_end_time_sec",
        "raw_timestamp_basis",
        "mapped_start_time_sec",
        "mapped_end_time_sec",
        "timestamp_basis",
        "timestamp_mapping_status",
        "timestamp_mapping_version",
        "timestamp_mapping",
    ):
        if key in segment:
            result[key] = _json_copy(segment[key], field=f"{field}.{key}")
    if "segment_id" in segment:
        result["segment_id"] = _text(segment.get("segment_id"), field=f"{field}.segment_id")
    return result


def _explicit_segments(row: Mapping[str, Any], *, field: str) -> list[dict[str, Any]]:
    for key in ("segments", "structured_segments"):
        if key not in row:
            continue
        raw_segments = _sequence(row.get(key), field=f"{field}.{key}")
        result: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_segments):
            segment = _mapping(raw, field=f"{field}.{key}[{index}]")
            result.append(_normalise_segment(segment, field=f"{field}.{key}[{index}]"))
        return result
    nested = row.get("model")
    if isinstance(nested, Mapping):
        return _explicit_segments(nested, field=f"{field}.model")
    return []


def _strict_json_loads(value: str) -> Any:
    """Decode one JSON value while rejecting duplicate object keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = child
        return result

    return json.loads(value, object_pairs_hook=reject_duplicates)


def _possible_truncation_warning(raw_text: str, error: json.JSONDecodeError) -> list[str]:
    """Return a diagnostic warning for a likely token-truncated JSON response.

    We deliberately do not repair a prefix or infer a missing closing token.  A
    model response that cannot be decoded remains ``INVALID``.  This small
    diagnostic is useful for distinguishing a strict-format failure caused by
    generation length from one caused by prose or an unsupported shape.
    """

    stripped = raw_text.strip()
    if not stripped or stripped[0] not in "[{":
        return []
    message = error.msg.casefold()
    likely_prefix_failure = (
        "unterminated" in message
        or "expecting value" in message
        or "expecting ',' delimiter" in message
        or "expecting property name" in message
    )
    if likely_prefix_failure or stripped[-1] not in "]}":
        return ["POSSIBLY_TRUNCATED_JSON"]
    return []


def _compat_segment_shape(
    value: Mapping[str, Any], *, field: str, warnings: list[str]
) -> dict[str, Any]:
    """Normalize harmless wrapper-shape variants emitted by local Qwen runs.

    Some Qwen generations have returned ``structured_labels`` as a one-item
    array even though the requested contract uses an object.  Unwrapping that
    explicit one-item container changes no label value and does not invent a
    boundary.  Ambiguous multi-item arrays are rejected rather than silently
    dropping claims or fabricating additional segments.
    """

    segment = dict(value)
    for key in ("structured_labels", "labels"):
        if key not in segment:
            continue
        nested = segment[key]
        if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes, bytearray)):
            continue
        if len(nested) == 1 and isinstance(nested[0], Mapping):
            segment[key] = dict(nested[0])
            warnings.append(f"{key.upper()}_ARRAY_COMPAT")
            continue
        if len(nested) == 0:
            # An explicit empty wrapper means no structured fields were
            # provided; represent that as an empty mapping so each field stays
            # ``NOT_MEASURED`` rather than inventing a value.
            segment[key] = {}
            warnings.append(f"{key.upper()}_EMPTY_ARRAY_COMPAT")
            continue
        raise ProductionStructuredAnnotationError(
            f"{field}.{key} array must contain exactly one object"
        )
    return segment


def _coarse_output_warnings(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    """Report prompt-policy deviations without rewriting model claims.

    The native runner asks for a small coarse vocabulary, but the parser must
    remain an observation adapter rather than a semantic classifier.  These
    warnings therefore retain every explicit segment and only make deviations
    visible to a later evaluator.
    """

    warnings: list[str] = []
    if len(segments) > 3:
        warnings.append("SEGMENT_COUNT_EXCEEDS_LIMIT")
    for segment in segments:
        evidence = segment.get("evidence")
        if (
            isinstance(evidence, Sequence)
            and not isinstance(evidence, (str, bytes, bytearray))
            and len(evidence) > 1
        ):
            warnings.append("EVIDENCE_COUNT_EXCEEDS_LIMIT")
        labels = segment.get("structured_labels")
        if isinstance(labels, Mapping):
            verb = labels.get("verb")
            if isinstance(verb, Mapping):
                verb = verb.get("value")
            if isinstance(verb, str) and verb.strip().casefold() in _COARSE_FILLER_VERBS:
                warnings.append("FILLER_VERB_PRESENT:" + verb.strip().casefold())
    return warnings


def parse_qwen_structured_output(raw_text: str | None) -> dict[str, Any]:
    """Parse Qwen's strict structured-segment response without invoking a model.

    The raw text remains the caller's responsibility to retain verbatim.  This
    function returns a small parser result suitable for a sidecar: malformed
    output is represented as ``INVALID`` with reason codes rather than being
    converted into a guessed segment.  A valid empty ``segments`` array is a
    parsed no-action observation and remains structurally distinct from invalid
    JSON.
    """

    errors: list[str] = []
    warnings: list[str] = []
    if raw_text is None:
        return {
            "parse_status": "INVALID",
            "timestamp_basis_explicit": False,
            "segments": [],
            "candidates": [],
            "errors": ["MODEL_NOT_RUN"],
            "warnings": warnings,
        }
    if not isinstance(raw_text, str):
        raise ProductionStructuredAnnotationError("raw_text must be a string or null")
    if not raw_text.strip():
        return {
            "parse_status": "INVALID",
            "timestamp_basis_explicit": False,
            "segments": [],
            "candidates": [],
            "errors": ["EMPTY_RESPONSE"],
            "warnings": warnings,
        }
    try:
        decoded = _strict_json_loads(raw_text)
    except json.JSONDecodeError as exc:
        warnings.extend(_possible_truncation_warning(raw_text, exc))
        return {
            "parse_status": "INVALID",
            "timestamp_basis_explicit": False,
            "segments": [],
            "candidates": [],
            "errors": ["INVALID_JSON", "STRICT_JSON_REQUIRED"],
            "warnings": list(dict.fromkeys(warnings)),
        }
    except (TypeError, ValueError):
        return {
            "parse_status": "INVALID",
            "timestamp_basis_explicit": False,
            "segments": [],
            "candidates": [],
            "errors": ["INVALID_JSON", "STRICT_JSON_REQUIRED"],
            "warnings": warnings,
        }
    if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes, bytearray)):
        # A common Qwen response is a bare segment array.  It is structurally
        # equivalent to the requested ``{"segments": [...]}`` wrapper, so we
        # accept it with an explicit compatibility warning.  No values or
        # boundaries are synthesized.
        decoded = {"segments": decoded}
        warnings.append("ROOT_ARRAY_COMPAT")
    if not isinstance(decoded, Mapping):
        return {
            "parse_status": "INVALID",
            "timestamp_basis_explicit": False,
            "segments": [],
            "candidates": [],
            "errors": ["JSON_OBJECT_REQUIRED"],
            "warnings": warnings,
        }
    # ``timestamp_basis`` was added for the explicit relative-time experiment.
    # It is optional so historical/source-absolute responses remain byte-shape
    # compatible with the parser (missing means source-absolute below).
    allowed_root = {"segments", "candidates", "timestamp_basis"}
    unknown_root = sorted(set(decoded) - allowed_root)
    if unknown_root:
        errors.append("UNKNOWN_ROOT_FIELDS:" + ",".join(unknown_root))
    timestamp_basis: str | None
    timestamp_basis_explicit = "timestamp_basis" in decoded
    if "timestamp_basis" not in decoded:
        timestamp_basis = TIMESTAMP_BASIS
        timestamp_basis_status = "MEASURED"
    else:
        raw_basis = decoded.get("timestamp_basis")
        if not isinstance(raw_basis, str) or not raw_basis.strip():
            timestamp_basis = None
            timestamp_basis_status = "INVALID"
            errors.append("TIMESTAMP_BASIS_INVALID")
        else:
            timestamp_basis = raw_basis.strip()
            timestamp_basis_status = (
                "MEASURED" if timestamp_basis in SUPPORTED_TIMESTAMP_BASES else "UNSUPPORTED"
            )
            if timestamp_basis_status == "UNSUPPORTED":
                # Keep the explicit segments parseable and visible.  The
                # projection/helper will refuse to reinterpret an unknown
                # clock; this is an observation boundary, not a parse failure.
                warnings.append("UNSUPPORTED_TIMESTAMP_BASIS:" + timestamp_basis)
    if "segments" not in decoded:
        errors.append("SEGMENTS_MISSING")
    segments: list[dict[str, Any]] = []
    if not errors:
        raw_segments = decoded.get("segments")
        if not isinstance(raw_segments, Sequence) or isinstance(
            raw_segments, (str, bytes, bytearray)
        ):
            errors.append("SEGMENTS_ARRAY_REQUIRED")
        else:
            allowed_segment = {
                "segment_id",
                "start_time_sec",
                "end_time_sec",
                "start_seconds",
                "end_seconds",
                "structured_labels",
                "labels",
                "verb",
                "noun",
                "attributes",
                "location",
                "hand",
                "confidence",
                "evidence",
                "status",
            }
            try:
                for index, raw_segment in enumerate(raw_segments):
                    if not isinstance(raw_segment, Mapping):
                        raise ProductionStructuredAnnotationError(
                            f"segments[{index}] must be an object"
                        )
                    unknown = sorted(set(raw_segment) - allowed_segment)
                    if unknown:
                        raise ProductionStructuredAnnotationError(
                            f"segments[{index}] contains unsupported fields: {', '.join(unknown)}"
                        )
                    compatible = _compat_segment_shape(
                        raw_segment, field=f"segments[{index}]", warnings=warnings
                    )
                    segments.append(_normalise_segment(compatible, field=f"segments[{index}]"))
            except ProductionStructuredAnnotationError as exc:
                errors.append(str(exc))
                segments = []
    candidates: list[Any] = []
    if "candidates" in decoded:
        raw_candidates = decoded.get("candidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(
            raw_candidates, (str, bytes, bytearray)
        ):
            errors.append("CANDIDATES_ARRAY_REQUIRED")
        else:
            candidates = cast(list[Any], _json_copy(raw_candidates, field="structured.candidates"))
    warnings.extend(_coarse_output_warnings(segments))
    return {
        "parse_status": "PARSED" if not errors else "INVALID",
        "timestamp_basis_explicit": timestamp_basis_explicit,
        "timestamp_basis": timestamp_basis,
        "timestamp_basis_status": timestamp_basis_status,
        "segments": segments,
        "candidates": candidates,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def parse_qwen_identity_only_output(raw_text: str | None) -> dict[str, Any]:
    """Parse the benchmark-only identity probe without temporal fields.

    This parser intentionally has no timestamp or segment repair path.  It
    validates only the three fields requested by the identity arm and returns
    the raw action separately so callers can keep canonical timing validation
    independent.  ``none visible`` and ``uncertain`` are explicit outcomes,
    not missing data.
    """

    result: dict[str, Any] = {
        "parse_status": "INVALID",
        "action": None,
        "confidence": None,
        "evidence": [],
        "errors": [],
        "warnings": [],
    }
    if raw_text is None:
        result["errors"] = ["MODEL_NOT_RUN"]
        return result
    if not isinstance(raw_text, str) or not raw_text.strip():
        result["errors"] = ["EMPTY_RESPONSE"]
        return result
    try:
        decoded = _strict_json_loads(raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        result["errors"] = ["INVALID_JSON", "STRICT_JSON_REQUIRED"]
        return result
    if not isinstance(decoded, Mapping):
        result["errors"] = ["JSON_OBJECT_REQUIRED"]
        return result
    unknown = sorted(set(decoded) - {"action", "confidence", "evidence"})
    if unknown:
        result["errors"].append("UNKNOWN_FIELDS:" + ",".join(unknown))
    action = decoded.get("action")
    if not isinstance(action, str) or not action.strip():
        result["errors"].append("ACTION_REQUIRED")
    else:
        action = action.strip().casefold()
        if action not in {*_IDENTITY_ACTIONS, "none visible", "uncertain"}:
            result["errors"].append("ACTION_OUT_OF_VOCABULARY:" + action)
        result["action"] = action
    confidence = decoded.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        result["errors"].append("CONFIDENCE_REQUIRED")
    elif not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        result["errors"].append("CONFIDENCE_OUT_OF_RANGE")
    else:
        result["confidence"] = float(confidence)
    evidence = decoded.get("evidence")
    if isinstance(evidence, str) and evidence.strip():
        result["evidence"] = [evidence.strip()]
    else:
        result["errors"].append("EVIDENCE_REQUIRED")
    result["parse_status"] = "PARSED" if not result["errors"] else "INVALID"
    return result


def map_qwen_relative_timestamps(
    parsed: Mapping[str, Any],
    *,
    window_start_seconds: float,
    window_end_seconds: float,
) -> dict[str, Any]:
    """Map an explicit Qwen window-relative result to source time.

    The model's raw response is retained by the caller in ``raw_text``.  This
    helper additionally keeps the numeric values emitted by the model on each
    mapped segment (``raw_start_time_sec``/``raw_end_time_sec`` and
    ``raw_timestamp_basis``), while the canonical ``start_time_sec`` and
    ``end_time_sec`` become source-absolute values.  Source-absolute results
    are returned as a deep copy without reinterpretation, preserving the
    historical route exactly.

    Unknown timestamp bases are not guessed: the parsed result is returned
    unchanged with an explicit ``UNSUPPORTED`` mapping status so a downstream
    projection can retain raw evidence while abstaining.  Malformed window
    bounds or relative segment offsets raise a contract error.
    """

    if not isinstance(parsed, Mapping):
        raise ProductionStructuredAnnotationError("parsed must be an object")
    start = _finite(
        window_start_seconds,
        field="window_start_seconds",
        minimum=0.0,
    )
    end = _finite(window_end_seconds, field="window_end_seconds", minimum=0.0)
    if end <= start:
        raise ProductionStructuredAnnotationError(
            "window_end_seconds must exceed window_start_seconds"
        )
    result = cast(dict[str, Any], _json_copy(parsed, field="parsed"))
    basis_value = result.get("timestamp_basis", TIMESTAMP_BASIS)
    if not isinstance(basis_value, str) or not basis_value.strip():
        raise ProductionStructuredAnnotationError("timestamp_basis must be non-empty text")
    basis = basis_value.strip()
    if basis == TIMESTAMP_BASIS:
        # Do not add provenance keys or alter existing source-absolute values;
        # this is the compatibility path for all historical sidecars.
        return result
    if basis != WINDOW_RELATIVE_TIMESTAMP_BASIS:
        result["timestamp_basis_status"] = "UNSUPPORTED"
        result["timestamp_mapping_status"] = "UNSUPPORTED"
        result["timestamp_mapping"] = {
            "version": TIMESTAMP_MAPPING_VERSION,
            "input_basis": basis,
            "output_basis": TIMESTAMP_BASIS,
            "window_start_seconds": start,
            "window_end_seconds": end,
            "status": "UNSUPPORTED",
        }
        return result

    duration = end - start
    raw_segments = result.get("segments", [])
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes, bytearray)):
        raise ProductionStructuredAnnotationError("parsed.segments must be an array")
    if not raw_segments:
        # A valid no-action response has no timestamps to reinterpret.  Keep
        # the explicit basis and report this separately from a failed mapping.
        result["timestamp_basis"] = TIMESTAMP_BASIS
        result["timestamp_basis_status"] = "MEASURED"
        result["timestamp_mapping_status"] = "NOT_APPLICABLE"
        result["timestamp_mapping"] = {
            "version": TIMESTAMP_MAPPING_VERSION,
            "input_basis": basis,
            "output_basis": TIMESTAMP_BASIS,
            "window_start_seconds": start,
            "window_end_seconds": end,
            "status": "NOT_APPLICABLE",
        }
        return result
    mapped_segments: list[dict[str, Any]] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise ProductionStructuredAnnotationError(f"parsed.segments[{index}] must be an object")
        segment = cast(dict[str, Any], _json_copy(raw_segment, field=f"parsed.segments[{index}]"))
        raw_start = segment.get("start_time_sec")
        raw_end = segment.get("end_time_sec")
        # Null boundaries are a valid explicit non-observation and remain null;
        # no fixed window boundary is fabricated.
        if raw_start is None and raw_end is None:
            segment.update(
                {
                    "raw_start_time_sec": None,
                    "raw_end_time_sec": None,
                    "raw_timestamp_basis": basis,
                    "mapped_start_time_sec": None,
                    "mapped_end_time_sec": None,
                    "timestamp_basis": TIMESTAMP_BASIS,
                    "timestamp_mapping_status": "NOT_MEASURED",
                    "timestamp_mapping_version": TIMESTAMP_MAPPING_VERSION,
                }
            )
            mapped_segments.append(segment)
            continue
        if raw_start is None or raw_end is None:
            raise ProductionStructuredAnnotationError(f"parsed.segments[{index}] PARTIAL_BOUNDARY")
        relative_start = _finite(
            raw_start,
            field=f"parsed.segments[{index}].start_time_sec",
            minimum=0.0,
        )
        relative_end = _finite(
            raw_end,
            field=f"parsed.segments[{index}].end_time_sec",
            minimum=0.0,
        )
        if relative_end <= relative_start:
            raise ProductionStructuredAnnotationError(
                f"parsed.segments[{index}] end_time_sec must exceed start_time_sec"
            )
        if relative_end > duration:
            raise ProductionStructuredAnnotationError(
                f"parsed.segments[{index}] RELATIVE_BOUNDARY_OUT_OF_WINDOW"
            )
        mapped_start = start + relative_start
        mapped_end = start + relative_end
        segment.update(
            {
                "raw_start_time_sec": relative_start,
                "raw_end_time_sec": relative_end,
                "raw_timestamp_basis": basis,
                "mapped_start_time_sec": mapped_start,
                "mapped_end_time_sec": mapped_end,
                "start_time_sec": mapped_start,
                "end_time_sec": mapped_end,
                "timestamp_basis": TIMESTAMP_BASIS,
                "timestamp_mapping_status": "MEASURED",
                "timestamp_mapping_version": TIMESTAMP_MAPPING_VERSION,
            }
        )
        mapped_segments.append(segment)
    result["segments"] = mapped_segments
    result["timestamp_basis"] = TIMESTAMP_BASIS
    result["timestamp_basis_status"] = "MEASURED"
    result["timestamp_mapping_status"] = "MAPPED"
    result["timestamp_mapping"] = {
        "version": TIMESTAMP_MAPPING_VERSION,
        "input_basis": basis,
        "output_basis": TIMESTAMP_BASIS,
        "window_start_seconds": start,
        "window_end_seconds": end,
        "status": "MAPPED",
    }
    return result


# Short compatibility aliases keep the helper easy to discover from runners
# without introducing a second implementation or wire contract.
map_relative_qwen_timestamps = map_qwen_relative_timestamps
map_qwen_timestamp_basis = map_qwen_relative_timestamps


def _model_rows_by_window(
    payload: Mapping[str, Any], *, model: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(_source_window_rows(payload, model=model)):
        window_id = _window_id(row, field=f"{model}.windows[{index}]")
        grouped.setdefault(window_id, []).append(row)
    return grouped


def _canonical_window_specs(
    sidecars: Mapping[str, Mapping[str, Any]],
    *,
    source_manifest: Mapping[str, Any] | None,
    window_specs: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build a source-bound window index from an explicit manifest or rows."""

    supplied = window_specs
    if supplied is None and source_manifest is not None:
        raw = source_manifest.get("windows")
        if raw is not None:
            supplied = _sequence(raw, field="source_manifest.windows")

    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: Mapping[str, Any], *, field: str, source_is_qwen: bool = False) -> None:
        window_id = _window_id(row, field=field)
        interval = _explicit_interval(row, field=field)
        observational_interval = False
        if interval is None and source_is_qwen:
            # Qwen's interval is an observation sampling interval, not an
            # action boundary.  It can seed a canonical window only when no
            # better source-bound interval exists.
            interval = _observational_interval(row, field=field)
            observational_interval = interval is not None
        if window_id in seen:
            existing = next(item for item in result if item["window_id"] == window_id)
            if (
                interval is not None
                and existing["interval"] is not None
                and any(abs(interval[i] - existing["interval"][i]) > 1e-3 for i in (0, 1))
            ):
                # Do not reject Qwen frame-overrun intervals when a canonical
                # WEMM/manifest interval already exists.
                if source_is_qwen and observational_interval:
                    return
                raise ProductionStructuredAnnotationError(
                    f"{field} interval does not bind existing window {window_id}"
                )
            if existing.get("ordinal") is None and "ordinal" in row:
                existing["ordinal"] = row["ordinal"]
            return
        ordinal = row.get("ordinal")
        if ordinal is not None and (
            isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0
        ):
            raise ProductionStructuredAnnotationError(
                f"{field}.ordinal must be a non-negative integer"
            )
        result.append({"window_id": window_id, "ordinal": ordinal, "interval": interval})
        seen.add(window_id)

    if supplied is not None:
        for index, raw in enumerate(supplied):
            add(_mapping(raw, field=f"window_specs[{index}]"), field=f"window_specs[{index}]")

    # Prefer explicit canonical intervals (WEMM/prod slots) before Qwen rows.
    for model in ("wemm", "mage", "qwen"):
        payload = sidecars.get(model)
        if payload is None:
            continue
        for index, row in enumerate(_source_window_rows(payload, model=model)):
            add(
                row,
                field=f"{model}.windows[{index}]",
                source_is_qwen=model == "qwen",
            )

    if not result:
        raise ProductionStructuredAnnotationError("no source-bound windows were found")
    # Assign stable ordinal order only when source rows did not provide one.
    for index, item in enumerate(result):
        if item["ordinal"] is None:
            item["ordinal"] = index
        if item["interval"] is None:
            raise ProductionStructuredAnnotationError(
                f"window {item['window_id']} has no source-bound interval"
            )
    result.sort(key=lambda item: (int(item["ordinal"]), item["window_id"]))
    return result


def _normalize_model_window(
    model: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    window_id: str,
    field: str,
) -> dict[str, Any]:
    if model == "mage" and not rows:
        return {
            "status": "BLOCKED",
            "measurement_status": "NOT_MEASURED",
            "candidates": [],
            "segments": [],
            "reason": "source-bound native codec/cache parity unavailable",
        }
    if not rows:
        return {
            "status": "NOT_RUN",
            "measurement_status": "NOT_MEASURED",
            "candidates": [],
            "segments": [],
        }

    # A model-output slot may be nested under ``model_outputs`` in the
    # production sidecar.  Flattening that wrapper is only structural; claims
    # remain untouched.
    source_rows: list[Mapping[str, Any]] = []
    for row in rows:
        nested_outputs = row.get("model_outputs")
        if isinstance(nested_outputs, Mapping) and isinstance(nested_outputs.get(model), Mapping):
            nested = dict(cast(Mapping[str, Any], nested_outputs[model]))
            for key in (
                "camera_id",
                "raw_text",
                "parsed_structured",
                "generation_warnings",
                "parse_error",
            ):
                if key in row and key not in nested:
                    nested[key] = row[key]
            source_rows.append(nested)
        elif isinstance(row.get("model"), Mapping):
            nested = dict(cast(Mapping[str, Any], row["model"]))
            for key in (
                "camera_id",
                "raw_text",
                "parsed_structured",
                "generation_warnings",
                "parse_error",
            ):
                if key in row and key not in nested:
                    nested[key] = row[key]
            source_rows.append(nested)
        else:
            source_rows.append(row)

    statuses = [
        _status(row.get("status"), field=f"{field}.status", default="NOT_RUN")
        for row in source_rows
    ]
    status = "SUCCEEDED" if any(item == "SUCCEEDED" for item in statuses) else statuses[0]
    if status == "BLOCKED":
        return {
            "status": "BLOCKED",
            "measurement_status": "NOT_MEASURED",
            "candidates": [],
            "segments": [],
            "reason": "source model reported BLOCKED",
        }

    candidates: list[Any] = []
    candidate_groups: list[dict[str, Any]] = []
    candidate_sources: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    parse_observations: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, row in enumerate(source_rows):
        row_field = f"{field}[{index}]"
        row_groups = _candidate_groups(row, model=model, field=row_field)
        row_candidates = cast(list[Any], row_groups[0]["candidates"]) if row_groups else []
        candidate_groups.extend(
            cast(
                list[dict[str, Any]],
                _json_copy(row_groups, field=f"{row_field}.candidate_groups"),
            )
        )
        if model == "qwen" and len(source_rows) > 1:
            # Keep one parsed claim per camera row.  This is not a new Top-K
            # ranking; it is a verbatim list of observed model claims.
            candidates.extend(row_candidates)
            candidate_sources.extend(_candidate_sources(row, field=row_field))
        elif index == 0:
            candidates = row_candidates
            candidate_sources.extend(_candidate_sources(row, field=row_field))
        explicit = _explicit_segments(row, field=row_field)
        if explicit:
            segments.extend(explicit)
        parsed = row.get("parsed_structured")
        if isinstance(parsed, Mapping):
            observation: dict[str, Any] = {}
            for key in ("camera_id", "parse_status", "errors", "warnings"):
                if key in row:
                    observation[key] = _json_copy(row[key], field=f"{row_field}.{key}")
                elif key in parsed:
                    observation[key] = _json_copy(parsed[key], field=f"{row_field}.parsed.{key}")
            for key in ("generation_warnings", "parse_error"):
                if key in row:
                    observation[key] = _json_copy(row[key], field=f"{row_field}.{key}")
            if observation:
                parse_observations.append(observation)
                for key in ("warnings", "generation_warnings"):
                    values = observation.get(key)
                    if isinstance(values, Sequence) and not isinstance(
                        values, (str, bytes, bytearray)
                    ):
                        warnings.extend(str(value) for value in values)
                if isinstance(observation.get("parse_error"), str):
                    warnings.append("STRUCTURED_PARSE_FAILURE")

    section: dict[str, Any] = {
        "status": status,
        "measurement_status": "MEASURED" if segments else "NOT_MEASURED",
        "candidates": candidates,
        "segments": segments,
    }
    if candidate_sources:
        section["candidate_sources"] = candidate_sources
    if candidate_groups:
        section["candidate_groups"] = candidate_groups
    if parse_observations:
        section["parse_observations"] = parse_observations
    if warnings:
        section["warnings"] = list(dict.fromkeys(warnings))
    # Preserve model-specific runtime provenance in a bounded, label-blind
    # shape without copying arbitrary source metadata or review fields.
    source_formats = [row.get("format") for row in rows if isinstance(row.get("format"), str)]
    if source_formats:
        section["source_formats"] = sorted(set(cast(list[str], source_formats)))
    return section


def build_structured_annotation_envelope(
    model_sidecars: Mapping[str, object],
    *,
    source_path: str | None = None,
    source_manifest: Mapping[str, Any] | str | Path | None = None,
    window_specs: Sequence[Mapping[str, Any]] | None = None,
    camera_count: int | None = None,
) -> dict[str, Any]:
    """Build a normalized envelope from already-recorded model sidecars.

    ``model_sidecars`` maps ``wemm``, ``qwen`` and optionally ``mage`` to a
    mapping or JSON path.  Missing model keys become ``NOT_RUN`` (or explicit
    Mage ``BLOCKED``).  ``source_manifest``/``window_specs`` are optional but
    recommended when combining sidecars whose window geometry is represented
    differently.  No fixed model window is emitted as a segment boundary.
    """

    if not isinstance(model_sidecars, Mapping):
        raise ProductionStructuredAnnotationError("model_sidecars must be an object")
    unexpected = sorted(set(model_sidecars) - set(MODEL_NAMES))
    if unexpected:
        raise ProductionStructuredAnnotationError(
            f"unsupported model sidecars: {', '.join(unexpected)}"
        )

    normalized_sidecars: dict[str, Mapping[str, Any]] = {}
    for model, raw in model_sidecars.items():
        normalized_sidecars[model] = _payload(raw, field=f"{model}_sidecar")

    manifest_payload: Mapping[str, Any] | None = None
    if source_manifest is not None:
        # A cohort manifest may legitimately contain a separate ``gold`` or
        # ``review`` section.  We read only source/window geometry and never
        # copy those fields into this label-blind envelope.
        manifest_payload = _payload(source_manifest, field="source_manifest", allow_gold=True)

    specs = _canonical_window_specs(
        normalized_sidecars,
        source_manifest=manifest_payload,
        window_specs=window_specs,
    )
    inferred_path = (
        source_path.strip() if isinstance(source_path, str) and source_path.strip() else None
    )
    if inferred_path is None:
        for payload in normalized_sidecars.values():
            inferred_path = _extract_source_path(payload)
            if inferred_path:
                break
    if inferred_path is None and manifest_payload is not None:
        inferred_path = _extract_source_path(manifest_payload)
    if inferred_path is None:
        raise ProductionStructuredAnnotationError(
            "source_path is required when sidecars do not expose a source path"
        )

    counts = [
        _extract_camera_count(payload)
        for payload in normalized_sidecars.values()
        if _extract_camera_count(payload) is not None
    ]
    resolved_camera_count = camera_count
    if resolved_camera_count is None and counts:
        resolved_camera_count = max(cast(list[int], counts))
    if resolved_camera_count is None:
        resolved_camera_count = 0
    if (
        isinstance(resolved_camera_count, bool)
        or not isinstance(resolved_camera_count, int)
        or resolved_camera_count < 0
    ):
        raise ProductionStructuredAnnotationError("camera_count must be a non-negative integer")

    grouped = {
        model: _model_rows_by_window(payload, model=model)
        for model, payload in normalized_sidecars.items()
    }
    output_windows: list[dict[str, Any]] = []
    for spec in specs:
        interval = cast(tuple[float, float], spec["interval"])
        window_id = cast(str, spec["window_id"])
        models: dict[str, Any] = {}
        for model in MODEL_NAMES:
            rows = grouped.get(model, {}).get(window_id, [])
            models[model] = _normalize_model_window(
                model,
                rows,
                window_id=window_id,
                field=f"windows[{window_id}].models.{model}",
            )
        output_windows.append(
            {
                "ordinal": int(spec["ordinal"]),
                "window_id": window_id,
                "start_time_sec": interval[0],
                "end_time_sec": interval[1],
                "timestamp_basis": TIMESTAMP_BASIS,
                "models": models,
            }
        )

    model_invoked = any(
        section["status"] not in {"NOT_RUN", "BLOCKED"}
        for window in output_windows
        for section in window["models"].values()
    )
    envelope: dict[str, Any] = {
        "format": STRUCTURED_ANNOTATION_ENVELOPE_VERSION,
        "authority": AUTHORITY,
        "production_eligible": False,
        "quality": {
            "measurement_status": "NOT_MEASURED",
            "quality_claim": False,
            "reason": (
                "structured fields and action boundaries require explicit model "
                "evidence and external review"
            ),
        },
        "source": {
            "path": inferred_path,
            "window_count": len(output_windows),
            "camera_count": resolved_camera_count,
        },
        "windows": output_windows,
        "contract": {
            "model_names": list(MODEL_NAMES),
            "candidate_claims_are_model_only": True,
            "gold_is_external": True,
            "gold_fields_included": False,
            "structured_label_representation": "value_status_object",
            "timestamp_basis": TIMESTAMP_BASIS,
            "fixed_window_is_not_action_boundary": True,
        },
        "controls": {
            "model_invoked": model_invoked,
            "source_media_decoded": False,
            "gold_included": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
        },
    }
    return normalize_structured_annotation_envelope(envelope)


def _validate_field_record(value: object, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    status = _text(record.get("status"), field=f"{field}.status").upper()
    if status not in _FIELD_STATUS_SET:
        raise ProductionStructuredAnnotationError(
            f"{field}.status must be one of {', '.join(FIELD_STATUSES)}"
        )
    if status == "MEASURED":
        if record.get("value") is None:
            raise ProductionStructuredAnnotationError(f"{field}.value cannot be null when measured")
        value_copy = _json_copy(record.get("value"), field=f"{field}.value")
    else:
        # Missing/unobservable fields must not smuggle a guessed value.
        if record.get("value") is not None and record.get("value") != "UNSPECIFIED":
            raise ProductionStructuredAnnotationError(f"{field}.value must be null for {status}")
        value_copy = record.get("value")
    return {"value": value_copy, "status": status}


def _validate_segment(
    value: object,
    *,
    field: str,
    window_start: float | None = None,
    window_end: float | None = None,
) -> dict[str, Any]:
    segment = _mapping(value, field=field)
    start = segment.get("start_time_sec")
    end = segment.get("end_time_sec")
    start_value: float | None
    end_value: float | None
    boundary_status = _text(
        segment.get("boundary_status"), field=f"{field}.boundary_status"
    ).upper()
    if boundary_status not in {"MEASURED", "NOT_MEASURED", "NOT_OBSERVABLE"}:
        raise ProductionStructuredAnnotationError(f"{field}.boundary_status is unsupported")
    if boundary_status == "MEASURED":
        start_value = _finite(start, field=f"{field}.start_time_sec", minimum=0.0)
        end_value = _finite(end, field=f"{field}.end_time_sec", minimum=0.0)
        if end_value <= start_value:
            raise ProductionStructuredAnnotationError(f"{field} end must exceed start")
        if (
            window_start is not None
            and window_end is not None
            and (start_value < window_start or end_value > window_end)
        ):
            # Keep the explicit claim and make the boundary failure visible;
            # never shift a source-absolute timestamp into the window.
            start_value, end_value = None, None
            boundary_status = "NOT_MEASURED"
            boundary_error = "SEGMENT_BOUNDARY_OUTSIDE_WINDOW"
        else:
            boundary_error = None
    else:
        if start is not None or end is not None:
            raise ProductionStructuredAnnotationError(
                f"{field} non-measured boundaries must be null"
            )
        start_value, end_value = None, None
        boundary_error = None
    labels = _mapping(segment.get("structured_labels"), field=f"{field}.structured_labels")
    if set(labels) != set(STRUCTURED_FIELDS):
        raise ProductionStructuredAnnotationError(
            f"{field}.structured_labels must contain exactly {', '.join(STRUCTURED_FIELDS)}"
        )
    labels_copy = {
        name: _validate_field_record(labels[name], field=f"{field}.structured_labels.{name}")
        for name in STRUCTURED_FIELDS
    }
    confidence_status = _text(
        segment.get("confidence_status"), field=f"{field}.confidence_status"
    ).upper()
    confidence = segment.get("confidence")
    if confidence_status not in _FIELD_STATUS_SET:
        raise ProductionStructuredAnnotationError(f"{field}.confidence_status is unsupported")
    if confidence_status == "MEASURED":
        confidence_value = _finite(confidence, field=f"{field}.confidence", minimum=0.0)
        if confidence_value > 1:
            raise ProductionStructuredAnnotationError(f"{field}.confidence must be between 0 and 1")
    else:
        if confidence is not None:
            raise ProductionStructuredAnnotationError(
                f"{field}.confidence must be null for {confidence_status}"
            )
        confidence_value = None
    evidence_status = _text(
        segment.get("evidence_status"), field=f"{field}.evidence_status"
    ).upper()
    if evidence_status not in _FIELD_STATUS_SET:
        raise ProductionStructuredAnnotationError(f"{field}.evidence_status is unsupported")
    evidence = segment.get("evidence")
    if evidence_status == "MEASURED":
        evidence_value = _json_copy(evidence, field=f"{field}.evidence")
    else:
        if evidence not in (None, []):
            raise ProductionStructuredAnnotationError(
                f"{field}.evidence must be empty for {evidence_status}"
            )
        evidence_value = []
    status = _text(segment.get("status"), field=f"{field}.status").upper()
    if status not in _SEGMENT_STATUS_SET:
        raise ProductionStructuredAnnotationError(f"{field}.status is unsupported")
    result: dict[str, Any] = {
        "start_time_sec": start_value,
        "end_time_sec": end_value,
        "boundary_status": boundary_status,
        "structured_labels": labels_copy,
        "confidence": confidence_value,
        "confidence_status": confidence_status,
        "evidence": evidence_value,
        "evidence_status": evidence_status,
        "status": status,
    }
    # Preserve benchmark-local timestamp mapping provenance in a normalized
    # envelope.  The canonical interval above is always the mapped/source
    # interval; raw relative offsets remain available for audit and comparison.
    for key in (
        "raw_start_time_sec",
        "raw_end_time_sec",
        "raw_timestamp_basis",
        "mapped_start_time_sec",
        "mapped_end_time_sec",
        "timestamp_basis",
        "timestamp_mapping_status",
        "timestamp_mapping_version",
        "timestamp_mapping",
    ):
        if key in segment:
            result[key] = _json_copy(segment[key], field=f"{field}.{key}")
    if boundary_error is not None:
        result["boundary_error"] = boundary_error
        if status == "MEASURED":
            result["status"] = "FAILED"
    elif "boundary_error" in segment:
        result["boundary_error"] = _text(
            segment.get("boundary_error"), field=f"{field}.boundary_error"
        )
    if "segment_id" in segment:
        result["segment_id"] = _text(segment.get("segment_id"), field=f"{field}.segment_id")
    return result


def normalize_structured_annotation_envelope(
    value: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate and canonicalize a structured annotation envelope.

    The function is intentionally label-blind: it validates structure and
    provenance only.  It never evaluates a candidate against gold.
    """

    payload = _payload(value, field="envelope")
    if payload.get("format") != STRUCTURED_ANNOTATION_ENVELOPE_VERSION:
        raise ProductionStructuredAnnotationError(
            "unsupported structured annotation envelope format"
        )
    if payload.get("authority") != AUTHORITY:
        raise ProductionStructuredAnnotationError(
            "envelope authority must be LOCAL_NONPRODUCTION_ONLY"
        )
    if payload.get("production_eligible") is not False:
        raise ProductionStructuredAnnotationError(
            "structured annotation envelope is non-production only"
        )
    _assert_no_gold_fields(payload, field="envelope")
    source = _mapping(payload.get("source"), field="envelope.source")
    source_path = _text(source.get("path"), field="envelope.source.path")
    window_count = source.get("window_count")
    camera_count = source.get("camera_count")
    if isinstance(window_count, bool) or not isinstance(window_count, int) or window_count < 0:
        raise ProductionStructuredAnnotationError(
            "envelope.source.window_count must be non-negative"
        )
    if isinstance(camera_count, bool) or not isinstance(camera_count, int) or camera_count < 0:
        raise ProductionStructuredAnnotationError(
            "envelope.source.camera_count must be non-negative"
        )
    windows_raw = _sequence(payload.get("windows"), field="envelope.windows")
    if len(windows_raw) != window_count:
        raise ProductionStructuredAnnotationError(
            "envelope.source.window_count does not match windows"
        )
    windows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(windows_raw):
        window = _mapping(raw, field=f"envelope.windows[{index}]")
        window_id = _window_id(window, field=f"envelope.windows[{index}]")
        if window_id in seen:
            raise ProductionStructuredAnnotationError(f"duplicate envelope window_id: {window_id}")
        seen.add(window_id)
        ordinal = window.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ProductionStructuredAnnotationError(
                f"envelope.windows[{index}].ordinal is invalid"
            )
        start = _finite(
            window.get("start_time_sec"),
            field=f"envelope.windows[{index}].start_time_sec",
            minimum=0.0,
        )
        end = _finite(
            window.get("end_time_sec"), field=f"envelope.windows[{index}].end_time_sec", minimum=0.0
        )
        if end <= start:
            raise ProductionStructuredAnnotationError(
                f"envelope.windows[{index}] end must exceed start"
            )
        timestamp_basis = window.get("timestamp_basis", TIMESTAMP_BASIS)
        if timestamp_basis != TIMESTAMP_BASIS:
            raise ProductionStructuredAnnotationError(
                f"envelope.windows[{index}].timestamp_basis must be {TIMESTAMP_BASIS}"
            )
        models_raw = _mapping(window.get("models"), field=f"envelope.windows[{index}].models")
        if set(models_raw) != set(MODEL_NAMES):
            raise ProductionStructuredAnnotationError(
                f"envelope.windows[{index}].models must contain exactly {', '.join(MODEL_NAMES)}"
            )
        models: dict[str, Any] = {}
        for model in MODEL_NAMES:
            section = _mapping(
                models_raw.get(model), field=f"envelope.windows[{index}].models.{model}"
            )
            status = _status(
                section.get("status"), field=f"envelope.windows[{index}].models.{model}.status"
            )
            measurement = _text(
                section.get("measurement_status"),
                field=f"envelope.windows[{index}].models.{model}.measurement_status",
            ).upper()
            if measurement not in {"MEASURED", "NOT_MEASURED"}:
                raise ProductionStructuredAnnotationError("model measurement_status is unsupported")
            candidates = _sequence(
                section.get("candidates"),
                field=f"envelope.windows[{index}].models.{model}.candidates",
            )
            candidates_copy = cast(
                list[Any],
                _json_copy(
                    candidates, field=f"envelope.windows[{index}].models.{model}.candidates"
                ),
            )
            segments_raw = _sequence(
                section.get("segments"),
                field=f"envelope.windows[{index}].models.{model}.segments",
            )
            segments = [
                _validate_segment(
                    segment,
                    field=f"envelope.windows[{index}].models.{model}.segments[{seg_index}]",
                    window_start=start,
                    window_end=end,
                )
                for seg_index, segment in enumerate(segments_raw)
            ]
            if segments and all(segment.get("status") == "FAILED" for segment in segments):
                # Explicit claims whose source-absolute boundaries fall outside
                # this window are retained for audit, but cannot count as a
                # measured structured observation.
                measurement = "NOT_MEASURED"
            if status == "BLOCKED" and (candidates_copy or segments):
                raise ProductionStructuredAnnotationError(
                    "BLOCKED model section cannot contain predictions"
                )
            if measurement == "MEASURED" and not segments:
                raise ProductionStructuredAnnotationError(
                    "MEASURED model section requires segments"
                )
            if measurement == "NOT_MEASURED" and any(
                segment.get("status") != "FAILED" for segment in segments
            ):
                raise ProductionStructuredAnnotationError(
                    "model sections with segments must be MEASURED"
                )
            model_result: dict[str, Any] = {
                "status": status,
                "measurement_status": measurement,
                "candidates": candidates_copy,
                "segments": segments,
            }
            for key in (
                "reason",
                "candidate_sources",
                "candidate_groups",
                "parse_observations",
                "warnings",
                "source_formats",
            ):
                if key in section:
                    model_result[key] = _json_copy(section[key], field=f"envelope...{model}.{key}")
            models[model] = model_result
        windows.append(
            {
                "ordinal": ordinal,
                "window_id": window_id,
                "start_time_sec": start,
                "end_time_sec": end,
                "timestamp_basis": TIMESTAMP_BASIS,
                "models": models,
            }
        )

    quality = _mapping(payload.get("quality"), field="envelope.quality")
    if quality.get("measurement_status") != "NOT_MEASURED":
        raise ProductionStructuredAnnotationError(
            "structured envelope quality must remain NOT_MEASURED"
        )
    if quality.get("quality_claim") is not False:
        raise ProductionStructuredAnnotationError(
            "structured envelope quality_claim must remain false"
        )
    contract = _mapping(payload.get("contract"), field="envelope.contract")
    if (
        tuple(_sequence(contract.get("model_names"), field="envelope.contract.model_names"))
        != MODEL_NAMES
    ):
        raise ProductionStructuredAnnotationError("envelope contract model_names are invalid")
    if contract.get("candidate_claims_are_model_only") is not True:
        raise ProductionStructuredAnnotationError("candidate claims must remain model-only")
    if contract.get("timestamp_basis", TIMESTAMP_BASIS) != TIMESTAMP_BASIS:
        raise ProductionStructuredAnnotationError(
            f"envelope contract timestamp_basis must be {TIMESTAMP_BASIS}"
        )
    if (
        contract.get("gold_is_external") is not True
        or contract.get("gold_fields_included") is not False
    ):
        raise ProductionStructuredAnnotationError("envelope must keep gold external and excluded")
    controls = _mapping(payload.get("controls"), field="envelope.controls")
    for key in (
        "model_invoked",
        "source_media_decoded",
        "gold_included",
        "predictions_copied_to_gold",
        "ontology_modified",
        "mapper_modified",
        "training_invoked",
    ):
        if not isinstance(controls.get(key), bool):
            raise ProductionStructuredAnnotationError(f"envelope.controls.{key} must be boolean")
    for key in (
        "gold_included",
        "predictions_copied_to_gold",
        "ontology_modified",
        "mapper_modified",
        "training_invoked",
    ):
        if controls[key] is not False:
            raise ProductionStructuredAnnotationError(f"envelope.controls.{key} must remain false")

    result = {
        "format": STRUCTURED_ANNOTATION_ENVELOPE_VERSION,
        "authority": AUTHORITY,
        "production_eligible": False,
        "quality": {
            "measurement_status": "NOT_MEASURED",
            "quality_claim": False,
            **(
                {"reason": _json_copy(quality["reason"], field="envelope.quality.reason")}
                if "reason" in quality
                else {}
            ),
        },
        "source": {
            "path": source_path,
            "window_count": window_count,
            "camera_count": camera_count,
        },
        "windows": windows,
        "contract": {
            "model_names": list(MODEL_NAMES),
            "candidate_claims_are_model_only": True,
            "gold_is_external": True,
            "gold_fields_included": False,
            "structured_label_representation": contract.get(
                "structured_label_representation", "value_status_object"
            ),
            "timestamp_basis": TIMESTAMP_BASIS,
            "fixed_window_is_not_action_boundary": bool(
                contract.get("fixed_window_is_not_action_boundary", True)
            ),
        },
        "controls": {
            key: bool(controls[key])
            for key in (
                "model_invoked",
                "source_media_decoded",
                "gold_included",
                "predictions_copied_to_gold",
                "ontology_modified",
                "mapper_modified",
                "training_invoked",
            )
        },
    }
    return copy.deepcopy(result)


validate_structured_annotation_envelope = normalize_structured_annotation_envelope
build_production_structured_annotation_envelope = build_structured_annotation_envelope
normalize_production_structured_annotation_envelope = normalize_structured_annotation_envelope
normalize_model_candidate = _candidate_list


__all__ = [
    "AUTHORITY",
    "FIELD_STATUSES",
    "MODEL_NAMES",
    "MODEL_STATUSES",
    "PRODUCTION_STRUCTURED_ANNOTATION_FORMAT",
    "QWEN_PRODUCTION_COARSE_FORCED_PROMPT",
    "QWEN_PRODUCTION_COARSE_FORCED_PROMPT_VERSION",
    "QWEN_PRODUCTION_COARSE_FORCED_STRICT_BOUNDARIES_PROMPT",
    "QWEN_PRODUCTION_COARSE_FORCED_STRICT_BOUNDARIES_PROMPT_VERSION",
    "QWEN_PRODUCTION_COARSE_PROMPT",
    "QWEN_PRODUCTION_COARSE_PROMPT_VERSION",
    "QWEN_PRODUCTION_COARSE_STRICT_BOUNDARIES_PROMPT",
    "QWEN_PRODUCTION_COARSE_STRICT_BOUNDARIES_PROMPT_VERSION",
    "QWEN_PRODUCTION_IDENTITY_DISAMBIGUATED_PROMPT",
    "QWEN_PRODUCTION_IDENTITY_DISAMBIGUATED_PROMPT_VERSION",
    "QWEN_PRODUCTION_IDENTITY_ONLY_PROMPT",
    "QWEN_PRODUCTION_IDENTITY_ONLY_PROMPT_VERSION",
    "QWEN_STRUCTURED_NATIVE_PROMPT",
    "QWEN_STRUCTURED_NATIVE_PROMPT_VERSION",
    "QWEN_STRUCTURED_NATIVE_RELATIVE_PROMPT",
    "QWEN_STRUCTURED_NATIVE_RELATIVE_PROMPT_VERSION",
    "QWEN_STRUCTURED_NATIVE_RELATIVE_STRICT_BOUNDARIES_PROMPT",
    "QWEN_STRUCTURED_NATIVE_RELATIVE_STRICT_BOUNDARIES_PROMPT_VERSION",
    "QWEN_STRUCTURED_NATIVE_STRICT_BOUNDARIES_PROMPT",
    "QWEN_STRUCTURED_NATIVE_STRICT_BOUNDARIES_PROMPT_VERSION",
    "RELATIVE_TIMESTAMP_BASIS",
    "SEGMENT_STATUSES",
    "STRUCTURED_ANNOTATION_ENVELOPE_VERSION",
    "STRUCTURED_FIELDS",
    "SUPPORTED_TIMESTAMP_BASES",
    "TIMESTAMP_BASIS",
    "TIMESTAMP_MAPPING_VERSION",
    "WINDOW_RELATIVE_TIMESTAMP_BASIS",
    "ProductionStructuredAnnotationContractError",
    "ProductionStructuredAnnotationError",
    "build_production_structured_annotation_envelope",
    "build_structured_annotation_envelope",
    "load_json",
    "map_qwen_relative_timestamps",
    "map_qwen_timestamp_basis",
    "map_relative_qwen_timestamps",
    "normalize_model_candidate",
    "normalize_production_structured_annotation_envelope",
    "normalize_structured_annotation_envelope",
    "parse_qwen_identity_only_output",
    "parse_qwen_structured_output",
    "validate_structured_annotation_envelope",
]
