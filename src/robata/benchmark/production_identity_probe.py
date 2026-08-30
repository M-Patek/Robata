"""Post-hoc diagnostics for the Qwen production identity-only sidecar.

The identity-only arm is a deliberately small probe.  It asks Qwen for an
action identity without timestamps or segment structure so that action
recognition can be inspected independently from temporal grounding.  This
module evaluates an already recorded sidecar; it never invokes a model,
decodes media, edits an ontology/Mapper, or writes labels.

The optional owner-confirmation input is *only* a surrogate reference.  It is
accepted for an exploratory overlap table when it explicitly declares that
official gold has not been established.  The evaluator always keeps official
quality ``NOT_MEASURED`` and never promotes a surrogate row to gold.
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
from typing import Any, Final, cast

PRODUCTION_IDENTITY_PROBE_VERSION: Final = "robata-production-qwen-identity-probe-v1"
PRODUCTION_IDENTITY_CANDIDATE_PROJECTION_VERSION: Final = (
    "robata-production-qwen-identity-candidate-projection-v1"
)
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
DIAGNOSTIC_STATUS: Final = "DIAGNOSTIC_ONLY"
SURROGATE_NON_GOLD: Final = "SURROGATE_NON_GOLD"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
IDENTITY_PROFILE: Final = "production_identity_only"
IDENTITY_PROFILES: Final = frozenset(
    {"production_identity_only", "production_identity_disambiguated"}
)
EXPECTED_SIDECAR_FORMAT: Final = "robata-production-qwen-structured-native-shadow-v1"

IDENTITY_ACTIONS: Final = (
    "pick up garment",
    "spread garment",
    "flatten garment",
    "adjust garment",
    "smooth garment",
    "fold garment",
    "none visible",
    "uncertain",
)
POSITIVE_ACTIONS: Final = frozenset(IDENTITY_ACTIONS[:6])
ABSTAIN_ACTIONS: Final = frozenset(IDENTITY_ACTIONS[6:])
# Values are compared after ``_normalise`` (which turns underscores into
# spaces), so keep the canonical normalized spellings here.
NATIVE_INPUT_MODES: Final = frozenset({"native video", "complete native video"})


class ProductionIdentityProbeError(ValueError):
    """Raised when a sidecar or surrogate reference is malformed."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionIdentityProbeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionIdentityProbeError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionIdentityProbeError(f"{field} must be a string")
    result = unicodedata.normalize("NFKC", value).strip()
    if not result and not allow_empty:
        raise ProductionIdentityProbeError(f"{field} must be non-empty")
    return result


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _finite_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        return None
    return result


def _copy_json(value: object, *, field: str) -> Any:
    """Copy JSON-compatible provenance without deriving an identity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionIdentityProbeError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionIdentityProbeError(f"{field} keys must be strings")
            copied[key] = _copy_json(child, field=f"{field}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionIdentityProbeError(f"{field} must be JSON-compatible")


def _load_payload(value: Mapping[str, Any] | str | Path, *, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(_mapping(value, field=field))
    path = Path(value).expanduser()
    try:
        decoded = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionIdentityProbeError(f"could not load {field} {path}: {exc}") from exc
    return dict(_mapping(decoded, field=str(path)))


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON object for the command-line evaluator."""

    return _load_payload(path, field="JSON input")


def _identity_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Choose the recorded parsed identity, without repairing it."""

    for key in ("parsed_identity", "raw_identity"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return row


def _parse_status(identity: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    value = identity.get("parse_status", row.get("identity_status"))
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return "NOT_RECORDED"


def _evidence(identity: Mapping[str, Any]) -> list[str]:
    value = identity.get("evidence")
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _native_completeness(row: Mapping[str, Any]) -> dict[str, Any]:
    """Measure native-video metadata independently of semantic parsing."""

    complete_flag = row.get("native_video_complete") is True
    input_mode = _normalise(row.get("input_mode"))
    native_mode = input_mode in NATIVE_INPUT_MODES
    visual = row.get("visual_input")
    visual_map = visual if isinstance(visual, Mapping) else None
    sequence_value = visual_map.get("content_sequence") if visual_map else None
    content_has_video = bool(
        isinstance(sequence_value, Sequence)
        and not isinstance(sequence_value, (str, bytes, bytearray))
        and any(_normalise(item) == "video" for item in sequence_value)
    )
    shapes = visual_map.get("processor_tensor_shapes") if visual_map else None
    visual_metadata = isinstance(visual_map, Mapping) and isinstance(shapes, Mapping)
    # The explicit runtime flag is authoritative.  Metadata checks are
    # reported separately so an older sidecar is diagnosable rather than
    # silently converted into a complete native claim.
    complete = complete_flag and native_mode and content_has_video and visual_metadata
    return {
        "complete": complete,
        "complete_flag": complete_flag,
        "native_input_mode": native_mode,
        "content_has_video": content_has_video,
        "visual_metadata": visual_metadata,
        "input_mode": row.get("input_mode"),
    }


def _reference_action(segment: Mapping[str, Any]) -> str | None:
    labels = segment.get("structured_labels", segment.get("labels"))
    source: Mapping[str, Any] = labels if isinstance(labels, Mapping) else segment
    verb = _normalise(source.get("verb", source.get("verb_code")))
    noun = _normalise(source.get("noun", source.get("object")))
    if not verb:
        return None
    # The owner-confirmation vocabulary intentionally uses garment as the
    # production noun.  This composition is lexical only; no semantic alias
    # table is applied.
    return " ".join(part for part in (verb, noun) if part)


def _reference_windows(reference: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_rows = reference.get("windows", reference.get("items", reference.get("decisions")))
    rows = _sequence(raw_rows, field="owner_confirmation.windows")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"owner_confirmation.windows[{index}]")
        window_id = _text(
            row.get("window_id"),
            field=f"owner_confirmation.windows[{index}].window_id",
        )
        if window_id in result:
            raise ProductionIdentityProbeError(f"duplicate reference window: {window_id}")
        # Terra review artifacts historically use ``recommendation`` while
        # compact owner-confirmation fixtures use ``decision``.  Preserve the
        # source decision instead of defaulting an omitted ``decision`` to
        # accept (which would incorrectly admit ABSTAIN rows).
        decision = row.get(
            "decision",
            row.get("recommendation", row.get("review_decision")),
        )
        segments_value = row.get("segments")
        gold = row.get("gold")
        if isinstance(gold, Mapping):
            if decision is None:
                decision = gold.get("status")
            if segments_value is None:
                segments_value = gold.get("segments", [])
        # An omitted/empty decision is not an implicit acceptance.  Review
        # artifacts may omit the field while a human decision is still
        # pending; treating that omission as ``accept`` would make the
        # optional surrogate overlap silently over-count eligible windows.
        decision_text = _normalise(decision) or "not_recorded"
        segments = _sequence(segments_value or [], field=f"{window_id}.segments")
        actions: list[str] = []
        for segment_index, raw_segment in enumerate(segments):
            segment = _mapping(raw_segment, field=f"{window_id}.segments[{segment_index}]")
            action = _reference_action(segment)
            if action and action not in actions:
                actions.append(action)
        eligible = decision_text in {"accept", "accepted", "edit", "split"} and bool(actions)
        result[window_id] = {
            "window_id": window_id,
            "decision": decision_text,
            "actions": actions,
            "eligible": eligible,
        }
    return result


def _validate_surrogate_reference(reference: Mapping[str, Any]) -> None:
    official = _normalise(reference.get("official_gold_status"))
    if official and official not in {"not established", "not measured"}:
        raise ProductionIdentityProbeError(
            "owner confirmation must explicitly remain non-gold (official_gold_status)"
        )
    if reference.get("production_eligible") is True:
        raise ProductionIdentityProbeError("owner confirmation cannot be production eligible")
    if reference.get("accepted_as_gold") is True or reference.get("official_gold") is True:
        raise ProductionIdentityProbeError("owner confirmation cannot claim official gold")
    status = _normalise(reference.get("status"))
    if "gold" in status and "non" not in status and "not established" not in status:
        raise ProductionIdentityProbeError("owner confirmation status appears to claim gold")


def _surrogate_overlap(
    rows: Sequence[Mapping[str, Any]], reference: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_surrogate_reference(reference)
    refs = _reference_windows(reference)
    # Multi-camera sidecars repeat each bounded window once per camera.  Keep
    # every row so an any-camera match is not lost to dictionary overwrite.
    by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row.get("window_id")), []).append(row)
    eligible = [item for item in refs.values() if item["eligible"]]
    hits = 0
    available = 0
    primary_hits = 0
    primary_available = 0
    per_window: list[dict[str, Any]] = []
    for item in eligible:
        predicted_rows = by_id.get(item["window_id"], [])
        predicted_actions: list[str] = []
        camera_actions: list[str] = []
        for predicted_row in predicted_rows:
            identity = _identity_mapping(predicted_row)
            predicted = _normalise(identity.get("action"))
            if predicted in IDENTITY_ACTIONS and predicted not in predicted_actions:
                predicted_actions.append(predicted)
            if predicted in IDENTITY_ACTIONS:
                camera_actions.append(predicted)
        expected = item["actions"][0]
        has_prediction = bool(predicted_actions)
        if has_prediction:
            available += 1
        exact = expected in predicted_actions
        if exact:
            hits += 1
        action_counts = Counter(camera_actions)
        max_count = max(action_counts.values(), default=0)
        primary_candidates = sorted(
            action for action, count in action_counts.items() if count == max_count
        )
        primary_action = primary_candidates[0] if len(primary_candidates) == 1 else None
        primary_is_available = primary_action is not None
        primary_exact = primary_action == expected
        if primary_is_available:
            primary_available += 1
        if primary_exact:
            primary_hits += 1
        per_window.append(
            {
                "window_id": item["window_id"],
                "expected_action": expected,
                "predicted_action": predicted_actions[0] if predicted_actions else None,
                "predicted_actions": predicted_actions,
                "camera_action_counts": dict(action_counts),
                "primary_action": primary_action,
                "primary_action_tie": len(primary_candidates) > 1,
                "primary_exact_match": primary_exact,
                "camera_count": len(predicted_rows),
                "prediction_available": has_prediction,
                "exact_match": exact,
            }
        )
    denominator = len(eligible)
    return {
        "status": SURROGATE_NON_GOLD,
        "measurement_status": SURROGATE_NON_GOLD,
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold": False,
        "reference_class": SURROGATE_NON_GOLD,
        "reference_window_count": len(refs),
        "eligible_window_count": denominator,
        "prediction_available_count": available,
        "exact_action_hits": hits,
        "exact_action_rate": hits / denominator if denominator else None,
        "prediction_coverage": available / denominator if denominator else None,
        "any_camera_action_hits": hits,
        "any_camera_action_rate": hits / denominator if denominator else None,
        "primary_action_hits": primary_hits,
        "primary_action_rate": primary_hits / denominator if denominator else None,
        "primary_action_available_count": primary_available,
        "primary_action_coverage": primary_available / denominator if denominator else None,
        "per_window": per_window,
    }


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def evaluate_production_identity_probe(
    sidecar: Mapping[str, Any] | str | Path,
    owner_confirmation: Mapping[str, Any] | str | Path | None = None,
    *,
    input_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate a recorded ``production_identity_only`` sidecar.

    All rates are operational diagnostics over the listed sidecar rows.  They
    are not accuracy unless the optional reference overlap is used, and that
    overlap is explicitly marked ``SURROGATE_NON_GOLD``.
    """

    document = _load_payload(sidecar, field="sidecar")
    if document.get("production_eligible") is True:
        raise ProductionIdentityProbeError("identity sidecar must be production-ineligible")
    declared_format = document.get("format")
    if declared_format is not None and declared_format != EXPECTED_SIDECAR_FORMAT:
        raise ProductionIdentityProbeError(f"sidecar format must be {EXPECTED_SIDECAR_FORMAT!r}")
    model = document.get("model")
    model_map = model if isinstance(model, Mapping) else {}
    declared_profile = model_map.get("label_profile")
    if declared_profile is not None and declared_profile not in IDENTITY_PROFILES:
        raise ProductionIdentityProbeError(
            f"sidecar label_profile must be one of {sorted(IDENTITY_PROFILES)!r}"
        )
    raw_rows = _sequence(document.get("windows"), field="sidecar.windows")
    if not raw_rows:
        raise ProductionIdentityProbeError("sidecar.windows must not be empty")

    rows: list[dict[str, Any]] = []
    seen_ids: set[tuple[str, str | None]] = set()
    parse_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    confidence_values: list[float] = []
    runtime_success = 0
    native_complete = 0
    native_flag = 0
    native_mode = 0
    native_visual = 0
    evidence_present = 0
    parsed_evidence_present = 0
    recognized_actions = 0
    parsed_recognized_actions = 0
    abstain_count = 0
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw, field=f"sidecar.windows[{index}]")
        window_id = _text(row.get("window_id"), field=f"sidecar.windows[{index}].window_id")
        camera_id = row.get("camera_id")
        camera_key = camera_id.strip() if isinstance(camera_id, str) and camera_id.strip() else None
        row_key = (window_id, camera_key)
        if row_key in seen_ids:
            raise ProductionIdentityProbeError(
                f"duplicate sidecar window/camera: {window_id}/{camera_key or '-'}"
            )
        seen_ids.add(row_key)
        identity = _identity_mapping(row)
        parse_status = _parse_status(identity, row)
        parse_counts[parse_status] += 1
        action_raw = identity.get("action")
        action = _normalise(action_raw) if isinstance(action_raw, str) else ""
        action_valid = action in IDENTITY_ACTIONS
        if action_valid:
            action_counts[action] += 1
            recognized_actions += 1
            if parse_status == "PARSED":
                parsed_recognized_actions += 1
            if action in ABSTAIN_ACTIONS:
                abstain_count += 1
        evidence = _evidence(identity)
        has_evidence = bool(evidence)
        if has_evidence:
            evidence_present += 1
            if parse_status == "PARSED":
                parsed_evidence_present += 1
        confidence = _finite_confidence(identity.get("confidence"))
        if confidence is not None:
            confidence_values.append(confidence)
        native = _native_completeness(row)
        native_complete += int(native["complete"])
        native_flag += int(native["complete_flag"])
        native_mode += int(native["native_input_mode"])
        native_visual += int(native["visual_metadata"])
        if _normalise(row.get("status")) in {"succeeded", "success"}:
            runtime_success += 1
        rows.append(
            {
                "window_id": window_id,
                "camera_id": camera_key,
                "status": row.get("status"),
                "parse_status": parse_status,
                "action": action or None,
                "action_valid": action_valid,
                "abstention": action in ABSTAIN_ACTIONS,
                "evidence": evidence,
                "evidence_present": has_evidence,
                "confidence": confidence,
                "confidence_valid": confidence is not None,
                "native_completeness": native,
                "raw_identity": _copy_json(identity, field=f"{window_id}.identity"),
            }
        )

    total = len(rows)
    parsed = parse_counts.get("PARSED", 0)
    invalid = total - parsed
    reference_overlap: dict[str, Any]
    reference_projection: dict[str, Any] | None = None
    if owner_confirmation is None:
        reference_overlap = {
            "status": "NOT_SUPPLIED",
            "measurement_status": OFFICIAL_QUALITY_STATUS,
            "official_quality_status": OFFICIAL_QUALITY_STATUS,
            "official_gold": False,
            "reference_class": None,
            "eligible_window_count": 0,
            "exact_action_hits": 0,
            "exact_action_rate": None,
            "prediction_coverage": None,
            "per_window": [],
        }
    else:
        reference_map = _load_payload(owner_confirmation, field="owner_confirmation")
        reference_projection = {
            "format": reference_map.get("format"),
            "status": reference_map.get("status"),
            "official_gold_status": reference_map.get("official_gold_status", "NOT_ESTABLISHED"),
            "human_adjudication": reference_map.get("human_adjudication"),
            "production_eligible": reference_map.get("production_eligible"),
        }
        reference_overlap = _surrogate_overlap(rows, reference_map)

    confidence_summary = {
        "valid_count": len(confidence_values),
        "denominator": total,
        "rate": _rate(len(confidence_values), total),
        "mean": statistics.fmean(confidence_values) if confidence_values else None,
        "median": statistics.median(confidence_values) if confidence_values else None,
        "minimum": min(confidence_values) if confidence_values else None,
        "maximum": max(confidence_values) if confidence_values else None,
    }
    metrics: dict[str, Any] = {
        "window_count": total,
        "runtime": {
            "succeeded": runtime_success,
            "failed_or_missing": total - runtime_success,
            "denominator": total,
            "rate": _rate(runtime_success, total),
        },
        "parse": {
            "parsed": parsed,
            "invalid": invalid,
            "not_recorded": parse_counts.get("NOT_RECORDED", 0),
            "denominator": total,
            "rate": _rate(parsed, total),
            "status_counts": dict(parse_counts),
        },
        "action": {
            "recognized": recognized_actions,
            "recognized_parsed": parsed_recognized_actions,
            "invalid_or_missing": total - recognized_actions,
            "denominator": total,
            "rate": _rate(recognized_actions, total),
            "among_parsed_rate": _rate(parsed_recognized_actions, parsed),
            "positive": sum(action_counts[action] for action in POSITIVE_ACTIONS),
            "abstentions": abstain_count,
            "counts": dict(action_counts),
        },
        "evidence": {
            "nonempty": evidence_present,
            "nonempty_parsed": parsed_evidence_present,
            "missing": total - evidence_present,
            "denominator": total,
            "rate": _rate(evidence_present, total),
            "among_parsed_rate": _rate(parsed_evidence_present, parsed),
        },
        "confidence": confidence_summary,
        "native_completeness": {
            "complete": native_complete,
            "denominator": total,
            "rate": _rate(native_complete, total),
            "complete_flag": native_flag,
            "native_input_mode": native_mode,
            "visual_metadata": native_visual,
            "complete_flag_rate": _rate(native_flag, total),
            "native_input_mode_rate": _rate(native_mode, total),
            "visual_metadata_rate": _rate(native_visual, total),
        },
    }
    result: dict[str, Any] = {
        "format": PRODUCTION_IDENTITY_PROBE_VERSION,
        "authority": AUTHORITY,
        "status": DIAGNOSTIC_STATUS,
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "production_eligible": False,
        "source": {
            "sidecar_format": document.get("format"),
            "sidecar_path": (input_paths or {}).get("sidecar"),
            "owner_confirmation_path": (input_paths or {}).get("owner_confirmation"),
            "window_count": total,
            "unique_window_count": len({row["window_id"] for row in rows}),
            "camera_count": len({row["camera_id"] for row in rows if row["camera_id"]}),
        },
        "model": {
            "identifier": model_map.get("identifier"),
            "label_profile": declared_profile or IDENTITY_PROFILE,
            "prompt_version": model_map.get("prompt_version"),
            "native_route": model_map.get("native_route"),
        },
        "metrics": metrics,
        # Keep direct metric aliases convenient for shell/post-hoc callers.
        "parse": metrics["parse"],
        "action": metrics["action"],
        "evidence": metrics["evidence"],
        "confidence": metrics["confidence"],
        "native_completeness": metrics["native_completeness"],
        "reference": reference_projection,
        "reference_overlap": reference_overlap,
        "windows": rows,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "raw_observation_preserved": True,
        },
        "limitations": [
            "Identity-only metrics describe recorded sidecar behavior, not production accuracy.",
            (
                "The optional owner-confirmation overlap is SURROGATE_NON_GOLD and "
                "official quality remains NOT_MEASURED."
            ),
            (
                "No temporal boundaries are evaluated because this probe intentionally "
                "omits timestamps and segments."
            ),
            (
                "Native completeness is a runtime/input contract check; it does not "
                "prove visual semantic correctness."
            ),
        ],
    }
    return cast(dict[str, Any], _copy_json(result, field="identity_probe"))


def project_identity_pending_candidates(
    sidecar: Mapping[str, Any] | str | Path,
    *,
    input_path: str | None = None,
) -> dict[str, Any]:
    """Project only clean identity rows into a pending review queue.

    The identity arm intentionally has no action boundaries.  Consequently a
    clean row becomes a *pending* candidate with null boundaries and an
    explicit ``IDENTITY_ONLY_NO_BOUNDARY`` reason.  Rows with invalid JSON,
    invalid fields, abstention identities, or incomplete native input remain
    visible as rejected diagnostics and never become candidates.  The complete
    input sidecar is copied under ``raw_sidecar`` for review traceability.
    """

    document = _load_payload(sidecar, field="sidecar")
    evaluated = evaluate_production_identity_probe(document)
    model = evaluated.get("model")
    model_map = model if isinstance(model, Mapping) else {}
    source_rows = _sequence(document.get("windows"), field="sidecar.windows")
    evaluated_rows = _sequence(evaluated.get("windows"), field="identity_probe.windows")
    if len(source_rows) != len(evaluated_rows):
        raise ProductionIdentityProbeError("sidecar/evaluation window counts differ")

    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    candidates_total = 0
    rejected_total = 0
    clean_rows = 0
    for index, (raw_source, raw_evaluated) in enumerate(
        zip(source_rows, evaluated_rows, strict=True)
    ):
        source_row = _mapping(raw_source, field=f"sidecar.windows[{index}]")
        observed = _mapping(raw_evaluated, field=f"identity_probe.windows[{index}]")
        window_id = _text(
            observed.get("window_id"), field=f"identity_probe.windows[{index}].window_id"
        )
        if window_id not in grouped:
            interval = source_row.get("interval", source_row.get("source_interval"))
            if interval is None and "start_seconds" in source_row and "end_seconds" in source_row:
                interval = [source_row.get("start_seconds"), source_row.get("end_seconds")]
            interval_copy = (
                _copy_json(interval, field=f"{window_id}.source_interval")
                if isinstance(interval, Sequence)
                and not isinstance(interval, (str, bytes, bytearray))
                else None
            )
            grouped[window_id] = {
                "window_id": window_id,
                "ordinal": source_row.get("ordinal", len(order)),
                "source_interval": interval_copy,
                "status": "ABSTAIN",
                "official_quality_status": OFFICIAL_QUALITY_STATUS,
                "official_gold_status": "NOT_ESTABLISHED",
                "quality_claim": False,
                "production_eligible": False,
                "automatic_eligible": False,
                "human_adjudication": "NOT_PERFORMED",
                "decision": "pending",
                "decision_options": ["accept", "edit", "split", "reject", "abstain"],
                "annotation_candidates": [],
                "rejected_identity_rows": [],
            }
            order.append(window_id)

        parse_status = str(observed.get("parse_status") or "NOT_RECORDED").upper()
        action = observed.get("action")
        action_text = str(action) if isinstance(action, str) else ""
        action_valid = observed.get("action_valid") is True
        evidence_present = observed.get("evidence_present") is True
        confidence_valid = observed.get("confidence_valid") is True
        native = observed.get("native_completeness")
        native_complete = isinstance(native, Mapping) and native.get("complete") is True
        runtime_status = str(observed.get("status") or "").upper()
        reasons: list[str] = []
        if parse_status != "PARSED":
            reasons.append("IDENTITY_PARSE_NOT_CLEAN")
        if not action_valid:
            reasons.append("ACTION_NOT_IN_VOCABULARY")
        elif action_text in ABSTAIN_ACTIONS:
            reasons.append(
                "NO_VISIBLE_ACTION" if action_text == "none visible" else "IDENTITY_UNCERTAIN"
            )
        if not evidence_present:
            reasons.append("EVIDENCE_NOT_MEASURED")
        if not confidence_valid:
            reasons.append("CONFIDENCE_NOT_MEASURED")
        if not native_complete:
            reasons.append("NATIVE_VIDEO_INCOMPLETE")
        if runtime_status not in {"SUCCEEDED", "SUCCESS"}:
            reasons.append("RUNTIME_NOT_SUCCEEDED")

        clean = (
            parse_status == "PARSED"
            and action_valid
            and action_text in POSITIVE_ACTIONS
            and evidence_present
            and confidence_valid
            and native_complete
            and runtime_status in {"SUCCEEDED", "SUCCESS"}
        )
        camera_id = observed.get("camera_id")
        camera_value = camera_id if isinstance(camera_id, str) and camera_id.strip() else None
        if clean:
            clean_rows += 1
            verb, noun = action_text.rsplit(" ", 1)
            candidate: dict[str, Any] = {
                "claim_id": (f"{window_id}:qwen-identity:{camera_value or 'single'}:{index}"),
                "source_claim_id": (
                    f"{window_id}:qwen-identity:{camera_value or 'single'}:{index}"
                ),
                "source_model": "qwen",
                "source_profile": model_map.get("label_profile"),
                "source_row_index": index,
                "camera_id": camera_value,
                "status": "PENDING_HUMAN_REVIEW",
                "automatic_eligible": False,
                "semantic_status": "NOT_CHECKED",
                "start_seconds": None,
                "end_seconds": None,
                "start_time_sec": None,
                "end_time_sec": None,
                "boundary_status": "NOT_MEASURED",
                "timestamp_basis": None,
                "verb": verb,
                "noun": noun,
                "attributes": None,
                "location": None,
                "hand": None,
                "field_status": {
                    "verb": "MEASURED",
                    "noun": "MEASURED",
                    "attributes": "NOT_MEASURED",
                    "location": "NOT_MEASURED",
                    "hand": "NOT_MEASURED",
                },
                "structured_labels": {
                    "verb": {"value": verb, "status": "MEASURED"},
                    "noun": {"value": noun, "status": "MEASURED"},
                    "attributes": {"value": None, "status": "NOT_MEASURED"},
                    "location": {"value": None, "status": "NOT_MEASURED"},
                    "hand": {"value": None, "status": "NOT_MEASURED"},
                },
                "label_text": action_text,
                "confidence": observed.get("confidence"),
                "evidence": _copy_json(observed.get("evidence", []), field=f"{window_id}.evidence"),
                "evidence_status": "MEASURED",
                "review_required": True,
                "accepted": False,
                "reason_codes": [
                    "IDENTITY_ONLY_NO_BOUNDARY",
                    "INDEPENDENT_REVIEW_REQUIRED",
                ],
                "raw_identity": _copy_json(
                    observed.get("raw_identity", {}), field=f"{window_id}.raw_identity"
                ),
            }
            candidate["raw_claim"] = candidate["raw_identity"]
            grouped[window_id]["annotation_candidates"].append(candidate)
            grouped[window_id]["status"] = "REVIEW_REQUIRED"
            candidates_total += 1
        else:
            rejected_total += 1
            grouped[window_id]["rejected_identity_rows"].append(
                {
                    "source_row_index": index,
                    "camera_id": camera_value,
                    "status": "ABSTAIN",
                    "action": action_text or None,
                    "reason_codes": list(dict.fromkeys(reasons or ["IDENTITY_NOT_CLEAN"])),
                    "raw_identity": _copy_json(
                        observed.get("raw_identity", {}), field=f"{window_id}.raw_identity"
                    ),
                }
            )

    windows = [grouped[window_id] for window_id in order]
    output = {
        "format": PRODUCTION_IDENTITY_CANDIDATE_PROJECTION_VERSION,
        "authority": AUTHORITY,
        "status": "REVIEW_REQUIRED" if candidates_total else "ABSTAIN",
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "production_eligible": False,
        "automatic_eligible": False,
        "model": _copy_json(model_map, field="sidecar.model"),
        "source": {
            "sidecar_format": document.get("format"),
            "sidecar_path": input_path,
            "window_count": len(windows),
            "source_row_count": len(source_rows),
            "unique_window_count": len(windows),
        },
        "windows": windows,
        # Keep the complete raw observation available to a reviewer.  This is
        # a plain JSON copy; no content identity is calculated.
        "raw_sidecar": _copy_json(document, field="raw_sidecar"),
        "metrics": {
            "window_count": len(windows),
            "source_row_count": len(source_rows),
            "clean_identity_row_count": clean_rows,
            "annotation_candidate_count": candidates_total,
            "rejected_identity_row_count": rejected_total,
            "review_required_window_count": sum(
                window["status"] == "REVIEW_REQUIRED" for window in windows
            ),
            "abstained_window_count": sum(window["status"] == "ABSTAIN" for window in windows),
        },
        "contract": {
            "identity_only": True,
            "boundaries_measured": False,
            "fixed_window_is_not_action_boundary": True,
            "clean_rows_only": True,
            "raw_sidecar_preserved": True,
            "automatic_eligible_always_false": True,
            "explicit_reviewer_decision_required": True,
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
            "raw_sidecar_preserved": True,
        },
        "limitations": [
            (
                "Identity-only rows do not provide action boundaries; candidates remain "
                "pending review with null boundaries."
            ),
            (
                "Rows are admitted structurally only; semantic identity still requires "
                "reviewer confirmation."
            ),
            "This benchmark-local projection is not a published production annotation schema.",
        ],
    }
    return cast(dict[str, Any], _copy_json(output, field="identity_candidate_projection"))


def render_identity_candidate_markdown(report: Mapping[str, Any]) -> str:
    """Render the pending identity-candidate projection."""

    metrics = report.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    lines = [
        "# Qwen identity pending-candidate projection",
        "",
        "> **REVIEW_REQUIRED/ABSTAIN.** Candidates are not gold and remain pending human review.",
        "",
        f"- Status: `{report.get('status', 'ABSTAIN')}`",
        f"- Windows: `{metrics.get('window_count', 0)}`",
        f"- Clean identity rows: `{metrics.get('clean_identity_row_count', 0)}`",
        f"- Pending candidates: `{metrics.get('annotation_candidate_count', 0)}`",
        f"- Rejected identity rows: `{metrics.get('rejected_identity_row_count', 0)}`",
        "",
        "| Window | Status | Candidates | Rejected |",
        "|---|---|---:|---:|",
    ]
    windows = report.get("windows", [])
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes, bytearray)):
        for window in windows:
            if not isinstance(window, Mapping):
                continue
            candidates = window.get("annotation_candidates", [])
            rejected = window.get("rejected_identity_rows", [])
            lines.append(
                f"| {window.get('window_id', '')} | {window.get('status', '')} | "
                f"{len(candidates) if isinstance(candidates, Sequence) else 0} | "
                f"{len(rejected) if isinstance(rejected, Sequence) else 0} |"
            )
    lines.extend(
        [
            "",
            (
                "Identity-only candidates have null action boundaries and require an "
                "explicit reviewer decision."
            ),
            "The complete raw sidecar is retained in JSON under `raw_sidecar`.",
            "",
        ]
    )
    return "\n".join(lines)


render_pending_markdown = render_identity_candidate_markdown


# Compatibility aliases for callers using the command name as the function.
evaluate_production_qwen_identity = evaluate_production_identity_probe
build_production_identity_probe = evaluate_production_identity_probe
build_identity_pending_candidates = project_identity_pending_candidates
project_production_identity_candidates = project_identity_pending_candidates


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact diagnostic summary."""

    metrics = report.get("metrics", {})
    parse = metrics.get("parse", {}) if isinstance(metrics, Mapping) else {}
    action = metrics.get("action", {}) if isinstance(metrics, Mapping) else {}
    evidence = metrics.get("evidence", {}) if isinstance(metrics, Mapping) else {}
    confidence = metrics.get("confidence", {}) if isinstance(metrics, Mapping) else {}
    native = metrics.get("native_completeness", {}) if isinstance(metrics, Mapping) else {}
    overlap = report.get("reference_overlap", {})

    def pct(value: object) -> str:
        return f"{float(value):.1%}" if isinstance(value, (int, float)) else "N/A"

    lines = [
        "# Qwen production identity-only probe",
        "",
        (
            "> **DIAGNOSTIC_ONLY.** Official quality is `NOT_MEASURED`; no model or "
            "media was run by the evaluator."
        ),
        "",
        f"- Windows: `{metrics.get('window_count', 0)}`",
        (
            f"- Parse: `{parse.get('parsed', 0)}/{parse.get('denominator', 0)}` "
            f"({pct(parse.get('rate'))})"
        ),
        (
            f"- Recognized action: `{action.get('recognized', 0)}/"
            f"{action.get('denominator', 0)}` ({pct(action.get('rate'))})"
        ),
        (
            f"- Non-empty evidence: `{evidence.get('nonempty', 0)}/"
            f"{evidence.get('denominator', 0)}` ({pct(evidence.get('rate'))})"
        ),
        (
            f"- Valid confidence: `{confidence.get('valid_count', 0)}/"
            f"{confidence.get('denominator', 0)}` ({pct(confidence.get('rate'))})"
        ),
        (
            f"- Native complete: `{native.get('complete', 0)}/"
            f"{native.get('denominator', 0)}` ({pct(native.get('rate'))})"
        ),
        "",
        "## Action counts",
        "",
        "| Action | Count |",
        "|---|---:|",
    ]
    counts = action.get("counts", {}) if isinstance(action, Mapping) else {}
    for label in IDENTITY_ACTIONS:
        lines.append(f"| {label} | {counts.get(label, 0)} |")
    lines.extend(["", "## Optional reference overlap", ""])
    if overlap.get("status") == SURROGATE_NON_GOLD:
        lines.extend(
            [
                "> **SURROGATE_NON_GOLD.** Owner-confirmation overlap is exploratory only.",
                "",
                f"- Eligible reference windows: `{overlap.get('eligible_window_count', 0)}`",
                (
                    f"- Exact action hits: `{overlap.get('exact_action_hits', 0)}` "
                    f"({pct(overlap.get('exact_action_rate'))}; any-camera availability)"
                ),
                (
                    f"- Primary aggregated action hits: `{overlap.get('primary_action_hits', 0)}` "
                    f"({pct(overlap.get('primary_action_rate'))})"
                ),
                (
                    f"- Prediction coverage: `{overlap.get('prediction_available_count', 0)}` "
                    f"({pct(overlap.get('prediction_coverage'))})"
                ),
            ]
        )
    else:
        lines.append("No owner-confirmation reference supplied; overlap is `NOT_MEASURED`.")
    lines.extend(
        [
            "",
            (
                "Per-window raw identity observations remain in the JSON report. "
                "This artifact is not a gold label file."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ABSTAIN_ACTIONS",
    "AUTHORITY",
    "DIAGNOSTIC_STATUS",
    "EXPECTED_SIDECAR_FORMAT",
    "IDENTITY_ACTIONS",
    "IDENTITY_PROFILE",
    "IDENTITY_PROFILES",
    "OFFICIAL_QUALITY_STATUS",
    "POSITIVE_ACTIONS",
    "PRODUCTION_IDENTITY_CANDIDATE_PROJECTION_VERSION",
    "PRODUCTION_IDENTITY_PROBE_VERSION",
    "SURROGATE_NON_GOLD",
    "ProductionIdentityProbeError",
    "build_identity_pending_candidates",
    "build_production_identity_probe",
    "evaluate_production_identity_probe",
    "evaluate_production_qwen_identity",
    "load_json",
    "project_identity_pending_candidates",
    "project_production_identity_candidates",
    "render_identity_candidate_markdown",
    "render_markdown",
    "render_pending_markdown",
]
