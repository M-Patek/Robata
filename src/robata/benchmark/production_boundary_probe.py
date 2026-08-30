"""Boundary-only diagnostics for the Qwen production shadow.

This module is deliberately separate from ``production_structured_annotation``.
It asks only for temporal evidence (a window-relative start/end pair, a
confidence and one short evidence phrase); action identity is supplied, when
available, as an untrusted model-observation sidecar context.  The parser is
strict and post-hoc: it never repairs timestamps, maps them to source time,
reads owner/Terra references, or changes the canonical production contract.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

PRODUCTION_BOUNDARY_PROBE_VERSION: Final = "robata-production-qwen-boundary-only-shadow-v1"
PRODUCTION_BOUNDARY_PROMPT_VERSION: Final = "qwen-production-boundary-only-v1"
PRODUCTION_BOUNDARY_PROMPT_BLIND_VERSION: Final = "qwen-production-boundary-only-blind-v1"
PRODUCTION_BOUNDARY_PROMPT_BOUNDED_V2_VERSION: Final = (
    "qwen-production-boundary-only-bounded-range-v2"
)
PRODUCTION_BOUNDARY_FRAME_PROMPT_VERSION: Final = "qwen-production-boundary-only-frame-ordinal-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
TIMESTAMP_BASIS: Final = "window_relative_seconds"
EXPECTED_IDENTITY_SIDECAR_FORMAT: Final = "robata-production-qwen-structured-native-shadow-v1"
IDENTITY_PROFILES: Final = frozenset(
    {"production_identity_only", "production_identity_disambiguated"}
)
BOUNDARY_STATUSES: Final = frozenset({"MEASURED", "UNCERTAIN", "NONE_VISIBLE", "ABSTAIN"})
UNRESOLVED_STATUSES: Final = frozenset({"UNCERTAIN", "NONE_VISIBLE", "ABSTAIN"})

QWEN_PRODUCTION_BOUNDARY_ONLY_PROMPT: Final = (
    "Review the complete bounded native video once. Return exactly one compact JSON object "
    "with only these keys: timestamp_basis, start_time_sec, end_time_sec, confidence, "
    'evidence, status. Use timestamp_basis exactly "window_relative_seconds": offsets '
    "start at this bounded clip, not source-absolute time. Report the smallest visible "
    "interval in which the hypothesized action's state-changing interaction occurs. Use "
    "null boundaries when the action is not visibly localizable. Confidence is a number "
    "from 0 to 1 or null. Evidence is one short directly visible fact or an empty string. "
    "Status is MEASURED when both boundaries are visible, UNCERTAIN when an action is "
    "plausible but boundaries are unclear, or NONE_VISIBLE when no relevant interaction "
    "appears. Do not output an action label, taxonomy ID, segment array, source timestamps, "
    "or prose outside the JSON object."
)

# A single-variable prompt arm for the observed failure mode where Qwen emits
# ``0,0`` placeholders or source-clock values (for example ``7.7``) even
# though the request is a four-second window.  The parser remains unchanged:
# it still rejects invalid claims rather than repairing them.
QWEN_PRODUCTION_BOUNDARY_ONLY_BOUNDED_V2_PROMPT: Final = (
    QWEN_PRODUCTION_BOUNDARY_ONLY_PROMPT
    + " The bounded clip duration is the only valid clock. Valid relative values "
    + "are within 0.0 and the clip duration; never emit source-absolute values or "
    + "a number larger than the clip duration. Do not use 0.0/0.0 as a placeholder. "
    + "If both onset and completion are not visibly localizable, set both boundaries "
    + "to null and set status to UNCERTAIN. For MEASURED, use a strictly positive "
    + "interval inside the clip and keep the interval as short as the visible action."
)

# A diagnostic arm for the observed clock confusion: Qwen may emit source-clock
# values despite being shown a bounded clip.  Frame ordinals are unambiguous in
# the ordered native input; the runner maps them using the actual sampled-frame
# timestamp table after generation.  This is intentionally a separate contract
# and never reinterprets seconds from the existing arms.
_QWEN_PRODUCTION_BOUNDARY_ONLY_FRAME_ORDINAL_PROMPT_TEMPLATE: Final = (
    "Review the complete bounded native video once. The input is an ordered sequence "
    "of exactly {frame_count_text} sampled frames, numbered 0 through {last_ordinal} in "
    "presentation order. "
    "Return exactly one compact JSON object with only these keys: coordinate_mode, "
    "start_frame_ordinal, end_frame_ordinal, confidence, evidence, status. Set "
    'coordinate_mode exactly to "sampled_frame_ordinal". Report the smallest '
    "visible interval in which the supplied identity hypothesis' state-changing "
    "interaction occurs. Use integer ordinals from 0 through {last_ordinal}; start must be "
    "strictly "
    "less than end. If an endpoint is unclear, use null for both ordinals and status "
    "UNCERTAIN. Status is MEASURED only when both endpoints are directly visible; use "
    "NONE_VISIBLE when no relevant interaction appears. Confidence is a number from 0 "
    "to 1 or null. Evidence is one short directly visible fact or an empty string. Do "
    "not output seconds, source timestamps, an action label, taxonomy ID, segment array, "
    "or prose outside the JSON object. Never use equal ordinals or invent an endpoint."
)


class ProductionBoundaryProbeError(ValueError):
    """Raised when a boundary sidecar or identity context is malformed."""


def frame_ordinal_prompt(frame_count: int = 8) -> str:
    """Build the frame-ordinal prompt for the actual sampled frame count.

    The native sampler is configurable; spelling the count and highest ordinal in
    the prompt keeps the model's coordinate system aligned with the recorded frame
    table and the strict post-hoc parser.
    """

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 2:
        raise ProductionBoundaryProbeError("frame_count must be an integer >= 2")
    return _QWEN_PRODUCTION_BOUNDARY_ONLY_FRAME_ORDINAL_PROMPT_TEMPLATE.format(
        frame_count_text="eight" if frame_count == 8 else str(frame_count),
        last_ordinal=frame_count - 1,
    )


# Preserve the original public constant (including its text for the historical
# eight-frame arm) for callers and previously recorded diagnostic artifacts.
QWEN_PRODUCTION_BOUNDARY_ONLY_FRAME_ORDINAL_PROMPT: Final = frame_ordinal_prompt(8)

QWEN_PRODUCTION_BOUNDARY_ONLY_BLIND_PROMPT: Final = QWEN_PRODUCTION_BOUNDARY_ONLY_PROMPT


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionBoundaryProbeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionBoundaryProbeError(f"{field} must be an array")
    return value


def _finite_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionBoundaryProbeError(f"{field} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ProductionBoundaryProbeError(f"{field} must be a finite number or null")
    return result


def _confidence(value: object) -> float | None:
    result = _finite_number(value, field="confidence")
    if result is not None and not 0.0 <= result <= 1.0:
        raise ProductionBoundaryProbeError("confidence must be between 0 and 1")
    return result


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionBoundaryProbeError(f"{field} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ProductionBoundaryProbeError(f"{field} must be non-empty")
    return result


def _copy_json(value: object, *, field: str) -> Any:
    """Copy JSON-compatible provenance without deriving an identity/hash."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionBoundaryProbeError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionBoundaryProbeError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionBoundaryProbeError(f"{field} must be JSON-compatible")


def load_json(
    value: Mapping[str, Any] | str | Path, *, field: str = "JSON input"
) -> dict[str, Any]:
    """Load a JSON object from a mapping or path."""

    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser()
    try:
        decoded = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionBoundaryProbeError(f"could not load {field} {path}: {exc}") from exc
    return dict(_mapping(decoded, field=str(path)))


def _invalid(*errors: str, warnings: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "parse_status": "INVALID",
        "timestamp_basis": None,
        "boundary_status": "INVALID",
        "start_time_sec": None,
        "end_time_sec": None,
        "confidence": None,
        "evidence": "",
        "errors": list(errors),
        "warnings": list(warnings),
    }


def parse_qwen_boundary_only_output(
    raw_text: str,
    *,
    window_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Parse one strict boundary-only response without repairing claims.

    ``window_duration_seconds`` is optional for unit-level parsing.  When
    supplied, measured boundaries must lie within that bounded native clip.
    Relative offsets are intentionally retained; this function never maps them
    to source-absolute time.
    """

    if not isinstance(raw_text, str) or not raw_text.strip():
        return _invalid("RAW_TEXT_MISSING")
    try:
        decoded = json.loads(raw_text)
    except (TypeError, ValueError):
        return _invalid("INVALID_JSON")
    if not isinstance(decoded, Mapping):
        return _invalid("JSON_OBJECT_REQUIRED")

    declared_basis = decoded.get("timestamp_basis")
    if not isinstance(declared_basis, str) or not declared_basis.strip():
        return _invalid("TIMESTAMP_BASIS_MISSING")
    if declared_basis.strip() != TIMESTAMP_BASIS:
        return _invalid("TIMESTAMP_BASIS_UNSUPPORTED")

    status_raw = decoded.get("status", "MEASURED")
    if not isinstance(status_raw, str) or not status_raw.strip():
        return _invalid("STATUS_INVALID")
    status = status_raw.strip().upper()
    if status not in BOUNDARY_STATUSES:
        return _invalid("STATUS_UNSUPPORTED")

    try:
        start = _finite_number(decoded.get("start_time_sec"), field="start_time_sec")
        end = _finite_number(decoded.get("end_time_sec"), field="end_time_sec")
        confidence = _confidence(decoded.get("confidence"))
        evidence = _text(decoded.get("evidence", ""), field="evidence", allow_empty=True)
    except ProductionBoundaryProbeError as exc:
        return _invalid(str(exc))

    if status in UNRESOLVED_STATUSES:
        if start is not None or end is not None:
            return _invalid("UNRESOLVED_BOUNDARY_MUST_BE_NULL")
        return {
            "parse_status": "PARSED",
            "timestamp_basis": TIMESTAMP_BASIS,
            "boundary_status": status,
            "start_time_sec": None,
            "end_time_sec": None,
            "confidence": confidence,
            "evidence": evidence,
            "errors": [],
            "warnings": [],
        }

    if start is None or end is None:
        return _invalid("MEASURED_BOUNDARY_PAIR_REQUIRED")
    if start < 0.0 or end < 0.0:
        return _invalid("BOUNDARY_NEGATIVE")
    if end <= start:
        return _invalid("BOUNDARY_END_NOT_AFTER_START")
    if window_duration_seconds is not None:
        duration = float(window_duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ProductionBoundaryProbeError(
                "window_duration_seconds must be positive and finite"
            )
        if end > duration:
            return _invalid("BOUNDARY_OUT_OF_RANGE")

    return {
        "parse_status": "PARSED",
        "timestamp_basis": TIMESTAMP_BASIS,
        "boundary_status": "MEASURED",
        "start_time_sec": start,
        "end_time_sec": end,
        "confidence": confidence,
        "evidence": evidence,
        "errors": [],
        "warnings": [],
    }


def parse_qwen_boundary_frame_output(
    raw_text: str,
    *,
    frame_count: int = 8,
) -> dict[str, Any]:
    """Parse the sampled-frame ordinal boundary diagnostic without mapping it."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        return _invalid("RAW_TEXT_MISSING")
    try:
        decoded = json.loads(raw_text)
    except (TypeError, ValueError):
        return _invalid("INVALID_JSON")
    if not isinstance(decoded, Mapping):
        return _invalid("JSON_OBJECT_REQUIRED")
    if str(decoded.get("coordinate_mode") or "").strip() != "sampled_frame_ordinal":
        return _invalid("COORDINATE_MODE_MISSING_OR_UNSUPPORTED")
    status_raw = decoded.get("status", "MEASURED")
    if not isinstance(status_raw, str) or not status_raw.strip():
        return _invalid("STATUS_INVALID")
    status = status_raw.strip().upper()
    if status not in BOUNDARY_STATUSES:
        return _invalid("STATUS_UNSUPPORTED")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 2:
        raise ProductionBoundaryProbeError("frame_count must be an integer >= 2")
    try:
        confidence = _confidence(decoded.get("confidence"))
        evidence = _text(decoded.get("evidence", ""), field="evidence", allow_empty=True)
    except ProductionBoundaryProbeError as exc:
        return _invalid(str(exc))
    start_raw, end_raw = decoded.get("start_frame_ordinal"), decoded.get("end_frame_ordinal")
    if status in UNRESOLVED_STATUSES:
        if start_raw is not None or end_raw is not None:
            return _invalid("UNRESOLVED_BOUNDARY_MUST_BE_NULL")
        return {
            "parse_status": "PARSED",
            "coordinate_mode": "sampled_frame_ordinal",
            "boundary_status": status,
            "start_frame_ordinal": None,
            "end_frame_ordinal": None,
            "confidence": confidence,
            "evidence": evidence,
            "errors": [],
            "warnings": [],
        }
    if isinstance(start_raw, bool) or not isinstance(start_raw, int):
        return _invalid("START_FRAME_ORDINAL_INVALID")
    if isinstance(end_raw, bool) or not isinstance(end_raw, int):
        return _invalid("END_FRAME_ORDINAL_INVALID")
    if start_raw < 0 or end_raw < 0 or start_raw >= frame_count or end_raw >= frame_count:
        return _invalid("FRAME_ORDINAL_OUT_OF_RANGE")
    if end_raw <= start_raw:
        return _invalid("FRAME_ORDINAL_END_NOT_AFTER_START")
    return {
        "parse_status": "PARSED",
        "coordinate_mode": "sampled_frame_ordinal",
        "boundary_status": "MEASURED",
        "start_frame_ordinal": start_raw,
        "end_frame_ordinal": end_raw,
        "confidence": confidence,
        "evidence": evidence,
        "errors": [],
        "warnings": [],
    }


def identity_context_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only a model-observed identity hypothesis from one sidecar row."""

    identity_value = row.get("parsed_identity", row.get("raw_identity"))
    identity = identity_value if isinstance(identity_value, Mapping) else row
    status = identity.get("parse_status", row.get("identity_status"))
    action = identity.get("action")
    if (
        isinstance(status, str)
        and status.strip().upper() == "PARSED"
        and isinstance(action, str)
        and action.strip()
    ):
        return {
            "status": "AVAILABLE",
            "action": action.strip(),
            "confidence": identity.get("confidence"),
            "evidence": _copy_json(identity.get("evidence", []), field="identity.evidence"),
            "source": "qwen_model_observation",
        }
    return {
        "status": "UNAVAILABLE",
        "reason": "IDENTITY_NOT_PARSED",
        "source": "qwen_model_observation",
    }


def _assert_identity_sidecar_safe(document: Mapping[str, Any]) -> None:
    """Reject owner/gold material from an identity context input."""

    if document.get("format") != EXPECTED_IDENTITY_SIDECAR_FORMAT:
        raise ProductionBoundaryProbeError(
            "identity sidecar format must be the Qwen structured-native shadow"
        )
    if document.get("production_eligible") is True:
        raise ProductionBoundaryProbeError("identity sidecar cannot be production eligible")
    controls = document.get("controls")
    if isinstance(controls, Mapping):
        for key in ("gold_included", "gold_read", "gold_written", "predictions_copied_to_gold"):
            if controls.get(key) is True:
                raise ProductionBoundaryProbeError("identity sidecar contains gold controls")
    model = document.get("model")
    if isinstance(model, Mapping):
        profile = model.get("label_profile")
        if profile is not None and profile not in IDENTITY_PROFILES:
            raise ProductionBoundaryProbeError(
                "identity sidecar label_profile is not an identity profile"
            )


def index_identity_sidecar(
    sidecar: Mapping[str, Any] | str | Path,
) -> dict[tuple[str, str | None], dict[str, Any]]:
    """Index Qwen identity observations by ``(window_id, camera_id)``."""

    document = load_json(sidecar, field="identity sidecar")
    _assert_identity_sidecar_safe(document)
    rows = _sequence(document.get("windows"), field="identity sidecar.windows")
    indexed: dict[tuple[str, str | None], dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"identity sidecar.windows[{index}]")
        window_id = row.get("window_id")
        if not isinstance(window_id, str) or not window_id.strip():
            raise ProductionBoundaryProbeError(
                f"identity sidecar.windows[{index}].window_id invalid"
            )
        camera = row.get("camera_id")
        camera_id = camera.strip() if isinstance(camera, str) and camera.strip() else None
        key = (window_id.strip(), camera_id)
        if key in indexed:
            raise ProductionBoundaryProbeError(
                f"duplicate identity row: {key[0]}:{key[1] or 'single'}"
            )
        context = identity_context_from_row(row)
        context["window_id"] = key[0]
        context["camera_id"] = camera_id
        context["source_row_index"] = index
        indexed[key] = context
    return indexed


def find_identity_context(
    indexed: Mapping[tuple[str, str | None], Mapping[str, Any]],
    *,
    window_id: str,
    camera_id: str,
) -> dict[str, Any]:
    """Return a copied context, preferring camera-specific then single rows."""

    value = indexed.get((window_id, camera_id)) or indexed.get((window_id, None))
    if value is None:
        return {"status": "NOT_SUPPLIED", "source": "qwen_model_observation"}
    return cast(dict[str, Any], _copy_json(value, field="identity_context"))


def evaluate_production_boundary_probe(
    sidecar: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compute non-gold boundary diagnostics from a recorded sidecar."""

    document = load_json(sidecar, field="boundary sidecar")
    rows = _sequence(document.get("windows"), field="boundary sidecar.windows")
    parsed = measured = unresolved = evidence = confidence = native_complete = 0
    row_reports: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"boundary sidecar.windows[{index}]")
        result = row.get("parsed_boundary")
        parsed_result = result if isinstance(result, Mapping) else {}
        status = str(parsed_result.get("parse_status", "INVALID")).upper()
        boundary_status = str(parsed_result.get("boundary_status", "INVALID")).upper()
        parsed += status == "PARSED"
        measured += status == "PARSED" and boundary_status == "MEASURED"
        unresolved += status == "PARSED" and boundary_status in UNRESOLVED_STATUSES
        evidence += bool(
            status == "PARSED"
            and isinstance(parsed_result.get("evidence"), str)
            and parsed_result.get("evidence")
        )
        confidence += bool(
            status == "PARSED" and isinstance(parsed_result.get("confidence"), (int, float))
        )
        native_complete += row.get("native_video_complete") is True
        row_reports.append(
            {
                "window_id": row.get("window_id"),
                "camera_id": row.get("camera_id"),
                "parse_status": status,
                "boundary_status": boundary_status,
                "identity_context_status": (
                    row.get("identity_context", {}).get("status")
                    if isinstance(row.get("identity_context"), Mapping)
                    else "NOT_SUPPLIED"
                ),
            }
        )
    total = len(rows)
    return {
        "format": PRODUCTION_BOUNDARY_PROBE_VERSION,
        "authority": AUTHORITY,
        "status": "DIAGNOSTIC_ONLY",
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "production_eligible": False,
        "source": {"sidecar_format": document.get("format"), "row_count": total},
        "metrics": {
            "row_count": total,
            "parse": {"parsed": parsed, "invalid": total - parsed},
            "boundary": {"measured": measured, "unresolved": unresolved},
            "evidence_present": evidence,
            "confidence_present": confidence,
            "native_video_complete": native_complete,
        },
        "windows": row_reports,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
        },
        "limitations": [
            "Boundary-only diagnostics are not official quality measurement.",
            (
                "Relative boundaries are retained and are never mapped to source time "
                "by this evaluator."
            ),
        ],
    }


__all__ = [
    "AUTHORITY",
    "BOUNDARY_STATUSES",
    "EXPECTED_IDENTITY_SIDECAR_FORMAT",
    "OFFICIAL_QUALITY_STATUS",
    "PRODUCTION_BOUNDARY_FRAME_PROMPT_VERSION",
    "PRODUCTION_BOUNDARY_PROBE_VERSION",
    "PRODUCTION_BOUNDARY_PROMPT_BLIND_VERSION",
    "PRODUCTION_BOUNDARY_PROMPT_BOUNDED_V2_VERSION",
    "PRODUCTION_BOUNDARY_PROMPT_VERSION",
    "QWEN_PRODUCTION_BOUNDARY_ONLY_BLIND_PROMPT",
    "QWEN_PRODUCTION_BOUNDARY_ONLY_BOUNDED_V2_PROMPT",
    "QWEN_PRODUCTION_BOUNDARY_ONLY_FRAME_ORDINAL_PROMPT",
    "QWEN_PRODUCTION_BOUNDARY_ONLY_PROMPT",
    "TIMESTAMP_BASIS",
    "ProductionBoundaryProbeError",
    "evaluate_production_boundary_probe",
    "find_identity_context",
    "frame_ordinal_prompt",
    "identity_context_from_row",
    "index_identity_sidecar",
    "load_json",
    "parse_qwen_boundary_frame_output",
    "parse_qwen_boundary_only_output",
]
