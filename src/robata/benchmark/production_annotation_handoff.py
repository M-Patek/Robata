"""Build a review-only production annotation handoff from model evidence.

This module joins the canonical label-blind structured envelope (or a direct
recorded model sidecar) with the
benchmark-local non-generative evidence verifier.  It is the last sidecar
boundary before an explicit reviewer decision: structurally reviewable Qwen
claims are rendered as annotation *candidates*, invalid claims remain visible
with reason codes, and every model's Top-K context is retained verbatim.

The output is not the published review-annotation wire contract and is never a
gold label.  It deliberately does not call a model, decode media, read gold,
modify the ontology/Mapper, train, or compute a hash/digest.  A reviewer must
still choose accept/edit/split/reject/abstain before any ordinary review-pack
bridge can create a human-authored result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from robata.benchmark.production_structured_annotation import (
    STRUCTURED_ANNOTATION_ENVELOPE_VERSION,
    ProductionStructuredAnnotationError,
    build_structured_annotation_envelope,
)
from robata.benchmark.production_structured_evidence_verifier import (
    StructuredEvidenceVerifierError,
    verify_production_structured_evidence,
)

PRODUCTION_ANNOTATION_HANDOFF_VERSION: Final = "robata-production-annotation-handoff-v1"
ANNOTATION_HANDOFF_VERSION: Final = PRODUCTION_ANNOTATION_HANDOFF_VERSION
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
OFFICIAL_GOLD_STATUS: Final = "NOT_ESTABLISHED"
HUMAN_ADJUDICATION_STATUS: Final = "NOT_PERFORMED"
DECISION_OPTIONS: Final = ("accept", "edit", "split", "reject", "abstain")
_OPTIONAL_FIELDS: Final = ("attributes", "location", "hand")


class ProductionAnnotationHandoffError(ValueError):
    """Raised when a structured envelope cannot form a review handoff."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionAnnotationHandoffError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionAnnotationHandoffError(f"{field} must be an array")
    return value


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _copy_json(value: object, *, field: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionAnnotationHandoffError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionAnnotationHandoffError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionAnnotationHandoffError(f"{field} must be JSON-compatible")


def _field_from_raw(raw_claim: Mapping[str, Any], field: str) -> tuple[Any, str]:
    labels = raw_claim.get("structured_labels", raw_claim.get("labels"))
    source: Mapping[str, Any] = labels if isinstance(labels, Mapping) else raw_claim
    raw = source.get(field)
    if isinstance(raw, Mapping):
        status = _text(raw.get("status")).upper()
        if not status:
            status = "NOT_OBSERVABLE" if raw.get("value") is None else "MEASURED"
        return raw.get("value"), status
    if raw is None:
        return None, "NOT_OBSERVABLE" if field in source else "NOT_MEASURED"
    return raw, "MEASURED"


def _label_text(fields: Mapping[str, Any]) -> str | None:
    verb = _text(fields.get("verb"))
    noun = _text(fields.get("noun"))
    if not verb or not noun:
        return None
    parts = [verb]
    attributes = _text(fields.get("attributes"))
    location = _text(fields.get("location"))
    hand = _text(fields.get("hand"))
    if attributes:
        parts.append(attributes)
    parts.append(noun)
    if location:
        parts.append(location)
    if hand:
        parts.extend(("with", hand))
    return " ".join(parts)


def _candidate_from_claim(claim: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(claim.get("raw_claim"), field="claim.raw_claim")
    interval = claim.get("interval")
    if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes, bytearray)):
        raise ProductionAnnotationHandoffError("reviewable claim interval is missing")
    if len(interval) != 2:
        raise ProductionAnnotationHandoffError("reviewable claim interval must have two values")
    start, end = _finite(interval[0]), _finite(interval[1])
    if start is None or end is None or end <= start:
        raise ProductionAnnotationHandoffError("reviewable claim interval is invalid")

    values: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for field in ("verb", "noun", *_OPTIONAL_FIELDS):
        value, status = _field_from_raw(raw, field)
        values[field] = value if status == "MEASURED" else None
        statuses[field] = status

    evidence = claim.get("evidence", [])
    evidence_values = (
        _copy_json(evidence, field=f"{claim.get('claim_id', 'claim')}.evidence")
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray))
        else []
    )
    candidate: dict[str, Any] = {
        "claim_id": claim.get("claim_id"),
        "source_model": claim.get("model", "qwen"),
        "source_claim_id": claim.get("claim_id"),
        "status": "PENDING_HUMAN_REVIEW",
        "automatic_eligible": False,
        "semantic_status": "NOT_CHECKED",
        "start_seconds": start,
        "end_seconds": end,
        "start_time_sec": start,
        "end_time_sec": end,
        "boundary_status": claim.get("boundary_status"),
        "timestamp_basis": claim.get("timestamp_basis"),
        "verb": values["verb"],
        "noun": values["noun"],
        "attributes": values["attributes"],
        "location": values["location"],
        "hand": values["hand"],
        "field_status": statuses,
        "structured_labels": {
            field: {"value": values[field], "status": statuses[field]}
            for field in ("verb", "noun", *_OPTIONAL_FIELDS)
        },
        "label_text": _label_text(values),
        "confidence": claim.get("raw_claim", {}).get("confidence"),
        "evidence": evidence_values,
        "evidence_status": claim.get("evidence_status"),
        "review_required": True,
        "accepted": False,
        "reason_codes": ["INDEPENDENT_REVIEW_REQUIRED", *claim.get("review_reason_codes", [])],
        "raw_claim": _copy_json(raw, field=f"{claim.get('claim_id', 'claim')}.raw_claim"),
    }
    candidate["reason_codes"] = list(dict.fromkeys(str(code) for code in candidate["reason_codes"]))
    return candidate


def _model_context(window: Mapping[str, Any]) -> dict[str, Any]:
    models = _mapping(window.get("models"), field="window.models")
    result: dict[str, Any] = {}
    for model in ("wemm", "qwen", "mage"):
        section = _mapping(models.get(model), field=f"window.models.{model}")
        result[model] = {
            "status": _text(section.get("status")).upper() or "NOT_RUN",
            "top_k": _copy_json(section.get("candidates", []), field=f"{model}.candidates"),
            "measurement_status": section.get("measurement_status", "NOT_MEASURED"),
            "candidate_sources": _copy_json(
                section.get("candidate_sources", []), field=f"{model}.candidate_sources"
            ),
            "candidate_groups": _copy_json(
                section.get("candidate_groups", []), field=f"{model}.candidate_groups"
            ),
            "structured_claim_count": len(
                section.get("segments", [])
                if isinstance(section.get("segments", []), Sequence)
                and not isinstance(section.get("segments", []), (str, bytes, bytearray))
                else []
            ),
            "retrieval_only": model == "wemm",
        }
    return result


def _canonical_context_envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return canonical window/model slots for an envelope or direct sidecar.

    The verifier accepts a recorded single-model sidecar as a convenience,
    while the handoff renderer needs the normalized ``models`` shape to retain
    all three model slots and WeMM's retrieval context.  Wrapping a direct
    sidecar here is structural only; it does not invoke a model or read media.
    """

    if payload.get("format") == STRUCTURED_ANNOTATION_ENVELOPE_VERSION:
        return payload
    model_key_by_format = {
        "robata-production-qwen-structured-native-shadow-v1": "qwen",
        "robata-production-qwen-shadow-v1": "qwen",
        "robata-production-wemm-shadow-v1": "wemm",
        "robata-production-wemm-vocabulary-shadow-v1": "wemm",
        "robata-production-mage-shadow-v1": "mage",
        "robata-production-mage-structured-native-shadow-v1": "mage",
    }
    model_key = model_key_by_format.get(str(payload.get("format")))
    if model_key is None:
        raise ProductionAnnotationHandoffError(
            "input must be a structured annotation envelope or supported model sidecar"
        )
    source = payload.get("source")
    source_path: str | None = None
    camera_count: int | None = None
    if isinstance(source, Mapping):
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
                source_path = candidate.strip()
                break
        raw_count = source.get("camera_count")
        if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count >= 0:
            camera_count = raw_count
    if source_path is None:
        for key in ("source_path", "media_path", "mcap_path", "video_path"):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                source_path = candidate.strip()
                break
    if source_path is None:
        raise ProductionAnnotationHandoffError("model sidecar source path is required")
    try:
        return cast(
            Mapping[str, Any],
            build_structured_annotation_envelope(
                {model_key: payload},
                source_path=source_path,
                camera_count=camera_count,
            ),
        )
    except ProductionStructuredAnnotationError as exc:
        raise ProductionAnnotationHandoffError(str(exc)) from exc


def build_production_annotation_handoff(
    envelope: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a review queue from a recorded structured envelope."""

    try:
        verification = verify_production_structured_evidence(envelope)
    except StructuredEvidenceVerifierError as exc:
        raise ProductionAnnotationHandoffError(str(exc)) from exc
    payload: Mapping[str, Any]
    if isinstance(envelope, Mapping):
        payload = envelope
    else:
        try:
            payload = _mapping(
                json.loads(Path(envelope).read_text(encoding="utf-8")), field="envelope"
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            raise ProductionAnnotationHandoffError(f"could not load envelope: {exc}") from exc
    # Use normalized windows for both canonical envelopes and direct
    # single-model sidecars.  Direct sidecars can have one row per camera,
    # whereas the handoff is intentionally one item per source window.
    context_envelope = _canonical_context_envelope(payload)
    raw_windows = _sequence(context_envelope.get("windows"), field="envelope.windows")
    verified_windows = _sequence(verification.get("windows"), field="verification.windows")
    if len(raw_windows) != len(verified_windows):
        raise ProductionAnnotationHandoffError("envelope/verifier window counts differ")

    windows: list[dict[str, Any]] = []
    for index, (raw_window, verified) in enumerate(zip(raw_windows, verified_windows, strict=True)):
        source_window = _mapping(raw_window, field=f"envelope.windows[{index}]")
        verified_window = _mapping(verified, field=f"verification.windows[{index}]")
        # New verifier versions expose all structured routes in
        # ``structured_claims_all`` (Qwen + optional Mage).  Fall back to the
        # historical Qwen-only field for old reports/replays.
        claims = _sequence(
            verified_window.get(
                "structured_claims_all", verified_window.get("structured_claims", [])
            ),
            field="verified.structured_claims_all",
        )
        reviewable = [
            _candidate_from_claim(_mapping(claim, field="verified.claim"))
            for claim in claims
            if _mapping(claim, field="verified.claim").get("eligible_for_review") is True
        ]
        rejected = [
            {
                "claim_id": claim.get("claim_id"),
                "source_model": claim.get("model", "qwen"),
                "status": "ABSTAIN",
                "reason_codes": _copy_json(
                    claim.get("reason_codes", []), field="claim.reason_codes"
                ),
                "raw_claim": _copy_json(claim.get("raw_claim"), field="claim.raw_claim"),
            }
            for claim in claims
            if _mapping(claim, field="verified.claim").get("eligible_for_review") is not True
        ]
        status = "REVIEW_REQUIRED" if reviewable else "ABSTAIN"
        windows.append(
            {
                "ordinal": source_window.get("ordinal"),
                "window_id": source_window.get("window_id"),
                "source_interval": [
                    source_window.get("start_time_sec"),
                    source_window.get("end_time_sec"),
                ],
                "status": status,
                "official_quality_status": OFFICIAL_QUALITY_STATUS,
                "official_gold_status": OFFICIAL_GOLD_STATUS,
                "quality_claim": False,
                "production_eligible": False,
                "automatic_eligible": False,
                "human_adjudication": HUMAN_ADJUDICATION_STATUS,
                "decision": "pending",
                "decision_options": list(DECISION_OPTIONS),
                "annotation_candidates": reviewable,
                "rejected_claims": rejected,
                "model_context": _model_context(source_window),
                "verifier": {
                    "structured_status": verified_window.get("structured_status"),
                    "structured_claim_count": verified_window.get(
                        "structured_claim_count_all",
                        verified_window.get("structured_claim_count", 0),
                    ),
                    "reviewable_claim_count": verified_window.get(
                        "reviewable_claim_count_all",
                        verified_window.get("reviewable_claim_count", 0),
                    ),
                    "invalid_claim_count": verified_window.get(
                        "invalid_claim_count_all",
                        verified_window.get("invalid_claim_count", 0),
                    ),
                    "models": _copy_json(verified_window.get("models", {}), field="window.models"),
                    "reason_codes": _copy_json(
                        verified_window.get("reason_codes", []), field="window.reason_codes"
                    ),
                    "parse_diagnostics_by_model": _copy_json(
                        verified_window.get("parse_diagnostics_by_model", {}),
                        field="window.parse_diagnostics_by_model",
                    ),
                    "parse_diagnostics": _copy_json(
                        verified_window.get("parse_diagnostics", []),
                        field="window.parse_diagnostics",
                    ),
                },
                "abstention": {
                    "abstained": not reviewable,
                    "reason_codes": _copy_json(
                        verified_window.get("reason_codes", []), field="window.abstention"
                    ),
                },
            }
        )

    metrics = {
        "window_count": len(windows),
        "review_required_window_count": sum(
            item["status"] == "REVIEW_REQUIRED" for item in windows
        ),
        "abstained_window_count": sum(item["status"] == "ABSTAIN" for item in windows),
        "annotation_candidate_count": sum(len(item["annotation_candidates"]) for item in windows),
        "rejected_claim_count": sum(len(item["rejected_claims"]) for item in windows),
        "annotation_candidate_count_by_model": {
            model: sum(
                sum(
                    candidate.get("source_model") == model
                    for candidate in item["annotation_candidates"]
                )
                for item in windows
            )
            for model in ("qwen", "mage")
        },
        "rejected_claim_count_by_model": {
            model: sum(
                sum(claim.get("source_model") == model for claim in item["rejected_claims"])
                for item in windows
            )
            for model in ("qwen", "mage")
        },
        "top_k_window_count": sum(
            any(context.get("top_k") for context in item["model_context"].values())
            for item in windows
        ),
    }
    source = _copy_json(verification.get("source", {}), field="verification.source")
    return {
        "format": PRODUCTION_ANNOTATION_HANDOFF_VERSION,
        "authority": AUTHORITY,
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": OFFICIAL_GOLD_STATUS,
        "quality_claim": False,
        "human_adjudication": HUMAN_ADJUDICATION_STATUS,
        "production_eligible": False,
        "automatic_eligible": False,
        "automatic_qualification": False,
        "status": "REVIEW_REQUIRED" if metrics["review_required_window_count"] else "ABSTAIN",
        "quality": {
            "measurement_status": OFFICIAL_QUALITY_STATUS,
            "quality_claim": False,
            "official_gold_status": OFFICIAL_GOLD_STATUS,
            "human_adjudication": HUMAN_ADJUDICATION_STATUS,
        },
        "source": source,
        "windows": windows,
        "metrics": metrics,
        "contract": {
            "annotation_fields": [
                "start_seconds",
                "end_seconds",
                "verb",
                "noun",
                "attributes",
                "location",
                "hand",
                "confidence",
                "evidence",
            ],
            "model_claims_are_not_gold": True,
            "structured_models_checked": ["qwen", "mage"],
            "wemm_retrieval_context_only": True,
            "top_k_preserved_verbatim": True,
            "invalid_claims_retained": True,
            "explicit_reviewer_decision_required": True,
            "fixed_window_is_not_action_boundary": True,
            "automatic_eligible_always_false": True,
            "automatic_qualification_disabled": True,
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
            "automatic_qualification": False,
        },
        "limitations": [
            (
                "Candidate eligibility is structural/provenance-only; semantic action "
                "identity is not checked."
            ),
            (
                "Every reviewable candidate still requires an independent source-bound "
                "reviewer decision."
            ),
            (
                "The sidecar is not schemas/v1/review-annotation.schema.json and cannot "
                "be published as a durable review record."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics", {}), field="report.metrics")
    quality = _mapping(report.get("quality", {}), field="report.quality")
    status = report.get("status", "ABSTAIN")
    measurement_status = quality.get("measurement_status", "NOT_MEASURED")
    review_required = metrics.get("review_required_window_count", 0)
    abstained = metrics.get("abstained_window_count", 0)
    candidates = metrics.get("annotation_candidate_count", 0)
    rejected = metrics.get("rejected_claim_count", 0)
    lines = [
        "# Production annotation handoff (review-only)",
        "",
        f"- Status: `{status}`",
        f"- Official quality: `{measurement_status}`",
        (
            f"- Windows: `{metrics.get('window_count', 0)}`; "
            f"review-required: `{review_required}`; abstained: `{abstained}`"
        ),
        (f"- Annotation candidates: `{candidates}`; rejected claims retained: `{rejected}`"),
        "",
        "| Window | Status | Candidates | Rejected | Top-K context |",
        "|---|---|---:|---:|---:|",
    ]
    windows = report.get("windows", [])
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes, bytearray)):
        for window in windows:
            if not isinstance(window, Mapping):
                continue
            context = window.get("model_context", {})
            top_k = (
                sum(
                    bool(item.get("top_k"))
                    for item in context.values()
                    if isinstance(item, Mapping)
                )
                if isinstance(context, Mapping)
                else 0
            )
            window_id = window.get("window_id", "")
            window_status = window.get("status", "")
            candidate_count = len(window.get("annotation_candidates", []))
            rejected_count = len(window.get("rejected_claims", []))
            lines.append(
                f"| {window_id} | {window_status} | {candidate_count} | "
                f"{rejected_count} | {top_k} models |"
            )
    lines.extend(
        [
            "",
            (
                "No model, media, gold, ontology, Mapper, training, or hash operation "
                "was performed; reviewer decision is pending."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ANNOTATION_HANDOFF_VERSION",
    "AUTHORITY",
    "DECISION_OPTIONS",
    "HUMAN_ADJUDICATION_STATUS",
    "OFFICIAL_GOLD_STATUS",
    "OFFICIAL_QUALITY_STATUS",
    "PRODUCTION_ANNOTATION_HANDOFF_VERSION",
    "ProductionAnnotationHandoffError",
    "build_production_annotation_handoff",
    "render_markdown",
]
