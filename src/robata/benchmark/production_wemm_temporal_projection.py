"""Project model-derived WeMM temporal intervals into a review draft.

The native WeMM runner keeps processing windows as context and emits an
additive ``temporal_resolution`` sidecar when its dense score track can
localise a candidate action.  Historically the annotation-draft bridge only
consumed the per-window proposals, which meant those model-derived boundaries
were visible in a sidecar but absent from the human review surface.

This module is a small, inference-free adapter for that seam.  It deliberately
keeps temporal proposals in a separate top-level collection instead of
attaching one interval to an arbitrary processing window.  Every projected
row remains editable, requires human review, and is explicitly ineligible for
automatic/gold publication.  The adapter accepts both a per-recording
``temporal_resolution`` object and the flattened aggregate
``temporal_segments`` alias.

No media/model/gold operation is performed and no hash or digest is calculated.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

FORMAT: Final = "robata-production-wemm-temporal-annotation-projection-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "PENDING_REVIEW"
BOUNDARY_STATUS: Final = "MODEL_PROBE_BOUND"
DECISION_OPTIONS: Final = ("accept", "edit", "split", "reject", "abstain")
TEMPORAL_INTERVAL_FIELDS: Final = (
    "start_seconds",
    "end_seconds",
    "verb",
    "noun",
    "attributes",
    "location",
    "hand",
    "confidence",
    "evidence",
    "camera_support",
    "top_k",
    "margin",
)
_FIELD_NAMES: Final = ("verb", "noun", "attributes", "location", "hand")
_SUPPORTED_SOURCE_FORMATS: Final = frozenset(
    {
        "robata-production-wemm-preannotation-v1",
        "robata-production-wemm-preannotation-review-pack-v1",
        "robata-production-wemm-review-pack-aggregate-v1",
    }
)
_GOLD_KEY_FRAGMENTS: Final = frozenset(
    {
        "gold",
        "groundtruth",
        "officialgold",
        "officialreference",
        "officiallabel",
        "humanannotation",
        "adjudicatedlabel",
    }
)


class ProductionWemmTemporalProjectionError(ValueError):
    """Raised when a temporal sidecar cannot be projected safely."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmTemporalProjectionError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmTemporalProjectionError(f"{field} must be an array")
    return value


def _text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _finite(value: object, *, field: str) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ProductionWemmTemporalProjectionError(f"{field} must be finite")
    return result


def _copy_json(value: object, *, field: str = "value") -> Any:
    """Detach JSON-compatible input and reject non-finite numbers."""

    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmTemporalProjectionError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmTemporalProjectionError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionWemmTemporalProjectionError(f"{field} must be JSON-compatible")


def _key_token(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _assert_no_gold(value: object, *, field: str = "input") -> None:
    """Reject an accidental official-label join at this model-only boundary."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProductionWemmTemporalProjectionError(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmTemporalProjectionError(f"{field} keys must be strings")
            token = _key_token(key)
            if token in _GOLD_KEY_FRAGMENTS:
                raise ProductionWemmTemporalProjectionError(
                    f"{field}.{key} contains gold/official annotation data"
                )
            _assert_no_gold(child, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_gold(child, field=f"{field}[{index}]")
        return
    raise ProductionWemmTemporalProjectionError(f"{field} must be JSON-compatible")


def _recording_id(source: Mapping[str, Any]) -> str | None:
    """Find a recording identifier without inventing one from a label."""

    candidates: list[Mapping[str, Any]] = [source]
    for key in ("source", "source_ref"):
        nested = source.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
            nested_source = nested.get("source")
            if isinstance(nested_source, Mapping):
                candidates.append(nested_source)
    for candidate in candidates:
        for key in ("recording_id", "source_id", "id"):
            value = _text(candidate.get(key))
            if value:
                return value
    return None


def _sidecar_segments(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    """Read and validate the optional temporal sidecar/alias.

    Per-recording sidecars use ``status=PROPOSALS_ONLY`` and
    ``production_eligible=false``.  The aggregate keeps the same segment rows
    under a compact wrapper with ``review_only=true``; both forms are accepted
    while each row still has to carry explicit review-only invariants.
    """

    raw_resolution = source.get("temporal_resolution")
    raw_alias = source.get("temporal_segments")
    resolution_segments: list[dict[str, Any]] | None = None
    resolution_format: str | None = None

    def validate_segment(raw: object, *, field: str) -> dict[str, Any]:
        segment = _mapping(raw, field=field)
        if segment.get("review_required") is not True:
            raise ProductionWemmTemporalProjectionError(f"{field}.review_required must be true")
        if segment.get("automatic_eligible") is not False:
            raise ProductionWemmTemporalProjectionError(f"{field}.automatic_eligible must be false")
        if segment.get("boundary_status") != BOUNDARY_STATUS:
            raise ProductionWemmTemporalProjectionError(
                f"{field}.boundary_status must be {BOUNDARY_STATUS}"
            )
        start = _finite(segment.get("start_seconds"), field=f"{field}.start_seconds")
        end = _finite(segment.get("end_seconds"), field=f"{field}.end_seconds")
        if start is None or end is None or start < 0.0 or end <= start:
            raise ProductionWemmTemporalProjectionError(
                f"{field} interval must satisfy 0 <= start < end"
            )
        # Preserve the row exactly; callers need all score/transition/camera
        # provenance for review and later error analysis.
        copied = _copy_json(segment, field=field)
        if not isinstance(copied, dict):  # pragma: no cover - helper invariant
            raise ProductionWemmTemporalProjectionError(f"{field} must be an object")
        return copied

    if raw_resolution is not None:
        resolution = _mapping(raw_resolution, field="temporal_resolution")
        resolution_format = _text(resolution.get("format"))
        status = _text(resolution.get("status"))
        review_only = resolution.get("review_only") is True
        if status is not None and status != "PROPOSALS_ONLY":
            raise ProductionWemmTemporalProjectionError(
                "temporal_resolution.status must be PROPOSALS_ONLY"
            )
        if status is None and not review_only:
            raise ProductionWemmTemporalProjectionError(
                "temporal_resolution must declare PROPOSALS_ONLY or review_only"
            )
        if (
            "production_eligible" in resolution
            and resolution.get("production_eligible") is not False
        ):
            raise ProductionWemmTemporalProjectionError(
                "temporal_resolution.production_eligible must be false"
            )
        raw_segments = _sequence(
            resolution.get("segments", []), field="temporal_resolution.segments"
        )
        resolution_segments = [
            validate_segment(raw, field=f"temporal_resolution.segments[{index}]")
            for index, raw in enumerate(raw_segments)
        ]

    alias_segments: list[dict[str, Any]] | None = None
    if raw_alias is not None:
        raw_values = _sequence(raw_alias, field="temporal_segments")
        alias_segments = [
            validate_segment(raw, field=f"temporal_segments[{index}]")
            for index, raw in enumerate(raw_values)
        ]

    if resolution_segments is None and alias_segments is None:
        return [], resolution_format
    if resolution_segments is not None and alias_segments is not None:
        if resolution_segments != alias_segments:
            raise ProductionWemmTemporalProjectionError(
                "temporal_segments does not match temporal_resolution.segments"
            )
        return resolution_segments, resolution_format
    return (
        resolution_segments if resolution_segments is not None else alias_segments or [],
        resolution_format,
    )


def _field_value(labels: Mapping[str, Any], name: str) -> tuple[Any, str]:
    raw = labels.get(name)
    if isinstance(raw, Mapping):
        value = raw.get("value")
        status = (
            _text(raw.get("status")) or ("MEASURED" if value is not None else "NOT_MEASURED")
        ).upper()
    else:
        value = raw
        status = "MEASURED" if raw is not None else "NOT_MEASURED"
    if status not in {"MEASURED", "NOT_MEASURED", "NOT_OBSERVABLE"}:
        status = "NOT_MEASURED"
    if status != "MEASURED":
        value = None
    return _copy_json(value, field=f"structured_labels.{name}"), status


def _camera_ids(value: object) -> list[str]:
    if isinstance(value, Mapping):
        value = list(value.keys())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return sorted({item.strip() for item in value if isinstance(item, str) and item.strip()})


def _evidence_cameras(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return sorted(
        {
            str(item.get("camera_id")).strip()
            for item in value
            if isinstance(item, Mapping)
            and isinstance(item.get("camera_id"), str)
            and item.get("camera_id", "").strip()
        }
    )


def _label_text(segment: Mapping[str, Any], values: Mapping[str, Any]) -> str | None:
    for key in ("label_text", "provisional_label", "text", "label"):
        value = _text(segment.get(key))
        if value:
            return value
    verb, noun = values.get("verb"), values.get("noun")
    if isinstance(verb, str) and isinstance(noun, str) and verb.strip() and noun.strip():
        return f"{verb.strip()} {noun.strip()}"
    return None


def _source_ref(
    segment: Mapping[str, Any], source: Mapping[str, Any], recording_id: str | None
) -> dict[str, Any]:
    raw = segment.get("source_ref")
    if not isinstance(raw, Mapping):
        raw = source.get("source_ref")
    if not isinstance(raw, Mapping):
        raw = source.get("source")
    copied = _copy_json(raw or {}, field="temporal.source_ref")
    if not isinstance(copied, dict):  # pragma: no cover - helper invariant
        copied = {}
    if recording_id is not None:
        copied.setdefault("recording_id", recording_id)
    return {key: value for key, value in copied.items()}


def _project_segment(
    segment: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    index: int,
    sidecar_format: str | None,
) -> dict[str, Any]:
    recording_id = _text(segment.get("recording_id")) or _recording_id(source)
    source_id = (
        _text(segment.get("temporal_segment_key"))
        or _text(segment.get("segment_id"))
        or _text(segment.get("proposal_id"))
        or f"temporal-{index:04d}"
    )
    segment_id = (
        source_id
        if "::temporal::" in source_id or recording_id is None
        else f"{recording_id}::temporal::{source_id}"
    )
    labels_raw = segment.get("structured_labels")
    labels = labels_raw if isinstance(labels_raw, Mapping) else segment
    values: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for name in _FIELD_NAMES:
        values[name], statuses[name] = _field_value(labels, name)
    evidence = _copy_json(segment.get("evidence", []), field=f"{segment_id}.evidence")
    if not isinstance(evidence, list):  # pragma: no cover - helper invariant
        evidence = []
    cameras = _camera_ids(segment.get("camera_support", segment.get("camera_ids", [])))
    if not cameras:
        cameras = _evidence_cameras(evidence)
    top_k_value = segment.get("top_k", segment.get("top_k_by_window", []))
    top_k = _copy_json(top_k_value, field=f"{segment_id}.top_k")
    if not isinstance(top_k, list):  # pragma: no cover - helper invariant
        top_k = []
    confidence = _finite(segment.get("confidence"), field=f"{segment_id}.confidence")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ProductionWemmTemporalProjectionError(
            f"{segment_id}.confidence must be between 0 and 1"
        )
    margin = _finite(segment.get("margin"), field=f"{segment_id}.margin")
    supporting_windows = _copy_json(
        segment.get("supporting_window_ids", []), field=f"{segment_id}.supporting_window_ids"
    )
    if not isinstance(supporting_windows, list):  # pragma: no cover - helper invariant
        supporting_windows = []
    boundary_method = _text(segment.get("boundary_method"))
    timestamp_basis = _text(segment.get("timestamp_basis")) or "MODEL_PROBE_BOUNDARY"
    return {
        "segment_id": segment_id,
        "source_temporal_segment_id": source_id,
        "recording_id": recording_id,
        "status": "PENDING_HUMAN_REVIEW",
        "editable": True,
        "accepted": False,
        "review_required": True,
        "automatic_eligible": False,
        "review_status": "PENDING",
        "decision": "pending",
        "decision_options": list(DECISION_OPTIONS),
        "start_seconds": _finite(segment.get("start_seconds"), field=f"{segment_id}.start_seconds"),
        "end_seconds": _finite(segment.get("end_seconds"), field=f"{segment_id}.end_seconds"),
        "start_time_sec": _finite(
            segment.get("start_seconds"), field=f"{segment_id}.start_seconds"
        ),
        "end_time_sec": _finite(segment.get("end_seconds"), field=f"{segment_id}.end_seconds"),
        "boundary_status": BOUNDARY_STATUS,
        "boundary_source": _text(segment.get("boundary_source")) or "wemm_temporal_score",
        "boundary_method": boundary_method,
        "boundary_confidence": _finite(
            segment.get("boundary_confidence"), field=f"{segment_id}.boundary_confidence"
        ),
        "timestamp_basis": timestamp_basis,
        # This is an action-boundary proposal.  The supporting windows remain
        # context and are never reinterpreted as its interval.
        "window_context": {
            "is_action_boundary": False,
            "supporting_window_ids": supporting_windows,
        },
        "provisional_id": _text(segment.get("provisional_id")),
        "label_text": _label_text(segment, values),
        "label_variant": _text(segment.get("label_variant")),
        "verb": values["verb"],
        "noun": values["noun"],
        "attributes": values["attributes"],
        "location": values["location"],
        "hand": values["hand"],
        "structured_labels": {
            name: {"value": values[name], "status": statuses[name]} for name in _FIELD_NAMES
        },
        "field_status": statuses,
        "confidence": confidence,
        "confidence_status": "MEASURED" if confidence is not None else "NOT_MEASURED",
        "evidence": evidence,
        "evidence_cameras": _evidence_cameras(evidence),
        "camera_support": cameras,
        "camera_support_count": segment.get("camera_support_count", len(cameras)),
        "top_k": top_k,
        "margin": margin,
        "temporal_score": _finite(segment.get("peak_score"), field=f"{segment_id}.peak_score"),
        "supporting_window_ids": supporting_windows,
        "source_model": "wemm",
        "source_sidecar_format": sidecar_format,
        "source_ref": _source_ref(segment, source, recording_id),
        "source_temporal_proposal": _copy_json(segment, field=f"{segment_id}.raw_proposal"),
        "review": {
            "decision": "pending",
            "reviewer_id": None,
            "reviewed_at": None,
            "notes": None,
        },
    }


def project_temporal_interval_proposals(source: Mapping[str, Any]) -> dict[str, Any]:
    """Build a detached review-only projection from a WeMM sidecar.

    The returned ``temporal_interval_proposals`` list is intentionally
    independent of ``windows``.  This lets a reviewer accept/edit a model
    interval without implying that the context window containing it is an
    action boundary.
    """

    payload = _mapping(source, field="source")
    _assert_no_gold(payload)
    source_format = _text(payload.get("format"))
    if source_format is not None and source_format not in _SUPPORTED_SOURCE_FORMATS:
        raise ProductionWemmTemporalProjectionError(f"unsupported source format {source_format!r}")
    segments, sidecar_format = _sidecar_segments(payload)
    proposals = [
        _project_segment(segment, source=payload, index=index, sidecar_format=sidecar_format)
        for index, segment in enumerate(segments)
    ]
    seen: set[str] = set()
    for proposal in proposals:
        proposal_id = str(proposal["segment_id"])
        if proposal_id in seen:
            raise ProductionWemmTemporalProjectionError(
                f"duplicate temporal proposal id: {proposal_id}"
            )
        seen.add(proposal_id)
    return {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": STATUS if proposals else "EMPTY",
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "recording_id": _recording_id(payload),
        "source_format": source_format,
        "sidecar_format": sidecar_format,
        "temporal_interval_proposals": proposals,
        "review_contract": {
            "decision_options": list(DECISION_OPTIONS),
            "temporal_interval_fields": list(TEMPORAL_INTERVAL_FIELDS),
            "boundary_status": BOUNDARY_STATUS,
            "proposals_only": True,
            "human_review_required": True,
            "review_required": True,
            "automatic_eligible": False,
            "window_context_only": True,
            "windows_are_not_action_boundaries": True,
            "top_k_preserved_verbatim": True,
        },
        "metrics": {
            "temporal_interval_proposal_count": len(proposals),
            "review_required_count": len(proposals),
            "automatic_eligible_count": 0,
            "measured_interval_count": len(proposals),
        },
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "temporal_sidecar_read": bool(
                segments
                or payload.get("temporal_resolution") is not None
                or payload.get("temporal_segments") is not None
            ),
            "temporal_proposals_projected": bool(proposals),
            "predictions_copied_to_gold": False,
            "hash_or_digest_computed": False,
        },
        "limitations": [
            "Temporal intervals are model-derived proposals, not measured gold boundaries.",
            (
                "Every temporal proposal requires explicit human "
                "accept/edit/split/reject/abstain review."
            ),
            "Supporting processing windows remain context only and are not action boundaries.",
        ],
    }


def attach_temporal_interval_proposals(
    draft: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach temporal proposals to an existing annotation draft.

    The input draft is copied and never mutated.  Existing window segments and
    metrics remain intact; temporal proposals are additive at the draft root so
    downstream consumers can opt in without changing the legacy window shape.
    """

    draft_map = _mapping(draft, field="draft")
    projection = project_temporal_interval_proposals(source)
    copied = _copy_json(draft_map, field="draft")
    if not isinstance(copied, dict):  # pragma: no cover - helper invariant
        raise ProductionWemmTemporalProjectionError("draft must be an object")
    proposals = _copy_json(
        projection["temporal_interval_proposals"], field="projection.temporal_interval_proposals"
    )
    copied["temporal_interval_proposals"] = proposals
    copied["temporal_interval_projection"] = {
        key: _copy_json(value, field=f"projection.{key}")
        for key, value in projection.items()
        if key not in {"temporal_interval_proposals"}
    }
    metrics = copied.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        copied["metrics"] = metrics
    metrics["temporal_interval_proposal_count"] = len(proposals)
    metrics["temporal_interval_review_required_count"] = len(proposals)
    metrics["temporal_interval_automatic_eligible_count"] = 0
    contract = copied.get("review_contract")
    if not isinstance(contract, dict):
        contract = {}
        copied["review_contract"] = contract
    contract["temporal_interval_fields"] = list(TEMPORAL_INTERVAL_FIELDS)
    contract["temporal_interval_proposals_separate"] = True
    contract["temporal_boundaries_are_model_proposals"] = True
    contract["temporal_proposals_require_human_review"] = True
    controls = copied.get("controls")
    if not isinstance(controls, dict):
        controls = {}
        copied["controls"] = controls
    controls["temporal_sidecar_read"] = projection["controls"]["temporal_sidecar_read"]
    controls["temporal_proposals_projected"] = bool(proposals)
    controls["temporal_proposals_to_gold"] = False
    limitations = copied.get("limitations")
    if not isinstance(limitations, list):
        limitations = []
        copied["limitations"] = limitations
    marker = (
        "Model-derived temporal proposals are separate from window segments and remain review-only."
    )
    if marker not in limitations:
        limitations.append(marker)
    return copied


def validate_temporal_interval_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a projection or attached draft's additive temporal fields."""

    payload = _mapping(value, field="projection")
    raw_projection = payload.get("temporal_interval_projection")
    target = raw_projection if isinstance(raw_projection, Mapping) else payload
    if target.get("format") != FORMAT:
        raise ProductionWemmTemporalProjectionError(f"projection format must be {FORMAT!r}")
    if target.get("authority") != AUTHORITY:
        raise ProductionWemmTemporalProjectionError("projection authority must remain local-only")
    if target.get("production_eligible") is not False:
        raise ProductionWemmTemporalProjectionError("projection production_eligible must be false")
    proposals_value = payload.get("temporal_interval_proposals")
    if proposals_value is None and target is not payload:
        proposals_value = target.get("temporal_interval_proposals", [])
    proposals = _sequence(proposals_value or [], field="temporal_interval_proposals")
    seen: set[str] = set()
    for index, raw in enumerate(proposals):
        proposal = _mapping(raw, field=f"temporal_interval_proposals[{index}]")
        proposal_id = _text(proposal.get("segment_id"))
        if proposal_id is None or proposal_id in seen:
            raise ProductionWemmTemporalProjectionError(
                f"temporal_interval_proposals[{index}].segment_id must be unique"
            )
        seen.add(proposal_id)
        if proposal.get("review_required") is not True:
            raise ProductionWemmTemporalProjectionError(
                f"{proposal_id}.review_required must be true"
            )
        if proposal.get("automatic_eligible") is not False:
            raise ProductionWemmTemporalProjectionError(
                f"{proposal_id}.automatic_eligible must be false"
            )
        if proposal.get("boundary_status") != BOUNDARY_STATUS:
            raise ProductionWemmTemporalProjectionError(
                f"{proposal_id}.boundary_status must be {BOUNDARY_STATUS}"
            )
        start = _finite(proposal.get("start_seconds"), field=f"{proposal_id}.start_seconds")
        end = _finite(proposal.get("end_seconds"), field=f"{proposal_id}.end_seconds")
        if start is None or end is None or start < 0.0 or end <= start:
            raise ProductionWemmTemporalProjectionError(
                f"{proposal_id} interval must satisfy 0 <= start < end"
            )
        top_k = _sequence(proposal.get("top_k", []), field=f"{proposal_id}.top_k")
        if not isinstance(top_k, Sequence) or isinstance(top_k, (str, bytes, bytearray)):
            raise ProductionWemmTemporalProjectionError(f"{proposal_id}.top_k must be an array")
        decision = _text(proposal.get("decision"))
        if decision not in {"pending", *DECISION_OPTIONS}:
            raise ProductionWemmTemporalProjectionError(f"{proposal_id}.decision is invalid")
        _assert_no_gold(proposal, field=proposal_id)
    detached = _copy_json(payload, field="projection")
    if not isinstance(detached, dict):  # pragma: no cover - helper invariant
        raise ProductionWemmTemporalProjectionError("projection must be an object")
    return detached


__all__ = [
    "AUTHORITY",
    "BOUNDARY_STATUS",
    "DECISION_OPTIONS",
    "FORMAT",
    "STATUS",
    "TEMPORAL_INTERVAL_FIELDS",
    "ProductionWemmTemporalProjectionError",
    "attach_temporal_interval_proposals",
    "project_temporal_interval_proposals",
    "validate_temporal_interval_projection",
]
