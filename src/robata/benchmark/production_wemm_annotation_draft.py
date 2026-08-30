"""Project recorded WeMM proposals into an editable production draft.

This is a deliberately small, inference-free bridge between the production
WeMM review pack and a human annotation surface.  It consumes the proposal
already written by the WeMM runner (or its aggregate) and exposes the fields
required by :mod:`data/source/annotation-principal.txt`:
``start_seconds``, ``end_seconds``, ``verb``, ``noun``, optional attributes,
location and hand, confidence, evidence, camera support, and the complete
Top-K context.

The bridge does *not* turn a processing window into an action span.  WeMM's
current proposals generally have ``proposal_interval.status=NOT_MEASURED``;
their draft segment therefore keeps null boundaries and records the window as
context only.  An explicitly source-bound proposal interval is copied only
when its status and numbers make that provenance clear.  Unknown, abstain and
split remain first-class review states and every draft is pending a human
decision.  The output is a local pre-annotation artifact, never gold or a
published annotation schema.

No model/media/gold operation is performed and no hash/digest is calculated.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

FORMAT: Final = "robata-production-wemm-annotation-draft-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
OFFICIAL_GOLD_STATUS: Final = "NOT_ESTABLISHED"
DECISION_OPTIONS: Final = ("accept", "edit", "split", "reject", "abstain")
ANNOTATION_FIELDS: Final = (
    "start_seconds",
    "end_seconds",
    "verb",
    "noun",
    "attributes",
    "location",
    "hand",
    "confidence",
    "evidence",
)
STRUCTURED_FIELDS: Final = ("verb", "noun", "attributes", "location", "hand")
REVIEW_CONTEXT_FIELDS: Final = (
    "camera_ids",
    "camera_support",
    "top_k",
    "margin",
)
REVIEW_REQUIRED_FIELDS: Final = (
    *ANNOTATION_FIELDS,
    "camera_support",
    "top_k",
    "margin",
)
WINDOW_REVIEW_CONTEXT_FIELDS: Final = ("camera_ids",)
STATUS_FIELDS: Final = (
    "window_status",
    "window_decision",
    "proposal_status",
    "decision",
    "split_hint",
)
WINDOW_CONTEXT_FIELDS: Final = ("start_seconds", "end_seconds")
PROVENANCE_FIELDS: Final = (
    "qa_status",
    "source_preflight_status",
    "review_pack_path",
    "archive_member",
    "source_path",
)
_SUPPORTED_FORMATS: Final = frozenset(
    {
        "robata-production-wemm-review-pack-aggregate-v1",
        "robata-production-wemm-preannotation-review-pack-v1",
        "robata-production-wemm-preannotation-v1",
    }
)
_VALID_BOUNDARY_STATUSES: Final = frozenset(
    {"MEASURED", "EXPLICIT", "SOURCE_BOUND", "SOURCE_ABSOLUTE"}
)
_WINDOW_ONLY_BOUNDARY_STATUSES: Final = frozenset(
    {"WINDOW_BOUND_ONLY", "FIXED_WINDOW", "WINDOW_CONTEXT_ONLY", "NOT_MEASURED", "NOT_OBSERVABLE"}
)
_VALID_WINDOW_STATUSES: Final = frozenset(
    {"PROPOSALS_AVAILABLE", "UNKNOWN", "ABSTAIN", "SPLIT", "PROVISIONAL"}
)
_VALID_PROPOSAL_STATUSES: Final = frozenset({"PROPOSED", "UNKNOWN", "ABSTAIN", "SPLIT"})
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


class ProductionWemmAnnotationDraftError(ValueError):
    """Raised when a recorded WeMM review artifact cannot be projected."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmAnnotationDraftError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmAnnotationDraftError(f"{field} must be an array")
    return value


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _copy_json(value: object, *, field: str = "value") -> Any:
    """Copy JSON-compatible input and reject non-finite numbers."""

    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmAnnotationDraftError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmAnnotationDraftError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionWemmAnnotationDraftError(f"{field} must be JSON-compatible")


def _key_token(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _assert_no_gold(value: object, *, field: str = "input") -> None:
    """Keep model-only input separate from official/human labels.

    Existing review packs intentionally contain fields named ``review`` and
    ``review_status``.  Only explicit gold/official/adjudicated key names are
    rejected here; this is a provenance check, not a heavy defensive layer.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProductionWemmAnnotationDraftError(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmAnnotationDraftError(f"{field} keys must be strings")
            token = _key_token(key)
            if token in _GOLD_KEY_FRAGMENTS:
                raise ProductionWemmAnnotationDraftError(
                    f"{field}.{key} contains gold/official annotation data"
                )
            _assert_no_gold(child, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_gold(child, field=f"{field}[{index}]")
        return
    raise ProductionWemmAnnotationDraftError(f"{field} must be JSON-compatible")


def _normalise_field(value: object, *, field: str) -> tuple[Any, str]:
    """Read one explicit structured field without parsing free prose."""

    if isinstance(value, Mapping):
        status = _text(value.get("status")).upper()
        raw_value = value.get("value")
        if not status:
            status = "MEASURED" if raw_value is not None else "NOT_OBSERVABLE"
    else:
        raw_value = value
        status = "MEASURED" if value is not None else "NOT_MEASURED"
    if status not in {"MEASURED", "NOT_MEASURED", "NOT_OBSERVABLE"}:
        # Preserve unknown provenance as an unmeasured field rather than
        # inventing a new semantic status in this adapter.
        status = "NOT_MEASURED"
    if status != "MEASURED" or raw_value is None:
        return None, status
    if isinstance(raw_value, str):
        raw_value = raw_value.strip() or None
        if raw_value is None:
            return None, "NOT_OBSERVABLE"
    try:
        return _copy_json(raw_value, field=f"{field}.value"), "MEASURED"
    except ProductionWemmAnnotationDraftError:
        raise


def _proposal_labels(proposal: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    raw_labels = proposal.get("structured_labels")
    labels: Mapping[str, Any] = raw_labels if isinstance(raw_labels, Mapping) else proposal
    values: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for field in STRUCTURED_FIELDS:
        values[field], statuses[field] = _normalise_field(
            labels.get(field), field=f"proposal.structured_labels.{field}"
        )
    return values, statuses


def _fallback_label_text(proposal: Mapping[str, Any], values: Mapping[str, Any]) -> str | None:
    raw = proposal.get("label_text")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    verb, noun = values.get("verb"), values.get("noun")
    if isinstance(verb, str) and verb.strip() and isinstance(noun, str) and noun.strip():
        return f"{verb.strip()} {noun.strip()}"
    return None


def _proposal_interval(
    proposal: Mapping[str, Any],
) -> tuple[float | None, float | None, str, str | None]:
    raw = proposal.get("proposal_interval")
    status: str | None = None
    if isinstance(raw, Mapping):
        status = _text(raw.get("status")).upper() or None
        start = _finite(raw.get("start_seconds", raw.get("start_time_sec")))
        end = _finite(raw.get("end_seconds", raw.get("end_time_sec")))
    elif (
        isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) and len(raw) == 2
    ):
        start, end = _finite(raw[0]), _finite(raw[1])
    else:
        start = _finite(proposal.get("start_seconds", proposal.get("start_time_sec")))
        end = _finite(proposal.get("end_seconds", proposal.get("end_time_sec")))
        status = _text(proposal.get("boundary_status")).upper() or None

    if status is None:
        status = "UNMARKED" if start is not None or end is not None else "NOT_MEASURED"
    basis = _text(proposal.get("timestamp_basis")) or None
    if status not in _VALID_BOUNDARY_STATUSES:
        return None, None, status, basis
    if start is None or end is None or start < 0 or end <= start:
        return None, None, "INVALID", basis
    return start, end, status, basis


def _camera_ids(value: object) -> list[str]:
    if isinstance(value, Mapping):
        # Camera support is emitted as a list by the current runner, but a
        # few older sidecars used ``{"cam_01": {...}}``.  Materialise the
        # mapping keys before the Sequence check; ``dict_keys`` itself is not
        # a Sequence and would otherwise silently erase valid support.
        value = list(value.keys())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return sorted({item.strip() for item in value if isinstance(item, str) and item.strip()})


def _evidence_cameras(evidence: Sequence[Any]) -> list[str]:
    cameras: set[str] = set()
    for item in evidence:
        if isinstance(item, Mapping):
            camera = item.get("camera_id")
            if isinstance(camera, str) and camera.strip():
                cameras.add(camera.strip())
    return sorted(cameras)


def _confidence(value: object) -> tuple[float | None, str]:
    if value is None:
        return None, "NOT_MEASURED"
    number = _finite(value)
    if number is None or not 0.0 <= number <= 1.0:
        return None, "INVALID"
    return number, "MEASURED"


def _context_interval(item: Mapping[str, Any]) -> tuple[float | None, float | None, str]:
    raw = item.get("source_interval", item.get("interval"))
    status = "WINDOW_CONTEXT_ONLY"
    if isinstance(raw, Mapping):
        start = _finite(raw.get("start_seconds", raw.get("start_time_sec")))
        end = _finite(raw.get("end_seconds", raw.get("end_time_sec")))
        raw_status = _text(raw.get("status")).upper()
        if raw_status:
            status = raw_status
    elif (
        isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) and len(raw) == 2
    ):
        start, end = _finite(raw[0]), _finite(raw[1])
    else:
        start = _finite(item.get("start_seconds", item.get("start_time_sec")))
        end = _finite(item.get("end_seconds", item.get("end_time_sec")))
    if start is None or end is None or start < 0 or end <= start:
        return None, None, "NOT_MEASURED"
    return start, end, status


def _input_rows(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], str]:
    fmt = _text(payload.get("format"))
    if fmt not in _SUPPORTED_FORMATS:
        raise ProductionWemmAnnotationDraftError(
            f"unsupported input format {fmt!r}; expected a production WeMM "
            "review/preannotation artifact"
        )
    raw = payload.get("items")
    if raw is None:
        raw = payload.get("windows")
    rows = _sequence(raw, field="input.items/windows")
    result: list[Mapping[str, Any]] = []
    for index, value in enumerate(rows):
        result.append(_mapping(value, field=f"input.items/windows[{index}]"))
    return result, fmt


def _recording_id(item: Mapping[str, Any], payload: Mapping[str, Any]) -> str | None:
    for source in (item, item.get("source_ref"), payload.get("source")):
        if not isinstance(source, Mapping):
            continue
        for key in ("recording_id", "source_id", "id"):
            value = _text(source.get(key))
            if value:
                return value
        nested = source.get("source")
        if isinstance(nested, Mapping):
            value = _text(nested.get("recording_id"))
            if value:
                return value
    return None


def _source_provenance(item: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Retain source/QA lineage alongside the editable draft.

    The aggregate review pack stores QA and preflight values under the item
    ``provenance`` object, while archive/path lineage lives under
    ``source_ref``.  Older pre-annotation packs may put those values directly
    under ``source``.  Keep all supplied snapshots and expose a small,
    consistent projection so a later reviewer can trace a draft without
    mistaking QA metadata for an action claim.
    """

    raw_provenance = item.get("provenance")
    raw_source_ref = item.get("source_ref")
    source_candidates: list[Mapping[str, Any]] = []
    # Some early per-window packs placed QA/path keys directly on the item;
    # inspect that row first, then the normalized provenance/ref snapshots.
    for raw in (item, raw_provenance, raw_source_ref):
        if isinstance(raw, Mapping):
            source_candidates.append(raw)
            nested = raw.get("source")
            if isinstance(nested, Mapping):
                source_candidates.append(nested)
    payload_source = payload.get("source")
    if isinstance(payload_source, Mapping):
        source_candidates.append(payload_source)

    def first_text(*keys: str) -> str | None:
        for candidate in source_candidates:
            for key in keys:
                value = _text(candidate.get(key))
                if value:
                    return value
        return None

    result: dict[str, Any] = {
        "qa_status": first_text("qa_status", "qa", "quality_status"),
        "source_preflight_status": first_text("source_preflight_status", "preflight_status"),
        "review_pack_path": first_text("review_pack_path"),
        "archive_member": first_text("archive_member"),
        "source_path": first_text("source_path", "path", "media_path", "mcap_path", "video_path"),
    }
    # Keep the unmodified input provenance snapshots as nested fields.  This
    # avoids dropping newly-added lineage keys while preserving a stable set
    # of fields for metrics and review tooling.
    if isinstance(raw_provenance, Mapping):
        result["input_provenance"] = _copy_json(
            raw_provenance, field="source_provenance.provenance"
        )
    if isinstance(raw_source_ref, Mapping):
        result["input_source_ref"] = _copy_json(
            raw_source_ref, field="source_provenance.source_ref"
        )
    result["window_context_only"] = True
    return result


def _recording_source_index(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index aggregate recording source snapshots for per-window review.

    The aggregate intentionally flattens its ``items`` rows and keeps the
    complete archive/path metadata under ``recordings``.  A draft reviewer
    needs that metadata next to a window because the staged path may already
    have been removed after inference.  This helper only copies recorded
    metadata; it never resolves paths or opens media.
    """

    result: dict[str, dict[str, Any]] = {}
    raw_recordings = payload.get("recordings")
    if isinstance(raw_recordings, Sequence) and not isinstance(
        raw_recordings, (str, bytes, bytearray)
    ):
        for index, raw_recording in enumerate(raw_recordings):
            if not isinstance(raw_recording, Mapping):
                continue
            recording_id = _text(raw_recording.get("recording_id"))
            if not recording_id:
                continue
            raw_source = raw_recording.get("source")
            source = raw_source if isinstance(raw_source, Mapping) else raw_recording
            # Aggregate recordings store a compact ``source_ref`` with the
            # complete source snapshot nested under ``source``.  Flatten the
            # two recorded views for reviewers while retaining the original
            # nested snapshot under ``source``.
            nested_source = source.get("source") if isinstance(source, Mapping) else None
            if isinstance(nested_source, Mapping):
                merged_source = dict(nested_source)
                merged_source.update(
                    {
                        key: value
                        for key, value in source.items()
                        if key != "source" and value is not None
                    }
                )
                merged_source["source"] = nested_source
                source = merged_source
            copied = _copy_json(source, field=f"input.recordings[{index}].source")
            if isinstance(copied, dict):
                copied.setdefault("recording_id", recording_id)
                result[recording_id] = copied
    if result:
        return result

    # A single-recording review/preannotation pack has no ``recordings``
    # array.  Preserve its source snapshot under the discovered recording ID.
    single_recording_id = _recording_id(payload, payload)
    raw_source = payload.get("source")
    if single_recording_id and isinstance(raw_source, Mapping):
        copied = _copy_json(raw_source, field="input.source")
        if isinstance(copied, dict):
            copied.setdefault("recording_id", single_recording_id)
            result[single_recording_id] = copied
    return result


def _attach_recording_source(
    window: dict[str, Any], source_index: Mapping[str, Mapping[str, Any]]
) -> None:
    """Attach a detached recording source snapshot to one draft window."""

    recording_id = _text(window.get("recording_id"))
    if not recording_id:
        return
    source = source_index.get(recording_id)
    if not isinstance(source, Mapping):
        return
    source_copy = _copy_json(source, field=f"{recording_id}.recording_source")
    window["recording_source"] = source_copy
    provenance = window.get("source_provenance")
    if not isinstance(provenance, dict):
        return
    # Keep the stable five-field projection while filling useful lineage that
    # is otherwise only present in aggregate.recordings.
    for key in (
        "archive_path",
        "manifest_format",
        "media_type",
        "camera_count",
        "common_duration_seconds",
        "common_start_timestamp_ns",
        "common_end_timestamp_ns",
        "path_lifecycle",
        "staging_lifecycle",
    ):
        if key not in provenance or provenance.get(key) is None:
            value = source.get(key)
            if value is not None:
                provenance[key] = _copy_json(value, field=f"{recording_id}.source.{key}")
    provenance["recording_source"] = source_copy
    source_ref = window.get("source_ref")
    if isinstance(source_ref, dict):
        for key in ("archive_path", "manifest_format", "media_type", "camera_count"):
            if key not in source_ref or source_ref.get(key) is None:
                value = source.get(key)
                if value is not None:
                    source_ref[key] = _copy_json(value, field=f"{recording_id}.source_ref.{key}")


def _window_status(item: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]]) -> str:
    raw = _text(item.get("window_status", item.get("status"))).upper()
    if raw in _VALID_WINDOW_STATUSES:
        return raw
    decision = _text(item.get("window_decision", item.get("decision"))).casefold()
    if decision == "abstain":
        return "ABSTAIN"
    if not proposals:
        return "UNKNOWN"
    proposal_statuses = {
        _text(proposal.get("proposal_status")).upper()
        for proposal in proposals
        if _text(proposal.get("proposal_status"))
    }
    if proposal_statuses and proposal_statuses <= {"ABSTAIN"}:
        return "ABSTAIN"
    if proposal_statuses and proposal_statuses <= {"UNKNOWN"}:
        return "UNKNOWN"
    if "SPLIT" in proposal_statuses:
        return "SPLIT"
    return "PROPOSALS_AVAILABLE"


def _proposal_segment(
    proposal: Mapping[str, Any],
    *,
    window_id: str,
    segment_index: int,
    context_interval: tuple[float | None, float | None, str],
) -> dict[str, Any]:
    values, statuses = _proposal_labels(proposal)
    label_text = _fallback_label_text(proposal, values)
    start, end, boundary_status, timestamp_basis = _proposal_interval(proposal)
    evidence_raw = proposal.get("evidence", [])
    if isinstance(evidence_raw, str):
        evidence: list[Any] = [evidence_raw] if evidence_raw.strip() else []
    elif isinstance(evidence_raw, Sequence) and not isinstance(
        evidence_raw, (str, bytes, bytearray)
    ):
        evidence = _copy_json(evidence_raw, field=f"{window_id}.proposal.evidence")
    else:
        evidence = []
    if not isinstance(evidence, list):  # pragma: no cover - copy invariant
        evidence = []
    confidence, confidence_status = _confidence(proposal.get("confidence"))
    camera_support = _camera_ids(proposal.get("camera_support", proposal.get("camera_ids", [])))
    evidence_cameras = _evidence_cameras(evidence)
    if not camera_support:
        camera_support = evidence_cameras
    proposal_status = _text(proposal.get("proposal_status")).upper() or "PROPOSED"
    if proposal_status not in _VALID_PROPOSAL_STATUSES:
        proposal_status = "UNKNOWN"
    proposal_id = _text(proposal.get("proposal_id")) or f"{window_id}-p{segment_index + 1:02d}"
    raw_source_decision = _text(proposal.get("decision")).casefold() or "pending"
    return {
        "segment_id": proposal_id,
        "source_proposal_id": proposal_id,
        "status": "PENDING_HUMAN_REVIEW",
        "editable": True,
        "accepted": False,
        "review_required": True,
        "automatic_eligible": False,
        "review_status": "PENDING",
        "decision": "pending",
        "source_decision": raw_source_decision,
        "decision_options": list(DECISION_OPTIONS),
        "review": {
            "decision": "pending",
            "reviewer_id": None,
            "reviewed_at": None,
            "notes": None,
        },
        "start_seconds": start,
        "end_seconds": end,
        "start_time_sec": start,
        "end_time_sec": end,
        "boundary_status": boundary_status,
        "timestamp_basis": timestamp_basis,
        "window_context": {
            "start_seconds": context_interval[0],
            "end_seconds": context_interval[1],
            "status": context_interval[2],
            "is_action_boundary": False,
        },
        "verb": values["verb"],
        "noun": values["noun"],
        "attributes": values["attributes"],
        "location": values["location"],
        "hand": values["hand"],
        "structured_labels": {
            field: {"value": values[field], "status": statuses[field]}
            for field in STRUCTURED_FIELDS
        },
        "field_status": statuses,
        "label_text": label_text,
        "confidence": confidence,
        "confidence_status": confidence_status,
        "evidence": evidence,
        "evidence_cameras": evidence_cameras,
        "camera_support": camera_support,
        "top_k": _copy_json(proposal.get("top_k", []), field=f"{window_id}.proposal.top_k"),
        "margin": _finite(proposal.get("margin")),
        "proposal_status": proposal_status,
        "unknown": proposal_status == "UNKNOWN",
        "abstain": proposal_status == "ABSTAIN",
        "split_hint": bool(proposal.get("split_hint", False)) or proposal_status == "SPLIT",
        "source_model": "wemm",
        "raw_proposal": _copy_json(proposal, field=f"{window_id}.proposal"),
    }


def _window_draft(
    item: Mapping[str, Any], payload: Mapping[str, Any], index: int
) -> dict[str, Any]:
    window_id = _text(item.get("window_id"))
    if not window_id:
        raise ProductionWemmAnnotationDraftError(f"items[{index}].window_id is required")
    raw_proposals = item.get("proposals", item.get("predictions", item.get("candidates", [])))
    proposals = [
        _mapping(value, field=f"{window_id}.proposals[{proposal_index}]")
        for proposal_index, value in enumerate(
            _sequence(raw_proposals, field=f"{window_id}.proposals")
        )
    ]
    context = _context_interval(item)
    status = _window_status(item, proposals)
    split_requested = status == "SPLIT" or any(bool(row.get("split_hint")) for row in proposals)

    # A normal WeMM window has one fused proposal.  If multiple proposals are
    # explicitly marked split, expose each as an editable segment; otherwise
    # keep alternatives in candidate_proposals and draft only the first one.
    # Track source indexes explicitly so UNKNOWN/ABSTAIN proposals are not
    # silently dropped from the review queue when they cannot form a segment.
    selected_indexes = (
        list(range(len(proposals))) if split_requested else ([0] if proposals else [])
    )
    segments: list[dict[str, Any]] = []
    represented_indexes: set[int] = set()
    for proposal_index in selected_indexes:
        proposal = proposals[proposal_index]
        if _text(proposal.get("proposal_status")).upper() in {"UNKNOWN", "ABSTAIN"}:
            continue
        segments.append(
            _proposal_segment(
                proposal,
                window_id=window_id,
                segment_index=proposal_index,
                context_interval=context,
            )
        )
        represented_indexes.add(proposal_index)
    if status in {"ABSTAIN", "UNKNOWN"} or not proposals:
        draft_status = status if status in {"ABSTAIN", "UNKNOWN"} else "UNKNOWN"
    elif split_requested:
        draft_status = "SPLIT_REVIEW"
    else:
        draft_status = "PROVISIONAL"

    raw_decision = _text(item.get("window_decision", item.get("decision"))).casefold()
    decision = raw_decision if raw_decision in DECISION_OPTIONS else "pending"
    review_status = _text(item.get("review_status")).upper() or "PENDING"
    camera_ids = _camera_ids(item.get("camera_ids", item.get("cameras", [])))
    if not camera_ids:
        camera_ids = sorted(
            {
                camera
                for proposal in proposals
                for camera in _camera_ids(
                    proposal.get("camera_support", proposal.get("camera_ids", []))
                )
            }
        )
    source_ref = item.get("source_ref", item.get("provenance"))
    source_ref_copy = (
        _copy_json(source_ref, field=f"{window_id}.source_ref") if source_ref else None
    )
    source_provenance = _source_provenance(item, payload)
    if source_ref_copy is None:
        # Keep a usable source reference even for older one-recording packs
        # that omitted ``source_ref``.  It is a locator snapshot, not a new
        # identity value.
        source_ref_copy = {
            "recording_id": _recording_id(item, payload),
            "review_pack_path": source_provenance.get("review_pack_path"),
            "source_path": source_provenance.get("source_path"),
            "archive_member": source_provenance.get("archive_member"),
        }
    return {
        "ordinal": item.get("ordinal", index),
        "window_id": window_id,
        "recording_id": _recording_id(item, payload),
        "camera_ids": camera_ids,
        "source_ref": source_ref_copy,
        "source_provenance": source_provenance,
        "qa_status": source_provenance.get("qa_status"),
        "source_preflight_status": source_provenance.get("source_preflight_status"),
        "source_interval": {
            "start_seconds": context[0],
            "end_seconds": context[1],
            "status": context[2],
            "is_action_boundary": False,
        },
        "window_status": status,
        "status": draft_status,
        "decision": decision,
        "review_status": review_status,
        "source_decision": raw_decision or "pending",
        "decision_options": list(DECISION_OPTIONS),
        "unknown": status == "UNKNOWN" or not proposals,
        "abstain": status == "ABSTAIN",
        "split_requested": split_requested,
        "annotation_draft": {
            "status": draft_status,
            "segments": segments,
            "editable": True,
            "human_review_required": True,
        },
        "candidate_proposals": _copy_json(
            [
                proposal
                for index, proposal in enumerate(proposals)
                if index not in represented_indexes
            ],
            field=f"{window_id}.candidate_proposals",
        ),
        "raw_candidates": _copy_json(
            item.get("raw_candidates", item.get("predictions", [])),
            field=f"{window_id}.raw_candidates",
        ),
        "review": {
            "status": review_status,
            "decision": decision,
            "reviewer_id": None,
            "reviewed_at": None,
            "notes": None,
        },
        "provenance": {
            "source_format": payload.get("format"),
            "model": "WeMM",
            "epic_ontology_used": False,
            "mapper_used": False,
            "window_context_only": True,
        },
    }


def build_wemm_annotation_draft(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build an editable, review-only draft from an existing WeMM artifact."""

    payload = _mapping(value, field="input")
    _assert_no_gold(payload)
    rows, source_format = _input_rows(payload)
    windows = [_window_draft(row, payload, index) for index, row in enumerate(rows)]
    recording_source_index = _recording_source_index(payload)
    for window in windows:
        _attach_recording_source(window, recording_source_index)
    source = _copy_json(payload.get("source", {}), field="input.source")
    raw_label_space: object = payload.get("label_space", {})
    if not isinstance(raw_label_space, Mapping):
        raw_label_space = {}
    # Aggregates intentionally flatten recording metadata.  Recover the
    # owner-scoped label-space snapshot from the first recording when the
    # aggregate itself has no top-level label_space; this is provenance only,
    # not a label join or a vocabulary rewrite.
    if not raw_label_space:
        recordings = payload.get("recordings")
        if isinstance(recordings, Sequence) and not isinstance(recordings, (str, bytes, bytearray)):
            for raw_recording in recordings:
                if not isinstance(raw_recording, Mapping):
                    continue
                candidate_space = raw_recording.get("label_space")
                if isinstance(candidate_space, Mapping):
                    raw_label_space = candidate_space
                    break
    label_space = _copy_json(raw_label_space, field="input.label_space")
    if not isinstance(label_space, dict):  # pragma: no cover - copy invariant
        label_space = {}
    # Preserve the input's owner-scoped label space but assert the key
    # provenance facts in the emitted draft.  We do not import EPIC labels or
    # map candidates into another catalog.
    label_space.setdefault("epic_ontology_used", False)
    label_space.setdefault("mapper_used", False)
    label_space.setdefault("kind", "OPEN_PROVISIONAL_PHRASES")
    status_counts = Counter(str(window["status"]) for window in windows)
    qa_status_counts = Counter(
        str(status).upper()
        for window in windows
        if (status := window["source_provenance"].get("qa_status"))
    )
    source_preflight_status_counts = Counter(
        str(status).upper()
        for window in windows
        if (status := window["source_provenance"].get("source_preflight_status"))
    )
    segment_count = sum(len(window["annotation_draft"]["segments"]) for window in windows)
    recording_count = len(
        {
            _text(window.get("recording_id"))
            for window in windows
            if _text(window.get("recording_id"))
        }
    )
    camera_window_input_count = sum(
        len(_camera_ids(window.get("camera_ids", []))) for window in windows
    )
    measured_boundaries = sum(
        sum(
            segment.get("boundary_status") in _VALID_BOUNDARY_STATUSES
            for segment in window["annotation_draft"]["segments"]
        )
        for window in windows
    )
    top_k_count = sum(
        sum(bool(segment.get("top_k")) for segment in window["annotation_draft"]["segments"])
        for window in windows
    )
    return {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": "PENDING_REVIEW" if windows else "EMPTY",
        "production_eligible": False,
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": OFFICIAL_GOLD_STATUS,
        "human_adjudication": "NOT_PERFORMED",
        "quality_claim": False,
        "source": source,
        "label_space": label_space,
        "windows": windows,
        "metrics": {
            "recording_count": recording_count,
            "window_count": len(windows),
            "segment_count": segment_count,
            "camera_window_input_count": camera_window_input_count,
            "provisional_window_count": status_counts.get("PROVISIONAL", 0),
            "split_review_window_count": status_counts.get("SPLIT_REVIEW", 0),
            "unknown_window_count": status_counts.get("UNKNOWN", 0),
            "abstain_window_count": status_counts.get("ABSTAIN", 0),
            "measured_boundary_segment_count": measured_boundaries,
            "unmeasured_boundary_segment_count": segment_count - measured_boundaries,
            "windows_with_top_k": top_k_count,
            "status_counts": dict(sorted(status_counts.items())),
            "qa_status_counts": dict(sorted(qa_status_counts.items())),
            "source_preflight_status_counts": dict(sorted(source_preflight_status_counts.items())),
        },
        "review_contract": {
            "decision_options": list(DECISION_OPTIONS),
            "annotation_fields": list(ANNOTATION_FIELDS),
            "structured_label_fields": list(STRUCTURED_FIELDS),
            "required_fields": list(REVIEW_REQUIRED_FIELDS),
            "review_context_fields": list(REVIEW_CONTEXT_FIELDS),
            "window_review_context_fields": list(WINDOW_REVIEW_CONTEXT_FIELDS),
            "window_context_fields": list(WINDOW_CONTEXT_FIELDS),
            "status_fields": list(STATUS_FIELDS),
            # Keep the source/QA provenance contract explicit at the top
            # level as well as on each window.  These fields describe where a
            # proposal came from; they are not an action label or a gold
            # signal.
            "provenance_fields": list(PROVENANCE_FIELDS),
            "model_predictions_are_not_gold": True,
            "human_review_required": True,
            "editable": True,
            "unknown_allowed": True,
            "abstain_allowed": True,
            "split_allowed": True,
            "fixed_window_is_not_action_boundary": True,
            "window_context_only": True,
            "top_k_preserved_verbatim": True,
            "source_format": source_format,
        },
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "predictions_copied_to_gold": False,
            "window_boundaries_as_actions": False,
            "epic_ontology_used": False,
            "mapper_used": False,
            "ontology_modified": False,
            "training_invoked": False,
        },
        "limitations": [
            "This artifact is a WeMM pre-annotation draft, not official gold.",
            "Processing-window intervals are context only; missing action boundaries remain null.",
            "Optional fields remain null unless WeMM explicitly supplied them.",
            "Every retained segment requires accept/edit/split/reject/abstain review.",
        ],
    }


def validate_wemm_annotation_draft(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a generated draft without reading media or review/gold data.

    The checker intentionally validates only the local draft contract.  It
    does not decide whether a label is semantically correct and does not
    promote any row to production eligibility.  A detached copy is returned
    so callers can safely inspect or write the validated artifact.
    """

    payload = _mapping(value, field="draft")
    _assert_no_gold(payload, field="draft")
    if payload.get("format") != FORMAT:
        raise ProductionWemmAnnotationDraftError(f"draft format must be {FORMAT!r}")
    if payload.get("authority") != AUTHORITY:
        raise ProductionWemmAnnotationDraftError("draft authority must remain local-only")
    if payload.get("production_eligible") is not False:
        raise ProductionWemmAnnotationDraftError("draft production_eligible must be false")
    if payload.get("official_quality_status") != OFFICIAL_QUALITY_STATUS:
        raise ProductionWemmAnnotationDraftError(
            "draft official_quality_status must remain NOT_MEASURED"
        )
    if payload.get("official_gold_status") != OFFICIAL_GOLD_STATUS:
        raise ProductionWemmAnnotationDraftError(
            "draft official_gold_status must remain NOT_ESTABLISHED"
        )
    contract = _mapping(payload.get("review_contract"), field="draft.review_contract")
    annotation_fields = _sequence(
        contract.get("annotation_fields", []), field="review_contract.annotation_fields"
    )
    for field in ANNOTATION_FIELDS:
        if field not in annotation_fields:
            raise ProductionWemmAnnotationDraftError(
                f"review_contract.annotation_fields missing {field!r}"
            )
    required_fields = _sequence(
        contract.get("required_fields", []), field="review_contract.required_fields"
    )
    for field in REVIEW_REQUIRED_FIELDS:
        if field not in required_fields:
            raise ProductionWemmAnnotationDraftError(
                f"review_contract.required_fields missing {field!r}"
            )
    review_context_fields = _sequence(
        contract.get("review_context_fields", []),
        field="review_contract.review_context_fields",
    )
    for field in REVIEW_CONTEXT_FIELDS:
        if field not in review_context_fields:
            raise ProductionWemmAnnotationDraftError(
                f"review_contract.review_context_fields missing {field!r}"
            )
    window_review_context_fields = _sequence(
        contract.get("window_review_context_fields", []),
        field="review_contract.window_review_context_fields",
    )
    for field in WINDOW_REVIEW_CONTEXT_FIELDS:
        if field not in window_review_context_fields:
            raise ProductionWemmAnnotationDraftError(
                f"review_contract.window_review_context_fields missing {field!r}"
            )
    window_context_fields = _sequence(
        contract.get("window_context_fields", []),
        field="review_contract.window_context_fields",
    )
    for field in WINDOW_CONTEXT_FIELDS:
        if field not in window_context_fields:
            raise ProductionWemmAnnotationDraftError(
                f"review_contract.window_context_fields missing {field!r}"
            )
    status_fields = _sequence(
        contract.get("status_fields", []), field="review_contract.status_fields"
    )
    for field in STATUS_FIELDS:
        if field not in status_fields:
            raise ProductionWemmAnnotationDraftError(
                f"review_contract.status_fields missing {field!r}"
            )
    provenance_fields = _sequence(
        contract.get("provenance_fields", []), field="review_contract.provenance_fields"
    )
    for field in PROVENANCE_FIELDS:
        if field not in provenance_fields:
            raise ProductionWemmAnnotationDraftError(
                f"review_contract.provenance_fields missing {field!r}"
            )
    if contract.get("fixed_window_is_not_action_boundary") is not True:
        raise ProductionWemmAnnotationDraftError(
            "review contract must keep fixed windows as context only"
        )
    controls = _mapping(payload.get("controls"), field="draft.controls")
    for key in ("gold_read", "gold_written", "predictions_copied_to_gold"):
        if controls.get(key) is not False:
            raise ProductionWemmAnnotationDraftError(f"controls.{key} must be false")

    windows = _sequence(payload.get("windows"), field="draft.windows")
    seen_ids: set[str] = set()
    measured_boundaries = 0
    segment_count = 0
    for index, raw_window in enumerate(windows):
        window = _mapping(raw_window, field=f"draft.windows[{index}]")
        window_id = _text(window.get("window_id"))
        if not window_id:
            raise ProductionWemmAnnotationDraftError(
                f"draft.windows[{index}].window_id is required"
            )
        if window_id in seen_ids:
            raise ProductionWemmAnnotationDraftError(f"duplicate draft window_id: {window_id}")
        seen_ids.add(window_id)
        recording_id = _text(window.get("recording_id"))
        if not recording_id:
            raise ProductionWemmAnnotationDraftError(f"{window_id}.recording_id is required")
        camera_ids = window.get("camera_ids")
        if not isinstance(camera_ids, Sequence) or isinstance(camera_ids, (str, bytes, bytearray)):
            raise ProductionWemmAnnotationDraftError(f"{window_id}.camera_ids must be an array")
        review_status = _text(window.get("review_status")).upper()
        if not review_status:
            raise ProductionWemmAnnotationDraftError(f"{window_id}.review_status is required")
        interval = _mapping(window.get("source_interval"), field=f"{window_id}.source_interval")
        context_status = _text(interval.get("status")).upper()
        context_start = _finite(interval.get("start_seconds"))
        context_end = _finite(interval.get("end_seconds"))
        if (
            context_start is None or context_end is None or context_end <= context_start
        ) and context_status != "NOT_MEASURED":
            raise ProductionWemmAnnotationDraftError(
                f"{window_id}.source_interval must be valid or NOT_MEASURED"
            )
        if interval.get("is_action_boundary") is not False:
            raise ProductionWemmAnnotationDraftError(
                f"{window_id}.source_interval must be context only"
            )
        decision = _text(window.get("decision")).casefold()
        if decision not in {"pending", *DECISION_OPTIONS}:
            raise ProductionWemmAnnotationDraftError(f"{window_id}.decision is invalid")
        draft = _mapping(window.get("annotation_draft"), field=f"{window_id}.annotation_draft")
        segments = _sequence(draft.get("segments"), field=f"{window_id}.annotation_draft.segments")
        segment_count += len(segments)
        for segment_index, raw_segment in enumerate(segments):
            segment = _mapping(
                raw_segment,
                field=f"{window_id}.annotation_draft.segments[{segment_index}]",
            )
            for required in (
                "segment_id",
                "status",
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
                "proposal_status",
                "unknown",
                "abstain",
                "split_hint",
                "decision",
                "decision_options",
                "source_decision",
                "automatic_eligible",
                "review_required",
                "review_status",
            ):
                if required not in segment:
                    raise ProductionWemmAnnotationDraftError(
                        f"{window_id}.segment[{segment_index}] missing {required!r}"
                    )
            if segment.get("status") != "PENDING_HUMAN_REVIEW":
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}] must remain pending review"
                )
            if segment.get("editable") is not True or segment.get("accepted") is not False:
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}] edit/accept flags are invalid"
                )
            if (
                segment.get("automatic_eligible") is not False
                or segment.get("review_required") is not True
            ):
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}] automatic/review flags are invalid"
                )
            if _text(segment.get("review_status")).upper() != "PENDING":
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}].review_status must remain PENDING"
                )
            decision = _text(segment.get("decision")).casefold()
            if decision not in {"pending", *DECISION_OPTIONS}:
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}].decision is invalid"
                )
            decision_options = _sequence(
                segment.get("decision_options", []),
                field=f"{window_id}.segment[{segment_index}].decision_options",
            )
            if any(option not in decision_options for option in DECISION_OPTIONS):
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}].decision_options are incomplete"
                )
            boundary_status = _text(segment.get("boundary_status")).upper()
            start = _finite(segment.get("start_seconds"))
            end = _finite(segment.get("end_seconds"))
            if boundary_status in _WINDOW_ONLY_BOUNDARY_STATUSES:
                if start is not None or end is not None:
                    raise ProductionWemmAnnotationDraftError(
                        f"{window_id}.segment[{segment_index}] context-only boundary must be null"
                    )
            elif boundary_status in _VALID_BOUNDARY_STATUSES:
                if start is None or end is None or end <= start:
                    raise ProductionWemmAnnotationDraftError(
                        f"{window_id}.segment[{segment_index}] measured boundary is invalid"
                    )
                if (
                    context_start is not None
                    and context_end is not None
                    and (start < context_start or end > context_end)
                ):
                    raise ProductionWemmAnnotationDraftError(
                        f"{window_id}.segment[{segment_index}] boundary exceeds context"
                    )
                measured_boundaries += 1
            else:
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}] boundary_status is invalid"
                )
            confidence = segment.get("confidence")
            if confidence is not None:
                number = _finite(confidence)
                if number is None or not 0 <= number <= 1:
                    raise ProductionWemmAnnotationDraftError(
                        f"{window_id}.segment[{segment_index}].confidence is invalid"
                    )
            if not isinstance(segment.get("evidence"), Sequence) or isinstance(
                segment.get("evidence"), (str, bytes, bytearray)
            ):
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}].evidence must be an array"
                )
            if not isinstance(segment.get("camera_support"), Sequence) or isinstance(
                segment.get("camera_support"), (str, bytes, bytearray)
            ):
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}].camera_support must be an array"
                )
            if not isinstance(segment.get("top_k"), Sequence) or isinstance(
                segment.get("top_k"), (str, bytes, bytearray)
            ):
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}].top_k must be an array"
                )
            margin = segment.get("margin")
            if margin is not None and _finite(margin) is None:
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.segment[{segment_index}].margin is invalid"
                )
        provenance = _mapping(
            window.get("source_provenance"), field=f"{window_id}.source_provenance"
        )
        for key in (
            "qa_status",
            "source_preflight_status",
            "review_pack_path",
            "archive_member",
            "source_path",
            "window_context_only",
        ):
            if key not in provenance:
                raise ProductionWemmAnnotationDraftError(
                    f"{window_id}.source_provenance missing {key!r}"
                )
        if provenance.get("window_context_only") is not True:
            raise ProductionWemmAnnotationDraftError(
                f"{window_id}.source_provenance.window_context_only must be true"
            )

    metrics = _mapping(payload.get("metrics"), field="draft.metrics")
    recording_count = len(
        {
            _text(window.get("recording_id"))
            for window in windows
            if _text(window.get("recording_id"))
        }
    )
    camera_window_input_count = sum(
        len(_camera_ids(window.get("camera_ids", []))) for window in windows
    )
    if metrics.get("recording_count") != recording_count:
        raise ProductionWemmAnnotationDraftError("draft.metrics.recording_count disagrees")
    if metrics.get("window_count") != len(windows):
        raise ProductionWemmAnnotationDraftError("draft.metrics.window_count disagrees")
    if metrics.get("segment_count") != segment_count:
        raise ProductionWemmAnnotationDraftError("draft.metrics.segment_count disagrees")
    if metrics.get("camera_window_input_count") != camera_window_input_count:
        raise ProductionWemmAnnotationDraftError(
            "draft.metrics.camera_window_input_count disagrees"
        )
    if metrics.get("measured_boundary_segment_count") != measured_boundaries:
        raise ProductionWemmAnnotationDraftError(
            "draft.metrics.measured_boundary_segment_count disagrees"
        )
    copied = _copy_json(payload, field="draft")
    if not isinstance(copied, dict):  # pragma: no cover - copy invariant
        raise ProductionWemmAnnotationDraftError("draft must be an object")
    return copied


def load_json(value: str | Path) -> dict[str, Any]:
    """Load a JSON object for the CLI."""

    try:
        payload = json.loads(Path(value).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionWemmAnnotationDraftError(f"could not load input: {exc}") from exc
    return dict(_mapping(payload, field="input"))


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise review summary without exposing a gold claim."""

    metrics = _mapping(report.get("metrics", {}), field="report.metrics")
    lines = [
        "# WeMM production annotation draft",
        "",
        f"- Status: `{report.get('status', 'UNKNOWN')}`",
        f"- Official quality: `{report.get('official_quality_status', OFFICIAL_QUALITY_STATUS)}`",
        f"- Official gold: `{report.get('official_gold_status', OFFICIAL_GOLD_STATUS)}`",
        (
            f"- Windows: `{metrics.get('window_count', 0)}`; "
            f"segments: `{metrics.get('segment_count', 0)}`"
        ),
        (
            f"- Provisional: `{metrics.get('provisional_window_count', 0)}`; "
            f"split review: `{metrics.get('split_review_window_count', 0)}`"
        ),
        (
            f"- Unknown: `{metrics.get('unknown_window_count', 0)}`; "
            f"abstain: `{metrics.get('abstain_window_count', 0)}`"
        ),
        (
            f"- Measured boundaries: `{metrics.get('measured_boundary_segment_count', 0)}`; "
            f"context-only/unmeasured: `{metrics.get('unmeasured_boundary_segment_count', 0)}`"
        ),
        "",
        "| Window | Status | Decision | Segments | Context interval |",
        "|---|---|---|---:|---|",
    ]
    raw_windows = report.get("windows", [])
    if isinstance(raw_windows, Sequence) and not isinstance(raw_windows, (str, bytes, bytearray)):
        for raw in raw_windows:
            if not isinstance(raw, Mapping):
                continue
            interval = raw.get("source_interval", {})
            if isinstance(interval, Mapping):
                display_interval = f"{interval.get('start_seconds')}-{interval.get('end_seconds')}"
            else:
                display_interval = "-"
            segments = raw.get("annotation_draft", {}).get("segments", [])
            count = len(segments) if isinstance(segments, Sequence) else 0
            lines.append(
                f"| {raw.get('window_id', '')} | {raw.get('status', '')} | "
                f"{raw.get('decision', 'pending')} | {count} | {display_interval} (context) |"
            )
    lines.extend(["", "Human review is required; this draft is not production gold.", ""])
    return "\n".join(lines)


__all__ = [
    "ANNOTATION_FIELDS",
    "AUTHORITY",
    "DECISION_OPTIONS",
    "FORMAT",
    "OFFICIAL_GOLD_STATUS",
    "OFFICIAL_QUALITY_STATUS",
    "PROVENANCE_FIELDS",
    "REVIEW_CONTEXT_FIELDS",
    "REVIEW_REQUIRED_FIELDS",
    "STATUS_FIELDS",
    "WINDOW_CONTEXT_FIELDS",
    "WINDOW_REVIEW_CONTEXT_FIELDS",
    "ProductionWemmAnnotationDraftError",
    "build_wemm_annotation_draft",
    "load_json",
    "render_markdown",
    "validate_wemm_annotation_draft",
]
