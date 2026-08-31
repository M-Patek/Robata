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
REFINED_BOUNDARY_STATUSES: Final = ("MODEL_REFINED", "MODEL_REFINEMENT_PENDING")
REFINED_BOUNDARY_STATUS: Final = "MODEL_REFINED"
REFINED_PENDING_STATUS: Final = "MODEL_REFINEMENT_PENDING"
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
REFINED_TEMPORAL_INTERVAL_FIELDS: Final = (
    *TEMPORAL_INTERVAL_FIELDS,
    "coarse_interval",
    "refinement_status",
    "onset_request_id",
    "offset_request_id",
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


def _optional_finite(value: object, *, field: str) -> float | None:
    """Validate an optional numeric field while distinguishing omission.

    Pending refinement rows may explicitly use ``None`` for unresolved
    boundaries, but a present string/list/object must not be silently treated
    as if the field were omitted.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionWemmTemporalProjectionError(f"{field} must be finite or null")
    return _finite(value, field=field)


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


def _validate_refined_segment(raw: object, *, field: str) -> dict[str, Any]:
    """Validate one adaptive row, retaining unresolved rows for review."""

    segment = _mapping(raw, field=field)
    if segment.get("review_required") is not True:
        raise ProductionWemmTemporalProjectionError(f"{field}.review_required must be true")
    if segment.get("automatic_eligible") is not False:
        raise ProductionWemmTemporalProjectionError(f"{field}.automatic_eligible must be false")
    boundary_status = _text(segment.get("boundary_status"))
    if boundary_status not in REFINED_BOUNDARY_STATUSES:
        raise ProductionWemmTemporalProjectionError(
            f"{field}.boundary_status must be one of {', '.join(REFINED_BOUNDARY_STATUSES)}"
        )
    coarse = _mapping(segment.get("coarse_interval"), field=f"{field}.coarse_interval")
    coarse_start = _finite(
        coarse.get("start_seconds"), field=f"{field}.coarse_interval.start_seconds"
    )
    coarse_end = _finite(coarse.get("end_seconds"), field=f"{field}.coarse_interval.end_seconds")
    if (
        coarse_start is None
        or coarse_end is None
        or coarse_start < 0.0
        or coarse_end <= coarse_start
    ):
        raise ProductionWemmTemporalProjectionError(
            f"{field}.coarse_interval must satisfy 0 <= start < end"
        )
    start = _optional_finite(segment.get("start_seconds"), field=f"{field}.start_seconds")
    end = _optional_finite(segment.get("end_seconds"), field=f"{field}.end_seconds")
    if boundary_status == REFINED_BOUNDARY_STATUS:
        if start is None or end is None or start < 0.0 or end <= start:
            raise ProductionWemmTemporalProjectionError(
                f"{field} interval must satisfy 0 <= start < end"
            )
    elif start is not None or end is not None:
        raise ProductionWemmTemporalProjectionError(f"{field} pending rows must omit start/end")
    copied = _copy_json(segment, field=field)
    if not isinstance(copied, dict):  # pragma: no cover - helper invariant
        raise ProductionWemmTemporalProjectionError(f"{field} must be an object")
    return copied


def _refined_sidecar_segments(
    source: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    """Collect adaptive rows from top-level and nested sidecar aliases.

    ``refined_segments`` is the producer spelling.  Review-pack aggregates
    additionally expose ``temporal_refinement_segments`` and
    ``refined_temporal_segments``; all aliases must agree when present.
    ``temporal_refinement`` is retained as a detached snapshot for callers
    that need the request/score lineage.
    """

    candidates: list[tuple[str, Sequence[Any]]] = []
    for key in ("refined_segments", "refined_temporal_segments", "temporal_refinement_segments"):
        if key not in source:
            continue
        candidates.append((key, _sequence(source.get(key), field=key)))

    snapshot: dict[str, Any] | None = None
    nested = source.get("temporal_refinement")
    if nested is not None:
        nested_map = _mapping(nested, field="temporal_refinement")
        _assert_no_gold(nested_map, field="temporal_refinement")
        nested_copy = _copy_json(nested_map, field="temporal_refinement")
        if not isinstance(nested_copy, dict):  # pragma: no cover
            raise ProductionWemmTemporalProjectionError("temporal_refinement must be an object")
        if nested_copy.get("production_eligible") is True:
            raise ProductionWemmTemporalProjectionError(
                "temporal_refinement.production_eligible must be false"
            )
        snapshot = nested_copy
        for key in ("refined_segments", "temporal_refinement_segments", "segments"):
            if key in nested_map:
                candidates.append(
                    (
                        f"temporal_refinement.{key}",
                        _sequence(nested_map.get(key), field=f"temporal_refinement.{key}"),
                    )
                )

    if not candidates:
        return [], snapshot, None

    normalised: list[tuple[str, list[dict[str, Any]]]] = []
    for key, rows in candidates:
        normalised.append(
            (
                key,
                [
                    _validate_refined_segment(row, field=f"{key}[{index}]")
                    for index, row in enumerate(rows)
                ],
            )
        )
    first_key, first_rows = normalised[0]
    for key, rows in normalised[1:]:
        if rows != first_rows:
            raise ProductionWemmTemporalProjectionError(f"{key} does not match {first_key}")
    sidecar_format = _text(snapshot.get("format")) if snapshot is not None else None
    return first_rows, snapshot, sidecar_format


def _row_action_identity(row: Mapping[str, Any]) -> str | None:
    """Return a conservative action identity for refined/coarse matching."""

    for key in ("provisional_id", "action_key", "label_text"):
        value = _text(row.get(key))
        if value:
            return value.casefold()
    return None


def _lineage_tokens(value: object) -> set[str]:
    """Return exact and namespace-neutral tokens for one lineage value.

    Aggregation adds ``recording::temporal::`` and
    ``recording::temporal-refined::`` prefixes to otherwise identical source
    segment IDs.  Comparing the final token lets those representations match
    without treating a repeated action label as sufficient identity.  A token
    is only used after exact matching has failed and still has to be unique.
    """

    text = _text(value)
    if text is None:
        return set()
    tokens = {text.casefold()}
    parts = [part for part in text.split("::") if part]
    if len(parts) > 1:
        tokens.add(parts[-1].casefold())
    return tokens


def _row_lineage_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in (
        "source_temporal_segment_id",
        "temporal_segment_key",
        "refined_temporal_segment_key",
        "segment_id",
    ):
        tokens.update(_lineage_tokens(row.get(key)))
    return tokens


def _recordings_compatible(left: object, right: object) -> bool:
    """Allow a missing aggregate recording ID without cross-record mixing."""

    left_id = _text(left)
    right_id = _text(right)
    return left_id is None or right_id is None or left_id == right_id


def _row_interval_signature(row: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return the source interval used by a coarse/refined match fallback."""

    raw = row.get("coarse_interval")
    if isinstance(raw, Mapping):
        start = _finite(raw.get("start_seconds"), field="coarse_interval.start_seconds")
        end = _finite(raw.get("end_seconds"), field="coarse_interval.end_seconds")
    else:
        start = _finite(row.get("start_seconds"), field="start_seconds")
        end = _finite(row.get("end_seconds"), field="end_seconds")
    if start is None or end is None or start < 0.0 or end <= start:
        return None
    return start, end


def _match_refined_to_coarse(
    coarse: Sequence[Mapping[str, Any]],
    refined: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, int], tuple[int, ...]]:
    """Match measured refined rows to coarse rows without positional guessing.

    The producer normally carries the original ``segment_id`` through the
    refined row, exposed as ``source_temporal_segment_id`` after projection.
    A unique action match is accepted only when an explicit ID has no exact
    coarse match; repeated actions are disambiguated by an equal coarse
    interval.  Ambiguous IDs or unmatched rows remain in the additive refined
    sidecar and do not replace a coarse primary proposal.
    """

    matches: dict[int, int] = {}
    unmatched: list[int] = []
    for refined_index, refined_row in enumerate(refined):
        if refined_row.get("boundary_status") != REFINED_BOUNDARY_STATUS:
            continue
        refined_recording = _text(refined_row.get("recording_id"))
        refined_source_id = _text(refined_row.get("source_temporal_segment_id"))

        candidates: list[int] = []
        exact_source_ambiguous = False
        if refined_source_id:
            candidates = [
                coarse_index
                for coarse_index, coarse_row in enumerate(coarse)
                if _recordings_compatible(coarse_row.get("recording_id"), refined_recording)
                and _text(coarse_row.get("source_temporal_segment_id")) == refined_source_id
            ]
            exact_source_ambiguous = len(candidates) > 1
            if not candidates:
                refined_tokens = _row_lineage_tokens(refined_row)
                if refined_tokens:
                    candidates = [
                        coarse_index
                        for coarse_index, coarse_row in enumerate(coarse)
                        if _recordings_compatible(coarse_row.get("recording_id"), refined_recording)
                        and refined_tokens.intersection(_row_lineage_tokens(coarse_row))
                    ]
        # An explicit source ID that maps to multiple coarse rows is unsafe:
        # do not let a broad action-label fallback guess which row it meant.
        # A zero-match ID can still occur when an aggregate rewrites the
        # coarse/refined namespaces, so use the conservative action/interval
        # fallback only in that case.
        if not candidates:
            action = _row_action_identity(refined_row)
            if action is not None:
                candidates = [
                    coarse_index
                    for coarse_index, coarse_row in enumerate(coarse)
                    if _recordings_compatible(coarse_row.get("recording_id"), refined_recording)
                    and _row_action_identity(coarse_row) == action
                ]
        if len(candidates) > 1 and not exact_source_ambiguous:
            refined_interval = _row_interval_signature(refined_row)
            if refined_interval is not None:
                candidates = [
                    coarse_index
                    for coarse_index in candidates
                    if _row_interval_signature(coarse[coarse_index]) == refined_interval
                ]
        if len(candidates) == 1 and candidates[0] not in matches.values():
            coarse_row = coarse[candidates[0]]
            refined_action = _row_action_identity(refined_row)
            coarse_action = _row_action_identity(coarse_row)
            # Source lineage is the strongest match key, but a conflicting
            # explicit action identity signals a stale/misaligned sidecar.
            # Leave it review-only rather than changing the primary row.
            if (
                refined_action is not None
                and coarse_action is not None
                and refined_action != coarse_action
            ):
                unmatched.append(refined_index)
                continue
            matches[candidates[0]] = refined_index
        else:
            unmatched.append(refined_index)
    return matches, tuple(unmatched)


def _refinement_metadata_snapshots(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Copy adaptive plan/result sidecars while enforcing review-only flags."""

    snapshots: dict[str, dict[str, Any]] = {}
    for key in (
        "temporal_refinement_plan",
        "temporal_refinement_fine_plan",
        "temporal_refinement_score_resolution",
    ):
        if key not in source:
            continue
        value = _mapping(source.get(key), field=key)
        _assert_no_gold(value, field=key)
        copied = _copy_json(value, field=key)
        if not isinstance(copied, dict):  # pragma: no cover - helper invariant
            raise ProductionWemmTemporalProjectionError(f"{key} must be an object")
        if copied.get("production_eligible") is True:
            raise ProductionWemmTemporalProjectionError(f"{key}.production_eligible must be false")
        snapshots[key] = copied
    return snapshots


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
        previous = copied.get("recording_id")
        if previous is not None and previous != recording_id:
            # The segment/source recording is the canonical lineage for a
            # direct projection call.  Keep a conflicting upstream value for
            # review instead of allowing a stale nested ID to disagree with
            # the top-level segment identity.
            existing_upstream = copied.get("upstream_lineage")
            upstream = dict(existing_upstream) if isinstance(existing_upstream, Mapping) else {}
            if existing_upstream is not None and not isinstance(existing_upstream, Mapping):
                upstream.setdefault(
                    "value", _copy_json(existing_upstream, field="source_ref.upstream_lineage")
                )
            upstream["recording_id"] = _copy_json(previous, field="source_ref.recording_id")
            copied["upstream_lineage"] = upstream
        copied["recording_id"] = recording_id
    return {key: value for key, value in copied.items()}


def _project_segment(
    segment: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    index: int,
    sidecar_format: str | None,
    boundary_status: str = BOUNDARY_STATUS,
    default_boundary_source: str = "wemm_temporal_score",
    id_namespace: str = "temporal",
    allow_unresolved: bool = False,
) -> dict[str, Any]:
    recording_id = _text(segment.get("recording_id")) or _recording_id(source)
    source_id = (
        _text(segment.get("temporal_segment_key"))
        or _text(segment.get("segment_id"))
        or _text(segment.get("proposal_id"))
        or f"temporal-{index:04d}"
    )
    namespace_token = id_namespace.strip(":") or "temporal"
    if f"::{namespace_token}::" in source_id:
        segment_id = source_id
    elif recording_id is None:
        segment_id = (
            source_id if namespace_token == "temporal" else f"{namespace_token}::{source_id}"
        )
    else:
        segment_id = f"{recording_id}::{namespace_token}::{source_id}"
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
    timestamp_basis = _text(segment.get("timestamp_basis")) or (
        "source_relative_seconds"
        if boundary_status in REFINED_BOUNDARY_STATUSES
        else "MODEL_PROBE_BOUNDARY"
    )
    raw_start = _finite(segment.get("start_seconds"), field=f"{segment_id}.start_seconds")
    raw_end = _finite(segment.get("end_seconds"), field=f"{segment_id}.end_seconds")
    if allow_unresolved:
        if boundary_status == REFINED_BOUNDARY_STATUS:
            if raw_start is None or raw_end is None or raw_start < 0.0 or raw_end <= raw_start:
                raise ProductionWemmTemporalProjectionError(
                    f"{segment_id} interval must satisfy 0 <= start < end"
                )
        elif raw_start is not None or raw_end is not None:
            raise ProductionWemmTemporalProjectionError(
                f"{segment_id} pending rows must omit start/end"
            )
    elif raw_start is None or raw_end is None or raw_start < 0.0 or raw_end <= raw_start:
        raise ProductionWemmTemporalProjectionError(
            f"{segment_id} interval must satisfy 0 <= start < end"
        )
    coarse_interval = _copy_json(
        segment.get("coarse_interval"), field=f"{segment_id}.coarse_interval"
    )
    refined_interval = _copy_json(
        segment.get("refined_interval"), field=f"{segment_id}.refined_interval"
    )
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
        "start_seconds": raw_start,
        "end_seconds": raw_end,
        "start_time_sec": raw_start,
        "end_time_sec": raw_end,
        "boundary_status": boundary_status,
        "boundary_source": _text(segment.get("boundary_source")) or default_boundary_source,
        "boundary_method": boundary_method,
        "boundary_confidence": _finite(
            segment.get("boundary_confidence"), field=f"{segment_id}.boundary_confidence"
        ),
        "timestamp_basis": timestamp_basis,
        "coarse_interval": coarse_interval,
        "refined_interval": refined_interval,
        "refinement_status": _text(segment.get("refinement_status")),
        "onset_request_id": _text(segment.get("onset_request_id")),
        "offset_request_id": _text(segment.get("offset_request_id")),
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
    refinement_metadata = _refinement_metadata_snapshots(payload)
    refined_segments_raw, temporal_refinement_snapshot, refined_sidecar_format = (
        _refined_sidecar_segments(payload)
    )
    coarse_proposals = [
        _project_segment(segment, source=payload, index=index, sidecar_format=sidecar_format)
        for index, segment in enumerate(segments)
    ]
    refined_segments = [
        _project_segment(
            segment,
            source=payload,
            index=index,
            sidecar_format=refined_sidecar_format,
            boundary_status=str(segment.get("boundary_status")),
            default_boundary_source="wemm_short_refinement",
            id_namespace="temporal-refined",
            allow_unresolved=True,
        )
        for index, segment in enumerate(refined_segments_raw)
    ]
    refined_interval_proposals = [
        proposal
        for proposal in refined_segments
        if proposal.get("boundary_status") == REFINED_BOUNDARY_STATUS
    ]

    # A measured refined row is the preferred primary proposal only when it can
    # be matched to one and only one coarse row.  This keeps the historical
    # coarse proposal as a deterministic fallback for pending, ambiguous, or
    # otherwise unmatched refinement output.  The complete coarse list remains
    # available as a sidecar for audit/review and is never silently replaced.
    refined_matches, unmatched_refined_indices = _match_refined_to_coarse(
        coarse_proposals, refined_segments
    )
    primary_proposals: list[dict[str, Any]] = []
    primary_selection: list[dict[str, Any]] = []
    for coarse_index, coarse_proposal in enumerate(coarse_proposals):
        refined_index = refined_matches.get(coarse_index)
        if refined_index is None:
            primary_proposals.append(coarse_proposal)
            primary_selection.append(
                {
                    "coarse_index": coarse_index,
                    "coarse_segment_id": coarse_proposal["segment_id"],
                    "primary_segment_id": coarse_proposal["segment_id"],
                    "selection": "COARSE_FALLBACK",
                    "refined_index": None,
                    "refined_segment_id": None,
                }
            )
            continue
        refined_proposal = refined_segments[refined_index]
        # Keep the primary surface detached from the refined sidecar.  Review
        # edits to one surface must not mutate the provenance copy on the
        # other surface (and vice versa).
        primary_proposals.append(
            _copy_json(refined_proposal, field="temporal_interval_proposals.refined")
        )
        primary_selection.append(
            {
                "coarse_index": coarse_index,
                "coarse_segment_id": coarse_proposal["segment_id"],
                "primary_segment_id": refined_proposal["segment_id"],
                "selection": "MODEL_REFINED",
                "refined_index": refined_index,
                "refined_segment_id": refined_proposal["segment_id"],
            }
        )

    coarse_ids = {str(proposal["segment_id"]) for proposal in coarse_proposals}
    refined_ids = {str(proposal["segment_id"]) for proposal in refined_segments}
    if coarse_ids.intersection(refined_ids):
        raise ProductionWemmTemporalProjectionError(
            "coarse and refined temporal proposal IDs must use separate namespaces"
        )
    seen: set[str] = set()
    for proposal in primary_proposals:
        proposal_id = str(proposal["segment_id"])
        if proposal_id in seen:
            raise ProductionWemmTemporalProjectionError(
                f"duplicate temporal proposal id: {proposal_id}"
            )
        seen.add(proposal_id)
    return {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": STATUS if (primary_proposals or refined_segments) else "EMPTY",
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "recording_id": _recording_id(payload),
        "source_format": source_format,
        "sidecar_format": sidecar_format,
        # Primary consumers receive a measured refined boundary when it was
        # matched safely; otherwise they receive the original coarse proposal.
        "temporal_interval_proposals": primary_proposals,
        # Keep the complete coarse projection independently available.  This
        # is intentionally not an alias of the primary list because a primary
        # row may now come from the refined namespace.
        "coarse_temporal_interval_proposals": _copy_json(
            coarse_proposals, field="coarse_temporal_interval_proposals"
        ),
        "temporal_interval_primary_selection": primary_selection,
        # Adaptive rows live in a separate namespace.  Pending rows remain
        # visible for review, while only measured rows are exposed as interval
        # proposals suitable for an interval-oriented consumer.
        "temporal_refinement_segments": refined_segments,
        "refined_temporal_segments": _copy_json(
            refined_segments, field="refined_temporal_segments"
        ),
        "refined_temporal_interval_proposals": refined_interval_proposals,
        "temporal_refinement": temporal_refinement_snapshot,
        **refinement_metadata,
        "review_contract": {
            "decision_options": list(DECISION_OPTIONS),
            "temporal_interval_fields": list(TEMPORAL_INTERVAL_FIELDS),
            "boundary_status": BOUNDARY_STATUS,
            "primary_boundary_statuses": [BOUNDARY_STATUS, REFINED_BOUNDARY_STATUS],
            "temporal_interval_primary_policy": ("matched_model_refined_else_coarse_fallback"),
            "coarse_temporal_interval_sidecar": True,
            "proposals_only": True,
            "human_review_required": True,
            "review_required": True,
            "automatic_eligible": False,
            "window_context_only": True,
            "windows_are_not_action_boundaries": True,
            "top_k_preserved_verbatim": True,
            "refined_temporal_interval_fields": list(REFINED_TEMPORAL_INTERVAL_FIELDS),
            "temporal_refinement_review_only": True,
            "refined_segments_review_only": True,
            "pending_refined_rows_visible": True,
        },
        "metrics": {
            "temporal_interval_proposal_count": len(primary_proposals),
            "review_required_count": len(primary_proposals),
            "automatic_eligible_count": 0,
            # Only a refined row that safely replaces a unique coarse row has
            # a measured boundary on the primary surface.  An orphan or
            # ambiguous refined row remains visible in the sidecar, but must
            # not inflate this metric and imply that it is usable by the
            # interval consumer.
            "measured_interval_count": len(refined_matches),
            "coarse_temporal_interval_proposal_count": len(coarse_proposals),
            "temporal_interval_primary_refined_count": len(refined_matches),
            "temporal_interval_coarse_fallback_count": len(coarse_proposals) - len(refined_matches),
            "temporal_interval_unmatched_refined_count": len(unmatched_refined_indices),
            "temporal_refinement_segment_count": len(refined_segments),
            "refined_temporal_interval_proposal_count": len(refined_interval_proposals),
            "refined_temporal_pending_count": len(refined_segments)
            - len(refined_interval_proposals),
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
                or refined_segments
                or temporal_refinement_snapshot is not None
            ),
            "temporal_proposals_projected": bool(primary_proposals or refined_segments),
            "temporal_refinement_projected": bool(refined_segments),
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
            "Adaptive refined rows are additive review evidence; pending rows are not "
            "measurable intervals.",
            "The primary temporal list uses matched measured refinement and falls back "
            "to the coarse proposal when matching is unavailable.",
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
    copied["coarse_temporal_interval_proposals"] = _copy_json(
        projection.get("coarse_temporal_interval_proposals", []),
        field="projection.coarse_temporal_interval_proposals",
    )
    copied["temporal_interval_primary_selection"] = _copy_json(
        projection.get("temporal_interval_primary_selection", []),
        field="projection.temporal_interval_primary_selection",
    )
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
    metrics["coarse_temporal_interval_proposal_count"] = len(
        copied["coarse_temporal_interval_proposals"]
    )
    metrics["temporal_interval_primary_refined_count"] = projection["metrics"].get(
        "temporal_interval_primary_refined_count", 0
    )
    metrics["measured_interval_count"] = projection["metrics"].get("measured_interval_count", 0)
    metrics["temporal_interval_coarse_fallback_count"] = projection["metrics"].get(
        "temporal_interval_coarse_fallback_count", 0
    )
    metrics["temporal_interval_unmatched_refined_count"] = projection["metrics"].get(
        "temporal_interval_unmatched_refined_count", 0
    )
    refined_segments = _copy_json(
        projection.get("temporal_refinement_segments", []),
        field="projection.temporal_refinement_segments",
    )
    refined_interval_proposals = _copy_json(
        projection.get("refined_temporal_interval_proposals", []),
        field="projection.refined_temporal_interval_proposals",
    )
    copied["temporal_refinement_segments"] = refined_segments
    copied["refined_temporal_segments"] = _copy_json(
        refined_segments, field="draft.refined_temporal_segments"
    )
    copied["refined_temporal_interval_proposals"] = refined_interval_proposals
    if projection.get("temporal_refinement") is not None:
        copied["temporal_refinement"] = _copy_json(
            projection.get("temporal_refinement"), field="projection.temporal_refinement"
        )
    for key in (
        "temporal_refinement_plan",
        "temporal_refinement_fine_plan",
        "temporal_refinement_score_resolution",
    ):
        if key in projection:
            copied[key] = _copy_json(projection[key], field=f"projection.{key}")
    metrics["temporal_refinement_segment_count"] = len(refined_segments)
    metrics["refined_temporal_interval_proposal_count"] = len(refined_interval_proposals)
    metrics["refined_temporal_pending_count"] = len(refined_segments) - len(
        refined_interval_proposals
    )
    contract = copied.get("review_contract")
    if not isinstance(contract, dict):
        contract = {}
        copied["review_contract"] = contract
    contract["temporal_interval_fields"] = list(TEMPORAL_INTERVAL_FIELDS)
    contract["temporal_interval_proposals_separate"] = True
    contract["temporal_boundaries_are_model_proposals"] = True
    contract["temporal_proposals_require_human_review"] = True
    contract["primary_boundary_statuses"] = [BOUNDARY_STATUS, REFINED_BOUNDARY_STATUS]
    contract["temporal_interval_primary_policy"] = "matched_model_refined_else_coarse_fallback"
    contract["coarse_temporal_interval_sidecar"] = True
    contract["refined_temporal_interval_fields"] = list(REFINED_TEMPORAL_INTERVAL_FIELDS)
    contract["temporal_refinement_review_only"] = True
    contract["refined_segments_review_only"] = True
    contract["pending_refined_rows_visible"] = True
    controls = copied.get("controls")
    if not isinstance(controls, dict):
        controls = {}
        copied["controls"] = controls
    controls["temporal_sidecar_read"] = projection["controls"]["temporal_sidecar_read"]
    controls["temporal_proposals_projected"] = bool(proposals or refined_segments)
    controls["temporal_refinement_projected"] = bool(refined_segments)
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
        boundary_status = _text(proposal.get("boundary_status"))
        if boundary_status not in {BOUNDARY_STATUS, REFINED_BOUNDARY_STATUS}:
            raise ProductionWemmTemporalProjectionError(
                f"{proposal_id}.boundary_status must be one of "
                f"{BOUNDARY_STATUS}, {REFINED_BOUNDARY_STATUS}"
            )
        start = _finite(proposal.get("start_seconds"), field=f"{proposal_id}.start_seconds")
        end = _finite(proposal.get("end_seconds"), field=f"{proposal_id}.end_seconds")
        if start is None or end is None or start < 0.0 or end <= start:
            raise ProductionWemmTemporalProjectionError(
                f"{proposal_id} interval must satisfy 0 <= start < end"
            )
        if boundary_status == REFINED_BOUNDARY_STATUS:
            coarse = _mapping(
                proposal.get("coarse_interval"), field=f"{proposal_id}.coarse_interval"
            )
            coarse_start = _finite(
                coarse.get("start_seconds"), field=f"{proposal_id}.coarse_interval.start_seconds"
            )
            coarse_end = _finite(
                coarse.get("end_seconds"), field=f"{proposal_id}.coarse_interval.end_seconds"
            )
            if (
                coarse_start is None
                or coarse_end is None
                or coarse_start < 0.0
                or coarse_end <= coarse_start
            ):
                raise ProductionWemmTemporalProjectionError(
                    f"{proposal_id}.coarse_interval must satisfy 0 <= start < end"
                )
        top_k = _sequence(proposal.get("top_k", []), field=f"{proposal_id}.top_k")
        if not isinstance(top_k, Sequence) or isinstance(top_k, (str, bytes, bytearray)):
            raise ProductionWemmTemporalProjectionError(f"{proposal_id}.top_k must be an array")
        decision = _text(proposal.get("decision"))
        if decision not in {"pending", *DECISION_OPTIONS}:
            raise ProductionWemmTemporalProjectionError(f"{proposal_id}.decision is invalid")
        _assert_no_gold(proposal, field=proposal_id)

    def _projected_coarse_rows(rows_value: object, *, field: str) -> list[Mapping[str, Any]]:
        rows = _sequence(rows_value or [], field=field)
        result: list[Mapping[str, Any]] = []
        local_ids: set[str] = set()
        for index, raw in enumerate(rows):
            row = _mapping(raw, field=f"{field}[{index}]")
            row_id = _text(row.get("segment_id"))
            if row_id is None or row_id in local_ids:
                raise ProductionWemmTemporalProjectionError(
                    f"{field}[{index}].segment_id must be unique"
                )
            local_ids.add(row_id)
            if row.get("review_required") is not True:
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id}.review_required must be true"
                )
            if row.get("automatic_eligible") is not False:
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id}.automatic_eligible must be false"
                )
            if row.get("boundary_status") != BOUNDARY_STATUS:
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id}.boundary_status must be {BOUNDARY_STATUS}"
                )
            start = _finite(row.get("start_seconds"), field=f"{row_id}.start_seconds")
            end = _finite(row.get("end_seconds"), field=f"{row_id}.end_seconds")
            if start is None or end is None or start < 0.0 or end <= start:
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id} interval must satisfy 0 <= start < end"
                )
            decision = _text(row.get("decision"))
            if decision not in {"pending", *DECISION_OPTIONS}:
                raise ProductionWemmTemporalProjectionError(f"{row_id}.decision is invalid")
            _assert_no_gold(row, field=row_id)
            result.append(row)
        return result

    def _projected_refined_rows(
        rows_value: object, *, field: str, measured_only: bool = False
    ) -> list[Mapping[str, Any]]:
        rows = _sequence(rows_value or [], field=field)
        result: list[Mapping[str, Any]] = []
        local_ids: set[str] = set()
        for index, raw in enumerate(rows):
            row = _mapping(raw, field=f"{field}[{index}]")
            row_id = _text(row.get("segment_id"))
            if row_id is None or row_id in local_ids:
                raise ProductionWemmTemporalProjectionError(
                    f"{field}[{index}].segment_id must be unique"
                )
            local_ids.add(row_id)
            if row.get("review_required") is not True:
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id}.review_required must be true"
                )
            if row.get("automatic_eligible") is not False:
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id}.automatic_eligible must be false"
                )
            boundary_status = _text(row.get("boundary_status"))
            if boundary_status not in REFINED_BOUNDARY_STATUSES:
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id}.boundary_status must be one of "
                    f"{', '.join(REFINED_BOUNDARY_STATUSES)}"
                )
            start = _optional_finite(row.get("start_seconds"), field=f"{row_id}.start_seconds")
            end = _optional_finite(row.get("end_seconds"), field=f"{row_id}.end_seconds")
            if boundary_status == REFINED_BOUNDARY_STATUS:
                if start is None or end is None or start < 0.0 or end <= start:
                    raise ProductionWemmTemporalProjectionError(
                        f"{row_id} interval must satisfy 0 <= start < end"
                    )
            elif start is not None or end is not None:
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id} pending rows must omit start/end"
                )
            if measured_only and boundary_status != REFINED_BOUNDARY_STATUS:
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id} measured refined proposal must use {REFINED_BOUNDARY_STATUS}"
                )
            coarse = _mapping(row.get("coarse_interval"), field=f"{row_id}.coarse_interval")
            coarse_start = _finite(
                coarse.get("start_seconds"), field=f"{row_id}.coarse_interval.start_seconds"
            )
            coarse_end = _finite(
                coarse.get("end_seconds"), field=f"{row_id}.coarse_interval.end_seconds"
            )
            if (
                coarse_start is None
                or coarse_end is None
                or coarse_start < 0.0
                or coarse_end <= coarse_start
            ):
                raise ProductionWemmTemporalProjectionError(
                    f"{row_id}.coarse_interval must satisfy 0 <= start < end"
                )
            _assert_no_gold(row, field=row_id)
            result.append(row)
        return result

    refined_value = payload.get("temporal_refinement_segments")
    if refined_value is None and target is not payload:
        refined_value = target.get("temporal_refinement_segments")
    refined_alias_value = payload.get("refined_temporal_segments")
    if refined_alias_value is None and target is not payload:
        refined_alias_value = target.get("refined_temporal_segments")
    refined_proposals_value = payload.get("refined_temporal_interval_proposals")
    if refined_proposals_value is None and target is not payload:
        refined_proposals_value = target.get("refined_temporal_interval_proposals")
    refined_rows = _projected_refined_rows(
        refined_value or [], field="temporal_refinement_segments"
    )
    if refined_alias_value is not None:
        alias_rows = _projected_refined_rows(refined_alias_value, field="refined_temporal_segments")
        if [dict(row) for row in alias_rows] != [dict(row) for row in refined_rows]:
            raise ProductionWemmTemporalProjectionError(
                "refined_temporal_segments does not match temporal_refinement_segments"
            )
    refined_proposals = _projected_refined_rows(
        refined_proposals_value or [],
        field="refined_temporal_interval_proposals",
        measured_only=True,
    )

    coarse_value = payload.get("coarse_temporal_interval_proposals")
    if coarse_value is None and target is not payload:
        coarse_value = target.get("coarse_temporal_interval_proposals")
    if coarse_value is None:
        # Reports produced before the primary-selection field existed are
        # still valid.  Their primary list was entirely coarse, so derive the
        # compatibility sidecar from those rows.  A refined primary row is
        # deliberately not treated as coarse here.
        coarse_rows = [row for row in proposals if row.get("boundary_status") == BOUNDARY_STATUS]
    else:
        coarse_rows = _projected_coarse_rows(
            coarse_value, field="coarse_temporal_interval_proposals"
        )

    coarse_sidecar_ids = {str(row.get("segment_id")) for row in coarse_rows}
    refined_ids_for_collision = {str(row.get("segment_id")) for row in refined_rows}
    if coarse_sidecar_ids.intersection(refined_ids_for_collision):
        raise ProductionWemmTemporalProjectionError(
            "coarse and refined temporal proposal IDs must use separate namespaces"
        )
    primary_refined_ids = {
        str(row.get("segment_id"))
        for row in proposals
        if row.get("boundary_status") == REFINED_BOUNDARY_STATUS
    }
    if not primary_refined_ids.issubset({str(row.get("segment_id")) for row in refined_rows}):
        raise ProductionWemmTemporalProjectionError(
            "primary refined temporal proposals must be drawn from temporal_refinement_segments"
        )
    primary_coarse_ids = {
        str(row.get("segment_id"))
        for row in proposals
        if row.get("boundary_status") == BOUNDARY_STATUS
    }
    if not primary_coarse_ids.issubset(coarse_sidecar_ids):
        raise ProductionWemmTemporalProjectionError(
            "primary coarse temporal proposals must be drawn from "
            "coarse_temporal_interval_proposals"
        )
    if refined_proposals:
        if not refined_rows:
            raise ProductionWemmTemporalProjectionError(
                "refined_temporal_interval_proposals must be drawn from "
                "temporal_refinement_segments"
            )
        refined_ids = {str(row.get("segment_id")) for row in refined_rows}
        if not {str(row.get("segment_id")) for row in refined_proposals}.issubset(refined_ids):
            raise ProductionWemmTemporalProjectionError(
                "refined_temporal_interval_proposals must be drawn from "
                "temporal_refinement_segments"
            )
    refinement_snapshot = payload.get("temporal_refinement")
    if refinement_snapshot is None and target is not payload:
        refinement_snapshot = target.get("temporal_refinement")
    if refinement_snapshot is not None:
        snapshot = _mapping(refinement_snapshot, field="temporal_refinement")
        if snapshot.get("production_eligible") is True:
            raise ProductionWemmTemporalProjectionError(
                "temporal_refinement.production_eligible must be false"
            )
        _assert_no_gold(snapshot, field="temporal_refinement")
    for key in (
        "temporal_refinement_plan",
        "temporal_refinement_fine_plan",
        "temporal_refinement_score_resolution",
    ):
        metadata_value = payload.get(key)
        if metadata_value is None and target is not payload:
            metadata_value = target.get(key)
        if metadata_value is None:
            continue
        metadata = _mapping(metadata_value, field=key)
        if metadata.get("production_eligible") is True:
            raise ProductionWemmTemporalProjectionError(f"{key}.production_eligible must be false")
        _assert_no_gold(metadata, field=key)
    detached = _copy_json(payload, field="projection")
    if not isinstance(detached, dict):  # pragma: no cover - helper invariant
        raise ProductionWemmTemporalProjectionError("projection must be an object")
    return detached


__all__ = [
    "AUTHORITY",
    "BOUNDARY_STATUS",
    "DECISION_OPTIONS",
    "FORMAT",
    "REFINED_BOUNDARY_STATUS",
    "REFINED_BOUNDARY_STATUSES",
    "REFINED_PENDING_STATUS",
    "REFINED_TEMPORAL_INTERVAL_FIELDS",
    "STATUS",
    "TEMPORAL_INTERVAL_FIELDS",
    "ProductionWemmTemporalProjectionError",
    "attach_temporal_interval_proposals",
    "project_temporal_interval_proposals",
    "validate_temporal_interval_projection",
]
