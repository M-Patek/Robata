"""Build a blank, source-bound production gold-collection template.

The production-shaped cohort has source windows but no independently
adjudicated action labels.  This module creates the smallest useful collection
queue for obtaining those labels without copying a model/Terra suggestion into
the gold record.  It is a local benchmark helper, not a published schema and
never reads media, invokes a model, edits ontology/Mapper, or computes an
identity/hash/digest.

``PENDING`` is the only lifecycle state emitted by the builder.  The later
``REVIEWED`` -> ``ADJUDICATED`` -> ``OFFICIAL_GOLD`` states are documented in
the contract so a reviewer can record the transition explicitly rather than
implicitly promoting a surrogate review.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

SOURCE_BOUND_GOLD_COLLECTION_VERSION: Final = "robata-production-source-bound-gold-collection-v1"
# Keep this input boundary aligned with the production-shaped cohort builder and
# the other production sidecars. In particular, an EPIC benchmark manifest must
# never be accepted as the source for a production gold queue.
PRODUCTION_COHORT_MANIFEST_FORMAT: Final = "robata-production-shaped-cohort-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
ANNOTATION_PRINCIPAL_REFERENCE: Final = "data/source/annotation-principal.txt"
DECISION_OPTIONS: Final = ("accept", "edit", "split", "reject", "abstain")
LIFECYCLE_STATES: Final = ("PENDING", "REVIEWED", "ADJUDICATED", "OFFICIAL_GOLD")
REQUIRED_SEGMENT_FIELDS: Final = (
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
STRUCTURED_LABEL_FIELDS: Final = ("verb", "noun", "attributes", "location", "hand")
DEFAULT_REVIEWER_SLOTS: Final = ("reviewer_a", "reviewer_b")


class ProductionGoldCollectionError(ValueError):
    """Raised when a source cohort cannot be converted to a blank queue."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionGoldCollectionError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionGoldCollectionError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionGoldCollectionError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    """Validate an optional textual reference without inventing a value."""

    if value is None:
        return None
    return _text(value, field=field)


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionGoldCollectionError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProductionGoldCollectionError(f"{field} must be a finite number")
    return result


def _copy_json(value: object, *, field: str) -> Any:
    """Copy JSON-shaped metadata without resolving paths or calculating IDs."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionGoldCollectionError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        # Do not silently coerce keys.  Coercion can collapse distinct input
        # keys (for example ``1`` and ``"1"``) and would make a copied source
        # record differ from the manifest it claims to bind.
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionGoldCollectionError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionGoldCollectionError(f"{field} must be JSON-compatible")


def _reviewer_slots(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is not None and (
        not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray))
    ):
        raise ProductionGoldCollectionError("reviewer_slots must be an array")
    raw = DEFAULT_REVIEWER_SLOTS if value is None else tuple(value)
    if not raw:
        raise ProductionGoldCollectionError("reviewer_slots must not be empty")
    slots: list[str] = []
    for index, slot in enumerate(raw):
        text = _text(slot, field=f"reviewer_slots[{index}]")
        if text in slots:
            raise ProductionGoldCollectionError(f"duplicate reviewer slot: {text}")
        slots.append(text)
    return tuple(slots)


def _source_camera_ids(source: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Validate optional source camera metadata and return its IDs.

    Older hand-authored fixtures may omit ``source.cameras`` entirely, so the
    field remains optional. When it is present, malformed entries (for example
    a missing ID becoming the string ``"None"``) would weaken source binding
    for every generated row and are rejected.
    """

    raw_cameras = source.get("cameras")
    if raw_cameras is None:
        camera_count = source.get("camera_count")
        if camera_count is not None and (
            isinstance(camera_count, bool) or not isinstance(camera_count, int) or camera_count < 0
        ):
            raise ProductionGoldCollectionError(
                "manifest.source.camera_count must be a non-negative integer"
            )
        return None
    cameras = _sequence(raw_cameras, field="manifest.source.cameras")
    if not cameras:
        raise ProductionGoldCollectionError("manifest.source.cameras must not be empty")
    ids: list[str] = []
    for index, raw_camera in enumerate(cameras):
        camera = _mapping(raw_camera, field=f"manifest.source.cameras[{index}]")
        camera_id = _text(
            camera.get("camera_id"), field=f"manifest.source.cameras[{index}].camera_id"
        )
        if camera_id in ids:
            raise ProductionGoldCollectionError(
                f"manifest.source.cameras contains duplicate {camera_id}"
            )
        ids.append(camera_id)
    camera_count = source.get("camera_count")
    if camera_count is not None and (
        isinstance(camera_count, bool) or not isinstance(camera_count, int) or camera_count < 0
    ):
        raise ProductionGoldCollectionError(
            "manifest.source.camera_count must be a non-negative integer"
        )
    if camera_count is not None and camera_count != len(ids):
        raise ProductionGoldCollectionError(
            "manifest.source.camera_count does not match manifest.source.cameras"
        )
    return tuple(ids)


def _source_camera_topics(
    source: Mapping[str, Any],
    camera_ids: Sequence[str] | None,
) -> dict[str, str] | None:
    """Return an optional, validated source camera-topic projection.

    The first production cohort manifests carry topic metadata on each camera
    row, while older hand-authored fixtures carry IDs only.  Preserve that
    compatibility, but reject *partial* or conflicting topic metadata instead
    of allowing one camera to silently bind to a different stream.
    """

    raw_cameras = source.get("cameras")
    if raw_cameras is None or camera_ids is None:
        return None
    cameras = _sequence(raw_cameras, field="manifest.source.cameras")
    if len(cameras) != len(camera_ids):
        # ``_source_camera_ids`` normally catches this through camera_count,
        # but keep the helper safe when called independently.
        raise ProductionGoldCollectionError(
            "manifest.source.cameras does not bind source camera IDs"
        )

    topics: dict[str, str] = {}
    present = False
    missing: list[str] = []
    for index, (raw_camera, camera_id) in enumerate(zip(cameras, camera_ids, strict=True)):
        camera = _mapping(raw_camera, field=f"manifest.source.cameras[{index}]")
        has_topic = "topic" in camera
        has_alias = "camera_topic" in camera
        if not has_topic and not has_alias:
            missing.append(camera_id)
            continue
        present = True
        if has_topic and has_alias:
            topic = _text(
                camera.get("topic"),
                field=f"manifest.source.cameras[{index}].topic",
            )
            alias = _text(
                camera.get("camera_topic"),
                field=f"manifest.source.cameras[{index}].camera_topic",
            )
            if topic != alias:
                raise ProductionGoldCollectionError(
                    f"manifest.source.cameras[{index}] topic aliases disagree"
                )
        else:
            key = "topic" if has_topic else "camera_topic"
            topic = _text(
                camera.get(key),
                field=f"manifest.source.cameras[{index}].{key}",
            )
        topics[camera_id] = topic

    if not present:
        return None
    if missing:
        raise ProductionGoldCollectionError(
            "manifest.source.cameras topic metadata is incomplete: " + ", ".join(missing)
        )
    if len(set(topics.values())) != len(topics):
        raise ProductionGoldCollectionError("manifest.source camera topics must be unique")
    return {camera_id: topics[camera_id] for camera_id in camera_ids}


def _camera_topics(
    value: object,
    *,
    field: str,
    camera_ids: Sequence[str],
) -> dict[str, str]:
    """Validate and normalize a complete per-window camera-topic map."""

    topics_raw = _mapping(value, field=field)
    topics: dict[str, str] = {}
    for raw_camera_id, raw_topic in topics_raw.items():
        if not isinstance(raw_camera_id, str):
            raise ProductionGoldCollectionError(f"{field} keys must be strings")
        camera_id = _text(raw_camera_id, field=f"{field}.camera_id")
        if camera_id in topics:
            raise ProductionGoldCollectionError(f"{field} contains duplicate camera ID {camera_id}")
        if camera_id not in camera_ids:
            raise ProductionGoldCollectionError(
                f"{field} contains camera not bound to window: {camera_id}"
            )
        topics[camera_id] = _text(raw_topic, field=f"{field}.{camera_id}")
    expected = tuple(camera_ids)
    if set(topics) != set(expected):
        raise ProductionGoldCollectionError(f"{field} must contain exactly the window camera IDs")
    if len(set(topics.values())) != len(topics):
        raise ProductionGoldCollectionError(f"{field} topics must be unique")
    # Mapping insertion order is not a source identity.  Emit a stable order
    # matching camera_ids so downstream consumers do not infer a new ordering.
    return {camera_id: topics[camera_id] for camera_id in expected}


def _window_source(
    window: Mapping[str, Any],
    *,
    source_path: str,
    index: int,
    expected_camera_ids: Sequence[str] | None = None,
    expected_camera_count: int | None = None,
    expected_camera_topics: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    start = _number(window.get("start_seconds"), field=f"windows[{index}].start_seconds")
    end = _number(window.get("end_seconds"), field=f"windows[{index}].end_seconds")
    if start < 0 or end <= start:
        raise ProductionGoldCollectionError(
            f"windows[{index}] must have 0 <= start_seconds < end_seconds"
        )
    camera_ids = _sequence(window.get("camera_ids"), field=f"windows[{index}].camera_ids")
    cameras: list[str] = []
    for cindex, camera_id in enumerate(camera_ids):
        value = _text(camera_id, field=f"windows[{index}].camera_ids[{cindex}]")
        if value in cameras:
            raise ProductionGoldCollectionError(
                f"windows[{index}].camera_ids contains duplicate {value}"
            )
        cameras.append(value)
    if not cameras:
        raise ProductionGoldCollectionError(f"windows[{index}].camera_ids must not be empty")
    if expected_camera_ids is not None and tuple(cameras) != tuple(expected_camera_ids):
        raise ProductionGoldCollectionError(
            f"windows[{index}].camera_ids must bind manifest.source.cameras"
        )
    if expected_camera_count is not None and len(cameras) != expected_camera_count:
        raise ProductionGoldCollectionError(
            f"windows[{index}].camera_ids must contain {expected_camera_count} cameras"
        )
    if "camera_topics" in window:
        camera_topics = _camera_topics(
            window["camera_topics"],
            field=f"windows[{index}].camera_topics",
            camera_ids=cameras,
        )
    elif expected_camera_topics is not None:
        camera_topics = {camera_id: expected_camera_topics[camera_id] for camera_id in cameras}
    else:
        camera_topics = {}
    if expected_camera_topics is not None and camera_topics != dict(expected_camera_topics):
        raise ProductionGoldCollectionError(
            f"windows[{index}].camera_topics must bind manifest source camera topics"
        )
    return {
        "path": source_path,
        "interval": [start, end],
        "camera_ids": cameras,
        "camera_topics": camera_topics,
        # Fixed cohort windows are context only; reviewers must enter action
        # boundaries in the segment rows below.
        "fixed_window": True,
        "action_boundary_status": "NOT_ESTABLISHED",
    }


def build_source_bound_gold_collection(
    manifest: Mapping[str, Any],
    *,
    reviewer_slots: Sequence[str] | None = None,
    manifest_reference: str | None = None,
    evidence_reference: str | None = None,
) -> dict[str, Any]:
    """Build a label-empty source-bound gold collection queue.

    ``manifest`` is expected to be a ``robata-production-shaped-cohort-v1``
    mapping.  The function copies only source/window metadata.  It deliberately
    does not copy ``review.segments``, model outputs, Terra labels, or any
    provisional vocabulary from the input.
    """

    payload = _mapping(manifest, field="manifest")
    manifest_format = _text(payload.get("format"), field="manifest.format")
    if manifest_format != PRODUCTION_COHORT_MANIFEST_FORMAT:
        raise ProductionGoldCollectionError(
            "manifest.format must be robata-production-shaped-cohort-v1"
        )
    # ``authority`` was added to the cohort manifest after the first local
    # fixtures. Treat an omitted value as the historical local default for
    # compatibility, but reject any explicit cross-namespace authority.
    authority = payload.get("authority", AUTHORITY)
    if authority != AUTHORITY:
        raise ProductionGoldCollectionError("manifest authority must be LOCAL_NONPRODUCTION_ONLY")
    source = _mapping(payload.get("source"), field="manifest.source")
    source_path = _text(source.get("path"), field="manifest.source.path")
    source_camera_ids = _source_camera_ids(source)
    source_camera_topics = _source_camera_topics(source, source_camera_ids)
    source_camera_count = source.get("camera_count")
    if source_camera_count is not None:
        # ``_source_camera_ids`` validates the type/range.  When the source
        # inventory is omitted, retain the declared count as a lightweight
        # binding check for every window rather than accepting arbitrary camera
        # coverage.
        source_camera_count = int(source_camera_count)
    media_type = _optional_text(source.get("media_type"), field="manifest.source.media_type")
    normalized_manifest_reference = _optional_text(manifest_reference, field="manifest_reference")
    normalized_evidence_reference = _optional_text(evidence_reference, field="evidence_reference")
    raw_windows = _sequence(payload.get("windows"), field="manifest.windows")
    if not raw_windows:
        raise ProductionGoldCollectionError("manifest.windows must not be empty")
    slots = _reviewer_slots(reviewer_slots)

    windows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_ordinals: set[int] = set()
    previous_interval: tuple[float, float] | None = None
    inferred_camera_ids: tuple[str, ...] | None = None
    inferred_camera_topics: dict[str, str] | None = None
    for index, raw_window in enumerate(raw_windows):
        window = _mapping(raw_window, field=f"manifest.windows[{index}]")
        window_id = _text(window.get("window_id"), field=f"windows[{index}].window_id")
        if window_id in seen_ids:
            raise ProductionGoldCollectionError(f"duplicate window_id: {window_id}")
        seen_ids.add(window_id)
        ordinal = window.get("ordinal", index)
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ProductionGoldCollectionError(
                f"windows[{index}].ordinal must be a non-negative integer"
            )
        if ordinal in seen_ordinals:
            raise ProductionGoldCollectionError(f"duplicate window ordinal: {ordinal}")
        if seen_ordinals and ordinal <= max(seen_ordinals):
            raise ProductionGoldCollectionError(
                "manifest window ordinals must be strictly increasing"
            )
        seen_ordinals.add(ordinal)
        source_window = _window_source(
            window,
            source_path=source_path,
            index=index,
            expected_camera_ids=source_camera_ids,
            expected_camera_count=source_camera_count,
            expected_camera_topics=source_camera_topics,
        )
        interval = tuple(source_window["interval"])
        if previous_interval is not None and interval[0] < previous_interval[1]:
            raise ProductionGoldCollectionError(
                "manifest windows must be ordered and non-overlapping"
            )
        previous_interval = (float(interval[0]), float(interval[1]))
        if source_camera_ids is None:
            current_camera_ids = tuple(source_window["camera_ids"])
            if inferred_camera_ids is None:
                inferred_camera_ids = current_camera_ids
            elif current_camera_ids != inferred_camera_ids:
                raise ProductionGoldCollectionError(
                    f"windows[{index}].camera_ids must be consistent across manifest windows"
                )
            current_topics = dict(source_window["camera_topics"])
            if inferred_camera_topics is None:
                inferred_camera_topics = current_topics
            elif current_topics != inferred_camera_topics:
                raise ProductionGoldCollectionError(
                    f"windows[{index}].camera_topics must be consistent across manifest windows"
                )

        reviewer_records: dict[str, dict[str, Any]] = {
            slot: {
                "slot": slot,
                "reviewer_id": None,
                "decision": None,
                "reviewed_at": None,
                "segments": [],
                "evidence_refs": [],
                "notes": None,
            }
            for slot in slots
        }
        windows.append(
            {
                "ordinal": ordinal,
                "window_id": window_id,
                "source": source_window,
                "review": {
                    "lifecycle_status": "PENDING",
                    "decision": None,
                    "decision_options": list(DECISION_OPTIONS),
                    "reviewer_slots": reviewer_records,
                    "adjudication": {
                        "status": "PENDING",
                        "adjudicator_id": None,
                        "adjudicated_at": None,
                        "decision": None,
                        "segments": [],
                        "evidence_refs": [],
                        "notes": None,
                    },
                },
                # Mirror the handoff's explicit window-level non-gold markers;
                # consumers should not have to inspect nested review/gold data
                # to determine whether a row is eligible for scoring.
                "official_quality_status": "NOT_MEASURED",
                "official_gold_status": "NOT_ESTABLISHED",
                "quality_claim": False,
                "human_adjudication": "NOT_PERFORMED",
                "production_eligible": False,
                "automatic_eligible": False,
                "automatic_qualification": False,
                "gold": {
                    "lifecycle_status": "PENDING",
                    "official_gold_status": "NOT_ESTABLISHED",
                    "segments": [],
                    "required_segment_fields": list(REQUIRED_SEGMENT_FIELDS),
                    "structured_label_fields": list(STRUCTURED_LABEL_FIELDS),
                    "provenance": {
                        "source_bound": True,
                        "source_path": source_path,
                        "camera_ids": list(source_window["camera_ids"]),
                        "source_interval": list(source_window["interval"]),
                        "action_boundaries_are_reviewer_entered": True,
                        "evidence_refs": [],
                        "reviewer_slots": list(slots),
                        "adjudicator_id": None,
                    },
                },
                "model_context": {
                    "wemm": {"status": "NOT_ATTACHED", "artifact_reference": None},
                    "qwen": {"status": "NOT_ATTACHED", "artifact_reference": None},
                    "mage": {"status": "NOT_ATTACHED", "artifact_reference": None},
                    "copied_into_gold": False,
                },
                "evidence": {
                    "source_path": source_path,
                    "camera_ids": list(source_window["camera_ids"]),
                    "surface_reference": normalized_evidence_reference,
                    "frame_refs": [],
                    "notes": None,
                },
            }
        )

    source_metadata = {
        "path": source_path,
        "media_type": media_type,
        "camera_count": source.get("camera_count"),
        "camera_ids": list(source_camera_ids or ()),
        "source_manifest_format": manifest_format,
        "source_manifest_reference": normalized_manifest_reference,
    }
    return {
        "format": SOURCE_BOUND_GOLD_COLLECTION_VERSION,
        "authority": AUTHORITY,
        "status": "PENDING",
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "human_adjudication": "NOT_PERFORMED",
        "production_eligible": False,
        "automatic_eligible": False,
        "automatic_qualification": False,
        "quality_measurement_status": "NOT_MEASURED",
        "source": source_metadata,
        "reviewer_slots": list(slots),
        "lifecycle": {
            "states": list(LIFECYCLE_STATES),
            "initial_state": "PENDING",
            "official_gold_requires_independent_adjudication": True,
        },
        "contract": {
            "decision_options": list(DECISION_OPTIONS),
            "required_segment_fields": list(REQUIRED_SEGMENT_FIELDS),
            "structured_label_fields": list(STRUCTURED_LABEL_FIELDS),
            "annotation_principal": {
                "source": ANNOTATION_PRINCIPAL_REFERENCE,
                "label_style": "concise action-attributes-object-location-hands",
                "consistent_verb_noun_pairs": True,
                "present_tense": True,
                "observable_only": True,
                "avoid_intent_assumptions": True,
                "explicit_object_interaction": True,
                "one_action_per_segment_when_possible": True,
                "split_when_visible_action_changes": True,
                "short_instruction_style": True,
            },
            "fixed_window_is_not_action_boundary": True,
            "model_outputs_are_context_only": True,
            "terra_or_agent_reviews_are_not_gold": True,
            "labels_must_be_entered_by_reviewer": True,
            "empty_segments_are_intentional": True,
        },
        "windows": windows,
        "controls": {
            "labels_inferred": False,
            "model_predictions_copied": False,
            "terra_labels_copied": False,
            "predictions_copied_to_gold": False,
            "gold_written": False,
            "official_gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "sha_or_digest_computed": False,
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render a compact operator-facing summary of a collection template."""

    source = _mapping(payload.get("source", {}), field="payload.source")
    windows = _sequence(payload.get("windows", []), field="payload.windows")
    lines = [
        "# Source-bound production gold collection template",
        "",
        f"- Status: `{payload.get('status', 'PENDING')}`",
        f"- Official gold: `{payload.get('official_gold_status', 'NOT_ESTABLISHED')}`",
        f"- Official quality: `{payload.get('official_quality_status', 'NOT_MEASURED')}`",
        f"- Quality measurement: `{payload.get('quality_measurement_status', 'NOT_MEASURED')}`",
        f"- Source: `{source.get('path', '')}`",
        f"- Windows: `{len(windows)}`",
        f"- Reviewer slots: `{', '.join(str(slot) for slot in payload.get('reviewer_slots', []))}`",
        "",
        "This file is a blank source-bound collection queue.  It contains no action labels.",
        "Fixed window intervals are context; reviewers must enter action boundaries.",
        (
            "Terra/agent/model suggestions remain outside the gold record until "
            "explicit independent adjudication."
        ),
        "",
        "| Window | Lifecycle | Decision | Segments |",
        "|---|---|---|---:|",
    ]
    for raw in windows:
        window = _mapping(raw, field="payload.windows[]")
        review = _mapping(window.get("review", {}), field="window.review")
        gold = _mapping(window.get("gold", {}), field="window.gold")
        lines.append(
            f"| {window.get('window_id', '')} | {review.get('lifecycle_status', 'PENDING')} | "
            f"{review.get('decision') or 'pending'} | {len(gold.get('segments', []))} |"
        )
    lines.extend(
        [
            "",
            "No model, media, ontology, Mapper, training, or hash operation is performed.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ANNOTATION_PRINCIPAL_REFERENCE",
    "AUTHORITY",
    "DECISION_OPTIONS",
    "DEFAULT_REVIEWER_SLOTS",
    "LIFECYCLE_STATES",
    "PRODUCTION_COHORT_MANIFEST_FORMAT",
    "REQUIRED_SEGMENT_FIELDS",
    "SOURCE_BOUND_GOLD_COLLECTION_VERSION",
    "STRUCTURED_LABEL_FIELDS",
    "ProductionGoldCollectionError",
    "build_source_bound_gold_collection",
    "render_markdown",
]
