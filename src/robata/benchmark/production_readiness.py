"""Lightweight readiness assessment for the production-shaped model cohort.

This is a benchmark-local planning/reporting seam.  It deliberately does not
load a model, decode media, infer labels, call the Mapper, or compute an
identity/hash.  Its purpose is to answer two separate questions before a
resource-heavy run is started:

* can the three model routes be invoked against the same source-bound windows?
* is there enough independent human/ontology evidence to measure quality?

The second question is intentionally stricter than the first.  An unlabeled raw
cohort may be useful for an exploratory model run, but its outputs are never a
quality result.  ``production_eligible`` is permanently false for this local
artifact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from .production_cohort import DEFAULT_CAMERA_TOPICS
from .production_model_output import (
    DEFAULT_NATIVE_MODEL_ROUTES,
    MODEL_NAMES,
    PRODUCTION_COHORT_MANIFEST_FORMAT,
    ProductionModelOutputError,
    validate_model_output_sidecar,
)

PRODUCTION_READINESS_VERSION: Final = "robata-production-three-model-readiness-v1"
LOCAL_NONPRODUCTION_AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
EXPECTED_CAMERA_IDS: Final = tuple(DEFAULT_CAMERA_TOPICS)


class ProductionReadinessError(ValueError):
    """Raised when a readiness input is malformed."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionReadinessError(f"{field} must be an object")
    return value


def _array(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionReadinessError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionReadinessError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, *, field: str) -> float:
    """Parse a source-relative time bound as a finite non-negative number."""

    if isinstance(value, bool):
        raise ProductionReadinessError(f"{field} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionReadinessError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or number < 0:
        raise ProductionReadinessError(f"{field} must be a finite number")
    return number


def _copy_json(value: object, *, field: str) -> Any:
    """Copy JSON-shaped metadata without importing or producing a digest."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionReadinessError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionReadinessError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionReadinessError(f"{field} must be JSON-compatible")


def _gate(
    name: str,
    status: str,
    reason: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "status": status,
        "reason": reason,
    }
    if details:
        result["details"] = _copy_json(details, field=f"gates.{name}.details")
    return result


def _source_camera_projection(
    source: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str] | None]:
    """Validate optional source camera inventory and return canonical IDs/topics."""

    expected = EXPECTED_CAMERA_IDS
    camera_count = source.get("camera_count")
    if camera_count is not None:
        if isinstance(camera_count, bool) or not isinstance(camera_count, int):
            raise ProductionReadinessError("manifest.source.camera_count must be an integer")
        if camera_count != len(expected):
            raise ProductionReadinessError(
                f"manifest.source.camera_count must be {len(expected)} for a six-camera cohort"
            )
    raw_cameras = source.get("cameras")
    if raw_cameras is None:
        return expected, None
    rows = _array(raw_cameras, field="manifest.source.cameras")
    if len(rows) != len(expected):
        raise ProductionReadinessError(
            f"manifest.source.cameras must contain exactly {len(expected)} cameras"
        )
    ids: list[str] = []
    topics: dict[str, str] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"manifest.source.cameras[{index}]")
        camera_id = _text(
            row.get("camera_id", row.get("id")),
            field=f"manifest.source.cameras[{index}].camera_id",
        )
        topic = _text(
            row.get("topic", row.get("camera_topic")),
            field=f"manifest.source.cameras[{index}].topic",
        )
        if camera_id in topics:
            raise ProductionReadinessError(f"duplicate source camera ID: {camera_id}")
        ids.append(camera_id)
        topics[camera_id] = topic
    if tuple(ids) != expected:
        raise ProductionReadinessError(
            "manifest.source.cameras must be ordered cam_01 through cam_06"
        )
    if len(set(topics.values())) != len(topics):
        raise ProductionReadinessError("manifest.source camera topics must be unique")
    return tuple(ids), topics


def _window_camera_topics(
    value: object,
    *,
    field: str,
    expected_ids: tuple[str, ...],
    expected_topics: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Validate a manifest window's optional camera-topic map."""

    if value is None:
        return None
    topics_raw = _mapping(value, field=field)
    if tuple(topics_raw) != expected_ids:
        raise ProductionReadinessError(f"{field} must contain cam_01 through cam_06 in order")
    topics: dict[str, str] = {
        camera_id: _text(topics_raw[camera_id], field=f"{field}.{camera_id}")
        for camera_id in expected_ids
    }
    if len(set(topics.values())) != len(topics):
        raise ProductionReadinessError(f"{field} topics must be unique")
    if expected_topics is not None and topics != dict(expected_topics):
        raise ProductionReadinessError(f"{field} does not bind manifest source camera topics")
    return topics


def _manifest_routes(window: Mapping[str, Any], *, field: str) -> dict[str, str]:
    """Read a per-window route map, defaulting to the declared native routes."""

    raw = window.get("model_routes")
    if raw is None:
        return dict(DEFAULT_NATIVE_MODEL_ROUTES)
    routes = _mapping(raw, field=f"{field}.model_routes")
    if set(routes) != set(MODEL_NAMES):
        raise ProductionReadinessError(
            f"{field}.model_routes must contain exactly WeMM, Qwen, and Mage"
        )
    return {
        model: _text(routes[model], field=f"{field}.model_routes.{model}") for model in MODEL_NAMES
    }


def _manifest_projection(manifest: Mapping[str, Any]) -> tuple[str, tuple[dict[str, Any], ...]]:
    if manifest.get("format") != PRODUCTION_COHORT_MANIFEST_FORMAT:
        raise ProductionReadinessError("manifest.format must be robata-production-shaped-cohort-v1")
    if manifest.get("authority") != LOCAL_NONPRODUCTION_AUTHORITY:
        raise ProductionReadinessError("manifest authority must be LOCAL_NONPRODUCTION_ONLY")
    source = _mapping(manifest.get("source"), field="manifest.source")
    source_path = _text(source.get("path"), field="manifest.source.path")
    expected_camera_ids, expected_camera_topics = _source_camera_projection(source)
    raw_windows = _array(manifest.get("windows"), field="manifest.windows")
    if not raw_windows:
        raise ProductionReadinessError("manifest.windows must not be empty")
    windows: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_ordinals: set[int] = set()
    for index, raw in enumerate(raw_windows):
        window = _mapping(raw, field=f"manifest.windows[{index}]")
        window_id = _text(window.get("window_id"), field=f"manifest.windows[{index}].window_id")
        if window_id in seen:
            raise ProductionReadinessError(f"duplicate manifest window_id: {window_id}")
        seen.add(window_id)
        ordinal = window.get("ordinal", index)
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ProductionReadinessError(
                f"manifest.windows[{index}].ordinal must be a non-negative integer"
            )
        if ordinal in seen_ordinals:
            raise ProductionReadinessError(f"duplicate manifest ordinal: {ordinal}")
        if windows and ordinal <= windows[-1]["ordinal"]:
            raise ProductionReadinessError("manifest ordinals must be strictly increasing")
        seen_ordinals.add(ordinal)
        start = _finite_number(
            window.get("start_seconds"), field=f"manifest.windows[{index}].start_seconds"
        )
        end = _finite_number(
            window.get("end_seconds"), field=f"manifest.windows[{index}].end_seconds"
        )
        if end <= start:
            raise ProductionReadinessError(f"manifest.windows[{index}] end must exceed start")
        if windows and start < windows[-1]["end_seconds"]:
            raise ProductionReadinessError("manifest windows must be ordered and non-overlapping")
        cameras = _array(window.get("camera_ids"), field=f"manifest.windows[{index}].camera_ids")
        camera_ids = tuple(
            _text(camera, field=f"manifest.windows[{index}].camera_ids[{camera_index}]")
            for camera_index, camera in enumerate(cameras)
        )
        if camera_ids != expected_camera_ids:
            raise ProductionReadinessError(
                f"manifest.windows[{index}].camera_ids must be cam_01 through cam_06 in order"
            )
        camera_topics = _window_camera_topics(
            window.get("camera_topics"),
            field=f"manifest.windows[{index}].camera_topics",
            expected_ids=expected_camera_ids,
            expected_topics=expected_camera_topics,
        )
        model_routes = _manifest_routes(window, field=f"manifest.windows[{index}]")
        windows.append(
            {
                "ordinal": ordinal,
                "window_id": window_id,
                "start_seconds": start,
                "end_seconds": end,
                "camera_ids": camera_ids,
                "camera_topics": camera_topics,
                "model_routes": model_routes,
            }
        )
    first_routes = windows[0]["model_routes"]
    if any(window["model_routes"] != first_routes for window in windows[1:]):
        raise ProductionReadinessError(
            "manifest model_routes must be consistent across all windows"
        )
    first_cameras = windows[0]["camera_ids"]
    if any(window["camera_ids"] != first_cameras for window in windows[1:]):
        raise ProductionReadinessError("manifest camera_ids must be consistent across all windows")
    return source_path, tuple(windows)


def _check_window_binding(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    field: str,
    require_cameras: bool = True,
) -> None:
    """Require a review/sidecar window to retain source-bound geometry."""

    for key in ("ordinal", "start_seconds", "end_seconds"):
        if key not in value:
            raise ProductionReadinessError(f"{field}.{key} is required for source binding")
    ordinal = value["ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal != expected["ordinal"]:
        raise ProductionReadinessError(f"{field}.ordinal does not bind manifest window")
    start = _finite_number(value["start_seconds"], field=f"{field}.start_seconds")
    end = _finite_number(value["end_seconds"], field=f"{field}.end_seconds")
    if start != expected["start_seconds"] or end != expected["end_seconds"]:
        raise ProductionReadinessError(f"{field} geometry does not bind manifest window")
    if end <= start:
        raise ProductionReadinessError(f"{field} end must exceed start")
    if require_cameras:
        cameras = _array(value.get("camera_ids"), field=f"{field}.camera_ids")
        camera_ids = tuple(
            _text(camera, field=f"{field}.camera_ids[{index}]")
            for index, camera in enumerate(cameras)
        )
        if camera_ids != tuple(expected["camera_ids"]):
            raise ProductionReadinessError(f"{field}.camera_ids do not bind manifest window")


def _validate_accepted_gold(
    gold: Mapping[str, Any], *, field: str, expected: Mapping[str, Any]
) -> int:
    """Validate the minimum observable gold contract before quality scoring."""

    segments = _array(gold.get("segments", []), field=f"{field}.segments")
    if not segments:
        return 0
    provenance = _mapping(gold.get("provenance"), field=f"{field}.provenance")
    for key in ("reviewer_id", "reviewed_at"):
        _text(provenance.get(key), field=f"{field}.provenance.{key}")
    if str(provenance.get("adjudication_status", "")).upper() not in {
        "ACCEPTED",
        "ADJUDICATED",
        "COMPLETED",
    }:
        raise ProductionReadinessError(
            f"{field}.provenance.adjudication_status must record completed adjudication"
        )
    for index, raw_segment in enumerate(segments):
        segment = _mapping(raw_segment, field=f"{field}.segments[{index}]")
        start = _finite_number(
            segment.get("start_seconds"),
            field=f"{field}.segments[{index}].start_seconds",
        )
        end = _finite_number(
            segment.get("end_seconds"),
            field=f"{field}.segments[{index}].end_seconds",
        )
        if start < expected["start_seconds"] or end > expected["end_seconds"] or end <= start:
            raise ProductionReadinessError(
                f"{field}.segments[{index}] is outside the source-bound window"
            )
        # The structured action contract requires at least verb and noun.  The
        # remaining attributes/location/hand fields may legitimately be null.
        for key in ("verb", "noun"):
            _text(segment.get(key), field=f"{field}.segments[{index}].{key}")
    return len(segments)


def _review_gate(
    review_pack: Mapping[str, Any] | None,
    *,
    source_path: str,
    windows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if review_pack is None:
        return _gate("human_review", "BLOCKED", "review pack was not supplied")
    if review_pack.get("format") != "robata-production-human-review-pack-v1":
        raise ProductionReadinessError("review_pack.format is not supported")
    if review_pack.get("authority") != LOCAL_NONPRODUCTION_AUTHORITY:
        raise ProductionReadinessError("review pack authority must be LOCAL_NONPRODUCTION_ONLY")
    if review_pack.get("source_manifest_format") != PRODUCTION_COHORT_MANIFEST_FORMAT:
        raise ProductionReadinessError(
            "review_pack.source_manifest_format does not bind cohort manifest"
        )
    source = _mapping(review_pack.get("source"), field="review_pack.source")
    review_source = _text(source.get("path"), field="review_pack.source.path")
    if review_source != source_path:
        raise ProductionReadinessError("review pack source does not bind manifest source")
    items = _array(review_pack.get("items"), field="review_pack.items")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(items):
        item = _mapping(raw, field=f"review_pack.items[{index}]")
        item_id = _text(item.get("window_id"), field=f"review_pack.items[{index}].window_id")
        if item_id in by_id:
            raise ProductionReadinessError(f"duplicate review window_id: {item_id}")
        by_id[item_id] = item
    missing = [window["window_id"] for window in windows if window["window_id"] not in by_id]
    extra = sorted(set(by_id) - {window["window_id"] for window in windows})
    if missing or extra:
        return _gate(
            "human_review",
            "BLOCKED",
            "review pack windows do not match manifest",
            details={"missing_windows": missing, "extra_windows": extra},
        )
    pending: list[str] = []
    accepted: list[str] = []
    for window in windows:
        item = by_id[window["window_id"]]
        item_source_path = _text(
            item.get("source_path"),
            field=f"review_pack.{window['window_id']}.source_path",
        )
        if item_source_path != source_path:
            raise ProductionReadinessError(
                f"review item {window['window_id']} source does not bind manifest source"
            )
        _check_window_binding(
            item,
            window,
            field=f"review_pack.{window['window_id']}",
            require_cameras=True,
        )
        gold = _mapping(item.get("gold"), field=f"review_pack.{window['window_id']}.gold")
        status = str(gold.get("status", "PENDING_HUMAN_REVIEW"))
        if status == "ACCEPTED":
            segment_count = _validate_accepted_gold(
                gold,
                field=f"review_pack.{window['window_id']}.gold",
                expected=window,
            )
            if segment_count:
                accepted.append(window["window_id"])
            else:
                pending.append(window["window_id"])
        else:
            pending.append(window["window_id"])
    if pending:
        return _gate(
            "human_review",
            "BLOCKED",
            "one or more windows lack accepted human gold",
            details={"accepted_windows": accepted, "pending_windows": pending},
        )
    return _gate(
        "human_review",
        "READY",
        "all windows have accepted non-empty human gold",
        details={"accepted_windows": accepted},
    )


def _approval_gate(
    name: str,
    value: Mapping[str, Any] | None,
    *,
    entry_keys: tuple[str, ...],
) -> dict[str, Any]:
    if value is None:
        return _gate(name, "BLOCKED", f"{name} was not supplied")
    approved = value.get("approved") is True or str(value.get("approval_status", "")).upper() in {
        "APPROVED",
        "READY",
    }
    entries: object | None = None
    entry_key: str | None = None
    for key in entry_keys:
        if key in value:
            entries = value[key]
            entry_key = key
            break
    if not approved:
        return _gate(name, "BLOCKED", f"{name} is not approved")
    if entries is None or entry_key is None:
        return _gate(name, "BLOCKED", f"{name} contains no explicit entries")
    try:
        if not _array(entries, field=f"{name}.{entry_key}"):
            return _gate(name, "BLOCKED", f"{name} contains no entries")
    except ProductionReadinessError:
        # Some mapping profiles use an object keyed by camera/action.  A
        # non-empty mapping is still an explicit profile, not inferred data.
        if not isinstance(entries, Mapping) or not entries:
            return _gate(name, "BLOCKED", f"{name} entries are malformed")
    return _gate(name, "READY", f"{name} is explicitly approved")


def _sidecar_gate(
    sidecar: Mapping[str, Any] | None,
    *,
    source_path: str,
    windows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if sidecar is None:
        return (
            _gate(
                "model_output_contract",
                "BLOCKED",
                "model-output sidecar was not supplied",
            ),
            None,
        )
    try:
        validated = validate_model_output_sidecar(sidecar)
    except ProductionModelOutputError as exc:
        raise ProductionReadinessError(f"invalid model-output sidecar: {exc}") from exc
    if validated["source"]["path"] != source_path:
        raise ProductionReadinessError("model-output sidecar source does not bind manifest source")
    sidecar_ids = {window["window_id"] for window in validated["windows"]}
    manifest_ids = {window["window_id"] for window in windows}
    if sidecar_ids != manifest_ids:
        return (
            _gate(
                "model_output_contract",
                "BLOCKED",
                "model-output sidecar windows do not match manifest",
                details={
                    "missing_windows": sorted(manifest_ids - sidecar_ids),
                    "extra_windows": sorted(sidecar_ids - manifest_ids),
                },
            ),
            validated,
        )
    # The sidecar validator binds every slot to its own top-level route map,
    # but the readiness boundary must also bind that map back to the cohort
    # manifest.  Otherwise a hand-authored sidecar could silently substitute a
    # different WeMM/Qwen/Mage representation while retaining the same windows.
    expected_routes = windows[0]["model_routes"]
    if any(window["model_routes"] != expected_routes for window in windows[1:]):
        raise ProductionReadinessError(
            "manifest model_routes must be consistent across all windows"
        )
    if validated["model_routes"] != expected_routes:
        raise ProductionReadinessError(
            "model-output sidecar model_routes do not bind manifest model_routes"
        )
    manifest_by_id = {window["window_id"]: window for window in windows}
    for sidecar_window in validated["windows"]:
        expected = manifest_by_id[sidecar_window["window_id"]]
        _check_window_binding(
            sidecar_window,
            expected,
            field=f"sidecar.{sidecar_window['window_id']}",
            require_cameras=True,
        )
    statuses = {
        model: [window["model_outputs"][model]["status"] for window in validated["windows"]]
        for model in MODEL_NAMES
    }
    not_run = [
        model for model, values in statuses.items() if all(value == "NOT_RUN" for value in values)
    ]
    succeeded = [
        model for model, values in statuses.items() if all(value == "SUCCEEDED" for value in values)
    ]
    if succeeded == list(MODEL_NAMES):
        status = "READY"
        reason = "all model slots succeeded"
    elif not_run == list(MODEL_NAMES):
        status = "READY"
        reason = "sidecar is initialized and ready for model invocation"
    else:
        status = "PARTIAL"
        reason = "some model slots are running, failed, cancelled, or incomplete"
    return (
        _gate(
            "model_output_contract",
            status,
            reason,
            details={"statuses": statuses, "all_models_succeeded": succeeded == list(MODEL_NAMES)},
        ),
        validated,
    )


def assess_production_readiness(
    manifest: Mapping[str, Any],
    *,
    review_pack: Mapping[str, Any] | None = None,
    sidecar: Mapping[str, Any] | None = None,
    ontology: Mapping[str, Any] | None = None,
    mapping: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess invocation and quality readiness without running any model."""

    manifest_value = _mapping(manifest, field="manifest")
    source_path, windows = _manifest_projection(manifest_value)
    source_gate = _gate(
        "source_windows",
        "READY",
        "source-bound manifest and bounded windows are structurally valid",
        details={"source_path": source_path, "window_count": len(windows)},
    )
    review_gate = _review_gate(review_pack, source_path=source_path, windows=windows)
    sidecar_gate, validated_sidecar = _sidecar_gate(
        sidecar, source_path=source_path, windows=windows
    )
    ontology_gate = _approval_gate(
        "ontology",
        ontology,
        entry_keys=("actions", "entries", "labels", "ontology"),
    )
    mapping_gate = _approval_gate(
        "mapping",
        mapping,
        entry_keys=("camera_mapping", "mappings", "entries", "topics"),
    )

    invocation_ready = all(
        gate["status"] == "READY"
        for gate in (source_gate, sidecar_gate, ontology_gate, mapping_gate)
    )
    model_statuses: dict[str, list[str]] = {}
    metrics_statuses: dict[str, list[str]] = {}
    if validated_sidecar is not None:
        model_statuses = {
            model: [
                window["model_outputs"][model]["status"] for window in validated_sidecar["windows"]
            ]
            for model in MODEL_NAMES
        }
        metrics_statuses = {
            model: [
                window["model_outputs"][model]["metrics"]["measurement_status"]
                for window in validated_sidecar["windows"]
            ]
            for model in MODEL_NAMES
        }
    models_complete = bool(model_statuses) and all(
        status == "SUCCEEDED" for statuses in model_statuses.values() for status in statuses
    )
    metrics_measured = bool(metrics_statuses) and all(
        status == "MEASURED" for statuses in metrics_statuses.values() for status in statuses
    )
    quality_ready = (
        invocation_ready
        and review_gate["status"] == "READY"
        and models_complete
        and metrics_measured
    )
    blockers: list[str] = []
    if not invocation_ready:
        for gate in (source_gate, sidecar_gate, ontology_gate, mapping_gate):
            if gate["status"] != "READY":
                blockers.append(f"{gate['name']}: {gate['reason']}")
    if not quality_ready:
        if review_gate["status"] != "READY":
            blockers.append(f"human_review: {review_gate['reason']}")
        if invocation_ready and (not models_complete or not metrics_measured):
            blockers.append("model outputs/metrics are not yet quality-measureable")

    execution_controls = (
        validated_sidecar.get("controls", {}) if validated_sidecar is not None else {}
    )
    return {
        "format": PRODUCTION_READINESS_VERSION,
        "authority": LOCAL_NONPRODUCTION_AUTHORITY,
        "production_eligible": False,
        "inference_readiness": "READY" if invocation_ready else "BLOCKED",
        "quality_readiness": "READY" if quality_ready else "NOT_MEASURED",
        "quality_measurement_status": "MEASURED" if quality_ready else "NOT_MEASURED",
        "gates": [source_gate, review_gate, sidecar_gate, ontology_gate, mapping_gate],
        "blockers": blockers,
        "source": {"path": source_path, "window_count": len(windows)},
        "model_statuses": model_statuses,
        "metrics_statuses": metrics_statuses,
        "controls": {
            "model_invoked": bool(execution_controls.get("model_invoked", False)),
            "gpu_invoked": bool(execution_controls.get("gpu_invoked", False)),
            "frames_decoded": bool(execution_controls.get("frames_decoded", False)),
            # These are immutable safety boundaries in the sidecar contract;
            # retaining explicit false values makes the report self-contained.
            "gold_included": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
        },
    }


build_production_readiness = assess_production_readiness
assess_three_model_readiness = assess_production_readiness


__all__ = [
    "LOCAL_NONPRODUCTION_AUTHORITY",
    "PRODUCTION_READINESS_VERSION",
    "ProductionReadinessError",
    "assess_production_readiness",
    "assess_three_model_readiness",
    "build_production_readiness",
]
