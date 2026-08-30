"""Label-blind output slots for the production-shaped model cohort.

The production-shaped cohort manifest binds source media and bounded windows,
but it intentionally contains no model output.  This module provides the small
benchmark-local sidecar used to collect observations from the three model
routes (WeMM, Qwen, and Mage) without changing the manifest or the human-review
pack.

The sidecar is *not* a product or published wire schema.  It is an explicit,
non-production artifact with one independent slot for every ``window_id`` and
model.  A slot may retain model claims and runtime artifacts, while gold labels
remain in the separate review contract.  In particular, this module never
copies ``manifest["gold"]`` or ``review`` data into an output slot and rejects
an attempted gold/annotation field in a supplied slot or prediction payload.

No model, processor, media decoder, mapper, ontology, or digest implementation
is imported here.  Initialisation is therefore safe to run before any heavy
runtime is available.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, TypeGuard, cast

PRODUCTION_MODEL_OUTPUT_SIDECAR_VERSION: Final = "robata-production-model-output-sidecar-v1"
"""Version of this benchmark-local sidecar contract."""

# ``FORMAT`` is a friendlier spelling for callers that use the manifest's
# ``format`` field directly.  Keep both names stable; no schema catalog entry
# is created for this local artifact.
PRODUCTION_MODEL_OUTPUT_FORMAT: Final = PRODUCTION_MODEL_OUTPUT_SIDECAR_VERSION

PRODUCTION_COHORT_MANIFEST_FORMAT: Final = "robata-production-shaped-cohort-v1"
LOCAL_NONPRODUCTION_AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"

MODEL_NAMES: Final = ("wemm", "qwen", "mage")
PRODUCTION_MODEL_NAMES: Final = MODEL_NAMES

# These are the route names emitted by ``production_cohort``.  They describe
# the representation boundary, not a claim that a model has run or that the
# route is production-authorized.
DEFAULT_NATIVE_MODEL_ROUTES: Final[dict[str, str]] = {
    "wemm": "complete_bounded_video_embedding",
    "qwen": "complete_native_video",
    "mage": "complete_bounded_native_codec",
}
DEFAULT_MODEL_ROUTES: Final[dict[str, str]] = DEFAULT_NATIVE_MODEL_ROUTES
NATIVE_MODEL_ROUTES: Final[dict[str, str]] = DEFAULT_NATIVE_MODEL_ROUTES

MODEL_OUTPUT_STATUSES: Final = (
    "NOT_RUN",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    "SKIPPED",
)
_STATUS_SET = frozenset(MODEL_OUTPUT_STATUSES)

_MEASUREMENT_NOT_MEASURED = "NOT_MEASURED"


class ProductionModelOutputError(ValueError):
    """Raised when a model-output sidecar violates its local contract."""


# A compatibility spelling for callers that use ``ContractError`` in their
# benchmark modules.
ProductionModelOutputContractError = ProductionModelOutputError


_GOLD_EXACT_KEYS = frozenset(
    {
        "gold",
        "goldstatus",
        "goldlabel",
        "goldlabels",
        "groundtruth",
        "groundtruthlabel",
        "groundtruthlabels",
        "officiallabel",
        "officiallabels",
        "officialreference",
        "annotation",
        "annotations",
        "humanlabel",
        "humanlabels",
        "review",
        "adjudication",
    }
)
_GOLD_KEY_FRAGMENTS = (
    "gold",
    "groundtruth",
    "goldlabel",
    "officiallabel",
    "officialreference",
    "humanlabel",
    "humanannotation",
)


def _normalised_key(value: str) -> str:
    """Normalise a key for the narrow gold-contamination guard."""

    return "".join(character for character in value.casefold() if character.isalnum())


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionModelOutputError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionModelOutputError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ProductionModelOutputError(f"{field} must be a finite non-negative number")
    return number


def _json_copy(value: object, *, field: str) -> Any:
    """Deep-copy JSON-compatible data and reject unsupported/non-finite values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionModelOutputError(f"{field} must not contain a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionModelOutputError(f"{field} mapping keys must be strings")
            result[key] = _json_copy(child, field=f"{field}.{key}")
        return result
    if _is_sequence(value):
        return [_json_copy(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionModelOutputError(f"{field} must be JSON-compatible")


def _assert_no_gold_fields(value: object, *, field: str) -> None:
    """Reject explicit gold/annotation keys while allowing model prose/claims.

    A prediction may legitimately contain fields such as ``verb`` and
    ``noun``; those are model claims.  The guard therefore checks field names
    that identify an official or human label rather than lexical values.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionModelOutputError(f"{field} must not contain a non-finite number")
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ProductionModelOutputError(f"{field} mapping keys must be strings")
            normalised = _normalised_key(raw_key)
            if normalised in _GOLD_EXACT_KEYS or any(
                fragment in normalised for fragment in _GOLD_KEY_FRAGMENTS
            ):
                raise ProductionModelOutputError(
                    f"{field}.{raw_key} must not contain gold or annotation data"
                )
            _assert_no_gold_fields(child, field=f"{field}.{raw_key}")
        return
    if _is_sequence(value):
        for index, child in enumerate(value):
            _assert_no_gold_fields(child, field=f"{field}[{index}]")
        return
    raise ProductionModelOutputError(f"{field} must be JSON-compatible")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionModelOutputError(f"{field} must be an object")
    return value


def _array(value: object, *, field: str) -> Sequence[Any]:
    if not _is_sequence(value):
        raise ProductionModelOutputError(f"{field} must be an array")
    return cast(Sequence[Any], value)


def _route_map(window: Mapping[str, Any], *, field: str) -> dict[str, str]:
    raw_routes = window.get("model_routes")
    if raw_routes is None:
        return dict(DEFAULT_NATIVE_MODEL_ROUTES)
    routes = _mapping(raw_routes, field=f"{field}.model_routes")
    unexpected = sorted(set(routes) - set(MODEL_NAMES))
    if unexpected:
        raise ProductionModelOutputError(
            f"{field}.model_routes contains unsupported models: {', '.join(unexpected)}"
        )
    result: dict[str, str] = {}
    for model in MODEL_NAMES:
        route = _required_text(
            routes.get(model, DEFAULT_NATIVE_MODEL_ROUTES[model]),
            field=f"{field}.model_routes.{model}",
        )
        # A route in this contract is required to be the named native route;
        # accepting an absent/empty route would make a slot ambiguous.  We do
        # not require a particular endpoint or provider here.
        result[model] = route
    return result


def _source_projection(manifest: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    source = _mapping(manifest.get("source"), field="manifest.source")
    source_path = _required_text(source.get("path"), field="manifest.source.path")
    projection: dict[str, Any] = {"path": source_path}
    for key in ("media_type", "camera_count"):
        if key in source:
            projection[key] = _json_copy(source[key], field=f"manifest.source.{key}")
    _assert_no_gold_fields(projection, field="sidecar.source")
    return source_path, projection


def _manifest_windows(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_windows = _array(manifest.get("windows"), field="manifest.windows")
    if not raw_windows:
        raise ProductionModelOutputError("manifest.windows must be non-empty")

    windows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_window in enumerate(raw_windows):
        window = _mapping(raw_window, field=f"manifest.windows[{index}]")
        window_id = _required_text(
            window.get("window_id"), field=f"manifest.windows[{index}].window_id"
        )
        if window_id in seen_ids:
            raise ProductionModelOutputError(f"duplicate manifest window_id: {window_id}")
        seen_ids.add(window_id)
        ordinal_value = window.get("ordinal", index)
        if (
            isinstance(ordinal_value, bool)
            or not isinstance(ordinal_value, int)
            or ordinal_value < 0
        ):
            raise ProductionModelOutputError(
                f"manifest.windows[{index}].ordinal must be a non-negative integer"
            )
        start = _finite_nonnegative(
            window.get("start_seconds"), field=f"manifest.windows[{index}].start_seconds"
        )
        end = _finite_nonnegative(
            window.get("end_seconds"), field=f"manifest.windows[{index}].end_seconds"
        )
        if end <= start:
            raise ProductionModelOutputError(
                f"manifest.windows[{index}] end_seconds must be greater than start_seconds"
            )
        camera_ids: list[str] | None = None
        if "camera_ids" in window:
            raw_camera_ids = _array(
                window.get("camera_ids"), field=f"manifest.windows[{index}].camera_ids"
            )
            camera_ids = [
                _required_text(
                    camera_id,
                    field=f"manifest.windows[{index}].camera_ids[{camera_index}]",
                )
                for camera_index, camera_id in enumerate(raw_camera_ids)
            ]
            if len(set(camera_ids)) != len(camera_ids):
                raise ProductionModelOutputError(
                    f"manifest.windows[{index}].camera_ids must be unique"
                )
        routes = _route_map(window, field=f"manifest.windows[{index}]")
        windows.append(
            {
                "ordinal": ordinal_value,
                "window_id": window_id,
                "start_seconds": start,
                "end_seconds": end,
                "camera_ids": camera_ids,
                "model_routes": routes,
            }
        )
    return tuple(windows)


def _initial_metrics() -> dict[str, Any]:
    # ``measurement_status`` follows the existing benchmark metric contracts;
    # ``status`` is retained as a compact sidecar-facing spelling.  Keeping
    # both equal makes the epistemic state difficult to omit in hand-authored
    # JSON while retaining compatibility with either consumer convention.
    return {
        "measurement_status": _MEASUREMENT_NOT_MEASURED,
        "status": _MEASUREMENT_NOT_MEASURED,
        "values": {},
    }


def _initial_artifact_lineage(
    *,
    manifest_format: str,
    source_path: str,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_manifest_format": manifest_format,
        "source_path": source_path,
        "window_id": window["window_id"],
        "window_ordinal": window["ordinal"],
        "input_artifacts": [],
        "output_artifacts": [],
        "raw_output_artifact": None,
        "parsed_output_artifact": None,
    }


def _initial_slot(
    *,
    model: str,
    route: str,
    manifest_format: str,
    source_path: str,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    window_id = str(window["window_id"])
    lineage = _initial_artifact_lineage(
        manifest_format=manifest_format,
        source_path=source_path,
        window=window,
    )
    return {
        "model": model,
        "window_id": window_id,
        "native_route": route,
        "status": "NOT_RUN",
        "predictions": [],
        "metrics": _initial_metrics(),
        "artifact_lineage": lineage,
    }


def build_model_output_sidecar(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Initialise independent WeMM/Qwen/Mage output slots from a cohort.

    Only source identity and window geometry are copied.  The manifest's
    ``gold`` and ``review`` sections are deliberately not copied.  The return
    value is a fresh JSON-shaped dictionary and can be serialised directly.
    """

    if not isinstance(manifest, Mapping):
        raise ProductionModelOutputError("manifest must be an object")
    manifest_format = _required_text(manifest.get("format"), field="manifest.format")
    if manifest_format != PRODUCTION_COHORT_MANIFEST_FORMAT:
        raise ProductionModelOutputError(
            "manifest.format must be robata-production-shaped-cohort-v1"
        )
    authority = manifest.get("authority", LOCAL_NONPRODUCTION_AUTHORITY)
    if authority != LOCAL_NONPRODUCTION_AUTHORITY:
        raise ProductionModelOutputError("manifest authority must be LOCAL_NONPRODUCTION_ONLY")
    source_path, source = _source_projection(manifest)
    windows = _manifest_windows(manifest)

    output_windows: list[dict[str, Any]] = []
    for window in windows:
        item: dict[str, Any] = {
            "ordinal": window["ordinal"],
            "window_id": window["window_id"],
            "start_seconds": window["start_seconds"],
            "end_seconds": window["end_seconds"],
            "model_outputs": {
                model: _initial_slot(
                    model=model,
                    route=window["model_routes"][model],
                    manifest_format=manifest_format,
                    source_path=source_path,
                    window=window,
                )
                for model in MODEL_NAMES
            },
        }
        if window["camera_ids"] is not None:
            item["camera_ids"] = list(window["camera_ids"])
        output_windows.append(item)

    # The current cohort manifest uses one route per model for every window.
    # Keep that useful summary at the top level, while retaining the exact
    # per-window route on each slot.  Reject a mixed-route manifest rather than
    # silently collapsing it into a misleading summary.
    first_routes = dict(windows[0]["model_routes"])
    if any(window["model_routes"] != first_routes for window in windows[1:]):
        raise ProductionModelOutputError(
            "manifest model_routes must be consistent across all windows"
        )

    sidecar = {
        "format": PRODUCTION_MODEL_OUTPUT_SIDECAR_VERSION,
        "authority": LOCAL_NONPRODUCTION_AUTHORITY,
        "source_manifest_format": manifest_format,
        "source": source,
        "model_routes": first_routes,
        "windows": output_windows,
        "contract": {
            "model_names": list(MODEL_NAMES),
            "predictions_are_model_claims": True,
            "gold_is_external": True,
            "gold_fields_included": False,
            "model_outputs_are_not_gold": True,
        },
        "controls": {
            "model_invoked": False,
            "gpu_invoked": False,
            "gold_included": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "frames_decoded": False,
        },
    }
    # The builder itself is a contract boundary; validate before exposing the
    # fresh object so future edits cannot accidentally widen the shape.
    return validate_model_output_sidecar(sidecar)


def _validate_metrics(value: object, *, field: str) -> dict[str, Any]:
    metrics = _mapping(value, field=field)
    measurement_status = metrics.get("measurement_status", metrics.get("status"))
    if measurement_status not in {_MEASUREMENT_NOT_MEASURED, "MEASURED"}:
        raise ProductionModelOutputError(
            f"{field}.measurement_status must be NOT_MEASURED or MEASURED"
        )
    status = metrics.get("status", measurement_status)
    if status != measurement_status:
        raise ProductionModelOutputError(f"{field}.status must match measurement_status")
    values = metrics.get("values", {})
    values_copy = _json_copy(values, field=f"{field}.values")
    _assert_no_gold_fields(values_copy, field=f"{field}.values")
    if measurement_status == _MEASUREMENT_NOT_MEASURED and values_copy not in ({}, None):
        raise ProductionModelOutputError(f"{field} marked NOT_MEASURED cannot retain metric values")
    return {
        "measurement_status": measurement_status,
        "status": status,
        "values": {} if values_copy is None else values_copy,
    }


def _validate_lineage(
    value: object,
    *,
    field: str,
    manifest_format: str,
    source_path: str,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = _mapping(value, field=field)
    _assert_no_gold_fields(lineage, field=field)
    if lineage.get("source_manifest_format") != manifest_format:
        raise ProductionModelOutputError(f"{field}.source_manifest_format does not bind sidecar")
    if lineage.get("source_path") != source_path:
        raise ProductionModelOutputError(f"{field}.source_path does not bind sidecar source")
    if lineage.get("window_id") != window["window_id"]:
        raise ProductionModelOutputError(f"{field}.window_id does not bind slot window")
    if lineage.get("window_ordinal") != window["ordinal"]:
        raise ProductionModelOutputError(f"{field}.window_ordinal does not bind slot window")
    for key in ("input_artifacts", "output_artifacts"):
        artifacts = _array(lineage.get(key), field=f"{field}.{key}")
        for index, artifact in enumerate(artifacts):
            _json_copy(artifact, field=f"{field}.{key}[{index}]")
    for key in ("raw_output_artifact", "parsed_output_artifact"):
        if lineage.get(key) is not None:
            _json_copy(lineage[key], field=f"{field}.{key}")
    return cast(dict[str, Any], _json_copy(lineage, field=field))


def _validate_slot(
    value: object,
    *,
    field: str,
    model: str,
    route: str,
    manifest_format: str,
    source_path: str,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    slot = _mapping(value, field=field)
    slot_model = _required_text(slot.get("model"), field=f"{field}.model")
    if slot_model != model:
        raise ProductionModelOutputError(f"{field}.model does not match its model key")
    slot_window_id = _required_text(slot.get("window_id"), field=f"{field}.window_id")
    if slot_window_id != window["window_id"]:
        raise ProductionModelOutputError(f"{field}.window_id does not match its window")
    native_route = _required_text(slot.get("native_route"), field=f"{field}.native_route")
    if native_route != route:
        raise ProductionModelOutputError(f"{field}.native_route does not match manifest route")
    status = _required_text(slot.get("status"), field=f"{field}.status")
    if status not in _STATUS_SET:
        raise ProductionModelOutputError(f"{field}.status is not a supported model-output status")
    predictions = _array(slot.get("predictions"), field=f"{field}.predictions")
    predictions_copy = _json_copy(predictions, field=f"{field}.predictions")
    _assert_no_gold_fields(predictions_copy, field=f"{field}.predictions")
    if status == "NOT_RUN" and predictions_copy:
        raise ProductionModelOutputError(f"{field}.NOT_RUN slot cannot retain predictions")
    metrics = _validate_metrics(slot.get("metrics"), field=f"{field}.metrics")
    lineage = _validate_lineage(
        slot.get("artifact_lineage"),
        field=f"{field}.artifact_lineage",
        manifest_format=manifest_format,
        source_path=source_path,
        window=window,
    )
    # Retain optional model/runtime metadata without prescribing a provider
    # wire shape.  Gold guards still apply to every optional value.
    optional: dict[str, Any] = {}
    for key, child in slot.items():
        if key in {
            "model",
            "window_id",
            "native_route",
            "status",
            "predictions",
            "metrics",
            "artifact_lineage",
        }:
            continue
        # Include the optional key itself in the recursive guard.  Checking
        # only its value would miss ``{"gold": {...}}`` because ``gold`` is
        # the container key rather than a nested value.
        _assert_no_gold_fields({key: child}, field=field)
        optional[key] = _json_copy(child, field=f"{field}.{key}")
    return {
        "model": model,
        "window_id": window["window_id"],
        "native_route": route,
        "status": status,
        "predictions": predictions_copy,
        "metrics": metrics,
        "artifact_lineage": lineage,
        **optional,
    }


def validate_model_output_sidecar(value: object) -> dict[str, Any]:
    """Validate and deep-copy a model-output sidecar.

    Validation is intentionally independent of any model runtime.  It checks
    route/window binding, status/metric invariants, artifact lineage, and the
    gold boundary; it does not score predictions or infer labels.
    """

    payload = _mapping(value, field="sidecar")
    format_value = _required_text(payload.get("format"), field="sidecar.format")
    if format_value != PRODUCTION_MODEL_OUTPUT_SIDECAR_VERSION:
        raise ProductionModelOutputError("sidecar.format is not supported")
    if payload.get("authority") != LOCAL_NONPRODUCTION_AUTHORITY:
        raise ProductionModelOutputError("sidecar authority must be LOCAL_NONPRODUCTION_ONLY")
    manifest_format = _required_text(
        payload.get("source_manifest_format"), field="sidecar.source_manifest_format"
    )
    if manifest_format != PRODUCTION_COHORT_MANIFEST_FORMAT:
        raise ProductionModelOutputError("sidecar source_manifest_format is not supported")
    source_path, source = _source_projection(payload)

    routes_raw = _mapping(payload.get("model_routes"), field="sidecar.model_routes")
    if set(routes_raw) != set(MODEL_NAMES):
        raise ProductionModelOutputError(
            "sidecar.model_routes must contain exactly WeMM, Qwen, and Mage"
        )
    routes = {
        model: _required_text(routes_raw.get(model), field=f"sidecar.model_routes.{model}")
        for model in MODEL_NAMES
    }

    raw_windows = _array(payload.get("windows"), field="sidecar.windows")
    if not raw_windows:
        raise ProductionModelOutputError("sidecar.windows must be non-empty")
    windows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_window in enumerate(raw_windows):
        window = _mapping(raw_window, field=f"sidecar.windows[{index}]")
        window_id = _required_text(
            window.get("window_id"), field=f"sidecar.windows[{index}].window_id"
        )
        if window_id in seen_ids:
            raise ProductionModelOutputError(f"duplicate sidecar window_id: {window_id}")
        seen_ids.add(window_id)
        ordinal_value = window.get("ordinal", index)
        if (
            isinstance(ordinal_value, bool)
            or not isinstance(ordinal_value, int)
            or ordinal_value < 0
        ):
            raise ProductionModelOutputError(
                f"sidecar.windows[{index}].ordinal must be a non-negative integer"
            )
        start = _finite_nonnegative(
            window.get("start_seconds"), field=f"sidecar.windows[{index}].start_seconds"
        )
        end = _finite_nonnegative(
            window.get("end_seconds"), field=f"sidecar.windows[{index}].end_seconds"
        )
        if end <= start:
            raise ProductionModelOutputError(
                f"sidecar.windows[{index}] end_seconds must be greater than start_seconds"
            )
        if "gold" in window:
            raise ProductionModelOutputError("sidecar windows must not contain gold data")
        camera_ids: list[str] | None = None
        if "camera_ids" in window:
            raw_camera_ids = _array(
                window.get("camera_ids"), field=f"sidecar.windows[{index}].camera_ids"
            )
            camera_ids = [
                _required_text(
                    camera_id,
                    field=f"sidecar.windows[{index}].camera_ids[{camera_index}]",
                )
                for camera_index, camera_id in enumerate(raw_camera_ids)
            ]

        binding_window = {
            "ordinal": ordinal_value,
            "window_id": window_id,
            "start_seconds": start,
            "end_seconds": end,
        }
        outputs_raw = _mapping(
            window.get("model_outputs"), field=f"sidecar.windows[{index}].model_outputs"
        )
        if set(outputs_raw) != set(MODEL_NAMES):
            raise ProductionModelOutputError(
                f"sidecar.windows[{index}].model_outputs must contain exactly WeMM, Qwen, and Mage"
            )
        outputs = {
            model: _validate_slot(
                outputs_raw.get(model),
                field=f"sidecar.windows[{index}].model_outputs.{model}",
                model=model,
                route=routes[model],
                manifest_format=manifest_format,
                source_path=source_path,
                window=binding_window,
            )
            for model in MODEL_NAMES
        }
        known_window_keys = {
            "ordinal",
            "window_id",
            "start_seconds",
            "end_seconds",
            "camera_ids",
            "model_outputs",
        }
        window_extras: dict[str, Any] = {}
        for key, child in window.items():
            if key in known_window_keys:
                continue
            # A sidecar window may carry harmless benchmark metadata, but a
            # key such as ``gold_status`` or ``review`` must fail closed.
            _assert_no_gold_fields({key: child}, field=f"sidecar.windows[{index}]")
            window_extras[key] = _json_copy(child, field=f"sidecar.windows[{index}].{key}")
        windows.append(
            {
                "ordinal": ordinal_value,
                "window_id": window_id,
                "start_seconds": start,
                "end_seconds": end,
                **({"camera_ids": camera_ids} if camera_ids is not None else {}),
                "model_outputs": outputs,
                **window_extras,
            }
        )

    contract_raw = _mapping(payload.get("contract"), field="sidecar.contract")
    model_names = _array(contract_raw.get("model_names"), field="sidecar.contract.model_names")
    if tuple(model_names) != MODEL_NAMES:
        raise ProductionModelOutputError("sidecar.contract.model_names must be WeMM, Qwen, Mage")
    for key in (
        "predictions_are_model_claims",
        "gold_is_external",
        "gold_fields_included",
        "model_outputs_are_not_gold",
    ):
        if not isinstance(contract_raw.get(key), bool):
            raise ProductionModelOutputError(f"sidecar.contract.{key} must be boolean")
    if (
        contract_raw["gold_is_external"] is not True
        or contract_raw["gold_fields_included"] is not False
    ):
        raise ProductionModelOutputError("sidecar contract must keep gold external and excluded")
    if contract_raw["model_outputs_are_not_gold"] is not True:
        raise ProductionModelOutputError("sidecar contract must mark model outputs as non-gold")
    for key, child in contract_raw.items():
        if key in {
            "model_names",
            "predictions_are_model_claims",
            "gold_is_external",
            "gold_fields_included",
            "model_outputs_are_not_gold",
        }:
            continue
        _assert_no_gold_fields({key: child}, field="sidecar.contract")

    controls = _mapping(payload.get("controls"), field="sidecar.controls")
    required_boolean_controls = (
        "model_invoked",
        "gpu_invoked",
        "gold_included",
        "predictions_copied_to_gold",
        "ontology_modified",
        "mapper_modified",
        "training_invoked",
        "frames_decoded",
    )
    for key in required_boolean_controls:
        if not isinstance(controls.get(key), bool):
            raise ProductionModelOutputError(f"sidecar.controls.{key} must be boolean")
    # Gold/authority boundaries are immutable even after a later local run;
    # execution observations may legitimately move from false to true.
    for key in (
        "gold_included",
        "predictions_copied_to_gold",
        "ontology_modified",
        "mapper_modified",
        "training_invoked",
    ):
        if controls[key] is not False:
            raise ProductionModelOutputError(f"sidecar.controls.{key} must remain false")
    if (
        any(
            window["model_outputs"][model]["status"] != "NOT_RUN"
            for window in windows
            for model in MODEL_NAMES
        )
        and controls["model_invoked"] is not True
    ):
        raise ProductionModelOutputError(
            "sidecar.controls.model_invoked must be true when a slot is not NOT_RUN"
        )
    for key, child in controls.items():
        if key not in required_boolean_controls:
            _assert_no_gold_fields({key: child}, field="sidecar.controls")
    _assert_no_gold_fields(source, field="sidecar.source")

    # Keep unknown top-level metadata only when it is JSON-safe and does not
    # smuggle gold into the sidecar.  Contract/control fields are checked above.
    known = {
        "format",
        "authority",
        "source_manifest_format",
        "source",
        "model_routes",
        "windows",
        "contract",
        "controls",
    }
    extras: dict[str, Any] = {}
    for key, child in payload.items():
        if key in known:
            continue
        _assert_no_gold_fields({key: child}, field="sidecar")
        extras[key] = _json_copy(child, field=f"sidecar.{key}")

    result = {
        "format": PRODUCTION_MODEL_OUTPUT_SIDECAR_VERSION,
        "authority": LOCAL_NONPRODUCTION_AUTHORITY,
        "source_manifest_format": manifest_format,
        "source": source,
        "model_routes": routes,
        "windows": windows,
        "contract": {
            "model_names": list(MODEL_NAMES),
            "predictions_are_model_claims": bool(contract_raw["predictions_are_model_claims"]),
            "gold_is_external": True,
            "gold_fields_included": False,
            "model_outputs_are_not_gold": True,
        },
        "controls": {key: bool(controls[key]) for key in required_boolean_controls},
        **extras,
    }
    return copy.deepcopy(result)


def update_model_output_slot(
    sidecar: Mapping[str, Any],
    *,
    window_id: str,
    model: str,
    status: str,
    predictions: Sequence[Any] = (),
    metrics: Mapping[str, Any] | None = None,
    artifact_lineage: Mapping[str, Any] | None = None,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a validated copy with one model/window slot replaced.

    This helper is intentionally narrow: the model and native route remain
    bound to the initialized slot, and supplied predictions/artifacts are
    checked by the same label-blind validator.  It is useful for a later local
    runner but does not invoke a model itself.  When a non-``NOT_RUN`` status
    is supplied, ``controls.model_invoked`` is set automatically unless an
    explicit controls mapping is provided.
    """

    payload = validate_model_output_sidecar(sidecar)
    target_window_id = _required_text(window_id, field="window_id")
    target_model = _required_text(model, field="model")
    if target_model not in MODEL_NAMES:
        raise ProductionModelOutputError(f"unsupported model: {target_model}")
    if status not in _STATUS_SET:
        raise ProductionModelOutputError(f"unsupported model-output status: {status}")
    prediction_values = _array(predictions, field="predictions")
    replacement_metrics = _json_copy(metrics, field="metrics") if metrics is not None else None
    replacement_lineage = (
        _json_copy(artifact_lineage, field="artifact_lineage")
        if artifact_lineage is not None
        else None
    )
    found = False
    for window in payload["windows"]:
        if window["window_id"] != target_window_id:
            continue
        found = True
        slot = window["model_outputs"][target_model]
        slot["status"] = status
        slot["predictions"] = list(prediction_values)
        if replacement_metrics is not None:
            slot["metrics"] = replacement_metrics
        if replacement_lineage is not None:
            slot["artifact_lineage"] = replacement_lineage
        break
    if not found:
        raise ProductionModelOutputError(f"unknown window_id: {target_window_id}")
    if controls is not None:
        controls_copy = _mapping(controls, field="controls")
        for key, child in controls_copy.items():
            if key not in payload["controls"]:
                raise ProductionModelOutputError(f"unsupported control: {key}")
            if not isinstance(child, bool):
                raise ProductionModelOutputError(f"controls.{key} must be boolean")
            payload["controls"][key] = child
    elif status != "NOT_RUN":
        # A terminal/running slot is evidence that at least one model was
        # invoked.  Callers can still explicitly set GPU/frame controls when
        # those observations are available.
        payload["controls"]["model_invoked"] = True
    return validate_model_output_sidecar(payload)


# Descriptive aliases make the builder easy to discover without creating a
# second contract implementation.
build_production_model_output_sidecar = build_model_output_sidecar
initialize_model_output_sidecar = build_model_output_sidecar
initialize_production_model_output_sidecar = build_model_output_sidecar
validate_production_model_output_sidecar = validate_model_output_sidecar
set_model_output_slot = update_model_output_slot


__all__ = [
    "DEFAULT_MODEL_ROUTES",
    "DEFAULT_NATIVE_MODEL_ROUTES",
    "LOCAL_NONPRODUCTION_AUTHORITY",
    "MODEL_NAMES",
    "MODEL_OUTPUT_STATUSES",
    "NATIVE_MODEL_ROUTES",
    "PRODUCTION_COHORT_MANIFEST_FORMAT",
    "PRODUCTION_MODEL_NAMES",
    "PRODUCTION_MODEL_OUTPUT_FORMAT",
    "PRODUCTION_MODEL_OUTPUT_SIDECAR_VERSION",
    "ProductionModelOutputContractError",
    "ProductionModelOutputError",
    "build_model_output_sidecar",
    "build_production_model_output_sidecar",
    "initialize_model_output_sidecar",
    "initialize_production_model_output_sidecar",
    "set_model_output_slot",
    "update_model_output_slot",
    "validate_model_output_sidecar",
    "validate_production_model_output_sidecar",
]
