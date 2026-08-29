"""Build a read-only WeMM frame/resolution ablation plan.

The production-shaped six-camera sample has a bounded ten-window cohort.  This
module describes a small, reproducible matrix for that cohort without opening
media, decoding frames, loading model weights, or deriving an identity.  The
matrix deliberately records the processor grid expected for each arm: the
video processor applies a *total* pixel budget, so changing the frame count can
also change the spatial grid.

The returned object is a planning artifact only.  It does not contain model
predictions and does not make an official quality claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

WEMM_FRAME_GRID_MATRIX_VERSION: Final = "robata-production-wemm-frame-grid-matrix-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
PLAN_STATUS: Final = "PLAN_ONLY"

CAMERA_IDS: Final = tuple(f"cam_{index:02d}" for index in range(1, 7))
EXPECTED_CAMERA_COUNT: Final = 6
EXPECTED_WINDOW_COUNT: Final = 10
EXPECTED_WINDOW_SECONDS: Final = 4.0
EXPECTED_REPRESENTED_DURATION_SECONDS: Final = 40.0
EXPECTED_COMMON_DURATION_SECONDS: Final = 40.833423
EXPECTED_EXCLUDED_TAIL_SECONDS: Final = 0.833423
EXPECTED_CAMERA_FRAME_COUNT: Final = 1226

WEMM_MODEL_IDENTIFIER: Final = "WeMM-Embedding-2B"
WEMM_DIMENSION: Final = 2048
BASELINE_FRAME_COUNT: Final = 4
CURRENT_TOTAL_PIXEL_BUDGET: Final = 262_144
HIGHER_TOTAL_PIXEL_BUDGET: Final = 524_288

BASELINE_GRID: Final = (2, 14, 16)
CURRENT_8_FRAME_GRID: Final = (4, 10, 12)
HIGHER_4_FRAME_GRID: Final = (2, 20, 24)
HIGHER_8_FRAME_GRID: Final = (4, 14, 16)


class WemmFrameGridMatrixError(ValueError):
    """Raised when a cohort or baseline cannot support this matrix."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WemmFrameGridMatrixError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WemmFrameGridMatrixError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WemmFrameGridMatrixError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WemmFrameGridMatrixError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise WemmFrameGridMatrixError(f"{field} must be >= {minimum}")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WemmFrameGridMatrixError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise WemmFrameGridMatrixError(f"{field} must be finite")
    return result


def _close(actual: object, expected: float, *, field: str, tolerance: float = 1e-4) -> float:
    result = _number(actual, field=field)
    if abs(result - expected) > tolerance:
        raise WemmFrameGridMatrixError(
            f"{field} must be approximately {expected:g}; got {result:g}"
        )
    return result


def _normalised_path(value: object, *, field: str) -> str:
    # This is only a comparison-friendly spelling.  It intentionally does not
    # touch the filesystem (and therefore cannot open an MCAP as a side effect).
    return _text(value, field=field).replace("\\", "/").casefold()


def _assert_false(parent: Mapping[str, Any], key: str, *, field: str) -> None:
    if key in parent and parent[key] is not False:
        raise WemmFrameGridMatrixError(f"{field}.{key} must be false for a plan-only input")


def _parse_grid(value: object, *, field: str) -> tuple[int, int, int]:
    """Parse the processor's common ``[[t, h, w]]`` representation."""

    outer = _sequence(value, field=field)
    if (
        len(outer) == 1
        and isinstance(outer[0], Sequence)
        and not isinstance(outer[0], (str, bytes, bytearray))
    ):
        row = outer[0]
    else:
        row = outer
    if len(row) != 3:
        raise WemmFrameGridMatrixError(f"{field} must contain a [t, h, w] triple")
    parsed = tuple(
        _integer(item, field=f"{field}[{index}]", minimum=1) for index, item in enumerate(row)
    )
    return parsed  # type: ignore[return-value]


def _window_rows(value: object, *, field: str) -> tuple[Mapping[str, Any], ...]:
    rows = _sequence(value, field=field)
    result: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        result.append(_mapping(row, field=f"{field}[{index}]"))
    return tuple(result)


def _validate_cohort(cohort: Mapping[str, Any]) -> dict[str, Any]:
    if cohort.get("format") != "robata-production-shaped-cohort-v1":
        raise WemmFrameGridMatrixError("cohort.format is not the production-shaped cohort format")
    if cohort.get("authority") != AUTHORITY:
        raise WemmFrameGridMatrixError("cohort.authority must be LOCAL_NONPRODUCTION_ONLY")

    source = _mapping(cohort.get("source"), field="cohort.source")
    source_path = _text(source.get("path"), field="cohort.source.path")
    if source.get("media_type") not in (None, "application/x-mcap"):
        raise WemmFrameGridMatrixError("cohort.source.media_type must be application/x-mcap")
    if _integer(source.get("camera_count"), field="cohort.source.camera_count", minimum=0) != (
        EXPECTED_CAMERA_COUNT
    ):
        raise WemmFrameGridMatrixError("cohort must contain exactly six cameras")

    cameras = _sequence(source.get("cameras"), field="cohort.source.cameras")
    if len(cameras) != EXPECTED_CAMERA_COUNT:
        raise WemmFrameGridMatrixError("cohort.source.cameras must contain six rows")
    camera_ids: list[str] = []
    for index, raw in enumerate(cameras):
        row = _mapping(raw, field=f"cohort.source.cameras[{index}]")
        camera_id = _text(row.get("camera_id"), field=f"cohort.source.cameras[{index}].camera_id")
        camera_ids.append(camera_id)
        if camera_id != CAMERA_IDS[index]:
            raise WemmFrameGridMatrixError(
                f"cohort camera order must be {list(CAMERA_IDS)}; got {camera_ids}"
            )
        frame_count = _integer(
            row.get("frame_count"),
            field=f"cohort.source.cameras[{index}].frame_count",
            minimum=0,
        )
        if frame_count != EXPECTED_CAMERA_FRAME_COUNT:
            raise WemmFrameGridMatrixError(
                f"{camera_id} must contain {EXPECTED_CAMERA_FRAME_COUNT} frames; got {frame_count}"
            )
        _close(
            row.get("duration_seconds"),
            EXPECTED_COMMON_DURATION_SECONDS,
            field=f"cohort.source.cameras[{index}].duration_seconds",
            tolerance=1e-3,
        )

    common_duration = _close(
        source.get("common_duration_seconds"),
        EXPECTED_COMMON_DURATION_SECONDS,
        field="cohort.source.common_duration_seconds",
        tolerance=1e-4,
    )
    policy = _mapping(cohort.get("window_policy"), field="cohort.window_policy")
    window_seconds = _close(
        policy.get("window_seconds"),
        EXPECTED_WINDOW_SECONDS,
        field="cohort.window_policy.window_seconds",
    )
    if policy.get("include_tail") is not False:
        raise WemmFrameGridMatrixError("cohort.window_policy.include_tail must be false")
    represented = _close(
        policy.get("represented_duration_seconds"),
        EXPECTED_REPRESENTED_DURATION_SECONDS,
        field="cohort.window_policy.represented_duration_seconds",
    )
    excluded_tail = _close(
        policy.get("excluded_tail_seconds"),
        EXPECTED_EXCLUDED_TAIL_SECONDS,
        field="cohort.window_policy.excluded_tail_seconds",
        tolerance=1e-4,
    )

    windows = _window_rows(cohort.get("windows"), field="cohort.windows")
    if len(windows) != EXPECTED_WINDOW_COUNT:
        raise WemmFrameGridMatrixError("cohort must contain exactly ten windows")
    window_ids: list[str] = []
    slots: list[dict[str, Any]] = []
    for index, row in enumerate(windows):
        ordinal = _integer(row.get("ordinal"), field=f"cohort.windows[{index}].ordinal", minimum=0)
        if ordinal != index:
            raise WemmFrameGridMatrixError("cohort window ordinals must be contiguous 0..9")
        window_id = _text(row.get("window_id"), field=f"cohort.windows[{index}].window_id")
        if window_id in window_ids:
            raise WemmFrameGridMatrixError(f"duplicate cohort window_id: {window_id}")
        window_ids.append(window_id)
        start = _close(
            row.get("start_seconds"),
            index * EXPECTED_WINDOW_SECONDS,
            field=f"cohort.windows[{index}].start_seconds",
        )
        end = _close(
            row.get("end_seconds"),
            (index + 1) * EXPECTED_WINDOW_SECONDS,
            field=f"cohort.windows[{index}].end_seconds",
        )
        duration = _close(
            row.get("duration_seconds"),
            EXPECTED_WINDOW_SECONDS,
            field=f"cohort.windows[{index}].duration_seconds",
        )
        listed_cameras = tuple(
            _text(item, field=f"cohort.windows[{index}].camera_ids[]")
            for item in _sequence(
                row.get("camera_ids"), field=f"cohort.windows[{index}].camera_ids"
            )
        )
        if listed_cameras != CAMERA_IDS:
            raise WemmFrameGridMatrixError(
                f"cohort.windows[{index}].camera_ids must list all six cameras in canonical order"
            )
        for camera_id in CAMERA_IDS:
            slots.append(
                {
                    "window_ordinal": ordinal,
                    "window_id": window_id,
                    "camera_id": camera_id,
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": duration,
                }
            )

    controls = _mapping(cohort.get("controls"), field="cohort.controls")
    for key in (
        "ontology_modified",
        "mapper_modified",
        "training_invoked",
        "heldout_100_opened",
        "sha_or_digest_computed",
        "frames_decoded",
    ):
        _assert_false(controls, key, field="cohort.controls")

    return {
        "source_path": source_path,
        "media_type": source.get("media_type", "application/x-mcap"),
        "camera_ids": list(CAMERA_IDS),
        "camera_count": EXPECTED_CAMERA_COUNT,
        "camera_frame_count": EXPECTED_CAMERA_FRAME_COUNT,
        "window_ids": window_ids,
        "window_count": EXPECTED_WINDOW_COUNT,
        "window_seconds": window_seconds,
        "represented_duration_seconds": represented,
        "common_duration_seconds": common_duration,
        "excluded_tail_seconds": excluded_tail,
        "slot_count": len(slots),
        "slots": slots,
    }


def _validate_baseline(
    baseline: Mapping[str, Any],
    *,
    cohort_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline.get("format") != "robata-production-wemm-vocabulary-shadow-v1":
        raise WemmFrameGridMatrixError("baseline.format is not a WeMM vocabulary shadow report")
    if baseline.get("authority") != AUTHORITY:
        raise WemmFrameGridMatrixError("baseline.authority must be LOCAL_NONPRODUCTION_ONLY")
    if baseline.get("status") != "SUCCEEDED":
        raise WemmFrameGridMatrixError("baseline.status must be SUCCEEDED")
    if baseline.get("official_quality_status") != "NOT_MEASURED":
        raise WemmFrameGridMatrixError("baseline official quality must remain NOT_MEASURED")
    if baseline.get("official_gold_status") != "NOT_ESTABLISHED":
        raise WemmFrameGridMatrixError("baseline official gold must remain NOT_ESTABLISHED")
    if baseline.get("quality_claim") is not False:
        raise WemmFrameGridMatrixError("baseline.quality_claim must be false")
    if baseline.get("production_eligible") is not False:
        raise WemmFrameGridMatrixError("baseline.production_eligible must be false")

    source = _mapping(baseline.get("source"), field="baseline.source")
    if _normalised_path(source.get("path"), field="baseline.source.path") != _normalised_path(
        cohort_summary["source_path"], field="cohort.source.path"
    ):
        raise WemmFrameGridMatrixError("baseline and cohort source paths do not match")
    if (
        _integer(source.get("window_count"), field="baseline.source.window_count", minimum=0)
        != (cohort_summary["window_count"])
    ):
        raise WemmFrameGridMatrixError("baseline window_count does not match cohort")
    if (
        _integer(source.get("camera_count"), field="baseline.source.camera_count", minimum=0)
        != (cohort_summary["camera_count"])
    ):
        raise WemmFrameGridMatrixError("baseline camera_count does not match cohort")

    model = _mapping(baseline.get("model"), field="baseline.model")
    if model.get("identifier") != WEMM_MODEL_IDENTIFIER:
        raise WemmFrameGridMatrixError("baseline model must be WeMM-Embedding-2B")
    if (
        _integer(model.get("dimension"), field="baseline.model.dimension", minimum=1)
        != WEMM_DIMENSION
    ):
        raise WemmFrameGridMatrixError("baseline embedding dimension must be 2048")
    if (
        _integer(model.get("frame_count"), field="baseline.model.frame_count", minimum=1)
        != BASELINE_FRAME_COUNT
    ):
        raise WemmFrameGridMatrixError("baseline frame_count must be four")
    if model.get("label_variant") != "canonical":
        raise WemmFrameGridMatrixError("baseline label_variant must be canonical")

    vocabulary = _mapping(baseline.get("vocabulary"), field="baseline.vocabulary")
    if vocabulary.get("owner_approved") is not True:
        raise WemmFrameGridMatrixError("baseline vocabulary must be owner approved")
    if vocabulary.get("production_eligible") is not False:
        raise WemmFrameGridMatrixError("baseline vocabulary must be non-production")
    for key in ("epic_ontology_used", "mapper_used"):
        if vocabulary.get(key) is not False:
            raise WemmFrameGridMatrixError(f"baseline.vocabulary.{key} must be false")

    controls = _mapping(baseline.get("controls"), field="baseline.controls")
    if controls.get("model_invoked") is not True:
        raise WemmFrameGridMatrixError("baseline.controls.model_invoked must be true")
    for key in (
        "gold_included",
        "predictions_are_gold",
        "existing_mapper_invoked",
        "ontology_modified",
        "mapper_modified",
        "training_invoked",
        "heldout_100_opened",
        "hash_or_sha_used",
        "ground_truth_used_in_encoder_input",
    ):
        _assert_false(controls, key, field="baseline.controls")

    expected_window_ids = tuple(cohort_summary["window_ids"])
    windows = _window_rows(baseline.get("windows"), field="baseline.windows")
    if len(windows) != len(expected_window_ids):
        raise WemmFrameGridMatrixError("baseline windows do not cover the cohort")

    nested_observation_count = 0
    nested_grids: list[tuple[int, int, int]] = []
    for index, row in enumerate(windows):
        ordinal = _integer(
            row.get("ordinal"), field=f"baseline.windows[{index}].ordinal", minimum=0
        )
        if ordinal != index:
            raise WemmFrameGridMatrixError("baseline window ordinals must be contiguous 0..9")
        window_id = _text(row.get("window_id"), field=f"baseline.windows[{index}].window_id")
        if window_id != expected_window_ids[index]:
            raise WemmFrameGridMatrixError(
                f"baseline window {index} does not match cohort window_id"
            )
        _close(
            row.get("start_seconds"),
            index * EXPECTED_WINDOW_SECONDS,
            field=f"baseline.windows[{index}].start_seconds",
        )
        _close(
            row.get("end_seconds"),
            (index + 1) * EXPECTED_WINDOW_SECONDS,
            field=f"baseline.windows[{index}].end_seconds",
        )
        model_row = _mapping(row.get("model"), field=f"baseline.windows[{index}].model")
        if model_row.get("status") != "SUCCEEDED":
            raise WemmFrameGridMatrixError(
                f"baseline.windows[{index}].model.status must be SUCCEEDED"
            )
        observations = model_row.get("input_observations")
        if observations is None:
            continue
        observation_rows = _sequence(
            observations, field=f"baseline.windows[{index}].model.input_observations"
        )
        if len(observation_rows) != EXPECTED_CAMERA_COUNT:
            raise WemmFrameGridMatrixError(
                f"baseline window {index} must contain six camera observations"
            )
        seen_cameras: set[str] = set()
        for obs_index, raw_obs in enumerate(observation_rows):
            obs = _mapping(
                raw_obs, field=f"baseline.windows[{index}].model.input_observations[{obs_index}]"
            )
            camera_id = _text(obs.get("camera_id"), field="baseline input observation camera_id")
            if camera_id not in CAMERA_IDS or camera_id in seen_cameras:
                raise WemmFrameGridMatrixError(
                    f"baseline window {index} has invalid/duplicate camera observations"
                )
            seen_cameras.add(camera_id)
            if obs.get("window_id") is not None and obs.get("window_id") != window_id:
                raise WemmFrameGridMatrixError(
                    "baseline input observation window_id does not match"
                )
            if (
                obs.get("frame_count") is not None
                and _integer(
                    obs.get("frame_count"),
                    field="baseline input observation.frame_count",
                    minimum=1,
                )
                != BASELINE_FRAME_COUNT
            ):
                raise WemmFrameGridMatrixError("baseline input observations must use four frames")
            model_observation = _mapping(
                obs.get("model_observation"), field="baseline input observation.model_observation"
            )
            if (
                model_observation.get("frame_count") is not None
                and _integer(
                    model_observation.get("frame_count"),
                    field="baseline model_observation.frame_count",
                    minimum=1,
                )
                != BASELINE_FRAME_COUNT
            ):
                raise WemmFrameGridMatrixError("baseline model observations must use four frames")
            grid_value = model_observation.get("video_grid_thw")
            if grid_value is not None:
                nested_grids.append(
                    _parse_grid(grid_value, field="baseline model_observation.video_grid_thw")
                )
            nested_observation_count += 1

    backend_rows = baseline.get("backend_observations")
    backend_video_count = 0
    backend_text_count = 0
    backend_grids: list[tuple[int, int, int]] = []
    if backend_rows is not None:
        for index, raw_row in enumerate(
            _sequence(backend_rows, field="baseline.backend_observations")
        ):
            row = _mapping(raw_row, field=f"baseline.backend_observations[{index}]")
            modality = row.get("modality")
            if modality == "text":
                backend_text_count += 1
                continue
            if modality != "video":
                continue
            backend_video_count += 1
            if (
                row.get("item_count") is not None
                and _integer(
                    row.get("item_count"), field="baseline video observation.item_count", minimum=1
                )
                != 1
            ):
                raise WemmFrameGridMatrixError("baseline video observations must have item_count=1")
            if (
                row.get("frame_count") is not None
                and _integer(
                    row.get("frame_count"),
                    field="baseline video observation.frame_count",
                    minimum=1,
                )
                != BASELINE_FRAME_COUNT
            ):
                raise WemmFrameGridMatrixError("baseline video observations must use four frames")
            for key in ("embedding_dimension", "requested_dimension"):
                if (
                    row.get(key) is not None
                    and _integer(row.get(key), field=f"baseline video observation.{key}", minimum=1)
                    != WEMM_DIMENSION
                ):
                    raise WemmFrameGridMatrixError(f"baseline video observation {key} must be 2048")
            grid_value = row.get("video_grid_thw")
            if grid_value is not None:
                backend_grids.append(
                    _parse_grid(grid_value, field="baseline video observation.video_grid_thw")
                )

    all_grids = [*nested_grids, *backend_grids]
    if not all_grids:
        raise WemmFrameGridMatrixError("baseline has no observed video_grid_thw metadata")
    if any(grid != BASELINE_GRID for grid in all_grids):
        observed = sorted({grid for grid in all_grids})
        raise WemmFrameGridMatrixError(
            f"baseline processor grid must be {list(BASELINE_GRID)} everywhere; observed {observed}"
        )
    expected_slots = int(cohort_summary["slot_count"])
    if nested_observation_count and nested_observation_count != expected_slots:
        raise WemmFrameGridMatrixError(
            "baseline input observations must cover all 60 camera-window slots"
        )
    if backend_video_count and backend_video_count != expected_slots:
        raise WemmFrameGridMatrixError("baseline backend video observations must contain 60 calls")
    if (
        nested_observation_count
        and backend_video_count
        and nested_observation_count != backend_video_count
    ):
        raise WemmFrameGridMatrixError("baseline nested and backend video call counts disagree")

    return {
        "artifact_format": str(baseline["format"]),
        "status": str(baseline["status"]),
        "model_identifier": str(model["identifier"]),
        "dimension": WEMM_DIMENSION,
        "frame_count": BASELINE_FRAME_COUNT,
        "label_variant": str(model["label_variant"]),
        "video_call_count": backend_video_count or nested_observation_count or expected_slots,
        "text_prototype_call_count": backend_text_count,
        "observed_video_grid_thw": [list(BASELINE_GRID)],
        "quality_status": str(baseline["official_quality_status"]),
        "gold_status": str(baseline["official_gold_status"]),
        "surrogate_quality_available": False,
    }


def _metric_contract() -> dict[str, Any]:
    return {
        "quality": {
            "official_status": "NOT_MEASURED",
            "surrogate_status": "OPTIONAL_REFERENCE_JOIN",
            "reference_join_required": True,
            "names": ["R@1", "R@5", "MRR", "Top1-Top2 margin"],
            "notes": (
                "R@1/R@5/MRR require an explicitly joined independent reference; "
                "predictions and owner-scoped vocabulary are not gold."
            ),
        },
        "cross_camera": {
            "names": ["camera agreement", "camera coverage"],
            "camera_count_expected": EXPECTED_CAMERA_COUNT,
            "per_window": True,
            "aggregate": True,
            "definition": (
                "camera agreement is the fraction of the six camera Top-1 labels "
                "equal to the fused window Top-1; coverage reports observed/expected cameras."
            ),
        },
        "resource": {
            "names": [
                "wall seconds",
                "wall seconds per camera-window",
                "processor seconds",
                "model seconds",
                "ranking/I-O seconds",
                "peak VRAM bytes",
                "peak allocated bytes",
            ],
            "telemetry_optional": ["peak_vram_bytes", "peak_allocated_bytes"],
            "phase_timers": ["processor_seconds", "model_seconds", "ranking_io_seconds"],
        },
    }


def _arm(
    *,
    arm_id: str,
    frame_count: int,
    pixel_budget: int,
    grid: tuple[int, int, int],
    grid_status: str,
    grid_source: str,
    slot_count: int,
) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "frame_count": frame_count,
        "total_pixel_budget": pixel_budget,
        "pixel_budget_cap": pixel_budget,
        "pixel_budget_unit": "pixels",
        "expected_video_grid_thw": [list(grid)],
        "expected_grid_thw": list(grid),
        "grid_expectation_status": grid_status,
        "grid_source": grid_source,
        "execution_status": "NOT_RUN",
        "planned_camera_count": EXPECTED_CAMERA_COUNT,
        "planned_window_count": EXPECTED_WINDOW_COUNT,
        "planned_camera_window_slot_count": slot_count,
        "planned_video_calls": slot_count,
        "planned_model_calls": slot_count,
        "planned_text_prototype_calls": 0,
        "optional_text_prototype_calls_if_cache_miss": 1,
        "text_prototype_policy": "reuse_baseline_embedding",
        "metrics_status": "NOT_MEASURED",
    }


def build_wemm_frame_grid_matrix(
    cohort: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the bounded cohort/baseline and return a four-arm plan."""

    cohort_summary = _validate_cohort(cohort)
    baseline_summary = _validate_baseline(baseline, cohort_summary=cohort_summary)
    slot_count = int(cohort_summary["slot_count"])

    arms = [
        _arm(
            arm_id="f4_current_budget",
            frame_count=4,
            pixel_budget=CURRENT_TOTAL_PIXEL_BUDGET,
            grid=BASELINE_GRID,
            grid_status="observed_baseline",
            grid_source="baseline.backend_observations_and_input_observations",
            slot_count=slot_count,
        ),
        _arm(
            arm_id="f8_current_budget",
            frame_count=8,
            pixel_budget=CURRENT_TOTAL_PIXEL_BUDGET,
            grid=CURRENT_8_FRAME_GRID,
            grid_status="probe_expected",
            grid_source="processor_only_probe_no_model_inference",
            slot_count=slot_count,
        ),
        _arm(
            arm_id="f4_higher_budget",
            frame_count=4,
            pixel_budget=HIGHER_TOTAL_PIXEL_BUDGET,
            grid=HIGHER_4_FRAME_GRID,
            grid_status="probe_expected",
            grid_source="processor_only_probe_no_model_inference",
            slot_count=slot_count,
        ),
        _arm(
            arm_id="f8_higher_budget",
            frame_count=8,
            pixel_budget=HIGHER_TOTAL_PIXEL_BUDGET,
            grid=HIGHER_8_FRAME_GRID,
            grid_status="probe_expected",
            grid_source="processor_only_probe_no_model_inference",
            slot_count=slot_count,
        ),
    ]
    arm_ids = [str(arm["arm_id"]) for arm in arms]

    controls = {
        "model_invoked": False,
        "wemm_invoked": False,
        "media_decoded": False,
        "frames_decoded": False,
        "full_37_mcap_opened": False,
        "heldout_100_opened": False,
        "ontology_modified": False,
        "mapper_modified": False,
        "training_invoked": False,
        "gold_read": False,
        "gold_written": False,
        "hash_or_digest_computed": False,
        "identity_computation": "none",
        "production_eligible": False,
    }
    matrix = {
        "matrix_id": "wemm_frame_grid_4x8_current_higher_v1",
        "factor_axes": {
            "frame_count": [4, 8],
            "total_pixel_budget": [CURRENT_TOTAL_PIXEL_BUDGET, HIGHER_TOTAL_PIXEL_BUDGET],
        },
        "arms": arms,
        "execution_order": arm_ids,
        "planned_camera_window_slot_count_per_arm": slot_count,
        "comparisons": [
            {
                "name": "frame_count_at_constant_current_budget",
                "arms": ["f4_current_budget", "f8_current_budget"],
                "held_constant": ["total_pixel_budget"],
                "grid_confound": True,
            },
            {
                "name": "frame_count_at_baseline_spatial_grid_expectation",
                "arms": ["f4_current_budget", "f8_higher_budget"],
                "held_constant": ["expected_video_grid_thw"],
                "budget_confound": True,
            },
            {
                "name": "resolution_at_four_frames",
                "arms": ["f4_current_budget", "f4_higher_budget"],
                "held_constant": ["frame_count"],
            },
            {
                "name": "resolution_at_eight_frames",
                "arms": ["f8_current_budget", "f8_higher_budget"],
                "held_constant": ["frame_count"],
            },
        ],
        "grid_interpretation": (
            "The processor applies a total pixel budget.  Current-budget 8-frame "
            "inputs therefore have a different expected spatial grid than the 4-frame "
            "baseline; non-baseline grids remain probe_expected until observed."
        ),
    }

    return {
        "format": WEMM_FRAME_GRID_MATRIX_VERSION,
        "authority": AUTHORITY,
        "status": PLAN_STATUS,
        "purpose": "bounded WeMM 4/8-frame by current/higher total-pixel-budget comparison",
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "production_eligible": False,
        "model_invoked": False,
        "media_decoded": False,
        "full_37_mcap_opened": False,
        "hash_or_digest_computed": False,
        "inputs": {
            "cohort_manifest_format": "robata-production-shaped-cohort-v1",
            "cohort_source_path": cohort_summary["source_path"],
            "baseline_report_format": baseline_summary["artifact_format"],
            "baseline_model": baseline_summary["model_identifier"],
        },
        "cohort": cohort_summary,
        "baseline": baseline_summary,
        "matrix": matrix,
        "metrics": _metric_contract(),
        "controls": controls,
        "limitations": [
            "This is a PLAN_ONLY artifact; no model call or media decode was performed.",
            (
                "The ten 4-second windows represent 40.0 seconds; the "
                "0.833423-second tail is excluded."
            ),
            "Independent human gold is not established, so official quality remains NOT_MEASURED.",
            (
                "Non-baseline processor grids are expectations from a processor-only probe, "
                "not measured model output."
            ),
            (
                "The plan is scoped to the six-camera sample cohort and does not open the "
                "full 37-MCAP archive."
            ),
        ],
    }


# Short alias useful to callers that use the noun in the CLI name.
build_frame_grid_matrix = build_wemm_frame_grid_matrix


def render_markdown(plan: Mapping[str, Any]) -> str:
    """Render a compact review note without changing the JSON plan."""

    matrix = _mapping(plan.get("matrix"), field="plan.matrix")
    arms = _sequence(matrix.get("arms"), field="plan.matrix.arms")
    lines = [
        "# WeMM frame/grid matrix",
        "",
        "> **PLAN_ONLY / LOCAL_NONPRODUCTION_ONLY.** No model or media was opened.",
        "",
        f"- Cohort slots per arm: `{plan.get('cohort', {}).get('slot_count', 0)}`",
        f"- Official quality: `{plan.get('official_quality_status', 'NOT_MEASURED')}`",
        "",
        "| Arm | Frames | Pixel budget | Expected grid | Grid status | Slots |",
        "|---|---:|---:|---|---|---:|",
    ]
    for raw_arm in arms:
        if not isinstance(raw_arm, Mapping):
            continue
        lines.append(
            f"| `{raw_arm.get('arm_id', '')}` | {raw_arm.get('frame_count', '')} | "
            f"{raw_arm.get('total_pixel_budget', '')} | `{raw_arm.get('expected_grid_thw', '')}` | "
            f"{raw_arm.get('grid_expectation_status', '')} | "
            f"{raw_arm.get('planned_camera_window_slot_count', '')} |"
        )
    lines.extend(
        [
            "",
            (
                "R@1/R@5/MRR require an explicit independent-reference join; margins and "
                "camera agreement are diagnostics."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AUTHORITY",
    "BASELINE_FRAME_COUNT",
    "BASELINE_GRID",
    "CAMERA_IDS",
    "CURRENT_8_FRAME_GRID",
    "CURRENT_TOTAL_PIXEL_BUDGET",
    "EXPECTED_CAMERA_COUNT",
    "EXPECTED_CAMERA_FRAME_COUNT",
    "EXPECTED_COMMON_DURATION_SECONDS",
    "EXPECTED_EXCLUDED_TAIL_SECONDS",
    "EXPECTED_REPRESENTED_DURATION_SECONDS",
    "EXPECTED_WINDOW_COUNT",
    "EXPECTED_WINDOW_SECONDS",
    "HIGHER_4_FRAME_GRID",
    "HIGHER_8_FRAME_GRID",
    "HIGHER_TOTAL_PIXEL_BUDGET",
    "PLAN_STATUS",
    "WEMM_DIMENSION",
    "WEMM_FRAME_GRID_MATRIX_VERSION",
    "WEMM_MODEL_IDENTIFIER",
    "WemmFrameGridMatrixError",
    "build_frame_grid_matrix",
    "build_wemm_frame_grid_matrix",
    "render_markdown",
]
