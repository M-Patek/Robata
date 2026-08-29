from __future__ import annotations

import pytest

from robata.benchmark.wemm_frame_grid_matrix import (
    BASELINE_GRID,
    CAMERA_IDS,
    CURRENT_8_FRAME_GRID,
    CURRENT_TOTAL_PIXEL_BUDGET,
    HIGHER_4_FRAME_GRID,
    HIGHER_8_FRAME_GRID,
    HIGHER_TOTAL_PIXEL_BUDGET,
    WemmFrameGridMatrixError,
    build_wemm_frame_grid_matrix,
)


def _cohort() -> dict[str, object]:
    cameras = [
        {
            "camera_id": camera_id,
            "topic": f"/robot0/sensor/camera{index}/compressed",
            "frame_count": 1226,
            "duration_seconds": 40.833423,
        }
        for index, camera_id in enumerate(CAMERA_IDS)
    ]
    windows = []
    for ordinal in range(10):
        windows.append(
            {
                "ordinal": ordinal,
                "window_id": f"sample-medium-w{ordinal:02d}",
                "start_seconds": ordinal * 4.0,
                "end_seconds": (ordinal + 1) * 4.0,
                "duration_seconds": 4.0,
                "camera_ids": list(CAMERA_IDS),
            }
        )
    return {
        "format": "robata-production-shaped-cohort-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {
            "path": "data/source/sample-medium.mcap",
            "media_type": "application/x-mcap",
            "camera_count": 6,
            "cameras": cameras,
            "common_duration_seconds": 40.833423,
        },
        "window_policy": {
            "window_seconds": 4.0,
            "include_tail": False,
            "represented_duration_seconds": 40.0,
            "excluded_tail_seconds": 0.833423,
        },
        "windows": windows,
        "controls": {
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "sha_or_digest_computed": False,
            "frames_decoded": False,
        },
    }


def _baseline() -> dict[str, object]:
    windows = []
    backend_observations = [
        {
            "modality": "text",
            "item_count": 6,
            "embedding_dimension": 2048,
            "requested_dimension": 2048,
            "video_grid_thw": [],
        }
    ]
    for ordinal in range(10):
        window_id = f"sample-medium-w{ordinal:02d}"
        observations = []
        for camera_id in CAMERA_IDS:
            model_observation = {
                "modality": "video",
                "item_count": 1,
                "frame_count": 4,
                "embedding_dimension": 2048,
                "requested_dimension": 2048,
                "video_grid_thw": [list(BASELINE_GRID)],
            }
            observations.append(
                {
                    "camera_id": camera_id,
                    "window_id": window_id,
                    "frame_count": 4,
                    "model_observation": model_observation,
                }
            )
            backend_observations.append(model_observation)
        windows.append(
            {
                "ordinal": ordinal,
                "window_id": window_id,
                "start_seconds": ordinal * 4.0,
                "end_seconds": (ordinal + 1) * 4.0,
                "model": {
                    "status": "SUCCEEDED",
                    "input_observations": observations,
                },
            }
        )
    return {
        "format": "robata-production-wemm-vocabulary-shadow-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "SUCCEEDED",
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "production_eligible": False,
        "source": {
            "path": "data/source/sample-medium.mcap",
            "window_count": 10,
            "camera_count": 6,
        },
        "model": {
            "identifier": "WeMM-Embedding-2B",
            "dimension": 2048,
            "label_variant": "canonical",
            "frame_count": 4,
        },
        "vocabulary": {
            "owner_approved": True,
            "production_eligible": False,
            "epic_ontology_used": False,
            "mapper_used": False,
        },
        "windows": windows,
        "backend_observations": backend_observations,
        "controls": {
            "model_invoked": True,
            "gold_included": False,
            "predictions_are_gold": False,
            "existing_mapper_invoked": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_sha_used": False,
            "ground_truth_used_in_encoder_input": False,
        },
    }


def test_valid_common_cohort_and_baseline_build_four_arms() -> None:
    plan = build_wemm_frame_grid_matrix(_cohort(), _baseline())

    assert plan["format"] == "robata-production-wemm-frame-grid-matrix-v1"
    assert plan["status"] == "PLAN_ONLY"
    assert plan["cohort"]["slot_count"] == 60  # type: ignore[index]
    assert plan["baseline"]["video_call_count"] == 60  # type: ignore[index]
    assert [arm["arm_id"] for arm in plan["matrix"]["arms"]] == [  # type: ignore[index]
        "f4_current_budget",
        "f8_current_budget",
        "f4_higher_budget",
        "f8_higher_budget",
    ]


def test_arm_grid_and_budget_matrix_is_explicit() -> None:
    plan = build_wemm_frame_grid_matrix(_cohort(), _baseline())
    arms = {arm["arm_id"]: arm for arm in plan["matrix"]["arms"]}  # type: ignore[index]
    assert (
        arms["f4_current_budget"]["frame_count"],
        arms["f4_current_budget"]["total_pixel_budget"],
    ) == (4, CURRENT_TOTAL_PIXEL_BUDGET)
    assert arms["f4_current_budget"]["expected_grid_thw"] == list(BASELINE_GRID)
    assert arms["f8_current_budget"]["expected_grid_thw"] == list(CURRENT_8_FRAME_GRID)
    assert arms["f4_higher_budget"]["expected_grid_thw"] == list(HIGHER_4_FRAME_GRID)
    assert arms["f8_higher_budget"]["expected_grid_thw"] == list(HIGHER_8_FRAME_GRID)
    assert arms["f4_higher_budget"]["total_pixel_budget"] == HIGHER_TOTAL_PIXEL_BUDGET
    assert all(arm["planned_camera_window_slot_count"] == 60 for arm in arms.values())
    assert arms["f4_current_budget"]["grid_expectation_status"] == "observed_baseline"
    assert all(
        arms[name]["grid_expectation_status"] == "probe_expected"
        for name in ("f8_current_budget", "f4_higher_budget", "f8_higher_budget")
    )


def test_wrong_camera_metadata_is_rejected() -> None:
    cohort = _cohort()
    cohort["source"]["cameras"][0]["camera_id"] = "cam_99"  # type: ignore[index]
    with pytest.raises(WemmFrameGridMatrixError, match="camera"):
        build_wemm_frame_grid_matrix(cohort, _baseline())


def test_wrong_window_metadata_is_rejected() -> None:
    baseline = _baseline()
    baseline["windows"][3]["window_id"] = "wrong-window"  # type: ignore[index]
    with pytest.raises(WemmFrameGridMatrixError, match="window"):
        build_wemm_frame_grid_matrix(_cohort(), baseline)


def test_wrong_processor_grid_is_rejected() -> None:
    baseline = _baseline()
    baseline["backend_observations"][7]["video_grid_thw"] = [[9, 9, 9]]  # type: ignore[index]
    with pytest.raises(WemmFrameGridMatrixError, match="grid"):
        build_wemm_frame_grid_matrix(_cohort(), baseline)


def test_plan_controls_prove_no_model_media_or_identity_work() -> None:
    plan = build_wemm_frame_grid_matrix(_cohort(), _baseline())
    for key in (
        "model_invoked",
        "media_decoded",
        "full_37_mcap_opened",
        "hash_or_digest_computed",
    ):
        assert plan[key] is False
        assert plan["controls"][key] is False  # type: ignore[index]
    assert plan["controls"]["identity_computation"] == "none"  # type: ignore[index]
