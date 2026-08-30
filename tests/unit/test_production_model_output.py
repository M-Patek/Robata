from __future__ import annotations

from copy import deepcopy

import pytest

from robata.benchmark.production_model_output import (
    DEFAULT_NATIVE_MODEL_ROUTES,
    MODEL_NAMES,
    PRODUCTION_MODEL_OUTPUT_SIDECAR_VERSION,
    ProductionModelOutputError,
    build_model_output_sidecar,
    update_model_output_slot,
    validate_model_output_sidecar,
)


def _manifest(*, windows: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "format": "robata-production-shaped-cohort-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {
            "path": "sample-medium.mcap",
            "media_type": "application/x-mcap",
            "camera_count": 6,
        },
        "gold": {
            "status": "PENDING_HUMAN_REVIEW",
            "segments": [],
        },
        "windows": windows
        or [
            {
                "ordinal": 0,
                "window_id": "sample-medium-w00",
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "camera_ids": ["cam_01", "cam_02"],
                "model_routes": dict(DEFAULT_NATIVE_MODEL_ROUTES),
                "gold_status": "PENDING_HUMAN_REVIEW",
                "review": {"segments": []},
            },
            {
                "ordinal": 1,
                "window_id": "sample-medium-w01",
                "start_seconds": 8.0,
                "end_seconds": 16.0,
                "camera_ids": ["cam_01", "cam_02"],
                "model_routes": dict(DEFAULT_NATIVE_MODEL_ROUTES),
            },
        ],
    }


def test_builder_initialises_three_independent_slots_without_gold() -> None:
    sidecar = build_model_output_sidecar(_manifest())

    assert sidecar["format"] == PRODUCTION_MODEL_OUTPUT_SIDECAR_VERSION
    assert sidecar["controls"]["gold_included"] is False
    assert sidecar["contract"]["gold_fields_included"] is False
    assert set(sidecar["model_routes"]) == set(MODEL_NAMES)
    assert len(sidecar["windows"]) == 2

    for window in sidecar["windows"]:
        assert "gold" not in window
        outputs = window["model_outputs"]
        assert set(outputs) == set(MODEL_NAMES)
        for model in MODEL_NAMES:
            slot = outputs[model]
            assert slot["model"] == model
            assert slot["window_id"] == window["window_id"]
            assert slot["native_route"] == DEFAULT_NATIVE_MODEL_ROUTES[model]
            assert slot["status"] == "NOT_RUN"
            assert slot["predictions"] == []
            assert slot["metrics"]["measurement_status"] == "NOT_MEASURED"
            assert slot["metrics"]["status"] == "NOT_MEASURED"
            assert slot["metrics"]["values"] == {}
            assert slot["artifact_lineage"]["window_id"] == window["window_id"]
            assert slot["artifact_lineage"]["input_artifacts"] == []


def test_sidecar_round_trip_is_independent_from_manifest_mutation() -> None:
    manifest = _manifest()
    sidecar = build_model_output_sidecar(manifest)
    manifest["windows"][0]["model_routes"]["qwen"] = "tampered-route"  # type: ignore[index]
    assert (
        sidecar["windows"][0]["model_outputs"]["qwen"]["native_route"]
        == (DEFAULT_NATIVE_MODEL_ROUTES["qwen"])
    )
    assert validate_model_output_sidecar(sidecar) == sidecar


def test_update_replaces_only_one_bound_model_window_slot() -> None:
    sidecar = build_model_output_sidecar(_manifest())
    updated = update_model_output_slot(
        sidecar,
        window_id="sample-medium-w00",
        model="qwen",
        status="SUCCEEDED",
        predictions=[{"verb": "turn", "noun": "tap"}],
        metrics={"measurement_status": "NOT_MEASURED", "status": "NOT_MEASURED", "values": {}},
    )

    assert updated["windows"][0]["model_outputs"]["qwen"]["status"] == "SUCCEEDED"
    assert updated["windows"][0]["model_outputs"]["qwen"]["predictions"] == [
        {"verb": "turn", "noun": "tap"}
    ]
    assert updated["windows"][0]["model_outputs"]["wemm"]["status"] == "NOT_RUN"
    assert sidecar["windows"][0]["model_outputs"]["qwen"]["status"] == "NOT_RUN"


@pytest.mark.parametrize(
    "location",
    (
        "slot",
        "prediction",
        "lineage",
    ),
)
def test_gold_cannot_be_injected_into_output_sidecar(location: str) -> None:
    sidecar = build_model_output_sidecar(_manifest())
    tampered = deepcopy(sidecar)
    slot = tampered["windows"][0]["model_outputs"]["wemm"]
    if location == "slot":
        slot["gold"] = {"segments": []}
    elif location == "prediction":
        slot["predictions"] = [{"ground_truth": "turn tap"}]
    else:
        slot["artifact_lineage"]["official_reference"] = "labels.json"

    with pytest.raises(ProductionModelOutputError, match=r"gold|annotation|sidecar"):
        validate_model_output_sidecar(tampered)


def test_not_measured_metrics_cannot_retain_values() -> None:
    sidecar = build_model_output_sidecar(_manifest())
    tampered = deepcopy(sidecar)
    tampered["windows"][0]["model_outputs"]["mage"]["metrics"]["values"] = {"latency_ms": 3}

    with pytest.raises(ProductionModelOutputError, match="NOT_MEASURED"):
        validate_model_output_sidecar(tampered)


def test_inconsistent_manifest_routes_fail_closed() -> None:
    windows = _manifest()["windows"]
    assert isinstance(windows, list)
    windows[1]["model_routes"] = {
        **DEFAULT_NATIVE_MODEL_ROUTES,
        "mage": "different-native-route",
    }

    with pytest.raises(ProductionModelOutputError, match="consistent"):
        build_model_output_sidecar(_manifest(windows=windows))
