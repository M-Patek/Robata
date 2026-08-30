from __future__ import annotations

from copy import deepcopy

import pytest

from robata.benchmark.production_cohort import DEFAULT_CAMERA_TOPICS
from robata.benchmark.production_model_output import build_model_output_sidecar
from robata.benchmark.production_readiness import (
    ProductionReadinessError,
    assess_production_readiness,
)


def _manifest() -> dict[str, object]:
    windows = [
        {
            "ordinal": 0,
            "window_id": "sample-w00",
            "start_seconds": 0.0,
            "end_seconds": 8.0,
            "camera_ids": list(DEFAULT_CAMERA_TOPICS),
        },
        {
            "ordinal": 1,
            "window_id": "sample-w01",
            "start_seconds": 8.0,
            "end_seconds": 16.0,
            "camera_ids": list(DEFAULT_CAMERA_TOPICS),
        },
    ]
    return {
        "format": "robata-production-shaped-cohort-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {"path": "data/source/sample-medium.mcap"},
        "windows": windows,
    }


def _approved_ontology() -> dict[str, object]:
    return {"approved": True, "actions": {"open cupboard": {"verb": "open"}}}


def _approved_mapping() -> dict[str, object]:
    return {"approved": True, "camera_mapping": {"cam_01": {"topic": "/cam/0"}}}


def _review_item(
    window: dict[str, object],
    *,
    status: str,
    accepted: bool = False,
) -> dict[str, object]:
    item: dict[str, object] = {
        "ordinal": window["ordinal"],
        "window_id": window["window_id"],
        "source_path": "data/source/sample-medium.mcap",
        "start_seconds": window["start_seconds"],
        "end_seconds": window["end_seconds"],
        "camera_ids": window["camera_ids"],
        "gold": {"status": status, "segments": []},
    }
    if accepted:
        item["gold"] = {
            "status": "ACCEPTED",
            "segments": [
                {
                    "start_seconds": window["start_seconds"],
                    "end_seconds": window["end_seconds"],
                    "verb": "open",
                    "noun": "cupboard",
                    "attributes": None,
                    "location": None,
                    "hand": None,
                }
            ],
            "provenance": {
                "reviewer_id": "reviewer-a",
                "reviewed_at": "2026-08-27T00:00:00Z",
                "adjudication_status": "ACCEPTED",
            },
        }
    return item


def test_unlabelled_cohort_is_not_quality_measureable() -> None:
    manifest = _manifest()
    sidecar = build_model_output_sidecar(manifest)
    report = assess_production_readiness(
        manifest,
        review_pack=None,
        sidecar=sidecar,
        ontology=_approved_ontology(),
        mapping=_approved_mapping(),
    )

    assert report["inference_readiness"] == "READY"
    assert report["quality_readiness"] == "NOT_MEASURED"
    assert report["quality_measurement_status"] == "NOT_MEASURED"
    assert report["production_eligible"] is False
    assert any("human_review" in blocker for blocker in report["blockers"])
    assert report["controls"]["model_invoked"] is False


def test_missing_ontology_and_mapping_block_invocation() -> None:
    manifest = _manifest()
    sidecar = build_model_output_sidecar(manifest)
    report = assess_production_readiness(manifest, sidecar=sidecar)

    assert report["inference_readiness"] == "BLOCKED"
    assert report["quality_measurement_status"] == "NOT_MEASURED"
    assert any("ontology" in blocker for blocker in report["blockers"])
    assert any("mapping" in blocker for blocker in report["blockers"])


def test_sidecar_source_mismatch_fails_closed() -> None:
    manifest = _manifest()
    sidecar = build_model_output_sidecar(manifest)
    altered = deepcopy(sidecar)
    altered["source"]["path"] = "other.mcap"  # type: ignore[index]

    with pytest.raises(ProductionReadinessError, match="does not bind"):
        assess_production_readiness(
            manifest,
            sidecar=altered,
            ontology=_approved_ontology(),
            mapping=_approved_mapping(),
        )


def test_pending_review_is_reported_per_window() -> None:
    manifest = _manifest()
    sidecar = build_model_output_sidecar(manifest)
    review = {
        "format": "robata-production-human-review-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source_manifest_format": "robata-production-shaped-cohort-v1",
        "source": {"path": "data/source/sample-medium.mcap"},
        "items": [
            _review_item(manifest["windows"][0], status="ACCEPTED", accepted=True),  # type: ignore[index]
            _review_item(manifest["windows"][1], status="PENDING_HUMAN_REVIEW"),  # type: ignore[index]
        ],
    }
    report = assess_production_readiness(
        manifest,
        review_pack=review,
        sidecar=sidecar,
        ontology=_approved_ontology(),
        mapping=_approved_mapping(),
    )

    human_gate = next(gate for gate in report["gates"] if gate["name"] == "human_review")
    assert human_gate["status"] == "BLOCKED"
    assert human_gate["details"]["pending_windows"] == ["sample-w01"]


def test_accepted_gold_does_not_measure_unrun_models() -> None:
    manifest = _manifest()
    sidecar = build_model_output_sidecar(manifest)
    review = {
        "format": "robata-production-human-review-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source_manifest_format": "robata-production-shaped-cohort-v1",
        "source": {"path": "data/source/sample-medium.mcap"},
        "items": [
            {
                **_review_item(window, status="ACCEPTED", accepted=True),
            }
            for window in manifest["windows"]  # type: ignore[index]
        ],
    }

    report = assess_production_readiness(
        manifest,
        review_pack=review,
        sidecar=sidecar,
        ontology=_approved_ontology(),
        mapping=_approved_mapping(),
    )

    assert report["inference_readiness"] == "READY"
    assert report["quality_readiness"] == "NOT_MEASURED"
    assert report["quality_measurement_status"] == "NOT_MEASURED"
    assert any("quality-measureable" in blocker for blocker in report["blockers"])


def test_all_succeeded_measured_slots_and_accepted_gold_measure_quality() -> None:
    """Quality cannot become READY until every model/window slot is measured."""

    manifest = _manifest()
    sidecar = build_model_output_sidecar(manifest)
    for window in sidecar["windows"]:  # type: ignore[index]
        for slot in window["model_outputs"].values():  # type: ignore[union-attr]
            slot["status"] = "SUCCEEDED"
            slot["predictions"] = [{"verb": "open", "noun": "cupboard"}]
            slot["metrics"] = {
                "measurement_status": "MEASURED",
                "status": "MEASURED",
                "values": {"latency_seconds": 1.0},
            }
    sidecar["controls"]["model_invoked"] = True  # type: ignore[index]

    review = {
        "format": "robata-production-human-review-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source_manifest_format": "robata-production-shaped-cohort-v1",
        "source": {"path": "data/source/sample-medium.mcap"},
        "items": [
            _review_item(window, status="ACCEPTED", accepted=True)
            for window in manifest["windows"]  # type: ignore[index]
        ],
    }

    report = assess_production_readiness(
        manifest,
        review_pack=review,
        sidecar=sidecar,
        ontology=_approved_ontology(),
        mapping=_approved_mapping(),
    )

    assert report["inference_readiness"] == "READY"
    assert report["quality_readiness"] == "READY"
    assert report["quality_measurement_status"] == "MEASURED"
    assert report["blockers"] == []


def test_review_window_mismatch_is_explicit() -> None:
    manifest = _manifest()
    review = {
        "format": "robata-production-human-review-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source_manifest_format": "robata-production-shaped-cohort-v1",
        "source": {"path": "data/source/sample-medium.mcap"},
        "items": [{"window_id": "unexpected", "gold": {"segments": []}}],
    }
    report = assess_production_readiness(manifest, review_pack=review)
    gate = next(gate for gate in report["gates"] if gate["name"] == "human_review")
    assert gate["status"] == "BLOCKED"
    assert gate["details"]["missing_windows"] == ["sample-w00", "sample-w01"]
    assert gate["details"]["extra_windows"] == ["unexpected"]


def test_manifest_requires_canonical_six_camera_window_mapping() -> None:
    manifest = _manifest()
    manifest["windows"][0]["camera_ids"] = [  # type: ignore[index]
        "cam_01",
        "cam_02",
        "cam_03",
        "cam_04",
        "cam_05",
        "other-camera",
    ]

    with pytest.raises(ProductionReadinessError, match="cam_01 through cam_06"):
        assess_production_readiness(manifest)


def test_manifest_source_camera_inventory_binds_window_topics() -> None:
    manifest = _manifest()
    manifest["source"]["camera_count"] = 6  # type: ignore[index]
    manifest["source"]["cameras"] = [  # type: ignore[index]
        {"camera_id": camera_id, "topic": f"/camera/{index}"}
        for index, camera_id in enumerate(DEFAULT_CAMERA_TOPICS)
    ]
    manifest["windows"][0]["camera_topics"] = {  # type: ignore[index]
        camera_id: f"/camera/{index}" for index, camera_id in enumerate(DEFAULT_CAMERA_TOPICS)
    }
    manifest["windows"][1]["camera_topics"] = dict(  # type: ignore[index]
        manifest["windows"][0]["camera_topics"]  # type: ignore[index]
    )
    manifest["windows"][1]["camera_topics"]["cam_06"] = "/camera/wrong"  # type: ignore[index]

    with pytest.raises(ProductionReadinessError, match="does not bind manifest source"):
        assess_production_readiness(manifest)


def test_sidecar_routes_must_bind_manifest_routes() -> None:
    manifest = _manifest()
    sidecar = build_model_output_sidecar(manifest)
    for window in manifest["windows"]:  # type: ignore[index]
        window["model_routes"] = {  # type: ignore[index]
            "wemm": "custom-wemm-route",
            "qwen": "custom-qwen-route",
            "mage": "custom-mage-route",
        }

    with pytest.raises(ProductionReadinessError, match="model_routes"):
        assess_production_readiness(
            manifest,
            sidecar=sidecar,
            ontology=_approved_ontology(),
            mapping=_approved_mapping(),
        )


def test_manifest_routes_must_be_consistent_across_windows() -> None:
    manifest = _manifest()
    manifest["windows"][1]["model_routes"] = {  # type: ignore[index]
        "wemm": "custom-wemm-route",
        "qwen": "custom-qwen-route",
        "mage": "custom-mage-route",
    }

    with pytest.raises(ProductionReadinessError, match="consistent"):
        assess_production_readiness(manifest)


@pytest.mark.parametrize("location", ("manifest", "review", "gold", "sidecar"))
def test_negative_time_bounds_fail_closed(location: str) -> None:
    """All source-bound geometry and accepted-gold bounds are non-negative."""

    manifest = _manifest()
    sidecar = build_model_output_sidecar(manifest)
    review = {
        "format": "robata-production-human-review-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source_manifest_format": "robata-production-shaped-cohort-v1",
        "source": {"path": "data/source/sample-medium.mcap"},
        "items": [
            _review_item(window, status="ACCEPTED", accepted=True)
            for window in manifest["windows"]  # type: ignore[index]
        ],
    }
    if location == "manifest":
        manifest["windows"][0]["start_seconds"] = -0.1  # type: ignore[index]
    elif location == "review":
        review["items"][0]["start_seconds"] = -0.1  # type: ignore[index]
    elif location == "gold":
        review["items"][0]["gold"]["segments"][0]["start_seconds"] = -0.1  # type: ignore[index]
    else:
        sidecar["windows"][0]["start_seconds"] = -0.1  # type: ignore[index]

    with pytest.raises(ProductionReadinessError, match="finite"):
        assess_production_readiness(
            manifest,
            review_pack=review,
            sidecar=sidecar,
            ontology=_approved_ontology(),
            mapping=_approved_mapping(),
        )
