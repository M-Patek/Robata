from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robata.benchmark.production_gold_collection import (
    DECISION_OPTIONS,
    LIFECYCLE_STATES,
    REQUIRED_SEGMENT_FIELDS,
    SOURCE_BOUND_GOLD_COLLECTION_VERSION,
    ProductionGoldCollectionError,
    build_source_bound_gold_collection,
)


def _manifest() -> dict[str, object]:
    return {
        "format": "robata-production-shaped-cohort-v1",
        "source": {
            "path": "data/source/sample-medium.mcap",
            "media_type": "application/x-mcap",
            "camera_count": 2,
            "cameras": [{"camera_id": "cam_01"}, {"camera_id": "cam_02"}],
        },
        "windows": [
            {
                "ordinal": 0,
                "window_id": "sample-medium-w00",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "camera_ids": ["cam_01", "cam_02"],
                "camera_topics": {"cam_01": "/camera/0", "cam_02": "/camera/1"},
                # This must not be copied into gold.
                "review": {"segments": [{"verb": "fold", "noun": "garment"}]},
            }
        ],
    }


def test_builder_is_blank_source_bound_and_lifecycle_explicit() -> None:
    payload = build_source_bound_gold_collection(
        _manifest(),
        manifest_reference="cohort.json",
        evidence_reference="review-surfaces.json",
    )
    assert payload["format"] == SOURCE_BOUND_GOLD_COLLECTION_VERSION
    assert payload["status"] == "PENDING"
    assert payload["official_gold_status"] == "NOT_ESTABLISHED"
    assert payload["official_quality_status"] == "NOT_MEASURED"
    assert payload["human_adjudication"] == "NOT_PERFORMED"
    assert payload["production_eligible"] is False
    assert payload["quality_measurement_status"] == "NOT_MEASURED"
    assert payload["lifecycle"]["states"] == list(LIFECYCLE_STATES)
    assert payload["reviewer_slots"] == ["reviewer_a", "reviewer_b"]
    window = payload["windows"][0]
    assert window["source"]["path"] == "data/source/sample-medium.mcap"
    assert window["source"]["interval"] == [0.0, 4.0]
    assert window["source"]["fixed_window"] is True
    assert window["source"]["action_boundary_status"] == "NOT_ESTABLISHED"
    assert window["gold"]["segments"] == []
    assert window["gold"]["required_segment_fields"] == list(REQUIRED_SEGMENT_FIELDS)
    assert window["review"]["decision"] is None
    assert window["review"]["decision_options"] == list(DECISION_OPTIONS)
    assert window["evidence"]["surface_reference"] == "review-surfaces.json"
    assert window["model_context"]["copied_into_gold"] is False
    assert payload["controls"]["terra_labels_copied"] is False
    assert payload["contract"]["annotation_principal"]["source"] == (
        "data/source/annotation-principal.txt"
    )
    principal = payload["contract"]["annotation_principal"]
    assert principal["consistent_verb_noun_pairs"] is True
    assert principal["short_instruction_style"] is True


def test_builder_does_not_copy_review_segments_or_accept_bad_windows() -> None:
    payload = build_source_bound_gold_collection(_manifest())
    assert payload["windows"][0]["gold"]["segments"] == []
    bad = _manifest()
    bad["windows"] = [
        {
            "ordinal": 0,
            "window_id": "w00",
            "start_seconds": 4.0,
            "end_seconds": 4.0,
            "camera_ids": ["cam_01"],
        }
    ]
    with pytest.raises(ProductionGoldCollectionError, match="start_seconds"):
        build_source_bound_gold_collection(bad)


def test_builder_rejects_duplicate_reviewer_slots_and_window_ids() -> None:
    with pytest.raises(ProductionGoldCollectionError, match="duplicate reviewer slot"):
        build_source_bound_gold_collection(_manifest(), reviewer_slots=["a", "a"])
    duplicate = _manifest()
    duplicate["windows"] = [*duplicate["windows"], duplicate["windows"][0]]  # type: ignore[index]
    with pytest.raises(ProductionGoldCollectionError, match="duplicate window_id"):
        build_source_bound_gold_collection(duplicate)


def test_builder_rejects_cross_namespace_manifest_and_empty_cohort() -> None:
    epic = _manifest()
    epic["format"] = "epic-kitchens-action-catalog-v1"
    with pytest.raises(ProductionGoldCollectionError, match=r"manifest\.format"):
        build_source_bound_gold_collection(epic)

    wrong_authority = _manifest()
    wrong_authority["authority"] = "EPIC_BENCHMARK"
    with pytest.raises(ProductionGoldCollectionError, match="authority"):
        build_source_bound_gold_collection(wrong_authority)

    empty = _manifest()
    empty["windows"] = []
    with pytest.raises(ProductionGoldCollectionError, match="windows must not be empty"):
        build_source_bound_gold_collection(empty)


def test_builder_rejects_duplicate_ordinals_and_camera_metadata_drift() -> None:
    duplicate = _manifest()
    duplicate["windows"] = [
        duplicate["windows"][0],  # type: ignore[index]
        {
            **duplicate["windows"][0],  # type: ignore[index]
            "window_id": "sample-medium-w01",
        },
    ]
    with pytest.raises(ProductionGoldCollectionError, match="duplicate window ordinal"):
        build_source_bound_gold_collection(duplicate)

    out_of_order = _manifest()
    out_of_order["windows"] = [
        {
            **out_of_order["windows"][0],  # type: ignore[index]
            "ordinal": 1,
            "window_id": "sample-medium-w01",
        },
        {
            **out_of_order["windows"][0],  # type: ignore[index]
            "ordinal": 0,
            "window_id": "sample-medium-w00b",
        },
    ]
    with pytest.raises(ProductionGoldCollectionError, match="strictly increasing"):
        build_source_bound_gold_collection(out_of_order)

    drift = _manifest()
    drift["windows"][0]["camera_ids"] = ["cam_02"]  # type: ignore[index]
    with pytest.raises(ProductionGoldCollectionError, match=r"bind manifest\.source\.cameras"):
        build_source_bound_gold_collection(drift)


def test_builder_binds_window_topics_to_source_inventory() -> None:
    manifest = _manifest()
    manifest["source"]["cameras"] = [  # type: ignore[index]
        {"camera_id": "cam_01", "topic": "/camera/0"},
        {"camera_id": "cam_02", "topic": "/camera/1"},
    ]
    altered = _manifest()
    altered["source"]["cameras"] = manifest["source"]["cameras"]  # type: ignore[index]
    altered["windows"][0]["camera_topics"]["cam_02"] = "/camera/wrong"  # type: ignore[index]

    with pytest.raises(
        ProductionGoldCollectionError,
        match=r"bind manifest source camera topics",
    ):
        build_source_bound_gold_collection(altered)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_topics", {"cam_01": "/camera/0"}, "exactly the window camera IDs"),
        ("camera_topics", {"cam_01": "/camera/0", "cam_02": ""}, "non-empty string"),
        ("camera_topics", {"cam_01": "/camera/0", "other": "/camera/1"}, "not bound"),
    ],
)
def test_builder_rejects_incomplete_or_unbound_window_topics(
    field: str, value: object, message: str
) -> None:
    manifest = _manifest()
    manifest["windows"][0][field] = value  # type: ignore[index]
    with pytest.raises(ProductionGoldCollectionError, match=message):
        build_source_bound_gold_collection(manifest)


def test_builder_uses_declared_camera_count_when_source_inventory_is_omitted() -> None:
    manifest = _manifest()
    manifest["source"].pop("cameras")  # type: ignore[index]
    manifest["windows"][0]["camera_ids"] = ["cam_01"]  # type: ignore[index]
    with pytest.raises(ProductionGoldCollectionError, match="contain 2 cameras"):
        build_source_bound_gold_collection(manifest)


def test_builder_rejects_overlapping_windows_and_camera_set_drift_without_inventory() -> None:
    manifest = _manifest()
    manifest["source"].pop("cameras")  # type: ignore[index]
    manifest["source"].pop("camera_count")  # type: ignore[index]
    first = manifest["windows"][0]  # type: ignore[index]
    manifest["windows"] = [
        first,
        {
            **first,
            "window_id": "sample-medium-w01",
            "ordinal": 1,
            "start_seconds": 3.0,
            "end_seconds": 5.0,
        },
    ]
    with pytest.raises(ProductionGoldCollectionError, match="non-overlapping"):
        build_source_bound_gold_collection(manifest)

    manifest = _manifest()
    manifest["source"].pop("cameras")  # type: ignore[index]
    manifest["source"].pop("camera_count")  # type: ignore[index]
    first = manifest["windows"][0]  # type: ignore[index]
    manifest["windows"] = [
        first,
        {
            **first,
            "window_id": "sample-medium-w01",
            "ordinal": 1,
            "start_seconds": 4.0,
            "end_seconds": 8.0,
            "camera_ids": ["cam_02"],
            "camera_topics": {"cam_02": "/camera/1"},
        },
    ]
    with pytest.raises(ProductionGoldCollectionError, match="consistent across"):
        build_source_bound_gold_collection(manifest)


def test_builder_rejects_string_reviewer_slots_and_non_string_metadata_keys() -> None:
    with pytest.raises(ProductionGoldCollectionError, match="reviewer_slots must be an array"):
        build_source_bound_gold_collection(_manifest(), reviewer_slots="reviewer_a")  # type: ignore[arg-type]

    manifest = _manifest()
    manifest["windows"][0]["camera_topics"] = {1: "/camera/0", "cam_02": "/camera/1"}  # type: ignore[index]
    with pytest.raises(ProductionGoldCollectionError, match="keys must be strings"):
        build_source_bound_gold_collection(manifest)


def test_builder_validates_source_topic_aliases_and_optional_references() -> None:
    manifest = _manifest()
    manifest["source"]["cameras"] = [  # type: ignore[index]
        {"camera_id": "cam_01", "topic": "/camera/0", "camera_topic": "/camera/0"},
        {"camera_id": "cam_02", "topic": "/camera/1", "camera_topic": "/camera/1"},
    ]
    payload = build_source_bound_gold_collection(
        manifest,
        manifest_reference=" cohort.json ",
        evidence_reference=" review.json ",
    )
    assert payload["source"]["source_manifest_reference"] == "cohort.json"
    assert payload["windows"][0]["evidence"]["surface_reference"] == "review.json"

    altered = _manifest()
    altered["source"]["cameras"] = [  # type: ignore[index]
        {"camera_id": "cam_01", "topic": "/camera/0", "camera_topic": "/camera/other"},
        {"camera_id": "cam_02", "topic": "/camera/1", "camera_topic": "/camera/1"},
    ]
    with pytest.raises(ProductionGoldCollectionError, match="aliases disagree"):
        build_source_bound_gold_collection(altered)

    with pytest.raises(ProductionGoldCollectionError, match="manifest_reference"):
        build_source_bound_gold_collection(_manifest(), manifest_reference=" ")


def test_cli_smoke_writes_json_and_markdown(tmp_path: Path) -> None:
    """Exercise the CLI with a self-contained manifest fixture.

    The production-shaped cohort fixture is intentionally local and ignored, so
    a clean checkout must not depend on ``.agent_tmp`` being present.
    """

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    output = tmp_path / "collection.json"
    output_md = tmp_path / "collection.md"
    script = Path(__file__).parents[2] / "scripts" / "build_production_gold_collection.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(manifest),
            "--output",
            str(output),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == SOURCE_BOUND_GOLD_COLLECTION_VERSION
    assert payload["source"]["source_manifest_reference"] == str(manifest)
    assert "blank source-bound collection queue" in output_md.read_text(encoding="utf-8")
