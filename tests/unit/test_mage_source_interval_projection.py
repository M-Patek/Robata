from __future__ import annotations

import json

import pytest

from robata.benchmark.mage_source_interval_projection import (
    MageSourceIntervalProjectionError,
    build_mage_source_interval_projection,
)


def _bounded_record(**changes: object) -> dict[str, object]:
    record: dict[str, object] = {
        "uid": "P32_07_0",
        "bounded_clip": {
            "path": "C:/bounded/P32_07_0.mp4",
            "start_seconds": 1.4,
            "end_seconds": 10.3,
        },
    }
    record.update(changes)
    return record


def _serialized_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {
            key
            for item_key, child in value.items()
            for key in ({item_key} | _serialized_keys(child))
        }
    if isinstance(value, list):
        return set().union(*(_serialized_keys(child) for child in value))
    return set()


def test_projection_keeps_only_source_location_fields_and_marks_missing_source_media() -> None:
    sensitive_value = "must-not-appear-in-projection"
    input_pack = {
        "record_count": 1,
        "records": [
            _bounded_record(
                official_reference={"value": sensitive_value},
                raw_output=sensitive_value,
                reveal=sensitive_value,
                sha256=sensitive_value,
                mapper=sensitive_value,
                adapter=sensitive_value,
            )
        ],
    }

    projection = build_mage_source_interval_projection(input_pack)

    record = projection["records"][0]
    assert record == {
        "uid": "P32_07_0",
        "source_video_id": None,
        "source_video_id_unavailable": True,
        "source_video_id_reason": "not present in the label-blind Mage bounded-case record",
        "source_video_path": None,
        "source_video_path_unavailable": True,
        "source_video_path_reason": "not present in the label-blind Mage bounded-case record",
        "source_interval_start_seconds": 1.4,
        "source_interval_start_seconds_unavailable": False,
        "source_interval_start_seconds_reason": "explicit bounded_clip.start_seconds",
        "source_interval_end_seconds": 10.3,
        "source_interval_end_seconds_unavailable": False,
        "source_interval_end_seconds_reason": "explicit bounded_clip.end_seconds",
        "bounded_clip_locator": "C:/bounded/P32_07_0.mp4",
        "bounded_clip_locator_unavailable": False,
        "bounded_clip_locator_reason": "bounded_clip.path is the available locator",
        "bounded_clip_path": "C:/bounded/P32_07_0.mp4",
        "bounded_clip_path_unavailable": False,
        "bounded_clip_path_reason": "explicit bounded_clip.path",
    }
    assert sensitive_value not in json.dumps(projection)
    assert not any(
        forbidden in key.casefold()
        for key in _serialized_keys(projection)
        for forbidden in ("official", "raw", "reveal", "sha", "hash", "mapper", "adapter")
    )
    assert projection["mechanical_overlap_readiness"] == {
        "status": "MAGE_SOURCE_BINDING_INCOMPLETE",
        "reason": "one or more bounded cases lack source binding fields",
        "incomplete_cases": [
            {
                "uid": "P32_07_0",
                "unavailable_fields": ["source_video_id", "source_video_path"],
            }
        ],
    }


def test_projection_uses_only_explicit_source_video_values() -> None:
    input_pack = {
        "records": [
            _bounded_record(
                source_video={
                    "identity": "P32_07",
                    "path": "C:/source/P32_07.MP4",
                },
                bounded_clip={
                    "locator": "bounded://P32_07_0",
                    "path": "C:/bounded/P32_07_0.mp4",
                    "start_seconds": 1.4,
                    "end_seconds": 10.3,
                },
            )
        ]
    }

    projection = build_mage_source_interval_projection(input_pack)

    record = projection["records"][0]
    assert record["source_video_id"] == "P32_07"
    assert record["source_video_id_unavailable"] is False
    assert record["source_video_path"] == "C:/source/P32_07.MP4"
    assert record["source_video_path_unavailable"] is False
    assert record["bounded_clip_locator"] == "bounded://P32_07_0"
    assert projection["mechanical_overlap_readiness"]["status"] == (
        "READY_FOR_EXTERNAL_QWEN_INTERVAL_INTERSECTION"
    )


def test_projection_marks_invalid_or_missing_bounded_fields_unavailable() -> None:
    projection = build_mage_source_interval_projection(
        {
            "records": [
                {
                    "uid": "P32_07_0",
                    "bounded_clip": {
                        "start_seconds": 4.0,
                        "end_seconds": 4.0,
                    },
                }
            ]
        }
    )

    record = projection["records"][0]
    assert record["bounded_clip_locator"] is None
    assert record["bounded_clip_locator_unavailable"] is True
    assert record["bounded_clip_locator_reason"] == "bounded_clip.locator is not present"
    assert record["bounded_clip_path"] is None
    assert record["bounded_clip_path_unavailable"] is True
    assert record["source_interval_start_seconds"] is None
    assert record["source_interval_end_seconds"] is None
    assert record["source_interval_start_seconds_reason"] == "bounded_clip interval is not positive"
    assert record["source_interval_end_seconds_reason"] == "bounded_clip interval is not positive"


def test_projection_rejects_duplicate_uids_and_bad_record_count() -> None:
    with pytest.raises(MageSourceIntervalProjectionError, match="duplicate UIDs"):
        build_mage_source_interval_projection({"records": [_bounded_record(), _bounded_record()]})
    with pytest.raises(MageSourceIntervalProjectionError, match="record_count"):
        build_mage_source_interval_projection({"record_count": 2, "records": [_bounded_record()]})
