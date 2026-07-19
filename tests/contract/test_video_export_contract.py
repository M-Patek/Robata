from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from robata.contracts import (
    CameraVideoExportManifest,
    CameraVideoTimestampRow,
    SchemaRegistry,
    SchemaValidationError,
)

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[2] / "schemas" / "v1"
SCHEMA_NAME = "camera-video-export-manifest"
CAMERA_IDS = tuple(f"cam_{number:02d}" for number in range(1, 7))


@pytest.fixture(scope="module")
def registry() -> SchemaRegistry:
    return SchemaRegistry(SCHEMA_DIRECTORY)


def _uuid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012x}"


def _camera(number: int) -> dict[str, Any]:
    return {
        "camera_id": CAMERA_IDS[number - 1],
        "source": {
            "topic": f"/camera/{number}",
            "channel_id": number,
            "schema_name": "foxglove.CompressedImage",
            "codec": "h264",
        },
        "input_message_count": 30,
        "source_first_observed_message_ns": "100",
        "source_last_observed_message_ns": "1000",
        "export_first_observed_source_message_ns": "200",
        "export_last_observed_source_message_ns": "900",
        "leading_drops": {
            "count": 1,
            "reason_code": "BEFORE_FIRST_DECODABLE_KEYFRAME",
            "first_source_ns": "100",
            "last_source_ns": "100",
        },
        "trailing_drops": {
            "count": 1,
            "reason_code": "AFTER_LAST_COMPLETE_SAMPLE",
            "first_source_ns": "1000",
            "last_source_ns": "1000",
        },
        "exported_packet_count": 28,
        "exported_frame_count": 28,
        "keyframe_count": 2,
        "width": 1600,
        "height": 1300,
        "video_artifact": {
            "uri": f"object://exports/{CAMERA_IDS[number - 1]}.mp4",
            "sha256": f"{number:x}" * 64,
            "bytes": 1234,
            "media_type": "video/mp4",
        },
        "timestamp_sidecar_artifact": {
            "uri": f"object://exports/{CAMERA_IDS[number - 1]}.timestamps.ndjson",
            "sha256": f"{number + 6:x}" * 64,
            "bytes": 456,
            "row_count": 28,
            "media_type": "application/x-ndjson",
        },
        "media_time_mapping": {
            "zero_source_ns": "200",
            "time_base_numerator": 1,
            "time_base_denominator": 1_000_000_000,
            "first_pts": 0,
            "last_pts": 700,
            "last_duration": 33,
            "tail_duration_policy": "MEDIAN_POSITIVE_INTERVAL",
            "rounding": "HALF_EVEN",
            "max_rounding_error_ns": "0",
        },
    }


def _local_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "execution_mode": "LOCAL_DEVELOPMENT_OVERRIDE",
        "recording_identity": "b" * 64,
        "source_content_sha256": "a" * 64,
        "source_size_bytes": 123456,
        "mapping_profile": {
            "version": "observed-v1",
            "digest": "c" * 64,
            "approved": False,
        },
        "ready_manifest_id": None,
        "alignment_id": None,
        "alignment_status": "UNVERIFIED",
        "exporter": {
            "name": "ffmpeg-exporter",
            "version": "1.0.0",
            "mode": "REMUX",
            "export_profile_id": "mp4-h264-analysis",
            "profile_version": "1.0.0",
            "canonical_config_sha256": "d" * 64,
        },
        "cameras": [_camera(number) for number in range(1, 7)],
    }


def _timestamp_row() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "export_profile_id": "direct-h264-remux-no-reordering",
        "export_profile_version": "1.0",
        "camera_id": "cam_01",
        "packet_index": 0,
        "source_sequence": 42,
        "source_log_time_ns": "1710000000000000000",
        "source_publish_time_ns": "1710000000000000001",
        "embedded_header_time_ns": "1710000000000000002",
        "relative_pts_ns": "0",
        "relative_dts_ns": "0",
        "duration_ns": "33333333",
        "time_base_numerator": 1,
        "time_base_denominator": 1_000_000_000,
        "is_keyframe": True,
        "duration_is_estimated": False,
    }


def _governed_manifest() -> dict[str, Any]:
    payload = _local_manifest()
    payload["execution_mode"] = "GOVERNED_READY"
    payload["ready_manifest_id"] = _uuid(1)
    payload["mapping_profile"]["approved"] = True
    return payload


def _model(payload: dict[str, Any]) -> CameraVideoExportManifest:
    return CameraVideoExportManifest.model_validate_json(json.dumps(payload))


def _timestamp_model(payload: dict[str, Any]) -> CameraVideoTimestampRow:
    return CameraVideoTimestampRow.model_validate_json(json.dumps(payload))


def _assert_rejected_by_both(
    registry: SchemaRegistry,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(SchemaValidationError):
        registry.validate(SCHEMA_NAME, payload)
    with pytest.raises(ValidationError):
        _model(payload)


def test_local_manifest_round_trips_through_schema_and_model(
    registry: SchemaRegistry,
) -> None:
    payload = _local_manifest()

    assert registry.validate(SCHEMA_NAME, payload) is payload
    manifest = _model(payload)
    serialized = manifest.model_dump(mode="json")

    assert serialized == payload
    assert registry.validate(SCHEMA_NAME, serialized) is serialized
    assert isinstance(manifest.cameras, tuple)


def test_governed_ready_does_not_claim_alignment(registry: SchemaRegistry) -> None:
    payload = _governed_manifest()

    registry.validate(SCHEMA_NAME, payload)
    manifest = _model(payload)

    assert manifest.ready_manifest_id == _uuid(1)
    assert manifest.alignment_id is None
    assert manifest.alignment_status.value == "UNVERIFIED"


def test_governed_manifest_may_carry_separate_valid_alignment(
    registry: SchemaRegistry,
) -> None:
    payload = _governed_manifest()
    payload["alignment_id"] = _uuid(2)
    payload["alignment_status"] = "VALID"

    registry.validate(SCHEMA_NAME, payload)
    assert _model(payload).alignment_id == _uuid(2)


@pytest.mark.parametrize("invalid_state", ["ready", "approved_mapping", "valid_alignment"])
def test_local_override_rejects_governed_evidence(
    registry: SchemaRegistry,
    invalid_state: str,
) -> None:
    payload = _local_manifest()
    if invalid_state == "ready":
        payload["ready_manifest_id"] = _uuid(1)
    elif invalid_state == "approved_mapping":
        payload["mapping_profile"]["approved"] = True
    else:
        payload["alignment_id"] = _uuid(2)
        payload["alignment_status"] = "VALID"

    _assert_rejected_by_both(registry, payload)


@pytest.mark.parametrize("invalid_state", ["missing_ready", "unapproved_mapping"])
def test_governed_mode_requires_ready_and_approved_mapping(
    registry: SchemaRegistry,
    invalid_state: str,
) -> None:
    payload = _governed_manifest()
    if invalid_state == "missing_ready":
        payload["ready_manifest_id"] = None
    else:
        payload["mapping_profile"]["approved"] = False

    _assert_rejected_by_both(registry, payload)


def test_valid_alignment_requires_alignment_id(registry: SchemaRegistry) -> None:
    payload = _governed_manifest()
    payload["alignment_status"] = "VALID"

    _assert_rejected_by_both(registry, payload)


@pytest.mark.parametrize("mutation", ["order", "cardinality"])
def test_camera_array_has_exact_canonical_order(
    registry: SchemaRegistry,
    mutation: str,
) -> None:
    payload = _local_manifest()
    if mutation == "order":
        payload["cameras"][0], payload["cameras"][1] = (
            payload["cameras"][1],
            payload["cameras"][0],
        )
    else:
        payload["cameras"].pop()

    _assert_rejected_by_both(registry, payload)


@pytest.mark.parametrize(
    "mutation",
    ["input_count", "sidecar_rows", "export_range", "leading_range"],
)
def test_model_enforces_export_accounting_and_time_ranges(mutation: str) -> None:
    payload = _local_manifest()
    camera = payload["cameras"][0]
    if mutation == "input_count":
        camera["input_message_count"] = 31
    elif mutation == "sidecar_rows":
        camera["timestamp_sidecar_artifact"]["row_count"] = 27
    elif mutation == "export_range":
        camera["export_first_observed_source_message_ns"] = "99"
    else:
        camera["leading_drops"]["first_source_ns"] = "250"
        camera["leading_drops"]["last_source_ns"] = "250"

    with pytest.raises(ValidationError):
        _model(payload)


def test_none_drop_provenance_requires_zero_count_and_null_times(
    registry: SchemaRegistry,
) -> None:
    payload = _local_manifest()
    camera = payload["cameras"][0]
    camera["leading_drops"] = {
        "count": 0,
        "reason_code": "NONE",
        "first_source_ns": None,
        "last_source_ns": None,
    }
    camera["input_message_count"] = 29

    registry.validate(SCHEMA_NAME, payload)
    _model(payload)

    camera["leading_drops"]["count"] = 1
    _assert_rejected_by_both(registry, payload)


def test_non_none_drop_provenance_requires_source_times(
    registry: SchemaRegistry,
) -> None:
    payload = _local_manifest()
    payload["cameras"][0]["trailing_drops"]["last_source_ns"] = None

    _assert_rejected_by_both(registry, payload)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("source_first_observed_message_ns", 100),
        ("source_first_observed_message_ns", "0100"),
    ],
)
def test_nanoseconds_are_canonical_wire_strings(
    registry: SchemaRegistry,
    field: str,
    invalid_value: object,
) -> None:
    payload = _local_manifest()
    payload["cameras"][0][field] = invalid_value

    _assert_rejected_by_both(registry, payload)


def test_contract_rejects_unknown_fields_and_bare_artifact_paths(
    registry: SchemaRegistry,
) -> None:
    payload = _local_manifest()
    payload["unexpected"] = True
    _assert_rejected_by_both(registry, payload)

    payload = _local_manifest()
    payload["cameras"][0]["video_artifact"]["uri"] = "exports/cam_01.mp4"
    _assert_rejected_by_both(registry, payload)


def test_integer_timebase_rejects_float_wire_value() -> None:
    payload = _local_manifest()
    payload["cameras"][0]["media_time_mapping"]["time_base_denominator"] = 1.0

    with pytest.raises(ValidationError):
        _model(payload)


@pytest.mark.parametrize("mutation", ["pts_order", "exclusive_end_overflow"])
def test_media_time_mapping_is_reconstructible(mutation: str) -> None:
    payload = _local_manifest()
    mapping = payload["cameras"][0]["media_time_mapping"]
    if mutation == "pts_order":
        mapping["first_pts"] = 701
    else:
        mapping["last_pts"] = 9223372036854775807

    with pytest.raises(ValidationError):
        _model(payload)


def test_manifest_and_nested_artifacts_are_immutable() -> None:
    manifest = _model(_local_manifest())

    with pytest.raises(ValidationError):
        manifest.source_size_bytes = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        manifest.cameras[0].video_artifact.bytes = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.cameras[0] = manifest.cameras[1]  # type: ignore[index]


def test_timestamp_row_round_trips_through_registered_schema(
    registry: SchemaRegistry,
) -> None:
    payload = _timestamp_row()

    assert registry.validate("camera-video-timestamp-row", payload) is payload
    row = _timestamp_model(payload)
    serialized = row.model_dump(mode="json")

    assert serialized == payload
    assert registry.validate("camera-video-timestamp-row", serialized) is serialized


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("source_log_time_ns", 1710000000000000000),
        ("duration_ns", "0"),
        ("relative_pts_ns", "-1"),
        ("time_base_denominator", 90000),
    ],
)
def test_timestamp_row_rejects_invalid_wire_values(
    registry: SchemaRegistry,
    field: str,
    invalid_value: object,
) -> None:
    payload = _timestamp_row()
    payload[field] = invalid_value

    with pytest.raises(SchemaValidationError):
        registry.validate("camera-video-timestamp-row", payload)
    with pytest.raises(ValidationError):
        _timestamp_model(payload)


def test_timestamp_row_is_closed_and_immutable(registry: SchemaRegistry) -> None:
    payload = _timestamp_row()
    payload["path"] = "sidecars/cam_01.ndjson"

    with pytest.raises(SchemaValidationError):
        registry.validate("camera-video-timestamp-row", payload)
    with pytest.raises(ValidationError):
        _timestamp_model(payload)

    row = _timestamp_model(_timestamp_row())
    with pytest.raises(ValidationError):
        row.packet_index = 1  # type: ignore[misc]


def test_timestamp_row_exclusive_end_must_fit_int64() -> None:
    payload = _timestamp_row()
    payload["relative_pts_ns"] = "9223372036854775807"

    with pytest.raises(ValidationError):
        _timestamp_model(payload)
