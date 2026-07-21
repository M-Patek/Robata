from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robata.contracts.schema_registry import (
    JSON_SCHEMA_DIALECT,
    SchemaNotFoundError,
    SchemaRegistry,
    SchemaValidationError,
)

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[2] / "schemas" / "v1"
CAMERA_IDS = tuple(f"cam_{number:02d}" for number in range(1, 7))


@pytest.fixture(scope="module")
def registry() -> SchemaRegistry:
    return SchemaRegistry(SCHEMA_DIRECTORY)


def _uuid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012x}"


def _camera(number: int) -> dict[str, Any]:
    return {
        "camera_id": CAMERA_IDS[number - 1],
        "role": f"view-{number}",
        "stream_id": _uuid(100 + number),
        "topic": f"/camera/{number}",
        "channel_id": number,
        "codec": "h264",
        "width": 1920,
        "height": 1080,
        "nominal_fps": 30.0,
        "source_start_ns": "1710000000000000000",
        "source_end_ns": "1710000001000000000",
        "frame_count": 30,
    }


def _mcap_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "mcap_id": _uuid(1),
        "validation_report_id": _uuid(4),
        "source": {
            "uri": "object://recordings/example.mcap",
            "version": "version-1",
            "sha256": "a" * 64,
            "bytes": 123456,
        },
        "recording": {
            "start_utc": "2026-07-18T10:00:00Z",
            "end_utc": "2026-07-18T10:00:01Z",
            "duration_ns": "1000000000",
            "timebase": "mcap_log_time",
        },
        "camera_count": 6,
        "camera_mapping_run_id": _uuid(2),
        "camera_mapping_version": "mapping-v1",
        "cameras": [_camera(number) for number in range(1, 7)],
        "ingested_at": "2026-07-18T10:01:00Z",
    }


def _alignment_camera(number: int) -> dict[str, Any]:
    return {
        "source_clock_id": f"clock-{number}",
        "source_timestamp_unit": "ns",
        "derived_drift_ppm": 0.0,
        "residual_p95_ns": "0",
        "max_error_ns": "0",
        "coverage": 1.0,
        "segments": [
            {
                "segment_id": _uuid(200 + number),
                "source_epoch_id": "epoch-0",
                "source_order_start": 0,
                "source_order_end": 30,
                "source_start_ns": "1710000000000000000",
                "source_end_ns": "1710000001000000000",
                "source_anchor_ns": "1710000000000000000",
                "canonical_anchor_ns": "0",
                "rate_numerator": "1",
                "rate_denominator": "1",
                "rounding": "HALF_EVEN",
            }
        ],
        "status": "VALID",
    }


def _alignment_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "alignment_id": _uuid(3),
        "mcap_id": _uuid(1),
        "camera_mapping_run_id": _uuid(2),
        "reference_timebase": "recording_relative_ns",
        "canonical_origin": {
            "source": "mcap_recording_start_in_reference_clock",
            "reference_timestamp_ns": "1710000000000000000",
            "utc": "2026-07-18T10:00:00Z",
        },
        "method": "mcap_log_time",
        "algorithm_version": "alignment-v1",
        "status": "VALID",
        "cameras": {
            camera_id: _alignment_camera(number)
            for number, camera_id in enumerate(CAMERA_IDS, start=1)
        },
        "policy_version": "clock-policy-v1",
        "created_at": "2026-07-18T10:02:00Z",
    }


def _validation_report() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "validation_report_id": _uuid(4),
        "mcap_id": _uuid(1),
        "recording_identity": "b" * 64,
        "source": {
            "uri": "object://recordings/example.mcap",
            "version": None,
            "sha256": "a" * 64,
            "bytes": 123456,
        },
        "mapping_policy": {"version": "mapping-v1", "digest": "c" * 64},
        "verdict": "INCONCLUSIVE",
        "discovered_video_stream_count": 5,
        "mapped_camera_count": 5,
        "errors": [
            {
                "code": "INVALID_CAMERA_COUNT",
                "message": "expected six mapped camera slots",
                "path": "$.mapped_camera_count",
                "camera_id": None,
                "stream_id": None,
            }
        ],
        "validated_at": "2026-07-18T10:01:00Z",
    }


def test_registry_checks_unique_2020_12_documents(registry: SchemaRegistry) -> None:
    assert registry.schema_names == (
        "alignment-manifest",
        "camera-video-export-manifest",
        "camera-video-timestamp-row",
        "canonical-primary-completion-detail",
        "common",
        "current-selection",
        "immutable-node-revision",
        "inference-attempt-selection",
        "inference-intent",
        "logical-node",
        "mcap-manifest",
        "mcap-validation-report",
        "model-inference",
        "orchestrator-enriched-output",
        "parsed-provider-claim-artifact",
        "primary-completion-record",
        "processing-run-node-membership",
        "provider-claim-payload",
        "raw-provider-response-artifact",
        "selected-attempt-output",
        "selection-decision",
    )
    assert len(registry.schema_ids) == len(set(registry.schema_ids))
    assert all(
        registry.get_schema(name)["$schema"] == JSON_SCHEMA_DIALECT
        for name in registry.schema_names
    )


@pytest.mark.parametrize(
    ("schema_name", "payload"),
    [
        ("mcap-manifest", _mcap_manifest()),
        ("alignment-manifest", _alignment_manifest()),
        ("mcap-validation-report", _validation_report()),
    ],
)
def test_valid_payloads_are_accepted(
    registry: SchemaRegistry, schema_name: str, payload: dict[str, Any]
) -> None:
    assert registry.validate(schema_name, payload) is payload


def test_ready_manifest_rejects_mutable_status(registry: SchemaRegistry) -> None:
    payload = _mcap_manifest()
    payload["status"] = "INVALID"

    with pytest.raises(SchemaValidationError) as caught:
        registry.validate("mcap-manifest.schema.json", payload)

    assert caught.value.path == "$.status"
    assert caught.value.validator == "additionalProperties"


def test_missing_camera_has_stable_path(registry: SchemaRegistry) -> None:
    payload = _alignment_manifest()
    del payload["cameras"]["cam_06"]

    with pytest.raises(SchemaValidationError) as caught:
        registry.validate("alignment-manifest", payload)

    assert caught.value.path == "$.cameras.cam_06"
    assert caught.value.validator == "required"


def test_extra_field_has_stable_path(registry: SchemaRegistry) -> None:
    payload = _mcap_manifest()
    payload["unexpected"] = True

    with pytest.raises(SchemaValidationError) as caught:
        registry.validate("mcap-manifest", payload)

    assert caught.value.path == "$.unexpected"
    assert caught.value.validator == "additionalProperties"


def test_numeric_nanoseconds_are_rejected(registry: SchemaRegistry) -> None:
    payload = _mcap_manifest()
    payload["recording"]["duration_ns"] = 1_000_000_000

    with pytest.raises(SchemaValidationError) as caught:
        registry.validate("mcap-manifest", payload)

    assert caught.value.path == "$.recording.duration_ns"


@pytest.mark.parametrize(
    "invalid_value",
    ["01", "-0", "9223372036854775808", "-9223372036854775809"],
)
def test_noncanonical_or_out_of_range_nanoseconds_are_rejected(
    registry: SchemaRegistry, invalid_value: str
) -> None:
    payload = _alignment_manifest()
    payload["canonical_origin"]["reference_timestamp_ns"] = invalid_value

    with pytest.raises(SchemaValidationError) as caught:
        registry.validate("alignment-manifest", payload)

    assert caught.value.path == "$.canonical_origin.reference_timestamp_ns"


def test_camera_array_requires_canonical_order(registry: SchemaRegistry) -> None:
    payload = _mcap_manifest()
    payload["cameras"][0], payload["cameras"][1] = (
        payload["cameras"][1],
        payload["cameras"][0],
    )

    with pytest.raises(SchemaValidationError) as caught:
        registry.validate("mcap-manifest", payload)

    assert caught.value.path == "$.cameras[0].camera_id"


def test_unknown_schema_uses_stable_exception(registry: SchemaRegistry) -> None:
    with pytest.raises(SchemaNotFoundError, match="schema is not registered: unknown"):
        registry.validate("unknown", {})
