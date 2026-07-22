from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from robata.application.canonical.media_quality import (
    LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID,
    LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION,
    FrameQualityObservation,
    FrameTimingEvidence,
    LocalMediaQualityReport,
    LocalQualityFlag,
    build_local_media_quality_report,
    load_registered_local_media_quality_report_document,
    registered_local_media_quality_report_document,
    validate_registered_local_media_quality_report_document,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import (
    SchemaPinMismatchError,
    SchemaRegistry,
    SchemaValidationError,
)


def _report() -> LocalMediaQualityReport:
    interval = NanosecondInterval(start_ns=0, end_ns=1_000_000_000)
    timings = {
        camera_id: (
            FrameTimingEvidence(
                camera_id=camera_id,
                packet_index=0,
                aligned_timestamp_ns=camera_index,
                source_timestamp_ns=1_000_000_000 + camera_index,
                source_sequence=0,
            ),
        )
        for camera_index, camera_id in enumerate(CAMERA_IDS)
    }
    observations = {
        camera_id: (
            FrameQualityObservation(
                camera_id=camera_id,
                packet_index=0,
                aligned_timestamp_ns=camera_index,
                source_timestamp_ns=1_000_000_000 + camera_index,
                grayscale_sha256=f"{camera_index + 1:064x}",
                mean_luma_milli=0,
                black_fraction_ppm=1_000_000,
                overexposed_fraction_ppm=0,
                edge_energy_milli=0,
                frame_delta_milli=None,
                flags=(
                    LocalQualityFlag.OBSERVED_BLACK_LUMA,
                    LocalQualityFlag.PROXY_LOW_EDGE_ENERGY,
                ),
            ),
        )
        for camera_index, camera_id in enumerate(CAMERA_IDS)
    }
    return build_local_media_quality_report(
        requested_max_duration_ns=1_000_000_000,
        recording_duration_ns=2_000_000_000,
        requested_interval=interval,
        timings=timings,
        frame_observations=observations,
    )


def test_report_document_validates_against_embedded_exact_registry_pin() -> None:
    registry = SchemaRegistry()
    registered = registry.resolve_version(
        LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID,
        LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION,
    )

    document = registered_local_media_quality_report_document(_report(), registry)

    assert document["schema_version"] == "1.0"
    assert document["schema_ref"] == registered.ref.model_dump(mode="json")
    assert document["cross_camera_skew"]["p50_ns"] == "5"
    assert document["cross_camera_skew"]["threshold_ns"] == "20000000"
    assert validate_registered_local_media_quality_report_document(document, registry) is document
    assert registry.validate_pinned(registered.ref, document) is document


def test_exact_persisted_report_loader_rejects_noncanonical_or_duplicate_json(
    tmp_path: Path,
) -> None:
    registry = SchemaRegistry()
    document = registered_local_media_quality_report_document(_report(), registry)
    path = tmp_path / "media-quality-report.json"
    exact_bytes = canonical_json_bytes(document)
    path.write_bytes(exact_bytes)

    assert load_registered_local_media_quality_report_document(path, registry) == document

    path.write_bytes(exact_bytes + b"\n")
    with pytest.raises(ValueError, match="exact canonical JSON"):
        load_registered_local_media_quality_report_document(path, registry)

    path.write_bytes(b'{"schema_version":"1.0",' + exact_bytes[1:])
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_registered_local_media_quality_report_document(path, registry)


def test_report_schema_is_closed_and_requires_the_exact_top_level_shape() -> None:
    registry = SchemaRegistry()
    registered = registry.resolve_version(
        LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID,
        LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION,
    )
    schema = registry.get_schema(registered.ref)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert len(schema["properties"]["camera_ledgers"]["prefixItems"]) == 6
    assert registered.entry.owner == "robata-canonical"
    assert registered.entry.projection_version == "local-media-quality-report-v1"
    assert registered.entry.compatibility_mode.value == "NONE"
    assert registered.entry.supported_predecessors == ()


def test_report_validator_rejects_forged_pin_and_structural_drift() -> None:
    registry = SchemaRegistry()
    document = registered_local_media_quality_report_document(_report(), registry)

    forged = deepcopy(document)
    forged["schema_ref"]["sha256"] = "0" * 64
    with pytest.raises(SchemaPinMismatchError):
        validate_registered_local_media_quality_report_document(forged, registry)

    unknown = deepcopy(document)
    unknown["unexpected"] = True
    with pytest.raises(SchemaValidationError, match="additional property"):
        validate_registered_local_media_quality_report_document(unknown, registry)

    numeric_nanoseconds = deepcopy(document)
    numeric_nanoseconds["cross_camera_skew"]["p50_ns"] = 5
    with pytest.raises(SchemaValidationError, match="not valid under any"):
        validate_registered_local_media_quality_report_document(numeric_nanoseconds, registry)


def test_report_validator_rejects_valid_shape_with_tampered_semantics() -> None:
    registry = SchemaRegistry()
    document = registered_local_media_quality_report_document(_report(), registry)

    tampered_report = deepcopy(document)
    tampered_report["window_limited"] = False
    with pytest.raises(ValueError, match="report semantic digest"):
        validate_registered_local_media_quality_report_document(tampered_report, registry)

    tampered_plan = deepcopy(document)
    tampered_plan["supplemental_targets"]["candidate_count"] += 1
    with pytest.raises(ValueError, match="neighbor target plan semantic digest"):
        validate_registered_local_media_quality_report_document(tampered_plan, registry)
