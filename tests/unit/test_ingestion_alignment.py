from __future__ import annotations

from pathlib import Path

import pytest

from robata.alignment import AlignmentError, AlignmentService, AlignmentStatus
from robata.contracts.cameras import SixCameraMap
from robata.contracts.mcap import MCAPValidationVerdict
from robata.contracts.schema_registry import default_schema_registry
from robata.ingestion import (
    ExactTopicMappingPolicy,
    MCAPIngestionService,
    MCAPValidator,
    StreamIndexer,
)
from robata.ingestion.indexer import IndexingCapabilityError
from robata.ingestion.service import IngestionCapabilityError, IngestionStateError
from robata.ports.ingestion import (
    ChannelInspection,
    DecoderProbeResult,
    McapInspection,
)

MCAP_ID = "00000000-0000-4000-8000-000000000001"
MAPPING_RUN_ID = "00000000-0000-4000-8000-000000000002"


class _Probe:
    def probe(self, source: Path, channel: ChannelInspection) -> DecoderProbeResult:
        return DecoderProbeResult(
            topic=channel.topic,
            codec="h264",
            success=True,
            width=640,
            height=480,
            first_decoded_timestamp_ns=channel.first_message_time_ns,
            messages_examined=2,
            decoded_frames=1,
            failures=(),
        )


class _Inspector:
    def __init__(self, inspection: McapInspection) -> None:
        self.inspection = inspection

    def inspect(self, source: Path) -> McapInspection:
        return self.inspection


class _UnavailableProbe:
    def probe(self, source: Path, channel: ChannelInspection) -> DecoderProbeResult:
        raise OSError("decoder infrastructure unavailable")


def _inspection() -> McapInspection:
    channels = tuple(
        ChannelInspection(
            channel_id=index,
            topic=f"/camera/{index}",
            schema_name="foxglove.CompressedImage",
            message_encoding="protobuf",
            message_count=3,
            first_message_time_ns=1_000_000_000,
            last_message_time_ns=3_000_000_000,
            monotonic=True,
            codec="h264",
            frame_id=None,
        )
        for index in range(1, 7)
    )
    return McapInspection(
        source=Path("sample.mcap"),
        source_size_bytes=5,
        source_sha256="0" * 64,
        header_profile="fixture",
        header_library="fixture",
        summary_available=True,
        channel_count=6,
        message_count=18,
        first_message_time_ns=1_000_000_000,
        last_message_time_ns=3_000_000_000,
        channels=channels,
    )


def _policy() -> ExactTopicMappingPolicy:
    topics = SixCameraMap.model_validate(
        {f"cam_{index:02d}": f"/camera/{index}" for index in range(1, 7)},
        strict=True,
    )
    return ExactTopicMappingPolicy(topics, version="fixture-v1")


def test_stream_indexer_is_deterministic_and_does_not_fabricate_frames() -> None:
    inspection = _inspection()
    first = StreamIndexer(mapping_policy=_policy(), decoder_probe=_Probe()).index_streams(
        "mcap-1", inspection
    )
    second = StreamIndexer(mapping_policy=_policy(), decoder_probe=_Probe()).index_streams(
        "mcap-1", inspection
    )

    assert first.status == "INDEXED"
    assert first.frame_indexes == ()
    assert first.camera_mapping_run.mapping_run_id == second.camera_mapping_run.mapping_run_id
    assert tuple(item.stream_id for item in first.stream_index) == tuple(
        item.stream_id for item in second.stream_index
    )


def test_stream_and_mapping_identity_is_content_based_not_alias_based() -> None:
    inspection = _inspection()
    indexer = StreamIndexer(mapping_policy=_policy(), decoder_probe=_Probe())

    first = indexer.index_streams("alias-mcap-1", inspection)
    second = indexer.index_streams("alias-mcap-2", inspection)

    assert tuple(item.stream_id for item in first.stream_index) == tuple(
        item.stream_id for item in second.stream_index
    )
    assert first.camera_mapping_run.mapping_run_id == second.camera_mapping_run.mapping_run_id
    assert first.camera_mapping_run.mcap_id == "alias-mcap-1"
    assert second.camera_mapping_run.mcap_id == "alias-mcap-2"


def test_stream_indexer_fails_closed_without_policy_or_probe() -> None:
    with pytest.raises(IndexingCapabilityError):
        StreamIndexer(decoder_probe=_Probe()).index_streams("mcap-1", _inspection())
    with pytest.raises(IndexingCapabilityError):
        StreamIndexer(mapping_policy=_policy()).index_streams("mcap-1", _inspection())


def test_ingestion_ready_requires_durability_evidence() -> None:
    inspection = _inspection()
    probe = _Probe()
    service = MCAPIngestionService(
        source_hasher=lambda _: ("0" * 64, 5),
        inspector=_Inspector(inspection),
        validator=MCAPValidator(decoder_probe=probe),
        indexer=StreamIndexer(mapping_policy=_policy(), decoder_probe=probe),
        source_resolver=lambda _: Path("sample.mcap"),
    )
    recording = service.discover("sample.mcap", 5)
    recording = service.hash_source(recording)
    recording = service.inspect(recording)
    recording = service.validate_mcap(recording)

    assert recording.status.value == "VALIDATING"
    report = service.publish_validation_report(recording)
    with pytest.raises(IngestionCapabilityError):
        service.publish_ready_manifest(
            recording,
            selected_validation_report=report,
        )
    assert service.get_state().value == "VALIDATING"
    assert service.ready_recording is None


def test_ingestion_publishes_registered_report_then_ready_manifest() -> None:
    inspection = _inspection()
    probe = _Probe()
    service = MCAPIngestionService(
        source_hasher=lambda _: ("0" * 64, 5),
        inspector=_Inspector(inspection),
        validator=MCAPValidator(decoder_probe=probe),
        indexer=StreamIndexer(mapping_policy=_policy(), decoder_probe=probe),
        source_resolver=lambda _: Path("sample.mcap"),
        source_durable_check=lambda _: True,
    )
    recording = service.discover("sample.mcap", 5)
    recording = service.hash_source(recording)
    recording = service.inspect(recording)
    recording = service.validate_mcap(recording)

    report = service.publish_validation_report(recording)
    manifest = service.publish_ready_manifest(
        recording,
        selected_validation_report=report,
    )
    ready_recording = service.ready_recording
    registry = default_schema_registry()
    report_ref = registry.resolve_version(
        "https://schemas.robata.dev/mcap-validation-report", "1.0.0"
    ).ref
    manifest_ref = registry.resolve_version("https://schemas.robata.dev/mcap-manifest", "1.0.0").ref
    report_payload = report.model_dump(mode="json")
    manifest_payload = manifest.model_dump(mode="json")

    assert report.verdict is MCAPValidationVerdict.VALID
    assert ready_recording is not None
    assert ready_recording.status.value == "READY"
    assert service.get_state().value == "READY"
    assert report.errors == ()
    assert registry.validate_pinned(report_ref, report_payload) is report_payload
    assert registry.validate_pinned(manifest_ref, manifest_payload) is manifest_payload
    assert "status" not in manifest_payload
    assert manifest.validation_report_id == report.validation_report_id
    assert [camera["camera_id"] for camera in manifest_payload["cameras"]] == [
        f"cam_{index:02d}" for index in range(1, 7)
    ]
    assert all("role" in camera for camera in manifest_payload["cameras"])
    assert (
        service.publish_ready_manifest(
            recording,
            selected_validation_report=report,
        )
        == manifest
    )

    forged_selection = report.model_copy(
        update={"validation_report_id": "00000000-0000-4000-8000-000000000099"}
    )
    with pytest.raises(IngestionStateError, match=r"not published by this run|prior evidence"):
        service.publish_ready_manifest(
            recording,
            selected_validation_report=forged_selection,
        )


def test_infrastructure_failure_publishes_inconclusive_report_only() -> None:
    inspection = _inspection()
    probe = _UnavailableProbe()
    service = MCAPIngestionService(
        source_hasher=lambda _: ("0" * 64, 5),
        inspector=_Inspector(inspection),
        validator=MCAPValidator(decoder_probe=probe),
        indexer=StreamIndexer(mapping_policy=_policy(), decoder_probe=probe),
        source_resolver=lambda _: Path("sample.mcap"),
        source_durable_check=lambda _: True,
        max_retries=0,
    )
    recording = service.discover("sample.mcap", 5)
    recording = service.hash_source(recording)
    recording = service.inspect(recording)
    recording = service.validate_mcap(recording)

    assert recording.status.value == "FAILED"
    report = service.publish_validation_report(recording)
    assert report.verdict is MCAPValidationVerdict.INCONCLUSIVE
    assert report.errors[0].code == "SOURCE_IO_ERROR"
    with pytest.raises(IngestionStateError):
        service.publish_ready_manifest(
            recording,
            selected_validation_report=report,
        )


def test_inconclusive_validation_can_reenter_validation_after_retry_wait() -> None:
    inspection = _inspection()
    probe = _UnavailableProbe()
    service = MCAPIngestionService(
        source_hasher=lambda _: ("0" * 64, 5),
        inspector=_Inspector(inspection),
        validator=MCAPValidator(decoder_probe=probe),
        indexer=StreamIndexer(mapping_policy=_policy(), decoder_probe=probe),
        source_resolver=lambda _: Path("sample.mcap"),
        max_retries=1,
    )
    recording = service.discover("sample.mcap", 5)
    recording = service.hash_source(recording)
    recording = service.inspect(recording)
    recording = service.validate_mcap(recording)

    assert recording.status.value == "RETRY_WAIT"
    assert service.publish_validation_report(recording).verdict is (
        MCAPValidationVerdict.INCONCLUSIVE
    )
    recording = service.retry_or_fail(recording)
    assert recording.status.value == "VALIDATING"
    recording = service.validate_mcap(recording)
    assert recording.status.value == "FAILED"


def test_alignment_requires_explicit_method_approval() -> None:
    timestamps = SixCameraMap.model_validate(
        {f"cam_{index:02d}": [1_000, 2_000, 3_000] for index in range(1, 7)},
        strict=True,
    )
    unverified = AlignmentService("policy-v1").align_recording(
        mcap_id=MCAP_ID,
        camera_mapping_run_id=MAPPING_RUN_ID,
        stream_timestamps=timestamps,
    )
    verified = AlignmentService("policy-v1", verified_methods=("mcap_log_time",)).align_recording(
        mcap_id=MCAP_ID,
        camera_mapping_run_id=MAPPING_RUN_ID,
        stream_timestamps=timestamps,
    )

    assert unverified.status is AlignmentStatus.UNVERIFIED
    assert verified.status is AlignmentStatus.VALID


def test_alignment_identity_excludes_alias_and_row_ids() -> None:
    timestamps = SixCameraMap.model_validate(
        {f"cam_{index:02d}": [1_000, 2_000, 3_000] for index in range(1, 7)},
        strict=True,
    )
    service = AlignmentService("policy-v1", verified_methods=("mcap_log_time",))
    first = service.align_recording(
        mcap_id="00000000-0000-4000-8000-000000000011",
        camera_mapping_run_id="00000000-0000-4000-8000-000000000012",
        source_content_sha256="a" * 64,
        camera_mapping_semantic_sha256="b" * 64,
        stream_timestamps=timestamps,
    )
    replay = service.align_recording(
        mcap_id="00000000-0000-4000-8000-000000000013",
        camera_mapping_run_id="00000000-0000-4000-8000-000000000014",
        source_content_sha256="a" * 64,
        camera_mapping_semantic_sha256="b" * 64,
        stream_timestamps=timestamps,
    )
    changed_source = service.align_recording(
        mcap_id="00000000-0000-4000-8000-000000000015",
        camera_mapping_run_id="00000000-0000-4000-8000-000000000016",
        source_content_sha256="c" * 64,
        camera_mapping_semantic_sha256="b" * 64,
        stream_timestamps=timestamps,
    )

    assert first.alignment_id == replay.alignment_id
    assert first.alignment_id != changed_source.alignment_id
    with pytest.raises(ValueError, match="must be supplied together"):
        service.align_recording(
            mcap_id=MCAP_ID,
            camera_mapping_run_id=MAPPING_RUN_ID,
            source_content_sha256="a" * 64,
            stream_timestamps=timestamps,
        )


def test_alignment_clock_reset_is_explicitly_degraded() -> None:
    timestamps = SixCameraMap.model_validate(
        {
            "cam_01": [1_000, 2_000, 3_000],
            "cam_02": [1_000, 2_000, 500],
            "cam_03": [1_000, 2_000, 3_000],
            "cam_04": [1_000, 2_000, 3_000],
            "cam_05": [1_000, 2_000, 3_000],
            "cam_06": [1_000, 2_000, 3_000],
        },
        strict=True,
    )
    run = AlignmentService("policy-v1", verified_methods=("mcap_log_time",)).align_recording(
        mcap_id=MCAP_ID,
        camera_mapping_run_id=MAPPING_RUN_ID,
        stream_timestamps=timestamps,
    )

    assert run.status is AlignmentStatus.DEGRADED


def test_alignment_publishes_registered_manifest_body() -> None:
    timestamps = SixCameraMap.model_validate(
        {f"cam_{index:02d}": [1_000, 2_000, 3_000] for index in range(1, 7)},
        strict=True,
    )
    service = AlignmentService("policy-v1", verified_methods=("mcap_log_time",))
    run = service.align_recording(
        mcap_id=MCAP_ID,
        camera_mapping_run_id=MAPPING_RUN_ID,
        stream_timestamps=timestamps,
        recording_start_utc="2026-07-19T12:00:00Z",
    )
    validation = service.validate_alignment(
        alignment_run=run,
        source_timestamps=timestamps,
    )
    published = service.publish_alignment_manifest(
        alignment_run=run,
        validation_result=validation,
        projections=[],
    )
    payload = published.model_dump(mode="json")
    registry = default_schema_registry()
    schema_ref = registry.resolve_version(
        "https://schemas.robata.dev/alignment-manifest", "1.0.0"
    ).ref

    assert registry.validate_pinned(schema_ref, payload) is payload
    assert payload["canonical_origin"] == {
        "source": "mcap_recording_start_in_reference_clock",
        "reference_timestamp_ns": "1000",
        "utc": "2026-07-19T12:00:00Z",
    }
    assert all("camera_id" not in camera for camera in payload["cameras"].values())

    first_metric = validation.per_camera[0]
    inconsistent = validation.model_copy(
        update={
            "per_camera": (
                first_metric.model_copy(
                    update={"residual_p95_ns": first_metric.residual_p95_ns + 1}
                ),
                *validation.per_camera[1:],
            )
        }
    )
    with pytest.raises(AlignmentError, match="manifest does not match validation evidence"):
        service.publish_alignment_manifest(
            alignment_run=run,
            validation_result=inconsistent,
            projections=[],
        )
