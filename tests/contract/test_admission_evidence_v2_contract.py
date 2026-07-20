from __future__ import annotations

import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

from robata.admission import (
    AdmissionContextResolver,
    AdmittedRecordingContextV2,
    AlignmentAdmissionOutcome,
    PrimaryAdmissionEvaluation,
    PrimaryAdmissionPolicy,
    SourceAdmissionOutcome,
)
from robata.contracts.admission_v2 import (
    ALIGNMENT_MANIFEST_V2_SCHEMA_ID,
    MCAP_READY_MANIFEST_V2_SCHEMA_ID,
    MCAP_VALIDATION_REPORT_V2_SCHEMA_ID,
    AlignmentManifestV2,
    CameraAlignmentV2,
    DecoderProbeEvidenceV2,
    DecoderProbeOutcome,
    EvidenceComponent,
    EvidenceDiagnosticClassification,
    EvidenceDiagnosticSeverity,
    MCAPReadyCameraV2,
    MCAPReadyManifestV2,
    MCAPReadySourceV2,
    MCAPValidationReportV2,
    MCAPValidationSourceV2,
    MCAPValidationVerdictV2,
    ProbedVideoStreamFactV2,
    SchemaSupportStatus,
    SemanticPolicyReference,
    SourceDurabilityEvidenceV2,
    StreamSchemaEvidenceV2,
    ValidationCameraMappingV2,
    ValidationCheckEvidenceV2,
    ValidationCheckOutcome,
    ValidationDiagnosticV2,
    alignment_manifest_v2_semantic_projection,
    compute_camera_mapping_semantic_sha256_v2,
    compute_stream_semantic_sha256_v2,
    mcap_ready_manifest_v2_semantic_projection,
    mcap_validation_report_v2_semantic_projection,
    validate_registered_admission_evidence_v2,
)
from robata.contracts.alignment import (
    AlignmentMethod,
    AlignmentSegment,
    AlignmentStatus,
    CanonicalOrigin,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import semantic_sha256
from robata.contracts.mcap import MCAPReadyRecording
from robata.contracts.schema_registry import SchemaPinMismatchError, SchemaRegistry


def _uuid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012x}"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _component(name: str) -> EvidenceComponent:
    return EvidenceComponent(
        name=name,
        version="1.0",
        code_sha256=_digest(f"{name}-code"),
        configuration_sha256=_digest(f"{name}-config"),
    )


def _policy(name: str) -> SemanticPolicyReference:
    return SemanticPolicyReference(
        version="1.0",
        semantic_sha256=_digest(f"{name}-policy"),
    )


def _schema_ref(registry: SchemaRegistry, schema_id: str):
    return registry.resolve_version(schema_id, "2.0.0").ref


def _valid_validation_fields(registry: SchemaRegistry) -> dict[str, Any]:
    source_content_sha256 = _digest("source")
    mapping_policy = _policy("mapping")
    probe = _component("decoder-probe")
    schema_policy = _policy("source-schema")
    stream_records: list[tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2]] = []
    for ordinal in range(1, 7):
        stream_id = _uuid(100 + ordinal)
        schema_fact = StreamSchemaEvidenceV2(
            stream_id=stream_id,
            stream_semantic_sha256="0" * 64,
            schema_name="foxglove.CompressedImage",
            schema_encoding="protobuf",
            schema_content_sha256=_digest("foxglove-schema"),
            support_status=SchemaSupportStatus.SUPPORTED,
            support_policy=schema_policy,
            diagnostic_ids=(),
        )
        probe_fact = ProbedVideoStreamFactV2(
            stream_id=stream_id,
            stream_semantic_sha256="0" * 64,
            topic=f"/camera/{ordinal}",
            channel_id=ordinal,
            message_encoding="protobuf",
            codec="h264",
            message_count=30,
            first_timestamp_ns=1_710_000_000_000_000_000,
            last_timestamp_ns=1_710_000_001_000_000_000,
            decoder_probe=DecoderProbeEvidenceV2(
                probe=probe,
                outcome=DecoderProbeOutcome.PASSED,
                decoded_frame_count=1,
                decoded_width=1600,
                decoded_height=1300,
                diagnostic_ids=(),
            ),
        )
        stream_digest = compute_stream_semantic_sha256_v2(
            source_content_sha256=source_content_sha256,
            schema_evidence=schema_fact,
            probed_stream_fact=probe_fact,
        )
        stream_records.append(
            (
                schema_fact.model_copy(update={"stream_semantic_sha256": stream_digest}),
                probe_fact.model_copy(update={"stream_semantic_sha256": stream_digest}),
            )
        )

    schema_evidence = tuple(
        sorted(
            (schema_fact for schema_fact, _ in stream_records),
            key=lambda item: (item.stream_semantic_sha256, item.schema_name),
        )
    )
    probed_stream_facts = tuple(
        sorted(
            (probe_fact for _, probe_fact in stream_records),
            key=lambda item: (
                item.stream_semantic_sha256,
                item.topic,
                item.channel_id,
            ),
        )
    )
    camera_mappings = tuple(
        ValidationCameraMappingV2(
            camera_id=camera_id,
            role=f"view-{ordinal}",
            stream_id=schema_fact.stream_id,
            stream_semantic_sha256=schema_fact.stream_semantic_sha256,
        )
        for ordinal, (camera_id, (schema_fact, _)) in enumerate(
            zip(CAMERA_IDS, stream_records, strict=True),
            start=1,
        )
    )
    camera_mapping_semantic_sha256 = compute_camera_mapping_semantic_sha256_v2(
        source_content_sha256=source_content_sha256,
        mapping_policy=mapping_policy,
        camera_mappings=camera_mappings,
    )
    return {
        "schema_version": "2.0",
        "schema_ref": _schema_ref(registry, MCAP_VALIDATION_REPORT_V2_SCHEMA_ID),
        "validation_report_id": _uuid(1),
        "validation_report_semantic_sha256": "0" * 64,
        "mcap_id": _uuid(2),
        "recording_identity": _digest("recording"),
        "source_content_sha256": source_content_sha256,
        "source": MCAPValidationSourceV2(
            uri="object://bucket/source.mcap",
            object_version="version-1",
            sha256=source_content_sha256,
            bytes=12345,
        ),
        "mapping_policy": mapping_policy,
        "camera_mapping_semantic_sha256": camera_mapping_semantic_sha256,
        "validator": _component("mcap-validator"),
        "checks": (
            ValidationCheckEvidenceV2(
                check_id="container",
                check_version="1.0",
                subject="source",
                outcome=ValidationCheckOutcome.PASS,
                diagnostic_ids=(),
            ),
            ValidationCheckEvidenceV2(
                check_id="decoder",
                check_version="1.0",
                subject="all-video-streams",
                outcome=ValidationCheckOutcome.PASS,
                diagnostic_ids=(),
            ),
            ValidationCheckEvidenceV2(
                check_id="mapping",
                check_version="1.0",
                subject="camera-mapping-candidate",
                outcome=ValidationCheckOutcome.PASS,
                diagnostic_ids=(),
            ),
            ValidationCheckEvidenceV2(
                check_id="schema",
                check_version="1.0",
                subject="all-video-streams",
                outcome=ValidationCheckOutcome.PASS,
                diagnostic_ids=(),
            ),
        ),
        "diagnostics": (),
        "schema_evidence": schema_evidence,
        "probed_stream_facts": probed_stream_facts,
        "camera_mappings": camera_mappings,
        "discovered_video_stream_count": 6,
        "mapped_camera_count": 6,
        "verdict": MCAPValidationVerdictV2.VALID,
        "validated_at": "2026-07-19T10:00:00Z",
    }


def _validation_report(
    registry: SchemaRegistry,
    **updates: Any,
) -> MCAPValidationReportV2:
    fields = _valid_validation_fields(registry)
    fields.update(updates)
    draft = MCAPValidationReportV2.model_construct(**fields)
    fields["validation_report_semantic_sha256"] = semantic_sha256(
        mcap_validation_report_v2_semantic_projection(draft)
    )
    return MCAPValidationReportV2.model_validate(fields, strict=True)


def _ready_fields(
    registry: SchemaRegistry,
    report: MCAPValidationReportV2,
) -> dict[str, Any]:
    probes_by_stream = {item.stream_id: item for item in report.probed_stream_facts}
    cameras_list: list[MCAPReadyCameraV2] = []
    for mapping in report.camera_mappings:
        probe_fact = probes_by_stream[mapping.stream_id]
        width = probe_fact.decoder_probe.decoded_width
        height = probe_fact.decoder_probe.decoded_height
        source_start_ns = probe_fact.first_timestamp_ns
        source_end_ns = probe_fact.last_timestamp_ns
        assert width is not None and height is not None
        assert source_start_ns is not None and source_end_ns is not None
        cameras_list.append(
            MCAPReadyCameraV2(
                camera_id=mapping.camera_id,
                role=mapping.role,
                stream_id=mapping.stream_id,
                stream_semantic_sha256=mapping.stream_semantic_sha256,
                topic=probe_fact.topic,
                channel_id=probe_fact.channel_id,
                codec=probe_fact.codec,
                width=width,
                height=height,
                nominal_fps=30.0,
                source_start_ns=source_start_ns,
                source_end_ns=source_end_ns,
                frame_count=probe_fact.message_count,
            )
        )
    cameras = tuple(cameras_list)
    return {
        "schema_version": "2.0",
        "schema_ref": _schema_ref(registry, MCAP_READY_MANIFEST_V2_SCHEMA_ID),
        "ready_manifest_id": _uuid(20),
        "ready_manifest_semantic_sha256": "0" * 64,
        "mcap_id": report.mcap_id,
        "recording_identity": report.recording_identity,
        "source_content_sha256": report.source_content_sha256,
        "source": MCAPReadySourceV2(
            artifact_id=_uuid(21),
            uri="object://bucket/source.mcap",
            object_version="version-1",
            sha256=report.source_content_sha256,
            bytes=12345,
        ),
        "source_durability": SourceDurabilityEvidenceV2(
            verifier=_component("durability-verifier"),
            outcome="PASS",
            verified_sha256=report.source_content_sha256,
            verified_bytes=12345,
        ),
        "validation_report_id": report.validation_report_id,
        "validation_report_semantic_sha256": report.validation_report_semantic_sha256,
        "validation_report_schema_ref": report.schema_ref,
        "camera_mapping_run_id": _uuid(22),
        "camera_mapping_semantic_sha256": report.camera_mapping_semantic_sha256,
        "mapping_policy": report.mapping_policy,
        "admission_policy": _policy("admission"),
        "recording": MCAPReadyRecording(
            start_utc="2026-07-19T09:00:00Z",
            end_utc="2026-07-19T09:00:01Z",
            duration_ns=1_000_000_000,
            timebase="mcap_log_time",
        ),
        "camera_count": 6,
        "cameras": cameras,
        "published_at": "2026-07-19T10:01:00Z",
    }


def _ready_manifest(
    registry: SchemaRegistry,
    report: MCAPValidationReportV2,
    **updates: Any,
) -> MCAPReadyManifestV2:
    fields = _ready_fields(registry, report)
    fields.update(updates)
    draft = MCAPReadyManifestV2.model_construct(**fields)
    fields["ready_manifest_semantic_sha256"] = semantic_sha256(
        mcap_ready_manifest_v2_semantic_projection(draft)
    )
    return MCAPReadyManifestV2.model_validate(fields, strict=True)


def _alignment_fields(
    registry: SchemaRegistry,
    ready: MCAPReadyManifestV2,
) -> dict[str, Any]:
    cameras: dict[str, CameraAlignmentV2] = {}
    for ordinal, camera_id in enumerate(CAMERA_IDS, start=1):
        cameras[camera_id.value] = CameraAlignmentV2(
            source_clock_id=f"clock-{ordinal}",
            source_timestamp_unit="ns",
            derived_drift_ppm=0.0,
            residual_p95_ns=0,
            max_error_ns=0,
            coverage=1.0,
            segments=(
                AlignmentSegment(
                    segment_id=_uuid(300 + ordinal),
                    source_epoch_id="epoch-0",
                    source_order_start=0,
                    source_order_end=30,
                    source_start_ns=1_710_000_000_000_000_000,
                    source_end_ns=1_710_000_001_000_000_000,
                    source_anchor_ns=1_710_000_000_000_000_000,
                    canonical_anchor_ns=0,
                    rate_numerator="1",
                    rate_denominator="1",
                    rounding="HALF_EVEN",
                ),
            ),
            status=AlignmentStatus.VALID,
            stream_id=ready.cameras[ordinal - 1].stream_id,
            stream_semantic_sha256=ready.cameras[ordinal - 1].stream_semantic_sha256,
        )
    return {
        "schema_version": "2.0",
        "schema_ref": _schema_ref(registry, ALIGNMENT_MANIFEST_V2_SCHEMA_ID),
        "alignment_id": _uuid(30),
        "alignment_semantic_sha256": "0" * 64,
        "mcap_id": ready.mcap_id,
        "recording_identity": ready.recording_identity,
        "source_content_sha256": ready.source_content_sha256,
        "ready_manifest_id": ready.ready_manifest_id,
        "ready_manifest_semantic_sha256": ready.ready_manifest_semantic_sha256,
        "ready_manifest_schema_ref": ready.schema_ref,
        "camera_mapping_run_id": ready.camera_mapping_run_id,
        "camera_mapping_semantic_sha256": ready.camera_mapping_semantic_sha256,
        "reference_timebase": "recording_relative_ns",
        "canonical_origin": CanonicalOrigin(
            source="mcap_recording_start_in_reference_clock",
            reference_timestamp_ns=1_710_000_000_000_000_000,
            utc="2026-07-19T09:00:00Z",
        ),
        "method": AlignmentMethod.MCAP_LOG_TIME,
        "algorithm": _component("offline-rational-alignment"),
        "status": AlignmentStatus.VALID,
        "cameras": cameras,
        "policy": _policy("clock-alignment"),
        "validator": _component("alignment-validator"),
        "checks": (
            ValidationCheckEvidenceV2(
                check_id="residual-and-coverage",
                check_version="1.0",
                subject="all-camera-transforms",
                outcome=ValidationCheckOutcome.PASS,
                diagnostic_ids=(),
            ),
        ),
        "diagnostics": (),
        "created_at": "2026-07-19T10:02:00Z",
    }


def _alignment_manifest(
    registry: SchemaRegistry,
    ready: MCAPReadyManifestV2,
    **updates: Any,
) -> AlignmentManifestV2:
    fields = _alignment_fields(registry, ready)
    fields.update(updates)
    draft = AlignmentManifestV2.model_construct(**fields)
    fields["alignment_semantic_sha256"] = semantic_sha256(
        alignment_manifest_v2_semantic_projection(draft)
    )
    return AlignmentManifestV2.model_validate(fields, strict=True)


def test_registered_v2_validation_ready_and_alignment_payloads_are_accepted() -> None:
    registry = SchemaRegistry()
    report = _validation_report(registry)
    ready = _ready_manifest(registry, report)
    alignment = _alignment_manifest(registry, ready)

    assert validate_registered_admission_evidence_v2(report, registry) is report
    assert validate_registered_admission_evidence_v2(ready, registry) is ready
    assert validate_registered_admission_evidence_v2(alignment, registry) is alignment


def test_v2_admission_context_cross_binds_registered_evidence() -> None:
    registry = SchemaRegistry()
    report = _validation_report(registry)
    ready = _ready_manifest(registry, report)
    alignment = _alignment_manifest(registry, ready)
    policy = PrimaryAdmissionPolicy.create(
        version="primary-v2",
        admissible_alignment_outcomes=(AlignmentAdmissionOutcome.VALID,),
    )
    evaluation = PrimaryAdmissionEvaluation(
        recording_identity=report.recording_identity,
        ready_manifest_id=ready.ready_manifest_id,
        ready_manifest_semantic_sha256=ready.ready_manifest_semantic_sha256,
        source_outcome=SourceAdmissionOutcome.READY,
        alignment_outcome=AlignmentAdmissionOutcome.VALID,
        alignment_id=alignment.alignment_id,
        alignment_semantic_sha256=alignment.alignment_semantic_sha256,
        policy_version=policy.version,
        policy_sha256=policy.semantic_sha256,
        admissible=True,
        reason_code="ADMISSIBLE",
    )

    context = AdmissionContextResolver().resolve_v2(
        evaluation=evaluation,
        policy=policy,
        validation_report=report,
        ready_manifest=ready,
        alignment_manifest=alignment,
        registry=registry,
    )

    assert isinstance(context, AdmittedRecordingContextV2)
    assert context.recording_identity == report.recording_identity
    assert context.camera_mapping_semantic_sha256 == ready.camera_mapping_semantic_sha256

    with pytest.raises(ValueError, match="v2 camera mapping digest"):
        AdmittedRecordingContextV2.model_validate(
            context.model_dump(mode="python")
            | {"camera_mapping_semantic_sha256": _digest("forged-mapping")}
        )


def test_exact_schema_quartet_is_resolved_before_wire_validation() -> None:
    registry = SchemaRegistry()
    report = _validation_report(registry)
    forged_ref = report.schema_ref.model_copy(update={"sha256": "0" * 64})
    forged = report.model_copy(update={"schema_ref": forged_ref})

    with pytest.raises(SchemaPinMismatchError):
        validate_registered_admission_evidence_v2(forged, registry)


def test_validation_identity_excludes_alias_row_ids_and_observation_clock() -> None:
    registry = SchemaRegistry()
    first = _validation_report(registry)
    moved_source = first.source.model_copy(
        update={"uri": "object://moved/source.mcap", "object_version": "version-2"}
    )
    second = _validation_report(
        registry,
        validation_report_id=_uuid(900),
        mcap_id=_uuid(901),
        source=moved_source,
        validated_at="2026-07-20T10:00:00Z",
    )

    assert second.validation_report_semantic_sha256 == first.validation_report_semantic_sha256
    assert mcap_validation_report_v2_semantic_projection(second) == (
        mcap_validation_report_v2_semantic_projection(first)
    )

    fields = _valid_validation_fields(registry)
    replacement_ids = {
        item.stream_id: _uuid(700 + ordinal)
        for ordinal, item in enumerate(fields["schema_evidence"], start=1)
    }
    fields["schema_evidence"] = tuple(
        item.model_copy(update={"stream_id": replacement_ids[item.stream_id]})
        for item in fields["schema_evidence"]
    )
    fields["probed_stream_facts"] = tuple(
        item.model_copy(update={"stream_id": replacement_ids[item.stream_id]})
        for item in fields["probed_stream_facts"]
    )
    fields["camera_mappings"] = tuple(
        item.model_copy(update={"stream_id": replacement_ids[item.stream_id]})
        for item in fields["camera_mappings"]
    )
    reallocated = _validation_report(registry, **fields)
    assert reallocated.validation_report_semantic_sha256 == (
        first.validation_report_semantic_sha256
    )


def test_infrastructure_probe_failure_is_inconclusive_not_invalid() -> None:
    registry = SchemaRegistry()
    fields = _valid_validation_fields(registry)
    probes = list(fields["probed_stream_facts"])
    first = probes[0]
    diagnostic = ValidationDiagnosticV2(
        diagnostic_id="decoder-unavailable",
        code="DECODER_INFRASTRUCTURE_UNAVAILABLE",
        message="decoder process unavailable",
        classification=EvidenceDiagnosticClassification.INFRASTRUCTURE,
        severity=EvidenceDiagnosticSeverity.ERROR,
        path="$.probed_stream_facts[0].decoder_probe",
        camera_id=None,
        stream_id=first.stream_id,
    )
    probes[0] = first.model_copy(
        update={
            "decoder_probe": DecoderProbeEvidenceV2(
                probe=_component("decoder-probe"),
                outcome=DecoderProbeOutcome.INCONCLUSIVE,
                decoded_frame_count=0,
                decoded_width=None,
                decoded_height=None,
                diagnostic_ids=(diagnostic.diagnostic_id,),
            )
        }
    )
    fields.update(
        {
            "checks": (
                ValidationCheckEvidenceV2(
                    check_id="decoder",
                    check_version="1.0",
                    subject="all-video-streams",
                    outcome=ValidationCheckOutcome.INCONCLUSIVE,
                    diagnostic_ids=(diagnostic.diagnostic_id,),
                ),
            ),
            "diagnostics": (diagnostic,),
            "probed_stream_facts": tuple(probes),
            "verdict": MCAPValidationVerdictV2.INCONCLUSIVE,
        }
    )
    report = _validation_report(registry, **fields)
    assert report.verdict is MCAPValidationVerdictV2.INCONCLUSIVE

    invalid_fields = dict(fields)
    invalid_fields["verdict"] = MCAPValidationVerdictV2.INVALID
    with pytest.raises(ValidationError, match="verdict does not match"):
        _validation_report(registry, **invalid_fields)


def test_ready_has_no_mutable_status_and_binds_selected_report_digest() -> None:
    registry = SchemaRegistry()
    report = _validation_report(registry)
    ready = _ready_manifest(registry, report)
    payload = ready.model_dump(mode="json")
    payload["status"] = "INVALID"

    with pytest.raises(ValueError, match="status"):
        registry.validate_pinned(ready.schema_ref, payload)
    stale_fields = _ready_fields(registry, report)
    stale_fields["ready_manifest_semantic_sha256"] = ready.ready_manifest_semantic_sha256
    stale_fields["validation_report_semantic_sha256"] = _digest("other-report")
    with pytest.raises(ValidationError, match="ready_manifest_semantic_sha256"):
        MCAPReadyManifestV2.model_validate(stale_fields, strict=True)


def test_ready_and_alignment_semantics_exclude_association_ids_and_publish_times() -> None:
    registry = SchemaRegistry()
    report = _validation_report(registry)
    first_ready = _ready_manifest(registry, report)
    moved_source = first_ready.source.model_copy(
        update={
            "artifact_id": _uuid(800),
            "uri": "object://moved/source.mcap",
            "object_version": "version-2",
        }
    )
    second_ready = _ready_manifest(
        registry,
        report,
        ready_manifest_id=_uuid(801),
        mcap_id=_uuid(802),
        source=moved_source,
        validation_report_id=_uuid(803),
        camera_mapping_run_id=_uuid(804),
        published_at="2026-07-20T10:01:00Z",
    )
    assert second_ready.ready_manifest_semantic_sha256 == (
        first_ready.ready_manifest_semantic_sha256
    )

    first_alignment = _alignment_manifest(registry, first_ready)
    cameras = {
        camera_id: camera.model_copy(
            update={
                "stream_id": _uuid(850 + ordinal),
                "segments": tuple(
                    segment.model_copy(update={"segment_id": _uuid(860 + ordinal)})
                    for segment in camera.segments
                ),
            }
        )
        for ordinal, (camera_id, camera) in enumerate(
            first_alignment.cameras.items(),
            start=1,
        )
    }
    second_alignment = _alignment_manifest(
        registry,
        first_ready,
        alignment_id=_uuid(870),
        mcap_id=_uuid(871),
        ready_manifest_id=_uuid(872),
        camera_mapping_run_id=_uuid(873),
        cameras=cameras,
        created_at="2026-07-20T10:02:00Z",
    )
    assert second_alignment.alignment_semantic_sha256 == (first_alignment.alignment_semantic_sha256)


def test_semantic_input_change_rejects_a_stale_digest() -> None:
    registry = SchemaRegistry()
    report = _validation_report(registry)
    fields = _valid_validation_fields(registry)
    fields["validation_report_semantic_sha256"] = report.validation_report_semantic_sha256
    fields["camera_mapping_semantic_sha256"] = _digest("changed-mapping")

    with pytest.raises(ValidationError, match="camera_mapping_semantic_sha256"):
        MCAPValidationReportV2.model_validate(fields, strict=True)
