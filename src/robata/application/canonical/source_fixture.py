"""Development source fixture that produces canonical admitted inputs.

The fixture contains only source-shaped observations: six camera streams,
source timestamps, and encoded frame bytes.  Admission evidence, alignment,
frame indexes, and materialized artifact facts are derived at load time.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.admission.context import AdmissionContextResolver, AdmittedRecordingContextV2
from robata.admission.ledger import (
    AlignmentAdmissionOutcome,
    PrimaryAdmissionEvaluation,
    PrimaryAdmissionPolicy,
    SourceAdmissionOutcome,
)
from robata.contracts.admission_v2 import (
    ADMISSION_EVIDENCE_V2_SCHEMA_VERSION,
    ALIGNMENT_MANIFEST_V2_SCHEMA_ID,
    MCAP_READY_MANIFEST_V2_SCHEMA_ID,
    MCAP_VALIDATION_REPORT_V2_SCHEMA_ID,
    AlignmentManifestV2,
    CameraAlignmentV2,
    DecoderProbeEvidenceV2,
    DecoderProbeOutcome,
    EvidenceComponent,
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
    alignment_manifest_v2_semantic_projection,
    compute_camera_mapping_semantic_sha256_v2,
    compute_stream_semantic_sha256_v2,
    mcap_ready_manifest_v2_semantic_projection,
    mcap_validation_report_v2_semantic_projection,
)
from robata.contracts.alignment import (
    AlignmentMethod,
    AlignmentSegment,
    AlignmentStatus,
    CanonicalOrigin,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import (
    INT64_MAX,
    NanosecondInterval,
    Nanoseconds,
    StrictModel,
)
from robata.contracts.hashing import exact_bytes_sha256, recording_identity, semantic_sha256
from robata.contracts.mcap import MCAPReadyRecording
from robata.contracts.sampling_plan import FrameBudget, OverflowPolicy, SamplingPlan
from robata.contracts.schema_registry import SchemaRegistry
from robata.sampling.materializer import (
    CameraSourceFrameIndex,
    CanonicalSixCameraFrameIndex,
    FrameAlignmentProjectionFact,
    IndexedSourceFrame,
    MaterializedArtifactManifest,
    MaterializedFrameArtifactFact,
)

FIXTURE_SOURCE_POLICY_VERSION = "canonical-source-fixture-v1"
FIXTURE_SAMPLING_POLICY_VERSION = "canonical-development-sampling-v1"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
Rfc3339Timestamp = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
            r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
        ),
    ),
]


class CanonicalSourceFixtureError(ValueError):
    """The development source fixture cannot produce canonical inputs."""


class FixtureSourceFrame(StrictModel):
    """One encoded source frame observed at an exact source timestamp."""

    source_timestamp_ns: Nanoseconds
    payload_base64: NonEmptyString


class FixtureCameraStream(StrictModel):
    """Source facts for one canonical camera stream."""

    camera_id: CameraId
    role: NonEmptyString
    topic: NonEmptyString
    channel_id: NonNegativeInt
    width: PositiveInt
    height: PositiveInt
    frames: Annotated[tuple[FixtureSourceFrame, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def validate_frame_order(self) -> Self:
        timestamps = tuple(frame.source_timestamp_ns for frame in self.frames)
        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError("fixture frame timestamps must be unique and strictly increasing")
        return self


class CanonicalSourceFixture(StrictModel):
    """A source-shaped six-camera fixture with no derived pipeline state."""

    schema_version: Literal["1.0"]
    source_clock_id: NonEmptyString
    recording_start_utc: Rfc3339Timestamp | None = None
    cameras: SixCameraMap[FixtureCameraStream]

    @model_validator(mode="after")
    def validate_camera_bindings(self) -> Self:
        topics: set[str] = set()
        channel_ids: set[int] = set()
        for camera_id, camera in self.cameras.items():
            if camera.camera_id is not camera_id:
                raise ValueError("fixture camera key must match nested camera_id")
            if camera.topic in topics or camera.channel_id in channel_ids:
                raise ValueError("fixture camera topics and channel IDs must be unique")
            topics.add(camera.topic)
            channel_ids.add(camera.channel_id)
            if camera.frames[-1].source_timestamp_ns == INT64_MAX:
                raise ValueError("fixture timestamps must allow a half-open source interval")
        origin_ns = min(camera.frames[0].source_timestamp_ns for camera in self.cameras.values())
        end_ns = max(camera.frames[-1].source_timestamp_ns for camera in self.cameras.values()) + 1
        if end_ns - origin_ns > INT64_MAX:
            raise ValueError("fixture canonical duration must fit signed int64 nanoseconds")
        return self


@dataclass(frozen=True, slots=True)
class CanonicalSourceBundle:
    """Canonical runner inputs derived from one immutable fixture source."""

    source_content_sha256: str
    admitted_context: AdmittedRecordingContextV2
    requested_interval: NanosecondInterval
    sampling_plan: SamplingPlan
    frame_index: CanonicalSixCameraFrameIndex
    _artifacts: Mapping[tuple[CameraId, str], MaterializedFrameArtifactFact]

    def resolve_artifact(
        self,
        camera_id: CameraId,
        frame: IndexedSourceFrame,
    ) -> MaterializedFrameArtifactFact | None:
        """Resolve exact fixture bytes for one indexed source frame."""

        return self._artifacts.get((camera_id, frame.source_frame_id))


def load_canonical_source_fixture(
    path: Path,
    *,
    schema_registry: SchemaRegistry | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CanonicalSourceBundle:
    """Load source facts and derive every canonical pre-run input."""

    source_path = Path(path)
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise CanonicalSourceFixtureError(f"cannot read fixture source: {exc}") from exc
    if not source_bytes:
        raise CanonicalSourceFixtureError("fixture source must not be empty")
    try:
        fixture = CanonicalSourceFixture.model_validate_json(source_bytes, strict=True)
    except ValidationError as exc:
        raise CanonicalSourceFixtureError(f"invalid fixture source: {exc}") from exc

    source_content_sha256 = exact_bytes_sha256(source_bytes)
    active_clock = clock or (lambda: datetime.now(tz=UTC))
    observed_at = _rfc3339(active_clock())
    bundle = _build_bundle(
        fixture=fixture,
        source_uri=source_path.resolve().as_uri(),
        source_content_sha256=source_content_sha256,
        source_bytes=len(source_bytes),
        schema_registry=schema_registry or SchemaRegistry(),
        observed_at=observed_at,
    )
    try:
        verified_source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise CanonicalSourceFixtureError(
            f"cannot verify fixture source after derivation: {exc}"
        ) from exc
    if verified_source_bytes != source_bytes:
        raise CanonicalSourceFixtureError("fixture source changed during derivation")
    return bundle


def _build_bundle(
    *,
    fixture: CanonicalSourceFixture,
    source_uri: str,
    source_content_sha256: str,
    source_bytes: int,
    schema_registry: SchemaRegistry,
    observed_at: str,
) -> CanonicalSourceBundle:
    decoded_payloads = _decode_payloads(fixture)
    mapping_policy = _policy("fixture-camera-mapping")
    admission_policy_ref = _policy("fixture-source-admission")
    alignment_policy_ref = _policy("fixture-clock-alignment")
    schema_support_policy = _policy("fixture-frame-schema")
    validator = _component("fixture-source-validator")
    decoder_probe = _component("fixture-png-probe")
    durability_verifier = _component("fixture-file-durability")
    alignment_algorithm = _component("fixture-rational-alignment")
    alignment_validator = _component("fixture-alignment-validator")

    recording_id = recording_identity(
        "robata-canonical-development-fixture-v1",
        source_content_sha256,
    )
    mcap_id = _stable_uuid("fixture-source", source_content_sha256)

    stream_records: dict[
        CameraId,
        tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2],
    ] = {}
    for camera_id in CAMERA_IDS:
        camera = fixture.cameras[camera_id]
        stream_id = _stable_uuid("fixture-stream", f"{source_content_sha256}:{camera_id.value}")
        schema_fact = StreamSchemaEvidenceV2(
            stream_id=stream_id,
            stream_semantic_sha256="0" * 64,
            schema_name="robata.fixture.EncodedImage",
            schema_encoding="json-base64",
            schema_content_sha256=semantic_sha256(
                {
                    "name": "robata.fixture.EncodedImage",
                    "version": FIXTURE_SOURCE_POLICY_VERSION,
                }
            ),
            support_status=SchemaSupportStatus.SUPPORTED,
            support_policy=schema_support_policy,
            diagnostic_ids=(),
        )
        probe_fact = ProbedVideoStreamFactV2(
            stream_id=stream_id,
            stream_semantic_sha256="0" * 64,
            topic=camera.topic,
            channel_id=camera.channel_id,
            message_encoding="fixture-json",
            codec="png",
            message_count=len(camera.frames),
            first_timestamp_ns=camera.frames[0].source_timestamp_ns,
            last_timestamp_ns=camera.frames[-1].source_timestamp_ns,
            decoder_probe=DecoderProbeEvidenceV2(
                probe=decoder_probe,
                outcome=DecoderProbeOutcome.PASSED,
                decoded_frame_count=len(camera.frames),
                decoded_width=camera.width,
                decoded_height=camera.height,
                diagnostic_ids=(),
            ),
        )
        stream_digest = compute_stream_semantic_sha256_v2(
            source_content_sha256=source_content_sha256,
            schema_evidence=schema_fact,
            probed_stream_fact=probe_fact,
        )
        stream_records[camera_id] = (
            schema_fact.model_copy(update={"stream_semantic_sha256": stream_digest}),
            probe_fact.model_copy(update={"stream_semantic_sha256": stream_digest}),
        )

    camera_mappings = tuple(
        ValidationCameraMappingV2(
            camera_id=camera_id,
            role=fixture.cameras[camera_id].role,
            stream_id=stream_records[camera_id][0].stream_id,
            stream_semantic_sha256=stream_records[camera_id][0].stream_semantic_sha256,
        )
        for camera_id in CAMERA_IDS
    )
    mapping_digest = compute_camera_mapping_semantic_sha256_v2(
        source_content_sha256=source_content_sha256,
        mapping_policy=mapping_policy,
        camera_mappings=camera_mappings,
    )
    validation_report = _validation_report(
        schema_registry=schema_registry,
        source_uri=source_uri,
        source_content_sha256=source_content_sha256,
        source_bytes=source_bytes,
        recording_identity_value=recording_id,
        mcap_id=mcap_id,
        mapping_policy=mapping_policy,
        mapping_digest=mapping_digest,
        validator=validator,
        stream_records=stream_records,
        camera_mappings=camera_mappings,
        observed_at=observed_at,
    )
    primary_policy = PrimaryAdmissionPolicy.create(
        version=FIXTURE_SOURCE_POLICY_VERSION,
        admissible_alignment_outcomes=(AlignmentAdmissionOutcome.VALID,),
    )
    ready_manifest = _ready_manifest(
        fixture=fixture,
        schema_registry=schema_registry,
        validation_report=validation_report,
        source_uri=source_uri,
        source_bytes=source_bytes,
        mapping_policy=mapping_policy,
        admission_policy=admission_policy_ref,
        durability_verifier=durability_verifier,
        stream_records=stream_records,
        observed_at=observed_at,
    )
    alignment_manifest = _alignment_manifest(
        fixture=fixture,
        schema_registry=schema_registry,
        ready_manifest=ready_manifest,
        policy=alignment_policy_ref,
        algorithm=alignment_algorithm,
        validator=alignment_validator,
        observed_at=observed_at,
    )
    evaluation = PrimaryAdmissionEvaluation(
        recording_identity=recording_id,
        ready_manifest_id=ready_manifest.ready_manifest_id,
        ready_manifest_semantic_sha256=ready_manifest.ready_manifest_semantic_sha256,
        source_outcome=SourceAdmissionOutcome.READY,
        alignment_outcome=AlignmentAdmissionOutcome.VALID,
        alignment_id=alignment_manifest.alignment_id,
        alignment_semantic_sha256=alignment_manifest.alignment_semantic_sha256,
        policy_version=primary_policy.version,
        policy_sha256=primary_policy.semantic_sha256,
        admissible=True,
        reason_code="ADMISSIBLE",
    )
    admitted_context = AdmissionContextResolver().resolve_v2(
        evaluation=evaluation,
        policy=primary_policy,
        validation_report=validation_report,
        ready_manifest=ready_manifest,
        alignment_manifest=alignment_manifest,
        registry=schema_registry,
    )
    frame_index, artifacts = _frame_index_and_artifacts(
        fixture=fixture,
        decoded_payloads=decoded_payloads,
        context=admitted_context,
        stream_records=stream_records,
    )
    requested_interval = NanosecondInterval(
        start_ns=0,
        end_ns=ready_manifest.recording.duration_ns,
    )
    sampling_plan = SamplingPlan(
        sampling_plan_id=_stable_uuid(
            "fixture-sampling-plan",
            FIXTURE_SAMPLING_POLICY_VERSION,
        ),
        version=FIXTURE_SAMPLING_POLICY_VERSION,
        qa_sampling_rate_fps=1.0,
        event_sampling_rate_fps=2.0,
        dense_sampling_rate_fps=2.0,
        frame_budget=FrameBudget(
            max_frames_per_camera=1_000,
            max_frames_total=6_000,
            overflow_policy=OverflowPolicy.SPLIT_WINDOW,
        ),
    )
    return CanonicalSourceBundle(
        source_content_sha256=source_content_sha256,
        admitted_context=admitted_context,
        requested_interval=requested_interval,
        sampling_plan=sampling_plan,
        frame_index=frame_index,
        _artifacts=MappingProxyType(artifacts),
    )


def _validation_report(
    *,
    schema_registry: SchemaRegistry,
    source_uri: str,
    source_content_sha256: str,
    source_bytes: int,
    recording_identity_value: str,
    mcap_id: str,
    mapping_policy: SemanticPolicyReference,
    mapping_digest: str,
    validator: EvidenceComponent,
    stream_records: Mapping[
        CameraId,
        tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2],
    ],
    camera_mappings: tuple[ValidationCameraMappingV2, ...],
    observed_at: str,
) -> MCAPValidationReportV2:
    schema_evidence = tuple(
        sorted(
            (record[0] for record in stream_records.values()),
            key=lambda item: (item.stream_semantic_sha256, item.schema_name),
        )
    )
    probed_stream_facts = tuple(
        sorted(
            (record[1] for record in stream_records.values()),
            key=lambda item: (
                item.stream_semantic_sha256,
                item.topic,
                item.channel_id,
            ),
        )
    )
    fields: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": schema_registry.resolve_version(
            MCAP_VALIDATION_REPORT_V2_SCHEMA_ID,
            ADMISSION_EVIDENCE_V2_SCHEMA_VERSION,
        ).ref,
        "validation_report_id": _stable_uuid("fixture-validation-row", source_content_sha256),
        "validation_report_semantic_sha256": "0" * 64,
        "mcap_id": mcap_id,
        "recording_identity": recording_identity_value,
        "source_content_sha256": source_content_sha256,
        "source": MCAPValidationSourceV2(
            uri=source_uri,
            object_version=source_content_sha256,
            sha256=source_content_sha256,
            bytes=source_bytes,
        ),
        "mapping_policy": mapping_policy,
        "camera_mapping_semantic_sha256": mapping_digest,
        "validator": validator,
        "checks": (
            _pass_check("camera-mapping", "six-camera mapping"),
            _pass_check("fixture-payloads", "all encoded source frames"),
            _pass_check("source-shape", "fixture recording"),
        ),
        "diagnostics": (),
        "schema_evidence": schema_evidence,
        "probed_stream_facts": probed_stream_facts,
        "camera_mappings": camera_mappings,
        "discovered_video_stream_count": 6,
        "mapped_camera_count": 6,
        "verdict": MCAPValidationVerdictV2.VALID,
        "validated_at": observed_at,
    }
    draft = MCAPValidationReportV2.model_construct(**fields)
    digest = semantic_sha256(mcap_validation_report_v2_semantic_projection(draft))
    fields["validation_report_semantic_sha256"] = digest
    fields["validation_report_id"] = _stable_uuid("fixture-validation-report", digest)
    return MCAPValidationReportV2.model_validate(fields, strict=True)


def _ready_manifest(
    *,
    fixture: CanonicalSourceFixture,
    schema_registry: SchemaRegistry,
    validation_report: MCAPValidationReportV2,
    source_uri: str,
    source_bytes: int,
    mapping_policy: SemanticPolicyReference,
    admission_policy: SemanticPolicyReference,
    durability_verifier: EvidenceComponent,
    stream_records: Mapping[
        CameraId,
        tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2],
    ],
    observed_at: str,
) -> MCAPReadyManifestV2:
    cameras: list[MCAPReadyCameraV2] = []
    for camera_id in CAMERA_IDS:
        camera = fixture.cameras[camera_id]
        schema_fact, probe_fact = stream_records[camera_id]
        duration_ns = camera.frames[-1].source_timestamp_ns - camera.frames[0].source_timestamp_ns
        cameras.append(
            MCAPReadyCameraV2(
                camera_id=camera_id,
                role=camera.role,
                stream_id=schema_fact.stream_id,
                stream_semantic_sha256=schema_fact.stream_semantic_sha256,
                topic=camera.topic,
                channel_id=camera.channel_id,
                codec=probe_fact.codec,
                width=camera.width,
                height=camera.height,
                nominal_fps=(len(camera.frames) - 1) * 1_000_000_000.0 / duration_ns,
                source_start_ns=camera.frames[0].source_timestamp_ns,
                source_end_ns=camera.frames[-1].source_timestamp_ns + 1,
                frame_count=len(camera.frames),
            )
        )
    source_origin_ns = min(
        camera.frames[0].source_timestamp_ns for camera in fixture.cameras.values()
    )
    source_end_ns = (
        max(camera.frames[-1].source_timestamp_ns for camera in fixture.cameras.values()) + 1
    )
    recording_duration_ns = source_end_ns - source_origin_ns
    fields: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": schema_registry.resolve_version(
            MCAP_READY_MANIFEST_V2_SCHEMA_ID,
            ADMISSION_EVIDENCE_V2_SCHEMA_VERSION,
        ).ref,
        "ready_manifest_id": _stable_uuid(
            "fixture-ready-row",
            validation_report.validation_report_semantic_sha256,
        ),
        "ready_manifest_semantic_sha256": "0" * 64,
        "mcap_id": validation_report.mcap_id,
        "recording_identity": validation_report.recording_identity,
        "source_content_sha256": validation_report.source_content_sha256,
        "source": MCAPReadySourceV2(
            artifact_id=_stable_uuid(
                "fixture-source-artifact",
                validation_report.source_content_sha256,
            ),
            uri=source_uri,
            object_version=validation_report.source_content_sha256,
            sha256=validation_report.source_content_sha256,
            bytes=source_bytes,
        ),
        "source_durability": SourceDurabilityEvidenceV2(
            verifier=durability_verifier,
            outcome="PASS",
            verified_sha256=validation_report.source_content_sha256,
            verified_bytes=source_bytes,
        ),
        "validation_report_id": validation_report.validation_report_id,
        "validation_report_semantic_sha256": (validation_report.validation_report_semantic_sha256),
        "validation_report_schema_ref": validation_report.schema_ref,
        "camera_mapping_run_id": _stable_uuid(
            "fixture-camera-mapping",
            validation_report.camera_mapping_semantic_sha256,
        ),
        "camera_mapping_semantic_sha256": (validation_report.camera_mapping_semantic_sha256),
        "mapping_policy": mapping_policy,
        "admission_policy": admission_policy,
        "recording": MCAPReadyRecording(
            start_utc=fixture.recording_start_utc,
            end_utc=None,
            duration_ns=recording_duration_ns,
            timebase="recording_relative_ns",
        ),
        "camera_count": 6,
        "cameras": tuple(cameras),
        "published_at": observed_at,
    }
    draft = MCAPReadyManifestV2.model_construct(**fields)
    digest = semantic_sha256(mcap_ready_manifest_v2_semantic_projection(draft))
    fields["ready_manifest_semantic_sha256"] = digest
    fields["ready_manifest_id"] = _stable_uuid("fixture-ready-manifest", digest)
    return MCAPReadyManifestV2.model_validate(fields, strict=True)


def _alignment_manifest(
    *,
    fixture: CanonicalSourceFixture,
    schema_registry: SchemaRegistry,
    ready_manifest: MCAPReadyManifestV2,
    policy: SemanticPolicyReference,
    algorithm: EvidenceComponent,
    validator: EvidenceComponent,
    observed_at: str,
) -> AlignmentManifestV2:
    ready_by_camera = {camera.camera_id: camera for camera in ready_manifest.cameras}
    cameras: dict[str, CameraAlignmentV2] = {}
    source_origin_ns = min(
        camera.frames[0].source_timestamp_ns for camera in fixture.cameras.values()
    )
    for camera_id in CAMERA_IDS:
        fixture_camera = fixture.cameras[camera_id]
        ready_camera = ready_by_camera[camera_id]
        first_timestamp = fixture_camera.frames[0].source_timestamp_ns
        segment = AlignmentSegment(
            segment_id=_stable_uuid(
                "fixture-alignment-segment",
                f"{ready_manifest.source_content_sha256}:{camera_id.value}",
            ),
            source_epoch_id="epoch-0",
            source_order_start=0,
            source_order_end=len(fixture_camera.frames),
            source_start_ns=first_timestamp,
            source_end_ns=fixture_camera.frames[-1].source_timestamp_ns + 1,
            source_anchor_ns=source_origin_ns,
            canonical_anchor_ns=0,
            rate_numerator="1",
            rate_denominator="1",
            rounding="HALF_EVEN",
        )
        cameras[camera_id.value] = CameraAlignmentV2(
            source_clock_id=fixture.source_clock_id,
            source_timestamp_unit="ns",
            derived_drift_ppm=0.0,
            residual_p95_ns=0,
            max_error_ns=0,
            coverage=1.0,
            segments=(segment,),
            status=AlignmentStatus.VALID,
            stream_id=ready_camera.stream_id,
            stream_semantic_sha256=ready_camera.stream_semantic_sha256,
        )
    fields: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": schema_registry.resolve_version(
            ALIGNMENT_MANIFEST_V2_SCHEMA_ID,
            ADMISSION_EVIDENCE_V2_SCHEMA_VERSION,
        ).ref,
        "alignment_id": _stable_uuid(
            "fixture-alignment-row",
            ready_manifest.ready_manifest_semantic_sha256,
        ),
        "alignment_semantic_sha256": "0" * 64,
        "mcap_id": ready_manifest.mcap_id,
        "recording_identity": ready_manifest.recording_identity,
        "source_content_sha256": ready_manifest.source_content_sha256,
        "ready_manifest_id": ready_manifest.ready_manifest_id,
        "ready_manifest_semantic_sha256": ready_manifest.ready_manifest_semantic_sha256,
        "ready_manifest_schema_ref": ready_manifest.schema_ref,
        "camera_mapping_run_id": ready_manifest.camera_mapping_run_id,
        "camera_mapping_semantic_sha256": ready_manifest.camera_mapping_semantic_sha256,
        "reference_timebase": "recording_relative_ns",
        "canonical_origin": CanonicalOrigin(
            source="fixture_shared_source_clock_origin",
            reference_timestamp_ns=source_origin_ns,
            utc=fixture.recording_start_utc,
        ),
        "method": AlignmentMethod.SENSOR_CLOCK,
        "algorithm": algorithm,
        "status": AlignmentStatus.VALID,
        "cameras": cameras,
        "policy": policy,
        "validator": validator,
        "checks": (_pass_check("rational-transform", "all camera frame timestamps"),),
        "diagnostics": (),
        "created_at": observed_at,
    }
    draft = AlignmentManifestV2.model_construct(**fields)
    digest = semantic_sha256(alignment_manifest_v2_semantic_projection(draft))
    fields["alignment_semantic_sha256"] = digest
    fields["alignment_id"] = _stable_uuid("fixture-alignment", digest)
    return AlignmentManifestV2.model_validate(fields, strict=True)


def _frame_index_and_artifacts(
    *,
    fixture: CanonicalSourceFixture,
    decoded_payloads: Mapping[tuple[CameraId, int], bytes],
    context: AdmittedRecordingContextV2,
    stream_records: Mapping[
        CameraId,
        tuple[StreamSchemaEvidenceV2, ProbedVideoStreamFactV2],
    ],
) -> tuple[
    CanonicalSixCameraFrameIndex,
    dict[tuple[CameraId, str], MaterializedFrameArtifactFact],
]:
    camera_indexes: dict[CameraId, CameraSourceFrameIndex] = {}
    artifacts: dict[tuple[CameraId, str], MaterializedFrameArtifactFact] = {}
    canonical_origin_ns = context.alignment_manifest.canonical_origin.reference_timestamp_ns
    for camera_id in CAMERA_IDS:
        camera = fixture.cameras[camera_id]
        stream = stream_records[camera_id][0]
        segment = context.alignment_manifest.cameras[camera_id.value].segments[0]
        indexed_frames: list[IndexedSourceFrame] = []
        for source_order, source_frame in enumerate(camera.frames):
            payload = decoded_payloads[(camera_id, source_order)]
            payload_digest = exact_bytes_sha256(payload)
            frame_identity = semantic_sha256(
                {
                    "source_content_sha256": context.source_content_sha256,
                    "camera_id": camera_id.value,
                    "source_order": source_order,
                    "source_timestamp_ns": str(source_frame.source_timestamp_ns),
                    "payload_sha256": payload_digest,
                }
            )
            source_frame_id = _stable_uuid("fixture-source-frame", frame_identity)
            indexed = IndexedSourceFrame(
                source_frame_id=source_frame_id,
                source_order=source_order,
                source_timestamp_ns=source_frame.source_timestamp_ns,
                source_locator={
                    "fixture_source_sha256": context.source_content_sha256,
                    "camera_id": camera_id.value,
                    "source_order": source_order,
                },
                decodable=True,
                alignment_projection=FrameAlignmentProjectionFact(
                    projection_id=_stable_uuid(
                        "fixture-frame-alignment",
                        f"{context.alignment_semantic_sha256}:{frame_identity}",
                    ),
                    alignment_id=context.alignment_manifest.alignment_id,
                    segment_id=segment.segment_id,
                    aligned_timestamp_ns=source_frame.source_timestamp_ns - canonical_origin_ns,
                ),
            )
            indexed_frames.append(indexed)
            artifacts[(camera_id, source_frame_id)] = MaterializedFrameArtifactFact(
                artifact=MaterializedArtifactManifest(
                    artifact_id=_stable_uuid("fixture-frame-artifact", frame_identity),
                    uri=(
                        f"fixture://frames/{context.source_content_sha256}/"
                        f"{camera_id.value}/{source_order}.png"
                    ),
                    sha256=payload_digest,
                    bytes=len(payload),
                    media_type="image/png",
                ),
                width=camera.width,
                height=camera.height,
                quality_flags=("DEVELOPMENT_FIXTURE_BYTES_VERIFIED",),
            )
        camera_indexes[camera_id] = CameraSourceFrameIndex(
            camera_id=camera_id,
            stream_id=stream.stream_id,
            stream_semantic_sha256=stream.stream_semantic_sha256,
            frames=tuple(indexed_frames),
        )
    return (
        CanonicalSixCameraFrameIndex(
            mcap_id=context.ready_manifest.mcap_id,
            camera_mapping_run_id=context.ready_manifest.camera_mapping_run_id,
            alignment_id=context.alignment_manifest.alignment_id,
            source_content_sha256=context.source_content_sha256,
            camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
            alignment_semantic_sha256=context.alignment_semantic_sha256,
            cameras=SixCameraMap[CameraSourceFrameIndex](camera_indexes),
        ),
        artifacts,
    )


def _decode_payloads(
    fixture: CanonicalSourceFixture,
) -> dict[tuple[CameraId, int], bytes]:
    try:
        import av
    except ModuleNotFoundError as exc:
        raise CanonicalSourceFixtureError(
            "fixture decoding requires the declared mcap development extra"
        ) from exc

    decoded: dict[tuple[CameraId, int], bytes] = {}
    for camera_id in CAMERA_IDS:
        for source_order, frame in enumerate(fixture.cameras[camera_id].frames):
            try:
                payload = base64.b64decode(frame.payload_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise CanonicalSourceFixtureError(
                    f"invalid base64 frame payload: {camera_id.value}/{source_order}"
                ) from exc
            if not payload.startswith(_PNG_SIGNATURE):
                raise CanonicalSourceFixtureError(
                    f"fixture frame is not PNG bytes: {camera_id.value}/{source_order}"
                )
            try:
                with av.open(BytesIO(payload), mode="r") as container:
                    frames = tuple(container.decode(video=0))
            except Exception as exc:
                raise CanonicalSourceFixtureError(
                    f"fixture frame cannot be decoded: {camera_id.value}/{source_order}"
                ) from exc
            if len(frames) != 1:
                raise CanonicalSourceFixtureError(
                    f"fixture PNG must decode exactly one frame: {camera_id.value}/{source_order}"
                )
            decoded_frame = frames[0]
            camera = fixture.cameras[camera_id]
            if (decoded_frame.width, decoded_frame.height) != (camera.width, camera.height):
                raise CanonicalSourceFixtureError(
                    "fixture frame dimensions do not match source facts: "
                    f"{camera_id.value}/{source_order}"
                )
            decoded[(camera_id, source_order)] = payload
    return decoded


def _component(name: str) -> EvidenceComponent:
    return EvidenceComponent(
        name=name,
        version=FIXTURE_SOURCE_POLICY_VERSION,
        code_sha256=semantic_sha256(
            {"component": name, "implementation": FIXTURE_SOURCE_POLICY_VERSION}
        ),
        configuration_sha256=semantic_sha256(
            {"component": name, "configuration": FIXTURE_SOURCE_POLICY_VERSION}
        ),
    )


def _policy(name: str) -> SemanticPolicyReference:
    return SemanticPolicyReference(
        version=FIXTURE_SOURCE_POLICY_VERSION,
        semantic_sha256=semantic_sha256({"policy": name, "version": FIXTURE_SOURCE_POLICY_VERSION}),
    )


def _pass_check(check_id: str, subject: str) -> ValidationCheckEvidenceV2:
    return ValidationCheckEvidenceV2(
        check_id=check_id,
        check_version=FIXTURE_SOURCE_POLICY_VERSION,
        subject=subject,
        outcome=ValidationCheckOutcome.PASS,
        diagnostic_ids=(),
    )


def _stable_uuid(namespace: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{value}"))


def _rfc3339(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "FIXTURE_SAMPLING_POLICY_VERSION",
    "FIXTURE_SOURCE_POLICY_VERSION",
    "CanonicalSourceBundle",
    "CanonicalSourceFixture",
    "CanonicalSourceFixtureError",
    "FixtureCameraStream",
    "FixtureSourceFrame",
    "load_canonical_source_fixture",
]
