"""Registered V2 source-admission and alignment evidence contracts.

The V1 wire documents remain readable, but they do not carry the complete
evidence and immutable registry pins required by Architecture V1.1 Sections
25.6 and 25.7.  These V2 models keep association UUIDs and source locators out
of semantic identity and bind every such association to a content digest.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.alignment import (
    AlignmentMethod,
    AlignmentSegment,
    AlignmentStatus,
    CameraAlignment,
    CanonicalOrigin,
)
from robata.contracts.cameras import CAMERA_ID_VALUES, CameraId
from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mcap import MCAPReadyRecording
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry, default_schema_registry

MCAP_VALIDATION_REPORT_V2_SCHEMA_ID = "https://schemas.robata.dev/mcap-validation-report"
MCAP_READY_MANIFEST_V2_SCHEMA_ID = "https://schemas.robata.dev/mcap-manifest"
ALIGNMENT_MANIFEST_V2_SCHEMA_ID = "https://schemas.robata.dev/alignment-manifest"
ADMISSION_EVIDENCE_V2_SCHEMA_VERSION = "2.0.0"

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


class EvidenceComponent(StrictModel):
    """Immutable executable component identity used to produce evidence."""

    name: NonEmptyString
    version: SchemaVersion
    code_sha256: Sha256Digest
    configuration_sha256: Sha256Digest


class SemanticPolicyReference(StrictModel):
    """Version and semantic digest of a policy candidate."""

    version: SchemaVersion
    semantic_sha256: Sha256Digest


class ValidationCheckOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceDiagnosticClassification(StrEnum):
    SOURCE = "SOURCE"
    POLICY = "POLICY"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class EvidenceDiagnosticSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationDiagnosticV2(StrictModel):
    """Stable diagnostic referenced by one or more explicit check outcomes."""

    diagnostic_id: NonEmptyString
    code: NonEmptyString
    message: NonEmptyString
    classification: EvidenceDiagnosticClassification
    severity: EvidenceDiagnosticSeverity
    path: str | None
    camera_id: CameraId | None
    stream_id: OpaqueUuid | None


class ValidationCheckEvidenceV2(StrictModel):
    """One named validator check and its complete terminal outcome."""

    check_id: NonEmptyString
    check_version: SchemaVersion
    subject: NonEmptyString
    outcome: ValidationCheckOutcome
    diagnostic_ids: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def require_canonical_diagnostic_ids(self) -> Self:
        if self.diagnostic_ids != tuple(sorted(set(self.diagnostic_ids))):
            raise ValueError("diagnostic_ids must be unique and lexically ordered")
        if (
            self.outcome
            in {
                ValidationCheckOutcome.FAIL,
                ValidationCheckOutcome.INCONCLUSIVE,
            }
            and not self.diagnostic_ids
        ):
            raise ValueError("FAIL and INCONCLUSIVE checks require diagnostics")
        return self


class SchemaSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class StreamSchemaEvidenceV2(StrictModel):
    """Exact source-schema fact and the policy decision for one video stream."""

    stream_id: OpaqueUuid
    stream_semantic_sha256: Sha256Digest
    schema_name: NonEmptyString
    schema_encoding: NonEmptyString
    schema_content_sha256: Sha256Digest | None
    support_status: SchemaSupportStatus
    support_policy: SemanticPolicyReference
    diagnostic_ids: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def require_schema_evidence(self) -> Self:
        if self.diagnostic_ids != tuple(sorted(set(self.diagnostic_ids))):
            raise ValueError("diagnostic_ids must be unique and lexically ordered")
        if self.support_status is not SchemaSupportStatus.SUPPORTED and not self.diagnostic_ids:
            raise ValueError("non-supported schema evidence requires diagnostics")
        return self


class DecoderProbeOutcome(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_RUN = "NOT_RUN"


class DecoderProbeEvidenceV2(StrictModel):
    """Decoder probe result; infrastructure uncertainty remains explicit."""

    probe: EvidenceComponent
    outcome: DecoderProbeOutcome
    decoded_frame_count: NonNegativeInt
    decoded_width: PositiveInt | None
    decoded_height: PositiveInt | None
    diagnostic_ids: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def require_probe_shape(self) -> Self:
        if self.diagnostic_ids != tuple(sorted(set(self.diagnostic_ids))):
            raise ValueError("diagnostic_ids must be unique and lexically ordered")
        if self.outcome is DecoderProbeOutcome.PASSED:
            if (
                self.decoded_frame_count < 1
                or self.decoded_width is None
                or self.decoded_height is None
            ):
                raise ValueError("PASSED decoder probe requires a decoded frame and dimensions")
        elif not self.diagnostic_ids:
            raise ValueError("non-passing decoder probe requires diagnostics")
        return self


class ProbedVideoStreamFactV2(StrictModel):
    """Container/index and decoder facts observed for one raw video stream."""

    stream_id: OpaqueUuid
    stream_semantic_sha256: Sha256Digest
    topic: NonEmptyString
    channel_id: NonNegativeInt
    message_encoding: NonEmptyString
    codec: NonEmptyString
    message_count: NonNegativeInt
    first_timestamp_ns: Nanoseconds | None
    last_timestamp_ns: Nanoseconds | None
    decoder_probe: DecoderProbeEvidenceV2

    @model_validator(mode="after")
    def require_timestamp_pair(self) -> Self:
        if (self.first_timestamp_ns is None) != (self.last_timestamp_ns is None):
            raise ValueError("stream timestamp bounds must both be present or both be absent")
        if (
            self.first_timestamp_ns is not None
            and self.last_timestamp_ns is not None
            and self.first_timestamp_ns > self.last_timestamp_ns
        ):
            raise ValueError("first_timestamp_ns must not exceed last_timestamp_ns")
        return self


class ValidationCameraMappingV2(StrictModel):
    """One mapping-policy candidate row, including invalid-report candidates."""

    camera_id: CameraId
    role: NonEmptyString
    stream_id: OpaqueUuid
    stream_semantic_sha256: Sha256Digest


class MCAPValidationSourceV2(StrictModel):
    """Verified bytes plus the non-identifying alias at validation time."""

    uri: NonEmptyString
    object_version: str | None
    sha256: Sha256Digest
    bytes: NonNegativeInt


class MCAPValidationVerdictV2(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"


class MCAPValidationReportV2(StrictModel):
    """Immutable and self-digesting complete source-validation evidence."""

    schema_version: Literal["2.0"]
    schema_ref: SchemaRef
    validation_report_id: OpaqueUuid
    validation_report_semantic_sha256: Sha256Digest
    mcap_id: OpaqueUuid
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    source: MCAPValidationSourceV2
    mapping_policy: SemanticPolicyReference
    camera_mapping_semantic_sha256: Sha256Digest
    validator: EvidenceComponent
    checks: Annotated[tuple[ValidationCheckEvidenceV2, ...], Field(min_length=1)]
    diagnostics: tuple[ValidationDiagnosticV2, ...]
    schema_evidence: tuple[StreamSchemaEvidenceV2, ...]
    probed_stream_facts: tuple[ProbedVideoStreamFactV2, ...]
    camera_mappings: tuple[ValidationCameraMappingV2, ...]
    discovered_video_stream_count: NonNegativeInt
    mapped_camera_count: NonNegativeInt
    verdict: MCAPValidationVerdictV2
    validated_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_complete_evidence(self) -> Self:
        _require_schema_ref(self.schema_ref, MCAP_VALIDATION_REPORT_V2_SCHEMA_ID)
        if self.source_content_sha256 != self.source.sha256:
            raise ValueError("source_content_sha256 must match source.sha256")
        if self.discovered_video_stream_count != len(self.probed_stream_facts):
            raise ValueError("discovered_video_stream_count must equal probed_stream_facts")
        if self.mapped_camera_count != len(self.camera_mappings):
            raise ValueError("mapped_camera_count must equal camera_mappings")

        _require_order(self.checks, key=lambda item: item.check_id, label="checks")
        _require_order(
            self.diagnostics,
            key=lambda item: item.diagnostic_id,
            label="diagnostics",
        )
        _require_order(
            self.schema_evidence,
            key=lambda item: (item.stream_semantic_sha256, item.schema_name),
            label="schema_evidence",
        )
        _require_order(
            self.probed_stream_facts,
            key=lambda item: (
                item.stream_semantic_sha256,
                item.topic,
                item.channel_id,
            ),
            label="probed_stream_facts",
        )
        _require_order(
            self.camera_mappings,
            key=lambda item: (
                item.camera_id.value,
                item.stream_semantic_sha256,
                item.role,
            ),
            label="camera_mappings",
        )

        diagnostic_ids = {item.diagnostic_id for item in self.diagnostics}
        referenced_diagnostics: set[str] = set()
        for check in self.checks:
            referenced_diagnostics.update(check.diagnostic_ids)
        for schema_fact in self.schema_evidence:
            referenced_diagnostics.update(schema_fact.diagnostic_ids)
        referenced_diagnostics.update(
            diagnostic_id
            for item in self.probed_stream_facts
            for diagnostic_id in item.decoder_probe.diagnostic_ids
        )
        if referenced_diagnostics != diagnostic_ids:
            raise ValueError("diagnostics must exactly match referenced diagnostic_ids")

        schema_by_stream = {item.stream_id: item for item in self.schema_evidence}
        probes_by_stream = {item.stream_id: item for item in self.probed_stream_facts}
        if len(schema_by_stream) != len(self.schema_evidence):
            raise ValueError("schema_evidence stream_id values must be unique")
        if len(probes_by_stream) != len(self.probed_stream_facts):
            raise ValueError("probed_stream_facts stream_id values must be unique")
        if set(schema_by_stream) != set(probes_by_stream):
            raise ValueError("schema_evidence and probed stream identities must match")
        for stream_id, schema_fact in schema_by_stream.items():
            probe_fact = probes_by_stream[stream_id]
            if schema_fact.stream_semantic_sha256 != probe_fact.stream_semantic_sha256:
                raise ValueError("schema and probe evidence must bind the same stream digest")
            expected_stream_digest = compute_stream_semantic_sha256_v2(
                source_content_sha256=self.source_content_sha256,
                schema_evidence=schema_fact,
                probed_stream_fact=probe_fact,
            )
            if schema_fact.stream_semantic_sha256 != expected_stream_digest:
                raise ValueError("stream_semantic_sha256 is inconsistent with source facts")
        for mapping in self.camera_mappings:
            mapped_probe_fact = probes_by_stream.get(mapping.stream_id)
            if (
                mapped_probe_fact is None
                or mapped_probe_fact.stream_semantic_sha256 != mapping.stream_semantic_sha256
            ):
                raise ValueError("camera mapping must bind a probed stream semantic digest")
        expected_mapping_digest = compute_camera_mapping_semantic_sha256_v2(
            source_content_sha256=self.source_content_sha256,
            mapping_policy=self.mapping_policy,
            camera_mappings=self.camera_mappings,
        )
        if self.camera_mapping_semantic_sha256 != expected_mapping_digest:
            raise ValueError("camera_mapping_semantic_sha256 is inconsistent")

        invalid_signal = (
            any(item.outcome is ValidationCheckOutcome.FAIL for item in self.checks)
            or any(
                item.support_status is SchemaSupportStatus.UNSUPPORTED
                for item in self.schema_evidence
            )
            or any(
                item.decoder_probe.outcome is DecoderProbeOutcome.FAILED
                for item in self.probed_stream_facts
            )
        )
        inconclusive_signal = (
            any(item.outcome is ValidationCheckOutcome.INCONCLUSIVE for item in self.checks)
            or any(
                item.support_status is SchemaSupportStatus.INCONCLUSIVE
                for item in self.schema_evidence
            )
            or any(
                item.decoder_probe.outcome
                in {DecoderProbeOutcome.INCONCLUSIVE, DecoderProbeOutcome.NOT_RUN}
                for item in self.probed_stream_facts
            )
        )
        expected_verdict = (
            MCAPValidationVerdictV2.INVALID
            if invalid_signal
            else (
                MCAPValidationVerdictV2.INCONCLUSIVE
                if inconclusive_signal
                else MCAPValidationVerdictV2.VALID
            )
        )
        if self.verdict is not expected_verdict:
            raise ValueError("verdict does not match check, schema, and probe evidence")

        if self.verdict is MCAPValidationVerdictV2.VALID:
            camera_ids = tuple(item.camera_id.value for item in self.camera_mappings)
            if camera_ids != CAMERA_ID_VALUES:
                raise ValueError("VALID report requires canonical cam_01 through cam_06 mappings")
            if len({item.stream_semantic_sha256 for item in self.camera_mappings}) != 6:
                raise ValueError("VALID report requires six distinct mapped stream digests")
            if any(
                item.support_status is not SchemaSupportStatus.SUPPORTED
                for item in self.schema_evidence
            ) or any(
                item.decoder_probe.outcome is not DecoderProbeOutcome.PASSED
                for item in self.probed_stream_facts
            ):
                raise ValueError("VALID report requires supported schemas and passing probes")
            if any(item.severity is EvidenceDiagnosticSeverity.ERROR for item in self.diagnostics):
                raise ValueError("VALID report cannot contain ERROR diagnostics")
        elif not self.diagnostics:
            raise ValueError("non-VALID report requires diagnostics")

        expected_digest = semantic_sha256(mcap_validation_report_v2_semantic_projection(self))
        if self.validation_report_semantic_sha256 != expected_digest:
            raise ValueError("validation_report_semantic_sha256 is inconsistent")
        return self


class MCAPReadySourceV2(StrictModel):
    """Durable raw-source association and its verified immutable bytes."""

    artifact_id: OpaqueUuid
    uri: NonEmptyString
    object_version: NonEmptyString
    sha256: Sha256Digest
    bytes: PositiveInt


class SourceDurabilityEvidenceV2(StrictModel):
    """Positive durability proof captured before READY publication."""

    verifier: EvidenceComponent
    outcome: Literal["PASS"]
    verified_sha256: Sha256Digest
    verified_bytes: PositiveInt


class MCAPReadyCameraV2(StrictModel):
    """One canonical READY camera bound to a stream semantic digest."""

    camera_id: CameraId
    role: NonEmptyString
    stream_id: OpaqueUuid
    stream_semantic_sha256: Sha256Digest
    topic: NonEmptyString
    channel_id: NonNegativeInt
    codec: NonEmptyString
    width: PositiveInt
    height: PositiveInt
    nominal_fps: PositiveFloat
    source_start_ns: Nanoseconds
    source_end_ns: Nanoseconds
    frame_count: PositiveInt

    @model_validator(mode="after")
    def require_nonempty_source_interval(self) -> Self:
        if self.source_start_ns >= self.source_end_ns:
            raise ValueError("camera source_start_ns must be less than source_end_ns")
        return self


class MCAPReadyManifestV2(StrictModel):
    """READY publication created only from selected valid and durable evidence."""

    schema_version: Literal["2.0"]
    schema_ref: SchemaRef
    ready_manifest_id: OpaqueUuid
    ready_manifest_semantic_sha256: Sha256Digest
    mcap_id: OpaqueUuid
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    source: MCAPReadySourceV2
    source_durability: SourceDurabilityEvidenceV2
    validation_report_id: OpaqueUuid
    validation_report_semantic_sha256: Sha256Digest
    validation_report_schema_ref: SchemaRef
    camera_mapping_run_id: OpaqueUuid
    camera_mapping_semantic_sha256: Sha256Digest
    mapping_policy: SemanticPolicyReference
    admission_policy: SemanticPolicyReference
    recording: MCAPReadyRecording
    camera_count: Literal[6]
    cameras: Annotated[tuple[MCAPReadyCameraV2, ...], Field(min_length=6, max_length=6)]
    published_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_ready_evidence(self) -> Self:
        _require_schema_ref(self.schema_ref, MCAP_READY_MANIFEST_V2_SCHEMA_ID)
        _require_schema_ref(
            self.validation_report_schema_ref,
            MCAP_VALIDATION_REPORT_V2_SCHEMA_ID,
        )
        if self.source_content_sha256 != self.source.sha256:
            raise ValueError("source_content_sha256 must match source.sha256")
        if (
            self.source_durability.verified_sha256 != self.source.sha256
            or self.source_durability.verified_bytes != self.source.bytes
        ):
            raise ValueError("source durability evidence must bind exact source bytes")
        actual_camera_ids = tuple(item.camera_id.value for item in self.cameras)
        if actual_camera_ids != CAMERA_ID_VALUES:
            raise ValueError("READY cameras must be ordered cam_01 through cam_06")
        if len({item.stream_semantic_sha256 for item in self.cameras}) != 6:
            raise ValueError("READY cameras require six distinct stream semantic digests")
        expected_mapping_digest = compute_camera_mapping_semantic_sha256_v2(
            source_content_sha256=self.source_content_sha256,
            mapping_policy=self.mapping_policy,
            camera_mappings=self.cameras,
        )
        if self.camera_mapping_semantic_sha256 != expected_mapping_digest:
            raise ValueError("camera_mapping_semantic_sha256 is inconsistent")
        expected_digest = semantic_sha256(mcap_ready_manifest_v2_semantic_projection(self))
        if self.ready_manifest_semantic_sha256 != expected_digest:
            raise ValueError("ready_manifest_semantic_sha256 is inconsistent")
        return self


class CameraAlignmentV2(CameraAlignment):
    """Alignment transform evidence bound to one source stream."""

    stream_id: OpaqueUuid
    stream_semantic_sha256: Sha256Digest


class AlignmentManifestV2(StrictModel):
    """Immutable six-camera alignment evidence with complete content bindings."""

    schema_version: Literal["2.0"]
    schema_ref: SchemaRef
    alignment_id: OpaqueUuid
    alignment_semantic_sha256: Sha256Digest
    mcap_id: OpaqueUuid
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    ready_manifest_id: OpaqueUuid
    ready_manifest_semantic_sha256: Sha256Digest
    ready_manifest_schema_ref: SchemaRef
    camera_mapping_run_id: OpaqueUuid
    camera_mapping_semantic_sha256: Sha256Digest
    reference_timebase: NonEmptyString
    canonical_origin: CanonicalOrigin
    method: AlignmentMethod
    algorithm: EvidenceComponent
    status: AlignmentStatus
    cameras: dict[str, CameraAlignmentV2]
    policy: SemanticPolicyReference
    validator: EvidenceComponent
    checks: Annotated[tuple[ValidationCheckEvidenceV2, ...], Field(min_length=1)]
    diagnostics: tuple[ValidationDiagnosticV2, ...]
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_alignment_evidence(self) -> Self:
        _require_schema_ref(self.schema_ref, ALIGNMENT_MANIFEST_V2_SCHEMA_ID)
        _require_schema_ref(self.ready_manifest_schema_ref, MCAP_READY_MANIFEST_V2_SCHEMA_ID)
        if tuple(self.cameras) != CAMERA_ID_VALUES:
            raise ValueError("alignment cameras must contain canonical cam_01 through cam_06 keys")
        if len({item.stream_semantic_sha256 for item in self.cameras.values()}) != 6:
            raise ValueError("alignment cameras require six distinct stream semantic digests")
        _require_order(self.checks, key=lambda item: item.check_id, label="checks")
        _require_order(
            self.diagnostics,
            key=lambda item: item.diagnostic_id,
            label="diagnostics",
        )
        diagnostic_ids = {item.diagnostic_id for item in self.diagnostics}
        referenced = {
            diagnostic_id for item in self.checks for diagnostic_id in item.diagnostic_ids
        }
        if referenced != diagnostic_ids:
            raise ValueError("alignment diagnostics must exactly match check references")

        camera_statuses = {item.status for item in self.cameras.values()}
        check_outcomes = {item.outcome for item in self.checks}
        if (
            AlignmentStatus.INVALID in camera_statuses
            or ValidationCheckOutcome.FAIL in check_outcomes
        ):
            expected_status = AlignmentStatus.INVALID
        elif (
            AlignmentStatus.UNVERIFIED in camera_statuses
            or ValidationCheckOutcome.INCONCLUSIVE in check_outcomes
        ):
            expected_status = AlignmentStatus.UNVERIFIED
        elif AlignmentStatus.DEGRADED in camera_statuses:
            expected_status = AlignmentStatus.DEGRADED
        else:
            expected_status = AlignmentStatus.VALID
        if self.status is not expected_status:
            raise ValueError("alignment status does not match camera and check evidence")
        if self.status is not AlignmentStatus.VALID and not self.diagnostics:
            raise ValueError("non-VALID alignment requires diagnostics")

        expected_digest = semantic_sha256(alignment_manifest_v2_semantic_projection(self))
        if self.alignment_semantic_sha256 != expected_digest:
            raise ValueError("alignment_semantic_sha256 is inconsistent")
        return self


AdmissionEvidenceV2 = MCAPValidationReportV2 | MCAPReadyManifestV2 | AlignmentManifestV2


def stream_semantic_projection_v2(
    *,
    source_content_sha256: str,
    schema_evidence: StreamSchemaEvidenceV2,
    probed_stream_fact: ProbedVideoStreamFactV2,
) -> dict[str, Any]:
    """Project one raw stream using source content instead of its association UUID."""

    if schema_evidence.stream_id != probed_stream_fact.stream_id:
        raise ValueError("schema and probe evidence must reference the same stream_id")
    return {
        "source_content_sha256": source_content_sha256,
        "topic": probed_stream_fact.topic,
        "channel_id": probed_stream_fact.channel_id,
        "message_encoding": probed_stream_fact.message_encoding,
        "codec": probed_stream_fact.codec,
        "schema_name": schema_evidence.schema_name,
        "schema_encoding": schema_evidence.schema_encoding,
        "schema_content_sha256": schema_evidence.schema_content_sha256,
    }


def compute_stream_semantic_sha256_v2(
    *,
    source_content_sha256: str,
    schema_evidence: StreamSchemaEvidenceV2,
    probed_stream_fact: ProbedVideoStreamFactV2,
) -> Sha256Digest:
    return semantic_sha256(
        stream_semantic_projection_v2(
            source_content_sha256=source_content_sha256,
            schema_evidence=schema_evidence,
            probed_stream_fact=probed_stream_fact,
        )
    )


def camera_mapping_semantic_projection_v2(
    *,
    source_content_sha256: str,
    mapping_policy: SemanticPolicyReference,
    camera_mappings: tuple[ValidationCameraMappingV2 | MCAPReadyCameraV2, ...],
) -> dict[str, Any]:
    """Project a mapping candidate without mapping-run or stream row UUIDs."""

    return {
        "source_content_sha256": source_content_sha256,
        "mapping_policy": mapping_policy.model_dump(mode="json"),
        "cameras": [
            {
                "camera_id": item.camera_id.value,
                "role": item.role,
                "stream_semantic_sha256": item.stream_semantic_sha256,
            }
            for item in camera_mappings
        ],
    }


def compute_camera_mapping_semantic_sha256_v2(
    *,
    source_content_sha256: str,
    mapping_policy: SemanticPolicyReference,
    camera_mappings: tuple[ValidationCameraMappingV2 | MCAPReadyCameraV2, ...],
) -> Sha256Digest:
    return semantic_sha256(
        camera_mapping_semantic_projection_v2(
            source_content_sha256=source_content_sha256,
            mapping_policy=mapping_policy,
            camera_mappings=camera_mappings,
        )
    )


def mcap_validation_report_v2_semantic_projection(
    report: MCAPValidationReportV2,
) -> dict[str, Any]:
    """Return the alias-, row-, and clock-independent validation projection."""

    return {
        "schema_version": report.schema_version,
        "schema_ref": _schema_ref_projection(report.schema_ref),
        "recording_identity": report.recording_identity,
        "source_content_sha256": report.source_content_sha256,
        "source_bytes": report.source.bytes,
        "mapping_policy": report.mapping_policy.model_dump(mode="json"),
        "camera_mapping_semantic_sha256": report.camera_mapping_semantic_sha256,
        "validator": report.validator.model_dump(mode="json"),
        "checks": [item.model_dump(mode="json") for item in report.checks],
        "diagnostics": [
            item.model_dump(mode="json", exclude={"stream_id"}) for item in report.diagnostics
        ],
        "schema_evidence": [
            item.model_dump(mode="json", exclude={"stream_id"}) for item in report.schema_evidence
        ],
        "probed_stream_facts": [
            item.model_dump(mode="json", exclude={"stream_id"})
            for item in report.probed_stream_facts
        ],
        "camera_mappings": [
            item.model_dump(mode="json", exclude={"stream_id"}) for item in report.camera_mappings
        ],
        "discovered_video_stream_count": report.discovered_video_stream_count,
        "mapped_camera_count": report.mapped_camera_count,
        "verdict": report.verdict.value,
    }


def mcap_ready_manifest_v2_semantic_projection(
    manifest: MCAPReadyManifestV2,
) -> dict[str, Any]:
    """Return the source-locator- and publication-independent READY projection."""

    return {
        "schema_version": manifest.schema_version,
        "schema_ref": _schema_ref_projection(manifest.schema_ref),
        "recording_identity": manifest.recording_identity,
        "source_content_sha256": manifest.source_content_sha256,
        "source_bytes": manifest.source.bytes,
        "source_durability": manifest.source_durability.model_dump(mode="json"),
        "validation_report_semantic_sha256": (manifest.validation_report_semantic_sha256),
        "validation_report_schema_ref": _schema_ref_projection(
            manifest.validation_report_schema_ref
        ),
        "camera_mapping_semantic_sha256": manifest.camera_mapping_semantic_sha256,
        "mapping_policy": manifest.mapping_policy.model_dump(mode="json"),
        "admission_policy": manifest.admission_policy.model_dump(mode="json"),
        "recording": manifest.recording.model_dump(mode="json"),
        "camera_count": manifest.camera_count,
        "cameras": [
            item.model_dump(mode="json", exclude={"stream_id"}) for item in manifest.cameras
        ],
    }


def alignment_manifest_v2_semantic_projection(
    manifest: AlignmentManifestV2,
) -> dict[str, Any]:
    """Return the row- and publication-independent alignment projection."""

    cameras: dict[str, Any] = {}
    for camera_id, camera in manifest.cameras.items():
        camera_projection = camera.model_dump(mode="json", exclude={"stream_id", "segments"})
        camera_projection["segments"] = [
            _alignment_segment_projection(segment) for segment in camera.segments
        ]
        cameras[camera_id] = camera_projection
    return {
        "schema_version": manifest.schema_version,
        "schema_ref": _schema_ref_projection(manifest.schema_ref),
        "recording_identity": manifest.recording_identity,
        "source_content_sha256": manifest.source_content_sha256,
        "ready_manifest_semantic_sha256": manifest.ready_manifest_semantic_sha256,
        "ready_manifest_schema_ref": _schema_ref_projection(manifest.ready_manifest_schema_ref),
        "camera_mapping_semantic_sha256": manifest.camera_mapping_semantic_sha256,
        "reference_timebase": manifest.reference_timebase,
        "canonical_origin": manifest.canonical_origin.model_dump(mode="json"),
        "method": manifest.method.value,
        "algorithm": manifest.algorithm.model_dump(mode="json"),
        "status": manifest.status.value,
        "cameras": cameras,
        "policy": manifest.policy.model_dump(mode="json"),
        "validator": manifest.validator.model_dump(mode="json"),
        "checks": [item.model_dump(mode="json") for item in manifest.checks],
        "diagnostics": [
            item.model_dump(mode="json", exclude={"stream_id"}) for item in manifest.diagnostics
        ],
    }


def validate_registered_admission_evidence_v2[
    EvidenceT: (MCAPValidationReportV2, MCAPReadyManifestV2, AlignmentManifestV2)
](
    evidence: EvidenceT,
    registry: SchemaRegistry | None = None,
) -> EvidenceT:
    """Resolve the exact quartet and validate the complete registered wire payload."""

    active_registry = registry or default_schema_registry()
    references = [evidence.schema_ref]
    if isinstance(evidence, MCAPReadyManifestV2):
        references.append(evidence.validation_report_schema_ref)
    elif isinstance(evidence, AlignmentManifestV2):
        references.append(evidence.ready_manifest_schema_ref)
    for reference in references:
        active_registry.resolve_exact(reference)
    active_registry.validate_pinned(
        evidence.schema_ref,
        evidence.model_dump(mode="json"),
    )
    return evidence


def _schema_ref_projection(reference: SchemaRef) -> dict[str, str]:
    return {
        "schema_id": reference.schema_id,
        "version": reference.version,
        "sha256": reference.sha256,
    }


def _alignment_segment_projection(segment: AlignmentSegment) -> dict[str, Any]:
    return segment.model_dump(mode="json", exclude={"segment_id"})


def _require_schema_ref(reference: SchemaRef, schema_id: str) -> None:
    if (
        reference.schema_id != schema_id
        or reference.version != ADMISSION_EVIDENCE_V2_SCHEMA_VERSION
    ):
        raise ValueError(f"schema_ref must identify {schema_id}@2.0.0")


def _require_order(
    items: tuple[Any, ...],
    *,
    key: Any,
    label: str,
) -> None:
    keys: list[Any] = [key(item) for item in items]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError(f"{label} must have unique canonical order")


__all__ = [
    "ADMISSION_EVIDENCE_V2_SCHEMA_VERSION",
    "ALIGNMENT_MANIFEST_V2_SCHEMA_ID",
    "MCAP_READY_MANIFEST_V2_SCHEMA_ID",
    "MCAP_VALIDATION_REPORT_V2_SCHEMA_ID",
    "AdmissionEvidenceV2",
    "AlignmentManifestV2",
    "CameraAlignmentV2",
    "DecoderProbeEvidenceV2",
    "DecoderProbeOutcome",
    "EvidenceComponent",
    "EvidenceDiagnosticClassification",
    "EvidenceDiagnosticSeverity",
    "MCAPReadyCameraV2",
    "MCAPReadyManifestV2",
    "MCAPReadySourceV2",
    "MCAPValidationReportV2",
    "MCAPValidationSourceV2",
    "MCAPValidationVerdictV2",
    "ProbedVideoStreamFactV2",
    "SchemaSupportStatus",
    "SemanticPolicyReference",
    "SourceDurabilityEvidenceV2",
    "StreamSchemaEvidenceV2",
    "ValidationCameraMappingV2",
    "ValidationCheckEvidenceV2",
    "ValidationCheckOutcome",
    "ValidationDiagnosticV2",
    "alignment_manifest_v2_semantic_projection",
    "camera_mapping_semantic_projection_v2",
    "compute_camera_mapping_semantic_sha256_v2",
    "compute_stream_semantic_sha256_v2",
    "mcap_ready_manifest_v2_semantic_projection",
    "mcap_validation_report_v2_semantic_projection",
    "stream_semantic_projection_v2",
    "validate_registered_admission_evidence_v2",
]
