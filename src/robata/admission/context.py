"""Fail-closed resolution of selected source and alignment admission evidence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ValidationError, model_validator

from robata.admission.ledger import (
    AlignmentAdmissionOutcome,
    PrimaryAdmissionEvaluation,
    PrimaryAdmissionPolicy,
    SourceAdmissionOutcome,
)
from robata.contracts.admission_v2 import (
    AlignmentManifestV2,
    MCAPReadyManifestV2,
    MCAPValidationReportV2,
    MCAPValidationVerdictV2,
    validate_registered_admission_evidence_v2,
)
from robata.contracts.alignment import AlignmentRun, AlignmentStatus
from robata.contracts.cameras import CAMERA_ID_VALUES
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mcap import (
    CameraMappingRun,
    MCAPReadyManifest,
    MCAPValidationReport,
    MCAPValidationVerdict,
)
from robata.contracts.schema_registry import SchemaRegistry


class AdmissionContextError(ValueError):
    """Selected ledger state and retrieved artifact bodies do not agree."""


class AdmittedRecordingContext(StrictModel):
    """Frozen, cross-bound evidence required before primary package creation."""

    schema_version: Literal["1.0"]
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    ready_manifest_id: OpaqueUuid
    validation_report_semantic_sha256: Sha256Digest
    ready_manifest_semantic_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    evaluation: PrimaryAdmissionEvaluation
    policy: PrimaryAdmissionPolicy
    validation_report: MCAPValidationReport
    ready_manifest: MCAPReadyManifest
    camera_mapping_run: CameraMappingRun
    alignment_run: AlignmentRun
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        _validate_cross_bindings(self)
        expected = semantic_sha256(admitted_recording_context_semantic_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("admitted context semantic_sha256 is inconsistent")
        return self


class AdmittedRecordingContextV2(StrictModel):
    """Registered, self-digesting admission evidence for the canonical path."""

    schema_version: Literal["2.0"]
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    ready_manifest_id: OpaqueUuid
    validation_report_semantic_sha256: Sha256Digest
    ready_manifest_semantic_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    evaluation: PrimaryAdmissionEvaluation
    policy: PrimaryAdmissionPolicy
    validation_report: MCAPValidationReportV2
    ready_manifest: MCAPReadyManifestV2
    alignment_manifest: AlignmentManifestV2
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        _validate_v2_cross_bindings(self)
        expected = semantic_sha256(admitted_recording_context_v2_semantic_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("v2 admitted context semantic_sha256 is inconsistent")
        return self


class AdmissionContextResolver:
    """Resolve ledger selection plus retrieved bodies into one trusted context."""

    def resolve(
        self,
        *,
        evaluation: PrimaryAdmissionEvaluation,
        policy: PrimaryAdmissionPolicy,
        ready_manifest_id: str,
        validation_report: MCAPValidationReport,
        validation_report_semantic_sha256: str,
        ready_manifest: MCAPReadyManifest,
        ready_manifest_semantic_sha256: str,
        camera_mapping_run: CameraMappingRun,
        camera_mapping_semantic_sha256: str,
        alignment_run: AlignmentRun,
        alignment_semantic_sha256: str,
    ) -> AdmittedRecordingContext:
        """Verify selected IDs, semantic bodies, and consuming-policy admission."""

        context_digest = semantic_sha256(
            _context_projection_values(
                recording_identity=evaluation.recording_identity,
                source_content_sha256=ready_manifest.source.sha256,
                validation_report_semantic_sha256=validation_report_semantic_sha256,
                ready_manifest_semantic_sha256=ready_manifest_semantic_sha256,
                camera_mapping_semantic_sha256=camera_mapping_semantic_sha256,
                alignment_semantic_sha256=alignment_semantic_sha256,
                alignment_outcome=evaluation.alignment_outcome,
                policy_sha256=policy.semantic_sha256,
            )
        )
        try:
            return AdmittedRecordingContext(
                schema_version="1.0",
                recording_identity=evaluation.recording_identity,
                source_content_sha256=ready_manifest.source.sha256,
                ready_manifest_id=ready_manifest_id,
                validation_report_semantic_sha256=validation_report_semantic_sha256,
                ready_manifest_semantic_sha256=ready_manifest_semantic_sha256,
                camera_mapping_semantic_sha256=camera_mapping_semantic_sha256,
                alignment_semantic_sha256=alignment_semantic_sha256,
                evaluation=evaluation,
                policy=policy,
                validation_report=validation_report,
                ready_manifest=ready_manifest,
                camera_mapping_run=camera_mapping_run,
                alignment_run=alignment_run,
                semantic_sha256=context_digest,
            )
        except (ValidationError, ValueError) as exc:
            raise AdmissionContextError(_error_message(exc)) from exc

    def resolve_v2(
        self,
        *,
        evaluation: PrimaryAdmissionEvaluation,
        policy: PrimaryAdmissionPolicy,
        validation_report: MCAPValidationReportV2,
        ready_manifest: MCAPReadyManifestV2,
        alignment_manifest: AlignmentManifestV2,
        registry: SchemaRegistry | None = None,
    ) -> AdmittedRecordingContextV2:
        """Resolve only complete, registered V2 evidence; no V1 upcast is attempted."""

        try:
            validate_registered_admission_evidence_v2(validation_report, registry)
            validate_registered_admission_evidence_v2(ready_manifest, registry)
            validate_registered_admission_evidence_v2(alignment_manifest, registry)
            return AdmittedRecordingContextV2(
                schema_version="2.0",
                recording_identity=validation_report.recording_identity,
                source_content_sha256=validation_report.source_content_sha256,
                ready_manifest_id=ready_manifest.ready_manifest_id,
                validation_report_semantic_sha256=(
                    validation_report.validation_report_semantic_sha256
                ),
                ready_manifest_semantic_sha256=ready_manifest.ready_manifest_semantic_sha256,
                camera_mapping_semantic_sha256=(validation_report.camera_mapping_semantic_sha256),
                alignment_semantic_sha256=alignment_manifest.alignment_semantic_sha256,
                evaluation=evaluation,
                policy=policy,
                validation_report=validation_report,
                ready_manifest=ready_manifest,
                alignment_manifest=alignment_manifest,
                semantic_sha256=semantic_sha256(
                    {
                        "recording_identity": validation_report.recording_identity,
                        "source_content_sha256": validation_report.source_content_sha256,
                        "validation_report_semantic_sha256": (
                            validation_report.validation_report_semantic_sha256
                        ),
                        "ready_manifest_semantic_sha256": (
                            ready_manifest.ready_manifest_semantic_sha256
                        ),
                        "camera_mapping_semantic_sha256": (
                            validation_report.camera_mapping_semantic_sha256
                        ),
                        "alignment_semantic_sha256": alignment_manifest.alignment_semantic_sha256,
                        "alignment_outcome": _alignment_outcome_v2(alignment_manifest.status).value,
                        "policy_sha256": policy.semantic_sha256,
                    }
                ),
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise AdmissionContextError(_error_message(exc)) from exc


def validation_report_semantic_projection(
    report: MCAPValidationReport,
) -> dict[str, object]:
    """Project source-validation meaning without aliases, row IDs, or clocks."""

    return {
        "schema_version": report.schema_version,
        "recording_identity": report.recording_identity,
        "source": {
            "sha256": report.source.sha256,
            "bytes": report.source.bytes,
        },
        "mapping_policy": report.mapping_policy.model_dump(mode="json"),
        "verdict": report.verdict.value,
        "discovered_video_stream_count": report.discovered_video_stream_count,
        "mapped_camera_count": report.mapped_camera_count,
        "errors": [item.model_dump(mode="json") for item in report.errors],
    }


def validation_report_semantic_digest(report: MCAPValidationReport) -> Sha256Digest:
    return semantic_sha256(validation_report_semantic_projection(report))


def camera_mapping_semantic_projection(
    mapping_run: CameraMappingRun,
    *,
    source_content_sha256: str,
    mapping_policy_sha256: str,
) -> dict[str, object]:
    """Project a mapping revision with content identity replacing association rows."""

    return {
        "source_content_sha256": source_content_sha256,
        "mapping_policy_version": mapping_run.mapping_policy_version,
        "mapping_policy_digest": mapping_policy_sha256,
        "status": mapping_run.status,
        "cameras": [item.model_dump(mode="json") for item in mapping_run.cameras],
    }


def camera_mapping_semantic_digest(
    mapping_run: CameraMappingRun,
    *,
    source_content_sha256: str,
    mapping_policy_sha256: str,
) -> Sha256Digest:
    return semantic_sha256(
        camera_mapping_semantic_projection(
            mapping_run,
            source_content_sha256=source_content_sha256,
            mapping_policy_sha256=mapping_policy_sha256,
        )
    )


def ready_manifest_semantic_projection(
    manifest: MCAPReadyManifest,
    *,
    validation_report_semantic_sha256: str,
    camera_mapping_semantic_sha256: str,
) -> dict[str, object]:
    """Project READY meaning with selected evidence represented by its digests."""

    return {
        "schema_version": manifest.schema_version,
        "validation_report_semantic_sha256": validation_report_semantic_sha256,
        "source": {
            "sha256": manifest.source.sha256,
            "bytes": manifest.source.bytes,
        },
        "recording": manifest.recording.model_dump(mode="json"),
        "camera_count": manifest.camera_count,
        "camera_mapping_semantic_sha256": camera_mapping_semantic_sha256,
        "camera_mapping_version": manifest.camera_mapping_version,
        "cameras": [item.model_dump(mode="json") for item in manifest.cameras],
    }


def ready_manifest_semantic_digest(
    manifest: MCAPReadyManifest,
    *,
    validation_report_semantic_sha256: str,
    camera_mapping_semantic_sha256: str,
) -> Sha256Digest:
    return semantic_sha256(
        ready_manifest_semantic_projection(
            manifest,
            validation_report_semantic_sha256=validation_report_semantic_sha256,
            camera_mapping_semantic_sha256=camera_mapping_semantic_sha256,
        )
    )


def alignment_run_semantic_projection(
    alignment_run: AlignmentRun,
    *,
    source_content_sha256: str,
    camera_mapping_semantic_sha256: str,
) -> dict[str, object]:
    """Project alignment meaning without artifact/segment row IDs or publish time."""

    cameras: dict[str, object] = {}
    for camera_id in CAMERA_ID_VALUES:
        camera = alignment_run.cameras[camera_id]
        cameras[camera_id] = {
            "source_clock_id": camera.source_clock_id,
            "source_timestamp_unit": camera.source_timestamp_unit,
            "derived_drift_ppm": camera.derived_drift_ppm,
            "residual_p95_ns": camera.residual_p95_ns,
            "max_error_ns": camera.max_error_ns,
            "coverage": camera.coverage,
            "segments": [
                segment.model_dump(mode="json", exclude={"segment_id"})
                for segment in camera.segments
            ],
            "status": camera.status.value,
        }
    return {
        "schema_version": alignment_run.schema_version,
        "source_content_sha256": source_content_sha256,
        "camera_mapping_semantic_sha256": camera_mapping_semantic_sha256,
        "reference_timebase": alignment_run.reference_timebase,
        "canonical_origin": alignment_run.canonical_origin.model_dump(mode="json"),
        "method": alignment_run.method.value,
        "algorithm_version": alignment_run.algorithm_version,
        "status": alignment_run.status.value,
        "cameras": cameras,
        "policy_version": alignment_run.policy_version,
    }


def alignment_run_semantic_digest(
    alignment_run: AlignmentRun,
    *,
    source_content_sha256: str,
    camera_mapping_semantic_sha256: str,
) -> Sha256Digest:
    return semantic_sha256(
        alignment_run_semantic_projection(
            alignment_run,
            source_content_sha256=source_content_sha256,
            camera_mapping_semantic_sha256=camera_mapping_semantic_sha256,
        )
    )


def admitted_recording_context_semantic_projection(
    context: AdmittedRecordingContext,
) -> dict[str, object]:
    return _context_projection_values(
        recording_identity=context.recording_identity,
        source_content_sha256=context.source_content_sha256,
        validation_report_semantic_sha256=context.validation_report_semantic_sha256,
        ready_manifest_semantic_sha256=context.ready_manifest_semantic_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=context.alignment_semantic_sha256,
        alignment_outcome=context.evaluation.alignment_outcome,
        policy_sha256=context.policy.semantic_sha256,
    )


def admitted_recording_context_v2_semantic_projection(
    context: AdmittedRecordingContextV2,
) -> dict[str, object]:
    """Return the digest-only identity of a V2 admission context."""

    return {
        "recording_identity": context.recording_identity,
        "source_content_sha256": context.source_content_sha256,
        "validation_report_semantic_sha256": context.validation_report_semantic_sha256,
        "ready_manifest_semantic_sha256": context.ready_manifest_semantic_sha256,
        "camera_mapping_semantic_sha256": context.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": context.alignment_semantic_sha256,
        "alignment_outcome": _alignment_outcome_v2(context.alignment_manifest.status).value,
        "policy_sha256": context.policy.semantic_sha256,
    }


def _validate_cross_bindings(context: AdmittedRecordingContext) -> None:
    evaluation = context.evaluation
    policy = context.policy
    report = context.validation_report
    manifest = context.ready_manifest
    mapping = context.camera_mapping_run
    alignment = context.alignment_run

    _require(evaluation.admissible, "primary admission evaluation is not admissible")
    _require(evaluation.reason_code == "ADMISSIBLE", "primary admission reason is not ADMISSIBLE")
    _require(
        evaluation.source_outcome is SourceAdmissionOutcome.READY,
        "primary admission did not select a READY source",
    )
    _require(
        evaluation.policy_version == policy.version
        and evaluation.policy_sha256 == policy.semantic_sha256,
        "primary admission policy does not match the supplied policy",
    )
    _require(
        evaluation.ready_manifest_id == context.ready_manifest_id,
        "selected READY manifest ID does not match the retrieved artifact",
    )
    _require(
        evaluation.ready_manifest_semantic_sha256 == context.ready_manifest_semantic_sha256,
        "selected READY manifest digest does not match the retrieved artifact",
    )
    _require(
        evaluation.alignment_id == alignment.alignment_id,
        "selected alignment ID does not match the retrieved artifact",
    )
    _require(
        evaluation.alignment_semantic_sha256 == context.alignment_semantic_sha256,
        "selected alignment digest does not match the retrieved artifact",
    )

    _require(report.verdict is MCAPValidationVerdict.VALID, "validation report is not VALID")
    _require(report.mapped_camera_count == 6, "VALID report does not prove six mapped cameras")
    _require(not report.errors, "VALID report contains diagnostics")
    _require(
        report.recording_identity == context.recording_identity == evaluation.recording_identity,
        "recording identity does not match selected source evidence",
    )
    _require(
        report.source.sha256 == manifest.source.sha256 == context.source_content_sha256,
        "source content digest does not match across admission evidence",
    )
    _require(
        report.source.bytes == manifest.source.bytes,
        "source byte count does not match across admission evidence",
    )
    _require(
        report.mcap_id == manifest.mcap_id == mapping.mcap_id == alignment.mcap_id,
        "MCAP association does not match across admission evidence",
    )
    _require(
        manifest.validation_report_id == report.validation_report_id,
        "READY manifest does not reference the supplied validation report",
    )
    _require(mapping.status == "PUBLISHED", "READY context requires a PUBLISHED mapping")
    _require(
        manifest.camera_mapping_run_id == mapping.mapping_run_id == alignment.camera_mapping_run_id,
        "camera mapping revision does not match across admission evidence",
    )
    _require(
        report.mapping_policy.version
        == mapping.mapping_policy_version
        == manifest.camera_mapping_version,
        "mapping policy version does not match across admission evidence",
    )
    mapping_rows = tuple((item.camera_id, item.role, item.stream_id) for item in mapping.cameras)
    manifest_rows = tuple(
        (item.camera_id.value, item.role, item.stream_id) for item in manifest.cameras
    )
    _require(
        tuple(item[0] for item in mapping_rows) == CAMERA_ID_VALUES,
        "mapping cameras are not in canonical six-camera order",
    )
    _require(len({item[2] for item in mapping_rows}) == 6, "mapping does not select six streams")
    _require(mapping_rows == manifest_rows, "READY camera rows do not match the mapping revision")

    expected_outcome = _alignment_outcome(alignment.status)
    _require(
        evaluation.alignment_outcome is expected_outcome,
        "alignment status does not match the selected ledger outcome",
    )
    _require(
        expected_outcome in policy.admissible_alignment_outcomes,
        "alignment outcome is not admissible under the supplied policy",
    )

    expected_report_digest = validation_report_semantic_digest(report)
    _require(
        context.validation_report_semantic_sha256 == expected_report_digest,
        "validation report body does not match its semantic digest",
    )
    expected_mapping_digest = camera_mapping_semantic_digest(
        mapping,
        source_content_sha256=context.source_content_sha256,
        mapping_policy_sha256=report.mapping_policy.digest,
    )
    _require(
        context.camera_mapping_semantic_sha256 == expected_mapping_digest,
        "camera mapping body does not match its semantic digest",
    )
    expected_manifest_digest = ready_manifest_semantic_digest(
        manifest,
        validation_report_semantic_sha256=context.validation_report_semantic_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
    )
    _require(
        context.ready_manifest_semantic_sha256 == expected_manifest_digest,
        "READY manifest body does not match its semantic digest",
    )
    expected_alignment_digest = alignment_run_semantic_digest(
        alignment,
        source_content_sha256=context.source_content_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
    )
    _require(
        context.alignment_semantic_sha256 == expected_alignment_digest,
        "alignment body does not match its semantic digest",
    )


def _validate_v2_cross_bindings(context: AdmittedRecordingContextV2) -> None:
    evaluation = context.evaluation
    policy = context.policy
    report = context.validation_report
    ready = context.ready_manifest
    alignment = context.alignment_manifest

    _require(evaluation.admissible, "v2 primary admission evaluation is not admissible")
    _require(
        evaluation.reason_code == "ADMISSIBLE", "v2 primary admission reason is not ADMISSIBLE"
    )
    _require(
        evaluation.source_outcome is SourceAdmissionOutcome.READY,
        "v2 primary admission did not select a READY source",
    )
    _require(
        evaluation.policy_version == policy.version
        and evaluation.policy_sha256 == policy.semantic_sha256,
        "v2 admission policy does not match the supplied policy",
    )
    _require(report.verdict is MCAPValidationVerdictV2.VALID, "v2 validation report is not VALID")
    _require(
        evaluation.ready_manifest_id == ready.ready_manifest_id == context.ready_manifest_id,
        "v2 selected READY manifest ID does not match",
    )
    _require(
        evaluation.ready_manifest_semantic_sha256
        == ready.ready_manifest_semantic_sha256
        == context.ready_manifest_semantic_sha256,
        "v2 selected READY manifest digest does not match",
    )
    _require(
        evaluation.alignment_id == alignment.alignment_id,
        "v2 selected alignment ID does not match",
    )
    _require(
        evaluation.alignment_semantic_sha256
        == alignment.alignment_semantic_sha256
        == context.alignment_semantic_sha256,
        "v2 selected alignment digest does not match",
    )
    _require(
        report.validation_report_id == ready.validation_report_id,
        "v2 READY manifest references another validation report",
    )
    _require(
        report.validation_report_semantic_sha256
        == ready.validation_report_semantic_sha256
        == context.validation_report_semantic_sha256,
        "v2 validation report digest is not bound by READY",
    )
    _require(
        report.schema_ref == ready.validation_report_schema_ref,
        "v2 READY validation schema reference differs",
    )
    _require(
        ready.ready_manifest_id == alignment.ready_manifest_id
        and ready.ready_manifest_semantic_sha256 == alignment.ready_manifest_semantic_sha256
        and ready.schema_ref == alignment.ready_manifest_schema_ref,
        "v2 alignment does not reference the selected READY manifest",
    )
    _require(
        report.mcap_id == ready.mcap_id == alignment.mcap_id,
        "v2 MCAP association does not match",
    )
    _require(
        report.recording_identity
        == ready.recording_identity
        == alignment.recording_identity
        == context.recording_identity
        == evaluation.recording_identity,
        "v2 recording identity does not match",
    )
    _require(
        report.source_content_sha256
        == ready.source_content_sha256
        == alignment.source_content_sha256
        == context.source_content_sha256,
        "v2 source content digest does not match",
    )
    _require(
        ready.source_durability.verified_sha256 == context.source_content_sha256
        and ready.source_durability.verified_bytes == ready.source.bytes,
        "v2 source durability evidence is not bound",
    )
    mapping_digest = report.camera_mapping_semantic_sha256
    _require(
        mapping_digest
        == ready.camera_mapping_semantic_sha256
        == alignment.camera_mapping_semantic_sha256
        == context.camera_mapping_semantic_sha256,
        "v2 camera mapping digest does not match",
    )
    _require(
        ready.camera_mapping_run_id == alignment.camera_mapping_run_id,
        "v2 camera mapping revision does not match",
    )
    _require(
        report.mapping_policy == ready.mapping_policy,
        "v2 mapping policy does not match",
    )

    report_rows = tuple(
        (
            item.camera_id.value,
            item.role,
            item.stream_id,
            item.stream_semantic_sha256,
        )
        for item in report.camera_mappings
    )
    ready_rows = tuple(
        (
            item.camera_id.value,
            item.role,
            item.stream_id,
            item.stream_semantic_sha256,
        )
        for item in ready.cameras
    )
    alignment_rows = tuple(
        (
            camera_id,
            camera.stream_id,
            camera.stream_semantic_sha256,
        )
        for camera_id, camera in alignment.cameras.items()
    )
    _require(
        tuple(row[0] for row in report_rows) == CAMERA_ID_VALUES,
        "v2 validation mappings are not in canonical camera order",
    )
    _require(
        report_rows == ready_rows,
        "v2 READY camera rows do not match validation mappings",
    )
    _require(
        tuple((row[0], row[2], row[3]) for row in ready_rows) == alignment_rows,
        "v2 alignment camera rows do not match READY mappings",
    )
    expected_outcome = _alignment_outcome_v2(alignment.status)
    _require(
        evaluation.alignment_outcome is expected_outcome,
        "v2 alignment status does not match admission outcome",
    )
    _require(
        expected_outcome in policy.admissible_alignment_outcomes,
        "v2 alignment outcome is not admissible under the supplied policy",
    )


def _alignment_outcome_v2(status: AlignmentStatus) -> AlignmentAdmissionOutcome:
    if status is AlignmentStatus.VALID:
        return AlignmentAdmissionOutcome.VALID
    if status is AlignmentStatus.DEGRADED:
        return AlignmentAdmissionOutcome.DEGRADED
    raise ValueError(f"alignment status {status.value} is not admissible for primary processing")


def _alignment_outcome(status: AlignmentStatus) -> AlignmentAdmissionOutcome:
    if status is AlignmentStatus.VALID:
        return AlignmentAdmissionOutcome.VALID
    if status is AlignmentStatus.DEGRADED:
        return AlignmentAdmissionOutcome.DEGRADED
    raise ValueError(f"alignment status {status.value} is not admissible for primary processing")


def _context_projection_values(
    *,
    recording_identity: str,
    source_content_sha256: str,
    validation_report_semantic_sha256: str,
    ready_manifest_semantic_sha256: str,
    camera_mapping_semantic_sha256: str,
    alignment_semantic_sha256: str,
    alignment_outcome: AlignmentAdmissionOutcome | None,
    policy_sha256: str,
) -> dict[str, object]:
    return {
        "recording_identity": recording_identity,
        "source_content_sha256": source_content_sha256,
        "validation_report_semantic_sha256": validation_report_semantic_sha256,
        "ready_manifest_semantic_sha256": ready_manifest_semantic_sha256,
        "camera_mapping_semantic_sha256": camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": alignment_semantic_sha256,
        "alignment_outcome": (alignment_outcome.value if alignment_outcome is not None else None),
        "policy_sha256": policy_sha256,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _error_message(exc: ValidationError | ValueError | TypeError) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False)
        if errors:
            return str(errors[0]["msg"]).removeprefix("Value error, ")
    return str(exc)


__all__ = [
    "AdmissionContextError",
    "AdmissionContextResolver",
    "AdmittedRecordingContext",
    "AdmittedRecordingContextV2",
    "admitted_recording_context_semantic_projection",
    "admitted_recording_context_v2_semantic_projection",
    "alignment_run_semantic_digest",
    "alignment_run_semantic_projection",
    "camera_mapping_semantic_digest",
    "camera_mapping_semantic_projection",
    "ready_manifest_semantic_digest",
    "ready_manifest_semantic_projection",
    "validation_report_semantic_digest",
    "validation_report_semantic_projection",
]
