from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.admission import (
    AdmissionContextError,
    AdmissionContextResolver,
    AlignmentAdmissionOutcome,
    PrimaryAdmissionEvaluation,
    PrimaryAdmissionPolicy,
    SourceAdmissionOutcome,
    admitted_recording_context_semantic_projection,
    alignment_run_semantic_digest,
    camera_mapping_semantic_digest,
    ready_manifest_semantic_digest,
    validation_report_semantic_digest,
)
from robata.alignment import AlignmentService, AlignmentStatus
from robata.contracts.cameras import CameraId, SixCameraMap
from robata.contracts.hashing import semantic_sha256
from robata.contracts.mcap import (
    CameraMapping,
    CameraMappingRun,
    MCAPMappingPolicyReference,
    MCAPReadyCamera,
    MCAPReadyManifest,
    MCAPReadyRecording,
    MCAPReadySource,
    MCAPValidationError,
    MCAPValidationReport,
    MCAPValidationSource,
    MCAPValidationVerdict,
)

NOW = "2026-07-19T16:00:00Z"
CONTENT = "a" * 64
RECORDING_IDENTITY = "b" * 64
MCAP_ID = str(UUID(int=1))
REPORT_ID = str(UUID(int=2))
READY_ID = str(UUID(int=3))
MAPPING_ID = str(UUID(int=4))
MAPPING_POLICY_DIGEST = "c" * 64


def _uuid(value: int) -> str:
    return str(UUID(int=value))


@dataclass(frozen=True)
class _Fixture:
    report: MCAPValidationReport
    report_digest: str
    manifest: MCAPReadyManifest
    manifest_digest: str
    mapping: CameraMappingRun
    mapping_digest: str
    alignment: object
    alignment_digest: str
    policy: PrimaryAdmissionPolicy
    evaluation: PrimaryAdmissionEvaluation

    def resolve(self, **updates: object):
        values: dict[str, object] = {
            "evaluation": self.evaluation,
            "policy": self.policy,
            "ready_manifest_id": READY_ID,
            "validation_report": self.report,
            "validation_report_semantic_sha256": self.report_digest,
            "ready_manifest": self.manifest,
            "ready_manifest_semantic_sha256": self.manifest_digest,
            "camera_mapping_run": self.mapping,
            "camera_mapping_semantic_sha256": self.mapping_digest,
            "alignment_run": self.alignment,
            "alignment_semantic_sha256": self.alignment_digest,
        }
        values.update(updates)
        return AdmissionContextResolver().resolve(**values)  # type: ignore[arg-type]


def _fixture(*, degraded: bool = False) -> _Fixture:
    streams = tuple(_uuid(100 + index) for index in range(1, 7))
    mapping = CameraMappingRun(
        mapping_run_id=MAPPING_ID,
        mcap_id=MCAP_ID,
        mapping_policy_version="mapping-v1",
        status="PUBLISHED",
        created_at=NOW,
        cameras=tuple(
            CameraMapping(
                camera_id=f"cam_{index:02d}",
                role=f"view-{index}",
                stream_id=streams[index - 1],
            )
            for index in range(1, 7)
        ),
    )
    report = MCAPValidationReport(
        schema_version="1.0",
        validation_report_id=REPORT_ID,
        mcap_id=MCAP_ID,
        recording_identity=RECORDING_IDENTITY,
        source=MCAPValidationSource(
            uri="file:///original/source.mcap",
            version="object-v1",
            sha256=CONTENT,
            bytes=123,
        ),
        mapping_policy=MCAPMappingPolicyReference(
            version="mapping-v1",
            digest=MAPPING_POLICY_DIGEST,
        ),
        verdict=MCAPValidationVerdict.VALID,
        discovered_video_stream_count=6,
        mapped_camera_count=6,
        errors=(),
        validated_at=NOW,
    )
    report_digest = validation_report_semantic_digest(report)
    mapping_digest = camera_mapping_semantic_digest(
        mapping,
        source_content_sha256=CONTENT,
        mapping_policy_sha256=MAPPING_POLICY_DIGEST,
    )
    manifest = MCAPReadyManifest(
        schema_version="1.0",
        mcap_id=MCAP_ID,
        validation_report_id=REPORT_ID,
        source=MCAPReadySource(
            uri="file:///original/source.mcap",
            version="object-v1",
            sha256=CONTENT,
            bytes=123,
        ),
        recording=MCAPReadyRecording(
            start_utc=NOW,
            end_utc="2026-07-19T16:01:00Z",
            duration_ns=60_000_000_000,
            timebase="mcap_log_time_ns",
        ),
        camera_count=6,
        camera_mapping_run_id=MAPPING_ID,
        camera_mapping_version="mapping-v1",
        cameras=tuple(
            MCAPReadyCamera(
                camera_id=CameraId(f"cam_{index:02d}"),
                role=f"view-{index}",
                stream_id=streams[index - 1],
                topic=f"/camera/{index}",
                channel_id=index,
                codec="h264",
                width=640,
                height=480,
                nominal_fps=30.0,
                source_start_ns=1_000,
                source_end_ns=4_000,
                frame_count=3,
            )
            for index in range(1, 7)
        ),
        ingested_at=NOW,
    )
    manifest_digest = ready_manifest_semantic_digest(
        manifest,
        validation_report_semantic_sha256=report_digest,
        camera_mapping_semantic_sha256=mapping_digest,
    )
    timestamp_values = {
        f"cam_{index:02d}": [1_000, 2_000, 500 if degraded and index == 2 else 3_000]
        for index in range(1, 7)
    }
    timestamps = SixCameraMap.model_validate(timestamp_values, strict=True)
    alignment = AlignmentService(
        "alignment-v1",
        verified_methods=("mcap_log_time",),
        clock=lambda: datetime(2026, 7, 19, 16, tzinfo=UTC),
    ).align_recording(
        mcap_id=MCAP_ID,
        camera_mapping_run_id=MAPPING_ID,
        source_content_sha256=CONTENT,
        camera_mapping_semantic_sha256=mapping_digest,
        stream_timestamps=timestamps,
        recording_start_utc=NOW,
    )
    expected_outcome = (
        AlignmentAdmissionOutcome.DEGRADED if degraded else AlignmentAdmissionOutcome.VALID
    )
    assert alignment.status.value == expected_outcome.value
    alignment_digest = alignment_run_semantic_digest(
        alignment,
        source_content_sha256=CONTENT,
        camera_mapping_semantic_sha256=mapping_digest,
    )
    policy = PrimaryAdmissionPolicy.create(
        version="primary-v1",
        admissible_alignment_outcomes=(
            AlignmentAdmissionOutcome.VALID,
            AlignmentAdmissionOutcome.DEGRADED,
        ),
    )
    evaluation = PrimaryAdmissionEvaluation(
        recording_identity=RECORDING_IDENTITY,
        ready_manifest_id=READY_ID,
        ready_manifest_semantic_sha256=manifest_digest,
        source_outcome=SourceAdmissionOutcome.READY,
        alignment_outcome=expected_outcome,
        alignment_id=alignment.alignment_id,
        alignment_semantic_sha256=alignment_digest,
        policy_version=policy.version,
        policy_sha256=policy.semantic_sha256,
        admissible=True,
        reason_code="ADMISSIBLE",
    )
    return _Fixture(
        report=report,
        report_digest=report_digest,
        manifest=manifest,
        manifest_digest=manifest_digest,
        mapping=mapping,
        mapping_digest=mapping_digest,
        alignment=alignment,
        alignment_digest=alignment_digest,
        policy=policy,
        evaluation=evaluation,
    )


def test_resolves_frozen_cross_bound_context() -> None:
    context = _fixture().resolve()

    assert context.recording_identity == RECORDING_IDENTITY
    assert context.source_content_sha256 == CONTENT
    assert context.ready_manifest_id == READY_ID
    assert context.semantic_sha256 == semantic_sha256(
        admitted_recording_context_semantic_projection(context)
    )
    with pytest.raises(ValidationError, match="frozen"):
        context.source_content_sha256 = "d" * 64


def test_rejects_forged_manifest_body_under_selected_digest() -> None:
    fixture = _fixture()
    forged_recording = fixture.manifest.recording.model_copy(
        update={"duration_ns": fixture.manifest.recording.duration_ns + 1}
    )
    forged_manifest = fixture.manifest.model_copy(update={"recording": forged_recording})

    with pytest.raises(AdmissionContextError, match="READY manifest body"):
        fixture.resolve(ready_manifest=forged_manifest)


def test_alias_relocation_does_not_change_semantic_admission() -> None:
    fixture = _fixture()
    moved_report = fixture.report.model_copy(
        update={
            "source": fixture.report.source.model_copy(
                update={"uri": "file:///moved/source.mcap", "version": "object-v2"}
            )
        }
    )
    moved_manifest = fixture.manifest.model_copy(
        update={
            "source": fixture.manifest.source.model_copy(
                update={"uri": "file:///moved/source.mcap", "version": "object-v2"}
            )
        }
    )

    context = fixture.resolve(
        validation_report=moved_report,
        ready_manifest=moved_manifest,
    )

    assert context.validation_report_semantic_sha256 == fixture.report_digest
    assert context.ready_manifest_semantic_sha256 == fixture.manifest_digest


def test_rejects_mcap_and_ready_artifact_mismatches() -> None:
    fixture = _fixture()
    forged_alignment = fixture.alignment.model_copy(update={"mcap_id": _uuid(999)})
    with pytest.raises(AdmissionContextError, match="MCAP association"):
        fixture.resolve(alignment_run=forged_alignment)

    with pytest.raises(AdmissionContextError, match="selected READY manifest ID"):
        fixture.resolve(ready_manifest_id=_uuid(998))


@pytest.mark.parametrize(
    "verdict",
    [MCAPValidationVerdict.INVALID, MCAPValidationVerdict.INCONCLUSIVE],
)
def test_rejects_nonvalid_validation_report(verdict: MCAPValidationVerdict) -> None:
    fixture = _fixture()
    diagnostic = MCAPValidationError(
        code="SOURCE_CHECK_FAILED",
        message="fixture diagnostic",
        path=None,
        camera_id=None,
        stream_id=None,
    )
    report = fixture.report.model_copy(update={"verdict": verdict, "errors": (diagnostic,)})

    with pytest.raises(AdmissionContextError, match="validation report is not VALID"):
        fixture.resolve(validation_report=report)


@pytest.mark.parametrize("status", [AlignmentStatus.INVALID, AlignmentStatus.UNVERIFIED])
def test_rejects_nonadmissible_alignment_body(status: AlignmentStatus) -> None:
    fixture = _fixture()
    alignment = fixture.alignment.model_copy(update={"status": status})

    with pytest.raises(AdmissionContextError, match="not admissible for primary processing"):
        fixture.resolve(alignment_run=alignment)


def test_degraded_alignment_requires_explicit_consuming_policy() -> None:
    fixture = _fixture(degraded=True)
    assert fixture.resolve().evaluation.alignment_outcome is AlignmentAdmissionOutcome.DEGRADED

    valid_only = PrimaryAdmissionPolicy.create(
        version="primary-valid-only-v1",
        admissible_alignment_outcomes=(AlignmentAdmissionOutcome.VALID,),
    )
    evaluation = PrimaryAdmissionEvaluation(
        **{
            **fixture.evaluation.model_dump(mode="python"),
            "policy_version": valid_only.version,
            "policy_sha256": valid_only.semantic_sha256,
        }
    )
    with pytest.raises(AdmissionContextError, match="not admissible under the supplied policy"):
        fixture.resolve(policy=valid_only, evaluation=evaluation)
