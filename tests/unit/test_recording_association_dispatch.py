from __future__ import annotations

from pathlib import Path

from robata.application.canonical.recording_association import (
    RecordingAssociationPublicationStatus,
    RecordingAssociationReportStore,
)
from robata.application.canonical.recording_association_dispatch import (
    CanonicalRecordingAssociationJob,
    CanonicalRecordingAssociationWorker,
    RecordingAssociationJobStore,
)
from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes
from robata.event_pipeline.recording_association import (
    AssociationAcceptedEvidenceRef,
    AssociationSourceActionRef,
    CompletedRecordingAssociationBinding,
    RecordingAssociationInput,
    RecordingAssociationPolicy,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _job() -> CanonicalRecordingAssociationJob:
    recording = CompletedRecordingAssociationBinding(
        completed_run_id="00000000-0000-4000-8000-000000000001",
        completed_recording_logical_key=(f"canonical-completed-recording-v1:{_digest(1)}"),
        completed_recording_semantic_sha256=_digest(1),
        completed_recording_exact_sha256=_digest(2),
        mcap_id="00000000-0000-4000-8000-000000000002",
        source_content_sha256=_digest(3),
        camera_mapping_semantic_sha256=_digest(4),
        alignment_semantic_sha256=_digest(5),
    )
    source_action = AssociationSourceActionRef(
        source_action_logical_key=f"action-evidence:{_digest(6)}",
        source_action_semantic_sha256=_digest(6),
    )
    evidence = AssociationAcceptedEvidenceRef(
        source_action=source_action,
        accepted_evidence_logical_key=f"action-observation:{_digest(7)}",
        accepted_evidence_semantic_sha256=_digest(7),
        accepted_evidence_exact_sha256=_digest(8),
        camera_id=CameraId.CAM_01,
        interval=NanosecondInterval(start_ns=10, end_ns=20),
        label="turn",
        confidence_millionths=900_000,
    )
    input_value = RecordingAssociationInput.create(
        source_action=source_action,
        mcap_id=recording.mcap_id,
        source_content_sha256=recording.source_content_sha256,
        camera_mapping_semantic_sha256=recording.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=recording.alignment_semantic_sha256,
        interval=NanosecondInterval(start_ns=10, end_ns=20),
        label="turn",
        accepted_evidence=(evidence,),
    )
    return CanonicalRecordingAssociationJob.create(
        recording=recording,
        policy=RecordingAssociationPolicy.create(
            version="recording-association-dispatch-test-v1",
        ),
        inputs=(input_value,),
    )


def test_detached_job_store_and_worker_replay_exact_derived_report(tmp_path: Path) -> None:
    job = _job()
    root = tmp_path / "recording-association"
    jobs = RecordingAssociationJobStore(root)

    stored, replayed = jobs.put_or_get(job)
    assert replayed is False
    assert stored == job
    assert jobs.job_path(job.semantic_sha256).read_bytes() == canonical_json_bytes(job)
    assert jobs.list_jobs() == (job,)

    worker = CanonicalRecordingAssociationWorker(
        jobs=jobs,
        reports=RecordingAssociationReportStore(root),
    )
    first = worker.drain()
    assert len(first) == 1
    assert first[0].publication.status is RecordingAssociationPublicationStatus.PUBLISHED
    assert first[0].publication.replayed is False
    assert first[0].publication.report is not None
    assert first[0].publication.report.recording == job.recording

    replay = worker.drain()
    assert len(replay) == 1
    assert replay[0].publication.status is RecordingAssociationPublicationStatus.REPLAYED
    assert replay[0].publication.replayed is True
    assert replay[0].publication.report == first[0].publication.report
