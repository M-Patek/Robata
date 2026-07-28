from __future__ import annotations

from pathlib import Path

import pytest

from robata.application.canonical.boundary_qualification import (
    BoundaryQualificationDispatchStatus,
    BoundaryQualificationPublicationStatus,
    BoundaryQualificationSidecarConflict,
    BoundaryQualificationSidecarStorageError,
    BoundaryQualificationSidecarStore,
    CanonicalBoundaryQualificationBridge,
    CanonicalBoundaryQualificationJob,
    CanonicalBoundaryQualificationWorker,
    boundary_qualification_case_from_execution,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes
from robata.event_pipeline.boundary_qualification import (
    BoundaryQualificationOutcome,
    BoundaryQualificationPolicy,
)
from robata.qa_pipeline.boundary_quality import (
    BoundaryCameraCondition,
    BoundaryCameraQualityEvidence,
    BoundaryQualityApplicability,
)
from tests.integration.test_canonical_offline import _claim_bytes, _harness, _run


def _boundary_execution(tmp_path: Path):
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path / "registry")
    result = _run(harness)
    assert len(result.boundary_refinement_executions) == 1
    return result.boundary_refinement_executions[0]


def _policy() -> BoundaryQualificationPolicy:
    return BoundaryQualificationPolicy.create(
        version="canonical-boundary-qualification-test-v1",
        excluded_conditions=(),
    )


def _quality_evidence(execution) -> tuple[BoundaryCameraQualityEvidence, ...]:
    role = execution.onset.role_result
    return tuple(
        BoundaryCameraQualityEvidence.create(
            mcap_id=role.mcap_id,
            recording_identity=role.recording_identity,
            source_content_sha256=role.source_content_sha256,
            camera_mapping_semantic_sha256=role.camera_mapping_semantic_sha256,
            alignment_semantic_sha256=role.alignment_semantic_sha256,
            camera_id=camera_id,
            qa_result_logical_key=f"p12-quality:{ordinal + 1:064x}",
            qa_result_semantic_sha256=f"{ordinal + 1:064x}",
            qa_result_exact_sha256=f"{ordinal + 101:064x}",
            condition=BoundaryCameraCondition.GOOD,
            applicability=BoundaryQualityApplicability.APPLICABLE,
            quality_millionths=900_000,
            policy_version="canonical-boundary-qualification-test-v1",
        )
        for ordinal, camera_id in enumerate(CAMERA_IDS)
    )


def test_detached_sidecar_seals_exact_boundary_execution_then_recovers_report(
    tmp_path: Path,
) -> None:
    execution = _boundary_execution(tmp_path)
    execution_bytes = canonical_json_bytes(execution)
    store = BoundaryQualificationSidecarStore(tmp_path / "sidecar")
    bridge = CanonicalBoundaryQualificationBridge(store)

    dispatched = bridge.enqueue(
        execution=execution,
        policy=_policy(),
        quality_evidence=_quality_evidence(execution),
    )

    assert dispatched.status is BoundaryQualificationDispatchStatus.ENQUEUED
    assert dispatched.replayed is False
    assert execution_bytes == canonical_json_bytes(execution)
    assert store.job_path(dispatched.job.semantic_sha256).read_bytes() == canonical_json_bytes(
        dispatched.job
    )

    first_worker = CanonicalBoundaryQualificationWorker(store)
    first = first_worker.drain()

    assert len(first) == 1
    assert first[0].status is BoundaryQualificationPublicationStatus.PUBLISHED
    assert first[0].report.case.source_boundary_result_logical_key == execution.result.logical_key
    assert first[0].report.case.source_boundary_result_exact_sha256
    assert first[0].report.outcome is BoundaryQualificationOutcome.CANDIDATE_REPORTED
    assert first[0].report.production_eligible is False
    assert b"NO_EVENTS" not in canonical_json_bytes(first[0].report)

    restarted_store = BoundaryQualificationSidecarStore(tmp_path / "sidecar")
    replay = CanonicalBoundaryQualificationWorker(restarted_store).drain()

    assert len(replay) == 1
    assert replay[0].status is BoundaryQualificationPublicationStatus.REPLAYED
    assert replay[0].report == first[0].report
    assert restarted_store.get_report(first[0].report.semantic_sha256) == first[0].report


def test_missing_quality_is_a_durable_baseline_comparison_not_an_event_claim(
    tmp_path: Path,
) -> None:
    execution = _boundary_execution(tmp_path)
    store = BoundaryQualificationSidecarStore(tmp_path / "sidecar")

    dispatched = CanonicalBoundaryQualificationBridge(store).enqueue(
        execution=execution,
        policy=_policy(),
    )
    published = CanonicalBoundaryQualificationWorker(store).drain()

    assert dispatched.status is BoundaryQualificationDispatchStatus.ENQUEUED
    assert len(published) == 1
    report = published[0].report
    assert report.outcome is BoundaryQualificationOutcome.QUALITY_NOT_APPLIED
    assert all(item.candidate == item.baseline for item in report.role_comparisons)
    assert report.case.quality_evidence == ()
    assert report.production_eligible is False


def test_sidecar_rejects_tampered_job_and_report_bytes(tmp_path: Path) -> None:
    execution = _boundary_execution(tmp_path)
    store = BoundaryQualificationSidecarStore(tmp_path / "sidecar")
    dispatch = CanonicalBoundaryQualificationBridge(store).enqueue(
        execution=execution,
        policy=_policy(),
    )
    publication = CanonicalBoundaryQualificationWorker(store).drain()[0]

    store.job_path(dispatch.job.semantic_sha256).write_bytes(b"{}")
    with pytest.raises(BoundaryQualificationSidecarStorageError, match="invalid boundary"):
        store.get_job(dispatch.job.semantic_sha256)

    conflicting_job = CanonicalBoundaryQualificationJob.create(
        case=boundary_qualification_case_from_execution(execution=execution),
        policy=BoundaryQualificationPolicy.create(
            version="canonical-boundary-qualification-conflict-v1",
            excluded_conditions=(),
        ),
    )
    store.job_path(dispatch.job.semantic_sha256).write_bytes(canonical_json_bytes(conflicting_job))
    with pytest.raises(BoundaryQualificationSidecarConflict, match="different immutable"):
        store.put_or_get_job(dispatch.job)

    store.report_path(publication.report.semantic_sha256).write_bytes(b"{}")
    with pytest.raises(BoundaryQualificationSidecarStorageError, match="invalid boundary"):
        store.get_report(publication.report.semantic_sha256)
