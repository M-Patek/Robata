from __future__ import annotations

import pytest

from robata.benchmark.boundary_qualification import (
    BoundaryQualificationFixtureTruth,
    build_boundary_qualification_fixture_metrics,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.event_pipeline.boundary_qualification import (
    BoundaryQualificationCameraObservation,
    BoundaryQualificationCase,
    BoundaryQualificationEngine,
    BoundaryQualificationObservationOutcome,
    BoundaryQualificationPolicy,
    BoundaryQualificationRoleInput,
)
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementOutcome,
    BoundaryRefinementRole,
)
from robata.qa_pipeline.boundary_quality import (
    BoundaryCameraCondition,
    BoundaryCameraQualityEvidence,
    BoundaryQualityApplicability,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _key(namespace: str, value: int) -> str:
    return f"{namespace}:{_digest(value)}"


def _role(role: BoundaryRefinementRole, value: int, center: int) -> BoundaryQualificationRoleInput:
    observations: list[BoundaryQualificationCameraObservation] = []
    for ordinal, camera_id in enumerate(CAMERA_IDS):
        source = value + ordinal
        if ordinal < 2:
            observed_center = center + ordinal * 2
            observations.append(
                BoundaryQualificationCameraObservation(
                    camera_id=camera_id,
                    source_camera_evidence_logical_key=_key("camera-evidence", source),
                    source_camera_evidence_semantic_sha256=_digest(source),
                    source_camera_evidence_exact_sha256=_digest(source + 100),
                    outcome=BoundaryQualificationObservationOutcome.OBSERVED,
                    observed_interval=NanosecondInterval(
                        start_ns=observed_center - 5,
                        end_ns=observed_center + 5,
                    ),
                    boundary_estimate_ns=observed_center,
                    uncertainty_ns=5,
                )
            )
        else:
            observations.append(
                BoundaryQualificationCameraObservation(
                    camera_id=camera_id,
                    source_camera_evidence_logical_key=_key("camera-evidence", source),
                    source_camera_evidence_semantic_sha256=_digest(source),
                    source_camera_evidence_exact_sha256=_digest(source + 100),
                    outcome=BoundaryQualificationObservationOutcome.NO_BOUNDARY,
                )
            )
    return BoundaryQualificationRoleInput(
        role=role,
        source_role_result_logical_key=_key("boundary-refinement-role", value + 20),
        source_role_result_semantic_sha256=_digest(value + 20),
        source_role_result_exact_sha256=_digest(value + 120),
        window_interval=NanosecondInterval(start_ns=0, end_ns=1_000),
        minimum_observed_cameras=2,
        camera_observations=tuple(observations),
        baseline_boundary_estimate_ns=center,
        baseline_uncertainty_ns=7,
        baseline_boundary_interval=NanosecondInterval(start_ns=center - 7, end_ns=center + 8),
        baseline_outcome=BoundaryRefinementOutcome.REFINED,
    )


def _quality(camera_id: CameraId, value: int) -> BoundaryCameraQualityEvidence:
    return BoundaryCameraQualityEvidence.create(
        mcap_id="00000000-0000-0000-0000-000000000001",
        recording_identity=_digest(1),
        source_content_sha256=_digest(2),
        camera_mapping_semantic_sha256=_digest(3),
        alignment_semantic_sha256=_digest(4),
        camera_id=camera_id,
        qa_result_logical_key=_key("qa-result", value),
        qa_result_semantic_sha256=_digest(value),
        qa_result_exact_sha256=_digest(value + 100),
        condition=BoundaryCameraCondition.GOOD,
        applicability=BoundaryQualityApplicability.APPLICABLE,
        quality_millionths=900_000,
        policy_version="p12-fixture-v1",
    )


def _report():
    onset = _role(BoundaryRefinementRole.ONSET, 100, 100)
    offset = _role(BoundaryRefinementRole.OFFSET, 200, 300)
    case = BoundaryQualificationCase(
        source_boundary_result_logical_key=_key("boundary-refinement", 50),
        source_boundary_result_semantic_sha256=_digest(50),
        source_boundary_result_exact_sha256=_digest(51),
        source_action_logical_key=_key("action", 52),
        source_action_semantic_sha256=_digest(52),
        mcap_id="00000000-0000-0000-0000-000000000001",
        recording_identity=_digest(1),
        source_content_sha256=_digest(2),
        camera_mapping_semantic_sha256=_digest(3),
        alignment_semantic_sha256=_digest(4),
        roles=(onset, offset),
        quality_evidence=tuple(
            _quality(camera_id, 500 + ordinal) for ordinal, camera_id in enumerate(CAMERA_IDS)
        ),
    )
    return BoundaryQualificationEngine(
        BoundaryQualificationPolicy.create(
            version="p12-fixture-v1",
            excluded_conditions=(),
        )
    ).compare(case)


def _truth(report) -> tuple[BoundaryQualificationFixtureTruth, ...]:
    return tuple(
        BoundaryQualificationFixtureTruth.create(
            source_role_result_logical_key=role.source_role_result_logical_key,
            source_role_result_semantic_sha256=role.source_role_result_semantic_sha256,
            role=role.role,
            action_label="turn",
            camera_condition=BoundaryCameraCondition.GOOD,
            boundary_estimate_ns=(101 if role.role is BoundaryRefinementRole.ONSET else 301),
        )
        for role in report.case.roles
    )


def test_fixture_metrics_are_stratified_and_non_representative() -> None:
    report = _report()
    metrics = build_boundary_qualification_fixture_metrics(
        fixture_id="p12-boundary-fixture",
        report=report,
        truth=_truth(report),
    )

    assert metrics.representative_measurement_status == "NOT_MEASURED"
    assert metrics.production_eligible is False
    assert len(metrics.strata) == 2
    assert {item.role for item in metrics.strata} == {
        BoundaryRefinementRole.ONSET,
        BoundaryRefinementRole.OFFSET,
    }
    assert all(item.action_label == "turn" for item in metrics.strata)
    assert all(item.camera_condition is BoundaryCameraCondition.GOOD for item in metrics.strata)
    assert all(item.baseline_mae_ns == 1.0 for item in metrics.strata)
    assert all(item.candidate_mae_ns == 1.0 for item in metrics.strata)
    assert all(item.baseline_interval_coverage == 1.0 for item in metrics.strata)
    assert metrics.logical_key.endswith(metrics.semantic_sha256)


def test_fixture_metrics_reject_truth_that_does_not_cover_the_report_roles() -> None:
    report = _report()
    with pytest.raises(ValueError, match="cover exactly"):
        build_boundary_qualification_fixture_metrics(
            fixture_id="p12-incomplete-truth",
            report=report,
            truth=_truth(report)[:1],
        )
