from __future__ import annotations

import pytest

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.event_pipeline.boundary_qualification import (
    BOUNDARY_QUALIFICATION_REPORT_LOGICAL_KEY_NAMESPACE,
    BoundaryQualificationCameraDisposition,
    BoundaryQualificationCameraObservation,
    BoundaryQualificationCase,
    BoundaryQualificationEngine,
    BoundaryQualificationObservationOutcome,
    BoundaryQualificationOutcome,
    BoundaryQualificationPolicy,
    BoundaryQualificationReasonCode,
    BoundaryQualificationReport,
    BoundaryQualificationRoleInput,
    boundary_qualification_report_projection,
    verify_boundary_qualification_report,
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


def _observation(
    *,
    camera_id: CameraId,
    center_ns: int | None,
    value: int,
) -> BoundaryQualificationCameraObservation:
    if center_ns is None:
        return BoundaryQualificationCameraObservation(
            camera_id=camera_id,
            source_camera_evidence_logical_key=_key("raw-camera-evidence", value),
            source_camera_evidence_semantic_sha256=_digest(value),
            source_camera_evidence_exact_sha256=_digest(value + 100),
            outcome=BoundaryQualificationObservationOutcome.NO_BOUNDARY,
            production_eligible=False,
        )
    interval = NanosecondInterval(start_ns=center_ns - 5, end_ns=center_ns + 5)
    return BoundaryQualificationCameraObservation(
        camera_id=camera_id,
        source_camera_evidence_logical_key=_key("raw-camera-evidence", value),
        source_camera_evidence_semantic_sha256=_digest(value),
        source_camera_evidence_exact_sha256=_digest(value + 100),
        outcome=BoundaryQualificationObservationOutcome.OBSERVED,
        observed_interval=interval,
        boundary_estimate_ns=center_ns,
        uncertainty_ns=5,
        production_eligible=False,
    )


def _role_input(
    *,
    role: BoundaryRefinementRole,
    centers: tuple[int | None, ...],
    value: int,
) -> BoundaryQualificationRoleInput:
    observations = tuple(
        _observation(camera_id=camera_id, center_ns=center, value=value + ordinal)
        for ordinal, (camera_id, center) in enumerate(zip(CAMERA_IDS, centers, strict=True))
    )
    observed = tuple(center for center in centers if center is not None)
    estimate = sorted(observed)[(len(observed) - 1) // 2]
    uncertainty = max(abs(center - estimate) + 5 for center in observed)
    return BoundaryQualificationRoleInput(
        role=role,
        source_role_result_logical_key=_key("boundary-refinement-role", value + 50),
        source_role_result_semantic_sha256=_digest(value + 50),
        source_role_result_exact_sha256=_digest(value + 60),
        window_interval=NanosecondInterval(start_ns=0, end_ns=1_000),
        minimum_observed_cameras=2,
        camera_observations=observations,
        baseline_boundary_estimate_ns=estimate,
        baseline_uncertainty_ns=uncertainty,
        baseline_boundary_interval=NanosecondInterval(
            start_ns=max(0, estimate - uncertainty),
            end_ns=min(1_000, estimate + uncertainty + 1),
        ),
        baseline_outcome=BoundaryRefinementOutcome.REFINED,
        production_eligible=False,
    )


def _quality(
    *,
    camera_id: CameraId,
    value: int,
    condition: BoundaryCameraCondition = BoundaryCameraCondition.GOOD,
    applicability: BoundaryQualityApplicability = BoundaryQualityApplicability.APPLICABLE,
) -> BoundaryCameraQualityEvidence:
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
        condition=condition,
        applicability=applicability,
        quality_millionths=(
            900_000 if applicability is BoundaryQualityApplicability.APPLICABLE else None
        ),
        policy_version="boundary-qualification-fixture-v1",
    )


def _case(
    quality_evidence: tuple[BoundaryCameraQualityEvidence, ...],
) -> BoundaryQualificationCase:
    return BoundaryQualificationCase(
        source_boundary_result_logical_key=_key("boundary-refinement", 40),
        source_boundary_result_semantic_sha256=_digest(40),
        source_boundary_result_exact_sha256=_digest(41),
        source_action_logical_key=_key("provisional-action", 42),
        source_action_semantic_sha256=_digest(42),
        mcap_id="00000000-0000-0000-0000-000000000001",
        recording_identity=_digest(1),
        source_content_sha256=_digest(2),
        camera_mapping_semantic_sha256=_digest(3),
        alignment_semantic_sha256=_digest(4),
        roles=(
            _role_input(
                role=BoundaryRefinementRole.ONSET,
                centers=(100, 101, 200, None, None, None),
                value=200,
            ),
            _role_input(
                role=BoundaryRefinementRole.OFFSET,
                centers=(300, 301, 400, None, None, None),
                value=300,
            ),
        ),
        quality_evidence=quality_evidence,
        production_eligible=False,
    )


def _quality_set() -> tuple[BoundaryCameraQualityEvidence, ...]:
    return tuple(
        _quality(
            camera_id=camera_id,
            value=500 + ordinal,
            condition=(
                BoundaryCameraCondition.DEGRADED
                if camera_id is CameraId.CAM_03
                else BoundaryCameraCondition.GOOD
            ),
        )
        for ordinal, camera_id in enumerate(CAMERA_IDS)
    )


def _policy(
    *, coverage: bool = False, minimum_observed_cameras: int = 2
) -> BoundaryQualificationPolicy:
    return BoundaryQualificationPolicy.create(
        version="boundary-qualification-fixture-v1",
        minimum_observed_cameras=minimum_observed_cameras,
        excluded_conditions=(BoundaryCameraCondition.DEGRADED,),
        calibrated_coverage_evidence_sha256=_digest(900) if coverage else None,
    )


def test_missing_quality_retains_exact_authoritative_baseline_and_replays() -> None:
    case = _case(_quality_set()[:1])
    report = BoundaryQualificationEngine(_policy()).compare(case)

    assert report.outcome is BoundaryQualificationOutcome.QUALITY_NOT_APPLIED
    assert all(
        comparison.candidate == comparison.baseline for comparison in report.role_comparisons
    )
    assert report.role_comparisons[0].camera_decisions[1].disposition is (
        BoundaryQualificationCameraDisposition.QUALITY_MISSING
    )
    assert verify_boundary_qualification_report(report) == report


def test_inapplicable_quality_retains_baseline_without_inventing_fallback() -> None:
    qualities = list(_quality_set())
    qualities[1] = _quality(
        camera_id=CameraId.CAM_02,
        value=701,
        applicability=BoundaryQualityApplicability.NOT_APPLICABLE,
    )
    report = BoundaryQualificationEngine(_policy()).compare(_case(tuple(qualities)))

    assert report.outcome is BoundaryQualificationOutcome.QUALITY_NOT_APPLIED
    assert all(
        comparison.candidate == comparison.baseline for comparison in report.role_comparisons
    )
    assert report.role_comparisons[0].camera_decisions[1].disposition is (
        BoundaryQualificationCameraDisposition.QUALITY_NOT_APPLICABLE
    )


def test_exclusion_that_leaves_too_few_cameras_is_indeterminate_without_fallback() -> None:
    report = BoundaryQualificationEngine(_policy(minimum_observed_cameras=3)).compare(
        _case(_quality_set())
    )

    assert report.outcome is BoundaryQualificationOutcome.CANDIDATE_INDETERMINATE
    assert all(
        comparison.candidate.outcome is BoundaryRefinementOutcome.INDETERMINATE
        for comparison in report.role_comparisons
    )
    assert report.role_comparisons[0].camera_decisions[2].disposition is (
        BoundaryQualificationCameraDisposition.EXCLUDED_CONDITION
    )


def test_narrowing_candidate_requires_calibrated_coverage_evidence() -> None:
    case = _case(_quality_set())
    retained = BoundaryQualificationEngine(_policy()).compare(case)
    qualified = BoundaryQualificationEngine(_policy(coverage=True)).compare(case)

    assert retained.outcome is BoundaryQualificationOutcome.BASELINE_RETAINED
    assert all(
        comparison.candidate == comparison.baseline for comparison in retained.role_comparisons
    )
    assert retained.role_comparisons[0].reason_code is (
        BoundaryQualificationReasonCode.NARROWING_REJECTED_NO_CALIBRATED_COVERAGE
    )
    assert qualified.outcome is BoundaryQualificationOutcome.CANDIDATE_REPORTED
    assert qualified.role_comparisons[0].candidate.boundary_estimate_ns == 100
    assert qualified.role_comparisons[0].candidate.uncertainty_ns == 6
    assert qualified.role_comparisons[0].candidate.boundary_interval == NanosecondInterval(
        start_ns=94,
        end_ns=107,
    )


def test_six_camera_baseline_and_content_addressed_replay_are_strict() -> None:
    case = _case(_quality_set())
    role = case.roles[0]
    with pytest.raises(ValueError, match="all six ordered cameras"):
        BoundaryQualificationRoleInput.model_validate(
            {
                **role.model_dump(mode="python"),
                "camera_observations": role.camera_observations[:-1],
            },
            strict=True,
        )

    report = BoundaryQualificationEngine(_policy(coverage=True)).compare(case)
    changed = report.role_comparisons[0].model_copy(
        update={"candidate": report.role_comparisons[0].baseline}
    )
    values: dict[str, object] = {
        "case": report.case,
        "policy": report.policy,
        "role_comparisons": (changed, report.role_comparisons[1]),
        "outcome": report.outcome,
        "projection_version": report.projection_version,
        "semantic_sha256": "0" * 64,
        "logical_key": (f"{BOUNDARY_QUALIFICATION_REPORT_LOGICAL_KEY_NAMESPACE}:{'0' * 64}"),
        "production_eligible": False,
    }
    draft = BoundaryQualificationReport.model_construct(**values)
    digest = semantic_sha256(boundary_qualification_report_projection(draft))
    tampered = BoundaryQualificationReport.model_validate(
        {
            **values,
            "semantic_sha256": digest,
            "logical_key": f"{BOUNDARY_QUALIFICATION_REPORT_LOGICAL_KEY_NAMESPACE}:{digest}",
        },
        strict=True,
    )
    with pytest.raises(ValueError, match="deterministic policy replay"):
        verify_boundary_qualification_report(tampered)
