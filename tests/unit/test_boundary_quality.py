from __future__ import annotations

import pytest

from robata.contracts.cameras import CameraId
from robata.qa_pipeline.boundary_quality import (
    BoundaryCameraCondition,
    BoundaryCameraQualityEvidence,
    BoundaryQualityApplicability,
    boundary_camera_quality_evidence_projection,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _key(namespace: str, value: int) -> str:
    return f"{namespace}:{_digest(value)}"


def _evidence(
    *,
    applicability: BoundaryQualityApplicability = BoundaryQualityApplicability.APPLICABLE,
    quality_millionths: int | None = 900_000,
    with_calibration: bool = True,
) -> BoundaryCameraQualityEvidence:
    return BoundaryCameraQualityEvidence.create(
        mcap_id="00000000-0000-0000-0000-000000000001",
        recording_identity=_digest(1),
        source_content_sha256=_digest(2),
        camera_mapping_semantic_sha256=_digest(3),
        alignment_semantic_sha256=_digest(4),
        camera_id=CameraId.CAM_01,
        qa_result_logical_key=_key("qa-result", 5),
        qa_result_semantic_sha256=_digest(5),
        qa_result_exact_sha256=_digest(6),
        condition=BoundaryCameraCondition.GOOD,
        applicability=applicability,
        quality_millionths=quality_millionths,
        policy_version="boundary-quality-fixture-v1",
        calibration_association_logical_key=(
            _key("calibration-association", 7) if with_calibration else None
        ),
        calibration_association_semantic_sha256=_digest(7) if with_calibration else None,
        calibration_association_exact_sha256=_digest(8) if with_calibration else None,
    )


def test_applicable_quality_is_content_addressed_and_explicitly_non_geometric() -> None:
    evidence = _evidence()
    projection = boundary_camera_quality_evidence_projection(evidence)

    assert evidence.logical_key.endswith(evidence.semantic_sha256)
    assert evidence.production_eligible is False
    assert projection["geometry_excluded"] is True
    assert "interval" not in projection
    assert evidence.calibration_association_exact_sha256 == _digest(8)


@pytest.mark.parametrize(
    ("applicability", "quality_millionths"),
    [
        (BoundaryQualityApplicability.MISSING, None),
        (BoundaryQualityApplicability.NOT_APPLICABLE, None),
    ],
)
def test_missing_or_inapplicable_quality_remains_explicit(
    applicability: BoundaryQualityApplicability,
    quality_millionths: int | None,
) -> None:
    evidence = _evidence(
        applicability=applicability,
        quality_millionths=quality_millionths,
        with_calibration=False,
    )

    assert evidence.applicability is applicability
    assert evidence.quality_millionths is None


def test_applicable_quality_requires_value_and_complete_calibration_citation() -> None:
    with pytest.raises(ValueError, match="requires a quality value"):
        _evidence(
            applicability=BoundaryQualityApplicability.APPLICABLE,
            quality_millionths=None,
        )

    base = _evidence()
    with pytest.raises(ValueError, match="complete or absent"):
        BoundaryCameraQualityEvidence.model_validate(
            {
                **base.model_dump(mode="python"),
                "calibration_association_exact_sha256": None,
            },
            strict=True,
        )


def test_tampered_quality_identity_or_foreign_qa_key_is_rejected() -> None:
    base = _evidence()
    with pytest.raises(ValueError, match="semantic identity"):
        BoundaryCameraQualityEvidence.model_validate(
            {**base.model_dump(mode="python"), "quality_millionths": 100_000},
            strict=True,
        )
    with pytest.raises(ValueError, match="QA result logical key"):
        BoundaryCameraQualityEvidence.model_validate(
            {
                **base.model_dump(mode="python"),
                "qa_result_logical_key": _key("qa-result", 999),
            },
            strict=True,
        )
