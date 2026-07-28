"""Local-only metrics for P12 boundary-qualification candidate reports.

The authoritative boundary reducer is not changed by this module.  It compares
a replay-verified candidate report with explicitly supplied fixture truth and
keeps every result non-representative until P15 provides governed labels.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import NanosecondInterval, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey
from robata.event_pipeline.boundary_qualification import (
    BoundaryQualificationReduction,
    BoundaryQualificationReport,
    verify_boundary_qualification_report,
)
from robata.event_pipeline.boundary_refinement import BoundaryRefinementRole
from robata.qa_pipeline.boundary_quality import BoundaryCameraCondition

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]

BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_PROJECTION_VERSION: Final = (
    "boundary-qualification-fixture-truth-semantic-v1"
)
BOUNDARY_QUALIFICATION_FIXTURE_METRICS_PROJECTION_VERSION: Final = (
    "boundary-qualification-fixture-metrics-semantic-v1"
)
BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_KEY_NAMESPACE: Final = (
    "boundary-qualification-fixture-truth-v1"
)
BOUNDARY_QUALIFICATION_FIXTURE_METRICS_KEY_NAMESPACE: Final = (
    "boundary-qualification-fixture-metrics-v1"
)


class BoundaryQualificationFixtureTruth(StrictModel):
    """One frozen local truth point for an exact authoritative role source."""

    source_role_result_logical_key: NodeLogicalKey
    source_role_result_semantic_sha256: Sha256Digest
    role: BoundaryRefinementRole
    action_label: NonEmptyString
    camera_condition: BoundaryCameraCondition
    boundary_estimate_ns: int
    projection_version: Literal["boundary-qualification-fixture-truth-semantic-v1"] = (
        BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        source_role_result_logical_key: str,
        source_role_result_semantic_sha256: str,
        role: BoundaryRefinementRole,
        action_label: str,
        camera_condition: BoundaryCameraCondition,
        boundary_estimate_ns: int,
    ) -> Self:
        values: dict[str, object] = {
            "source_role_result_logical_key": source_role_result_logical_key,
            "source_role_result_semantic_sha256": source_role_result_semantic_sha256,
            "role": role,
            "action_label": action_label,
            "camera_condition": camera_condition,
            "boundary_estimate_ns": boundary_estimate_ns,
            "projection_version": BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = cls.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_KEY_NAMESPACE}:{'0' * 64}",
            source_role_result_logical_key=source_role_result_logical_key,
            source_role_result_semantic_sha256=source_role_result_semantic_sha256,
            role=role,
            action_label=action_label,
            camera_condition=camera_condition,
            boundary_estimate_ns=boundary_estimate_ns,
            projection_version=BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_PROJECTION_VERSION,
            production_eligible=False,
        )
        digest = semantic_sha256(boundary_qualification_fixture_truth_projection(draft))
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if (
            self.source_role_result_logical_key.rsplit(":", 1)[-1]
            != self.source_role_result_semantic_sha256
        ):
            raise ValueError("fixture truth role logical key is inconsistent")
        digest = semantic_sha256(boundary_qualification_fixture_truth_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("fixture truth semantic identity is inconsistent")
        if self.logical_key != f"{BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_KEY_NAMESPACE}:{digest}":
            raise ValueError("fixture truth logical key is inconsistent")
        return self


class BoundaryQualificationStratumMetrics(StrictModel):
    """Error and coverage outcomes for one role/class/condition stratum."""

    role: BoundaryRefinementRole
    action_label: NonEmptyString
    camera_condition: BoundaryCameraCondition
    sample_count: NonNegativeInt
    baseline_mae_ns: NonNegativeFloat
    candidate_mae_ns: NonNegativeFloat | None = None
    candidate_refined_count: NonNegativeInt
    candidate_indeterminate_count: NonNegativeInt
    baseline_interval_coverage: UnitInterval
    candidate_interval_coverage: UnitInterval | None = None

    @model_validator(mode="after")
    def validate_population(self) -> Self:
        if self.sample_count == 0:
            raise ValueError("boundary qualification stratum requires fixture samples")
        if self.candidate_refined_count + self.candidate_indeterminate_count != self.sample_count:
            raise ValueError("candidate outcomes must cover the stratum")
        has_candidates = self.candidate_refined_count > 0
        if has_candidates != (
            self.candidate_mae_ns is not None and self.candidate_interval_coverage is not None
        ):
            raise ValueError("candidate metrics must match refined candidate population")
        return self


class BoundaryQualificationFixtureMetrics(StrictModel):
    """Content-addressed local-fixture metrics, never representative evidence."""

    fixture_id: NonEmptyString
    report_logical_key: NodeLogicalKey
    report_semantic_sha256: Sha256Digest
    policy_semantic_sha256: Sha256Digest
    truth: tuple[BoundaryQualificationFixtureTruth, ...]
    strata: tuple[BoundaryQualificationStratumMetrics, ...]
    representative_measurement_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    projection_version: Literal["boundary-qualification-fixture-metrics-semantic-v1"] = (
        BOUNDARY_QUALIFICATION_FIXTURE_METRICS_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.report_logical_key.rsplit(":", 1)[-1] != self.report_semantic_sha256:
            raise ValueError("fixture metrics report logical key is inconsistent")
        expected_truth = tuple(sorted(self.truth, key=_truth_sort_key))
        if self.truth != expected_truth:
            raise ValueError("fixture truth must be unique and canonically ordered")
        if len({item.source_role_result_logical_key for item in self.truth}) != len(self.truth):
            raise ValueError("fixture truth may contain each role result once")
        expected_strata = tuple(sorted(self.strata, key=_stratum_sort_key))
        if self.strata != expected_strata:
            raise ValueError("fixture strata must be canonically ordered")
        digest = semantic_sha256(boundary_qualification_fixture_metrics_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("fixture boundary metrics semantic identity is inconsistent")
        if self.logical_key != f"{BOUNDARY_QUALIFICATION_FIXTURE_METRICS_KEY_NAMESPACE}:{digest}":
            raise ValueError("fixture boundary metrics logical key is inconsistent")
        return self


def boundary_qualification_fixture_truth_projection(
    truth: BoundaryQualificationFixtureTruth,
) -> dict[str, object]:
    return {
        "semantic_projection_version": truth.projection_version,
        "source_role_result_logical_key": truth.source_role_result_logical_key,
        "source_role_result_semantic_sha256": truth.source_role_result_semantic_sha256,
        "role": truth.role.value,
        "action_label": truth.action_label,
        "camera_condition": truth.camera_condition.value,
        "boundary_estimate_ns": str(truth.boundary_estimate_ns),
        "production_eligible": truth.production_eligible,
    }


def boundary_qualification_fixture_metrics_projection(
    metrics: BoundaryQualificationFixtureMetrics,
) -> dict[str, object]:
    return {
        "semantic_projection_version": metrics.projection_version,
        "fixture_id": metrics.fixture_id,
        "report_logical_key": metrics.report_logical_key,
        "report_semantic_sha256": metrics.report_semantic_sha256,
        "policy_semantic_sha256": metrics.policy_semantic_sha256,
        "truth": [boundary_qualification_fixture_truth_projection(item) for item in metrics.truth],
        "strata": [item.model_dump(mode="json") for item in metrics.strata],
        "representative_measurement_status": metrics.representative_measurement_status,
        "production_eligible": metrics.production_eligible,
        "qualification_scope": "local-fixture-only; representative-boundary-quality-is-p15",
    }


def build_boundary_qualification_fixture_metrics(
    *,
    fixture_id: str,
    report: BoundaryQualificationReport,
    truth: Sequence[BoundaryQualificationFixtureTruth],
) -> BoundaryQualificationFixtureMetrics:
    """Aggregate exact role comparisons by role, label, and camera condition."""

    checked = verify_boundary_qualification_report(report)
    ordered_truth = tuple(sorted(truth, key=_truth_sort_key))
    comparison_by_key = {
        role_input.source_role_result_logical_key: comparison
        for role_input, comparison in zip(checked.case.roles, checked.role_comparisons, strict=True)
    }
    if set(item.source_role_result_logical_key for item in ordered_truth) != set(comparison_by_key):
        raise ValueError("fixture truth must cover exactly the report role inputs")
    grouped: defaultdict[
        tuple[BoundaryRefinementRole, str, BoundaryCameraCondition],
        list[
            tuple[
                BoundaryQualificationFixtureTruth,
                BoundaryQualificationReduction,
                BoundaryQualificationReduction,
            ]
        ],
    ] = defaultdict(list)
    for item in ordered_truth:
        comparison = comparison_by_key[item.source_role_result_logical_key]
        grouped[(item.role, item.action_label, item.camera_condition)].append(
            (item, comparison.baseline, comparison.candidate)
        )
    strata = tuple(
        _build_stratum(key, rows)
        for key, rows in sorted(grouped.items(), key=lambda item: _stratum_key(item[0]))
    )
    values: dict[str, Any] = {
        "fixture_id": fixture_id,
        "report_logical_key": checked.logical_key,
        "report_semantic_sha256": checked.semantic_sha256,
        "policy_semantic_sha256": checked.policy.semantic_sha256,
        "truth": ordered_truth,
        "strata": strata,
        "representative_measurement_status": "NOT_MEASURED",
        "projection_version": BOUNDARY_QUALIFICATION_FIXTURE_METRICS_PROJECTION_VERSION,
        "production_eligible": False,
    }
    draft = BoundaryQualificationFixtureMetrics.model_construct(
        semantic_sha256="0" * 64,
        logical_key=f"{BOUNDARY_QUALIFICATION_FIXTURE_METRICS_KEY_NAMESPACE}:{'0' * 64}",
        **values,
    )
    digest = semantic_sha256(boundary_qualification_fixture_metrics_projection(draft))
    return BoundaryQualificationFixtureMetrics.model_validate(
        {
            **values,
            "semantic_sha256": digest,
            "logical_key": f"{BOUNDARY_QUALIFICATION_FIXTURE_METRICS_KEY_NAMESPACE}:{digest}",
        },
        strict=True,
    )


def _build_stratum(
    key: tuple[BoundaryRefinementRole, str, BoundaryCameraCondition],
    rows: Sequence[
        tuple[
            BoundaryQualificationFixtureTruth,
            BoundaryQualificationReduction,
            BoundaryQualificationReduction,
        ]
    ],
) -> BoundaryQualificationStratumMetrics:
    role, action_label, camera_condition = key
    baseline_errors = [
        _absolute_error(baseline, truth.boundary_estimate_ns) for truth, baseline, _ in rows
    ]
    candidate_rows = [
        (truth, candidate)
        for truth, _, candidate in rows
        if candidate.boundary_estimate_ns is not None and candidate.boundary_interval is not None
    ]
    return BoundaryQualificationStratumMetrics(
        role=role,
        action_label=action_label,
        camera_condition=camera_condition,
        sample_count=len(rows),
        baseline_mae_ns=sum(baseline_errors) / len(baseline_errors),
        candidate_mae_ns=(
            sum(
                _absolute_error(candidate, truth.boundary_estimate_ns)
                for truth, candidate in candidate_rows
            )
            / len(candidate_rows)
            if candidate_rows
            else None
        ),
        candidate_refined_count=len(candidate_rows),
        candidate_indeterminate_count=len(rows) - len(candidate_rows),
        baseline_interval_coverage=sum(
            _contains(baseline.boundary_interval, truth.boundary_estimate_ns)
            for truth, baseline, _ in rows
        )
        / len(rows),
        candidate_interval_coverage=(
            sum(
                _contains(candidate.boundary_interval, truth.boundary_estimate_ns)
                for truth, candidate in candidate_rows
            )
            / len(candidate_rows)
            if candidate_rows
            else None
        ),
    )


def _absolute_error(reduction: BoundaryQualificationReduction, truth_estimate_ns: int) -> float:
    if reduction.boundary_estimate_ns is None:
        raise ValueError("fixture baseline must reproduce an authoritative refined estimate")
    return float(abs(reduction.boundary_estimate_ns - truth_estimate_ns))


def _contains(interval: NanosecondInterval | None, value: int) -> bool:
    return interval is not None and interval.start_ns <= value < interval.end_ns


def _truth_sort_key(item: BoundaryQualificationFixtureTruth) -> tuple[str, str]:
    return item.source_role_result_logical_key, item.semantic_sha256


def _stratum_key(
    key: tuple[BoundaryRefinementRole, str, BoundaryCameraCondition],
) -> tuple[str, str, str]:
    return key[0].value, key[1], key[2].value


def _stratum_sort_key(item: BoundaryQualificationStratumMetrics) -> tuple[str, str, str]:
    return item.role.value, item.action_label, item.camera_condition.value


__all__ = [
    "BOUNDARY_QUALIFICATION_FIXTURE_METRICS_KEY_NAMESPACE",
    "BOUNDARY_QUALIFICATION_FIXTURE_METRICS_PROJECTION_VERSION",
    "BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_KEY_NAMESPACE",
    "BOUNDARY_QUALIFICATION_FIXTURE_TRUTH_PROJECTION_VERSION",
    "BoundaryQualificationFixtureMetrics",
    "BoundaryQualificationFixtureTruth",
    "BoundaryQualificationStratumMetrics",
    "boundary_qualification_fixture_metrics_projection",
    "boundary_qualification_fixture_truth_projection",
    "build_boundary_qualification_fixture_metrics",
]
