"""Non-authoritative quality qualification for canonical boundary reductions.

The authoritative boundary reducer remains ``median-low-max-envelope-v1`` in
``boundary_refinement``.  This module only compares a quality-filtered
candidate against an exact copy of that reducer's inputs and outputs.  Its
content-addressed reports are deliberately not ActionEvent revisions and can
never become primary completion authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementOutcome,
    BoundaryRefinementRole,
)
from robata.qa_pipeline.boundary_quality import (
    BoundaryCameraCondition,
    BoundaryCameraQualityEvidence,
    BoundaryQualityApplicability,
)

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
MinimumCameraCount = Annotated[int, Field(strict=True, ge=1, le=6)]
QualityMillionths = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]

BOUNDARY_QUALIFICATION_POLICY_PROJECTION_VERSION: Final = (
    "boundary-qualification-policy-semantic-v1"
)
BOUNDARY_QUALIFICATION_REPORT_PROJECTION_VERSION: Final = (
    "boundary-qualification-report-semantic-v1"
)
BOUNDARY_QUALIFICATION_REPORT_LOGICAL_KEY_NAMESPACE: Final = "boundary-qualification-report-v1"
BOUNDARY_QUALIFICATION_REDUCER_VERSION: Final = "median-low-max-envelope-v1"


class BoundaryQualificationError(ValueError):
    """A qualification input cannot be compared deterministically."""


class BoundaryQualificationObservationOutcome(StrEnum):
    """The copied authoritative status of one camera slot."""

    OBSERVED = "OBSERVED"
    NO_BOUNDARY = "NO_BOUNDARY"
    INDETERMINATE = "INDETERMINATE"


class BoundaryQualificationCameraDisposition(StrEnum):
    """Why one camera was or was not included in a candidate reducer."""

    NOT_OBSERVED = "NOT_OBSERVED"
    SELECTED = "SELECTED"
    EXCLUDED_CONDITION = "EXCLUDED_CONDITION"
    EXCLUDED_QUALITY_THRESHOLD = "EXCLUDED_QUALITY_THRESHOLD"
    QUALITY_MISSING = "QUALITY_MISSING"
    QUALITY_NOT_APPLICABLE = "QUALITY_NOT_APPLICABLE"
    QUALITY_NOT_APPLIED = "QUALITY_NOT_APPLIED"


class BoundaryQualificationOutcome(StrEnum):
    """A non-authoritative comparison outcome, never an event disposition."""

    QUALITY_NOT_APPLIED = "QUALITY_NOT_APPLIED"
    CANDIDATE_REPORTED = "CANDIDATE_REPORTED"
    CANDIDATE_INDETERMINATE = "CANDIDATE_INDETERMINATE"
    BASELINE_RETAINED = "BASELINE_RETAINED"
    MIXED = "MIXED"


class BoundaryQualificationReasonCode(StrEnum):
    """Exact policy reason for a role-level comparison outcome."""

    QUALITY_NOT_APPLIED = "QUALITY_NOT_APPLIED"
    INSUFFICIENT_SELECTED_CAMERAS = "INSUFFICIENT_SELECTED_CAMERAS"
    NARROWING_REJECTED_NO_CALIBRATED_COVERAGE = "NARROWING_REJECTED_NO_CALIBRATED_COVERAGE"
    CANDIDATE_REPORTED = "CANDIDATE_REPORTED"


class BoundaryQualificationCameraObservation(StrictModel):
    """Exact copied per-camera evidence from one authoritative role result."""

    camera_id: CameraId
    source_camera_evidence_logical_key: NodeLogicalKey
    source_camera_evidence_semantic_sha256: Sha256Digest
    source_camera_evidence_exact_sha256: Sha256Digest
    outcome: BoundaryQualificationObservationOutcome
    observed_interval: NanosecondInterval | None = None
    boundary_estimate_ns: Nanoseconds | None = None
    uncertainty_ns: NonNegativeInt | None = None
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        _require_logical_key_digest(
            self.source_camera_evidence_logical_key,
            self.source_camera_evidence_semantic_sha256,
            "source camera evidence",
        )
        values = (
            self.observed_interval,
            self.boundary_estimate_ns,
            self.uncertainty_ns,
        )
        if self.outcome is not BoundaryQualificationObservationOutcome.OBSERVED:
            if any(value is not None for value in values):
                raise ValueError("unobserved qualification camera cannot carry geometry")
            return self
        if any(value is None for value in values):
            raise ValueError("observed qualification camera requires interval and reduction")
        assert self.observed_interval is not None
        assert self.boundary_estimate_ns is not None
        assert self.uncertainty_ns is not None
        expected_estimate = (
            self.observed_interval.start_ns + self.observed_interval.duration_ns // 2
        )
        expected_uncertainty = (self.observed_interval.duration_ns + 1) // 2
        if (
            self.boundary_estimate_ns != expected_estimate
            or self.uncertainty_ns != expected_uncertainty
        ):
            raise ValueError("camera reduction differs from authoritative interval semantics")
        return self


class BoundaryQualificationReduction(StrictModel):
    """One role reduction after retaining a specific camera subset."""

    selected_camera_ids: tuple[CameraId, ...]
    boundary_estimate_ns: Nanoseconds | None
    uncertainty_ns: NonNegativeInt | None
    boundary_interval: NanosecondInterval | None
    outcome: BoundaryRefinementOutcome
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_reduction(self) -> Self:
        if self.selected_camera_ids != _canonical_camera_ids(self.selected_camera_ids):
            raise ValueError("reduction camera IDs must be unique and canonically ordered")
        values = (
            self.boundary_estimate_ns,
            self.uncertainty_ns,
            self.boundary_interval,
        )
        if self.outcome is BoundaryRefinementOutcome.REFINED:
            if not self.selected_camera_ids or any(value is None for value in values):
                raise ValueError("refined qualification reduction lacks geometry")
        elif any(value is not None for value in values):
            raise ValueError("indeterminate qualification reduction cannot expose geometry")
        return self


class BoundaryQualificationRoleInput(StrictModel):
    """Complete role snapshot used to prove the baseline was not changed."""

    role: BoundaryRefinementRole
    source_role_result_logical_key: NodeLogicalKey
    source_role_result_semantic_sha256: Sha256Digest
    source_role_result_exact_sha256: Sha256Digest
    window_interval: NanosecondInterval
    minimum_observed_cameras: MinimumCameraCount
    camera_observations: tuple[BoundaryQualificationCameraObservation, ...]
    baseline_boundary_estimate_ns: Nanoseconds | None
    baseline_uncertainty_ns: NonNegativeInt | None
    baseline_boundary_interval: NanosecondInterval | None
    baseline_outcome: BoundaryRefinementOutcome
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_role_input(self) -> Self:
        _require_logical_key_digest(
            self.source_role_result_logical_key,
            self.source_role_result_semantic_sha256,
            "source role result",
        )
        if tuple(item.camera_id for item in self.camera_observations) != CAMERA_IDS:
            raise ValueError("qualification role input requires all six ordered cameras")
        for item in self.camera_observations:
            if item.observed_interval is not None and not _interval_inside(
                item.observed_interval, self.window_interval
            ):
                raise ValueError("qualification camera interval lies outside its role window")
        expected = _reduce_observations(
            observations=self.camera_observations,
            minimum_observed_cameras=self.minimum_observed_cameras,
            window_interval=self.window_interval,
        )
        actual = BoundaryQualificationReduction(
            selected_camera_ids=tuple(
                item.camera_id
                for item in self.camera_observations
                if item.outcome is BoundaryQualificationObservationOutcome.OBSERVED
            ),
            boundary_estimate_ns=self.baseline_boundary_estimate_ns,
            uncertainty_ns=self.baseline_uncertainty_ns,
            boundary_interval=self.baseline_boundary_interval,
            outcome=self.baseline_outcome,
            production_eligible=False,
        )
        if actual != expected:
            raise ValueError("qualification baseline differs from authoritative reducer")
        return self


class BoundaryQualificationPolicy(StrictModel):
    """Frozen candidate-only camera-exclusion policy."""

    version: SchemaVersion
    minimum_observed_cameras: MinimumCameraCount = 2
    minimum_quality_millionths: QualityMillionths = 0
    excluded_conditions: tuple[BoundaryCameraCondition, ...] = (
        BoundaryCameraCondition.INCOMPLETE,
        BoundaryCameraCondition.UNKNOWN,
        BoundaryCameraCondition.UNUSABLE,
    )
    calibrated_coverage_evidence_sha256: Sha256Digest | None = None
    reducer_version: Literal["median-low-max-envelope-v1"] = BOUNDARY_QUALIFICATION_REDUCER_VERSION
    projection_version: Literal["boundary-qualification-policy-semantic-v1"] = (
        BOUNDARY_QUALIFICATION_POLICY_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        version: str,
        minimum_observed_cameras: int = 2,
        minimum_quality_millionths: int = 0,
        excluded_conditions: Sequence[BoundaryCameraCondition] = (
            BoundaryCameraCondition.INCOMPLETE,
            BoundaryCameraCondition.UNKNOWN,
            BoundaryCameraCondition.UNUSABLE,
        ),
        calibrated_coverage_evidence_sha256: str | None = None,
    ) -> Self:
        values: dict[str, Any] = {
            "version": version,
            "minimum_observed_cameras": minimum_observed_cameras,
            "minimum_quality_millionths": minimum_quality_millionths,
            "excluded_conditions": tuple(
                sorted(set(excluded_conditions), key=lambda item: item.value)
            ),
            "calibrated_coverage_evidence_sha256": calibrated_coverage_evidence_sha256,
            "reducer_version": BOUNDARY_QUALIFICATION_REDUCER_VERSION,
            "projection_version": BOUNDARY_QUALIFICATION_POLICY_PROJECTION_VERSION,
            "production_eligible": False,
        }
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": semantic_sha256(
                    _boundary_qualification_policy_projection_values(values)
                ),
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.excluded_conditions != tuple(
            sorted(set(self.excluded_conditions), key=lambda item: item.value)
        ):
            raise ValueError("excluded quality conditions must be unique and ordered")
        if self.semantic_sha256 != semantic_sha256(boundary_qualification_policy_projection(self)):
            raise ValueError("boundary qualification policy semantic identity is inconsistent")
        return self


class BoundaryQualificationCase(StrictModel):
    """Exact completed-action context for one read-only qualification comparison."""

    source_boundary_result_logical_key: NodeLogicalKey
    source_boundary_result_semantic_sha256: Sha256Digest
    source_boundary_result_exact_sha256: Sha256Digest
    source_action_logical_key: NodeLogicalKey
    source_action_semantic_sha256: Sha256Digest
    mcap_id: OpaqueUuid
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    roles: tuple[BoundaryQualificationRoleInput, BoundaryQualificationRoleInput]
    quality_evidence: tuple[BoundaryCameraQualityEvidence, ...] = ()
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        _require_logical_key_digest(
            self.source_boundary_result_logical_key,
            self.source_boundary_result_semantic_sha256,
            "source boundary result",
        )
        _require_logical_key_digest(
            self.source_action_logical_key,
            self.source_action_semantic_sha256,
            "source action",
        )
        if tuple(item.role for item in self.roles) != (
            BoundaryRefinementRole.ONSET,
            BoundaryRefinementRole.OFFSET,
        ):
            raise ValueError("qualification case requires ordered ONSET and OFFSET role inputs")
        expected_quality = tuple(sorted(self.quality_evidence, key=_quality_sort_key))
        if self.quality_evidence != expected_quality:
            raise ValueError("qualification quality evidence must be canonically ordered")
        quality_camera_ids = tuple(item.camera_id for item in self.quality_evidence)
        if len(quality_camera_ids) != len(set(quality_camera_ids)):
            raise ValueError("qualification quality evidence must have unique cameras")
        for item in self.quality_evidence:
            actual_lineage = (
                item.mcap_id,
                item.recording_identity,
                item.source_content_sha256,
                item.camera_mapping_semantic_sha256,
                item.alignment_semantic_sha256,
            )
            expected_lineage = (
                self.mcap_id,
                self.recording_identity,
                self.source_content_sha256,
                self.camera_mapping_semantic_sha256,
                self.alignment_semantic_sha256,
            )
            if actual_lineage != expected_lineage:
                raise ValueError("quality evidence has foreign recording lineage")
        return self


class BoundaryQualificationCameraDecision(StrictModel):
    """Candidate disposition and exact QA citation for one camera slot."""

    camera_id: CameraId
    source_camera_evidence_logical_key: NodeLogicalKey
    quality_evidence_logical_key: NodeLogicalKey | None = None
    disposition: BoundaryQualificationCameraDisposition
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if (
            self.disposition
            in {
                BoundaryQualificationCameraDisposition.SELECTED,
                BoundaryQualificationCameraDisposition.EXCLUDED_CONDITION,
                BoundaryQualificationCameraDisposition.EXCLUDED_QUALITY_THRESHOLD,
                BoundaryQualificationCameraDisposition.QUALITY_NOT_APPLICABLE,
                BoundaryQualificationCameraDisposition.QUALITY_NOT_APPLIED,
            }
            and self.quality_evidence_logical_key is None
        ):
            raise ValueError("quality-based camera disposition requires QA evidence citation")
        if (
            self.disposition is BoundaryQualificationCameraDisposition.QUALITY_MISSING
            and self.quality_evidence_logical_key is not None
        ):
            raise ValueError("missing-quality camera disposition cannot cite QA evidence")
        return self


class BoundaryQualificationRoleComparison(StrictModel):
    """Baseline and candidate reductions for exactly one boundary role."""

    role: BoundaryRefinementRole
    baseline: BoundaryQualificationReduction
    candidate: BoundaryQualificationReduction
    camera_decisions: tuple[BoundaryQualificationCameraDecision, ...]
    outcome: BoundaryQualificationOutcome
    reason_code: BoundaryQualificationReasonCode
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if tuple(item.camera_id for item in self.camera_decisions) != CAMERA_IDS:
            raise ValueError("qualification comparison requires all six ordered camera decisions")
        if self.outcome is BoundaryQualificationOutcome.MIXED:
            raise ValueError("mixed outcome is only valid for an action-level report")
        if (
            self.outcome
            in {
                BoundaryQualificationOutcome.QUALITY_NOT_APPLIED,
                BoundaryQualificationOutcome.BASELINE_RETAINED,
            }
            and self.candidate != self.baseline
        ):
            raise ValueError("baseline-retaining qualification outcome must retain baseline values")
        if (
            self.outcome is BoundaryQualificationOutcome.CANDIDATE_INDETERMINATE
            and self.candidate.outcome is not BoundaryRefinementOutcome.INDETERMINATE
        ):
            raise ValueError("indeterminate qualification outcome requires indeterminate candidate")
        if (
            self.outcome is BoundaryQualificationOutcome.CANDIDATE_REPORTED
            and self.candidate.outcome is not BoundaryRefinementOutcome.REFINED
        ):
            raise ValueError("reported qualification candidate must be refined")
        expected_reasons = {
            BoundaryQualificationOutcome.QUALITY_NOT_APPLIED: (
                BoundaryQualificationReasonCode.QUALITY_NOT_APPLIED
            ),
            BoundaryQualificationOutcome.CANDIDATE_REPORTED: (
                BoundaryQualificationReasonCode.CANDIDATE_REPORTED
            ),
            BoundaryQualificationOutcome.CANDIDATE_INDETERMINATE: (
                BoundaryQualificationReasonCode.INSUFFICIENT_SELECTED_CAMERAS
            ),
            BoundaryQualificationOutcome.BASELINE_RETAINED: (
                BoundaryQualificationReasonCode.NARROWING_REJECTED_NO_CALIBRATED_COVERAGE
            ),
        }
        if self.reason_code is not expected_reasons[self.outcome]:
            raise ValueError("qualification reason code does not match comparison outcome")
        return self


class BoundaryQualificationReport(StrictModel):
    """Content-addressed P12 comparison evidence, never an authoritative result."""

    case: BoundaryQualificationCase
    policy: BoundaryQualificationPolicy
    role_comparisons: tuple[
        BoundaryQualificationRoleComparison,
        BoundaryQualificationRoleComparison,
    ]
    outcome: BoundaryQualificationOutcome
    projection_version: Literal["boundary-qualification-report-semantic-v1"] = (
        BOUNDARY_QUALIFICATION_REPORT_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        expected_roles = tuple(item.role for item in self.case.roles)
        actual_roles = tuple(item.role for item in self.role_comparisons)
        if actual_roles != expected_roles:
            raise ValueError("qualification report comparisons must retain ordered role separation")
        for source_role, comparison in zip(self.case.roles, self.role_comparisons, strict=True):
            expected = _baseline_reduction(source_role)
            if comparison.baseline != expected:
                raise ValueError("qualification report baseline differs from source role input")
            expected_decision_keys = tuple(
                item.source_camera_evidence_logical_key for item in source_role.camera_observations
            )
            actual_decision_keys = tuple(
                item.source_camera_evidence_logical_key for item in comparison.camera_decisions
            )
            if actual_decision_keys != expected_decision_keys:
                raise ValueError("qualification camera decisions do not retain raw evidence")
        if self.outcome is not _report_outcome(self.role_comparisons):
            raise ValueError("qualification report outcome does not match role comparisons")
        digest = semantic_sha256(boundary_qualification_report_projection(self))
        if (
            self.semantic_sha256 != digest
            or self.logical_key != f"{BOUNDARY_QUALIFICATION_REPORT_LOGICAL_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("boundary qualification report semantic identity is inconsistent")
        return self


class BoundaryQualificationEngine:
    """Compare quality-filtered candidates without changing authoritative boundaries."""

    def __init__(self, policy: BoundaryQualificationPolicy) -> None:
        self._policy = BoundaryQualificationPolicy.model_validate(
            policy.model_dump(mode="python"), strict=True
        )

    @property
    def policy(self) -> BoundaryQualificationPolicy:
        return self._policy

    def compare(self, case: BoundaryQualificationCase) -> BoundaryQualificationReport:
        """Produce one immutable candidate report from an exact case snapshot."""

        checked_case = BoundaryQualificationCase.model_validate(
            case.model_dump(mode="python"), strict=True
        )
        comparisons = tuple(
            self._compare_role(
                role_input=role_input,
                quality_by_camera={item.camera_id: item for item in checked_case.quality_evidence},
            )
            for role_input in checked_case.roles
        )
        role_comparisons = (comparisons[0], comparisons[1])
        values: dict[str, Any] = {
            "case": checked_case,
            "policy": self._policy,
            "role_comparisons": role_comparisons,
            "outcome": _report_outcome(role_comparisons),
            "projection_version": BOUNDARY_QUALIFICATION_REPORT_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = BoundaryQualificationReport.model_construct(
            **values,
            semantic_sha256="0" * 64,
            logical_key=(f"{BOUNDARY_QUALIFICATION_REPORT_LOGICAL_KEY_NAMESPACE}:{'0' * 64}"),
        )
        digest = semantic_sha256(boundary_qualification_report_projection(draft))
        return BoundaryQualificationReport.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{BOUNDARY_QUALIFICATION_REPORT_LOGICAL_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    def derive(self, case: BoundaryQualificationCase) -> BoundaryQualificationReport:
        """Alias for callers that use the report as derived qualification evidence."""

        return self.compare(case)

    def _compare_role(
        self,
        *,
        role_input: BoundaryQualificationRoleInput,
        quality_by_camera: dict[CameraId, BoundaryCameraQualityEvidence],
    ) -> BoundaryQualificationRoleComparison:
        baseline = _baseline_reduction(role_input)
        observed = tuple(
            item
            for item in role_input.camera_observations
            if item.outcome is BoundaryQualificationObservationOutcome.OBSERVED
        )
        quality_complete = bool(observed) and all(
            (quality := quality_by_camera.get(item.camera_id)) is not None
            and quality.applicability is BoundaryQualityApplicability.APPLICABLE
            for item in observed
        )
        if not quality_complete:
            unapplied_decisions = _quality_not_applied_decisions(
                role_input.camera_observations,
                quality_by_camera,
            )
            return BoundaryQualificationRoleComparison(
                role=role_input.role,
                baseline=baseline,
                candidate=baseline,
                camera_decisions=unapplied_decisions,
                outcome=BoundaryQualificationOutcome.QUALITY_NOT_APPLIED,
                reason_code=BoundaryQualificationReasonCode.QUALITY_NOT_APPLIED,
                production_eligible=False,
            )

        decisions: list[BoundaryQualificationCameraDecision] = []
        selected: list[BoundaryQualificationCameraObservation] = []
        for observation in role_input.camera_observations:
            quality = quality_by_camera.get(observation.camera_id)
            if observation.outcome is not BoundaryQualificationObservationOutcome.OBSERVED:
                decisions.append(
                    _camera_decision(
                        observation,
                        quality,
                        BoundaryQualificationCameraDisposition.NOT_OBSERVED,
                    )
                )
            elif quality is None:
                raise BoundaryQualificationError("quality completeness changed during comparison")
            elif quality.condition in self._policy.excluded_conditions:
                decisions.append(
                    _camera_decision(
                        observation,
                        quality,
                        BoundaryQualificationCameraDisposition.EXCLUDED_CONDITION,
                    )
                )
            elif quality.quality_millionths is None:
                raise BoundaryQualificationError("applicable quality lacks a quality value")
            elif quality.quality_millionths < self._policy.minimum_quality_millionths:
                decisions.append(
                    _camera_decision(
                        observation,
                        quality,
                        BoundaryQualificationCameraDisposition.EXCLUDED_QUALITY_THRESHOLD,
                    )
                )
            else:
                selected.append(observation)
                decisions.append(
                    _camera_decision(
                        observation,
                        quality,
                        BoundaryQualificationCameraDisposition.SELECTED,
                    )
                )
        candidate = _reduce_observations(
            observations=selected,
            minimum_observed_cameras=self._policy.minimum_observed_cameras,
            window_interval=role_input.window_interval,
        )
        if candidate.outcome is BoundaryRefinementOutcome.INDETERMINATE:
            outcome = BoundaryQualificationOutcome.CANDIDATE_INDETERMINATE
            reason_code = BoundaryQualificationReasonCode.INSUFFICIENT_SELECTED_CAMERAS
        elif _narrows_uncertainty(candidate, baseline) and (
            self._policy.calibrated_coverage_evidence_sha256 is None
        ):
            candidate = baseline
            outcome = BoundaryQualificationOutcome.BASELINE_RETAINED
            reason_code = BoundaryQualificationReasonCode.NARROWING_REJECTED_NO_CALIBRATED_COVERAGE
        else:
            outcome = BoundaryQualificationOutcome.CANDIDATE_REPORTED
            reason_code = BoundaryQualificationReasonCode.CANDIDATE_REPORTED
        return BoundaryQualificationRoleComparison(
            role=role_input.role,
            baseline=baseline,
            candidate=candidate,
            camera_decisions=tuple(decisions),
            outcome=outcome,
            reason_code=reason_code,
            production_eligible=False,
        )


def boundary_qualification_policy_projection(
    policy: BoundaryQualificationPolicy,
) -> dict[str, object]:
    """Return the explicit semantic projection for a candidate-only policy."""

    return _boundary_qualification_policy_projection_values(policy.model_dump(mode="python"))


def boundary_qualification_case_projection(
    case: BoundaryQualificationCase,
) -> dict[str, object]:
    """Return all copied geometry and QA lineage required for deterministic replay."""

    return {
        "source_boundary_result_logical_key": case.source_boundary_result_logical_key,
        "source_boundary_result_semantic_sha256": case.source_boundary_result_semantic_sha256,
        "source_boundary_result_exact_sha256": case.source_boundary_result_exact_sha256,
        "source_action_logical_key": case.source_action_logical_key,
        "source_action_semantic_sha256": case.source_action_semantic_sha256,
        "mcap_id": case.mcap_id,
        "recording_identity": case.recording_identity,
        "source_content_sha256": case.source_content_sha256,
        "camera_mapping_semantic_sha256": case.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": case.alignment_semantic_sha256,
        "roles": [item.model_dump(mode="json") for item in case.roles],
        "quality_evidence": [item.model_dump(mode="json") for item in case.quality_evidence],
        "production_eligible": case.production_eligible,
        "identity_scope": "read-only-boundary-qualification-case",
    }


def boundary_qualification_report_projection(
    report: BoundaryQualificationReport,
) -> dict[str, object]:
    """Return report semantics without its derived report identity."""

    return {
        "semantic_projection_version": report.projection_version,
        "case": boundary_qualification_case_projection(report.case),
        "policy": boundary_qualification_policy_projection(report.policy),
        "role_comparisons": [item.model_dump(mode="json") for item in report.role_comparisons],
        "outcome": report.outcome.value,
        "production_eligible": report.production_eligible,
        "identity_scope": "candidate-qualification-not-authoritative-boundary-or-event-revision",
    }


def verify_boundary_qualification_report(
    report: BoundaryQualificationReport,
) -> BoundaryQualificationReport:
    """Reject a syntactically valid report that does not replay from exact inputs."""

    checked = BoundaryQualificationReport.model_validate(
        report.model_dump(mode="python"), strict=True
    )
    expected = BoundaryQualificationEngine(checked.policy).compare(checked.case)
    if expected.model_dump(mode="json") != checked.model_dump(mode="json"):
        raise ValueError("boundary qualification report does not match deterministic policy replay")
    return checked


def _boundary_qualification_policy_projection_values(
    values: dict[str, Any],
) -> dict[str, object]:
    conditions = tuple(values["excluded_conditions"])
    return {
        "semantic_projection_version": values["projection_version"],
        "version": values["version"],
        "minimum_observed_cameras": values["minimum_observed_cameras"],
        "minimum_quality_millionths": values["minimum_quality_millionths"],
        "excluded_conditions": [
            item.value if isinstance(item, BoundaryCameraCondition) else item for item in conditions
        ],
        "calibrated_coverage_evidence_sha256": values["calibrated_coverage_evidence_sha256"],
        "reducer_version": values["reducer_version"],
        "production_eligible": values["production_eligible"],
        "authority": "non-authoritative-qualification-only",
    }


def _baseline_reduction(
    role_input: BoundaryQualificationRoleInput,
) -> BoundaryQualificationReduction:
    return _reduce_observations(
        observations=role_input.camera_observations,
        minimum_observed_cameras=role_input.minimum_observed_cameras,
        window_interval=role_input.window_interval,
    )


def _reduce_observations(
    *,
    observations: Sequence[BoundaryQualificationCameraObservation],
    minimum_observed_cameras: int,
    window_interval: NanosecondInterval,
) -> BoundaryQualificationReduction:
    slots = tuple(
        item
        for item in observations
        if item.outcome is BoundaryQualificationObservationOutcome.OBSERVED
    )
    selected_camera_ids = tuple(item.camera_id for item in slots)
    if selected_camera_ids != _canonical_camera_ids(selected_camera_ids):
        raise BoundaryQualificationError("candidate reducer received unordered camera evidence")
    if len(slots) < minimum_observed_cameras:
        return BoundaryQualificationReduction(
            selected_camera_ids=selected_camera_ids,
            boundary_estimate_ns=None,
            uncertainty_ns=None,
            boundary_interval=None,
            outcome=BoundaryRefinementOutcome.INDETERMINATE,
            production_eligible=False,
        )
    centers: list[int] = []
    intervals: list[NanosecondInterval] = []
    for slot in slots:
        if slot.observed_interval is None or slot.boundary_estimate_ns is None:
            raise BoundaryQualificationError("observed camera lacks interval reduction")
        centers.append(slot.boundary_estimate_ns)
        intervals.append(slot.observed_interval)
    estimate = _median_low_int(centers)
    uncertainty = max(
        abs(center - estimate) + (interval.duration_ns + 1) // 2
        for center, interval in zip(centers, intervals, strict=True)
    )
    interval = NanosecondInterval(
        start_ns=max(window_interval.start_ns, estimate - uncertainty),
        end_ns=min(window_interval.end_ns, estimate + uncertainty + 1),
    )
    return BoundaryQualificationReduction(
        selected_camera_ids=selected_camera_ids,
        boundary_estimate_ns=estimate,
        uncertainty_ns=uncertainty,
        boundary_interval=interval,
        outcome=BoundaryRefinementOutcome.REFINED,
        production_eligible=False,
    )


def _quality_not_applied_decisions(
    observations: Sequence[BoundaryQualificationCameraObservation],
    quality_by_camera: dict[CameraId, BoundaryCameraQualityEvidence],
) -> tuple[BoundaryQualificationCameraDecision, ...]:
    decisions: list[BoundaryQualificationCameraDecision] = []
    for observation in observations:
        quality = quality_by_camera.get(observation.camera_id)
        if observation.outcome is not BoundaryQualificationObservationOutcome.OBSERVED:
            disposition = BoundaryQualificationCameraDisposition.NOT_OBSERVED
        elif quality is None:
            disposition = BoundaryQualificationCameraDisposition.QUALITY_MISSING
        elif quality.applicability is not BoundaryQualityApplicability.APPLICABLE:
            disposition = BoundaryQualificationCameraDisposition.QUALITY_NOT_APPLICABLE
        else:
            disposition = BoundaryQualificationCameraDisposition.QUALITY_NOT_APPLIED
        decisions.append(_camera_decision(observation, quality, disposition))
    return tuple(decisions)


def _camera_decision(
    observation: BoundaryQualificationCameraObservation,
    quality: BoundaryCameraQualityEvidence | None,
    disposition: BoundaryQualificationCameraDisposition,
) -> BoundaryQualificationCameraDecision:
    return BoundaryQualificationCameraDecision(
        camera_id=observation.camera_id,
        source_camera_evidence_logical_key=observation.source_camera_evidence_logical_key,
        quality_evidence_logical_key=(quality.logical_key if quality is not None else None),
        disposition=disposition,
        production_eligible=False,
    )


def _narrows_uncertainty(
    candidate: BoundaryQualificationReduction,
    baseline: BoundaryQualificationReduction,
) -> bool:
    return (
        candidate.outcome is BoundaryRefinementOutcome.REFINED
        and baseline.outcome is BoundaryRefinementOutcome.REFINED
        and candidate.uncertainty_ns is not None
        and baseline.uncertainty_ns is not None
        and candidate.uncertainty_ns < baseline.uncertainty_ns
    )


def _report_outcome(
    comparisons: Sequence[BoundaryQualificationRoleComparison],
) -> BoundaryQualificationOutcome:
    outcomes = tuple(item.outcome for item in comparisons)
    if not outcomes:
        raise ValueError("qualification report requires role comparisons")
    if len(set(outcomes)) == 1:
        return outcomes[0]
    return BoundaryQualificationOutcome.MIXED


def _canonical_camera_ids(camera_ids: Sequence[CameraId]) -> tuple[CameraId, ...]:
    return tuple(sorted(set(camera_ids), key=CAMERA_IDS.index))


def _quality_sort_key(
    evidence: BoundaryCameraQualityEvidence,
) -> tuple[int, str]:
    return CAMERA_IDS.index(evidence.camera_id), evidence.logical_key


def _require_logical_key_digest(logical_key: str, digest: str, subject: str) -> None:
    if logical_key.rsplit(":", 1)[-1] != digest:
        raise ValueError(f"{subject} logical key must end with its semantic SHA-256")


def _interval_inside(inner: NanosecondInterval, outer: NanosecondInterval) -> bool:
    return inner.start_ns >= outer.start_ns and inner.end_ns <= outer.end_ns


def _median_low_int(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


__all__ = [
    "BOUNDARY_QUALIFICATION_POLICY_PROJECTION_VERSION",
    "BOUNDARY_QUALIFICATION_REDUCER_VERSION",
    "BOUNDARY_QUALIFICATION_REPORT_LOGICAL_KEY_NAMESPACE",
    "BOUNDARY_QUALIFICATION_REPORT_PROJECTION_VERSION",
    "BoundaryQualificationCameraDecision",
    "BoundaryQualificationCameraDisposition",
    "BoundaryQualificationCameraObservation",
    "BoundaryQualificationCase",
    "BoundaryQualificationEngine",
    "BoundaryQualificationError",
    "BoundaryQualificationObservationOutcome",
    "BoundaryQualificationOutcome",
    "BoundaryQualificationPolicy",
    "BoundaryQualificationReasonCode",
    "BoundaryQualificationReduction",
    "BoundaryQualificationReport",
    "BoundaryQualificationRoleComparison",
    "BoundaryQualificationRoleInput",
    "boundary_qualification_case_projection",
    "boundary_qualification_policy_projection",
    "boundary_qualification_report_projection",
    "verify_boundary_qualification_report",
]
