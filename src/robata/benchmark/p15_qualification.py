"""Content-addressed P15 Pareto and external-gate qualification package.

The package is deliberately an internal qualification artifact, not a released
wire payload.  It binds P0's frozen scope to P1-P14 evidence references and
retains every unresolved external gate.  It can describe a measured failure,
but never turns any technical result into production authorization.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.measurement_truth import (
    EvidenceClass,
    MeasurementStatus,
    ScopeEvidenceRegister,
)
from robata.contracts.phase_contract_decisions import (
    OptimizationPhase,
    PhaseContractDecisionKind,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
P15_QUALIFICATION_PACKAGE_VERSION: Final = "p15-qualification-package-v1"
_EXPECTED_GATE_IDS: Final = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
_REQUIRED_PHASES: Final = tuple(
    phase for phase in OptimizationPhase if phase is not OptimizationPhase.P15
)


class P15ExternalGateId(StrEnum):
    """Ordered external gates declared by the P15 Blueprint."""

    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    E6 = "E6"


class P15ExternalGateStatus(StrEnum):
    """External-gate truth without conflating absence and a measured failure."""

    NOT_MEASURED = "NOT_MEASURED"
    MEASURED_PENDING_INDEPENDENT_REVIEW = "MEASURED_PENDING_INDEPENDENT_REVIEW"
    MEASURED_FAILED = "MEASURED_FAILED"
    PENDING_INDEPENDENT_REVIEW = "PENDING_INDEPENDENT_REVIEW"


class P15TradeoffAxis(StrEnum):
    """Axes retained separately when selecting an operating point."""

    THROUGHPUT = "THROUGHPUT"
    QUALITY = "QUALITY"
    RELIABILITY = "RELIABILITY"
    LATENCY = "LATENCY"
    DEADLINE = "DEADLINE"
    RESOURCE = "RESOURCE"
    COST = "COST"


class P15PhaseArtifactReference(StrictModel):
    """Exact locally available evidence from one preceding Blueprint phase."""

    phase: OptimizationPhase
    artifact_id: NonEmptyString
    artifact_uri: NonEmptyString
    artifact_sha256: Sha256Digest
    scope_digest: Sha256Digest
    evidence_class: EvidenceClass
    measurement_status: MeasurementStatus
    summary: NonEmptyString
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.phase is OptimizationPhase.P15:
            raise ValueError("P15 evidence package may only reference preceding P1-P14 artifacts")
        if self.evidence_class is EvidenceClass.PRODUCTION_QUALIFIED:
            raise ValueError("phase evidence cannot self-label as production qualified")
        return self


class P15ParetoSelection(StrictModel):
    """Chosen operating point with explicit multi-axis rationale, never a scalar score."""

    source_artifact_id: NonEmptyString
    candidate_operating_point_ids: tuple[NonEmptyString, ...] = Field(min_length=2)
    selected_operating_point_id: NonEmptyString
    tradeoff_axes: tuple[P15TradeoffAxis, ...]
    preference_rationale: NonEmptyString
    measurement_status: MeasurementStatus
    production_eligible: Literal[False] = False

    @field_validator("candidate_operating_point_ids")
    @classmethod
    def validate_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("Pareto candidate operating point IDs must be unique and ordered")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.selected_operating_point_id not in self.candidate_operating_point_ids:
            raise ValueError("selected operating point must belong to the Pareto candidate set")
        if self.tradeoff_axes != tuple(P15TradeoffAxis):
            raise ValueError("Pareto selection must retain every tradeoff axis in canonical order")
        return self


class P15ExternalGateEvidence(StrictModel):
    """One P15 gate state with a proof artifact or an explicit unresolved reason."""

    gate_id: P15ExternalGateId
    status: P15ExternalGateStatus
    unresolved_reason: NonEmptyString
    supporting_artifact_uri: NonEmptyString | None = None
    supporting_artifact_sha256: Sha256Digest | None = None
    threshold_failure: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        has_uri = self.supporting_artifact_uri is not None
        has_digest = self.supporting_artifact_sha256 is not None
        if has_uri != has_digest:
            raise ValueError("gate artifact URI and digest must be supplied together")
        measured = {
            P15ExternalGateStatus.MEASURED_PENDING_INDEPENDENT_REVIEW,
            P15ExternalGateStatus.MEASURED_FAILED,
        }
        if self.status in measured and not has_uri:
            raise ValueError("measured gate states require a supporting artifact")
        if self.status is P15ExternalGateStatus.NOT_MEASURED and has_uri:
            raise ValueError("unmeasured gates cannot claim a supporting artifact")
        if self.status is P15ExternalGateStatus.MEASURED_FAILED:
            if self.threshold_failure is None:
                raise ValueError("measured failed gates require a threshold failure")
        elif self.threshold_failure is not None:
            raise ValueError("only measured failed gates may retain a threshold failure")
        if self.gate_id is P15ExternalGateId.E6:
            if self.status is not P15ExternalGateStatus.PENDING_INDEPENDENT_REVIEW:
                raise ValueError("E6 must remain pending independent review")
            if has_uri or self.threshold_failure is not None:
                raise ValueError("E6 cannot claim a technical artifact or threshold result")
        elif self.status is P15ExternalGateStatus.PENDING_INDEPENDENT_REVIEW:
            raise ValueError("only E6 may be pending independent review")
        return self


class P15UnresolvedRisk(StrictModel):
    """One explicit unresolved item owned by a non-qualified external gate."""

    risk_id: NonEmptyString
    gate_id: P15ExternalGateId
    description: NonEmptyString
    required_follow_up: NonEmptyString


class P15QualificationPackage(StrictModel):
    """P15 aggregate retaining scope, Pareto choice, evidence, gates, and risks."""

    package_version: Literal["p15-qualification-package-v1"] = P15_QUALIFICATION_PACKAGE_VERSION
    package_sha256: Sha256Digest
    scope_evidence_register: ScopeEvidenceRegister
    phase_artifacts: tuple[P15PhaseArtifactReference, ...] = Field(min_length=14)
    pareto_selection: P15ParetoSelection
    external_gates: tuple[P15ExternalGateEvidence, ...] = Field(min_length=7)
    unresolved_risks: tuple[P15UnresolvedRisk, ...] = Field(min_length=1)
    technical_requirements_satisfied: bool
    qualification_status: Literal["PENDING_INDEPENDENT_REVIEW"] = "PENDING_INDEPENDENT_REVIEW"
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        scope_evidence_register: ScopeEvidenceRegister,
        phase_artifacts: tuple[P15PhaseArtifactReference, ...],
        pareto_selection: P15ParetoSelection,
        external_gates: tuple[P15ExternalGateEvidence, ...],
        unresolved_risks: tuple[P15UnresolvedRisk, ...],
    ) -> Self:
        """Create a content-addressed P15 package without granting release authority."""

        technical_requirements_satisfied = all(
            gate.status is P15ExternalGateStatus.MEASURED_PENDING_INDEPENDENT_REVIEW
            for gate in external_gates
            if gate.gate_id is not P15ExternalGateId.E6
        )
        values: dict[str, object] = {
            "scope_evidence_register": scope_evidence_register,
            "phase_artifacts": phase_artifacts,
            "pareto_selection": pareto_selection,
            "external_gates": external_gates,
            "unresolved_risks": unresolved_risks,
            "technical_requirements_satisfied": technical_requirements_satisfied,
            "qualification_status": "PENDING_INDEPENDENT_REVIEW",
            "production_eligible": False,
        }
        draft = cls.model_construct(
            package_version=P15_QUALIFICATION_PACKAGE_VERSION,
            package_sha256="0" * 64,
            scope_evidence_register=scope_evidence_register,
            phase_artifacts=phase_artifacts,
            pareto_selection=pareto_selection,
            external_gates=external_gates,
            unresolved_risks=unresolved_risks,
            technical_requirements_satisfied=technical_requirements_satisfied,
            qualification_status="PENDING_INDEPENDENT_REVIEW",
            production_eligible=False,
        )
        package_sha256 = semantic_sha256(p15_qualification_package_projection(draft))
        return cls.model_validate(
            {
                "package_version": P15_QUALIFICATION_PACKAGE_VERSION,
                "package_sha256": package_sha256,
                **values,
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_package(self) -> Self:
        ScopeEvidenceRegister.model_validate(
            self.scope_evidence_register.model_dump(mode="python"), strict=True
        )
        p15_decision = self.scope_evidence_register.phase_contract_decisions.decision_for(
            OptimizationPhase.P15
        )
        if p15_decision.decision is not PhaseContractDecisionKind.NO_CONTRACT_CHANGE:
            raise ValueError("P15 package requires the frozen no-contract-change decision")
        artifact_keys = tuple((item.phase, item.artifact_id) for item in self.phase_artifacts)
        phase_order = tuple(OptimizationPhase)
        if artifact_keys != tuple(
            sorted(artifact_keys, key=lambda item: (phase_order.index(item[0]), item[1]))
        ):
            raise ValueError("phase artifacts must be canonically ordered")
        artifact_ids = tuple(item.artifact_id for item in self.phase_artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("phase artifact IDs must be unique")
        artifact_phases = {item.phase for item in self.phase_artifacts}
        if artifact_phases != set(_REQUIRED_PHASES):
            raise ValueError("P15 package requires at least one artifact for every P1-P14 phase")
        scope_digest = self.scope_evidence_register.scope.scope_digest
        if any(item.scope_digest != scope_digest for item in self.phase_artifacts):
            raise ValueError("phase artifacts must bind the frozen P0 scope digest")
        if self.pareto_selection.source_artifact_id not in set(artifact_ids):
            raise ValueError("Pareto selection must cite one retained phase artifact")
        gate_ids = tuple(gate.gate_id.value for gate in self.external_gates)
        if gate_ids != _EXPECTED_GATE_IDS:
            raise ValueError("P15 external gates must retain E0 through E6 in order")
        unresolved_gate_ids = {
            gate.gate_id
            for gate in self.external_gates
            if gate.status is not P15ExternalGateStatus.MEASURED_PENDING_INDEPENDENT_REVIEW
        }
        risk_keys = tuple((risk.gate_id.value, risk.risk_id) for risk in self.unresolved_risks)
        if risk_keys != tuple(sorted(risk_keys)) or len(risk_keys) != len(set(risk_keys)):
            raise ValueError("unresolved risks must have unique canonical gate and risk IDs")
        if not unresolved_gate_ids.issubset({risk.gate_id for risk in self.unresolved_risks}):
            raise ValueError("every unresolved external gate requires a risk-register entry")
        expected_technical_requirements_satisfied = not unresolved_gate_ids - {P15ExternalGateId.E6}
        if self.technical_requirements_satisfied != expected_technical_requirements_satisfied:
            raise ValueError("technical requirements state does not match external gate evidence")
        expected_sha256 = semantic_sha256(p15_qualification_package_projection(self))
        if self.package_sha256 != expected_sha256:
            raise ValueError("package_sha256 does not match the P15 qualification package")
        return self

    def as_dict(self) -> dict[str, object]:
        """Return the complete JSON-ready evidence package."""

        return self.model_dump(mode="json")


def p15_qualification_package_projection(
    package: P15QualificationPackage,
) -> dict[str, object]:
    """Return the content-addressed P15 package preimage."""

    return {
        "package_version": package.package_version,
        "scope_evidence_register_digest": package.scope_evidence_register.register_digest,
        "scope_digest": package.scope_evidence_register.scope.scope_digest,
        "phase_artifacts": [item.model_dump(mode="json") for item in package.phase_artifacts],
        "pareto_selection": package.pareto_selection.model_dump(mode="json"),
        "external_gates": [item.model_dump(mode="json") for item in package.external_gates],
        "unresolved_risks": [item.model_dump(mode="json") for item in package.unresolved_risks],
        "technical_requirements_satisfied": package.technical_requirements_satisfied,
        "qualification_status": package.qualification_status,
        "production_eligible": package.production_eligible,
    }


__all__ = [
    "P15_QUALIFICATION_PACKAGE_VERSION",
    "P15ExternalGateEvidence",
    "P15ExternalGateId",
    "P15ExternalGateStatus",
    "P15ParetoSelection",
    "P15PhaseArtifactReference",
    "P15QualificationPackage",
    "P15TradeoffAxis",
    "P15UnresolvedRisk",
    "p15_qualification_package_projection",
]
