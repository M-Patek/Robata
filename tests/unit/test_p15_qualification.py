"""Tests for the internal P15 evidence package and external-gate matrix."""

from __future__ import annotations

import json

import pytest

from robata.benchmark.p15_qualification import (
    P15ExternalGateEvidence,
    P15ExternalGateId,
    P15ExternalGateStatus,
    P15ParetoSelection,
    P15PhaseArtifactReference,
    P15QualificationPackage,
    P15TradeoffAxis,
    P15UnresolvedRisk,
)
from robata.contracts.measurement_truth import (
    EvidenceClass,
    MeasurementAxes,
    MeasurementEnvironment,
    MeasurementExecutionMode,
    MeasurementStatus,
    MeasurementWorkload,
    ScopeDigestInputs,
    ScopeEvidenceRegister,
    ScopeFingerprint,
)
from robata.contracts.phase_contract_decisions import OptimizationPhase


def _digest(value: int) -> str:
    return f"{value:064x}"


def _scope_evidence() -> ScopeEvidenceRegister:
    workload_digest = _digest(4)
    scope = ScopeFingerprint.create(
        inputs=ScopeDigestInputs(
            code_revision="p15-local-code-revision",
            code_digest=_digest(1),
            schema_catalog_digest=_digest(2),
            workload_digest=workload_digest,
            policy_digest=_digest(3),
            identity_formula_version="1.0",
            identity_projection_digest=_digest(5),
            seam_versions=("p15:qualification-package-v1",),
        )
    )
    return ScopeEvidenceRegister.create(
        scope=scope,
        evidence_class=EvidenceClass.LOCAL_CONFORMANCE,
        execution_mode=MeasurementExecutionMode.FRESH,
        workload=MeasurementWorkload(
            workload_fingerprint=workload_digest,
            recording_count=2,
            camera_count=6,
            recording_duration_ns=1_000,
        ),
        environment=MeasurementEnvironment(
            provider="local-fixture",
            provider_mode="LOCAL",
            hardware="local-cpu",
        ),
        axes=MeasurementAxes(recording_hours=1.0, camera_hours=6.0),
        observed_at="2026-01-01T00:00:00Z",
        measurement_status=MeasurementStatus.NOT_MEASURED,
    )


def _artifacts(scope: ScopeEvidenceRegister) -> tuple[P15PhaseArtifactReference, ...]:
    return tuple(
        P15PhaseArtifactReference(
            phase=phase,
            artifact_id=f"{phase.value.lower()}-evidence",
            artifact_uri=f"local://qualification/{phase.value.lower()}.json",
            artifact_sha256=_digest(100 + ordinal),
            scope_digest=scope.scope.scope_digest,
            evidence_class=EvidenceClass.LOCAL_CONFORMANCE,
            measurement_status=MeasurementStatus.NOT_MEASURED,
            summary=f"{phase.value} local conformance evidence",
        )
        for ordinal, phase in enumerate(
            (phase for phase in OptimizationPhase if phase is not OptimizationPhase.P15),
            start=1,
        )
    )


def _selection() -> P15ParetoSelection:
    return P15ParetoSelection(
        source_artifact_id="p8-evidence",
        candidate_operating_point_ids=("balanced", "high-recall"),
        selected_operating_point_id="balanced",
        tradeoff_axes=tuple(P15TradeoffAxis),
        preference_rationale="Retains the local Pareto frontier without collapsing its axes.",
        measurement_status=MeasurementStatus.NOT_MEASURED,
    )


def _gates() -> tuple[P15ExternalGateEvidence, ...]:
    return tuple(
        P15ExternalGateEvidence(
            gate_id=gate_id,
            status=(
                P15ExternalGateStatus.PENDING_INDEPENDENT_REVIEW
                if gate_id is P15ExternalGateId.E6
                else P15ExternalGateStatus.NOT_MEASURED
            ),
            unresolved_reason=(
                "Independent release review is pending."
                if gate_id is P15ExternalGateId.E6
                else "Representative external evidence has not been measured."
            ),
        )
        for gate_id in P15ExternalGateId
    )


def _risks() -> tuple[P15UnresolvedRisk, ...]:
    return tuple(
        P15UnresolvedRisk(
            risk_id=f"risk-{gate_id.value.lower()}",
            gate_id=gate_id,
            description=f"{gate_id.value} evidence remains unresolved.",
            required_follow_up=f"Run the declared {gate_id.value} qualification gate.",
        )
        for gate_id in P15ExternalGateId
    )


def _package() -> P15QualificationPackage:
    scope = _scope_evidence()
    return P15QualificationPackage.create(
        scope_evidence_register=scope,
        phase_artifacts=_artifacts(scope),
        pareto_selection=_selection(),
        external_gates=_gates(),
        unresolved_risks=_risks(),
    )


def test_package_replays_scope_phase_evidence_pareto_and_unresolved_gates() -> None:
    first = _package()
    second = _package()
    payload = first.as_dict()

    assert first == second
    assert first.production_eligible is False
    assert first.technical_requirements_satisfied is False
    assert first.external_gates[-1].status is P15ExternalGateStatus.PENDING_INDEPENDENT_REVIEW
    assert {item.phase for item in first.phase_artifacts} == {
        phase for phase in OptimizationPhase if phase is not OptimizationPhase.P15
    }
    assert P15QualificationPackage.model_validate_json(json.dumps(payload)) == first


def test_measured_failure_retains_its_artifact_and_cannot_be_relabelled_unmeasured() -> None:
    gate = P15ExternalGateEvidence(
        gate_id=P15ExternalGateId.E1,
        status=P15ExternalGateStatus.MEASURED_FAILED,
        unresolved_reason="Boundary coverage missed the frozen threshold.",
        supporting_artifact_uri="object://qualification/e1-boundary.json",
        supporting_artifact_sha256=_digest(500),
        threshold_failure="coverage below the registered minimum",
    )
    assert gate.threshold_failure == "coverage below the registered minimum"

    with pytest.raises(ValueError, match="supporting artifact"):
        P15ExternalGateEvidence(
            gate_id=P15ExternalGateId.E1,
            status=P15ExternalGateStatus.MEASURED_FAILED,
            unresolved_reason="Boundary coverage missed the frozen threshold.",
            threshold_failure="coverage below the registered minimum",
        )
    with pytest.raises(ValueError, match="threshold failure"):
        P15ExternalGateEvidence(
            gate_id=P15ExternalGateId.E1,
            status=P15ExternalGateStatus.NOT_MEASURED,
            unresolved_reason="Boundary coverage is unavailable.",
            threshold_failure="coverage below the registered minimum",
        )


def test_package_requires_every_phase_artifact_and_every_unresolved_gate_risk() -> None:
    scope = _scope_evidence()
    artifacts = _artifacts(scope)
    missing_p14 = (*artifacts[:-1], artifacts[-2].model_copy(update={"artifact_id": "p13-extra"}))
    with pytest.raises(ValueError, match="every P1-P14"):
        P15QualificationPackage.create(
            scope_evidence_register=scope,
            phase_artifacts=missing_p14,
            pareto_selection=_selection(),
            external_gates=_gates(),
            unresolved_risks=_risks(),
        )

    with pytest.raises(ValueError, match="risk-register"):
        P15QualificationPackage.create(
            scope_evidence_register=scope,
            phase_artifacts=_artifacts(scope),
            pareto_selection=_selection(),
            external_gates=_gates(),
            unresolved_risks=_risks()[1:],
        )
