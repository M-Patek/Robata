from __future__ import annotations

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import semantic_sha256
from robata.contracts.phase_contract_decisions import (
    ContractSurface,
    OptimizationPhase,
    PhaseContractDecision,
    PhaseContractDecisionKind,
    PhaseContractDecisionRegister,
    default_phase_contract_decision_register,
    phase_contract_decision_register_projection,
)
from robata.runtime.measurement_truth import build_profile_evidence_register
from tests.unit.test_canonical_profile import _v3_report


def test_default_register_covers_every_downstream_blueprint_phase() -> None:
    register = default_phase_contract_decision_register()

    assert tuple(item.phase for item in register.decisions) == tuple(OptimizationPhase)
    assert all(item.dispatchable for item in register.decisions)
    assert all(
        ContractSurface.PUBLISHED_SCHEMA not in item.versioned_surfaces
        for item in register.decisions
    )


def test_register_digest_rejects_tampered_decisions() -> None:
    register = default_phase_contract_decision_register()
    tampered = register.model_dump(mode="python")
    tampered["decisions"] = (
        *register.decisions[:-1],
        register.decisions[-1].model_copy(update={"rationale": "tampered"}),
    )

    with pytest.raises(ValidationError, match="register_digest"):
        PhaseContractDecisionRegister.model_validate(tampered, strict=True)


def test_unresolved_decision_blocks_its_phase_but_remains_recordable() -> None:
    register = default_phase_contract_decision_register()
    unresolved = register.decisions[0].model_copy(
        update={"decision": PhaseContractDecisionKind.UNRESOLVED}
    )
    draft = register.model_copy(update={"decisions": (unresolved, *register.decisions[1:])})
    payload = draft.model_dump(mode="python")
    payload["register_digest"] = semantic_sha256(phase_contract_decision_register_projection(draft))
    pending = PhaseContractDecisionRegister.model_validate(payload, strict=True)

    with pytest.raises(ValueError, match="P1 has an unresolved"):
        pending.require_dispatchable(OptimizationPhase.P1)


def test_internal_versioned_decision_cannot_hide_a_published_wire_change() -> None:
    with pytest.raises(ValidationError, match="published changes require"):
        PhaseContractDecision(
            phase=OptimizationPhase.P1,
            decision=PhaseContractDecisionKind.INTERNAL_VERSIONED_CHANGE,
            preserved_surfaces=(ContractSurface.IDENTITY_FORMULA,),
            versioned_surfaces=(ContractSurface.WIRE_SHAPE,),
            rationale="invalid",
            guardrails=("invalid",),
        )


def test_scope_evidence_binds_all_dispatchable_phase_contract_decisions() -> None:
    evidence = build_profile_evidence_register(
        _v3_report(replayed=False),
        observed_at="2026-01-01T00:00:00Z",
    )

    assert tuple(item.phase for item in evidence.phase_contract_decisions.decisions) == tuple(
        OptimizationPhase
    )
    for phase in OptimizationPhase:
        assert evidence.phase_contract_decisions.require_dispatchable(phase).dispatchable
