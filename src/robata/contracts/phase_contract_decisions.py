"""P0 contract decisions for the optimization Blueprint.

This internal qualification register is not a published wire payload. It records the
contract branch selected before a downstream phase may change implementation details.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
PHASE_CONTRACT_DECISION_REGISTER_VERSION: Final[Literal["phase-contract-decision-register-v1"]] = (
    "phase-contract-decision-register-v1"
)


class OptimizationPhase(StrEnum):
    """Blueprint phases that require a P0 contract decision."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
    P6 = "P6"
    P7 = "P7"
    P8 = "P8"
    P9 = "P9"
    P10 = "P10"
    P11 = "P11"
    P12 = "P12"
    P13 = "P13"
    P14 = "P14"
    P15 = "P15"


class ContractSurface(StrEnum):
    """Contract surface evaluated before dispatch."""

    PUBLISHED_SCHEMA = "PUBLISHED_SCHEMA"
    WIRE_SHAPE = "WIRE_SHAPE"
    IDENTITY_FORMULA = "IDENTITY_FORMULA"
    DERIVED_ARTIFACT_IDENTITY = "DERIVED_ARTIFACT_IDENTITY"
    LOGICAL_KEY = "LOGICAL_KEY"
    IDEMPOTENCY_KEY = "IDEMPOTENCY_KEY"
    FENCE = "FENCE"
    SEMANTIC_PROJECTION = "SEMANTIC_PROJECTION"
    POLICY = "POLICY"


class PhaseContractDecisionKind(StrEnum):
    """Recorded outcome of a phase contract review."""

    NO_CONTRACT_CHANGE = "NO_CONTRACT_CHANGE"
    INTERNAL_VERSIONED_CHANGE = "INTERNAL_VERSIONED_CHANGE"
    PUBLISHED_CHANGE_REGISTERED = "PUBLISHED_CHANGE_REGISTERED"
    UNRESOLVED = "UNRESOLVED"


class PhaseContractDecision(StrictModel):
    """One immutable phase decision and its validity guardrails."""

    phase: OptimizationPhase
    decision: PhaseContractDecisionKind
    preserved_surfaces: tuple[ContractSurface, ...]
    versioned_surfaces: tuple[ContractSurface, ...] = ()
    rationale: NonEmptyString
    guardrails: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def validate_resolution(self) -> Self:
        if not self.preserved_surfaces:
            raise ValueError("a phase decision must name preserved contract surfaces")
        if tuple(sorted(set(self.preserved_surfaces))) != self.preserved_surfaces:
            raise ValueError("preserved_surfaces must be unique and ordered")
        if tuple(sorted(set(self.versioned_surfaces))) != self.versioned_surfaces:
            raise ValueError("versioned_surfaces must be unique and ordered")
        if set(self.preserved_surfaces) & set(self.versioned_surfaces):
            raise ValueError("a surface cannot be both preserved and versioned")
        if not self.guardrails:
            raise ValueError("a phase decision must state at least one guardrail")
        if (
            self.decision is PhaseContractDecisionKind.NO_CONTRACT_CHANGE
            and self.versioned_surfaces
        ):
            raise ValueError("no-contract-change decisions cannot version surfaces")
        if self.decision is PhaseContractDecisionKind.INTERNAL_VERSIONED_CHANGE:
            if not self.versioned_surfaces:
                raise ValueError("internal-versioned decisions must name versioned surfaces")
            if (
                ContractSurface.PUBLISHED_SCHEMA in self.versioned_surfaces
                or ContractSurface.WIRE_SHAPE in self.versioned_surfaces
            ):
                raise ValueError("published changes require a registered published decision")
        if self.decision is PhaseContractDecisionKind.PUBLISHED_CHANGE_REGISTERED and not (
            ContractSurface.PUBLISHED_SCHEMA in self.versioned_surfaces
            or ContractSurface.WIRE_SHAPE in self.versioned_surfaces
        ):
            raise ValueError("published decisions must version a schema or wire surface")
        return self

    @property
    def dispatchable(self) -> bool:
        return self.decision is not PhaseContractDecisionKind.UNRESOLVED


def phase_contract_decision_register_projection(
    register: PhaseContractDecisionRegister,
) -> dict[str, object]:
    """Return the stable content-addressed register projection."""

    if not isinstance(register, PhaseContractDecisionRegister):
        raise TypeError("register must be a PhaseContractDecisionRegister")
    return register.model_dump(mode="json", exclude={"register_digest"})


class PhaseContractDecisionRegister(StrictModel):
    """Complete ordered P0 register for P1 through P15."""

    schema_version: Literal["1.0"]
    register_version: Literal["phase-contract-decision-register-v1"]
    register_digest: Sha256Digest
    decisions: tuple[PhaseContractDecision, ...]

    @classmethod
    def create(cls, *, decisions: tuple[PhaseContractDecision, ...]) -> Self:
        draft = cls.model_construct(
            schema_version="1.0",
            register_version=PHASE_CONTRACT_DECISION_REGISTER_VERSION,
            register_digest="0" * 64,
            decisions=decisions,
        )
        return cls(
            schema_version="1.0",
            register_version=PHASE_CONTRACT_DECISION_REGISTER_VERSION,
            register_digest=semantic_sha256(phase_contract_decision_register_projection(draft)),
            decisions=decisions,
        )

    @model_validator(mode="after")
    def validate_register(self) -> Self:
        if tuple(item.phase for item in self.decisions) != tuple(OptimizationPhase):
            raise ValueError("phase decisions must cover P1 through P15 exactly once in order")
        if self.register_digest != semantic_sha256(
            phase_contract_decision_register_projection(self)
        ):
            raise ValueError("register_digest does not match the phase contract decisions")
        return self

    def decision_for(self, phase: OptimizationPhase) -> PhaseContractDecision:
        if not isinstance(phase, OptimizationPhase):
            raise TypeError("phase must be an OptimizationPhase")
        return self.decisions[tuple(OptimizationPhase).index(phase)]

    def require_dispatchable(self, phase: OptimizationPhase) -> PhaseContractDecision:
        decision = self.decision_for(phase)
        if not decision.dispatchable:
            raise ValueError(f"{phase.value} has an unresolved contract decision")
        return decision


_ALL_CORE_SURFACES: Final[tuple[ContractSurface, ...]] = (
    ContractSurface.DERIVED_ARTIFACT_IDENTITY,
    ContractSurface.FENCE,
    ContractSurface.IDEMPOTENCY_KEY,
    ContractSurface.IDENTITY_FORMULA,
    ContractSurface.LOGICAL_KEY,
    ContractSurface.PUBLISHED_SCHEMA,
    ContractSurface.SEMANTIC_PROJECTION,
    ContractSurface.WIRE_SHAPE,
)


def _decision(
    phase: OptimizationPhase,
    decision: PhaseContractDecisionKind,
    rationale: str,
    guardrails: tuple[str, ...],
    versioned_surfaces: tuple[ContractSurface, ...] = (),
) -> PhaseContractDecision:
    return PhaseContractDecision(
        phase=phase,
        decision=decision,
        preserved_surfaces=tuple(
            surface for surface in _ALL_CORE_SURFACES if surface not in versioned_surfaces
        ),
        versioned_surfaces=versioned_surfaces,
        rationale=rationale,
        guardrails=guardrails,
    )


def default_phase_contract_decision_register() -> PhaseContractDecisionRegister:
    """Return the current P0 decisions selected for the Blueprint."""

    no_change = PhaseContractDecisionKind.NO_CONTRACT_CHANGE
    internal = PhaseContractDecisionKind.INTERNAL_VERSIONED_CHANGE
    policy = (ContractSurface.POLICY,)
    decision_rows = (
        (
            OptimizationPhase.P1,
            no_change,
            "SQLite observation and batching preserve existing durable authority contracts.",
            (
                "Keep recovery, leases, fences, CAS, and same-database publication unchanged.",
                "Run provider and media work outside SQLite transactions.",
            ),
            (),
        ),
        (
            OptimizationPhase.P2,
            no_change,
            "Durability barriers expose the same exact artifact bytes and existing lineage.",
            ("Commit authority after verification, fsync, atomic exposure, and reconciliation.",),
            (),
        ),
        (
            OptimizationPhase.P3,
            no_change,
            "Completion profiling preserves the v3 ordered collection-root formula.",
            ("Keep all eleven count/root pairs and the complete canonical ordered digest list.",),
            (),
        ),
        (
            OptimizationPhase.P4,
            internal,
            "Backend provenance and cache policy stay internal; released formulas do not change.",
            (
                "Require derived-byte parity or bind changed bytes through existing identities.",
                "Record fallback before dependent publication.",
            ),
            policy,
        ),
        (
            OptimizationPhase.P5,
            internal,
            "JPEG remains opt-in and policy-bound; PNG is still the canonical default.",
            (
                "Bind encoder config and media bytes through existing identities.",
                "Changing a published schema or default needs a registered version decision.",
            ),
            (ContractSurface.DERIVED_ARTIFACT_IDENTITY, ContractSurface.POLICY),
        ),
        (
            OptimizationPhase.P6,
            no_change,
            "Provider batching changes transport grouping, not logical invocation ownership.",
            ("Keep one evidence chain per logical invocation.",),
            (),
        ),
        (
            OptimizationPhase.P7,
            no_change,
            "Concurrent execution preserves leases, fences, and terminal acceptance.",
            ("Enqueue only after durable claim and retain one deterministic EOS finalization.",),
            (),
        ),
        (
            OptimizationPhase.P8,
            no_change,
            "Backpressure is an internal timing controller and cannot change terminal semantics.",
            ("Persist controller state and limit actions to bounded timing controls.",),
            (),
        ),
        (
            OptimizationPhase.P9,
            internal,
            "Calibration lineage is inference-only; Product QA wire values remain uncalibrated.",
            (
                "A calibrated Product QA field needs a separately registered version.",
                "Missing or inapplicable calibration follows the declared raw-score path.",
            ),
            policy,
        ),
        (
            OptimizationPhase.P10,
            internal,
            "Adaptive decisions are durable and bound to existing accepted-evidence lineage.",
            ("Seal decisions before extra targets; never infer NO_EVENTS from QA.",),
            (
                ContractSurface.IDEMPOTENCY_KEY,
                ContractSurface.LOGICAL_KEY,
                ContractSurface.POLICY,
                ContractSurface.SEMANTIC_PROJECTION,
            ),
        ),
        (
            OptimizationPhase.P11,
            internal,
            "Association is an async derived report and does not alter released completion.",
            ("Cite accepted evidence; do not mutate released stages or event revisions.",),
            (
                ContractSurface.IDEMPOTENCY_KEY,
                ContractSurface.LOGICAL_KEY,
                ContractSurface.POLICY,
                ContractSurface.SEMANTIC_PROJECTION,
            ),
        ),
        (
            OptimizationPhase.P12,
            no_change,
            "Qualification compares candidates while the authoritative reducer remains unchanged.",
            ("Changed estimators require registered reducer, migration, and replay.",),
            (),
        ),
        (
            OptimizationPhase.P13,
            internal,
            "Review selection is an immutable internal pool decision and never blocks completion.",
            ("Bind pool, policy, terms, ranking, and budget in an append-only decision.",),
            (
                ContractSurface.IDEMPOTENCY_KEY,
                ContractSurface.LOGICAL_KEY,
                ContractSurface.POLICY,
                ContractSurface.SEMANTIC_PROJECTION,
            ),
        ),
        (
            OptimizationPhase.P14,
            internal,
            "Vector rows are async versioned projections over structured-authoritative candidates.",
            ("Preserve tenant isolation, revision linkage, and structured fallback.",),
            (
                ContractSurface.IDEMPOTENCY_KEY,
                ContractSurface.LOGICAL_KEY,
                ContractSurface.POLICY,
                ContractSurface.SEMANTIC_PROJECTION,
            ),
        ),
        (
            OptimizationPhase.P15,
            no_change,
            "The Pareto package cannot self-authorize a contract or production release.",
            ("Keep E0-E5 states explicit and leave E6 to independent review.",),
            (),
        ),
    )
    return PhaseContractDecisionRegister.create(
        decisions=tuple(_decision(*row) for row in decision_rows)
    )


__all__ = [
    "PHASE_CONTRACT_DECISION_REGISTER_VERSION",
    "ContractSurface",
    "OptimizationPhase",
    "PhaseContractDecision",
    "PhaseContractDecisionKind",
    "PhaseContractDecisionRegister",
    "default_phase_contract_decision_register",
    "phase_contract_decision_register_projection",
]
