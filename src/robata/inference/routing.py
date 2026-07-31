"""Plane-separated model routing for production and experiment traffic.

The production plane can produce exactly one authoritative model result.  The
experiment plane can run paired or shadow work, but its route decisions are
observational and therefore cannot select a production result.  Dispatch and
evidence persistence remain the responsibility of the inference orchestrator.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1, le=256)]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelRoutingError(ValueError):
    """Raised when route configuration or route lookup is invalid."""


class RoutePlane(StrEnum):
    """The authority plane which owns a model route."""

    PRODUCTION = "PRODUCTION"
    EXPERIMENT = "EXPERIMENT"


class RouteMode(StrEnum):
    """The dispatch shape selected by a route."""

    PRIMARY = "PRIMARY"
    SHADOW = "SHADOW"
    PAIRED = "PAIRED"
    CANARY = "CANARY"


class ModelRouteRole(StrEnum):
    """The stable role of a deployment within a route."""

    CONTROL = "CONTROL"
    CANDIDATE = "CANDIDATE"


class DispatchDisposition(StrEnum):
    """Whether a dispatch may enter the authoritative selection path."""

    AUTHORITATIVE = "AUTHORITATIVE"
    OBSERVATION = "OBSERVATION"


class ExperimentInputRepresentation(StrEnum):
    """How the two model routes represent the same source input."""

    IDENTICAL_FRAME_RENDERING = "IDENTICAL_FRAME_RENDERING"
    MODEL_SPECIFIC_RENDERING = "MODEL_SPECIFIC_RENDERING"


class ExperimentIsolationProfile(StrEnum):
    """Resource placement used when interpreting a paired experiment."""

    INDEPENDENT_EQUAL_HARDWARE = "INDEPENDENT_EQUAL_HARDWARE"
    COLOCATED_SHARED_HARDWARE = "COLOCATED_SHARED_HARDWARE"


class ModelDeployment(StrictModel):
    """Pinned, non-secret model deployment facts used by a route.

    ``endpoint_config_digest`` identifies a separately managed endpoint
    configuration without carrying a URL or credential through experiment
    records.  A deployment is intentionally distinct from a provider: two
    endpoints serving the same provider or model still need separate capacity
    and provenance identities.
    """

    schema_version: Literal["1.0"] = "1.0"
    deployment_id: NonEmptyString
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    adapter_version: SchemaVersion
    capability_snapshot_id: OpaqueUuid
    capability_snapshot_digest: Sha256Digest
    endpoint_config_digest: Sha256Digest
    max_concurrent_requests: PositiveInt


class ProductionRouteAuthorization(StrictModel):
    """References external qualification and release evidence.

    This is deliberately a reference, not a local claim that a route is
    qualified. The release process must validate the referenced evidence
    before a route is deployed in a production composition.
    """

    qualification_report_ref: NonEmptyString
    qualification_report_sha256: Sha256Digest
    release_decision_ref: NonEmptyString
    release_decision_sha256: Sha256Digest


class ProductionRouteAuthorizationVerifier(Protocol):
    """External port that verifies the exact evidence for a serving canary."""

    def verify(
        self,
        *,
        authorization: ProductionRouteAuthorization,
        control: ModelDeployment,
        candidate: ModelDeployment,
    ) -> bool:
        """Return ``True`` only when the referenced evidence authorizes this pair."""


class ExperimentContract(StrictModel):
    """Frozen identity and fairness conditions for one two-model experiment.

    The contract digest is the required logical dependency for every later
    experiment dispatch. It prevents independent experiments over the same
    media from reusing an inference identity or terminal accidentally.
    """

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: NonEmptyString
    contract_version: SchemaVersion
    workload_manifest_sha256: Sha256Digest
    arrival_schedule_sha256: Sha256Digest
    comparison_config_sha256: Sha256Digest
    input_representation: ExperimentInputRepresentation
    isolation_profile: ExperimentIsolationProfile
    control: ModelDeployment
    candidate: ModelDeployment
    selection_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_deployments(self) -> Self:
        _validate_distinct_deployments(control=self.control, candidate=self.candidate)
        return self

    @property
    def contract_digest(self) -> Sha256Digest:
        return _route_configuration_digest(self)


class RouteDispatch(StrictModel):
    """One model dispatch selected by a route decision."""

    deployment_id: NonEmptyString
    role: ModelRouteRole
    disposition: DispatchDisposition


class ModelRouteDecision(StrictModel):
    """A deterministic route decision bound to one immutable input identity."""

    schema_version: Literal["1.0"] = "1.0"
    plane: RoutePlane
    mode: RouteMode
    route_id: NonEmptyString
    route_configuration_digest: Sha256Digest
    input_identity_sha256: Sha256Digest
    experiment_id: NonEmptyString | None = None
    dispatches: tuple[RouteDispatch, ...]

    @model_validator(mode="after")
    def validate_authority_boundary(self) -> Self:
        deployment_ids = tuple(item.deployment_id for item in self.dispatches)
        if len(deployment_ids) != len(set(deployment_ids)):
            raise ValueError("route decisions cannot dispatch one deployment more than once")
        authoritative = tuple(
            item
            for item in self.dispatches
            if item.disposition is DispatchDisposition.AUTHORITATIVE
        )
        if self.plane is RoutePlane.PRODUCTION:
            if self.experiment_id is not None:
                raise ValueError("production decisions cannot carry an experiment_id")
            if self.mode not in {RouteMode.PRIMARY, RouteMode.CANARY}:
                raise ValueError("production decisions require PRIMARY or CANARY mode")
            if len(self.dispatches) != 1 or len(authoritative) != 1:
                raise ValueError("production decisions require one authoritative dispatch")
        else:
            if self.experiment_id is None:
                raise ValueError("experiment decisions require an experiment_id")
            if self.mode not in {RouteMode.PAIRED, RouteMode.SHADOW}:
                raise ValueError("experiment decisions require PAIRED or SHADOW mode")
            if authoritative:
                raise ValueError("experiment decisions cannot produce authoritative dispatches")
        return self

    @property
    def authoritative_deployment_id(self) -> str | None:
        """Return the sole authoritative deployment, if this is a production decision."""

        for dispatch in self.dispatches:
            if dispatch.disposition is DispatchDisposition.AUTHORITATIVE:
                return dispatch.deployment_id
        return None


def _route_configuration_digest(route: StrictModel) -> Sha256Digest:
    return semantic_sha256(route.model_dump(mode="json"))


def endpoint_config_digest(endpoint_config: StrictModel) -> Sha256Digest:
    """Return the stable identity used to bind a deployment to its endpoint config."""

    if not isinstance(endpoint_config, StrictModel):
        raise TypeError("endpoint_config must be a StrictModel")
    return semantic_sha256(endpoint_config.model_dump(mode="json"))


def _require_input_digest(value: object) -> Sha256Digest:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ModelRoutingError("input_identity_sha256 must be a lowercase SHA-256 digest")
    return value


def _sample(*, input_identity_sha256: Sha256Digest, route_id: str, policy_version: str) -> float:
    preimage = "\x1f".join((input_identity_sha256, route_id, policy_version)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(preimage).digest(), "big") / (1 << 256)


def _validate_distinct_deployments(*, control: ModelDeployment, candidate: ModelDeployment) -> None:
    if control.deployment_id == candidate.deployment_id:
        raise ValueError("control and candidate deployments must be distinct")


class ProductionRoute(StrictModel):
    """The stable authoritative model route with external evidence references."""

    schema_version: Literal["1.0"] = "1.0"
    route_id: NonEmptyString
    policy_version: SchemaVersion
    deployment: ModelDeployment
    authorization: ProductionRouteAuthorization

    @property
    def configuration_digest(self) -> Sha256Digest:
        return _route_configuration_digest(self)

    def decide(self, *, input_identity_sha256: str) -> ModelRouteDecision:
        digest = _require_input_digest(input_identity_sha256)
        return ModelRouteDecision(
            plane=RoutePlane.PRODUCTION,
            mode=RouteMode.PRIMARY,
            route_id=self.route_id,
            route_configuration_digest=self.configuration_digest,
            input_identity_sha256=digest,
            dispatches=(
                RouteDispatch(
                    deployment_id=self.deployment.deployment_id,
                    role=ModelRouteRole.CONTROL,
                    disposition=DispatchDisposition.AUTHORITATIVE,
                ),
            ),
        )


class ExperimentRoute(StrictModel):
    """A deterministic, non-authoritative two-model experiment route.

    ``PAIRED`` replays both deployments for selected inputs.  ``SHADOW`` only
    dispatches the candidate and relies on the production control result being
    retained separately.  Neither mode creates an authoritative selection.
    """

    schema_version: Literal["1.0"] = "1.0"
    route_id: NonEmptyString
    policy_version: SchemaVersion
    mode: Literal[RouteMode.PAIRED, RouteMode.SHADOW]
    sample_ratio: UnitInterval
    contract: ExperimentContract

    @property
    def configuration_digest(self) -> Sha256Digest:
        return _route_configuration_digest(self)

    def selected(self, *, input_identity_sha256: str) -> bool:
        digest = _require_input_digest(input_identity_sha256)
        if self.sample_ratio == 0.0:
            return False
        if self.sample_ratio == 1.0:
            return True
        return (
            _sample(
                input_identity_sha256=digest,
                route_id=self.route_id,
                policy_version=self.policy_version,
            )
            < self.sample_ratio
        )

    def decide(self, *, input_identity_sha256: str) -> ModelRouteDecision:
        digest = _require_input_digest(input_identity_sha256)
        dispatches: tuple[RouteDispatch, ...]
        if not self.selected(input_identity_sha256=digest):
            dispatches = ()
        elif self.mode is RouteMode.PAIRED:
            dispatches = (
                RouteDispatch(
                    deployment_id=self.contract.control.deployment_id,
                    role=ModelRouteRole.CONTROL,
                    disposition=DispatchDisposition.OBSERVATION,
                ),
                RouteDispatch(
                    deployment_id=self.contract.candidate.deployment_id,
                    role=ModelRouteRole.CANDIDATE,
                    disposition=DispatchDisposition.OBSERVATION,
                ),
            )
        else:
            dispatches = (
                RouteDispatch(
                    deployment_id=self.contract.candidate.deployment_id,
                    role=ModelRouteRole.CANDIDATE,
                    disposition=DispatchDisposition.OBSERVATION,
                ),
            )
        return ModelRouteDecision(
            plane=RoutePlane.EXPERIMENT,
            mode=self.mode,
            route_id=self.route_id,
            route_configuration_digest=self.configuration_digest,
            input_identity_sha256=digest,
            experiment_id=self.contract.experiment_id,
            dispatches=dispatches,
        )


class CanaryRoute(StrictModel):
    """A candidate-serving route that deterministically selects a cohort.

    A production composition must validate ``authorization`` before accepting
    this route for live authority.
    """

    schema_version: Literal["1.0"] = "1.0"
    route_id: NonEmptyString
    policy_version: SchemaVersion
    candidate_ratio: UnitInterval
    control: ModelDeployment
    candidate: ModelDeployment
    authorization: ProductionRouteAuthorization

    @model_validator(mode="after")
    def validate_deployments(self) -> Self:
        _validate_distinct_deployments(control=self.control, candidate=self.candidate)
        return self

    @property
    def configuration_digest(self) -> Sha256Digest:
        return _route_configuration_digest(self)

    def candidate_selected(self, *, input_identity_sha256: str) -> bool:
        digest = _require_input_digest(input_identity_sha256)
        if self.candidate_ratio == 0.0:
            return False
        if self.candidate_ratio == 1.0:
            return True
        return (
            _sample(
                input_identity_sha256=digest,
                route_id=self.route_id,
                policy_version=self.policy_version,
            )
            < self.candidate_ratio
        )

    def decide(self, *, input_identity_sha256: str) -> ModelRouteDecision:
        digest = _require_input_digest(input_identity_sha256)
        selected_candidate = self.candidate_selected(input_identity_sha256=digest)
        deployment = self.candidate if selected_candidate else self.control
        role = ModelRouteRole.CANDIDATE if selected_candidate else ModelRouteRole.CONTROL
        return ModelRouteDecision(
            plane=RoutePlane.PRODUCTION,
            mode=RouteMode.CANARY,
            route_id=self.route_id,
            route_configuration_digest=self.configuration_digest,
            input_identity_sha256=digest,
            dispatches=(
                RouteDispatch(
                    deployment_id=deployment.deployment_id,
                    role=role,
                    disposition=DispatchDisposition.AUTHORITATIVE,
                ),
            ),
        )


class ModelRouter:
    """Route production and experiment traffic through separate explicit methods.

    Production calls never fan out to an experiment.  Experiment calls require
    an experiment identifier and always yield observational dispatches.  The
    later execution bridge can map deployment identifiers to independent
    adapter/orchestrator instances without changing this authority decision.
    """

    def __init__(
        self,
        *,
        production: ProductionRoute,
        experiments: Mapping[str, ExperimentRoute] | None = None,
        canary: CanaryRoute | None = None,
        canary_authorization_verifier: ProductionRouteAuthorizationVerifier | None = None,
    ) -> None:
        if not isinstance(production, ProductionRoute):
            raise TypeError("production must be ProductionRoute")
        try:
            production = ProductionRoute.model_validate(production.model_dump())
        except ValueError as error:
            raise ModelRoutingError("production route configuration is invalid") from error
        if canary is not None and not isinstance(canary, CanaryRoute):
            raise TypeError("canary must be CanaryRoute")
        if canary is not None:
            try:
                canary = CanaryRoute.model_validate(canary.model_dump())
            except ValueError as error:
                raise ModelRoutingError("canary route configuration is invalid") from error
            if canary.control.deployment_id != production.deployment.deployment_id:
                raise ModelRoutingError(
                    "canary control deployment must match the configured production deployment"
                )
            if canary_authorization_verifier is None:
                raise ModelRoutingError("serving canary requires verified external authorization")
            verify = getattr(canary_authorization_verifier, "verify", None)
            if not callable(verify):
                raise TypeError("canary_authorization_verifier must define verify")
            try:
                authorized = verify(
                    authorization=canary.authorization,
                    control=canary.control,
                    candidate=canary.candidate,
                )
            except Exception as error:
                raise ModelRoutingError(
                    "serving canary authorization verification failed"
                ) from error
            if authorized is not True:
                raise ModelRoutingError("serving canary authorization was not verified")
        normalized_experiments: dict[str, ExperimentRoute] = {}
        for experiment_id, route in (experiments or {}).items():
            if not isinstance(experiment_id, str) or not experiment_id:
                raise ModelRoutingError("experiment route keys must be nonempty strings")
            if not isinstance(route, ExperimentRoute):
                raise TypeError("experiment routes must be ExperimentRoute instances")
            try:
                route = ExperimentRoute.model_validate(route.model_dump())
            except ValueError as error:
                raise ModelRoutingError("experiment route configuration is invalid") from error
            if route.contract.experiment_id != experiment_id:
                raise ModelRoutingError("experiment route key must match route experiment_id")
            if experiment_id in normalized_experiments:
                raise ModelRoutingError("experiment identifiers must be unique")
            normalized_experiments[experiment_id] = route
        self._production = production
        self._canary = canary
        self._experiments = normalized_experiments

    @property
    def production(self) -> ProductionRoute:
        return self._production

    @property
    def canary(self) -> CanaryRoute | None:
        return self._canary

    @property
    def experiments(self) -> tuple[ExperimentRoute, ...]:
        return tuple(self._experiments.values())

    def route_production(self, *, input_identity_sha256: str) -> ModelRouteDecision:
        """Return the only authoritative decision available from this router."""

        if self._canary is not None:
            return self._canary.decide(input_identity_sha256=input_identity_sha256)
        return self._production.decide(input_identity_sha256=input_identity_sha256)

    def route_experiment(
        self, *, experiment_id: str, input_identity_sha256: str
    ) -> ModelRouteDecision:
        """Return a non-authoritative paired or shadow decision for one experiment."""

        if not isinstance(experiment_id, str) or not experiment_id:
            raise ModelRoutingError("experiment_id must be a nonempty string")
        route = self._experiments.get(experiment_id)
        if route is None:
            raise ModelRoutingError(f"unknown experiment route: {experiment_id!r}")
        return route.decide(input_identity_sha256=input_identity_sha256)


__all__ = [
    "CanaryRoute",
    "DispatchDisposition",
    "ExperimentContract",
    "ExperimentInputRepresentation",
    "ExperimentIsolationProfile",
    "ExperimentRoute",
    "ModelDeployment",
    "ModelRouteDecision",
    "ModelRouteRole",
    "ModelRouter",
    "ModelRoutingError",
    "ProductionRoute",
    "ProductionRouteAuthorization",
    "ProductionRouteAuthorizationVerifier",
    "RouteDispatch",
    "RouteMode",
    "RoutePlane",
    "endpoint_config_digest",
]
