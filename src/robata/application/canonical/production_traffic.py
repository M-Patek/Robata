"""Production-plane traffic binding for canonical Robata workers.

The canonical composition owns the reviewed primary route and PostgreSQL
authority.  The inference routing module owns the explicit production and
experiment planes.  This module is the small operational seam between them:
it converts the canonical primary route into an inference ``ModelRouter``,
validates optional shadow/canary routes, and emits non-secret sidecars that a
source-specific worker can persist next to its E2E trace.

No experiment result is ever returned as an authoritative production result.
The shadow helper is deliberately non-blocking and delegates lifecycle and
bounded concurrency to :class:`ProductionShadowCoordinator`; durable queues,
automatic promotion, and automatic rollback remain outside this bridge.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, Self, cast

from pydantic import StringConstraints, model_validator

from robata.application.canonical.production_routing import (
    ModelDeployment as CanonicalModelDeployment,
)
from robata.application.canonical.production_routing import (
    ProductionRoute as CanonicalProductionRoute,
)
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.inference.experiment_execution import (
    ExperimentInvocation,
)
from robata.inference.production_shadow import (
    ProductionShadowBudget,
    ProductionShadowCoordinator,
    ProductionShadowExecutor,
    ProductionShadowObservation,
)
from robata.inference.routing import (
    CanaryRoute,
    DispatchDisposition,
    ExperimentRoute,
    ModelDeployment,
    ModelRouteDecision,
    ModelRouter,
    ModelRoutingError,
    ProductionRoute,
    ProductionRouteAuthorization,
    ProductionRouteAuthorizationVerifier,
    RouteMode,
    RoutePlane,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]


class ProductionTrafficError(ValueError):
    """Raised when a production traffic binding is unsafe or inconsistent."""


class PrimaryRouteReleaseVerifier(Protocol):
    """External verifier for the exact canonical primary release evidence."""

    def verify_primary(self, *, authorization: object, deployment: object) -> bool:
        """Return true only when the canonical primary release is admitted."""


class ShadowSubmitHook(Protocol):
    """The narrow nonblocking candidate hook exposed to a production worker."""

    def __call__(
        self,
        *,
        primary_inference_id: str,
        invocation: ExperimentInvocation,
    ) -> ProductionShadowObservation | None:
        """Submit observation-only candidate work without awaiting it."""


class ProductionTrafficReadiness(StrictModel):
    """Non-secret route-binding readiness sidecar.

    This sidecar describes what is bound, not a release decision.  In
    particular, promotion and rollback are explicit false values so a worker
    cannot mistake this bridge for a rollout controller.
    """

    schema_version: Literal["robata-production-traffic-readiness-v1"] = (
        "robata-production-traffic-readiness-v1"
    )
    state: Literal["READY", "NOT_READY"]
    production_plane: Literal["PRIMARY", "CANARY"]
    primary_route_id: NonEmptyString
    primary_deployment_id: NonEmptyString
    primary_route_configuration_digest: Sha256Digest
    primary_release_verified: bool
    shadow_enabled: bool
    shadow_submit_available: bool
    shadow_route_id: NonEmptyString | None = None
    shadow_experiment_id: NonEmptyString | None = None
    shadow_candidate_deployment_id: NonEmptyString | None = None
    canary_enabled: bool
    canary_authorization_verified: bool
    canary_route_id: NonEmptyString | None = None
    canary_candidate_deployment_id: NonEmptyString | None = None
    canonical_authority_plane: Literal["PRODUCTION"] = "PRODUCTION"
    experiment_authority_allowed: Literal[False] = False
    automatic_promotion: Literal[False] = False
    automatic_rollback: Literal[False] = False
    blockers: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.state == "READY" and self.blockers:
            raise ValueError("ready traffic binding cannot contain blockers")
        if self.shadow_enabled:
            if (
                self.shadow_route_id is None
                or self.shadow_experiment_id is None
                or self.shadow_candidate_deployment_id is None
            ):
                raise ValueError("enabled shadow binding requires route and candidate facts")
        elif any(
            value is not None
            for value in (
                self.shadow_route_id,
                self.shadow_experiment_id,
                self.shadow_candidate_deployment_id,
            )
        ):
            raise ValueError("disabled shadow binding cannot contain route facts")
        if self.canary_enabled:
            if self.canary_route_id is None or self.canary_candidate_deployment_id is None:
                raise ValueError("enabled canary binding requires route and candidate facts")
            if not self.canary_authorization_verified:
                raise ValueError("enabled canary binding requires verified authorization")
        elif any(
            value is not None
            for value in (self.canary_route_id, self.canary_candidate_deployment_id)
        ):
            raise ValueError("disabled canary binding cannot contain route facts")
        return self

    @property
    def configuration_digest(self) -> Sha256Digest:
        """Return a content digest suitable for a worker sidecar reference."""

        return semantic_sha256(self.model_dump(mode="json"))


class ProductionTrafficRoutePlan(StrictModel):
    """One deterministic production/observation decision for an input identity."""

    schema_version: Literal["robata-production-traffic-route-plan-v1"] = (
        "robata-production-traffic-route-plan-v1"
    )
    input_identity_sha256: Sha256Digest
    primary: ModelRouteDecision
    shadow: ModelRouteDecision | None = None
    canonical_authoritative_deployment_id: NonEmptyString
    experiment_authoritative_deployment_id: None = None
    shadow_selected: bool
    shadow_observation_only: Literal[True] = True
    plan_digest: Sha256Digest = "0" * 64

    @model_validator(mode="after")
    def validate_authority_boundary(self) -> Self:
        if self.primary.plane is not RoutePlane.PRODUCTION:
            raise ValueError("production traffic plan primary decision must be production-plane")
        if self.primary.authoritative_deployment_id != self.canonical_authoritative_deployment_id:
            raise ValueError("canonical authoritative deployment does not match primary decision")
        if self.shadow is None:
            if self.shadow_selected:
                raise ValueError("shadow_selected requires a shadow decision")
        else:
            if self.shadow.plane is not RoutePlane.EXPERIMENT:
                raise ValueError("shadow decision must be experiment-plane")
            if self.shadow.mode is not RouteMode.SHADOW:
                raise ValueError("production shadow plan requires SHADOW mode")
            if any(
                dispatch.disposition is DispatchDisposition.AUTHORITATIVE
                for dispatch in self.shadow.dispatches
            ):
                raise ValueError("shadow decision cannot contain authoritative dispatches")
            if self.shadow_selected != bool(self.shadow.dispatches):
                raise ValueError("shadow_selected does not match the deterministic decision")
        calculated = semantic_sha256(
            {
                "input_identity_sha256": self.input_identity_sha256,
                "primary": self.primary.model_dump(mode="json"),
                "shadow": self.shadow.model_dump(mode="json") if self.shadow is not None else None,
                "canonical_authoritative_deployment_id": self.canonical_authoritative_deployment_id,
                "shadow_selected": self.shadow_selected,
            }
        )
        # ``plan_digest`` is populated by ``build_route_plan``.  Accept the
        # zero sentinel only while Pydantic constructs an instance from a
        # caller; a non-sentinel value must match exactly.
        if self.plan_digest != "0" * 64 and self.plan_digest != calculated:
            raise ValueError("traffic route plan digest does not match its contents")
        object.__setattr__(self, "plan_digest", calculated)
        return self


def canonical_deployment_to_inference(
    deployment: CanonicalModelDeployment,
) -> ModelDeployment:
    """Convert a canonical deployment without changing any pinned facts."""

    if not isinstance(deployment, CanonicalModelDeployment):
        raise TypeError("deployment must be the canonical application ModelDeployment")
    try:
        return ModelDeployment.model_validate(deployment.model_dump(mode="json"), strict=True)
    except ValueError as error:
        raise ProductionTrafficError(
            "canonical primary deployment is not inference-route compatible"
        ) from error


def canonical_route_to_inference(
    route: CanonicalProductionRoute,
) -> ProductionRoute:
    """Convert one canonical primary route into the plane-separated route type."""

    if not isinstance(route, CanonicalProductionRoute):
        raise TypeError("route must be the canonical application ProductionRoute")
    deployment = canonical_deployment_to_inference(route.deployment)
    try:
        authorization = ProductionRouteAuthorization.model_validate(
            route.authorization.model_dump(mode="json"), strict=True
        )
        return ProductionRoute(
            route_id=route.route_id,
            policy_version=route.policy_version,
            deployment=deployment,
            authorization=authorization,
        )
    except ValueError as error:
        raise ProductionTrafficError(
            "canonical primary route is not inference-route compatible"
        ) from error


def _deployment_exact_match(left: ModelDeployment, right: ModelDeployment) -> bool:
    """Compare all pinned deployment facts, not just the deployment id."""

    return left == right


def _ensure_independent_candidate(
    *, control: ModelDeployment, candidate: ModelDeployment, label: str
) -> None:
    if candidate.deployment_id == control.deployment_id:
        raise ProductionTrafficError(
            f"{label} candidate deployment must be independent from primary control"
        )
    # A new deployment identifier cannot be used as a false isolation claim if
    # it points at the same endpoint configuration.  Capability snapshots may
    # legitimately be shared for equivalent model builds, so the endpoint
    # identity is the required separation boundary here.
    if candidate.endpoint_config_digest == control.endpoint_config_digest:
        raise ProductionTrafficError(
            f"{label} candidate deployment must use an independent endpoint configuration"
        )


def _verify_primary_release(
    *,
    route: CanonicalProductionRoute,
    verifier: PrimaryRouteReleaseVerifier | None,
) -> bool:
    if verifier is None:
        return False
    verify = getattr(verifier, "verify_primary", None)
    if not callable(verify):
        raise ProductionTrafficError("primary release verifier must define verify_primary")
    try:
        result = verify(authorization=route.authorization, deployment=route.deployment)
    except Exception as error:
        raise ProductionTrafficError("primary route release verification failed") from error
    if result is not True:
        raise ProductionTrafficError("primary route release was not verified")
    return True


class ProductionTrafficBridge:
    """Bind canonical primary traffic to isolated shadow/canary planes."""

    @classmethod
    def from_canonical_primary_route(
        cls, route: CanonicalProductionRoute, **kwargs: object
    ) -> ProductionTrafficBridge:
        """Construct a bridge from the reviewed canonical route object."""

        return cls(canonical_primary_route=route, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_bootstrap(cls, bootstrap: object, **kwargs: object) -> ProductionTrafficBridge:
        """Construct a bridge from a loaded production bootstrap document."""

        route = getattr(bootstrap, "primary_route", None)
        if not isinstance(route, CanonicalProductionRoute):
            raise TypeError("bootstrap must expose a canonical primary_route")
        return cls(canonical_primary_route=route, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_runtime(cls, runtime: object, **kwargs: object) -> ProductionTrafficBridge:
        """Construct a bridge from a production runtime's pinned primary route."""

        route = getattr(runtime, "primary_route", None)
        if not isinstance(route, CanonicalProductionRoute):
            raise TypeError("runtime must expose a canonical primary_route")
        return cls(canonical_primary_route=route, **kwargs)  # type: ignore[arg-type]

    def __init__(
        self,
        *,
        canonical_primary_route: CanonicalProductionRoute,
        shadow_route: ExperimentRoute | None = None,
        canary_route: CanaryRoute | None = None,
        canary_authorization_verifier: ProductionRouteAuthorizationVerifier | None = None,
        primary_release_verifier: PrimaryRouteReleaseVerifier | None = None,
        require_primary_release_verification: bool = False,
        shadow_coordinator: ProductionShadowCoordinator | None = None,
        shadow_executor: ProductionShadowExecutor | None = None,
        shadow_budget: ProductionShadowBudget | None = None,
    ) -> None:
        if not isinstance(canonical_primary_route, CanonicalProductionRoute):
            raise TypeError("canonical_primary_route must be canonical ProductionRoute")
        inference_primary = canonical_route_to_inference(canonical_primary_route)
        primary_verified = _verify_primary_release(
            route=canonical_primary_route,
            verifier=primary_release_verifier,
        )
        if require_primary_release_verification and not primary_verified:
            raise ProductionTrafficError("primary route release verifier is required")

        if canary_route is not None and not isinstance(canary_route, CanaryRoute):
            raise TypeError("canary_route must be an inference CanaryRoute")
        if canary_route is not None:
            if not _deployment_exact_match(canary_route.control, inference_primary.deployment):
                raise ProductionTrafficError(
                    "canary control deployment must exactly match primary route"
                )
            _ensure_independent_candidate(
                control=inference_primary.deployment,
                candidate=canary_route.candidate,
                label="canary",
            )
            if canary_authorization_verifier is None:
                raise ProductionTrafficError(
                    "canary route requires an external authorization verifier"
                )

        if shadow_route is not None and not isinstance(shadow_route, ExperimentRoute):
            raise TypeError("shadow_route must be an inference ExperimentRoute")
        if shadow_route is not None:
            if shadow_route.mode is not RouteMode.SHADOW:
                raise ProductionTrafficError("shadow_route must use SHADOW mode")
            if not _deployment_exact_match(
                shadow_route.contract.control, inference_primary.deployment
            ):
                raise ProductionTrafficError(
                    "shadow control deployment must exactly match primary route"
                )
            _ensure_independent_candidate(
                control=inference_primary.deployment,
                candidate=shadow_route.contract.candidate,
                label="shadow",
            )

        if shadow_coordinator is not None and shadow_route is None:
            raise ProductionTrafficError("shadow coordinator requires a shadow route")
        if shadow_executor is not None and shadow_budget is None:
            raise ProductionTrafficError("shadow executor requires a shadow budget")
        if shadow_budget is not None and shadow_executor is None:
            raise ProductionTrafficError("shadow budget requires a shadow executor")
        if shadow_coordinator is not None and (
            shadow_executor is not None or shadow_budget is not None
        ):
            raise ProductionTrafficError(
                "provide either shadow_coordinator or executor/budget, not both"
            )
        if shadow_coordinator is None and shadow_route is not None and shadow_executor is not None:
            assert shadow_budget is not None
            try:
                shadow_coordinator = ProductionShadowCoordinator(
                    route=shadow_route,
                    executor=shadow_executor,
                    budget=shadow_budget,
                )
            except Exception as error:
                raise ProductionTrafficError(
                    "shadow coordinator configuration is invalid"
                ) from error

        try:
            router = ModelRouter(
                production=inference_primary,
                experiments=(
                    {shadow_route.contract.experiment_id: shadow_route}
                    if shadow_route is not None
                    else None
                ),
                canary=canary_route,
                canary_authorization_verifier=canary_authorization_verifier,
            )
        except (ModelRoutingError, TypeError, ValueError) as error:
            raise ProductionTrafficError(
                f"plane-separated model router configuration is invalid: {error}"
            ) from error

        if shadow_coordinator is not None and shadow_coordinator.route != shadow_route:
            raise ProductionTrafficError("shadow coordinator route does not match shadow route")

        self._canonical_primary_route = canonical_primary_route
        self._primary_route = inference_primary
        self._router = router
        self._shadow_route = shadow_route
        self._canary_route = canary_route
        self._shadow_coordinator = shadow_coordinator
        self._primary_release_verified = primary_verified
        self._canary_authorization_verified = (
            canary_route is None or canary_authorization_verifier is not None
        )

    @property
    def canonical_primary_route(self) -> CanonicalProductionRoute:
        return self._canonical_primary_route

    @property
    def primary_route(self) -> ProductionRoute:
        return self._primary_route

    @property
    def router(self) -> ModelRouter:
        return self._router

    @property
    def shadow_route(self) -> ExperimentRoute | None:
        return self._shadow_route

    @property
    def canary_route(self) -> CanaryRoute | None:
        return self._canary_route

    @property
    def shadow_submit_available(self) -> bool:
        return self._shadow_coordinator is not None

    @property
    def readiness(self) -> ProductionTrafficReadiness:
        blockers: list[str] = []
        if not self._primary_release_verified:
            blockers.append("PRIMARY_ROUTE_RELEASE_UNVERIFIED")
        if self._shadow_route is not None and self._shadow_coordinator is None:
            blockers.append("SHADOW_SUBMIT_HOOK_UNBOUND")
        state: Literal["READY", "NOT_READY"] = "READY" if not blockers else "NOT_READY"
        return ProductionTrafficReadiness(
            state=state,
            production_plane="CANARY" if self._canary_route is not None else "PRIMARY",
            primary_route_id=self._primary_route.route_id,
            primary_deployment_id=self._primary_route.deployment.deployment_id,
            primary_route_configuration_digest=self._primary_route.configuration_digest,
            primary_release_verified=self._primary_release_verified,
            shadow_enabled=self._shadow_route is not None,
            shadow_submit_available=self.shadow_submit_available,
            shadow_route_id=self._shadow_route.route_id if self._shadow_route else None,
            shadow_experiment_id=(
                self._shadow_route.contract.experiment_id if self._shadow_route else None
            ),
            shadow_candidate_deployment_id=(
                self._shadow_route.contract.candidate.deployment_id if self._shadow_route else None
            ),
            canary_enabled=self._canary_route is not None,
            canary_authorization_verified=self._canary_authorization_verified,
            canary_route_id=self._canary_route.route_id if self._canary_route else None,
            canary_candidate_deployment_id=(
                self._canary_route.candidate.deployment_id if self._canary_route else None
            ),
            blockers=tuple(blockers),
        )

    @property
    def readiness_sidecar(self) -> ProductionTrafficReadiness:
        """Alias used by workers that persist sidecars alongside traces."""

        return self.readiness

    def require_ready(self) -> ProductionTrafficReadiness:
        """Fail closed before a worker starts authoritative traffic."""

        readiness = self.readiness
        if readiness.state != "READY":
            raise ProductionTrafficError(
                "production traffic bridge is not ready: " + ", ".join(readiness.blockers)
            )
        return readiness

    def decide(self, *, input_identity_sha256: str) -> ProductionTrafficRoutePlan:
        """Return deterministic primary and optional observation-only decisions."""

        primary = self._router.route_production(input_identity_sha256=input_identity_sha256)
        shadow = (
            self._router.route_experiment(
                experiment_id=self._shadow_route.contract.experiment_id,
                input_identity_sha256=input_identity_sha256,
            )
            if self._shadow_route is not None
            else None
        )
        return ProductionTrafficRoutePlan(
            input_identity_sha256=input_identity_sha256,
            primary=primary,
            shadow=shadow,
            canonical_authoritative_deployment_id=cast(str, primary.authoritative_deployment_id),
            shadow_selected=bool(shadow and shadow.dispatches),
        )

    def route_plan(self, *, input_identity_sha256: str) -> ProductionTrafficRoutePlan:
        """Alias for ``decide`` used by source-specific workers."""

        return self.decide(input_identity_sha256=input_identity_sha256)

    def decide_ready(self, *, input_identity_sha256: str) -> ProductionTrafficRoutePlan:
        """Require all configured admission checks before issuing a worker plan."""

        self.require_ready()
        return self.decide(input_identity_sha256=input_identity_sha256)

    def route_production(self, *, input_identity_sha256: str) -> ModelRouteDecision:
        """Return only the authoritative production decision."""

        return self._router.route_production(input_identity_sha256=input_identity_sha256)

    def submit_shadow(
        self,
        *,
        primary_inference_id: str,
        invocation: ExperimentInvocation,
    ) -> ProductionShadowObservation | None:
        """Submit shadow work without awaiting candidate completion.

        ``None`` means no shadow route is configured.  A configured route with
        no executor is represented by a NOT_READY sidecar rather than silently
        claiming that candidate work was submitted.
        """

        if self._shadow_route is None:
            return None
        if self._shadow_coordinator is None:
            raise ProductionTrafficError("shadow submit hook is not bound")
        try:
            return self._shadow_coordinator.submit(
                primary_inference_id=primary_inference_id,
                invocation=invocation,
            )
        except Exception as error:
            raise ProductionTrafficError("nonblocking shadow submit failed") from error

    def submit_shadow_nonblocking(
        self,
        *,
        primary_inference_id: str,
        invocation: ExperimentInvocation,
    ) -> ProductionShadowObservation | None:
        """Explicitly named alias for the worker-facing shadow hook."""

        return self.submit_shadow(
            primary_inference_id=primary_inference_id,
            invocation=invocation,
        )

    async def drain_shadow(self) -> None:
        """Drain in-process shadow tasks during controlled worker shutdown."""

        if self._shadow_coordinator is not None:
            await self._shadow_coordinator.drain()

    def shadow_observations(self) -> tuple[ProductionShadowObservation, ...]:
        """Return local shadow observations; never a canonical result."""

        if self._shadow_coordinator is None:
            return ()
        return self._shadow_coordinator.observations

    @property
    def route_plan_digest(self) -> Sha256Digest:
        """Digest the immutable route/readiness binding for worker provenance."""

        return semantic_sha256(
            {
                "primary": self._primary_route.model_dump(mode="json"),
                "shadow": self._shadow_route.model_dump(mode="json")
                if self._shadow_route is not None
                else None,
                "canary": self._canary_route.model_dump(mode="json")
                if self._canary_route is not None
                else None,
                "readiness": self.readiness.model_dump(mode="json"),
            }
        )


# Public construction aliases keep call sites readable while retaining one
# implementation and one validation path.
def build_production_traffic_bridge(**kwargs: object) -> ProductionTrafficBridge:
    """Build a fail-closed plane-separated production traffic bridge."""

    return ProductionTrafficBridge(**kwargs)  # type: ignore[arg-type]


def bind_production_traffic(**kwargs: object) -> ProductionTrafficBridge:
    """Alias for ``build_production_traffic_bridge``."""

    return build_production_traffic_bridge(**kwargs)


__all__ = [
    "PrimaryRouteReleaseVerifier",
    "ProductionTrafficBridge",
    "ProductionTrafficError",
    "ProductionTrafficReadiness",
    "ProductionTrafficRoutePlan",
    "ShadowSubmitHook",
    "bind_production_traffic",
    "build_production_traffic_bridge",
    "canonical_deployment_to_inference",
    "canonical_route_to_inference",
]
