from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.inference.routing import (
    CanaryRoute,
    DispatchDisposition,
    ExperimentContract,
    ExperimentInputRepresentation,
    ExperimentIsolationProfile,
    ExperimentRoute,
    ModelDeployment,
    ModelRouteDecision,
    ModelRouter,
    ModelRouteRole,
    ModelRoutingError,
    ProductionRoute,
    ProductionRouteAuthorization,
    ProductionRouteAuthorizationVerifier,
    RouteDispatch,
    RouteMode,
    RoutePlane,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _deployment(*, value: int, name: str) -> ModelDeployment:
    return ModelDeployment(
        deployment_id=f"runpod-{name}-4b-{value}",
        provider="runpod",
        model_name=name,
        model_version="2026.07.0",
        adapter_version="1.0",
        capability_snapshot_id=_uuid(value),
        capability_snapshot_digest=_digest(value),
        endpoint_config_digest=_digest(value + 100),
        max_concurrent_requests=4,
    )


def _authorization() -> ProductionRouteAuthorization:
    return ProductionRouteAuthorization(
        qualification_report_ref="r2://qualification/qwen-control.json",
        qualification_report_sha256=_digest(200),
        release_decision_ref="r2://release/decision.json",
        release_decision_sha256=_digest(201),
    )


def _production(*, deployment: ModelDeployment) -> ProductionRoute:
    return ProductionRoute(
        route_id="production-qwen-control",
        policy_version="1.0",
        deployment=deployment,
        authorization=_authorization(),
    )


def _contract(
    *,
    control: ModelDeployment,
    candidate: ModelDeployment,
    workload_digest: int = 300,
) -> ExperimentContract:
    return ExperimentContract(
        experiment_id="mage-vs-qwen-4b-dev-01",
        contract_version="1.0",
        workload_manifest_sha256=_digest(workload_digest),
        arrival_schedule_sha256=_digest(301),
        comparison_config_sha256=_digest(302),
        input_representation=ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING,
        isolation_profile=ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE,
        control=control,
        candidate=candidate,
    )


def _experiment(
    *,
    control: ModelDeployment,
    candidate: ModelDeployment,
    mode: RouteMode = RouteMode.PAIRED,
    sample_ratio: float = 1.0,
) -> ExperimentRoute:
    return ExperimentRoute(
        route_id="mage-vs-qwen-4b-route",
        policy_version="1.0",
        mode=mode,
        sample_ratio=sample_ratio,
        contract=_contract(control=control, candidate=candidate),
    )


class _RecordingCanaryAuthorizationVerifier:
    def __init__(self, *, accepted: bool = True, failure: Exception | None = None) -> None:
        self.accepted = accepted
        self.failure = failure
        self.calls: list[tuple[ProductionRouteAuthorization, ModelDeployment, ModelDeployment]] = []

    def verify(
        self,
        *,
        authorization: ProductionRouteAuthorization,
        control: ModelDeployment,
        candidate: ModelDeployment,
    ) -> bool:
        self.calls.append((authorization, control, candidate))
        if self.failure is not None:
            raise self.failure
        return self.accepted


def test_production_route_has_exactly_one_authoritative_control_dispatch() -> None:
    control = _deployment(value=1, name="Qwen3-VL-4B-Instruct")
    router = ModelRouter(production=_production(deployment=control))

    decision = router.route_production(input_identity_sha256=_digest(1_000))

    assert decision.plane is RoutePlane.PRODUCTION
    assert decision.mode is RouteMode.PRIMARY
    assert decision.experiment_id is None
    assert decision.authoritative_deployment_id == control.deployment_id
    assert decision.dispatches == (
        RouteDispatch(
            deployment_id=control.deployment_id,
            role=ModelRouteRole.CONTROL,
            disposition=DispatchDisposition.AUTHORITATIVE,
        ),
    )


def test_paired_experiment_is_observational_and_bound_to_its_contract() -> None:
    control = _deployment(value=1, name="Qwen3-VL-4B-Instruct")
    candidate = _deployment(value=2, name="Mage-VL-4B")
    experiment = _experiment(control=control, candidate=candidate)
    router = ModelRouter(
        production=_production(deployment=control),
        experiments={experiment.contract.experiment_id: experiment},
    )

    decision = router.route_experiment(
        experiment_id=experiment.contract.experiment_id,
        input_identity_sha256=_digest(1_001),
    )

    assert decision.plane is RoutePlane.EXPERIMENT
    assert decision.mode is RouteMode.PAIRED
    assert decision.experiment_id == experiment.contract.experiment_id
    assert decision.authoritative_deployment_id is None
    assert tuple(item.role for item in decision.dispatches) == (
        ModelRouteRole.CONTROL,
        ModelRouteRole.CANDIDATE,
    )
    assert all(item.disposition is DispatchDisposition.OBSERVATION for item in decision.dispatches)
    changed_contract = _contract(
        control=control,
        candidate=candidate,
        workload_digest=333,
    )
    assert changed_contract.contract_digest != experiment.contract.contract_digest


def test_experiment_sampling_is_stable_and_honors_zero_and_one_boundaries() -> None:
    control = _deployment(value=1, name="Qwen3-VL-4B-Instruct")
    candidate = _deployment(value=2, name="Mage-VL-4B")
    zero = _experiment(control=control, candidate=candidate, sample_ratio=0.0)
    full = _experiment(control=control, candidate=candidate, sample_ratio=1.0)
    partial = _experiment(control=control, candidate=candidate, sample_ratio=0.37)
    digest = _digest(1_002)

    assert zero.decide(input_identity_sha256=digest).dispatches == ()
    assert len(full.decide(input_identity_sha256=digest).dispatches) == 2
    assert partial.decide(input_identity_sha256=digest) == partial.decide(
        input_identity_sha256=digest
    )


def test_shadow_experiment_dispatches_only_the_candidate_without_authority() -> None:
    control = _deployment(value=1, name="Qwen3-VL-4B-Instruct")
    candidate = _deployment(value=2, name="Mage-VL-4B")
    experiment = _experiment(
        control=control,
        candidate=candidate,
        mode=RouteMode.SHADOW,
    )

    decision = experiment.decide(input_identity_sha256=_digest(1_003))

    assert decision.dispatches == (
        RouteDispatch(
            deployment_id=candidate.deployment_id,
            role=ModelRouteRole.CANDIDATE,
            disposition=DispatchDisposition.OBSERVATION,
        ),
    )
    assert decision.authoritative_deployment_id is None


def test_canary_requires_the_production_control_and_remains_one_authoritative_dispatch() -> None:
    control = _deployment(value=1, name="Qwen3-VL-4B-Instruct")
    candidate = _deployment(value=2, name="Mage-VL-4B")
    production = _production(deployment=control)
    canary = CanaryRoute(
        route_id="mage-canary",
        policy_version="1.0",
        candidate_ratio=1.0,
        control=control,
        candidate=candidate,
        authorization=_authorization(),
    )
    verifier = _RecordingCanaryAuthorizationVerifier()
    router = ModelRouter(
        production=production,
        canary=canary,
        canary_authorization_verifier=verifier,
    )

    decision = router.route_production(input_identity_sha256=_digest(1_004))

    assert decision.mode is RouteMode.CANARY
    assert decision.authoritative_deployment_id == candidate.deployment_id
    assert decision.dispatches[0].role is ModelRouteRole.CANDIDATE
    assert len(verifier.calls) == 1
    authorization, verified_control, verified_candidate = verifier.calls[0]
    assert authorization.qualification_report_ref == canary.authorization.qualification_report_ref
    assert (
        authorization.qualification_report_sha256
        == canary.authorization.qualification_report_sha256
    )
    assert authorization.release_decision_ref == canary.authorization.release_decision_ref
    assert authorization.release_decision_sha256 == canary.authorization.release_decision_sha256
    assert verified_control == control
    assert verified_candidate == candidate

    wrong_control = _deployment(value=3, name="Other-Control")
    with pytest.raises(ModelRoutingError, match="control deployment"):
        ModelRouter(
            production=production,
            canary=canary.model_copy(update={"control": wrong_control}),
            canary_authorization_verifier=verifier,
        )


def test_canary_is_rejected_without_an_authorization_verifier() -> None:
    control = _deployment(value=1, name="Qwen3-VL-4B-Instruct")
    candidate = _deployment(value=2, name="Mage-VL-4B")
    canary = CanaryRoute(
        route_id="mage-canary",
        policy_version="1.0",
        candidate_ratio=0.5,
        control=control,
        candidate=candidate,
        authorization=_authorization(),
    )

    with pytest.raises(ModelRoutingError, match="authorization"):
        ModelRouter(production=_production(deployment=control), canary=canary)


@pytest.mark.parametrize(
    "verifier",
    [
        _RecordingCanaryAuthorizationVerifier(accepted=False),
        _RecordingCanaryAuthorizationVerifier(failure=RuntimeError("unavailable")),
    ],
)
def test_canary_is_rejected_when_authorization_verification_fails(
    verifier: ProductionRouteAuthorizationVerifier,
) -> None:
    control = _deployment(value=1, name="Qwen3-VL-4B-Instruct")
    candidate = _deployment(value=2, name="Mage-VL-4B")
    canary = CanaryRoute(
        route_id="mage-canary",
        policy_version="1.0",
        candidate_ratio=0.5,
        control=control,
        candidate=candidate,
        authorization=_authorization(),
    )

    with pytest.raises(ModelRoutingError, match="authorization"):
        ModelRouter(
            production=_production(deployment=control),
            canary=canary,
            canary_authorization_verifier=verifier,
        )


@pytest.mark.parametrize(
    "canary_update",
    [
        {"candidate_ratio": 1.01},
        {
            "authorization": _authorization().model_copy(
                update={"release_decision_sha256": "not-a-sha256"}
            )
        },
    ],
)
def test_router_revalidates_model_copy_canary_mutations_at_construction(
    canary_update: dict[str, object],
) -> None:
    control = _deployment(value=1, name="Qwen3-VL-4B-Instruct")
    candidate = _deployment(value=2, name="Mage-VL-4B")
    canary = CanaryRoute(
        route_id="mage-canary",
        policy_version="1.0",
        candidate_ratio=0.5,
        control=control,
        candidate=candidate,
        authorization=_authorization(),
    ).model_copy(update=canary_update)

    with pytest.raises((ValidationError, ModelRoutingError)):
        ModelRouter(
            production=_production(deployment=control),
            canary=canary,
            canary_authorization_verifier=_RecordingCanaryAuthorizationVerifier(),
        )


def test_experiment_contract_and_decision_fail_closed_on_authority_errors() -> None:
    control = _deployment(value=1, name="Qwen3-VL-4B-Instruct")
    candidate = _deployment(value=2, name="Mage-VL-4B")

    with pytest.raises(ValidationError, match="selection_eligible"):
        ExperimentContract.model_validate(
            _contract(control=control, candidate=candidate).model_dump()
            | {"selection_eligible": True}
        )
    with pytest.raises(ValidationError, match="cannot produce authoritative"):
        ModelRouteDecision(
            plane=RoutePlane.EXPERIMENT,
            mode=RouteMode.PAIRED,
            route_id="invalid",
            route_configuration_digest=_digest(400),
            input_identity_sha256=_digest(401),
            experiment_id="experiment",
            dispatches=(
                RouteDispatch(
                    deployment_id=control.deployment_id,
                    role=ModelRouteRole.CONTROL,
                    disposition=DispatchDisposition.AUTHORITATIVE,
                ),
            ),
        )
    router = ModelRouter(production=_production(deployment=control))
    with pytest.raises(ModelRoutingError, match="unknown experiment"):
        router.route_experiment(
            experiment_id="unknown",
            input_identity_sha256=_digest(402),
        )
