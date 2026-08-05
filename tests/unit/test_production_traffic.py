"""Focused tests for the canonical production traffic bridge."""

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from robata.application.canonical.production_routing import (
    ModelDeployment as CanonicalModelDeployment,
)
from robata.application.canonical.production_routing import (
    ProductionRoute as CanonicalProductionRoute,
)
from robata.application.canonical.production_routing import (
    ProductionRouteAuthorization as CanonicalProductionRouteAuthorization,
)
from robata.application.canonical.production_traffic import (
    ProductionTrafficBridge,
    ProductionTrafficError,
    canonical_route_to_inference,
)
from robata.inference.experiment_execution import ExperimentInvocation
from robata.inference.models import VisionTask
from robata.inference.production_shadow import InMemoryProductionShadowBudget
from robata.inference.routing import (
    CanaryRoute,
    ExperimentContract,
    ExperimentInputRepresentation,
    ExperimentIsolationProfile,
    ExperimentRoute,
    ModelDeployment,
    ProductionRouteAuthorization,
    RouteMode,
    RoutePlane,
)


def _digest(number: int) -> str:
    return f"{number:064x}"


def _uuid(number: int) -> str:
    return str(UUID(int=number))


def _canonical_primary() -> CanonicalProductionRoute:
    deployment = CanonicalModelDeployment(
        deployment_id="qwen-control",
        provider="runpod",
        model_name="Qwen3-VL-4B-Instruct",
        model_version="2026.08.05",
        adapter_version="runpod-adapter-v1",
        capability_snapshot_id=_uuid(1),
        capability_snapshot_digest=_digest(1),
        endpoint_config_digest=_digest(2),
        max_concurrent_requests=1,
    )
    return CanonicalProductionRoute(
        route_id="production-qwen-control",
        policy_version="1.0",
        deployment=deployment,
        authorization=CanonicalProductionRouteAuthorization(
            qualification_report_ref="r2://reports/qwen.json",
            qualification_report_sha256=_digest(3),
            release_decision_ref="r2://releases/qwen.json",
            release_decision_sha256=_digest(4),
        ),
    )


def _inference_deployment(*, deployment_id: str, endpoint: int, name: str) -> ModelDeployment:
    return ModelDeployment(
        deployment_id=deployment_id,
        provider="runpod",
        model_name=name,
        model_version="2026.08.05",
        adapter_version="runpod-adapter-v1",
        capability_snapshot_id=_uuid(endpoint + 10),
        capability_snapshot_digest=_digest(endpoint + 20),
        endpoint_config_digest=_digest(endpoint),
        max_concurrent_requests=1,
    )


def _shadow_route(primary: ModelDeployment) -> ExperimentRoute:
    candidate = _inference_deployment(
        deployment_id="mage-shadow",
        endpoint=30,
        name="Mage-VL-4B",
    )
    return ExperimentRoute(
        route_id="mage-shadow-route",
        policy_version="1.0",
        mode=RouteMode.SHADOW,
        sample_ratio=1.0,
        contract=ExperimentContract(
            experiment_id="mage-shadow-01",
            contract_version="1.0",
            workload_manifest_sha256=_digest(40),
            arrival_schedule_sha256=_digest(41),
            comparison_config_sha256=_digest(42),
            input_representation=ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING,
            isolation_profile=ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE,
            control=primary,
            candidate=candidate,
        ),
    )


def _authorization() -> ProductionRouteAuthorization:
    return ProductionRouteAuthorization(
        qualification_report_ref="r2://reports/mage.json",
        qualification_report_sha256=_digest(50),
        release_decision_ref="r2://releases/mage.json",
        release_decision_sha256=_digest(51),
    )


class _Verifier:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[tuple[object, object, object]] = []

    def verify(self, *, authorization: object, control: object, candidate: object) -> bool:
        self.calls.append((authorization, control, candidate))
        return self.result


class _PrimaryVerifier:
    def __init__(self, result: bool = True) -> None:
        self.result = result

    def verify_primary(self, *, authorization: object, deployment: object) -> bool:
        del authorization, deployment
        return self.result


class _FailingShadowExecutor:
    async def execute(self, **kwargs: object) -> object:
        del kwargs
        raise RuntimeError("candidate unavailable")


def test_canonical_primary_is_converted_without_changing_pinned_facts() -> None:
    canonical = _canonical_primary()
    inference = canonical_route_to_inference(canonical)

    assert inference.route_id == canonical.route_id
    assert inference.deployment.model_dump(mode="json") == canonical.deployment.model_dump(
        mode="json"
    )
    assert inference.authorization.model_dump(mode="json") == canonical.authorization.model_dump(
        mode="json"
    )


def test_bridge_emits_authoritative_primary_and_observation_only_shadow_plan() -> None:
    canonical = _canonical_primary()
    primary = canonical_route_to_inference(canonical).deployment
    bridge = ProductionTrafficBridge(
        canonical_primary_route=canonical,
        shadow_route=_shadow_route(primary),
    )

    plan = bridge.decide(input_identity_sha256=_digest(100))

    assert plan.primary.plane is RoutePlane.PRODUCTION
    assert plan.primary.authoritative_deployment_id == "qwen-control"
    assert plan.shadow is not None
    assert plan.shadow.plane is RoutePlane.EXPERIMENT
    assert plan.shadow.mode is RouteMode.SHADOW
    assert plan.shadow_selected is True
    assert plan.experiment_authoritative_deployment_id is None
    assert bridge.readiness.state == "NOT_READY"
    assert "SHADOW_SUBMIT_HOOK_UNBOUND" in bridge.readiness.blockers
    assert bridge.readiness.automatic_promotion is False
    assert bridge.readiness.automatic_rollback is False


def test_shadow_submit_is_nonblocking_and_never_authoritative() -> None:
    canonical = _canonical_primary()
    primary = canonical_route_to_inference(canonical).deployment
    route = _shadow_route(primary)
    bridge = ProductionTrafficBridge(
        canonical_primary_route=canonical,
        shadow_route=route,
        primary_release_verifier=_PrimaryVerifier(),
        shadow_executor=_FailingShadowExecutor(),
        shadow_budget=InMemoryProductionShadowBudget(maximum_in_flight=1),
    )
    invocation = ExperimentInvocation(
        source_workload_manifest_sha256=_digest(40),
        input_identity_sha256=_digest(99),
        task=VisionTask.ACTION_EVIDENCE,
        package_set_id=None,
        mcap_id="mcap-1",
        camera_mapping_run_id="mapping-1",
        alignment_id="alignment-1",
        start_ns=0,
        end_ns=1,
        package_inputs=(),
        control=None,
        candidate=None,
        comparison_config={},
    )

    async def run() -> None:
        observation = bridge.submit_shadow(
            primary_inference_id=_uuid(99),
            invocation=invocation,
        )
        assert observation is not None
        assert observation.status.value == "QUEUED"
        assert bridge.readiness.state == "READY"
        await bridge.drain_shadow()

    asyncio.run(run())
    assert bridge.shadow_observations()[0].status.value == "FAILED"
    assert bridge.readiness.experiment_authority_allowed is False


def test_shadow_control_requires_exact_primary_match() -> None:
    canonical = _canonical_primary()
    primary = canonical_route_to_inference(canonical).deployment
    mismatched = primary.model_copy(update={"adapter_version": "different"})
    route = _shadow_route(mismatched)

    with pytest.raises(ProductionTrafficError, match="exactly match"):
        ProductionTrafficBridge(canonical_primary_route=canonical, shadow_route=route)


def test_shadow_candidate_requires_independent_endpoint() -> None:
    canonical = _canonical_primary()
    primary = canonical_route_to_inference(canonical).deployment
    route = _shadow_route(primary).model_copy(
        update={
            "contract": _shadow_route(primary).contract.model_copy(
                update={
                    "candidate": _inference_deployment(
                        deployment_id="mage-shadow",
                        endpoint=2,
                        name="Mage-VL-4B",
                    )
                }
            )
        }
    )

    with pytest.raises(ProductionTrafficError, match="independent endpoint"):
        ProductionTrafficBridge(canonical_primary_route=canonical, shadow_route=route)


def test_canary_requires_exact_external_authorization_and_primary_control() -> None:
    canonical = _canonical_primary()
    primary = canonical_route_to_inference(canonical).deployment
    candidate = _inference_deployment(
        deployment_id="mage-canary",
        endpoint=60,
        name="Mage-VL-4B",
    )
    canary = CanaryRoute(
        route_id="mage-canary-route",
        policy_version="1.0",
        candidate_ratio=0.25,
        control=primary,
        candidate=candidate,
        authorization=_authorization(),
    )
    verifier = _Verifier()
    bridge = ProductionTrafficBridge(
        canonical_primary_route=canonical,
        canary_route=canary,
        canary_authorization_verifier=verifier,
        primary_release_verifier=_PrimaryVerifier(),
    )

    assert bridge.readiness.state == "READY"
    assert bridge.readiness.production_plane == "CANARY"
    assert bridge.readiness.canary_authorization_verified is True
    assert len(verifier.calls) == 1
    assert (
        bridge.route_production(input_identity_sha256=_digest(101)).plane is RoutePlane.PRODUCTION
    )


def test_canary_requires_exact_primary_control_and_independent_endpoint() -> None:
    canonical = _canonical_primary()
    primary = canonical_route_to_inference(canonical).deployment
    mismatched_control = primary.model_copy(update={"adapter_version": "different"})
    candidate = _inference_deployment(
        deployment_id="mage-canary",
        endpoint=60,
        name="Mage-VL-4B",
    )
    canary = CanaryRoute(
        route_id="mage-canary-route",
        policy_version="1.0",
        candidate_ratio=0.25,
        control=mismatched_control,
        candidate=candidate,
        authorization=_authorization(),
    )
    with pytest.raises(ProductionTrafficError, match="exactly match"):
        ProductionTrafficBridge(
            canonical_primary_route=canonical,
            canary_route=canary,
            canary_authorization_verifier=_Verifier(),
        )

    same_endpoint_candidate = candidate.model_copy(
        update={"endpoint_config_digest": primary.endpoint_config_digest}
    )
    same_endpoint_canary = canary.model_copy(
        update={"control": primary, "candidate": same_endpoint_candidate}
    )
    with pytest.raises(ProductionTrafficError, match="independent endpoint"):
        ProductionTrafficBridge(
            canonical_primary_route=canonical,
            canary_route=same_endpoint_canary,
            canary_authorization_verifier=_Verifier(),
        )


def test_canary_verification_fails_closed() -> None:
    canonical = _canonical_primary()
    primary = canonical_route_to_inference(canonical).deployment
    candidate = _inference_deployment(
        deployment_id="mage-canary",
        endpoint=60,
        name="Mage-VL-4B",
    )
    canary = CanaryRoute(
        route_id="mage-canary-route",
        policy_version="1.0",
        candidate_ratio=0.25,
        control=primary,
        candidate=candidate,
        authorization=_authorization(),
    )

    with pytest.raises(ProductionTrafficError, match="external authorization"):
        ProductionTrafficBridge(canonical_primary_route=canonical, canary_route=canary)
    with pytest.raises(ProductionTrafficError, match="authorization"):
        ProductionTrafficBridge(
            canonical_primary_route=canonical,
            canary_route=canary,
            canary_authorization_verifier=_Verifier(False),
        )


def test_readiness_is_not_ready_without_primary_release_evidence() -> None:
    bridge = ProductionTrafficBridge(canonical_primary_route=_canonical_primary())

    assert bridge.readiness.state == "NOT_READY"
    assert "PRIMARY_ROUTE_RELEASE_UNVERIFIED" in bridge.readiness.blockers
    with pytest.raises(ProductionTrafficError, match="PRIMARY_ROUTE_RELEASE_UNVERIFIED"):
        bridge.require_ready()
    with pytest.raises(ProductionTrafficError, match="PRIMARY_ROUTE_RELEASE_UNVERIFIED"):
        bridge.decide_ready(input_identity_sha256=_digest(101))


def test_primary_release_verifier_is_optional_but_fail_closed_when_supplied() -> None:
    canonical = _canonical_primary()

    class _PrimaryVerifier:
        def verify_primary(self, *, authorization: object, deployment: object) -> bool:
            assert authorization == canonical.authorization
            assert deployment == canonical.deployment
            return False

    with pytest.raises(ProductionTrafficError, match="release was not verified"):
        ProductionTrafficBridge(
            canonical_primary_route=canonical,
            primary_release_verifier=_PrimaryVerifier(),
        )
