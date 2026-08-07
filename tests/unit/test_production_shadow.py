from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from robata.inference.experiment_execution import ExperimentExecutionError
from robata.inference.models import InferenceStatus
from robata.inference.production_shadow import (
    InMemoryProductionShadowBudget,
    ProductionShadowCoordinator,
    ProductionShadowError,
    ProductionShadowStatus,
)
from robata.inference.routing import RouteMode
from tests.unit.test_experiment_execution import _fixture, _uuid


class _BlockingExecutor:
    def __init__(
        self,
        *,
        candidate_status: InferenceStatus = InferenceStatus.SUCCEEDED,
        error: Exception | None = None,
    ) -> None:
        self.release = asyncio.Event()
        self.calls = []
        self._candidate_status = candidate_status
        self._error = error

    async def execute(self, *, route, decision, invocation):
        del route, decision
        self.calls.append(invocation)
        await self.release.wait()
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            comparison=SimpleNamespace(comparison_id=_uuid(900), candidate=None),
            candidate_terminal=SimpleNamespace(
                inference_id=_uuid(901),
                status=self._candidate_status,
                output_valid=self._candidate_status is InferenceStatus.SUCCEEDED,
            ),
        )


class _MalformedExecutor:
    async def execute(self, *, route, decision, invocation):
        del route, decision, invocation
        return object()


class _TaskCreationFailureLoop:
    def create_task(self, coroutine):
        coroutine.close()
        raise RuntimeError("task creation failed")


def test_production_shadow_releases_budget_when_task_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(mode=RouteMode.SHADOW)
    budget = InMemoryProductionShadowBudget(maximum_in_flight=1)
    coordinator = ProductionShadowCoordinator(
        route=fixture.route,
        executor=fixture.coordinator,
        budget=budget,
    )
    monkeypatch.setattr(
        "robata.inference.production_shadow.asyncio.get_running_loop",
        lambda: _TaskCreationFailureLoop(),
    )

    observation = coordinator.submit(
        primary_inference_id=_uuid(905),
        invocation=fixture.invocation,
    )

    assert observation.status is ProductionShadowStatus.FAILED
    assert observation.error_type == "RuntimeError"
    assert budget.in_flight == 0
    assert coordinator.pending_count == 0


def test_production_shadow_records_malformed_executor_result_as_failure() -> None:
    async def scenario() -> None:
        fixture = _fixture(mode=RouteMode.SHADOW)
        budget = InMemoryProductionShadowBudget(maximum_in_flight=1)
        coordinator = ProductionShadowCoordinator(
            route=fixture.route,
            executor=_MalformedExecutor(),
            budget=budget,
        )

        coordinator.submit(primary_inference_id=_uuid(906), invocation=fixture.invocation)
        await coordinator.drain()

        observation = coordinator.observations[0]
        assert observation.status is ProductionShadowStatus.FAILED
        assert observation.error_type == "AttributeError"
        assert budget.in_flight == 0
        assert coordinator.pending_count == 0

    asyncio.run(scenario())


class _RaisingBudget:
    def __init__(self) -> None:
        self.release_calls = 0

    def try_reserve(self) -> bool:
        raise RuntimeError("budget unavailable")

    def release(self) -> None:
        self.release_calls += 1


class _TerminalValidationErrorExecutor:
    async def execute(self, *, route, decision, invocation):
        del route, decision, invocation
        return SimpleNamespace(
            comparison=SimpleNamespace(
                comparison_id=_uuid(907),
                candidate=SimpleNamespace(
                    terminal_validation_error="primary_inference_id does not match dispatch"
                ),
            ),
            candidate_terminal=SimpleNamespace(
                inference_id=_uuid(908),
                status=InferenceStatus.SUCCEEDED,
                output_valid=True,
            ),
        )


def test_production_shadow_records_budget_port_failure_without_propagating() -> None:
    async def scenario() -> None:
        fixture = _fixture(mode=RouteMode.SHADOW)
        budget = _RaisingBudget()
        coordinator = ProductionShadowCoordinator(
            route=fixture.route,
            executor=fixture.coordinator,
            budget=budget,
        )

        observation = coordinator.submit(
            primary_inference_id=_uuid(909),
            invocation=fixture.invocation,
        )

        assert observation.status is ProductionShadowStatus.FAILED
        assert observation.error_type == "RuntimeError"
        assert observation.error_detail == "budget unavailable"
        assert budget.release_calls == 0
        assert coordinator.pending_count == 0

    asyncio.run(scenario())


def test_production_shadow_marks_p17_terminal_validation_error_as_failure() -> None:
    async def scenario() -> None:
        fixture = _fixture(mode=RouteMode.SHADOW)
        budget = InMemoryProductionShadowBudget(maximum_in_flight=1)
        coordinator = ProductionShadowCoordinator(
            route=fixture.route,
            executor=_TerminalValidationErrorExecutor(),
            budget=budget,
        )

        coordinator.submit(primary_inference_id=_uuid(907), invocation=fixture.invocation)
        await coordinator.drain()

        observation = coordinator.observations[0]
        assert observation.status is ProductionShadowStatus.FAILED
        assert observation.error_type == "ExperimentTerminalValidationError"
        assert observation.error_detail == "primary_inference_id does not match dispatch"
        assert budget.in_flight == 0

    asyncio.run(scenario())


def test_production_shadow_is_nonblocking_bounded_and_deduplicated() -> None:
    async def scenario() -> None:
        fixture = _fixture(mode=RouteMode.SHADOW)
        invocation = fixture.invocation
        executor = _BlockingExecutor()
        budget = InMemoryProductionShadowBudget(maximum_in_flight=1)
        coordinator = ProductionShadowCoordinator(
            route=fixture.route,
            executor=executor,
            budget=budget,
        )

        queued = coordinator.submit(
            primary_inference_id=_uuid(910),
            invocation=invocation,
        )
        duplicate = coordinator.submit(
            primary_inference_id=_uuid(910),
            invocation=invocation,
        )
        assert queued.status is ProductionShadowStatus.QUEUED
        assert duplicate == queued
        assert coordinator.pending_count == 1
        assert budget.in_flight == 1
        assert executor.calls == []

        await asyncio.sleep(0)
        assert len(executor.calls) == 1
        assert executor.calls[0].primary_inference_id == _uuid(910)
        assert executor.calls[0].control is None

        skipped = coordinator.submit(
            primary_inference_id=_uuid(911),
            invocation=invocation,
        )
        assert skipped.status is ProductionShadowStatus.SKIPPED_BUDGET
        assert coordinator.pending_count == 1

        executor.release.set()
        await coordinator.drain()
        completed = coordinator.observations[0]
        assert completed.status is ProductionShadowStatus.SUCCEEDED
        assert completed.candidate_inference_id == _uuid(901)
        assert budget.in_flight == 0
        assert coordinator.pending_count == 0

    asyncio.run(scenario())


def test_production_shadow_records_candidate_cancellation_and_executor_failure() -> None:
    async def scenario() -> None:
        fixture = _fixture(mode=RouteMode.SHADOW)
        invocation = fixture.invocation

        cancelled_executor = _BlockingExecutor(candidate_status=InferenceStatus.CANCELLED)
        cancelled_budget = InMemoryProductionShadowBudget(maximum_in_flight=1)
        cancelled = ProductionShadowCoordinator(
            route=fixture.route,
            executor=cancelled_executor,
            budget=cancelled_budget,
        )
        cancelled.submit(primary_inference_id=_uuid(920), invocation=invocation)
        cancelled_executor.release.set()
        await cancelled.drain()
        assert cancelled.observations[0].status is ProductionShadowStatus.CANCELLED
        assert cancelled_budget.in_flight == 0

        failed_executor = _BlockingExecutor(error=RuntimeError("candidate transport failed"))
        failed_budget = InMemoryProductionShadowBudget(maximum_in_flight=1)
        failed = ProductionShadowCoordinator(
            route=fixture.route,
            executor=failed_executor,
            budget=failed_budget,
        )
        failed.submit(primary_inference_id=_uuid(921), invocation=invocation)
        failed_executor.release.set()
        await failed.drain()
        observation = failed.observations[0]
        assert observation.status is ProductionShadowStatus.FAILED
        assert observation.error_type == "RuntimeError"
        assert failed_budget.in_flight == 0

    asyncio.run(scenario())


def test_production_shadow_preserves_primary_lineage_and_canonical_isolation() -> None:
    async def scenario() -> None:
        fixture = _fixture(mode=RouteMode.SHADOW)
        invocation = fixture.invocation
        coordinator = ProductionShadowCoordinator(
            route=fixture.route,
            executor=fixture.coordinator,
            budget=InMemoryProductionShadowBudget(maximum_in_flight=1),
        )

        queued = coordinator.submit(
            primary_inference_id=_uuid(930),
            invocation=invocation,
        )
        assert queued.status is ProductionShadowStatus.QUEUED
        await coordinator.drain()

        terminal = fixture.candidate_ledger.list_terminals()[0]
        assert terminal.shadow is True
        assert terminal.primary_inference_id == _uuid(930)
        assert fixture.candidate_ledger.list_selections() == ()
        assert fixture.control_ledger.list_terminals() == ()
        assert fixture.control_ledger.list_selections() == ()
        assert coordinator.observations[0].comparison_id is not None

    asyncio.run(scenario())


def test_production_shadow_observes_p17_candidate_cancellation() -> None:
    async def scenario() -> None:
        fixture = _fixture(candidate_mode="cancelled", mode=RouteMode.SHADOW)
        budget = InMemoryProductionShadowBudget(maximum_in_flight=1)
        coordinator = ProductionShadowCoordinator(
            route=fixture.route,
            executor=fixture.coordinator,
            budget=budget,
        )

        coordinator.submit(primary_inference_id=_uuid(935), invocation=fixture.invocation)
        await coordinator.drain()

        assert coordinator.observations[0].status is ProductionShadowStatus.CANCELLED
        assert fixture.candidate_ledger.list_terminals()[0].status is InferenceStatus.CANCELLED
        assert fixture.candidate_ledger.list_selections() == ()
        assert budget.in_flight == 0

    asyncio.run(scenario())


def test_primary_lineage_is_rejected_for_paired_execution() -> None:
    fixture = _fixture(mode=RouteMode.PAIRED)
    invocation = replace(fixture.invocation, primary_inference_id=_uuid(940))

    with pytest.raises(ExperimentExecutionError, match="only valid for SHADOW"):
        asyncio.run(
            fixture.coordinator.execute(
                route=fixture.route,
                decision=fixture.decision,
                invocation=invocation,
            )
        )

    assert fixture.control_adapter.infer_calls == 0
    assert fixture.candidate_adapter.infer_calls == 0


def test_production_shadow_rejects_paired_routes_and_prepopulated_lineage() -> None:
    paired_fixture = _fixture(mode=RouteMode.PAIRED)
    with pytest.raises(ProductionShadowError, match="requires a SHADOW"):
        ProductionShadowCoordinator(
            route=paired_fixture.route,
            executor=paired_fixture.coordinator,
            budget=InMemoryProductionShadowBudget(maximum_in_flight=1),
        )

    fixture = _fixture(mode=RouteMode.SHADOW)
    coordinator = ProductionShadowCoordinator(
        route=fixture.route,
        executor=fixture.coordinator,
        budget=InMemoryProductionShadowBudget(maximum_in_flight=1),
    )
    with pytest.raises(ProductionShadowError, match="must not prepopulate"):
        coordinator.submit(
            primary_inference_id=_uuid(941),
            invocation=replace(
                fixture.invocation,
                primary_inference_id=_uuid(942),
            ),
        )


def test_selected_shadow_requires_an_event_loop_before_reserving_budget() -> None:
    fixture = _fixture(mode=RouteMode.SHADOW)
    budget = InMemoryProductionShadowBudget(maximum_in_flight=1)
    coordinator = ProductionShadowCoordinator(
        route=fixture.route,
        executor=fixture.coordinator,
        budget=budget,
    )

    with pytest.raises(ProductionShadowError, match="running event loop"):
        coordinator.submit(primary_inference_id=_uuid(943), invocation=fixture.invocation)

    assert budget.in_flight == 0
