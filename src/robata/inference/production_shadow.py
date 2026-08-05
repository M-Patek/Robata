"""Nonblocking bridge from completed production work to P17 shadow execution.

This is intentionally an in-process composition helper rather than a durable
scheduler or a serving policy.  It never awaits candidate work from the
production caller and delegates every candidate invocation to the existing
experiment coordinator, which writes it as ``shadow=True`` and therefore
cannot create a canonical selection.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from robata.contracts.logical_nodes import OpaqueUuid
from robata.inference.experiment_execution import (
    ExperimentExecutionResult,
    ExperimentInvocation,
)
from robata.inference.models import InferenceStatus, ModelInference
from robata.inference.routing import ExperimentRoute, ModelRouteDecision, RouteMode

_OPAQUE_UUID_ADAPTER = TypeAdapter(OpaqueUuid)


class ProductionShadowError(ValueError):
    """Raised when a local production-shadow composition is invalid."""


class ProductionShadowStatus(StrEnum):
    """Local observation state; this is not a published rollout state machine."""

    NOT_SELECTED = "NOT_SELECTED"
    SKIPPED_BUDGET = "SKIPPED_BUDGET"
    QUEUED = "QUEUED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProductionShadowBudget(Protocol):
    """Small bounded-concurrency port for observation-only candidate work."""

    def try_reserve(self) -> bool:
        """Reserve one candidate slot without waiting."""

    def release(self) -> None:
        """Release a previously reserved candidate slot."""


class ProductionShadowExecutor(Protocol):
    """The P17 execution port consumed by this nonblocking bridge."""

    async def execute(
        self,
        *,
        route: ExperimentRoute,
        decision: ModelRouteDecision,
        invocation: ExperimentInvocation,
    ) -> ExperimentExecutionResult:
        """Run one non-authoritative experiment decision."""


class InMemoryProductionShadowBudget:
    """A deliberately small, event-loop-local in-flight budget."""

    def __init__(self, *, maximum_in_flight: int) -> None:
        if (
            isinstance(maximum_in_flight, bool)
            or not isinstance(maximum_in_flight, int)
            or maximum_in_flight < 1
        ):
            raise ProductionShadowError("maximum_in_flight must be a positive integer")
        self._maximum_in_flight = maximum_in_flight
        self._in_flight = 0

    @property
    def maximum_in_flight(self) -> int:
        """Return the fixed local candidate limit."""

        return self._maximum_in_flight

    @property
    def in_flight(self) -> int:
        """Return the currently reserved candidate slots."""

        return self._in_flight

    def try_reserve(self) -> bool:
        """Reserve one slot if capacity remains, without queueing."""

        if self._in_flight >= self._maximum_in_flight:
            return False
        self._in_flight += 1
        return True

    def release(self) -> None:
        """Release one slot and detect bridge accounting misuse."""

        if self._in_flight < 1:
            raise ProductionShadowError("production shadow budget release underflow")
        self._in_flight -= 1


@dataclass(frozen=True, slots=True)
class ProductionShadowObservation:
    """A local lifecycle observation linked to one committed primary attempt."""

    route_id: str
    experiment_id: str
    primary_inference_id: str
    input_identity_sha256: str
    status: ProductionShadowStatus
    comparison_id: str | None = None
    candidate_inference_id: str | None = None
    candidate_status: InferenceStatus | None = None
    error_type: str | None = None
    error_detail: str | None = None


class ProductionShadowCoordinator:
    """Start selected candidate shadow work without delaying production callers.

    This object owns only in-process task tracking.  P17 owns the candidate
    evidence and the candidate terminal remains selection-ineligible.  A
    durable restart/recovery queue, rollout policy, and serving rollback are
    intentionally outside this helper.
    """

    def __init__(
        self,
        *,
        route: ExperimentRoute,
        executor: ProductionShadowExecutor,
        budget: ProductionShadowBudget,
    ) -> None:
        if not isinstance(route, ExperimentRoute):
            raise TypeError("route must be ExperimentRoute")
        try:
            checked_route = ExperimentRoute.model_validate(route.model_dump())
        except ValueError as exc:
            raise ProductionShadowError("production shadow route is invalid") from exc
        if checked_route.mode is not RouteMode.SHADOW:
            raise ProductionShadowError("production shadow requires a SHADOW experiment route")
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("executor must define execute")
        if not callable(getattr(budget, "try_reserve", None)) or not callable(
            getattr(budget, "release", None)
        ):
            raise TypeError("budget must define try_reserve and release")
        self._route = checked_route
        self._executor = executor
        self._budget = budget
        self._observations: dict[tuple[str, str], ProductionShadowObservation] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def route(self) -> ExperimentRoute:
        """Return the frozen shadow route used by this bridge."""

        return self._route

    @property
    def observations(self) -> tuple[ProductionShadowObservation, ...]:
        """Return observations in submission order."""

        return tuple(self._observations.values())

    @property
    def pending_count(self) -> int:
        """Return the number of candidate tasks still owned by this bridge."""

        return len(self._tasks)

    def submit(
        self,
        *,
        primary_inference_id: str,
        invocation: ExperimentInvocation,
    ) -> ProductionShadowObservation:
        """Record or start a candidate task without awaiting its completion."""

        primary_id = self._validate_primary_inference_id(primary_inference_id)
        if not isinstance(invocation, ExperimentInvocation):
            raise TypeError("invocation must be ExperimentInvocation")
        if invocation.primary_inference_id is not None:
            raise ProductionShadowError(
                "production shadow invocation must not prepopulate primary_inference_id"
            )
        key = (primary_id, invocation.input_identity_sha256)
        existing = self._observations.get(key)
        if existing is not None:
            return existing

        decision = self._route.decide(input_identity_sha256=invocation.input_identity_sha256)
        if not decision.dispatches:
            return self._record(
                key,
                self._observation(
                    primary_inference_id=primary_id,
                    input_identity_sha256=invocation.input_identity_sha256,
                    status=ProductionShadowStatus.NOT_SELECTED,
                ),
            )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise ProductionShadowError(
                "production shadow submit requires a running event loop"
            ) from exc

        try:
            reserved = self._budget.try_reserve()
        except Exception as exc:
            return self._record(
                key,
                self._observation(
                    primary_inference_id=primary_id,
                    input_identity_sha256=invocation.input_identity_sha256,
                    status=ProductionShadowStatus.FAILED,
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                ),
            )
        if not reserved:
            return self._record(
                key,
                self._observation(
                    primary_inference_id=primary_id,
                    input_identity_sha256=invocation.input_identity_sha256,
                    status=ProductionShadowStatus.SKIPPED_BUDGET,
                ),
            )

        queued = self._record(
            key,
            self._observation(
                primary_inference_id=primary_id,
                input_identity_sha256=invocation.input_identity_sha256,
                status=ProductionShadowStatus.QUEUED,
            ),
        )
        try:
            task = loop.create_task(
                self._execute(
                    key=key,
                    decision=decision,
                    invocation=replace(
                        invocation,
                        control=None,
                        primary_inference_id=primary_id,
                    ),
                )
            )
        except Exception as exc:
            self._budget.release()
            return self._record(
                key,
                self._observation(
                    primary_inference_id=primary_id,
                    input_identity_sha256=invocation.input_identity_sha256,
                    status=ProductionShadowStatus.FAILED,
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                ),
            )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return queued

    async def drain(self) -> None:
        """Wait for bridge-owned candidate tasks during controlled shutdown/tests."""

        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _execute(
        self,
        *,
        key: tuple[str, str],
        decision: ModelRouteDecision,
        invocation: ExperimentInvocation,
    ) -> None:
        try:
            result = await self._executor.execute(
                route=self._route,
                decision=decision,
                invocation=invocation,
            )
            terminal = result.candidate_terminal
            observation = self._result_observation(
                invocation=invocation,
                result=result,
                terminal=terminal,
            )
        except asyncio.CancelledError:
            self._record(
                key,
                self._observation(
                    primary_inference_id=invocation.primary_inference_id,
                    input_identity_sha256=invocation.input_identity_sha256,
                    status=ProductionShadowStatus.CANCELLED,
                ),
            )
            raise
        except Exception as exc:
            self._record(
                key,
                self._observation(
                    primary_inference_id=invocation.primary_inference_id,
                    input_identity_sha256=invocation.input_identity_sha256,
                    status=ProductionShadowStatus.FAILED,
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                ),
            )
        else:
            self._record(key, observation)
        finally:
            self._budget.release()

    def _observation(
        self,
        *,
        primary_inference_id: str | None,
        input_identity_sha256: str,
        status: ProductionShadowStatus,
        comparison_id: str | None = None,
        candidate_inference_id: str | None = None,
        candidate_status: InferenceStatus | None = None,
        error_type: str | None = None,
        error_detail: str | None = None,
    ) -> ProductionShadowObservation:
        assert primary_inference_id is not None
        return ProductionShadowObservation(
            route_id=self._route.route_id,
            experiment_id=self._route.contract.experiment_id,
            primary_inference_id=primary_inference_id,
            input_identity_sha256=input_identity_sha256,
            status=status,
            comparison_id=comparison_id,
            candidate_inference_id=candidate_inference_id,
            candidate_status=candidate_status,
            error_type=error_type,
            error_detail=error_detail,
        )

    def _result_observation(
        self,
        *,
        invocation: ExperimentInvocation,
        result: ExperimentExecutionResult,
        terminal: ModelInference | None,
    ) -> ProductionShadowObservation:
        candidate = result.comparison.candidate
        terminal_validation_error = (
            candidate.terminal_validation_error if candidate is not None else None
        )
        candidate_status = (
            terminal.status
            if terminal is not None
            else candidate.status
            if candidate is not None
            else None
        )
        status = ProductionShadowStatus.SUCCEEDED
        if terminal_validation_error is not None:
            status = ProductionShadowStatus.FAILED
        elif candidate_status is InferenceStatus.CANCELLED or (
            candidate is not None and candidate.execution_error_type == "CancelledError"
        ):
            status = ProductionShadowStatus.CANCELLED
        elif (
            terminal is None
            or terminal.status is not InferenceStatus.SUCCEEDED
            or not terminal.output_valid
        ):
            status = ProductionShadowStatus.FAILED
        return self._observation(
            primary_inference_id=invocation.primary_inference_id,
            input_identity_sha256=invocation.input_identity_sha256,
            status=status,
            comparison_id=result.comparison.comparison_id,
            candidate_inference_id=(
                terminal.inference_id
                if terminal is not None
                else candidate.inference_id
                if candidate is not None
                else None
            ),
            candidate_status=candidate_status,
            error_type=(
                "ExperimentTerminalValidationError"
                if terminal_validation_error is not None
                else None
            ),
            error_detail=terminal_validation_error,
        )

    def _record(
        self,
        key: tuple[str, str],
        observation: ProductionShadowObservation,
    ) -> ProductionShadowObservation:
        self._observations[key] = observation
        return observation

    @staticmethod
    def _validate_primary_inference_id(value: str) -> str:
        try:
            return _OPAQUE_UUID_ADAPTER.validate_python(value)
        except ValidationError as exc:
            raise ProductionShadowError("primary_inference_id must be a lowercase UUID") from exc


__all__ = [
    "InMemoryProductionShadowBudget",
    "ProductionShadowBudget",
    "ProductionShadowCoordinator",
    "ProductionShadowError",
    "ProductionShadowExecutor",
    "ProductionShadowObservation",
    "ProductionShadowStatus",
]
