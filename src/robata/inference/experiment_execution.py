"""Isolated, non-authoritative execution for paired model experiments.

This module deliberately sits beside, rather than inside, the canonical
inference path.  It maps an ``ExperimentRoute`` decision to separately
constructed orchestrators and retains a local generic comparison sidecar.
Published inference schemas remain unchanged until experiment evidence needs
to become a durable product contract.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal, Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, TypeAdapter, ValidationError

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import CanonicalizationError, semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.inference.adapter import PackageInput
from robata.inference.input_plan import InferenceInputPlan
from robata.inference.models import (
    InferenceStatus,
    ModelInference,
    ModelInferenceUsage,
    VisionTask,
)
from robata.inference.orchestrator import InferenceOrchestrator
from robata.inference.routing import (
    DispatchDisposition,
    ExperimentContract,
    ExperimentInputRepresentation,
    ExperimentRoute,
    ModelDeployment,
    ModelRouteDecision,
    ModelRouteRole,
    RouteMode,
    RoutePlane,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
_OPAQUE_UUID_ADAPTER = TypeAdapter(OpaqueUuid)


class ExperimentExecutionError(ValueError):
    """Raised when an experiment cannot be safely dispatched or compared."""


class ExperimentExecutionConflictError(ExperimentExecutionError):
    """Raised when an append-only comparison is replayed with different evidence."""


class ExperimentComparisonStatus(StrEnum):
    """Outcome of a route decision and, when available, its paired comparison."""

    NOT_SELECTED = "NOT_SELECTED"
    SINGLE_OBSERVATION = "SINGLE_OBSERVATION"
    AGREEMENT = "AGREEMENT"
    DIFFERENCE = "DIFFERENCE"
    CONTROL_FAILURE = "CONTROL_FAILURE"
    CANDIDATE_FAILURE = "CANDIDATE_FAILURE"
    BOTH_FAILURE = "BOTH_FAILURE"
    NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass(frozen=True, slots=True)
class ExperimentTargetInput:
    """One deployment-specific immutable input plan and selected call part."""

    input_plan: InferenceInputPlan
    input_plan_part_ordinal: int | None = None


@dataclass(frozen=True, slots=True)
class ExperimentInvocation:
    """Shared source invocation with separate provider-targeted input plans.

    ``source_workload_manifest_sha256`` is supplied by the workload scheduler;
    it binds this execution to the immutable workload manifest named in the
    experiment contract.  The input plans retain their own request-catalog and
    rendering provenance because a plan is target-bound by design.
    """

    source_workload_manifest_sha256: str
    input_identity_sha256: str
    task: VisionTask
    package_set_id: str | None
    mcap_id: str
    camera_mapping_run_id: str
    alignment_id: str
    start_ns: int
    end_ns: int
    package_inputs: tuple[PackageInput, ...]
    control: ExperimentTargetInput | None
    candidate: ExperimentTargetInput | None
    comparison_config: Mapping[str, object]
    input_config: Mapping[str, object] = field(default_factory=dict)
    sampling_config: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)
    attempt: int = 1
    retry_count: int = 0
    primary_inference_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentDeploymentBinding:
    """One independently constructed deployment-scoped orchestrator."""

    deployment: ModelDeployment
    orchestrator: InferenceOrchestrator


class ExperimentDeploymentRegistry:
    """Resolve deployment facts to isolated experiment execution scopes.

    A registry refuses shared orchestrator or ledger objects.  An
    ``InferenceOrchestrator`` owns its microbatch queue, so separate instances
    also separate that queue.  Adapter and execution-gate identity are checked
    for the concrete task just before a paired dispatch.
    """

    def __init__(self, *, bindings: Mapping[str, ExperimentDeploymentBinding]) -> None:
        normalized: dict[str, ExperimentDeploymentBinding] = {}
        orchestrator_ids: set[int] = set()
        ledger_ids: set[int] = set()
        for deployment_id, binding in bindings.items():
            if not isinstance(deployment_id, str) or not deployment_id:
                raise ExperimentExecutionError("deployment registry keys must be nonempty strings")
            if not isinstance(binding, ExperimentDeploymentBinding):
                raise TypeError("deployment registry values must be ExperimentDeploymentBinding")
            if binding.deployment.deployment_id != deployment_id:
                raise ExperimentExecutionError(
                    "deployment registry key does not match binding deployment_id"
                )
            orchestrator_identity = id(binding.orchestrator)
            ledger_identity = id(binding.orchestrator.ledger)
            if orchestrator_identity in orchestrator_ids:
                raise ExperimentExecutionError(
                    "experiment deployments must use separate orchestrator instances"
                )
            if ledger_identity in ledger_ids:
                raise ExperimentExecutionError(
                    "experiment deployments must use separate inference ledgers"
                )
            orchestrator_ids.add(orchestrator_identity)
            ledger_ids.add(ledger_identity)
            normalized[deployment_id] = binding
        self._bindings = normalized

    @property
    def bindings(self) -> tuple[ExperimentDeploymentBinding, ...]:
        """Return the configured deployment bindings in insertion order."""

        return tuple(self._bindings.values())

    def resolve(self, deployment: ModelDeployment) -> ExperimentDeploymentBinding:
        """Return the exact binding for an immutable deployment descriptor."""

        binding = self._bindings.get(deployment.deployment_id)
        if binding is None:
            raise ExperimentExecutionError(
                f"no experiment executor is registered for {deployment.deployment_id!r}"
            )
        if binding.deployment != deployment:
            raise ExperimentExecutionError(
                "registered deployment facts differ from the experiment contract"
            )
        return binding

    def validate_pair_isolation(
        self,
        *,
        task: VisionTask,
        control: ExperimentDeploymentBinding,
        candidate: ExperimentDeploymentBinding,
    ) -> None:
        """Check the concrete task has independent resources on both sides."""

        if control.orchestrator is candidate.orchestrator:
            raise ExperimentExecutionError(
                "paired deployments must use separate orchestrator instances"
            )
        if control.orchestrator.ledger is candidate.orchestrator.ledger:
            raise ExperimentExecutionError("paired deployments must use separate ledgers")
        self._validate_policy_binding(task=task, binding=control)
        self._validate_policy_binding(task=task, binding=candidate)
        if control.orchestrator.adapter_for(task) is candidate.orchestrator.adapter_for(task):
            raise ExperimentExecutionError("paired deployments must use separate adapters")
        if control.orchestrator.execution_gate is candidate.orchestrator.execution_gate:
            raise ExperimentExecutionError("paired deployments must use separate execution gates")
        if _logical_model_identity(control.deployment) == _logical_model_identity(
            candidate.deployment
        ):
            raise ExperimentExecutionError(
                "paired deployments with identical inference identities are unsupported"
            )

    @staticmethod
    def _validate_policy_binding(*, task: VisionTask, binding: ExperimentDeploymentBinding) -> None:
        policy = binding.orchestrator.policy_for(task)
        deployment = binding.deployment
        policy_identity = (
            policy.provider,
            policy.model_name,
            policy.model_version,
            policy.adapter_version,
        )
        deployment_identity = (
            deployment.provider,
            deployment.model_name,
            deployment.model_version,
            deployment.adapter_version,
        )
        if policy_identity != deployment_identity:
            raise ExperimentExecutionError(
                "deployment facts do not match the configured inference policy"
            )


class ExperimentFieldDelta(StrictModel):
    """One generic deterministic difference between control and candidate output."""

    path: NonEmptyString
    control: object | None = None
    candidate: object | None = None
    severity: NonEmptyString


class ExperimentSideOutcome(StrictModel):
    """Minimal terminal or dispatch-error evidence retained for one model side."""

    schema_version: Literal["1.0"] = "1.0"
    role: ModelRouteRole
    deployment_id: NonEmptyString
    inference_id: OpaqueUuid | None = None
    logical_invocation_id: OpaqueUuid | None = None
    status: InferenceStatus | None = None
    output_valid: bool | None = None
    retry_count: NonNegativeInt | None = None
    latency_ms: NonNegativeInt | None = None
    usage: ModelInferenceUsage | None = None
    failure_code: NonEmptyString | None = None
    execution_error_type: NonEmptyString | None = None
    execution_error_detail: str | None = None
    terminal_validation_error: str | None = None


class ExperimentPairComparison(StrictModel):
    """Internal append-only generic comparison sidecar for one route decision."""

    schema_version: Literal["1.0"] = "1.0"
    comparison_id: OpaqueUuid
    experiment_id: NonEmptyString
    experiment_contract_digest: Sha256Digest
    workload_manifest_sha256: Sha256Digest
    route_id: NonEmptyString
    route_configuration_digest: Sha256Digest
    input_identity_sha256: Sha256Digest
    comparison_config_sha256: Sha256Digest
    input_representation: ExperimentInputRepresentation
    attempt: NonNegativeInt
    retry_count: NonNegativeInt
    status: ExperimentComparisonStatus
    comparable: bool
    control_inference_id: OpaqueUuid | None = None
    candidate_inference_id: OpaqueUuid | None = None
    control: ExperimentSideOutcome | None = None
    candidate: ExperimentSideOutcome | None = None
    field_deltas: tuple[ExperimentFieldDelta, ...] = ()


@dataclass(frozen=True, slots=True)
class ExperimentExecutionResult:
    """Comparison evidence plus the independently persisted terminal attempts."""

    comparison: ExperimentPairComparison
    control_terminal: ModelInference | None
    candidate_terminal: ModelInference | None


class ExperimentComparisonLedger(Protocol):
    """Append-only local persistence port for internal comparison sidecars."""

    def append(self, comparison: ExperimentPairComparison) -> ExperimentPairComparison:
        """Append a comparison or return the exact idempotent replay."""
        ...


class InMemoryExperimentComparisonLedger:
    """Reference append-only sidecar ledger for development execution and tests."""

    def __init__(self) -> None:
        self._comparisons: dict[str, ExperimentPairComparison] = {}

    @property
    def comparisons(self) -> tuple[ExperimentPairComparison, ...]:
        """Return comparisons in insertion order."""

        return tuple(self._comparisons.values())

    def append(self, comparison: ExperimentPairComparison) -> ExperimentPairComparison:
        existing = self._comparisons.get(comparison.comparison_id)
        if existing is not None:
            if existing != comparison:
                raise ExperimentExecutionConflictError(
                    "experiment comparison identity already has different evidence"
                )
            return existing
        self._comparisons[comparison.comparison_id] = comparison
        return comparison


@dataclass(frozen=True, slots=True)
class _ComparisonOptions:
    ignored_paths: tuple[str, ...]
    global_numeric_tolerance: float
    numeric_tolerances: Mapping[str, float]
    severity_by_path: Mapping[str, str]
    default_severity: str


@dataclass(frozen=True, slots=True)
class _DispatchResult:
    role: ModelRouteRole
    deployment: ModelDeployment
    target: ExperimentTargetInput
    terminal: ModelInference | None
    error: BaseException | None = None
    terminal_validation_error: str | None = None


class ExperimentExecutionCoordinator:
    """Execute only experiment-plane route decisions through isolated scopes."""

    def __init__(
        self,
        *,
        registry: ExperimentDeploymentRegistry,
        comparison_ledger: ExperimentComparisonLedger | None = None,
    ) -> None:
        if not isinstance(registry, ExperimentDeploymentRegistry):
            raise TypeError("registry must be ExperimentDeploymentRegistry")
        self._registry = registry
        self._comparison_ledger = comparison_ledger or InMemoryExperimentComparisonLedger()

    @property
    def comparison_ledger(self) -> ExperimentComparisonLedger:
        """Return the local append-only comparison sidecar."""

        return self._comparison_ledger

    async def execute(
        self,
        *,
        route: ExperimentRoute,
        decision: ModelRouteDecision,
        invocation: ExperimentInvocation,
    ) -> ExperimentExecutionResult:
        """Run a sampled experiment decision without any canonical selection path."""

        self._validate_route_decision(route=route, decision=decision)
        config_digest, options = self._validate_invocation(
            route=route,
            decision=decision,
            invocation=invocation,
        )

        if not decision.dispatches:
            comparison = self._append_comparison(
                route=route,
                decision=decision,
                invocation=invocation,
                comparison_config_sha256=config_digest,
                status=ExperimentComparisonStatus.NOT_SELECTED,
                comparable=False,
                control=None,
                candidate=None,
                field_deltas=(),
            )
            return ExperimentExecutionResult(
                comparison=comparison,
                control_terminal=None,
                candidate_terminal=None,
            )

        dispatch_by_role = {dispatch.role: dispatch for dispatch in decision.dispatches}
        control_binding: ExperimentDeploymentBinding | None = None
        candidate_binding: ExperimentDeploymentBinding | None = None
        control_target: ExperimentTargetInput | None = None
        candidate_target: ExperimentTargetInput | None = None

        if ModelRouteRole.CONTROL in dispatch_by_role:
            control_binding = self._registry.resolve(route.contract.control)
            control_target = self._require_target_input(
                invocation.control,
                role=ModelRouteRole.CONTROL,
            )
            self._validate_target_input(
                contract=route.contract,
                invocation=invocation,
                deployment=control_binding.deployment,
                target=control_target,
            )
        if ModelRouteRole.CANDIDATE in dispatch_by_role:
            candidate_binding = self._registry.resolve(route.contract.candidate)
            candidate_target = self._require_target_input(
                invocation.candidate,
                role=ModelRouteRole.CANDIDATE,
            )
            self._validate_target_input(
                contract=route.contract,
                invocation=invocation,
                deployment=candidate_binding.deployment,
                target=candidate_target,
            )

        if route.mode is RouteMode.PAIRED:
            assert control_binding is not None and control_target is not None
            assert candidate_binding is not None and candidate_target is not None
            self._registry.validate_pair_isolation(
                task=invocation.task,
                control=control_binding,
                candidate=candidate_binding,
            )
            self._validate_paired_inputs(
                representation=route.contract.input_representation,
                control=control_target,
                candidate=candidate_target,
            )

        coroutines: list[Awaitable[_DispatchResult]] = []
        ordered_roles: list[ModelRouteRole] = []
        if control_binding is not None and control_target is not None:
            ordered_roles.append(ModelRouteRole.CONTROL)
            coroutines.append(
                self._dispatch(
                    route=route,
                    invocation=invocation,
                    binding=control_binding,
                    role=ModelRouteRole.CONTROL,
                    target=control_target,
                )
            )
        if candidate_binding is not None and candidate_target is not None:
            ordered_roles.append(ModelRouteRole.CANDIDATE)
            coroutines.append(
                self._dispatch(
                    route=route,
                    invocation=invocation,
                    binding=candidate_binding,
                    role=ModelRouteRole.CANDIDATE,
                    target=candidate_target,
                )
            )
        settled = await asyncio.gather(*coroutines, return_exceptions=True)
        outcomes: dict[ModelRouteRole, _DispatchResult] = {}
        for role, result in zip(ordered_roles, settled, strict=True):
            if isinstance(result, _DispatchResult):
                outcomes[role] = result
                continue
            binding = control_binding if role is ModelRouteRole.CONTROL else candidate_binding
            target = control_target if role is ModelRouteRole.CONTROL else candidate_target
            assert binding is not None and target is not None
            error = result if isinstance(result, BaseException) else RuntimeError(str(result))
            outcomes[role] = _DispatchResult(
                role=role,
                deployment=binding.deployment,
                target=target,
                terminal=None,
                error=error,
            )

        control_result = outcomes.get(ModelRouteRole.CONTROL)
        candidate_result = outcomes.get(ModelRouteRole.CANDIDATE)
        control_outcome = self._side_outcome(control_result)
        candidate_outcome = self._side_outcome(candidate_result)
        status, comparable, deltas = self._compare(
            representation=route.contract.input_representation,
            mode=route.mode,
            control=control_result,
            candidate=candidate_result,
            options=options,
        )
        comparison = self._append_comparison(
            route=route,
            decision=decision,
            invocation=invocation,
            comparison_config_sha256=config_digest,
            status=status,
            comparable=comparable,
            control=control_outcome,
            candidate=candidate_outcome,
            field_deltas=deltas,
        )
        return ExperimentExecutionResult(
            comparison=comparison,
            control_terminal=control_result.terminal if control_result is not None else None,
            candidate_terminal=(
                candidate_result.terminal if candidate_result is not None else None
            ),
        )

    @staticmethod
    def _validate_route_decision(*, route: ExperimentRoute, decision: ModelRouteDecision) -> None:
        if not isinstance(route, ExperimentRoute):
            raise TypeError("route must be ExperimentRoute")
        if not isinstance(decision, ModelRouteDecision):
            raise TypeError("decision must be ModelRouteDecision")
        if decision.plane is not RoutePlane.EXPERIMENT:
            raise ExperimentExecutionError(
                "experiment coordinator only accepts experiment decisions"
            )
        if (
            decision.route_id != route.route_id
            or decision.route_configuration_digest != route.configuration_digest
            or decision.mode is not route.mode
            or decision.experiment_id != route.contract.experiment_id
        ):
            raise ExperimentExecutionError(
                "route decision does not match the configured experiment"
            )
        if any(
            dispatch.disposition is not DispatchDisposition.OBSERVATION
            for dispatch in decision.dispatches
        ):
            raise ExperimentExecutionError("experiment dispatches must be observation-only")

        expected: tuple[tuple[ModelRouteRole, str], ...]
        if not decision.dispatches:
            expected = ()
        elif route.mode is RouteMode.PAIRED:
            expected = (
                (ModelRouteRole.CONTROL, route.contract.control.deployment_id),
                (ModelRouteRole.CANDIDATE, route.contract.candidate.deployment_id),
            )
        else:
            expected = ((ModelRouteRole.CANDIDATE, route.contract.candidate.deployment_id),)
        actual = tuple((dispatch.role, dispatch.deployment_id) for dispatch in decision.dispatches)
        if actual != expected:
            raise ExperimentExecutionError(
                "route decision dispatches do not match the experiment mode"
            )

    @staticmethod
    def _validate_invocation(
        *,
        route: ExperimentRoute,
        decision: ModelRouteDecision,
        invocation: ExperimentInvocation,
    ) -> tuple[Sha256Digest, _ComparisonOptions]:
        if not isinstance(invocation, ExperimentInvocation):
            raise TypeError("invocation must be ExperimentInvocation")
        contract = route.contract
        if invocation.source_workload_manifest_sha256 != contract.workload_manifest_sha256:
            raise ExperimentExecutionError(
                "invocation workload manifest does not match the contract"
            )
        if invocation.input_identity_sha256 != decision.input_identity_sha256:
            raise ExperimentExecutionError("invocation identity does not match the route decision")
        if (
            isinstance(invocation.start_ns, bool)
            or isinstance(invocation.end_ns, bool)
            or not isinstance(invocation.start_ns, int)
            or not isinstance(invocation.end_ns, int)
            or invocation.start_ns >= invocation.end_ns
        ):
            raise ExperimentExecutionError("experiment interval must be nonempty")
        if (
            isinstance(invocation.attempt, bool)
            or not isinstance(invocation.attempt, int)
            or invocation.attempt < 1
            or isinstance(invocation.retry_count, bool)
            or not isinstance(invocation.retry_count, int)
            or invocation.retry_count < 0
            or invocation.retry_count >= invocation.attempt
        ):
            raise ExperimentExecutionError("experiment retry values are invalid")
        if invocation.primary_inference_id is not None:
            if route.mode is not RouteMode.SHADOW:
                raise ExperimentExecutionError(
                    "primary inference linkage is only valid for SHADOW routes"
                )
            try:
                _OPAQUE_UUID_ADAPTER.validate_python(invocation.primary_inference_id)
            except ValidationError as exc:
                raise ExperimentExecutionError(
                    "primary_inference_id must be a lowercase UUID"
                ) from exc
        if "logical_dependency_sha256" in invocation.input_config:
            raise ExperimentExecutionError(
                "experiment logical dependency is derived from the frozen contract"
            )
        try:
            config = dict(invocation.comparison_config)
            config_digest = semantic_sha256(config)
        except (CanonicalizationError, TypeError, ValueError) as exc:
            raise ExperimentExecutionError(
                "comparison configuration is not canonical JSON"
            ) from exc
        if config_digest != contract.comparison_config_sha256:
            raise ExperimentExecutionError("comparison configuration does not match the contract")
        return config_digest, _comparison_options(config)

    @staticmethod
    def _require_target_input(
        target: ExperimentTargetInput | None, *, role: ModelRouteRole
    ) -> ExperimentTargetInput:
        if target is None:
            raise ExperimentExecutionError(f"{role.value.lower()} input plan is required")
        if not isinstance(target, ExperimentTargetInput):
            raise TypeError(f"{role.value.lower()} input must be ExperimentTargetInput")
        return target

    @staticmethod
    def _validate_target_input(
        *,
        contract: ExperimentContract,
        invocation: ExperimentInvocation,
        deployment: ModelDeployment,
        target: ExperimentTargetInput,
    ) -> None:
        plan = target.input_plan
        if (
            plan.subject.task is not invocation.task
            or plan.request_catalog.task is not invocation.task
        ):
            raise ExperimentExecutionError(
                "input plan task does not match the experiment invocation"
            )
        if target.input_plan_part_ordinal is not None and (
            isinstance(target.input_plan_part_ordinal, bool)
            or not isinstance(target.input_plan_part_ordinal, int)
            or target.input_plan_part_ordinal < 0
            or target.input_plan_part_ordinal >= len(plan.call_plan.parts)
        ):
            raise ExperimentExecutionError("input plan part ordinal is outside the call plan")
        target_identity = (
            plan.target.provider,
            plan.target.model_name,
            plan.target.model_version,
            plan.target.adapter_version,
            plan.target.capability_snapshot_id,
            plan.target.capability_snapshot_sha256,
        )
        deployment_identity = (
            deployment.provider,
            deployment.model_name,
            deployment.model_version,
            deployment.adapter_version,
            deployment.capability_snapshot_id,
            deployment.capability_snapshot_digest,
        )
        if target_identity != deployment_identity:
            raise ExperimentExecutionError("input plan target does not match its deployment")
        if plan.subject.request_catalog_sha256 != plan.request_catalog.semantic_sha256:
            raise ExperimentExecutionError("input plan source catalog binding is inconsistent")
        expected_packages = tuple(
            (
                item.package_id,
                item.ordinal,
                item.semantic_content_sha256,
                item.manifest_bytes_sha256,
            )
            for item in plan.subject.packages
        )
        actual_packages = tuple(
            (
                item.package_id,
                item.ordinal,
                item.package_semantic_content_sha256,
                item.package_manifest_sha256,
            )
            for item in invocation.package_inputs
        )
        if actual_packages != expected_packages:
            raise ExperimentExecutionError("input plan packages do not match the shared source")

    @staticmethod
    def _validate_paired_inputs(
        *,
        representation: ExperimentInputRepresentation,
        control: ExperimentTargetInput,
        candidate: ExperimentTargetInput,
    ) -> None:
        control_plan = control.input_plan
        candidate_plan = candidate.input_plan
        if (
            control_plan.request_catalog.semantic_sha256
            != candidate_plan.request_catalog.semantic_sha256
            or control_plan.subject != candidate_plan.subject
        ):
            raise ExperimentExecutionError(
                "paired input plans do not share the same source catalog"
            )
        if representation is ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING:
            control_prompt = control_plan.prompt_output
            candidate_prompt = candidate_plan.prompt_output
            if (
                _selected_rendered_digest(control) != _selected_rendered_digest(candidate)
                or control_prompt.prompt_version != candidate_prompt.prompt_version
                or control_prompt.prompt_sha256 != candidate_prompt.prompt_sha256
                or (
                    control_prompt.provider_response_schema_sha256
                    != candidate_prompt.provider_response_schema_sha256
                )
            ):
                raise ExperimentExecutionError(
                    "identical-frame comparison requires identical rendering, prompt, and schema"
                )

    async def _dispatch(
        self,
        *,
        route: ExperimentRoute,
        invocation: ExperimentInvocation,
        binding: ExperimentDeploymentBinding,
        role: ModelRouteRole,
        target: ExperimentTargetInput,
    ) -> _DispatchResult:
        try:
            terminal = await binding.orchestrator.orchestrate(
                task=invocation.task,
                package_set_id=invocation.package_set_id,
                mcap_id=invocation.mcap_id,
                camera_mapping_run_id=invocation.camera_mapping_run_id,
                alignment_id=invocation.alignment_id,
                start_ns=invocation.start_ns,
                end_ns=invocation.end_ns,
                package_inputs=invocation.package_inputs,
                input_plan=target.input_plan,
                input_plan_part_ordinal=target.input_plan_part_ordinal,
                input_config=dict(invocation.input_config),
                logical_dependency_sha256=route.contract.contract_digest,
                sampling_config=dict(invocation.sampling_config),
                metadata=dict(invocation.metadata),
                attempt=invocation.attempt,
                retry_count=invocation.retry_count,
                shadow=True,
                experiment_id=route.contract.experiment_id,
                shadow_route_id=route.route_id,
                primary_inference_id=invocation.primary_inference_id,
            )
        except Exception as exc:
            return _DispatchResult(
                role=role,
                deployment=binding.deployment,
                target=target,
                terminal=None,
                error=exc,
            )
        try:
            self._validate_terminal(
                contract=route.contract,
                route=route,
                invocation=invocation,
                deployment=binding.deployment,
                target=target,
                terminal=terminal,
            )
        except ExperimentExecutionError as exc:
            return _DispatchResult(
                role=role,
                deployment=binding.deployment,
                target=target,
                terminal=terminal,
                terminal_validation_error=str(exc),
            )
        return _DispatchResult(
            role=role,
            deployment=binding.deployment,
            target=target,
            terminal=terminal,
        )

    @staticmethod
    def _validate_terminal(
        *,
        contract: ExperimentContract,
        route: ExperimentRoute,
        invocation: ExperimentInvocation,
        deployment: ModelDeployment,
        target: ExperimentTargetInput,
        terminal: ModelInference,
    ) -> None:
        expected = {
            "stage": invocation.task,
            "provider": deployment.provider,
            "model_name": deployment.model_name,
            "model_version": deployment.model_version,
            "adapter_version": deployment.adapter_version,
            "capability_snapshot_id": deployment.capability_snapshot_id,
            "capability_snapshot_digest": deployment.capability_snapshot_digest,
            "mcap_id": invocation.mcap_id,
            "package_set_id": invocation.package_set_id,
            "camera_mapping_run_id": invocation.camera_mapping_run_id,
            "alignment_id": invocation.alignment_id,
            "start_ns": invocation.start_ns,
            "end_ns": invocation.end_ns,
            "experiment_id": contract.experiment_id,
            "shadow_route_id": route.route_id,
            "primary_inference_id": invocation.primary_inference_id,
            "shadow": True,
            "input_plan_id": target.input_plan.input_plan_id,
            "input_plan_semantic_sha256": target.input_plan.semantic_sha256,
            "input_plan_part_ordinal": target.input_plan_part_ordinal,
            "rendered_input_digest": _selected_rendered_digest(target),
        }
        mismatches = [
            field
            for field, expected_value in expected.items()
            if getattr(terminal, field) != expected_value
        ]
        if terminal.input_config.get("logical_dependency_sha256") != contract.contract_digest:
            mismatches.append("logical_dependency_sha256")
        if mismatches:
            raise ExperimentExecutionError(
                "experiment terminal is inconsistent with its dispatch: " + ", ".join(mismatches)
            )

    @staticmethod
    def _side_outcome(result: _DispatchResult | None) -> ExperimentSideOutcome | None:
        if result is None:
            return None
        terminal = result.terminal
        if terminal is None:
            error = result.error
            return ExperimentSideOutcome(
                role=result.role,
                deployment_id=result.deployment.deployment_id,
                execution_error_type=(
                    type(error).__name__ if error is not None else "UnknownError"
                ),
                execution_error_detail=(
                    str(error) if error is not None else "dispatch returned no terminal"
                ),
            )
        return ExperimentSideOutcome(
            role=result.role,
            deployment_id=result.deployment.deployment_id,
            inference_id=terminal.inference_id,
            logical_invocation_id=terminal.logical_invocation_id,
            status=terminal.status,
            output_valid=terminal.output_valid,
            retry_count=terminal.retry_count,
            latency_ms=terminal.latency_ms,
            usage=terminal.usage,
            failure_code=terminal.failure.code if terminal.failure is not None else None,
            terminal_validation_error=result.terminal_validation_error,
        )

    @staticmethod
    def _compare(
        *,
        representation: ExperimentInputRepresentation,
        mode: RouteMode,
        control: _DispatchResult | None,
        candidate: _DispatchResult | None,
        options: _ComparisonOptions,
    ) -> tuple[ExperimentComparisonStatus, bool, tuple[ExperimentFieldDelta, ...]]:
        if mode is RouteMode.SHADOW:
            return ExperimentComparisonStatus.SINGLE_OBSERVATION, False, ()
        assert control is not None and candidate is not None
        control_success = _successful_terminal(control)
        candidate_success = _successful_terminal(candidate)
        if not control_success and not candidate_success:
            return ExperimentComparisonStatus.BOTH_FAILURE, False, ()
        if not control_success:
            return ExperimentComparisonStatus.CONTROL_FAILURE, False, ()
        if not candidate_success:
            return ExperimentComparisonStatus.CANDIDATE_FAILURE, False, ()
        if representation is ExperimentInputRepresentation.MODEL_SPECIFIC_RENDERING:
            return ExperimentComparisonStatus.NOT_COMPARABLE, False, ()

        assert control.terminal is not None and candidate.terminal is not None
        assert control.terminal.normalized_output is not None
        assert candidate.terminal.normalized_output is not None
        deltas = _compute_field_deltas(
            control=control.terminal.normalized_output,
            candidate=candidate.terminal.normalized_output,
            options=options,
        )
        if deltas:
            return ExperimentComparisonStatus.DIFFERENCE, True, deltas
        return ExperimentComparisonStatus.AGREEMENT, True, ()

    def _append_comparison(
        self,
        *,
        route: ExperimentRoute,
        decision: ModelRouteDecision,
        invocation: ExperimentInvocation,
        comparison_config_sha256: Sha256Digest,
        status: ExperimentComparisonStatus,
        comparable: bool,
        control: ExperimentSideOutcome | None,
        candidate: ExperimentSideOutcome | None,
        field_deltas: tuple[ExperimentFieldDelta, ...],
    ) -> ExperimentPairComparison:
        comparison_id = _comparison_id(
            contract=route.contract,
            decision=decision,
            invocation=invocation,
            comparison_config_sha256=comparison_config_sha256,
            control=control,
            candidate=candidate,
        )
        comparison = ExperimentPairComparison(
            comparison_id=comparison_id,
            experiment_id=route.contract.experiment_id,
            experiment_contract_digest=route.contract.contract_digest,
            workload_manifest_sha256=route.contract.workload_manifest_sha256,
            route_id=route.route_id,
            route_configuration_digest=route.configuration_digest,
            input_identity_sha256=decision.input_identity_sha256,
            comparison_config_sha256=comparison_config_sha256,
            input_representation=route.contract.input_representation,
            attempt=invocation.attempt,
            retry_count=invocation.retry_count,
            status=status,
            comparable=comparable,
            control_inference_id=control.inference_id if control is not None else None,
            candidate_inference_id=(candidate.inference_id if candidate is not None else None),
            control=control,
            candidate=candidate,
            field_deltas=field_deltas,
        )
        return self._comparison_ledger.append(comparison)


def _logical_model_identity(deployment: ModelDeployment) -> tuple[str, str, str, str, str]:
    """Return fields the existing orchestrator uses in a logical invocation."""

    return (
        deployment.provider,
        deployment.model_name,
        deployment.model_version,
        deployment.adapter_version,
        deployment.capability_snapshot_digest,
    )


def _selected_rendered_digest(target: ExperimentTargetInput) -> str:
    if target.input_plan_part_ordinal is None:
        return target.input_plan.rendering_sha256
    return target.input_plan.call_plan.parts[target.input_plan_part_ordinal].item_manifest_sha256


def _successful_terminal(result: _DispatchResult) -> bool:
    terminal = result.terminal
    return (
        terminal is not None
        and result.error is None
        and result.terminal_validation_error is None
        and terminal.status is InferenceStatus.SUCCEEDED
        and terminal.output_valid
        and terminal.normalized_output is not None
        and terminal.failure is None
    )


def _comparison_id(
    *,
    contract: ExperimentContract,
    decision: ModelRouteDecision,
    invocation: ExperimentInvocation,
    comparison_config_sha256: Sha256Digest,
    control: ExperimentSideOutcome | None,
    candidate: ExperimentSideOutcome | None,
) -> str:
    digest = semantic_sha256(
        {
            "experiment_contract_digest": contract.contract_digest,
            "route_configuration_digest": decision.route_configuration_digest,
            "input_identity_sha256": decision.input_identity_sha256,
            "comparison_config_sha256": comparison_config_sha256,
            "attempt": invocation.attempt,
            "retry_count": invocation.retry_count,
            "control": _side_identity(control),
            "candidate": _side_identity(candidate),
        }
    )
    return str(uuid5(NAMESPACE_URL, f"robata:experiment-comparison:{digest}"))


def _side_identity(side: ExperimentSideOutcome | None) -> dict[str, str] | None:
    if side is None:
        return None
    if side.inference_id is not None:
        return {"inference_id": side.inference_id}
    return {
        "execution_error_type": side.execution_error_type or "UnknownError",
        "execution_error_detail": side.execution_error_detail or "",
    }


def _comparison_options(config: Mapping[str, object]) -> _ComparisonOptions:
    if not isinstance(config, Mapping):
        raise ExperimentExecutionError("comparison configuration must be a mapping")
    ignored_paths = _string_sequence(
        config.get("ignored_paths", config.get("ignore_paths", ())),
        field="ignored_paths",
    )
    numeric_tolerance = _nonnegative_number(
        config.get("numeric_tolerance", 0.0),
        field="numeric_tolerance",
    )
    numeric_tolerances = _path_numbers(
        config.get("numeric_tolerances", {}),
        field="numeric_tolerances",
    )
    severity_by_path = _path_strings(
        config.get("severity_by_path", {}),
        field="severity_by_path",
    )
    default_severity = config.get("default_severity", "MATERIAL")
    if not isinstance(default_severity, str) or not default_severity:
        raise ExperimentExecutionError("default_severity must be a nonempty string")
    return _ComparisonOptions(
        ignored_paths=ignored_paths,
        global_numeric_tolerance=numeric_tolerance,
        numeric_tolerances=numeric_tolerances,
        severity_by_path=severity_by_path,
        default_severity=default_severity,
    )


def _compute_field_deltas(
    *,
    control: dict[str, object],
    candidate: dict[str, object],
    options: _ComparisonOptions,
) -> tuple[ExperimentFieldDelta, ...]:
    """Compare two normalized outputs without provider-specific field names."""

    deltas: list[ExperimentFieldDelta] = []

    def is_ignored(path: str) -> bool:
        return any(
            path == ignored or path.startswith(f"{ignored}.") or path.startswith(f"{ignored}[")
            for ignored in options.ignored_paths
        )

    def severity(path: str) -> str:
        if path in options.severity_by_path:
            return options.severity_by_path[path]
        matches = [
            configured_path
            for configured_path in options.severity_by_path
            if path.startswith(configured_path)
        ]
        return (
            options.severity_by_path[max(matches, key=len)] if matches else options.default_severity
        )

    def tolerance(path: str) -> float:
        if path in options.numeric_tolerances:
            return options.numeric_tolerances[path]
        matches = [
            configured_path
            for configured_path in options.numeric_tolerances
            if path.startswith(configured_path)
        ]
        return (
            options.numeric_tolerances[max(matches, key=len)]
            if matches
            else options.global_numeric_tolerance
        )

    def add(path: str, left: object | None, right: object | None) -> None:
        if not is_ignored(path):
            deltas.append(
                ExperimentFieldDelta(
                    path=path,
                    control=deepcopy(left),
                    candidate=deepcopy(right),
                    severity=severity(path),
                )
            )

    def compare(left: object, right: object, path: str) -> None:
        if is_ignored(path):
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if any(not isinstance(key, str) for key in (*left.keys(), *right.keys())):
                raise ExperimentExecutionError("normalized output dictionary keys must be strings")
            for key in sorted(set(left) | set(right)):
                child_path = f"{path}.{key}" if path else key
                if key not in left:
                    add(child_path, None, right[key])
                elif key not in right:
                    add(child_path, left[key], None)
                else:
                    compare(left[key], right[key], child_path)
            return
        if (
            isinstance(left, Sequence)
            and not isinstance(left, (str, bytes))
            and isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
        ):
            for index in range(max(len(left), len(right))):
                child_path = f"{path or '$'}[{index}]"
                if index >= len(left):
                    add(child_path, None, right[index])
                elif index >= len(right):
                    add(child_path, left[index], None)
                else:
                    compare(left[index], right[index], child_path)
            return
        if _numbers_equal(left, right, tolerance(path or "$")):
            return
        if left != right:
            add(path or "$", left, right)

    compare(control, candidate, "")
    return tuple(deltas)


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExperimentExecutionError(f"{field} must be a sequence of nonempty strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ExperimentExecutionError(f"{field} must be a sequence of nonempty strings")
    return result


def _nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentExecutionError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ExperimentExecutionError(f"{field} must be a finite nonnegative number")
    return number


def _path_numbers(value: object, *, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ExperimentExecutionError(f"{field} must be a mapping")
    result: dict[str, float] = {}
    for path, number in value.items():
        if not isinstance(path, str) or not path:
            raise ExperimentExecutionError(f"{field} keys must be nonempty strings")
        result[path] = _nonnegative_number(number, field=f"{field}.{path}")
    return result


def _path_strings(value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ExperimentExecutionError(f"{field} must be a mapping")
    result: dict[str, str] = {}
    for path, severity in value.items():
        if not isinstance(path, str) or not path:
            raise ExperimentExecutionError(f"{field} keys must be nonempty strings")
        if not isinstance(severity, str) or not severity:
            raise ExperimentExecutionError(f"{field} values must be nonempty strings")
        result[path] = severity
    return result


def _numbers_equal(left: object, right: object, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    left_number = float(left)
    right_number = float(right)
    return (
        math.isfinite(left_number)
        and math.isfinite(right_number)
        and abs(left_number - right_number) <= tolerance
    )


__all__ = [
    "ExperimentComparisonLedger",
    "ExperimentComparisonStatus",
    "ExperimentDeploymentBinding",
    "ExperimentDeploymentRegistry",
    "ExperimentExecutionConflictError",
    "ExperimentExecutionCoordinator",
    "ExperimentExecutionError",
    "ExperimentExecutionResult",
    "ExperimentFieldDelta",
    "ExperimentInvocation",
    "ExperimentPairComparison",
    "ExperimentSideOutcome",
    "ExperimentTargetInput",
    "InMemoryExperimentComparisonLedger",
]
