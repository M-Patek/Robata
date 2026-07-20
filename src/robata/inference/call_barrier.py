"""Durable call-part completion and reduction for inference input plans.

The generic queue barrier owns terminal membership state. This module binds
that state to the exact immutable members declared by an InferenceInputPlan,
persists selected part outcomes, and publishes at most one ordered reduction.
Production storage adapters must provide the same append-only conflict and
atomicity guarantees as the in-memory reference implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Annotated, Literal, Protocol, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import CanonicalizationError, semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid, Rfc3339Timestamp
from robata.inference.input_plan import InferenceCallPart, InferenceInputPlan
from robata.inference.models import (
    InferenceAttemptSelection,
    InferenceFailure,
    InferenceStatus,
    ModelInference,
    inference_attempt_selection_logical_key,
)
from robata.queue.barrier import AggregateStatus, BarrierCoordinator, ReductionPolicy
from robata.queue.stage import StageStatus

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class InferenceCallBarrierError(RuntimeError):
    """Base error for call barrier declaration, completion, and reduction."""


class InferenceCallBarrierConflictError(InferenceCallBarrierError):
    """Raised when an append-only record is replayed with different content."""


class InferenceCallBarrierOpenError(InferenceCallBarrierError):
    """Raised when reduction is requested before every declared part is terminal."""


class InferenceCallBarrierFailedError(InferenceCallBarrierError):
    """Raised when a required part failed and reduction is therefore forbidden."""


class InferenceCallReductionConfigurationError(InferenceCallBarrierError):
    """Raised when the exact versioned reduction implementation is unavailable."""


class InferenceCallBarrierDefinition(StrictModel):
    """Immutable declaration of the exact call-plan member set."""

    schema_version: Literal["1.0"] = "1.0"
    barrier_id: OpaqueUuid
    barrier_semantic_sha256: Sha256Digest
    barrier_logical_key: NonEmptyString
    input_plan_semantic_sha256: Sha256Digest
    call_plan_sha256: Sha256Digest
    part_count: PositiveInt
    expected_part_semantic_sha256s: tuple[Sha256Digest, ...]
    expected_part_logical_keys: tuple[NonEmptyString, ...]
    expected_part_idempotency_keys: tuple[NonEmptyString, ...]
    reduction_policy: NonEmptyString
    reduction_policy_version: SchemaVersion
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_expected_members(self) -> Self:
        if (
            len(self.expected_part_semantic_sha256s) != self.part_count
            or len(self.expected_part_logical_keys) != self.part_count
            or len(self.expected_part_idempotency_keys) != self.part_count
        ):
            raise ValueError("barrier definition must declare every expected call part")
        if len(set(self.expected_part_semantic_sha256s)) != self.part_count:
            raise ValueError("barrier part semantic identities must be unique")
        if len(set(self.expected_part_logical_keys)) != self.part_count:
            raise ValueError("barrier part logical keys must be unique")
        if len(set(self.expected_part_idempotency_keys)) != self.part_count:
            raise ValueError("barrier part idempotency keys must be unique")
        return self


class InferenceCallPartCompletion(StrictModel):
    """One selected final outcome bound to an exact declared call part."""

    schema_version: Literal["1.0"] = "1.0"
    completion_id: OpaqueUuid
    completion_semantic_sha256: Sha256Digest
    barrier_id: OpaqueUuid
    barrier_semantic_sha256: Sha256Digest
    input_plan_semantic_sha256: Sha256Digest
    call_plan_sha256: Sha256Digest
    part_ordinal: NonNegativeInt
    part_count: PositiveInt
    part_semantic_sha256: Sha256Digest
    part_logical_key: NonEmptyString
    part_idempotency_key: NonEmptyString
    inference_id: OpaqueUuid
    logical_invocation_id: OpaqueUuid
    selection_id: OpaqueUuid | None
    selection_policy_version: SchemaVersion | None
    selection_decision_logical_key: NodeLogicalKey | None
    attempt: PositiveInt
    status: InferenceStatus
    normalized_output: dict[str, object] | None
    normalized_output_sha256: Sha256Digest | None
    raw_output_artifact_id: NonEmptyString | None
    failure: InferenceFailure | None
    completed_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_terminal_shape_and_identity(self) -> Self:
        succeeded = self.status is InferenceStatus.SUCCEEDED
        if succeeded:
            if (
                self.normalized_output is None
                or self.normalized_output_sha256 is None
                or self.raw_output_artifact_id is None
                or self.selection_id is None
                or self.selection_policy_version is None
                or self.selection_decision_logical_key is None
                or self.failure is not None
            ):
                raise ValueError(
                    "successful part completion requires normalized and raw output references"
                )
            try:
                expected_output = semantic_sha256(self.normalized_output)
            except (CanonicalizationError, TypeError, ValueError) as exc:
                raise ValueError("part normalized output is not canonical JSON") from exc
            if self.normalized_output_sha256 != expected_output:
                raise ValueError("part normalized output digest is inconsistent")
            expected_selection_key = inference_attempt_selection_logical_key(
                logical_invocation_id=self.logical_invocation_id,
                policy_version=self.selection_policy_version,
            )
            if self.selection_decision_logical_key != expected_selection_key:
                raise ValueError("part selection decision logical key is inconsistent")
        elif (
            self.normalized_output is not None
            or self.normalized_output_sha256 is not None
            or self.selection_id is not None
            or self.selection_policy_version is not None
            or self.selection_decision_logical_key is not None
            or self.failure is None
        ):
            raise ValueError("failed part completion requires only failure details")

        expected = _completion_semantic_sha256(
            barrier_semantic_sha256=self.barrier_semantic_sha256,
            input_plan_semantic_sha256=self.input_plan_semantic_sha256,
            call_plan_sha256=self.call_plan_sha256,
            part_semantic_sha256=self.part_semantic_sha256,
            part_idempotency_key=self.part_idempotency_key,
            inference_id=self.inference_id,
            logical_invocation_id=self.logical_invocation_id,
            selection_id=self.selection_id,
            selection_policy_version=self.selection_policy_version,
            selection_decision_logical_key=self.selection_decision_logical_key,
            attempt=self.attempt,
            status=self.status,
            normalized_output_sha256=self.normalized_output_sha256,
            raw_output_artifact_id=self.raw_output_artifact_id,
            failure=self.failure,
        )
        if self.completion_semantic_sha256 != expected:
            raise ValueError("part completion semantic digest is inconsistent")
        if self.completion_id != _stable_uuid("inference-call-completion", expected):
            raise ValueError("part completion identity is inconsistent")
        return self


class InferenceCallReduction(StrictModel):
    """One immutable, ordered logical result published after barrier success."""

    schema_version: Literal["1.0"] = "1.0"
    reduction_id: OpaqueUuid
    reduction_semantic_sha256: Sha256Digest
    barrier_id: OpaqueUuid
    barrier_semantic_sha256: Sha256Digest
    input_plan_semantic_sha256: Sha256Digest
    call_plan_sha256: Sha256Digest
    reduction_policy: NonEmptyString
    reduction_policy_version: SchemaVersion
    output_schema_sha256: Sha256Digest
    ordered_completion_ids: tuple[OpaqueUuid, ...]
    ordered_part_semantic_sha256s: tuple[Sha256Digest, ...]
    ordered_normalized_output_sha256s: tuple[Sha256Digest, ...]
    ordered_selection_decision_logical_keys: tuple[NodeLogicalKey, ...]
    normalized_output: dict[str, object]
    normalized_output_sha256: Sha256Digest
    reduced_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_reduction_identity(self) -> Self:
        count = len(self.ordered_completion_ids)
        if (
            count == 0
            or len(self.ordered_part_semantic_sha256s) != count
            or len(self.ordered_normalized_output_sha256s) != count
            or len(self.ordered_selection_decision_logical_keys) != count
        ):
            raise ValueError("reduction must bind equal nonempty ordered member tuples")
        try:
            expected_output = semantic_sha256(self.normalized_output)
        except (CanonicalizationError, TypeError, ValueError) as exc:
            raise ValueError("reduced output is not canonical JSON") from exc
        if self.normalized_output_sha256 != expected_output:
            raise ValueError("reduced output digest is inconsistent")
        expected = _reduction_semantic_sha256(
            barrier_semantic_sha256=self.barrier_semantic_sha256,
            input_plan_semantic_sha256=self.input_plan_semantic_sha256,
            call_plan_sha256=self.call_plan_sha256,
            reduction_policy=self.reduction_policy,
            reduction_policy_version=self.reduction_policy_version,
            output_schema_sha256=self.output_schema_sha256,
            ordered_part_semantic_sha256s=self.ordered_part_semantic_sha256s,
            ordered_normalized_output_sha256s=self.ordered_normalized_output_sha256s,
            ordered_selection_decision_logical_keys=(self.ordered_selection_decision_logical_keys),
            normalized_output_sha256=self.normalized_output_sha256,
        )
        if self.reduction_semantic_sha256 != expected:
            raise ValueError("reduction semantic digest is inconsistent")
        if self.reduction_id != _stable_uuid("inference-call-reduction", expected):
            raise ValueError("reduction identity is inconsistent")
        return self


class InferenceCallReducer(Protocol):
    """Exact policy/version implementation for deterministic part reduction."""

    def reduce(
        self,
        *,
        input_plan: InferenceInputPlan,
        ordered_completions: tuple[InferenceCallPartCompletion, ...],
    ) -> Mapping[str, object]: ...


class InferenceCallBarrierStorage(Protocol):
    """Append-only persistence port for call barriers, completions, and reductions."""

    def append_definition(
        self, definition: InferenceCallBarrierDefinition
    ) -> InferenceCallBarrierDefinition: ...

    def get_definition(self, barrier_id: str) -> InferenceCallBarrierDefinition | None: ...

    def append_completion(
        self, completion: InferenceCallPartCompletion
    ) -> InferenceCallPartCompletion: ...

    def list_completions(self, barrier_id: str) -> tuple[InferenceCallPartCompletion, ...]: ...

    def append_reduction(self, reduction: InferenceCallReduction) -> InferenceCallReduction: ...

    def get_reduction(self, barrier_id: str) -> InferenceCallReduction | None: ...


class InMemoryInferenceCallBarrierStorage:
    """Thread-safe append-only reference storage for local execution and tests."""

    def __init__(self) -> None:
        self._definitions: dict[str, InferenceCallBarrierDefinition] = {}
        self._completions: dict[str, dict[str, InferenceCallPartCompletion]] = {}
        self._reductions: dict[str, InferenceCallReduction] = {}
        self._lock = RLock()

    def append_definition(
        self, definition: InferenceCallBarrierDefinition
    ) -> InferenceCallBarrierDefinition:
        barrier_id = definition.barrier_id
        with self._lock:
            existing = self._definitions.get(barrier_id)
            if existing is not None and existing != definition:
                raise InferenceCallBarrierConflictError(
                    f"conflicting call barrier definition: {barrier_id}"
                )
            self._definitions[barrier_id] = definition
            self._completions.setdefault(barrier_id, {})
            return definition

    def get_definition(self, barrier_id: str) -> InferenceCallBarrierDefinition | None:
        with self._lock:
            return self._definitions.get(str(barrier_id))

    def append_completion(
        self, completion: InferenceCallPartCompletion
    ) -> InferenceCallPartCompletion:
        barrier_id = completion.barrier_id
        with self._lock:
            definition = self._definitions.get(barrier_id)
            if definition is None:
                raise KeyError(f"unknown inference call barrier: {barrier_id}")
            ordinal = completion.part_ordinal
            if (
                ordinal >= definition.part_count
                or completion.part_count != definition.part_count
                or completion.barrier_semantic_sha256 != definition.barrier_semantic_sha256
                or completion.input_plan_semantic_sha256 != definition.input_plan_semantic_sha256
                or completion.call_plan_sha256 != definition.call_plan_sha256
                or completion.part_semantic_sha256
                != definition.expected_part_semantic_sha256s[ordinal]
                or completion.part_logical_key != definition.expected_part_logical_keys[ordinal]
                or completion.part_idempotency_key
                != definition.expected_part_idempotency_keys[ordinal]
            ):
                raise InferenceCallBarrierConflictError(
                    "completion does not match the declared call barrier member"
                )
            members = self._completions[barrier_id]
            existing = members.get(completion.part_semantic_sha256)
            if existing is not None and existing != completion:
                raise InferenceCallBarrierConflictError(
                    f"call part already has a different final completion: {ordinal}"
                )
            members[completion.part_semantic_sha256] = completion
            return completion

    def list_completions(self, barrier_id: str) -> tuple[InferenceCallPartCompletion, ...]:
        with self._lock:
            barrier_key = str(barrier_id)
            if barrier_key not in self._definitions:
                raise KeyError(f"unknown inference call barrier: {barrier_id}")
            return tuple(self._completions[barrier_key].values())

    def append_reduction(self, reduction: InferenceCallReduction) -> InferenceCallReduction:
        barrier_id = reduction.barrier_id
        with self._lock:
            definition = self._definitions.get(barrier_id)
            if definition is None:
                raise KeyError(f"unknown inference call barrier: {barrier_id}")
            ordered = tuple(
                sorted(
                    self._completions[barrier_id].values(),
                    key=lambda completion: completion.part_ordinal,
                )
            )
            if (
                len(ordered) != definition.part_count
                or reduction.barrier_semantic_sha256 != definition.barrier_semantic_sha256
                or reduction.input_plan_semantic_sha256 != definition.input_plan_semantic_sha256
                or reduction.call_plan_sha256 != definition.call_plan_sha256
                or reduction.reduction_policy != definition.reduction_policy
                or reduction.reduction_policy_version != definition.reduction_policy_version
                or reduction.ordered_completion_ids != tuple(item.completion_id for item in ordered)
                or reduction.ordered_part_semantic_sha256s
                != tuple(item.part_semantic_sha256 for item in ordered)
                or reduction.ordered_normalized_output_sha256s
                != tuple(_required_output_digest(item) for item in ordered)
                or reduction.ordered_selection_decision_logical_keys
                != tuple(_required_selection_key(item) for item in ordered)
            ):
                raise InferenceCallBarrierConflictError(
                    "reduction does not match the completed declared member set"
                )
            existing = self._reductions.get(barrier_id)
            if existing is not None and existing != reduction:
                raise InferenceCallBarrierConflictError(
                    f"call barrier already has a different reduction: {barrier_id}"
                )
            self._reductions[barrier_id] = reduction
            return reduction

    def get_reduction(self, barrier_id: str) -> InferenceCallReduction | None:
        with self._lock:
            return self._reductions.get(str(barrier_id))


class InferenceCallBarrierCoordinator:
    """Bind orchestrated call-part attempts to a durable ordered barrier."""

    def __init__(
        self,
        *,
        barriers: BarrierCoordinator,
        storage: InferenceCallBarrierStorage,
        reducers: Mapping[tuple[str, str], InferenceCallReducer] | None = None,
    ) -> None:
        self._barriers = barriers
        self._storage = storage
        self._reducers = dict(reducers or {})

    def declare(
        self,
        input_plan: InferenceInputPlan,
        *,
        created_at: str,
    ) -> InferenceCallBarrierDefinition:
        """Declare or replay the exact immutable member set before dispatch."""

        plan = _validated_plan(input_plan)
        call_plan = plan.call_plan
        barrier = self._barriers.create_barrier(
            call_plan.barrier_logical_key,
            len(call_plan.parts),
            ReductionPolicy(
                version=call_plan.reduction_policy_version,
                required_count=len(call_plan.parts),
                degradable_count=0,
            ),
        )
        definition = InferenceCallBarrierDefinition(
            barrier_id=barrier.barrier_id,
            barrier_semantic_sha256=call_plan.barrier_semantic_sha256,
            barrier_logical_key=call_plan.barrier_logical_key,
            input_plan_semantic_sha256=plan.semantic_sha256,
            call_plan_sha256=call_plan.call_plan_sha256,
            part_count=len(call_plan.parts),
            expected_part_semantic_sha256s=tuple(
                part.part_semantic_sha256 for part in call_plan.parts
            ),
            expected_part_logical_keys=tuple(part.part_logical_key for part in call_plan.parts),
            expected_part_idempotency_keys=tuple(part.idempotency_key for part in call_plan.parts),
            reduction_policy=call_plan.reduction_policy,
            reduction_policy_version=call_plan.reduction_policy_version,
            created_at=created_at,
        )
        existing = self._storage.get_definition(barrier.barrier_id)
        if existing is not None:
            if _definition_projection(existing) != _definition_projection(definition):
                raise InferenceCallBarrierConflictError(
                    "call barrier identity already has a different plan binding"
                )
            return existing
        return self._storage.append_definition(definition)

    def submit_part_terminal(
        self,
        input_plan: InferenceInputPlan,
        inference: ModelInference,
        *,
        selection: InferenceAttemptSelection | None = None,
        failure_is_final: bool = False,
    ) -> InferenceCallPartCompletion:
        """Persist one selected terminal, then idempotently advance its barrier."""

        plan = _validated_plan(input_plan)
        definition = self._require_definition(plan)
        part = _bound_part(plan, inference)
        raw_output_artifact_id = _raw_output_artifact_id(inference)
        succeeded = inference.status is InferenceStatus.SUCCEEDED
        if succeeded:
            if not inference.output_valid or inference.normalized_output is None:
                raise InferenceCallBarrierError(
                    "successful call part requires valid normalized output"
                )
            if inference.failure is not None:
                raise InferenceCallBarrierError(
                    "successful call part cannot contain failure details"
                )
            if raw_output_artifact_id is None:
                raise InferenceCallBarrierError(
                    "successful call part requires a raw output artifact reference"
                )
            if (
                selection is None
                or selection.inference_id != inference.inference_id
                or selection.logical_invocation_id != inference.logical_invocation_id
                or selection.selection_decision_logical_key
                != inference_attempt_selection_logical_key(
                    logical_invocation_id=inference.logical_invocation_id,
                    policy_version=selection.policy_version,
                )
            ):
                raise InferenceCallBarrierError(
                    "successful call part requires its exact persisted attempt selection"
                )
            normalized_output = dict(inference.normalized_output)
            normalized_output_sha256: Sha256Digest | None = semantic_sha256(normalized_output)
            failure = None
        else:
            if selection is not None:
                raise InferenceCallBarrierError(
                    "failed call part cannot reference an attempt selection"
                )
            if not failure_is_final:
                raise InferenceCallBarrierError(
                    "non-success call part requires explicit final-failure confirmation"
                )
            if inference.output_valid or inference.failure is None:
                raise InferenceCallBarrierError(
                    "failed call part must carry invalid output and failure details"
                )
            normalized_output = None
            normalized_output_sha256 = None
            failure = inference.failure

        completion_digest = _completion_semantic_sha256(
            barrier_semantic_sha256=definition.barrier_semantic_sha256,
            input_plan_semantic_sha256=plan.semantic_sha256,
            call_plan_sha256=plan.call_plan.call_plan_sha256,
            part_semantic_sha256=part.part_semantic_sha256,
            part_idempotency_key=part.idempotency_key,
            inference_id=inference.inference_id,
            logical_invocation_id=inference.logical_invocation_id,
            selection_id=selection.selection_id if selection is not None else None,
            selection_policy_version=(selection.policy_version if selection is not None else None),
            selection_decision_logical_key=(
                selection.selection_decision_logical_key if selection is not None else None
            ),
            attempt=inference.attempt,
            status=inference.status,
            normalized_output_sha256=normalized_output_sha256,
            raw_output_artifact_id=raw_output_artifact_id,
            failure=failure,
        )
        completion = InferenceCallPartCompletion(
            completion_id=_stable_uuid("inference-call-completion", completion_digest),
            completion_semantic_sha256=completion_digest,
            barrier_id=definition.barrier_id,
            barrier_semantic_sha256=definition.barrier_semantic_sha256,
            input_plan_semantic_sha256=plan.semantic_sha256,
            call_plan_sha256=plan.call_plan.call_plan_sha256,
            part_ordinal=part.ordinal,
            part_count=part.part_count,
            part_semantic_sha256=part.part_semantic_sha256,
            part_logical_key=part.part_logical_key,
            part_idempotency_key=part.idempotency_key,
            inference_id=inference.inference_id,
            logical_invocation_id=inference.logical_invocation_id,
            selection_id=selection.selection_id if selection is not None else None,
            selection_policy_version=(selection.policy_version if selection is not None else None),
            selection_decision_logical_key=(
                selection.selection_decision_logical_key if selection is not None else None
            ),
            attempt=inference.attempt,
            status=inference.status,
            normalized_output=normalized_output,
            normalized_output_sha256=normalized_output_sha256,
            raw_output_artifact_id=raw_output_artifact_id,
            failure=failure,
            completed_at=inference.completed_at,
        )
        stored = self._storage.append_completion(completion)
        self._barriers.submit_member(
            definition.barrier_id,
            part.part_logical_key,
            _stage_status(inference.status),
        )
        return stored

    def get_aggregate_status(self, input_plan: InferenceInputPlan) -> AggregateStatus:
        """Return the generic durable barrier state for this exact plan."""

        definition = self._require_definition(_validated_plan(input_plan))
        return self._barriers.get_aggregate_status(definition.barrier_id)

    def reduce(
        self,
        input_plan: InferenceInputPlan,
        *,
        reduced_at: str,
    ) -> InferenceCallReduction:
        """Publish one deterministic reduction only after every part succeeds."""

        plan = _validated_plan(input_plan)
        definition = self._require_definition(plan)
        existing = self._storage.get_reduction(definition.barrier_id)
        if existing is not None:
            return existing

        aggregate = self._barriers.get_aggregate_status(definition.barrier_id)
        if not aggregate.is_complete:
            raise InferenceCallBarrierOpenError(
                "cannot reduce before every declared call part is terminal"
            )
        if aggregate.overall_status is not StageStatus.SUCCEEDED:
            raise InferenceCallBarrierFailedError(
                "cannot reduce a call barrier containing a failed required part"
            )

        completions = tuple(
            sorted(
                self._storage.list_completions(definition.barrier_id),
                key=lambda completion: completion.part_ordinal,
            )
        )
        _validate_completed_member_set(definition, completions)
        reducer = self._reducers.get(
            (definition.reduction_policy, definition.reduction_policy_version)
        )
        if reducer is None:
            raise InferenceCallReductionConfigurationError(
                "no reducer is registered for the exact policy and version"
            )
        output = reducer.reduce(
            input_plan=plan,
            ordered_completions=completions,
        )
        if not isinstance(output, Mapping):
            raise InferenceCallReductionConfigurationError(
                "inference call reducer must return a mapping"
            )
        normalized_output = dict(output)
        if any(not isinstance(key, str) or not key for key in normalized_output):
            raise InferenceCallReductionConfigurationError(
                "reduced output keys must be nonempty strings"
            )
        try:
            output_sha256 = semantic_sha256(normalized_output)
        except (CanonicalizationError, TypeError, ValueError) as exc:
            raise InferenceCallReductionConfigurationError(
                "reduced output must be canonical JSON"
            ) from exc

        completion_ids = tuple(item.completion_id for item in completions)
        part_digests = tuple(item.part_semantic_sha256 for item in completions)
        output_digests = tuple(_required_output_digest(item) for item in completions)
        selection_keys = tuple(_required_selection_key(item) for item in completions)
        reduction_digest = _reduction_semantic_sha256(
            barrier_semantic_sha256=definition.barrier_semantic_sha256,
            input_plan_semantic_sha256=plan.semantic_sha256,
            call_plan_sha256=plan.call_plan.call_plan_sha256,
            reduction_policy=definition.reduction_policy,
            reduction_policy_version=definition.reduction_policy_version,
            output_schema_sha256=plan.prompt_output.provider_response_schema_sha256,
            ordered_part_semantic_sha256s=part_digests,
            ordered_normalized_output_sha256s=output_digests,
            ordered_selection_decision_logical_keys=selection_keys,
            normalized_output_sha256=output_sha256,
        )
        reduction = InferenceCallReduction(
            reduction_id=_stable_uuid("inference-call-reduction", reduction_digest),
            reduction_semantic_sha256=reduction_digest,
            barrier_id=definition.barrier_id,
            barrier_semantic_sha256=definition.barrier_semantic_sha256,
            input_plan_semantic_sha256=plan.semantic_sha256,
            call_plan_sha256=plan.call_plan.call_plan_sha256,
            reduction_policy=definition.reduction_policy,
            reduction_policy_version=definition.reduction_policy_version,
            output_schema_sha256=plan.prompt_output.provider_response_schema_sha256,
            ordered_completion_ids=completion_ids,
            ordered_part_semantic_sha256s=part_digests,
            ordered_normalized_output_sha256s=output_digests,
            ordered_selection_decision_logical_keys=selection_keys,
            normalized_output=normalized_output,
            normalized_output_sha256=output_sha256,
            reduced_at=reduced_at,
        )
        return self._storage.append_reduction(reduction)

    def _require_definition(self, plan: InferenceInputPlan) -> InferenceCallBarrierDefinition:
        barrier_id = _barrier_id(plan.call_plan.barrier_logical_key)
        definition = self._storage.get_definition(barrier_id)
        if definition is None:
            raise InferenceCallBarrierError(
                "input plan call barrier must be declared before completion"
            )
        expected: dict[str, object] = {
            "barrier_semantic_sha256": plan.call_plan.barrier_semantic_sha256,
            "barrier_logical_key": plan.call_plan.barrier_logical_key,
            "input_plan_semantic_sha256": plan.semantic_sha256,
            "call_plan_sha256": plan.call_plan.call_plan_sha256,
            "part_count": len(plan.call_plan.parts),
            "expected_part_semantic_sha256s": tuple(
                part.part_semantic_sha256 for part in plan.call_plan.parts
            ),
            "expected_part_logical_keys": tuple(
                part.part_logical_key for part in plan.call_plan.parts
            ),
            "expected_part_idempotency_keys": tuple(
                part.idempotency_key for part in plan.call_plan.parts
            ),
            "reduction_policy": plan.call_plan.reduction_policy,
            "reduction_policy_version": plan.call_plan.reduction_policy_version,
        }
        actual = definition.model_dump(
            mode="python",
            exclude={"schema_version", "barrier_id", "created_at"},
        )
        if actual != expected:
            raise InferenceCallBarrierConflictError(
                "declared call barrier does not match the immutable input plan"
            )
        return definition


def _validated_plan(input_plan: InferenceInputPlan) -> InferenceInputPlan:
    if not isinstance(input_plan, InferenceInputPlan):
        raise InferenceCallBarrierError("input_plan must be an InferenceInputPlan")
    try:
        return InferenceInputPlan.model_validate(input_plan.model_dump(mode="python"))
    except ValueError as exc:
        raise InferenceCallBarrierError("input plan failed immutable contract validation") from exc


def _bound_part(
    input_plan: InferenceInputPlan,
    inference: ModelInference,
) -> InferenceCallPart:
    ordinal = inference.input_plan_part_ordinal
    if ordinal is None or ordinal >= len(input_plan.call_plan.parts):
        raise InferenceCallBarrierError(
            "terminal inference does not reference a declared call part"
        )
    part = input_plan.call_plan.parts[ordinal]
    expected_packages = tuple(package.package_id for package in input_plan.subject.packages)
    if (
        inference.input_plan_semantic_sha256 != input_plan.semantic_sha256
        or inference.input_plan_part_count != part.part_count
        or inference.input_plan_part_semantic_sha256 != part.part_semantic_sha256
        or inference.provider_idempotency_key != part.idempotency_key
        or inference.rendered_input_digest != part.item_manifest_sha256
        or inference.stage is not input_plan.subject.task
        or inference.package_ids != expected_packages
    ):
        raise InferenceCallBarrierError(
            "terminal inference identity does not match its declared call part"
        )
    return part


def _validate_completed_member_set(
    definition: InferenceCallBarrierDefinition,
    completions: tuple[InferenceCallPartCompletion, ...],
) -> None:
    if len(completions) != definition.part_count:
        raise InferenceCallBarrierOpenError(
            "persisted part completions do not cover the declared member set"
        )
    if tuple(item.part_ordinal for item in completions) != tuple(range(definition.part_count)):
        raise InferenceCallBarrierConflictError(
            "persisted part completion ordinals are not exact and contiguous"
        )
    if (
        tuple(item.part_semantic_sha256 for item in completions)
        != definition.expected_part_semantic_sha256s
    ):
        raise InferenceCallBarrierConflictError(
            "persisted part completions do not match declared identities"
        )
    if any(item.status is not InferenceStatus.SUCCEEDED for item in completions):
        raise InferenceCallBarrierFailedError(
            "successful barrier state contains a non-success part completion"
        )


def _required_output_digest(completion: InferenceCallPartCompletion) -> Sha256Digest:
    if completion.normalized_output_sha256 is None:
        raise InferenceCallBarrierFailedError(
            "successful call part is missing its normalized output digest"
        )
    return completion.normalized_output_sha256


def _required_selection_key(completion: InferenceCallPartCompletion) -> NodeLogicalKey:
    if completion.selection_decision_logical_key is None:
        raise InferenceCallBarrierFailedError(
            "successful call part is missing its selection decision logical key"
        )
    return completion.selection_decision_logical_key


def _raw_output_artifact_id(inference: ModelInference) -> str | None:
    raw_output = inference.raw_output
    if raw_output is None:
        return None
    artifact_id = raw_output.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise InferenceCallBarrierError("terminal raw output artifact reference is invalid")
    return artifact_id


def _definition_projection(
    definition: InferenceCallBarrierDefinition,
) -> dict[str, object]:
    return definition.model_dump(mode="python", exclude={"created_at"})


def _stage_status(status: InferenceStatus) -> StageStatus:
    if status is InferenceStatus.SUCCEEDED:
        return StageStatus.SUCCEEDED
    if status is InferenceStatus.CANCELLED:
        return StageStatus.CANCELLED
    if status is InferenceStatus.INVALID_OUTPUT:
        return StageStatus.QUARANTINED
    return StageStatus.FAILED


def _completion_semantic_sha256(
    *,
    barrier_semantic_sha256: str,
    input_plan_semantic_sha256: str,
    call_plan_sha256: str,
    part_semantic_sha256: str,
    part_idempotency_key: str,
    inference_id: str,
    logical_invocation_id: str,
    selection_id: str | None,
    selection_policy_version: str | None,
    selection_decision_logical_key: str | None,
    attempt: int,
    status: InferenceStatus,
    normalized_output_sha256: str | None,
    raw_output_artifact_id: str | None,
    failure: InferenceFailure | None,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "barrier_semantic_sha256": barrier_semantic_sha256,
            "input_plan_semantic_sha256": input_plan_semantic_sha256,
            "call_plan_sha256": call_plan_sha256,
            "part_semantic_sha256": part_semantic_sha256,
            "part_idempotency_key": part_idempotency_key,
            "inference_id": inference_id,
            "logical_invocation_id": logical_invocation_id,
            "selection_id": selection_id,
            "selection_policy_version": selection_policy_version,
            "selection_decision_logical_key": selection_decision_logical_key,
            "attempt": attempt,
            "status": status,
            "normalized_output_sha256": normalized_output_sha256,
            "raw_output_artifact_id": raw_output_artifact_id,
            "failure": failure,
        }
    )


def _reduction_semantic_sha256(
    *,
    barrier_semantic_sha256: str,
    input_plan_semantic_sha256: str,
    call_plan_sha256: str,
    reduction_policy: str,
    reduction_policy_version: str,
    output_schema_sha256: str,
    ordered_part_semantic_sha256s: tuple[str, ...],
    ordered_normalized_output_sha256s: tuple[str, ...],
    ordered_selection_decision_logical_keys: tuple[str, ...],
    normalized_output_sha256: str,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "barrier_semantic_sha256": barrier_semantic_sha256,
            "input_plan_semantic_sha256": input_plan_semantic_sha256,
            "call_plan_sha256": call_plan_sha256,
            "reduction_policy": reduction_policy,
            "reduction_policy_version": reduction_policy_version,
            "output_schema_sha256": output_schema_sha256,
            "ordered_part_semantic_sha256s": ordered_part_semantic_sha256s,
            "ordered_normalized_output_sha256s": ordered_normalized_output_sha256s,
            "ordered_selection_decision_logical_keys": (ordered_selection_decision_logical_keys),
            "normalized_output_sha256": normalized_output_sha256,
        }
    )


def _stable_uuid(namespace: str, digest: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{digest}"))


def _barrier_id(logical_key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:barrier:{logical_key}"))


__all__ = [
    "InMemoryInferenceCallBarrierStorage",
    "InferenceCallBarrierConflictError",
    "InferenceCallBarrierCoordinator",
    "InferenceCallBarrierDefinition",
    "InferenceCallBarrierError",
    "InferenceCallBarrierFailedError",
    "InferenceCallBarrierOpenError",
    "InferenceCallBarrierStorage",
    "InferenceCallPartCompletion",
    "InferenceCallReducer",
    "InferenceCallReduction",
    "InferenceCallReductionConfigurationError",
]
