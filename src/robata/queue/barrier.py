"""Provider-neutral fan-out/reduction barriers.

The queue broker is intentionally outside this module.  A barrier is durable
processing truth: workers submit terminal member outcomes, and a reducer may
publish only after the declared member set is terminal.  The in-memory
storage implementation is a deterministic local scaffold; a production
adapter must provide equivalent atomic read/modify/write semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock
from typing import Annotated
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from robata.contracts.common import StrictModel
from robata.contracts.logical_nodes import OpaqueUuid
from robata.queue.models import DependencyCriticality, NonEmptyString, NonNegativeInt
from robata.queue.stage import StageStatus

PositiveInt = Annotated[int, Field(strict=True, ge=1)]

_SUCCESS_OUTCOMES = frozenset(
    {
        StageStatus.SUCCEEDED,
        StageStatus.SKIPPED_POLICY,
        StageStatus.SKIPPED_NOT_NEEDED,
    }
)
_FAILURE_OUTCOMES = frozenset(
    {
        StageStatus.FAILED,
        StageStatus.CANCELLED,
        StageStatus.EXPIRED,
        StageStatus.QUARANTINED,
        StageStatus.INCOMPLETE,
    }
)


class Barrier(StrictModel):
    """A synchronization barrier definition.

    ``expected_member_count`` may be zero.  Empty fan-outs are represented as
    a closed barrier with an explicit ``empty_semantics`` outcome instead of
    leaving downstream work waiting forever.
    """

    barrier_id: OpaqueUuid
    logical_key: NonEmptyString
    expected_member_count: NonNegativeInt
    empty_semantics: NonEmptyString
    reduction_policy: NonEmptyString
    status: NonEmptyString
    required_success_count: NonNegativeInt = 0
    max_degraded_failures: NonNegativeInt = 0


class BarrierMember(StrictModel):
    """One terminal member outcome and its dependency criticality."""

    work_item_id: NonEmptyString
    criticality: DependencyCriticality = DependencyCriticality.REQUIRED
    outcome: StageStatus


class BarrierState(StrictModel):
    """Runtime state of a barrier as members are submitted."""

    barrier_id: OpaqueUuid
    completed_members: NonNegativeInt = 0
    pending_members: NonNegativeInt = 0
    failed_members: NonNegativeInt = 0
    status: NonEmptyString


class AggregateStatus(StrictModel):
    """Criticality-aware aggregate outcome for a barrier cohort."""

    overall_status: StageStatus
    degraded_cameras: NonNegativeInt = 0
    failed_cameras: NonNegativeInt = 0
    is_complete: bool = False


class ReductionPolicy(StrictModel):
    """Rules for determining when a barrier is satisfactorily complete.

    ``required_count`` is the minimum number of successful/explicitly skipped
    members.  ``degradable_count`` bounds failed members marked DEGRADABLE;
    REQUIRED failures always make the completed aggregate INCOMPLETE, while
    OPTIONAL failures are recorded without blocking primary completion.
    """

    version: NonEmptyString
    required_count: NonNegativeInt
    degradable_count: NonNegativeInt = 0


class BarrierStorage:
    """Storage boundary for barrier definitions, members, and state.

    ``add_member`` must be atomic with its state update in a durable adapter.
    The local implementation below uses a process lock and immutable models.
    """

    def get_barrier(self, barrier_id: str) -> Barrier | None:
        raise NotImplementedError

    def save_barrier(self, barrier: Barrier) -> None:
        raise NotImplementedError

    def get_state(self, barrier_id: str) -> BarrierState | None:
        raise NotImplementedError

    def save_state(self, state: BarrierState) -> None:
        raise NotImplementedError

    def add_member(
        self,
        barrier_id: str,
        work_item_id: str,
        outcome: StageStatus,
        criticality: DependencyCriticality = DependencyCriticality.REQUIRED,
    ) -> None:
        raise NotImplementedError

    def get_members(self, barrier_id: str) -> tuple[BarrierMember, ...]:
        raise NotImplementedError


def _terminal_failure(outcome: StageStatus) -> bool:
    return outcome in _FAILURE_OUTCOMES


def _aggregate(
    barrier: Barrier,
    members: Iterable[BarrierMember],
    *,
    is_complete: bool,
) -> AggregateStatus:
    """Apply the documented criticality policy to one member snapshot."""

    member_list = tuple(members)
    failed = sum(_terminal_failure(member.outcome) for member in member_list)
    degraded = sum(
        _terminal_failure(member.outcome) and member.criticality is DependencyCriticality.DEGRADABLE
        for member in member_list
    )

    if barrier.expected_member_count == 0:
        empty_status = StageStatus(barrier.empty_semantics)
        return AggregateStatus(
            overall_status=empty_status,
            degraded_cameras=0,
            failed_cameras=0,
            is_complete=True,
        )

    if not is_complete:
        return AggregateStatus(
            overall_status=StageStatus.PENDING,
            degraded_cameras=degraded,
            failed_cameras=failed,
            is_complete=False,
        )

    successful = sum(member.outcome in _SUCCESS_OUTCOMES for member in member_list)
    required_failed = any(
        member.criticality is DependencyCriticality.REQUIRED
        and member.outcome not in _SUCCESS_OUTCOMES
        for member in member_list
    )
    acceptable = (
        not required_failed
        and successful >= barrier.required_success_count
        and degraded <= barrier.max_degraded_failures
    )
    return AggregateStatus(
        overall_status=StageStatus.SUCCEEDED if acceptable else StageStatus.INCOMPLETE,
        degraded_cameras=degraded,
        failed_cameras=failed,
        is_complete=True,
    )


class InMemoryBarrierStorage(BarrierStorage):
    """Thread-safe, deterministic barrier storage for local execution/tests."""

    def __init__(self) -> None:
        self._barriers: dict[str, Barrier] = {}
        self._states: dict[str, BarrierState] = {}
        self._members: dict[str, dict[str, BarrierMember]] = {}
        self._lock = RLock()

    def get_barrier(self, barrier_id: str) -> Barrier | None:
        with self._lock:
            return self._barriers.get(str(barrier_id))

    def save_barrier(self, barrier: Barrier) -> None:
        barrier_id = str(barrier.barrier_id)
        with self._lock:
            existing = self._barriers.get(barrier_id)
            if existing is not None and existing != barrier:
                raise ValueError(f"barrier already exists with different definition: {barrier_id}")
            self._barriers[barrier_id] = barrier
            self._members.setdefault(barrier_id, {})
            self._states.setdefault(
                barrier_id,
                BarrierState(
                    barrier_id=barrier_id,
                    pending_members=barrier.expected_member_count,
                    status=barrier.status,
                ),
            )

    def get_state(self, barrier_id: str) -> BarrierState | None:
        with self._lock:
            return self._states.get(str(barrier_id))

    def save_state(self, state: BarrierState) -> None:
        barrier_id = str(state.barrier_id)
        with self._lock:
            if barrier_id not in self._barriers:
                raise KeyError(f"unknown barrier: {barrier_id}")
            self._states[barrier_id] = state

    def get_members(self, barrier_id: str) -> tuple[BarrierMember, ...]:
        with self._lock:
            members = self._members.get(str(barrier_id))
            if members is None:
                raise KeyError(f"unknown barrier: {barrier_id}")
            return tuple(members.values())

    def add_member(
        self,
        barrier_id: str,
        work_item_id: str,
        outcome: StageStatus,
        criticality: DependencyCriticality = DependencyCriticality.REQUIRED,
    ) -> None:
        barrier_key = str(barrier_id)
        if not isinstance(work_item_id, str) or not work_item_id.strip():
            raise ValueError("work_item_id must be a non-empty string")
        if not isinstance(outcome, StageStatus):
            raise ValueError("barrier members must submit a StageStatus")
        if outcome not in _SUCCESS_OUTCOMES and outcome not in _FAILURE_OUTCOMES:
            raise ValueError("barrier members must submit a terminal StageStatus")
        try:
            criticality = DependencyCriticality(criticality)
        except ValueError as exc:
            raise ValueError("invalid dependency criticality") from exc

        with self._lock:
            barrier = self._barriers.get(barrier_key)
            if barrier is None:
                raise KeyError(f"unknown barrier: {barrier_key}")
            members = self._members[barrier_key]
            existing = members.get(work_item_id)
            if existing is not None:
                if existing.outcome is outcome and existing.criticality is criticality:
                    return
                raise ValueError(f"conflicting replay for barrier member: {work_item_id}")
            if len(members) >= barrier.expected_member_count:
                raise ValueError(f"barrier member capacity exceeded: {barrier_key}")
            members[work_item_id] = BarrierMember(
                work_item_id=work_item_id,
                criticality=criticality,
                outcome=outcome,
            )
            complete = len(members) == barrier.expected_member_count
            aggregate = _aggregate(barrier, members.values(), is_complete=complete)
            self._states[barrier_key] = BarrierState(
                barrier_id=barrier_key,
                completed_members=len(members),
                pending_members=barrier.expected_member_count - len(members),
                failed_members=aggregate.failed_cameras,
                status=(
                    "CLOSED"
                    if complete and aggregate.overall_status is not StageStatus.INCOMPLETE
                    else "FAILED"
                    if complete
                    else "OPEN"
                ),
            )


class BarrierCoordinator:
    """Create barriers and expose criticality-aware aggregate state."""

    def __init__(self, storage: BarrierStorage) -> None:
        self._storage = storage

    def create_barrier(
        self,
        logical_key: str,
        expected_members: int,
        reduction_policy: ReductionPolicy,
        *,
        empty_semantics: str = StageStatus.SKIPPED_NOT_NEEDED.value,
    ) -> Barrier:
        """Create or replay a deterministic barrier definition.

        The UUID is derived solely from the logical key, so redelivered
        planning work cannot create duplicate barriers.
        """

        if not isinstance(logical_key, str) or not logical_key.strip():
            raise ValueError("logical_key must be a non-empty string")
        if isinstance(expected_members, bool) or not isinstance(expected_members, int):
            raise ValueError("expected_members must be an integer")
        if expected_members < 0:
            raise ValueError("expected_members must be non-negative")
        if reduction_policy.required_count > expected_members:
            raise ValueError("required_count cannot exceed expected_members")
        if reduction_policy.degradable_count > expected_members:
            raise ValueError("degradable_count cannot exceed expected_members")
        try:
            empty_outcome = StageStatus(empty_semantics)
        except (TypeError, ValueError) as exc:
            raise ValueError("empty_semantics must identify a terminal StageStatus") from exc
        if empty_outcome not in _SUCCESS_OUTCOMES and empty_outcome not in _FAILURE_OUTCOMES:
            raise ValueError("empty_semantics must identify a terminal StageStatus")

        barrier_id = str(uuid5(NAMESPACE_URL, f"robata:barrier:{logical_key}"))
        status = (
            "FAILED"
            if expected_members == 0 and empty_outcome in _FAILURE_OUTCOMES
            else "CLOSED"
            if expected_members == 0
            else "OPEN"
        )
        barrier = Barrier(
            barrier_id=barrier_id,
            logical_key=logical_key,
            expected_member_count=expected_members,
            empty_semantics=empty_semantics,
            reduction_policy=reduction_policy.version,
            status=status,
            required_success_count=reduction_policy.required_count,
            max_degraded_failures=reduction_policy.degradable_count,
        )
        self._storage.save_barrier(barrier)
        if self._storage.get_state(barrier_id) is None:
            self._storage.save_state(
                BarrierState(
                    barrier_id=barrier_id,
                    completed_members=0,
                    pending_members=expected_members,
                    failed_members=0,
                    status=status,
                )
            )
        return barrier

    def submit_member(
        self,
        barrier_id: str,
        work_item_id: str,
        outcome: StageStatus,
        criticality: DependencyCriticality = DependencyCriticality.REQUIRED,
    ) -> BarrierState:
        """Atomically submit one terminal member outcome and return state."""

        try:
            outcome = StageStatus(outcome)
            criticality = DependencyCriticality(criticality)
        except ValueError as exc:
            raise ValueError("invalid barrier member outcome or criticality") from exc
        self._storage.add_member(
            barrier_id,
            work_item_id,
            outcome,
            criticality,
        )
        state = self._storage.get_state(barrier_id)
        if state is None:
            raise KeyError(f"storage did not return state for barrier: {barrier_id}")
        return state

    def is_complete(self, barrier_id: str) -> bool:
        """Return whether the barrier has reached a terminal state."""

        state = self._storage.get_state(barrier_id)
        return state is not None and state.status in {"CLOSED", "FAILED"}

    def get_aggregate_status(self, barrier_id: str) -> AggregateStatus:
        """Compute aggregate status from one storage snapshot."""

        barrier = self._storage.get_barrier(barrier_id)
        state = self._storage.get_state(barrier_id)
        if barrier is None or state is None:
            raise KeyError(f"unknown barrier: {barrier_id}")
        return _aggregate(
            barrier,
            self._storage.get_members(barrier_id),
            is_complete=state.status in {"CLOSED", "FAILED"},
        )


__all__ = [
    "AggregateStatus",
    "Barrier",
    "BarrierCoordinator",
    "BarrierMember",
    "BarrierState",
    "BarrierStorage",
    "InMemoryBarrierStorage",
    "ReductionPolicy",
]
