"""Fan-out/reduction barrier with criticality-aware aggregation."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from robata.contracts.common import StrictModel
from robata.contracts.logical_nodes import OpaqueUuid
from robata.queue.models import DependencyCriticality, NonEmptyString, NonNegativeInt
from robata.queue.stage import StageStatus


PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class Barrier(StrictModel):
    """A synchronization barrier definition.

    Barriers group related work items so that downstream stages can wait for
    an entire cohort to complete before proceeding.
    """

    barrier_id: OpaqueUuid
    logical_key: NonEmptyString
    expected_member_count: PositiveInt
    empty_semantics: NonEmptyString
    reduction_policy: NonEmptyString
    status: NonEmptyString


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

    ``required_count`` is the minimum number of members that must succeed.
    ``degradable_count`` is the maximum number of members that may complete
    with degraded status before the barrier is considered failed.
    """

    version: NonEmptyString
    required_count: PositiveInt
    degradable_count: NonNegativeInt = 0


class BarrierStorage:
    """Abstract storage interface for barrier state persistence.

    Implementations must provide atomic read-modify-write semantics for
    barrier member submission and state transitions.
    """

    def get_barrier(self, barrier_id: str) -> Barrier | None:
        """Retrieve a barrier by its ID."""
        ...

    def save_barrier(self, barrier: Barrier) -> None:
        """Persist a barrier definition."""
        ...

    def get_state(self, barrier_id: str) -> BarrierState | None:
        """Retrieve the runtime state for a barrier."""
        ...

    def save_state(self, state: BarrierState) -> None:
        """Persist barrier runtime state."""
        ...

    def add_member(self, barrier_id: str, work_item_id: str, outcome: StageStatus) -> None:
        """Record a member outcome for a barrier."""
        ...


class BarrierCoordinator:
    """Fan-out/reduction barrier with criticality-aware aggregation.

    The coordinator manages barrier lifecycle: creation, member submission,
    and aggregate status computation.  It delegates persistence to a
    :class:`BarrierStorage` implementation.
    """

    def __init__(self, storage: BarrierStorage) -> None:
        self._storage = storage

    def create_barrier(
        self,
        logical_key: str,
        expected_members: int,
        reduction_policy: ReductionPolicy,
    ) -> Barrier:
        """Create a new barrier with the given parameters.

        Returns the created :class:`Barrier`.
        """
        # TODO: generate barrier_id and persist via storage
        barrier_id = "00000000-0000-0000-0000-000000000000"
        barrier = Barrier(
            barrier_id=barrier_id,
            logical_key=logical_key,
            expected_member_count=expected_members,
            empty_semantics="fail",
            reduction_policy=reduction_policy.version,
            status="OPEN",
        )
        self._storage.save_barrier(barrier)
        return barrier

    def submit_member(
        self,
        barrier_id: str,
        work_item_id: str,
        outcome: StageStatus,
    ) -> BarrierState:
        """Submit a member outcome to a barrier.

        Returns the updated :class:`BarrierState` after submission.
        """
        # TODO: atomically update barrier state via storage
        state = BarrierState(barrier_id=barrier_id, status="OPEN")
        self._storage.save_state(state)
        return state

    def is_complete(self, barrier_id: str) -> bool:
        """Return whether the barrier has reached a terminal state."""
        state = self._storage.get_state(barrier_id)
        if state is None:
            return False
        return state.status in {"CLOSED", "FAILED"}

    def get_aggregate_status(self, barrier_id: str) -> AggregateStatus:
        """Compute the criticality-aware aggregate status for a barrier.

        Returns an :class:`AggregateStatus` summarizing the overall outcome,
        including degraded and failed camera counts.
        """
        # TODO: compute aggregate based on member criticality and outcomes
        return AggregateStatus(
            overall_status=StageStatus.SUCCEEDED,
            is_complete=self.is_complete(barrier_id),
        )


__all__ = [
    "AggregateStatus",
    "Barrier",
    "BarrierCoordinator",
    "BarrierState",
    "BarrierStorage",
    "ReductionPolicy",
]
