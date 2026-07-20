"""Contract tests for provider-neutral fan-out/reduction barriers."""

from __future__ import annotations

import pytest

from robata.queue import (
    BarrierCoordinator,
    DependencyCriticality,
    InMemoryBarrierStorage,
    ReductionPolicy,
    StageStatus,
)


def _coordinator() -> tuple[BarrierCoordinator, InMemoryBarrierStorage]:
    storage = InMemoryBarrierStorage()
    return BarrierCoordinator(storage), storage


def test_barrier_replay_is_deterministic_and_member_submission_is_idempotent() -> None:
    coordinator, storage = _coordinator()
    policy = ReductionPolicy(version="qa-six-v1", required_count=2, degradable_count=1)
    barrier = coordinator.create_barrier("qa:mcap-1", 3, policy)

    state = coordinator.submit_member(
        barrier.barrier_id,
        "cam-01-work",
        StageStatus.SUCCEEDED,
    )
    replayed_state = coordinator.submit_member(
        barrier.barrier_id,
        "cam-01-work",
        StageStatus.SUCCEEDED,
    )
    replayed_barrier = coordinator.create_barrier("qa:mcap-1", 3, policy)

    assert replayed_barrier.barrier_id == barrier.barrier_id
    assert state == replayed_state
    assert storage.get_state(barrier.barrier_id) == state
    assert state.completed_members == 1
    assert state.pending_members == 2
    aggregate = coordinator.get_aggregate_status(barrier.barrier_id)
    assert aggregate.overall_status is StageStatus.PENDING


def test_degradable_failure_closes_a_complete_barrier_without_cleaning_evidence() -> None:
    coordinator, _ = _coordinator()
    barrier = coordinator.create_barrier(
        "qa:mcap-degraded",
        3,
        ReductionPolicy(version="qa-six-v1", required_count=2, degradable_count=1),
    )
    coordinator.submit_member(barrier.barrier_id, "cam-01", StageStatus.SUCCEEDED)
    coordinator.submit_member(
        barrier.barrier_id,
        "cam-02",
        StageStatus.FAILED,
        DependencyCriticality.DEGRADABLE,
    )
    state = coordinator.submit_member(barrier.barrier_id, "cam-03", StageStatus.SUCCEEDED)

    aggregate = coordinator.get_aggregate_status(barrier.barrier_id)
    assert state.status == "CLOSED"
    assert aggregate.overall_status is StageStatus.SUCCEEDED
    assert aggregate.degraded_cameras == 1
    assert aggregate.failed_cameras == 1
    assert aggregate.is_complete is True


def test_required_failure_persists_incomplete_and_blocks_clean_aggregate() -> None:
    coordinator, _ = _coordinator()
    barrier = coordinator.create_barrier(
        "qa:mcap-required-failure",
        2,
        ReductionPolicy(version="qa-six-v1", required_count=1, degradable_count=1),
    )
    coordinator.submit_member(barrier.barrier_id, "cam-01", StageStatus.FAILED)
    state = coordinator.submit_member(
        barrier.barrier_id,
        "optional-index",
        StageStatus.SUCCEEDED,
        DependencyCriticality.OPTIONAL,
    )

    aggregate = coordinator.get_aggregate_status(barrier.barrier_id)
    assert state.status == "FAILED"
    assert aggregate.overall_status is StageStatus.INCOMPLETE
    assert aggregate.failed_cameras == 1
    assert aggregate.is_complete is True


def test_zero_child_barrier_closes_with_explicit_empty_semantics() -> None:
    coordinator, _ = _coordinator()
    barrier = coordinator.create_barrier(
        "event:mcap-no-proposals",
        0,
        ReductionPolicy(version="event-empty-v1", required_count=0),
        empty_semantics=StageStatus.SKIPPED_NOT_NEEDED.value,
    )

    aggregate = coordinator.get_aggregate_status(barrier.barrier_id)
    assert barrier.status == "CLOSED"
    assert coordinator.is_complete(barrier.barrier_id) is True
    assert aggregate.overall_status is StageStatus.SKIPPED_NOT_NEEDED
    assert aggregate.is_complete is True


def test_zero_child_barrier_can_fail_with_explicit_terminal_semantics() -> None:
    coordinator, _ = _coordinator()
    barrier = coordinator.create_barrier(
        "event:mcap-empty-failure",
        0,
        ReductionPolicy(version="event-empty-v1", required_count=0),
        empty_semantics=StageStatus.FAILED.value,
    )

    assert barrier.status == "FAILED"
    assert coordinator.is_complete(barrier.barrier_id) is True
    assert coordinator.get_aggregate_status(barrier.barrier_id).overall_status is StageStatus.FAILED


@pytest.mark.parametrize("empty_semantics", ["UNKNOWN", StageStatus.PENDING, StageStatus.RUNNING])
def test_zero_child_barrier_rejects_unknown_or_nonterminal_empty_semantics(
    empty_semantics: str,
) -> None:
    coordinator, _ = _coordinator()

    with pytest.raises(ValueError, match="terminal StageStatus"):
        coordinator.create_barrier(
            "event:mcap-invalid-empty-semantics",
            0,
            ReductionPolicy(version="event-empty-v1", required_count=0),
            empty_semantics=empty_semantics,
        )


def test_conflicting_or_excess_member_replay_fails_closed() -> None:
    coordinator, _ = _coordinator()
    barrier = coordinator.create_barrier(
        "qa:mcap-conflict",
        1,
        ReductionPolicy(version="qa-six-v1", required_count=1),
    )
    coordinator.submit_member(barrier.barrier_id, "cam-01", StageStatus.SUCCEEDED)

    with pytest.raises(ValueError, match="conflicting replay"):
        coordinator.submit_member(barrier.barrier_id, "cam-01", StageStatus.FAILED)
    with pytest.raises(ValueError, match="capacity exceeded"):
        coordinator.submit_member(barrier.barrier_id, "cam-02", StageStatus.SUCCEEDED)
