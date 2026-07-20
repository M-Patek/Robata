"""Tests for deterministic local weighted-fair dispatching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from robata.queue.dispatcher import DispatcherConfig, StageDispatcher
from robata.queue.stage import Stage

WORK_ITEM_ID = "00000000-0000-5000-8000-000000000001"
RESERVATION_IDS = (
    "00000000-0000-5000-8000-000000000010",
    "00000000-0000-5000-8000-000000000011",
    "00000000-0000-5000-8000-000000000012",
    "00000000-0000-5000-8000-000000000013",
)


@dataclass
class _FakeClock:
    now: datetime

    def __call__(self) -> datetime:
        return self.now

    def advance_ms(self, milliseconds: int) -> None:
        self.now += timedelta(milliseconds=milliseconds)


def _dispatcher(
    *,
    max_concurrent: int = 4,
    ttl_ms: int = 1_000,
) -> tuple[StageDispatcher, _FakeClock]:
    clock = _FakeClock(datetime(2026, 7, 19, 12, 0, tzinfo=UTC))
    reservation_ids = iter(RESERVATION_IDS)
    config = DispatcherConfig(
        version="local-fair-v1",
        stage_weights={
            Stage.QA_COARSE_PLAN: 1.0,
            Stage.EVENT_PROPOSAL_PLAN: 2.0,
            Stage.VALUE_SCORE: 0.0,
        },
        max_concurrent=max_concurrent,
        fairness_window_ms=ttl_ms,
    )
    return (
        StageDispatcher(
            config,
            clock=clock,
            id_factory=lambda: next(reservation_ids),
        ),
        clock,
    )


def test_weighted_load_allows_two_event_slots_for_one_coarse_slot() -> None:
    dispatcher, _ = _dispatcher()
    assert dispatcher.dispatch(Stage.QA_COARSE_PLAN, WORK_ITEM_ID).accepted is True
    dispatcher.reserve_capacity(Stage.QA_COARSE_PLAN, 1.0)

    coarse_again = dispatcher.dispatch(Stage.QA_COARSE_PLAN, WORK_ITEM_ID)
    assert coarse_again.accepted is False
    assert coarse_again.reason == "stage is above its weighted fair share"

    dispatcher.reserve_capacity(Stage.EVENT_PROPOSAL_PLAN, 1.0)
    assert dispatcher.dispatch(Stage.EVENT_PROPOSAL_PLAN, WORK_ITEM_ID).accepted is True
    dispatcher.reserve_capacity(Stage.EVENT_PROPOSAL_PLAN, 1.0)

    assert dispatcher.dispatch(Stage.QA_COARSE_PLAN, WORK_ITEM_ID).accepted is True


def test_capacity_full_reports_wait_and_expiry_reopens_admission() -> None:
    dispatcher, clock = _dispatcher(max_concurrent=2, ttl_ms=750)
    first = dispatcher.reserve_capacity(Stage.QA_COARSE_PLAN, 1.0)
    dispatcher.reserve_capacity(Stage.EVENT_PROPOSAL_PLAN, 1.0)

    result = dispatcher.dispatch(Stage.EVENT_PROPOSAL_PLAN, WORK_ITEM_ID)
    assert result.accepted is False
    assert result.reason == "dispatcher capacity is full"
    assert result.estimated_wait_ms == 750
    assert first.reserved_at == "2026-07-19T12:00:00Z"
    assert first.expires_at == "2026-07-19T12:00:00.750000Z"

    clock.advance_ms(750)
    assert dispatcher.sweep_expired() == 2
    assert dispatcher.active_reservation_count == 0
    assert dispatcher.dispatch(Stage.EVENT_PROPOSAL_PLAN, WORK_ITEM_ID).accepted is True


def test_release_is_exact_idempotent_and_active_snapshot_is_stable() -> None:
    dispatcher, _ = _dispatcher()
    coarse = dispatcher.reserve_capacity(Stage.QA_COARSE_PLAN, 1.0)
    event = dispatcher.reserve_capacity(Stage.EVENT_PROPOSAL_PLAN, 1.0)

    assert dispatcher.list_active_reservations() == (coarse, event)
    dispatcher.release_capacity(coarse)
    dispatcher.release_capacity(coarse)

    assert dispatcher.active_reservation_count == 1
    assert dispatcher.list_active_reservations(Stage.EVENT_PROPOSAL_PLAN) == (event,)


def test_unknown_disabled_and_invalid_weights_fail_closed() -> None:
    dispatcher, _ = _dispatcher()

    unknown = dispatcher.dispatch(Stage.ALIGN, WORK_ITEM_ID)
    disabled = dispatcher.dispatch(Stage.VALUE_SCORE, WORK_ITEM_ID)
    invalid_id = dispatcher.dispatch(Stage.QA_COARSE_PLAN, "not-a-uuid")
    assert unknown.accepted is False and unknown.reason == "stage is not configured"
    assert disabled.accepted is False and disabled.reason == "stage weight is zero"
    assert invalid_id.accepted is False

    with pytest.raises(ValueError, match="stage is not configured"):
        dispatcher.reserve_capacity(Stage.ALIGN, 1.0)
    with pytest.raises(ValueError, match="stage weight is zero"):
        dispatcher.reserve_capacity(Stage.VALUE_SCORE, 1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        dispatcher.reserve_capacity(Stage.QA_COARSE_PLAN, 0.0)


def test_id_collision_and_naive_clock_fail_closed() -> None:
    repeated_id = RESERVATION_IDS[0]
    config = DispatcherConfig(
        version="local-fair-v1",
        stage_weights={Stage.QA_COARSE_PLAN: 1.0},
        max_concurrent=2,
        fairness_window_ms=1_000,
    )
    clock = _FakeClock(datetime(2026, 7, 19, 12, 0, tzinfo=UTC))
    dispatcher = StageDispatcher(config, clock=clock, id_factory=lambda: repeated_id)
    dispatcher.reserve_capacity(Stage.QA_COARSE_PLAN, 1.0)
    with pytest.raises(ValueError, match="ID collision"):
        dispatcher.reserve_capacity(Stage.QA_COARSE_PLAN, 1.0)

    naive = StageDispatcher(config, clock=lambda: datetime(2026, 7, 19, 12, 0))
    with pytest.raises(ValueError, match="timezone-aware"):
        naive.dispatch(Stage.QA_COARSE_PLAN, WORK_ITEM_ID)

    clock.now -= timedelta(seconds=1)
    with pytest.raises(ValueError, match="clock moved backwards"):
        dispatcher.dispatch(Stage.QA_COARSE_PLAN, WORK_ITEM_ID)
