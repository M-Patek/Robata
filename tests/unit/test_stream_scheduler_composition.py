from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import UUID

import pytest

from robata.adapters.sqlite_stream_work_ledger import SQLiteStreamWorkLedgerConflict
from robata.adapters.sqlite_work_scheduler import (
    SQLiteWorkScheduler,
    WorkFenceError,
    WorkNotFoundError,
)
from robata.application.canonical.bounded_media import (
    BoundedWindowPlan,
    CameraStreamFacts,
    CameraWindowPlan,
    PlannerFinish,
    WindowClosureReason,
    WindowMember,
)
from robata.application.canonical.stream_scheduler import (
    DEFAULT_STREAM_BACKPRESSURE_CONFIG,
    DurableStreamWindowScheduler,
    EosSealInputs,
    StreamBackpressureThrottle,
    StreamSchedulerCompositionError,
    StreamSchedulerSchemaRefs,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import NanosecondInterval
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    AuthorityBinding,
    CameraAbsenceReason,
    ChannelBinding,
    StreamPolicyBinding,
    StreamPurpose,
    StreamStage,
    TerminalOutcome,
)
from robata.contracts.stream_planning import create_expected_window_plan
from robata.contracts.stream_source import PreEosCaptureSubject, create_pre_eos_capture_subject
from robata.queue.backpressure import BackpressureConfig, PressureClass
from robata.queue.models import WorkItemState
from robata.queue.stage import DependencyCriticality
from robata.queue.stream_models import (
    StreamTerminalEvidence,
    StreamWorkItemState,
    StreamWorkLease,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _schema(value: int) -> SchemaRef:
    return SchemaRef(
        schema_id=f"https://schemas.robata.dev/test-{value}",
        version="1.0.0",
        artifact_id=_uuid(1000 + value),
        sha256=_digest(2000 + value),
    )


def _capture() -> PreEosCaptureSubject:
    authority = AuthorityBinding(
        authority_id="authority",
        authority_epoch=1,
        policy_version="authority-v1",
        initial_binding_semantic_sha256=_digest(10),
    )
    return create_pre_eos_capture_subject(
        schema_ref=_schema(1),
        capture_authority_id="capture-authority",
        capture_authority_epoch=1,
        capture_assignment_policy_version="capture-assignment-v1",
        acquisition_id="acquisition-1",
        acquisition_epoch=1,
        channel_bindings=tuple(
            ChannelBinding(
                camera_id=camera_id,
                source_channel_id=f"channel-{camera_id.value}",
                source_channel_epoch=1,
                channel_binding_semantic_sha256=_digest(20 + index),
            )
            for index, camera_id in enumerate(CAMERA_IDS)
        ),
        mapping_authority=authority,
        clock_authority=authority,
    )


def _schema_refs() -> StreamSchedulerSchemaRefs:
    return StreamSchedulerSchemaRefs(
        incremental_window=_schema(2),
        expected_declaration=_schema(3),
        expected_plan_seal=_schema(4),
        stream_work_plan=_schema(5),
        terminal_member=_schema(6),
        terminal_closure=_schema(7),
    )


def _window(capture: PreEosCaptureSubject, ordinal: int) -> BoundedWindowPlan:
    start = ordinal * 1_000_000_000
    interval = NanosecondInterval(start_ns=start, end_ns=start + 1_000_000_000)
    camera_plans = tuple(
        CameraWindowPlan(
            camera_id=camera_id,
            members=(
                WindowMember(
                    camera_id=camera_id,
                    interval=interval,
                    absence_reason=CameraAbsenceReason.ABSENT,
                    absence_evidence_sha256=_digest(100 + ordinal * 10 + index),
                ),
            ),
        )
        for index, camera_id in enumerate(CAMERA_IDS)
    )
    return BoundedWindowPlan(
        ordinal=ordinal,
        requested_interval=interval,
        effective_interval=interval,
        camera_plans=camera_plans,
        quality_targets=(),
        quality_gaps=(),
        watermark_ns=interval.end_ns + 300_000_000,
        closure_reason=WindowClosureReason.WATERMARK,
        capture_scope_digest=capture.capture_scope_digest,
        mapping_semantic_sha256=_digest(30),
        clock_or_alignment_semantic_sha256=_digest(31),
        window_policy_version="window-v1",
        quality_policy_version="quality-v1",
        purpose=StreamPurpose.QA_COARSE,
    )


def _finish() -> PlannerFinish:
    return PlannerFinish(
        closed_segments=(),
        quality_targets=(),
        windows=(),
        facts=tuple(
            CameraStreamFacts(
                camera_id=camera_id,
                packet_count=0,
                payload_bytes=0,
                first_timestamp_ns=None,
                last_timestamp_ns=None,
                first_sequence=None,
                last_sequence=None,
                sequence_gap_count=0,
            )
            for camera_id in CAMERA_IDS
        ),
    )


def _eos(duration_ns: int) -> EosSealInputs:
    return EosSealInputs(
        eos_source_receipt_semantic_sha256=_digest(40),
        final_source_timeline_semantic_sha256=_digest(41),
        final_duration_ns=duration_ns,
        ordered_six_channel_health_closure_sha256=_digest(42),
        mapping_closure_semantic_sha256=_digest(43),
        clock_or_alignment_closure_semantic_sha256=_digest(44),
    )


def _composition(
    root: Path,
    *,
    capture: PreEosCaptureSubject,
    scheduler: SQLiteWorkScheduler,
    observer=None,
    planner_version: str = "planner-v1",
    stream_run_id: int = 60,
    database_path: Path | None = None,
    backpressure_config: BackpressureConfig | None = None,
) -> DurableStreamWindowScheduler:
    policy = StreamPolicyBinding(version="policy-v1", semantic_sha256=_digest(50))
    plan = create_expected_window_plan(
        schema_ref=_schema(8),
        capture_scope_digest=capture.capture_scope_digest,
        segmentation_policy=policy,
        window_policy=policy,
        watermark_policy=policy,
        lateness_policy=policy,
        idle_source_policy=policy,
        planner_version=planner_version,
    )
    return DurableStreamWindowScheduler(
        database_path=scheduler.database_path if database_path is None else database_path,
        execution_scheduler=scheduler,
        expected_plan=plan,
        source_subject=capture.reference(),
        stream_run_id=_uuid(stream_run_id),
        schema_refs=_schema_refs(),
        dag_config_semantic_sha256=_digest(61),
        backpressure_config=(
            DEFAULT_STREAM_BACKPRESSURE_CONFIG
            if backpressure_config is None
            else backpressure_config
        ),
        clock=lambda: _NOW,
        boundary_observer=observer,
    )


def _terminal_evidence(
    value: int,
    completed_at: datetime,
    *,
    outcome: TerminalOutcome = TerminalOutcome.SUCCEEDED,
) -> StreamTerminalEvidence:
    return StreamTerminalEvidence(
        outcome=outcome,
        evidence_ref={
            "artifact_id": _uuid(200 + value),
            "exact_sha256": _digest(300 + value),
            "byte_count": value + 1,
            "media_type": "application/json",
            "schema_ref": _schema(20 + value),
        },
        terminal_policy_version="stream-terminal-policy-v1",
        completed_at=completed_at.isoformat(),
    )


def _plan_for(
    composition: DurableStreamWindowScheduler,
    stage: StreamStage,
    ordinal: int = 0,
):
    plans = [plan for plan in composition.work_plans() if plan.stage is stage]
    return plans[ordinal]


def _complete(
    composition: DurableStreamWindowScheduler,
    stage: StreamStage,
    *,
    base_seconds: int,
    outcome: TerminalOutcome = TerminalOutcome.SUCCEEDED,
) -> None:
    plan = _plan_for(composition, stage)
    claim = composition.claim(
        "worker",
        30,
        work_item_id=plan.work_item_id,
        now=_NOW + timedelta(seconds=base_seconds),
    )
    assert claim is not None
    composition.start(claim.lease, now=_NOW + timedelta(seconds=base_seconds + 1))
    completed = composition.complete(
        claim.lease,
        _terminal_evidence(
            base_seconds,
            _NOW + timedelta(seconds=base_seconds + 2),
            outcome=outcome,
        ),
        now=_NOW + timedelta(seconds=base_seconds + 2),
    )
    expected_state = {
        TerminalOutcome.SUCCEEDED: StreamWorkItemState.SUCCEEDED,
        TerminalOutcome.SKIPPED_NOT_NEEDED: StreamWorkItemState.SKIPPED_NOT_NEEDED,
    }[outcome]
    assert completed.state is expected_state


def _complete_window_chain(
    composition: DurableStreamWindowScheduler,
    *,
    start_seconds: int = 0,
) -> None:
    for offset, stage in enumerate(
        (
            StreamStage.WINDOW,
            StreamStage.QA_COARSE,
            StreamStage.QA_DENSE,
            StreamStage.EVENT_PROPOSAL,
            StreamStage.WINDOW_REDUCTION,
        )
    ):
        _complete(composition, stage, base_seconds=start_seconds + offset * 3)


def test_declaration_is_durable_before_child_projection_and_restart_recovers(
    tmp_path: Path,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")

    def crash(boundary: str) -> None:
        if boundary == "expected_declaration_durable":
            raise RuntimeError("injected after declaration")

    composition = _composition(
        tmp_path,
        capture=capture,
        scheduler=execution,
        observer=crash,
    )
    with pytest.raises(RuntimeError, match="injected"):
        composition.append_window(_window(capture, 0))

    with sqlite3.connect(tmp_path / "work.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM expected_windows").fetchone()[0] == 1
        root_id = connection.execute(
            "SELECT work_item_id FROM stream_work_plans ORDER BY role_order LIMIT 1"
        ).fetchone()[0]
    with pytest.raises(WorkNotFoundError):
        execution.get(root_id)

    reopened = _composition(tmp_path, capture=capture, scheduler=execution)
    assert len(reopened.declarations()) == 1
    assert len(reopened.work_plans()) == 5
    assert execution.get(root_id).state is WorkItemState.READY
    assert reopened.append_window(_window(capture, 0)) == reopened.declarations()[0]


def test_projection_crash_before_batch_publish_replays_once_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)

    def crash_before_publish(_work_item_ids: object) -> int:
        raise RuntimeError("injected after execution batch projection")

    monkeypatch.setattr(composition._ledger, "mark_published_many", crash_before_publish)
    with pytest.raises(RuntimeError, match="after execution batch projection"):
        composition.append_window(_window(capture, 0))

    with sqlite3.connect(execution.database_path) as connection:
        publication_states = tuple(
            row[0]
            for row in connection.execute(
                "SELECT publication_state FROM stream_work_plans ORDER BY role_order"
            )
        )
        projected_count = connection.execute("SELECT COUNT(*) FROM work_items").fetchone()[0]
    assert publication_states == ("PENDING",) * 5
    assert projected_count == 5

    reopened = _composition(tmp_path, capture=capture, scheduler=execution)
    root = _plan_for(reopened, StreamStage.WINDOW)
    assert execution.get(root.work_item_id).state is WorkItemState.READY
    assert reopened.recover() == 0
    with sqlite3.connect(execution.database_path) as connection:
        publication_states = tuple(
            row[0]
            for row in connection.execute(
                "SELECT publication_state FROM stream_work_plans ORDER BY role_order"
            )
        )
        projected_count = connection.execute("SELECT COUNT(*) FROM work_items").fetchone()[0]
    assert publication_states == ("PUBLISHED",) * 5
    assert projected_count == 5


def test_hot_stream_reads_do_not_scan_pending_terminal_intents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    root = _plan_for(composition, StreamStage.WINDOW)

    def reject_pending_terminal_scan(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("normal hot reads must not scan all pending terminal intents")

    monkeypatch.setattr(composition._ledger, "pending_work_rows", reject_pending_terminal_scan)
    assert composition.get(root.work_item_id).state is StreamWorkItemState.READY
    assert composition.backlog(now=_NOW).active_backlog == 5
    assert len(composition.bounded_drain_scope(1)) == 1
    assert len(composition.work_items(recover_graph=False)) == 5


def test_backpressure_throttles_only_new_windows_and_recovers_same_policy(
    tmp_path: Path,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    policy = BackpressureConfig(
        version="test-stream-pressure-v1",
        queue_depth_threshold=0,
        oldest_age_threshold_ms=60_000,
        backlog_slope_threshold=100.0,
    )
    composition = _composition(
        tmp_path,
        capture=capture,
        scheduler=execution,
        backpressure_config=policy,
    )

    first = composition.append_window(_window(capture, 0))
    pressure = composition.backpressure_snapshot()
    assert pressure.decision.pressure_class is PressureClass.THROTTLED
    assert pressure.decision.signals == ("QUEUE_DEPTH",)
    assert pressure.decision.shedding_actions == ("THROTTLE_LEDGER",)

    with pytest.raises(StreamBackpressureThrottle) as caught:
        composition.append_window(_window(capture, 1))
    assert caught.value.decision.policy_version == policy.version
    assert composition.declarations() == (first,)
    assert len(composition.work_plans()) == 5

    recovered = DurableStreamWindowScheduler.recover_registered(
        execution_scheduler=execution,
        stream_run_id=_uuid(60),
        clock=lambda: _NOW,
    )[0]
    assert recovered.backpressure_snapshot().decision.pressure_class is PressureClass.THROTTLED
    assert recovered.append_window(_window(capture, 0)) == first

    _complete_window_chain(recovered)
    assert recovered.backpressure_snapshot().decision.pressure_class is PressureClass.NORMAL
    recovered.append_window(_window(capture, 1))
    assert len(recovered.declarations()) == 2


def test_composition_requires_scheduler_authority_database_path(tmp_path: Path) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")

    with pytest.raises(ValueError, match="share database_path"):
        _composition(
            tmp_path,
            capture=capture,
            scheduler=execution,
            database_path=tmp_path / "other.sqlite3",
        )


def test_existing_ordinal_replay_rejects_changed_watermark_source_facts(
    tmp_path: Path,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    original = _window(capture, 0)
    composition.append_window(original)
    changed = replace(
        original,
        watermark_ns=None,
        closure_reason=WindowClosureReason.EOS,
    )

    with pytest.raises(StreamSchedulerCompositionError, match="source facts"):
        composition.append_window(changed)
    assert len(composition.declarations()) == 1


def test_append_is_rejected_after_planner_eos(tmp_path: Path) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    first = _window(capture, 0)
    composition.append_window(first)
    composition.seal(_finish())

    assert composition.append_window(first) == composition.declarations()[0]
    with pytest.raises(SQLiteStreamWorkLedgerConflict, match="planner EOS"):
        composition.append_window(_window(capture, 1))
    assert len(composition.declarations()) == 1


def test_append_and_planner_eos_are_serialized_in_one_authority_database(
    tmp_path: Path,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    barrier = Barrier(2)

    def append_once() -> RuntimeError | None:
        barrier.wait()
        try:
            composition.append_window(_window(capture, 0))
        except RuntimeError as error:
            return error
        return None

    def seal_once() -> RuntimeError | None:
        barrier.wait()
        try:
            composition.seal(_finish())
        except RuntimeError as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        append_future = pool.submit(append_once)
        seal_future = pool.submit(seal_once)
        append_error = append_future.result()
        seal_error = seal_future.result()

    assert seal_error is None
    assert append_error is None or isinstance(append_error, SQLiteStreamWorkLedgerConflict)
    with sqlite3.connect(execution.database_path) as connection:
        planner_eos, window_count = connection.execute(
            """
            SELECT planner_eos_sha256,
                   (SELECT COUNT(*) FROM expected_windows)
            FROM stream_plans
            """
        ).fetchone()
    assert planner_eos is not None
    assert window_count == (1 if append_error is None else 0)


def test_start_and_heartbeat_reject_work_from_another_stream_graph(
    tmp_path: Path,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    first = _composition(tmp_path, capture=capture, scheduler=execution)
    second = _composition(
        tmp_path,
        capture=capture,
        scheduler=execution,
        planner_version="planner-v2",
        stream_run_id=62,
    )
    first.append_window(_window(capture, 0))
    second.append_window(_window(capture, 0))
    foreign = _plan_for(second, StreamStage.WINDOW)
    claim = second.claim("worker", 20, work_item_id=foreign.work_item_id, now=_NOW)
    assert claim is not None
    before = execution.get(foreign.work_item_id)

    with pytest.raises(StreamSchedulerCompositionError, match="another stream graph"):
        first.start(claim.lease, now=_NOW + timedelta(seconds=1))
    with pytest.raises(StreamSchedulerCompositionError, match="another stream graph"):
        first.heartbeat(claim.lease, 20, now=_NOW + timedelta(seconds=1))

    after = execution.get(foreign.work_item_id)
    assert after.state is WorkItemState.LEASED
    assert after.row_version == before.row_version
    assert after.lease_expires_at == before.lease_expires_at


def test_stale_fence_cannot_commit_stream_terminal_fact(tmp_path: Path) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    root = _plan_for(composition, StreamStage.WINDOW)

    old = composition.claim("old-worker", 2, work_item_id=root.work_item_id, now=_NOW)
    assert old is not None
    composition.start(old.lease, now=_NOW + timedelta(seconds=1))
    new = composition.claim(
        "new-worker",
        20,
        work_item_id=root.work_item_id,
        now=_NOW + timedelta(seconds=3),
    )
    assert new is not None
    composition.start(new.lease, now=_NOW + timedelta(seconds=4))

    with pytest.raises(WorkFenceError):
        composition.complete(
            old.lease,
            _terminal_evidence(1, _NOW + timedelta(seconds=5)),
            now=_NOW + timedelta(seconds=5),
        )
    completed = composition.complete(
        new.lease,
        _terminal_evidence(2, _NOW + timedelta(seconds=6)),
        now=_NOW + timedelta(seconds=6),
    )
    assert completed.state is StreamWorkItemState.SUCCEEDED
    assert completed.terminal_evidence == _terminal_evidence(2, _NOW + timedelta(seconds=6))


def test_forged_future_fence_does_not_poison_pending_terminal(tmp_path: Path) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    root = _plan_for(composition, StreamStage.WINDOW)

    claim = composition.claim("worker", 20, work_item_id=root.work_item_id, now=_NOW)
    assert claim is not None
    composition.start(claim.lease, now=_NOW + timedelta(seconds=1))
    forged = StreamWorkLease(
        work_item_id=claim.lease.work_item_id,
        worker_id=claim.lease.worker_id,
        lease_epoch=claim.lease.lease_epoch + 100,
        fencing_token=_uuid(9999),
        lease_expires_at=claim.lease.lease_expires_at,
    )

    with pytest.raises(WorkFenceError):
        composition.complete(
            forged,
            _terminal_evidence(3, _NOW + timedelta(seconds=2)),
            now=_NOW + timedelta(seconds=2),
        )

    with sqlite3.connect(tmp_path / "work.sqlite3") as connection:
        pending = connection.execute(
            """
            SELECT pending_terminal_json, pending_lease_epoch, pending_fencing_token
            FROM stream_work_plans WHERE work_item_id = ?
            """,
            (root.work_item_id,),
        ).fetchone()
    assert pending == (None, None, None)

    evidence = _terminal_evidence(4, _NOW + timedelta(seconds=3))
    completed = composition.complete(
        claim.lease,
        evidence,
        now=_NOW + timedelta(seconds=3),
    )
    assert completed.state is StreamWorkItemState.SUCCEEDED
    assert completed.terminal_evidence == evidence


def test_forged_capability_expiry_cannot_create_pending_terminal(
    tmp_path: Path,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    root = _plan_for(composition, StreamStage.WINDOW)
    claim = composition.claim("worker", 20, work_item_id=root.work_item_id, now=_NOW)
    assert claim is not None
    composition.start(claim.lease, now=_NOW + timedelta(seconds=1))
    forged = claim.lease.model_copy(
        update={"lease_expires_at": (_NOW + timedelta(days=1)).isoformat()}
    )

    with pytest.raises(WorkFenceError, match="stale, expired, or inactive"):
        composition.complete(
            forged,
            _terminal_evidence(8, _NOW + timedelta(seconds=2)),
            now=_NOW + timedelta(seconds=2),
        )

    with sqlite3.connect(execution.database_path) as connection:
        pending = connection.execute(
            """
            SELECT pending_terminal_json, pending_lease_epoch, pending_fencing_token
            FROM stream_work_plans WHERE work_item_id = ?
            """,
            (root.work_item_id,),
        ).fetchone()
    assert pending == (None, None, None)

def test_backdated_evidence_cannot_revive_expired_authority_lease(
    tmp_path: Path,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    root = _plan_for(composition, StreamStage.WINDOW)
    claim = composition.claim("worker", 2, work_item_id=root.work_item_id, now=_NOW)
    assert claim is not None
    composition.start(claim.lease, now=_NOW + timedelta(seconds=1))

    with pytest.raises(WorkFenceError):
        composition.complete(
            claim.lease,
            _terminal_evidence(5, _NOW + timedelta(seconds=1)),
            now=_NOW + timedelta(seconds=3),
        )

    with sqlite3.connect(execution.database_path) as connection:
        pending = connection.execute(
            """
            SELECT pending_terminal_json, pending_lease_epoch, pending_fencing_token
            FROM stream_work_plans WHERE work_item_id = ?
            """,
            (root.work_item_id,),
        ).fetchone()
    assert pending == (None, None, None)


def test_terminal_policy_mismatch_is_rejected_before_pending_intent(
    tmp_path: Path,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    root = _plan_for(composition, StreamStage.WINDOW)
    claim = composition.claim("worker", 20, work_item_id=root.work_item_id, now=_NOW)
    assert claim is not None
    composition.start(claim.lease, now=_NOW + timedelta(seconds=1))
    mismatched = _terminal_evidence(6, _NOW + timedelta(seconds=2)).model_copy(
        update={"terminal_policy_version": "other-policy-v1"}
    )

    with pytest.raises(StreamSchedulerCompositionError, match="policy pin"):
        composition.complete(
            claim.lease,
            mismatched,
            now=_NOW + timedelta(seconds=2),
        )

    evidence = _terminal_evidence(7, _NOW + timedelta(seconds=3))
    completed = composition.complete(
        claim.lease,
        evidence,
        now=_NOW + timedelta(seconds=3),
    )
    assert completed.terminal_evidence == evidence


def test_terminal_execution_crash_is_reconciled_without_redispatch(tmp_path: Path) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    crashed = False
    armed = False

    def crash(boundary: str) -> None:
        nonlocal crashed, armed
        if boundary == "execution_terminal_committed" and armed and not crashed:
            crashed = True
            raise RuntimeError("injected after execution commit")

    composition = _composition(tmp_path, capture=capture, scheduler=execution, observer=crash)
    composition.append_window(_window(capture, 0))
    _complete(composition, StreamStage.WINDOW, base_seconds=0)
    _complete(composition, StreamStage.QA_COARSE, base_seconds=3)
    _complete(composition, StreamStage.QA_DENSE, base_seconds=6)
    _complete(composition, StreamStage.EVENT_PROPOSAL, base_seconds=9)
    armed = True
    reduction = _plan_for(composition, StreamStage.WINDOW_REDUCTION)
    claim = composition.claim(
        "worker", 30, work_item_id=reduction.work_item_id, now=_NOW + timedelta(seconds=12)
    )
    assert claim is not None
    composition.start(claim.lease, now=_NOW + timedelta(seconds=13))
    with pytest.raises(RuntimeError, match="execution commit"):
        composition.complete(
            claim.lease,
            _terminal_evidence(21, _NOW + timedelta(seconds=14)),
            now=_NOW + timedelta(seconds=14),
        )
    assert execution.get(reduction.work_item_id).state is WorkItemState.SUCCEEDED

    with sqlite3.connect(execution.database_path) as connection:
        connection.execute("DROP INDEX stream_work_pending_terminal")

    reopened = _composition(tmp_path, capture=capture, scheduler=execution)
    assert reopened.get(reduction.work_item_id).state is StreamWorkItemState.SUCCEEDED
    assert len(reopened.terminal_members()) == 1
    assert reopened.terminal_member_count() == 1
    assert reopened.terminal_member_at(0) == reopened.terminal_members()[0]
    assert reopened.terminal_member_at(1) is None
    assert len(execution.list_attempts(reduction.work_item_id)) == 1

    with sqlite3.connect(execution.database_path) as connection:
        index_sql = connection.execute(
            """
            SELECT sql FROM sqlite_schema
            WHERE type = 'index' AND name = 'stream_work_pending_terminal'
            """
        ).fetchone()
        query_plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT work_item_id FROM stream_work_plans
            WHERE plan_key = ? AND pending_terminal_json IS NOT NULL
            ORDER BY work_item_id
            """,
            (reopened.plan_key,),
        ).fetchall()
    assert index_sql is not None and "WHERE pending_terminal_json IS NOT NULL" in index_sql[0]
    assert any("stream_work_pending_terminal" in row[3] for row in query_plan)


def test_startup_audit_rejects_work_row_column_drift(tmp_path: Path) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    with sqlite3.connect(execution.database_path) as connection:
        connection.execute(
            """
            UPDATE stream_work_plans SET role_order = 99
            WHERE expected_ordinal = 0 AND role_order = 0
            """
        )

    with pytest.raises(StreamSchedulerCompositionError, match="canonical DAG plan"):
        _composition(tmp_path, capture=capture, scheduler=execution)


def test_seal_replay_requires_exact_finalization_companion(tmp_path: Path) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    composition.seal(_finish())
    composition.finalize_eos(_eos(1_000_000_000))
    with sqlite3.connect(execution.database_path) as connection:
        connection.execute("DELETE FROM stream_work_plans WHERE expected_ordinal IS NULL")

    with pytest.raises(StreamSchedulerCompositionError, match="finalization work"):
        composition.finalize_eos(_eos(1_000_000_000))


def test_omitted_declared_member_keeps_finalization_gated(tmp_path: Path) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    composition.append_window(_window(capture, 1))
    composition.seal(_finish())
    composition.finalize_eos(_eos(2_000_000_000))
    composition.mark_export_barrier_complete(
        export_manifest_semantic_sha256=_digest(70), completed_member_count=6
    )

    with pytest.raises(StreamSchedulerCompositionError, match="missing declared ordinals"):
        composition.close_finalization_gate()
    finalization = _plan_for(composition, StreamStage.FINALIZATION)
    with pytest.raises(WorkNotFoundError):
        execution.get(finalization.work_item_id)
    snapshot = composition.backlog(now=_NOW + timedelta(seconds=10))
    assert snapshot.declared_window_count == 2
    assert snapshot.expected_plan_sealed
    assert snapshot.terminal_member_count == 0
    assert snapshot.export_barrier_complete
    assert not snapshot.finalization_published
    assert ("GATED", 1) in snapshot.state_counts


def test_complete_barriers_publish_ready_finalization_and_query_age(tmp_path: Path) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))
    composition.seal(_finish())
    composition.finalize_eos(_eos(1_000_000_000))

    _complete_window_chain(composition)
    composition.mark_export_barrier_complete(
        export_manifest_semantic_sha256=_digest(71), completed_member_count=6
    )
    closure = composition.close_finalization_gate()
    assert closure.complete
    assert composition.terminal_closure() == closure

    finalization = _plan_for(composition, StreamStage.FINALIZATION)
    assert execution.get(finalization.work_item_id).state is WorkItemState.READY
    snapshot = composition.backlog(now=_NOW + timedelta(seconds=25))
    assert snapshot.finalization_published
    assert snapshot.active_backlog == 1
    assert snapshot.oldest_active_age_seconds == 25.0


def test_full_window_dag_has_deadlines_and_governed_skip_releases_downstream(
    tmp_path: Path,
) -> None:
    capture = _capture()
    execution = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    composition = _composition(tmp_path, capture=capture, scheduler=execution)
    composition.append_window(_window(capture, 0))

    plans = composition.work_plans()
    assert tuple(plan.stage for plan in plans) == (
        StreamStage.WINDOW,
        StreamStage.QA_COARSE,
        StreamStage.QA_DENSE,
        StreamStage.EVENT_PROPOSAL,
        StreamStage.WINDOW_REDUCTION,
    )
    assert all(plan.sla_deadline_at is not None for plan in plans)
    assert all(plan.execution_expiry_at is None for plan in plans)
    assert plans[0].ordered_dependencies == ()
    by_stage = {plan.stage: plan for plan in plans}
    assert {
        dependency.upstream_work_logical_key
        for dependency in by_stage[StreamStage.EVENT_PROPOSAL].ordered_dependencies
    } == {
        by_stage[StreamStage.QA_COARSE].work_logical_key,
        by_stage[StreamStage.QA_DENSE].work_logical_key,
    }
    assert {
        dependency.upstream_work_logical_key
        for dependency in by_stage[StreamStage.WINDOW_REDUCTION].ordered_dependencies
    } == {
        by_stage[stage].work_logical_key
        for stage in (
            StreamStage.WINDOW,
            StreamStage.QA_COARSE,
            StreamStage.QA_DENSE,
            StreamStage.EVENT_PROPOSAL,
        )
    }
    assert all(
        dependency.criticality is DependencyCriticality.DEGRADABLE
        for plan in plans
        for dependency in plan.ordered_dependencies
    )

    _complete(composition, StreamStage.WINDOW, base_seconds=0)
    _complete(composition, StreamStage.QA_COARSE, base_seconds=3)
    _complete(
        composition,
        StreamStage.QA_DENSE,
        base_seconds=6,
        outcome=TerminalOutcome.SKIPPED_NOT_NEEDED,
    )
    event = _plan_for(composition, StreamStage.EVENT_PROPOSAL)
    assert execution.get(event.work_item_id).state is WorkItemState.READY
