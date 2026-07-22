from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.adapters.sqlite_barrier import SQLiteBarrierStorage
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler
from robata.contracts.schema_registry import SchemaPinMismatchError, SchemaRegistry
from robata.queue.barrier import BarrierCoordinator, ReductionPolicy
from robata.queue.models import (
    WorkDependency,
    WorkItemPlan,
    WorkItemSubjectType,
)
from robata.queue.stage import DependencyCriticality, Stage, StageStatus
from robata.queue.wire import (
    PERSISTED_BARRIER_SCHEMA_ID,
    PERSISTED_BARRIER_SCHEMA_VERSION,
    WORK_MESSAGE_SCHEMA_ID,
    WORK_MESSAGE_SCHEMA_VERSION,
    PersistedBarrier,
    WorkMessage,
    validate_registered_work_message,
)

_NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)
_RUN_ID = str(UUID(int=1))
_MCAP_ID = str(UUID(int=2))


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _plan(value: int) -> WorkItemPlan:
    return WorkItemPlan(
        schema_version="1.0",
        work_item_id=_uuid(100 + value),
        work_logical_key=f"work:{value}",
        run_id=_RUN_ID,
        mcap_id=_MCAP_ID,
        stage=Stage.QA_COARSE_PLAN,
        subject_type=WorkItemSubjectType.MCAP,
        subject_id=_uuid(200 + value),
        input_digest=f"{value:064x}",
        config_digest=f"{value + 10:064x}",
        priority=value,
        sla_deadline_at=(_NOW + timedelta(hours=1)).isoformat(),
        execution_expiry_at=(_NOW + timedelta(hours=2)).isoformat(),
        max_attempts=3,
        trace_id=f"trace-{value}",
        created_at=_NOW.isoformat(),
    )


def test_work_message_projects_only_an_active_authoritative_lease(tmp_path: Path) -> None:
    registry = SchemaRegistry()
    reference = registry.resolve_version(
        WORK_MESSAGE_SCHEMA_ID,
        WORK_MESSAGE_SCHEMA_VERSION,
    ).ref
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    upstream_plan = _plan(1)
    downstream_plan = _plan(2)
    scheduler.plan(upstream_plan)
    dependency = WorkDependency(
        dependency_id=_uuid(301),
        downstream_work_item_id=downstream_plan.work_item_id,
        upstream_work_item_id=upstream_plan.work_item_id,
        criticality=DependencyCriticality.REQUIRED,
    )
    ready = scheduler.plan(downstream_plan, (dependency,))

    with pytest.raises(ValueError, match="leased or running"):
        WorkMessage.from_ledger(
            ready,
            scheduler.dependencies(ready.work_item_id),
            schema_ref=reference,
        )

    upstream_claim = scheduler.claim("worker-a", 30, now=_NOW)
    assert upstream_claim is not None
    scheduler.start(upstream_claim.lease, now=_NOW + timedelta(seconds=1))
    scheduler.succeed(upstream_claim.lease, now=_NOW + timedelta(seconds=2))
    downstream_claim = scheduler.claim("worker-b", 30, now=_NOW + timedelta(seconds=3))
    assert downstream_claim is not None
    assert downstream_claim.work_item.work_item_id == downstream_plan.work_item_id

    message = WorkMessage.from_ledger(
        downstream_claim.work_item,
        scheduler.dependencies(downstream_plan.work_item_id),
        schema_ref=reference,
    )

    assert message.fencing_token == downstream_claim.lease.fencing_token
    assert message.lease_epoch == downstream_claim.lease.lease_epoch
    assert message.attempt == 1
    assert tuple(item.work_item_id for item in message.dependencies) == (
        upstream_plan.work_item_id,
    )
    assert validate_registered_work_message(message, registry) == message

    forged = message.model_copy(
        update={"schema_ref": reference.model_copy(update={"sha256": "0" * 64})}
    )
    with pytest.raises(SchemaPinMismatchError):
        validate_registered_work_message(forged, registry)


def test_sqlite_exposes_versioned_atomic_persisted_barrier_snapshot(
    tmp_path: Path,
) -> None:
    registry = SchemaRegistry()
    reference = registry.resolve_version(
        PERSISTED_BARRIER_SCHEMA_ID,
        PERSISTED_BARRIER_SCHEMA_VERSION,
    ).ref
    database = tmp_path / "barrier.sqlite3"
    storage = SQLiteBarrierStorage(database)
    coordinator = BarrierCoordinator(storage)
    barrier = coordinator.create_barrier(
        "qa:recording-1",
        2,
        ReductionPolicy(version="qa-reduce-v1", required_count=1, degradable_count=1),
    )

    initial = storage.get_persisted_barrier_snapshot(
        barrier.barrier_id,
        schema_ref=reference,
    )
    assert initial is not None
    assert initial.state_version == 0
    assert initial.status == "OPEN"
    assert initial.members == ()

    coordinator.submit_member(
        barrier.barrier_id,
        "work-b",
        StageStatus.FAILED,
        DependencyCriticality.DEGRADABLE,
    )
    coordinator.submit_member(
        barrier.barrier_id,
        "work-a",
        StageStatus.SUCCEEDED,
        DependencyCriticality.REQUIRED,
    )
    completed = SQLiteBarrierStorage(database).get_persisted_barrier_snapshot(
        barrier.barrier_id,
        schema_ref=reference,
    )

    assert completed is not None
    assert completed.state_version == 2
    assert completed.status == "CLOSED"
    assert completed.completed_members == 2
    assert completed.pending_members == 0
    assert completed.failed_members == 1
    assert tuple(item.work_item_id for item in completed.members) == (
        "work-a",
        "work-b",
    )

    invalid = completed.model_dump(mode="python")
    invalid["completed_members"] = 1
    with pytest.raises(ValidationError, match="completed_members"):
        PersistedBarrier.model_validate(invalid, strict=True)


def test_registered_work_and_barrier_schemas_are_closed_exact_contracts() -> None:
    registry = SchemaRegistry()
    for schema_id, version in (
        (WORK_MESSAGE_SCHEMA_ID, WORK_MESSAGE_SCHEMA_VERSION),
        (PERSISTED_BARRIER_SCHEMA_ID, PERSISTED_BARRIER_SCHEMA_VERSION),
    ):
        registered = registry.resolve_version(schema_id, version)
        document = registry.get_schema(registered.ref)

        assert document["additionalProperties"] is False
        assert set(document["required"]) == set(document["properties"])
        assert document["properties"]["schema_version"]["const"] == "1.0"
        assert registered.entry.compatibility_mode.value == "NONE"
        assert registered.entry.supported_predecessors == ()
