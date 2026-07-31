from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from robata.adapters.postgres_authority import (
    PostgresAuthorityConfigurationError,
    PostgresCanonicalAuthority,
    active_postgres_authority_transaction_operation,
    require_outside_postgres_authority_transaction,
)
from robata.adapters.postgres_capture_authority import PostgresCaptureAuthority
from robata.adapters.postgres_stream_work_ledger import PostgresStreamWorkLedger
from robata.adapters.postgres_work_scheduler import PostgresWorkScheduler
from robata.adapters.sqlite_work_scheduler import WorkFenceError
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import AuthorityBinding, ChannelBinding
from robata.contracts.stream_source import PreEosCaptureSubject
from robata.queue.models import (
    WorkAttemptOutcome,
    WorkDependency,
    WorkItemPlan,
    WorkItemState,
    WorkItemSubjectType,
)
from robata.queue.stage import DependencyCriticality, Stage
from robata.runtime.observability import RuntimeProfileRecorder

_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_RUN_ID = "00000000-0000-4000-8000-000000000001"
_MCAP_ID = "00000000-0000-4000-8000-000000000002"


class _Cursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def fetchone(self) -> sqlite3.Row | None:
        return self._cursor.fetchone()

    def fetchall(self) -> Sequence[sqlite3.Row]:
        return self._cursor.fetchall()


class _EmptyCursor:
    rowcount = 0

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> tuple[object, ...]:
        return ()


class _PostgresSqliteHarness:
    """DB-API double that executes portable scheduler SQL against SQLite.

    It retains PostgreSQL placeholders and lock clauses in ``queries`` for assertions,
    while stripping only PostgreSQL-only transaction syntax for local semantic tests.
    """

    def __init__(self) -> None:
        self._connection = sqlite3.connect(":memory:", isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self.queries: list[str] = []
        self.begin_count = 0
        self.commit_count = 0
        self.rollback_count = 0
        _create_harness_schema(self._connection)

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> _Cursor | _EmptyCursor:
        self.queries.append(query)
        normalized = query.strip()
        if normalized.startswith("BEGIN"):
            self.begin_count += 1
            self._connection.execute("BEGIN")
            return _EmptyCursor()
        if (
            normalized.startswith("SET LOCAL")
            or normalized.startswith("SELECT set_config")
            or normalized.startswith("SELECT pg_advisory_xact_lock")
        ):
            return _EmptyCursor()
        portable = re.sub(r"\s+FOR UPDATE(?:\s+SKIP LOCKED)?", "", query)
        portable = portable.replace("IS NOT DISTINCT FROM", "IS")
        portable = portable.replace("%s", "?")
        return _Cursor(self._connection.execute(portable, () if params is None else params))

    def commit(self) -> None:
        self.commit_count += 1
        self._connection.commit()

    def rollback(self) -> None:
        self.rollback_count += 1
        self._connection.rollback()

    def close(self) -> None:
        # Authority operations intentionally use short-lived connections in production.
        # This semantic double keeps one in-memory database alive across those operations.
        return None


def _create_harness_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE work_items (
            tenant_id TEXT NOT NULL DEFAULT 'tenant-test',
            schema_version TEXT NOT NULL,
            work_item_id TEXT NOT NULL,
            work_logical_key TEXT NOT NULL,
            run_id TEXT NOT NULL,
            mcap_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            subject_type TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            input_digest TEXT NOT NULL,
            config_digest TEXT NOT NULL,
            priority INTEGER NOT NULL,
            sla_deadline_at TEXT,
            execution_expiry_at TEXT,
            max_attempts INTEGER NOT NULL,
            trace_id TEXT,
            created_at TEXT NOT NULL,
            state TEXT NOT NULL,
            cancel_requested INTEGER NOT NULL,
            lease_epoch INTEGER NOT NULL,
            fencing_token TEXT,
            leased_by TEXT,
            lease_expires_at TEXT,
            attempt INTEGER NOT NULL,
            retry_not_before_at TEXT,
            terminal_reason_code TEXT,
            terminal_reason_detail TEXT,
            result_reference TEXT,
            result_sha256 TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL,
            row_version INTEGER NOT NULL,
            PRIMARY KEY (tenant_id, work_item_id),
            UNIQUE (tenant_id, work_logical_key)
        );
        CREATE TABLE work_dependencies (
            tenant_id TEXT NOT NULL DEFAULT 'tenant-test',
            dependency_id TEXT NOT NULL,
            downstream_work_item_id TEXT NOT NULL,
            upstream_work_item_id TEXT NOT NULL,
            criticality TEXT NOT NULL,
            PRIMARY KEY (tenant_id, dependency_id),
            UNIQUE (tenant_id, downstream_work_item_id, upstream_work_item_id)
        );
        CREATE TABLE work_attempts (
            tenant_id TEXT NOT NULL DEFAULT 'tenant-test',
            work_item_id TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            lease_epoch INTEGER NOT NULL,
            fencing_token TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            claimed_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            outcome TEXT NOT NULL,
            error_code TEXT,
            error_detail TEXT,
            PRIMARY KEY (tenant_id, work_item_id, attempt_number),
            UNIQUE (tenant_id, work_item_id, lease_epoch),
            UNIQUE (tenant_id, fencing_token)
        );
        CREATE TABLE stream_plans (
            tenant_id TEXT NOT NULL DEFAULT 'tenant-test',
            plan_key TEXT NOT NULL,
            plan_json BLOB NOT NULL,
            source_subject_json BLOB NOT NULL,
            composition_config_json BLOB NOT NULL,
            planner_eos_sha256 TEXT,
            seal_json BLOB,
            terminal_closure_json BLOB,
            export_manifest_sha256 TEXT,
            export_member_count INTEGER,
            PRIMARY KEY (tenant_id, plan_key)
        );
        CREATE TABLE expected_windows (
            tenant_id TEXT NOT NULL DEFAULT 'tenant-test',
            plan_key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            declaration_json BLOB NOT NULL,
            window_json BLOB NOT NULL,
            terminal_member_json BLOB,
            PRIMARY KEY (tenant_id, plan_key, ordinal)
        );
        CREATE TABLE stream_work_plans (
            tenant_id TEXT NOT NULL DEFAULT 'tenant-test',
            work_item_id TEXT NOT NULL,
            work_logical_key TEXT NOT NULL,
            plan_key TEXT NOT NULL,
            expected_ordinal INTEGER,
            role_order INTEGER NOT NULL,
            stage TEXT NOT NULL,
            plan_json BLOB NOT NULL,
            publication_state TEXT NOT NULL,
            terminal_evidence_json BLOB,
            pending_terminal_json BLOB,
            pending_lease_epoch INTEGER,
            pending_fencing_token TEXT,
            PRIMARY KEY (tenant_id, work_item_id),
            UNIQUE (tenant_id, work_logical_key)
        );
        CREATE TABLE stream_backpressure_controllers (
            tenant_id TEXT NOT NULL DEFAULT 'tenant-test',
            plan_key TEXT NOT NULL,
            controller_key TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            owner_fence INTEGER NOT NULL,
            state_json BLOB NOT NULL,
            PRIMARY KEY (tenant_id, plan_key, controller_key)
        );
        CREATE TABLE capture_authority_metadata (
            tenant_id TEXT NOT NULL DEFAULT 'tenant-test',
            singleton INTEGER NOT NULL,
            capture_authority_id TEXT NOT NULL,
            capture_authority_epoch INTEGER NOT NULL,
            capture_assignment_policy_version TEXT NOT NULL,
            next_acquisition_sequence INTEGER NOT NULL,
            PRIMARY KEY (tenant_id, singleton)
        );
        CREATE TABLE capture_authority_receipts (
            tenant_id TEXT NOT NULL DEFAULT 'tenant-test',
            receipt_slot TEXT NOT NULL,
            acquisition_sequence INTEGER NOT NULL,
            request_json BLOB NOT NULL,
            subject_json BLOB NOT NULL,
            capture_scope_digest TEXT NOT NULL,
            PRIMARY KEY (tenant_id, receipt_slot),
            UNIQUE (tenant_id, acquisition_sequence),
            UNIQUE (tenant_id, capture_scope_digest)
        );
        """
    )


@pytest.fixture
def _authority() -> tuple[PostgresCanonicalAuthority, _PostgresSqliteHarness]:
    harness = _PostgresSqliteHarness()
    return PostgresCanonicalAuthority(lambda: harness), harness


def _uuid(value: int) -> str:
    return f"00000000-0000-4000-8000-{value:012d}"


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _plan(
    value: int,
    *,
    max_attempts: int = 3,
    priority: int = 0,
) -> WorkItemPlan:
    return WorkItemPlan(
        work_item_id=_uuid(100 + value),
        work_logical_key=f"work:{value}",
        run_id=_RUN_ID,
        mcap_id=_MCAP_ID,
        stage=Stage.QA_COARSE_PLAN,
        subject_type=WorkItemSubjectType.MCAP,
        subject_id=_uuid(200 + value),
        input_digest="a" * 64,
        config_digest="b" * 64,
        priority=priority,
        max_attempts=max_attempts,
        created_at=_timestamp(_BASE),
    )


def _schema() -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/pre-eos-capture-subject",
        version="1.0.0",
        artifact_id=str(UUID(int=1)),
        sha256="1" * 64,
    )


def _bindings() -> tuple[ChannelBinding, ...]:
    return tuple(
        ChannelBinding(
            camera_id=camera_id,
            source_channel_id=f"source-{camera_id.value}",
            source_channel_epoch=1,
            channel_binding_semantic_sha256=f"{index + 10:064x}",
        )
        for index, camera_id in enumerate(CAMERA_IDS)
    )


def _binding() -> AuthorityBinding:
    return AuthorityBinding(
        authority_id="source-authority",
        authority_epoch=1,
        policy_version="source-policy-v1",
        initial_binding_semantic_sha256="2" * 64,
    )


def test_postgres_scheduler_preserves_fence_and_claim_state_machine(
    _authority: tuple[PostgresCanonicalAuthority, _PostgresSqliteHarness],
) -> None:
    authority, harness = _authority
    scheduler = PostgresWorkScheduler(authority)
    plan = _plan(1, priority=10)

    assert scheduler.plan(plan).state is WorkItemState.READY
    assert scheduler.plan(plan).work_item_id == plan.work_item_id
    claim = scheduler.claim("worker-a", 10, now=_BASE)
    assert claim is not None
    assert claim.work_item.state is WorkItemState.LEASED
    assert (
        scheduler.start(claim.lease, now=_BASE + timedelta(seconds=1)).state
        is WorkItemState.RUNNING
    )
    renewed = scheduler.heartbeat(claim.lease, 10, now=_BASE + timedelta(seconds=2))

    with pytest.raises(WorkFenceError, match="stale, expired, or inactive"):
        scheduler.succeed(claim.lease, now=_BASE + timedelta(seconds=3))

    terminal = scheduler.succeed(
        renewed,
        result_reference="results/window-1.json",
        result_sha256="c" * 64,
        now=_BASE + timedelta(seconds=3),
    )
    assert terminal.state is WorkItemState.SUCCEEDED
    assert terminal.fencing_token is None
    assert scheduler.list_attempts(plan.work_item_id)[0].outcome is WorkAttemptOutcome.SUCCEEDED
    assert any("FOR UPDATE SKIP LOCKED" in query for query in harness.queries)
    assert any("pg_advisory_xact_lock" in query for query in harness.queries)
    assert harness.begin_count == harness.commit_count + harness.rollback_count


def test_postgres_scheduler_recovers_leases_and_cascades_required_failure(
    _authority: tuple[PostgresCanonicalAuthority, _PostgresSqliteHarness],
) -> None:
    authority, _harness = _authority
    scheduler = PostgresWorkScheduler(authority)
    upstream = _plan(2, max_attempts=1)
    downstream = _plan(3)
    dependency = WorkDependency(
        dependency_id=_uuid(900),
        downstream_work_item_id=downstream.work_item_id,
        upstream_work_item_id=upstream.work_item_id,
        criticality=DependencyCriticality.REQUIRED,
    )
    scheduler.plan(upstream)
    assert scheduler.plan(downstream, (dependency,)).state is WorkItemState.PLANNED
    claim = scheduler.claim_and_start("worker-a", 5, work_item_id=upstream.work_item_id, now=_BASE)
    assert claim is not None
    assert scheduler.reconcile(now=_BASE + timedelta(seconds=5)) == 1
    assert scheduler.get(upstream.work_item_id).state is WorkItemState.FAILED_PERMANENT
    assert scheduler.get(downstream.work_item_id).state is WorkItemState.FAILED_PERMANENT
    assert scheduler.list_attempts(upstream.work_item_id)[0].outcome is WorkAttemptOutcome.ABANDONED


def test_stream_and_capture_extensions_share_one_postgres_authority(
    _authority: tuple[PostgresCanonicalAuthority, _PostgresSqliteHarness],
) -> None:
    authority, harness = _authority
    scheduler = PostgresWorkScheduler(authority)
    ledger = PostgresStreamWorkLedger(scheduler)
    capture = PostgresCaptureAuthority(
        scheduler,
        capture_authority_id="postgres-capture-authority",
        capture_authority_epoch=1,
        capture_assignment_policy_version="assignment-v1",
    )

    ledger.register_plan(
        plan_key="stream-plan",
        plan_json=b'{"plan":true}',
        source_subject_json=b'{"source":true}',
        composition_config_json=b'{"composition":true}',
    )
    assert ledger.append_window(
        plan_key="stream-plan",
        ordinal=0,
        declaration_json=b'{"ordinal":0}',
        window_json=b'{"window":0}',
        work_plans=(),
    )
    source = _binding()
    issued = capture.issue("input/slot-1", _schema(), _bindings(), source, source)
    replayed = capture.issue("input/slot-1", _schema(), _bindings(), source, source)

    assert isinstance(issued, PreEosCaptureSubject)
    assert replayed == issued
    assert issued.acquisition_id == "postgres-capture-authority:1"
    assert scheduler.authority is ledger.authority.authority is capture.authority
    assert harness.begin_count == harness.commit_count + harness.rollback_count


def test_postgres_authority_rejects_nested_provider_work(
    _authority: tuple[PostgresCanonicalAuthority, _PostgresSqliteHarness],
) -> None:
    authority, _harness = _authority

    def operation(_connection: object) -> None:
        assert active_postgres_authority_transaction_operation() == "test.nested"
        with pytest.raises(PostgresAuthorityConfigurationError):
            require_outside_postgres_authority_transaction(activity="provider invocation")

    authority.run_authority_transaction(
        write=False,
        operation_name="test.nested",
        operation=operation,
    )


def test_postgres_authority_observes_transactions_and_retries_without_operation_ids() -> None:
    harness = _PostgresSqliteHarness()
    recorder = RuntimeProfileRecorder()
    authority = PostgresCanonicalAuthority(
        lambda: harness,
        serialization_retries=1,
        runtime_observer=recorder,
    )
    calls = 0

    class _SerializationFailure(Exception):
        sqlstate = "40001"

    def operation(_connection: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _SerializationFailure()
        return "committed"

    assert (
        authority.run_authority_transaction(
            write=True,
            operation_name="sensitive.operation.name",
            operation=operation,
        )
        == "committed"
    )

    snapshot = recorder.snapshot()

    assert "sensitive.operation.name" not in snapshot.model_dump_json()
    assert calls == 2
    assert len(snapshot.spans) == 1
    assert snapshot.spans[0].name == "postgres.authority.transaction"
    assert [(item.name, item.value) for item in snapshot.spans[0].attributes] == [
        ("operation_family", "OTHER"),
        ("write", True),
    ]
    assert {
        (counter.name, tuple((item.name, item.value) for item in counter.attributes)): counter.value
        for counter in snapshot.counters
    } == {
        (
            "postgres.authority.transaction_attempts",
            (("operation_family", "OTHER"), ("write", True)),
        ): 2,
        (
            "postgres.authority.transaction_retries",
            (("operation_family", "OTHER"), ("write", True)),
        ): 1,
    }


def test_p22_migration_is_transaction_runner_compatible() -> None:
    migration = (
        Path(__file__).parents[2] / "db" / "migrations" / "0002_work_and_stream_authority.sql"
    )
    sql = migration.read_text(encoding="utf-8")
    assert "CREATE TABLE robata_canonical.work_items" in sql
    assert "CREATE TABLE robata_canonical.capture_authority_receipts" in sql
    assert "tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id()" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "CREATE POLICY" in sql
    assert "BYTEA" in sql
    assert not re.search(r"(?m)^BEGIN;|^COMMIT;", sql)
