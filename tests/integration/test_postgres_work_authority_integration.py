"""Optional real-PostgreSQL proof for P22 canonical work authority.

Set ``ROBATA_TEST_POSTGRES_DSN`` only for an isolated disposable database. The test
applies the checked-in migration set, so it must never point at a shared deployment.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest

from robata.adapters.postgres_authority import (
    PostgresCanonicalAuthority,
    PostgresConnection,
    psycopg_connection_factory,
)
from robata.adapters.postgres_capture_authority import PostgresCaptureAuthority
from robata.adapters.postgres_migrations import PostgresMigrationRunner
from robata.adapters.postgres_stream_work_ledger import PostgresStreamWorkLedger
from robata.adapters.postgres_work_scheduler import (
    PostgresWorkNotFoundError,
    PostgresWorkScheduler,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import AuthorityBinding, ChannelBinding
from robata.queue.models import WorkItem, WorkItemPlan, WorkItemState, WorkItemSubjectType
from robata.queue.stage import Stage

_DSN = os.environ.get("ROBATA_TEST_POSTGRES_DSN")
_BASE = datetime(2026, 1, 1, tzinfo=UTC)

pytestmark = pytest.mark.skipif(
    not _DSN,
    reason="requires isolated ROBATA_TEST_POSTGRES_DSN",
)


def test_real_postgres_migration_and_fenced_work_lifecycle() -> None:
    assert _DSN is not None
    factory = psycopg_connection_factory(
        _DSN,
        application_name="robata-p22-integration",
    )
    repository_root = Path(__file__).parents[2]
    runner = PostgresMigrationRunner(factory, repository_root / "db" / "migrations")
    application = runner.apply()
    assert set(application.applied_ids).union(application.already_applied_ids) >= {"0001", "0002"}

    authority = PostgresCanonicalAuthority(
        factory,
        tenant_setting="robata.tenant_id",
        tenant_id="p22-integration",
    )
    startup = authority.verify_startup()
    assert startup.backend_kind == "POSTGRESQL"
    scheduler = PostgresWorkScheduler(authority)
    work_item_id = str(uuid4())
    plan = WorkItemPlan(
        work_item_id=work_item_id,
        work_logical_key=f"postgres-work:{uuid4()}",
        run_id=str(uuid4()),
        mcap_id=str(uuid4()),
        stage=Stage.QA_COARSE_PLAN,
        subject_type=WorkItemSubjectType.MCAP,
        subject_id=str(uuid4()),
        input_digest="a" * 64,
        config_digest="b" * 64,
        created_at=_BASE.isoformat(timespec="microseconds"),
    )

    assert scheduler.plan(plan).state is WorkItemState.READY
    claim = scheduler.claim_and_start(
        "postgres-integration-worker",
        30,
        work_item_id=work_item_id,
        now=_BASE,
    )
    assert claim is not None
    renewed = scheduler.heartbeat(claim.lease, 30, now=_BASE + timedelta(seconds=1))
    terminal = scheduler.succeed(
        renewed,
        result_reference=f"r2://test/{work_item_id}",
        result_sha256="b" * 64,
        now=_BASE + timedelta(seconds=2),
    )
    assert terminal.state is WorkItemState.SUCCEEDED
    assert scheduler.list_attempts(work_item_id)[0].lease_epoch == 1

    ledger = PostgresStreamWorkLedger(scheduler)
    plan_key = f"stream-plan:{uuid4()}"
    ledger.register_plan(
        plan_key=plan_key,
        plan_json=b'{"plan":true}',
        source_subject_json=b'{"source":true}',
        composition_config_json=b'{"composition":true}',
    )
    assert ledger.append_window(
        plan_key=plan_key,
        ordinal=0,
        declaration_json=b'{"ordinal":0}',
        window_json=b'{"window":0}',
        work_plans=(),
    )
    assert ledger.next_window_ordinal(plan_key) == 1

    capture = PostgresCaptureAuthority(
        scheduler,
        capture_authority_id="postgres-integration-capture",
        capture_authority_epoch=1,
        capture_assignment_policy_version="assignment-v1",
    )
    source_authority = AuthorityBinding(
        authority_id="source-authority",
        authority_epoch=1,
        policy_version="source-policy-v1",
        initial_binding_semantic_sha256="c" * 64,
    )
    bindings = tuple(
        ChannelBinding(
            camera_id=camera_id,
            source_channel_id=f"source-{camera_id.value}",
            source_channel_epoch=1,
            channel_binding_semantic_sha256=f"{index + 1:064x}",
        )
        for index, camera_id in enumerate(CAMERA_IDS)
    )
    schema_ref = SchemaRef(
        schema_id="https://schemas.robata.dev/pre-eos-capture-subject",
        version="1.0.0",
        artifact_id=str(uuid4()),
        sha256="d" * 64,
    )
    receipt_slot = f"input/{uuid4()}"
    issued = capture.issue(
        receipt_slot,
        schema_ref,
        bindings,
        source_authority,
        source_authority,
    )
    assert (
        capture.issue(
            receipt_slot,
            schema_ref,
            bindings,
            source_authority,
            source_authority,
        )
        == issued
    )


def test_real_postgres_forced_rls_isolates_tenant_work_items() -> None:
    assert _DSN is not None
    factory = psycopg_connection_factory(
        _DSN,
        application_name="robata-p22-rls-integration",
    )
    repository_root = Path(__file__).parents[2]
    PostgresMigrationRunner(factory, repository_root / "db" / "migrations").apply()

    role_name = f"robata_p22_rls_{uuid4().hex}"
    administrator = factory()
    try:
        administrator.execute(f"CREATE ROLE {role_name} NOLOGIN")
        administrator.execute(f"GRANT USAGE ON SCHEMA robata_canonical TO {role_name}")
        administrator.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
            f"IN SCHEMA robata_canonical TO {role_name}"
        )
    finally:
        administrator.close()

    def restricted_factory() -> PostgresConnection:
        connection = factory()
        connection.execute(f"SET ROLE {role_name}")
        return connection

    def stored_tenants(authority: PostgresCanonicalAuthority, work_item_id: str) -> tuple[str, ...]:
        def operation(connection: PostgresConnection) -> tuple[str, ...]:
            rows = connection.execute(
                "SELECT tenant_id FROM work_items WHERE work_item_id = %s",
                (work_item_id,),
            ).fetchall()
            return tuple(str(row["tenant_id"]) for row in rows)

        return authority.run_authority_transaction(
            write=False,
            operation_name="test.rls_tenants",
            operation=operation,
        )

    authority_a = PostgresCanonicalAuthority(
        restricted_factory,
        tenant_setting="robata.tenant_id",
        tenant_id="p22-tenant-a",
    )
    authority_b = PostgresCanonicalAuthority(
        restricted_factory,
        tenant_setting="robata.tenant_id",
        tenant_id="p22-tenant-b",
    )
    work_item_id = str(uuid4())
    plan = WorkItemPlan(
        work_item_id=work_item_id,
        work_logical_key=f"postgres-rls-work:{uuid4()}",
        run_id=str(uuid4()),
        mcap_id=str(uuid4()),
        stage=Stage.QA_COARSE_PLAN,
        subject_type=WorkItemSubjectType.MCAP,
        subject_id=str(uuid4()),
        input_digest="a" * 64,
        config_digest="b" * 64,
        created_at=_BASE.isoformat(timespec="microseconds"),
    )
    scheduler_a = PostgresWorkScheduler(authority_a)
    scheduler_b = PostgresWorkScheduler(authority_b)

    try:
        assert scheduler_a.plan(plan).state is WorkItemState.READY
        assert stored_tenants(authority_a, work_item_id) == ("p22-tenant-a",)
        assert stored_tenants(authority_b, work_item_id) == ()
        with pytest.raises(PostgresWorkNotFoundError):
            scheduler_b.get(work_item_id)

        # Composite tenant keys permit the same replay identity in an isolated tenant.
        assert scheduler_b.plan(plan).state is WorkItemState.READY
        assert stored_tenants(authority_a, work_item_id) == ("p22-tenant-a",)
        assert stored_tenants(authority_b, work_item_id) == ("p22-tenant-b",)
    finally:
        cleanup = factory()
        try:
            cleanup.execute(f"DROP OWNED BY {role_name}")
            cleanup.execute(f"DROP ROLE {role_name}")
        finally:
            cleanup.close()


def test_real_postgres_concurrent_exact_plan_replays_once() -> None:
    assert _DSN is not None
    factory = psycopg_connection_factory(
        _DSN,
        application_name="robata-p22-concurrency-integration",
    )
    repository_root = Path(__file__).parents[2]
    PostgresMigrationRunner(factory, repository_root / "db" / "migrations").apply()
    authority = PostgresCanonicalAuthority(
        factory,
        tenant_setting="robata.tenant_id",
        tenant_id="p22-concurrency",
    )
    scheduler = PostgresWorkScheduler(authority)
    plan = WorkItemPlan(
        work_item_id=str(uuid4()),
        work_logical_key=f"postgres-concurrent-work:{uuid4()}",
        run_id=str(uuid4()),
        mcap_id=str(uuid4()),
        stage=Stage.QA_COARSE_PLAN,
        subject_type=WorkItemSubjectType.MCAP,
        subject_id=str(uuid4()),
        input_digest="a" * 64,
        config_digest="b" * 64,
        created_at=_BASE.isoformat(timespec="microseconds"),
    )
    barrier = Barrier(2)

    def plan_at_once() -> WorkItem:
        barrier.wait()
        return scheduler.plan(plan)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(plan_at_once) for _ in range(2))
        results = tuple(future.result() for future in futures)

    assert tuple(item.work_item_id for item in results) == (plan.work_item_id, plan.work_item_id)
    assert scheduler.items_for_run(plan.run_id) == (results[0],)
