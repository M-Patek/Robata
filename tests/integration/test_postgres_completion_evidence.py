"""Real PostgreSQL coverage for canonical completion, evidence, barriers, and delivery.

Set ``ROBATA_TEST_POSTGRES_DSN`` to opt into this integration test. The target
database is intentionally not created or destroyed by the test; it applies the
checked-in immutable migration set and isolates its records with a unique RLS
tenant setting.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from robata.adapters.postgres_authority import (
    ConnectionFactory,
    PostgresCanonicalAuthority,
    PostgresConnection,
    psycopg_connection_factory,
)
from robata.adapters.postgres_completion_evidence import (
    PostgresBarrierStorage,
    PostgresInferenceEvidenceLedger,
    PostgresPrimaryCompletionRepository,
    PostgresPrimaryOutboxDeliveryStore,
    verify_completion_evidence_schema,
)
from robata.adapters.postgres_migrations import PostgresMigrationRunner
from robata.application.canonical_run_membership import CanonicalProcessingRunContext
from robata.contracts.schema_registry import SchemaRegistry
from robata.queue.barrier import BarrierCoordinator
from robata.queue.outbox import OutboxDeliveryStatus, OutboxRetryPolicy
from robata.queue.stage import StageStatus
from tests.integration.test_sqlite_primary_completion import _run_case
from tests.unit.test_sqlite_barrier import _completion, _declare, _reduction
from tests.unit.test_sqlite_inference_evidence import _build_after_raw, _intent

_TEST_DSN = os.environ.get("ROBATA_TEST_POSTGRES_DSN")


@pytest.fixture(scope="session")
def postgres_factory() -> ConnectionFactory:
    if not _TEST_DSN:
        pytest.skip("ROBATA_TEST_POSTGRES_DSN is required for PostgreSQL integration tests")
    return psycopg_connection_factory(_TEST_DSN, application_name="robata-postgres-integration")


@pytest.fixture(scope="session", autouse=True)
def migrated_postgres(postgres_factory: ConnectionFactory) -> None:
    migrations = Path(__file__).resolve().parents[2] / "db" / "migrations"
    application = PostgresMigrationRunner(postgres_factory, migrations).apply()
    assert "0003" in application.applied_ids + application.already_applied_ids


@pytest.fixture
def postgres_runtime_factory(
    postgres_factory: ConnectionFactory,
) -> Iterator[ConnectionFactory]:
    """Use a non-superuser role so forced RLS is exercised in the real-PG test."""

    role_name = f"robata_p23_rls_{uuid4().hex}"
    administrator = postgres_factory()
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
        connection = postgres_factory()
        connection.execute(f"SET ROLE {role_name}")
        return connection

    try:
        yield restricted_factory
    finally:
        cleanup = postgres_factory()
        try:
            cleanup.execute(f"DROP OWNED BY {role_name}")
            cleanup.execute(f"DROP ROLE {role_name}")
        finally:
            cleanup.close()


def _authority(factory: ConnectionFactory, tenant_id: str) -> PostgresCanonicalAuthority:
    return PostgresCanonicalAuthority(
        factory,
        tenant_setting="robata.tenant_id",
        tenant_id=tenant_id,
    )


def test_postgres_completion_outbox_evidence_and_barrier_happy_paths(
    postgres_runtime_factory: ConnectionFactory,
    tmp_path: Path,
) -> None:
    authority = _authority(postgres_runtime_factory, f"postgres-integration-{uuid4().hex}")
    verify_completion_evidence_schema(authority)

    ledger = PostgresInferenceEvidenceLedger(authority, SchemaRegistry())
    fixture, intent, raw_data = _intent()
    assert ledger.append_intent(intent) == intent
    stored_raw = ledger.append(
        request_id=intent.request_id,
        provider_request_id="postgres-integration-provider",
        data=raw_data,
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored_raw.artifact_id,
        stored_raw.provider_request_id,
    )
    assert ledger.append_terminal_and_selection(evidence.terminal, evidence.selection) == (
        evidence.terminal,
        evidence.selection,
    )
    assert ledger.append_accepted_lineage(
        evidence.parsed,
        evidence.selected,
        evidence.enriched,
    ) == (evidence.parsed, evidence.selected, evidence.enriched)
    ledger.verify_completion_seal()
    assert ledger.get_enriched_output(evidence.enriched.artifact_id) == evidence.enriched

    barrier_storage = PostgresBarrierStorage(authority)
    definition = _declare(barrier_storage)
    coordinator = BarrierCoordinator(barrier_storage)
    completions = tuple(_completion(definition, ordinal) for ordinal in range(2))
    for completion in reversed(completions):
        assert barrier_storage.append_completion(completion) == completion
        coordinator.submit_member(
            definition.barrier_id,
            completion.part_logical_key,
            StageStatus.SUCCEEDED,
        )
    reduction = _reduction(definition, completions)
    assert barrier_storage.append_reduction(reduction) == reduction
    assert barrier_storage.list_completions(definition.barrier_id) == completions
    assert barrier_storage.get_reduction(definition.barrier_id) == reduction

    _, _preparation_repository, command = _run_case(
        tmp_path / "completion-preparation",
        run_value=uuid4().int % 1_000_000_000,
    )
    completion_repository = PostgresPrimaryCompletionRepository(authority)
    processing_run = command.detail.processing_run
    run_context = CanonicalProcessingRunContext.fresh(
        run_id=processing_run.run_id,
        recording_identity=processing_run.recording_identity,
        mcap_id=processing_run.mcap_id,
        pipeline_version=processing_run.pipeline_version,
        config_sha256=processing_run.config_sha256,
        started_at=processing_run.started_at,
    )
    assert completion_repository.begin_run(run_context) == run_context.to_record()
    prepared_identities = command.detail.prepared_identities
    assert prepared_identities is not None
    snapshot = completion_repository.snapshot(processing_run.recording_identity)
    assert (snapshot.generation, snapshot.fence) == (
        prepared_identities.expected_generation,
        prepared_identities.expected_fence,
    )
    first = completion_repository.commit(command)
    replay = completion_repository.commit(command)
    assert not first.replayed
    assert replay.replayed
    assert first.committed.outbox

    def clock() -> datetime:
        return datetime(2026, 7, 21, 12, tzinfo=UTC)

    delivery_store = PostgresPrimaryOutboxDeliveryStore(
        authority,
        retry_policy=OutboxRetryPolicy(
            version="postgres-integration-v1",
            max_attempts=3,
            base_delay_seconds=1.0,
            max_delay_seconds=10.0,
        ),
        clock=clock,
    )
    claim = delivery_store.claim(
        worker_id="postgres-integration-worker", lease_duration=timedelta(seconds=30)
    )
    assert claim is not None
    acknowledged = delivery_store.acknowledge(claim)
    assert acknowledged.status is OutboxDeliveryStatus.DELIVERED
    assert (
        delivery_store.claim(
            worker_id="postgres-integration-worker",
            lease_duration=timedelta(seconds=30),
        )
        is None
    )

    ledger.close()
