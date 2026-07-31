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
from robata.adapters.postgres_r2_artifacts import PostgresR2ArtifactAuthority
from robata.adapters.r2_object_store import R2ObjectStore, R2ObjectStoreConfig
from robata.application.canonical_run_membership import CanonicalProcessingRunContext
from robata.contracts.hashing import exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRegistry
from robata.queue.barrier import BarrierCoordinator
from robata.queue.outbox import OutboxDeliveryStatus, OutboxRetryPolicy
from robata.queue.stage import StageStatus
from tests.integration.test_sqlite_primary_completion import _run_case
from tests.unit.test_r2_object_store import _S3Double
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
    assert "0005" in application.applied_ids + application.already_applied_ids


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
            f"GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA robata_canonical TO {role_name}"
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


def _r2_observation_kinds(
    authority: PostgresCanonicalAuthority,
    artifact_id: str,
) -> tuple[str, ...]:
    def operation(connection: PostgresConnection) -> tuple[str, ...]:
        rows = connection.execute(
            """
            SELECT observation_kind
            FROM raw_provider_r2_artifact_observations
            WHERE artifact_id = %s
            ORDER BY observed_at, observation_id
            """,
            (artifact_id,),
        ).fetchall()
        return tuple(str(row["observation_kind"]) for row in rows)

    return authority.run_authority_transaction(
        write=False,
        operation_name="test.r2_observation_kinds",
        operation=operation,
    )


def test_postgres_r2_receipts_cannot_begin_committed(
    postgres_runtime_factory: ConnectionFactory,
) -> None:
    tenant_id = f"postgres-r2-receipt-{uuid4().hex}"
    authority = _authority(postgres_runtime_factory, tenant_id)
    ledger = PostgresInferenceEvidenceLedger(authority, SchemaRegistry())
    _fixture, intent, raw_data = _intent()
    assert ledger.append_intent(intent) == intent

    artifact_id = str(uuid4())
    provider_request_id = f"postgres-r2-receipt:{uuid4()}"
    raw_digest = exact_bytes_sha256(raw_data)

    def insert_initially_committed(connection: PostgresConnection) -> None:
        connection.execute(
            """
            INSERT INTO raw_provider_r2_artifact_receipts (
                artifact_id, inference_id, request_id, provider_request_id,
                exact_bytes_sha256, byte_count, media_type, payload_bytes,
                logical_key, object_uri, object_version, r2_config_sha256,
                state, committed_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, 'application/json', %s,
                %s, %s, %s, %s, 'COMMITTED', CURRENT_TIMESTAMP
            )
            """,
            (
                artifact_id,
                intent.inference_id,
                intent.request_id,
                provider_request_id,
                raw_digest,
                len(raw_data),
                raw_data,
                f"raw-provider-responses/receipt-test/{artifact_id}",
                (
                    "r2://robata-test/raw-provider-responses/receipt-test/"
                    f"{artifact_id}/.robata-versions/raw-v1-{raw_digest}"
                ),
                f"raw-v1-{raw_digest}",
                "0" * 64,
            ),
        )

    with pytest.raises(Exception, match="raw provider R2 artifact receipts must begin staged"):
        authority.run_authority_transaction(
            write=True,
            operation_name="test.r2_receipt_initially_committed",
            operation=insert_initially_committed,
        )


def test_postgres_completion_outbox_evidence_and_barrier_happy_paths(
    postgres_runtime_factory: ConnectionFactory,
    tmp_path: Path,
) -> None:
    tenant_id = f"postgres-integration-{uuid4().hex}"
    authority = _authority(postgres_runtime_factory, tenant_id)
    verify_completion_evidence_schema(authority)
    r2_client = _S3Double()
    r2_object_store = R2ObjectStore(
        R2ObjectStoreConfig(
            endpoint_url="https://account-id.r2.cloudflarestorage.com",
            bucket="robata-test",
            prefix="canonical-evidence",
        ),
        r2_client,
    )
    artifact_authority = PostgresR2ArtifactAuthority(
        authority,
        r2_object_store,
        tenant_id=tenant_id,
    )
    artifact_authority.verify_startup()

    ledger = PostgresInferenceEvidenceLedger(
        authority,
        SchemaRegistry(),
        artifact_authority=artifact_authority,
    )
    fixture, intent, raw_data = _intent()
    assert ledger.append_intent(intent) == intent
    r2_client.fail_after_write_once = True
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
    assert r2_client.put_count == 1
    assert _r2_observation_kinds(authority, stored_raw.artifact_id) == ("PUT_VERIFIED",)
    assert ledger.get(stored_raw.artifact_id) == stored_raw
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
