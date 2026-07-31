"""Optional real-PostgreSQL proof for the canonical committed-run read model.

Set ``ROBATA_TEST_POSTGRES_DSN`` only to an isolated disposable database. The
test applies checked-in migrations and uses a random tenant.
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

from robata.adapters.postgres_authority import (
    PostgresCanonicalAuthority,
    PostgresConnection,
    psycopg_connection_factory,
)
from robata.adapters.postgres_migrations import PostgresMigrationRunner
from robata.adapters.postgres_read_model import (
    EvidenceDocumentKind,
    PostgresCanonicalReadModel,
)

_DSN = os.environ.get("ROBATA_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not _DSN,
    reason="requires isolated ROBATA_TEST_POSTGRES_DSN",
)


def test_real_postgres_read_model_keeps_committed_and_evidence_bytes_exact() -> None:
    assert _DSN is not None
    factory = psycopg_connection_factory(_DSN, application_name="robata-read-model-integration")
    repository_root = Path(__file__).parents[2]
    PostgresMigrationRunner(factory, repository_root / "db" / "migrations").apply()

    authority = PostgresCanonicalAuthority(
        factory,
        tenant_setting="robata.tenant_id",
        tenant_id=f"read-model-{uuid4()}",
    )
    identifiers = _insert_minimal_graph(authority)
    read_model = PostgresCanonicalReadModel(authority)

    startup = read_model.verify_startup()
    summary = read_model.get_committed_run(identifiers["run_id"])
    detail = read_model.get_committed_run_detail(identifiers["run_id"])
    work = read_model.list_work_stages(identifiers["run_id"])
    outbox = read_model.list_outbox_delivery(identifiers["run_id"])
    evidence = read_model.list_evidence(logical_invocation_id=identifiers["logical_invocation_id"])
    raw = read_model.get_raw_provider_response(identifiers["raw_artifact_id"])
    document = read_model.get_evidence_document(
        EvidenceDocumentKind.RAW_PROVIDER_ARTIFACT,
        identifiers["raw_artifact_id"],
    )

    assert startup.tenant_id.startswith("read-model-")
    assert summary is not None and summary.work_item_count == 1
    assert summary.undelivered_outbox_count == 1
    assert detail is not None and detail.run_document.exact_bytes == identifiers["run_bytes"]
    assert detail.committed_document.exact_bytes == identifiers["committed_bytes"]
    assert work[0].state == "SUCCEEDED"
    assert outbox[0].delivery_status == "PENDING"
    assert evidence[0].model_name == "mage-vl"
    assert raw is not None and raw.raw_bytes == identifiers["raw_bytes"]
    assert document is not None and document.exact_bytes == identifiers["artifact_bytes"]

    other_tenant = PostgresCanonicalReadModel(
        PostgresCanonicalAuthority(
            factory,
            tenant_setting="robata.tenant_id",
            tenant_id=f"read-model-other-{uuid4()}",
        )
    )
    assert other_tenant.get_committed_run(identifiers["run_id"]) is None
    assert other_tenant.get_raw_provider_response(identifiers["raw_artifact_id"]) is None


def _insert_minimal_graph(authority: PostgresCanonicalAuthority) -> dict[str, str | bytes]:
    run_id = str(uuid4())
    recording_identity = f"recording:{uuid4()}"
    mcap_id = str(uuid4())
    detail_artifact_id = str(uuid4())
    event_id = str(uuid4())
    assignment_key = f"assignment:{uuid4()}"
    outbox_id = str(uuid4())
    work_item_id = str(uuid4())
    inference_id = str(uuid4())
    request_id = str(uuid4())
    logical_invocation_id = f"invocation:{uuid4()}"
    raw_artifact_id = str(uuid4())
    run_bytes = b'{"run":"canonical"}'
    command_bytes = b'{"command":"canonical"}'
    committed_bytes = b'{"committed":"canonical"}'
    detail_bytes = b'{"detail":"canonical"}'
    assignment_bytes = b'{"assignment":"canonical"}'
    outbox_bytes = b'{"outbox":"canonical"}'
    intent_bytes = b'{"intent":"canonical"}'
    raw_bytes = b'{"raw":"canonical"}'
    terminal_bytes = b'{"terminal":"canonical"}'
    artifact_bytes = b'{"artifact":"canonical"}'

    def insert(connection: PostgresConnection) -> None:
        connection.execute(
            """
            INSERT INTO primary_runs (
                run_id, recording_identity, mcap_id, pipeline_version, config_sha256,
                started_at, primary_status, completed_at, run_version, command_sha256,
                run_json, run_json_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, 'SUCCEEDED', %s, 1, %s, %s, %s)
            """,
            (
                run_id,
                recording_identity,
                mcap_id,
                "pipeline-v1",
                "a" * 64,
                "2026-01-01T00:00:00.000000Z",
                "2026-01-01T00:01:00.000000Z",
                _digest(command_bytes),
                run_bytes,
                _digest(run_bytes),
            ),
        )
        connection.execute(
            """
            INSERT INTO detailed_results (
                artifact_id, exact_bytes_sha256, byte_count, schema_id, schema_version,
                schema_artifact_id, schema_sha256, payload_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                detail_artifact_id,
                _digest(detail_bytes),
                len(detail_bytes),
                "https://schemas.robata.dev/detailed-result",
                "1.0.0",
                str(uuid4()),
                "b" * 64,
                detail_bytes,
            ),
        )
        connection.execute(
            """
            INSERT INTO primary_completions (
                run_id, command_sha256, command_json, command_json_sha256,
                committed_json, committed_json_sha256, detailed_result_artifact_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                _digest(command_bytes),
                command_bytes,
                _digest(command_bytes),
                committed_bytes,
                _digest(committed_bytes),
                detail_artifact_id,
            ),
        )
        connection.execute(
            """
            INSERT INTO event_registry_partitions (recording_identity, generation, fence)
            VALUES (%s, 0, 1)
            """,
            (recording_identity,),
        )
        connection.execute(
            """
            INSERT INTO stable_event_identities (
                event_id, recording_identity, payload_json, payload_json_sha256
            ) VALUES (%s, %s, %s, %s)
            """,
            (event_id, recording_identity, assignment_bytes, _digest(assignment_bytes)),
        )
        connection.execute(
            """
            INSERT INTO event_identity_assignments (
                assignment_logical_key, recording_identity, event_hypothesis_logical_key,
                identity_policy_version, identity_policy_sha256, event_id, payload_json,
                payload_json_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                assignment_key,
                recording_identity,
                f"hypothesis:{uuid4()}",
                "identity-v1",
                "c" * 64,
                event_id,
                assignment_bytes,
                _digest(assignment_bytes),
            ),
        )
        connection.execute(
            """
            INSERT INTO primary_outbox (
                outbox_id, completion_run_id, recording_identity, outbox_ordinal,
                assignment_logical_key, payload_json, payload_json_sha256
            ) VALUES (%s, %s, %s, 0, %s, %s, %s)
            """,
            (
                outbox_id,
                run_id,
                recording_identity,
                assignment_key,
                outbox_bytes,
                _digest(outbox_bytes),
            ),
        )
        connection.execute(
            """
            INSERT INTO primary_outbox_deliveries (
                outbox_id, status, attempt_count, lease_epoch, next_attempt_at,
                retry_policy_version, max_attempts, base_delay_seconds, max_delay_seconds
            ) VALUES (%s, 'PENDING', 0, 0, NOW(), 'retry-v1', 3, 1, 10)
            """,
            (outbox_id,),
        )
        connection.execute(
            """
            INSERT INTO work_items (
                schema_version, work_item_id, work_logical_key, run_id, mcap_id, stage,
                subject_type, subject_id, input_digest, config_digest, priority, max_attempts,
                created_at, state, cancel_requested, lease_epoch, attempt, completed_at,
                updated_at, row_version
            ) VALUES (
                '1.0', %s, %s, %s, %s, 'QA_COARSE_PLAN', 'MCAP', %s, %s, %s, 0, 1,
                %s, 'SUCCEEDED', 0, 0, 1, %s, %s, 1
            )
            """,
            (
                work_item_id,
                f"work:{uuid4()}",
                run_id,
                mcap_id,
                str(uuid4()),
                "d" * 64,
                "e" * 64,
                "2026-01-01T00:00:00.000000Z",
                "2026-01-01T00:01:00.000000Z",
                "2026-01-01T00:01:00.000000Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO inference_intents (
                inference_id, logical_invocation_id, request_id, contract_schema_id,
                contract_version, contract_artifact_id, contract_sha256, payload_json,
                payload_sha256
            ) VALUES (%s, %s, %s, %s, '1.0.0', %s, %s, %s, %s)
            """,
            (
                inference_id,
                logical_invocation_id,
                request_id,
                "https://schemas.robata.dev/inference-intent",
                str(uuid4()),
                "f" * 64,
                intent_bytes,
                _digest(intent_bytes),
            ),
        )
        connection.execute(
            """
            INSERT INTO raw_provider_responses (
                artifact_id, inference_id, request_id, provider_request_id,
                exact_bytes_sha256, media_type, byte_count, raw_bytes
            ) VALUES (%s, %s, %s, %s, %s, 'application/json', %s, %s)
            """,
            (
                raw_artifact_id,
                inference_id,
                request_id,
                f"provider-request:{uuid4()}",
                _digest(raw_bytes),
                len(raw_bytes),
                raw_bytes,
            ),
        )
        connection.execute(
            """
            INSERT INTO model_inference_terminals (
                inference_id, logical_invocation_id, request_id, status, shadow, output_valid,
                raw_artifact_id, contract_schema_id, contract_version, contract_artifact_id,
                contract_sha256, payload_json, payload_sha256
            ) VALUES (%s, %s, %s, 'SUCCEEDED', 0, 1, %s, %s, '1.0.0', %s, %s, %s, %s)
            """,
            (
                inference_id,
                logical_invocation_id,
                request_id,
                raw_artifact_id,
                "https://schemas.robata.dev/model-inference",
                str(uuid4()),
                "1" * 64,
                terminal_bytes,
                _digest(terminal_bytes),
            ),
        )
        connection.execute(
            """
            INSERT INTO raw_provider_artifacts (
                artifact_id, inference_id, request_id, exact_bytes_sha256, byte_count,
                media_type, provider_request_id, provider, model_name, model_version,
                created_at, contract_schema_id, contract_version, contract_artifact_id,
                contract_sha256, payload_json, payload_sha256
            ) VALUES (
                %s, %s, %s, %s, %s, 'application/json', %s, 'runpod', 'mage-vl', '4b',
                %s, %s, '1.0.0', %s, %s, %s, %s
            )
            """,
            (
                raw_artifact_id,
                inference_id,
                request_id,
                _digest(raw_bytes),
                len(raw_bytes),
                f"provider-request:{uuid4()}",
                "2026-01-01T00:00:30.000000Z",
                "https://schemas.robata.dev/raw-provider-response",
                str(uuid4()),
                "2" * 64,
                artifact_bytes,
                _digest(artifact_bytes),
            ),
        )

    authority.run_authority_transaction(
        write=True,
        operation_name="test.read_model.insert_minimal_graph",
        operation=insert,
    )
    return {
        "run_id": run_id,
        "logical_invocation_id": logical_invocation_id,
        "raw_artifact_id": raw_artifact_id,
        "run_bytes": run_bytes,
        "committed_bytes": committed_bytes,
        "raw_bytes": raw_bytes,
        "artifact_bytes": artifact_bytes,
    }


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()
