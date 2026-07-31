from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Any

import pytest

from robata.adapters.postgres_authority import PostgresCanonicalAuthority
from robata.adapters.postgres_read_model import (
    EvidenceDocumentKind,
    PostgresCanonicalReadModel,
    PostgresReadModelConfigurationError,
    PostgresReadModelIntegrityError,
)


class _Cursor:
    def __init__(self, rows: Sequence[dict[str, object]]) -> None:
        self._rows = tuple(rows)

    @property
    def rowcount(self) -> int:
        return len(self._rows)

    def fetchone(self) -> dict[str, object] | None:
        return None if not self._rows else self._rows[0]

    def fetchall(self) -> tuple[dict[str, object], ...]:
        return self._rows


class _ReadModelConnection:
    def __init__(
        self,
        *,
        tenant_id: str | None = "tenant-test",
        raw_digest: str | None = None,
        migration_ids: tuple[str, ...] = ("0001", "0002", "0003"),
    ) -> None:
        self.tenant_id = tenant_id
        self.raw_digest = raw_digest or _digest(_RAW_BYTES)
        self.migration_ids = migration_ids
        self.queries: list[tuple[str, Sequence[object] | None]] = []
        self.commit_count = 0
        self.rollback_count = 0

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> _Cursor:
        self.queries.append((query, params))
        normalized = " ".join(query.split())
        if normalized.startswith("BEGIN") or normalized.startswith("SET LOCAL"):
            return _Cursor(())
        if "set_config" in normalized:
            return _Cursor(())
        if "current_setting" in normalized:
            return _Cursor(({"tenant_id": self.tenant_id},))
        if "information_schema.tables" in normalized:
            return _Cursor(tuple({"table_name": name} for name in _TABLES))
        if "robata_ops.schema_migrations" in normalized:
            return _Cursor(
                tuple({"migration_id": migration_id} for migration_id in self.migration_ids)
            )
        if "r.run_json" in normalized:
            return _Cursor((_DETAIL_ROW,))
        if "FROM primary_runs AS r" in normalized:
            return _Cursor((_SUMMARY_ROW,))
        if "FROM work_items" in normalized:
            return _Cursor((_WORK_ROW,))
        if "FROM primary_outbox AS o" in normalized:
            return _Cursor((_OUTBOX_ROW,))
        if "FROM model_inference_terminals AS t" in normalized:
            return _Cursor((_EVIDENCE_ROW,))
        if "FROM raw_provider_responses" in normalized:
            row = dict(_RAW_ROW)
            row["exact_bytes_sha256"] = self.raw_digest
            return _Cursor((row,))
        if "AS document_id" in normalized:
            return _Cursor((_EVIDENCE_DOCUMENT_ROW,))
        raise AssertionError(f"unexpected SQL: {normalized}")

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        return None


_RAW_BYTES = b'{"raw":"provider-response"}'
_RUN_BYTES = b'{"run":"run-1"}'
_COMMAND_BYTES = b'{"command":"run-1"}'
_COMMITTED_BYTES = b'{"committed":"run-1"}'
_DETAIL_BYTES = b'{"detail":"result-1"}'
_EVIDENCE_BYTES = b'{"artifact":"evidence-1"}'


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


_TABLES = (
    "primary_runs",
    "primary_completions",
    "detailed_results",
    "primary_outbox",
    "primary_outbox_deliveries",
    "work_items",
    "inference_intents",
    "model_inference_terminals",
    "raw_provider_responses",
    "raw_provider_artifacts",
    "parsed_provider_claims",
    "selected_attempt_outputs",
    "enriched_provider_outputs",
)

_SUMMARY_ROW: dict[str, object] = {
    "run_id": "run-1",
    "recording_identity": "recording-1",
    "mcap_id": "mcap-1",
    "pipeline_version": "pipeline-v1",
    "config_sha256": "a" * 64,
    "started_at": "2026-01-01T00:00:00.000000Z",
    "completed_at": "2026-01-01T00:01:00.000000Z",
    "primary_status": "SUCCEEDED",
    "run_version": 1,
    "command_sha256": _digest(_COMMAND_BYTES),
    "committed_json_sha256": _digest(_COMMITTED_BYTES),
    "detailed_result_artifact_id": "detail-1",
    "work_item_count": 3,
    "terminal_work_item_count": 2,
    "outbox_count": 2,
    "undelivered_outbox_count": 1,
}

_DETAIL_ROW: dict[str, object] = {
    "run_json": memoryview(_RUN_BYTES),
    "run_json_sha256": _digest(_RUN_BYTES),
    "command_json": _COMMAND_BYTES,
    "command_json_sha256": _digest(_COMMAND_BYTES),
    "committed_json": _COMMITTED_BYTES,
    "committed_json_sha256": _digest(_COMMITTED_BYTES),
    "payload_json": _DETAIL_BYTES,
    "exact_bytes_sha256": _digest(_DETAIL_BYTES),
    "byte_count": len(_DETAIL_BYTES),
}

_WORK_ROW: dict[str, object] = {
    "stage": "QA_COARSE_PLAN",
    "state": "SUCCEEDED",
    "work_item_count": 2,
    "attempts_started": 3,
    "latest_updated_at": "2026-01-01T00:01:00.000000Z",
}

_OUTBOX_ROW: dict[str, object] = {
    "outbox_id": "outbox-1",
    "outbox_ordinal": 0,
    "assignment_logical_key": "assignment-1",
    "payload_json_sha256": "b" * 64,
    "delivered_at": None,
    "delivery_status": "PENDING",
    "attempt_count": 0,
    "next_attempt_at": "2026-01-01T00:01:01.000000Z",
    "lease_epoch": 2,
}

_EVIDENCE_ROW: dict[str, object] = {
    "inference_id": "inference-1",
    "logical_invocation_id": "invocation-1",
    "request_id": "request-1",
    "terminal_status": "SUCCEEDED",
    "shadow": 0,
    "output_valid": 1,
    "raw_artifact_id": "raw-1",
    "provider": "runpod",
    "model_name": "mage-vl",
    "model_version": "4b",
    "provider_request_id": "provider-request-1",
    "created_at": "2026-01-01T00:00:30.000000Z",
    "exact_bytes_sha256": _digest(_RAW_BYTES),
    "byte_count": len(_RAW_BYTES),
    "media_type": "application/json",
}

_RAW_ROW: dict[str, object] = {
    "artifact_id": "raw-1",
    "inference_id": "inference-1",
    "request_id": "request-1",
    "provider_request_id": "provider-request-1",
    "media_type": "application/json",
    "exact_bytes_sha256": _digest(_RAW_BYTES),
    "byte_count": len(_RAW_BYTES),
    "raw_bytes": _RAW_BYTES,
}

_EVIDENCE_DOCUMENT_ROW: dict[str, object] = {
    "document_id": "raw-1",
    "payload_json": _EVIDENCE_BYTES,
    "payload_sha256": _digest(_EVIDENCE_BYTES),
}


def _read_model(connection: _ReadModelConnection) -> PostgresCanonicalReadModel:
    authority = PostgresCanonicalAuthority(
        lambda: connection,
        tenant_setting="robata.tenant_id",
        tenant_id="tenant-test",
    )
    return PostgresCanonicalReadModel(authority)


def test_postgres_read_model_returns_verified_snapshots_through_read_only_rls() -> None:
    connection = _ReadModelConnection()
    read_model = _read_model(connection)

    startup = read_model.verify_startup()
    detail = read_model.get_committed_run_detail("run-1")
    stages = read_model.list_work_stages("run-1")
    outbox = read_model.list_outbox_delivery("run-1")
    evidence = read_model.list_evidence(logical_invocation_id="invocation-1")
    raw = read_model.get_raw_provider_response("raw-1")
    document = read_model.get_evidence_document(
        EvidenceDocumentKind.RAW_PROVIDER_ARTIFACT,
        "raw-1",
    )

    assert startup.tenant_id == "tenant-test"
    assert detail is not None
    assert detail.summary.undelivered_outbox_count == 1
    assert detail.run_document.exact_bytes == _RUN_BYTES
    assert detail.detailed_result_document.exact_bytes == _DETAIL_BYTES
    assert stages[0].attempts_started == 3
    assert outbox[0].delivery_status == "PENDING"
    assert evidence[0].model_name == "mage-vl"
    assert evidence[0].output_valid is True
    assert raw is not None and raw.raw_bytes == _RAW_BYTES
    assert document is not None and document.exact_bytes == _EVIDENCE_BYTES
    assert any(
        "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY" in query
        for query, _ in connection.queries
    )
    assert any("SELECT set_config" in query for query, _ in connection.queries)
    assert all("sqlite" not in query.lower() for query, _ in connection.queries)


def test_postgres_read_model_rejects_raw_byte_digest_mismatch() -> None:
    read_model = _read_model(_ReadModelConnection(raw_digest="0" * 64))

    with pytest.raises(PostgresReadModelIntegrityError, match="SHA-256"):
        read_model.get_raw_provider_response("raw-1")


def test_postgres_read_model_requires_transaction_local_tenant_context() -> None:
    connection = _ReadModelConnection(tenant_id=None)
    authority = PostgresCanonicalAuthority(lambda: connection)
    read_model = PostgresCanonicalReadModel(authority)

    with pytest.raises(PostgresReadModelConfigurationError, match="tenant_setting"):
        read_model.verify_startup()


def test_postgres_read_model_requires_checked_migration_baseline() -> None:
    read_model = _read_model(_ReadModelConnection(migration_ids=("0001",)))

    with pytest.raises(PostgresReadModelConfigurationError, match="0002, 0003"):
        read_model.verify_startup()


@pytest.mark.parametrize("limit", [0, 501, True, "100"])
def test_postgres_read_model_rejects_unbounded_or_invalid_pages(limit: Any) -> None:
    read_model = _read_model(_ReadModelConnection())

    with pytest.raises((TypeError, ValueError)):
        read_model.list_committed_runs(limit=limit)
