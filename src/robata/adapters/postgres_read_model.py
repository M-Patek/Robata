"""Tenant-scoped PostgreSQL read projections for committed Robata work.

This adapter is deliberately a *read side*.  It never reconstructs a canonical
document from decoded JSON, and it never opens a local SQLite database.  Every
operation executes through :class:`PostgresCanonicalAuthority`, which gives the
query a repeatable-read transaction and the transaction-local tenant setting
required by the canonical table RLS policies.

The indexed fields below make a UI/API view efficient.  When callers request a
canonical document or raw provider response, the original ``bytea`` is returned
unchanged only after its persisted SHA-256 and length have been checked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import TypeVar

from robata.adapters.postgres_authority import (
    PostgresAuthorityError,
    PostgresCanonicalAuthority,
    PostgresConnection,
    Row,
)

_ResultT = TypeVar("_ResultT")
_TENANT_SETTING = "robata.tenant_id"
_MAX_PAGE_SIZE = 500


class PostgresReadModelError(RuntimeError):
    """Base error for the canonical PostgreSQL read side."""


class PostgresReadModelConfigurationError(PostgresReadModelError):
    """The adapter is not bound to the tenant/RLS authority required for reads."""


class PostgresReadModelStorageError(PostgresReadModelError):
    """PostgreSQL could not provide one consistent read-model snapshot."""


class PostgresReadModelIntegrityError(PostgresReadModelError):
    """A persisted canonical byte sequence does not match its declared digest."""


class EvidenceDocumentKind(StrEnum):
    """Canonical evidence documents that can be expanded without JSON re-encoding."""

    RAW_PROVIDER_ARTIFACT = "RAW_PROVIDER_ARTIFACT"
    PARSED_PROVIDER_CLAIM = "PARSED_PROVIDER_CLAIM"
    SELECTED_ATTEMPT_OUTPUT = "SELECTED_ATTEMPT_OUTPUT"
    ENRICHED_PROVIDER_OUTPUT = "ENRICHED_PROVIDER_OUTPUT"


@dataclass(frozen=True, slots=True)
class CanonicalBytesDocument:
    """One immutable canonical document, represented by its original bytes."""

    document_id: str
    kind: str
    exact_bytes: bytes
    sha256: str

    @property
    def byte_count(self) -> int:
        """Return the exact persisted payload size without decoding it."""

        return len(self.exact_bytes)


@dataclass(frozen=True, slots=True)
class CommittedRunSummary:
    """Compact immutable completion metadata plus live operational counters."""

    run_id: str
    recording_identity: str
    mcap_id: str
    pipeline_version: str
    config_sha256: str
    started_at: str
    completed_at: str
    primary_status: str
    run_version: int
    command_sha256: str
    committed_json_sha256: str
    detailed_result_artifact_id: str
    work_item_count: int
    terminal_work_item_count: int
    outbox_count: int
    undelivered_outbox_count: int


@dataclass(frozen=True, slots=True)
class CommittedRunDetail:
    """A committed run and its three exact canonical documents."""

    summary: CommittedRunSummary
    run_document: CanonicalBytesDocument
    command_document: CanonicalBytesDocument
    committed_document: CanonicalBytesDocument
    detailed_result_document: CanonicalBytesDocument


@dataclass(frozen=True, slots=True)
class WorkStageSummary:
    """One state bucket for a run's durable scheduler work."""

    stage: str
    state: str
    work_item_count: int
    attempts_started: int
    latest_updated_at: str


@dataclass(frozen=True, slots=True)
class OutboxDeliverySummary:
    """Delivery metadata only; the outbox payload remains canonical storage."""

    outbox_id: str
    outbox_ordinal: int
    assignment_logical_key: str
    payload_sha256: str
    delivered_at: str | None
    delivery_status: str | None
    attempt_count: int | None
    next_attempt_at: str | None
    lease_epoch: int | None


@dataclass(frozen=True, slots=True)
class EvidenceArtifactSummary:
    """Indexed evidence graph information without decoding provider payload bytes."""

    inference_id: str
    logical_invocation_id: str
    request_id: str
    terminal_status: str
    shadow: bool
    output_valid: bool
    raw_artifact_id: str | None
    provider: str | None
    model_name: str | None
    model_version: str | None
    provider_request_id: str | None
    created_at: str | None
    exact_bytes_sha256: str | None
    byte_count: int | None
    media_type: str | None


@dataclass(frozen=True, slots=True)
class RawProviderResponse:
    """A verified raw provider response returned as exact bytes."""

    artifact_id: str
    inference_id: str
    request_id: str
    provider_request_id: str
    media_type: str
    exact_bytes_sha256: str
    raw_bytes: bytes

    @property
    def byte_count(self) -> int:
        """Return the original response size."""

        return len(self.raw_bytes)


@dataclass(frozen=True, slots=True)
class PostgresReadModelStartup:
    """Verified, credential-free facts about the read-model deployment."""

    backend_kind: str
    schema: str
    tenant_id: str
    required_tables: tuple[str, ...]
    required_migration_ids: tuple[str, ...]


_REQUIRED_MIGRATION_IDS = ("0001", "0002", "0003")

_REQUIRED_TABLES = (
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

_EVIDENCE_DOCUMENT_TABLES: dict[EvidenceDocumentKind, tuple[str, str]] = {
    EvidenceDocumentKind.RAW_PROVIDER_ARTIFACT: ("raw_provider_artifacts", "artifact_id"),
    EvidenceDocumentKind.PARSED_PROVIDER_CLAIM: ("parsed_provider_claims", "artifact_id"),
    EvidenceDocumentKind.SELECTED_ATTEMPT_OUTPUT: ("selected_attempt_outputs", "selection_id"),
    EvidenceDocumentKind.ENRICHED_PROVIDER_OUTPUT: ("enriched_provider_outputs", "artifact_id"),
}

_COMMITTED_RUN_SELECT = """
    SELECT
        r.run_id,
        r.recording_identity,
        r.mcap_id,
        r.pipeline_version,
        r.config_sha256,
        r.started_at,
        r.completed_at,
        r.primary_status,
        r.run_version,
        c.command_sha256,
        c.committed_json_sha256,
        c.detailed_result_artifact_id,
        (
            SELECT COUNT(*)
            FROM work_items AS w
            WHERE w.tenant_id = r.tenant_id AND w.run_id = r.run_id
        ) AS work_item_count,
        (
            SELECT COUNT(*)
            FROM work_items AS w
            WHERE w.tenant_id = r.tenant_id AND w.run_id = r.run_id
              AND w.state IN (
                  'SUCCEEDED', 'FAILED_PERMANENT', 'SKIPPED_POLICY',
                  'SKIPPED_NOT_NEEDED', 'CANCELLED', 'EXPIRED', 'INVALIDATED'
              )
        ) AS terminal_work_item_count,
        (
            SELECT COUNT(*)
            FROM primary_outbox AS o
            WHERE o.tenant_id = r.tenant_id AND o.completion_run_id = r.run_id
        ) AS outbox_count,
        (
            SELECT COUNT(*)
            FROM primary_outbox AS o
            WHERE o.tenant_id = r.tenant_id AND o.completion_run_id = r.run_id
              AND o.delivered_at IS NULL
        ) AS undelivered_outbox_count
    FROM primary_runs AS r
    JOIN primary_completions AS c ON c.tenant_id = r.tenant_id AND c.run_id = r.run_id
    WHERE r.tenant_id = robata_canonical.current_tenant_id()
"""


class PostgresCanonicalReadModel:
    """Read-only PostgreSQL projection for API and frontend consumers.

    ``logical_invocation_id`` is the evidence scope because the current canonical
    inference evidence contract deliberately does not invent a ``run_id`` edge.
    A caller must provide a real linkage from its own application boundary rather
    than have this projection infer one from decoded payload bytes.
    """

    backend_kind = "POSTGRESQL"

    def __init__(self, authority: PostgresCanonicalAuthority) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        self._authority = authority

    @property
    def authority(self) -> PostgresCanonicalAuthority:
        """Return the sole tenant/RLS transaction authority used for all reads."""

        return self._authority

    @property
    def required_tables(self) -> tuple[str, ...]:
        """Return the checked-in migration tables this projection depends upon."""

        return _REQUIRED_TABLES

    @property
    def required_migration_ids(self) -> tuple[str, ...]:
        """Return the immutable migration IDs needed by this read projection."""

        return _REQUIRED_MIGRATION_IDS

    def verify_startup(self) -> PostgresReadModelStartup:
        """Verify migration tables and a nonempty transaction-local tenant context."""

        def operation(connection: PostgresConnection) -> PostgresReadModelStartup:
            tenant_id = _require_tenant_context(connection)
            rows = connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = ANY(%s)
                """,
                (self._authority.schema, list(_REQUIRED_TABLES)),
            ).fetchall()
            found = {_required_text(row, "table_name") for row in rows}
            missing = sorted(set(_REQUIRED_TABLES).difference(found))
            if missing:
                raise PostgresReadModelConfigurationError(
                    "PostgreSQL read-model migration is incomplete: " + ", ".join(missing)
                )
            migration_rows = connection.execute(
                """
                SELECT migration_id
                FROM robata_ops.schema_migrations
                WHERE migration_id = ANY(%s)
                """,
                (list(_REQUIRED_MIGRATION_IDS),),
            ).fetchall()
            applied_migration_ids = {_required_text(row, "migration_id") for row in migration_rows}
            missing_migration_ids = sorted(
                set(_REQUIRED_MIGRATION_IDS).difference(applied_migration_ids)
            )
            if missing_migration_ids:
                raise PostgresReadModelConfigurationError(
                    "PostgreSQL read-model migrations are missing: "
                    + ", ".join(missing_migration_ids)
                )
            return PostgresReadModelStartup(
                backend_kind=self.backend_kind,
                schema=self._authority.schema,
                tenant_id=tenant_id,
                required_tables=_REQUIRED_TABLES,
                required_migration_ids=_REQUIRED_MIGRATION_IDS,
            )

        return self._read("read_model.verify_startup", operation)

    def get_committed_run(self, run_id: str) -> CommittedRunSummary | None:
        """Return one committed run's metadata and durable operational counters."""

        checked_run_id = _nonempty(run_id, "run_id")

        def operation(connection: PostgresConnection) -> CommittedRunSummary | None:
            row = connection.execute(
                _COMMITTED_RUN_SELECT + " AND r.run_id = %s",
                (checked_run_id,),
            ).fetchone()
            return None if row is None else _committed_run_summary(row)

        return self._read("read_model.get_committed_run", operation)

    def list_committed_runs(self, *, limit: int = 50) -> tuple[CommittedRunSummary, ...]:
        """List completed runs in deterministic newest-completion order."""

        checked_limit = _page_size(limit)

        def operation(connection: PostgresConnection) -> tuple[CommittedRunSummary, ...]:
            rows = connection.execute(
                _COMMITTED_RUN_SELECT + " ORDER BY r.completed_at DESC, r.run_id ASC LIMIT %s",
                (checked_limit,),
            ).fetchall()
            return tuple(_committed_run_summary(row) for row in rows)

        return self._read("read_model.list_committed_runs", operation)

    def get_committed_run_detail(self, run_id: str) -> CommittedRunDetail | None:
        """Return a run's original committed documents after byte-level verification."""

        checked_run_id = _nonempty(run_id, "run_id")

        def operation(connection: PostgresConnection) -> CommittedRunDetail | None:
            summary_row = connection.execute(
                _COMMITTED_RUN_SELECT + " AND r.run_id = %s",
                (checked_run_id,),
            ).fetchone()
            if summary_row is None:
                return None
            detail_row = connection.execute(
                """
                SELECT
                    r.run_json,
                    r.run_json_sha256,
                    c.command_json,
                    c.command_json_sha256,
                    c.committed_json,
                    c.committed_json_sha256,
                    d.payload_json,
                    d.exact_bytes_sha256,
                    d.byte_count
                FROM primary_runs AS r
                JOIN primary_completions AS c ON c.tenant_id = r.tenant_id AND c.run_id = r.run_id
                JOIN detailed_results AS d
                  ON d.tenant_id = c.tenant_id AND d.artifact_id = c.detailed_result_artifact_id
                WHERE r.run_id = %s
                """,
                (checked_run_id,),
            ).fetchone()
            if detail_row is None:
                raise PostgresReadModelIntegrityError(
                    "committed run references a missing detailed result"
                )
            summary = _committed_run_summary(summary_row)
            return CommittedRunDetail(
                summary=summary,
                run_document=_canonical_document(
                    document_id=summary.run_id,
                    kind="PRIMARY_RUN",
                    raw=_required_bytes(detail_row, "run_json"),
                    expected_sha256=_required_text(detail_row, "run_json_sha256"),
                ),
                command_document=_canonical_document(
                    document_id=summary.run_id,
                    kind="PRIMARY_COMPLETION_COMMAND",
                    raw=_required_bytes(detail_row, "command_json"),
                    expected_sha256=_required_text(detail_row, "command_json_sha256"),
                ),
                committed_document=_canonical_document(
                    document_id=summary.run_id,
                    kind="COMMITTED_PRIMARY_COMPLETION",
                    raw=_required_bytes(detail_row, "committed_json"),
                    expected_sha256=_required_text(detail_row, "committed_json_sha256"),
                ),
                detailed_result_document=_canonical_document(
                    document_id=summary.detailed_result_artifact_id,
                    kind="DETAILED_RESULT",
                    raw=_required_bytes(detail_row, "payload_json"),
                    expected_sha256=_required_text(detail_row, "exact_bytes_sha256"),
                    expected_byte_count=_required_int(detail_row, "byte_count"),
                ),
            )

        return self._read("read_model.get_committed_run_detail", operation)

    def list_work_stages(self, run_id: str) -> tuple[WorkStageSummary, ...]:
        """Return scheduler state buckets for one run without mutating scheduler state."""

        checked_run_id = _nonempty(run_id, "run_id")

        def operation(connection: PostgresConnection) -> tuple[WorkStageSummary, ...]:
            rows = connection.execute(
                """
                SELECT
                    stage,
                    state,
                    COUNT(*) AS work_item_count,
                    COALESCE(SUM(attempt), 0) AS attempts_started,
                    MAX(updated_at) AS latest_updated_at
                FROM work_items
                WHERE tenant_id = robata_canonical.current_tenant_id() AND run_id = %s
                GROUP BY stage, state
                ORDER BY stage ASC, state ASC
                """,
                (checked_run_id,),
            ).fetchall()
            return tuple(
                WorkStageSummary(
                    stage=_required_text(row, "stage"),
                    state=_required_text(row, "state"),
                    work_item_count=_required_int(row, "work_item_count"),
                    attempts_started=_required_int(row, "attempts_started"),
                    latest_updated_at=_required_text(row, "latest_updated_at"),
                )
                for row in rows
            )

        return self._read("read_model.list_work_stages", operation)

    def list_outbox_delivery(
        self,
        run_id: str,
        *,
        limit: int = 100,
    ) -> tuple[OutboxDeliverySummary, ...]:
        """Return fenced delivery state for a committed run's outbox messages."""

        checked_run_id = _nonempty(run_id, "run_id")
        checked_limit = _page_size(limit)

        def operation(connection: PostgresConnection) -> tuple[OutboxDeliverySummary, ...]:
            rows = connection.execute(
                """
                SELECT
                    o.outbox_id,
                    o.outbox_ordinal,
                    o.assignment_logical_key,
                    o.payload_json_sha256,
                    o.delivered_at,
                    d.status AS delivery_status,
                    d.attempt_count,
                    d.next_attempt_at,
                    d.lease_epoch
                FROM primary_outbox AS o
                LEFT JOIN primary_outbox_deliveries AS d
                  ON d.tenant_id = o.tenant_id AND d.outbox_id = o.outbox_id
                WHERE o.tenant_id = robata_canonical.current_tenant_id()
                  AND o.completion_run_id = %s
                ORDER BY o.outbox_ordinal ASC, o.outbox_id ASC
                LIMIT %s
                """,
                (checked_run_id, checked_limit),
            ).fetchall()
            return tuple(
                OutboxDeliverySummary(
                    outbox_id=_required_text(row, "outbox_id"),
                    outbox_ordinal=_required_int(row, "outbox_ordinal"),
                    assignment_logical_key=_required_text(row, "assignment_logical_key"),
                    payload_sha256=_required_text(row, "payload_json_sha256"),
                    delivered_at=_optional_timestamp(row, "delivered_at"),
                    delivery_status=_optional_text(row, "delivery_status"),
                    attempt_count=_optional_int(row, "attempt_count"),
                    next_attempt_at=_optional_timestamp(row, "next_attempt_at"),
                    lease_epoch=_optional_int(row, "lease_epoch"),
                )
                for row in rows
            )

        return self._read("read_model.list_outbox_delivery", operation)

    def list_evidence(
        self,
        *,
        logical_invocation_id: str | None = None,
        limit: int = 100,
    ) -> tuple[EvidenceArtifactSummary, ...]:
        """List terminal inference evidence by its explicit logical invocation scope."""

        checked_limit = _page_size(limit)
        checked_invocation_id = (
            None
            if logical_invocation_id is None
            else _nonempty(logical_invocation_id, "logical_invocation_id")
        )

        def operation(connection: PostgresConnection) -> tuple[EvidenceArtifactSummary, ...]:
            where_sql = (
                " WHERE t.tenant_id = robata_canonical.current_tenant_id()"
                if checked_invocation_id is None
                else (
                    " WHERE t.tenant_id = robata_canonical.current_tenant_id()"
                    " AND t.logical_invocation_id = %s"
                )
            )
            params: Sequence[object] = (
                (checked_limit,)
                if checked_invocation_id is None
                else (checked_invocation_id, checked_limit)
            )
            rows = connection.execute(
                """
                SELECT
                    t.inference_id,
                    t.logical_invocation_id,
                    t.request_id,
                    t.status AS terminal_status,
                    t.shadow,
                    t.output_valid,
                    t.raw_artifact_id,
                    a.provider,
                    a.model_name,
                    a.model_version,
                    a.provider_request_id,
                    a.created_at,
                    raw.exact_bytes_sha256,
                    raw.byte_count,
                    raw.media_type
                FROM model_inference_terminals AS t
                JOIN inference_intents AS i
                  ON i.tenant_id = t.tenant_id
                 AND i.inference_id = t.inference_id
                 AND i.request_id = t.request_id
                LEFT JOIN raw_provider_artifacts AS a
                  ON a.tenant_id = t.tenant_id
                 AND a.inference_id = t.inference_id
                 AND a.artifact_id = t.raw_artifact_id
                LEFT JOIN raw_provider_responses AS raw
                  ON raw.tenant_id = t.tenant_id
                 AND raw.inference_id = t.inference_id
                 AND raw.artifact_id = t.raw_artifact_id
                """
                + where_sql
                + """
                ORDER BY a.created_at DESC NULLS LAST, t.inference_id ASC
                LIMIT %s
                """,
                params,
            ).fetchall()
            return tuple(_evidence_summary(row) for row in rows)

        return self._read("read_model.list_evidence", operation)

    def get_raw_provider_response(self, artifact_id: str) -> RawProviderResponse | None:
        """Return exact raw provider bytes only when their stored digest still verifies."""

        checked_artifact_id = _nonempty(artifact_id, "artifact_id")

        def operation(connection: PostgresConnection) -> RawProviderResponse | None:
            row = connection.execute(
                """
                SELECT
                    artifact_id,
                    inference_id,
                    request_id,
                    provider_request_id,
                    media_type,
                    exact_bytes_sha256,
                    byte_count,
                    raw_bytes
                FROM raw_provider_responses
                WHERE tenant_id = robata_canonical.current_tenant_id() AND artifact_id = %s
                """,
                (checked_artifact_id,),
            ).fetchone()
            if row is None:
                return None
            raw_bytes = _verified_bytes(
                raw=_required_bytes(row, "raw_bytes"),
                expected_sha256=_required_text(row, "exact_bytes_sha256"),
                expected_byte_count=_required_int(row, "byte_count"),
                label=f"raw provider response {checked_artifact_id}",
            )
            return RawProviderResponse(
                artifact_id=_required_text(row, "artifact_id"),
                inference_id=_required_text(row, "inference_id"),
                request_id=_required_text(row, "request_id"),
                provider_request_id=_required_text(row, "provider_request_id"),
                media_type=_required_text(row, "media_type"),
                exact_bytes_sha256=_required_text(row, "exact_bytes_sha256"),
                raw_bytes=raw_bytes,
            )

        return self._read("read_model.get_raw_provider_response", operation)

    def get_evidence_document(
        self,
        kind: EvidenceDocumentKind,
        document_id: str,
    ) -> CanonicalBytesDocument | None:
        """Expand a canonical evidence document without parsing or re-serializing it."""

        if not isinstance(kind, EvidenceDocumentKind):
            raise TypeError("kind must be EvidenceDocumentKind")
        checked_document_id = _nonempty(document_id, "document_id")
        table_name, identifier_column = _EVIDENCE_DOCUMENT_TABLES[kind]

        def operation(connection: PostgresConnection) -> CanonicalBytesDocument | None:
            # Both identifiers are module constants, never user-provided SQL.
            row = connection.execute(
                f"""
                SELECT {identifier_column} AS document_id, payload_json, payload_sha256
                FROM {table_name}
                WHERE tenant_id = robata_canonical.current_tenant_id() AND {identifier_column} = %s
                """,
                (checked_document_id,),
            ).fetchone()
            if row is None:
                return None
            return _canonical_document(
                document_id=_required_text(row, "document_id"),
                kind=kind.value,
                raw=_required_bytes(row, "payload_json"),
                expected_sha256=_required_text(row, "payload_sha256"),
            )

        return self._read("read_model.get_evidence_document", operation)

    def _read(
        self,
        operation_name: str,
        operation: Callable[[PostgresConnection], _ResultT],
    ) -> _ResultT:
        def guarded(connection: PostgresConnection) -> _ResultT:
            _require_tenant_context(connection)
            return operation(connection)

        try:
            return self._authority.run_authority_transaction(
                write=False,
                operation_name=operation_name,
                operation=guarded,
            )
        except PostgresReadModelError:
            raise
        except PostgresAuthorityError as error:
            raise PostgresReadModelStorageError(
                f"PostgreSQL read-model authority failed: {error}"
            ) from error
        except Exception as error:
            raise PostgresReadModelStorageError(
                f"PostgreSQL read-model query failed: {error}"
            ) from error


def _require_tenant_context(connection: PostgresConnection) -> str:
    row = connection.execute(
        "SELECT current_setting(%s, true) AS tenant_id",
        (_TENANT_SETTING,),
    ).fetchone()
    if row is None:
        raise PostgresReadModelConfigurationError(
            "PostgreSQL read model cannot verify transaction-local tenant context"
        )
    tenant_id = _optional_text(row, "tenant_id")
    if tenant_id is None:
        raise PostgresReadModelConfigurationError(
            "PostgreSQL read model requires tenant_setting='robata.tenant_id' and a tenant_id"
        )
    return tenant_id


def _committed_run_summary(row: Row) -> CommittedRunSummary:
    return CommittedRunSummary(
        run_id=_required_text(row, "run_id"),
        recording_identity=_required_text(row, "recording_identity"),
        mcap_id=_required_text(row, "mcap_id"),
        pipeline_version=_required_text(row, "pipeline_version"),
        config_sha256=_required_text(row, "config_sha256"),
        started_at=_required_text(row, "started_at"),
        completed_at=_required_text(row, "completed_at"),
        primary_status=_required_text(row, "primary_status"),
        run_version=_required_int(row, "run_version"),
        command_sha256=_required_text(row, "command_sha256"),
        committed_json_sha256=_required_text(row, "committed_json_sha256"),
        detailed_result_artifact_id=_required_text(row, "detailed_result_artifact_id"),
        work_item_count=_required_int(row, "work_item_count"),
        terminal_work_item_count=_required_int(row, "terminal_work_item_count"),
        outbox_count=_required_int(row, "outbox_count"),
        undelivered_outbox_count=_required_int(row, "undelivered_outbox_count"),
    )


def _evidence_summary(row: Row) -> EvidenceArtifactSummary:
    return EvidenceArtifactSummary(
        inference_id=_required_text(row, "inference_id"),
        logical_invocation_id=_required_text(row, "logical_invocation_id"),
        request_id=_required_text(row, "request_id"),
        terminal_status=_required_text(row, "terminal_status"),
        shadow=_required_bool(row, "shadow"),
        output_valid=_required_bool(row, "output_valid"),
        raw_artifact_id=_optional_text(row, "raw_artifact_id"),
        provider=_optional_text(row, "provider"),
        model_name=_optional_text(row, "model_name"),
        model_version=_optional_text(row, "model_version"),
        provider_request_id=_optional_text(row, "provider_request_id"),
        created_at=_optional_text(row, "created_at"),
        exact_bytes_sha256=_optional_text(row, "exact_bytes_sha256"),
        byte_count=_optional_int(row, "byte_count"),
        media_type=_optional_text(row, "media_type"),
    )


def _canonical_document(
    *,
    document_id: str,
    kind: str,
    raw: bytes,
    expected_sha256: str,
    expected_byte_count: int | None = None,
) -> CanonicalBytesDocument:
    verified = _verified_bytes(
        raw=raw,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label=f"canonical {kind} document {document_id}",
    )
    return CanonicalBytesDocument(
        document_id=document_id,
        kind=kind,
        exact_bytes=verified,
        sha256=expected_sha256,
    )


def _verified_bytes(
    *,
    raw: bytes,
    expected_sha256: str,
    expected_byte_count: int | None,
    label: str,
) -> bytes:
    if expected_byte_count is not None and len(raw) != expected_byte_count:
        raise PostgresReadModelIntegrityError(
            f"{label} byte count does not match persisted metadata"
        )
    actual_sha256 = sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PostgresReadModelIntegrityError(f"{label} SHA-256 does not match persisted metadata")
    return raw


def _required_text(row: Row, name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise PostgresReadModelStorageError(f"PostgreSQL read-model row lacks nonempty {name}")
    return value


def _optional_text(row: Row, name: str) -> str | None:
    value = row.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PostgresReadModelStorageError(f"PostgreSQL read-model row has invalid {name}")
    return value


def _optional_timestamp(row: Row, name: str) -> str | None:
    value = row.get(name)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    raise PostgresReadModelStorageError(f"PostgreSQL read-model row has invalid {name}")


def _required_int(row: Row, name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PostgresReadModelStorageError(f"PostgreSQL read-model row has invalid {name}")
    return value


def _optional_int(row: Row, name: str) -> int | None:
    value = row.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PostgresReadModelStorageError(f"PostgreSQL read-model row has invalid {name}")
    return value


def _required_bool(row: Row, name: str) -> bool:
    value = row.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise PostgresReadModelStorageError(f"PostgreSQL read-model row has invalid {name}")


def _required_bytes(row: Row, name: str) -> bytes:
    value = row.get(name)
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return bytes(value)
    raise PostgresReadModelStorageError(f"PostgreSQL read-model row has invalid {name}")


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _page_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("limit must be an integer")
    if not 1 <= value <= _MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
    return value


__all__ = [
    "CanonicalBytesDocument",
    "CommittedRunDetail",
    "CommittedRunSummary",
    "EvidenceArtifactSummary",
    "EvidenceDocumentKind",
    "OutboxDeliverySummary",
    "PostgresCanonicalReadModel",
    "PostgresReadModelConfigurationError",
    "PostgresReadModelError",
    "PostgresReadModelIntegrityError",
    "PostgresReadModelStartup",
    "PostgresReadModelStorageError",
    "RawProviderResponse",
    "WorkStageSummary",
]
