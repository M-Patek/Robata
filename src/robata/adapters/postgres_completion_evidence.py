"""PostgreSQL/Supabase adapters for canonical completion, barriers, and outbox delivery.

The canonical payload format remains the existing registered Pydantic/wire
contracts.  This module deliberately reuses the already-tested aggregate
decision logic from the local adapters while replacing its storage boundary
with a PostgreSQL transaction supplied by :class:`PostgresCanonicalAuthority`.
No SQLite connection, database file, or local fallback is opened here.

The compatibility cursor is intentionally narrow: it only translates DB-API
placeholders and turns the outer PostgreSQL transaction into the transaction
surface expected by the shared aggregate logic.  PostgreSQL remains the sole
authority, with RLS tenant context and the migration's `bytea` canonical facts.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar, cast

from robata.adapters.postgres_authority import (
    PostgresAuthorityStorageError,
    PostgresCanonicalAuthority,
    PostgresConnection,
    PostgresCursor,
)
from robata.adapters.postgres_r2_artifacts import PostgresR2ArtifactAuthority
from robata.adapters.sqlite_barrier import SQLiteBarrierStorage
from robata.adapters.sqlite_inference_evidence import (
    _CONTRACT_VERSION,
    INFERENCE_ATTEMPT_SELECTION_SCHEMA_ID,
    INFERENCE_INTENT_SCHEMA_ID,
    MODEL_INFERENCE_SCHEMA_ID,
    PARSED_PROVIDER_CLAIM_SCHEMA_ID,
    RAW_PROVIDER_RESPONSE_SCHEMA_ID,
    SELECTED_ATTEMPT_OUTPUT_SCHEMA_ID,
    SQLiteInferenceEvidenceLedger,
    SQLiteInferenceEvidenceLedgerError,
    _LedgerCache,
)
from robata.adapters.sqlite_outbox import (
    SQLitePrimaryOutboxDeliveryStore,
    _fencing_token,
    _require_now,
    _rfc3339,
    _validate_lease_duration,
)
from robata.adapters.sqlite_primary_completion import SQLitePrimaryCompletionRepository
from robata.application.canonical.primary_completion import (
    CommittedPrimaryCompletion,
    PreparedPrimaryCompletionCommand,
    PrimaryCompletionCommand,
    PrimaryCompletionCommitResult,
    PrimaryCompletionError,
    PrimaryCompletionErrorCode,
)
from robata.application.canonical_run_membership import (
    CanonicalProcessingRunContext,
    CanonicalProcessingRunRecord,
)
from robata.contracts.schema_registry import (
    SchemaRegistry,
    SchemaRegistryError,
    default_schema_registry,
)
from robata.event_pipeline.identity_registry import EventIdentityOutboxRecord, EventRegistrySnapshot
from robata.inference.call_barrier import InferenceCallBarrierError
from robata.inference.enrichment import ENRICHED_OUTPUT_SCHEMA_ID, ENRICHED_OUTPUT_SCHEMA_VERSION
from robata.queue.outbox import (
    Clock,
    OutboxDeliveryClaim,
    OutboxDeliveryError,
    OutboxDeliverySnapshot,
    OutboxRetryPolicy,
)
from robata.runtime.observability import RuntimeAttributeValue, RuntimeObserver

_ResultT = TypeVar("_ResultT")


class PostgresCanonicalAdapterError(RuntimeError):
    """A PostgreSQL canonical adapter cannot perform an authoritative operation."""


class PostgresBarrierStorageError(InferenceCallBarrierError, PostgresCanonicalAdapterError):
    """PostgreSQL barrier persistence or its canonical bytes failed closed."""


class PostgresInferenceEvidenceLedgerError(
    SQLiteInferenceEvidenceLedgerError,
    PostgresCanonicalAdapterError,
):
    """PostgreSQL persistence of the immutable inference evidence graph failed."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _PostgresCompatibilityOperationalError(sqlite3.OperationalError):
    """SQLite-compatible surface that retains a PostgreSQL SQLSTATE for retries."""

    sqlstate: str | None = None


class _PostgresCursorCompatibility:
    """Map an injected PostgreSQL cursor to the small SQLite adapter read surface."""

    def __init__(self, cursor: PostgresCursor) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def fetchone(self) -> dict[str, object] | None:
        row = self._cursor.fetchone()
        return None if row is None else _postgres_row(row)

    def fetchall(self) -> tuple[dict[str, object], ...]:
        return tuple(_postgres_row(row) for row in self._cursor.fetchall())


class _PostgresConnectionCompatibility:
    """Translate existing parameterized aggregate SQL to PostgreSQL DB-API SQL.

    The outer ``PostgresCanonicalAuthority`` already owns the actual transaction.
    ``commit``, ``rollback`` and ``close`` are intentionally no-ops so a local
    aggregate replay cannot accidentally split the shared PostgreSQL transaction.
    """

    def __init__(self, connection: PostgresConnection) -> None:
        self._connection = connection

    @property
    def in_transaction(self) -> bool:
        return True

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> _PostgresCursorCompatibility:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if query.strip().upper().startswith("BEGIN"):
            return _PostgresCursorCompatibility(_EmptyPostgresCursor())
        try:
            cursor = self._connection.execute(
                _postgres_placeholders(query),
                None if params is None else tuple(_postgres_parameter(value) for value in params),
            )
        except Exception as error:
            # Existing aggregate code maps sqlite errors to its public, stable
            # domain errors. Preserve that behavior without exposing driver types.
            converted = _PostgresCompatibilityOperationalError(str(error))
            sqlstate = getattr(error, "sqlstate", None)
            if isinstance(sqlstate, str):
                converted.sqlstate = sqlstate
            raise converted from error
        return _PostgresCursorCompatibility(cursor)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class _EmptyPostgresCursor:
    """No-op cursor returned for nested SQLite-style BEGIN statements."""

    @property
    def rowcount(self) -> int:
        return 0

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> tuple[dict[str, object], ...]:
        return ()


def _postgres_placeholders(query: str) -> str:
    """Translate local aggregate SQL without interpolating values.

    The local primary aggregate has one SQLite conflict target that omits the
    production tenant partition. PostgreSQL must name the composite authority
    key so the defaulted tenant value participates in the idempotent insert.
    """

    translated = query.replace("?", "%s")
    return translated.replace(
        "ON CONFLICT (recording_identity) DO NOTHING",
        "ON CONFLICT (tenant_id, recording_identity) DO NOTHING",
    )


def _postgres_parameter(value: object) -> object:
    if isinstance(value, memoryview):
        return value.tobytes()

    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    return value


def _postgres_row(row: object) -> dict[str, object]:
    if not isinstance(row, dict):
        try:
            converted = dict(cast(Any, row))
        except (TypeError, ValueError) as error:
            raise PostgresCanonicalAdapterError("PostgreSQL row must be mapping-shaped") from error
    else:
        converted = dict(row)
    return {key: _postgres_value(value) for key, value in converted.items()}


def _postgres_value(value: object) -> object:
    if isinstance(value, bool):
        # Keep the inherited SQLite aggregate readers strict and deterministic.
        return int(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    return value


class _PostgresAdapterTransactionMixin:
    """Bind a local aggregate method to one authority-owned PostgreSQL transaction."""

    _authority: PostgresCanonicalAuthority
    _active_connection: ContextVar[_PostgresConnectionCompatibility | None]

    def _run_postgres_operation(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        active = self._active_connection.get()
        if active is not None:
            return operation()

        def execute(connection: PostgresConnection) -> _ResultT:
            token = self._active_connection.set(_PostgresConnectionCompatibility(connection))
            try:
                return operation()
            finally:
                self._active_connection.reset(token)

        try:
            return self._authority.run_authority_transaction(
                write=write,
                operation_name=operation_name,
                operation=execute,
            )
        except PostgresAuthorityStorageError as error:
            raise PostgresCanonicalAdapterError(
                f"PostgreSQL canonical authority operation failed: {operation_name}"
            ) from error

    def _postgres_compat_connection(self) -> _PostgresConnectionCompatibility:
        connection = self._active_connection.get()
        if connection is None:
            raise PostgresCanonicalAdapterError(
                "PostgreSQL adapter operation was invoked outside its authority transaction"
            )
        return connection


class PostgresBarrierStorage(_PostgresAdapterTransactionMixin, SQLiteBarrierStorage):
    """PostgreSQL-backed implementation of generic and inference-call barriers.

    ``SQLiteBarrierStorage`` contributes only pure model validation, byte checks,
    and deterministic state derivation. Its file initialization and transaction
    methods are bypassed entirely.
    """

    def __init__(
        self,
        authority: PostgresCanonicalAuthority,
        *,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        self._authority = authority
        self._active_connection = ContextVar("postgres_barrier_connection", default=None)
        self._runtime_observer = runtime_observer
        self._database_path = Path(f"postgres-{authority.schema}-barriers")

    def _transaction(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[Any], _ResultT],
    ) -> _ResultT:
        return self._run_postgres_operation(
            write=write,
            operation_name=f"barrier.{operation_name}",
            operation=lambda: operation(self._postgres_compat_connection()),
        )


class PostgresInferenceEvidenceLedger(
    _PostgresAdapterTransactionMixin,
    SQLiteInferenceEvidenceLedger,
):
    """PostgreSQL-backed, append-only inference evidence graph.

    The model and lineage validation rules are deliberately shared with the local
    ledger. PostgreSQL does not reuse the local process cache: each operation
    reconstructs the typed graph inside the authority-owned snapshot transaction.
    That gives independent workers a complete persisted-state audit before they
    append an immutable fact, while serializable retries preserve idempotency.
    """

    def __init__(
        self,
        authority: PostgresCanonicalAuthority,
        schema_registry: SchemaRegistry,
        *,
        runtime_observer: RuntimeObserver | None = None,
        artifact_authority: PostgresR2ArtifactAuthority | None = None,
    ) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        if not isinstance(schema_registry, SchemaRegistry):
            raise TypeError("schema_registry must be a SchemaRegistry")
        if artifact_authority is not None and not isinstance(
            artifact_authority, PostgresR2ArtifactAuthority
        ):
            raise TypeError("artifact_authority must be PostgresR2ArtifactAuthority or None")
        self._authority = authority
        self._artifact_authority = artifact_authority
        self._active_connection = ContextVar("postgres_inference_evidence_connection", default=None)
        self._state_lock = RLock()
        self._runtime_observer = runtime_observer
        self._schema_registry = schema_registry
        self._database_path = Path(f"postgres-{authority.schema}-inference-evidence")
        self._raw_bytes_cas_root: Path | None = None
        self._closed = False
        self._cache_validation_connection = None
        self._cache = cast(_LedgerCache, None)
        self._cache_data_version: int | None = None
        try:
            self._pins = {
                "intent": self._resolve_pin(INFERENCE_INTENT_SCHEMA_ID, _CONTRACT_VERSION),
                "terminal": self._resolve_pin(MODEL_INFERENCE_SCHEMA_ID, _CONTRACT_VERSION),
                "selection": self._resolve_pin(
                    INFERENCE_ATTEMPT_SELECTION_SCHEMA_ID,
                    _CONTRACT_VERSION,
                ),
                "raw_artifact": self._resolve_pin(
                    RAW_PROVIDER_RESPONSE_SCHEMA_ID,
                    _CONTRACT_VERSION,
                ),
                "parsed": self._resolve_pin(PARSED_PROVIDER_CLAIM_SCHEMA_ID, _CONTRACT_VERSION),
                "selected": self._resolve_pin(
                    SELECTED_ATTEMPT_OUTPUT_SCHEMA_ID,
                    _CONTRACT_VERSION,
                ),
                "enriched": self._resolve_pin(
                    ENRICHED_OUTPUT_SCHEMA_ID,
                    ENRICHED_OUTPUT_SCHEMA_VERSION,
                ),
            }
        except SQLiteInferenceEvidenceLedgerError as error:
            raise PostgresInferenceEvidenceLedgerError(
                "cannot resolve inference evidence schema governance"
            ) from error

    @property
    def artifact_authority(self) -> PostgresR2ArtifactAuthority | None:
        """Return the optional required-production R2 mirror authority."""

        return self._artifact_authority

    def close(self) -> None:
        """Mark this composition-owned adapter closed without owning a DB connection."""

        with self._state_lock:
            self._closed = True

    def verify_integrity(self) -> None:
        """Perform a complete typed-graph and canonical-byte audit in PostgreSQL."""

        with self._state_lock:
            self._require_open()

            def audit() -> None:
                connection: Any = self._postgres_compat_connection()
                self._cache = self._cache_from_database(connection, self._load_state(connection))

            try:
                self._run_postgres_operation(
                    write=False,
                    operation_name="inference_evidence.verify_integrity",
                    operation=audit,
                )
            except PostgresCanonicalAdapterError as error:
                raise PostgresInferenceEvidenceLedgerError(
                    "cannot verify PostgreSQL inference evidence integrity"
                ) from error
            if self._artifact_authority is not None:
                try:
                    self._artifact_authority.verify_records(self._cache.state.raw.values())
                except Exception as error:
                    raise PostgresInferenceEvidenceLedgerError(
                        "cannot verify immutable R2 raw-provider evidence mirrors"
                    ) from error

    def verify_completion_seal(self) -> None:
        """Require a fresh persisted-state evidence audit at the completion boundary."""

        # PostgreSQL writers are independent processes, so a process-local cache
        # cannot prove currentness. A read-only serializable snapshot is the seal.
        self.verify_integrity()

    def _require_open(self) -> None:
        if self._closed:
            raise PostgresInferenceEvidenceLedgerError(
                "PostgreSQL inference evidence ledger is closed"
            )

    def _cached_read(
        self,
        *,
        operation_name: str,
        cached: Callable[[_LedgerCache], _ResultT],
        database: Callable[[Any], _ResultT],
    ) -> _ResultT:
        """Read from one fully audited PostgreSQL snapshot, never a stale local cache."""

        del database
        with self._state_lock:
            self._require_open()

            def read() -> _ResultT:
                connection: Any = self._postgres_compat_connection()
                cache = self._cache_from_database(connection, self._load_state(connection))
                self._cache = cache
                return deepcopy(cached(cache))

            try:
                return self._run_postgres_operation(
                    write=False,
                    operation_name=f"inference_evidence.{operation_name}",
                    operation=read,
                )
            except PostgresCanonicalAdapterError as error:
                raise PostgresInferenceEvidenceLedgerError(
                    f"cannot read PostgreSQL inference evidence: {operation_name}"
                ) from error

    def _cached_write(
        self,
        *,
        operation_name: str,
        operation: Callable[[Any, _LedgerCache], tuple[_ResultT, _LedgerCache]],
    ) -> _ResultT:
        """Append one fact against a complete graph in one serializable transaction."""

        with self._state_lock:
            self._require_open()

            def append() -> _ResultT:
                connection: Any = self._postgres_compat_connection()
                cache = self._cache_from_database(connection, self._load_state(connection))
                previous_validation_connection = self._cache_validation_connection
                self._cache_validation_connection = cast(Any, connection)
                try:
                    result, committed_cache = operation(connection, cache)
                    self._cache = committed_cache
                    return deepcopy(result)
                finally:
                    self._cache_validation_connection = previous_validation_connection

            try:
                return self._run_postgres_operation(
                    write=True,
                    operation_name=f"inference_evidence.{operation_name}",
                    operation=append,
                )
            except SQLiteInferenceEvidenceLedgerError:
                raise
            except sqlite3.IntegrityError as error:
                raise PostgresInferenceEvidenceLedgerError(
                    "PostgreSQL rejected append-only inference evidence"
                ) from error
            except sqlite3.Error as error:
                raise PostgresInferenceEvidenceLedgerError(
                    "PostgreSQL inference evidence transaction failed"
                ) from error
            except PostgresCanonicalAdapterError as error:
                raise PostgresInferenceEvidenceLedgerError(
                    f"cannot append PostgreSQL inference evidence: {operation_name}"
                ) from error

    def get(self, artifact_id: str) -> Any:
        record = SQLiteInferenceEvidenceLedger.get(self, artifact_id)
        if self._artifact_authority is not None:
            self._artifact_authority.verify_record(record)
        return record

    def list_records(self) -> tuple[Any, ...]:
        records = SQLiteInferenceEvidenceLedger.list_records(self)
        if self._artifact_authority is not None:
            self._artifact_authority.verify_records(records)
        return records

    def _persist_raw_bytes(self, record: Any) -> bytes:
        """Mirror raw evidence to immutable R2 before its PostgreSQL append."""

        if self._artifact_authority is not None:
            self._artifact_authority.stage_and_verify(record)
        return bytes(record.data)


class PostgresPrimaryCompletionRepository(
    _PostgresAdapterTransactionMixin,
    SQLitePrimaryCompletionRepository,
):
    """PostgreSQL aggregate authority for run completion, identity, and outbox facts."""

    def __init__(
        self,
        authority: PostgresCanonicalAuthority,
        *,
        registry: SchemaRegistry | None = None,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        self._authority = authority
        self._active_connection = ContextVar("postgres_primary_completion_connection", default=None)
        self._runtime_observer = runtime_observer
        self._path = Path(f"postgres-{authority.schema}-primary-completion")
        try:
            self._registry = registry or default_schema_registry()
            self._outbox_schema_ref = self._registry.resolve_version(
                "https://schemas.robata.dev/event-identity-outbox-record",
                "1.0.0",
            ).ref
        except SchemaRegistryError as error:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                f"cannot resolve primary completion schema governance: {error}",
            ) from error

    def begin_run(
        self,
        context: CanonicalProcessingRunContext,
    ) -> CanonicalProcessingRunRecord:
        def begin() -> CanonicalProcessingRunRecord:
            if isinstance(context, CanonicalProcessingRunContext):
                self._lock_recording(context.recording_identity)
            return SQLitePrimaryCompletionRepository.begin_run(self, context)

        return self._run_postgres_operation(
            write=True,
            operation_name="primary_completion.begin_run",
            operation=begin,
        )

    def snapshot(self, recording_identity: str) -> EventRegistrySnapshot:
        def snapshot() -> EventRegistrySnapshot:
            if isinstance(recording_identity, str) and recording_identity:
                self._lock_recording(recording_identity)
            return SQLitePrimaryCompletionRepository.snapshot(self, recording_identity)

        return self._run_postgres_operation(
            write=True,
            operation_name="primary_completion.snapshot",
            operation=snapshot,
        )

    def get(self, run_id: str) -> CommittedPrimaryCompletion | None:
        return self._run_postgres_operation(
            write=False,
            operation_name="primary_completion.get",
            operation=lambda: SQLitePrimaryCompletionRepository.get(self, run_id),
        )

    def list_outbox(self, recording_identity: str) -> tuple[EventIdentityOutboxRecord, ...]:
        return self._run_postgres_operation(
            write=False,
            operation_name="primary_completion.list_outbox",
            operation=lambda: SQLitePrimaryCompletionRepository.list_outbox(
                self, recording_identity
            ),
        )

    def _commit_checked(
        self,
        checked: PrimaryCompletionCommand,
        *,
        prepared: PreparedPrimaryCompletionCommand | None = None,
    ) -> PrimaryCompletionCommitResult:
        def commit() -> PrimaryCompletionCommitResult:
            self._lock_recording(checked.detail.recording_identity)
            return SQLitePrimaryCompletionRepository._commit_checked(
                self,
                checked,
                prepared=prepared,
            )

        return self._run_postgres_operation(
            write=True,
            operation_name="primary_completion.commit",
            operation=commit,
        )

    def _lock_recording(self, recording_identity: str) -> None:
        """Serialize aggregate mutations for one recording without a process-local lock."""

        self._postgres_compat_connection().execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (recording_identity,),
        )

    def _connect(self) -> Any:
        return self._postgres_compat_connection()

    @contextmanager
    def _observed_transaction_scope(
        self,
        connection: Any,
        *,
        write: bool,
        operation_name: str,
    ) -> Iterator[dict[str, RuntimeAttributeValue]]:
        del connection
        yield {"operation": operation_name, "write": write}

    def _commit_observed(
        self,
        connection: Any,
        attributes: dict[str, RuntimeAttributeValue],
        *,
        use_commit_hook: bool = False,
    ) -> None:
        del connection, attributes, use_commit_hook

    def _rollback_observed(
        self, connection: Any, attributes: dict[str, RuntimeAttributeValue]
    ) -> None:
        del connection, attributes


class PostgresPrimaryOutboxDeliveryStore(
    _PostgresAdapterTransactionMixin,
    SQLitePrimaryOutboxDeliveryStore,
):
    """PostgreSQL fenced delivery state over canonical primary outbox facts."""

    def __init__(
        self,
        authority: PostgresCanonicalAuthority,
        *,
        retry_policy: OutboxRetryPolicy,
        clock: Clock | None = None,
        registry: SchemaRegistry | None = None,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        if not isinstance(retry_policy, OutboxRetryPolicy):
            raise TypeError("retry_policy must be OutboxRetryPolicy")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._authority = authority
        self._active_connection = ContextVar("postgres_outbox_delivery_connection", default=None)
        self._path = Path(f"postgres-{authority.schema}-outbox-delivery")
        self._retry_policy = retry_policy
        self._clock: Clock = clock if clock is not None else _utc_now
        self._runtime_observer = runtime_observer
        try:
            self._registry = registry or default_schema_registry()
        except SchemaRegistryError as error:
            raise PostgresCanonicalAdapterError(
                "cannot resolve primary outbox schema governance"
            ) from error

    def claim(
        self,
        *,
        worker_id: str,
        lease_duration: timedelta,
    ) -> OutboxDeliveryClaim | None:
        """Lease one eligible row with PostgreSQL row-level skip locking.

        The inherited local store is used for wire decoding and fence checks, but
        PostgreSQL workers must never select the same eligible delivery row and
        then race on its conditional update.
        """

        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a nonempty string")
        _validate_lease_duration(lease_duration)
        now = _require_now(self._clock)
        now_text = _rfc3339(now)
        expires_at = _rfc3339(now + lease_duration)

        def claim() -> OutboxDeliveryClaim | None:
            connection: Any = self._postgres_compat_connection()
            self._discover(connection, now_text)
            self._expire_leases(connection, now_text)
            row = connection.execute(
                """
                SELECT d.outbox_id, d.attempt_count, d.lease_epoch
                FROM primary_outbox_deliveries AS d
                JOIN primary_outbox AS o ON o.outbox_id = d.outbox_id
                WHERE o.delivered_at IS NULL
                  AND d.status IN ('PENDING', 'RETRY_WAIT')
                  AND d.next_attempt_at <= ?
                ORDER BY o.completion_run_id, o.outbox_ordinal, o.outbox_id
                LIMIT 1
                FOR UPDATE OF d SKIP LOCKED
                """,
                (now_text,),
            ).fetchone()
            if row is None:
                return None
            outbox_id = str(row["outbox_id"])
            previous_epoch = int(cast(Any, row["lease_epoch"]))
            lease_epoch = previous_epoch + 1
            attempt_count = int(cast(Any, row["attempt_count"])) + 1
            token = _fencing_token(outbox_id, lease_epoch)
            cursor = connection.execute(
                """
                UPDATE primary_outbox_deliveries
                SET status = 'LEASED',
                    attempt_count = ?,
                    lease_epoch = ?,
                    fencing_token = ?,
                    claimed_by = ?,
                    lease_expires_at = ?,
                    last_error = NULL
                WHERE outbox_id = ?
                  AND status IN ('PENDING', 'RETRY_WAIT')
                  AND lease_epoch = ?
                """,
                (
                    attempt_count,
                    lease_epoch,
                    token,
                    worker_id,
                    expires_at,
                    outbox_id,
                    previous_epoch,
                ),
            )
            if cursor.rowcount != 1:
                raise OutboxDeliveryError("outbox claim lost its atomic state transition")
            claimed = self._joined_row(connection, outbox_id)
            return OutboxDeliveryClaim(
                message=self._message_from_row(claimed),
                delivery=self._snapshot_from_row(claimed),
            )

        try:
            return self._run_postgres_operation(
                write=True,
                operation_name="outbox_delivery.claim",
                operation=claim,
            )
        except OutboxDeliveryError:
            raise
        except (PostgresCanonicalAdapterError, sqlite3.Error, TypeError, ValueError) as error:
            raise OutboxDeliveryError(
                f"cannot claim PostgreSQL primary outbox row: {error}"
            ) from error

    def acknowledge(self, claim: OutboxDeliveryClaim) -> OutboxDeliverySnapshot:
        return self._run_postgres_operation(
            write=True,
            operation_name="outbox_delivery.acknowledge",
            operation=lambda: SQLitePrimaryOutboxDeliveryStore.acknowledge(self, claim),
        )

    def record_failure(self, claim: OutboxDeliveryClaim, error: str) -> OutboxDeliverySnapshot:
        return self._run_postgres_operation(
            write=True,
            operation_name="outbox_delivery.record_failure",
            operation=lambda: SQLitePrimaryOutboxDeliveryStore.record_failure(self, claim, error),
        )

    def get(self, outbox_id: str) -> OutboxDeliverySnapshot | None:
        return self._run_postgres_operation(
            write=False,
            operation_name="outbox_delivery.get",
            operation=lambda: SQLitePrimaryOutboxDeliveryStore.get(self, outbox_id),
        )

    def list_dead_letters(self, *, limit: int = 100) -> tuple[OutboxDeliverySnapshot, ...]:
        return self._run_postgres_operation(
            write=False,
            operation_name="outbox_delivery.list_dead_letters",
            operation=lambda: SQLitePrimaryOutboxDeliveryStore.list_dead_letters(self, limit=limit),
        )

    def _connect(self) -> Any:
        return self._postgres_compat_connection()


def verify_completion_evidence_schema(authority: PostgresCanonicalAuthority) -> None:
    """Check P23 tables and forced RLS before canonical work starts."""

    required = (
        "primary_runs",
        "event_registry_partitions",
        "stable_event_identities",
        "event_identity_assignments",
        "event_identity_relations",
        "action_event_publications",
        "detailed_results",
        "primary_completions",
        "primary_outbox",
        "primary_outbox_deliveries",
        "inference_intents",
        "raw_provider_responses",
        "raw_provider_r2_artifact_receipts",
        "raw_provider_r2_artifact_observations",
        "model_inference_terminals",
        "raw_provider_artifacts",
        "inference_attempt_selections",
        "parsed_provider_claims",
        "selected_attempt_outputs",
        "enriched_provider_outputs",
        "calibration_artifacts",
        "inference_calibration_associations",
        "barrier_definitions",
        "barrier_states",
        "barrier_members",
        "inference_call_barrier_definitions",
        "inference_call_part_completions",
        "inference_call_reductions",
    )

    def operation(connection: PostgresConnection) -> None:
        rows = connection.execute(
            """
            SELECT c.relname AS table_name,
                   c.relrowsecurity AS row_security,
                   c.relforcerowsecurity AS force_row_security
            FROM pg_catalog.pg_class AS c
            JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
              AND c.relkind = 'r'
              AND c.relname = ANY(%s)
            """,
            (authority.schema, list(required)),
        ).fetchall()
        found = {str(row["table_name"]) for row in rows}
        missing = sorted(set(required).difference(found))
        if missing:
            raise PostgresCanonicalAdapterError(
                "PostgreSQL completion/evidence migration is incomplete: " + ", ".join(missing)
            )
        unprotected = sorted(
            str(row["table_name"])
            for row in rows
            if not bool(row["row_security"]) or not bool(row["force_row_security"])
        )
        if unprotected:
            raise PostgresCanonicalAdapterError(
                "PostgreSQL completion/evidence tables must force row-level security: "
                + ", ".join(unprotected)
            )

    authority.run_authority_transaction(
        write=False,
        operation_name="verify_completion_evidence_schema",
        operation=operation,
    )


__all__ = [
    "PostgresBarrierStorage",
    "PostgresBarrierStorageError",
    "PostgresCanonicalAdapterError",
    "PostgresInferenceEvidenceLedger",
    "PostgresInferenceEvidenceLedgerError",
    "PostgresPrimaryCompletionRepository",
    "PostgresPrimaryOutboxDeliveryStore",
    "verify_completion_evidence_schema",
]
