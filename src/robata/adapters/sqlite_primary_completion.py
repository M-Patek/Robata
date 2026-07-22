"""Aggregate SQLite authority for local canonical primary completion.

This adapter owns one database and one final ``BEGIN IMMEDIATE`` transaction.
It does not call the standalone identity or revision adapters.  Large-scale
production storage and delivery remain external policy decisions; the local
adapter persists exact detailed-result bytes and pending outbox records so that
restart and replay semantics are executable now.
"""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from functools import cache
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ValidationError

from robata.application.canonical.action_event_revision import (
    PreparedInitialActionEventRevision,
)
from robata.application.canonical.primary_completion import (
    CommittedPrimaryCompletion,
    PrimaryCompletionCommand,
    PrimaryCompletionCommitResult,
    PrimaryCompletionError,
    PrimaryCompletionErrorCode,
)
from robata.application.canonical_run_membership import (
    CanonicalProcessingRunContext,
    CanonicalProcessingRunRecord,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.primary_completion import validate_registered_primary_completion_record
from robata.contracts.schema_registry import (
    SchemaRegistry,
    SchemaRegistryError,
    default_schema_registry,
)
from robata.event_pipeline.identity_registry import (
    EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_ID,
    EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_VERSION,
    EventIdentityAssignment,
    EventIdentityBatchResult,
    EventIdentityOutboxRecord,
    EventIdentityOutboxWireRecord,
    EventIdentityRelation,
    EventRegistrySnapshot,
    StableEventIdentity,
    validate_registered_event_identity_outbox_wire_record,
)

_APPLICATION_ID: Final = 0x52504341  # "RPCA"
_SCHEMA_VERSION: Final = 2
_BUSY_TIMEOUT_MS: Final = 30_000


def _immutable_table_triggers(table: str, label: str) -> tuple[str, str]:
    return (
        f"""
        CREATE TRIGGER {table}_no_update
        BEFORE UPDATE ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{label} are append-only');
        END
        """,
        f"""
        CREATE TRIGGER {table}_no_delete
        BEFORE DELETE ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{label} are append-only');
        END
        """,
    )


_OUTBOX_DELIVERY_SCHEMA_STATEMENTS: Final = (
    """
    CREATE TABLE primary_outbox_deliveries (
        outbox_id TEXT PRIMARY KEY,
        status TEXT NOT NULL CHECK (
            status IN ('PENDING', 'LEASED', 'RETRY_WAIT', 'DELIVERED', 'DEAD_LETTER')
        ),
        attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
        lease_epoch INTEGER NOT NULL CHECK (lease_epoch >= 0),
        fencing_token TEXT,
        claimed_by TEXT,
        lease_expires_at TEXT,
        next_attempt_at TEXT NOT NULL,
        retry_policy_version TEXT NOT NULL,
        max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
        base_delay_seconds REAL NOT NULL CHECK (base_delay_seconds >= 0),
        max_delay_seconds REAL NOT NULL CHECK (
            max_delay_seconds >= base_delay_seconds
        ),
        last_error TEXT,
        delivered_at TEXT,
        dead_lettered_at TEXT,
        FOREIGN KEY (outbox_id) REFERENCES primary_outbox (outbox_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        CHECK (
            status <> 'LEASED'
            OR (
                fencing_token IS NOT NULL
                AND claimed_by IS NOT NULL
                AND lease_expires_at IS NOT NULL
            )
        ),
        CHECK ((status = 'DELIVERED') = (delivered_at IS NOT NULL)),
        CHECK ((status = 'DEAD_LETTER') = (dead_lettered_at IS NOT NULL))
    )
    """,
    """
    CREATE INDEX primary_outbox_delivery_claim_idx
        ON primary_outbox_deliveries (status, next_attempt_at, outbox_id)
    """,
    """
    CREATE TRIGGER primary_outbox_immutable_fields
    BEFORE UPDATE OF
        outbox_id,
        completion_run_id,
        recording_identity,
        outbox_ordinal,
        assignment_logical_key,
        payload_json,
        payload_json_sha256
    ON primary_outbox
    BEGIN
        SELECT RAISE(ABORT, 'primary outbox facts are append-only');
    END
    """,
    """
    CREATE TRIGGER primary_outbox_delivery_is_monotonic
    BEFORE UPDATE OF delivered_at ON primary_outbox
    WHEN OLD.delivered_at IS NOT NULL OR NEW.delivered_at IS NULL
    BEGIN
        SELECT RAISE(ABORT, 'primary outbox delivery acknowledgement is monotonic');
    END
    """,
    """
    CREATE TRIGGER primary_outbox_deliveries_no_delete
    BEFORE DELETE ON primary_outbox_deliveries
    BEGIN
        SELECT RAISE(ABORT, 'primary outbox delivery rows cannot be deleted');
    END
    """,
)


_SCHEMA_STATEMENTS: Final = (
    """
    CREATE TABLE primary_runs (
        run_id TEXT PRIMARY KEY,
        recording_identity TEXT NOT NULL,
        mcap_id TEXT NOT NULL,
        pipeline_version TEXT NOT NULL,
        config_sha256 TEXT NOT NULL,
        started_at TEXT NOT NULL,
        primary_status TEXT NOT NULL,
        completed_at TEXT,
        run_version INTEGER NOT NULL CHECK (run_version IN (0, 1)),
        command_sha256 TEXT,
        run_json BLOB NOT NULL,
        run_json_sha256 TEXT NOT NULL,
        UNIQUE (run_id, recording_identity),
        CHECK (
            (primary_status = 'RUNNING' AND completed_at IS NULL
                AND run_version = 0 AND command_sha256 IS NULL)
            OR
            (primary_status IN ('SUCCEEDED', 'NO_EVENTS') AND completed_at IS NOT NULL
                AND run_version = 1 AND command_sha256 IS NOT NULL)
        )
    )
    """,
    """
    CREATE TABLE event_registry_partitions (
        recording_identity TEXT PRIMARY KEY,
        generation INTEGER NOT NULL CHECK (generation >= 0),
        fence INTEGER NOT NULL CHECK (fence >= 1)
    )
    """,
    """
    CREATE TABLE stable_event_identities (
        event_id TEXT PRIMARY KEY,
        recording_identity TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        payload_json_sha256 TEXT NOT NULL,
        UNIQUE (recording_identity, event_id),
        FOREIGN KEY (recording_identity)
            REFERENCES event_registry_partitions (recording_identity)
            ON UPDATE RESTRICT ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE event_identity_assignments (
        assignment_logical_key TEXT PRIMARY KEY,
        recording_identity TEXT NOT NULL,
        event_hypothesis_logical_key TEXT NOT NULL,
        identity_policy_version TEXT NOT NULL,
        identity_policy_sha256 TEXT NOT NULL,
        event_id TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        payload_json_sha256 TEXT NOT NULL,
        UNIQUE (
            recording_identity,
            event_hypothesis_logical_key,
            identity_policy_version,
            identity_policy_sha256
        ),
        UNIQUE (recording_identity, assignment_logical_key),
        FOREIGN KEY (recording_identity, event_id)
            REFERENCES stable_event_identities (recording_identity, event_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE event_identity_relations (
        relation_logical_key TEXT PRIMARY KEY,
        recording_identity TEXT NOT NULL,
        assignment_logical_key TEXT NOT NULL,
        from_event_id TEXT NOT NULL,
        to_event_id TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        payload_json_sha256 TEXT NOT NULL,
        FOREIGN KEY (recording_identity, assignment_logical_key)
            REFERENCES event_identity_assignments (
                recording_identity, assignment_logical_key
            ) ON UPDATE RESTRICT ON DELETE RESTRICT,
        FOREIGN KEY (recording_identity, from_event_id)
            REFERENCES stable_event_identities (recording_identity, event_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        FOREIGN KEY (recording_identity, to_event_id)
            REFERENCES stable_event_identities (recording_identity, event_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        CHECK (from_event_id <> to_event_id)
    )
    """,
    """
    CREATE TABLE action_event_publications (
        subject_id TEXT PRIMARY KEY,
        recording_identity TEXT NOT NULL,
        event_id TEXT NOT NULL UNIQUE,
        revision_logical_key TEXT NOT NULL UNIQUE,
        selection_decision_logical_key TEXT NOT NULL UNIQUE,
        publication_json BLOB NOT NULL,
        publication_json_sha256 TEXT NOT NULL,
        FOREIGN KEY (recording_identity, event_id)
            REFERENCES stable_event_identities (recording_identity, event_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE detailed_results (
        artifact_id TEXT PRIMARY KEY,
        exact_bytes_sha256 TEXT NOT NULL UNIQUE,
        byte_count INTEGER NOT NULL CHECK (byte_count > 0),
        schema_id TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        schema_artifact_id TEXT NOT NULL,
        schema_sha256 TEXT NOT NULL,
        payload_json BLOB NOT NULL
    )
    """,
    """
    CREATE TABLE primary_completions (
        run_id TEXT PRIMARY KEY,
        command_sha256 TEXT NOT NULL,
        command_json BLOB NOT NULL,
        command_json_sha256 TEXT NOT NULL,
        committed_json BLOB NOT NULL,
        committed_json_sha256 TEXT NOT NULL,
        detailed_result_artifact_id TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES primary_runs (run_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        FOREIGN KEY (detailed_result_artifact_id)
            REFERENCES detailed_results (artifact_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE primary_outbox (
        outbox_id TEXT PRIMARY KEY,
        completion_run_id TEXT NOT NULL,
        recording_identity TEXT NOT NULL,
        outbox_ordinal INTEGER NOT NULL CHECK (outbox_ordinal >= 0),
        assignment_logical_key TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        payload_json_sha256 TEXT NOT NULL,
        delivered_at TEXT,
        UNIQUE (completion_run_id, outbox_ordinal),
        FOREIGN KEY (completion_run_id) REFERENCES primary_completions (run_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        FOREIGN KEY (completion_run_id, recording_identity)
            REFERENCES primary_runs (run_id, recording_identity)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        FOREIGN KEY (recording_identity, assignment_logical_key)
            REFERENCES event_identity_assignments (
                recording_identity, assignment_logical_key
            ) ON UPDATE RESTRICT ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX primary_outbox_pending_idx
        ON primary_outbox (recording_identity, delivered_at, outbox_id)
    """,
    *_OUTBOX_DELIVERY_SCHEMA_STATEMENTS,
    *_immutable_table_triggers("stable_event_identities", "stable event identities"),
    *_immutable_table_triggers("event_identity_assignments", "event identity assignments"),
    *_immutable_table_triggers("event_identity_relations", "event identity relations"),
    *_immutable_table_triggers("action_event_publications", "ActionEvent publications"),
    *_immutable_table_triggers("detailed_results", "detailed results"),
    *_immutable_table_triggers("primary_completions", "primary completions"),
)

_V1_TO_V2_SCHEMA_STATEMENTS: Final = _OUTBOX_DELIVERY_SCHEMA_STATEMENTS


def _primary_schema_fingerprint(connection: sqlite3.Connection) -> tuple[tuple[object, ...], ...]:
    rows = connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    return tuple(tuple(row) for row in rows)


@cache
def _expected_primary_schema_fingerprint() -> tuple[tuple[object, ...], ...]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        return _primary_schema_fingerprint(connection)
    finally:
        connection.close()


def _primary_schema_is_current(connection: sqlite3.Connection) -> bool:
    return _primary_schema_fingerprint(connection) == _expected_primary_schema_fingerprint()


class SQLitePrimaryCompletionRepository:
    """One local authority for run completion, event publication, and outbox."""

    def __init__(
        self,
        path: Path,
        *,
        registry: SchemaRegistry | None = None,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be pathlib.Path")
        self._path = path.resolve()
        try:
            self._registry = registry or default_schema_registry()
            self._outbox_schema_ref = self._registry.resolve_version(
                EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_ID,
                EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_VERSION,
            ).ref
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        except PrimaryCompletionError:
            raise
        except SchemaRegistryError as error:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                f"cannot resolve primary completion schema governance: {error}",
            ) from error
        except (OSError, sqlite3.Error) as error:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.TRANSACTION_FAILED,
                f"cannot initialize primary completion database: {error}",
            ) from error

    @property
    def path(self) -> Path:
        return self._path

    def begin_run(
        self,
        context: CanonicalProcessingRunContext,
    ) -> CanonicalProcessingRunRecord:
        if not isinstance(context, CanonicalProcessingRunContext):
            raise TypeError("context must be CanonicalProcessingRunContext")
        candidate = context.to_record()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM primary_runs WHERE run_id = ?",
                (candidate.run_id,),
            ).fetchone()
            if row is not None:
                stored = self._run_from_row(row)
                if _run_binding(stored) != _run_binding(candidate):
                    raise PrimaryCompletionError(
                        PrimaryCompletionErrorCode.RUN_CONFLICT,
                        "run ID already has a different immutable binding",
                    )
                connection.rollback()
                return stored

            payload = canonical_json_bytes(candidate)
            connection.execute(
                """
                INSERT INTO primary_runs (
                    run_id, recording_identity, mcap_id, pipeline_version,
                    config_sha256, started_at, primary_status, completed_at,
                    run_version, command_sha256, run_json, run_json_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', NULL, 0, NULL, ?, ?)
                """,
                (
                    candidate.run_id,
                    candidate.recording_identity,
                    candidate.mcap_id,
                    candidate.pipeline_version,
                    candidate.config_sha256,
                    candidate.started_at,
                    sqlite3.Binary(payload),
                    exact_bytes_sha256(payload),
                ),
            )
            connection.commit()
            return candidate
        except PrimaryCompletionError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.TRANSACTION_FAILED,
                f"cannot begin primary run: {error}",
            ) from error
        finally:
            connection.close()

    def snapshot(self, recording_identity: str) -> EventRegistrySnapshot:
        if not isinstance(recording_identity, str) or not recording_identity:
            raise TypeError("recording_identity must be a nonempty string")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO event_registry_partitions (
                    recording_identity, generation, fence
                ) VALUES (?, 0, 1)
                ON CONFLICT (recording_identity) DO NOTHING
                """,
                (recording_identity,),
            )
            partition = connection.execute(
                """
                SELECT generation, fence
                FROM event_registry_partitions
                WHERE recording_identity = ?
                """,
                (recording_identity,),
            ).fetchone()
            assert partition is not None
            identities = tuple(
                self._identity_from_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM stable_event_identities
                    WHERE recording_identity = ? ORDER BY event_id
                    """,
                    (recording_identity,),
                ).fetchall()
            )
            assignments = tuple(
                sorted(
                    (
                        self._assignment_from_row(row)
                        for row in connection.execute(
                            """
                            SELECT * FROM event_identity_assignments
                            WHERE recording_identity = ?
                            """,
                            (recording_identity,),
                        ).fetchall()
                    ),
                    key=lambda item: (
                        item.event_hypothesis_logical_key,
                        item.identity_policy_version,
                        item.identity_policy_sha256,
                        item.assignment_logical_key,
                    ),
                )
            )
            current_revisions = tuple(
                sorted(
                    (
                        self._publication_from_row(row).current_revision
                        for row in connection.execute(
                            """
                            SELECT * FROM action_event_publications
                            WHERE recording_identity = ?
                            """,
                            (recording_identity,),
                        ).fetchall()
                    ),
                    key=lambda item: item.event_id,
                )
            )
            connection.commit()
            return EventRegistrySnapshot(
                schema_version="1.0",
                recording_identity=recording_identity,
                generation=int(partition["generation"]),
                fence=int(partition["fence"]),
                identities=identities,
                current_revisions=current_revisions,
                assignments=assignments,
            )
        except (PrimaryCompletionError, ValidationError):
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.TRANSACTION_FAILED,
                f"cannot read event registry snapshot: {error}",
            ) from error
        finally:
            connection.close()

    def get(self, run_id: str) -> CommittedPrimaryCompletion | None:
        if not isinstance(run_id, str) or not run_id:
            raise TypeError("run_id must be a nonempty string")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM primary_completions WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return self._committed_from_row(connection, row)
        except PrimaryCompletionError:
            raise
        except sqlite3.Error as error:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.TRANSACTION_FAILED,
                f"cannot recover primary completion: {error}",
            ) from error
        finally:
            connection.close()

    def commit(self, command: PrimaryCompletionCommand) -> PrimaryCompletionCommitResult:
        checked = self._validate_command(command)
        connection = self._connect()
        commit_attempted = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM primary_completions WHERE run_id = ?",
                (checked.detail.run_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["command_sha256"]) != checked.command_sha256:
                    raise PrimaryCompletionError(
                        PrimaryCompletionErrorCode.RUN_CONFLICT,
                        "run already completed with a different command",
                    )
                committed = self._committed_from_row(connection, existing)
                connection.rollback()
                return PrimaryCompletionCommitResult(committed=committed, replayed=True)

            run_row = connection.execute(
                "SELECT * FROM primary_runs WHERE run_id = ?",
                (checked.detail.run_id,),
            ).fetchone()
            if run_row is None:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.RUN_CONFLICT,
                    "begin_run must persist the run binding before completion",
                )
            stored_run = self._run_from_row(run_row)
            if (
                stored_run.primary_status.value != "RUNNING"
                or int(run_row["run_version"]) != 0
                or _run_binding(stored_run) != _run_binding(checked.detail.processing_run)
            ):
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.STALE_RUN,
                    "processing run is no longer the expected RUNNING binding",
                )

            identity_result = self._apply_identity_and_publications(connection, checked)
            detail_bytes = canonical_json_bytes(checked.detail)
            self._insert_or_verify_detail(connection, checked, detail_bytes)
            self._after_staged_facts(connection, checked)
            outbox = identity_result.outbox if identity_result is not None else ()
            committed = CommittedPrimaryCompletion(
                schema_version="1.0",
                command_sha256=checked.command_sha256,
                processing_run=checked.detail.processing_run,
                completion=checked.completion,
                detail=checked.detail,
                identity_result=identity_result,
                action_event_publications=checked.detail.action_event_publications,
                outbox=outbox,
                evidence_references=checked.evidence_references,
            )
            command_bytes = canonical_json_bytes(checked)
            committed_bytes = canonical_json_bytes(committed)
            connection.execute(
                """
                INSERT INTO primary_completions (
                    run_id, command_sha256, command_json, command_json_sha256,
                    committed_json, committed_json_sha256,
                    detailed_result_artifact_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.detail.run_id,
                    checked.command_sha256,
                    sqlite3.Binary(command_bytes),
                    exact_bytes_sha256(command_bytes),
                    sqlite3.Binary(committed_bytes),
                    exact_bytes_sha256(committed_bytes),
                    checked.completion.detailed_result.artifact_id,
                ),
            )
            for ordinal, item in enumerate(outbox):
                self._insert_outbox(
                    connection,
                    checked.detail.run_id,
                    ordinal,
                    item,
                )

            terminal_bytes = canonical_json_bytes(checked.detail.processing_run)
            updated = connection.execute(
                """
                UPDATE primary_runs
                SET primary_status = ?, completed_at = ?, run_version = 1,
                    command_sha256 = ?, run_json = ?, run_json_sha256 = ?
                WHERE run_id = ? AND primary_status = 'RUNNING'
                    AND run_version = 0 AND command_sha256 IS NULL
                """,
                (
                    checked.detail.status,
                    checked.detail.processing_run.completed_at,
                    checked.command_sha256,
                    sqlite3.Binary(terminal_bytes),
                    exact_bytes_sha256(terminal_bytes),
                    checked.detail.run_id,
                ),
            )
            if updated.rowcount != 1:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.STALE_RUN,
                    "processing run completion compare-and-swap did not match",
                )

            commit_attempted = True
            self._commit_connection(connection)
            return PrimaryCompletionCommitResult(committed=committed, replayed=False)
        except PrimaryCompletionError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except (ValidationError, TypeError, ValueError) as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INVALID_COMMAND,
                f"primary completion command is invalid: {error}",
            ) from error
        except sqlite3.IntegrityError as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                f"primary completion constraint failed: {error}",
            ) from error
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            if commit_attempted:
                recovered = self._recover_uncertain_commit(checked)
                if recovered is not None:
                    return PrimaryCompletionCommitResult(
                        committed=recovered,
                        replayed=True,
                    )
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.TRANSACTION_FAILED,
                f"primary completion transaction failed: {error}",
            ) from error
        finally:
            connection.close()

    def list_outbox(
        self,
        recording_identity: str,
    ) -> tuple[EventIdentityOutboxRecord, ...]:
        if not isinstance(recording_identity, str) or not recording_identity:
            raise TypeError("recording_identity must be a nonempty string")
        connection = self._connect()
        try:
            return tuple(
                self._outbox_from_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM primary_outbox
                    WHERE recording_identity = ?
                    ORDER BY completion_run_id, outbox_ordinal
                    """,
                    (recording_identity,),
                ).fetchall()
            )
        except sqlite3.Error as error:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.TRANSACTION_FAILED,
                f"cannot read primary outbox: {error}",
            ) from error
        finally:
            connection.close()

    def _validate_command(self, command: object) -> PrimaryCompletionCommand:
        if not isinstance(command, PrimaryCompletionCommand):
            raise TypeError("command must be PrimaryCompletionCommand")
        try:
            validated = PrimaryCompletionCommand.model_validate(
                command.model_dump(mode="python"),
                strict=True,
            )
            self._registry.validate_pinned(
                validated.detail.schema_ref,
                validated.detail.model_dump(mode="json"),
            )
            validate_registered_primary_completion_record(
                validated.completion,
                self._registry,
            )
        except (SchemaRegistryError, ValidationError, TypeError, ValueError) as error:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INVALID_COMMAND,
                f"primary completion schema validation failed: {error}",
            ) from error
        return validated

    def _apply_identity_and_publications(
        self,
        connection: sqlite3.Connection,
        command: PrimaryCompletionCommand,
    ) -> EventIdentityBatchResult | None:
        detail = command.detail
        if detail.status == "NO_EVENTS":
            return None
        prepared = detail.prepared_identities
        assert prepared is not None
        action_batch = detail.action_event_publications
        if (
            action_batch.expected_generation != prepared.expected_generation
            or action_batch.expected_fence != prepared.expected_fence
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INVALID_COMMAND,
                "identity and ActionEvent preparations use different fences",
            )
        partition = connection.execute(
            """
            SELECT generation, fence FROM event_registry_partitions
            WHERE recording_identity = ?
            """,
            (prepared.recording_identity,),
        ).fetchone()
        if partition is None or (
            int(partition["generation"]) != prepared.expected_generation
            or int(partition["fence"]) != prepared.expected_fence
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.STALE_IDENTITY,
                "event identity generation or fence is stale",
            )

        mutation = prepared.mutation
        if mutation is not None:
            for identity in mutation.identities:
                self._insert_identity(connection, identity)
            for assignment in mutation.assignments:
                self._insert_assignment(connection, assignment)
            for relation in mutation.relations:
                self._insert_relation(connection, relation)

        for assignment in prepared.assignments:
            self._verify_assignment(connection, assignment)
        for publication in action_batch.publications:
            self._insert_or_verify_publication(connection, publication)

        if mutation is None:
            final_generation = prepared.expected_generation
            final_fence = prepared.expected_fence
            new_identities: tuple[StableEventIdentity, ...] = ()
            relations: tuple[EventIdentityRelation, ...] = ()
            outbox: tuple[EventIdentityOutboxRecord, ...] = ()
        else:
            updated = connection.execute(
                """
                UPDATE event_registry_partitions
                SET generation = ?, fence = fence + 1
                WHERE recording_identity = ? AND generation = ? AND fence = ?
                """,
                (
                    mutation.next_generation,
                    prepared.recording_identity,
                    prepared.expected_generation,
                    prepared.expected_fence,
                ),
            )
            if updated.rowcount != 1:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.STALE_IDENTITY,
                    "event identity compare-and-swap did not match",
                )
            final_generation = mutation.next_generation
            final_fence = prepared.expected_fence + 1
            new_identities = mutation.identities
            relations = mutation.relations
            outbox = mutation.outbox
        return EventIdentityBatchResult(
            recording_identity=prepared.recording_identity,
            initial_generation=prepared.expected_generation,
            final_generation=final_generation,
            fence=final_fence,
            assignments=prepared.assignments,
            new_identities=new_identities,
            relations=relations,
            outbox=outbox,
            replayed_assignment_logical_keys=(prepared.replayed_assignment_logical_keys),
        )

    def _insert_identity(
        self,
        connection: sqlite3.Connection,
        identity: StableEventIdentity,
    ) -> None:
        if (
            connection.execute(
                "SELECT 1 FROM stable_event_identities WHERE event_id = ?",
                (identity.event_id,),
            ).fetchone()
            is not None
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.STALE_IDENTITY,
                "prepared mutation reallocates an existing event ID",
            )
        payload = canonical_json_bytes(identity)
        connection.execute(
            """
            INSERT INTO stable_event_identities (
                event_id, recording_identity, payload_json, payload_json_sha256
            ) VALUES (?, ?, ?, ?)
            """,
            (
                identity.event_id,
                identity.recording_identity,
                sqlite3.Binary(payload),
                exact_bytes_sha256(payload),
            ),
        )

    def _insert_assignment(
        self,
        connection: sqlite3.Connection,
        assignment: EventIdentityAssignment,
    ) -> None:
        conflict = connection.execute(
            """
            SELECT 1 FROM event_identity_assignments
            WHERE assignment_logical_key = ? OR (
                recording_identity = ? AND event_hypothesis_logical_key = ?
                AND identity_policy_version = ? AND identity_policy_sha256 = ?
            )
            """,
            (
                assignment.assignment_logical_key,
                assignment.recording_identity,
                assignment.event_hypothesis_logical_key,
                assignment.identity_policy_version,
                assignment.identity_policy_sha256,
            ),
        ).fetchone()
        if conflict is not None:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.STALE_IDENTITY,
                "prepared mutation repeats an existing identity assignment",
            )
        payload = canonical_json_bytes(assignment)
        connection.execute(
            """
            INSERT INTO event_identity_assignments (
                assignment_logical_key, recording_identity,
                event_hypothesis_logical_key, identity_policy_version,
                identity_policy_sha256, event_id, payload_json,
                payload_json_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assignment.assignment_logical_key,
                assignment.recording_identity,
                assignment.event_hypothesis_logical_key,
                assignment.identity_policy_version,
                assignment.identity_policy_sha256,
                assignment.event_id,
                sqlite3.Binary(payload),
                exact_bytes_sha256(payload),
            ),
        )

    def _insert_relation(
        self,
        connection: sqlite3.Connection,
        relation: EventIdentityRelation,
    ) -> None:
        payload = canonical_json_bytes(relation)
        connection.execute(
            """
            INSERT INTO event_identity_relations (
                relation_logical_key, recording_identity,
                assignment_logical_key, from_event_id, to_event_id,
                payload_json, payload_json_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relation.relation_logical_key,
                relation.recording_identity,
                relation.assignment_logical_key,
                relation.from_event_id,
                relation.to_event_id,
                sqlite3.Binary(payload),
                exact_bytes_sha256(payload),
            ),
        )

    def _verify_assignment(
        self,
        connection: sqlite3.Connection,
        assignment: EventIdentityAssignment,
    ) -> None:
        row = connection.execute(
            """
            SELECT * FROM event_identity_assignments
            WHERE assignment_logical_key = ?
            """,
            (assignment.assignment_logical_key,),
        ).fetchone()
        if row is None or self._assignment_from_row(row) != assignment:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.STALE_IDENTITY,
                "prepared assignment is absent or differs from aggregate state",
            )

    def _insert_or_verify_publication(
        self,
        connection: sqlite3.Connection,
        publication: PreparedInitialActionEventRevision,
    ) -> None:
        row = connection.execute(
            """
            SELECT * FROM action_event_publications
            WHERE subject_id = ? OR event_id = ?
                OR revision_logical_key = ? OR selection_decision_logical_key = ?
            """,
            (
                publication.subject.node_logical_key,
                publication.payload.event_id,
                publication.revision.revision_logical_key,
                publication.selection.selection_decision_logical_key,
            ),
        ).fetchone()
        if row is not None:
            if self._publication_from_row(row) != publication:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.RUN_CONFLICT,
                    "ActionEvent genesis publication conflicts with existing state",
                )
            return
        payload = canonical_json_bytes(publication)
        connection.execute(
            """
            INSERT INTO action_event_publications (
                subject_id, recording_identity, event_id,
                revision_logical_key, selection_decision_logical_key,
                publication_json, publication_json_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                publication.subject.node_logical_key,
                publication.payload.recording_identity,
                publication.payload.event_id,
                publication.revision.revision_logical_key,
                publication.selection.selection_decision_logical_key,
                sqlite3.Binary(payload),
                exact_bytes_sha256(payload),
            ),
        )

    def _insert_or_verify_detail(
        self,
        connection: sqlite3.Connection,
        command: PrimaryCompletionCommand,
        payload: bytes,
    ) -> None:
        reference = command.completion.detailed_result
        payload_digest = exact_bytes_sha256(payload)
        if payload_digest != reference.exact_bytes_sha256 or len(payload) != reference.byte_count:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INVALID_COMMAND,
                "detailed result bytes do not match their exact reference",
            )
        row = connection.execute(
            """
            SELECT * FROM detailed_results
            WHERE artifact_id = ? OR exact_bytes_sha256 = ?
            """,
            (reference.artifact_id, reference.exact_bytes_sha256),
        ).fetchone()
        if row is not None:
            if (
                str(row["artifact_id"]) != reference.artifact_id
                or str(row["exact_bytes_sha256"]) != reference.exact_bytes_sha256
                or int(row["byte_count"]) != reference.byte_count
                or str(row["schema_id"]) != reference.schema_ref.schema_id
                or str(row["schema_version"]) != reference.schema_ref.version
                or str(row["schema_artifact_id"]) != reference.schema_ref.artifact_id
                or str(row["schema_sha256"]) != reference.schema_ref.sha256
                or bytes(row["payload_json"]) != payload
                or exact_bytes_sha256(bytes(row["payload_json"])) != str(row["exact_bytes_sha256"])
            ):
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "content-addressed detailed result conflicts with stored bytes",
                )
            return
        schema_ref = reference.schema_ref
        connection.execute(
            """
            INSERT INTO detailed_results (
                artifact_id, exact_bytes_sha256, byte_count, schema_id,
                schema_version, schema_artifact_id, schema_sha256, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reference.artifact_id,
                reference.exact_bytes_sha256,
                reference.byte_count,
                schema_ref.schema_id,
                schema_ref.version,
                schema_ref.artifact_id,
                schema_ref.sha256,
                sqlite3.Binary(payload),
            ),
        )

    def _insert_outbox(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        ordinal: int,
        outbox: EventIdentityOutboxRecord,
    ) -> None:
        try:
            wire = EventIdentityOutboxWireRecord.from_record(
                outbox, schema_ref=self._outbox_schema_ref
            )
            checked = validate_registered_event_identity_outbox_wire_record(wire, self._registry)
        except (SchemaRegistryError, ValidationError, TypeError, ValueError) as error:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INVALID_COMMAND,
                f"successor outbox record is not exact-schema valid: {error}",
            ) from error
        payload = canonical_json_bytes(checked)
        connection.execute(
            """
            INSERT INTO primary_outbox (
                outbox_id, completion_run_id, recording_identity,
                outbox_ordinal, assignment_logical_key, payload_json,
                payload_json_sha256, delivered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                checked.outbox_id,
                run_id,
                checked.recording_identity,
                ordinal,
                checked.assignment_logical_key,
                sqlite3.Binary(payload),
                exact_bytes_sha256(payload),
            ),
        )

    def _recover_uncertain_commit(
        self,
        command: PrimaryCompletionCommand,
    ) -> CommittedPrimaryCompletion | None:
        try:
            recovered = self.get(command.detail.run_id)
        except PrimaryCompletionError:
            return None
        if recovered is None:
            return None
        if recovered.command_sha256 != command.command_sha256:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.RUN_CONFLICT,
                "uncertain commit recovered a different command",
            )
        return recovered

    def _committed_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> CommittedPrimaryCompletion:
        committed = _decode_model(
            row,
            payload_column="committed_json",
            digest_column="committed_json_sha256",
            model=CommittedPrimaryCompletion,
        )
        command = _decode_model(
            row,
            payload_column="command_json",
            digest_column="command_json_sha256",
            model=PrimaryCompletionCommand,
        )
        if (
            committed.processing_run.run_id != str(row["run_id"])
            or committed.command_sha256 != str(row["command_sha256"])
            or command.command_sha256 != committed.command_sha256
            or command.detail != committed.detail
            or command.completion != committed.completion
            or command.evidence_references != committed.evidence_references
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored completion columns disagree with canonical JSON",
            )
        try:
            validate_registered_primary_completion_record(
                committed.completion,
                self._registry,
            )
            self._registry.validate_pinned(
                committed.detail.schema_ref,
                committed.detail.model_dump(mode="json"),
            )
            for evidence_reference in committed.evidence_references:
                self._registry.resolve_exact(evidence_reference.schema_ref)
        except (SchemaRegistryError, ValidationError, TypeError, ValueError) as error:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored completion does not carry valid exact schema governance",
            ) from error
        detail_row = connection.execute(
            "SELECT * FROM detailed_results WHERE artifact_id = ?",
            (committed.completion.detailed_result.artifact_id,),
        ).fetchone()
        detail_bytes = canonical_json_bytes(committed.detail)
        reference = committed.completion.detailed_result
        if detail_row is None:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored detailed result is absent",
            )
        stored_detail_bytes = bytes(detail_row["payload_json"])
        if (
            stored_detail_bytes != detail_bytes
            or exact_bytes_sha256(stored_detail_bytes) != reference.exact_bytes_sha256
            or str(detail_row["exact_bytes_sha256"]) != reference.exact_bytes_sha256
            or int(detail_row["byte_count"]) != reference.byte_count
            or str(detail_row["schema_id"]) != reference.schema_ref.schema_id
            or str(detail_row["schema_version"]) != reference.schema_ref.version
            or str(detail_row["schema_artifact_id"]) != reference.schema_ref.artifact_id
            or str(detail_row["schema_sha256"]) != reference.schema_ref.sha256
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored detailed result is corrupt",
            )
        run_row = connection.execute(
            "SELECT * FROM primary_runs WHERE run_id = ?",
            (committed.processing_run.run_id,),
        ).fetchone()
        if run_row is None or self._run_from_row(run_row) != committed.processing_run:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored run does not match committed completion",
            )
        outbox = tuple(
            self._outbox_from_row(item)
            for item in connection.execute(
                """
                SELECT * FROM primary_outbox
                WHERE completion_run_id = ? ORDER BY outbox_ordinal
                """,
                (committed.processing_run.run_id,),
            ).fetchall()
        )
        if outbox != committed.outbox:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored outbox does not match committed completion",
            )
        self._verify_committed_closure(connection, committed)
        return committed

    def _verify_committed_closure(
        self,
        connection: sqlite3.Connection,
        committed: CommittedPrimaryCompletion,
    ) -> None:
        for publication in committed.action_event_publications.publications:
            row = connection.execute(
                "SELECT * FROM action_event_publications WHERE subject_id = ?",
                (publication.subject.node_logical_key,),
            ).fetchone()
            if row is None or self._publication_from_row(row) != publication:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "committed ActionEvent publication is absent or corrupt",
                )

        identity = committed.identity_result
        if identity is None:
            return
        partition = connection.execute(
            """
            SELECT generation, fence FROM event_registry_partitions
            WHERE recording_identity = ?
            """,
            (identity.recording_identity,),
        ).fetchone()
        if partition is None or (
            int(partition["generation"]) < identity.final_generation
            or int(partition["fence"]) < identity.fence
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "committed identity partition has regressed",
            )
        for assignment in identity.assignments:
            row = connection.execute(
                """
                SELECT * FROM event_identity_assignments
                WHERE assignment_logical_key = ?
                """,
                (assignment.assignment_logical_key,),
            ).fetchone()
            if row is None or self._assignment_from_row(row) != assignment:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "committed identity assignment is absent or corrupt",
                )
        for stable_identity in identity.new_identities:
            row = connection.execute(
                "SELECT * FROM stable_event_identities WHERE event_id = ?",
                (stable_identity.event_id,),
            ).fetchone()
            if row is None or self._identity_from_row(row) != stable_identity:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "committed stable event identity is absent or corrupt",
                )
        for relation in identity.relations:
            row = connection.execute(
                """
                SELECT * FROM event_identity_relations
                WHERE relation_logical_key = ?
                """,
                (relation.relation_logical_key,),
            ).fetchone()
            if row is None or self._relation_from_row(row) != relation:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "committed identity relation is absent or corrupt",
                )

    def _run_from_row(self, row: sqlite3.Row) -> CanonicalProcessingRunRecord:
        run = _decode_model(
            row,
            payload_column="run_json",
            digest_column="run_json_sha256",
            model=CanonicalProcessingRunRecord,
        )
        columns = (
            str(row["run_id"]),
            str(row["recording_identity"]),
            str(row["mcap_id"]),
            str(row["pipeline_version"]),
            str(row["config_sha256"]),
            str(row["started_at"]),
            str(row["primary_status"]),
            None if row["completed_at"] is None else str(row["completed_at"]),
        )
        expected = (
            run.run_id,
            run.recording_identity,
            run.mcap_id,
            run.pipeline_version,
            run.config_sha256,
            run.started_at,
            run.primary_status.value,
            run.completed_at,
        )
        if columns != expected:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored run columns disagree with canonical JSON",
            )
        return run

    def _identity_from_row(self, row: sqlite3.Row) -> StableEventIdentity:
        identity = _decode_model(
            row,
            payload_column="payload_json",
            digest_column="payload_json_sha256",
            model=StableEventIdentity,
        )
        if identity.event_id != str(row["event_id"]) or identity.recording_identity != str(
            row["recording_identity"]
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored event identity columns disagree with canonical JSON",
            )
        return identity

    def _assignment_from_row(self, row: sqlite3.Row) -> EventIdentityAssignment:
        assignment = _decode_model(
            row,
            payload_column="payload_json",
            digest_column="payload_json_sha256",
            model=EventIdentityAssignment,
        )
        if (
            assignment.assignment_logical_key != str(row["assignment_logical_key"])
            or assignment.recording_identity != str(row["recording_identity"])
            or assignment.event_id != str(row["event_id"])
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored assignment columns disagree with canonical JSON",
            )
        return assignment

    def _relation_from_row(self, row: sqlite3.Row) -> EventIdentityRelation:
        relation = _decode_model(
            row,
            payload_column="payload_json",
            digest_column="payload_json_sha256",
            model=EventIdentityRelation,
        )
        if (
            relation.relation_logical_key != str(row["relation_logical_key"])
            or relation.recording_identity != str(row["recording_identity"])
            or relation.assignment_logical_key != str(row["assignment_logical_key"])
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored identity relation columns disagree with canonical JSON",
            )
        return relation

    def _publication_from_row(
        self,
        row: sqlite3.Row,
    ) -> PreparedInitialActionEventRevision:
        publication = _decode_model(
            row,
            payload_column="publication_json",
            digest_column="publication_json_sha256",
            model=PreparedInitialActionEventRevision,
        )
        if (
            publication.subject.node_logical_key != str(row["subject_id"])
            or publication.payload.recording_identity != str(row["recording_identity"])
            or publication.payload.event_id != str(row["event_id"])
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored ActionEvent columns disagree with canonical JSON",
            )
        return publication

    def _outbox_from_row(self, row: sqlite3.Row) -> EventIdentityOutboxRecord:
        wire = _decode_model(
            row,
            payload_column="payload_json",
            digest_column="payload_json_sha256",
            model=EventIdentityOutboxWireRecord,
        )
        try:
            wire = validate_registered_event_identity_outbox_wire_record(wire, self._registry)
        except (SchemaRegistryError, ValidationError, TypeError, ValueError) as error:
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored outbox does not carry a valid exact schema pin",
            ) from error
        if (
            wire.outbox_id != str(row["outbox_id"])
            or wire.recording_identity != str(row["recording_identity"])
            or wire.assignment_logical_key != str(row["assignment_logical_key"])
        ):
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "stored outbox columns disagree with canonical JSON",
            )
        return wire.to_record()

    def _migrate_v1_outbox_payloads(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT * FROM primary_outbox ORDER BY completion_run_id, outbox_ordinal"
        ).fetchall()
        for row in rows:
            record = _decode_model(
                row,
                payload_column="payload_json",
                digest_column="payload_json_sha256",
                model=EventIdentityOutboxRecord,
            )
            try:
                wire = EventIdentityOutboxWireRecord.from_record(
                    record,
                    schema_ref=self._outbox_schema_ref,
                )
                checked = validate_registered_event_identity_outbox_wire_record(
                    wire,
                    self._registry,
                )
            except (SchemaRegistryError, ValidationError, TypeError, ValueError) as error:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "cannot upcast v1 primary outbox to its exact Wire schema",
                ) from error
            payload = canonical_json_bytes(checked)
            cursor = connection.execute(
                """
                UPDATE primary_outbox
                SET payload_json = ?, payload_json_sha256 = ?
                WHERE outbox_id = ? AND payload_json_sha256 = ?
                """,
                (
                    sqlite3.Binary(payload),
                    exact_bytes_sha256(payload),
                    str(row["outbox_id"]),
                    str(row["payload_json_sha256"]),
                ),
            )
            if cursor.rowcount != 1:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "v1 primary outbox migration lost its target row",
                )

    def _initialize(self) -> None:
        connection = self._open_unchecked()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if user_version == _SCHEMA_VERSION and application_id == _APPLICATION_ID:
                if not _primary_schema_is_current(connection):
                    raise PrimaryCompletionError(
                        PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                        "primary completion DDL does not match the canonical schema",
                    )
                return
            if user_version == 1 and application_id == _APPLICATION_ID:
                connection.execute("BEGIN IMMEDIATE")
                locked_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if locked_version == _SCHEMA_VERSION:
                    if not _primary_schema_is_current(connection):
                        raise PrimaryCompletionError(
                            PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                            "concurrently migrated primary completion DDL is not canonical",
                        )
                    connection.commit()
                    return
                if locked_version != 1:
                    raise PrimaryCompletionError(
                        PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                        "primary completion schema changed during migration",
                    )
                self._migrate_v1_outbox_payloads(connection)
                for statement in _V1_TO_V2_SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                if not _primary_schema_is_current(connection):
                    raise PrimaryCompletionError(
                        PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                        "migrated primary completion DDL is not canonical",
                    )
                connection.commit()
                return
            if user_version != 0 or application_id != 0:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "database application_id or user_version belongs to another schema",
                )
            existing = connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
            if existing:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "refusing to initialize over an existing SQLite schema",
                )
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            if not _primary_schema_is_current(connection):
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "new primary completion DDL is not canonical",
                )
            connection.commit()
        except Exception:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        finally:
            connection.close()

    def _open_unchecked(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._path,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA trusted_schema = OFF")
            return connection
        except sqlite3.Error:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._open_unchecked()
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if application_id != _APPLICATION_ID or user_version != _SCHEMA_VERSION:
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "primary completion database header is incompatible",
                )
            if not _primary_schema_is_current(connection):
                raise PrimaryCompletionError(
                    PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                    "primary completion DDL does not match the canonical schema",
                )
            return connection
        except PrimaryCompletionError:
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as error:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise PrimaryCompletionError(
                PrimaryCompletionErrorCode.INTEGRITY_ERROR,
                "cannot open or verify the primary completion database",
            ) from error

    def _commit_connection(self, connection: sqlite3.Connection) -> None:
        """Narrow test hook for commit-outcome uncertainty."""

        connection.commit()

    def _after_staged_facts(
        self,
        connection: sqlite3.Connection,
        command: PrimaryCompletionCommand,
    ) -> None:
        """Narrow test hook for rollback after aggregate facts are staged."""

        del connection, command


def _decode_model[ModelT: BaseModel](
    row: sqlite3.Row,
    *,
    payload_column: str,
    digest_column: str,
    model: type[ModelT],
) -> ModelT:
    raw = bytes(row[payload_column])
    if exact_bytes_sha256(raw) != str(row[digest_column]):
        raise PrimaryCompletionError(
            PrimaryCompletionErrorCode.INTEGRITY_ERROR,
            f"stored {payload_column} exact digest is corrupt",
        )
    try:
        value = model.model_validate_json(raw, strict=True)
    except ValidationError as error:
        raise PrimaryCompletionError(
            PrimaryCompletionErrorCode.INTEGRITY_ERROR,
            f"stored {payload_column} is not a valid {model.__name__}",
        ) from error
    if canonical_json_bytes(value) != raw:
        raise PrimaryCompletionError(
            PrimaryCompletionErrorCode.INTEGRITY_ERROR,
            f"stored {payload_column} is not canonical JSON",
        )
    return value


def _run_binding(record: CanonicalProcessingRunRecord) -> tuple[str, ...]:
    return (
        record.run_id,
        record.recording_identity,
        record.mcap_id,
        record.pipeline_version,
        record.config_sha256,
        record.started_at,
    )


__all__ = ["SQLitePrimaryCompletionRepository"]
