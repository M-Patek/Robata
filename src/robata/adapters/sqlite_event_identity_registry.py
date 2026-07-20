"""Durable SQLite adapter for the event identity registry repository port.

The database is one registry, while compare-and-swap state is partitioned by
recording identity.  Every commit takes SQLite's writer lock before reading the
partition fence, then inserts all immutable rows and advances that fence in one
transaction.  The globally unique identity primary key prevents an allocator
from assigning the same event ID to two recordings, including across processes.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import suppress
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ValidationError

from robata.contracts.hashing import CanonicalizationError, canonical_json_bytes
from robata.event_pipeline.identity_registry import (
    CrossRecordingEventIdentityError,
    EventCurrentRevisionReference,
    EventIdentityAssignment,
    EventIdentityConflictError,
    EventIdentityInputError,
    EventIdentityOutboxRecord,
    EventIdentityRegistryError,
    EventIdentityRegistryMutation,
    EventIdentityRelation,
    EventRegistrySnapshot,
    StableEventIdentity,
    StaleEventRegistryFenceError,
)

_RECORDING_IDENTITY_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA_VERSION: Final = 1
_APPLICATION_ID: Final = 0x52454952  # "REIR": Robata event identity registry.
_BUSY_TIMEOUT_MS: Final = 30_000

_SCHEMA_SQL: Final = """
CREATE TABLE IF NOT EXISTS event_registry_partitions (
    recording_identity TEXT PRIMARY KEY,
    generation INTEGER NOT NULL CHECK (generation >= 0),
    fence INTEGER NOT NULL CHECK (fence >= 1)
);

CREATE TABLE IF NOT EXISTS stable_event_identities (
    event_id TEXT PRIMARY KEY,
    recording_identity TEXT NOT NULL,
    created_generation INTEGER NOT NULL CHECK (created_generation >= 1),
    payload_json BLOB NOT NULL,
    UNIQUE (recording_identity, event_id),
    FOREIGN KEY (recording_identity)
        REFERENCES event_registry_partitions (recording_identity)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS event_current_revisions (
    recording_identity TEXT NOT NULL,
    event_id TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    PRIMARY KEY (recording_identity, event_id),
    FOREIGN KEY (recording_identity, event_id)
        REFERENCES stable_event_identities (recording_identity, event_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS event_identity_assignments (
    assignment_logical_key TEXT PRIMARY KEY,
    recording_identity TEXT NOT NULL,
    event_hypothesis_logical_key TEXT NOT NULL,
    identity_policy_version TEXT NOT NULL,
    identity_policy_sha256 TEXT NOT NULL,
    event_id TEXT NOT NULL,
    registry_generation INTEGER NOT NULL CHECK (registry_generation >= 1),
    payload_json BLOB NOT NULL,
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
);

CREATE TABLE IF NOT EXISTS event_identity_relations (
    relation_logical_key TEXT PRIMARY KEY,
    recording_identity TEXT NOT NULL,
    assignment_logical_key TEXT NOT NULL,
    from_event_id TEXT NOT NULL,
    to_event_id TEXT NOT NULL,
    registry_generation INTEGER NOT NULL CHECK (registry_generation >= 1),
    payload_json BLOB NOT NULL,
    FOREIGN KEY (recording_identity, assignment_logical_key)
        REFERENCES event_identity_assignments (recording_identity, assignment_logical_key)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (recording_identity, from_event_id)
        REFERENCES stable_event_identities (recording_identity, event_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (recording_identity, to_event_id)
        REFERENCES stable_event_identities (recording_identity, event_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (from_event_id <> to_event_id)
);

CREATE TABLE IF NOT EXISTS event_identity_outbox (
    outbox_id TEXT PRIMARY KEY,
    recording_identity TEXT NOT NULL,
    assignment_logical_key TEXT NOT NULL,
    registry_generation INTEGER NOT NULL CHECK (registry_generation >= 1),
    payload_json BLOB NOT NULL,
    UNIQUE (recording_identity, assignment_logical_key),
    FOREIGN KEY (recording_identity, assignment_logical_key)
        REFERENCES event_identity_assignments (recording_identity, assignment_logical_key)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS stable_event_identities_recording_idx
    ON stable_event_identities (recording_identity, event_id);
CREATE INDEX IF NOT EXISTS event_identity_assignments_recording_idx
    ON event_identity_assignments (
        recording_identity,
        event_hypothesis_logical_key,
        identity_policy_version,
        identity_policy_sha256,
        assignment_logical_key
    );
CREATE INDEX IF NOT EXISTS event_identity_relations_recording_idx
    ON event_identity_relations (recording_identity, relation_logical_key);
CREATE INDEX IF NOT EXISTS event_identity_outbox_recording_idx
    ON event_identity_outbox (recording_identity, outbox_id);

CREATE TRIGGER IF NOT EXISTS stable_event_identities_no_update
BEFORE UPDATE ON stable_event_identities
BEGIN
    SELECT RAISE(ABORT, 'stable event identities are append-only');
END;
CREATE TRIGGER IF NOT EXISTS stable_event_identities_no_delete
BEFORE DELETE ON stable_event_identities
BEGIN
    SELECT RAISE(ABORT, 'stable event identities are append-only');
END;
CREATE TRIGGER IF NOT EXISTS stable_event_identities_no_reinsert
BEFORE INSERT ON stable_event_identities
WHEN EXISTS (
    SELECT 1 FROM stable_event_identities WHERE event_id = NEW.event_id
)
BEGIN
    SELECT RAISE(ABORT, 'stable event identities are append-only');
END;
CREATE TRIGGER IF NOT EXISTS event_identity_assignments_no_update
BEFORE UPDATE ON event_identity_assignments
BEGIN
    SELECT RAISE(ABORT, 'event identity assignments are append-only');
END;
CREATE TRIGGER IF NOT EXISTS event_identity_assignments_no_delete
BEFORE DELETE ON event_identity_assignments
BEGIN
    SELECT RAISE(ABORT, 'event identity assignments are append-only');
END;
CREATE TRIGGER IF NOT EXISTS event_identity_assignments_no_reinsert
BEFORE INSERT ON event_identity_assignments
WHEN EXISTS (
    SELECT 1
    FROM event_identity_assignments
    WHERE assignment_logical_key = NEW.assignment_logical_key
       OR (
           recording_identity = NEW.recording_identity
           AND event_hypothesis_logical_key = NEW.event_hypothesis_logical_key
           AND identity_policy_version = NEW.identity_policy_version
           AND identity_policy_sha256 = NEW.identity_policy_sha256
       )
)
BEGIN
    SELECT RAISE(ABORT, 'event identity assignments are append-only');
END;
CREATE TRIGGER IF NOT EXISTS event_identity_relations_no_update
BEFORE UPDATE ON event_identity_relations
BEGIN
    SELECT RAISE(ABORT, 'event identity relations are append-only');
END;
CREATE TRIGGER IF NOT EXISTS event_identity_relations_no_delete
BEFORE DELETE ON event_identity_relations
BEGIN
    SELECT RAISE(ABORT, 'event identity relations are append-only');
END;
CREATE TRIGGER IF NOT EXISTS event_identity_relations_no_reinsert
BEFORE INSERT ON event_identity_relations
WHEN EXISTS (
    SELECT 1
    FROM event_identity_relations
    WHERE relation_logical_key = NEW.relation_logical_key
)
BEGIN
    SELECT RAISE(ABORT, 'event identity relations are append-only');
END;
CREATE TRIGGER IF NOT EXISTS event_identity_outbox_no_update
BEFORE UPDATE ON event_identity_outbox
BEGIN
    SELECT RAISE(ABORT, 'event identity outbox rows are append-only');
END;
CREATE TRIGGER IF NOT EXISTS event_identity_outbox_no_delete
BEFORE DELETE ON event_identity_outbox
BEGIN
    SELECT RAISE(ABORT, 'event identity outbox rows are append-only');
END;
CREATE TRIGGER IF NOT EXISTS event_identity_outbox_no_reinsert
BEFORE INSERT ON event_identity_outbox
WHEN EXISTS (
    SELECT 1
    FROM event_identity_outbox
    WHERE outbox_id = NEW.outbox_id
       OR (
           recording_identity = NEW.recording_identity
           AND assignment_logical_key = NEW.assignment_logical_key
       )
)
BEGIN
    SELECT RAISE(ABORT, 'event identity outbox rows are append-only');
END;

PRAGMA application_id = 1380272466;
PRAGMA user_version = 1;
"""


def _split_sql_script(script: str) -> tuple[str, ...]:
    statements: list[str] = []
    pending: list[str] = []
    for line in script.splitlines(keepends=True):
        pending.append(line)
        candidate = "".join(pending)
        if sqlite3.complete_statement(candidate):
            statement = candidate.strip()
            if statement:
                statements.append(statement)
            pending.clear()
    if any(item.strip() for item in pending):
        raise AssertionError("event identity SQLite schema contains an incomplete statement")
    return tuple(statements)


_SCHEMA_STATEMENTS: Final = _split_sql_script(_SCHEMA_SQL)


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _database_schema_fingerprint(connection: sqlite3.Connection) -> tuple[object, ...]:
    object_rows = connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    objects = tuple(tuple(row) for row in object_rows)
    table_names = tuple(
        _row_text(row, "name") for row in object_rows if _row_text(row, "type") == "table"
    )
    table_facts: list[tuple[object, ...]] = []
    for table_name in table_names:
        quoted_table = _quote_sqlite_identifier(table_name)
        columns = tuple(
            tuple(row)
            for row in connection.execute(f"PRAGMA table_xinfo({quoted_table})").fetchall()
        )
        index_rows = connection.execute(f"PRAGMA index_list({quoted_table})").fetchall()
        indexes: list[tuple[object, ...]] = []
        for index_row in index_rows:
            index_name = _row_text(index_row, "name")
            quoted_index = _quote_sqlite_identifier(index_name)
            indexes.append(
                (
                    *tuple(index_row),
                    tuple(
                        tuple(row)
                        for row in connection.execute(
                            f"PRAGMA index_xinfo({quoted_index})"
                        ).fetchall()
                    ),
                )
            )
        foreign_keys = tuple(
            tuple(row)
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted_table})").fetchall()
        )
        table_facts.append((table_name, columns, tuple(indexes), foreign_keys))
    return objects, tuple(table_facts)


@cache
def _expected_schema_fingerprint() -> tuple[object, ...]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        return _database_schema_fingerprint(connection)
    finally:
        connection.close()


class SQLiteEventIdentityRegistryError(EventIdentityRegistryError):
    """SQLite storage, schema, or persisted-integrity failure."""


class SQLiteEventIdentityRegistryUncertainCommitError(SQLiteEventIdentityRegistryError):
    """A failed commit could not be reconciled to one exact durable outcome."""


class SQLiteEventIdentityRegistryRepository:
    """Local durable conformance event identity registry backed by stdlib SQLite.

    ``event_current_revisions`` is a read-only input projection here. Its separately
    fenced revision-selection owner must populate it; this adapter never authors or
    advances current-revision selection state.
    """

    def __init__(self, database_path: Path) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a pathlib.Path")
        try:
            if database_path.exists() and database_path.is_symlink():
                raise SQLiteEventIdentityRegistryError(
                    f"event identity database must not be a symlink: {database_path}"
                )
            parent = database_path.parent
            if parent.exists() and parent.is_symlink():
                raise SQLiteEventIdentityRegistryError(
                    f"event identity database parent must not be a symlink: {parent}"
                )
            parent.mkdir(parents=True, exist_ok=True)
            if not parent.is_dir():
                raise SQLiteEventIdentityRegistryError(
                    f"event identity database parent is not a directory: {parent}"
                )
            self._database_path = parent.resolve(strict=True) / database_path.name
        except SQLiteEventIdentityRegistryError:
            raise
        except OSError as exc:
            raise SQLiteEventIdentityRegistryError(
                f"cannot prepare event identity database path {database_path}: {exc}"
            ) from exc
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the resolved SQLite database path."""

        return self._database_path

    def snapshot(self, recording_identity: str) -> EventRegistrySnapshot:
        """Read a transactionally consistent recording-scoped registry snapshot."""

        checked_identity = _validate_recording_identity(recording_identity)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._verify_database(connection)
            snapshot = self._snapshot_in_transaction(connection, checked_identity)
            self._commit(connection)
            return snapshot
        except EventIdentityRegistryError:
            _rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            _rollback_quietly(connection)
            raise SQLiteEventIdentityRegistryError(
                f"cannot read event identity snapshot for {checked_identity}: {exc}"
            ) from exc
        finally:
            connection.close()

    def commit(self, mutation: EventIdentityRegistryMutation) -> EventRegistrySnapshot:
        """Atomically append a mutation and advance its exact generation and fence."""

        checked = _validate_mutation_contract(mutation)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_database(connection)
            before = self._snapshot_in_transaction(connection, checked.recording_identity)
            if before.generation != checked.expected_generation or before.fence != checked.fence:
                raise StaleEventRegistryFenceError(checked.recording_identity)

            existing_relations = self._relations_in_transaction(
                connection, checked.recording_identity
            )
            existing_outbox = self._outbox_in_transaction(connection, checked.recording_identity)
            self._validate_commit(
                connection,
                snapshot=before,
                existing_relations=existing_relations,
                existing_outbox=existing_outbox,
                mutation=checked,
            )

            connection.execute(
                """
                INSERT OR IGNORE INTO event_registry_partitions (
                    recording_identity, generation, fence
                ) VALUES (?, 0, 1)
                """,
                (checked.recording_identity,),
            )
            self._insert_identities(connection, checked.identities)
            self._insert_assignments(connection, checked.assignments)
            self._insert_relations(connection, checked.relations)
            self._insert_outbox(connection, checked.outbox)
            cursor = connection.execute(
                """
                UPDATE event_registry_partitions
                SET generation = ?, fence = fence + 1
                WHERE recording_identity = ? AND generation = ? AND fence = ?
                """,
                (
                    checked.next_generation,
                    checked.recording_identity,
                    checked.expected_generation,
                    checked.fence,
                ),
            )
            if cursor.rowcount != 1:
                raise StaleEventRegistryFenceError(checked.recording_identity)

            committed = self._snapshot_in_transaction(connection, checked.recording_identity)
            committed_relations = self._relations_in_transaction(
                connection, checked.recording_identity
            )
            committed_outbox = self._outbox_in_transaction(connection, checked.recording_identity)
            self._validate_auxiliary_integrity(
                snapshot=committed,
                relations=committed_relations,
                outbox=committed_outbox,
            )
            if not self._matches_exact_mutation_state(
                snapshot=committed,
                relations=committed_relations,
                outbox=committed_outbox,
                before=before,
                existing_relations=existing_relations,
                existing_outbox=existing_outbox,
                mutation=checked,
            ):
                raise SQLiteEventIdentityRegistryError(
                    "tentative event identity state does not match the exact mutation"
                )
            try:
                self._commit(connection)
            except sqlite3.Error as exc:
                _rollback_quietly(connection)
                try:
                    recovered = self._recover_uncertain_commit(
                        before=before,
                        existing_relations=existing_relations,
                        existing_outbox=existing_outbox,
                        mutation=checked,
                    )
                except EventIdentityRegistryError:
                    raise SQLiteEventIdentityRegistryUncertainCommitError(
                        "event identity commit outcome is uncertain and reconciliation failed"
                    ) from exc
                if recovered is None:
                    raise SQLiteEventIdentityRegistryUncertainCommitError(
                        "event identity commit outcome did not match the exact expected mutation"
                    ) from exc
                return recovered
            return committed
        except EventIdentityRegistryError:
            _rollback_quietly(connection)
            raise
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(connection)
            raise EventIdentityConflictError(
                f"SQLite rejected an append-only event identity mutation: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(connection)
            raise SQLiteEventIdentityRegistryError(
                f"event identity transaction failed for {checked.recording_identity}: {exc}"
            ) from exc
        finally:
            connection.close()

    def list_relations(self, recording_identity: str) -> tuple[EventIdentityRelation, ...]:
        """Return immutable correction edges in canonical logical-key order."""

        checked_identity = _validate_recording_identity(recording_identity)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._verify_database(connection)
            relations = self._relations_in_transaction(connection, checked_identity)
            self._commit(connection)
            return relations
        except EventIdentityRegistryError:
            _rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            _rollback_quietly(connection)
            raise SQLiteEventIdentityRegistryError(
                f"cannot read event identity relations for {checked_identity}: {exc}"
            ) from exc
        finally:
            connection.close()

    def list_outbox(self, recording_identity: str) -> tuple[EventIdentityOutboxRecord, ...]:
        """Return committed transactional outbox records in canonical ID order."""

        checked_identity = _validate_recording_identity(recording_identity)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._verify_database(connection)
            outbox = self._outbox_in_transaction(connection, checked_identity)
            self._commit(connection)
            return outbox
        except EventIdentityRegistryError:
            _rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            _rollback_quietly(connection)
            raise SQLiteEventIdentityRegistryError(
                f"cannot read event identity outbox for {checked_identity}: {exc}"
            ) from exc
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            preflight_user_version = _pragma_int(connection, "user_version")
            preflight_application_id = _pragma_int(connection, "application_id")
            preflight_schema_row = connection.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'
                ) AS has_user_schema
                """
            ).fetchone()
            if preflight_schema_row is None:
                raise SQLiteEventIdentityRegistryError("SQLite schema inventory returned no value")
            preflight_has_schema = _row_int(preflight_schema_row, "has_user_schema") == 1
            if preflight_user_version == 0:
                if preflight_application_id != 0 or preflight_has_schema:
                    raise SQLiteEventIdentityRegistryError(
                        "refusing to adopt a nonempty or claimed unversioned SQLite database"
                    )
            elif preflight_user_version != _SCHEMA_VERSION:
                raise SQLiteEventIdentityRegistryError(
                    f"unsupported event identity database schema version: {preflight_user_version}"
                )
            else:
                self._verify_schema(connection)

            journal_row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            journal_mode: object = None if journal_row is None else journal_row[0]
            if not isinstance(journal_mode, str) or journal_mode.lower() != "wal":
                raise SQLiteEventIdentityRegistryError("SQLite WAL mode could not be enabled")

            connection.execute("BEGIN IMMEDIATE")
            user_version = _pragma_int(connection, "user_version")
            application_id = _pragma_int(connection, "application_id")
            schema_row = connection.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'
                ) AS has_user_schema
                """
            ).fetchone()
            if schema_row is None:
                raise SQLiteEventIdentityRegistryError("SQLite schema inventory returned no value")
            has_user_schema = _row_int(schema_row, "has_user_schema") == 1

            if user_version == 0:
                if application_id != 0 or has_user_schema:
                    raise SQLiteEventIdentityRegistryError(
                        "refusing to adopt a nonempty or claimed unversioned SQLite database"
                    )
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
            elif user_version != _SCHEMA_VERSION:
                raise SQLiteEventIdentityRegistryError(
                    f"unsupported event identity database schema version: {user_version}"
                )
            elif application_id != _APPLICATION_ID:
                raise SQLiteEventIdentityRegistryError(
                    "event identity database has an unexpected SQLite application identity"
                )

            self._verify_database(connection)
            connection.commit()
        except EventIdentityRegistryError:
            if connection is not None:
                _rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                _rollback_quietly(connection)
            raise SQLiteEventIdentityRegistryError(
                f"cannot initialize SQLite event identity registry: {exc}"
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        if self._database_path.is_symlink():
            raise SQLiteEventIdentityRegistryError(
                f"event identity database became a symlink: {self._database_path}"
            )
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA recursive_triggers = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA trusted_schema = OFF")
            if _pragma_int(connection, "foreign_keys") != 1:
                raise sqlite3.OperationalError("foreign-key enforcement is disabled")
            if _pragma_int(connection, "recursive_triggers") != 1:
                raise sqlite3.OperationalError("recursive trigger enforcement is disabled")
            if _pragma_int(connection, "synchronous") != 2:
                raise sqlite3.OperationalError("SQLite FULL synchronous mode is disabled")
            if _pragma_int(connection, "busy_timeout") != _BUSY_TIMEOUT_MS:
                raise sqlite3.OperationalError("SQLite busy timeout is inconsistent")
            if _pragma_int(connection, "trusted_schema") != 0:
                raise sqlite3.OperationalError("SQLite trusted schema mode is enabled")
            return connection
        except sqlite3.Error as exc:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise SQLiteEventIdentityRegistryError(
                f"cannot open SQLite event identity registry: {exc}"
            ) from exc

    def _verify_database(self, connection: sqlite3.Connection) -> None:
        self._verify_schema(connection)
        journal_row = connection.execute("PRAGMA journal_mode").fetchone()
        journal_mode: object = None if journal_row is None else journal_row[0]
        if not isinstance(journal_mode, str) or journal_mode.lower() != "wal":
            raise SQLiteEventIdentityRegistryError(
                "SQLite event identity database is not in WAL mode"
            )
        self._verify_database_health(connection)

    def _verify_schema(self, connection: sqlite3.Connection) -> None:
        if _pragma_int(connection, "application_id") != _APPLICATION_ID:
            raise SQLiteEventIdentityRegistryError(
                "event identity database has an unexpected SQLite application identity"
            )
        if _pragma_int(connection, "user_version") != _SCHEMA_VERSION:
            raise SQLiteEventIdentityRegistryError(
                "event identity database schema version changed unexpectedly"
            )
        if _database_schema_fingerprint(connection) != _expected_schema_fingerprint():
            raise SQLiteEventIdentityRegistryError(
                "SQLite event identity DDL does not match the canonical schema"
            )

    def _verify_database_health(self, connection: sqlite3.Connection) -> None:
        check_row = connection.execute("PRAGMA quick_check(1)").fetchone()
        check_result: object = None if check_row is None else check_row[0]
        if check_result != "ok":
            raise SQLiteEventIdentityRegistryError(
                f"SQLite quick_check failed for event identity database: {check_result!r}"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise SQLiteEventIdentityRegistryError(
                "SQLite foreign-key check found orphaned event identity rows"
            )
        self._verify_all_persisted_state(connection)

    def _verify_all_persisted_state(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT recording_identity
            FROM event_registry_partitions
            ORDER BY recording_identity
            """
        ).fetchall()
        for row in rows:
            recording_identity = _row_text(row, "recording_identity")
            if _RECORDING_IDENTITY_PATTERN.fullmatch(recording_identity) is None:
                raise SQLiteEventIdentityRegistryError(
                    "persisted event identity partition has an invalid recording identity"
                )
            snapshot = self._snapshot_in_transaction(connection, recording_identity)
            relations = self._relations_in_transaction(connection, recording_identity)
            outbox = self._outbox_in_transaction(connection, recording_identity)
            self._validate_auxiliary_integrity(
                snapshot=snapshot,
                relations=relations,
                outbox=outbox,
            )

    @staticmethod
    def _validate_snapshot_integrity(
        snapshot: EventRegistrySnapshot,
        *,
        partition_exists: bool,
    ) -> None:
        if snapshot.fence != snapshot.generation + 1:
            raise SQLiteEventIdentityRegistryError(
                "event identity partition fence is inconsistent with its generation"
            )
        if partition_exists and snapshot.generation == 0:
            raise SQLiteEventIdentityRegistryError(
                "persisted event identity partition cannot remain at generation zero"
            )

        identity_by_id = {item.event_id: item for item in snapshot.identities}
        revision_ids = tuple(item.event_id for item in snapshot.current_revisions)
        if len(set(revision_ids)) != len(revision_ids) or not set(revision_ids).issubset(
            identity_by_id
        ):
            raise SQLiteEventIdentityRegistryError(
                "current-revision projection references an unknown event identity"
            )

        assignment_logical_keys = tuple(
            item.assignment_logical_key for item in snapshot.assignments
        )
        assignment_idempotency_keys = tuple(
            _assignment_idempotency_key(item) for item in snapshot.assignments
        )
        if len(set(assignment_logical_keys)) != len(assignment_logical_keys) or len(
            set(assignment_idempotency_keys)
        ) != len(assignment_idempotency_keys):
            raise SQLiteEventIdentityRegistryError(
                "persisted event identity assignments violate uniqueness"
            )

        assignment_generations: set[int] = set()
        creation_assignments: dict[tuple[str, int], list[EventIdentityAssignment]] = {}
        for assignment in snapshot.assignments:
            identity = identity_by_id.get(assignment.event_id)
            if identity is None:
                raise SQLiteEventIdentityRegistryError(
                    "persisted assignment references an unknown event identity"
                )
            if not (
                identity.created_generation <= assignment.registry_generation <= snapshot.generation
            ):
                raise SQLiteEventIdentityRegistryError(
                    "persisted assignment generation is outside its identity partition"
                )

            for candidate in assignment.candidates:
                candidate_identity = identity_by_id.get(candidate.event_id)
                if (
                    candidate_identity is None
                    or candidate_identity.created_generation > assignment.registry_generation
                ):
                    raise SQLiteEventIdentityRegistryError(
                        "persisted assignment candidate is outside its identity partition"
                    )

            disposition = assignment.disposition.value
            expected_relation_values = {
                candidate.relation.value for candidate in assignment.candidates
            }
            expected_relation_values.add(
                "SAME_EVENT" if disposition == "REUSED" else "NEW_IDENTITY"
            )
            if {item.value for item in assignment.relation} != expected_relation_values:
                raise SQLiteEventIdentityRegistryError(
                    "persisted assignment relation summary does not match its candidates"
                )

            same_event_candidates = tuple(
                candidate
                for candidate in assignment.candidates
                if candidate.relation.value == "SAME_EVENT"
            )
            if disposition == "REUSED":
                if (
                    len(same_event_candidates) != 1
                    or same_event_candidates[0].event_id != assignment.event_id
                ):
                    raise SQLiteEventIdentityRegistryError(
                        "persisted reused assignment lacks its exact selected identity"
                    )
            elif disposition == "AMBIGUOUS":
                if not assignment.candidates or any(
                    candidate.relation.value != "POSSIBLE_MATCH"
                    or candidate.event_id == assignment.event_id
                    for candidate in assignment.candidates
                ):
                    raise SQLiteEventIdentityRegistryError(
                        "persisted ambiguous assignment has invalid candidates"
                    )
            elif any(
                candidate.relation.value in {"SAME_EVENT", "POSSIBLE_MATCH"}
                or candidate.event_id == assignment.event_id
                for candidate in assignment.candidates
            ):
                raise SQLiteEventIdentityRegistryError(
                    "persisted created assignment has invalid candidates"
                )

            assignment_generations.add(assignment.registry_generation)
            if disposition in {"CREATED", "AMBIGUOUS"}:
                if (
                    identity.created_generation != assignment.registry_generation
                    or identity.created_by_hypothesis_logical_key
                    != assignment.event_hypothesis_logical_key
                    or identity.creation_disposition != disposition
                ):
                    raise SQLiteEventIdentityRegistryError(
                        "created or ambiguous assignment does not create its exact identity"
                    )
                creation_assignments.setdefault(
                    (assignment.event_id, assignment.registry_generation), []
                ).append(assignment)

        if snapshot.generation == 0:
            if snapshot.identities or snapshot.current_revisions or snapshot.assignments:
                raise SQLiteEventIdentityRegistryError(
                    "generation-zero event identity snapshot contains durable rows"
                )
        else:
            ordered_generations = tuple(sorted(assignment_generations))
            if (
                not ordered_generations
                or ordered_generations[0] != 1
                or ordered_generations[-1] != snapshot.generation
                or any(
                    current != previous + 1 for previous, current in pairwise(ordered_generations)
                )
            ):
                raise SQLiteEventIdentityRegistryError(
                    "persisted assignment generations do not form a complete partition history"
                )

        for identity in snapshot.identities:
            if identity.created_generation > snapshot.generation:
                raise SQLiteEventIdentityRegistryError(
                    "persisted event identity was created after the partition generation"
                )
            creators = creation_assignments.get(
                (identity.event_id, identity.created_generation), []
            )
            if len(creators) != 1:
                raise SQLiteEventIdentityRegistryError(
                    "persisted event identity lacks one exact creation assignment"
                )
            creator = creators[0]
            if (
                creator.event_hypothesis_logical_key != identity.created_by_hypothesis_logical_key
                or creator.disposition.value != identity.creation_disposition
            ):
                raise SQLiteEventIdentityRegistryError(
                    "persisted event identity creation lineage is inconsistent"
                )

    @staticmethod
    def _validate_auxiliary_integrity(
        *,
        snapshot: EventRegistrySnapshot,
        relations: tuple[EventIdentityRelation, ...],
        outbox: tuple[EventIdentityOutboxRecord, ...],
    ) -> None:
        assignment_by_key = {item.assignment_logical_key: item for item in snapshot.assignments}
        identity_ids = {item.event_id for item in snapshot.identities}
        expected_relation_facts = {
            (
                snapshot.recording_identity,
                assignment.assignment_logical_key,
                candidate.event_id,
                assignment.event_id,
                candidate.relation.value,
                candidate.score,
                candidate.reason,
                assignment.identity_policy_version,
                assignment.identity_policy_sha256,
                assignment.registry_generation,
            )
            for assignment in snapshot.assignments
            for candidate in assignment.candidates
            if candidate.relation.value != "SAME_EVENT"
        }
        actual_relation_facts = {
            (
                relation.recording_identity,
                relation.assignment_logical_key,
                relation.from_event_id,
                relation.to_event_id,
                relation.relation.value,
                relation.score,
                relation.reason,
                relation.identity_policy_version,
                relation.identity_policy_sha256,
                relation.registry_generation,
            )
            for relation in relations
        }
        for relation in relations:
            assignment = assignment_by_key.get(relation.assignment_logical_key)
            if assignment is None:
                raise SQLiteEventIdentityRegistryError(
                    "persisted event identity relation references an unknown assignment"
                )
            if (
                relation.from_event_id not in identity_ids
                or relation.to_event_id not in identity_ids
            ):
                raise SQLiteEventIdentityRegistryError(
                    "persisted event identity relation lineage is inconsistent"
                )
        if len(actual_relation_facts) != len(relations) or (
            actual_relation_facts != expected_relation_facts
        ):
            raise SQLiteEventIdentityRegistryError(
                "persisted relations do not exactly match assignment candidates"
            )

        outbox_by_assignment = {item.assignment_logical_key: item for item in outbox}
        if len(outbox_by_assignment) != len(outbox) or set(outbox_by_assignment) != set(
            assignment_by_key
        ):
            raise SQLiteEventIdentityRegistryError(
                "persisted transactional outbox does not exactly cover assignments"
            )
        if any(
            item.registry_generation
            != assignment_by_key[item.assignment_logical_key].registry_generation
            for item in outbox
        ):
            raise SQLiteEventIdentityRegistryError(
                "persisted transactional outbox generation is inconsistent"
            )

    def _snapshot_in_transaction(
        self,
        connection: sqlite3.Connection,
        recording_identity: str,
    ) -> EventRegistrySnapshot:
        partition = connection.execute(
            """
            SELECT generation, fence
            FROM event_registry_partitions
            WHERE recording_identity = ?
            """,
            (recording_identity,),
        ).fetchone()
        if partition is None:
            generation = 0
            fence = 1
        else:
            generation = _row_int(partition, "generation")
            fence = _row_int(partition, "fence")

        identity_rows = connection.execute(
            """
            SELECT event_id, recording_identity, created_generation, payload_json
            FROM stable_event_identities
            WHERE recording_identity = ?
            ORDER BY event_id
            """,
            (recording_identity,),
        ).fetchall()
        identities = tuple(self._identity_from_row(row) for row in identity_rows)

        revision_rows = connection.execute(
            """
            SELECT event_id, recording_identity, payload_json
            FROM event_current_revisions
            WHERE recording_identity = ?
            ORDER BY event_id
            """,
            (recording_identity,),
        ).fetchall()
        revisions = tuple(self._revision_from_row(row) for row in revision_rows)

        assignment_rows = connection.execute(
            """
            SELECT
                assignment_logical_key,
                recording_identity,
                event_hypothesis_logical_key,
                identity_policy_version,
                identity_policy_sha256,
                event_id,
                registry_generation,
                payload_json
            FROM event_identity_assignments
            WHERE recording_identity = ?
            ORDER BY
                event_hypothesis_logical_key,
                identity_policy_version,
                identity_policy_sha256,
                assignment_logical_key
            """,
            (recording_identity,),
        ).fetchall()
        assignments = tuple(self._assignment_from_row(row) for row in assignment_rows)
        try:
            snapshot = EventRegistrySnapshot(
                schema_version="1.0",
                recording_identity=recording_identity,
                generation=generation,
                fence=fence,
                identities=identities,
                current_revisions=revisions,
                assignments=assignments,
            )
        except (TypeError, ValueError) as exc:
            raise SQLiteEventIdentityRegistryError(
                f"persisted event identity snapshot is invalid for {recording_identity}"
            ) from exc
        self._validate_snapshot_integrity(
            snapshot,
            partition_exists=partition is not None,
        )
        return snapshot

    def _relations_in_transaction(
        self,
        connection: sqlite3.Connection,
        recording_identity: str,
    ) -> tuple[EventIdentityRelation, ...]:
        rows = connection.execute(
            """
            SELECT
                relation_logical_key,
                recording_identity,
                assignment_logical_key,
                from_event_id,
                to_event_id,
                registry_generation,
                payload_json
            FROM event_identity_relations
            WHERE recording_identity = ?
            ORDER BY relation_logical_key
            """,
            (recording_identity,),
        ).fetchall()
        return tuple(self._relation_from_row(row) for row in rows)

    def _outbox_in_transaction(
        self,
        connection: sqlite3.Connection,
        recording_identity: str,
    ) -> tuple[EventIdentityOutboxRecord, ...]:
        rows = connection.execute(
            """
            SELECT
                outbox_id,
                recording_identity,
                assignment_logical_key,
                registry_generation,
                payload_json
            FROM event_identity_outbox
            WHERE recording_identity = ?
            ORDER BY outbox_id
            """,
            (recording_identity,),
        ).fetchall()
        return tuple(self._outbox_from_row(row) for row in rows)

    def _validate_commit(
        self,
        connection: sqlite3.Connection,
        *,
        snapshot: EventRegistrySnapshot,
        existing_relations: tuple[EventIdentityRelation, ...],
        existing_outbox: tuple[EventIdentityOutboxRecord, ...],
        mutation: EventIdentityRegistryMutation,
    ) -> None:
        new_identity_ids = tuple(item.event_id for item in mutation.identities)
        if len(set(new_identity_ids)) != len(new_identity_ids):
            raise EventIdentityConflictError("mutation repeats a new event ID")

        existing_identity_ids = {item.event_id for item in snapshot.identities}
        if existing_identity_ids.intersection(new_identity_ids):
            raise EventIdentityConflictError("mutation reallocates an existing event ID")
        for event_id, owner in self._event_owners(connection, new_identity_ids):
            if owner != mutation.recording_identity:
                raise CrossRecordingEventIdentityError(
                    "event ID is already owned by another recording"
                )
            raise EventIdentityConflictError(f"mutation reallocates existing event ID {event_id}")

        known_event_ids = existing_identity_ids | set(new_identity_ids)
        existing_assignment_keys = {
            _assignment_idempotency_key(item) for item in snapshot.assignments
        }
        existing_assignment_logical_keys = {
            item.assignment_logical_key for item in snapshot.assignments
        }
        mutation_assignment_keys: set[tuple[str, str, str]] = set()
        mutation_assignment_logical_keys: set[str] = set()
        for assignment in mutation.assignments:
            key = _assignment_idempotency_key(assignment)
            if key in existing_assignment_keys or key in mutation_assignment_keys:
                raise EventIdentityConflictError(
                    "hypothesis already has an assignment for this exact policy"
                )
            if (
                assignment.assignment_logical_key in existing_assignment_logical_keys
                or assignment.assignment_logical_key in mutation_assignment_logical_keys
            ):
                raise EventIdentityConflictError("assignment logical key conflicts")
            if assignment.event_id not in known_event_ids:
                raise EventIdentityConflictError(
                    "assignment references an unknown recording-scoped event ID"
                )
            mutation_assignment_keys.add(key)
            mutation_assignment_logical_keys.add(assignment.assignment_logical_key)

        relation_keys = tuple(item.relation_logical_key for item in mutation.relations)
        existing_relation_keys = {item.relation_logical_key for item in existing_relations}
        if len(set(relation_keys)) != len(relation_keys) or existing_relation_keys.intersection(
            relation_keys
        ):
            raise EventIdentityConflictError("relation logical key conflicts")
        for relation in mutation.relations:
            if relation.assignment_logical_key not in mutation_assignment_logical_keys:
                raise EventIdentityConflictError(
                    "relation does not belong to this mutation's assignment"
                )
            if (
                relation.from_event_id not in known_event_ids
                or relation.to_event_id not in known_event_ids
                or relation.from_event_id == relation.to_event_id
            ):
                raise EventIdentityConflictError(
                    "relation endpoints must be distinct identities in this recording"
                )

        outbox_ids = tuple(item.outbox_id for item in mutation.outbox)
        existing_outbox_ids = {item.outbox_id for item in existing_outbox}
        if len(set(outbox_ids)) != len(outbox_ids) or existing_outbox_ids.intersection(outbox_ids):
            raise EventIdentityConflictError("outbox ID conflicts")
        if {
            item.assignment_logical_key for item in mutation.outbox
        } != mutation_assignment_logical_keys:
            raise EventIdentityConflictError(
                "transactional outbox does not exactly cover assignments"
            )

    def _event_owners(
        self,
        connection: sqlite3.Connection,
        event_ids: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        if not event_ids:
            return ()
        placeholders = ", ".join("?" for _ in event_ids)
        rows = connection.execute(
            f"""
            SELECT event_id, recording_identity
            FROM stable_event_identities
            WHERE event_id IN ({placeholders})
            ORDER BY event_id
            """,
            event_ids,
        ).fetchall()
        return tuple(
            (_row_text(row, "event_id"), _row_text(row, "recording_identity")) for row in rows
        )

    def _insert_identities(
        self,
        connection: sqlite3.Connection,
        identities: tuple[StableEventIdentity, ...],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO stable_event_identities (
                event_id, recording_identity, created_generation, payload_json
            ) VALUES (?, ?, ?, ?)
            """,
            tuple(
                (
                    item.event_id,
                    item.recording_identity,
                    item.created_generation,
                    sqlite3.Binary(canonical_json_bytes(item)),
                )
                for item in identities
            ),
        )

    def _insert_assignments(
        self,
        connection: sqlite3.Connection,
        assignments: tuple[EventIdentityAssignment, ...],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO event_identity_assignments (
                assignment_logical_key,
                recording_identity,
                event_hypothesis_logical_key,
                identity_policy_version,
                identity_policy_sha256,
                event_id,
                registry_generation,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(
                (
                    item.assignment_logical_key,
                    item.recording_identity,
                    item.event_hypothesis_logical_key,
                    item.identity_policy_version,
                    item.identity_policy_sha256,
                    item.event_id,
                    item.registry_generation,
                    sqlite3.Binary(canonical_json_bytes(item)),
                )
                for item in assignments
            ),
        )

    def _insert_relations(
        self,
        connection: sqlite3.Connection,
        relations: tuple[EventIdentityRelation, ...],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO event_identity_relations (
                relation_logical_key,
                recording_identity,
                assignment_logical_key,
                from_event_id,
                to_event_id,
                registry_generation,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(
                (
                    item.relation_logical_key,
                    item.recording_identity,
                    item.assignment_logical_key,
                    item.from_event_id,
                    item.to_event_id,
                    item.registry_generation,
                    sqlite3.Binary(canonical_json_bytes(item)),
                )
                for item in relations
            ),
        )

    def _insert_outbox(
        self,
        connection: sqlite3.Connection,
        outbox: tuple[EventIdentityOutboxRecord, ...],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO event_identity_outbox (
                outbox_id,
                recording_identity,
                assignment_logical_key,
                registry_generation,
                payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            tuple(
                (
                    item.outbox_id,
                    item.recording_identity,
                    item.assignment_logical_key,
                    item.registry_generation,
                    sqlite3.Binary(canonical_json_bytes(item)),
                )
                for item in outbox
            ),
        )

    def _identity_from_row(self, row: sqlite3.Row) -> StableEventIdentity:
        identity = _model_from_blob(
            StableEventIdentity,
            _row_bytes(row, "payload_json"),
            "stable event identity",
        )
        if (
            identity.event_id != _row_text(row, "event_id")
            or identity.recording_identity != _row_text(row, "recording_identity")
            or identity.created_generation != _row_int(row, "created_generation")
        ):
            raise SQLiteEventIdentityRegistryError(
                "stable event identity payload does not match indexed columns"
            )
        return identity

    def _revision_from_row(self, row: sqlite3.Row) -> EventCurrentRevisionReference:
        revision = _model_from_blob(
            EventCurrentRevisionReference,
            _row_bytes(row, "payload_json"),
            "event current revision",
        )
        if revision.event_id != _row_text(
            row, "event_id"
        ) or revision.recording_identity != _row_text(row, "recording_identity"):
            raise SQLiteEventIdentityRegistryError(
                "event current revision payload does not match indexed columns"
            )
        return revision

    def _assignment_from_row(self, row: sqlite3.Row) -> EventIdentityAssignment:
        assignment = _model_from_blob(
            EventIdentityAssignment,
            _row_bytes(row, "payload_json"),
            "event identity assignment",
        )
        expected_columns: tuple[tuple[str, object], ...] = (
            ("assignment_logical_key", assignment.assignment_logical_key),
            ("recording_identity", assignment.recording_identity),
            ("event_hypothesis_logical_key", assignment.event_hypothesis_logical_key),
            ("identity_policy_version", assignment.identity_policy_version),
            ("identity_policy_sha256", assignment.identity_policy_sha256),
            ("event_id", assignment.event_id),
        )
        if any(_row_text(row, column) != value for column, value in expected_columns) or (
            _row_int(row, "registry_generation") != assignment.registry_generation
        ):
            raise SQLiteEventIdentityRegistryError(
                "event identity assignment payload does not match indexed columns"
            )
        return assignment

    def _relation_from_row(self, row: sqlite3.Row) -> EventIdentityRelation:
        relation = _model_from_blob(
            EventIdentityRelation,
            _row_bytes(row, "payload_json"),
            "event identity relation",
        )
        expected_columns: tuple[tuple[str, object], ...] = (
            ("relation_logical_key", relation.relation_logical_key),
            ("recording_identity", relation.recording_identity),
            ("assignment_logical_key", relation.assignment_logical_key),
            ("from_event_id", relation.from_event_id),
            ("to_event_id", relation.to_event_id),
        )
        if any(_row_text(row, column) != value for column, value in expected_columns) or (
            _row_int(row, "registry_generation") != relation.registry_generation
        ):
            raise SQLiteEventIdentityRegistryError(
                "event identity relation payload does not match indexed columns"
            )
        return relation

    def _outbox_from_row(self, row: sqlite3.Row) -> EventIdentityOutboxRecord:
        outbox = _model_from_blob(
            EventIdentityOutboxRecord,
            _row_bytes(row, "payload_json"),
            "event identity outbox",
        )
        expected_columns: tuple[tuple[str, object], ...] = (
            ("outbox_id", outbox.outbox_id),
            ("recording_identity", outbox.recording_identity),
            ("assignment_logical_key", outbox.assignment_logical_key),
        )
        if any(_row_text(row, column) != value for column, value in expected_columns) or (
            _row_int(row, "registry_generation") != outbox.registry_generation
        ):
            raise SQLiteEventIdentityRegistryError(
                "event identity outbox payload does not match indexed columns"
            )
        return outbox

    def _recover_uncertain_commit(
        self,
        *,
        before: EventRegistrySnapshot,
        existing_relations: tuple[EventIdentityRelation, ...],
        existing_outbox: tuple[EventIdentityOutboxRecord, ...],
        mutation: EventIdentityRegistryMutation,
    ) -> EventRegistrySnapshot | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._verify_database(connection)
            snapshot = self._snapshot_in_transaction(
                connection,
                mutation.recording_identity,
            )
            relations = self._relations_in_transaction(
                connection,
                mutation.recording_identity,
            )
            outbox = self._outbox_in_transaction(
                connection,
                mutation.recording_identity,
            )
            connection.commit()
            if not self._matches_exact_mutation_state(
                snapshot=snapshot,
                relations=relations,
                outbox=outbox,
                before=before,
                existing_relations=existing_relations,
                existing_outbox=existing_outbox,
                mutation=mutation,
            ):
                return None
            return snapshot
        except EventIdentityRegistryError:
            _rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            _rollback_quietly(connection)
            raise SQLiteEventIdentityRegistryError(
                "cannot reconcile uncertain event identity commit"
            ) from exc
        finally:
            connection.close()

    @staticmethod
    def _matches_exact_mutation_state(
        *,
        snapshot: EventRegistrySnapshot,
        relations: tuple[EventIdentityRelation, ...],
        outbox: tuple[EventIdentityOutboxRecord, ...],
        before: EventRegistrySnapshot,
        existing_relations: tuple[EventIdentityRelation, ...],
        existing_outbox: tuple[EventIdentityOutboxRecord, ...],
        mutation: EventIdentityRegistryMutation,
    ) -> bool:
        expected_identities = tuple(
            sorted((*before.identities, *mutation.identities), key=lambda item: item.event_id)
        )
        expected_assignments = tuple(
            sorted(
                (*before.assignments, *mutation.assignments),
                key=_assignment_storage_sort_key,
            )
        )
        expected_relations = tuple(
            sorted(
                (*existing_relations, *mutation.relations),
                key=lambda item: item.relation_logical_key,
            )
        )
        expected_outbox = tuple(
            sorted((*existing_outbox, *mutation.outbox), key=lambda item: item.outbox_id)
        )
        return (
            snapshot.recording_identity == mutation.recording_identity
            and snapshot.generation == mutation.next_generation
            and snapshot.fence == mutation.fence + 1
            and snapshot.identities == expected_identities
            and snapshot.assignments == expected_assignments
            and relations == expected_relations
            and outbox == expected_outbox
        )

    def _commit(self, connection: sqlite3.Connection) -> None:
        """Narrow commit hook retained for deterministic fault-injection tests."""

        connection.commit()


def _validate_recording_identity(recording_identity: str) -> str:
    if (
        not isinstance(recording_identity, str)
        or _RECORDING_IDENTITY_PATTERN.fullmatch(recording_identity) is None
    ):
        raise EventIdentityInputError("recording_identity must be a lowercase SHA-256 digest")
    return recording_identity


def _validate_mutation_contract(
    mutation: EventIdentityRegistryMutation,
) -> EventIdentityRegistryMutation:
    if not isinstance(mutation, EventIdentityRegistryMutation):
        raise TypeError("mutation must be an EventIdentityRegistryMutation")
    try:
        return EventIdentityRegistryMutation.model_validate(
            mutation.model_dump(mode="python"), strict=True
        )
    except ValueError as exc:
        raise EventIdentityInputError("registry mutation failed validation") from exc


def _assignment_idempotency_key(
    assignment: EventIdentityAssignment,
) -> tuple[str, str, str]:
    return (
        assignment.event_hypothesis_logical_key,
        assignment.identity_policy_version,
        assignment.identity_policy_sha256,
    )


def _assignment_storage_sort_key(
    assignment: EventIdentityAssignment,
) -> tuple[str, str, str, str]:
    return (*_assignment_idempotency_key(assignment), assignment.assignment_logical_key)


def _model_from_blob[ModelT: BaseModel](
    model_type: type[ModelT],
    raw: bytes,
    description: str,
) -> ModelT:
    try:
        value = model_type.model_validate_json(raw, strict=True)
        if canonical_json_bytes(value) != raw:
            raise SQLiteEventIdentityRegistryError(f"persisted {description} is not canonical JSON")
        return value
    except SQLiteEventIdentityRegistryError:
        raise
    except (CanonicalizationError, ValidationError, TypeError, ValueError) as exc:
        raise SQLiteEventIdentityRegistryError(
            f"persisted {description} failed strict validation"
        ) from exc


def _row_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise SQLiteEventIdentityRegistryError(f"SQLite column {column!r} is not text")
    return value


def _row_int(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SQLiteEventIdentityRegistryError(f"SQLite column {column!r} is not an integer")
    return value


def _row_bytes(row: sqlite3.Row, column: str) -> bytes:
    value: object = row[column]
    if not isinstance(value, bytes):
        raise SQLiteEventIdentityRegistryError(f"SQLite column {column!r} is not a blob")
    return value


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None:
        raise SQLiteEventIdentityRegistryError(f"SQLite PRAGMA {name} returned no value")
    value: object = row[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SQLiteEventIdentityRegistryError(f"SQLite PRAGMA {name} did not return an integer")
    return value


def _rollback_quietly(connection: sqlite3.Connection) -> None:
    with suppress(sqlite3.Error):
        connection.rollback()


__all__ = [
    "SQLiteEventIdentityRegistryError",
    "SQLiteEventIdentityRegistryRepository",
    "SQLiteEventIdentityRegistryUncertainCommitError",
]
