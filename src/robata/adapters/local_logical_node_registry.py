"""SQLite-backed run-independent logical-node registry."""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Final, cast

from pydantic import TypeAdapter, ValidationError

from robata.contracts.common import SchemaVersion
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.logical_nodes import (
    KeyNamespace,
    LogicalNode,
    NodeLogicalKey,
    NodeType,
    OpaqueUuid,
    ProcessingRunNodeMembership,
    Rfc3339Timestamp,
    RunNodeDisposition,
    RunNodeRole,
)
from robata.contracts.revisions import (
    CurrentSelection,
    ImmutableNodeRevision,
    RevisionEligibility,
    SelectionDecision,
    create_selection_decision,
)
from robata.ports.logical_node_registry import (
    ExistingNodeDisposition,
    LogicalNodeRegistryError,
    LogicalNodeRegistryErrorCode,
    PublishedRunNodeMembership,
    VerifiedLogicalNode,
)
from robata.ports.revision_registry import (
    PublishedRevision,
    PublishedSelection,
    RevisionSelectionRegistryError,
    RevisionSelectionRegistryErrorCode,
    VerifiedRevisionSubject,
)

_NODE_TYPE_ADAPTER: Final[TypeAdapter[NodeType]] = TypeAdapter(NodeType)
_NODE_KEY_ADAPTER: Final[TypeAdapter[NodeLogicalKey]] = TypeAdapter(NodeLogicalKey)
_UUID_ADAPTER: Final[TypeAdapter[OpaqueUuid]] = TypeAdapter(OpaqueUuid)
_ROLE_ADAPTER: Final[TypeAdapter[RunNodeRole]] = TypeAdapter(RunNodeRole)
_TIMESTAMP_ADAPTER: Final[TypeAdapter[Rfc3339Timestamp]] = TypeAdapter(Rfc3339Timestamp)
_KEY_NAMESPACE_ADAPTER: Final[TypeAdapter[KeyNamespace]] = TypeAdapter(KeyNamespace)
_SCHEMA_VERSION_ADAPTER: Final[TypeAdapter[SchemaVersion]] = TypeAdapter(SchemaVersion)
_CURRENT_PROJECTION_VERSION: Final[SchemaVersion] = "current-selection-v1"


def _normalize_schema_sql(value: str) -> str:
    return " ".join(value.split())


_SCHEMA_V1_SQL = """
CREATE TABLE IF NOT EXISTS logical_nodes (
    node_type TEXT NOT NULL,
    node_logical_key TEXT NOT NULL,
    schema_version TEXT NOT NULL CHECK (schema_version = '1.0'),
    key_namespace TEXT NOT NULL,
    semantic_sha256 TEXT NOT NULL,
    identity_policy_version TEXT NOT NULL,
    node_json BLOB NOT NULL,
    node_json_sha256 TEXT NOT NULL,
    PRIMARY KEY (node_type, node_logical_key)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS processing_run_nodes (
    run_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    node_logical_key TEXT NOT NULL,
    role TEXT NOT NULL,
    schema_version TEXT NOT NULL CHECK (schema_version = '1.0'),
    disposition TEXT NOT NULL CHECK (
        disposition IN ('CREATED', 'REUSED', 'INVALIDATED', 'OBSERVED')
    ),
    first_work_item_id TEXT NOT NULL,
    attached_at TEXT NOT NULL,
    membership_json BLOB NOT NULL,
    membership_json_sha256 TEXT NOT NULL,
    PRIMARY KEY (run_id, node_type, node_logical_key, role),
    FOREIGN KEY (node_type, node_logical_key)
        REFERENCES logical_nodes (node_type, node_logical_key)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS processing_run_nodes_node_idx
    ON processing_run_nodes (node_type, node_logical_key, run_id, role);

CREATE UNIQUE INDEX IF NOT EXISTS processing_run_nodes_creator_idx
    ON processing_run_nodes (node_type, node_logical_key)
    WHERE disposition = 'CREATED';

PRAGMA user_version = 1;
"""

_REVISION_SCHEMA_STATEMENTS: Final = (
    """
    CREATE TABLE immutable_node_revisions (
        subject_type TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        revision_id TEXT NOT NULL,
        schema_version TEXT NOT NULL CHECK (schema_version = '1.0'),
        revision_key_namespace TEXT NOT NULL,
        revision_logical_key TEXT NOT NULL,
        semantic_sha256 TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        lineage_sha256 TEXT NOT NULL,
        status_at_publication TEXT NOT NULL,
        eligibility_at_publication TEXT NOT NULL CHECK (
            eligibility_at_publication IN ('ELIGIBLE', 'INELIGIBLE')
        ),
        revision_policy_version TEXT NOT NULL,
        supersedes_revision_id TEXT,
        supersedes_revision_logical_key TEXT,
        published_at TEXT NOT NULL,
        revision_json BLOB NOT NULL,
        revision_json_sha256 TEXT NOT NULL,
        PRIMARY KEY (subject_type, subject_id, revision_id),
        FOREIGN KEY (subject_type, subject_id)
            REFERENCES logical_nodes (node_type, node_logical_key)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        FOREIGN KEY (
            subject_type,
            subject_id,
            supersedes_revision_id,
            supersedes_revision_logical_key
        )
            REFERENCES immutable_node_revisions (
                subject_type,
                subject_id,
                revision_id,
                revision_logical_key
            )
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        CHECK (
            (supersedes_revision_id IS NULL AND supersedes_revision_logical_key IS NULL)
            OR
            (
                supersedes_revision_id IS NOT NULL
                AND supersedes_revision_logical_key IS NOT NULL
            )
        )
    ) WITHOUT ROWID
    """,
    """
    CREATE UNIQUE INDEX immutable_node_revisions_id_idx
        ON immutable_node_revisions (revision_id)
    """,
    """
    CREATE UNIQUE INDEX immutable_node_revisions_logical_key_idx
        ON immutable_node_revisions (subject_type, subject_id, revision_logical_key)
    """,
    """
    CREATE UNIQUE INDEX immutable_node_revisions_owner_identity_idx
        ON immutable_node_revisions (
            subject_type, subject_id, revision_id, revision_logical_key
        )
    """,
    """
    CREATE TABLE selection_decisions (
        subject_type TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        selection_decision_id TEXT NOT NULL,
        schema_version TEXT NOT NULL CHECK (schema_version = '1.0'),
        selection_key_namespace TEXT NOT NULL,
        selection_decision_logical_key TEXT NOT NULL,
        semantic_sha256 TEXT NOT NULL,
        selected_revision_id TEXT NOT NULL,
        selected_revision_logical_key TEXT NOT NULL,
        previous_selection_decision_id TEXT,
        previous_selection_decision_logical_key TEXT,
        selection_sequence INTEGER NOT NULL CHECK (selection_sequence >= 1),
        selection_policy_version TEXT NOT NULL,
        projection_version TEXT NOT NULL,
        selected_at TEXT NOT NULL,
        decision_json BLOB NOT NULL,
        decision_json_sha256 TEXT NOT NULL,
        PRIMARY KEY (subject_type, subject_id, selection_decision_id),
        FOREIGN KEY (
            subject_type, subject_id, selected_revision_id, selected_revision_logical_key
        )
            REFERENCES immutable_node_revisions (
                subject_type, subject_id, revision_id, revision_logical_key
            )
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        FOREIGN KEY (
            subject_type,
            subject_id,
            previous_selection_decision_id,
            previous_selection_decision_logical_key
        )
            REFERENCES selection_decisions (
                subject_type,
                subject_id,
                selection_decision_id,
                selection_decision_logical_key
            )
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        CHECK (
            (
                previous_selection_decision_id IS NULL
                AND previous_selection_decision_logical_key IS NULL
            )
            OR
            (
                previous_selection_decision_id IS NOT NULL
                AND previous_selection_decision_logical_key IS NOT NULL
            )
        )
    ) WITHOUT ROWID
    """,
    """
    CREATE UNIQUE INDEX selection_decisions_id_idx
        ON selection_decisions (selection_decision_id)
    """,
    """
    CREATE UNIQUE INDEX selection_decisions_logical_key_idx
        ON selection_decisions (
            subject_type, subject_id, selection_decision_logical_key
        )
    """,
    """
    CREATE UNIQUE INDEX selection_decisions_sequence_idx
        ON selection_decisions (subject_type, subject_id, selection_sequence)
    """,
    """
    CREATE UNIQUE INDEX selection_decisions_owner_identity_idx
        ON selection_decisions (
            subject_type,
            subject_id,
            selection_decision_id,
            selection_decision_logical_key
        )
    """,
    """
    CREATE UNIQUE INDEX selection_decisions_projection_source_idx
        ON selection_decisions (
            subject_type,
            subject_id,
            selection_decision_id,
            selected_revision_id,
            selection_policy_version,
            projection_version,
            selected_at
        )
    """,
    """
    CREATE UNIQUE INDEX selection_decisions_genesis_idx
        ON selection_decisions (subject_type, subject_id)
        WHERE previous_selection_decision_id IS NULL
    """,
    """
    CREATE UNIQUE INDEX selection_decisions_successor_idx
        ON selection_decisions (
            subject_type, subject_id, previous_selection_decision_id
        )
        WHERE previous_selection_decision_id IS NOT NULL
    """,
    """
    CREATE TABLE current_selections (
        subject_type TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        schema_version TEXT NOT NULL CHECK (schema_version = '1.0'),
        selected_revision_id TEXT NOT NULL,
        selection_decision_id TEXT NOT NULL,
        selection_policy_version TEXT NOT NULL,
        projection_version TEXT NOT NULL,
        selected_at TEXT NOT NULL,
        current_json BLOB NOT NULL,
        current_json_sha256 TEXT NOT NULL,
        PRIMARY KEY (subject_type, subject_id),
        FOREIGN KEY (subject_type, subject_id, selected_revision_id)
            REFERENCES immutable_node_revisions (subject_type, subject_id, revision_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        FOREIGN KEY (
            subject_type,
            subject_id,
            selection_decision_id,
            selected_revision_id,
            selection_policy_version,
            projection_version,
            selected_at
        )
            REFERENCES selection_decisions (
                subject_type,
                subject_id,
                selection_decision_id,
                selected_revision_id,
                selection_policy_version,
                projection_version,
                selected_at
            )
            ON UPDATE RESTRICT ON DELETE RESTRICT
    ) WITHOUT ROWID
    """,
    """
    CREATE TRIGGER immutable_node_revisions_no_update
    BEFORE UPDATE ON immutable_node_revisions
    BEGIN
        SELECT RAISE(ABORT, 'immutable node revisions cannot be updated');
    END
    """,
    """
    CREATE TRIGGER immutable_node_revisions_no_delete
    BEFORE DELETE ON immutable_node_revisions
    BEGIN
        SELECT RAISE(ABORT, 'immutable node revisions cannot be deleted');
    END
    """,
    """
    CREATE TRIGGER selection_decisions_no_update
    BEFORE UPDATE ON selection_decisions
    BEGIN
        SELECT RAISE(ABORT, 'selection decisions cannot be updated');
    END
    """,
    """
    CREATE TRIGGER selection_decisions_no_delete
    BEFORE DELETE ON selection_decisions
    BEGIN
        SELECT RAISE(ABORT, 'selection decisions cannot be deleted');
    END
    """,
)

_EXPECTED_TABLE_COLUMNS: Final = {
    "logical_nodes": (
        "node_type",
        "node_logical_key",
        "schema_version",
        "key_namespace",
        "semantic_sha256",
        "identity_policy_version",
        "node_json",
        "node_json_sha256",
    ),
    "processing_run_nodes": (
        "run_id",
        "node_type",
        "node_logical_key",
        "role",
        "schema_version",
        "disposition",
        "first_work_item_id",
        "attached_at",
        "membership_json",
        "membership_json_sha256",
    ),
    "immutable_node_revisions": (
        "subject_type",
        "subject_id",
        "revision_id",
        "schema_version",
        "revision_key_namespace",
        "revision_logical_key",
        "semantic_sha256",
        "payload_sha256",
        "lineage_sha256",
        "status_at_publication",
        "eligibility_at_publication",
        "revision_policy_version",
        "supersedes_revision_id",
        "supersedes_revision_logical_key",
        "published_at",
        "revision_json",
        "revision_json_sha256",
    ),
    "selection_decisions": (
        "subject_type",
        "subject_id",
        "selection_decision_id",
        "schema_version",
        "selection_key_namespace",
        "selection_decision_logical_key",
        "semantic_sha256",
        "selected_revision_id",
        "selected_revision_logical_key",
        "previous_selection_decision_id",
        "previous_selection_decision_logical_key",
        "selection_sequence",
        "selection_policy_version",
        "projection_version",
        "selected_at",
        "decision_json",
        "decision_json_sha256",
    ),
    "current_selections": (
        "subject_type",
        "subject_id",
        "schema_version",
        "selected_revision_id",
        "selection_decision_id",
        "selection_policy_version",
        "projection_version",
        "selected_at",
        "current_json",
        "current_json_sha256",
    ),
}
_EXPECTED_TABLE_TYPES: Final = {
    "logical_nodes": ("TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "BLOB", "TEXT"),
    "processing_run_nodes": (
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "BLOB",
        "TEXT",
    ),
    "immutable_node_revisions": (
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "BLOB",
        "TEXT",
    ),
    "selection_decisions": (
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "INTEGER",
        "TEXT",
        "TEXT",
        "TEXT",
        "BLOB",
        "TEXT",
    ),
    "current_selections": (
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "BLOB",
        "TEXT",
    ),
}
_EXPECTED_PRIMARY_KEYS: Final = {
    "logical_nodes": ("node_type", "node_logical_key"),
    "processing_run_nodes": ("run_id", "node_type", "node_logical_key", "role"),
    "immutable_node_revisions": ("subject_type", "subject_id", "revision_id"),
    "selection_decisions": ("subject_type", "subject_id", "selection_decision_id"),
    "current_selections": ("subject_type", "subject_id"),
}
_EXPECTED_NULLABLE_COLUMNS: Final = {
    "logical_nodes": frozenset(),
    "processing_run_nodes": frozenset(),
    "immutable_node_revisions": frozenset(
        {"supersedes_revision_id", "supersedes_revision_logical_key"}
    ),
    "selection_decisions": frozenset(
        {
            "previous_selection_decision_id",
            "previous_selection_decision_logical_key",
        }
    ),
    "current_selections": frozenset(),
}
_EXPECTED_INDEX_COLUMNS: Final = {
    "processing_run_nodes_creator_idx": ("node_type", "node_logical_key"),
    "processing_run_nodes_node_idx": (
        "node_type",
        "node_logical_key",
        "run_id",
        "role",
    ),
    "immutable_node_revisions_id_idx": ("revision_id",),
    "immutable_node_revisions_logical_key_idx": (
        "subject_type",
        "subject_id",
        "revision_logical_key",
    ),
    "immutable_node_revisions_owner_identity_idx": (
        "subject_type",
        "subject_id",
        "revision_id",
        "revision_logical_key",
    ),
    "selection_decisions_id_idx": ("selection_decision_id",),
    "selection_decisions_logical_key_idx": (
        "subject_type",
        "subject_id",
        "selection_decision_logical_key",
    ),
    "selection_decisions_sequence_idx": (
        "subject_type",
        "subject_id",
        "selection_sequence",
    ),
    "selection_decisions_owner_identity_idx": (
        "subject_type",
        "subject_id",
        "selection_decision_id",
        "selection_decision_logical_key",
    ),
    "selection_decisions_projection_source_idx": (
        "subject_type",
        "subject_id",
        "selection_decision_id",
        "selected_revision_id",
        "selection_policy_version",
        "projection_version",
        "selected_at",
    ),
    "selection_decisions_genesis_idx": ("subject_type", "subject_id"),
    "selection_decisions_successor_idx": (
        "subject_type",
        "subject_id",
        "previous_selection_decision_id",
    ),
}
_EXPECTED_INDEX_FLAGS: Final = {
    "processing_run_nodes_creator_idx": (1, 1),
    "processing_run_nodes_node_idx": (0, 0),
    "immutable_node_revisions_id_idx": (1, 0),
    "immutable_node_revisions_logical_key_idx": (1, 0),
    "immutable_node_revisions_owner_identity_idx": (1, 0),
    "selection_decisions_id_idx": (1, 0),
    "selection_decisions_logical_key_idx": (1, 0),
    "selection_decisions_sequence_idx": (1, 0),
    "selection_decisions_owner_identity_idx": (1, 0),
    "selection_decisions_projection_source_idx": (1, 0),
    "selection_decisions_genesis_idx": (1, 1),
    "selection_decisions_successor_idx": (1, 1),
}
_EXPECTED_SCHEMA_SQL_V1: Final = {
    "logical_nodes": _normalize_schema_sql(
        """
        CREATE TABLE logical_nodes (
            node_type TEXT NOT NULL,
            node_logical_key TEXT NOT NULL,
            schema_version TEXT NOT NULL CHECK (schema_version = '1.0'),
            key_namespace TEXT NOT NULL,
            semantic_sha256 TEXT NOT NULL,
            identity_policy_version TEXT NOT NULL,
            node_json BLOB NOT NULL,
            node_json_sha256 TEXT NOT NULL,
            PRIMARY KEY (node_type, node_logical_key)
        ) WITHOUT ROWID
        """
    ),
    "processing_run_nodes": _normalize_schema_sql(
        """
        CREATE TABLE processing_run_nodes (
            run_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            node_logical_key TEXT NOT NULL,
            role TEXT NOT NULL,
            schema_version TEXT NOT NULL CHECK (schema_version = '1.0'),
            disposition TEXT NOT NULL CHECK (
                disposition IN ('CREATED', 'REUSED', 'INVALIDATED', 'OBSERVED')
            ),
            first_work_item_id TEXT NOT NULL,
            attached_at TEXT NOT NULL,
            membership_json BLOB NOT NULL,
            membership_json_sha256 TEXT NOT NULL,
            PRIMARY KEY (run_id, node_type, node_logical_key, role),
            FOREIGN KEY (node_type, node_logical_key)
                REFERENCES logical_nodes (node_type, node_logical_key)
                ON UPDATE RESTRICT ON DELETE RESTRICT
        ) WITHOUT ROWID
        """
    ),
    "processing_run_nodes_creator_idx": _normalize_schema_sql(
        """
        CREATE UNIQUE INDEX processing_run_nodes_creator_idx
            ON processing_run_nodes (node_type, node_logical_key)
            WHERE disposition = 'CREATED'
        """
    ),
    "processing_run_nodes_node_idx": _normalize_schema_sql(
        """
        CREATE INDEX processing_run_nodes_node_idx
            ON processing_run_nodes (node_type, node_logical_key, run_id, role)
        """
    ),
}

_REVISION_SCHEMA_NAMES: Final = (
    "immutable_node_revisions",
    "immutable_node_revisions_id_idx",
    "immutable_node_revisions_logical_key_idx",
    "immutable_node_revisions_owner_identity_idx",
    "selection_decisions",
    "selection_decisions_id_idx",
    "selection_decisions_logical_key_idx",
    "selection_decisions_sequence_idx",
    "selection_decisions_owner_identity_idx",
    "selection_decisions_projection_source_idx",
    "selection_decisions_genesis_idx",
    "selection_decisions_successor_idx",
    "current_selections",
    "immutable_node_revisions_no_update",
    "immutable_node_revisions_no_delete",
    "selection_decisions_no_update",
    "selection_decisions_no_delete",
)
_EXPECTED_REVISION_SCHEMA_SQL: Final = {
    name: _normalize_schema_sql(statement)
    for name, statement in zip(
        _REVISION_SCHEMA_NAMES,
        _REVISION_SCHEMA_STATEMENTS,
        strict=True,
    )
}
_EXPECTED_SCHEMA_SQL_V2: Final = _EXPECTED_SCHEMA_SQL_V1 | _EXPECTED_REVISION_SCHEMA_SQL


class LocalLogicalNodeRegistry:
    """Local durable registry whose transaction attaches node and run together."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                "registry root must be a pathlib.Path",
            )
        try:
            if root.is_symlink():
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                    f"registry root must not be a symlink: {root}",
                )
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                    f"registry root is not a directory: {root}",
                )
            self._root = root.resolve(strict=True)
            self._database_path = self._root / "logical-nodes.sqlite3"
            if self._database_path.is_symlink():
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                    f"registry database must not be a symlink: {self._database_path}",
                )
        except LogicalNodeRegistryError:
            raise
        except OSError as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.STORAGE_IO_ERROR,
                f"cannot initialize logical-node storage at {root}: {error}",
            ) from error
        self._initialize_database()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def database_path(self) -> Path:
        return self._database_path

    def _initialize_database(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            user_version = _pragma_int(connection, "user_version")
            if user_version not in (0, 1, 2):
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                    f"unsupported local logical-node schema version: {user_version}",
                )
            journal_row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            journal_mode: object = None if journal_row is None else journal_row[0]
            if not isinstance(journal_mode, str) or journal_mode.lower() != "wal":
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.STORAGE_IO_ERROR,
                    "SQLite WAL mode could not be enabled",
                )
            if user_version == 0:
                connection.executescript(_SCHEMA_V1_SQL)
                user_version = 1
            if user_version == 1:
                self._verify_database_schema(connection, expected_version=1)
                self._verify_database_health(connection)
                self._migrate_v1_to_v2(connection)
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
        except LogicalNodeRegistryError:
            raise
        except sqlite3.Error as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.STORAGE_IO_ERROR,
                f"cannot initialize SQLite logical-node registry: {error}",
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        if self._database_path.is_symlink():
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"registry database became a symlink: {self._database_path}",
            )
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=30.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA trusted_schema = OFF")
            if _pragma_int(connection, "foreign_keys") != 1:
                raise sqlite3.OperationalError("foreign-key enforcement is disabled")
            return connection
        except sqlite3.Error as error:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.STORAGE_IO_ERROR,
                f"cannot open SQLite logical-node registry: {error}",
            ) from error

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        try:
            connection.execute("BEGIN IMMEDIATE")
            current_version = _pragma_int(connection, "user_version")
            if current_version == 2:
                connection.rollback()
                return
            if current_version != 1:
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                    f"cannot migrate logical-node schema version {current_version}",
                )
            self._verify_database_schema(connection, expected_version=1)
            self._verify_database_health(connection)
            for statement in _REVISION_SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
        except LogicalNodeRegistryError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.STORAGE_IO_ERROR,
                f"cannot migrate SQLite logical-node registry to schema version 2: {error}",
            ) from error

    def _verify_database_schema(
        self,
        connection: sqlite3.Connection,
        *,
        expected_version: int = 2,
    ) -> None:
        if _pragma_int(connection, "foreign_keys") != 1:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.STORAGE_IO_ERROR,
                "SQLite foreign-key enforcement could not be enabled",
            )
        if expected_version not in (1, 2):
            raise AssertionError("expected_version must be 1 or 2")
        if _pragma_int(connection, "user_version") != expected_version:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"logical-node registry schema version is not {expected_version}",
            )
        schema_rows = connection.execute(
            """
            SELECT name, sql
            FROM sqlite_schema
            WHERE sql IS NOT NULL
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        schema_sql = {
            _row_text(row, "name"): _normalize_schema_sql(_row_text(row, "sql"))
            for row in schema_rows
        }
        expected_schema_sql = (
            _EXPECTED_SCHEMA_SQL_V1 if expected_version == 1 else _EXPECTED_SCHEMA_SQL_V2
        )
        if schema_sql != expected_schema_sql:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "SQLite logical-node DDL does not match the canonical schema",
            )
        table_names = (
            ("logical_nodes", "processing_run_nodes")
            if expected_version == 1
            else tuple(_EXPECTED_TABLE_COLUMNS)
        )
        for table in table_names:
            expected = _EXPECTED_TABLE_COLUMNS[table]
            rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
            actual = tuple(_row_text(row, "name") for row in rows)
            expected_notnull = tuple(
                0 if name in _EXPECTED_NULLABLE_COLUMNS[table] else 1 for name in expected
            )
            actual_notnull = tuple(_row_int(row, "notnull") for row in rows)
            if actual != expected or actual_notnull != expected_notnull:
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                    f"SQLite table {table!r} does not match the required schema",
                )
            actual_types = tuple(_row_text(row, "type") for row in rows)
            if actual_types != _EXPECTED_TABLE_TYPES[table]:
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                    f"SQLite table {table!r} has invalid declared column types",
                )
            primary_key = tuple(
                name
                for _, name in sorted(
                    (_row_int(row, "pk"), _row_text(row, "name"))
                    for row in rows
                    if _row_int(row, "pk") > 0
                )
            )
            if primary_key != _EXPECTED_PRIMARY_KEYS[table]:
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                    f"SQLite table {table!r} has an invalid primary key",
                )
        indexes: dict[str, sqlite3.Row] = {}
        for table in table_names:
            for row in connection.execute(f"PRAGMA index_list({table})").fetchall():
                if _row_text(row, "origin") == "c":
                    indexes[_row_text(row, "name")] = row
        expected_index_names = {
            name
            for name, sql in expected_schema_sql.items()
            if sql.startswith("CREATE INDEX ") or sql.startswith("CREATE UNIQUE INDEX ")
        }
        if set(indexes) != expected_index_names:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "SQLite registry indexes do not match the canonical schema",
            )
        for index_name in sorted(expected_index_names):
            expected_columns = _EXPECTED_INDEX_COLUMNS[index_name]
            index_row = indexes.get(index_name)
            expected_unique, expected_partial = _EXPECTED_INDEX_FLAGS[index_name]
            if (
                index_row is None
                or _row_int(index_row, "unique") != expected_unique
                or _row_int(index_row, "partial") != expected_partial
                or _row_text(index_row, "origin") != "c"
            ):
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                    f"SQLite index {index_name!r} has invalid flags",
                )
            columns = tuple(
                _row_text(row, "name")
                for row in connection.execute(f"PRAGMA index_info({index_name})").fetchall()
            )
            if columns != expected_columns:
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                    f"SQLite index {index_name!r} has invalid columns",
                )
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(processing_run_nodes)"
        ).fetchall()
        actual_foreign_key = tuple(
            (
                _row_text(row, "table"),
                _row_text(row, "from"),
                _row_text(row, "to"),
                _row_text(row, "on_update"),
                _row_text(row, "on_delete"),
            )
            for row in sorted(foreign_keys, key=lambda row: _row_int(row, "seq"))
        )
        expected_foreign_key = (
            ("logical_nodes", "node_type", "node_type", "RESTRICT", "RESTRICT"),
            (
                "logical_nodes",
                "node_logical_key",
                "node_logical_key",
                "RESTRICT",
                "RESTRICT",
            ),
        )
        if actual_foreign_key != expected_foreign_key:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "SQLite processing-run membership foreign key is invalid",
            )

    def attach_run_node(
        self,
        *,
        node: LogicalNode,
        run_id: OpaqueUuid,
        role: RunNodeRole,
        first_work_item_id: OpaqueUuid,
        attached_at: Rfc3339Timestamp,
        existing_node_disposition: ExistingNodeDisposition = RunNodeDisposition.REUSED,
    ) -> PublishedRunNodeMembership:
        """Atomically attach a run while deriving whether the node was created or reused."""

        checked_node, requested = self._validate_attach_request(
            node=node,
            run_id=run_id,
            role=role,
            first_work_item_id=first_work_item_id,
            attached_at=attached_at,
            existing_node_disposition=existing_node_disposition,
        )
        node_bytes = canonical_json_bytes(checked_node)
        expected_membership = requested
        node_inserted = False
        membership_inserted = False
        commit_error: sqlite3.Error | None = None
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            node_row = self._select_node_row(
                connection,
                checked_node.node_type,
                checked_node.node_logical_key,
            )
            membership_row = self._select_membership_row(connection, requested.identity)

            if membership_row is not None:
                if node_row is None:
                    raise LogicalNodeRegistryError(
                        LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                        "stored membership references an absent logical node",
                    )
                verified = self._verified_node_from_connection(
                    connection,
                    checked_node.node_type,
                    checked_node.node_logical_key,
                )
                if canonical_json_bytes(verified.node) != node_bytes:
                    raise LogicalNodeRegistryError(
                        LogicalNodeRegistryErrorCode.NODE_CONFLICT,
                        "logical-node identity is already bound to different immutable content",
                    )
                stored_membership = self._membership_from_row(membership_row)
                if not self._is_idempotent_retry(stored_membership, requested):
                    raise LogicalNodeRegistryError(
                        LogicalNodeRegistryErrorCode.MEMBERSHIP_CONFLICT,
                        "run-node membership identity is already bound to different content",
                    )
                connection.rollback()
                return PublishedRunNodeMembership(
                    node=verified.node,
                    membership=stored_membership,
                    node_inserted=False,
                    membership_inserted=False,
                )

            if node_row is None:
                if requested.disposition is not RunNodeDisposition.REUSED:
                    raise LogicalNodeRegistryError(
                        LogicalNodeRegistryErrorCode.NODE_NOT_FOUND,
                        f"{requested.disposition.value} requires an existing verified node",
                    )
                expected_membership = requested.model_copy(
                    update={"disposition": RunNodeDisposition.CREATED}
                )
                self._insert_node(connection, checked_node, node_bytes)
                node_inserted = True
            else:
                verified = self._verified_node_from_connection(
                    connection,
                    checked_node.node_type,
                    checked_node.node_logical_key,
                )
                if canonical_json_bytes(verified.node) != node_bytes:
                    raise LogicalNodeRegistryError(
                        LogicalNodeRegistryErrorCode.NODE_CONFLICT,
                        "logical-node identity is already bound to different immutable content",
                    )

            membership_bytes = canonical_json_bytes(expected_membership)
            self._insert_membership(connection, expected_membership, membership_bytes)
            membership_inserted = True
            try:
                self._commit(connection)
            except sqlite3.Error as error:
                commit_error = error
                with suppress(sqlite3.Error):
                    connection.rollback()
        except LogicalNodeRegistryError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.TRANSACTION_FAILED,
                f"logical-node attach transaction failed: {error}",
            ) from error
        finally:
            connection.close()

        if commit_error is not None:
            recovered = self._recover_uncertain_commit(checked_node, expected_membership)
            if recovered is None:
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.TRANSACTION_FAILED,
                    f"logical-node attach commit failed: {commit_error}",
                ) from commit_error
            return PublishedRunNodeMembership(
                node=recovered.node,
                membership=expected_membership,
                node_inserted=None,
                membership_inserted=None,
            )

        verified = self.verify_node(checked_node.node_type, checked_node.node_logical_key)
        membership = next(
            (
                candidate
                for candidate in verified.memberships
                if candidate.identity == expected_membership.identity
            ),
            None,
        )
        if membership != expected_membership:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "committed membership does not match the attach result",
            )
        return PublishedRunNodeMembership(
            node=verified.node,
            membership=membership,
            node_inserted=node_inserted,
            membership_inserted=membership_inserted,
        )

    def lookup_node(
        self,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
    ) -> LogicalNode | None:
        checked_type, checked_key = self._validate_node_identity(node_type, node_logical_key)
        connection = self._connect()
        try:
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            if self._select_node_row(connection, checked_type, checked_key) is None:
                return None
            return self._verified_node_from_connection(
                connection,
                checked_type,
                checked_key,
            ).node
        except LogicalNodeRegistryError:
            raise
        except sqlite3.Error as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot look up logical node: {error}",
            ) from error
        finally:
            connection.close()

    def lookup_membership(
        self,
        run_id: OpaqueUuid,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
        role: RunNodeRole,
    ) -> ProcessingRunNodeMembership | None:
        identity = self._validate_membership_identity(
            run_id,
            node_type,
            node_logical_key,
            role,
        )
        connection = self._connect()
        try:
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            row = self._select_membership_row(connection, identity)
            if row is None:
                return None
            membership = self._membership_from_row(row)
            verified = self._verified_node_from_connection(connection, identity[1], identity[2])
            if membership not in verified.memberships:
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                    "membership is absent from its node's verified membership set",
                )
            return membership
        except LogicalNodeRegistryError:
            raise
        except sqlite3.Error as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot look up run-node membership: {error}",
            ) from error
        finally:
            connection.close()

    def list_run_memberships(
        self,
        run_id: OpaqueUuid,
    ) -> tuple[ProcessingRunNodeMembership, ...]:
        checked_run = self._validate_uuid(run_id, "run_id")
        connection = self._connect()
        try:
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            rows = connection.execute(
                """
                SELECT *
                FROM processing_run_nodes
                WHERE run_id = ?
                ORDER BY node_type, node_logical_key, role
                """,
                (checked_run,),
            ).fetchall()
            memberships = tuple(self._membership_from_row(row) for row in rows)
            for membership in memberships:
                verified = self._verified_node_from_connection(
                    connection,
                    membership.node_type,
                    membership.node_logical_key,
                )
                if membership not in verified.memberships:
                    raise LogicalNodeRegistryError(
                        LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                        "run membership is absent from its verified node",
                    )
            return memberships
        except LogicalNodeRegistryError:
            raise
        except sqlite3.Error as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot list run-node memberships: {error}",
            ) from error
        finally:
            connection.close()

    def list_node_memberships(
        self,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
    ) -> tuple[ProcessingRunNodeMembership, ...]:
        checked_type, checked_key = self._validate_node_identity(node_type, node_logical_key)
        connection = self._connect()
        try:
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            if self._select_node_row(connection, checked_type, checked_key) is None:
                return ()
            return self._verified_node_from_connection(
                connection,
                checked_type,
                checked_key,
            ).memberships
        except LogicalNodeRegistryError:
            raise
        except sqlite3.Error as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot list node-run memberships: {error}",
            ) from error
        finally:
            connection.close()

    def verify_node(
        self,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
    ) -> VerifiedLogicalNode:
        checked_type, checked_key = self._validate_node_identity(node_type, node_logical_key)
        connection = self._connect()
        try:
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            if self._select_node_row(connection, checked_type, checked_key) is None:
                raise LogicalNodeRegistryError(
                    LogicalNodeRegistryErrorCode.NODE_NOT_FOUND,
                    f"logical node does not exist: {(checked_type, checked_key)!r}",
                )
            return self._verified_node_from_connection(connection, checked_type, checked_key)
        except LogicalNodeRegistryError:
            raise
        except sqlite3.Error as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot verify logical node: {error}",
            ) from error
        finally:
            connection.close()

    def publish_revision(self, revision: ImmutableNodeRevision) -> PublishedRevision:
        """Publish or resolve one immutable revision under an existing logical node."""

        checked = self._validate_revision(revision)
        revision_bytes = canonical_json_bytes(checked)
        commit_error: sqlite3.Error | None = None
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            self._verified_revision_subject_from_connection(
                connection,
                checked.subject_type,
                checked.subject_id,
            )
            id_row = self._select_revision_by_global_id(
                connection,
                checked.revision_id,
            )
            if id_row is not None:
                stored = self._revision_from_row(id_row)
                if not self._same_revision_command(stored, checked):
                    raise RevisionSelectionRegistryError(
                        RevisionSelectionRegistryErrorCode.REVISION_CONFLICT,
                        "revision_id is already bound to different immutable semantics",
                    )
                connection.rollback()
                return PublishedRevision(revision=stored, inserted=False)

            key_row = self._select_revision_by_logical_key(
                connection,
                checked.subject_type,
                checked.subject_id,
                checked.revision_logical_key,
            )
            if key_row is not None:
                stored = self._revision_from_row(key_row)
                if not self._same_revision_command(stored, checked):
                    raise RevisionSelectionRegistryError(
                        RevisionSelectionRegistryErrorCode.REVISION_CONFLICT,
                        "revision logical key is bound to different immutable semantics",
                    )
                connection.rollback()
                return PublishedRevision(revision=stored, inserted=False)

            self._verify_superseded_revision(connection, checked)
            self._insert_revision(connection, checked, revision_bytes)
            try:
                self._commit(connection)
            except sqlite3.Error as error:
                commit_error = error
                with suppress(sqlite3.Error):
                    connection.rollback()
        except RevisionSelectionRegistryError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except LogicalNodeRegistryError as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED,
                f"immutable revision publication failed: {error}",
            ) from error
        finally:
            connection.close()

        if commit_error is not None:
            recovered = self._recover_uncertain_revision_commit(checked)
            if recovered is None:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED,
                    f"immutable revision commit failed: {commit_error}",
                ) from commit_error
            return PublishedRevision(revision=recovered, inserted=None)

        committed_revision = self.lookup_revision(
            checked.subject_type,
            checked.subject_id,
            checked.revision_id,
        )
        if committed_revision is None or not self._same_revision_command(
            committed_revision,
            checked,
        ):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "committed immutable revision cannot be verified",
            )
        return PublishedRevision(revision=committed_revision, inserted=True)

    def select_revision(
        self,
        *,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
        selected_revision_id: OpaqueUuid,
        selection_decision_id: OpaqueUuid,
        selection_key_namespace: KeyNamespace,
        expected_previous_selection_decision_id: OpaqueUuid | None,
        selection_policy_version: SchemaVersion,
        selected_at: Rfc3339Timestamp,
    ) -> PublishedSelection:
        """Append one decision and compare-and-swap the replaceable projection."""

        subject = self._validate_revision_subject(subject_type, subject_id)
        checked_revision_id = self._validate_revision_uuid(
            selected_revision_id,
            "selected_revision_id",
        )
        checked_decision_id = self._validate_revision_uuid(
            selection_decision_id,
            "selection_decision_id",
        )
        checked_namespace = self._validate_revision_namespace(selection_key_namespace)
        checked_previous_id = (
            None
            if expected_previous_selection_decision_id is None
            else self._validate_revision_uuid(
                expected_previous_selection_decision_id,
                "expected_previous_selection_decision_id",
            )
        )
        checked_policy = self._validate_revision_policy(selection_policy_version)
        checked_selected_at = self._validate_revision_timestamp(selected_at, "selected_at")
        candidate: SelectionDecision | None = None
        commit_error: sqlite3.Error | None = None
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            verified = self._verified_revision_subject_from_connection(
                connection,
                *subject,
            )
            target = next(
                (
                    revision
                    for revision in verified.revisions
                    if revision.revision_id == checked_revision_id
                ),
                None,
            )
            if target is None:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.REVISION_NOT_FOUND,
                    f"selected revision does not exist under subject: {checked_revision_id}",
                )
            if target.eligibility_at_publication is not RevisionEligibility.ELIGIBLE:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.REVISION_INELIGIBLE,
                    "selected revision was ineligible at publication",
                )

            id_row = self._select_decision_by_global_id(
                connection,
                checked_decision_id,
            )
            if id_row is not None:
                stored = self._decision_from_row(id_row)
                if not self._decision_matches_request(
                    stored,
                    subject=subject,
                    selected_revision_id=checked_revision_id,
                    selection_key_namespace=checked_namespace,
                    previous_selection_decision_id=checked_previous_id,
                    selection_policy_version=checked_policy,
                ):
                    raise RevisionSelectionRegistryError(
                        RevisionSelectionRegistryErrorCode.SELECTION_CONFLICT,
                        "selection_decision_id is bound to a different immutable command",
                    )
                if verified.current is None:
                    raise RevisionSelectionRegistryError(
                        RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                        "stored selection decision has no current projection",
                    )
                connection.rollback()
                return PublishedSelection(
                    decision=stored,
                    current=verified.current,
                    decision_inserted=False,
                    projection_advanced=False,
                )

            previous: SelectionDecision | None = None
            if checked_previous_id is not None:
                previous = next(
                    (
                        decision
                        for decision in verified.decisions
                        if decision.selection_decision_id == checked_previous_id
                    ),
                    None,
                )
                if previous is None:
                    raise RevisionSelectionRegistryError(
                        RevisionSelectionRegistryErrorCode.STALE_SELECTION,
                        "expected previous selection decision does not exist",
                    )
            candidate = create_selection_decision(
                selection_decision_id=checked_decision_id,
                selection_key_namespace=checked_namespace,
                subject_type=subject[0],
                subject_id=subject[1],
                selected_revision_id=target.revision_id,
                selected_revision_logical_key=target.revision_logical_key,
                previous_selection_decision_id=(
                    None if previous is None else previous.selection_decision_id
                ),
                previous_selection_decision_logical_key=(
                    None if previous is None else previous.selection_decision_logical_key
                ),
                selection_sequence=(1 if previous is None else previous.selection_sequence + 1),
                selection_policy_version=checked_policy,
                projection_version=_CURRENT_PROJECTION_VERSION,
                selected_at=checked_selected_at,
            )
            semantic_retry = next(
                (
                    decision
                    for decision in verified.decisions
                    if decision.selection_decision_logical_key
                    == candidate.selection_decision_logical_key
                ),
                None,
            )
            if semantic_retry is not None:
                if not self._same_selection_command(semantic_retry, candidate):
                    raise RevisionSelectionRegistryError(
                        RevisionSelectionRegistryErrorCode.SELECTION_CONFLICT,
                        "selection logical key is bound to different immutable semantics",
                    )
                if verified.current is None:
                    raise RevisionSelectionRegistryError(
                        RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                        "stored selection decision has no current projection",
                    )
                connection.rollback()
                return PublishedSelection(
                    decision=semantic_retry,
                    current=verified.current,
                    decision_inserted=False,
                    projection_advanced=False,
                )

            actual_previous_id = (
                None if verified.current is None else verified.current.selection_decision_id
            )
            if actual_previous_id != checked_previous_id:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.STALE_SELECTION,
                    "current selection no longer matches the expected predecessor",
                )

            decision_bytes = canonical_json_bytes(candidate)
            current = self._current_from_decision(candidate)
            current_bytes = canonical_json_bytes(current)
            self._insert_selection_decision(connection, candidate, decision_bytes)
            if checked_previous_id is None:
                self._insert_current_selection(connection, current, current_bytes)
            else:
                updated = self._compare_and_swap_current_selection(
                    connection,
                    expected_previous_id=checked_previous_id,
                    current=current,
                    current_bytes=current_bytes,
                )
                if not updated:
                    raise RevisionSelectionRegistryError(
                        RevisionSelectionRegistryErrorCode.STALE_SELECTION,
                        "current-selection compare-and-swap did not match",
                    )
            try:
                self._commit(connection)
            except sqlite3.Error as error:
                commit_error = error
                with suppress(sqlite3.Error):
                    connection.rollback()
        except RevisionSelectionRegistryError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except LogicalNodeRegistryError as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise self._revision_error_from_logical(error) from error
        except (ValidationError, ValueError) as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INVALID_REQUEST,
                f"selection command is invalid: {error}",
            ) from error
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED,
                f"selection transaction failed: {error}",
            ) from error
        finally:
            connection.close()

        if candidate is None:
            raise AssertionError("selection candidate was not constructed")
        if commit_error is not None:
            recovered = self._recover_uncertain_selection_commit(candidate)
            if recovered is None:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED,
                    f"selection commit failed: {commit_error}",
                ) from commit_error
            decision, current = recovered
            return PublishedSelection(
                decision=decision,
                current=current,
                decision_inserted=None,
                projection_advanced=None,
            )

        verified = self.verify_subject(*subject)
        committed_decision = next(
            (
                item
                for item in verified.decisions
                if item.selection_decision_id == candidate.selection_decision_id
            ),
            None,
        )
        if committed_decision != candidate or verified.current is None:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "committed selection decision cannot be verified",
            )
        return PublishedSelection(
            decision=committed_decision,
            current=verified.current,
            decision_inserted=True,
            projection_advanced=True,
        )

    def lookup_revision(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
        revision_id: OpaqueUuid,
    ) -> ImmutableNodeRevision | None:
        subject = self._validate_revision_subject(subject_type, subject_id)
        checked_revision_id = self._validate_revision_uuid(revision_id, "revision_id")
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            if self._select_node_row(connection, *subject) is None:
                return None
            verified = self._verified_revision_subject_from_connection(
                connection,
                *subject,
            )
            return next(
                (
                    revision
                    for revision in verified.revisions
                    if revision.revision_id == checked_revision_id
                ),
                None,
            )
        except RevisionSelectionRegistryError:
            raise
        except LogicalNodeRegistryError as error:
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot look up immutable revision: {error}",
            ) from error
        finally:
            with suppress(sqlite3.Error):
                connection.rollback()
            connection.close()

    def lookup_selection_decision(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
        selection_decision_id: OpaqueUuid,
    ) -> SelectionDecision | None:
        subject = self._validate_revision_subject(subject_type, subject_id)
        checked_decision_id = self._validate_revision_uuid(
            selection_decision_id,
            "selection_decision_id",
        )
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            if self._select_node_row(connection, *subject) is None:
                return None
            verified = self._verified_revision_subject_from_connection(
                connection,
                *subject,
            )
            return next(
                (
                    decision
                    for decision in verified.decisions
                    if decision.selection_decision_id == checked_decision_id
                ),
                None,
            )
        except RevisionSelectionRegistryError:
            raise
        except LogicalNodeRegistryError as error:
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot look up selection decision: {error}",
            ) from error
        finally:
            with suppress(sqlite3.Error):
                connection.rollback()
            connection.close()

    def lookup_current_selection(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> CurrentSelection | None:
        subject = self._validate_revision_subject(subject_type, subject_id)
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            if self._select_node_row(connection, *subject) is None:
                return None
            return self._verified_revision_subject_from_connection(
                connection,
                *subject,
            ).current
        except RevisionSelectionRegistryError:
            raise
        except LogicalNodeRegistryError as error:
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot look up current selection: {error}",
            ) from error
        finally:
            with suppress(sqlite3.Error):
                connection.rollback()
            connection.close()

    def list_revisions(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> tuple[ImmutableNodeRevision, ...]:
        subject = self._validate_revision_subject(subject_type, subject_id)
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            if self._select_node_row(connection, *subject) is None:
                return ()
            return self._verified_revision_subject_from_connection(
                connection,
                *subject,
            ).revisions
        except RevisionSelectionRegistryError:
            raise
        except LogicalNodeRegistryError as error:
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot list immutable revisions: {error}",
            ) from error
        finally:
            with suppress(sqlite3.Error):
                connection.rollback()
            connection.close()

    def list_selection_decisions(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> tuple[SelectionDecision, ...]:
        subject = self._validate_revision_subject(subject_type, subject_id)
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            if self._select_node_row(connection, *subject) is None:
                return ()
            return self._verified_revision_subject_from_connection(
                connection,
                *subject,
            ).decisions
        except RevisionSelectionRegistryError:
            raise
        except LogicalNodeRegistryError as error:
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot list selection decisions: {error}",
            ) from error
        finally:
            with suppress(sqlite3.Error):
                connection.rollback()
            connection.close()

    def verify_subject(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> VerifiedRevisionSubject:
        subject = self._validate_revision_subject(subject_type, subject_id)
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            if self._select_node_row(connection, *subject) is None:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.SUBJECT_NOT_FOUND,
                    f"revision subject does not exist: {subject!r}",
                )
            return self._verified_revision_subject_from_connection(
                connection,
                *subject,
            )
        except RevisionSelectionRegistryError:
            raise
        except LogicalNodeRegistryError as error:
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot verify revision subject: {error}",
            ) from error
        finally:
            with suppress(sqlite3.Error):
                connection.rollback()
            connection.close()

    def rebuild_current_projection(self) -> tuple[CurrentSelection, ...]:
        """Replace the current projection from verified append-only decision chains."""

        expected: tuple[CurrentSelection, ...] = ()
        commit_error: sqlite3.Error | None = None
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            expected = self._expected_current_projection_from_connection(connection)
            connection.execute("DELETE FROM current_selections")
            for current in expected:
                current_bytes = canonical_json_bytes(current)
                self._insert_current_selection(connection, current, current_bytes)
            try:
                self._commit(connection)
            except sqlite3.Error as error:
                commit_error = error
                with suppress(sqlite3.Error):
                    connection.rollback()
        except RevisionSelectionRegistryError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except LogicalNodeRegistryError as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED,
                f"current-projection rebuild failed: {error}",
            ) from error
        finally:
            connection.close()

        if commit_error is not None:
            try:
                return self._read_all_current_projection()
            except RevisionSelectionRegistryError as recovery_error:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED,
                    "current-projection rebuild commit failed and the post-commit "
                    f"state cannot be verified: {recovery_error}",
                ) from commit_error
        return expected

    def _connect_revision(self) -> sqlite3.Connection:
        try:
            return self._connect()
        except LogicalNodeRegistryError as error:
            raise self._revision_error_from_logical(error) from error

    @staticmethod
    def _revision_error_from_logical(
        error: LogicalNodeRegistryError,
    ) -> RevisionSelectionRegistryError:
        mapping = {
            LogicalNodeRegistryErrorCode.INVALID_REQUEST: (
                RevisionSelectionRegistryErrorCode.INVALID_REQUEST
            ),
            LogicalNodeRegistryErrorCode.NODE_NOT_FOUND: (
                RevisionSelectionRegistryErrorCode.SUBJECT_NOT_FOUND
            ),
            LogicalNodeRegistryErrorCode.TRANSACTION_FAILED: (
                RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED
            ),
            LogicalNodeRegistryErrorCode.STORAGE_IO_ERROR: (
                RevisionSelectionRegistryErrorCode.STORAGE_IO_ERROR
            ),
            LogicalNodeRegistryErrorCode.INTEGRITY_ERROR: (
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR
            ),
        }
        return RevisionSelectionRegistryError(
            mapping.get(
                error.code,
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
            ),
            str(error),
        )

    @staticmethod
    def _validate_revision(revision: object) -> ImmutableNodeRevision:
        if not isinstance(revision, ImmutableNodeRevision):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INVALID_REQUEST,
                "revision must be an ImmutableNodeRevision",
            )
        try:
            return ImmutableNodeRevision.model_validate(
                revision.model_dump(mode="python"),
                strict=True,
            )
        except (ValidationError, ValueError) as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INVALID_REQUEST,
                f"revision is invalid: {error}",
            ) from error

    @classmethod
    def _validate_revision_subject(
        cls,
        subject_type: object,
        subject_id: object,
    ) -> tuple[NodeType, NodeLogicalKey]:
        try:
            return cls._validate_node_identity(subject_type, subject_id)
        except LogicalNodeRegistryError as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INVALID_REQUEST,
                str(error),
            ) from error

    @classmethod
    def _validate_revision_uuid(cls, value: object, field: str) -> OpaqueUuid:
        try:
            return cls._validate_uuid(value, field)
        except LogicalNodeRegistryError as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INVALID_REQUEST,
                str(error),
            ) from error

    @staticmethod
    def _validate_revision_namespace(value: object) -> KeyNamespace:
        try:
            return _KEY_NAMESPACE_ADAPTER.validate_python(value, strict=True)
        except ValidationError as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INVALID_REQUEST,
                "selection_key_namespace is not canonical",
            ) from error

    @staticmethod
    def _validate_revision_policy(value: object) -> SchemaVersion:
        try:
            return _SCHEMA_VERSION_ADAPTER.validate_python(value, strict=True)
        except ValidationError as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INVALID_REQUEST,
                "selection_policy_version is not canonical",
            ) from error

    @staticmethod
    def _validate_revision_timestamp(
        value: object,
        field: str,
    ) -> Rfc3339Timestamp:
        try:
            checked = _TIMESTAMP_ADAPTER.validate_python(value, strict=True)
            normalized = f"{checked[:-1]}+00:00" if checked.endswith("Z") else checked
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("timezone is absent")
            return checked
        except (ValidationError, ValueError) as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INVALID_REQUEST,
                f"{field} must be a valid timezone-bearing RFC3339 timestamp",
            ) from error

    @staticmethod
    def _same_revision_command(
        stored: ImmutableNodeRevision,
        requested: ImmutableNodeRevision,
    ) -> bool:
        excluded = {"revision_id", "published_at"}
        return stored.model_dump(exclude=excluded) == requested.model_dump(exclude=excluded)

    @staticmethod
    def _same_selection_command(
        stored: SelectionDecision,
        requested: SelectionDecision,
    ) -> bool:
        excluded = {"selection_decision_id", "selected_at"}
        return stored.model_dump(exclude=excluded) == requested.model_dump(exclude=excluded)

    @staticmethod
    def _decision_matches_request(
        stored: SelectionDecision,
        *,
        subject: tuple[NodeType, NodeLogicalKey],
        selected_revision_id: OpaqueUuid,
        selection_key_namespace: KeyNamespace,
        previous_selection_decision_id: OpaqueUuid | None,
        selection_policy_version: SchemaVersion,
    ) -> bool:
        return (
            (stored.subject_type, stored.subject_id) == subject
            and stored.selected_revision_id == selected_revision_id
            and stored.selection_key_namespace == selection_key_namespace
            and stored.previous_selection_decision_id == previous_selection_decision_id
            and stored.selection_policy_version == selection_policy_version
            and stored.projection_version == _CURRENT_PROJECTION_VERSION
        )

    def _validate_attach_request(
        self,
        *,
        node: LogicalNode,
        run_id: object,
        role: object,
        first_work_item_id: object,
        attached_at: object,
        existing_node_disposition: object,
    ) -> tuple[LogicalNode, ProcessingRunNodeMembership]:
        if not isinstance(node, LogicalNode):
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                "node must be a LogicalNode",
            )
        if (
            not isinstance(existing_node_disposition, RunNodeDisposition)
            or existing_node_disposition is RunNodeDisposition.CREATED
        ):
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                "existing_node_disposition must be REUSED, INVALIDATED, or OBSERVED",
            )
        checked_run = self._validate_uuid(run_id, "run_id")
        checked_work = self._validate_uuid(first_work_item_id, "first_work_item_id")
        checked_role = self._validate_role(role)
        try:
            checked_attached_at = _TIMESTAMP_ADAPTER.validate_python(attached_at, strict=True)
        except ValidationError as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                "attached_at must be an RFC3339 timestamp",
            ) from error
        try:
            checked_node = LogicalNode.model_validate(
                node.model_dump(mode="json"),
                strict=True,
            )
            membership = ProcessingRunNodeMembership(
                schema_version="1.0",
                run_id=checked_run,
                node_type=checked_node.node_type,
                node_logical_key=checked_node.node_logical_key,
                role=checked_role,
                disposition=existing_node_disposition,
                first_work_item_id=checked_work,
                attached_at=checked_attached_at,
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                f"invalid logical-node attach request: {error}",
            ) from error
        return checked_node, membership

    @staticmethod
    def _validate_node_identity(
        node_type: object,
        node_logical_key: object,
    ) -> tuple[NodeType, NodeLogicalKey]:
        try:
            return (
                _NODE_TYPE_ADAPTER.validate_python(node_type, strict=True),
                _NODE_KEY_ADAPTER.validate_python(node_logical_key, strict=True),
            )
        except ValidationError as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                f"invalid logical-node identity: {error}",
            ) from error

    @staticmethod
    def _validate_uuid(value: object, field: str) -> OpaqueUuid:
        try:
            return _UUID_ADAPTER.validate_python(value, strict=True)
        except ValidationError as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                f"{field} must be a lowercase UUID-shaped opaque ID",
            ) from error

    @classmethod
    def _validate_membership_identity(
        cls,
        run_id: object,
        node_type: object,
        node_logical_key: object,
        role: object,
    ) -> tuple[OpaqueUuid, NodeType, NodeLogicalKey, RunNodeRole]:
        checked_run = cls._validate_uuid(run_id, "run_id")
        checked_type, checked_key = cls._validate_node_identity(node_type, node_logical_key)
        checked_role = cls._validate_role(role)
        return checked_run, checked_type, checked_key, checked_role

    @staticmethod
    def _validate_role(value: object) -> RunNodeRole:
        try:
            return _ROLE_ADAPTER.validate_python(value, strict=True)
        except ValidationError as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INVALID_REQUEST,
                "role must be an uppercase canonical token",
            ) from error

    @staticmethod
    def _select_node_row(
        connection: sqlite3.Connection,
        node_type: str,
        node_logical_key: str,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                """
                SELECT *
                FROM logical_nodes
                WHERE node_type = ? AND node_logical_key = ?
                """,
                (node_type, node_logical_key),
            ).fetchone(),
        )

    @staticmethod
    def _select_membership_row(
        connection: sqlite3.Connection,
        identity: tuple[str, str, str, str],
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                """
                SELECT *
                FROM processing_run_nodes
                WHERE run_id = ?
                  AND node_type = ?
                  AND node_logical_key = ?
                  AND role = ?
                """,
                identity,
            ).fetchone(),
        )

    @staticmethod
    def _select_revision_by_global_id(
        connection: sqlite3.Connection,
        revision_id: str,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                """
                SELECT *
                FROM immutable_node_revisions
                WHERE revision_id = ?
                """,
                (revision_id,),
            ).fetchone(),
        )

    @staticmethod
    def _select_revision_by_logical_key(
        connection: sqlite3.Connection,
        subject_type: str,
        subject_id: str,
        revision_logical_key: str,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                """
                SELECT *
                FROM immutable_node_revisions
                WHERE subject_type = ?
                  AND subject_id = ?
                  AND revision_logical_key = ?
                """,
                (subject_type, subject_id, revision_logical_key),
            ).fetchone(),
        )

    @staticmethod
    def _select_decision_by_global_id(
        connection: sqlite3.Connection,
        selection_decision_id: str,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                """
                SELECT *
                FROM selection_decisions
                WHERE selection_decision_id = ?
                """,
                (selection_decision_id,),
            ).fetchone(),
        )

    @staticmethod
    def _select_current_row(
        connection: sqlite3.Connection,
        subject_type: str,
        subject_id: str,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                """
                SELECT *
                FROM current_selections
                WHERE subject_type = ? AND subject_id = ?
                """,
                (subject_type, subject_id),
            ).fetchone(),
        )

    @staticmethod
    def _insert_node(
        connection: sqlite3.Connection,
        node: LogicalNode,
        node_bytes: bytes,
    ) -> None:
        connection.execute(
            """
            INSERT INTO logical_nodes (
                node_type,
                node_logical_key,
                schema_version,
                key_namespace,
                semantic_sha256,
                identity_policy_version,
                node_json,
                node_json_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.node_type,
                node.node_logical_key,
                node.schema_version,
                node.key_namespace,
                node.semantic_sha256,
                node.identity_policy_version,
                node_bytes,
                exact_bytes_sha256(node_bytes),
            ),
        )

    @staticmethod
    def _insert_membership(
        connection: sqlite3.Connection,
        membership: ProcessingRunNodeMembership,
        membership_bytes: bytes,
    ) -> None:
        connection.execute(
            """
            INSERT INTO processing_run_nodes (
                run_id,
                node_type,
                node_logical_key,
                role,
                schema_version,
                disposition,
                first_work_item_id,
                attached_at,
                membership_json,
                membership_json_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                membership.run_id,
                membership.node_type,
                membership.node_logical_key,
                membership.role,
                membership.schema_version,
                membership.disposition.value,
                membership.first_work_item_id,
                membership.attached_at,
                membership_bytes,
                exact_bytes_sha256(membership_bytes),
            ),
        )

    @staticmethod
    def _insert_revision(
        connection: sqlite3.Connection,
        revision: ImmutableNodeRevision,
        revision_bytes: bytes,
    ) -> None:
        connection.execute(
            """
            INSERT INTO immutable_node_revisions (
                subject_type,
                subject_id,
                revision_id,
                schema_version,
                revision_key_namespace,
                revision_logical_key,
                semantic_sha256,
                payload_sha256,
                lineage_sha256,
                status_at_publication,
                eligibility_at_publication,
                revision_policy_version,
                supersedes_revision_id,
                supersedes_revision_logical_key,
                published_at,
                revision_json,
                revision_json_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision.subject_type,
                revision.subject_id,
                revision.revision_id,
                revision.schema_version,
                revision.revision_key_namespace,
                revision.revision_logical_key,
                revision.semantic_sha256,
                revision.payload_sha256,
                revision.lineage_sha256,
                revision.status_at_publication,
                revision.eligibility_at_publication.value,
                revision.revision_policy_version,
                revision.supersedes_revision_id,
                revision.supersedes_revision_logical_key,
                revision.published_at,
                revision_bytes,
                exact_bytes_sha256(revision_bytes),
            ),
        )

    @staticmethod
    def _insert_selection_decision(
        connection: sqlite3.Connection,
        decision: SelectionDecision,
        decision_bytes: bytes,
    ) -> None:
        connection.execute(
            """
            INSERT INTO selection_decisions (
                subject_type,
                subject_id,
                selection_decision_id,
                schema_version,
                selection_key_namespace,
                selection_decision_logical_key,
                semantic_sha256,
                selected_revision_id,
                selected_revision_logical_key,
                previous_selection_decision_id,
                previous_selection_decision_logical_key,
                selection_sequence,
                selection_policy_version,
                projection_version,
                selected_at,
                decision_json,
                decision_json_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.subject_type,
                decision.subject_id,
                decision.selection_decision_id,
                decision.schema_version,
                decision.selection_key_namespace,
                decision.selection_decision_logical_key,
                decision.semantic_sha256,
                decision.selected_revision_id,
                decision.selected_revision_logical_key,
                decision.previous_selection_decision_id,
                decision.previous_selection_decision_logical_key,
                decision.selection_sequence,
                decision.selection_policy_version,
                decision.projection_version,
                decision.selected_at,
                decision_bytes,
                exact_bytes_sha256(decision_bytes),
            ),
        )

    @staticmethod
    def _insert_current_selection(
        connection: sqlite3.Connection,
        current: CurrentSelection,
        current_bytes: bytes,
    ) -> None:
        connection.execute(
            """
            INSERT INTO current_selections (
                subject_type,
                subject_id,
                schema_version,
                selected_revision_id,
                selection_decision_id,
                selection_policy_version,
                projection_version,
                selected_at,
                current_json,
                current_json_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                current.subject_type,
                current.subject_id,
                current.schema_version,
                current.selected_revision_id,
                current.selection_decision_id,
                current.selection_policy_version,
                current.projection_version,
                current.selected_at,
                current_bytes,
                exact_bytes_sha256(current_bytes),
            ),
        )

    @staticmethod
    def _compare_and_swap_current_selection(
        connection: sqlite3.Connection,
        *,
        expected_previous_id: OpaqueUuid,
        current: CurrentSelection,
        current_bytes: bytes,
    ) -> bool:
        cursor = connection.execute(
            """
            UPDATE current_selections
            SET schema_version = ?,
                selected_revision_id = ?,
                selection_decision_id = ?,
                selection_policy_version = ?,
                projection_version = ?,
                selected_at = ?,
                current_json = ?,
                current_json_sha256 = ?
            WHERE subject_type = ?
              AND subject_id = ?
              AND selection_decision_id = ?
            """,
            (
                current.schema_version,
                current.selected_revision_id,
                current.selection_decision_id,
                current.selection_policy_version,
                current.projection_version,
                current.selected_at,
                current_bytes,
                exact_bytes_sha256(current_bytes),
                current.subject_type,
                current.subject_id,
                expected_previous_id,
            ),
        )
        return cursor.rowcount == 1

    @staticmethod
    def _is_idempotent_retry(
        stored: ProcessingRunNodeMembership,
        requested: ProcessingRunNodeMembership,
    ) -> bool:
        if stored == requested:
            return True
        if (
            stored.disposition is RunNodeDisposition.CREATED
            and requested.disposition is RunNodeDisposition.REUSED
        ):
            return stored.model_copy(update={"disposition": RunNodeDisposition.REUSED}) == requested
        return False

    def _node_from_row(self, row: sqlite3.Row) -> LogicalNode:
        raw = _row_bytes(row, "node_json")
        stored_digest = _row_text(row, "node_json_sha256")
        if exact_bytes_sha256(raw) != stored_digest:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "stored logical-node record digest is corrupt",
            )
        try:
            node = LogicalNode.model_validate_json(raw, strict=True)
        except (ValidationError, ValueError) as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"stored logical-node document is invalid: {error}",
            ) from error
        if canonical_json_bytes(node) != raw:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "stored logical-node document is not canonical JSON",
            )
        normalized = (
            _row_text(row, "schema_version"),
            _row_text(row, "node_type"),
            _row_text(row, "node_logical_key"),
            _row_text(row, "key_namespace"),
            _row_text(row, "semantic_sha256"),
            _row_text(row, "identity_policy_version"),
        )
        expected = (
            node.schema_version,
            node.node_type,
            node.node_logical_key,
            node.key_namespace,
            node.semantic_sha256,
            node.identity_policy_version,
        )
        if normalized != expected:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "logical-node normalized columns disagree with canonical JSON",
            )
        return node

    def _membership_from_row(self, row: sqlite3.Row) -> ProcessingRunNodeMembership:
        raw = _row_bytes(row, "membership_json")
        stored_digest = _row_text(row, "membership_json_sha256")
        if exact_bytes_sha256(raw) != stored_digest:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "stored membership record digest is corrupt",
            )
        try:
            membership = ProcessingRunNodeMembership.model_validate_json(raw, strict=True)
        except (ValidationError, ValueError) as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"stored run-node membership document is invalid: {error}",
            ) from error
        if canonical_json_bytes(membership) != raw:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "stored membership document is not canonical JSON",
            )
        normalized = (
            _row_text(row, "schema_version"),
            _row_text(row, "run_id"),
            _row_text(row, "node_type"),
            _row_text(row, "node_logical_key"),
            _row_text(row, "role"),
            _row_text(row, "disposition"),
            _row_text(row, "first_work_item_id"),
            _row_text(row, "attached_at"),
        )
        expected = (
            membership.schema_version,
            membership.run_id,
            membership.node_type,
            membership.node_logical_key,
            membership.role,
            membership.disposition.value,
            membership.first_work_item_id,
            membership.attached_at,
        )
        if normalized != expected:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "membership normalized columns disagree with canonical JSON",
            )
        return membership

    def _revision_from_row(self, row: sqlite3.Row) -> ImmutableNodeRevision:
        raw = _row_bytes(row, "revision_json")
        if exact_bytes_sha256(raw) != _row_text(row, "revision_json_sha256"):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "stored immutable-revision record digest is corrupt",
            )
        try:
            revision = ImmutableNodeRevision.model_validate_json(raw, strict=True)
        except (ValidationError, ValueError) as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"stored immutable-revision document is invalid: {error}",
            ) from error
        if canonical_json_bytes(revision) != raw:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "stored immutable-revision document is not canonical JSON",
            )
        normalized = (
            _row_text(row, "subject_type"),
            _row_text(row, "subject_id"),
            _row_text(row, "revision_id"),
            _row_text(row, "schema_version"),
            _row_text(row, "revision_key_namespace"),
            _row_text(row, "revision_logical_key"),
            _row_text(row, "semantic_sha256"),
            _row_text(row, "payload_sha256"),
            _row_text(row, "lineage_sha256"),
            _row_text(row, "status_at_publication"),
            _row_text(row, "eligibility_at_publication"),
            _row_text(row, "revision_policy_version"),
            _row_optional_text(row, "supersedes_revision_id"),
            _row_optional_text(row, "supersedes_revision_logical_key"),
            _row_text(row, "published_at"),
        )
        expected = (
            revision.subject_type,
            revision.subject_id,
            revision.revision_id,
            revision.schema_version,
            revision.revision_key_namespace,
            revision.revision_logical_key,
            revision.semantic_sha256,
            revision.payload_sha256,
            revision.lineage_sha256,
            revision.status_at_publication,
            revision.eligibility_at_publication.value,
            revision.revision_policy_version,
            revision.supersedes_revision_id,
            revision.supersedes_revision_logical_key,
            revision.published_at,
        )
        if normalized != expected:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "immutable-revision columns disagree with canonical JSON",
            )
        return revision

    def _decision_from_row(self, row: sqlite3.Row) -> SelectionDecision:
        raw = _row_bytes(row, "decision_json")
        if exact_bytes_sha256(raw) != _row_text(row, "decision_json_sha256"):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "stored selection-decision record digest is corrupt",
            )
        try:
            decision = SelectionDecision.model_validate_json(raw, strict=True)
        except (ValidationError, ValueError) as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"stored selection-decision document is invalid: {error}",
            ) from error
        if canonical_json_bytes(decision) != raw:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "stored selection-decision document is not canonical JSON",
            )
        normalized = (
            _row_text(row, "subject_type"),
            _row_text(row, "subject_id"),
            _row_text(row, "selection_decision_id"),
            _row_text(row, "schema_version"),
            _row_text(row, "selection_key_namespace"),
            _row_text(row, "selection_decision_logical_key"),
            _row_text(row, "semantic_sha256"),
            _row_text(row, "selected_revision_id"),
            _row_text(row, "selected_revision_logical_key"),
            _row_optional_text(row, "previous_selection_decision_id"),
            _row_optional_text(row, "previous_selection_decision_logical_key"),
            _row_int(row, "selection_sequence"),
            _row_text(row, "selection_policy_version"),
            _row_text(row, "projection_version"),
            _row_text(row, "selected_at"),
        )
        expected = (
            decision.subject_type,
            decision.subject_id,
            decision.selection_decision_id,
            decision.schema_version,
            decision.selection_key_namespace,
            decision.selection_decision_logical_key,
            decision.semantic_sha256,
            decision.selected_revision_id,
            decision.selected_revision_logical_key,
            decision.previous_selection_decision_id,
            decision.previous_selection_decision_logical_key,
            decision.selection_sequence,
            decision.selection_policy_version,
            decision.projection_version,
            decision.selected_at,
        )
        if normalized != expected:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "selection-decision columns disagree with canonical JSON",
            )
        return decision

    def _current_from_row(self, row: sqlite3.Row) -> CurrentSelection:
        raw = _row_bytes(row, "current_json")
        if exact_bytes_sha256(raw) != _row_text(row, "current_json_sha256"):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "stored current-selection record digest is corrupt",
            )
        try:
            current = CurrentSelection.model_validate_json(raw, strict=True)
        except (ValidationError, ValueError) as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"stored current-selection document is invalid: {error}",
            ) from error
        if canonical_json_bytes(current) != raw:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "stored current-selection document is not canonical JSON",
            )
        normalized = (
            _row_text(row, "subject_type"),
            _row_text(row, "subject_id"),
            _row_text(row, "schema_version"),
            _row_text(row, "selected_revision_id"),
            _row_text(row, "selection_decision_id"),
            _row_text(row, "selection_policy_version"),
            _row_text(row, "projection_version"),
            _row_text(row, "selected_at"),
        )
        expected = (
            current.subject_type,
            current.subject_id,
            current.schema_version,
            current.selected_revision_id,
            current.selection_decision_id,
            current.selection_policy_version,
            current.projection_version,
            current.selected_at,
        )
        if normalized != expected:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "current-selection columns disagree with canonical JSON",
            )
        return current

    @staticmethod
    def _current_from_decision(decision: SelectionDecision) -> CurrentSelection:
        return CurrentSelection(
            schema_version="1.0",
            subject_type=decision.subject_type,
            subject_id=decision.subject_id,
            selected_revision_id=decision.selected_revision_id,
            selection_decision_id=decision.selection_decision_id,
            selection_policy_version=decision.selection_policy_version,
            projection_version=decision.projection_version,
            selected_at=decision.selected_at,
        )

    def _verified_node_from_connection(
        self,
        connection: sqlite3.Connection,
        node_type: str,
        node_logical_key: str,
    ) -> VerifiedLogicalNode:
        row = self._select_node_row(connection, node_type, node_logical_key)
        if row is None:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.NODE_NOT_FOUND,
                f"logical node does not exist: {(node_type, node_logical_key)!r}",
            )
        node = self._node_from_row(row)
        if node.identity != (node_type, node_logical_key):
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "logical-node lookup identity disagrees with its canonical document",
            )
        membership_rows = connection.execute(
            """
            SELECT *
            FROM processing_run_nodes
            WHERE node_type = ? AND node_logical_key = ?
            ORDER BY run_id, role
            """,
            (node_type, node_logical_key),
        ).fetchall()
        memberships = tuple(self._membership_from_row(item) for item in membership_rows)
        if not memberships:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "logical node has no creating run membership",
            )
        if any(
            membership.node_type != node_type or membership.node_logical_key != node_logical_key
            for membership in memberships
        ):
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "logical node contains a membership for another node identity",
            )
        identities = tuple(membership.identity for membership in memberships)
        if len(identities) != len(set(identities)):
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "logical node contains duplicate membership identities",
            )
        creators = tuple(
            membership
            for membership in memberships
            if membership.disposition is RunNodeDisposition.CREATED
        )
        if len(creators) != 1:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "logical node must have exactly one immutable CREATED membership",
            )
        return VerifiedLogicalNode(node=node, memberships=memberships)

    def _verify_superseded_revision(
        self,
        connection: sqlite3.Connection,
        revision: ImmutableNodeRevision,
    ) -> None:
        if revision.supersedes_revision_id is None:
            return
        if (
            revision.supersedes_revision_id == revision.revision_id
            or revision.supersedes_revision_logical_key == revision.revision_logical_key
        ):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.REVISION_CONFLICT,
                "a revision cannot supersede itself",
            )
        row = self._select_revision_by_global_id(
            connection,
            revision.supersedes_revision_id,
        )
        if row is None:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.REVISION_NOT_FOUND,
                "superseded revision does not exist",
            )
        superseded = self._revision_from_row(row)
        if (
            superseded.subject_type != revision.subject_type
            or superseded.subject_id != revision.subject_id
            or superseded.revision_logical_key != revision.supersedes_revision_logical_key
        ):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.REVISION_CONFLICT,
                "superseded revision is not the referenced revision under this subject",
            )

    def _verified_revision_history_from_connection(
        self,
        connection: sqlite3.Connection,
        subject_type: str,
        subject_id: str,
    ) -> tuple[
        LogicalNode,
        tuple[ImmutableNodeRevision, ...],
        tuple[SelectionDecision, ...],
        CurrentSelection | None,
    ]:
        node = self._verified_node_from_connection(
            connection,
            subject_type,
            subject_id,
        ).node
        revision_rows = connection.execute(
            """
            SELECT *
            FROM immutable_node_revisions
            WHERE subject_type = ? AND subject_id = ?
            ORDER BY revision_logical_key, revision_id
            """,
            (subject_type, subject_id),
        ).fetchall()
        revisions = tuple(self._revision_from_row(row) for row in revision_rows)
        if any(
            (revision.subject_type, revision.subject_id) != (subject_type, subject_id)
            for revision in revisions
        ):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "revision row belongs to a different subject",
            )
        revisions_by_id = {revision.revision_id: revision for revision in revisions}
        revisions_by_key = {revision.revision_logical_key: revision for revision in revisions}
        if len(revisions_by_id) != len(revisions) or len(revisions_by_key) != len(revisions):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "revision subject contains duplicate immutable identities",
            )
        for revision in revisions:
            if revision.supersedes_revision_id is None:
                continue
            if (
                revision.supersedes_revision_id == revision.revision_id
                or revision.supersedes_revision_logical_key == revision.revision_logical_key
            ):
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                    "stored revision supersedes itself",
                )
            superseded = revisions_by_id.get(revision.supersedes_revision_id)
            if (
                superseded is None
                or superseded.revision_logical_key != revision.supersedes_revision_logical_key
            ):
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                    "stored supersedes lineage is absent or mismatched",
                )

        decision_rows = connection.execute(
            """
            SELECT *
            FROM selection_decisions
            WHERE subject_type = ? AND subject_id = ?
            ORDER BY selection_sequence, selection_decision_id
            """,
            (subject_type, subject_id),
        ).fetchall()
        decisions = tuple(self._decision_from_row(row) for row in decision_rows)
        if any(
            (decision.subject_type, decision.subject_id) != (subject_type, subject_id)
            for decision in decisions
        ):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "selection-decision row belongs to a different subject",
            )
        decision_ids = tuple(decision.selection_decision_id for decision in decisions)
        decision_keys = tuple(decision.selection_decision_logical_key for decision in decisions)
        if len(set(decision_ids)) != len(decisions) or len(set(decision_keys)) != len(decisions):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "selection chain contains duplicate immutable identities",
            )
        if tuple(decision.selection_sequence for decision in decisions) != tuple(
            range(1, len(decisions) + 1)
        ):
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "selection chain sequence is not contiguous from one",
            )
        previous: SelectionDecision | None = None
        for decision in decisions:
            expected_previous = (
                (None, None)
                if previous is None
                else (
                    previous.selection_decision_id,
                    previous.selection_decision_logical_key,
                )
            )
            if (
                decision.previous_selection_decision_id,
                decision.previous_selection_decision_logical_key,
            ) != expected_previous:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                    "selection decisions do not form one linear predecessor chain",
                )
            selected = revisions_by_id.get(decision.selected_revision_id)
            if (
                selected is None
                or selected.revision_logical_key != decision.selected_revision_logical_key
            ):
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                    "selection decision references an absent or mismatched revision",
                )
            if selected.eligibility_at_publication is not RevisionEligibility.ELIGIBLE:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                    "selection decision references an ineligible revision",
                )
            if decision.projection_version != _CURRENT_PROJECTION_VERSION:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                    "selection decision uses an unsupported projection version",
                )
            previous = decision
        expected_current = None if previous is None else self._current_from_decision(previous)
        return node, revisions, decisions, expected_current

    def _verified_revision_subject_from_connection(
        self,
        connection: sqlite3.Connection,
        subject_type: str,
        subject_id: str,
    ) -> VerifiedRevisionSubject:
        node, revisions, decisions, expected_current = (
            self._verified_revision_history_from_connection(
                connection,
                subject_type,
                subject_id,
            )
        )
        current_row = self._select_current_row(connection, subject_type, subject_id)
        current = None if current_row is None else self._current_from_row(current_row)
        if current != expected_current:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                "current projection does not equal the verified selection-chain tail",
            )
        return VerifiedRevisionSubject(
            node=node,
            revisions=revisions,
            decisions=decisions,
            current=current,
        )

    def _expected_current_projection_from_connection(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[CurrentSelection, ...]:
        subject_rows = connection.execute(
            """
            SELECT subject_type, subject_id FROM immutable_node_revisions
            UNION
            SELECT subject_type, subject_id FROM selection_decisions
            ORDER BY subject_type, subject_id
            """
        ).fetchall()
        expected: list[CurrentSelection] = []
        for row in subject_rows:
            _, _, _, current = self._verified_revision_history_from_connection(
                connection,
                _row_text(row, "subject_type"),
                _row_text(row, "subject_id"),
            )
            if current is not None:
                expected.append(current)
        return tuple(expected)

    @staticmethod
    def _verify_database_health(
        connection: sqlite3.Connection,
    ) -> None:
        quick_check = connection.execute("PRAGMA quick_check(1)").fetchone()
        result: object = None if quick_check is None else quick_check[0]
        if result != "ok":
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                f"SQLite quick_check failed: {result!r}",
            )
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_rows:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
                "SQLite foreign-key check found orphaned registry rows",
            )

    def _recover_uncertain_commit(
        self,
        node: LogicalNode,
        expected_membership: ProcessingRunNodeMembership,
    ) -> VerifiedLogicalNode | None:
        try:
            verified = self.verify_node(node.node_type, node.node_logical_key)
        except LogicalNodeRegistryError as error:
            if error.code is LogicalNodeRegistryErrorCode.NODE_NOT_FOUND:
                return None
            raise
        if canonical_json_bytes(verified.node) != canonical_json_bytes(node):
            return None
        committed = next(
            (
                membership
                for membership in verified.memberships
                if membership.identity == expected_membership.identity
            ),
            None,
        )
        if committed != expected_membership:
            return None
        return verified

    def _recover_uncertain_revision_commit(
        self,
        expected: ImmutableNodeRevision,
    ) -> ImmutableNodeRevision | None:
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            row = self._select_revision_by_logical_key(
                connection,
                expected.subject_type,
                expected.subject_id,
                expected.revision_logical_key,
            )
            if row is None:
                return None
            stored = self._revision_from_row(row)
            if not self._same_revision_command(stored, expected):
                return None
            verified = self._verified_revision_subject_from_connection(
                connection,
                expected.subject_type,
                expected.subject_id,
            )
            if stored not in verified.revisions:
                return None
            return stored
        except RevisionSelectionRegistryError as error:
            if error.code is RevisionSelectionRegistryErrorCode.SUBJECT_NOT_FOUND:
                return None
            raise
        except LogicalNodeRegistryError as error:
            if error.code is LogicalNodeRegistryErrorCode.NODE_NOT_FOUND:
                return None
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot recover uncertain revision commit: {error}",
            ) from error
        finally:
            with suppress(sqlite3.Error):
                connection.rollback()
            connection.close()

    def _recover_uncertain_selection_commit(
        self,
        expected: SelectionDecision,
    ) -> tuple[SelectionDecision, CurrentSelection] | None:
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            verified = self._verified_revision_subject_from_connection(
                connection,
                expected.subject_type,
                expected.subject_id,
            )
            stored = next(
                (
                    decision
                    for decision in verified.decisions
                    if decision.selection_decision_logical_key
                    == expected.selection_decision_logical_key
                ),
                None,
            )
            if (
                stored is None
                or not self._same_selection_command(stored, expected)
                or verified.current is None
            ):
                return None
            return stored, verified.current
        except RevisionSelectionRegistryError as error:
            if error.code is RevisionSelectionRegistryErrorCode.SUBJECT_NOT_FOUND:
                return None
            raise
        except LogicalNodeRegistryError as error:
            if error.code is LogicalNodeRegistryErrorCode.NODE_NOT_FOUND:
                return None
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot recover uncertain selection commit: {error}",
            ) from error
        finally:
            with suppress(sqlite3.Error):
                connection.rollback()
            connection.close()

    def _read_all_current_projection(self) -> tuple[CurrentSelection, ...]:
        connection = self._connect_revision()
        try:
            connection.execute("BEGIN")
            self._verify_database_schema(connection)
            self._verify_database_health(connection)
            expected = self._expected_current_projection_from_connection(connection)
            rows = connection.execute(
                """
                SELECT *
                FROM current_selections
                ORDER BY subject_type, subject_id
                """
            ).fetchall()
            actual = tuple(self._current_from_row(row) for row in rows)
            if actual != expected:
                raise RevisionSelectionRegistryError(
                    RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                    "current projection does not match verified decision-chain tails",
                )
            return actual
        except RevisionSelectionRegistryError:
            raise
        except LogicalNodeRegistryError as error:
            raise self._revision_error_from_logical(error) from error
        except sqlite3.Error as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot verify the rebuilt current projection: {error}",
            ) from error
        finally:
            with suppress(sqlite3.Error):
                connection.rollback()
            connection.close()

    def _commit(self, connection: sqlite3.Connection) -> None:
        connection.commit()


def _row_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise LogicalNodeRegistryError(
            LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
            f"SQLite column {column!r} is not text",
        )
    return value


def _row_optional_text(row: sqlite3.Row, column: str) -> str | None:
    value: object = row[column]
    if value is not None and not isinstance(value, str):
        raise RevisionSelectionRegistryError(
            RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR,
            f"SQLite column {column!r} is neither text nor null",
        )
    return value


def _row_int(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogicalNodeRegistryError(
            LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
            f"SQLite column {column!r} is not an integer",
        )
    return value


def _row_bytes(row: sqlite3.Row, column: str) -> bytes:
    value: object = row[column]
    if not isinstance(value, bytes):
        raise LogicalNodeRegistryError(
            LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
            f"SQLite column {column!r} is not a BLOB",
        )
    return value


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None:
        raise LogicalNodeRegistryError(
            LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
            f"SQLite PRAGMA {name} returned no value",
        )
    value: object = row[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogicalNodeRegistryError(
            LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
            f"SQLite PRAGMA {name} returned a non-integer value",
        )
    return value


__all__ = ["LocalLogicalNodeRegistry"]
