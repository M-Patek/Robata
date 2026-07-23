"""Local durable storage for generic and inference-call barriers."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from functools import cache
from pathlib import Path
from typing import Final, TypeVar

from pydantic import BaseModel, ValidationError

from robata.contracts.hashing import CanonicalizationError, canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.inference.call_barrier import (
    InferenceCallBarrierConflictError,
    InferenceCallBarrierDefinition,
    InferenceCallBarrierError,
    InferenceCallBarrierStorage,
    InferenceCallPartCompletion,
    InferenceCallReduction,
)
from robata.inference.models import InferenceStatus
from robata.queue.barrier import (
    Barrier,
    BarrierMember,
    BarrierState,
    BarrierStorage,
)
from robata.queue.models import DependencyCriticality
from robata.queue.stage import StageStatus
from robata.queue.wire import (
    PersistedBarrier,
    validate_registered_persisted_barrier,
)
from robata.runtime.observability import (
    RuntimeAttributeValue,
    RuntimeObserver,
    runtime_increment,
    runtime_span,
)

_APPLICATION_ID: Final = 0x52424152  # "RBAR"
_SCHEMA_VERSION: Final = 1
_BUSY_TIMEOUT_MS: Final = 30_000

_SUCCESS_OUTCOMES: Final = frozenset(
    {
        StageStatus.SUCCEEDED,
        StageStatus.SKIPPED_POLICY,
        StageStatus.SKIPPED_NOT_NEEDED,
    }
)
_FAILURE_OUTCOMES: Final = frozenset(
    {
        StageStatus.FAILED,
        StageStatus.CANCELLED,
        StageStatus.EXPIRED,
        StageStatus.QUARANTINED,
        StageStatus.INCOMPLETE,
    }
)
_TERMINAL_OUTCOMES: Final = _SUCCESS_OUTCOMES | _FAILURE_OUTCOMES

_SCHEMA_SQL: Final = """
CREATE TABLE barrier_definitions (
    barrier_id TEXT PRIMARY KEY,
    logical_key TEXT NOT NULL UNIQUE,
    expected_member_count INTEGER NOT NULL CHECK (expected_member_count >= 0),
    empty_semantics TEXT NOT NULL,
    reduction_policy TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED', 'FAILED')),
    required_success_count INTEGER NOT NULL CHECK (
        required_success_count >= 0 AND required_success_count <= expected_member_count
    ),
    max_degraded_failures INTEGER NOT NULL CHECK (
        max_degraded_failures >= 0 AND max_degraded_failures <= expected_member_count
    ),
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL
);

CREATE TABLE barrier_states (
    barrier_id TEXT PRIMARY KEY,
    state_version INTEGER NOT NULL CHECK (state_version >= 0),
    completed_members INTEGER NOT NULL CHECK (completed_members >= 0),
    pending_members INTEGER NOT NULL CHECK (pending_members >= 0),
    failed_members INTEGER NOT NULL CHECK (failed_members >= 0),
    status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED', 'FAILED')),
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    FOREIGN KEY (barrier_id) REFERENCES barrier_definitions (barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE barrier_members (
    barrier_id TEXT NOT NULL,
    work_item_id TEXT NOT NULL,
    criticality TEXT NOT NULL CHECK (criticality IN ('REQUIRED', 'DEGRADABLE', 'OPTIONAL')),
    outcome TEXT NOT NULL CHECK (
        outcome IN (
            'SUCCEEDED', 'SKIPPED_POLICY', 'SKIPPED_NOT_NEEDED',
            'FAILED', 'CANCELLED', 'EXPIRED', 'QUARANTINED', 'INCOMPLETE'
        )
    ),
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    PRIMARY KEY (barrier_id, work_item_id),
    FOREIGN KEY (barrier_id) REFERENCES barrier_definitions (barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) WITHOUT ROWID;

CREATE TABLE inference_call_barrier_definitions (
    barrier_id TEXT PRIMARY KEY,
    barrier_semantic_sha256 TEXT NOT NULL,
    barrier_logical_key TEXT NOT NULL,
    input_plan_semantic_sha256 TEXT NOT NULL,
    call_plan_sha256 TEXT NOT NULL,
    part_count INTEGER NOT NULL CHECK (part_count > 0),
    reduction_policy TEXT NOT NULL,
    reduction_policy_version TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    FOREIGN KEY (barrier_id) REFERENCES barrier_definitions (barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE inference_call_part_completions (
    barrier_id TEXT NOT NULL,
    part_ordinal INTEGER NOT NULL CHECK (part_ordinal >= 0),
    part_count INTEGER NOT NULL CHECK (part_count > 0),
    part_semantic_sha256 TEXT NOT NULL,
    part_logical_key TEXT NOT NULL,
    part_idempotency_key TEXT NOT NULL,
    completion_id TEXT NOT NULL UNIQUE,
    completion_semantic_sha256 TEXT NOT NULL,
    inference_id TEXT NOT NULL,
    logical_invocation_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('SUCCEEDED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'INVALID_OUTPUT')
    ),
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    PRIMARY KEY (barrier_id, part_ordinal),
    UNIQUE (barrier_id, part_semantic_sha256),
    FOREIGN KEY (barrier_id) REFERENCES inference_call_barrier_definitions (barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) WITHOUT ROWID;

CREATE TABLE inference_call_reductions (
    barrier_id TEXT PRIMARY KEY,
    reduction_id TEXT NOT NULL UNIQUE,
    reduction_semantic_sha256 TEXT NOT NULL,
    normalized_output_sha256 TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    FOREIGN KEY (barrier_id) REFERENCES inference_call_barrier_definitions (barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
"""

_IMMUTABLE_TABLES: Final = (
    "barrier_definitions",
    "barrier_members",
    "inference_call_barrier_definitions",
    "inference_call_part_completions",
    "inference_call_reductions",
)


class SQLiteBarrierStorageError(InferenceCallBarrierError):
    """SQLite schema, transaction, or persisted barrier data failed closed."""


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
        raise AssertionError("barrier SQLite schema is incomplete")
    return tuple(statements)


def _immutable_triggers(table: str) -> tuple[str, str]:
    return (
        f"""
        CREATE TRIGGER {table}_no_update
        BEFORE UPDATE ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{table} is append-only');
        END
        """,
        f"""
        CREATE TRIGGER {table}_no_delete
        BEFORE DELETE ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{table} is append-only');
        END
        """,
    )


_SCHEMA_STATEMENTS: Final = (
    *_split_sql_script(_SCHEMA_SQL),
    *(statement for table in _IMMUTABLE_TABLES for statement in _immutable_triggers(table)),
    """
    CREATE TRIGGER barrier_states_no_delete
    BEFORE DELETE ON barrier_states
    BEGIN
        SELECT RAISE(ABORT, 'barrier_states cannot be deleted');
    END
    """,
)

_ResultT = TypeVar("_ResultT")


class SQLiteBarrierStorage(BarrierStorage, InferenceCallBarrierStorage):
    """One SQLite authority for generic and inference-call barrier facts."""

    def __init__(
        self,
        database_path: Path,
        *,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a pathlib.Path")
        self._runtime_observer = runtime_observer
        try:
            if database_path.exists() and database_path.is_symlink():
                raise SQLiteBarrierStorageError(
                    f"barrier database must not be a symlink: {database_path}"
                )
            parent = database_path.parent
            if parent.exists() and parent.is_symlink():
                raise SQLiteBarrierStorageError(
                    f"barrier database parent must not be a symlink: {parent}"
                )
            parent.mkdir(parents=True, exist_ok=True)
            if not parent.is_dir():
                raise SQLiteBarrierStorageError(
                    f"barrier database parent is not a directory: {parent}"
                )
            self._database_path = parent.resolve(strict=True) / database_path.name
        except SQLiteBarrierStorageError:
            raise
        except OSError as exc:
            raise SQLiteBarrierStorageError(
                f"cannot prepare barrier database path {database_path}"
            ) from exc
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def get_barrier(self, barrier_id: str) -> Barrier | None:
        key = _nonempty_string(barrier_id, "barrier_id")

        def read(connection: sqlite3.Connection) -> Barrier | None:
            row = connection.execute(
                "SELECT * FROM barrier_definitions WHERE barrier_id = ?",
                (key,),
            ).fetchone()
            return None if row is None else self._barrier_from_row(row)

        return self._transaction(write=False, operation_name="get_barrier", operation=read)

    def save_barrier(self, barrier: Barrier) -> None:
        checked = _strict_model(barrier, Barrier, "barrier")
        _validate_barrier_definition(checked)

        def save(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT * FROM barrier_definitions WHERE barrier_id = ?",
                (checked.barrier_id,),
            ).fetchone()
            if row is not None:
                if self._barrier_from_row(row) != checked:
                    raise ValueError(
                        f"barrier already exists with different definition: {checked.barrier_id}"
                    )
                self._load_generic_barrier(connection, checked.barrier_id)
                return

            payload, digest = _model_payload(checked)
            connection.execute(
                """
                INSERT INTO barrier_definitions (
                    barrier_id, logical_key, expected_member_count, empty_semantics,
                    reduction_policy, status, required_success_count,
                    max_degraded_failures, payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.barrier_id,
                    checked.logical_key,
                    checked.expected_member_count,
                    checked.empty_semantics,
                    checked.reduction_policy,
                    checked.status,
                    checked.required_success_count,
                    checked.max_degraded_failures,
                    sqlite3.Binary(payload),
                    digest,
                ),
            )
            state = _derive_state(checked, ())
            state_payload, state_digest = _model_payload(state)
            connection.execute(
                """
                INSERT INTO barrier_states (
                    barrier_id, state_version, completed_members, pending_members,
                    failed_members, status, payload_json, payload_sha256
                ) VALUES (?, 0, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.barrier_id,
                    state.completed_members,
                    state.pending_members,
                    state.failed_members,
                    state.status,
                    sqlite3.Binary(state_payload),
                    state_digest,
                ),
            )

        self._transaction(write=True, operation_name="save_barrier", operation=save)

    def get_state(self, barrier_id: str) -> BarrierState | None:
        key = _nonempty_string(barrier_id, "barrier_id")

        def read(connection: sqlite3.Connection) -> BarrierState | None:
            row = connection.execute(
                "SELECT 1 FROM barrier_definitions WHERE barrier_id = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            _barrier, state, _members, _version = self._load_generic_barrier(connection, key)
            return state

        return self._transaction(write=False, operation_name="get_state", operation=read)

    def save_state(self, state: BarrierState) -> None:
        checked = _strict_model(state, BarrierState, "barrier state")

        def save(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT 1 FROM barrier_definitions WHERE barrier_id = ?",
                (checked.barrier_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown barrier: {checked.barrier_id}")
            _barrier, stored, _members, _version = self._load_generic_barrier(
                connection,
                checked.barrier_id,
            )
            if stored != checked:
                raise ValueError(
                    "barrier state is derived from its immutable definition and terminal members"
                )

        self._transaction(write=True, operation_name="save_state", operation=save)

    def add_member(
        self,
        barrier_id: str,
        work_item_id: str,
        outcome: StageStatus,
        criticality: DependencyCriticality = DependencyCriticality.REQUIRED,
    ) -> None:
        barrier_key = _nonempty_string(barrier_id, "barrier_id")
        work_key = _nonempty_string(work_item_id, "work_item_id")
        try:
            checked_outcome = StageStatus(outcome)
            checked_criticality = DependencyCriticality(criticality)
        except ValueError as exc:
            raise ValueError("invalid barrier member outcome or criticality") from exc
        if checked_outcome not in _TERMINAL_OUTCOMES:
            raise ValueError("barrier members must submit a terminal StageStatus")
        candidate = BarrierMember(
            work_item_id=work_key,
            criticality=checked_criticality,
            outcome=checked_outcome,
        )

        def add(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT 1 FROM barrier_definitions WHERE barrier_id = ?",
                (barrier_key,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown barrier: {barrier_key}")
            barrier, _state, members, state_version = self._load_generic_barrier(
                connection,
                barrier_key,
            )
            existing = next(
                (member for member in members if member.work_item_id == work_key),
                None,
            )
            if existing is not None:
                if existing == candidate:
                    return
                raise ValueError(f"conflicting replay for barrier member: {work_key}")
            if len(members) >= barrier.expected_member_count:
                raise ValueError(f"barrier member capacity exceeded: {barrier_key}")

            self._validate_call_member(connection, barrier_key, candidate)
            payload, digest = _model_payload(candidate)
            connection.execute(
                """
                INSERT INTO barrier_members (
                    barrier_id, work_item_id, criticality, outcome,
                    payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    barrier_key,
                    work_key,
                    checked_criticality.value,
                    checked_outcome.value,
                    sqlite3.Binary(payload),
                    digest,
                ),
            )
            updated_members = (*members, candidate)
            updated_state = _derive_state(barrier, updated_members)
            state_payload, state_digest = _model_payload(updated_state)
            cursor = connection.execute(
                """
                UPDATE barrier_states
                SET state_version = state_version + 1,
                    completed_members = ?, pending_members = ?,
                    failed_members = ?, status = ?,
                    payload_json = ?, payload_sha256 = ?
                WHERE barrier_id = ? AND state_version = ?
                """,
                (
                    updated_state.completed_members,
                    updated_state.pending_members,
                    updated_state.failed_members,
                    updated_state.status,
                    sqlite3.Binary(state_payload),
                    state_digest,
                    barrier_key,
                    state_version,
                ),
            )
            if cursor.rowcount != 1:
                raise SQLiteBarrierStorageError(
                    f"barrier state compare-and-swap did not match: {barrier_key}"
                )

        self._transaction(write=True, operation_name="add_member", operation=add)

    def get_members(self, barrier_id: str) -> tuple[BarrierMember, ...]:
        key = _nonempty_string(barrier_id, "barrier_id")

        def read(connection: sqlite3.Connection) -> tuple[BarrierMember, ...]:
            row = connection.execute(
                "SELECT 1 FROM barrier_definitions WHERE barrier_id = ?",
                (key,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown barrier: {key}")
            _barrier, _state, members, _version = self._load_generic_barrier(connection, key)
            return members

        return self._transaction(write=False, operation_name="get_members", operation=read)

    def get_persisted_barrier_snapshot(
        self,
        barrier_id: str,
        *,
        schema_ref: SchemaRef,
    ) -> PersistedBarrier | None:
        """Read one definition/state/member snapshot in one SQLite transaction.

        state_version lets consumers detect a concurrent member submission,
        so a mixed read can never be mistaken for durable barrier truth.
        """

        key = _nonempty_string(barrier_id, "barrier_id")

        def read(connection: sqlite3.Connection) -> PersistedBarrier | None:
            row = connection.execute(
                "SELECT 1 FROM barrier_definitions WHERE barrier_id = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            barrier, state, members, state_version = self._load_generic_barrier(
                connection,
                key,
            )
            snapshot = PersistedBarrier.from_snapshot(
                barrier,
                state,
                members,
                state_version,
                schema_ref=schema_ref,
            )
            return validate_registered_persisted_barrier(snapshot)

        return self._transaction(
            write=False,
            operation_name="get_persisted_barrier_snapshot",
            operation=read,
        )

    def append_definition(
        self,
        definition: InferenceCallBarrierDefinition,
    ) -> InferenceCallBarrierDefinition:
        checked = _strict_model(
            definition,
            InferenceCallBarrierDefinition,
            "inference call barrier definition",
        )

        def append(connection: sqlite3.Connection) -> InferenceCallBarrierDefinition:
            barrier_row = connection.execute(
                "SELECT * FROM barrier_definitions WHERE barrier_id = ?",
                (checked.barrier_id,),
            ).fetchone()
            if barrier_row is None:
                raise KeyError(f"unknown inference call barrier: {checked.barrier_id}")
            barrier = self._barrier_from_row(barrier_row)
            _validate_call_definition_binding(barrier, checked)

            row = connection.execute(
                """
                SELECT * FROM inference_call_barrier_definitions
                WHERE barrier_id = ?
                """,
                (checked.barrier_id,),
            ).fetchone()
            if row is not None:
                stored = self._call_definition_from_row(row)
                if stored != checked:
                    raise InferenceCallBarrierConflictError(
                        f"conflicting call barrier definition: {checked.barrier_id}"
                    )
                return stored

            payload, digest = _model_payload(checked)
            connection.execute(
                """
                INSERT INTO inference_call_barrier_definitions (
                    barrier_id, barrier_semantic_sha256, barrier_logical_key,
                    input_plan_semantic_sha256, call_plan_sha256, part_count,
                    reduction_policy, reduction_policy_version,
                    payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.barrier_id,
                    checked.barrier_semantic_sha256,
                    checked.barrier_logical_key,
                    checked.input_plan_semantic_sha256,
                    checked.call_plan_sha256,
                    checked.part_count,
                    checked.reduction_policy,
                    checked.reduction_policy_version,
                    sqlite3.Binary(payload),
                    digest,
                ),
            )
            return checked

        return self._transaction(
            write=True,
            operation_name="append_definition",
            operation=append,
        )

    def get_definition(self, barrier_id: str) -> InferenceCallBarrierDefinition | None:
        key = _nonempty_string(barrier_id, "barrier_id")

        def read(connection: sqlite3.Connection) -> InferenceCallBarrierDefinition | None:
            row = connection.execute(
                """
                SELECT * FROM inference_call_barrier_definitions
                WHERE barrier_id = ?
                """,
                (key,),
            ).fetchone()
            if row is None:
                return None
            definition = self._call_definition_from_row(row)
            barrier_row = connection.execute(
                "SELECT * FROM barrier_definitions WHERE barrier_id = ?",
                (key,),
            ).fetchone()
            if barrier_row is None:
                raise SQLiteBarrierStorageError(
                    "persisted inference call barrier has no generic definition"
                )
            _validate_call_definition_binding(self._barrier_from_row(barrier_row), definition)
            return definition

        return self._transaction(write=False, operation_name="get_definition", operation=read)

    def append_completion(
        self,
        completion: InferenceCallPartCompletion,
    ) -> InferenceCallPartCompletion:
        checked = _strict_model(
            completion,
            InferenceCallPartCompletion,
            "inference call part completion",
        )

        def append(connection: sqlite3.Connection) -> InferenceCallPartCompletion:
            definition = self._require_call_definition(connection, checked.barrier_id)
            _validate_completion_binding(definition, checked)
            row = connection.execute(
                """
                SELECT * FROM inference_call_part_completions
                WHERE barrier_id = ? AND part_ordinal = ?
                """,
                (checked.barrier_id, checked.part_ordinal),
            ).fetchone()
            if row is not None:
                stored = self._completion_from_row(row)
                if stored != checked:
                    raise InferenceCallBarrierConflictError(
                        "call part already has a different final completion: "
                        f"{checked.part_ordinal}"
                    )
                return stored

            semantic_row = connection.execute(
                """
                SELECT * FROM inference_call_part_completions
                WHERE barrier_id = ? AND part_semantic_sha256 = ?
                """,
                (checked.barrier_id, checked.part_semantic_sha256),
            ).fetchone()
            if semantic_row is not None:
                raise InferenceCallBarrierConflictError(
                    "call part semantic identity is already bound to another ordinal"
                )
            identity_row = connection.execute(
                """
                SELECT * FROM inference_call_part_completions
                WHERE completion_id = ?
                """,
                (checked.completion_id,),
            ).fetchone()
            if identity_row is not None:
                raise InferenceCallBarrierConflictError(
                    "call completion identity is already bound to another member"
                )

            payload, digest = _model_payload(checked)
            connection.execute(
                """
                INSERT INTO inference_call_part_completions (
                    barrier_id, part_ordinal, part_count, part_semantic_sha256,
                    part_logical_key, part_idempotency_key, completion_id,
                    completion_semantic_sha256, inference_id, logical_invocation_id,
                    status, payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.barrier_id,
                    checked.part_ordinal,
                    checked.part_count,
                    checked.part_semantic_sha256,
                    checked.part_logical_key,
                    checked.part_idempotency_key,
                    checked.completion_id,
                    checked.completion_semantic_sha256,
                    checked.inference_id,
                    checked.logical_invocation_id,
                    checked.status.value,
                    sqlite3.Binary(payload),
                    digest,
                ),
            )
            return checked

        return self._transaction(
            write=True,
            operation_name="append_completion",
            operation=append,
        )

    def list_completions(
        self,
        barrier_id: str,
    ) -> tuple[InferenceCallPartCompletion, ...]:
        key = _nonempty_string(barrier_id, "barrier_id")

        def read(connection: sqlite3.Connection) -> tuple[InferenceCallPartCompletion, ...]:
            definition = self._require_call_definition(connection, key)
            completions = tuple(
                self._completion_from_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM inference_call_part_completions
                    WHERE barrier_id = ? ORDER BY part_ordinal
                    """,
                    (key,),
                ).fetchall()
            )
            for item in completions:
                _validate_completion_binding(definition, item)
            return completions

        return self._transaction(
            write=False,
            operation_name="list_completions",
            operation=read,
        )

    def append_reduction(
        self,
        reduction: InferenceCallReduction,
    ) -> InferenceCallReduction:
        checked = _strict_model(
            reduction,
            InferenceCallReduction,
            "inference call reduction",
        )

        def append(connection: sqlite3.Connection) -> InferenceCallReduction:
            definition = self._require_call_definition(connection, checked.barrier_id)
            completions = self._call_completions(connection, definition)
            self._validate_reduction(connection, definition, completions, checked)
            row = connection.execute(
                "SELECT * FROM inference_call_reductions WHERE barrier_id = ?",
                (checked.barrier_id,),
            ).fetchone()
            if row is not None:
                stored = self._reduction_from_row(row)
                if stored != checked:
                    raise InferenceCallBarrierConflictError(
                        f"call barrier already has a different reduction: {checked.barrier_id}"
                    )
                return stored
            identity_row = connection.execute(
                "SELECT * FROM inference_call_reductions WHERE reduction_id = ?",
                (checked.reduction_id,),
            ).fetchone()
            if identity_row is not None:
                raise InferenceCallBarrierConflictError(
                    "call reduction identity is already bound to another barrier"
                )

            payload, digest = _model_payload(checked)
            connection.execute(
                """
                INSERT INTO inference_call_reductions (
                    barrier_id, reduction_id, reduction_semantic_sha256,
                    normalized_output_sha256, payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.barrier_id,
                    checked.reduction_id,
                    checked.reduction_semantic_sha256,
                    checked.normalized_output_sha256,
                    sqlite3.Binary(payload),
                    digest,
                ),
            )
            return checked

        return self._transaction(
            write=True,
            operation_name="append_reduction",
            operation=append,
        )

    def get_reduction(self, barrier_id: str) -> InferenceCallReduction | None:
        key = _nonempty_string(barrier_id, "barrier_id")

        def read(connection: sqlite3.Connection) -> InferenceCallReduction | None:
            row = connection.execute(
                "SELECT * FROM inference_call_reductions WHERE barrier_id = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            definition = self._require_call_definition(connection, key)
            completions = self._call_completions(connection, definition)
            reduction = self._reduction_from_row(row)
            self._validate_reduction(connection, definition, completions, reduction)
            return reduction

        return self._transaction(write=False, operation_name="get_reduction", operation=read)

    def _load_generic_barrier(
        self,
        connection: sqlite3.Connection,
        barrier_id: str,
    ) -> tuple[Barrier, BarrierState, tuple[BarrierMember, ...], int]:
        barrier_row = connection.execute(
            "SELECT * FROM barrier_definitions WHERE barrier_id = ?",
            (barrier_id,),
        ).fetchone()
        if barrier_row is None:
            raise KeyError(f"unknown barrier: {barrier_id}")
        state_row = connection.execute(
            "SELECT * FROM barrier_states WHERE barrier_id = ?",
            (barrier_id,),
        ).fetchone()
        if state_row is None:
            raise SQLiteBarrierStorageError(f"persisted barrier has no durable state: {barrier_id}")
        barrier = self._barrier_from_row(barrier_row)
        state, state_version = self._state_from_row(state_row)
        members = tuple(
            self._member_from_row(row)
            for row in connection.execute(
                """
                SELECT * FROM barrier_members
                WHERE barrier_id = ? ORDER BY work_item_id
                """,
                (barrier_id,),
            ).fetchall()
        )
        if len(members) > barrier.expected_member_count:
            raise SQLiteBarrierStorageError(
                f"persisted barrier member capacity is exceeded: {barrier_id}"
            )
        derived = _derive_state(barrier, members)
        if state != derived:
            raise SQLiteBarrierStorageError(
                f"persisted barrier state disagrees with its terminal members: {barrier_id}"
            )
        return barrier, state, members, state_version

    def _validate_call_member(
        self,
        connection: sqlite3.Connection,
        barrier_id: str,
        member: BarrierMember,
    ) -> None:
        definition_row = connection.execute(
            """
            SELECT * FROM inference_call_barrier_definitions
            WHERE barrier_id = ?
            """,
            (barrier_id,),
        ).fetchone()
        if definition_row is None:
            return
        definition = self._call_definition_from_row(definition_row)
        if member.work_item_id not in definition.expected_part_logical_keys:
            raise ValueError(
                "inference call barrier member is not declared by its immutable definition"
            )
        completion_row = connection.execute(
            """
            SELECT * FROM inference_call_part_completions
            WHERE barrier_id = ? AND part_logical_key = ?
            """,
            (barrier_id, member.work_item_id),
        ).fetchone()
        if completion_row is None:
            raise ValueError("inference call barrier member requires its durable part completion")
        completion = self._completion_from_row(completion_row)
        _validate_completion_binding(definition, completion)
        if member.outcome is not _stage_status(completion.status):
            raise ValueError(
                "inference call barrier member outcome disagrees with its part completion"
            )

    def _require_call_definition(
        self,
        connection: sqlite3.Connection,
        barrier_id: str,
    ) -> InferenceCallBarrierDefinition:
        row = connection.execute(
            """
            SELECT * FROM inference_call_barrier_definitions
            WHERE barrier_id = ?
            """,
            (barrier_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown inference call barrier: {barrier_id}")
        definition = self._call_definition_from_row(row)
        barrier_row = connection.execute(
            "SELECT * FROM barrier_definitions WHERE barrier_id = ?",
            (barrier_id,),
        ).fetchone()
        if barrier_row is None:
            raise SQLiteBarrierStorageError(
                "persisted inference call barrier has no generic definition"
            )
        _validate_call_definition_binding(self._barrier_from_row(barrier_row), definition)
        return definition

    def _call_completions(
        self,
        connection: sqlite3.Connection,
        definition: InferenceCallBarrierDefinition,
    ) -> tuple[InferenceCallPartCompletion, ...]:
        completions = tuple(
            self._completion_from_row(row)
            for row in connection.execute(
                """
                SELECT * FROM inference_call_part_completions
                WHERE barrier_id = ? ORDER BY part_ordinal
                """,
                (definition.barrier_id,),
            ).fetchall()
        )
        for item in completions:
            _validate_completion_binding(definition, item)
        return completions

    def _validate_reduction(
        self,
        connection: sqlite3.Connection,
        definition: InferenceCallBarrierDefinition,
        completions: tuple[InferenceCallPartCompletion, ...],
        reduction: InferenceCallReduction,
    ) -> None:
        if len(completions) != definition.part_count or tuple(
            item.part_ordinal for item in completions
        ) != tuple(range(definition.part_count)):
            raise InferenceCallBarrierConflictError(
                "reduction requires exact contiguous call-part completion coverage"
            )
        if any(item.status is not InferenceStatus.SUCCEEDED for item in completions):
            raise InferenceCallBarrierConflictError(
                "reduction cannot include a failed call-part completion"
            )
        _barrier, state, members, _version = self._load_generic_barrier(
            connection,
            definition.barrier_id,
        )
        if state.status != "CLOSED" or len(members) != definition.part_count:
            raise InferenceCallBarrierConflictError(
                "reduction requires a successfully closed generic barrier"
            )
        output_digests = tuple(item.normalized_output_sha256 for item in completions)
        selection_keys = tuple(item.selection_decision_logical_key for item in completions)
        if any(item is None for item in output_digests) or any(
            item is None for item in selection_keys
        ):
            raise InferenceCallBarrierConflictError(
                "successful call completions lack required output or selection identities"
            )
        expected = (
            reduction.barrier_id == definition.barrier_id
            and reduction.barrier_semantic_sha256 == definition.barrier_semantic_sha256
            and reduction.input_plan_semantic_sha256 == definition.input_plan_semantic_sha256
            and reduction.call_plan_sha256 == definition.call_plan_sha256
            and reduction.reduction_policy == definition.reduction_policy
            and reduction.reduction_policy_version == definition.reduction_policy_version
            and reduction.ordered_completion_ids
            == tuple(item.completion_id for item in completions)
            and reduction.ordered_part_semantic_sha256s
            == tuple(item.part_semantic_sha256 for item in completions)
            and reduction.ordered_normalized_output_sha256s == output_digests
            and reduction.ordered_selection_decision_logical_keys == selection_keys
        )
        if not expected:
            raise InferenceCallBarrierConflictError(
                "reduction does not match the completed declared member set"
            )

    def _barrier_from_row(self, row: sqlite3.Row) -> Barrier:
        value = _decode_model(row, Barrier, "barrier definition")
        _require_columns(
            row,
            (
                ("barrier_id", value.barrier_id),
                ("logical_key", value.logical_key),
                ("expected_member_count", value.expected_member_count),
                ("empty_semantics", value.empty_semantics),
                ("reduction_policy", value.reduction_policy),
                ("status", value.status),
                ("required_success_count", value.required_success_count),
                ("max_degraded_failures", value.max_degraded_failures),
            ),
            "barrier definition",
        )
        try:
            _validate_barrier_definition(value)
        except ValueError as exc:
            raise SQLiteBarrierStorageError(
                "persisted barrier definition is semantically invalid"
            ) from exc
        return value

    def _state_from_row(self, row: sqlite3.Row) -> tuple[BarrierState, int]:
        value = _decode_model(row, BarrierState, "barrier state")
        _require_columns(
            row,
            (
                ("barrier_id", value.barrier_id),
                ("completed_members", value.completed_members),
                ("pending_members", value.pending_members),
                ("failed_members", value.failed_members),
                ("status", value.status),
            ),
            "barrier state",
        )
        return value, _row_int(row, "state_version")

    def _member_from_row(self, row: sqlite3.Row) -> BarrierMember:
        value = _decode_model(row, BarrierMember, "barrier member")
        _require_columns(
            row,
            (
                ("work_item_id", value.work_item_id),
                ("criticality", value.criticality.value),
                ("outcome", value.outcome.value),
            ),
            "barrier member",
        )
        return value

    def _call_definition_from_row(
        self,
        row: sqlite3.Row,
    ) -> InferenceCallBarrierDefinition:
        value = _decode_model(
            row,
            InferenceCallBarrierDefinition,
            "inference call barrier definition",
        )
        _require_columns(
            row,
            (
                ("barrier_id", value.barrier_id),
                ("barrier_semantic_sha256", value.barrier_semantic_sha256),
                ("barrier_logical_key", value.barrier_logical_key),
                ("input_plan_semantic_sha256", value.input_plan_semantic_sha256),
                ("call_plan_sha256", value.call_plan_sha256),
                ("part_count", value.part_count),
                ("reduction_policy", value.reduction_policy),
                ("reduction_policy_version", value.reduction_policy_version),
            ),
            "inference call barrier definition",
        )
        return value

    def _completion_from_row(self, row: sqlite3.Row) -> InferenceCallPartCompletion:
        value = _decode_model(
            row,
            InferenceCallPartCompletion,
            "inference call part completion",
        )
        _require_columns(
            row,
            (
                ("barrier_id", value.barrier_id),
                ("part_ordinal", value.part_ordinal),
                ("part_count", value.part_count),
                ("part_semantic_sha256", value.part_semantic_sha256),
                ("part_logical_key", value.part_logical_key),
                ("part_idempotency_key", value.part_idempotency_key),
                ("completion_id", value.completion_id),
                ("completion_semantic_sha256", value.completion_semantic_sha256),
                ("inference_id", value.inference_id),
                ("logical_invocation_id", value.logical_invocation_id),
                ("status", value.status.value),
            ),
            "inference call part completion",
        )
        return value

    def _reduction_from_row(self, row: sqlite3.Row) -> InferenceCallReduction:
        value = _decode_model(row, InferenceCallReduction, "inference call reduction")
        _require_columns(
            row,
            (
                ("barrier_id", value.barrier_id),
                ("reduction_id", value.reduction_id),
                ("reduction_semantic_sha256", value.reduction_semantic_sha256),
                ("normalized_output_sha256", value.normalized_output_sha256),
            ),
            "inference call reduction",
        )
        return value

    def _initialize_database(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._open()
            with self._observed_transaction_scope(
                connection,
                write=False,
                operation_name="initialize_preflight",
            ):
                preflight_version = _pragma_int(connection, "user_version")
                preflight_application_id = _pragma_int(connection, "application_id")
                preflight_has_schema = _has_user_schema(connection)
                if preflight_version == 0:
                    if preflight_application_id != 0 or preflight_has_schema:
                        raise SQLiteBarrierStorageError(
                            "refusing to adopt a nonempty or claimed unversioned barrier database"
                        )
                elif (
                    preflight_version != _SCHEMA_VERSION
                    or preflight_application_id != _APPLICATION_ID
                ):
                    raise SQLiteBarrierStorageError(
                        "barrier database header belongs to another schema version"
                    )

            journal = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if journal is None or not isinstance(journal[0], str) or journal[0].lower() != "wal":
                raise SQLiteBarrierStorageError("SQLite WAL mode could not be enabled")

            with self._observed_transaction_scope(
                connection,
                write=True,
                operation_name="initialize_schema",
            ):
                user_version = _pragma_int(connection, "user_version")
                application_id = _pragma_int(connection, "application_id")
                has_schema = _has_user_schema(connection)
                if user_version == 0:
                    if application_id != 0 or has_schema:
                        raise SQLiteBarrierStorageError(
                            "refusing to adopt a nonempty or claimed unversioned barrier database"
                        )
                    for statement in _SCHEMA_STATEMENTS:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                elif user_version != _SCHEMA_VERSION or application_id != _APPLICATION_ID:
                    raise SQLiteBarrierStorageError(
                        "barrier database header belongs to another schema version"
                    )
                self._verify_database(connection)
        except SQLiteBarrierStorageError:
            if connection is not None:
                _rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                _rollback_quietly(connection)
            raise SQLiteBarrierStorageError(
                f"cannot initialize SQLite barrier storage: {exc}"
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def _open(self) -> sqlite3.Connection:
        if self._database_path.is_symlink():
            raise SQLiteBarrierStorageError(
                f"barrier database became a symlink: {self._database_path}"
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
                raise sqlite3.OperationalError("foreign keys are disabled")
            if _pragma_int(connection, "recursive_triggers") != 1:
                raise sqlite3.OperationalError("recursive triggers are disabled")
            if _pragma_int(connection, "synchronous") != 2:
                raise sqlite3.OperationalError("FULL synchronous mode is disabled")
            if _pragma_int(connection, "busy_timeout") != _BUSY_TIMEOUT_MS:
                raise sqlite3.OperationalError("busy timeout is inconsistent")
            if _pragma_int(connection, "trusted_schema") != 0:
                raise sqlite3.OperationalError("trusted schema mode is enabled")
            return connection
        except sqlite3.Error as exc:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise SQLiteBarrierStorageError(f"cannot open SQLite barrier storage: {exc}") from exc

    def _transaction(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[sqlite3.Connection], _ResultT],
    ) -> _ResultT:
        connection = self._open()
        try:
            with self._observed_transaction_scope(
                connection,
                write=write,
                operation_name=operation_name,
            ):
                self._verify_header(connection)
                result = operation(connection)
            return result
        except (InferenceCallBarrierError, KeyError, TypeError, ValueError):
            raise
        except sqlite3.IntegrityError as exc:
            raise SQLiteBarrierStorageError(f"SQLite rejected barrier persistence: {exc}") from exc
        except sqlite3.Error as exc:
            raise SQLiteBarrierStorageError(f"barrier storage transaction failed: {exc}") from exc
        finally:
            connection.close()

    @contextmanager
    def _observed_transaction_scope(
        self,
        connection: sqlite3.Connection,
        *,
        write: bool,
        operation_name: str,
    ) -> Iterator[None]:
        attributes: dict[str, RuntimeAttributeValue] = {
            "operation": operation_name,
            "write": write,
        }
        with runtime_span(
            self._runtime_observer,
            "sqlite.barrier.transaction",
            attributes,
        ):
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            runtime_increment(
                self._runtime_observer,
                "sqlite.barrier.transactions",
                attributes=attributes,
            )
            try:
                yield
            except BaseException:
                if connection.in_transaction:
                    try:
                        connection.rollback()
                    except sqlite3.Error:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.barrier.rollback_failures",
                            attributes=attributes,
                        )
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.barrier.transaction_outcomes_unknown",
                            attributes=attributes,
                        )
                    else:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.barrier.rollbacks",
                            attributes=attributes,
                        )
                else:
                    runtime_increment(
                        self._runtime_observer,
                        "sqlite.barrier.transaction_outcomes_unknown",
                        attributes=attributes,
                    )
                raise
            try:
                connection.commit()
            except BaseException:
                runtime_increment(
                    self._runtime_observer,
                    "sqlite.barrier.commit_failures",
                    attributes=attributes,
                )
                if connection.in_transaction:
                    try:
                        connection.rollback()
                    except sqlite3.Error:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.barrier.rollback_failures",
                            attributes=attributes,
                        )
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.barrier.transaction_outcomes_unknown",
                            attributes=attributes,
                        )
                    else:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.barrier.rollbacks",
                            attributes=attributes,
                        )
                else:
                    runtime_increment(
                        self._runtime_observer,
                        "sqlite.barrier.transaction_outcomes_unknown",
                        attributes=attributes,
                    )
                raise
            runtime_increment(
                self._runtime_observer,
                "sqlite.barrier.commits",
                attributes=attributes,
            )

    def _verify_database(self, connection: sqlite3.Connection) -> None:
        self._verify_header(connection)
        journal = connection.execute("PRAGMA journal_mode").fetchone()
        if journal is None or not isinstance(journal[0], str) or journal[0].lower() != "wal":
            raise SQLiteBarrierStorageError("barrier database is not in WAL mode")
        if _database_schema_fingerprint(connection) != _expected_schema_fingerprint():
            raise SQLiteBarrierStorageError(
                "SQLite barrier DDL does not match the canonical schema"
            )
        quick = connection.execute("PRAGMA quick_check(1)").fetchone()
        if quick is None or quick[0] != "ok":
            raise SQLiteBarrierStorageError(
                f"SQLite quick_check failed for barrier storage: {quick!r}"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise SQLiteBarrierStorageError("SQLite foreign-key check found orphaned barrier rows")

    def _verify_header(self, connection: sqlite3.Connection) -> None:
        if _pragma_int(connection, "application_id") != _APPLICATION_ID:
            raise SQLiteBarrierStorageError("barrier database application identity changed")
        if _pragma_int(connection, "user_version") != _SCHEMA_VERSION:
            raise SQLiteBarrierStorageError("barrier database schema version changed")


def _strict_model[ModelT: BaseModel](
    value: object,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    try:
        return model_type.model_validate(value.model_dump(mode="python"), strict=True)
    except ValidationError as exc:
        raise ValueError(f"{label} failed strict validation") from exc


def _model_payload(value: BaseModel) -> tuple[bytes, str]:
    try:
        payload = canonical_json_bytes(value)
    except (CanonicalizationError, TypeError, ValueError) as exc:
        raise ValueError(f"{type(value).__name__} is not canonical JSON") from exc
    return payload, exact_bytes_sha256(payload)


def _decode_model[ModelT: BaseModel](
    row: sqlite3.Row,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    raw = _row_bytes(row, "payload_json")
    if _row_text(row, "payload_sha256") != exact_bytes_sha256(raw):
        raise SQLiteBarrierStorageError(f"persisted {label} exact digest is corrupt")
    try:
        value = model_type.model_validate_json(raw, strict=True)
        if canonical_json_bytes(value) != raw:
            raise SQLiteBarrierStorageError(f"persisted {label} is not canonical JSON")
        return value
    except SQLiteBarrierStorageError:
        raise
    except (CanonicalizationError, ValidationError, TypeError, ValueError) as exc:
        raise SQLiteBarrierStorageError(f"persisted {label} failed strict validation") from exc


def _validate_barrier_definition(barrier: Barrier) -> None:
    try:
        empty_outcome = StageStatus(barrier.empty_semantics)
    except ValueError as exc:
        raise ValueError("empty_semantics must identify a terminal StageStatus") from exc
    if empty_outcome not in _TERMINAL_OUTCOMES:
        raise ValueError("empty_semantics must identify a terminal StageStatus")
    if barrier.required_success_count > barrier.expected_member_count:
        raise ValueError("required success count cannot exceed expected members")
    if barrier.max_degraded_failures > barrier.expected_member_count:
        raise ValueError("degradable failures cannot exceed expected members")
    expected_status = (
        "FAILED"
        if barrier.expected_member_count == 0 and empty_outcome in _FAILURE_OUTCOMES
        else "CLOSED"
        if barrier.expected_member_count == 0
        else "OPEN"
    )
    if barrier.status != expected_status:
        raise ValueError("barrier initial status is inconsistent with its member count")


def _derive_state(
    barrier: Barrier,
    members: tuple[BarrierMember, ...],
) -> BarrierState:
    completed = len(members)
    failed = sum(member.outcome in _FAILURE_OUTCOMES for member in members)
    if completed > barrier.expected_member_count:
        raise SQLiteBarrierStorageError(
            f"persisted barrier has too many terminal members: {barrier.barrier_id}"
        )
    if barrier.expected_member_count == 0:
        status = barrier.status
    elif completed < barrier.expected_member_count:
        status = "OPEN"
    else:
        successful = sum(member.outcome in _SUCCESS_OUTCOMES for member in members)
        required_failed = any(
            member.criticality is DependencyCriticality.REQUIRED
            and member.outcome not in _SUCCESS_OUTCOMES
            for member in members
        )
        degraded = sum(
            member.criticality is DependencyCriticality.DEGRADABLE
            and member.outcome in _FAILURE_OUTCOMES
            for member in members
        )
        acceptable = (
            not required_failed
            and successful >= barrier.required_success_count
            and degraded <= barrier.max_degraded_failures
        )
        status = "CLOSED" if acceptable else "FAILED"
    return BarrierState(
        barrier_id=barrier.barrier_id,
        completed_members=completed,
        pending_members=barrier.expected_member_count - completed,
        failed_members=failed,
        status=status,
    )


def _validate_call_definition_binding(
    barrier: Barrier,
    definition: InferenceCallBarrierDefinition,
) -> None:
    if (
        barrier.barrier_id != definition.barrier_id
        or barrier.logical_key != definition.barrier_logical_key
        or barrier.expected_member_count != definition.part_count
        or barrier.required_success_count != definition.part_count
        or barrier.max_degraded_failures != 0
        or barrier.reduction_policy != definition.reduction_policy_version
    ):
        raise InferenceCallBarrierConflictError(
            "call barrier definition does not match its generic barrier"
        )


def _validate_completion_binding(
    definition: InferenceCallBarrierDefinition,
    completion: InferenceCallPartCompletion,
) -> None:
    ordinal = completion.part_ordinal
    if (
        ordinal >= definition.part_count
        or completion.barrier_id != definition.barrier_id
        or completion.part_count != definition.part_count
        or completion.barrier_semantic_sha256 != definition.barrier_semantic_sha256
        or completion.input_plan_semantic_sha256 != definition.input_plan_semantic_sha256
        or completion.call_plan_sha256 != definition.call_plan_sha256
        or completion.part_semantic_sha256 != definition.expected_part_semantic_sha256s[ordinal]
        or completion.part_logical_key != definition.expected_part_logical_keys[ordinal]
        or completion.part_idempotency_key != definition.expected_part_idempotency_keys[ordinal]
    ):
        raise InferenceCallBarrierConflictError(
            "completion does not match the declared call barrier member"
        )


def _stage_status(status: InferenceStatus) -> StageStatus:
    if status is InferenceStatus.SUCCEEDED:
        return StageStatus.SUCCEEDED
    if status is InferenceStatus.CANCELLED:
        return StageStatus.CANCELLED
    if status is InferenceStatus.INVALID_OUTPUT:
        return StageStatus.QUARANTINED
    return StageStatus.FAILED


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a nonempty string")
    return value


def _require_columns(
    row: sqlite3.Row,
    expected: tuple[tuple[str, object], ...],
    label: str,
) -> None:
    for column, value in expected:
        if row[column] != value:
            raise SQLiteBarrierStorageError(
                f"persisted {label} indexed column {column} disagrees with canonical JSON"
            )


def _row_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise SQLiteBarrierStorageError(f"persisted barrier column {column} must be text")
    return value


def _row_int(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SQLiteBarrierStorageError(f"persisted barrier column {column} must be an integer")
    return value


def _row_bytes(row: sqlite3.Row, column: str) -> bytes:
    value: object = row[column]
    if not isinstance(value, bytes):
        raise SQLiteBarrierStorageError(f"persisted barrier column {column} must be bytes")
    return value


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    value: object = None if row is None else row[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SQLiteBarrierStorageError(f"SQLite PRAGMA {name} returned no integer")
    return value


def _has_user_schema(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'
        )
        """
    ).fetchone()
    value: object = None if row is None else row[0]
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise SQLiteBarrierStorageError("SQLite schema inventory returned no value")
    return value == 1


def _database_schema_fingerprint(
    connection: sqlite3.Connection,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    )


@cache
def _expected_schema_fingerprint() -> tuple[tuple[object, ...], ...]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        return _database_schema_fingerprint(connection)
    finally:
        connection.close()


def _rollback_quietly(connection: sqlite3.Connection) -> None:
    with suppress(sqlite3.Error):
        connection.rollback()


__all__ = [
    "SQLiteBarrierStorage",
    "SQLiteBarrierStorageError",
]
