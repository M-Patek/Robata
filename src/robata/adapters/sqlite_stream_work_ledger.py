"""SQLite persistence for the local pre-EOS stream composition.

The adapter stores exact canonical JSON bytes and publication bookkeeping. It
does not interpret stream identities or project them into executable work.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler, WorkFenceError

_EXTENSION_NAME = "stream-work-ledger"
_EXTENSION_SCHEMA_VERSION = 2
_EXTENSION_OBJECT_NAMES = frozenset(
    {
        "stream_plans",
        "expected_windows",
        "stream_work_plans",
        "stream_work_plan_order",
    }
)
_EXTENSION_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE stream_plans (
        plan_key TEXT PRIMARY KEY,
        plan_json BLOB NOT NULL,
        source_subject_json BLOB NOT NULL,
        composition_config_json BLOB NOT NULL,
        planner_eos_sha256 TEXT,
        seal_json BLOB,
        terminal_closure_json BLOB,
        export_manifest_sha256 TEXT,
        export_member_count INTEGER
    )
    """,
    """
    CREATE TABLE expected_windows (
        plan_key TEXT NOT NULL REFERENCES stream_plans(plan_key),
        ordinal INTEGER NOT NULL,
        declaration_json BLOB NOT NULL,
        window_json BLOB NOT NULL,
        terminal_member_json BLOB,
        PRIMARY KEY (plan_key, ordinal)
    )
    """,
    """
    CREATE TABLE stream_work_plans (
        work_item_id TEXT PRIMARY KEY,
        work_logical_key TEXT NOT NULL UNIQUE,
        plan_key TEXT NOT NULL REFERENCES stream_plans(plan_key),
        expected_ordinal INTEGER,
        role_order INTEGER NOT NULL,
        stage TEXT NOT NULL,
        plan_json BLOB NOT NULL,
        publication_state TEXT NOT NULL
            CHECK (publication_state IN ('GATED', 'PENDING', 'PUBLISHED')),
        terminal_evidence_json BLOB,
        pending_terminal_json BLOB,
        pending_lease_epoch INTEGER,
        pending_fencing_token TEXT
    )
    """,
    """
    CREATE INDEX stream_work_plan_order
    ON stream_work_plans(plan_key, expected_ordinal, role_order)
    """,
)
_ADDITIVE_INDEX_STATEMENTS = (
    """
    CREATE INDEX IF NOT EXISTS stream_work_pending_terminal
    ON stream_work_plans(plan_key, work_item_id)
    WHERE pending_terminal_json IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS stream_work_pending_publication
    ON stream_work_plans(plan_key, expected_ordinal, role_order, work_item_id)
    WHERE publication_state = 'PENDING'
    """,
    """
    CREATE INDEX IF NOT EXISTS stream_work_published_lookup
    ON stream_work_plans(plan_key, work_item_id)
    WHERE publication_state = 'PUBLISHED'
    """,
)
_EXTENSION_REQUIRED_COLUMNS = {
    "stream_plans": frozenset(
        {
            "plan_key",
            "plan_json",
            "source_subject_json",
            "composition_config_json",
            "planner_eos_sha256",
            "seal_json",
            "terminal_closure_json",
            "export_manifest_sha256",
            "export_member_count",
        }
    ),
}
_EXTENSION_REQUIRED_COLUMNS["expected_windows"] = frozenset(
    {
        "plan_key",
        "ordinal",
        "declaration_json",
        "window_json",
        "terminal_member_json",
    }
)
_EXTENSION_REQUIRED_COLUMNS["stream_work_plans"] = frozenset(
    {
        "work_item_id",
        "work_logical_key",
        "plan_key",
        "expected_ordinal",
        "role_order",
        "stage",
        "plan_json",
        "publication_state",
        "terminal_evidence_json",
        "pending_terminal_json",
        "pending_lease_epoch",
        "pending_fencing_token",
    }
)
_V1_TO_V2_PENDING_COLUMNS = (
    ("pending_terminal_json", "BLOB"),
    ("pending_lease_epoch", "INTEGER"),
    ("pending_fencing_token", "TEXT"),
)


class SQLiteStreamWorkLedgerError(RuntimeError):
    """The local stream ledger cannot preserve its durable contract."""


class SQLiteStreamWorkLedgerConflict(SQLiteStreamWorkLedgerError):
    """An exact replay differs from already-persisted stream state."""


@dataclass(frozen=True, slots=True)
class StoredStreamPlan:
    plan_key: str
    plan_json: bytes
    source_subject_json: bytes
    composition_config_json: bytes
    planner_eos_sha256: str | None
    seal_json: bytes | None
    terminal_closure_json: bytes | None
    export_manifest_sha256: str | None
    export_member_count: int | None


@dataclass(frozen=True, slots=True)
class StoredExpectedWindow:
    plan_key: str
    ordinal: int
    declaration_json: bytes
    window_json: bytes
    terminal_member_json: bytes | None


@dataclass(frozen=True, slots=True)
class StoredStreamWorkPlan:
    work_item_id: str
    work_logical_key: str
    plan_key: str
    expected_ordinal: int | None
    role_order: int
    stage: str
    plan_json: bytes
    publication_state: str

    terminal_evidence_json: bytes | None = None
    pending_terminal_json: bytes | None = None
    pending_lease_epoch: int | None = None
    pending_fencing_token: str | None = None


@dataclass(frozen=True, slots=True)
class StoredStreamWorkExecution:
    """One bounded stream-plan row joined to its execution state."""

    work: StoredStreamWorkPlan
    execution_state: str
    execution_created_at: str


@dataclass(frozen=True, slots=True)
class StoredStreamBacklog:
    """SQL-side aggregate used by the streaming hot path."""

    state_counts: tuple[tuple[str, int], ...]
    active_backlog: int
    oldest_active_created_at: str | None
    declared_window_count: int
    expected_plan_sealed: bool
    terminal_member_count: int
    export_manifest_sha256: str | None
    export_member_count: int | None
    finalization_published: bool


@dataclass(frozen=True, slots=True)
class NewStreamWorkPlan:
    work_item_id: str
    work_logical_key: str
    expected_ordinal: int | None
    role_order: int
    stage: str
    plan_json: bytes
    publication_state: str


@dataclass(frozen=True, slots=True)
class NewStreamWindow:
    """One immutable expected-window append and its child work plans."""

    ordinal: int
    declaration_json: bytes
    window_json: bytes
    work_plans: tuple[NewStreamWorkPlan, ...]


class SQLiteStreamWorkLedger:
    """Exact-byte ledger for declarations, typed plans, seals, and closures."""

    def __init__(self, authority: SQLiteWorkScheduler) -> None:
        if not isinstance(authority, SQLiteWorkScheduler):
            raise TypeError("authority must be SQLiteWorkScheduler")
        self._authority = authority
        self._database_path = authority.database_path
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def register_plan(
        self,
        *,
        plan_key: str,
        plan_json: bytes,
        source_subject_json: bytes,
        composition_config_json: bytes,
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT * FROM stream_plans WHERE plan_key = ?", (plan_key,)
            ).fetchone()
            if row is not None:
                existing = _plan_from_row(row)
                if (
                    existing.plan_json != plan_json
                    or existing.source_subject_json != source_subject_json
                    or existing.composition_config_json != composition_config_json
                ):
                    raise SQLiteStreamWorkLedgerConflict(
                        "composition replay changed plan or policy pins"
                    )
                return
            connection.execute(
                """
                INSERT INTO stream_plans (
                    plan_key, plan_json, source_subject_json, composition_config_json,
                    planner_eos_sha256, seal_json, terminal_closure_json,
                    export_manifest_sha256, export_member_count
                ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
                """,
                (
                    plan_key,
                    sqlite3.Binary(plan_json),
                    sqlite3.Binary(source_subject_json),
                    sqlite3.Binary(composition_config_json),
                ),
            )

        self._run(write=True, operation_name="register_plan", operation=operation)

    def get_plan(self, plan_key: str) -> StoredStreamPlan:
        def operation(connection: sqlite3.Connection) -> StoredStreamPlan:
            row = connection.execute(
                "SELECT * FROM stream_plans WHERE plan_key = ?", (plan_key,)
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("expected plan is not registered")
            return _plan_from_row(row)

        return self._run(write=False, operation_name="get_plan", operation=operation)

    def plans(self) -> tuple[StoredStreamPlan, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[StoredStreamPlan, ...]:
            rows = connection.execute("SELECT * FROM stream_plans ORDER BY plan_key").fetchall()
            return tuple(_plan_from_row(row) for row in rows)

        return self._run(write=False, operation_name="plans", operation=operation)

    def append_windows(
        self,
        *,
        plan_key: str,
        windows: Sequence[NewStreamWindow],
    ) -> tuple[bool, ...]:
        """Atomically append a contiguous batch of windows.

        Existing members are checked as exact replays. New members are inserted in
        ordinal order and remain PENDING until the scheduler projects their
        execution plans, so a crash between those boundaries is recoverable.
        """

        if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence):
            raise TypeError("windows must be a sequence")
        checked = tuple(windows)
        if not checked:
            return ()
        for window in checked:
            if not isinstance(window, NewStreamWindow):
                raise TypeError("windows must contain NewStreamWindow values")
            if (
                isinstance(window.ordinal, bool)
                or not isinstance(window.ordinal, int)
                or window.ordinal < 0
            ):
                raise ValueError("window ordinal must be a nonnegative integer")
            if not isinstance(window.declaration_json, bytes) or not isinstance(
                window.window_json, bytes
            ):
                raise TypeError("window JSON payloads must be bytes")
            if any(work.expected_ordinal != window.ordinal for work in window.work_plans):
                raise SQLiteStreamWorkLedgerConflict(
                    "stream child work ordinal does not match its window"
                )
        ordinals = tuple(window.ordinal for window in checked)
        if len(set(ordinals)) != len(ordinals) or ordinals != tuple(sorted(ordinals)):
            raise SQLiteStreamWorkLedgerConflict("window batch ordinals must be unique and ordered")

        def operation(connection: sqlite3.Connection) -> tuple[bool, ...]:
            plan = connection.execute(
                """
                SELECT planner_eos_sha256, seal_json
                FROM stream_plans WHERE plan_key = ?
                """,
                (plan_key,),
            ).fetchone()
            if plan is None:
                raise SQLiteStreamWorkLedgerError("expected plan is not registered")
            tail = connection.execute(
                """
                SELECT ordinal FROM expected_windows
                WHERE plan_key = ?
                ORDER BY ordinal DESC
                LIMIT 1
                """,
                (plan_key,),
            ).fetchone()
            next_ordinal = 0 if tail is None else _int(tail, "ordinal") + 1
            inserted: list[bool] = []
            for window in checked:
                existing = connection.execute(
                    """
                    SELECT * FROM expected_windows
                    WHERE plan_key = ? AND ordinal = ?
                    """,
                    (plan_key, window.ordinal),
                ).fetchone()
                if existing is not None:
                    stored = _window_from_row(existing)
                    if (
                        stored.declaration_json != window.declaration_json
                        or stored.window_json != window.window_json
                    ):
                        raise SQLiteStreamWorkLedgerConflict(
                            "expected-window replay changed exact bytes"
                        )
                    _verify_existing_work_rows(
                        connection,
                        plan_key=plan_key,
                        expected_ordinal=window.ordinal,
                        expected=window.work_plans,
                    )
                    inserted.append(False)
                    continue
                if plan["planner_eos_sha256"] is not None or plan["seal_json"] is not None:
                    raise SQLiteStreamWorkLedgerConflict(
                        "cannot append a new window after planner EOS"
                    )
                if next_ordinal != window.ordinal:
                    raise SQLiteStreamWorkLedgerConflict(
                        "expected windows must be appended in contiguous planner order"
                    )
                connection.execute(
                    """
                    INSERT INTO expected_windows (
                        plan_key, ordinal, declaration_json, window_json, terminal_member_json
                    ) VALUES (?, ?, ?, ?, NULL)
                    """,
                    (
                        plan_key,
                        window.ordinal,
                        sqlite3.Binary(window.declaration_json),
                        sqlite3.Binary(window.window_json),
                    ),
                )
                for work in window.work_plans:
                    _insert_work(connection, plan_key, work)
                next_ordinal += 1
                inserted.append(True)
            return tuple(inserted)

        return self._run(write=True, operation_name="append_windows", operation=operation)

    def append_window(
        self,
        *,
        plan_key: str,
        ordinal: int,
        declaration_json: bytes,
        window_json: bytes,
        work_plans: Sequence[NewStreamWorkPlan],
    ) -> bool:
        """Atomically append one declaration and its children."""

        return self.append_windows(
            plan_key=plan_key,
            windows=(
                NewStreamWindow(
                    ordinal=ordinal,
                    declaration_json=declaration_json,
                    window_json=window_json,
                    work_plans=tuple(work_plans),
                ),
            ),
        )[0]

    def windows(self, plan_key: str) -> tuple[StoredExpectedWindow, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[StoredExpectedWindow, ...]:
            rows = connection.execute(
                """
                SELECT * FROM expected_windows WHERE plan_key = ? ORDER BY ordinal
                """,
                (plan_key,),
            ).fetchall()
            return tuple(_window_from_row(row) for row in rows)

        return self._run(write=False, operation_name="windows", operation=operation)

    def windows_for_ordinals(
        self,
        plan_key: str,
        ordinals: Sequence[int],
    ) -> tuple[StoredExpectedWindow, ...]:
        """Load only the expected-window rows needed for a bounded append/replay."""

        if isinstance(ordinals, (str, bytes)) or not isinstance(ordinals, Sequence):
            raise TypeError("ordinals must be a sequence")
        checked = tuple(ordinals)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in checked
        ):
            raise ValueError("ordinals must contain nonnegative integers")
        if len(set(checked)) != len(checked):
            raise ValueError("ordinals must be unique")
        if not checked:
            return ()
        placeholders = ", ".join("?" for _value in checked)

        def operation(connection: sqlite3.Connection) -> tuple[StoredExpectedWindow, ...]:
            rows = connection.execute(
                f"""
                SELECT * FROM expected_windows
                WHERE plan_key = ? AND ordinal IN ({placeholders})
                ORDER BY ordinal
                """,
                (plan_key, *checked),
            ).fetchall()
            return tuple(_window_from_row(row) for row in rows)

        return self._run(
            write=False,
            operation_name="windows_for_ordinals",
            operation=operation,
        )

    def window_count(self, plan_key: str) -> int:
        def operation(connection: sqlite3.Connection) -> int:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM expected_windows WHERE plan_key = ?",
                (plan_key,),
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("window count query returned no row")
            return _int(row, "count")

        return self._run(write=False, operation_name="window_count", operation=operation)

    def next_window_ordinal(self, plan_key: str) -> int:
        """Return the next append ordinal through the ordered window index."""

        def operation(connection: sqlite3.Connection) -> int:
            row = connection.execute(
                """
                SELECT ordinal FROM expected_windows
                WHERE plan_key = ?
                ORDER BY ordinal DESC
                LIMIT 1
                """,
                (plan_key,),
            ).fetchone()
            if row is None:
                return 0
            return _int(row, "ordinal") + 1

        return self._run(
            write=False,
            operation_name="next_window_ordinal",
            operation=operation,
        )

    def window_at(self, plan_key: str, ordinal: int) -> StoredExpectedWindow | None:
        def operation(connection: sqlite3.Connection) -> StoredExpectedWindow | None:
            row = connection.execute(
                """
                SELECT * FROM expected_windows WHERE plan_key = ? AND ordinal = ?
                """,
                (plan_key, ordinal),
            ).fetchone()
            return None if row is None else _window_from_row(row)

        return self._run(write=False, operation_name="window_at", operation=operation)

    def terminal_member_at(self, plan_key: str, ordinal: int) -> bytes | None:
        def operation(connection: sqlite3.Connection) -> bytes | None:
            row = connection.execute(
                """
                SELECT terminal_member_json FROM expected_windows
                WHERE plan_key = ? AND ordinal = ?
                """,
                (plan_key, ordinal),
            ).fetchone()
            return None if row is None else _optional_bytes(row, "terminal_member_json")

        return self._run(
            write=False,
            operation_name="terminal_member_at",
            operation=operation,
        )

    def terminal_member_count(self, plan_key: str) -> int:
        def operation(connection: sqlite3.Connection) -> int:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM expected_windows
                WHERE plan_key = ? AND terminal_member_json IS NOT NULL
                """,
                (plan_key,),
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("terminal member count query returned no row")
            return _int(row, "count")

        return self._run(
            write=False,
            operation_name="terminal_member_count",
            operation=operation,
        )

    def set_planner_eos(self, plan_key: str, finish_sha256: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT planner_eos_sha256 FROM stream_plans WHERE plan_key = ?", (plan_key,)
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("expected plan is not registered")
            current = cast(str | None, row["planner_eos_sha256"])
            if current is not None and current != finish_sha256:
                raise SQLiteStreamWorkLedgerConflict("planner EOS replay changed exact facts")
            connection.execute(
                "UPDATE stream_plans SET planner_eos_sha256 = ? WHERE plan_key = ?",
                (finish_sha256, plan_key),
            )

        self._run(write=True, operation_name="set_planner_eos", operation=operation)

    def store_seal_and_finalization(
        self,
        *,
        plan_key: str,
        seal_json: bytes,
        expected_declaration_jsons: Sequence[bytes],
        finalization: NewStreamWorkPlan,
    ) -> bool:
        def operation(connection: sqlite3.Connection) -> bool:
            row = connection.execute(
                "SELECT planner_eos_sha256, seal_json FROM stream_plans WHERE plan_key = ?",
                (plan_key,),
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("expected plan is not registered")
            if row["planner_eos_sha256"] is None:
                raise SQLiteStreamWorkLedgerConflict("planner EOS must be durable before seal")
            declaration_rows = connection.execute(
                """
                SELECT ordinal, declaration_json FROM expected_windows
                WHERE plan_key = ? ORDER BY ordinal
                """,
                (plan_key,),
            ).fetchall()
            persisted_declarations = tuple(
                _bytes(value, "declaration_json") for value in declaration_rows
            )
            if tuple(_int(value, "ordinal") for value in declaration_rows) != tuple(
                range(len(expected_declaration_jsons))
            ) or persisted_declarations != tuple(expected_declaration_jsons):
                raise SQLiteStreamWorkLedgerConflict("EOS seal declarations changed before commit")
            existing = _optional_bytes(row, "seal_json")
            if existing is not None:
                if existing != seal_json:
                    raise SQLiteStreamWorkLedgerConflict("EOS seal replay changed source facts")
                _verify_existing_work_rows(
                    connection,
                    plan_key=plan_key,
                    expected_ordinal=None,
                    expected=(finalization,),
                )
                return False
            connection.execute(
                "UPDATE stream_plans SET seal_json = ? WHERE plan_key = ?",
                (sqlite3.Binary(seal_json), plan_key),
            )
            _insert_work(connection, plan_key, finalization)
            return True

        return self._run(
            write=True,
            operation_name="store_seal_and_finalization",
            operation=operation,
        )

    def mark_export_barrier(
        self,
        *,
        plan_key: str,
        manifest_sha256: str,
        member_count: int,
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                """
                SELECT export_manifest_sha256, export_member_count
                FROM stream_plans WHERE plan_key = ?
                """,
                (plan_key,),
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("expected plan is not registered")
            existing = cast(str | None, row["export_manifest_sha256"])
            existing_count = cast(int | None, row["export_member_count"])
            if existing is not None and (existing, existing_count) != (
                manifest_sha256,
                member_count,
            ):
                raise SQLiteStreamWorkLedgerConflict("export barrier replay changed manifest")
            connection.execute(
                """
                UPDATE stream_plans SET export_manifest_sha256 = ?, export_member_count = ?
                WHERE plan_key = ?
                """,
                (manifest_sha256, member_count, plan_key),
            )

        self._run(write=True, operation_name="mark_export_barrier", operation=operation)

    def store_closure_and_open_finalization(
        self,
        *,
        plan_key: str,
        closure_json: bytes,
        finalization_stage: str,
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT terminal_closure_json FROM stream_plans WHERE plan_key = ?",
                (plan_key,),
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("expected plan is not registered")
            existing = _optional_bytes(row, "terminal_closure_json")
            if existing is not None and existing != closure_json:
                raise SQLiteStreamWorkLedgerConflict(
                    "terminal closure replay changed accepted members"
                )
            finalization_rows = connection.execute(
                """
                SELECT publication_state FROM stream_work_plans
                WHERE plan_key = ? AND stage = ?
                """,
                (plan_key, finalization_stage),
            ).fetchall()
            if len(finalization_rows) != 1:
                raise SQLiteStreamWorkLedgerError(
                    "terminal closure requires exact finalization work"
                )
            publication_state = _text(finalization_rows[0], "publication_state")
            if existing is None and publication_state != "GATED":
                raise SQLiteStreamWorkLedgerConflict("unclosed finalization work must remain gated")
            if existing is not None and publication_state == "GATED":
                raise SQLiteStreamWorkLedgerConflict("closed finalization work cannot remain gated")
            connection.execute(
                "UPDATE stream_plans SET terminal_closure_json = ? WHERE plan_key = ?",
                (sqlite3.Binary(closure_json), plan_key),
            )
            cursor = connection.execute(
                """
                UPDATE stream_work_plans SET publication_state = 'PENDING'
                WHERE plan_key = ? AND stage = ? AND publication_state = 'GATED'
                """,
                (plan_key, finalization_stage),
            )
            if cursor.rowcount not in {0, 1}:
                raise SQLiteStreamWorkLedgerError("finalization gate has duplicate work rows")

        self._run(
            write=True,
            operation_name="store_closure_and_open_finalization",
            operation=operation,
        )

    def work_plans(self, plan_key: str) -> tuple[StoredStreamWorkPlan, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                """
                SELECT * FROM stream_work_plans WHERE plan_key = ?
                ORDER BY COALESCE(expected_ordinal, 2147483647), role_order, work_item_id
                """,
                (plan_key,),
            ).fetchall()
            return tuple(_work_from_row(row) for row in rows)

        return self._run(write=False, operation_name="work_plans", operation=operation)

    def pending_publication_work_rows(
        self,
        plan_key: str,
    ) -> tuple[StoredStreamWorkPlan, ...]:
        """Return only stream rows whose execution projection is not yet published.

        This is a recovery boundary: normal append publishes the bounded batch directly,
        while a restart can replay just the durable PENDING rows left by a crash.
        """

        def operation(connection: sqlite3.Connection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                """
                SELECT * FROM stream_work_plans
                WHERE plan_key = ? AND publication_state = 'PENDING'
                ORDER BY COALESCE(expected_ordinal, 2147483647), role_order, work_item_id
                """,
                (plan_key,),
            ).fetchall()
            return tuple(_work_from_row(row) for row in rows)

        return self._run(
            write=False,
            operation_name="pending_publication_work_rows",
            operation=operation,
        )

    def next_ready_work(self, plan_key: str) -> StoredStreamWorkPlan | None:
        """Return the highest-priority published READY row for this graph only."""

        def operation(connection: sqlite3.Connection) -> StoredStreamWorkPlan | None:
            row = connection.execute(
                """
                SELECT stream.*
                FROM stream_work_plans AS stream
                JOIN work_items AS execution
                  ON execution.work_item_id = stream.work_item_id
                WHERE stream.plan_key = ?
                  AND stream.publication_state = 'PUBLISHED'
                  AND execution.state = 'READY'
                ORDER BY
                    execution.priority DESC,
                    CASE WHEN execution.sla_deadline_at IS NULL THEN 1 ELSE 0 END,
                    execution.sla_deadline_at,
                    execution.created_at,
                    execution.work_item_id
                LIMIT 1
                """,
                (plan_key,),
            ).fetchone()
            return None if row is None else _work_from_row(row)

        return self._run(write=False, operation_name="next_ready_work", operation=operation)

    def work_plans_for_ordinals(
        self,
        plan_key: str,
        ordinals: Sequence[int],
    ) -> tuple[StoredStreamWorkPlan, ...]:
        """Load child plans for a bounded set of window ordinals in one lookup."""

        if isinstance(ordinals, (str, bytes)) or not isinstance(ordinals, Sequence):
            raise TypeError("ordinals must be a sequence")
        checked = tuple(ordinals)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in checked
        ):
            raise ValueError("ordinals must contain nonnegative integers")
        if len(set(checked)) != len(checked):
            raise ValueError("ordinals must be unique")
        if not checked:
            return ()
        placeholders = ", ".join("?" for _value in checked)

        def operation(connection: sqlite3.Connection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                f"""
                SELECT * FROM stream_work_plans
                WHERE plan_key = ? AND expected_ordinal IN ({placeholders})
                ORDER BY expected_ordinal, role_order, work_item_id
                """,
                (plan_key, *checked),
            ).fetchall()
            return tuple(_work_from_row(row) for row in rows)

        return self._run(
            write=False,
            operation_name="work_plans_for_ordinals",
            operation=operation,
        )

    def work_plans_for_ids(
        self,
        work_item_ids: Sequence[str],
    ) -> tuple[StoredStreamWorkPlan, ...]:
        """Load a bounded set of stream rows through the primary-key index."""

        if isinstance(work_item_ids, (str, bytes)) or not isinstance(work_item_ids, Sequence):
            raise TypeError("work_item_ids must be a sequence")
        checked_ids = tuple(work_item_ids)
        if any(not isinstance(value, str) or not value for value in checked_ids):
            raise ValueError("work_item_ids must contain non-empty strings")
        if len(set(checked_ids)) != len(checked_ids):
            raise ValueError("work_item_ids must be unique")
        if not checked_ids:
            return ()
        placeholders = ", ".join("?" for _value in checked_ids)

        def operation(connection: sqlite3.Connection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                f"""
                SELECT * FROM stream_work_plans
                WHERE work_item_id IN ({placeholders})
                """,
                checked_ids,
            ).fetchall()
            by_id = {_text(row, "work_item_id"): _work_from_row(row) for row in rows}
            if set(by_id) != set(checked_ids):
                raise SQLiteStreamWorkLedgerError("stream work lookup row is absent")
            return tuple(by_id[value] for value in checked_ids)

        return self._run(write=False, operation_name="work_plans_for_ids", operation=operation)

    def work_plans_for_ordinal(
        self,
        plan_key: str,
        ordinal: int,
    ) -> tuple[StoredStreamWorkPlan, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                """
                SELECT * FROM stream_work_plans
                WHERE plan_key = ? AND expected_ordinal = ?
                ORDER BY role_order, work_item_id
                """,
                (plan_key, ordinal),
            ).fetchall()
            return tuple(_work_from_row(row) for row in rows)

        return self._run(
            write=False,
            operation_name="work_plans_for_ordinal",
            operation=operation,
        )

    def bounded_execution_scope(
        self,
        *,
        plan_key: str,
        max_active: int,
        terminal_states: Sequence[str],
        finalization_stage: str,
    ) -> tuple[StoredStreamWorkExecution, ...]:
        """Return bounded active work plus its direct upstream evidence rows."""

        if isinstance(max_active, bool) or max_active <= 0:
            raise ValueError("max_active must be positive")
        checked_terminal_states = tuple(terminal_states)
        if not checked_terminal_states:
            raise ValueError("terminal_states must not be empty")
        placeholders = ", ".join("?" for _state in checked_terminal_states)

        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[StoredStreamWorkExecution, ...]:
            rows = connection.execute(
                f"""
                WITH active(work_item_id) AS (
                    SELECT stream.work_item_id
                    FROM stream_work_plans AS stream
                    JOIN work_items AS execution
                      ON execution.work_item_id = stream.work_item_id
                    WHERE stream.plan_key = ?
                      AND stream.publication_state = 'PUBLISHED'
                      AND stream.stage <> ?
                      AND execution.state NOT IN ({placeholders})
                    ORDER BY
                      COALESCE(stream.expected_ordinal, 2147483647),
                      stream.role_order,
                      stream.work_item_id
                    LIMIT ?
                ),
                scoped(work_item_id) AS (
                    SELECT work_item_id FROM active
                    UNION
                    SELECT dependency.upstream_work_item_id
                    FROM work_dependencies AS dependency
                    JOIN active
                      ON active.work_item_id = dependency.downstream_work_item_id
                )
                SELECT
                    stream.*,
                    execution.state AS execution_state,
                    execution.created_at AS execution_created_at
                FROM scoped
                JOIN stream_work_plans AS stream
                  ON stream.work_item_id = scoped.work_item_id
                JOIN work_items AS execution
                  ON execution.work_item_id = scoped.work_item_id
                WHERE stream.plan_key = ?
                ORDER BY
                  COALESCE(stream.expected_ordinal, 2147483647),
                  stream.role_order,
                  stream.work_item_id
                """,
                (
                    plan_key,
                    finalization_stage,
                    *checked_terminal_states,
                    max_active,
                    plan_key,
                ),
            ).fetchall()
            return tuple(_work_execution_from_row(row) for row in rows)

        return self._run(
            write=False,
            operation_name="bounded_execution_scope",
            operation=operation,
        )

    def backlog_projection(
        self,
        *,
        plan_key: str,
        terminal_states: Sequence[str],
        finalization_stage: str,
    ) -> StoredStreamBacklog:
        """Aggregate one graph without deserializing its immutable history."""

        checked_terminal_states = frozenset(terminal_states)
        if not checked_terminal_states:
            raise ValueError("terminal_states must not be empty")

        def operation(connection: sqlite3.Connection) -> StoredStreamBacklog:
            metadata = connection.execute(
                """
                SELECT
                    plan.seal_json,
                    plan.export_manifest_sha256,
                    plan.export_member_count,
                    COUNT(window.ordinal) AS declared_window_count,
                    COUNT(window.terminal_member_json) AS terminal_member_count
                FROM stream_plans AS plan
                LEFT JOIN expected_windows AS window
                  ON window.plan_key = plan.plan_key
                WHERE plan.plan_key = ?
                GROUP BY plan.plan_key
                """,
                (plan_key,),
            ).fetchone()
            if metadata is None:
                raise SQLiteStreamWorkLedgerError("expected plan is not registered")
            groups = connection.execute(
                """
                SELECT
                    stream.publication_state,
                    execution.state AS execution_state,
                    COUNT(*) AS state_count,
                    MIN(
                        COALESCE(
                            execution.created_at,
                            json_extract(stream.plan_json, '$.created_at')
                        )
                    ) AS oldest_created_at
                FROM stream_work_plans AS stream
                LEFT JOIN work_items AS execution
                  ON execution.work_item_id = stream.work_item_id
                WHERE stream.plan_key = ?
                GROUP BY stream.publication_state, execution.state
                """,
                (plan_key,),
            ).fetchall()
            finalization = connection.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM stream_work_plans
                    WHERE plan_key = ? AND stage = ?
                      AND publication_state <> 'GATED'
                ) AS published
                """,
                (plan_key, finalization_stage),
            ).fetchone()
            if finalization is None:
                raise SQLiteStreamWorkLedgerError("finalization query returned no row")

            counts: dict[str, int] = {}
            active_backlog = 0
            oldest_active: str | None = None
            for row in groups:
                publication_state = _text(row, "publication_state")
                execution_state = _optional_text(row, "execution_state")
                state_count = _int(row, "state_count")
                if publication_state == "GATED":
                    state_key = "GATED"
                elif execution_state is None:
                    raise SQLiteStreamWorkLedgerError(
                        "published stream work lacks its execution projection"
                    )
                else:
                    state_key = execution_state
                counts[state_key] = counts.get(state_key, 0) + state_count
                if publication_state == "GATED" or execution_state not in checked_terminal_states:
                    active_backlog += state_count
                    created_at = _optional_text(row, "oldest_created_at")
                    if created_at is not None and (
                        oldest_active is None or created_at < oldest_active
                    ):
                        oldest_active = created_at

            return StoredStreamBacklog(
                state_counts=tuple(sorted(counts.items())),
                active_backlog=active_backlog,
                oldest_active_created_at=oldest_active,
                declared_window_count=_int(metadata, "declared_window_count"),
                expected_plan_sealed=_optional_bytes(metadata, "seal_json") is not None,
                terminal_member_count=_int(metadata, "terminal_member_count"),
                export_manifest_sha256=_optional_text(metadata, "export_manifest_sha256"),
                export_member_count=_optional_int(metadata, "export_member_count"),
                finalization_published=bool(_int(finalization, "published")),
            )

        return self._run(
            write=False,
            operation_name="backlog_projection",
            operation=operation,
        )

    def get_work(self, work_item_id: str) -> StoredStreamWorkPlan:
        def operation(connection: sqlite3.Connection) -> StoredStreamWorkPlan:
            row = connection.execute(
                "SELECT * FROM stream_work_plans WHERE work_item_id = ?", (work_item_id,)
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("work item is not in this stream graph")
            return _work_from_row(row)

        return self._run(write=False, operation_name="get_work", operation=operation)

    def get_work_by_key(self, logical_key: str) -> StoredStreamWorkPlan:
        def operation(connection: sqlite3.Connection) -> StoredStreamWorkPlan:
            row = connection.execute(
                "SELECT * FROM stream_work_plans WHERE work_logical_key = ?", (logical_key,)
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("upstream stream work is not durable")
            return _work_from_row(row)

        return self._run(write=False, operation_name="get_work_by_key", operation=operation)

    def mark_published(self, work_item_id: str) -> None:
        """Publish one already-projected work row with legacy no-op semantics."""

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                UPDATE stream_work_plans SET publication_state = 'PUBLISHED'
                WHERE work_item_id = ? AND publication_state = 'PENDING'
                """,
                (work_item_id,),
            )

        self._run(write=True, operation_name="mark_published", operation=operation)

    def mark_published_many(self, work_item_ids: Sequence[str]) -> int:
        """Atomically publish one bounded batch of projected PENDING rows.

        The stream declaration remains durable before this transition. If a process
        stops after projection but before this operation, startup recovery can replay
        the same immutable work plans and retry this exact update safely.
        """

        if isinstance(work_item_ids, (str, bytes)) or not isinstance(work_item_ids, Sequence):
            raise TypeError("work_item_ids must be a sequence")
        checked_ids = tuple(work_item_ids)
        if not checked_ids:
            return 0
        if any(not isinstance(value, str) or not value for value in checked_ids):
            raise ValueError("work_item_ids must contain non-empty strings")
        if len(set(checked_ids)) != len(checked_ids):
            raise ValueError("work_item_ids must be unique")
        placeholders = ", ".join("?" for _value in checked_ids)

        def operation(connection: sqlite3.Connection) -> int:
            rows = connection.execute(
                f"""
                SELECT work_item_id, publication_state
                FROM stream_work_plans
                WHERE work_item_id IN ({placeholders})
                """,
                checked_ids,
            ).fetchall()
            states = {_text(row, "work_item_id"): _text(row, "publication_state") for row in rows}
            if set(states) != set(checked_ids):
                raise SQLiteStreamWorkLedgerError("work publication row is absent")
            if any(state not in {"PENDING", "PUBLISHED"} for state in states.values()):
                raise SQLiteStreamWorkLedgerConflict("gated work cannot be published")
            cursor = connection.execute(
                f"""
                UPDATE stream_work_plans SET publication_state = 'PUBLISHED'
                WHERE work_item_id IN ({placeholders}) AND publication_state = 'PENDING'
                """,
                checked_ids,
            )
            return cursor.rowcount

        return self._run(write=True, operation_name="mark_published_many", operation=operation)

    def store_pending_terminal(
        self,
        *,
        work_item_id: str,
        payload: bytes,
        lease_epoch: int,
        fencing_token: str,
        worker_id: str | None = None,
        authority_now: str | None = None,
        lease_expires_at: str | None = None,
    ) -> bool:
        """Store one acceptance intent; return false if it was already accepted.

        When a current lease is supplied, the authority checks that fence in this same
        transaction before recording a new intent. This retains the crash-visible intent
        before execution terminal transition without adding a separate point-read
        transaction on every normal completion. ``lease_expires_at`` optionally binds
        the full capability returned by the scheduler; callers on the stream path pass
        it so a forged expiry cannot create a pending intent.
        """

        if (worker_id is None) != (authority_now is None):
            raise ValueError("worker_id and authority_now must be supplied together")

        def operation(connection: sqlite3.Connection) -> bool:
            row = connection.execute(
                """
                SELECT terminal_evidence_json, pending_terminal_json,
                       pending_lease_epoch, pending_fencing_token
                FROM stream_work_plans WHERE work_item_id = ?
                """,
                (work_item_id,),
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("terminal work is not in this graph")
            accepted = _optional_bytes(row, "terminal_evidence_json")
            if accepted is not None:
                if accepted != payload:
                    raise SQLiteStreamWorkLedgerConflict(
                        "terminal replay changed accepted stream evidence"
                    )
                return False
            if worker_id is not None and authority_now is not None:
                execution = connection.execute(
                    """
                    SELECT state, lease_epoch, fencing_token, leased_by, lease_expires_at
                    FROM work_items WHERE work_item_id = ?
                    """,
                    (work_item_id,),
                ).fetchone()
                persisted_lease_expires_at = (
                    None if execution is None else _optional_text(execution, "lease_expires_at")
                )
                if (
                    execution is None
                    or _text(execution, "state") not in {"LEASED", "RUNNING"}
                    or _int(execution, "lease_epoch") != lease_epoch
                    or _optional_text(execution, "fencing_token") != fencing_token
                    or _optional_text(execution, "leased_by") != worker_id
                    or persisted_lease_expires_at is None
                    or (
                        lease_expires_at is not None
                        and persisted_lease_expires_at != lease_expires_at
                    )
                    or persisted_lease_expires_at <= authority_now
                ):
                    raise WorkFenceError("work lease is stale, expired, or inactive")
            pending = _optional_bytes(row, "pending_terminal_json")
            pending_epoch = cast(int | None, row["pending_lease_epoch"])
            pending_token = cast(str | None, row["pending_fencing_token"])
            if pending is not None and pending_epoch is not None:
                if (
                    pending == payload
                    and pending_epoch == lease_epoch
                    and pending_token == fencing_token
                ):
                    return True
                if pending_epoch >= lease_epoch:
                    raise SQLiteStreamWorkLedgerConflict(
                        "a newer or conflicting terminal acceptance is pending"
                    )
            connection.execute(
                """
                UPDATE stream_work_plans
                SET pending_terminal_json = ?, pending_lease_epoch = ?,
                    pending_fencing_token = ? WHERE work_item_id = ?
                """,
                (
                    sqlite3.Binary(payload),
                    lease_epoch,
                    fencing_token,
                    work_item_id,
                ),
            )
            return True

        return self._run(
            write=True,
            operation_name="store_pending_terminal",
            operation=operation,
        )

    def pending_work_item_ids(self, plan_key: str) -> tuple[str, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[str, ...]:
            rows = connection.execute(
                """
                SELECT work_item_id FROM stream_work_plans
                WHERE plan_key = ? AND pending_terminal_json IS NOT NULL
                ORDER BY work_item_id
                """,
                (plan_key,),
            ).fetchall()
            return tuple(_text(row, "work_item_id") for row in rows)

        return self._run(
            write=False,
            operation_name="pending_work_item_ids",
            operation=operation,
        )

    def pending_work_rows(self, plan_key: str) -> tuple[StoredStreamWorkPlan, ...]:
        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                """
                SELECT * FROM stream_work_plans
                WHERE plan_key = ? AND pending_terminal_json IS NOT NULL
                ORDER BY work_item_id
                """,
                (plan_key,),
            ).fetchall()
            return tuple(_work_from_row(row) for row in rows)

        return self._run(
            write=False,
            operation_name="pending_work_rows",
            operation=operation,
        )

    def accept_pending_terminal(
        self,
        *,
        work_item_id: str,
        expected_pending_json: bytes,
        terminal_member_json: bytes | None,
        expected_ordinal: int | None,
    ) -> None:
        """Atomically promote a bound pending terminal and optional window member."""

        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                """
                SELECT terminal_evidence_json, pending_terminal_json
                FROM stream_work_plans WHERE work_item_id = ?
                """,
                (work_item_id,),
            ).fetchone()
            if row is None:
                raise SQLiteStreamWorkLedgerError("terminal work is not in this graph")
            accepted = _optional_bytes(row, "terminal_evidence_json")
            if accepted is not None:
                if accepted != expected_pending_json:
                    raise SQLiteStreamWorkLedgerConflict(
                        "terminal replay changed accepted stream evidence"
                    )
                return
            pending = _optional_bytes(row, "pending_terminal_json")
            if pending != expected_pending_json:
                raise SQLiteStreamWorkLedgerConflict("pending terminal changed before acceptance")
            if terminal_member_json is not None:
                if expected_ordinal is None:
                    raise SQLiteStreamWorkLedgerError(
                        "window terminal member requires an expected ordinal"
                    )
                expected = connection.execute(
                    """
                    SELECT terminal_member_json FROM expected_windows
                    WHERE plan_key = (
                        SELECT plan_key FROM stream_work_plans WHERE work_item_id = ?
                    ) AND ordinal = ?
                    """,
                    (work_item_id, expected_ordinal),
                ).fetchone()
                if expected is None:
                    raise SQLiteStreamWorkLedgerError("window reduction lacks expected declaration")
                existing_member = _optional_bytes(expected, "terminal_member_json")
                if existing_member is not None and existing_member != terminal_member_json:
                    raise SQLiteStreamWorkLedgerConflict(
                        "terminal member replay changed accepted evidence"
                    )
            connection.execute(
                """
                UPDATE stream_work_plans
                SET terminal_evidence_json = pending_terminal_json,
                    pending_terminal_json = NULL,
                    pending_lease_epoch = NULL,
                    pending_fencing_token = NULL
                WHERE work_item_id = ?
                """,
                (work_item_id,),
            )
            if terminal_member_json is not None and expected_ordinal is not None:
                connection.execute(
                    """
                    UPDATE expected_windows SET terminal_member_json = ?
                    WHERE plan_key = (
                        SELECT plan_key FROM stream_work_plans WHERE work_item_id = ?
                    ) AND ordinal = ?
                    """,
                    (sqlite3.Binary(terminal_member_json), work_item_id, expected_ordinal),
                )

        self._run(
            write=True,
            operation_name="accept_pending_terminal",
            operation=operation,
        )

    def _initialize_database(self) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stream_extension_metadata (
                    extension_name TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
                )
                """
            )
            metadata = connection.execute(
                """
                SELECT schema_version FROM stream_extension_metadata
                WHERE extension_name = ?
                """,
                (_EXTENSION_NAME,),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE name IN (?, ?, ?, ?)
                """,
                tuple(sorted(_EXTENSION_OBJECT_NAMES)),
            ).fetchall()
            existing_objects = {_text(row, "name") for row in rows}
            if metadata is None:
                if existing_objects:
                    raise SQLiteStreamWorkLedgerError(
                        "refusing to adopt unversioned stream extension tables"
                    )
                for statement in _EXTENSION_SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO stream_extension_metadata (extension_name, schema_version)
                    VALUES (?, ?)
                    """,
                    (_EXTENSION_NAME, _EXTENSION_SCHEMA_VERSION),
                )
            else:
                version = _int(metadata, "schema_version")
                if version == 1:
                    if existing_objects != _EXTENSION_OBJECT_NAMES:
                        raise SQLiteStreamWorkLedgerError(
                            "stream extension v1 schema inventory changed"
                        )
                    _migrate_v1_to_v2(connection)
                    cursor = connection.execute(
                        "UPDATE stream_extension_metadata SET schema_version = ? "
                        "WHERE extension_name = ? AND schema_version = 1",
                        (_EXTENSION_SCHEMA_VERSION, _EXTENSION_NAME),
                    )
                    if cursor.rowcount != 1:
                        raise SQLiteStreamWorkLedgerError("stream migration metadata row missing")
                    version = _EXTENSION_SCHEMA_VERSION
                if version != _EXTENSION_SCHEMA_VERSION:
                    raise SQLiteStreamWorkLedgerError(
                        "stream extension belongs to another schema version"
                    )
                if existing_objects != _EXTENSION_OBJECT_NAMES:
                    raise SQLiteStreamWorkLedgerError("stream extension schema inventory changed")
            _validate_extension_columns(connection)
            for statement in _ADDITIVE_INDEX_STATEMENTS:
                connection.execute(statement)

        self._run(
            write=True,
            operation_name="initialize_extension",
            operation=operation,
        )

    def _run[T](
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[sqlite3.Connection], T],
    ) -> T:
        return self._authority.run_authority_transaction(
            write=write,
            operation_name=f"stream_work.{operation_name}",
            operation=operation,
        )


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    table = "stream_work_plans"
    columns = _table_columns(connection, table)
    for name, declaration in _V1_TO_V2_PENDING_COLUMNS:
        if name not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _validate_extension_columns(connection: sqlite3.Connection) -> None:
    for table, required in _EXTENSION_REQUIRED_COLUMNS.items():
        missing = required.difference(_table_columns(connection, table))
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise SQLiteStreamWorkLedgerError(
                f"{table} is missing required columns: {missing_text}"
            )


def _table_columns(connection: sqlite3.Connection, table: str) -> frozenset[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return frozenset(_text(row, "name") for row in rows)


def _insert_work(
    connection: sqlite3.Connection,
    plan_key: str,
    work: NewStreamWorkPlan,
) -> None:
    connection.execute(
        """
        INSERT INTO stream_work_plans (
            work_item_id, work_logical_key, plan_key, expected_ordinal,
            role_order, stage, plan_json, publication_state,
            terminal_evidence_json, pending_terminal_json,
            pending_lease_epoch, pending_fencing_token
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
        """,
        (
            work.work_item_id,
            work.work_logical_key,
            plan_key,
            work.expected_ordinal,
            work.role_order,
            work.stage,
            sqlite3.Binary(work.plan_json),
            work.publication_state,
        ),
    )


def _verify_existing_work_rows(
    connection: sqlite3.Connection,
    *,
    plan_key: str,
    expected_ordinal: int | None,
    expected: Sequence[NewStreamWorkPlan],
) -> None:
    rows = connection.execute(
        """
        SELECT * FROM stream_work_plans
        WHERE plan_key = ? AND expected_ordinal IS ?
        ORDER BY role_order, work_item_id
        """,
        (plan_key, expected_ordinal),
    ).fetchall()
    stored = tuple(_work_from_row(row) for row in rows)
    if len(stored) != len(expected):
        raise SQLiteStreamWorkLedgerConflict("stream work replay lacks exact companion rows")
    for current, candidate in zip(stored, expected, strict=True):
        allowed_states = (
            {"GATED", "PENDING", "PUBLISHED"}
            if candidate.publication_state == "GATED"
            else {candidate.publication_state, "PUBLISHED"}
        )
        if (
            current.work_item_id != candidate.work_item_id
            or current.work_logical_key != candidate.work_logical_key
            or current.plan_key != plan_key
            or current.expected_ordinal != candidate.expected_ordinal
            or current.role_order != candidate.role_order
            or current.stage != candidate.stage
            or current.plan_json != candidate.plan_json
            or current.publication_state not in allowed_states
        ):
            raise SQLiteStreamWorkLedgerConflict("stream work replay changed exact companion rows")


def _plan_from_row(row: sqlite3.Row) -> StoredStreamPlan:
    return StoredStreamPlan(
        plan_key=_text(row, "plan_key"),
        plan_json=_bytes(row, "plan_json"),
        source_subject_json=_bytes(row, "source_subject_json"),
        composition_config_json=_bytes(row, "composition_config_json"),
        planner_eos_sha256=_optional_text(row, "planner_eos_sha256"),
        seal_json=_optional_bytes(row, "seal_json"),
        terminal_closure_json=_optional_bytes(row, "terminal_closure_json"),
        export_manifest_sha256=_optional_text(row, "export_manifest_sha256"),
        export_member_count=_optional_int(row, "export_member_count"),
    )


def _window_from_row(row: sqlite3.Row) -> StoredExpectedWindow:
    return StoredExpectedWindow(
        plan_key=_text(row, "plan_key"),
        ordinal=_int(row, "ordinal"),
        declaration_json=_bytes(row, "declaration_json"),
        window_json=_bytes(row, "window_json"),
        terminal_member_json=_optional_bytes(row, "terminal_member_json"),
    )


def _work_from_row(row: sqlite3.Row) -> StoredStreamWorkPlan:
    return StoredStreamWorkPlan(
        work_item_id=_text(row, "work_item_id"),
        work_logical_key=_text(row, "work_logical_key"),
        plan_key=_text(row, "plan_key"),
        expected_ordinal=_optional_int(row, "expected_ordinal"),
        role_order=_int(row, "role_order"),
        stage=_text(row, "stage"),
        plan_json=_bytes(row, "plan_json"),
        publication_state=_text(row, "publication_state"),
        terminal_evidence_json=_optional_bytes(row, "terminal_evidence_json"),
        pending_terminal_json=_optional_bytes(row, "pending_terminal_json"),
        pending_lease_epoch=_optional_int(row, "pending_lease_epoch"),
        pending_fencing_token=_optional_text(row, "pending_fencing_token"),
    )


def _work_execution_from_row(row: sqlite3.Row) -> StoredStreamWorkExecution:
    return StoredStreamWorkExecution(
        work=_work_from_row(row),
        execution_state=_text(row, "execution_state"),
        execution_created_at=_text(row, "execution_created_at"),
    )


def _bytes(row: sqlite3.Row, field: str) -> bytes:
    value: object = row[field]
    if not isinstance(value, bytes):
        raise SQLiteStreamWorkLedgerError(f"persisted {field} must be bytes")
    return value


def _optional_bytes(row: sqlite3.Row, field: str) -> bytes | None:
    value: object = row[field]
    if value is not None and not isinstance(value, bytes):
        raise SQLiteStreamWorkLedgerError(f"persisted {field} must be bytes or null")
    return value


def _text(row: sqlite3.Row, field: str) -> str:
    value: object = row[field]
    if not isinstance(value, str):
        raise SQLiteStreamWorkLedgerError(f"persisted {field} must be text")
    return value


def _optional_text(row: sqlite3.Row, field: str) -> str | None:
    value: object = row[field]
    if value is not None and not isinstance(value, str):
        raise SQLiteStreamWorkLedgerError(f"persisted {field} must be text or null")
    return value


def _int(row: sqlite3.Row, field: str) -> int:
    value: object = row[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SQLiteStreamWorkLedgerError(f"persisted {field} must be an integer")
    return value


def _optional_int(row: sqlite3.Row, field: str) -> int | None:
    value: object = row[field]
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise SQLiteStreamWorkLedgerError(f"persisted {field} must be an integer or null")
    return value


__all__ = [
    "NewStreamWindow",
    "NewStreamWorkPlan",
    "SQLiteStreamWorkLedger",
    "SQLiteStreamWorkLedgerConflict",
    "SQLiteStreamWorkLedgerError",
    "StoredExpectedWindow",
    "StoredStreamBacklog",
    "StoredStreamPlan",
    "StoredStreamWorkExecution",
    "StoredStreamWorkPlan",
]
