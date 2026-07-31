"""PostgreSQL persistence for canonical pre-EOS stream work state.

Exact canonical JSON stays in bytea. This adapter only owns declarations, work-plan
bookkeeping, pending terminals, closure gates, and timing-controller state; it does
not perform provider or media activity inside its transaction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from robata.adapters.postgres_authority import PostgresConnection, Row
from robata.adapters.postgres_work_scheduler import PostgresWorkScheduler
from robata.adapters.sqlite_stream_work_ledger import (
    NewStreamWindow,
    NewStreamWorkPlan,
    StoredExpectedWindow,
    StoredStreamBacklog,
    StoredStreamBackpressureController,
    StoredStreamPlan,
    StoredStreamWorkExecution,
    StoredStreamWorkPlan,
)
from robata.adapters.sqlite_work_scheduler import WorkFenceError
from robata.queue.models import TERMINAL_WORK_STATES

_TERMINAL_WORK_STATE_VALUES = tuple(sorted(state.value for state in TERMINAL_WORK_STATES))


class PostgresStreamWorkLedgerError(RuntimeError):
    """The PostgreSQL stream ledger cannot preserve its durable contract."""


class PostgresStreamWorkLedgerConflict(PostgresStreamWorkLedgerError):
    """An exact replay differs from already-persisted stream state."""


class PostgresStreamWorkLedgerFairnessThrottle(PostgresStreamWorkLedgerError):
    """A recording would exceed its durable active-window share in one partition."""

    def __init__(
        self,
        *,
        plan_key: str,
        controller_key: str,
        current_active_window_count: int,
        least_peer_active_window_count: int,
        requested_new_window_count: int,
        allowed_new_window_count: int,
    ) -> None:
        self.plan_key = plan_key
        self.controller_key = controller_key
        self.current_active_window_count = current_active_window_count
        self.least_peer_active_window_count = least_peer_active_window_count
        self.requested_new_window_count = requested_new_window_count
        self.allowed_new_window_count = allowed_new_window_count
        super().__init__(
            "recording admission is throttled by durable active-window fairness "
            f"for partition {controller_key}: requested={requested_new_window_count}, "
            f"allowed={allowed_new_window_count}, "
            f"current_active={current_active_window_count}, "
            f"least_peer_active={least_peer_active_window_count}"
        )


class PostgresStreamWorkLedger:
    """Exact-byte stream ledger sharing one PostgreSQL scheduler authority."""

    backend_kind = "POSTGRESQL"

    def __init__(self, authority: PostgresWorkScheduler) -> None:
        if not isinstance(authority, PostgresWorkScheduler):
            raise TypeError("authority must be PostgresWorkScheduler")
        self._authority = authority

    @property
    def authority(self) -> PostgresWorkScheduler:
        return self._authority

    @property
    def schema(self) -> str:
        return self._authority.schema

    def register_plan(
        self,
        *,
        plan_key: str,
        plan_json: bytes,
        source_subject_json: bytes,
        composition_config_json: bytes,
    ) -> None:
        def operation(connection: PostgresConnection) -> None:
            row = connection.execute(
                "SELECT * FROM stream_plans WHERE plan_key = %s", (plan_key,)
            ).fetchone()
            if row is not None:
                existing = _plan_from_row(row)
                if (
                    existing.plan_json != plan_json
                    or existing.source_subject_json != source_subject_json
                    or existing.composition_config_json != composition_config_json
                ):
                    raise PostgresStreamWorkLedgerConflict(
                        "composition replay changed plan or policy pins"
                    )
                return
            connection.execute(
                """
                INSERT INTO stream_plans (
                    plan_key, plan_json, source_subject_json, composition_config_json,
                    planner_eos_sha256, seal_json, terminal_closure_json,
                    export_manifest_sha256, export_member_count
                ) VALUES (%s, %s, %s, %s, NULL, NULL, NULL, NULL, NULL)
                """,
                (
                    plan_key,
                    (plan_json),
                    (source_subject_json),
                    (composition_config_json),
                ),
            )

        self._run(write=True, operation_name="register_plan", operation=operation)

    def get_plan(self, plan_key: str) -> StoredStreamPlan:
        def operation(connection: PostgresConnection) -> StoredStreamPlan:
            row = connection.execute(
                "SELECT * FROM stream_plans WHERE plan_key = %s", (plan_key,)
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("expected plan is not registered")
            return _plan_from_row(row)

        return self._run(write=False, operation_name="get_plan", operation=operation)

    def plans(self) -> tuple[StoredStreamPlan, ...]:
        def operation(connection: PostgresConnection) -> tuple[StoredStreamPlan, ...]:
            rows = connection.execute("SELECT * FROM stream_plans ORDER BY plan_key").fetchall()
            return tuple(_plan_from_row(row) for row in rows)

        return self._run(write=False, operation_name="plans", operation=operation)

    def claim_backpressure_controller(
        self,
        *,
        plan_key: str,
        controller_key: str,
        policy_version: str,
        owner_id: str,
        initial_state_json: bytes,
    ) -> StoredStreamBackpressureController:
        """Acquire the owner fence for one controller without losing its state."""

        _require_nonempty_text(plan_key, "plan_key")
        _require_nonempty_text(controller_key, "controller_key")
        _require_nonempty_text(policy_version, "policy_version")
        _require_nonempty_text(owner_id, "owner_id")
        _require_nonempty_bytes(initial_state_json, "initial_state_json")

        def operation(connection: PostgresConnection) -> StoredStreamBackpressureController:
            plan = connection.execute(
                "SELECT 1 FROM stream_plans WHERE plan_key = %s",
                (plan_key,),
            ).fetchone()
            if plan is None:
                raise PostgresStreamWorkLedgerError("expected plan is not registered")
            row = connection.execute(
                """
                SELECT * FROM stream_backpressure_controllers
                WHERE plan_key = %s AND controller_key = %s
                """,
                (plan_key, controller_key),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO stream_backpressure_controllers (
                        plan_key, controller_key, policy_version, owner_id,
                        owner_fence, state_json
                    ) VALUES (%s, %s, %s, %s, 1, %s)
                    """,
                    (
                        plan_key,
                        controller_key,
                        policy_version,
                        owner_id,
                        (initial_state_json),
                    ),
                )
                return StoredStreamBackpressureController(
                    plan_key=plan_key,
                    controller_key=controller_key,
                    policy_version=policy_version,
                    owner_id=owner_id,
                    owner_fence=1,
                    state_json=initial_state_json,
                )
            stored = _backpressure_controller_from_row(row)
            if stored.policy_version != policy_version:
                raise PostgresStreamWorkLedgerConflict(
                    "backpressure controller policy version changed"
                )
            if stored.owner_id == owner_id:
                return stored
            next_fence = stored.owner_fence + 1
            cursor = connection.execute(
                """
                UPDATE stream_backpressure_controllers
                SET owner_id = %s, owner_fence = %s
                WHERE plan_key = %s AND controller_key = %s AND owner_fence = %s
                """,
                (owner_id, next_fence, plan_key, controller_key, stored.owner_fence),
            )
            if cursor.rowcount != 1:
                raise WorkFenceError("backpressure controller ownership changed")
            return StoredStreamBackpressureController(
                plan_key=stored.plan_key,
                controller_key=stored.controller_key,
                policy_version=stored.policy_version,
                owner_id=owner_id,
                owner_fence=next_fence,
                state_json=stored.state_json,
            )

        return self._run(
            write=True,
            operation_name="claim_backpressure_controller",
            operation=operation,
        )

    def save_backpressure_controller(
        self,
        controller: StoredStreamBackpressureController,
        *,
        state_json: bytes,
    ) -> StoredStreamBackpressureController:
        """Persist the successor state using the current owner fence."""

        if not isinstance(controller, StoredStreamBackpressureController):
            raise TypeError("controller must be StoredStreamBackpressureController")
        _require_nonempty_bytes(state_json, "state_json")

        def operation(connection: PostgresConnection) -> StoredStreamBackpressureController:
            cursor = connection.execute(
                """
                UPDATE stream_backpressure_controllers
                SET state_json = %s
                WHERE plan_key = %s AND controller_key = %s
                  AND owner_id = %s AND owner_fence = %s AND policy_version = %s
                """,
                (
                    (state_json),
                    controller.plan_key,
                    controller.controller_key,
                    controller.owner_id,
                    controller.owner_fence,
                    controller.policy_version,
                ),
            )
            if cursor.rowcount != 1:
                raise WorkFenceError("backpressure controller fence is stale")
            return StoredStreamBackpressureController(
                plan_key=controller.plan_key,
                controller_key=controller.controller_key,
                policy_version=controller.policy_version,
                owner_id=controller.owner_id,
                owner_fence=controller.owner_fence,
                state_json=state_json,
            )

        return self._run(
            write=True,
            operation_name="save_backpressure_controller",
            operation=operation,
        )

    def append_windows(
        self,
        *,
        plan_key: str,
        windows: Sequence[NewStreamWindow],
        controller_key: str | None = None,
        controller_policy_version: str | None = None,
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
        if (controller_key is None) != (controller_policy_version is None):
            raise ValueError(
                "controller_key and controller_policy_version must be supplied together"
            )
        if controller_key is not None and controller_policy_version is not None:
            _require_nonempty_text(controller_key, "controller_key")
            _require_nonempty_text(controller_policy_version, "controller_policy_version")
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
                raise PostgresStreamWorkLedgerConflict(
                    "stream child work ordinal does not match its window"
                )
        ordinals = tuple(window.ordinal for window in checked)
        if len(set(ordinals)) != len(ordinals) or ordinals != tuple(sorted(ordinals)):
            raise PostgresStreamWorkLedgerConflict(
                "window batch ordinals must be unique and ordered"
            )

        def operation(connection: PostgresConnection) -> tuple[bool, ...]:
            plan = connection.execute(
                """
                SELECT planner_eos_sha256, seal_json
                FROM stream_plans WHERE plan_key = %s
                """,
                (plan_key,),
            ).fetchone()
            if plan is None:
                raise PostgresStreamWorkLedgerError("expected plan is not registered")

            tail = connection.execute(
                """
                SELECT ordinal FROM expected_windows
                WHERE plan_key = %s
                ORDER BY ordinal DESC
                LIMIT 1
                """,
                (plan_key,),
            ).fetchone()
            next_ordinal = 0 if tail is None else _int(tail, "ordinal") + 1
            inserted: list[bool] = []
            new_windows: list[NewStreamWindow] = []
            for window in checked:
                existing = connection.execute(
                    """
                    SELECT * FROM expected_windows
                    WHERE plan_key = %s AND ordinal = %s
                    """,
                    (plan_key, window.ordinal),
                ).fetchone()
                if existing is not None:
                    stored = _window_from_row(existing)
                    if (
                        stored.declaration_json != window.declaration_json
                        or stored.window_json != window.window_json
                    ):
                        raise PostgresStreamWorkLedgerConflict(
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
                    raise PostgresStreamWorkLedgerConflict(
                        "cannot append a new window after planner EOS"
                    )
                if next_ordinal != window.ordinal:
                    raise PostgresStreamWorkLedgerConflict(
                        "expected windows must be appended in contiguous planner order"
                    )
                new_windows.append(window)
                next_ordinal += 1
                inserted.append(True)

            if new_windows and controller_key is not None and controller_policy_version is not None:
                _enforce_recording_fair_admission(
                    connection,
                    plan_key=plan_key,
                    controller_key=controller_key,
                    controller_policy_version=controller_policy_version,
                    requested_new_window_count=len(new_windows),
                )

            for window, did_insert in zip(checked, inserted, strict=True):
                if not did_insert:
                    continue
                connection.execute(
                    """
                    INSERT INTO expected_windows (
                        plan_key, ordinal, declaration_json, window_json, terminal_member_json
                    ) VALUES (%s, %s, %s, %s, NULL)
                    """,
                    (
                        plan_key,
                        window.ordinal,
                        (window.declaration_json),
                        (window.window_json),
                    ),
                )
                for work in window.work_plans:
                    _insert_work(connection, plan_key, work)
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
        controller_key: str | None = None,
        controller_policy_version: str | None = None,
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
            controller_key=controller_key,
            controller_policy_version=controller_policy_version,
        )[0]

    def windows(self, plan_key: str) -> tuple[StoredExpectedWindow, ...]:
        def operation(connection: PostgresConnection) -> tuple[StoredExpectedWindow, ...]:
            rows = connection.execute(
                """
                SELECT * FROM expected_windows WHERE plan_key = %s ORDER BY ordinal
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
        placeholders = ", ".join("%s" for _value in checked)

        def operation(connection: PostgresConnection) -> tuple[StoredExpectedWindow, ...]:
            rows = connection.execute(
                f"""
                SELECT * FROM expected_windows
                WHERE plan_key = %s AND ordinal IN ({placeholders})
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
        def operation(connection: PostgresConnection) -> int:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM expected_windows WHERE plan_key = %s",
                (plan_key,),
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("window count query returned no row")
            return _int(row, "count")

        return self._run(write=False, operation_name="window_count", operation=operation)

    def next_window_ordinal(self, plan_key: str) -> int:
        """Return the next append ordinal through the ordered window index."""

        def operation(connection: PostgresConnection) -> int:
            row = connection.execute(
                """
                SELECT ordinal FROM expected_windows
                WHERE plan_key = %s
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
        def operation(connection: PostgresConnection) -> StoredExpectedWindow | None:
            row = connection.execute(
                """
                SELECT * FROM expected_windows WHERE plan_key = %s AND ordinal = %s
                """,
                (plan_key, ordinal),
            ).fetchone()
            return None if row is None else _window_from_row(row)

        return self._run(write=False, operation_name="window_at", operation=operation)

    def terminal_member_at(self, plan_key: str, ordinal: int) -> bytes | None:
        def operation(connection: PostgresConnection) -> bytes | None:
            row = connection.execute(
                """
                SELECT terminal_member_json FROM expected_windows
                WHERE plan_key = %s AND ordinal = %s
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
        def operation(connection: PostgresConnection) -> int:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM expected_windows
                WHERE plan_key = %s AND terminal_member_json IS NOT NULL
                """,
                (plan_key,),
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("terminal member count query returned no row")
            return _int(row, "count")

        return self._run(
            write=False,
            operation_name="terminal_member_count",
            operation=operation,
        )

    def set_planner_eos(self, plan_key: str, finish_sha256: str) -> None:
        def operation(connection: PostgresConnection) -> None:
            row = connection.execute(
                "SELECT planner_eos_sha256 FROM stream_plans WHERE plan_key = %s", (plan_key,)
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("expected plan is not registered")
            current = cast(str | None, row["planner_eos_sha256"])
            if current is not None and current != finish_sha256:
                raise PostgresStreamWorkLedgerConflict("planner EOS replay changed exact facts")
            connection.execute(
                "UPDATE stream_plans SET planner_eos_sha256 = %s WHERE plan_key = %s",
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
        def operation(connection: PostgresConnection) -> bool:
            row = connection.execute(
                "SELECT planner_eos_sha256, seal_json FROM stream_plans WHERE plan_key = %s",
                (plan_key,),
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("expected plan is not registered")
            if row["planner_eos_sha256"] is None:
                raise PostgresStreamWorkLedgerConflict("planner EOS must be durable before seal")
            declaration_rows = connection.execute(
                """
                SELECT ordinal, declaration_json FROM expected_windows
                WHERE plan_key = %s ORDER BY ordinal
                """,
                (plan_key,),
            ).fetchall()
            persisted_declarations = tuple(
                _bytes(value, "declaration_json") for value in declaration_rows
            )
            if tuple(_int(value, "ordinal") for value in declaration_rows) != tuple(
                range(len(expected_declaration_jsons))
            ) or persisted_declarations != tuple(expected_declaration_jsons):
                raise PostgresStreamWorkLedgerConflict(
                    "EOS seal declarations changed before commit"
                )
            existing = _optional_bytes(row, "seal_json")
            if existing is not None:
                if existing != seal_json:
                    raise PostgresStreamWorkLedgerConflict("EOS seal replay changed source facts")
                _verify_existing_work_rows(
                    connection,
                    plan_key=plan_key,
                    expected_ordinal=None,
                    expected=(finalization,),
                )
                return False
            connection.execute(
                "UPDATE stream_plans SET seal_json = %s WHERE plan_key = %s",
                ((seal_json), plan_key),
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
        def operation(connection: PostgresConnection) -> None:
            row = connection.execute(
                """
                SELECT export_manifest_sha256, export_member_count
                FROM stream_plans WHERE plan_key = %s
                """,
                (plan_key,),
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("expected plan is not registered")
            existing = cast(str | None, row["export_manifest_sha256"])
            existing_count = cast(int | None, row["export_member_count"])
            if existing is not None and (existing, existing_count) != (
                manifest_sha256,
                member_count,
            ):
                raise PostgresStreamWorkLedgerConflict("export barrier replay changed manifest")
            connection.execute(
                """
                UPDATE stream_plans SET export_manifest_sha256 = %s, export_member_count = %s
                WHERE plan_key = %s
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
        def operation(connection: PostgresConnection) -> None:
            row = connection.execute(
                "SELECT terminal_closure_json FROM stream_plans WHERE plan_key = %s",
                (plan_key,),
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("expected plan is not registered")
            existing = _optional_bytes(row, "terminal_closure_json")
            if existing is not None and existing != closure_json:
                raise PostgresStreamWorkLedgerConflict(
                    "terminal closure replay changed accepted members"
                )
            finalization_rows = connection.execute(
                """
                SELECT publication_state FROM stream_work_plans
                WHERE plan_key = %s AND stage = %s
                """,
                (plan_key, finalization_stage),
            ).fetchall()
            if len(finalization_rows) != 1:
                raise PostgresStreamWorkLedgerError(
                    "terminal closure requires exact finalization work"
                )
            publication_state = _text(finalization_rows[0], "publication_state")
            if existing is None and publication_state != "GATED":
                raise PostgresStreamWorkLedgerConflict(
                    "unclosed finalization work must remain gated"
                )
            if existing is not None and publication_state == "GATED":
                raise PostgresStreamWorkLedgerConflict(
                    "closed finalization work cannot remain gated"
                )
            connection.execute(
                "UPDATE stream_plans SET terminal_closure_json = %s WHERE plan_key = %s",
                ((closure_json), plan_key),
            )
            cursor = connection.execute(
                """
                UPDATE stream_work_plans SET publication_state = 'PENDING'
                WHERE plan_key = %s AND stage = %s AND publication_state = 'GATED'
                """,
                (plan_key, finalization_stage),
            )
            if cursor.rowcount not in {0, 1}:
                raise PostgresStreamWorkLedgerError("finalization gate has duplicate work rows")

        self._run(
            write=True,
            operation_name="store_closure_and_open_finalization",
            operation=operation,
        )

    def work_plans(self, plan_key: str) -> tuple[StoredStreamWorkPlan, ...]:
        def operation(connection: PostgresConnection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                """
                SELECT * FROM stream_work_plans WHERE plan_key = %s
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

        def operation(connection: PostgresConnection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                """
                SELECT * FROM stream_work_plans
                WHERE plan_key = %s AND publication_state = 'PENDING'
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

        def operation(connection: PostgresConnection) -> StoredStreamWorkPlan | None:
            row = connection.execute(
                """
                SELECT stream.*
                FROM stream_work_plans AS stream
                JOIN work_items AS execution
                  ON execution.work_item_id = stream.work_item_id
                WHERE stream.plan_key = %s
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
        placeholders = ", ".join("%s" for _value in checked)

        def operation(connection: PostgresConnection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                f"""
                SELECT * FROM stream_work_plans
                WHERE plan_key = %s AND expected_ordinal IN ({placeholders})
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
        placeholders = ", ".join("%s" for _value in checked_ids)

        def operation(connection: PostgresConnection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                f"""
                SELECT * FROM stream_work_plans
                WHERE work_item_id IN ({placeholders})
                """,
                checked_ids,
            ).fetchall()
            by_id = {_text(row, "work_item_id"): _work_from_row(row) for row in rows}
            if set(by_id) != set(checked_ids):
                raise PostgresStreamWorkLedgerError("stream work lookup row is absent")
            return tuple(by_id[value] for value in checked_ids)

        return self._run(write=False, operation_name="work_plans_for_ids", operation=operation)

    def work_plans_for_ordinal(
        self,
        plan_key: str,
        ordinal: int,
    ) -> tuple[StoredStreamWorkPlan, ...]:
        def operation(connection: PostgresConnection) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                """
                SELECT * FROM stream_work_plans
                WHERE plan_key = %s AND expected_ordinal = %s
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
        placeholders = ", ".join("%s" for _state in checked_terminal_states)

        def operation(
            connection: PostgresConnection,
        ) -> tuple[StoredStreamWorkExecution, ...]:
            rows = connection.execute(
                f"""
                WITH active(work_item_id) AS (
                    SELECT stream.work_item_id
                    FROM stream_work_plans AS stream
                    JOIN work_items AS execution
                      ON execution.work_item_id = stream.work_item_id
                    WHERE stream.plan_key = %s
                      AND stream.publication_state = 'PUBLISHED'
                      AND stream.stage <> %s
                      AND execution.state NOT IN ({placeholders})
                    ORDER BY
                      COALESCE(stream.expected_ordinal, 2147483647),
                      stream.role_order,
                      stream.work_item_id
                    LIMIT %s
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
                WHERE stream.plan_key = %s
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

        def operation(connection: PostgresConnection) -> StoredStreamBacklog:
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
                WHERE plan.plan_key = %s
                GROUP BY plan.plan_key
                """,
                (plan_key,),
            ).fetchone()
            if metadata is None:
                raise PostgresStreamWorkLedgerError("expected plan is not registered")
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
                WHERE stream.plan_key = %s
                GROUP BY stream.publication_state, execution.state
                """,
                (plan_key,),
            ).fetchall()
            finalization = connection.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM stream_work_plans
                    WHERE plan_key = %s AND stage = %s
                      AND publication_state <> 'GATED'
                ) AS published
                """,
                (plan_key, finalization_stage),
            ).fetchone()
            if finalization is None:
                raise PostgresStreamWorkLedgerError("finalization query returned no row")

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
                    raise PostgresStreamWorkLedgerError(
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
        def operation(connection: PostgresConnection) -> StoredStreamWorkPlan:
            row = connection.execute(
                "SELECT * FROM stream_work_plans WHERE work_item_id = %s", (work_item_id,)
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("work item is not in this stream graph")
            return _work_from_row(row)

        return self._run(write=False, operation_name="get_work", operation=operation)

    def get_work_by_key(self, logical_key: str) -> StoredStreamWorkPlan:
        def operation(connection: PostgresConnection) -> StoredStreamWorkPlan:
            row = connection.execute(
                "SELECT * FROM stream_work_plans WHERE work_logical_key = %s", (logical_key,)
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("upstream stream work is not durable")
            return _work_from_row(row)

        return self._run(write=False, operation_name="get_work_by_key", operation=operation)

    def mark_published(self, work_item_id: str) -> None:
        """Publish one already-projected work row with legacy no-op semantics."""

        def operation(connection: PostgresConnection) -> None:
            connection.execute(
                """
                UPDATE stream_work_plans SET publication_state = 'PUBLISHED'
                WHERE work_item_id = %s AND publication_state = 'PENDING'
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
        placeholders = ", ".join("%s" for _value in checked_ids)

        def operation(connection: PostgresConnection) -> int:
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
                raise PostgresStreamWorkLedgerError("work publication row is absent")
            if any(state not in {"PENDING", "PUBLISHED"} for state in states.values()):
                raise PostgresStreamWorkLedgerConflict("gated work cannot be published")
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

        def operation(connection: PostgresConnection) -> bool:
            row = connection.execute(
                """
                SELECT terminal_evidence_json, pending_terminal_json,
                       pending_lease_epoch, pending_fencing_token
                FROM stream_work_plans WHERE work_item_id = %s
                """,
                (work_item_id,),
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("terminal work is not in this graph")
            accepted = _optional_bytes(row, "terminal_evidence_json")
            if accepted is not None:
                if accepted != payload:
                    raise PostgresStreamWorkLedgerConflict(
                        "terminal replay changed accepted stream evidence"
                    )
                return False
            if worker_id is not None and authority_now is not None:
                execution = connection.execute(
                    """
                    SELECT state, lease_epoch, fencing_token, leased_by, lease_expires_at
                    FROM work_items WHERE work_item_id = %s
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
                    raise PostgresStreamWorkLedgerConflict(
                        "a newer or conflicting terminal acceptance is pending"
                    )
            connection.execute(
                """
                UPDATE stream_work_plans
                SET pending_terminal_json = %s, pending_lease_epoch = %s,
                    pending_fencing_token = %s WHERE work_item_id = %s
                """,
                (
                    (payload),
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
        def operation(connection: PostgresConnection) -> tuple[str, ...]:
            rows = connection.execute(
                """
                SELECT work_item_id FROM stream_work_plans
                WHERE plan_key = %s AND pending_terminal_json IS NOT NULL
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
            connection: PostgresConnection,
        ) -> tuple[StoredStreamWorkPlan, ...]:
            rows = connection.execute(
                """
                SELECT * FROM stream_work_plans
                WHERE plan_key = %s AND pending_terminal_json IS NOT NULL
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

        def operation(connection: PostgresConnection) -> None:
            row = connection.execute(
                """
                SELECT terminal_evidence_json, pending_terminal_json
                FROM stream_work_plans WHERE work_item_id = %s
                """,
                (work_item_id,),
            ).fetchone()
            if row is None:
                raise PostgresStreamWorkLedgerError("terminal work is not in this graph")
            accepted = _optional_bytes(row, "terminal_evidence_json")
            if accepted is not None:
                if accepted != expected_pending_json:
                    raise PostgresStreamWorkLedgerConflict(
                        "terminal replay changed accepted stream evidence"
                    )
                return
            pending = _optional_bytes(row, "pending_terminal_json")
            if pending != expected_pending_json:
                raise PostgresStreamWorkLedgerConflict("pending terminal changed before acceptance")
            if terminal_member_json is not None:
                if expected_ordinal is None:
                    raise PostgresStreamWorkLedgerError(
                        "window terminal member requires an expected ordinal"
                    )
                expected = connection.execute(
                    """
                    SELECT terminal_member_json FROM expected_windows
                    WHERE plan_key = (
                        SELECT plan_key FROM stream_work_plans WHERE work_item_id = %s
                    ) AND ordinal = %s
                    """,
                    (work_item_id, expected_ordinal),
                ).fetchone()
                if expected is None:
                    raise PostgresStreamWorkLedgerError(
                        "window reduction lacks expected declaration"
                    )
                existing_member = _optional_bytes(expected, "terminal_member_json")
                if existing_member is not None and existing_member != terminal_member_json:
                    raise PostgresStreamWorkLedgerConflict(
                        "terminal member replay changed accepted evidence"
                    )
            connection.execute(
                """
                UPDATE stream_work_plans
                SET terminal_evidence_json = pending_terminal_json,
                    pending_terminal_json = NULL,
                    pending_lease_epoch = NULL,
                    pending_fencing_token = NULL
                WHERE work_item_id = %s
                """,
                (work_item_id,),
            )
            if terminal_member_json is not None and expected_ordinal is not None:
                connection.execute(
                    """
                    UPDATE expected_windows SET terminal_member_json = %s
                    WHERE plan_key = (
                        SELECT plan_key FROM stream_work_plans WHERE work_item_id = %s
                    ) AND ordinal = %s
                    """,
                    ((terminal_member_json), work_item_id, expected_ordinal),
                )

        self._run(
            write=True,
            operation_name="accept_pending_terminal",
            operation=operation,
        )

    def _run[T](
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[PostgresConnection], T],
    ) -> T:
        return self._authority.run_authority_transaction(
            write=write,
            operation_name=f"stream_work.{operation_name}",
            operation=operation,
        )


def _enforce_recording_fair_admission(
    connection: PostgresConnection,
    *,
    plan_key: str,
    controller_key: str,
    controller_policy_version: str,
    requested_new_window_count: int,
) -> None:
    """Keep one partition's durable active windows balanced across recordings.

    The fairness source is the authoritative stream/work ledger, rather than
    process-local requests. A window remains active while any of its fixed DAG
    work is pending or nonterminal. This includes a durable PENDING projection
    after a crash, so reopening the database reconstructs the exact share.
    """

    controller = connection.execute(
        """
        SELECT policy_version FROM stream_backpressure_controllers
        WHERE plan_key = %s AND controller_key = %s
        """,
        (plan_key, controller_key),
    ).fetchone()
    if controller is None:
        raise PostgresStreamWorkLedgerError(
            "backpressure fairness requires a registered partition controller"
        )
    if _text(controller, "policy_version") != controller_policy_version:
        raise PostgresStreamWorkLedgerConflict(
            "backpressure fairness policy does not match this recording controller"
        )

    policy_rows = connection.execute(
        """
        SELECT DISTINCT policy_version FROM stream_backpressure_controllers
        WHERE controller_key = %s ORDER BY policy_version
        """,
        (controller_key,),
    ).fetchall()
    if any(_text(row, "policy_version") != controller_policy_version for row in policy_rows):
        raise PostgresStreamWorkLedgerConflict(
            "backpressure partition controller key has conflicting policy versions"
        )

    terminal_placeholders = ", ".join("%s" for _ in _TERMINAL_WORK_STATE_VALUES)
    rows = connection.execute(
        f"""
        SELECT stream.plan_key,
               COUNT(DISTINCT stream.expected_ordinal) AS active_window_count
        FROM stream_work_plans AS stream
        JOIN stream_backpressure_controllers AS controller
          ON controller.plan_key = stream.plan_key
        LEFT JOIN work_items AS execution
          ON execution.work_item_id = stream.work_item_id
        WHERE controller.controller_key = %s
          AND stream.expected_ordinal IS NOT NULL
          AND (
              stream.publication_state = 'PENDING'
              OR execution.work_item_id IS NULL
              OR execution.state NOT IN ({terminal_placeholders})
          )
        GROUP BY stream.plan_key
        ORDER BY stream.plan_key
        """,
        (controller_key, *_TERMINAL_WORK_STATE_VALUES),
    ).fetchall()
    active_by_plan = {_text(row, "plan_key"): _int(row, "active_window_count") for row in rows}
    peer_active_counts = tuple(
        count for candidate, count in active_by_plan.items() if candidate != plan_key
    )
    if not peer_active_counts:
        return

    current_active = active_by_plan.get(plan_key, 0)
    least_peer_active = min(peer_active_counts)
    allowed_new = max(0, least_peer_active + 1 - current_active)
    if requested_new_window_count > allowed_new:
        raise PostgresStreamWorkLedgerFairnessThrottle(
            plan_key=plan_key,
            controller_key=controller_key,
            current_active_window_count=current_active,
            least_peer_active_window_count=least_peer_active,
            requested_new_window_count=requested_new_window_count,
            allowed_new_window_count=allowed_new,
        )


def _insert_work(
    connection: PostgresConnection,
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
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, NULL, NULL)
        """,
        (
            work.work_item_id,
            work.work_logical_key,
            plan_key,
            work.expected_ordinal,
            work.role_order,
            work.stage,
            (work.plan_json),
            work.publication_state,
        ),
    )


def _verify_existing_work_rows(
    connection: PostgresConnection,
    *,
    plan_key: str,
    expected_ordinal: int | None,
    expected: Sequence[NewStreamWorkPlan],
) -> None:
    rows = connection.execute(
        """
        SELECT * FROM stream_work_plans
        WHERE plan_key = %s AND expected_ordinal IS NOT DISTINCT FROM %s
        ORDER BY role_order, work_item_id
        """,
        (plan_key, expected_ordinal),
    ).fetchall()
    stored = tuple(_work_from_row(row) for row in rows)
    if len(stored) != len(expected):
        raise PostgresStreamWorkLedgerConflict("stream work replay lacks exact companion rows")
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
            raise PostgresStreamWorkLedgerConflict(
                "stream work replay changed exact companion rows"
            )


def _plan_from_row(row: Row) -> StoredStreamPlan:
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


def _backpressure_controller_from_row(
    row: Row,
) -> StoredStreamBackpressureController:
    return StoredStreamBackpressureController(
        plan_key=_text(row, "plan_key"),
        controller_key=_text(row, "controller_key"),
        policy_version=_text(row, "policy_version"),
        owner_id=_text(row, "owner_id"),
        owner_fence=_int(row, "owner_fence"),
        state_json=_bytes(row, "state_json"),
    )


def _window_from_row(row: Row) -> StoredExpectedWindow:
    return StoredExpectedWindow(
        plan_key=_text(row, "plan_key"),
        ordinal=_int(row, "ordinal"),
        declaration_json=_bytes(row, "declaration_json"),
        window_json=_bytes(row, "window_json"),
        terminal_member_json=_optional_bytes(row, "terminal_member_json"),
    )


def _work_from_row(row: Row) -> StoredStreamWorkPlan:
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


def _work_execution_from_row(row: Row) -> StoredStreamWorkExecution:
    return StoredStreamWorkExecution(
        work=_work_from_row(row),
        execution_state=_text(row, "execution_state"),
        execution_created_at=_text(row, "execution_created_at"),
    )


def _require_nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _require_nonempty_bytes(value: object, field: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{field} must be non-empty bytes")
    return value


def _bytes(row: Row, field: str) -> bytes:
    value: object = row[field]
    if not isinstance(value, bytes):
        raise PostgresStreamWorkLedgerError(f"persisted {field} must be bytes")
    return value


def _optional_bytes(row: Row, field: str) -> bytes | None:
    value: object = row[field]
    if value is not None and not isinstance(value, bytes):
        raise PostgresStreamWorkLedgerError(f"persisted {field} must be bytes or null")
    return value


def _text(row: Row, field: str) -> str:
    value: object = row[field]
    if not isinstance(value, str):
        raise PostgresStreamWorkLedgerError(f"persisted {field} must be text")
    return value


def _optional_text(row: Row, field: str) -> str | None:
    value: object = row[field]
    if value is not None and not isinstance(value, str):
        raise PostgresStreamWorkLedgerError(f"persisted {field} must be text or null")
    return value


def _int(row: Row, field: str) -> int:
    value: object = row[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PostgresStreamWorkLedgerError(f"persisted {field} must be an integer")
    return value


def _optional_int(row: Row, field: str) -> int | None:
    value: object = row[field]
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise PostgresStreamWorkLedgerError(f"persisted {field} must be an integer or null")
    return value


__all__ = [
    "NewStreamWindow",
    "NewStreamWorkPlan",
    "PostgresStreamWorkLedger",
    "PostgresStreamWorkLedgerConflict",
    "PostgresStreamWorkLedgerError",
    "PostgresStreamWorkLedgerFairnessThrottle",
    "StoredExpectedWindow",
    "StoredStreamBacklog",
    "StoredStreamBackpressureController",
    "StoredStreamPlan",
    "StoredStreamWorkExecution",
    "StoredStreamWorkPlan",
]
