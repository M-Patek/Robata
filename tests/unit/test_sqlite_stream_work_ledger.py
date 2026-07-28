import sqlite3
from pathlib import Path

import pytest

from robata.adapters.sqlite_stream_work_ledger import (
    SQLiteStreamWorkLedger,
    SQLiteStreamWorkLedgerConflict,
)
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler, WorkFenceError


def test_v1_pending_terminal_columns_migrate_and_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "work.sqlite3"
    SQLiteWorkScheduler(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE stream_extension_metadata (
                extension_name TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
            );
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
            );
            CREATE TABLE expected_windows (
                plan_key TEXT NOT NULL REFERENCES stream_plans(plan_key),
                ordinal INTEGER NOT NULL,
                declaration_json BLOB NOT NULL,
                window_json BLOB NOT NULL,
                terminal_member_json BLOB,
                PRIMARY KEY (plan_key, ordinal)
            );
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
                terminal_evidence_json BLOB
            );
            CREATE INDEX stream_work_plan_order
            ON stream_work_plans(plan_key, expected_ordinal, role_order);
            INSERT INTO stream_extension_metadata(extension_name, schema_version)
            VALUES ('stream-work-ledger', 1);
            """
        )

    SQLiteStreamWorkLedger(SQLiteWorkScheduler(database_path))
    with sqlite3.connect(database_path) as connection:
        version = connection.execute(
            """
            SELECT schema_version FROM stream_extension_metadata
            WHERE extension_name = 'stream-work-ledger'
            """
        ).fetchone()
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(stream_work_plans)").fetchall()
        }

    assert version == (3,)
    assert {
        "pending_terminal_json",
        "pending_lease_epoch",
        "pending_fencing_token",
    } <= columns

    SQLiteStreamWorkLedger(SQLiteWorkScheduler(database_path))


def test_backpressure_controller_owner_fence_and_policy_pin(tmp_path: Path) -> None:
    database_path = tmp_path / "work.sqlite3"
    ledger = SQLiteStreamWorkLedger(SQLiteWorkScheduler(database_path))
    ledger.register_plan(
        plan_key="plan-key",
        plan_json=b"{}",
        source_subject_json=b"{}",
        composition_config_json=b"{}",
    )

    first = ledger.claim_backpressure_controller(
        plan_key="plan-key",
        controller_key="window-admission",
        policy_version="policy-v1",
        owner_id="owner-one",
        initial_state_json=b'{"state":"initial"}',
    )
    same_owner = ledger.claim_backpressure_controller(
        plan_key="plan-key",
        controller_key="window-admission",
        policy_version="policy-v1",
        owner_id="owner-one",
        initial_state_json=b'{"state":"ignored"}',
    )
    assert same_owner == first

    second = ledger.claim_backpressure_controller(
        plan_key="plan-key",
        controller_key="window-admission",
        policy_version="policy-v1",
        owner_id="owner-two",
        initial_state_json=b'{"state":"ignored"}',
    )
    assert second.owner_fence == first.owner_fence + 1
    assert second.state_json == first.state_json
    with pytest.raises(WorkFenceError):
        ledger.save_backpressure_controller(first, state_json=b'{"state":"stale"}')

    saved = ledger.save_backpressure_controller(second, state_json=b'{"state":"next"}')
    assert saved.state_json == b'{"state":"next"}'
    with pytest.raises(SQLiteStreamWorkLedgerConflict, match="policy version"):
        ledger.claim_backpressure_controller(
            plan_key="plan-key",
            controller_key="window-admission",
            policy_version="policy-v2",
            owner_id="owner-three",
            initial_state_json=b'{"state":"ignored"}',
        )
