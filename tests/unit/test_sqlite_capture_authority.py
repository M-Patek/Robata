from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

from robata.adapters.sqlite_capture_authority import (
    SQLiteCaptureAuthorityConflict,
    SQLiteLocalCaptureAuthority,
)
from robata.adapters.sqlite_stream_work_ledger import SQLiteStreamWorkLedger
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import AuthorityBinding, ChannelBinding
from robata.contracts.stream_source import PreEosCaptureSubject


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _schema(value: int = 1) -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/pre-eos-capture-subject",
        version="1.0.0",
        artifact_id=_uuid(value),
        sha256=_digest(value),
    )


def _bindings() -> tuple[ChannelBinding, ...]:
    return tuple(
        ChannelBinding(
            camera_id=camera_id,
            source_channel_id=f"source-{camera_id.value}",
            source_channel_epoch=1,
            channel_binding_semantic_sha256=_digest(100 + index),
        )
        for index, camera_id in enumerate(CAMERA_IDS)
    )


def _source_authority(value: int = 1) -> AuthorityBinding:
    return AuthorityBinding(
        authority_id=f"source-authority-{value}",
        authority_epoch=1,
        policy_version="source-policy-v1",
        initial_binding_semantic_sha256=_digest(200 + value),
    )


def _capture_authority(database_path: Path) -> SQLiteLocalCaptureAuthority:
    return SQLiteLocalCaptureAuthority(
        SQLiteWorkScheduler(database_path),
        capture_authority_id="local-capture-authority",
        capture_authority_epoch=3,
        capture_assignment_policy_version="assignment-v1",
    )


def _issue(
    authority: SQLiteLocalCaptureAuthority,
    receipt_slot: str,
    *,
    schema_ref: SchemaRef | None = None,
    mapping_authority: AuthorityBinding | None = None,
) -> PreEosCaptureSubject:
    source_authority = _source_authority()
    return authority.issue(
        receipt_slot,
        _schema() if schema_ref is None else schema_ref,
        _bindings(),
        source_authority if mapping_authority is None else mapping_authority,
        source_authority,
    )


def test_fresh_issue_reopens_as_exact_replay(tmp_path: Path) -> None:
    database_path = tmp_path / "work.sqlite3"
    first = _issue(_capture_authority(database_path), "input/slot-1")

    replayed = _issue(_capture_authority(database_path), "input/slot-1")

    assert replayed == first
    assert replayed.acquisition_id == "local-capture-authority:1"
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT subject_json FROM capture_authority_receipts
            WHERE receipt_slot = 'input/slot-1'
            """
        ).fetchone()
        next_sequence = connection.execute(
            """
            SELECT next_acquisition_sequence FROM capture_authority_metadata
            WHERE singleton = 1
            """
        ).fetchone()
    assert row is not None and row[0] == canonical_json_bytes(first)
    assert next_sequence == (2,)


def test_receipt_slot_rejects_changed_inputs(tmp_path: Path) -> None:
    authority = _capture_authority(tmp_path / "work.sqlite3")
    _issue(authority, "slot-1")

    with pytest.raises(SQLiteCaptureAuthorityConflict, match="changed"):
        _issue(authority, "slot-1", mapping_authority=_source_authority(2))


def test_identical_bindings_in_two_receipts_get_distinct_scopes(tmp_path: Path) -> None:
    authority = _capture_authority(tmp_path / "work.sqlite3")

    first = _issue(authority, "slot-1")
    second = _issue(authority, "slot-2")

    assert first.channel_bindings == second.channel_bindings
    assert first.acquisition_id == "local-capture-authority:1"
    assert second.acquisition_id == "local-capture-authority:2"
    assert first.capture_scope_id != second.capture_scope_id
    assert first.capture_scope_digest != second.capture_scope_digest


def test_capture_and_stream_extensions_share_scheduler_database(tmp_path: Path) -> None:
    database_path = tmp_path / "work.sqlite3"
    scheduler = SQLiteWorkScheduler(database_path)
    SQLiteStreamWorkLedger(scheduler)
    authority = SQLiteLocalCaptureAuthority(
        scheduler,
        capture_authority_id="local-capture-authority",
        capture_authority_epoch=3,
        capture_assignment_policy_version="assignment-v1",
    )

    issued = _issue(authority, "slot-1")
    reopened_ledger = SQLiteStreamWorkLedger(SQLiteWorkScheduler(database_path))

    assert reopened_ledger.database_path == database_path
    assert issued.capture_authority_id == "local-capture-authority"
    with sqlite3.connect(database_path) as connection:
        extensions = connection.execute(
            """
            SELECT extension_name FROM stream_extension_metadata
            ORDER BY extension_name
            """
        ).fetchall()
        scheduler_table = connection.execute(
            "SELECT name FROM sqlite_schema WHERE name = 'work_items'"
        ).fetchone()
    assert extensions == [("local-capture-authority",), ("stream-work-ledger",)]
    assert scheduler_table == ("work_items",)
