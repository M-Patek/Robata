from __future__ import annotations

import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import (
    LogicalNode,
    RunNodeDisposition,
    logical_node_from_semantic_digest,
)
from robata.contracts.revisions import (
    ImmutableNodeRevision,
    RevisionEligibility,
    create_immutable_node_revision,
)
from robata.ports.logical_node_registry import (
    LogicalNodeRegistryError,
    LogicalNodeRegistryErrorCode,
    PublishedRunNodeMembership,
)

_ATTACHED_AT = "2026-07-20T12:00:00Z"
_PUBLISHED_AT = "2026-07-20T13:00:00Z"
_SELECTED_AT = "2026-07-20T14:00:00Z"


def _uuid(number: int) -> str:
    return f"00000000-0000-5000-8000-{number:012x}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _node(seed: str = "default") -> LogicalNode:
    return logical_node_from_semantic_digest(
        node_type="CAMERA_VIDEO_EXPORT",
        key_namespace="camera-video-export:v1",
        semantic_sha256=semantic_sha256(
            {
                "export_config_sha256": _digest(f"config:{seed}"),
                "mapping_profile_sha256": _digest(f"mapping:{seed}"),
                "source_content_sha256": _digest(f"source:{seed}"),
            }
        ),
        identity_policy_version="camera-video-export-v1",
    )


def _attach(
    registry: LocalLogicalNodeRegistry,
    node: LogicalNode,
    *,
    run: int = 1,
    work: int = 101,
    attached_at: str = _ATTACHED_AT,
) -> PublishedRunNodeMembership:
    return registry.attach_run_node(
        node=node,
        run_id=_uuid(run),
        role="OUTPUT",
        first_work_item_id=_uuid(work),
        attached_at=attached_at,
        existing_node_disposition=RunNodeDisposition.REUSED,
    )


def _row_counts(registry: LocalLogicalNodeRegistry) -> tuple[int, int]:
    with sqlite3.connect(registry.database_path) as connection:
        node_row = connection.execute("SELECT COUNT(*) FROM logical_nodes").fetchone()
        membership_row = connection.execute("SELECT COUNT(*) FROM processing_run_nodes").fetchone()
    assert node_row is not None
    assert membership_row is not None
    return int(node_row[0]), int(membership_row[0])


def _revision(node: LogicalNode, number: int = 1) -> ImmutableNodeRevision:
    return create_immutable_node_revision(
        revision_id=_uuid(1_000 + number),
        subject_type=node.node_type,
        subject_id=node.node_logical_key,
        revision_key_namespace="camera-video-revision:v1",
        payload_sha256=_digest(f"payload:{number}"),
        lineage_sha256=_digest(f"lineage:{number}"),
        status_at_publication="READY",
        eligibility_at_publication=RevisionEligibility.ELIGIBLE,
        revision_policy_version="camera-video-revision-policy-v1",
        supersedes_revision_id=None,
        supersedes_revision_logical_key=None,
        published_at=_PUBLISHED_AT,
    )


def test_two_runs_share_one_node_with_created_and_reused_memberships(
    tmp_path: Path,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()

    created = _attach(registry, node, run=1, work=101)
    reused = _attach(
        registry,
        node,
        run=2,
        work=202,
        attached_at="2026-07-20T12:01:00Z",
    )

    assert created.membership.disposition is RunNodeDisposition.CREATED
    assert reused.membership.disposition is RunNodeDisposition.REUSED
    assert created.node_inserted is True
    assert reused.node_inserted is False
    assert _row_counts(registry) == (1, 2)
    assert set(registry.verify_node(*node.identity).memberships) == {
        created.membership,
        reused.membership,
    }


def test_exact_same_run_retry_preserves_original_created_membership(
    tmp_path: Path,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()

    first = _attach(registry, node)
    retry = _attach(registry, node)

    assert retry.membership == first.membership
    assert retry.membership.disposition is RunNodeDisposition.CREATED
    assert retry.node_inserted is False
    assert retry.membership_inserted is False
    assert _row_counts(registry) == (1, 1)


def test_restart_preserves_verified_node_and_run_memberships(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    node = _node()
    registry = LocalLogicalNodeRegistry(root)
    created = _attach(registry, node, run=1, work=101)
    reused = _attach(
        registry,
        node,
        run=2,
        work=202,
        attached_at="2026-07-20T12:01:00Z",
    )

    reopened = LocalLogicalNodeRegistry(root)

    assert reopened.lookup_node(*node.identity) == node
    assert set(reopened.verify_node(*node.identity).memberships) == {
        created.membership,
        reused.membership,
    }
    assert _row_counts(reopened) == (1, 2)


def test_concurrent_distinct_runs_converge_on_one_node(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    node = _node()
    registries = (LocalLogicalNodeRegistry(root), LocalLogicalNodeRegistry(root))
    barrier = threading.Barrier(2)

    def publish(index: int) -> PublishedRunNodeMembership:
        barrier.wait(timeout=10)
        return _attach(
            registries[index],
            node,
            run=index + 1,
            work=101 + index,
            attached_at=f"2026-07-20T12:00:0{index}Z",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(publish, range(2)))

    assert sorted(result.membership.disposition.value for result in results) == [
        "CREATED",
        "REUSED",
    ]
    assert sum(result.node_inserted is True for result in results) == 1
    assert all(result.membership_inserted is True for result in results)
    verified = LocalLogicalNodeRegistry(root).verify_node(*node.identity)
    assert {membership.run_id for membership in verified.memberships} == {_uuid(1), _uuid(2)}
    assert _row_counts(registries[0]) == (1, 2)


def test_membership_insert_failure_rolls_back_node_and_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()

    def fail_insert(
        _connection: sqlite3.Connection,
        _membership: object,
        _membership_bytes: bytes,
    ) -> None:
        raise sqlite3.OperationalError("injected membership insert failure")

    monkeypatch.setattr(registry, "_insert_membership", fail_insert)

    with pytest.raises(LogicalNodeRegistryError) as caught:
        _attach(registry, node)

    assert caught.value.code is LogicalNodeRegistryErrorCode.TRANSACTION_FAILED
    assert _row_counts(registry) == (0, 0)
    assert registry.lookup_node(*node.identity) is None


def test_noncanonical_node_record_tamper_fails_closed(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    _attach(registry, node)

    with sqlite3.connect(registry.database_path) as connection:
        row = connection.execute(
            "SELECT node_json FROM logical_nodes WHERE node_type = ? AND node_logical_key = ?",
            node.identity,
        ).fetchone()
        assert row is not None
        tampered = bytes(row[0]) + b" "
        connection.execute(
            """
            UPDATE logical_nodes
            SET node_json = ?, node_json_sha256 = ?
            WHERE node_type = ? AND node_logical_key = ?
            """,
            (tampered, hashlib.sha256(tampered).hexdigest(), *node.identity),
        )

    with pytest.raises(LogicalNodeRegistryError) as caught:
        registry.lookup_node(*node.identity)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


def test_normalized_membership_column_tamper_fails_closed(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    published = _attach(registry, node)

    with sqlite3.connect(registry.database_path) as connection:
        connection.execute(
            """
            UPDATE processing_run_nodes SET attached_at = ?
            WHERE run_id = ? AND node_type = ? AND node_logical_key = ? AND role = ?
            """,
            ("2026-07-20T23:59:59Z", *published.membership.identity),
        )

    with pytest.raises(LogicalNodeRegistryError) as caught:
        registry.lookup_membership(*published.membership.identity)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


def test_missing_creator_membership_invalidates_node(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    creator = _attach(registry, node, run=1, work=101)
    _attach(
        registry,
        node,
        run=2,
        work=202,
        attached_at="2026-07-20T12:01:00Z",
    )

    with sqlite3.connect(registry.database_path) as connection:
        connection.execute(
            """
            DELETE FROM processing_run_nodes
            WHERE run_id = ? AND node_type = ? AND node_logical_key = ? AND role = ?
            """,
            creator.membership.identity,
        )

    with pytest.raises(LogicalNodeRegistryError) as caught:
        registry.verify_node(*node.identity)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


def test_foreign_key_health_check_rejects_orphaned_membership(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    published = _attach(registry, node)

    with sqlite3.connect(registry.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "DELETE FROM logical_nodes WHERE node_type = ? AND node_logical_key = ?",
            node.identity,
        )

    assert _row_counts(registry) == (0, 1)
    with pytest.raises(LogicalNodeRegistryError) as caught:
        registry.list_run_memberships(published.membership.run_id)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


@pytest.mark.parametrize(
    ("table", "identity_column", "identity_number", "mutation"),
    [
        ("immutable_node_revisions", "revision_id", 1_001, "update"),
        ("immutable_node_revisions", "revision_id", 1_001, "delete"),
        ("selection_decisions", "selection_decision_id", 10_001, "update"),
        ("selection_decisions", "selection_decision_id", 10_001, "delete"),
    ],
)
def test_revision_history_is_database_enforced_append_only(
    tmp_path: Path,
    table: str,
    identity_column: str,
    identity_number: int,
    mutation: str,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    _attach(registry, node)
    revision = _revision(node)
    registry.publish_revision(revision)
    selected = registry.select_revision(
        subject_type=node.node_type,
        subject_id=node.node_logical_key,
        selected_revision_id=revision.revision_id,
        selection_decision_id=_uuid(10_001),
        selection_key_namespace="camera-video-selection:v1",
        expected_previous_selection_decision_id=None,
        selection_policy_version="camera-video-selection-policy-v1",
        selected_at=_SELECTED_AT,
    )

    statement = (
        f"UPDATE {table} SET schema_version = '1.0' WHERE {identity_column} = ?"
        if mutation == "update"
        else f"DELETE FROM {table} WHERE {identity_column} = ?"
    )
    with (
        sqlite3.connect(registry.database_path) as connection,
        pytest.raises(sqlite3.IntegrityError),
    ):
        connection.execute(statement, (_uuid(identity_number),))

    verified = registry.verify_subject(*node.identity)
    assert verified.revisions == (revision,)
    assert verified.decisions == (selected.decision,)
    assert verified.current == selected.current


def test_revision_and_current_selection_survive_restart(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry = LocalLogicalNodeRegistry(root)
    node = _node()
    _attach(registry, node)
    revision = _revision(node)
    registry.publish_revision(revision)
    selected = registry.select_revision(
        subject_type=node.node_type,
        subject_id=node.node_logical_key,
        selected_revision_id=revision.revision_id,
        selection_decision_id=_uuid(10_001),
        selection_key_namespace="camera-video-selection:v1",
        expected_previous_selection_decision_id=None,
        selection_policy_version="camera-video-selection-policy-v1",
        selected_at=_SELECTED_AT,
    )

    reopened = LocalLogicalNodeRegistry(root)

    assert reopened.lookup_revision(*node.identity, revision.revision_id) == revision
    assert reopened.lookup_current_selection(*node.identity) == selected.current
    verified = reopened.verify_subject(*node.identity)
    assert verified.revisions == (revision,)
    assert verified.decisions == (selected.decision,)
    assert verified.current == selected.current
    assert reopened.rebuild_current_projection() == (selected.current,)
