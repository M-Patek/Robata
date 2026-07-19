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
from robata.ports.logical_node_registry import (
    LogicalNodeRegistryError,
    LogicalNodeRegistryErrorCode,
    PublishedRunNodeMembership,
)

_ATTACHED_AT = "2026-07-18T12:00:00Z"


def _uuid(number: int) -> str:
    return f"00000000-0000-5000-8000-{number:012x}"


def _node(
    seed: str = "default",
    *,
    node_type: str = "CAMERA_VIDEO_EXPORT",
    identity_policy_version: str = "camera-video-export-v1",
) -> LogicalNode:
    return logical_node_from_semantic_digest(
        node_type=node_type,
        key_namespace="camera-video-export:v1",
        semantic_sha256=semantic_sha256(
            {
                "export_config_sha256": hashlib.sha256(f"config:{seed}".encode()).hexdigest(),
                "mapping_profile_sha256": hashlib.sha256(f"mapping:{seed}".encode()).hexdigest(),
                "source_content_sha256": hashlib.sha256(f"source:{seed}".encode()).hexdigest(),
            }
        ),
        identity_policy_version=identity_policy_version,
    )


def _attach(
    registry: LocalLogicalNodeRegistry,
    node: LogicalNode,
    *,
    run: int = 1,
    role: str = "OUTPUT",
    work: int = 101,
    attached_at: str = _ATTACHED_AT,
    disposition: RunNodeDisposition = RunNodeDisposition.REUSED,
) -> PublishedRunNodeMembership:
    return registry.attach_run_node(
        node=node,
        run_id=_uuid(run),
        role=role,
        first_work_item_id=_uuid(work),
        attached_at=attached_at,
        existing_node_disposition=disposition,
    )


def _row_counts(registry: LocalLogicalNodeRegistry) -> tuple[int, int]:
    with sqlite3.connect(registry.database_path) as connection:
        node_count = connection.execute("SELECT COUNT(*) FROM logical_nodes").fetchone()
        membership_count = connection.execute(
            "SELECT COUNT(*) FROM processing_run_nodes"
        ).fetchone()
    assert node_count is not None
    assert membership_count is not None
    return int(node_count[0]), int(membership_count[0])


def test_first_attach_derives_created_and_persists_both_rows(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()

    published = _attach(registry, node)

    assert published.node == node
    assert published.membership.disposition is RunNodeDisposition.CREATED
    assert published.node_inserted is True
    assert published.membership_inserted is True
    assert _row_counts(registry) == (1, 1)
    assert registry.verify_node(*node.identity).memberships == (published.membership,)


def test_exact_membership_retry_returns_original_created_without_inserts(
    tmp_path: Path,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    first = _attach(registry, node)

    replay = _attach(registry, node)

    assert replay.node == first.node
    assert replay.membership == first.membership
    assert replay.membership.disposition is RunNodeDisposition.CREATED
    assert replay.node_inserted is False
    assert replay.membership_inserted is False
    assert _row_counts(registry) == (1, 1)


def test_second_run_reuses_one_node_and_preserves_both_memberships(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    creator = _attach(registry, node, run=1, work=101)

    reused = _attach(
        registry,
        node,
        run=2,
        work=202,
        attached_at="2026-07-18T12:01:00Z",
    )

    assert reused.membership.disposition is RunNodeDisposition.REUSED
    assert reused.node_inserted is False
    assert reused.membership_inserted is True
    assert _row_counts(registry) == (1, 2)
    verified = registry.verify_node(*node.identity)
    assert verified.node == node
    assert set(verified.memberships) == {creator.membership, reused.membership}


def test_same_run_and_node_can_have_distinct_role_memberships(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    output = _attach(registry, node, role="OUTPUT", work=101)

    audit = _attach(
        registry,
        node,
        role="AUDIT_INPUT",
        work=102,
        attached_at="2026-07-18T12:00:01Z",
    )

    assert output.membership.disposition is RunNodeDisposition.CREATED
    assert audit.membership.disposition is RunNodeDisposition.REUSED
    assert output.membership.identity != audit.membership.identity
    assert registry.lookup_membership(_uuid(1), *node.identity, "OUTPUT") == output.membership
    assert registry.lookup_membership(_uuid(1), *node.identity, "AUDIT_INPUT") == audit.membership
    assert _row_counts(registry) == (1, 2)


def test_invalidated_and_observed_attach_to_existing_node_without_mutation(
    tmp_path: Path,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    creator = _attach(registry, node, run=1, work=101)

    invalidated = _attach(
        registry,
        node,
        run=2,
        work=202,
        attached_at="2026-07-18T12:01:00Z",
        disposition=RunNodeDisposition.INVALIDATED,
    )
    observed = _attach(
        registry,
        node,
        run=3,
        role="AUDIT_INPUT",
        work=303,
        attached_at="2026-07-18T12:02:00Z",
        disposition=RunNodeDisposition.OBSERVED,
    )

    assert invalidated.membership.disposition is RunNodeDisposition.INVALIDATED
    assert observed.membership.disposition is RunNodeDisposition.OBSERVED
    assert registry.lookup_node(*node.identity) == node
    assert registry.lookup_membership(*creator.membership.identity) == creator.membership
    assert set(registry.list_node_memberships(*node.identity)) == {
        creator.membership,
        invalidated.membership,
        observed.membership,
    }
    assert _row_counts(registry) == (1, 3)


@pytest.mark.parametrize(
    "disposition",
    [RunNodeDisposition.INVALIDATED, RunNodeDisposition.OBSERVED],
)
def test_noncreating_dispositions_reject_missing_node_without_rows(
    tmp_path: Path,
    disposition: RunNodeDisposition,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")

    with pytest.raises(LogicalNodeRegistryError) as caught:
        _attach(registry, _node(), disposition=disposition)

    assert caught.value.code is LogicalNodeRegistryErrorCode.NODE_NOT_FOUND
    assert _row_counts(registry) == (0, 0)


@pytest.mark.parametrize(
    "disposition",
    [RunNodeDisposition.CREATED, "REUSED"],
    ids=["created-is-derived", "string-is-not-coerced"],
)
def test_attach_rejects_created_or_untyped_existing_disposition_without_rows(
    tmp_path: Path,
    disposition: object,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()

    with pytest.raises(LogicalNodeRegistryError) as caught:
        registry.attach_run_node(
            node=node,
            run_id=_uuid(1),
            role="OUTPUT",
            first_work_item_id=_uuid(101),
            attached_at=_ATTACHED_AT,
            existing_node_disposition=disposition,  # type: ignore[arg-type]
        )

    assert caught.value.code is LogicalNodeRegistryErrorCode.INVALID_REQUEST
    assert _row_counts(registry) == (0, 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"work": 999},
        {"attached_at": "2026-07-18T12:00:01Z"},
        {"disposition": RunNodeDisposition.OBSERVED},
    ],
    ids=["first-work-item", "attached-at", "disposition"],
)
def test_existing_membership_rejects_changed_immutable_content(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    original = _attach(registry, node)
    request: dict[str, object] = {
        "work": 101,
        "attached_at": _ATTACHED_AT,
        "disposition": RunNodeDisposition.REUSED,
    }
    request.update(changes)

    with pytest.raises(LogicalNodeRegistryError) as caught:
        _attach(
            registry,
            node,
            work=int(request["work"]),
            attached_at=str(request["attached_at"]),
            disposition=request["disposition"],  # type: ignore[arg-type]
        )

    assert caught.value.code is LogicalNodeRegistryErrorCode.MEMBERSHIP_CONFLICT
    assert registry.lookup_membership(*original.membership.identity) == original.membership
    assert _row_counts(registry) == (1, 1)


def test_existing_node_identity_rejects_different_policy_content(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    _attach(registry, node)
    conflicting = node.model_copy(update={"identity_policy_version": "different-policy-v2"})

    with pytest.raises(LogicalNodeRegistryError) as caught:
        _attach(registry, conflicting, run=2, work=202)

    assert caught.value.code is LogicalNodeRegistryErrorCode.NODE_CONFLICT
    assert registry.lookup_node(*node.identity) == node
    assert _row_counts(registry) == (1, 1)


def test_run_and_node_queries_have_canonical_order_independent_of_insert_order(
    tmp_path: Path,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    alpha_first = _node("alpha-first", node_type="ALPHA_NODE")
    alpha_second = _node("alpha-second", node_type="ALPHA_NODE")
    zeta = _node("zeta", node_type="ZETA_NODE")

    _attach(registry, zeta, run=50, role="OUTPUT", work=501)
    _attach(registry, alpha_second, run=50, role="INPUT", work=502)
    _attach(registry, alpha_first, run=50, role="Z_ROLE", work=503)
    _attach(registry, alpha_first, run=50, role="A_ROLE", work=504)

    run_memberships = registry.list_run_memberships(_uuid(50))
    run_order = tuple(
        (item.node_type, item.node_logical_key, item.role) for item in run_memberships
    )
    assert run_order == tuple(sorted(run_order))

    _attach(registry, alpha_first, run=60, role="B_ROLE", work=601)
    _attach(registry, alpha_first, run=40, role="M_ROLE", work=401)
    node_memberships = registry.list_node_memberships(*alpha_first.identity)
    node_order = tuple((item.run_id, item.role) for item in node_memberships)
    assert node_order == tuple(sorted(node_order))
    assert registry.verify_node(*alpha_first.identity).memberships == node_memberships


def test_reopening_registry_preserves_verified_node_and_memberships(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    node = _node()
    registry = LocalLogicalNodeRegistry(root)
    created = _attach(registry, node, run=1, work=101)
    reused = _attach(registry, node, run=2, work=202)

    reopened = LocalLogicalNodeRegistry(root)

    verified = reopened.verify_node(*node.identity)
    assert verified.node == node
    assert set(verified.memberships) == {created.membership, reused.membership}
    assert _row_counts(reopened) == (1, 2)


def test_failure_before_commit_rolls_back_node_and_membership(
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


def test_commit_after_success_error_recovers_the_committed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    registry = LocalLogicalNodeRegistry(root)
    node = _node()

    def commit_then_report_error(connection: sqlite3.Connection) -> None:
        connection.commit()
        raise sqlite3.OperationalError("injected uncertain commit result")

    monkeypatch.setattr(registry, "_commit", commit_then_report_error)

    published = _attach(registry, node)

    assert published.membership.disposition is RunNodeDisposition.CREATED
    assert published.node_inserted is None
    assert published.membership_inserted is None
    assert LocalLogicalNodeRegistry(root).verify_node(*node.identity).memberships == (
        published.membership,
    )


def test_commit_error_before_commit_rolls_back_and_new_adapter_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    registry = LocalLogicalNodeRegistry(root)
    node = _node()

    def fail_before_commit(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected pre-commit failure")

    monkeypatch.setattr(registry, "_commit", fail_before_commit)

    with pytest.raises(LogicalNodeRegistryError) as caught:
        _attach(registry, node)

    assert caught.value.code is LogicalNodeRegistryErrorCode.TRANSACTION_FAILED
    assert _row_counts(registry) == (0, 0)

    retry_registry = LocalLogicalNodeRegistry(root)
    retry = _attach(retry_registry, node)
    assert retry.membership.disposition is RunNodeDisposition.CREATED
    assert retry.node_inserted is True
    assert retry.membership_inserted is True
    assert _row_counts(retry_registry) == (1, 1)


def test_uncertain_commit_recovery_does_not_claim_competitor_inserts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    node = _node()
    registry_a = LocalLogicalNodeRegistry(root)
    registry_b = LocalLogicalNodeRegistry(root)
    competitor_results: list[PublishedRunNodeMembership] = []
    original_recover = registry_a._recover_uncertain_commit

    def fail_before_commit(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected pre-commit failure for A")

    def competitor_then_recover(
        recovery_node: LogicalNode,
        expected_membership: object,
    ) -> object:
        competitor_results.append(_attach(registry_b, node))
        return original_recover(recovery_node, expected_membership)  # type: ignore[arg-type]

    monkeypatch.setattr(registry_a, "_commit", fail_before_commit)
    monkeypatch.setattr(registry_a, "_recover_uncertain_commit", competitor_then_recover)

    recovered = _attach(registry_a, node)

    assert len(competitor_results) == 1
    competitor = competitor_results[0]
    assert competitor.node_inserted is True
    assert competitor.membership_inserted is True
    assert competitor.membership.disposition is RunNodeDisposition.CREATED
    assert recovered.node == competitor.node
    assert recovered.membership == competitor.membership
    assert recovered.node_inserted is None
    assert recovered.membership_inserted is None
    assert _row_counts(registry_a) == (1, 1)


def test_concurrent_runs_converge_on_one_node_with_created_and_reused(
    tmp_path: Path,
) -> None:
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
            attached_at=f"2026-07-18T12:00:0{index}Z",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(publish, range(2)))

    assert sorted(result.membership.disposition.value for result in results) == [
        "CREATED",
        "REUSED",
    ]
    assert sum(result.node_inserted for result in results) == 1
    assert all(result.membership_inserted for result in results)
    verified = LocalLogicalNodeRegistry(root).verify_node(*node.identity)
    assert {membership.run_id for membership in verified.memberships} == {_uuid(1), _uuid(2)}
    assert _row_counts(registries[0]) == (1, 2)


def test_concurrent_exact_membership_attach_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    node = _node()
    registries = (LocalLogicalNodeRegistry(root), LocalLogicalNodeRegistry(root))
    barrier = threading.Barrier(2)

    def publish(index: int) -> PublishedRunNodeMembership:
        barrier.wait(timeout=10)
        return _attach(registries[index], node)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(publish, range(2)))

    assert all(result.membership.disposition is RunNodeDisposition.CREATED for result in results)
    assert results[0].membership == results[1].membership
    assert sum(result.node_inserted for result in results) == 1
    assert sum(result.membership_inserted for result in results) == 1
    assert _row_counts(registries[0]) == (1, 1)


@pytest.mark.parametrize("tamper", ["json", "digest", "normalized"])
def test_verified_node_lookup_rejects_node_record_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    _attach(registry, node)

    with sqlite3.connect(registry.database_path) as connection:
        if tamper == "json":
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
        elif tamper == "digest":
            connection.execute(
                """
                UPDATE logical_nodes SET node_json_sha256 = ?
                WHERE node_type = ? AND node_logical_key = ?
                """,
                ("0" * 64, *node.identity),
            )
        else:
            connection.execute(
                """
                UPDATE logical_nodes SET identity_policy_version = ?
                WHERE node_type = ? AND node_logical_key = ?
                """,
                ("tampered-policy", *node.identity),
            )

    with pytest.raises(LogicalNodeRegistryError) as caught:
        registry.lookup_node(*node.identity)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


@pytest.mark.parametrize("tamper", ["json", "digest", "normalized"])
def test_verified_membership_lookup_rejects_membership_record_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    published = _attach(registry, node)
    identity = published.membership.identity

    with sqlite3.connect(registry.database_path) as connection:
        if tamper == "json":
            row = connection.execute(
                """
                SELECT membership_json FROM processing_run_nodes
                WHERE run_id = ? AND node_type = ? AND node_logical_key = ? AND role = ?
                """,
                identity,
            ).fetchone()
            assert row is not None
            tampered = bytes(row[0]) + b" "
            connection.execute(
                """
                UPDATE processing_run_nodes
                SET membership_json = ?, membership_json_sha256 = ?
                WHERE run_id = ? AND node_type = ? AND node_logical_key = ? AND role = ?
                """,
                (tampered, hashlib.sha256(tampered).hexdigest(), *identity),
            )
        elif tamper == "digest":
            connection.execute(
                """
                UPDATE processing_run_nodes SET membership_json_sha256 = ?
                WHERE run_id = ? AND node_type = ? AND node_logical_key = ? AND role = ?
                """,
                ("0" * 64, *identity),
            )
        else:
            connection.execute(
                """
                UPDATE processing_run_nodes SET attached_at = ?
                WHERE run_id = ? AND node_type = ? AND node_logical_key = ? AND role = ?
                """,
                ("2026-07-18T23:59:59Z", *identity),
            )

    with pytest.raises(LogicalNodeRegistryError) as caught:
        registry.lookup_membership(*identity)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


def test_removing_the_created_membership_invalidates_verified_node(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    creator = _attach(registry, node, run=1, work=101)
    _attach(registry, node, run=2, work=202)

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


def test_foreign_key_restricts_deleting_node_with_membership(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")
    node = _node()
    _attach(registry, node)

    with sqlite3.connect(registry.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM logical_nodes WHERE node_type = ? AND node_logical_key = ?",
                node.identity,
            )

    assert registry.verify_node(*node.identity).node == node


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


def test_unsupported_database_user_version_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    root.mkdir()
    with sqlite3.connect(root / "logical-nodes.sqlite3") as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(LogicalNodeRegistryError) as caught:
        LocalLogicalNodeRegistry(root)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


def test_partial_existing_schema_fails_closed_instead_of_being_repaired(
    tmp_path: Path,
) -> None:
    root = tmp_path / "registry"
    root.mkdir()
    with sqlite3.connect(root / "logical-nodes.sqlite3") as connection:
        connection.execute("CREATE TABLE logical_nodes (bogus TEXT NOT NULL)")
        connection.execute("PRAGMA user_version = 1")

    with pytest.raises(LogicalNodeRegistryError) as caught:
        LocalLogicalNodeRegistry(root)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


def test_full_schema_with_wrong_declared_column_type_fails_during_initialization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "registry"
    registry = LocalLogicalNodeRegistry(root)

    with sqlite3.connect(registry.database_path) as connection:
        row = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'logical_nodes'"
        ).fetchone()
        assert row is not None
        original_sql = str(row[0])
        tampered_sql = original_sql.replace(
            "semantic_sha256 TEXT NOT NULL",
            "semantic_sha256 INTEGER NOT NULL",
            1,
        )
        assert tampered_sql != original_sql
        schema_version_row = connection.execute("PRAGMA schema_version").fetchone()
        assert schema_version_row is not None
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_schema SET sql = ? WHERE type = 'table' AND name = 'logical_nodes'",
            (tampered_sql,),
        )
        connection.execute(f"PRAGMA schema_version = {int(schema_version_row[0]) + 1}")
        connection.execute("PRAGMA writable_schema = OFF")

    with pytest.raises(LogicalNodeRegistryError) as caught:
        LocalLogicalNodeRegistry(root)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR
