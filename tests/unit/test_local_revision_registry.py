from __future__ import annotations

import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.logical_nodes import (
    LogicalNode,
    RunNodeDisposition,
    logical_node_from_semantic_digest,
)
from robata.contracts.revisions import (
    CurrentSelection,
    ImmutableNodeRevision,
    RevisionEligibility,
    create_immutable_node_revision,
)
from robata.ports.logical_node_registry import (
    LogicalNodeRegistryError,
    LogicalNodeRegistryErrorCode,
)
from robata.ports.revision_registry import (
    PublishedSelection,
    RevisionSelectionRegistryError,
    RevisionSelectionRegistryErrorCode,
)

_PUBLISHED_AT = "2026-07-18T12:00:00Z"
_SELECTED_AT = "2026-07-18T13:00:00Z"


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


def _attach_subject(registry: LocalLogicalNodeRegistry, node: LogicalNode, run: int = 1) -> None:
    registry.attach_run_node(
        node=node,
        run_id=_uuid(run),
        role="OUTPUT",
        first_work_item_id=_uuid(100 + run),
        attached_at=f"2026-07-18T12:00:{run:02d}Z",
        existing_node_disposition=RunNodeDisposition.REUSED,
    )


def _registry_with_subject(
    root: Path,
    *,
    seed: str = "default",
    run: int = 1,
) -> tuple[LocalLogicalNodeRegistry, LogicalNode]:
    registry = LocalLogicalNodeRegistry(root)
    node = _node(seed)
    _attach_subject(registry, node, run)
    return registry, node


def _revision(
    node: LogicalNode,
    number: int,
    *,
    semantic_number: int | None = None,
    eligibility: RevisionEligibility = RevisionEligibility.ELIGIBLE,
    supersedes: ImmutableNodeRevision | None = None,
    supersedes_revision_id: str | None = None,
    supersedes_revision_logical_key: str | None = None,
    published_at: str | None = None,
) -> ImmutableNodeRevision:
    semantic_number = number if semantic_number is None else semantic_number
    if supersedes is not None:
        supersedes_revision_id = supersedes.revision_id
        supersedes_revision_logical_key = supersedes.revision_logical_key
    return create_immutable_node_revision(
        revision_id=_uuid(1_000 + number),
        subject_type=node.node_type,
        subject_id=node.node_logical_key,
        revision_key_namespace="camera-video-revision:v1",
        payload_sha256=_digest(f"payload:{semantic_number}"),
        lineage_sha256=_digest(f"lineage:{semantic_number}"),
        status_at_publication="READY",
        eligibility_at_publication=eligibility,
        revision_policy_version="camera-video-revision-policy-v1",
        supersedes_revision_id=supersedes_revision_id,
        supersedes_revision_logical_key=supersedes_revision_logical_key,
        published_at=published_at or _PUBLISHED_AT,
    )


def _select(
    registry: LocalLogicalNodeRegistry,
    node: LogicalNode,
    revision: ImmutableNodeRevision,
    decision_number: int,
    *,
    expected_previous: str | None = None,
    selected_at: str = _SELECTED_AT,
) -> PublishedSelection:
    return registry.select_revision(
        subject_type=node.node_type,
        subject_id=node.node_logical_key,
        selected_revision_id=revision.revision_id,
        selection_decision_id=_uuid(10_000 + decision_number),
        selection_key_namespace="camera-video-selection:v1",
        expected_previous_selection_decision_id=expected_previous,
        selection_policy_version="camera-video-selection-policy-v1",
        selected_at=selected_at,
    )


def _revision_counts(registry: LocalLogicalNodeRegistry) -> tuple[int, int, int]:
    with sqlite3.connect(registry.database_path) as connection:
        rows = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            for table in (
                "immutable_node_revisions",
                "selection_decisions",
                "current_selections",
            )
        )
    assert all(row is not None for row in rows)
    return tuple(int(row[0]) for row in rows if row is not None)  # type: ignore[return-value]


def _assert_revision_error(
    caught: pytest.ExceptionInfo[RevisionSelectionRegistryError],
    code: RevisionSelectionRegistryErrorCode,
) -> None:
    assert caught.value.code is code


def test_publish_revision_persists_and_resolves_all_verified_reads(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    revision = _revision(node, 1)

    published = registry.publish_revision(revision)

    assert published.revision == revision
    assert published.inserted is True
    assert registry.lookup_revision(*node.identity, revision.revision_id) == revision
    assert registry.list_revisions(*node.identity) == (revision,)
    verified = registry.verify_subject(*node.identity)
    assert verified.node == node
    assert verified.revisions == (revision,)
    assert verified.decisions == ()
    assert verified.current is None
    assert _revision_counts(registry) == (1, 0, 0)


def test_publish_semantic_retry_returns_original_uuid_and_audit_time(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    first = _revision(node, 1)
    retry = _revision(
        node,
        2,
        semantic_number=1,
        published_at="2026-07-18T12:59:59Z",
    )

    original = registry.publish_revision(first)
    replay = registry.publish_revision(retry)

    assert replay.revision == original.revision
    assert replay.revision.revision_id == first.revision_id
    assert replay.revision.published_at == first.published_at
    assert replay.inserted is False
    assert _revision_counts(registry) == (1, 0, 0)


def test_publish_same_revision_id_with_different_content_conflicts(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    first = _revision(node, 1)
    conflicting = _revision(node, 1, semantic_number=2)
    registry.publish_revision(first)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        registry.publish_revision(conflicting)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.REVISION_CONFLICT)
    assert registry.list_revisions(*node.identity) == (first,)


def test_publish_revision_can_supersede_revision_under_same_subject(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    first = _revision(node, 1)
    second = _revision(node, 2, supersedes=first)
    registry.publish_revision(first)

    published = registry.publish_revision(second)

    assert published.revision == second
    assert set(registry.list_revisions(*node.identity)) == {first, second}


def test_publish_revision_rejects_missing_superseded_revision(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    missing_target = _revision(node, 99)
    candidate = _revision(
        node,
        2,
        supersedes_revision_id=missing_target.revision_id,
        supersedes_revision_logical_key=missing_target.revision_logical_key,
    )

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        registry.publish_revision(candidate)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.REVISION_NOT_FOUND)
    assert _revision_counts(registry) == (0, 0, 0)


def test_publish_revision_rejects_cross_subject_supersedes(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry, first_node = _registry_with_subject(root, seed="first", run=1)
    second_node = _node("second")
    _attach_subject(registry, second_node, run=2)
    foreign = _revision(first_node, 1)
    registry.publish_revision(foreign)
    candidate = _revision(second_node, 2, supersedes=foreign)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        registry.publish_revision(candidate)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.REVISION_CONFLICT)
    assert registry.list_revisions(*second_node.identity) == ()


def test_publish_revision_rejects_self_supersedes(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    referenced_key = _revision(node, 99).revision_logical_key
    candidate = _revision(
        node,
        1,
        supersedes_revision_id=_uuid(1_001),
        supersedes_revision_logical_key=referenced_key,
    )
    assert candidate.supersedes_revision_id == candidate.revision_id

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        registry.publish_revision(candidate)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.REVISION_CONFLICT)
    assert _revision_counts(registry) == (0, 0, 0)


def test_initial_selection_appends_decision_and_projects_current(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    revision = _revision(node, 1)
    registry.publish_revision(revision)

    selected = _select(registry, node, revision, 1)

    assert selected.decision_inserted is True
    assert selected.projection_advanced is True
    assert selected.decision.selection_sequence == 1
    assert selected.decision.previous_selection_decision_id is None
    assert selected.current.selected_revision_id == revision.revision_id
    assert selected.current.selection_decision_id == selected.decision.selection_decision_id
    assert (
        registry.lookup_selection_decision(
            *node.identity,
            selected.decision.selection_decision_id,
        )
        == selected.decision
    )
    assert registry.lookup_current_selection(*node.identity) == selected.current
    assert registry.list_selection_decisions(*node.identity) == (selected.decision,)
    assert _revision_counts(registry) == (1, 1, 1)


def test_selection_rejects_ineligible_revision_without_writes(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    revision = _revision(node, 1, eligibility=RevisionEligibility.INELIGIBLE)
    registry.publish_revision(revision)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        _select(registry, node, revision, 1)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.REVISION_INELIGIBLE)
    assert _revision_counts(registry) == (1, 0, 0)


def test_ineligible_revision_precedes_stale_cas_without_selection_side_effects(
    tmp_path: Path,
) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    eligible = _revision(node, 1)
    ineligible = _revision(node, 2, eligibility=RevisionEligibility.INELIGIBLE)
    registry.publish_revision(eligible)
    registry.publish_revision(ineligible)
    existing = _select(registry, node, eligible, 1)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        _select(registry, node, ineligible, 2, expected_previous=None)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.REVISION_INELIGIBLE)
    assert registry.list_selection_decisions(*node.identity) == (existing.decision,)
    assert registry.lookup_current_selection(*node.identity) == existing.current
    assert _revision_counts(registry) == (2, 1, 1)


def test_selection_rejects_missing_revision_without_writes(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    missing = _revision(node, 99)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        _select(registry, node, missing, 1)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.REVISION_NOT_FOUND)
    assert _revision_counts(registry) == (0, 0, 0)


def test_stale_selection_cas_preserves_existing_decision_and_current(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    first_revision = _revision(node, 1)
    second_revision = _revision(node, 2)
    registry.publish_revision(first_revision)
    registry.publish_revision(second_revision)
    first = _select(registry, node, first_revision, 1)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        _select(registry, node, second_revision, 2, expected_previous=None)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.STALE_SELECTION)
    assert registry.list_selection_decisions(*node.identity) == (first.decision,)
    assert registry.lookup_current_selection(*node.identity) == first.current
    assert _revision_counts(registry) == (2, 1, 1)


def test_second_selection_preserves_revision_bytes_and_history_and_retry_does_not_rewind(
    tmp_path: Path,
) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    first_revision = _revision(node, 1)
    second_revision = _revision(node, 2, supersedes=first_revision)
    registry.publish_revision(first_revision)
    registry.publish_revision(second_revision)
    first = _select(registry, node, first_revision, 1)
    with sqlite3.connect(registry.database_path) as connection:
        revision_bytes_before = tuple(
            bytes(row[0])
            for row in connection.execute(
                "SELECT revision_json FROM immutable_node_revisions ORDER BY revision_id"
            )
        )

    second = _select(
        registry,
        node,
        second_revision,
        2,
        expected_previous=first.decision.selection_decision_id,
        selected_at="2026-07-18T13:01:00Z",
    )
    replay = _select(
        registry,
        node,
        first_revision,
        1,
        selected_at="2026-07-18T23:59:59Z",
    )

    assert second.decision.selection_sequence == 2
    assert second.decision.previous_selection_decision_id == first.decision.selection_decision_id
    assert registry.list_selection_decisions(*node.identity) == (
        first.decision,
        second.decision,
    )
    assert replay.decision == first.decision
    assert replay.current == second.current
    assert replay.decision_inserted is False
    assert replay.projection_advanced is False
    assert registry.lookup_current_selection(*node.identity) == second.current
    with sqlite3.connect(registry.database_path) as connection:
        revision_bytes_after = tuple(
            bytes(row[0])
            for row in connection.execute(
                "SELECT revision_json FROM immutable_node_revisions ORDER BY revision_id"
            )
        )
    assert revision_bytes_after == revision_bytes_before
    assert _revision_counts(registry) == (2, 2, 1)


def _current_for_decision(selection: PublishedSelection) -> CurrentSelection:
    decision = selection.decision
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


@pytest.mark.parametrize("drift", ["delete", "stale", "corrupt"])
def test_rebuild_repairs_missing_stale_or_corrupt_current_projection(
    tmp_path: Path,
    drift: str,
) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    first_revision = _revision(node, 1)
    second_revision = _revision(node, 2)
    registry.publish_revision(first_revision)
    registry.publish_revision(second_revision)
    first = _select(registry, node, first_revision, 1)
    second = _select(
        registry,
        node,
        second_revision,
        2,
        expected_previous=first.decision.selection_decision_id,
        selected_at="2026-07-18T13:01:00Z",
    )

    with sqlite3.connect(registry.database_path) as connection:
        if drift == "delete":
            connection.execute(
                "DELETE FROM current_selections WHERE subject_type = ? AND subject_id = ?",
                node.identity,
            )
        elif drift == "stale":
            stale = _current_for_decision(first)
            raw = canonical_json_bytes(stale)
            connection.execute(
                """
                UPDATE current_selections
                SET schema_version = ?, selected_revision_id = ?, selection_decision_id = ?,
                    selection_policy_version = ?, projection_version = ?, selected_at = ?,
                    current_json = ?, current_json_sha256 = ?
                WHERE subject_type = ? AND subject_id = ?
                """,
                (
                    stale.schema_version,
                    stale.selected_revision_id,
                    stale.selection_decision_id,
                    stale.selection_policy_version,
                    stale.projection_version,
                    stale.selected_at,
                    raw,
                    hashlib.sha256(raw).hexdigest(),
                    *node.identity,
                ),
            )
        else:
            connection.execute(
                """
                UPDATE current_selections SET current_json_sha256 = ?
                WHERE subject_type = ? AND subject_id = ?
                """,
                ("0" * 64, *node.identity),
            )

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        registry.lookup_current_selection(*node.identity)
    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR)

    rebuilt = registry.rebuild_current_projection()

    assert rebuilt == (second.current,)
    assert registry.lookup_current_selection(*node.identity) == second.current
    assert registry.list_selection_decisions(*node.identity) == (
        first.decision,
        second.decision,
    )


def test_rebuild_is_deterministic_for_multiple_subjects(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry, first_node = _registry_with_subject(root, seed="z", run=1)
    second_node = _node("a")
    _attach_subject(registry, second_node, run=2)
    first_revision = _revision(first_node, 1)
    second_revision = _revision(second_node, 2)
    registry.publish_revision(first_revision)
    registry.publish_revision(second_revision)
    first = _select(registry, first_node, first_revision, 1)
    second = _select(registry, second_node, second_revision, 2)
    expected = tuple(
        current
        for _, current in sorted(
            ((first_node.identity, first.current), (second_node.identity, second.current))
        )
    )

    assert registry.rebuild_current_projection() == expected
    assert registry.rebuild_current_projection() == expected


def test_rebuild_returns_commit_snapshot_when_selection_advances_after_lock_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    setup, node = _registry_with_subject(root)
    first_revision = _revision(node, 1)
    second_revision = _revision(node, 2)
    setup.publish_revision(first_revision)
    setup.publish_revision(second_revision)
    first = _select(setup, node, first_revision, 1)
    rebuilder = LocalLogicalNodeRegistry(root)
    writer = LocalLogicalNodeRegistry(root)
    rebuild_committed = threading.Event()
    selection_completed = threading.Event()

    def advance_selection() -> PublishedSelection:
        assert rebuild_committed.wait(timeout=10)
        try:
            return _select(
                writer,
                node,
                second_revision,
                2,
                expected_previous=first.decision.selection_decision_id,
                selected_at="2026-07-18T13:01:00Z",
            )
        finally:
            selection_completed.set()

    def commit_then_wait_for_selection(connection: sqlite3.Connection) -> None:
        connection.commit()
        rebuild_committed.set()
        assert selection_completed.wait(timeout=10)

    monkeypatch.setattr(rebuilder, "_commit", commit_then_wait_for_selection)

    with ThreadPoolExecutor(max_workers=1) as executor:
        selection_future = executor.submit(advance_selection)
        rebuilt = rebuilder.rebuild_current_projection()
        second = selection_future.result(timeout=10)

    assert rebuilt == (first.current,)
    after_commit = rebuilder.verify_subject(*node.identity)
    assert after_commit.decisions == (first.decision, second.decision)
    assert after_commit.current == second.current


def test_concurrent_initial_selections_have_one_cas_winner(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    setup, node = _registry_with_subject(root)
    revisions = (_revision(node, 1), _revision(node, 2))
    for revision in revisions:
        setup.publish_revision(revision)
    registries = (LocalLogicalNodeRegistry(root), LocalLogicalNodeRegistry(root))
    barrier = threading.Barrier(2)

    def select(index: int) -> PublishedSelection | RevisionSelectionRegistryError:
        barrier.wait(timeout=10)
        try:
            return _select(registries[index], node, revisions[index], index + 1)
        except RevisionSelectionRegistryError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(select, range(2)))

    successes = tuple(item for item in outcomes if isinstance(item, PublishedSelection))
    failures = tuple(item for item in outcomes if isinstance(item, RevisionSelectionRegistryError))
    assert len(successes) == 1
    assert len(failures) == 1
    assert failures[0].code is RevisionSelectionRegistryErrorCode.STALE_SELECTION
    verified = LocalLogicalNodeRegistry(root).verify_subject(*node.identity)
    assert verified.decisions == (successes[0].decision,)
    assert verified.current == successes[0].current
    assert _revision_counts(setup) == (2, 1, 1)


def test_concurrent_exact_initial_selection_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    setup, node = _registry_with_subject(root)
    revision = _revision(node, 1)
    setup.publish_revision(revision)
    registries = (LocalLogicalNodeRegistry(root), LocalLogicalNodeRegistry(root))
    barrier = threading.Barrier(2)

    def select(index: int) -> PublishedSelection:
        barrier.wait(timeout=10)
        return _select(registries[index], node, revision, 1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(select, range(2)))

    assert results[0].decision == results[1].decision
    assert results[0].current == results[1].current
    assert sum(result.decision_inserted is True for result in results) == 1
    assert sum(result.decision_inserted is False for result in results) == 1
    assert sum(result.projection_advanced is True for result in results) == 1
    assert sum(result.projection_advanced is False for result in results) == 1
    assert _revision_counts(setup) == (1, 1, 1)


def test_verified_read_keeps_one_snapshot_when_selection_commits_between_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    setup, node = _registry_with_subject(root)
    first_revision = _revision(node, 1)
    second_revision = _revision(node, 2)
    setup.publish_revision(first_revision)
    setup.publish_revision(second_revision)
    first = _select(setup, node, first_revision, 1)
    reader = LocalLogicalNodeRegistry(root)
    writer = LocalLogicalNodeRegistry(root)
    decision_decoded = threading.Event()
    writer_committed = threading.Event()
    original_decision_from_row = reader._decision_from_row

    def pause_after_decision_query(row: sqlite3.Row):
        decision = original_decision_from_row(row)
        if not decision_decoded.is_set():
            decision_decoded.set()
            assert writer_committed.wait(timeout=10)
        return decision

    monkeypatch.setattr(reader, "_decision_from_row", pause_after_decision_query)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(reader.verify_subject, *node.identity)
        try:
            assert decision_decoded.wait(timeout=10)
            second = _select(
                writer,
                node,
                second_revision,
                2,
                expected_previous=first.decision.selection_decision_id,
                selected_at="2026-07-18T13:01:00Z",
            )
        finally:
            writer_committed.set()
        during_commit = future.result(timeout=10)

    assert during_commit.revisions == tuple(
        sorted(
            (first_revision, second_revision),
            key=lambda revision: (revision.revision_logical_key, revision.revision_id),
        )
    )
    assert during_commit.decisions == (first.decision,)
    assert during_commit.current == first.current
    after_commit = reader.verify_subject(*node.identity)
    assert after_commit.decisions == (first.decision, second.decision)
    assert after_commit.current == second.current


def test_selection_semantic_retry_returns_original_decision_identity(tmp_path: Path) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    revision = _revision(node, 1)
    registry.publish_revision(revision)
    first = _select(registry, node, revision, 1)

    retry = _select(
        registry,
        node,
        revision,
        2,
        selected_at="2026-07-18T23:59:59Z",
    )

    assert retry.decision == first.decision
    assert retry.current == first.current
    assert retry.decision_inserted is False
    assert retry.projection_advanced is False
    assert _revision_counts(registry) == (1, 1, 1)


def _drop_trigger(connection: sqlite3.Connection, trigger_name: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_schema WHERE type = 'trigger' AND name = ?",
        (trigger_name,),
    ).fetchone()
    assert row is not None
    trigger_sql = str(row[0])
    connection.execute(f"DROP TRIGGER {trigger_name}")
    return trigger_sql


def _drop_update_trigger(connection: sqlite3.Connection, table: str) -> str:
    return _drop_trigger(connection, f"{table}_no_update")


@pytest.mark.parametrize("entity", ["revision", "decision", "current"])
@pytest.mark.parametrize("tamper", ["record", "digest", "column"])
def test_verified_revision_reads_reject_record_column_and_digest_tamper(
    tmp_path: Path,
    entity: str,
    tamper: str,
) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    revision = _revision(node, 1)
    registry.publish_revision(revision)
    selection = _select(registry, node, revision, 1)
    table = {
        "revision": "immutable_node_revisions",
        "decision": "selection_decisions",
        "current": "current_selections",
    }[entity]
    json_column = {
        "revision": "revision_json",
        "decision": "decision_json",
        "current": "current_json",
    }[entity]
    digest_column = f"{json_column}_sha256"
    where_column = {
        "revision": "revision_id",
        "decision": "selection_decision_id",
        "current": "subject_id",
    }[entity]
    where_value = {
        "revision": revision.revision_id,
        "decision": selection.decision.selection_decision_id,
        "current": node.node_logical_key,
    }[entity]

    with sqlite3.connect(registry.database_path) as connection:
        restore_trigger = None
        if entity != "current":
            restore_trigger = _drop_update_trigger(connection, table)
        if tamper == "record":
            row = connection.execute(
                f"SELECT {json_column} FROM {table} WHERE {where_column} = ?",
                (where_value,),
            ).fetchone()
            assert row is not None
            raw = bytes(row[0]) + b" "
            connection.execute(
                f"""
                UPDATE {table} SET {json_column} = ?, {digest_column} = ?
                WHERE {where_column} = ?
                """,
                (raw, hashlib.sha256(raw).hexdigest(), where_value),
            )
        elif tamper == "digest":
            connection.execute(
                f"UPDATE {table} SET {digest_column} = ? WHERE {where_column} = ?",
                ("0" * 64, where_value),
            )
        elif entity == "revision":
            connection.execute(
                "UPDATE immutable_node_revisions SET status_at_publication = ? "
                "WHERE revision_id = ?",
                ("TAMPERED", where_value),
            )
        elif entity == "decision":
            connection.execute(
                "UPDATE selection_decisions SET selection_key_namespace = ? "
                "WHERE selection_decision_id = ?",
                ("tampered-selection:v1", where_value),
            )
        else:
            connection.execute(
                "UPDATE current_selections SET selected_at = ? WHERE subject_id = ?",
                ("2026-07-18T23:59:59Z", where_value),
            )
        if restore_trigger is not None:
            connection.execute(restore_trigger)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        if entity == "revision":
            registry.lookup_revision(*node.identity, revision.revision_id)
        elif entity == "decision":
            registry.lookup_selection_decision(
                *node.identity,
                selection.decision.selection_decision_id,
            )
        else:
            registry.lookup_current_selection(*node.identity)
    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR)


@pytest.mark.parametrize(
    ("table", "operation"),
    [
        ("immutable_node_revisions", "UPDATE"),
        ("immutable_node_revisions", "DELETE"),
        ("selection_decisions", "UPDATE"),
        ("selection_decisions", "DELETE"),
    ],
)
def test_database_triggers_reject_update_and_delete_of_immutable_rows(
    tmp_path: Path,
    table: str,
    operation: str,
) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    revision = _revision(node, 1)
    registry.publish_revision(revision)
    selection = _select(registry, node, revision, 1)
    statement = (
        f"UPDATE {table} SET schema_version = schema_version"
        if operation == "UPDATE"
        else f"DELETE FROM {table}"
    )

    with (
        sqlite3.connect(registry.database_path) as connection,
        pytest.raises(sqlite3.IntegrityError),
    ):
        connection.execute(statement)

    verified = registry.verify_subject(*node.identity)
    assert verified.revisions == (revision,)
    assert verified.decisions == (selection.decision,)
    assert verified.current == selection.current


def test_rebuild_rejects_missing_authoritative_tail_instead_of_falling_back(
    tmp_path: Path,
) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    first_revision = _revision(node, 1)
    second_revision = _revision(node, 2)
    registry.publish_revision(first_revision)
    registry.publish_revision(second_revision)
    first = _select(registry, node, first_revision, 1)
    second = _select(
        registry,
        node,
        second_revision,
        2,
        expected_previous=first.decision.selection_decision_id,
        selected_at="2026-07-18T13:01:00Z",
    )

    with sqlite3.connect(registry.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        trigger_sql = _drop_trigger(connection, "selection_decisions_no_delete")
        connection.execute(
            """
            DELETE FROM selection_decisions
            WHERE subject_type = ? AND subject_id = ? AND selection_decision_id = ?
            """,
            (*node.identity, second.decision.selection_decision_id),
        )
        connection.execute(trigger_sql)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        registry.rebuild_current_projection()
    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.INTEGRITY_ERROR)

    with sqlite3.connect(registry.database_path) as connection:
        decisions = connection.execute(
            """
            SELECT selection_decision_id FROM selection_decisions
            WHERE subject_type = ? AND subject_id = ?
            ORDER BY selection_sequence
            """,
            node.identity,
        ).fetchall()
        current = connection.execute(
            """
            SELECT selection_decision_id FROM current_selections
            WHERE subject_type = ? AND subject_id = ?
            """,
            node.identity,
        ).fetchone()
    assert decisions == [(first.decision.selection_decision_id,)]
    assert current == (second.decision.selection_decision_id,)


def test_rebuild_precommit_error_maps_to_transaction_failed_and_keeps_current_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    revision = _revision(node, 1)
    registry.publish_revision(revision)
    _select(registry, node, revision, 1)
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute(
            "DELETE FROM current_selections WHERE subject_type = ? AND subject_id = ?",
            node.identity,
        )

    def fail_before_commit(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected pre-commit rebuild failure")

    monkeypatch.setattr(registry, "_commit", fail_before_commit)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        registry.rebuild_current_projection()
    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED)

    with sqlite3.connect(registry.database_path) as connection:
        current_count = connection.execute(
            """
            SELECT COUNT(*) FROM current_selections
            WHERE subject_type = ? AND subject_id = ?
            """,
            node.identity,
        ).fetchone()
    assert current_count == (0,)


def test_supersedes_composite_foreign_key_rejects_mismatched_logical_key(
    tmp_path: Path,
) -> None:
    registry, node = _registry_with_subject(tmp_path / "registry")
    first = _revision(node, 1)
    registry.publish_revision(first)
    candidate = _revision(node, 2, supersedes=first)
    wrong_predecessor_key = _revision(node, 99).revision_logical_key
    assert wrong_predecessor_key != first.revision_logical_key
    raw = canonical_json_bytes(candidate)
    values = (
        candidate.subject_type,
        candidate.subject_id,
        candidate.revision_id,
        candidate.schema_version,
        candidate.revision_key_namespace,
        candidate.revision_logical_key,
        candidate.semantic_sha256,
        candidate.payload_sha256,
        candidate.lineage_sha256,
        candidate.status_at_publication,
        candidate.eligibility_at_publication.value,
        candidate.revision_policy_version,
        candidate.supersedes_revision_id,
        wrong_predecessor_key,
        candidate.published_at,
        raw,
        hashlib.sha256(raw).hexdigest(),
    )

    with sqlite3.connect(registry.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
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
                values,
            )

    assert registry.list_revisions(*node.identity) == (first,)


def test_revision_commit_error_before_commit_rolls_back_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    registry, node = _registry_with_subject(root)
    revision = _revision(node, 1)

    def fail_before_commit(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected pre-commit revision failure")

    monkeypatch.setattr(registry, "_commit", fail_before_commit)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        registry.publish_revision(revision)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED)
    assert _revision_counts(registry) == (0, 0, 0)
    retry = LocalLogicalNodeRegistry(root).publish_revision(revision)
    assert retry.revision == revision
    assert retry.inserted is True


def test_revision_commit_error_after_commit_recovers_without_false_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    registry, node = _registry_with_subject(root)
    revision = _revision(node, 1)

    def commit_then_fail(connection: sqlite3.Connection) -> None:
        connection.commit()
        raise sqlite3.OperationalError("injected uncertain revision commit")

    monkeypatch.setattr(registry, "_commit", commit_then_fail)

    published = registry.publish_revision(revision)

    assert published.revision == revision
    assert published.inserted is None
    assert (
        LocalLogicalNodeRegistry(root).lookup_revision(
            *node.identity,
            revision.revision_id,
        )
        == revision
    )


def test_selection_commit_error_before_commit_rolls_back_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    registry, node = _registry_with_subject(root)
    revision = _revision(node, 1)
    registry.publish_revision(revision)

    def fail_before_commit(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected pre-commit selection failure")

    monkeypatch.setattr(registry, "_commit", fail_before_commit)

    with pytest.raises(RevisionSelectionRegistryError) as caught:
        _select(registry, node, revision, 1)

    _assert_revision_error(caught, RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED)
    assert _revision_counts(registry) == (1, 0, 0)
    retry = _select(LocalLogicalNodeRegistry(root), node, revision, 1)
    assert retry.decision_inserted is True
    assert retry.projection_advanced is True


def test_selection_commit_error_after_commit_recovers_without_false_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    registry, node = _registry_with_subject(root)
    revision = _revision(node, 1)
    registry.publish_revision(revision)

    def commit_then_fail(connection: sqlite3.Connection) -> None:
        connection.commit()
        raise sqlite3.OperationalError("injected uncertain selection commit")

    monkeypatch.setattr(registry, "_commit", commit_then_fail)

    selected = _select(registry, node, revision, 1)

    assert selected.decision_inserted is None
    assert selected.projection_advanced is None
    verified = LocalLogicalNodeRegistry(root).verify_subject(*node.identity)
    assert verified.decisions == (selected.decision,)
    assert verified.current == selected.current


def test_new_database_uses_schema_v2_with_revision_tables_and_triggers(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path / "registry")

    with sqlite3.connect(registry.database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()
        rows = connection.execute(
            "SELECT type, name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()

    assert version == (2,)
    objects = {(str(row[0]), str(row[1])) for row in rows}
    assert {
        ("table", "immutable_node_revisions"),
        ("table", "selection_decisions"),
        ("table", "current_selections"),
        ("trigger", "immutable_node_revisions_no_update"),
        ("trigger", "immutable_node_revisions_no_delete"),
        ("trigger", "selection_decisions_no_update"),
        ("trigger", "selection_decisions_no_delete"),
    } <= objects


def test_v1_database_migrates_to_v2_without_changing_node_history(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry, node = _registry_with_subject(root)
    before = registry.verify_node(*node.identity)
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute("DROP TABLE current_selections")
        connection.execute("DROP TABLE selection_decisions")
        connection.execute("DROP TABLE immutable_node_revisions")
        connection.execute("PRAGMA user_version = 1")

    migrated = LocalLogicalNodeRegistry(root)

    assert migrated.verify_node(*node.identity) == before
    with sqlite3.connect(migrated.database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()
    assert version == (2,)
    revision = _revision(node, 1)
    assert migrated.publish_revision(revision).revision == revision


def test_v2_constructor_fails_closed_on_current_projection_foreign_key_orphan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "registry"
    registry, node = _registry_with_subject(root)
    revision = _revision(node, 1)
    registry.publish_revision(revision)
    _select(registry, node, revision, 1)
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute(
            """
            UPDATE current_selections SET selected_revision_id = ?
            WHERE subject_type = ? AND subject_id = ?
            """,
            (_uuid(999_999), *node.identity),
        )

    with pytest.raises(LogicalNodeRegistryError) as caught:
        LocalLogicalNodeRegistry(root)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


def test_unsupported_user_version_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry = LocalLogicalNodeRegistry(root)
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(LogicalNodeRegistryError) as caught:
        LocalLogicalNodeRegistry(root)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR


def test_v2_revision_schema_drift_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry = LocalLogicalNodeRegistry(root)
    with sqlite3.connect(registry.database_path) as connection:
        row = connection.execute(
            "SELECT sql FROM sqlite_schema "
            "WHERE type = 'table' AND name = 'immutable_node_revisions'"
        ).fetchone()
        assert row is not None
        original_sql = str(row[0])
        tampered_sql = original_sql.replace(
            "payload_sha256 TEXT NOT NULL",
            "payload_sha256 INTEGER NOT NULL",
            1,
        )
        assert tampered_sql != original_sql
        schema_version = connection.execute("PRAGMA schema_version").fetchone()
        assert schema_version is not None
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_schema SET sql = ? "
            "WHERE type = 'table' AND name = 'immutable_node_revisions'",
            (tampered_sql,),
        )
        connection.execute(f"PRAGMA schema_version = {int(schema_version[0]) + 1}")
        connection.execute("PRAGMA writable_schema = OFF")

    with pytest.raises(LogicalNodeRegistryError) as caught:
        LocalLogicalNodeRegistry(root)
    assert caught.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR
