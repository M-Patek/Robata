from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.ports.logical_node_registry import (
    LogicalNodeRegistryError,
    LogicalNodeRegistryErrorCode,
)
from robata.runtime.observability import (
    RuntimeProfileRecorder,
    RuntimeSpanStatus,
)
from tests.unit.test_local_logical_node_registry import (
    _SELECTED_AT,
    _attach,
    _node,
    _revision,
    _uuid,
)


def _counter_map(
    recorder: RuntimeProfileRecorder,
    name: str,
) -> dict[tuple[tuple[str, object], ...], int]:
    return {
        tuple((attribute.name, attribute.value) for attribute in counter.attributes): counter.value
        for counter in recorder.snapshot().counters
        if counter.name == name
    }


def test_observes_exact_success_reads_and_commit_failure_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RuntimeProfileRecorder()
    registry = LocalLogicalNodeRegistry(
        tmp_path / "registry",
        runtime_observer=recorder,
    )
    node = _node("observed-success")
    _attach(registry, node)
    revision = _revision(node)
    registry.publish_revision(revision)
    registry.select_revision(
        subject_type=node.node_type,
        subject_id=node.node_logical_key,
        selected_revision_id=revision.revision_id,
        selection_decision_id=_uuid(10_001),
        selection_key_namespace="camera-video-selection:v1",
        expected_previous_selection_decision_id=None,
        selection_policy_version="camera-video-selection-policy-v1",
        selected_at=_SELECTED_AT,
    )

    def fail_commit(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected commit failure")

    monkeypatch.setattr(registry, "_commit", fail_commit)
    with pytest.raises(LogicalNodeRegistryError) as raised:
        _attach(
            registry,
            _node("observed-rollback"),
            run=2,
            work=202,
        )
    assert raised.value.code is LogicalNodeRegistryErrorCode.TRANSACTION_FAILED

    with sqlite3.connect(registry.database_path) as connection:
        node_count = connection.execute("SELECT COUNT(*) FROM logical_nodes").fetchone()
        membership_count = connection.execute(
            "SELECT COUNT(*) FROM processing_run_nodes"
        ).fetchone()
    assert node_count == (1,)
    assert membership_count == (1,)

    runtime = recorder.snapshot()
    assert (
        sum(span.name == "sqlite.logical_node_registry.initialization" for span in runtime.spans)
        == 1
    )
    assert _counter_map(recorder, "sqlite.logical_node_registry.transactions") == {
        (("operation", "attach_run_node"), ("write", True)): 2,
        (("operation", "initialize_schema"), ("write", True)): 1,
        (("operation", "lookup_revision"), ("write", False)): 1,
        (("operation", "migrate_v1_to_v2"), ("write", True)): 1,
        (("operation", "publish_revision"), ("write", True)): 1,
        (("operation", "select_revision"), ("write", True)): 1,
        (("operation", "verify_node"), ("write", False)): 2,
        (("operation", "verify_subject"), ("write", False)): 1,
    }
    assert _counter_map(recorder, "sqlite.logical_node_registry.commits") == {
        (("operation", "attach_run_node"), ("write", True)): 1,
        (("operation", "initialize_schema"), ("write", True)): 1,
        (("operation", "migrate_v1_to_v2"), ("write", True)): 1,
        (("operation", "publish_revision"), ("write", True)): 1,
        (("operation", "select_revision"), ("write", True)): 1,
    }
    assert _counter_map(recorder, "sqlite.logical_node_registry.rollbacks") == {
        (("operation", "attach_run_node"), ("write", True)): 1,
        (("operation", "lookup_revision"), ("write", False)): 1,
        (("operation", "verify_node"), ("write", False)): 2,
        (("operation", "verify_subject"), ("write", False)): 1,
    }
    assert _counter_map(recorder, "sqlite.logical_node_registry.commit_failures") == {
        (("operation", "attach_run_node"), ("write", True)): 1,
    }
    assert (
        _counter_map(
            recorder,
            "sqlite.logical_node_registry.transaction_outcomes_unknown",
        )
        == {}
    )
    transaction_spans = tuple(
        span for span in runtime.spans if span.name == "sqlite.logical_node_registry.transaction"
    )
    assert len(transaction_spans) == 9
    assert sum(span.status is RuntimeSpanStatus.ERROR for span in transaction_spans) == 2


def test_business_failure_is_observed_as_rollback_not_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RuntimeProfileRecorder()
    registry = LocalLogicalNodeRegistry(
        tmp_path / "registry",
        runtime_observer=recorder,
    )

    def fail_health(_connection: sqlite3.Connection) -> None:
        raise LogicalNodeRegistryError(
            LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
            "injected business failure",
        )

    monkeypatch.setattr(registry, "_verify_database_health", fail_health)
    with pytest.raises(LogicalNodeRegistryError) as raised:
        _attach(registry, _node("business-failure"))
    assert raised.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR

    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM logical_nodes").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM processing_run_nodes").fetchone() == (0,)
    expected = {(("operation", "attach_run_node"), ("write", True)): 1}
    assert _counter_map(recorder, "sqlite.logical_node_registry.rollbacks") == expected
    assert _counter_map(recorder, "sqlite.logical_node_registry.commit_failures") == {}
    assert (
        _counter_map(
            recorder,
            "sqlite.logical_node_registry.transaction_outcomes_unknown",
        )
        == {}
    )


def test_commit_that_raises_after_commit_is_observed_as_unknown_and_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RuntimeProfileRecorder()
    registry = LocalLogicalNodeRegistry(
        tmp_path / "registry",
        runtime_observer=recorder,
    )

    def commit_then_raise(connection: sqlite3.Connection) -> None:
        connection.commit()
        raise sqlite3.OperationalError("injected post-commit failure")

    monkeypatch.setattr(registry, "_commit", commit_then_raise)
    published = _attach(registry, _node("uncertain-commit"))
    assert published.node_inserted is None
    assert published.membership_inserted is None

    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM logical_nodes").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM processing_run_nodes").fetchone() == (1,)
    expected = {(("operation", "attach_run_node"), ("write", True)): 1}
    assert _counter_map(recorder, "sqlite.logical_node_registry.commit_failures") == expected
    assert (
        _counter_map(
            recorder,
            "sqlite.logical_node_registry.transaction_outcomes_unknown",
        )
        == expected
    )
    assert (
        _counter_map(recorder, "sqlite.logical_node_registry.rollbacks").get(
            (("operation", "attach_run_node"), ("write", True))
        )
        is None
    )


def test_rollback_failure_is_observed_as_unknown_and_preserves_business_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RuntimeProfileRecorder()
    registry = LocalLogicalNodeRegistry(
        tmp_path / "registry",
        runtime_observer=recorder,
    )

    def fail_health(_connection: sqlite3.Connection) -> None:
        raise LogicalNodeRegistryError(
            LogicalNodeRegistryErrorCode.INTEGRITY_ERROR,
            "injected business failure",
        )

    def fail_rollback(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected rollback failure")

    monkeypatch.setattr(registry, "_verify_database_health", fail_health)
    monkeypatch.setattr(registry, "_rollback", fail_rollback)
    with pytest.raises(LogicalNodeRegistryError) as raised:
        _attach(registry, _node("rollback-failure"))
    assert raised.value.code is LogicalNodeRegistryErrorCode.INTEGRITY_ERROR

    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM logical_nodes").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM processing_run_nodes").fetchone() == (0,)
    expected = {(("operation", "attach_run_node"), ("write", True)): 1}
    assert _counter_map(recorder, "sqlite.logical_node_registry.rollback_failures") == expected
    assert (
        _counter_map(
            recorder,
            "sqlite.logical_node_registry.transaction_outcomes_unknown",
        )
        == expected
    )
    assert _counter_map(recorder, "sqlite.logical_node_registry.commit_failures") == {}
