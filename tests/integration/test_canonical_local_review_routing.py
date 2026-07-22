from __future__ import annotations

from pathlib import Path

from robata.adapters.sqlite_review_queue import SQLiteReviewQueue
from robata.application.canonical.local_review_routing import (
    route_local_review_after_completion,
)
from robata.contracts.schema_registry import SchemaRegistry
from robata.review.routing import ReviewRoutingDisposition
from tests.integration.test_sqlite_primary_completion import _run_case


class _UnavailableReviewQueue:
    def enqueue(self, _task):  # type: ignore[no-untyped-def]
        raise RuntimeError("review service unavailable")


def test_committed_result_routes_idempotent_nonblocking_review(tmp_path: Path) -> None:
    _, repository, command = _run_case(tmp_path, run_value=95_001)
    committed = repository.commit(command).committed
    registry = SchemaRegistry()
    state_root = tmp_path / "local-operations"
    state_root.mkdir()

    first = route_local_review_after_completion(
        committed,
        state_root=state_root,
        registry=registry,
    )
    replay = route_local_review_after_completion(
        committed,
        state_root=state_root,
        registry=registry,
    )

    assert first.disposition is ReviewRoutingDisposition.ENQUEUED
    assert replay.disposition is ReviewRoutingDisposition.ALREADY_ENQUEUED
    assert replay.review_task_id == first.review_task_id
    open_tasks = SQLiteReviewQueue(
        state_root / "review-queue.sqlite3",
        registry=registry,
    ).list_open()
    assert len(open_tasks) == 1
    assert open_tasks[0].task.blocking is False
    assert open_tasks[0].task.reason_codes == ("LOCAL_CONFORMANCE_SAMPLE",)


def test_review_failure_cannot_replace_committed_primary_result(tmp_path: Path) -> None:
    _, repository, command = _run_case(tmp_path, run_value=95_002)
    committed = repository.commit(command).committed
    state_root = tmp_path / "local-operations"
    state_root.mkdir()

    result = route_local_review_after_completion(
        committed,
        state_root=state_root,
        registry=SchemaRegistry(),
        queue=_UnavailableReviewQueue(),  # type: ignore[arg-type]
    )

    assert result.disposition is ReviewRoutingDisposition.ROUTING_FAILED
    assert result.failure_type == "RuntimeError"
    assert committed.completion.run_id == command.completion.run_id
