from __future__ import annotations

from pathlib import Path

from robata.adapters.sqlite_primary_completion import SQLitePrimaryCompletionRepository
from robata.adapters.sqlite_review_queue import SQLiteReviewQueue
from robata.application.canonical.local_review_routing import (
    route_local_review_after_completion,
)
from robata.application.canonical.media_quality_binding import (
    derive_local_media_quality_binding,
)
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRegistry
from robata.review.models import ReviewTrigger
from robata.review.routing import ReviewRoutingDisposition
from tests.integration.test_canonical_offline import (
    _claim_bytes,
    _harness,
    _processing_run,
    _run,
)
from tests.integration.test_sqlite_primary_completion import _prepare_command, _run_case
from tests.unit.test_canonical_media_quality_binding import _report as _quality_report


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


def test_media_quality_flags_route_nonblocking_qa_degradation(tmp_path: Path) -> None:
    _, repository, command = _run_case(tmp_path, run_value=95_004)
    committed = repository.commit(command).committed
    registry = SchemaRegistry()
    state_root = tmp_path / "local-operations"
    state_root.mkdir()
    binding = derive_local_media_quality_binding(_quality_report(with_flags=True))

    routed = route_local_review_after_completion(
        committed,
        state_root=state_root,
        registry=registry,
        media_quality_binding=binding,
    )

    assert routed.disposition is ReviewRoutingDisposition.ENQUEUED
    task = (
        SQLiteReviewQueue(
            state_root / "review-queue.sqlite3",
            registry=registry,
        )
        .list_open()[0]
        .task
    )
    assert task.trigger is ReviewTrigger.QA_DEGRADATION
    assert task.priority == 5
    assert task.subject.subject_type == "LOCAL_MEDIA_QUALITY_REPORT"
    assert task.subject.subject_id == (f"media-quality-report:{binding.report_semantic_sha256}")
    assert task.reason_codes == (
        "LOCAL_MEDIA_QUALITY_OBSERVED_BLACK_LUMA",
        "LOCAL_MEDIA_QUALITY_PROXY_LOW_EDGE_ENERGY",
    )
    assert task.blocking is False


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


def test_early_no_events_routes_completion_subject_for_sampling(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "logical",
        event_proposal_response_factory=lambda _request: canonical_json_bytes(
            {"claims": [], "abstained": False}
        ),
    )
    repository = SQLitePrimaryCompletionRepository(tmp_path / "completion.sqlite3")
    processing_run = _processing_run(
        harness,
        run_id="00000000-0000-5000-8000-000000095003",
    )
    repository.begin_run(processing_run)
    result = _run(harness, processing_run=processing_run)
    committed = repository.commit(
        _prepare_command(repository=repository, harness=harness, result=result)
    ).committed
    assert committed.detail.output_decision is None

    state_root = tmp_path / "local-operations"
    state_root.mkdir()
    routed = route_local_review_after_completion(
        committed,
        state_root=state_root,
        registry=SchemaRegistry(),
    )

    assert routed.disposition is ReviewRoutingDisposition.ENQUEUED
    task = (
        SQLiteReviewQueue(
            state_root / "review-queue.sqlite3",
            registry=SchemaRegistry(),
        )
        .list_open()[0]
        .task
    )
    assert task.subject.subject_type == "CANONICAL_PRIMARY_COMPLETION"
    assert task.subject.subject_id == (f"primary-completion:{committed.completion.semantic_sha256}")
    assert task.reason_codes == ("LOCAL_CONFORMANCE_NO_EVENTS_SAMPLE",)
