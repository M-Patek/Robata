"""Durability, fencing, SLA, and replay tests for the local review queue."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest

from robata.adapters.sqlite_review_queue import SQLiteReviewQueue
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.ports.review_queue import ReviewQueueError, ReviewQueueErrorCode
from robata.review.models import (
    ReviewAdjudication,
    ReviewAnnotation,
    ReviewReopenCommand,
    ReviewRequest,
    ReviewRoutingRule,
    ReviewSubject,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTrigger,
    create_nonblocking_review_routing_policy,
    create_review_annotation,
    create_review_reopen_command,
    create_review_task,
)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _assert_registered_payload(payload: bytes) -> None:
    document = json.loads(payload)
    assert document["schema_version"] == "1.0"
    assert set(document["schema_ref"]) == {
        "schema_id",
        "version",
        "artifact_id",
        "sha256",
    }


def _tamper_schema_ref(
    database: Path,
    payload_kind: Literal["task", "annotation", "reopen"],
) -> None:
    statements = {
        "task": (
            "DROP TRIGGER review_tasks_immutable_definition",
            "SELECT task_json FROM review_tasks LIMIT 1",
            "UPDATE review_tasks SET task_json = ?, task_exact_sha256 = ?",
        ),
        "annotation": (
            "DROP TRIGGER review_annotations_no_update",
            "SELECT annotation_json FROM review_annotations LIMIT 1",
            "UPDATE review_annotations SET annotation_json = ?, annotation_exact_sha256 = ?",
        ),
        "reopen": (
            "DROP TRIGGER review_reopen_commands_no_update",
            "SELECT command_json FROM review_reopen_commands LIMIT 1",
            "UPDATE review_reopen_commands SET command_json = ?, command_exact_sha256 = ?",
        ),
    }
    drop_trigger, select_payload, update_payload = statements[payload_kind]
    connection = sqlite3.connect(database)
    try:
        connection.execute(drop_trigger)
        row = connection.execute(select_payload).fetchone()
        assert row is not None
        document = json.loads(bytes(row[0]))
        document["schema_ref"]["artifact_id"] = _uuid(9_999)
        payload = canonical_json_bytes(document)
        connection.execute(
            update_payload,
            (sqlite3.Binary(payload), exact_bytes_sha256(payload)),
        )
        connection.commit()
    finally:
        connection.close()


def _policy():  # type: ignore[no-untyped-def]
    return create_nonblocking_review_routing_policy(
        policy_version="review-routing-7",
        rules=(
            ReviewRoutingRule(
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                priority=20,
                sla_ns=100,
            ),
            ReviewRoutingRule(
                trigger=ReviewTrigger.IDENTITY_AMBIGUITY,
                priority=2,
                sla_ns=50,
            ),
        ),
    )


def _task(
    identity: int,
    *,
    trigger: ReviewTrigger = ReviewTrigger.LOW_CONFIDENCE,
    requested_at_ns: int = 1_000,
) -> ReviewTask:
    request = ReviewRequest(
        request_id=_uuid(identity),
        subject=ReviewSubject(
            subject_type="EVENT_HYPOTHESIS",
            subject_id=f"event-hypothesis:{_digest(100 + identity)}",
            recording_identity=_digest(200 + identity),
        ),
        trigger=trigger,
        reason_codes=(trigger.value,),
        requested_at_ns=requested_at_ns,
    )
    task = create_review_task(request, _policy())
    assert task is not None
    return task


def _annotation(task: ReviewTask, *, fence: int, owner: str, decision: str = "ACCEPT"):
    return create_review_annotation(
        task=task,
        lease_fence=fence,
        lease_owner=owner,
        reviewer_id="reviewer-1",
        adjudication=ReviewAdjudication(
            decision_code=decision,
            reason_codes=("EVIDENCE_VERIFIED",),
            comment="Reviewed against the immutable evidence bundle.",
        ),
        authored_at_ns=1_020 + fence,
    )


def _persist_review_history(
    database: Path,
) -> tuple[SQLiteReviewQueue, ReviewTask, ReviewAnnotation, ReviewReopenCommand]:
    queue = SQLiteReviewQueue(database)
    task = _task(1)
    queue.enqueue(task)
    lease = queue.claim_next(worker_id="worker-a", now_ns=1_010, lease_duration_ns=100)
    assert lease is not None
    annotation = _annotation(task, fence=lease.lease_fence, owner="worker-a")
    queue.submit_annotation(annotation, now_ns=1_030)
    command = create_review_reopen_command(
        reopen_id=_uuid(900),
        review_task_id=task.review_task_id,
        expected_annotation_id=annotation.annotation_id,
        reason_code="NEW_EVIDENCE",
        requested_at_ns=1_040,
    )
    assert queue.reopen(command).applied is True
    return queue, task, annotation, command


def test_priority_claim_submit_and_exact_replay(tmp_path: Path) -> None:
    queue = SQLiteReviewQueue(tmp_path / "review.sqlite3")
    low_priority = _task(1)
    high_priority = _task(2, trigger=ReviewTrigger.IDENTITY_AMBIGUITY)

    assert queue.enqueue(low_priority).inserted is True
    assert queue.enqueue(low_priority).inserted is False
    queue.enqueue(high_priority)

    lease = queue.claim_next(worker_id="worker-a", now_ns=1_010, lease_duration_ns=100)
    assert lease is not None
    assert lease.task == high_priority
    assert lease.lease_fence == 1

    annotation = _annotation(high_priority, fence=lease.lease_fence, owner="worker-a")
    assert queue.submit_annotation(annotation, now_ns=1_030).inserted is True
    assert queue.submit_annotation(annotation, now_ns=1_030).inserted is False

    completed = queue.get_task(high_priority.review_task_id)
    assert completed is not None
    assert completed.status is ReviewTaskStatus.COMPLETED
    assert completed.completed_annotation_id == annotation.annotation_id
    assert queue.list_annotations(high_priority.review_task_id) == (annotation,)
    assert queue.list_open()[0].task == low_priority


def test_task_and_annotation_identity_conflicts_fail_closed(tmp_path: Path) -> None:
    queue = SQLiteReviewQueue(tmp_path / "review.sqlite3")
    task = _task(1)
    queue.enqueue(task)

    conflicting_task = _task(1, requested_at_ns=1_001)
    with pytest.raises(ReviewQueueError) as task_error:
        queue.enqueue(conflicting_task)
    assert task_error.value.code is ReviewQueueErrorCode.TASK_CONFLICT

    lease = queue.claim_next(worker_id="worker-a", now_ns=1_010, lease_duration_ns=100)
    assert lease is not None
    accepted = _annotation(task, fence=lease.lease_fence, owner="worker-a")
    queue.submit_annotation(accepted, now_ns=1_030)

    conflicting_annotation = _annotation(
        task,
        fence=lease.lease_fence,
        owner="worker-a",
        decision="REJECT",
    )
    with pytest.raises(ReviewQueueError) as annotation_error:
        queue.submit_annotation(conflicting_annotation, now_ns=1_030)
    assert annotation_error.value.code is ReviewQueueErrorCode.ANNOTATION_CONFLICT


def test_sla_overdue_is_visible_independently_of_lease_state(tmp_path: Path) -> None:
    queue = SQLiteReviewQueue(tmp_path / "review.sqlite3")
    task = _task(1)
    queue.enqueue(task)

    assert task.due_at_ns == 1_100
    assert queue.list_overdue(now_ns=1_100) == ()
    overdue = queue.list_overdue(now_ns=1_101)
    assert tuple(item.task for item in overdue) == (task,)
    assert overdue[0].is_overdue(1_101)

    lease = queue.claim_next(worker_id="worker-a", now_ns=1_101, lease_duration_ns=100)
    assert lease is not None
    assert queue.list_overdue(now_ns=1_102)[0].status is ReviewTaskStatus.LEASED


def test_expired_lease_recovery_increments_fence_and_rejects_stale_submit(
    tmp_path: Path,
) -> None:
    database = tmp_path / "review.sqlite3"
    first = SQLiteReviewQueue(database)
    task = _task(1)
    first.enqueue(task)
    first_lease = first.claim_next(
        worker_id="worker-a",
        now_ns=1_010,
        lease_duration_ns=10,
    )
    assert first_lease is not None
    stale_annotation = _annotation(task, fence=1, owner="worker-a")

    with pytest.raises(ReviewQueueError) as expired:
        first.submit_annotation(stale_annotation, now_ns=1_020)
    assert expired.value.code is ReviewQueueErrorCode.LEASE_EXPIRED

    recovered = SQLiteReviewQueue(database)
    second_lease = recovered.claim_next(
        worker_id="worker-b",
        now_ns=1_020,
        lease_duration_ns=100,
    )
    assert second_lease is not None
    assert second_lease.lease_fence == 2

    with pytest.raises(ReviewQueueError) as stale:
        recovered.submit_annotation(stale_annotation, now_ns=1_021)
    assert stale.value.code is ReviewQueueErrorCode.STALE_FENCE

    current_annotation = _annotation(task, fence=2, owner="worker-b")
    recovered.submit_annotation(current_annotation, now_ns=1_030)
    assert recovered.get_task(task.review_task_id).status is ReviewTaskStatus.COMPLETED  # type: ignore[union-attr]


def test_reopen_is_durable_idempotent_and_preserves_annotation_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "review.sqlite3"
    queue = SQLiteReviewQueue(database)
    task = _task(1)
    queue.enqueue(task)
    lease = queue.claim_next(worker_id="worker-a", now_ns=1_010, lease_duration_ns=100)
    assert lease is not None
    annotation = _annotation(task, fence=1, owner="worker-a")
    queue.submit_annotation(annotation, now_ns=1_030)

    command = create_review_reopen_command(
        reopen_id=_uuid(900),
        review_task_id=task.review_task_id,
        expected_annotation_id=annotation.annotation_id,
        reason_code="NEW_EVIDENCE",
        requested_at_ns=1_040,
    )
    first = queue.reopen(command)
    replay = queue.reopen(command)
    assert first.applied is True
    assert first.snapshot.status is ReviewTaskStatus.PENDING
    assert replay.applied is False

    conflict = create_review_reopen_command(
        reopen_id=command.reopen_id,
        review_task_id=task.review_task_id,
        expected_annotation_id=annotation.annotation_id,
        reason_code="POLICY_CHANGE",
        requested_at_ns=1_040,
    )
    with pytest.raises(ReviewQueueError) as conflict_error:
        queue.reopen(conflict)
    assert conflict_error.value.code is ReviewQueueErrorCode.REOPEN_CONFLICT

    recovered = SQLiteReviewQueue(database)
    assert recovered.list_annotations(task.review_task_id) == (annotation,)
    new_lease = recovered.claim_next(
        worker_id="worker-b",
        now_ns=1_050,
        lease_duration_ns=100,
    )
    assert new_lease is not None
    assert new_lease.lease_fence == 2


def test_registered_payloads_survive_database_reopen_and_exact_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "review.sqlite3"
    _, task, annotation, command = _persist_review_history(database)

    connection = sqlite3.connect(database)
    try:
        task_payload = bytes(connection.execute("SELECT task_json FROM review_tasks").fetchone()[0])
        annotation_payload = bytes(
            connection.execute("SELECT annotation_json FROM review_annotations").fetchone()[0]
        )
        reopen_payload = bytes(
            connection.execute("SELECT command_json FROM review_reopen_commands").fetchone()[0]
        )
    finally:
        connection.close()
    _assert_registered_payload(task_payload)
    _assert_registered_payload(annotation_payload)
    _assert_registered_payload(reopen_payload)

    recovered = SQLiteReviewQueue(database)
    snapshot = recovered.get_task(task.review_task_id)
    assert snapshot is not None
    assert snapshot.task == task
    assert recovered.list_annotations(task.review_task_id) == (annotation,)
    assert recovered.enqueue(task).inserted is False
    assert recovered.submit_annotation(annotation, now_ns=1_050).inserted is False
    reopen_replay = recovered.reopen(command)
    assert reopen_replay.command == command
    assert reopen_replay.applied is False


@pytest.mark.parametrize("payload_kind", ("task", "annotation", "reopen"))
def test_registered_pin_tampering_fails_closed_after_exact_digest_is_recomputed(
    tmp_path: Path,
    payload_kind: Literal["task", "annotation", "reopen"],
) -> None:
    database = tmp_path / "review.sqlite3"
    _, task, _, command = _persist_review_history(database)
    _tamper_schema_ref(database, payload_kind)

    recovered = SQLiteReviewQueue(database)
    with pytest.raises(ReviewQueueError) as error:
        if payload_kind == "task":
            recovered.get_task(task.review_task_id)
        elif payload_kind == "annotation":
            recovered.list_annotations(task.review_task_id)
        else:
            recovered.reopen(command)
    assert error.value.code is ReviewQueueErrorCode.INTEGRITY_ERROR
    assert "registered schema validation failed" in str(error.value)


def test_v1_database_is_rejected_instead_of_reinterpreting_persisted_payloads(
    tmp_path: Path,
) -> None:
    database = tmp_path / "review.sqlite3"
    queue = SQLiteReviewQueue(database)
    queue.enqueue(_task(1))

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ReviewQueueError) as error:
        SQLiteReviewQueue(database)
    assert error.value.code is ReviewQueueErrorCode.INTEGRITY_ERROR
