"""Section 25.8 nonblocking review routing tests."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pytest

from robata.ports.review_queue import EnqueuedReviewTask
from robata.review.models import (
    ReviewRequest,
    ReviewRoutingRule,
    ReviewSubject,
    ReviewTrigger,
    create_nonblocking_review_routing_policy,
    create_review_task,
)
from robata.review.routing import (
    BlockingReviewPolicyUnavailableError,
    NonBlockingReviewRouter,
    ReviewRoutingDisposition,
    route_review_after_primary,
)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _subject() -> ReviewSubject:
    return ReviewSubject(
        subject_type="EVENT_HYPOTHESIS",
        subject_id=f"event-hypothesis:{_digest(10)}",
        recording_identity=_digest(11),
    )


def _request(trigger: ReviewTrigger = ReviewTrigger.LOW_CONFIDENCE) -> ReviewRequest:
    return ReviewRequest(
        request_id=_uuid(1),
        subject=_subject(),
        trigger=trigger,
        reason_codes=(trigger.value,),
        requested_at_ns=1_000,
    )


def _policy():  # type: ignore[no-untyped-def]
    return create_nonblocking_review_routing_policy(
        policy_version="review-routing-7",
        rules=(
            ReviewRoutingRule(
                trigger=ReviewTrigger.REVIEW_SAMPLING,
                priority=20,
                sla_ns=500,
            ),
            ReviewRoutingRule(
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                priority=5,
                sla_ns=100,
            ),
        ),
    )


def test_versioned_policy_is_canonical_and_routes_explicit_triggers() -> None:
    policy = _policy()
    replayed = create_nonblocking_review_routing_policy(
        policy_version="review-routing-7",
        rules=tuple(reversed(policy.rules)),
    )

    assert replayed == policy
    assert tuple(rule.trigger for rule in policy.rules) == (
        ReviewTrigger.LOW_CONFIDENCE,
        ReviewTrigger.REVIEW_SAMPLING,
    )

    task = create_review_task(_request(), policy)
    assert task is not None
    assert task.blocking is False
    assert task.priority == 5
    assert task.due_at_ns == 1_100

    not_governed = create_review_task(_request(ReviewTrigger.QA_DEGRADATION), policy)
    assert not_governed is None


@dataclass
class _FailingQueue:
    calls: int = 0

    def enqueue(self, _task):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise RuntimeError("review infrastructure unavailable")


@dataclass
class _RecordingQueue:
    inserted: bool

    def enqueue(self, task):  # type: ignore[no-untyped-def]
        return EnqueuedReviewTask(task=task, inserted=self.inserted)


def test_review_queue_failure_cannot_withhold_or_replace_primary_result() -> None:
    queue = _FailingQueue()
    router = NonBlockingReviewRouter(policy=_policy(), queue=queue)  # type: ignore[arg-type]
    primary_result = object()

    outcome = route_review_after_primary(
        primary_result,
        router=router,
        request=_request(),
    )

    assert outcome.primary_result is primary_result
    assert outcome.review.disposition is ReviewRoutingDisposition.ROUTING_FAILED
    assert outcome.review.failure_type == "RuntimeError"
    assert queue.calls == 1


@pytest.mark.parametrize(
    ("inserted", "expected"),
    [
        (True, ReviewRoutingDisposition.ENQUEUED),
        (False, ReviewRoutingDisposition.ALREADY_ENQUEUED),
    ],
)
def test_router_reports_enqueue_replay_without_changing_primary_semantics(
    inserted: bool,
    expected: ReviewRoutingDisposition,
) -> None:
    router = NonBlockingReviewRouter(
        policy=_policy(),
        queue=_RecordingQueue(inserted),  # type: ignore[arg-type]
    )

    receipt = router.route(_request())

    assert receipt.disposition is expected
    assert receipt.review_task_id is not None


def test_blocking_review_fails_closed_without_governed_policy() -> None:
    router = NonBlockingReviewRouter(
        policy=_policy(),
        queue=_RecordingQueue(True),  # type: ignore[arg-type]
    )

    with pytest.raises(
        BlockingReviewPolicyUnavailableError,
        match="blocking_review_policy",
    ):
        router.require_blocking_review(_request())
