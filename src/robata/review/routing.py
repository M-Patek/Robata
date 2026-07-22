"""Best-effort routing that preserves already-produced primary results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from robata.ports.review_queue import ReviewQueue
from robata.review.models import (
    NonBlockingReviewRoutingPolicy,
    ReviewRequest,
    create_review_task,
)


class ReviewRoutingDisposition(StrEnum):
    """Observable result of nonblocking downstream review routing."""

    ENQUEUED = "ENQUEUED"
    ALREADY_ENQUEUED = "ALREADY_ENQUEUED"
    NOT_ROUTED = "NOT_ROUTED"
    ROUTING_FAILED = "ROUTING_FAILED"


@dataclass(frozen=True, slots=True)
class ReviewRoutingReceipt:
    """Routing outcome that never substitutes for primary pipeline state."""

    disposition: ReviewRoutingDisposition
    review_task_id: str | None = None
    failure_type: str | None = None


class BlockingReviewPolicyUnavailableError(RuntimeError):
    """Blocking review cannot run without an explicit governed policy."""


class NonBlockingReviewRouter:
    """Route optional review work while containing downstream queue failures."""

    def __init__(
        self,
        *,
        policy: NonBlockingReviewRoutingPolicy,
        queue: ReviewQueue,
    ) -> None:
        if not isinstance(policy, NonBlockingReviewRoutingPolicy):
            raise TypeError("policy must be a NonBlockingReviewRoutingPolicy")
        self._policy = policy
        self._queue = queue

    def route(self, request: ReviewRequest) -> ReviewRoutingReceipt:
        """Route after primary completion; all downstream failures become receipts."""

        try:
            task = create_review_task(request, self._policy)
            if task is None:
                return ReviewRoutingReceipt(ReviewRoutingDisposition.NOT_ROUTED)
            result = self._queue.enqueue(task)
        except Exception as exc:
            # This is the section 25.8 containment boundary. The primary result
            # already exists and review infrastructure cannot invalidate it.
            return ReviewRoutingReceipt(
                ReviewRoutingDisposition.ROUTING_FAILED,
                failure_type=type(exc).__name__,
            )
        disposition = (
            ReviewRoutingDisposition.ENQUEUED
            if result.inserted
            else ReviewRoutingDisposition.ALREADY_ENQUEUED
        )
        return ReviewRoutingReceipt(disposition, review_task_id=result.task.review_task_id)

    def require_blocking_review(self, _request: ReviewRequest) -> None:
        """Fail closed until a complete governed blocking policy exists."""

        raise BlockingReviewPolicyUnavailableError(
            "blocking review requires a governed blocking_review_policy with named "
            "risk classes, ownership, capacity, deadlines, fallback, and metrics"
        )


@dataclass(frozen=True, slots=True)
class PrimaryResultWithReview[PrimaryResultT]:
    """An unchanged primary result accompanied by independent review state."""

    primary_result: PrimaryResultT
    review: ReviewRoutingReceipt


def route_review_after_primary[PrimaryResultT](
    primary_result: PrimaryResultT,
    *,
    router: NonBlockingReviewRouter,
    request: ReviewRequest,
) -> PrimaryResultWithReview[PrimaryResultT]:
    """Attach best-effort review routing without changing or withholding primary output."""

    return PrimaryResultWithReview(
        primary_result=primary_result,
        review=router.route(request),
    )


__all__ = [
    "BlockingReviewPolicyUnavailableError",
    "NonBlockingReviewRouter",
    "PrimaryResultWithReview",
    "ReviewRoutingDisposition",
    "ReviewRoutingReceipt",
    "route_review_after_primary",
]
