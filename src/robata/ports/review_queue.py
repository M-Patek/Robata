"""Durable boundary for asynchronous human-review work."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from robata.review.models import (
    ReviewAnnotation,
    ReviewLease,
    ReviewReopenCommand,
    ReviewTask,
    ReviewTaskSnapshot,
)


class ReviewQueueErrorCode(StrEnum):
    """Stable local review queue failure categories."""

    INVALID_REQUEST = "INVALID_REQUEST"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_CONFLICT = "TASK_CONFLICT"
    NOT_CLAIMABLE = "NOT_CLAIMABLE"
    STALE_FENCE = "STALE_FENCE"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    ANNOTATION_CONFLICT = "ANNOTATION_CONFLICT"
    REOPEN_CONFLICT = "REOPEN_CONFLICT"
    STORAGE_IO_ERROR = "STORAGE_IO_ERROR"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"


class ReviewQueueError(RuntimeError):
    """Review queue failure carrying a stable machine-readable code."""

    def __init__(self, code: ReviewQueueErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class EnqueuedReviewTask:
    """Verified task plus exact insertion attribution."""

    task: ReviewTask
    inserted: bool


@dataclass(frozen=True, slots=True)
class SubmittedReviewAnnotation:
    """Verified immutable annotation plus exact insertion attribution."""

    annotation: ReviewAnnotation
    inserted: bool


@dataclass(frozen=True, slots=True)
class ReopenedReviewTask:
    """Reopen command result; history remains append-only."""

    command: ReviewReopenCommand
    snapshot: ReviewTaskSnapshot
    applied: bool


class ReviewQueue(Protocol):
    """Priority/SLA queue with durable leases, fences, annotations, and reopen."""

    def enqueue(self, task: ReviewTask) -> EnqueuedReviewTask:
        """Insert or exactly replay one immutable review task."""

    def claim_next(
        self,
        *,
        worker_id: str,
        now_ns: int,
        lease_duration_ns: int,
    ) -> ReviewLease | None:
        """Claim the highest-priority pending or expired task."""

    def renew_lease(
        self,
        *,
        review_task_id: str,
        worker_id: str,
        lease_fence: int,
        now_ns: int,
        lease_duration_ns: int,
    ) -> ReviewLease:
        """Extend one live lease without changing its fence."""

    def submit_annotation(
        self,
        annotation: ReviewAnnotation,
        *,
        now_ns: int,
    ) -> SubmittedReviewAnnotation:
        """Atomically append an annotation and complete its fenced task attempt."""

    def reopen(self, command: ReviewReopenCommand) -> ReopenedReviewTask:
        """Reopen completed work while retaining its annotation history."""

    def get_task(self, review_task_id: str) -> ReviewTaskSnapshot | None:
        """Return independently visible queue state."""

    def list_open(self, *, limit: int | None = None) -> tuple[ReviewTaskSnapshot, ...]:
        """List pending and leased tasks in deterministic scheduling order.

        ``limit`` bounds the storage read when callers only need a queue head.
        """

    def list_overdue(self, *, now_ns: int) -> tuple[ReviewTaskSnapshot, ...]:
        """List incomplete tasks strictly beyond their SLA deadline."""

    def list_annotations(self, review_task_id: str) -> tuple[ReviewAnnotation, ...]:
        """Return append-only annotation history ordered by lease fence."""


__all__ = [
    "EnqueuedReviewTask",
    "ReopenedReviewTask",
    "ReviewQueue",
    "ReviewQueueError",
    "ReviewQueueErrorCode",
    "SubmittedReviewAnnotation",
]
