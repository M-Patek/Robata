"""Deterministic nonblocking review routing for local canonical completions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from robata.adapters.sqlite_review_queue import SQLiteReviewQueue
from robata.application.canonical.primary_completion import CommittedPrimaryCompletion
from robata.contracts.schema_registry import SchemaRegistry
from robata.ports.review_queue import ReviewQueue
from robata.review.models import (
    ReviewRequest,
    ReviewRoutingRule,
    ReviewSubject,
    ReviewTrigger,
    create_nonblocking_review_routing_policy,
)
from robata.review.routing import NonBlockingReviewRouter, ReviewRoutingReceipt

LOCAL_CANONICAL_REVIEW_POLICY_VERSION = "canonical-local-nonblocking-review-v1"
_NANOSECONDS_PER_SECOND = 1_000_000_000


def route_local_review_after_completion(
    committed: CommittedPrimaryCompletion,
    *,
    state_root: Path,
    registry: SchemaRegistry,
    queue: ReviewQueue | None = None,
) -> ReviewRoutingReceipt:
    """Route a committed result without changing or withholding primary truth."""

    if not isinstance(committed, CommittedPrimaryCompletion):
        raise TypeError("committed must be a CommittedPrimaryCompletion")
    if not isinstance(state_root, Path):
        raise TypeError("state_root must be pathlib.Path")
    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be a SchemaRegistry")

    decision = committed.detail.output_decision
    if decision is None:
        raise ValueError("committed canonical detail has no output decision")
    trigger, reason_codes = _review_reason(decision.decision)
    request = ReviewRequest(
        request_id=str(
            uuid5(
                NAMESPACE_URL,
                ":".join(
                    (
                        "robata:canonical-local-review-request",
                        LOCAL_CANONICAL_REVIEW_POLICY_VERSION,
                        committed.processing_run.run_id,
                        decision.semantic_sha256,
                        trigger.value,
                    )
                ),
            )
        ),
        subject=ReviewSubject(
            subject_type="CANONICAL_OUTPUT_DECISION",
            subject_id=f"output-decision:{decision.semantic_sha256}",
            recording_identity=committed.completion.recording_identity,
        ),
        trigger=trigger,
        reason_codes=reason_codes,
        requested_at_ns=_timestamp_ns(committed.processing_run.completed_at),
    )
    policy = create_nonblocking_review_routing_policy(
        policy_version=LOCAL_CANONICAL_REVIEW_POLICY_VERSION,
        rules=(
            ReviewRoutingRule(
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                priority=10,
                sla_ns=4 * 60 * 60 * _NANOSECONDS_PER_SECOND,
            ),
            ReviewRoutingRule(
                trigger=ReviewTrigger.REVIEW_SAMPLING,
                priority=100,
                sla_ns=24 * 60 * 60 * _NANOSECONDS_PER_SECOND,
            ),
        ),
    )
    active_queue = queue or SQLiteReviewQueue(
        state_root / "review-queue.sqlite3",
        registry=registry,
    )
    return NonBlockingReviewRouter(policy=policy, queue=active_queue).route(request)


def _review_reason(decision: str) -> tuple[ReviewTrigger, tuple[str, ...]]:
    if decision == "ABSTAINED":
        return ReviewTrigger.LOW_CONFIDENCE, ("CANONICAL_OUTPUT_ABSTAINED",)
    if decision in {"ADMITTED", "NO_EVENTS"}:
        return ReviewTrigger.REVIEW_SAMPLING, ("LOCAL_CONFORMANCE_SAMPLE",)
    raise ValueError(f"unsupported canonical output decision: {decision}")


def _timestamp_ns(value: str | None) -> int:
    if value is None:
        raise ValueError("committed processing run has no completion timestamp")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("processing-run completion timestamp must include a timezone")
    utc = parsed.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = utc - epoch
    return (
        delta.days * 86_400 * _NANOSECONDS_PER_SECOND
        + delta.seconds * _NANOSECONDS_PER_SECOND
        + delta.microseconds * 1_000
    )


__all__ = [
    "LOCAL_CANONICAL_REVIEW_POLICY_VERSION",
    "route_local_review_after_completion",
]
