"""Deterministic nonblocking review routing for local canonical completions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from robata.adapters.sqlite_review_queue import SQLiteReviewQueue
from robata.application.canonical.media_quality_binding import LocalMediaQualityBinding
from robata.application.canonical.primary_completion import CommittedPrimaryCompletion
from robata.contracts.common import StrictModel
from robata.contracts.schema_registry import SchemaRegistry
from robata.ports.review_queue import ReviewQueue
from robata.review.models import (
    ReviewRequest,
    ReviewRoutingRule,
    ReviewSubject,
    ReviewTrigger,
    create_nonblocking_review_routing_policy,
)
from robata.review.routing import (
    NonBlockingReviewRouter,
    ReviewRoutingDisposition,
)

LOCAL_CANONICAL_REVIEW_POLICY_VERSION = "canonical-local-nonblocking-review-v2"
_NANOSECONDS_PER_SECOND = 1_000_000_000


class LocalReviewRoutingSummary(StrictModel):
    """Operator-visible state for best-effort local review routing."""

    model_version: Literal["canonical-local-review-routing-summary-v1"] = (
        "canonical-local-review-routing-summary-v1"
    )
    disposition: ReviewRoutingDisposition
    review_task_id: str | None = None
    failure_type: str | None = None


def route_local_review_after_completion(
    committed: CommittedPrimaryCompletion,
    *,
    state_root: Path,
    registry: SchemaRegistry,
    queue: ReviewQueue | None = None,
    media_quality_binding: LocalMediaQualityBinding | None = None,
) -> LocalReviewRoutingSummary:
    """Route a committed result without changing or withholding primary truth."""

    if not isinstance(committed, CommittedPrimaryCompletion):
        raise TypeError("committed must be a CommittedPrimaryCompletion")
    if not isinstance(state_root, Path):
        raise TypeError("state_root must be pathlib.Path")
    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be a SchemaRegistry")

    decision = committed.detail.output_decision
    reason_codes: tuple[str, ...]
    if media_quality_binding is not None and media_quality_binding.requires_review:
        trigger = ReviewTrigger.QA_DEGRADATION
        reason_codes = tuple(
            f"LOCAL_MEDIA_QUALITY_{item.flag.value}" for item in media_quality_binding.flag_counts
        )
        subject_type = "LOCAL_MEDIA_QUALITY_REPORT"
        subject_digest = media_quality_binding.report_semantic_sha256
        subject_id = f"media-quality-report:{subject_digest}"
    elif decision is None:
        if committed.detail.status != "NO_EVENTS":
            raise ValueError("committed canonical detail has no output decision")
        trigger = ReviewTrigger.REVIEW_SAMPLING
        reason_codes = ("LOCAL_CONFORMANCE_NO_EVENTS_SAMPLE",)
        subject_type = "CANONICAL_PRIMARY_COMPLETION"
        subject_digest = committed.completion.semantic_sha256
        subject_id = f"primary-completion:{subject_digest}"
    else:
        trigger, reason_codes = _review_reason(decision.decision)
        subject_type = "CANONICAL_OUTPUT_DECISION"
        subject_digest = decision.semantic_sha256
        subject_id = f"output-decision:{subject_digest}"
    request = ReviewRequest(
        request_id=str(
            uuid5(
                NAMESPACE_URL,
                ":".join(
                    (
                        "robata:canonical-local-review-request",
                        LOCAL_CANONICAL_REVIEW_POLICY_VERSION,
                        committed.processing_run.run_id,
                        subject_digest,
                        trigger.value,
                    )
                ),
            )
        ),
        subject=ReviewSubject(
            subject_type=subject_type,
            subject_id=subject_id,
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
                trigger=ReviewTrigger.QA_DEGRADATION,
                priority=5,
                sla_ns=2 * 60 * 60 * _NANOSECONDS_PER_SECOND,
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
    receipt = NonBlockingReviewRouter(policy=policy, queue=active_queue).route(request)
    return LocalReviewRoutingSummary(
        disposition=receipt.disposition,
        review_task_id=receipt.review_task_id,
        failure_type=receipt.failure_type,
    )


def failed_local_review_routing(error: Exception) -> LocalReviewRoutingSummary:
    """Represent setup failures after primary completion without raising them."""

    return LocalReviewRoutingSummary(
        disposition=ReviewRoutingDisposition.ROUTING_FAILED,
        failure_type=type(error).__name__,
    )


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
    "LocalReviewRoutingSummary",
    "failed_local_review_routing",
    "route_local_review_after_completion",
]
