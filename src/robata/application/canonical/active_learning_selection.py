"""Detached canonical bridge for immutable P13 review-pool selection.

Review routing remains an independent best-effort operation after completion.
This bridge only freezes already-created review tasks and explicit term evidence,
then persists a decision through the independent selection store.  It never
calls a runner, changes queue priority, or changes primary completion state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from robata.adapters.sqlite_review_selection import ReviewSelectionStore
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.ports.review_queue import ReviewQueue, SubmittedReviewAnnotation
from robata.review.active_learning import (
    ActiveLearningCandidate,
    ActiveLearningPoolSnapshot,
    ActiveLearningSelectionDecision,
    ActiveLearningSelectionPolicy,
    ActiveLearningSelector,
    ActiveLearningSourceReference,
    ActiveLearningTermApplicability,
    ActiveLearningTermEvidence,
    ActiveLearningTermKind,
)
from robata.review.models import (
    ReviewAnnotation,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTrigger,
)
from robata.runtime.observability import RuntimeObserver, runtime_increment, runtime_span

LOCAL_CANONICAL_ACTIVE_LEARNING_POOL_VERSION: Final = "canonical-local-review-pool-v1"
LOCAL_CANONICAL_ACTIVE_LEARNING_POLICY_VERSION: Final = "canonical-local-review-policy-v1"
LOCAL_CANONICAL_ACTIVE_LEARNING_SELECTION_BUDGET: Final = 1
LOCAL_CANONICAL_ACTIVE_LEARNING_MAX_POOL_CANDIDATES: Final = 64
_LOCAL_REVIEW_TASK_EVIDENCE_NAMESPACE: Final = "canonical-local-review-task-v1"


class LocalActiveLearningSelectionDisposition(StrEnum):
    """Observable outcome of a sidecar selection attempt."""

    PERSISTED = "PERSISTED"
    REPLAYED = "REPLAYED"
    SKIPPED_ROUTED_TASK_NOT_OPEN = "SKIPPED_ROUTED_TASK_NOT_OPEN"
    FAILED = "FAILED"


class LocalActiveLearningAnnotationLineageDisposition(StrEnum):
    """Observable outcome of detached annotation-lineage persistence."""

    APPENDED = "APPENDED"
    REPLAYED = "REPLAYED"
    SKIPPED_TASK_NOT_SELECTED = "SKIPPED_TASK_NOT_SELECTED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class LocalActiveLearningAnnotationLineageSummary:
    """Accepted review submission plus its independent best-effort lineage result."""

    submission: SubmittedReviewAnnotation
    disposition: LocalActiveLearningAnnotationLineageDisposition
    matched_decision_semantic_sha256es: tuple[Sha256Digest, ...] = ()
    appended_decision_semantic_sha256es: tuple[Sha256Digest, ...] = ()
    failure_type: str | None = None


class LocalActiveLearningSelectionSummary(StrictModel):
    """A detached selection result that cannot alter review or completion state."""

    model_version: Literal["canonical-local-active-learning-selection-summary-v1"] = (
        "canonical-local-active-learning-selection-summary-v1"
    )
    disposition: LocalActiveLearningSelectionDisposition
    decision_semantic_sha256: Sha256Digest | None = None
    pool_semantic_sha256: Sha256Digest | None = None
    selected_review_task_ids: tuple[OpaqueUuid, ...] = ()
    failure_type: str | None = None


class CanonicalActiveLearningSelectionBridge:
    """Create and persist detached immutable decisions from routed review tasks."""

    def __init__(
        self,
        *,
        selector: ActiveLearningSelector,
        store: ReviewSelectionStore,
    ) -> None:
        if not isinstance(selector, ActiveLearningSelector):
            raise TypeError("selector must be an ActiveLearningSelector")
        if not isinstance(store, ReviewSelectionStore):
            raise TypeError("store must be a ReviewSelectionStore")
        self._selector = selector
        self._store = store

    @staticmethod
    def candidate_from_routed_task(
        task: ReviewTask,
        *,
        terms: Iterable[ActiveLearningTermEvidence],
    ) -> ActiveLearningCandidate:
        """Bind an existing task's exact bytes to explicit selection terms."""

        if not isinstance(task, ReviewTask):
            raise TypeError("task must be a ReviewTask")
        return ActiveLearningCandidate.from_review_task(
            task,
            review_task_exact_sha256=exact_bytes_sha256(canonical_json_bytes(task)),
            terms=terms,
        )

    def select_and_publish(
        self,
        *,
        pool_version: str,
        candidates: Iterable[ActiveLearningCandidate],
        policy: ActiveLearningSelectionPolicy,
        budget: int,
    ) -> tuple[ActiveLearningSelectionDecision, bool]:
        """Select once from a frozen pool and append its exact immutable decision."""

        pool = ActiveLearningPoolSnapshot.create(
            pool_version=pool_version,
            candidates=tuple(candidates),
        )
        decision = self._selector.select(pool=pool, policy=policy, budget=budget)
        return self._store.put_or_get(decision)


def submit_local_review_annotation_with_active_learning_lineage(
    *,
    queue: ReviewQueue,
    state_root: Path,
    annotation: ReviewAnnotation,
    now_ns: int,
    runtime_observer: RuntimeObserver | None = None,
) -> LocalActiveLearningAnnotationLineageSummary:
    """Submit review authority first, then best-effort append frozen-decision lineage.

    The queue remains authoritative for annotation acceptance and task completion.
    Selection persistence is intentionally detached: every selection-store failure is
    represented in the returned summary after the accepted submission is retained.
    """

    submission = queue.submit_annotation(annotation, now_ns=now_ns)
    try:
        with runtime_span(runtime_observer, "review.active_learning.annotation_lineage"):
            if not isinstance(state_root, Path):
                raise TypeError("state_root must be a pathlib.Path")
            store = ReviewSelectionStore(state_root / "review-selection.sqlite3")
            decisions = tuple(
                decision
                for decision in store.list_decisions()
                if submission.annotation.review_task_id in decision.selected_review_task_ids
            )
            if not decisions:
                result = LocalActiveLearningAnnotationLineageSummary(
                    submission=submission,
                    disposition=(
                        LocalActiveLearningAnnotationLineageDisposition.SKIPPED_TASK_NOT_SELECTED
                    ),
                )
            else:
                appended = tuple(
                    decision.semantic_sha256
                    for decision in decisions
                    if store.append_annotation_lineage(
                        decision=decision,
                        annotation=submission.annotation,
                    )
                )
                result = LocalActiveLearningAnnotationLineageSummary(
                    submission=submission,
                    disposition=(
                        LocalActiveLearningAnnotationLineageDisposition.APPENDED
                        if appended
                        else LocalActiveLearningAnnotationLineageDisposition.REPLAYED
                    ),
                    matched_decision_semantic_sha256es=tuple(
                        decision.semantic_sha256 for decision in decisions
                    ),
                    appended_decision_semantic_sha256es=appended,
                )
    except Exception as error:
        result = LocalActiveLearningAnnotationLineageSummary(
            submission=submission,
            disposition=LocalActiveLearningAnnotationLineageDisposition.FAILED,
            failure_type=type(error).__name__,
        )
    runtime_increment(
        runtime_observer,
        "review.active_learning_annotation_lineage_outcomes",
        attributes={"disposition": result.disposition.value},
    )
    return result


def dispatch_local_active_learning_selection(
    *,
    queue: ReviewQueue,
    state_root: Path,
    routed_task_id: str,
    runtime_observer: RuntimeObserver | None = None,
) -> LocalActiveLearningSelectionSummary:
    """Freeze and persist a bounded pool after durable review routing.

    The local routing path has no scored uncertainty, disagreement, coverage, or
    diversity evidence.  Those dimensions are retained as missing evidence rather
    than silently becoming zero.  The only local ranking term is an explicit,
    deterministic recency rank derived from the immutable review tasks in this
    frozen pool.  Existing queue priority remains the selector's first rank layer.
    """

    try:
        if not isinstance(state_root, Path):
            raise TypeError("state_root must be a pathlib.Path")
        if not isinstance(routed_task_id, str) or not routed_task_id:
            raise TypeError("routed_task_id must be a nonempty string")
        with runtime_span(runtime_observer, "review.active_learning.select"):
            tasks = _bounded_open_tasks(queue=queue, routed_task_id=routed_task_id)
            if not tasks:
                result = LocalActiveLearningSelectionSummary(
                    disposition=(
                        LocalActiveLearningSelectionDisposition.SKIPPED_ROUTED_TASK_NOT_OPEN
                    )
                )
            else:
                recency_values = _recency_values(tasks)
                bridge = CanonicalActiveLearningSelectionBridge(
                    selector=ActiveLearningSelector(),
                    store=ReviewSelectionStore(state_root / "review-selection.sqlite3"),
                )
                decision, replayed = bridge.select_and_publish(
                    pool_version=LOCAL_CANONICAL_ACTIVE_LEARNING_POOL_VERSION,
                    candidates=tuple(
                        bridge.candidate_from_routed_task(
                            task,
                            terms=_local_routed_task_terms(
                                task,
                                recency_value=recency_values[task.review_task_id],
                            ),
                        )
                        for task in tasks
                    ),
                    policy=_local_active_learning_policy(),
                    budget=LOCAL_CANONICAL_ACTIVE_LEARNING_SELECTION_BUDGET,
                )
                result = LocalActiveLearningSelectionSummary(
                    disposition=(
                        LocalActiveLearningSelectionDisposition.REPLAYED
                        if replayed
                        else LocalActiveLearningSelectionDisposition.PERSISTED
                    ),
                    decision_semantic_sha256=decision.semantic_sha256,
                    pool_semantic_sha256=decision.pool.semantic_sha256,
                    selected_review_task_ids=decision.selected_review_task_ids,
                )
    except Exception as error:
        result = LocalActiveLearningSelectionSummary(
            disposition=LocalActiveLearningSelectionDisposition.FAILED,
            failure_type=type(error).__name__,
        )
    runtime_increment(
        runtime_observer,
        "review.active_learning_selection_outcomes",
        attributes={"disposition": result.disposition.value},
    )
    return result


def _bounded_open_tasks(*, queue: ReviewQueue, routed_task_id: str) -> tuple[ReviewTask, ...]:
    """Freeze at most one bounded queue head while retaining the routed task."""

    routed_snapshot = queue.get_task(routed_task_id)
    if routed_snapshot is None or routed_snapshot.status is ReviewTaskStatus.COMPLETED:
        return ()
    if LOCAL_CANONICAL_ACTIVE_LEARNING_MAX_POOL_CANDIDATES < 1:
        raise ValueError("local active-learning pool limit must be positive")
    snapshots = queue.list_open(limit=LOCAL_CANONICAL_ACTIVE_LEARNING_MAX_POOL_CANDIDATES)
    tasks = tuple(item.task for item in snapshots)
    task_ids = tuple(task.review_task_id for task in tasks)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("review queue returned duplicate open review task IDs")
    if any(task.review_task_id == routed_task_id for task in tasks):
        return tuple(sorted(tasks, key=lambda task: task.review_task_id))
    return tuple(
        sorted(
            (
                *tasks[: LOCAL_CANONICAL_ACTIVE_LEARNING_MAX_POOL_CANDIDATES - 1],
                routed_snapshot.task,
            ),
            key=lambda task: task.review_task_id,
        )
    )


def _recency_values(tasks: tuple[ReviewTask, ...]) -> dict[str, int]:
    """Rank recent tasks without assigning a synthetic missing-data value."""

    if not tasks:
        raise ValueError("recency requires at least one review task")
    newest_first = tuple(
        sorted(tasks, key=lambda task: (-task.requested_at_ns, task.review_task_id))
    )
    count = len(newest_first)
    return {
        task.review_task_id: ((count - ordinal) * 1_000_000) // count
        for ordinal, task in enumerate(newest_first)
    }


def _local_routed_task_terms(
    task: ReviewTask,
    *,
    recency_value: int,
) -> tuple[ActiveLearningTermEvidence, ...]:
    """Retain unavailable local terms as missing and cite task-backed recency."""

    task_exact_sha256 = exact_bytes_sha256(canonical_json_bytes(task))
    recency_source = ActiveLearningSourceReference(
        logical_key=f"{_LOCAL_REVIEW_TASK_EVIDENCE_NAMESPACE}:{task.semantic_sha256}",
        semantic_sha256=task.semantic_sha256,
        exact_sha256=task_exact_sha256,
        source_revision=task.routing_policy_version,
    )
    terms: list[ActiveLearningTermEvidence] = []
    for kind in ActiveLearningTermKind:
        if kind is ActiveLearningTermKind.RECENCY:
            terms.append(
                ActiveLearningTermEvidence(
                    kind=kind,
                    applicability=ActiveLearningTermApplicability.APPLICABLE,
                    value_millionths=recency_value,
                    source=recency_source,
                    reason_codes=("LOCAL_REVIEW_REQUESTED_AT_RECENCY",),
                )
            )
        else:
            terms.append(
                ActiveLearningTermEvidence(
                    kind=kind,
                    applicability=ActiveLearningTermApplicability.MISSING,
                    reason_codes=(f"LOCAL_REVIEW_{kind.value}_NOT_RECORDED",),
                )
            )
    return tuple(terms)


def _local_active_learning_policy() -> ActiveLearningSelectionPolicy:
    """Return the local policy without claiming unavailable score-model revisions."""

    return ActiveLearningSelectionPolicy.create(
        policy_version=LOCAL_CANONICAL_ACTIVE_LEARNING_POLICY_VERSION,
        eligible_triggers=tuple(ReviewTrigger),
        ranking_terms=(ActiveLearningTermKind.RECENCY,),
    )


__all__ = [
    "LOCAL_CANONICAL_ACTIVE_LEARNING_MAX_POOL_CANDIDATES",
    "LOCAL_CANONICAL_ACTIVE_LEARNING_POLICY_VERSION",
    "LOCAL_CANONICAL_ACTIVE_LEARNING_POOL_VERSION",
    "LOCAL_CANONICAL_ACTIVE_LEARNING_SELECTION_BUDGET",
    "CanonicalActiveLearningSelectionBridge",
    "LocalActiveLearningAnnotationLineageDisposition",
    "LocalActiveLearningAnnotationLineageSummary",
    "LocalActiveLearningSelectionDisposition",
    "LocalActiveLearningSelectionSummary",
    "dispatch_local_active_learning_selection",
    "submit_local_review_annotation_with_active_learning_lineage",
]
