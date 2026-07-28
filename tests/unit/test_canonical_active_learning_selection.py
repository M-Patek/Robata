"""Tests for the detached P13 canonical active-learning bridge."""

from __future__ import annotations

import pytest

import robata.application.canonical.active_learning_selection as selection_module
from robata.adapters.sqlite_review_selection import ReviewSelectionStore
from robata.application.canonical.active_learning_selection import (
    CanonicalActiveLearningSelectionBridge,
    LocalActiveLearningAnnotationLineageDisposition,
    LocalActiveLearningSelectionDisposition,
    dispatch_local_active_learning_selection,
    submit_local_review_annotation_with_active_learning_lineage,
)
from robata.ports.review_queue import SubmittedReviewAnnotation
from robata.review.active_learning import (
    ActiveLearningSelectionPolicy,
    ActiveLearningSelector,
    ActiveLearningSourceReference,
    ActiveLearningTermApplicability,
    ActiveLearningTermEvidence,
    ActiveLearningTermKind,
)
from robata.review.models import (
    ReviewAdjudication,
    ReviewAnnotation,
    ReviewRequest,
    ReviewRoutingRule,
    ReviewSubject,
    ReviewTask,
    ReviewTaskSnapshot,
    ReviewTaskStatus,
    ReviewTrigger,
    create_nonblocking_review_routing_policy,
    create_review_annotation,
    create_review_task,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _task(value: int = 1, *, priority: int = 10, requested_at_ns: int = 1_000) -> ReviewTask:
    routing = create_nonblocking_review_routing_policy(
        policy_version="canonical-selection-routing-v1",
        rules=(
            ReviewRoutingRule(
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                priority=priority,
                sla_ns=100,
            ),
        ),
    )
    task = create_review_task(
        ReviewRequest(
            request_id=f"00000000-0000-0000-0000-{value:012d}",
            subject=ReviewSubject(
                subject_type="EVENT_HYPOTHESIS",
                subject_id=f"event-hypothesis:{_digest(value + 1)}",
                recording_identity=_digest(value + 2),
            ),
            trigger=ReviewTrigger.LOW_CONFIDENCE,
            reason_codes=("LOW_CONFIDENCE",),
            requested_at_ns=requested_at_ns,
        ),
        routing,
    )
    assert task is not None
    return task


def _terms() -> tuple[ActiveLearningTermEvidence, ...]:
    return tuple(
        ActiveLearningTermEvidence(
            kind=kind,
            applicability=ActiveLearningTermApplicability.APPLICABLE,
            value_millionths=500_000 + ordinal,
            source=ActiveLearningSourceReference(
                logical_key=f"canonical-selection-evidence-v1:{_digest(100 + ordinal)}",
                semantic_sha256=_digest(100 + ordinal),
                exact_sha256=_digest(200 + ordinal),
            ),
            reason_codes=(f"{kind.value}_OBSERVED",),
        )
        for ordinal, kind in enumerate(ActiveLearningTermKind)
    )


def _annotation(task: ReviewTask) -> ReviewAnnotation:
    return create_review_annotation(
        task=task,
        lease_fence=1,
        lease_owner="canonical-selection-worker",
        reviewer_id="reviewer-1",
        adjudication=ReviewAdjudication(
            decision_code="ACCEPT",
            reason_codes=("EVIDENCE_VERIFIED",),
            comment="Detached lineage must not alter review acceptance.",
        ),
        authored_at_ns=2_000,
    )


class _SubmittingReviewQueue:
    def __init__(self) -> None:
        self.events: list[str] = []

    def submit_annotation(
        self,
        annotation: ReviewAnnotation,
        *,
        now_ns: int,
    ) -> SubmittedReviewAnnotation:
        self.events.append(f"submit:{now_ns}")
        return SubmittedReviewAnnotation(annotation=annotation, inserted=True)


def test_bridge_binds_routed_task_and_replays_persisted_selection(tmp_path) -> None:
    bridge = CanonicalActiveLearningSelectionBridge(
        selector=ActiveLearningSelector(),
        store=ReviewSelectionStore(tmp_path / "selection.sqlite"),
    )
    task = _task()
    candidate = bridge.candidate_from_routed_task(task, terms=_terms())
    policy = ActiveLearningSelectionPolicy.create(
        policy_version="canonical-selection-policy-v1",
        eligible_triggers=(ReviewTrigger.LOW_CONFIDENCE,),
        ranking_terms=tuple(ActiveLearningTermKind),
    )

    decision, replayed = bridge.select_and_publish(
        pool_version="canonical-selection-pool-v1",
        candidates=(candidate,),
        policy=policy,
        budget=1,
    )
    replay, replayed_again = bridge.select_and_publish(
        pool_version="canonical-selection-pool-v1",
        candidates=(candidate,),
        policy=policy,
        budget=1,
    )

    assert candidate.review_task_id == task.review_task_id
    assert candidate.review_task_semantic_sha256 == task.semantic_sha256
    assert decision.selected_review_task_ids == (task.review_task_id,)
    assert replay == decision
    assert replayed is False
    assert replayed_again is True


def test_annotation_lineage_store_failure_cannot_reject_submitted_annotation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task()
    annotation = _annotation(task)
    queue = _SubmittingReviewQueue()

    def unavailable_store(_database_path):  # type: ignore[no-untyped-def]
        queue.events.append("selection-store")
        raise RuntimeError("selection storage unavailable")

    monkeypatch.setattr(selection_module, "ReviewSelectionStore", unavailable_store)
    result = submit_local_review_annotation_with_active_learning_lineage(
        queue=queue,  # type: ignore[arg-type]
        state_root=tmp_path,
        annotation=annotation,
        now_ns=2_001,
    )

    assert queue.events == ["submit:2001", "selection-store"]
    assert result.submission == SubmittedReviewAnnotation(annotation=annotation, inserted=True)
    assert result.disposition is LocalActiveLearningAnnotationLineageDisposition.FAILED
    assert result.failure_type == "RuntimeError"


class _OpenReviewQueue:
    def __init__(self, *tasks: ReviewTask) -> None:
        self._snapshots = tuple(
            ReviewTaskSnapshot(task=task, status=ReviewTaskStatus.PENDING) for task in tasks
        )
        self.open_limits: list[int | None] = []

    def get_task(self, review_task_id: str) -> ReviewTaskSnapshot | None:
        return next(
            (
                snapshot
                for snapshot in self._snapshots
                if snapshot.task.review_task_id == review_task_id
            ),
            None,
        )

    def list_open(self, *, limit: int | None = None) -> tuple[ReviewTaskSnapshot, ...]:
        self.open_limits.append(limit)
        return self._snapshots if limit is None else self._snapshots[:limit]


def test_local_dispatch_bounds_pool_preserves_priority_and_does_not_zero_missing_terms(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _task(1, priority=1, requested_at_ns=1_000)
    excluded = _task(2, priority=2, requested_at_ns=2_000)
    routed = _task(3, priority=100, requested_at_ns=3_000)
    queue = _OpenReviewQueue(first, excluded, routed)
    monkeypatch.setattr(
        selection_module,
        "LOCAL_CANONICAL_ACTIVE_LEARNING_MAX_POOL_CANDIDATES",
        2,
    )

    selection = dispatch_local_active_learning_selection(
        queue=queue,  # type: ignore[arg-type]
        state_root=tmp_path,
        routed_task_id=routed.review_task_id,
    )

    assert selection.disposition is LocalActiveLearningSelectionDisposition.PERSISTED
    assert selection.decision_semantic_sha256 is not None
    decision = ReviewSelectionStore(tmp_path / "review-selection.sqlite3").get(
        selection.decision_semantic_sha256
    )
    assert decision is not None
    assert {item.review_task_id for item in decision.pool.candidates} == {
        first.review_task_id,
        routed.review_task_id,
    }
    assert queue.open_limits == [2]
    assert decision.selected_review_task_ids == (first.review_task_id,)
    routed_candidate = next(
        item for item in decision.pool.candidates if item.review_task_id == routed.review_task_id
    )
    assert routed_candidate.priority_evidence.priority == routed.priority
    terms = {item.kind: item for item in routed_candidate.terms}
    assert terms[ActiveLearningTermKind.RECENCY].value_millionths == 1_000_000
    assert terms[ActiveLearningTermKind.RECENCY].source is not None
    assert terms[ActiveLearningTermKind.RECENCY].source.semantic_sha256 == routed.semantic_sha256
    assert all(
        terms[kind].applicability is ActiveLearningTermApplicability.MISSING
        and terms[kind].value_millionths is None
        for kind in ActiveLearningTermKind
        if kind is not ActiveLearningTermKind.RECENCY
    )
