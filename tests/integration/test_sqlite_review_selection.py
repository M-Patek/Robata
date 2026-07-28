from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier as ThreadBarrier
from uuid import UUID

import pytest

from robata.adapters.sqlite_review_selection import (
    ReviewSelectionStore,
    ReviewSelectionStoreIntegrityError,
)
from robata.application.canonical.active_learning_selection import (
    CanonicalActiveLearningSelectionBridge,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.review.active_learning import (
    ActiveLearningCandidate,
    ActiveLearningPoolSnapshot,
    ActiveLearningSelectionPolicy,
    ActiveLearningSelector,
    ActiveLearningSourceReference,
    ActiveLearningTermApplicability,
    ActiveLearningTermEvidence,
    ActiveLearningTermKind,
)
from robata.review.models import (
    ReviewAdjudication,
    ReviewRequest,
    ReviewRoutingRule,
    ReviewSubject,
    ReviewTask,
    ReviewTrigger,
    create_nonblocking_review_routing_policy,
    create_review_annotation,
    create_review_task,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _task(value: int) -> ReviewTask:
    routing = create_nonblocking_review_routing_policy(
        policy_version="selection-routing-v1",
        rules=(
            ReviewRoutingRule(
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                priority=10,
                sla_ns=100,
            ),
        ),
    )
    task = create_review_task(
        ReviewRequest(
            request_id=_uuid(value),
            subject=ReviewSubject(
                subject_type="EVENT_HYPOTHESIS",
                subject_id=f"event-hypothesis:{_digest(value + 100)}",
                recording_identity=_digest(value + 200),
            ),
            trigger=ReviewTrigger.LOW_CONFIDENCE,
            reason_codes=("LOW_CONFIDENCE",),
            requested_at_ns=1_000 + value,
        ),
        routing,
    )
    assert task is not None
    return task


def _candidate(task: ReviewTask, value: int) -> ActiveLearningCandidate:
    terms = tuple(
        ActiveLearningTermEvidence(
            kind=kind,
            applicability=ActiveLearningTermApplicability.APPLICABLE,
            value_millionths=900_000 - ordinal,
            source=ActiveLearningSourceReference(
                logical_key=f"selection-term:{_digest(value + ordinal)}",
                semantic_sha256=_digest(value + ordinal),
                exact_sha256=_digest(value + 1_000 + ordinal),
            ),
            reason_codes=(f"{kind.value}_OBSERVED",),
        )
        for ordinal, kind in enumerate(ActiveLearningTermKind)
    )
    return ActiveLearningCandidate.from_review_task(
        task,
        review_task_exact_sha256=exact_bytes_sha256(canonical_json_bytes(task)),
        terms=terms,
    )


def _decision(*tasks: ReviewTask):
    pool = ActiveLearningPoolSnapshot.create(
        pool_version="selection-pool-v1",
        candidates=tuple(
            _candidate(task, 100 + ordinal * 100) for ordinal, task in enumerate(tasks)
        ),
    )
    policy = ActiveLearningSelectionPolicy.create(
        policy_version="selection-policy-v1",
        eligible_triggers=(ReviewTrigger.LOW_CONFIDENCE,),
        ranking_terms=tuple(ActiveLearningTermKind),
    )
    return ActiveLearningSelector().select(pool=pool, policy=policy, budget=1)


def _annotation(task: ReviewTask):
    return create_review_annotation(
        task=task,
        lease_fence=1,
        lease_owner="selection-worker",
        reviewer_id="reviewer-1",
        adjudication=ReviewAdjudication(
            decision_code="ACCEPT",
            reason_codes=("EVIDENCE_VERIFIED",),
            comment="Late labels remain separate from the frozen selection.",
        ),
        authored_at_ns=2_000,
    )


def test_selection_store_replays_exact_decision_across_restart(tmp_path) -> None:
    decision = _decision(_task(1), _task(2))
    database = tmp_path / "selection.sqlite"

    fresh, replayed = ReviewSelectionStore(database).put_or_get(decision)
    replay, replayed_again = ReviewSelectionStore(database).put_or_get(decision)
    restarted = ReviewSelectionStore(database).get(decision.semantic_sha256)
    decisions = ReviewSelectionStore(database).list_decisions()

    assert fresh == decision
    assert replay == decision
    assert restarted == decision
    assert decisions == (decision,)
    assert replayed is False
    assert replayed_again is True


def test_concurrent_selection_persists_one_immutable_decision(tmp_path) -> None:
    first_task = _task(11)
    second_task = _task(12)
    expected = _decision(first_task, second_task)
    database = tmp_path / "selection.sqlite"
    store = ReviewSelectionStore(database)
    bridge = CanonicalActiveLearningSelectionBridge(
        selector=ActiveLearningSelector(),
        store=store,
    )
    barrier = ThreadBarrier(2)

    def select_at_once():
        barrier.wait()
        return bridge.select_and_publish(
            pool_version=expected.pool.pool_version,
            candidates=expected.pool.candidates,
            policy=expected.policy,
            budget=expected.budget,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _unused: select_at_once(), range(2)))

    assert tuple(decision for decision, _replayed in outcomes) == (expected, expected)
    assert sorted(replayed for _decision_value, replayed in outcomes) == [False, True]
    assert store.list_decisions() == (expected,)


def test_annotation_lineage_is_append_only_and_cannot_rewrite_selection(tmp_path) -> None:
    first_task = _task(3)
    second_task = _task(4)
    decision = _decision(first_task, second_task)
    task_by_id = {
        first_task.review_task_id: first_task,
        second_task.review_task_id: second_task,
    }
    selected_task = task_by_id[decision.selected_review_task_ids[0]]
    unselected_task = next(
        task for task_id, task in task_by_id.items() if task_id != selected_task.review_task_id
    )
    store = ReviewSelectionStore(tmp_path / "selection.sqlite")
    store.put_or_get(decision)

    annotation = _annotation(selected_task)
    assert store.append_annotation_lineage(decision=decision, annotation=annotation) is True
    assert store.append_annotation_lineage(decision=decision, annotation=annotation) is False
    assert store.list_annotations(decision.semantic_sha256) == (annotation,)
    assert store.get(decision.semantic_sha256) == decision

    with pytest.raises(ReviewSelectionStoreIntegrityError, match="selected by the frozen"):
        store.append_annotation_lineage(
            decision=decision,
            annotation=_annotation(unselected_task),
        )
