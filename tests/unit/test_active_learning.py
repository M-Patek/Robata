"""Focused tests for immutable P13 active-learning selection."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.review.active_learning import (
    ActiveLearningCandidate,
    ActiveLearningCandidateDisposition,
    ActiveLearningModelRevision,
    ActiveLearningPoolSnapshot,
    ActiveLearningSelectionPolicy,
    ActiveLearningSelector,
    ActiveLearningSourceReference,
    ActiveLearningTermApplicability,
    ActiveLearningTermEvidence,
    ActiveLearningTermKind,
    ExistingReviewPriorityEvidence,
    verify_active_learning_selection_decision,
)
from robata.review.models import ReviewTrigger


def _digest(value: int) -> str:
    return f"{value:064x}"


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _source(value: int) -> ActiveLearningSourceReference:
    digest = _digest(value)
    return ActiveLearningSourceReference(
        logical_key=f"active-learning-source-v1:{digest}",
        semantic_sha256=digest,
        exact_sha256=_digest(value + 1_000),
        source_revision="source-v1",
    )


def _term(
    kind: ActiveLearningTermKind,
    *,
    value: int,
    applicability: ActiveLearningTermApplicability = ActiveLearningTermApplicability.APPLICABLE,
    source_value: int,
) -> ActiveLearningTermEvidence:
    if applicability is ActiveLearningTermApplicability.APPLICABLE:
        return ActiveLearningTermEvidence(
            kind=kind,
            applicability=applicability,
            value_millionths=value,
            source=_source(source_value),
            reason_codes=(f"{kind.value}_OBSERVED",),
        )
    return ActiveLearningTermEvidence(
        kind=kind,
        applicability=applicability,
        reason_codes=(f"{kind.value}_{applicability.value}",),
    )


def _candidate(
    value: int,
    *,
    priority: int = 10,
    trigger: ReviewTrigger = ReviewTrigger.REVIEW_SAMPLING,
    term_value: int = 500_000,
    applicability: dict[ActiveLearningTermKind, ActiveLearningTermApplicability] | None = None,
) -> ActiveLearningCandidate:
    applicability = applicability or {}
    terms = tuple(
        _term(
            kind,
            value=term_value,
            applicability=applicability.get(kind, ActiveLearningTermApplicability.APPLICABLE),
            source_value=value * 100 + ordinal,
        )
        for ordinal, kind in enumerate(ActiveLearningTermKind, start=1)
    )
    return ActiveLearningCandidate.create(
        review_task_id=_uuid(value),
        review_task_semantic_sha256=_digest(value + 10),
        review_task_exact_sha256=_digest(value + 20),
        recording_identity=_digest(value + 30),
        priority_evidence=ExistingReviewPriorityEvidence(
            trigger=trigger,
            priority=priority,
            reason_codes=("EXISTING_ROUTING_PRIORITY",),
        ),
        terms=terms,
    )


def _policy(
    *,
    ranking_terms: tuple[ActiveLearningTermKind, ...] = tuple(ActiveLearningTermKind),
) -> ActiveLearningSelectionPolicy:
    return ActiveLearningSelectionPolicy.create(
        policy_version="active-learning-policy-v1",
        eligible_triggers=(ReviewTrigger.REVIEW_SAMPLING, ReviewTrigger.LOW_CONFIDENCE),
        ranking_terms=ranking_terms,
        model_revisions=(
            ActiveLearningModelRevision(
                model_name="quality-model",
                model_version="quality-model-v1",
                model_semantic_sha256=_digest(9_000),
            ),
        ),
    )


def _pool(*candidates: ActiveLearningCandidate) -> ActiveLearningPoolSnapshot:
    return ActiveLearningPoolSnapshot.create(
        pool_version="active-learning-pool-v1",
        candidates=candidates,
    )


def _ordered_task_ids(decision) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
    return tuple(
        item.candidate.review_task_id
        for item in sorted(
            (item for item in decision.candidate_decisions if item.rank is not None),
            key=lambda item: item.rank,
        )
    )


def test_pool_and_decision_replay_are_independent_of_input_order() -> None:
    first = _candidate(2, priority=20, term_value=900_000)
    second = _candidate(1, priority=5, term_value=100_000)
    policy = _policy()

    forward = _pool(first, second)
    reversed_pool = _pool(second, first)
    selector = ActiveLearningSelector()
    forward_decision = selector.select(pool=forward, policy=policy, budget=2)
    replayed = selector.select(pool=reversed_pool, policy=policy, budget=2)

    assert forward == reversed_pool
    assert forward_decision == replayed
    assert forward_decision.selected_review_task_ids == (_uuid(1), _uuid(2))
    assert verify_active_learning_selection_decision(forward_decision) == forward_decision


def test_existing_priority_is_first_layer_and_task_id_is_final_tie_break() -> None:
    same_priority_later_id = _candidate(9, priority=1, term_value=100_000)
    same_priority_earlier_id = _candidate(4, priority=1, term_value=100_000)
    lower_priority_with_higher_terms = _candidate(2, priority=5, term_value=1_000_000)

    decision = ActiveLearningSelector().select(
        pool=_pool(
            lower_priority_with_higher_terms,
            same_priority_later_id,
            same_priority_earlier_id,
        ),
        policy=_policy(),
        budget=3,
    )

    assert _ordered_task_ids(decision) == (_uuid(4), _uuid(9), _uuid(2))
    assert decision.selected_review_task_ids == (_uuid(4), _uuid(9), _uuid(2))


def test_zero_and_exhausted_budget_keep_ranked_candidates_and_reasons() -> None:
    pool = _pool(_candidate(1), _candidate(2), _candidate(3))
    selector = ActiveLearningSelector()

    zero = selector.select(pool=pool, policy=_policy(), budget=0)
    one = selector.select(pool=pool, policy=_policy(), budget=1)

    assert zero.selected_review_task_ids == ()
    assert all(
        item.disposition is ActiveLearningCandidateDisposition.BUDGET_EXHAUSTED
        for item in zero.candidate_decisions
    )
    assert all("BUDGET_ZERO" in item.reason_codes for item in zero.candidate_decisions)
    assert tuple(item.rank for item in zero.candidate_decisions) == (1, 2, 3)

    assert len(one.selected_review_task_ids) == 1
    assert _ordered_task_ids(one)[:1] == one.selected_review_task_ids
    assert [item.disposition for item in one.candidate_decisions].count(
        ActiveLearningCandidateDisposition.BUDGET_EXHAUSTED
    ) == 2


def test_pool_rejects_duplicate_review_task_identity() -> None:
    candidate = _candidate(1)

    with pytest.raises(ValueError, match="unique review task IDs"):
        _pool(candidate, candidate)


def test_missing_or_inapplicable_required_terms_are_not_defaulted_to_zero() -> None:
    missing = _candidate(
        1,
        applicability={
            ActiveLearningTermKind.UNCERTAINTY: ActiveLearningTermApplicability.MISSING,
        },
    )
    not_applicable = _candidate(
        2,
        applicability={
            ActiveLearningTermKind.DIVERSITY: ActiveLearningTermApplicability.NOT_APPLICABLE,
        },
    )
    decision = ActiveLearningSelector().select(
        pool=_pool(missing, not_applicable),
        policy=_policy(),
        budget=2,
    )
    by_task_id = {item.candidate.review_task_id: item for item in decision.candidate_decisions}

    assert decision.selected_review_task_ids == ()
    assert (
        by_task_id[_uuid(1)].disposition is ActiveLearningCandidateDisposition.MISSING_REQUIRED_TERM
    )
    assert "MISSING_REQUIRED_UNCERTAINTY" in by_task_id[_uuid(1)].reason_codes
    assert (
        by_task_id[_uuid(2)].disposition
        is ActiveLearningCandidateDisposition.NOT_APPLICABLE_REQUIRED_TERM
    )
    assert "NOT_APPLICABLE_REQUIRED_DIVERSITY" in by_task_id[_uuid(2)].reason_codes
    with pytest.raises(ValidationError, match="cannot carry a default value"):
        ActiveLearningTermEvidence(
            kind=ActiveLearningTermKind.UNCERTAINTY,
            applicability=ActiveLearningTermApplicability.MISSING,
            value_millionths=0,
            reason_codes=("UNCERTAINTY_MISSING",),
        )


def test_late_pool_inputs_cannot_change_a_frozen_verified_decision() -> None:
    policy = _policy()
    initial = ActiveLearningSelector().select(
        pool=_pool(_candidate(1, priority=10)),
        policy=policy,
        budget=1,
    )
    late_pool = _pool(_candidate(1, priority=10), _candidate(2, priority=0))
    later = ActiveLearningSelector().select(pool=late_pool, policy=policy, budget=1)

    assert initial.selected_review_task_ids == (_uuid(1),)
    assert later.selected_review_task_ids == (_uuid(2),)
    assert verify_active_learning_selection_decision(initial) == initial

    tampered = initial.model_copy(update={"pool": late_pool})
    with pytest.raises(ValueError):
        verify_active_learning_selection_decision(tampered)
