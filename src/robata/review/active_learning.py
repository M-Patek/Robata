"""Immutable, nonblocking active-learning selection over review work.

The selector is deliberately independent from review routing and queue ownership.
It freezes caller-supplied review candidates into a content-addressed pool, ranks
only evidence that the policy explicitly requires, and records an immutable
decision for every pool member.  It never creates, claims, reorders, or blocks a
review task.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from robata.contracts.common import INT64_MAX, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.review.models import ReviewTask, ReviewTrigger

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
ReasonCode = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$",
    ),
]
NonNegativeInteger = Annotated[int, Field(strict=True, ge=0, le=INT64_MAX)]
PositiveInteger = Annotated[int, Field(strict=True, ge=1, le=INT64_MAX)]
Millionths = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]

ACTIVE_LEARNING_CANDIDATE_PROJECTION_VERSION: Final = "active-learning-candidate-semantic-v1"
ACTIVE_LEARNING_POOL_PROJECTION_VERSION: Final = "active-learning-pool-semantic-v1"
ACTIVE_LEARNING_POLICY_PROJECTION_VERSION: Final = "active-learning-policy-semantic-v1"
ACTIVE_LEARNING_DECISION_PROJECTION_VERSION: Final = "active-learning-decision-semantic-v1"

ACTIVE_LEARNING_CANDIDATE_KEY_NAMESPACE: Final = "active-learning-candidate-v1"
ACTIVE_LEARNING_POOL_KEY_NAMESPACE: Final = "active-learning-pool-v1"
ACTIVE_LEARNING_POLICY_KEY_NAMESPACE: Final = "active-learning-policy-v1"
ACTIVE_LEARNING_DECISION_KEY_NAMESPACE: Final = "active-learning-decision-v1"


class ActiveLearningTermKind(StrEnum):
    """Evidence dimensions retained for every candidate in a frozen pool."""

    UNCERTAINTY = "UNCERTAINTY"
    DISAGREEMENT = "DISAGREEMENT"
    COVERAGE = "COVERAGE"
    DIVERSITY = "DIVERSITY"
    RECENCY = "RECENCY"


class ActiveLearningTermApplicability(StrEnum):
    """Whether a term has a usable value; missing evidence never becomes zero."""

    APPLICABLE = "APPLICABLE"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ActiveLearningCandidateDisposition(StrEnum):
    """Immutable disposition of one candidate in one selection decision."""

    SELECTED = "SELECTED"
    INELIGIBLE_TRIGGER = "INELIGIBLE_TRIGGER"
    MISSING_REQUIRED_TERM = "MISSING_REQUIRED_TERM"
    NOT_APPLICABLE_REQUIRED_TERM = "NOT_APPLICABLE_REQUIRED_TERM"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ActiveLearningSourceReference(StrictModel):
    """Exact source citation for an applicable ranking term."""

    logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    exact_sha256: Sha256Digest
    source_revision: SchemaVersion | None = None


class ActiveLearningTermEvidence(StrictModel):
    """One auditable fixed-point term for one review candidate."""

    kind: ActiveLearningTermKind
    applicability: ActiveLearningTermApplicability
    value_millionths: Millionths | None = None
    source: ActiveLearningSourceReference | None = None
    reason_codes: tuple[ReasonCode, ...]

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("term reason_codes must be nonempty, unique, and sorted")
        return value

    @model_validator(mode="after")
    def validate_applicability(self) -> Self:
        if self.applicability is ActiveLearningTermApplicability.APPLICABLE:
            if self.value_millionths is None or self.source is None:
                raise ValueError("applicable terms require a value and exact source reference")
        elif self.value_millionths is not None or self.source is not None:
            raise ValueError("missing or inapplicable terms cannot carry a default value or source")
        return self


class ExistingReviewPriorityEvidence(StrictModel):
    """Original nonblocking routing priority, retained ahead of new ranking terms."""

    trigger: ReviewTrigger
    priority: NonNegativeInteger
    reason_codes: tuple[ReasonCode, ...]

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("priority reason_codes must be nonempty, unique, and sorted")
        return value


def _term_order(value: ActiveLearningTermKind) -> int:
    return tuple(ActiveLearningTermKind).index(value)


def _canonical_terms(
    terms: Iterable[ActiveLearningTermEvidence],
) -> tuple[ActiveLearningTermEvidence, ...]:
    return tuple(sorted(tuple(terms), key=lambda item: _term_order(item.kind)))


def _canonical_reason_codes(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _candidate_projection_values(
    *,
    review_task_id: str,
    review_task_semantic_sha256: str,
    review_task_exact_sha256: str,
    recording_identity: str,
    priority_evidence: ExistingReviewPriorityEvidence,
    terms: tuple[ActiveLearningTermEvidence, ...],
) -> dict[str, object]:
    return {
        "semantic_projection_version": ACTIVE_LEARNING_CANDIDATE_PROJECTION_VERSION,
        "review_task_id": review_task_id,
        "review_task_semantic_sha256": review_task_semantic_sha256,
        "review_task_exact_sha256": review_task_exact_sha256,
        "recording_identity": recording_identity,
        "existing_review_priority": {
            "trigger": priority_evidence.trigger.value,
            "priority": priority_evidence.priority,
            "reason_codes": list(priority_evidence.reason_codes),
        },
        "terms": [
            {
                "kind": term.kind.value,
                "applicability": term.applicability.value,
                "value_millionths": term.value_millionths,
                "source": None if term.source is None else term.source.model_dump(mode="json"),
                "reason_codes": list(term.reason_codes),
            }
            for term in terms
        ],
    }


class ActiveLearningCandidate(StrictModel):
    """A task plus its complete, frozen active-learning evidence vector."""

    model_version: Literal["active-learning-candidate-v1"] = "active-learning-candidate-v1"
    review_task_id: OpaqueUuid
    review_task_semantic_sha256: Sha256Digest
    review_task_exact_sha256: Sha256Digest
    recording_identity: Sha256Digest
    priority_evidence: ExistingReviewPriorityEvidence
    terms: tuple[ActiveLearningTermEvidence, ...]
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey

    @classmethod
    def create(
        cls,
        *,
        review_task_id: str,
        review_task_semantic_sha256: str,
        review_task_exact_sha256: str,
        recording_identity: str,
        priority_evidence: ExistingReviewPriorityEvidence,
        terms: Iterable[ActiveLearningTermEvidence],
    ) -> Self:
        """Freeze one candidate independently of queue state or wall-clock time."""

        canonical_terms = _canonical_terms(terms)
        digest = semantic_sha256(
            _candidate_projection_values(
                review_task_id=review_task_id,
                review_task_semantic_sha256=review_task_semantic_sha256,
                review_task_exact_sha256=review_task_exact_sha256,
                recording_identity=recording_identity,
                priority_evidence=priority_evidence,
                terms=canonical_terms,
            )
        )
        return cls(
            review_task_id=review_task_id,
            review_task_semantic_sha256=review_task_semantic_sha256,
            review_task_exact_sha256=review_task_exact_sha256,
            recording_identity=recording_identity,
            priority_evidence=priority_evidence,
            terms=canonical_terms,
            semantic_sha256=digest,
            logical_key=f"{ACTIVE_LEARNING_CANDIDATE_KEY_NAMESPACE}:{digest}",
        )

    @classmethod
    def from_review_task(
        cls,
        task: ReviewTask,
        *,
        review_task_exact_sha256: str,
        terms: Iterable[ActiveLearningTermEvidence],
    ) -> Self:
        """Create a candidate while retaining the task's exact routing inputs."""

        if not isinstance(task, ReviewTask):
            raise TypeError("task must be a ReviewTask")
        return cls.create(
            review_task_id=task.review_task_id,
            review_task_semantic_sha256=task.semantic_sha256,
            review_task_exact_sha256=review_task_exact_sha256,
            recording_identity=task.subject.recording_identity,
            priority_evidence=ExistingReviewPriorityEvidence(
                trigger=task.trigger,
                priority=task.priority,
                reason_codes=task.reason_codes,
            ),
            terms=terms,
        )

    @model_validator(mode="after")
    def validate_identity_and_terms(self) -> Self:
        expected_terms = tuple(ActiveLearningTermKind)
        if tuple(term.kind for term in self.terms) != expected_terms:
            raise ValueError("candidate terms must contain every kind once in canonical order")
        expected_digest = semantic_sha256(active_learning_candidate_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("candidate semantic_sha256 does not match frozen evidence")
        expected_key = f"{ACTIVE_LEARNING_CANDIDATE_KEY_NAMESPACE}:{expected_digest}"
        if self.logical_key != expected_key:
            raise ValueError("candidate logical_key does not match semantic_sha256")
        return self


def active_learning_candidate_projection(candidate: ActiveLearningCandidate) -> dict[str, object]:
    """Return the immutable semantic projection for one pool member."""

    if not isinstance(candidate, ActiveLearningCandidate):
        raise TypeError("candidate must be an ActiveLearningCandidate")
    return _candidate_projection_values(
        review_task_id=candidate.review_task_id,
        review_task_semantic_sha256=candidate.review_task_semantic_sha256,
        review_task_exact_sha256=candidate.review_task_exact_sha256,
        recording_identity=candidate.recording_identity,
        priority_evidence=candidate.priority_evidence,
        terms=candidate.terms,
    )


def _pool_projection_values(
    *,
    pool_version: str,
    candidates: tuple[ActiveLearningCandidate, ...],
) -> dict[str, object]:
    return {
        "semantic_projection_version": ACTIVE_LEARNING_POOL_PROJECTION_VERSION,
        "pool_version": pool_version,
        "candidates": [
            {
                "review_task_id": item.review_task_id,
                "candidate_semantic_sha256": item.semantic_sha256,
                "candidate_logical_key": item.logical_key,
            }
            for item in candidates
        ],
    }


class ActiveLearningPoolSnapshot(StrictModel):
    """Immutable candidate pool, independent of mutable task lease state."""

    model_version: Literal["active-learning-pool-v1"] = "active-learning-pool-v1"
    pool_version: SchemaVersion
    candidates: tuple[ActiveLearningCandidate, ...]
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey

    @classmethod
    def create(
        cls,
        *,
        pool_version: SchemaVersion,
        candidates: Iterable[ActiveLearningCandidate],
    ) -> Self:
        """Canonicalize a complete eligibility snapshot before selection starts."""

        canonical_candidates = tuple(
            sorted(tuple(candidates), key=lambda item: item.review_task_id)
        )
        digest = semantic_sha256(
            _pool_projection_values(pool_version=pool_version, candidates=canonical_candidates)
        )
        return cls(
            pool_version=pool_version,
            candidates=canonical_candidates,
            semantic_sha256=digest,
            logical_key=f"{ACTIVE_LEARNING_POOL_KEY_NAMESPACE}:{digest}",
        )

    @model_validator(mode="after")
    def validate_identity_and_members(self) -> Self:
        if self.candidates != tuple(sorted(self.candidates, key=lambda item: item.review_task_id)):
            raise ValueError("pool candidates must be ordered by review_task_id")
        task_ids = tuple(item.review_task_id for item in self.candidates)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("pool candidates must have unique review task IDs")
        task_digests = tuple(item.review_task_semantic_sha256 for item in self.candidates)
        if len(task_digests) != len(set(task_digests)):
            raise ValueError("pool candidates must have unique review task semantic digests")
        expected_digest = semantic_sha256(active_learning_pool_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("pool semantic_sha256 does not match frozen candidates")
        expected_key = f"{ACTIVE_LEARNING_POOL_KEY_NAMESPACE}:{expected_digest}"
        if self.logical_key != expected_key:
            raise ValueError("pool logical_key does not match semantic_sha256")
        return self


def active_learning_pool_projection(pool: ActiveLearningPoolSnapshot) -> dict[str, object]:
    """Return the semantic projection for an immutable review pool."""

    if not isinstance(pool, ActiveLearningPoolSnapshot):
        raise TypeError("pool must be an ActiveLearningPoolSnapshot")
    return _pool_projection_values(pool_version=pool.pool_version, candidates=pool.candidates)


class ActiveLearningModelRevision(StrictModel):
    """One model or feature revision bound into a selection policy."""

    model_name: NonEmptyString
    model_version: SchemaVersion
    model_semantic_sha256: Sha256Digest


def _policy_projection_values(
    *,
    policy_version: str,
    eligible_triggers: tuple[ReviewTrigger, ...],
    ranking_terms: tuple[ActiveLearningTermKind, ...],
    model_revisions: tuple[ActiveLearningModelRevision, ...],
) -> dict[str, object]:
    return {
        "semantic_projection_version": ACTIVE_LEARNING_POLICY_PROJECTION_VERSION,
        "policy_version": policy_version,
        "eligible_triggers": [item.value for item in eligible_triggers],
        "existing_priority_order": "ASCENDING",
        "ranking_terms": [item.value for item in ranking_terms],
        "ranking_term_order": "DESCENDING_MILLIONTHS",
        "final_tie_break": "REVIEW_TASK_ID_ASCENDING",
        "model_revisions": [item.model_dump(mode="json") for item in model_revisions],
    }


class ActiveLearningSelectionPolicy(StrictModel):
    """Versioned ranking policy with no implicit score substitution."""

    model_version: Literal["active-learning-policy-v1"] = "active-learning-policy-v1"
    policy_version: SchemaVersion
    eligible_triggers: tuple[ReviewTrigger, ...]
    ranking_terms: tuple[ActiveLearningTermKind, ...]
    model_revisions: tuple[ActiveLearningModelRevision, ...] = ()
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey

    @classmethod
    def create(
        cls,
        *,
        policy_version: SchemaVersion,
        eligible_triggers: Iterable[ReviewTrigger],
        ranking_terms: Iterable[ActiveLearningTermKind],
        model_revisions: Iterable[ActiveLearningModelRevision] = (),
    ) -> Self:
        """Freeze the policy, including model revisions and tie-breaking behavior."""

        canonical_triggers = tuple(sorted(tuple(eligible_triggers), key=lambda item: item.value))
        ordered_terms = tuple(ranking_terms)
        canonical_models = tuple(
            sorted(tuple(model_revisions), key=lambda item: (item.model_name, item.model_version))
        )
        digest = semantic_sha256(
            _policy_projection_values(
                policy_version=policy_version,
                eligible_triggers=canonical_triggers,
                ranking_terms=ordered_terms,
                model_revisions=canonical_models,
            )
        )
        return cls(
            policy_version=policy_version,
            eligible_triggers=canonical_triggers,
            ranking_terms=ordered_terms,
            model_revisions=canonical_models,
            semantic_sha256=digest,
            logical_key=f"{ACTIVE_LEARNING_POLICY_KEY_NAMESPACE}:{digest}",
        )

    @model_validator(mode="after")
    def validate_identity_and_order(self) -> Self:
        if not self.eligible_triggers:
            raise ValueError("selection policy requires at least one eligible trigger")
        if self.eligible_triggers != tuple(
            sorted(set(self.eligible_triggers), key=lambda item: item.value)
        ):
            raise ValueError("eligible triggers must be unique and canonically ordered")
        if not self.ranking_terms or len(self.ranking_terms) != len(set(self.ranking_terms)):
            raise ValueError("ranking terms must be nonempty and unique")
        if any(term not in ActiveLearningTermKind for term in self.ranking_terms):
            raise ValueError("ranking terms must be active-learning term kinds")
        if self.model_revisions != tuple(
            sorted(self.model_revisions, key=lambda item: (item.model_name, item.model_version))
        ):
            raise ValueError("model revisions must be in canonical order")
        names = tuple(item.model_name for item in self.model_revisions)
        if len(names) != len(set(names)):
            raise ValueError("model revision names must be unique")
        expected_digest = semantic_sha256(active_learning_policy_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("selection policy semantic_sha256 does not match its inputs")
        expected_key = f"{ACTIVE_LEARNING_POLICY_KEY_NAMESPACE}:{expected_digest}"
        if self.logical_key != expected_key:
            raise ValueError("selection policy logical_key does not match semantic_sha256")
        return self


def active_learning_policy_projection(policy: ActiveLearningSelectionPolicy) -> dict[str, object]:
    """Return the full semantic projection for a selector policy."""

    if not isinstance(policy, ActiveLearningSelectionPolicy):
        raise TypeError("policy must be an ActiveLearningSelectionPolicy")
    return _policy_projection_values(
        policy_version=policy.policy_version,
        eligible_triggers=policy.eligible_triggers,
        ranking_terms=policy.ranking_terms,
        model_revisions=policy.model_revisions,
    )


class ActiveLearningCandidateDecision(StrictModel):
    """Selection result and reason for one candidate; it embeds all source terms."""

    candidate: ActiveLearningCandidate
    disposition: ActiveLearningCandidateDisposition
    rank: PositiveInteger | None = None
    reason_codes: tuple[ReasonCode, ...]

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("selection reason_codes must be nonempty, unique, and sorted")
        return value

    @model_validator(mode="after")
    def validate_rank_shape(self) -> Self:
        ranked = {
            ActiveLearningCandidateDisposition.SELECTED,
            ActiveLearningCandidateDisposition.BUDGET_EXHAUSTED,
        }
        if (self.disposition in ranked) != (self.rank is not None):
            raise ValueError("only ranked candidates may carry a rank")
        return self


def _rank_value(item: ActiveLearningCandidateDecision) -> int:
    if item.rank is None:
        raise ValueError("ranked candidate requires a rank")
    return item.rank


def _decision_projection_values(
    *,
    pool: ActiveLearningPoolSnapshot,
    policy: ActiveLearningSelectionPolicy,
    budget: int,
    candidate_decisions: tuple[ActiveLearningCandidateDecision, ...],
    selected_review_task_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "semantic_projection_version": ACTIVE_LEARNING_DECISION_PROJECTION_VERSION,
        "pool_semantic_sha256": pool.semantic_sha256,
        "pool_logical_key": pool.logical_key,
        "policy_semantic_sha256": policy.semantic_sha256,
        "policy_logical_key": policy.logical_key,
        "budget": budget,
        "candidate_decisions": [
            {
                "review_task_id": item.candidate.review_task_id,
                "candidate_semantic_sha256": item.candidate.semantic_sha256,
                "disposition": item.disposition.value,
                "rank": item.rank,
                "reason_codes": list(item.reason_codes),
            }
            for item in candidate_decisions
        ],
        "selected_review_task_ids": list(selected_review_task_ids),
    }


class ActiveLearningSelectionDecision(StrictModel):
    """Content-addressed selection result that can be replayed without live inputs."""

    model_version: Literal["active-learning-decision-v1"] = "active-learning-decision-v1"
    pool: ActiveLearningPoolSnapshot
    policy: ActiveLearningSelectionPolicy
    budget: NonNegativeInteger
    candidate_decisions: tuple[ActiveLearningCandidateDecision, ...]
    selected_review_task_ids: tuple[OpaqueUuid, ...]
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey

    @classmethod
    def create(
        cls,
        *,
        pool: ActiveLearningPoolSnapshot,
        policy: ActiveLearningSelectionPolicy,
        budget: int,
        candidate_decisions: Iterable[ActiveLearningCandidateDecision],
    ) -> Self:
        """Seal a deterministic decision after all candidate dispositions are known."""

        canonical_decisions = tuple(
            sorted(tuple(candidate_decisions), key=lambda item: item.candidate.review_task_id)
        )
        selected_ids = tuple(
            item.candidate.review_task_id
            for item in sorted(
                (
                    item
                    for item in canonical_decisions
                    if item.disposition is ActiveLearningCandidateDisposition.SELECTED
                ),
                key=_rank_value,
            )
        )
        digest = semantic_sha256(
            _decision_projection_values(
                pool=pool,
                policy=policy,
                budget=budget,
                candidate_decisions=canonical_decisions,
                selected_review_task_ids=selected_ids,
            )
        )
        return cls(
            pool=pool,
            policy=policy,
            budget=budget,
            candidate_decisions=canonical_decisions,
            selected_review_task_ids=selected_ids,
            semantic_sha256=digest,
            logical_key=f"{ACTIVE_LEARNING_DECISION_KEY_NAMESPACE}:{digest}",
        )

    @model_validator(mode="after")
    def validate_identity_and_coverage(self) -> Self:
        if self.candidate_decisions != tuple(
            sorted(self.candidate_decisions, key=lambda item: item.candidate.review_task_id)
        ):
            raise ValueError("candidate decisions must be ordered by review_task_id")
        pool_by_task_id = {item.review_task_id: item for item in self.pool.candidates}
        item_task_ids = tuple(item.candidate.review_task_id for item in self.candidate_decisions)
        if set(item_task_ids) != set(pool_by_task_id) or len(item_task_ids) != len(pool_by_task_id):
            raise ValueError(
                "selection decision must account for every pool candidate exactly once"
            )
        for item in self.candidate_decisions:
            if pool_by_task_id[item.candidate.review_task_id] != item.candidate:
                raise ValueError("selection candidate must match the frozen pool candidate")
        ranked = tuple(item for item in self.candidate_decisions if item.rank is not None)
        if tuple(sorted(_rank_value(item) for item in ranked)) != tuple(range(1, len(ranked) + 1)):
            raise ValueError("ranked candidates must have contiguous ranks starting at one")
        selected = tuple(
            item.candidate.review_task_id
            for item in sorted(
                (
                    item
                    for item in self.candidate_decisions
                    if item.disposition is ActiveLearningCandidateDisposition.SELECTED
                ),
                key=_rank_value,
            )
        )
        if self.selected_review_task_ids != selected:
            raise ValueError("selected_review_task_ids must match selected rank order")
        if len(selected) > self.budget:
            raise ValueError("selection cannot exceed its budget")
        expected_digest = semantic_sha256(active_learning_decision_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("selection decision semantic_sha256 does not match frozen inputs")
        expected_key = f"{ACTIVE_LEARNING_DECISION_KEY_NAMESPACE}:{expected_digest}"
        if self.logical_key != expected_key:
            raise ValueError("selection decision logical_key does not match semantic_sha256")
        return self


def active_learning_decision_projection(
    decision: ActiveLearningSelectionDecision,
) -> dict[str, object]:
    """Return the semantic projection of an immutable selection decision."""

    if not isinstance(decision, ActiveLearningSelectionDecision):
        raise TypeError("decision must be an ActiveLearningSelectionDecision")
    return _decision_projection_values(
        pool=decision.pool,
        policy=decision.policy,
        budget=decision.budget,
        candidate_decisions=decision.candidate_decisions,
        selected_review_task_ids=decision.selected_review_task_ids,
    )


class ActiveLearningSelector:
    """Pure deterministic selector; storage and delivery are intentionally external."""

    def select(
        self,
        *,
        pool: ActiveLearningPoolSnapshot,
        policy: ActiveLearningSelectionPolicy,
        budget: int,
    ) -> ActiveLearningSelectionDecision:
        """Return an exhaustive, replayable disposition for every frozen candidate."""

        if not isinstance(pool, ActiveLearningPoolSnapshot):
            raise TypeError("pool must be an ActiveLearningPoolSnapshot")
        if not isinstance(policy, ActiveLearningSelectionPolicy):
            raise TypeError("policy must be an ActiveLearningSelectionPolicy")
        if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= INT64_MAX:
            raise ValueError("budget must be a nonnegative signed int64 integer")

        decisions_by_task_id: dict[str, ActiveLearningCandidateDecision] = {}
        rankable: list[ActiveLearningCandidate] = []
        for candidate in pool.candidates:
            disposition, reasons = _candidate_eligibility(candidate, policy)
            if disposition is None:
                rankable.append(candidate)
                continue
            decisions_by_task_id[candidate.review_task_id] = ActiveLearningCandidateDecision(
                candidate=candidate,
                disposition=disposition,
                reason_codes=reasons,
            )

        ordered = tuple(sorted(rankable, key=lambda item: _ranking_key(item, policy)))
        for rank, candidate in enumerate(ordered, start=1):
            selected = rank <= budget
            if selected:
                disposition = ActiveLearningCandidateDisposition.SELECTED
                reasons = ("SELECTED_WITHIN_BUDGET",)
            else:
                disposition = ActiveLearningCandidateDisposition.BUDGET_EXHAUSTED
                budget_reason = "BUDGET_ZERO" if budget == 0 else "BUDGET_EXHAUSTED"
                reasons = _canonical_reason_codes(("BUDGET_EXHAUSTED", budget_reason))
            decisions_by_task_id[candidate.review_task_id] = ActiveLearningCandidateDecision(
                candidate=candidate,
                disposition=disposition,
                rank=rank,
                reason_codes=reasons,
            )

        return ActiveLearningSelectionDecision.create(
            pool=pool,
            policy=policy,
            budget=budget,
            candidate_decisions=tuple(decisions_by_task_id.values()),
        )


def _candidate_eligibility(
    candidate: ActiveLearningCandidate,
    policy: ActiveLearningSelectionPolicy,
) -> tuple[ActiveLearningCandidateDisposition | None, tuple[str, ...]]:
    if candidate.priority_evidence.trigger not in policy.eligible_triggers:
        return (
            ActiveLearningCandidateDisposition.INELIGIBLE_TRIGGER,
            _canonical_reason_codes(
                ("TRIGGER_NOT_ELIGIBLE", f"TRIGGER_{candidate.priority_evidence.trigger.value}")
            ),
        )
    terms = {item.kind: item for item in candidate.terms}
    missing = tuple(
        term
        for term in policy.ranking_terms
        if terms[term].applicability is ActiveLearningTermApplicability.MISSING
    )
    if missing:
        return (
            ActiveLearningCandidateDisposition.MISSING_REQUIRED_TERM,
            _canonical_reason_codes(f"MISSING_REQUIRED_{item.value}" for item in missing),
        )
    not_applicable = tuple(
        term
        for term in policy.ranking_terms
        if terms[term].applicability is ActiveLearningTermApplicability.NOT_APPLICABLE
    )
    if not_applicable:
        return (
            ActiveLearningCandidateDisposition.NOT_APPLICABLE_REQUIRED_TERM,
            _canonical_reason_codes(
                f"NOT_APPLICABLE_REQUIRED_{item.value}" for item in not_applicable
            ),
        )
    return None, ()


def _ranking_key(
    candidate: ActiveLearningCandidate,
    policy: ActiveLearningSelectionPolicy,
) -> tuple[object, ...]:
    """Existing priority wins first; terms and task ID only resolve later layers."""

    terms = {item.kind: item for item in candidate.terms}
    term_values = tuple(terms[term].value_millionths for term in policy.ranking_terms)
    if any(value is None for value in term_values):
        raise ValueError("eligible candidate has no ranking value")
    term_layers = tuple(-value for value in term_values if value is not None)
    return (
        candidate.priority_evidence.priority,
        *term_layers,
        candidate.review_task_id,
    )


def verify_active_learning_selection_decision(
    decision: ActiveLearningSelectionDecision,
) -> ActiveLearningSelectionDecision:
    """Recompute a decision exclusively from its frozen pool, policy, and budget."""

    if not isinstance(decision, ActiveLearningSelectionDecision):
        raise TypeError("decision must be an ActiveLearningSelectionDecision")
    checked = ActiveLearningSelectionDecision.model_validate(
        decision.model_dump(mode="python"),
        strict=True,
    )
    expected = ActiveLearningSelector().select(
        pool=checked.pool,
        policy=checked.policy,
        budget=checked.budget,
    )
    if checked != expected:
        raise ValueError("selection decision does not reproduce from its frozen inputs")
    return checked


__all__ = [
    "ACTIVE_LEARNING_CANDIDATE_KEY_NAMESPACE",
    "ACTIVE_LEARNING_CANDIDATE_PROJECTION_VERSION",
    "ACTIVE_LEARNING_DECISION_KEY_NAMESPACE",
    "ACTIVE_LEARNING_DECISION_PROJECTION_VERSION",
    "ACTIVE_LEARNING_POLICY_KEY_NAMESPACE",
    "ACTIVE_LEARNING_POLICY_PROJECTION_VERSION",
    "ACTIVE_LEARNING_POOL_KEY_NAMESPACE",
    "ACTIVE_LEARNING_POOL_PROJECTION_VERSION",
    "ActiveLearningCandidate",
    "ActiveLearningCandidateDecision",
    "ActiveLearningCandidateDisposition",
    "ActiveLearningModelRevision",
    "ActiveLearningPoolSnapshot",
    "ActiveLearningSelectionDecision",
    "ActiveLearningSelectionPolicy",
    "ActiveLearningSelector",
    "ActiveLearningSourceReference",
    "ActiveLearningTermApplicability",
    "ActiveLearningTermEvidence",
    "ActiveLearningTermKind",
    "ExistingReviewPriorityEvidence",
    "active_learning_candidate_projection",
    "active_learning_decision_projection",
    "active_learning_policy_projection",
    "active_learning_pool_projection",
    "verify_active_learning_selection_decision",
]
