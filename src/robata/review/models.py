"""Governed contracts for asynchronous human review."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, field_validator, model_validator

from robata.contracts.common import INT64_MAX, Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, NodeType, OpaqueUuid
from robata.contracts.revisions import ImmutableNodeRevision, SelectionDecision
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry, default_schema_registry

REVIEW_ROUTING_PROJECTION_VERSION: Final = "review-routing-policy-internal-v1"
REVIEW_TASK_PROJECTION_VERSION: Final = "review-task-internal-v1"
REVIEW_ANNOTATION_PROJECTION_VERSION: Final = "review-annotation-internal-v1"
REVIEW_REOPEN_PROJECTION_VERSION: Final = "review-reopen-internal-v1"

REVIEW_TASK_SCHEMA_ID: Final = "https://schemas.robata.dev/review-task"
REVIEW_TASK_SCHEMA_VERSION: Final = "1.0.0"
REVIEW_TASK_WIRE_VERSION: Literal["1.0"] = "1.0"

REVIEW_ANNOTATION_SCHEMA_ID: Final = "https://schemas.robata.dev/review-annotation"
REVIEW_ANNOTATION_SCHEMA_VERSION: Final = "1.0.0"
REVIEW_ANNOTATION_WIRE_VERSION: Literal["1.0"] = "1.0"

REVIEW_REOPEN_COMMAND_SCHEMA_ID: Final = "https://schemas.robata.dev/review-reopen-command"
REVIEW_REOPEN_COMMAND_SCHEMA_VERSION: Final = "1.0.0"
REVIEW_REOPEN_COMMAND_WIRE_VERSION: Literal["1.0"] = "1.0"

ReviewCode = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$",
    ),
]
ReviewActorId = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
ReviewComment = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=8192)]
ReviewPriority = Annotated[int, Field(strict=True, ge=0, le=INT64_MAX)]
PositiveNanoseconds = Annotated[int, Field(strict=True, ge=1, le=INT64_MAX)]
NonNegativeCounter = Annotated[int, Field(strict=True, ge=0, le=INT64_MAX)]
PositiveFence = Annotated[int, Field(strict=True, ge=1, le=INT64_MAX)]


def _resolve_review_schema_ref(
    schema_ref: SchemaRef | None,
    *,
    schema_id: str,
    schema_version: str,
) -> SchemaRef:
    if schema_ref is not None:
        return schema_ref
    return default_schema_registry().resolve_version(schema_id, schema_version).ref


def _require_review_schema_ref(
    schema_ref: SchemaRef,
    *,
    schema_id: str,
    schema_version: str,
) -> None:
    if schema_ref.schema_id != schema_id or schema_ref.version != schema_version:
        raise ValueError(f"schema_ref must identify {schema_id}@{schema_version}")


class ReviewTrigger(StrEnum):
    """The review triggers explicitly named by Architecture V1 section 25.8."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    DISAGREEMENT = "DISAGREEMENT"
    IDENTITY_AMBIGUITY = "IDENTITY_AMBIGUITY"
    QA_DEGRADATION = "QA_DEGRADATION"
    REVIEW_SAMPLING = "REVIEW_SAMPLING"


class ReviewTaskStatus(StrEnum):
    """Local review queue lifecycle."""

    PENDING = "PENDING"
    LEASED = "LEASED"
    COMPLETED = "COMPLETED"


class ReviewSubject(StrictModel):
    """Immutable logical subject and optional selected revision under review."""

    subject_type: NodeType
    subject_id: NodeLogicalKey
    recording_identity: Sha256Digest
    selected_revision_logical_key: NodeLogicalKey | None = None


class ReviewRoutingRule(StrictModel):
    """Priority and SLA for one governed nonblocking trigger."""

    trigger: ReviewTrigger
    priority: ReviewPriority
    sla_ns: PositiveNanoseconds


def _routing_policy_projection(
    *,
    policy_version: SchemaVersion,
    rules: tuple[ReviewRoutingRule, ...],
) -> dict[str, object]:
    return {
        "semantic_projection_version": REVIEW_ROUTING_PROJECTION_VERSION,
        "policy_version": policy_version,
        "mode": "NONBLOCKING",
        "rules": [rule.model_dump(mode="json") for rule in rules],
    }


class NonBlockingReviewRoutingPolicy(StrictModel):
    """Versioned routing policy that cannot authorize blocking review."""

    model_version: Literal["review-routing-policy-internal-v1"] = (
        "review-routing-policy-internal-v1"
    )
    policy_version: SchemaVersion
    semantic_sha256: Sha256Digest
    rules: tuple[ReviewRoutingRule, ...]

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        expected_order = tuple(sorted(self.rules, key=lambda item: item.trigger.value))
        if self.rules != expected_order:
            raise ValueError("review routing rules must be ordered by trigger")
        if len({rule.trigger for rule in self.rules}) != len(self.rules):
            raise ValueError("review routing policy must define each trigger at most once")
        expected_digest = semantic_sha256(
            _routing_policy_projection(policy_version=self.policy_version, rules=self.rules)
        )
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match review routing policy")
        return self

    def rule_for(self, trigger: ReviewTrigger) -> ReviewRoutingRule | None:
        """Return the exact trigger rule, or None when review is not routed."""

        return next((rule for rule in self.rules if rule.trigger is trigger), None)


def create_nonblocking_review_routing_policy(
    *,
    policy_version: SchemaVersion,
    rules: tuple[ReviewRoutingRule, ...],
) -> NonBlockingReviewRoutingPolicy:
    """Canonicalize rules and bind them to a versioned semantic digest."""

    canonical_rules = tuple(sorted(rules, key=lambda item: item.trigger.value))
    digest = semantic_sha256(
        _routing_policy_projection(policy_version=policy_version, rules=canonical_rules)
    )
    return NonBlockingReviewRoutingPolicy(
        policy_version=policy_version,
        semantic_sha256=digest,
        rules=canonical_rules,
    )


class ReviewRequest(StrictModel):
    """Caller-authored request; no score can implicitly turn it into blocking work."""

    request_id: OpaqueUuid
    subject: ReviewSubject
    trigger: ReviewTrigger
    reason_codes: tuple[ReviewCode, ...]
    requested_at_ns: Nanoseconds

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("review request requires at least one reason code")
        if value != tuple(sorted(set(value))):
            raise ValueError("review reason codes must be unique and sorted")
        return value


def _review_task_projection(
    *,
    request: ReviewRequest,
    rule: ReviewRoutingRule,
    due_at_ns: int,
    policy: NonBlockingReviewRoutingPolicy,
) -> dict[str, object]:
    return {
        "semantic_projection_version": REVIEW_TASK_PROJECTION_VERSION,
        "request_id": request.request_id,
        "subject": request.subject.model_dump(mode="json"),
        "trigger": request.trigger.value,
        "reason_codes": list(request.reason_codes),
        "priority": rule.priority,
        "requested_at_ns": str(request.requested_at_ns),
        "due_at_ns": str(due_at_ns),
        "routing_policy_version": policy.policy_version,
        "routing_policy_sha256": policy.semantic_sha256,
        "blocking": False,
    }


class ReviewTask(StrictModel):
    """Immutable nonblocking work definition persisted by the local review queue."""

    schema_version: Literal["1.0"]
    schema_ref: SchemaRef
    model_version: Literal["review-task-internal-v1"]
    review_task_id: OpaqueUuid
    semantic_sha256: Sha256Digest
    request_id: OpaqueUuid
    subject: ReviewSubject
    trigger: ReviewTrigger
    reason_codes: tuple[ReviewCode, ...]
    priority: ReviewPriority
    requested_at_ns: Nanoseconds
    due_at_ns: Nanoseconds
    routing_policy_version: SchemaVersion
    routing_policy_sha256: Sha256Digest
    blocking: Literal[False]

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("review reason codes must be nonempty, unique, and sorted")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _require_review_schema_ref(
            self.schema_ref,
            schema_id=REVIEW_TASK_SCHEMA_ID,
            schema_version=REVIEW_TASK_SCHEMA_VERSION,
        )
        if self.due_at_ns <= self.requested_at_ns:
            raise ValueError("review due_at_ns must be later than requested_at_ns")
        request = ReviewRequest(
            request_id=self.request_id,
            subject=self.subject,
            trigger=self.trigger,
            reason_codes=self.reason_codes,
            requested_at_ns=self.requested_at_ns,
        )
        rule = ReviewRoutingRule(
            trigger=self.trigger,
            priority=self.priority,
            sla_ns=self.due_at_ns - self.requested_at_ns,
        )
        policy = NonBlockingReviewRoutingPolicy.model_construct(
            policy_version=self.routing_policy_version,
            semantic_sha256=self.routing_policy_sha256,
            rules=(),
        )
        expected_digest = semantic_sha256(
            _review_task_projection(
                request=request,
                rule=rule,
                due_at_ns=self.due_at_ns,
                policy=policy,
            )
        )
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match review task")
        expected_id = str(uuid5(NAMESPACE_URL, f"robata:review-task:{expected_digest}"))
        if self.review_task_id != expected_id:
            raise ValueError("review_task_id does not match review task semantic identity")
        return self


def create_review_task(
    request: ReviewRequest,
    policy: NonBlockingReviewRoutingPolicy,
    *,
    schema_ref: SchemaRef | None = None,
) -> ReviewTask | None:
    """Route one request under a nonblocking policy, returning None when omitted."""

    if not isinstance(request, ReviewRequest):
        raise TypeError("request must be a ReviewRequest")
    if not isinstance(policy, NonBlockingReviewRoutingPolicy):
        raise TypeError("policy must be a NonBlockingReviewRoutingPolicy")
    rule = policy.rule_for(request.trigger)
    if rule is None:
        return None
    due_at_ns = request.requested_at_ns + rule.sla_ns
    if due_at_ns > INT64_MAX:
        raise ValueError("review SLA deadline exceeds signed int64 nanoseconds")
    digest = semantic_sha256(
        _review_task_projection(
            request=request,
            rule=rule,
            due_at_ns=due_at_ns,
            policy=policy,
        )
    )
    return ReviewTask(
        schema_version=REVIEW_TASK_WIRE_VERSION,
        schema_ref=_resolve_review_schema_ref(
            schema_ref,
            schema_id=REVIEW_TASK_SCHEMA_ID,
            schema_version=REVIEW_TASK_SCHEMA_VERSION,
        ),
        model_version="review-task-internal-v1",
        review_task_id=str(uuid5(NAMESPACE_URL, f"robata:review-task:{digest}")),
        semantic_sha256=digest,
        request_id=request.request_id,
        subject=request.subject,
        trigger=request.trigger,
        reason_codes=request.reason_codes,
        priority=rule.priority,
        requested_at_ns=request.requested_at_ns,
        due_at_ns=due_at_ns,
        routing_policy_version=policy.policy_version,
        routing_policy_sha256=policy.semantic_sha256,
        blocking=False,
    )


class ReviewAdjudication(StrictModel):
    """Immutable reviewer decision and optional authored revision transition."""

    decision_code: ReviewCode
    reason_codes: tuple[ReviewCode, ...] = ()
    comment: ReviewComment | None = None
    authored_revision: ImmutableNodeRevision | None = None
    selection_decision: SelectionDecision | None = None

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("adjudication reason codes must be unique and sorted")
        return value

    @model_validator(mode="after")
    def validate_revision_selection(self) -> Self:
        if (self.authored_revision is None) != (self.selection_decision is None):
            raise ValueError(
                "authored_revision and selection_decision must be both absent or both present"
            )
        if self.authored_revision is None or self.selection_decision is None:
            return self
        revision = self.authored_revision
        selection = self.selection_decision
        if (revision.subject_type, revision.subject_id) != (
            selection.subject_type,
            selection.subject_id,
        ):
            raise ValueError("authored revision and selection must have the same subject")
        if selection.selected_revision_id != revision.revision_id:
            raise ValueError("selection must select the authored revision ID")
        if selection.selected_revision_logical_key != revision.revision_logical_key:
            raise ValueError("selection must select the authored revision logical key")
        return self


def _annotation_projection(
    *,
    task: ReviewTask,
    lease_fence: int,
    lease_owner: str,
    reviewer_id: str,
    adjudication: ReviewAdjudication,
    authored_at_ns: int,
) -> dict[str, object]:
    return {
        "semantic_projection_version": REVIEW_ANNOTATION_PROJECTION_VERSION,
        "review_task_id": task.review_task_id,
        "review_task_semantic_sha256": task.semantic_sha256,
        "subject": task.subject.model_dump(mode="json"),
        "lease_fence": lease_fence,
        "lease_owner": lease_owner,
        "reviewer_id": reviewer_id,
        "adjudication": adjudication.model_dump(mode="json"),
        "authored_at_ns": str(authored_at_ns),
    }


class ReviewAnnotation(StrictModel):
    """Append-only annotation bound to one valid leased attempt."""

    schema_version: Literal["1.0"]
    schema_ref: SchemaRef
    model_version: Literal["review-annotation-internal-v1"]
    annotation_id: OpaqueUuid
    semantic_sha256: Sha256Digest
    review_task_id: OpaqueUuid
    review_task_semantic_sha256: Sha256Digest
    subject: ReviewSubject
    lease_fence: PositiveFence
    lease_owner: ReviewActorId
    reviewer_id: ReviewActorId
    adjudication: ReviewAdjudication
    authored_at_ns: Nanoseconds

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _require_review_schema_ref(
            self.schema_ref,
            schema_id=REVIEW_ANNOTATION_SCHEMA_ID,
            schema_version=REVIEW_ANNOTATION_SCHEMA_VERSION,
        )
        revision = self.adjudication.authored_revision
        if revision is not None and (revision.subject_type, revision.subject_id) != (
            self.subject.subject_type,
            self.subject.subject_id,
        ):
            raise ValueError("authored revision must belong to the reviewed subject")
        task = ReviewTask.model_construct(
            review_task_id=self.review_task_id,
            semantic_sha256=self.review_task_semantic_sha256,
            subject=self.subject,
        )
        expected_digest = semantic_sha256(
            _annotation_projection(
                task=task,
                lease_fence=self.lease_fence,
                lease_owner=self.lease_owner,
                reviewer_id=self.reviewer_id,
                adjudication=self.adjudication,
                authored_at_ns=self.authored_at_ns,
            )
        )
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match review annotation")
        expected_id = str(uuid5(NAMESPACE_URL, f"robata:review-annotation:{expected_digest}"))
        if self.annotation_id != expected_id:
            raise ValueError("annotation_id does not match review annotation semantic identity")
        return self


def create_review_annotation(
    *,
    task: ReviewTask,
    lease_fence: int,
    lease_owner: str,
    reviewer_id: str,
    adjudication: ReviewAdjudication,
    authored_at_ns: int,
    schema_ref: SchemaRef | None = None,
) -> ReviewAnnotation:
    """Create an immutable annotation for a claimed review task."""

    projection = _annotation_projection(
        task=task,
        lease_fence=lease_fence,
        lease_owner=lease_owner,
        reviewer_id=reviewer_id,
        adjudication=adjudication,
        authored_at_ns=authored_at_ns,
    )
    digest = semantic_sha256(projection)
    return ReviewAnnotation(
        schema_version=REVIEW_ANNOTATION_WIRE_VERSION,
        schema_ref=_resolve_review_schema_ref(
            schema_ref,
            schema_id=REVIEW_ANNOTATION_SCHEMA_ID,
            schema_version=REVIEW_ANNOTATION_SCHEMA_VERSION,
        ),
        model_version="review-annotation-internal-v1",
        annotation_id=str(uuid5(NAMESPACE_URL, f"robata:review-annotation:{digest}")),
        semantic_sha256=digest,
        review_task_id=task.review_task_id,
        review_task_semantic_sha256=task.semantic_sha256,
        subject=task.subject,
        lease_fence=lease_fence,
        lease_owner=lease_owner,
        reviewer_id=reviewer_id,
        adjudication=adjudication,
        authored_at_ns=authored_at_ns,
    )


def _reopen_projection(
    *,
    review_task_id: str,
    expected_annotation_id: str,
    reason_code: str,
    requested_at_ns: int,
) -> dict[str, object]:
    return {
        "semantic_projection_version": REVIEW_REOPEN_PROJECTION_VERSION,
        "review_task_id": review_task_id,
        "expected_annotation_id": expected_annotation_id,
        "reason_code": reason_code,
        "requested_at_ns": str(requested_at_ns),
    }


class ReviewReopenCommand(StrictModel):
    """Idempotent command to reopen one completed review without deleting history."""

    schema_version: Literal["1.0"]
    schema_ref: SchemaRef
    model_version: Literal["review-reopen-internal-v1"]
    reopen_id: OpaqueUuid
    semantic_sha256: Sha256Digest
    review_task_id: OpaqueUuid
    expected_annotation_id: OpaqueUuid
    reason_code: ReviewCode
    requested_at_ns: Nanoseconds

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        _require_review_schema_ref(
            self.schema_ref,
            schema_id=REVIEW_REOPEN_COMMAND_SCHEMA_ID,
            schema_version=REVIEW_REOPEN_COMMAND_SCHEMA_VERSION,
        )
        expected = semantic_sha256(
            _reopen_projection(
                review_task_id=self.review_task_id,
                expected_annotation_id=self.expected_annotation_id,
                reason_code=self.reason_code,
                requested_at_ns=self.requested_at_ns,
            )
        )
        if self.semantic_sha256 != expected:
            raise ValueError("semantic_sha256 does not match review reopen command")
        return self


def create_review_reopen_command(
    *,
    reopen_id: OpaqueUuid,
    review_task_id: OpaqueUuid,
    expected_annotation_id: OpaqueUuid,
    reason_code: ReviewCode,
    requested_at_ns: int,
    schema_ref: SchemaRef | None = None,
) -> ReviewReopenCommand:
    """Bind a caller-owned command ID to exact reopen semantics."""

    digest = semantic_sha256(
        _reopen_projection(
            review_task_id=review_task_id,
            expected_annotation_id=expected_annotation_id,
            reason_code=reason_code,
            requested_at_ns=requested_at_ns,
        )
    )
    return ReviewReopenCommand(
        schema_version=REVIEW_REOPEN_COMMAND_WIRE_VERSION,
        schema_ref=_resolve_review_schema_ref(
            schema_ref,
            schema_id=REVIEW_REOPEN_COMMAND_SCHEMA_ID,
            schema_version=REVIEW_REOPEN_COMMAND_SCHEMA_VERSION,
        ),
        model_version="review-reopen-internal-v1",
        reopen_id=reopen_id,
        semantic_sha256=digest,
        review_task_id=review_task_id,
        expected_annotation_id=expected_annotation_id,
        reason_code=reason_code,
        requested_at_ns=requested_at_ns,
    )


def validate_registered_review_task(
    task: ReviewTask,
    registry: SchemaRegistry | None = None,
) -> ReviewTask:
    """Strictly revalidate a review task against its exact registered schema."""

    checked = ReviewTask.model_validate(task.model_dump(mode="python"), strict=True)
    active_registry = registry or default_schema_registry()
    active_registry.resolve_exact(checked.schema_ref)
    active_registry.validate_pinned(checked.schema_ref, checked.model_dump(mode="json"))
    return checked


def validate_registered_review_annotation(
    annotation: ReviewAnnotation,
    registry: SchemaRegistry | None = None,
) -> ReviewAnnotation:
    """Strictly revalidate an annotation against its exact registered schema."""

    checked = ReviewAnnotation.model_validate(annotation.model_dump(mode="python"), strict=True)
    active_registry = registry or default_schema_registry()
    active_registry.resolve_exact(checked.schema_ref)
    active_registry.validate_pinned(checked.schema_ref, checked.model_dump(mode="json"))
    return checked


def validate_registered_review_reopen_command(
    command: ReviewReopenCommand,
    registry: SchemaRegistry | None = None,
) -> ReviewReopenCommand:
    """Strictly revalidate a reopen command against its exact registered schema."""

    checked = ReviewReopenCommand.model_validate(command.model_dump(mode="python"), strict=True)
    active_registry = registry or default_schema_registry()
    active_registry.resolve_exact(checked.schema_ref)
    active_registry.validate_pinned(checked.schema_ref, checked.model_dump(mode="json"))
    return checked


class ReviewTaskSnapshot(StrictModel):
    """Independently visible mutable queue state around an immutable task."""

    task: ReviewTask
    status: ReviewTaskStatus
    lease_fence: NonNegativeCounter = 0
    attempt_count: NonNegativeCounter = 0
    lease_owner: ReviewActorId | None = None
    lease_expires_at_ns: Nanoseconds | None = None
    completed_annotation_id: OpaqueUuid | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.attempt_count != self.lease_fence:
            raise ValueError("attempt_count must equal the monotonically increasing lease fence")
        if self.status is ReviewTaskStatus.LEASED:
            if self.lease_owner is None or self.lease_expires_at_ns is None:
                raise ValueError("leased review task requires complete lease metadata")
            if self.lease_fence == 0:
                raise ValueError("leased review task requires a positive fence")
            if self.completed_annotation_id is not None:
                raise ValueError("leased review task cannot have a completed annotation")
        else:
            if self.lease_owner is not None or self.lease_expires_at_ns is not None:
                raise ValueError("only leased review tasks may retain lease metadata")
            completed = self.status is ReviewTaskStatus.COMPLETED
            if completed != (self.completed_annotation_id is not None):
                raise ValueError("completed status and annotation ID must be present together")
        return self

    def is_overdue(self, now_ns: int) -> bool:
        """Return whether this incomplete task is strictly beyond its SLA deadline."""

        if isinstance(now_ns, bool) or not isinstance(now_ns, int):
            raise TypeError("now_ns must be an integer")
        return self.status is not ReviewTaskStatus.COMPLETED and now_ns > self.task.due_at_ns


class ReviewLease(StrictModel):
    """Fenced authority to submit one annotation."""

    task: ReviewTask
    worker_id: ReviewActorId
    lease_fence: PositiveFence
    lease_expires_at_ns: Nanoseconds


__all__ = [
    "REVIEW_ANNOTATION_PROJECTION_VERSION",
    "REVIEW_ANNOTATION_SCHEMA_ID",
    "REVIEW_ANNOTATION_SCHEMA_VERSION",
    "REVIEW_ANNOTATION_WIRE_VERSION",
    "REVIEW_REOPEN_COMMAND_SCHEMA_ID",
    "REVIEW_REOPEN_COMMAND_SCHEMA_VERSION",
    "REVIEW_REOPEN_COMMAND_WIRE_VERSION",
    "REVIEW_REOPEN_PROJECTION_VERSION",
    "REVIEW_ROUTING_PROJECTION_VERSION",
    "REVIEW_TASK_PROJECTION_VERSION",
    "REVIEW_TASK_SCHEMA_ID",
    "REVIEW_TASK_SCHEMA_VERSION",
    "REVIEW_TASK_WIRE_VERSION",
    "NonBlockingReviewRoutingPolicy",
    "PositiveNanoseconds",
    "ReviewActorId",
    "ReviewAdjudication",
    "ReviewAnnotation",
    "ReviewCode",
    "ReviewLease",
    "ReviewPriority",
    "ReviewReopenCommand",
    "ReviewRequest",
    "ReviewRoutingRule",
    "ReviewSubject",
    "ReviewTask",
    "ReviewTaskSnapshot",
    "ReviewTaskStatus",
    "ReviewTrigger",
    "create_nonblocking_review_routing_policy",
    "create_review_annotation",
    "create_review_reopen_command",
    "create_review_task",
    "validate_registered_review_annotation",
    "validate_registered_review_reopen_command",
    "validate_registered_review_task",
]
