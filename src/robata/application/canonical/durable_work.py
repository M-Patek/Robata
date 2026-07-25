"""Durable execution of the canonical primary-publication boundary.

The offline runner is intentionally not presented as a set of independently
schedulable stages. Its real durable boundary is primary publication: event
identity, ActionEvent genesis, run completion, and outbox facts commit together.
This module gives that existing boundary one fenced work item and reconciles the
small cross-database window where primary completion commits before the work
ledger records success.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ValidationError

from robata.application.canonical.primary_completion import (
    CommittedPrimaryCompletion,
    PreparedPrimaryCompletionCommand,
    PrimaryCompletionCommand,
    PrimaryCompletionCommitResult,
    PrimaryCompletionRepository,
)
from robata.contracts.hashing import (
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)
from robata.queue.models import (
    WorkDependency,
    WorkItem,
    WorkItemPlan,
    WorkItemState,
    WorkItemSubjectType,
    WorkLease,
    WorkLeaseClaim,
)
from robata.queue.stage import Stage

CANONICAL_ACTION_PUBLISH_WORK_POLICY_VERSION = "canonical-action-publish-work-v1"
CANONICAL_ACTION_PUBLISH_WORK_LEASE_SECONDS = 60
CANONICAL_ACTION_PUBLISH_WORK_MAX_ATTEMPTS = 3


class CanonicalDurableWorkError(RuntimeError):
    """The publication work ledger cannot truthfully match primary completion."""


class DurableWorkScheduler(Protocol):
    """Scheduler surface needed by the local primary-publication worker."""

    def plan(
        self,
        plan: WorkItemPlan,
        dependencies: Sequence[WorkDependency] = (),
    ) -> WorkItem: ...

    def get(self, work_item_id: str) -> WorkItem: ...

    def reconcile(self, *, now: datetime | None = None) -> int: ...

    def claim(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        work_item_id: str | None = None,
        now: datetime | None = None,
    ) -> WorkLeaseClaim | None: ...

    def start(self, lease: WorkLease, *, now: datetime | None = None) -> WorkItem: ...

    def heartbeat(
        self,
        lease: WorkLease,
        lease_duration_seconds: int,
        *,
        now: datetime | None = None,
    ) -> WorkLease: ...

    def succeed(
        self,
        lease: WorkLease,
        *,
        result_reference: str | None = None,
        result_sha256: str | None = None,
        now: datetime | None = None,
    ) -> WorkItem: ...


def canonical_action_publish_plan_from_command(
    command: PrimaryCompletionCommand,
) -> WorkItemPlan:
    """Derive execution-local work identity without changing domain identities."""

    checked = _strict_model(command, PrimaryCompletionCommand, "command")
    return _action_publish_plan_from_checked(checked)


def canonical_action_publish_plan_from_prepared(
    prepared: PreparedPrimaryCompletionCommand,
) -> WorkItemPlan:
    """Derive work identity from a process-local prepared completion command."""

    if not isinstance(prepared, PreparedPrimaryCompletionCommand):
        raise TypeError("prepared must be PreparedPrimaryCompletionCommand")
    if not prepared.is_canonical_preparation:
        raise CanonicalDurableWorkError("prepared completion lacks canonical provenance")
    return _action_publish_plan_from_checked(prepared.command)


def _action_publish_plan_from_checked(command: PrimaryCompletionCommand) -> WorkItemPlan:
    return _action_publish_plan(
        run_id=command.detail.run_id,
        mcap_id=command.detail.mcap_id,
        command_sha256=command.command_sha256,
        config_sha256=command.detail.execution_policy_sha256,
        created_at=command.detail.processing_run.started_at,
    )


def canonical_action_publish_plan_from_committed(
    committed: CommittedPrimaryCompletion,
) -> WorkItemPlan:
    """Reconstruct the exact plan after completion won the crash race."""

    checked = _strict_model(committed, CommittedPrimaryCompletion, "committed")
    return _action_publish_plan(
        run_id=checked.detail.run_id,
        mcap_id=checked.detail.mcap_id,
        command_sha256=checked.command_sha256,
        config_sha256=checked.detail.execution_policy_sha256,
        created_at=checked.processing_run.started_at,
    )


class CanonicalActionPublishWorkCoordinator:
    """Fence and reconcile the one actual canonical publication work unit."""

    def __init__(
        self,
        *,
        scheduler: DurableWorkScheduler,
        repository: PrimaryCompletionRepository,
        clock: Callable[[], datetime] | None = None,
        lease_duration_seconds: int = CANONICAL_ACTION_PUBLISH_WORK_LEASE_SECONDS,
    ) -> None:
        if isinstance(lease_duration_seconds, bool) or not isinstance(lease_duration_seconds, int):
            raise TypeError("lease_duration_seconds must be an integer")
        if lease_duration_seconds < 1:
            raise ValueError("lease_duration_seconds must be positive")
        self._scheduler = scheduler
        self._repository = repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lease_duration_seconds = lease_duration_seconds

    def commit(
        self,
        command: PrimaryCompletionCommand,
    ) -> PrimaryCompletionCommitResult:
        """Strictly validate and commit an arbitrary publication command."""

        checked = _strict_model(command, PrimaryCompletionCommand, "command")
        return self._commit_checked(
            checked,
            plan=_action_publish_plan_from_checked(checked),
            commit_repository=lambda: self._repository.commit(checked),
        )

    def commit_prepared(
        self,
        prepared: PreparedPrimaryCompletionCommand,
    ) -> PrimaryCompletionCommitResult:
        """Commit a directly adjacent, already-validated local command."""

        plan = canonical_action_publish_plan_from_prepared(prepared)
        return self._commit_checked(
            prepared.command,
            plan=plan,
            commit_repository=lambda: self._repository.commit_prepared(prepared),
        )

    def _commit_checked(
        self,
        command: PrimaryCompletionCommand,
        *,
        plan: WorkItemPlan,
        commit_repository: Callable[[], PrimaryCompletionCommitResult],
    ) -> PrimaryCompletionCommitResult:
        existing = self._repository.get(command.detail.run_id)
        if existing is not None:
            if existing.command_sha256 != command.command_sha256:
                raise CanonicalDurableWorkError(
                    "primary completion exists with a different publication command"
                )
            self.reconcile(existing)
            return PrimaryCompletionCommitResult(committed=existing, replayed=True)

        item = self._scheduler.plan(plan)
        if item.state is WorkItemState.SUCCEEDED:
            raise CanonicalDurableWorkError(
                "publication work succeeded without authoritative primary completion"
            )
        lease = self._claim_ready(plan)
        now = self._now()
        self._scheduler.start(lease, now=now)
        lease = self._scheduler.heartbeat(
            lease,
            self._lease_duration_seconds,
            now=now,
        )

        committed = commit_repository()
        self._succeed(lease, committed.committed)
        return committed

    def reconcile(self, committed: CommittedPrimaryCompletion) -> WorkItem:
        """Make an existing primary completion visible as succeeded durable work."""

        checked = _strict_model(committed, CommittedPrimaryCompletion, "committed")
        plan = canonical_action_publish_plan_from_committed(checked)
        self._scheduler.plan(plan)
        self._scheduler.reconcile(now=self._now())
        item = self._scheduler.get(plan.work_item_id)
        reference, digest = _committed_result_binding(checked)
        if item.state is WorkItemState.SUCCEEDED:
            if item.result_reference != reference or item.result_sha256 != digest:
                raise CanonicalDurableWorkError(
                    "succeeded publication work has a different completion binding"
                )
            return item
        if item.state in {WorkItemState.LEASED, WorkItemState.RUNNING}:
            lease = _active_lease(item)
            if lease.worker_id != _worker_id(plan):
                raise CanonicalDurableWorkError(
                    "committed publication is still leased by another worker"
                )
        elif item.state is WorkItemState.READY:
            lease = self._claim_ready(plan)
        else:
            raise CanonicalDurableWorkError(
                f"committed publication has irreconcilable work state {item.state.value}"
            )

        now = self._now()
        if item.state is WorkItemState.LEASED:
            self._scheduler.start(lease, now=now)
        lease = self._scheduler.heartbeat(
            lease,
            self._lease_duration_seconds,
            now=now,
        )
        return self._succeed(lease, checked)

    def _claim_ready(self, plan: WorkItemPlan) -> WorkLease:
        now = self._now()
        self._scheduler.reconcile(now=now)
        claim = self._scheduler.claim(
            _worker_id(plan),
            self._lease_duration_seconds,
            work_item_id=plan.work_item_id,
            now=now,
        )
        if claim is None:
            item = self._scheduler.get(plan.work_item_id)
            raise CanonicalDurableWorkError(
                f"publication work is not claimable from state {item.state.value}"
            )
        if claim.work_item.work_item_id != plan.work_item_id:
            raise CanonicalDurableWorkError("scheduler claimed unrelated publication work")
        return claim.lease

    def _succeed(
        self,
        lease: WorkLease,
        committed: CommittedPrimaryCompletion,
    ) -> WorkItem:
        reference, digest = _committed_result_binding(committed)
        return self._scheduler.succeed(
            lease,
            result_reference=reference,
            result_sha256=digest,
            now=self._now(),
        )

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise CanonicalDurableWorkError("durable-work clock must return datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalDurableWorkError("durable-work clock must be timezone-aware")
        return value.astimezone(UTC)


def _action_publish_plan(
    *,
    run_id: str,
    mcap_id: str,
    command_sha256: str,
    config_sha256: str,
    created_at: str,
) -> WorkItemPlan:
    projection = {
        "work_policy_version": CANONICAL_ACTION_PUBLISH_WORK_POLICY_VERSION,
        "run_id": run_id,
        "mcap_id": mcap_id,
        "stage": Stage.ACTION_PUBLISH.value,
        "command_sha256": command_sha256,
        "config_sha256": config_sha256,
    }
    digest = semantic_sha256(projection)
    return WorkItemPlan(
        work_item_id=str(
            uuid5(
                NAMESPACE_URL,
                f"robata:canonical-action-publish-work:{digest}",
            )
        ),
        work_logical_key=f"canonical-action-publish-work:{digest}",
        run_id=run_id,
        mcap_id=mcap_id,
        stage=Stage.ACTION_PUBLISH,
        subject_type=WorkItemSubjectType.MCAP,
        subject_id=mcap_id,
        input_digest=command_sha256,
        config_digest=config_sha256,
        priority=100,
        max_attempts=CANONICAL_ACTION_PUBLISH_WORK_MAX_ATTEMPTS,
        trace_id=f"canonical-run:{run_id}",
        created_at=created_at,
    )


def _active_lease(item: WorkItem) -> WorkLease:
    if (
        item.state not in {WorkItemState.LEASED, WorkItemState.RUNNING}
        or item.fencing_token is None
        or item.leased_by is None
        or item.lease_expires_at is None
    ):
        raise CanonicalDurableWorkError("publication work lacks a complete active lease")
    return WorkLease(
        work_item_id=item.work_item_id,
        worker_id=item.leased_by,
        lease_epoch=item.lease_epoch,
        fencing_token=item.fencing_token,
        lease_expires_at=item.lease_expires_at,
    )


def _committed_result_binding(
    committed: CommittedPrimaryCompletion,
) -> tuple[str, str]:
    checked = _strict_model(committed, CommittedPrimaryCompletion, "committed")
    return (
        f"primary-completion:{checked.detail.run_id}",
        exact_bytes_sha256(canonical_json_bytes(checked)),
    )


def _worker_id(plan: WorkItemPlan) -> str:
    return f"canonical-local-action-publish:{plan.work_item_id}"


def _strict_model[T: BaseModel](value: object, model_type: type[T], label: str) -> T:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    try:
        return model_type.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, ValidationError) as error:
        raise ValueError(f"{label} failed strict validation") from error


__all__ = [
    "CANONICAL_ACTION_PUBLISH_WORK_LEASE_SECONDS",
    "CANONICAL_ACTION_PUBLISH_WORK_MAX_ATTEMPTS",
    "CANONICAL_ACTION_PUBLISH_WORK_POLICY_VERSION",
    "CanonicalActionPublishWorkCoordinator",
    "CanonicalDurableWorkError",
    "DurableWorkScheduler",
    "canonical_action_publish_plan_from_command",
    "canonical_action_publish_plan_from_committed",
    "canonical_action_publish_plan_from_prepared",
]
