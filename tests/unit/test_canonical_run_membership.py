from pathlib import Path

import pytest
from pydantic import ValidationError

from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.application.canonical_run_membership import (
    CanonicalProcessingRunContext,
    CanonicalProcessingRunDeadlineStatus,
    CanonicalProcessingRunMode,
    CanonicalProcessingRunPrimaryStatus,
    CanonicalProcessingRunShadowStatus,
    CanonicalRunMembershipError,
    CanonicalRunMembershipErrorCode,
    CanonicalRunMembershipJournal,
    canonical_first_work_item_id,
)
from robata.contracts.logical_nodes import RunNodeDisposition, logical_node_from_semantic_digest

STARTED = "2026-07-20T12:00:00Z"


def _uuid(value: int) -> str:
    return f"00000000-0000-4000-8000-{value:012x}"


def _node(value: str):  # type: ignore[no-untyped-def]
    return logical_node_from_semantic_digest(
        node_type="FUSION_REDUCTION",
        key_namespace="canonical-fusion-reduction:v1",
        semantic_sha256=value * 64,
        identity_policy_version="canonical-fusion-reduction-v1",
    )


def _context(run_id: str = _uuid(1)) -> CanonicalProcessingRunContext:
    return CanonicalProcessingRunContext.fresh(
        run_id=run_id,
        recording_identity="a" * 64,
        mcap_id=_uuid(2),
        pipeline_version="canonical-offline-v1",
        config_sha256="b" * 64,
        started_at=STARTED,
    )


def _journal(root: Path, run_id: str = _uuid(1)) -> CanonicalRunMembershipJournal:
    return CanonicalRunMembershipJournal(
        context=_context(run_id), registry=LocalLogicalNodeRegistry(root)
    )


def test_context_record_and_explicit_resume_preserve_run_facts() -> None:
    context = _context()

    assert context.mode is CanonicalProcessingRunMode.FRESH
    assert context.primary_status is CanonicalProcessingRunPrimaryStatus.RUNNING
    assert context.shadow_status is CanonicalProcessingRunShadowStatus.NOT_SCHEDULED
    assert context.deadline_status is CanonicalProcessingRunDeadlineStatus.UNRESOLVED
    assert context.deadline_at is context.completed_at is None
    assert "schema_version" not in type(context).model_fields

    record = context.to_record()
    assert (
        record.run_id,
        record.recording_identity,
        record.mcap_id,
        record.pipeline_version,
        record.config_sha256,
        record.started_at,
    ) == (
        context.run_id,
        context.recording_identity,
        context.mcap_id,
        context.pipeline_version,
        context.config_sha256,
        context.started_at,
    )
    resumed = CanonicalProcessingRunContext.resume(record)
    assert resumed.mode is CanonicalProcessingRunMode.RESUME
    assert resumed.run_id == context.run_id

    terminal = record.complete(
        CanonicalProcessingRunPrimaryStatus.SUCCEEDED, "2026-07-20T12:01:00Z"
    )
    with pytest.raises(ValueError, match="only a RUNNING"):
        CanonicalProcessingRunContext.resume(terminal)
    assert (
        terminal.complete(
            CanonicalProcessingRunPrimaryStatus.SUCCEEDED,
            "2026-07-20T12:01:00Z",
        )
        is terminal
    )


def test_context_is_closed_and_strict() -> None:
    values = _context().model_dump(mode="python")
    with pytest.raises(ValidationError):
        CanonicalProcessingRunContext.model_validate({**values, "extra": True}, strict=True)
    with pytest.raises(ValidationError):
        CanonicalProcessingRunContext.model_validate({**values, "run_id": 1}, strict=True)


def test_attach_derives_work_id_preserves_order_and_is_idempotent(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    second = _node("2")
    first = _node("1")

    second_membership = journal.attach(second, "FUSION_OUTPUT", STARTED)
    first_membership = journal.attach(first, "FUSION_INPUT", "2026-07-20T12:00:01Z")
    retried = journal.attach(second, "FUSION_OUTPUT", STARTED)

    assert journal.memberships == (second_membership, first_membership)
    assert retried == second_membership
    assert second_membership.first_work_item_id == canonical_first_work_item_id(
        run_id=journal.context.run_id, node=second, role="FUSION_OUTPUT"
    )
    assert first_membership.first_work_item_id == canonical_first_work_item_id(
        run_id=journal.context.run_id, node=first, role="FUSION_INPUT"
    )


def test_fresh_replay_reuses_node_with_distinct_execution_work_id(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path)
    node = _node("3")
    first = CanonicalRunMembershipJournal(context=_context(_uuid(10)), registry=registry)
    replay = CanonicalRunMembershipJournal(context=_context(_uuid(11)), registry=registry)

    created = first.attach(node, "FUSION_OUTPUT", STARTED)
    reused = replay.attach(node, "FUSION_OUTPUT", STARTED)

    assert created.disposition is RunNodeDisposition.CREATED
    assert reused.disposition is RunNodeDisposition.REUSED
    assert created.node_logical_key == reused.node_logical_key == node.node_logical_key
    assert created.first_work_item_id != reused.first_work_item_id


def test_conflicts_and_invalid_temporal_facts_fail_closed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    node = _node("4")
    journal.attach(node, "FUSION_OUTPUT", STARTED)

    with pytest.raises(CanonicalRunMembershipError) as changed_time:
        journal.attach(node, "FUSION_OUTPUT", "2026-07-20T12:00:01Z")
    assert changed_time.value.code is CanonicalRunMembershipErrorCode.MEMBERSHIP_CONFLICT

    conflicting = node.model_copy(update={"identity_policy_version": "different-v1"})
    with pytest.raises(CanonicalRunMembershipError) as changed_node:
        journal.attach(conflicting, "FUSION_OUTPUT", STARTED)
    assert changed_node.value.code is CanonicalRunMembershipErrorCode.MEMBERSHIP_CONFLICT

    with pytest.raises(CanonicalRunMembershipError) as early:
        journal.attach(_node("5"), "FUSION_OUTPUT", "2026-07-20T11:59:59Z")
    assert early.value.code is CanonicalRunMembershipErrorCode.INVALID_REQUEST


def test_completion_blocks_new_memberships(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.attach(_node("6"), "FUSION_OUTPUT", "2026-07-20T12:00:01Z")
    record = journal.complete(
        CanonicalProcessingRunPrimaryStatus.NO_EVENTS,
    )

    assert record.primary_status is CanonicalProcessingRunPrimaryStatus.NO_EVENTS
    assert record.completed_at == "2026-07-20T12:00:01Z"
    with pytest.raises(CanonicalRunMembershipError) as completed:
        journal.attach(_node("7"), "FUSION_OUTPUT", "2026-07-20T12:00:03Z")
    assert completed.value.code is CanonicalRunMembershipErrorCode.RUN_ALREADY_COMPLETED


def test_completion_time_is_stable_across_exact_same_run_replay(tmp_path: Path) -> None:
    registry = LocalLogicalNodeRegistry(tmp_path)
    first = CanonicalRunMembershipJournal(context=_context(), registry=registry)
    replay = CanonicalRunMembershipJournal(context=_context(), registry=registry)

    first.attach(_node("8"), "FUSION_INPUT", "2026-07-20T12:00:01Z")
    first.attach(_node("9"), "FUSION_OUTPUT", "2026-07-20T12:00:03Z")
    replay.attach(_node("8"), "FUSION_INPUT", "2026-07-20T12:00:01Z")
    replay.attach(_node("9"), "FUSION_OUTPUT", "2026-07-20T12:00:03Z")

    first_record = first.complete(CanonicalProcessingRunPrimaryStatus.SUCCEEDED)
    replay_record = replay.complete(CanonicalProcessingRunPrimaryStatus.SUCCEEDED)

    assert first_record == replay_record
    assert first_record.completed_at == "2026-07-20T12:00:03Z"


def test_completion_without_memberships_uses_started_at(tmp_path: Path) -> None:
    journal = _journal(tmp_path)

    record = journal.complete(CanonicalProcessingRunPrimaryStatus.MATERIALIZATION_FAILED)

    assert record.completed_at == STARTED
