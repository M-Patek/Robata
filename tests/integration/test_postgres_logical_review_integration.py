"""Optional real-PostgreSQL proof for canonical logical and review authority."""

from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import UUID, uuid4

import pytest

from robata.adapters.postgres_authority import (
    ConnectionFactory,
    PostgresCanonicalAuthority,
    PostgresConnection,
    psycopg_connection_factory,
)
from robata.adapters.postgres_completion_evidence import PostgresPrimaryCompletionRepository
from robata.adapters.postgres_logical_review import (
    PostgresLogicalNodeRegistry,
    PostgresReviewQueue,
)
from robata.adapters.postgres_migrations import PostgresMigrationRunner
from robata.adapters.postgres_run_projection import PostgresCommittedRunProjection
from robata.application.canonical_run_membership import CanonicalProcessingRunContext
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import RunNodeDisposition, logical_node_from_semantic_digest
from robata.contracts.revisions import (
    RevisionEligibility,
    create_immutable_node_revision,
)
from robata.ports.review_queue import ReviewQueueError, ReviewQueueErrorCode
from robata.review.models import (
    ReviewAdjudication,
    ReviewRequest,
    ReviewRoutingRule,
    ReviewSubject,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTrigger,
    create_nonblocking_review_routing_policy,
    create_review_annotation,
    create_review_reopen_command,
    create_review_task,
)
from tests.integration.test_sqlite_primary_completion import _run_case

_TEST_DSN = os.environ.get("ROBATA_TEST_POSTGRES_DSN")
_ATTACHED_AT = "2026-07-20T12:00:00Z"
_PUBLISHED_AT = "2026-07-20T13:00:00Z"
_SELECTED_AT = "2026-07-20T14:00:00Z"


@pytest.fixture(scope="session")
def postgres_factory() -> ConnectionFactory:
    if not _TEST_DSN:
        pytest.skip("ROBATA_TEST_POSTGRES_DSN is required for PostgreSQL integration tests")
    return psycopg_connection_factory(_TEST_DSN, application_name="robata-p24-integration")


@pytest.fixture(scope="session", autouse=True)
def migrated_postgres(postgres_factory: ConnectionFactory) -> None:
    migration_root = Path(__file__).resolve().parents[2] / "db" / "migrations"
    applied = PostgresMigrationRunner(postgres_factory, migration_root).apply()
    assert "0004" in applied.applied_ids + applied.already_applied_ids


@pytest.fixture
def restricted_postgres_factory(
    postgres_factory: ConnectionFactory,
) -> Iterator[ConnectionFactory]:
    """Exercise FORCE RLS through a short-lived non-owner role."""

    role_name = f"robata_p24_rls_{uuid4().hex}"
    administrator = postgres_factory()
    try:
        administrator.execute(f"CREATE ROLE {role_name} NOLOGIN")
        administrator.execute(f"GRANT USAGE ON SCHEMA robata_canonical TO {role_name}")
        administrator.execute(
            "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
            f"IN SCHEMA robata_canonical TO {role_name}"
        )
    finally:
        administrator.close()

    def factory() -> PostgresConnection:
        connection = postgres_factory()
        connection.execute(f"SET ROLE {role_name}")
        return connection

    try:
        yield factory
    finally:
        cleanup = postgres_factory()
        try:
            cleanup.execute(f"DROP OWNED BY {role_name}")
            cleanup.execute(f"DROP ROLE {role_name}")
        finally:
            cleanup.close()


def _authority(factory: ConnectionFactory, tenant_id: str) -> PostgresCanonicalAuthority:
    return PostgresCanonicalAuthority(
        factory,
        tenant_setting="robata.tenant_id",
        tenant_id=tenant_id,
    )


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: str) -> str:
    return semantic_sha256({"integration": value})


def _node() -> object:
    return logical_node_from_semantic_digest(
        node_type="CAMERA_VIDEO_EXPORT",
        key_namespace="camera-video-export:v1",
        semantic_sha256=_digest("node"),
        identity_policy_version="camera-video-export-v1",
    )


def _revision(node: object, number: int, *, supersedes: object | None = None) -> object:
    return create_immutable_node_revision(
        revision_id=_uuid(1_000 + number),
        subject_type=node.node_type,  # type: ignore[union-attr]
        subject_id=node.node_logical_key,  # type: ignore[union-attr]
        revision_key_namespace="camera-video-revision:v1",
        payload_sha256=_digest(f"payload:{number}"),
        lineage_sha256=_digest(f"lineage:{number}"),
        status_at_publication="READY",
        eligibility_at_publication=RevisionEligibility.ELIGIBLE,
        revision_policy_version="camera-video-revision-policy-v1",
        supersedes_revision_id=(
            None if supersedes is None else supersedes.revision_id  # type: ignore[union-attr]
        ),
        supersedes_revision_logical_key=(
            None if supersedes is None else supersedes.revision_logical_key  # type: ignore[union-attr]
        ),
        published_at=_PUBLISHED_AT,
    )


def _review_task() -> ReviewTask:
    policy = create_nonblocking_review_routing_policy(
        policy_version="postgres-review-routing-v1",
        rules=(
            ReviewRoutingRule(
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                priority=2,
                sla_ns=100,
            ),
        ),
    )
    task = create_review_task(
        ReviewRequest(
            request_id=_uuid(5_000),
            subject=ReviewSubject(
                subject_type="EVENT_HYPOTHESIS",
                subject_id="event-hypothesis:" + "a" * 64,
                recording_identity="b" * 64,
            ),
            trigger=ReviewTrigger.LOW_CONFIDENCE,
            reason_codes=(ReviewTrigger.LOW_CONFIDENCE.value,),
            requested_at_ns=1_000,
        ),
        policy,
    )
    assert task is not None
    return task


def test_postgres_logical_revision_review_and_rls_lifecycle(
    restricted_postgres_factory: ConnectionFactory,
) -> None:
    tenant_a = f"p24-a-{uuid4().hex}"
    tenant_b = f"p24-b-{uuid4().hex}"
    authority_a = _authority(restricted_postgres_factory, tenant_a)
    authority_b = _authority(restricted_postgres_factory, tenant_b)
    registry_a = PostgresLogicalNodeRegistry(authority_a)
    registry_b = PostgresLogicalNodeRegistry(authority_b)
    queue_a = PostgresReviewQueue(authority_a)
    queue_b = PostgresReviewQueue(authority_b)
    registry_a.verify_startup()
    queue_a.verify_startup()

    node = _node()
    created = registry_a.attach_run_node(
        node=node,  # type: ignore[arg-type]
        run_id=_uuid(1),
        role="OUTPUT",
        first_work_item_id=_uuid(101),
        attached_at=_ATTACHED_AT,
        existing_node_disposition=RunNodeDisposition.REUSED,
    )
    created_replay = registry_a.attach_run_node(
        node=node,  # type: ignore[arg-type]
        run_id=_uuid(1),
        role="OUTPUT",
        first_work_item_id=_uuid(101),
        attached_at=_ATTACHED_AT,
        existing_node_disposition=RunNodeDisposition.REUSED,
    )
    reused = registry_a.attach_run_node(
        node=node,  # type: ignore[arg-type]
        run_id=_uuid(2),
        role="OUTPUT",
        first_work_item_id=_uuid(102),
        attached_at="2026-07-20T12:01:00Z",
        existing_node_disposition=RunNodeDisposition.REUSED,
    )
    assert created.node_inserted is True
    assert created.membership.disposition is RunNodeDisposition.CREATED
    assert created_replay.node_inserted is False
    assert created_replay.membership_inserted is False
    assert created_replay.membership == created.membership
    assert reused.membership.disposition is RunNodeDisposition.REUSED
    assert registry_a.lookup_node(*node.identity) == node  # type: ignore[union-attr]
    assert (
        registry_a.lookup_membership(  # type: ignore[union-attr]
            _uuid(1),
            node.node_type,
            node.node_logical_key,
            "OUTPUT",
        )
        == created.membership
    )
    assert registry_a.list_run_memberships(_uuid(1)) == (created.membership,)
    assert registry_a.list_node_memberships(*node.identity) == (  # type: ignore[union-attr]
        created.membership,
        reused.membership,
    )
    assert registry_a.verify_node(*node.identity).memberships == (  # type: ignore[union-attr]
        created.membership,
        reused.membership,
    )

    first_revision = _revision(node, 1)
    assert registry_a.publish_revision(first_revision).inserted is True  # type: ignore[arg-type]
    assert registry_a.publish_revision(first_revision).inserted is False  # type: ignore[arg-type]
    first_selection = registry_a.select_revision(
        subject_type=node.node_type,  # type: ignore[union-attr]
        subject_id=node.node_logical_key,  # type: ignore[union-attr]
        selected_revision_id=first_revision.revision_id,  # type: ignore[union-attr]
        selection_decision_id=_uuid(10_001),
        selection_key_namespace="camera-video-selection:v1",
        expected_previous_selection_decision_id=None,
        selection_policy_version="camera-video-selection-policy-v1",
        selected_at=_SELECTED_AT,
    )
    first_selection_replay = registry_a.select_revision(
        subject_type=node.node_type,  # type: ignore[union-attr]
        subject_id=node.node_logical_key,  # type: ignore[union-attr]
        selected_revision_id=first_revision.revision_id,  # type: ignore[union-attr]
        selection_decision_id=_uuid(10_001),
        selection_key_namespace="camera-video-selection:v1",
        expected_previous_selection_decision_id=None,
        selection_policy_version="camera-video-selection-policy-v1",
        selected_at=_SELECTED_AT,
    )
    assert first_selection_replay.decision_inserted is False
    assert first_selection_replay.projection_advanced is False
    assert first_selection_replay.current == first_selection.current
    second_revision = _revision(node, 2, supersedes=first_revision)
    assert registry_a.publish_revision(second_revision).inserted is True  # type: ignore[arg-type]
    second_selection = registry_a.select_revision(
        subject_type=node.node_type,  # type: ignore[union-attr]
        subject_id=node.node_logical_key,  # type: ignore[union-attr]
        selected_revision_id=second_revision.revision_id,  # type: ignore[union-attr]
        selection_decision_id=_uuid(10_002),
        selection_key_namespace="camera-video-selection:v1",
        expected_previous_selection_decision_id=first_selection.decision.selection_decision_id,
        selection_policy_version="camera-video-selection-policy-v1",
        selected_at="2026-07-20T14:01:00Z",
    )
    verified = registry_a.verify_subject(*node.identity)  # type: ignore[union-attr]
    assert verified.revisions == (first_revision, second_revision)
    assert verified.current == second_selection.current
    assert (
        registry_a.lookup_revision(  # type: ignore[union-attr]
            node.node_type,
            node.node_logical_key,
            first_revision.revision_id,
        )
        == first_revision
    )
    assert (
        registry_a.lookup_selection_decision(  # type: ignore[union-attr]
            node.node_type,
            node.node_logical_key,
            first_selection.decision.selection_decision_id,
        )
        == first_selection.decision
    )
    assert registry_a.lookup_current_selection(*node.identity) == second_selection.current  # type: ignore[union-attr]
    assert registry_a.list_revisions(*node.identity) == (  # type: ignore[union-attr]
        first_revision,
        second_revision,
    )
    assert registry_a.list_selection_decisions(*node.identity) == (  # type: ignore[union-attr]
        first_selection.decision,
        second_selection.decision,
    )
    assert registry_a.rebuild_current_projection() == (second_selection.current,)

    task = _review_task()
    assert queue_a.enqueue(task).inserted is True
    assert queue_a.enqueue(task).inserted is False
    lease = queue_a.claim_next(worker_id="review-worker-a", now_ns=1_010, lease_duration_ns=100)
    assert lease is not None
    renewed = queue_a.renew_lease(
        review_task_id=task.review_task_id,
        worker_id=lease.worker_id,
        lease_fence=lease.lease_fence,
        now_ns=1_020,
        lease_duration_ns=200,
    )
    assert renewed.lease_fence == lease.lease_fence
    assert renewed.lease_expires_at_ns == 1_220
    stale_annotation = create_review_annotation(
        task=task,
        lease_fence=renewed.lease_fence + 1,
        lease_owner=renewed.worker_id,
        reviewer_id="reviewer-a",
        adjudication=ReviewAdjudication(decision_code="ACCEPT"),
        authored_at_ns=1_030,
    )
    with pytest.raises(ReviewQueueError) as stale_error:
        queue_a.submit_annotation(stale_annotation, now_ns=1_040)
    assert stale_error.value.code is ReviewQueueErrorCode.STALE_FENCE
    annotation = create_review_annotation(
        task=task,
        lease_fence=renewed.lease_fence,
        lease_owner=renewed.worker_id,
        reviewer_id="reviewer-a",
        adjudication=ReviewAdjudication(decision_code="ACCEPT"),
        authored_at_ns=1_050,
    )
    assert queue_a.submit_annotation(annotation, now_ns=1_060).inserted is True
    assert queue_a.submit_annotation(annotation, now_ns=1_070).inserted is False
    reopen_command = create_review_reopen_command(
        reopen_id=_uuid(9_000),
        review_task_id=task.review_task_id,
        expected_annotation_id=annotation.annotation_id,
        reason_code="NEW_EVIDENCE",
        requested_at_ns=1_080,
    )
    reopened = queue_a.reopen(reopen_command)
    assert reopened.applied is True
    assert reopened.snapshot.status is ReviewTaskStatus.PENDING
    assert queue_a.reopen(reopen_command).applied is False
    replacement_lease = queue_a.claim_next(
        worker_id="review-worker-b",
        now_ns=1_090,
        lease_duration_ns=100,
    )
    assert replacement_lease is not None
    assert replacement_lease.lease_fence == lease.lease_fence + 1
    snapshot = queue_a.get_task(task.review_task_id)
    assert snapshot is not None
    assert snapshot.lease_fence == replacement_lease.lease_fence
    assert tuple(item.task.review_task_id for item in queue_a.list_open(limit=1)) == (
        task.review_task_id,
    )
    assert tuple(item.task.review_task_id for item in queue_a.list_overdue(now_ns=1_200)) == (
        task.review_task_id,
    )
    assert queue_a.list_annotations(task.review_task_id) == (annotation,)

    # The same identity can exist in another tenant, but no fact is visible or
    # mutable across the transaction-local RLS tenant boundary.
    assert registry_b.lookup_node(*node.identity) is None  # type: ignore[union-attr]
    assert queue_b.get_task(task.review_task_id) is None
    other_created = registry_b.attach_run_node(
        node=node,  # type: ignore[arg-type]
        run_id=_uuid(1),
        role="OUTPUT",
        first_work_item_id=_uuid(101),
        attached_at=_ATTACHED_AT,
        existing_node_disposition=RunNodeDisposition.REUSED,
    )
    assert other_created.membership.disposition is RunNodeDisposition.CREATED
    assert registry_a.lookup_current_selection(*node.identity) == second_selection.current  # type: ignore[union-attr]


def test_postgres_run_projection_reads_only_committed_completion_bytes(
    restricted_postgres_factory: ConnectionFactory,
    tmp_path: Path,
) -> None:
    authority = _authority(restricted_postgres_factory, f"p24-read-{uuid4().hex}")
    _, _preparation_repository, command = _run_case(
        tmp_path / "completion-preparation",
        run_value=uuid4().int % 1_000_000_000,
    )
    processing_run = command.detail.processing_run
    completion_repository = PostgresPrimaryCompletionRepository(authority)
    run_context = CanonicalProcessingRunContext.fresh(
        run_id=processing_run.run_id,
        recording_identity=processing_run.recording_identity,
        mcap_id=processing_run.mcap_id,
        pipeline_version=processing_run.pipeline_version,
        config_sha256=processing_run.config_sha256,
        started_at=processing_run.started_at,
    )
    assert completion_repository.begin_run(run_context) == run_context.to_record()
    prepared_identities = command.detail.prepared_identities
    assert prepared_identities is not None
    partition = completion_repository.snapshot(processing_run.recording_identity)
    assert (partition.generation, partition.fence) == (
        prepared_identities.expected_generation,
        prepared_identities.expected_fence,
    )
    committed = completion_repository.commit(command)
    projection = PostgresCommittedRunProjection(authority)

    projection.verify_startup()
    projection.health_check()
    listed = projection.list_runs()
    assert [item.run_id for item in listed.runs] == [processing_run.run_id]
    snapshot = projection.snapshot(processing_run.run_id)
    assert snapshot.cursor == exact_bytes_sha256(canonical_json_bytes(committed.committed))
    assert snapshot.run.run_id == processing_run.run_id
    assert snapshot.run.recording_identity == processing_run.recording_identity


def test_postgres_concurrent_logical_node_creation_converges(
    restricted_postgres_factory: ConnectionFactory,
) -> None:
    authority = _authority(restricted_postgres_factory, f"p24-concurrent-{uuid4().hex}")
    registries = (PostgresLogicalNodeRegistry(authority), PostgresLogicalNodeRegistry(authority))
    node = _node()
    barrier = Barrier(2)

    def attach(index: int):  # type: ignore[no-untyped-def]
        barrier.wait(timeout=10)
        return registries[index].attach_run_node(
            node=node,  # type: ignore[arg-type]
            run_id=_uuid(100 + index),
            role="OUTPUT",
            first_work_item_id=_uuid(200 + index),
            attached_at=f"2026-07-20T12:00:0{index}Z",
            existing_node_disposition=RunNodeDisposition.REUSED,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(attach, range(2)))

    assert sorted(item.membership.disposition.value for item in results) == ["CREATED", "REUSED"]
    assert sum(item.node_inserted is True for item in results) == 1
    assert all(item.membership_inserted is True for item in results)
    verified = registries[0].verify_node(*node.identity)  # type: ignore[union-attr]
    assert {item.run_id for item in verified.memberships} == {_uuid(100), _uuid(101)}
