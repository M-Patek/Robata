from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from robata.adapters import SQLitePrimaryCompletionRepository
from robata.application.canonical.models import CanonicalOfflineRunStatus
from robata.application.canonical.primary_completion import (
    CanonicalPrimaryCompletionDetail,
    primary_completion_command_projection,
)
from robata.application.canonical.result_validation import CanonicalOfflineRunResult
from robata.application.canonical_offline import (
    PrimaryCompletionCommand,
    PrimaryCompletionError,
    PrimaryCompletionErrorCode,
    create_primary_completion_command,
    prepare_initial_action_event_publications,
)
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.primary_completion import (
    PrimaryCompletionOutcome,
    PrimaryCompletionRecord,
    PrimaryCompletionTerminalStage,
    primary_completion_record_semantic_projection,
)
from robata.event_pipeline.identity_registry import (
    EventIdentityPolicyRef,
    EventIdentityRegistryService,
    ExactFingerprintEventIdentityResolver,
)
from robata.inference.adapter import VisionInferenceRequest
from robata.inference.enrichment import OrchestratorEnrichedOutput, ProviderObservation
from tests.integration.test_canonical_offline import (
    _action_evidence_claim_bytes,
    _claim_bytes,
    _coarse_claim_bytes,
    _digest,
    _event_proposal_claim_bytes,
    _Harness,
    _harness,
    _processing_run,
    _run,
    _SequenceEventIdAllocator,
)


def _uuid(value: int) -> str:
    return f"00000000-0000-5000-8000-{value:012x}"



def _prepare_command(
    *,
    repository: SQLitePrimaryCompletionRepository,
    harness: _Harness,
    result: CanonicalOfflineRunResult,
) -> PrimaryCompletionCommand:
    prepared = None
    if result.status is CanonicalOfflineRunStatus.SUCCEEDED:
        outputs = tuple(
            item.enriched_output for item in result.part_results if item.enriched_output is not None
        )
        assert outputs
        checked_outputs: tuple[OrchestratorEnrichedOutput, ...] = outputs
        completed_at = result.processing_run.completed_at
        assert completed_at is not None
        service = EventIdentityRegistryService(
            repository=None,
            resolver=ExactFingerprintEventIdentityResolver(
                EventIdentityPolicyRef(
                    version="exact-fingerprint-v1",
                    semantic_sha256=_digest("exact-fingerprint-v1"),
                )
            ),
            allocator=_SequenceEventIdAllocator(),
            output_admission_policy=harness.execution_policy.output_admission_policy,
        )
        prepared = service.prepare_batch(
            snapshot=repository.snapshot(harness.context.recording_identity),
            admitted_context=harness.context,
            hypotheses=result.hypotheses,
            enriched_outputs=checked_outputs,
            decided_at=completed_at,
        )
    publications = prepare_initial_action_event_publications(
        context=harness.context,
        result=result,
        prepared_identities=prepared,
        execution_policy=harness.execution_policy,
    )
    return create_primary_completion_command(
        result=result,
        prepared_identities=prepared,
        action_event_publications=publications,
    )


def _run_case(
    root: Path,
    *,
    run_value: int,
) -> tuple[_Harness, SQLitePrimaryCompletionRepository, PrimaryCompletionCommand]:
    harness = _harness(_claim_bytes, logical_registry_root=root / "logical")
    repository = SQLitePrimaryCompletionRepository(root / "completion.sqlite3")
    processing_run = _processing_run(harness, run_id=_uuid(run_value))
    assert repository.begin_run(processing_run) == processing_run.to_record()
    result = _run(harness, processing_run=processing_run)
    return (
        harness,
        repository,
        _prepare_command(
            repository=repository,
            harness=harness,
            result=result,
        ),
    )


def test_atomic_completion_survives_reopen_and_exact_replay(tmp_path: Path) -> None:
    harness, repository, command = _run_case(tmp_path, run_value=80_001)

    first = repository.commit(command)

    assert first.replayed is False
    assert first.committed.completion.outcome is PrimaryCompletionOutcome.PRIMARY_COMPLETE
    assert first.committed.completion.schema_version == "3.0"
    assert first.committed.completion.schema_ref.version == "3.0.0"
    assert first.committed.completion.terminal_stage is PrimaryCompletionTerminalStage.FINAL_FUSION
    assert first.committed.detail.schema_version == "4.0"
    assert first.committed.detail.schema_ref.version == "4.0.0"
    assert (
        first.committed.detail.semantic_projection_version
        == "canonical-primary-completion-detail-semantic-v4"
    )
    assert first.committed.detail.processing_run.pipeline_version == "canonical-offline-v5"
    assert first.committed.detail.production_eligible is False
    assert first.committed.detail.coarse_qa_result is not None
    assert first.committed.detail.event_proposal_result is not None
    assert first.committed.detail.candidate_reduction_result is not None
    assert first.committed.detail.action_evidence_executions
    assert first.committed.detail.provisional_fusion_result is not None
    assert first.committed.detail.boundary_refinement_executions
    assert first.committed.detail.final_fusion_context is not None
    assert first.committed.detail.dense_qa_executions == ()
    qa_memberships = tuple(
        item
        for item in first.committed.detail.run_memberships
        if item.node_type == "QA_COMPLETION_RESULT" or item.role == "QA_COMPLETION"
    )
    assert len(qa_memberships) == 1
    assert qa_memberships[0].node_logical_key == (
        f"qa-completion-result:{first.committed.detail.qa_completion_result.semantic_sha256}"
    )
    assert first.committed.detail.qa_completion_result.final_aggregate is not None
    assert first.committed.detail.qa_completion_result.production_eligible is False
    coarse_memberships = tuple(
        item
        for item in first.committed.detail.run_memberships
        if item.node_type == "COARSE_QA_RESULT" or item.role == "COARSE_QA"
    )
    assert len(coarse_memberships) == 1
    assert coarse_memberships[0].node_logical_key == (
        "coarse-qa-result:"
        f"{first.committed.detail.qa_completion_result.coarse_result_semantic_sha256}"
    )
    assert first.committed.identity_result is not None
    assert first.committed.identity_result.final_generation == 1
    assert first.committed.identity_result.fence == 2
    assert len(first.committed.action_event_publications.publications) == 1
    assert len(first.committed.outbox) == 1

    reopened = SQLitePrimaryCompletionRepository(repository.path)
    assert reopened.get(command.detail.run_id) == first.committed
    replay = reopened.commit(command)

    assert replay.replayed is True
    assert replay.committed == first.committed
    assert reopened.list_outbox(harness.context.recording_identity) == first.committed.outbox
    snapshot = reopened.snapshot(harness.context.recording_identity)
    assert snapshot.generation == 1
    assert snapshot.fence == 2
    assert len(snapshot.identities) == 1
    assert len(snapshot.current_revisions) == 1


def _assert_early_no_event_completion_replays(
    *,
    harness: _Harness,
    repository: SQLitePrimaryCompletionRepository,
    run_value: int,
    terminal_stage: PrimaryCompletionTerminalStage,
) -> None:
    processing_run = _processing_run(harness, run_id=_uuid(run_value))
    repository.begin_run(processing_run)
    result = _run(harness, processing_run=processing_run)
    assert result.status is CanonicalOfflineRunStatus.NO_EVENTS

    command = _prepare_command(repository=repository, harness=harness, result=result)
    completion = command.completion
    assert command.detail.status == "NO_EVENTS"
    assert command.detail.input_plan is None
    assert command.detail.part_results == ()
    assert completion.outcome is PrimaryCompletionOutcome.PRIMARY_COMPLETE_NO_EVENTS
    assert completion.terminal_stage is terminal_stage
    assert completion.barrier_member_count == 0
    assert completion.barrier_definition_semantic_sha256 is None
    assert completion.barrier_reduction_semantic_sha256 is None
    assert completion.output_decision_semantic_sha256 is None
    assert completion.output_admission_policy_version is None
    assert completion.output_admission_policy_sha256 is None
    if terminal_stage is PrimaryCompletionTerminalStage.EVENT_PROPOSAL:
        assert result.provisional_fusion_result is None
        assert result.candidate_reduction_result is not None
        assert completion.terminal_evidence_semantic_sha256 == (
            result.candidate_reduction_result.semantic_sha256
        )
    else:
        assert result.provisional_fusion_result is not None
        assert completion.terminal_evidence_semantic_sha256 == (
            result.provisional_fusion_result.semantic_sha256
        )

    first = repository.commit(command)
    assert first.replayed is False
    assert first.committed.identity_result is None
    assert first.committed.outbox == ()
    reopened = SQLitePrimaryCompletionRepository(repository.path)
    assert reopened.get(processing_run.run_id) == first.committed
    replay = reopened.commit(command)
    assert replay.replayed is True
    assert replay.committed == first.committed
    assert reopened.snapshot(harness.context.recording_identity).generation == 0
    assert reopened.list_outbox(harness.context.recording_identity) == ()


def test_event_proposal_no_events_commits_and_replays_atomically(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "logical",
        event_proposal_response_factory=lambda _request: canonical_json_bytes(
            {"claims": [], "abstained": False}
        ),
    )
    repository = SQLitePrimaryCompletionRepository(tmp_path / "completion.sqlite3")

    _assert_early_no_event_completion_replays(
        harness=harness,
        repository=repository,
        run_value=80_016,
        terminal_stage=PrimaryCompletionTerminalStage.EVENT_PROPOSAL,
    )


def test_action_no_events_commits_and_replays_atomically(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "logical",
        action_evidence_response_factory=lambda request: _action_evidence_claim_bytes(
            request,
            observation=ProviderObservation.NO_EVENT,
        ),
    )
    repository = SQLitePrimaryCompletionRepository(tmp_path / "completion.sqlite3")

    _assert_early_no_event_completion_replays(
        harness=harness,
        repository=repository,
        run_value=80_017,
        terminal_stage=PrimaryCompletionTerminalStage.PROVISIONAL_FUSION,
    )


def test_v5_detail_v4_rejects_missing_qa_completion_membership_proof(
    tmp_path: Path,
) -> None:
    _, _, command = _run_case(tmp_path, run_value=80_010)
    fields = command.detail.model_dump(mode="python")
    fields["run_memberships"] = tuple(
        item
        for item in command.detail.run_memberships
        if item.node_type != "QA_COMPLETION_RESULT" and item.role != "QA_COMPLETION"
    )

    with pytest.raises(ValidationError, match="exact ordered lineage prefix"):
        CanonicalPrimaryCompletionDetail.model_validate(fields, strict=True)


def test_v5_detail_v4_rejects_tampered_qa_completion_membership_key(
    tmp_path: Path,
) -> None:
    _, _, command = _run_case(tmp_path, run_value=80_011)
    fields = command.detail.model_dump(mode="python")
    fields["run_memberships"] = tuple(
        item.model_copy(update={"node_logical_key": f"qa-completion-result:{_digest('wrong-qa')}"})
        if item.node_type == "QA_COMPLETION_RESULT"
        else item
        for item in command.detail.run_memberships
    )

    with pytest.raises(ValidationError, match="exact ordered lineage prefix"):
        CanonicalPrimaryCompletionDetail.model_validate(fields, strict=True)


def test_v5_detail_v4_rejects_tampered_coarse_membership_digest(
    tmp_path: Path,
) -> None:
    _, _, command = _run_case(tmp_path, run_value=80_012)
    fields = command.detail.model_dump(mode="python")
    fields["run_memberships"] = tuple(
        item.model_copy(update={"node_logical_key": f"coarse-qa-result:{_digest('wrong-coarse')}"})
        if item.node_type == "COARSE_QA_RESULT"
        else item
        for item in command.detail.run_memberships
    )

    with pytest.raises(ValidationError, match="exact ordered lineage prefix"):
        CanonicalPrimaryCompletionDetail.model_validate(fields, strict=True)


def test_v5_detail_v4_persists_exact_dense_qa_execution_facts(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "logical",
        coarse_response_factory=lambda request: _coarse_claim_bytes(
            request,
            observation=ProviderObservation.DEGRADED,
        ),
    )
    repository = SQLitePrimaryCompletionRepository(tmp_path / "completion.sqlite3")
    processing_run = _processing_run(harness, run_id=_uuid(80_013))
    repository.begin_run(processing_run)
    result = _run(harness, processing_run=processing_run)
    assert result.dense_qa_executions
    assert result.qa_completion_result is not None
    assert result.qa_completion_result.dense_result is not None
    command = _prepare_command(repository=repository, harness=harness, result=result)

    assert command.detail.dense_qa_executions == result.dense_qa_executions
    dense_roles = tuple(
        item.role for item in command.detail.run_memberships if item.role.startswith("DENSE_QA_")
    )
    assert dense_roles[-1] == "DENSE_QA_RESULT"

    committed = repository.commit(command).committed
    reopened = SQLitePrimaryCompletionRepository(repository.path)
    assert reopened.get(command.detail.run_id) == committed
    assert committed.detail.dense_qa_executions == result.dense_qa_executions


def test_v5_detail_v4_rejects_missing_dense_execution_facts(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "logical",
        coarse_response_factory=lambda request: _coarse_claim_bytes(
            request,
            observation=ProviderObservation.DEGRADED,
        ),
    )
    repository = SQLitePrimaryCompletionRepository(tmp_path / "completion.sqlite3")
    processing_run = _processing_run(harness, run_id=_uuid(80_014))
    repository.begin_run(processing_run)
    result = _run(harness, processing_run=processing_run)
    command = _prepare_command(repository=repository, harness=harness, result=result)
    fields = command.detail.model_dump(mode="python")
    fields["dense_qa_executions"] = ()

    with pytest.raises(ValidationError, match="dense QA result does not exactly cover"):
        CanonicalPrimaryCompletionDetail.model_validate(fields, strict=True)


def test_v5_detail_v4_rejects_missing_dense_result_membership(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "logical",
        coarse_response_factory=lambda request: _coarse_claim_bytes(
            request,
            observation=ProviderObservation.DEGRADED,
        ),
    )
    repository = SQLitePrimaryCompletionRepository(tmp_path / "completion.sqlite3")
    processing_run = _processing_run(harness, run_id=_uuid(80_015))
    repository.begin_run(processing_run)
    result = _run(harness, processing_run=processing_run)
    command = _prepare_command(repository=repository, harness=harness, result=result)
    fields = command.detail.model_dump(mode="python")
    fields["run_memberships"] = tuple(
        item for item in command.detail.run_memberships if item.role != "DENSE_QA_RESULT"
    )

    with pytest.raises(ValidationError, match="exact ordered lineage prefix"):
        CanonicalPrimaryCompletionDetail.model_validate(fields, strict=True)


class _CommitThenLoseResponseRepository(SQLitePrimaryCompletionRepository):
    def _commit_connection(self, connection: sqlite3.Connection) -> None:
        super()._commit_connection(connection)
        raise sqlite3.OperationalError("simulated lost commit response")


class _FailAfterStagedFactsRepository(SQLitePrimaryCompletionRepository):
    def _after_staged_facts(
        self,
        connection: sqlite3.Connection,
        command: PrimaryCompletionCommand,
    ) -> None:
        del connection, command
        raise sqlite3.OperationalError("simulated failure after staged aggregate facts")


def test_lost_commit_response_recovers_without_duplicate_outbox(tmp_path: Path) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path / "logical")
    repository = _CommitThenLoseResponseRepository(tmp_path / "completion.sqlite3")
    processing_run = _processing_run(harness, run_id=_uuid(80_002))
    repository.begin_run(processing_run)
    result = _run(harness, processing_run=processing_run)
    command = _prepare_command(repository=repository, harness=harness, result=result)

    recovered = repository.commit(command)

    assert recovered.replayed is True
    assert recovered.committed.command_sha256 == command.command_sha256
    assert len(repository.list_outbox(harness.context.recording_identity)) == 1
    reopened = SQLitePrimaryCompletionRepository(repository.path)
    replay = reopened.commit(command)
    assert replay.replayed is True
    assert replay.committed == recovered.committed
    assert len(reopened.list_outbox(harness.context.recording_identity)) == 1


def test_stale_identity_fence_rolls_back_second_run(tmp_path: Path) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path / "logical")
    repository = SQLitePrimaryCompletionRepository(tmp_path / "completion.sqlite3")
    first_run = _processing_run(harness, run_id=_uuid(80_003))
    second_run = _processing_run(harness, run_id=_uuid(80_004))
    repository.begin_run(first_run)
    repository.begin_run(second_run)
    first_result = _run(harness, processing_run=first_run)
    second_result = _run(harness, processing_run=second_run)
    first_command = _prepare_command(
        repository=repository,
        harness=harness,
        result=first_result,
    )
    second_command = _prepare_command(
        repository=repository,
        harness=harness,
        result=second_result,
    )

    repository.commit(first_command)
    with pytest.raises(PrimaryCompletionError) as caught:
        repository.commit(second_command)

    assert caught.value.code is PrimaryCompletionErrorCode.STALE_IDENTITY
    assert repository.get(second_run.run_id) is None
    assert repository.begin_run(second_run).primary_status.value == "RUNNING"
    assert len(repository.list_outbox(harness.context.recording_identity)) == 1
    snapshot = repository.snapshot(harness.context.recording_identity)
    assert snapshot.generation == 1
    assert snapshot.fence == 2


def test_repository_revalidates_model_copy_before_any_write(tmp_path: Path) -> None:
    harness, repository, command = _run_case(tmp_path, run_value=80_005)
    wrong_reference = command.completion.detailed_result.model_copy(
        update={"artifact_id": _uuid(90_005)}
    )
    tampered_completion = command.completion.model_copy(update={"detailed_result": wrong_reference})
    tampered_command = command.model_copy(update={"completion": tampered_completion})

    with pytest.raises(PrimaryCompletionError) as caught:
        repository.commit(tampered_command)

    assert caught.value.code is PrimaryCompletionErrorCode.INVALID_COMMAND
    assert repository.get(command.detail.run_id) is None
    assert (
        repository.begin_run(
            _processing_run(harness, run_id=command.detail.run_id)
        ).primary_status.value
        == "RUNNING"
    )
    assert repository.snapshot(harness.context.recording_identity).generation == 0


def test_repository_rejects_compact_record_that_disagrees_with_detail(tmp_path: Path) -> None:
    _harness_value, repository, command = _run_case(tmp_path, run_value=80_006)
    draft_completion = command.completion.model_copy(
        update={
            "output_admission_policy_version": "different-output-policy-v1",
            "semantic_sha256": "0" * 64,
        }
    )
    fields = draft_completion.model_dump(mode="python")
    fields["semantic_sha256"] = semantic_sha256(
        primary_completion_record_semantic_projection(draft_completion)
    )
    alternate_completion = PrimaryCompletionRecord.model_validate(fields, strict=True)
    draft_command = command.model_copy(
        update={
            "command_sha256": "0" * 64,
            "completion": alternate_completion,
        }
    )
    tampered_command = draft_command.model_copy(
        update={
            "command_sha256": semantic_sha256(primary_completion_command_projection(draft_command))
        }
    )

    with pytest.raises(PrimaryCompletionError) as caught:
        repository.commit(tampered_command)

    assert caught.value.code is PrimaryCompletionErrorCode.INVALID_COMMAND
    assert repository.get(command.detail.run_id) is None


def test_failure_after_staging_rolls_back_every_aggregate_fact(tmp_path: Path) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path / "logical")
    repository = _FailAfterStagedFactsRepository(tmp_path / "completion.sqlite3")
    processing_run = _processing_run(harness, run_id=_uuid(80_007))
    repository.begin_run(processing_run)
    result = _run(harness, processing_run=processing_run)
    command = _prepare_command(repository=repository, harness=harness, result=result)

    with pytest.raises(PrimaryCompletionError) as caught:
        repository.commit(command)

    assert caught.value.code is PrimaryCompletionErrorCode.TRANSACTION_FAILED
    reopened = SQLitePrimaryCompletionRepository(repository.path)
    assert reopened.get(processing_run.run_id) is None
    assert reopened.begin_run(processing_run).primary_status.value == "RUNNING"
    snapshot = reopened.snapshot(harness.context.recording_identity)
    assert snapshot.generation == 0
    assert snapshot.fence == 1
    assert snapshot.identities == ()
    assert snapshot.assignments == ()
    assert snapshot.current_revisions == ()
    assert reopened.list_outbox(harness.context.recording_identity) == ()
    with sqlite3.connect(repository.path) as connection:
        assert connection.execute("SELECT count(*) FROM detailed_results").fetchone()[0] == 0
        assert (
            connection.execute("SELECT count(*) FROM action_event_publications").fetchone()[0] == 0
        )
        assert connection.execute("SELECT count(*) FROM primary_completions").fetchone()[0] == 0


def test_new_run_reuses_identity_and_publication_without_duplicate_outbox(
    tmp_path: Path,
) -> None:
    harness, repository, first_command = _run_case(tmp_path, run_value=80_008)
    first = repository.commit(first_command)
    second_run = _processing_run(harness, run_id=_uuid(80_009))
    repository.begin_run(second_run)
    second_result = _run(harness, processing_run=second_run)
    second_command = _prepare_command(
        repository=repository,
        harness=harness,
        result=second_result,
    )
    assert second_command.detail.prepared_identities is not None
    assert second_command.detail.prepared_identities.mutation is None

    second = repository.commit(second_command)

    assert second.replayed is False
    assert second.committed.identity_result is not None
    assert second.committed.identity_result.final_generation == 1
    assert second.committed.outbox == ()
    reopened = SQLitePrimaryCompletionRepository(repository.path)
    assert reopened.get(first_command.detail.run_id) == first.committed
    assert reopened.get(second_command.detail.run_id) == second.committed
    assert len(reopened.list_outbox(harness.context.recording_identity)) == 1


def test_multi_event_outbox_order_survives_reopen(tmp_path: Path) -> None:
    action_labels: dict[str, str] = {}

    def action_response(request: VisionInferenceRequest) -> bytes:
        assert request.input_plan is not None
        plan_sha256 = request.input_plan.semantic_sha256
        label = action_labels.get(plan_sha256)
        if label is None:
            labels = ("grasp", "place")
            assert len(action_labels) < len(labels)
            label = labels[len(action_labels)]
            action_labels[plan_sha256] = label
        return _action_evidence_claim_bytes(request, label=label)

    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "logical",
        event_proposal_response_factory=lambda request: _event_proposal_claim_bytes(
            request,
            intervals=((100_000_000, 400_000_000), (400_000_000, 900_000_000)),
            labels=("grasp", "place"),
        ),
        action_evidence_response_factory=action_response,
    )
    repository = SQLitePrimaryCompletionRepository(tmp_path / "completion.sqlite3")
    processing_run = _processing_run(harness, run_id=_uuid(80_010))
    repository.begin_run(processing_run)
    result = _run(harness, processing_run=processing_run)
    assert set(action_labels.values()) == {"grasp", "place"}
    command = _prepare_command(repository=repository, harness=harness, result=result)
    assert command.detail.prepared_identities is not None
    assert command.detail.prepared_identities.mutation is not None
    assert len(command.detail.prepared_identities.mutation.outbox) == 2

    committed = repository.commit(command).committed
    reopened = SQLitePrimaryCompletionRepository(repository.path)

    assert reopened.get(command.detail.run_id) == committed
    assert reopened.list_outbox(harness.context.recording_identity) == committed.outbox
