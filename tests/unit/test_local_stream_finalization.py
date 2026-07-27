from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from robata.adapters.sqlite_outbox import SQLiteIdempotentOutboxSink
from robata.adapters.sqlite_stream_delivery import (
    SQLiteStreamDeliveryAuthority,
    SQLiteStreamDeliveryConflict,
    SQLiteStreamDeliveryError,
)
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler
from robata.application.canonical.bounded_media import (
    BoundedWindowPlan,
    CameraStreamFacts,
    CameraWindowPlan,
    PlannerFinish,
    WindowClosureReason,
    WindowMember,
)
from robata.application.canonical.local_stream_finalization import (
    FinalRecordingFacts,
    LocalConformanceStreamFinalizer,
    LocalStreamFinalizationError,
    LocalStreamFinalizationSchemaRefs,
    load_completed_local_stream_recording_result,
)
from robata.application.canonical.stream_scheduler import (
    DurableStreamWindowScheduler,
    EosSealInputs,
    StreamSchedulerSchemaRefs,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    AuthorityBinding,
    CameraAbsenceReason,
    ChannelBinding,
    StreamPolicyBinding,
    StreamPurpose,
    StreamStage,
    StreamSubjectType,
    TerminalOutcome,
)
from robata.contracts.stream_inference import create_stream_window_result
from robata.contracts.stream_planning import (
    StreamWorkItemPlan,
    create_expected_window_plan,
)
from robata.contracts.stream_source import create_pre_eos_capture_subject
from robata.inference.models import InferenceStatus, ModelInference, ModelInferenceUsage, VisionTask
from robata.queue.outbox import (
    OutboxDeliveryStatus,
    OutboxFenceError,
    OutboxRelay,
    OutboxRetryPolicy,
)
from robata.queue.stream_models import (
    StreamTerminalEvidence,
    StreamWorkItem,
    StreamWorkItemState,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _provider_fixture_model(task: VisionTask = VisionTask.QA_COARSE) -> ModelInference:
    ordinal = {
        VisionTask.QA_COARSE: 1,
        VisionTask.QA_DENSE: 2,
        VisionTask.EVENT_PROPOSAL: 3,
    }[task]
    return ModelInference(
        schema_version='1.0',
        inference_id=str(uuid5(NAMESPACE_URL, f'p5-test-inference:{task.value}')),
        logical_invocation_id=str(
            uuid5(NAMESPACE_URL, f'p5-test-logical-invocation:{task.value}')
        ),
        request_id=str(uuid5(NAMESPACE_URL, f'p5-test-request:{task.value}')),
        idempotency_key=f'p5-test-idempotency-{task.value}',
        mcap_id='00000000-0000-0000-0000-000000000401',
        package_set_id='00000000-0000-0000-0000-000000000402',
        package_id=None,
        package_ids=(f'p5-test-package-{ordinal}',),
        camera_mapping_run_id='00000000-0000-0000-0000-000000000403',
        alignment_id='00000000-0000-0000-0000-000000000404',
        start_ns=0,
        end_ns=1_000_000_000,
        stage=task,
        provider='local-test-provider',
        model_name='local-test-model',
        model_version='1.0',
        adapter_version='local-test-adapter-v1',
        prompt_version='local-test-prompt-v1',
        prompt_artifact_id='local-test-prompt',
        prompt_sha256=f'{500 + ordinal:064x}',
        rendered_input_digest=f'{510 + ordinal:064x}',
        input_plan_id=None,
        input_plan_semantic_sha256=None,
        input_plan_part_ordinal=None,
        input_plan_part_count=None,
        input_plan_part_semantic_sha256=None,
        output_schema_id='local-test-output',
        output_schema_version='1.0',
        output_schema_artifact_id='local-test-output-schema',
        output_schema_sha256=f'{520 + ordinal:064x}',
        capability_snapshot_id='00000000-0000-0000-0000-000000000405',
        capability_snapshot_digest=f'{530 + ordinal:064x}',
        input_manifest_set_sha256=f'{540 + ordinal:064x}',
        input_config={'input_images': 1},
        sampling_config={'policy': 'p5-test-v1'},
        generation_config={'temperature': 0},
        provider_idempotency_key=f'p5-test-provider-idempotency-{task.value}',
        provider_request_id=f'p5-test-provider-request-{task.value}',
        experiment_id=None,
        shadow_route_id=None,
        primary_inference_id=None,
        shadow=False,
        attempt=1,
        retry_count=0,
        status=InferenceStatus.SUCCEEDED,
        queued_at='2026-07-20T00:00:00Z',
        started_at='2026-07-20T00:00:00Z',
        completed_at='2026-07-20T00:00:00Z',
        latency_ms=1,
        raw_output={'fixture': task.value},
        normalized_output={'label': task.value},
        output_valid=True,
        reported_confidence=None,
        calibrated_confidence=None,
        usage=ModelInferenceUsage(input_frames=1, input_images=1),
        failure=None,
        created_at='2026-07-20T00:00:00Z',
    )


def _schema(value: int) -> SchemaRef:
    return SchemaRef(
        schema_id=f"https://schemas.robata.dev/local-stream-test-{value}",
        version="1.0.0",
        artifact_id=_uuid(1000 + value),
        sha256=_digest(2000 + value),
    )


def _recording_result_schema(version: str = "4.0.0") -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/local-stream-recording-result",
        version=version,
        artifact_id=_uuid(1023),
        sha256=_digest(2023),
    )


def _composition(
    tmp_path: Path,
    *,
    close_eos: bool = True,
    window_count: int = 1,
) -> DurableStreamWindowScheduler:
    authority = AuthorityBinding(
        authority_id="local-authority",
        authority_epoch=1,
        policy_version="local-authority-v1",
        initial_binding_semantic_sha256=_digest(10),
    )
    capture = create_pre_eos_capture_subject(
        schema_ref=_schema(1),
        capture_authority_id="local-capture-authority",
        capture_authority_epoch=1,
        capture_assignment_policy_version="local-capture-v1",
        acquisition_id="local-acquisition",
        acquisition_epoch=1,
        channel_bindings=tuple(
            ChannelBinding(
                camera_id=camera_id,
                source_channel_id=f"channel-{camera_id.value}",
                source_channel_epoch=1,
                channel_binding_semantic_sha256=_digest(20 + ordinal),
            )
            for ordinal, camera_id in enumerate(CAMERA_IDS)
        ),
        mapping_authority=authority,
        clock_authority=authority,
    )
    policy = StreamPolicyBinding(version="local-policy-v1", semantic_sha256=_digest(30))
    expected = create_expected_window_plan(
        schema_ref=_schema(2),
        capture_scope_digest=capture.capture_scope_digest,
        segmentation_policy=policy,
        window_policy=policy,
        watermark_policy=policy,
        lateness_policy=policy,
        idle_source_policy=policy,
        planner_version="local-planner-v1",
    )
    execution = SQLiteWorkScheduler(tmp_path / "stream.sqlite3")
    composition = DurableStreamWindowScheduler(
        database_path=execution.database_path,
        execution_scheduler=execution,
        expected_plan=expected,
        source_subject=capture.reference(),
        stream_run_id=_uuid(50),
        schema_refs=StreamSchedulerSchemaRefs(
            incremental_window=_schema(3),
            expected_declaration=_schema(4),
            expected_plan_seal=_schema(5),
            stream_work_plan=_schema(6),
            terminal_member=_schema(7),
            terminal_closure=_schema(8),
        ),
        dag_config_semantic_sha256=_digest(40),
        clock=lambda: _NOW,
    )
    for window_ordinal in range(window_count):
        interval = NanosecondInterval(
            start_ns=window_ordinal * 1_000_000_000,
            end_ns=(window_ordinal + 1) * 1_000_000_000,
        )
        composition.append_window(
            BoundedWindowPlan(
                ordinal=window_ordinal,
                requested_interval=interval,
                effective_interval=interval,
                camera_plans=tuple(
                    CameraWindowPlan(
                        camera_id=camera_id,
                        members=(
                            WindowMember(
                                camera_id=camera_id,
                                interval=interval,
                                absence_reason=CameraAbsenceReason.ABSENT,
                                absence_evidence_sha256=_digest(
                                    60 + window_ordinal * len(CAMERA_IDS) + ordinal
                                ),
                            ),
                        ),
                    )
                    for ordinal, camera_id in enumerate(CAMERA_IDS)
                ),
                quality_targets=(),
                quality_gaps=(),
                watermark_ns=interval.end_ns + 300_000_000,
                closure_reason=WindowClosureReason.WATERMARK,
                capture_scope_digest=capture.capture_scope_digest,
                mapping_semantic_sha256=_digest(70),
                clock_or_alignment_semantic_sha256=_digest(71),
                window_policy_version="local-window-v1",
                quality_policy_version="local-quality-v1",
                purpose=StreamPurpose.QA_COARSE,
            )
        )
    if close_eos:
        _close_eos(composition)
    return composition


def _close_eos(composition: DurableStreamWindowScheduler) -> None:
    composition.seal(
        PlannerFinish(
            closed_segments=(),
            quality_targets=(),
            windows=(),
            facts=tuple(
                CameraStreamFacts(
                    camera_id=camera_id,
                    packet_count=0,
                    payload_bytes=0,
                    first_timestamp_ns=None,
                    last_timestamp_ns=None,
                    first_sequence=None,
                    last_sequence=None,
                    sequence_gap_count=0,
                )
                for camera_id in CAMERA_IDS
            ),
        )
    )
    composition.finalize_eos(
        EosSealInputs(
            eos_source_receipt_semantic_sha256=_digest(80),
            final_source_timeline_semantic_sha256=_digest(81),
            final_duration_ns=1_000_000_000,
            ordered_six_channel_health_closure_sha256=_digest(82),
            mapping_closure_semantic_sha256=_digest(83),
            clock_or_alignment_closure_semantic_sha256=_digest(84),
        )
    )
    composition.mark_export_barrier_complete(
        export_manifest_semantic_sha256=_digest(85),
        completed_member_count=6,
    )


def _finalizer(
    composition: DurableStreamWindowScheduler,
    tmp_path: Path,
    *,
    policy: str = "local-test-mock-v1",
    now: datetime = _NOW + timedelta(hours=1),
    stage_terminal_executor: (
        Callable[[StreamWorkItemPlan], StreamTerminalEvidence | None] | None
    ) = None,
    model_inference_schema_ref: SchemaRef | None = None,
    recover_graph_before_execute: bool = True,
) -> LocalConformanceStreamFinalizer:
    return LocalConformanceStreamFinalizer(
        scheduler=composition,
        delivery_authority=SQLiteStreamDeliveryAuthority(
            SQLiteWorkScheduler(composition.database_path),
            retry_policy=OutboxRetryPolicy(
                version="local-stream-delivery-test-v1",
                max_attempts=3,
                base_delay_seconds=1,
                max_delay_seconds=4,
            ),
            clock=lambda: now,
        ),
        artifact_root=tmp_path / "artifacts",
        schema_refs=LocalStreamFinalizationSchemaRefs(
            local_work_receipt=_schema(20),
            stream_window_result=_schema(21),
            recording_finalization=_schema(22),
            stream_recording_result=_recording_result_schema(),
            window_inference_plan=_schema(23),
            window_semantic_evidence_v2=_schema(24),
            stream_inference_identity=_schema(25),
            stream_inference_attempt=_schema(26),
            stream_inference_intent=_schema(27),
            stream_accepted_call=_schema(28),
            stream_inference_terminal=_schema(29),
            model_inference=model_inference_schema_ref,
        ),
        final_recording=FinalRecordingFacts(
            final_source_subject_type="MCAP_RECORDING",
            final_source_subject_id=_uuid(90),
            final_source_exact_sha256=_digest(91),
            final_recording_identity=_digest(92),
            final_duration_ns=1_000_000_000,
        ),
        window_purpose=StreamPurpose.QA_COARSE,
        mock_executor_policy_version=policy,
        recover_graph_before_execute=recover_graph_before_execute,
        stage_terminal_executor=stage_terminal_executor,
        clock=lambda: now,
    )


def test_local_executor_completes_dag_closure_and_finalization(tmp_path: Path) -> None:
    composition = _composition(tmp_path)
    finalizer = _finalizer(composition, tmp_path)

    outcome = finalizer.execute()

    assert outcome.evidence_class == "LOCAL_CONFORMANCE"
    assert outcome.newly_executed_work_count == 6
    assert len(outcome.window_results) == 1
    result = outcome.window_results[0]
    assert result.terminal_outcome is TerminalOutcome.NO_EVENTS
    assert len(result.accepted_terminals) == 1
    assert outcome.terminal_closure.complete
    assert outcome.terminal_closure.members[0].terminal_outcome is TerminalOutcome.NO_EVENTS
    assert outcome.finalization_work.stage is StreamStage.FINALIZATION
    assert outcome.finalization_work.state is StreamWorkItemState.SUCCEEDED
    assert tuple(
        mapping.incremental_subject_type
        for mapping in outcome.recording_finalization.ordered_subject_mappings
    ) == (StreamSubjectType.INCREMENTAL_WINDOW,)
    final_ref = outcome.finalization_work.terminal_evidence_ref
    assert final_ref is not None
    assert finalizer.artifact_path_for(final_ref).is_file()
    assert outcome.recording_result.output_decision == "NO_EVENTS"
    recovered_result, recovered_ref = load_completed_local_stream_recording_result(
        scheduler=composition,
        artifact_root=tmp_path / "artifacts",
        schema_ref=_recording_result_schema(),
    )
    assert recovered_result == outcome.recording_result
    assert recovered_ref == outcome.recording_result_evidence_ref

    delivery = SQLiteStreamDeliveryAuthority(
        SQLiteWorkScheduler(composition.database_path),
        retry_policy=OutboxRetryPolicy(
            version="local-stream-delivery-test-v1",
            max_attempts=3,
            base_delay_seconds=1,
            max_delay_seconds=4,
        ),
        clock=lambda: _NOW + timedelta(hours=1),
    )
    sink = SQLiteIdempotentOutboxSink(
        tmp_path / "stream-sink.sqlite3",
        clock=lambda: _NOW + timedelta(hours=1),
    )
    relay = OutboxRelay(
        store=delivery,
        sink=sink,
        worker_id="stream-test-relay",
        lease_duration=timedelta(minutes=1),
    )
    assert relay.deliver_once() is not None
    assert relay.deliver_once() is not None
    assert relay.deliver_once() is None
    assert sink.count() == 2
    assert delivery.reconcile() == 0
    assert relay.deliver_once() is None


def test_stream_relay_recovers_after_publish_before_ack_without_duplicate_fact(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)
    outcome = _finalizer(composition, tmp_path).execute()
    now = [_NOW + timedelta(hours=2)]
    authority = SQLiteWorkScheduler(composition.database_path)
    delivery = SQLiteStreamDeliveryAuthority(
        authority,
        retry_policy=OutboxRetryPolicy(
            version="local-stream-delivery-test-v1",
            max_attempts=3,
            base_delay_seconds=1,
            max_delay_seconds=4,
        ),
        clock=lambda: now[0],
    )
    sink = SQLiteIdempotentOutboxSink(
        tmp_path / "stream-crash-sink.sqlite3",
        clock=lambda: now[0],
    )

    abandoned = delivery.claim(
        worker_id="stream-relay-crashed",
        lease_duration=timedelta(seconds=10),
    )
    assert abandoned is not None
    assert abandoned.message.payload == canonical_json_bytes(outcome.window_results[0])
    sink.publish(abandoned.message)
    assert sink.count() == 1
    leased = delivery.get(abandoned.message.outbox_id)
    assert leased is not None
    assert leased.status is OutboxDeliveryStatus.LEASED
    assert leased.lease_epoch == abandoned.delivery.lease_epoch
    assert leased.attempt_count == 1

    now[0] += timedelta(seconds=11)
    recovered = delivery.claim(
        worker_id="stream-relay-recovered",
        lease_duration=timedelta(seconds=10),
    )
    assert recovered is not None
    assert recovered.message == abandoned.message
    assert recovered.delivery.lease_epoch == abandoned.delivery.lease_epoch + 1
    assert recovered.delivery.attempt_count == 2
    assert recovered.delivery.fencing_token != abandoned.delivery.fencing_token

    sink.publish(recovered.message)
    delivered = delivery.acknowledge(recovered)

    assert delivered.status is OutboxDeliveryStatus.DELIVERED
    assert delivered.attempt_count == 2
    assert sink.count() == 1
    with pytest.raises(OutboxFenceError, match="stale"):
        delivery.acknowledge(abandoned)

    result = outcome.window_results[0]
    counts = authority.run_authority_transaction(
        write=False,
        operation_name="test.read_recovered_stream_delivery",
        operation=lambda connection: (
            connection.execute(
                "SELECT COUNT(*) FROM stream_window_results WHERE window_result_key = ?",
                (result.window_result_key,),
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM stream_delivery_outbox WHERE outbox_id = ?",
                (delivered.outbox_id,),
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM stream_outbox_deliveries WHERE outbox_id = ?",
                (delivered.outbox_id,),
            ).fetchone()[0],
        ),
    )
    assert counts == (1, 1, 1)


def test_eos_result_loading_reuses_batched_execution_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = _composition(tmp_path, window_count=3)
    reduction_ids = {
        plan.work_item_id
        for plan in composition.work_plans()
        if plan.stage is StreamStage.WINDOW_REDUCTION
    }
    original_get = composition.get

    def reject_reduction_point_read(work_item_id: str) -> StreamWorkItem:
        if work_item_id in reduction_ids:
            raise AssertionError("EOS must reuse the batched execution snapshot")
        return original_get(work_item_id)

    monkeypatch.setattr(composition, "get", reject_reduction_point_read)

    outcome = _finalizer(composition, tmp_path).execute()

    assert len(outcome.window_results) == 3
    assert outcome.finalization_work.state is StreamWorkItemState.SUCCEEDED


def test_drain_ready_is_bounded_replay_safe_and_execute_finishes_eos(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path, close_eos=False)
    finalizer = _finalizer(composition, tmp_path)

    def terminal_nonfinal_ids() -> set[str]:
        return {
            plan.work_item_id
            for plan in composition.work_plans()
            if plan.stage is not StreamStage.FINALIZATION
            and composition.get(plan.work_item_id).terminal_evidence_ref is not None
        }

    assert finalizer.drain_ready(max_items=2) == 2
    first_batch = terminal_nonfinal_ids()
    assert len(first_batch) == 2

    (composition,) = DurableStreamWindowScheduler.recover_registered(
        execution_scheduler=SQLiteWorkScheduler(composition.database_path),
        stream_run_id=_uuid(50),
        clock=lambda: _NOW,
    )
    finalizer = _finalizer(composition, tmp_path)
    assert finalizer.drain_ready(max_items=2) == 2
    second_batch = terminal_nonfinal_ids()
    assert len(second_batch) == 4
    assert first_batch < second_batch

    assert finalizer.drain_ready(max_items=20) == 1
    completed = terminal_nonfinal_ids()
    assert len(completed) == 5
    assert finalizer.drain_ready(max_items=20) == 0
    assert terminal_nonfinal_ids() == completed

    with pytest.raises(
        LocalStreamFinalizationError,
        match="expected window plan must be sealed",
    ):
        finalizer.execute()

    _close_eos(composition)
    outcome = finalizer.execute()

    assert outcome.newly_executed_work_count == 1
    assert outcome.finalization_work.state is StreamWorkItemState.SUCCEEDED
    assert outcome.terminal_closure.complete
    assert len(outcome.window_results) == 1


def test_pre_eos_restart_resumes_owned_active_lease_without_second_provider_dispatch(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path, close_eos=False)
    qa_plan = next(
        plan for plan in composition.work_plans() if plan.stage is StreamStage.QA_COARSE
    )
    assert _finalizer(composition, tmp_path).drain_ready(max_items=1) == 1
    assert composition.get(qa_plan.work_item_id).state is StreamWorkItemState.READY
    model_inference_schema = SchemaRef(
        schema_id="https://schemas.robata.dev/model-inference",
        version="1.0.0",
        artifact_id=_uuid(301),
        sha256=_digest(302),
    )
    provider_payload = b"persisted-provider-terminal"
    provider_payload = canonical_json_bytes(_provider_fixture_model())
    provider_ref = ArtifactEvidenceRef(
        artifact_id=_uuid(303),
        exact_sha256=exact_bytes_sha256(provider_payload),
        byte_count=len(provider_payload),
        media_type="application/json",
        schema_ref=model_inference_schema,
    )
    provider_path = (
        tmp_path
        / "artifacts"
        / provider_ref.exact_sha256[:2]
        / f"{provider_ref.exact_sha256}.json"
    )
    provider_path.parent.mkdir(parents=True)
    provider_path.write_bytes(provider_payload)
    provider_terminal = StreamTerminalEvidence(
        outcome=TerminalOutcome.SUCCEEDED,
        evidence_ref=provider_ref,
        terminal_policy_version="stream-terminal-policy-v1",
        completed_at=_NOW.isoformat(),
    )
    hook_calls = 0
    provider_dispatches = 0

    def provider_terminal_executor(plan: StreamWorkItemPlan) -> StreamTerminalEvidence:
        nonlocal hook_calls, provider_dispatches
        assert plan.work_item_id == qa_plan.work_item_id
        hook_calls += 1
        if hook_calls == 1:
            provider_dispatches += 1
        return provider_terminal

    # The first call represents provider evidence persisted immediately before
    # a crash prevents scheduler.complete() from accepting the stream terminal.
    claim = composition.claim_and_start(
        "local-conformance-stream-worker",
        300,
        work_item_id=qa_plan.work_item_id,
        now=_NOW + timedelta(hours=1),
    )
    assert claim is not None
    assert provider_terminal_executor(qa_plan) == provider_terminal
    assert composition.get(qa_plan.work_item_id).state is StreamWorkItemState.RUNNING

    # A local restart retains the original fence and attempt. The finalizer
    # re-enters the idempotent provider hook, whose durable ledger would reuse
    # the persisted terminal rather than send another provider request.
    (recovered,) = DurableStreamWindowScheduler.recover_registered(
        execution_scheduler=SQLiteWorkScheduler(composition.database_path),
        stream_run_id=_uuid(50),
        clock=lambda: _NOW,
    )
    finalizer = _finalizer(
        recovered,
        tmp_path,
        stage_terminal_executor=provider_terminal_executor,
        model_inference_schema_ref=model_inference_schema,
        recover_graph_before_execute=False,
    )

    assert finalizer.drain_ready(max_items=1) == 1
    accepted = recovered.get(qa_plan.work_item_id)
    assert accepted.state is StreamWorkItemState.SUCCEEDED
    assert accepted.terminal_evidence_ref == provider_ref
    attempts = SQLiteWorkScheduler(recovered.database_path).list_attempts(
        qa_plan.work_item_id
    )
    assert len(attempts) == 1
    assert hook_calls == 2
    assert provider_dispatches == 1


def test_finalization_restart_resumes_owned_active_lease_without_second_attempt(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)
    finalizer = _finalizer(composition, tmp_path)
    nonfinal_count = sum(
        plan.stage is not StreamStage.FINALIZATION for plan in composition.work_plans()
    )
    assert finalizer.drain_ready(max_items=nonfinal_count) == nonfinal_count
    composition.close_finalization_gate()

    final_plan = next(
        plan for plan in composition.work_plans() if plan.stage is StreamStage.FINALIZATION
    )
    claim = composition.claim_and_start(
        "local-conformance-stream-worker",
        300,
        work_item_id=final_plan.work_item_id,
        now=_NOW + timedelta(hours=1),
        recover_graph=False,
    )
    assert claim is not None
    assert composition.get(final_plan.work_item_id).state is StreamWorkItemState.RUNNING

    recovered = _finalizer(
        composition,
        tmp_path,
        now=_NOW + timedelta(hours=1, minutes=1),
        recover_graph_before_execute=False,
    ).execute()

    assert recovered.finalization_work.state is StreamWorkItemState.SUCCEEDED
    assert recovered.newly_executed_work_count == 1
    attempts = SQLiteWorkScheduler(composition.database_path).list_attempts(
        final_plan.work_item_id
    )
    assert len(attempts) == 1


def test_provider_terminal_outcome_mismatch_fails_closed_before_window_reduction(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path, close_eos=False)
    model_schema = SchemaRef(
        schema_id='https://schemas.robata.dev/model-inference',
        version='1.0.0',
        artifact_id=_uuid(401),
        sha256=_digest(402),
    )
    model = _provider_fixture_model()
    payload = canonical_json_bytes(model)
    reference = ArtifactEvidenceRef(
        artifact_id=_uuid(403),
        exact_sha256=exact_bytes_sha256(payload),
        byte_count=len(payload),
        media_type='application/json',
        schema_ref=model_schema,
    )
    artifact_path = (
        tmp_path
        / 'artifacts'
        / reference.exact_sha256[:2]
        / f'{reference.exact_sha256}.json'
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(payload)

    def provider_terminal_executor(plan: StreamWorkItemPlan) -> StreamTerminalEvidence | None:
        if plan.stage is not StreamStage.QA_COARSE:
            return None
        return StreamTerminalEvidence(
            outcome=TerminalOutcome.FAILED,
            evidence_ref=reference,
            terminal_policy_version='stream-terminal-policy-v1',
            completed_at=_NOW.isoformat(),
            reason_code='TEST_CONTRADICTORY_TERMINAL',
        )

    with pytest.raises(
        LocalStreamFinalizationError,
        match='provider terminal outcome conflicts',
    ):
        _finalizer(
            composition,
            tmp_path,
            stage_terminal_executor=provider_terminal_executor,
            model_inference_schema_ref=model_schema,
        ).drain_ready(max_items=2)


def test_provider_stage_cannot_fall_back_to_local_receipt_when_model_schema_is_pinned(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path, close_eos=False)
    model_schema = SchemaRef(
        schema_id='https://schemas.robata.dev/model-inference',
        version='1.0.0',
        artifact_id=_uuid(451),
        sha256=_digest(452),
    )

    with pytest.raises(LocalStreamFinalizationError, match='must provide a model inference'):
        _finalizer(
            composition,
            tmp_path,
            stage_terminal_executor=lambda _plan: None,
            model_inference_schema_ref=model_schema,
        ).drain_ready(max_items=2)


def test_provider_stage_schema_pin_requires_a_provider_executor(tmp_path: Path) -> None:
    composition = _composition(tmp_path, close_eos=False)
    model_schema = SchemaRef(
        schema_id='https://schemas.robata.dev/model-inference',
        version='1.0.0',
        artifact_id=_uuid(461),
        sha256=_digest(462),
    )

    with pytest.raises(LocalStreamFinalizationError, match='must provide a model inference'):
        _finalizer(
            composition,
            tmp_path,
            model_inference_schema_ref=model_schema,
        ).drain_ready(max_items=2)


def test_valid_failed_provider_terminal_degrades_window_and_closes_eos(tmp_path: Path) -> None:
    composition = _composition(tmp_path)
    model_schema = SchemaRef(
        schema_id='https://schemas.robata.dev/model-inference',
        version='1.0.0',
        artifact_id=_uuid(411),
        sha256=_digest(412),
    )
    task_by_stage = {
        StreamStage.QA_COARSE: VisionTask.QA_COARSE,
        StreamStage.QA_DENSE: VisionTask.QA_DENSE,
        StreamStage.EVENT_PROPOSAL: VisionTask.EVENT_PROPOSAL,
    }

    def provider_terminal_executor(plan: StreamWorkItemPlan) -> StreamTerminalEvidence | None:
        task = task_by_stage.get(plan.stage)
        if task is None:
            return None
        model = _provider_fixture_model(task)
        outcome = TerminalOutcome.SUCCEEDED
        reason_code = None
        if plan.stage is StreamStage.QA_COARSE:
            model = model.model_copy(
                update={
                    'status': InferenceStatus.FAILED,
                    'raw_output': None,
                    'normalized_output': None,
                    'output_valid': False,
                }
            )
            outcome = TerminalOutcome.FAILED
            reason_code = 'TEST_PROVIDER_FAILURE'
        payload = canonical_json_bytes(model)
        reference = ArtifactEvidenceRef(
            artifact_id=_uuid(420 + len(payload)),
            exact_sha256=exact_bytes_sha256(payload),
            byte_count=len(payload),
            media_type='application/json',
            schema_ref=model_schema,
        )
        artifact_path = (
            tmp_path
            / 'artifacts'
            / reference.exact_sha256[:2]
            / f'{reference.exact_sha256}.json'
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(payload)
        return StreamTerminalEvidence(
            outcome=outcome,
            evidence_ref=reference,
            terminal_policy_version='stream-terminal-policy-v1',
            completed_at=_NOW.isoformat(),
            reason_code=reason_code,
        )

    outcome = _finalizer(
        composition,
        tmp_path,
        stage_terminal_executor=provider_terminal_executor,
        model_inference_schema_ref=model_schema,
    ).execute()

    assert outcome.window_results[0].terminal_outcome is TerminalOutcome.ABSTAINED
    assert outcome.terminal_closure.complete
    assert outcome.recording_result.output_decision == 'ABSTAINED'

def test_exact_replay_executes_no_work_and_policy_change_fails_closed(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)
    first = _finalizer(composition, tmp_path).execute()
    before = tuple(sorted(path.name for path in (tmp_path / "artifacts").rglob("*.json")))

    replay = _finalizer(
        composition,
        tmp_path,
        now=_NOW + timedelta(days=1),
    ).execute()
    after = tuple(sorted(path.name for path in (tmp_path / "artifacts").rglob("*.json")))

    assert replay.newly_executed_work_count == 0
    assert replay.window_results == first.window_results
    assert replay.recording_finalization == first.recording_finalization
    assert after == before

    with pytest.raises(LocalStreamFinalizationError, match="executor policy"):
        _finalizer(composition, tmp_path, policy="local-test-mock-v2").execute()


def test_all_expected_windows_reduce_once_and_exactly_replay(tmp_path: Path) -> None:
    composition = _composition(tmp_path, window_count=2)
    first = _finalizer(composition, tmp_path).execute()

    assert first.terminal_closure.complete
    assert first.terminal_closure.expected_member_count == 2
    assert len(first.window_results) == 2
    assert len(first.recording_finalization.ordered_subject_mappings) == 2

    recovered = DurableStreamWindowScheduler.recover_registered(
        execution_scheduler=SQLiteWorkScheduler(composition.database_path),
        stream_run_id=_uuid(50),
        clock=lambda: _NOW,
    )
    assert len(recovered) == 1
    replay = _finalizer(recovered[0], tmp_path, now=_NOW + timedelta(days=1)).execute()

    assert replay.newly_executed_work_count == 0
    assert replay.recording_finalization == first.recording_finalization
    assert replay.recording_result == first.recording_result


def test_window_publication_rolls_back_and_recovers_after_injected_outbox_failure(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)
    first = _finalizer(composition, tmp_path)
    authority = SQLiteWorkScheduler(composition.database_path)
    authority.run_authority_transaction(
        write=True,
        operation_name="test.inject_stream_outbox_failure",
        operation=lambda connection: connection.execute(
            """
            CREATE TRIGGER injected_stream_outbox_failure
            BEFORE INSERT ON stream_delivery_outbox
            BEGIN
                SELECT RAISE(ABORT, 'injected stream outbox failure');
            END
            """
        ),
    )

    with pytest.raises(
        SQLiteStreamDeliveryError,
        match="stream delivery commit_window_reduction failed",
    ):
        first.execute()

    counts = authority.run_authority_transaction(
        write=False,
        operation_name="test.read_stream_rollback",
        operation=lambda connection: (
            connection.execute("SELECT COUNT(*) FROM stream_window_results").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM stream_delivery_outbox").fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM expected_windows WHERE terminal_member_json IS NOT NULL"
            ).fetchone()[0],
        ),
    )
    assert counts == (0, 0, 0)

    authority.run_authority_transaction(
        write=True,
        operation_name="test.remove_stream_outbox_failure",
        operation=lambda connection: connection.execute(
            "DROP TRIGGER injected_stream_outbox_failure"
        ),
    )
    recovered = _finalizer(
        composition,
        tmp_path,
        now=_NOW + timedelta(hours=2),
    ).execute()
    assert recovered.finalization_work.state is StreamWorkItemState.SUCCEEDED
    assert len(recovered.window_results) == 1


def test_delivery_authority_rejects_failed_required_window_before_transaction(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)
    finalizer = _finalizer(composition, tmp_path)
    assert finalizer.drain_ready(max_items=4) == 4
    reduction = next(
        plan for plan in composition.work_plans() if plan.stage is StreamStage.WINDOW_REDUCTION
    )
    claim = composition.claim(
        "failed-window-worker",
        300,
        work_item_id=reduction.work_item_id,
        now=_NOW + timedelta(hours=1),
    )
    assert claim is not None
    composition.start(claim.lease, now=_NOW + timedelta(hours=1))
    result = create_stream_window_result(
        schema_ref=_schema(21),
        window_subject=reduction.subject,
        purpose=StreamPurpose.QA_COARSE,
        terminal_outcome=TerminalOutcome.FAILED,
        accepted_terminals=(),
        result_semantic_evidence_sha256=_digest(3000),
        result_evidence_ref=ArtifactEvidenceRef(
            artifact_id=_uuid(3001),
            exact_sha256=_digest(3002),
            byte_count=1,
            media_type="application/json",
            schema_ref=_schema(20),
        ),
        reduction_policy_version="failed-window-test-v1",
        created_at=_NOW.isoformat(),
    )
    payload = canonical_json_bytes(result)
    evidence = StreamTerminalEvidence(
        outcome=TerminalOutcome.FAILED,
        evidence_ref=ArtifactEvidenceRef(
            artifact_id=_uuid(3003),
            exact_sha256=exact_bytes_sha256(payload),
            byte_count=len(payload),
            media_type="application/json",
            schema_ref=result.schema_ref,
        ),
        terminal_policy_version="stream-terminal-policy-v1",
        completed_at=_NOW.isoformat(),
        reason_code="FAILED_REQUIRED_WINDOW",
    )
    member = composition.prepare_window_terminal_member(claim.lease, evidence)
    delivery = SQLiteStreamDeliveryAuthority(
        SQLiteWorkScheduler(composition.database_path),
        retry_policy=OutboxRetryPolicy(
            version="local-stream-delivery-test-v1",
            max_attempts=3,
            base_delay_seconds=1,
            max_delay_seconds=4,
        ),
        clock=lambda: _NOW + timedelta(hours=1),
    )

    with pytest.raises(
        SQLiteStreamDeliveryConflict,
        match="failed or incomplete required window",
    ):
        delivery.commit_window_reduction(
            lease=claim.lease,
            terminal_evidence=evidence,
            terminal_member=member,
            result=result,
            topic="robata.stream.window-results.v1",
            message_key=result.window_result_key,
            now=_NOW + timedelta(hours=1),
        )

    assert composition.get(reduction.work_item_id).state is StreamWorkItemState.RUNNING
    counts = SQLiteWorkScheduler(composition.database_path).run_authority_transaction(
        write=False,
        operation_name="test.failed_window_not_committed",
        operation=lambda connection: (
            connection.execute("SELECT COUNT(*) FROM stream_window_results").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM stream_delivery_outbox").fetchone()[0],
        ),
    )
    assert counts == (0, 0)
