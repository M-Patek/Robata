from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from robata.adapters.sqlite_inference_evidence import (
    SQLiteInferenceEvidenceLedger,
    SQLiteInferenceEvidenceLedgerError,
)
from robata.adapters.sqlite_primary_completion import SQLitePrimaryCompletionRepository
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler, WorkStorageError
from robata.application.canonical import local_composition as local_composition_module
from robata.application.canonical import local_outbox_delivery as local_outbox_delivery_module
from robata.application.canonical.durable_work import (
    canonical_action_publish_plan_from_committed,
)
from robata.application.canonical.local_composition import (
    CanonicalLocalCompositionError,
    CanonicalLocalCompositionErrorCode,
    CanonicalLocalRunReceipt,
    run_local_canonical_fixture,
)
from robata.application.canonical.local_outbox_delivery import LocalOutboxDeliveryOutcome
from robata.application.canonical.parallel_service import (
    CanonicalLocalFixtureJob,
    CanonicalLocalRecordingService,
)
from robata.application.canonical.primary_completion import PreparedPrimaryCompletionCommand
from robata.application.canonical.recording_association import (
    RecordingAssociationPublicationStatus,
    RecordingAssociationReportStore,
)
from robata.application.canonical.recording_association_dispatch import (
    CanonicalRecordingAssociationWorker,
    RecordingAssociationJobStore,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.queue.models import WorkAttemptOutcome, WorkItemState
from robata.review.routing import ReviewRoutingDisposition
from robata.runtime.observability import RuntimeProfileRecorder

SOURCE_FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical" / "source-recording.json"


def _assert_local_conformance(receipt: CanonicalLocalRunReceipt) -> None:
    assert receipt.schema_version == "1.0"
    assert receipt.ok is True
    assert receipt.status == "SUCCEEDED"
    assert receipt.network_call_count == 0
    assert receipt.evidence_class == "LOCAL_CONFORMANCE"
    assert receipt.production_eligible is False


def _primary_completion_bytes(state_dir: Path) -> tuple[bytes, str, bytes, str]:
    with sqlite3.connect(state_dir / "primary-completion.sqlite3") as connection:
        detail = connection.execute(
            "SELECT payload_json, exact_bytes_sha256 FROM detailed_results"
        ).fetchone()
        completion = connection.execute(
            "SELECT command_json, command_json_sha256 FROM primary_completions"
        ).fetchone()
    assert detail is not None
    assert completion is not None
    return detail[0], detail[1], completion[0], completion[1]


def _sidecar_json_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
    }


def _counter_has_status(
    recorder: RuntimeProfileRecorder,
    *,
    name: str,
    status: str,
) -> bool:
    return any(
        counter.name == name
        and dict((attribute.name, attribute.value) for attribute in counter.attributes).get(
            "status"
        )
        == status
        for counter in recorder.snapshot().counters
    )


def test_local_command_commits_then_exactly_replays_one_run(tmp_path: Path) -> None:
    state_dir = tmp_path / "canonical-state"

    first = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="integration-exact-replay",
    )

    _assert_local_conformance(first)
    assert first.replayed is False
    assert first.fixture_inference_calls > 0
    assert first.event_ids
    assert first.revision_ids
    assert first.outbox_ids
    assert len(first.event_ids) == len(first.revision_ids) == len(first.outbox_ids)
    assert first.outbox_count == len(first.outbox_ids)
    assert first.model_version == "canonical-local-run-receipt-v4"
    assert first.media_quality_binding is None
    assert first.supplemental_qa_evidence is None
    assert first.review_routing.disposition is ReviewRoutingDisposition.ENQUEUED
    assert first.outbox_delivery.outcome is LocalOutboxDeliveryOutcome.DELIVERED
    assert first.outbox_delivery.delivered_count == first.outbox_count
    assert first.outbox_delivery.outbox_ids == first.outbox_ids
    assert (state_dir / "work-scheduler.sqlite3").is_file()
    assert (state_dir / "runs" / first.run_id / "inference-call-barrier.sqlite3").is_file()

    replay = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="integration-exact-replay",
    )

    _assert_local_conformance(replay)
    assert replay.replayed is True
    assert replay.fixture_inference_calls == 0
    assert replay.run_id == first.run_id
    assert replay.recording_identity == first.recording_identity
    assert replay.status == first.status
    assert replay.command_sha256 == first.command_sha256
    assert replay.completion_semantic_sha256 == first.completion_semantic_sha256
    assert replay.event_ids == first.event_ids
    assert replay.revision_ids == first.revision_ids
    assert replay.outbox_ids == first.outbox_ids
    assert replay.outbox_count == first.outbox_count
    assert replay.outbox_delivery.outcome is LocalOutboxDeliveryOutcome.DELIVERED
    assert replay.outbox_delivery.relay_attempt_count == 0
    assert replay.review_routing.disposition is ReviewRoutingDisposition.ALREADY_ENQUEUED
    sink_path = state_dir / "outbox-sink.sqlite3"
    with sqlite3.connect(sink_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM delivered_outbox_messages").fetchone()[
            0
        ] == len(first.outbox_ids)
        connection.execute("DROP TRIGGER delivered_outbox_messages_no_update")
        connection.commit()

    unreconciled = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="integration-exact-replay",
    )
    assert unreconciled.ok is True
    assert unreconciled.replayed is True
    assert unreconciled.outbox_delivery.outcome is LocalOutboxDeliveryOutcome.FAILED
    assert unreconciled.outbox_delivery.last_error is not None
    assert "cannot reconcile delivered rows" in unreconciled.outbox_delivery.last_error


def test_local_completion_queues_then_worker_publishes_detached_association_report(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "canonical-state"
    receipt = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="recording-association-detached",
    )

    _assert_local_conformance(receipt)
    job_root = state_dir / "recording-association"
    jobs = RecordingAssociationJobStore(job_root)
    queued = jobs.list_jobs()
    assert len(queued) == 1
    job = queued[0]
    assert job.inputs
    assert not (job_root / "reports").exists()

    repository = SQLitePrimaryCompletionRepository(state_dir / "primary-completion.sqlite3")
    primary_before = repository.get(receipt.run_id)
    assert primary_before is not None
    assert job.recording.completed_run_id == receipt.run_id
    assert (
        job.recording.completed_recording_semantic_sha256 == primary_before.detail.semantic_sha256
    )
    assert (
        job.recording.completed_recording_exact_sha256
        == primary_before.completion.detailed_result.exact_bytes_sha256
    )

    reports = RecordingAssociationReportStore(job_root)
    worker = CanonicalRecordingAssociationWorker(jobs=jobs, reports=reports)
    first_execution = worker.drain()
    assert len(first_execution) == 1
    publication = first_execution[0].publication
    assert publication.status is RecordingAssociationPublicationStatus.PUBLISHED
    assert publication.replayed is False
    assert publication.report is not None
    assert publication.report.recording == job.recording
    assert publication.report.inputs == job.inputs
    assert repository.get(receipt.run_id) == primary_before

    replay = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="recording-association-detached",
    )
    _assert_local_conformance(replay)
    assert replay.replayed is True
    assert replay.event_ids == receipt.event_ids
    assert replay.revision_ids == receipt.revision_ids
    assert replay.outbox_ids == receipt.outbox_ids
    assert jobs.list_jobs() == queued

    replay_execution = worker.drain()
    assert len(replay_execution) == 1
    assert replay_execution[0].publication.status is RecordingAssociationPublicationStatus.REPLAYED
    assert replay_execution[0].publication.replayed is True
    assert replay_execution[0].publication.report == publication.report
    assert repository.get(receipt.run_id) == primary_before


def test_disabling_recording_association_preserves_v3_v4_primary_completion_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_state_dir = tmp_path / "association-enabled"
    enabled = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=enabled_state_dir,
        run_key="recording-association-disabled-bytes",
    )
    _assert_local_conformance(enabled)
    enabled_primary = SQLitePrimaryCompletionRepository(
        enabled_state_dir / "primary-completion.sqlite3"
    ).get(enabled.run_id)
    assert enabled_primary is not None
    assert enabled_primary.completion.schema_version == "3.0"
    assert enabled_primary.detail.schema_version == "4.0"
    assert RecordingAssociationJobStore(enabled_state_dir / "recording-association").list_jobs()

    # P11 is detached from primary authority. Turning off its composition hook
    # may suppress the derived job only; it cannot alter released V3/V4 bytes.
    def disable_association(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        local_composition_module,
        "_enqueue_local_recording_association_dispatch",
        disable_association,
    )
    disabled_state_dir = tmp_path / "association-disabled"
    disabled = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=disabled_state_dir,
        run_key="recording-association-disabled-bytes",
    )
    _assert_local_conformance(disabled)
    disabled_primary = SQLitePrimaryCompletionRepository(
        disabled_state_dir / "primary-completion.sqlite3"
    ).get(disabled.run_id)
    assert disabled_primary is not None
    assert not (disabled_state_dir / "recording-association").exists()

    assert disabled.run_id == enabled.run_id
    assert canonical_json_bytes(disabled_primary.completion) == canonical_json_bytes(
        enabled_primary.completion
    )
    assert canonical_json_bytes(disabled_primary.detail) == canonical_json_bytes(
        enabled_primary.detail
    )


def test_local_completion_survives_recording_association_enqueue_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"
    recorder = RuntimeProfileRecorder()

    def fail_dispatch(**_kwargs: object) -> None:
        raise OSError("association dispatch storage unavailable")

    monkeypatch.setattr(
        local_composition_module,
        "enqueue_recording_association_after_completion",
        fail_dispatch,
    )
    receipt = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="recording-association-enqueue-failure",
        runtime_observer=recorder,
    )

    _assert_local_conformance(receipt)
    assert (
        SQLitePrimaryCompletionRepository(state_dir / "primary-completion.sqlite3").get(
            receipt.run_id
        )
        is not None
    )
    assert any(
        counter.name == "recording_association.dispatches"
        and dict((attribute.name, attribute.value) for attribute in counter.attributes).get(
            "status"
        )
        == "FAILED"
        for counter in recorder.snapshot().counters
    )


def test_local_completion_persists_calibration_adaptive_and_boundary_sidecars(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "canonical-state"
    fresh = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="p9-p10-p12-sidecars",
    )

    _assert_local_conformance(fresh)
    primary_before = _primary_completion_bytes(state_dir)
    repository = SQLitePrimaryCompletionRepository(state_dir / "primary-completion.sqlite3")
    committed = repository.get(fresh.run_id)
    assert committed is not None

    with sqlite3.connect(state_dir / "inference-evidence.sqlite3") as connection:
        artifact_count = connection.execute(
            "SELECT COUNT(*) FROM calibration_artifacts"
        ).fetchone()[0]
        calibration_rows = connection.execute(
            """
            SELECT outcome, raw_score, calibrated_probability, calibration_artifact_id
            FROM inference_calibration_associations
            """
        ).fetchall()
    assert artifact_count == 0
    assert calibration_rows
    assert all(row[2] is None and row[3] is None for row in calibration_rows)
    assert any(
        row[0] == "RAW_FALLBACK_MISSING_ARTIFACT" and row[1] == pytest.approx(0.8)
        for row in calibration_rows
    )

    decision_database = state_dir / "adaptive-sampling-decisions.sqlite3"
    with sqlite3.connect(decision_database) as connection:
        decision_rows = connection.execute(
            "SELECT payload_json FROM adaptive_sampling_decisions"
        ).fetchall()
    assert len(decision_rows) == 1
    decision_bytes = decision_rows[0][0]
    decision = json.loads(decision_bytes)
    assert decision["outcome"] == "DENSE_QA_ALREADY_COMPLETE"
    assert decision["incremental_targets"] == []
    assert decision["no_additional_work_proof"]["proof_kind"] == "DENSE_QA_COARSE_COMPLETE"
    assert b"NO_EVENTS" not in decision_bytes

    adaptive_root = state_dir / "adaptive-sampling-executions"
    terminal_bytes = _sidecar_json_bytes(adaptive_root)
    assert len(terminal_bytes) == 2
    terminal_payload = next(
        json.loads(payload)
        for path, payload in terminal_bytes.items()
        if path.startswith("terminals/")
    )
    assert terminal_payload["terminal_kind"] == "NO_ADDITIONAL_WORK"
    assert terminal_payload["no_additional_work_outcome"] == "DENSE_QA_ALREADY_COMPLETE"
    assert b"NO_EVENTS" not in canonical_json_bytes(terminal_payload)

    boundary_root = state_dir / "boundary-qualification"
    boundary_bytes = _sidecar_json_bytes(boundary_root)
    boundary_jobs = [path for path in boundary_bytes if path.startswith("jobs/")]
    boundary_reports = [path for path in boundary_bytes if path.startswith("reports/")]
    assert len(boundary_jobs) == len(committed.detail.boundary_refinement_executions)
    assert len(boundary_reports) == len(boundary_jobs)
    assert all(
        json.loads(boundary_bytes[path])["production_eligible"] is False
        and b"NO_EVENTS" not in boundary_bytes[path]
        for path in boundary_reports
    )

    replay = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="p9-p10-p12-sidecars",
    )

    _assert_local_conformance(replay)
    assert replay.replayed is True
    assert replay.fixture_inference_calls == 0
    assert _primary_completion_bytes(state_dir) == primary_before
    with sqlite3.connect(decision_database) as connection:
        replayed_decision = connection.execute(
            "SELECT payload_json FROM adaptive_sampling_decisions"
        ).fetchall()
    assert replayed_decision == [(decision_bytes,)]
    assert _sidecar_json_bytes(adaptive_root) == terminal_bytes
    assert _sidecar_json_bytes(boundary_root) == boundary_bytes


def test_local_sidecar_failures_do_not_replace_or_mutate_primary_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"
    recorder = RuntimeProfileRecorder()
    original_adaptive_execute = (
        local_composition_module.CanonicalAdaptiveSamplingBridge.execute_for_canonical_result
    )
    original_boundary_drain = local_composition_module.CanonicalBoundaryQualificationWorker.drain

    def fail_adaptive_execute(*_args: object, **_kwargs: object) -> None:
        raise OSError("adaptive sidecar unavailable")

    def fail_boundary_drain(*_args: object, **_kwargs: object) -> None:
        raise OSError("boundary sidecar unavailable")

    monkeypatch.setattr(
        local_composition_module.CanonicalAdaptiveSamplingBridge,
        "execute_for_canonical_result",
        fail_adaptive_execute,
    )
    monkeypatch.setattr(
        local_composition_module.CanonicalBoundaryQualificationWorker,
        "drain",
        fail_boundary_drain,
    )
    fresh = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="p10-p12-failure-isolation",
        runtime_observer=recorder,
    )

    _assert_local_conformance(fresh)
    primary_before = _primary_completion_bytes(state_dir)
    assert _counter_has_status(
        recorder,
        name="adaptive_sampling.dispatches",
        status="FAILED",
    )
    assert _counter_has_status(
        recorder,
        name="boundary_qualification.dispatches",
        status="FAILED",
    )

    monkeypatch.setattr(
        local_composition_module.CanonicalAdaptiveSamplingBridge,
        "execute_for_canonical_result",
        original_adaptive_execute,
    )
    monkeypatch.setattr(
        local_composition_module.CanonicalBoundaryQualificationWorker,
        "drain",
        original_boundary_drain,
    )
    replay = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="p10-p12-failure-isolation",
    )

    _assert_local_conformance(replay)
    assert replay.replayed is True
    assert _primary_completion_bytes(state_dir) == primary_before
    assert (state_dir / "adaptive-sampling-executions" / "terminals").is_dir()
    assert list((state_dir / "boundary-qualification" / "reports").glob("*.json"))


def test_runtime_observation_preserves_canonical_identity_and_replay(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "canonical-state"
    recorder = RuntimeProfileRecorder()

    observed = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="instrumented-exact-replay",
        runtime_observer=recorder,
    )
    profile = recorder.snapshot()
    replay_recorder = RuntimeProfileRecorder()
    replay = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="instrumented-exact-replay",
        runtime_observer=replay_recorder,
    )
    replay_profile = replay_recorder.snapshot()

    _assert_local_conformance(observed)
    assert observed.replayed is False
    assert replay.replayed is True
    assert replay.fixture_inference_calls == 0
    assert replay.run_id == observed.run_id
    assert replay.recording_identity == observed.recording_identity
    assert replay.command_sha256 == observed.command_sha256
    assert replay.completion_semantic_sha256 == observed.completion_semantic_sha256
    assert replay.event_ids == observed.event_ids
    assert replay.revision_ids == observed.revision_ids
    assert replay.outbox_ids == observed.outbox_ids
    span_names = {span.name for span in profile.spans}
    assert {
        "canonical.composition",
        "inference.pipeline",
        "inference.orchestration",
        "inference.provider_dispatch",
        "sqlite.inference_evidence.transaction",
        "sqlite.inference_evidence.begin",
        "sqlite.inference_evidence.integrity_check",
        "sqlite.inference_evidence.operation",
        "sqlite.inference_evidence.commit",
        "sqlite.barrier.transaction",
        "sqlite.work_scheduler.transaction",
        "sqlite.logical_node_registry.transaction",
        "sqlite.primary_completion.transaction",
        "sqlite.outbox_delivery.transaction",
        "sqlite.outbox_sink.transaction",
        "sqlite.review_queue.transaction",
        "completion.evidence.audit",
        "completion.command.serialize_validate",
        "completion.command.detail.semantic_hash",
        "completion.command.detail.schema_validate",
        "completion.command.detail.model_validate",
        "completion.command.detail.registry_validate",
        "completion.command.detail.hypothesis_binding_sort",
        "completion.command.detail.serialize",
        "completion.command.ordered_roots",
        "completion.command.roots.leaf_prepare",
        "completion.command.roots.compute",
        "completion.command.completion.schema_validate",
        "completion.command.completion.registry_validate",
        "completion.command.command.semantic_hash",
        "completion.command.command.schema_validate",
        "completion.command.command.model_validate",
        "completion.command.command.serialize",
        "completion.command.processing_run.serialize",
        "completion.commit",
        "completion.commit.validate",
        "completion.commit.identity",
        "completion.commit.detail",
        "completion.commit.detail_artifact",
        "completion.commit.detail_artifact.verify",
        "completion.commit.detail_artifact.insert",
        "completion.commit.serialize",
        "completion.commit.committed.serialize",
        "completion.commit.primary_record_persist",
        "completion.commit.outbox",
        "completion.commit.run_close",
        "completion.commit.authoritative",
        "completion.commit.database_commit",
        "delivery.outbox.reconcile",
        "review.route",
    } <= span_names
    transaction_count = sum(
        counter.value
        for counter in profile.counters
        if counter.name == "sqlite.inference_evidence.transactions"
    )
    assert transaction_count > observed.fixture_inference_calls
    completion_counter_names = {
        "completion.command.detail_bytes",
        "completion.command.command_bytes",
        "completion.command.processing_run_bytes",
        "completion.command.root_collections",
        "completion.command.root_leaves",
    }
    assert all(
        any(counter.name == name and counter.value > 0 for counter in profile.counters)
        for name in completion_counter_names
    )
    assert (
        sum(
            counter.value
            for counter in profile.counters
            if counter.name == "completion.command.root_collections"
        )
        == 11
    )
    provider_dispatch_count = sum(
        counter.value
        for counter in profile.counters
        if counter.name == "inference.provider_dispatches"
    )
    assert provider_dispatch_count == observed.fixture_inference_calls
    assert (
        sum(
            counter.value
            for counter in profile.counters
            if counter.name == "durable_work.terminal_outcomes"
        )
        == 1
    )
    replay_span_names = {span.name for span in replay_profile.spans}
    assert "delivery.outbox.reconcile" in replay_span_names
    assert "delivery.outbox.relay" not in replay_span_names
    assert "inference.provider_dispatch" not in replay_span_names
    assert not any(
        counter.name == "inference.provider_dispatches" for counter in replay_profile.counters
    )
    assert (
        sum(
            counter.value
            for counter in replay_profile.counters
            if counter.name == "durable_work.terminal_outcomes"
        )
        == 1
    )
    assert any(
        counter.name == "delivery.outbox.outcomes"
        and any(
            attribute.name == "outcome" and attribute.value == replay.outbox_delivery.outcome.value
            for attribute in counter.attributes
        )
        for counter in replay_profile.counters
    )
    assert any(
        counter.name == "review.routing_outcomes"
        and any(
            attribute.name == "disposition"
            and attribute.value == replay.review_routing.disposition.value
            for attribute in counter.attributes
        )
        for counter in replay_profile.counters
    )


def test_committed_completion_reconciles_publish_work_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"
    original_succeed = SQLiteWorkScheduler.succeed
    injected = False

    def fail_first_publish_succeed(self, lease, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal injected
        if not injected and self.database_path.name == "work-scheduler.sqlite3":
            injected = True
            raise WorkStorageError("injected crash after primary completion commit")
        return original_succeed(self, lease, **kwargs)

    monkeypatch.setattr(SQLiteWorkScheduler, "succeed", fail_first_publish_succeed)
    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_fixture(
            source_path=SOURCE_FIXTURE,
            state_dir=state_dir,
            run_key="post-commit-pre-work-success-crash",
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    with sqlite3.connect(state_dir / "primary-completion.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM primary_completions").fetchone()[0] == 1

    monkeypatch.setattr(SQLiteWorkScheduler, "succeed", original_succeed)
    recovered = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="post-commit-pre-work-success-crash",
    )

    assert recovered.replayed is True
    assert recovered.fixture_inference_calls == 0
    repository = SQLitePrimaryCompletionRepository(state_dir / "primary-completion.sqlite3")
    committed = repository.get(recovered.run_id)
    assert committed is not None
    plan = canonical_action_publish_plan_from_committed(committed)
    scheduler = SQLiteWorkScheduler(state_dir / "work-scheduler.sqlite3")
    work = scheduler.get(plan.work_item_id)
    assert work.state is WorkItemState.SUCCEEDED
    assert work.result_reference == f"primary-completion:{recovered.run_id}"
    assert [item.outcome for item in scheduler.list_attempts(plan.work_item_id)] == [
        WorkAttemptOutcome.SUCCEEDED
    ]


def test_local_command_new_run_reuses_inference_event_and_revision(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "canonical-state"
    first = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="integration-first-run",
    )

    second = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="integration-second-run",
    )

    _assert_local_conformance(second)
    assert second.replayed is False
    assert second.fixture_inference_calls == 0
    assert second.run_id != first.run_id
    assert second.recording_identity == first.recording_identity
    assert second.event_ids == first.event_ids
    assert second.revision_ids == first.revision_ids
    assert second.outbox_ids == ()
    assert second.outbox_count == 0
    assert second.outbox_delivery.outcome is LocalOutboxDeliveryOutcome.NOT_APPLICABLE
    first_barrier = state_dir / "runs" / first.run_id / "inference-call-barrier.sqlite3"
    second_barrier = state_dir / "runs" / second.run_id / "inference-call-barrier.sqlite3"
    assert first_barrier.is_file()
    assert second_barrier.is_file()
    assert first_barrier != second_barrier


def test_recovered_completion_delivers_pending_outbox_after_post_commit_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"
    original_reconcile = local_composition_module.reconcile_local_primary_outbox

    def crash_before_relay(**_kwargs: object) -> None:
        raise SystemExit("injected crash after primary commit")

    monkeypatch.setattr(
        local_composition_module,
        "reconcile_local_primary_outbox",
        crash_before_relay,
    )
    with pytest.raises(SystemExit, match="after primary commit"):
        run_local_canonical_fixture(
            source_path=SOURCE_FIXTURE,
            state_dir=state_dir,
            run_key="post-commit-pre-relay-crash",
        )

    primary_path = state_dir / "primary-completion.sqlite3"
    with sqlite3.connect(primary_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM primary_completions").fetchone()[0] == 1
        pending = connection.execute(
            "SELECT COUNT(*) FROM primary_outbox WHERE delivered_at IS NULL"
        ).fetchone()[0]
        assert pending > 0
    assert not (state_dir / "outbox-sink.sqlite3").exists()

    monkeypatch.setattr(
        local_composition_module,
        "reconcile_local_primary_outbox",
        original_reconcile,
    )
    recovered = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="post-commit-pre-relay-crash",
    )

    assert recovered.replayed is True
    assert recovered.fixture_inference_calls == 0
    assert recovered.outbox_delivery.outcome is LocalOutboxDeliveryOutcome.DELIVERED
    assert recovered.outbox_delivery.delivered_count == recovered.outbox_count
    with sqlite3.connect(primary_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM primary_outbox WHERE delivered_at IS NULL"
            ).fetchone()[0]
            == 0
        )


def test_sink_failure_is_durable_without_replacing_primary_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"

    def fail_publish(_sink: object, _message: object) -> None:
        raise RuntimeError("injected local sink outage")

    monkeypatch.setattr(
        local_outbox_delivery_module.SQLiteIdempotentOutboxSink,
        "publish",
        fail_publish,
    )
    receipt = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="sink-failure-does-not-replace-primary",
    )

    assert receipt.ok is True
    assert receipt.status == "SUCCEEDED"
    assert receipt.outbox_delivery.outcome is LocalOutboxDeliveryOutcome.PENDING
    assert receipt.outbox_delivery.retry_wait_count == receipt.outbox_count
    assert receipt.outbox_delivery.delivered_count == 0
    assert receipt.outbox_delivery.last_error == "RuntimeError: injected local sink outage"
    with sqlite3.connect(state_dir / "primary-completion.sqlite3") as connection:
        assert (
            connection.execute(
                "SELECT primary_status FROM primary_runs WHERE run_id = ?",
                (receipt.run_id,),
            ).fetchone()[0]
            == "SUCCEEDED"
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM primary_completions WHERE run_id = ?",
                (receipt.run_id,),
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM primary_outbox WHERE delivered_at IS NULL"
            ).fetchone()[0]
            == receipt.outbox_count
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM primary_outbox_deliveries WHERE status = 'RETRY_WAIT'"
            ).fetchone()[0]
            == receipt.outbox_count
        )


def test_local_command_policy_change_cannot_replay_stale_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"
    first = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="policy-bound-recovery",
    )
    original_factory = local_composition_module._local_inference_policies

    def changed_policies(registry):  # type: ignore[no-untyped-def]
        coarse, dense, proposal, action, boundary, fusion = original_factory(registry)
        return (
            coarse.model_copy(
                update={
                    "policy_version": "offline-coarse-qa-model-policy-v2",
                    "prompt_version": "coarse-qa-prompt-v2",
                    "prompt_artifact_id": local_composition_module._stable_uuid(
                        "canonical-local-prompt",
                        "coarse-qa-prompt-v2",
                    ),
                    "prompt_sha256": local_composition_module.exact_bytes_sha256(
                        b"robata canonical local coarse QA prompt v2"
                    ),
                }
            ),
            dense,
            proposal,
            action,
            boundary,
            fusion,
        )

    monkeypatch.setattr(
        local_composition_module,
        "_local_inference_policies",
        changed_policies,
    )
    recovered_run_ids: list[str] = []
    original_get = local_composition_module.SQLitePrimaryCompletionRepository.get

    def recording_get(repository, run_id):  # type: ignore[no-untyped-def]
        recovered_run_ids.append(run_id)
        return original_get(repository, run_id)

    monkeypatch.setattr(
        local_composition_module.SQLitePrimaryCompletionRepository,
        "get",
        recording_get,
    )

    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_fixture(
            source_path=SOURCE_FIXTURE,
            state_dir=state_dir,
            run_key="policy-bound-recovery",
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.RUN_NOT_COMPLETABLE
    assert "REUSED identity assignments require a prior selection chain" in str(caught.value)
    assert len(recovered_run_ids) == 1
    assert recovered_run_ids[0] != first.run_id


def test_local_command_maps_invalid_state_schema_to_structured_error(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "canonical-state"
    state_dir.mkdir()
    with sqlite3.connect(state_dir / "inference-evidence.sqlite3") as connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_fixture(
            source_path=SOURCE_FIXTURE,
            state_dir=state_dir,
            run_key="invalid-state",
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    assert "unsupported inference evidence schema version" in str(caught.value)


def test_local_command_seals_inference_evidence_before_primary_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"
    sealed_paths: list[Path] = []
    full_audit_paths: list[Path] = []
    boundary_events: list[str] = []
    original_seal = SQLiteInferenceEvidenceLedger.verify_completion_seal
    original_commit = SQLitePrimaryCompletionRepository.commit_prepared
    captured_prepared: list[PreparedPrimaryCompletionCommand] = []

    def recording_seal(ledger: SQLiteInferenceEvidenceLedger) -> None:
        sealed_paths.append(ledger.database_path)
        boundary_events.append("seal")
        original_seal(ledger)

    def unexpected_full_audit(ledger: SQLiteInferenceEvidenceLedger) -> None:
        full_audit_paths.append(ledger.database_path)
        raise AssertionError("normal precommit completion must not rescan the evidence ledger")

    def recording_commit(repository, prepared):  # type: ignore[no-untyped-def]
        boundary_events.append("commit")
        captured_prepared.append(prepared)
        return original_commit(repository, prepared)

    def unexpected_strict_validation(
        _repository: SQLitePrimaryCompletionRepository,
        _command: object,
    ) -> None:
        raise AssertionError("prepared completion must not revalidate the full command")

    monkeypatch.setattr(SQLiteInferenceEvidenceLedger, "verify_completion_seal", recording_seal)
    monkeypatch.setattr(SQLiteInferenceEvidenceLedger, "verify_integrity", unexpected_full_audit)
    monkeypatch.setattr(SQLitePrimaryCompletionRepository, "commit_prepared", recording_commit)
    monkeypatch.setattr(
        SQLitePrimaryCompletionRepository,
        "_validate_command",
        unexpected_strict_validation,
    )

    receipt = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="pre-completion-evidence-seal",
    )

    _assert_local_conformance(receipt)
    assert receipt.replayed is False
    assert sealed_paths == [(state_dir / "inference-evidence.sqlite3").resolve()]
    assert full_audit_paths == []
    assert boundary_events == ["seal", "commit"]
    assert len(captured_prepared) == 1
    prepared = captured_prepared[0]
    assert prepared.is_canonical_preparation is True
    command = prepared.command
    assert prepared.detail_bytes == canonical_json_bytes(command.detail)
    assert prepared.command_bytes == canonical_json_bytes(command)
    assert prepared.processing_run_bytes == canonical_json_bytes(command.detail.processing_run)
    with sqlite3.connect(state_dir / "primary-completion.sqlite3") as connection:
        detail_row = connection.execute(
            "SELECT payload_json, exact_bytes_sha256 FROM detailed_results"
        ).fetchone()
        completion_row = connection.execute(
            "SELECT command_json, command_json_sha256 FROM primary_completions"
        ).fetchone()
        run_row = connection.execute(
            "SELECT run_json, run_json_sha256 FROM primary_runs WHERE run_id = ?",
            (command.detail.run_id,),
        ).fetchone()
    assert detail_row == (
        prepared.detail_bytes,
        prepared.detail_exact_bytes_sha256,
    )
    assert completion_row == (
        prepared.command_bytes,
        prepared.command_exact_bytes_sha256,
    )
    assert run_row == (
        prepared.processing_run_bytes,
        prepared.processing_run_exact_bytes_sha256,
    )

    tampered_detail = prepared.detail_bytes + b" "
    with pytest.raises(ValueError, match="detail bytes"):
        replace(
            prepared,
            detail_bytes=tampered_detail,
            detail_exact_bytes_sha256=exact_bytes_sha256(tampered_detail),
        )

    tampered_command = prepared.command_bytes + b" "
    with pytest.raises(ValueError, match="command bytes"):
        replace(
            prepared,
            command_bytes=tampered_command,
            command_exact_bytes_sha256=exact_bytes_sha256(tampered_command),
        )

    tampered_processing_run = prepared.processing_run_bytes + b" "
    with pytest.raises(ValueError, match="processing-run bytes"):
        replace(
            prepared,
            processing_run_bytes=tampered_processing_run,
            processing_run_exact_bytes_sha256=exact_bytes_sha256(tampered_processing_run),
        )


def test_failed_precommit_evidence_seal_publishes_no_primary_fact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"

    def fail_seal(_ledger: SQLiteInferenceEvidenceLedger) -> None:
        raise SQLiteInferenceEvidenceLedgerError("injected completion seal failure")

    monkeypatch.setattr(SQLiteInferenceEvidenceLedger, "verify_completion_seal", fail_seal)

    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_fixture(
            source_path=SOURCE_FIXTURE,
            state_dir=state_dir,
            run_key="failed-precommit-evidence-seal",
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    with sqlite3.connect(state_dir / "primary-completion.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM primary_completions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM primary_outbox").fetchone()[0] == 0


def test_external_inference_evidence_data_version_change_fails_seal_before_primary_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"
    sealed_paths: list[Path] = []
    original_seal = SQLiteInferenceEvidenceLedger.verify_completion_seal
    original_prepare = local_composition_module.prepare_initial_action_event_publications

    def externally_advance_data_version(*args, **kwargs):  # type: ignore[no-untyped-def]
        publications = original_prepare(*args, **kwargs)
        evidence_database = state_dir / "inference-evidence.sqlite3"
        with sqlite3.connect(evidence_database) as connection:
            # Write the same valid schema version through a second connection. SQLite
            # still advances the first connection's data_version without changing the
            # ledger's contract, so the completion seal must reject its stale cache.
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            connection.execute(f"PRAGMA user_version = {int(schema_version)}")
            connection.commit()
        return publications

    def recording_seal(ledger: SQLiteInferenceEvidenceLedger) -> None:
        sealed_paths.append(ledger.database_path)
        original_seal(ledger)

    monkeypatch.setattr(
        local_composition_module,
        "prepare_initial_action_event_publications",
        externally_advance_data_version,
    )
    monkeypatch.setattr(SQLiteInferenceEvidenceLedger, "verify_completion_seal", recording_seal)

    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_fixture(
            source_path=SOURCE_FIXTURE,
            state_dir=state_dir,
            run_key="stale-inference-evidence-seal",
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    assert sealed_paths == [(state_dir / "inference-evidence.sqlite3").resolve()]
    with sqlite3.connect(state_dir / "primary-completion.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM primary_completions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM primary_outbox").fetchone()[0] == 0


def test_parallel_local_commands_share_bounded_provider_and_replay_independently(
    tmp_path: Path,
) -> None:
    """Four state-affine recordings can share one bounded provider dispatcher."""

    jobs: list[CanonicalLocalFixtureJob] = []
    source_document = json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    for ordinal in range(4):
        # The fixture's source hash defines recording identity.  Give each
        # concurrent job distinct immutable source bytes, rather than treating
        # four copies of one recording as independent production identities.
        document = dict(source_document)
        document["source_clock_id"] = f"parallel-fixture-clock-{ordinal}"
        source = tmp_path / f"source-{ordinal}.json"
        source.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        jobs.append(
            CanonicalLocalFixtureJob(
                source_path=source,
                state_dir=tmp_path / f"state-{ordinal}",
                run_key="parallel-local-command",
            )
        )

    with CanonicalLocalRecordingService(
        recording_worker_count=4,
        ingress_queue_capacity=4,
        provider_concurrency=1,
        provider_queue_capacity=1,
    ) as service:
        fresh = service.run_fixtures(jobs)
        snapshot_after_fresh = service.snapshot
        replay = service.run_fixtures(jobs)
        snapshot_after_replay = service.snapshot

    assert len(fresh) == len(replay) == 4
    assert all(receipt.ok and not receipt.replayed for receipt in fresh)
    assert all(receipt.ok and receipt.replayed for receipt in replay)
    assert len({receipt.recording_identity for receipt in fresh}) == 4
    assert len({receipt.run_id for receipt in fresh}) == 4
    assert tuple(receipt.run_id for receipt in replay) == tuple(receipt.run_id for receipt in fresh)
    assert tuple(receipt.recording_identity for receipt in replay) == tuple(
        receipt.recording_identity for receipt in fresh
    )
    assert all(receipt.fixture_inference_calls > 0 for receipt in fresh)
    assert all(receipt.fixture_inference_calls == 0 for receipt in replay)

    assert snapshot_after_fresh.completed == 4
    assert snapshot_after_fresh.fresh_receipts == 4
    assert snapshot_after_fresh.replayed_receipts == 0
    assert snapshot_after_fresh.max_active == 4
    assert snapshot_after_fresh.claimed_state_dir_count == 0
    assert snapshot_after_fresh.unique_state_dir_count == 4
    assert snapshot_after_fresh.state_dir_claim_conflicts == 0
    assert snapshot_after_fresh.provider_queue.completed > 0
    assert snapshot_after_fresh.provider_queue.max_queue_depth <= 1

    assert snapshot_after_replay.completed == 8
    assert snapshot_after_replay.fresh_receipts == 4
    assert snapshot_after_replay.replayed_receipts == 4
    assert snapshot_after_replay.provider_queue.completed == (
        snapshot_after_fresh.provider_queue.completed
    )
