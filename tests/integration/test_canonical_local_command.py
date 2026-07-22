from __future__ import annotations

import sqlite3
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
from robata.queue.models import WorkAttemptOutcome, WorkItemState
from robata.review.routing import ReviewRoutingDisposition

SOURCE_FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical" / "source-recording.json"


def _assert_local_conformance(receipt: CanonicalLocalRunReceipt) -> None:
    assert receipt.schema_version == "1.0"
    assert receipt.ok is True
    assert receipt.status == "SUCCEEDED"
    assert receipt.network_call_count == 0
    assert receipt.evidence_class == "LOCAL_CONFORMANCE"
    assert receipt.production_eligible is False


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


def test_local_command_audits_inference_evidence_before_primary_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"
    audited_paths: list[Path] = []
    boundary_events: list[str] = []
    original_verify = SQLiteInferenceEvidenceLedger.verify_integrity
    original_commit = SQLitePrimaryCompletionRepository.commit

    def recording_verify(ledger: SQLiteInferenceEvidenceLedger) -> None:
        audited_paths.append(ledger.database_path)
        boundary_events.append("audit")
        original_verify(ledger)

    def recording_commit(repository, command):  # type: ignore[no-untyped-def]
        boundary_events.append("commit")
        return original_commit(repository, command)

    monkeypatch.setattr(SQLiteInferenceEvidenceLedger, "verify_integrity", recording_verify)
    monkeypatch.setattr(SQLitePrimaryCompletionRepository, "commit", recording_commit)

    receipt = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="pre-completion-evidence-audit",
    )

    _assert_local_conformance(receipt)
    assert receipt.replayed is False
    assert audited_paths == [(state_dir / "inference-evidence.sqlite3").resolve()]
    assert boundary_events == ["audit", "commit"]


def test_failed_precommit_evidence_audit_publishes_no_primary_fact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "canonical-state"

    def fail_audit(_ledger: SQLiteInferenceEvidenceLedger) -> None:
        raise SQLiteInferenceEvidenceLedgerError("injected integrity failure")

    monkeypatch.setattr(SQLiteInferenceEvidenceLedger, "verify_integrity", fail_audit)

    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_fixture(
            source_path=SOURCE_FIXTURE,
            state_dir=state_dir,
            run_key="failed-precommit-evidence-audit",
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    with sqlite3.connect(state_dir / "primary-completion.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM primary_completions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM primary_outbox").fetchone()[0] == 0
