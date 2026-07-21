from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from robata.application.canonical import local_composition as local_composition_module
from robata.application.canonical.local_composition import (
    CanonicalLocalCompositionError,
    CanonicalLocalCompositionErrorCode,
    CanonicalLocalRunReceipt,
    run_local_canonical_fixture,
)

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
    first_barrier = state_dir / "runs" / first.run_id / "inference-call-barrier.sqlite3"
    second_barrier = state_dir / "runs" / second.run_id / "inference-call-barrier.sqlite3"
    assert first_barrier.is_file()
    assert second_barrier.is_file()
    assert first_barrier != second_barrier


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
