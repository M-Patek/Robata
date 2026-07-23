"""Durability and conflict tests for the local SQLite barrier authority."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier as ThreadBarrier
from uuid import UUID

import pytest

import robata.inference.call_barrier as call_barrier_module
from robata.adapters.sqlite_barrier import (
    SQLiteBarrierStorage,
    SQLiteBarrierStorageError,
)
from robata.contracts.hashing import semantic_sha256
from robata.inference.call_barrier import (
    InferenceCallBarrierConflictError,
    InferenceCallBarrierDefinition,
    InferenceCallPartCompletion,
    InferenceCallReduction,
)
from robata.inference.models import (
    InferenceStatus,
    inference_attempt_selection_logical_key,
)
from robata.queue.barrier import Barrier, BarrierCoordinator, ReductionPolicy
from robata.queue.stage import StageStatus
from robata.runtime.observability import RuntimeProfileRecorder

NOW = "2026-07-20T12:00:00Z"


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def test_observes_exact_transaction_boundaries_and_actual_rollback(
    tmp_path: Path,
) -> None:
    recorder = RuntimeProfileRecorder()
    storage = SQLiteBarrierStorage(
        tmp_path / "barriers.sqlite3",
        runtime_observer=recorder,
    )
    barrier = Barrier(
        barrier_id=_uuid(900),
        logical_key="observed-barrier",
        expected_member_count=1,
        empty_semantics=StageStatus.SKIPPED_NOT_NEEDED.value,
        reduction_policy="reduce-1",
        status="OPEN",
        required_success_count=1,
        max_degraded_failures=0,
    )
    storage.save_barrier(barrier)

    conflicting = barrier.model_copy(update={"logical_key": "conflicting-barrier"})
    with pytest.raises(ValueError, match="different definition"):
        storage.save_barrier(conflicting)

    snapshot = recorder.snapshot()
    transactions = tuple(
        counter for counter in snapshot.counters if counter.name == "sqlite.barrier.transactions"
    )
    commits = sum(
        counter.value for counter in snapshot.counters if counter.name == "sqlite.barrier.commits"
    )
    rollbacks = sum(
        counter.value for counter in snapshot.counters if counter.name == "sqlite.barrier.rollbacks"
    )
    writes = {
        attribute.value
        for counter in transactions
        for attribute in counter.attributes
        if attribute.name == "write"
    }

    assert sum(counter.value for counter in transactions) == 4
    assert commits == 3
    assert rollbacks == 1
    assert writes == {False, True}
    assert sum(span.name == "sqlite.barrier.transaction" for span in snapshot.spans) == 4


def _declare(storage: SQLiteBarrierStorage) -> InferenceCallBarrierDefinition:
    logical_key = f"inference-input-barrier:{_digest(10)}"
    barrier = BarrierCoordinator(storage).create_barrier(
        logical_key,
        2,
        ReductionPolicy(version="reduce-1", required_count=2),
    )
    definition = InferenceCallBarrierDefinition(
        barrier_id=barrier.barrier_id,
        barrier_semantic_sha256=_digest(11),
        barrier_logical_key=logical_key,
        input_plan_semantic_sha256=_digest(12),
        call_plan_sha256=_digest(13),
        part_count=2,
        expected_part_semantic_sha256s=(_digest(20), _digest(21)),
        expected_part_logical_keys=("call-part:0", "call-part:1"),
        expected_part_idempotency_keys=("idempotency:0", "idempotency:1"),
        reduction_policy="ordered-concat",
        reduction_policy_version="reduce-1",
        created_at=NOW,
    )
    return storage.append_definition(definition)


def _completion(
    definition: InferenceCallBarrierDefinition,
    ordinal: int,
    *,
    identity_offset: int = 0,
) -> InferenceCallPartCompletion:
    inference_id = _uuid(100 + ordinal + identity_offset)
    logical_invocation_id = _uuid(200 + ordinal + identity_offset)
    selection_id = _uuid(300 + ordinal + identity_offset)
    selection_policy_version = "selection-1"
    selection_key = inference_attempt_selection_logical_key(
        logical_invocation_id=logical_invocation_id,
        policy_version=selection_policy_version,
    )
    normalized_output = {"label": f"label-{ordinal}"}
    output_digest = semantic_sha256(normalized_output)
    completion_digest = call_barrier_module._completion_semantic_sha256(
        barrier_semantic_sha256=definition.barrier_semantic_sha256,
        input_plan_semantic_sha256=definition.input_plan_semantic_sha256,
        call_plan_sha256=definition.call_plan_sha256,
        part_semantic_sha256=definition.expected_part_semantic_sha256s[ordinal],
        part_idempotency_key=definition.expected_part_idempotency_keys[ordinal],
        inference_id=inference_id,
        logical_invocation_id=logical_invocation_id,
        selection_id=selection_id,
        selection_policy_version=selection_policy_version,
        selection_decision_logical_key=selection_key,
        attempt=1,
        status=InferenceStatus.SUCCEEDED,
        normalized_output_sha256=output_digest,
        raw_output_artifact_id=f"raw:{ordinal}:{identity_offset}",
        failure=None,
    )
    return InferenceCallPartCompletion(
        completion_id=call_barrier_module._stable_uuid(
            "inference-call-completion",
            completion_digest,
        ),
        completion_semantic_sha256=completion_digest,
        barrier_id=definition.barrier_id,
        barrier_semantic_sha256=definition.barrier_semantic_sha256,
        input_plan_semantic_sha256=definition.input_plan_semantic_sha256,
        call_plan_sha256=definition.call_plan_sha256,
        part_ordinal=ordinal,
        part_count=definition.part_count,
        part_semantic_sha256=definition.expected_part_semantic_sha256s[ordinal],
        part_logical_key=definition.expected_part_logical_keys[ordinal],
        part_idempotency_key=definition.expected_part_idempotency_keys[ordinal],
        inference_id=inference_id,
        logical_invocation_id=logical_invocation_id,
        selection_id=selection_id,
        selection_policy_version=selection_policy_version,
        selection_decision_logical_key=selection_key,
        attempt=1,
        status=InferenceStatus.SUCCEEDED,
        normalized_output=normalized_output,
        normalized_output_sha256=output_digest,
        raw_output_artifact_id=f"raw:{ordinal}:{identity_offset}",
        failure=None,
        completed_at=NOW,
    )


def _reduction(
    definition: InferenceCallBarrierDefinition,
    completions: tuple[InferenceCallPartCompletion, ...],
) -> InferenceCallReduction:
    normalized_output = {
        "label": "|".join(str(item.normalized_output["label"]) for item in completions)
    }
    output_digest = semantic_sha256(normalized_output)
    part_digests = tuple(item.part_semantic_sha256 for item in completions)
    output_digests = tuple(
        item.normalized_output_sha256
        for item in completions
        if item.normalized_output_sha256 is not None
    )
    selection_keys = tuple(
        item.selection_decision_logical_key
        for item in completions
        if item.selection_decision_logical_key is not None
    )
    reduction_digest = call_barrier_module._reduction_semantic_sha256(
        barrier_semantic_sha256=definition.barrier_semantic_sha256,
        input_plan_semantic_sha256=definition.input_plan_semantic_sha256,
        call_plan_sha256=definition.call_plan_sha256,
        reduction_policy=definition.reduction_policy,
        reduction_policy_version=definition.reduction_policy_version,
        output_schema_sha256=_digest(40),
        ordered_part_semantic_sha256s=part_digests,
        ordered_normalized_output_sha256s=output_digests,
        ordered_selection_decision_logical_keys=selection_keys,
        normalized_output_sha256=output_digest,
    )
    return InferenceCallReduction(
        reduction_id=call_barrier_module._stable_uuid(
            "inference-call-reduction",
            reduction_digest,
        ),
        reduction_semantic_sha256=reduction_digest,
        barrier_id=definition.barrier_id,
        barrier_semantic_sha256=definition.barrier_semantic_sha256,
        input_plan_semantic_sha256=definition.input_plan_semantic_sha256,
        call_plan_sha256=definition.call_plan_sha256,
        reduction_policy=definition.reduction_policy,
        reduction_policy_version=definition.reduction_policy_version,
        output_schema_sha256=_digest(40),
        ordered_completion_ids=tuple(item.completion_id for item in completions),
        ordered_part_semantic_sha256s=part_digests,
        ordered_normalized_output_sha256s=output_digests,
        ordered_selection_decision_logical_keys=selection_keys,
        normalized_output=normalized_output,
        normalized_output_sha256=output_digest,
        reduced_at=NOW,
    )


def test_barrier_facts_reopen_and_exactly_replay(tmp_path: Path) -> None:
    database = tmp_path / "barriers.sqlite3"
    storage = SQLiteBarrierStorage(database)
    definition = _declare(storage)
    coordinator = BarrierCoordinator(storage)
    completions = tuple(_completion(definition, ordinal) for ordinal in range(2))

    for completion in reversed(completions):
        assert storage.append_completion(completion) == completion
        coordinator.submit_member(
            definition.barrier_id,
            completion.part_logical_key,
            StageStatus.SUCCEEDED,
        )
    reduction = _reduction(definition, completions)
    assert storage.append_reduction(reduction) == reduction

    reopened = SQLiteBarrierStorage(database)
    assert reopened.get_definition(definition.barrier_id) == definition
    assert reopened.list_completions(definition.barrier_id) == completions
    assert reopened.get_state(definition.barrier_id).status == "CLOSED"  # type: ignore[union-attr]
    assert reopened.get_reduction(definition.barrier_id) == reduction
    assert reopened.append_definition(definition) == definition
    for completion in completions:
        assert reopened.append_completion(completion) == completion
        BarrierCoordinator(reopened).submit_member(
            definition.barrier_id,
            completion.part_logical_key,
            StageStatus.SUCCEEDED,
        )
    assert reopened.append_reduction(reduction) == reduction

    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT count(*) FROM inference_call_part_completions").fetchone()[0]
            == 2
        )
        assert connection.execute("SELECT count(*) FROM barrier_members").fetchone()[0] == 2
        assert (
            connection.execute("SELECT count(*) FROM inference_call_reductions").fetchone()[0] == 1
        )


def test_replay_repairs_crash_between_completion_and_member(tmp_path: Path) -> None:
    database = tmp_path / "barriers.sqlite3"
    first = SQLiteBarrierStorage(database)
    definition = _declare(first)
    completion = _completion(definition, 0)

    first.append_completion(completion)
    assert first.get_state(definition.barrier_id).completed_members == 0  # type: ignore[union-attr]

    reopened = SQLiteBarrierStorage(database)
    assert reopened.append_completion(completion) == completion
    state = BarrierCoordinator(reopened).submit_member(
        definition.barrier_id,
        completion.part_logical_key,
        StageStatus.SUCCEEDED,
    )
    replayed = BarrierCoordinator(reopened).submit_member(
        definition.barrier_id,
        completion.part_logical_key,
        StageStatus.SUCCEEDED,
    )

    assert state == replayed
    assert state.completed_members == 1
    assert state.pending_members == 1
    assert len(reopened.list_completions(definition.barrier_id)) == 1
    assert len(reopened.get_members(definition.barrier_id)) == 1


def test_concurrent_conflicting_completions_have_one_durable_winner(
    tmp_path: Path,
) -> None:
    database = tmp_path / "barriers.sqlite3"
    first = SQLiteBarrierStorage(database)
    definition = _declare(first)
    second = SQLiteBarrierStorage(database)
    candidates = (
        _completion(definition, 0, identity_offset=0),
        _completion(definition, 0, identity_offset=1_000),
    )
    start = ThreadBarrier(2)

    def append(item: tuple[SQLiteBarrierStorage, InferenceCallPartCompletion]) -> object:
        storage, candidate = item
        start.wait(timeout=5)
        try:
            return storage.append_completion(candidate)
        except InferenceCallBarrierConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(append, zip((first, second), candidates, strict=True)))

    assert sum(isinstance(item, InferenceCallPartCompletion) for item in results) == 1
    assert sum(isinstance(item, InferenceCallBarrierConflictError) for item in results) == 1
    persisted = first.list_completions(definition.barrier_id)
    assert len(persisted) == 1
    assert persisted[0] in candidates


def test_tampered_state_row_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "barriers.sqlite3"
    storage = SQLiteBarrierStorage(database)
    barrier = BarrierCoordinator(storage).create_barrier(
        "tamper-state",
        1,
        ReductionPolicy(version="reduce-1", required_count=1),
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE barrier_states SET completed_members = 1
            WHERE barrier_id = ?
            """,
            (barrier.barrier_id,),
        )
        connection.commit()

    with pytest.raises(SQLiteBarrierStorageError, match="indexed column completed_members"):
        storage.get_state(barrier.barrier_id)
