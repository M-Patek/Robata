from __future__ import annotations

from uuid import UUID

import pytest

from robata.benchmark.active_learning import (
    ActiveLearningHeldOutObservation,
    ActiveLearningPoolObservation,
    ActiveLearningSplit,
    ActiveLearningSplitProtocol,
    ActiveLearningSplitRecord,
    build_active_learning_fixture_metrics,
)
from robata.review.active_learning import (
    ActiveLearningCandidate,
    ActiveLearningPoolSnapshot,
    ActiveLearningSelectionPolicy,
    ActiveLearningSelector,
    ActiveLearningSourceReference,
    ActiveLearningTermApplicability,
    ActiveLearningTermEvidence,
    ActiveLearningTermKind,
    ExistingReviewPriorityEvidence,
)
from robata.review.models import ReviewTrigger


def _digest(value: int) -> str:
    return f"{value:064x}"


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _candidate(value: int, recording_identity: str) -> ActiveLearningCandidate:
    return ActiveLearningCandidate.create(
        review_task_id=_uuid(value),
        review_task_semantic_sha256=_digest(value + 10),
        review_task_exact_sha256=_digest(value + 20),
        recording_identity=recording_identity,
        priority_evidence=ExistingReviewPriorityEvidence(
            trigger=ReviewTrigger.REVIEW_SAMPLING,
            priority=value,
            reason_codes=("EXISTING_PRIORITY",),
        ),
        terms=tuple(
            ActiveLearningTermEvidence(
                kind=kind,
                applicability=ActiveLearningTermApplicability.APPLICABLE,
                value_millionths=900_000 - ordinal,
                source=ActiveLearningSourceReference(
                    logical_key=f"term-source:{_digest(value * 100 + ordinal)}",
                    semantic_sha256=_digest(value * 100 + ordinal),
                    exact_sha256=_digest(value * 100 + ordinal + 1_000),
                ),
                reason_codes=(f"{kind.value}_OBSERVED",),
            )
            for ordinal, kind in enumerate(ActiveLearningTermKind)
        ),
    )


def _protocol() -> ActiveLearningSplitProtocol:
    records = (
        ActiveLearningSplitRecord(
            mcap_id=_uuid(1),
            recording_identity=_digest(1),
            capture_group="capture-a",
            camera_group="camera-a",
            time_group="time-a",
        ),
        ActiveLearningSplitRecord(
            mcap_id=_uuid(2),
            recording_identity=_digest(2),
            capture_group="capture-b",
            camera_group="camera-b",
            time_group="time-b",
        ),
        ActiveLearningSplitRecord(
            mcap_id=_uuid(3),
            recording_identity=_digest(3),
            capture_group="capture-c",
            camera_group="camera-c",
            time_group="time-c",
        ),
    )
    return ActiveLearningSplitProtocol.create(
        split_version="active-learning-split-v1",
        records=records,
        assignments={
            _uuid(1): ActiveLearningSplit.DEVELOPMENT,
            _uuid(2): ActiveLearningSplit.CALIBRATION,
            _uuid(3): ActiveLearningSplit.FROZEN_TEST,
        },
    )


def _decision():
    first = _candidate(11, _digest(1))
    second = _candidate(12, _digest(2))
    pool = ActiveLearningPoolSnapshot.create(
        pool_version="active-learning-pool-v1",
        candidates=(first, second),
    )
    policy = ActiveLearningSelectionPolicy.create(
        policy_version="active-learning-policy-v1",
        eligible_triggers=(ReviewTrigger.REVIEW_SAMPLING,),
        ranking_terms=tuple(ActiveLearningTermKind),
    )
    return ActiveLearningSelector().select(pool=pool, policy=policy, budget=1)


def _pool_observations(decision):
    mcap_by_recording = {_digest(1): _uuid(1), _digest(2): _uuid(2)}
    return tuple(
        ActiveLearningPoolObservation(
            review_task_id=item.candidate.review_task_id,
            review_task_semantic_sha256=item.candidate.review_task_semantic_sha256,
            mcap_id=mcap_by_recording[item.candidate.recording_identity],
            recording_identity=item.candidate.recording_identity,
            subgroup=(
                "hand-left" if item.candidate.recording_identity == _digest(1) else "hand-right"
            ),
            annotation_arrived=item.candidate.review_task_id in decision.selected_review_task_ids,
            agreement_millionths=(
                900_000
                if item.candidate.review_task_id in decision.selected_review_task_ids
                else None
            ),
            yield_positive=(
                True if item.candidate.review_task_id in decision.selected_review_task_ids else None
            ),
        )
        for item in decision.candidate_decisions
    )


def test_fixture_metrics_keep_selection_and_frozen_held_out_populations_separate() -> None:
    decision = _decision()
    metrics = build_active_learning_fixture_metrics(
        decision=decision,
        split_protocol=_protocol(),
        pool_observations=_pool_observations(decision),
        held_out_observations=(
            ActiveLearningHeldOutObservation(
                mcap_id=_uuid(3),
                subgroup="hand-left",
                agreement_millionths=800_000,
                yield_positive=False,
            ),
        ),
    )

    assert metrics.representative_measurement_status == "NOT_MEASURED"
    assert metrics.production_eligible is False
    assert metrics.pool_count == 2
    assert metrics.selected_count == 1
    assert metrics.annotation_count == 1
    assert metrics.frozen_held_out_count == 1
    assert metrics.frozen_held_out_yield_millionths == 0
    assert len(metrics.subgroup_metrics) == 2
    assert metrics.logical_key.endswith(metrics.semantic_sha256)


def test_split_rejects_connected_capture_camera_or_time_groups_crossing_roles() -> None:
    first = ActiveLearningSplitRecord(
        mcap_id=_uuid(21),
        recording_identity=_digest(21),
        capture_group="shared-capture",
        camera_group="camera-a",
        time_group="time-a",
    )
    second = ActiveLearningSplitRecord(
        mcap_id=_uuid(22),
        recording_identity=_digest(22),
        capture_group="shared-capture",
        camera_group="camera-b",
        time_group="time-b",
    )

    with pytest.raises(ValueError, match="connected groups"):
        ActiveLearningSplitProtocol.create(
            split_version="active-learning-split-v1",
            records=(first, second),
            assignments={
                first.mcap_id: ActiveLearningSplit.DEVELOPMENT,
                second.mcap_id: ActiveLearningSplit.FROZEN_TEST,
            },
        )


def test_frozen_test_record_cannot_enter_the_selection_pool() -> None:
    decision = _decision()
    pool_observations = list(_pool_observations(decision))
    pool_observations[0] = pool_observations[0].model_copy(update={"mcap_id": _uuid(3)})

    with pytest.raises(ValueError, match="frozen-test records"):
        build_active_learning_fixture_metrics(
            decision=decision,
            split_protocol=_protocol(),
            pool_observations=tuple(pool_observations),
            held_out_observations=(),
        )
