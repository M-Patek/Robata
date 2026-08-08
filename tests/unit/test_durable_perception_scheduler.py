from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from robata.application.canonical.mage_stream import (
    AbsoluteNanosecondInterval,
    MageStreamPolicy,
    MageStreamRecording,
    plan_mage_stream,
)
from robata.perception.durable_scheduler import (
    DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
    DurablePerceptionWorkFenceError,
    DurablePerceptionWorkState,
    DurablePerceptionWorkStateError,
    SQLitePerceptionWorkScheduler,
)
from robata.perception.pipeline import PerceptionStage

_BASE = datetime(2026, 8, 8, tzinfo=UTC)
_DIGEST = "a" * 64


def _plan(duration_ns: int = 16_000_000_000):
    return plan_mage_stream(
        recording=MageStreamRecording(
            recording_key="durable-perception-recording",
            recording_exact_sha256="b" * 64,
            interval=AbsoluteNanosecondInterval(0, duration_ns),
        ),
        policy=MageStreamPolicy(
            scan_segment_duration_ns=8_000_000_000,
            reasoning_horizon_duration_ns=8_000_000_000,
        ),
    )


def _work(
    scheduler: SQLitePerceptionWorkScheduler, run_key: str, ordinal: int, stage: PerceptionStage
):
    return next(item for item in scheduler.context_work(run_key, ordinal) if item.stage is stage)


def _complete(
    scheduler: SQLitePerceptionWorkScheduler,
    work_item_id: str,
    *,
    now: datetime,
    marker: str,
):
    claim = scheduler.claim_and_start("durable-worker", 30, work_item_id=work_item_id, now=now)
    assert claim is not None
    return scheduler.succeed(
        claim.lease,
        result_reference=f"artifact://{marker}",
        result_sha256=_DIGEST,
        now=now + timedelta(seconds=1),
    )


def test_registers_non_overlapping_context_work_with_one_observation_per_segment(
    tmp_path: Path,
) -> None:
    scheduler = SQLitePerceptionWorkScheduler(tmp_path / "perception.sqlite3")
    run = scheduler.register_plan(_plan(), codec_policy_version="mage-native-codec-v2", now=_BASE)
    snapshot = scheduler.snapshot(run.run_key)

    assert run.scheduler_policy_version == DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION
    assert [item.focus_segment_ordinal for item in snapshot.contexts] == [0, 1]
    assert [(item.interval_start_ns, item.interval_end_ns) for item in snapshot.contexts] == [
        (0, 8_000_000_000),
        (8_000_000_000, 16_000_000_000),
    ]
    counts = {item.stage: item for item in snapshot.stage_counts}
    assert counts[PerceptionStage.MEDIA_SCAN].planned == 2
    assert counts[PerceptionStage.PERCEPTION_OBSERVE].planned == 2
    assert counts[PerceptionStage.OBSERVATION_PROJECT].planned == 2
    assert counts[PerceptionStage.TEMPORAL_RECONCILE].planned == 2
    assert counts[PerceptionStage.FINALIZE].planned == 1
    assert counts[PerceptionStage.FUSION].planned == 0
    assert counts[PerceptionStage.PERCEPTION_REFINE].planned == 0
    assert snapshot.normal_observation_work_count == 2
    assert snapshot.refinement_work_count == 0

    first = {item.stage: item for item in scheduler.context_work(run.run_key, 0)}
    second = {item.stage: item for item in scheduler.context_work(run.run_key, 1)}
    assert scheduler.dependencies(first[PerceptionStage.PERCEPTION_OBSERVE].work_item_id) == (
        first[PerceptionStage.MEDIA_SCAN].work_item_id,
    )
    assert set(scheduler.dependencies(second[PerceptionStage.TEMPORAL_RECONCILE].work_item_id)) == {
        second[PerceptionStage.OBSERVATION_PROJECT].work_item_id,
        first[PerceptionStage.TEMPORAL_RECONCILE].work_item_id,
    }
    assert all("QWEN" not in item.stage.value for item in scheduler.items_for_run(run.run_key))


def test_registration_is_exact_replay_safe_and_scopes_alternate_codec_configuration_to_new_run(
    tmp_path: Path,
) -> None:
    scheduler = SQLitePerceptionWorkScheduler(tmp_path / "perception.sqlite3")
    plan = _plan()
    first = scheduler.register_plan(plan, codec_policy_version="mage-native-codec-v2", now=_BASE)
    replay = scheduler.register_plan(plan, codec_policy_version="mage-native-codec-v2", now=_BASE)

    assert replay == first
    assert len(scheduler.items_for_run(first.run_key)) == 9
    alternate = scheduler.register_plan(
        plan, codec_policy_version="mage-native-codec-v3", now=_BASE
    )
    assert alternate.run_key != first.run_key
    assert alternate.plan_key == first.plan_key
    assert alternate.config_sha256 != first.config_sha256


def test_expired_lease_recovers_with_new_fence_and_rejects_stale_terminal_acceptance(
    tmp_path: Path,
) -> None:
    database = tmp_path / "perception.sqlite3"
    scheduler = SQLitePerceptionWorkScheduler(database)
    run = scheduler.register_plan(_plan(8_000_000_000), codec_policy_version="codec-v2", now=_BASE)
    media = _work(scheduler, run.run_key, 0, PerceptionStage.MEDIA_SCAN)
    first = scheduler.claim_and_start("first-worker", 5, work_item_id=media.work_item_id, now=_BASE)
    assert first is not None
    assert first.item.state is DurablePerceptionWorkState.RUNNING

    restarted = SQLitePerceptionWorkScheduler(database)
    assert restarted.reconcile(now=_BASE + timedelta(seconds=6)) == 1
    assert restarted.get(media.work_item_id).state is DurablePerceptionWorkState.READY
    second = restarted.claim_and_start(
        "second-worker", 30, work_item_id=media.work_item_id, now=_BASE + timedelta(seconds=7)
    )
    assert second is not None
    assert second.lease.lease_epoch == first.lease.lease_epoch + 1
    with pytest.raises(DurablePerceptionWorkFenceError):
        restarted.succeed(
            first.lease,
            result_reference="artifact://stale",
            result_sha256=_DIGEST,
            now=_BASE + timedelta(seconds=8),
        )
    accepted = restarted.succeed(
        second.lease,
        result_reference="artifact://media-health",
        result_sha256=_DIGEST,
        now=_BASE + timedelta(seconds=8),
    )
    assert accepted.state is DurablePerceptionWorkState.SUCCEEDED


def test_stage_dependencies_and_terminal_acceptance_form_a_recoverable_minimal_closed_loop(
    tmp_path: Path,
) -> None:
    scheduler = SQLitePerceptionWorkScheduler(tmp_path / "perception.sqlite3")
    run = scheduler.register_plan(_plan(), codec_policy_version="codec-v2", now=_BASE)
    cursor = _BASE
    for ordinal in (0, 1):
        for stage in (
            PerceptionStage.MEDIA_SCAN,
            PerceptionStage.PERCEPTION_OBSERVE,
            PerceptionStage.OBSERVATION_PROJECT,
            PerceptionStage.TEMPORAL_RECONCILE,
        ):
            item = _work(scheduler, run.run_key, ordinal, stage)
            accepted = _complete(
                scheduler, item.work_item_id, now=cursor, marker=f"{ordinal}-{stage.value}"
            )
            assert accepted.state is DurablePerceptionWorkState.SUCCEEDED
            cursor += timedelta(seconds=2)

    finalization = next(
        item
        for item in scheduler.items_for_run(run.run_key)
        if item.stage is PerceptionStage.FINALIZE
    )
    assert scheduler.get(finalization.work_item_id).state is DurablePerceptionWorkState.PLANNED
    assert (
        scheduler.claim("durable-worker", 30, work_item_id=finalization.work_item_id, now=cursor)
        is None
    )

    sealed = scheduler.seal_derived_work(run.run_key, now=cursor)
    assert sealed.derived_work_sealed is True
    assert scheduler.get(finalization.work_item_id).state is DurablePerceptionWorkState.READY
    result = _complete(scheduler, finalization.work_item_id, now=cursor, marker="finalize")
    assert result.state is DurablePerceptionWorkState.SUCCEEDED
    counts = {item.stage: item for item in scheduler.snapshot(run.run_key).stage_counts}
    assert counts[PerceptionStage.PERCEPTION_OBSERVE].succeeded == 2
    assert counts[PerceptionStage.FINALIZE].succeeded == 1


def test_derived_work_closure_binds_refine_to_fusion_and_blocks_finalization(
    tmp_path: Path,
) -> None:
    scheduler = SQLitePerceptionWorkScheduler(tmp_path / "perception.sqlite3")
    run = scheduler.register_plan(_plan(8_000_000_000), codec_policy_version="codec-v2", now=_BASE)
    for index, stage in enumerate(
        (
            PerceptionStage.MEDIA_SCAN,
            PerceptionStage.PERCEPTION_OBSERVE,
            PerceptionStage.OBSERVATION_PROJECT,
            PerceptionStage.TEMPORAL_RECONCILE,
        )
    ):
        item = _work(scheduler, run.run_key, 0, stage)
        _complete(
            scheduler,
            item.work_item_id,
            now=_BASE + timedelta(seconds=index * 2),
            marker=stage.value,
        )

    fusion = scheduler.schedule_derived(
        run_key=run.run_key,
        focus_segment_ordinal=0,
        stage=PerceptionStage.FUSION,
        input_sha256="c" * 64,
        config_sha256="d" * 64,
        now=_BASE + timedelta(seconds=10),
    )
    assert fusion.state is DurablePerceptionWorkState.READY
    refine = scheduler.schedule_derived(
        run_key=run.run_key,
        focus_segment_ordinal=0,
        stage=PerceptionStage.PERCEPTION_REFINE,
        input_sha256="e" * 64,
        config_sha256="d" * 64,
        upstream_work_item_id=fusion.work_item_id,
        now=_BASE + timedelta(seconds=11),
    )
    assert refine.derived_from_work_item_id == fusion.work_item_id
    assert refine.state is DurablePerceptionWorkState.PLANNED
    assert fusion.work_item_id in scheduler.dependencies(refine.work_item_id)

    finalization = next(
        item
        for item in scheduler.items_for_run(run.run_key)
        if item.stage is PerceptionStage.FINALIZE
    )
    assert fusion.work_item_id in scheduler.dependencies(finalization.work_item_id)
    assert refine.work_item_id in scheduler.dependencies(finalization.work_item_id)
    assert scheduler.get(finalization.work_item_id).state is DurablePerceptionWorkState.PLANNED

    with pytest.raises(DurablePerceptionWorkStateError, match="SUCCEEDED"):
        scheduler.seal_derived_work(run.run_key, now=_BASE + timedelta(seconds=12))

    _complete(scheduler, fusion.work_item_id, now=_BASE + timedelta(seconds=13), marker="fusion")
    assert scheduler.get(refine.work_item_id).state is DurablePerceptionWorkState.READY
    with pytest.raises(DurablePerceptionWorkStateError, match="SUCCEEDED"):
        scheduler.seal_derived_work(run.run_key, now=_BASE + timedelta(seconds=15))

    _complete(scheduler, refine.work_item_id, now=_BASE + timedelta(seconds=16), marker="refine")
    sealed = scheduler.seal_derived_work(run.run_key, now=_BASE + timedelta(seconds=18))
    assert sealed.derived_work_sealed is True
    assert scheduler.get(finalization.work_item_id).state is DurablePerceptionWorkState.READY
    replayed = scheduler.register_plan(
        _plan(8_000_000_000),
        codec_policy_version="codec-v2",
        now=_BASE + timedelta(seconds=18),
    )
    assert replayed.run_key == run.run_key
    assert replayed.derived_work_sealed is True

    with pytest.raises(DurablePerceptionWorkStateError, match="sealed"):
        scheduler.schedule_derived(
            run_key=run.run_key,
            focus_segment_ordinal=0,
            stage=PerceptionStage.FUSION,
            input_sha256="f" * 64,
            config_sha256="d" * 64,
            now=_BASE + timedelta(seconds=19),
        )
    with pytest.raises(ValueError, match="upstream FUSION"):
        scheduler.schedule_derived(
            run_key=run.run_key,
            focus_segment_ordinal=0,
            stage=PerceptionStage.PERCEPTION_REFINE,
            input_sha256="f" * 64,
            config_sha256="d" * 64,
            now=_BASE + timedelta(seconds=19),
        )

    result = _complete(
        scheduler, finalization.work_item_id, now=_BASE + timedelta(seconds=20), marker="finalize"
    )
    assert result.state is DurablePerceptionWorkState.SUCCEEDED


def test_default_composition_factory_constructs_vnext_sqlite_authority_only(tmp_path: Path) -> None:
    from robata.application.canonical.perception_composition import (
        LEGACY_QWEN_WINDOW_PROFILE,
        PerceptionCompositionSelectionError,
        create_default_vnext_perception_scheduler,
    )

    scheduler = create_default_vnext_perception_scheduler(tmp_path / "default.sqlite3")
    assert scheduler.scheduler_policy_version == DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION
    with pytest.raises(PerceptionCompositionSelectionError, match="legacy"):
        create_default_vnext_perception_scheduler(
            tmp_path / "legacy.sqlite3",
            profile=LEGACY_QWEN_WINDOW_PROFILE,
        )
