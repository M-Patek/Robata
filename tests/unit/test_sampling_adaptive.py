from __future__ import annotations

import pytest
from pydantic import ValidationError

from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.sampling import (
    ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION,
    AdaptiveResolutionMode,
    CanonicalAdaptiveGridSegment,
    FrozenAdaptiveResolutionRequest,
    FrozenAdaptiveTriggerArtifactRef,
    ResolvedAdaptivePlan,
    ResolvedAdaptiveTarget,
    SamplingGrid,
    adaptive_target_plan_semantic_projection,
    resolve_frozen_adaptive_targets,
)
from robata.sampling.adaptive import (
    AdaptiveCoveragePlanner,
    AdaptiveCoveragePolicy,
    AdaptiveUpgradeReason,
    AdaptiveUpgradeRequest,
    AdaptiveUpgradeTargetRole,
)


def _artifact(
    *,
    artifact_id: str = "10000000-0000-4000-8000-000000000001",
    exact_bytes_sha256: str = "1" * 64,
    semantic_digest: str = "2" * 64,
    schema_id: str = "https://schemas.robata.dev/adaptive-trigger",
    schema_version: str = "1.0.0",
    schema_artifact_id: str = "20000000-0000-4000-8000-000000000001",
    schema_sha256: str = "5" * 64,
) -> FrozenAdaptiveTriggerArtifactRef:
    return FrozenAdaptiveTriggerArtifactRef(
        artifact_id=artifact_id,
        schema_ref=SchemaRef(
            schema_id=schema_id,
            version=schema_version,
            artifact_id=schema_artifact_id,
            sha256=schema_sha256,
        ),
        exact_bytes_sha256=exact_bytes_sha256,
        semantic_sha256=semantic_digest,
    )


def _segment(
    start_ns: int,
    end_ns: int,
    *,
    rate_num: int = 2,
    rate_den: int = 1,
) -> CanonicalAdaptiveGridSegment:
    return CanonicalAdaptiveGridSegment(
        interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        rate_num=rate_num,
        rate_den=rate_den,
    )


def _request(
    *,
    effective_interval: NanosecondInterval,
    resolution_mode: AdaptiveResolutionMode,
    segments: tuple[CanonicalAdaptiveGridSegment, ...] = (),
    explicit_targets_ns: tuple[int, ...] = (),
    max_target_count: int = 10,
    trigger_artifact: FrozenAdaptiveTriggerArtifactRef | None = None,
    grid_origin_ns: int = 0,
) -> FrozenAdaptiveResolutionRequest:
    return FrozenAdaptiveResolutionRequest(
        trigger_artifact=trigger_artifact or _artifact(),
        camera_id=CameraId.CAM_01,
        effective_interval=effective_interval,
        grid_origin_ns=grid_origin_ns,
        resolution_mode=resolution_mode,
        segments=segments,
        explicit_targets_ns=explicit_targets_ns,
        max_target_count=max_target_count,
        resolution_policy_version="adaptive-resolution-policy-v1",
    )


def _forged_plan_values(
    plan: ResolvedAdaptivePlan,
    targets: tuple[ResolvedAdaptiveTarget, ...],
) -> dict[str, object]:
    forged = plan.model_copy(update={"targets": targets})
    values = forged.model_dump()
    values["semantic_sha256"] = semantic_sha256(adaptive_target_plan_semantic_projection(forged))
    return values


def test_grid_segments_are_phase_stable_across_clipping_gaps_and_negative_k() -> None:
    interval = NanosecondInterval(start_ns=-750_000_000, end_ns=1_250_000_000)
    request = _request(
        effective_interval=interval,
        resolution_mode=AdaptiveResolutionMode.GRID_SEGMENTS,
        segments=(
            _segment(-750_000_000, 250_000_000),
            _segment(750_000_000, 1_250_000_000),
        ),
    )

    first = resolve_frozen_adaptive_targets(request)
    replay = resolve_frozen_adaptive_targets(request)

    assert tuple(target.target_ns for target in first.targets) == (
        -500_000_000,
        0,
        1_000_000_000,
    )
    assert tuple((target.segment_ordinal, target.grid_k) for target in first.targets) == (
        (0, -1),
        (0, 0),
        (1, 2),
    )
    assert first == replay
    assert first.model_dump_json() == replay.model_dump_json()
    projection = adaptive_target_plan_semantic_projection(first)
    assert projection["grid_origin_ns"] == "0"
    assert projection["segments"] == [
        {
            "start_ns": "-750000000",
            "end_ns": "250000000",
            "rate_num": "2",
            "rate_den": "1",
        },
        {
            "start_ns": "750000000",
            "end_ns": "1250000000",
            "rate_num": "2",
            "rate_den": "1",
        },
    ]
    assert first.semantic_sha256 == semantic_sha256(projection)


def test_explicit_targets_are_preserved_without_grid_provenance() -> None:
    request = _request(
        effective_interval=NanosecondInterval(start_ns=-100, end_ns=1_000),
        resolution_mode=AdaptiveResolutionMode.EXPLICIT_TARGETS,
        explicit_targets_ns=(-100, 0, 999),
        max_target_count=3,
        grid_origin_ns=37,
    )

    plan = resolve_frozen_adaptive_targets(request)

    assert plan.explicit_targets_ns == (-100, 0, 999)
    assert tuple(target.target_ns for target in plan.targets) == (-100, 0, 999)
    assert all(target.segment_ordinal is None and target.grid_k is None for target in plan.targets)
    projection = adaptive_target_plan_semantic_projection(plan)
    assert projection["trigger_artifact"] == {
        "schema_ref": {
            "schema_id": "https://schemas.robata.dev/adaptive-trigger",
            "version": "1.0.0",
        },
        "semantic_sha256": "2" * 64,
    }
    assert projection["camera_id"] == "cam_01"
    assert projection["effective_interval"] == {"start_ns": "-100", "end_ns": "1000"}
    assert projection["grid_origin_ns"] == "37"
    assert projection["resolution_policy_version"] == "adaptive-resolution-policy-v1"
    assert (
        projection["semantic_projection_version"]
        == ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION
    )
    assert projection["max_target_count"] == 3
    assert projection["explicit_targets_ns"] == ["-100", "0", "999"]


def test_identity_uses_semantic_artifact_and_schema_identity_only() -> None:
    interval = NanosecondInterval(start_ns=0, end_ns=1_000)
    base = _request(
        effective_interval=interval,
        resolution_mode=AdaptiveResolutionMode.EXPLICIT_TARGETS,
        explicit_targets_ns=(0,),
    )
    exact_alias = _request(
        effective_interval=interval,
        resolution_mode=AdaptiveResolutionMode.EXPLICIT_TARGETS,
        explicit_targets_ns=(0,),
        trigger_artifact=_artifact(exact_bytes_sha256="3" * 64),
    )
    schema_exact_alias = _request(
        effective_interval=interval,
        resolution_mode=AdaptiveResolutionMode.EXPLICIT_TARGETS,
        explicit_targets_ns=(0,),
        trigger_artifact=_artifact(
            schema_artifact_id="20000000-0000-4000-8000-000000000002",
            schema_sha256="6" * 64,
        ),
    )
    semantic_change = _request(
        effective_interval=interval,
        resolution_mode=AdaptiveResolutionMode.EXPLICIT_TARGETS,
        explicit_targets_ns=(0,),
        trigger_artifact=_artifact(semantic_digest="4" * 64),
    )
    schema_version_change = _request(
        effective_interval=interval,
        resolution_mode=AdaptiveResolutionMode.EXPLICIT_TARGETS,
        explicit_targets_ns=(0,),
        trigger_artifact=_artifact(schema_version="1.1.0"),
    )
    row_alias = _request(
        effective_interval=interval,
        resolution_mode=AdaptiveResolutionMode.EXPLICIT_TARGETS,
        explicit_targets_ns=(0,),
        trigger_artifact=_artifact(
            artifact_id="10000000-0000-4000-8000-000000000002",
        ),
    )

    base_digest = resolve_frozen_adaptive_targets(base).semantic_sha256
    assert base_digest == resolve_frozen_adaptive_targets(exact_alias).semantic_sha256
    assert base_digest == resolve_frozen_adaptive_targets(schema_exact_alias).semantic_sha256
    assert base_digest == resolve_frozen_adaptive_targets(row_alias).semantic_sha256
    assert base_digest != resolve_frozen_adaptive_targets(semantic_change).semantic_sha256
    assert base_digest != resolve_frozen_adaptive_targets(schema_version_change).semantic_sha256


def test_semantic_projection_version_is_fixed() -> None:
    request = _request(
        effective_interval=NanosecondInterval(start_ns=0, end_ns=1),
        resolution_mode=AdaptiveResolutionMode.EXPLICIT_TARGETS,
        explicit_targets_ns=(0,),
    )
    values = request.model_dump()
    values["semantic_projection_version"] = "caller-selected-version"

    with pytest.raises(ValidationError, match="adaptive-target-plan-semantic-v1"):
        FrozenAdaptiveResolutionRequest.model_validate(values)


def test_non_reduced_segment_rate_is_rejected() -> None:
    with pytest.raises(ValidationError, match="reduced rational"):
        _segment(0, 1_000_000_000, rate_num=2, rate_den=4)


def test_overlapping_or_out_of_order_segments_are_rejected() -> None:
    with pytest.raises(ValidationError, match="ordered and non-overlapping"):
        _request(
            effective_interval=NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
            resolution_mode=AdaptiveResolutionMode.GRID_SEGMENTS,
            segments=(
                _segment(1_000_000_000, 2_000_000_000),
                _segment(0, 1_500_000_000),
            ),
        )


@pytest.mark.parametrize("targets", [(0, 0), (1, 0)])
def test_explicit_targets_must_be_strictly_increasing(targets: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        _request(
            effective_interval=NanosecondInterval(start_ns=0, end_ns=10),
            resolution_mode=AdaptiveResolutionMode.EXPLICIT_TARGETS,
            explicit_targets_ns=targets,
        )


@pytest.mark.parametrize("resolution_mode", list(AdaptiveResolutionMode))
def test_target_budget_is_fail_closed_for_both_input_modes(
    resolution_mode: AdaptiveResolutionMode,
) -> None:
    interval = NanosecondInterval(start_ns=0, end_ns=1_000_000_000)
    if resolution_mode is AdaptiveResolutionMode.GRID_SEGMENTS:
        request = _request(
            effective_interval=interval,
            resolution_mode=resolution_mode,
            segments=(_segment(0, 1_000_000_000),),
            max_target_count=1,
        )
    else:
        request = _request(
            effective_interval=interval,
            resolution_mode=resolution_mode,
            explicit_targets_ns=(0, 500_000_000),
            max_target_count=1,
        )

    with pytest.raises(ValueError, match="exceeds max_target_count"):
        resolve_frozen_adaptive_targets(request)


def test_extreme_rate_budget_failure_enumerates_only_budget_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        effective_interval=NanosecondInterval(start_ns=0, end_ns=1_000_000_000),
        resolution_mode=AdaptiveResolutionMode.GRID_SEGMENTS,
        segments=(_segment(0, 1_000_000_000, rate_num=10**18),),
        max_target_count=1,
    )
    calls: list[int] = []
    original_target_ns = SamplingGrid.target_ns

    def counted_target_ns(grid: SamplingGrid, k: int) -> int:
        calls.append(k)
        if len(calls) > request.max_target_count + 1:
            raise AssertionError("resolver scanned past the bounded unique-target budget")
        return original_target_ns(grid, k)

    monkeypatch.setattr(SamplingGrid, "target_ns", counted_target_ns)

    with pytest.raises(ValueError, match="exceeds max_target_count"):
        resolve_frozen_adaptive_targets(request)

    assert len(calls) == request.max_target_count + 1


def test_strict_plan_validation_recomputes_complete_targets_after_rehash() -> None:
    plan = resolve_frozen_adaptive_targets(
        _request(
            effective_interval=NanosecondInterval(start_ns=0, end_ns=3),
            resolution_mode=AdaptiveResolutionMode.GRID_SEGMENTS,
            segments=(_segment(0, 3, rate_num=2_000_000_000),),
            max_target_count=4,
        )
    )
    assert tuple((target.target_ns, target.grid_k) for target in plan.targets) == (
        (0, -1),
        (1, 2),
        (2, 3),
    )

    missing = plan.targets[:-1]
    extra = (
        *plan.targets,
        ResolvedAdaptiveTarget(
            ordinal=3,
            target_ns=2,
            segment_ordinal=0,
            grid_k=4,
        ),
    )
    wrong_lowest_k = (
        ResolvedAdaptiveTarget(
            ordinal=0,
            target_ns=0,
            segment_ordinal=0,
            grid_k=0,
        ),
        *plan.targets[1:],
    )

    for forged_targets in (missing, extra, wrong_lowest_k):
        with pytest.raises(ValidationError, match="complete canonical target set"):
            ResolvedAdaptivePlan.model_validate(_forged_plan_values(plan, forged_targets))


def _coverage_policy(
    *,
    base_target_budget_per_camera: int = 2,
    context_offsets_ns: tuple[int, ...] = (-500_000_000, 500_000_000),
    max_targets_per_camera: int = 5,
    max_targets_total: int = 30,
    max_upgrade_requests: int = 1_024,
) -> AdaptiveCoveragePolicy:
    return AdaptiveCoveragePolicy(
        version="adaptive-coverage-v1",
        base_rate_num=1,
        base_target_budget_per_camera=base_target_budget_per_camera,
        context_offsets_ns=context_offsets_ns,
        max_upgrade_requests=max_upgrade_requests,
        max_targets_per_camera=max_targets_per_camera,
        max_targets_total=max_targets_total,
    )


def _coverage_target(
    plan: object,
    camera_id: CameraId,
    target_ns: int,
) -> object:
    targets = plan.targets
    return next(
        target
        for target in targets
        if target.camera_id is camera_id and target.target_ns == target_ns
    )


def test_coverage_planner_reserves_phase_stable_base_budget_for_every_camera() -> None:
    plan = AdaptiveCoveragePlanner(_coverage_policy(base_target_budget_per_camera=3)).plan(
        NanosecondInterval(start_ns=0, end_ns=10_000_000_000)
    )

    assert plan.base_target_count == 18
    for camera_id in CameraId:
        assert [
            target.target_ns
            for target in plan.targets
            if target.camera_id is camera_id and target.base_coverage
        ] == [0, 4_000_000_000, 9_000_000_000]
    assert len(plan.targets) == 18


def test_coverage_planner_bounded_base_selection_skips_extreme_grid_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_enumerated(
        _grid: SamplingGrid,
        _start_ns: int,
        _end_ns: int,
    ) -> object:
        raise AssertionError("coverage planner must not enumerate every extreme-rate target")

    monkeypatch.setattr(SamplingGrid, "iter_unique_targets", fail_if_enumerated)

    plan = AdaptiveCoveragePlanner(
        AdaptiveCoveragePolicy(
            version="adaptive-coverage-v1",
            base_rate_num=10**18,
            base_target_budget_per_camera=2,
            max_targets_per_camera=2,
            max_targets_total=12,
        )
    ).plan(NanosecondInterval(start_ns=0, end_ns=1_000_000_000))

    assert plan.base_target_count == 12
    assert len(plan.targets) == 12


def test_coverage_planner_preserves_available_unique_targets_on_rounded_high_rate_grid() -> None:
    plan = AdaptiveCoveragePlanner(
        AdaptiveCoveragePolicy(
            version="adaptive-coverage-v1",
            base_rate_num=2_000_000_000,
            base_target_budget_per_camera=4,
            max_targets_per_camera=4,
            max_targets_total=24,
        )
    ).plan(NanosecondInterval(start_ns=0, end_ns=4))

    for camera_id in CameraId:
        assert [
            target.target_ns
            for target in plan.targets
            if target.camera_id is camera_id and target.base_coverage
        ] == [0, 1, 2, 3]


def test_coverage_planner_caps_base_budget_at_available_unique_targets() -> None:
    plan = AdaptiveCoveragePlanner(
        AdaptiveCoveragePolicy(
            version="adaptive-coverage-v1",
            base_rate_num=1,
            base_target_budget_per_camera=3,
            max_targets_per_camera=3,
            max_targets_total=18,
        )
    ).plan(NanosecondInterval(start_ns=0, end_ns=1))

    assert plan.base_target_count == 6
    for camera_id in CameraId:
        assert [
            target.target_ns
            for target in plan.targets
            if target.camera_id is camera_id and target.base_coverage
        ] == [0]

    high_rate_plan = AdaptiveCoveragePlanner(
        AdaptiveCoveragePolicy(
            version="adaptive-coverage-v1",
            base_rate_num=2_000_000_000,
            base_target_budget_per_camera=4,
            max_targets_per_camera=4,
            max_targets_total=24,
        )
    ).plan(NanosecondInterval(start_ns=0, end_ns=2))
    assert high_rate_plan.base_target_count == 12
    for camera_id in CameraId:
        assert [
            target.target_ns
            for target in high_rate_plan.targets
            if target.camera_id is camera_id and target.base_coverage
        ] == [0, 1]


def test_coverage_planner_keeps_trigger_and_pre_post_context_for_all_upgrade_reasons() -> None:
    interval = NanosecondInterval(start_ns=0, end_ns=2_000_000_000)
    requests = (
        AdaptiveUpgradeRequest(
            camera_id=CameraId.CAM_01,
            trigger_timestamp_ns=1_000_000_000,
            reason=AdaptiveUpgradeReason.SOURCE_QUALITY_SIGNAL,
        ),
        AdaptiveUpgradeRequest(
            camera_id=CameraId.CAM_02,
            trigger_timestamp_ns=1_000_000_000,
            reason=AdaptiveUpgradeReason.COARSE_UNCERTAINTY,
        ),
        AdaptiveUpgradeRequest(
            camera_id=CameraId.CAM_03,
            trigger_timestamp_ns=1_000_000_000,
            reason=AdaptiveUpgradeReason.CROSS_CAMERA_DISAGREEMENT,
        ),
        AdaptiveUpgradeRequest(
            camera_id=CameraId.CAM_04,
            trigger_timestamp_ns=1_000_000_000,
            reason=AdaptiveUpgradeReason.EVENT_CANDIDATE,
        ),
        AdaptiveUpgradeRequest(
            camera_id=CameraId.CAM_05,
            trigger_timestamp_ns=1_000_000_000,
            reason=AdaptiveUpgradeReason.BOUNDARY_REFINEMENT,
        ),
    )
    planner = AdaptiveCoveragePlanner(_coverage_policy())

    plan = planner.plan(interval, requests)
    replay = planner.plan(interval, tuple(reversed(requests)))

    assert plan == replay
    assert plan.base_target_count == 12
    assert plan.upgrade_coordinate_count == 15
    assert plan.upgrade_targets_added == 10
    assert plan.upgrade_coordinates_deduplicated_into_base == 5
    assert {
        provenance.reason for target in plan.targets for provenance in target.upgrade_provenance
    } == set(AdaptiveUpgradeReason)

    original_observation = _coverage_target(plan, CameraId.CAM_01, 1_000_000_000)
    assert original_observation.base_coverage is True
    assert original_observation.upgrade_provenance == (original_observation.upgrade_provenance[0],)
    assert original_observation.upgrade_provenance[0].role is AdaptiveUpgradeTargetRole.TRIGGER

    assert (
        _coverage_target(
            plan,
            CameraId.CAM_01,
            500_000_000,
        )
        .upgrade_provenance[0]
        .role
        is AdaptiveUpgradeTargetRole.PRE_CONTEXT
    )
    assert (
        _coverage_target(
            plan,
            CameraId.CAM_01,
            1_500_000_000,
        )
        .upgrade_provenance[0]
        .role
        is AdaptiveUpgradeTargetRole.POST_CONTEXT
    )


def test_coverage_planner_retains_clipped_edge_context_with_its_original_trigger() -> None:
    plan = AdaptiveCoveragePlanner(_coverage_policy()).plan(
        NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        (
            AdaptiveUpgradeRequest(
                camera_id=CameraId.CAM_01,
                trigger_timestamp_ns=100_000_000,
                reason=AdaptiveUpgradeReason.SOURCE_QUALITY_SIGNAL,
            ),
        ),
    )

    clipped_pre_context = _coverage_target(plan, CameraId.CAM_01, 0).upgrade_provenance[0]
    assert clipped_pre_context.role is AdaptiveUpgradeTargetRole.PRE_CONTEXT
    assert clipped_pre_context.trigger_timestamp_ns == 100_000_000
    assert clipped_pre_context.context_offset_ns == -500_000_000
    assert clipped_pre_context.context_clipped is True

    trigger = _coverage_target(plan, CameraId.CAM_01, 100_000_000).upgrade_provenance[0]
    assert trigger.role is AdaptiveUpgradeTargetRole.TRIGGER
    assert trigger.trigger_timestamp_ns == 100_000_000


def test_coverage_planner_prioritizes_triggers_and_reports_per_camera_budget_drops() -> None:
    plan = AdaptiveCoveragePlanner(
        _coverage_policy(
            context_offsets_ns=(-200_000_000, 200_000_000),
            max_targets_per_camera=3,
            max_targets_total=18,
        )
    ).plan(
        NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        (
            AdaptiveUpgradeRequest(
                camera_id=CameraId.CAM_01,
                trigger_timestamp_ns=500_000_000,
                reason=AdaptiveUpgradeReason.EVENT_CANDIDATE,
            ),
        ),
    )

    assert plan.upgrade_coordinate_count == 3
    assert plan.upgrade_targets_added == 1
    assert plan.dropped_by_per_camera_budget == 2
    assert plan.dropped_by_total_budget == 0
    assert (
        _coverage_target(
            plan,
            CameraId.CAM_01,
            500_000_000,
        )
        .upgrade_provenance[0]
        .role
        is AdaptiveUpgradeTargetRole.TRIGGER
    )


def test_coverage_planner_applies_total_budget_after_base_and_per_camera_reservations() -> None:
    plan = AdaptiveCoveragePlanner(
        _coverage_policy(
            context_offsets_ns=(-200_000_000, -100_000_000, 100_000_000, 200_000_000),
            max_targets_per_camera=7,
            max_targets_total=13,
        )
    ).plan(
        NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        (
            AdaptiveUpgradeRequest(
                camera_id=CameraId.CAM_01,
                trigger_timestamp_ns=500_000_000,
                reason=AdaptiveUpgradeReason.EVENT_CANDIDATE,
            ),
        ),
    )

    assert plan.base_target_count == 12
    assert plan.upgrade_coordinate_count == 5
    assert plan.upgrade_targets_added == 1
    assert plan.dropped_by_per_camera_budget == 0
    assert plan.dropped_by_total_budget == 4
    assert (
        _coverage_target(
            plan,
            CameraId.CAM_01,
            500_000_000,
        )
        .upgrade_provenance[0]
        .role
        is AdaptiveUpgradeTargetRole.TRIGGER
    )


def test_coverage_planner_rejects_upgrade_input_beyond_its_declared_bound() -> None:
    planner = AdaptiveCoveragePlanner(_coverage_policy(max_upgrade_requests=1))
    interval = NanosecondInterval(start_ns=0, end_ns=2_000_000_000)
    requests = (
        AdaptiveUpgradeRequest(
            camera_id=CameraId.CAM_01,
            trigger_timestamp_ns=100_000_000,
            reason=AdaptiveUpgradeReason.SOURCE_QUALITY_SIGNAL,
        ),
        AdaptiveUpgradeRequest(
            camera_id=CameraId.CAM_02,
            trigger_timestamp_ns=200_000_000,
            reason=AdaptiveUpgradeReason.EVENT_CANDIDATE,
        ),
    )

    with pytest.raises(ValueError, match="max_upgrade_requests"):
        planner.plan(interval, requests)


def test_coverage_planner_fails_closed_when_budget_would_erase_an_original_trigger() -> None:
    planner = AdaptiveCoveragePlanner(
        _coverage_policy(
            context_offsets_ns=(-100_000_000, 100_000_000),
            max_targets_total=13,
        )
    )

    with pytest.raises(ValueError, match="cannot preserve every original upgrade trigger"):
        planner.plan(
            NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
            (
                AdaptiveUpgradeRequest(
                    camera_id=CameraId.CAM_01,
                    trigger_timestamp_ns=500_000_000,
                    reason=AdaptiveUpgradeReason.SOURCE_QUALITY_SIGNAL,
                ),
                AdaptiveUpgradeRequest(
                    camera_id=CameraId.CAM_02,
                    trigger_timestamp_ns=250_000_000,
                    reason=AdaptiveUpgradeReason.BOUNDARY_REFINEMENT,
                ),
            ),
        )
