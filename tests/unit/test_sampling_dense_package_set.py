from __future__ import annotations

from hashlib import sha256
from itertools import pairwise
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from robata.contracts.common import NanosecondInterval
from robata.contracts.pipeline import SamplingPurpose
from robata.contracts.sampling_plan import (
    FrameBudget,
    OverflowPolicy,
    PerCameraOverride,
    SamplingPlan,
)
from robata.sampling.dense import (
    DenseSamplingPlanner,
    DenseSplitPolicy,
    frame_counts_for_interval,
    sampling_plan_projection,
)
from robata.sampling.package_set import (
    MaterializedPackageRef,
    PackageLineage,
    PackageSetBuilder,
    TemporalPackageSet,
    sampling_plan_digest,
)

SECOND = 1_000_000_000


def _plan(
    *,
    qa_fps: float = 1.0,
    dense_fps: float = 5.0,
    max_per_camera: int = 4,
    max_total: int = 24,
    per_camera: tuple[PerCameraOverride, ...] = (),
    overflow_policy: OverflowPolicy = OverflowPolicy.SPLIT_WINDOW,
) -> SamplingPlan:
    return SamplingPlan(
        sampling_plan_id="sampling-plan-1",
        version="sampling-v1",
        qa_sampling_rate_fps=qa_fps,
        event_sampling_rate_fps=2.0,
        dense_sampling_rate_fps=dense_fps,
        per_camera=per_camera,
        adaptive_policy=None,
        frame_budget=FrameBudget(
            max_frames_per_camera=max_per_camera,
            max_frames_total=max_total,
            overflow_policy=overflow_policy,
        ),
    )


def _policy(*, overlap_ns: int = 100_000_000, plan: SamplingPlan | None = None) -> DenseSplitPolicy:
    if plan is None:
        return DenseSplitPolicy(version="dense-v1", overlap_ns=overlap_ns)
    return DenseSplitPolicy(
        version="dense-v1",
        overlap_ns=overlap_ns,
        max_frames_per_camera=plan.frame_budget.max_frames_per_camera,
        max_frames_total=plan.frame_budget.max_frames_total,
    )


def _window(
    start_ns: int = 0,
    end_ns: int = 2 * SECOND,
    *,
    purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
) -> SimpleNamespace:
    interval = NanosecondInterval(start_ns=start_ns, end_ns=end_ns)
    return SimpleNamespace(
        window_id="window-1",
        mcap_id="mcap-1",
        camera_mapping_run_id="mapping-1",
        requested_interval=interval,
        interval=interval,
        purpose=purpose,
    )


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _lineage(
    plan: SamplingPlan,
    *,
    purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
) -> PackageLineage:
    return PackageLineage(
        source_content_sha256=_digest("source"),
        window_semantic_sha256=_digest("window"),
        camera_mapping_semantic_sha256=_digest("mapping"),
        alignment_semantic_sha256=_digest("alignment"),
        sampling_plan_sha256=sampling_plan_digest(plan, purpose=purpose),
    )


def _build_package_set(
    builder: PackageSetBuilder,
    window: SimpleNamespace,
    plan: SamplingPlan,
    alignment_id: str = "alignment-1",
    *,
    content_seed: str = "content",
    manifest_seed: str | None = None,
    package_id_prefix: str = "package",
) -> TemporalPackageSet:
    parts = builder.plan_parts(window, plan)
    members = tuple(
        MaterializedPackageRef(
            ordinal=part.ordinal,
            package_id=f"{package_id_prefix}-{part.ordinal}",
            package_semantic_content_sha256=_digest(f"{content_seed}:semantic:{part.ordinal}"),
            package_manifest_sha256=_digest(
                f"{manifest_seed or content_seed}:manifest:{part.ordinal}"
            ),
        )
        for part in parts
    )
    return builder.build_package_set(
        window,
        plan,
        alignment_id,
        lineage=_lineage(
            plan,
            purpose=getattr(window, "purpose", SamplingPurpose.ACTION_DENSE),
        ),
        materialized_members=members,
        created_at="2026-07-19T00:00:00Z",
    )


def test_dense_split_uses_six_camera_total_budget_and_exact_progress() -> None:
    plan = _plan(max_per_camera=4, max_total=24)
    planner = DenseSamplingPlanner(_policy(plan=plan), plan)
    windows = planner.plan_dense_windows(
        [NanosecondInterval(start_ns=0, end_ns=2 * SECOND)],
        padding_ns=0,
        recording_duration_ns=2 * SECOND,
    )

    assert len(windows) == 3
    assert windows[0].interval.start_ns == 0
    assert windows[-1].interval.end_ns == 2 * SECOND
    for window in windows:
        counts = frame_counts_for_interval(window.interval, plan)
        assert all(count <= 4 for count in counts)
        assert sum(counts) <= 24
    for previous, current in pairwise(windows):
        assert current.interval.start_ns > previous.interval.start_ns
        assert previous.overlap_after_ns == current.overlap_before_ns


def test_total_budget_can_be_stricter_than_per_camera_budget() -> None:
    plan = _plan(max_per_camera=10, max_total=12)
    planner = DenseSamplingPlanner(_policy(overlap_ns=0, plan=plan), plan)
    windows = planner.plan_dense_windows(
        [NanosecondInterval(start_ns=0, end_ns=SECOND)],
        padding_ns=0,
        recording_duration_ns=SECOND,
    )

    assert len(windows) == 3
    assert [window.interval.duration_ns for window in windows] == [
        400_000_000,
        400_000_000,
        200_000_000,
    ]
    assert all(sum(frame_counts_for_interval(window.interval, plan)) <= 12 for window in windows)


def test_dense_rate_is_taken_from_sampling_plan_not_a_five_fps_placeholder() -> None:
    plan = _plan(dense_fps=2.0, max_per_camera=4, max_total=24)
    planner = DenseSamplingPlanner(_policy(overlap_ns=0, plan=plan), plan)
    windows = planner.plan_dense_windows(
        [NanosecondInterval(start_ns=0, end_ns=3 * SECOND)],
        padding_ns=0,
        recording_duration_ns=3 * SECOND,
    )

    assert len(windows) == 2
    assert [window.interval.duration_ns for window in windows] == [2 * SECOND, SECOND]


def test_coarse_qa_planning_uses_uniform_qa_rate_not_dense_rate_or_overrides() -> None:
    plan = _plan(
        qa_fps=1.0,
        dense_fps=5.0,
        max_per_camera=1,
        max_total=6,
        per_camera=(PerCameraOverride(camera_id="cam_01", dense_sampling_rate_fps=9.0),),
    )
    interval = NanosecondInterval(start_ns=0, end_ns=SECOND)

    projection = sampling_plan_projection(plan, purpose=SamplingPurpose.QA_COARSE)
    assert projection["purpose"] == "QA_COARSE"
    assert projection["strategy"] == "UNIFORM"
    assert projection["qa_rate"] == {"numerator": 1, "denominator": 1}
    assert set(
        (rate["numerator"], rate["denominator"]) for rate in projection["per_camera"].values()
    ) == {(1, 1)}
    assert (
        frame_counts_for_interval(
            interval,
            plan,
            purpose=SamplingPurpose.QA_COARSE,
        )
        == (1,) * 6
    )

    parts = PackageSetBuilder("reduce-v1").plan_parts(
        _window(0, SECOND, purpose=SamplingPurpose.QA_COARSE),
        plan,
    )
    assert len(parts) == 1
    assert sampling_plan_digest(plan, purpose=SamplingPurpose.QA_COARSE) != (
        sampling_plan_digest(plan)
    )


def test_qa_dense_planning_uses_uniform_dense_rate_and_distinct_projection() -> None:
    plan = _plan(
        dense_fps=5.0,
        max_per_camera=5,
        max_total=30,
        per_camera=(PerCameraOverride(camera_id="cam_01", dense_sampling_rate_fps=9.0),),
    )
    interval = NanosecondInterval(start_ns=0, end_ns=SECOND)

    projection = sampling_plan_projection(plan, purpose=SamplingPurpose.QA_DENSE)
    action_projection = sampling_plan_projection(plan)

    assert projection["purpose"] == "QA_DENSE"
    assert projection["strategy"] == "DENSE"
    assert projection["dense_rate"] == {"numerator": 5, "denominator": 1}
    assert set(
        (rate["numerator"], rate["denominator"]) for rate in projection["per_camera"].values()
    ) == {(5, 1)}
    assert (
        frame_counts_for_interval(
            interval,
            plan,
            purpose=SamplingPurpose.QA_DENSE,
        )
        == (5,) * 6
    )

    assert "purpose" not in action_projection
    assert "strategy" not in action_projection
    assert action_projection["per_camera"]["cam_01"] == {
        "numerator": 9,
        "denominator": 1,
    }
    assert sampling_plan_digest(plan, purpose=SamplingPurpose.QA_DENSE) != (
        sampling_plan_digest(plan)
    )

    parts = PackageSetBuilder("reduce-v1").plan_parts(
        _window(0, SECOND, purpose=SamplingPurpose.QA_DENSE),
        plan,
    )
    assert len(parts) == 1


def test_provider_neutral_plan_rejects_other_sampling_purposes() -> None:
    with pytest.raises(ValueError, match="only QA_COARSE, QA_DENSE, and ACTION_DENSE"):
        sampling_plan_projection(_plan(), purpose=SamplingPurpose.EVENT_PROPOSAL)


def test_dense_input_order_does_not_change_semantic_ids() -> None:
    plan = _plan()
    policy = _policy(plan=plan)
    candidates = (
        NanosecondInterval(start_ns=3 * SECOND, end_ns=4 * SECOND),
        NanosecondInterval(start_ns=0, end_ns=2 * SECOND),
    )
    first = DenseSamplingPlanner(policy, plan).plan_dense_windows(candidates, 0, 5 * SECOND)
    second = DenseSamplingPlanner(policy, plan).plan_dense_windows(
        tuple(reversed(candidates)), 0, 5 * SECOND
    )

    assert [window.window_id for window in first] == [window.window_id for window in second]
    assert [window.model_dump() for window in first] == [window.model_dump() for window in second]


def test_duplicate_candidates_are_preserved_with_stable_distinct_ids() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    candidate = NanosecondInterval(start_ns=0, end_ns=SECOND)
    windows = DenseSamplingPlanner(_policy(plan=plan), plan).plan_dense_windows(
        [candidate, candidate],
        padding_ns=0,
        recording_duration_ns=SECOND,
    )

    assert len(windows) == 2
    assert windows[0].window_id != windows[1].window_id


def test_padding_is_requested_but_clipping_is_effective() -> None:
    plan = _plan(max_per_camera=10, max_total=60)
    windows = DenseSamplingPlanner(_policy(plan=plan), plan).plan_dense_windows(
        [NanosecondInterval(start_ns=100_000_000, end_ns=200_000_000)],
        padding_ns=200_000_000,
        recording_duration_ns=400_000_000,
    )

    assert len(windows) == 1
    assert windows[0].requested_interval == NanosecondInterval(
        start_ns=-100_000_000,
        end_ns=400_000_000,
    )
    assert windows[0].interval == NanosecondInterval(start_ns=0, end_ns=400_000_000)


def test_overlap_equal_to_admissible_span_fails_closed() -> None:
    plan = _plan(max_per_camera=4, max_total=24)
    policy = _policy(overlap_ns=800_000_000, plan=plan)
    with pytest.raises(ValueError, match="strictly less"):
        DenseSamplingPlanner(policy, plan).plan_dense_windows(
            [NanosecondInterval(start_ns=0, end_ns=2 * SECOND)],
            padding_ns=0,
            recording_duration_ns=2 * SECOND,
        )


def test_package_set_builder_emits_valid_ordered_members() -> None:
    plan = _plan()
    package_set = _build_package_set(
        PackageSetBuilder("reduce-v1", _policy(plan=plan)),
        _window(),
        plan,
    )

    assert package_set.split_reason == "FRAME_BUDGET"
    assert not hasattr(package_set, "capability_snapshot_digest")
    assert [member.ordinal for member in package_set.members] == [0, 1, 2]
    assert [member.part_count for member in package_set.members] == [3, 3, 3]
    assert package_set.members[0].start_ns == package_set.start_ns
    assert package_set.members[-1].end_ns == package_set.end_ns
    assert package_set.members[0].overlap_after_ns == package_set.members[1].overlap_before_ns


def test_package_set_ids_and_digests_are_stable_for_plan_override_order() -> None:
    overrides = (
        PerCameraOverride(camera_id="cam_02", dense_sampling_rate_fps=7.0),
        PerCameraOverride(camera_id="cam_04", dense_sampling_rate_fps=3.0),
    )
    reversed_overrides = tuple(reversed(overrides))
    first_plan = _plan(per_camera=overrides)
    second_plan = _plan(per_camera=reversed_overrides)
    policy = _policy(plan=first_plan)
    first = _build_package_set(
        PackageSetBuilder("reduce-v1", policy),
        _window(),
        first_plan,
    )
    second = _build_package_set(
        PackageSetBuilder("reduce-v1", policy),
        _window(),
        second_plan,
    )

    assert first.package_set_id == second.package_set_id
    assert first.split_plan_digest == second.split_plan_digest
    assert first.member_manifest_sha256 == second.member_manifest_sha256
    assert [member.package_id for member in first.members] == [
        member.package_id for member in second.members
    ]


def test_sampling_semantic_change_changes_package_set_identity() -> None:
    first_plan = _plan(dense_fps=1.0, max_per_camera=20, max_total=120)
    second_plan = _plan(dense_fps=2.0, max_per_camera=20, max_total=120)
    first = _build_package_set(
        PackageSetBuilder("reduce-v1"),
        _window(0, SECOND),
        first_plan,
    )
    second = _build_package_set(
        PackageSetBuilder("reduce-v1"),
        _window(0, SECOND),
        second_plan,
    )

    assert first.split_group_id != second.split_group_id
    assert first.split_plan_digest != second.split_plan_digest
    assert first.package_set_id != second.package_set_id
    assert first.members[0].package_id == second.members[0].package_id


def test_package_set_rejects_gaps_and_mismatched_digest() -> None:
    plan = _plan()
    package_set = _build_package_set(
        PackageSetBuilder("reduce-v1", _policy(plan=plan)),
        _window(),
        plan,
    )
    payload = package_set.model_dump()
    payload["members"][1]["start_ns"] += 1_000_000_000
    with pytest.raises(ValidationError, match=r"gap|overlap|member effective"):
        TemporalPackageSet.model_validate(payload)

    payload = package_set.model_dump()
    payload["member_manifest_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="member_manifest_sha256"):
        TemporalPackageSet.model_validate(payload)


def test_non_split_package_has_none_reason() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    package_set = _build_package_set(
        PackageSetBuilder("reduce-v1"),
        _window(0, SECOND),
        plan,
        "a1",
    )
    assert package_set.split_reason == "NONE"
    assert len(package_set.members) == 1


def test_provider_limit_is_not_a_package_split_reason() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    package_set = _build_package_set(
        PackageSetBuilder("reduce-v1"),
        _window(0, SECOND),
        plan,
        "a1",
    )
    payload = package_set.model_dump()
    payload["split_reason"] = "PROVIDER_LIMIT"
    with pytest.raises(ValidationError):
        TemporalPackageSet.model_validate(payload)

    payload = package_set.model_dump()
    payload["capability_snapshot_digest"] = _digest("provider-capability")
    with pytest.raises(ValidationError):
        TemporalPackageSet.model_validate(payload)


def test_package_content_changes_manifest_and_set_but_not_split_plan() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    builder = PackageSetBuilder("reduce-v1")
    first = _build_package_set(builder, _window(0, SECOND), plan, content_seed="first")
    second = _build_package_set(builder, _window(0, SECOND), plan, content_seed="second")

    assert first.split_plan_digest == second.split_plan_digest
    assert first.split_group_id == second.split_group_id
    assert first.member_manifest_sha256 != second.member_manifest_sha256
    assert first.package_set_id != second.package_set_id


def test_exact_package_manifest_relocation_does_not_change_package_set_identity() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    builder = PackageSetBuilder("reduce-v1")
    first = _build_package_set(
        builder,
        _window(0, SECOND),
        plan,
        content_seed="same-content",
        manifest_seed="first-serialization",
    )
    relocated = _build_package_set(
        builder,
        _window(0, SECOND),
        plan,
        content_seed="same-content",
        manifest_seed="second-serialization",
    )

    assert first.members[0].package_manifest_sha256 != (
        relocated.members[0].package_manifest_sha256
    )
    assert first.member_manifest_sha256 == relocated.member_manifest_sha256
    assert first.package_set_id == relocated.package_set_id


def test_forged_member_semantic_content_is_rejected_by_manifest_binding() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    package_set = _build_package_set(
        PackageSetBuilder("reduce-v1"),
        _window(0, SECOND),
        plan,
    )
    payload = package_set.model_dump()
    payload["members"][0]["package_semantic_content_sha256"] = _digest("forged")

    with pytest.raises(ValidationError, match="member_manifest_sha256"):
        TemporalPackageSet.model_validate(payload)


def test_package_set_builder_requires_materialized_lineage_and_content() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    builder = PackageSetBuilder("reduce-v1")

    with pytest.raises(ValueError, match="lineage"):
        builder.build_package_set(
            _window(0, SECOND),
            plan,
            "alignment-1",
            materialized_members=(),
            created_at="2026-07-19T00:00:00Z",
        )


def test_package_set_identity_excludes_row_ids_and_clock_time() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    builder = PackageSetBuilder("reduce-v1")
    first = _build_package_set(builder, _window(0, SECOND), plan)
    moved_window = SimpleNamespace(
        window_id="window-2",
        mcap_id="mcap-2",
        camera_mapping_run_id="mapping-2",
        requested_interval=NanosecondInterval(start_ns=0, end_ns=SECOND),
        interval=NanosecondInterval(start_ns=0, end_ns=SECOND),
    )
    second = _build_package_set(
        builder,
        moved_window,
        plan,
        alignment_id="alignment-2",
        package_id_prefix="other-package",
    ).model_copy(update={"created_at": "2026-07-20T00:00:00Z"})

    assert first.members[0].package_id != second.members[0].package_id
    assert first.window_id != second.window_id
    assert first.package_set_id == second.package_set_id
    assert first.split_group_id == second.split_group_id
    assert first.member_manifest_sha256 == second.member_manifest_sha256

    with pytest.raises(ValueError, match="materialized"):
        builder.build_package_set(
            _window(0, SECOND),
            plan,
            "alignment-1",
            lineage=_lineage(plan),
            created_at="2026-07-19T00:00:00Z",
        )


def test_sampling_plan_row_id_does_not_change_package_identity() -> None:
    first_plan = _plan(max_per_camera=20, max_total=120)
    second_plan = first_plan.model_copy(update={"sampling_plan_id": "another-row-id"})
    builder = PackageSetBuilder("reduce-v1")

    first = _build_package_set(builder, _window(0, SECOND), first_plan)
    second = _build_package_set(builder, _window(0, SECOND), second_plan)

    assert sampling_plan_digest(first_plan) == sampling_plan_digest(second_plan)
    assert first.package_set_id == second.package_set_id
    assert first.split_group_id == second.split_group_id
