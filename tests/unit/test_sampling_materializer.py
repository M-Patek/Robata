from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID

import pytest

from robata.admission import (
    AdmissionContextResolver,
    AlignmentAdmissionOutcome,
    PrimaryAdmissionEvaluation,
    PrimaryAdmissionPolicy,
    SourceAdmissionOutcome,
)
from robata.contracts.alignment import (
    AlignmentMethod,
    AlignmentRun,
    AlignmentSegment,
    AlignmentStatus,
    CameraAlignment,
    CanonicalOrigin,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.contracts.sampling_plan import FrameBudget, OverflowPolicy, SamplingPlan
from robata.contracts.schema_registry import SchemaRegistry
from robata.contracts.temporal import PackageLineage
from robata.inference import (
    ApplicableProviderLimits,
    InferenceInputPlanner,
    InputPlanPreparer,
    InputPlanTarget,
    InputPreparationError,
    PromptOutputContract,
    ProviderReferenceCatalog,
    ProviderRenderingPolicy,
    TransformOperation,
    VisionTask,
)
from robata.inference.input_plan import INFERENCE_INPUT_PLANNER_VERSION
from robata.sampling import (
    CameraSourceFrameIndex,
    CanonicalSixCameraFrameIndex,
    FrameAlignmentProjectionFact,
    IndexedSourceFrame,
    MaterializedArtifactManifest,
    MaterializedCameraStatus,
    MaterializedFrameArtifactFact,
    OfflineTemporalPackageMaterializer,
    PackageMaterializationError,
    PackageMaterializationErrorCode,
    PackageSetBuilder,
    SelectionStatus,
    TemporalPackageMaterializationPolicy,
)
from robata.sampling.dense import IntervalPart
from robata.sampling.package_set import sampling_plan_digest
from tests.contract.test_admission_evidence_v2_contract import (
    _alignment_manifest,
    _ready_manifest,
    _validation_report,
)

SECOND = 1_000_000_000
SOURCE_ORIGIN_NS = 10_000_000_000


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _sampling_plan(row_id: str = "sampling-plan-row-a") -> SamplingPlan:
    return SamplingPlan(
        sampling_plan_id=row_id,
        version="sampling-v1",
        qa_sampling_rate_fps=1.0,
        event_sampling_rate_fps=2.0,
        dense_sampling_rate_fps=2.0,
        frame_budget=FrameBudget(
            max_frames_per_camera=20,
            max_frames_total=120,
            overflow_policy=OverflowPolicy.SPLIT_WINDOW,
        ),
    )


def _lineage(plan: SamplingPlan) -> PackageLineage:
    return PackageLineage(
        source_content_sha256=_digest("source-content"),
        window_semantic_sha256=_digest("window"),
        camera_mapping_semantic_sha256=_digest("mapping"),
        alignment_semantic_sha256=_digest("alignment"),
        sampling_plan_sha256=sampling_plan_digest(plan),
    )


def _alignment(alias: int = 0) -> AlignmentRun:
    alignment_id = _uuid(100 + alias)
    cameras: dict[str, CameraAlignment] = {}
    for ordinal, camera_id in enumerate(CAMERA_IDS):
        cameras[camera_id.value] = CameraAlignment(
            source_clock_id=f"clock-{camera_id.value}",
            source_timestamp_unit="ns",
            derived_drift_ppm=0.0,
            residual_p95_ns=0,
            max_error_ns=0,
            coverage=1.0,
            segments=(
                AlignmentSegment(
                    segment_id=_uuid(1_000 + alias * 100 + ordinal),
                    source_epoch_id="epoch-0",
                    source_order_start=0,
                    source_order_end=100,
                    source_start_ns=SOURCE_ORIGIN_NS,
                    source_end_ns=SOURCE_ORIGIN_NS + 3 * SECOND,
                    source_anchor_ns=SOURCE_ORIGIN_NS,
                    canonical_anchor_ns=0,
                    rate_numerator="1",
                    rate_denominator="1",
                    rounding="HALF_EVEN",
                ),
            ),
            status=AlignmentStatus.VALID,
        )
    return AlignmentRun(
        schema_version="1.0",
        alignment_id=alignment_id,
        mcap_id=_uuid(200 + alias),
        camera_mapping_run_id=_uuid(300 + alias),
        reference_timebase="recording_relative_ns",
        canonical_origin=CanonicalOrigin(
            source="fixture",
            reference_timestamp_ns=SOURCE_ORIGIN_NS,
            utc=None,
        ),
        method=AlignmentMethod.MCAP_LOG_TIME,
        algorithm_version="alignment-v1",
        status=AlignmentStatus.VALID,
        cameras=cameras,
        policy_version="alignment-policy-v1",
        created_at=f"2026-07-{19 + alias:02d}T00:00:00Z",
    )


def _frame_index(
    alignment: AlignmentRun,
    lineage: PackageLineage,
    *,
    alias: int = 0,
    empty_camera: CameraId | None = CameraId.CAM_06,
    forge_projection: bool = False,
) -> CanonicalSixCameraFrameIndex:
    cameras: dict[CameraId, CameraSourceFrameIndex] = {}
    aligned_times = (250_000_000, 900_000_000, 1_100_000_000)
    for camera_id in CAMERA_IDS:
        frames: list[IndexedSourceFrame] = []
        segment_id = alignment.cameras[camera_id.value].segments[0].segment_id
        if camera_id is not empty_camera:
            for source_order, aligned_timestamp_ns in enumerate(aligned_times):
                projected = aligned_timestamp_ns
                if forge_projection and camera_id is CameraId.CAM_01 and source_order == 0:
                    projected += 1
                frames.append(
                    IndexedSourceFrame(
                        source_frame_id=(
                            f"source-frame-row-{alias}-{camera_id.value}-{source_order}"
                        ),
                        source_order=source_order,
                        source_timestamp_ns=SOURCE_ORIGIN_NS + aligned_timestamp_ns,
                        source_locator={
                            "camera_id": camera_id.value,
                            "message_offset": source_order,
                        },
                        decodable=True,
                        alignment_projection=FrameAlignmentProjectionFact(
                            projection_id=(
                                f"projection-row-{alias}-{camera_id.value}-{source_order}"
                            ),
                            alignment_id=alignment.alignment_id,
                            segment_id=segment_id,
                            aligned_timestamp_ns=projected,
                        ),
                    )
                )
        cameras[camera_id] = CameraSourceFrameIndex(
            camera_id=camera_id,
            stream_id=f"stream-row-{alias}-{camera_id.value}",
            stream_semantic_sha256=_digest(f"stream:{camera_id.value}"),
            frames=tuple(frames),
        )
    return CanonicalSixCameraFrameIndex(
        mcap_id=alignment.mcap_id,
        camera_mapping_run_id=alignment.camera_mapping_run_id,
        alignment_id=alignment.alignment_id,
        source_content_sha256=lineage.source_content_sha256,
        camera_mapping_semantic_sha256=lineage.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=lineage.alignment_semantic_sha256,
        cameras=SixCameraMap[CameraSourceFrameIndex](cameras),
    )


def _part() -> IntervalPart:
    interval = NanosecondInterval(start_ns=0, end_ns=1_500_000_000)
    return IntervalPart(
        requested_interval=interval,
        effective_interval=interval,
        ordinal=0,
        part_count=1,
        overlap_before_ns=0,
        overlap_after_ns=0,
    )


def _policy() -> TemporalPackageMaterializationPolicy:
    return TemporalPackageMaterializationPolicy(
        version="materialization-v1",
        grid_origin_ns=0,
        selection_tolerance_ns=300_000_000,
        tie_break_policy_version="nearest-v1",
        dedupe_policy_version="one-source-frame-v1",
        producer_version="offline-materializer-v1",
        extractor_version="fixture-png-v1",
    )


def _resolver(
    alias: int = 0,
    *,
    missing: tuple[CameraId, int] | None = None,
):
    def resolve(
        camera_id: CameraId,
        frame: IndexedSourceFrame,
    ) -> MaterializedFrameArtifactFact | None:
        if missing == (camera_id, frame.source_order):
            return None
        label = f"{camera_id.value}:{frame.source_order}"
        return MaterializedFrameArtifactFact(
            artifact=MaterializedArtifactManifest(
                artifact_id=f"artifact-row-{alias}-{label}",
                uri=f"memory://frames/{alias}/{camera_id.value}/{frame.source_order}.png",
                sha256=_digest(f"png:{label}"),
                bytes=100 + frame.source_order,
                media_type="image/png",
            ),
            width=320,
            height=180,
            quality_flags=("FIXTURE_BYTES_VERIFIED",),
        )

    return resolve


def _materialize(
    *,
    alias: int = 0,
    plan: SamplingPlan | None = None,
    missing: tuple[CameraId, int] | None = None,
    forge_projection: bool = False,
):
    resolved_plan = plan or _sampling_plan()
    lineage = _lineage(resolved_plan)
    alignment = _alignment(alias)
    frame_index = _frame_index(
        alignment,
        lineage,
        alias=alias,
        forge_projection=forge_projection,
    )
    return OfflineTemporalPackageMaterializer(_policy()).materialize(
        part=_part(),
        sampling_plan=resolved_plan,
        alignment_run=alignment,
        frame_index=frame_index,
        lineage=lineage,
        window_id=f"window-row-{alias}",
        artifact_resolver=_resolver(alias, missing=missing),
        created_at=f"2026-07-{19 + alias:02d}T01:00:00Z",
    )


def test_materializer_records_every_target_and_canonical_empty_camera() -> None:
    output = _materialize()
    package = output.package

    assert tuple(package.cameras.keys()) == CAMERA_IDS
    cam_01 = package.cameras[CameraId.CAM_01]
    assert tuple(target.status for target in cam_01.targets) == (
        SelectionStatus.SELECTED,
        SelectionStatus.DEDUPLICATED_FRAME,
        SelectionStatus.SELECTED,
    )
    assert cam_01.targets[0].actual_timestamp_ns == 250_000_000
    assert cam_01.targets[1].selected_frame_ordinal == 0
    assert cam_01.targets[2].actual_timestamp_ns == 900_000_000
    assert len(cam_01.frames) == 2
    assert all(frame.materialized_artifact is not None for frame in cam_01.frames)

    empty = package.cameras[CameraId.CAM_06]
    assert empty.status is MaterializedCameraStatus.NO_FRAME
    assert len(empty.frames) == 0
    assert (
        tuple(target.status for target in empty.targets)
        == (SelectionStatus.NO_FRAME_WITHIN_TOLERANCE,) * 3
    )
    assert package.frame_count_total == 10
    assert output.package_ref.package_id == package.package_id


def test_materialized_ref_is_accepted_by_package_set_builder() -> None:
    plan = _sampling_plan()
    output = _materialize(plan=plan)
    interval = NanosecondInterval(start_ns=0, end_ns=1_500_000_000)
    window = SimpleNamespace(
        window_id=output.package.window_id,
        mcap_id=output.package.mcap_id,
        camera_mapping_run_id=output.package.camera_mapping_run_id,
        requested_interval=interval,
        interval=interval,
    )

    package_set = PackageSetBuilder("reduce-v1").build_package_set(
        window,
        plan,
        output.package.alignment_id,
        lineage=_lineage(plan),
        materialized_members=(output.package_ref,),
        created_at="2026-07-19T02:00:00Z",
    )

    assert package_set.members[0].package_id == output.package.package_id
    assert package_set.members[0].package_manifest_sha256 == output.package_manifest_sha256


def test_semantic_identity_excludes_alias_run_and_clock_fields() -> None:
    first_plan = _sampling_plan("sampling-plan-row-a")
    second_plan = _sampling_plan("sampling-plan-row-b")
    first = _materialize(alias=0, plan=first_plan)
    second = _materialize(alias=1, plan=second_plan)

    assert first.package.package_id == second.package.package_id
    assert first.package.semantic_content_sha256 == second.package.semantic_content_sha256
    assert [frame.frame_id for frame in first.package.cameras[CameraId.CAM_01].frames] == [
        frame.frame_id for frame in second.package.cameras[CameraId.CAM_01].frames
    ]
    assert first.manifest_bytes != second.manifest_bytes
    assert first.package_manifest_sha256 != second.package_manifest_sha256


def test_forged_alignment_projection_fails_closed() -> None:
    with pytest.raises(PackageMaterializationError) as raised:
        _materialize(forge_projection=True)

    assert raised.value.code is PackageMaterializationErrorCode.ALIGNMENT_MISMATCH


def test_selected_frame_without_materialized_artifact_fails_closed() -> None:
    with pytest.raises(PackageMaterializationError) as raised:
        _materialize(missing=(CameraId.CAM_01, 0))

    assert raised.value.code is PackageMaterializationErrorCode.MISSING_ARTIFACT


def _v2_context():
    registry = SchemaRegistry()
    report = _validation_report(registry)
    ready = _ready_manifest(registry, report)
    alignment = _alignment_manifest(registry, ready)
    policy = PrimaryAdmissionPolicy.create(
        version="primary-v2",
        admissible_alignment_outcomes=(AlignmentAdmissionOutcome.VALID,),
    )
    evaluation = PrimaryAdmissionEvaluation(
        recording_identity=report.recording_identity,
        ready_manifest_id=ready.ready_manifest_id,
        ready_manifest_semantic_sha256=ready.ready_manifest_semantic_sha256,
        source_outcome=SourceAdmissionOutcome.READY,
        alignment_outcome=AlignmentAdmissionOutcome.VALID,
        alignment_id=alignment.alignment_id,
        alignment_semantic_sha256=alignment.alignment_semantic_sha256,
        policy_version=policy.version,
        policy_sha256=policy.semantic_sha256,
        admissible=True,
        reason_code="ADMISSIBLE",
    )
    return AdmissionContextResolver().resolve_v2(
        evaluation=evaluation,
        policy=policy,
        validation_report=report,
        ready_manifest=ready,
        alignment_manifest=alignment,
        registry=registry,
    )


def _v2_frame_index(context, lineage: PackageLineage) -> CanonicalSixCameraFrameIndex:
    cameras: dict[CameraId, CameraSourceFrameIndex] = {}
    aligned_times = (250_000_000, 700_000_000, 900_000_000)
    for camera_id in CAMERA_IDS:
        alignment = context.alignment_manifest.cameras[camera_id.value]
        segment = alignment.segments[0]
        frames = ()
        if camera_id is not CameraId.CAM_06:
            frames = tuple(
                IndexedSourceFrame(
                    source_frame_id=f"v2-frame-{camera_id.value}-{source_order}",
                    source_order=source_order,
                    source_timestamp_ns=segment.source_anchor_ns + aligned_timestamp_ns,
                    source_locator={
                        "camera_id": camera_id.value,
                        "message_offset": source_order,
                    },
                    decodable=True,
                    alignment_projection=FrameAlignmentProjectionFact(
                        projection_id=(f"v2-projection-{camera_id.value}-{source_order}"),
                        alignment_id=context.alignment_manifest.alignment_id,
                        segment_id=segment.segment_id,
                        aligned_timestamp_ns=aligned_timestamp_ns,
                    ),
                )
                for source_order, aligned_timestamp_ns in enumerate(aligned_times)
            )
        cameras[camera_id] = CameraSourceFrameIndex(
            camera_id=camera_id,
            stream_id=alignment.stream_id,
            stream_semantic_sha256=alignment.stream_semantic_sha256,
            frames=frames,
        )
    return CanonicalSixCameraFrameIndex(
        mcap_id=context.ready_manifest.mcap_id,
        camera_mapping_run_id=context.ready_manifest.camera_mapping_run_id,
        alignment_id=context.alignment_manifest.alignment_id,
        source_content_sha256=context.source_content_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=context.alignment_semantic_sha256,
        cameras=SixCameraMap[CameraSourceFrameIndex](cameras),
    )


def test_v2_admitted_materialization_cross_binds_each_stream() -> None:
    context = _v2_context()
    plan = _sampling_plan()
    lineage = PackageLineage(
        source_content_sha256=context.source_content_sha256,
        window_semantic_sha256=_digest("v2-window"),
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=context.alignment_semantic_sha256,
        sampling_plan_sha256=sampling_plan_digest(plan),
    )
    frame_index = _v2_frame_index(context, lineage)

    output = OfflineTemporalPackageMaterializer(_policy()).materialize_admitted(
        part=_part(),
        sampling_plan=plan,
        admitted_context=context,
        frame_index=frame_index,
        lineage=lineage,
        window_id="v2-window-row",
        artifact_resolver=_resolver(),
        created_at="2026-07-19T02:00:00Z",
    )

    assert output.package.alignment_id == context.alignment_manifest.alignment_id
    assert output.package.cameras[CameraId.CAM_01].stream_semantic_sha256 == (
        context.alignment_manifest.cameras[CameraId.CAM_01.value].stream_semantic_sha256
    )

    forged_cameras = frame_index.cameras.model_dump(mode="python")
    forged_cameras[CameraId.CAM_01.value]["stream_semantic_sha256"] = _digest("forged-stream")
    forged_index = frame_index.model_copy(
        update={
            "cameras": SixCameraMap[CameraSourceFrameIndex].model_validate(
                forged_cameras,
                strict=True,
            )
        }
    )
    with pytest.raises(PackageMaterializationError) as raised:
        OfflineTemporalPackageMaterializer(_policy()).materialize_admitted(
            part=_part(),
            sampling_plan=plan,
            admitted_context=context,
            frame_index=forged_index,
            lineage=lineage,
            window_id="v2-window-row",
            artifact_resolver=_resolver(),
            created_at="2026-07-19T02:00:00Z",
        )
    assert raised.value.code is PackageMaterializationErrorCode.ALIGNMENT_MISMATCH


def _input_target() -> InputPlanTarget:
    return InputPlanTarget(
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="1.0",
        planner_version=INFERENCE_INPUT_PLANNER_VERSION,
        capability_snapshot_id=_uuid(9_000),
        capability_snapshot_sha256=_digest("capability"),
    )


def _prompt_output() -> PromptOutputContract:
    return PromptOutputContract(
        prompt_version="prompt-v1",
        prompt_sha256=_digest("prompt"),
        rendered_message_sha256=_digest("rendered-prompt"),
        provider_response_schema_sha256=_digest("provider-claim-schema"),
        enriched_domain_schema_sha256=_digest("enriched-output-schema"),
        protocol_mode="json-schema",
        tool_mode="none",
    )


def _preparer() -> InputPlanPreparer:
    return InputPlanPreparer(
        InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION),
        ProviderRenderingPolicy(
            version="render-v1",
            transform_policy_version="identity-v1",
            idempotency_policy_version="idempotency-v1",
            reduction_policy="ordered-claims-v1",
            reduction_policy_version="1.0",
            input_tokens_per_item=2,
            fixed_input_tokens_per_part=1,
            accepted_media_types=("image/png",),
        ),
    )


def test_materialized_package_prepares_complete_catalog_and_explicit_call_parts() -> None:
    materialized = _materialize()
    before = materialized.package
    plan = _preparer().prepare(
        packages=(materialized,),
        task=VisionTask.QA_DENSE,
        request_catalog_id=_uuid(9_001),
        input_plan_id=_uuid(9_002),
        target=_input_target(),
        prompt_output=_prompt_output(),
        applicable_limits=ApplicableProviderLimits(
            max_images_per_request=3,
            max_pixels_per_image=320 * 180,
            max_payload_bytes_per_request=1_000,
            max_input_tokens_per_request=7,
        ),
        created_at="2026-07-19T03:00:00Z",
    )

    assert materialized.package == before
    assert plan.subject.packages[0].package_id == materialized.package.package_id
    assert plan.subject.packages[0].manifest_bytes_sha256 == (materialized.package_manifest_sha256)
    assert tuple(camera.camera_id for camera in plan.request_catalog.packages[0].cameras) == (
        CAMERA_IDS
    )
    assert plan.request_catalog.packages[0].cameras[-1].frames == ()
    assert tuple(item.provider_item_ordinal for item in plan.rendered_items) == tuple(range(10))
    assert all(item.transform.operation is TransformOperation.NONE for item in plan.rendered_items)
    assert tuple(
        (
            part.start_item_ordinal,
            part.end_item_ordinal_exclusive,
            part.measured_input_tokens,
        )
        for part in plan.call_plan.parts
    ) == ((0, 3, 7), (3, 6, 7), (6, 9, 7), (9, 10, 3))
    assert (
        plan.prompt_output.provider_response_schema_sha256
        != plan.prompt_output.enriched_domain_schema_sha256
    )


def test_staged_preparation_derives_prompt_tokens_without_an_input_plan_cycle() -> None:
    materialized = _materialize()
    preparer = _preparer()
    limits = ApplicableProviderLimits(
        max_images_per_request=20,
        max_pixels_per_image=320 * 180,
        max_payload_bytes_per_request=20_000,
        max_input_tokens_per_request=100,
    )
    prepared = preparer.prepare_rendering(
        packages=(materialized,),
        task=VisionTask.QA_DENSE,
        request_catalog_id=_uuid(9_010),
        applicable_limits=limits,
        created_at="2026-07-19T03:00:00Z",
    )
    prompt_entries = ProviderReferenceCatalog.derive_entries(
        request_catalog_sha256=prepared.request_catalog.semantic_sha256,
        rendered_items=prepared.rendered_items,
        token_policy_version="token-v1",
    )
    rendered_message_sha256 = semantic_sha256(
        {
            "task": prepared.task.value,
            "evidence_tokens": [item.correlation_token for item in prompt_entries],
        }
    )
    prompt = _prompt_output().model_copy(
        update={"rendered_message_sha256": rendered_message_sha256}
    )
    plan = preparer.finalize(
        prepared=prepared,
        input_plan_id=_uuid(9_011),
        target=_input_target(),
        prompt_output=prompt,
        created_at="2026-07-19T03:00:00Z",
    )
    catalog = ProviderReferenceCatalog.build(
        input_plan=plan,
        reference_catalog_id=_uuid(9_012),
        token_policy_version="token-v1",
        created_at="2026-07-19T03:00:00Z",
    )

    assert catalog.entries == prompt_entries
    assert plan.prompt_output.rendered_message_sha256 == rendered_message_sha256


def test_input_preparation_rejects_an_unsplittable_single_frame() -> None:
    materialized = _materialize()
    with pytest.raises(InputPreparationError, match="max_pixels_per_image"):
        _preparer().prepare(
            packages=(materialized,),
            task=VisionTask.QA_DENSE,
            request_catalog_id=_uuid(9_003),
            input_plan_id=_uuid(9_004),
            target=_input_target(),
            prompt_output=_prompt_output(),
            applicable_limits=ApplicableProviderLimits(
                max_images_per_request=1,
                max_pixels_per_image=1,
                max_payload_bytes_per_request=1_000,
                max_input_tokens_per_request=10,
            ),
            created_at="2026-07-19T03:00:00Z",
        )


def test_input_preparation_wraps_render_factory_failures_with_frame_context() -> None:
    materialized = _materialize()

    def fail_render(*_args: object) -> object:
        raise RuntimeError("fixture renderer unavailable")

    with pytest.raises(
        InputPreparationError,
        match=r"camera cam_01, frame 0: fixture renderer unavailable",
    ):
        _preparer().prepare(
            packages=(materialized,),
            task=VisionTask.QA_DENSE,
            request_catalog_id=_uuid(9_005),
            input_plan_id=_uuid(9_006),
            target=_input_target(),
            prompt_output=_prompt_output(),
            applicable_limits=ApplicableProviderLimits(
                max_images_per_request=10,
                max_pixels_per_image=320 * 180,
                max_payload_bytes_per_request=10_000,
                max_input_tokens_per_request=100,
            ),
            created_at="2026-07-19T03:00:00Z",
            rendered_item_factory=fail_render,  # type: ignore[arg-type]
        )


def test_rendering_policy_requires_canonical_media_type_order() -> None:
    with pytest.raises(ValueError, match="unique and lexically ordered"):
        ProviderRenderingPolicy(
            version="render-v1",
            transform_policy_version="identity-v1",
            idempotency_policy_version="idempotency-v1",
            reduction_policy="ordered-claims-v1",
            reduction_policy_version="1.0",
            accepted_media_types=("image/webp", "image/png"),
        )
