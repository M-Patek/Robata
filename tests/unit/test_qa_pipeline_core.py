from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from robata.contracts import CameraId, NanosecondInterval
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.pipeline import (
    CameraQAClaim,
    CameraQAResult,
    CameraQAStatus,
    RecordingQAStatus,
)
from robata.contracts.schema_registry import SchemaRegistry
from robata.contracts.temporal import (
    PackageLineage,
    SplitReason,
    TemporalPackageSet,
    TemporalPackageSetMember,
    compute_member_manifest_sha256,
    compute_split_plan_digest,
    derive_package_set_id,
    derive_split_group_id,
)
from robata.inference.adapter import JsonSchemaRef
from robata.inference.enrichment import (
    ENRICHED_OUTPUT_SCHEMA_ID,
    ENRICHED_OUTPUT_SCHEMA_VERSION,
    PROVIDER_CLAIM_SCHEMA_ID,
    EnrichmentAuthorityContext,
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    ProviderClaimEnricher,
    ProviderClaimInterval,
    ProviderClaimKind,
    ProviderClaimPayload,
    ProviderObservation,
    ProviderReferenceCatalog,
    ProviderTaskClaim,
    RawProviderResponseArtifact,
    SelectedAttemptOutput,
    orchestrator_enriched_output_projection,
)
from robata.inference.input_plan import (
    INFERENCE_INPUT_PLANNER_VERSION,
    ApplicableProviderLimits,
    CallPartSpec,
    CatalogCamera,
    CatalogFrame,
    CatalogPackage,
    FrameTransform,
    InferenceInputPlan,
    InferenceInputPlanner,
    InputPlanTarget,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    TransformOperation,
)
from robata.inference.models import (
    InferenceAttemptSelection,
    VisionTask,
    inference_attempt_selection_logical_key,
)
from robata.qa_pipeline.aggregate import QAAggregationPolicy, QAAggregator
from robata.qa_pipeline.coarse import (
    CoarseQAProjectionError,
    CoarseQAProjector,
    CoarseQAStatus,
)
from robata.qa_pipeline.suspicion_reducer import (
    SuspiciousInterval,
    SuspiciousIntervalReducer,
)

NOW = "2026-07-20T12:00:00Z"


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:test:{label}"))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _camera_result(
    camera_id: CameraId,
    status: CameraQAStatus,
    *,
    score: float | None = None,
    mcap_id: str | None = None,
    start_ns: int = 0,
    end_ns: int = 10_000,
) -> CameraQAResult:
    return CameraQAResult(
        qa_result_id=_id(f"qa:{camera_id.value}:{status.value}:{start_ns}:{end_ns}"),
        mcap_id=mcap_id or _id("mcap"),
        package_id=_id(f"package:{camera_id.value}:{start_ns}:{end_ns}"),
        inference_id=_id(f"inference:{camera_id.value}:{start_ns}:{end_ns}"),
        camera_id=camera_id,
        claim=CameraQAClaim(
            camera_id=camera_id,
            observed_interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
            status=status,
            issues=(),
            reported_score=score,
            frame_ordinals=(),
        ),
        evidence_frame_ids=(),
    )


def test_qa_aggregate_is_canonical_incomplete_and_non_promotable() -> None:
    statuses = {
        CameraId.CAM_01: CameraQAStatus.GOOD,
        CameraId.CAM_02: CameraQAStatus.GOOD,
        CameraId.CAM_03: CameraQAStatus.GOOD,
        CameraId.CAM_04: CameraQAStatus.GOOD,
        CameraId.CAM_05: CameraQAStatus.GOOD,
        CameraId.CAM_06: CameraQAStatus.INCOMPLETE,
    }
    results = [
        _camera_result(camera_id, statuses[camera_id], score=index / 10)
        for index, camera_id in enumerate(reversed(tuple(CameraId)), start=1)
    ]

    aggregate = QAAggregator().aggregate_camera_results(results)

    assert aggregate.overall_status is RecordingQAStatus.INCOMPLETE
    assert aggregate.usable_camera_count == 5
    assert aggregate.camera_result_ids == tuple(
        next(result.qa_result_id for result in results if result.camera_id is camera_id)
        for camera_id in CameraId
    )
    assert aggregate.model_score == pytest.approx(0.35)
    assert aggregate.deterministic_quality is None
    assert aggregate.policy_version == "local-development-v1"
    assert aggregate.promotion_eligible is False


def test_qa_aggregate_rejects_mixed_recordings_and_scopes() -> None:
    results = [_camera_result(camera_id, CameraQAStatus.GOOD) for camera_id in CameraId]
    results[-1] = _camera_result(
        CameraId.CAM_06,
        CameraQAStatus.GOOD,
        mcap_id=_id("other-mcap"),
    )
    with pytest.raises(ValueError, match="one MCAP"):
        QAAggregator().aggregate_camera_results(results)

    results[-1] = _camera_result(
        CameraId.CAM_06,
        CameraQAStatus.GOOD,
        start_ns=1,
    )
    with pytest.raises(ValueError, match="exact scope"):
        QAAggregator().aggregate_camera_results(results)


def test_unresolved_qa_policy_cannot_claim_promotion() -> None:
    with pytest.raises(ValidationError, match="cannot be promotable"):
        QAAggregationPolicy(
            version="unapproved-v1",
            degraded_min_usable=4,
            status_quality={status: 0.0 for status in CameraQAStatus},
            promotion_eligible=True,
        )


def test_suspicious_reduction_is_cross_camera_deterministic_and_clipped() -> None:
    intervals = (
        SuspiciousInterval(
            start_ns=100,
            end_ns=200,
            camera_id=CameraId.CAM_02,
            issue_type="BLUR",
            confidence=0.8,
        ),
        SuspiciousInterval(
            start_ns=210,
            end_ns=300,
            camera_id=CameraId.CAM_01,
            issue_type="OCCLUSION",
            confidence=0.9,
        ),
        SuspiciousInterval(
            start_ns=900,
            end_ns=980,
            camera_id=CameraId.CAM_03,
            issue_type="EXPOSURE",
            confidence=0.7,
        ),
    )
    reducer = SuspiciousIntervalReducer()

    first = reducer.reduce(
        intervals,
        padding_ns=50,
        max_gap_ns=20,
        recording_duration_ns=1_000,
        policy_version="qa-reduce-v2",
    )
    replay = reducer.reduce(
        tuple(reversed(intervals)),
        padding_ns=50,
        max_gap_ns=20,
        recording_duration_ns=1_000,
        policy_version="qa-reduce-v2",
    )

    assert first == replay
    assert [(item.start_ns, item.end_ns) for item in first] == [(50, 350), (850, 1_000)]
    assert first[0].cameras == (CameraId.CAM_01, CameraId.CAM_02)
    assert first[0].merged_from_count == 2
    assert len({source.interval_id for source in first[0].source_intervals}) == 2
    assert first[0].reduction_policy_version.max_gap_ns == 20
    assert first[0].reduction_policy_version.clip_to_recording_bounds is True


def test_suspicious_interval_and_reducer_fail_closed() -> None:
    with pytest.raises(ValidationError, match="must be non-empty"):
        SuspiciousInterval(
            start_ns=5,
            end_ns=5,
            camera_id=CameraId.CAM_01,
            issue_type="BLUR",
            confidence=0.5,
        )
    with pytest.raises(ValueError, match="must be nonnegative"):
        SuspiciousIntervalReducer().reduce((), padding_ns=-1)


def _schema_ref(registry: SchemaRegistry, schema_id: str) -> JsonSchemaRef:
    version = ENRICHED_OUTPUT_SCHEMA_VERSION if schema_id == ENRICHED_OUTPUT_SCHEMA_ID else "1.0.0"
    registered = registry.resolve_version(schema_id, version).ref
    return JsonSchemaRef(
        schema_id=registered.schema_id,
        version=registered.version,
        artifact_id=registered.artifact_id,
        sha256=registered.sha256,
    )


def _coarse_package_set() -> TemporalPackageSet:
    lineage = PackageLineage(
        source_content_sha256=_digest(1),
        window_semantic_sha256=_digest(2),
        camera_mapping_semantic_sha256=_digest(3),
        alignment_semantic_sha256=_digest(4),
        sampling_plan_sha256=_digest(5),
    )
    member = TemporalPackageSetMember(
        package_id=_id("coarse-package"),
        ordinal=0,
        part_count=1,
        requested_start_ns=0,
        requested_end_ns=100,
        start_ns=0,
        end_ns=100,
        overlap_before_ns=0,
        overlap_after_ns=0,
        package_semantic_content_sha256=_digest(6),
        package_manifest_sha256=_digest(7),
    )
    members = (member,)
    split_digest = compute_split_plan_digest(
        lineage=lineage,
        split_reason=SplitReason.NONE,
        split_policy_version="coarse-split-v1",
        members=members,
    )
    split_group_id = derive_split_group_id(
        lineage=lineage,
        requested_start_ns=0,
        requested_end_ns=100,
        start_ns=0,
        end_ns=100,
        split_plan_digest=split_digest,
    )
    member_manifest = compute_member_manifest_sha256(members)
    return TemporalPackageSet(
        schema_version="1.0",
        package_set_id=derive_package_set_id(
            split_group_id=split_group_id,
            member_manifest_sha256=member_manifest,
            reduction_policy_version="coarse-package-reduce-v1",
        ),
        split_group_id=split_group_id,
        mcap_id=_id("coarse-mcap"),
        window_id=_id("coarse-window"),
        camera_mapping_run_id=_id("coarse-mapping"),
        alignment_id=_id("coarse-alignment"),
        lineage=lineage,
        requested_start_ns=0,
        requested_end_ns=100,
        start_ns=0,
        end_ns=100,
        split_reason=SplitReason.NONE,
        split_policy_version="coarse-split-v1",
        split_plan_digest=split_digest,
        members=members,
        member_manifest_sha256=member_manifest,
        reduction_policy_version="coarse-package-reduce-v1",
        created_at=NOW,
    )


def _coarse_input_plan(
    package_set: TemporalPackageSet,
    provider_schema: JsonSchemaRef,
    enriched_schema: JsonSchemaRef,
) -> InferenceInputPlan:
    member = package_set.members[0]
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    cameras = tuple(
        CatalogCamera(
            camera_id=camera_id,
            ordinal=ordinal,
            frames=(
                CatalogFrame(
                    frame_id=_id(f"coarse-frame-{ordinal}"),
                    ordinal=0,
                    aligned_timestamp_ns=10 + ordinal,
                    source_timestamp_ns=1_000 + ordinal,
                    source_artifact_uri=f"object://coarse/source/{ordinal}",
                    source_artifact_sha256=_digest(100 + ordinal),
                    source_artifact_bytes=100,
                    media_type="image/png",
                    encoding="png",
                    width=64,
                    height=48,
                ),
            ),
        )
        for ordinal, camera_id in enumerate(CAMERA_IDS)
    )
    package = CatalogPackage(
        package_id=member.package_id,
        ordinal=member.ordinal,
        semantic_content_sha256=member.package_semantic_content_sha256,
        manifest_bytes_sha256=member.package_manifest_sha256,
        cameras=cameras,
    )
    catalog = planner.build_request_catalog(
        request_catalog_id=_id("coarse-catalog"),
        task=VisionTask.QA_COARSE,
        packages=(package,),
        created_at=NOW,
    )
    rendered_items = tuple(
        RenderedProviderItem(
            provider_item_ordinal=ordinal,
            package_id=package.package_id,
            package_ordinal=package.ordinal,
            camera_id=camera.camera_id,
            camera_ordinal=camera.ordinal,
            frame_id=camera.frames[0].frame_id,
            frame_ordinal=0,
            aligned_timestamp_ns=camera.frames[0].aligned_timestamp_ns,
            source_timestamp_ns=camera.frames[0].source_timestamp_ns,
            source_artifact_sha256=camera.frames[0].source_artifact_sha256,
            artifact=RenderedArtifact(
                artifact_id=_id(f"coarse-rendered-{ordinal}"),
                uri=f"object://coarse/rendered/{ordinal}",
                sha256=camera.frames[0].source_artifact_sha256,
                byte_count=100,
                media_type="image/png",
                encoding="png",
                width=64,
                height=48,
            ),
            transform=FrameTransform.create(
                operation=TransformOperation.NONE,
                policy_version="coarse-render-v1",
            ),
        )
        for ordinal, camera in enumerate(cameras)
    )
    return planner.build(
        input_plan_id=_id("coarse-input-plan"),
        created_at=NOW,
        request_catalog=catalog,
        target=InputPlanTarget(
            provider="local-fixture",
            model_name="fixture-vision",
            model_version="1.0",
            adapter_version="fixture-adapter-v1",
            planner_version=INFERENCE_INPUT_PLANNER_VERSION,
            capability_snapshot_id=_id("coarse-capability"),
            capability_snapshot_sha256=_digest(200),
        ),
        rendered_items=rendered_items,
        prompt_output=PromptOutputContract(
            prompt_version="coarse-prompt-v1",
            prompt_sha256=_digest(201),
            rendered_message_sha256=_digest(202),
            provider_response_schema_sha256=provider_schema.sha256,
            enriched_domain_schema_sha256=enriched_schema.sha256,
            protocol_mode="json-schema",
            tool_mode="none",
        ),
        applicable_limits=ApplicableProviderLimits(
            max_images_per_request=6,
            max_pixels_per_image=64 * 48,
            max_payload_bytes_per_request=1_000,
            max_input_tokens_per_request=100,
        ),
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=6,
                measured_input_tokens=50,
            ),
        ),
        idempotency_policy_version="coarse-idempotency-v1",
        reduction_policy="ordered-claim-union",
        reduction_policy_version="coarse-claim-reduce-v1",
    )


def _coarse_enriched_output(
    package_set: TemporalPackageSet,
    plan: InferenceInputPlan,
    registry: SchemaRegistry,
    provider_schema: JsonSchemaRef,
    enriched_schema: JsonSchemaRef,
    statuses: dict[CameraId, ProviderObservation],
) -> OrchestratorEnrichedOutput:
    reference_catalog = ProviderReferenceCatalog.build(
        input_plan=plan,
        reference_catalog_id=_id("coarse-reference-catalog"),
        token_policy_version="coarse-token-v1",
        created_at=NOW,
    )
    payload = ProviderClaimPayload(
        claims=tuple(
            ProviderTaskClaim(
                claim_ordinal=ordinal,
                kind=ProviderClaimKind.QA_OBSERVATION,
                package_ordinal=0,
                camera_ordinal=ordinal,
                interval=ProviderClaimInterval(start_ns=0, end_ns=100),
                label="coarse-screen",
                observation=statuses[camera_id],
                evidence_tokens=(reference_catalog.entries[ordinal].correlation_token,),
                model_reported_score=None,
                conflict_codes=(),
            )
            for ordinal, camera_id in enumerate(CAMERA_IDS)
        ),
        abstained=False,
    )
    inference_id = _id("coarse-inference")
    raw = RawProviderResponseArtifact.from_bytes(
        data=canonical_json_bytes(payload.model_dump(mode="json")),
        artifact_id=_id("coarse-raw"),
        media_type="application/json",
        provider_request_id="coarse-provider-request",
        inference_id=inference_id,
        provider="local-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        created_at=NOW,
    )
    parsed = ParsedProviderClaimArtifact.create(
        artifact_id=_id("coarse-parsed"),
        raw_response=raw,
        provider_claim_schema=provider_schema,
        task=VisionTask.QA_COARSE,
        payload=payload,
        parser_version="coarse-parser-v1",
        created_at=NOW,
    )
    logical_invocation_id = _id("coarse-logical-invocation")
    selection_policy = "coarse-selection-v1"
    selection = InferenceAttemptSelection(
        schema_version="1.0",
        selection_id=_id("coarse-selection"),
        inference_id=inference_id,
        logical_invocation_id=logical_invocation_id,
        policy_version=selection_policy,
        selection_reason="FIRST_SCHEMA_VALID_SUCCESS",
        selection_decision_logical_key=inference_attempt_selection_logical_key(
            logical_invocation_id=logical_invocation_id,
            policy_version=selection_policy,
        ),
        selected_at=NOW,
    )
    selected = SelectedAttemptOutput.create(parsed, selection)
    authority = EnrichmentAuthorityContext(
        recording_identity=_digest(203),
        mcap_id=package_set.mcap_id,
        camera_mapping_run_id=package_set.camera_mapping_run_id,
        alignment_id=package_set.alignment_id,
        inference_id=inference_id,
        logical_invocation_id=logical_invocation_id,
        prompt_version=plan.prompt_output.prompt_version,
        prompt_artifact_id=_id("coarse-prompt"),
        prompt_sha256=plan.prompt_output.prompt_sha256,
        work_node_type="INFERENCE_ENRICHMENT",
        work_node_logical_key=f"inference-work:{_digest(204)}",
    )
    return ProviderClaimEnricher(registry).enrich(
        input_plan=plan,
        reference_catalog=reference_catalog,
        parsed_claims=parsed,
        selected_attempt=selected,
        authority=authority,
        enriched_output_schema=enriched_schema,
        enrichment_policy_version="coarse-enrichment-v1",
        artifact_id=_id("coarse-enriched"),
        created_at=NOW,
    )


def _coarse_fixture(
    statuses: dict[CameraId, ProviderObservation] | None = None,
) -> tuple[TemporalPackageSet, InferenceInputPlan, OrchestratorEnrichedOutput]:
    registry = SchemaRegistry()
    provider_schema = _schema_ref(registry, PROVIDER_CLAIM_SCHEMA_ID)
    enriched_schema = _schema_ref(registry, ENRICHED_OUTPUT_SCHEMA_ID)
    package_set = _coarse_package_set()
    plan = _coarse_input_plan(package_set, provider_schema, enriched_schema)
    output = _coarse_enriched_output(
        package_set,
        plan,
        registry,
        provider_schema,
        enriched_schema,
        statuses or {camera_id: ProviderObservation.GOOD for camera_id in CAMERA_IDS},
    )
    return package_set, plan, output


def _replace_enriched_claims(
    output: OrchestratorEnrichedOutput,
    claims: tuple,
) -> OrchestratorEnrichedOutput:
    provisional = output.model_copy(update={"claims": claims})
    values = provisional.model_dump(mode="python")
    values["semantic_sha256"] = semantic_sha256(
        orchestrator_enriched_output_projection(provisional)
    )
    return OrchestratorEnrichedOutput.model_validate(values, strict=True)


def test_coarse_projector_completes_all_good_without_promoting_claims() -> None:
    package_set, plan, output = _coarse_fixture()

    result = CoarseQAProjector().project(
        package_set=package_set,
        input_plan=plan,
        enriched_outputs=(output,),
    )
    replay = CoarseQAProjector().project(
        package_set=package_set,
        input_plan=plan,
        enriched_outputs=(output,),
    )

    assert result == replay
    assert result.local_status is CoarseQAStatus.COMPLETE
    assert result.complete is True
    assert result.requires_dense is False
    assert result.production_eligible is False
    assert tuple(item.camera_id for item in result.package_camera_results) == CAMERA_IDS
    assert result.package_camera_results[0].claim == output.claims[0]
    assert result.package_camera_results[0].claim.evidence == output.claims[0].evidence
    assert result.source_outputs[0].artifact_id == output.artifact_id


def test_coarse_projector_routes_degraded_or_unusable_to_dense() -> None:
    statuses = {camera_id: ProviderObservation.GOOD for camera_id in CAMERA_IDS}
    statuses[CameraId.CAM_02] = ProviderObservation.DEGRADED
    statuses[CameraId.CAM_05] = ProviderObservation.UNUSABLE
    package_set, plan, output = _coarse_fixture(statuses)

    result = CoarseQAProjector().project(
        package_set=package_set,
        input_plan=plan,
        enriched_outputs=(output,),
    )

    assert result.local_status is CoarseQAStatus.REQUIRES_DENSE
    assert result.complete is False
    assert result.requires_dense is True
    assert result.production_eligible is False


def test_coarse_projector_stops_unknown_as_incomplete() -> None:
    statuses = {camera_id: ProviderObservation.GOOD for camera_id in CAMERA_IDS}
    statuses[CameraId.CAM_04] = ProviderObservation.UNKNOWN
    package_set, plan, output = _coarse_fixture(statuses)

    result = CoarseQAProjector().project(
        package_set=package_set,
        input_plan=plan,
        enriched_outputs=(output,),
    )

    assert result.local_status is CoarseQAStatus.INCOMPLETE
    assert result.complete is False
    assert result.requires_dense is False
    assert result.production_eligible is False


def test_coarse_projector_rejects_missing_camera_and_out_of_package_interval() -> None:
    package_set, plan, output = _coarse_fixture()
    missing_camera = _replace_enriched_claims(output, output.claims[:-1])
    with pytest.raises(CoarseQAProjectionError, match="cover every package/camera"):
        CoarseQAProjector().project(
            package_set=package_set,
            input_plan=plan,
            enriched_outputs=(missing_camera,),
        )

    first = output.claims[0].model_copy(
        update={"interval": ProviderClaimInterval(start_ns=0, end_ns=101)}
    )
    outside_package = _replace_enriched_claims(output, (first, *output.claims[1:]))
    with pytest.raises(CoarseQAProjectionError, match="interval must lie inside"):
        CoarseQAProjector().project(
            package_set=package_set,
            input_plan=plan,
            enriched_outputs=(outside_package,),
        )
