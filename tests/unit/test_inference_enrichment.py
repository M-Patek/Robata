from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.schema_registry import SchemaRegistry, SchemaValidationError
from robata.inference.adapter import JsonSchemaRef
from robata.inference.enrichment import (
    ENRICHED_OUTPUT_SCHEMA_ID,
    PROVIDER_CLAIM_SCHEMA_ID,
    EnrichmentAuthorityContext,
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    ProviderClaimEnricher,
    ProviderClaimEnrichmentError,
    ProviderClaimInterval,
    ProviderClaimKind,
    ProviderClaimPayload,
    ProviderObservation,
    ProviderReferenceCatalog,
    ProviderTaskClaim,
    RawProviderResponseArtifact,
    SelectedAttemptOutput,
    enrichment_logical_digest,
    orchestrator_enriched_output_projection,
)
from robata.inference.input_plan import (
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
from robata.inference.models import VisionTask

NOW = "2026-07-19T12:00:00Z"
TASK = VisionTask.ACTION_EVIDENCE


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _schema_ref(registry: SchemaRegistry, schema_id: str) -> JsonSchemaRef:
    registered = registry.resolve_version(schema_id, "1.0.0").ref
    return JsonSchemaRef(
        schema_id=registered.schema_id,
        version=registered.version,
        artifact_id=registered.artifact_id,
        sha256=registered.sha256,
    )


def _input_plan(
    provider_schema: JsonSchemaRef,
    enriched_schema: JsonSchemaRef,
) -> InferenceInputPlan:
    planner = InferenceInputPlanner("planner-1")
    cameras = tuple(
        CatalogCamera(
            camera_id=camera_id,
            ordinal=ordinal,
            frames=(
                CatalogFrame(
                    frame_id=_uuid(100 + ordinal),
                    ordinal=0,
                    aligned_timestamp_ns=10 + ordinal,
                    source_timestamp_ns=1_000 + ordinal,
                    source_artifact_uri=f"object://source/cam-{ordinal}",
                    source_artifact_sha256=_digest(200 + ordinal),
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
        package_id=_uuid(300),
        ordinal=0,
        semantic_content_sha256=_digest(301),
        manifest_bytes_sha256=_digest(302),
        cameras=cameras,
    )
    catalog = planner.build_request_catalog(
        request_catalog_id=_uuid(303),
        task=TASK,
        packages=(package,),
        created_at=NOW,
    )
    items = tuple(
        RenderedProviderItem(
            provider_item_ordinal=ordinal,
            package_id=package.package_id,
            package_ordinal=package.ordinal,
            camera_id=camera.camera_id,
            camera_ordinal=camera.ordinal,
            frame_id=frame.frame_id,
            frame_ordinal=frame.ordinal,
            aligned_timestamp_ns=frame.aligned_timestamp_ns,
            source_timestamp_ns=frame.source_timestamp_ns,
            source_artifact_sha256=frame.source_artifact_sha256,
            artifact=RenderedArtifact(
                artifact_id=_uuid(400 + ordinal),
                uri=f"object://rendered/cam-{ordinal}",
                sha256=frame.source_artifact_sha256,
                byte_count=frame.source_artifact_bytes,
                media_type=frame.media_type,
                encoding=frame.encoding,
                width=frame.width,
                height=frame.height,
            ),
            transform=FrameTransform.create(
                operation=TransformOperation.NONE,
                policy_version="render-1",
            ),
        )
        for ordinal, (camera, frame) in enumerate(
            (camera, frame) for camera in cameras for frame in camera.frames
        )
    )
    return planner.build(
        input_plan_id=_uuid(500),
        created_at=NOW,
        request_catalog=catalog,
        target=InputPlanTarget(
            provider="local-fake",
            model_name="vision-model",
            model_version="1.0",
            adapter_version="adapter-1",
            planner_version="planner-1",
            capability_snapshot_id=_uuid(501),
            capability_snapshot_sha256=_digest(502),
        ),
        rendered_items=items,
        prompt_output=PromptOutputContract(
            prompt_version="prompt-1",
            prompt_sha256=_digest(503),
            rendered_message_sha256=_digest(504),
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
        idempotency_policy_version="idempotency-1",
        reduction_policy="ordered-claim-union",
        reduction_policy_version="reduce-1",
    )


def _payload(reference_catalog: ProviderReferenceCatalog) -> ProviderClaimPayload:
    return ProviderClaimPayload(
        claims=tuple(
            ProviderTaskClaim(
                claim_ordinal=ordinal,
                kind=ProviderClaimKind.ACTION_OBSERVATION,
                package_ordinal=0,
                camera_ordinal=ordinal,
                interval=ProviderClaimInterval(start_ns=0, end_ns=100),
                label="grasp",
                observation=ProviderObservation.SUPPORTING,
                evidence_tokens=(reference_catalog.entries[ordinal].correlation_token,),
                model_reported_score=0.8 if ordinal == 0 else None,
                conflict_codes=(),
            )
            for ordinal in range(6)
        ),
        abstained=False,
    )


def _parsed(
    *,
    payload: ProviderClaimPayload,
    provider_schema: JsonSchemaRef,
    row_offset: int = 0,
) -> ParsedProviderClaimArtifact:
    raw = RawProviderResponseArtifact.from_bytes(
        data=canonical_json_bytes(payload.model_dump(mode="json")),
        artifact_id=_uuid(600 + row_offset),
        media_type="application/json",
        provider_request_id=f"provider-request-{row_offset}",
        inference_id=_uuid(601),
        provider="local-fake",
        model_name="vision-model",
        model_version="1.0",
        created_at=NOW,
    )
    return ParsedProviderClaimArtifact.create(
        artifact_id=_uuid(700 + row_offset),
        raw_response=raw,
        provider_claim_schema=provider_schema,
        task=TASK,
        payload=payload,
        parser_version="parser-1",
        created_at=NOW,
    )


@dataclass(frozen=True)
class _Fixture:
    registry: SchemaRegistry
    provider_schema: JsonSchemaRef
    enriched_schema: JsonSchemaRef
    plan: InferenceInputPlan
    reference_catalog: ProviderReferenceCatalog
    payload: ProviderClaimPayload
    parsed: ParsedProviderClaimArtifact
    selected: SelectedAttemptOutput
    authority: EnrichmentAuthorityContext


def _fixture() -> _Fixture:
    registry = SchemaRegistry()
    provider_schema = _schema_ref(registry, PROVIDER_CLAIM_SCHEMA_ID)
    enriched_schema = _schema_ref(registry, ENRICHED_OUTPUT_SCHEMA_ID)
    plan = _input_plan(provider_schema, enriched_schema)
    reference_catalog = ProviderReferenceCatalog.build(
        input_plan=plan,
        reference_catalog_id=_uuid(800),
        token_policy_version="token-1",
        created_at=NOW,
    )
    payload = _payload(reference_catalog)
    parsed = _parsed(payload=payload, provider_schema=provider_schema)
    selected = SelectedAttemptOutput.create(parsed)
    authority = EnrichmentAuthorityContext(
        recording_identity=_digest(801),
        mcap_id=_uuid(802),
        camera_mapping_run_id=_uuid(803),
        alignment_id=_uuid(804),
        inference_id=selected.inference_id,
        logical_invocation_id=_uuid(805),
        prompt_version="prompt-1",
        prompt_artifact_id=_uuid(806),
        prompt_sha256=_digest(503),
        work_node_type="INFERENCE_ENRICHMENT",
        work_node_logical_key=f"inference-work:{_digest(807)}",
    )
    return _Fixture(
        registry=registry,
        provider_schema=provider_schema,
        enriched_schema=enriched_schema,
        plan=plan,
        reference_catalog=reference_catalog,
        payload=payload,
        parsed=parsed,
        selected=selected,
        authority=authority,
    )


def _enrich(
    fixture: _Fixture,
    *,
    parsed: ParsedProviderClaimArtifact | None = None,
    selected: SelectedAttemptOutput | None = None,
    enriched_schema: JsonSchemaRef | None = None,
) -> OrchestratorEnrichedOutput:
    return ProviderClaimEnricher(fixture.registry).enrich(
        input_plan=fixture.plan,
        reference_catalog=fixture.reference_catalog,
        parsed_claims=parsed or fixture.parsed,
        selected_attempt=selected or fixture.selected,
        authority=fixture.authority,
        enriched_output_schema=enriched_schema or fixture.enriched_schema,
        enrichment_policy_version="enrichment-1",
        artifact_id=_uuid(900),
        created_at=NOW,
    )


def test_enriches_untrusted_claims_with_authoritative_lineage() -> None:
    fixture = _fixture()
    output = _enrich(fixture)

    assert fixture.parsed.raw_response.artifact_id != fixture.parsed.artifact_id
    assert fixture.parsed.artifact_id != output.artifact_id
    assert len(output.claims) == 6
    first = output.claims[0]
    assert first.package_id == fixture.plan.request_catalog.packages[0].package_id
    assert first.camera_id is CAMERA_IDS[0]
    assert first.evidence[0].frame_id == (
        fixture.plan.request_catalog.packages[0].cameras[0].frames[0].frame_id
    )
    assert first.evidence[0].source_artifact_uri == "object://source/cam-0"
    assert first.model_reported_confidence is not None
    assert first.model_reported_confidence.kind == "MODEL_REPORTED_UNCALIBRATED"
    assert first.model_reported_confidence.semantics == "provider_self_report"
    assert first.model_reported_confidence.producer_id == fixture.selected.inference_id

    expected_key_digest = enrichment_logical_digest(
        selected_attempt_output_sha256=fixture.selected.output_sha256,
        request_catalog_sha256=fixture.plan.request_catalog.semantic_sha256,
        target_schema_sha256=fixture.enriched_schema.sha256,
        enrichment_policy_version="enrichment-1",
    )
    assert output.enrichment_logical_key == f"orchestrator-enrichment:{expected_key_digest}"
    assert output.semantic_sha256 == semantic_sha256(
        orchestrator_enriched_output_projection(output)
    )
    registered = fixture.registry.resolve_version(ENRICHED_OUTPUT_SCHEMA_ID, "1.0.0")
    assert fixture.registry.validate_pinned(
        registered.ref, output.model_dump(mode="json")
    ) == output.model_dump(mode="json")


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "candidate_event_id",
        "inference_id",
        "package_id",
        "prompt_sha256",
        "probability",
        "calibrated_confidence",
    ],
)
def test_provider_boundary_rejects_authoritative_or_trusted_fields(
    forbidden_field: str,
) -> None:
    fixture = _fixture()
    payload = fixture.payload.model_dump(mode="json")
    payload["claims"][0][forbidden_field] = _uuid(999)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderClaimPayload.model_validate(payload)
    provider_ref = fixture.registry.resolve_version(PROVIDER_CLAIM_SCHEMA_ID, "1.0.0").ref
    with pytest.raises(SchemaValidationError, match="additional property"):
        fixture.registry.validate_pinned(provider_ref, payload)


def test_provider_tokens_fail_closed_when_missing_duplicate_or_out_of_catalog() -> None:
    fixture = _fixture()
    claim = fixture.payload.claims[0]
    with pytest.raises(ValidationError, match="at least one evidence token"):
        ProviderTaskClaim.model_validate({**claim.model_dump(mode="python"), "evidence_tokens": []})
    with pytest.raises(ValidationError, match="must be unique"):
        ProviderTaskClaim.model_validate(
            {
                **claim.model_dump(mode="python"),
                "evidence_tokens": [
                    claim.evidence_tokens[0],
                    claim.evidence_tokens[0],
                ],
            }
        )

    forged_payload = fixture.payload.model_dump(mode="python")
    forged_payload["claims"][0]["evidence_tokens"] = [f"ref:{'f' * 64}"]
    parsed = _parsed(
        payload=ProviderClaimPayload.model_validate(forged_payload),
        provider_schema=fixture.provider_schema,
        row_offset=10,
    )
    with pytest.raises(ProviderClaimEnrichmentError, match="outside the request catalog"):
        _enrich(fixture, parsed=parsed, selected=SelectedAttemptOutput.create(parsed))


def test_camera_coverage_and_task_schema_bindings_are_fail_closed() -> None:
    fixture = _fixture()
    incomplete = ProviderClaimPayload(
        claims=fixture.payload.claims[:-1],
        abstained=False,
    )
    parsed = _parsed(
        payload=incomplete,
        provider_schema=fixture.provider_schema,
        row_offset=20,
    )
    with pytest.raises(ProviderClaimEnrichmentError, match="cover each"):
        _enrich(fixture, parsed=parsed, selected=SelectedAttemptOutput.create(parsed))

    forged_schema = fixture.enriched_schema.model_copy(update={"sha256": _digest(9999)})
    with pytest.raises(ProviderClaimEnrichmentError, match="not bound by input plan"):
        _enrich(fixture, enriched_schema=forged_schema)


def test_enriched_identity_tampering_is_rejected() -> None:
    output = _enrich(_fixture())
    payload = output.model_dump(mode="python")
    payload["claims"][0]["claim_id"] = _uuid(9999)
    with pytest.raises(ValidationError, match="claim ID is inconsistent"):
        OrchestratorEnrichedOutput.model_validate(payload)

    payload = output.model_dump(mode="python")
    payload["semantic_sha256"] = _digest(9999)
    with pytest.raises(ValidationError, match="semantic_sha256 is inconsistent"):
        OrchestratorEnrichedOutput.model_validate(payload)


def test_artifact_locator_changes_do_not_change_selected_output_or_enrichment_key() -> None:
    fixture = _fixture()
    second_parsed = _parsed(
        payload=fixture.payload,
        provider_schema=fixture.provider_schema,
        row_offset=50,
    )
    second_selected = SelectedAttemptOutput.create(second_parsed)

    assert second_parsed.semantic_sha256 == fixture.parsed.semantic_sha256
    assert second_selected.output_sha256 == fixture.selected.output_sha256
    assert (
        _enrich(
            fixture,
            parsed=second_parsed,
            selected=second_selected,
        ).enrichment_logical_key
        == _enrich(fixture).enrichment_logical_key
    )
