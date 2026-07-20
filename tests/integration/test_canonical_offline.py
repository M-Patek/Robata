from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid5

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical_offline import (
    CanonicalOfflineExecutionPolicy,
    CanonicalOfflinePipeline,
    CanonicalOfflineRunResult,
    CanonicalOfflineRunStatus,
)
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRegistry
from robata.contracts.temporal import PackageLineage
from robata.event_pipeline.identity_registry import (
    EventIdentityPolicyRef,
    EventIdentityRegistryService,
    ExactFingerprintEventIdentityResolver,
    InMemoryEventIdentityRegistryRepository,
    ProductionOutputAdmissionPolicyRef,
)
from robata.inference.adapter import (
    JsonSchemaRef,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.enrichment import (
    ENRICHED_OUTPUT_SCHEMA_ID,
    PROVIDER_CLAIM_SCHEMA_ID,
    ProviderReferenceCatalog,
)
from robata.inference.input_plan import InferenceInputPlanner
from robata.inference.models import (
    ConcurrencyClass,
    InferenceFailure,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    Retryability,
    VisionTask,
)
from robata.inference.offline_fixture import (
    InMemoryRawProviderBytesStore,
    OfflineFixtureResponse,
    OfflineFixtureVisionAdapter,
    ProviderResponseParseCode,
    StrictProviderClaimParser,
)
from robata.inference.orchestrator import InferencePolicy
from robata.inference.preparation import (
    InputPlanPreparer,
    ProviderRenderingPolicy,
)
from robata.sampling.materializer import (
    CanonicalSixCameraFrameIndex,
    OfflineTemporalPackageMaterializer,
)
from robata.sampling.package_set import PackageSetBuilder, sampling_plan_digest
from tests.unit.test_sampling_materializer import (
    _policy as _materialization_policy,
)
from tests.unit.test_sampling_materializer import (
    _resolver as _artifact_resolver,
)
from tests.unit.test_sampling_materializer import (
    _sampling_plan,
    _v2_context,
    _v2_frame_index,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-19T12:00:00Z"
TOKEN_POLICY_VERSION = "provider-token-v1"
PARSER_VERSION = "strict-provider-claim-v1"
REDUCTION_POLICY = "ordered-claims-v1"
REDUCTION_POLICY_VERSION = "1.0"
REQUESTED_INTERVAL = NanosecondInterval(start_ns=0, end_ns=1_000_000_000)


def _uuid(number: int) -> str:
    return str(UUID(int=number))


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _schema_ref(registry: SchemaRegistry, schema_id: str) -> JsonSchemaRef:
    registered = registry.resolve_version(schema_id, "1.0.0").ref
    return JsonSchemaRef(
        schema_id=registered.schema_id,
        version=registered.version,
        artifact_id=registered.artifact_id,
        sha256=registered.sha256,
    )


def _capabilities(*, max_images_per_request: int = 20) -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(1_000),
        snapshot_digest=_digest(f"capabilities:{max_images_per_request}"),
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        supported_tasks=(VisionTask.FUSION_ADJUDICATION,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=max_images_per_request,
        max_pixels_per_image=320 * 180,
        max_payload_bytes=1_000_000,
        max_input_tokens=10_000,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.SERIAL,
        data_handling_policy_version="offline-data-v1",
        observed_at=NOW_TEXT,
    )


def _claim_bytes(request: VisionInferenceRequest, *, abstained: bool = False) -> bytes:
    if abstained:
        return canonical_json_bytes({"claims": [], "abstained": True})
    assert request.input_plan is not None
    entries = ProviderReferenceCatalog.derive_entries(
        request_catalog_sha256=request.input_plan.request_catalog.semantic_sha256,
        rendered_items=request.input_plan.rendered_items,
        token_policy_version=TOKEN_POLICY_VERSION,
    )
    return canonical_json_bytes(
        {
            "claims": [
                {
                    "claim_ordinal": 0,
                    "kind": "FUSION_HYPOTHESIS",
                    "package_ordinal": None,
                    "camera_ordinal": None,
                    "interval": {"start_ns": "200000000", "end_ns": "800000000"},
                    "label": "grasp",
                    "observation": "PROPOSED",
                    "evidence_tokens": [entries[0].correlation_token],
                    "model_reported_score": 0.8,
                    "conflict_codes": [],
                }
            ],
            "abstained": False,
        }
    )


def _failure(
    request: VisionInferenceRequest,
    *,
    retryability: Retryability,
    status: InferenceStatus,
) -> VisionInferenceFailure:
    return VisionInferenceFailure(
        status=status,
        provider_request_id=None,
        provider=request.provider,
        model_name=request.model_name,
        model_version=request.model_version,
        schema_valid=False,
        usage=VisionUsage(input_frames=0, input_images=0),
        latency_ms=0,
        failure=InferenceFailure(
            code="OFFLINE_PROVIDER_FAILURE",
            detail="deterministic fixture failure",
            retryability=retryability,
        ),
    )


class _SequenceEventIdAllocator:
    version = "integration-sequence-v1"

    def __init__(self) -> None:
        self._next = 10_000

    def allocate(self, **_kwargs: object) -> str:
        allocated = _uuid(self._next)
        self._next += 1
        return allocated


class _RawReferenceMismatchAdapter(OfflineFixtureVisionAdapter):
    async def infer(
        self,
        request: VisionInferenceRequest,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        outcome = await super().infer(request)
        if isinstance(outcome, VisionInferenceSuccess):
            return outcome.model_copy(update={"raw_output_artifact_id": _uuid(99_999)})
        return outcome


@dataclass(frozen=True)
class _Harness:
    pipeline: CanonicalOfflinePipeline
    repository: InMemoryEventIdentityRegistryRepository
    raw_store: InMemoryRawProviderBytesStore
    context: AdmittedRecordingContextV2
    frame_index: CanonicalSixCameraFrameIndex


def _harness(
    response_factory: Callable[[VisionInferenceRequest], OfflineFixtureResponse],
    *,
    max_images_per_request: int = 20,
    max_attempts: int = 2,
    mismatch_raw_reference: bool = False,
) -> _Harness:
    registry = SchemaRegistry()
    provider_schema = _schema_ref(registry, PROVIDER_CLAIM_SCHEMA_ID)
    enriched_schema = _schema_ref(registry, ENRICHED_OUTPUT_SCHEMA_ID)
    output_policy = ProductionOutputAdmissionPolicyRef(
        version="fusion-output-admission-v1",
        semantic_sha256=_digest("fusion-output-admission-v1"),
    )
    execution_policy = CanonicalOfflineExecutionPolicy.create(
        policy_version="canonical-offline-v1",
        window_policy_version="root-window-v1",
        token_policy_version=TOKEN_POLICY_VERSION,
        parser_version=PARSER_VERSION,
        enrichment_policy_version="enrichment-v1",
        projector_policy_version="fusion-projector-v1",
        reduction_policy=REDUCTION_POLICY,
        reduction_policy_version=REDUCTION_POLICY_VERSION,
        max_attempts=max_attempts,
        output_admission_policy=output_policy,
    )
    inference_policy = InferencePolicy(
        policy_version="offline-model-policy-v1",
        task=VisionTask.FUSION_ADJUDICATION,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="offline-adapter-v1",
        prompt_version="fusion-prompt-v1",
        prompt_artifact_id=_uuid(1_001),
        prompt_sha256=_digest("fusion-prompt-v1"),
        output_schema=provider_schema,
        enriched_output_schema=enriched_schema,
        generation_config={"temperature": 0},
        timeout_ms=1_000,
        selection_policy_version="select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="offline-data-v1",
    )
    raw_store = InMemoryRawProviderBytesStore()
    adapter_class = (
        _RawReferenceMismatchAdapter if mismatch_raw_reference else OfflineFixtureVisionAdapter
    )
    adapter = adapter_class(
        capabilities=_capabilities(max_images_per_request=max_images_per_request),
        raw_store=raw_store,
        parser=StrictProviderClaimParser(registry, parser_version=PARSER_VERSION),
        response_factory=response_factory,
    )
    repository = InMemoryEventIdentityRegistryRepository()
    identity_policy = EventIdentityPolicyRef(
        version="exact-fingerprint-v1",
        semantic_sha256=_digest("exact-fingerprint-v1"),
    )
    identity_registry = EventIdentityRegistryService(
        repository=repository,
        resolver=ExactFingerprintEventIdentityResolver(identity_policy),
        allocator=_SequenceEventIdAllocator(),
        output_admission_policy=output_policy,
    )
    input_preparer = InputPlanPreparer(
        InferenceInputPlanner("planner-v1"),
        ProviderRenderingPolicy(
            version="render-v1",
            transform_policy_version="identity-v1",
            idempotency_policy_version="idempotency-v1",
            reduction_policy=REDUCTION_POLICY,
            reduction_policy_version=REDUCTION_POLICY_VERSION,
            input_tokens_per_item=2,
            fixed_input_tokens_per_part=1,
            accepted_media_types=("image/png",),
        ),
    )
    pipeline = CanonicalOfflinePipeline(
        package_builder=PackageSetBuilder(REDUCTION_POLICY_VERSION),
        materializer=OfflineTemporalPackageMaterializer(_materialization_policy()),
        input_preparer=input_preparer,
        adapter=adapter,
        inference_policy=inference_policy,
        schema_registry=registry,
        identity_registry=identity_registry,
        execution_policy=execution_policy,
        clock=lambda: NOW,
    )
    context = _v2_context()
    plan = _sampling_plan()
    frame_index = _v2_frame_index(
        context,
        PackageLineage(
            source_content_sha256=context.source_content_sha256,
            window_semantic_sha256=_digest("placeholder-window"),
            camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
            alignment_semantic_sha256=context.alignment_semantic_sha256,
            sampling_plan_sha256=sampling_plan_digest(plan),
        ),
    )
    return _Harness(
        pipeline=pipeline,
        repository=repository,
        raw_store=raw_store,
        context=context,
        frame_index=frame_index,
    )


def _run(
    harness: _Harness,
    *,
    requested_interval: NanosecondInterval = REQUESTED_INTERVAL,
) -> CanonicalOfflineRunResult:
    return asyncio.run(
        harness.pipeline.run(
            admitted_context=harness.context,
            requested_interval=requested_interval,
            sampling_plan=_sampling_plan(),
            frame_index=harness.frame_index,
            artifact_resolver=_artifact_resolver(),
        )
    )


def _barrier_id(result: CanonicalOfflineRunResult) -> str:
    assert result.input_plan is not None
    logical_key = result.input_plan.call_plan.barrier_logical_key
    return str(uuid5(NAMESPACE_URL, f"robata:barrier:{logical_key}"))


def _assert_offline(result: CanonicalOfflineRunResult, harness: _Harness) -> None:
    assert result.network_call_count == 0
    assert harness.pipeline.adapter.network_call_count == 0


def test_success_connects_raw_claim_enrichment_and_recording_scoped_identity() -> None:
    harness = _harness(_claim_bytes)

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert result.attempt_count == result.adapter_infer_calls == 1
    assert result.terminal is not None
    assert result.selection is not None
    assert result.barrier_reduction is not None
    assert result.raw_response is not None
    assert result.parsed_claims is not None
    assert result.selected_output is not None
    assert result.enriched_output is not None
    assert result.output_decision is not None
    assert result.output_decision.decision == "PRODUCTION_ADMITTED"
    assert len(result.hypotheses) == 1
    assert result.identity_result is not None
    assignment = result.identity_result.assignments[0]
    assert assignment.event_id == _uuid(10_000)
    assert assignment.recording_identity == harness.context.recording_identity
    assert result.hypotheses[0].production_output_admission == (
        result.output_decision.production_output_admission
    )
    snapshot = harness.repository.snapshot(harness.context.recording_identity)
    assert snapshot.generation == 1
    assert tuple(item.event_id for item in snapshot.identities) == (_uuid(10_000),)
    assert len(snapshot.assignments) == 1
    assert len(harness.repository.list_outbox(harness.context.recording_identity)) == 1
    assert len(harness.raw_store.list_records()) == 1
    _assert_offline(result, harness)


def test_exact_replay_reuses_selected_success_without_new_side_effects() -> None:
    harness = _harness(_claim_bytes)
    first = _run(harness)
    assert first.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.identity_result is not None
    first_snapshot = harness.repository.snapshot(harness.context.recording_identity)
    first_outbox = harness.repository.list_outbox(harness.context.recording_identity)
    first_raw = harness.raw_store.list_records()

    replay = _run(harness)

    assert replay.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert replay.run_id == first.run_id
    assert replay.terminal == first.terminal
    assert replay.selection == first.selection
    assert replay.barrier_reduction == first.barrier_reduction
    assert replay.enriched_output == first.enriched_output
    assert replay.adapter_infer_calls == 0
    assert harness.pipeline.adapter.infer_calls == 1
    assert len(harness.pipeline.ledger.list_intents()) == 1
    assert len(harness.pipeline.ledger.list_terminals()) == 1
    assert len(harness.pipeline.ledger.list_selections()) == 1
    assert harness.raw_store.list_records() == first_raw
    replay_snapshot = harness.repository.snapshot(harness.context.recording_identity)
    assert replay_snapshot == first_snapshot
    assert replay_snapshot.generation == 1
    assert harness.repository.list_outbox(harness.context.recording_identity) == first_outbox
    assert replay.identity_result is not None
    assert replay.identity_result.new_identities == ()
    assert replay.identity_result.outbox == ()
    assert len(replay.identity_result.replayed_assignment_logical_keys) == 1
    _assert_offline(replay, harness)


def test_nonoverlapping_root_window_fails_before_capability_or_dispatch() -> None:
    harness = _harness(_claim_bytes)
    duration = harness.context.ready_manifest.recording.duration_ns

    result = _run(
        harness,
        requested_interval=NanosecondInterval(
            start_ns=duration + 1,
            end_ns=duration + 2,
        ),
    )

    assert result.status is CanonicalOfflineRunStatus.CONFIGURATION_FAILED
    assert result.error is not None
    assert result.error.code == "INVALID_ROOT_WINDOW"
    assert result.window is None
    assert result.input_plan is None
    assert result.attempt_count == result.adapter_infer_calls == 0
    assert harness.pipeline.adapter.capability_calls == 0
    assert harness.pipeline.adapter.infer_calls == 0
    assert harness.pipeline.ledger.list_intents() == ()
    assert harness.raw_store.list_records() == ()
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)


def test_abstention_returns_no_events_without_mutating_identity_registry() -> None:
    harness = _harness(lambda request: _claim_bytes(request, abstained=True))

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.NO_EVENTS
    assert result.output_decision is not None
    assert result.output_decision.decision == "NO_EVENTS"
    assert result.enriched_output is not None and result.enriched_output.abstained
    assert result.hypotheses == ()
    assert result.identity_result is None
    snapshot = harness.repository.snapshot(harness.context.recording_identity)
    assert snapshot.generation == 0
    assert snapshot.identities == snapshot.assignments == ()
    assert harness.repository.list_outbox(harness.context.recording_identity) == ()
    _assert_offline(result, harness)


def test_retryable_failure_is_not_barrier_terminal_and_second_attempt_succeeds() -> None:
    calls = 0

    def retry_then_succeed(request: VisionInferenceRequest) -> OfflineFixtureResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _failure(
                request,
                retryability=Retryability.RETRYABLE,
                status=InferenceStatus.TIMEOUT,
            )
        return _claim_bytes(request)

    harness = _harness(retry_then_succeed)

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert calls == result.attempt_count == result.adapter_infer_calls == 2
    terminals = harness.pipeline.ledger.list_terminals()
    assert tuple(item.status for item in terminals) == (
        InferenceStatus.TIMEOUT,
        InferenceStatus.SUCCEEDED,
    )
    assert len(harness.pipeline.ledger.list_selections()) == 1
    assert result.barrier_reduction is not None
    completions = harness.pipeline.call_barrier_storage.list_completions(_barrier_id(result))
    assert len(completions) == 1
    assert completions[0].status is InferenceStatus.SUCCEEDED
    assert len(harness.raw_store.list_records()) == 1
    _assert_offline(result, harness)


def test_permanent_failure_has_final_barrier_completion_but_no_reduction_or_identity() -> None:
    harness = _harness(
        lambda request: _failure(
            request,
            retryability=Retryability.PERMANENT,
            status=InferenceStatus.FAILED,
        )
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INFERENCE_FAILED
    assert result.attempt_count == result.adapter_infer_calls == 1
    assert result.barrier_reduction is None
    assert result.selection is None
    assert result.identity_result is None
    assert harness.pipeline.ledger.list_selections() == ()
    barrier_id = _barrier_id(result)
    assert harness.pipeline.call_barrier_storage.get_definition(barrier_id) is not None
    assert len(harness.pipeline.call_barrier_storage.list_completions(barrier_id)) == 1
    assert harness.pipeline.call_barrier_storage.get_reduction(barrier_id) is None
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    assert harness.raw_store.list_records() == ()
    _assert_offline(result, harness)


def test_duplicate_json_key_is_invalid_output_and_exact_raw_bytes_are_retained() -> None:
    response = b'{"claims":[],"abstained":true,"abstained":false}'
    harness = _harness(lambda _request: response)

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INVALID_OUTPUT
    assert result.error is not None
    assert result.error.code == ProviderResponseParseCode.DUPLICATE_JSON_KEY.value
    assert result.barrier_reduction is None
    assert result.identity_result is None
    records = harness.raw_store.list_records()
    assert len(records) == 1
    assert records[0].data == response
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)


def test_selected_terminal_raw_reference_mismatch_fails_closed() -> None:
    harness = _harness(_claim_bytes, mismatch_raw_reference=True)

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INVALID_OUTPUT
    assert result.error is not None
    assert result.error.code == "SELECTED_RAW_OUTPUT_INVALID"
    assert result.terminal is not None
    assert result.terminal.raw_output == {"artifact_id": _uuid(99_999)}
    assert result.barrier_reduction is not None
    assert result.raw_response is None
    assert result.parsed_claims is None
    assert result.identity_result is None
    records = harness.raw_store.list_records()
    assert len(records) == 1
    assert records[0].artifact_id != _uuid(99_999)
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)


def test_provider_limit_multi_part_blocks_before_dispatch_and_barrier_declaration() -> None:
    harness = _harness(_claim_bytes, max_images_per_request=3)

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.BLOCKED_MULTI_PART
    assert result.input_plan is not None
    assert len(result.input_plan.call_plan.parts) > 1
    assert result.attempt_count == result.adapter_infer_calls == 0
    assert harness.pipeline.adapter.infer_calls == 0
    assert harness.pipeline.ledger.list_intents() == ()
    assert harness.pipeline.call_barrier_storage.get_definition(_barrier_id(result)) is None
    assert harness.raw_store.list_records() == ()
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)
