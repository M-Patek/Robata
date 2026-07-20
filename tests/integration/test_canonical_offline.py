from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import count
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from pydantic import ValidationError

from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.adapters.sqlite_inference_evidence import SQLiteInferenceEvidenceLedger
from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical_offline import (
    CANONICAL_OFFLINE_PIPELINE_VERSION,
    CanonicalOfflineConfigurationError,
    CanonicalOfflineExecutionPolicy,
    CanonicalOfflinePartStatus,
    CanonicalOfflinePipeline,
    CanonicalOfflineRunResult,
    CanonicalOfflineRunStatus,
)
from robata.application.canonical_run_membership import CanonicalProcessingRunContext
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.logical_nodes import RunNodeDisposition
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
    ENRICHED_OUTPUT_SCHEMA_VERSION,
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
    RawProviderBytesStore,
    StrictProviderClaimParser,
)
from robata.inference.orchestrator import InferenceLedgerError, InferencePolicy
from robata.inference.preparation import (
    InputPlanPreparer,
    ProviderRenderingPolicy,
)
from robata.ports.logical_node_registry import (
    LogicalNodeRegistryError,
    LogicalNodeRegistryErrorCode,
)
from robata.sampling.materializer import (
    CanonicalSixCameraFrameIndex,
    FrameArtifactResolver,
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
_RUN_ID_COUNTER = count(20_000)


def _uuid(number: int) -> str:
    return str(UUID(int=number))


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _schema_ref(registry: SchemaRegistry, schema_id: str) -> JsonSchemaRef:
    version = ENRICHED_OUTPUT_SCHEMA_VERSION if schema_id == ENRICHED_OUTPUT_SCHEMA_ID else "1.0.0"
    registered = registry.resolve_version(schema_id, version).ref
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


def _claim_bytes(
    request: VisionInferenceRequest,
    *,
    abstained: bool = False,
    evidence_provider_item_ordinal: int | None = None,
) -> bytes:
    if abstained:
        return canonical_json_bytes({"claims": [], "abstained": True})
    assert request.input_plan is not None
    assert request.input_plan_part_ordinal is not None
    entries = ProviderReferenceCatalog.derive_entries(
        request_catalog_sha256=request.input_plan.request_catalog.semantic_sha256,
        rendered_items=request.input_plan.rendered_items,
        token_policy_version=TOKEN_POLICY_VERSION,
    )
    part = request.input_plan.call_plan.parts[request.input_plan_part_ordinal]
    item_ordinal = (
        part.start_item_ordinal
        if evidence_provider_item_ordinal is None
        else evidence_provider_item_ordinal
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
                    "evidence_tokens": [entries[item_ordinal].correlation_token],
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
    raw_store: RawProviderBytesStore
    inference_evidence: SQLiteInferenceEvidenceLedger | None
    context: AdmittedRecordingContextV2
    frame_index: CanonicalSixCameraFrameIndex
    execution_policy: CanonicalOfflineExecutionPolicy
    logical_node_registry: LocalLogicalNodeRegistry


def _harness(
    response_factory: Callable[[VisionInferenceRequest], OfflineFixtureResponse],
    *,
    logical_registry_root: Path,
    max_images_per_request: int = 20,
    max_attempts: int = 2,
    mismatch_raw_reference: bool = False,
    repository: InMemoryEventIdentityRegistryRepository | None = None,
    inference_evidence_path: Path | None = None,
    clock: Callable[[], datetime] | None = None,
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
    inference_evidence = (
        SQLiteInferenceEvidenceLedger(inference_evidence_path, registry)
        if inference_evidence_path is not None
        else None
    )
    raw_store: RawProviderBytesStore = (
        inference_evidence if inference_evidence is not None else InMemoryRawProviderBytesStore()
    )
    adapter_class = (
        _RawReferenceMismatchAdapter if mismatch_raw_reference else OfflineFixtureVisionAdapter
    )
    adapter = adapter_class(
        capabilities=_capabilities(max_images_per_request=max_images_per_request),
        raw_store=raw_store,
        parser=StrictProviderClaimParser(registry, parser_version=PARSER_VERSION),
        response_factory=response_factory,
    )
    repository = repository or InMemoryEventIdentityRegistryRepository()
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
    logical_node_registry = LocalLogicalNodeRegistry(logical_registry_root)
    pipeline = CanonicalOfflinePipeline(
        package_builder=PackageSetBuilder(REDUCTION_POLICY_VERSION),
        materializer=OfflineTemporalPackageMaterializer(_materialization_policy()),
        input_preparer=input_preparer,
        adapter=adapter,
        inference_policy=inference_policy,
        schema_registry=registry,
        identity_registry=identity_registry,
        logical_node_registry=logical_node_registry,
        execution_policy=execution_policy,
        inference_ledger=inference_evidence,
        evidence_store=inference_evidence,
        clock=clock if clock is not None else lambda: NOW,
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
        inference_evidence=inference_evidence,
        context=context,
        frame_index=frame_index,
        execution_policy=execution_policy,
        logical_node_registry=logical_node_registry,
    )


def _processing_run(
    harness: _Harness,
    *,
    run_id: str,
    started_at: str = NOW_TEXT,
) -> CanonicalProcessingRunContext:
    return CanonicalProcessingRunContext.fresh(
        run_id=run_id,
        recording_identity=harness.context.recording_identity,
        mcap_id=harness.context.ready_manifest.mcap_id,
        pipeline_version=CANONICAL_OFFLINE_PIPELINE_VERSION,
        config_sha256=harness.execution_policy.semantic_sha256,
        started_at=started_at,
    )


def _run(
    harness: _Harness,
    *,
    requested_interval: NanosecondInterval = REQUESTED_INTERVAL,
    processing_run: CanonicalProcessingRunContext | None = None,
    artifact_resolver: FrameArtifactResolver | None = None,
) -> CanonicalOfflineRunResult:
    active_run = processing_run or _processing_run(
        harness,
        run_id=_uuid(next(_RUN_ID_COUNTER)),
    )
    return asyncio.run(
        harness.pipeline.run(
            processing_run=active_run,
            admitted_context=harness.context,
            requested_interval=requested_interval,
            sampling_plan=_sampling_plan(),
            frame_index=harness.frame_index,
            artifact_resolver=(
                artifact_resolver if artifact_resolver is not None else _artifact_resolver()
            ),
        )
    )


def _barrier_id(result: CanonicalOfflineRunResult) -> str:
    assert result.input_plan is not None
    logical_key = result.input_plan.call_plan.barrier_logical_key
    return str(uuid5(NAMESPACE_URL, f"robata:barrier:{logical_key}"))


def _assert_offline(result: CanonicalOfflineRunResult, harness: _Harness) -> None:
    assert result.network_call_count == 0
    assert harness.pipeline.adapter.network_call_count == 0


def _revalidate_result(
    result: CanonicalOfflineRunResult,
    **updates: object,
) -> CanonicalOfflineRunResult:
    values = result.model_dump(mode="python")
    values.update(updates)
    return CanonicalOfflineRunResult.model_validate(values, strict=True)


def _inference_evidence_counts(database_path: Path) -> dict[str, int]:
    queries = {
        "inference_intents": "SELECT COUNT(*) FROM inference_intents",
        "raw_provider_responses": "SELECT COUNT(*) FROM raw_provider_responses",
        "model_inference_terminals": "SELECT COUNT(*) FROM model_inference_terminals",
        "inference_attempt_selections": "SELECT COUNT(*) FROM inference_attempt_selections",
        "raw_provider_artifacts": "SELECT COUNT(*) FROM raw_provider_artifacts",
        "parsed_provider_claims": "SELECT COUNT(*) FROM parsed_provider_claims",
        "selected_attempt_outputs": "SELECT COUNT(*) FROM selected_attempt_outputs",
        "enriched_provider_outputs": "SELECT COUNT(*) FROM enriched_provider_outputs",
    }
    with sqlite3.connect(database_path) as connection:
        return {
            table: int(connection.execute(query).fetchone()[0]) for table, query in queries.items()
        }


def test_success_connects_raw_claim_enrichment_and_recording_scoped_identity(
    tmp_path: Path,
) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path)

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
    assert harness.pipeline.evidence_store.get_parsed_claim(result.parsed_claims.artifact_id) == (
        result.parsed_claims
    )
    assert harness.pipeline.evidence_store.get_selected_output(result.selection.selection_id) == (
        result.selected_output
    )
    assert harness.pipeline.evidence_store.get_enriched_output(
        result.enriched_output.artifact_id
    ) == (result.enriched_output)
    assert result.processing_run.run_id == result.run_id
    assert result.processing_run.primary_status.value == result.status.value
    assert result.mcap_id == harness.context.ready_manifest.mcap_id
    assert result.execution_policy_sha256 == harness.execution_policy.semantic_sha256
    assert tuple(item.node_type for item in result.run_memberships) == (
        "TEMPORAL_WINDOW",
        "TEMPORAL_PACKAGE_SET",
        "INFERENCE_INPUT_PLAN",
        "INFERENCE_CALL_PART",
        "INFERENCE_CALL_BARRIER",
        "INFERENCE_ATTEMPT_SELECTION",
        "PARSED_PROVIDER_CLAIM",
        "SELECTED_ATTEMPT_OUTPUT",
        "ORCHESTRATOR_ENRICHMENT",
        "INFERENCE_CALL_REDUCTION",
        "FUSION_REDUCTION",
        "OUTPUT_ADMISSION_DECISION",
        "EVENT_HYPOTHESIS",
    )
    assert all(item.disposition is RunNodeDisposition.CREATED for item in result.run_memberships)
    assert {
        item.identity for item in harness.logical_node_registry.list_run_memberships(result.run_id)
    } == {item.identity for item in result.run_memberships}
    _assert_offline(result, harness)


def test_run_membership_failure_stops_before_event_identity_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path)
    attach = harness.logical_node_registry.attach_run_node

    def reject_event_hypothesis(**kwargs: object):  # type: ignore[no-untyped-def]
        if getattr(kwargs["node"], "node_type", None) == "EVENT_HYPOTHESIS":
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.TRANSACTION_FAILED,
                "injected event-hypothesis membership failure",
            )
        return attach(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        harness.logical_node_registry,
        "attach_run_node",
        reject_event_hypothesis,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.RUN_MEMBERSHIP_FAILED
    assert result.processing_run.primary_status.value == result.status.value
    assert result.error is not None
    assert result.error.stage.value == "RUN_MEMBERSHIP"
    assert result.output_decision is not None
    assert result.hypotheses
    assert result.identity_result is None
    assert result.run_memberships[-1].role == "OUTPUT_DECISION"
    assert all(item.node_type != "EVENT_HYPOTHESIS" for item in result.run_memberships)
    snapshot = harness.repository.snapshot(harness.context.recording_identity)
    assert snapshot.generation == 0
    assert snapshot.assignments == ()
    assert harness.repository.list_outbox(harness.context.recording_identity) == ()


def test_exact_replay_reuses_selected_success_without_new_side_effects(
    tmp_path: Path,
) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
    )
    processing_run = _processing_run(harness, run_id=_uuid(21_000))
    first = _run(harness, processing_run=processing_run)
    assert first.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.input_plan is not None
    assert first.identity_result is not None
    part_count = len(first.input_plan.call_plan.parts)
    assert part_count > 1
    assert len(first.part_results) == part_count
    first_snapshot = harness.repository.snapshot(harness.context.recording_identity)
    first_outbox = harness.repository.list_outbox(harness.context.recording_identity)
    first_raw = harness.raw_store.list_records()

    replay = _run(harness, processing_run=processing_run)

    assert replay.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert replay.run_id == first.run_id
    assert replay.processing_run == first.processing_run
    assert replay.run_memberships == first.run_memberships
    assert replay.part_results == first.part_results
    assert replay.barrier_reduction == first.barrier_reduction
    assert replay.fusion_reduction == first.fusion_reduction
    assert replay.hypotheses == first.hypotheses
    assert replay.terminal is replay.selection is replay.enriched_output is None
    assert replay.adapter_infer_calls == 0
    assert harness.pipeline.adapter.infer_calls == part_count
    assert len(harness.pipeline.ledger.list_intents()) == part_count
    assert len(harness.pipeline.ledger.list_terminals()) == part_count
    assert len(harness.pipeline.ledger.list_selections()) == part_count
    assert harness.raw_store.list_records() == first_raw
    assert len(first_raw) == part_count
    replay_snapshot = harness.repository.snapshot(harness.context.recording_identity)
    assert replay_snapshot == first_snapshot
    assert replay_snapshot.generation == 1
    assert harness.repository.list_outbox(harness.context.recording_identity) == first_outbox
    assert replay.identity_result is not None
    assert replay.identity_result.new_identities == ()
    assert replay.identity_result.outbox == ()
    assert len(replay.identity_result.replayed_assignment_logical_keys) == part_count
    _assert_offline(replay, harness)


def test_exact_same_run_replay_keeps_completion_time_with_advancing_clock(
    tmp_path: Path,
) -> None:
    clock_calls = count()

    def advancing_clock() -> datetime:
        return NOW + timedelta(microseconds=next(clock_calls))

    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        clock=advancing_clock,
    )
    processing_run = _processing_run(harness, run_id=_uuid(21_001))

    first = _run(harness, processing_run=processing_run)
    replay = _run(harness, processing_run=processing_run)

    assert first.status is replay.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.processing_run == replay.processing_run
    assert first.processing_run.completed_at == processing_run.started_at


def test_finish_does_not_reread_a_clock_that_would_move_backwards(tmp_path: Path) -> None:
    clock_call_count = 0

    def rewinding_clock() -> datetime:
        nonlocal clock_call_count
        value = NOW if clock_call_count == 0 else NOW - timedelta(seconds=1)
        clock_call_count += 1
        return value

    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        clock=rewinding_clock,
    )

    result = _run(
        harness,
        processing_run=_processing_run(harness, run_id=_uuid(21_002)),
        artifact_resolver=lambda _camera_id, _frame: None,
    )

    assert result.status is CanonicalOfflineRunStatus.MATERIALIZATION_FAILED
    assert result.processing_run.completed_at == result.processing_run.started_at
    assert clock_call_count == 1


def test_fresh_runs_reuse_the_full_logical_chain_across_clock_facts(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
    )
    first_context = _processing_run(
        harness,
        run_id=_uuid(22_001),
        started_at="2026-07-19T11:59:58Z",
    )
    second_context = _processing_run(
        harness,
        run_id=_uuid(22_002),
        started_at="2026-07-19T11:59:59Z",
    )

    first = _run(harness, processing_run=first_context)
    second = _run(harness, processing_run=second_context)

    assert first.status is second.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.run_id != second.run_id
    assert first.package_set is not None and second.package_set is not None
    assert tuple(item.package_manifest_sha256 for item in first.package_set.members) != tuple(
        item.package_manifest_sha256 for item in second.package_set.members
    )
    assert first.package_set.package_set_id == second.package_set.package_set_id
    assert first.input_plan is not None and second.input_plan is not None
    assert first.input_plan.semantic_sha256 == second.input_plan.semantic_sha256
    assert tuple(
        (item.node_type, item.node_logical_key, item.role) for item in first.run_memberships
    ) == tuple(
        (item.node_type, item.node_logical_key, item.role) for item in second.run_memberships
    )
    assert all(item.disposition is RunNodeDisposition.CREATED for item in first.run_memberships)
    assert all(item.disposition is RunNodeDisposition.REUSED for item in second.run_memberships)
    assert second.adapter_infer_calls == 0
    assert harness.pipeline.adapter.infer_calls == len(first.input_plan.call_plan.parts)
    assert second.identity_result is not None
    assert second.identity_result.new_identities == ()
    assert second.identity_result.outbox == ()


def test_sqlite_inference_evidence_recovers_across_fresh_pipeline_instances(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "inference-evidence.sqlite3"
    logical_registry_root = tmp_path / "logical-registry"
    repository = InMemoryEventIdentityRegistryRepository()
    first_harness = _harness(
        _claim_bytes,
        logical_registry_root=logical_registry_root,
        max_images_per_request=3,
        repository=repository,
        inference_evidence_path=evidence_path,
    )
    first = _run(
        first_harness,
        processing_run=_processing_run(first_harness, run_id=_uuid(22_301)),
    )

    assert first.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.input_plan is not None
    part_count = len(first.input_plan.call_plan.parts)
    assert part_count > 1
    first_counts = _inference_evidence_counts(evidence_path)
    assert set(first_counts.values()) == {part_count}

    second_factory_calls = 0

    def provider_dispatch_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal second_factory_calls
        second_factory_calls += 1
        raise AssertionError("persisted selected evidence must prevent provider dispatch")

    second_now = NOW + timedelta(seconds=1)
    second_harness = _harness(
        provider_dispatch_must_not_run,
        logical_registry_root=logical_registry_root,
        max_images_per_request=3,
        repository=repository,
        inference_evidence_path=evidence_path,
        clock=lambda: second_now,
    )
    second = _run(
        second_harness,
        processing_run=_processing_run(
            second_harness,
            run_id=_uuid(22_302),
            started_at="2026-07-19T12:00:01Z",
        ),
    )

    assert first_harness.pipeline is not second_harness.pipeline
    assert first_harness.pipeline.adapter is not second_harness.pipeline.adapter
    assert first_harness.inference_evidence is not None
    assert second_harness.inference_evidence is not None
    assert first_harness.inference_evidence is not second_harness.inference_evidence
    assert first_harness.pipeline.ledger is first_harness.inference_evidence
    assert first_harness.pipeline.evidence_store is first_harness.inference_evidence
    assert first_harness.raw_store is first_harness.inference_evidence
    assert second_harness.pipeline.ledger is second_harness.inference_evidence
    assert second_harness.pipeline.evidence_store is second_harness.inference_evidence
    assert second_harness.raw_store is second_harness.inference_evidence
    assert first_harness.logical_node_registry is not second_harness.logical_node_registry
    assert first_harness.inference_evidence.database_path == evidence_path.resolve()
    assert second_harness.inference_evidence.database_path == evidence_path.resolve()
    assert second.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert second.run_id != first.run_id
    assert second_factory_calls == 0
    assert second.adapter_infer_calls == 0
    assert second_harness.pipeline.adapter.infer_calls == 0
    assert tuple(
        (item.parsed_claims, item.selected_output, item.enriched_output)
        for item in second.part_results
    ) == tuple(
        (item.parsed_claims, item.selected_output, item.enriched_output)
        for item in first.part_results
    )
    assert _inference_evidence_counts(evidence_path) == first_counts
    assert second_harness.raw_store.list_records() == first_harness.raw_store.list_records()
    assert all(item.disposition is RunNodeDisposition.REUSED for item in second.run_memberships)
    assert second.identity_result is not None
    assert second.identity_result.new_identities == ()
    assert second.identity_result.outbox == ()


def test_fresh_retry_attempt_reuses_run_independent_fusion_and_identity(tmp_path: Path) -> None:
    repository = InMemoryEventIdentityRegistryRepository()
    first_harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
        repository=repository,
    )
    first = _run(first_harness)
    assert first.status is CanonicalOfflineRunStatus.SUCCEEDED

    calls = 0

    def retry_first_call(request: VisionInferenceRequest) -> OfflineFixtureResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _failure(
                request,
                retryability=Retryability.RETRYABLE,
                status=InferenceStatus.TIMEOUT,
            )
        return _claim_bytes(request)

    second_harness = _harness(
        retry_first_call,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
        repository=repository,
    )
    second = _run(second_harness)
    assert second.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.input_plan is not None and second.input_plan is not None
    assert first.input_plan.semantic_sha256 == second.input_plan.semantic_sha256
    assert len(first.part_results) == len(second.part_results) > 1

    first_part = first.part_results[0]
    second_part = second.part_results[0]
    assert first_part.terminal.inference_id != second_part.terminal.inference_id
    assert first_part.selection is not None and second_part.selection is not None
    assert first_part.selection.selection_decision_logical_key == (
        second_part.selection.selection_decision_logical_key
    )
    assert first_part.selected_output is not None and second_part.selected_output is not None
    assert first_part.selected_output.output_sha256 == second_part.selected_output.output_sha256
    assert first_part.enriched_output is not None and second_part.enriched_output is not None
    assert first_part.enriched_output.enrichment_logical_key == (
        second_part.enriched_output.enrichment_logical_key
    )
    assert first_part.enriched_output.semantic_sha256 != (
        second_part.enriched_output.semantic_sha256
    )

    assert first.barrier_reduction is not None and second.barrier_reduction is not None
    assert first.barrier_reduction.ordered_completion_ids != (
        second.barrier_reduction.ordered_completion_ids
    )
    assert first.barrier_reduction.reduction_semantic_sha256 == (
        second.barrier_reduction.reduction_semantic_sha256
    )
    assert first.fusion_reduction is not None and second.fusion_reduction is not None
    assert first.fusion_reduction.reduction_logical_key == (
        second.fusion_reduction.reduction_logical_key
    )
    assert first.fusion_reduction.semantic_sha256 == second.fusion_reduction.semantic_sha256
    assert first.output_decision is not None and second.output_decision is not None
    assert first.output_decision.decision_id == second.output_decision.decision_id
    assert first.output_decision.semantic_sha256 == second.output_decision.semantic_sha256
    assert tuple(item.event_hypothesis_logical_key for item in first.hypotheses) == tuple(
        item.event_hypothesis_logical_key for item in second.hypotheses
    )
    assert tuple(item.semantic_sha256 for item in first.hypotheses) == tuple(
        item.semantic_sha256 for item in second.hypotheses
    )

    assert first.identity_result is not None and second.identity_result is not None
    assert second.identity_result.new_identities == ()
    assert second.identity_result.outbox == ()
    assert len(second.identity_result.replayed_assignment_logical_keys) == len(first.hypotheses)
    snapshot = repository.snapshot(first_harness.context.recording_identity)
    assert snapshot.generation == 1


def test_nonoverlapping_root_window_fails_before_capability_or_dispatch(
    tmp_path: Path,
) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path)
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


def test_processing_run_must_bind_recording_and_execution_policy(tmp_path: Path) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path)
    processing_run = _processing_run(harness, run_id=_uuid(22_100)).model_copy(
        update={"config_sha256": _digest("wrong-execution-policy")}
    )

    with pytest.raises(CanonicalOfflineConfigurationError, match="processing run"):
        _run(harness, processing_run=processing_run)

    assert harness.pipeline.adapter.capability_calls == 0
    assert harness.pipeline.adapter.infer_calls == 0
    assert harness.logical_node_registry.list_run_memberships(processing_run.run_id) == ()


def test_run_result_rejects_tampered_binding_and_membership_proof(tmp_path: Path) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path)
    result = _run(harness)
    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED

    with pytest.raises(ValidationError, match="terminal processing-run record"):
        _revalidate_result(result, mcap_id=_uuid(22_201))
    with pytest.raises(ValidationError, match="terminal processing-run record"):
        _revalidate_result(
            result,
            execution_policy_sha256=_digest("tampered-execution-policy"),
        )
    wrong_mcap_id = _uuid(22_202)
    with pytest.raises(ValidationError, match="root window MCAP"):
        _revalidate_result(
            result,
            mcap_id=wrong_mcap_id,
            processing_run=result.processing_run.model_copy(update={"mcap_id": wrong_mcap_id}),
        )
    wrong_policy_sha256 = _digest("consistently-tampered-execution-policy")
    with pytest.raises(ValidationError, match="input plan or run binding"):
        _revalidate_result(
            result,
            execution_policy_sha256=wrong_policy_sha256,
            processing_run=result.processing_run.model_copy(
                update={"config_sha256": wrong_policy_sha256}
            ),
        )
    with pytest.raises(ValidationError, match="complete nonempty membership lineage"):
        _revalidate_result(result, run_memberships=())

    reordered = (
        result.run_memberships[1],
        result.run_memberships[0],
        *result.run_memberships[2:],
    )
    with pytest.raises(ValidationError, match="exact ordered lineage prefix"):
        _revalidate_result(result, run_memberships=reordered)
    wrong_role = (
        result.run_memberships[0].model_copy(update={"role": "PACKAGE_SET"}),
        *result.run_memberships[1:],
    )
    with pytest.raises(ValidationError, match="exact ordered lineage prefix"):
        _revalidate_result(result, run_memberships=wrong_role)
    wrong_work_item = (
        result.run_memberships[0].model_copy(update={"first_work_item_id": _uuid(22_203)}),
        *result.run_memberships[1:],
    )
    with pytest.raises(ValidationError, match="canonical attachment"):
        _revalidate_result(result, run_memberships=wrong_work_item)


def test_membership_failed_result_rejects_published_identity_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success_harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "success",
    )
    successful = _run(success_harness)
    assert successful.identity_result is not None

    failure_harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "failure",
    )
    attach = failure_harness.logical_node_registry.attach_run_node

    def reject_event_hypothesis(**kwargs: object):  # type: ignore[no-untyped-def]
        if getattr(kwargs["node"], "node_type", None) == "EVENT_HYPOTHESIS":
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.TRANSACTION_FAILED,
                "injected event-hypothesis membership failure",
            )
        return attach(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        failure_harness.logical_node_registry,
        "attach_run_node",
        reject_event_hypothesis,
    )
    failed = _run(failure_harness)
    assert failed.status is CanonicalOfflineRunStatus.RUN_MEMBERSHIP_FAILED

    with pytest.raises(ValidationError, match="cannot carry a published identity result"):
        _revalidate_result(failed, identity_result=successful.identity_result)


def test_all_required_parts_abstain_without_mutating_identity_registry(tmp_path: Path) -> None:
    harness = _harness(
        lambda request: _claim_bytes(request, abstained=True),
        logical_registry_root=tmp_path,
        max_images_per_request=3,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.ABSTAINED
    assert result.input_plan is not None
    part_count = len(result.input_plan.call_plan.parts)
    assert part_count > 1
    assert len(result.part_results) == part_count
    assert all(
        item.status is CanonicalOfflinePartStatus.ENRICHED
        and item.enriched_output is not None
        and item.enriched_output.abstained
        for item in result.part_results
    )
    assert result.barrier_reduction is not None
    assert result.fusion_reduction is not None
    assert result.fusion_reduction.outcome == "ALL_PARTS_ABSTAINED"
    assert result.output_decision is not None
    assert result.output_decision.decision == "ABSTAINED"
    assert len(result.output_decision.source_enrichments) == part_count
    assert result.enriched_output is None
    assert result.hypotheses == ()
    assert result.identity_result is None
    snapshot = harness.repository.snapshot(harness.context.recording_identity)
    assert snapshot.generation == 0
    assert snapshot.identities == snapshot.assignments == ()
    assert harness.repository.list_outbox(harness.context.recording_identity) == ()
    completions = harness.pipeline.call_barrier_storage.list_completions(_barrier_id(result))
    assert len(completions) == part_count
    _assert_offline(result, harness)


def test_retry_budget_is_independent_per_part_and_barrier_sees_only_final_success(
    tmp_path: Path,
) -> None:
    calls_by_part: dict[int, int] = {}

    def retry_then_succeed(request: VisionInferenceRequest) -> OfflineFixtureResponse:
        assert request.input_plan_part_ordinal is not None
        part_ordinal = request.input_plan_part_ordinal
        calls_by_part[part_ordinal] = calls_by_part.get(part_ordinal, 0) + 1
        if part_ordinal == 1 and calls_by_part[part_ordinal] == 1:
            return _failure(
                request,
                retryability=Retryability.RETRYABLE,
                status=InferenceStatus.TIMEOUT,
            )
        return _claim_bytes(request)

    harness = _harness(
        retry_then_succeed,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert result.input_plan is not None
    part_count = len(result.input_plan.call_plan.parts)
    assert part_count > 1
    assert calls_by_part == {ordinal: 2 if ordinal == 1 else 1 for ordinal in range(part_count)}
    assert result.attempt_count == result.adapter_infer_calls == part_count + 1
    assert tuple(item.orchestration_attempt_count for item in result.part_results) == tuple(
        2 if ordinal == 1 else 1 for ordinal in range(part_count)
    )
    terminals = harness.pipeline.ledger.list_terminals()
    assert len(terminals) == part_count + 1
    assert sum(item.status is InferenceStatus.TIMEOUT for item in terminals) == 1
    assert sum(item.status is InferenceStatus.SUCCEEDED for item in terminals) == part_count
    assert len(harness.pipeline.ledger.list_selections()) == part_count
    assert result.barrier_reduction is not None
    completions = harness.pipeline.call_barrier_storage.list_completions(_barrier_id(result))
    assert len(completions) == part_count
    assert all(item.status is InferenceStatus.SUCCEEDED for item in completions)
    assert len(harness.raw_store.list_records()) == part_count
    _assert_offline(result, harness)


def test_one_permanent_part_failure_does_not_skip_remaining_required_parts(
    tmp_path: Path,
) -> None:
    seen_parts: list[int] = []

    def fail_one_part(request: VisionInferenceRequest) -> OfflineFixtureResponse:
        assert request.input_plan_part_ordinal is not None
        seen_parts.append(request.input_plan_part_ordinal)
        if request.input_plan_part_ordinal == 1:
            return _failure(
                request,
                retryability=Retryability.PERMANENT,
                status=InferenceStatus.FAILED,
            )
        return _claim_bytes(request)

    harness = _harness(
        fail_one_part,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INCOMPLETE
    assert result.input_plan is not None
    part_count = len(result.input_plan.call_plan.parts)
    assert part_count > 1
    assert seen_parts == list(range(part_count))
    assert result.attempt_count == result.adapter_infer_calls == part_count
    assert len(result.part_results) == part_count
    assert result.part_results[1].status is CanonicalOfflinePartStatus.TERMINAL_FAILED
    assert all(
        item.status is CanonicalOfflinePartStatus.ENRICHED
        for ordinal, item in enumerate(result.part_results)
        if ordinal != 1
    )
    assert result.barrier_reduction is None
    assert result.selection is None
    assert result.identity_result is None
    assert len(harness.pipeline.ledger.list_selections()) == part_count - 1
    barrier_id = _barrier_id(result)
    assert harness.pipeline.call_barrier_storage.get_definition(barrier_id) is not None
    completions = harness.pipeline.call_barrier_storage.list_completions(barrier_id)
    assert len(completions) == part_count
    assert sum(item.status is InferenceStatus.FAILED for item in completions) == 1
    assert harness.pipeline.call_barrier_storage.get_reduction(barrier_id) is None
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    assert len(harness.raw_store.list_records()) == part_count - 1
    with pytest.raises(ValidationError, match="complete retained membership lineage"):
        _revalidate_result(result, run_memberships=result.run_memberships[:-1])
    _assert_offline(result, harness)


def test_duplicate_json_key_makes_required_part_incomplete_and_retains_raw_bytes(
    tmp_path: Path,
) -> None:
    response = b'{"claims":[],"abstained":true,"abstained":false}'
    harness = _harness(lambda _request: response, logical_registry_root=tmp_path)

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INCOMPLETE
    assert result.error is not None
    assert result.error.code == "REQUIRED_CALL_PARTS_INCOMPLETE"
    assert len(result.part_results) == 1
    failed_part = result.part_results[0]
    assert failed_part.status is CanonicalOfflinePartStatus.TERMINAL_FAILED
    assert failed_part.error is not None
    assert failed_part.error.code == ProviderResponseParseCode.DUPLICATE_JSON_KEY.value
    assert result.barrier_reduction is None
    assert result.identity_result is None
    records = harness.raw_store.list_records()
    assert len(records) == 1
    assert records[0].data == response
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)


def test_selected_terminal_raw_reference_mismatch_fails_closed(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        mismatch_raw_reference=True,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INVALID_OUTPUT
    assert result.error is not None
    assert result.error.code == "SELECTED_RAW_OUTPUT_INVALID"
    assert result.terminal is not None
    assert result.terminal.raw_output == {"artifact_id": _uuid(99_999)}
    assert result.barrier_reduction is None
    assert result.raw_response is None
    assert result.parsed_claims is None
    assert result.identity_result is None
    records = harness.raw_store.list_records()
    assert len(records) == 1
    assert records[0].artifact_id != _uuid(99_999)
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)


def test_persisted_inference_evidence_conflict_is_a_structured_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path)

    def reject_parsed_claim(_artifact: object) -> object:
        raise InferenceLedgerError("injected parsed evidence conflict")

    monkeypatch.setattr(
        harness.pipeline.evidence_store,
        "append_parsed_claim",
        reject_parsed_claim,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INVALID_OUTPUT
    assert result.error is not None
    assert result.error.stage.value == "PARSING"
    assert result.error.code == "INFERENCE_EVIDENCE_CONFLICT"
    assert "injected parsed evidence conflict" in result.error.detail
    assert result.identity_result is None


def test_provider_limit_multi_part_reduces_complete_ordered_call_set(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert result.input_plan is not None
    part_count = len(result.input_plan.call_plan.parts)
    assert part_count > 1
    assert len(result.part_results) == part_count
    assert all(item.status is CanonicalOfflinePartStatus.ENRICHED for item in result.part_results)
    assert result.attempt_count == result.adapter_infer_calls == part_count
    assert result.terminal is result.selection is result.enriched_output is None
    assert result.barrier_reduction is not None
    assert len(result.barrier_reduction.ordered_completion_ids) == part_count
    assert result.fusion_reduction is not None
    assert result.fusion_reduction.outcome == "CLAIMS"
    assert tuple(item.fusion_output_ordinal for item in result.fusion_reduction.claims) == tuple(
        range(part_count)
    )
    assert len(result.hypotheses) == part_count
    assert result.identity_result is not None
    assert len(result.identity_result.assignments) == part_count
    completions = harness.pipeline.call_barrier_storage.list_completions(_barrier_id(result))
    assert tuple(item.part_ordinal for item in completions) == tuple(range(part_count))
    assert len(harness.raw_store.list_records()) == part_count
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 1
    _assert_offline(result, harness)


def test_mixed_required_part_abstention_is_incomplete_without_reduction(
    tmp_path: Path,
) -> None:
    def mixed_response(request: VisionInferenceRequest) -> OfflineFixtureResponse:
        assert request.input_plan_part_ordinal is not None
        return _claim_bytes(request, abstained=request.input_plan_part_ordinal == 0)

    harness = _harness(
        mixed_response,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INCOMPLETE
    assert result.error is not None
    assert result.input_plan is not None
    part_count = len(result.input_plan.call_plan.parts)
    assert part_count > 1
    assert len(result.part_results) == part_count
    assert all(item.status is CanonicalOfflinePartStatus.ENRICHED for item in result.part_results)
    assert result.part_results[0].enriched_output is not None
    assert result.part_results[0].enriched_output.abstained
    assert all(
        item.enriched_output is not None and not item.enriched_output.abstained
        for item in result.part_results[1:]
    )
    assert result.barrier_reduction is None
    assert result.fusion_reduction is None
    assert result.output_decision is None
    assert result.identity_result is None
    barrier_id = _barrier_id(result)
    assert len(harness.pipeline.call_barrier_storage.list_completions(barrier_id)) == part_count
    assert harness.pipeline.call_barrier_storage.get_reduction(barrier_id) is None
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)


def test_out_of_part_evidence_is_rejected_after_all_parts_reach_terminal(
    tmp_path: Path,
) -> None:
    def out_of_scope_on_second_part(
        request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        assert request.input_plan_part_ordinal is not None
        if request.input_plan_part_ordinal == 1:
            return _claim_bytes(request, evidence_provider_item_ordinal=0)
        return _claim_bytes(request)

    harness = _harness(
        out_of_scope_on_second_part,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INVALID_OUTPUT
    assert result.error is not None
    assert result.error.code == "ENRICHMENT_REJECTED"
    assert result.input_plan is not None
    part_count = len(result.input_plan.call_plan.parts)
    assert part_count > 1
    assert len(result.part_results) == part_count
    invalid_part = result.part_results[1]
    assert invalid_part.status is CanonicalOfflinePartStatus.POST_SELECTION_INVALID
    assert invalid_part.raw_response is not None
    assert invalid_part.parsed_claims is not None
    assert invalid_part.selected_output is not None
    assert invalid_part.enriched_output is None
    assert all(
        item.status is CanonicalOfflinePartStatus.ENRICHED
        for ordinal, item in enumerate(result.part_results)
        if ordinal != 1
    )
    assert result.barrier_reduction is None
    assert result.fusion_reduction is None
    assert result.output_decision is None
    assert result.identity_result is None
    completions = harness.pipeline.call_barrier_storage.list_completions(_barrier_id(result))
    assert len(completions) == part_count
    assert all(item.status is InferenceStatus.SUCCEEDED for item in completions)
    assert len(harness.raw_store.list_records()) == part_count
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)
