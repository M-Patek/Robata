"""Concrete, restartable LOCAL_CONFORMANCE composition for the canonical path."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.adapters.sqlite_barrier import (
    SQLiteBarrierStorage,
    SQLiteBarrierStorageError,
)
from robata.adapters.sqlite_inference_evidence import (
    SQLiteInferenceEvidenceLedger,
    SQLiteInferenceEvidenceLedgerError,
)
from robata.adapters.sqlite_primary_completion import SQLitePrimaryCompletionRepository
from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.action_event_revision import (
    CanonicalActionEventRevisionError,
    prepare_initial_action_event_publications,
)
from robata.application.canonical.models import (
    CANONICAL_OFFLINE_PIPELINE_VERSION,
    CanonicalOfflineExecutionPolicy,
    CanonicalOfflineRunStatus,
)
from robata.application.canonical.output_admission import (
    CANONICAL_FINAL_FUSION_CONTEXT_METADATA_KEY,
    CanonicalFinalFusionContext,
)
from robata.application.canonical.primary_completion import (
    CommittedPrimaryCompletion,
    PrimaryCompletionError,
    create_primary_completion_command,
)
from robata.application.canonical.runner import CanonicalOfflinePipeline
from robata.application.canonical.source_fixture import (
    CanonicalSourceBundle,
    CanonicalSourceFixture,
    CanonicalSourceFixtureError,
    load_canonical_source_fixture,
)
from robata.application.canonical_run_membership import CanonicalProcessingRunContext
from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval, StrictModel
from robata.contracts.hashing import (
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)
from robata.contracts.sampling_plan import SamplingPlan
from robata.contracts.schema_registry import SchemaRegistry
from robata.event_pipeline.identity_registry import (
    EventIdAllocator,
    EventIdentityPolicyRef,
    EventIdentityRegistryError,
    EventIdentityRegistryService,
    ExactFingerprintEventIdentityResolver,
    PlatformEnrichedEventHypothesis,
    ProductionOutputAdmissionPolicyRef,
)
from robata.event_pipeline.provisional_fusion import (
    LOCAL_PROVISIONAL_FUSION_POLICY_VERSION,
    ProvisionalFusionPolicy,
)
from robata.inference.adapter import JsonSchemaRef, VisionInferenceRequest
from robata.inference.enrichment import (
    ENRICHED_OUTPUT_SCHEMA_ID,
    ENRICHED_OUTPUT_SCHEMA_VERSION,
    PROVIDER_CLAIM_SCHEMA_ID,
    ProviderClaimInterval,
    ProviderClaimKind,
    ProviderClaimPayload,
    ProviderObservation,
    ProviderReferenceCatalog,
    ProviderTaskClaim,
)
from robata.inference.input_plan import (
    INFERENCE_INPUT_PLANNER_VERSION,
    InferenceInputPlanner,
    RenderedProviderItem,
)
from robata.inference.models import (
    ConcurrencyClass,
    InputMode,
    ModelCapabilities,
    VisionTask,
)
from robata.inference.offline_fixture import (
    OfflineFixtureVisionAdapter,
    StrictProviderClaimParser,
)
from robata.inference.orchestrator import InferencePolicy
from robata.inference.preparation import InputPlanPreparer, ProviderRenderingPolicy
from robata.ports.logical_node_registry import LogicalNodeRegistryError
from robata.qa_pipeline.coarse import LOCAL_COARSE_QA_POLICY_VERSION
from robata.qa_pipeline.completion import LOCAL_QA_COMPLETION_POLICY_VERSION
from robata.qa_pipeline.dense import DenseQAPlanningPolicy
from robata.sampling.materializer import (
    CanonicalSixCameraFrameIndex,
    IndexedSourceFrame,
    MaterializedFrameArtifactFact,
    OfflineTemporalPackageMaterializer,
    TemporalPackageMaterializationPolicy,
)
from robata.sampling.package_set import PackageSetBuilder

LOCAL_CANONICAL_COMPOSITION_VERSION = "canonical-local-composition-v14"
LOCAL_CANONICAL_EXECUTION_CLOCK_VERSION = "canonical-local-execution-clock-v1"
LOCAL_CANONICAL_EXECUTION_TIME = "2026-07-20T00:00:00Z"
_LOCAL_CANONICAL_EXECUTION_DATETIME: Final = datetime(2026, 7, 20, tzinfo=UTC)
LOCAL_CANONICAL_TOKEN_POLICY_VERSION = "provider-token-v1"
LOCAL_CANONICAL_PARSER_VERSION = "strict-provider-claim-v1"
LOCAL_CANONICAL_REDUCTION_POLICY = "ordered-claims-v1"
LOCAL_CANONICAL_REDUCTION_POLICY_VERSION = "1.0"
LOCAL_CANONICAL_EVENT_ALLOCATOR_VERSION = "canonical-local-event-uuid5-v1"


class CanonicalLocalCompositionErrorCode(StrEnum):
    """Small command-boundary error vocabulary for the local composition."""

    INVALID_REQUEST = "INVALID_REQUEST"
    SOURCE_INVALID = "SOURCE_INVALID"
    LOCAL_STATE_FAILED = "LOCAL_STATE_FAILED"
    RUN_NOT_COMPLETABLE = "RUN_NOT_COMPLETABLE"
    COMPLETION_FAILED = "COMPLETION_FAILED"


class CanonicalLocalCompositionError(RuntimeError):
    """The local command could not produce an authoritative completion."""

    def __init__(
        self,
        code: CanonicalLocalCompositionErrorCode,
        detail: str,
    ) -> None:
        super().__init__(detail)
        self.code = code


class CanonicalLocalRunReceipt(StrictModel):
    """Compact operator view of one committed local-conformance run."""

    schema_version: Literal["1.0"]
    ok: Literal[True]
    run_id: str
    recording_identity: str
    status: Literal["SUCCEEDED", "NO_EVENTS"]
    command_sha256: str
    completion_semantic_sha256: str
    event_ids: tuple[str, ...]
    revision_ids: tuple[str, ...]
    outbox_ids: tuple[str, ...]
    outbox_count: int
    replayed: bool
    fixture_inference_calls: int
    network_call_count: Literal[0]
    evidence_class: Literal["LOCAL_CONFORMANCE"]
    production_eligible: Literal[False]


@dataclass(frozen=True, slots=True)
class _LocalCanonicalRuntime:
    registry: SchemaRegistry
    execution_policy: CanonicalOfflineExecutionPolicy
    pipeline: CanonicalOfflinePipeline


class _CanonicalSourceInputs(Protocol):
    @property
    def source_content_sha256(self) -> str: ...

    @property
    def admitted_context(self) -> AdmittedRecordingContextV2: ...

    @property
    def requested_interval(self) -> NanosecondInterval: ...

    @property
    def sampling_plan(self) -> SamplingPlan: ...

    @property
    def frame_index(self) -> CanonicalSixCameraFrameIndex: ...

    def resolve_artifact(
        self,
        camera_id: CameraId,
        frame: IndexedSourceFrame,
    ) -> MaterializedFrameArtifactFact | None: ...


_CanonicalSourceLoader = Callable[[SchemaRegistry], _CanonicalSourceInputs]


class _DeterministicLocalEventIdAllocator(EventIdAllocator):
    """Allocate stable proposed IDs so a pre-commit crash can be replayed exactly."""

    @property
    def version(self) -> str:
        return LOCAL_CANONICAL_EVENT_ALLOCATOR_VERSION

    def allocate(
        self,
        *,
        recording_identity: str,
        hypothesis: PlatformEnrichedEventHypothesis,
        registry_generation: int,
    ) -> str:
        material = ":".join(
            (
                self.version,
                recording_identity,
                hypothesis.event_hypothesis_logical_key,
                hypothesis.semantic_sha256,
                str(registry_generation),
            )
        )
        return str(uuid5(NAMESPACE_URL, f"robata:canonical-local-event:{material}"))


def run_local_canonical_fixture(
    source_path: Path,
    state_dir: Path,
    run_key: str = "primary",
) -> CanonicalLocalRunReceipt:
    """Run or recover the complete local canonical path from a JSON source fixture."""

    source = _require_path(source_path, "source_path")
    state_root = _require_path(state_dir, "state_dir")
    _require_run_key(run_key)
    source_sha256, _, clock_value = _source_run_binding(source)
    source_binding_sha256 = semantic_sha256(
        {
            "source_binding_policy_version": "canonical-local-source-binding-v1",
            "source_kind": "json-source-fixture",
            "source_content_sha256": source_sha256,
        }
    )

    def load_source(registry: SchemaRegistry) -> CanonicalSourceBundle:
        return load_canonical_source_fixture(
            source,
            schema_registry=registry,
            clock=lambda: clock_value,
        )

    return _run_local_canonical(
        state_root=state_root,
        run_key=run_key,
        source_sha256=source_sha256,
        source_binding_sha256=source_binding_sha256,
        source_loader=load_source,
    )


def run_local_canonical_mcap(
    source_path: Path,
    mapping_config: Path,
    state_dir: Path,
    run_key: str = "primary",
    *,
    allow_unapproved_profile: bool = False,
) -> CanonicalLocalRunReceipt:
    """Run or recover the complete local canonical path from a real MCAP."""

    from robata.application.canonical.mcap_source import (
        CanonicalMcapSourceBundle,
        CanonicalMcapSourceError,
        authorize_mcap_mapping,
        load_canonical_mcap_source,
    )

    source = _require_path(source_path, "source_path")
    mapping = _require_path(mapping_config, "mapping_config")
    state_root = _require_path(state_dir, "state_dir")
    _require_run_key(run_key)
    if not isinstance(allow_unapproved_profile, bool):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "allow_unapproved_profile must be a boolean",
        )
    try:
        # Authorization must precede the first source read.
        authorization = authorize_mcap_mapping(
            mapping,
            allow_unapproved_profile=allow_unapproved_profile,
        )
        source_sha256 = _hash_source_file(source, label="MCAP source")
    except CanonicalMcapSourceError as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
            str(error),
        ) from error

    source_binding_sha256 = semantic_sha256(
        {
            "source_binding_policy_version": "canonical-local-source-binding-v1",
            "source_kind": "mcap",
            "source_content_sha256": source_sha256,
            "mapping_profile_semantic_sha256": authorization.semantic_sha256,
        }
    )

    def load_source(registry: SchemaRegistry) -> CanonicalMcapSourceBundle:
        try:
            return load_canonical_mcap_source(
                source,
                authorization=authorization,
                state_dir=(
                    state_root
                    / "mcap"
                    / _stable_uuid("canonical-mcap-source-state", source_binding_sha256)
                ),
                expected_source_sha256=source_sha256,
                schema_registry=registry,
                clock=lambda: _LOCAL_CANONICAL_EXECUTION_DATETIME,
            )
        except CanonicalMcapSourceError as error:
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
                str(error),
            ) from error

    return _run_local_canonical(
        state_root=state_root,
        run_key=run_key,
        source_sha256=source_sha256,
        source_binding_sha256=source_binding_sha256,
        source_loader=load_source,
    )


def _run_local_canonical(
    *,
    state_root: Path,
    run_key: str,
    source_sha256: str,
    source_binding_sha256: str,
    source_loader: _CanonicalSourceLoader,
) -> CanonicalLocalRunReceipt:
    """Run the shared canonical flow after source authorization and binding."""

    started_at = LOCAL_CANONICAL_EXECUTION_TIME
    clock_value = _LOCAL_CANONICAL_EXECUTION_DATETIME
    registry = SchemaRegistry()
    execution_policy = _execution_policy()
    (
        coarse_qa_policy,
        dense_qa_policy,
        event_proposal_policy,
        action_evidence_policy,
        boundary_refinement_policy,
        inference_policy,
    ) = _local_inference_policies(registry)
    runtime_policy_sha256 = _local_runtime_policy_sha256(
        coarse_qa_policy=coarse_qa_policy,
        dense_qa_policy=dense_qa_policy,
        event_proposal_policy=event_proposal_policy,
        action_evidence_policy=action_evidence_policy,
        boundary_refinement_policy=boundary_refinement_policy,
        inference_policy=inference_policy,
    )
    run_id = _run_id(
        source_binding_sha256=source_binding_sha256,
        execution_policy_sha256=execution_policy.semantic_sha256,
        runtime_policy_sha256=runtime_policy_sha256,
        run_key=run_key,
    )
    try:
        state_root.mkdir(parents=True, exist_ok=True)
        completion_repository = SQLitePrimaryCompletionRepository(
            state_root / "primary-completion.sqlite3",
            registry=registry,
        )
        recovered = completion_repository.get(run_id)
        if recovered is not None:
            return _receipt(
                recovered,
                replayed=True,
                fixture_inference_calls=0,
            )

        bundle = source_loader(registry)
        if bundle.source_content_sha256 != source_sha256:
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
                "source bytes changed while preparing the canonical run",
            )
        runtime = _build_runtime(
            state_root=state_root,
            run_id=run_id,
            registry=registry,
            execution_policy=execution_policy,
            coarse_qa_policy=coarse_qa_policy,
            dense_qa_policy=dense_qa_policy,
            event_proposal_policy=event_proposal_policy,
            action_evidence_policy=action_evidence_policy,
            boundary_refinement_policy=boundary_refinement_policy,
            inference_policy=inference_policy,
            clock_value=clock_value,
            observed_at=started_at,
        )
        processing_run = CanonicalProcessingRunContext.fresh(
            run_id=run_id,
            recording_identity=bundle.admitted_context.recording_identity,
            mcap_id=bundle.admitted_context.ready_manifest.mcap_id,
            pipeline_version=CANONICAL_OFFLINE_PIPELINE_VERSION,
            config_sha256=execution_policy.semantic_sha256,
            started_at=started_at,
        )
        processing_run = CanonicalProcessingRunContext.resume(
            completion_repository.begin_run(processing_run)
        )
        result = asyncio.run(
            runtime.pipeline.run(
                processing_run=processing_run,
                admitted_context=bundle.admitted_context,
                requested_interval=bundle.requested_interval,
                sampling_plan=bundle.sampling_plan,
                frame_index=bundle.frame_index,
                artifact_resolver=bundle.resolve_artifact,
            )
        )
        if result.status not in {
            CanonicalOfflineRunStatus.SUCCEEDED,
            CanonicalOfflineRunStatus.NO_EVENTS,
        }:
            detail = result.error.detail if result.error is not None else "no diagnostic"
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.RUN_NOT_COMPLETABLE,
                f"canonical run ended as {result.status.value}: {detail}",
            )

        prepared_identities = None
        if result.status is CanonicalOfflineRunStatus.SUCCEEDED:
            enriched_outputs = tuple(
                item.enriched_output
                for item in result.part_results
                if item.enriched_output is not None
            )
            if not enriched_outputs:
                raise CanonicalLocalCompositionError(
                    CanonicalLocalCompositionErrorCode.RUN_NOT_COMPLETABLE,
                    "successful canonical run has no enriched outputs",
                )
            decided_at = result.processing_run.completed_at
            if decided_at is None:
                raise CanonicalLocalCompositionError(
                    CanonicalLocalCompositionErrorCode.RUN_NOT_COMPLETABLE,
                    "successful canonical run has no completion timestamp",
                )
            identity_service = EventIdentityRegistryService(
                repository=None,
                resolver=ExactFingerprintEventIdentityResolver(_identity_policy()),
                allocator=_DeterministicLocalEventIdAllocator(),
                output_admission_policy=execution_policy.output_admission_policy,
            )
            prepared_identities = identity_service.prepare_batch(
                snapshot=completion_repository.snapshot(result.recording_identity),
                admitted_context=bundle.admitted_context,
                hypotheses=result.hypotheses,
                enriched_outputs=enriched_outputs,
                decided_at=decided_at,
            )

        publications = prepare_initial_action_event_publications(
            context=bundle.admitted_context,
            result=result,
            prepared_identities=prepared_identities,
            execution_policy=execution_policy,
        )
        command = create_primary_completion_command(
            result=result,
            prepared_identities=prepared_identities,
            action_event_publications=publications,
            registry=registry,
        )
        commit_result = completion_repository.commit(command)
        return _receipt(
            commit_result.committed,
            replayed=commit_result.replayed,
            fixture_inference_calls=result.adapter_infer_calls,
        )
    except CanonicalLocalCompositionError:
        raise
    except (CanonicalSourceFixtureError, ValidationError) as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
            str(error),
        ) from error
    except (
        SQLiteBarrierStorageError,
        SQLiteInferenceEvidenceLedgerError,
        LogicalNodeRegistryError,
    ) as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED,
            str(error),
        ) from error
    except (PrimaryCompletionError, EventIdentityRegistryError) as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.COMPLETION_FAILED,
            str(error),
        ) from error
    except (CanonicalActionEventRevisionError, OSError, ValueError) as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.RUN_NOT_COMPLETABLE,
            str(error),
        ) from error


def _build_runtime(
    *,
    state_root: Path,
    run_id: str,
    registry: SchemaRegistry,
    execution_policy: CanonicalOfflineExecutionPolicy,
    coarse_qa_policy: InferencePolicy,
    dense_qa_policy: InferencePolicy,
    event_proposal_policy: InferencePolicy,
    action_evidence_policy: InferencePolicy,
    boundary_refinement_policy: InferencePolicy,
    inference_policy: InferencePolicy,
    clock_value: datetime,
    observed_at: str,
) -> _LocalCanonicalRuntime:
    """Compose runtime adapters from the exact policies bound into the run ID."""

    inference_evidence = SQLiteInferenceEvidenceLedger(
        state_root / "inference-evidence.sqlite3",
        registry,
    )
    barrier_storage = SQLiteBarrierStorage(
        state_root / "runs" / run_id / "inference-call-barrier.sqlite3"
    )
    parser = StrictProviderClaimParser(
        registry,
        parser_version=LOCAL_CANONICAL_PARSER_VERSION,
    )
    adapter = OfflineFixtureVisionAdapter(
        capabilities=_capabilities(observed_at),
        raw_store=inference_evidence,
        parser=parser,
        response_factory=_fixture_claim_bytes,
    )
    input_preparer = InputPlanPreparer(
        InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION),
        ProviderRenderingPolicy(
            version="render-v1",
            transform_policy_version="identity-v1",
            idempotency_policy_version="idempotency-v1",
            reduction_policy=LOCAL_CANONICAL_REDUCTION_POLICY,
            reduction_policy_version=LOCAL_CANONICAL_REDUCTION_POLICY_VERSION,
            input_tokens_per_item=2,
            fixed_input_tokens_per_part=1,
            accepted_media_types=("image/png",),
        ),
    )
    pipeline = CanonicalOfflinePipeline(
        package_builder=PackageSetBuilder(LOCAL_CANONICAL_REDUCTION_POLICY_VERSION),
        materializer=OfflineTemporalPackageMaterializer(
            TemporalPackageMaterializationPolicy(
                version="materialization-v1",
                grid_origin_ns=0,
                selection_tolerance_ns=300_000_000,
                tie_break_policy_version="nearest-v1",
                dedupe_policy_version="one-source-frame-v1",
                producer_version="offline-materializer-v1",
                extractor_version="fixture-png-v1",
            )
        ),
        input_preparer=input_preparer,
        adapter=adapter,
        raw_store=inference_evidence,
        parser=parser,
        coarse_qa_policy=coarse_qa_policy,
        dense_qa_policy=dense_qa_policy,
        event_proposal_policy=event_proposal_policy,
        action_evidence_policy=action_evidence_policy,
        boundary_refinement_policy=boundary_refinement_policy,
        inference_policy=inference_policy,
        schema_registry=registry,
        logical_node_registry=LocalLogicalNodeRegistry(state_root / "logical-nodes"),
        execution_policy=execution_policy,
        inference_ledger=inference_evidence,
        evidence_store=inference_evidence,
        barrier_storage=barrier_storage,
        call_barrier_storage=barrier_storage,
        clock=lambda: clock_value,
    )
    return _LocalCanonicalRuntime(
        registry=registry,
        execution_policy=execution_policy,
        pipeline=pipeline,
    )


def _local_inference_policies(
    registry: SchemaRegistry,
) -> tuple[
    InferencePolicy,
    InferencePolicy,
    InferencePolicy,
    InferencePolicy,
    InferencePolicy,
    InferencePolicy,
]:
    """Build the task policies before completion recovery is considered."""

    provider_schema = _schema_ref(registry, PROVIDER_CLAIM_SCHEMA_ID, "1.0.0")
    enriched_schema = _schema_ref(
        registry,
        ENRICHED_OUTPUT_SCHEMA_ID,
        ENRICHED_OUTPUT_SCHEMA_VERSION,
    )
    coarse_qa_policy = InferencePolicy(
        policy_version="offline-coarse-qa-model-policy-v1",
        task=VisionTask.QA_COARSE,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="offline-adapter-v1",
        prompt_version="coarse-qa-prompt-v1",
        prompt_artifact_id=_stable_uuid(
            "canonical-local-prompt",
            "coarse-qa-prompt-v1",
        ),
        prompt_sha256=exact_bytes_sha256(b"robata canonical local coarse QA prompt v1"),
        output_schema=provider_schema,
        enriched_output_schema=enriched_schema,
        generation_config={"temperature": 0},
        timeout_ms=1_000,
        selection_policy_version="select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="offline-data-v1",
    )
    dense_qa_policy = InferencePolicy(
        policy_version="offline-dense-qa-model-policy-v1",
        task=VisionTask.QA_DENSE,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="offline-adapter-v1",
        prompt_version="dense-qa-prompt-v1",
        prompt_artifact_id=_stable_uuid(
            "canonical-local-prompt",
            "dense-qa-prompt-v1",
        ),
        prompt_sha256=exact_bytes_sha256(b"robata canonical local dense QA prompt v1"),
        output_schema=provider_schema,
        enriched_output_schema=enriched_schema,
        generation_config={"temperature": 0},
        timeout_ms=1_000,
        selection_policy_version="select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="offline-data-v1",
    )
    event_proposal_policy = InferencePolicy(
        policy_version="offline-event-proposal-model-policy-v1",
        task=VisionTask.EVENT_PROPOSAL,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="offline-adapter-v1",
        prompt_version="event-proposal-prompt-v1",
        prompt_artifact_id=_stable_uuid("canonical-local-prompt", "event-proposal-prompt-v1"),
        prompt_sha256=exact_bytes_sha256(b"robata canonical local event proposal prompt v1"),
        output_schema=provider_schema,
        enriched_output_schema=enriched_schema,
        generation_config={"temperature": 0},
        timeout_ms=1_000,
        selection_policy_version="select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="offline-data-v1",
    )
    action_evidence_policy = InferencePolicy(
        policy_version="offline-action-evidence-model-policy-v1",
        task=VisionTask.ACTION_EVIDENCE,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="offline-adapter-v1",
        prompt_version="action-evidence-prompt-v1",
        prompt_artifact_id=_stable_uuid("canonical-local-prompt", "action-evidence-prompt-v1"),
        prompt_sha256=exact_bytes_sha256(b"robata canonical local action evidence prompt v1"),
        output_schema=provider_schema,
        enriched_output_schema=enriched_schema,
        generation_config={"temperature": 0},
        timeout_ms=1_000,
        selection_policy_version="select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="offline-data-v1",
    )
    boundary_refinement_policy = InferencePolicy(
        policy_version="offline-boundary-refinement-model-policy-v1",
        task=VisionTask.BOUNDARY_REFINEMENT,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="offline-adapter-v1",
        prompt_version="boundary-refinement-prompt-v1",
        prompt_artifact_id=_stable_uuid("canonical-local-prompt", "boundary-refinement-prompt-v1"),
        prompt_sha256=exact_bytes_sha256(b"robata canonical local boundary refinement prompt v1"),
        output_schema=provider_schema,
        enriched_output_schema=enriched_schema,
        generation_config={"temperature": 0},
        timeout_ms=1_000,
        selection_policy_version="select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="offline-data-v1",
    )
    inference_policy = InferencePolicy(
        policy_version="offline-model-policy-v2",
        task=VisionTask.FUSION_ADJUDICATION,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="offline-adapter-v1",
        prompt_version="fusion-prompt-v2",
        prompt_artifact_id=_stable_uuid("canonical-local-prompt", "fusion-prompt-v2"),
        prompt_sha256=exact_bytes_sha256(b"robata canonical local fusion prompt v2"),
        output_schema=provider_schema,
        enriched_output_schema=enriched_schema,
        generation_config={"temperature": 0},
        timeout_ms=1_000,
        selection_policy_version="select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="offline-data-v1",
    )
    return (
        coarse_qa_policy,
        dense_qa_policy,
        event_proposal_policy,
        action_evidence_policy,
        boundary_refinement_policy,
        inference_policy,
    )


def _local_runtime_policy_sha256(
    *,
    coarse_qa_policy: InferencePolicy,
    dense_qa_policy: InferencePolicy,
    event_proposal_policy: InferencePolicy,
    action_evidence_policy: InferencePolicy,
    boundary_refinement_policy: InferencePolicy,
    inference_policy: InferencePolicy,
) -> str:
    """Bind automatic recovery identity to the policies that drive the run."""

    return semantic_sha256(
        {
            "semantic_projection_version": "canonical-local-runtime-policy-v8",
            "pipeline_version": CANONICAL_OFFLINE_PIPELINE_VERSION,
            "coarse_qa_inference_policy": coarse_qa_policy.model_dump(mode="json"),
            "coarse_qa_projection_policy_version": LOCAL_COARSE_QA_POLICY_VERSION,
            "qa_completion_policy_version": LOCAL_QA_COMPLETION_POLICY_VERSION,
            "dense_qa_inference_policy": dense_qa_policy.model_dump(mode="json"),
            "event_proposal_inference_policy": event_proposal_policy.model_dump(mode="json"),
            "action_evidence_inference_policy": action_evidence_policy.model_dump(mode="json"),
            "boundary_refinement_inference_policy": (
                boundary_refinement_policy.model_dump(mode="json")
            ),
            "boundary_refinement_policy_version": "local-boundary-refinement-v1",
            "provisional_fusion_policy": ProvisionalFusionPolicy.create(
                version=LOCAL_PROVISIONAL_FUSION_POLICY_VERSION
            ).model_dump(mode="json"),
            "dense_qa_planning_policy": DenseQAPlanningPolicy(
                version=LOCAL_QA_COMPLETION_POLICY_VERSION
            ).model_dump(mode="json"),
            "fusion_inference_policy": inference_policy.model_dump(mode="json"),
        }
    )


def _execution_policy() -> CanonicalOfflineExecutionPolicy:
    output_policy = ProductionOutputAdmissionPolicyRef(
        version="fusion-output-admission-v1",
        semantic_sha256=semantic_sha256(
            {
                "semantic_projection_version": "local-output-admission-policy-v1",
                "evidence_class": "LOCAL_CONFORMANCE",
                "production_eligible": False,
            }
        ),
    )
    return CanonicalOfflineExecutionPolicy.create(
        policy_version="canonical-offline-v2",
        window_policy_version="root-window-v1",
        token_policy_version=LOCAL_CANONICAL_TOKEN_POLICY_VERSION,
        parser_version=LOCAL_CANONICAL_PARSER_VERSION,
        enrichment_policy_version="enrichment-v1",
        projector_policy_version="fusion-projector-v2",
        reduction_policy=LOCAL_CANONICAL_REDUCTION_POLICY,
        reduction_policy_version=LOCAL_CANONICAL_REDUCTION_POLICY_VERSION,
        provisional_fusion_policy_version=LOCAL_PROVISIONAL_FUSION_POLICY_VERSION,
        boundary_refinement_policy_version="local-boundary-refinement-v1",
        max_attempts=2,
        output_admission_policy=output_policy,
    )


def _identity_policy() -> EventIdentityPolicyRef:
    return EventIdentityPolicyRef(
        version="exact-fingerprint-v1",
        semantic_sha256=semantic_sha256(
            {
                "semantic_projection_version": "event-identity-policy-v1",
                "resolver": "exact-platform-semantic-fingerprint",
            }
        ),
    )


def _capabilities(observed_at: str) -> ModelCapabilities:
    facts: dict[str, object] = {
        "provider": "offline-fixture",
        "model_name": "fixture-vision",
        "model_version": "1.0",
        "supported_tasks": [
            VisionTask.QA_COARSE.value,
            VisionTask.QA_DENSE.value,
            VisionTask.EVENT_PROPOSAL.value,
            VisionTask.ACTION_EVIDENCE.value,
            VisionTask.BOUNDARY_REFINEMENT.value,
            VisionTask.FUSION_ADJUDICATION.value,
        ],
        "input_modes": [InputMode.MULTI_IMAGE.value],
        "accepted_media_types": ["image/png"],
        "max_images_per_request": 120,
        "max_pixels_per_image": 33_177_600,
        "max_payload_bytes": 64_000_000,
        "max_input_tokens": 100_000,
        "supports_json_schema": True,
        "supports_provider_idempotency": True,
        "concurrency_class": ConcurrencyClass.SERIAL.value,
        "data_handling_policy_version": "offline-data-v1",
        "observed_at": observed_at,
    }
    digest = semantic_sha256(
        {
            "semantic_projection_version": "local-fixture-capabilities-v1",
            **facts,
        }
    )
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_stable_uuid("canonical-local-capabilities", digest),
        snapshot_digest=digest,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        supported_tasks=(
            VisionTask.QA_COARSE,
            VisionTask.QA_DENSE,
            VisionTask.EVENT_PROPOSAL,
            VisionTask.ACTION_EVIDENCE,
            VisionTask.BOUNDARY_REFINEMENT,
            VisionTask.FUSION_ADJUDICATION,
        ),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=120,
        max_pixels_per_image=33_177_600,
        max_payload_bytes=64_000_000,
        max_input_tokens=100_000,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.SERIAL,
        data_handling_policy_version="offline-data-v1",
        observed_at=observed_at,
    )


def _fixture_claim_bytes(request: VisionInferenceRequest) -> bytes:
    plan = request.input_plan
    part_ordinal = request.input_plan_part_ordinal
    if plan is None or part_ordinal is None:
        raise ValueError("local fixture response requires one explicit input-plan part")
    part = plan.call_plan.parts[part_ordinal]
    items = plan.rendered_items[part.start_item_ordinal : part.end_item_ordinal_exclusive]
    if not items:
        raise ValueError("local fixture response cannot cite an empty input-plan part")
    entries = ProviderReferenceCatalog.derive_entries(
        request_catalog_sha256=plan.request_catalog.semantic_sha256,
        rendered_items=plan.rendered_items,
        token_policy_version=LOCAL_CANONICAL_TOKEN_POLICY_VERSION,
    )
    if request.task in {VisionTask.QA_COARSE, VisionTask.QA_DENSE}:
        first_by_coordinate: dict[tuple[int, int], RenderedProviderItem] = {}
        for item in plan.rendered_items:
            first_by_coordinate.setdefault(
                (item.package_ordinal, item.camera_ordinal),
                item,
            )
        part_coordinates = tuple(
            item
            for item in first_by_coordinate.values()
            if part.start_item_ordinal
            <= item.provider_item_ordinal
            < part.end_item_ordinal_exclusive
        )
        claims = tuple(
            ProviderTaskClaim(
                claim_ordinal=claim_ordinal,
                kind=ProviderClaimKind.QA_OBSERVATION,
                package_ordinal=item.package_ordinal,
                camera_ordinal=item.camera_ordinal,
                interval=ProviderClaimInterval(
                    start_ns=item.aligned_timestamp_ns,
                    end_ns=item.aligned_timestamp_ns + 1,
                ),
                label=None,
                observation=ProviderObservation.GOOD,
                evidence_tokens=(entries[item.provider_item_ordinal].correlation_token,),
                model_reported_score=None,
                conflict_codes=(),
            )
            for claim_ordinal, item in enumerate(part_coordinates)
        )
        return canonical_json_bytes(
            ProviderClaimPayload(
                claims=claims,
                abstained=False,
            )
        )
    if request.task is VisionTask.ACTION_EVIDENCE:
        by_coordinate: dict[tuple[int, int], list[RenderedProviderItem]] = {}
        for item in items:
            by_coordinate.setdefault((item.package_ordinal, item.camera_ordinal), []).append(item)
        claims = tuple(
            ProviderTaskClaim(
                claim_ordinal=claim_ordinal,
                kind=ProviderClaimKind.ACTION_OBSERVATION,
                package_ordinal=coordinate[0],
                camera_ordinal=coordinate[1],
                interval=ProviderClaimInterval(
                    start_ns=min(item.aligned_timestamp_ns for item in coordinate_items),
                    end_ns=max(item.aligned_timestamp_ns for item in coordinate_items) + 1,
                ),
                label="fixture-action",
                observation=ProviderObservation.SUPPORTING,
                evidence_tokens=tuple(
                    entries[item.provider_item_ordinal].correlation_token
                    for item in coordinate_items
                ),
                model_reported_score=0.8,
                conflict_codes=(),
            )
            for claim_ordinal, (coordinate, coordinate_items) in enumerate(
                sorted(by_coordinate.items())
            )
        )
        return canonical_json_bytes(ProviderClaimPayload(claims=claims, abstained=False))
    if request.task is VisionTask.BOUNDARY_REFINEMENT:
        role = request.metadata.get("boundary_refinement_role")
        if role not in {"ONSET", "OFFSET"}:
            raise ValueError("boundary fixture request lacks its orchestrator-owned role")
        anchor_text = request.metadata.get("boundary_anchor_ns")
        if anchor_text is None:
            raise ValueError("boundary fixture request lacks its orchestrator-owned anchor")
        try:
            boundary_anchor_ns = int(anchor_text)
        except ValueError as exc:
            raise ValueError("boundary fixture request carries an invalid anchor") from exc
        boundary_items_by_coordinate: dict[tuple[int, int], list[RenderedProviderItem]] = {}
        for item in items:
            boundary_items_by_coordinate.setdefault(
                (item.package_ordinal, item.camera_ordinal), []
            ).append(item)
        package_ordinals = sorted({item.package_ordinal for item in plan.rendered_items})
        observed_package_ordinal = min(
            package_ordinals,
            key=lambda package_ordinal: min(
                abs(item.aligned_timestamp_ns - boundary_anchor_ns)
                for item in plan.rendered_items
                if item.package_ordinal == package_ordinal
            ),
        )
        boundary_claims: list[ProviderTaskClaim] = []
        for claim_ordinal, (coordinate, coordinate_items) in enumerate(
            sorted(boundary_items_by_coordinate.items())
        ):
            observed = coordinate[0] == observed_package_ordinal
            ordered_items = sorted(
                coordinate_items,
                key=lambda item: (
                    abs(item.aligned_timestamp_ns - boundary_anchor_ns),
                    item.aligned_timestamp_ns,
                    item.frame_ordinal,
                ),
            )
            cited = (ordered_items[0],)
            boundary_claims.append(
                ProviderTaskClaim(
                    claim_ordinal=claim_ordinal,
                    kind=ProviderClaimKind.BOUNDARY_OBSERVATION,
                    package_ordinal=coordinate[0],
                    camera_ordinal=coordinate[1],
                    interval=(
                        ProviderClaimInterval(
                            start_ns=cited[0].aligned_timestamp_ns,
                            end_ns=cited[-1].aligned_timestamp_ns + 1,
                        )
                        if observed
                        else None
                    ),
                    label=None,
                    observation=(
                        ProviderObservation.OBSERVED
                        if observed
                        else ProviderObservation.NO_BOUNDARY
                    ),
                    evidence_tokens=(
                        tuple(
                            entries[item.provider_item_ordinal].correlation_token for item in cited
                        )
                        if observed
                        else ()
                    ),
                    model_reported_score=0.8 if observed else None,
                    conflict_codes=(),
                )
            )
        return canonical_json_bytes(
            ProviderClaimPayload(claims=tuple(boundary_claims), abstained=False)
        )
    if request.task is VisionTask.EVENT_PROPOSAL:
        start_ns = min(item.aligned_timestamp_ns for item in items)
        end_ns = max(item.aligned_timestamp_ns for item in items) + 1
        proposal_tokens = tuple(
            entries[item.provider_item_ordinal].correlation_token
            for item in items
            if item.camera_ordinal in range(6)
        )
        payload = ProviderClaimPayload(
            claims=(
                ProviderTaskClaim(
                    claim_ordinal=0,
                    kind=ProviderClaimKind.EVENT_PROPOSAL,
                    package_ordinal=None,
                    camera_ordinal=None,
                    interval=ProviderClaimInterval(start_ns=start_ns, end_ns=end_ns),
                    label="fixture-action",
                    observation=ProviderObservation.PROPOSED,
                    evidence_tokens=tuple(dict.fromkeys(proposal_tokens)),
                    model_reported_score=0.8,
                    conflict_codes=(),
                ),
            ),
            abstained=False,
        )
        return canonical_json_bytes(payload)
    if request.task is not VisionTask.FUSION_ADJUDICATION:
        raise ValueError(f"unsupported local fixture task: {request.task.value}")
    context_json = request.metadata.get(CANONICAL_FINAL_FUSION_CONTEXT_METADATA_KEY)
    if context_json is None:
        raise ValueError("local final fusion request lacks its refined-action context")
    final_context = CanonicalFinalFusionContext.model_validate_json(
        context_json,
        strict=True,
    )
    assigned_actions = tuple(
        action
        for action in final_context.actions
        if action.action_ordinal % part.part_count == part.ordinal
    )
    claims = tuple(
        ProviderTaskClaim(
            claim_ordinal=claim_ordinal,
            kind=ProviderClaimKind.FUSION_HYPOTHESIS,
            package_ordinal=None,
            camera_ordinal=None,
            interval=ProviderClaimInterval(
                start_ns=action.refined_interval.start_ns,
                end_ns=action.refined_interval.end_ns,
            ),
            label=action.label,
            observation=ProviderObservation.PROPOSED,
            evidence_tokens=(
                entries[
                    min(
                        items,
                        key=lambda item: (
                            abs(
                                item.aligned_timestamp_ns
                                - (
                                    action.refined_interval.start_ns
                                    + action.refined_interval.end_ns
                                )
                                // 2
                            ),
                            item.provider_item_ordinal,
                        ),
                    ).provider_item_ordinal
                ].correlation_token,
            ),
            model_reported_score=0.8,
            conflict_codes=(),
        )
        for claim_ordinal, action in enumerate(assigned_actions)
    )
    return canonical_json_bytes(
        ProviderClaimPayload(
            claims=claims,
            abstained=False,
        )
    )


def _source_run_binding(path: Path) -> tuple[str, str, datetime]:
    try:
        source_bytes = path.read_bytes()
    except OSError as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
            f"cannot read fixture source: {error}",
        ) from error
    try:
        CanonicalSourceFixture.model_validate_json(source_bytes, strict=True)
    except ValidationError as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
            f"invalid fixture source: {error}",
        ) from error
    return (
        exact_bytes_sha256(source_bytes),
        LOCAL_CANONICAL_EXECUTION_TIME,
        _LOCAL_CANONICAL_EXECUTION_DATETIME,
    )


def _hash_source_file(path: Path, *, label: str) -> str:
    digest = hashlib.sha256()
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError(f"{path} is not a regular file")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
            f"cannot read {label}: {error}",
        ) from error
    return digest.hexdigest()


def _receipt(
    committed: CommittedPrimaryCompletion,
    *,
    replayed: bool,
    fixture_inference_calls: int,
) -> CanonicalLocalRunReceipt:
    publications = committed.action_event_publications.publications
    return CanonicalLocalRunReceipt(
        schema_version="1.0",
        ok=True,
        run_id=committed.processing_run.run_id,
        recording_identity=committed.processing_run.recording_identity,
        status=committed.detail.status,
        command_sha256=committed.command_sha256,
        completion_semantic_sha256=committed.completion.semantic_sha256,
        event_ids=tuple(item.payload.event_id for item in publications),
        revision_ids=tuple(item.revision.revision_id for item in publications),
        outbox_ids=tuple(item.outbox_id for item in committed.outbox),
        outbox_count=len(committed.outbox),
        replayed=replayed,
        fixture_inference_calls=fixture_inference_calls,
        network_call_count=0,
        evidence_class="LOCAL_CONFORMANCE",
        production_eligible=False,
    )


def _schema_ref(registry: SchemaRegistry, schema_id: str, version: str) -> JsonSchemaRef:
    ref = registry.resolve_version(schema_id, version).ref
    return JsonSchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _run_id(
    *,
    source_binding_sha256: str,
    execution_policy_sha256: str,
    runtime_policy_sha256: str,
    run_key: str,
) -> str:
    material = ":".join(
        (
            LOCAL_CANONICAL_COMPOSITION_VERSION,
            LOCAL_CANONICAL_EXECUTION_CLOCK_VERSION,
            LOCAL_CANONICAL_EXECUTION_TIME,
            source_binding_sha256,
            execution_policy_sha256,
            runtime_policy_sha256,
            run_key,
        )
    )
    return str(uuid5(NAMESPACE_URL, f"robata:canonical-local-run:{material}"))


def _stable_uuid(namespace: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{value}"))


def _require_path(value: Path, field: str) -> Path:
    if not isinstance(value, Path):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            f"{field} must be pathlib.Path",
        )
    return value.resolve()


def _require_run_key(run_key: str) -> None:
    if not isinstance(run_key, str) or not run_key.strip() or len(run_key) > 128:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "run_key must be a nonblank string of at most 128 characters",
        )


__all__ = [
    "LOCAL_CANONICAL_COMPOSITION_VERSION",
    "LOCAL_CANONICAL_EXECUTION_CLOCK_VERSION",
    "LOCAL_CANONICAL_EXECUTION_TIME",
    "CanonicalLocalCompositionError",
    "CanonicalLocalCompositionErrorCode",
    "CanonicalLocalRunReceipt",
    "run_local_canonical_fixture",
    "run_local_canonical_mcap",
]
