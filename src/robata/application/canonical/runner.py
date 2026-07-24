"""Canonical offline pipeline composition and state progression."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import cast

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.boundary_windows import (
    CanonicalBoundaryRefinementWindow,
    canonical_boundary_refinement_lineage,
)
from robata.application.canonical.logical_nodes import (
    canonical_call_barrier_logical_node,
    canonical_call_part_logical_node,
    canonical_call_reduction_logical_node,
    canonical_candidate_event_logical_node,
    canonical_candidate_reduction_logical_node,
    canonical_coarse_qa_logical_node,
    canonical_dense_qa_result_logical_node,
    canonical_enrichment_logical_node,
    canonical_event_hypothesis_logical_node,
    canonical_event_proposal_result_logical_node,
    canonical_fusion_reduction_logical_node,
    canonical_input_plan_logical_node,
    canonical_output_decision_logical_node,
    canonical_package_set_logical_node,
    canonical_parsed_claim_logical_node,
    canonical_provisional_fusion_result_logical_node,
    canonical_provisional_physical_action_logical_node,
    canonical_qa_completion_logical_node,
    canonical_root_window_logical_node,
    canonical_selected_output_logical_node,
    canonical_selection_logical_node,
)
from robata.application.canonical.models import (
    CanonicalCandidateDenseWindow,
    CanonicalOfflineConfigurationError,
    CanonicalOfflineError,
    CanonicalOfflineExecutionPolicy,
    CanonicalOfflinePartResult,
    CanonicalOfflinePartStatus,
    CanonicalOfflineRunStatus,
    CanonicalOfflineStage,
    CanonicalRootWindow,
    _canonical_error,
    _strict_context,
    canonical_candidate_dense_lineage,
    canonical_lineage,
)
from robata.application.canonical.output_admission import (
    CANONICAL_FINAL_FUSION_CONTEXT_METADATA_KEY,
    CanonicalFinalFusionContext,
    FusionEventHypothesisProjector,
    validate_final_fusion_reduction,
)
from robata.application.canonical.projections import _stable_uuid
from robata.application.canonical.reduction import (
    _build_canonical_fusion_reduction,
    _OrderedProviderClaimReducer,
    _reduce_provider_claim_payloads,
)
from robata.application.canonical.result_validation import (
    CanonicalActionEvidenceExecution,
    CanonicalBoundaryRefinementExecution,
    CanonicalBoundaryRefinementPassExecution,
    CanonicalDenseQAExecution,
    CanonicalOfflineRunResult,
    canonical_action_evidence_execution_membership_lineage,
    canonical_boundary_refinement_execution_membership_lineage,
    canonical_dense_qa_execution_membership_lineage,
)
from robata.application.canonical.runner_support import (
    _package_inputs,
    _rendered_prompt_bytes,
    _require_canonical_uuid,
    _rfc3339_datetime,
    _schema_ref,
    _terminal_raw_artifact_id,
    _timestamp,
    _utc_now,
    _validate_input_plan_chain,
    _validate_materialized_chain,
    _validate_package_set_chain,
    _validate_processing_run_binding,
    _validated_capabilities,
)
from robata.application.canonical_run_membership import (
    CanonicalProcessingRunContext,
    CanonicalProcessingRunPrimaryStatus,
    CanonicalRunMembershipError,
    CanonicalRunMembershipJournal,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import (
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)
from robata.contracts.logical_nodes import LogicalNode, RunNodeRole
from robata.contracts.pipeline import CameraQAStatus, SamplingPurpose
from robata.contracts.sampling_plan import SamplingPlan
from robata.contracts.schema_registry import SchemaRegistry
from robata.contracts.temporal import PackageLineage, TemporalPackageSet
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementOutcome,
    BoundaryRefinementPolicy,
    BoundaryRefinementProjectionError,
    BoundaryRefinementProjector,
    BoundaryRefinementRole,
)
from robata.event_pipeline.candidate import (
    CandidateReducer,
    CandidateReductionError,
    CandidateReductionPolicy,
    CanonicalCandidateEvent,
)
from robata.event_pipeline.evidence import (
    ActionEvidenceOutcome,
    ActionEvidenceProjectionError,
    ActionEvidenceProjector,
)
from robata.event_pipeline.proposer import EventProposalProjector
from robata.event_pipeline.provisional_fusion import (
    ProvisionalFusionError,
    ProvisionalFusionPolicy,
    ProvisionalPhysicalAction,
    ProvisionalPhysicalActionFuser,
)
from robata.inference.adapter import (
    BatchVisionModelAdapter,
    JsonSchemaRef,
    PackageInput,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionModelAdapter,
)
from robata.inference.call_barrier import (
    InferenceCallBarrierCoordinator,
    InferenceCallBarrierError,
    InferenceCallBarrierStorage,
    InferenceCallReduction,
    InMemoryInferenceCallBarrierStorage,
)
from robata.inference.enrichment import (
    ENRICHED_OUTPUT_SCHEMA_ID,
    PROVIDER_CLAIM_SCHEMA_ID,
    EnrichmentAuthorityContext,
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    ProviderClaimEnricher,
    ProviderClaimEnrichmentError,
    ProviderClaimKind,
    ProviderClaimPayload,
    ProviderReferenceCatalog,
    RawProviderResponseArtifact,
    SelectedAttemptOutput,
    enrichment_logical_digest,
)
from robata.inference.evidence import (
    InferenceEvidenceStore,
    InferenceEvidenceStoreError,
    InMemoryInferenceEvidenceStore,
)
from robata.inference.input_plan import (
    INPUT_PLAN_UUID_NAMESPACE,
    REQUEST_CATALOG_UUID_NAMESPACE,
    InferenceCallPart,
    InferenceInputPlan,
    InputPlanTarget,
    PromptOutputContract,
)
from robata.inference.models import (
    InferenceAttemptSelection,
    InferenceStatus,
    ModelCapabilities,
    ModelInference,
    Retryability,
    VisionTask,
)
from robata.inference.offline_fixture import (
    RawProviderBytesStore,
    RawProviderBytesStoreError,
    StrictProviderClaimParseError,
    StrictProviderClaimParser,
)
from robata.inference.orchestrator import (
    InferenceLedger,
    InferenceLedgerError,
    InferenceOrchestrationError,
    InferenceOrchestrator,
    InferencePolicy,
    InMemoryInferenceLedger,
)
from robata.inference.preparation import (
    InputPlanPreparer,
    InputPreparationError,
    RenderedItemFactory,
    applicable_limits_from_capabilities,
)
from robata.ports.logical_node_registry import LogicalNodeRegistry, LogicalNodeRegistryError
from robata.qa_pipeline.coarse import (
    CoarseQAProjectionError,
    CoarseQAProjector,
    CoarseQAResult,
)
from robata.qa_pipeline.completion import (
    QACompletionProjector,
    QACompletionStatus,
)
from robata.qa_pipeline.dense import (
    CameraDenseResult,
    DenseQAInputPlanRef,
    DenseQAOutputRef,
    DenseQAPackageRef,
    DenseQAProjectionError,
    DenseQAProjector,
    DenseQAUnitEvidence,
    DenseQAWorkUnit,
)
from robata.queue.barrier import BarrierCoordinator, BarrierStorage, InMemoryBarrierStorage
from robata.queue.stage import StageStatus
from robata.runtime.observability import (
    RuntimeObserver,
    runtime_increment,
    runtime_span,
)
from robata.sampling.materializer import (
    CanonicalSixCameraFrameIndex,
    FrameArtifactResolver,
    MaterializedTemporalPackage,
    OfflineTemporalPackageMaterializer,
    PackageMaterializationError,
)
from robata.sampling.package_set import PackageSetBuilder, sampling_plan_digest


class _CanonicalRunMembershipPublicationError(RuntimeError):
    """A typed node or its immutable run attachment could not be published."""


class _QAStageError(RuntimeError):
    """One concrete QA inference stage stopped before its deterministic projection."""

    def __init__(
        self,
        status: CanonicalOfflineRunStatus,
        error: CanonicalOfflineError,
    ) -> None:
        super().__init__(error.detail)
        self.status = status
        self.error = error


class _DenseQAExecutionError(RuntimeError):
    """One planned dense unit could not produce complete normalized evidence."""

    def __init__(self, error: CanonicalOfflineError) -> None:
        super().__init__(error.detail)
        self.error = error


class _BoundaryRefinementExecutionError(RuntimeError):
    """One provisional action could not close both boundary roles."""

    def __init__(
        self,
        status: CanonicalOfflineRunStatus,
        error: CanonicalOfflineError,
    ) -> None:
        super().__init__(error.detail)
        self.status = status
        self.error = error


class _ActionEvidenceExecutionError(RuntimeError):
    """One candidate could not produce a complete canonical action-evidence set."""

    def __init__(
        self,
        status: CanonicalOfflineRunStatus,
        error: CanonicalOfflineError,
    ) -> None:
        super().__init__(error.detail)
        self.status = status
        self.error = error


def _canonical_call_dependency_sha256(
    dependency_config: Mapping[str, object] | None,
) -> str | None:
    dependencies = dict(dependency_config or {})
    if not dependencies:
        return None
    return semantic_sha256(
        {
            "semantic_projection_version": "canonical-call-dependency-v1",
            "dependencies": dependencies,
        }
    )


class _CountingVisionModelAdapter:
    """Observe actual adapter dispatches without extending the provider port."""

    def __init__(
        self,
        delegate: VisionModelAdapter,
        *,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        self._delegate = delegate
        self._runtime_observer = runtime_observer
        self._infer_calls = 0

    @property
    def provider(self) -> str:
        return self._delegate.provider

    @property
    def infer_calls(self) -> int:
        return self._infer_calls

    async def capabilities(
        self,
        model_name: str,
        model_version: str,
    ) -> ModelCapabilities:
        return await self._delegate.capabilities(model_name, model_version)

    async def infer(
        self,
        request: VisionInferenceRequest,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        self._infer_calls += 1
        attributes = {
            "provider": self.provider,
            "task": request.task.value,
        }
        runtime_increment(
            self._runtime_observer,
            "inference.provider_dispatches",
            attributes=attributes,
        )
        with runtime_span(
            self._runtime_observer,
            "inference.provider_dispatch",
            attributes,
        ):
            outcome = await self._delegate.infer(request)
        runtime_increment(
            self._runtime_observer,
            "inference.provider_outcomes",
            attributes={
                **attributes,
                "outcome": type(outcome).__name__,
            },
        )
        return outcome


class _CountingBatchVisionModelAdapter(_CountingVisionModelAdapter):
    """Preserve provider dispatch metrics for a native batch-capable adapter."""

    async def infer_batch(
        self,
        requests: tuple[VisionInferenceRequest, ...],
    ) -> tuple[VisionInferenceSuccess | VisionInferenceFailure, ...]:
        batch_delegate = cast(BatchVisionModelAdapter, self._delegate)
        self._infer_calls += len(requests)
        attributes: dict[str, str | int] = {
            "provider": self.provider,
            "batch_size": len(requests),
        }
        runtime_increment(
            self._runtime_observer,
            "inference.provider_batch_dispatches",
            attributes=attributes,
        )
        with runtime_span(
            self._runtime_observer,
            "inference.provider_batch_dispatch",
            attributes,
        ):
            outcomes = await batch_delegate.infer_batch(requests)
        for outcome in outcomes:
            runtime_increment(
                self._runtime_observer,
                "inference.provider_outcomes",
                attributes={
                    "provider": self.provider,
                    "outcome": type(outcome).__name__,
                },
            )
        return outcomes


class CanonicalOfflinePipeline:
    """Run the canonical post-admission path through provider-neutral ports."""

    def __init__(
        self,
        *,
        package_builder: PackageSetBuilder,
        materializer: OfflineTemporalPackageMaterializer,
        input_preparer: InputPlanPreparer,
        adapter: VisionModelAdapter,
        raw_store: RawProviderBytesStore,
        parser: StrictProviderClaimParser,
        coarse_qa_policy: InferencePolicy,
        dense_qa_policy: InferencePolicy,
        event_proposal_policy: InferencePolicy | None = None,
        action_evidence_policy: InferencePolicy,
        boundary_refinement_policy: InferencePolicy,
        inference_policy: InferencePolicy,
        schema_registry: SchemaRegistry,
        logical_node_registry: LogicalNodeRegistry,
        execution_policy: CanonicalOfflineExecutionPolicy,
        inference_ledger: InferenceLedger | None = None,
        evidence_store: InferenceEvidenceStore | None = None,
        barrier_storage: BarrierStorage | None = None,
        call_barrier_storage: InferenceCallBarrierStorage | None = None,
        max_concurrent_call_parts: int = 1,
        max_inference_batch_size: int = 8,
        max_inference_batch_queue_delay_ms: int = 5,
        clock: Callable[[], datetime] | None = None,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not isinstance(package_builder, PackageSetBuilder):
            raise TypeError("package_builder must be a PackageSetBuilder")
        if not isinstance(materializer, OfflineTemporalPackageMaterializer):
            raise TypeError("materializer must be an OfflineTemporalPackageMaterializer")
        if not isinstance(input_preparer, InputPlanPreparer):
            raise TypeError("input_preparer must be an InputPlanPreparer")
        if not isinstance(raw_store, RawProviderBytesStore):
            raise TypeError("raw_store must implement RawProviderBytesStore")
        if not isinstance(parser, StrictProviderClaimParser):
            raise TypeError("parser must be a StrictProviderClaimParser")
        if not isinstance(coarse_qa_policy, InferencePolicy):
            raise TypeError("coarse_qa_policy must be an InferencePolicy")
        if not isinstance(dense_qa_policy, InferencePolicy):
            raise TypeError("dense_qa_policy must be an InferencePolicy")
        if event_proposal_policy is None:
            event_proposal_policy = inference_policy.model_copy(
                update={
                    "policy_version": f"{inference_policy.policy_version}-event-proposal",
                    "task": VisionTask.EVENT_PROPOSAL,
                    "prompt_version": "event-proposal-derived-v1",
                }
            )
        if not isinstance(event_proposal_policy, InferencePolicy):
            raise TypeError("event_proposal_policy must be an InferencePolicy")
        if not isinstance(action_evidence_policy, InferencePolicy):
            raise TypeError("action_evidence_policy must be an InferencePolicy")
        if not isinstance(boundary_refinement_policy, InferencePolicy):
            raise TypeError("boundary_refinement_policy must be an InferencePolicy")
        if not isinstance(inference_policy, InferencePolicy):
            raise TypeError("inference_policy must be an InferencePolicy")
        if not isinstance(schema_registry, SchemaRegistry):
            raise TypeError("schema_registry must be a SchemaRegistry")
        if not callable(getattr(logical_node_registry, "attach_run_node", None)):
            raise TypeError("logical_node_registry must implement attach_run_node")
        if not isinstance(execution_policy, CanonicalOfflineExecutionPolicy):
            raise TypeError("execution_policy must be a CanonicalOfflineExecutionPolicy")
        if (inference_ledger is None) != (evidence_store is None):
            raise CanonicalOfflineConfigurationError(
                "durable inference evidence requires both ledger and evidence store"
            )
        if (barrier_storage is None) != (call_barrier_storage is None):
            raise CanonicalOfflineConfigurationError(
                "durable call barriers require both generic and call barrier storage"
            )
        if evidence_store is not None:
            if not isinstance(evidence_store, InferenceEvidenceStore):
                raise TypeError("evidence_store must implement InferenceEvidenceStore")
            ledger_identity: object = inference_ledger
            raw_store_identity: object = raw_store
            evidence_store_identity: object = evidence_store
            if (
                ledger_identity is not evidence_store_identity
                or raw_store_identity is not evidence_store_identity
            ):
                raise CanonicalOfflineConfigurationError(
                    "durable inference ledger, raw store, and evidence store must be one object"
                )
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if (
            isinstance(max_concurrent_call_parts, bool)
            or not isinstance(max_concurrent_call_parts, int)
            or not 1 <= max_concurrent_call_parts <= 64
        ):
            raise ValueError("max_concurrent_call_parts must be between one and 64")
        if (
            isinstance(max_inference_batch_size, bool)
            or not isinstance(max_inference_batch_size, int)
            or not 1 <= max_inference_batch_size <= 256
        ):
            raise ValueError("max_inference_batch_size must be between one and 256")
        if (
            isinstance(max_inference_batch_queue_delay_ms, bool)
            or not isinstance(max_inference_batch_queue_delay_ms, int)
            or not 0 <= max_inference_batch_queue_delay_ms <= 1_000
        ):
            raise ValueError("max_inference_batch_queue_delay_ms must be between zero and 1000")
        self._package_builder = package_builder
        self._materializer = materializer
        self._input_preparer = input_preparer
        self._adapter = adapter
        counting_adapter_type = (
            _CountingBatchVisionModelAdapter
            if callable(getattr(adapter, "infer_batch", None))
            else _CountingVisionModelAdapter
        )
        self._dispatch_adapter = counting_adapter_type(adapter, runtime_observer=runtime_observer)
        self._raw_store = raw_store
        self._parser = parser
        self._coarse_qa_policy = coarse_qa_policy
        self._dense_qa_policy = dense_qa_policy
        self._event_proposal_policy = event_proposal_policy
        self._action_evidence_policy = action_evidence_policy
        self._boundary_refinement_policy = boundary_refinement_policy
        self._inference_policy = inference_policy
        self._schema_registry = schema_registry
        self._logical_node_registry = logical_node_registry
        self._execution_policy = execution_policy
        self._max_concurrent_call_parts = max_concurrent_call_parts
        self._clock = clock or _utc_now
        self._runtime_observer = runtime_observer
        self._validate_configuration()

        self._ledger = (
            inference_ledger if inference_ledger is not None else InMemoryInferenceLedger()
        )
        self._evidence_store = (
            evidence_store if evidence_store is not None else InMemoryInferenceEvidenceStore()
        )
        schema_registry.resolve_exact(_schema_ref(coarse_qa_policy.output_schema))
        schema_registry.resolve_exact(_schema_ref(dense_qa_policy.output_schema))
        schema_registry.resolve_exact(_schema_ref(event_proposal_policy.output_schema))
        schema_registry.resolve_exact(_schema_ref(action_evidence_policy.output_schema))
        schema_registry.resolve_exact(_schema_ref(boundary_refinement_policy.output_schema))
        schema_registry.resolve_exact(_schema_ref(inference_policy.output_schema))
        self._orchestrator = InferenceOrchestrator(
            adapters={adapter.provider: self._dispatch_adapter},
            task_policies={
                coarse_qa_policy.task: coarse_qa_policy,
                dense_qa_policy.task: dense_qa_policy,
                event_proposal_policy.task: event_proposal_policy,
                action_evidence_policy.task: action_evidence_policy,
                boundary_refinement_policy.task: boundary_refinement_policy,
                inference_policy.task: inference_policy,
            },
            schema_artifacts={
                item.ref.artifact_id: item.document_bytes for item in schema_registry.entries
            },
            ledger=self._ledger,
            max_batch_size=max_inference_batch_size,
            max_batch_queue_delay_ms=max_inference_batch_queue_delay_ms,
            clock=self._clock,
        )
        self._barrier_storage = (
            barrier_storage if barrier_storage is not None else InMemoryBarrierStorage()
        )
        self._call_barrier_storage = (
            call_barrier_storage
            if call_barrier_storage is not None
            else InMemoryInferenceCallBarrierStorage()
        )
        reducer_key = (
            execution_policy.reduction_policy,
            execution_policy.reduction_policy_version,
        )
        self._call_barrier = InferenceCallBarrierCoordinator(
            barriers=BarrierCoordinator(self._barrier_storage),
            storage=self._call_barrier_storage,
            reducers={reducer_key: _OrderedProviderClaimReducer()},
        )
        self._enricher = ProviderClaimEnricher(schema_registry)
        self._coarse_qa_projector = CoarseQAProjector()
        self._qa_completion_projector = QACompletionProjector()
        self._dense_qa_projector = DenseQAProjector(self._qa_completion_projector.policy_version)
        self._event_proposal_projector = EventProposalProjector()
        self._action_evidence_projector = ActionEvidenceProjector()
        self._candidate_reducer = CandidateReducer(
            CandidateReductionPolicy(version="candidate-reduction-v2")
        )
        self._provisional_fuser = ProvisionalPhysicalActionFuser(
            ProvisionalFusionPolicy.create(
                version=execution_policy.provisional_fusion_policy_version
            )
        )
        self._boundary_refinement_projector = BoundaryRefinementProjector(
            BoundaryRefinementPolicy.create(
                version=execution_policy.boundary_refinement_policy_version
            )
        )
        self._projector = FusionEventHypothesisProjector(
            policy=execution_policy.output_admission_policy,
            projector_version=execution_policy.projector_policy_version,
        )

    @property
    def ledger(self) -> InferenceLedger:
        return self._ledger

    @property
    def evidence_store(self) -> InferenceEvidenceStore:
        return self._evidence_store

    @property
    def call_barrier_storage(self) -> InferenceCallBarrierStorage:
        return self._call_barrier_storage

    @property
    def barrier_storage(self) -> BarrierStorage:
        return self._barrier_storage

    @property
    def adapter(self) -> VisionModelAdapter:
        return self._adapter

    async def run(
        self,
        *,
        processing_run: CanonicalProcessingRunContext,
        admitted_context: AdmittedRecordingContextV2,
        requested_interval: NanosecondInterval,
        sampling_plan: SamplingPlan,
        frame_index: CanonicalSixCameraFrameIndex,
        artifact_resolver: FrameArtifactResolver,
        rendered_item_factory: RenderedItemFactory | None = None,
    ) -> CanonicalOfflineRunResult:
        """Execute one exact local run, retaining every admitted intermediate."""

        context = _strict_context(admitted_context)
        if not isinstance(processing_run, CanonicalProcessingRunContext):
            raise TypeError("processing_run must be a CanonicalProcessingRunContext")
        run_context = CanonicalProcessingRunContext.model_validate(
            processing_run.model_dump(mode="python"), strict=True
        )
        _validate_processing_run_binding(
            processing_run=run_context,
            admitted_context=context,
            execution_policy=self._execution_policy,
        )
        if not isinstance(requested_interval, NanosecondInterval):
            raise TypeError("requested_interval must be a NanosecondInterval")
        if not isinstance(sampling_plan, SamplingPlan):
            raise TypeError("sampling_plan must be a SamplingPlan")
        if not isinstance(frame_index, CanonicalSixCameraFrameIndex):
            raise TypeError("frame_index must be a CanonicalSixCameraFrameIndex")
        if not callable(artifact_resolver):
            raise TypeError("artifact_resolver must be callable")
        if rendered_item_factory is not None and not callable(rendered_item_factory):
            raise TypeError("rendered_item_factory must be callable")

        observed_at = _timestamp(self._clock)
        if _rfc3339_datetime(run_context.started_at) > _rfc3339_datetime(observed_at):
            raise CanonicalOfflineConfigurationError(
                "processing run started_at cannot be later than the pipeline clock"
            )
        created_at = run_context.started_at
        run_id = run_context.run_id
        journal = CanonicalRunMembershipJournal(
            context=run_context,
            registry=self._logical_node_registry,
        )
        infer_calls_before = self._dispatch_adapter.infer_calls
        part_results_accumulator: list[CanonicalOfflinePartResult] = []
        state: dict[str, object] = {
            "schema_version": "1.0",
            "run_id": run_id,
            "processing_run": journal.record,
            "run_memberships": (),
            "recording_identity": context.recording_identity,
            "mcap_id": context.ready_manifest.mcap_id,
            "execution_policy_sha256": self._execution_policy.semantic_sha256,
            "window": None,
            "materialized_package_ids": (),
            "package_set": None,
            "coarse_qa_result": None,
            "dense_qa_executions": (),
            "qa_completion_result": None,
            "event_proposal_result": None,
            "candidate_reduction_result": None,
            "action_evidence_executions": (),
            "provisional_fusion_result": None,
            "boundary_refinement_executions": (),
            "final_fusion_context": None,
            "input_plan": None,
            "reference_catalog": None,
            "part_results": (),
            "barrier_reduction": None,
            "fusion_reduction": None,
            "terminal": None,
            "selection": None,
            "raw_response": None,
            "parsed_claims": None,
            "selected_output": None,
            "enriched_output": None,
            "output_decision": None,
            "hypotheses": (),
            "identity_result": None,
            "attempt_count": 0,
        }

        def finish(
            status: CanonicalOfflineRunStatus,
            error: CanonicalOfflineError | None = None,
        ) -> CanonicalOfflineRunResult:
            part_results = tuple(part_results_accumulator)
            state["part_results"] = part_results
            state["attempt_count"] = sum(item.orchestration_attempt_count for item in part_results)
            if len(part_results) == 1 and part_results[0].part_count == 1:
                only = part_results[0]
                state.update(
                    {
                        "terminal": only.terminal,
                        "selection": only.selection,
                        "raw_response": only.raw_response,
                        "parsed_claims": only.parsed_claims,
                        "selected_output": only.selected_output,
                        "enriched_output": only.enriched_output,
                    }
                )
            elif part_results:
                state.update(
                    {
                        "terminal": None,
                        "selection": None,
                        "raw_response": None,
                        "parsed_claims": None,
                        "selected_output": None,
                        "enriched_output": None,
                    }
                )
            completed_run = journal.complete(
                CanonicalProcessingRunPrimaryStatus(status.value),
            )
            return CanonicalOfflineRunResult.model_validate(
                {
                    **state,
                    "processing_run": completed_run,
                    "run_memberships": journal.memberships,
                    "status": status,
                    "adapter_infer_calls": (
                        self._dispatch_adapter.infer_calls - infer_calls_before
                    ),
                    "error": error,
                },
                strict=True,
            )

        def attach_nodes(
            entries: Sequence[tuple[LogicalNode, RunNodeRole]],
        ) -> None:
            try:
                for node, role in entries:
                    journal.attach(node, role, created_at)
            except (CanonicalRunMembershipError, LogicalNodeRegistryError) as exc:
                raise _CanonicalRunMembershipPublicationError(str(exc)) from exc

        def membership_failure(error: object) -> CanonicalOfflineRunResult:
            return finish(
                CanonicalOfflineRunStatus.RUN_MEMBERSHIP_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.RUN_MEMBERSHIP,
                    "RUN_NODE_PUBLICATION_FAILED",
                    error,
                ),
            )

        try:
            if context.alignment_manifest.reference_timebase != "recording_relative_ns":
                raise CanonicalOfflineConfigurationError(
                    "canonical window requires recording_relative_ns alignment"
                )
            window = CanonicalRootWindow.from_context(
                context=context,
                requested_interval=requested_interval,
                purpose=SamplingPurpose.QA_COARSE,
                window_policy_version=self._execution_policy.window_policy_version,
                created_at=created_at,
            )
            state["window"] = window
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.CONFIGURATION_FAILED,
                _canonical_error(CanonicalOfflineStage.WINDOW, "INVALID_ROOT_WINDOW", exc),
            )
        try:
            attach_nodes(((canonical_root_window_logical_node(window), "ROOT_WINDOW"),))
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        try:
            lineage = canonical_lineage(
                context=context,
                window=window,
                sampling_plan=sampling_plan,
            )
            planned_parts = self._package_builder.plan_parts(window, sampling_plan)
            if not planned_parts:
                raise CanonicalOfflineConfigurationError("root window produced no parts")
            materialized = tuple(
                self._materializer.materialize_admitted(
                    part=part,
                    sampling_plan=sampling_plan,
                    purpose=window.purpose,
                    admitted_context=context,
                    frame_index=frame_index,
                    lineage=lineage,
                    window_id=window.window_id,
                    artifact_resolver=artifact_resolver,
                    created_at=created_at,
                )
                for part in planned_parts
            )
            _validate_materialized_chain(
                context=context,
                window=window,
                lineage=lineage,
                planned_parts=planned_parts,
                materialized=materialized,
            )
            package_set = self._package_builder.build_package_set(
                window,
                sampling_plan,
                context.alignment_manifest.alignment_id,
                lineage=lineage,
                materialized_members=tuple(item.package_ref for item in materialized),
                created_at=created_at,
            )
            _validate_package_set_chain(
                context=context,
                window=window,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                reduction_policy_version=self._package_builder.reduction_policy_version,
            )
            state["materialized_package_ids"] = tuple(
                item.package.package_id for item in materialized
            )
            state["package_set"] = package_set
        except PackageMaterializationError as exc:
            return finish(
                CanonicalOfflineRunStatus.MATERIALIZATION_FAILED,
                _canonical_error(CanonicalOfflineStage.MATERIALIZATION, exc.code.value, exc),
            )
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.MATERIALIZATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.MATERIALIZATION,
                    "PACKAGE_CHAIN_INVALID",
                    exc,
                ),
            )
        try:
            attach_nodes(((canonical_package_set_logical_node(package_set), "PACKAGE_SET"),))
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        try:
            coarse_input_plan, coarse_reference_catalog = await self._prepare_inference_stage(
                task=VisionTask.QA_COARSE,
                inference_policy=self._coarse_qa_policy,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                created_at=created_at,
                rendered_item_factory=rendered_item_factory,
            )
        except asyncio.CancelledError:
            raise
        except (
            InputPreparationError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            return finish(
                CanonicalOfflineRunStatus.CONFIGURATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.PREPARATION,
                    "QA_COARSE_INPUT_PLAN_INVALID",
                    exc,
                ),
            )
        try:
            coarse_qa_result = await self._execute_coarse_qa_call_plan(
                context=context,
                window=window,
                sampling_plan=sampling_plan,
                package_set=package_set,
                input_plan=coarse_input_plan,
                reference_catalog=coarse_reference_catalog,
                created_at=created_at,
            )
        except _QAStageError as exc:
            return finish(exc.status, exc.error)

        state["coarse_qa_result"] = coarse_qa_result
        coarse_qa_node = canonical_coarse_qa_logical_node(coarse_qa_result)
        try:
            attach_nodes(((coarse_qa_node, "COARSE_QA"),))
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)
        try:
            qa_recording_interval = NanosecondInterval(
                start_ns=0,
                end_ns=context.ready_manifest.recording.duration_ns,
            )
            qa_completion_result = self._qa_completion_projector.project(
                coarse_qa_result,
                recording_interval=qa_recording_interval,
            )
        except (TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "QA_COMPLETION_PROJECTION_REJECTED",
                    exc,
                ),
            )
        state["qa_completion_result"] = qa_completion_result
        if qa_completion_result.status is QACompletionStatus.DENSE_REQUIRED:
            dense_executions: list[CanonicalDenseQAExecution] = []
            manifest = qa_completion_result.dense_work_manifest
            for unit in manifest.units:
                try:
                    execution = await self._execute_dense_qa_unit(
                        unit=unit,
                        preliminary_completion_semantic_sha256=(
                            qa_completion_result.semantic_sha256
                        ),
                        dense_manifest_semantic_digest=manifest.semantic_digest,
                        context=context,
                        sampling_plan=sampling_plan,
                        frame_index=frame_index,
                        artifact_resolver=artifact_resolver,
                        created_at=created_at,
                        rendered_item_factory=rendered_item_factory,
                    )
                except asyncio.CancelledError:
                    raise
                except _DenseQAExecutionError as exc:
                    blocked = self._qa_completion_projector.block(
                        coarse_qa_result,
                        exc.error.code,
                        recording_interval=qa_recording_interval,
                    )
                    state["qa_completion_result"] = blocked
                    blocked_node = canonical_qa_completion_logical_node(blocked)
                    try:
                        attach_nodes(((blocked_node, "QA_COMPLETION"),))
                    except (
                        TypeError,
                        ValueError,
                        _CanonicalRunMembershipPublicationError,
                    ) as membership_exc:
                        return membership_failure(membership_exc)
                    return finish(CanonicalOfflineRunStatus.INCOMPLETE, exc.error)

                dense_executions.append(execution)
                state["dense_qa_executions"] = tuple(dense_executions)
                try:
                    attach_nodes(canonical_dense_qa_execution_membership_lineage(execution))
                except (
                    TypeError,
                    ValueError,
                    _CanonicalRunMembershipPublicationError,
                ) as exc:
                    return membership_failure(exc)

            try:
                dense_result = self._dense_qa_projector.project(
                    manifest,
                    tuple(item.unit_evidence for item in dense_executions),
                )
                qa_completion_result = self._qa_completion_projector.project(
                    coarse_qa_result,
                    dense_result,
                    recording_interval=qa_recording_interval,
                )
            except (DenseQAProjectionError, TypeError, ValueError) as exc:
                failure_code = "QA_DENSE_RESULT_REJECTED"
                blocked = self._qa_completion_projector.block(
                    coarse_qa_result,
                    failure_code,
                    recording_interval=qa_recording_interval,
                )
                state["qa_completion_result"] = blocked
                blocked_node = canonical_qa_completion_logical_node(blocked)
                try:
                    attach_nodes(((blocked_node, "QA_COMPLETION"),))
                except (
                    TypeError,
                    ValueError,
                    _CanonicalRunMembershipPublicationError,
                ) as membership_exc:
                    return membership_failure(membership_exc)
                return finish(
                    CanonicalOfflineRunStatus.INCOMPLETE,
                    _canonical_error(
                        CanonicalOfflineStage.REDUCTION,
                        failure_code,
                        exc,
                    ),
                )
            state["qa_completion_result"] = qa_completion_result
            qa_completion_node = canonical_qa_completion_logical_node(qa_completion_result)
            dense_result_node = canonical_dense_qa_result_logical_node(dense_result)
            try:
                attach_nodes(
                    (
                        (dense_result_node, "DENSE_QA_RESULT"),
                        (qa_completion_node, "QA_COMPLETION"),
                    )
                )
            except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
                return membership_failure(exc)
        else:
            qa_completion_node = canonical_qa_completion_logical_node(qa_completion_result)
            try:
                attach_nodes(((qa_completion_node, "QA_COMPLETION"),))
            except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
                return membership_failure(exc)

        if qa_completion_result.status is QACompletionStatus.QA_INCOMPLETE:
            detail = (
                "dense QA did not resolve every required six-camera observation"
                if qa_completion_result.dense_result is not None
                else "coarse QA contains unknown evidence and cannot admit downstream work"
            )
            return finish(
                CanonicalOfflineRunStatus.INCOMPLETE,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "QA_INCOMPLETE",
                    detail,
                ),
            )

        proposal_dependency = {
            "qa_completion_semantic_sha256": qa_completion_node.semantic_sha256,
            "qa_completion_policy_version": qa_completion_result.policy_version,
        }
        try:
            proposal_input_plan, proposal_reference_catalog = await self._prepare_inference_stage(
                task=VisionTask.EVENT_PROPOSAL,
                inference_policy=self._event_proposal_policy,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                created_at=created_at,
                rendered_item_factory=rendered_item_factory,
                dependency_config=proposal_dependency,
            )
            (
                _proposal_part_results,
                _proposal_barrier_reduction,
                proposal_enriched_outputs,
            ) = await self._execute_qa_call_plan(
                task=VisionTask.EVENT_PROPOSAL,
                inference_policy=self._event_proposal_policy,
                context=context,
                window=window,
                sampling_plan=sampling_plan,
                package_set=package_set,
                input_plan=proposal_input_plan,
                reference_catalog=proposal_reference_catalog,
                created_at=created_at,
                code_prefix="EVENT_PROPOSAL",
                dependency_config=proposal_dependency,
            )
            proposal_result = self._event_proposal_projector.project(
                input_plan=proposal_input_plan,
                package_set=package_set,
                enriched_outputs=proposal_enriched_outputs,
            )
            state["event_proposal_result"] = proposal_result
            attach_nodes(
                (
                    (
                        canonical_event_proposal_result_logical_node(proposal_result),
                        "EVENT_PROPOSAL_RESULT",
                    ),
                )
            )
            candidate_result = self._candidate_reducer.reduce(
                proposal_result,
                package_set=package_set,
            )
            state["candidate_reduction_result"] = candidate_result
            attach_nodes(
                (
                    (
                        canonical_candidate_reduction_logical_node(candidate_result),
                        "CANDIDATE_REDUCTION",
                    ),
                    *(
                        (canonical_candidate_event_logical_node(item), "CANDIDATE_EVENT")
                        for item in candidate_result.candidates
                    ),
                )
            )
            if candidate_result.no_events:
                return finish(CanonicalOfflineRunStatus.NO_EVENTS, None)
        except asyncio.CancelledError:
            raise
        except _QAStageError as exc:
            return finish(exc.status, exc.error)
        except _CanonicalRunMembershipPublicationError as exc:
            return membership_failure(exc)
        except (
            CandidateReductionError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "EVENT_PROPOSAL_REJECTED",
                    exc,
                ),
            )

        action_evidence_executions: list[CanonicalActionEvidenceExecution] = []
        for candidate in candidate_result.candidates:
            try:
                action_execution = await self._execute_action_evidence_candidate(
                    candidate=candidate,
                    parent_window=window,
                    qa_completion_semantic_sha256=qa_completion_node.semantic_sha256,
                    candidate_reduction_semantic_sha256=candidate_result.semantic_sha256,
                    context=context,
                    sampling_plan=sampling_plan,
                    frame_index=frame_index,
                    artifact_resolver=artifact_resolver,
                    created_at=created_at,
                    rendered_item_factory=rendered_item_factory,
                )
                action_evidence_executions.append(action_execution)
                state["action_evidence_executions"] = tuple(action_evidence_executions)
                attach_nodes(
                    canonical_action_evidence_execution_membership_lineage(action_execution)
                )
            except asyncio.CancelledError:
                raise
            except _ActionEvidenceExecutionError as exc:
                return finish(exc.status, exc.error)
            except _CanonicalRunMembershipPublicationError as exc:
                return membership_failure(exc)
            except (TypeError, ValueError) as exc:
                return finish(
                    CanonicalOfflineRunStatus.INVALID_OUTPUT,
                    _canonical_error(
                        CanonicalOfflineStage.REDUCTION,
                        "ACTION_EVIDENCE_EXECUTION_REJECTED",
                        exc,
                    ),
                )

        indeterminate = tuple(
            item
            for item in action_evidence_executions
            if item.evidence_result.outcome is ActionEvidenceOutcome.INDETERMINATE
        )
        if indeterminate:
            return finish(
                CanonicalOfflineRunStatus.INCOMPLETE,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "ACTION_EVIDENCE_INDETERMINATE",
                    "one or more candidates lack positive or complete negative action evidence",
                ),
            )
        try:
            provisional_fusion_result = self._provisional_fuser.fuse(
                candidate_result,
                tuple(item.evidence_result for item in action_evidence_executions),
            )
            state["provisional_fusion_result"] = provisional_fusion_result
            attach_nodes(
                (
                    (
                        canonical_provisional_fusion_result_logical_node(provisional_fusion_result),
                        "PROVISIONAL_ACTION_FUSION_RESULT",
                    ),
                    *(
                        (
                            canonical_provisional_physical_action_logical_node(action),
                            "PROVISIONAL_PHYSICAL_ACTION",
                        )
                        for action in provisional_fusion_result.actions
                    ),
                )
            )
        except _CanonicalRunMembershipPublicationError as exc:
            return membership_failure(exc)
        except (ProvisionalFusionError, TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "PROVISIONAL_FUSION_REJECTED",
                    exc,
                ),
            )

        if provisional_fusion_result.no_actions:
            return finish(CanonicalOfflineRunStatus.NO_EVENTS, None)

        action_evidence_digests = tuple(
            item.evidence_result.semantic_sha256 for item in action_evidence_executions
        )
        boundary_refinement_executions: list[CanonicalBoundaryRefinementExecution] = []
        for action in provisional_fusion_result.actions:
            try:
                boundary_execution = await self._execute_boundary_refinement_action(
                    action=action,
                    provisional_fusion_semantic_sha256=(provisional_fusion_result.semantic_sha256),
                    parent_window=window,
                    qa_completion_semantic_sha256=qa_completion_node.semantic_sha256,
                    candidate_reduction_semantic_sha256=candidate_result.semantic_sha256,
                    action_evidence_result_semantic_sha256s=action_evidence_digests,
                    context=context,
                    sampling_plan=sampling_plan,
                    frame_index=frame_index,
                    artifact_resolver=artifact_resolver,
                    created_at=created_at,
                    rendered_item_factory=rendered_item_factory,
                )
                boundary_refinement_executions.append(boundary_execution)
                state["boundary_refinement_executions"] = tuple(boundary_refinement_executions)
                attach_nodes(
                    canonical_boundary_refinement_execution_membership_lineage(boundary_execution)
                )
            except asyncio.CancelledError:
                raise
            except _BoundaryRefinementExecutionError as exc:
                return finish(exc.status, exc.error)
            except _CanonicalRunMembershipPublicationError as exc:
                return membership_failure(exc)
            except (TypeError, ValueError) as exc:
                return finish(
                    CanonicalOfflineRunStatus.INVALID_OUTPUT,
                    _canonical_error(
                        CanonicalOfflineStage.REDUCTION,
                        "BOUNDARY_REFINEMENT_EXECUTION_REJECTED",
                        exc,
                    ),
                )

        if any(
            item.result.outcome is BoundaryRefinementOutcome.INDETERMINATE
            for item in boundary_refinement_executions
        ):
            return finish(
                CanonicalOfflineRunStatus.INCOMPLETE,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "BOUNDARY_REFINEMENT_INDETERMINATE",
                    "one or more provisional actions lack complete onset/offset evidence",
                ),
            )
        if len(boundary_refinement_executions) != len(provisional_fusion_result.actions):
            return finish(
                CanonicalOfflineRunStatus.INCOMPLETE,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "BOUNDARY_REFINEMENT_COVERAGE_INCOMPLETE",
                    "boundary refinement did not preserve provisional-action cardinality",
                ),
            )
        try:
            final_fusion_context = CanonicalFinalFusionContext.from_boundary_results(
                results=tuple(item.result for item in boundary_refinement_executions),
                recording_identity=context.recording_identity,
                policy_version=self._execution_policy.projector_policy_version,
            )
            state["final_fusion_context"] = final_fusion_context
        except (TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "FINAL_FUSION_CONTEXT_REJECTED",
                    exc,
                ),
            )

        fusion_dependency = {
            "qa_completion_semantic_sha256": qa_completion_node.semantic_sha256,
            "qa_completion_policy_version": qa_completion_result.policy_version,
            "event_proposal_semantic_sha256": proposal_result.semantic_sha256,
            "candidate_reduction_semantic_sha256": candidate_result.semantic_sha256,
            "action_evidence_result_semantic_sha256s": [
                item.evidence_result.semantic_sha256 for item in action_evidence_executions
            ],
            "provisional_fusion_semantic_sha256": provisional_fusion_result.semantic_sha256,
            "final_fusion_policy_version": final_fusion_context.policy_version,
            "final_fusion_context_semantic_sha256": final_fusion_context.semantic_sha256,
            "final_fusion_context": final_fusion_context.model_dump(mode="json"),
        }
        try:
            input_plan, reference_catalog = await self._prepare_inference_stage(
                task=VisionTask.FUSION_ADJUDICATION,
                inference_policy=self._inference_policy,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                created_at=created_at,
                rendered_item_factory=rendered_item_factory,
                dependency_config=fusion_dependency,
            )
            state["input_plan"] = input_plan
            state["reference_catalog"] = reference_catalog
        except asyncio.CancelledError:
            raise
        except (
            InputPreparationError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            return finish(
                CanonicalOfflineRunStatus.CONFIGURATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.PREPARATION,
                    "INPUT_PLAN_INVALID",
                    exc,
                ),
            )
        try:
            attach_nodes(((canonical_input_plan_logical_node(input_plan), "INPUT_PLAN"),))
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        package_inputs = _package_inputs(package_set)
        try:
            self._call_barrier.declare(input_plan, created_at=created_at)
        except InferenceCallBarrierError as exc:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "BARRIER_DECLARATION_FAILED",
                    exc,
                ),
            )
        try:
            attach_nodes(
                (
                    *(
                        (
                            canonical_call_part_logical_node(input_plan, part),
                            "CALL_PART",
                        )
                        for part in input_plan.call_plan.parts
                    ),
                    (canonical_call_barrier_logical_node(input_plan), "CALL_BARRIER"),
                )
            )
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        return await self._execute_declared_call_plan(
            context=context,
            window=window,
            sampling_plan=sampling_plan,
            package_set=package_set,
            package_inputs=package_inputs,
            input_plan=input_plan,
            reference_catalog=reference_catalog,
            created_at=created_at,
            state=state,
            part_results=part_results_accumulator,
            finish=finish,
            attach_nodes=attach_nodes,
            membership_failure=membership_failure,
            final_fusion_context=final_fusion_context,
            dependency_config=fusion_dependency,
        )

    async def _prepare_inference_stage(
        self,
        *,
        task: VisionTask,
        inference_policy: InferencePolicy,
        lineage: PackageLineage,
        package_set: TemporalPackageSet,
        materialized: tuple[MaterializedTemporalPackage, ...],
        created_at: str,
        rendered_item_factory: RenderedItemFactory | None,
        dependency_config: Mapping[str, object] | None = None,
    ) -> tuple[InferenceInputPlan, ProviderReferenceCatalog]:
        with runtime_span(
            self._runtime_observer,
            "inference.prepare_serialize",
            {"task": task.value},
        ):
            return await self._prepare_inference_stage_unobserved(
                task=task,
                inference_policy=inference_policy,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                created_at=created_at,
                rendered_item_factory=rendered_item_factory,
                dependency_config=dependency_config,
            )

    async def _prepare_inference_stage_unobserved(
        self,
        *,
        task: VisionTask,
        inference_policy: InferencePolicy,
        lineage: PackageLineage,
        package_set: TemporalPackageSet,
        materialized: tuple[MaterializedTemporalPackage, ...],
        created_at: str,
        rendered_item_factory: RenderedItemFactory | None,
        dependency_config: Mapping[str, object] | None = None,
    ) -> tuple[InferenceInputPlan, ProviderReferenceCatalog]:
        capabilities = await self._adapter.capabilities(
            inference_policy.model_name,
            inference_policy.model_version,
        )
        capabilities = _validated_capabilities(
            capabilities,
            inference_policy=inference_policy,
            input_preparer=self._input_preparer,
        )
        applicable_limits = applicable_limits_from_capabilities(
            max_images_per_request=capabilities.max_images_per_request,
            max_pixels_per_image=capabilities.max_pixels_per_image,
            max_payload_bytes=capabilities.max_payload_bytes,
            max_input_tokens=capabilities.max_input_tokens,
        )
        target = InputPlanTarget(
            provider=capabilities.provider,
            model_name=capabilities.model_name,
            model_version=capabilities.model_version,
            adapter_version=inference_policy.adapter_version,
            planner_version=self._input_preparer.planner_version,
            capability_snapshot_id=capabilities.snapshot_id,
            capability_snapshot_sha256=capabilities.snapshot_digest,
        )
        request_catalog_id = _stable_uuid(
            REQUEST_CATALOG_UUID_NAMESPACE,
            lineage,
            task.value,
            inference_policy.policy_version,
            inference_policy.prompt_sha256,
            self._execution_policy.semantic_sha256,
            capabilities.snapshot_digest,
        )
        prepared = self._input_preparer.prepare_rendering(
            packages=materialized,
            task=task,
            request_catalog_id=request_catalog_id,
            applicable_limits=applicable_limits,
            created_at=created_at,
            rendered_item_factory=rendered_item_factory,
        )
        prompt_entries = ProviderReferenceCatalog.derive_entries(
            request_catalog_sha256=prepared.request_catalog.semantic_sha256,
            rendered_items=prepared.rendered_items,
            token_policy_version=self._execution_policy.token_policy_version,
        )
        rendered_prompt = _rendered_prompt_bytes(
            inference_policy=inference_policy,
            request_catalog_sha256=prepared.request_catalog.semantic_sha256,
            token_policy_version=self._execution_policy.token_policy_version,
            entries=prompt_entries,
            logical_dependency_sha256=_canonical_call_dependency_sha256(dependency_config),
        )
        enriched_schema = self._required_enriched_schema(inference_policy)
        prompt_output = PromptOutputContract(
            prompt_version=inference_policy.prompt_version,
            prompt_sha256=inference_policy.prompt_sha256,
            rendered_message_sha256=exact_bytes_sha256(rendered_prompt),
            provider_response_schema_sha256=inference_policy.output_schema.sha256,
            enriched_domain_schema_sha256=enriched_schema.sha256,
            protocol_mode="json-schema",
            tool_mode="none",
        )
        input_plan_id = _stable_uuid(
            INPUT_PLAN_UUID_NAMESPACE,
            prepared.request_catalog.semantic_sha256,
            target.capability_snapshot_sha256,
            exact_bytes_sha256(rendered_prompt),
            self._execution_policy.semantic_sha256,
        )
        input_plan = self._input_preparer.finalize(
            prepared=prepared,
            input_plan_id=input_plan_id,
            target=target,
            prompt_output=prompt_output,
            created_at=created_at,
        )
        reference_catalog = ProviderReferenceCatalog.build(
            input_plan=input_plan,
            reference_catalog_id=_stable_uuid(
                "provider-reference-catalog",
                input_plan.semantic_sha256,
                self._execution_policy.token_policy_version,
            ),
            token_policy_version=self._execution_policy.token_policy_version,
            created_at=created_at,
        )
        if reference_catalog.entries != prompt_entries:
            raise CanonicalOfflineConfigurationError(
                "rendered prompt tokens differ from finalized reference catalog"
            )
        _validate_input_plan_chain(
            package_set=package_set,
            materialized=materialized,
            input_plan=input_plan,
            reference_catalog=reference_catalog,
            inference_policy=inference_policy,
            execution_policy=self._execution_policy,
            capabilities=capabilities,
        )
        return input_plan, reference_catalog

    async def _orchestrate_call_part(
        self,
        *,
        task: VisionTask,
        inference_policy: InferencePolicy,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow
        | CanonicalCandidateDenseWindow
        | CanonicalBoundaryRefinementWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        package_inputs: tuple[PackageInput, ...],
        input_plan: InferenceInputPlan,
        part: InferenceCallPart,
        dependency_config: Mapping[str, object] | None = None,
    ) -> tuple[ModelInference, InferenceAttemptSelection | None, int]:
        terminal: ModelInference | None = None
        dependency_fields = dict(dependency_config or {})
        logical_dependency_sha256 = _canonical_call_dependency_sha256(dependency_fields)
        request_metadata: dict[str, str] = {}
        boundary_role = dependency_fields.get("boundary_refinement_role")
        if boundary_role is not None:
            if boundary_role not in {"ONSET", "OFFSET"}:
                raise CanonicalOfflineConfigurationError(
                    "boundary refinement dependency carries an invalid role"
                )
            request_metadata["boundary_refinement_role"] = str(boundary_role)
            boundary_anchor_ns = dependency_fields.get("boundary_anchor_ns")
            if isinstance(boundary_anchor_ns, bool) or not isinstance(boundary_anchor_ns, int):
                raise CanonicalOfflineConfigurationError(
                    "boundary refinement dependency lacks an integer anchor"
                )
            request_metadata["boundary_anchor_ns"] = str(boundary_anchor_ns)

        final_context_value = dependency_fields.get("final_fusion_context")
        if final_context_value is not None:
            if task is not VisionTask.FUSION_ADJUDICATION:
                raise CanonicalOfflineConfigurationError(
                    "final fusion context is attached to a non-fusion task"
                )
            try:
                final_context = CanonicalFinalFusionContext.model_validate_json(
                    canonical_json_bytes(final_context_value),
                    strict=True,
                )
            except ValueError as exc:
                raise CanonicalOfflineConfigurationError(
                    "final fusion dependency context is invalid"
                ) from exc
            if (
                dependency_fields.get("final_fusion_context_semantic_sha256")
                != final_context.semantic_sha256
                or dependency_fields.get("final_fusion_policy_version")
                != final_context.policy_version
            ):
                raise CanonicalOfflineConfigurationError(
                    "final fusion dependency identity is inconsistent"
                )
            request_metadata[CANONICAL_FINAL_FUSION_CONTEXT_METADATA_KEY] = canonical_json_bytes(
                final_context.model_dump(mode="json")
            ).decode("utf-8")
        elif task is VisionTask.FUSION_ADJUDICATION:
            raise CanonicalOfflineConfigurationError(
                "final fusion request lacks its refined-action context"
            )

        for attempt in range(1, self._execution_policy.max_attempts + 1):
            terminal = await self._orchestrator.orchestrate(
                task=task,
                package_set_id=package_set.package_set_id,
                mcap_id=context.ready_manifest.mcap_id,
                camera_mapping_run_id=context.ready_manifest.camera_mapping_run_id,
                alignment_id=context.alignment_manifest.alignment_id,
                start_ns=window.interval.start_ns,
                end_ns=window.interval.end_ns,
                package_inputs=package_inputs,
                rendered_input_digest=part.item_manifest_sha256,
                input_plan=input_plan,
                input_plan_part_ordinal=part.ordinal,
                input_config={
                    "canonical_execution_policy_sha256": (self._execution_policy.semantic_sha256),
                    **dependency_fields,
                    **(
                        {"logical_dependency_sha256": logical_dependency_sha256}
                        if logical_dependency_sha256 is not None
                        else {}
                    ),
                },
                logical_dependency_sha256=logical_dependency_sha256,
                sampling_config={
                    "sampling_plan_version": sampling_plan.version,
                    "sampling_plan_sha256": sampling_plan_digest(
                        sampling_plan,
                        purpose=window.purpose,
                    ),
                },
                metadata=request_metadata,
                attempt=attempt,
                retry_count=attempt - 1,
            )
            if terminal.status is InferenceStatus.SUCCEEDED:
                selection = self._ledger.get_selection(
                    terminal.logical_invocation_id,
                    inference_policy.selection_policy_version,
                )
                if selection is None:
                    raise CanonicalOfflineConfigurationError(
                        "successful invocation has no persisted selection"
                    )
                selected = self._ledger.get_terminal(selection.inference_id)
                if (
                    selected is None
                    or selected.status is not InferenceStatus.SUCCEEDED
                    or not selected.output_valid
                    or selected.input_plan_part_ordinal != part.ordinal
                ):
                    raise CanonicalOfflineConfigurationError(
                        "persisted selection does not reference the valid part success"
                    )
                return selected, selection, attempt

            failure = terminal.failure
            retryable = failure is not None and failure.retryability in {
                Retryability.RETRYABLE,
                Retryability.RATE_LIMITED,
            }
            if not retryable or attempt == self._execution_policy.max_attempts:
                return terminal, None, attempt
        assert terminal is not None
        return terminal, None, self._execution_policy.max_attempts

    def _build_selected_part_lineage(
        self,
        *,
        task: VisionTask,
        inference_policy: InferencePolicy,
        context: AdmittedRecordingContextV2,
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        part: InferenceCallPart,
        selected_terminal: ModelInference,
        selection: InferenceAttemptSelection,
    ) -> tuple[
        RawProviderResponseArtifact | None,
        ParsedProviderClaimArtifact | None,
        SelectedAttemptOutput | None,
        OrchestratorEnrichedOutput | None,
        CanonicalOfflineError | None,
    ]:
        try:
            if (
                selection.inference_id != selected_terminal.inference_id
                or selection.logical_invocation_id != selected_terminal.logical_invocation_id
            ):
                raise CanonicalOfflineConfigurationError(
                    "selection does not reference the selected terminal"
                )
            raw_artifact_id = _terminal_raw_artifact_id(selected_terminal)
            stored_raw = self._raw_store.get(raw_artifact_id)
            if (
                stored_raw.artifact_id != raw_artifact_id
                or stored_raw.request_id != selected_terminal.request_id
                or stored_raw.provider_request_id != selected_terminal.provider_request_id
            ):
                raise CanonicalOfflineConfigurationError(
                    "stored raw bytes do not match the selected terminal request"
                )
            artifact_created_at = selected_terminal.completed_at
            expected_raw = RawProviderResponseArtifact.from_bytes(
                data=stored_raw.data,
                artifact_id=stored_raw.artifact_id,
                media_type=stored_raw.media_type,
                provider_request_id=stored_raw.provider_request_id,
                inference_id=selected_terminal.inference_id,
                provider=selected_terminal.provider,
                model_name=selected_terminal.model_name,
                model_version=selected_terminal.model_version,
                created_at=artifact_created_at,
            )
            parsed_artifact_id = _stable_uuid(
                "parsed-provider-claim",
                selected_terminal.inference_id,
                stored_raw.exact_bytes_sha256,
                inference_policy.output_schema.sha256,
                self._execution_policy.parser_version,
            )
            parsed_claims = self._evidence_store.get_parsed_claim(parsed_artifact_id)
            if parsed_claims is None:
                parsed_claims = self._parser.parse_artifact(
                    stored=stored_raw,
                    inference_id=selected_terminal.inference_id,
                    provider=selected_terminal.provider,
                    model_name=selected_terminal.model_name,
                    model_version=selected_terminal.model_version,
                    provider_claim_schema=inference_policy.output_schema,
                    task=task,
                    artifact_id=parsed_artifact_id,
                    created_at=artifact_created_at,
                )
            if (
                parsed_claims.artifact_id != parsed_artifact_id
                or parsed_claims.raw_response != expected_raw
                or parsed_claims.provider_claim_schema != inference_policy.output_schema
                or parsed_claims.task is not task
                or parsed_claims.parser_version != self._execution_policy.parser_version
                or parsed_claims.created_at != artifact_created_at
                or parsed_claims.payload.model_dump(mode="json")
                != selected_terminal.normalized_output
            ):
                raise CanonicalOfflineConfigurationError(
                    "persisted parsed claims differ from the selected terminal"
                )
            expected_selected = SelectedAttemptOutput.create(parsed_claims, selection)
            selected_output = self._evidence_store.get_selected_output(selection.selection_id)
            if selected_output is None:
                selected_output = expected_selected
            elif selected_output != expected_selected:
                raise InferenceEvidenceStoreError(
                    "persisted selected output differs from its selection and parsed claim"
                )
        except (InferenceEvidenceStoreError, InferenceLedgerError) as exc:
            return (
                None,
                None,
                None,
                None,
                _canonical_error(
                    CanonicalOfflineStage.PARSING,
                    "INFERENCE_EVIDENCE_CONFLICT",
                    exc,
                ),
            )
        except (
            RawProviderBytesStoreError,
            StrictProviderClaimParseError,
            TypeError,
            ValueError,
        ) as exc:
            return (
                None,
                None,
                None,
                None,
                _canonical_error(
                    CanonicalOfflineStage.PARSING,
                    "SELECTED_RAW_OUTPUT_INVALID",
                    exc,
                ),
            )

        try:
            work_digest = semantic_sha256(
                {
                    "recording_identity": context.recording_identity,
                    "input_plan_semantic_sha256": input_plan.semantic_sha256,
                    "input_plan_part_semantic_sha256": part.part_semantic_sha256,
                    "selected_attempt_output_sha256": selected_output.output_sha256,
                    "enrichment_policy_version": (self._execution_policy.enrichment_policy_version),
                }
            )
            authority = EnrichmentAuthorityContext(
                recording_identity=context.recording_identity,
                mcap_id=context.ready_manifest.mcap_id,
                camera_mapping_run_id=context.ready_manifest.camera_mapping_run_id,
                alignment_id=context.alignment_manifest.alignment_id,
                inference_id=selected_terminal.inference_id,
                logical_invocation_id=selected_terminal.logical_invocation_id,
                prompt_version=inference_policy.prompt_version,
                prompt_artifact_id=inference_policy.prompt_artifact_id,
                prompt_sha256=inference_policy.prompt_sha256,
                work_node_type="INFERENCE_ENRICHMENT",
                work_node_logical_key=f"inference-work:{work_digest}",
            )
            enriched_schema = self._required_enriched_schema(inference_policy)
            logical_digest = enrichment_logical_digest(
                selected_attempt_output_sha256=selected_output.output_sha256,
                request_catalog_sha256=input_plan.request_catalog.semantic_sha256,
                target_schema_sha256=enriched_schema.sha256,
                enrichment_policy_version=self._execution_policy.enrichment_policy_version,
            )
            enriched_artifact_id = _stable_uuid("orchestrator-enrichment", logical_digest)
            enriched_output = self._evidence_store.get_enriched_output(enriched_artifact_id)
            if enriched_output is None:
                enriched_output = self._enricher.enrich(
                    input_plan=input_plan,
                    input_plan_part_ordinal=part.ordinal,
                    reference_catalog=reference_catalog,
                    parsed_claims=parsed_claims,
                    selected_attempt=selected_output,
                    authority=authority,
                    enriched_output_schema=enriched_schema,
                    enrichment_policy_version=self._execution_policy.enrichment_policy_version,
                    artifact_id=enriched_artifact_id,
                    created_at=selected_terminal.completed_at,
                )
            elif (
                enriched_output.artifact_id != enriched_artifact_id
                or enriched_output.enrichment_logical_key
                != f"orchestrator-enrichment:{logical_digest}"
                or enriched_output.task is not task
                or enriched_output.selected_attempt != selected_output
                or enriched_output.request_catalog_id
                != input_plan.request_catalog.request_catalog_id
                or enriched_output.request_catalog_sha256
                != input_plan.request_catalog.semantic_sha256
                or enriched_output.reference_catalog_id != reference_catalog.reference_catalog_id
                or enriched_output.reference_catalog_sha256 != reference_catalog.semantic_sha256
                or enriched_output.input_plan_id != input_plan.input_plan_id
                or enriched_output.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or enriched_output.provider_claim_schema != inference_policy.output_schema
                or enriched_output.enriched_output_schema != enriched_schema
                or enriched_output.enrichment_policy_version
                != self._execution_policy.enrichment_policy_version
                or enriched_output.authority != authority
                or enriched_output.created_at != selected_terminal.completed_at
            ):
                raise InferenceEvidenceStoreError(
                    "persisted enriched output differs from the current semantic lineage"
                )
            (
                persisted_parsed,
                persisted_selected,
                persisted_enriched,
            ) = self._evidence_store.append_accepted_lineage(
                parsed_claims,
                selected_output,
                enriched_output,
            )
            if (
                persisted_parsed != parsed_claims
                or persisted_selected != selected_output
                or persisted_enriched != enriched_output
            ):
                raise InferenceEvidenceStoreError(
                    "persisted accepted lineage differs from the completed call lineage"
                )
            parsed_claims = persisted_parsed
            selected_output = persisted_selected
            enriched_output = persisted_enriched
        except (InferenceEvidenceStoreError, InferenceLedgerError) as exc:
            return (
                parsed_claims.raw_response,
                parsed_claims,
                selected_output,
                None,
                _canonical_error(
                    CanonicalOfflineStage.ENRICHMENT,
                    "INFERENCE_EVIDENCE_CONFLICT",
                    exc,
                ),
            )
        except (
            RawProviderBytesStoreError,
            ProviderClaimEnrichmentError,
            TypeError,
            ValueError,
        ) as exc:
            return (
                parsed_claims.raw_response,
                parsed_claims,
                selected_output,
                None,
                _canonical_error(
                    CanonicalOfflineStage.ENRICHMENT,
                    "ENRICHMENT_REJECTED",
                    exc,
                ),
            )
        return (
            parsed_claims.raw_response,
            parsed_claims,
            selected_output,
            enriched_output,
            None,
        )

    async def _execute_one_call_part(
        self,
        *,
        task: VisionTask,
        inference_policy: InferencePolicy,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow
        | CanonicalCandidateDenseWindow
        | CanonicalBoundaryRefinementWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        package_inputs: tuple[PackageInput, ...],
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        part: InferenceCallPart,
        dependency_config: Mapping[str, object] | None = None,
    ) -> CanonicalOfflinePartResult:
        call_attributes = {
            "part_count": part.part_count,
            "part_ordinal": part.ordinal,
            "task": task.value,
        }
        runtime_increment(
            self._runtime_observer,
            "inference.call_parts",
            attributes={"task": task.value},
        )
        with runtime_span(
            self._runtime_observer,
            "inference.orchestration",
            call_attributes,
        ):
            terminal, selection, attempts_used = await self._orchestrate_call_part(
                task=task,
                inference_policy=inference_policy,
                context=context,
                window=window,
                sampling_plan=sampling_plan,
                package_set=package_set,
                package_inputs=package_inputs,
                input_plan=input_plan,
                part=part,
                dependency_config=dependency_config,
            )
        runtime_increment(
            self._runtime_observer,
            "inference.orchestration_attempts",
            attempts_used,
            {"task": task.value},
        )
        if terminal.status is not InferenceStatus.SUCCEEDED:
            runtime_increment(
                self._runtime_observer,
                "inference.call_outcomes",
                attributes={"status": terminal.status.value, "task": task.value},
            )
            failure = terminal.failure
            error = _canonical_error(
                CanonicalOfflineStage.INFERENCE,
                failure.code if failure is not None else terminal.status.value,
                failure.detail if failure is not None else terminal.status.value,
            )
            completion = self._call_barrier.submit_part_terminal(
                input_plan,
                terminal,
                failure_is_final=True,
            )
            return CanonicalOfflinePartResult(
                schema_version="1.0",
                part_ordinal=part.ordinal,
                part_count=part.part_count,
                part_semantic_sha256=part.part_semantic_sha256,
                status=CanonicalOfflinePartStatus.TERMINAL_FAILED,
                orchestration_attempt_count=attempts_used,
                terminal=terminal,
                selection=None,
                completion=completion,
                raw_response=None,
                parsed_claims=None,
                selected_output=None,
                enriched_output=None,
                error=error,
            )

        if selection is None:
            raise CanonicalOfflineConfigurationError(
                "selected success is missing its selection decision"
            )
        with runtime_span(
            self._runtime_observer,
            "inference.parse_validate_enrich",
            call_attributes,
        ):
            raw, parsed, selected_output, enriched, lineage_error = (
                self._build_selected_part_lineage(
                    task=task,
                    inference_policy=inference_policy,
                    context=context,
                    input_plan=input_plan,
                    reference_catalog=reference_catalog,
                    part=part,
                    selected_terminal=terminal,
                    selection=selection,
                )
            )
        runtime_increment(
            self._runtime_observer,
            "inference.call_outcomes",
            attributes={
                "status": ("SUCCEEDED" if lineage_error is None else "POST_SELECTION_INVALID"),
                "task": task.value,
            },
        )
        completion = self._call_barrier.submit_part_terminal(
            input_plan,
            terminal,
            selection=selection,
        )
        return CanonicalOfflinePartResult(
            schema_version="1.0",
            part_ordinal=part.ordinal,
            part_count=part.part_count,
            part_semantic_sha256=part.part_semantic_sha256,
            status=(
                CanonicalOfflinePartStatus.ENRICHED
                if lineage_error is None
                else CanonicalOfflinePartStatus.POST_SELECTION_INVALID
            ),
            orchestration_attempt_count=attempts_used,
            terminal=terminal,
            selection=selection,
            completion=completion,
            raw_response=raw,
            parsed_claims=parsed,
            selected_output=selected_output,
            enriched_output=enriched,
            error=lineage_error,
        )

    async def _execute_call_plan_parts(
        self,
        *,
        task: VisionTask,
        inference_policy: InferencePolicy,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow
        | CanonicalCandidateDenseWindow
        | CanonicalBoundaryRefinementWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        package_inputs: tuple[PackageInput, ...],
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        dependency_config: Mapping[str, object] | None,
    ) -> tuple[CanonicalOfflinePartResult, ...]:
        semaphore = asyncio.Semaphore(self._max_concurrent_call_parts)

        async def execute(part: InferenceCallPart) -> CanonicalOfflinePartResult:
            async with semaphore:
                return await self._execute_one_call_part(
                    task=task,
                    inference_policy=inference_policy,
                    context=context,
                    window=window,
                    sampling_plan=sampling_plan,
                    package_set=package_set,
                    package_inputs=package_inputs,
                    input_plan=input_plan,
                    reference_catalog=reference_catalog,
                    part=part,
                    dependency_config=dependency_config,
                )

        tasks = tuple(asyncio.create_task(execute(part)) for part in input_plan.call_plan.parts)
        try:
            return tuple(await asyncio.gather(*tasks))
        except BaseException:
            for pending_task in tasks:
                if not pending_task.done():
                    pending_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _execute_coarse_qa_call_plan(
        self,
        *,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        created_at: str,
    ) -> CoarseQAResult:
        _, _, enriched_outputs = await self._execute_qa_call_plan(
            task=VisionTask.QA_COARSE,
            inference_policy=self._coarse_qa_policy,
            context=context,
            window=window,
            sampling_plan=sampling_plan,
            package_set=package_set,
            input_plan=input_plan,
            reference_catalog=reference_catalog,
            created_at=created_at,
            code_prefix="QA_COARSE",
        )
        try:
            return self._coarse_qa_projector.project(
                package_set=package_set,
                input_plan=input_plan,
                enriched_outputs=enriched_outputs,
            )
        except (CoarseQAProjectionError, TypeError, ValueError) as exc:
            raise _QAStageError(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "QA_COARSE_PROJECTION_REJECTED",
                    exc,
                ),
            ) from exc

    async def _execute_qa_call_plan(
        self,
        *,
        task: VisionTask,
        inference_policy: InferencePolicy,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow
        | CanonicalCandidateDenseWindow
        | CanonicalBoundaryRefinementWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        created_at: str,
        code_prefix: str,
        dependency_config: Mapping[str, object] | None = None,
    ) -> tuple[
        tuple[CanonicalOfflinePartResult, ...],
        InferenceCallReduction,
        tuple[OrchestratorEnrichedOutput, ...],
    ]:
        with runtime_span(
            self._runtime_observer,
            "inference.call_plan",
            {
                "part_count": len(input_plan.call_plan.parts),
                "task": task.value,
            },
        ):
            return await self._execute_qa_call_plan_unobserved(
                task=task,
                inference_policy=inference_policy,
                context=context,
                window=window,
                sampling_plan=sampling_plan,
                package_set=package_set,
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                created_at=created_at,
                code_prefix=code_prefix,
                dependency_config=dependency_config,
            )

    async def _execute_qa_call_plan_unobserved(
        self,
        *,
        task: VisionTask,
        inference_policy: InferencePolicy,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow
        | CanonicalCandidateDenseWindow
        | CanonicalBoundaryRefinementWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        created_at: str,
        code_prefix: str,
        dependency_config: Mapping[str, object] | None = None,
    ) -> tuple[
        tuple[CanonicalOfflinePartResult, ...],
        InferenceCallReduction,
        tuple[OrchestratorEnrichedOutput, ...],
    ]:
        """Execute one supported plan through the shared terminal-evidence chain."""

        if task not in {
            VisionTask.QA_COARSE,
            VisionTask.QA_DENSE,
            VisionTask.EVENT_PROPOSAL,
            VisionTask.ACTION_EVIDENCE,
            VisionTask.BOUNDARY_REFINEMENT,
        }:
            raise CanonicalOfflineConfigurationError(
                "shared call plan received an unsupported terminal-evidence task"
            )
        if inference_policy.task is not task:
            raise CanonicalOfflineConfigurationError("shared call policy does not match its task")
        package_inputs = _package_inputs(package_set)
        try:
            self._call_barrier.declare(input_plan, created_at=created_at)
        except InferenceCallBarrierError as exc:
            raise _QAStageError(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    f"{code_prefix}_BARRIER_DECLARATION_FAILED",
                    exc,
                ),
            ) from exc

        try:
            part_results = list(
                await self._execute_call_plan_parts(
                    task=task,
                    inference_policy=inference_policy,
                    context=context,
                    window=window,
                    sampling_plan=sampling_plan,
                    package_set=package_set,
                    package_inputs=package_inputs,
                    input_plan=input_plan,
                    reference_catalog=reference_catalog,
                    dependency_config=dependency_config,
                )
            )
        except asyncio.CancelledError:
            raise
        except (
            InferenceOrchestrationError,
            InferenceCallBarrierError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            raise _QAStageError(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    f"{code_prefix}_CALL_PART_EXECUTION_FAILED",
                    exc,
                ),
            ) from exc

        try:
            aggregate = self._call_barrier.get_aggregate_status(input_plan)
        except InferenceCallBarrierError as exc:
            raise _QAStageError(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    f"{code_prefix}_BARRIER_STATUS_FAILED",
                    exc,
                ),
            ) from exc
        if not aggregate.is_complete:
            raise _QAStageError(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    f"{code_prefix}_BARRIER_NOT_TERMINAL",
                    f"declared {task.value} barrier remained open",
                ),
            )
        if aggregate.overall_status is StageStatus.INCOMPLETE:
            failed_ordinals = tuple(
                item.part_ordinal
                for item in part_results
                if item.status is CanonicalOfflinePartStatus.TERMINAL_FAILED
            )
            raise _QAStageError(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    f"{code_prefix}_REQUIRED_PARTS_INCOMPLETE",
                    f"required terminal failures at part ordinals {failed_ordinals}",
                ),
            )
        if aggregate.overall_status is not StageStatus.SUCCEEDED:
            raise _QAStageError(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    f"{code_prefix}_BARRIER_STATUS_INVALID",
                    aggregate.overall_status.value,
                ),
            )

        invalid = next(
            (
                item
                for item in part_results
                if item.status is CanonicalOfflinePartStatus.POST_SELECTION_INVALID
            ),
            None,
        )
        if invalid is not None:
            assert invalid.error is not None
            raise _QAStageError(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                invalid.error,
            )
        enriched_outputs = tuple(
            item.enriched_output for item in part_results if item.enriched_output is not None
        )
        if len(enriched_outputs) != len(input_plan.call_plan.parts):
            raise _QAStageError(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.ENRICHMENT,
                    f"{code_prefix}_ENRICHMENT_SET_INCOMPLETE",
                    f"successful {task.value} barrier lacks exact enriched output coverage",
                ),
            )

        try:
            barrier_reduction = self._call_barrier.reduce(
                input_plan,
                reduced_at=created_at,
            )
            parsed_payloads = tuple(
                item.parsed_claims.payload
                for item in part_results
                if item.parsed_claims is not None
            )
            expected_payload = _reduce_provider_claim_payloads(parsed_payloads)
            reduced_payload = ProviderClaimPayload.model_validate(
                barrier_reduction.normalized_output,
                strict=False,
            )
            if reduced_payload != expected_payload:
                raise CanonicalOfflineConfigurationError(
                    f"{task.value} barrier reduction differs from exact parsed claims"
                )
            return tuple(part_results), barrier_reduction, enriched_outputs
        except (
            InferenceCallBarrierError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            raise _QAStageError(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    f"{code_prefix}_BARRIER_REDUCTION_REJECTED",
                    exc,
                ),
            ) from exc

    async def _execute_boundary_refinement_action(
        self,
        *,
        action: ProvisionalPhysicalAction,
        provisional_fusion_semantic_sha256: str,
        parent_window: CanonicalRootWindow,
        qa_completion_semantic_sha256: str,
        candidate_reduction_semantic_sha256: str,
        action_evidence_result_semantic_sha256s: tuple[str, ...],
        context: AdmittedRecordingContextV2,
        sampling_plan: SamplingPlan,
        frame_index: CanonicalSixCameraFrameIndex,
        artifact_resolver: FrameArtifactResolver,
        created_at: str,
        rendered_item_factory: RenderedItemFactory | None,
    ) -> CanonicalBoundaryRefinementExecution:
        passes: dict[BoundaryRefinementRole, CanonicalBoundaryRefinementPassExecution] = {}
        for role in (BoundaryRefinementRole.ONSET, BoundaryRefinementRole.OFFSET):
            passes[role] = await self._execute_boundary_refinement_role(
                action=action,
                role=role,
                provisional_fusion_semantic_sha256=provisional_fusion_semantic_sha256,
                parent_window=parent_window,
                qa_completion_semantic_sha256=qa_completion_semantic_sha256,
                candidate_reduction_semantic_sha256=candidate_reduction_semantic_sha256,
                action_evidence_result_semantic_sha256s=(action_evidence_result_semantic_sha256s),
                context=context,
                sampling_plan=sampling_plan,
                frame_index=frame_index,
                artifact_resolver=artifact_resolver,
                created_at=created_at,
                rendered_item_factory=rendered_item_factory,
            )
        try:
            onset = passes[BoundaryRefinementRole.ONSET]
            offset = passes[BoundaryRefinementRole.OFFSET]
            result = self._boundary_refinement_projector.reduce(
                action=action,
                onset=onset.role_result,
                offset=offset.role_result,
            )
            return CanonicalBoundaryRefinementExecution(
                action=action,
                onset=onset,
                offset=offset,
                result=result,
                production_eligible=False,
            )
        except (BoundaryRefinementProjectionError, TypeError, ValueError) as exc:
            raise _BoundaryRefinementExecutionError(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "BOUNDARY_REFINEMENT_REDUCTION_REJECTED",
                    exc,
                ),
            ) from exc

    async def _execute_boundary_refinement_role(
        self,
        *,
        action: ProvisionalPhysicalAction,
        role: BoundaryRefinementRole,
        provisional_fusion_semantic_sha256: str,
        parent_window: CanonicalRootWindow,
        qa_completion_semantic_sha256: str,
        candidate_reduction_semantic_sha256: str,
        action_evidence_result_semantic_sha256s: tuple[str, ...],
        context: AdmittedRecordingContextV2,
        sampling_plan: SamplingPlan,
        frame_index: CanonicalSixCameraFrameIndex,
        artifact_resolver: FrameArtifactResolver,
        created_at: str,
        rendered_item_factory: RenderedItemFactory | None,
    ) -> CanonicalBoundaryRefinementPassExecution:
        policy = self._boundary_refinement_projector.policy
        try:
            window = CanonicalBoundaryRefinementWindow.from_context(
                context=context,
                action=action,
                provisional_fusion_semantic_sha256=provisional_fusion_semantic_sha256,
                parent_window=parent_window,
                role=role,
                padding_before_ns=policy.padding_before_ns,
                padding_after_ns=policy.padding_after_ns,
                window_policy_version=(
                    f"{self._execution_policy.window_policy_version}-boundary-refinement-v1"
                ),
                created_at=created_at,
            )
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            raise _BoundaryRefinementExecutionError(
                CanonicalOfflineRunStatus.CONFIGURATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.WINDOW,
                    f"BOUNDARY_REFINEMENT_{role.value}_WINDOW_INVALID",
                    exc,
                ),
            ) from exc

        try:
            lineage = canonical_boundary_refinement_lineage(
                context=context,
                window=window,
                sampling_plan=sampling_plan,
            )
            planned_parts = self._package_builder.plan_parts(window, sampling_plan)
            if not planned_parts:
                raise CanonicalOfflineConfigurationError(
                    f"boundary {role.value} window produced no package parts"
                )
            materialized = tuple(
                self._materializer.materialize_admitted(
                    part=part,
                    sampling_plan=sampling_plan,
                    purpose=SamplingPurpose.BOUNDARY_REFINEMENT,
                    admitted_context=context,
                    frame_index=frame_index,
                    lineage=lineage,
                    window_id=window.window_id,
                    artifact_resolver=artifact_resolver,
                    created_at=created_at,
                )
                for part in planned_parts
            )
            _validate_materialized_chain(
                context=context,
                window=window,
                lineage=lineage,
                planned_parts=planned_parts,
                materialized=materialized,
            )
            package_set = self._package_builder.build_package_set(
                window,
                sampling_plan,
                context.alignment_manifest.alignment_id,
                lineage=lineage,
                materialized_members=tuple(item.package_ref for item in materialized),
                created_at=created_at,
            )
            _validate_package_set_chain(
                context=context,
                window=window,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                reduction_policy_version=self._package_builder.reduction_policy_version,
            )
        except PackageMaterializationError as exc:
            raise _BoundaryRefinementExecutionError(
                CanonicalOfflineRunStatus.MATERIALIZATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.MATERIALIZATION,
                    f"BOUNDARY_REFINEMENT_{role.value}_{exc.code.value}",
                    exc,
                ),
            ) from exc
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            raise _BoundaryRefinementExecutionError(
                CanonicalOfflineRunStatus.MATERIALIZATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.MATERIALIZATION,
                    f"BOUNDARY_REFINEMENT_{role.value}_PACKAGE_CHAIN_INVALID",
                    exc,
                ),
            ) from exc

        dependency = {
            "qa_completion_semantic_sha256": qa_completion_semantic_sha256,
            "candidate_reduction_semantic_sha256": candidate_reduction_semantic_sha256,
            "action_evidence_result_semantic_sha256s": list(
                action_evidence_result_semantic_sha256s
            ),
            "provisional_fusion_semantic_sha256": provisional_fusion_semantic_sha256,
            "provisional_physical_action_logical_key": action.logical_key,
            "provisional_physical_action_semantic_sha256": action.semantic_sha256,
            "boundary_refinement_role": role.value,
            "boundary_anchor_ns": (
                action.coarse_interval.start_ns
                if role is BoundaryRefinementRole.ONSET
                else action.coarse_interval.end_ns
            ),
            "boundary_refinement_window_semantic_sha256": window.semantic_sha256,
            "boundary_refinement_policy_version": policy.version,
            "boundary_refinement_policy_semantic_sha256": policy.semantic_sha256,
        }
        try:
            input_plan, reference_catalog = await self._prepare_inference_stage(
                task=VisionTask.BOUNDARY_REFINEMENT,
                inference_policy=self._boundary_refinement_policy,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                created_at=created_at,
                rendered_item_factory=rendered_item_factory,
                dependency_config=dependency,
            )
        except asyncio.CancelledError:
            raise
        except (
            InputPreparationError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            raise _BoundaryRefinementExecutionError(
                CanonicalOfflineRunStatus.CONFIGURATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.PREPARATION,
                    f"BOUNDARY_REFINEMENT_{role.value}_INPUT_PLAN_INVALID",
                    exc,
                ),
            ) from exc

        try:
            part_results, barrier_reduction, enriched_outputs = await self._execute_qa_call_plan(
                task=VisionTask.BOUNDARY_REFINEMENT,
                inference_policy=self._boundary_refinement_policy,
                context=context,
                window=window,
                sampling_plan=sampling_plan,
                package_set=package_set,
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                created_at=created_at,
                code_prefix=f"BOUNDARY_REFINEMENT_{role.value}",
                dependency_config=dependency,
            )
            role_result = self._boundary_refinement_projector.project_role(
                action=action,
                role=role,
                input_plan=input_plan,
                package_set=package_set,
                enriched_outputs=enriched_outputs,
                alignment_manifest=context.alignment_manifest,
            )
            return CanonicalBoundaryRefinementPassExecution(
                window=window,
                alignment_manifest=context.alignment_manifest,
                package_set=package_set,
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                part_results=part_results,
                barrier_reduction=barrier_reduction,
                role_result=role_result,
                production_eligible=False,
            )
        except asyncio.CancelledError:
            raise
        except _QAStageError as exc:
            raise _BoundaryRefinementExecutionError(exc.status, exc.error) from exc
        except (BoundaryRefinementProjectionError, TypeError, ValueError) as exc:
            raise _BoundaryRefinementExecutionError(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    f"BOUNDARY_REFINEMENT_{role.value}_PROJECTION_REJECTED",
                    exc,
                ),
            ) from exc

    async def _execute_action_evidence_candidate(
        self,
        *,
        candidate: CanonicalCandidateEvent,
        parent_window: CanonicalRootWindow,
        qa_completion_semantic_sha256: str,
        candidate_reduction_semantic_sha256: str,
        context: AdmittedRecordingContextV2,
        sampling_plan: SamplingPlan,
        frame_index: CanonicalSixCameraFrameIndex,
        artifact_resolver: FrameArtifactResolver,
        created_at: str,
        rendered_item_factory: RenderedItemFactory | None,
    ) -> CanonicalActionEvidenceExecution:
        try:
            window = CanonicalCandidateDenseWindow.from_context(
                context=context,
                candidate=candidate,
                parent_window=parent_window,
                window_policy_version=(
                    f"{self._execution_policy.window_policy_version}-candidate-dense-v1"
                ),
                created_at=created_at,
            )
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            raise _ActionEvidenceExecutionError(
                CanonicalOfflineRunStatus.CONFIGURATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.WINDOW,
                    "ACTION_EVIDENCE_WINDOW_INVALID",
                    exc,
                ),
            ) from exc

        try:
            lineage = canonical_candidate_dense_lineage(
                context=context,
                window=window,
                sampling_plan=sampling_plan,
            )
            planned_parts = self._package_builder.plan_parts(window, sampling_plan)
            if not planned_parts:
                raise CanonicalOfflineConfigurationError(
                    "candidate ACTION_DENSE window produced no package parts"
                )
            materialized = tuple(
                self._materializer.materialize_admitted(
                    part=part,
                    sampling_plan=sampling_plan,
                    purpose=SamplingPurpose.ACTION_DENSE,
                    admitted_context=context,
                    frame_index=frame_index,
                    lineage=lineage,
                    window_id=window.window_id,
                    artifact_resolver=artifact_resolver,
                    created_at=created_at,
                )
                for part in planned_parts
            )
            _validate_materialized_chain(
                context=context,
                window=window,
                lineage=lineage,
                planned_parts=planned_parts,
                materialized=materialized,
            )
            package_set = self._package_builder.build_package_set(
                window,
                sampling_plan,
                context.alignment_manifest.alignment_id,
                lineage=lineage,
                materialized_members=tuple(item.package_ref for item in materialized),
                created_at=created_at,
            )
            _validate_package_set_chain(
                context=context,
                window=window,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                reduction_policy_version=self._package_builder.reduction_policy_version,
            )
        except PackageMaterializationError as exc:
            raise _ActionEvidenceExecutionError(
                CanonicalOfflineRunStatus.MATERIALIZATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.MATERIALIZATION,
                    f"ACTION_EVIDENCE_{exc.code.value}",
                    exc,
                ),
            ) from exc
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            raise _ActionEvidenceExecutionError(
                CanonicalOfflineRunStatus.MATERIALIZATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.MATERIALIZATION,
                    "ACTION_EVIDENCE_PACKAGE_CHAIN_INVALID",
                    exc,
                ),
            ) from exc

        dependency = {
            "qa_completion_semantic_sha256": qa_completion_semantic_sha256,
            "candidate_reduction_semantic_sha256": candidate_reduction_semantic_sha256,
            "candidate_logical_key": candidate.candidate_logical_key,
            "action_dense_window_semantic_sha256": window.semantic_sha256,
            "action_evidence_projection_version": (self._action_evidence_projector.policy_version),
        }
        try:
            input_plan, reference_catalog = await self._prepare_inference_stage(
                task=VisionTask.ACTION_EVIDENCE,
                inference_policy=self._action_evidence_policy,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                created_at=created_at,
                rendered_item_factory=rendered_item_factory,
                dependency_config=dependency,
            )
        except asyncio.CancelledError:
            raise
        except (
            InputPreparationError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            raise _ActionEvidenceExecutionError(
                CanonicalOfflineRunStatus.CONFIGURATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.PREPARATION,
                    "ACTION_EVIDENCE_INPUT_PLAN_INVALID",
                    exc,
                ),
            ) from exc

        try:
            part_results, barrier_reduction, enriched_outputs = await self._execute_qa_call_plan(
                task=VisionTask.ACTION_EVIDENCE,
                inference_policy=self._action_evidence_policy,
                context=context,
                window=window,
                sampling_plan=sampling_plan,
                package_set=package_set,
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                created_at=created_at,
                code_prefix="ACTION_EVIDENCE",
                dependency_config=dependency,
            )
            evidence_result = self._action_evidence_projector.project(
                input_plan=input_plan,
                package_set=package_set,
                candidate=candidate,
                enriched_outputs=enriched_outputs,
            )
            return CanonicalActionEvidenceExecution(
                candidate=candidate,
                window=window,
                package_set=package_set,
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                part_results=part_results,
                barrier_reduction=barrier_reduction,
                evidence_result=evidence_result,
                production_eligible=False,
            )
        except asyncio.CancelledError:
            raise
        except _QAStageError as exc:
            raise _ActionEvidenceExecutionError(exc.status, exc.error) from exc
        except (ActionEvidenceProjectionError, TypeError, ValueError) as exc:
            raise _ActionEvidenceExecutionError(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "ACTION_EVIDENCE_PROJECTION_REJECTED",
                    exc,
                ),
            ) from exc

    async def _execute_dense_qa_unit(
        self,
        *,
        unit: DenseQAWorkUnit,
        preliminary_completion_semantic_sha256: str,
        dense_manifest_semantic_digest: str,
        context: AdmittedRecordingContextV2,
        sampling_plan: SamplingPlan,
        frame_index: CanonicalSixCameraFrameIndex,
        artifact_resolver: FrameArtifactResolver,
        created_at: str,
        rendered_item_factory: RenderedItemFactory | None,
    ) -> CanonicalDenseQAExecution:
        try:
            window = CanonicalRootWindow.from_context(
                context=context,
                requested_interval=unit.effective_interval,
                purpose=SamplingPurpose.QA_DENSE,
                window_policy_version=self._execution_policy.window_policy_version,
                created_at=created_at,
            )
            if window.interval != unit.effective_interval:
                raise CanonicalOfflineConfigurationError(
                    "dense work unit differs from its admitted execution window"
                )
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            raise _DenseQAExecutionError(
                _canonical_error(
                    CanonicalOfflineStage.WINDOW,
                    "QA_DENSE_WINDOW_INVALID",
                    exc,
                )
            ) from exc

        try:
            lineage = canonical_lineage(
                context=context,
                window=window,
                sampling_plan=sampling_plan,
            )
            planned_parts = self._package_builder.plan_parts(window, sampling_plan)
            if not planned_parts:
                raise CanonicalOfflineConfigurationError(
                    "dense work unit produced no package parts"
                )
            materialized = tuple(
                self._materializer.materialize_admitted(
                    part=part,
                    sampling_plan=sampling_plan,
                    purpose=SamplingPurpose.QA_DENSE,
                    admitted_context=context,
                    frame_index=frame_index,
                    lineage=lineage,
                    window_id=window.window_id,
                    artifact_resolver=artifact_resolver,
                    created_at=created_at,
                )
                for part in planned_parts
            )
            _validate_materialized_chain(
                context=context,
                window=window,
                lineage=lineage,
                planned_parts=planned_parts,
                materialized=materialized,
            )
            package_set = self._package_builder.build_package_set(
                window,
                sampling_plan,
                context.alignment_manifest.alignment_id,
                lineage=lineage,
                materialized_members=tuple(item.package_ref for item in materialized),
                created_at=created_at,
            )
            _validate_package_set_chain(
                context=context,
                window=window,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                reduction_policy_version=self._package_builder.reduction_policy_version,
            )
        except PackageMaterializationError as exc:
            raise _DenseQAExecutionError(
                _canonical_error(
                    CanonicalOfflineStage.MATERIALIZATION,
                    f"QA_DENSE_{exc.code.value}",
                    exc,
                )
            ) from exc
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            raise _DenseQAExecutionError(
                _canonical_error(
                    CanonicalOfflineStage.MATERIALIZATION,
                    "QA_DENSE_PACKAGE_CHAIN_INVALID",
                    exc,
                )
            ) from exc

        dependency = {
            "qa_completion_semantic_sha256": preliminary_completion_semantic_sha256,
            "dense_work_manifest_semantic_sha256": dense_manifest_semantic_digest,
            "dense_work_unit_semantic_sha256": unit.semantic_digest,
        }
        try:
            input_plan, reference_catalog = await self._prepare_inference_stage(
                task=VisionTask.QA_DENSE,
                inference_policy=self._dense_qa_policy,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                created_at=created_at,
                rendered_item_factory=rendered_item_factory,
                dependency_config=dependency,
            )
        except asyncio.CancelledError:
            raise
        except (
            InputPreparationError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            raise _DenseQAExecutionError(
                _canonical_error(
                    CanonicalOfflineStage.PREPARATION,
                    "QA_DENSE_INPUT_PLAN_INVALID",
                    exc,
                )
            ) from exc

        try:
            part_results, barrier_reduction, enriched_outputs = await self._execute_qa_call_plan(
                task=VisionTask.QA_DENSE,
                inference_policy=self._dense_qa_policy,
                context=context,
                window=window,
                sampling_plan=sampling_plan,
                package_set=package_set,
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                created_at=created_at,
                code_prefix="QA_DENSE",
                dependency_config=dependency,
            )
            unit_evidence = self._build_dense_qa_unit_evidence(
                unit=unit,
                context=context,
                package_set=package_set,
                input_plan=input_plan,
                enriched_outputs=enriched_outputs,
            )
            return CanonicalDenseQAExecution(
                work_unit=unit,
                window=window,
                package_set=package_set,
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                part_results=part_results,
                barrier_reduction=barrier_reduction,
                unit_evidence=unit_evidence,
                production_eligible=False,
            )
        except asyncio.CancelledError:
            raise
        except _QAStageError as exc:
            raise _DenseQAExecutionError(exc.error) from exc
        except (DenseQAProjectionError, TypeError, ValueError) as exc:
            raise _DenseQAExecutionError(
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "QA_DENSE_EVIDENCE_REJECTED",
                    exc,
                )
            ) from exc

    @staticmethod
    def _build_dense_qa_unit_evidence(
        *,
        unit: DenseQAWorkUnit,
        context: AdmittedRecordingContextV2,
        package_set: TemporalPackageSet,
        input_plan: InferenceInputPlan,
        enriched_outputs: Sequence[OrchestratorEnrichedOutput],
    ) -> DenseQAUnitEvidence:
        """Normalize exact QA_DENSE lineage without assigning new authority."""

        if (
            input_plan.subject.task is not VisionTask.QA_DENSE
            or input_plan.request_catalog.task is not VisionTask.QA_DENSE
            or package_set.mcap_id != context.ready_manifest.mcap_id
            or package_set.camera_mapping_run_id != context.ready_manifest.camera_mapping_run_id
            or package_set.alignment_id != context.alignment_manifest.alignment_id
        ):
            raise DenseQAProjectionError("dense execution lineage is not QA_DENSE-bound")

        packages = tuple(
            DenseQAPackageRef(
                package_id=member.package_id,
                ordinal=member.ordinal,
                interval=NanosecondInterval(
                    start_ns=member.start_ns,
                    end_ns=member.end_ns,
                ),
                semantic_content_sha256=member.package_semantic_content_sha256,
                manifest_sha256=member.package_manifest_sha256,
            )
            for member in package_set.members
        )
        input_ref = DenseQAInputPlanRef(
            input_plan_id=input_plan.input_plan_id,
            semantic_sha256=input_plan.semantic_sha256,
            task="QA_DENSE",
            package_ids=tuple(item.package_id for item in packages),
        )
        output_refs: dict[str, DenseQAOutputRef] = {}
        observations: dict[tuple[int, CameraId], CameraDenseResult] = {}
        for output in enriched_outputs:
            if (
                output.task is not VisionTask.QA_DENSE
                or output.abstained
                or output.input_plan_id != input_plan.input_plan_id
                or output.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or output.request_catalog_id != input_plan.request_catalog.request_catalog_id
                or output.request_catalog_sha256 != input_plan.request_catalog.semantic_sha256
                or output.authority.mcap_id != package_set.mcap_id
                or output.authority.camera_mapping_run_id != package_set.camera_mapping_run_id
                or output.authority.alignment_id != package_set.alignment_id
            ):
                raise DenseQAProjectionError(
                    "enriched QA_DENSE output is not bound to its exact execution"
                )
            if output.artifact_id in output_refs:
                raise DenseQAProjectionError("duplicate QA_DENSE enriched output")
            source = DenseQAOutputRef(
                artifact_id=output.artifact_id,
                semantic_sha256=output.semantic_sha256,
                enrichment_logical_key=output.enrichment_logical_key,
                inference_id=output.selected_attempt.inference_id,
                input_plan_id=input_plan.input_plan_id,
                input_plan_semantic_sha256=input_plan.semantic_sha256,
                task="QA_DENSE",
            )
            output_refs[output.artifact_id] = source
            for claim in output.claims:
                if (
                    claim.kind is not ProviderClaimKind.QA_OBSERVATION
                    or claim.package_id is None
                    or claim.package_ordinal is None
                    or claim.camera_id is None
                    or claim.package_ordinal >= len(packages)
                ):
                    raise DenseQAProjectionError(
                        "QA_DENSE accepts only package-camera QA observations"
                    )
                coordinate = (claim.package_ordinal, claim.camera_id)
                if coordinate in observations:
                    raise DenseQAProjectionError(
                        "QA_DENSE observations must cover each coordinate once"
                    )
                try:
                    local_status = CameraQAStatus(claim.observation.value)
                except ValueError as exc:
                    raise DenseQAProjectionError(
                        "QA_DENSE observation has no local QA status"
                    ) from exc
                observations[coordinate] = CameraDenseResult(
                    package_id=claim.package_id,
                    package_ordinal=claim.package_ordinal,
                    camera_id=claim.camera_id,
                    local_status=local_status,
                    source_output=source,
                    claim=claim,
                    production_eligible=False,
                )

        expected_coordinates = tuple(
            (package.ordinal, camera_id) for package in packages for camera_id in CAMERA_IDS
        )
        if set(observations) != set(expected_coordinates):
            raise DenseQAProjectionError(
                "QA_DENSE enriched outputs lack exact package-by-six coverage"
            )
        return DenseQAUnitEvidence(
            unit_id=unit.unit_id,
            unit_semantic_digest=unit.semantic_digest,
            mcap_id=package_set.mcap_id,
            camera_mapping_run_id=package_set.camera_mapping_run_id,
            camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
            alignment_id=package_set.alignment_id,
            alignment_semantic_sha256=context.alignment_semantic_sha256,
            package_set_id=package_set.package_set_id,
            split_plan_digest=package_set.split_plan_digest,
            member_manifest_sha256=package_set.member_manifest_sha256,
            packages=packages,
            input_plan=input_ref,
            source_outputs=tuple(output_refs[artifact_id] for artifact_id in sorted(output_refs)),
            package_camera_results=tuple(
                observations[coordinate] for coordinate in expected_coordinates
            ),
            production_eligible=False,
        )

    async def _execute_declared_call_plan(
        self,
        *,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        package_inputs: tuple[PackageInput, ...],
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        created_at: str,
        state: dict[str, object],
        part_results: list[CanonicalOfflinePartResult],
        finish: Callable[
            [CanonicalOfflineRunStatus, CanonicalOfflineError | None],
            CanonicalOfflineRunResult,
        ],
        attach_nodes: Callable[
            [Sequence[tuple[LogicalNode, RunNodeRole]]],
            None,
        ],
        membership_failure: Callable[[object], CanonicalOfflineRunResult],
        final_fusion_context: CanonicalFinalFusionContext,
        dependency_config: Mapping[str, object] | None = None,
    ) -> CanonicalOfflineRunResult:
        try:
            results = await self._execute_call_plan_parts(
                task=VisionTask.FUSION_ADJUDICATION,
                inference_policy=self._inference_policy,
                context=context,
                window=window,
                sampling_plan=sampling_plan,
                package_set=package_set,
                package_inputs=package_inputs,
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                dependency_config=dependency_config,
            )
        except asyncio.CancelledError:
            raise
        except (
            InferenceOrchestrationError,
            InferenceCallBarrierError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "CALL_PART_EXECUTION_FAILED",
                    exc,
                ),
            )
        for result in results:
            part_results.append(result)
            try:
                nodes: list[tuple[LogicalNode, RunNodeRole]] = []
                if result.selection is not None:
                    nodes.append(
                        (canonical_selection_logical_node(result.selection), "ATTEMPT_SELECTION")
                    )
                if result.parsed_claims is not None:
                    nodes.append(
                        (canonical_parsed_claim_logical_node(result.parsed_claims), "PARSED_CLAIM")
                    )
                if result.selected_output is not None:
                    nodes.append(
                        (
                            canonical_selected_output_logical_node(result.selected_output),
                            "SELECTED_OUTPUT",
                        )
                    )
                if result.enriched_output is not None:
                    nodes.append(
                        (
                            canonical_enrichment_logical_node(result.enriched_output),
                            "ENRICHED_OUTPUT",
                        )
                    )
                attach_nodes(nodes)
            except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
                return membership_failure(exc)

        try:
            aggregate = self._call_barrier.get_aggregate_status(input_plan)
        except InferenceCallBarrierError as exc:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "BARRIER_STATUS_FAILED",
                    exc,
                ),
            )
        if not aggregate.is_complete:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "BARRIER_NOT_TERMINAL",
                    "declared call barrier remained open after every local part execution",
                ),
            )
        if aggregate.overall_status is StageStatus.INCOMPLETE:
            failed_ordinals = tuple(
                item.part_ordinal
                for item in part_results
                if item.status is CanonicalOfflinePartStatus.TERMINAL_FAILED
            )
            return finish(
                CanonicalOfflineRunStatus.INCOMPLETE,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "REQUIRED_CALL_PARTS_INCOMPLETE",
                    f"required terminal failures at part ordinals {failed_ordinals}",
                ),
            )
        if aggregate.overall_status is not StageStatus.SUCCEEDED:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "BARRIER_STATUS_INVALID",
                    aggregate.overall_status.value,
                ),
            )

        invalid = next(
            (
                item
                for item in part_results
                if item.status is CanonicalOfflinePartStatus.POST_SELECTION_INVALID
            ),
            None,
        )
        if invalid is not None:
            assert invalid.error is not None
            return finish(CanonicalOfflineRunStatus.INVALID_OUTPUT, invalid.error)

        enriched_outputs = tuple(
            item.enriched_output for item in part_results if item.enriched_output is not None
        )
        if len(enriched_outputs) != len(input_plan.call_plan.parts):
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.ENRICHMENT,
                    "PART_ENRICHMENT_SET_INCOMPLETE",
                    "successful barrier lacks exact enriched output coverage",
                ),
            )
        abstention_flags = tuple(item.abstained for item in enriched_outputs)
        if any(abstention_flags) and not all(abstention_flags):
            return finish(
                CanonicalOfflineRunStatus.INCOMPLETE,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "PARTIAL_REQUIRED_ABSTENTION",
                    "required call parts mixed provider claims and abstentions",
                ),
            )

        try:
            barrier_reduction = self._call_barrier.reduce(
                input_plan,
                reduced_at=created_at,
            )
            parsed_payloads = tuple(
                item.parsed_claims.payload
                for item in part_results
                if item.parsed_claims is not None
            )
            expected_payload = _reduce_provider_claim_payloads(parsed_payloads)
            reduced_payload = ProviderClaimPayload.model_validate(
                barrier_reduction.normalized_output,
                strict=False,
            )
            if reduced_payload != expected_payload:
                raise CanonicalOfflineConfigurationError(
                    "barrier reduction differs from exact parsed part claims"
                )
            fusion_reduction = _build_canonical_fusion_reduction(
                input_plan=input_plan,
                barrier_reduction=barrier_reduction,
                part_results=tuple(part_results),
                created_at=created_at,
            )
            state["barrier_reduction"] = barrier_reduction
            state["fusion_reduction"] = fusion_reduction
        except (
            InferenceCallBarrierError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "FUSION_REDUCTION_REJECTED",
                    exc,
                ),
            )
        try:
            attach_nodes(
                (
                    (
                        canonical_call_reduction_logical_node(barrier_reduction),
                        "CALL_REDUCTION",
                    ),
                    (
                        canonical_fusion_reduction_logical_node(fusion_reduction),
                        "FUSION_REDUCTION",
                    ),
                )
            )
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        try:
            validate_final_fusion_reduction(
                context=final_fusion_context,
                fusion_reduction=fusion_reduction,
            )
        except (TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "FINAL_FUSION_CLOSURE_REJECTED",
                    exc,
                ),
            )

        try:
            decision, hypotheses = self._projector.project(
                context=context,
                fusion_reduction=fusion_reduction,
                enriched_outputs=enriched_outputs,
                interval=window.interval,
            )
            state["output_decision"] = decision
            state["hypotheses"] = hypotheses
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.OUTPUT_ADMISSION,
                    "OUTPUT_ADMISSION_REJECTED",
                    exc,
                ),
            )
        try:
            attach_nodes(
                (
                    (canonical_output_decision_logical_node(decision), "OUTPUT_DECISION"),
                    *(
                        (canonical_event_hypothesis_logical_node(item), "EVENT_HYPOTHESIS")
                        for item in hypotheses
                    ),
                )
            )
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        if decision.decision == "ABSTAINED":
            return finish(CanonicalOfflineRunStatus.ABSTAINED, None)
        if decision.decision == "NO_EVENTS":
            return finish(CanonicalOfflineRunStatus.NO_EVENTS, None)
        return finish(CanonicalOfflineRunStatus.SUCCEEDED, None)

    def _validate_configuration(self) -> None:
        execution = self._execution_policy
        for policy, expected_task in (
            (self._coarse_qa_policy, VisionTask.QA_COARSE),
            (self._dense_qa_policy, VisionTask.QA_DENSE),
            (self._event_proposal_policy, VisionTask.EVENT_PROPOSAL),
            (self._action_evidence_policy, VisionTask.ACTION_EVIDENCE),
            (self._boundary_refinement_policy, VisionTask.BOUNDARY_REFINEMENT),
            (self._inference_policy, VisionTask.FUSION_ADJUDICATION),
        ):
            if policy.task is not expected_task:
                raise CanonicalOfflineConfigurationError(
                    f"canonical pipeline requires {expected_task.value} policy"
                )
            if policy.provider != self._adapter.provider:
                raise CanonicalOfflineConfigurationError(
                    "inference policy provider does not match offline adapter"
                )
            if policy.output_schema.schema_id != PROVIDER_CLAIM_SCHEMA_ID:
                raise CanonicalOfflineConfigurationError(
                    "inference policy must pin the provider-claim schema"
                )
            enriched_schema = policy.enriched_output_schema
            if enriched_schema is None or enriched_schema.schema_id != ENRICHED_OUTPUT_SCHEMA_ID:
                raise CanonicalOfflineConfigurationError(
                    "inference policy must pin the enriched-output schema"
                )
            if policy.output_schema.sha256 == enriched_schema.sha256:
                raise CanonicalOfflineConfigurationError(
                    "provider and enriched schemas must remain distinct"
                )
            self._schema_registry.resolve_exact(_schema_ref(policy.output_schema))
            self._schema_registry.resolve_exact(_schema_ref(enriched_schema))
            _require_canonical_uuid(policy.prompt_artifact_id, "prompt_artifact_id")
        if self._parser.schema_registry is not self._schema_registry:
            raise CanonicalOfflineConfigurationError(
                "provider claim parser and pipeline must share one schema registry"
            )
        if self._parser.parser_version != execution.parser_version:
            raise CanonicalOfflineConfigurationError(
                "parser version does not match execution policy"
            )
        rendering_policy = self._input_preparer.policy
        if (
            rendering_policy.reduction_policy != execution.reduction_policy
            or rendering_policy.reduction_policy_version != execution.reduction_policy_version
            or self._package_builder.reduction_policy_version != execution.reduction_policy_version
        ):
            raise CanonicalOfflineConfigurationError(
                "package, rendering, and execution reduction policies differ"
            )

    @staticmethod
    def _required_enriched_schema(inference_policy: InferencePolicy) -> JsonSchemaRef:
        schema = inference_policy.enriched_output_schema
        if schema is None:
            raise CanonicalOfflineConfigurationError("enriched output schema is missing")
        return schema


__all__ = ["CanonicalOfflinePipeline"]
