"""Canonical offline pipeline composition and state progression."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import datetime

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.logical_nodes import (
    canonical_call_barrier_logical_node,
    canonical_call_part_logical_node,
    canonical_call_reduction_logical_node,
    canonical_enrichment_logical_node,
    canonical_event_hypothesis_logical_node,
    canonical_fusion_reduction_logical_node,
    canonical_input_plan_logical_node,
    canonical_output_decision_logical_node,
    canonical_package_set_logical_node,
    canonical_parsed_claim_logical_node,
    canonical_root_window_logical_node,
    canonical_selected_output_logical_node,
    canonical_selection_logical_node,
)
from robata.application.canonical.models import (
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
    canonical_lineage,
)
from robata.application.canonical.output_admission import FusionEventHypothesisProjector
from robata.application.canonical.projections import _stable_uuid
from robata.application.canonical.reduction import (
    _build_canonical_fusion_reduction,
    _OrderedProviderClaimReducer,
    _reduce_provider_claim_payloads,
)
from robata.application.canonical.result_validation import CanonicalOfflineRunResult
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
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import LogicalNode, RunNodeRole
from robata.contracts.pipeline import SamplingPurpose
from robata.contracts.sampling_plan import SamplingPlan
from robata.contracts.schema_registry import SchemaRegistry
from robata.contracts.temporal import TemporalPackageSet
from robata.inference.adapter import JsonSchemaRef, PackageInput
from robata.inference.call_barrier import (
    InferenceCallBarrierCoordinator,
    InferenceCallBarrierError,
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
    ModelInference,
    Retryability,
    VisionTask,
)
from robata.inference.offline_fixture import (
    OfflineFixtureVisionAdapter,
    RawProviderBytesStoreError,
    StrictProviderClaimParseError,
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
from robata.queue.barrier import BarrierCoordinator, InMemoryBarrierStorage
from robata.queue.stage import StageStatus
from robata.sampling.materializer import (
    CanonicalSixCameraFrameIndex,
    FrameArtifactResolver,
    OfflineTemporalPackageMaterializer,
    PackageMaterializationError,
)
from robata.sampling.package_set import PackageSetBuilder, sampling_plan_digest


class _CanonicalRunMembershipPublicationError(RuntimeError):
    """A typed node or its immutable run attachment could not be published."""


class CanonicalOfflinePipeline:
    """Run the canonical post-admission path without network-capable adapters."""

    def __init__(
        self,
        *,
        package_builder: PackageSetBuilder,
        materializer: OfflineTemporalPackageMaterializer,
        input_preparer: InputPlanPreparer,
        adapter: OfflineFixtureVisionAdapter,
        inference_policy: InferencePolicy,
        schema_registry: SchemaRegistry,
        logical_node_registry: LogicalNodeRegistry,
        execution_policy: CanonicalOfflineExecutionPolicy,
        inference_ledger: InferenceLedger | None = None,
        evidence_store: InferenceEvidenceStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(package_builder, PackageSetBuilder):
            raise TypeError("package_builder must be a PackageSetBuilder")
        if not isinstance(materializer, OfflineTemporalPackageMaterializer):
            raise TypeError("materializer must be an OfflineTemporalPackageMaterializer")
        if not isinstance(input_preparer, InputPlanPreparer):
            raise TypeError("input_preparer must be an InputPlanPreparer")
        if not isinstance(adapter, OfflineFixtureVisionAdapter):
            raise TypeError("canonical pipeline accepts only OfflineFixtureVisionAdapter")
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
        if evidence_store is not None:
            if not isinstance(evidence_store, InferenceEvidenceStore):
                raise TypeError("evidence_store must implement InferenceEvidenceStore")
            ledger_identity: object = inference_ledger
            raw_store_identity: object = adapter.raw_store
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
        self._package_builder = package_builder
        self._materializer = materializer
        self._input_preparer = input_preparer
        self._adapter = adapter
        self._inference_policy = inference_policy
        self._schema_registry = schema_registry
        self._logical_node_registry = logical_node_registry
        self._execution_policy = execution_policy
        self._clock = clock or _utc_now
        self._validate_configuration()

        self._ledger = (
            inference_ledger if inference_ledger is not None else InMemoryInferenceLedger()
        )
        self._evidence_store = (
            evidence_store if evidence_store is not None else InMemoryInferenceEvidenceStore()
        )
        schema_registry.resolve_exact(_schema_ref(inference_policy.output_schema))
        self._orchestrator = InferenceOrchestrator(
            adapters={adapter.provider: adapter},
            task_policies={inference_policy.task: inference_policy},
            schema_artifacts={
                item.ref.artifact_id: item.document_bytes for item in schema_registry.entries
            },
            ledger=self._ledger,
            clock=self._clock,
        )
        self._call_barrier_storage = InMemoryInferenceCallBarrierStorage()
        reducer_key = (
            execution_policy.reduction_policy,
            execution_policy.reduction_policy_version,
        )
        self._call_barrier = InferenceCallBarrierCoordinator(
            barriers=BarrierCoordinator(InMemoryBarrierStorage()),
            storage=self._call_barrier_storage,
            reducers={reducer_key: _OrderedProviderClaimReducer()},
        )
        self._enricher = ProviderClaimEnricher(schema_registry)
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
    def call_barrier_storage(self) -> InMemoryInferenceCallBarrierStorage:
        return self._call_barrier_storage

    @property
    def adapter(self) -> OfflineFixtureVisionAdapter:
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
        infer_calls_before = self._adapter.infer_calls
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
            if self._adapter.network_call_count != 0:
                raise CanonicalOfflineConfigurationError(
                    "offline fixture adapter reported a network call"
                )
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
                    "adapter_infer_calls": self._adapter.infer_calls - infer_calls_before,
                    "network_call_count": 0,
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
                purpose=SamplingPurpose.ACTION_DENSE,
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
            capabilities = await self._adapter.capabilities(
                self._inference_policy.model_name,
                self._inference_policy.model_version,
            )
            capabilities = _validated_capabilities(
                capabilities,
                inference_policy=self._inference_policy,
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
                adapter_version=self._inference_policy.adapter_version,
                planner_version=self._input_preparer.planner_version,
                capability_snapshot_id=capabilities.snapshot_id,
                capability_snapshot_sha256=capabilities.snapshot_digest,
            )
            request_catalog_id = _stable_uuid(
                REQUEST_CATALOG_UUID_NAMESPACE,
                lineage,
                self._execution_policy.semantic_sha256,
                capabilities.snapshot_digest,
            )
            prepared = self._input_preparer.prepare_rendering(
                packages=materialized,
                task=VisionTask.FUSION_ADJUDICATION,
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
                inference_policy=self._inference_policy,
                request_catalog_sha256=prepared.request_catalog.semantic_sha256,
                token_policy_version=self._execution_policy.token_policy_version,
                entries=prompt_entries,
            )
            enriched_schema = self._required_enriched_schema()
            prompt_output = PromptOutputContract(
                prompt_version=self._inference_policy.prompt_version,
                prompt_sha256=self._inference_policy.prompt_sha256,
                rendered_message_sha256=exact_bytes_sha256(rendered_prompt),
                provider_response_schema_sha256=self._inference_policy.output_schema.sha256,
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
                inference_policy=self._inference_policy,
                execution_policy=self._execution_policy,
                capabilities=capabilities,
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
        )

    async def _orchestrate_call_part(
        self,
        *,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        package_inputs: tuple[PackageInput, ...],
        input_plan: InferenceInputPlan,
        part: InferenceCallPart,
    ) -> tuple[ModelInference, InferenceAttemptSelection | None, int]:
        terminal: ModelInference | None = None
        for attempt in range(1, self._execution_policy.max_attempts + 1):
            terminal = await self._orchestrator.orchestrate(
                task=VisionTask.FUSION_ADJUDICATION,
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
                    "canonical_execution_policy_sha256": (self._execution_policy.semantic_sha256)
                },
                sampling_config={
                    "sampling_plan_version": sampling_plan.version,
                    "sampling_plan_sha256": sampling_plan_digest(sampling_plan),
                },
                metadata={},
                attempt=attempt,
                retry_count=attempt - 1,
            )
            if terminal.status is InferenceStatus.SUCCEEDED:
                selection = self._ledger.get_selection(
                    terminal.logical_invocation_id,
                    self._inference_policy.selection_policy_version,
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
            stored_raw = self._adapter.raw_store.get(raw_artifact_id)
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
                self._inference_policy.output_schema.sha256,
                self._execution_policy.parser_version,
            )
            parsed_claims = self._evidence_store.get_parsed_claim(parsed_artifact_id)
            if parsed_claims is None:
                parsed_claims = self._adapter.parser.parse_artifact(
                    stored=stored_raw,
                    inference_id=selected_terminal.inference_id,
                    provider=selected_terminal.provider,
                    model_name=selected_terminal.model_name,
                    model_version=selected_terminal.model_version,
                    provider_claim_schema=self._inference_policy.output_schema,
                    task=VisionTask.FUSION_ADJUDICATION,
                    artifact_id=parsed_artifact_id,
                    created_at=artifact_created_at,
                )
                parsed_claims = self._evidence_store.append_parsed_claim(parsed_claims)
            if (
                parsed_claims.artifact_id != parsed_artifact_id
                or parsed_claims.raw_response != expected_raw
                or parsed_claims.provider_claim_schema != self._inference_policy.output_schema
                or parsed_claims.task is not VisionTask.FUSION_ADJUDICATION
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
                selected_output = self._evidence_store.append_selected_output(expected_selected)
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
                prompt_version=self._inference_policy.prompt_version,
                prompt_artifact_id=self._inference_policy.prompt_artifact_id,
                prompt_sha256=self._inference_policy.prompt_sha256,
                work_node_type="INFERENCE_ENRICHMENT",
                work_node_logical_key=f"inference-work:{work_digest}",
            )
            enriched_schema = self._required_enriched_schema()
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
                enriched_output = self._evidence_store.append_enriched_output(enriched_output)
            elif (
                enriched_output.artifact_id != enriched_artifact_id
                or enriched_output.enrichment_logical_key
                != f"orchestrator-enrichment:{logical_digest}"
                or enriched_output.task is not VisionTask.FUSION_ADJUDICATION
                or enriched_output.selected_attempt != selected_output
                or enriched_output.request_catalog_id
                != input_plan.request_catalog.request_catalog_id
                or enriched_output.request_catalog_sha256
                != input_plan.request_catalog.semantic_sha256
                or enriched_output.reference_catalog_id != reference_catalog.reference_catalog_id
                or enriched_output.reference_catalog_sha256 != reference_catalog.semantic_sha256
                or enriched_output.input_plan_id != input_plan.input_plan_id
                or enriched_output.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or enriched_output.provider_claim_schema != self._inference_policy.output_schema
                or enriched_output.enriched_output_schema != enriched_schema
                or enriched_output.enrichment_policy_version
                != self._execution_policy.enrichment_policy_version
                or enriched_output.authority != authority
                or enriched_output.created_at != selected_terminal.completed_at
            ):
                raise InferenceEvidenceStoreError(
                    "persisted enriched output differs from the current semantic lineage"
                )
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
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        package_inputs: tuple[PackageInput, ...],
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        part: InferenceCallPart,
    ) -> CanonicalOfflinePartResult:
        terminal, selection, attempts_used = await self._orchestrate_call_part(
            context=context,
            window=window,
            sampling_plan=sampling_plan,
            package_set=package_set,
            package_inputs=package_inputs,
            input_plan=input_plan,
            part=part,
        )
        if terminal.status is not InferenceStatus.SUCCEEDED:
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
        raw, parsed, selected_output, enriched, lineage_error = self._build_selected_part_lineage(
            context=context,
            input_plan=input_plan,
            reference_catalog=reference_catalog,
            part=part,
            selected_terminal=terminal,
            selection=selection,
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
    ) -> CanonicalOfflineRunResult:
        for part in input_plan.call_plan.parts:
            try:
                result = await self._execute_one_call_part(
                    context=context,
                    window=window,
                    sampling_plan=sampling_plan,
                    package_set=package_set,
                    package_inputs=package_inputs,
                    input_plan=input_plan,
                    reference_catalog=reference_catalog,
                    part=part,
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
        policy = self._inference_policy
        execution = self._execution_policy
        if policy.task is not VisionTask.FUSION_ADJUDICATION:
            raise CanonicalOfflineConfigurationError(
                "canonical pipeline requires FUSION_ADJUDICATION policy"
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
        if self._adapter.parser.schema_registry is not self._schema_registry:
            raise CanonicalOfflineConfigurationError(
                "offline parser and pipeline must share one schema registry"
            )
        if self._adapter.parser.parser_version != execution.parser_version:
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
        _require_canonical_uuid(policy.prompt_artifact_id, "prompt_artifact_id")

    def _required_enriched_schema(self) -> JsonSchemaRef:
        schema = self._inference_policy.enriched_output_schema
        if schema is None:
            raise CanonicalOfflineConfigurationError("enriched output schema is missing")
        return schema


__all__ = ["CanonicalOfflinePipeline"]
