"""Concrete, restartable LOCAL_CONFORMANCE composition for the canonical path."""

from __future__ import annotations

import asyncio
import hashlib
import sys
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.adapters.sqlite_barrier import (
    SQLiteBarrierStorage,
    SQLiteBarrierStorageError,
)
from robata.adapters.sqlite_inference_evidence import (
    MODEL_INFERENCE_SCHEMA_ID,
    SQLiteInferenceEvidenceLedger,
    SQLiteInferenceEvidenceLedgerError,
)
from robata.adapters.sqlite_outbox import SQLiteIdempotentOutboxSink
from robata.adapters.sqlite_primary_completion import SQLitePrimaryCompletionRepository
from robata.adapters.sqlite_stream_delivery import SQLiteStreamDeliveryAuthority
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler, WorkSchedulerError
from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.action_event_revision import (
    CanonicalActionEventRevisionError,
    prepare_initial_action_event_publications,
)
from robata.application.canonical.durable_work import (
    CanonicalActionPublishWorkCoordinator,
    CanonicalDurableWorkError,
)
from robata.application.canonical.local_outbox_delivery import (
    LocalOutboxDeliverySummary,
    failed_local_outbox_delivery,
    reconcile_local_primary_outbox,
)
from robata.application.canonical.local_review_routing import (
    LocalReviewRoutingSummary,
    failed_local_review_routing,
    route_local_review_after_completion,
)
from robata.application.canonical.local_stream_finalization import (
    LOCAL_STREAM_CAUSAL_REDUCTION_POLICY_VERSION,
    LOCAL_STREAM_MOCK_EXECUTOR_POLICY_VERSION,
    LOCAL_STREAM_WORK_RECEIPT_SCHEMA_ID,
    LOCAL_STREAM_WORK_RECEIPT_SCHEMA_VERSION,
    FinalRecordingFacts,
    LocalConformanceStreamFinalizer,
    LocalStreamFinalizationError,
    LocalStreamFinalizationOutcome,
    LocalStreamFinalizationSchemaRefs,
    load_completed_local_stream_recording_result,
)
from robata.application.canonical.local_supplemental_qa import (
    LOCAL_SUPPLEMENTAL_DEDUPE_POLICY_VERSION,
    LOCAL_SUPPLEMENTAL_QA_RUNTIME_VERSION,
    LOCAL_SUPPLEMENTAL_SELECTION_TOLERANCE_NS,
    LOCAL_SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION,
    build_and_publish_local_supplemental_qa_evidence,
    load_and_verify_local_supplemental_qa_evidence,
)
from robata.application.canonical.media_quality import (
    LOCAL_MEDIA_QUALITY_POLICY_VERSION,
    LOCAL_MEDIA_QUALITY_REPORT_FORMAT_VERSION,
    LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID,
    LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION,
    LOCAL_NEIGHBOR_TARGET_POLICY_VERSION,
    LocalMediaQualityReport,
    load_registered_local_media_quality_report_document,
)
from robata.application.canonical.media_quality_binding import (
    LOCAL_MEDIA_QUALITY_BINDING_PROJECTION_VERSION,
    LocalMediaQualityBinding,
    derive_local_media_quality_binding_document,
)
from robata.application.canonical.media_quality_source_binding import (
    MEDIA_QUALITY_SOURCE_BINDING_PROJECTION_VERSION,
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
    CANONICAL_PRIMARY_COMPLETION_COMMAND_PROJECTION_VERSION,
    CommittedPrimaryCompletion,
    PrimaryCompletionError,
    PrimaryCompletionEvidenceReference,
    PrimaryCompletionEvidenceRole,
    prepare_primary_completion_command,
)
from robata.application.canonical.product_qa import (
    product_qa_context_from_media_quality_report,
)
from robata.application.canonical.result_validation import CanonicalOfflineRunResult
from robata.application.canonical.runner import CanonicalOfflinePipeline
from robata.application.canonical.source_fixture import (
    CanonicalSourceBundle,
    CanonicalSourceFixture,
    CanonicalSourceFixtureError,
    load_canonical_source_fixture,
)
from robata.application.canonical.stream_recording_reduction import (
    LOCAL_STREAM_RECORDING_REDUCTION_POLICY_VERSION,
    LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
    LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_ID,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_VERSION,
    LocalStreamRecordingResult,
    LocalStreamRecordingResultV2,
    LocalStreamRecordingResultV3,
    LocalStreamRecordingResultV4,
)
from robata.application.canonical.stream_scheduler import DurableStreamWindowScheduler
from robata.application.canonical_run_membership import CanonicalProcessingRunContext
from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval, StrictModel
from robata.contracts.hashing import (
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)
from robata.contracts.local_stream_causal import (
    LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_ID,
    LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_VERSION,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_ID,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_VERSION,
)
from robata.contracts.sampling_plan import SamplingPlan
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.contracts.stream_common import ArtifactEvidenceRef, StreamPurpose
from robata.contracts.stream_finalization import (
    RECORDING_FINALIZATION_SCHEMA_ID,
    RECORDING_FINALIZATION_SCHEMA_VERSION,
)
from robata.contracts.stream_inference import (
    STREAM_ACCEPTED_CALL_SCHEMA_ID,
    STREAM_ACCEPTED_CALL_SCHEMA_VERSION,
    STREAM_INFERENCE_INTENT_SCHEMA_ID,
    STREAM_INFERENCE_INTENT_SCHEMA_VERSION,
    STREAM_INFERENCE_TERMINAL_SCHEMA_ID,
    STREAM_INFERENCE_TERMINAL_SCHEMA_VERSION,
    STREAM_WINDOW_RESULT_SCHEMA_ID,
    STREAM_WINDOW_RESULT_SCHEMA_VERSION,
)
from robata.contracts.stream_planning import StreamWorkItemPlan
from robata.contracts.stream_window import (
    STREAM_INFERENCE_ATTEMPT_SCHEMA_ID,
    STREAM_INFERENCE_ATTEMPT_SCHEMA_VERSION,
    STREAM_INFERENCE_SCHEMA_ID,
    STREAM_INFERENCE_SCHEMA_VERSION,
)
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
from robata.inference.adapter import (
    JsonSchemaRef,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionModelAdapter,
)
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
from robata.qa_pipeline.supplemental import (
    LOCAL_SUPPLEMENTAL_QA_DENSE_POLICY_VERSION,
    SUPPLEMENTAL_QA_DENSE_INPUT_PROJECTION_VERSION,
    SUPPLEMENTAL_QA_DENSE_RESULT_PROJECTION_VERSION,
)
from robata.qa_pipeline.supplemental_wire import (
    LOCAL_SUPPLEMENTAL_QA_EVIDENCE_PROJECTION_VERSION,
    LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID,
    LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION,
    LocalSupplementalQaEvidence,
)
from robata.queue.outbox import OutboxRelay, OutboxRetryPolicy
from robata.queue.stream_models import StreamTerminalEvidence
from robata.runtime.observability import (
    RuntimeObserver,
    runtime_increment,
    runtime_span,
)
from robata.sampling.materializer import (
    CanonicalSixCameraFrameIndex,
    IndexedSourceFrame,
    MaterializedFrameArtifactFact,
    OfflineTemporalPackageMaterializer,
    TemporalPackageMaterializationPolicy,
)
from robata.sampling.package_set import PackageSetBuilder
from robata.sampling.supplemental import (
    SUPPLEMENTAL_MATERIALIZER_VERSION,
    SUPPLEMENTAL_PACKAGE_PROJECTION_VERSION,
    SUPPLEMENTAL_PACKAGE_SCHEMA_VERSION,
    SUPPLEMENTAL_TARGET_PLAN_PROJECTION_VERSION,
)

if TYPE_CHECKING:
    from robata.application.canonical.mcap_source import McapMediaProcessingPolicy


LOCAL_CANONICAL_COMPOSITION_VERSION = "canonical-local-composition-v21"
LOCAL_CANONICAL_EXECUTION_CLOCK_VERSION = "canonical-local-execution-clock-v1"
LOCAL_CANONICAL_EXECUTION_TIME = "2026-07-20T00:00:00Z"
_LOCAL_CANONICAL_EXECUTION_DATETIME: Final = datetime(2026, 7, 20, tzinfo=UTC)
LOCAL_CANONICAL_TOKEN_POLICY_VERSION = "provider-token-v1"
LOCAL_CANONICAL_PARSER_VERSION = "strict-provider-claim-v1"
LOCAL_CANONICAL_REDUCTION_POLICY = "ordered-claims-v1"
LOCAL_CANONICAL_REDUCTION_POLICY_VERSION = "1.0"
LOCAL_CANONICAL_EVENT_ALLOCATOR_VERSION = "canonical-local-event-uuid5-v1"
# Runtime-only dispatch bounds. They deliberately stay outside the canonical
# execution policy: compatible batch grouping must not change identity/evidence.
LOCAL_CANONICAL_MAX_CONCURRENT_CALL_PARTS: Final = 6
LOCAL_CANONICAL_MAX_INFERENCE_BATCH_SIZE: Final = 8
LOCAL_CANONICAL_MAX_INFERENCE_BATCH_QUEUE_DELAY_MS: Final = 5
LOCAL_CANONICAL_RUN_RECEIPT_MODEL_VERSION: Final[Literal["canonical-local-run-receipt-v4"]] = (
    "canonical-local-run-receipt-v4"
)
LOCAL_CANONICAL_RUNTIME_DESCRIPTOR_MODEL_VERSION: Final[
    Literal["canonical-local-runtime-descriptor-v1"]
] = "canonical-local-runtime-descriptor-v1"
# Kept equal to the finalizer's default, but pinned explicitly for the P5
# factory context so pre-EOS artifacts and EOS terminal acceptance agree.
LOCAL_CANONICAL_STREAM_TERMINAL_POLICY_VERSION: Final = "stream-terminal-policy-v1"


class CanonicalLocalCompositionErrorCode(StrEnum):
    """Small command-boundary error vocabulary for the local composition."""

    INVALID_REQUEST = "INVALID_REQUEST"
    SOURCE_INVALID = "SOURCE_INVALID"
    LOCAL_STATE_FAILED = "LOCAL_STATE_FAILED"
    RUN_NOT_COMPLETABLE = "RUN_NOT_COMPLETABLE"
    COMPLETION_FAILED = "COMPLETION_FAILED"
    BACKPRESSURE = "BACKPRESSURE"


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
    model_version: Literal["canonical-local-run-receipt-v4"]
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
    outbox_delivery: LocalOutboxDeliverySummary
    media_quality_binding: LocalMediaQualityBinding | None
    supplemental_qa_evidence: LocalSupplementalQaEvidence | None
    review_routing: LocalReviewRoutingSummary
    replayed: bool
    fixture_inference_calls: int
    network_call_count: Literal[0]
    evidence_class: Literal["LOCAL_CONFORMANCE"]
    production_eligible: Literal[False]


class LocalCanonicalRuntimeDescriptor(StrictModel):
    """Non-authoritative policy pins for reproducible local profiling."""

    schema_version: Literal["1.0"]
    model_version: Literal["canonical-local-runtime-descriptor-v1"]
    composition_version: str
    pipeline_version: str
    execution_policy_semantic_sha256: str
    runtime_policy_semantic_sha256: str
    input_planner_version: str
    parser_version: str
    inference_policy_versions: tuple[str, ...]
    evidence_class: Literal["LOCAL_CONFORMANCE"]
    production_eligible: Literal[False]


@dataclass(frozen=True, slots=True)
class _LocalCanonicalRuntime:
    registry: SchemaRegistry
    execution_policy: CanonicalOfflineExecutionPolicy
    pipeline: CanonicalOfflinePipeline
    inference_evidence: SQLiteInferenceEvidenceLedger


_StageTerminalExecutor = Callable[[StreamWorkItemPlan], StreamTerminalEvidence | None]


@dataclass(frozen=True, slots=True)
class LocalPreEosExecutorContext:
    """Runtime-only dependencies for a real provider-neutral pre-EOS executor.

    The factory is deliberately responsible for deterministic conversion from a
    ``StreamWorkItemPlan`` to canonical invocation inputs.  The stream plan does
    not contain enough admitted package/rendered-input lineage for this
    composition layer to safely invent those inputs.
    """

    pipeline: CanonicalOfflinePipeline
    artifact_root: Path
    model_inference_schema_ref: SchemaRef
    terminal_policy_version: str


LocalPreEosExecutorFactory = Callable[[LocalPreEosExecutorContext], _StageTerminalExecutor]


def local_canonical_runtime_descriptor() -> LocalCanonicalRuntimeDescriptor:
    """Return the exact policy pins used by the local canonical composition."""

    registry = SchemaRegistry()
    execution_policy = _execution_policy()
    policies = _local_inference_policies(registry)
    runtime_policy_sha256 = _local_runtime_policy_sha256(
        coarse_qa_policy=policies[0],
        dense_qa_policy=policies[1],
        event_proposal_policy=policies[2],
        action_evidence_policy=policies[3],
        boundary_refinement_policy=policies[4],
        inference_policy=policies[5],
    )
    return LocalCanonicalRuntimeDescriptor(
        schema_version="1.0",
        model_version=LOCAL_CANONICAL_RUNTIME_DESCRIPTOR_MODEL_VERSION,
        composition_version=LOCAL_CANONICAL_COMPOSITION_VERSION,
        pipeline_version=CANONICAL_OFFLINE_PIPELINE_VERSION,
        execution_policy_semantic_sha256=execution_policy.semantic_sha256,
        runtime_policy_semantic_sha256=runtime_policy_sha256,
        input_planner_version=INFERENCE_INPUT_PLANNER_VERSION,
        parser_version=LOCAL_CANONICAL_PARSER_VERSION,
        inference_policy_versions=tuple(
            f"{policy.task.value}:{policy.policy_version}" for policy in policies
        ),
        evidence_class="LOCAL_CONFORMANCE",
        production_eligible=False,
    )


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


class LocalProviderCallDispatcher(Protocol):
    """Runtime-only dispatch boundary shared by local recording workers.

    Provider topology is injected rather than included in the canonical command,
    run identity, or persisted result.  A dispatcher owns admission and
    concurrency across pipeline instances, while each instance retains its own
    recording-affine evidence ledger.
    """

    async def dispatch(
        self,
        operation: Callable[[], Awaitable[object]],
    ) -> object:
        """Run one provider operation under the shared runtime bound."""


class _DispatchingVisionModelAdapter:
    """Apply a shared runtime dispatcher without changing the provider port."""

    def __init__(
        self,
        delegate: VisionModelAdapter,
        dispatcher: LocalProviderCallDispatcher,
    ) -> None:
        if not callable(getattr(delegate, "capabilities", None)) or not callable(
            getattr(delegate, "infer", None)
        ):
            raise TypeError("delegate must implement VisionModelAdapter")
        if not callable(getattr(dispatcher, "dispatch", None)):
            raise TypeError("provider_dispatcher must implement async dispatch")
        self._delegate = delegate
        self._dispatcher = dispatcher

    @property
    def provider(self) -> str:
        return self._delegate.provider

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        outcome = await self._dispatcher.dispatch(
            lambda: self._delegate.capabilities(model_name, model_version)
        )
        if not isinstance(outcome, ModelCapabilities):
            raise TypeError("provider dispatcher returned an invalid capability result")
        return outcome

    async def infer(
        self,
        request: VisionInferenceRequest,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        outcome = await self._dispatcher.dispatch(lambda: self._delegate.infer(request))
        if not isinstance(outcome, (VisionInferenceSuccess, VisionInferenceFailure)):
            raise TypeError("provider dispatcher returned an invalid inference result")
        return outcome


class _DispatchingBatchVisionModelAdapter(_DispatchingVisionModelAdapter):
    """Preserve a delegate's native batch boundary under shared dispatch."""

    async def infer_batch(
        self,
        requests: tuple[VisionInferenceRequest, ...],
    ) -> tuple[VisionInferenceSuccess | VisionInferenceFailure, ...]:
        infer_batch = getattr(self._delegate, "infer_batch", None)
        if not callable(infer_batch):
            raise TypeError("delegate does not implement infer_batch")
        outcome = await self._dispatcher.dispatch(lambda: infer_batch(requests))
        if not isinstance(outcome, tuple) or not all(
            isinstance(item, (VisionInferenceSuccess, VisionInferenceFailure)) for item in outcome
        ):
            raise TypeError("provider dispatcher returned an invalid batch inference result")
        return cast(tuple[VisionInferenceSuccess | VisionInferenceFailure, ...], outcome)


_CanonicalSourceLoader = Callable[
    [SchemaRegistry, SQLiteWorkScheduler, str, _StageTerminalExecutor | None],
    _CanonicalSourceInputs,
]
_MediaQualityDocumentLoader = Callable[[SchemaRegistry], dict[str, object]]
_SupplementalQaEvidenceBuilder = Callable[
    [SchemaRegistry, _CanonicalSourceInputs],
    LocalSupplementalQaEvidence | None,
]
_SupplementalQaEvidenceLoader = Callable[
    [SchemaRegistry],
    LocalSupplementalQaEvidence | None,
]


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
    *,
    runtime_observer: RuntimeObserver | None = None,
    provider_dispatcher: LocalProviderCallDispatcher | None = None,
) -> CanonicalLocalRunReceipt:
    """Run or recover the complete local canonical path from a JSON source fixture."""

    source = _require_path(source_path, "source_path")
    state_root = _require_path(state_dir, "state_dir")
    _require_run_key(run_key)
    _require_provider_dispatcher(provider_dispatcher)
    with runtime_span(runtime_observer, "source.fixture.read_validate"):
        source_sha256, _, clock_value = _source_run_binding(source)
    source_binding_sha256 = semantic_sha256(
        {
            "source_binding_policy_version": "canonical-local-source-binding-v1",
            "source_kind": "json-source-fixture",
            "source_content_sha256": source_sha256,
        }
    )

    def load_source(
        registry: SchemaRegistry,
        _execution_scheduler: SQLiteWorkScheduler,
        _stream_run_id: str,
        _stage_terminal_executor: _StageTerminalExecutor | None = None,
    ) -> CanonicalSourceBundle:
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
        runtime_observer=runtime_observer,
        provider_dispatcher=provider_dispatcher,
    )


def run_local_canonical_mcap(
    source_path: Path,
    mapping_config: Path,
    state_dir: Path,
    run_key: str = "primary",
    *,
    allow_unapproved_profile: bool = False,
    max_duration_ns: int | None = None,
    media_processing_policy: McapMediaProcessingPolicy | None = None,
    runtime_observer: RuntimeObserver | None = None,
    provider_dispatcher: LocalProviderCallDispatcher | None = None,
    stage_terminal_executor: _StageTerminalExecutor | None = None,
    pre_eos_executor_factory: LocalPreEosExecutorFactory | None = None,
) -> CanonicalLocalRunReceipt:
    """Run or recover the complete local canonical path from a real MCAP.

    ``pre_eos_executor_factory`` is the P5 provider-neutral integration point.
    It receives the constructed canonical runtime before source preparation and
    returns the one hook used for both incremental pre-EOS work and any work
    still pending at EOS. ``stage_terminal_executor`` remains available for
    direct injection, but the two options are mutually exclusive.
    """

    from robata.application.canonical.mcap_source import (
        DEFAULT_MCAP_MEDIA_PROCESSING_POLICY,
        CanonicalMcapSourceBundle,
        CanonicalMcapSourceError,
        McapMediaProcessingPolicy,
        authorize_mcap_mapping,
        load_canonical_mcap_source,
        mcap_media_processing_policy_projection,
    )

    source = _require_path(source_path, "source_path")
    mapping = _require_path(mapping_config, "mapping_config")
    state_root = _require_path(state_dir, "state_dir")
    _require_run_key(run_key)
    _require_provider_dispatcher(provider_dispatcher)
    if stage_terminal_executor is not None and not callable(stage_terminal_executor):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "stage_terminal_executor must be callable or None",
        )
    if pre_eos_executor_factory is not None and not callable(pre_eos_executor_factory):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "pre_eos_executor_factory must be callable or None",
        )
    if stage_terminal_executor is not None and pre_eos_executor_factory is not None:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "stage_terminal_executor and pre_eos_executor_factory are mutually exclusive",
        )
    if not isinstance(allow_unapproved_profile, bool):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "allow_unapproved_profile must be a boolean",
        )
    if max_duration_ns is not None and (
        isinstance(max_duration_ns, bool)
        or not isinstance(max_duration_ns, int)
        or max_duration_ns <= 0
    ):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "max_duration_ns must be a positive integer or None",
        )
    resolved_media_processing_policy = (
        DEFAULT_MCAP_MEDIA_PROCESSING_POLICY
        if media_processing_policy is None
        else media_processing_policy
    )
    if not isinstance(resolved_media_processing_policy, McapMediaProcessingPolicy):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "media_processing_policy must be McapMediaProcessingPolicy or None",
        )
    try:
        # Authorization must precede the first source read.
        with runtime_span(runtime_observer, "source.mapping.authorize"):
            authorization = authorize_mcap_mapping(
                mapping,
                allow_unapproved_profile=allow_unapproved_profile,
            )
        with runtime_span(runtime_observer, "source.hash"):
            source_sha256 = _hash_source_file(source, label="MCAP source")
    except CanonicalMcapSourceError as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
            str(error),
        ) from error

    quality_registry = SchemaRegistry()
    quality_schema_ref = quality_registry.resolve_version(
        LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID,
        LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION,
    ).ref
    supplemental_schema_ref = quality_registry.resolve_version(
        LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID,
        LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION,
    ).ref
    source_binding_sha256 = semantic_sha256(
        {
            "source_binding_policy_version": "canonical-local-mcap-source-binding-v7",
            "source_kind": "mcap",
            "source_content_sha256": source_sha256,
            "mapping_profile_semantic_sha256": authorization.semantic_sha256,
            "primary_completion_command_projection_version": (
                CANONICAL_PRIMARY_COMPLETION_COMMAND_PROJECTION_VERSION
            ),
            "max_duration_ns": None if max_duration_ns is None else str(max_duration_ns),
            "media_processing_policy": mcap_media_processing_policy_projection(
                resolved_media_processing_policy
            ),
            "media_quality_policy_version": LOCAL_MEDIA_QUALITY_POLICY_VERSION,
            "neighbor_target_policy_version": LOCAL_NEIGHBOR_TARGET_POLICY_VERSION,
            "media_quality_report_format_version": (LOCAL_MEDIA_QUALITY_REPORT_FORMAT_VERSION),
            "media_quality_report_schema_ref": quality_schema_ref.model_dump(mode="json"),
            "media_quality_binding_projection_version": (
                LOCAL_MEDIA_QUALITY_BINDING_PROJECTION_VERSION
            ),
            "media_quality_source_binding_projection_version": (
                MEDIA_QUALITY_SOURCE_BINDING_PROJECTION_VERSION
            ),
            "supplemental_qa_runtime_version": LOCAL_SUPPLEMENTAL_QA_RUNTIME_VERSION,
            "supplemental_target_plan_projection_version": (
                SUPPLEMENTAL_TARGET_PLAN_PROJECTION_VERSION
            ),
            "supplemental_selection_tolerance_ns": str(LOCAL_SUPPLEMENTAL_SELECTION_TOLERANCE_NS),
            "supplemental_tie_break_policy_version": (LOCAL_SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION),
            "supplemental_dedupe_policy_version": (LOCAL_SUPPLEMENTAL_DEDUPE_POLICY_VERSION),
            "supplemental_materializer_version": SUPPLEMENTAL_MATERIALIZER_VERSION,
            "supplemental_package_schema_version": SUPPLEMENTAL_PACKAGE_SCHEMA_VERSION,
            "supplemental_package_projection_version": (SUPPLEMENTAL_PACKAGE_PROJECTION_VERSION),
            "supplemental_qa_consumer_policy_version": (LOCAL_SUPPLEMENTAL_QA_DENSE_POLICY_VERSION),
            "supplemental_qa_input_projection_version": (
                SUPPLEMENTAL_QA_DENSE_INPUT_PROJECTION_VERSION
            ),
            "supplemental_qa_result_projection_version": (
                SUPPLEMENTAL_QA_DENSE_RESULT_PROJECTION_VERSION
            ),
            "supplemental_qa_evidence_projection_version": (
                LOCAL_SUPPLEMENTAL_QA_EVIDENCE_PROJECTION_VERSION
            ),
            "supplemental_qa_evidence_schema_ref": (
                supplemental_schema_ref.model_dump(mode="json")
            ),
        }
    )

    mcap_state_root = (
        state_root / "mcap" / _stable_uuid("canonical-mcap-source-state", source_binding_sha256)
    )

    def load_source(
        registry: SchemaRegistry,
        execution_scheduler: SQLiteWorkScheduler,
        stream_run_id: str,
        resolved_stage_terminal_executor: _StageTerminalExecutor | None = None,
    ) -> CanonicalMcapSourceBundle:
        try:
            return load_canonical_mcap_source(
                source,
                authorization=authorization,
                state_dir=mcap_state_root,
                expected_source_sha256=source_sha256,
                schema_registry=registry,
                clock=lambda: _LOCAL_CANONICAL_EXECUTION_DATETIME,
                max_duration_ns=max_duration_ns,
                media_processing_policy=resolved_media_processing_policy,
                runtime_observer=runtime_observer,
                execution_scheduler=execution_scheduler,
                stream_run_id=stream_run_id,
                stream_artifact_root=state_root / "stream-artifacts",
                stage_terminal_executor=resolved_stage_terminal_executor,
            )
        except CanonicalMcapSourceError as error:
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
                str(error),
            ) from error

    def load_media_quality_document(registry: SchemaRegistry) -> dict[str, object]:
        try:
            return load_registered_local_media_quality_report_document(
                mcap_state_root / "media-quality-report.json",
                registry,
            )
        except (OSError, TypeError, ValueError) as error:
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED,
                f"invalid persisted media quality report: {error}",
            ) from error

    def build_supplemental_qa_evidence(
        registry: SchemaRegistry,
        bundle: _CanonicalSourceInputs,
    ) -> LocalSupplementalQaEvidence | None:
        if not isinstance(bundle, CanonicalMcapSourceBundle):
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED,
                "MCAP supplemental QA received an incompatible source bundle",
            )
        try:
            return build_and_publish_local_supplemental_qa_evidence(
                bundle=bundle,
                state_dir=mcap_state_root,
                registry=registry,
                created_at=LOCAL_CANONICAL_EXECUTION_TIME,
            )
        except (OSError, TypeError, ValueError) as error:
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED,
                f"local supplemental QA failed: {error}",
            ) from error

    def load_supplemental_qa_evidence(
        registry: SchemaRegistry,
    ) -> LocalSupplementalQaEvidence | None:
        try:
            return load_and_verify_local_supplemental_qa_evidence(
                media_quality_document=load_media_quality_document(registry),
                state_dir=mcap_state_root,
                expected_source_content_sha256=source_sha256,
                registry=registry,
            )
        except (OSError, TypeError, ValueError) as error:
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED,
                f"invalid persisted supplemental QA evidence: {error}",
            ) from error

    return _run_local_canonical(
        state_root=state_root,
        run_key=run_key,
        source_sha256=source_sha256,
        source_binding_sha256=source_binding_sha256,
        source_loader=load_source,
        media_quality_document_loader=load_media_quality_document,
        supplemental_qa_evidence_builder=build_supplemental_qa_evidence,
        supplemental_qa_evidence_loader=load_supplemental_qa_evidence,
        runtime_observer=runtime_observer,
        provider_dispatcher=provider_dispatcher,
        stage_terminal_executor=stage_terminal_executor,
        pre_eos_executor_factory=pre_eos_executor_factory,
    )


def _run_local_canonical(
    *,
    state_root: Path,
    run_key: str,
    source_sha256: str,
    source_binding_sha256: str,
    source_loader: _CanonicalSourceLoader,
    media_quality_document_loader: _MediaQualityDocumentLoader | None = None,
    supplemental_qa_evidence_builder: _SupplementalQaEvidenceBuilder | None = None,
    supplemental_qa_evidence_loader: _SupplementalQaEvidenceLoader | None = None,
    runtime_observer: RuntimeObserver | None = None,
    provider_dispatcher: LocalProviderCallDispatcher | None = None,
    stage_terminal_executor: _StageTerminalExecutor | None = None,
    pre_eos_executor_factory: LocalPreEosExecutorFactory | None = None,
) -> CanonicalLocalRunReceipt:
    with runtime_span(runtime_observer, "canonical.composition"):
        return _run_local_canonical_inner(
            state_root=state_root,
            run_key=run_key,
            source_sha256=source_sha256,
            source_binding_sha256=source_binding_sha256,
            source_loader=source_loader,
            media_quality_document_loader=media_quality_document_loader,
            supplemental_qa_evidence_builder=supplemental_qa_evidence_builder,
            supplemental_qa_evidence_loader=supplemental_qa_evidence_loader,
            runtime_observer=runtime_observer,
            provider_dispatcher=provider_dispatcher,
            stage_terminal_executor=stage_terminal_executor,
            pre_eos_executor_factory=pre_eos_executor_factory,
        )


def _run_local_canonical_inner(
    *,
    state_root: Path,
    run_key: str,
    source_sha256: str,
    source_binding_sha256: str,
    source_loader: _CanonicalSourceLoader,
    media_quality_document_loader: _MediaQualityDocumentLoader | None = None,
    supplemental_qa_evidence_builder: _SupplementalQaEvidenceBuilder | None = None,
    supplemental_qa_evidence_loader: _SupplementalQaEvidenceLoader | None = None,
    runtime_observer: RuntimeObserver | None = None,
    provider_dispatcher: LocalProviderCallDispatcher | None = None,
    stage_terminal_executor: _StageTerminalExecutor | None = None,
    pre_eos_executor_factory: LocalPreEosExecutorFactory | None = None,
) -> CanonicalLocalRunReceipt:
    """Run the shared canonical flow after source authorization and binding."""

    if (supplemental_qa_evidence_builder is None) != (supplemental_qa_evidence_loader is None):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "supplemental QA builder and loader must be configured together",
        )
    if stage_terminal_executor is not None and not callable(stage_terminal_executor):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "stage_terminal_executor must be callable or None",
        )
    if pre_eos_executor_factory is not None and not callable(pre_eos_executor_factory):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "pre_eos_executor_factory must be callable or None",
        )
    if stage_terminal_executor is not None and pre_eos_executor_factory is not None:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "stage_terminal_executor and pre_eos_executor_factory are mutually exclusive",
        )
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
    # The persistent evidence ledger belongs to the one local composition.  Keep it
    # alive across every stage, then close it before this invocation releases its
    # state directory (including on a failed pipeline).
    runtime: _LocalCanonicalRuntime | None = None

    def build_runtime() -> _LocalCanonicalRuntime:
        return _build_runtime(
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
            runtime_observer=runtime_observer,
            provider_dispatcher=provider_dispatcher,
        )

    try:
        with runtime_span(runtime_observer, "completion.storage.open"):
            state_root.mkdir(parents=True, exist_ok=True)
            completion_repository = SQLitePrimaryCompletionRepository(
                state_root / "primary-completion.sqlite3",
                registry=registry,
                runtime_observer=runtime_observer,
            )
            work_scheduler = SQLiteWorkScheduler(
                state_root / "work-scheduler.sqlite3",
                runtime_observer=runtime_observer,
            )
            publish_work = CanonicalActionPublishWorkCoordinator(
                scheduler=work_scheduler,
                repository=completion_repository,
            )
            stream_schedulers = DurableStreamWindowScheduler.recover_registered(
                execution_scheduler=work_scheduler,
                stream_run_id=run_id,
                clock=lambda: _LOCAL_CANONICAL_EXECUTION_DATETIME,
            )
            runtime_increment(
                runtime_observer,
                "stream.scheduler.recovered_graphs",
                len(stream_schedulers),
            )
        with runtime_span(runtime_observer, "completion.recovery.lookup"):
            recovered = completion_repository.get(run_id)
        if recovered is not None:
            runtime_increment(
                runtime_observer,
                "canonical.execution_paths",
                attributes={"replayed": True},
            )
            with runtime_span(runtime_observer, "completion.scheduler.reconcile"):
                publish_work.reconcile(recovered)
            with runtime_span(runtime_observer, "stream.finalization.recover"):
                recording_result_references = tuple(
                    reference
                    for reference in recovered.evidence_references
                    if reference.role is PrimaryCompletionEvidenceRole.STREAM_RECORDING_RESULT
                )
                if len(recording_result_references) != len(stream_schedulers):
                    raise CanonicalLocalCompositionError(
                        CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED,
                        "stream scheduler and authoritative recording evidence disagree",
                    )
                stream_recording_evidence = tuple(
                    load_completed_local_stream_recording_result(
                        scheduler=scheduler,
                        artifact_root=state_root / "stream-artifacts",
                        schema_ref=reference.schema_ref,
                        expected_exact_sha256=reference.exact_bytes_sha256,
                        expected_byte_count=reference.byte_count,
                    )
                    for scheduler, reference in zip(
                        stream_schedulers,
                        recording_result_references,
                        strict=True,
                    )
                )
                _relay_local_stream_outbox(stream_schedulers, state_root)
            runtime_increment(
                runtime_observer,
                "durable_work.terminal_outcomes",
                attributes={
                    "replayed": True,
                    "stage": "ACTION_PUBLISH",
                    "state": "SUCCEEDED",
                },
            )
            with runtime_span(runtime_observer, "delivery.outbox.reconcile"):
                outbox_delivery = _reconcile_local_outbox(
                    recovered,
                    state_root=state_root,
                    primary_database_path=completion_repository.path,
                    registry=registry,
                    runtime_observer=runtime_observer,
                )
            with runtime_span(runtime_observer, "quality.evidence.load"):
                media_quality_document, media_quality_binding = _load_local_media_quality_evidence(
                    media_quality_document_loader,
                    registry,
                )
                supplemental_qa_evidence = (
                    None
                    if supplemental_qa_evidence_loader is None
                    else supplemental_qa_evidence_loader(registry)
                )
            evidence_references = _local_completion_evidence_references(
                media_quality_document=media_quality_document,
                supplemental_qa_evidence=supplemental_qa_evidence,
                stream_recording_evidence=stream_recording_evidence,
            )
            if recovered.evidence_references != evidence_references:
                raise CanonicalLocalCompositionError(
                    CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED,
                    "persisted local evidence differs from authoritative completion references",
                )
            return _receipt(
                recovered,
                replayed=True,
                fixture_inference_calls=0,
                state_root=state_root,
                primary_database_path=completion_repository.path,
                registry=registry,
                media_quality_binding=media_quality_binding,
                supplemental_qa_evidence=supplemental_qa_evidence,
                outbox_delivery=outbox_delivery,
                runtime_observer=runtime_observer,
            )

        runtime_increment(
            runtime_observer,
            "canonical.execution_paths",
            attributes={"replayed": False},
        )
        resolved_stage_terminal_executor = stage_terminal_executor
        if pre_eos_executor_factory is not None:
            # Source preparation incrementally drains ready windows.  Build the
            # canonical runtime first so the real provider-neutral executor can
            # use the same orchestrator/evidence ledger before the first window
            # becomes eligible for pre-EOS execution.
            with runtime_span(runtime_observer, "canonical.runtime.build"):
                runtime = build_runtime()
            context = LocalPreEosExecutorContext(
                pipeline=runtime.pipeline,
                artifact_root=state_root / "stream-artifacts",
                model_inference_schema_ref=registry.resolve_version(
                    MODEL_INFERENCE_SCHEMA_ID,
                    "1.0.0",
                ).ref,
                terminal_policy_version=LOCAL_CANONICAL_STREAM_TERMINAL_POLICY_VERSION,
            )
            with runtime_span(runtime_observer, "stream.pre_eos_executor.build"):
                resolved_stage_terminal_executor = pre_eos_executor_factory(context)
            if not callable(resolved_stage_terminal_executor):
                raise CanonicalLocalCompositionError(
                    CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
                    "pre_eos_executor_factory must return a callable stage terminal executor",
                )
        with runtime_span(runtime_observer, "source.prepare"):
            bundle = source_loader(
                registry,
                work_scheduler,
                run_id,
                resolved_stage_terminal_executor,
            )
        if bundle.source_content_sha256 != source_sha256:
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
                "source bytes changed while preparing the canonical run",
            )
        stream_schedulers = DurableStreamWindowScheduler.recover_registered(
            execution_scheduler=work_scheduler,
            stream_run_id=run_id,
            clock=lambda: _LOCAL_CANONICAL_EXECUTION_DATETIME,
        )
        runtime_increment(
            runtime_observer,
            "stream.scheduler.prepared_graphs",
            len(stream_schedulers),
        )
        with runtime_span(runtime_observer, "quality.evidence.prepare"):
            media_quality_document, media_quality_binding = _load_local_media_quality_evidence(
                media_quality_document_loader,
                registry,
            )
            supplemental_qa_evidence = (
                None
                if supplemental_qa_evidence_builder is None
                else supplemental_qa_evidence_builder(registry, bundle)
            )
        if runtime is None:
            with runtime_span(runtime_observer, "canonical.runtime.build"):
                runtime = build_runtime()
        processing_run = CanonicalProcessingRunContext.fresh(
            run_id=run_id,
            recording_identity=bundle.admitted_context.recording_identity,
            mcap_id=bundle.admitted_context.ready_manifest.mcap_id,
            pipeline_version=CANONICAL_OFFLINE_PIPELINE_VERSION,
            config_sha256=execution_policy.semantic_sha256,
            started_at=started_at,
        )
        with runtime_span(runtime_observer, "completion.run.begin"):
            processing_run = CanonicalProcessingRunContext.resume(
                completion_repository.begin_run(processing_run)
            )
        source_media_quality_report = getattr(bundle, "media_quality_report", None)
        product_qa_context = (
            None
            if source_media_quality_report is None
            else product_qa_context_from_media_quality_report(
                cast(LocalMediaQualityReport, source_media_quality_report)
            )
        )
        with runtime_span(runtime_observer, "inference.pipeline"):
            result = asyncio.run(
                runtime.pipeline.run(
                    processing_run=processing_run,
                    admitted_context=bundle.admitted_context,
                    requested_interval=bundle.requested_interval,
                    sampling_plan=bundle.sampling_plan,
                    frame_index=bundle.frame_index,
                    artifact_resolver=bundle.resolve_artifact,
                    product_qa_context=product_qa_context,
                )
            )
        runtime_increment(
            runtime_observer,
            "inference.fixture_calls",
            result.adapter_infer_calls,
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

        with runtime_span(runtime_observer, "stream.finalization.execute"):
            stream_finalizations = _finalize_local_stream_graphs(
                schedulers=stream_schedulers,
                state_root=state_root,
                registry=registry,
                bundle=bundle,
                canonical_result=result,
                stage_terminal_executor=resolved_stage_terminal_executor,
            )
        runtime_increment(
            runtime_observer,
            "stream.finalization.completed_graphs",
            len(stream_finalizations),
        )
        runtime_increment(
            runtime_observer,
            "stream.finalization.executed_work",
            sum(item.newly_executed_work_count for item in stream_finalizations),
        )
        evidence_references = _local_completion_evidence_references(
            media_quality_document=media_quality_document,
            supplemental_qa_evidence=supplemental_qa_evidence,
            stream_recording_evidence=tuple(
                (item.recording_result, item.recording_result_evidence_ref)
                for item in stream_finalizations
            ),
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
            with runtime_span(runtime_observer, "completion.identity.prepare"):
                prepared_identities = identity_service.prepare_batch(
                    snapshot=completion_repository.snapshot(result.recording_identity),
                    admitted_context=bundle.admitted_context,
                    hypotheses=result.hypotheses,
                    enriched_outputs=enriched_outputs,
                    decided_at=decided_at,
                )

        with runtime_span(runtime_observer, "completion.publications.prepare"):
            publications = prepare_initial_action_event_publications(
                context=bundle.admitted_context,
                result=result,
                prepared_identities=prepared_identities,
                execution_policy=execution_policy,
            )
        with runtime_span(
            runtime_observer,
            "completion.evidence.audit",
            {"mode": "incremental"},
        ):
            runtime.inference_evidence.verify_completion_seal()
        with runtime_span(runtime_observer, "completion.command.serialize_validate"):
            prepared_command = prepare_primary_completion_command(
                result=result,
                prepared_identities=prepared_identities,
                action_event_publications=publications,
                evidence_references=evidence_references,
                registry=registry,
            )
        with runtime_span(runtime_observer, "completion.commit"):
            commit_result = publish_work.commit_prepared(prepared_command)
        runtime_increment(
            runtime_observer,
            "durable_work.terminal_outcomes",
            attributes={
                "replayed": commit_result.replayed,
                "stage": "ACTION_PUBLISH",
                "state": "SUCCEEDED",
            },
        )
        return _receipt(
            commit_result.committed,
            replayed=commit_result.replayed,
            fixture_inference_calls=result.adapter_infer_calls,
            state_root=state_root,
            primary_database_path=completion_repository.path,
            registry=registry,
            media_quality_binding=media_quality_binding,
            supplemental_qa_evidence=supplemental_qa_evidence,
            runtime_observer=runtime_observer,
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
        LocalStreamFinalizationError,
        LogicalNodeRegistryError,
        WorkSchedulerError,
        CanonicalDurableWorkError,
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
    finally:
        # Do not let a close failure replace an authoritative pipeline exception.
        active_exception = sys.exc_info()[0] is not None
        if runtime is not None:
            try:
                runtime.inference_evidence.close()
            except SQLiteInferenceEvidenceLedgerError as error:
                if not active_exception:
                    raise CanonicalLocalCompositionError(
                        CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED,
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
    runtime_observer: RuntimeObserver | None = None,
    provider_dispatcher: LocalProviderCallDispatcher | None = None,
) -> _LocalCanonicalRuntime:
    """Compose runtime adapters from the exact policies bound into the run ID."""

    inference_evidence = SQLiteInferenceEvidenceLedger(
        state_root / "inference-evidence.sqlite3",
        registry,
        raw_bytes_cas_root=state_root / "raw-provider-cas",
        runtime_observer=runtime_observer,
    )
    try:
        barrier_storage = SQLiteBarrierStorage(
            state_root / "runs" / run_id / "inference-call-barrier.sqlite3",
            runtime_observer=runtime_observer,
        )
        parser = StrictProviderClaimParser(
            registry,
            parser_version=LOCAL_CANONICAL_PARSER_VERSION,
        )
        adapter: VisionModelAdapter = OfflineFixtureVisionAdapter(
            capabilities=_capabilities(observed_at),
            raw_store=inference_evidence,
            parser=parser,
            response_factory=_fixture_claim_bytes,
        )
        if provider_dispatcher is not None:
            wrapper_type = (
                _DispatchingBatchVisionModelAdapter
                if callable(getattr(adapter, "infer_batch", None))
                else _DispatchingVisionModelAdapter
            )
            adapter = wrapper_type(adapter, provider_dispatcher)
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
            logical_node_registry=LocalLogicalNodeRegistry(
                state_root / "logical-nodes",
                runtime_observer=runtime_observer,
            ),
            execution_policy=execution_policy,
            max_concurrent_call_parts=LOCAL_CANONICAL_MAX_CONCURRENT_CALL_PARTS,
            max_inference_batch_size=LOCAL_CANONICAL_MAX_INFERENCE_BATCH_SIZE,
            max_inference_batch_queue_delay_ms=(LOCAL_CANONICAL_MAX_INFERENCE_BATCH_QUEUE_DELAY_MS),
            inference_ledger=inference_evidence,
            evidence_store=inference_evidence,
            barrier_storage=barrier_storage,
            call_barrier_storage=barrier_storage,
            clock=lambda: clock_value,
            runtime_observer=runtime_observer,
        )
        return _LocalCanonicalRuntime(
            registry=registry,
            execution_policy=execution_policy,
            pipeline=pipeline,
            inference_evidence=inference_evidence,
        )
    except BaseException:
        # Construction failed before ownership could transfer to the local runtime.
        with suppress(SQLiteInferenceEvidenceLedgerError):
            inference_evidence.close()
        raise


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
            "semantic_projection_version": "canonical-local-runtime-policy-v10",
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
            "local_stream_mock_executor_policy_version": (
                LOCAL_STREAM_MOCK_EXECUTOR_POLICY_VERSION
            ),
            "local_stream_reduction_policy_version": (LOCAL_STREAM_CAUSAL_REDUCTION_POLICY_VERSION),
            "local_stream_recording_reduction_policy_version": (
                LOCAL_STREAM_RECORDING_REDUCTION_POLICY_VERSION
            ),
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
        for item in items:
            first_by_coordinate.setdefault(
                (item.package_ordinal, item.camera_ordinal),
                item,
            )
        part_coordinates = tuple(first_by_coordinate.values())
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


def _load_local_media_quality_evidence(
    loader: _MediaQualityDocumentLoader | None,
    registry: SchemaRegistry,
) -> tuple[Mapping[str, object] | None, LocalMediaQualityBinding | None]:
    if loader is None:
        return None, None
    document = loader(registry)
    try:
        binding = derive_local_media_quality_binding_document(document, registry)
    except (TypeError, ValueError) as error:
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED,
            f"invalid persisted media quality binding: {error}",
        ) from error
    return document, binding


def _local_completion_evidence_references(
    *,
    media_quality_document: Mapping[str, object] | None,
    supplemental_qa_evidence: LocalSupplementalQaEvidence | None,
    stream_recording_evidence: tuple[
        tuple[
            LocalStreamRecordingResult
            | LocalStreamRecordingResultV2
            | LocalStreamRecordingResultV3
            | LocalStreamRecordingResultV4,
            ArtifactEvidenceRef,
        ],
        ...,
    ] = (),
) -> tuple[PrimaryCompletionEvidenceReference, ...]:
    references: list[PrimaryCompletionEvidenceReference] = []
    if len(stream_recording_evidence) > 1:
        raise ValueError("one primary completion may bind only one stream recording result")
    if media_quality_document is not None:
        references.append(
            _registered_evidence_reference(
                role=PrimaryCompletionEvidenceRole.MEDIA_QUALITY_REPORT,
                document=media_quality_document,
            )
        )
    if supplemental_qa_evidence is not None:
        references.append(
            _registered_evidence_reference(
                role=PrimaryCompletionEvidenceRole.SUPPLEMENTAL_QA_EVIDENCE,
                document=supplemental_qa_evidence.model_dump(mode="json"),
            )
        )
    for result, artifact_ref in stream_recording_evidence:
        payload = canonical_json_bytes(result)
        if (
            artifact_ref.schema_ref != result.schema_ref
            or artifact_ref.exact_sha256 != exact_bytes_sha256(payload)
            or artifact_ref.byte_count != len(payload)
        ):
            raise ValueError("stream recording result differs from its content-addressed artifact")
        references.append(
            PrimaryCompletionEvidenceReference(
                role=PrimaryCompletionEvidenceRole.STREAM_RECORDING_RESULT,
                schema_ref=result.schema_ref,
                semantic_sha256=result.recording_result_semantic_sha256,
                exact_bytes_sha256=artifact_ref.exact_sha256,
                byte_count=artifact_ref.byte_count,
            )
        )
    return tuple(sorted(references, key=lambda reference: reference.role.value))


def _registered_evidence_reference(
    *,
    role: PrimaryCompletionEvidenceRole,
    document: Mapping[str, object],
) -> PrimaryCompletionEvidenceReference:
    payload = canonical_json_bytes(document)
    semantic_digest = document.get("semantic_sha256")
    if not isinstance(semantic_digest, str):
        raise ValueError("registered evidence semantic_sha256 must be a string")
    return PrimaryCompletionEvidenceReference(
        role=role,
        schema_ref=SchemaRef.model_validate(document.get("schema_ref"), strict=True),
        semantic_sha256=semantic_digest,
        exact_bytes_sha256=exact_bytes_sha256(payload),
        byte_count=len(payload),
    )


def _finalize_local_stream_graphs(
    *,
    schedulers: tuple[DurableStreamWindowScheduler, ...],
    state_root: Path,
    registry: SchemaRegistry,
    bundle: _CanonicalSourceInputs,
    canonical_result: CanonicalOfflineRunResult,
    stage_terminal_executor: _StageTerminalExecutor | None = None,
) -> tuple[LocalStreamFinalizationOutcome, ...]:
    if not schedulers:
        return ()
    schema_refs = LocalStreamFinalizationSchemaRefs(
        local_work_receipt=registry.resolve_version(
            LOCAL_STREAM_WORK_RECEIPT_SCHEMA_ID,
            LOCAL_STREAM_WORK_RECEIPT_SCHEMA_VERSION,
        ).ref,
        stream_window_result=registry.resolve_version(
            STREAM_WINDOW_RESULT_SCHEMA_ID,
            STREAM_WINDOW_RESULT_SCHEMA_VERSION,
        ).ref,
        recording_finalization=registry.resolve_version(
            RECORDING_FINALIZATION_SCHEMA_ID,
            RECORDING_FINALIZATION_SCHEMA_VERSION,
        ).ref,
        stream_recording_result=registry.resolve_version(
            LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
            LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION,
        ).ref,
        window_inference_plan=registry.resolve_version(
            LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_ID,
            LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_VERSION,
        ).ref,
        window_semantic_evidence_v2=registry.resolve_version(
            LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_ID,
            LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_VERSION,
        ).ref,
        stream_inference_identity=registry.resolve_version(
            STREAM_INFERENCE_SCHEMA_ID,
            STREAM_INFERENCE_SCHEMA_VERSION,
        ).ref,
        stream_inference_attempt=registry.resolve_version(
            STREAM_INFERENCE_ATTEMPT_SCHEMA_ID,
            STREAM_INFERENCE_ATTEMPT_SCHEMA_VERSION,
        ).ref,
        stream_inference_intent=registry.resolve_version(
            STREAM_INFERENCE_INTENT_SCHEMA_ID,
            STREAM_INFERENCE_INTENT_SCHEMA_VERSION,
        ).ref,
        stream_accepted_call=registry.resolve_version(
            STREAM_ACCEPTED_CALL_SCHEMA_ID,
            STREAM_ACCEPTED_CALL_SCHEMA_VERSION,
        ).ref,
        stream_inference_terminal=registry.resolve_version(
            STREAM_INFERENCE_TERMINAL_SCHEMA_ID,
            STREAM_INFERENCE_TERMINAL_SCHEMA_VERSION,
        ).ref,
        window_semantic_evidence=registry.resolve_version(
            LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_ID,
            LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_VERSION,
        ).ref,
        # The source-side pre-EOS executor can already have completed typed QA/event
        # terminals by the time this EOS coordinator is constructed.  Pin the
        # existing inference artifact here so replay reads that evidence rather than
        # treating it as a local-conformance receipt.
        model_inference=registry.resolve_version(
            MODEL_INFERENCE_SCHEMA_ID,
            "1.0.0",
        ).ref,
    )
    ready = bundle.admitted_context.ready_manifest
    final_recording = FinalRecordingFacts(
        final_source_subject_type="MCAP_RECORDING",
        final_source_subject_id=ready.mcap_id,
        final_source_exact_sha256=bundle.source_content_sha256,
        final_recording_identity=ready.recording_identity,
        final_duration_ns=ready.recording.duration_ns,
    )
    delivery = _local_stream_delivery_authority(schedulers)
    outcomes = tuple(
        LocalConformanceStreamFinalizer(
            scheduler=scheduler,
            delivery_authority=delivery,
            artifact_root=state_root / "stream-artifacts",
            schema_refs=schema_refs,
            final_recording=final_recording,
            canonical_result=canonical_result,
            source_timeline_origin_ns=(
                bundle.admitted_context.alignment_manifest.canonical_origin.reference_timestamp_ns
            ),
            canonical_requested_interval=bundle.requested_interval,
            window_purpose=StreamPurpose.EVENT_PROPOSAL,
            terminal_policy_version=LOCAL_CANONICAL_STREAM_TERMINAL_POLICY_VERSION,
            recover_graph_before_execute=False,
            stage_terminal_executor=stage_terminal_executor,
            clock=lambda: _LOCAL_CANONICAL_EXECUTION_DATETIME,
        ).execute()
        for scheduler in schedulers
    )
    _relay_local_stream_outbox(schedulers, state_root, authority=delivery)
    return outcomes


def _local_stream_delivery_authority(
    schedulers: tuple[DurableStreamWindowScheduler, ...],
) -> SQLiteStreamDeliveryAuthority:
    if not schedulers:
        raise ValueError("stream delivery authority requires a scheduler")
    database_path = schedulers[0].database_path
    if any(scheduler.database_path != database_path for scheduler in schedulers):
        raise LocalStreamFinalizationError(
            "one canonical run cannot span multiple stream authority databases"
        )
    return SQLiteStreamDeliveryAuthority(
        SQLiteWorkScheduler(database_path),
        retry_policy=OutboxRetryPolicy(
            version="local-stream-delivery-retry-v1",
            max_attempts=3,
            base_delay_seconds=1,
            max_delay_seconds=30,
        ),
        clock=lambda: _LOCAL_CANONICAL_EXECUTION_DATETIME,
    )


def _relay_local_stream_outbox(
    schedulers: tuple[DurableStreamWindowScheduler, ...],
    state_root: Path,
    *,
    authority: SQLiteStreamDeliveryAuthority | None = None,
) -> int:
    if not schedulers:
        return 0
    store = authority or _local_stream_delivery_authority(schedulers)
    store.reconcile()
    relay = OutboxRelay(
        store=store,
        sink=SQLiteIdempotentOutboxSink(
            state_root / "stream-outbox-sink.sqlite3",
            clock=lambda: _LOCAL_CANONICAL_EXECUTION_DATETIME,
        ),
        worker_id="local-stream-outbox-relay",
        lease_duration=timedelta(minutes=5),
    )
    delivered = 0
    while relay.deliver_once() is not None:
        delivered += 1
    return delivered


def _reconcile_local_outbox(
    committed: CommittedPrimaryCompletion,
    *,
    state_root: Path,
    primary_database_path: Path,
    registry: SchemaRegistry,
    runtime_observer: RuntimeObserver | None = None,
) -> LocalOutboxDeliverySummary:
    try:
        return reconcile_local_primary_outbox(
            primary_database_path=primary_database_path,
            sink_database_path=state_root / "outbox-sink.sqlite3",
            outbox=committed.outbox,
            registry=registry,
            runtime_observer=runtime_observer,
        )
    except Exception as error:
        # Completion is already authoritative; delivery remains observable recovery work.
        return failed_local_outbox_delivery(committed.outbox, error)


def _receipt(
    committed: CommittedPrimaryCompletion,
    *,
    replayed: bool,
    fixture_inference_calls: int,
    state_root: Path,
    primary_database_path: Path,
    registry: SchemaRegistry,
    media_quality_binding: LocalMediaQualityBinding | None,
    supplemental_qa_evidence: LocalSupplementalQaEvidence | None,
    outbox_delivery: LocalOutboxDeliverySummary | None = None,
    runtime_observer: RuntimeObserver | None = None,
) -> CanonicalLocalRunReceipt:
    publications = committed.action_event_publications.publications
    if outbox_delivery is None:
        with runtime_span(runtime_observer, "delivery.outbox.reconcile"):
            delivery = _reconcile_local_outbox(
                committed,
                state_root=state_root,
                primary_database_path=primary_database_path,
                registry=registry,
                runtime_observer=runtime_observer,
            )
    else:
        delivery = outbox_delivery
    runtime_increment(
        runtime_observer,
        "delivery.outbox.outcomes",
        attributes={"outcome": delivery.outcome.value, "replayed": replayed},
    )
    if delivery.relay_attempt_count:
        runtime_increment(
            runtime_observer,
            "delivery.outbox.relay_attempts",
            delivery.relay_attempt_count,
            {"replayed": replayed},
        )
    with runtime_span(runtime_observer, "review.route"):
        try:
            review = route_local_review_after_completion(
                committed,
                state_root=state_root,
                registry=registry,
                media_quality_binding=media_quality_binding,
                runtime_observer=runtime_observer,
            )
        except Exception as error:
            # Review is downstream of primary completion and cannot replace its truth.
            review = failed_local_review_routing(error)
    runtime_increment(
        runtime_observer,
        "review.routing_outcomes",
        attributes={"disposition": review.disposition.value, "replayed": replayed},
    )
    if committed.outbox:
        runtime_increment(
            runtime_observer,
            "delivery.outbox.committed_rows_observed",
            len(committed.outbox),
            {"replayed": replayed},
        )
    runtime_increment(
        runtime_observer,
        "completion.receipts",
        attributes={"replayed": replayed, "status": committed.detail.status},
    )
    return CanonicalLocalRunReceipt(
        schema_version="1.0",
        model_version=LOCAL_CANONICAL_RUN_RECEIPT_MODEL_VERSION,
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
        outbox_delivery=delivery,
        media_quality_binding=media_quality_binding,
        supplemental_qa_evidence=supplemental_qa_evidence,
        review_routing=review,
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


def _require_provider_dispatcher(
    value: LocalProviderCallDispatcher | None,
) -> None:
    if value is not None and not callable(getattr(value, "dispatch", None)):
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.INVALID_REQUEST,
            "provider_dispatcher must implement async dispatch",
        )


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
    "LOCAL_CANONICAL_STREAM_TERMINAL_POLICY_VERSION",
    "CanonicalLocalCompositionError",
    "CanonicalLocalCompositionErrorCode",
    "CanonicalLocalRunReceipt",
    "LocalPreEosExecutorContext",
    "LocalPreEosExecutorFactory",
    "LocalProviderCallDispatcher",
    "run_local_canonical_fixture",
    "run_local_canonical_mcap",
]
