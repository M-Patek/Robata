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
from robata.adapters.sqlite_barrier import SQLiteBarrierStorage
from robata.adapters.sqlite_inference_evidence import SQLiteInferenceEvidenceLedger
from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.output_admission import (
    CANONICAL_FINAL_FUSION_CONTEXT_METADATA_KEY,
    CanonicalFinalFusionContext,
)
from robata.application.canonical_offline import (
    CANONICAL_OFFLINE_PIPELINE_VERSION,
    CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE,
    CanonicalActionEvidenceExecution,
    CanonicalOfflineConfigurationError,
    CanonicalOfflineExecutionPolicy,
    CanonicalOfflinePartStatus,
    CanonicalOfflinePipeline,
    CanonicalOfflineRunResult,
    CanonicalOfflineRunStatus,
    CanonicalOutputAdmissionDecision,
    _stable_uuid,
    canonical_output_decision_projection,
)
from robata.application.canonical_run_membership import CanonicalProcessingRunContext
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.logical_nodes import RunNodeDisposition
from robata.contracts.pipeline import SamplingPurpose
from robata.contracts.schema_registry import SchemaRegistry
from robata.contracts.temporal import PackageLineage
from robata.event_pipeline.candidate import (
    CANDIDATE_EVENT_LOGICAL_KEY_NAMESPACE,
    CANDIDATE_REDUCTION_LOGICAL_KEY_NAMESPACE,
    CandidateReductionResult,
    CanonicalCandidateEvent,
)
from robata.event_pipeline.identity_registry import (
    AdmissionEvidenceClass,
    AdmissionProof,
    EventIdentityPolicyRef,
    EventIdentityRegistryService,
    ExactFingerprintEventIdentityResolver,
    InMemoryEventIdentityRegistryRepository,
    OutputAdmissionProof,
    PlatformEnrichedEventHypothesis,
    ProductionAdmittedHypothesisFact,
    ProductionOutputAdmissionPolicyRef,
)
from robata.event_pipeline.proposer import EventProposalOutcome
from robata.event_pipeline.provisional_fusion import (
    ProvisionalFusionError,
    ProvisionalPhysicalActionFuser,
)
from robata.inference.adapter import (
    JsonSchemaRef,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.call_barrier import InferenceCallReduction
from robata.inference.enrichment import (
    ENRICHED_OUTPUT_SCHEMA_ID,
    ENRICHED_OUTPUT_SCHEMA_VERSION,
    PROVIDER_CLAIM_SCHEMA_ID,
    ProviderClaimKind,
    ProviderObservation,
    ProviderReferenceCatalog,
)
from robata.inference.input_plan import (
    INFERENCE_INPUT_PLANNER_VERSION,
    INPUT_PLAN_UUID_NAMESPACE,
    REQUEST_CATALOG_UUID_NAMESPACE,
    InferenceInputPlanner,
    RenderedProviderItem,
)
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
from robata.qa_pipeline.completion import DenseQAOutcome, QACompletionStatus
from robata.qa_pipeline.dense import DenseQAStatus
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
        snapshot_digest=_digest(
            f"capabilities:qa-coarse+qa-dense+proposal+action+boundary+fusion:{max_images_per_request}"
        ),
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


def _coarse_claim_bytes(
    request: VisionInferenceRequest,
    *,
    observation: ProviderObservation = ProviderObservation.GOOD,
    model_reported_score: float | None = None,
) -> bytes:
    """Build deterministic QA observations for either QA sampling pass."""

    assert request.task in {VisionTask.QA_COARSE, VisionTask.QA_DENSE}
    assert request.input_plan is not None
    assert request.input_plan_part_ordinal is not None
    plan = request.input_plan
    part = plan.call_plan.parts[request.input_plan_part_ordinal]
    entries = ProviderReferenceCatalog.derive_entries(
        request_catalog_sha256=plan.request_catalog.semantic_sha256,
        rendered_items=plan.rendered_items,
        token_policy_version=TOKEN_POLICY_VERSION,
    )
    first_by_coordinate: dict[tuple[int, int], RenderedProviderItem] = {}
    for item in plan.rendered_items:
        first_by_coordinate.setdefault(
            (item.package_ordinal, item.camera_ordinal),
            item,
        )
    claimed_items = tuple(
        item
        for item in first_by_coordinate.values()
        if part.start_item_ordinal <= item.provider_item_ordinal < part.end_item_ordinal_exclusive
    )
    return canonical_json_bytes(
        {
            "claims": [
                {
                    "claim_ordinal": claim_ordinal,
                    "kind": "QA_OBSERVATION",
                    "package_ordinal": item.package_ordinal,
                    "camera_ordinal": item.camera_ordinal,
                    "interval": {
                        "start_ns": str(item.aligned_timestamp_ns),
                        "end_ns": str(item.aligned_timestamp_ns + 1),
                    },
                    "label": None,
                    "observation": observation.value,
                    "evidence_tokens": [entries[item.provider_item_ordinal].correlation_token],
                    "model_reported_score": model_reported_score,
                    "conflict_codes": [],
                }
                for claim_ordinal, item in enumerate(claimed_items)
            ],
            "abstained": False,
        }
    )


def _claim_bytes(
    request: VisionInferenceRequest,
    *,
    abstained: bool = False,
    evidence_provider_item_ordinal: int | None = None,
    action_limit: int | None = None,
) -> bytes:
    if abstained:
        return canonical_json_bytes({"claims": [], "abstained": True})
    assert request.task is VisionTask.FUSION_ADJUDICATION
    assert request.input_plan is not None
    assert request.input_plan_part_ordinal is not None
    context_json = request.metadata.get(CANONICAL_FINAL_FUSION_CONTEXT_METADATA_KEY)
    assert context_json is not None
    context = CanonicalFinalFusionContext.model_validate_json(context_json, strict=True)
    entries = ProviderReferenceCatalog.derive_entries(
        request_catalog_sha256=request.input_plan.request_catalog.semantic_sha256,
        rendered_items=request.input_plan.rendered_items,
        token_policy_version=TOKEN_POLICY_VERSION,
    )
    part = request.input_plan.call_plan.parts[request.input_plan_part_ordinal]
    part_items = request.input_plan.rendered_items[
        part.start_item_ordinal : part.end_item_ordinal_exclusive
    ]
    assigned_actions = tuple(
        action
        for action in context.actions
        if action.action_ordinal % part.part_count == part.ordinal
    )
    if action_limit is not None:
        assigned_actions = tuple(
            action for action in assigned_actions if action.action_ordinal < action_limit
        )
    if evidence_provider_item_ordinal is not None and not assigned_actions:
        # A narrow negative-test hook: emit one otherwise valid action with an invalid part token.
        assigned_actions = context.actions[:1]
    claims: list[dict[str, object]] = []
    for claim_ordinal, action in enumerate(assigned_actions):
        midpoint_ns = (action.refined_interval.start_ns + action.refined_interval.end_ns) // 2
        item_ordinal = (
            min(
                part_items,
                key=lambda item: (
                    abs(item.aligned_timestamp_ns - midpoint_ns),
                    item.provider_item_ordinal,
                ),
            ).provider_item_ordinal
            if evidence_provider_item_ordinal is None
            else evidence_provider_item_ordinal
        )
        claims.append(
            {
                "claim_ordinal": claim_ordinal,
                "kind": "FUSION_HYPOTHESIS",
                "package_ordinal": None,
                "camera_ordinal": None,
                "interval": {
                    "start_ns": str(action.refined_interval.start_ns),
                    "end_ns": str(action.refined_interval.end_ns),
                },
                "label": action.label,
                "observation": "PROPOSED",
                "evidence_tokens": [entries[item_ordinal].correlation_token],
                "model_reported_score": 0.8,
                "conflict_codes": [],
            }
        )
    return canonical_json_bytes({"claims": claims, "abstained": False})


def _event_proposal_claim_bytes(
    request: VisionInferenceRequest,
    *,
    intervals: tuple[tuple[int, int], ...] = ((200_000_000, 800_000_000),),
    labels: tuple[str, ...] | None = None,
) -> bytes:
    assert request.task is VisionTask.EVENT_PROPOSAL
    assert request.input_plan is not None
    assert request.input_plan_part_ordinal is not None
    plan = request.input_plan
    part = plan.call_plan.parts[request.input_plan_part_ordinal]
    entries = ProviderReferenceCatalog.derive_entries(
        request_catalog_sha256=plan.request_catalog.semantic_sha256,
        rendered_items=plan.rendered_items,
        token_policy_version=TOKEN_POLICY_VERSION,
    )
    item_ordinals = range(part.start_item_ordinal, part.end_item_ordinal_exclusive)
    tokens = [entries[ordinal].correlation_token for ordinal in item_ordinals]
    resolved_labels = labels or tuple("grasp" for _ in intervals)
    if len(resolved_labels) != len(intervals):
        raise ValueError("event proposal labels must match intervals")
    return canonical_json_bytes(
        {
            "claims": [
                {
                    "claim_ordinal": claim_ordinal,
                    "kind": "EVENT_PROPOSAL",
                    "package_ordinal": None,
                    "camera_ordinal": None,
                    "interval": {"start_ns": str(start_ns), "end_ns": str(end_ns)},
                    "label": resolved_labels[claim_ordinal],
                    "observation": "PROPOSED",
                    "evidence_tokens": tokens,
                    "model_reported_score": 0.8,
                    "conflict_codes": [],
                }
                for claim_ordinal, (start_ns, end_ns) in enumerate(intervals)
            ],
            "abstained": False,
        }
    )


def _action_evidence_claim_bytes(
    request: VisionInferenceRequest,
    *,
    observation: ProviderObservation = ProviderObservation.SUPPORTING,
    label: str = "grasp",
) -> bytes:
    assert request.task is VisionTask.ACTION_EVIDENCE
    assert request.input_plan is not None
    assert request.input_plan_part_ordinal is not None
    plan = request.input_plan
    part = plan.call_plan.parts[request.input_plan_part_ordinal]
    entries = ProviderReferenceCatalog.derive_entries(
        request_catalog_sha256=plan.request_catalog.semantic_sha256,
        rendered_items=plan.rendered_items,
        token_policy_version=TOKEN_POLICY_VERSION,
    )
    by_coordinate: dict[tuple[int, int], list[RenderedProviderItem]] = {}
    for item in plan.rendered_items[part.start_item_ordinal : part.end_item_ordinal_exclusive]:
        by_coordinate.setdefault((item.package_ordinal, item.camera_ordinal), []).append(item)
    claims: list[dict[str, object]] = []
    for claim_ordinal, (coordinate, items) in enumerate(sorted(by_coordinate.items())):
        missing = observation is ProviderObservation.MISSING
        claims.append(
            {
                "claim_ordinal": claim_ordinal,
                "kind": "ACTION_OBSERVATION",
                "package_ordinal": coordinate[0],
                "camera_ordinal": coordinate[1],
                "interval": (
                    None
                    if missing
                    else {
                        "start_ns": str(min(item.aligned_timestamp_ns for item in items)),
                        "end_ns": str(max(item.aligned_timestamp_ns for item in items) + 1),
                    }
                ),
                "label": label,
                "observation": observation.value,
                "evidence_tokens": (
                    []
                    if missing
                    else [entries[item.provider_item_ordinal].correlation_token for item in items]
                ),
                "model_reported_score": None,
                "conflict_codes": [],
            }
        )
    return canonical_json_bytes({"claims": claims, "abstained": False})


def _boundary_refinement_claim_bytes(
    request: VisionInferenceRequest,
    *,
    observation: ProviderObservation = ProviderObservation.OBSERVED,
) -> bytes:
    assert request.task is VisionTask.BOUNDARY_REFINEMENT
    assert request.metadata.get("boundary_refinement_role") in {"ONSET", "OFFSET"}
    assert request.metadata.get("boundary_anchor_ns") is not None
    assert request.input_plan is not None
    assert request.input_plan_part_ordinal is not None
    plan = request.input_plan
    part = plan.call_plan.parts[request.input_plan_part_ordinal]
    entries = ProviderReferenceCatalog.derive_entries(
        request_catalog_sha256=plan.request_catalog.semantic_sha256,
        rendered_items=plan.rendered_items,
        token_policy_version=TOKEN_POLICY_VERSION,
    )
    by_coordinate: dict[tuple[int, int], list[RenderedProviderItem]] = {}
    for item in plan.rendered_items[part.start_item_ordinal : part.end_item_ordinal_exclusive]:
        by_coordinate.setdefault((item.package_ordinal, item.camera_ordinal), []).append(item)
    all_timestamps = tuple(item.aligned_timestamp_ns for item in plan.rendered_items)
    midpoint = (min(all_timestamps) + max(all_timestamps)) // 2
    observed_package = min(
        {item.package_ordinal for item in plan.rendered_items},
        key=lambda package_ordinal: min(
            abs(item.aligned_timestamp_ns - midpoint)
            for item in plan.rendered_items
            if item.package_ordinal == package_ordinal
        ),
    )
    claims: list[dict[str, object]] = []
    for claim_ordinal, (coordinate, coordinate_items) in enumerate(sorted(by_coordinate.items())):
        is_observed = (
            coordinate[0] == observed_package and observation is ProviderObservation.OBSERVED
        )
        ordered = sorted(
            coordinate_items,
            key=lambda item: (
                abs(item.aligned_timestamp_ns - midpoint),
                item.aligned_timestamp_ns,
                item.frame_ordinal,
            ),
        )
        cited = tuple(sorted(ordered[:2], key=lambda item: item.aligned_timestamp_ns))
        claim_observation = (
            ProviderObservation.OBSERVED
            if is_observed
            else (
                observation
                if coordinate[0] == observed_package
                else ProviderObservation.NO_BOUNDARY
            )
        )
        claims.append(
            {
                "claim_ordinal": claim_ordinal,
                "kind": "BOUNDARY_OBSERVATION",
                "package_ordinal": coordinate[0],
                "camera_ordinal": coordinate[1],
                "interval": (
                    {
                        "start_ns": str(cited[0].aligned_timestamp_ns),
                        "end_ns": str(cited[-1].aligned_timestamp_ns + 1),
                    }
                    if is_observed
                    else None
                ),
                "label": None,
                "observation": claim_observation.value,
                "evidence_tokens": (
                    [entries[item.provider_item_ordinal].correlation_token for item in cited]
                    if is_observed
                    else []
                ),
                "model_reported_score": None,
                "conflict_codes": [],
            }
        )
    return canonical_json_bytes({"claims": claims, "abstained": False})


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
        if request.task is VisionTask.FUSION_ADJUDICATION and isinstance(
            outcome, VisionInferenceSuccess
        ):
            return outcome.model_copy(update={"raw_output_artifact_id": _uuid(99_999)})
        return outcome


class _ProtocolOnlyVisionAdapter:
    """Test double that exposes only the provider-neutral adapter surface."""

    def __init__(
        self,
        delegate: OfflineFixtureVisionAdapter,
        *,
        mismatch_raw_reference: bool = False,
    ) -> None:
        self._delegate = delegate
        self._mismatch_raw_reference = mismatch_raw_reference
        self._dispatch_calls = 0

    @property
    def provider(self) -> str:
        return self._delegate.provider

    @property
    def dispatch_calls(self) -> int:
        return self._dispatch_calls

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
        self._dispatch_calls += 1
        outcome = await self._delegate.infer(request)
        if (
            self._mismatch_raw_reference
            and request.task is VisionTask.FUSION_ADJUDICATION
            and isinstance(outcome, VisionInferenceSuccess)
        ):
            return outcome.model_copy(update={"raw_output_artifact_id": _uuid(99_999)})
        return outcome


@dataclass(frozen=True)
class _Harness:
    pipeline: CanonicalOfflinePipeline
    fixture_adapter: OfflineFixtureVisionAdapter
    repository: InMemoryEventIdentityRegistryRepository
    raw_store: RawProviderBytesStore
    inference_evidence: SQLiteInferenceEvidenceLedger | None
    barrier_storage: SQLiteBarrierStorage | None
    context: AdmittedRecordingContextV2
    frame_index: CanonicalSixCameraFrameIndex
    execution_policy: CanonicalOfflineExecutionPolicy
    logical_node_registry: LocalLogicalNodeRegistry
    protocol_adapter: _ProtocolOnlyVisionAdapter | None


def _harness(
    response_factory: Callable[[VisionInferenceRequest], OfflineFixtureResponse],
    *,
    logical_registry_root: Path,
    coarse_response_factory: Callable[[VisionInferenceRequest], OfflineFixtureResponse]
    | None = None,
    dense_response_factory: Callable[[VisionInferenceRequest], OfflineFixtureResponse]
    | None = None,
    event_proposal_response_factory: Callable[[VisionInferenceRequest], OfflineFixtureResponse]
    | None = None,
    action_evidence_response_factory: Callable[[VisionInferenceRequest], OfflineFixtureResponse]
    | None = None,
    boundary_refinement_response_factory: Callable[[VisionInferenceRequest], OfflineFixtureResponse]
    | None = None,
    max_images_per_request: int = 20,
    max_attempts: int = 2,
    mismatch_raw_reference: bool = False,
    protocol_only_adapter: bool = False,
    repository: InMemoryEventIdentityRegistryRepository | None = None,
    inference_evidence_path: Path | None = None,
    barrier_path: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    frame_index_override: CanonicalSixCameraFrameIndex | None = None,
) -> _Harness:
    registry = SchemaRegistry()
    provider_schema = _schema_ref(registry, PROVIDER_CLAIM_SCHEMA_ID)
    enriched_schema = _schema_ref(registry, ENRICHED_OUTPUT_SCHEMA_ID)
    output_policy = ProductionOutputAdmissionPolicyRef(
        version="fusion-output-admission-v1",
        semantic_sha256=_digest("fusion-output-admission-v1"),
    )
    execution_policy = CanonicalOfflineExecutionPolicy.create(
        policy_version="canonical-offline-v2",
        window_policy_version="root-window-v1",
        token_policy_version=TOKEN_POLICY_VERSION,
        parser_version=PARSER_VERSION,
        enrichment_policy_version="enrichment-v1",
        projector_policy_version="fusion-projector-v2",
        reduction_policy=REDUCTION_POLICY,
        reduction_policy_version=REDUCTION_POLICY_VERSION,
        provisional_fusion_policy_version="local-provisional-action-fusion-v1",
        boundary_refinement_policy_version="local-boundary-refinement-v1",
        max_attempts=max_attempts,
        output_admission_policy=output_policy,
    )
    inference_policy = InferencePolicy(
        policy_version="offline-model-policy-v2",
        task=VisionTask.FUSION_ADJUDICATION,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="offline-adapter-v1",
        prompt_version="fusion-prompt-v2",
        prompt_artifact_id=_uuid(1_001),
        prompt_sha256=_digest("fusion-prompt-v2"),
        output_schema=provider_schema,
        enriched_output_schema=enriched_schema,
        generation_config={"temperature": 0},
        timeout_ms=1_000,
        selection_policy_version="select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="offline-data-v1",
    )
    coarse_qa_policy = InferencePolicy(
        policy_version="offline-coarse-qa-model-policy-v1",
        task=VisionTask.QA_COARSE,
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="offline-adapter-v1",
        prompt_version="coarse-qa-prompt-v1",
        prompt_artifact_id=_uuid(1_002),
        prompt_sha256=_digest("coarse-qa-prompt-v1"),
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
        prompt_artifact_id=_uuid(1_003),
        prompt_sha256=_digest("dense-qa-prompt-v1"),
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
        prompt_artifact_id=_uuid(1_004),
        prompt_sha256=_digest("event-proposal-prompt-v1"),
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
        prompt_artifact_id=_uuid(1_005),
        prompt_sha256=_digest("action-evidence-prompt-v1"),
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
        prompt_artifact_id=_uuid(1_006),
        prompt_sha256=_digest("boundary-refinement-prompt-v1"),
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
    barrier_storage = SQLiteBarrierStorage(barrier_path) if barrier_path is not None else None
    raw_store: RawProviderBytesStore = (
        inference_evidence if inference_evidence is not None else InMemoryRawProviderBytesStore()
    )
    parser = StrictProviderClaimParser(registry, parser_version=PARSER_VERSION)
    fixture_adapter_class = (
        _RawReferenceMismatchAdapter
        if mismatch_raw_reference and not protocol_only_adapter
        else OfflineFixtureVisionAdapter
    )

    def staged_response_factory(
        request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        if request.task is VisionTask.QA_COARSE:
            factory = coarse_response_factory or _coarse_claim_bytes
            return factory(request)
        if request.task is VisionTask.QA_DENSE:
            factory = dense_response_factory or _coarse_claim_bytes
            return factory(request)
        if request.task is VisionTask.EVENT_PROPOSAL:
            factory = event_proposal_response_factory or _event_proposal_claim_bytes
            return factory(request)
        if request.task is VisionTask.ACTION_EVIDENCE:
            factory = action_evidence_response_factory or _action_evidence_claim_bytes
            return factory(request)
        if request.task is VisionTask.BOUNDARY_REFINEMENT:
            factory = boundary_refinement_response_factory or _boundary_refinement_claim_bytes
            return factory(request)
        return response_factory(request)

    fixture_adapter = fixture_adapter_class(
        capabilities=_capabilities(max_images_per_request=max_images_per_request),
        raw_store=raw_store,
        parser=parser,
        response_factory=staged_response_factory,
    )
    protocol_adapter = (
        _ProtocolOnlyVisionAdapter(
            fixture_adapter,
            mismatch_raw_reference=mismatch_raw_reference,
        )
        if protocol_only_adapter
        else None
    )
    adapter = protocol_adapter or fixture_adapter
    repository = repository or InMemoryEventIdentityRegistryRepository()
    input_preparer = InputPlanPreparer(
        InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION),
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
        raw_store=raw_store,
        parser=parser,
        coarse_qa_policy=coarse_qa_policy,
        dense_qa_policy=dense_qa_policy,
        event_proposal_policy=event_proposal_policy,
        action_evidence_policy=action_evidence_policy,
        boundary_refinement_policy=boundary_refinement_policy,
        inference_policy=inference_policy,
        schema_registry=registry,
        logical_node_registry=logical_node_registry,
        execution_policy=execution_policy,
        inference_ledger=inference_evidence,
        evidence_store=inference_evidence,
        barrier_storage=barrier_storage,
        call_barrier_storage=barrier_storage,
        clock=clock if clock is not None else lambda: NOW,
    )
    context = _v2_context()
    plan = _sampling_plan()
    frame_index = frame_index_override or _v2_frame_index(
        context,
        PackageLineage(
            source_content_sha256=context.source_content_sha256,
            window_semantic_sha256=_digest("placeholder-window"),
            camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
            alignment_semantic_sha256=context.alignment_semantic_sha256,
            sampling_plan_sha256=sampling_plan_digest(
                plan,
                purpose=SamplingPurpose.QA_COARSE,
            ),
        ),
        empty_camera=None,
    )
    return _Harness(
        pipeline=pipeline,
        fixture_adapter=fixture_adapter,
        repository=repository,
        raw_store=raw_store,
        inference_evidence=inference_evidence,
        barrier_storage=barrier_storage,
        context=context,
        frame_index=frame_index,
        execution_policy=execution_policy,
        logical_node_registry=logical_node_registry,
        protocol_adapter=protocol_adapter,
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
    assert result.adapter_infer_calls >= 0
    assert harness.fixture_adapter.network_call_count == 0


def _boundary_dispatch_count(result: CanonicalOfflineRunResult) -> int:
    """Count the exact retained ONSET/OFFSET call parts."""

    return sum(
        len(execution.onset.part_results) + len(execution.offset.part_results)
        for execution in result.boundary_refinement_executions
    )


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


def test_success_connects_raw_claim_enrichment_and_local_hypothesis(
    tmp_path: Path,
) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        protocol_only_adapter=True,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert harness.protocol_adapter is not None
    assert harness.pipeline.adapter is harness.protocol_adapter
    assert not isinstance(harness.pipeline.adapter, OfflineFixtureVisionAdapter)
    assert result.package_set is not None
    assert result.coarse_qa_result is not None
    assert result.coarse_qa_result.complete
    assert not result.coarse_qa_result.production_eligible
    assert result.qa_completion_result is not None
    assert result.qa_completion_result.status is QACompletionStatus.QA_COMPLETE
    assert (
        result.qa_completion_result.dense_work_manifest.outcome is DenseQAOutcome.SKIPPED_NOT_NEEDED
    )
    assert result.qa_completion_result.dense_work_manifest.items == ()
    assert result.qa_completion_result.final_aggregate is not None
    assert (
        tuple(item.camera_id for item in result.qa_completion_result.final_aggregate.camera_results)
        == CAMERA_IDS
    )
    assert result.qa_completion_result.production_eligible is False
    assert len(result.action_evidence_executions) == 1
    action_execution = result.action_evidence_executions[0]
    assert action_execution.window.purpose is SamplingPurpose.ACTION_DENSE
    assert action_execution.window.source_subject_type == "CANDIDATE_EVENT"
    assert (
        action_execution.window.source_subject_logical_key
        == action_execution.candidate.candidate_logical_key
    )
    assert action_execution.window.context_truncated
    assert action_execution.window.requested_interval == NanosecondInterval(
        start_ns=-300_000_000,
        end_ns=1_300_000_000,
    )
    assert action_execution.window.interval == REQUESTED_INTERVAL
    assert action_execution.evidence_result.outcome.value == "SUPPORTED"
    assert (
        tuple(
            camera_id for camera_id, _ in action_execution.evidence_result.camera_evidence.items()
        )
        == CAMERA_IDS
    )
    assert all(
        action_execution.evidence_result.camera_evidence[camera_id].observations
        for camera_id in CAMERA_IDS
    )
    assert action_execution.production_eligible is False
    assert action_execution.candidate.candidate_logical_key.startswith(
        f"{CANDIDATE_EVENT_LOGICAL_KEY_NAMESPACE}:"
    )
    assert result.candidate_reduction_result is not None
    assert result.candidate_reduction_result.logical_key.startswith(
        f"{CANDIDATE_REDUCTION_LOGICAL_KEY_NAMESPACE}:"
    )
    assert result.provisional_fusion_result is not None
    assert not result.provisional_fusion_result.production_eligible
    assert len(result.provisional_fusion_result.actions) == 1
    provisional_action = result.provisional_fusion_result.actions[0]
    assert provisional_action.ordinal == 0
    assert provisional_action.label == "grasp"
    assert not provisional_action.production_eligible
    assert tuple(provisional_action.camera_evidence.keys()) == CAMERA_IDS
    assert tuple(
        item.action_evidence_logical_key for item in provisional_action.source_candidates
    ) == (action_execution.evidence_result.logical_key,)

    assert len(result.boundary_refinement_executions) == 1
    boundary_execution = result.boundary_refinement_executions[0]
    assert boundary_execution.action == provisional_action
    assert boundary_execution.result.source_action_logical_key == provisional_action.logical_key
    assert boundary_execution.result.outcome.value == "REFINED"
    assert boundary_execution.result.refined_interval is not None
    assert boundary_execution.result.onset == boundary_execution.onset.role_result
    assert boundary_execution.result.offset == boundary_execution.offset.role_result
    assert boundary_execution.result.used_fallback is False
    assert boundary_execution.production_eligible is False
    for expected_role, pass_execution in (
        ("ONSET", boundary_execution.onset),
        ("OFFSET", boundary_execution.offset),
    ):
        assert pass_execution.window.purpose is SamplingPurpose.BOUNDARY_REFINEMENT
        assert pass_execution.window.refinement_role.value == expected_role
        assert pass_execution.window.source_subject_type == "PROVISIONAL_PHYSICAL_ACTION"
        assert pass_execution.window.source_subject_logical_key == provisional_action.logical_key
        assert pass_execution.input_plan.subject.task is VisionTask.BOUNDARY_REFINEMENT
        assert len(pass_execution.part_results) == len(pass_execution.input_plan.call_plan.parts)
        assert all(
            item.status is CanonicalOfflinePartStatus.ENRICHED
            for item in pass_execution.part_results
        )
        assert pass_execution.role_result.role.value == expected_role
        assert pass_execution.role_result.outcome.value == "REFINED"
        assert pass_execution.role_result.observed_camera_count == len(CAMERA_IDS)
        assert tuple(pass_execution.role_result.camera_evidence.keys()) == CAMERA_IDS
        assert pass_execution.role_result.used_fallback is False
        assert pass_execution.production_eligible is False

    candidate_digest = action_execution.candidate.candidate_logical_key.rsplit(":", 1)[-1]
    stale_candidate = action_execution.candidate.model_dump(mode="python")
    stale_candidate["candidate_logical_key"] = f"candidate-event:{candidate_digest}"
    stale_candidate["candidate_event_id"] = str(
        uuid5(NAMESPACE_URL, f"robata:candidate-event:{candidate_digest}")
    )
    with pytest.raises(ValidationError, match="candidate identity"):
        CanonicalCandidateEvent.model_validate(stale_candidate, strict=True)

    stale_reduction = result.candidate_reduction_result.model_dump(mode="python")
    stale_reduction["logical_key"] = (
        f"candidate-reduction:{result.candidate_reduction_result.semantic_sha256}"
    )
    with pytest.raises(ValidationError, match="candidate reduction identity"):
        CandidateReductionResult.model_validate(stale_reduction, strict=True)

    forged_execution = action_execution.model_dump(mode="python")
    forged_inference_id = _uuid(99_998)
    forged_execution["evidence_result"]["source_outputs"][0]["source_inference_id"] = (
        forged_inference_id
    )
    for camera_id in CAMERA_IDS:
        for observation in forged_execution["evidence_result"]["camera_evidence"][camera_id.value][
            "observations"
        ]:
            if observation["source"]["output"]["part_ordinal"] == 0:
                observation["source"]["output"]["source_inference_id"] = forged_inference_id
    with pytest.raises(
        ValidationError,
        match="normalized action evidence differs from retained enriched claims",
    ):
        CanonicalActionEvidenceExecution.model_validate(forged_execution, strict=True)

    assert result.input_plan is not None
    target = result.input_plan.target
    request_catalog = result.input_plan.request_catalog
    request_uuid_parts = (
        result.package_set.lineage,
        VisionTask.FUSION_ADJUDICATION.value,
        "offline-model-policy-v2",
        _digest("fusion-prompt-v2"),
        harness.execution_policy.semantic_sha256,
        target.capability_snapshot_sha256,
    )
    input_plan_uuid_parts = (
        request_catalog.semantic_sha256,
        target.capability_snapshot_sha256,
        result.input_plan.prompt_output.rendered_message_sha256,
        harness.execution_policy.semantic_sha256,
    )
    assert request_catalog.request_catalog_id == _stable_uuid(
        REQUEST_CATALOG_UUID_NAMESPACE,
        *request_uuid_parts,
    )
    assert request_catalog.request_catalog_id != _stable_uuid(
        "provider-request-catalog",
        *request_uuid_parts,
    )
    assert result.input_plan.input_plan_id == _stable_uuid(
        INPUT_PLAN_UUID_NAMESPACE,
        *input_plan_uuid_parts,
    )
    assert result.input_plan.input_plan_id != _stable_uuid(
        "inference-input-plan",
        *input_plan_uuid_parts,
    )
    assert result.attempt_count == 1
    expected_dispatches = (
        len(result.coarse_qa_result.source_outputs)
        + sum(len(item.part_results) for item in result.action_evidence_executions)
        + _boundary_dispatch_count(result)
        + 2 * len(result.part_results)
    )
    assert result.adapter_infer_calls == expected_dispatches
    assert harness.protocol_adapter.dispatch_calls == expected_dispatches
    assert result.terminal is not None
    assert result.selection is not None
    assert result.barrier_reduction is not None
    assert result.raw_response is not None
    assert result.parsed_claims is not None
    assert result.selected_output is not None
    assert result.enriched_output is not None
    assert result.output_decision is not None
    assert result.output_decision.decision == "ADMITTED"
    assert result.output_decision.evidence_class is AdmissionEvidenceClass.LOCAL_CONFORMANCE
    assert result.output_decision.production_eligible is False
    assert b"PRODUCTION_ADMITTED" not in canonical_json_bytes(result.model_dump(mode="json"))
    assert len(result.hypotheses) == 1
    assert result.identity_result is None
    assert result.hypotheses[0].production_output_admission == (
        result.output_decision.production_output_admission
    )
    snapshot = harness.repository.snapshot(harness.context.recording_identity)
    assert snapshot.generation == 0
    assert snapshot.identities == ()
    assert snapshot.assignments == ()
    assert harness.repository.list_outbox(harness.context.recording_identity) == ()
    assert len(harness.raw_store.list_records()) == expected_dispatches
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
    membership_roles = tuple(item.role for item in result.run_memberships)
    assert "BOUNDARY_ONSET_WINDOW" in membership_roles
    assert "BOUNDARY_ONSET_RESULT" in membership_roles
    assert "BOUNDARY_OFFSET_WINDOW" in membership_roles
    assert "BOUNDARY_OFFSET_RESULT" in membership_roles
    assert "BOUNDARY_REFINEMENT_RESULT" in membership_roles
    assert tuple(item.node_type for item in result.run_memberships[-3:]) == (
        "FUSION_REDUCTION",
        "OUTPUT_ADMISSION_DECISION",
        "EVENT_HYPOTHESIS",
    )
    assert all(item.disposition is RunNodeDisposition.CREATED for item in result.run_memberships)
    assert {
        item.identity for item in harness.logical_node_registry.list_run_memberships(result.run_id)
    } == {item.identity for item in result.run_memberships}
    _assert_offline(result, harness)


def test_zero_event_proposals_complete_without_fusion_and_replay_without_dispatch(
    tmp_path: Path,
) -> None:
    fusion_calls = 0

    def fusion_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        raise AssertionError("zero proposals must stop before fusion")

    harness = _harness(
        fusion_must_not_run,
        logical_registry_root=tmp_path,
        event_proposal_response_factory=lambda _request: canonical_json_bytes(
            {"claims": [], "abstained": False}
        ),
        protocol_only_adapter=True,
    )
    processing_run = _processing_run(harness, run_id=_uuid(21_101))

    first = _run(harness, processing_run=processing_run)
    first_raw = harness.raw_store.list_records()
    replay = _run(harness, processing_run=processing_run)

    assert first.status is replay.status is CanonicalOfflineRunStatus.NO_EVENTS
    assert first.event_proposal_result is not None
    assert first.event_proposal_result.outcome is EventProposalOutcome.NO_EVENTS
    assert first.event_proposal_result.proposals == ()
    assert not first.event_proposal_result.production_eligible
    assert first.candidate_reduction_result is not None
    assert first.candidate_reduction_result.no_events
    assert not first.candidate_reduction_result.production_eligible
    assert first.input_plan is None
    assert first.part_results == ()
    assert first.fusion_reduction is None
    assert first.output_decision is None
    assert first.hypotheses == ()
    assert replay.event_proposal_result == first.event_proposal_result
    assert replay.candidate_reduction_result == first.candidate_reduction_result
    assert replay.run_memberships == first.run_memberships
    assert first.adapter_infer_calls == 2
    assert replay.adapter_infer_calls == 0
    assert harness.protocol_adapter is not None
    assert harness.protocol_adapter.dispatch_calls == 2
    assert harness.raw_store.list_records() == first_raw
    assert len(first_raw) == 2
    assert fusion_calls == 0
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    assert harness.repository.list_outbox(harness.context.recording_identity) == ()
    _assert_offline(first, harness)
    _assert_offline(replay, harness)


def test_event_proposal_abstention_is_not_reported_as_no_events(tmp_path: Path) -> None:
    fusion_calls = 0

    def fusion_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        raise AssertionError("proposal abstention must stop before fusion")

    harness = _harness(
        fusion_must_not_run,
        logical_registry_root=tmp_path,
        event_proposal_response_factory=lambda _request: canonical_json_bytes(
            {"claims": [], "abstained": True}
        ),
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INVALID_OUTPUT
    assert result.error is not None
    assert result.error.code == "SELECTED_RAW_OUTPUT_INVALID"
    assert result.event_proposal_result is None
    assert result.candidate_reduction_result is None
    assert result.input_plan is None
    assert result.fusion_reduction is None
    assert result.output_decision is None
    assert result.hypotheses == ()
    assert fusion_calls == 0
    _assert_offline(result, harness)


def test_event_proposal_permanent_failure_stops_before_projection_and_fusion(
    tmp_path: Path,
) -> None:
    fusion_calls = 0

    def fusion_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        raise AssertionError("failed proposal stage must stop before fusion")

    harness = _harness(
        fusion_must_not_run,
        logical_registry_root=tmp_path,
        event_proposal_response_factory=lambda request: _failure(
            request,
            retryability=Retryability.PERMANENT,
            status=InferenceStatus.FAILED,
        ),
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INFERENCE_FAILED
    assert result.error is not None
    assert result.error.code == "EVENT_PROPOSAL_REQUIRED_PARTS_INCOMPLETE"
    assert result.event_proposal_result is None
    assert result.candidate_reduction_result is None
    assert result.input_plan is None
    assert result.fusion_reduction is None
    assert result.output_decision is None
    assert result.hypotheses == ()
    assert result.identity_result is None
    assert fusion_calls == 0
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    assert harness.repository.list_outbox(harness.context.recording_identity) == ()
    _assert_offline(result, harness)


def test_action_evidence_all_no_event_stops_before_fusion(
    tmp_path: Path,
) -> None:
    fusion_calls = 0

    def fusion_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        raise AssertionError("negative action evidence must stop before fusion")

    harness = _harness(
        fusion_must_not_run,
        logical_registry_root=tmp_path,
        action_evidence_response_factory=lambda request: _action_evidence_claim_bytes(
            request,
            observation=ProviderObservation.NO_EVENT,
        ),
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.NO_EVENTS
    assert result.error is None
    assert len(result.action_evidence_executions) == 1
    assert result.action_evidence_executions[0].evidence_result.outcome.value == "NO_ACTION"
    assert result.input_plan is None
    assert result.part_results == ()
    assert result.fusion_reduction is None
    assert result.output_decision is None
    assert result.hypotheses == ()
    assert result.adapter_infer_calls == 3
    assert fusion_calls == 0
    _assert_offline(result, harness)


def test_sparse_action_camera_cannot_be_reported_as_no_events(
    tmp_path: Path,
) -> None:
    context = _v2_context()
    plan = _sampling_plan()
    full_index = _v2_frame_index(
        context,
        PackageLineage(
            source_content_sha256=context.source_content_sha256,
            window_semantic_sha256=_digest("placeholder-window"),
            camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
            alignment_semantic_sha256=context.alignment_semantic_sha256,
            sampling_plan_sha256=sampling_plan_digest(
                plan,
                purpose=SamplingPurpose.QA_COARSE,
            ),
        ),
        empty_camera=None,
    )
    sparse_index_payload = full_index.model_dump(mode="python")
    cam_06_frames = sparse_index_payload["cameras"][CameraId.CAM_06.value]["frames"]
    sparse_index_payload["cameras"][CameraId.CAM_06.value]["frames"] = (cam_06_frames[0],)
    sparse_index = CanonicalSixCameraFrameIndex.model_validate(
        sparse_index_payload,
        strict=True,
    )

    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        event_proposal_response_factory=lambda request: _event_proposal_claim_bytes(
            request,
            intervals=((800_000_000, 900_000_000),),
        ),
        action_evidence_response_factory=lambda request: _action_evidence_claim_bytes(
            request,
            observation=ProviderObservation.NO_EVENT,
        ),
        frame_index_override=sparse_index,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INCOMPLETE
    assert result.error is not None
    assert result.error.code == "ACTION_EVIDENCE_INDETERMINATE"
    assert len(result.action_evidence_executions) == 1
    evidence = result.action_evidence_executions[0].evidence_result
    assert evidence.outcome.value == "INDETERMINATE"
    assert evidence.camera_evidence[CameraId.CAM_06].observations == ()
    assert evidence.camera_evidence[CameraId.CAM_06].outcome.value == "INDETERMINATE"
    assert result.input_plan is None
    assert result.fusion_reduction is None
    assert result.output_decision is None
    assert result.hypotheses == ()
    _assert_offline(result, harness)


def test_action_evidence_permanent_failure_stops_before_fusion(
    tmp_path: Path,
) -> None:
    fusion_calls = 0

    def fusion_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        raise AssertionError("failed action evidence must stop before fusion")

    harness = _harness(
        fusion_must_not_run,
        logical_registry_root=tmp_path,
        action_evidence_response_factory=lambda request: _failure(
            request,
            retryability=Retryability.PERMANENT,
            status=InferenceStatus.FAILED,
        ),
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INFERENCE_FAILED
    assert result.error is not None
    assert result.error.code == "ACTION_EVIDENCE_REQUIRED_PARTS_INCOMPLETE"
    assert result.action_evidence_executions == ()
    assert result.input_plan is None
    assert result.fusion_reduction is None
    assert result.output_decision is None
    assert result.hypotheses == ()
    assert result.adapter_infer_calls == 3
    assert fusion_calls == 0
    _assert_offline(result, harness)


def test_boundary_offset_failure_stops_without_fake_refined_result(
    tmp_path: Path,
) -> None:
    fusion_calls = 0
    boundary_roles: list[str] = []

    def fusion_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        raise AssertionError("failed boundary refinement must stop before final fusion")

    def boundary_response(
        request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        role = request.metadata["boundary_refinement_role"]
        boundary_roles.append(role)
        if role == "OFFSET":
            return _failure(
                request,
                retryability=Retryability.PERMANENT,
                status=InferenceStatus.FAILED,
            )
        return _boundary_refinement_claim_bytes(request)

    harness = _harness(
        fusion_must_not_run,
        logical_registry_root=tmp_path,
        boundary_refinement_response_factory=boundary_response,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INFERENCE_FAILED
    assert result.error is not None
    assert result.error.code == "BOUNDARY_REFINEMENT_OFFSET_REQUIRED_PARTS_INCOMPLETE"
    assert result.provisional_fusion_result is not None
    assert len(result.provisional_fusion_result.actions) == 1
    assert result.boundary_refinement_executions == ()
    assert boundary_roles == ["ONSET", "OFFSET"]
    assert result.input_plan is None
    assert result.part_results == ()
    assert result.output_decision is None
    assert result.hypotheses == ()
    assert fusion_calls == 0
    _assert_offline(result, harness)


def test_candidate_reduction_merges_nested_connected_proposal_intervals(
    tmp_path: Path,
) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        event_proposal_response_factory=lambda request: _event_proposal_claim_bytes(
            request,
            intervals=(
                (100_000_000, 900_000_000),
                (200_000_000, 300_000_000),
                (800_000_000, 950_000_000),
            ),
        ),
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert result.event_proposal_result is not None
    assert len(result.event_proposal_result.proposals) == 3
    assert result.candidate_reduction_result is not None
    assert len(result.candidate_reduction_result.candidates) == 1
    candidate = result.candidate_reduction_result.candidates[0]
    assert candidate.effective_interval == NanosecondInterval(
        start_ns=100_000_000,
        end_ns=950_000_000,
    )
    assert len(candidate.source_proposal_logical_keys) == 3
    assert not candidate.production_eligible
    _assert_offline(result, harness)


def test_multiple_candidates_complete_one_final_fusion_path(
    tmp_path: Path,
) -> None:
    fusion_calls = 0

    def fusion_response(request: VisionInferenceRequest) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        return _claim_bytes(request)

    action_labels_by_plan: dict[str, str] = {}

    def action_evidence_response(request: VisionInferenceRequest) -> bytes:
        assert request.input_plan is not None
        plan_key = request.input_plan.input_plan_id
        if plan_key not in action_labels_by_plan:
            action_labels_by_plan[plan_key] = ("grasp", "place")[len(action_labels_by_plan)]
        return _action_evidence_claim_bytes(
            request,
            label=action_labels_by_plan[plan_key],
        )

    harness = _harness(
        fusion_response,
        logical_registry_root=tmp_path,
        event_proposal_response_factory=lambda request: _event_proposal_claim_bytes(
            request,
            intervals=(
                (100_000_000, 400_000_000),
                (400_000_000, 900_000_000),
            ),
            labels=("grasp", "place"),
        ),
        action_evidence_response_factory=action_evidence_response,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert result.error is None
    assert result.candidate_reduction_result is not None
    assert len(result.candidate_reduction_result.candidates) == 2
    assert (
        tuple(item.candidate for item in result.action_evidence_executions)
        == result.candidate_reduction_result.candidates
    )
    assert len(result.action_evidence_executions) == 2
    assert len({item.window.window_logical_key for item in result.action_evidence_executions}) == 2
    assert all(
        item.evidence_result.outcome.value == "SUPPORTED"
        for item in result.action_evidence_executions
    )
    assert result.provisional_fusion_result is not None
    assert tuple(
        (item.ordinal, item.label) for item in result.provisional_fusion_result.actions
    ) == ((0, "grasp"), (1, "place"))
    assert all(not item.production_eligible for item in result.provisional_fusion_result.actions)
    assert (
        tuple(item.action for item in result.boundary_refinement_executions)
        == result.provisional_fusion_result.actions
    )
    assert len(result.boundary_refinement_executions) == 2
    assert all(
        item.result.outcome.value == "REFINED"
        and item.onset.role_result.outcome.value == "REFINED"
        and item.offset.role_result.outcome.value == "REFINED"
        and not item.production_eligible
        for item in result.boundary_refinement_executions
    )
    assert result.final_fusion_context is not None
    assert tuple(item.label for item in result.final_fusion_context.actions) == (
        "grasp",
        "place",
    )
    assert tuple(item.refined_interval for item in result.final_fusion_context.actions) == tuple(
        item.result.refined_interval for item in result.boundary_refinement_executions
    )
    assert result.input_plan is not None
    assert fusion_calls == len(result.part_results)
    assert result.fusion_reduction is not None
    assert result.fusion_reduction.outcome == "CLAIMS"
    assert tuple(
        sorted(
            (
                item.representative.interval.start_ns,
                item.representative.interval.end_ns,
                item.representative.label,
            )
            for item in result.fusion_reduction.claims
            if item.representative.interval is not None
        )
    ) == tuple(
        sorted(
            (
                item.refined_interval.start_ns,
                item.refined_interval.end_ns,
                item.label,
            )
            for item in result.final_fusion_context.actions
        )
    )
    assert len(result.hypotheses) == 2
    assert result.output_decision is not None
    assert result.output_decision.production_eligible is False
    assert all(not item.production_admission.production_eligible for item in result.hypotheses)
    fuser = ProvisionalPhysicalActionFuser(result.provisional_fusion_result.policy)
    reversed_evidence = tuple(
        item.evidence_result for item in reversed(result.action_evidence_executions)
    )
    assert (
        fuser.fuse(result.candidate_reduction_result, reversed_evidence)
        == result.provisional_fusion_result
    )
    with pytest.raises(ProvisionalFusionError, match="exactly cover"):
        fuser.fuse(
            result.candidate_reduction_result,
            reversed_evidence[:-1],
        )
    _assert_offline(result, harness)


def test_final_fusion_explicit_empty_result_is_no_events(tmp_path: Path) -> None:
    harness = _harness(
        lambda request: _claim_bytes(request, action_limit=0),
        logical_registry_root=tmp_path,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.NO_EVENTS
    assert result.error is None
    assert result.final_fusion_context is not None
    assert len(result.final_fusion_context.actions) == 1
    assert result.fusion_reduction is not None
    assert result.fusion_reduction.outcome == "NO_SURVIVING_EVENTS"
    assert result.output_decision is not None
    assert result.output_decision.decision == "NO_EVENTS"
    assert result.output_decision.production_eligible is False
    assert result.hypotheses == ()
    assert result.identity_result is None
    _assert_offline(result, harness)


def test_final_fusion_missing_refined_action_fails_closed(tmp_path: Path) -> None:
    action_labels_by_plan: dict[str, str] = {}

    def action_evidence_response(request: VisionInferenceRequest) -> bytes:
        assert request.input_plan is not None
        plan_key = request.input_plan.input_plan_id
        if plan_key not in action_labels_by_plan:
            action_labels_by_plan[plan_key] = ("grasp", "place")[len(action_labels_by_plan)]
        return _action_evidence_claim_bytes(
            request,
            label=action_labels_by_plan[plan_key],
        )

    harness = _harness(
        lambda request: _claim_bytes(request, action_limit=1),
        logical_registry_root=tmp_path,
        event_proposal_response_factory=lambda request: _event_proposal_claim_bytes(
            request,
            intervals=(
                (100_000_000, 400_000_000),
                (400_000_000, 900_000_000),
            ),
            labels=("grasp", "place"),
        ),
        action_evidence_response_factory=action_evidence_response,
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INVALID_OUTPUT
    assert result.error is not None
    assert result.error.code == "FINAL_FUSION_CLOSURE_REJECTED"
    assert result.final_fusion_context is not None
    assert len(result.final_fusion_context.actions) == 2
    assert result.fusion_reduction is not None
    assert len(result.fusion_reduction.claims) == 1
    assert result.output_decision is None
    assert result.hypotheses == ()
    assert result.identity_result is None
    _assert_offline(result, harness)


def test_degraded_coarse_qa_executes_dense_then_continues_to_fusion(
    tmp_path: Path,
) -> None:
    fusion_calls = 0

    def fusion_response(request: VisionInferenceRequest) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        return _claim_bytes(request)

    harness = _harness(
        fusion_response,
        logical_registry_root=tmp_path,
        coarse_response_factory=lambda request: _coarse_claim_bytes(
            request,
            observation=ProviderObservation.DEGRADED,
        ),
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert result.error is None
    assert result.coarse_qa_result is not None
    assert result.coarse_qa_result.requires_dense
    assert not result.coarse_qa_result.complete
    assert not result.coarse_qa_result.production_eligible
    assert result.qa_completion_result is not None
    assert result.qa_completion_result.status is QACompletionStatus.QA_COMPLETE
    assert result.qa_completion_result.dense_work_manifest.outcome is DenseQAOutcome.DENSE_REQUIRED
    assert result.qa_completion_result.dense_result is not None
    assert result.qa_completion_result.dense_result.status is DenseQAStatus.COMPLETE
    assert result.qa_completion_result.final_aggregate is not None
    assert result.qa_completion_result.dense_work_manifest.items
    assert all(
        item.context_camera_ids == CAMERA_IDS
        for item in result.qa_completion_result.dense_work_manifest.items
    )
    assert all(
        item.source.source_output in result.coarse_qa_result.source_outputs
        for item in result.qa_completion_result.dense_work_manifest.items
    )
    assert len(result.dense_qa_executions) == len(
        result.qa_completion_result.dense_work_manifest.units
    )
    dense_execution = result.dense_qa_executions[0]
    assert dense_execution.window.purpose is SamplingPurpose.QA_DENSE
    assert dense_execution.unit_evidence.production_eligible is False
    assert all(
        item.local_status.value == "GOOD"
        for item in dense_execution.unit_evidence.package_camera_results
    )
    assert result.input_plan is not None
    assert result.part_results
    assert result.adapter_infer_calls > len(result.coarse_qa_result.source_outputs)
    assert fusion_calls == len(result.part_results)
    assert result.barrier_reduction is not None
    assert result.fusion_reduction is not None
    assert result.output_decision is not None
    assert result.hypotheses
    assert result.identity_result is None
    roles = tuple(item.role for item in result.run_memberships)
    assert roles.index("COARSE_QA") < roles.index("DENSE_QA_WINDOW")
    assert roles.index("DENSE_QA_CALL_REDUCTION") < roles.index("DENSE_QA_RESULT")
    assert roles.index("DENSE_QA_RESULT") < roles.index("QA_COMPLETION")
    assert roles.index("QA_COMPLETION") < roles.index("INPUT_PLAN")
    _assert_offline(result, harness)


def test_required_dense_terminal_failure_blocks_fusion(tmp_path: Path) -> None:
    fusion_calls = 0

    def fusion_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        raise AssertionError("failed required dense QA must block fusion")

    harness = _harness(
        fusion_must_not_run,
        logical_registry_root=tmp_path,
        coarse_response_factory=lambda request: _coarse_claim_bytes(
            request,
            observation=ProviderObservation.DEGRADED,
        ),
        dense_response_factory=lambda request: _failure(
            request,
            retryability=Retryability.NON_RETRYABLE,
            status=InferenceStatus.FAILED,
        ),
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INCOMPLETE
    assert result.error is not None
    assert result.error.code == "QA_DENSE_REQUIRED_PARTS_INCOMPLETE"
    assert result.qa_completion_result is not None
    assert result.qa_completion_result.status is QACompletionStatus.QA_INCOMPLETE
    assert result.qa_completion_result.dense_result is None
    assert result.qa_completion_result.dense_failure_code == "QA_DENSE_REQUIRED_PARTS_INCOMPLETE"
    assert result.qa_completion_result.dense_work_manifest.units
    assert result.dense_qa_executions == ()
    assert result.input_plan is None
    assert result.part_results == ()
    assert fusion_calls == 0
    assert result.run_memberships[-1].role == "QA_COMPLETION"
    _assert_offline(result, harness)


def test_unknown_coarse_qa_persists_incomplete_gate_without_dense_or_fusion(
    tmp_path: Path,
) -> None:
    fusion_calls = 0

    def fusion_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        raise AssertionError("incomplete QA must stop before fusion")

    harness = _harness(
        fusion_must_not_run,
        logical_registry_root=tmp_path,
        coarse_response_factory=lambda request: _coarse_claim_bytes(
            request,
            observation=ProviderObservation.UNKNOWN,
        ),
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INCOMPLETE
    assert result.error is not None and result.error.code == "QA_INCOMPLETE"
    assert result.qa_completion_result is not None
    assert result.qa_completion_result.status is QACompletionStatus.QA_INCOMPLETE
    assert (
        result.qa_completion_result.dense_work_manifest.outcome is DenseQAOutcome.BLOCKED_INCOMPLETE
    )
    assert result.qa_completion_result.dense_work_manifest.items == ()
    assert result.qa_completion_result.final_aggregate is None
    assert result.input_plan is None
    assert fusion_calls == 0
    assert result.coarse_qa_result is not None
    assert result.adapter_infer_calls == len(result.coarse_qa_result.source_outputs)
    assert result.run_memberships[-1].node_type == "QA_COMPLETION_RESULT"


def test_coarse_dependency_changes_fusion_plan_and_provider_idempotency(
    tmp_path: Path,
) -> None:
    first_harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "first",
        coarse_response_factory=lambda request: _coarse_claim_bytes(
            request,
            model_reported_score=0.25,
        ),
    )
    changed_harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path / "changed",
        coarse_response_factory=lambda request: _coarse_claim_bytes(
            request,
            model_reported_score=0.75,
        ),
    )

    first = _run(first_harness)
    changed = _run(changed_harness)

    assert first.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert changed.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.coarse_qa_result != changed.coarse_qa_result
    assert first.input_plan is not None
    assert changed.input_plan is not None
    assert first.input_plan.prompt_output.rendered_message_sha256 != (
        changed.input_plan.prompt_output.rendered_message_sha256
    )
    assert first.input_plan.semantic_sha256 != changed.input_plan.semantic_sha256
    assert first.input_plan.call_plan.barrier_semantic_sha256 != (
        changed.input_plan.call_plan.barrier_semantic_sha256
    )
    assert tuple(item.idempotency_key for item in first.input_plan.call_plan.parts) != tuple(
        item.idempotency_key for item in changed.input_plan.call_plan.parts
    )
    assert tuple(item.terminal.logical_invocation_id for item in first.part_results) != tuple(
        item.terminal.logical_invocation_id for item in changed.part_results
    )
    assert tuple(item.terminal.provider_idempotency_key for item in first.part_results) != tuple(
        item.terminal.provider_idempotency_key for item in changed.part_results
    )


def test_coarse_terminal_failure_returns_structured_inference_failure(
    tmp_path: Path,
) -> None:
    fusion_calls = 0

    def fusion_must_not_run(_request: VisionInferenceRequest) -> OfflineFixtureResponse:
        nonlocal fusion_calls
        fusion_calls += 1
        raise AssertionError("failed coarse QA must stop before fusion")

    harness = _harness(
        fusion_must_not_run,
        logical_registry_root=tmp_path,
        coarse_response_factory=lambda request: _failure(
            request,
            retryability=Retryability.PERMANENT,
            status=InferenceStatus.FAILED,
        ),
    )

    result = _run(harness)

    assert result.status is CanonicalOfflineRunStatus.INFERENCE_FAILED
    assert result.error is not None
    assert result.error.code == "QA_COARSE_REQUIRED_PARTS_INCOMPLETE"
    assert result.coarse_qa_result is None
    assert result.input_plan is None
    assert result.part_results == ()
    assert result.adapter_infer_calls == 1
    assert fusion_calls == 0
    assert tuple(item.task for item in harness.pipeline.ledger.list_intents()) == (
        VisionTask.QA_COARSE,
    )
    assert tuple(item.node_type for item in result.run_memberships) == (
        "TEMPORAL_WINDOW",
        "TEMPORAL_PACKAGE_SET",
    )
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
        protocol_only_adapter=True,
    )
    processing_run = _processing_run(harness, run_id=_uuid(21_000))
    first = _run(harness, processing_run=processing_run)
    assert first.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.input_plan is not None
    assert first.identity_result is None
    part_count = len(first.input_plan.call_plan.parts)
    assert part_count > 1
    assert len(first.part_results) == part_count
    assert first.coarse_qa_result is not None
    coarse_part_count = len(first.coarse_qa_result.source_outputs)
    assert coarse_part_count == part_count
    first_snapshot = harness.repository.snapshot(harness.context.recording_identity)
    first_outbox = harness.repository.list_outbox(harness.context.recording_identity)
    first_raw = harness.raw_store.list_records()

    replay = _run(harness, processing_run=processing_run)

    assert replay.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert replay.run_id == first.run_id
    assert replay.processing_run == first.processing_run
    assert replay.run_memberships == first.run_memberships
    assert replay.qa_completion_result == first.qa_completion_result
    assert replay.event_proposal_result == first.event_proposal_result
    assert replay.candidate_reduction_result == first.candidate_reduction_result
    assert replay.action_evidence_executions == first.action_evidence_executions
    assert replay.provisional_fusion_result == first.provisional_fusion_result
    assert replay.boundary_refinement_executions == first.boundary_refinement_executions
    assert replay.final_fusion_context == first.final_fusion_context
    assert replay.part_results == first.part_results
    assert replay.barrier_reduction == first.barrier_reduction
    assert replay.fusion_reduction == first.fusion_reduction
    assert replay.hypotheses == first.hypotheses
    assert replay.terminal is replay.selection is replay.enriched_output is None
    assert replay.adapter_infer_calls == 0
    assert harness.protocol_adapter is not None
    action_part_count = sum(
        len(item.input_plan.call_plan.parts) for item in first.action_evidence_executions
    )
    total_part_count = (
        coarse_part_count + action_part_count + _boundary_dispatch_count(first) + 2 * part_count
    )
    assert harness.protocol_adapter.dispatch_calls == total_part_count
    assert len(harness.pipeline.ledger.list_intents()) == total_part_count
    assert len(harness.pipeline.ledger.list_terminals()) == total_part_count
    assert len(harness.pipeline.ledger.list_selections()) == total_part_count
    assert harness.raw_store.list_records() == first_raw
    assert len(first_raw) == total_part_count
    replay_snapshot = harness.repository.snapshot(harness.context.recording_identity)
    assert replay_snapshot == first_snapshot
    assert replay_snapshot.generation == 0
    assert harness.repository.list_outbox(harness.context.recording_identity) == first_outbox
    assert first_outbox == ()
    assert replay.identity_result is None
    _assert_offline(replay, harness)


def test_exact_replay_reuses_dense_qa_chain_without_new_side_effects(
    tmp_path: Path,
) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        coarse_response_factory=lambda request: _coarse_claim_bytes(
            request,
            observation=ProviderObservation.DEGRADED,
        ),
        protocol_only_adapter=True,
    )
    processing_run = _processing_run(harness, run_id=_uuid(21_004))

    first = _run(harness, processing_run=processing_run)

    assert first.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.dense_qa_executions
    assert first.qa_completion_result is not None
    assert first.qa_completion_result.dense_result is not None
    assert first.coarse_qa_result is not None
    first_raw = harness.raw_store.list_records()
    expected_dispatches = (
        len(first.coarse_qa_result.source_outputs)
        + sum(len(item.part_results) for item in first.dense_qa_executions)
        + sum(len(item.part_results) for item in first.action_evidence_executions)
        + _boundary_dispatch_count(first)
        + 2 * len(first.part_results)
    )
    assert harness.protocol_adapter is not None
    assert harness.protocol_adapter.dispatch_calls == expected_dispatches

    replay = _run(harness, processing_run=processing_run)

    assert replay.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert replay.processing_run == first.processing_run
    assert replay.run_memberships == first.run_memberships
    assert replay.dense_qa_executions == first.dense_qa_executions
    assert replay.qa_completion_result == first.qa_completion_result
    assert replay.event_proposal_result == first.event_proposal_result
    assert replay.candidate_reduction_result == first.candidate_reduction_result
    assert replay.action_evidence_executions == first.action_evidence_executions
    assert replay.provisional_fusion_result == first.provisional_fusion_result
    assert replay.boundary_refinement_executions == first.boundary_refinement_executions
    assert replay.final_fusion_context == first.final_fusion_context
    assert replay.part_results == first.part_results
    assert replay.fusion_reduction == first.fusion_reduction
    assert replay.hypotheses == first.hypotheses
    assert replay.adapter_infer_calls == 0
    assert harness.protocol_adapter.dispatch_calls == expected_dispatches
    assert harness.raw_store.list_records() == first_raw
    assert len(harness.pipeline.ledger.list_intents()) == expected_dispatches
    assert len(harness.pipeline.ledger.list_terminals()) == expected_dispatches
    assert len(harness.pipeline.ledger.list_selections()) == expected_dispatches
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
    assert first.event_proposal_result is not None
    assert second.event_proposal_result is not None
    assert first.event_proposal_result.semantic_sha256 == (
        second.event_proposal_result.semantic_sha256
    )
    assert first.candidate_reduction_result is not None
    assert second.candidate_reduction_result is not None
    assert tuple(
        item.candidate_logical_key for item in first.candidate_reduction_result.candidates
    ) == tuple(item.candidate_logical_key for item in second.candidate_reduction_result.candidates)
    assert tuple(
        item.evidence_result.logical_key for item in first.action_evidence_executions
    ) == tuple(item.evidence_result.logical_key for item in second.action_evidence_executions)
    assert first.provisional_fusion_result is not None
    assert second.provisional_fusion_result is not None
    assert first.provisional_fusion_result.logical_key == (
        second.provisional_fusion_result.logical_key
    )
    assert tuple(item.logical_key for item in first.provisional_fusion_result.actions) == tuple(
        item.logical_key for item in second.provisional_fusion_result.actions
    )
    assert first.output_decision == second.output_decision
    assert first.hypotheses == second.hypotheses
    assert tuple(
        (item.node_type, item.node_logical_key, item.role) for item in first.run_memberships
    ) == tuple(
        (item.node_type, item.node_logical_key, item.role) for item in second.run_memberships
    )
    assert all(item.disposition is RunNodeDisposition.CREATED for item in first.run_memberships)
    assert all(item.disposition is RunNodeDisposition.REUSED for item in second.run_memberships)
    assert second.adapter_infer_calls == 0
    assert first.coarse_qa_result is not None
    assert harness.pipeline.adapter.infer_calls == (
        len(first.coarse_qa_result.source_outputs)
        + sum(len(item.part_results) for item in first.action_evidence_executions)
        + _boundary_dispatch_count(first)
        + 2 * len(first.input_plan.call_plan.parts)
    )
    assert first.identity_result is second.identity_result is None
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0


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
    assert first.coarse_qa_result is not None
    coarse_part_count = len(first.coarse_qa_result.source_outputs)
    assert coarse_part_count == part_count
    first_counts = _inference_evidence_counts(evidence_path)
    action_part_count = sum(
        len(item.input_plan.call_plan.parts) for item in first.action_evidence_executions
    )
    assert set(first_counts.values()) == {
        coarse_part_count + action_part_count + _boundary_dispatch_count(first) + 2 * part_count
    }

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
    assert first.identity_result is second.identity_result is None
    assert repository.snapshot(first_harness.context.recording_identity).generation == 0


def test_sqlite_barrier_recovers_same_run_after_reduction_commit_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "inference-evidence.sqlite3"
    barrier_path = tmp_path / "call-barriers.sqlite3"
    logical_registry_root = tmp_path / "logical-registry"
    first_harness = _harness(
        _claim_bytes,
        logical_registry_root=logical_registry_root,
        max_images_per_request=3,
        inference_evidence_path=evidence_path,
        barrier_path=barrier_path,
    )
    first_storage = first_harness.barrier_storage
    assert first_storage is not None
    assert first_harness.pipeline.barrier_storage is first_storage
    assert first_harness.pipeline.call_barrier_storage is first_storage
    processing_run = _processing_run(first_harness, run_id=_uuid(22_303))

    persisted_during_interruption: list[InferenceCallReduction] = []
    append_reduction = first_storage.append_reduction

    class SimulatedProcessInterruption(BaseException):
        pass

    def append_reduction_then_interrupt(
        reduction: InferenceCallReduction,
    ) -> InferenceCallReduction:
        stored = append_reduction(reduction)
        claims = stored.normalized_output.get("claims")
        is_final_fusion = isinstance(claims, list) and any(
            isinstance(claim, dict)
            and claim.get("kind") == ProviderClaimKind.FUSION_HYPOTHESIS.value
            for claim in claims
        )
        if is_final_fusion:
            persisted_during_interruption.append(stored)
            raise SimulatedProcessInterruption
        return stored

    monkeypatch.setattr(
        first_storage,
        "append_reduction",
        append_reduction_then_interrupt,
    )

    with pytest.raises(SimulatedProcessInterruption):
        _run(first_harness, processing_run=processing_run)

    assert len(persisted_during_interruption) == 1
    persisted_reduction = persisted_during_interruption[0]
    barrier_id = persisted_reduction.barrier_id
    persisted_definition = first_storage.get_definition(barrier_id)
    persisted_completions = first_storage.list_completions(barrier_id)
    persisted_barrier = first_storage.get_barrier(barrier_id)
    persisted_state = first_storage.get_state(barrier_id)
    persisted_members = first_storage.get_members(barrier_id)
    assert persisted_definition is not None
    assert len(persisted_completions) > 1
    assert persisted_reduction == first_storage.get_reduction(barrier_id)
    assert persisted_barrier is not None
    assert persisted_state is not None and persisted_state.status == "CLOSED"
    assert len(persisted_members) == len(persisted_completions)
    first_evidence_counts = _inference_evidence_counts(evidence_path)

    second_factory_calls = 0

    def provider_dispatch_must_not_run(
        _request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal second_factory_calls
        second_factory_calls += 1
        raise AssertionError("persisted selected evidence must prevent provider dispatch")

    second_harness = _harness(
        provider_dispatch_must_not_run,
        logical_registry_root=logical_registry_root,
        max_images_per_request=3,
        inference_evidence_path=evidence_path,
        barrier_path=barrier_path,
        clock=lambda: NOW + timedelta(seconds=1),
    )
    second_storage = second_harness.barrier_storage
    assert second_storage is not None
    assert second_storage is not first_storage
    assert second_harness.pipeline.barrier_storage is second_storage
    assert second_harness.pipeline.call_barrier_storage is second_storage

    recovered = _run(second_harness, processing_run=processing_run)

    assert recovered.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert recovered.run_id == processing_run.run_id
    assert _barrier_id(recovered) == barrier_id
    assert second_factory_calls == 0
    assert recovered.adapter_infer_calls == 0
    assert second_harness.fixture_adapter.infer_calls == 0
    assert second_storage.get_definition(barrier_id) == persisted_definition
    assert second_storage.list_completions(barrier_id) == persisted_completions
    assert second_storage.get_reduction(barrier_id) == persisted_reduction
    assert recovered.barrier_reduction == persisted_reduction
    assert second_storage.get_barrier(barrier_id) == persisted_barrier
    assert second_storage.get_state(barrier_id) == persisted_state
    assert second_storage.get_members(barrier_id) == persisted_members
    assert _inference_evidence_counts(evidence_path) == first_evidence_counts


def test_fresh_retry_attempt_reuses_run_independent_fusion_and_hypotheses(
    tmp_path: Path,
) -> None:
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

    proposal_calls = 0

    def retry_first_proposal(
        request: VisionInferenceRequest,
    ) -> OfflineFixtureResponse:
        nonlocal proposal_calls
        proposal_calls += 1
        if proposal_calls == 1:
            return _failure(
                request,
                retryability=Retryability.RETRYABLE,
                status=InferenceStatus.TIMEOUT,
            )
        return _event_proposal_claim_bytes(request)

    second_harness = _harness(
        retry_first_call,
        logical_registry_root=tmp_path,
        max_images_per_request=3,
        repository=repository,
        event_proposal_response_factory=retry_first_proposal,
    )
    second = _run(second_harness)
    assert second.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.input_plan is not None and second.input_plan is not None
    assert first.input_plan.semantic_sha256 == second.input_plan.semantic_sha256
    assert first.event_proposal_result is not None
    assert second.event_proposal_result is not None
    assert tuple(
        item.source.source_inference_id for item in first.event_proposal_result.proposals
    ) != tuple(item.source.source_inference_id for item in second.event_proposal_result.proposals)
    assert first.event_proposal_result.semantic_sha256 == (
        second.event_proposal_result.semantic_sha256
    )
    assert first.event_proposal_result.logical_key == second.event_proposal_result.logical_key
    assert first.candidate_reduction_result is not None
    assert second.candidate_reduction_result is not None
    assert first.candidate_reduction_result.semantic_sha256 == (
        second.candidate_reduction_result.semantic_sha256
    )
    assert tuple(
        item.candidate_logical_key for item in first.candidate_reduction_result.candidates
    ) == tuple(item.candidate_logical_key for item in second.candidate_reduction_result.candidates)
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

    assert first.identity_result is second.identity_result is None
    snapshot = repository.snapshot(first_harness.context.recording_identity)
    assert snapshot.generation == 0


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


def test_pre_v4_running_processing_run_cannot_resume_under_v4(tmp_path: Path) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path)
    legacy = CanonicalProcessingRunContext.fresh(
        run_id=_uuid(22_101),
        recording_identity=harness.context.recording_identity,
        mcap_id=harness.context.ready_manifest.mcap_id,
        pipeline_version="canonical-offline-v3",
        config_sha256=harness.execution_policy.semantic_sha256,
        started_at=NOW_TEXT,
    )
    resumed = CanonicalProcessingRunContext.resume(legacy.to_record())

    assert CANONICAL_OFFLINE_PIPELINE_VERSION == "canonical-offline-v5"
    with pytest.raises(CanonicalOfflineConfigurationError, match="processing run"):
        _run(harness, processing_run=resumed)

    assert harness.pipeline.adapter.capability_calls == 0
    assert harness.pipeline.adapter.infer_calls == 0
    assert harness.logical_node_registry.list_run_memberships(resumed.run_id) == ()


def test_run_result_rejects_tampered_binding_and_membership_proof(tmp_path: Path) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path)
    result = _run(harness)
    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert result.output_decision is not None

    with pytest.raises(
        ValidationError,
        match="boundary refinement executions require their complete upstream closure",
    ):
        _revalidate_result(result, qa_completion_result=None)

    v1_decision = result.output_decision.model_dump(mode="python")
    v1_decision["schema_version"] = "1.0"
    with pytest.raises(ValidationError, match=r"2\.0"):
        _revalidate_result(result, output_decision=v1_decision)

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
    with pytest.raises(
        ValidationError,
        match="boundary calls do not bind the exact upstream run closure",
    ):
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


def test_run_result_requires_exact_local_evidence_proof(tmp_path: Path) -> None:
    harness = _harness(_claim_bytes, logical_registry_root=tmp_path)
    result = _run(harness)
    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert result.output_decision is not None
    assert len(result.hypotheses) == 1
    decision = result.output_decision
    output_proof = decision.production_output_admission
    assert output_proof is not None
    hypothesis = result.hypotheses[0]

    def rebound_decision(
        *,
        evidence_class: AdmissionEvidenceClass,
        output_proof: OutputAdmissionProof,
        admitted_claim_ordinals: tuple[int, ...],
    ) -> CanonicalOutputAdmissionDecision:
        draft = decision.model_copy(
            update={
                "evidence_class": evidence_class,
                "production_eligible": False,
                "production_output_admission": output_proof,
                "admitted_claim_ordinals": admitted_claim_ordinals,
            }
        )
        digest = semantic_sha256(canonical_output_decision_projection(draft))
        return draft.model_copy(
            update={
                "semantic_sha256": digest,
                "decision_id": _stable_uuid(
                    CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE,
                    digest,
                ),
            }
        )

    benchmark_admission = AdmissionProof.from_context(
        harness.context,
        evidence_class=AdmissionEvidenceClass.GOVERNED_BENCHMARK,
    )
    benchmark_output = OutputAdmissionProof.create(
        recording_identity=harness.context.recording_identity,
        source_enrichments=decision.source_enrichments,
        admitted_hypothesis_facts=output_proof.admitted_hypothesis_facts,
        policy=harness.execution_policy.output_admission_policy,
        evidence_class=AdmissionEvidenceClass.GOVERNED_BENCHMARK,
    )
    benchmark_hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=hypothesis.recording_identity,
        effective_interval=hypothesis.effective_interval,
        semantic_fingerprint_sha256=hypothesis.semantic_fingerprint_sha256,
        fusion_logical_key=hypothesis.fusion_logical_key,
        fusion_output_ordinal=hypothesis.fusion_output_ordinal,
        source_enrichments=hypothesis.source_enrichments,
        production_admission=benchmark_admission,
        production_output_admission=benchmark_output,
    )
    benchmark_decision = rebound_decision(
        evidence_class=AdmissionEvidenceClass.GOVERNED_BENCHMARK,
        output_proof=benchmark_output,
        admitted_claim_ordinals=decision.admitted_claim_ordinals,
    )

    with pytest.raises(ValidationError, match="decisions require LOCAL_CONFORMANCE"):
        _revalidate_result(
            result,
            output_decision=benchmark_decision,
            hypotheses=(benchmark_hypothesis,),
        )
    with pytest.raises(ValidationError, match="one exact local proof lineage"):
        _revalidate_result(result, hypotheses=(benchmark_hypothesis,))

    extra_fact = ProductionAdmittedHypothesisFact(
        fusion_output_ordinal=99,
        effective_interval=NanosecondInterval(start_ns=900_000_000, end_ns=900_000_001),
        semantic_fingerprint_sha256=_digest("extra-hypothesis"),
        fusion_logical_key=f"fusion:{_digest('extra-hypothesis')}",
    )
    expanded_output = OutputAdmissionProof.create(
        recording_identity=harness.context.recording_identity,
        source_enrichments=decision.source_enrichments,
        admitted_hypothesis_facts=(
            *output_proof.admitted_hypothesis_facts,
            extra_fact,
        ),
        policy=harness.execution_policy.output_admission_policy,
    )
    expanded_hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=hypothesis.recording_identity,
        effective_interval=hypothesis.effective_interval,
        semantic_fingerprint_sha256=hypothesis.semantic_fingerprint_sha256,
        fusion_logical_key=hypothesis.fusion_logical_key,
        fusion_output_ordinal=hypothesis.fusion_output_ordinal,
        source_enrichments=hypothesis.source_enrichments,
        production_admission=hypothesis.production_admission,
        production_output_admission=expanded_output,
    )
    expanded_decision = rebound_decision(
        evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
        output_proof=expanded_output,
        admitted_claim_ordinals=tuple(
            sorted((*decision.admitted_claim_ordinals, extra_fact.fusion_output_ordinal))
        ),
    )

    with pytest.raises(ValidationError, match="exactly cover the run hypotheses"):
        _revalidate_result(
            result,
            output_decision=expanded_decision,
            hypotheses=(expanded_hypothesis,),
        )


def test_local_result_rejects_authoritative_identity_result(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
    )
    successful = _run(harness)
    assert successful.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert successful.identity_result is None

    identity_policy = EventIdentityPolicyRef(
        version="exact-fingerprint-v1",
        semantic_sha256=_digest("exact-fingerprint-v1"),
    )
    identity_registry = EventIdentityRegistryService(
        repository=harness.repository,
        resolver=ExactFingerprintEventIdentityResolver(identity_policy),
        allocator=_SequenceEventIdAllocator(),
        output_admission_policy=harness.execution_policy.output_admission_policy,
    )
    enriched_outputs = tuple(
        item.enriched_output for item in successful.part_results if item.enriched_output is not None
    )
    identity_result = identity_registry.assign_batch(
        admitted_context=harness.context,
        hypotheses=successful.hypotheses,
        enriched_outputs=enriched_outputs,
        decided_at=NOW_TEXT,
    )

    with pytest.raises(ValidationError, match="authoritative identity assignments"):
        _revalidate_result(successful, identity_result=identity_result)


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
    assert result.coarse_qa_result is not None
    coarse_part_count = len(result.coarse_qa_result.source_outputs)
    action_part_count = sum(len(item.part_results) for item in result.action_evidence_executions)
    assert coarse_part_count == part_count
    assert result.attempt_count == part_count + 1
    assert (
        result.adapter_infer_calls
        == coarse_part_count
        + action_part_count
        + _boundary_dispatch_count(result)
        + 2 * part_count
        + 1
    )
    assert tuple(item.orchestration_attempt_count for item in result.part_results) == tuple(
        2 if ordinal == 1 else 1 for ordinal in range(part_count)
    )
    terminals = harness.pipeline.ledger.list_terminals()
    assert (
        len(terminals)
        == coarse_part_count
        + action_part_count
        + _boundary_dispatch_count(result)
        + 2 * part_count
        + 1
    )
    assert sum(item.status is InferenceStatus.TIMEOUT for item in terminals) == 1
    assert sum(item.status is InferenceStatus.SUCCEEDED for item in terminals) == (
        coarse_part_count + action_part_count + _boundary_dispatch_count(result) + 2 * part_count
    )
    assert len(harness.pipeline.ledger.list_selections()) == (
        coarse_part_count + action_part_count + _boundary_dispatch_count(result) + 2 * part_count
    )
    assert result.barrier_reduction is not None
    completions = harness.pipeline.call_barrier_storage.list_completions(_barrier_id(result))
    assert len(completions) == part_count
    assert all(item.status is InferenceStatus.SUCCEEDED for item in completions)
    assert len(harness.raw_store.list_records()) == (
        coarse_part_count + action_part_count + _boundary_dispatch_count(result) + 2 * part_count
    )
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
    assert result.coarse_qa_result is not None
    coarse_part_count = len(result.coarse_qa_result.source_outputs)
    action_part_count = sum(len(item.part_results) for item in result.action_evidence_executions)
    assert coarse_part_count == part_count
    assert result.attempt_count == part_count
    assert (
        result.adapter_infer_calls
        == coarse_part_count + action_part_count + _boundary_dispatch_count(result) + 2 * part_count
    )
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
    assert len(harness.pipeline.ledger.list_selections()) == (
        coarse_part_count
        + action_part_count
        + _boundary_dispatch_count(result)
        + 2 * part_count
        - 1
    )
    barrier_id = _barrier_id(result)
    assert harness.pipeline.call_barrier_storage.get_definition(barrier_id) is not None
    completions = harness.pipeline.call_barrier_storage.list_completions(barrier_id)
    assert len(completions) == part_count
    assert sum(item.status is InferenceStatus.FAILED for item in completions) == 1
    assert harness.pipeline.call_barrier_storage.get_reduction(barrier_id) is None
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    assert len(harness.raw_store.list_records()) == (
        coarse_part_count
        + action_part_count
        + _boundary_dispatch_count(result)
        + 2 * part_count
        - 1
    )
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
    assert len(records) == 6
    assert sum(item.data == response for item in records) == 1
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)


def test_selected_terminal_raw_reference_mismatch_fails_closed(tmp_path: Path) -> None:
    harness = _harness(
        _claim_bytes,
        logical_registry_root=tmp_path,
        mismatch_raw_reference=True,
        protocol_only_adapter=True,
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
    assert len(records) == 6
    assert all(item.artifact_id != _uuid(99_999) for item in records)
    assert harness.protocol_adapter is not None
    assert harness.protocol_adapter.dispatch_calls == 6
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
    assert result.coarse_qa_result is not None
    coarse_part_count = len(result.coarse_qa_result.source_outputs)
    action_part_count = sum(len(item.part_results) for item in result.action_evidence_executions)
    assert coarse_part_count == part_count
    assert result.attempt_count == part_count
    assert (
        result.adapter_infer_calls
        == coarse_part_count + action_part_count + _boundary_dispatch_count(result) + 2 * part_count
    )
    assert result.terminal is result.selection is result.enriched_output is None
    assert result.barrier_reduction is not None
    assert len(result.barrier_reduction.ordered_completion_ids) == part_count
    assert result.fusion_reduction is not None
    assert result.fusion_reduction.outcome == "CLAIMS"
    assert tuple(item.fusion_output_ordinal for item in result.fusion_reduction.claims) == (0,)
    assert len(result.hypotheses) == 1
    assert result.identity_result is None
    completions = harness.pipeline.call_barrier_storage.list_completions(_barrier_id(result))
    assert tuple(item.part_ordinal for item in completions) == tuple(range(part_count))
    assert len(harness.raw_store.list_records()) == (
        coarse_part_count + action_part_count + _boundary_dispatch_count(result) + 2 * part_count
    )
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
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
    assert result.coarse_qa_result is not None
    assert len(harness.raw_store.list_records()) == (
        len(result.coarse_qa_result.source_outputs)
        + sum(len(item.part_results) for item in result.action_evidence_executions)
        + _boundary_dispatch_count(result)
        + 2 * part_count
    )
    assert harness.repository.snapshot(harness.context.recording_identity).generation == 0
    _assert_offline(result, harness)
