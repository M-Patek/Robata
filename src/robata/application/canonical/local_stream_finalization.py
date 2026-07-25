"""Deterministic local-conformance execution of the durable stream DAG.

This module is deliberately non-production.  It proves scheduling,
window-local causal inference lineage, incremental result, EOS closure, and
finalization recovery with an explicit deterministic local mock.  The mock
never claims governed provider or production-qualified evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import ConfigDict

from robata.adapters.sqlite_stream_delivery import (
    PreparedWindowReductionEvidence,
    SQLiteStreamDeliveryAuthority,
)
from robata.application.canonical.result_validation import CanonicalOfflineRunResult
from robata.application.canonical.stream_recording_reduction import (
    LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
    LOCAL_STREAM_RECORDING_RESULT_SCHEMA_VERSION,
    LOCAL_STREAM_RECORDING_RESULT_V2_SCHEMA_VERSION,
    LOCAL_STREAM_RECORDING_RESULT_V3_SCHEMA_VERSION,
    LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION,
    LocalStreamCanonicalTruth,
    LocalStreamQaCameraReference,
    LocalStreamRecordingResult,
    LocalStreamRecordingResultV2,
    LocalStreamRecordingResultV3,
    LocalStreamRecordingResultV4,
    LocalStreamSemanticIntervalReference,
    LocalStreamWindowSemanticEvidence,
    create_local_stream_recording_result,
    create_local_stream_recording_result_v2,
    create_local_stream_recording_result_v3,
    create_local_stream_recording_result_v4,
    validate_local_stream_recording_result_v3_truth,
)
from robata.application.canonical.stream_scheduler import (
    DurableStreamWindowScheduler,
    StreamDrainWorkSnapshot,
)
from robata.contracts.common import (
    NanosecondInterval,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.inference import InferenceStatus, ModelInferenceUsage
from robata.contracts.local_stream_causal import (
    LocalStreamStageEvidenceReference,
    LocalStreamWindowSemanticEvidenceV2,
    LocalStreamWindowSemanticStatus,
    create_local_stream_window_inference_plan,
    create_local_stream_window_semantic_evidence_v2,
)
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    NonEmptyString,
    StreamPurpose,
    StreamStage,
    StreamSubjectType,
    TerminalOutcome,
)
from robata.contracts.stream_finalization import (
    FinalizationSubjectMapping,
    RecordingFinalizationMap,
    WindowTerminalClosure,
    create_recording_finalization_map,
)
from robata.contracts.stream_inference import (
    StreamInferenceIntent,
    StreamInferenceTerminal,
    StreamInputPlanReference,
    StreamWindowResult,
    create_stream_accepted_call_evidence,
    create_stream_inference_intent,
    create_stream_inference_terminal,
    create_stream_window_result,
    reference_stream_accepted_call,
    reference_stream_inference_intent,
    reference_stream_inference_terminal,
)
from robata.contracts.stream_planning import (
    ExpectedWindowDeclaration,
    StreamWorkItemPlan,
)
from robata.contracts.stream_window import (
    create_stream_inference_attempt_identity,
    create_stream_inference_identity,
)
from robata.inference.models import (
    InferenceStatus as ProviderInferenceStatus,
)
from robata.inference.models import (
    ModelInference as ProviderModelInference,
)
from robata.inference.models import (
    VisionTask as ProviderVisionTask,
)
from robata.queue.stream_models import (
    TERMINAL_STREAM_WORK_STATES,
    StreamTerminalEvidence,
    StreamWorkItem,
    StreamWorkItemState,
)

LOCAL_STREAM_MOCK_EXECUTOR_POLICY_VERSION: Final = (
    "local-conformance-provider-neutral-stream-executor-v1"
)
LOCAL_STREAM_REDUCTION_POLICY_VERSION: Final = (
    "local-conformance-provider-neutral-window-reduction-v1"
)
LOCAL_STREAM_CAUSAL_REDUCTION_POLICY_VERSION: Final = "local-conformance-window-causal-reduction-v2"
LOCAL_STREAM_CAUSAL_CLOSURE_PROJECTION_VERSION: Final = (
    "local-stream-six-camera-slot-closure-semantic-v1"
)
LOCAL_STREAM_MOCK_PROVIDER_VERSION: Final = "local-conformance-window-mock-v1"
LOCAL_STREAM_RECEIPT_VERSION: Final = "local-stream-work-receipt-v1"
LOCAL_STREAM_WORK_RECEIPT_SCHEMA_ID: Final = "https://schemas.robata.dev/local-stream-work-receipt"
LOCAL_STREAM_WORK_RECEIPT_SCHEMA_VERSION: Final = "1.0.0"
LOCAL_STREAM_FINAL_MAPPING_VERSION: Final = "local-stream-final-window-mapping-v1"
LOCAL_CONFORMANCE_EVIDENCE_CLASS: Final = "LOCAL_CONFORMANCE"
# The pre-EOS bridge deliberately has a small surface: it only owns the three
# existing canonical provider tasks that appear in the window DAG.  WINDOW and
# reduction work retain their scheduler/local-reduction responsibilities.
_PRE_EOS_PROVIDER_TASK_BY_STAGE: Final[dict[StreamStage, ProviderVisionTask]] = {
    StreamStage.QA_COARSE: ProviderVisionTask.QA_COARSE,
    StreamStage.QA_DENSE: ProviderVisionTask.QA_DENSE,
    StreamStage.EVENT_PROPOSAL: ProviderVisionTask.EVENT_PROPOSAL,
}
_ARTIFACT_NAMESPACE: Final = uuid5(NAMESPACE_URL, "robata:local-conformance-stream-artifact-v1")


class LocalStreamFinalizationError(RuntimeError):
    """Local stream execution conflicts with durable state or exact artifacts."""


@dataclass(frozen=True, slots=True)
class LocalStreamFinalizationSchemaRefs:
    """Registered pins for artifacts emitted by the local executor."""

    local_work_receipt: SchemaRef
    stream_window_result: SchemaRef
    recording_finalization: SchemaRef
    stream_recording_result: SchemaRef
    window_inference_plan: SchemaRef
    window_semantic_evidence_v2: SchemaRef
    stream_inference_identity: SchemaRef
    stream_inference_attempt: SchemaRef
    stream_inference_intent: SchemaRef
    stream_accepted_call: SchemaRef
    stream_inference_terminal: SchemaRef
    window_semantic_evidence: SchemaRef | None = None
    # P5 may complete a provider-neutral QA/event stage before EOS. Such a
    # terminal is an existing ``model-inference`` artifact, not a
    # LOCAL_CONFORMANCE receipt. Keeping this optional preserves the fast
    # conformance-only mode and avoids changing published local-stream schemas.
    model_inference: SchemaRef | None = None


@dataclass(frozen=True, slots=True)
class FinalRecordingFacts:
    """Authoritative final-recording facts supplied by the canonical run."""

    final_source_subject_type: str
    final_source_subject_id: str
    final_source_exact_sha256: str
    final_recording_identity: str
    final_duration_ns: int


@dataclass(frozen=True, slots=True)
class LocalStreamFinalizationOutcome:
    """Recovered or newly completed local-conformance stream result."""

    window_results: tuple[StreamWindowResult, ...]
    terminal_closure: WindowTerminalClosure
    recording_finalization: RecordingFinalizationMap
    recording_result: (
        LocalStreamRecordingResult
        | LocalStreamRecordingResultV2
        | LocalStreamRecordingResultV3
        | LocalStreamRecordingResultV4
    )
    recording_result_evidence_ref: ArtifactEvidenceRef
    finalization_work: StreamWorkItem
    newly_executed_work_count: int
    canonical_truth: LocalStreamCanonicalTruth | None = None
    evidence_class: str = LOCAL_CONFORMANCE_EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class _PreparedStreamWorkCompletion:
    terminal_evidence: StreamTerminalEvidence
    window_result: StreamWindowResult | None = None
    causal_evidence: PreparedWindowReductionEvidence | None = None


class LocalStreamWorkReceipt(StrictModel):
    """Exact non-production evidence emitted for one local stream-work item."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": ("https://schemas.robata.dev/v1/local-stream-work-receipt.schema.json"),
        },
    )

    schema_version: Literal["1.0"] = "1.0"
    schema_ref: SchemaRef
    model_version: Literal["local-stream-work-receipt-v1"] = LOCAL_STREAM_RECEIPT_VERSION
    evidence_class: Literal["LOCAL_CONFORMANCE"] = LOCAL_CONFORMANCE_EVIDENCE_CLASS
    production_eligible: Literal[False] = False
    executor_policy_version: SchemaVersion
    plan_key: NonEmptyString
    work_item_id: OpaqueUuid
    work_logical_key: NonEmptyString
    stage: StreamStage
    subject_type: StreamSubjectType
    subject_key: NonEmptyString
    subject_semantic_sha256: Sha256Digest
    input_semantic_sha256: Sha256Digest
    config_semantic_sha256: Sha256Digest
    ordered_upstream_terminal_exact_sha256_values: tuple[Sha256Digest, ...]
    result: Literal["ABSTAINED_NO_PROVIDER", "LOCAL_CONFORMANCE_STAGE_COMPLETE"]


class _ExactLocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, payload: bytes, schema_ref: SchemaRef) -> ArtifactEvidenceRef:
        digest = hashlib.sha256(payload).hexdigest()
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as output:
                output.write(payload)
        except FileExistsError as error:
            if path.read_bytes() != payload:
                raise LocalStreamFinalizationError(
                    "content-addressed local artifact conflicts with existing bytes"
                ) from error
        return ArtifactEvidenceRef(
            artifact_id=str(uuid5(_ARTIFACT_NAMESPACE, digest)),
            exact_sha256=digest,
            byte_count=len(payload),
            media_type="application/json",
            schema_ref=schema_ref,
        )

    def read(self, reference: ArtifactEvidenceRef) -> bytes:
        path = self._path(reference.exact_sha256)
        try:
            payload = path.read_bytes()
        except FileNotFoundError as error:
            raise LocalStreamFinalizationError(
                "durable stream terminal references a missing local artifact"
            ) from error
        if (
            len(payload) != reference.byte_count
            or hashlib.sha256(payload).hexdigest() != reference.exact_sha256
        ):
            raise LocalStreamFinalizationError(
                "local stream artifact bytes do not match their exact reference"
            )
        return payload

    def path_for(self, reference: ArtifactEvidenceRef) -> Path:
        return self._path(reference.exact_sha256)

    def _path(self, digest: str) -> Path:
        return self._root / digest[:2] / f"{digest}.json"


def _declared_stream_timeline(
    declarations: tuple[ExpectedWindowDeclaration, ...],
) -> tuple[int, NanosecondInterval]:
    """Return the explicit source origin and relative requested span."""

    if not declarations:
        raise LocalStreamFinalizationError("stream graph has no expected window declarations")
    first = declarations[0]
    if first.ordinal != 0:
        raise LocalStreamFinalizationError("stream declarations must begin at ordinal zero")
    origin_ns = first.requested_interval.start_ns
    if any(
        declaration.requested_interval.start_ns < origin_ns
        or declaration.requested_interval.end_ns <= origin_ns
        for declaration in declarations
    ):
        raise LocalStreamFinalizationError(
            "stream declarations do not share one nonnegative timeline origin"
        )
    end_ns = max(declaration.requested_interval.end_ns - origin_ns for declaration in declarations)
    return origin_ns, NanosecondInterval(start_ns=0, end_ns=end_ns)


def _normalize_stream_effective_interval(
    effective_interval: NanosecondInterval,
    *,
    source_timeline_origin_ns: int,
    canonical_requested_interval: NanosecondInterval,
) -> NanosecondInterval:
    """Map an absolute stream interval into the recording-relative request."""

    start_ns = (
        canonical_requested_interval.start_ns
        + effective_interval.start_ns
        - source_timeline_origin_ns
    )
    end_ns = (
        canonical_requested_interval.start_ns
        + effective_interval.end_ns
        - source_timeline_origin_ns
    )
    start_ns = max(start_ns, canonical_requested_interval.start_ns)
    end_ns = min(end_ns, canonical_requested_interval.end_ns)
    if start_ns >= end_ns:
        raise LocalStreamFinalizationError(
            "stream window has no recording-relative overlap with the requested interval"
        )
    return NanosecondInterval(start_ns=start_ns, end_ns=end_ns)


class LocalConformanceStreamFinalizer:
    """Execute and finalize one already-declared durable stream graph.

    The coordinator is intended to run after canonical local analysis has
    produced final-recording facts and before primary completion is committed.
    It is restartable: terminal work and content-addressed artifacts are read
    back and validated instead of being regenerated.
    """

    def __init__(
        self,
        *,
        scheduler: DurableStreamWindowScheduler,
        delivery_authority: SQLiteStreamDeliveryAuthority,
        artifact_root: str | Path,
        schema_refs: LocalStreamFinalizationSchemaRefs,
        final_recording: FinalRecordingFacts | None = None,
        canonical_result: CanonicalOfflineRunResult | None = None,
        source_timeline_origin_ns: int | None = None,
        canonical_requested_interval: NanosecondInterval | None = None,
        window_purpose: StreamPurpose,
        mock_executor_policy_version: str = LOCAL_STREAM_MOCK_EXECUTOR_POLICY_VERSION,
        terminal_policy_version: str = "stream-terminal-policy-v1",
        worker_id: str = "local-conformance-stream-worker",
        lease_duration_seconds: int = 300,
        recover_graph_before_execute: bool = True,
        stage_terminal_executor: (
            Callable[[StreamWorkItemPlan], StreamTerminalEvidence | None] | None
        ) = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(scheduler, DurableStreamWindowScheduler):
            raise TypeError("scheduler must be DurableStreamWindowScheduler")
        if not isinstance(delivery_authority, SQLiteStreamDeliveryAuthority):
            raise TypeError("delivery_authority must be SQLiteStreamDeliveryAuthority")
        if not isinstance(window_purpose, StreamPurpose):
            raise TypeError("window_purpose must be StreamPurpose")
        if canonical_result is not None and not isinstance(
            canonical_result, CanonicalOfflineRunResult
        ):
            raise TypeError("canonical_result must be CanonicalOfflineRunResult")
        if canonical_result is not None and (
            source_timeline_origin_ns is None or canonical_requested_interval is None
        ):
            raise ValueError(
                "canonical stream finalization requires an explicit timeline origin and interval"
            )
        if source_timeline_origin_ns is not None and (
            isinstance(source_timeline_origin_ns, bool)
            or not isinstance(source_timeline_origin_ns, int)
        ):
            raise TypeError("source_timeline_origin_ns must be an integer")
        if canonical_requested_interval is not None and not isinstance(
            canonical_requested_interval, NanosecondInterval
        ):
            raise TypeError("canonical_requested_interval must be NanosecondInterval")
        if not mock_executor_policy_version:
            raise ValueError("mock_executor_policy_version must be non-empty")
        if not terminal_policy_version:
            raise ValueError("terminal_policy_version must be non-empty")
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        if isinstance(lease_duration_seconds, bool) or lease_duration_seconds <= 0:
            raise ValueError("lease_duration_seconds must be positive")
        if not isinstance(recover_graph_before_execute, bool):
            raise TypeError("recover_graph_before_execute must be bool")
        if stage_terminal_executor is not None and not callable(stage_terminal_executor):
            raise TypeError("stage_terminal_executor must be callable or None")
        self._scheduler = scheduler
        self._delivery = delivery_authority
        self._artifacts = _ExactLocalArtifactStore(Path(artifact_root))
        self._schema_refs = schema_refs
        self._final_recording = final_recording
        self._canonical_result = canonical_result
        self._source_timeline_origin_ns = source_timeline_origin_ns
        self._canonical_requested_interval = canonical_requested_interval
        self._window_purpose = window_purpose
        self._mock_policy = mock_executor_policy_version
        self._terminal_policy = terminal_policy_version
        self._worker_id = worker_id
        self._lease_seconds = lease_duration_seconds
        self._recover_graph_before_execute = recover_graph_before_execute
        self._stage_terminal_executor = stage_terminal_executor
        self._clock = clock or (lambda: datetime.now(UTC))

    def execute(self) -> LocalStreamFinalizationOutcome:
        """Complete every window DAG, close EOS gates, and complete finalization."""

        if self._final_recording is None:
            raise LocalStreamFinalizationError("recording facts are required for EOS finalization")
        if self._scheduler.expected_plan_seal() is None:
            raise LocalStreamFinalizationError(
                "expected window plan must be sealed before local stream execution"
            )
        if not self._scheduler.export_barrier().complete:
            raise LocalStreamFinalizationError(
                "six-camera export barrier must complete before local stream execution"
            )
        all_plans = self._scheduler.work_plans()
        nonfinal = tuple(plan for plan in all_plans if plan.stage is not StreamStage.FINALIZATION)
        executed = self.drain_ready(max_items=max(1, len(nonfinal)))
        items_by_id = {
            item.work_item_id: item for item in self._scheduler.work_items(recover_graph=False)
        }
        unresolved = tuple(
            plan.work_item_id
            for plan in nonfinal
            if items_by_id[plan.work_item_id].state not in TERMINAL_STREAM_WORK_STATES
        )
        if unresolved:
            raise LocalStreamFinalizationError(
                f"stream DAG made no progress; unresolved work: {unresolved}"
            )

        window_results = self._load_window_results(nonfinal, items_by_id)
        closure = self._scheduler.terminal_closure()
        if closure is None:
            closure = self._scheduler.close_finalization_gate()
        final_plan = self._finalization_plan(all_plans)
        finalization = self._create_finalization(window_results, closure, final_plan)
        finalization_ref = self._artifacts.put(
            canonical_json_bytes(finalization),
            self._schema_refs.recording_finalization,
        )
        semantic_evidence = tuple(
            LocalStreamWindowSemanticEvidenceV2.model_validate_json(
                self._artifacts.read(result.result_evidence_ref),
                strict=True,
            )
            for result in window_results
        )
        canonical_truth: LocalStreamCanonicalTruth | None = None
        source_origin: int | None = None
        canonical_interval: NanosecondInterval | None = None
        if self._canonical_result is not None:
            canonical_truth = create_local_stream_canonical_truth(self._canonical_result)
            declarations = self._scheduler.declarations()
            if len(declarations) != len(window_results):
                raise LocalStreamFinalizationError(
                    "sealed declarations and window results do not align"
                )
            source_origin, declared_interval = _declared_stream_timeline(declarations)
            if source_origin != self._source_timeline_origin_ns:
                raise LocalStreamFinalizationError(
                    "stream timeline origin disagrees with admitted alignment evidence"
                )
            canonical_interval = self._canonical_requested_interval
            if canonical_interval is None:
                raise LocalStreamFinalizationError(
                    "canonical stream finalization lacks its requested interval"
                )
            if (
                canonical_interval.start_ns != declared_interval.start_ns
                or canonical_interval.end_ns > declared_interval.end_ns
            ):
                raise LocalStreamFinalizationError(
                    "stream declarations do not cover the canonical requested interval"
                )
        recording_result = create_local_stream_recording_result_v4(
            schema_ref=self._schema_refs.stream_recording_result,
            window_results=window_results,
            window_semantic_evidence=semantic_evidence,
            terminal_closure=closure,
            recording_finalization=finalization,
            source_timeline_origin_ns=source_origin,
            canonical_requested_interval=canonical_interval,
        )
        if canonical_truth is not None:
            try:
                validate_local_stream_recording_result_v3_truth(
                    recording_result,
                    canonical_truth,
                    source_timeline_origin_ns=source_origin,
                    canonical_requested_interval=canonical_interval,
                )
            except ValueError as error:
                raise LocalStreamFinalizationError(
                    "causal stream result is incompatible with canonical recording truth"
                ) from error
        recording_result_ref = self._artifacts.put(
            canonical_json_bytes(recording_result),
            self._schema_refs.stream_recording_result,
        )
        self._delivery.publish_recording_finalization(
            plan_key=self._scheduler.plan_key,
            finalization=finalization,
            topic="robata.stream.recording-finalizations.v1",
            message_key=finalization.finalization_key,
            created_at=self._now(),
        )

        final_item = self._scheduler.get(final_plan.work_item_id)
        if final_item.state not in TERMINAL_STREAM_WORK_STATES:
            claim = self._scheduler.claim_and_start(
                self._worker_id,
                self._lease_seconds,
                work_item_id=final_plan.work_item_id,
                now=self._now(),
                recover_graph=False,
            )
            if claim is None:
                raise LocalStreamFinalizationError(
                    "finalization gate is closed but finalization work is not claimable"
                )
            final_item = self._scheduler.complete(
                claim.lease,
                StreamTerminalEvidence(
                    outcome=TerminalOutcome.SUCCEEDED,
                    evidence_ref=finalization_ref,
                    terminal_policy_version=self._terminal_policy,
                    completed_at=final_plan.created_at,
                ),
                now=self._now(),
            )
            executed += 1
        else:
            existing = final_item.terminal_evidence_ref
            if existing is None:
                raise LocalStreamFinalizationError(
                    "terminal finalization work lacks exact evidence"
                )
            if canonical_json_bytes(finalization) != self._artifacts.read(existing):
                raise LocalStreamFinalizationError(
                    "finalization replay changed exact recording-finalization bytes"
                )

        return LocalStreamFinalizationOutcome(
            window_results=window_results,
            terminal_closure=closure,
            recording_finalization=finalization,
            recording_result=recording_result,
            recording_result_evidence_ref=recording_result_ref,
            finalization_work=final_item,
            newly_executed_work_count=executed,
            canonical_truth=canonical_truth,
        )

    def drain_ready(
        self,
        max_items: int,
        *,
        scope: tuple[StreamDrainWorkSnapshot, ...] | None = None,
    ) -> int:
        """Complete at most ``max_items`` ready non-finalization work items."""

        if isinstance(max_items, bool) or not isinstance(max_items, int):
            raise TypeError("max_items must be an integer")
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if self._recover_graph_before_execute:
            self._scheduler.recover()

        bounded_scope = self._scheduler.bounded_drain_scope(max_items) if scope is None else scope
        plans_by_key = {item.plan.work_logical_key: item.plan for item in bounded_scope}
        terminal_refs_by_key = {
            item.plan.work_logical_key: item.terminal_evidence.evidence_ref
            for item in bounded_scope
            if item.terminal_evidence is not None
        }
        pending = {
            item.plan.work_item_id: item.plan
            for item in bounded_scope
            if item.plan.stage is not StreamStage.FINALIZATION and not item.is_terminal
        }
        if not pending:
            return 0
        declarations_by_window = (
            {declaration.window_key: declaration for declaration in self._scheduler.declarations()}
            if any(plan.stage is StreamStage.WINDOW_REDUCTION for plan in pending.values())
            else {}
        )
        executed = 0
        while pending and executed < max_items:
            progress = False
            for plan in tuple(pending.values()):
                if executed >= max_items:
                    break
                claim = self._scheduler.claim_and_start(
                    self._worker_id,
                    self._lease_seconds,
                    work_item_id=plan.work_item_id,
                    now=self._now(),
                    recover_graph=False,
                )
                if (
                    claim is None
                    and self._stage_terminal_executor is not None
                    and plan.stage in _PRE_EOS_PROVIDER_TASK_BY_STAGE
                ):
                    # A provider terminal may already be durably persisted while a
                    # process dies before the stream scheduler accepts it.  Do not
                    # reclaim or re-dispatch the work: resume only the exact live
                    # fence still owned by this local worker, then let the canonical
                    # provider ledger replay its evidence through the same hook.
                    claim = self._scheduler.resume_owned_active(
                        self._worker_id,
                        self._lease_seconds,
                        work_item_id=plan.work_item_id,
                        now=self._now(),
                    )
                if claim is None:
                    continue
                prepared = self._prepare_work_completion(
                    plan,
                    plans_by_key,
                    terminal_refs_by_key,
                    declarations_by_window,
                )
                evidence = prepared.terminal_evidence
                if plan.stage is StreamStage.WINDOW_REDUCTION:
                    result = prepared.window_result
                    causal_evidence = prepared.causal_evidence
                    if result is None or causal_evidence is None:
                        raise LocalStreamFinalizationError(
                            "window reduction did not prepare its causal result closure"
                        )
                    member = self._scheduler.prepare_window_terminal_member(
                        claim.lease,
                        evidence,
                    )
                    self._delivery.commit_window_reduction(
                        lease=claim.lease,
                        terminal_evidence=evidence,
                        terminal_member=member,
                        result=result,
                        causal_evidence=causal_evidence,
                        topic="robata.stream.window-results.v1",
                        message_key=result.window_result_key,
                        now=self._now(),
                    )
                else:
                    self._scheduler.complete(claim.lease, evidence, now=self._now())
                terminal_refs_by_key[plan.work_logical_key] = evidence.evidence_ref
                executed += 1
                del pending[plan.work_item_id]
                progress = True
            if not progress:
                break
        return executed

    def artifact_path_for(self, reference: ArtifactEvidenceRef) -> Path:
        """Resolve an explicit local locator for one exact artifact reference."""

        return self._artifacts.path_for(reference)

    def _prepare_work_completion(
        self,
        plan: StreamWorkItemPlan,
        plans_by_key: dict[str, StreamWorkItemPlan],
        terminal_refs_by_key: dict[str, ArtifactEvidenceRef],
        declarations_by_window: dict[str, ExpectedWindowDeclaration],
    ) -> _PreparedStreamWorkCompletion:
        if plan.stage is StreamStage.WINDOW_REDUCTION:
            declaration = declarations_by_window.get(plan.subject.subject_key)
            if declaration is None:
                raise LocalStreamFinalizationError(
                    "window reduction lacks its expected declaration"
                )
            return self._prepare_causal_window_reduction(
                plan,
                declaration,
                plans_by_key,
                terminal_refs_by_key,
            )
        # The normal local path intentionally remains a deterministic mock.
        # When a P5 provider-neutral executor is configured, however, it gets
        # first refusal for ready QA/event work. The scheduler continues to
        # own claims, leases, and terminal acceptance; this hook owns neither
        # scheduling state nor QA/event reduction semantics.
        if self._stage_terminal_executor is not None:
            terminal = self._stage_terminal_executor(plan)
            if terminal is not None:
                checked_terminal = StreamTerminalEvidence.model_validate(
                    terminal.model_dump(mode="python"), strict=True
                )
                if plan.stage not in _PRE_EOS_PROVIDER_TASK_BY_STAGE:
                    raise LocalStreamFinalizationError(
                        "provider-neutral execution is only valid for QA/event window stages"
                    )
                model_inference_ref = self._schema_refs.model_inference
                if (
                    model_inference_ref is None
                    or checked_terminal.evidence_ref.schema_ref != model_inference_ref
                ):
                    raise LocalStreamFinalizationError(
                        "provider-neutral terminal must reference the registered "
                        "model inference artifact"
                    )
                if checked_terminal.terminal_policy_version != self._terminal_policy:
                    raise LocalStreamFinalizationError(
                        "provider-neutral terminal violates the stream terminal policy"
                    )
                return _PreparedStreamWorkCompletion(terminal_evidence=checked_terminal)
        receipt = self._receipt(plan, plans_by_key, terminal_refs_by_key)
        receipt_ref = self._artifacts.put(
            canonical_json_bytes(receipt), self._schema_refs.local_work_receipt
        )
        return _PreparedStreamWorkCompletion(
            terminal_evidence=StreamTerminalEvidence(
                outcome=TerminalOutcome.SUCCEEDED,
                evidence_ref=receipt_ref,
                terminal_policy_version=self._terminal_policy,
                completed_at=plan.created_at,
            )
        )

    def _prepare_causal_window_reduction(
        self,
        plan: StreamWorkItemPlan,
        declaration: ExpectedWindowDeclaration,
        plans_by_key: dict[str, StreamWorkItemPlan],
        terminal_refs_by_key: dict[str, ArtifactEvidenceRef],
    ) -> _PreparedStreamWorkCompletion:
        upstream = self._causal_upstream_stage_evidence(
            plan,
            plans_by_key,
            terminal_refs_by_key,
        )
        closure_digest = semantic_sha256(
            {
                "projection_version": LOCAL_STREAM_CAUSAL_CLOSURE_PROJECTION_VERSION,
                "ordered_six_camera_slots": [
                    slot.model_dump(mode="json")
                    for slot in declaration.ordered_six_slot_segment_or_explicit_absence_closure
                ],
            }
        )
        inference_plan = create_local_stream_window_inference_plan(
            schema_ref=self._schema_refs.window_inference_plan,
            plan_key=self._scheduler.plan_key,
            expected_ordinal=declaration.ordinal,
            window_key=declaration.window_key,
            window_semantic_sha256=declaration.window_semantic_sha256,
            effective_interval=declaration.effective_interval,
            input_plan_semantic_sha256=plan.input_semantic_sha256,
            six_camera_slot_closure_semantic_sha256=closure_digest,
            ordered_upstream_stage_evidence=upstream,
        )
        inference_plan_ref = self._artifacts.put(
            canonical_json_bytes(inference_plan),
            self._schema_refs.window_inference_plan,
        )
        has_media = any(
            slot.kind == "SEGMENT"
            or (
                slot.kind == "SEGMENT_SEQUENCE"
                and any(member.kind == "SEGMENT" for member in slot.ordered_members)
            )
            for slot in declaration.ordered_six_slot_segment_or_explicit_absence_closure
        )
        semantic_status: LocalStreamWindowSemanticStatus = "PROPOSED" if has_media else "NO_EVENTS"
        semantic_evidence = create_local_stream_window_semantic_evidence_v2(
            schema_ref=self._schema_refs.window_semantic_evidence_v2,
            plan=inference_plan,
            plan_ref=inference_plan_ref,
            semantic_status=semantic_status,
            proposal_label="fixture-action" if has_media else None,
        )
        semantic_ref = self._artifacts.put(
            canonical_json_bytes(semantic_evidence),
            self._schema_refs.window_semantic_evidence_v2,
        )
        logical_identity = create_stream_inference_identity(
            schema_ref=self._schema_refs.stream_inference_identity,
            window_key=declaration.window_key,
            window_semantic_sha256=declaration.window_semantic_sha256,
            purpose=self._window_purpose,
            input_plan_semantic_sha256=inference_plan.input_plan_semantic_sha256,
        )
        attempt_identity = create_stream_inference_attempt_identity(
            schema_ref=self._schema_refs.stream_inference_attempt,
            stream_inference_logical_id=logical_identity.stream_inference_logical_id,
            attempt_number=1,
        )
        intent = create_stream_inference_intent(
            schema_ref=self._schema_refs.stream_inference_intent,
            window_subject=plan.subject,
            logical_identity=logical_identity,
            attempt_identity=attempt_identity,
            input_plan=StreamInputPlanReference(
                input_plan_id=inference_plan.input_plan_id,
                input_plan_semantic_sha256=inference_plan.input_plan_semantic_sha256,
                exact_artifact_ref=inference_plan_ref,
            ),
            provider_idempotency_key=(
                f"{LOCAL_STREAM_MOCK_PROVIDER_VERSION}:"
                f"{logical_identity.stream_inference_logical_id}"
            ),
            dispatch_policy_version=self._mock_policy,
            created_at=plan.created_at,
        )
        intent_ref = self._artifacts.put(
            canonical_json_bytes(intent),
            self._schema_refs.stream_inference_intent,
        )
        accepted_call = create_stream_accepted_call_evidence(
            schema_ref=self._schema_refs.stream_accepted_call,
            intent_ref=reference_stream_inference_intent(intent, intent_ref),
            status=InferenceStatus.SUCCEEDED,
            provider_exchange_ref=semantic_ref,
            output_valid=True,
            usage=ModelInferenceUsage(
                input_frames=sum(
                    slot.kind != "ABSENCE"
                    for slot in declaration.ordered_six_slot_segment_or_explicit_absence_closure
                ),
                input_images=0,
            ),
            latency_ms=0,
            completed_at=plan.created_at,
            provider_request_id=(
                f"{LOCAL_STREAM_MOCK_PROVIDER_VERSION}:{attempt_identity.inference_attempt_id}"
            ),
            output_semantic_sha256=semantic_evidence.semantic_sha256,
            normalized_output_ref=semantic_ref,
        )
        accepted_call_ref = self._artifacts.put(
            canonical_json_bytes(accepted_call),
            self._schema_refs.stream_accepted_call,
        )
        inference_terminal = create_stream_inference_terminal(
            schema_ref=self._schema_refs.stream_inference_terminal,
            logical_identity=logical_identity,
            attempt_identity=attempt_identity,
            intent_ref=reference_stream_inference_intent(intent, intent_ref),
            accepted_call_ref=reference_stream_accepted_call(
                accepted_call,
                accepted_call_ref,
            ),
            status=InferenceStatus.SUCCEEDED,
            terminal_policy_version=self._terminal_policy,
            completed_at=plan.created_at,
        )
        inference_terminal_ref = self._artifacts.put(
            canonical_json_bytes(inference_terminal),
            self._schema_refs.stream_inference_terminal,
        )
        outcome = TerminalOutcome.SUCCEEDED if has_media else TerminalOutcome.NO_EVENTS
        result = create_stream_window_result(
            schema_ref=self._schema_refs.stream_window_result,
            window_subject=plan.subject,
            purpose=self._window_purpose,
            terminal_outcome=outcome,
            accepted_terminals=(
                reference_stream_inference_terminal(
                    inference_terminal,
                    inference_terminal_ref,
                ),
            ),
            result_semantic_evidence_sha256=semantic_evidence.semantic_sha256,
            result_evidence_ref=semantic_ref,
            reduction_policy_version=LOCAL_STREAM_CAUSAL_REDUCTION_POLICY_VERSION,
            created_at=plan.created_at,
        )
        result_ref = self._artifacts.put(
            canonical_json_bytes(result),
            self._schema_refs.stream_window_result,
        )
        return _PreparedStreamWorkCompletion(
            terminal_evidence=StreamTerminalEvidence(
                outcome=outcome,
                evidence_ref=result_ref,
                terminal_policy_version=self._terminal_policy,
                completed_at=plan.created_at,
            ),
            window_result=result,
            causal_evidence=PreparedWindowReductionEvidence(
                inference_plan=inference_plan,
                semantic_evidence=semantic_evidence,
                intent=intent,
                accepted_call=accepted_call,
                inference_terminal=inference_terminal,
            ),
        )

    def _causal_upstream_stage_evidence(
        self,
        plan: StreamWorkItemPlan,
        plans_by_key: dict[str, StreamWorkItemPlan],
        terminal_refs_by_key: dict[str, ArtifactEvidenceRef],
    ) -> tuple[LocalStreamStageEvidenceReference, ...]:
        by_stage: dict[StreamStage, LocalStreamStageEvidenceReference] = {}
        for dependency in plan.ordered_dependencies:
            upstream_plan = plans_by_key.get(dependency.upstream_work_logical_key)
            if upstream_plan is None:
                raise LocalStreamFinalizationError(
                    "causal window reduction lacks an upstream work plan"
                )
            reference = terminal_refs_by_key.get(upstream_plan.work_logical_key)
            if reference is None:
                raise LocalStreamFinalizationError(
                    "causal window reduction lacks terminal upstream evidence"
                )
            expected_provider_task = _PRE_EOS_PROVIDER_TASK_BY_STAGE.get(upstream_plan.stage)
            if (
                expected_provider_task is not None
                and self._schema_refs.model_inference is not None
                and reference.schema_ref == self._schema_refs.model_inference
            ):
                try:
                    inference = ProviderModelInference.model_validate_json(
                        self._artifacts.read(reference),
                        strict=True,
                    )
                except ValueError as error:
                    raise LocalStreamFinalizationError(
                        "causal upstream provider evidence is not a model inference"
                    ) from error
                if (
                    inference.stage is not expected_provider_task
                    or inference.status is not ProviderInferenceStatus.SUCCEEDED
                    or not inference.output_valid
                ):
                    raise LocalStreamFinalizationError(
                        "causal upstream model inference is not a valid successful stage terminal"
                    )
                evidence_stage = upstream_plan.stage
                evidence_key = upstream_plan.work_logical_key
                evidence_sha = semantic_sha256(inference.model_dump(mode="json"))
            else:
                try:
                    receipt = LocalStreamWorkReceipt.model_validate_json(
                        self._artifacts.read(reference),
                        strict=True,
                    )
                except ValueError as error:
                    raise LocalStreamFinalizationError(
                        "causal upstream evidence is not a local stream receipt"
                    ) from error
                if (
                    receipt.executor_policy_version != self._mock_policy
                    or receipt.work_item_id != upstream_plan.work_item_id
                    or receipt.work_logical_key != upstream_plan.work_logical_key
                    or receipt.stage is not upstream_plan.stage
                ):
                    raise LocalStreamFinalizationError(
                        "causal upstream receipt conflicts with its durable work plan"
                    )
                evidence_stage = receipt.stage
                evidence_key = receipt.work_logical_key
                evidence_sha = semantic_sha256(receipt.model_dump(mode="json"))
            if evidence_stage in by_stage:
                raise LocalStreamFinalizationError(
                    "causal window reduction has duplicate upstream stages"
                )
            by_stage[evidence_stage] = LocalStreamStageEvidenceReference(
                stage=evidence_stage,
                work_logical_key=evidence_key,
                terminal_evidence_ref=reference,
                evidence_semantic_sha256=evidence_sha,
            )
        stages = (
            StreamStage.WINDOW,
            StreamStage.QA_COARSE,
            StreamStage.QA_DENSE,
            StreamStage.EVENT_PROPOSAL,
        )
        if tuple(stage for stage in stages if stage in by_stage) != stages:
            raise LocalStreamFinalizationError(
                "causal window reduction lacks its complete upstream stage closure"
            )
        return tuple(by_stage[stage] for stage in stages)

    def _receipt(
        self,
        plan: StreamWorkItemPlan,
        plans_by_key: dict[str, StreamWorkItemPlan],
        terminal_refs_by_key: dict[str, ArtifactEvidenceRef],
    ) -> LocalStreamWorkReceipt:
        upstream_refs: list[str] = []
        for dependency in plan.ordered_dependencies:
            upstream_plan = plans_by_key[dependency.upstream_work_logical_key]
            reference = terminal_refs_by_key.get(upstream_plan.work_logical_key)
            if reference is None:
                raise LocalStreamFinalizationError(
                    "claimable stream work lacks a terminal upstream dependency"
                )
            upstream_refs.append(reference.exact_sha256)
        return LocalStreamWorkReceipt(
            schema_ref=self._schema_refs.local_work_receipt,
            executor_policy_version=self._mock_policy,
            plan_key=self._scheduler.plan_key,
            work_item_id=plan.work_item_id,
            work_logical_key=plan.work_logical_key,
            stage=plan.stage,
            subject_type=plan.subject.subject_type,
            subject_key=plan.subject.subject_key,
            subject_semantic_sha256=plan.subject.subject_semantic_sha256,
            input_semantic_sha256=plan.input_semantic_sha256,
            config_semantic_sha256=plan.config_semantic_sha256,
            ordered_upstream_terminal_exact_sha256_values=tuple(upstream_refs),
            result=(
                "ABSTAINED_NO_PROVIDER"
                if plan.stage is StreamStage.WINDOW_REDUCTION
                else "LOCAL_CONFORMANCE_STAGE_COMPLETE"
            ),
        )

    def _load_window_results(
        self,
        plans: tuple[StreamWorkItemPlan, ...],
        items_by_id: dict[str, StreamWorkItem],
    ) -> tuple[StreamWindowResult, ...]:
        by_window: dict[str, StreamWindowResult] = {}
        for plan in plans:
            if plan.stage is not StreamStage.WINDOW_REDUCTION:
                continue
            item = items_by_id.get(plan.work_item_id)
            if item is None:
                raise LocalStreamFinalizationError(
                    "window reduction lacks its batched execution snapshot"
                )
            reference = item.terminal_evidence_ref
            if reference is None:
                raise LocalStreamFinalizationError(
                    "window reduction completed without an exact result reference"
                )
            payload = self._artifacts.read(reference)
            try:
                result = StreamWindowResult.model_validate_json(payload, strict=True)
            except ValueError as error:
                raise LocalStreamFinalizationError(
                    "window reduction evidence is not a StreamWindowResult"
                ) from error
            if (
                result.window_subject != plan.subject
                or result.terminal_outcome
                not in {TerminalOutcome.SUCCEEDED, TerminalOutcome.NO_EVENTS}
                or result.reduction_policy_version != LOCAL_STREAM_CAUSAL_REDUCTION_POLICY_VERSION
                or len(result.accepted_terminals) != 1
            ):
                raise LocalStreamFinalizationError(
                    "window result is not the expected local causal reduction"
                )
            try:
                semantic_evidence = LocalStreamWindowSemanticEvidenceV2.model_validate_json(
                    self._artifacts.read(result.result_evidence_ref),
                    strict=True,
                )
                terminal_reference = result.accepted_terminals[0]
                inference_terminal = StreamInferenceTerminal.model_validate_json(
                    self._artifacts.read(terminal_reference.artifact_ref),
                    strict=True,
                )
                intent = StreamInferenceIntent.model_validate_json(
                    self._artifacts.read(inference_terminal.intent_ref.artifact_ref),
                    strict=True,
                )
            except ValueError as error:
                raise LocalStreamFinalizationError(
                    "window result causal artifacts are invalid"
                ) from error
            expected_status = (
                "PROPOSED" if result.terminal_outcome is TerminalOutcome.SUCCEEDED else "NO_EVENTS"
            )
            if intent.dispatch_policy_version != self._mock_policy:
                raise LocalStreamFinalizationError(
                    "window result conflicts with the local executor policy"
                )
            if (
                semantic_evidence.schema_ref != self._schema_refs.window_semantic_evidence_v2
                or semantic_evidence.plan_key != self._scheduler.plan_key
                or semantic_evidence.window_key != plan.subject.subject_key
                or semantic_evidence.window_semantic_sha256 != plan.subject.subject_semantic_sha256
                or semantic_evidence.semantic_status != expected_status
                or semantic_evidence.semantic_sha256 != result.result_semantic_evidence_sha256
                or inference_terminal.schema_ref != self._schema_refs.stream_inference_terminal
                or inference_terminal.terminal_semantic_sha256
                != terminal_reference.terminal_semantic_sha256
                or inference_terminal.logical_identity.window_key != plan.subject.subject_key
                or inference_terminal.logical_identity.window_semantic_sha256
                != plan.subject.subject_semantic_sha256
                or inference_terminal.logical_identity.purpose is not result.purpose
                or inference_terminal.status is not InferenceStatus.SUCCEEDED
                or intent.schema_ref != self._schema_refs.stream_inference_intent
                or intent.intent_semantic_sha256
                != inference_terminal.intent_ref.intent_semantic_sha256
            ):
                raise LocalStreamFinalizationError(
                    "window result causal artifacts conflict with durable identities"
                )
            if plan.subject.subject_key in by_window:
                raise LocalStreamFinalizationError(
                    "window reduction produced duplicate durable results"
                )
            by_window[plan.subject.subject_key] = result
        ordered: list[StreamWindowResult] = []
        for declaration in self._scheduler.declarations():
            ordered_result = by_window.get(declaration.window_key)
            if ordered_result is None:
                raise LocalStreamFinalizationError(
                    "ordered expected window lacks a local reduction result"
                )
            ordered.append(ordered_result)
        return tuple(ordered)

    def _create_finalization(
        self,
        results: tuple[StreamWindowResult, ...],
        closure: WindowTerminalClosure,
        final_plan: StreamWorkItemPlan,
    ) -> RecordingFinalizationMap:
        final_recording = self._final_recording
        if final_recording is None:
            raise LocalStreamFinalizationError("recording facts are required for EOS finalization")
        seal = self._scheduler.expected_plan_seal()
        if seal is None:
            raise LocalStreamFinalizationError("expected plan seal disappeared")
        export = self._scheduler.export_barrier()
        manifest_digest = export.export_manifest_semantic_sha256
        if manifest_digest is None:
            raise LocalStreamFinalizationError("export barrier lacks manifest identity")
        source = final_plan.source_subject
        mappings = tuple(
            FinalizationSubjectMapping(
                incremental_subject_type=StreamSubjectType.INCREMENTAL_WINDOW,
                incremental_subject_key=result.window_subject.subject_key,
                incremental_subject_semantic_sha256=(result.window_subject.subject_semantic_sha256),
                final_subject_type="FINAL_WINDOW",
                final_subject_key=f"final-window-v1:{final_digest}",
                final_subject_semantic_sha256=final_digest,
            )
            for result in results
            for final_digest in (
                semantic_sha256(
                    {
                        "version": LOCAL_STREAM_FINAL_MAPPING_VERSION,
                        "final_recording_identity": (final_recording.final_recording_identity),
                        "window_result_semantic_sha256": (result.window_result_semantic_sha256),
                    }
                ),
            )
        )
        return create_recording_finalization_map(
            schema_ref=self._schema_refs.recording_finalization,
            capture_scope_key=source.capture_scope_key,
            capture_scope_digest=source.capture_scope_digest,
            final_source_subject_type=final_recording.final_source_subject_type,
            final_source_subject_id=final_recording.final_source_subject_id,
            final_source_exact_sha256=final_recording.final_source_exact_sha256,
            final_recording_identity=final_recording.final_recording_identity,
            final_duration_ns=final_recording.final_duration_ns,
            final_mapping_semantic_sha256=seal.mapping_closure_semantic_sha256,
            final_alignment_semantic_sha256=(seal.clock_or_alignment_closure_semantic_sha256),
            expected_plan_seal_semantic_sha256=seal.seal_semantic_sha256,
            window_terminal_closure_semantic_sha256=closure.terminal_closure_digest,
            export_manifest_semantic_sha256=manifest_digest,
            ordered_subject_mappings=mappings,
        )

    def _finalization_plan(
        self,
        all_plans: tuple[StreamWorkItemPlan, ...],
    ) -> StreamWorkItemPlan:
        plans = tuple(plan for plan in all_plans if plan.stage is StreamStage.FINALIZATION)
        if len(plans) != 1:
            raise LocalStreamFinalizationError(
                "sealed stream graph must contain exactly one finalization plan"
            )
        return plans[0]

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise LocalStreamFinalizationError("local stream clock must be timezone-aware")
        return now.astimezone(UTC)


def create_local_stream_canonical_truth(
    result: CanonicalOfflineRunResult,
) -> LocalStreamCanonicalTruth:
    """Project the existing rich canonical mock result without adding model claims."""

    qa = result.qa_completion_result
    proposals = result.event_proposal_result
    candidates = result.candidate_reduction_result
    if qa is None or qa.final_aggregate is None or proposals is None or candidates is None:
        raise LocalStreamFinalizationError(
            "canonical result lacks six-camera QA, proposal, or candidate closure"
        )
    qa_references = tuple(
        LocalStreamQaCameraReference(
            camera_id=camera.camera_id,
            semantic_sha256=semantic_sha256(camera.model_dump(mode="json")),
        )
        for camera in qa.final_aggregate.camera_results
    )

    def reference(
        *,
        kind: Literal[
            "EVENT_PROPOSAL",
            "CANDIDATE",
            "ACTION",
            "BOUNDARY",
            "HYPOTHESIS",
        ],
        logical_key: str,
        digest: str,
        interval: object,
    ) -> LocalStreamSemanticIntervalReference:
        from robata.contracts.common import NanosecondInterval

        return LocalStreamSemanticIntervalReference(
            kind=kind,
            logical_key=logical_key,
            semantic_sha256=digest,
            interval=NanosecondInterval.model_validate(interval, strict=True),
        )

    proposal_references = tuple(
        reference(
            kind="EVENT_PROPOSAL",
            logical_key=proposal.source_proposal_logical_key,
            digest=semantic_sha256(proposal.model_dump(mode="json")),
            interval=proposal.interval,
        )
        for proposal in proposals.proposals
    )
    candidate_references = tuple(
        reference(
            kind="CANDIDATE",
            logical_key=candidate.candidate_logical_key,
            digest=candidate_event_digest,
            interval=candidate.effective_interval,
        )
        for candidate in candidates.candidates
        for candidate_event_digest in (candidate.candidate_logical_key.rsplit(":", 1)[-1],)
    )
    fusion = result.provisional_fusion_result
    action_references = tuple(
        reference(
            kind="ACTION",
            logical_key=action.logical_key,
            digest=action.semantic_sha256,
            interval=action.coarse_interval,
        )
        for action in (() if fusion is None else fusion.actions)
    )
    boundary_references = tuple(
        reference(
            kind="BOUNDARY",
            logical_key=execution.result.logical_key,
            digest=execution.result.semantic_sha256,
            interval=execution.result.refined_interval or execution.result.coarse_interval,
        )
        for execution in result.boundary_refinement_executions
    )
    hypothesis_references = tuple(
        reference(
            kind="HYPOTHESIS",
            logical_key=hypothesis.event_hypothesis_logical_key,
            digest=hypothesis.semantic_sha256,
            interval=hypothesis.effective_interval,
        )
        for hypothesis in result.hypotheses
    )
    sort_key = lambda item: (  # noqa: E731
        item.interval.start_ns,
        item.interval.end_ns,
        item.kind,
        item.logical_key,
        item.semantic_sha256,
    )
    decision = result.output_decision
    if decision is None:
        if result.status.value != "NO_EVENTS":
            raise LocalStreamFinalizationError(
                "canonical result lacks its recording output decision"
            )
        output_decision: Literal["ADMITTED", "NO_EVENTS", "ABSTAINED"] = "NO_EVENTS"
        output_decision_digest = None
    else:
        output_decision = decision.decision
        output_decision_digest = decision.semantic_sha256
    return LocalStreamCanonicalTruth(
        six_camera_qa_semantic_sha256=qa.semantic_sha256,
        qa_camera_references=qa_references,
        event_proposal_result_semantic_sha256=proposals.semantic_sha256,
        proposal_references=tuple(sorted(proposal_references, key=sort_key)),
        candidate_reduction_semantic_sha256=candidates.semantic_sha256,
        candidate_references=tuple(sorted(candidate_references, key=sort_key)),
        provisional_fusion_semantic_sha256=None if fusion is None else fusion.semantic_sha256,
        action_references=tuple(sorted(action_references, key=sort_key)),
        boundary_closure_semantic_sha256=semantic_sha256(
            {
                "projection_version": "local-stream-boundary-closure-v1",
                "ordered_boundary_semantic_sha256_values": [
                    execution.result.semantic_sha256
                    for execution in result.boundary_refinement_executions
                ],
            }
        ),
        boundary_references=tuple(sorted(boundary_references, key=sort_key)),
        output_decision=output_decision,
        output_decision_semantic_sha256=output_decision_digest,
        hypothesis_references=tuple(sorted(hypothesis_references, key=sort_key)),
    )


def load_completed_local_stream_finalization(
    *,
    scheduler: DurableStreamWindowScheduler,
    artifact_root: str | Path,
) -> RecordingFinalizationMap:
    """Verify and load the exact finalization artifact on completion replay."""

    plans = tuple(plan for plan in scheduler.work_plans() if plan.stage is StreamStage.FINALIZATION)
    if len(plans) != 1:
        raise LocalStreamFinalizationError(
            "sealed stream graph must contain exactly one finalization plan"
        )
    item = scheduler.get(plans[0].work_item_id)
    reference = item.terminal_evidence_ref
    if item.state is not StreamWorkItemState.SUCCEEDED or reference is None:
        raise LocalStreamFinalizationError(
            "primary completion requires a successful stream finalization artifact"
        )
    payload = _ExactLocalArtifactStore(Path(artifact_root)).read(reference)
    try:
        finalization = RecordingFinalizationMap.model_validate_json(payload, strict=True)
    except ValueError as error:
        raise LocalStreamFinalizationError(
            "finalization work evidence is not a RecordingFinalizationMap"
        ) from error
    closure = scheduler.terminal_closure()
    if (
        closure is None
        or finalization.capture_scope_digest != plans[0].source_subject.capture_scope_digest
        or finalization.window_terminal_closure_semantic_sha256 != closure.terminal_closure_digest
        or finalization.export_manifest_semantic_sha256
        != scheduler.export_barrier().export_manifest_semantic_sha256
    ):
        raise LocalStreamFinalizationError(
            "recording finalization does not bind the recovered stream barriers"
        )
    return finalization


def load_completed_local_stream_recording_result(
    *,
    scheduler: DurableStreamWindowScheduler,
    artifact_root: str | Path,
    schema_ref: SchemaRef,
    expected_exact_sha256: str | None = None,
    expected_byte_count: int | None = None,
) -> tuple[
    LocalStreamRecordingResult
    | LocalStreamRecordingResultV2
    | LocalStreamRecordingResultV3
    | LocalStreamRecordingResultV4,
    ArtifactEvidenceRef,
]:
    """Rebuild and verify the exact recording reduction from durable stream facts."""

    if (
        schema_ref.schema_id != LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID
        or schema_ref.version
        not in {
            LOCAL_STREAM_RECORDING_RESULT_SCHEMA_VERSION,
            LOCAL_STREAM_RECORDING_RESULT_V2_SCHEMA_VERSION,
            LOCAL_STREAM_RECORDING_RESULT_V3_SCHEMA_VERSION,
            LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION,
        }
    ):
        raise LocalStreamFinalizationError(
            "completed stream recording result uses an unsupported schema"
        )
    if (expected_exact_sha256 is None) != (expected_byte_count is None):
        raise LocalStreamFinalizationError(
            "exact recording evidence digest and byte count must be supplied together"
        )
    root = Path(artifact_root)
    artifacts = _ExactLocalArtifactStore(root)
    finalization = load_completed_local_stream_finalization(
        scheduler=scheduler,
        artifact_root=root,
    )
    closure = scheduler.terminal_closure()
    if closure is None:
        raise LocalStreamFinalizationError(
            "completed stream finalization lacks its terminal closure"
        )
    reduction_plans = {
        plan.subject.subject_key: plan
        for plan in scheduler.work_plans()
        if plan.stage is StreamStage.WINDOW_REDUCTION
    }
    declarations = scheduler.declarations()
    ordered_results: list[StreamWindowResult] = []
    for declaration in declarations:
        plan = reduction_plans.get(declaration.window_key)
        if plan is None:
            raise LocalStreamFinalizationError(
                "completed stream finalization lacks a window reduction plan"
            )
        item = scheduler.get(plan.work_item_id)
        reference = item.terminal_evidence_ref
        if item.state not in TERMINAL_STREAM_WORK_STATES or reference is None:
            raise LocalStreamFinalizationError(
                "completed stream finalization has nonterminal window reduction"
            )
        try:
            result = StreamWindowResult.model_validate_json(
                artifacts.read(reference),
                strict=True,
            )
        except ValueError as error:
            raise LocalStreamFinalizationError(
                "window reduction evidence is not a StreamWindowResult"
            ) from error
        if (
            result.schema_ref != reference.schema_ref
            or result.window_subject.subject_key != declaration.window_key
            or result.window_subject.subject_semantic_sha256 != declaration.window_semantic_sha256
        ):
            raise LocalStreamFinalizationError(
                "window reduction evidence conflicts with its expected declaration"
            )
        ordered_results.append(result)
    if schema_ref.version == LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION:
        v4_semantic_evidence: list[LocalStreamWindowSemanticEvidenceV2] = []
        for result in ordered_results:
            try:
                v4_evidence = LocalStreamWindowSemanticEvidenceV2.model_validate_json(
                    artifacts.read(result.result_evidence_ref),
                    strict=True,
                )
            except ValueError as error:
                raise LocalStreamFinalizationError(
                    "v4 recording result references invalid causal window evidence"
                ) from error
            v4_semantic_evidence.append(v4_evidence)

        persisted_v4: LocalStreamRecordingResultV4 | None = None
        v4_committed_payload: bytes | None = None
        v4_committed_ref: ArtifactEvidenceRef | None = None
        v4_source_origin: int | None = None
        v4_canonical_interval: NanosecondInterval | None = None
        if expected_exact_sha256 is not None and expected_byte_count is not None:
            v4_committed_ref = ArtifactEvidenceRef(
                artifact_id=str(uuid5(_ARTIFACT_NAMESPACE, expected_exact_sha256)),
                exact_sha256=expected_exact_sha256,
                byte_count=expected_byte_count,
                media_type="application/json",
                schema_ref=schema_ref,
            )
            v4_committed_payload = artifacts.read(v4_committed_ref)
            try:
                persisted_v4 = LocalStreamRecordingResultV4.model_validate_json(
                    v4_committed_payload,
                    strict=True,
                )
            except ValueError as error:
                raise LocalStreamFinalizationError(
                    "committed stream recording evidence is not a v4 result"
                ) from error
            v4_source_origin = persisted_v4.source_timeline_origin_ns
            v4_canonical_interval = persisted_v4.canonical_requested_interval

        rebuilt_v4 = create_local_stream_recording_result_v4(
            schema_ref=schema_ref,
            window_results=tuple(ordered_results),
            window_semantic_evidence=tuple(v4_semantic_evidence),
            terminal_closure=closure,
            recording_finalization=finalization,
            source_timeline_origin_ns=v4_source_origin,
            canonical_requested_interval=v4_canonical_interval,
        )
        rebuilt_payload = canonical_json_bytes(rebuilt_v4)
        if persisted_v4 is not None and (
            rebuilt_v4 != persisted_v4 or rebuilt_payload != v4_committed_payload
        ):
            raise LocalStreamFinalizationError(
                "v4 recording replay does not reproduce committed exact bytes"
            )
        if v4_committed_ref is None:
            v4_committed_ref = artifacts.put(rebuilt_payload, schema_ref)
        return rebuilt_v4, v4_committed_ref
    if schema_ref.version == LOCAL_STREAM_RECORDING_RESULT_V3_SCHEMA_VERSION:
        v3_semantic_evidence: list[LocalStreamWindowSemanticEvidenceV2] = []
        for result in ordered_results:
            try:
                v3_evidence = LocalStreamWindowSemanticEvidenceV2.model_validate_json(
                    artifacts.read(result.result_evidence_ref),
                    strict=True,
                )
            except ValueError as error:
                raise LocalStreamFinalizationError(
                    "v3 recording result references invalid causal window evidence"
                ) from error
            v3_semantic_evidence.append(v3_evidence)

        persisted_v3: LocalStreamRecordingResultV3 | None = None
        v3_committed_payload: bytes | None = None
        v3_committed_ref: ArtifactEvidenceRef | None = None
        v3_source_origin: int | None = None
        v3_canonical_interval: NanosecondInterval | None = None
        if expected_exact_sha256 is not None and expected_byte_count is not None:
            v3_committed_ref = ArtifactEvidenceRef(
                artifact_id=str(uuid5(_ARTIFACT_NAMESPACE, expected_exact_sha256)),
                exact_sha256=expected_exact_sha256,
                byte_count=expected_byte_count,
                media_type="application/json",
                schema_ref=schema_ref,
            )
            v3_committed_payload = artifacts.read(v3_committed_ref)
            try:
                persisted_v3 = LocalStreamRecordingResultV3.model_validate_json(
                    v3_committed_payload,
                    strict=True,
                )
            except ValueError as error:
                raise LocalStreamFinalizationError(
                    "committed stream recording evidence is not a v3 result"
                ) from error
            v3_source_origin = persisted_v3.source_timeline_origin_ns
            v3_canonical_interval = persisted_v3.canonical_requested_interval

        rebuilt_v3 = create_local_stream_recording_result_v3(
            schema_ref=schema_ref,
            window_results=tuple(ordered_results),
            window_semantic_evidence=tuple(v3_semantic_evidence),
            terminal_closure=closure,
            recording_finalization=finalization,
            source_timeline_origin_ns=v3_source_origin,
            canonical_requested_interval=v3_canonical_interval,
        )
        rebuilt_payload = canonical_json_bytes(rebuilt_v3)
        if persisted_v3 is not None and (
            rebuilt_v3 != persisted_v3 or rebuilt_payload != v3_committed_payload
        ):
            raise LocalStreamFinalizationError(
                "v3 recording replay does not reproduce committed exact bytes"
            )
        if v3_committed_ref is None:
            v3_committed_ref = artifacts.put(rebuilt_payload, schema_ref)
        return rebuilt_v3, v3_committed_ref
    if schema_ref.version == "2.0.0":
        if expected_exact_sha256 is None or expected_byte_count is None:
            raise LocalStreamFinalizationError(
                "v2 recording replay requires the committed exact artifact reference"
            )
        v2_committed_ref = ArtifactEvidenceRef(
            artifact_id=str(uuid5(_ARTIFACT_NAMESPACE, expected_exact_sha256)),
            exact_sha256=expected_exact_sha256,
            byte_count=expected_byte_count,
            media_type="application/json",
            schema_ref=schema_ref,
        )
        v2_committed_payload = artifacts.read(v2_committed_ref)
        try:
            persisted_v2 = LocalStreamRecordingResultV2.model_validate_json(
                v2_committed_payload,
                strict=True,
            )
        except ValueError as error:
            raise LocalStreamFinalizationError(
                "committed stream recording evidence is not a v2 result"
            ) from error
        v2_semantic_evidence: list[
            tuple[LocalStreamWindowSemanticEvidence, ArtifactEvidenceRef]
        ] = []
        for reference in persisted_v2.ordered_window_semantic_evidence_refs:
            try:
                v2_evidence = LocalStreamWindowSemanticEvidence.model_validate_json(
                    artifacts.read(reference),
                    strict=True,
                )
            except ValueError as error:
                raise LocalStreamFinalizationError(
                    "v2 recording result references invalid window semantic evidence"
                ) from error
            v2_semantic_evidence.append((v2_evidence, reference))
        v2_source_origin, _declared_interval = _declared_stream_timeline(declarations)
        relative_effective_end = max(
            declaration.effective_interval.end_ns - v2_source_origin for declaration in declarations
        )
        v2_canonical_interval = NanosecondInterval(
            start_ns=0,
            end_ns=min(
                finalization.final_duration_ns,
                relative_effective_end,
            ),
        )
        if len(v2_semantic_evidence) != len(declarations):
            raise LocalStreamFinalizationError(
                "v2 recording result does not cover every declared window"
            )
        for declaration, (v2_evidence, _reference) in zip(
            declarations,
            v2_semantic_evidence,
            strict=True,
        ):
            expected_interval = _normalize_stream_effective_interval(
                declaration.effective_interval,
                source_timeline_origin_ns=v2_source_origin,
                canonical_requested_interval=v2_canonical_interval,
            )
            if (
                v2_evidence.expected_ordinal != declaration.ordinal
                or v2_evidence.window_key != declaration.window_key
                or v2_evidence.window_semantic_sha256 != declaration.window_semantic_sha256
                or v2_evidence.effective_interval != expected_interval
            ):
                raise LocalStreamFinalizationError(
                    "v2 window semantic evidence is not recording-relative to its declaration"
                )
        rebuilt_v2 = create_local_stream_recording_result_v2(
            schema_ref=schema_ref,
            window_results=tuple(ordered_results),
            window_semantic_evidence=tuple(v2_semantic_evidence),
            terminal_closure=closure,
            recording_finalization=finalization,
        )
        if rebuilt_v2 != persisted_v2 or canonical_json_bytes(rebuilt_v2) != v2_committed_payload:
            raise LocalStreamFinalizationError(
                "v2 recording replay does not reproduce committed exact bytes"
            )
        return rebuilt_v2, v2_committed_ref

    recording_result = create_local_stream_recording_result(
        schema_ref=schema_ref,
        window_results=tuple(ordered_results),
        terminal_closure=closure,
        recording_finalization=finalization,
    )
    reference = artifacts.put(
        canonical_json_bytes(recording_result),
        schema_ref,
    )
    if expected_exact_sha256 is not None and (
        reference.exact_sha256 != expected_exact_sha256
        or reference.byte_count != expected_byte_count
    ):
        raise LocalStreamFinalizationError(
            "v1 recording replay does not reproduce committed exact bytes"
        )
    return recording_result, reference


__all__ = [
    "LOCAL_CONFORMANCE_EVIDENCE_CLASS",
    "LOCAL_STREAM_MOCK_EXECUTOR_POLICY_VERSION",
    "LOCAL_STREAM_REDUCTION_POLICY_VERSION",
    "LOCAL_STREAM_WORK_RECEIPT_SCHEMA_ID",
    "LOCAL_STREAM_WORK_RECEIPT_SCHEMA_VERSION",
    "FinalRecordingFacts",
    "LocalConformanceStreamFinalizer",
    "LocalStreamFinalizationError",
    "LocalStreamFinalizationOutcome",
    "LocalStreamFinalizationSchemaRefs",
    "LocalStreamWorkReceipt",
    "create_local_stream_canonical_truth",
    "load_completed_local_stream_finalization",
    "load_completed_local_stream_recording_result",
]
