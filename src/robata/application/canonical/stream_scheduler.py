"""Durable local composition for the pre-EOS stream work graph.

Expected declarations are committed before an internal execution projection is
sent to the local scheduler. The projection is not stream Wire evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Self
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, ValidationError

from robata.adapters.sqlite_stream_work_ledger import (
    NewStreamWindow,
    NewStreamWorkPlan,
    SQLiteStreamWorkLedger,
    SQLiteStreamWorkLedgerFairnessThrottle,
    StoredExpectedWindow,
    StoredStreamBackpressureController,
    StoredStreamWorkPlan,
)
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler, WorkFenceError
from robata.application.canonical.bounded_media import (
    BoundedWindowPlan,
    PlannerEmission,
    PlannerFinish,
    SinglePassPlanningSink,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    PreEosCaptureSubjectRef,
    StreamStage,
    StreamSubjectRef,
    StreamSubjectType,
    TerminalOutcome,
)
from robata.contracts.stream_finalization import (
    WindowTerminalClosure,
    WindowTerminalMember,
    create_window_terminal_closure,
    create_window_terminal_member,
)
from robata.contracts.stream_planning import (
    ExpectedWindowDeclaration,
    ExpectedWindowPlan,
    ExpectedWindowPlanSeal,
    StreamWorkDependency,
    StreamWorkItemPlan,
    create_expected_window_declaration,
    create_expected_window_plan_seal,
    create_stream_work_item_plan,
)
from robata.contracts.stream_window import IncrementalWindow
from robata.queue.backpressure import (
    AdmissionDecision,
    BackpressureConfig,
    BackpressureController,
    BackpressureControllerState,
    BackpressureRuntimeSignals,
    PressureClass,
    QueueMetrics,
)
from robata.queue.models import (
    TERMINAL_WORK_STATES,
    WorkDependency,
    WorkItem,
    WorkItemPlan,
    WorkItemState,
    WorkItemSubjectType,
    WorkLease,
)
from robata.queue.stage import DependencyCriticality, Stage
from robata.queue.stream_models import (
    StreamTerminalEvidence,
    StreamWorkItem,
    StreamWorkItemState,
    StreamWorkLease,
    StreamWorkLeaseClaim,
)

INTERNAL_STREAM_EXECUTION_PROJECTION_VERSION: Final = "internal-stream-execution-projection-v1"
STREAM_WINDOW_DAG_POLICY_VERSION: Final = "stream-window-dag-v4"
WATERMARK_SOURCE_FACTS_PROJECTION_VERSION: Final = "bounded-watermark-source-facts-v1"
PLANNER_EOS_PROJECTION_VERSION: Final = "bounded-planner-eos-v2"

_INTERNAL_EXECUTION_NAMESPACE: Final = UUID("d13666a6-112f-5ab7-9f70-240de5557fe1")
_INTERNAL_DEPENDENCY_NAMESPACE: Final = UUID("67d7e565-bc31-55e1-9168-57331a0c6eb3")

_WINDOW_DAG_TOPOLOGY: Final = (
    (StreamStage.WINDOW, ()),
    (StreamStage.QA_COARSE, (StreamStage.WINDOW,)),
    (StreamStage.QA_DENSE, (StreamStage.QA_COARSE,)),
    (
        StreamStage.EVENT_PROPOSAL,
        (StreamStage.QA_COARSE, StreamStage.QA_DENSE),
    ),
    (
        StreamStage.WINDOW_REDUCTION,
        (
            StreamStage.WINDOW,
            StreamStage.QA_COARSE,
            StreamStage.QA_DENSE,
            StreamStage.EVENT_PROPOSAL,
        ),
    ),
)
_WINDOW_DAG_STAGES: Final = tuple(stage for stage, _dependencies in _WINDOW_DAG_TOPOLOGY)
_BACKPRESSURE_CONTROLLER_KEY: Final = "stream-window-admission"

# Local-conformance scheduling budgets. They are operational fields rather than
# logical-identity inputs and remain explicitly unqualified for production SLOs.
_LOCAL_STAGE_SLA_STEP_SECONDS: Final = 5 * 60
DEFAULT_STREAM_BACKPRESSURE_CONFIG: Final = BackpressureConfig(
    version="local-stream-backpressure-v1",
    queue_depth_threshold=256,
    oldest_age_threshold_ms=30_000,
    backlog_slope_threshold=128.0,
)

_STAGE_EXECUTION_PROJECTION: Final = {
    StreamStage.SEGMENT: Stage.MCAP_INGEST,
    StreamStage.WINDOW: Stage.QA_COARSE_PLAN,
    StreamStage.QA_COARSE: Stage.QWEN_QA_COARSE,
    StreamStage.QA_DENSE: Stage.QWEN_QA_DENSE,
    StreamStage.EVENT_PROPOSAL: Stage.QWEN_EVENT_PROPOSAL,
    StreamStage.ACTION_DENSE: Stage.QWEN_ACTION_EVIDENCE,
    StreamStage.BOUNDARY_REFINEMENT: Stage.QWEN_BOUNDARY,
    StreamStage.WINDOW_REDUCTION: Stage.QA_AGGREGATE,
    StreamStage.FINALIZATION: Stage.ACTION_PUBLISH,
}

_TERMINAL_STATE_BY_OUTCOME: Final = {
    TerminalOutcome.SUCCEEDED: StreamWorkItemState.SUCCEEDED,
    TerminalOutcome.SKIPPED_POLICY: StreamWorkItemState.SKIPPED_POLICY,
    TerminalOutcome.SKIPPED_NOT_NEEDED: StreamWorkItemState.SKIPPED_NOT_NEEDED,
    TerminalOutcome.FAILED: StreamWorkItemState.FAILED,
    TerminalOutcome.CANCELLED: StreamWorkItemState.CANCELLED,
    TerminalOutcome.EXPIRED: StreamWorkItemState.EXPIRED,
    TerminalOutcome.QUARANTINED: StreamWorkItemState.QUARANTINED,
    TerminalOutcome.LATE_INPUT: StreamWorkItemState.LATE_INPUT,
    TerminalOutcome.INCOMPLETE: StreamWorkItemState.INCOMPLETE,
    TerminalOutcome.ABSTAINED: StreamWorkItemState.ABSTAINED,
    TerminalOutcome.NO_EVENTS: StreamWorkItemState.NO_EVENTS,
    TerminalOutcome.INVALIDATED: StreamWorkItemState.INVALIDATED,
}

_NONTERMINAL_STREAM_STATE: Final = {
    WorkItemState.PLANNED: StreamWorkItemState.PLANNED,
    WorkItemState.READY: StreamWorkItemState.READY,
    WorkItemState.LEASED: StreamWorkItemState.LEASED,
    WorkItemState.RUNNING: StreamWorkItemState.RUNNING,
    WorkItemState.RETRY_WAIT: StreamWorkItemState.RETRY_WAIT,
}


class StreamSchedulerCompositionError(RuntimeError):
    """The stream graph conflicts with its exact replay or execution state."""


class StreamBackpressureThrottle(StreamSchedulerCompositionError):
    """A new window was durably retained upstream but cannot enter this DAG yet."""

    def __init__(self, decision: AdmissionDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason or "stream window admission is throttled")


@dataclass(frozen=True, slots=True)
class StreamSchedulerSchemaRefs:
    """Pins for the already-registered WP1 contract family."""

    incremental_window: SchemaRef
    expected_declaration: SchemaRef
    expected_plan_seal: SchemaRef
    stream_work_plan: SchemaRef
    terminal_member: SchemaRef
    terminal_closure: SchemaRef


class _StoredSchedulerSchemaRefs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    incremental_window: SchemaRef
    expected_declaration: SchemaRef
    expected_plan_seal: SchemaRef
    stream_work_plan: SchemaRef
    terminal_member: SchemaRef
    terminal_closure: SchemaRef


class _StoredSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    stream_run_id: str
    dag_config_semantic_sha256: str
    terminal_policy_version: str
    backpressure_config: dict[str, object]
    schema_refs: _StoredSchedulerSchemaRefs


@dataclass(frozen=True, slots=True)
class EosSealInputs:
    """Source-authority facts required to make the expected plan immutable."""

    eos_source_receipt_semantic_sha256: str
    final_source_timeline_semantic_sha256: str
    final_duration_ns: int
    ordered_six_channel_health_closure_sha256: str
    mapping_closure_semantic_sha256: str
    clock_or_alignment_closure_semantic_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "eos_source_receipt_semantic_sha256",
            "final_source_timeline_semantic_sha256",
            "ordered_six_channel_health_closure_sha256",
            "mapping_closure_semantic_sha256",
            "clock_or_alignment_closure_semantic_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if isinstance(self.final_duration_ns, bool) or not isinstance(self.final_duration_ns, int):
            raise TypeError("final_duration_ns must be an integer")
        if self.final_duration_ns < 0:
            raise ValueError("final_duration_ns must be nonnegative")


@dataclass(frozen=True, slots=True)
class StreamBacklogSnapshot:
    """Small query projection for local capacity and recovery checks."""

    state_counts: tuple[tuple[str, int], ...]
    active_backlog: int
    oldest_active_age_seconds: float | None
    declared_window_count: int
    expected_plan_sealed: bool
    terminal_member_count: int
    export_barrier_complete: bool
    finalization_published: bool


@dataclass(frozen=True, slots=True)
class StreamBackpressureSnapshot:
    """One explicit operational decision over the current durable backlog."""

    metrics: QueueMetrics
    decision: AdmissionDecision


@dataclass(frozen=True, slots=True)
class StreamDrainWorkSnapshot:
    """Bounded execution row and direct evidence used by one drain call."""

    plan: StreamWorkItemPlan
    execution_state: WorkItemState
    terminal_evidence: StreamTerminalEvidence | None

    @property
    def is_terminal(self) -> bool:
        return self.execution_state in TERMINAL_WORK_STATES


@dataclass(frozen=True, slots=True)
class StreamExportBarrierSnapshot:
    """Local fixed-six export barrier pointing at the governed manifest."""

    expected_member_count: int
    completed_member_count: int
    export_manifest_semantic_sha256: str | None

    @property
    def complete(self) -> bool:
        return (
            self.expected_member_count == len(CAMERA_IDS)
            and self.completed_member_count == self.expected_member_count
            and self.export_manifest_semantic_sha256 is not None
        )


class DurableStreamWindowScheduler(SinglePassPlanningSink):
    """Persist expected windows and compose a restartable local stream DAG."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        execution_scheduler: SQLiteWorkScheduler,
        expected_plan: ExpectedWindowPlan,
        source_subject: PreEosCaptureSubjectRef,
        stream_run_id: str,
        schema_refs: StreamSchedulerSchemaRefs,
        dag_config_semantic_sha256: str,
        terminal_policy_version: str = "stream-terminal-policy-v1",
        backpressure_signal_provider: Callable[[], BackpressureRuntimeSignals | None] | None = None,
        backpressure_config: BackpressureConfig = DEFAULT_STREAM_BACKPRESSURE_CONFIG,
        clock: Callable[[], datetime] | None = None,
        boundary_observer: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(execution_scheduler, SQLiteWorkScheduler):
            raise TypeError("execution_scheduler must be SQLiteWorkScheduler")
        requested_path = Path(database_path).resolve()
        authority_path = execution_scheduler.database_path.resolve()
        if requested_path != authority_path:
            raise ValueError("stream composition and execution scheduler must share database_path")
        self._ledger = SQLiteStreamWorkLedger(execution_scheduler)
        self._execution_scheduler = execution_scheduler
        # Immutable work-plan rows are safe to retain after publication. The authority
        # remains the source of lifecycle state; this cache only avoids re-reading the
        # same graph metadata between a bounded scope and its exact claim.
        self._work_row_cache: dict[str, StoredStreamWorkPlan] = {}
        self._expected_plan = _strict_model(expected_plan, ExpectedWindowPlan, "expected_plan")
        self._source_subject = _strict_model(
            source_subject, PreEosCaptureSubjectRef, "source_subject"
        )
        self._stream_run_id = _require_uuid(stream_run_id, "stream_run_id")
        self._schema_refs = schema_refs
        self._dag_config_semantic_sha256 = _require_sha256(
            dag_config_semantic_sha256, "dag_config_semantic_sha256"
        )
        if not isinstance(terminal_policy_version, str) or not terminal_policy_version:
            raise ValueError("terminal_policy_version must be non-empty")
        self._terminal_policy_version = terminal_policy_version
        self._backpressure_config = _strict_model(
            backpressure_config,
            BackpressureConfig,
            "backpressure_config",
        )
        if backpressure_signal_provider is not None and not callable(backpressure_signal_provider):
            raise TypeError("backpressure_signal_provider must be callable or None")
        self._backpressure_signal_provider = backpressure_signal_provider
        self._backpressure = BackpressureController(self._backpressure_config)
        self._backpressure_controller_owner_id = f"stream-backpressure:{uuid4()}"
        self._backpressure_controller: StoredStreamBackpressureController
        self._backpressure_state: BackpressureControllerState
        self._clock = clock or (lambda: datetime.now(UTC))
        self._boundary_observer = boundary_observer
        if self._expected_plan.capture_scope_digest != self._source_subject.capture_scope_digest:
            raise ValueError("expected plan and source subject must share capture_scope_digest")
        self._register_plan()
        self._backpressure_controller, self._backpressure_state = (
            self._claim_backpressure_controller()
        )
        self._verify_storage()
        self.recover()

    @classmethod
    def recover_registered(
        cls,
        *,
        execution_scheduler: SQLiteWorkScheduler,
        stream_run_id: str,
        backpressure_signal_provider: Callable[[], BackpressureRuntimeSignals | None] | None = None,
        clock: Callable[[], datetime] | None = None,
        boundary_observer: Callable[[str], None] | None = None,
    ) -> tuple[Self, ...]:
        """Reconstruct every exact persisted graph owned by one canonical run."""

        checked_run_id = _require_uuid(stream_run_id, "stream_run_id")
        ledger = SQLiteStreamWorkLedger(execution_scheduler)
        recovered: list[Self] = []
        for stored in ledger.plans():
            config = _parse_exact(
                stored.composition_config_json,
                _StoredSchedulerConfig,
                "stream scheduler composition config",
            )
            if config.stream_run_id != checked_run_id:
                continue
            if config.version != STREAM_WINDOW_DAG_POLICY_VERSION:
                raise StreamSchedulerCompositionError(
                    "persisted stream scheduler policy version is unsupported"
                )
            recovered.append(
                cls(
                    database_path=execution_scheduler.database_path,
                    execution_scheduler=execution_scheduler,
                    expected_plan=_parse_exact(
                        stored.plan_json,
                        ExpectedWindowPlan,
                        "expected window plan",
                    ),
                    source_subject=_parse_exact(
                        stored.source_subject_json,
                        PreEosCaptureSubjectRef,
                        "pre-EOS source subject",
                    ),
                    stream_run_id=checked_run_id,
                    schema_refs=_schema_refs_from_stored(config.schema_refs),
                    backpressure_signal_provider=backpressure_signal_provider,
                    dag_config_semantic_sha256=config.dag_config_semantic_sha256,
                    terminal_policy_version=config.terminal_policy_version,
                    backpressure_config=_stored_backpressure_config(config.backpressure_config),
                    clock=clock,
                    boundary_observer=boundary_observer,
                )
            )
        return tuple(recovered)

    @property
    def database_path(self) -> Path:
        return self._ledger.database_path

    @property
    def execution_scheduler(self) -> SQLiteWorkScheduler:
        """Return the shared authority used for every stream-ledger operation."""

        return self._execution_scheduler

    @property
    def backpressure_controller_key(self) -> str:
        """Return the declared provider/partition controller key."""

        return self._backpressure_config.controller_key

    @property
    def backpressure_controller_limit(self) -> int:
        """Return the last durable controller limit without creating an observation."""

        return self._backpressure_state.current_limit

    @property
    def plan_key(self) -> str:
        return self._expected_plan.plan_key

    def append_emission(self, emission: PlannerEmission) -> None:
        """Commit all windows from one planner emission as one bounded batch."""

        if not isinstance(emission, PlannerEmission):
            raise TypeError("emission must be PlannerEmission")
        self._append_window_batch(emission.windows)

    def append_windows(self, windows: Sequence[BoundedWindowPlan]) -> None:
        """Commit one explicit bounded, ordered planner window batch."""

        self._append_window_batch(windows)

    def _append_window_batch(self, windows: Sequence[BoundedWindowPlan]) -> None:
        """Commit a bounded sequence of windows as one durable batch."""

        windows = tuple(windows)
        if not windows:
            return
        if any(
            not isinstance(window, BoundedWindowPlan)
            or window.capture_scope_digest != self._expected_plan.capture_scope_digest
            for window in windows
        ):
            raise StreamSchedulerCompositionError(
                "emission contains a window from another capture scope"
            )
        ordinals = tuple(window.ordinal for window in windows)
        if len(set(ordinals)) != len(ordinals) or ordinals != tuple(sorted(ordinals)):
            raise StreamSchedulerCompositionError(
                "emission windows must have unique ordered ordinals"
            )

        persisted_count = self._ledger.next_window_ordinal(self.plan_key)
        lookup_ordinals = {ordinal for ordinal in ordinals if ordinal < persisted_count}
        lookup_ordinals.update(ordinal - 1 for ordinal in ordinals if ordinal > 0)
        persisted = self._ledger.windows_for_ordinals(
            self.plan_key,
            tuple(sorted(lookup_ordinals)),
        )
        persisted_by_ordinal = {row.ordinal: row for row in persisted}
        next_new_ordinal = persisted_count
        new_window_count = 0
        for ordinal in ordinals:
            if ordinal < persisted_count:
                if ordinal not in persisted_by_ordinal:
                    raise StreamSchedulerCompositionError(
                        "persisted expected windows are not contiguous"
                    )
                continue
            if ordinal != next_new_ordinal:
                raise StreamSchedulerCompositionError(
                    "expected windows must be appended in contiguous planner order"
                )
            new_window_count += 1
            next_new_ordinal += 1
        self._admit_window_batch(new_window_count)

        existing_ordinals = tuple(ordinal for ordinal in ordinals if ordinal < persisted_count)
        persisted_children = self._ledger.work_plans_for_ordinals(
            self.plan_key,
            existing_ordinals,
        )
        children_by_ordinal: dict[int, list[StoredStreamWorkPlan]] = {}
        for row in persisted_children:
            if row.expected_ordinal is not None:
                children_by_ordinal.setdefault(row.expected_ordinal, []).append(row)

        declarations_by_ordinal: dict[int, ExpectedWindowDeclaration] = {}
        append_batch: list[NewStreamWindow] = []
        work_item_ids: list[str] = []
        existing_checks: list[tuple[StoredExpectedWindow, str]] = []
        for window in windows:
            wire_window = window.to_incremental_window(self._schema_refs.incremental_window)
            if window.ordinal == 0:
                previous_chain = None
            elif window.ordinal - 1 in declarations_by_ordinal:
                previous_chain = declarations_by_ordinal[window.ordinal - 1].append_chain_sha256
            else:
                previous_row = persisted_by_ordinal.get(window.ordinal - 1)
                if previous_row is None:
                    raise StreamSchedulerCompositionError(
                        "expected windows must preserve contiguous predecessor state"
                    )
                previous = _parse_exact(
                    previous_row.declaration_json,
                    ExpectedWindowDeclaration,
                    "previous declaration",
                )
                previous_chain = previous.append_chain_sha256
            existing = persisted_by_ordinal.get(window.ordinal)
            created_at: str | None = None
            companion_rows = tuple(children_by_ordinal.get(window.ordinal, ()))
            if existing is not None:
                stored_declaration = _parse_exact(
                    existing.declaration_json,
                    ExpectedWindowDeclaration,
                    "expected declaration",
                )
                stored_window = _parse_exact(
                    existing.window_json,
                    IncrementalWindow,
                    "incremental window",
                )
                if len(companion_rows) != len(_WINDOW_DAG_STAGES):
                    raise StreamSchedulerCompositionError(
                        "expected-window replay lacks exact companion work"
                    )
                first_plan = _parse_exact(
                    companion_rows[0].plan_json,
                    StreamWorkItemPlan,
                    "stream work plan",
                )
                created_at = first_plan.created_at
            declaration = create_expected_window_declaration(
                schema_ref=self._schema_refs.expected_declaration,
                plan_key=self.plan_key,
                ordinal=window.ordinal,
                window_key=wire_window.window_key,
                window_semantic_sha256=wire_window.window_semantic_sha256,
                requested_interval=wire_window.requested_interval,
                effective_interval=wire_window.effective_interval,
                ordered_six_slot_segment_or_explicit_absence_closure=wire_window.camera_closure,
                watermark_source_facts_sha256=_watermark_source_facts_sha256(window),
                previous_append_chain_sha256=previous_chain,
            )
            if existing is not None and (
                canonical_json_bytes(stored_declaration) != canonical_json_bytes(declaration)
                or canonical_json_bytes(stored_window) != canonical_json_bytes(wire_window)
            ):
                raise StreamSchedulerCompositionError(
                    "expected-window replay changed declaration source facts"
                )
            plans = self._window_work_plans(wire_window, created_at=created_at)
            new_work = tuple(
                _new_stored_work_plan(
                    plan,
                    expected_ordinal=window.ordinal,
                    role_order=role_order,
                    publication_state="PENDING",
                )
                for role_order, plan in enumerate(plans)
            )
            append_batch.append(
                NewStreamWindow(
                    ordinal=window.ordinal,
                    declaration_json=canonical_json_bytes(declaration),
                    window_json=canonical_json_bytes(wire_window),
                    work_plans=new_work,
                )
            )
            declarations_by_ordinal[window.ordinal] = declaration
            work_item_ids.extend(plan.work_item_id for plan in plans)
            if existing is not None:
                existing_checks.append((existing, plans[-1].work_item_id))

        try:
            inserted = self._ledger.append_windows(
                plan_key=self.plan_key,
                windows=tuple(append_batch),
                controller_key=self._backpressure_config.controller_key,
                controller_policy_version=self._backpressure_config.version,
            )
        except SQLiteStreamWorkLedgerFairnessThrottle as error:
            self._observe("window_admission_throttled")
            raise StreamBackpressureThrottle(self._recording_fairness_decision(error)) from error
        for did_insert in inserted:
            if did_insert:
                self._observe("expected_declaration_durable")
        published_rows = self._ledger.work_plans_for_ids(
            tuple(work_item_ids),
        )
        self._publish_work_rows(published_rows)
        for existing, reduction_id in existing_checks:
            self._verify_existing_window_member(existing, reduction_id)

    def _admit_window_batch(self, new_window_count: int) -> None:
        """Apply admission in planner order without admitting past the queue bound."""

        if isinstance(new_window_count, bool) or not isinstance(new_window_count, int):
            raise TypeError("new_window_count must be an integer")
        if new_window_count < 0:
            raise ValueError("new_window_count must be nonnegative")
        if new_window_count == 0:
            return

        pressure = self.backpressure_snapshot()
        metrics = pressure.metrics
        state = self._backpressure_state
        observed_at_ms = state.last_observed_at_ms
        if observed_at_ms is None:
            raise StreamSchedulerCompositionError("backpressure observation was not persisted")

        for index in range(new_window_count):
            if index:
                decision, state = self._backpressure.evaluate(
                    _STAGE_EXECUTION_PROJECTION[StreamStage.WINDOW],
                    metrics,
                    state,
                    observed_at_ms=observed_at_ms,
                )
            else:
                decision = pressure.decision
            if not decision.admitted:
                self._observe("window_admission_throttled")
                raise StreamBackpressureThrottle(decision)
            projected_depth = metrics.depth + len(_WINDOW_DAG_STAGES)
            metrics = QueueMetrics(
                depth=projected_depth,
                oldest_age_ms=metrics.oldest_age_ms,
                arrival_rate=metrics.arrival_rate,
                service_rate=metrics.service_rate,
                backlog_slope=metrics.backlog_slope,
                provider_quota=metrics.provider_quota,
                worker_utilization=metrics.worker_utilization,
            )

    def append_window(self, window: BoundedWindowPlan) -> ExpectedWindowDeclaration:
        """Idempotently append one expected window and its fixed local DAG."""

        if not isinstance(window, BoundedWindowPlan):
            raise TypeError("window must be BoundedWindowPlan")
        if window.capture_scope_digest != self._expected_plan.capture_scope_digest:
            raise StreamSchedulerCompositionError("window belongs to another capture scope")
        wire_window = window.to_incremental_window(self._schema_refs.incremental_window)
        stored_window_count = self._ledger.next_window_ordinal(self.plan_key)
        if window.ordinal > stored_window_count:
            raise StreamSchedulerCompositionError(
                "expected windows must be appended in contiguous planner order"
            )
        existing = (
            self._ledger.window_at(self.plan_key, window.ordinal)
            if window.ordinal < stored_window_count
            else None
        )
        if existing is None:
            pressure = self.backpressure_snapshot()
            if not pressure.decision.admitted:
                self._observe("window_admission_throttled")
                raise StreamBackpressureThrottle(pressure.decision)
        previous_chain = None
        if window.ordinal > 0:
            previous_row = self._ledger.window_at(self.plan_key, window.ordinal - 1)
            if previous_row is None:
                raise StreamSchedulerCompositionError(
                    "expected windows must preserve contiguous predecessor state"
                )
            previous = _parse_exact(
                previous_row.declaration_json,
                ExpectedWindowDeclaration,
                "previous declaration",
            )
            previous_chain = previous.append_chain_sha256
        declaration = create_expected_window_declaration(
            schema_ref=self._schema_refs.expected_declaration,
            plan_key=self.plan_key,
            ordinal=window.ordinal,
            window_key=wire_window.window_key,
            window_semantic_sha256=wire_window.window_semantic_sha256,
            requested_interval=wire_window.requested_interval,
            effective_interval=wire_window.effective_interval,
            ordered_six_slot_segment_or_explicit_absence_closure=wire_window.camera_closure,
            watermark_source_facts_sha256=_watermark_source_facts_sha256(window),
            previous_append_chain_sha256=previous_chain,
        )
        created_at: str | None = None
        companion_rows: tuple[StoredStreamWorkPlan, ...] = ()
        if existing is not None:
            stored_declaration = _parse_exact(
                existing.declaration_json,
                ExpectedWindowDeclaration,
                "expected declaration",
            )
            stored_window = _parse_exact(
                existing.window_json, IncrementalWindow, "incremental window"
            )
            if canonical_json_bytes(stored_declaration) != canonical_json_bytes(
                declaration
            ) or canonical_json_bytes(stored_window) != canonical_json_bytes(wire_window):
                raise StreamSchedulerCompositionError(
                    "expected-window replay changed declaration source facts"
                )
            companion_rows = self._ledger.work_plans_for_ordinal(
                self.plan_key,
                window.ordinal,
            )
            if len(companion_rows) != len(_WINDOW_DAG_STAGES):
                raise StreamSchedulerCompositionError(
                    "expected-window replay lacks exact companion work"
                )
            first_plan = _parse_exact(
                companion_rows[0].plan_json,
                StreamWorkItemPlan,
                "stream work plan",
            )
            created_at = first_plan.created_at
        new_work = tuple(
            _new_stored_work_plan(
                plan,
                expected_ordinal=window.ordinal,
                role_order=role_order,
                publication_state="PENDING",
            )
            for role_order, plan in enumerate(
                self._window_work_plans(wire_window, created_at=created_at)
            )
        )
        try:
            inserted = self._ledger.append_window(
                plan_key=self.plan_key,
                ordinal=window.ordinal,
                declaration_json=canonical_json_bytes(declaration),
                window_json=canonical_json_bytes(wire_window),
                work_plans=new_work,
                controller_key=self._backpressure_config.controller_key,
                controller_policy_version=self._backpressure_config.version,
            )
        except SQLiteStreamWorkLedgerFairnessThrottle as error:
            self._observe("window_admission_throttled")
            raise StreamBackpressureThrottle(self._recording_fairness_decision(error)) from error
        if inserted:
            self._observe("expected_declaration_durable")
        self._publish_work_rows(new_work if inserted else companion_rows)
        if existing is not None:
            self._verify_existing_window_member(existing, new_work[-1].work_item_id)
            return stored_declaration
        return declaration

    def seal(self, finish: PlannerFinish) -> None:
        """Persist planner EOS after appending every final partial window."""

        if not isinstance(finish, PlannerFinish):
            raise TypeError("finish must be PlannerFinish")
        self._append_window_batch(finish.windows)
        finish_sha256 = _planner_finish_sha256(finish)
        self._ledger.set_planner_eos(self.plan_key, finish_sha256)
        self._observe("planner_eos_durable")

    def finalize_eos(self, inputs: EosSealInputs) -> ExpectedWindowPlanSeal:
        """Commit the expected-set seal and a gated finalization work plan."""

        if not isinstance(inputs, EosSealInputs):
            raise TypeError("inputs must be EosSealInputs")
        declarations = self.declarations()
        candidate = create_expected_window_plan_seal(
            schema_ref=self._schema_refs.expected_plan_seal,
            plan=self._expected_plan,
            declarations=declarations,
            eos_source_receipt_semantic_sha256=inputs.eos_source_receipt_semantic_sha256,
            final_source_timeline_semantic_sha256=inputs.final_source_timeline_semantic_sha256,
            final_duration_ns=inputs.final_duration_ns,
            ordered_six_channel_health_closure_sha256=(
                inputs.ordered_six_channel_health_closure_sha256
            ),
            mapping_closure_semantic_sha256=inputs.mapping_closure_semantic_sha256,
            clock_or_alignment_closure_semantic_sha256=(
                inputs.clock_or_alignment_closure_semantic_sha256
            ),
        )
        stored_plan = self._ledger.get_plan(self.plan_key)
        if stored_plan.planner_eos_sha256 is None:
            raise StreamSchedulerCompositionError("planner EOS must be durable before seal")
        existing_seal: ExpectedWindowPlanSeal | None = None
        finalization_created_at: str | None = None
        if stored_plan.seal_json is not None:
            existing_seal = _parse_exact(
                stored_plan.seal_json, ExpectedWindowPlanSeal, "expected plan seal"
            )
            if canonical_json_bytes(existing_seal) != canonical_json_bytes(candidate):
                raise StreamSchedulerCompositionError("EOS seal replay changed source facts")
            finalization_rows = tuple(
                row
                for row in self._ledger.work_plans(self.plan_key)
                if row.expected_ordinal is None
            )
            if len(finalization_rows) != 1:
                raise StreamSchedulerCompositionError(
                    "EOS seal replay lacks exact finalization work"
                )
            finalization_created_at = _parse_exact(
                finalization_rows[0].plan_json,
                StreamWorkItemPlan,
                "finalization work plan",
            ).created_at
        finalization = self._finalization_work_plan(
            candidate,
            created_at=finalization_created_at,
        )
        inserted = self._ledger.store_seal_and_finalization(
            plan_key=self.plan_key,
            seal_json=canonical_json_bytes(candidate),
            expected_declaration_jsons=tuple(
                canonical_json_bytes(declaration) for declaration in declarations
            ),
            finalization=_new_stored_work_plan(
                finalization,
                expected_ordinal=None,
                role_order=len(_WINDOW_DAG_STAGES),
                publication_state="GATED",
            ),
        )
        if inserted:
            self._observe("expected_plan_seal_durable")
        return candidate if existing_seal is None else existing_seal

    def mark_export_barrier_complete(
        self,
        *,
        export_manifest_semantic_sha256: str,
        completed_member_count: int,
    ) -> StreamExportBarrierSnapshot:
        """Record the fixed-six export barrier without inventing a broker fact."""

        digest = _require_sha256(export_manifest_semantic_sha256, "export_manifest_semantic_sha256")
        if completed_member_count != len(CAMERA_IDS):
            raise StreamSchedulerCompositionError(
                "export barrier requires exactly six completed camera members"
            )
        self._ledger.mark_export_barrier(
            plan_key=self.plan_key,
            manifest_sha256=digest,
            member_count=completed_member_count,
        )
        return self.export_barrier()

    def close_finalization_gate(self) -> WindowTerminalClosure:
        """Close terminal/export barriers and publish finalization work."""

        seal = self.expected_plan_seal()
        if seal is None:
            raise StreamSchedulerCompositionError("expected plan must be sealed before closure")
        declarations = self.declarations()
        members = self.terminal_members()
        member_ordinals = {member.expected_ordinal for member in members}
        if len(members) != len(declarations):
            missing = tuple(
                declaration.ordinal
                for declaration in declarations
                if declaration.ordinal not in member_ordinals
            )
            raise StreamSchedulerCompositionError(
                f"terminal closure is missing declared ordinals: {missing}"
            )
        closure = create_window_terminal_closure(
            schema_ref=self._schema_refs.terminal_closure,
            plan_seal=seal,
            expected_declarations=declarations,
            members=members,
        )
        if not self.export_barrier().complete:
            raise StreamSchedulerCompositionError(
                "six-camera export barrier must complete before finalization publication"
            )
        self._ledger.store_closure_and_open_finalization(
            plan_key=self.plan_key,
            closure_json=canonical_json_bytes(closure),
            finalization_stage=StreamStage.FINALIZATION.value,
        )
        self._observe("finalization_gate_durable")
        self.recover()
        return closure

    def recover(self) -> int:
        """Reconcile crash intents and replay only un-published work projections."""

        accepted = self._recover_terminal_acceptances()
        return accepted + self._publish_work_rows(
            self._ledger.pending_publication_work_rows(self.plan_key)
        )

    def _publish_work_rows(
        self,
        rows: Sequence[NewStreamWorkPlan | StoredStreamWorkPlan],
    ) -> int:
        """Project one bounded PENDING batch, then atomically expose it for execution."""

        pending = tuple(row for row in rows if row.publication_state == "PENDING")
        if not pending:
            return 0
        parsed = tuple(
            _parse_exact(row.plan_json, StreamWorkItemPlan, "stream work plan") for row in pending
        )
        plans_by_key = {plan.work_logical_key: plan for plan in parsed}
        projections = tuple(
            self._internal_execution_projection(plan, plans_by_key=plans_by_key) for plan in parsed
        )
        self._execution_scheduler.plan_many(projections)
        for _plan in parsed:
            self._observe("internal_execution_projected")
        self._ledger.mark_published_many(tuple(plan.work_item_id for plan in parsed))
        for row in pending:
            self._remember_published_work_row(row)
        return len(parsed)

    def claim(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        work_item_id: str | None = None,
        now: datetime | None = None,
        recover_graph: bool = True,
    ) -> StreamWorkLeaseClaim | None:
        """Claim only work owned by this graph and return a typed lease."""

        checked_now = self._checked_now(now)
        if recover_graph:
            self.recover()
        stored: StoredStreamWorkPlan | None
        if work_item_id is not None:
            stored = self._cached_published_work_row(work_item_id)
            if stored is None:
                stored = self._stored_work_row(work_item_id, require_published=True)
        else:
            # Do not reconstruct the whole graph merely to find one claim candidate.
            # SQLite selects the one READY row made visible by the latest transition.
            stored = self._ledger.next_ready_work(self.plan_key)
        if stored is None:
            return None
        plan = _parse_exact(stored.plan_json, StreamWorkItemPlan, "stream work plan")
        claim = self._execution_scheduler.claim(
            worker_id,
            lease_duration_seconds,
            work_item_id=plan.work_item_id,
            now=checked_now,
        )
        if claim is None:
            return None
        return StreamWorkLeaseClaim(
            work_item=self._stream_item_from(
                plan,
                claim.work_item,
                evidence=None,
            ),
            lease=_stream_lease(claim.lease),
        )

    def claim_and_start(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        work_item_id: str | None = None,
        now: datetime | None = None,
        recover_graph: bool = True,
    ) -> StreamWorkLeaseClaim | None:
        """Claim and start one graph-owned item for a same-process executor."""

        checked_now = self._checked_now(now)
        if recover_graph:
            self.recover()
        stored: StoredStreamWorkPlan | None
        if work_item_id is not None:
            stored = self._cached_published_work_row(work_item_id)
            if stored is None:
                stored = self._stored_work_row(work_item_id, require_published=True)
        else:
            stored = self._ledger.next_ready_work(self.plan_key)
        if stored is None:
            return None
        plan = _parse_exact(stored.plan_json, StreamWorkItemPlan, "stream work plan")
        claim = self._execution_scheduler.claim_and_start(
            worker_id,
            lease_duration_seconds,
            work_item_id=plan.work_item_id,
            now=checked_now,
        )
        if claim is None:
            return None
        return StreamWorkLeaseClaim(
            work_item=self._stream_item_from(
                plan,
                claim.work_item,
                evidence=None,
            ),
            lease=_stream_lease(claim.lease),
        )

    def resume_owned_active(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        work_item_id: str,
        now: datetime | None = None,
    ) -> StreamWorkLeaseClaim | None:
        """Resume one still-active lease without taking it from another worker.

        This is intentionally narrower than a claim or lease-recovery operation:
        it preserves the persisted fence and attempt and only works when the
        authority still records ``worker_id`` as the owner.  It is used after a
        process restart at the pre-EOS provider/terminal-acceptance boundary, so
        a canonical evidence ledger can replay the already-persisted provider
        terminal without scheduling a second dispatch.
        """

        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be non-empty")
        if isinstance(lease_duration_seconds, bool) or not isinstance(lease_duration_seconds, int):
            raise TypeError("lease_duration_seconds must be an integer")
        if lease_duration_seconds <= 0:
            raise ValueError("lease_duration_seconds must be positive")
        stored = self._stored_work_row(work_item_id, require_published=True)
        _parse_exact(stored.plan_json, StreamWorkItemPlan, "stream work plan")
        execution = self._execution_scheduler.get(work_item_id)
        if (
            execution.state not in {WorkItemState.LEASED, WorkItemState.RUNNING}
            or execution.leased_by != worker_id
            or execution.fencing_token is None
            or execution.lease_expires_at is None
        ):
            return None
        lease = StreamWorkLease(
            work_item_id=execution.work_item_id,
            worker_id=worker_id,
            lease_epoch=execution.lease_epoch,
            fencing_token=execution.fencing_token,
            lease_expires_at=execution.lease_expires_at,
        )
        checked_now = self._checked_now(now)
        try:
            renewed = self.heartbeat(
                lease,
                lease_duration_seconds,
                now=checked_now,
            )
            execution_item = self.start(renewed, now=checked_now)
        except WorkFenceError:
            # A concurrent authority transition won the race.  Do not steal the
            # work or create a new attempt; the ordinary scheduler path decides
            # whether it becomes claimable later.
            return None
        return StreamWorkLeaseClaim(
            work_item=execution_item,
            lease=renewed,
        )

    def start(
        self,
        lease: StreamWorkLease,
        *,
        now: datetime | None = None,
    ) -> StreamWorkItem:
        checked = _strict_model(lease, StreamWorkLease, "lease")
        plan = self._stored_work_plan(checked.work_item_id, require_published=True)
        item = self._execution_scheduler.start(
            _execution_lease(checked), now=self._checked_now(now)
        )
        return self._stream_item_from(plan, item, evidence=None)

    def heartbeat(
        self,
        lease: StreamWorkLease,
        lease_duration_seconds: int,
        *,
        now: datetime | None = None,
    ) -> StreamWorkLease:
        checked = _strict_model(lease, StreamWorkLease, "lease")
        self._stored_work_plan(checked.work_item_id, require_published=True)
        renewed = self._execution_scheduler.heartbeat(
            _execution_lease(checked),
            lease_duration_seconds,
            now=self._checked_now(now),
        )
        return _stream_lease(renewed)

    def fail_retryable(
        self,
        lease: StreamWorkLease,
        *,
        error_code: str,
        error_detail: str | None = None,
        now: datetime | None = None,
    ) -> StreamWorkItem:
        """Record a fenced execution failure without inventing terminal evidence.

        A worker can fail before it has prepared a typed stream terminal. Keep that
        distinction durable by returning the execution item to the scheduler's
        retry path instead of synthesizing a stream terminal fact.
        """

        checked = _strict_model(lease, StreamWorkLease, "lease")
        stored = self._stored_work_row(
            checked.work_item_id,
            require_published=True,
        )
        plan = _parse_exact(stored.plan_json, StreamWorkItemPlan, "stream work plan")
        active_execution = self._execution_scheduler.get(checked.work_item_id)
        if active_execution.attempt >= active_execution.max_attempts:
            raise StreamSchedulerCompositionError(
                "pre-terminal execution cannot exhaust attempts without typed stream evidence"
            )
        execution = self._execution_scheduler.fail(
            _execution_lease(checked),
            error_code=error_code,
            error_detail=error_detail,
            retryable=True,
            now=self._checked_now(now),
        )
        return self._stream_item_from(
            plan,
            execution,
            evidence=self._accepted_evidence(stored=stored),
        )

    def complete(
        self,
        lease: StreamWorkLease,
        terminal_evidence: StreamTerminalEvidence,
        *,
        now: datetime | None = None,
    ) -> StreamWorkItem:
        """Accept one fenced terminal fact across the two local ledgers."""

        checked_lease = _strict_model(lease, StreamWorkLease, "lease")
        checked_evidence = _strict_model(
            terminal_evidence, StreamTerminalEvidence, "terminal_evidence"
        )
        stored = self._stored_work_row(
            checked_lease.work_item_id,
            require_published=True,
        )
        plan = _parse_exact(stored.plan_json, StreamWorkItemPlan, "stream work plan")
        if checked_evidence.terminal_policy_version != self._terminal_policy_version:
            raise StreamSchedulerCompositionError(
                "terminal evidence does not use the composition policy pin"
            )
        checked_now = self._checked_now(now)
        if _parse_timestamp(checked_evidence.completed_at) > checked_now:
            raise StreamSchedulerCompositionError(
                "terminal evidence cannot complete after authority time"
            )
        accepted = self._accepted_evidence(stored=stored)
        if accepted is not None:
            if canonical_json_bytes(accepted) != canonical_json_bytes(checked_evidence):
                raise StreamSchedulerCompositionError(
                    "terminal replay changed accepted stream evidence"
                )
            return self._stream_item_from(
                plan,
                self._execution_scheduler.get(checked_lease.work_item_id),
                evidence=accepted,
            )
        self._store_pending_terminal(
            checked_lease,
            checked_evidence,
            authority_now=checked_now,
        )
        execution = self._execution_scheduler.succeed(
            _execution_lease(checked_lease),
            result_reference=_terminal_result_reference(checked_evidence),
            result_sha256=checked_evidence.evidence_ref.exact_sha256,
            now=checked_now,
        )
        self._observe("execution_terminal_committed")
        self._accept_pending_terminal(
            checked_lease.work_item_id,
            stored=stored,
            execution=execution,
            pending_raw=canonical_json_bytes(checked_evidence),
        )
        return self._stream_item_from(
            plan,
            execution,
            evidence=checked_evidence,
        )

    def prepare_window_terminal_member(
        self,
        lease: StreamWorkLease,
        terminal_evidence: StreamTerminalEvidence,
    ) -> WindowTerminalMember:
        """Build the exact reduction member for an external atomic commit."""

        checked_lease = _strict_model(lease, StreamWorkLease, "lease")
        checked_evidence = _strict_model(
            terminal_evidence, StreamTerminalEvidence, "terminal_evidence"
        )
        if checked_evidence.terminal_policy_version != self._terminal_policy_version:
            raise StreamSchedulerCompositionError(
                "terminal evidence does not use the composition policy pin"
            )
        stored = self._ledger.get_work(checked_lease.work_item_id)
        plan = _parse_exact(stored.plan_json, StreamWorkItemPlan, "stream work plan")
        if plan.stage is not StreamStage.WINDOW_REDUCTION:
            raise StreamSchedulerCompositionError(
                "only WINDOW_REDUCTION has a window terminal member"
            )
        ordinal = stored.expected_ordinal
        expected_window = (
            None if ordinal is None else self._ledger.window_at(self.plan_key, ordinal)
        )
        if ordinal is None or expected_window is None:
            raise StreamSchedulerCompositionError("window reduction lacks its expected declaration")
        declaration = _parse_exact(
            expected_window.declaration_json,
            ExpectedWindowDeclaration,
            "expected declaration",
        )
        return create_window_terminal_member(
            schema_ref=self._schema_refs.terminal_member,
            plan_key=self.plan_key,
            expected_ordinal=ordinal,
            window_key=declaration.window_key,
            window_semantic_sha256=declaration.window_semantic_sha256,
            terminal_outcome=checked_evidence.outcome,
            terminal_work_item_id=plan.work_item_id,
            terminal_work_logical_key=plan.work_logical_key,
            terminal_evidence_ref=checked_evidence.evidence_ref,
            terminal_policy_version=checked_evidence.terminal_policy_version,
        )

    def get(self, work_item_id: str) -> StreamWorkItem:
        """Read one stream item without turning a normal read into graph recovery."""

        stored = self._stored_work_row(work_item_id, require_published=False)
        plan = _parse_exact(stored.plan_json, StreamWorkItemPlan, "stream work plan")
        evidence = self._accepted_evidence(stored=stored)
        return self._stream_item_from(
            plan,
            self._execution_scheduler.get(work_item_id),
            evidence=evidence,
        )

    def declarations(self) -> tuple[ExpectedWindowDeclaration, ...]:
        return tuple(
            _parse_exact(
                row.declaration_json,
                ExpectedWindowDeclaration,
                "expected declaration",
            )
            for row in self._ledger.windows(self.plan_key)
        )

    def work_plans(self) -> tuple[StreamWorkItemPlan, ...]:
        return tuple(plan for plan, _state in self._stored_work_plans_with_state())

    def bounded_drain_scope(self, max_items: int) -> tuple[StreamDrainWorkSnapshot, ...]:
        """Load bounded active work and only the direct evidence it consumes."""

        if isinstance(max_items, bool) or not isinstance(max_items, int):
            raise TypeError("max_items must be an integer")
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        rows = self._ledger.bounded_execution_scope(
            plan_key=self.plan_key,
            max_active=max_items,
            terminal_states=tuple(sorted(state.value for state in TERMINAL_WORK_STATES)),
            finalization_stage=StreamStage.FINALIZATION.value,
        )
        snapshots: list[StreamDrainWorkSnapshot] = []
        for row in rows:
            self._remember_work_row(row.work)
            snapshots.append(
                StreamDrainWorkSnapshot(
                    plan=_parse_exact(
                        row.work.plan_json,
                        StreamWorkItemPlan,
                        "bounded stream work plan",
                    ),
                    execution_state=WorkItemState(row.execution_state),
                    terminal_evidence=(
                        None
                        if row.work.terminal_evidence_json is None
                        else _parse_exact(
                            row.work.terminal_evidence_json,
                            StreamTerminalEvidence,
                            "bounded stream terminal evidence",
                        )
                    ),
                )
            )
        return tuple(snapshots)

    def work_items(self, *, recover_graph: bool = True) -> tuple[StreamWorkItem, ...]:
        """Read this graph's published work from batched authority snapshots."""

        if recover_graph:
            self.recover()
        rows = self._ledger.work_plans(self.plan_key)
        executions = {
            item.work_item_id: item
            for item in self._execution_scheduler.items_for_run(self._stream_run_id)
        }
        snapshots: list[StreamWorkItem] = []
        for row in rows:
            self._remember_work_row(row)
            if row.publication_state != "PUBLISHED":
                continue
            execution = executions.get(row.work_item_id)
            if execution is None:
                raise StreamSchedulerCompositionError(
                    "published stream work lacks its execution projection"
                )
            plan = _parse_exact(row.plan_json, StreamWorkItemPlan, "stream work plan")
            evidence = (
                None
                if row.terminal_evidence_json is None
                else _parse_exact(
                    row.terminal_evidence_json,
                    StreamTerminalEvidence,
                    "stream terminal evidence",
                )
            )
            snapshots.append(self._stream_item_from(plan, execution, evidence=evidence))
        return tuple(snapshots)

    def terminal_members(self) -> tuple[WindowTerminalMember, ...]:
        return tuple(
            _parse_exact(
                row.terminal_member_json,
                WindowTerminalMember,
                "window terminal member",
            )
            for row in self._ledger.windows(self.plan_key)
            if row.terminal_member_json is not None
        )

    def terminal_member_at(self, ordinal: int) -> WindowTerminalMember | None:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("ordinal must be an integer")
        if ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        raw = self._ledger.terminal_member_at(self.plan_key, ordinal)
        if raw is None:
            return None
        member = _parse_exact(raw, WindowTerminalMember, "window terminal member")
        if member.expected_ordinal != ordinal:
            raise StreamSchedulerCompositionError(
                "window terminal member disagrees with its durable ordinal"
            )
        return member

    def terminal_member_count(self) -> int:
        return self._ledger.terminal_member_count(self.plan_key)

    def expected_plan_seal(self) -> ExpectedWindowPlanSeal | None:
        raw = self._ledger.get_plan(self.plan_key).seal_json
        return (
            None if raw is None else _parse_exact(raw, ExpectedWindowPlanSeal, "expected plan seal")
        )

    def terminal_closure(self) -> WindowTerminalClosure | None:
        raw = self._ledger.get_plan(self.plan_key).terminal_closure_json
        return (
            None
            if raw is None
            else _parse_exact(raw, WindowTerminalClosure, "window terminal closure")
        )

    def export_barrier(self) -> StreamExportBarrierSnapshot:
        stored_plan = self._ledger.get_plan(self.plan_key)
        digest = stored_plan.export_manifest_sha256
        count = stored_plan.export_member_count
        return StreamExportBarrierSnapshot(
            expected_member_count=len(CAMERA_IDS),
            completed_member_count=0 if count is None else count,
            export_manifest_semantic_sha256=digest,
        )

    def backlog(self, *, now: datetime | None = None) -> StreamBacklogSnapshot:
        """Return counts and oldest age without implicitly advancing state."""

        checked_now = self._checked_now(now)
        projection = self._ledger.backlog_projection(
            plan_key=self.plan_key,
            terminal_states=tuple(sorted(state.value for state in TERMINAL_WORK_STATES)),
            finalization_stage=StreamStage.FINALIZATION.value,
        )
        oldest = (
            None
            if projection.oldest_active_created_at is None
            else max(
                0.0,
                (
                    checked_now - _parse_timestamp(projection.oldest_active_created_at)
                ).total_seconds(),
            )
        )
        export_barrier = StreamExportBarrierSnapshot(
            expected_member_count=len(CAMERA_IDS),
            completed_member_count=(
                0 if projection.export_member_count is None else projection.export_member_count
            ),
            export_manifest_semantic_sha256=projection.export_manifest_sha256,
        )
        return StreamBacklogSnapshot(
            state_counts=projection.state_counts,
            active_backlog=projection.active_backlog,
            oldest_active_age_seconds=oldest,
            declared_window_count=projection.declared_window_count,
            expected_plan_sealed=projection.expected_plan_sealed,
            terminal_member_count=projection.terminal_member_count,
            export_barrier_complete=export_barrier.complete,
            finalization_published=projection.finalization_published,
        )

    def backpressure_snapshot(
        self,
        *,
        now: datetime | None = None,
        metrics: QueueMetrics | None = None,
    ) -> StreamBackpressureSnapshot:
        """Persist one timing-only controller observation over durable backlog."""

        checked_now = self._checked_now(now)
        observed_at_ms = int(checked_now.timestamp() * 1_000)
        backlog = self.backlog(now=checked_now)
        durable_metrics, sample_counts = self._durable_backpressure_metrics(
            backlog,
            observed_at_ms=observed_at_ms,
        )
        if metrics is None:
            runtime_signals = self._runtime_backpressure_signals()
            metrics = durable_metrics.model_copy(
                update={
                    "provider_quota": runtime_signals.provider_quota,
                    "worker_utilization": runtime_signals.worker_utilization,
                }
            )
        else:
            metrics = _strict_model(metrics, QueueMetrics, "metrics")
        decision, next_state = self._backpressure.evaluate(
            _STAGE_EXECUTION_PROJECTION[StreamStage.WINDOW],
            metrics,
            self._backpressure_state,
            observed_at_ms=observed_at_ms,
        )
        total_work_count, terminal_work_count, backlog_depth = sample_counts
        next_state = next_state.model_copy(
            update={
                "last_arrival_count": total_work_count,
                "last_service_count": terminal_work_count,
                "last_backlog_depth": backlog_depth,
            }
        )
        try:
            controller = self._ledger.save_backpressure_controller(
                self._backpressure_controller,
                state_json=canonical_json_bytes(next_state),
            )
        except WorkFenceError as error:
            raise StreamSchedulerCompositionError(
                "backpressure controller ownership changed"
            ) from error
        self._backpressure_controller = controller
        self._backpressure_state = next_state
        return StreamBackpressureSnapshot(
            metrics=metrics,
            decision=decision,
        )

    def _runtime_backpressure_signals(self) -> BackpressureRuntimeSignals:
        """Read non-durable provider/executor signals without inventing unavailable values."""

        provider = self._backpressure_signal_provider
        if provider is None:
            return BackpressureRuntimeSignals()
        try:
            observed = provider()
        except Exception as error:
            raise StreamSchedulerCompositionError(
                "backpressure runtime signal provider failed"
            ) from error
        if observed is None:
            return BackpressureRuntimeSignals()
        try:
            return _strict_model(
                observed,
                BackpressureRuntimeSignals,
                "backpressure runtime signals",
            )
        except (TypeError, ValueError) as error:
            raise StreamSchedulerCompositionError(
                "backpressure runtime signal provider returned invalid signals"
            ) from error

    def _recording_fairness_decision(
        self,
        error: SQLiteStreamWorkLedgerFairnessThrottle,
    ) -> AdmissionDecision:
        """Map a durable partition-share rejection onto the timing-only controller surface."""

        return AdmissionDecision(
            admitted=False,
            policy_version=self._backpressure_config.version,
            pressure_class=PressureClass.THROTTLED,
            signals=("RECORDING_FAIRNESS",),
            shedding_actions=("THROTTLE_LEDGER",),
            reason=(
                "stream window admission is throttled by RECORDING_FAIRNESS "
                f"for partition {error.controller_key}"
            ),
            suggested_delay_ms=1_000,
            controller_limit=self._backpressure_state.current_limit,
            controller_mode=self._backpressure_config.controller_mode,
        )

    def _durable_backpressure_metrics(
        self,
        backlog: StreamBacklogSnapshot,
        *,
        observed_at_ms: int,
    ) -> tuple[QueueMetrics, tuple[int, int, int]]:
        """Derive rate observations from monotonically durable work transitions."""

        total_work_count = sum(count for _state, count in backlog.state_counts)
        terminal_work_count = max(0, total_work_count - backlog.active_backlog)
        oldest_age_ms = (
            0
            if backlog.oldest_active_age_seconds is None
            else max(0, int(backlog.oldest_active_age_seconds * 1_000))
        )
        arrival_rate: float | None = None
        service_rate: float | None = None
        backlog_slope: float | None = None
        state = self._backpressure_state
        if (
            state.last_observed_at_ms is not None
            and state.last_arrival_count is not None
            and state.last_service_count is not None
            and state.last_backlog_depth is not None
        ):
            elapsed_ms = observed_at_ms - state.last_observed_at_ms
            if elapsed_ms >= self._backpressure_config.minimum_rate_observation_interval_ms:
                elapsed_seconds = elapsed_ms / 1_000
                arrival_rate = max(
                    0.0,
                    (total_work_count - state.last_arrival_count) / elapsed_seconds,
                )
                service_rate = max(
                    0.0,
                    (terminal_work_count - state.last_service_count) / elapsed_seconds,
                )
                backlog_slope = (
                    backlog.active_backlog - state.last_backlog_depth
                ) / elapsed_seconds

        return (
            QueueMetrics(
                depth=backlog.active_backlog,
                oldest_age_ms=oldest_age_ms,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                backlog_slope=backlog_slope,
            ),
            (total_work_count, terminal_work_count, backlog.active_backlog),
        )

    def _claim_backpressure_controller(
        self,
    ) -> tuple[StoredStreamBackpressureController, BackpressureControllerState]:
        """Fence one local owner while preserving the last canonical controller state."""

        controller_key = self._backpressure_config.controller_key
        initial_state = self._backpressure.initial_state(controller_key)
        controller = self._ledger.claim_backpressure_controller(
            plan_key=self.plan_key,
            controller_key=controller_key,
            policy_version=self._backpressure_config.version,
            owner_id=self._backpressure_controller_owner_id,
            initial_state_json=canonical_json_bytes(initial_state),
        )
        state = _parse_exact(
            controller.state_json,
            BackpressureControllerState,
            "backpressure controller state",
        )
        if (
            controller.plan_key != self.plan_key
            or controller.controller_key != controller_key
            or controller.policy_version != self._backpressure_config.version
            or state.controller_key != controller_key
            or state.policy_version != self._backpressure_config.version
            or state.controller_version != self._backpressure_config.controller_version
        ):
            raise StreamSchedulerCompositionError(
                "persisted backpressure controller does not match composition policy"
            )
        if not (
            self._backpressure_config.minimum_limit
            <= state.current_limit
            <= self._backpressure_config.maximum_limit
        ):
            raise StreamSchedulerCompositionError(
                "persisted backpressure controller limit is outside policy bounds"
            )
        return controller, state

    def _window_work_plans(
        self,
        window: IncrementalWindow,
        *,
        created_at: str | None = None,
    ) -> tuple[StreamWorkItemPlan, ...]:
        plans: list[StreamWorkItemPlan] = []
        authority_created_at = self._timestamp_now() if created_at is None else created_at
        authority_created = _parse_timestamp(authority_created_at)
        plans_by_stage: dict[StreamStage, StreamWorkItemPlan] = {}
        for stage_index, (stage, upstream_stages) in enumerate(_WINDOW_DAG_TOPOLOGY):
            dependencies = tuple(
                sorted(
                    (
                        StreamWorkDependency(
                            upstream_work_logical_key=plans_by_stage[
                                upstream_stage
                            ].work_logical_key,
                            criticality=DependencyCriticality.DEGRADABLE,
                        )
                        for upstream_stage in upstream_stages
                    ),
                    key=lambda value: value.upstream_work_logical_key,
                )
            )
            input_digest = semantic_sha256(
                {
                    "version": STREAM_WINDOW_DAG_POLICY_VERSION,
                    "window_semantic_sha256": window.window_semantic_sha256,
                    "stage": stage.value,
                    "ordered_upstream_work": [
                        {
                            "work_logical_key": value.upstream_work_logical_key,
                            "criticality": value.criticality.value,
                        }
                        for value in dependencies
                    ],
                }
            )
            plan = create_stream_work_item_plan(
                schema_ref=self._schema_refs.stream_work_plan,
                stream_run_id=self._stream_run_id,
                source_subject=self._source_subject,
                stage=stage,
                subject=window.reference(),
                input_semantic_sha256=input_digest,
                config_semantic_sha256=self._dag_config_semantic_sha256,
                ordered_dependencies=dependencies,
                sla_deadline_at=_format_timestamp(
                    authority_created
                    + timedelta(seconds=_LOCAL_STAGE_SLA_STEP_SECONDS * (stage_index + 1))
                ),
                created_at=authority_created_at,
            )
            plans.append(plan)
            plans_by_stage[stage] = plan
        return tuple(plans)

    def _finalization_work_plan(
        self,
        seal: ExpectedWindowPlanSeal,
        *,
        created_at: str | None = None,
    ) -> StreamWorkItemPlan:
        reductions = tuple(
            plan
            for plan, _state in self._stored_work_plans_with_state()
            if plan.stage is StreamStage.WINDOW_REDUCTION
        )
        dependencies = tuple(
            sorted(
                (
                    StreamWorkDependency(
                        upstream_work_logical_key=plan.work_logical_key,
                        criticality=DependencyCriticality.REQUIRED,
                    )
                    for plan in reductions
                ),
                key=lambda value: value.upstream_work_logical_key,
            )
        )
        subject = StreamSubjectRef(
            subject_type=StreamSubjectType.EXPECTED_WINDOW_PLAN_SEAL,
            subject_key=f"expected-window-plan-seal-v1:{seal.seal_semantic_sha256}",
            subject_semantic_sha256=seal.seal_semantic_sha256,
            capture_scope_digest=seal.capture_scope_digest,
            identity_policy_version=seal.seal_projection_version,
            schema_ref=seal.schema_ref,
        )
        authority_created_at = self._timestamp_now() if created_at is None else created_at
        authority_created = _parse_timestamp(authority_created_at)
        return create_stream_work_item_plan(
            schema_ref=self._schema_refs.stream_work_plan,
            stream_run_id=self._stream_run_id,
            source_subject=self._source_subject,
            stage=StreamStage.FINALIZATION,
            subject=subject,
            input_semantic_sha256=semantic_sha256(
                {
                    "version": STREAM_WINDOW_DAG_POLICY_VERSION,
                    "expected_plan_seal_semantic_sha256": seal.seal_semantic_sha256,
                    "terminal_closure_required": True,
                    "fixed_six_export_barrier_required": True,
                }
            ),
            config_semantic_sha256=self._dag_config_semantic_sha256,
            ordered_dependencies=dependencies,
            priority=100,
            sla_deadline_at=_format_timestamp(
                authority_created
                + timedelta(seconds=_LOCAL_STAGE_SLA_STEP_SECONDS * (len(_WINDOW_DAG_STAGES) + 1))
            ),
            created_at=authority_created_at,
        )

    def _internal_execution_projection(
        self,
        plan: StreamWorkItemPlan,
        *,
        plans_by_key: Mapping[str, StreamWorkItemPlan] | None = None,
    ) -> tuple[WorkItemPlan, tuple[WorkDependency, ...]]:
        """Project to V1 execution state without creating stream Wire data."""

        source_surrogate = str(
            uuid5(
                _INTERNAL_EXECUTION_NAMESPACE,
                f"source:{plan.source_subject.capture_scope_digest}",
            )
        )
        subject_surrogate = str(
            uuid5(
                _INTERNAL_EXECUTION_NAMESPACE,
                f"subject:{plan.subject.subject_type.value}:{plan.subject.subject_key}",
            )
        )
        execution = WorkItemPlan(
            work_item_id=plan.work_item_id,
            work_logical_key=plan.work_logical_key,
            run_id=plan.stream_run_id,
            mcap_id=source_surrogate,
            stage=_STAGE_EXECUTION_PROJECTION[plan.stage],
            subject_type=WorkItemSubjectType.WINDOW,
            subject_id=subject_surrogate,
            input_digest=plan.input_semantic_sha256,
            config_digest=semantic_sha256(
                {
                    "projection_version": INTERNAL_STREAM_EXECUTION_PROJECTION_VERSION,
                    "stream_config_semantic_sha256": plan.config_semantic_sha256,
                }
            ),
            priority=plan.priority,
            sla_deadline_at=plan.sla_deadline_at,
            execution_expiry_at=plan.execution_expiry_at,
            max_attempts=plan.max_attempts,
            trace_id=plan.trace_id,
            created_at=plan.created_at,
        )
        dependencies: list[WorkDependency] = []
        for dependency in plan.ordered_dependencies:
            upstream = (
                None
                if plans_by_key is None
                else plans_by_key.get(dependency.upstream_work_logical_key)
            )
            if upstream is None:
                upstream = self._stored_work_plan_by_key(dependency.upstream_work_logical_key)
            dependency_id = str(
                uuid5(
                    _INTERNAL_DEPENDENCY_NAMESPACE,
                    ":".join(
                        (
                            INTERNAL_STREAM_EXECUTION_PROJECTION_VERSION,
                            plan.work_item_id,
                            upstream.work_item_id,
                            dependency.criticality.value,
                        )
                    ),
                )
            )
            dependencies.append(
                WorkDependency(
                    dependency_id=dependency_id,
                    downstream_work_item_id=plan.work_item_id,
                    upstream_work_item_id=upstream.work_item_id,
                    criticality=dependency.criticality,
                )
            )
        return execution, tuple(sorted(dependencies, key=lambda value: value.dependency_id))

    def _stream_item(
        self,
        execution: WorkItem,
        *,
        stored: StoredStreamWorkPlan | None = None,
    ) -> StreamWorkItem:
        work_row = (
            self._stored_work_row(execution.work_item_id, require_published=False)
            if stored is None
            else stored
        )
        plan = _parse_exact(work_row.plan_json, StreamWorkItemPlan, "stream work plan")
        evidence = self._accepted_evidence(stored=work_row)
        return self._stream_item_from(plan, execution, evidence=evidence)

    def _stream_item_from(
        self,
        plan: StreamWorkItemPlan,
        execution: WorkItem,
        *,
        evidence: StreamTerminalEvidence | None,
    ) -> StreamWorkItem:
        state: StreamWorkItemState
        if evidence is not None:
            if execution.state is not WorkItemState.SUCCEEDED:
                raise StreamSchedulerCompositionError(
                    "accepted terminal evidence lacks succeeded execution projection"
                )
            state = _TERMINAL_STATE_BY_OUTCOME[evidence.outcome]
        else:
            nonterminal_state = _NONTERMINAL_STREAM_STATE.get(execution.state)
            if nonterminal_state is None:
                raise StreamSchedulerCompositionError(
                    "terminal execution projection lacks exact stream terminal evidence"
                )
            state = nonterminal_state
        return StreamWorkItem(
            **plan.model_dump(mode="python"),
            state=state,
            cancel_requested=execution.cancel_requested,
            lease_epoch=execution.lease_epoch,
            fencing_token=execution.fencing_token,
            leased_by=execution.leased_by,
            lease_expires_at=execution.lease_expires_at,
            attempt=execution.attempt,
            retry_not_before_at=execution.retry_not_before_at,
            terminal_evidence=evidence,
            updated_at=execution.updated_at,
            row_version=execution.row_version,
        )

    def _store_pending_terminal(
        self,
        lease: StreamWorkLease,
        evidence: StreamTerminalEvidence,
        *,
        authority_now: datetime,
    ) -> None:
        self._ledger.store_pending_terminal(
            work_item_id=lease.work_item_id,
            payload=canonical_json_bytes(evidence),
            lease_epoch=lease.lease_epoch,
            fencing_token=lease.fencing_token,
            worker_id=lease.worker_id,
            authority_now=_format_timestamp(authority_now),
            lease_expires_at=lease.lease_expires_at,
        )

    def _recover_terminal_acceptances(self) -> int:
        accepted = 0
        for stored in self._ledger.pending_work_rows(self.plan_key):
            execution = self._execution_scheduler.get(stored.work_item_id)
            if execution.state is not WorkItemState.SUCCEEDED:
                continue
            self._accept_pending_terminal(
                stored.work_item_id,
                stored=stored,
                execution=execution,
            )
            accepted += 1
        return accepted

    def _accept_pending_terminal(
        self,
        work_item_id: str,
        *,
        stored: StoredStreamWorkPlan | None = None,
        execution: WorkItem | None = None,
        pending_raw: bytes | None = None,
    ) -> None:
        execution = self._execution_scheduler.get(work_item_id) if execution is None else execution
        if execution.state is not WorkItemState.SUCCEEDED:
            raise StreamSchedulerCompositionError(
                "execution projection is not committed for terminal acceptance"
            )
        work_row = self._ledger.get_work(work_item_id) if stored is None else stored
        if work_row.work_item_id != work_item_id or work_row.plan_key != self.plan_key:
            raise StreamSchedulerCompositionError(
                "pending terminal work belongs to another stream graph"
            )
        if work_row.terminal_evidence_json is not None:
            return
        accepted_pending_raw = (
            work_row.pending_terminal_json if pending_raw is None else pending_raw
        )
        if accepted_pending_raw is None:
            raise StreamSchedulerCompositionError("committed work lacks pending terminal fact")
        evidence = _parse_exact(
            accepted_pending_raw,
            StreamTerminalEvidence,
            "pending terminal evidence",
        )
        if (
            execution.result_reference != _terminal_result_reference(evidence)
            or execution.result_sha256 != evidence.evidence_ref.exact_sha256
        ):
            raise StreamSchedulerCompositionError(
                "execution result does not bind to pending terminal evidence"
            )
        plan = _parse_exact(work_row.plan_json, StreamWorkItemPlan, "stream work plan")
        terminal_member_raw: bytes | None = None
        ordinal = work_row.expected_ordinal
        if plan.stage is StreamStage.WINDOW_REDUCTION:
            if ordinal is None:
                raise StreamSchedulerCompositionError(
                    "window reduction is missing expected ordinal"
                )
            window = self._ledger.window_at(self.plan_key, ordinal)
            if window is None:
                raise StreamSchedulerCompositionError("window reduction lacks expected declaration")
            declaration = _parse_exact(
                window.declaration_json,
                ExpectedWindowDeclaration,
                "expected declaration",
            )
            member = create_window_terminal_member(
                schema_ref=self._schema_refs.terminal_member,
                plan_key=self.plan_key,
                expected_ordinal=ordinal,
                window_key=declaration.window_key,
                window_semantic_sha256=declaration.window_semantic_sha256,
                terminal_outcome=evidence.outcome,
                terminal_work_item_id=plan.work_item_id,
                terminal_work_logical_key=plan.work_logical_key,
                terminal_evidence_ref=evidence.evidence_ref,
                terminal_policy_version=evidence.terminal_policy_version,
            )
            terminal_member_raw = canonical_json_bytes(member)
        self._ledger.accept_pending_terminal(
            work_item_id=work_item_id,
            expected_pending_json=accepted_pending_raw,
            terminal_member_json=terminal_member_raw,
            expected_ordinal=ordinal,
        )

    def _accepted_evidence(
        self,
        work_item_id: str | None = None,
        *,
        stored: StoredStreamWorkPlan | None = None,
    ) -> StreamTerminalEvidence | None:
        if stored is None:
            if work_item_id is None:
                raise TypeError("work_item_id is required when stored is absent")
            stored = self._stored_work_row(work_item_id, require_published=False)
        raw = stored.terminal_evidence_json
        return (
            None
            if raw is None
            else _parse_exact(raw, StreamTerminalEvidence, "stream terminal evidence")
        )

    def _remember_work_row(self, stored: StoredStreamWorkPlan) -> None:
        if stored.plan_key == self.plan_key:
            self._work_row_cache[stored.work_item_id] = stored

    def _remember_published_work_row(
        self,
        row: NewStreamWorkPlan | StoredStreamWorkPlan,
    ) -> None:
        if isinstance(row, StoredStreamWorkPlan):
            self._remember_work_row(replace(row, publication_state="PUBLISHED"))
            return
        self._remember_work_row(
            StoredStreamWorkPlan(
                work_item_id=row.work_item_id,
                work_logical_key=row.work_logical_key,
                plan_key=self.plan_key,
                expected_ordinal=row.expected_ordinal,
                role_order=row.role_order,
                stage=row.stage,
                plan_json=row.plan_json,
                publication_state="PUBLISHED",
            )
        )

    def _cached_published_work_row(self, work_item_id: str) -> StoredStreamWorkPlan | None:
        stored = self._work_row_cache.get(work_item_id)
        if (
            stored is None
            or stored.plan_key != self.plan_key
            or stored.publication_state != "PUBLISHED"
        ):
            return None
        return stored

    def _stored_work_row(
        self,
        work_item_id: str,
        *,
        require_published: bool,
    ) -> StoredStreamWorkPlan:
        stored = self._ledger.get_work(work_item_id)
        if stored.plan_key != self.plan_key:
            raise StreamSchedulerCompositionError("work item belongs to another stream graph")
        if require_published and stored.publication_state != "PUBLISHED":
            raise StreamSchedulerCompositionError("work item is not published for execution")
        self._remember_work_row(stored)
        return stored

    def _stored_work_plan(
        self,
        work_item_id: str,
        *,
        require_published: bool,
    ) -> StreamWorkItemPlan:
        stored = self._stored_work_row(work_item_id, require_published=require_published)
        return _parse_exact(stored.plan_json, StreamWorkItemPlan, "stream work plan")

    def _stored_work_plan_by_key(self, logical_key: str) -> StreamWorkItemPlan:
        stored = self._ledger.get_work_by_key(logical_key)
        if stored.plan_key != self.plan_key:
            raise StreamSchedulerCompositionError("upstream work belongs to another stream graph")
        return _parse_exact(
            stored.plan_json,
            StreamWorkItemPlan,
            "upstream stream work plan",
        )

    def _stored_work_plans_with_state(
        self,
    ) -> tuple[tuple[StreamWorkItemPlan, str], ...]:
        plans: list[tuple[StreamWorkItemPlan, str]] = []
        for row in self._ledger.work_plans(self.plan_key):
            self._remember_work_row(row)
            plans.append(
                (
                    _parse_exact(row.plan_json, StreamWorkItemPlan, "stream work plan"),
                    row.publication_state,
                )
            )
        return tuple(plans)

    def _verify_existing_window_member(
        self,
        window_row: StoredExpectedWindow,
        reduction_work_item_id: str,
    ) -> None:
        reduction_row = self._ledger.get_work(reduction_work_item_id)
        member_raw = window_row.terminal_member_json
        evidence_raw = reduction_row.terminal_evidence_json
        if member_raw is None and evidence_raw is None:
            return
        if member_raw is None or evidence_raw is None:
            raise StreamSchedulerCompositionError(
                "window member and accepted reduction evidence must coexist"
            )
        declaration = _parse_exact(
            window_row.declaration_json,
            ExpectedWindowDeclaration,
            "expected declaration",
        )
        evidence = _parse_exact(
            evidence_raw,
            StreamTerminalEvidence,
            "stream terminal evidence",
        )
        plan = _parse_exact(
            reduction_row.plan_json,
            StreamWorkItemPlan,
            "stream work plan",
        )
        expected = create_window_terminal_member(
            schema_ref=self._schema_refs.terminal_member,
            plan_key=self.plan_key,
            expected_ordinal=window_row.ordinal,
            window_key=declaration.window_key,
            window_semantic_sha256=declaration.window_semantic_sha256,
            terminal_outcome=evidence.outcome,
            terminal_work_item_id=plan.work_item_id,
            terminal_work_logical_key=plan.work_logical_key,
            terminal_evidence_ref=evidence.evidence_ref,
            terminal_policy_version=evidence.terminal_policy_version,
        )
        if canonical_json_bytes(expected) != member_raw:
            raise StreamSchedulerCompositionError(
                "window member does not bind to accepted reduction evidence"
            )

    def _register_plan(self) -> None:
        config_bytes = canonical_json_bytes(
            {
                "version": STREAM_WINDOW_DAG_POLICY_VERSION,
                "stream_run_id": self._stream_run_id,
                "dag_config_semantic_sha256": self._dag_config_semantic_sha256,
                "terminal_policy_version": self._terminal_policy_version,
                "backpressure_config": _backpressure_config_storage_payload(
                    self._backpressure_config,
                ),
                "schema_refs": {
                    "incremental_window": self._schema_refs.incremental_window,
                    "expected_declaration": self._schema_refs.expected_declaration,
                    "expected_plan_seal": self._schema_refs.expected_plan_seal,
                    "stream_work_plan": self._schema_refs.stream_work_plan,
                    "terminal_member": self._schema_refs.terminal_member,
                    "terminal_closure": self._schema_refs.terminal_closure,
                },
            }
        )
        plan_bytes = canonical_json_bytes(self._expected_plan)
        source_bytes = canonical_json_bytes(self._source_subject)
        self._ledger.register_plan(
            plan_key=self.plan_key,
            plan_json=plan_bytes,
            source_subject_json=source_bytes,
            composition_config_json=config_bytes,
        )

    def _verify_storage(self) -> None:
        stored_plan = self._ledger.get_plan(self.plan_key)
        parsed_plan = _parse_exact(
            stored_plan.plan_json, ExpectedWindowPlan, "expected window plan"
        )
        parsed_source = _parse_exact(
            stored_plan.source_subject_json,
            PreEosCaptureSubjectRef,
            "pre-EOS source subject",
        )
        if canonical_json_bytes(parsed_plan) != canonical_json_bytes(
            self._expected_plan
        ) or canonical_json_bytes(parsed_source) != canonical_json_bytes(self._source_subject):
            raise StreamSchedulerCompositionError(
                "stored plan or source differs from composition authority"
            )

        windows = self._ledger.windows(self.plan_key)
        if tuple(row.ordinal for row in windows) != tuple(range(len(windows))):
            raise StreamSchedulerCompositionError(
                "stored expected-window ordinals are not contiguous"
            )
        work_rows = self._ledger.work_plans(self.plan_key)
        parsed_work: dict[str, StreamWorkItemPlan] = {}
        work_rows_by_ordinal: dict[int | None, list[StoredStreamWorkPlan]] = {}
        for work_row in work_rows:
            work_rows_by_ordinal.setdefault(work_row.expected_ordinal, []).append(work_row)
            work_plan = _parse_exact(work_row.plan_json, StreamWorkItemPlan, "stream work plan")
            if (
                work_row.work_item_id != work_plan.work_item_id
                or work_row.work_logical_key != work_plan.work_logical_key
                or work_row.plan_key != self.plan_key
                or work_row.stage != work_plan.stage.value
            ):
                raise StreamSchedulerCompositionError(
                    "stream work row columns do not match plan JSON"
                )
            if work_plan.work_item_id in parsed_work:
                raise StreamSchedulerCompositionError("stream work rows contain duplicate IDs")
            parsed_work[work_plan.work_item_id] = work_plan
            accepted = (
                None
                if work_row.terminal_evidence_json is None
                else _parse_exact(
                    work_row.terminal_evidence_json,
                    StreamTerminalEvidence,
                    "terminal_evidence_json",
                )
            )
            pending = (
                None
                if work_row.pending_terminal_json is None
                else _parse_exact(
                    work_row.pending_terminal_json,
                    StreamTerminalEvidence,
                    "pending_terminal_json",
                )
            )
            if accepted is not None and pending is not None:
                raise StreamSchedulerCompositionError(
                    "accepted and pending terminal evidence cannot coexist"
                )
            pending_fence = (
                work_row.pending_lease_epoch,
                work_row.pending_fencing_token,
            )
            if (pending is None) != all(value is None for value in pending_fence):
                raise StreamSchedulerCompositionError(
                    "pending terminal evidence requires a complete fence"
                )
            for evidence in (accepted, pending):
                if (
                    evidence is not None
                    and evidence.terminal_policy_version != self._terminal_policy_version
                ):
                    raise StreamSchedulerCompositionError(
                        "stored terminal evidence violates the composition policy pin"
                    )

        declarations: list[ExpectedWindowDeclaration] = []
        previous_chain: str | None = None
        for window_row in windows:
            declaration = _parse_exact(
                window_row.declaration_json,
                ExpectedWindowDeclaration,
                "expected declaration",
            )
            window = _parse_exact(
                window_row.window_json,
                IncrementalWindow,
                "incremental window",
            )
            if (
                declaration.plan_key != self.plan_key
                or declaration.ordinal != window_row.ordinal
                or declaration.window_key != window.window_key
                or declaration.window_semantic_sha256 != window.window_semantic_sha256
                or declaration.requested_interval != window.requested_interval
                or declaration.effective_interval != window.effective_interval
                or declaration.ordered_six_slot_segment_or_explicit_absence_closure
                != window.camera_closure
                or declaration.previous_append_chain_sha256 != previous_chain
            ):
                raise StreamSchedulerCompositionError(
                    "expected declaration does not bind to its window row"
                )
            previous_chain = declaration.append_chain_sha256
            declarations.append(declaration)

            companions = tuple(work_rows_by_ordinal.get(window_row.ordinal, ()))
            if len(companions) != len(_WINDOW_DAG_STAGES):
                raise StreamSchedulerCompositionError(
                    "expected window lacks its exact canonical DAG"
                )
            first = parsed_work[companions[0].work_item_id]
            expected_plans = self._window_work_plans(
                window,
                created_at=first.created_at,
            )
            for role_order, (row, expected) in enumerate(
                zip(companions, expected_plans, strict=True)
            ):
                self._verify_stored_work_row(
                    row,
                    expected,
                    expected_ordinal=window_row.ordinal,
                    role_order=role_order,
                    allowed_publication_states={"PENDING", "PUBLISHED"},
                )
            self._verify_existing_window_member(
                window_row,
                expected_plans[-1].work_item_id,
            )

        if any(
            row.expected_ordinal is not None and not 0 <= row.expected_ordinal < len(windows)
            for row in work_rows
        ):
            raise StreamSchedulerCompositionError(
                "stream work references an absent expected ordinal"
            )

        declarations_tuple = tuple(declarations)
        finalization_rows = tuple(work_rows_by_ordinal.get(None, ()))
        seal: ExpectedWindowPlanSeal | None = None
        if stored_plan.seal_json is None:
            if finalization_rows:
                raise StreamSchedulerCompositionError(
                    "unsealed plan cannot contain finalization work"
                )
            if stored_plan.terminal_closure_json is not None:
                raise StreamSchedulerCompositionError(
                    "unsealed plan cannot contain terminal closure"
                )
        else:
            if stored_plan.planner_eos_sha256 is None:
                raise StreamSchedulerCompositionError(
                    "expected plan seal requires durable planner EOS"
                )
            seal = _parse_exact(
                stored_plan.seal_json,
                ExpectedWindowPlanSeal,
                "expected plan seal",
            )
            expected_seal = create_expected_window_plan_seal(
                schema_ref=self._schema_refs.expected_plan_seal,
                plan=self._expected_plan,
                declarations=declarations_tuple,
                eos_source_receipt_semantic_sha256=(seal.eos_source_receipt_semantic_sha256),
                final_source_timeline_semantic_sha256=(seal.final_source_timeline_semantic_sha256),
                final_duration_ns=seal.final_duration_ns,
                ordered_six_channel_health_closure_sha256=(
                    seal.ordered_six_channel_health_closure_sha256
                ),
                mapping_closure_semantic_sha256=(seal.mapping_closure_semantic_sha256),
                clock_or_alignment_closure_semantic_sha256=(
                    seal.clock_or_alignment_closure_semantic_sha256
                ),
            )
            if canonical_json_bytes(expected_seal) != stored_plan.seal_json:
                raise StreamSchedulerCompositionError(
                    "expected plan seal does not bind to current plan declarations"
                )
            if len(finalization_rows) != 1:
                raise StreamSchedulerCompositionError(
                    "sealed plan requires exact finalization work"
                )
            stored_finalization = parsed_work[finalization_rows[0].work_item_id]
            expected_finalization = self._finalization_work_plan(
                seal,
                created_at=stored_finalization.created_at,
            )
            finalization_states = (
                {"GATED"} if stored_plan.terminal_closure_json is None else {"PENDING", "PUBLISHED"}
            )
            self._verify_stored_work_row(
                finalization_rows[0],
                expected_finalization,
                expected_ordinal=None,
                role_order=len(_WINDOW_DAG_STAGES),
                allowed_publication_states=finalization_states,
            )

        members = tuple(
            _parse_exact(
                row.terminal_member_json,
                WindowTerminalMember,
                "window terminal member",
            )
            for row in windows
            if row.terminal_member_json is not None
        )
        if stored_plan.terminal_closure_json is not None:
            if seal is None:
                raise StreamSchedulerCompositionError(
                    "terminal closure requires an expected plan seal"
                )
            closure = _parse_exact(
                stored_plan.terminal_closure_json,
                WindowTerminalClosure,
                "window terminal closure",
            )
            expected_closure = create_window_terminal_closure(
                schema_ref=self._schema_refs.terminal_closure,
                plan_seal=seal,
                expected_declarations=declarations_tuple,
                members=members,
            )
            if canonical_json_bytes(expected_closure) != canonical_json_bytes(closure):
                raise StreamSchedulerCompositionError(
                    "terminal closure does not bind to plan members"
                )
            if stored_plan.export_manifest_sha256 is None or stored_plan.export_member_count != len(
                CAMERA_IDS
            ):
                raise StreamSchedulerCompositionError(
                    "terminal closure requires the fixed-six export barrier"
                )

    def _verify_stored_work_row(
        self,
        row: StoredStreamWorkPlan,
        expected: StreamWorkItemPlan,
        *,
        expected_ordinal: int | None,
        role_order: int,
        allowed_publication_states: set[str],
    ) -> None:
        if (
            row.work_item_id != expected.work_item_id
            or row.work_logical_key != expected.work_logical_key
            or row.plan_key != self.plan_key
            or row.expected_ordinal != expected_ordinal
            or row.role_order != role_order
            or row.stage != expected.stage.value
            or row.plan_json != canonical_json_bytes(expected)
            or row.publication_state not in allowed_publication_states
        ):
            raise StreamSchedulerCompositionError(
                "stream work row does not match its canonical DAG plan"
            )

    def _timestamp_now(self) -> str:
        return self._checked_now(None).isoformat(timespec="microseconds")

    def _checked_now(self, value: datetime | None) -> datetime:
        candidate = self._clock() if value is None else value
        if not isinstance(candidate, datetime):
            raise TypeError("stream scheduler clock must return datetime")
        if candidate.tzinfo is None or candidate.utcoffset() is None:
            raise ValueError("stream scheduler clock must be timezone-aware")
        return candidate.astimezone(UTC)

    def _observe(self, boundary: str) -> None:
        if self._boundary_observer is not None:
            self._boundary_observer(boundary)


def _watermark_source_facts_sha256(window: BoundedWindowPlan) -> str:
    return semantic_sha256(
        {
            "version": WATERMARK_SOURCE_FACTS_PROJECTION_VERSION,
            "window_ordinal": window.ordinal,
            "watermark_ns": None if window.watermark_ns is None else str(window.watermark_ns),
            "closure_reason": window.closure_reason.value,
            "window_planning_sha256": window.planning_sha256,
        }
    )


def _planner_finish_sha256(finish: PlannerFinish) -> str:
    return semantic_sha256(
        {
            "version": PLANNER_EOS_PROJECTION_VERSION,
            "closed_segment_semantic_sha256_values": [
                segment.semantic_sha256 for segment in finish.closed_segments
            ],
            "quality_targets": [
                {
                    "camera_id": target.camera_id.value,
                    "bucket_ordinal": str(target.bucket_ordinal),
                    "requested_target_ns": str(target.requested_target_ns),
                    "packet_traversal_index": str(target.packet.traversal_index),
                    "packet_payload_sha256": target.packet.payload_sha256,
                    "policy_version": target.policy_version,
                }
                for target in finish.quality_targets
            ],
            "window_planning_sha256_values": [window.planning_sha256 for window in finish.windows],
            "camera_facts": [
                {
                    "camera_id": fact.camera_id.value,
                    "packet_count": str(fact.packet_count),
                    "payload_bytes": str(fact.payload_bytes),
                    "first_timestamp_ns": (
                        None if fact.first_timestamp_ns is None else str(fact.first_timestamp_ns)
                    ),
                    "last_timestamp_ns": (
                        None if fact.last_timestamp_ns is None else str(fact.last_timestamp_ns)
                    ),
                    "first_sequence": (
                        None if fact.first_sequence is None else str(fact.first_sequence)
                    ),
                    "last_sequence": (
                        None if fact.last_sequence is None else str(fact.last_sequence)
                    ),
                    "sequence_gap_count": str(fact.sequence_gap_count),
                }
                for fact in finish.facts
            ],
        }
    )


def _stream_lease(lease: WorkLease) -> StreamWorkLease:
    return StreamWorkLease(
        work_item_id=lease.work_item_id,
        worker_id=lease.worker_id,
        lease_epoch=lease.lease_epoch,
        fencing_token=lease.fencing_token,
        lease_expires_at=lease.lease_expires_at,
    )


def _execution_lease(lease: StreamWorkLease) -> WorkLease:
    return WorkLease(
        work_item_id=lease.work_item_id,
        worker_id=lease.worker_id,
        lease_epoch=lease.lease_epoch,
        fencing_token=lease.fencing_token,
        lease_expires_at=lease.lease_expires_at,
    )


def _terminal_result_reference(evidence: StreamTerminalEvidence) -> str:
    return f"stream-terminal-evidence:{evidence.evidence_ref.artifact_id}"


def _new_stored_work_plan(
    plan: StreamWorkItemPlan,
    *,
    expected_ordinal: int | None,
    role_order: int,
    publication_state: str,
) -> NewStreamWorkPlan:
    return NewStreamWorkPlan(
        work_item_id=plan.work_item_id,
        work_logical_key=plan.work_logical_key,
        expected_ordinal=expected_ordinal,
        role_order=role_order,
        stage=plan.stage.value,
        plan_json=canonical_json_bytes(plan),
        publication_state=publication_state,
    )


def _strict_model[T: BaseModel](value: object, model_type: type[T], label: str) -> T:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    try:
        return model_type.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, ValidationError) as error:
        raise ValueError(f"{label} failed strict validation") from error


def _backpressure_config_storage_payload(config: BackpressureConfig) -> dict[str, object]:
    """Keep a default fixed policy byte-compatible with pre-controller plans."""

    legacy_fields = (
        "version",
        "queue_depth_threshold",
        "oldest_age_threshold_ms",
        "backlog_slope_threshold",
    )
    legacy_payload = {field: getattr(config, field) for field in legacy_fields}
    default = BackpressureConfig(**legacy_payload)
    controller_fields = (
        "controller_version",
        "controller_key",
        "worker_utilization_threshold",
        "controller_mode",
        "minimum_limit",
        "maximum_limit",
        "additive_increase",
        "minimum_rate_observation_interval_ms",
        "multiplicative_decrease",
        "cooldown_ms",
    )
    if all(getattr(config, field) == getattr(default, field) for field in controller_fields):
        return legacy_payload
    return config.model_dump(mode="json")


def _stored_backpressure_config(value: Mapping[str, object]) -> BackpressureConfig:
    """Strictly restore either the legacy fixed shape or an explicit controller shape."""

    try:
        return BackpressureConfig.model_validate_json(
            canonical_json_bytes(dict(value)),
            strict=True,
        )
    except (TypeError, ValidationError, ValueError) as error:
        raise StreamSchedulerCompositionError("persisted backpressure config is invalid") from error


def _parse_exact[T: BaseModel](payload: bytes, model_type: type[T], label: str) -> T:
    try:
        value = model_type.model_validate_json(payload, strict=True)
    except (UnicodeDecodeError, ValidationError, ValueError) as error:
        raise StreamSchedulerCompositionError(f"persisted {label} is invalid") from error
    if canonical_json_bytes(value) != payload:
        raise StreamSchedulerCompositionError(f"persisted {label} is not canonical JSON")
    return value


def _schema_refs_from_stored(
    value: _StoredSchedulerSchemaRefs,
) -> StreamSchedulerSchemaRefs:
    return StreamSchedulerSchemaRefs(
        incremental_window=value.incremental_window,
        expected_declaration=value.expected_declaration,
        expected_plan_seal=value.expected_plan_seal,
        stream_work_plan=value.stream_work_plan,
        terminal_member=value.terminal_member,
        terminal_closure=value.terminal_closure,
    )


def _require_uuid(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a UUID string")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a UUID string") from error
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be a lowercase canonical UUID")
    return value


def _require_sha256(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _parse_timestamp(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


__all__ = [
    "DEFAULT_STREAM_BACKPRESSURE_CONFIG",
    "INTERNAL_STREAM_EXECUTION_PROJECTION_VERSION",
    "PLANNER_EOS_PROJECTION_VERSION",
    "STREAM_WINDOW_DAG_POLICY_VERSION",
    "WATERMARK_SOURCE_FACTS_PROJECTION_VERSION",
    "DurableStreamWindowScheduler",
    "EosSealInputs",
    "StreamBacklogSnapshot",
    "StreamBackpressureSnapshot",
    "StreamBackpressureThrottle",
    "StreamDrainWorkSnapshot",
    "StreamExportBarrierSnapshot",
    "StreamSchedulerCompositionError",
    "StreamSchedulerSchemaRefs",
]
