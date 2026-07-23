"""Durable local composition for the pre-EOS stream work graph.

Expected declarations are committed before an internal execution projection is
sent to the local scheduler. The projection is not stream Wire evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final
from uuid import UUID, uuid5

from pydantic import BaseModel, ValidationError

from robata.adapters.sqlite_stream_work_ledger import (
    NewStreamWorkPlan,
    SQLiteStreamWorkLedger,
    StoredExpectedWindow,
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
from robata.queue.models import (
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
STREAM_WINDOW_DAG_POLICY_VERSION: Final = "stream-window-dag-v2"
WATERMARK_SOURCE_FACTS_PROJECTION_VERSION: Final = "bounded-watermark-source-facts-v1"
PLANNER_EOS_PROJECTION_VERSION: Final = "bounded-planner-eos-v1"

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
    (StreamStage.ACTION_DENSE, (StreamStage.EVENT_PROPOSAL,)),
    (StreamStage.BOUNDARY_REFINEMENT, (StreamStage.ACTION_DENSE,)),
    (
        StreamStage.WINDOW_REDUCTION,
        (
            StreamStage.WINDOW,
            StreamStage.QA_COARSE,
            StreamStage.QA_DENSE,
            StreamStage.EVENT_PROPOSAL,
            StreamStage.ACTION_DENSE,
            StreamStage.BOUNDARY_REFINEMENT,
        ),
    ),
)
_WINDOW_DAG_STAGES: Final = tuple(stage for stage, _dependencies in _WINDOW_DAG_TOPOLOGY)

# Local-conformance scheduling budgets. They are operational fields rather than
# logical-identity inputs and remain explicitly unqualified for production SLOs.
_LOCAL_STAGE_SLA_STEP_SECONDS: Final = 5 * 60

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


@dataclass(frozen=True, slots=True)
class StreamSchedulerSchemaRefs:
    """Pins for the already-registered WP1 contract family."""

    incremental_window: SchemaRef
    expected_declaration: SchemaRef
    expected_plan_seal: SchemaRef
    stream_work_plan: SchemaRef
    terminal_member: SchemaRef
    terminal_closure: SchemaRef


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
        self._clock = clock or (lambda: datetime.now(UTC))
        self._boundary_observer = boundary_observer
        if self._expected_plan.capture_scope_digest != self._source_subject.capture_scope_digest:
            raise ValueError("expected plan and source subject must share capture_scope_digest")
        self._register_plan()
        self._verify_storage()
        self.recover()

    @property
    def database_path(self) -> Path:
        return self._ledger.database_path

    @property
    def plan_key(self) -> str:
        return self._expected_plan.plan_key

    def append_emission(self, emission: PlannerEmission) -> None:
        """Commit every closed window before projecting child execution work."""

        if not isinstance(emission, PlannerEmission):
            raise TypeError("emission must be PlannerEmission")
        for window in emission.windows:
            self.append_window(window)

    def append_window(self, window: BoundedWindowPlan) -> ExpectedWindowDeclaration:
        """Idempotently append one expected window and its fixed local DAG."""

        if not isinstance(window, BoundedWindowPlan):
            raise TypeError("window must be BoundedWindowPlan")
        if window.capture_scope_digest != self._expected_plan.capture_scope_digest:
            raise StreamSchedulerCompositionError("window belongs to another capture scope")
        wire_window = window.to_incremental_window(self._schema_refs.incremental_window)
        stored_windows = self._ledger.windows(self.plan_key)
        if window.ordinal > len(stored_windows):
            raise StreamSchedulerCompositionError(
                "expected windows must be appended in contiguous planner order"
            )
        previous_chain = None
        if window.ordinal > 0:
            previous = _parse_exact(
                stored_windows[window.ordinal - 1].declaration_json,
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
        existing = stored_windows[window.ordinal] if window.ordinal < len(stored_windows) else None
        created_at: str | None = None
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
            companion_rows = tuple(
                row
                for row in self._ledger.work_plans(self.plan_key)
                if row.expected_ordinal == window.ordinal
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
        inserted = self._ledger.append_window(
            plan_key=self.plan_key,
            ordinal=window.ordinal,
            declaration_json=canonical_json_bytes(declaration),
            window_json=canonical_json_bytes(wire_window),
            work_plans=new_work,
        )
        if inserted:
            self._observe("expected_declaration_durable")
        self.recover()
        if existing is not None:
            self._verify_existing_window_member(existing, new_work[-1].work_item_id)
            return stored_declaration
        return declaration

    def seal(self, finish: PlannerFinish) -> None:
        """Persist planner EOS after appending every final partial window."""

        if not isinstance(finish, PlannerFinish):
            raise TypeError("finish must be PlannerFinish")
        for window in finish.windows:
            self.append_window(window)
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
        """Reconcile terminals and replay scheduler projections idempotently."""

        accepted = self._recover_terminal_acceptances()
        published = 0
        for plan, publication_state in self._stored_work_plans_with_state():
            if publication_state == "GATED":
                continue
            projected, dependencies = self._internal_execution_projection(plan)
            self._execution_scheduler.plan(projected, dependencies)
            self._observe("internal_execution_projected")
            if publication_state != "PUBLISHED":
                self._ledger.mark_published(plan.work_item_id)
                published += 1
        return accepted + published

    def claim(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        work_item_id: str | None = None,
        now: datetime | None = None,
    ) -> StreamWorkLeaseClaim | None:
        """Claim only work owned by this graph and return a typed lease."""

        checked_now = self._checked_now(now)
        self.recover()
        self._execution_scheduler.reconcile(now=checked_now)
        candidates: tuple[str, ...]
        if work_item_id is not None:
            self._stored_work_plan(work_item_id, require_published=True)
            candidates = (work_item_id,)
        else:
            ready: list[tuple[int, str, str]] = []
            for plan, state in self._stored_work_plans_with_state():
                if state != "PUBLISHED":
                    continue
                item = self._execution_scheduler.get(plan.work_item_id)
                if item.state is WorkItemState.READY:
                    ready.append((-plan.priority, plan.created_at, plan.work_item_id))
            candidates = tuple(value[2] for value in sorted(ready))
        for candidate in candidates:
            claim = self._execution_scheduler.claim(
                worker_id,
                lease_duration_seconds,
                work_item_id=candidate,
                now=checked_now,
            )
            if claim is not None:
                return StreamWorkLeaseClaim(
                    work_item=self._stream_item(claim.work_item),
                    lease=_stream_lease(claim.lease),
                )
        return None

    def start(
        self,
        lease: StreamWorkLease,
        *,
        now: datetime | None = None,
    ) -> StreamWorkItem:
        checked = _strict_model(lease, StreamWorkLease, "lease")
        self._stored_work_plan(checked.work_item_id, require_published=True)
        item = self._execution_scheduler.start(
            _execution_lease(checked), now=self._checked_now(now)
        )
        return self._stream_item(item)

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
        self._stored_work_plan(checked_lease.work_item_id, require_published=True)
        if checked_evidence.terminal_policy_version != self._terminal_policy_version:
            raise StreamSchedulerCompositionError(
                "terminal evidence does not use the composition policy pin"
            )
        checked_now = self._checked_now(now)
        if _parse_timestamp(checked_evidence.completed_at) > checked_now:
            raise StreamSchedulerCompositionError(
                "terminal evidence cannot complete after authority time"
            )
        accepted = self._accepted_evidence(checked_lease.work_item_id)
        if accepted is not None:
            if canonical_json_bytes(accepted) != canonical_json_bytes(checked_evidence):
                raise StreamSchedulerCompositionError(
                    "terminal replay changed accepted stream evidence"
                )
            return self.get(checked_lease.work_item_id)
        self._require_current_execution_lease(checked_lease, checked_now)
        self._store_pending_terminal(checked_lease, checked_evidence)
        self._execution_scheduler.succeed(
            _execution_lease(checked_lease),
            result_reference=_terminal_result_reference(checked_evidence),
            result_sha256=checked_evidence.evidence_ref.exact_sha256,
            now=checked_now,
        )
        self._observe("execution_terminal_committed")
        self._accept_pending_terminal(checked_lease.work_item_id)
        return self.get(checked_lease.work_item_id)

    def get(self, work_item_id: str) -> StreamWorkItem:
        self._recover_terminal_acceptances()
        self._stored_work_plan(work_item_id, require_published=False)
        return self._stream_item(self._execution_scheduler.get(work_item_id))

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
        self._recover_terminal_acceptances()
        counts: Counter[str] = Counter()
        active_created: list[datetime] = []
        finalization_published = False
        for plan, publication_state in self._stored_work_plans_with_state():
            if publication_state == "GATED":
                counts["GATED"] += 1
                active_created.append(_parse_timestamp(plan.created_at))
                continue
            item = self._execution_scheduler.get(plan.work_item_id)
            counts[item.state.value] += 1
            if item.state not in {
                WorkItemState.SUCCEEDED,
                WorkItemState.FAILED_PERMANENT,
                WorkItemState.SKIPPED_POLICY,
                WorkItemState.SKIPPED_NOT_NEEDED,
                WorkItemState.CANCELLED,
                WorkItemState.EXPIRED,
                WorkItemState.INVALIDATED,
            }:
                active_created.append(_parse_timestamp(plan.created_at))
            if plan.stage is StreamStage.FINALIZATION and publication_state == "PUBLISHED":
                finalization_published = True
        oldest = None
        if active_created:
            oldest = max(0.0, (checked_now - min(active_created)).total_seconds())
        return StreamBacklogSnapshot(
            state_counts=tuple(sorted(counts.items())),
            active_backlog=len(active_created),
            oldest_active_age_seconds=oldest,
            declared_window_count=len(self.declarations()),
            expected_plan_sealed=self.expected_plan_seal() is not None,
            terminal_member_count=len(self.terminal_members()),
            export_barrier_complete=self.export_barrier().complete,
            finalization_published=finalization_published,
        )

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
                    + timedelta(
                        seconds=_LOCAL_STAGE_SLA_STEP_SECONDS * (stage_index + 1)
                    )
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
                + timedelta(
                    seconds=_LOCAL_STAGE_SLA_STEP_SECONDS * (len(_WINDOW_DAG_STAGES) + 1)
                )
            ),
            created_at=authority_created_at,
        )

    def _internal_execution_projection(
        self, plan: StreamWorkItemPlan
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

    def _stream_item(self, execution: WorkItem) -> StreamWorkItem:
        plan = self._stored_work_plan(execution.work_item_id, require_published=False)
        evidence = self._accepted_evidence(execution.work_item_id)
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
    ) -> None:
        self._ledger.store_pending_terminal(
            work_item_id=lease.work_item_id,
            payload=canonical_json_bytes(evidence),
            lease_epoch=lease.lease_epoch,
            fencing_token=lease.fencing_token,
        )

    def _require_current_execution_lease(
        self,
        lease: StreamWorkLease,
        authority_now: datetime,
    ) -> None:
        execution = self._execution_scheduler.get(lease.work_item_id)
        if (
            execution.state not in {WorkItemState.LEASED, WorkItemState.RUNNING}
            or execution.work_item_id != lease.work_item_id
            or execution.leased_by != lease.worker_id
            or execution.lease_epoch != lease.lease_epoch
            or execution.fencing_token != lease.fencing_token
            or execution.lease_expires_at is None
            or _parse_timestamp(execution.lease_expires_at) <= authority_now
        ):
            raise WorkFenceError("work lease is stale, expired, or inactive")

    def _recover_terminal_acceptances(self) -> int:
        accepted = 0
        for work_item_id in self._ledger.pending_work_item_ids():
            execution = self._execution_scheduler.get(work_item_id)
            if execution.state is not WorkItemState.SUCCEEDED:
                continue
            self._accept_pending_terminal(work_item_id)
            accepted += 1
        return accepted

    def _accept_pending_terminal(self, work_item_id: str) -> None:
        execution = self._execution_scheduler.get(work_item_id)
        if execution.state is not WorkItemState.SUCCEEDED:
            raise StreamSchedulerCompositionError(
                "execution projection is not committed for terminal acceptance"
            )
        stored = self._ledger.get_work(work_item_id)
        if stored.terminal_evidence_json is not None:
            return
        pending_raw = stored.pending_terminal_json
        if pending_raw is None:
            raise StreamSchedulerCompositionError("committed work lacks pending terminal fact")
        evidence = _parse_exact(pending_raw, StreamTerminalEvidence, "pending terminal evidence")
        if (
            execution.result_reference != _terminal_result_reference(evidence)
            or execution.result_sha256 != evidence.evidence_ref.exact_sha256
        ):
            raise StreamSchedulerCompositionError(
                "execution result does not bind to pending terminal evidence"
            )
        plan = _parse_exact(stored.plan_json, StreamWorkItemPlan, "stream work plan")
        terminal_member_raw: bytes | None = None
        ordinal = stored.expected_ordinal
        if plan.stage is StreamStage.WINDOW_REDUCTION:
            if ordinal is None:
                raise StreamSchedulerCompositionError(
                    "window reduction is missing expected ordinal"
                )
            windows = self._ledger.windows(self.plan_key)
            if ordinal >= len(windows) or windows[ordinal].ordinal != ordinal:
                raise StreamSchedulerCompositionError("window reduction lacks expected declaration")
            declaration = _parse_exact(
                windows[ordinal].declaration_json,
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
            expected_pending_json=pending_raw,
            terminal_member_json=terminal_member_raw,
            expected_ordinal=ordinal,
        )

    def _accepted_evidence(self, work_item_id: str) -> StreamTerminalEvidence | None:
        raw = self._ledger.get_work(work_item_id).terminal_evidence_json
        return (
            None
            if raw is None
            else _parse_exact(raw, StreamTerminalEvidence, "stream terminal evidence")
        )

    def _stored_work_plan(
        self,
        work_item_id: str,
        *,
        require_published: bool,
    ) -> StreamWorkItemPlan:
        stored = self._ledger.get_work(work_item_id)
        if stored.plan_key != self.plan_key:
            raise StreamSchedulerCompositionError("work item belongs to another stream graph")
        if require_published and stored.publication_state != "PUBLISHED":
            raise StreamSchedulerCompositionError("work item is not published for execution")
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
        return tuple(
            (
                _parse_exact(row.plan_json, StreamWorkItemPlan, "stream work plan"),
                row.publication_state,
            )
            for row in self._ledger.work_plans(self.plan_key)
        )

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
        for work_row in work_rows:
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

            companions = tuple(
                row for row in work_rows if row.expected_ordinal == window_row.ordinal
            )
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
        finalization_rows = tuple(row for row in work_rows if row.expected_ordinal is None)
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
                    "bucket_ordinal": target.bucket_ordinal,
                    "requested_target_ns": str(target.requested_target_ns),
                    "packet_traversal_index": target.packet.traversal_index,
                    "packet_payload_sha256": target.packet.payload_sha256,
                    "policy_version": target.policy_version,
                }
                for target in finish.quality_targets
            ],
            "window_planning_sha256_values": [window.planning_sha256 for window in finish.windows],
            "camera_facts": [
                {
                    "camera_id": fact.camera_id.value,
                    "packet_count": fact.packet_count,
                    "payload_bytes": fact.payload_bytes,
                    "first_timestamp_ns": fact.first_timestamp_ns,
                    "last_timestamp_ns": fact.last_timestamp_ns,
                    "first_sequence": fact.first_sequence,
                    "last_sequence": fact.last_sequence,
                    "sequence_gap_count": fact.sequence_gap_count,
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


def _parse_exact[T: BaseModel](payload: bytes, model_type: type[T], label: str) -> T:
    try:
        value = model_type.model_validate_json(payload, strict=True)
    except (UnicodeDecodeError, ValidationError, ValueError) as error:
        raise StreamSchedulerCompositionError(f"persisted {label} is invalid") from error
    if canonical_json_bytes(value) != payload:
        raise StreamSchedulerCompositionError(f"persisted {label} is not canonical JSON")
    return value


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
    "INTERNAL_STREAM_EXECUTION_PROJECTION_VERSION",
    "PLANNER_EOS_PROJECTION_VERSION",
    "STREAM_WINDOW_DAG_POLICY_VERSION",
    "WATERMARK_SOURCE_FACTS_PROJECTION_VERSION",
    "DurableStreamWindowScheduler",
    "EosSealInputs",
    "StreamBacklogSnapshot",
    "StreamExportBarrierSnapshot",
    "StreamSchedulerCompositionError",
    "StreamSchedulerSchemaRefs",
]
