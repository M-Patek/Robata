"""Deterministic stream work planning and expected-window append contracts."""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from typing import Any, Literal, Self, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import model_validator

from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    CameraSlotClosure,
    DependencyCriticality,
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    PreEosCaptureSubjectRef,
    SixCameraSlotClosure,
    StreamPolicyBinding,
    StreamStage,
    StreamSubjectRef,
    validate_rfc3339,
)

STREAM_WORK_WIRE_VERSION: Literal["1.0"] = "1.0"
STREAM_WORK_PLAN_SCHEMA_ID = "https://schemas.robata.dev/stream-work-plan"
STREAM_WORK_PLAN_SCHEMA_VERSION = "1.0.0"
EXPECTED_WINDOW_PLAN_SCHEMA_ID = "https://schemas.robata.dev/expected-window-plan"
EXPECTED_WINDOW_DECLARATION_SCHEMA_ID = "https://schemas.robata.dev/expected-window-declaration"
EXPECTED_WINDOW_SEAL_SCHEMA_ID = "https://schemas.robata.dev/expected-window-plan-seal"
EXPECTED_WINDOW_PLAN_SCHEMA_VERSION = "1.0.0"
EXPECTED_WINDOW_DECLARATION_SCHEMA_VERSION = "1.0.0"
EXPECTED_WINDOW_SEAL_SCHEMA_VERSION = "1.0.0"
EXPECTED_WINDOW_PLAN_WIRE_VERSION: Literal["1.0"] = "1.0"
EXPECTED_WINDOW_DECLARATION_WIRE_VERSION: Literal["1.0"] = "1.0"
EXPECTED_WINDOW_SEAL_WIRE_VERSION: Literal["1.0"] = "1.0"
WORK_PROJECTION_VERSION = "stream-work-plan-semantic-v1"
WORK_KEY_POLICY_VERSION = "stream-work-key-v1"
WORK_KEY_NAMESPACE = "stream-work-v1"
PLAN_PROJECTION_VERSION = "expected-window-plan-semantic-v1"
PLAN_IDENTITY_POLICY_VERSION = "expected-window-plan-identity-v1"
PLAN_KEY_NAMESPACE = "expected-window-plan-v1"
DECLARATION_PROJECTION_VERSION = "expected-window-declaration-semantic-v1"
APPEND_CHAIN_VERSION = "expected-window-plan-append-v1"
EXPECTED_MEMBER_ROOT_VERSION = "expected-window-member-root-v1"
CHILD_DELIVERY_NAMESPACE = "expected-window-child-delivery-v1"


def _namespace(label: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"robata:stream-namespace:{label}")


STREAM_WORK_V1_NAMESPACE = _namespace(WORK_KEY_NAMESPACE)
EXPECTED_WINDOW_PLAN_V1_NAMESPACE = _namespace(PLAN_KEY_NAMESPACE)
EXPECTED_WINDOW_CHILD_DELIVERY_V1_NAMESPACE = _namespace(CHILD_DELIVERY_NAMESPACE)


class ExpectedWindowPlanState(StrEnum):
    OPEN = "OPEN"
    SEALED = "SEALED"


def _interval_projection(interval: NanosecondInterval) -> dict[str, str]:
    return {"start_ns": str(interval.start_ns), "end_ns": str(interval.end_ns)}


def _closure_projection(slots: tuple[CameraSlotClosure, ...]) -> list[dict[str, object]]:
    return [slot.model_dump(mode="json") for slot in slots]


class StreamWorkDependency(StrictModel):
    """Canonical dependency projection; criticality is identity-bearing."""

    upstream_work_logical_key: NonEmptyString
    criticality: DependencyCriticality = DependencyCriticality.REQUIRED


class StreamWorkItemPlan(StrictModel):
    """Execution-scoped stream work plan, distinct from V1 MCAP work."""

    schema_version: Literal["1.0"] = STREAM_WORK_WIRE_VERSION
    schema_ref: SchemaRef
    work_item_id: OpaqueUuid
    work_logical_key: NonEmptyString
    stream_run_id: OpaqueUuid
    source_subject: PreEosCaptureSubjectRef
    stage: StreamStage
    subject: StreamSubjectRef
    ordered_dependencies: tuple[StreamWorkDependency, ...] = ()
    input_semantic_sha256: Sha256Digest
    config_semantic_sha256: Sha256Digest
    work_projection_version: SchemaVersion = WORK_PROJECTION_VERSION
    work_key_policy_version: SchemaVersion = WORK_KEY_POLICY_VERSION
    priority: NonNegativeInt = 0
    sla_deadline_at: Rfc3339Timestamp | None = None
    execution_expiry_at: Rfc3339Timestamp | None = None
    max_attempts: PositiveInt = 3
    trace_id: NonEmptyString | None = None
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.work_projection_version != WORK_PROJECTION_VERSION:
            raise ValueError("stream work uses the registered projection version")
        if self.work_key_policy_version != WORK_KEY_POLICY_VERSION:
            raise ValueError("stream work uses the registered key policy version")
        if self.subject.capture_scope_digest != self.source_subject.capture_scope_digest:
            raise ValueError("stream work subject must bind to source capture_scope_digest")
        if tuple(self.ordered_dependencies) != tuple(
            sorted(self.ordered_dependencies, key=lambda dep: dep.upstream_work_logical_key)
        ):
            raise ValueError("stream work dependencies must be canonically ordered")
        if len({dep.upstream_work_logical_key for dep in self.ordered_dependencies}) != len(
            self.ordered_dependencies
        ):
            raise ValueError("stream work dependency keys must be unique")
        if self.sla_deadline_at is not None:
            validate_rfc3339(self.sla_deadline_at, "sla_deadline_at")
        if self.execution_expiry_at is not None:
            validate_rfc3339(self.execution_expiry_at, "execution_expiry_at")
        expected = stream_work_semantic_sha256(self)
        if self.work_logical_key != derive_work_logical_key(expected):
            raise ValueError("work_logical_key does not match the work projection")
        if self.work_item_id != derive_work_item_id(expected):
            raise ValueError("work_item_id does not match work_logical_key")
        return self

    @property
    def semantic_sha256(self) -> Sha256Digest:
        return stream_work_semantic_sha256(self)


def stream_work_semantic_projection(plan: StreamWorkItemPlan) -> dict[str, object]:
    """Identity projection; authority ``created_at`` and schema pin are excluded."""

    return {
        "work_projection_version": plan.work_projection_version,
        "work_key_policy_version": plan.work_key_policy_version,
        "stream_run_id": plan.stream_run_id,
        "capture_scope_digest": plan.source_subject.capture_scope_digest,
        "stage": plan.stage.value,
        "typed_subject_key": plan.subject.subject_key,
        "typed_subject_semantic_sha256": plan.subject.subject_semantic_sha256,
        "ordered_dependency_projections": [
            {
                "upstream_work_logical_key": dependency.upstream_work_logical_key,
                "criticality": dependency.criticality.value,
            }
            for dependency in plan.ordered_dependencies
        ],
        "input_semantic_sha256": plan.input_semantic_sha256,
        "config_semantic_sha256": plan.config_semantic_sha256,
    }


def stream_work_semantic_sha256(plan: StreamWorkItemPlan) -> Sha256Digest:
    return semantic_sha256(stream_work_semantic_projection(plan))


def derive_work_logical_key(work_digest: Sha256Digest) -> str:
    return f"{WORK_KEY_NAMESPACE}:{work_digest}"


def derive_work_item_id(work_digest: Sha256Digest) -> OpaqueUuid:
    return str(uuid5(STREAM_WORK_V1_NAMESPACE, derive_work_logical_key(work_digest)))


def create_stream_work_item_plan(
    *,
    schema_ref: SchemaRef,
    stream_run_id: OpaqueUuid,
    source_subject: PreEosCaptureSubjectRef,
    stage: StreamStage,
    subject: StreamSubjectRef,
    input_semantic_sha256: Sha256Digest,
    config_semantic_sha256: Sha256Digest,
    ordered_dependencies: tuple[StreamWorkDependency, ...] = (),
    priority: int = 0,
    sla_deadline_at: str | None = None,
    execution_expiry_at: str | None = None,
    max_attempts: int = 3,
    trace_id: str | None = None,
    created_at: Rfc3339Timestamp,
) -> StreamWorkItemPlan:
    values = {
        "schema_ref": schema_ref,
        "stream_run_id": stream_run_id,
        "source_subject": source_subject,
        "stage": stage,
        "subject": subject,
        "input_semantic_sha256": input_semantic_sha256,
        "config_semantic_sha256": config_semantic_sha256,
        "ordered_dependencies": ordered_dependencies,
        "priority": priority,
        "sla_deadline_at": sla_deadline_at,
        "execution_expiry_at": execution_expiry_at,
        "max_attempts": max_attempts,
        "trace_id": trace_id,
        "created_at": created_at,
    }
    # Build the preimage directly so authority timestamps cannot accidentally enter it.
    projection = {
        "work_projection_version": WORK_PROJECTION_VERSION,
        "work_key_policy_version": WORK_KEY_POLICY_VERSION,
        "stream_run_id": stream_run_id,
        "capture_scope_digest": source_subject.capture_scope_digest,
        "stage": stage.value,
        "typed_subject_key": subject.subject_key,
        "typed_subject_semantic_sha256": subject.subject_semantic_sha256,
        "ordered_dependency_projections": [
            {
                "upstream_work_logical_key": dependency.upstream_work_logical_key,
                "criticality": dependency.criticality.value,
            }
            for dependency in sorted(
                ordered_dependencies, key=lambda dep: dep.upstream_work_logical_key
            )
        ],
        "input_semantic_sha256": input_semantic_sha256,
        "config_semantic_sha256": config_semantic_sha256,
    }
    digest = semantic_sha256(projection)
    return StreamWorkItemPlan(
        work_item_id=derive_work_item_id(digest),
        work_logical_key=derive_work_logical_key(digest),
        **cast(dict[str, Any], values),
    )


class ExpectedWindowPlan(StrictModel):
    """Source/policy-derived expected set, independent of execution outcomes."""

    schema_version: Literal["1.0"] = EXPECTED_WINDOW_PLAN_WIRE_VERSION
    schema_ref: SchemaRef
    plan_id: OpaqueUuid
    plan_key: NonEmptyString
    plan_digest: Sha256Digest
    capture_scope_digest: Sha256Digest
    segmentation_policy_version: SchemaVersion
    segmentation_policy_semantic_sha256: Sha256Digest
    window_policy_version: SchemaVersion
    window_policy_semantic_sha256: Sha256Digest
    watermark_policy_version: SchemaVersion
    watermark_policy_semantic_sha256: Sha256Digest
    lateness_policy_version: SchemaVersion
    lateness_policy_semantic_sha256: Sha256Digest
    idle_source_policy_version: SchemaVersion
    idle_source_policy_semantic_sha256: Sha256Digest
    planner_version: SchemaVersion
    plan_projection_version: SchemaVersion = PLAN_PROJECTION_VERSION
    plan_identity_policy_version: SchemaVersion = PLAN_IDENTITY_POLICY_VERSION
    state: ExpectedWindowPlanState = ExpectedWindowPlanState.OPEN

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.plan_projection_version != PLAN_PROJECTION_VERSION:
            raise ValueError("expected-window plan uses the registered projection version")
        if self.plan_identity_policy_version != PLAN_IDENTITY_POLICY_VERSION:
            raise ValueError("expected-window plan uses the registered identity policy version")
        expected = expected_window_plan_semantic_sha256(self)
        if self.plan_digest != expected:
            raise ValueError("plan_digest does not match policy projection")
        if self.plan_key != derive_plan_key(expected):
            raise ValueError("plan_key does not match plan_digest")
        if self.plan_id != derive_plan_id(expected):
            raise ValueError("plan_id does not match plan_key")
        return self


def expected_window_plan_semantic_projection(plan: ExpectedWindowPlan) -> dict[str, object]:
    return {
        "plan_projection_version": plan.plan_projection_version,
        "plan_identity_policy_version": plan.plan_identity_policy_version,
        "capture_scope_digest": plan.capture_scope_digest,
        "segmentation_policy_binding": {
            "version": plan.segmentation_policy_version,
            "semantic_sha256": plan.segmentation_policy_semantic_sha256,
        },
        "window_policy_binding": {
            "version": plan.window_policy_version,
            "semantic_sha256": plan.window_policy_semantic_sha256,
        },
        "watermark_policy_binding": {
            "version": plan.watermark_policy_version,
            "semantic_sha256": plan.watermark_policy_semantic_sha256,
        },
        "lateness_policy_binding": {
            "version": plan.lateness_policy_version,
            "semantic_sha256": plan.lateness_policy_semantic_sha256,
        },
        "idle_source_policy_binding": {
            "version": plan.idle_source_policy_version,
            "semantic_sha256": plan.idle_source_policy_semantic_sha256,
        },
        "planner_version": plan.planner_version,
    }


def expected_window_plan_semantic_sha256(plan: ExpectedWindowPlan) -> Sha256Digest:
    return semantic_sha256(expected_window_plan_semantic_projection(plan))


def derive_plan_key(plan_digest: Sha256Digest) -> str:
    return f"{PLAN_KEY_NAMESPACE}:{plan_digest}"


def derive_plan_id(plan_digest: Sha256Digest) -> OpaqueUuid:
    return str(uuid5(EXPECTED_WINDOW_PLAN_V1_NAMESPACE, derive_plan_key(plan_digest)))


def create_expected_window_plan(
    *,
    schema_ref: SchemaRef,
    capture_scope_digest: Sha256Digest,
    segmentation_policy: StreamPolicyBinding,
    window_policy: StreamPolicyBinding,
    watermark_policy: StreamPolicyBinding,
    lateness_policy: StreamPolicyBinding,
    idle_source_policy: StreamPolicyBinding,
    planner_version: str,
) -> ExpectedWindowPlan:
    values = {
        "schema_ref": schema_ref,
        "capture_scope_digest": capture_scope_digest,
        "segmentation_policy_version": segmentation_policy.version,
        "segmentation_policy_semantic_sha256": segmentation_policy.semantic_sha256,
        "window_policy_version": window_policy.version,
        "window_policy_semantic_sha256": window_policy.semantic_sha256,
        "watermark_policy_version": watermark_policy.version,
        "watermark_policy_semantic_sha256": watermark_policy.semantic_sha256,
        "lateness_policy_version": lateness_policy.version,
        "lateness_policy_semantic_sha256": lateness_policy.semantic_sha256,
        "idle_source_policy_version": idle_source_policy.version,
        "idle_source_policy_semantic_sha256": idle_source_policy.semantic_sha256,
        "planner_version": planner_version,
    }
    projection = {
        "plan_projection_version": PLAN_PROJECTION_VERSION,
        "plan_identity_policy_version": PLAN_IDENTITY_POLICY_VERSION,
        "capture_scope_digest": capture_scope_digest,
        "segmentation_policy_binding": segmentation_policy.model_dump(mode="json"),
        "window_policy_binding": window_policy.model_dump(mode="json"),
        "watermark_policy_binding": watermark_policy.model_dump(mode="json"),
        "lateness_policy_binding": lateness_policy.model_dump(mode="json"),
        "idle_source_policy_binding": idle_source_policy.model_dump(mode="json"),
        "planner_version": planner_version,
    }
    digest = semantic_sha256(projection)
    return ExpectedWindowPlan(
        plan_id=derive_plan_id(digest),
        plan_key=derive_plan_key(digest),
        plan_digest=digest,
        **cast(dict[str, Any], values),
    )


class ExpectedWindowDeclaration(StrictModel):
    """Immutable contiguous append record in an open expected-window plan."""

    schema_version: Literal["1.0"] = EXPECTED_WINDOW_DECLARATION_WIRE_VERSION
    schema_ref: SchemaRef
    plan_key: NonEmptyString
    ordinal: NonNegativeInt
    window_key: NonEmptyString
    window_semantic_sha256: Sha256Digest
    requested_interval: NanosecondInterval
    effective_interval: NanosecondInterval
    ordered_six_slot_segment_or_explicit_absence_closure: tuple[CameraSlotClosure, ...]
    watermark_source_facts_sha256: Sha256Digest
    declaration_projection_version: SchemaVersion = DECLARATION_PROJECTION_VERSION
    declaration_semantic_sha256: Sha256Digest
    previous_append_chain_sha256: Sha256Digest | None = None
    append_chain_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_declaration(self) -> Self:
        if self.declaration_projection_version != DECLARATION_PROJECTION_VERSION:
            raise ValueError("expected-window declaration uses the registered projection version")
        if (
            self.effective_interval.start_ns < self.requested_interval.start_ns
            or self.effective_interval.end_ns > self.requested_interval.end_ns
        ):
            raise ValueError(
                "declaration effective interval must be contained by requested interval"
            )
        if self.ordinal == 0 and self.previous_append_chain_sha256 is not None:
            raise ValueError("ordinal zero declaration must start a fresh append chain")
        if self.ordinal > 0 and self.previous_append_chain_sha256 is None:
            raise ValueError("nonzero declaration must reference the previous append chain")
        SixCameraSlotClosure(slots=self.ordered_six_slot_segment_or_explicit_absence_closure)
        expected = expected_window_declaration_semantic_sha256(self)
        if self.declaration_semantic_sha256 != expected:
            raise ValueError("declaration_semantic_sha256 does not match declaration fields")
        expected_chain = compute_append_chain_sha256(
            plan_key=self.plan_key,
            ordinal=self.ordinal,
            declaration_semantic_sha256=self.declaration_semantic_sha256,
            previous=self.previous_append_chain_sha256,
        )
        if self.append_chain_sha256 != expected_chain:
            raise ValueError("append_chain_sha256 does not match declaration chain")
        return self


def expected_window_declaration_semantic_projection(
    declaration: ExpectedWindowDeclaration,
) -> dict[str, object]:
    return {
        "declaration_projection_version": declaration.declaration_projection_version,
        "plan_key": declaration.plan_key,
        "ordinal": declaration.ordinal,
        "window_key": declaration.window_key,
        "window_semantic_sha256": declaration.window_semantic_sha256,
        "requested_interval": _interval_projection(declaration.requested_interval),
        "effective_interval": _interval_projection(declaration.effective_interval),
        "ordered_six_slot_segment_or_explicit_absence_closure": _closure_projection(
            declaration.ordered_six_slot_segment_or_explicit_absence_closure
        ),
        "watermark_source_facts_sha256": declaration.watermark_source_facts_sha256,
    }


def expected_window_declaration_semantic_sha256(
    declaration: ExpectedWindowDeclaration,
) -> Sha256Digest:
    return semantic_sha256(expected_window_declaration_semantic_projection(declaration))


def compute_append_chain_sha256(
    *,
    plan_key: str,
    ordinal: int,
    declaration_semantic_sha256: Sha256Digest,
    previous: Sha256Digest | None,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "version": APPEND_CHAIN_VERSION,
            "plan_key": plan_key,
            "ordinal": ordinal,
            "declaration_semantic_sha256": declaration_semantic_sha256,
            "previous": previous,
        }
    )


def create_expected_window_declaration(
    *,
    schema_ref: SchemaRef,
    plan_key: str,
    ordinal: int,
    window_key: str,
    window_semantic_sha256: Sha256Digest,
    requested_interval: NanosecondInterval,
    effective_interval: NanosecondInterval,
    ordered_six_slot_segment_or_explicit_absence_closure: tuple[CameraSlotClosure, ...],
    watermark_source_facts_sha256: Sha256Digest,
    previous_append_chain_sha256: Sha256Digest | None,
) -> ExpectedWindowDeclaration:
    projection = {
        "declaration_projection_version": DECLARATION_PROJECTION_VERSION,
        "plan_key": plan_key,
        "ordinal": ordinal,
        "window_key": window_key,
        "window_semantic_sha256": window_semantic_sha256,
        "requested_interval": _interval_projection(requested_interval),
        "effective_interval": _interval_projection(effective_interval),
        "ordered_six_slot_segment_or_explicit_absence_closure": _closure_projection(
            ordered_six_slot_segment_or_explicit_absence_closure
        ),
        "watermark_source_facts_sha256": watermark_source_facts_sha256,
    }
    declaration_digest = semantic_sha256(projection)
    chain_digest = compute_append_chain_sha256(
        plan_key=plan_key,
        ordinal=ordinal,
        declaration_semantic_sha256=declaration_digest,
        previous=previous_append_chain_sha256,
    )
    return ExpectedWindowDeclaration(
        schema_ref=schema_ref,
        plan_key=plan_key,
        ordinal=ordinal,
        window_key=window_key,
        window_semantic_sha256=window_semantic_sha256,
        requested_interval=requested_interval,
        effective_interval=effective_interval,
        ordered_six_slot_segment_or_explicit_absence_closure=(
            ordered_six_slot_segment_or_explicit_absence_closure
        ),
        watermark_source_facts_sha256=watermark_source_facts_sha256,
        declaration_semantic_sha256=declaration_digest,
        previous_append_chain_sha256=previous_append_chain_sha256,
        append_chain_sha256=chain_digest,
    )


def compute_ordered_expected_member_root(
    declarations: tuple[ExpectedWindowDeclaration, ...],
) -> Sha256Digest:
    """Hash the canonical expected membership independently of work outcomes."""

    ordered = tuple(sorted(declarations, key=lambda declaration: declaration.ordinal))
    return semantic_sha256(
        {
            "version": EXPECTED_MEMBER_ROOT_VERSION,
            "ordered_expected_members": [
                {
                    "ordinal": declaration.ordinal,
                    "window_key": declaration.window_key,
                    "window_semantic_sha256": declaration.window_semantic_sha256,
                    "declaration_semantic_sha256": declaration.declaration_semantic_sha256,
                }
                for declaration in ordered
            ],
        }
    )


def derive_child_delivery_id(
    *,
    plan_key: str,
    ordinal: int,
    declaration_semantic_sha256: Sha256Digest,
    child_work_logical_key: str,
) -> OpaqueUuid:
    material = f"{plan_key}:{ordinal}:{declaration_semantic_sha256}:{child_work_logical_key}"
    return str(uuid5(EXPECTED_WINDOW_CHILD_DELIVERY_V1_NAMESPACE, material))


class ExpectedWindowPlanSeal(StrictModel):
    """One-way EOS seal derived only from source/policy facts."""

    schema_version: Literal["1.0"] = EXPECTED_WINDOW_SEAL_WIRE_VERSION
    schema_ref: SchemaRef
    plan_key: NonEmptyString
    capture_scope_digest: Sha256Digest
    eos_source_receipt_semantic_sha256: Sha256Digest
    final_source_timeline_semantic_sha256: Sha256Digest
    final_duration_ns: int
    ordered_six_channel_health_closure_sha256: Sha256Digest
    mapping_closure_semantic_sha256: Sha256Digest
    clock_or_alignment_closure_semantic_sha256: Sha256Digest
    segmentation_policy_version: SchemaVersion
    segmentation_policy_semantic_sha256: Sha256Digest
    window_policy_version: SchemaVersion
    window_policy_semantic_sha256: Sha256Digest
    watermark_policy_version: SchemaVersion
    watermark_policy_semantic_sha256: Sha256Digest
    lateness_policy_version: SchemaVersion
    lateness_policy_semantic_sha256: Sha256Digest
    idle_source_policy_version: SchemaVersion
    idle_source_policy_semantic_sha256: Sha256Digest
    planner_version: SchemaVersion
    expected_member_count: NonNegativeInt
    first_ordinal: NonNegativeInt | None = None
    last_ordinal_or_none: NonNegativeInt | None = None
    final_append_chain_sha256: Sha256Digest | None = None
    ordered_expected_member_root_sha256: Sha256Digest
    seal_projection_version: SchemaVersion = "expected-window-plan-seal-semantic-v1"
    seal_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_seal(self) -> Self:
        if self.seal_projection_version != "expected-window-plan-seal-semantic-v1":
            raise ValueError("expected-window seal uses the registered projection version")
        if self.final_duration_ns < 0:
            raise ValueError("final_duration_ns must be nonnegative")
        if self.expected_member_count == 0:
            if self.first_ordinal is not None or self.last_ordinal_or_none is not None:
                raise ValueError("empty expected plan cannot have ordinal bounds")
        elif self.first_ordinal != 0 or self.last_ordinal_or_none != self.expected_member_count - 1:
            raise ValueError("nonempty expected plan must have contiguous ordinal bounds")
        expected = expected_window_plan_seal_semantic_sha256(self)
        if self.seal_semantic_sha256 != expected:
            raise ValueError("seal_semantic_sha256 does not match seal projection")
        return self


def expected_window_plan_seal_semantic_projection(
    seal: ExpectedWindowPlanSeal,
) -> dict[str, object]:
    return {
        "seal_projection_version": seal.seal_projection_version,
        "plan_key": seal.plan_key,
        "capture_scope_digest": seal.capture_scope_digest,
        "eos_source_receipt_semantic_sha256": seal.eos_source_receipt_semantic_sha256,
        "final_source_timeline_semantic_sha256": seal.final_source_timeline_semantic_sha256,
        "final_duration_ns": str(seal.final_duration_ns),
        "ordered_six_channel_health_closure_sha256": seal.ordered_six_channel_health_closure_sha256,
        "mapping_closure_semantic_sha256": seal.mapping_closure_semantic_sha256,
        "clock_or_alignment_closure_semantic_sha256": (
            seal.clock_or_alignment_closure_semantic_sha256
        ),
        "segmentation_policy_binding": {
            "version": seal.segmentation_policy_version,
            "semantic_sha256": seal.segmentation_policy_semantic_sha256,
        },
        "window_policy_binding": {
            "version": seal.window_policy_version,
            "semantic_sha256": seal.window_policy_semantic_sha256,
        },
        "watermark_policy_binding": {
            "version": seal.watermark_policy_version,
            "semantic_sha256": seal.watermark_policy_semantic_sha256,
        },
        "lateness_policy_binding": {
            "version": seal.lateness_policy_version,
            "semantic_sha256": seal.lateness_policy_semantic_sha256,
        },
        "idle_source_policy_binding": {
            "version": seal.idle_source_policy_version,
            "semantic_sha256": seal.idle_source_policy_semantic_sha256,
        },
        "planner_version": seal.planner_version,
        "expected_member_count": seal.expected_member_count,
        "first_ordinal": seal.first_ordinal,
        "last_ordinal_or_none": seal.last_ordinal_or_none,
        "final_append_chain_sha256": seal.final_append_chain_sha256,
        "ordered_expected_member_root_sha256": seal.ordered_expected_member_root_sha256,
    }


def expected_window_plan_seal_semantic_sha256(seal: ExpectedWindowPlanSeal) -> Sha256Digest:
    return semantic_sha256(expected_window_plan_seal_semantic_projection(seal))


def create_expected_window_plan_seal(
    *,
    schema_ref: SchemaRef,
    plan: ExpectedWindowPlan,
    declarations: tuple[ExpectedWindowDeclaration, ...],
    eos_source_receipt_semantic_sha256: Sha256Digest,
    final_source_timeline_semantic_sha256: Sha256Digest,
    final_duration_ns: int,
    ordered_six_channel_health_closure_sha256: Sha256Digest,
    mapping_closure_semantic_sha256: Sha256Digest,
    clock_or_alignment_closure_semantic_sha256: Sha256Digest,
) -> ExpectedWindowPlanSeal:
    if plan.state is not ExpectedWindowPlanState.OPEN:
        raise ValueError("only an open expected-window plan may be sealed")
    ordered = tuple(sorted(declarations, key=lambda declaration: declaration.ordinal))
    if tuple(declaration.ordinal for declaration in ordered) != tuple(range(len(ordered))):
        raise ValueError("expected-window declarations must be contiguous")
    if any(declaration.plan_key != plan.plan_key for declaration in ordered):
        raise ValueError("expected-window declaration plan_key does not match plan")
    for previous, current in pairwise(ordered):
        if current.previous_append_chain_sha256 != previous.append_chain_sha256:
            raise ValueError("expected-window declaration append chain is discontinuous")

    expected_count = len(ordered)
    values = {
        "schema_ref": schema_ref,
        "plan_key": plan.plan_key,
        "capture_scope_digest": plan.capture_scope_digest,
        "eos_source_receipt_semantic_sha256": eos_source_receipt_semantic_sha256,
        "final_source_timeline_semantic_sha256": final_source_timeline_semantic_sha256,
        "final_duration_ns": final_duration_ns,
        "ordered_six_channel_health_closure_sha256": (ordered_six_channel_health_closure_sha256),
        "mapping_closure_semantic_sha256": mapping_closure_semantic_sha256,
        "clock_or_alignment_closure_semantic_sha256": (clock_or_alignment_closure_semantic_sha256),
        "segmentation_policy_version": plan.segmentation_policy_version,
        "segmentation_policy_semantic_sha256": plan.segmentation_policy_semantic_sha256,
        "window_policy_version": plan.window_policy_version,
        "window_policy_semantic_sha256": plan.window_policy_semantic_sha256,
        "watermark_policy_version": plan.watermark_policy_version,
        "watermark_policy_semantic_sha256": plan.watermark_policy_semantic_sha256,
        "lateness_policy_version": plan.lateness_policy_version,
        "lateness_policy_semantic_sha256": plan.lateness_policy_semantic_sha256,
        "idle_source_policy_version": plan.idle_source_policy_version,
        "idle_source_policy_semantic_sha256": plan.idle_source_policy_semantic_sha256,
        "planner_version": plan.planner_version,
        "expected_member_count": expected_count,
        "first_ordinal": 0 if ordered else None,
        "last_ordinal_or_none": expected_count - 1 if ordered else None,
        "final_append_chain_sha256": ordered[-1].append_chain_sha256 if ordered else None,
        "ordered_expected_member_root_sha256": compute_ordered_expected_member_root(ordered),
    }
    draft = ExpectedWindowPlanSeal.model_construct(
        seal_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = expected_window_plan_seal_semantic_sha256(draft)
    return ExpectedWindowPlanSeal(
        seal_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


__all__ = [
    "APPEND_CHAIN_VERSION",
    "CHILD_DELIVERY_NAMESPACE",
    "DECLARATION_PROJECTION_VERSION",
    "EXPECTED_MEMBER_ROOT_VERSION",
    "EXPECTED_WINDOW_CHILD_DELIVERY_V1_NAMESPACE",
    "EXPECTED_WINDOW_DECLARATION_SCHEMA_ID",
    "EXPECTED_WINDOW_DECLARATION_SCHEMA_VERSION",
    "EXPECTED_WINDOW_DECLARATION_WIRE_VERSION",
    "EXPECTED_WINDOW_PLAN_SCHEMA_ID",
    "EXPECTED_WINDOW_PLAN_SCHEMA_VERSION",
    "EXPECTED_WINDOW_PLAN_V1_NAMESPACE",
    "EXPECTED_WINDOW_PLAN_WIRE_VERSION",
    "EXPECTED_WINDOW_SEAL_SCHEMA_ID",
    "EXPECTED_WINDOW_SEAL_SCHEMA_VERSION",
    "EXPECTED_WINDOW_SEAL_WIRE_VERSION",
    "PLAN_IDENTITY_POLICY_VERSION",
    "PLAN_KEY_NAMESPACE",
    "PLAN_PROJECTION_VERSION",
    "STREAM_WORK_PLAN_SCHEMA_ID",
    "STREAM_WORK_PLAN_SCHEMA_VERSION",
    "STREAM_WORK_V1_NAMESPACE",
    "STREAM_WORK_WIRE_VERSION",
    "WORK_KEY_NAMESPACE",
    "WORK_KEY_POLICY_VERSION",
    "WORK_PROJECTION_VERSION",
    "ExpectedWindowDeclaration",
    "ExpectedWindowPlan",
    "ExpectedWindowPlanSeal",
    "ExpectedWindowPlanState",
    "StreamWorkDependency",
    "StreamWorkItemPlan",
    "compute_append_chain_sha256",
    "compute_ordered_expected_member_root",
    "create_expected_window_declaration",
    "create_expected_window_plan",
    "create_expected_window_plan_seal",
    "create_stream_work_item_plan",
    "derive_child_delivery_id",
    "derive_plan_id",
    "derive_plan_key",
    "derive_work_item_id",
    "derive_work_logical_key",
    "expected_window_declaration_semantic_projection",
    "expected_window_declaration_semantic_sha256",
    "expected_window_plan_seal_semantic_projection",
    "expected_window_plan_seal_semantic_sha256",
    "expected_window_plan_semantic_projection",
    "expected_window_plan_semantic_sha256",
    "stream_work_semantic_projection",
    "stream_work_semantic_sha256",
]
