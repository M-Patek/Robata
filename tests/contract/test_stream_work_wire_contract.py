from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    CameraAbsenceReason,
    StreamCameraAbsence,
    StreamPurpose,
    StreamStage,
    TerminalOutcome,
)
from robata.contracts.stream_planning import (
    StreamWorkDependency,
    create_stream_work_item_plan,
)
from robata.contracts.stream_source import (
    AuthorityBinding,
    ChannelBinding,
    create_pre_eos_capture_subject,
)
from robata.contracts.stream_window import create_incremental_window
from robata.queue.stage import DependencyCriticality
from robata.queue.stream_models import (
    StreamTerminalEvidence,
    StreamWorkAttempt,
    StreamWorkAttemptOutcome,
    StreamWorkItem,
    StreamWorkItemState,
    StreamWorkLease,
    StreamWorkLeaseClaim,
    SupportedWorkContractPin,
    WorkerCapabilityClaim,
)
from robata.queue.stream_wire import (
    STREAM_WORK_MESSAGE_SCHEMA_ID,
    STREAM_WORK_MESSAGE_SCHEMA_VERSION,
    STREAM_WORK_PLAN_SCHEMA_ID,
    StreamWorkMessage,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _schema(schema_id: str, value: int) -> SchemaRef:
    return SchemaRef(
        schema_id=schema_id,
        version="1.0.0",
        artifact_id=_uuid(value),
        sha256=_digest(value + 100),
    )


def _capture():
    channels = tuple(
        ChannelBinding(
            camera_id=CameraId(f"cam_0{index}"),
            source_channel_id=f"source-{index}",
            source_channel_epoch=1,
            channel_binding_semantic_sha256=_digest(index),
        )
        for index in range(1, 7)
    )
    authority = AuthorityBinding(
        authority_id="authority",
        authority_epoch=1,
        policy_version="policy-v1",
        initial_binding_semantic_sha256=_digest(20),
    )
    return create_pre_eos_capture_subject(
        schema_ref=_schema("https://schemas.robata.dev/stream-capture", 1),
        capture_authority_id="capture-authority",
        capture_authority_epoch=1,
        capture_assignment_policy_version="assignment-v1",
        acquisition_id="acquisition-1",
        acquisition_epoch=1,
        channel_bindings=channels,
        mapping_authority=authority,
        clock_authority=authority,
    )


def _plan(*, criticality: DependencyCriticality = DependencyCriticality.REQUIRED):
    capture = _capture()
    absent = tuple(
        StreamCameraAbsence(camera_id=CameraId(f"cam_0{index}"), reason=CameraAbsenceReason.ABSENT)
        for index in range(1, 7)
    )
    window = create_incremental_window(
        schema_ref=_schema("https://schemas.robata.dev/stream-window", 2),
        capture_scope_digest=capture.capture_scope_digest,
        purpose=StreamPurpose.QA_COARSE,
        requested_interval=NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        effective_interval=NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        ordered_six_slot_segment_or_explicit_absence_closure=absent,
        mapping_semantic_sha256=_digest(30),
        clock_or_alignment_semantic_sha256=_digest(31),
        window_policy_version="window-policy-v1",
    )
    return create_stream_work_item_plan(
        schema_ref=_schema(STREAM_WORK_PLAN_SCHEMA_ID, 3),
        stream_run_id=_uuid(4),
        source_subject=capture.reference(),
        stage=StreamStage.QA_COARSE,
        subject=window.reference(),
        input_semantic_sha256=_digest(40),
        config_semantic_sha256=_digest(41),
        ordered_dependencies=(
            StreamWorkDependency(
                upstream_work_logical_key="stream-work-v1:" + _digest(42),
                criticality=criticality,
            ),
        ),
        created_at=_NOW.isoformat(),
    )


def _evidence(outcome: TerminalOutcome = TerminalOutcome.SUCCEEDED) -> StreamTerminalEvidence:
    return StreamTerminalEvidence(
        outcome=outcome,
        evidence_ref={
            "artifact_id": _uuid(50),
            "exact_sha256": _digest(51),
            "byte_count": 1,
            "media_type": "application/json",
            "schema_ref": _schema("https://schemas.robata.dev/terminal-evidence", 52),
        },
        terminal_policy_version="terminal-policy-v1",
        completed_at=(_NOW + timedelta(seconds=1)).isoformat(),
        reason_code=None if outcome is TerminalOutcome.SUCCEEDED else "EXPLICIT_OUTCOME",
    )


def _leased_item() -> StreamWorkItem:
    plan = _plan()
    return StreamWorkItem(
        **plan.model_dump(mode="python"),
        state=StreamWorkItemState.LEASED,
        lease_epoch=2,
        fencing_token="fence-2",
        leased_by="worker-a",
        lease_expires_at=(_NOW + timedelta(minutes=1)).isoformat(),
        attempt=1,
        updated_at=_NOW.isoformat(),
    )


def test_dependency_criticality_is_part_of_stream_work_identity() -> None:
    required = _plan(criticality=DependencyCriticality.REQUIRED)
    optional = _plan(criticality=DependencyCriticality.OPTIONAL)
    assert required.work_logical_key != optional.work_logical_key
    assert required.work_item_id != optional.work_item_id


def test_stream_plan_rejects_a_subject_from_another_capture() -> None:
    plan = _plan()
    with pytest.raises(ValidationError, match="capture_scope_digest"):
        create_stream_work_item_plan(
            schema_ref=plan.schema_ref,
            stream_run_id=plan.stream_run_id,
            source_subject=plan.source_subject,
            stage=plan.stage,
            subject=plan.subject.model_copy(update={"capture_scope_digest": _digest(999)}),
            input_semantic_sha256=plan.input_semantic_sha256,
            config_semantic_sha256=plan.config_semantic_sha256,
            ordered_dependencies=plan.ordered_dependencies,
            created_at=plan.created_at,
        )


def test_terminal_stream_work_requires_exact_evidence_for_every_outcome() -> None:
    plan = _plan()
    completed = StreamWorkItem(
        **plan.model_dump(mode="python"),
        state=StreamWorkItemState.NO_EVENTS,
        terminal_evidence=_evidence(TerminalOutcome.NO_EVENTS),
        updated_at=(_NOW + timedelta(seconds=1)).isoformat(),
    )
    assert completed.terminal_evidence is not None
    assert completed.terminal_evidence.evidence_ref.byte_count == 1

    late_input = StreamWorkItem(
        **plan.model_dump(mode="python"),
        state=StreamWorkItemState.LATE_INPUT,
        terminal_evidence=_evidence(TerminalOutcome.LATE_INPUT),
        updated_at=(_NOW + timedelta(seconds=1)).isoformat(),
    )
    assert late_input.terminal_outcome is TerminalOutcome.LATE_INPUT

    with pytest.raises(ValidationError, match="requires terminal evidence"):
        StreamWorkItem(
            **plan.model_dump(mode="python"),
            state=StreamWorkItemState.FAILED,
            updated_at=_NOW.isoformat(),
        )


def test_terminal_evidence_detail_requires_a_reason_code() -> None:
    with pytest.raises(ValidationError, match="reason detail"):
        StreamTerminalEvidence(
            outcome=TerminalOutcome.ABSTAINED,
            evidence_ref=_evidence().evidence_ref,
            terminal_policy_version="terminal-policy-v1",
            completed_at=_NOW.isoformat(),
            reason_detail="detail without a stable code",
        )


def test_active_lease_claim_and_message_preserve_criticality_and_pins() -> None:
    item = _leased_item()
    lease = StreamWorkLease(
        work_item_id=item.work_item_id,
        worker_id=item.leased_by or "",
        lease_epoch=item.lease_epoch,
        fencing_token=item.fencing_token or "",
        lease_expires_at=item.lease_expires_at or "",
    )
    claim = StreamWorkLeaseClaim(work_item=item, lease=lease)
    message = StreamWorkMessage.from_ledger(
        claim.work_item,
        plan_schema_ref=claim.work_item.schema_ref,
        schema_ref=_schema(STREAM_WORK_MESSAGE_SCHEMA_ID, 60),
    )
    assert message.ordered_dependencies[0].criticality is DependencyCriticality.REQUIRED
    assert message.capability_pin().plan_schema_ref == item.schema_ref
    assert message.capability_pin().message_schema_ref.version == STREAM_WORK_MESSAGE_SCHEMA_VERSION

    tampered = message.model_dump(mode="python")
    tampered["ordered_dependencies"][0]["criticality"] = DependencyCriticality.OPTIONAL
    with pytest.raises(ValidationError, match="work_logical_key"):
        StreamWorkMessage.model_validate(tampered, strict=True)


def test_message_requires_active_lease_and_source_binding() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="leased or running"):
        StreamWorkMessage.from_ledger(
            StreamWorkItem(
                **plan.model_dump(mode="python"),
                state=StreamWorkItemState.READY,
                updated_at=_NOW.isoformat(),
            ),
            plan_schema_ref=plan.schema_ref,
            schema_ref=_schema(STREAM_WORK_MESSAGE_SCHEMA_ID, 61),
        )

    item = _leased_item()
    with pytest.raises(ValueError, match="authoritative stream work plan pin"):
        StreamWorkMessage.from_ledger(
            item,
            plan_schema_ref=_schema(STREAM_WORK_PLAN_SCHEMA_ID, 62),
            schema_ref=_schema(STREAM_WORK_MESSAGE_SCHEMA_ID, 63),
        )


def test_worker_capability_claim_is_exact_and_canonically_ordered() -> None:
    plan_ref = _schema(STREAM_WORK_PLAN_SCHEMA_ID, 70)
    message_ref = _schema(STREAM_WORK_MESSAGE_SCHEMA_ID, 71)
    pin = SupportedWorkContractPin(
        plan_schema_ref=plan_ref,
        message_schema_ref=message_ref,
        work_projection_version="stream-work-plan-semantic-v1",
        work_key_policy_version="stream-work-key-v1",
    )
    claim = WorkerCapabilityClaim(worker_id="worker-a", supported_work_contracts=(pin,))
    assert claim.supports(
        plan_schema_ref=plan_ref,
        message_schema_ref=message_ref,
        work_projection_version="stream-work-plan-semantic-v1",
        work_key_policy_version="stream-work-key-v1",
    )
    assert not claim.supports(
        plan_schema_ref=plan_ref,
        message_schema_ref=message_ref.model_copy(update={"sha256": _digest(999)}),
        work_projection_version="stream-work-plan-semantic-v1",
        work_key_policy_version="stream-work-key-v1",
    )


def test_completed_attempts_cannot_omit_evidence() -> None:
    with pytest.raises(ValidationError, match="terminal evidence"):
        StreamWorkAttempt(
            work_item_id=_uuid(80),
            attempt_number=1,
            lease_epoch=1,
            fencing_token="fence",
            worker_id="worker-a",
            claimed_at=_NOW.isoformat(),
            completed_at=(_NOW + timedelta(seconds=1)).isoformat(),
            outcome=StreamWorkAttemptOutcome.SUCCEEDED,
        )
