"""Local conformance tests for the pre-EOS stream identity chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import NanosecondInterval
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    AuthorityBinding,
    CameraAbsenceReason,
    ChannelBinding,
    DependencyCriticality,
    SixCameraSlotClosure,
    StreamArtifactRef,
    StreamCameraAbsence,
    StreamIntervalAbsence,
    StreamPolicyBinding,
    StreamPurpose,
    StreamSegmentSequence,
    StreamStage,
    StreamSubjectType,
    TerminalOutcome,
)
from robata.contracts.stream_finalization import (
    FinalizationSubjectMapping,
    create_recording_finalization_map,
    create_window_terminal_closure,
    create_window_terminal_member,
)
from robata.contracts.stream_planning import (
    StreamWorkDependency,
    create_expected_window_declaration,
    create_expected_window_plan,
    create_expected_window_plan_seal,
    create_stream_work_item_plan,
    stream_work_semantic_sha256,
)
from robata.contracts.stream_source import (
    create_pre_eos_capture_subject,
    create_stream_segment_manifest,
)
from robata.contracts.stream_window import (
    create_incremental_window,
    create_stream_inference_attempt_identity,
    create_stream_inference_identity,
)

_ROOT = Path(__file__).resolve().parents[2]
_VECTOR = _ROOT / "conformance" / "stream_identity_chain_v1.json"
_DIGEST = "a" * 64
_INTERVAL = NanosecondInterval(start_ns=0, end_ns=2_000_000_000)


def _chain() -> dict[str, object]:
    schema_ref = SchemaRef(
        schema_id="https://schemas.robata.dev/x",
        version="1.0.0",
        artifact_id="00000000-0000-0000-0000-000000000001",
        sha256=_DIGEST,
    )
    bindings = tuple(
        ChannelBinding(
            camera_id=camera_id,
            source_channel_id=f"ch{index}",
            source_channel_epoch=1,
            channel_binding_semantic_sha256=_DIGEST,
        )
        for index, camera_id in enumerate(CAMERA_IDS, 1)
    )
    authority = AuthorityBinding(
        authority_id="authority-map-clock",
        authority_epoch=1,
        policy_version="v1",
        initial_binding_semantic_sha256=_DIGEST,
    )
    capture = create_pre_eos_capture_subject(
        schema_ref=schema_ref,
        capture_authority_id="capture-authority",
        capture_authority_epoch=1,
        capture_assignment_policy_version="assign-v1",
        acquisition_id="acq-001",
        acquisition_epoch=1,
        channel_bindings=bindings,
        mapping_authority=authority,
        clock_authority=authority,
    )
    segments = tuple(
        create_stream_segment_manifest(
            schema_ref=schema_ref,
            capture_scope_digest=capture.capture_scope_digest,
            camera_id=camera_id,
            requested_interval=_INTERVAL,
            effective_interval=_INTERVAL,
            ordered_packet_or_sequence_closure=("seq-1", "seq-2"),
            exact_content_sha256=_DIGEST,
            mapping_semantic_sha256=_DIGEST,
            clock_or_alignment_semantic_sha256=_DIGEST,
            segmentation_policy_version="segment-v1",
        )
        for camera_id in CAMERA_IDS
    )
    slots = tuple(segment.reference() for segment in segments)
    window = create_incremental_window(
        schema_ref=schema_ref,
        capture_scope_digest=capture.capture_scope_digest,
        purpose=StreamPurpose.QA_COARSE,
        requested_interval=_INTERVAL,
        effective_interval=_INTERVAL,
        ordered_six_slot_segment_or_explicit_absence_closure=slots,
        mapping_semantic_sha256=_DIGEST,
        clock_or_alignment_semantic_sha256=_DIGEST,
        window_policy_version="window-v1",
    )
    policy = StreamPolicyBinding(version="v1", semantic_sha256=_DIGEST)
    expected_plan = create_expected_window_plan(
        schema_ref=schema_ref,
        capture_scope_digest=capture.capture_scope_digest,
        segmentation_policy=policy,
        window_policy=policy,
        watermark_policy=policy,
        lateness_policy=policy,
        idle_source_policy=policy,
        planner_version="planner-v1",
    )
    declaration = create_expected_window_declaration(
        schema_ref=schema_ref,
        plan_key=expected_plan.plan_key,
        ordinal=0,
        window_key=window.window_key,
        window_semantic_sha256=window.window_semantic_sha256,
        requested_interval=window.requested_interval,
        effective_interval=window.effective_interval,
        ordered_six_slot_segment_or_explicit_absence_closure=slots,
        watermark_source_facts_sha256=_DIGEST,
        previous_append_chain_sha256=None,
    )
    seal = create_expected_window_plan_seal(
        schema_ref=schema_ref,
        plan=expected_plan,
        declarations=(declaration,),
        eos_source_receipt_semantic_sha256=_DIGEST,
        final_source_timeline_semantic_sha256=_DIGEST,
        final_duration_ns=2_000_000_000,
        ordered_six_channel_health_closure_sha256=_DIGEST,
        mapping_closure_semantic_sha256=_DIGEST,
        clock_or_alignment_closure_semantic_sha256=_DIGEST,
    )
    inference = create_stream_inference_identity(
        schema_ref=schema_ref,
        window_key=window.window_key,
        window_semantic_sha256=window.window_semantic_sha256,
        purpose=StreamPurpose.QA_COARSE,
        input_plan_semantic_sha256=_DIGEST,
    )
    inference_attempt = create_stream_inference_attempt_identity(
        schema_ref=schema_ref,
        stream_inference_logical_id=inference.stream_inference_logical_id,
        attempt_number=1,
    )
    work = create_stream_work_item_plan(
        schema_ref=schema_ref,
        stream_run_id="00000000-0000-0000-0000-000000000002",
        source_subject=capture.reference(),
        stage=StreamStage.QA_COARSE,
        subject=window.reference(),
        input_semantic_sha256=_DIGEST,
        config_semantic_sha256=_DIGEST,
        created_at="2026-01-01T00:00:00Z",
    )
    evidence = StreamArtifactRef(
        artifact_id="00000000-0000-0000-0000-000000000002",
        exact_sha256=_DIGEST,
        byte_count=1,
        media_type="application/json",
        schema_ref=schema_ref,
    )
    terminal_member = create_window_terminal_member(
        schema_ref=schema_ref,
        plan_key=expected_plan.plan_key,
        expected_ordinal=0,
        window_key=window.window_key,
        window_semantic_sha256=window.window_semantic_sha256,
        terminal_outcome=TerminalOutcome.FAILED,
        terminal_work_item_id=work.work_item_id,
        terminal_work_logical_key=work.work_logical_key,
        terminal_evidence_ref=evidence,
        terminal_policy_version="terminal-v1",
    )
    closure = create_window_terminal_closure(
        schema_ref=schema_ref,
        plan_seal=seal,
        expected_declarations=(declaration,),
        members=(terminal_member,),
    )
    finalization = create_recording_finalization_map(
        schema_ref=schema_ref,
        capture_scope_key=capture.capture_scope_key,
        capture_scope_digest=capture.capture_scope_digest,
        final_source_subject_type="RECORDING",
        final_source_subject_id="00000000-0000-0000-0000-000000000004",
        final_source_exact_sha256=_DIGEST,
        final_recording_identity=_DIGEST,
        final_duration_ns=2_000_000_000,
        final_mapping_semantic_sha256=_DIGEST,
        final_alignment_semantic_sha256=_DIGEST,
        expected_plan_seal_semantic_sha256=seal.seal_semantic_sha256,
        window_terminal_closure_semantic_sha256=closure.terminal_closure_digest,
        export_manifest_semantic_sha256=_DIGEST,
        ordered_subject_mappings=(
            FinalizationSubjectMapping(
                incremental_subject_type=StreamSubjectType.INCREMENTAL_WINDOW,
                incremental_subject_key=window.window_key,
                incremental_subject_semantic_sha256=window.window_semantic_sha256,
                final_subject_type="WINDOW",
                final_subject_key=f"window-final:{_DIGEST}",
                final_subject_semantic_sha256=_DIGEST,
            ),
        ),
    )
    return locals()


def test_full_chain_matches_checked_in_vector() -> None:
    vector = json.loads(_VECTOR.read_text(encoding="utf-8"))
    chain = _chain()
    capture = chain["capture"]
    segments = chain["segments"]
    window = chain["window"]
    inference = chain["inference"]
    inference_attempt = chain["inference_attempt"]
    work = chain["work"]
    expected_plan = chain["expected_plan"]
    declaration = chain["declaration"]
    seal = chain["seal"]
    closure = chain["closure"]
    finalization = chain["finalization"]

    assert capture.capture_scope_digest == vector["capture"]["capture_scope_digest"]
    assert capture.capture_scope_key == vector["capture"]["capture_scope_key"]
    assert capture.capture_scope_id == vector["capture"]["capture_scope_id"]
    assert [
        {
            "camera_id": segment.camera_id.value,
            "segment_semantic_sha256": segment.segment_semantic_sha256,
            "segment_key": segment.segment_key,
            "segment_id": segment.segment_id,
        }
        for segment in segments
    ] == vector["segments"]
    assert {
        "window_semantic_sha256": window.window_semantic_sha256,
        "window_key": window.window_key,
        "window_id": window.window_id,
    } == vector["window"]
    assert {
        "inference_semantic_sha256": inference.inference_semantic_sha256,
        "inference_key": inference.inference_key,
        "stream_inference_logical_id": inference.stream_inference_logical_id,
    } == vector["inference"]
    assert {
        "inference_attempt_key": inference_attempt.inference_attempt_key,
        "inference_attempt_id": inference_attempt.inference_attempt_id,
    } == vector["inference_attempt"]
    assert {
        "work_logical_key": work.work_logical_key,
        "work_item_id": work.work_item_id,
        "created_at": work.created_at,
    } == vector["work"]
    assert {
        "plan_key": expected_plan.plan_key,
        "plan_digest": expected_plan.plan_digest,
    } == vector["plan"]
    assert {
        "declaration_semantic_sha256": declaration.declaration_semantic_sha256,
        "append_chain_sha256": declaration.append_chain_sha256,
    } == vector["declaration"]
    assert {"seal_semantic_sha256": seal.seal_semantic_sha256} == vector["seal"]
    assert {
        "terminal_member_semantic_sha256": closure.members[0].member_semantic_sha256,
        "terminal_member_root": closure.terminal_member_root,
        "terminal_closure_digest": closure.terminal_closure_digest,
    } == vector["closure"]
    assert {
        "finalization_key": finalization.finalization_key,
        "finalization_semantic_sha256": finalization.finalization_semantic_sha256,
    } == vector["finalization"]


def test_created_at_is_provenance_but_dependency_criticality_is_identity() -> None:
    chain = _chain()
    base = chain["work"]
    changed_time = create_stream_work_item_plan(
        schema_ref=base.schema_ref,
        stream_run_id=base.stream_run_id,
        source_subject=base.source_subject,
        stage=base.stage,
        subject=base.subject,
        input_semantic_sha256=base.input_semantic_sha256,
        config_semantic_sha256=base.config_semantic_sha256,
        created_at="2030-01-01T00:00:00Z",
    )
    assert changed_time.work_logical_key == base.work_logical_key
    assert stream_work_semantic_sha256(changed_time) == stream_work_semantic_sha256(base)

    required = base.model_copy(
        update={
            "ordered_dependencies": (
                StreamWorkDependency(
                    upstream_work_logical_key=f"stream-work-v1:{'b' * 64}",
                    criticality=DependencyCriticality.REQUIRED,
                ),
            )
        }
    )
    optional = required.model_copy(
        update={
            "ordered_dependencies": (
                StreamWorkDependency(
                    upstream_work_logical_key=f"stream-work-v1:{'b' * 64}",
                    criticality=DependencyCriticality.OPTIONAL,
                ),
            )
        }
    )
    # model_copy intentionally does not re-run validators; the projection still
    # demonstrates that changing criticality changes the identity preimage.
    assert stream_work_semantic_sha256(required) != stream_work_semantic_sha256(optional)


def test_inference_retries_keep_logical_identity_and_change_attempt_identity() -> None:
    chain = _chain()
    inference = chain["inference"]
    first_attempt = chain["inference_attempt"]
    retry = create_stream_inference_attempt_identity(
        schema_ref=inference.schema_ref,
        stream_inference_logical_id=inference.stream_inference_logical_id,
        attempt_number=2,
    )

    assert first_attempt.stream_inference_logical_id == inference.stream_inference_logical_id
    assert retry.stream_inference_logical_id == inference.stream_inference_logical_id
    assert retry.inference_attempt_key != first_attempt.inference_attempt_key
    assert retry.inference_attempt_id != first_attempt.inference_attempt_id


def test_capture_identity_binds_authority_channel_facts() -> None:
    chain = _chain()
    capture = chain["capture"]
    changed_bindings = list(capture.channel_bindings)
    changed_bindings[0] = changed_bindings[0].model_copy(
        update={"channel_binding_semantic_sha256": "b" * 64}
    )
    changed = create_pre_eos_capture_subject(
        schema_ref=capture.schema_ref,
        capture_authority_id=capture.capture_authority_id,
        capture_authority_epoch=capture.capture_authority_epoch,
        capture_assignment_policy_version=capture.capture_assignment_policy_version,
        acquisition_id=capture.acquisition_id,
        acquisition_epoch=capture.acquisition_epoch,
        channel_bindings=tuple(changed_bindings),
        mapping_authority=capture.mapping_authority,
        clock_authority=capture.clock_authority,
    )

    assert changed.capture_scope_digest != capture.capture_scope_digest
    tampered = capture.model_dump(mode="python")
    tampered["channel_bindings"][0]["channel_binding_semantic_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="capture_scope_digest"):
        type(capture).model_validate(tampered)


def test_six_slot_closure_keeps_absence_explicit_and_ordered() -> None:
    chain = _chain()
    segments = chain["segments"]
    slots = [segment.reference() for segment in segments]
    slots[2] = StreamCameraAbsence(
        camera_id=CAMERA_IDS[2],
        reason=CameraAbsenceReason.BLACK,
        evidence_sha256=_DIGEST,
    )
    closure = SixCameraSlotClosure(slots=tuple(slots))
    assert closure.slots[2].kind == "ABSENCE"
    degraded_slots = list(slots)
    degraded_slots[2] = StreamCameraAbsence(
        camera_id=CAMERA_IDS[2],
        reason=CameraAbsenceReason.DEGRADED,
        evidence_sha256=_DIGEST,
    )
    base_window = chain["window"]
    degraded_window = create_incremental_window(
        schema_ref=base_window.schema_ref,
        capture_scope_digest=base_window.capture_scope_digest,
        purpose=base_window.purpose,
        requested_interval=base_window.requested_interval,
        effective_interval=base_window.effective_interval,
        ordered_six_slot_segment_or_explicit_absence_closure=tuple(degraded_slots),
        mapping_semantic_sha256=base_window.mapping_semantic_sha256,
        clock_or_alignment_semantic_sha256=base_window.clock_or_alignment_semantic_sha256,
        window_policy_version=base_window.window_policy_version,
    )
    absent_window = create_incremental_window(
        schema_ref=base_window.schema_ref,
        capture_scope_digest=base_window.capture_scope_digest,
        purpose=base_window.purpose,
        requested_interval=base_window.requested_interval,
        effective_interval=base_window.effective_interval,
        ordered_six_slot_segment_or_explicit_absence_closure=tuple(slots),
        mapping_semantic_sha256=base_window.mapping_semantic_sha256,
        clock_or_alignment_semantic_sha256=base_window.clock_or_alignment_semantic_sha256,
        window_policy_version=base_window.window_policy_version,
    )
    assert degraded_window.window_semantic_sha256 != absent_window.window_semantic_sha256
    cross_capture_slots = list(chain["slots"])
    cross_capture_slots[0] = cross_capture_slots[0].model_copy(
        update={"capture_scope_digest": "b" * 64}
    )
    with pytest.raises(ValueError, match="capture_scope_digest"):
        create_incremental_window(
            schema_ref=base_window.schema_ref,
            capture_scope_digest=base_window.capture_scope_digest,
            purpose=base_window.purpose,
            requested_interval=base_window.requested_interval,
            effective_interval=base_window.effective_interval,
            ordered_six_slot_segment_or_explicit_absence_closure=tuple(cross_capture_slots),
            mapping_semantic_sha256=base_window.mapping_semantic_sha256,
            clock_or_alignment_semantic_sha256=base_window.clock_or_alignment_semantic_sha256,
            window_policy_version=base_window.window_policy_version,
        )
    with pytest.raises(ValueError, match="ordered"):
        SixCameraSlotClosure(slots=tuple(reversed(slots)))


def test_window_slot_can_carry_an_ordered_nonempty_segment_sequence() -> None:
    chain = _chain()
    segments = chain["segments"]
    assert isinstance(segments, tuple)
    first = segments[0]
    second = create_stream_segment_manifest(
        schema_ref=first.schema_ref,
        capture_scope_digest=first.capture_scope_digest,
        camera_id=first.camera_id,
        requested_interval=NanosecondInterval(start_ns=2_000_000_000, end_ns=3_000_000_000),
        effective_interval=NanosecondInterval(start_ns=2_000_000_000, end_ns=3_000_000_000),
        ordered_packet_or_sequence_closure=("seq-3", "seq-4"),
        exact_content_sha256="b" * 64,
        mapping_semantic_sha256=_DIGEST,
        clock_or_alignment_semantic_sha256=_DIGEST,
        segmentation_policy_version="segment-v1",
    )
    sequence = StreamSegmentSequence(
        camera_id=first.camera_id,
        capture_scope_digest=first.capture_scope_digest,
        ordered_members=(first.reference(), second.reference()),
    )
    slots = tuple(segment.reference() for segment in segments)
    window = create_incremental_window(
        schema_ref=first.schema_ref,
        capture_scope_digest=first.capture_scope_digest,
        purpose=StreamPurpose.QA_COARSE,
        requested_interval=NanosecondInterval(start_ns=0, end_ns=3_000_000_000),
        effective_interval=NanosecondInterval(start_ns=0, end_ns=3_000_000_000),
        ordered_six_slot_segment_or_explicit_absence_closure=(sequence, *slots[1:]),
        mapping_semantic_sha256=_DIGEST,
        clock_or_alignment_semantic_sha256=_DIGEST,
        window_policy_version="window-v1",
    )
    assert window.camera_closure[0] == sequence
    with pytest.raises(ValueError, match="at least one"):
        StreamSegmentSequence(
            camera_id=first.camera_id,
            capture_scope_digest=first.capture_scope_digest,
            ordered_members=(),
        )
    with pytest.raises(ValueError, match="camera and capture"):
        StreamSegmentSequence(
            camera_id=first.camera_id,
            capture_scope_digest=first.capture_scope_digest,
            ordered_members=(first.reference(), segments[1].reference()),
        )


def test_window_sequence_preserves_evidenced_partial_interval_absence() -> None:
    chain = _chain()
    segments = chain["segments"]
    assert isinstance(segments, tuple)
    first = segments[0]
    gap = StreamIntervalAbsence(
        camera_id=first.camera_id,
        capture_scope_digest=first.capture_scope_digest,
        interval=NanosecondInterval(start_ns=1_000_000_000, end_ns=2_000_000_000),
        reason=CameraAbsenceReason.GAP,
        evidence_sha256="b" * 64,
    )
    sequence = StreamSegmentSequence(
        camera_id=first.camera_id,
        capture_scope_digest=first.capture_scope_digest,
        ordered_members=(first.reference(), gap),
    )
    slots = tuple(segment.reference() for segment in segments)
    window = create_incremental_window(
        schema_ref=first.schema_ref,
        capture_scope_digest=first.capture_scope_digest,
        purpose=StreamPurpose.QA_COARSE,
        requested_interval=NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        effective_interval=NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        ordered_six_slot_segment_or_explicit_absence_closure=(sequence, *slots[1:]),
        mapping_semantic_sha256=_DIGEST,
        clock_or_alignment_semantic_sha256=_DIGEST,
        window_policy_version="window-v1",
    )
    assert window.camera_closure[0] == sequence
    assert sequence.ordered_members[1] == gap

    with pytest.raises(ValueError, match="must be unique"):
        StreamSegmentSequence(
            camera_id=first.camera_id,
            capture_scope_digest=first.capture_scope_digest,
            ordered_members=(gap, gap),
        )
    with pytest.raises(ValueError, match="camera and capture"):
        StreamSegmentSequence(
            camera_id=first.camera_id,
            capture_scope_digest=first.capture_scope_digest,
            ordered_members=(gap.model_copy(update={"capture_scope_digest": "c" * 64}),),
        )


def test_terminal_closure_cannot_omit_a_sealed_member() -> None:
    chain = _chain()
    with pytest.raises(ValueError, match="every sealed expected member"):
        create_window_terminal_closure(
            schema_ref=chain["schema_ref"],
            plan_seal=chain["seal"],
            expected_declarations=(chain["declaration"],),
            members=(),
        )
