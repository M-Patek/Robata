from __future__ import annotations

from uuid import UUID

import pytest

from robata.application.canonical.stream_recording_reduction import (
    LocalStreamCanonicalTruth,
    LocalStreamQaCameraReference,
    LocalStreamRecordingReductionError,
    LocalStreamRecordingResultV4,
    LocalStreamSemanticIntervalReference,
    LocalStreamWindowSemanticEvidence,
    LocalStreamWindowSemanticEvidenceV2,
    create_local_stream_recording_result,
    create_local_stream_recording_result_v2,
    create_local_stream_recording_result_v3,
    create_local_stream_recording_result_v4,
    create_local_stream_window_semantic_evidence,
    validate_local_stream_recording_result_v3_truth,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    StreamPurpose,
    StreamSubjectRef,
    StreamSubjectType,
    TerminalOutcome,
)
from robata.contracts.stream_finalization import (
    FinalizationSubjectMapping,
    RecordingFinalizationMap,
    WindowTerminalClosure,
    compute_terminal_member_root,
    create_recording_finalization_map,
    create_window_terminal_member,
    terminal_closure_semantic_sha256,
)
from robata.contracts.stream_inference import (
    StreamWindowResult,
    create_stream_window_result,
)
from robata.contracts.stream_planning import derive_work_item_id


def _digest(value: int) -> str:
    return f"{value:064x}"


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _schema(value: int = 1) -> SchemaRef:
    return SchemaRef(
        schema_id=f"https://schemas.robata.dev/stream-recording-reduction-test-{value}",
        version="1.0.0",
        artifact_id=_uuid(100 + value),
        sha256=_digest(200 + value),
    )


def _recording_result_schema() -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/local-stream-recording-result",
        version="1.0.0",
        artifact_id=_uuid(199),
        sha256=_digest(299),
    )


def _recording_result_v2_schema() -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/local-stream-recording-result",
        version="2.0.0",
        artifact_id=_uuid(198),
        sha256=_digest(298),
    )


def _window_semantic_evidence_schema() -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/local-stream-window-semantic-evidence",
        version="1.0.0",
        artifact_id=_uuid(197),
        sha256=_digest(297),
    )


def _evidence(value: int) -> ArtifactEvidenceRef:
    return ArtifactEvidenceRef(
        artifact_id=_uuid(300 + value),
        exact_sha256=_digest(400 + value),
        byte_count=value + 1,
        media_type="application/json",
        schema_ref=_schema(2),
    )


def _window_result(
    ordinal: int,
    *,
    capture_digest: str,
    outcome: TerminalOutcome,
) -> StreamWindowResult:
    window_digest = _digest(500 + ordinal)
    return create_stream_window_result(
        schema_ref=_schema(3),
        window_subject=StreamSubjectRef(
            subject_type=StreamSubjectType.INCREMENTAL_WINDOW,
            subject_key=f"incremental-window-v1:{window_digest}",
            subject_semantic_sha256=window_digest,
            capture_scope_digest=capture_digest,
            identity_policy_version="incremental-window-identity-v1",
            schema_ref=_schema(4),
        ),
        purpose=StreamPurpose.QA_COARSE,
        terminal_outcome=outcome,
        accepted_terminals=(),
        result_semantic_evidence_sha256=_digest(600 + ordinal),
        result_evidence_ref=_evidence(ordinal),
        reduction_policy_version="test-window-reduction-v1",
        created_at="2026-01-01T00:00:00Z",
    )


def _closure(
    results: tuple[StreamWindowResult, ...],
    *,
    plan_key: str,
    seal_digest: str,
) -> WindowTerminalClosure:
    result_payloads = tuple(canonical_json_bytes(result) for result in results)
    members = tuple(
        create_window_terminal_member(
            schema_ref=_schema(5),
            plan_key=plan_key,
            expected_ordinal=ordinal,
            window_key=result.window_subject.subject_key,
            window_semantic_sha256=result.window_subject.subject_semantic_sha256,
            terminal_outcome=result.terminal_outcome,
            terminal_work_item_id=derive_work_item_id(_digest(700 + ordinal)),
            terminal_work_logical_key=f"stream-work-v1:{_digest(700 + ordinal)}",
            terminal_evidence_ref=ArtifactEvidenceRef(
                artifact_id=_uuid(650 + ordinal),
                exact_sha256=exact_bytes_sha256(result_payloads[ordinal]),
                byte_count=len(result_payloads[ordinal]),
                media_type="application/json",
                schema_ref=result.schema_ref,
            ),
            terminal_policy_version="test-terminal-v1",
        )
        for ordinal, result in enumerate(results)
    )
    root = compute_terminal_member_root(
        plan_seal_semantic_sha256=seal_digest,
        members=members,
    )
    draft = WindowTerminalClosure.model_construct(
        schema_ref=_schema(6),
        plan_key=plan_key,
        plan_seal_semantic_sha256=seal_digest,
        expected_member_count=len(members),
        members=members,
        terminal_member_root=root,
        terminal_closure_digest="0" * 64,
    )
    return WindowTerminalClosure(
        schema_ref=draft.schema_ref,
        plan_key=plan_key,
        plan_seal_semantic_sha256=seal_digest,
        expected_member_count=len(members),
        members=members,
        terminal_member_root=root,
        terminal_closure_digest=terminal_closure_semantic_sha256(draft),
    )


def _inputs(
    outcomes: tuple[TerminalOutcome, ...] = (
        TerminalOutcome.ABSTAINED,
        TerminalOutcome.ABSTAINED,
    ),
    *,
    capture_digest: str | None = None,
) -> tuple[
    tuple[StreamWindowResult, ...],
    WindowTerminalClosure,
    RecordingFinalizationMap,
]:
    capture = capture_digest or _digest(800)
    results = tuple(
        _window_result(ordinal, capture_digest=capture, outcome=outcome)
        for ordinal, outcome in enumerate(outcomes)
    )
    plan_key = f"expected-window-plan-v1:{_digest(801)}"
    seal_digest = _digest(802)
    closure = _closure(results, plan_key=plan_key, seal_digest=seal_digest)
    mappings = tuple(
        FinalizationSubjectMapping(
            incremental_subject_type=StreamSubjectType.INCREMENTAL_WINDOW,
            incremental_subject_key=result.window_subject.subject_key,
            incremental_subject_semantic_sha256=(result.window_subject.subject_semantic_sha256),
            final_subject_type="FINAL_WINDOW",
            final_subject_key=f"final-window-v1:{_digest(900 + ordinal)}",
            final_subject_semantic_sha256=_digest(900 + ordinal),
        )
        for ordinal, result in enumerate(results)
    )
    finalization = create_recording_finalization_map(
        schema_ref=_schema(7),
        capture_scope_key=f"pre-eos-capture-v1:{capture}",
        capture_scope_digest=capture,
        final_source_subject_type="MCAP_RECORDING",
        final_source_subject_id=_uuid(950),
        final_source_exact_sha256=_digest(951),
        final_recording_identity=_digest(952),
        final_duration_ns=2_000_000_000,
        final_mapping_semantic_sha256=_digest(953),
        final_alignment_semantic_sha256=_digest(954),
        expected_plan_seal_semantic_sha256=seal_digest,
        window_terminal_closure_semantic_sha256=closure.terminal_closure_digest,
        export_manifest_semantic_sha256=_digest(955),
        ordered_subject_mappings=mappings,
    )
    return results, closure, finalization


def _canonical_truth() -> LocalStreamCanonicalTruth:
    shared_interval = NanosecondInterval(
        start_ns=500_000_000,
        end_ns=1_500_000_000,
    )
    return LocalStreamCanonicalTruth(
        six_camera_qa_semantic_sha256=_digest(1001),
        qa_camera_references=tuple(
            LocalStreamQaCameraReference(
                camera_id=camera_id,
                semantic_sha256=_digest(1010 + ordinal),
            )
            for ordinal, camera_id in enumerate(CAMERA_IDS)
        ),
        event_proposal_result_semantic_sha256=_digest(1020),
        proposal_references=(
            LocalStreamSemanticIntervalReference(
                kind="EVENT_PROPOSAL",
                logical_key=f"event-proposal-v1:{_digest(1021)}",
                semantic_sha256=_digest(1021),
                interval=shared_interval,
            ),
        ),
        candidate_reduction_semantic_sha256=_digest(1030),
        candidate_references=(),
        provisional_fusion_semantic_sha256=_digest(1040),
        action_references=(),
        boundary_closure_semantic_sha256=_digest(1050),
        boundary_references=(),
        output_decision="ADMITTED",
        output_decision_semantic_sha256=_digest(1060),
        hypothesis_references=(
            LocalStreamSemanticIntervalReference(
                kind="HYPOTHESIS",
                logical_key=f"event-hypothesis-v1:{_digest(1061)}",
                semantic_sha256=_digest(1061),
                interval=shared_interval,
            ),
        ),
    )


def _window_semantic_evidence(
    results: tuple[StreamWindowResult, ...],
    closure: WindowTerminalClosure,
) -> tuple[
    tuple[LocalStreamWindowSemanticEvidence, ArtifactEvidenceRef],
    ...,
]:
    truth = _canonical_truth()
    pairs: list[tuple[LocalStreamWindowSemanticEvidence, ArtifactEvidenceRef]] = []
    for ordinal, result in enumerate(results):
        evidence = create_local_stream_window_semantic_evidence(
            schema_ref=_window_semantic_evidence_schema(),
            plan_key=closure.plan_key,
            expected_ordinal=ordinal,
            window_result=result,
            effective_interval=NanosecondInterval(
                start_ns=ordinal * 1_000_000_000,
                end_ns=(ordinal + 1) * 1_000_000_000,
            ),
            canonical_truth=truth,
        )
        payload = canonical_json_bytes(evidence)
        pairs.append(
            (
                evidence,
                ArtifactEvidenceRef(
                    artifact_id=_uuid(1100 + ordinal),
                    exact_sha256=exact_bytes_sha256(payload),
                    byte_count=len(payload),
                    media_type="application/json",
                    schema_ref=evidence.schema_ref,
                ),
            )
        )
    return tuple(pairs)


def test_reduction_is_invariant_to_supplied_window_result_order() -> None:
    results, closure, finalization = _inputs()

    ordered = create_local_stream_recording_result(
        schema_ref=_recording_result_schema(),
        window_results=results,
        terminal_closure=closure,
        recording_finalization=finalization,
    )
    reversed_input = create_local_stream_recording_result(
        schema_ref=_recording_result_schema(),
        window_results=tuple(reversed(results)),
        terminal_closure=closure,
        recording_finalization=finalization,
    )

    assert reversed_input == ordered
    assert ordered.ordered_window_result_semantic_sha256_values == tuple(
        result.window_result_semantic_sha256 for result in results
    )


def test_reduction_rejects_a_missing_sealed_window() -> None:
    results, closure, finalization = _inputs()

    with pytest.raises(LocalStreamRecordingReductionError, match="every sealed"):
        create_local_stream_recording_result(
            schema_ref=_recording_result_schema(),
            window_results=results[:1],
            terminal_closure=closure,
            recording_finalization=finalization,
        )


def test_reduction_rejects_a_window_from_another_capture() -> None:
    results, closure, finalization = _inputs()
    foreign = _window_result(
        1,
        capture_digest=_digest(999),
        outcome=TerminalOutcome.ABSTAINED,
    )

    with pytest.raises(LocalStreamRecordingReductionError, match="terminal closure"):
        create_local_stream_recording_result(
            schema_ref=_recording_result_schema(),
            window_results=(results[0], foreign),
            terminal_closure=closure,
            recording_finalization=finalization,
        )


def test_local_abstained_windows_reduce_to_nonproduction_abstention() -> None:
    results, closure, finalization = _inputs()

    reduced = create_local_stream_recording_result(
        schema_ref=_recording_result_schema(),
        window_results=results,
        terminal_closure=closure,
        recording_finalization=finalization,
    )

    assert reduced.output_decision == "ABSTAINED"
    assert reduced.accepted_terminals == ()
    assert reduced.evidence_class == "LOCAL_CONFORMANCE"
    assert reduced.production_eligible is False


def test_v2_reduction_is_order_invariant_and_deduplicates_cross_window_truth() -> None:
    results, closure, finalization = _inputs()
    evidence = _window_semantic_evidence(results, closure)

    ordered = create_local_stream_recording_result_v2(
        schema_ref=_recording_result_v2_schema(),
        window_results=results,
        window_semantic_evidence=evidence,
        terminal_closure=closure,
        recording_finalization=finalization,
    )
    reversed_input = create_local_stream_recording_result_v2(
        schema_ref=_recording_result_v2_schema(),
        window_results=tuple(reversed(results)),
        window_semantic_evidence=tuple(reversed(evidence)),
        terminal_closure=closure,
        recording_finalization=finalization,
    )

    assert reversed_input == ordered
    assert ordered.schema_version == "2.0"
    assert ordered.output_decision == "ADMITTED"
    assert ordered.cross_window_duplicate_reference_count == 2
    assert ordered.ordered_window_semantic_evidence_refs == tuple(
        reference for _, reference in evidence
    )
    assert ordered.proposal_references == _canonical_truth().proposal_references
    assert ordered.hypothesis_references == _canonical_truth().hypothesis_references
    assert ordered.production_eligible is False


def test_v2_reduction_rejects_tampered_window_semantic_artifact() -> None:
    results, closure, finalization = _inputs()
    evidence = list(_window_semantic_evidence(results, closure))
    item, reference = evidence[0]
    evidence[0] = (
        item,
        reference.model_copy(update={"exact_sha256": _digest(1200)}),
    )

    with pytest.raises(
        LocalStreamRecordingReductionError,
        match="content-addressed reference",
    ):
        create_local_stream_recording_result_v2(
            schema_ref=_recording_result_v2_schema(),
            window_results=results,
            window_semantic_evidence=tuple(evidence),
            terminal_closure=closure,
            recording_finalization=finalization,
        )


def test_v2_reduction_rejects_failed_required_window() -> None:
    results, closure, finalization = _inputs((TerminalOutcome.ABSTAINED, TerminalOutcome.FAILED))
    evidence = _window_semantic_evidence(results, closure)

    with pytest.raises(
        LocalStreamRecordingReductionError,
        match="failed or incomplete required window",
    ):
        create_local_stream_recording_result_v2(
            schema_ref=_recording_result_v2_schema(),
            window_results=results,
            window_semantic_evidence=evidence,
            terminal_closure=closure,
            recording_finalization=finalization,
        )


def _recording_result_v3_schema() -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/local-stream-recording-result",
        version="3.0.0",
        artifact_id=_uuid(196),
        sha256=_digest(296),
    )


def _v3_inputs(
    *, source_timeline_origin_ns: int = 0
) -> tuple[
    tuple[StreamWindowResult, ...],
    tuple[LocalStreamWindowSemanticEvidenceV2, ...],
    WindowTerminalClosure,
    RecordingFinalizationMap,
]:
    capture = _digest(1300)
    plan_key = f"expected-window-plan-v1:{_digest(1301)}"
    seal_digest = _digest(1302)
    base_results = tuple(
        _window_result(ordinal, capture_digest=capture, outcome=TerminalOutcome.ABSTAINED)
        for ordinal in range(2)
    )
    evidence: list[LocalStreamWindowSemanticEvidenceV2] = []
    for ordinal, result in enumerate(base_results):
        interval = NanosecondInterval(
            start_ns=source_timeline_origin_ns + ordinal * 1_000_000_000,
            end_ns=source_timeline_origin_ns + 2_000_000_000,
        )
        values = {
            "schema_ref": _schema(20),
            "plan_key": plan_key,
            "plan_semantic_sha256": _digest(1310),
            "window_inference_plan_ref": _evidence(40 + ordinal),
            "expected_ordinal": ordinal,
            "window_key": result.window_subject.subject_key,
            "window_semantic_sha256": result.window_subject.subject_semantic_sha256,
            "effective_interval": interval,
            "input_plan_semantic_sha256": _digest(1320),
            "six_camera_slot_closure_semantic_sha256": _digest(1330),
            "semantic_status": "PROPOSED",
            "proposal_label": "fixture-action",
            "proposal_interval": interval,
            "proposal_semantic_sha256": _digest(1340 + ordinal),
        }
        draft = LocalStreamWindowSemanticEvidenceV2.model_construct(
            semantic_sha256="0" * 64,
            **values,
        )
        evidence.append(
            LocalStreamWindowSemanticEvidenceV2(
                semantic_sha256=semantic_sha256(
                    draft.model_dump(mode="json", exclude={"schema_ref", "semantic_sha256"})
                ),
                **values,
            )
        )
    results: list[StreamWindowResult] = []
    for base, s2 in zip(base_results, evidence, strict=True):
        payload = canonical_json_bytes(s2)
        results.append(
            create_stream_window_result(
                schema_ref=base.schema_ref,
                window_subject=base.window_subject,
                purpose=base.purpose,
                terminal_outcome=base.terminal_outcome,
                accepted_terminals=base.accepted_terminals,
                result_semantic_evidence_sha256=s2.semantic_sha256,
                result_evidence_ref=ArtifactEvidenceRef(
                    artifact_id=_uuid(1400 + s2.expected_ordinal),
                    exact_sha256=exact_bytes_sha256(payload),
                    byte_count=len(payload),
                    media_type="application/json",
                    schema_ref=s2.schema_ref,
                ),
                reduction_policy_version=base.reduction_policy_version,
                created_at=base.created_at,
            )
        )
    ordered_results = tuple(results)
    closure = _closure(ordered_results, plan_key=plan_key, seal_digest=seal_digest)
    mappings = tuple(
        FinalizationSubjectMapping(
            incremental_subject_type=StreamSubjectType.INCREMENTAL_WINDOW,
            incremental_subject_key=result.window_subject.subject_key,
            incremental_subject_semantic_sha256=result.window_subject.subject_semantic_sha256,
            final_subject_type="FINAL_WINDOW",
            final_subject_key=f"final-window-v1:{_digest(1410 + ordinal)}",
            final_subject_semantic_sha256=_digest(1410 + ordinal),
        )
        for ordinal, result in enumerate(ordered_results)
    )
    finalization = create_recording_finalization_map(
        schema_ref=_schema(21),
        capture_scope_key=f"pre-eos-capture-v1:{capture}",
        capture_scope_digest=capture,
        final_source_subject_type="MCAP_RECORDING",
        final_source_subject_id=_uuid(1420),
        final_source_exact_sha256=_digest(1421),
        final_recording_identity=_digest(1422),
        final_duration_ns=2_000_000_000,
        final_mapping_semantic_sha256=_digest(1423),
        final_alignment_semantic_sha256=_digest(1424),
        expected_plan_seal_semantic_sha256=seal_digest,
        window_terminal_closure_semantic_sha256=closure.terminal_closure_digest,
        export_manifest_semantic_sha256=_digest(1425),
        ordered_subject_mappings=mappings,
    )
    return ordered_results, tuple(evidence), closure, finalization


def test_v3_causal_reduction_binds_w1_to_s2_and_merges_adjacent_fragments() -> None:
    results, evidence, closure, finalization = _v3_inputs()

    reduced = create_local_stream_recording_result_v3(
        schema_ref=_recording_result_v3_schema(),
        window_results=results,
        window_semantic_evidence=evidence,
        terminal_closure=closure,
        recording_finalization=finalization,
    )

    assert reduced.schema_version == "3.0"
    assert reduced.production_eligible is False
    assert reduced.output_decision == "ADMITTED"
    assert len(reduced.merged_hypotheses) == 1
    assert reduced.merged_hypotheses[0].label == "fixture-action"
    assert reduced.merged_hypotheses[0].interval == NanosecondInterval(
        start_ns=0,
        end_ns=2_000_000_000,
    )
    assert reduced.merged_hypotheses[0].source_ordinals == (0, 1)

    # Canonical semantic digests are unrelated to the local causal S2 digests.
    validate_local_stream_recording_result_v3_truth(reduced, _canonical_truth())


def test_v3_rejects_out_of_order_or_unbound_s2() -> None:
    results, evidence, closure, finalization = _v3_inputs()

    with pytest.raises(LocalStreamRecordingReductionError, match="out of order"):
        create_local_stream_recording_result_v3(
            schema_ref=_recording_result_v3_schema(),
            window_results=results,
            window_semantic_evidence=tuple(reversed(evidence)),
            terminal_closure=closure,
            recording_finalization=finalization,
        )

    unbound = results[0].model_copy(update={"result_semantic_evidence_sha256": _digest(1500)})
    with pytest.raises(LocalStreamRecordingReductionError, match="exactly bind S2"):
        create_local_stream_recording_result_v3(
            schema_ref=_recording_result_v3_schema(),
            window_results=(unbound, results[1]),
            window_semantic_evidence=evidence,
            terminal_closure=closure,
            recording_finalization=finalization,
        )


def _recording_result_v4_schema() -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/local-stream-recording-result",
        version="4.0.0",
        artifact_id=_uuid(195),
        sha256=_digest(295),
    )


def test_v4_encodes_real_epoch_origin_as_canonical_decimal_string() -> None:
    source_origin_ns = 1_742_000_000_000_000_000
    results, evidence, closure, finalization = _v3_inputs(
        source_timeline_origin_ns=source_origin_ns
    )
    reduced = create_local_stream_recording_result_v4(
        schema_ref=_recording_result_v4_schema(),
        window_results=results,
        window_semantic_evidence=evidence,
        terminal_closure=closure,
        recording_finalization=finalization,
        source_timeline_origin_ns=source_origin_ns,
        canonical_requested_interval=NanosecondInterval(
            start_ns=0,
            end_ns=2_000_000_000,
        ),
    )

    payload = canonical_json_bytes(reduced)
    assert b'"source_timeline_origin_ns":"1742000000000000000"' in payload
    assert reduced.source_timeline_origin_ns == source_origin_ns
    assert reduced.merged_hypotheses[0].interval == NanosecondInterval(
        start_ns=0,
        end_ns=2_000_000_000,
    )
    assert LocalStreamRecordingResultV4.model_validate_json(payload) == reduced
