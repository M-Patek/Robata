from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.admission import (
    AdmissionLedgerError,
    AlignmentAdmissionOutcome,
    InMemoryAdmissionLedger,
    PrimaryAdmissionPolicy,
    SourceAdmissionOutcome,
    create_alignment_admission_decision,
    create_source_admission_decision,
    create_source_alias_observation,
)

NOW = "2026-07-19T16:00:00Z"
LATER = "2026-07-19T16:01:00Z"


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _pending(
    *,
    recording_identity: str = _digest(1),
    content: str = _digest(2),
):
    return create_source_admission_decision(
        recording_identity=recording_identity,
        source_content_sha256=content,
        sequence=0,
        outcome=SourceAdmissionOutcome.PENDING,
        policy_version="source-policy-1",
        policy_sha256=_digest(3),
        decided_at=NOW,
    )


def _ready(pending, *, report_id: str = _uuid(10), manifest_id: str = _uuid(11)):
    return create_source_admission_decision(
        recording_identity=pending.recording_identity,
        source_content_sha256=pending.source_content_sha256,
        sequence=1,
        outcome=SourceAdmissionOutcome.READY,
        validation_report_id=report_id,
        validation_report_semantic_sha256=_digest(12),
        ready_manifest_id=manifest_id,
        ready_manifest_semantic_sha256=_digest(13),
        policy_version="source-policy-1",
        policy_sha256=_digest(3),
        predecessor_logical_key=pending.decision_logical_key,
        decided_at=LATER,
    )


def _alignment(
    ready,
    *,
    sequence: int,
    outcome: AlignmentAdmissionOutcome,
    predecessor: str | None,
    alignment: bool = False,
    evidence: bool = False,
    reason: str | None = None,
):
    return create_alignment_admission_decision(
        ready_manifest_id=ready.ready_manifest_id,
        ready_manifest_semantic_sha256=ready.ready_manifest_semantic_sha256,
        sequence=sequence,
        outcome=outcome,
        policy_version="alignment-policy-1",
        policy_sha256=_digest(20),
        predecessor_logical_key=predecessor,
        alignment_id=_uuid(21) if alignment else None,
        alignment_semantic_sha256=_digest(22) if alignment else None,
        validation_evidence_sha256=_digest(23) if evidence else None,
        reason_code=reason,
        decided_at=LATER,
    )


def test_source_aliases_do_not_inflate_unique_content_denominator() -> None:
    ledger = InMemoryAdmissionLedger()
    pending = _pending()
    first = create_source_alias_observation(
        source_uri="file:///first.mcap",
        source_version="v1",
        source_content_sha256=pending.source_content_sha256,
        recording_identity=pending.recording_identity,
        observed_at=NOW,
    )
    second = create_source_alias_observation(
        source_uri="file:///moved.mcap",
        source_version="v1",
        source_content_sha256=pending.source_content_sha256,
        recording_identity=pending.recording_identity,
        observed_at=LATER,
    )

    assert ledger.observe_alias(first) == first
    assert ledger.observe_alias(first.model_copy(update={"observed_at": LATER})) == first
    ledger.observe_alias(second)
    ledger.append_source_decision(pending)

    counts = ledger.reconcile_sources()
    assert counts.discovered_unique_content == 1
    assert counts.alias_observations == 2
    assert counts.pending == 1


def test_same_alias_version_cannot_change_content_binding() -> None:
    ledger = InMemoryAdmissionLedger()
    first = create_source_alias_observation(
        source_uri="file:///source.mcap",
        source_version="v1",
        source_content_sha256=_digest(1),
        recording_identity=_digest(2),
        observed_at=NOW,
    )
    conflicting = create_source_alias_observation(
        source_uri=first.source_uri,
        source_version=first.source_version,
        source_content_sha256=_digest(3),
        recording_identity=_digest(4),
        observed_at=LATER,
    )
    ledger.observe_alias(first)

    with pytest.raises(AdmissionLedgerError, match="different content"):
        ledger.observe_alias(conflicting)


def test_source_decision_chain_is_append_only_and_stale_writes_fail() -> None:
    ledger = InMemoryAdmissionLedger()
    pending = ledger.append_source_decision(_pending())
    ready = ledger.append_source_decision(_ready(pending))

    assert ledger.append_source_decision(ready.model_copy(update={"decided_at": NOW})) == ready
    stale = create_source_admission_decision(
        recording_identity=pending.recording_identity,
        source_content_sha256=pending.source_content_sha256,
        sequence=2,
        outcome=SourceAdmissionOutcome.QUARANTINED,
        policy_version="source-policy-2",
        policy_sha256=_digest(30),
        predecessor_logical_key=pending.decision_logical_key,
        reason_code="SECURITY_HOLD",
        decided_at=LATER,
    )
    with pytest.raises(AdmissionLedgerError, match="predecessor is stale"):
        ledger.append_source_decision(stale)

    assert ledger.current_source(pending.recording_identity) == ready
    assert ledger.source_history(pending.recording_identity) == (pending, ready)


def test_invalid_or_inconclusive_source_never_enters_alignment_cohort() -> None:
    ledger = InMemoryAdmissionLedger()
    pending = ledger.append_source_decision(_pending())
    invalid = create_source_admission_decision(
        recording_identity=pending.recording_identity,
        source_content_sha256=pending.source_content_sha256,
        sequence=1,
        outcome=SourceAdmissionOutcome.INVALID,
        validation_report_id=_uuid(31),
        validation_report_semantic_sha256=_digest(32),
        policy_version="source-policy-1",
        policy_sha256=_digest(3),
        predecessor_logical_key=pending.decision_logical_key,
        reason_code="INVALID_CAMERA_COUNT",
        decided_at=LATER,
    )
    ledger.append_source_decision(invalid)
    fake_ready = _ready(pending)
    not_scheduled = _alignment(
        fake_ready,
        sequence=0,
        outcome=AlignmentAdmissionOutcome.NOT_SCHEDULED,
        predecessor=None,
    )

    with pytest.raises(AdmissionLedgerError, match="currently selected READY"):
        ledger.append_alignment_decision(not_scheduled)

    counts = ledger.reconcile_sources()
    assert counts.invalid == 1
    assert counts.ready == 0
    assert ledger.reconcile_alignments().ready_manifest_cohort == 0


def test_alignment_accounting_and_primary_admission_are_independent() -> None:
    ledger = InMemoryAdmissionLedger()
    pending = ledger.append_source_decision(_pending())
    ready = ledger.append_source_decision(_ready(pending))
    not_scheduled = ledger.append_alignment_decision(
        _alignment(
            ready,
            sequence=0,
            outcome=AlignmentAdmissionOutcome.NOT_SCHEDULED,
            predecessor=None,
        )
    )
    queued = ledger.append_alignment_decision(
        _alignment(
            ready,
            sequence=1,
            outcome=AlignmentAdmissionOutcome.QUEUED,
            predecessor=not_scheduled.decision_logical_key,
        )
    )
    running = ledger.append_alignment_decision(
        _alignment(
            ready,
            sequence=2,
            outcome=AlignmentAdmissionOutcome.RUNNING,
            predecessor=queued.decision_logical_key,
            alignment=True,
        )
    )
    degraded = ledger.append_alignment_decision(
        _alignment(
            ready,
            sequence=3,
            outcome=AlignmentAdmissionOutcome.DEGRADED,
            predecessor=running.decision_logical_key,
            alignment=True,
            evidence=True,
        )
    )

    valid_only = PrimaryAdmissionPolicy.create(
        version="primary-valid-only-1",
        admissible_alignment_outcomes=(AlignmentAdmissionOutcome.VALID,),
    )
    allow_degraded = PrimaryAdmissionPolicy.create(
        version="primary-degraded-1",
        admissible_alignment_outcomes=(
            AlignmentAdmissionOutcome.VALID,
            AlignmentAdmissionOutcome.DEGRADED,
        ),
    )

    rejected = ledger.evaluate_primary(
        recording_identity=pending.recording_identity,
        policy=valid_only,
    )
    accepted = ledger.evaluate_primary(
        recording_identity=pending.recording_identity,
        policy=allow_degraded,
    )
    assert rejected.admissible is False
    assert rejected.reason_code == "ALIGNMENT_NOT_ADMISSIBLE"
    assert accepted.admissible is True
    assert accepted.ready_manifest_id == ready.ready_manifest_id
    assert accepted.ready_manifest_semantic_sha256 == ready.ready_manifest_semantic_sha256
    assert accepted.alignment_outcome is AlignmentAdmissionOutcome.DEGRADED
    assert accepted.alignment_id == degraded.alignment_id
    assert accepted.alignment_semantic_sha256 == degraded.alignment_semantic_sha256
    assert ledger.current_alignment(ready.ready_manifest_id) == degraded

    source_counts = ledger.reconcile_sources()
    alignment_counts = ledger.reconcile_alignments()
    assert source_counts.ready == 1
    assert alignment_counts.ready_manifest_cohort == 1
    assert alignment_counts.degraded == 1


def test_projection_rebuild_preserves_verified_chain_tails() -> None:
    ledger = InMemoryAdmissionLedger()
    pending = ledger.append_source_decision(_pending())
    ready = ledger.append_source_decision(_ready(pending))
    not_scheduled = ledger.append_alignment_decision(
        _alignment(
            ready,
            sequence=0,
            outcome=AlignmentAdmissionOutcome.NOT_SCHEDULED,
            predecessor=None,
        )
    )
    ledger.append_alignment_decision(
        _alignment(
            ready,
            sequence=1,
            outcome=AlignmentAdmissionOutcome.CANCELLED,
            predecessor=not_scheduled.decision_logical_key,
            reason="OPERATOR_CANCELLED",
        )
    )
    before_source = ledger.current_source(pending.recording_identity)
    before_alignment = ledger.current_alignment(ready.ready_manifest_id)

    ledger.rebuild_current_projections()

    assert ledger.current_source(pending.recording_identity) == before_source
    assert ledger.current_alignment(ready.ready_manifest_id) == before_alignment


def test_semantic_retry_excludes_row_ids_and_clock_but_keeps_first_refs() -> None:
    pending = _pending()
    first = _ready(pending, report_id=_uuid(40), manifest_id=_uuid(41))
    replay = _ready(pending, report_id=_uuid(42), manifest_id=_uuid(43)).model_copy(
        update={"decided_at": NOW}
    )
    assert replay.semantic_sha256 == first.semantic_sha256
    assert replay.decision_logical_key == first.decision_logical_key

    ledger = InMemoryAdmissionLedger()
    ledger.append_source_decision(pending)
    assert ledger.append_source_decision(first) == first
    assert ledger.append_source_decision(replay) == first
    assert ledger.current_source(pending.recording_identity).ready_manifest_id == _uuid(41)


def test_decision_shapes_fail_closed_before_ledger_append() -> None:
    pending = _pending()
    with pytest.raises(ValidationError, match="READY requires"):
        create_source_admission_decision(
            recording_identity=pending.recording_identity,
            source_content_sha256=pending.source_content_sha256,
            sequence=1,
            outcome=SourceAdmissionOutcome.READY,
            validation_report_id=_uuid(50),
            validation_report_semantic_sha256=_digest(51),
            policy_version="source-policy-1",
            policy_sha256=_digest(3),
            predecessor_logical_key=pending.decision_logical_key,
            decided_at=LATER,
        )

    ready = _ready(pending)
    with pytest.raises(ValidationError, match="terminal alignment verdict"):
        _alignment(
            ready,
            sequence=0,
            outcome=AlignmentAdmissionOutcome.VALID,
            predecessor=None,
            alignment=True,
            evidence=False,
        )
