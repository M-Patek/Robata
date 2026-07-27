from __future__ import annotations

from pathlib import Path

import pytest

from robata.application.canonical.local_composition import run_local_canonical_fixture
from robata.application.canonical.qualification_evidence import (
    CanonicalRecoveryEvidenceClass,
    CanonicalRecoveryQualificationEvidence,
    CanonicalRecoveryScenario,
    build_canonical_recovery_qualification_evidence,
)
from robata.contracts.hashing import exact_bytes_sha256

SOURCE_FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical" / "source-recording.json"


def test_local_recovery_evidence_binds_fresh_replay_and_durable_identities(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "canonical-state"
    fresh = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="recovery-evidence",
    )
    replay = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="recovery-evidence",
    )
    assert fresh.review_routing.review_task_id is not None

    evidence = build_canonical_recovery_qualification_evidence(
        workload_fingerprint=exact_bytes_sha256(b"local-recovery-workload"),
        run_namespace="local-recovery-evidence",
        scenario=CanonicalRecoveryScenario.SOAK,
        scenario_evidence_sha256=exact_bytes_sha256(b"local-soak-observation"),
        fresh_receipt=fresh,
        replay_receipt=replay,
        authoritative_terminal_ids=(fresh.completion_semantic_sha256,),
        outbox_delivery_ids=fresh.outbox_ids,
        review_task_ids=(fresh.review_routing.review_task_id,),
        failure_observed=False,
        recovery_completed=True,
        run_duration_ns=1,
    )

    assert evidence.evidence_class is CanonicalRecoveryEvidenceClass.LOCAL_CONFORMANCE
    assert evidence.production_eligible is False
    assert evidence.fresh.run_id == evidence.replay.run_id == fresh.run_id
    assert evidence.fresh.command_sha256 == evidence.replay.command_sha256
    assert evidence.authoritative_terminal_count == 1
    assert evidence.outbox_delivery_ids == fresh.outbox_ids
    assert evidence.review_task_ids == (fresh.review_routing.review_task_id,)

    failed_reconciliation = build_canonical_recovery_qualification_evidence(
        workload_fingerprint=exact_bytes_sha256(b"local-recovery-workload"),
        run_namespace="local-recovery-evidence",
        scenario=CanonicalRecoveryScenario.BROKER_FAILURE,
        scenario_evidence_sha256=exact_bytes_sha256(b"local-transport-observation"),
        fresh_receipt=fresh,
        replay_receipt=replay,
        authoritative_terminal_ids=(fresh.completion_semantic_sha256,),
        outbox_delivery_ids=fresh.outbox_ids + fresh.outbox_ids,
        review_task_ids=(fresh.review_routing.review_task_id,),
        failure_observed=True,
        recovery_completed=False,
        run_duration_ns=1,
    )
    assert failed_reconciliation.duplicate_outbox_delivery_count == len(fresh.outbox_ids)

    missing_provider_failure = evidence.model_dump(mode="python")
    missing_provider_failure["scenario"] = CanonicalRecoveryScenario.PROVIDER_TIMEOUT
    with pytest.raises(ValueError, match="completed failure scenario"):
        CanonicalRecoveryQualificationEvidence.model_validate(missing_provider_failure)

    duplicate_terminal = evidence.model_dump(mode="python")
    duplicate_terminal["scenario"] = CanonicalRecoveryScenario.RESTART_REPLAY
    duplicate_terminal["failure_observed"] = True
    duplicate_terminal["authoritative_terminal_ids"] = (
        fresh.completion_semantic_sha256,
        fresh.completion_semantic_sha256,
    )
    duplicate_terminal["authoritative_terminal_count"] = 2
    duplicate_terminal["duplicate_terminal_count"] = 1
    with pytest.raises(ValueError, match="one non-duplicate authoritative terminal"):
        CanonicalRecoveryQualificationEvidence.model_validate(duplicate_terminal)

    duplicate_review = evidence.model_dump(mode="python")
    duplicate_review["scenario"] = CanonicalRecoveryScenario.BROKER_FAILURE
    duplicate_review["failure_observed"] = True
    duplicate_review["review_task_ids"] = (
        fresh.review_routing.review_task_id,
        fresh.review_routing.review_task_id,
    )
    duplicate_review["review_task_count"] = 2
    duplicate_review["duplicate_review_task_count"] = 1
    with pytest.raises(ValueError, match="review tasks do not match"):
        CanonicalRecoveryQualificationEvidence.model_validate(duplicate_review)

    upgraded = evidence.model_dump(mode="python")
    upgraded["evidence_class"] = CanonicalRecoveryEvidenceClass.PRODUCTION_QUALIFICATION
    with pytest.raises(ValueError, match="cannot upgrade receipt provenance"):
        CanonicalRecoveryQualificationEvidence.model_validate(upgraded)

    pending_outbox = evidence.model_dump(mode="python")
    pending_outbox["fresh"] = evidence.fresh.model_copy(
        update={"outbox_delivery_outcome": "PENDING"}
    )
    with pytest.raises(ValueError, match="delivered outbox"):
        CanonicalRecoveryQualificationEvidence.model_validate(pending_outbox)

    failed_review_routing = evidence.model_dump(mode="python")
    failed_review_routing["fresh"] = evidence.fresh.model_copy(
        update={"review_routing_disposition": "ROUTING_FAILED"}
    )
    with pytest.raises(ValueError, match="reconciled review"):
        CanonicalRecoveryQualificationEvidence.model_validate(failed_review_routing)

    missing_backlog = evidence.model_dump(mode="python")
    missing_backlog["scenario"] = CanonicalRecoveryScenario.BACKLOG_DRAIN
    with pytest.raises(ValueError, match="nonzero backlog"):
        CanonicalRecoveryQualificationEvidence.model_validate(missing_backlog)
