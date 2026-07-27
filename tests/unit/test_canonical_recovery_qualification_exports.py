from __future__ import annotations

from robata.application.canonical import (
    CanonicalRecoveryEvidenceClass,
    CanonicalRecoveryQualificationEvidence,
    CanonicalRecoveryReceiptEvidence,
    CanonicalRecoveryScenario,
    build_canonical_recovery_qualification_evidence,
    qualification_evidence,
)


def test_canonical_package_lazily_exports_recovery_qualification_types() -> None:
    assert CanonicalRecoveryEvidenceClass is qualification_evidence.CanonicalRecoveryEvidenceClass
    assert (
        CanonicalRecoveryQualificationEvidence
        is qualification_evidence.CanonicalRecoveryQualificationEvidence
    )
    assert (
        CanonicalRecoveryReceiptEvidence is qualification_evidence.CanonicalRecoveryReceiptEvidence
    )
    assert CanonicalRecoveryScenario is qualification_evidence.CanonicalRecoveryScenario
    assert (
        build_canonical_recovery_qualification_evidence
        is qualification_evidence.build_canonical_recovery_qualification_evidence
    )
