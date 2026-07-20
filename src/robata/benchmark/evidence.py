"""Governed evidence identity for promotion-capable benchmark measurements."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import StringConstraints, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
EvidenceContextIdentity = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^benchmark-evidence:[0-9a-f]{64}$",
    ),
]


def benchmark_evidence_context_projection(
    *,
    benchmark_id: str,
    benchmark_manifest_digest: str,
    governed_corpus_digest: str,
    ground_truth_manifest_digest: str,
    grouped_split_manifest_digest: str,
    data_split: str,
    governance_approved: bool,
    governance_approval_id: str,
    governance_approval_digest: str,
    governance_policy_version: str,
) -> dict[str, str | bool]:
    """Return the complete semantic preimage for a benchmark evidence context."""

    return {
        "domain": "robata.benchmark-evidence-context",
        "schema_version": "1.0",
        "benchmark_id": benchmark_id,
        "benchmark_manifest_digest": benchmark_manifest_digest,
        "governed_corpus_digest": governed_corpus_digest,
        "ground_truth_manifest_digest": ground_truth_manifest_digest,
        "grouped_split_manifest_digest": grouped_split_manifest_digest,
        "data_split": data_split,
        "governance_approved": governance_approved,
        "governance_approval_id": governance_approval_id,
        "governance_approval_digest": governance_approval_digest,
        "governance_policy_version": governance_policy_version,
    }


class BenchmarkEvidenceContext(StrictModel):
    """Immutable identity of the governed inputs behind benchmark evidence.

    The context does not grant governance approval. It binds a benchmark result
    to an externally approved record and to the exact frozen manifests used by
    the calculation. Promotion still requires the referenced records to exist
    in the owning governance system.
    """

    schema_version: Literal["1.0"]
    context_identity: EvidenceContextIdentity
    context_digest: Sha256Digest
    benchmark_id: OpaqueUuid
    benchmark_manifest_digest: Sha256Digest
    governed_corpus_digest: Sha256Digest
    ground_truth_manifest_digest: Sha256Digest
    grouped_split_manifest_digest: Sha256Digest
    data_split: Literal["FROZEN_TEST"]
    governance_approved: Literal[True]
    governance_approval_id: NonEmptyString
    governance_approval_digest: Sha256Digest
    governance_policy_version: SchemaVersion

    @classmethod
    def create(
        cls,
        *,
        benchmark_id: OpaqueUuid,
        benchmark_manifest_digest: Sha256Digest,
        governed_corpus_digest: Sha256Digest,
        ground_truth_manifest_digest: Sha256Digest,
        grouped_split_manifest_digest: Sha256Digest,
        data_split: Literal["FROZEN_TEST"],
        governance_approved: Literal[True],
        governance_approval_id: NonEmptyString,
        governance_approval_digest: Sha256Digest,
        governance_policy_version: SchemaVersion,
    ) -> Self:
        """Build a context and derive its content-addressed identity."""

        projection = benchmark_evidence_context_projection(
            benchmark_id=benchmark_id,
            benchmark_manifest_digest=benchmark_manifest_digest,
            governed_corpus_digest=governed_corpus_digest,
            ground_truth_manifest_digest=ground_truth_manifest_digest,
            grouped_split_manifest_digest=grouped_split_manifest_digest,
            data_split=data_split,
            governance_approved=governance_approved,
            governance_approval_id=governance_approval_id,
            governance_approval_digest=governance_approval_digest,
            governance_policy_version=governance_policy_version,
        )
        digest = semantic_sha256(projection)
        return cls(
            schema_version="1.0",
            context_identity=f"benchmark-evidence:{digest}",
            context_digest=digest,
            benchmark_id=benchmark_id,
            benchmark_manifest_digest=benchmark_manifest_digest,
            governed_corpus_digest=governed_corpus_digest,
            ground_truth_manifest_digest=ground_truth_manifest_digest,
            grouped_split_manifest_digest=grouped_split_manifest_digest,
            data_split=data_split,
            governance_approved=governance_approved,
            governance_approval_id=governance_approval_id,
            governance_approval_digest=governance_approval_digest,
            governance_policy_version=governance_policy_version,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        projection = benchmark_evidence_context_projection(
            benchmark_id=self.benchmark_id,
            benchmark_manifest_digest=self.benchmark_manifest_digest,
            governed_corpus_digest=self.governed_corpus_digest,
            ground_truth_manifest_digest=self.ground_truth_manifest_digest,
            grouped_split_manifest_digest=self.grouped_split_manifest_digest,
            data_split=self.data_split,
            governance_approved=self.governance_approved,
            governance_approval_id=self.governance_approval_id,
            governance_approval_digest=self.governance_approval_digest,
            governance_policy_version=self.governance_policy_version,
        )
        expected_digest = semantic_sha256(projection)
        if self.context_digest != expected_digest:
            raise ValueError("context_digest does not match the evidence context")
        if self.context_identity != f"benchmark-evidence:{expected_digest}":
            raise ValueError("context_identity does not match context_digest")
        return self


__all__ = [
    "BenchmarkEvidenceContext",
    "EvidenceContextIdentity",
    "benchmark_evidence_context_projection",
]
