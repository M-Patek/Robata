"""Deterministic candidate reduction for the canonical event path."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.contracts.pipeline import CameraEvidenceStatus, ProposalCameraClaim
from robata.contracts.temporal import TemporalPackageSet
from robata.event_pipeline.proposer import (
    EventProposalResult,
    NormalizedEventProposal,
)

CANDIDATE_EVENT_SEMANTIC_PROJECTION_VERSION: Final = "candidate-event-semantic-v2"
CANDIDATE_EVENT_LOGICAL_KEY_NAMESPACE: Final = "candidate-event-v2"
CANDIDATE_EVENT_UUID_DERIVATION_NAMESPACE: Final = "robata:candidate-event-v2"
CANDIDATE_REDUCTION_POLICY_PROJECTION_VERSION: Final = "candidate-reduction-policy-semantic-v2"
CANDIDATE_REDUCTION_SEMANTIC_PROJECTION_VERSION: Final = "candidate-reduction-semantic-v2"
CANDIDATE_REDUCTION_LOGICAL_KEY_NAMESPACE: Final = "candidate-reduction-v2"


class ValidationIssue(StrictModel):
    code: str
    message: str


class ValidationResult(StrictModel):
    is_valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()


class CandidateReductionError(ValueError):
    """Candidate reduction cannot preserve canonical lineage."""


class CandidateReductionPolicy(StrictModel):
    version: SchemaVersion
    merge_gap_ns: int = 0
    dense_padding_before_ns: int = 500_000_000
    dense_padding_after_ns: int = 500_000_000
    max_candidates: int = 128
    ontology_version: SchemaVersion = "fixture-action-ontology-v1"

    def validate_policy(self) -> CandidateReductionPolicy:
        if (
            self.merge_gap_ns < 0
            or self.dense_padding_before_ns < 0
            or self.dense_padding_after_ns < 0
        ):
            raise ValueError("candidate reduction durations must be nonnegative")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        return self


class CanonicalCandidateEvent(StrictModel):
    candidate_event_id: OpaqueUuid
    candidate_logical_key: NodeLogicalKey
    mcap_id: OpaqueUuid
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    source_package_ids: tuple[OpaqueUuid, ...]
    source_proposal_logical_keys: tuple[NodeLogicalKey, ...]
    effective_interval: NanosecondInterval
    requested_dense_interval: NanosecondInterval
    label: str
    reducer_policy_version: SchemaVersion
    reducer_policy_semantic_sha256: Sha256Digest
    ontology_version: SchemaVersion
    generation: int = 0
    parent_candidate_logical_key: NodeLogicalKey | None = None
    camera_coverage: SixCameraMap[ProposalCameraClaim]
    production_eligible: bool = False

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        digest = candidate_event_semantic_sha256(self)
        if (
            self.candidate_logical_key != f"{CANDIDATE_EVENT_LOGICAL_KEY_NAMESPACE}:{digest}"
            or self.candidate_event_id
            != str(
                uuid5(
                    NAMESPACE_URL,
                    f"{CANDIDATE_EVENT_UUID_DERIVATION_NAMESPACE}:{digest}",
                )
            )
        ):
            raise ValueError("candidate identity does not match its semantic projection")
        return self


class CandidateReductionResult(StrictModel):
    policy: CandidateReductionPolicy
    policy_semantic_sha256: Sha256Digest
    source_event_proposal_result_semantic_sha256: Sha256Digest
    candidates: tuple[CanonicalCandidateEvent, ...]
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: bool = False

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        policy_digest = candidate_policy_semantic_sha256(self.policy)
        digest = candidate_reduction_semantic_sha256(
            source_event_proposal_result_semantic_sha256=(
                self.source_event_proposal_result_semantic_sha256
            ),
            policy_semantic_sha256=policy_digest,
            candidate_logical_keys=tuple(item.candidate_logical_key for item in self.candidates),
        )
        if (
            self.policy_semantic_sha256 != policy_digest
            or self.semantic_sha256 != digest
            or self.logical_key != f"{CANDIDATE_REDUCTION_LOGICAL_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("candidate reduction identity does not match its projection")
        return self

    @property
    def no_events(self) -> bool:
        return not self.candidates


def candidate_policy_semantic_sha256(policy: CandidateReductionPolicy) -> Sha256Digest:
    return semantic_sha256(
        {
            "semantic_projection_version": CANDIDATE_REDUCTION_POLICY_PROJECTION_VERSION,
            "policy": policy.model_dump(mode="json"),
        }
    )


def candidate_event_semantic_sha256(candidate: CanonicalCandidateEvent) -> Sha256Digest:
    return _candidate_event_identity_sha256(
        source_content_sha256=candidate.source_content_sha256,
        camera_mapping_semantic_sha256=candidate.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=candidate.alignment_semantic_sha256,
        source_proposal_logical_keys=candidate.source_proposal_logical_keys,
        reducer_policy_semantic_sha256=candidate.reducer_policy_semantic_sha256,
        effective_interval=candidate.effective_interval,
        requested_dense_interval=candidate.requested_dense_interval,
        label=candidate.label,
        generation=candidate.generation,
        camera_coverage=candidate.camera_coverage,
    )


def _candidate_event_identity_sha256(
    *,
    source_content_sha256: Sha256Digest,
    camera_mapping_semantic_sha256: Sha256Digest,
    alignment_semantic_sha256: Sha256Digest,
    source_proposal_logical_keys: Sequence[NodeLogicalKey],
    reducer_policy_semantic_sha256: Sha256Digest,
    effective_interval: NanosecondInterval,
    requested_dense_interval: NanosecondInterval,
    label: str,
    generation: int,
    camera_coverage: SixCameraMap[ProposalCameraClaim],
) -> Sha256Digest:
    return semantic_sha256(
        {
            "semantic_projection_version": CANDIDATE_EVENT_SEMANTIC_PROJECTION_VERSION,
            "source_content_sha256": source_content_sha256,
            "camera_mapping_semantic_sha256": camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": alignment_semantic_sha256,
            "source_proposal_logical_keys": list(source_proposal_logical_keys),
            "reducer_policy_semantic_sha256": reducer_policy_semantic_sha256,
            "effective_interval": effective_interval.model_dump(mode="json"),
            "requested_dense_interval": requested_dense_interval.model_dump(mode="json"),
            "normalized_label": label,
            "generation": generation,
            "camera_coverage": camera_coverage.model_dump(mode="json"),
        }
    )


def candidate_reduction_semantic_sha256(
    *,
    source_event_proposal_result_semantic_sha256: Sha256Digest,
    policy_semantic_sha256: Sha256Digest,
    candidate_logical_keys: Sequence[NodeLogicalKey],
) -> Sha256Digest:
    return semantic_sha256(
        {
            "semantic_projection_version": CANDIDATE_REDUCTION_SEMANTIC_PROJECTION_VERSION,
            "source_event_proposal_result_semantic_sha256": (
                source_event_proposal_result_semantic_sha256
            ),
            "policy_semantic_sha256": policy_semantic_sha256,
            "candidate_logical_keys": list(candidate_logical_keys),
        }
    )


class CandidateReducer:
    """Merge deterministic proposals without run or attempt identity."""

    def __init__(self, policy: CandidateReductionPolicy) -> None:
        self._policy = policy.validate_policy()

    @property
    def policy(self) -> CandidateReductionPolicy:
        return self._policy

    def reduce(
        self,
        proposal_result: EventProposalResult,
        *,
        package_set: TemporalPackageSet,
    ) -> CandidateReductionResult:
        valid_package_ids = {member.package_id for member in package_set.members}
        if any(
            proposal.mcap_id != package_set.mcap_id
            or not set(proposal.source_package_ids) <= valid_package_ids
            or proposal.interval.start_ns < package_set.start_ns
            or proposal.interval.end_ns > package_set.end_ns
            for proposal in proposal_result.proposals
        ):
            raise CandidateReductionError("proposal is foreign to the candidate package set")
        ordered = sorted(
            proposal_result.proposals,
            key=lambda item: (
                item.interval.start_ns,
                item.interval.end_ns,
                item.label,
                item.source_proposal_logical_key,
            ),
        )
        groups: list[list[NormalizedEventProposal]] = []
        for proposal in ordered:
            if groups and self._can_merge(groups[-1], proposal):
                groups[-1].append(proposal)
            else:
                groups.append([proposal])
        if len(groups) > self._policy.max_candidates:
            raise CandidateReductionError("candidate reduction exceeded max_candidates")
        policy_digest = candidate_policy_semantic_sha256(self._policy)
        candidates = tuple(
            self._build_candidate(group, package_set, policy_digest) for group in groups
        )
        digest = candidate_reduction_semantic_sha256(
            source_event_proposal_result_semantic_sha256=proposal_result.semantic_sha256,
            policy_semantic_sha256=policy_digest,
            candidate_logical_keys=tuple(item.candidate_logical_key for item in candidates),
        )
        return CandidateReductionResult(
            policy=self._policy,
            policy_semantic_sha256=policy_digest,
            source_event_proposal_result_semantic_sha256=proposal_result.semantic_sha256,
            candidates=candidates,
            semantic_sha256=digest,
            logical_key=f"{CANDIDATE_REDUCTION_LOGICAL_KEY_NAMESPACE}:{digest}",
        )

    def _can_merge(
        self,
        group: Sequence[NormalizedEventProposal],
        right: NormalizedEventProposal,
    ) -> bool:
        return (
            group[0].label == right.label
            and right.interval.start_ns
            <= max(item.interval.end_ns for item in group) + self._policy.merge_gap_ns
        )

    def _build_candidate(
        self,
        group: Sequence[NormalizedEventProposal],
        package_set: TemporalPackageSet,
        policy_digest: Sha256Digest,
    ) -> CanonicalCandidateEvent:
        if not group:
            raise CandidateReductionError("candidate group cannot be empty")
        start_ns = min(item.interval.start_ns for item in group)
        end_ns = max(item.interval.end_ns for item in group)
        effective = NanosecondInterval(start_ns=start_ns, end_ns=end_ns)
        # Preserve requested padding.  The candidate-scoped ACTION_DENSE window
        # owns clipping against admitted recording bounds; clipping here would
        # silently erase context truncation when the proposal root is narrower.
        dense = NanosecondInterval(
            start_ns=start_ns - self._policy.dense_padding_before_ns,
            end_ns=end_ns + self._policy.dense_padding_after_ns,
        )
        source_keys = tuple(sorted({item.source_proposal_logical_key for item in group}))
        package_ids = tuple(
            sorted({package_id for item in group for package_id in item.source_package_ids})
        )
        coverage = self._merge_coverage(group)
        digest = _candidate_event_identity_sha256(
            source_content_sha256=package_set.lineage.source_content_sha256,
            camera_mapping_semantic_sha256=(package_set.lineage.camera_mapping_semantic_sha256),
            alignment_semantic_sha256=package_set.lineage.alignment_semantic_sha256,
            source_proposal_logical_keys=source_keys,
            reducer_policy_semantic_sha256=policy_digest,
            effective_interval=effective,
            requested_dense_interval=dense,
            label=group[0].label,
            generation=0,
            camera_coverage=coverage,
        )
        return CanonicalCandidateEvent(
            candidate_event_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"{CANDIDATE_EVENT_UUID_DERIVATION_NAMESPACE}:{digest}",
                )
            ),
            candidate_logical_key=f"{CANDIDATE_EVENT_LOGICAL_KEY_NAMESPACE}:{digest}",
            mcap_id=package_set.mcap_id,
            source_content_sha256=package_set.lineage.source_content_sha256,
            camera_mapping_semantic_sha256=(package_set.lineage.camera_mapping_semantic_sha256),
            alignment_semantic_sha256=package_set.lineage.alignment_semantic_sha256,
            source_package_ids=package_ids,
            source_proposal_logical_keys=source_keys,
            effective_interval=effective,
            requested_dense_interval=dense,
            label=group[0].label,
            reducer_policy_version=self._policy.version,
            reducer_policy_semantic_sha256=policy_digest,
            ontology_version=self._policy.ontology_version,
            generation=0,
            camera_coverage=coverage,
        )

    @staticmethod
    def _merge_coverage(
        group: Sequence[NormalizedEventProposal],
    ) -> SixCameraMap[ProposalCameraClaim]:
        coverage: dict[CameraId, ProposalCameraClaim] = {}
        for camera_id in CAMERA_IDS:
            ordinals = sorted(
                {
                    ordinal
                    for proposal in group
                    for ordinal in proposal.camera_coverage[camera_id].frame_ordinals
                    if proposal.camera_coverage[camera_id].status
                    is not CameraEvidenceStatus.MISSING
                }
            )
            coverage[camera_id] = ProposalCameraClaim(
                camera_id=camera_id,
                status=CameraEvidenceStatus.SUPPORTING
                if ordinals
                else CameraEvidenceStatus.MISSING,
                frame_ordinals=tuple(ordinals),
            )
        return SixCameraMap(coverage)


class CandidateEventManager:
    """Legacy manager retained as explicit fail-closed compatibility surface."""

    def merge_candidates(self, candidates: Sequence[object]) -> Sequence[object]:
        raise NotImplementedError(
            "CandidateEventManager.merge_candidates is non-runnable; use CandidateReducer"
        )

    def split_candidate(self, candidate: object, split_points: Sequence[int]) -> Sequence[object]:
        raise NotImplementedError(
            "CandidateEventManager.split_candidate is non-runnable; use versioned candidate policy"
        )

    def validate_candidate(self, candidate: object) -> object:
        raise NotImplementedError(
            "CandidateEventManager.validate_candidate is non-runnable; use CandidateReducer output"
        )


__all__ = [
    "CANDIDATE_EVENT_LOGICAL_KEY_NAMESPACE",
    "CANDIDATE_EVENT_SEMANTIC_PROJECTION_VERSION",
    "CANDIDATE_EVENT_UUID_DERIVATION_NAMESPACE",
    "CANDIDATE_REDUCTION_LOGICAL_KEY_NAMESPACE",
    "CANDIDATE_REDUCTION_POLICY_PROJECTION_VERSION",
    "CANDIDATE_REDUCTION_SEMANTIC_PROJECTION_VERSION",
    "CandidateEventManager",
    "CandidateReducer",
    "CandidateReductionError",
    "CandidateReductionPolicy",
    "CandidateReductionResult",
    "CanonicalCandidateEvent",
    "ValidationIssue",
    "ValidationResult",
]
