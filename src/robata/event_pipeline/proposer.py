"""Deterministic EVENT_PROPOSAL normalization for the canonical path."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import model_validator

from robata.contracts.cameras import CAMERA_IDS, SixCameraMap
from robata.contracts.common import (
    NanosecondInterval,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.contracts.pipeline import CameraEvidenceStatus, ProposalCameraClaim
from robata.contracts.temporal import TemporalPackageSet
from robata.inference.enrichment import (
    EnrichedEvidenceReference,
    EnrichedProviderClaim,
    OrchestratorEnrichedOutput,
)
from robata.inference.input_plan import InferenceInputPlan
from robata.inference.models import VisionTask


class EventProposalError(ValueError):
    """A provider proposal cannot be admitted as canonical evidence."""


class EventProposalOutcome(StrEnum):
    CLAIMS = "CLAIMS"
    NO_EVENTS = "NO_EVENTS"


class EventProposerConfig(StrictModel):
    """Legacy configuration retained for compatibility; execution is fail-closed."""

    version: SchemaVersion
    min_proposal_duration_ns: int
    max_proposals_per_recording: int
    overlap_threshold: float


class TemporalSignal(StrictModel):
    timestamp_ns: int
    signal_type: str
    strength: float


class MCAPRecording(StrictModel):
    mcap_id: OpaqueUuid
    duration_ns: int


class EventProposalOutputRef(StrictModel):
    source_inference_id: OpaqueUuid
    source_input_plan_semantic_sha256: Sha256Digest
    claim_ordinal: int
    enrichment_logical_key: NodeLogicalKey
    claim_id: OpaqueUuid


class NormalizedEventProposal(StrictModel):
    source_proposal_logical_key: NodeLogicalKey
    source: EventProposalOutputRef
    mcap_id: OpaqueUuid
    source_package_ids: tuple[OpaqueUuid, ...]
    interval: NanosecondInterval
    label: str
    model_reported_score: float | None
    camera_coverage: SixCameraMap[ProposalCameraClaim]
    evidence: tuple[EnrichedEvidenceReference, ...]
    production_eligible: bool = False


class EventProposalResult(StrictModel):
    task: VisionTask
    input_plan_semantic_sha256: Sha256Digest
    outcome: EventProposalOutcome
    proposals: tuple[NormalizedEventProposal, ...]
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: bool = False

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if (self.outcome is EventProposalOutcome.CLAIMS) != bool(self.proposals):
            raise ValueError("proposal CLAIMS outcome must exactly match nonempty proposals")
        digest = event_proposal_result_semantic_sha256(
            input_plan_semantic_sha256=self.input_plan_semantic_sha256,
            outcome=self.outcome,
            proposal_logical_keys=tuple(
                item.source_proposal_logical_key for item in self.proposals
            ),
        )
        if self.semantic_sha256 != digest or self.logical_key != f"event-proposal-result:{digest}":
            raise ValueError(
                "event proposal result identity does not match its semantic projection"
            )
        return self


def event_proposal_result_semantic_sha256(
    *,
    input_plan_semantic_sha256: Sha256Digest,
    outcome: EventProposalOutcome,
    proposal_logical_keys: Sequence[NodeLogicalKey],
) -> Sha256Digest:
    return semantic_sha256(
        {
            "semantic_projection_version": EventProposalProjector.policy_version,
            "input_plan_semantic_sha256": input_plan_semantic_sha256,
            "outcome": outcome,
            "proposal_logical_keys": list(proposal_logical_keys),
        }
    )


class EventProposalProjector:
    policy_version: SchemaVersion = "event-proposal-semantic-v1"

    def project(
        self,
        *,
        input_plan: InferenceInputPlan,
        package_set: TemporalPackageSet,
        enriched_outputs: Sequence[OrchestratorEnrichedOutput],
    ) -> EventProposalResult:
        if input_plan.subject.task is not VisionTask.EVENT_PROPOSAL:
            raise EventProposalError("projector requires EVENT_PROPOSAL input plan")
        if input_plan.request_catalog.task is not VisionTask.EVENT_PROPOSAL:
            raise EventProposalError("request catalog task is not EVENT_PROPOSAL")
        package_ids = tuple(member.package_id for member in package_set.members)
        if tuple(item.package_id for item in input_plan.subject.packages) != package_ids:
            raise EventProposalError("proposal input plan does not bind package-set members")
        if not enriched_outputs:
            raise EventProposalError("proposal projection requires enriched outputs")
        if any(output.task is not VisionTask.EVENT_PROPOSAL for output in enriched_outputs):
            raise EventProposalError("proposal enrichment task mismatch")
        if any(
            output.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or output.request_catalog_sha256 != input_plan.request_catalog.semantic_sha256
            for output in enriched_outputs
        ):
            raise EventProposalError("proposal enrichment is foreign to the input plan")
        if any(output.abstained for output in enriched_outputs):
            raise EventProposalError("EVENT_PROPOSAL outputs cannot abstain")
        proposals = tuple(
            sorted(
                (
                    self._normalize_claim(output, claim, package_set)
                    for output in enriched_outputs
                    for claim in output.claims
                ),
                key=lambda item: (
                    item.interval.start_ns,
                    item.interval.end_ns,
                    item.label,
                    item.source_proposal_logical_key,
                ),
            )
        )
        outcome = EventProposalOutcome.CLAIMS if proposals else EventProposalOutcome.NO_EVENTS
        digest = event_proposal_result_semantic_sha256(
            input_plan_semantic_sha256=input_plan.semantic_sha256,
            outcome=outcome,
            proposal_logical_keys=tuple(item.source_proposal_logical_key for item in proposals),
        )
        return EventProposalResult(
            task=VisionTask.EVENT_PROPOSAL,
            input_plan_semantic_sha256=input_plan.semantic_sha256,
            outcome=outcome,
            proposals=proposals,
            semantic_sha256=digest,
            logical_key=f"event-proposal-result:{digest}",
        )

    def _normalize_claim(
        self,
        output: OrchestratorEnrichedOutput,
        claim: EnrichedProviderClaim,
        package_set: TemporalPackageSet,
    ) -> NormalizedEventProposal:
        if claim.kind.value != "EVENT_PROPOSAL":
            raise EventProposalError("non-proposal claim reached proposal projector")
        if (
            claim.package_id is not None
            or claim.package_ordinal is not None
            or claim.camera_id is not None
        ):
            raise EventProposalError("proposal claim cannot author package/camera identity")
        if claim.interval is None or not claim.evidence:
            raise EventProposalError("proposal claim requires interval and evidence")
        interval = NanosecondInterval(
            start_ns=claim.interval.start_ns, end_ns=claim.interval.end_ns
        )
        if interval.start_ns < package_set.start_ns or interval.end_ns > package_set.end_ns:
            raise EventProposalError("proposal interval is outside package-set bounds")
        valid_ids = {member.package_id for member in package_set.members}
        package_ids = tuple(sorted({item.package_id for item in claim.evidence}))
        if not set(package_ids) <= valid_ids:
            raise EventProposalError("proposal evidence cites a package outside the package set")
        coverage = {
            camera_id: ProposalCameraClaim(
                camera_id=camera_id,
                status=(
                    CameraEvidenceStatus.SUPPORTING
                    if any(item.camera_id is camera_id for item in claim.evidence)
                    else CameraEvidenceStatus.MISSING
                ),
                frame_ordinals=tuple(
                    sorted(
                        {
                            item.frame_ordinal
                            for item in claim.evidence
                            if item.camera_id is camera_id
                        }
                    )
                ),
            )
            for camera_id in CAMERA_IDS
        }
        label = str(claim.label or "unlabeled-action").strip().lower()
        if not label:
            raise EventProposalError("proposal label cannot be empty")
        digest = semantic_sha256(
            {
                "semantic_projection_version": self.policy_version,
                "source_content_sha256": package_set.lineage.source_content_sha256,
                "camera_mapping_semantic_sha256": (
                    package_set.lineage.camera_mapping_semantic_sha256
                ),
                "alignment_semantic_sha256": package_set.lineage.alignment_semantic_sha256,
                "enrichment_logical_key": output.enrichment_logical_key,
                "claim_ordinal": claim.claim_ordinal,
                "interval": interval.model_dump(mode="json"),
                "label": label,
                "evidence_coordinates": sorted(
                    (item.package_ordinal, item.camera_ordinal, item.frame_ordinal)
                    for item in claim.evidence
                ),
            }
        )
        source = EventProposalOutputRef(
            source_inference_id=output.authority.inference_id,
            source_input_plan_semantic_sha256=output.input_plan_semantic_sha256,
            claim_ordinal=claim.claim_ordinal,
            enrichment_logical_key=output.enrichment_logical_key,
            claim_id=claim.claim_id,
        )
        return NormalizedEventProposal(
            source_proposal_logical_key=f"event-proposal:{digest}",
            source=source,
            mcap_id=package_set.mcap_id,
            source_package_ids=package_ids,
            interval=interval,
            label=label,
            model_reported_score=(
                claim.model_reported_confidence.value
                if claim.model_reported_confidence is not None
                else None
            ),
            camera_coverage=SixCameraMap(coverage),
            evidence=tuple(
                sorted(
                    claim.evidence,
                    key=lambda item: (
                        item.package_ordinal,
                        item.camera_ordinal,
                        item.frame_ordinal,
                    ),
                )
            ),
        )


class EventProposer:
    """Legacy API retained only as an explicit fail-closed compatibility surface."""

    def __init__(self, config: EventProposerConfig) -> None:
        self._config = config

    def propose(
        self,
        qa_complete_recording: MCAPRecording,
        coarse_packages: Sequence[object],
        temporal_signals: Sequence[TemporalSignal],
    ) -> Sequence[object]:
        raise NotImplementedError(
            "EventProposer.propose is non-runnable; use EventProposalProjector with enriched output"
        )


__all__ = [
    "EventProposalError",
    "EventProposalOutcome",
    "EventProposalOutputRef",
    "EventProposalProjector",
    "EventProposalResult",
    "EventProposer",
    "EventProposerConfig",
    "MCAPRecording",
    "NormalizedEventProposal",
    "TemporalSignal",
]
