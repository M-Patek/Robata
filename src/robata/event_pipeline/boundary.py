"""Boundary Refinement: precise onset/offset estimation (Section 13.3).

Separates action detection from boundary estimation.  Outputs include
onset/end intervals or uncertainty, with sources of uncertainty explicitly
tracked: sample spacing, alignment residual, camera disagreement,
visibility, and package-edge contact.
"""

from __future__ import annotations

from statistics import median_low
from typing import Annotated
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.pipeline import (
    BoundaryRefinement,
    BoundaryStatus,
    CandidateEvent,
)


class BoundaryRefinementPolicy(StrictModel):
    """Versioned deterministic policy for fusing per-camera boundary claims."""

    version: SchemaVersion
    minimum_observed_cameras: Annotated[int, Field(strict=True, ge=1, le=6)] = 2
    allow_candidate_fallback: bool = True


class RefinedEvent(StrictModel):
    """A candidate event with refined temporal boundaries.

    Carries onset/offset intervals or explicit uncertainty rather than
    scalar timestamps alone.
    """

    event_id: OpaqueUuid
    interval: NanosecondInterval
    onset_interval: NanosecondInterval | None = None
    offset_interval: NanosecondInterval | None = None
    uncertainty_ns: Nanoseconds
    observed_camera_count: Annotated[int, Field(strict=True, ge=0, le=6)]
    used_fallback: bool
    policy_version: SchemaVersion
    production_eligible: bool = False


class BoundaryRefiner:
    """Refine coarse event boundaries using dense boundary evidence.

    Detection and boundary estimation are separate stages:
    - coarse event and uncertainty
      -> padded onset/end refinement windows
      -> dense timestamp-based sampling
      -> Qwen BOUNDARY_REFINEMENT
      -> per-camera onset/end evidence
      -> final fusion and ActionEvent revision

    Sources of uncertainty:
    - Sample spacing
    - Alignment residual
    - Camera disagreement
    - Visibility
    - Package-edge contact
    """

    def __init__(self, policy: BoundaryRefinementPolicy | None = None) -> None:
        self._policy = policy or BoundaryRefinementPolicy(
            version="local-boundary-v1",
            minimum_observed_cameras=2,
            allow_candidate_fallback=True,
        )

    @staticmethod
    def _estimate(
        intervals: tuple[NanosecondInterval, ...],
    ) -> tuple[int, int]:
        centers = tuple(interval.start_ns + interval.duration_ns // 2 for interval in intervals)
        estimate = median_low(sorted(centers))
        uncertainty = max(
            abs(center - estimate) + (interval.duration_ns + 1) // 2
            for center, interval in zip(centers, intervals, strict=True)
        )
        return estimate, uncertainty

    def refine(
        self,
        coarse_event: CandidateEvent,
        boundary_evidence: BoundaryRefinement,
    ) -> RefinedEvent:
        """Refine the temporal boundaries of a coarse event.

        Args:
            coarse_event: The candidate event with coarse interval.
            boundary_evidence: Per-camera boundary refinement evidence
                from the BOUNDARY_REFINEMENT inference task.

        Returns:
            A :class:`RefinedEvent` with onset/offset intervals or
            explicit uncertainty.
        """
        observed_claims = tuple(
            claim
            for claim in boundary_evidence.cameras.values()
            if claim.status is BoundaryStatus.OBSERVED
        )
        for claim in boundary_evidence.cameras.values():
            if claim.observed_interval is None:
                continue
            dense = coarse_event.dense_interval
            observed = claim.observed_interval
            if observed.start_ns < dense.start_ns or observed.end_ns > dense.end_ns:
                raise ValueError("boundary observations must lie in the candidate dense interval")

        onset_intervals = tuple(
            claim.onset_interval for claim in observed_claims if claim.onset_interval is not None
        )
        offset_intervals = tuple(
            claim.offset_interval for claim in observed_claims if claim.offset_interval is not None
        )
        sufficient = (
            len(observed_claims) >= self._policy.minimum_observed_cameras
            and len(onset_intervals) == len(observed_claims)
            and len(offset_intervals) == len(observed_claims)
        )
        used_fallback = not sufficient
        if sufficient:
            start_ns, start_uncertainty = self._estimate(onset_intervals)
            end_ns, end_uncertainty = self._estimate(offset_intervals)
            if start_ns >= end_ns:
                used_fallback = True
        if used_fallback:
            if not self._policy.allow_candidate_fallback:
                raise ValueError("insufficient valid boundary evidence and fallback is disabled")
            interval = coarse_event.proposal.interval
            start_ns, end_ns = interval.start_ns, interval.end_ns
            start_uncertainty = max(0, interval.start_ns - coarse_event.dense_interval.start_ns)
            end_uncertainty = max(0, coarse_event.dense_interval.end_ns - interval.end_ns)
            onset_interval = None
            offset_interval = None
        else:
            interval = NanosecondInterval(start_ns=start_ns, end_ns=end_ns)
            onset_interval = NanosecondInterval(
                start_ns=max(coarse_event.dense_interval.start_ns, start_ns - start_uncertainty),
                end_ns=min(coarse_event.dense_interval.end_ns, start_ns + start_uncertainty + 1),
            )
            offset_interval = NanosecondInterval(
                start_ns=max(coarse_event.dense_interval.start_ns, end_ns - end_uncertainty),
                end_ns=min(coarse_event.dense_interval.end_ns, end_ns + end_uncertainty + 1),
            )
        uncertainty = max(start_uncertainty, end_uncertainty)
        digest = semantic_sha256(
            {
                "candidate_event_id": coarse_event.candidate_event_id,
                "boundary_evidence": boundary_evidence,
                "interval": interval,
                "policy_version": self._policy.version,
                "used_fallback": used_fallback,
            }
        )
        event_id = str(uuid5(NAMESPACE_URL, f"robata:refined-event:{digest}"))
        return RefinedEvent(
            event_id=event_id,
            interval=interval,
            onset_interval=onset_interval,
            offset_interval=offset_interval,
            uncertainty_ns=uncertainty,
            observed_camera_count=len(observed_claims),
            used_fallback=used_fallback,
            policy_version=self._policy.version,
            production_eligible=False,
        )


__all__ = [
    "BoundaryRefinementPolicy",
    "BoundaryRefiner",
    "RefinedEvent",
]
