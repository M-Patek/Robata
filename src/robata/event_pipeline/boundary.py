"""Boundary Refinement: precise onset/offset estimation (Section 13.3).

Separates action detection from boundary estimation.  Outputs include
onset/end intervals or uncertainty, with sources of uncertainty explicitly
tracked: sample spacing, alignment residual, camera disagreement,
visibility, and package-edge contact.
"""

from __future__ import annotations

from robata.contracts.common import NanosecondInterval, Nanoseconds, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mainline import (
    BoundaryRefinement,
    CandidateEvent,
    NonEmptyString,
    SchemaVersion,
)


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
        # Skeleton: boundary refinement to be implemented per Section 13.3.
        _ = coarse_event
        _ = boundary_evidence
        raise NotImplementedError("refine is a skeleton")


__all__ = [
    "BoundaryRefiner",
    "RefinedEvent",
]
