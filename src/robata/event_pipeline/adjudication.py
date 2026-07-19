"""Fusion Adjudication: resolve conflicts among fusion decisions (Section 15).

Adjudicates multiple fusion decisions under a versioned policy,
producing final decisions, abstentions, and rationale.
"""

from __future__ import annotations

from collections.abc import Sequence

from robata.contracts.common import StrictModel
from robata.contracts.mainline import NonEmptyString, SchemaVersion


class FusionDecision(StrictModel):
    """One physical-event hypothesis output by the fusion engine.

    Mirrors the decision shape from :mod:`robata.event_pipeline.fusion`
    so adjudication can operate without importing the full fusion module.
    """

    event_id: str
    interval_start_ns: int
    interval_end_ns: int
    action_type: NonEmptyString
    confidence: float
    ambiguity_state: NonEmptyString


class AdjudicationPolicy(StrictModel):
    """Versioned policy for fusion adjudication."""

    version: SchemaVersion
    conflict_resolution_strategy: NonEmptyString


class AdjudicationResult(StrictModel):
    """Outcome of fusion adjudication.

    Contains final accepted decisions, abstained decisions, and the
    rationale for each outcome.
    """

    final_decisions: tuple[FusionDecision, ...]
    abstained: tuple[FusionDecision, ...]
    rationale: NonEmptyString


class FusionAdjudicator:
    """Adjudicate fusion decisions under a versioned policy.

    Resolves conflicts among multiple fusion hypotheses, handles
    ambiguous cases, and produces a final adjudication result with
    explicit rationale.
    """

    def adjudicate(
        self,
        decisions: Sequence[FusionDecision],
        policy: AdjudicationPolicy,
    ) -> AdjudicationResult:
        """Adjudicate a set of fusion decisions.

        Args:
            decisions: Fusion decisions to adjudicate.
            policy: Versioned adjudication policy controlling conflict
                resolution strategy.

        Returns:
            An :class:`AdjudicationResult` with final decisions,
            abstentions, and rationale.
        """
        # Skeleton: adjudication logic to be implemented per Section 15.
        _ = decisions
        _ = policy
        return AdjudicationResult(
            final_decisions=(),
            abstained=(),
            rationale="Skeleton: adjudication not yet implemented.",
        )


__all__ = [
    "AdjudicationPolicy",
    "AdjudicationResult",
    "FusionAdjudicator",
    "FusionDecision",
]
