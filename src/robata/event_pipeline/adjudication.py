"""Deterministic adjudication of local multi-view fusion hypotheses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from robata.contracts.common import SchemaVersion, StrictModel
from robata.contracts.pipeline import NonEmptyString
from robata.event_pipeline.fusion import FusionDecision


class AdjudicationPolicy(StrictModel):
    """Versioned conflict policy; it never upgrades provider evidence."""

    version: SchemaVersion
    conflict_resolution_strategy: Literal[
        "HIGHEST_CONFIDENCE",
        "ABSTAIN_ON_CONFLICT",
        "KEEP_NON_OVERLAPPING",
    ]
    confidence_margin: float = Field(strict=True, ge=0.0, le=1.0, default=0.0)


class AdjudicationResult(StrictModel):
    """Accepted and explicitly abstained fusion hypotheses."""

    final_decisions: tuple[FusionDecision, ...]
    abstained: tuple[FusionDecision, ...]
    rationale: NonEmptyString


def _overlap(left: FusionDecision, right: FusionDecision) -> bool:
    return (
        left.interval.start_ns < right.interval.end_ns
        and right.interval.start_ns < left.interval.end_ns
    )


class FusionAdjudicator:
    """Resolve overlapping hypotheses without silently dropping evidence."""

    def adjudicate(
        self,
        decisions: Sequence[FusionDecision],
        policy: AdjudicationPolicy,
    ) -> AdjudicationResult:
        if not isinstance(policy, AdjudicationPolicy):
            raise TypeError("policy must be an AdjudicationPolicy")
        ordered = sorted(
            decisions,
            key=lambda item: (
                -item.confidence,
                item.event_id,
                item.interval.start_ns,
                item.interval.end_ns,
            ),
        )
        by_id: dict[str, FusionDecision] = {}
        for decision in ordered:
            existing = by_id.get(decision.event_id)
            if existing is not None and existing != decision:
                raise ValueError("conflicting replay for one fusion event_id")
            by_id[decision.event_id] = decision

        final: list[FusionDecision] = []
        abstained: list[FusionDecision] = []
        reasons: list[str] = []
        for decision in ordered:
            if decision in abstained or decision in final:
                continue
            if decision.ambiguity_state != "RESOLVED":
                abstained.append(decision)
                reasons.append(f"{decision.event_id}:AMBIGUOUS")
                continue
            conflicts = [candidate for candidate in final if _overlap(decision, candidate)]
            if not conflicts:
                final.append(decision)
                continue
            winner = max(
                (decision, *conflicts),
                key=lambda item: (item.confidence, item.event_id),
            )
            close = any(
                abs(winner.confidence - candidate.confidence) <= policy.confidence_margin
                for candidate in (decision, *conflicts)
                if candidate is not winner
            )
            if policy.conflict_resolution_strategy == "ABSTAIN_ON_CONFLICT" or (
                close and policy.confidence_margin > 0
            ):
                final[:] = [item for item in final if item not in conflicts]
                abstained.extend((*conflicts, decision))
                reasons.append(f"{decision.event_id}:CONFLICT_ABSTAIN")
            elif policy.conflict_resolution_strategy in {
                "HIGHEST_CONFIDENCE",
                "KEEP_NON_OVERLAPPING",
            }:
                final[:] = [item for item in final if item not in conflicts]
                final.append(winner)
                abstained.extend(item for item in (*conflicts, decision) if item is not winner)
                reasons.append(f"{winner.event_id}:CONFLICT_WINNER")
            else:
                raise ValueError("unsupported conflict resolution strategy")

        # Preserve stable order in both append-only projections.
        final = sorted(final, key=lambda item: (item.interval.start_ns, item.event_id))
        abstained = sorted(
            {item.event_id: item for item in abstained}.values(),
            key=lambda item: (item.interval.start_ns, item.event_id),
        )
        rationale = (
            f"policy={policy.version};strategy={policy.conflict_resolution_strategy};"
            f"final={len(final)};abstained={len(abstained)}"
            + (";" + ",".join(sorted(reasons)) if reasons else ";no_conflicts")
        )
        return AdjudicationResult(
            final_decisions=tuple(final),
            abstained=tuple(abstained),
            rationale=rationale,
        )


__all__ = [
    "AdjudicationPolicy",
    "AdjudicationResult",
    "FusionAdjudicator",
    "FusionDecision",
]
