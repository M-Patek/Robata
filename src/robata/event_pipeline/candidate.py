"""CandidateEvent management: merge, split, and validate candidates."""

from __future__ import annotations

from collections.abc import Sequence

from robata.contracts.common import StrictModel
from robata.contracts.pipeline import (
    CandidateEvent,
    NonEmptyString,
)


class ValidationIssue(StrictModel):
    """A single validation finding."""

    code: NonEmptyString
    message: NonEmptyString


class ValidationResult(StrictModel):
    """Outcome of candidate validation."""

    is_valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()


class CandidateEventManager:
    """Manage candidate event lifecycle: merge, split, and validate.

    Validation rules (Section 13.1):
    - Canonical intervals are half-open, non-empty, and within MCAP duration.
    - camera_coverage contains exactly cam_01 through cam_06.
    - Overlap/merge never discards source proposal IDs.
    - Boundary/content changes create new immutable candidate IDs.
    """

    def merge_candidates(
        self,
        candidates: Sequence[CandidateEvent],
    ) -> Sequence[CandidateEvent]:
        """Merge overlapping or adjacent candidates into unified candidates.

        Preserves every source proposal ID in the merged result's lineage.
        Does not mutate input candidates; returns new immutable candidates.

        Args:
            candidates: Candidate events to merge.

        Returns:
            Merged candidate events.
        """
        raise NotImplementedError(
            "CandidateEventManager.merge_candidates is a non-runnable architecture skeleton; "
            "candidate merge policy is not implemented."
        )

    def split_candidate(
        self,
        candidate: CandidateEvent,
        split_points: Sequence[int],
    ) -> Sequence[CandidateEvent]:
        """Split a candidate at specified nanosecond timestamps.

        Creates new immutable candidate IDs for each resulting segment.
        The original candidate is not mutated.

        Args:
            candidate: The candidate to split.
            split_points: Nanosecond timestamps at which to split the
                candidate interval.

        Returns:
            New candidate segments.
        """
        raise NotImplementedError(
            "CandidateEventManager.split_candidate is a non-runnable architecture skeleton; "
            "candidate split policy is not implemented."
        )

    def validate_candidate(self, candidate: CandidateEvent) -> ValidationResult:
        """Validate a single candidate against architecture rules.

        Checks:
        - Interval is half-open, non-empty, and within MCAP duration.
        - camera_coverage contains cam_01 through cam_06.
        - Source proposal IDs are preserved.

        Args:
            candidate: The candidate to validate.

        Returns:
            Validation result with is_valid, errors, and warnings.
        """
        raise NotImplementedError(
            "CandidateEventManager.validate_candidate is a non-runnable architecture skeleton; "
            "candidate validation policy is not implemented."
        )


__all__ = [
    "CandidateEventManager",
    "ValidationIssue",
    "ValidationResult",
]
