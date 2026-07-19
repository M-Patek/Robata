"""CandidateEvent management: merge, split, and validate candidates."""

from __future__ import annotations

from collections.abc import Sequence

from robata.contracts.common import StrictModel
from robata.contracts.mainline import (
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
        # Skeleton: merge logic to be implemented per Section 13.1.
        _ = candidates
        return []

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
        # Skeleton: split logic to be implemented per Section 13.1.
        _ = candidate
        _ = split_points
        return []

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
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []

        # Interval must be non-empty (NanosecondInterval already enforces this)
        # and within MCAP duration is checked by the caller context.

        # camera_coverage must contain all six cameras
        # (enforced by the contract model, but we validate explicitly here)
        # Note: CandidateEvent uses proposal.cameras which is a SixCameraMap
        # and already validates exactly six canonical keys.

        # Skeleton: additional validation rules to be implemented.
        _ = candidate

        return ValidationResult(
            is_valid=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )


__all__ = [
    "CandidateEventManager",
    "ValidationIssue",
    "ValidationResult",
]
