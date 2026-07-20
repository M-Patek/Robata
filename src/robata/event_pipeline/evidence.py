"""ActionEvidence extraction (Section 13.2).

Builds dense six-camera evidence packages, runs per-camera and optional
cross-view hypothesis generation, normalizes evidence, and performs
provisional fusion to separate simultaneous physical actions.
"""

from __future__ import annotations

from robata.contracts.mainline import (
    ActionEvidence,
    CandidateEvent,
    TemporalVisualPackage,
)


class ActionEvidenceExtractor:
    """Extract action evidence from a dense package for a candidate event.

    Steps:
    1. Add context padding and clip to recording bounds.
    2. Construct a dense six-camera TemporalVisualPackage.
    3. Generate per-camera hypotheses and optional cross-view hypotheses.
    4. Normalize exactly six evidence entries.
    5. Provisionally fuse to separate simultaneous physical actions.
    """

    def extract_evidence(
        self,
        candidate: CandidateEvent,
        dense_package: TemporalVisualPackage,
    ) -> ActionEvidence:
        """Extract action evidence for a candidate from a dense package.

        Args:
            candidate: The candidate event being analyzed.
            dense_package: A dense six-camera TemporalVisualPackage covering
                the candidate interval plus context padding.

        Returns:
            Normalized six-camera action evidence.
        """
        # Skeleton: evidence extraction to be implemented per Section 13.2.
        _ = candidate
        _ = dense_package
        # Return a placeholder; real implementation will construct from
        # inference results.
        raise NotImplementedError("extract_evidence is a skeleton")


__all__ = [
    "ActionEvidenceExtractor",
]
