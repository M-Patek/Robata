"""Future contract for ActionEvidence extraction (Section 13.2).

This non-runnable skeleton will build dense six-camera evidence packages and
normalize per-camera and cross-view evidence. Extraction remains unimplemented.
"""

from __future__ import annotations

from robata.contracts.pipeline import (
    ActionEvidence,
    CandidateEvent,
    TemporalVisualPackage,
)


class ActionEvidenceExtractor:
    """Future action-evidence extractor; currently non-runnable.

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
        _ = candidate
        _ = dense_package
        raise NotImplementedError("extract_evidence is a skeleton")


__all__ = [
    "ActionEvidenceExtractor",
]
