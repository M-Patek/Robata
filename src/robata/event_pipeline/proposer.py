"""Event proposal: high-recall action interval proposal."""

from __future__ import annotations

from collections.abc import Sequence

from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    StrictModel,
)
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mainline import (
    CandidateEvent,
    NonEmptyString,
    TemporalVisualPackage,
    UnitInterval,
)


class EventProposerConfig(StrictModel):
    """Configuration for the EventProposer.

    Controls recall-oriented proposal generation, overlap handling,
    and per-recording limits.
    """

    version: SchemaVersion
    min_proposal_duration_ns: Nanoseconds
    max_proposals_per_recording: int
    overlap_threshold: UnitInterval


class TemporalSignal(StrictModel):
    """Lightweight temporal change feature used to guide event proposals."""

    timestamp_ns: Nanoseconds
    signal_type: NonEmptyString
    strength: UnitInterval


class MCAPRecording(StrictModel):
    """Minimal recording reference for the proposer.

    Carries the MCAP identity and overall duration so the proposer can
    validate that proposals lie within recording bounds.
    """

    mcap_id: OpaqueUuid
    duration_ns: Nanoseconds


class EventProposer:
    """High-recall action interval proposal.

    Maximizes proposal recall, finds intervals of likely physical change,
    reduces duplicates caused by window overlap, preserves every source
    proposal, and bounds dense-stage expansion.
    """

    def __init__(self, config: EventProposerConfig) -> None:
        self._config = config

    def propose(
        self,
        qa_complete_recording: MCAPRecording,
        coarse_packages: Sequence[TemporalVisualPackage],
        temporal_signals: Sequence[TemporalSignal],
    ) -> Sequence[CandidateEvent]:
        """Generate high-recall candidate events from coarse packages and signals.

        Args:
            qa_complete_recording: The QA-complete recording context.
            coarse_packages: Coarse-sampled six-camera packages covering the
                recording interval.
            temporal_signals: Lightweight temporal change features (motion,
                scene change, etc.) used to guide proposals.

        Returns:
            A sequence of :class:`CandidateEvent` records representing
            proposed action intervals.  May be empty when no likely
            physical change is detected.
        """
        # Skeleton: concrete algorithm to be implemented per Section 13.
        _ = qa_complete_recording
        _ = coarse_packages
        _ = temporal_signals
        return []


__all__ = [
    "EventProposer",
    "EventProposerConfig",
    "MCAPRecording",
    "TemporalSignal",
]
