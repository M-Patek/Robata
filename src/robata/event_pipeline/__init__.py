"""Event pipeline: proposal, evidence, boundary refinement, fusion, and adjudication."""

from robata.event_pipeline.adjudication import (
    AdjudicationPolicy,
    AdjudicationResult,
    FusionAdjudicator,
    FusionDecision,
)
from robata.event_pipeline.boundary import BoundaryRefiner, RefinedEvent
from robata.event_pipeline.candidate import (
    CandidateEventManager,
    ValidationIssue,
    ValidationResult,
)
from robata.event_pipeline.evidence import ActionEvidenceExtractor
from robata.event_pipeline.fusion import FusionDecision, FusionEngine, FusionPolicy
from robata.event_pipeline.proposer import (
    EventProposer,
    EventProposerConfig,
    MCAPRecording,
    TemporalSignal,
)

__all__ = [
    "ActionEvidenceExtractor",
    "AdjudicationPolicy",
    "AdjudicationResult",
    "BoundaryRefiner",
    "CandidateEventManager",
    "EventProposer",
    "EventProposerConfig",
    "FusionAdjudicator",
    "FusionDecision",
    "FusionEngine",
    "FusionPolicy",
    "MCAPRecording",
    "RefinedEvent",
    "TemporalSignal",
    "ValidationIssue",
    "ValidationResult",
]
