"""Deterministic timestamp-grid sampling primitives."""

from robata.sampling.adaptive import (
    AdaptiveSampler,
    AdaptiveSamplingPolicy,
    AdaptiveSamplingResult,
    AdaptiveSignal,
    SignalDetector,
    SignalTrigger,
)
from robata.sampling.dense import (
    DenseSamplingPlanner,
    DenseSplitPolicy,
)
from robata.sampling.grid import (
    NANOSECONDS_PER_SECOND,
    CandidateFrame,
    FrameCandidate,
    FrameSelection,
    SamplingGrid,
    SamplingRate,
    SamplingTarget,
    SelectionStatus,
    TargetSelection,
    select_nearest_frames,
)
from robata.sampling.package_set import (
    PackageSetBuilder,
    TemporalPackageSet,
    TemporalPackageSetMember,
)
from robata.sampling.signals import (
    BlurDetector,
    MotionEnergyDetector,
    SceneChangeDetector,
)

__all__ = [
    "NANOSECONDS_PER_SECOND",
    "AdaptiveSampler",
    "AdaptiveSamplingPolicy",
    "AdaptiveSamplingResult",
    "AdaptiveSignal",
    "BlurDetector",
    "CandidateFrame",
    "DenseSamplingPlanner",
    "DenseSplitPolicy",
    "FrameCandidate",
    "FrameSelection",
    "MotionEnergyDetector",
    "PackageSetBuilder",
    "SamplingGrid",
    "SamplingRate",
    "SamplingTarget",
    "SceneChangeDetector",
    "SelectionStatus",
    "SignalDetector",
    "SignalTrigger",
    "TargetSelection",
    "TemporalPackageSet",
    "TemporalPackageSetMember",
    "select_nearest_frames",
]
