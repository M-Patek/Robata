"""QA pipeline package.

This package contains the two-stage QA pipeline implementation
(Coarse -> Dense -> Final) as defined in Architecture V1 Section 12.
"""

from robata.qa_pipeline.aggregate import (
    QAAggregationPolicy,
    QAAggregator,
    RecordingQAResult,
)
from robata.qa_pipeline.coarse import (
    CameraCoarseResult,
    CoarseQAPipeline,
    CoarseQAResult,
    SamplingPlan,
)
from robata.qa_pipeline.coarse import (
    SuspiciousInterval as CoarseSuspiciousInterval,
)
from robata.qa_pipeline.completion import (
    DenseQAOutcome,
    DenseQAWorkManifest,
    QACompletionAggregate,
    QACompletionProjector,
    QACompletionResult,
    QACompletionStatus,
)
from robata.qa_pipeline.dense import (
    CameraDenseResult,
    CoarseQAObservationRef,
    DenseQAInputPlanRef,
    DenseQAOutputRef,
    DenseQAPackageRef,
    DenseQAPlanner,
    DenseQAPlanningPolicy,
    DenseQAProjectionError,
    DenseQAProjector,
    DenseQAResult,
    DenseQAStatus,
    DenseQAUnitEvidence,
    DenseQAUnitResult,
    DenseQAWorkItem,
    DenseQAWorkUnit,
)
from robata.qa_pipeline.fast_detector import (
    ContainerCheckResult,
    DecodeGapResult,
    FastDetector,
    FastDetectorConfig,
    StreamIntegrityResult,
    TimestampCheckResult,
    VideoStream,
)
from robata.qa_pipeline.stages import (
    CameraQAStatus,
    QAIssueSeverity,
    QAStage,
    QAStageIssue,
    QAStageResult,
    RecordingQAStatus,
)
from robata.qa_pipeline.suspicion_reducer import (
    ReducedInterval,
    ReductionPolicyVersion,
    SourceIntervalRef,
    SuspiciousInterval,
    SuspiciousIntervalReducer,
)

__all__ = [
    "CameraCoarseResult",
    "CameraDenseResult",
    "CameraQAStatus",
    "CoarseQAObservationRef",
    "CoarseQAPipeline",
    "CoarseQAResult",
    "CoarseSuspiciousInterval",
    "ContainerCheckResult",
    "DecodeGapResult",
    "DenseQAInputPlanRef",
    "DenseQAOutcome",
    "DenseQAOutputRef",
    "DenseQAPackageRef",
    "DenseQAPlanner",
    "DenseQAPlanningPolicy",
    "DenseQAProjectionError",
    "DenseQAProjector",
    "DenseQAResult",
    "DenseQAStatus",
    "DenseQAUnitEvidence",
    "DenseQAUnitResult",
    "DenseQAWorkItem",
    "DenseQAWorkManifest",
    "DenseQAWorkUnit",
    "FastDetector",
    "FastDetectorConfig",
    "QAAggregationPolicy",
    "QAAggregator",
    "QACompletionAggregate",
    "QACompletionProjector",
    "QACompletionResult",
    "QACompletionStatus",
    "QAIssueSeverity",
    "QAStage",
    "QAStageIssue",
    "QAStageResult",
    "RecordingQAResult",
    "RecordingQAStatus",
    "ReducedInterval",
    "ReductionPolicyVersion",
    "SamplingPlan",
    "SourceIntervalRef",
    "StreamIntegrityResult",
    "SuspiciousInterval",
    "SuspiciousIntervalReducer",
    "TimestampCheckResult",
    "VideoStream",
]
