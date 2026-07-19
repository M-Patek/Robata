"""QA pipeline package.

This package contains the two-stage QA pipeline implementation
(Coarse -> Dense -> Final) as defined in Architecture V1 Section 12.
"""

from robata.qa_pipeline.aggregate import (
    QAAggregator,
    RecordingQAResult,
)
from robata.qa_pipeline.coarse import (
    CameraCoarseResult,
    CoarseQAPipeline,
    CoarseQAResult,
    SamplingPlan,
    SuspiciousInterval,
)
from robata.qa_pipeline.dense import (
    CameraDenseResult,
    DenseQAPipeline,
    DenseQAResult,
    IntervalDenseResult,
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
    QAStage,
    QAStageIssue,
    QAStageResult,
    QAIssueSeverity,
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
    # stages
    "CameraQAStatus",
    "QAStage",
    "QAStageIssue",
    "QAStageResult",
    "QAIssueSeverity",
    "RecordingQAStatus",
    # fast_detector
    "ContainerCheckResult",
    "DecodeGapResult",
    "FastDetector",
    "FastDetectorConfig",
    "StreamIntegrityResult",
    "TimestampCheckResult",
    "VideoStream",
    # coarse
    "CameraCoarseResult",
    "CoarseQAPipeline",
    "CoarseQAResult",
    "SamplingPlan",
    "SuspiciousInterval",
    # suspicion_reducer
    "ReducedInterval",
    "ReductionPolicyVersion",
    "SourceIntervalRef",
    "SuspiciousIntervalReducer",
    # dense
    "CameraDenseResult",
    "DenseQAPipeline",
    "DenseQAResult",
    "IntervalDenseResult",
    # aggregate
    "QAAggregator",
    "RecordingQAResult",
]
