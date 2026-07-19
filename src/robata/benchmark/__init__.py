"""Benchmark framework for versioned, reproducible pipeline evaluation.

Implements Section 18 of the Architecture Design V1: a versioned benchmark
manifest, ground-truth annotations, data splits, experiment matrices,
metrics calculation, statistical analysis, and promotion gates.
"""

from robata.benchmark.experiments import (
    CameraAblationExperiment,
    DenseSamplingExperiment,
    EventProposalExperiment,
    ExperimentMatrix,
    QAExperiment,
    ShadowComparisonExperiment,
)
from robata.benchmark.ground_truth import (
    BoundaryExample,
    CameraEvidenceAnnotation,
    GroundTruthAnnotation,
    InterAnnotatorAgreement,
    Interval,
    PhysicalActionAnnotation,
    QAIssueAnnotation,
    RecordingUsability,
)
from robata.benchmark.metrics import (
    BoundaryMetrics,
    CalibrationMetrics,
    EventMetrics,
    MetricsCalculator,
    QAMetrics,
)
from robata.benchmark.models import (
    BenchmarkManifest,
    DataSplit,
    StratificationDimension,
)
from robata.benchmark.promotion import (
    BenchmarkResults,
    GateCategory,
    GateResult,
    PromotionDecision,
    PromotionEvaluator,
    PromotionGate,
    PromotionGateRegistry,
)
from robata.benchmark.splits import (
    DataSplitResult,
    DataSplitter,
    SplitConfig,
)
from robata.benchmark.statistics import (
    BootstrapResult,
    ConfidenceInterval,
    McNemarResult,
    StatisticalAnalyzer,
)

__all__ = [
    # models
    "BenchmarkManifest",
    # promotion
    "BenchmarkResults",
    # statistics
    "BootstrapResult",
    # ground_truth
    "BoundaryExample",
    # metrics
    "BoundaryMetrics",
    "CalibrationMetrics",
    # experiments
    "CameraAblationExperiment",
    "CameraEvidenceAnnotation",
    "ConfidenceInterval",
    "DataSplit",
    # splits
    "DataSplitResult",
    "DataSplitter",
    "DenseSamplingExperiment",
    "EventMetrics",
    "EventProposalExperiment",
    "ExperimentMatrix",
    "GateCategory",
    "GateResult",
    "GroundTruthAnnotation",
    "InterAnnotatorAgreement",
    "Interval",
    "McNemarResult",
    "MetricsCalculator",
    "PhysicalActionAnnotation",
    "PromotionDecision",
    "PromotionEvaluator",
    "PromotionGate",
    "PromotionGateRegistry",
    "QAExperiment",
    "QAIssueAnnotation",
    "QAMetrics",
    "RecordingUsability",
    "ShadowComparisonExperiment",
    "SplitConfig",
    "StatisticalAnalyzer",
    "StratificationDimension",
]
