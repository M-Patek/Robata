"""Benchmark framework for versioned, reproducible pipeline evaluation.

Implements Section 18 of the Architecture Design V1: a versioned benchmark
manifest, ground-truth annotations, data splits, experiment matrices,
metrics calculation, statistical analysis, and promotion gates.
"""

from robata.benchmark.evidence import (
    BenchmarkEvidenceContext,
    EvidenceContextIdentity,
    benchmark_evidence_context_projection,
)
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
    EvidenceBoundMetrics,
    MeasurementStatus,
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
    SplitMetadataError,
    SplitRecord,
)
from robata.benchmark.statistics import (
    BootstrapResult,
    ConfidenceInterval,
    McNemarResult,
    StatisticalAnalyzer,
)

__all__ = [
    "BenchmarkEvidenceContext",
    "BenchmarkManifest",
    "BenchmarkResults",
    "BootstrapResult",
    "BoundaryExample",
    "BoundaryMetrics",
    "CalibrationMetrics",
    "CameraAblationExperiment",
    "CameraEvidenceAnnotation",
    "ConfidenceInterval",
    "DataSplit",
    "DataSplitResult",
    "DataSplitter",
    "DenseSamplingExperiment",
    "EventMetrics",
    "EventProposalExperiment",
    "EvidenceBoundMetrics",
    "EvidenceContextIdentity",
    "ExperimentMatrix",
    "GateCategory",
    "GateResult",
    "GroundTruthAnnotation",
    "InterAnnotatorAgreement",
    "Interval",
    "McNemarResult",
    "MeasurementStatus",
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
    "SplitMetadataError",
    "SplitRecord",
    "StatisticalAnalyzer",
    "StratificationDimension",
    "benchmark_evidence_context_projection",
]
