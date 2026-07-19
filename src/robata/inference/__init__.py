"""Inference orchestrator, adapter protocol, and models.

Architecture V1.1 — Sections 9 (VisionModelAdapter), 10 (Qwen primary path),
and 11 (GPT shadow path).
"""

from robata.inference.adapter import (
    JsonSchemaRef,
    NormalizedOutputEnvelope,
    PackageInput,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionModelAdapter,
    VisionUsage,
)
from robata.inference.evaluation import (
    EvaluationResult,
    EvaluationService,
    FieldDelta,
)
from robata.inference.models import (
    CapabilitySnapshot,
    ConcurrencyClass,
    InferenceAttemptSelection,
    InferenceFailure,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    ModelDisagreementSample,
    ModelInference,
    ModelInferenceUsage,
    ProductionDecision,
    Retryability,
    ShadowRoute,
    ShadowRouteStatus,
    ShadowSelectionReason,
    VisionTask,
)
from robata.inference.orchestrator import InferenceOrchestrator
from robata.inference.shadow import ShadowRouter

__all__ = [
    # Adapter
    "JsonSchemaRef",
    "NormalizedOutputEnvelope",
    "PackageInput",
    "VisionInferenceFailure",
    "VisionInferenceRequest",
    "VisionInferenceSuccess",
    "VisionModelAdapter",
    "VisionUsage",
    # Evaluation
    "EvaluationResult",
    "EvaluationService",
    "FieldDelta",
    # Models
    "CapabilitySnapshot",
    "ConcurrencyClass",
    "InferenceAttemptSelection",
    "InferenceFailure",
    "InferenceStatus",
    "InputMode",
    "ModelCapabilities",
    "ModelDisagreementSample",
    "ModelInference",
    "ModelInferenceUsage",
    "ProductionDecision",
    "Retryability",
    "ShadowRoute",
    "ShadowRouteStatus",
    "ShadowSelectionReason",
    "VisionTask",
    # Orchestrator
    "InferenceOrchestrator",
    # Shadow
    "ShadowRouter",
]
