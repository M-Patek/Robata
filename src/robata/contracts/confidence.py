"""Confidence value contract with full provenance.

Every confidence value carries its semantics, producer, and calibration lineage.
Bare numbers in ``[0,1]`` are never interchangeable merely because they share a range.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.common import StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class ConfidenceKind(StrEnum):
    """How a confidence value was produced."""

    MODEL_REPORTED = "MODEL_REPORTED"
    CALIBRATED = "CALIBRATED"
    POLICY_DERIVED = "POLICY_DERIVED"
    DETERMINISTIC = "DETERMINISTIC"


class ConfidenceProducerType(StrEnum):
    """What kind of system produced the confidence value."""

    MODEL_ATTEMPT = "MODEL_ATTEMPT"
    CALIBRATOR = "CALIBRATOR"
    POLICY = "POLICY"
    ALGORITHM = "ALGORITHM"


class ConfidenceValue(StrictModel):
    """A confidence value with explicit semantics and provenance.

    The stored form gives each confidence value a ``confidence_id``;
    nested JSON may embed the same fields for transport.
    """

    value: float | None = Field(
        ...,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Confidence value in [0,1] or null when unavailable.",
    )
    kind: ConfidenceKind
    semantics: NonEmptyString
    producer_type: ConfidenceProducerType
    producer_id: NonEmptyString
    producer_version: NonEmptyString
    calibration_artifact_id: NonEmptyString | None = None
    source_confidence_ids: tuple[NonEmptyString, ...] = ()


class ReportedConfidence(ConfidenceValue):
    """Model-reported confidence before calibration."""

    kind: ConfidenceKind = ConfidenceKind.MODEL_REPORTED
    producer_type: ConfidenceProducerType = ConfidenceProducerType.MODEL_ATTEMPT
    semantics: NonEmptyString = "provider_self_report"


class CalibratedConfidence(ConfidenceValue):
    """Confidence after calibration."""

    kind: ConfidenceKind = ConfidenceKind.CALIBRATED
    producer_type: ConfidenceProducerType = ConfidenceProducerType.CALIBRATOR
    semantics: NonEmptyString = "P(task_output_correct)"


class PolicyDerivedConfidence(ConfidenceValue):
    """Confidence derived from a versioned policy."""

    kind: ConfidenceKind = ConfidenceKind.POLICY_DERIVED
    producer_type: ConfidenceProducerType = ConfidenceProducerType.POLICY


class DeterministicConfidence(ConfidenceValue):
    """Deterministic confidence from an algorithm."""

    kind: ConfidenceKind = ConfidenceKind.DETERMINISTIC
    producer_type: ConfidenceProducerType = ConfidenceProducerType.ALGORITHM


__all__ = [
    "CalibratedConfidence",
    "ConfidenceKind",
    "ConfidenceProducerType",
    "ConfidenceValue",
    "DeterministicConfidence",
    "PolicyDerivedConfidence",
    "ReportedConfidence",
]
