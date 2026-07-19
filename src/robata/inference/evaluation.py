"""Paired evaluation for Qwen vs GPT shadow inference.

Architecture V1.1 — Section 11.3 (Paired evaluation and disagreements).
"""

from __future__ import annotations

from robata.contracts.common import Nanoseconds, SchemaVersion, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.inference.models import (
    ModelDisagreementSample,
    ModelInference,
    NonEmptyString,
    ShadowSelectionReason,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FieldDelta(StrictModel):
    """A single field-level disagreement between Qwen and GPT outputs."""

    path: NonEmptyString
    qwen: object | None = None
    gpt: object | None = None
    severity: NonEmptyString


class EvaluationResult(StrictModel):
    """Structured result of a paired model evaluation."""

    schema_version: Literal["1.0"]
    evaluation_id: OpaqueUuid
    qwen_inference_id: OpaqueUuid
    gpt_inference_id: OpaqueUuid
    shadow_route_id: OpaqueUuid
    field_deltas: tuple[FieldDelta, ...]
    status: NonEmptyString
    comparison_contract_version: SchemaVersion
    created_at: Rfc3339Timestamp


# ---------------------------------------------------------------------------
# EvaluationService
# ---------------------------------------------------------------------------


class EvaluationService:
    """Compare paired Qwen and GPT inference results.

    An evaluation job starts only after stored Qwen and GPT results exist
    for the same immutable package, task, prompt contract, and comparable
    input configuration. Stores structured field-level agreement and
    differences as described in Architecture V1.1 Section 11.3.
    """

    def evaluate_pair(
        self,
        *,
        qwen_inference: ModelInference,
        gpt_inference: ModelInference,
        comparison_contract_version: SchemaVersion,
    ) -> EvaluationResult:
        """Compare Qwen vs GPT inference outputs.

        Performs a full paired evaluation including structured field-level
        comparison, schema validity, abstention, retries, latency, usage,
        and cost differences.

        Args:
            qwen_inference: The primary Qwen inference result.
            gpt_inference: The shadow GPT inference result.
            comparison_contract_version: Version of the comparison contract.

        Returns:
            An ``EvaluationResult`` containing the paired comparison.
        """
        raise NotImplementedError

    def compute_disagreement(
        self,
        *,
        qwen_normalized_output: dict[str, object],
        gpt_normalized_output: dict[str, object],
        comparison_config: dict[str, object],
    ) -> tuple[FieldDelta, ...]:
        """Compute field-level deltas between Qwen and GPT normalized outputs.

        Compares action labels, objects, hands, QA issues, and boundary
        differences at the field level. A provider failure is also an
        evaluation outcome.

        Args:
            qwen_normalized_output: Normalized output from the Qwen inference.
            gpt_normalized_output: Normalized output from the GPT inference.
            comparison_config: Configuration for the comparison algorithm.

        Returns:
            A tuple of ``FieldDelta`` records, one per disagreeing field.
        """
        raise NotImplementedError

    def persist_disagreement(
        self,
        *,
        evaluation_result: EvaluationResult,
        shadow_reason: ShadowSelectionReason,
        mcap_id: OpaqueUuid,
        start_ns: Nanoseconds,
        end_ns: Nanoseconds,
        package_set_id: OpaqueUuid,
        camera_mapping_run_id: OpaqueUuid,
        alignment_id: OpaqueUuid,
    ) -> ModelDisagreementSample:
        """Store a ``ModelDisagreementSample`` record.

        Persists the evaluation result as an append-only disagreement
        sample. The inference pair plus comparison-contract version is
        unique. A provider failure is also an evaluation outcome, but it
        cannot alter the production Qwen result.

        Args:
            evaluation_result: The result of the paired evaluation.
            shadow_reason: Whether the sample was random, hard-case, or both.
            mcap_id: The MCAP recording identifier.
            start_ns: Start of the temporal window in nanoseconds.
            end_ns: End of the temporal window in nanoseconds.
            package_set_id: The package set identifier.
            camera_mapping_run_id: The camera mapping run identifier.
            alignment_id: The alignment identifier.

        Returns:
            The persisted ``ModelDisagreementSample``.
        """
        raise NotImplementedError


__all__ = [
    "EvaluationResult",
    "EvaluationService",
    "FieldDelta",
]
