"""Promotion gates for benchmark evaluation (Section 18.6).

Defines gate categories, individual gates, gate registries, and the
evaluator that checks whether benchmark results meet promotion criteria.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.benchmark.evidence import BenchmarkEvidenceContext
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class GateCategory(StrEnum):
    """Categories of promotion gates (Section 18.6).

    Each gate category corresponds to a specific aspect of the pipeline
    that must meet registered thresholds before promotion.
    """

    DATA_LINEAGE = "DATA_LINEAGE"
    ALIGNMENT = "ALIGNMENT"
    QA = "QA"
    EVENT_PROPOSAL = "EVENT_PROPOSAL"
    ACTION_BOUNDARY = "ACTION_BOUNDARY"
    STRUCTURED_OUTPUT = "STRUCTURED_OUTPUT"
    PRIMARY_REGRESSION = "PRIMARY_REGRESSION"
    SHADOW_ISOLATION = "SHADOW_ISOLATION"
    CAPACITY = "CAPACITY"
    COST = "COST"


class PromotionGate(StrictModel):
    """One promotion gate with threshold and failure action.

    Each gate row requires metric definition, numeric threshold/margin,
    denominator, required strata, data split, owner, effective date,
    failure action, and version.
    """

    gate_id: OpaqueUuid
    category: GateCategory
    metric_definition: NonEmptyString
    threshold: float = Field(strict=True, allow_inf_nan=False)
    margin: float = Field(strict=True, ge=0.0, allow_inf_nan=False)
    comparison: Literal["GTE", "LTE", "EQ"] = "GTE"
    denominator: NonEmptyString
    required_strata: tuple[NonEmptyString, ...]
    data_split: NonEmptyString
    owner: NonEmptyString
    effective_date: NonEmptyString
    failure_action: NonEmptyString
    version: NonEmptyString

    @model_validator(mode="after")
    def validate_gate(self) -> PromotionGate:
        if not self.required_strata:
            raise ValueError("promotion gates must declare required strata")
        if len(self.required_strata) != len(set(self.required_strata)):
            raise ValueError("required_strata must be unique")
        return self


class PromotionGateRegistry(StrictModel):
    """Frozen registry of promotion gates for a benchmark.

    A frozen-test or production promotion run is invalid when any
    required gate row is missing.
    """

    registry_id: OpaqueUuid
    gates: tuple[PromotionGate, ...]
    benchmark_id: OpaqueUuid
    evidence_context_digest: Sha256Digest
    frozen_at: datetime

    @model_validator(mode="after")
    def validate_registry(self) -> PromotionGateRegistry:
        gate_ids = tuple(gate.gate_id for gate in self.gates)
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("promotion gate IDs must be unique")
        if self.frozen_at.tzinfo is None or self.frozen_at.utcoffset() is None:
            raise ValueError("frozen_at must be timezone-aware")
        return self


class GateResult(StrictModel):
    """Result of evaluating one promotion gate."""

    category: GateCategory
    passed: bool
    evidence: dict[NonEmptyString, Any]
    threshold: float
    actual_value: float | None = None


class PromotionDecision(StrictModel):
    """Final promotion decision after evaluating all gates.

    Approved only when all required gates pass.
    """

    approved: bool
    rejected_gates: tuple[GateResult, ...]
    approved_gates: tuple[GateResult, ...]
    timestamp: datetime
    validation_errors: tuple[NonEmptyString, ...] = ()


class BenchmarkResults:
    """Benchmark values plus the identity/split/stratum evidence they came from."""

    def __init__(self, results: dict[str, Any]) -> None:
        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary")
        self._results = dict(results)
        self._evidence_context, self._evidence_context_error = self._parse_evidence_context()

    def _parse_evidence_context(
        self,
    ) -> tuple[BenchmarkEvidenceContext | None, str | None]:
        raw_context = self._results.get("evidence_context")
        if raw_context is None:
            return None, "MISSING_EVIDENCE_CONTEXT"
        try:
            context = BenchmarkEvidenceContext.model_validate(raw_context)
        except ValidationError:
            return None, "INVALID_EVIDENCE_CONTEXT"

        declared_digest = self._results.get("evidence_context_digest")
        declared_identity = self._results.get("evidence_context_identity")
        if declared_digest is None or declared_identity is None:
            return context, "MISSING_EVIDENCE_CONTEXT_BINDING"
        if (
            declared_digest != context.context_digest
            or declared_identity != context.context_identity
        ):
            return context, "EVIDENCE_CONTEXT_BINDING_MISMATCH"

        benchmark_id = self._results.get("benchmark_id")
        data_split = self._results.get("data_split")
        if benchmark_id != context.benchmark_id or data_split != context.data_split:
            return context, "EVIDENCE_CONTEXT_RESULT_MISMATCH"
        return context, None

    @staticmethod
    def _lookup(value: Any, key: str, default: Any = None) -> Any:
        for part in key.split("."):
            if isinstance(value, Mapping):
                if part not in value:
                    return default
                value = value[part]
            else:
                value = getattr(value, part, default)
                if value is default:
                    return default
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self._lookup(self._results, key, default)

    @property
    def benchmark_id(self) -> str | None:
        value = self._results.get("benchmark_id")
        return value if isinstance(value, str) else None

    @property
    def data_split(self) -> str | None:
        value = self._results.get("data_split")
        return value if isinstance(value, str) else None

    @property
    def evidence_context(self) -> BenchmarkEvidenceContext | None:
        return self._evidence_context

    @property
    def evidence_context_error(self) -> str | None:
        return self._evidence_context_error

    def stratum(self, name: str) -> Mapping[str, Any] | None:
        strata = self._results.get("strata")
        if not isinstance(strata, Mapping):
            return None
        value = strata.get(name)
        return value if isinstance(value, Mapping) else None

    @staticmethod
    def _measurement_binding(value: Any) -> tuple[Any, Any, Any]:
        if isinstance(value, Mapping):
            return (
                value.get("measurement_status"),
                value.get("evidence_context_digest"),
                value.get("evidence_context_identity"),
            )
        return (
            getattr(value, "measurement_status", None),
            getattr(value, "evidence_context_digest", None),
            getattr(value, "evidence_context_identity", None),
        )

    def measurement_error(self, value: Any) -> str | None:
        """Return the reason a metric is not governed measured evidence."""

        if self._results.get("measurement_status") != "MEASURED":
            return "NOT_MEASURED"
        if self.evidence_context_error is not None:
            return self.evidence_context_error
        context = self.evidence_context
        if context is None:
            return "MISSING_EVIDENCE_CONTEXT"

        status, digest, identity = self._measurement_binding(value)
        if status is not None:
            if status != "MEASURED":
                return "NOT_MEASURED"
            if digest != context.context_digest or identity != context.context_identity:
                return "METRIC_EVIDENCE_CONTEXT_MISMATCH"
        return None

    def measured(self, value: Any) -> bool:
        """Return whether a metric carries validated governed evidence."""

        return self.measurement_error(value) is None

    def stratum_measurement_error(
        self,
        stratum: Mapping[str, Any],
        value: Any,
    ) -> str | None:
        """Validate a stratum against the same governed evidence context."""

        metric_error = self.measurement_error(value)
        if metric_error is not None:
            return metric_error
        if stratum.get("measurement_status") != "MEASURED":
            return "STRATUM_NOT_MEASURED"
        context = self.evidence_context
        if context is None:
            return "MISSING_EVIDENCE_CONTEXT"
        if (
            stratum.get("evidence_context_digest") != context.context_digest
            or stratum.get("evidence_context_identity") != context.context_identity
        ):
            return "STRATUM_EVIDENCE_CONTEXT_MISMATCH"
        return None

    def measured_stratum(self, stratum: Mapping[str, Any], value: Any) -> bool:
        """Require the global run and stratum to share governed evidence."""

        return self.stratum_measurement_error(stratum, value) is None


class PromotionEvaluator:
    """Evaluate benchmark results against promotion gates.

    Checks each gate category and produces a PromotionDecision.
    """

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    def evaluate(
        self,
        results: BenchmarkResults,
        gates: PromotionGateRegistry,
    ) -> PromotionDecision:
        """Evaluate all gates and return a promotion decision.

        Args:
            results: Benchmark results to evaluate.
            gates: Frozen registry of promotion gates.

        Returns:
            PromotionDecision with approved/rejected status and evidence.
        """
        validation_errors: list[str] = []
        if results.benchmark_id is None:
            validation_errors.append("MISSING_BENCHMARK_ID")
        elif results.benchmark_id != gates.benchmark_id:
            validation_errors.append("BENCHMARK_ID_MISMATCH")
        if results.evidence_context_error is not None:
            validation_errors.append(results.evidence_context_error)
        elif (
            results.evidence_context is None
            or results.evidence_context.context_digest != gates.evidence_context_digest
        ):
            validation_errors.append("EVIDENCE_CONTEXT_REGISTRY_MISMATCH")

        rejected: list[GateResult] = []
        approved: list[GateResult] = []

        present_categories = {gate.category for gate in gates.gates}
        for category in GateCategory:
            if category not in present_categories:
                rejected.append(self._missing_gate_result(category))

        for gate in gates.gates:
            result = self._check_gate(gate, results)
            if result.passed:
                approved.append(result)
            else:
                rejected.append(result)

        return PromotionDecision(
            approved=bool(gates.gates) and not rejected and not validation_errors,
            rejected_gates=tuple(rejected),
            approved_gates=tuple(approved),
            timestamp=self._clock(),
            validation_errors=tuple(validation_errors),
        )

    def _check_gate(self, gate: PromotionGate, results: BenchmarkResults) -> GateResult:
        """Check a single gate against results.

        Args:
            gate: The promotion gate to check.
            results: Benchmark results.

        Returns:
            GateResult with pass/fail status.
        """
        checkers = {
            GateCategory.DATA_LINEAGE: self.check_data_lineage,
            GateCategory.ALIGNMENT: self.check_alignment,
            GateCategory.QA: self.check_qa,
            GateCategory.EVENT_PROPOSAL: self.check_event_proposal,
            GateCategory.ACTION_BOUNDARY: self.check_action_boundary,
            GateCategory.STRUCTURED_OUTPUT: self.check_structured_output,
            GateCategory.PRIMARY_REGRESSION: self.check_primary_regression,
            GateCategory.SHADOW_ISOLATION: self.check_shadow_isolation,
            GateCategory.CAPACITY: self.check_capacity,
            GateCategory.COST: self.check_cost,
        }
        checker = checkers.get(gate.category)
        if checker is None:
            return self._default_check(results, gate)
        return checker(results, gate)

    def _missing_gate_result(self, category: GateCategory) -> GateResult:
        return GateResult(
            category=category,
            passed=False,
            evidence={"reason": "MISSING_REQUIRED_GATE"},
            threshold=0.0,
            actual_value=None,
        )

    def _default_check(
        self,
        results: BenchmarkResults,
        gate: PromotionGate,
    ) -> GateResult:
        _ = results
        return GateResult(
            category=gate.category,
            passed=False,
            evidence={"reason": "UNSUPPORTED_GATE_CATEGORY"},
            threshold=gate.threshold,
            actual_value=None,
        )

    @staticmethod
    def _actual_value(value: Any) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Mapping) and isinstance(value.get("value"), (int, float)):
            return float(value["value"])
        candidate = getattr(value, "value", None)
        if isinstance(candidate, (int, float)):
            return float(candidate)
        return None

    def _evaluate_registered_gate(
        self,
        category: GateCategory,
        results: BenchmarkResults,
        gate: PromotionGate | None,
    ) -> GateResult:
        if gate is None:
            return self._missing_gate_result(category)
        sentinel = object()
        evidence_value = results.get(gate.metric_definition, sentinel)
        common_evidence: dict[str, Any] = {
            "gate_id": gate.gate_id,
            "metric_definition": gate.metric_definition,
            "comparison": gate.comparison,
            "margin": gate.margin,
            "denominator": gate.denominator,
            "required_strata": list(gate.required_strata),
            "data_split": gate.data_split,
        }
        if evidence_value is sentinel:
            return GateResult(
                category=category,
                passed=False,
                evidence={**common_evidence, "reason": "MISSING_METRIC"},
                threshold=gate.threshold,
                actual_value=None,
            )
        measurement_error = results.measurement_error(evidence_value)
        if measurement_error is not None:
            return GateResult(
                category=category,
                passed=False,
                evidence={**common_evidence, "reason": measurement_error},
                threshold=gate.threshold,
                actual_value=None,
            )
        context = results.evidence_context
        if context is None:
            return GateResult(
                category=category,
                passed=False,
                evidence={**common_evidence, "reason": "MISSING_EVIDENCE_CONTEXT"},
                threshold=gate.threshold,
                actual_value=None,
            )
        if results.data_split != gate.data_split:
            return GateResult(
                category=category,
                passed=False,
                evidence={
                    **common_evidence,
                    "reason": "DATA_SPLIT_MISMATCH",
                    "actual_data_split": results.data_split,
                },
                threshold=gate.threshold,
                actual_value=None,
            )
        actual = self._actual_value(evidence_value)
        if actual is None:
            return GateResult(
                category=category,
                passed=False,
                evidence={**common_evidence, "reason": "NON_NUMERIC_METRIC"},
                threshold=gate.threshold,
                actual_value=None,
            )

        stratum_values: dict[str, float] = {}
        for stratum_name in gate.required_strata:
            stratum = results.stratum(stratum_name)
            if stratum is None:
                return GateResult(
                    category=category,
                    passed=False,
                    evidence={
                        **common_evidence,
                        "reason": "MISSING_REQUIRED_STRATUM",
                        "missing_stratum": stratum_name,
                    },
                    threshold=gate.threshold,
                    actual_value=None,
                )
            sentinel = object()
            stratum_evidence = BenchmarkResults._lookup(
                stratum,
                gate.metric_definition,
                sentinel,
            )
            if stratum_evidence is sentinel:
                return GateResult(
                    category=category,
                    passed=False,
                    evidence={
                        **common_evidence,
                        "reason": "MISSING_STRATUM_METRIC",
                        "missing_stratum": stratum_name,
                    },
                    threshold=gate.threshold,
                    actual_value=None,
                )
            stratum_error = results.stratum_measurement_error(stratum, stratum_evidence)
            if stratum_error is not None:
                return GateResult(
                    category=category,
                    passed=False,
                    evidence={
                        **common_evidence,
                        "reason": "STRATUM_NOT_MEASURED",
                        "unmeasured_stratum": stratum_name,
                        "evidence_error": stratum_error,
                    },
                    threshold=gate.threshold,
                    actual_value=None,
                )
            stratum_actual = self._actual_value(stratum_evidence)
            if stratum_actual is None:
                return GateResult(
                    category=category,
                    passed=False,
                    evidence={
                        **common_evidence,
                        "reason": "NON_NUMERIC_STRATUM_METRIC",
                        "invalid_stratum": stratum_name,
                    },
                    threshold=gate.threshold,
                    actual_value=None,
                )
            stratum_values[stratum_name] = stratum_actual

        values = [actual, *stratum_values.values()]
        if gate.comparison == "GTE":
            passed = all(value >= gate.threshold - gate.margin for value in values)
            reported_actual = min(values)
        elif gate.comparison == "LTE":
            passed = all(value <= gate.threshold + gate.margin for value in values)
            reported_actual = max(values)
        else:
            passed = all(abs(value - gate.threshold) <= gate.margin for value in values)
            reported_actual = max(values, key=lambda value: abs(value - gate.threshold))
        return GateResult(
            category=category,
            passed=passed,
            evidence={
                **common_evidence,
                "reason": "THRESHOLD_MET" if passed else "THRESHOLD_NOT_MET",
                "measurement_status": "MEASURED",
                "evidence_context_digest": context.context_digest,
                "evidence_context_identity": context.context_identity,
                "stratum_values": stratum_values,
            },
            threshold=gate.threshold,
            actual_value=reported_actual,
        )

    def check_data_lineage(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check data lineage gate.

        Every accepted MCAP has six mapped cameras; every published QA/event/inference
        traces to exact source, camera, timestamps, package, and versions.
        """
        return self._evaluate_registered_gate(GateCategory.DATA_LINEAGE, results, gate)

    def check_alignment(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check alignment gate.

        Declared p95/p99 residual/skew tolerance met on target rigs and clock modes.
        """
        return self._evaluate_registered_gate(GateCategory.ALIGNMENT, results, gate)

    def check_qa(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check QA gate.

        Critical issue recall, per-class/macro quality, recording false accept/reject,
        and calibration meet registered thresholds.
        """
        return self._evaluate_registered_gate(GateCategory.QA, results, gate)

    def check_event_proposal(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check event proposal gate.

        Recall at registered temporal IoU/tolerance meets target before dense
        workload is optimized.
        """
        return self._evaluate_registered_gate(GateCategory.EVENT_PROPOSAL, results, gate)

    def check_action_boundary(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check action/boundary gate.

        Action/object/hand quality, temporal IoU, and boundary error meet targets
        by relevant strata.
        """
        return self._evaluate_registered_gate(GateCategory.ACTION_BOUNDARY, results, gate)

    def check_structured_output(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check structured output gate.

        Valid/repair/abstention rates meet the registered operational budget.
        """
        return self._evaluate_registered_gate(GateCategory.STRUCTURED_OUTPUT, results, gate)

    def check_primary_regression(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check primary regression gate.

        New Qwen/model/prompt/sampling/fusion policy does not exceed the registered
        quality-regression margin.
        """
        return self._evaluate_registered_gate(GateCategory.PRIMARY_REGRESSION, results, gate)

    def check_shadow_isolation(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check shadow isolation gate.

        GPT saturation/failure does not materially change Qwen critical-path latency,
        success, or deadline compliance.
        """
        return self._evaluate_registered_gate(GateCategory.SHADOW_ISOLATION, results, gate)

    def check_capacity(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check capacity gate.

        Sustained measured capacity, backlog drain, deadline compliance, and headroom
        pass both workload interpretations.
        """
        return self._evaluate_registered_gate(GateCategory.CAPACITY, results, gate)

    def check_cost(
        self,
        results: BenchmarkResults,
        gate: PromotionGate | None = None,
    ) -> GateResult:
        """Check cost gate.

        Cost per recording hour, camera-video hour, package, and event stays under
        an approved budget.
        """
        return self._evaluate_registered_gate(GateCategory.COST, results, gate)


__all__ = [
    "BenchmarkResults",
    "GateCategory",
    "GateResult",
    "PromotionDecision",
    "PromotionEvaluator",
    "PromotionGate",
    "PromotionGateRegistry",
]
