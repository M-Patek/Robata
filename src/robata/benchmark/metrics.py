"""Metrics calculation for benchmark evaluation (Section 18.3).

Implements calculators for QA, event proposal, boundary refinement,
and calibration metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, isnan
from statistics import mean, median
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.benchmark.evidence import BenchmarkEvidenceContext, EvidenceContextIdentity
from robata.contracts.common import INT64_MAX, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
MeasurementStatus = Literal["NOT_MEASURED", "MEASURED"]
MetricPolicyIdentity = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^benchmark-metric-policy:[0-9a-f]{64}$",
    ),
]
NonNegativeNanoseconds = Annotated[int, Field(strict=True, ge=0, le=INT64_MAX)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


def benchmark_metric_policy_projection(
    *,
    policy_version: str,
    critical_issue_codes: tuple[str, ...],
    event_iou_thresholds: tuple[float, ...],
    event_start_end_tolerance_ns: int,
    boundary_tolerance_ns: int,
    calibration_bin_count: int,
    governance_approval_id: str | None,
    governance_approval_digest: str | None,
    governance_policy_version: str | None,
) -> dict[str, object]:
    """Return the complete semantic preimage for a metric-definition policy."""

    return {
        "domain": "robata.benchmark-metric-policy",
        "schema_version": "1.0",
        "policy_version": policy_version,
        "critical_issue_codes": critical_issue_codes,
        "event_iou_thresholds": event_iou_thresholds,
        "event_start_end_tolerance_ns": str(event_start_end_tolerance_ns),
        "boundary_tolerance_ns": str(boundary_tolerance_ns),
        "calibration_bin_count": calibration_bin_count,
        "governance_approval_id": governance_approval_id,
        "governance_approval_digest": governance_approval_digest,
        "governance_policy_version": governance_policy_version,
    }


class BenchmarkMetricPolicy(StrictModel):
    """Versioned definitions used by every benchmark metric calculation.

    A policy can be useful for local engineering without an approval binding. A
    calculation may claim MEASURED only when all three approval fields match
    the supplied BenchmarkEvidenceContext.
    """

    schema_version: Literal["1.0"]
    policy_identity: MetricPolicyIdentity
    policy_digest: Sha256Digest
    policy_version: SchemaVersion
    critical_issue_codes: tuple[NonEmptyString, ...] = Field(min_length=1)
    event_iou_thresholds: tuple[UnitInterval, ...] = Field(min_length=1)
    event_start_end_tolerance_ns: NonNegativeNanoseconds
    boundary_tolerance_ns: NonNegativeNanoseconds
    calibration_bin_count: PositiveInt
    governance_approval_id: NonEmptyString | None = None
    governance_approval_digest: Sha256Digest | None = None
    governance_policy_version: SchemaVersion | None = None

    @classmethod
    def create(
        cls,
        *,
        policy_version: SchemaVersion,
        critical_issue_codes: tuple[NonEmptyString, ...],
        event_iou_thresholds: tuple[UnitInterval, ...],
        event_start_end_tolerance_ns: NonNegativeNanoseconds,
        boundary_tolerance_ns: NonNegativeNanoseconds,
        calibration_bin_count: PositiveInt,
        governance_approval_id: NonEmptyString | None = None,
        governance_approval_digest: Sha256Digest | None = None,
        governance_policy_version: SchemaVersion | None = None,
    ) -> Self:
        """Build a policy and derive its content-addressed identity."""

        projection = benchmark_metric_policy_projection(
            policy_version=policy_version,
            critical_issue_codes=critical_issue_codes,
            event_iou_thresholds=event_iou_thresholds,
            event_start_end_tolerance_ns=event_start_end_tolerance_ns,
            boundary_tolerance_ns=boundary_tolerance_ns,
            calibration_bin_count=calibration_bin_count,
            governance_approval_id=governance_approval_id,
            governance_approval_digest=governance_approval_digest,
            governance_policy_version=governance_policy_version,
        )
        digest = semantic_sha256(projection)
        return cls(
            schema_version="1.0",
            policy_identity=f"benchmark-metric-policy:{digest}",
            policy_digest=digest,
            policy_version=policy_version,
            critical_issue_codes=critical_issue_codes,
            event_iou_thresholds=event_iou_thresholds,
            event_start_end_tolerance_ns=event_start_end_tolerance_ns,
            boundary_tolerance_ns=boundary_tolerance_ns,
            calibration_bin_count=calibration_bin_count,
            governance_approval_id=governance_approval_id,
            governance_approval_digest=governance_approval_digest,
            governance_policy_version=governance_policy_version,
        )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if tuple(sorted(set(self.critical_issue_codes))) != self.critical_issue_codes:
            raise ValueError("critical_issue_codes must be unique and sorted")
        if any(threshold <= 0.0 for threshold in self.event_iou_thresholds):
            raise ValueError("event_iou_thresholds must lie in (0, 1]")
        if tuple(sorted(set(self.event_iou_thresholds))) != self.event_iou_thresholds:
            raise ValueError("event_iou_thresholds must be unique and strictly increasing")
        approval_fields = (
            self.governance_approval_id,
            self.governance_approval_digest,
            self.governance_policy_version,
        )
        if any(value is not None for value in approval_fields) and not all(
            value is not None for value in approval_fields
        ):
            raise ValueError(
                "metric policy governance approval fields must be all present or absent"
            )
        projection = benchmark_metric_policy_projection(
            policy_version=self.policy_version,
            critical_issue_codes=self.critical_issue_codes,
            event_iou_thresholds=self.event_iou_thresholds,
            event_start_end_tolerance_ns=self.event_start_end_tolerance_ns,
            boundary_tolerance_ns=self.boundary_tolerance_ns,
            calibration_bin_count=self.calibration_bin_count,
            governance_approval_id=self.governance_approval_id,
            governance_approval_digest=self.governance_approval_digest,
            governance_policy_version=self.governance_policy_version,
        )
        expected_digest = semantic_sha256(projection)
        if self.policy_digest != expected_digest:
            raise ValueError("policy_digest does not match the metric policy")
        if self.policy_identity != f"benchmark-metric-policy:{expected_digest}":
            raise ValueError("policy_identity does not match policy_digest")
        return self


class EvidenceBoundMetrics(StrictModel):
    """Metric values whose measurement claim is bound to governed evidence."""

    measurement_status: MeasurementStatus = "NOT_MEASURED"
    evidence_context_digest: Sha256Digest | None = None
    evidence_context_identity: EvidenceContextIdentity | None = None
    metric_policy_identity: MetricPolicyIdentity
    metric_policy_digest: Sha256Digest
    metric_policy_version: SchemaVersion

    @model_validator(mode="after")
    def validate_measurement_binding(self) -> Self:
        has_digest = self.evidence_context_digest is not None
        has_identity = self.evidence_context_identity is not None
        if self.measurement_status == "MEASURED":
            if not has_digest or not has_identity:
                raise ValueError("MEASURED metrics require an evidence context binding")
            expected_identity = f"benchmark-evidence:{self.evidence_context_digest}"
            if self.evidence_context_identity != expected_identity:
                raise ValueError("evidence context identity does not match its digest")
        elif has_digest or has_identity:
            raise ValueError("NOT_MEASURED metrics cannot claim an evidence context binding")
        if self.metric_policy_identity != f"benchmark-metric-policy:{self.metric_policy_digest}":
            raise ValueError("metric policy identity does not match its digest")
        return self


def _measurement_binding(
    context: BenchmarkEvidenceContext | None,
    policy: BenchmarkMetricPolicy,
) -> tuple[MeasurementStatus, str | None, str | None]:
    if context is None:
        return "NOT_MEASURED", None, None
    expected_approval = (
        context.governance_approval_id,
        context.governance_approval_digest,
        context.governance_policy_version,
    )
    actual_approval = (
        policy.governance_approval_id,
        policy.governance_approval_digest,
        policy.governance_policy_version,
    )
    if actual_approval != expected_approval:
        raise ValueError(
            "MEASURED metrics require a metric policy bound to the evidence governance approval"
        )
    return "MEASURED", context.context_digest, context.context_identity


class QAMetrics(EvidenceBoundMetrics):
    """Quality assurance metrics for one benchmark run.

    Covers per-issue precision/recall/F1, macro/micro aggregates,
    critical issue recall, temporal IoU, and recording-level measures.
    """

    per_issue_precision: dict[NonEmptyString, float]
    per_issue_recall: dict[NonEmptyString, float]
    per_issue_f1: dict[NonEmptyString, float]
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    critical_issue_recall: float
    temporal_iou: float
    recording_precision: float
    recording_recall: float
    false_accept_rate: float
    false_reject_rate: float
    sample_count: int = Field(default=0, ge=0, strict=True)


class EventMetrics(EvidenceBoundMetrics):
    """Event proposal metrics for one benchmark run.

    Covers recall at policy-defined IoU thresholds, average recall,
    start/end hit rate, miss rate, and false candidate rates.
    """

    recall_at_iou: dict[str, float]  # policy-defined decimal threshold keys
    average_recall: float
    start_end_hit_rate: float
    miss_rate_by_class: dict[NonEmptyString, float]
    false_candidates_per_hour: float
    duplicate_rate: float
    overmerge_rate: float
    oversplit_rate: float
    sample_count: int = Field(default=0, ge=0, strict=True)


class BoundaryMetrics(EvidenceBoundMetrics):
    """Boundary refinement metrics for one benchmark run.

    Covers start/end MAE, median/p95 error, temporal IoU,
    within-tolerance rate, and classification accuracy.
    """

    start_mae: NonNegativeFloat
    end_mae: NonNegativeFloat
    median_error: NonNegativeFloat
    p95_error: NonNegativeFloat
    temporal_iou: float
    within_tolerance_rate: float
    classification_accuracy: float
    object_accuracy: float
    hand_accuracy: float
    sample_count: int = Field(default=0, ge=0, strict=True)


class CalibrationMetrics(EvidenceBoundMetrics):
    """Calibration metrics for one benchmark run.

    Expected calibration error, Brier score, abstention rate, and unknown rate.
    """

    ece: float
    brier_score: float
    abstention_rate: float
    unknown_rate: float
    sample_count: int = Field(default=0, ge=0, strict=True)


def _records(value: Any, name: str) -> list[Any]:
    """Normalize benchmark inputs without imposing a storage dependency."""

    if isinstance(value, Mapping):
        for key in ("records", "items", "predictions", "proposals", "annotations"):
            if key in value:
                return _records(value[key], name)
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = list(value)
        if not result:
            raise ValueError(f"{name} must contain at least one record")
        return result
    raise TypeError(f"{name} must be a mapping or a non-empty sequence")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _interval(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    start = _field(value, "start_ns")
    end = _field(value, "end_ns")
    if start is None or end is None:
        if isinstance(value, Sequence) and len(value) == 2:
            start, end = value
        else:
            return None
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise ValueError("benchmark intervals require exact integer nanoseconds")
    if not start < end:
        return None
    return start, end


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def _code(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(_field(value, "code", _field(value, "label", "UNKNOWN")))


def _event_class(value: Any) -> str | None:
    """Return the explicit event class used for one-to-one matching."""

    for field_name in ("label", "label_hint", "action_type", "action_type_id", "verb"):
        field_value = _field(value, field_name)
        if field_value is not None:
            return str(field_value)
    return None


def _stable_record_key(value: Any) -> str:
    """Return an order-independent tie-break key for an immutable record."""

    try:
        return canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError):
        return repr(value)


def _record_id(value: Any, index: int) -> str:
    return str(_field(value, "recording_id", _field(value, "mcap_id", index)))


def _explicit_record_id(value: Any) -> str | None:
    record_id = _field(value, "recording_id", _field(value, "mcap_id"))
    return str(record_id) if record_id is not None else None


def _physical_event_key(value: Any) -> tuple[str, str]:
    record_id = _explicit_record_id(value)
    if record_id is None:
        raise ValueError("boundary records require recording_id or mcap_id")
    event_id = None
    for field_name in ("action_id", "event_id", "physical_event_id", "action_event_id"):
        event_id = _field(value, field_name)
        if event_id is not None:
            break
    if event_id is None:
        raise ValueError("boundary records require an explicit physical-event identity")
    return record_id, str(event_id)


def _issues(value: Any) -> list[Any]:
    issues = _field(value, "issues")
    if issues is None and _field(value, "code") is not None:
        return [value]
    if issues is None:
        return []
    return list(issues)


def _event_interval(value: Any) -> tuple[int, int] | None:
    return _interval(_field(value, "interval", value))


def _metric(value: float) -> float:
    if isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _boundary_hit(prediction: Any, target: Any, tolerance_ns: int) -> bool:
    predicted_interval = _event_interval(prediction)
    target_interval = _event_interval(target)
    if predicted_interval is None or target_interval is None:
        return False
    return (
        abs(predicted_interval[0] - target_interval[0]) <= tolerance_ns
        and abs(predicted_interval[1] - target_interval[1]) <= tolerance_ns
    )


class MetricsCalculator:
    """Calculate benchmark metrics from predictions and ground truth.

    The calculator accepts immutable contract models or plain mappings. It
    deliberately rejects an empty cohort so an absent benchmark cannot look
    like a measured all-zero result. Every result is bound to an explicit,
    content-addressed metric-definition policy. Values calculated without a
    governed evidence context remain useful locally but are explicitly not
    measured.
    """

    def __init__(self, policy: BenchmarkMetricPolicy) -> None:
        if not isinstance(policy, BenchmarkMetricPolicy):
            raise TypeError("policy must be a BenchmarkMetricPolicy")
        self._policy = policy

    @property
    def policy(self) -> BenchmarkMetricPolicy:
        return self._policy

    def calculate_qa_metrics(
        self,
        predictions: Any,
        ground_truth: Any,
        *,
        evidence_context: BenchmarkEvidenceContext | None = None,
    ) -> QAMetrics:
        """Calculate QA metrics from predictions and ground truth.

        Args:
            predictions: Model QA predictions.
            ground_truth: Ground-truth QA annotations.
            evidence_context: Frozen governed manifests and approval identity.

        Returns:
            QAMetrics populated with per-issue and aggregate measures.
        """
        predicted = _records(predictions, "predictions")
        truth = _records(ground_truth, "ground_truth")
        pred_by_record: dict[str, list[Any]] = {
            _record_id(record, index): _issues(record) for index, record in enumerate(predicted)
        }
        truth_by_record: dict[str, list[Any]] = {
            _record_id(record, index): _issues(record) for index, record in enumerate(truth)
        }
        codes = sorted(
            {_code(issue) for issues in pred_by_record.values() for issue in issues}
            | {_code(issue) for issues in truth_by_record.values() for issue in issues}
        )
        precision: dict[str, float] = {}
        recall: dict[str, float] = {}
        f1: dict[str, float] = {}
        total_tp = total_fp = total_fn = 0
        matched_ious: list[float] = []
        for code in codes:
            tp = fp = fn = 0
            for record_id in sorted(set(pred_by_record) | set(truth_by_record)):
                pred_items = [
                    item for item in pred_by_record.get(record_id, ()) if _code(item) == code
                ]
                true_items = [
                    item for item in truth_by_record.get(record_id, ()) if _code(item) == code
                ]
                used: set[int] = set()
                for item in pred_items:
                    item_interval = _event_interval(item)
                    candidates: list[tuple[int, float]] = []
                    for index, other in enumerate(true_items):
                        if index in used:
                            continue
                        other_interval = _event_interval(other)
                        overlap = (
                            1.0
                            if item_interval is None or other_interval is None
                            else _iou(item_interval, other_interval)
                        )
                        candidates.append((index, overlap))
                    if candidates:
                        best_index, best_iou = max(
                            candidates,
                            key=lambda pair: (pair[1], -pair[0]),
                        )
                        if best_iou > 0.0:
                            used.add(best_index)
                            tp += 1
                            matched_ious.append(best_iou)
                        else:
                            fp += 1
                    else:
                        fp += 1
                fn += len(true_items) - len(used)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            precision[code] = tp / (tp + fp) if tp + fp else 0.0
            recall[code] = tp / (tp + fn) if tp + fn else 0.0
            f1[code] = (
                2 * precision[code] * recall[code] / (precision[code] + recall[code])
                if precision[code] + recall[code]
                else 0.0
            )
        macro_f1 = mean(f1.values()) if f1 else 0.0
        micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        critical = set(self._policy.critical_issue_codes)
        critical_tp = critical_fn = 0
        for record_id in sorted(set(pred_by_record) | set(truth_by_record)):
            pred_codes = {_code(item) for item in pred_by_record.get(record_id, ())}
            true_codes = {_code(item) for item in truth_by_record.get(record_id, ())}
            critical_tp += len((pred_codes & true_codes) & critical)
            critical_fn += len(true_codes & critical - pred_codes)
        true_recordings = {key for key, items in truth_by_record.items() if items}
        pred_recordings = {key for key, items in pred_by_record.items() if items}
        recording_tp = len(true_recordings & pred_recordings)
        recording_precision = recording_tp / len(pred_recordings) if pred_recordings else 0.0
        recording_recall = recording_tp / len(true_recordings) if true_recordings else 0.0
        negative_count = max(
            1,
            len(set(pred_by_record) | set(truth_by_record)) - len(true_recordings),
        )
        false_accept_rate = len(pred_recordings - true_recordings) / negative_count
        false_reject_rate = len(true_recordings - pred_recordings) / max(1, len(true_recordings))
        measurement_status, context_digest, context_identity = _measurement_binding(
            evidence_context, self._policy
        )
        return QAMetrics(
            per_issue_precision=precision,
            per_issue_recall=recall,
            per_issue_f1=f1,
            macro_f1=_metric(macro_f1),
            micro_precision=_metric(micro_precision),
            micro_recall=_metric(micro_recall),
            micro_f1=_metric(micro_f1),
            critical_issue_recall=_metric(
                critical_tp / (critical_tp + critical_fn) if critical_tp + critical_fn else 0.0
            ),
            temporal_iou=_metric(mean(matched_ious) if matched_ious else 0.0),
            recording_precision=_metric(recording_precision),
            recording_recall=_metric(recording_recall),
            false_accept_rate=_metric(false_accept_rate),
            false_reject_rate=_metric(false_reject_rate),
            sample_count=len(truth),
            measurement_status=measurement_status,
            evidence_context_digest=context_digest,
            evidence_context_identity=context_identity,
            metric_policy_identity=self._policy.policy_identity,
            metric_policy_digest=self._policy.policy_digest,
            metric_policy_version=self._policy.policy_version,
        )

    def calculate_event_metrics(
        self,
        proposals: Any,
        ground_truth: Any,
        *,
        evidence_context: BenchmarkEvidenceContext | None = None,
    ) -> EventMetrics:
        """Calculate event proposal metrics from proposals and ground truth.

        Args:
            proposals: Model event proposals.
            ground_truth: Ground-truth event annotations.
            evidence_context: Frozen governed manifests and approval identity.

        Returns:
            EventMetrics populated with policy-defined recall and error rates.
        """
        predicted = _records(proposals, "proposals")
        truth = _records(ground_truth, "ground_truth")
        predicted_record_ids = tuple(_explicit_record_id(item) for item in predicted)
        truth_record_ids = tuple(_explicit_record_id(item) for item in truth)
        if any(record_id is None for record_id in (*predicted_record_ids, *truth_record_ids)):
            raise ValueError("event metrics require recording_id or mcap_id on every record")
        edges: list[tuple[int, int, float]] = []
        for proposal_index, proposal in enumerate(predicted):
            proposal_interval = _event_interval(proposal)
            if proposal_interval is None:
                continue
            for truth_index, target in enumerate(truth):
                if predicted_record_ids[proposal_index] != truth_record_ids[truth_index]:
                    continue
                if _event_class(proposal) != _event_class(target):
                    continue
                target_interval = _event_interval(target)
                if target_interval is None:
                    continue
                overlap = _iou(proposal_interval, target_interval)
                if overlap > 0.0:
                    edges.append((proposal_index, truth_index, overlap))

        prediction_keys = tuple(_stable_record_key(item) for item in predicted)
        truth_keys = tuple(_stable_record_key(item) for item in truth)
        # Global greedy assignment is deterministic for a frozen manifest and
        # uses semantic record content before the positional fallback.
        used_predictions: set[int] = set()
        used_truth: set[int] = set()
        assignments: list[tuple[int, int, float]] = []
        for proposal_index, truth_index, overlap in sorted(
            edges,
            key=lambda edge: (
                -edge[2],
                prediction_keys[edge[0]],
                truth_keys[edge[1]],
                edge[0],
                edge[1],
            ),
        ):
            if proposal_index in used_predictions or truth_index in used_truth:
                continue
            used_predictions.add(proposal_index)
            used_truth.add(truth_index)
            assignments.append((proposal_index, truth_index, overlap))

        assignments.sort(key=lambda edge: edge[0])
        pairs = [
            (predicted[proposal_index], truth[truth_index], overlap)
            for proposal_index, truth_index, overlap in assignments
        ]
        thresholds = self._policy.event_iou_thresholds
        recall_at_iou = {
            str(threshold): _metric(
                sum(overlap >= threshold for _, _, overlap in pairs) / len(truth) if truth else 0.0
            )
            for threshold in thresholds
        }
        classes = sorted(
            event_class
            for event_class in {_event_class(item) for item in truth}
            if event_class is not None
        )
        misses: dict[str, float] = {}
        for label in classes:
            class_indices = [
                index for index, item in enumerate(truth) if _event_class(item) == label
            ]
            misses[label] = _metric(
                sum(index not in used_truth for index in class_indices) / max(1, len(class_indices))
            )
        durations_by_record: dict[str, float] = {}
        for index, item in enumerate(truth):
            raw_duration = _field(item, "duration_ns")
            if raw_duration is None:
                continue
            duration = float(raw_duration)
            if not isfinite(duration) or duration <= 0:
                raise ValueError("event duration_ns values must be positive finite numbers")
            record_id = truth_record_ids[index]
            if record_id is None:
                raise ValueError("event metrics require recording_id or mcap_id")
            previous_duration = durations_by_record.get(record_id)
            if previous_duration is not None and previous_duration != duration:
                raise ValueError("one recording cannot have conflicting duration_ns values")
            durations_by_record[record_id] = duration
        durations = list(durations_by_record.values())
        if any(duration <= 0 for duration in durations):
            raise ValueError("event duration_ns values must be positive")
        false_candidate_count = len(predicted) - len(pairs)
        if false_candidate_count and not durations:
            raise ValueError(
                "ground truth must provide duration_ns when false candidates are present"
            )
        hours = sum(durations) / 3_600_000_000_000 if durations else 1.0
        unmatched_with_overlap = {
            proposal_index
            for proposal_index, _, _ in edges
            if proposal_index not in used_predictions
        }
        duplicate_count = len(unmatched_with_overlap)
        overmerge_extra = sum(
            max(0, sum(edge[0] == proposal_index for edge in edges) - 1)
            for proposal_index in range(len(predicted))
        )
        oversplit_extra = sum(
            max(0, sum(edge[1] == truth_index for edge in edges) - 1)
            for truth_index in range(len(truth))
        )
        measurement_status, context_digest, context_identity = _measurement_binding(
            evidence_context, self._policy
        )
        return EventMetrics(
            recall_at_iou=recall_at_iou,
            average_recall=_metric(mean(recall_at_iou.values())),
            start_end_hit_rate=_metric(
                sum(
                    _boundary_hit(
                        prediction,
                        target,
                        self._policy.event_start_end_tolerance_ns,
                    )
                    for prediction, target, _ in pairs
                )
                / max(1, len(truth))
            ),
            miss_rate_by_class=misses,
            false_candidates_per_hour=false_candidate_count / hours,
            duplicate_rate=_metric(duplicate_count / max(1, len(predicted))),
            overmerge_rate=_metric(overmerge_extra / max(1, len(truth))),
            oversplit_rate=_metric(oversplit_extra / max(1, len(truth))),
            sample_count=len(truth),
            measurement_status=measurement_status,
            evidence_context_digest=context_digest,
            evidence_context_identity=context_identity,
            metric_policy_identity=self._policy.policy_identity,
            metric_policy_digest=self._policy.policy_digest,
            metric_policy_version=self._policy.policy_version,
        )

    def calculate_boundary_metrics(
        self,
        refined: Any,
        ground_truth: Any,
        *,
        evidence_context: BenchmarkEvidenceContext | None = None,
    ) -> BoundaryMetrics:
        """Calculate boundary refinement metrics from refined predictions and ground truth.

        Args:
            refined: Refined boundary predictions.
            ground_truth: Ground-truth boundary annotations.
            evidence_context: Frozen governed manifests and approval identity.

        Returns:
            BoundaryMetrics populated with MAE, IoU, and accuracy measures.
        """
        predicted = _records(refined, "refined")
        truth = _records(ground_truth, "ground_truth")
        predicted_by_event = {_physical_event_key(item): item for item in predicted}
        truth_by_event = {_physical_event_key(item): item for item in truth}
        if len(predicted_by_event) != len(predicted) or len(truth_by_event) != len(truth):
            raise ValueError("boundary inputs cannot contain duplicate physical-event identities")
        if predicted_by_event.keys() != truth_by_event.keys():
            raise ValueError("refined and ground_truth must contain the same physical events")
        pairs = [
            (predicted_by_event[key], truth_by_event[key]) for key in sorted(predicted_by_event)
        ]
        start_errors: list[float] = []
        end_errors: list[float] = []
        overlaps: list[float] = []
        class_hits = object_hits = hand_hits = 0
        for prediction, target in pairs:
            pred_interval = _event_interval(prediction)
            target_interval = _event_interval(target)
            if pred_interval is None or target_interval is None:
                continue
            start_errors.append(abs(pred_interval[0] - target_interval[0]))
            end_errors.append(abs(pred_interval[1] - target_interval[1]))
            overlaps.append(_iou(pred_interval, target_interval))
            class_hits += _code(prediction) == _code(target)
            object_hits += _field(prediction, "object") == _field(target, "object")
            hand_hits += _field(prediction, "hand") == _field(target, "hand")
        errors = start_errors + end_errors
        if not errors:
            raise ValueError("boundary records must contain valid intervals")
        measurement_status, context_digest, context_identity = _measurement_binding(
            evidence_context, self._policy
        )
        return BoundaryMetrics(
            start_mae=mean(start_errors),
            end_mae=mean(end_errors),
            median_error=median(errors),
            p95_error=sorted(errors)[min(len(errors) - 1, int(len(errors) * 0.95))],
            temporal_iou=mean(overlaps),
            within_tolerance_rate=sum(
                error <= self._policy.boundary_tolerance_ns for error in errors
            )
            / len(errors),
            classification_accuracy=class_hits / len(pairs),
            object_accuracy=object_hits / len(pairs),
            hand_accuracy=hand_hits / len(pairs),
            sample_count=len(pairs),
            measurement_status=measurement_status,
            evidence_context_digest=context_digest,
            evidence_context_identity=context_identity,
            metric_policy_identity=self._policy.policy_identity,
            metric_policy_digest=self._policy.policy_digest,
            metric_policy_version=self._policy.policy_version,
        )

    def calculate_calibration(
        self,
        predictions: Any,
        ground_truth: Any,
        *,
        evidence_context: BenchmarkEvidenceContext | None = None,
    ) -> CalibrationMetrics:
        """Calculate calibration metrics from predictions and ground truth.

        Args:
            predictions: Model predictions with confidence scores.
            ground_truth: Ground-truth labels.
            evidence_context: Frozen governed manifests and approval identity.

        Returns:
            CalibrationMetrics populated with ECE, Brier score, and rates.
        """
        predicted = _records(predictions, "predictions")
        truth = _records(ground_truth, "ground_truth")
        if len(predicted) != len(truth):
            raise ValueError("predictions and ground_truth must have the same length")
        pairs: list[tuple[float, int]] = []
        abstentions = unknowns = 0
        for prediction, target in zip(predicted, truth, strict=True):
            confidence = _field(prediction, "confidence", _field(prediction, "score"))
            label = _field(target, "label", _field(target, "value", target))
            if confidence is None:
                abstentions += 1
                continue
            try:
                confidence_float = float(confidence)
            except (TypeError, ValueError) as exc:
                raise ValueError("confidence values must be numeric") from exc
            if not 0.0 <= confidence_float <= 1.0:
                raise ValueError("confidence values must lie in [0, 1]")
            target_value = int(bool(label))
            pairs.append((confidence_float, target_value))
            if _field(prediction, "status") in {"UNKNOWN", "ABSTAIN", "ABSTAINED"}:
                unknowns += 1
        if not pairs:
            raise ValueError("at least one numeric confidence is required")
        brier = mean((confidence - label) ** 2 for confidence, label in pairs)
        bin_count = self._policy.calibration_bin_count
        bins: list[list[tuple[float, int]]] = [[] for _ in range(bin_count)]
        for pair in pairs:
            bins[min(bin_count - 1, int(pair[0] * bin_count))].append(pair)
        ece = sum(
            len(bucket)
            / len(pairs)
            * abs(mean(confidence for confidence, _ in bucket) - mean(label for _, label in bucket))
            for bucket in bins
            if bucket
        )
        measurement_status, context_digest, context_identity = _measurement_binding(
            evidence_context, self._policy
        )
        return CalibrationMetrics(
            ece=ece,
            brier_score=brier,
            abstention_rate=abstentions / len(predicted),
            unknown_rate=unknowns / len(predicted),
            sample_count=len(predicted),
            measurement_status=measurement_status,
            evidence_context_digest=context_digest,
            evidence_context_identity=context_identity,
            metric_policy_identity=self._policy.policy_identity,
            metric_policy_digest=self._policy.policy_digest,
            metric_policy_version=self._policy.policy_version,
        )


__all__ = [
    "BenchmarkMetricPolicy",
    "BoundaryMetrics",
    "CalibrationMetrics",
    "EventMetrics",
    "EvidenceBoundMetrics",
    "MeasurementStatus",
    "MetricsCalculator",
    "QAMetrics",
    "benchmark_metric_policy_projection",
]
