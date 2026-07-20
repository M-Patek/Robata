from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from robata.benchmark import (
    BenchmarkEvidenceContext,
    BenchmarkResults,
    GateCategory,
    MetricsCalculator,
    PromotionEvaluator,
    PromotionGate,
    PromotionGateRegistry,
)
from robata.contracts.hashing import semantic_sha256


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:benchmark-test:{label}"))


def _digest(label: str) -> str:
    return semantic_sha256({"benchmark-test": label})


def _context(
    benchmark_id: str | None = None,
    *,
    governed_corpus_label: str = "governed-corpus",
) -> BenchmarkEvidenceContext:
    return BenchmarkEvidenceContext.create(
        benchmark_id=benchmark_id or _id("benchmark"),
        benchmark_manifest_digest=_digest("benchmark-manifest"),
        governed_corpus_digest=_digest(governed_corpus_label),
        ground_truth_manifest_digest=_digest("ground-truth-manifest"),
        grouped_split_manifest_digest=_digest("grouped-split-manifest"),
        data_split="FROZEN_TEST",
        governance_approved=True,
        governance_approval_id="approval-2026-07-19",
        governance_approval_digest=_digest("governance-approval"),
        governance_policy_version="governance-policy-1.0",
    )


def _gate(category: GateCategory, *, actual_key: str, comparison: str = "GTE") -> PromotionGate:
    return PromotionGate(
        gate_id=_id(f"gate:{category.value}"),
        category=category,
        metric_definition=actual_key,
        threshold=0.8 if comparison == "GTE" else 0.5,
        margin=0.0,
        comparison=comparison,
        denominator="frozen-test-items",
        required_strata=("all",),
        data_split="FROZEN_TEST",
        owner="benchmark-owner",
        effective_date="2026-07-19",
        failure_action="REJECT_PROMOTION",
        version="1.0",
    )


def _results(
    values: dict[str, object],
    *,
    benchmark_id: str | None = None,
    data_split: str = "FROZEN_TEST",
    include_strata: bool = True,
    stratum_values: dict[str, object] | None = None,
    evidence_context: BenchmarkEvidenceContext | None = None,
) -> BenchmarkResults:
    resolved_benchmark_id = benchmark_id or (
        evidence_context.benchmark_id if evidence_context is not None else _id("benchmark")
    )
    context = evidence_context or _context(resolved_benchmark_id)
    payload = {
        "benchmark_id": resolved_benchmark_id,
        "data_split": data_split,
        "measurement_status": "MEASURED",
        "evidence_context": context,
        "evidence_context_digest": context.context_digest,
        "evidence_context_identity": context.context_identity,
        **values,
    }
    if include_strata:
        payload["strata"] = {
            "all": {
                "measurement_status": "MEASURED",
                "evidence_context_digest": context.context_digest,
                "evidence_context_identity": context.context_identity,
                **(stratum_values if stratum_values is not None else values),
            }
        }
    return BenchmarkResults(payload)


def test_metrics_reject_empty_cohorts_instead_of_returning_zero_evidence() -> None:
    calculator = MetricsCalculator()
    with pytest.raises(ValueError, match="at least one record"):
        calculator.calculate_qa_metrics([], [])
    with pytest.raises(ValueError, match="at least one record"):
        calculator.calculate_event_metrics([], [])


def test_offline_metrics_calculate_perfect_fixture() -> None:
    calculator = MetricsCalculator()
    context = _context()
    qa_record = {
        "recording_id": "recording-1",
        "issues": ({"code": "MOTION_BLUR", "start_ns": 0, "end_ns": 10},),
    }
    event = {
        "recording_id": "recording-1",
        "action_id": "action-1",
        "label": "grasp",
        "object": "cup",
        "hand": "right",
        "interval": {"start_ns": 0, "end_ns": 10},
    }

    ungoverned_qa = calculator.calculate_qa_metrics((qa_record,), (qa_record,))
    qa = calculator.calculate_qa_metrics(
        (qa_record,),
        (qa_record,),
        evidence_context=context,
    )
    events = calculator.calculate_event_metrics(
        (event,),
        (event,),
        evidence_context=context,
    )
    boundaries = calculator.calculate_boundary_metrics(
        (event,),
        (event,),
        evidence_context=context,
    )
    calibration = calculator.calculate_calibration(
        ({"confidence": 1.0},),
        ({"label": True},),
        evidence_context=context,
    )

    assert ungoverned_qa.measurement_status == "NOT_MEASURED"
    assert ungoverned_qa.evidence_context_digest is None
    assert qa.measurement_status == "MEASURED"
    assert qa.evidence_context_digest == context.context_digest
    assert qa.evidence_context_identity == context.context_identity
    assert {
        events.measurement_status,
        boundaries.measurement_status,
        calibration.measurement_status,
    } == {"MEASURED"}
    assert qa.micro_f1 == 1.0
    assert qa.critical_issue_recall == 1.0
    assert events.recall_at_iou == {"0.3": 1.0, "0.5": 1.0, "0.7": 1.0}
    assert events.mAP == 1.0
    assert boundaries.temporal_iou == 1.0
    assert boundaries.start_mae == 0.0
    assert calibration.ece == 0.0
    assert calibration.brier_score == 0.0


def test_evidence_context_is_frozen_and_content_addressed() -> None:
    context = _context()
    same_context = _context()
    assert context.context_digest == same_context.context_digest
    assert context.context_identity == f"benchmark-evidence:{context.context_digest}"

    tampered = context.model_dump(mode="python")
    tampered["governed_corpus_digest"] = _digest("other-corpus")
    with pytest.raises(ValidationError, match="context_digest"):
        BenchmarkEvidenceContext.model_validate(tampered)

    validation_context = context.model_dump(mode="python")
    validation_context["data_split"] = "VALIDATION"
    with pytest.raises(ValidationError, match="FROZEN_TEST"):
        BenchmarkEvidenceContext.model_validate(validation_context)


def test_measured_metric_binding_cannot_be_added_to_local_output() -> None:
    record = {"recording_id": "recording-1", "issues": ()}
    local_metrics = MetricsCalculator().calculate_qa_metrics((record,), (record,))
    forged = local_metrics.model_dump(mode="python")
    forged["measurement_status"] = "MEASURED"

    with pytest.raises(ValidationError, match="evidence context binding"):
        type(local_metrics).model_validate(forged)


def test_event_metrics_require_class_matching_and_separate_false_candidates() -> None:
    calculator = MetricsCalculator()
    duration_ns = 3_600_000_000_000
    proposals = (
        {
            "recording_id": "recording-1",
            "label": "grasp",
            "interval": {"start_ns": 0, "end_ns": 10},
        },
        {
            "recording_id": "recording-1",
            "label": "grasp",
            "interval": {"start_ns": 0, "end_ns": 10},
        },
        {
            "recording_id": "recording-1",
            "label": "reach",
            "interval": {"start_ns": 0, "end_ns": 10},
        },
    )
    ground_truth = (
        {
            "recording_id": "recording-1",
            "label": "grasp",
            "interval": {"start_ns": 0, "end_ns": 10},
            "duration_ns": duration_ns,
        },
    )

    metrics = calculator.calculate_event_metrics(proposals, ground_truth)

    assert metrics.recall_at_iou["0.5"] == 1.0
    assert metrics.false_candidates_per_hour == 2.0
    assert metrics.duplicate_rate == pytest.approx(1 / 3)
    assert metrics.oversplit_rate == 1.0
    assert metrics.miss_rate_by_class == {"grasp": 0.0}


def test_event_metrics_do_not_invent_a_time_denominator() -> None:
    with pytest.raises(ValueError, match="duration_ns"):
        MetricsCalculator().calculate_event_metrics(
            (
                {
                    "recording_id": "recording-1",
                    "label": "false",
                    "interval": {"start_ns": 0, "end_ns": 10},
                },
            ),
            (
                {
                    "recording_id": "recording-1",
                    "label": "truth",
                    "interval": {"start_ns": 0, "end_ns": 10},
                },
            ),
        )


def test_event_metrics_deduplicate_recording_duration_and_use_semantic_ties() -> None:
    calculator = MetricsCalculator()
    recording_duration = 3_600_000_000_000
    truth = (
        {
            "recording_id": "recording-1",
            "label": "grasp",
            "interval": {"start_ns": 0, "end_ns": 10},
            "duration_ns": recording_duration,
        },
        {
            "recording_id": "recording-1",
            "label": "reach",
            "interval": {"start_ns": 20, "end_ns": 30},
            "duration_ns": recording_duration,
        },
    )
    proposals = (
        {
            "recording_id": "recording-1",
            "label": "reach",
            "interval": {"start_ns": 20, "end_ns": 30},
        },
        {
            "recording_id": "recording-1",
            "label": "grasp",
            "interval": {"start_ns": 0, "end_ns": 10},
        },
        {
            "recording_id": "recording-1",
            "label": "other",
            "interval": {"start_ns": 40, "end_ns": 50},
        },
    )

    metrics = calculator.calculate_event_metrics(proposals, truth)

    assert metrics.false_candidates_per_hour == 1.0
    assert metrics.recall_at_iou["0.5"] == 1.0


def test_event_and_boundary_metrics_reject_cross_record_or_unpaired_inputs() -> None:
    calculator = MetricsCalculator()
    with pytest.raises(ValueError, match="duration_ns"):
        calculator.calculate_event_metrics(
            (
                {
                    "recording_id": "recording-a",
                    "label": "grasp",
                    "interval": {"start_ns": 0, "end_ns": 10},
                },
            ),
            (
                {
                    "recording_id": "recording-b",
                    "label": "grasp",
                    "interval": {"start_ns": 0, "end_ns": 10},
                },
            ),
        )

    boundary = {
        "recording_id": "recording-a",
        "action_id": "action-1",
        "label": "grasp",
        "interval": {"start_ns": 0, "end_ns": 10},
    }
    with pytest.raises(ValueError, match="same physical events"):
        calculator.calculate_boundary_metrics(
            (boundary,),
            ({**boundary, "action_id": "action-2"},),
        )


def test_promotion_gate_requires_explicit_measured_evidence() -> None:
    gate = _gate(GateCategory.QA, actual_key="qa.micro_f1")
    evaluator = PromotionEvaluator()

    missing = evaluator.check_qa(BenchmarkResults({}), gate)
    unmeasured = evaluator.check_qa(
        BenchmarkResults({"qa": {"micro_f1": 0.95}}),
        gate,
    )
    measured = evaluator.check_qa(_results({"qa": {"micro_f1": 0.95}}), gate)
    forged_status = evaluator.check_qa(
        BenchmarkResults(
            {
                "benchmark_id": _id("benchmark"),
                "data_split": "FROZEN_TEST",
                "measurement_status": "MEASURED",
                "qa": {"micro_f1": 0.95},
            }
        ),
        gate,
    )

    assert missing.passed is False
    assert missing.evidence["reason"] == "MISSING_METRIC"
    assert unmeasured.passed is False
    assert unmeasured.evidence["reason"] == "NOT_MEASURED"
    assert forged_status.passed is False
    assert forged_status.evidence["reason"] == "MISSING_EVIDENCE_CONTEXT"
    assert measured.passed is True
    assert measured.actual_value == 0.95


def test_promotion_gate_enforces_split_and_every_required_stratum() -> None:
    gate = _gate(GateCategory.QA, actual_key="qa.micro_f1")
    evaluator = PromotionEvaluator()

    wrong_split = evaluator.check_qa(
        _results({"qa": {"micro_f1": 0.95}}, data_split="VALIDATION"),
        gate,
    )
    missing_stratum = evaluator.check_qa(
        _results({"qa": {"micro_f1": 0.95}}, include_strata=False),
        gate,
    )
    below_floor = evaluator.check_qa(
        _results(
            {"qa": {"micro_f1": 0.95}},
            stratum_values={"qa": {"micro_f1": 0.7}},
        ),
        gate,
    )

    assert wrong_split.evidence["reason"] == "EVIDENCE_CONTEXT_RESULT_MISMATCH"
    assert missing_stratum.evidence["reason"] == "MISSING_REQUIRED_STRATUM"
    assert below_floor.passed is False
    assert below_floor.actual_value == 0.7
    assert below_floor.evidence["reason"] == "THRESHOLD_NOT_MET"


def test_promotion_rejects_tampered_context_and_stratum_binding() -> None:
    gate = _gate(GateCategory.QA, actual_key="qa.micro_f1")
    context = _context()
    tampered_context = context.model_dump(mode="python")
    tampered_context["ground_truth_manifest_digest"] = _digest("other-ground-truth")
    tampered = BenchmarkResults(
        {
            "benchmark_id": context.benchmark_id,
            "data_split": "FROZEN_TEST",
            "measurement_status": "MEASURED",
            "evidence_context": tampered_context,
            "evidence_context_digest": context.context_digest,
            "evidence_context_identity": context.context_identity,
            "qa": {"micro_f1": 0.95},
        }
    )
    tampered_result = PromotionEvaluator().check_qa(tampered, gate)
    assert tampered_result.evidence["reason"] == "INVALID_EVIDENCE_CONTEXT"

    wrong_stratum = _results(
        {"qa": {"micro_f1": 0.95}},
        stratum_values={
            "evidence_context_digest": _digest("wrong-context"),
            "qa": {"micro_f1": 0.95},
        },
    )
    stratum_result = PromotionEvaluator().check_qa(wrong_stratum, gate)
    assert stratum_result.evidence["reason"] == "STRATUM_NOT_MEASURED"
    assert stratum_result.evidence["evidence_error"] == "STRATUM_EVIDENCE_CONTEXT_MISMATCH"


def test_full_registered_gate_set_can_pass_and_missing_category_rejects() -> None:
    gates = tuple(
        _gate(
            category,
            actual_key=f"metrics.{category.value.lower()}",
            comparison="LTE" if category is GateCategory.COST else "GTE",
        )
        for category in GateCategory
    )
    registry = PromotionGateRegistry(
        registry_id=_id("registry"),
        gates=gates,
        benchmark_id=_id("benchmark"),
        evidence_context_digest=_context().context_digest,
        frozen_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    metric_values = {
        category.value.lower(): (0.4 if category is GateCategory.COST else 0.9)
        for category in GateCategory
    }
    clock_value = datetime(2026, 7, 20, tzinfo=UTC)
    evaluator = PromotionEvaluator(clock=lambda: clock_value)

    approved = evaluator.evaluate(_results({"metrics": metric_values}), registry)
    incomplete = evaluator.evaluate(
        _results({"metrics": metric_values}),
        registry.model_copy(update={"gates": gates[:-1]}),
    )

    assert approved.approved is True
    assert approved.timestamp == clock_value
    assert len(approved.approved_gates) == len(GateCategory)
    assert incomplete.approved is False
    assert any(
        result.evidence.get("reason") == "MISSING_REQUIRED_GATE"
        for result in incomplete.rejected_gates
    )


def test_registry_evaluation_rejects_wrong_benchmark_identity() -> None:
    gates = tuple(
        _gate(category, actual_key=f"metrics.{category.value.lower()}") for category in GateCategory
    )
    registry = PromotionGateRegistry(
        registry_id=_id("registry-identity"),
        gates=gates,
        benchmark_id=_id("benchmark"),
        evidence_context_digest=_context().context_digest,
        frozen_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    metric_values = {category.value.lower(): 0.9 for category in GateCategory}

    decision = PromotionEvaluator().evaluate(
        _results({"metrics": metric_values}, benchmark_id=_id("wrong-benchmark")),
        registry,
    )

    assert decision.approved is False
    assert decision.validation_errors == (
        "BENCHMARK_ID_MISMATCH",
        "EVIDENCE_CONTEXT_REGISTRY_MISMATCH",
    )


def test_registry_pins_the_expected_evidence_context() -> None:
    expected_context = _context()
    other_context = _context(governed_corpus_label="different-governed-corpus")
    registry = PromotionGateRegistry(
        registry_id=_id("registry-context"),
        gates=(_gate(GateCategory.QA, actual_key="qa.micro_f1"),),
        benchmark_id=_id("benchmark"),
        evidence_context_digest=expected_context.context_digest,
        frozen_at=datetime(2026, 7, 19, tzinfo=UTC),
    )

    decision = PromotionEvaluator().evaluate(
        _results(
            {"qa": {"micro_f1": 0.95}},
            evidence_context=other_context,
        ),
        registry,
    )

    assert decision.approved is False
    assert "EVIDENCE_CONTEXT_REGISTRY_MISMATCH" in decision.validation_errors
