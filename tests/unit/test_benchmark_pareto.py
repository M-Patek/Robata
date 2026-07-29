from __future__ import annotations

import pytest

from robata.benchmark import (
    BenchmarkEvidenceContext,
    BenchmarkMetricPolicy,
    BoundaryMetrics,
    EventMetrics,
    LocalSamplingDenseParetoReport,
    LocalSamplingDensePolicyObservation,
    QAMetrics,
    build_local_sampling_dense_pareto_report,
)
from robata.contracts.hashing import semantic_sha256


def _digest(label: str) -> str:
    return semantic_sha256({"local-pareto-test": label})


def _context() -> BenchmarkEvidenceContext:
    return BenchmarkEvidenceContext.create(
        benchmark_id="00000000-0000-0000-0000-000000000001",
        benchmark_manifest_digest=_digest("benchmark-manifest"),
        governed_corpus_digest=_digest("governed-corpus"),
        ground_truth_manifest_digest=_digest("ground-truth-manifest"),
        grouped_split_manifest_digest=_digest("grouped-split-manifest"),
        data_split="FROZEN_TEST",
        governance_approved=True,
        governance_approval_id="approval-local-pareto-test",
        governance_approval_digest=_digest("governance-approval"),
        governance_policy_version="governance-policy-1.0",
    )


def _metric_policy(
    context: BenchmarkEvidenceContext | None = None,
) -> BenchmarkMetricPolicy:
    return BenchmarkMetricPolicy.create(
        policy_version="local-fixture-metrics-1.0",
        critical_issue_codes=("BLACK_SCREEN",),
        event_iou_thresholds=(0.5,),
        event_start_end_tolerance_ns=100,
        boundary_tolerance_ns=100,
        calibration_bin_count=4,
        governance_approval_id=(context.governance_approval_id if context is not None else None),
        governance_approval_digest=(
            context.governance_approval_digest if context is not None else None
        ),
        governance_policy_version=(
            context.governance_policy_version if context is not None else None
        ),
    )


def _qa_metrics(policy: BenchmarkMetricPolicy, value: float) -> QAMetrics:
    return QAMetrics(
        per_issue_precision={"BLACK_SCREEN": value},
        per_issue_recall={"BLACK_SCREEN": value},
        per_issue_f1={"BLACK_SCREEN": value},
        macro_f1=value,
        micro_precision=value,
        micro_recall=value,
        micro_f1=value,
        critical_issue_recall=value,
        temporal_iou=value,
        recording_precision=value,
        recording_recall=value,
        false_accept_rate=1.0 - value,
        false_reject_rate=1.0 - value,
        sample_count=12,
        metric_policy_identity=policy.policy_identity,
        metric_policy_digest=policy.policy_digest,
        metric_policy_version=policy.policy_version,
    )


def _event_metrics(policy: BenchmarkMetricPolicy, value: float) -> EventMetrics:
    return EventMetrics(
        recall_at_iou={"0.5": value},
        average_recall=value,
        start_end_hit_rate=value,
        miss_rate_by_class={"ACTION": 1.0 - value},
        false_candidates_per_hour=1.0 - value,
        duplicate_rate=0.0,
        overmerge_rate=0.0,
        oversplit_rate=0.0,
        sample_count=12,
        metric_policy_identity=policy.policy_identity,
        metric_policy_digest=policy.policy_digest,
        metric_policy_version=policy.policy_version,
    )


def _boundary_metrics(policy: BenchmarkMetricPolicy, value: float) -> BoundaryMetrics:
    return BoundaryMetrics(
        start_mae=1.0 - value,
        end_mae=1.0 - value,
        median_error=1.0 - value,
        p95_error=1.0 - value,
        temporal_iou=value,
        within_tolerance_rate=value,
        classification_accuracy=value,
        object_accuracy=value,
        hand_accuracy=value,
        sample_count=12,
        metric_policy_identity=policy.policy_identity,
        metric_policy_digest=policy.policy_digest,
        metric_policy_version=policy.policy_version,
    )


def _observation(
    policy_id: str,
    *,
    base_sampling_fps: float,
    dense_sampling_fps: float,
    quality: float,
    unique_images: int,
    provider_images: int,
    logical_calls: int,
    cpu_time_ns: int,
    policy: BenchmarkMetricPolicy | None = None,
) -> LocalSamplingDensePolicyObservation:
    metric_policy = policy or _metric_policy()
    return LocalSamplingDensePolicyObservation(
        policy_id=policy_id,
        policy_version="adaptive-cascade-v1",
        base_sampling_fps=base_sampling_fps,
        dense_sampling_fps=dense_sampling_fps,
        qa_metrics=_qa_metrics(metric_policy, quality),
        event_metrics=_event_metrics(metric_policy, quality),
        boundary_metrics=_boundary_metrics(metric_policy, quality),
        unique_image_count=unique_images,
        provider_image_count=provider_images,
        logical_call_count=logical_calls,
        cpu_time_ns=cpu_time_ns,
    )


def _policy_matrix() -> tuple[LocalSamplingDensePolicyObservation, ...]:
    policy = _metric_policy()
    return (
        _observation(
            "high-recall",
            base_sampling_fps=1.0,
            dense_sampling_fps=10.0,
            quality=0.9,
            unique_images=30,
            provider_images=36,
            logical_calls=9,
            cpu_time_ns=30_000_000,
            policy=policy,
        ),
        _observation(
            "dominated",
            base_sampling_fps=2.0,
            dense_sampling_fps=20.0,
            quality=0.6,
            unique_images=50,
            provider_images=60,
            logical_calls=16,
            cpu_time_ns=50_000_000,
            policy=policy,
        ),
        _observation(
            "balanced",
            base_sampling_fps=0.5,
            dense_sampling_fps=5.0,
            quality=0.8,
            unique_images=20,
            provider_images=24,
            logical_calls=6,
            cpu_time_ns=20_000_000,
            policy=policy,
        ),
        _observation(
            "base-only",
            base_sampling_fps=0.25,
            dense_sampling_fps=0.0,
            quality=0.7,
            unique_images=10,
            provider_images=12,
            logical_calls=3,
            cpu_time_ns=10_000_000,
            policy=policy,
        ),
    )


def test_local_pareto_report_compares_quality_and_cost_without_production_claim() -> None:
    report = build_local_sampling_dense_pareto_report(
        fixture_manifest_digest=_digest("fixture-manifest"),
        pipeline_version="canonical-fixture-v1",
        model_identifier="local-fixture-provider",
        prompt_version="qa-event-prompt-v1",
        policies=_policy_matrix(),
    )

    assert report.measurement_status == "NOT_MEASURED"
    assert report.production_quality_status == "NOT_MEASURED"
    assert report.production_eligible is False
    assert tuple(policy.policy_id for policy in report.policies) == (
        "balanced",
        "base-only",
        "dominated",
        "high-recall",
    )
    assert report.pareto_policy_ids == ("balanced", "base-only", "high-recall")
    assert report.is_pareto_optimal("balanced") is True
    assert report.is_pareto_optimal("dominated") is False
    with pytest.raises(ValueError, match="not present"):
        report.is_pareto_optimal("missing")

    payload = report.as_dict()
    assert payload["evidence_class"] == "LOCAL_CONFORMANCE"
    assert payload["policies"][0]["provider_image_count"] == 24
    assert payload["policies"][0]["logical_call_count"] == 6
    assert payload["policies"][0]["cpu_time_ns"] == 20_000_000

    markdown = report.render_markdown()
    assert "QA macro F1 (local)" in markdown
    assert "Provider images" in markdown
    assert "Logical calls" in markdown
    assert "CPU ms" in markdown
    assert "Production quality: NOT_MEASURED" in markdown
    assert "not production quality" in markdown


def test_local_pareto_report_rejects_insufficient_or_tampered_policy_matrix() -> None:
    matrix = _policy_matrix()
    with pytest.raises(ValueError, match="at least"):
        build_local_sampling_dense_pareto_report(
            fixture_manifest_digest=_digest("fixture-manifest"),
            pipeline_version="canonical-fixture-v1",
            model_identifier="local-fixture-provider",
            prompt_version="qa-event-prompt-v1",
            policies=matrix[:2],
        )

    report = build_local_sampling_dense_pareto_report(
        fixture_manifest_digest=_digest("fixture-manifest"),
        pipeline_version="canonical-fixture-v1",
        model_identifier="local-fixture-provider",
        prompt_version="qa-event-prompt-v1",
        policies=matrix,
    )
    payload = report.model_dump(mode="python")
    payload["pareto_policy_ids"] = ("dominated",)
    with pytest.raises(ValueError, match="pareto_policy_ids"):
        LocalSamplingDenseParetoReport.model_validate(payload)


def test_local_pareto_rows_reject_governed_metric_claims() -> None:
    context = _context()
    policy = _metric_policy(context)
    row = _observation(
        "governed-claim",
        base_sampling_fps=1.0,
        dense_sampling_fps=10.0,
        quality=0.9,
        unique_images=30,
        provider_images=36,
        logical_calls=9,
        cpu_time_ns=30_000_000,
        policy=policy,
    )
    qa_with_production_claim = QAMetrics.model_validate(
        {
            **row.qa_metrics.model_dump(mode="python"),
            "measurement_status": "MEASURED",
            "evidence_context_digest": context.context_digest,
            "evidence_context_identity": context.context_identity,
        }
    )

    with pytest.raises(ValueError, match="qa_metrics must be NOT_MEASURED"):
        LocalSamplingDensePolicyObservation(
            policy_id=row.policy_id,
            policy_version=row.policy_version,
            base_sampling_fps=row.base_sampling_fps,
            dense_sampling_fps=row.dense_sampling_fps,
            qa_metrics=qa_with_production_claim,
            event_metrics=row.event_metrics,
            boundary_metrics=row.boundary_metrics,
            unique_image_count=row.unique_image_count,
            provider_image_count=row.provider_image_count,
            logical_call_count=row.logical_call_count,
            cpu_time_ns=row.cpu_time_ns,
        )
