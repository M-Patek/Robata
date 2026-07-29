"""Non-promotional local Pareto reports for sampling and dense-QA policies.

This module deliberately keeps fixture quality values separate from governed benchmark
measurements. A report can compare local policy trade-offs, but it cannot be used as
production-quality evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.benchmark.metrics import BoundaryMetrics, EventMetrics, QAMetrics
from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]

_LOCAL_PARETO_REPORT_VERSION: Final[Literal["local-sampling-dense-pareto-v1"]] = (
    "local-sampling-dense-pareto-v1"
)
_LOCAL_QUALITY_METRIC_NAMES = (
    "qa_macro_f1",
    "event_average_recall",
    "boundary_temporal_iou",
)
_LOCAL_COST_METRIC_NAMES = (
    "unique_image_count",
    "provider_image_count",
    "logical_call_count",
    "cpu_time_ns",
)


class LocalSamplingDensePolicyObservation(StrictModel):
    """One local-fixture observation for one sampling/dense policy.

    The embedded metrics reuse the benchmark calculator's vocabulary, but each must
    remain NOT_MEASURED. They are useful for local regression comparison only; they
    do not bind to a governed corpus, labels, or production-quality claim.
    """

    policy_id: NonEmptyString
    policy_version: SchemaVersion
    base_sampling_fps: PositiveFiniteFloat
    dense_sampling_fps: NonNegativeFiniteFloat
    qa_metrics: QAMetrics
    event_metrics: EventMetrics
    boundary_metrics: BoundaryMetrics
    unique_image_count: NonNegativeInt
    provider_image_count: NonNegativeInt
    logical_call_count: NonNegativeInt
    cpu_time_ns: NonNegativeInt

    @model_validator(mode="after")
    def validate_local_only_metrics(self) -> Self:
        metrics = (
            ("qa_metrics", self.qa_metrics),
            ("event_metrics", self.event_metrics),
            ("boundary_metrics", self.boundary_metrics),
        )
        for name, metric in metrics:
            if metric.measurement_status != "NOT_MEASURED":
                raise ValueError(f"{name} must be NOT_MEASURED in a local Pareto report")
            if (
                metric.evidence_context_digest is not None
                or metric.evidence_context_identity is not None
            ):
                raise ValueError(f"{name} cannot bind governed evidence in a local Pareto report")

        metric_policy = (
            self.qa_metrics.metric_policy_identity,
            self.qa_metrics.metric_policy_digest,
            self.qa_metrics.metric_policy_version,
        )
        for name, metric in metrics[1:]:
            if (
                metric.metric_policy_identity,
                metric.metric_policy_digest,
                metric.metric_policy_version,
            ) != metric_policy:
                raise ValueError(f"{name} must use the same metric policy as qa_metrics")

        for name, value in self.quality_metric_values.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        return self

    @property
    def quality_metric_values(self) -> dict[str, float]:
        """Return the fixed local quality objectives, all higher-is-better."""

        return {
            "qa_macro_f1": self.qa_metrics.macro_f1,
            "event_average_recall": self.event_metrics.average_recall,
            "boundary_temporal_iou": self.boundary_metrics.temporal_iou,
        }

    @property
    def cost_metric_values(self) -> dict[str, int]:
        """Return the fixed cost objectives, all lower-is-better."""

        return {
            "unique_image_count": self.unique_image_count,
            "provider_image_count": self.provider_image_count,
            "logical_call_count": self.logical_call_count,
            "cpu_time_ns": self.cpu_time_ns,
        }


class LocalSamplingDenseParetoReport(StrictModel):
    """A reproducible, explicitly non-production sampling/dense Pareto report."""

    report_version: Literal["local-sampling-dense-pareto-v1"] = _LOCAL_PARETO_REPORT_VERSION
    fixture_manifest_digest: Sha256Digest
    pipeline_version: SchemaVersion
    model_identifier: NonEmptyString
    prompt_version: SchemaVersion
    evidence_class: Literal["LOCAL_CONFORMANCE"] = "LOCAL_CONFORMANCE"
    provider_mode: Literal["LOCAL_OFFLINE_FIXTURE", "NO_PROVIDER_CALLS"] = "LOCAL_OFFLINE_FIXTURE"
    measurement_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    production_quality_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    production_eligible: Literal[False] = False
    policies: tuple[LocalSamplingDensePolicyObservation, ...] = Field(min_length=3)
    pareto_policy_ids: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        policy_ids = tuple(policy.policy_id for policy in self.policies)
        if policy_ids != tuple(sorted(policy_ids)):
            raise ValueError("policies must be sorted by policy_id")
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("policy_id values must be unique")

        sampling_dense_configurations = {
            (policy.base_sampling_fps, policy.dense_sampling_fps) for policy in self.policies
        }
        if len(sampling_dense_configurations) < 3:
            raise ValueError("report must compare at least three sampling/dense configurations")

        policy_metric_binding = _metric_policy_binding(self.policies[0])
        for policy in self.policies[1:]:
            if _metric_policy_binding(policy) != policy_metric_binding:
                raise ValueError("all policies must use the same benchmark metric policy")

        expected_frontier = _pareto_policy_ids(self.policies)
        if self.pareto_policy_ids != expected_frontier:
            raise ValueError("pareto_policy_ids do not match the policy observations")
        return self

    @property
    def metric_policy_identity(self) -> str:
        """Return the shared metric policy identity used by all local rows."""

        return self.policies[0].qa_metrics.metric_policy_identity

    @property
    def metric_policy_digest(self) -> str:
        """Return the shared metric policy digest used by all local rows."""

        return self.policies[0].qa_metrics.metric_policy_digest

    @property
    def metric_policy_version(self) -> str:
        """Return the shared metric policy version used by all local rows."""

        return self.policies[0].qa_metrics.metric_policy_version

    def is_pareto_optimal(self, policy_id: str) -> bool:
        """Return whether a named policy belongs to the local frontier."""

        if policy_id not in {policy.policy_id for policy in self.policies}:
            raise ValueError("policy_id is not present in this report")
        return policy_id in self.pareto_policy_ids

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready payload with explicit local-only provenance."""

        return self.model_dump(mode="json")

    def render_markdown(self) -> str:
        """Render a compact local report without implying production quality."""

        lines = [
            "# Local sampling/dense Pareto report",
            "",
            f"- Fixture manifest: {self.fixture_manifest_digest}",
            f"- Pipeline version: {self.pipeline_version}",
            f"- Model: {self.model_identifier}",
            f"- Prompt version: {self.prompt_version}",
            f"- Evidence class: {self.evidence_class}",
            f"- Provider mode: {self.provider_mode}",
            "- Quality measurement status: NOT_MEASURED",
            "- Production quality: NOT_MEASURED - local fixture metrics are",
            "  not production quality.",
            "",
            "| Policy | Base FPS | Dense FPS | QA macro F1 (local) | Event recall (local) | "
            "Boundary IoU (local) | Unique images | Provider images | Logical calls | CPU ms | "
            "Pareto |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        frontier = set(self.pareto_policy_ids)
        for policy in self.policies:
            quality = policy.quality_metric_values
            costs = policy.cost_metric_values
            policy_id = policy.policy_id.replace("|", "\\|")
            lines.append(
                "| "
                f"{policy_id} | {policy.base_sampling_fps:g} | {policy.dense_sampling_fps:g} | "
                f"{quality['qa_macro_f1']:.6f} | {quality['event_average_recall']:.6f} | "
                f"{quality['boundary_temporal_iou']:.6f} | {costs['unique_image_count']} | "
                f"{costs['provider_image_count']} | {costs['logical_call_count']} | "
                f"{costs['cpu_time_ns'] / 1_000_000:.3f} | "
                f"{'YES' if policy.policy_id in frontier else 'NO'} |"
            )
        return "\n".join(lines) + "\n"


def build_local_sampling_dense_pareto_report(
    *,
    fixture_manifest_digest: Sha256Digest,
    pipeline_version: SchemaVersion,
    model_identifier: NonEmptyString,
    prompt_version: SchemaVersion,
    policies: Iterable[LocalSamplingDensePolicyObservation],
    provider_mode: Literal["LOCAL_OFFLINE_FIXTURE", "NO_PROVIDER_CALLS"] = (
        "LOCAL_OFFLINE_FIXTURE"
    ),
) -> LocalSamplingDenseParetoReport:
    """Build a deterministic local-only report from at least three policy rows."""

    checked_policies = tuple(policies)
    if not all(
        isinstance(policy, LocalSamplingDensePolicyObservation) for policy in checked_policies
    ):
        raise TypeError("policies must contain LocalSamplingDensePolicyObservation values")
    ordered_policies = tuple(sorted(checked_policies, key=lambda policy: policy.policy_id))
    return LocalSamplingDenseParetoReport(
        fixture_manifest_digest=fixture_manifest_digest,
        pipeline_version=pipeline_version,
        model_identifier=model_identifier,
        prompt_version=prompt_version,
        provider_mode=provider_mode,
        policies=ordered_policies,
        pareto_policy_ids=_pareto_policy_ids(ordered_policies),
    )


def _metric_policy_binding(
    policy: LocalSamplingDensePolicyObservation,
) -> tuple[str, str, str]:
    return (
        policy.qa_metrics.metric_policy_identity,
        policy.qa_metrics.metric_policy_digest,
        policy.qa_metrics.metric_policy_version,
    )


def _pareto_policy_ids(
    policies: tuple[LocalSamplingDensePolicyObservation, ...],
) -> tuple[str, ...]:
    return tuple(
        policy.policy_id
        for policy in policies
        if not any(
            _dominates(other, policy) for other in policies if other.policy_id != policy.policy_id
        )
    )


def _dominates(
    left: LocalSamplingDensePolicyObservation,
    right: LocalSamplingDensePolicyObservation,
) -> bool:
    """Return whether left is no worse in every objective and better in one."""

    quality_left = left.quality_metric_values
    quality_right = right.quality_metric_values
    cost_left = left.cost_metric_values
    cost_right = right.cost_metric_values
    quality_not_worse = all(
        quality_left[name] >= quality_right[name] for name in _LOCAL_QUALITY_METRIC_NAMES
    )
    cost_not_worse = all(cost_left[name] <= cost_right[name] for name in _LOCAL_COST_METRIC_NAMES)
    strictly_better = any(
        quality_left[name] > quality_right[name] for name in _LOCAL_QUALITY_METRIC_NAMES
    ) or any(cost_left[name] < cost_right[name] for name in _LOCAL_COST_METRIC_NAMES)
    return quality_not_worse and cost_not_worse and strictly_better


__all__ = [
    "LocalSamplingDenseParetoReport",
    "LocalSamplingDensePolicyObservation",
    "build_local_sampling_dense_pareto_report",
]
