"""Fixture-only quality accounting for P11 recording association.

The association report remains a nonblocking, non-authoritative derived
artifact.  These metrics make local fixture behavior inspectable without
claiming representative association precision or recall; P15 owns that
qualification once governed labels and a representative scope are available.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey
from robata.event_pipeline.recording_association import (
    AssociationPairDisposition,
    AssociationSourceActionRef,
    RecordingAssociationReport,
    verify_recording_association_report,
)

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
MetricMillionths = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]

RECORDING_ASSOCIATION_FIXTURE_TRUTH_PROJECTION_VERSION: Final = (
    "recording-association-fixture-truth-semantic-v1"
)
RECORDING_ASSOCIATION_FIXTURE_TRUTH_LOGICAL_KEY_NAMESPACE: Final = (
    "recording-association-fixture-truth-v1"
)
RECORDING_ASSOCIATION_FIXTURE_METRICS_PROJECTION_VERSION: Final = (
    "recording-association-fixture-metrics-semantic-v1"
)
RECORDING_ASSOCIATION_FIXTURE_METRICS_LOGICAL_KEY_NAMESPACE: Final = (
    "recording-association-fixture-metrics-v1"
)


class AssociationFixtureTruthCluster(StrictModel):
    """One adjudicated association set used only by a local fixture."""

    source_actions: tuple[AssociationSourceActionRef, ...]

    @model_validator(mode="after")
    def validate_cluster(self) -> Self:
        expected = tuple(sorted(self.source_actions, key=_source_action_sort_key))
        source_keys = tuple(item.source_action_logical_key for item in self.source_actions)
        if len(self.source_actions) < 2:
            raise ValueError("fixture truth cluster requires at least two source actions")
        if self.source_actions != expected or len(source_keys) != len(set(source_keys)):
            raise ValueError("fixture truth cluster source actions must be unique and ordered")
        return self


class AssociationFixtureTruth(StrictModel):
    """Content-addressed expected associations for deterministic local fixtures."""

    fixture_id: NonEmptyString
    source_clusters: tuple[AssociationFixtureTruthCluster, ...]
    projection_version: Literal["recording-association-fixture-truth-semantic-v1"] = (
        RECORDING_ASSOCIATION_FIXTURE_TRUTH_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        fixture_id: str,
        source_clusters: Sequence[AssociationFixtureTruthCluster],
    ) -> Self:
        values: dict[str, Any] = {
            "fixture_id": fixture_id,
            "source_clusters": tuple(sorted(source_clusters, key=_truth_cluster_sort_key)),
            "projection_version": RECORDING_ASSOCIATION_FIXTURE_TRUTH_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = cls.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{RECORDING_ASSOCIATION_FIXTURE_TRUTH_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
            **values,
        )
        digest = semantic_sha256(recording_association_fixture_truth_projection(draft))
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": (
                    f"{RECORDING_ASSOCIATION_FIXTURE_TRUTH_LOGICAL_KEY_NAMESPACE}:{digest}"
                ),
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_truth(self) -> Self:
        expected = tuple(sorted(self.source_clusters, key=_truth_cluster_sort_key))
        if self.source_clusters != expected:
            raise ValueError("fixture truth clusters must be canonically ordered")
        source_keys = tuple(
            action.source_action_logical_key
            for cluster in self.source_clusters
            for action in cluster.source_actions
        )
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("fixture truth source action cannot appear in multiple clusters")
        digest = semantic_sha256(recording_association_fixture_truth_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("fixture truth semantic identity is inconsistent")
        if (
            self.logical_key
            != f"{RECORDING_ASSOCIATION_FIXTURE_TRUTH_LOGICAL_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("fixture truth logical key is inconsistent")
        return self


class RecordingAssociationFixtureMetrics(StrictModel):
    """A content-addressed fixture result, explicitly not a representative claim."""

    fixture_truth: AssociationFixtureTruth
    report_logical_key: NodeLogicalKey
    report_semantic_sha256: Sha256Digest
    policy_semantic_sha256: Sha256Digest
    input_count: NonNegativeInt
    associated_input_count: NonNegativeInt
    predicted_associated_pair_count: NonNegativeInt
    expected_associated_pair_count: NonNegativeInt
    true_positive_pair_count: NonNegativeInt
    false_positive_pair_count: NonNegativeInt
    false_negative_pair_count: NonNegativeInt
    ambiguous_pair_count: NonNegativeInt
    precision_millionths: MetricMillionths
    recall_millionths: MetricMillionths
    f1_millionths: MetricMillionths
    association_coverage_millionths: MetricMillionths
    representative_measurement_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    projection_version: Literal["recording-association-fixture-metrics-semantic-v1"] = (
        RECORDING_ASSOCIATION_FIXTURE_METRICS_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.report_logical_key.rsplit(":", 1)[-1] != self.report_semantic_sha256:
            raise ValueError("fixture metrics report logical key is inconsistent")
        if self.associated_input_count > self.input_count:
            raise ValueError("associated input count cannot exceed input count")
        if self.predicted_associated_pair_count != (
            self.true_positive_pair_count + self.false_positive_pair_count
        ):
            raise ValueError("predicted pair count does not match precision population")
        if self.expected_associated_pair_count != (
            self.true_positive_pair_count + self.false_negative_pair_count
        ):
            raise ValueError("expected pair count does not match recall population")
        expected_precision = _ratio_millionths(
            self.true_positive_pair_count,
            self.predicted_associated_pair_count,
        )
        expected_recall = _ratio_millionths(
            self.true_positive_pair_count,
            self.expected_associated_pair_count,
        )
        expected_f1 = _f1_millionths(expected_precision, expected_recall)
        expected_coverage = _ratio_millionths(
            self.associated_input_count,
            self.input_count,
        )
        if (
            self.precision_millionths != expected_precision
            or self.recall_millionths != expected_recall
            or self.f1_millionths != expected_f1
            or self.association_coverage_millionths != expected_coverage
        ):
            raise ValueError("fixture association metrics do not match their populations")
        digest = semantic_sha256(recording_association_fixture_metrics_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("fixture association metrics semantic identity is inconsistent")
        if (
            self.logical_key
            != f"{RECORDING_ASSOCIATION_FIXTURE_METRICS_LOGICAL_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("fixture association metrics logical key is inconsistent")
        return self


def recording_association_fixture_truth_projection(
    truth: AssociationFixtureTruth,
) -> dict[str, object]:
    """Return the full semantic preimage for fixture association truth."""

    return {
        "semantic_projection_version": truth.projection_version,
        "fixture_id": truth.fixture_id,
        "source_clusters": [item.model_dump(mode="json") for item in truth.source_clusters],
        "production_eligible": truth.production_eligible,
    }


def recording_association_fixture_metrics_projection(
    metrics: RecordingAssociationFixtureMetrics,
) -> dict[str, object]:
    """Return the complete non-authoritative metric projection."""

    return {
        "semantic_projection_version": metrics.projection_version,
        "fixture_truth": recording_association_fixture_truth_projection(metrics.fixture_truth),
        "fixture_truth_semantic_sha256": metrics.fixture_truth.semantic_sha256,
        "report_logical_key": metrics.report_logical_key,
        "report_semantic_sha256": metrics.report_semantic_sha256,
        "policy_semantic_sha256": metrics.policy_semantic_sha256,
        "input_count": metrics.input_count,
        "associated_input_count": metrics.associated_input_count,
        "predicted_associated_pair_count": metrics.predicted_associated_pair_count,
        "expected_associated_pair_count": metrics.expected_associated_pair_count,
        "true_positive_pair_count": metrics.true_positive_pair_count,
        "false_positive_pair_count": metrics.false_positive_pair_count,
        "false_negative_pair_count": metrics.false_negative_pair_count,
        "ambiguous_pair_count": metrics.ambiguous_pair_count,
        "precision_millionths": metrics.precision_millionths,
        "recall_millionths": metrics.recall_millionths,
        "f1_millionths": metrics.f1_millionths,
        "association_coverage_millionths": metrics.association_coverage_millionths,
        "representative_measurement_status": metrics.representative_measurement_status,
        "production_eligible": metrics.production_eligible,
        "qualification_scope": "local-fixture-only; representative-association-quality-is-p15",
    }


def build_recording_association_fixture_metrics(
    *,
    fixture_id: str,
    report: RecordingAssociationReport,
    expected_clusters: Sequence[AssociationFixtureTruthCluster],
) -> RecordingAssociationFixtureMetrics:
    """Compare a replay-verified report with local fixture associations only."""

    checked = verify_recording_association_report(report)
    truth = AssociationFixtureTruth.create(
        fixture_id=fixture_id,
        source_clusters=expected_clusters,
    )
    report_keys = {item.source_action.source_action_logical_key for item in checked.inputs}
    truth_keys = {
        action.source_action_logical_key
        for cluster in truth.source_clusters
        for action in cluster.source_actions
    }
    if not truth_keys.issubset(report_keys):
        raise ValueError("fixture truth references a source action absent from the report")

    predicted_pairs = _cluster_pairs(tuple(cluster.source_actions for cluster in checked.clusters))
    expected_pairs = _cluster_pairs(
        tuple(cluster.source_actions for cluster in truth.source_clusters)
    )
    true_positive = len(predicted_pairs & expected_pairs)
    false_positive = len(predicted_pairs - expected_pairs)
    false_negative = len(expected_pairs - predicted_pairs)
    associated_keys = {
        action.source_action_logical_key
        for cluster in checked.clusters
        for action in cluster.source_actions
    }
    ambiguous_pair_count = sum(
        decision.disposition is AssociationPairDisposition.AMBIGUOUS
        for decision in checked.pair_decisions
    )
    values: dict[str, Any] = {
        "fixture_truth": truth,
        "report_logical_key": checked.logical_key,
        "report_semantic_sha256": checked.semantic_sha256,
        "policy_semantic_sha256": checked.policy.semantic_sha256,
        "input_count": len(checked.inputs),
        "associated_input_count": len(associated_keys),
        "predicted_associated_pair_count": len(predicted_pairs),
        "expected_associated_pair_count": len(expected_pairs),
        "true_positive_pair_count": true_positive,
        "false_positive_pair_count": false_positive,
        "false_negative_pair_count": false_negative,
        "ambiguous_pair_count": ambiguous_pair_count,
        "precision_millionths": _ratio_millionths(true_positive, len(predicted_pairs)),
        "recall_millionths": _ratio_millionths(true_positive, len(expected_pairs)),
        "f1_millionths": _f1_millionths(
            _ratio_millionths(true_positive, len(predicted_pairs)),
            _ratio_millionths(true_positive, len(expected_pairs)),
        ),
        "association_coverage_millionths": _ratio_millionths(
            len(associated_keys),
            len(checked.inputs),
        ),
        "representative_measurement_status": "NOT_MEASURED",
        "projection_version": RECORDING_ASSOCIATION_FIXTURE_METRICS_PROJECTION_VERSION,
        "production_eligible": False,
    }
    draft = RecordingAssociationFixtureMetrics.model_construct(
        semantic_sha256="0" * 64,
        logical_key=f"{RECORDING_ASSOCIATION_FIXTURE_METRICS_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
        **values,
    )
    digest = semantic_sha256(recording_association_fixture_metrics_projection(draft))
    return RecordingAssociationFixtureMetrics.model_validate(
        {
            **values,
            "semantic_sha256": digest,
            "logical_key": (
                f"{RECORDING_ASSOCIATION_FIXTURE_METRICS_LOGICAL_KEY_NAMESPACE}:{digest}"
            ),
        },
        strict=True,
    )


def _source_action_sort_key(item: AssociationSourceActionRef) -> tuple[str, str]:
    return item.source_action_logical_key, item.source_action_semantic_sha256


def _truth_cluster_sort_key(
    cluster: AssociationFixtureTruthCluster,
) -> tuple[str, ...]:
    return tuple(item.source_action_logical_key for item in cluster.source_actions)


def _cluster_pairs(
    clusters: Sequence[Sequence[AssociationSourceActionRef]],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for cluster in clusters:
        for left, right in combinations(cluster, 2):
            left_key = left.source_action_logical_key
            right_key = right.source_action_logical_key
            pairs.add((min(left_key, right_key), max(left_key, right_key)))
    return pairs


def _ratio_millionths(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return 1_000_000
    return (numerator * 1_000_000) // denominator


def _f1_millionths(precision_millionths: int, recall_millionths: int) -> int:
    denominator = precision_millionths + recall_millionths
    if denominator == 0:
        return 0
    return (2 * precision_millionths * recall_millionths) // denominator


__all__ = [
    "RECORDING_ASSOCIATION_FIXTURE_METRICS_LOGICAL_KEY_NAMESPACE",
    "RECORDING_ASSOCIATION_FIXTURE_METRICS_PROJECTION_VERSION",
    "RECORDING_ASSOCIATION_FIXTURE_TRUTH_LOGICAL_KEY_NAMESPACE",
    "RECORDING_ASSOCIATION_FIXTURE_TRUTH_PROJECTION_VERSION",
    "AssociationFixtureTruth",
    "AssociationFixtureTruthCluster",
    "RecordingAssociationFixtureMetrics",
    "build_recording_association_fixture_metrics",
    "recording_association_fixture_metrics_projection",
    "recording_association_fixture_truth_projection",
]
