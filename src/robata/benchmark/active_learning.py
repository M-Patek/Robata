"""Leakage-safe local metrics for immutable P13 review selection.

The selector pool and held-out evaluation population remain separate.  This
module freezes capture/camera/time connected groups and reports local fixture
metrics only; it never changes routing, completion, or model promotion.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.review.active_learning import (
    ActiveLearningSelectionDecision,
    verify_active_learning_selection_decision,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
MetricMillionths = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

ACTIVE_LEARNING_SPLIT_PROJECTION_VERSION: Final = "active-learning-split-semantic-v1"
ACTIVE_LEARNING_METRICS_PROJECTION_VERSION: Final = "active-learning-fixture-metrics-semantic-v1"
ACTIVE_LEARNING_SPLIT_KEY_NAMESPACE: Final = "active-learning-split-v1"
ACTIVE_LEARNING_METRICS_KEY_NAMESPACE: Final = "active-learning-fixture-metrics-v1"


class ActiveLearningSplit(StrEnum):
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    FROZEN_TEST = "frozen_test"


class ActiveLearningSplitRecord(StrictModel):
    """Required connected-group metadata for one recording."""

    mcap_id: OpaqueUuid
    recording_identity: Sha256Digest
    capture_group: NonEmptyString
    camera_group: NonEmptyString
    time_group: NonEmptyString


class ActiveLearningSplitProtocol(StrictModel):
    """Content-addressed transitive capture/camera/time split assignment."""

    split_version: NonEmptyString
    records: tuple[ActiveLearningSplitRecord, ...]
    assignments: dict[OpaqueUuid, ActiveLearningSplit]
    leakage_group_ids: dict[OpaqueUuid, NonEmptyString]
    projection_version: Literal["active-learning-split-semantic-v1"] = (
        ACTIVE_LEARNING_SPLIT_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        split_version: str,
        records: Sequence[ActiveLearningSplitRecord],
        assignments: Mapping[str, ActiveLearningSplit | str],
    ) -> Self:
        ordered_records = tuple(sorted(records, key=lambda item: item.mcap_id))
        values: dict[str, object] = {
            "split_version": split_version,
            "records": ordered_records,
            "assignments": dict(
                sorted(
                    (mcap_id, ActiveLearningSplit(value)) for mcap_id, value in assignments.items()
                )
            ),
            "leakage_group_ids": _connected_group_ids(ordered_records),
            "projection_version": ACTIVE_LEARNING_SPLIT_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = cls.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{ACTIVE_LEARNING_SPLIT_KEY_NAMESPACE}:{'0' * 64}",
            split_version=split_version,
            records=ordered_records,
            assignments=dict(
                sorted(
                    (mcap_id, ActiveLearningSplit(value)) for mcap_id, value in assignments.items()
                )
            ),
            leakage_group_ids=_connected_group_ids(ordered_records),
            projection_version=ACTIVE_LEARNING_SPLIT_PROJECTION_VERSION,
            production_eligible=False,
        )
        digest = semantic_sha256(active_learning_split_protocol_projection(draft))
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{ACTIVE_LEARNING_SPLIT_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        if not self.records:
            raise ValueError("active-learning split requires at least one record")
        if self.records != tuple(sorted(self.records, key=lambda item: item.mcap_id)):
            raise ValueError("active-learning split records must be canonically ordered")
        mcap_ids = tuple(item.mcap_id for item in self.records)
        if len(mcap_ids) != len(set(mcap_ids)):
            raise ValueError("active-learning split MCAP IDs must be unique")
        if set(self.assignments) != set(mcap_ids):
            raise ValueError("active-learning split must assign every record exactly once")
        if dict(sorted(self.assignments.items())) != self.assignments:
            raise ValueError("active-learning split assignments must be canonically ordered")
        if self.leakage_group_ids != _connected_group_ids(self.records):
            raise ValueError("active-learning split leakage groups do not reproduce")
        group_roles: defaultdict[str, set[ActiveLearningSplit]] = defaultdict(set)
        for mcap_id, group_id in self.leakage_group_ids.items():
            group_roles[group_id].add(self.assignments[mcap_id])
        if any(len(roles) != 1 for roles in group_roles.values()):
            raise ValueError("capture/camera/time connected groups may not cross split roles")
        digest = semantic_sha256(active_learning_split_protocol_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("active-learning split semantic identity is inconsistent")
        if self.logical_key != f"{ACTIVE_LEARNING_SPLIT_KEY_NAMESPACE}:{digest}":
            raise ValueError("active-learning split logical key is inconsistent")
        return self

    def role_for(self, mcap_id: str) -> ActiveLearningSplit:
        try:
            return self.assignments[mcap_id]
        except KeyError as error:
            raise ValueError("MCAP is absent from the active-learning split") from error


class ActiveLearningPoolObservation(StrictModel):
    """One selector-pool candidate with optional late annotation outcome."""

    review_task_id: OpaqueUuid
    review_task_semantic_sha256: Sha256Digest
    mcap_id: OpaqueUuid
    recording_identity: Sha256Digest
    subgroup: NonEmptyString
    annotation_arrived: bool
    agreement_millionths: MetricMillionths | None = None
    yield_positive: bool | None = None

    @model_validator(mode="after")
    def validate_annotation_values(self) -> Self:
        values = (self.agreement_millionths, self.yield_positive)
        if self.annotation_arrived != all(value is not None for value in values):
            raise ValueError("annotation outcomes must be complete when annotation has arrived")
        return self


class ActiveLearningHeldOutObservation(StrictModel):
    """Frozen-test outcome, never a selected review task."""

    mcap_id: OpaqueUuid
    subgroup: NonEmptyString
    agreement_millionths: MetricMillionths
    yield_positive: bool


class ActiveLearningSubgroupMetrics(StrictModel):
    subgroup: NonEmptyString
    pool_count: NonNegativeInt
    selected_count: NonNegativeInt
    annotation_count: NonNegativeInt
    selection_coverage_millionths: MetricMillionths
    selection_bias_delta_millionths: int

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.selected_count > self.pool_count or self.annotation_count > self.selected_count:
            raise ValueError("subgroup selection counts are inconsistent")
        if self.selection_coverage_millionths != _ratio_millionths(
            self.selected_count, self.pool_count
        ):
            raise ValueError("subgroup selection coverage does not match counts")
        return self


class ActiveLearningFixtureMetrics(StrictModel):
    """Content-addressed local selection, agreement, coverage, and yield report."""

    split_protocol: ActiveLearningSplitProtocol
    decision_logical_key: NodeLogicalKey
    decision_semantic_sha256: Sha256Digest
    pool_observations: tuple[ActiveLearningPoolObservation, ...]
    held_out_observations: tuple[ActiveLearningHeldOutObservation, ...]
    pool_count: NonNegativeInt
    selected_count: NonNegativeInt
    annotation_count: NonNegativeInt
    selection_coverage_millionths: MetricMillionths
    selected_annotation_yield_millionths: MetricMillionths | None
    selected_annotation_agreement_millionths: MetricMillionths | None
    frozen_held_out_count: NonNegativeInt
    frozen_held_out_yield_millionths: MetricMillionths | None
    frozen_held_out_agreement_millionths: MetricMillionths | None
    subgroup_metrics: tuple[ActiveLearningSubgroupMetrics, ...]
    representative_measurement_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    projection_version: Literal["active-learning-fixture-metrics-semantic-v1"] = (
        ACTIVE_LEARNING_METRICS_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.decision_logical_key.rsplit(":", 1)[-1] != self.decision_semantic_sha256:
            raise ValueError("fixture metrics decision logical key is inconsistent")
        if self.pool_count != len(self.pool_observations):
            raise ValueError("pool count must match pool observations")
        if self.selected_count > self.pool_count or self.annotation_count > self.selected_count:
            raise ValueError("selection counts are inconsistent")
        if self.selection_coverage_millionths != _ratio_millionths(
            self.selected_count, self.pool_count
        ):
            raise ValueError("selection coverage does not match counts")
        selected_values = (
            self.selected_annotation_yield_millionths,
            self.selected_annotation_agreement_millionths,
        )
        if (self.annotation_count > 0) != all(value is not None for value in selected_values):
            raise ValueError("selected annotation metrics must match the annotation population")
        held_out_values = (
            self.frozen_held_out_yield_millionths,
            self.frozen_held_out_agreement_millionths,
        )
        if (self.frozen_held_out_count > 0) != all(value is not None for value in held_out_values):
            raise ValueError("held-out metrics must match the frozen-test population")
        if self.subgroup_metrics != tuple(
            sorted(self.subgroup_metrics, key=lambda item: item.subgroup)
        ):
            raise ValueError("subgroup metrics must be canonically ordered")
        digest = semantic_sha256(active_learning_fixture_metrics_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("active-learning fixture metrics semantic identity is inconsistent")
        if self.logical_key != f"{ACTIVE_LEARNING_METRICS_KEY_NAMESPACE}:{digest}":
            raise ValueError("active-learning fixture metrics logical key is inconsistent")
        return self


def active_learning_split_protocol_projection(
    protocol: ActiveLearningSplitProtocol,
) -> dict[str, object]:
    return {
        "semantic_projection_version": protocol.projection_version,
        "split_version": protocol.split_version,
        "records": [item.model_dump(mode="json") for item in protocol.records],
        "assignments": {mcap_id: role.value for mcap_id, role in protocol.assignments.items()},
        "leakage_group_ids": protocol.leakage_group_ids,
        "production_eligible": protocol.production_eligible,
        "leakage_policy": "recording-capture-camera-time-transitive-closure",
    }


def active_learning_fixture_metrics_projection(
    metrics: ActiveLearningFixtureMetrics,
) -> dict[str, object]:
    return {
        "semantic_projection_version": metrics.projection_version,
        "split_protocol": active_learning_split_protocol_projection(metrics.split_protocol),
        "split_protocol_semantic_sha256": metrics.split_protocol.semantic_sha256,
        "decision_logical_key": metrics.decision_logical_key,
        "decision_semantic_sha256": metrics.decision_semantic_sha256,
        "pool_observations": [item.model_dump(mode="json") for item in metrics.pool_observations],
        "held_out_observations": [
            item.model_dump(mode="json") for item in metrics.held_out_observations
        ],
        "pool_count": metrics.pool_count,
        "selected_count": metrics.selected_count,
        "annotation_count": metrics.annotation_count,
        "selection_coverage_millionths": metrics.selection_coverage_millionths,
        "selected_annotation_yield_millionths": metrics.selected_annotation_yield_millionths,
        "selected_annotation_agreement_millionths": (
            metrics.selected_annotation_agreement_millionths
        ),
        "frozen_held_out_count": metrics.frozen_held_out_count,
        "frozen_held_out_yield_millionths": metrics.frozen_held_out_yield_millionths,
        "frozen_held_out_agreement_millionths": (metrics.frozen_held_out_agreement_millionths),
        "subgroup_metrics": [item.model_dump(mode="json") for item in metrics.subgroup_metrics],
        "representative_measurement_status": metrics.representative_measurement_status,
        "production_eligible": metrics.production_eligible,
        "qualification_scope": "local-fixture-only; grouped-held-out-review-yield-is-p15",
    }


def build_active_learning_fixture_metrics(
    *,
    decision: ActiveLearningSelectionDecision,
    split_protocol: ActiveLearningSplitProtocol,
    pool_observations: Sequence[ActiveLearningPoolObservation],
    held_out_observations: Sequence[ActiveLearningHeldOutObservation],
) -> ActiveLearningFixtureMetrics:
    """Build a local report while enforcing selector/frozen-test separation."""

    checked_decision = verify_active_learning_selection_decision(decision)
    candidates = {
        item.candidate.review_task_id: item.candidate
        for item in checked_decision.candidate_decisions
    }
    selected_ids = set(checked_decision.selected_review_task_ids)
    pool = tuple(sorted(pool_observations, key=lambda item: item.review_task_id))
    if {item.review_task_id for item in pool} != set(candidates) or len(pool) != len(candidates):
        raise ValueError(
            "pool observations must cover every frozen decision candidate exactly once"
        )
    for item in pool:
        candidate = candidates[item.review_task_id]
        if candidate.review_task_semantic_sha256 != item.review_task_semantic_sha256:
            raise ValueError("pool observation task digest differs from frozen candidate")
        if candidate.recording_identity != item.recording_identity:
            raise ValueError("pool observation recording identity differs from frozen candidate")
        if split_protocol.role_for(item.mcap_id) is ActiveLearningSplit.FROZEN_TEST:
            raise ValueError("frozen-test records may not enter the active-learning selection pool")
    held_out = tuple(sorted(held_out_observations, key=lambda item: (item.mcap_id, item.subgroup)))
    if any(
        split_protocol.role_for(item.mcap_id) is not ActiveLearningSplit.FROZEN_TEST
        for item in held_out
    ):
        raise ValueError("held-out observations must be assigned to frozen_test")
    selected = tuple(item for item in pool if item.review_task_id in selected_ids)
    annotated = tuple(item for item in selected if item.annotation_arrived)
    coverage = _ratio_millionths(len(selected), len(pool))
    groups: defaultdict[str, list[ActiveLearningPoolObservation]] = defaultdict(list)
    for item in pool:
        groups[item.subgroup].append(item)
    subgroup_metrics = tuple(
        _subgroup_metrics(name, rows, selected_ids=selected_ids, overall_rate=coverage)
        for name, rows in sorted(groups.items())
    )
    values: dict[str, Any] = {
        "split_protocol": split_protocol,
        "decision_logical_key": checked_decision.logical_key,
        "decision_semantic_sha256": checked_decision.semantic_sha256,
        "pool_observations": pool,
        "held_out_observations": held_out,
        "pool_count": len(pool),
        "selected_count": len(selected),
        "annotation_count": len(annotated),
        "selection_coverage_millionths": coverage,
        "selected_annotation_yield_millionths": _positive_rate(annotated),
        "selected_annotation_agreement_millionths": _mean_agreement(annotated),
        "frozen_held_out_count": len(held_out),
        "frozen_held_out_yield_millionths": _positive_rate(held_out),
        "frozen_held_out_agreement_millionths": _mean_agreement(held_out),
        "subgroup_metrics": subgroup_metrics,
        "representative_measurement_status": "NOT_MEASURED",
        "projection_version": ACTIVE_LEARNING_METRICS_PROJECTION_VERSION,
        "production_eligible": False,
    }
    draft = ActiveLearningFixtureMetrics.model_construct(
        semantic_sha256="0" * 64,
        logical_key=f"{ACTIVE_LEARNING_METRICS_KEY_NAMESPACE}:{'0' * 64}",
        **values,
    )
    digest = semantic_sha256(active_learning_fixture_metrics_projection(draft))
    return ActiveLearningFixtureMetrics.model_validate(
        {
            **values,
            "semantic_sha256": digest,
            "logical_key": f"{ACTIVE_LEARNING_METRICS_KEY_NAMESPACE}:{digest}",
        },
        strict=True,
    )


def _connected_group_ids(records: Sequence[ActiveLearningSplitRecord]) -> dict[str, str]:
    parent = {record.mcap_id: record.mcap_id for record in records}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    seen: dict[tuple[str, str], str] = {}
    for record in records:
        for dimension, value in (
            ("recording", record.recording_identity),
            ("capture", record.capture_group),
            ("camera", record.camera_group),
            ("time", record.time_group),
        ):
            union(record.mcap_id, seen.setdefault((dimension, value), record.mcap_id))
    members: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        members[find(record.mcap_id)].append(record.mcap_id)
    result: dict[str, str] = {}
    for member_ids in members.values():
        ordered = tuple(sorted(member_ids))
        group_id = f"review-selection-group:{semantic_sha256({'mcap_ids': ordered})}"
        result.update({mcap_id: group_id for mcap_id in ordered})
    return dict(sorted(result.items()))


def _subgroup_metrics(
    subgroup: str,
    rows: Sequence[ActiveLearningPoolObservation],
    *,
    selected_ids: set[str],
    overall_rate: int,
) -> ActiveLearningSubgroupMetrics:
    selected = tuple(item for item in rows if item.review_task_id in selected_ids)
    return ActiveLearningSubgroupMetrics(
        subgroup=subgroup,
        pool_count=len(rows),
        selected_count=len(selected),
        annotation_count=sum(item.annotation_arrived is True for item in selected),
        selection_coverage_millionths=_ratio_millionths(len(selected), len(rows)),
        selection_bias_delta_millionths=_ratio_millionths(len(selected), len(rows)) - overall_rate,
    )


def _ratio_millionths(numerator: int, denominator: int) -> int:
    return 0 if denominator == 0 else (numerator * 1_000_000) // denominator


def _positive_rate(
    observations: Sequence[ActiveLearningPoolObservation | ActiveLearningHeldOutObservation],
) -> int | None:
    if not observations:
        return None
    values = [item.yield_positive for item in observations]
    if any(value is None for value in values):
        raise ValueError("reported observations require complete positive outcomes")
    return _ratio_millionths(sum(value is True for value in values), len(values))


def _mean_agreement(
    observations: Sequence[ActiveLearningPoolObservation | ActiveLearningHeldOutObservation],
) -> int | None:
    if not observations:
        return None
    values = [item.agreement_millionths for item in observations]
    if any(value is None for value in values):
        raise ValueError("reported observations require complete agreement values")
    return sum(value for value in values if value is not None) // len(values)


__all__ = [
    "ACTIVE_LEARNING_METRICS_KEY_NAMESPACE",
    "ACTIVE_LEARNING_METRICS_PROJECTION_VERSION",
    "ACTIVE_LEARNING_SPLIT_KEY_NAMESPACE",
    "ACTIVE_LEARNING_SPLIT_PROJECTION_VERSION",
    "ActiveLearningFixtureMetrics",
    "ActiveLearningHeldOutObservation",
    "ActiveLearningPoolObservation",
    "ActiveLearningSplit",
    "ActiveLearningSplitProtocol",
    "ActiveLearningSplitRecord",
    "ActiveLearningSubgroupMetrics",
    "active_learning_fixture_metrics_projection",
    "active_learning_split_protocol_projection",
    "build_active_learning_fixture_metrics",
]
