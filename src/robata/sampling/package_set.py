"""Deterministic provider-neutral temporal package-set planning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.pipeline import SamplingPurpose
from robata.contracts.sampling_plan import SamplingPlan
from robata.contracts.temporal import (
    PackageLineage,
    SplitReason,
    TemporalPackageSet,
    TemporalPackageSetMember,
    compute_member_manifest_sha256,
    compute_split_plan_digest,
    derive_package_set_id,
    derive_split_group_id,
)
from robata.sampling.dense import DenseSplitPolicy, IntervalPart, plan_interval_parts

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class MaterializedPackageRef(StrictModel):
    """Immutable content identity supplied after frame materialization."""

    ordinal: NonNegativeInt
    package_id: NonEmptyString
    package_semantic_content_sha256: Sha256Digest
    package_manifest_sha256: Sha256Digest


class PackageSetBuilder:
    """Build a package set only after materialized member content is known."""

    def __init__(
        self,
        reduction_policy_version: SchemaVersion,
        split_policy: DenseSplitPolicy | None = None,
    ) -> None:
        if not isinstance(reduction_policy_version, str) or not reduction_policy_version:
            raise ValueError("reduction_policy_version must be a nonempty string")
        self._reduction_policy_version = reduction_policy_version
        self._split_policy = split_policy

    @property
    def reduction_policy_version(self) -> SchemaVersion:
        return self._reduction_policy_version

    @property
    def split_policy(self) -> DenseSplitPolicy | None:
        return self._split_policy

    def plan_parts(
        self,
        window: object,
        sampling_plan: SamplingPlan,
    ) -> tuple[IntervalPart, ...]:
        """Return provider-neutral coordinates before materialization."""

        requested, effective = _window_intervals(window)
        policy = self._policy(sampling_plan)
        return plan_interval_parts(
            requested,
            effective,
            sampling_plan,
            overlap_ns=policy.overlap_ns,
            split_policy=policy,
            purpose=_window_sampling_purpose(window),
        )

    def build_package_set(
        self,
        window: object,
        sampling_plan: SamplingPlan,
        alignment_id: str,
        *,
        lineage: PackageLineage | None = None,
        materialized_members: Sequence[MaterializedPackageRef] | None = None,
        created_at: str | None = None,
    ) -> TemporalPackageSet:
        """Construct a validated set from materialized package identities.

        A split plan without selected frames is not a package set.  Callers
        must provide the semantic and exact manifest digests produced by the
        frame materializer; coordinate-only placeholders are rejected.
        """

        if not isinstance(sampling_plan, SamplingPlan):
            raise TypeError("sampling_plan must be a SamplingPlan")
        if not isinstance(alignment_id, str) or not alignment_id:
            raise ValueError("alignment_id must be a nonempty string")
        if lineage is None:
            raise ValueError("package lineage is required after materialization")
        if not isinstance(lineage, PackageLineage):
            raise TypeError("lineage must be a PackageLineage")
        if materialized_members is None:
            raise ValueError("materialized package identities are required")
        if created_at is None:
            raise ValueError("created_at is required for a durable package set")

        window_id, mcap_id, mapping_id, requested, effective = _window_fields(window)
        if mapping_id is None:
            raise ValueError("window must expose a camera_mapping_run_id")
        parts = self.plan_parts(window, sampling_plan)
        refs = tuple(materialized_members)
        if len(refs) != len(parts):
            raise ValueError("materialized member count must equal planned part count")
        if tuple(ref.ordinal for ref in refs) != tuple(range(len(refs))):
            raise ValueError("materialized member ordinals must be consecutive and ordered")

        split_reason = SplitReason.NONE if len(parts) == 1 else SplitReason.FRAME_BUDGET
        members = tuple(_member_from_part(ref, part) for ref, part in zip(refs, parts, strict=True))
        split_policy = self._policy(sampling_plan)
        split_plan_digest = compute_split_plan_digest(
            lineage=lineage,
            split_reason=split_reason,
            split_policy_version=split_policy.version,
            members=members,
        )
        split_group_id = derive_split_group_id(
            lineage=lineage,
            requested_start_ns=requested.start_ns,
            requested_end_ns=requested.end_ns,
            start_ns=effective.start_ns,
            end_ns=effective.end_ns,
            split_plan_digest=split_plan_digest,
        )
        member_manifest_sha256 = compute_member_manifest_sha256(members)
        package_set_id = derive_package_set_id(
            split_group_id=split_group_id,
            member_manifest_sha256=member_manifest_sha256,
            reduction_policy_version=self._reduction_policy_version,
        )
        return TemporalPackageSet(
            schema_version="1.0",
            package_set_id=package_set_id,
            split_group_id=split_group_id,
            mcap_id=mcap_id,
            window_id=window_id,
            camera_mapping_run_id=mapping_id,
            alignment_id=alignment_id,
            lineage=lineage,
            requested_start_ns=requested.start_ns,
            requested_end_ns=requested.end_ns,
            start_ns=effective.start_ns,
            end_ns=effective.end_ns,
            split_reason=split_reason,
            split_policy_version=split_policy.version,
            split_plan_digest=split_plan_digest,
            members=members,
            member_manifest_sha256=member_manifest_sha256,
            reduction_policy_version=self._reduction_policy_version,
            created_at=created_at,
        )

    def _policy(self, sampling_plan: SamplingPlan) -> DenseSplitPolicy:
        return self._split_policy or DenseSplitPolicy(
            version=sampling_plan.version,
            overlap_ns=0,
        )


def _window_fields(
    window: object,
) -> tuple[str, str, str | None, NanosecondInterval, NanosecondInterval]:
    window_id = getattr(window, "window_id", None)
    mcap_id = getattr(window, "mcap_id", None)
    mapping_id = getattr(window, "camera_mapping_run_id", None)
    requested = getattr(window, "requested_interval", None)
    effective = getattr(window, "interval", None)
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("window must expose a nonempty window_id")
    if not isinstance(mcap_id, str) or not mcap_id:
        raise ValueError("window must expose a nonempty mcap_id")
    if mapping_id is not None and (not isinstance(mapping_id, str) or not mapping_id):
        raise ValueError("camera_mapping_run_id must be null or a nonempty string")
    if not isinstance(requested, NanosecondInterval) or not isinstance(
        effective, NanosecondInterval
    ):
        raise ValueError("window must expose requested_interval and interval contracts")
    return window_id, mcap_id, mapping_id, requested, effective


def _window_intervals(window: object) -> tuple[NanosecondInterval, NanosecondInterval]:
    _, _, _, requested, effective = _window_fields(window)
    return requested, effective


def _window_sampling_purpose(window: object) -> SamplingPurpose:
    purpose = getattr(window, "purpose", SamplingPurpose.ACTION_DENSE)
    if not isinstance(purpose, SamplingPurpose):
        raise TypeError("window purpose must be a SamplingPurpose")
    return purpose


def _member_from_part(
    ref: MaterializedPackageRef,
    part: IntervalPart,
) -> TemporalPackageSetMember:
    return TemporalPackageSetMember(
        package_id=ref.package_id,
        ordinal=part.ordinal,
        part_count=part.part_count,
        requested_start_ns=part.requested_interval.start_ns,
        requested_end_ns=part.requested_interval.end_ns,
        start_ns=part.effective_interval.start_ns,
        end_ns=part.effective_interval.end_ns,
        overlap_before_ns=part.overlap_before_ns,
        overlap_after_ns=part.overlap_after_ns,
        package_semantic_content_sha256=ref.package_semantic_content_sha256,
        package_manifest_sha256=ref.package_manifest_sha256,
    )


def sampling_plan_digest(
    sampling_plan: SamplingPlan,
    *,
    purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
) -> Sha256Digest:
    """Return a canonical digest suitable for PackageLineage."""

    from robata.sampling.dense import sampling_plan_projection

    return semantic_sha256(sampling_plan_projection(sampling_plan, purpose=purpose))


__all__ = [
    "MaterializedPackageRef",
    "PackageLineage",
    "PackageSetBuilder",
    "SplitReason",
    "TemporalPackageSet",
    "TemporalPackageSetMember",
    "sampling_plan_digest",
]
