"""Provider-neutral temporal package and frame contracts.

The package-set contract is deliberately owned by the contracts layer.
Sampling plans may decide how a source interval is split, while provider-
specific rendering and call limits belong to InferenceInputPlan.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import (
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import Rfc3339Timestamp

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class SplitReason(StrEnum):
    """Provider-neutral reasons for splitting a temporal package set."""

    FRAME_BUDGET = "FRAME_BUDGET"
    NONE = "NONE"


class PackageLineage(StrictModel):
    """Semantic lineage required for run-independent package identity."""

    source_content_sha256: Sha256Digest
    window_semantic_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    sampling_plan_sha256: Sha256Digest


class TemporalPackageSetMember(StrictModel):
    """One materialized, ordered member of a temporal package set."""

    package_id: NonEmptyString
    ordinal: NonNegativeInt
    part_count: PositiveInt
    requested_start_ns: Nanoseconds
    requested_end_ns: Nanoseconds
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    overlap_before_ns: Nanoseconds = 0
    overlap_after_ns: Nanoseconds = 0
    package_semantic_content_sha256: Sha256Digest
    package_manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        if self.requested_start_ns >= self.requested_end_ns:
            raise ValueError("member requested interval must be nonempty")
        if self.start_ns >= self.end_ns:
            raise ValueError("member effective interval must be nonempty")
        if self.start_ns < self.requested_start_ns or self.end_ns > self.requested_end_ns:
            raise ValueError("member effective interval must be contained by requested interval")
        if self.overlap_before_ns < 0 or self.overlap_after_ns < 0:
            raise ValueError("member overlap values must be nonnegative")
        span = self.end_ns - self.start_ns
        if self.overlap_before_ns >= span or self.overlap_after_ns >= span:
            raise ValueError("member overlap must be strictly less than its effective span")
        if self.ordinal >= self.part_count:
            raise ValueError("member ordinal must be less than part_count")
        return self


class TemporalPackageSet(StrictModel):
    """Immutable provider-neutral package set after frame materialization."""

    schema_version: Literal["1.0"]
    package_set_id: NonEmptyString
    split_group_id: NonEmptyString
    mcap_id: NonEmptyString
    window_id: NonEmptyString
    camera_mapping_run_id: NonEmptyString
    alignment_id: NonEmptyString
    lineage: PackageLineage
    requested_start_ns: Nanoseconds
    requested_end_ns: Nanoseconds
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    split_reason: SplitReason
    split_policy_version: SchemaVersion
    split_plan_digest: Sha256Digest
    members: tuple[TemporalPackageSetMember, ...]
    member_manifest_sha256: Sha256Digest
    reduction_policy_version: SchemaVersion
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_package_set(self) -> Self:
        if self.requested_start_ns >= self.requested_end_ns:
            raise ValueError("package-set requested interval must be nonempty")
        if self.start_ns >= self.end_ns:
            raise ValueError("package-set effective interval must be nonempty")
        if self.start_ns < self.requested_start_ns or self.end_ns > self.requested_end_ns:
            raise ValueError(
                "package-set effective interval must be contained by requested interval"
            )
        if not self.members:
            raise ValueError("package set must contain at least one member")

        part_count = len(self.members)
        ordinals = tuple(member.ordinal for member in self.members)
        if ordinals != tuple(range(part_count)):
            raise ValueError("member ordinals must be consecutive and stored in order")
        if any(member.part_count != part_count for member in self.members):
            raise ValueError("every member part_count must equal the package-set size")
        if self.members[0].start_ns != self.start_ns:
            raise ValueError("first member must start at the package-set effective start")
        if self.members[-1].end_ns != self.end_ns:
            raise ValueError("last member must end at the package-set effective end")
        if self.members[0].overlap_before_ns != 0:
            raise ValueError("first member cannot have overlap_before_ns")
        if self.members[-1].overlap_after_ns != 0:
            raise ValueError("last member cannot have overlap_after_ns")

        for member in self.members:
            if member.start_ns < self.start_ns or member.end_ns > self.end_ns:
                raise ValueError("member effective interval must lie within the package set")
            if (
                member.requested_start_ns < self.requested_start_ns
                or member.requested_end_ns > self.requested_end_ns
            ):
                raise ValueError("member requested interval must lie within the package set")

        for previous, current in zip(self.members, self.members[1:], strict=False):
            if current.start_ns <= previous.start_ns or current.end_ns <= previous.end_ns:
                raise ValueError("member intervals must make strict temporal progress")
            if current.start_ns > previous.end_ns:
                raise ValueError("member effective intervals must cover without gaps")
            overlap = previous.end_ns - current.start_ns
            if previous.overlap_after_ns != overlap or current.overlap_before_ns != overlap:
                raise ValueError("adjacent member overlap metadata is inconsistent")

        expected_reason = SplitReason.NONE if part_count == 1 else SplitReason.FRAME_BUDGET
        if self.split_reason is not expected_reason:
            raise ValueError("split_reason must agree with the number of members")
        if part_count == 1 and (
            self.members[0].overlap_before_ns or self.members[0].overlap_after_ns
        ):
            raise ValueError("the unsplit member cannot have overlap")

        expected_split_digest = compute_split_plan_digest(
            lineage=self.lineage,
            split_reason=self.split_reason,
            split_policy_version=self.split_policy_version,
            members=self.members,
        )
        if self.split_plan_digest != expected_split_digest:
            raise ValueError("split_plan_digest does not match the package coordinates")
        expected_group = derive_split_group_id(
            lineage=self.lineage,
            requested_start_ns=self.requested_start_ns,
            requested_end_ns=self.requested_end_ns,
            start_ns=self.start_ns,
            end_ns=self.end_ns,
            split_plan_digest=self.split_plan_digest,
        )
        if self.split_group_id != expected_group:
            raise ValueError("split_group_id does not match semantic lineage")
        expected_manifest = compute_member_manifest_sha256(self.members)
        if self.member_manifest_sha256 != expected_manifest:
            raise ValueError("member_manifest_sha256 does not match materialized members")
        expected_package_set_id = derive_package_set_id(
            split_group_id=self.split_group_id,
            member_manifest_sha256=self.member_manifest_sha256,
            reduction_policy_version=self.reduction_policy_version,
        )
        if self.package_set_id != expected_package_set_id:
            raise ValueError("package_set_id does not match the package manifest")
        return self


def member_coordinate(member: TemporalPackageSetMember) -> dict[str, int | str]:
    """Return the coordinate projection used by split identity."""

    return {
        "ordinal": member.ordinal,
        "part_count": member.part_count,
        "requested_start_ns": str(member.requested_start_ns),
        "requested_end_ns": str(member.requested_end_ns),
        "start_ns": str(member.start_ns),
        "end_ns": str(member.end_ns),
        "overlap_before_ns": str(member.overlap_before_ns),
        "overlap_after_ns": str(member.overlap_after_ns),
    }


def compute_split_plan_digest(
    *,
    lineage: PackageLineage,
    split_reason: SplitReason,
    split_policy_version: str,
    members: Sequence[TemporalPackageSetMember],
) -> Sha256Digest:
    """Hash source/policy lineage and ordered coordinates, excluding row IDs."""

    return semantic_sha256(
        {
            "lineage": lineage,
            "split_reason": split_reason.value,
            "split_policy_version": split_policy_version,
            "member_coordinates": [member_coordinate(member) for member in members],
        }
    )


def compute_member_manifest_sha256(
    members: Sequence[TemporalPackageSetMember],
) -> Sha256Digest:
    """Hash ordered materialized content and coordinates, excluding package row IDs."""

    return semantic_sha256(
        [
            {
                "coordinate": member_coordinate(member),
                "package_semantic_content_sha256": member.package_semantic_content_sha256,
                "package_manifest_sha256": member.package_manifest_sha256,
            }
            for member in members
        ]
    )


def derive_split_group_id(
    *,
    lineage: PackageLineage,
    requested_start_ns: int,
    requested_end_ns: int,
    start_ns: int,
    end_ns: int,
    split_plan_digest: str,
) -> str:
    """Derive a run-independent split-group identifier."""

    digest = semantic_sha256(
        {
            "lineage": lineage,
            "requested_start_ns": str(requested_start_ns),
            "requested_end_ns": str(requested_end_ns),
            "start_ns": str(start_ns),
            "end_ns": str(end_ns),
            "split_plan_digest": split_plan_digest,
        }
    )
    return str(uuid5(NAMESPACE_URL, f"robata:split-group:{digest}"))


def derive_package_set_id(
    *,
    split_group_id: str,
    member_manifest_sha256: str,
    reduction_policy_version: str,
) -> str:
    """Derive a package-set identifier from semantic content, not row IDs."""

    digest = semantic_sha256(
        {
            "split_group_id": split_group_id,
            "member_manifest_sha256": member_manifest_sha256,
            "reduction_policy_version": reduction_policy_version,
        }
    )
    return str(uuid5(NAMESPACE_URL, f"robata:package-set:{digest}"))


class FrameSelectionManifest(StrictModel):
    """One selected frame within a temporal package."""

    frame_id: NonEmptyString
    alignment_projection_id: NonEmptyString
    ordinal: NonNegativeInt
    aligned_timestamp_ns: Nanoseconds
    source_timestamp_ns: Nanoseconds
    delta_to_target_ns: Nanoseconds
    source_locator: dict[str, object]
    materialized_artifact: dict[str, object] | None = None
    width: PositiveInt
    height: PositiveInt
    quality_flags: tuple[NonEmptyString, ...] = ()


class CameraSamplingSummary(StrictModel):
    """Sampling metadata for one camera within a package."""

    strategy: NonEmptyString
    target_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    actual_fps: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
    target_count: NonNegativeInt
    actual_count: NonNegativeInt
    missed_targets: NonNegativeInt
    trigger_features: tuple[NonEmptyString, ...] = ()


class TemporalPackageCameraEntry(StrictModel):
    """One camera entry within a temporal visual package."""

    status: NonEmptyString
    stream_id: NonEmptyString | None = None
    frames: tuple[FrameSelectionManifest, ...] = ()
    sampling: CameraSamplingSummary | None = None
    missing_reason: NonEmptyString | None = None


__all__ = [
    "CameraSamplingSummary",
    "FrameSelectionManifest",
    "PackageLineage",
    "SplitReason",
    "TemporalPackageCameraEntry",
    "TemporalPackageSet",
    "TemporalPackageSetMember",
    "compute_member_manifest_sha256",
    "compute_split_plan_digest",
    "derive_package_set_id",
    "derive_split_group_id",
    "member_coordinate",
]
