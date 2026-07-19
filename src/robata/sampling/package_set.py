"""TemporalPackageSet management for split groups and reduction policies.

Implements :class:`PackageSetBuilder` which constructs immutable
:class:`TemporalPackageSet` records from a temporal window, a sampling plan,
and an alignment identifier.  The builder handles:

- ``split_group_id`` generation
- ``member_manifest_sha256`` computation
- ``reduction_policy_version`` management
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import canonical_json_bytes

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class TemporalPackageSetMember(StrictModel):
    """One ordered member of a :class:`TemporalPackageSet`."""

    package_id: NonEmptyString
    ordinal: NonNegativeInt
    part_count: NonNegativeInt
    requested_start_ns: Nanoseconds
    requested_end_ns: Nanoseconds
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    overlap_before_ns: Nanoseconds
    overlap_after_ns: Nanoseconds


class TemporalPackageSet(StrictModel):
    """Immutable set of one or more packages produced from a single window.

    When no split is required the set contains exactly one member.
    When a window exceeds frame-budget or provider limits, the set
    contains multiple overlapping members that share a ``split_group_id``
    and are reduced downstream under ``reduction_policy_version``.
    """

    schema_version: Annotated[str, Field(strict=True, pattern=r"^1\.0$")]
    package_set_id: NonEmptyString
    split_group_id: NonEmptyString
    mcap_id: NonEmptyString
    window_id: NonEmptyString
    camera_mapping_run_id: NonEmptyString | None
    alignment_id: NonEmptyString | None
    requested_start_ns: Nanoseconds
    requested_end_ns: Nanoseconds
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    split_reason: Annotated[
        str,
        StringConstraints(strict=True, pattern=r"^(FRAME_BUDGET|PROVIDER_LIMIT|NONE)$"),
    ]
    split_policy_version: SchemaVersion
    capability_snapshot_digest: Sha256Digest | None
    split_plan_digest: Sha256Digest
    members: tuple[TemporalPackageSetMember, ...]
    member_manifest_sha256: Sha256Digest
    reduction_policy_version: SchemaVersion


class SamplingPlan:
    """Duck-typed sampling plan accepted by :class:`PackageSetBuilder`.

    Expected attributes:
    - ``sampling_plan_id`` (str)
    - ``version`` (str)
    - ``strategy_by_camera`` (dict)
    - ``limits`` (dict)
    """


class PackageSetBuilder:
    """Build a :class:`TemporalPackageSet` from a window and sampling plan.

    The builder is deterministic: the same inputs always produce the same
    ``split_group_id`` and ``member_manifest_sha256``.
    """

    def __init__(self, reduction_policy_version: SchemaVersion) -> None:
        self._reduction_policy_version = reduction_policy_version

    @property
    def reduction_policy_version(self) -> SchemaVersion:
        """The versioned reduction policy used by this builder."""
        return self._reduction_policy_version

    def build_package_set(
        self,
        window: object,  # duck-typed TemporalWindow
        sampling_plan: SamplingPlan,
        alignment_id: str,
    ) -> TemporalPackageSet:
        """Construct an immutable :class:`TemporalPackageSet`.

        Args:
            window: A temporal window exposing ``window_id``, ``mcap_id``,
                ``camera_mapping_run_id``, ``requested_interval``,
                ``interval``, and optionally ``parent_window_id``.
            sampling_plan: The sampling plan used for this window.
            alignment_id: The alignment identifier for temporal projection.

        Returns:
            A validated :class:`TemporalPackageSet` with computed digests.
        """
        window_id = getattr(window, "window_id", "")
        mcap_id = getattr(window, "mcap_id", "")
        camera_mapping_run_id = getattr(window, "camera_mapping_run_id", None)
        requested = getattr(window, "requested_interval", None)
        effective = getattr(window, "interval", None)

        if not window_id or not mcap_id or requested is None or effective is None:
            raise ValueError("window must expose window_id, mcap_id, requested_interval, interval")

        # Derive deterministic split_group_id from window lineage
        split_group_id = _derive_split_group_id(
            window_id=window_id,
            mcap_id=mcap_id,
            alignment_id=alignment_id,
            sampling_plan_id=getattr(sampling_plan, "sampling_plan_id", ""),
            requested_interval=requested,
        )

        # Derive package_set_id from split_group_id and a stable counter seed
        package_set_id = _derive_package_set_id(
            split_group_id=split_group_id,
            window_id=window_id,
        )

        # Build members — for the skeleton we emit a single member
        member = TemporalPackageSetMember(
            package_id=f"pkg-{package_set_id}-0",
            ordinal=0,
            part_count=1,
            requested_start_ns=requested.start_ns,
            requested_end_ns=requested.end_ns,
            start_ns=effective.start_ns,
            end_ns=effective.end_ns,
            overlap_before_ns=0,
            overlap_after_ns=0,
        )
        members = (member,)

        # Compute member_manifest_sha256
        member_manifest_sha256 = _compute_member_manifest_sha256(members)

        # Compute split_plan_digest (excludes package IDs and digest outputs)
        split_plan_digest = _compute_split_plan_digest(
            window_id=window_id,
            split_reason="NONE",
            split_policy_version=getattr(sampling_plan, "version", "unknown"),
            capability_snapshot=None,
            members=members,
        )

        return TemporalPackageSet(
            schema_version="1.0",
            package_set_id=package_set_id,
            split_group_id=split_group_id,
            mcap_id=mcap_id,
            window_id=window_id,
            camera_mapping_run_id=camera_mapping_run_id,
            alignment_id=alignment_id,
            requested_start_ns=requested.start_ns,
            requested_end_ns=requested.end_ns,
            start_ns=effective.start_ns,
            end_ns=effective.end_ns,
            split_reason="NONE",
            split_policy_version=getattr(sampling_plan, "version", "unknown"),
            capability_snapshot_digest=None,
            split_plan_digest=split_plan_digest,
            members=members,
            member_manifest_sha256=member_manifest_sha256,
            reduction_policy_version=self._reduction_policy_version,
        )


def _derive_split_group_id(
    *,
    window_id: str,
    mcap_id: str,
    alignment_id: str,
    sampling_plan_id: str,
    requested_interval: NanosecondInterval,
) -> str:
    """Derive a deterministic split_group_id from window lineage."""
    preimage = canonical_json_bytes(
        {
            "window_id": window_id,
            "mcap_id": mcap_id,
            "alignment_id": alignment_id,
            "sampling_plan_id": sampling_plan_id,
            "requested_start_ns": str(requested_interval.start_ns),
            "requested_end_ns": str(requested_interval.end_ns),
        }
    )
    digest = hashlib.sha256(preimage).hexdigest()
    return f"sg-{digest[:32]}"


def _derive_package_set_id(*, split_group_id: str, window_id: str) -> str:
    """Derive a deterministic package_set_id."""
    preimage = canonical_json_bytes(
        {
            "split_group_id": split_group_id,
            "window_id": window_id,
            "ordinal": 0,
        }
    )
    digest = hashlib.sha256(preimage).hexdigest()
    return f"ps-{digest[:32]}"


def _compute_member_manifest_sha256(
    members: Sequence[TemporalPackageSetMember],
) -> Sha256Digest:
    """Hash the ordered member tuple (ordinal, bounds, overlap) without IDs.

    The projection excludes ``package_id`` and ``member_manifest_sha256``
    itself so that child identities do not depend on a parent hash that
    depends on the children.
    """
    projection = []
    for member in members:
        projection.append(
            {
                "ordinal": member.ordinal,
                "part_count": member.part_count,
                "requested_start_ns": str(member.requested_start_ns),
                "requested_end_ns": str(member.requested_end_ns),
                "start_ns": str(member.start_ns),
                "end_ns": str(member.end_ns),
                "overlap_before_ns": str(member.overlap_before_ns),
                "overlap_after_ns": str(member.overlap_after_ns),
            }
        )
    preimage = canonical_json_bytes(projection)
    return hashlib.sha256(preimage).hexdigest()


def _compute_split_plan_digest(
    *,
    window_id: str,
    split_reason: str,
    split_policy_version: str,
    capability_snapshot: str | None,
    members: Sequence[TemporalPackageSetMember],
) -> Sha256Digest:
    """Hash the split plan excluding package IDs and digest outputs."""
    member_coords = []
    for member in members:
        member_coords.append(
            {
                "ordinal": member.ordinal,
                "part_count": member.part_count,
                "requested_start_ns": str(member.requested_start_ns),
                "requested_end_ns": str(member.requested_end_ns),
                "start_ns": str(member.start_ns),
                "end_ns": str(member.end_ns),
                "overlap_before_ns": str(member.overlap_before_ns),
                "overlap_after_ns": str(member.overlap_after_ns),
            }
        )
    preimage = canonical_json_bytes(
        {
            "window_id": window_id,
            "split_reason": split_reason,
            "split_policy_version": split_policy_version,
            "capability_snapshot": capability_snapshot,
            "member_coords": member_coords,
        }
    )
    return hashlib.sha256(preimage).hexdigest()


__all__ = [
    "PackageSetBuilder",
    "TemporalPackageSet",
    "TemporalPackageSetMember",
]
