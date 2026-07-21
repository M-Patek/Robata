"""Role-bound canonical windows for provisional-action boundary refinement."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.models import (
    CanonicalOfflineConfigurationError,
    CanonicalRootWindow,
    _strict_context,
)
from robata.application.canonical.projections import (
    _stable_uuid,
    canonical_root_window_projection_values,
)
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid, Rfc3339Timestamp
from robata.contracts.pipeline import SamplingPurpose
from robata.contracts.sampling_plan import SamplingPlan
from robata.contracts.temporal import PackageLineage
from robata.event_pipeline.boundary_refinement import BoundaryRefinementRole
from robata.event_pipeline.provisional_fusion import ProvisionalPhysicalAction
from robata.sampling.package_set import sampling_plan_digest

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
BOUNDARY_WINDOW_PROJECTION_VERSION = "boundary-refinement-window-semantic-v1"
BOUNDARY_WINDOW_UUID_NAMESPACE = "canonical-boundary-refinement-window-v1"


def boundary_refinement_window_projection(
    window: CanonicalBoundaryRefinementWindow,
) -> dict[str, object]:
    """Return the run-independent identity projection for one role window."""

    return _boundary_refinement_window_projection_values(
        {
            "source_content_sha256": window.source_content_sha256,
            "camera_mapping_semantic_sha256": window.camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": window.alignment_semantic_sha256,
            "requested_interval": window.requested_interval,
            "interval": window.interval,
            "purpose": window.purpose,
            "window_policy_version": window.window_policy_version,
            "source_subject_type": window.source_subject_type,
            "source_subject_logical_key": window.source_subject_logical_key,
            "parent_window_logical_key": window.parent_window_logical_key,
            "source_lineage_sha256": window.source_lineage_sha256,
            "refinement_role": window.refinement_role,
            "generation": window.generation,
            "provisional_action_semantic_sha256": (window.provisional_action_semantic_sha256),
            "provisional_fusion_semantic_sha256": (window.provisional_fusion_semantic_sha256),
            "coarse_interval": window.coarse_interval,
            "boundary_anchor_ns": window.boundary_anchor_ns,
            "padding_before_ns": window.padding_before_ns,
            "padding_after_ns": window.padding_after_ns,
            "context_truncated": window.context_truncated,
            "production_eligible": window.production_eligible,
        }
    )


def _boundary_refinement_window_projection_values(
    values: dict[str, object],
) -> dict[str, object]:
    coarse = values["coarse_interval"]
    role = values["refinement_role"]
    if not isinstance(coarse, NanosecondInterval):
        raise TypeError("boundary coarse interval must be a NanosecondInterval")
    if not isinstance(role, BoundaryRefinementRole):
        raise TypeError("boundary refinement role is invalid")
    return {
        "semantic_projection_version": BOUNDARY_WINDOW_PROJECTION_VERSION,
        **canonical_root_window_projection_values(values),
        "refinement_role": role.value,
        "provisional_action_semantic_sha256": values["provisional_action_semantic_sha256"],
        "provisional_fusion_semantic_sha256": values["provisional_fusion_semantic_sha256"],
        "coarse_interval": coarse.model_dump(mode="json"),
        "boundary_anchor_ns": str(values["boundary_anchor_ns"]),
        "padding_before_ns": str(values["padding_before_ns"]),
        "padding_after_ns": str(values["padding_after_ns"]),
        "context_truncated": values["context_truncated"],
        "production_eligible": values["production_eligible"],
    }


class CanonicalBoundaryRefinementWindow(StrictModel):
    """One ONSET or OFFSET dense window bound to a provisional action."""

    schema_version: Literal["1.0"]
    window_id: OpaqueUuid
    window_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    mcap_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid
    alignment_id: OpaqueUuid
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    recording_duration_ns: PositiveInt
    reference_timebase: Literal["recording_relative_ns"]
    provisional_action_logical_key: NodeLogicalKey
    provisional_action_semantic_sha256: Sha256Digest
    provisional_fusion_semantic_sha256: Sha256Digest
    coarse_interval: NanosecondInterval
    boundary_anchor_ns: Nanoseconds
    padding_before_ns: NonNegativeInt
    padding_after_ns: PositiveInt
    requested_interval: NanosecondInterval
    interval: NanosecondInterval
    context_truncated: bool
    purpose: Literal[SamplingPurpose.BOUNDARY_REFINEMENT]
    window_policy_version: SchemaVersion
    source_subject_type: Literal["PROVISIONAL_PHYSICAL_ACTION"]
    source_subject_logical_key: NodeLogicalKey
    parent_window_logical_key: NodeLogicalKey
    source_lineage_sha256: Sha256Digest
    refinement_role: BoundaryRefinementRole
    generation: PositiveInt
    created_at: Rfc3339Timestamp
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        anchor = (
            self.coarse_interval.start_ns
            if self.refinement_role is BoundaryRefinementRole.ONSET
            else self.coarse_interval.end_ns
        )
        requested = NanosecondInterval(
            start_ns=anchor - self.padding_before_ns,
            end_ns=anchor + self.padding_after_ns,
        )
        if (
            self.boundary_anchor_ns != anchor
            or self.requested_interval != requested
            or self.source_subject_logical_key != self.provisional_action_logical_key
            or self.interval.start_ns < 0
            or self.interval.end_ns > self.recording_duration_ns
            or self.interval.start_ns < requested.start_ns
            or self.interval.end_ns > requested.end_ns
            or self.context_truncated != (self.interval != requested)
        ):
            raise ValueError("boundary refinement window has inconsistent subject or bounds")
        digest = semantic_sha256(boundary_refinement_window_projection(self))
        if (
            self.semantic_sha256 != digest
            or self.window_logical_key != f"temporal-window:{digest}"
            or self.window_id != _stable_uuid(BOUNDARY_WINDOW_UUID_NAMESPACE, digest)
        ):
            raise ValueError("boundary refinement window identity is inconsistent")
        return self

    @classmethod
    def from_context(
        cls,
        *,
        context: AdmittedRecordingContextV2,
        action: ProvisionalPhysicalAction,
        provisional_fusion_semantic_sha256: str,
        parent_window: CanonicalRootWindow,
        role: BoundaryRefinementRole,
        padding_before_ns: int,
        padding_after_ns: int,
        window_policy_version: str,
        created_at: str,
    ) -> Self:
        checked_context = _strict_context(context)
        checked_action = ProvisionalPhysicalAction.model_validate(
            action.model_dump(mode="python"), strict=True
        )
        checked_parent = CanonicalRootWindow.model_validate(
            parent_window.model_dump(mode="python"), strict=True
        )
        if padding_before_ns < 0 or padding_after_ns <= 0:
            raise CanonicalOfflineConfigurationError(
                "boundary padding must be nonnegative before and positive after"
            )
        if (
            checked_action.mcap_id != checked_context.ready_manifest.mcap_id
            or checked_action.source_content_sha256 != checked_context.source_content_sha256
            or checked_action.camera_mapping_semantic_sha256
            != checked_context.camera_mapping_semantic_sha256
            or checked_action.alignment_semantic_sha256 != checked_context.alignment_semantic_sha256
            or checked_parent.recording_identity != checked_context.recording_identity
            or checked_parent.mcap_id != checked_action.mcap_id
        ):
            raise CanonicalOfflineConfigurationError(
                "boundary action/window lineage differs from admission context"
            )
        anchor = (
            checked_action.coarse_interval.start_ns
            if role is BoundaryRefinementRole.ONSET
            else checked_action.coarse_interval.end_ns
        )
        requested = NanosecondInterval(
            start_ns=anchor - padding_before_ns,
            end_ns=anchor + padding_after_ns,
        )
        duration = checked_context.ready_manifest.recording.duration_ns
        effective_start = max(0, requested.start_ns)
        effective_end = min(duration, requested.end_ns)
        if effective_start >= effective_end:
            raise CanonicalOfflineConfigurationError(
                "boundary refinement request does not overlap the recording"
            )
        effective = NanosecondInterval(
            start_ns=effective_start,
            end_ns=effective_end,
        )
        source_lineage = semantic_sha256(
            {
                "semantic_projection_version": ("boundary-refinement-window-source-lineage-v1"),
                "admission_context_semantic_sha256": checked_context.semantic_sha256,
                "provisional_fusion_semantic_sha256": (provisional_fusion_semantic_sha256),
                "provisional_action_logical_key": checked_action.logical_key,
                "parent_window_semantic_sha256": checked_parent.semantic_sha256,
                "refinement_role": role.value,
            }
        )
        values: dict[str, Any] = {
            "source_content_sha256": checked_context.source_content_sha256,
            "camera_mapping_semantic_sha256": (checked_context.camera_mapping_semantic_sha256),
            "alignment_semantic_sha256": checked_context.alignment_semantic_sha256,
            "requested_interval": requested,
            "interval": effective,
            "purpose": SamplingPurpose.BOUNDARY_REFINEMENT,
            "window_policy_version": window_policy_version,
            "source_subject_type": "PROVISIONAL_PHYSICAL_ACTION",
            "source_subject_logical_key": checked_action.logical_key,
            "parent_window_logical_key": checked_parent.window_logical_key,
            "source_lineage_sha256": source_lineage,
            "refinement_role": role,
            "generation": checked_parent.generation + 1,
            "provisional_action_semantic_sha256": checked_action.semantic_sha256,
            "provisional_fusion_semantic_sha256": provisional_fusion_semantic_sha256,
            "coarse_interval": checked_action.coarse_interval,
            "boundary_anchor_ns": anchor,
            "padding_before_ns": padding_before_ns,
            "padding_after_ns": padding_after_ns,
            "context_truncated": effective != requested,
            "production_eligible": False,
        }
        digest = semantic_sha256(_boundary_refinement_window_projection_values(values))
        return cls(
            schema_version="1.0",
            window_id=_stable_uuid(BOUNDARY_WINDOW_UUID_NAMESPACE, digest),
            window_logical_key=f"temporal-window:{digest}",
            semantic_sha256=digest,
            recording_identity=checked_context.recording_identity,
            source_content_sha256=checked_context.source_content_sha256,
            mcap_id=checked_context.ready_manifest.mcap_id,
            camera_mapping_run_id=checked_context.ready_manifest.camera_mapping_run_id,
            alignment_id=checked_context.alignment_manifest.alignment_id,
            camera_mapping_semantic_sha256=(checked_context.camera_mapping_semantic_sha256),
            alignment_semantic_sha256=checked_context.alignment_semantic_sha256,
            recording_duration_ns=duration,
            reference_timebase="recording_relative_ns",
            provisional_action_logical_key=checked_action.logical_key,
            provisional_action_semantic_sha256=checked_action.semantic_sha256,
            provisional_fusion_semantic_sha256=provisional_fusion_semantic_sha256,
            coarse_interval=checked_action.coarse_interval,
            boundary_anchor_ns=anchor,
            padding_before_ns=padding_before_ns,
            padding_after_ns=padding_after_ns,
            requested_interval=requested,
            interval=effective,
            context_truncated=effective != requested,
            purpose=SamplingPurpose.BOUNDARY_REFINEMENT,
            window_policy_version=window_policy_version,
            source_subject_type="PROVISIONAL_PHYSICAL_ACTION",
            source_subject_logical_key=checked_action.logical_key,
            parent_window_logical_key=checked_parent.window_logical_key,
            source_lineage_sha256=source_lineage,
            refinement_role=role,
            generation=checked_parent.generation + 1,
            created_at=created_at,
            production_eligible=False,
        )


def canonical_boundary_refinement_lineage(
    *,
    context: AdmittedRecordingContextV2,
    window: CanonicalBoundaryRefinementWindow,
    sampling_plan: SamplingPlan,
) -> PackageLineage:
    """Build exact package lineage for one role-bound boundary window."""

    checked_context = _strict_context(context)
    checked_window = CanonicalBoundaryRefinementWindow.model_validate(
        window.model_dump(mode="python"), strict=True
    )
    checked_plan = SamplingPlan.model_validate(sampling_plan.model_dump(mode="python"), strict=True)
    if (
        checked_window.recording_identity != checked_context.recording_identity
        or checked_window.source_content_sha256 != checked_context.source_content_sha256
        or checked_window.mcap_id != checked_context.ready_manifest.mcap_id
        or checked_window.camera_mapping_run_id
        != checked_context.ready_manifest.camera_mapping_run_id
        or checked_window.alignment_id != checked_context.alignment_manifest.alignment_id
        or checked_window.camera_mapping_semantic_sha256
        != checked_context.camera_mapping_semantic_sha256
        or checked_window.alignment_semantic_sha256 != checked_context.alignment_semantic_sha256
    ):
        raise CanonicalOfflineConfigurationError(
            "boundary refinement window differs from admission context"
        )
    return PackageLineage(
        source_content_sha256=checked_context.source_content_sha256,
        window_semantic_sha256=checked_window.semantic_sha256,
        camera_mapping_semantic_sha256=(checked_context.camera_mapping_semantic_sha256),
        alignment_semantic_sha256=checked_context.alignment_semantic_sha256,
        sampling_plan_sha256=sampling_plan_digest(
            checked_plan,
            purpose=SamplingPurpose.BOUNDARY_REFINEMENT,
        ),
    )


__all__ = [
    "BOUNDARY_WINDOW_PROJECTION_VERSION",
    "BoundaryRefinementRole",
    "CanonicalBoundaryRefinementWindow",
    "boundary_refinement_window_projection",
    "canonical_boundary_refinement_lineage",
]
