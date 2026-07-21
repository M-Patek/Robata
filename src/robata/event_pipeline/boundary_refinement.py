"""Authoritative dual-role boundary normalization and deterministic reduction.

``BOUNDARY_OBSERVATION.interval`` has exactly one meaning in this module: it
is the uncertainty interval for the role supplied to :meth:`project_role`.
ONSET and OFFSET are projected independently from their own immutable window,
package set, input plan, and enriched outputs.  The two role results are only
combined after both trust closures have been proved.

This is intentionally a local-conformance surface.  It never falls back to a
coarse interval, never assigns an event ID, and can never claim production
eligibility.  Provider scores are retained as uncalibrated audit evidence but
do not participate in reduction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.admission_v2 import AlignmentManifestV2
from robata.contracts.alignment import AlignmentStatus
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.contracts.temporal import TemporalPackageSet, TemporalPackageSetMember
from robata.event_pipeline.provisional_fusion import ProvisionalPhysicalAction
from robata.inference.enrichment import (
    EnrichedEvidenceReference,
    EnrichedProviderClaim,
    OrchestratorEnrichedOutput,
    ProviderClaimKind,
    ProviderObservation,
)
from robata.inference.input_plan import InferenceCallPart, InferenceInputPlan
from robata.inference.models import VisionTask

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
CameraCount = Annotated[int, Field(strict=True, ge=0, le=6)]
MinimumCameraCount = Annotated[int, Field(strict=True, ge=1, le=6)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
AmbiguityCode = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$"),
]

BOUNDARY_REFINEMENT_POLICY_PROJECTION_VERSION: Final = "boundary-refinement-policy-semantic-v1"
BOUNDARY_OBSERVATION_PROJECTION_VERSION: Final = "boundary-observation-semantic-v1"
BOUNDARY_ROLE_RESULT_PROJECTION_VERSION: Final = "boundary-refinement-role-semantic-v1"
BOUNDARY_RESULT_PROJECTION_VERSION: Final = "boundary-refinement-result-semantic-v1"
BOUNDARY_OBSERVATION_LOGICAL_KEY_NAMESPACE: Final = "boundary-observation-v1"
BOUNDARY_ROLE_RESULT_LOGICAL_KEY_NAMESPACE: Final = "boundary-refinement-role-v1"
BOUNDARY_RESULT_LOGICAL_KEY_NAMESPACE: Final = "boundary-refinement-v1"
LOCAL_BOUNDARY_REFINEMENT_POLICY_VERSION: Final = "local-boundary-refinement-v1"
BOUNDARY_REDUCER_VERSION: Final = "median-low-max-envelope-v1"


class BoundaryRefinementProjectionError(ValueError):
    """Boundary evidence does not close over its authoritative inputs."""


class BoundaryRefinementRole(StrEnum):
    """A role-bound provider interval; roles must never be inferred from labels."""

    ONSET = "ONSET"
    OFFSET = "OFFSET"


class BoundaryCameraOutcome(StrEnum):
    """Deterministic status of one of the six canonical camera slots."""

    OBSERVED = "OBSERVED"
    NO_BOUNDARY = "NO_BOUNDARY"
    INDETERMINATE = "INDETERMINATE"


class BoundaryRefinementOutcome(StrEnum):
    """Role-level and action-level boundary reduction status."""

    REFINED = "REFINED"
    INDETERMINATE = "INDETERMINATE"


class BoundaryRefinementPolicy(StrictModel):
    """Versioned local policy; fallback is structurally unavailable."""

    version: SchemaVersion
    minimum_observed_cameras: MinimumCameraCount = 2
    padding_before_ns: NonNegativeInt = 500_000_000
    padding_after_ns: PositiveInt = 500_000_000
    reducer_version: Literal["median-low-max-envelope-v1"] = BOUNDARY_REDUCER_VERSION
    projection_version: Literal["boundary-refinement-policy-semantic-v1"] = (
        BOUNDARY_REFINEMENT_POLICY_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    fallback_enabled: Literal[False] = False
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        version: str = LOCAL_BOUNDARY_REFINEMENT_POLICY_VERSION,
        minimum_observed_cameras: int = 2,
        padding_before_ns: int = 500_000_000,
        padding_after_ns: int = 500_000_000,
    ) -> Self:
        """Build the policy while deriving its immutable semantic digest."""

        values: dict[str, Any] = {
            "version": version,
            "minimum_observed_cameras": minimum_observed_cameras,
            "padding_before_ns": padding_before_ns,
            "padding_after_ns": padding_after_ns,
            "reducer_version": BOUNDARY_REDUCER_VERSION,
            "projection_version": BOUNDARY_REFINEMENT_POLICY_PROJECTION_VERSION,
            "fallback_enabled": False,
            "production_eligible": False,
        }
        digest = semantic_sha256(_boundary_policy_projection_values(values))
        return cls.model_validate({**values, "semantic_sha256": digest}, strict=True)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.semantic_sha256 != semantic_sha256(
            boundary_refinement_policy_semantic_projection(self)
        ):
            raise ValueError("boundary refinement policy semantic identity is inconsistent")
        return self


class BoundaryRefinementPackageRef(StrictModel):
    """One exact ordered dense package retained for audit."""

    package_id: OpaqueUuid
    package_ordinal: NonNegativeInt
    interval: NanosecondInterval
    semantic_content_sha256: Sha256Digest
    manifest_sha256: Sha256Digest


class BoundaryRefinementOutputRef(StrictModel):
    """Selected call-part output; UUID fields remain provenance locators only."""

    part_ordinal: NonNegativeInt
    part_semantic_sha256: Sha256Digest
    source_inference_id: OpaqueUuid
    source_artifact_id: OpaqueUuid
    selected_output_sha256: Sha256Digest
    selection_decision_logical_key: NodeLogicalKey
    enrichment_logical_key: NodeLogicalKey


class BoundaryRefinementClaimRef(StrictModel):
    """Exact enriched claim locator inside one selected call-part output."""

    output: BoundaryRefinementOutputRef
    claim_ordinal: NonNegativeInt
    claim_id: OpaqueUuid


class NormalizedBoundaryObservation(StrictModel):
    """One role-bound package-camera boundary observation."""

    logical_key: NodeLogicalKey
    source: BoundaryRefinementClaimRef
    package_id: OpaqueUuid
    package_ordinal: NonNegativeInt
    camera_id: CameraId
    role: BoundaryRefinementRole
    boundary_interval: NanosecondInterval | None
    label: str | None
    observation: ProviderObservation
    evidence: tuple[EnrichedEvidenceReference, ...]
    provider_conflict_codes: tuple[str, ...]
    model_reported_score: UnitInterval | None
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.observation not in _BOUNDARY_OBSERVATIONS:
            raise ValueError("invalid BOUNDARY_OBSERVATION status")
        if self.observation is ProviderObservation.OBSERVED and (
            self.boundary_interval is None or not self.evidence
        ):
            raise ValueError("OBSERVED boundary claims require interval and evidence")
        if self.observation is ProviderObservation.MISSING and (
            self.boundary_interval is not None or self.evidence
        ):
            raise ValueError("MISSING boundary claims cannot assert interval or evidence")
        if self.evidence != tuple(sorted(self.evidence, key=_evidence_sort_key)):
            raise ValueError("boundary evidence must be canonically ordered")
        if self.provider_conflict_codes != tuple(sorted(set(self.provider_conflict_codes))):
            raise ValueError("provider conflict codes must be unique and ordered")
        digest = semantic_sha256(boundary_observation_semantic_projection(self))
        expected_key = f"{BOUNDARY_OBSERVATION_LOGICAL_KEY_NAMESPACE}:{digest}"
        if self.logical_key != expected_key:
            raise ValueError("boundary observation logical identity is inconsistent")
        return self


class CameraBoundaryEvidence(StrictModel):
    """One explicit camera slot for one refinement role."""

    camera_id: CameraId
    role: BoundaryRefinementRole
    observations: tuple[NormalizedBoundaryObservation, ...]
    outcome: BoundaryCameraOutcome
    observed_interval: NanosecondInterval | None
    boundary_estimate_ns: Nanoseconds | None
    uncertainty_ns: Nanoseconds | None
    alignment_status: AlignmentStatus
    alignment_residual_p95_ns: NonNegativeInt
    alignment_max_error_ns: NonNegativeInt
    ambiguity_codes: tuple[AmbiguityCode, ...]
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_slot(self) -> Self:
        if any(
            item.camera_id is not self.camera_id or item.role is not self.role
            for item in self.observations
        ):
            raise ValueError("camera boundary slot contains foreign evidence")
        expected = tuple(sorted(self.observations, key=_observation_sort_key))
        keys = tuple(item.logical_key for item in self.observations)
        if self.observations != expected or len(keys) != len(set(keys)):
            raise ValueError("camera boundary observations must be unique and ordered")
        if self.alignment_status in {AlignmentStatus.INVALID, AlignmentStatus.UNVERIFIED}:
            raise ValueError("camera boundary evidence requires verified alignment")
        if self.ambiguity_codes != tuple(sorted(set(self.ambiguity_codes))):
            raise ValueError("camera ambiguity codes must be unique and ordered")
        if self.outcome is BoundaryCameraOutcome.OBSERVED:
            if (
                self.observed_interval is None
                or self.boundary_estimate_ns is None
                or self.uncertainty_ns is None
                or self.uncertainty_ns < 0
                or not self.observations
            ):
                raise ValueError("observed camera boundary lacks its deterministic estimate")
        elif any(
            value is not None
            for value in (
                self.observed_interval,
                self.boundary_estimate_ns,
                self.uncertainty_ns,
            )
        ):
            raise ValueError("non-observed camera boundary cannot expose an estimate")
        return self


class BoundaryRefinementRoleResult(StrictModel):
    """Exact six-camera reduction for one independently projected role."""

    task: Literal[VisionTask.BOUNDARY_REFINEMENT] = VisionTask.BOUNDARY_REFINEMENT
    role: BoundaryRefinementRole
    source_action_logical_key: NodeLogicalKey
    source_action_semantic_sha256: Sha256Digest
    source_action_policy_semantic_sha256: Sha256Digest
    source_action_ordinal: NonNegativeInt
    action_label: str
    coarse_interval: NanosecondInterval
    coarse_anchor_ns: Nanoseconds
    mcap_id: OpaqueUuid
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    camera_mapping_run_id: OpaqueUuid
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_id: OpaqueUuid
    alignment_semantic_sha256: Sha256Digest
    window_semantic_sha256: Sha256Digest
    requested_window_interval: NanosecondInterval
    window_interval: NanosecondInterval
    package_set_id: OpaqueUuid
    split_plan_digest: Sha256Digest
    member_manifest_sha256: Sha256Digest
    packages: tuple[BoundaryRefinementPackageRef, ...]
    input_plan_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    source_outputs: tuple[BoundaryRefinementOutputRef, ...]
    camera_evidence: SixCameraMap[CameraBoundaryEvidence]
    observed_camera_count: CameraCount
    boundary_estimate_ns: Nanoseconds | None
    uncertainty_ns: Nanoseconds | None
    boundary_interval: NanosecondInterval | None
    outcome: BoundaryRefinementOutcome
    ambiguity_codes: tuple[AmbiguityCode, ...]
    policy: BoundaryRefinementPolicy
    projection_version: Literal["boundary-refinement-role-semantic-v1"] = (
        BOUNDARY_ROLE_RESULT_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    used_fallback: Literal[False] = False
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _validate_role_result_shape(self)
        digest = semantic_sha256(boundary_role_result_semantic_projection(self))
        expected_key = f"{BOUNDARY_ROLE_RESULT_LOGICAL_KEY_NAMESPACE}:{digest}"
        if self.semantic_sha256 != digest or self.logical_key != expected_key:
            raise ValueError("boundary role result logical identity is inconsistent")
        return self


class BoundaryRefinementResult(StrictModel):
    """Action-level ONSET/OFFSET reduction; it is not an event registry record."""

    source_action_logical_key: NodeLogicalKey
    source_action_semantic_sha256: Sha256Digest
    source_action_ordinal: NonNegativeInt
    action_label: str
    coarse_interval: NanosecondInterval
    mcap_id: OpaqueUuid
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    onset: BoundaryRefinementRoleResult
    offset: BoundaryRefinementRoleResult
    refined_interval: NanosecondInterval | None
    onset_interval: NanosecondInterval | None
    offset_interval: NanosecondInterval | None
    onset_estimate_ns: Nanoseconds | None
    offset_estimate_ns: Nanoseconds | None
    uncertainty_ns: Nanoseconds | None
    outcome: BoundaryRefinementOutcome
    ambiguity_codes: tuple[AmbiguityCode, ...]
    policy: BoundaryRefinementPolicy
    projection_version: Literal["boundary-refinement-result-semantic-v1"] = (
        BOUNDARY_RESULT_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    used_fallback: Literal[False] = False
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _validate_action_result_shape(self)
        digest = semantic_sha256(boundary_refinement_result_semantic_projection(self))
        expected_key = f"{BOUNDARY_RESULT_LOGICAL_KEY_NAMESPACE}:{digest}"
        if self.semantic_sha256 != digest or self.logical_key != expected_key:
            raise ValueError("boundary refinement result logical identity is inconsistent")
        return self


_BOUNDARY_OBSERVATIONS = frozenset(
    {
        ProviderObservation.OBSERVED,
        ProviderObservation.NO_BOUNDARY,
        ProviderObservation.OCCLUDED,
        ProviderObservation.UNUSABLE,
        ProviderObservation.MISSING,
    }
)


def boundary_refinement_policy_semantic_projection(
    policy: BoundaryRefinementPolicy,
) -> dict[str, object]:
    """Return the exact versioned policy identity."""

    return _boundary_policy_projection_values(
        {
            "version": policy.version,
            "minimum_observed_cameras": policy.minimum_observed_cameras,
            "padding_before_ns": policy.padding_before_ns,
            "padding_after_ns": policy.padding_after_ns,
            "reducer_version": policy.reducer_version,
            "projection_version": policy.projection_version,
            "fallback_enabled": policy.fallback_enabled,
            "production_eligible": policy.production_eligible,
        }
    )


def _boundary_policy_projection_values(values: Mapping[str, object]) -> dict[str, object]:
    return {
        "semantic_projection_version": values["projection_version"],
        "version": values["version"],
        "minimum_observed_cameras": values["minimum_observed_cameras"],
        "padding_before_ns": str(values["padding_before_ns"]),
        "padding_after_ns": str(values["padding_after_ns"]),
        "reducer_version": values["reducer_version"],
        "fallback_enabled": values["fallback_enabled"],
        "production_eligible": values["production_eligible"],
    }


def boundary_output_semantic_projection(
    output: BoundaryRefinementOutputRef,
) -> dict[str, object]:
    """Return selected-output content without run/artifact locator UUIDs."""

    return {
        "part_ordinal": output.part_ordinal,
        "part_semantic_sha256": output.part_semantic_sha256,
        "selected_output_sha256": output.selected_output_sha256,
        "selection_decision_logical_key": output.selection_decision_logical_key,
    }


def boundary_observation_semantic_projection(
    observation: NormalizedBoundaryObservation,
) -> dict[str, object]:
    """Return the run-independent identity of one role-bound camera claim."""

    return {
        "semantic_projection_version": BOUNDARY_OBSERVATION_PROJECTION_VERSION,
        "source_output": boundary_output_semantic_projection(observation.source.output),
        "claim_ordinal": observation.source.claim_ordinal,
        "package_ordinal": observation.package_ordinal,
        "camera_id": observation.camera_id.value,
        "role": observation.role.value,
        "boundary_interval": _interval_projection(observation.boundary_interval),
        "label": observation.label,
        "observation": observation.observation.value,
        "evidence_coordinates": [
            _evidence_semantic_projection(item) for item in observation.evidence
        ],
        "provider_conflict_codes": list(observation.provider_conflict_codes),
        "model_reported_score": observation.model_reported_score,
        "production_eligible": observation.production_eligible,
    }


def boundary_role_result_semantic_projection(
    result: BoundaryRefinementRoleResult,
) -> dict[str, object]:
    """Return role identity across action, package, provider, and six-camera facts."""

    return {
        "semantic_projection_version": result.projection_version,
        "role": result.role.value,
        "source_action_logical_key": result.source_action_logical_key,
        "source_action_semantic_sha256": result.source_action_semantic_sha256,
        "source_action_policy_semantic_sha256": result.source_action_policy_semantic_sha256,
        "source_action_ordinal": result.source_action_ordinal,
        "action_label": result.action_label,
        "coarse_interval": result.coarse_interval.model_dump(mode="json"),
        "coarse_anchor_ns": str(result.coarse_anchor_ns),
        "source_content_sha256": result.source_content_sha256,
        "camera_mapping_semantic_sha256": result.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": result.alignment_semantic_sha256,
        "window_semantic_sha256": result.window_semantic_sha256,
        "requested_window_interval": result.requested_window_interval.model_dump(mode="json"),
        "window_interval": result.window_interval.model_dump(mode="json"),
        "split_plan_digest": result.split_plan_digest,
        "member_manifest_sha256": result.member_manifest_sha256,
        "package_members": [
            {
                "package_ordinal": item.package_ordinal,
                "interval": item.interval.model_dump(mode="json"),
                "semantic_content_sha256": item.semantic_content_sha256,
            }
            for item in result.packages
        ],
        "input_plan_semantic_sha256": result.input_plan_semantic_sha256,
        "source_outputs": [
            boundary_output_semantic_projection(item) for item in result.source_outputs
        ],
        "camera_evidence": {
            camera_id.value: _camera_semantic_projection(result.camera_evidence[camera_id])
            for camera_id in CAMERA_IDS
        },
        "observed_camera_count": result.observed_camera_count,
        "boundary_estimate_ns": _nanoseconds_projection(result.boundary_estimate_ns),
        "uncertainty_ns": _nanoseconds_projection(result.uncertainty_ns),
        "boundary_interval": _interval_projection(result.boundary_interval),
        "outcome": result.outcome.value,
        "ambiguity_codes": list(result.ambiguity_codes),
        "policy_semantic_sha256": result.policy.semantic_sha256,
        "used_fallback": result.used_fallback,
        "production_eligible": result.production_eligible,
    }


def boundary_refinement_result_semantic_projection(
    result: BoundaryRefinementResult,
) -> dict[str, object]:
    """Return action-level identity without event or run locator assignment."""

    return {
        "semantic_projection_version": result.projection_version,
        "source_action_logical_key": result.source_action_logical_key,
        "source_action_semantic_sha256": result.source_action_semantic_sha256,
        "source_action_ordinal": result.source_action_ordinal,
        "action_label": result.action_label,
        "coarse_interval": result.coarse_interval.model_dump(mode="json"),
        "source_content_sha256": result.source_content_sha256,
        "camera_mapping_semantic_sha256": result.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": result.alignment_semantic_sha256,
        "onset_logical_key": result.onset.logical_key,
        "onset_semantic_sha256": result.onset.semantic_sha256,
        "offset_logical_key": result.offset.logical_key,
        "offset_semantic_sha256": result.offset.semantic_sha256,
        "refined_interval": _interval_projection(result.refined_interval),
        "onset_interval": _interval_projection(result.onset_interval),
        "offset_interval": _interval_projection(result.offset_interval),
        "onset_estimate_ns": _nanoseconds_projection(result.onset_estimate_ns),
        "offset_estimate_ns": _nanoseconds_projection(result.offset_estimate_ns),
        "uncertainty_ns": _nanoseconds_projection(result.uncertainty_ns),
        "outcome": result.outcome.value,
        "ambiguity_codes": list(result.ambiguity_codes),
        "policy_semantic_sha256": result.policy.semantic_sha256,
        "used_fallback": result.used_fallback,
        "production_eligible": result.production_eligible,
    }


class BoundaryRefinementProjector:
    """Normalize independently role-bound outputs, then reduce the two roles."""

    def __init__(self, policy: BoundaryRefinementPolicy) -> None:
        self._policy = BoundaryRefinementPolicy.model_validate(
            policy.model_dump(mode="python"), strict=True
        )

    @property
    def policy(self) -> BoundaryRefinementPolicy:
        return self._policy

    def project_role(
        self,
        *,
        action: ProvisionalPhysicalAction,
        role: BoundaryRefinementRole,
        input_plan: InferenceInputPlan,
        package_set: TemporalPackageSet,
        enriched_outputs: Sequence[OrchestratorEnrichedOutput],
        alignment_manifest: AlignmentManifestV2,
    ) -> BoundaryRefinementRoleResult:
        """Project one ONSET or OFFSET role from an exact enriched-output closure."""

        checked_action = ProvisionalPhysicalAction.model_validate(
            action.model_dump(mode="python"), strict=True
        )
        if not isinstance(role, BoundaryRefinementRole):
            raise BoundaryRefinementProjectionError("boundary refinement role is invalid")
        checked_plan = InferenceInputPlan.model_validate(
            input_plan.model_dump(mode="python"), strict=True
        )
        checked_packages = TemporalPackageSet.model_validate(
            package_set.model_dump(mode="python"), strict=True
        )
        checked_outputs = tuple(
            OrchestratorEnrichedOutput.model_validate(output.model_dump(mode="python"), strict=True)
            for output in enriched_outputs
        )
        checked_alignment = AlignmentManifestV2.model_validate(
            alignment_manifest.model_dump(mode="python"), strict=True
        )
        _validate_projection_lineage(
            action=checked_action,
            role=role,
            policy=self._policy,
            input_plan=checked_plan,
            package_set=checked_packages,
            enriched_outputs=checked_outputs,
            alignment_manifest=checked_alignment,
        )

        package_refs = tuple(_package_ref(member) for member in checked_packages.members)
        output_refs: list[BoundaryRefinementOutputRef] = []
        by_camera: dict[CameraId, list[NormalizedBoundaryObservation]] = {
            camera_id: [] for camera_id in CAMERA_IDS
        }
        for part, output in zip(checked_plan.call_plan.parts, checked_outputs, strict=True):
            output_ref = _output_ref(part, output)
            output_refs.append(output_ref)
            visible_items = {
                item.provider_item_ordinal: item
                for item in checked_plan.rendered_items[
                    part.start_item_ordinal : part.end_item_ordinal_exclusive
                ]
            }
            for claim in output.claims:
                _validate_claim_evidence_visible(
                    claim=claim,
                    visible_items=visible_items,
                    package_set=checked_packages,
                )
                observation = _normalize_boundary_observation(
                    claim=claim,
                    output_ref=output_ref,
                    role=role,
                    package_set=checked_packages,
                )
                by_camera[observation.camera_id].append(observation)

        window_interval = NanosecondInterval(
            start_ns=checked_packages.start_ns,
            end_ns=checked_packages.end_ns,
        )
        camera_evidence = SixCameraMap[CameraBoundaryEvidence](
            {
                camera_id: _build_camera_evidence(
                    camera_id=camera_id,
                    role=role,
                    observations=tuple(by_camera[camera_id]),
                    alignment_manifest=checked_alignment,
                    package_set=checked_packages,
                    window_interval=window_interval,
                )
                for camera_id in CAMERA_IDS
            }
        )
        observed_count = sum(
            camera_evidence[camera_id].outcome is BoundaryCameraOutcome.OBSERVED
            for camera_id in CAMERA_IDS
        )
        estimate, uncertainty, boundary_interval, outcome, ambiguity_codes = _role_reduction_values(
            camera_evidence=camera_evidence,
            minimum_observed_cameras=self._policy.minimum_observed_cameras,
            window_interval=window_interval,
        )
        anchor = (
            checked_action.coarse_interval.start_ns
            if role is BoundaryRefinementRole.ONSET
            else checked_action.coarse_interval.end_ns
        )
        values: dict[str, Any] = {
            "task": VisionTask.BOUNDARY_REFINEMENT,
            "role": role,
            "source_action_logical_key": checked_action.logical_key,
            "source_action_semantic_sha256": checked_action.semantic_sha256,
            "source_action_policy_semantic_sha256": checked_action.policy_semantic_sha256,
            "source_action_ordinal": checked_action.ordinal,
            "action_label": checked_action.label,
            "coarse_interval": checked_action.coarse_interval,
            "coarse_anchor_ns": anchor,
            "mcap_id": checked_action.mcap_id,
            "recording_identity": checked_alignment.recording_identity,
            "source_content_sha256": checked_action.source_content_sha256,
            "camera_mapping_run_id": checked_alignment.camera_mapping_run_id,
            "camera_mapping_semantic_sha256": checked_action.camera_mapping_semantic_sha256,
            "alignment_id": checked_alignment.alignment_id,
            "alignment_semantic_sha256": checked_action.alignment_semantic_sha256,
            "window_semantic_sha256": checked_packages.lineage.window_semantic_sha256,
            "requested_window_interval": NanosecondInterval(
                start_ns=checked_packages.requested_start_ns,
                end_ns=checked_packages.requested_end_ns,
            ),
            "window_interval": window_interval,
            "package_set_id": checked_packages.package_set_id,
            "split_plan_digest": checked_packages.split_plan_digest,
            "member_manifest_sha256": checked_packages.member_manifest_sha256,
            "packages": package_refs,
            "input_plan_id": checked_plan.input_plan_id,
            "input_plan_semantic_sha256": checked_plan.semantic_sha256,
            "source_outputs": tuple(output_refs),
            "camera_evidence": camera_evidence,
            "observed_camera_count": observed_count,
            "boundary_estimate_ns": estimate,
            "uncertainty_ns": uncertainty,
            "boundary_interval": boundary_interval,
            "outcome": outcome,
            "ambiguity_codes": ambiguity_codes,
            "policy": self._policy,
            "projection_version": BOUNDARY_ROLE_RESULT_PROJECTION_VERSION,
            "used_fallback": False,
            "production_eligible": False,
        }
        draft = BoundaryRefinementRoleResult.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{BOUNDARY_ROLE_RESULT_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
            **values,
        )
        digest = semantic_sha256(boundary_role_result_semantic_projection(draft))
        return BoundaryRefinementRoleResult.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{BOUNDARY_ROLE_RESULT_LOGICAL_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    def reduce(
        self,
        *,
        action: ProvisionalPhysicalAction,
        onset: BoundaryRefinementRoleResult,
        offset: BoundaryRefinementRoleResult,
    ) -> BoundaryRefinementResult:
        """Combine exact ONSET/OFFSET results without coarse-event fallback."""

        checked_action = ProvisionalPhysicalAction.model_validate(
            action.model_dump(mode="python"), strict=True
        )
        checked_onset = BoundaryRefinementRoleResult.model_validate(
            onset.model_dump(mode="python"), strict=True
        )
        checked_offset = BoundaryRefinementRoleResult.model_validate(
            offset.model_dump(mode="python"), strict=True
        )
        _validate_role_pair(
            action=checked_action,
            onset=checked_onset,
            offset=checked_offset,
            policy=self._policy,
        )
        (
            refined_interval,
            onset_interval,
            offset_interval,
            onset_estimate,
            offset_estimate,
            uncertainty,
            outcome,
            ambiguity_codes,
        ) = _action_reduction_values(checked_onset, checked_offset)
        values: dict[str, Any] = {
            "source_action_logical_key": checked_action.logical_key,
            "source_action_semantic_sha256": checked_action.semantic_sha256,
            "source_action_ordinal": checked_action.ordinal,
            "action_label": checked_action.label,
            "coarse_interval": checked_action.coarse_interval,
            "mcap_id": checked_action.mcap_id,
            "source_content_sha256": checked_action.source_content_sha256,
            "camera_mapping_semantic_sha256": checked_action.camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": checked_action.alignment_semantic_sha256,
            "onset": checked_onset,
            "offset": checked_offset,
            "refined_interval": refined_interval,
            "onset_interval": onset_interval,
            "offset_interval": offset_interval,
            "onset_estimate_ns": onset_estimate,
            "offset_estimate_ns": offset_estimate,
            "uncertainty_ns": uncertainty,
            "outcome": outcome,
            "ambiguity_codes": ambiguity_codes,
            "policy": self._policy,
            "projection_version": BOUNDARY_RESULT_PROJECTION_VERSION,
            "used_fallback": False,
            "production_eligible": False,
        }
        draft = BoundaryRefinementResult.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{BOUNDARY_RESULT_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
            **values,
        )
        digest = semantic_sha256(boundary_refinement_result_semantic_projection(draft))
        return BoundaryRefinementResult.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{BOUNDARY_RESULT_LOGICAL_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )


def _validate_projection_lineage(
    *,
    action: ProvisionalPhysicalAction,
    role: BoundaryRefinementRole,
    policy: BoundaryRefinementPolicy,
    input_plan: InferenceInputPlan,
    package_set: TemporalPackageSet,
    enriched_outputs: Sequence[OrchestratorEnrichedOutput],
    alignment_manifest: AlignmentManifestV2,
) -> None:
    if input_plan.subject.task is not VisionTask.BOUNDARY_REFINEMENT:
        raise BoundaryRefinementProjectionError(
            "projector requires a BOUNDARY_REFINEMENT input plan"
        )
    if input_plan.request_catalog.task is not VisionTask.BOUNDARY_REFINEMENT:
        raise BoundaryRefinementProjectionError("request catalog task is not BOUNDARY_REFINEMENT")
    if (
        package_set.mcap_id != action.mcap_id
        or package_set.lineage.source_content_sha256 != action.source_content_sha256
        or package_set.lineage.camera_mapping_semantic_sha256
        != action.camera_mapping_semantic_sha256
        or package_set.lineage.alignment_semantic_sha256 != action.alignment_semantic_sha256
    ):
        raise BoundaryRefinementProjectionError(
            "boundary package set differs from its provisional action"
        )
    anchor = (
        action.coarse_interval.start_ns
        if role is BoundaryRefinementRole.ONSET
        else action.coarse_interval.end_ns
    )
    policy_start = anchor - policy.padding_before_ns
    policy_end = anchor + policy.padding_after_ns
    if (
        package_set.requested_start_ns < policy_start
        or package_set.requested_end_ns > policy_end
        or package_set.requested_start_ns > anchor
        or package_set.requested_end_ns < anchor
        or package_set.start_ns > anchor
        or package_set.end_ns < anchor
    ):
        raise BoundaryRefinementProjectionError(
            "boundary package window is not the role-bound policy window"
        )
    if alignment_manifest.status in {AlignmentStatus.INVALID, AlignmentStatus.UNVERIFIED} or any(
        camera.status in {AlignmentStatus.INVALID, AlignmentStatus.UNVERIFIED}
        for camera in alignment_manifest.cameras.values()
    ):
        raise BoundaryRefinementProjectionError(
            "boundary refinement requires verified per-camera alignment"
        )
    if (
        alignment_manifest.mcap_id != action.mcap_id
        or alignment_manifest.source_content_sha256 != action.source_content_sha256
        or alignment_manifest.camera_mapping_run_id != package_set.camera_mapping_run_id
        or alignment_manifest.camera_mapping_semantic_sha256
        != action.camera_mapping_semantic_sha256
        or alignment_manifest.alignment_id != package_set.alignment_id
        or alignment_manifest.alignment_semantic_sha256 != action.alignment_semantic_sha256
    ):
        raise BoundaryRefinementProjectionError(
            "alignment manifest differs from the action/package trust closure"
        )
    subject = input_plan.subject.packages
    if len(subject) != len(package_set.members) or any(
        planned.package_id != member.package_id
        or planned.ordinal != member.ordinal
        or planned.semantic_content_sha256 != member.package_semantic_content_sha256
        or planned.manifest_bytes_sha256 != member.package_manifest_sha256
        for planned, member in zip(subject, package_set.members, strict=True)
    ):
        raise BoundaryRefinementProjectionError(
            "boundary input plan does not bind the exact package members"
        )
    parts = input_plan.call_plan.parts
    if not enriched_outputs or len(enriched_outputs) != len(parts):
        raise BoundaryRefinementProjectionError(
            "boundary projection requires one enriched output per call part"
        )
    for part, output in zip(parts, enriched_outputs, strict=True):
        if (
            output.task is not VisionTask.BOUNDARY_REFINEMENT
            or output.abstained
            or output.input_plan_id != input_plan.input_plan_id
            or output.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or output.request_catalog_id != input_plan.request_catalog.request_catalog_id
            or output.request_catalog_sha256 != input_plan.request_catalog.semantic_sha256
            or output.provider_claim_schema.sha256
            != input_plan.prompt_output.provider_response_schema_sha256
            or output.enriched_output_schema.sha256
            != input_plan.prompt_output.enriched_domain_schema_sha256
            or output.authority.recording_identity != alignment_manifest.recording_identity
            or output.authority.mcap_id != package_set.mcap_id
            or output.authority.camera_mapping_run_id != package_set.camera_mapping_run_id
            or output.authority.alignment_id != package_set.alignment_id
            or output.authority.prompt_version != input_plan.prompt_output.prompt_version
            or output.authority.prompt_sha256 != input_plan.prompt_output.prompt_sha256
        ):
            raise BoundaryRefinementProjectionError(
                "boundary enrichment is foreign to its role input plan"
            )
        expected_coordinates = {
            (item.package_ordinal, item.camera_ordinal)
            for item in input_plan.rendered_items[
                part.start_item_ordinal : part.end_item_ordinal_exclusive
            ]
        }
        actual_coordinates = [
            (claim.package_ordinal, CAMERA_IDS.index(claim.camera_id))
            for claim in output.claims
            if claim.kind is ProviderClaimKind.BOUNDARY_OBSERVATION
            and claim.package_ordinal is not None
            and claim.camera_id is not None
        ]
        if (
            any(claim.kind is not ProviderClaimKind.BOUNDARY_OBSERVATION for claim in output.claims)
            or len(actual_coordinates) != len(set(actual_coordinates))
            or set(actual_coordinates) != expected_coordinates
        ):
            raise BoundaryRefinementProjectionError(
                "each call part must cover its exact package-camera coordinates"
            )


def _package_ref(
    member: TemporalPackageSetMember,
) -> BoundaryRefinementPackageRef:
    return BoundaryRefinementPackageRef(
        package_id=member.package_id,
        package_ordinal=member.ordinal,
        interval=NanosecondInterval(
            start_ns=member.start_ns,
            end_ns=member.end_ns,
        ),
        semantic_content_sha256=member.package_semantic_content_sha256,
        manifest_sha256=member.package_manifest_sha256,
    )


def _output_ref(
    part: InferenceCallPart,
    output: OrchestratorEnrichedOutput,
) -> BoundaryRefinementOutputRef:
    selected = output.selected_attempt
    return BoundaryRefinementOutputRef(
        part_ordinal=part.ordinal,
        part_semantic_sha256=part.part_semantic_sha256,
        source_inference_id=output.authority.inference_id,
        source_artifact_id=output.artifact_id,
        selected_output_sha256=selected.output_sha256,
        selection_decision_logical_key=selected.selection_decision_logical_key,
        enrichment_logical_key=output.enrichment_logical_key,
    )


def _claim_ref(
    output_ref: BoundaryRefinementOutputRef,
    claim: EnrichedProviderClaim,
) -> BoundaryRefinementClaimRef:
    return BoundaryRefinementClaimRef(
        output=output_ref,
        claim_ordinal=claim.claim_ordinal,
        claim_id=claim.claim_id,
    )


def _normalize_boundary_observation(
    *,
    claim: EnrichedProviderClaim,
    output_ref: BoundaryRefinementOutputRef,
    role: BoundaryRefinementRole,
    package_set: TemporalPackageSet,
) -> NormalizedBoundaryObservation:
    if (
        claim.kind is not ProviderClaimKind.BOUNDARY_OBSERVATION
        or claim.package_id is None
        or claim.package_ordinal is None
        or claim.camera_id is None
        or claim.package_ordinal >= len(package_set.members)
    ):
        raise BoundaryRefinementProjectionError(
            "boundary observation lacks package-camera authority"
        )
    member = package_set.members[claim.package_ordinal]
    if claim.package_id != member.package_id:
        raise BoundaryRefinementProjectionError("boundary observation cites a foreign package")
    interval = _claim_interval(claim)
    if claim.observation not in _BOUNDARY_OBSERVATIONS:
        raise BoundaryRefinementProjectionError("invalid BOUNDARY_OBSERVATION status")
    if claim.observation is ProviderObservation.OBSERVED and (
        interval is None or not claim.evidence
    ):
        raise BoundaryRefinementProjectionError(
            "OBSERVED boundary claims require interval and evidence"
        )
    if claim.observation is ProviderObservation.MISSING and (
        interval is not None or claim.evidence
    ):
        raise BoundaryRefinementProjectionError(
            "MISSING boundary claims cannot assert interval or evidence"
        )
    if interval is not None and not _interval_inside_member(interval, member):
        raise BoundaryRefinementProjectionError(
            "role boundary interval lies outside its exact package"
        )
    values: dict[str, Any] = {
        "source": _claim_ref(output_ref, claim),
        "package_id": claim.package_id,
        "package_ordinal": claim.package_ordinal,
        "camera_id": claim.camera_id,
        "role": role,
        "boundary_interval": interval,
        "label": _normalized_label(claim.label),
        "observation": claim.observation,
        "evidence": tuple(sorted(claim.evidence, key=_evidence_sort_key)),
        "provider_conflict_codes": tuple(sorted(set(claim.conflict_codes))),
        "model_reported_score": (
            claim.model_reported_confidence.value
            if claim.model_reported_confidence is not None
            else None
        ),
        "production_eligible": False,
    }
    draft = NormalizedBoundaryObservation.model_construct(
        logical_key=f"{BOUNDARY_OBSERVATION_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
        **values,
    )
    digest = semantic_sha256(boundary_observation_semantic_projection(draft))
    return NormalizedBoundaryObservation.model_validate(
        {
            **values,
            "logical_key": f"{BOUNDARY_OBSERVATION_LOGICAL_KEY_NAMESPACE}:{digest}",
        },
        strict=True,
    )


def _validate_claim_evidence_visible(
    *,
    claim: EnrichedProviderClaim,
    visible_items: Mapping[int, object],
    package_set: TemporalPackageSet,
) -> None:
    for evidence in claim.evidence:
        rendered = visible_items.get(evidence.provider_item_ordinal)
        if rendered is None:
            raise BoundaryRefinementProjectionError(
                "boundary evidence is outside its selected call part"
            )
        if evidence.package_ordinal >= len(package_set.members):
            raise BoundaryRefinementProjectionError(
                "boundary evidence cites an unknown package ordinal"
            )
        member = package_set.members[evidence.package_ordinal]
        # Replayed evidence keeps its original exact manifest; logical equivalence is
        # established by package content plus the exact camera/frame/time facts below.
        if (
            getattr(rendered, "package_id", None) != evidence.package_id
            or getattr(rendered, "package_ordinal", None) != evidence.package_ordinal
            or getattr(rendered, "camera_id", None) is not evidence.camera_id
            or getattr(rendered, "camera_ordinal", None) != evidence.camera_ordinal
            or getattr(rendered, "frame_id", None) != evidence.frame_id
            or getattr(rendered, "frame_ordinal", None) != evidence.frame_ordinal
            or getattr(rendered, "aligned_timestamp_ns", None) != evidence.aligned_timestamp_ns
            or getattr(rendered, "source_timestamp_ns", None) != evidence.source_timestamp_ns
            or getattr(rendered, "source_artifact_sha256", None) != evidence.source_artifact_sha256
            or evidence.package_id != member.package_id
            or evidence.package_semantic_content_sha256 != member.package_semantic_content_sha256
        ):
            raise BoundaryRefinementProjectionError(
                "boundary evidence differs from authoritative input coordinates"
            )


def _build_camera_evidence(
    *,
    camera_id: CameraId,
    role: BoundaryRefinementRole,
    observations: Sequence[NormalizedBoundaryObservation],
    alignment_manifest: AlignmentManifestV2,
    package_set: TemporalPackageSet,
    window_interval: NanosecondInterval,
) -> CameraBoundaryEvidence:
    ordered = tuple(sorted(observations, key=_observation_sort_key))
    alignment = alignment_manifest.cameras[camera_id.value]
    package_intervals = {
        member.ordinal: NanosecondInterval(
            start_ns=member.start_ns,
            end_ns=member.end_ns,
        )
        for member in package_set.members
    }
    (
        outcome,
        observed_interval,
        estimate,
        uncertainty,
        ambiguity_codes,
    ) = _camera_reduction_values(
        observations=ordered,
        alignment_status=alignment.status,
        alignment_max_error_ns=alignment.max_error_ns,
        window_interval=window_interval,
        package_intervals=package_intervals,
    )
    return CameraBoundaryEvidence(
        camera_id=camera_id,
        role=role,
        observations=ordered,
        outcome=outcome,
        observed_interval=observed_interval,
        boundary_estimate_ns=estimate,
        uncertainty_ns=uncertainty,
        alignment_status=alignment.status,
        alignment_residual_p95_ns=alignment.residual_p95_ns,
        alignment_max_error_ns=alignment.max_error_ns,
        ambiguity_codes=ambiguity_codes,
        production_eligible=False,
    )


def _camera_reduction_values(
    *,
    observations: Sequence[NormalizedBoundaryObservation],
    alignment_status: AlignmentStatus,
    alignment_max_error_ns: int,
    window_interval: NanosecondInterval,
    package_intervals: Mapping[int, NanosecondInterval],
) -> tuple[
    BoundaryCameraOutcome,
    NanosecondInterval | None,
    int | None,
    int | None,
    tuple[str, ...],
]:
    observed = tuple(
        item for item in observations if item.observation is ProviderObservation.OBSERVED
    )
    codes: set[str] = set()
    if alignment_status is AlignmentStatus.DEGRADED:
        codes.add("ALIGNMENT_DEGRADED")
    statuses = {item.observation for item in observations}
    if ProviderObservation.OCCLUDED in statuses:
        codes.add("CAMERA_OCCLUDED")
    if ProviderObservation.UNUSABLE in statuses:
        codes.add("CAMERA_UNUSABLE")
    if ProviderObservation.MISSING in statuses:
        codes.add("CAMERA_MISSING")
    if observed and len(observed) != len(observations):
        codes.add("MIXED_CAMERA_OBSERVATIONS")
    if not observed:
        outcome = (
            BoundaryCameraOutcome.NO_BOUNDARY
            if observations
            and all(item.observation is ProviderObservation.NO_BOUNDARY for item in observations)
            else BoundaryCameraOutcome.INDETERMINATE
        )
        return outcome, None, None, None, tuple(sorted(codes))

    raw_intervals: list[NanosecondInterval] = []
    for item in observed:
        interval = item.boundary_interval
        if interval is None:
            raise ValueError("observed boundary is missing its interval")
        package_interval = package_intervals.get(item.package_ordinal)
        if package_interval is None or not _interval_inside(interval, package_interval):
            raise ValueError("observed boundary references an undeclared package interval")
        if (
            interval.start_ns == package_interval.start_ns
            or interval.end_ns == package_interval.end_ns
        ):
            codes.add("PACKAGE_EDGE_CONTACT")
        raw_intervals.append(interval)

    raw_start = min(interval.start_ns for interval in raw_intervals)
    raw_end = max(interval.end_ns for interval in raw_intervals)
    expanded_start = raw_start - alignment_max_error_ns
    expanded_end = raw_end + alignment_max_error_ns
    if expanded_start < window_interval.start_ns or expanded_end > window_interval.end_ns:
        codes.add("WINDOW_EDGE_CONTACT")
    clipped_start = max(window_interval.start_ns, expanded_start)
    clipped_end = min(window_interval.end_ns, expanded_end)
    observed_interval = NanosecondInterval(
        start_ns=clipped_start,
        end_ns=clipped_end,
    )
    estimate = observed_interval.start_ns + observed_interval.duration_ns // 2
    uncertainty = (observed_interval.duration_ns + 1) // 2
    return (
        BoundaryCameraOutcome.OBSERVED,
        observed_interval,
        estimate,
        uncertainty,
        tuple(sorted(codes)),
    )


def _role_reduction_values(
    *,
    camera_evidence: SixCameraMap[CameraBoundaryEvidence],
    minimum_observed_cameras: int,
    window_interval: NanosecondInterval,
) -> tuple[
    int | None,
    int | None,
    NanosecondInterval | None,
    BoundaryRefinementOutcome,
    tuple[str, ...],
]:
    slots = tuple(
        camera_evidence[camera_id]
        for camera_id in CAMERA_IDS
        if camera_evidence[camera_id].outcome is BoundaryCameraOutcome.OBSERVED
    )
    codes = {
        code for camera_id in CAMERA_IDS for code in camera_evidence[camera_id].ambiguity_codes
    }
    if len(slots) < minimum_observed_cameras:
        codes.add("INSUFFICIENT_OBSERVED_CAMERAS")
        return (
            None,
            None,
            None,
            BoundaryRefinementOutcome.INDETERMINATE,
            tuple(sorted(codes)),
        )

    intervals: list[NanosecondInterval] = []
    centers: list[int] = []
    for slot in slots:
        interval = slot.observed_interval
        estimate = slot.boundary_estimate_ns
        if interval is None or estimate is None:
            raise ValueError("observed camera slot lacks reduction values")
        intervals.append(interval)
        centers.append(estimate)
    estimate = _median_low_int(centers)
    uncertainty = max(
        abs(center - estimate) + (interval.duration_ns + 1) // 2
        for center, interval in zip(centers, intervals, strict=True)
    )
    raw_start = estimate - uncertainty
    raw_end = estimate + uncertainty + 1
    if raw_start < window_interval.start_ns or raw_end > window_interval.end_ns:
        codes.add("ROLE_WINDOW_EDGE_CONTACT")
    if len(set(centers)) > 1:
        codes.add("CROSS_CAMERA_DISAGREEMENT")
    interval = NanosecondInterval(
        start_ns=max(window_interval.start_ns, raw_start),
        end_ns=min(window_interval.end_ns, raw_end),
    )
    return (
        estimate,
        uncertainty,
        interval,
        BoundaryRefinementOutcome.REFINED,
        tuple(sorted(codes)),
    )


def _action_reduction_values(
    onset: BoundaryRefinementRoleResult,
    offset: BoundaryRefinementRoleResult,
) -> tuple[
    NanosecondInterval | None,
    NanosecondInterval | None,
    NanosecondInterval | None,
    int | None,
    int | None,
    int | None,
    BoundaryRefinementOutcome,
    tuple[str, ...],
]:
    codes = {f"ONSET_{code}" for code in onset.ambiguity_codes}
    codes.update(f"OFFSET_{code}" for code in offset.ambiguity_codes)
    if onset.outcome is not BoundaryRefinementOutcome.REFINED:
        codes.add("ONSET_INDETERMINATE")
    if offset.outcome is not BoundaryRefinementOutcome.REFINED:
        codes.add("OFFSET_INDETERMINATE")
    if (
        onset.outcome is not BoundaryRefinementOutcome.REFINED
        or offset.outcome is not BoundaryRefinementOutcome.REFINED
    ):
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            BoundaryRefinementOutcome.INDETERMINATE,
            tuple(sorted(codes)),
        )

    onset_estimate = onset.boundary_estimate_ns
    offset_estimate = offset.boundary_estimate_ns
    onset_interval = onset.boundary_interval
    offset_interval = offset.boundary_interval
    onset_uncertainty = onset.uncertainty_ns
    offset_uncertainty = offset.uncertainty_ns
    if (
        onset_estimate is None
        or offset_estimate is None
        or onset_interval is None
        or offset_interval is None
        or onset_uncertainty is None
        or offset_uncertainty is None
    ):
        raise ValueError("refined role result lacks deterministic reduction values")
    if onset_estimate >= offset_estimate:
        codes.add("BOUNDARY_ORDER_CONFLICT")
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            BoundaryRefinementOutcome.INDETERMINATE,
            tuple(sorted(codes)),
        )
    if onset_interval.end_ns > offset_interval.start_ns:
        codes.add("BOUNDARY_UNCERTAINTY_OVERLAP")
    return (
        NanosecondInterval(
            start_ns=onset_estimate,
            end_ns=offset_estimate,
        ),
        onset_interval,
        offset_interval,
        onset_estimate,
        offset_estimate,
        max(onset_uncertainty, offset_uncertainty),
        BoundaryRefinementOutcome.REFINED,
        tuple(sorted(codes)),
    )


def _validate_role_result_shape(result: BoundaryRefinementRoleResult) -> None:
    if tuple(item.package_ordinal for item in result.packages) != tuple(
        range(len(result.packages))
    ):
        raise ValueError("boundary packages must be stored in ordinal order")
    if len({item.package_id for item in result.packages}) != len(result.packages):
        raise ValueError("boundary package IDs must be unique")
    if tuple(item.part_ordinal for item in result.source_outputs) != tuple(
        range(len(result.source_outputs))
    ):
        raise ValueError("boundary outputs must be stored in call-part order")
    if len({item.part_semantic_sha256 for item in result.source_outputs}) != len(
        result.source_outputs
    ):
        raise ValueError("boundary output part identities must be unique")
    expected_anchor = (
        result.coarse_interval.start_ns
        if result.role is BoundaryRefinementRole.ONSET
        else result.coarse_interval.end_ns
    )
    if result.coarse_anchor_ns != expected_anchor:
        raise ValueError("boundary role anchor differs from the coarse action")
    if (
        result.window_interval.start_ns < result.requested_window_interval.start_ns
        or result.window_interval.end_ns > result.requested_window_interval.end_ns
        or result.window_interval.start_ns > result.coarse_anchor_ns
        or result.window_interval.end_ns < result.coarse_anchor_ns
    ):
        raise ValueError("boundary role window bounds are inconsistent")

    package_by_ordinal = {item.package_ordinal: item for item in result.packages}
    output_refs = tuple(result.source_outputs)
    for camera_id in CAMERA_IDS:
        slot = result.camera_evidence[camera_id]
        if slot.camera_id is not camera_id or slot.role is not result.role:
            raise ValueError("boundary camera slot identity is inconsistent")
        expected_camera = _camera_reduction_values(
            observations=slot.observations,
            alignment_status=slot.alignment_status,
            alignment_max_error_ns=slot.alignment_max_error_ns,
            window_interval=result.window_interval,
            package_intervals={
                ordinal: package.interval for ordinal, package in package_by_ordinal.items()
            },
        )
        actual_camera = (
            slot.outcome,
            slot.observed_interval,
            slot.boundary_estimate_ns,
            slot.uncertainty_ns,
            slot.ambiguity_codes,
        )
        if actual_camera != expected_camera:
            raise ValueError("camera boundary reduction is inconsistent")
        if slot.observed_interval is not None and not _interval_inside(
            slot.observed_interval, result.window_interval
        ):
            raise ValueError("camera boundary interval lies outside its role window")
        for observation in slot.observations:
            package = package_by_ordinal.get(observation.package_ordinal)
            if (
                package is None
                or observation.package_id != package.package_id
                or observation.source.output not in output_refs
            ):
                raise ValueError("boundary observation references undeclared lineage")
            if observation.boundary_interval is not None and not _interval_inside(
                observation.boundary_interval, package.interval
            ):
                raise ValueError("boundary observation lies outside its package")
            if any(
                evidence.package_id != package.package_id
                or evidence.package_ordinal != package.package_ordinal
                or evidence.package_semantic_content_sha256 != package.semantic_content_sha256
                or evidence.camera_id is not camera_id
                for evidence in observation.evidence
            ):
                raise ValueError("boundary evidence references undeclared lineage")

    observed_count = sum(
        result.camera_evidence[camera_id].outcome is BoundaryCameraOutcome.OBSERVED
        for camera_id in CAMERA_IDS
    )
    expected_reduction = _role_reduction_values(
        camera_evidence=result.camera_evidence,
        minimum_observed_cameras=result.policy.minimum_observed_cameras,
        window_interval=result.window_interval,
    )
    actual_reduction = (
        result.boundary_estimate_ns,
        result.uncertainty_ns,
        result.boundary_interval,
        result.outcome,
        result.ambiguity_codes,
    )
    if result.observed_camera_count != observed_count or actual_reduction != expected_reduction:
        raise ValueError("boundary role reduction is inconsistent")


def _validate_role_pair(
    *,
    action: ProvisionalPhysicalAction,
    onset: BoundaryRefinementRoleResult,
    offset: BoundaryRefinementRoleResult,
    policy: BoundaryRefinementPolicy,
) -> None:
    if (
        onset.role is not BoundaryRefinementRole.ONSET
        or offset.role is not BoundaryRefinementRole.OFFSET
    ):
        raise BoundaryRefinementProjectionError(
            "action reduction requires exact ONSET and OFFSET role results"
        )
    if onset.policy != policy or offset.policy != policy:
        raise BoundaryRefinementProjectionError(
            "boundary role results use a foreign reduction policy"
        )
    if onset.window_semantic_sha256 == offset.window_semantic_sha256:
        raise BoundaryRefinementProjectionError(
            "ONSET and OFFSET require independently role-bound windows"
        )
    expected_action = (
        action.logical_key,
        action.semantic_sha256,
        action.policy_semantic_sha256,
        action.ordinal,
        action.label,
        action.coarse_interval,
        action.mcap_id,
        action.source_content_sha256,
        action.camera_mapping_semantic_sha256,
        action.alignment_semantic_sha256,
    )
    for result in (onset, offset):
        actual_action = (
            result.source_action_logical_key,
            result.source_action_semantic_sha256,
            result.source_action_policy_semantic_sha256,
            result.source_action_ordinal,
            result.action_label,
            result.coarse_interval,
            result.mcap_id,
            result.source_content_sha256,
            result.camera_mapping_semantic_sha256,
            result.alignment_semantic_sha256,
        )
        if actual_action != expected_action:
            raise BoundaryRefinementProjectionError(
                "boundary role result differs from its provisional action"
            )
    onset_context = (
        onset.recording_identity,
        onset.mcap_id,
        onset.source_content_sha256,
        onset.camera_mapping_run_id,
        onset.camera_mapping_semantic_sha256,
        onset.alignment_id,
        onset.alignment_semantic_sha256,
    )
    offset_context = (
        offset.recording_identity,
        offset.mcap_id,
        offset.source_content_sha256,
        offset.camera_mapping_run_id,
        offset.camera_mapping_semantic_sha256,
        offset.alignment_id,
        offset.alignment_semantic_sha256,
    )
    if onset_context != offset_context:
        raise BoundaryRefinementProjectionError(
            "ONSET and OFFSET do not share one recording trust closure"
        )


def _validate_action_result_shape(result: BoundaryRefinementResult) -> None:
    if (
        result.onset.role is not BoundaryRefinementRole.ONSET
        or result.offset.role is not BoundaryRefinementRole.OFFSET
        or result.onset.policy != result.policy
        or result.offset.policy != result.policy
        or result.onset.window_semantic_sha256 == result.offset.window_semantic_sha256
    ):
        raise ValueError("boundary action result has invalid role closure")
    expected_source = (
        result.source_action_logical_key,
        result.source_action_semantic_sha256,
        result.source_action_ordinal,
        result.action_label,
        result.coarse_interval,
        result.mcap_id,
        result.source_content_sha256,
        result.camera_mapping_semantic_sha256,
        result.alignment_semantic_sha256,
    )
    for role_result in (result.onset, result.offset):
        actual_source = (
            role_result.source_action_logical_key,
            role_result.source_action_semantic_sha256,
            role_result.source_action_ordinal,
            role_result.action_label,
            role_result.coarse_interval,
            role_result.mcap_id,
            role_result.source_content_sha256,
            role_result.camera_mapping_semantic_sha256,
            role_result.alignment_semantic_sha256,
        )
        if actual_source != expected_source:
            raise ValueError("boundary action result differs from its role source")
    expected_reduction = _action_reduction_values(result.onset, result.offset)
    actual_reduction = (
        result.refined_interval,
        result.onset_interval,
        result.offset_interval,
        result.onset_estimate_ns,
        result.offset_estimate_ns,
        result.uncertainty_ns,
        result.outcome,
        result.ambiguity_codes,
    )
    if actual_reduction != expected_reduction:
        raise ValueError("boundary action reduction is inconsistent")


def _camera_semantic_projection(slot: CameraBoundaryEvidence) -> dict[str, object]:
    return {
        "camera_id": slot.camera_id.value,
        "role": slot.role.value,
        "observation_logical_keys": [item.logical_key for item in slot.observations],
        "outcome": slot.outcome.value,
        "observed_interval": _interval_projection(slot.observed_interval),
        "boundary_estimate_ns": _nanoseconds_projection(slot.boundary_estimate_ns),
        "uncertainty_ns": _nanoseconds_projection(slot.uncertainty_ns),
        "alignment_status": slot.alignment_status.value,
        "alignment_residual_p95_ns": str(slot.alignment_residual_p95_ns),
        "alignment_max_error_ns": str(slot.alignment_max_error_ns),
        "ambiguity_codes": list(slot.ambiguity_codes),
        "production_eligible": slot.production_eligible,
    }


def _evidence_semantic_projection(
    evidence: EnrichedEvidenceReference,
) -> dict[str, object]:
    return {
        "provider_item_ordinal": evidence.provider_item_ordinal,
        "package_ordinal": evidence.package_ordinal,
        "package_semantic_content_sha256": (evidence.package_semantic_content_sha256),
        "camera_ordinal": evidence.camera_ordinal,
        "frame_ordinal": evidence.frame_ordinal,
        "aligned_timestamp_ns": str(evidence.aligned_timestamp_ns),
        "source_timestamp_ns": str(evidence.source_timestamp_ns),
        "source_artifact_sha256": evidence.source_artifact_sha256,
    }


def _evidence_sort_key(
    evidence: EnrichedEvidenceReference,
) -> tuple[int, int, int, int]:
    return (
        evidence.provider_item_ordinal,
        evidence.package_ordinal,
        evidence.camera_ordinal,
        evidence.frame_ordinal,
    )


def _observation_sort_key(
    observation: NormalizedBoundaryObservation,
) -> tuple[int, int, int, str]:
    return (
        observation.source.output.part_ordinal,
        observation.package_ordinal,
        observation.source.claim_ordinal,
        observation.logical_key,
    )


def _claim_interval(claim: EnrichedProviderClaim) -> NanosecondInterval | None:
    if claim.interval is None:
        return None
    return NanosecondInterval(
        start_ns=claim.interval.start_ns,
        end_ns=claim.interval.end_ns,
    )


def _interval_inside_member(
    interval: NanosecondInterval,
    member: TemporalPackageSetMember,
) -> bool:
    return interval.start_ns >= member.start_ns and interval.end_ns <= member.end_ns


def _interval_inside(
    inner: NanosecondInterval,
    outer: NanosecondInterval,
) -> bool:
    return inner.start_ns >= outer.start_ns and inner.end_ns <= outer.end_ns


def _interval_projection(interval: NanosecondInterval | None) -> object:
    return interval.model_dump(mode="json") if interval is not None else None


def _nanoseconds_projection(value: int | None) -> str | None:
    return str(value) if value is not None else None


def _normalized_label(label: str | None) -> str | None:
    if label is None:
        return None
    normalized = label.strip().lower()
    if not normalized:
        raise BoundaryRefinementProjectionError("boundary observation label cannot be blank")
    return normalized


def _median_low_int(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


__all__ = [
    "BOUNDARY_OBSERVATION_LOGICAL_KEY_NAMESPACE",
    "BOUNDARY_OBSERVATION_PROJECTION_VERSION",
    "BOUNDARY_REFINEMENT_POLICY_PROJECTION_VERSION",
    "BOUNDARY_RESULT_LOGICAL_KEY_NAMESPACE",
    "BOUNDARY_RESULT_PROJECTION_VERSION",
    "BOUNDARY_ROLE_RESULT_LOGICAL_KEY_NAMESPACE",
    "BOUNDARY_ROLE_RESULT_PROJECTION_VERSION",
    "LOCAL_BOUNDARY_REFINEMENT_POLICY_VERSION",
    "BoundaryCameraOutcome",
    "BoundaryRefinementClaimRef",
    "BoundaryRefinementOutcome",
    "BoundaryRefinementOutputRef",
    "BoundaryRefinementPackageRef",
    "BoundaryRefinementPolicy",
    "BoundaryRefinementProjectionError",
    "BoundaryRefinementProjector",
    "BoundaryRefinementResult",
    "BoundaryRefinementRole",
    "BoundaryRefinementRoleResult",
    "CameraBoundaryEvidence",
    "NormalizedBoundaryObservation",
    "boundary_observation_semantic_projection",
    "boundary_output_semantic_projection",
    "boundary_refinement_policy_semantic_projection",
    "boundary_refinement_result_semantic_projection",
    "boundary_role_result_semantic_projection",
]
