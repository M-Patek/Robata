"""Provider-neutral materialization for frozen supplemental explicit targets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import Rfc3339Timestamp
from robata.contracts.schema_registry import SchemaRef
from robata.sampling.grid import (
    FrameCandidate,
    SamplingTarget,
    SelectionStatus,
    select_nearest_frames,
)
from robata.sampling.materializer import (
    CanonicalSixCameraFrameIndex,
    FrameArtifactResolver,
    IndexedSourceFrame,
    MaterializedArtifactManifest,
    MaterializedFrameArtifactFact,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

SUPPLEMENTAL_TARGET_PLAN_PROJECTION_VERSION = "supplemental-explicit-target-plan-semantic-v2"
SUPPLEMENTAL_PACKAGE_PROJECTION_VERSION = "supplemental-temporal-package-semantic-v2"
SUPPLEMENTAL_PACKAGE_SCHEMA_VERSION = "2.0"
SUPPLEMENTAL_MATERIALIZER_VERSION = "supplemental-explicit-target-materializer-v2"
SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION = "nearest-absolute-delta-v1"
SUPPLEMENTAL_DEDUPE_POLICY_VERSION = "one-source-frame-v1"
MEDIA_QUALITY_REPORT_SCHEMA_ID = "https://schemas.robata.dev/media-quality-report"
MEDIA_QUALITY_REPORT_SCHEMA_VERSION = "1.0.0"


class SupplementalEvidenceClass(StrEnum):
    """Promotion class for local deterministic media evidence."""

    LOCAL_CONFORMANCE = "LOCAL_CONFORMANCE"


class RegisteredMediaQualitySourceBinding(StrictModel):
    """Provider-neutral, self-validating report and admitted-source lineage."""

    report_schema_ref: SchemaRef
    report_semantic_sha256: Sha256Digest
    supplemental_target_plan_semantic_sha256: Sha256Digest
    media_quality_binding_semantic_sha256: Sha256Digest
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    projection_version: Literal["media-quality-source-binding-semantic-v1"] = (
        "media-quality-source-binding-semantic-v1"
    )
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = semantic_sha256(media_quality_source_binding_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("media-quality source binding semantic_sha256 is inconsistent")
        return self


def media_quality_source_binding_projection(
    binding: RegisteredMediaQualitySourceBinding,
) -> dict[str, object]:
    """Return the complete versioned report/source lineage projection."""

    return {
        "projection_version": binding.projection_version,
        "report_schema_ref": binding.report_schema_ref,
        "report_semantic_sha256": binding.report_semantic_sha256,
        "supplemental_target_plan_semantic_sha256": (
            binding.supplemental_target_plan_semantic_sha256
        ),
        "media_quality_binding_semantic_sha256": (binding.media_quality_binding_semantic_sha256),
        "source_content_sha256": binding.source_content_sha256,
        "camera_mapping_semantic_sha256": binding.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": binding.alignment_semantic_sha256,
    }


class FrozenSupplementalTarget(StrictModel):
    """One explicitly resolved target copied from immutable trigger evidence."""

    ordinal: NonNegativeInt
    camera_id: CameraId
    target_ns: Nanoseconds


class FrozenSupplementalTargetPlan(StrictModel):
    """Frozen explicit targets whose complete coordinates define logical identity."""

    plan_id: NonEmptyString
    semantic_sha256: Sha256Digest
    source_report_schema_ref: SchemaRef
    source_report_semantic_sha256: Sha256Digest
    source_target_plan_semantic_sha256: Sha256Digest
    source_binding: RegisteredMediaQualitySourceBinding
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    effective_interval: NanosecondInterval
    targets: tuple[FrozenSupplementalTarget, ...]
    selection_tolerance_ns: Annotated[Nanoseconds, Field(ge=0)]
    tie_break_policy_version: SchemaVersion
    dedupe_policy_version: SchemaVersion
    target_policy_version: SchemaVersion
    projection_version: Literal["supplemental-explicit-target-plan-semantic-v2"] = (
        "supplemental-explicit-target-plan-semantic-v2"
    )
    evidence_class: Literal[SupplementalEvidenceClass.LOCAL_CONFORMANCE] = (
        SupplementalEvidenceClass.LOCAL_CONFORMANCE
    )
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if (
            self.source_report_schema_ref.schema_id != MEDIA_QUALITY_REPORT_SCHEMA_ID
            or self.source_report_schema_ref.version != MEDIA_QUALITY_REPORT_SCHEMA_VERSION
        ):
            raise ValueError("source_report_schema_ref must identify media-quality-report@1.0.0")
        if (
            self.source_binding.report_schema_ref != self.source_report_schema_ref
            or self.source_binding.report_semantic_sha256 != self.source_report_semantic_sha256
            or self.source_binding.supplemental_target_plan_semantic_sha256
            != self.source_target_plan_semantic_sha256
            or self.source_binding.source_content_sha256 != self.source_content_sha256
            or self.source_binding.camera_mapping_semantic_sha256
            != self.camera_mapping_semantic_sha256
            or self.source_binding.alignment_semantic_sha256 != self.alignment_semantic_sha256
        ):
            raise ValueError("source_binding does not bind the frozen target plan lineage")
        if not self.targets:
            raise ValueError("supplemental target plan must contain at least one target")
        expected_order = tuple(
            sorted(
                self.targets,
                key=lambda target: (target.target_ns, CAMERA_IDS.index(target.camera_id)),
            )
        )
        if self.targets != expected_order:
            raise ValueError("supplemental targets must use canonical camera/time order")
        if tuple(target.ordinal for target in self.targets) != tuple(range(len(self.targets))):
            raise ValueError("supplemental target ordinals must be contiguous from zero")
        coordinates = tuple((target.camera_id, target.target_ns) for target in self.targets)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("supplemental target coordinates must be unique")
        if any(not self.effective_interval.contains(target.target_ns) for target in self.targets):
            raise ValueError("supplemental targets must lie inside effective_interval")
        expected_digest = semantic_sha256(supplemental_target_plan_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match supplemental target plan")
        if self.plan_id != _stable_id("supplemental-target-plan-v2", expected_digest):
            raise ValueError("plan_id does not match supplemental target plan identity")
        return self


def supplemental_target_plan_projection(
    plan: FrozenSupplementalTargetPlan,
) -> dict[str, object]:
    """Return the versioned logical projection for a frozen explicit target plan."""

    return {
        "projection_version": plan.projection_version,
        "source_report_schema_ref": plan.source_report_schema_ref,
        "source_report_semantic_sha256": plan.source_report_semantic_sha256,
        "source_target_plan_semantic_sha256": plan.source_target_plan_semantic_sha256,
        "source_binding_semantic_sha256": plan.source_binding.semantic_sha256,
        "source_content_sha256": plan.source_content_sha256,
        "camera_mapping_semantic_sha256": plan.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": plan.alignment_semantic_sha256,
        "effective_interval": plan.effective_interval,
        "resolution_mode": "EXPLICIT_TARGETS",
        "targets": plan.targets,
        "selection_tolerance_ns": str(plan.selection_tolerance_ns),
        "tie_break_policy_version": plan.tie_break_policy_version,
        "dedupe_policy_version": plan.dedupe_policy_version,
        "target_policy_version": plan.target_policy_version,
        "evidence_class": plan.evidence_class.value,
        "production_eligible": plan.production_eligible,
    }


def build_frozen_supplemental_target_plan(
    *,
    source_binding: RegisteredMediaQualitySourceBinding,
    effective_interval: NanosecondInterval,
    targets: Sequence[tuple[CameraId, Nanoseconds]],
    selection_tolerance_ns: Nanoseconds,
    tie_break_policy_version: str,
    dedupe_policy_version: str,
    target_policy_version: str,
) -> FrozenSupplementalTargetPlan:
    """Freeze canonical explicit coordinates before package identity is computed."""

    ordered = tuple(sorted(set(targets), key=lambda item: (item[1], CAMERA_IDS.index(item[0]))))
    frozen_targets = tuple(
        FrozenSupplementalTarget(ordinal=ordinal, camera_id=camera_id, target_ns=target_ns)
        for ordinal, (camera_id, target_ns) in enumerate(ordered)
    )
    values: dict[str, Any] = {
        "source_report_schema_ref": source_binding.report_schema_ref,
        "source_report_semantic_sha256": source_binding.report_semantic_sha256,
        "source_target_plan_semantic_sha256": (
            source_binding.supplemental_target_plan_semantic_sha256
        ),
        "source_binding": source_binding,
        "source_content_sha256": source_binding.source_content_sha256,
        "camera_mapping_semantic_sha256": (source_binding.camera_mapping_semantic_sha256),
        "alignment_semantic_sha256": source_binding.alignment_semantic_sha256,
        "effective_interval": effective_interval,
        "targets": frozen_targets,
        "selection_tolerance_ns": selection_tolerance_ns,
        "tie_break_policy_version": tie_break_policy_version,
        "dedupe_policy_version": dedupe_policy_version,
        "target_policy_version": target_policy_version,
        "projection_version": SUPPLEMENTAL_TARGET_PLAN_PROJECTION_VERSION,
        "evidence_class": SupplementalEvidenceClass.LOCAL_CONFORMANCE,
        "production_eligible": False,
    }
    draft = FrozenSupplementalTargetPlan.model_construct(
        plan_id="pending",
        semantic_sha256="0" * 64,
        **values,
    )
    digest = semantic_sha256(supplemental_target_plan_projection(draft))
    return FrozenSupplementalTargetPlan.model_validate(
        {
            **values,
            "plan_id": _stable_id("supplemental-target-plan-v2", digest),
            "semantic_sha256": digest,
        },
        strict=True,
    )


class SupplementalSourceFrameRef(StrictModel):
    """Exact selected or failed source-frame provenance."""

    source_frame_id: NonEmptyString
    aligned_timestamp_ns: Nanoseconds
    source_timestamp_ns: Nanoseconds
    source_locator: dict[str, str | int | bool]
    alignment_projection_id: NonEmptyString


class SupplementalSelectedArtifact(StrictModel):
    """Provider-neutral immutable visual artifact facts."""

    artifact: MaterializedArtifactManifest
    width: Annotated[int, Field(strict=True, ge=1)]
    height: Annotated[int, Field(strict=True, ge=1)]
    quality_flags: tuple[NonEmptyString, ...] = ()


class SupplementalTargetOutcome(StrictModel):
    """Auditable nearest-frame outcome for one frozen explicit target."""

    target: FrozenSupplementalTarget
    status: SelectionStatus
    delta_to_target_ns: Nanoseconds | None = None
    source_frame: SupplementalSourceFrameRef | None = None
    selected_artifact: SupplementalSelectedArtifact | None = None
    reused_selected_target_ordinal: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status is SelectionStatus.NO_FRAME_WITHIN_TOLERANCE:
            if any(
                value is not None
                for value in (
                    self.delta_to_target_ns,
                    self.source_frame,
                    self.selected_artifact,
                    self.reused_selected_target_ordinal,
                )
            ):
                raise ValueError("a missed supplemental target cannot contain frame facts")
            return self
        if self.source_frame is None or self.delta_to_target_ns is None:
            raise ValueError("a source-associated supplemental outcome requires frame provenance")
        if (
            self.source_frame.aligned_timestamp_ns - self.target.target_ns
            != self.delta_to_target_ns
        ):
            raise ValueError("supplemental target delta is inconsistent")
        if self.status is SelectionStatus.SELECTED:
            if self.selected_artifact is None or self.reused_selected_target_ordinal is not None:
                raise ValueError("a selected supplemental target requires its artifact")
        elif self.status is SelectionStatus.DEDUPLICATED_FRAME:
            if self.selected_artifact is not None or self.reused_selected_target_ordinal is None:
                raise ValueError("a deduplicated target must reference its selected target")
        elif self.selected_artifact is not None or self.reused_selected_target_ordinal is not None:
            raise ValueError("a decode-failed target cannot contain materialized evidence")
        return self


class ProviderNeutralSupplementalPackage(StrictModel):
    """Immutable explicit-target visual package kept separate from grid-only V1."""

    schema_version: Literal["2.0"] = "2.0"
    package_id: NonEmptyString
    semantic_content_sha256: Sha256Digest
    target_plan_id: NonEmptyString
    target_plan_semantic_sha256: Sha256Digest
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    outcomes: tuple[SupplementalTargetOutcome, ...]
    selected_artifact_count: NonNegativeInt
    materializer_version: SchemaVersion
    projection_version: Literal["supplemental-temporal-package-semantic-v2"] = (
        "supplemental-temporal-package-semantic-v2"
    )
    evidence_class: Literal[SupplementalEvidenceClass.LOCAL_CONFORMANCE] = (
        SupplementalEvidenceClass.LOCAL_CONFORMANCE
    )
    production_eligible: Literal[False] = False
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if tuple(item.target.ordinal for item in self.outcomes) != tuple(range(len(self.outcomes))):
            raise ValueError("supplemental outcomes must cover ordered plan targets exactly")
        selected = {
            item.target.ordinal: item
            for item in self.outcomes
            if item.status is SelectionStatus.SELECTED
        }
        assignments_by_locator: dict[tuple[CameraId, bytes], list[SupplementalTargetOutcome]] = {}
        for item in self.outcomes:
            if item.status in {
                SelectionStatus.SELECTED,
                SelectionStatus.DEDUPLICATED_FRAME,
            }:
                assert item.source_frame is not None
                locator = (
                    item.target.camera_id,
                    canonical_json_bytes(item.source_frame.source_locator),
                )
                assignments_by_locator.setdefault(locator, []).append(item)
        for assignments in assignments_by_locator.values():
            selected_assignments = tuple(
                item for item in assignments if item.status is SelectionStatus.SELECTED
            )
            if len(selected_assignments) != 1:
                raise ValueError("one source locator must have exactly one selected target")
            winner = min(
                assignments,
                key=lambda item: (
                    abs(item.delta_to_target_ns or 0),
                    item.target.target_ns,
                    item.target.ordinal,
                ),
            )
            selected_assignment = selected_assignments[0]
            if selected_assignment is not winner:
                raise ValueError("selected target does not match the dedupe winner policy")
            if any(
                item.reused_selected_target_ordinal != selected_assignment.target.ordinal
                for item in assignments
                if item.status is SelectionStatus.DEDUPLICATED_FRAME
            ):
                raise ValueError("deduplicated target does not reference the policy winner")
        for item in self.outcomes:
            reused = item.reused_selected_target_ordinal
            if reused is not None:
                source = selected.get(reused)
                if source is None or source.source_frame != item.source_frame:
                    raise ValueError("deduplicated target does not resolve to a selected frame")
        if self.selected_artifact_count != len(selected):
            raise ValueError("selected_artifact_count is inconsistent")
        expected_digest = semantic_sha256(supplemental_package_projection(self))
        if self.semantic_content_sha256 != expected_digest:
            raise ValueError("semantic_content_sha256 does not match supplemental package")
        if self.package_id != _stable_id("supplemental-temporal-package-v2", expected_digest):
            raise ValueError("package_id does not match supplemental package identity")
        return self


def _source_frame_projection(frame: SupplementalSourceFrameRef) -> dict[str, object]:
    return {
        "aligned_timestamp_ns": str(frame.aligned_timestamp_ns),
        "source_timestamp_ns": str(frame.source_timestamp_ns),
        "source_locator": frame.source_locator,
    }


def _artifact_projection(artifact: SupplementalSelectedArtifact) -> dict[str, object]:
    return {
        "sha256": artifact.artifact.sha256,
        "bytes": artifact.artifact.bytes,
        "media_type": artifact.artifact.media_type,
        "width": artifact.width,
        "height": artifact.height,
        "quality_flags": artifact.quality_flags,
    }


def supplemental_package_projection(
    package: ProviderNeutralSupplementalPackage,
) -> dict[str, object]:
    """Project semantic package content, excluding locators and wall-clock fields."""

    return {
        "schema_version": package.schema_version,
        "projection_version": package.projection_version,
        "target_plan_id": package.target_plan_id,
        "target_plan_semantic_sha256": package.target_plan_semantic_sha256,
        "source_content_sha256": package.source_content_sha256,
        "camera_mapping_semantic_sha256": package.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": package.alignment_semantic_sha256,
        "outcomes": [
            {
                "target": item.target,
                "status": item.status.value,
                "delta_to_target_ns": (
                    None if item.delta_to_target_ns is None else str(item.delta_to_target_ns)
                ),
                "source_frame": (
                    None
                    if item.source_frame is None
                    else _source_frame_projection(item.source_frame)
                ),
                "selected_artifact": (
                    None
                    if item.selected_artifact is None
                    else _artifact_projection(item.selected_artifact)
                ),
                "reused_selected_target_ordinal": item.reused_selected_target_ordinal,
            }
            for item in package.outcomes
        ],
        "selected_artifact_count": package.selected_artifact_count,
        "materializer_version": package.materializer_version,
        "evidence_class": package.evidence_class.value,
        "production_eligible": package.production_eligible,
    }


@dataclass(frozen=True, slots=True)
class MaterializedSupplementalPackage:
    """Semantic package plus exact canonical manifest evidence."""

    package: ProviderNeutralSupplementalPackage
    manifest_bytes: bytes
    manifest_sha256: Sha256Digest

    def __post_init__(self) -> None:
        if self.manifest_bytes != canonical_json_bytes(self.package):
            raise ValueError("manifest_bytes must equal canonical supplemental package bytes")
        if exact_bytes_sha256(self.manifest_bytes) != self.manifest_sha256:
            raise ValueError("manifest_sha256 does not match supplemental package bytes")


class SupplementalPackageMaterializationError(ValueError):
    """Frozen plan and verified frame index cannot produce a valid package."""


class ExplicitTargetPackageMaterializer:
    """Select and materialize exact explicit targets without inventing a grid rate."""

    def __init__(self, version: str = SUPPLEMENTAL_MATERIALIZER_VERSION) -> None:
        if version != SUPPLEMENTAL_MATERIALIZER_VERSION:
            raise ValueError(f"materializer version must be {SUPPLEMENTAL_MATERIALIZER_VERSION!r}")
        self._version = version

    def materialize(
        self,
        *,
        plan: FrozenSupplementalTargetPlan,
        frame_index: CanonicalSixCameraFrameIndex,
        artifact_resolver: FrameArtifactResolver
        | Callable[[CameraId, IndexedSourceFrame], MaterializedFrameArtifactFact | None],
        created_at: str,
    ) -> MaterializedSupplementalPackage:
        """Resolve every frozen coordinate and bind selected immutable artifacts."""

        _validate_plan_index_binding(plan, frame_index)
        if plan.tie_break_policy_version != SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION:
            raise SupplementalPackageMaterializationError(
                "supplemental target plan uses an unsupported tie-break policy"
            )
        if plan.dedupe_policy_version != SUPPLEMENTAL_DEDUPE_POLICY_VERSION:
            raise SupplementalPackageMaterializationError(
                "supplemental target plan uses an unsupported dedupe policy"
            )
        if not callable(artifact_resolver):
            raise TypeError("artifact_resolver must be callable")

        outcomes: dict[int, SupplementalTargetOutcome] = {}
        for camera_id in CAMERA_IDS:
            camera_targets = tuple(
                target for target in plan.targets if target.camera_id is camera_id
            )
            if not camera_targets:
                continue
            source_index = frame_index.cameras[camera_id]
            by_locator = {
                canonical_json_bytes(frame.source_locator): frame for frame in source_index.frames
            }
            selections = select_nearest_frames(
                (
                    SamplingTarget(k=target.ordinal, target_ns=target.target_ns)
                    for target in camera_targets
                ),
                (
                    FrameCandidate(
                        aligned_timestamp_ns=frame.alignment_projection.aligned_timestamp_ns,
                        source_timestamp_ns=frame.source_timestamp_ns,
                        source_locator_bytes=canonical_json_bytes(frame.source_locator),
                        decodable=frame.decodable,
                    )
                    for frame in source_index.frames
                ),
                interval_start_ns=plan.effective_interval.start_ns,
                interval_end_ns=plan.effective_interval.end_ns,
                selection_tolerance_ns=plan.selection_tolerance_ns,
            )
            selected_by_locator = {
                selection.frame.source_locator_bytes: selection.target.k
                for selection in selections
                if selection.status is SelectionStatus.SELECTED and selection.frame is not None
            }
            targets_by_ordinal = {target.ordinal: target for target in camera_targets}
            for selection in selections:
                target = targets_by_ordinal[selection.target.k]
                frame = (
                    None
                    if selection.frame is None
                    else by_locator[selection.frame.source_locator_bytes]
                )
                frame_ref = None if frame is None else _source_frame_ref(frame)
                selected_artifact = None
                reused = None
                if selection.status is SelectionStatus.SELECTED:
                    assert frame is not None
                    resolved = artifact_resolver(camera_id, frame)
                    if resolved is None:
                        raise SupplementalPackageMaterializationError(
                            "selected supplemental frame has no immutable artifact"
                        )
                    selected_artifact = _selected_artifact(resolved)
                elif selection.status is SelectionStatus.DEDUPLICATED_FRAME:
                    assert selection.frame is not None
                    reused = selected_by_locator.get(selection.frame.source_locator_bytes)
                    if reused is None:
                        raise SupplementalPackageMaterializationError(
                            "deduplicated supplemental frame has no selected target"
                        )
                outcomes[target.ordinal] = SupplementalTargetOutcome(
                    target=target,
                    status=selection.status,
                    delta_to_target_ns=selection.delta_to_target_ns,
                    source_frame=frame_ref,
                    selected_artifact=selected_artifact,
                    reused_selected_target_ordinal=reused,
                )

        ordered = tuple(outcomes[ordinal] for ordinal in range(len(plan.targets)))
        values: dict[str, Any] = {
            "target_plan_id": plan.plan_id,
            "target_plan_semantic_sha256": plan.semantic_sha256,
            "source_content_sha256": plan.source_content_sha256,
            "camera_mapping_semantic_sha256": plan.camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": plan.alignment_semantic_sha256,
            "outcomes": ordered,
            "selected_artifact_count": sum(
                item.status is SelectionStatus.SELECTED for item in ordered
            ),
            "materializer_version": self._version,
            "projection_version": SUPPLEMENTAL_PACKAGE_PROJECTION_VERSION,
            "evidence_class": SupplementalEvidenceClass.LOCAL_CONFORMANCE,
            "production_eligible": False,
            "created_at": created_at,
        }
        draft = ProviderNeutralSupplementalPackage.model_construct(
            package_id="pending",
            semantic_content_sha256="0" * 64,
            schema_version=SUPPLEMENTAL_PACKAGE_SCHEMA_VERSION,
            **values,
        )
        digest = semantic_sha256(supplemental_package_projection(draft))
        package = ProviderNeutralSupplementalPackage.model_validate(
            {
                **values,
                "schema_version": SUPPLEMENTAL_PACKAGE_SCHEMA_VERSION,
                "package_id": _stable_id("supplemental-temporal-package-v2", digest),
                "semantic_content_sha256": digest,
            },
            strict=True,
        )
        manifest_bytes = canonical_json_bytes(package)
        return MaterializedSupplementalPackage(
            package=package,
            manifest_bytes=manifest_bytes,
            manifest_sha256=exact_bytes_sha256(manifest_bytes),
        )


def _validate_plan_index_binding(
    plan: FrozenSupplementalTargetPlan,
    frame_index: CanonicalSixCameraFrameIndex,
) -> None:
    if not isinstance(plan, FrozenSupplementalTargetPlan):
        raise TypeError("plan must be a FrozenSupplementalTargetPlan")
    if not isinstance(frame_index, CanonicalSixCameraFrameIndex):
        raise TypeError("frame_index must be a CanonicalSixCameraFrameIndex")
    if (
        frame_index.source_content_sha256 != plan.source_content_sha256
        or frame_index.camera_mapping_semantic_sha256 != plan.camera_mapping_semantic_sha256
        or frame_index.alignment_semantic_sha256 != plan.alignment_semantic_sha256
    ):
        raise SupplementalPackageMaterializationError(
            "supplemental target plan does not bind the frame index lineage"
        )
    if any(
        frame.alignment_projection.alignment_id != frame_index.alignment_id
        for camera in frame_index.cameras.values()
        for frame in camera.frames
    ):
        raise SupplementalPackageMaterializationError(
            "supplemental frame index contains a foreign alignment projection"
        )


def _source_frame_ref(frame: IndexedSourceFrame) -> SupplementalSourceFrameRef:
    return SupplementalSourceFrameRef(
        source_frame_id=frame.source_frame_id,
        aligned_timestamp_ns=frame.alignment_projection.aligned_timestamp_ns,
        source_timestamp_ns=frame.source_timestamp_ns,
        source_locator=frame.source_locator,
        alignment_projection_id=frame.alignment_projection.projection_id,
    )


def _selected_artifact(fact: MaterializedFrameArtifactFact) -> SupplementalSelectedArtifact:
    return SupplementalSelectedArtifact(
        artifact=fact.artifact,
        width=fact.width,
        height=fact.height,
        quality_flags=fact.quality_flags,
    )


def _stable_id(kind: str, digest: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{kind}:{digest}"))


__all__ = [
    "SUPPLEMENTAL_DEDUPE_POLICY_VERSION",
    "SUPPLEMENTAL_MATERIALIZER_VERSION",
    "SUPPLEMENTAL_PACKAGE_PROJECTION_VERSION",
    "SUPPLEMENTAL_TARGET_PLAN_PROJECTION_VERSION",
    "SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION",
    "ExplicitTargetPackageMaterializer",
    "FrozenSupplementalTarget",
    "FrozenSupplementalTargetPlan",
    "MaterializedSupplementalPackage",
    "ProviderNeutralSupplementalPackage",
    "SupplementalEvidenceClass",
    "SupplementalPackageMaterializationError",
    "SupplementalSelectedArtifact",
    "SupplementalSourceFrameRef",
    "SupplementalTargetOutcome",
    "build_frozen_supplemental_target_plan",
    "supplemental_package_projection",
    "supplemental_target_plan_projection",
]
