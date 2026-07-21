"""Offline, provider-neutral materialization of canonical temporal packages.

The materializer consumes already-indexed source frames and immutable artifact
facts. It never decodes media, invents pixels, or calls a model/provider.
Semantic identity excludes execution/publication row IDs, while the separate
manifest digest hashes the exact canonical manifest bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Literal, Never, Protocol, Self, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, field_validator, model_validator

from robata.admission.context import AdmittedRecordingContextV2
from robata.alignment.rational_time import RationalTransformSegment
from robata.contracts.admission_v2 import AlignmentManifestV2
from robata.contracts.alignment import AlignmentRun
from robata.contracts.artifacts import ArtifactUri, MediaType
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import Rfc3339Timestamp
from robata.contracts.pipeline import SamplingPurpose, SamplingStrategy
from robata.contracts.sampling_plan import SamplingPlan
from robata.contracts.temporal import CameraSamplingSummary, FrameSelectionManifest, PackageLineage
from robata.sampling.dense import IntervalPart, sampling_plan_projection
from robata.sampling.grid import (
    NANOSECONDS_PER_SECOND,
    FrameCandidate,
    SamplingGrid,
    SamplingRate,
    SelectionStatus,
)
from robata.sampling.package_set import MaterializedPackageRef, sampling_plan_digest

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
GridIndex = Annotated[int, Field(strict=True)]
LocatorScalar = str | int | bool
AlignmentEvidence = AlignmentRun | AlignmentManifestV2


class PackageMaterializationErrorCode(StrEnum):
    """Stable failure classes for the offline materialization boundary."""

    INVALID_INPUT = "INVALID_INPUT"
    ALIGNMENT_MISMATCH = "ALIGNMENT_MISMATCH"
    FRAME_BUDGET_EXCEEDED = "FRAME_BUDGET_EXCEEDED"
    MISSING_ARTIFACT = "MISSING_ARTIFACT"
    INVALID_ARTIFACT = "INVALID_ARTIFACT"
    EMPTY_PACKAGE = "EMPTY_PACKAGE"


class PackageMaterializationError(RuntimeError):
    """A fail-closed materialization error with a machine-readable code."""

    def __init__(self, code: PackageMaterializationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class FrameAlignmentProjectionFact(StrictModel):
    """Persisted alignment projection associated with one indexed source frame."""

    projection_id: NonEmptyString
    alignment_id: NonEmptyString
    segment_id: NonEmptyString
    aligned_timestamp_ns: Nanoseconds


class IndexedSourceFrame(StrictModel):
    """One source-frame fact in canonical source order."""

    source_frame_id: NonEmptyString
    source_order: NonNegativeInt
    source_timestamp_ns: Nanoseconds
    source_locator: dict[str, LocatorScalar]
    decodable: bool
    alignment_projection: FrameAlignmentProjectionFact

    @field_validator("source_locator")
    @classmethod
    def require_canonical_locator(
        cls,
        value: dict[str, LocatorScalar],
    ) -> dict[str, LocatorScalar]:
        if not value:
            raise ValueError("source_locator must not be empty")
        canonical_json_bytes(value)
        return dict(value)


class CameraSourceFrameIndex(StrictModel):
    """Canonical source-order index for one camera stream."""

    camera_id: CameraId
    stream_id: NonEmptyString
    stream_semantic_sha256: Sha256Digest
    frames: tuple[IndexedSourceFrame, ...]

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        orders = tuple(frame.source_order for frame in self.frames)
        if orders != tuple(sorted(orders)) or len(orders) != len(set(orders)):
            raise ValueError("camera frame index must use unique ascending source_order values")
        frame_ids = tuple(frame.source_frame_id for frame in self.frames)
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("camera frame index must not repeat source_frame_id")
        locators = tuple(canonical_json_bytes(frame.source_locator) for frame in self.frames)
        if len(locators) != len(set(locators)):
            raise ValueError("camera frame index must not repeat a canonical source locator")
        return self


class CanonicalSixCameraFrameIndex(StrictModel):
    """Exactly six aligned source-frame indexes bound to semantic lineage."""

    mcap_id: NonEmptyString
    camera_mapping_run_id: NonEmptyString
    alignment_id: NonEmptyString
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    cameras: SixCameraMap[CameraSourceFrameIndex]

    @model_validator(mode="after")
    def validate_camera_keys(self) -> Self:
        for camera_id, camera in self.cameras.items():
            if camera.camera_id is not camera_id:
                raise ValueError("camera index keys must match nested camera_id values")
        return self


class MaterializedArtifactManifest(StrictModel):
    """Immutable artifact identity embedded in a selected-frame manifest."""

    artifact_id: NonEmptyString
    uri: ArtifactUri
    sha256: Sha256Digest
    bytes: PositiveInt
    media_type: MediaType


class MaterializedFrameArtifactFact(StrictModel):
    """Artifact facts returned by the caller after real frame materialization."""

    artifact: MaterializedArtifactManifest
    width: PositiveInt
    height: PositiveInt
    quality_flags: tuple[NonEmptyString, ...] = ()


class FrameArtifactResolver(Protocol):
    """Resolve immutable artifact facts for a selected source frame."""

    def __call__(
        self,
        camera_id: CameraId,
        frame: IndexedSourceFrame,
    ) -> MaterializedFrameArtifactFact | None:
        """Return facts for real materialized bytes, or None when absent."""


class TemporalPackageMaterializationPolicy(StrictModel):
    """Versioned, provider-neutral frame-selection and materialization policy."""

    version: SchemaVersion
    grid_origin_ns: Nanoseconds = 0
    selection_tolerance_ns: Annotated[Nanoseconds, Field(ge=0)]
    tie_break_policy_version: SchemaVersion
    dedupe_policy_version: SchemaVersion
    producer_version: SchemaVersion
    extractor_version: SchemaVersion


class GridTargetMaterialization(StrictModel):
    """Auditable outcome for every unique rounded grid target."""

    index: GridIndex
    target_ns: Nanoseconds
    status: SelectionStatus
    actual_timestamp_ns: Nanoseconds | None = None
    source_timestamp_ns: Nanoseconds | None = None
    delta_to_target_ns: Nanoseconds | None = None
    source_frame_id: NonEmptyString | None = None
    alignment_projection_id: NonEmptyString | None = None
    source_locator: dict[str, LocatorScalar] | None = None
    selected_frame_ordinal: NonNegativeInt | None = None
    tie_break_policy_version: SchemaVersion
    dedupe_policy_version: SchemaVersion

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        source_values = (
            self.actual_timestamp_ns,
            self.source_timestamp_ns,
            self.delta_to_target_ns,
            self.source_frame_id,
            self.alignment_projection_id,
            self.source_locator,
        )
        if self.status is SelectionStatus.NO_FRAME_WITHIN_TOLERANCE:
            if any(value is not None for value in source_values):
                raise ValueError("a no-frame target cannot contain source-frame facts")
            if self.selected_frame_ordinal is not None:
                raise ValueError("a no-frame target cannot reference a selected frame")
            return self

        if any(value is None for value in source_values):
            raise ValueError("a source-associated target requires complete frame provenance")
        assert self.actual_timestamp_ns is not None
        assert self.delta_to_target_ns is not None
        if self.actual_timestamp_ns - self.target_ns != self.delta_to_target_ns:
            raise ValueError("target delta must equal actual timestamp minus target timestamp")
        if self.status is SelectionStatus.DECODE_FAILED:
            if self.selected_frame_ordinal is not None:
                raise ValueError("a decode-failed target cannot reference a selected frame")
        elif self.selected_frame_ordinal is None:
            raise ValueError("selected and deduplicated targets must resolve a selected frame")
        return self


class MaterializedCameraStatus(StrEnum):
    """Evidence availability for one camera in the materialized package."""

    AVAILABLE = "AVAILABLE"
    NO_FRAME = "NO_FRAME"
    CORRUPT = "CORRUPT"


class MaterializedTemporalPackageCamera(StrictModel):
    """One canonical camera entry with its complete target ledger."""

    camera_id: CameraId
    status: MaterializedCameraStatus
    stream_id: NonEmptyString
    stream_semantic_sha256: Sha256Digest
    rate_numerator: PositiveInt
    rate_denominator: PositiveInt
    grid_origin_ns: Nanoseconds
    selection_tolerance_ns: Annotated[Nanoseconds, Field(ge=0)]
    targets: tuple[GridTargetMaterialization, ...]
    frames: tuple[FrameSelectionManifest, ...]
    sampling: CameraSamplingSummary
    missing_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_camera_manifest(self) -> Self:
        coordinates = tuple((target.index, target.target_ns) for target in self.targets)
        if coordinates != tuple(sorted(coordinates)):
            raise ValueError("grid targets must be stored in ascending index/time order")
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("grid target coordinates must be unique")

        frame_ordinals = tuple(frame.ordinal for frame in self.frames)
        if frame_ordinals != tuple(range(len(self.frames))):
            raise ValueError("selected frame ordinals must be contiguous from zero")
        for frame in self.frames:
            if frame.materialized_artifact is None:
                raise ValueError("every selected frame requires a materialized artifact")
            MaterializedArtifactManifest.model_validate(frame.materialized_artifact)

        selected_targets = tuple(
            target for target in self.targets if target.status is SelectionStatus.SELECTED
        )
        if len(selected_targets) != len(self.frames):
            raise ValueError("selected target count must equal selected frame count")
        for target in self.targets:
            ordinal = target.selected_frame_ordinal
            if ordinal is None:
                continue
            if ordinal >= len(self.frames):
                raise ValueError("target selected_frame_ordinal is out of range")
            frame = self.frames[ordinal]
            if (
                frame.aligned_timestamp_ns != target.actual_timestamp_ns
                or frame.source_timestamp_ns != target.source_timestamp_ns
                or frame.source_locator != target.source_locator
                or frame.alignment_projection_id != target.alignment_projection_id
            ):
                raise ValueError("target provenance does not match its selected frame")
            if (
                target.status is SelectionStatus.SELECTED
                and frame.delta_to_target_ns != target.delta_to_target_ns
            ):
                raise ValueError("selected target delta does not match its selected frame")

        target_count = len(self.targets)
        actual_count = len(self.frames)
        if (
            self.sampling.target_count != target_count
            or self.sampling.actual_count != actual_count
            or self.sampling.missed_targets != target_count - actual_count
        ):
            raise ValueError("sampling counts must match target and selected-frame manifests")

        has_decode_failure = any(
            target.status is SelectionStatus.DECODE_FAILED for target in self.targets
        )
        expected_status = (
            MaterializedCameraStatus.CORRUPT
            if has_decode_failure
            else (
                MaterializedCameraStatus.AVAILABLE
                if self.frames
                else MaterializedCameraStatus.NO_FRAME
            )
        )
        if self.status is not expected_status:
            raise ValueError("camera status does not match target outcomes")
        if self.status is MaterializedCameraStatus.AVAILABLE:
            if self.missing_reason is not None:
                raise ValueError("AVAILABLE camera cannot carry a missing reason")
        elif self.missing_reason is None:
            raise ValueError("non-AVAILABLE camera requires a missing reason")
        return self


class MaterializedIntervalPart(StrictModel):
    """Serializable package-member coordinates copied from an IntervalPart."""

    requested_interval: NanosecondInterval
    effective_interval: NanosecondInterval
    ordinal: NonNegativeInt
    part_count: PositiveInt
    overlap_before_ns: Annotated[Nanoseconds, Field(ge=0)]
    overlap_after_ns: Annotated[Nanoseconds, Field(ge=0)]

    @model_validator(mode="after")
    def validate_part(self) -> Self:
        if self.effective_interval.start_ns < self.requested_interval.start_ns:
            raise ValueError("effective interval must be contained by requested interval")
        if self.effective_interval.end_ns > self.requested_interval.end_ns:
            raise ValueError("effective interval must be contained by requested interval")
        if self.ordinal >= self.part_count:
            raise ValueError("part ordinal must be less than part_count")
        duration = self.effective_interval.duration_ns
        if self.overlap_before_ns >= duration or self.overlap_after_ns >= duration:
            raise ValueError("part overlap must be less than its effective duration")
        if self.part_count == 1 and (self.overlap_before_ns or self.overlap_after_ns):
            raise ValueError("an unsplit part cannot carry overlap")
        return self


class ProviderNeutralTemporalPackage(StrictModel):
    """Immutable six-camera manifest produced before provider input planning."""

    schema_version: Literal["1.0"]
    package_id: NonEmptyString
    semantic_content_sha256: Sha256Digest
    mcap_id: NonEmptyString
    window_id: NonEmptyString
    camera_mapping_run_id: NonEmptyString
    alignment_id: NonEmptyString
    lineage: PackageLineage
    part: MaterializedIntervalPart
    sampling_plan_id: NonEmptyString
    sampling_plan_version: SchemaVersion
    sampling_plan_sha256: Sha256Digest
    cameras: SixCameraMap[MaterializedTemporalPackageCamera]
    frame_count_total: PositiveInt
    materialization_policy_version: SchemaVersion
    producer_version: SchemaVersion
    extractor_version: SchemaVersion
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_package(self) -> Self:
        if self.sampling_plan_sha256 != self.lineage.sampling_plan_sha256:
            raise ValueError("sampling plan digest must match package lineage")
        actual_count = 0
        for camera_id, camera in self.cameras.items():
            if camera.camera_id is not camera_id:
                raise ValueError("camera manifest keys must match nested camera_id values")
            actual_count += len(camera.frames)
            for target in camera.targets:
                if not self.part.effective_interval.contains(target.target_ns):
                    raise ValueError("every grid target must lie in the effective interval")
            for frame in camera.frames:
                if not self.part.effective_interval.contains(frame.aligned_timestamp_ns):
                    raise ValueError("every selected frame must lie in the effective interval")
        if actual_count != self.frame_count_total:
            raise ValueError("frame_count_total must equal all selected frame counts")
        expected_digest = semantic_sha256(package_semantic_projection(self))
        if self.semantic_content_sha256 != expected_digest:
            raise ValueError("semantic_content_sha256 does not match package content")
        if self.package_id != derive_temporal_package_id(expected_digest):
            raise ValueError("package_id does not match semantic package content")
        return self


@dataclass(frozen=True, slots=True)
class MaterializedTemporalPackage:
    """Package manifest, exact bytes/digest, and package-set reference."""

    package: ProviderNeutralTemporalPackage
    manifest_bytes: bytes
    package_manifest_sha256: Sha256Digest

    def __post_init__(self) -> None:
        if self.manifest_bytes != canonical_json_bytes(self.package):
            raise ValueError("manifest_bytes must be the canonical package manifest bytes")
        if exact_bytes_sha256(self.manifest_bytes) != self.package_manifest_sha256:
            raise ValueError("package_manifest_sha256 must hash the exact manifest bytes")

    @property
    def package_ref(self) -> MaterializedPackageRef:
        """Return the exact reference accepted by PackageSetBuilder."""

        return MaterializedPackageRef(
            ordinal=self.package.part.ordinal,
            package_id=self.package.package_id,
            package_semantic_content_sha256=self.package.semantic_content_sha256,
            package_manifest_sha256=self.package_manifest_sha256,
        )


def derive_temporal_package_id(semantic_content_sha256: str) -> str:
    """Derive a run-independent package ID from semantic content."""

    return str(
        uuid5(
            NAMESPACE_URL,
            f"robata:provider-neutral-temporal-package:v1:{semantic_content_sha256}",
        )
    )


def _artifact_semantic_projection(frame: FrameSelectionManifest) -> dict[str, object]:
    assert frame.materialized_artifact is not None
    artifact = MaterializedArtifactManifest.model_validate(frame.materialized_artifact)
    return {
        "sha256": artifact.sha256,
        "bytes": artifact.bytes,
        "media_type": artifact.media_type,
        "width": frame.width,
        "height": frame.height,
        "quality_flags": frame.quality_flags,
    }


def _target_semantic_projection(
    target: GridTargetMaterialization,
    frames: tuple[FrameSelectionManifest, ...],
) -> dict[str, object]:
    artifact: dict[str, object] | None = None
    if target.status is SelectionStatus.SELECTED:
        assert target.selected_frame_ordinal is not None
        artifact = _artifact_semantic_projection(frames[target.selected_frame_ordinal])
    return {
        "index": target.index,
        "target_ns": str(target.target_ns),
        "status": target.status.value,
        "actual_timestamp_ns": (
            None if target.actual_timestamp_ns is None else str(target.actual_timestamp_ns)
        ),
        "source_timestamp_ns": (
            None if target.source_timestamp_ns is None else str(target.source_timestamp_ns)
        ),
        "delta_to_target_ns": (
            None if target.delta_to_target_ns is None else str(target.delta_to_target_ns)
        ),
        "source_locator": target.source_locator,
        "tie_break_policy_version": target.tie_break_policy_version,
        "dedupe_policy_version": target.dedupe_policy_version,
        "materialized_artifact": artifact,
    }


def package_semantic_projection(package: ProviderNeutralTemporalPackage) -> dict[str, object]:
    """Project package semantics while excluding row IDs, URIs, and wall-clock time."""

    return {
        "schema_version": package.schema_version,
        "lineage": package.lineage,
        "part": package.part,
        "sampling_plan_sha256": package.sampling_plan_sha256,
        "sampling_plan_version": package.sampling_plan_version,
        "materialization_policy_version": package.materialization_policy_version,
        "producer_version": package.producer_version,
        "extractor_version": package.extractor_version,
        "cameras": [
            {
                "camera_id": camera_id.value,
                "status": camera.status.value,
                "stream_semantic_sha256": camera.stream_semantic_sha256,
                "rate": {
                    "numerator": camera.rate_numerator,
                    "denominator": camera.rate_denominator,
                },
                "grid_origin_ns": str(camera.grid_origin_ns),
                "selection_tolerance_ns": str(camera.selection_tolerance_ns),
                "targets": [
                    _target_semantic_projection(target, camera.frames) for target in camera.targets
                ],
            }
            for camera_id, camera in package.cameras.items()
        ],
    }


class OfflineTemporalPackageMaterializer:
    """Select indexed frames and bind caller-supplied immutable artifact facts."""

    def __init__(self, policy: TemporalPackageMaterializationPolicy) -> None:
        if not isinstance(policy, TemporalPackageMaterializationPolicy):
            raise TypeError("policy must be a TemporalPackageMaterializationPolicy")
        self._policy = policy

    @property
    def policy(self) -> TemporalPackageMaterializationPolicy:
        return self._policy

    def materialize(
        self,
        *,
        part: IntervalPart,
        sampling_plan: SamplingPlan,
        purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
        alignment_run: AlignmentEvidence,
        frame_index: CanonicalSixCameraFrameIndex,
        lineage: PackageLineage,
        window_id: str,
        artifact_resolver: FrameArtifactResolver
        | Callable[[CameraId, IndexedSourceFrame], MaterializedFrameArtifactFact | None],
        created_at: str,
    ) -> MaterializedTemporalPackage:
        """Build one complete provider-neutral package for a planned interval part."""

        if not isinstance(part, IntervalPart):
            raise TypeError("part must be an IntervalPart")
        if not isinstance(sampling_plan, SamplingPlan):
            raise TypeError("sampling_plan must be a SamplingPlan")
        if not isinstance(alignment_run, (AlignmentRun, AlignmentManifestV2)):
            raise TypeError("alignment_run must be registered alignment evidence")
        if not isinstance(frame_index, CanonicalSixCameraFrameIndex):
            raise TypeError("frame_index must be a CanonicalSixCameraFrameIndex")
        if not isinstance(lineage, PackageLineage):
            raise TypeError("lineage must be a PackageLineage")
        if not isinstance(window_id, str) or not window_id:
            raise ValueError("window_id must be a nonempty string")
        if not callable(artifact_resolver):
            raise TypeError("artifact_resolver must be callable")

        part_manifest = _part_manifest(part)
        _validate_bindings(
            sampling_plan=sampling_plan,
            purpose=purpose,
            alignment_run=alignment_run,
            frame_index=frame_index,
            lineage=lineage,
        )
        plan_projection = sampling_plan_projection(sampling_plan, purpose=purpose)
        per_camera = cast(dict[str, dict[str, int]], plan_projection["per_camera"])
        strategy = _materialization_strategy(purpose)

        cameras: dict[CameraId, MaterializedTemporalPackageCamera] = {}
        for camera_id in CAMERA_IDS:
            rate = per_camera[camera_id.value]
            cameras[camera_id] = self._materialize_camera(
                camera_id=camera_id,
                source_index=frame_index.cameras[camera_id],
                alignment_run=alignment_run,
                alignment_semantic_sha256=lineage.alignment_semantic_sha256,
                interval=part_manifest.effective_interval,
                rate=SamplingRate(rate["numerator"], rate["denominator"]),
                strategy=strategy,
                artifact_resolver=artifact_resolver,
            )

        target_counts = tuple(len(cameras[camera_id].targets) for camera_id in CAMERA_IDS)
        budget = sampling_plan.frame_budget
        if any(count > budget.max_frames_per_camera for count in target_counts):
            _fail(
                PackageMaterializationErrorCode.FRAME_BUDGET_EXCEEDED,
                "materialized part exceeds the per-camera frame budget",
            )
        if sum(target_counts) > budget.max_frames_total:
            _fail(
                PackageMaterializationErrorCode.FRAME_BUDGET_EXCEEDED,
                "materialized part exceeds the total frame budget",
            )

        frame_count_total = sum(len(camera.frames) for camera in cameras.values())
        if frame_count_total == 0:
            _fail(
                PackageMaterializationErrorCode.EMPTY_PACKAGE,
                "materialization selected no frames across all six cameras",
            )

        camera_map = SixCameraMap[MaterializedTemporalPackageCamera](cameras)
        provisional = ProviderNeutralTemporalPackage.model_construct(
            schema_version="1.0",
            package_id="pending",
            semantic_content_sha256="0" * 64,
            mcap_id=alignment_run.mcap_id,
            window_id=window_id,
            camera_mapping_run_id=alignment_run.camera_mapping_run_id,
            alignment_id=alignment_run.alignment_id,
            lineage=lineage,
            part=part_manifest,
            sampling_plan_id=sampling_plan.sampling_plan_id,
            sampling_plan_version=sampling_plan.version,
            sampling_plan_sha256=lineage.sampling_plan_sha256,
            cameras=camera_map,
            frame_count_total=frame_count_total,
            materialization_policy_version=self._policy.version,
            producer_version=self._policy.producer_version,
            extractor_version=self._policy.extractor_version,
            created_at=created_at,
        )
        semantic_digest = semantic_sha256(package_semantic_projection(provisional))
        package = ProviderNeutralTemporalPackage(
            schema_version="1.0",
            package_id=derive_temporal_package_id(semantic_digest),
            semantic_content_sha256=semantic_digest,
            mcap_id=alignment_run.mcap_id,
            window_id=window_id,
            camera_mapping_run_id=alignment_run.camera_mapping_run_id,
            alignment_id=alignment_run.alignment_id,
            lineage=lineage,
            part=part_manifest,
            sampling_plan_id=sampling_plan.sampling_plan_id,
            sampling_plan_version=sampling_plan.version,
            sampling_plan_sha256=lineage.sampling_plan_sha256,
            cameras=camera_map,
            frame_count_total=frame_count_total,
            materialization_policy_version=self._policy.version,
            producer_version=self._policy.producer_version,
            extractor_version=self._policy.extractor_version,
            created_at=created_at,
        )
        manifest_bytes = canonical_json_bytes(package)
        return MaterializedTemporalPackage(
            package=package,
            manifest_bytes=manifest_bytes,
            package_manifest_sha256=exact_bytes_sha256(manifest_bytes),
        )

    def materialize_admitted(
        self,
        *,
        part: IntervalPart,
        sampling_plan: SamplingPlan,
        purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
        admitted_context: AdmittedRecordingContextV2,
        frame_index: CanonicalSixCameraFrameIndex,
        lineage: PackageLineage,
        window_id: str,
        artifact_resolver: FrameArtifactResolver
        | Callable[[CameraId, IndexedSourceFrame], MaterializedFrameArtifactFact | None],
        created_at: str,
    ) -> MaterializedTemporalPackage:
        """Materialize only from selected registered V2 admission evidence."""

        if not isinstance(admitted_context, AdmittedRecordingContextV2):
            raise TypeError("admitted_context must be an AdmittedRecordingContextV2")
        try:
            context = AdmittedRecordingContextV2.model_validate(
                admitted_context.model_dump(mode="python"), strict=True
            )
        except ValueError as exc:
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                f"V2 admission context failed validation: {exc}",
            )
        _validate_admitted_bindings(
            context=context,
            frame_index=frame_index,
            lineage=lineage,
        )
        return self.materialize(
            part=part,
            sampling_plan=sampling_plan,
            purpose=purpose,
            alignment_run=context.alignment_manifest,
            frame_index=frame_index,
            lineage=lineage,
            window_id=window_id,
            artifact_resolver=artifact_resolver,
            created_at=created_at,
        )

    def _materialize_camera(
        self,
        *,
        camera_id: CameraId,
        source_index: CameraSourceFrameIndex,
        alignment_run: AlignmentEvidence,
        alignment_semantic_sha256: str,
        interval: NanosecondInterval,
        rate: SamplingRate,
        strategy: SamplingStrategy,
        artifact_resolver: FrameArtifactResolver
        | Callable[[CameraId, IndexedSourceFrame], MaterializedFrameArtifactFact | None],
    ) -> MaterializedTemporalPackageCamera:
        frames_by_locator = _validate_and_index_projections(
            camera_id=camera_id,
            source_index=source_index,
            alignment_run=alignment_run,
        )
        candidates = tuple(
            FrameCandidate(
                aligned_timestamp_ns=frame.alignment_projection.aligned_timestamp_ns,
                source_timestamp_ns=frame.source_timestamp_ns,
                source_locator_bytes=locator,
                decodable=frame.decodable,
            )
            for locator, frame in frames_by_locator.items()
        )
        grid = SamplingGrid(grid_origin_ns=self._policy.grid_origin_ns, rate=rate)
        selections = grid.select_frames(
            candidates,
            interval.start_ns,
            interval.end_ns,
            self._policy.selection_tolerance_ns,
        )

        selected_frames: list[FrameSelectionManifest] = []
        selected_ordinal_by_locator: dict[bytes, int] = {}
        for selection in selections:
            if selection.status is not SelectionStatus.SELECTED:
                continue
            assert selection.frame is not None
            assert selection.delta_to_target_ns is not None
            selected_source_frame = frames_by_locator[selection.frame.source_locator_bytes]
            artifact = _resolve_artifact(
                artifact_resolver,
                camera_id,
                selected_source_frame,
            )
            ordinal = len(selected_frames)
            selected_ordinal_by_locator[selection.frame.source_locator_bytes] = ordinal
            selected_frames.append(
                FrameSelectionManifest(
                    frame_id=_derive_frame_selection_id(
                        camera_id=camera_id,
                        alignment_semantic_sha256=alignment_semantic_sha256,
                        target_index=selection.k,
                        target_ns=selection.target_ns,
                        source_frame=selected_source_frame,
                        artifact=artifact,
                    ),
                    alignment_projection_id=(
                        selected_source_frame.alignment_projection.projection_id
                    ),
                    ordinal=ordinal,
                    aligned_timestamp_ns=selection.frame.aligned_timestamp_ns,
                    source_timestamp_ns=selection.frame.source_timestamp_ns,
                    delta_to_target_ns=selection.delta_to_target_ns,
                    source_locator=dict(selected_source_frame.source_locator),
                    materialized_artifact=artifact.artifact.model_dump(mode="json"),
                    width=artifact.width,
                    height=artifact.height,
                    quality_flags=artifact.quality_flags,
                )
            )

        target_results: list[GridTargetMaterialization] = []
        for selection in selections:
            target_source_frame: IndexedSourceFrame | None = None
            selected_ordinal: int | None = None
            if selection.frame is not None:
                target_source_frame = frames_by_locator[selection.frame.source_locator_bytes]
                selected_ordinal = selected_ordinal_by_locator.get(
                    selection.frame.source_locator_bytes
                )
            target_results.append(
                GridTargetMaterialization(
                    index=selection.k,
                    target_ns=selection.target_ns,
                    status=selection.status,
                    actual_timestamp_ns=(
                        None if selection.frame is None else selection.frame.aligned_timestamp_ns
                    ),
                    source_timestamp_ns=(
                        None if selection.frame is None else selection.frame.source_timestamp_ns
                    ),
                    delta_to_target_ns=selection.delta_to_target_ns,
                    source_frame_id=(
                        None if target_source_frame is None else target_source_frame.source_frame_id
                    ),
                    alignment_projection_id=(
                        None
                        if target_source_frame is None
                        else target_source_frame.alignment_projection.projection_id
                    ),
                    source_locator=(
                        None
                        if target_source_frame is None
                        else dict(target_source_frame.source_locator)
                    ),
                    selected_frame_ordinal=(
                        None
                        if selection.status is SelectionStatus.DECODE_FAILED
                        else selected_ordinal
                    ),
                    tie_break_policy_version=self._policy.tie_break_policy_version,
                    dedupe_policy_version=self._policy.dedupe_policy_version,
                )
            )

        has_decode_failure = any(
            target.status is SelectionStatus.DECODE_FAILED for target in target_results
        )
        status = (
            MaterializedCameraStatus.CORRUPT
            if has_decode_failure
            else (
                MaterializedCameraStatus.AVAILABLE
                if selected_frames
                else MaterializedCameraStatus.NO_FRAME
            )
        )
        missing_reason = None
        if status is MaterializedCameraStatus.CORRUPT:
            missing_reason = "DECODE_FAILED"
        elif status is MaterializedCameraStatus.NO_FRAME:
            missing_reason = (
                "NO_FRAME_WITHIN_TOLERANCE" if target_results else "NO_GRID_TARGET_IN_INTERVAL"
            )

        actual_count = len(selected_frames)
        actual_fps = (
            0.0
            if actual_count == 0
            else float(Fraction(actual_count * NANOSECONDS_PER_SECOND, interval.duration_ns))
        )
        return MaterializedTemporalPackageCamera(
            camera_id=camera_id,
            status=status,
            stream_id=source_index.stream_id,
            stream_semantic_sha256=source_index.stream_semantic_sha256,
            rate_numerator=rate.numerator,
            rate_denominator=rate.denominator,
            grid_origin_ns=self._policy.grid_origin_ns,
            selection_tolerance_ns=self._policy.selection_tolerance_ns,
            targets=tuple(target_results),
            frames=tuple(selected_frames),
            sampling=CameraSamplingSummary(
                strategy=strategy.value,
                target_fps=float(Fraction(rate.numerator, rate.denominator)),
                actual_fps=actual_fps,
                target_count=len(target_results),
                actual_count=actual_count,
                missed_targets=len(target_results) - actual_count,
            ),
            missing_reason=missing_reason,
        )


def _materialization_strategy(purpose: SamplingPurpose) -> SamplingStrategy:
    if purpose is SamplingPurpose.QA_COARSE:
        return SamplingStrategy.UNIFORM
    if purpose in {
        SamplingPurpose.QA_DENSE,
        SamplingPurpose.ACTION_DENSE,
        SamplingPurpose.BOUNDARY_REFINEMENT,
    }:
        return SamplingStrategy.DENSE
    raise ValueError(
        "provider-neutral materialization currently supports only QA_COARSE, "
        "QA_DENSE, ACTION_DENSE, and BOUNDARY_REFINEMENT"
    )


def _fail(code: PackageMaterializationErrorCode, message: str) -> Never:
    raise PackageMaterializationError(code, message)


def _part_manifest(part: IntervalPart) -> MaterializedIntervalPart:
    try:
        return MaterializedIntervalPart(
            requested_interval=part.requested_interval,
            effective_interval=part.effective_interval,
            ordinal=part.ordinal,
            part_count=part.part_count,
            overlap_before_ns=part.overlap_before_ns,
            overlap_after_ns=part.overlap_after_ns,
        )
    except ValueError as exc:
        _fail(PackageMaterializationErrorCode.INVALID_INPUT, str(exc))


def _validate_bindings(
    *,
    sampling_plan: SamplingPlan,
    purpose: SamplingPurpose,
    alignment_run: AlignmentEvidence,
    frame_index: CanonicalSixCameraFrameIndex,
    lineage: PackageLineage,
) -> None:
    if frame_index.mcap_id != alignment_run.mcap_id:
        _fail(PackageMaterializationErrorCode.ALIGNMENT_MISMATCH, "MCAP binding differs")
    if frame_index.camera_mapping_run_id != alignment_run.camera_mapping_run_id:
        _fail(PackageMaterializationErrorCode.ALIGNMENT_MISMATCH, "mapping binding differs")
    if frame_index.alignment_id != alignment_run.alignment_id:
        _fail(PackageMaterializationErrorCode.ALIGNMENT_MISMATCH, "alignment binding differs")
    if frame_index.source_content_sha256 != lineage.source_content_sha256:
        _fail(PackageMaterializationErrorCode.ALIGNMENT_MISMATCH, "source digest differs")
    if frame_index.camera_mapping_semantic_sha256 != lineage.camera_mapping_semantic_sha256:
        _fail(PackageMaterializationErrorCode.ALIGNMENT_MISMATCH, "mapping digest differs")
    if frame_index.alignment_semantic_sha256 != lineage.alignment_semantic_sha256:
        _fail(PackageMaterializationErrorCode.ALIGNMENT_MISMATCH, "alignment digest differs")
    if isinstance(alignment_run, AlignmentManifestV2):
        if alignment_run.source_content_sha256 != frame_index.source_content_sha256:
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                "V2 alignment source digest differs",
            )
        if (
            alignment_run.camera_mapping_semantic_sha256
            != frame_index.camera_mapping_semantic_sha256
        ):
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                "V2 alignment mapping digest differs",
            )
        if alignment_run.alignment_semantic_sha256 != frame_index.alignment_semantic_sha256:
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                "V2 alignment semantic digest differs",
            )
    if sampling_plan_digest(sampling_plan, purpose=purpose) != lineage.sampling_plan_sha256:
        _fail(PackageMaterializationErrorCode.INVALID_INPUT, "sampling plan digest differs")


def _validate_admitted_bindings(
    *,
    context: AdmittedRecordingContextV2,
    frame_index: CanonicalSixCameraFrameIndex,
    lineage: PackageLineage,
) -> None:
    manifest = context.ready_manifest
    alignment = context.alignment_manifest
    expected = (
        (frame_index.mcap_id, manifest.mcap_id, "MCAP ID"),
        (
            frame_index.camera_mapping_run_id,
            manifest.camera_mapping_run_id,
            "camera mapping run ID",
        ),
        (frame_index.alignment_id, alignment.alignment_id, "alignment ID"),
        (
            frame_index.source_content_sha256,
            context.source_content_sha256,
            "source content digest",
        ),
        (
            frame_index.camera_mapping_semantic_sha256,
            context.camera_mapping_semantic_sha256,
            "camera mapping semantic digest",
        ),
        (
            frame_index.alignment_semantic_sha256,
            context.alignment_semantic_sha256,
            "alignment semantic digest",
        ),
        (
            lineage.source_content_sha256,
            context.source_content_sha256,
            "lineage source digest",
        ),
        (
            lineage.camera_mapping_semantic_sha256,
            context.camera_mapping_semantic_sha256,
            "lineage mapping digest",
        ),
        (
            lineage.alignment_semantic_sha256,
            context.alignment_semantic_sha256,
            "lineage alignment digest",
        ),
    )
    for actual, selected, label in expected:
        if actual != selected:
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                f"admitted {label} differs",
            )

    ready_by_camera = {item.camera_id: item for item in manifest.cameras}
    for camera_id in CAMERA_IDS:
        source_index = frame_index.cameras[camera_id]
        ready_camera = ready_by_camera[camera_id]
        aligned_camera = alignment.cameras[camera_id.value]
        if not (source_index.stream_id == ready_camera.stream_id == aligned_camera.stream_id):
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                f"{camera_id.value} stream ID differs across admitted evidence",
            )
        if not (
            source_index.stream_semantic_sha256
            == ready_camera.stream_semantic_sha256
            == aligned_camera.stream_semantic_sha256
        ):
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                f"{camera_id.value} stream semantic digest differs",
            )


def _validate_and_index_projections(
    *,
    camera_id: CameraId,
    source_index: CameraSourceFrameIndex,
    alignment_run: AlignmentEvidence,
) -> dict[bytes, IndexedSourceFrame]:
    camera_alignment = alignment_run.cameras[camera_id.value]
    if isinstance(alignment_run, AlignmentManifestV2):
        v2_camera_alignment = alignment_run.cameras[camera_id.value]
        if (
            source_index.stream_id != v2_camera_alignment.stream_id
            or source_index.stream_semantic_sha256 != v2_camera_alignment.stream_semantic_sha256
        ):
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                f"{camera_id.value} frame index does not bind the V2 alignment stream",
            )
    segments = {segment.segment_id: segment for segment in camera_alignment.segments}
    indexed: dict[bytes, IndexedSourceFrame] = {}
    for frame in source_index.frames:
        projection = frame.alignment_projection
        if projection.alignment_id != alignment_run.alignment_id:
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                f"{camera_id.value} frame projection references another alignment",
            )
        segment_contract = segments.get(projection.segment_id)
        if segment_contract is None:
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                f"{camera_id.value} frame projection references an unknown segment",
            )
        segment = RationalTransformSegment(
            source_order_start=segment_contract.source_order_start,
            source_order_end=segment_contract.source_order_end,
            source_start_ns=segment_contract.source_start_ns,
            source_end_ns=segment_contract.source_end_ns,
            source_anchor_ns=segment_contract.source_anchor_ns,
            canonical_anchor_ns=segment_contract.canonical_anchor_ns,
            rate_numerator=int(segment_contract.rate_numerator),
            rate_denominator=int(segment_contract.rate_denominator),
            source_epoch_id=segment_contract.source_epoch_id,
            segment_id=segment_contract.segment_id,
        )
        try:
            expected = segment.apply(
                frame.source_timestamp_ns,
                source_order=frame.source_order,
            )
        except (ValueError, OverflowError) as exc:
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                f"{camera_id.value} frame is outside its alignment segment: {exc}",
            )
        if expected != projection.aligned_timestamp_ns:
            _fail(
                PackageMaterializationErrorCode.ALIGNMENT_MISMATCH,
                f"{camera_id.value} aligned timestamp differs from the rational transform",
            )
        indexed[canonical_json_bytes(frame.source_locator)] = frame
    return indexed


def _resolve_artifact(
    resolver: FrameArtifactResolver
    | Callable[[CameraId, IndexedSourceFrame], MaterializedFrameArtifactFact | None],
    camera_id: CameraId,
    frame: IndexedSourceFrame,
) -> MaterializedFrameArtifactFact:
    try:
        artifact = resolver(camera_id, frame)
    except Exception as exc:
        raise PackageMaterializationError(
            PackageMaterializationErrorCode.MISSING_ARTIFACT,
            f"artifact resolution failed for {camera_id.value}/{frame.source_frame_id}: {exc}",
        ) from exc
    if artifact is None:
        label = f"{camera_id.value}/{frame.source_frame_id}"
        _fail(
            PackageMaterializationErrorCode.MISSING_ARTIFACT,
            f"selected frame lacks materialized artifact: {label}",
        )
    if not isinstance(artifact, MaterializedFrameArtifactFact):
        _fail(
            PackageMaterializationErrorCode.INVALID_ARTIFACT,
            "artifact resolver must return MaterializedFrameArtifactFact or None",
        )
    return artifact


def _derive_frame_selection_id(
    *,
    camera_id: CameraId,
    alignment_semantic_sha256: str,
    target_index: int,
    target_ns: int,
    source_frame: IndexedSourceFrame,
    artifact: MaterializedFrameArtifactFact,
) -> str:
    digest = semantic_sha256(
        {
            "camera_id": camera_id.value,
            "alignment_semantic_sha256": alignment_semantic_sha256,
            "target_index": target_index,
            "target_ns": str(target_ns),
            "source_timestamp_ns": str(source_frame.source_timestamp_ns),
            "aligned_timestamp_ns": str(source_frame.alignment_projection.aligned_timestamp_ns),
            "source_locator": source_frame.source_locator,
            "artifact": {
                "sha256": artifact.artifact.sha256,
                "bytes": artifact.artifact.bytes,
                "media_type": artifact.artifact.media_type,
                "width": artifact.width,
                "height": artifact.height,
                "quality_flags": artifact.quality_flags,
            },
        }
    )
    return str(uuid5(NAMESPACE_URL, f"robata:frame-selection:v1:{digest}"))


__all__ = [
    "CameraSourceFrameIndex",
    "CanonicalSixCameraFrameIndex",
    "FrameAlignmentProjectionFact",
    "FrameArtifactResolver",
    "GridTargetMaterialization",
    "IndexedSourceFrame",
    "MaterializedArtifactManifest",
    "MaterializedCameraStatus",
    "MaterializedFrameArtifactFact",
    "MaterializedIntervalPart",
    "MaterializedTemporalPackage",
    "MaterializedTemporalPackageCamera",
    "OfflineTemporalPackageMaterializer",
    "PackageMaterializationError",
    "PackageMaterializationErrorCode",
    "ProviderNeutralTemporalPackage",
    "TemporalPackageMaterializationPolicy",
    "derive_temporal_package_id",
    "package_semantic_projection",
]
