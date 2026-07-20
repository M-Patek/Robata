"""Strict provider-specific input planning over provider-neutral packages.

The contracts implement the Architecture V1.1 Section 25.2 boundary. A
request catalog binds provider-local order to authoritative rows, while its
semantic digest and the input-plan digest exclude row UUIDs, locators, and
wall-clock fields. This first planner slice is one source frame to one provider
item; tiling or grouping requires a later lossless provenance contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.artifacts import ArtifactUri, MediaType
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.inference.models import VisionTask

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
TransformValue = str | int | bool
type FrozenTuple[T] = Annotated[tuple[T, ...], Field(strict=False)]

INFERENCE_INPUT_PLANNER_VERSION: Final = "inference-input-planner-v2"
REQUEST_CATALOG_SEMANTIC_PROJECTION_VERSION: Final = "request-catalog-semantic-v2"
INPUT_PLAN_SEMANTIC_PROJECTION_VERSION: Final = "inference-input-plan-semantic-v2"
CALL_PLAN_SEMANTIC_PROJECTION_VERSION: Final = "inference-call-plan-semantic-v2"
CALL_PART_SEMANTIC_PROJECTION_VERSION: Final = "inference-call-part-semantic-v2"
CALL_BARRIER_SEMANTIC_PROJECTION_VERSION: Final = "inference-call-barrier-semantic-v2"
CALL_IDEMPOTENCY_KEY_POLICY_VERSION: Final = "inference-call-idempotency-key-v2"

REQUEST_CATALOG_UUID_NAMESPACE: Final = "provider-request-catalog-v2"
INPUT_PLAN_UUID_NAMESPACE: Final = "inference-input-plan-v2"
INPUT_PLAN_LOGICAL_KEY_NAMESPACE: Final = "inference-input-plan-v2"
CALL_PART_LOGICAL_KEY_NAMESPACE: Final = "inference-input-call-part-v2"
CALL_BARRIER_LOGICAL_KEY_NAMESPACE: Final = "inference-input-barrier-v2"
CALL_IDEMPOTENCY_KEY_NAMESPACE: Final = "inference-input-call-v2"


class TransformOperation(StrEnum):
    """Supported one-frame-to-one-item transform classes."""

    NONE = "NONE"
    RESIZE = "RESIZE"
    RESIZE_TRANSCODE = "RESIZE_TRANSCODE"
    TRANSCODE = "TRANSCODE"


class LimitMetric(StrEnum):
    """Provider request limits measured by this planner."""

    IMAGES_PER_REQUEST = "IMAGES_PER_REQUEST"
    INPUT_TOKENS_PER_REQUEST = "INPUT_TOKENS_PER_REQUEST"
    PAYLOAD_BYTES_PER_REQUEST = "PAYLOAD_BYTES_PER_REQUEST"
    PIXELS_PER_IMAGE = "PIXELS_PER_IMAGE"


class LimitDecisionStatus(StrEnum):
    """Explicit result of comparing a measurement with a provider limit."""

    FAIL = "FAIL"
    PASS = "PASS"


class InputPlanError(ValueError):
    """Base error for deterministic input planning failures."""


class InputPlanLimitError(InputPlanError):
    """Raised when an explicit call plan still exceeds a provider limit."""

    def __init__(self, decisions: tuple[ProviderLimitDecision, ...]) -> None:
        self.decisions = decisions
        failed = ", ".join(
            decision.metric.value
            for decision in decisions
            if decision.status is LimitDecisionStatus.FAIL
        )
        super().__init__(f"explicit call plan exceeds provider limits: {failed}")


class CatalogFrame(StrictModel):
    """One immutable source frame in provider-local request-catalog order."""

    frame_id: OpaqueUuid
    ordinal: NonNegativeInt
    aligned_timestamp_ns: Nanoseconds
    source_timestamp_ns: Nanoseconds
    source_artifact_uri: ArtifactUri
    source_artifact_sha256: Sha256Digest
    source_artifact_bytes: PositiveInt
    media_type: MediaType
    encoding: NonEmptyString
    width: PositiveInt
    height: PositiveInt


class CatalogCamera(StrictModel):
    """One canonical camera entry, including explicit empty views."""

    camera_id: CameraId
    ordinal: NonNegativeInt
    frames: FrozenTuple[CatalogFrame]

    @model_validator(mode="after")
    def validate_frames(self) -> Self:
        if self.ordinal >= len(CAMERA_IDS) or CAMERA_IDS[self.ordinal] is not self.camera_id:
            raise ValueError("camera ordinal must match canonical camera order")
        ordinals = tuple(frame.ordinal for frame in self.frames)
        if ordinals != tuple(range(len(self.frames))):
            raise ValueError("catalog frame ordinals must be contiguous from zero")
        timestamps = tuple(frame.aligned_timestamp_ns for frame in self.frames)
        if timestamps != tuple(sorted(timestamps)):
            raise ValueError("catalog frames must preserve nondecreasing aligned time order")
        if len({frame.frame_id for frame in self.frames}) != len(self.frames):
            raise ValueError("catalog frame IDs must be unique within a camera")
        return self


class CatalogPackage(StrictModel):
    """One ordered package with semantic identity and exact audit evidence."""

    package_id: OpaqueUuid
    ordinal: NonNegativeInt
    semantic_content_sha256: Sha256Digest
    manifest_bytes_sha256: Sha256Digest
    cameras: FrozenTuple[CatalogCamera]

    @model_validator(mode="after")
    def validate_cameras(self) -> Self:
        if tuple(camera.camera_id for camera in self.cameras) != CAMERA_IDS:
            raise ValueError("catalog packages must contain all six cameras in canonical order")
        return self


class RequestCatalog(StrictModel):
    """Immutable authoritative catalog with a run-independent semantic digest."""

    schema_version: Literal["1.0"]
    request_catalog_id: OpaqueUuid
    task: VisionTask
    packages: FrozenTuple[CatalogPackage]
    semantic_sha256: Sha256Digest
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_catalog(self) -> Self:
        if not self.packages:
            raise ValueError("request catalog must contain at least one package")
        if tuple(package.ordinal for package in self.packages) != tuple(range(len(self.packages))):
            raise ValueError("catalog package ordinals must be contiguous from zero")
        package_ids = tuple(package.package_id for package in self.packages)
        if len(set(package_ids)) != len(package_ids):
            raise ValueError("catalog package IDs must be unique")
        frame_ids = tuple(
            frame.frame_id
            for package in self.packages
            for camera in package.cameras
            for frame in camera.frames
        )
        if not frame_ids:
            raise ValueError("request catalog must contain at least one source frame")
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("catalog frame IDs must be globally unique")
        expected = semantic_sha256(request_catalog_semantic_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("request catalog semantic_sha256 is inconsistent")
        return self


class SubjectPackageDigest(StrictModel):
    """Ordered semantic identity plus exact-byte audit evidence for one package."""

    package_id: OpaqueUuid
    ordinal: NonNegativeInt
    semantic_content_sha256: Sha256Digest
    manifest_bytes_sha256: Sha256Digest


class InputPlanSubject(StrictModel):
    """Provider-neutral evidence bound by an input plan."""

    task: VisionTask
    packages: FrozenTuple[SubjectPackageDigest]
    request_catalog_sha256: Sha256Digest


class InputPlanTarget(StrictModel):
    """Pinned provider/model target and capability evidence."""

    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    adapter_version: SchemaVersion
    planner_version: SchemaVersion
    capability_snapshot_id: OpaqueUuid
    capability_snapshot_sha256: Sha256Digest


class TransformParameter(StrictModel):
    """Canonical scalar transform parameter; floats never enter identity."""

    name: Annotated[
        str,
        StringConstraints(strict=True, pattern=r"^[a-z][a-z0-9_]*$"),
    ]
    value: TransformValue

    @model_validator(mode="after")
    def reject_empty_text(self) -> Self:
        if isinstance(self.value, str) and not self.value:
            raise ValueError("string transform parameters must be nonempty")
        return self


class FrameTransform(StrictModel):
    """Versioned transform parameters with a self-validating digest."""

    operation: TransformOperation
    policy_version: SchemaVersion
    parameters: FrozenTuple[TransformParameter]
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_transform(self) -> Self:
        names = tuple(parameter.name for parameter in self.parameters)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("transform parameters must have unique canonical name order")
        if self.operation is TransformOperation.NONE and self.parameters:
            raise ValueError("NONE transforms cannot carry parameters")
        if self.operation is not TransformOperation.NONE and not self.parameters:
            raise ValueError("material transforms require explicit parameters")
        if self.semantic_sha256 != semantic_sha256(_transform_projection(self)):
            raise ValueError("transform semantic_sha256 is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        operation: TransformOperation,
        policy_version: str,
        parameters: Sequence[TransformParameter] = (),
    ) -> FrameTransform:
        """Create a transform while deriving its semantic digest."""

        ordered = tuple(parameters)
        projection = {
            "operation": operation.value,
            "policy_version": policy_version,
            "parameters": [parameter.model_dump(mode="json") for parameter in ordered],
        }
        return cls(
            operation=operation,
            policy_version=policy_version,
            parameters=ordered,
            semantic_sha256=semantic_sha256(projection),
        )


class RenderedArtifact(StrictModel):
    """Exact provider-facing bytes; row identity and URI remain locators only."""

    artifact_id: OpaqueUuid
    uri: ArtifactUri
    sha256: Sha256Digest
    byte_count: PositiveInt
    media_type: MediaType
    encoding: NonEmptyString
    width: PositiveInt
    height: PositiveInt


class RenderedProviderItem(StrictModel):
    """One provider item with exact package/frame provenance."""

    provider_item_ordinal: NonNegativeInt
    package_id: OpaqueUuid
    package_ordinal: NonNegativeInt
    camera_id: CameraId
    camera_ordinal: NonNegativeInt
    frame_id: OpaqueUuid
    frame_ordinal: NonNegativeInt
    aligned_timestamp_ns: Nanoseconds
    source_timestamp_ns: Nanoseconds
    source_artifact_sha256: Sha256Digest
    artifact: RenderedArtifact
    transform: FrameTransform


class PromptOutputContract(StrictModel):
    """Pinned prompt, wire response, and enriched-domain contracts."""

    prompt_version: SchemaVersion
    prompt_sha256: Sha256Digest
    rendered_message_sha256: Sha256Digest
    provider_response_schema_sha256: Sha256Digest
    enriched_domain_schema_sha256: Sha256Digest
    protocol_mode: NonEmptyString
    tool_mode: NonEmptyString


class ApplicableProviderLimits(StrictModel):
    """Pinned per-request limits; ``None`` means explicitly unbounded."""

    max_images_per_request: PositiveInt | None
    max_pixels_per_image: PositiveInt | None
    max_payload_bytes_per_request: PositiveInt | None
    max_input_tokens_per_request: PositiveInt | None


class MeasuredProviderLimits(StrictModel):
    """Worst measured usage across the explicit call parts."""

    total_provider_items: PositiveInt
    max_images_per_request: NonNegativeInt
    max_pixels_per_image: NonNegativeInt
    max_payload_bytes_per_request: PositiveInt
    max_input_tokens_per_request: NonNegativeInt


class ProviderLimitDecision(StrictModel):
    """One explicit, self-consistent provider limit comparison."""

    metric: LimitMetric
    measured_value: NonNegativeInt
    applicable_limit: PositiveInt | None
    status: LimitDecisionStatus

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        expected = (
            LimitDecisionStatus.PASS
            if self.applicable_limit is None or self.measured_value <= self.applicable_limit
            else LimitDecisionStatus.FAIL
        )
        if self.status is not expected:
            raise ValueError("limit decision does not match its measurement and limit")
        return self


class CallPartSpec(StrictModel):
    """Caller-selected contiguous provider item range; the planner never auto-splits."""

    start_item_ordinal: NonNegativeInt
    end_item_ordinal_exclusive: PositiveInt
    measured_input_tokens: NonNegativeInt

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.start_item_ordinal >= self.end_item_ordinal_exclusive:
            raise ValueError("call part range must be nonempty")
        return self


class InferenceCallPart(StrictModel):
    """One ordered provider call joined by the input-plan barrier."""

    ordinal: NonNegativeInt
    part_count: PositiveInt
    start_item_ordinal: NonNegativeInt
    end_item_ordinal_exclusive: PositiveInt
    overlap_before_items: NonNegativeInt
    overlap_after_items: NonNegativeInt
    measured_input_tokens: NonNegativeInt
    item_manifest_sha256: Sha256Digest
    part_semantic_sha256: Sha256Digest
    part_logical_key: NonEmptyString
    idempotency_key: NonEmptyString


class InferenceCallPlan(StrictModel):
    """Ordered calls, durable logical barrier, and deterministic reduction."""

    call_plan_sha256: Sha256Digest
    parts: FrozenTuple[InferenceCallPart]
    idempotency_policy_version: SchemaVersion
    barrier_semantic_sha256: Sha256Digest
    barrier_logical_key: NonEmptyString
    reduction_policy: NonEmptyString
    reduction_policy_version: SchemaVersion

    @model_validator(mode="after")
    def validate_part_shape(self) -> Self:
        if not self.parts:
            raise ValueError("call plan must contain at least one explicit part")
        count = len(self.parts)
        if tuple(part.ordinal for part in self.parts) != tuple(range(count)):
            raise ValueError("call part ordinals must be contiguous and stored in order")
        if any(part.part_count != count for part in self.parts):
            raise ValueError("every call part_count must equal the call plan size")
        if len({part.part_semantic_sha256 for part in self.parts}) != count:
            raise ValueError("call part semantic identities must be unique")
        if len({part.part_logical_key for part in self.parts}) != count:
            raise ValueError("call part logical keys must be unique")
        if len({part.idempotency_key for part in self.parts}) != count:
            raise ValueError("call part idempotency keys must be unique")
        if self.parts[0].start_item_ordinal != 0:
            raise ValueError("first call part must start at provider item zero")
        if self.parts[0].overlap_before_items != 0:
            raise ValueError("first call part cannot have overlap_before_items")
        if self.parts[-1].overlap_after_items != 0:
            raise ValueError("last call part cannot have overlap_after_items")
        for previous, current in zip(self.parts, self.parts[1:], strict=False):
            if (
                current.start_item_ordinal <= previous.start_item_ordinal
                or current.end_item_ordinal_exclusive <= previous.end_item_ordinal_exclusive
            ):
                raise ValueError("call parts must make strict ordered progress")
            if current.start_item_ordinal > previous.end_item_ordinal_exclusive:
                raise ValueError("call parts cannot leave provider item gaps")
            overlap = previous.end_item_ordinal_exclusive - current.start_item_ordinal
            if previous.overlap_after_items != overlap or current.overlap_before_items != overlap:
                raise ValueError("call part overlap metadata is inconsistent")
        return self


class InferenceInputPlan(StrictModel):
    """Immutable executable provider-specific plan with run-independent identity."""

    schema_version: Literal["1.0"]
    input_plan_id: OpaqueUuid
    request_catalog: RequestCatalog
    subject: InputPlanSubject
    target: InputPlanTarget
    rendered_items: FrozenTuple[RenderedProviderItem]
    rendering_sha256: Sha256Digest
    prompt_output: PromptOutputContract
    applicable_limits: ApplicableProviderLimits
    measured_limits: MeasuredProviderLimits
    limit_decisions: FrozenTuple[ProviderLimitDecision]
    call_plan: InferenceCallPlan
    semantic_sha256: Sha256Digest
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        _validate_complete_plan(self)
        return self


class InferenceInputPlanner:
    """Build strict input plans from explicit provider rendering and call choices."""

    def __init__(self, planner_version: SchemaVersion) -> None:
        if planner_version != INFERENCE_INPUT_PLANNER_VERSION:
            raise ValueError(f"planner_version must be {INFERENCE_INPUT_PLANNER_VERSION!r}")
        self._planner_version = planner_version

    @property
    def planner_version(self) -> SchemaVersion:
        return self._planner_version

    def build_request_catalog(
        self,
        *,
        request_catalog_id: str,
        task: VisionTask,
        packages: Sequence[CatalogPackage],
        created_at: str,
    ) -> RequestCatalog:
        """Create a catalog digest without row IDs, exact manifests, locators, or time."""

        package_tuple = tuple(packages)
        projection = _request_catalog_projection(task=task, packages=package_tuple)
        return RequestCatalog(
            schema_version="1.0",
            request_catalog_id=request_catalog_id,
            task=task,
            packages=package_tuple,
            semantic_sha256=semantic_sha256(projection),
            created_at=created_at,
        )

    def build(
        self,
        *,
        input_plan_id: str,
        created_at: str,
        request_catalog: RequestCatalog,
        target: InputPlanTarget,
        rendered_items: Sequence[RenderedProviderItem],
        prompt_output: PromptOutputContract,
        applicable_limits: ApplicableProviderLimits,
        call_parts: Sequence[CallPartSpec],
        idempotency_policy_version: SchemaVersion,
        reduction_policy: str,
        reduction_policy_version: SchemaVersion,
    ) -> InferenceInputPlan:
        """Build an executable plan, rejecting loss or an over-limit call plan."""

        if target.planner_version != self._planner_version:
            raise InputPlanError("target planner_version does not match this planner")
        items = tuple(rendered_items)
        specs = tuple(call_parts)
        if not specs:
            raise InputPlanError(
                "call parts must be explicit; implicit provider splitting is forbidden"
            )
        subject = _subject_from_catalog(request_catalog)
        _validate_rendered_items(request_catalog, items)
        rendering_sha256 = semantic_sha256([_rendered_item_projection(item) for item in items])
        parts = _build_call_parts(items=items, specs=specs)
        call_plan_sha256 = _call_plan_digest(
            subject=subject,
            target=target,
            rendering_sha256=rendering_sha256,
            prompt_output=prompt_output,
            parts=parts,
            idempotency_policy_version=idempotency_policy_version,
            reduction_policy=reduction_policy,
            reduction_policy_version=reduction_policy_version,
        )
        parts = tuple(
            part.model_copy(
                update={
                    "part_semantic_sha256": _part_semantic_sha256(call_plan_sha256, part),
                    "part_logical_key": _logical_key(
                        CALL_PART_LOGICAL_KEY_NAMESPACE,
                        _part_semantic_sha256(call_plan_sha256, part),
                    ),
                    "idempotency_key": _idempotency_key(
                        call_plan_sha256, part, idempotency_policy_version
                    ),
                }
            )
            for part in parts
        )
        barrier_digest = _barrier_semantic_sha256(
            call_plan_sha256=call_plan_sha256,
            part_count=len(parts),
            reduction_policy=reduction_policy,
            reduction_policy_version=reduction_policy_version,
        )
        call_plan = InferenceCallPlan(
            call_plan_sha256=call_plan_sha256,
            parts=parts,
            idempotency_policy_version=idempotency_policy_version,
            barrier_semantic_sha256=barrier_digest,
            barrier_logical_key=_logical_key(
                CALL_BARRIER_LOGICAL_KEY_NAMESPACE,
                barrier_digest,
            ),
            reduction_policy=reduction_policy,
            reduction_policy_version=reduction_policy_version,
        )
        measured = _measure_provider_limits(items, parts)
        decisions = _limit_decisions(measured, applicable_limits)
        if any(decision.status is LimitDecisionStatus.FAIL for decision in decisions):
            raise InputPlanLimitError(decisions)

        provisional = InferenceInputPlan.model_construct(
            schema_version="1.0",
            input_plan_id=input_plan_id,
            request_catalog=request_catalog,
            subject=subject,
            target=target,
            rendered_items=items,
            rendering_sha256=rendering_sha256,
            prompt_output=prompt_output,
            applicable_limits=applicable_limits,
            measured_limits=measured,
            limit_decisions=decisions,
            call_plan=call_plan,
            semantic_sha256="0" * 64,
            created_at=created_at,
        )
        return InferenceInputPlan(
            schema_version="1.0",
            input_plan_id=input_plan_id,
            request_catalog=request_catalog,
            subject=subject,
            target=target,
            rendered_items=items,
            rendering_sha256=rendering_sha256,
            prompt_output=prompt_output,
            applicable_limits=applicable_limits,
            measured_limits=measured,
            limit_decisions=decisions,
            call_plan=call_plan,
            semantic_sha256=semantic_sha256(input_plan_semantic_projection(provisional)),
            created_at=created_at,
        )


def request_catalog_semantic_projection(catalog: RequestCatalog) -> dict[str, object]:
    """Return catalog identity without row IDs, exact manifests, locators, or time."""

    return _request_catalog_projection(task=catalog.task, packages=catalog.packages)


def input_plan_semantic_projection(plan: InferenceInputPlan) -> dict[str, object]:
    """Return the complete stable input-plan identity projection."""

    return {
        "semantic_projection_version": INPUT_PLAN_SEMANTIC_PROJECTION_VERSION,
        "schema_version": plan.schema_version,
        "subject": _subject_projection(plan.subject),
        "target": _target_projection(plan.target),
        "rendered_items": [_rendered_item_projection(item) for item in plan.rendered_items],
        "rendering_sha256": plan.rendering_sha256,
        "prompt_output": plan.prompt_output.model_dump(mode="json"),
        "applicable_limits": plan.applicable_limits.model_dump(mode="json"),
        "measured_limits": plan.measured_limits.model_dump(mode="json"),
        "limit_decisions": [decision.model_dump(mode="json") for decision in plan.limit_decisions],
        "call_plan": plan.call_plan.model_dump(mode="json"),
    }


def _request_catalog_projection(
    *,
    task: VisionTask,
    packages: Sequence[CatalogPackage],
) -> dict[str, object]:
    return {
        "semantic_projection_version": REQUEST_CATALOG_SEMANTIC_PROJECTION_VERSION,
        "task": task.value,
        "packages": [_catalog_package_projection(package) for package in packages],
    }


def _catalog_package_projection(package: CatalogPackage) -> dict[str, object]:
    return {
        "ordinal": package.ordinal,
        "semantic_content_sha256": package.semantic_content_sha256,
        "cameras": [
            {
                "camera_id": camera.camera_id.value,
                "ordinal": camera.ordinal,
                "frames": [
                    {
                        "ordinal": frame.ordinal,
                        "aligned_timestamp_ns": str(frame.aligned_timestamp_ns),
                        "source_timestamp_ns": str(frame.source_timestamp_ns),
                        "source_artifact_sha256": frame.source_artifact_sha256,
                        "source_artifact_bytes": frame.source_artifact_bytes,
                        "media_type": frame.media_type,
                        "encoding": frame.encoding,
                        "width": frame.width,
                        "height": frame.height,
                    }
                    for frame in camera.frames
                ],
            }
            for camera in package.cameras
        ],
    }


def _subject_from_catalog(catalog: RequestCatalog) -> InputPlanSubject:
    return InputPlanSubject(
        task=catalog.task,
        packages=tuple(
            SubjectPackageDigest(
                package_id=package.package_id,
                ordinal=package.ordinal,
                semantic_content_sha256=package.semantic_content_sha256,
                manifest_bytes_sha256=package.manifest_bytes_sha256,
            )
            for package in catalog.packages
        ),
        request_catalog_sha256=catalog.semantic_sha256,
    )


def _subject_projection(subject: InputPlanSubject) -> dict[str, object]:
    return {
        "task": subject.task.value,
        "packages": [
            {
                "ordinal": package.ordinal,
                "semantic_content_sha256": package.semantic_content_sha256,
            }
            for package in subject.packages
        ],
        "request_catalog_sha256": subject.request_catalog_sha256,
    }


def _target_projection(target: InputPlanTarget) -> dict[str, object]:
    return {
        "provider": target.provider,
        "model_name": target.model_name,
        "model_version": target.model_version,
        "adapter_version": target.adapter_version,
        "planner_version": target.planner_version,
        "capability_snapshot_sha256": target.capability_snapshot_sha256,
    }


def _transform_projection(transform: FrameTransform) -> dict[str, object]:
    return {
        "operation": transform.operation.value,
        "policy_version": transform.policy_version,
        "parameters": [parameter.model_dump(mode="json") for parameter in transform.parameters],
    }


def _rendered_item_projection(item: RenderedProviderItem) -> dict[str, object]:
    return {
        "provider_item_ordinal": item.provider_item_ordinal,
        "package_ordinal": item.package_ordinal,
        "camera_id": item.camera_id.value,
        "camera_ordinal": item.camera_ordinal,
        "frame_ordinal": item.frame_ordinal,
        "aligned_timestamp_ns": str(item.aligned_timestamp_ns),
        "source_timestamp_ns": str(item.source_timestamp_ns),
        "source_artifact_sha256": item.source_artifact_sha256,
        "artifact": {
            "sha256": item.artifact.sha256,
            "byte_count": item.artifact.byte_count,
            "media_type": item.artifact.media_type,
            "encoding": item.artifact.encoding,
            "width": item.artifact.width,
            "height": item.artifact.height,
        },
        "transform": {
            **_transform_projection(item.transform),
            "semantic_sha256": item.transform.semantic_sha256,
        },
    }


def _catalog_frame_sequence(
    catalog: RequestCatalog,
) -> tuple[tuple[CatalogPackage, CatalogCamera, CatalogFrame], ...]:
    return tuple(
        (package, camera, frame)
        for package in catalog.packages
        for camera in package.cameras
        for frame in camera.frames
    )


def _validate_rendered_items(
    catalog: RequestCatalog,
    items: Sequence[RenderedProviderItem],
) -> None:
    expected = _catalog_frame_sequence(catalog)
    if len(items) != len(expected):
        raise InputPlanError("rendering must map every catalog frame exactly once")
    if tuple(item.provider_item_ordinal for item in items) != tuple(range(len(items))):
        raise InputPlanError("provider item ordinals must be contiguous and stored in order")

    for item, (package, camera, frame) in zip(items, expected, strict=True):
        actual_binding = (
            item.package_id,
            item.package_ordinal,
            item.camera_id,
            item.camera_ordinal,
            item.frame_id,
            item.frame_ordinal,
            item.aligned_timestamp_ns,
            item.source_timestamp_ns,
            item.source_artifact_sha256,
        )
        expected_binding = (
            package.package_id,
            package.ordinal,
            camera.camera_id,
            camera.ordinal,
            frame.frame_id,
            frame.ordinal,
            frame.aligned_timestamp_ns,
            frame.source_timestamp_ns,
            frame.source_artifact_sha256,
        )
        if actual_binding != expected_binding:
            raise InputPlanError("rendering must preserve package, camera, and frame catalog order")
        _validate_transform_result(frame, item)


def _validate_transform_result(frame: CatalogFrame, item: RenderedProviderItem) -> None:
    artifact = item.artifact
    operation = item.transform.operation
    same_dimensions = (artifact.width, artifact.height) == (frame.width, frame.height)
    same_representation = (
        artifact.media_type == frame.media_type and artifact.encoding == frame.encoding
    )
    same_bytes = (
        artifact.sha256 == frame.source_artifact_sha256
        and artifact.byte_count == frame.source_artifact_bytes
    )

    if operation is TransformOperation.NONE:
        if not same_dimensions or not same_representation or not same_bytes:
            raise InputPlanError("NONE transform must preserve source bytes and representation")
    elif operation is TransformOperation.RESIZE:
        if same_dimensions or not same_representation or same_bytes:
            raise InputPlanError("RESIZE must change dimensions only within the representation")
    elif operation is TransformOperation.TRANSCODE:
        if not same_dimensions or same_bytes:
            raise InputPlanError("TRANSCODE must preserve dimensions and create new bytes")
    elif operation is TransformOperation.RESIZE_TRANSCODE and (same_dimensions or same_bytes):
        raise InputPlanError("RESIZE_TRANSCODE must change dimensions and bytes")


def _raw_part_projection(part: InferenceCallPart) -> dict[str, object]:
    return {
        "ordinal": part.ordinal,
        "part_count": part.part_count,
        "start_item_ordinal": part.start_item_ordinal,
        "end_item_ordinal_exclusive": part.end_item_ordinal_exclusive,
        "overlap_before_items": part.overlap_before_items,
        "overlap_after_items": part.overlap_after_items,
        "measured_input_tokens": part.measured_input_tokens,
        "item_manifest_sha256": part.item_manifest_sha256,
    }


def _build_call_parts(
    *,
    items: tuple[RenderedProviderItem, ...],
    specs: tuple[CallPartSpec, ...],
) -> tuple[InferenceCallPart, ...]:
    count = len(specs)
    parts: list[InferenceCallPart] = []
    for ordinal, spec in enumerate(specs):
        if spec.end_item_ordinal_exclusive > len(items):
            raise InputPlanError("call part range exceeds rendered provider items")
        overlap_before = (
            0
            if ordinal == 0
            else specs[ordinal - 1].end_item_ordinal_exclusive - spec.start_item_ordinal
        )
        overlap_after = (
            0
            if ordinal == count - 1
            else spec.end_item_ordinal_exclusive - specs[ordinal + 1].start_item_ordinal
        )
        if overlap_before < 0 or overlap_after < 0:
            raise InputPlanError("call parts cannot leave provider item gaps")
        manifest = semantic_sha256(
            [
                _rendered_item_projection(item)
                for item in items[spec.start_item_ordinal : spec.end_item_ordinal_exclusive]
            ]
        )
        parts.append(
            InferenceCallPart(
                ordinal=ordinal,
                part_count=count,
                start_item_ordinal=spec.start_item_ordinal,
                end_item_ordinal_exclusive=spec.end_item_ordinal_exclusive,
                overlap_before_items=overlap_before,
                overlap_after_items=overlap_after,
                measured_input_tokens=spec.measured_input_tokens,
                item_manifest_sha256=manifest,
                part_semantic_sha256=f"{ordinal + 1:064x}",
                part_logical_key=f"pending-part-{ordinal}",
                idempotency_key=f"pending-part-{ordinal}",
            )
        )
    if parts[0].start_item_ordinal != 0 or parts[-1].end_item_ordinal_exclusive != len(items):
        raise InputPlanError("explicit call parts must cover every rendered provider item")
    # Exercise the local ordering/overlap validator before deriving identities.
    InferenceCallPlan(
        call_plan_sha256="0" * 64,
        parts=tuple(parts),
        idempotency_policy_version="pending",
        barrier_semantic_sha256="0" * 64,
        barrier_logical_key="pending",
        reduction_policy="pending",
        reduction_policy_version="pending",
    )
    return tuple(parts)


def _call_plan_digest(
    *,
    subject: InputPlanSubject,
    target: InputPlanTarget,
    rendering_sha256: str,
    prompt_output: PromptOutputContract,
    parts: tuple[InferenceCallPart, ...],
    idempotency_policy_version: str,
    reduction_policy: str,
    reduction_policy_version: str,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "semantic_projection_version": CALL_PLAN_SEMANTIC_PROJECTION_VERSION,
            "subject": _subject_projection(subject),
            "target": _target_projection(target),
            "rendering_sha256": rendering_sha256,
            "prompt_output": prompt_output.model_dump(mode="json"),
            "parts": [_raw_part_projection(part) for part in parts],
            "idempotency_policy_version": idempotency_policy_version,
            "reduction_policy": reduction_policy,
            "reduction_policy_version": reduction_policy_version,
        }
    )


def _idempotency_key(
    call_plan_sha256: str,
    part: InferenceCallPart,
    policy_version: str,
) -> str:
    part_digest = _part_semantic_sha256(call_plan_sha256, part)
    digest = semantic_sha256(
        {
            "key_policy_version": CALL_IDEMPOTENCY_KEY_POLICY_VERSION,
            "part_semantic_sha256": part_digest,
            "provider_policy_version": policy_version,
        }
    )
    return _logical_key(CALL_IDEMPOTENCY_KEY_NAMESPACE, digest)


def _part_semantic_sha256(
    call_plan_sha256: str,
    part: InferenceCallPart,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "semantic_projection_version": CALL_PART_SEMANTIC_PROJECTION_VERSION,
            "call_plan_sha256": call_plan_sha256,
            "part": _raw_part_projection(part),
        }
    )


def _barrier_semantic_sha256(
    *,
    call_plan_sha256: str,
    part_count: int,
    reduction_policy: str,
    reduction_policy_version: str,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "semantic_projection_version": CALL_BARRIER_SEMANTIC_PROJECTION_VERSION,
            "call_plan_sha256": call_plan_sha256,
            "part_count": part_count,
            "reduction_policy": reduction_policy,
            "reduction_policy_version": reduction_policy_version,
        }
    )


def _logical_key(namespace: str, digest: str) -> str:
    return f"{namespace}:{digest}"


def _measure_provider_limits(
    items: Sequence[RenderedProviderItem],
    parts: Sequence[InferenceCallPart],
) -> MeasuredProviderLimits:
    images: list[int] = []
    pixels: list[int] = []
    payloads: list[int] = []
    tokens: list[int] = []
    for part in parts:
        part_items = items[part.start_item_ordinal : part.end_item_ordinal_exclusive]
        image_items = tuple(
            item for item in part_items if item.artifact.media_type.startswith("image/")
        )
        images.append(len(image_items))
        pixels.append(
            max(
                (item.artifact.width * item.artifact.height for item in image_items),
                default=0,
            )
        )
        payloads.append(sum(item.artifact.byte_count for item in part_items))
        tokens.append(part.measured_input_tokens)
    return MeasuredProviderLimits(
        total_provider_items=len(items),
        max_images_per_request=max(images),
        max_pixels_per_image=max(pixels),
        max_payload_bytes_per_request=max(payloads),
        max_input_tokens_per_request=max(tokens),
    )


def _limit_decisions(
    measured: MeasuredProviderLimits,
    limits: ApplicableProviderLimits,
) -> tuple[ProviderLimitDecision, ...]:
    values = (
        (
            LimitMetric.IMAGES_PER_REQUEST,
            measured.max_images_per_request,
            limits.max_images_per_request,
        ),
        (
            LimitMetric.PIXELS_PER_IMAGE,
            measured.max_pixels_per_image,
            limits.max_pixels_per_image,
        ),
        (
            LimitMetric.PAYLOAD_BYTES_PER_REQUEST,
            measured.max_payload_bytes_per_request,
            limits.max_payload_bytes_per_request,
        ),
        (
            LimitMetric.INPUT_TOKENS_PER_REQUEST,
            measured.max_input_tokens_per_request,
            limits.max_input_tokens_per_request,
        ),
    )
    return tuple(
        ProviderLimitDecision(
            metric=metric,
            measured_value=value,
            applicable_limit=limit,
            status=(
                LimitDecisionStatus.PASS
                if limit is None or value <= limit
                else LimitDecisionStatus.FAIL
            ),
        )
        for metric, value, limit in values
    )


def _validate_complete_plan(plan: InferenceInputPlan) -> None:
    if plan.target.planner_version != INFERENCE_INPUT_PLANNER_VERSION:
        raise ValueError(
            f"input plan target planner_version must be {INFERENCE_INPUT_PLANNER_VERSION!r}"
        )
    expected_subject = _subject_from_catalog(plan.request_catalog)
    if plan.subject != expected_subject:
        raise ValueError("input plan subject does not match its request catalog")
    _validate_rendered_items(plan.request_catalog, plan.rendered_items)
    expected_rendering = semantic_sha256(
        [_rendered_item_projection(item) for item in plan.rendered_items]
    )
    if plan.rendering_sha256 != expected_rendering:
        raise ValueError("rendering_sha256 is inconsistent")
    if not plan.call_plan.parts:
        raise ValueError("call plan must contain explicit parts")
    if plan.call_plan.parts[-1].end_item_ordinal_exclusive != len(plan.rendered_items):
        raise ValueError("call plan must cover every rendered provider item")

    expected_call_digest = _call_plan_digest(
        subject=plan.subject,
        target=plan.target,
        rendering_sha256=plan.rendering_sha256,
        prompt_output=plan.prompt_output,
        parts=plan.call_plan.parts,
        idempotency_policy_version=plan.call_plan.idempotency_policy_version,
        reduction_policy=plan.call_plan.reduction_policy,
        reduction_policy_version=plan.call_plan.reduction_policy_version,
    )
    if plan.call_plan.call_plan_sha256 != expected_call_digest:
        raise ValueError("call_plan_sha256 is inconsistent")
    for part in plan.call_plan.parts:
        items = plan.rendered_items[part.start_item_ordinal : part.end_item_ordinal_exclusive]
        expected_manifest = semantic_sha256([_rendered_item_projection(item) for item in items])
        if part.item_manifest_sha256 != expected_manifest:
            raise ValueError("call part item_manifest_sha256 is inconsistent")
        expected_part_digest = _part_semantic_sha256(expected_call_digest, part)
        if (
            part.part_semantic_sha256 != expected_part_digest
            or part.part_logical_key
            != _logical_key(CALL_PART_LOGICAL_KEY_NAMESPACE, expected_part_digest)
        ):
            raise ValueError("call part semantic identity is inconsistent")
        expected_key = _idempotency_key(
            expected_call_digest,
            part,
            plan.call_plan.idempotency_policy_version,
        )
        if part.idempotency_key != expected_key:
            raise ValueError("call part idempotency_key is inconsistent")

    expected_barrier = _barrier_semantic_sha256(
        call_plan_sha256=expected_call_digest,
        part_count=len(plan.call_plan.parts),
        reduction_policy=plan.call_plan.reduction_policy,
        reduction_policy_version=plan.call_plan.reduction_policy_version,
    )
    if (
        plan.call_plan.barrier_semantic_sha256 != expected_barrier
        or plan.call_plan.barrier_logical_key
        != _logical_key(CALL_BARRIER_LOGICAL_KEY_NAMESPACE, expected_barrier)
    ):
        raise ValueError("call plan barrier identity is inconsistent")

    expected_measured = _measure_provider_limits(plan.rendered_items, plan.call_plan.parts)
    if plan.measured_limits != expected_measured:
        raise ValueError("measured provider limits are inconsistent")
    expected_decisions = _limit_decisions(expected_measured, plan.applicable_limits)
    if plan.limit_decisions != expected_decisions:
        raise ValueError("provider limit decisions are inconsistent")
    if any(decision.status is LimitDecisionStatus.FAIL for decision in plan.limit_decisions):
        raise ValueError("an executable input plan cannot contain failed provider limits")
    expected_semantic = semantic_sha256(input_plan_semantic_projection(plan))
    if plan.semantic_sha256 != expected_semantic:
        raise ValueError("input plan semantic_sha256 is inconsistent")


__all__ = [
    "CALL_BARRIER_LOGICAL_KEY_NAMESPACE",
    "CALL_BARRIER_SEMANTIC_PROJECTION_VERSION",
    "CALL_IDEMPOTENCY_KEY_NAMESPACE",
    "CALL_IDEMPOTENCY_KEY_POLICY_VERSION",
    "CALL_PART_LOGICAL_KEY_NAMESPACE",
    "CALL_PART_SEMANTIC_PROJECTION_VERSION",
    "CALL_PLAN_SEMANTIC_PROJECTION_VERSION",
    "INFERENCE_INPUT_PLANNER_VERSION",
    "INPUT_PLAN_LOGICAL_KEY_NAMESPACE",
    "INPUT_PLAN_SEMANTIC_PROJECTION_VERSION",
    "INPUT_PLAN_UUID_NAMESPACE",
    "REQUEST_CATALOG_SEMANTIC_PROJECTION_VERSION",
    "REQUEST_CATALOG_UUID_NAMESPACE",
    "ApplicableProviderLimits",
    "CallPartSpec",
    "CatalogCamera",
    "CatalogFrame",
    "CatalogPackage",
    "FrameTransform",
    "InferenceCallPart",
    "InferenceCallPlan",
    "InferenceInputPlan",
    "InferenceInputPlanner",
    "InputPlanError",
    "InputPlanLimitError",
    "InputPlanSubject",
    "InputPlanTarget",
    "LimitDecisionStatus",
    "LimitMetric",
    "MeasuredProviderLimits",
    "PromptOutputContract",
    "ProviderLimitDecision",
    "RenderedArtifact",
    "RenderedProviderItem",
    "RequestCatalog",
    "SubjectPackageDigest",
    "TransformOperation",
    "TransformParameter",
    "input_plan_semantic_projection",
    "request_catalog_semantic_projection",
]
