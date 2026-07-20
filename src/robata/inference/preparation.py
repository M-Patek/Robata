"""Prepare immutable temporal packages for a provider-specific input plan.

This module is deliberately a preparation boundary.  It reads materialized
provider-neutral packages, creates an authoritative request catalog, renders
one provider item per source frame, and chooses explicit call parts under the
currently pinned limits.  It never decodes media, calls a provider, or mutates
the package set.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated, Protocol, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.inference.input_plan import (
    ApplicableProviderLimits,
    CallPartSpec,
    CatalogCamera,
    CatalogFrame,
    CatalogPackage,
    FrameTransform,
    InferenceInputPlan,
    InferenceInputPlanner,
    InputPlanTarget,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    RequestCatalog,
    TransformOperation,
)
from robata.inference.models import VisionTask
from robata.sampling.materializer import (
    MaterializedArtifactManifest,
    MaterializedTemporalPackage,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
_OPAQUE_UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class InputPreparationError(ValueError):
    """The immutable package set cannot satisfy the requested render/limit contract."""


class ProviderRenderingPolicy(StrictModel):
    """Versioned, deterministic rendering and token-estimation policy."""

    version: NonEmptyString
    transform_policy_version: NonEmptyString
    idempotency_policy_version: NonEmptyString
    reduction_policy: NonEmptyString
    reduction_policy_version: NonEmptyString
    input_tokens_per_item: PositiveInt = 1
    fixed_input_tokens_per_part: NonNegativeInt = 0
    accepted_media_types: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def require_canonical_media_types(self) -> Self:
        if self.accepted_media_types != tuple(sorted(set(self.accepted_media_types))):
            raise ValueError("accepted_media_types must be unique and lexically ordered")
        return self


class RenderedItemFactory(Protocol):
    """Resolve a provider artifact and transform for one immutable source frame."""

    def __call__(
        self,
        package: MaterializedTemporalPackage,
        camera_id: CameraId,
        frame_ordinal: int,
        source_artifact: MaterializedArtifactManifest,
    ) -> tuple[RenderedArtifact, FrameTransform]:
        """Return exact provider bytes metadata and its explicit transform."""


class PreparedProviderRendering(StrictModel):
    """Provider rendering staged before the exact prompt contract is bound."""

    task: VisionTask
    request_catalog: RequestCatalog
    rendered_items: tuple[RenderedProviderItem, ...]
    applicable_limits: ApplicableProviderLimits
    call_parts: tuple[CallPartSpec, ...]

    @model_validator(mode="after")
    def validate_preparation(self) -> Self:
        if self.request_catalog.task is not self.task:
            raise ValueError("prepared request catalog task is inconsistent")
        ordinals = tuple(item.provider_item_ordinal for item in self.rendered_items)
        if ordinals != tuple(range(len(self.rendered_items))):
            raise ValueError("prepared provider item ordinals must be contiguous")
        if not self.rendered_items or not self.call_parts:
            raise ValueError("prepared rendering must contain items and call parts")
        return self


def applicable_limits_from_capabilities(
    *,
    max_images_per_request: int | None,
    max_pixels_per_image: int | None,
    max_payload_bytes: int | None,
    max_input_tokens: int | None,
) -> ApplicableProviderLimits:
    """Map a capability snapshot into the plan's provider-limit contract."""

    return ApplicableProviderLimits(
        max_images_per_request=max_images_per_request,
        max_pixels_per_image=max_pixels_per_image,
        max_payload_bytes_per_request=max_payload_bytes,
        max_input_tokens_per_request=max_input_tokens,
    )


class InputPlanPreparer:
    """Build a strict input plan from materialized package manifests."""

    def __init__(
        self,
        planner: InferenceInputPlanner,
        policy: ProviderRenderingPolicy,
    ) -> None:
        if not isinstance(planner, InferenceInputPlanner):
            raise TypeError("planner must be an InferenceInputPlanner")
        if not isinstance(policy, ProviderRenderingPolicy):
            raise TypeError("policy must be a ProviderRenderingPolicy")
        self._planner = planner
        self._policy = policy

    @property
    def policy(self) -> ProviderRenderingPolicy:
        return self._policy

    @property
    def planner_version(self) -> str:
        return self._planner.planner_version

    def prepare(
        self,
        *,
        packages: Sequence[MaterializedTemporalPackage],
        task: VisionTask,
        request_catalog_id: str,
        input_plan_id: str,
        target: InputPlanTarget,
        prompt_output: PromptOutputContract,
        applicable_limits: ApplicableProviderLimits,
        created_at: str,
        rendered_item_factory: RenderedItemFactory | None = None,
    ) -> InferenceInputPlan:
        """Materialize the catalog/rendering/call-part side of one input plan."""

        prepared = self.prepare_rendering(
            packages=packages,
            task=task,
            request_catalog_id=request_catalog_id,
            applicable_limits=applicable_limits,
            created_at=created_at,
            rendered_item_factory=rendered_item_factory,
        )
        return self.finalize(
            prepared=prepared,
            input_plan_id=input_plan_id,
            target=target,
            prompt_output=prompt_output,
            created_at=created_at,
        )

    def prepare_rendering(
        self,
        *,
        packages: Sequence[MaterializedTemporalPackage],
        task: VisionTask,
        request_catalog_id: str,
        applicable_limits: ApplicableProviderLimits,
        created_at: str,
        rendered_item_factory: RenderedItemFactory | None = None,
    ) -> PreparedProviderRendering:
        """Build catalog, rendered items, and call parts without a prompt digest cycle."""

        package_tuple = tuple(packages)
        if not package_tuple:
            raise InputPreparationError("at least one materialized package is required")
        if tuple(item.package.part.ordinal for item in package_tuple) != tuple(
            range(len(package_tuple))
        ):
            raise InputPreparationError("materialized package ordinals must be contiguous")

        catalogs = tuple(self._catalog_package(item) for item in package_tuple)
        catalog = self._planner.build_request_catalog(
            request_catalog_id=request_catalog_id,
            task=task,
            packages=catalogs,
            created_at=created_at,
        )
        factory = rendered_item_factory
        rendered_items_list: list[RenderedProviderItem] = []
        for package in package_tuple:
            for camera_id in CAMERA_IDS:
                for frame in package.package.cameras[camera_id].frames:
                    item = self._render_item(package, camera_id, frame, factory)
                    rendered_items_list.append(
                        item.model_copy(
                            update={
                                "provider_item_ordinal": len(rendered_items_list),
                            }
                        )
                    )
        rendered_items = tuple(rendered_items_list)
        call_parts = _partition_items(
            rendered_items,
            applicable_limits=applicable_limits,
            policy=self._policy,
        )
        try:
            return PreparedProviderRendering(
                task=task,
                request_catalog=catalog,
                rendered_items=rendered_items,
                applicable_limits=applicable_limits,
                call_parts=call_parts,
            )
        except (TypeError, ValueError) as exc:
            raise InputPreparationError(str(exc)) from exc

    def finalize(
        self,
        *,
        prepared: PreparedProviderRendering,
        input_plan_id: str,
        target: InputPlanTarget,
        prompt_output: PromptOutputContract,
        created_at: str,
    ) -> InferenceInputPlan:
        """Bind the exact prompt/output contract and finish an executable plan."""

        if not isinstance(prepared, PreparedProviderRendering):
            raise TypeError("prepared must be a PreparedProviderRendering")
        try:
            return self._planner.build(
                input_plan_id=input_plan_id,
                created_at=created_at,
                request_catalog=prepared.request_catalog,
                target=target,
                rendered_items=prepared.rendered_items,
                prompt_output=prompt_output,
                applicable_limits=prepared.applicable_limits,
                call_parts=prepared.call_parts,
                idempotency_policy_version=self._policy.idempotency_policy_version,
                reduction_policy=self._policy.reduction_policy,
                reduction_policy_version=self._policy.reduction_policy_version,
            )
        except (TypeError, ValueError) as exc:
            raise InputPreparationError(str(exc)) from exc

    def _catalog_package(self, materialized: MaterializedTemporalPackage) -> CatalogPackage:
        package = materialized.package
        cameras = tuple(
            CatalogCamera(
                camera_id=camera_id,
                ordinal=ordinal,
                frames=tuple(
                    CatalogFrame(
                        frame_id=_opaque_id(
                            "frame",
                            frame.frame_id,
                            frame.materialized_artifact,
                        ),
                        ordinal=frame.ordinal,
                        aligned_timestamp_ns=frame.aligned_timestamp_ns,
                        source_timestamp_ns=frame.source_timestamp_ns,
                        source_artifact_uri=_artifact(frame).uri,
                        source_artifact_sha256=_artifact(frame).sha256,
                        source_artifact_bytes=_artifact(frame).bytes,
                        media_type=_artifact(frame).media_type,
                        encoding=_encoding(_artifact(frame).media_type),
                        width=frame.width,
                        height=frame.height,
                    )
                    for frame in package.cameras[camera_id].frames
                ),
            )
            for ordinal, camera_id in enumerate(CAMERA_IDS)
        )
        try:
            return CatalogPackage(
                package_id=_opaque_id("package", package.package_id),
                ordinal=package.part.ordinal,
                semantic_content_sha256=package.semantic_content_sha256,
                manifest_bytes_sha256=materialized.package_manifest_sha256,
                cameras=cameras,
            )
        except (TypeError, ValueError) as exc:
            raise InputPreparationError(f"invalid materialized package: {exc}") from exc

    def _render_item(
        self,
        materialized: MaterializedTemporalPackage,
        camera_id: CameraId,
        frame: object,
        factory: RenderedItemFactory | None,
    ) -> RenderedProviderItem:
        # The manifest type is intentionally checked at this boundary instead of
        # relying on a duck-typed provider adapter.
        from robata.contracts.temporal import FrameSelectionManifest

        if not isinstance(frame, FrameSelectionManifest):
            raise InputPreparationError("materialized camera contains an invalid frame")
        source = _artifact(frame)
        if factory is None:
            artifact, transform = _default_rendered_item(
                materialized,
                camera_id,
                frame.ordinal,
                source,
                transform_policy_version=self._policy.transform_policy_version,
            )
        else:
            try:
                artifact, transform = factory(
                    materialized,
                    camera_id,
                    frame.ordinal,
                    source,
                )
            except Exception as exc:
                raise InputPreparationError(
                    "rendered item factory failed for "
                    f"package {materialized.package.package_id}, "
                    f"camera {camera_id.value}, frame {frame.ordinal}: {exc}"
                ) from exc
        if not isinstance(artifact, RenderedArtifact) or not isinstance(transform, FrameTransform):
            raise InputPreparationError("rendered item factory returned invalid contracts")
        if transform.policy_version != self._policy.transform_policy_version:
            raise InputPreparationError(
                "rendered item transform policy does not match the pinned rendering policy"
            )
        if self._policy.accepted_media_types and (
            artifact.media_type not in self._policy.accepted_media_types
        ):
            raise InputPreparationError(
                f"rendered media type is not accepted: {artifact.media_type}"
            )
        package = materialized.package
        package_id = _opaque_id("package", package.package_id)
        frame_id = _opaque_id("frame", frame.frame_id, frame.materialized_artifact)
        camera_ordinal = CAMERA_IDS.index(camera_id)
        return RenderedProviderItem(
            provider_item_ordinal=0,  # assigned by the canonical flattening pass
            package_id=package_id,
            package_ordinal=package.part.ordinal,
            camera_id=camera_id,
            camera_ordinal=camera_ordinal,
            frame_id=frame_id,
            frame_ordinal=frame.ordinal,
            aligned_timestamp_ns=frame.aligned_timestamp_ns,
            source_timestamp_ns=frame.source_timestamp_ns,
            source_artifact_sha256=source.sha256,
            artifact=artifact,
            transform=transform,
        )


def _artifact(frame: object) -> MaterializedArtifactManifest:
    from robata.contracts.temporal import FrameSelectionManifest

    if not isinstance(frame, FrameSelectionManifest) or frame.materialized_artifact is None:
        raise InputPreparationError("selected frame has no materialized artifact")
    try:
        return MaterializedArtifactManifest.model_validate(frame.materialized_artifact, strict=True)
    except (TypeError, ValueError) as exc:
        raise InputPreparationError(f"invalid materialized artifact: {exc}") from exc


def _opaque_id(domain: str, value: object, extra: object | None = None) -> str:
    text = str(value)
    if _OPAQUE_UUID_PATTERN.fullmatch(text) is not None:
        return text
    if extra is not None:
        text += f":{semantic_sha256(extra)}"
    return str(uuid5(NAMESPACE_URL, f"robata:input-plan:{domain}:v1:{text}"))


def _encoding(media_type: str) -> str:
    subtype = media_type.split("/", 1)[1]
    return subtype.split("+", 1)[0].lower()


def _default_rendered_item(
    package: MaterializedTemporalPackage,
    camera_id: CameraId,
    frame_ordinal: int,
    source_artifact: MaterializedArtifactManifest,
    *,
    transform_policy_version: str,
) -> tuple[RenderedArtifact, FrameTransform]:
    frame = package.package.cameras[camera_id].frames[frame_ordinal]
    return (
        RenderedArtifact(
            artifact_id=_opaque_id(
                "rendered-artifact",
                source_artifact.sha256,
                {
                    "bytes": source_artifact.bytes,
                    "media_type": source_artifact.media_type,
                    "width": frame.width,
                    "height": frame.height,
                },
            ),
            uri=source_artifact.uri,
            sha256=source_artifact.sha256,
            byte_count=source_artifact.bytes,
            media_type=source_artifact.media_type,
            encoding=_encoding(source_artifact.media_type),
            width=frame.width,
            height=frame.height,
        ),
        FrameTransform.create(
            operation=TransformOperation.NONE,
            policy_version=transform_policy_version,
        ),
    )


def _partition_items(
    items: Sequence[RenderedProviderItem],
    *,
    applicable_limits: ApplicableProviderLimits,
    policy: ProviderRenderingPolicy,
) -> tuple[CallPartSpec, ...]:
    if not items:
        raise InputPreparationError("materialized packages contain no selected frames")

    parts: list[CallPartSpec] = []
    start = 0
    images = 0
    payload = 0
    tokens = policy.fixed_input_tokens_per_part
    for ordinal, item in enumerate(items):
        is_image = item.artifact.media_type.startswith("image/")
        item_images = 1 if is_image else 0
        item_payload = item.artifact.byte_count
        item_tokens = policy.input_tokens_per_item
        if (
            applicable_limits.max_pixels_per_image is not None
            and item.artifact.width * item.artifact.height > applicable_limits.max_pixels_per_image
        ):
            raise InputPreparationError(
                f"item {ordinal} exceeds max_pixels_per_image before call partitioning"
            )

        exceeds = _exceeds(
            images + item_images,
            payload + item_payload,
            tokens + item_tokens,
            applicable_limits,
        )
        if exceeds and ordinal == start:
            raise InputPreparationError(
                f"item {ordinal} exceeds provider limits and cannot be split"
            )
        if exceeds:
            parts.append(
                CallPartSpec(
                    start_item_ordinal=start,
                    end_item_ordinal_exclusive=ordinal,
                    measured_input_tokens=tokens,
                )
            )
            start = ordinal
            images = 0
            payload = 0
            tokens = policy.fixed_input_tokens_per_part
        images += item_images
        payload += item_payload
        tokens += item_tokens

    parts.append(
        CallPartSpec(
            start_item_ordinal=start,
            end_item_ordinal_exclusive=len(items),
            measured_input_tokens=tokens,
        )
    )
    return tuple(parts)


def _exceeds(
    images: int,
    payload: int,
    tokens: int,
    limits: ApplicableProviderLimits,
) -> bool:
    return any(
        (
            limits.max_images_per_request is not None and images > limits.max_images_per_request,
            limits.max_payload_bytes_per_request is not None
            and payload > limits.max_payload_bytes_per_request,
            limits.max_input_tokens_per_request is not None
            and tokens > limits.max_input_tokens_per_request,
        )
    )


__all__ = [
    "InputPlanPreparer",
    "InputPreparationError",
    "PreparedProviderRendering",
    "ProviderRenderingPolicy",
    "RenderedItemFactory",
    "applicable_limits_from_capabilities",
]
