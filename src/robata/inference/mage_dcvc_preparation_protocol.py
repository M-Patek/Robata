"""Internal JSONL contracts for the endpoint-external Mage DCVC preparation worker.

These models are an operational process boundary, not a published product schema.
They intentionally bind the effective provider implementation and configuration while
keeping local mount paths out of semantic identities.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Final, Literal, Self

from pydantic import Field, JsonValue, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256

MAGE_DCVC_PROVIDER_VERSION: Final = "robata-mage-dcvc-provider-v2"
MAGE_DCVC_RECIPE_VERSION: Final = "mage-dcvc-readiness-explicit-v2"
MAGE_DCVC_EFFECTIVE_CONFIG_VERSION: Final = "mage-dcvc-effective-config-v2"
MAGE_DCVC_PREPARATION_IDENTITY_VERSION: Final = "mage-dcvc-preparation-identity-v2"
MAGE_DCVC_PREPARATION_REQUEST_VERSION: Final = "mage-dcvc-preparation-request-v2"
MAGE_DCVC_PREPARATION_ARTIFACT_VERSION: Final = "mage-dcvc-preparation-artifact-v2"
MAGE_DCVC_PREPARATION_RESPONSE_VERSION: Final = "mage-dcvc-preparation-response-v2"
MAGE_DCVC_PREPARATION_SIDECAR_NAME: Final = ".robata-dcvc-preparation-v2.json"
MAGE_DCVC_TEMP_MARKER_VERSION: Final = "mage-dcvc-preparation-temp-v2"
MAGE_DCVC_TEMP_MARKER_NAME: Final = ".robata-dcvc-preparation-temp-v2.json"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16_384)]
SafeIdentifier = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._:-]+$"),
]
PreparationDevice = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(?:cpu|cuda(?::[0-9]+)?)$"),
]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0)]
UnitFloat = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]


class MageDcvcEffectiveConfig(StrictModel):
    """One immutable provider configuration loaded for the worker lifetime.

    ``sampled_frame_count`` controls which frames become observations. It is not an
    encode-step limit: the current recurrent DCVC path advances through the final
    sampled frame, which is stated explicitly by ``encoded_frame_extent``.
    """

    effective_config_version: Literal["mage-dcvc-effective-config-v2"] = (
        MAGE_DCVC_EFFECTIVE_CONFIG_VERSION
    )
    provider_version: Literal["robata-mage-dcvc-provider-v2"] = MAGE_DCVC_PROVIDER_VERSION
    recipe_version: Literal["mage-dcvc-readiness-explicit-v2"] = MAGE_DCVC_RECIPE_VERSION
    provider_implementation_sha256: Sha256Digest
    intra_checkpoint_sha256: Sha256Digest
    inter_checkpoint_sha256: Sha256Digest
    engine: Literal["dcvc-rt"] = "dcvc-rt"
    preparation_device: PreparationDevice
    device_concurrency_policy: Literal["exclusive-shared-device-v1", "separate-device-v1"]
    precision: Literal["fp16"] = "fp16"
    qp: Annotated[int, Field(strict=True, ge=0, le=63)] = 42
    reset_interval: PositiveInt = 64
    intra_period: Annotated[int, Field(strict=True, ge=-1)] = -1
    max_side: NonNegativeInt = 0
    target_canvas: PositiveInt = 32
    sampled_frame_count: PositiveInt = 256
    sequence_length_frames: Literal[0] = 0
    canvas_token_side: None = None
    encoded_frame_extent: Literal["through-last-sampled-frame"] = "through-last-sampled-frame"
    grouping_mode: Literal["readiness"] = "readiness"
    readiness_sum_threshold_mode: Literal["auto"] = "auto"
    group_size: PositiveInt = 32
    images_per_group: PositiveInt = 4
    patch: Literal[16] = 16
    max_pixels: PositiveInt = 150_000
    min_group_frames: PositiveInt = 8
    max_group_frames: PositiveInt = 128
    readiness_coverage_bins: PositiveInt = 3
    readiness_delta_ratio: UnitFloat = 0.05
    bitcost_grid: Literal["sub"] = "sub"
    bitcost_percentile: Annotated[int, Field(strict=True, ge=0, le=100)] = 99
    bitcost_log_scale: Literal[True] = True
    decode_backsearch_max: PositiveInt = 16
    canvas_format: Literal["jpg"] = "jpg"
    per_frame_cap_ratio: Annotated[float, Field(strict=True, gt=0.0)] = 1.2
    bottom_attenuation: UnitFloat = 0.5
    bottom_band_ratio: UnitFloat = 0.1
    threshold_scale: Annotated[float, Field(strict=True, gt=0.0)] = 1.0
    random_select: bool = False
    random_seed: NonNegativeInt = 0
    effective_config_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_effective_config(self) -> Self:
        if self.intra_period == 0:
            raise ValueError("intra_period must be -1 or a positive frame interval")
        if self.max_side != 0 and self.max_side < self.patch:
            raise ValueError("max_side must be 0 or at least one patch wide")
        if self.min_group_frames > self.max_group_frames:
            raise ValueError("min_group_frames must not exceed max_group_frames")
        if self.target_canvas % self.images_per_group != 0:
            raise ValueError("target_canvas must be divisible by images_per_group")
        expected_sampled = (self.target_canvas // self.images_per_group) * self.group_size
        if self.sampled_frame_count != expected_sampled:
            raise ValueError("sampled_frame_count must follow the Mage canvas formula")
        if self.effective_config_sha256 != mage_dcvc_effective_config_sha256(self):
            raise ValueError("effective_config_sha256 does not match the effective configuration")
        return self


class MageDcvcPreparationRequest(StrictModel):
    """One idempotent segment-preparation request sent to the resident worker."""

    request_version: Literal["mage-dcvc-preparation-request-v2"] = (
        MAGE_DCVC_PREPARATION_REQUEST_VERSION
    )
    request_id: SafeIdentifier
    source_path: NonEmptyString
    source_content_sha256: Sha256Digest
    source_byte_count: PositiveInt
    output_relative_path: NonEmptyString
    effective_config_sha256: Sha256Digest
    preparation_identity: Sha256Digest

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        path = PurePosixPath(self.output_relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("output_relative_path must be a safe relative POSIX path")
        if self.preparation_identity != mage_dcvc_preparation_identity(
            source_content_sha256=self.source_content_sha256,
            source_byte_count=self.source_byte_count,
            effective_config_sha256=self.effective_config_sha256,
        ):
            raise ValueError("preparation_identity does not match the request projection")
        return self


class MageDcvcPreparedAsset(StrictModel):
    """One exact provider asset committed by a preparation job."""

    relative_path: NonEmptyString
    byte_count: PositiveInt
    sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_relative_path(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("relative_path must be a safe relative POSIX path")
        if self.relative_path in {
            MAGE_DCVC_PREPARATION_SIDECAR_NAME,
            MAGE_DCVC_TEMP_MARKER_NAME,
        }:
            raise ValueError("worker metadata files cannot be provider assets")
        return self


class MageDcvcPreparationArtifact(StrictModel):
    """Exact-byte manifest committed beside one provider output directory."""

    artifact_version: Literal["mage-dcvc-preparation-artifact-v2"] = (
        MAGE_DCVC_PREPARATION_ARTIFACT_VERSION
    )
    preparation_identity: Sha256Digest
    effective_config_sha256: Sha256Digest
    provider_version: Literal["robata-mage-dcvc-provider-v2"] = MAGE_DCVC_PROVIDER_VERSION
    recipe_version: Literal["mage-dcvc-readiness-explicit-v2"] = MAGE_DCVC_RECIPE_VERSION
    provider_implementation_sha256: Sha256Digest
    source_content_sha256: Sha256Digest
    source_byte_count: PositiveInt
    assets: tuple[MageDcvcPreparedAsset, ...]
    provider_metadata: dict[str, JsonValue]
    artifact_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        relative_paths = tuple(asset.relative_path for asset in self.assets)
        if not relative_paths:
            raise ValueError("at least one provider asset is required")
        if relative_paths != tuple(sorted(relative_paths)):
            raise ValueError("assets must be sorted by relative_path")
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("asset relative paths must be unique")
        if self.artifact_semantic_sha256 != mage_dcvc_artifact_semantic_sha256(self):
            raise ValueError("artifact_semantic_sha256 does not match the artifact projection")
        return self


class MageDcvcPreparationResponse(StrictModel):
    """One JSONL response; failures never contain an admitted artifact."""

    response_version: Literal["mage-dcvc-preparation-response-v2"] = (
        MAGE_DCVC_PREPARATION_RESPONSE_VERSION
    )
    request_id: SafeIdentifier
    status: Literal["BUILT", "VERIFIED_HIT", "REJECTED", "BUSY", "FAILED"]
    preparation_identity: Sha256Digest | None = None
    artifact_semantic_sha256: Sha256Digest | None = None
    output_directory: NonEmptyString | None = None
    wall_seconds: NonNegativeFloat
    error_code: SafeIdentifier | None = None
    error_message: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        admitted = self.status in {"BUILT", "VERIFIED_HIT"}
        artifact_fields = (
            self.preparation_identity,
            self.artifact_semantic_sha256,
            self.output_directory,
        )
        if admitted and any(value is None for value in artifact_fields):
            raise ValueError("admitted responses require identity, artifact digest, and directory")
        if not admitted and any(value is not None for value in artifact_fields):
            raise ValueError("non-admitted responses cannot contain artifact fields")
        if admitted and (self.error_code is not None or self.error_message is not None):
            raise ValueError("admitted responses cannot contain an error")
        if not admitted and (self.error_code is None or self.error_message is None):
            raise ValueError("non-admitted responses require an error code and message")
        return self


class MageDcvcTempMarker(StrictModel):
    """Marker proving a staging directory belongs to one exact worker job."""

    marker_version: Literal["mage-dcvc-preparation-temp-v2"] = MAGE_DCVC_TEMP_MARKER_VERSION
    request_id: SafeIdentifier
    preparation_identity: Sha256Digest
    effective_config_sha256: Sha256Digest


def mage_dcvc_effective_config_sha256(config: MageDcvcEffectiveConfig) -> Sha256Digest:
    """Hash every effective behavior field while excluding only the self digest."""

    return semantic_sha256(config.model_dump(mode="json", exclude={"effective_config_sha256"}))


def mage_dcvc_preparation_identity(
    *,
    source_content_sha256: Sha256Digest,
    source_byte_count: int,
    effective_config_sha256: Sha256Digest,
) -> Sha256Digest:
    """Address one prepared segment independently of local transport paths."""

    return semantic_sha256(
        {
            "identity_version": MAGE_DCVC_PREPARATION_IDENTITY_VERSION,
            "source_content_sha256": source_content_sha256,
            "source_byte_count": source_byte_count,
            "effective_config_sha256": effective_config_sha256,
        }
    )


def mage_dcvc_artifact_semantic_sha256(
    artifact: MageDcvcPreparationArtifact,
) -> Sha256Digest:
    """Hash the complete exact-byte artifact projection except its self digest."""

    return semantic_sha256(artifact.model_dump(mode="json", exclude={"artifact_semantic_sha256"}))


__all__ = [
    "MAGE_DCVC_EFFECTIVE_CONFIG_VERSION",
    "MAGE_DCVC_PREPARATION_ARTIFACT_VERSION",
    "MAGE_DCVC_PREPARATION_IDENTITY_VERSION",
    "MAGE_DCVC_PREPARATION_REQUEST_VERSION",
    "MAGE_DCVC_PREPARATION_RESPONSE_VERSION",
    "MAGE_DCVC_PREPARATION_SIDECAR_NAME",
    "MAGE_DCVC_PROVIDER_VERSION",
    "MAGE_DCVC_RECIPE_VERSION",
    "MAGE_DCVC_TEMP_MARKER_NAME",
    "MAGE_DCVC_TEMP_MARKER_VERSION",
    "MageDcvcEffectiveConfig",
    "MageDcvcPreparationArtifact",
    "MageDcvcPreparationRequest",
    "MageDcvcPreparationResponse",
    "MageDcvcPreparedAsset",
    "MageDcvcTempMarker",
    "mage_dcvc_artifact_semantic_sha256",
    "mage_dcvc_effective_config_sha256",
    "mage_dcvc_preparation_identity",
]
