"""Single-route Mage authority and small-encoder shadow seams.

The first Robata vNext qualification deliberately runs one camera and one
resident Mage decoder.  This module makes that decision explicit while keeping a
provider-neutral interface for a future lightweight camera encoder.  The shadow
objects are diagnostic only: they cannot publish facts or replace the native Mage
authority result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, Protocol

from pydantic import Field, model_validator

from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.perception_stream import PerceptionContextManifest

SINGLE_CAMERA_AUTHORITY_POLICY_VERSION: Final = "single-camera-mage-authority-v1"
SMALL_ENCODER_OBSERVATION_VERSION: Final = "small-encoder-observation-v1"
SMALL_ENCODER_SHADOW_COMPARISON_VERSION: Final = "small-encoder-shadow-comparison-v1"


class SmallEncoderShadowMode(StrEnum):
    """Admission mode for an unqualified lightweight encoder."""

    DISABLED = "DISABLED"
    SHADOW_ONLY = "SHADOW_ONLY"


class SmallEncoderQuality(StrEnum):
    """Coarse quality emitted by a candidate camera encoder."""

    USABLE = "USABLE"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"
    UNKNOWN = "UNKNOWN"


class SingleCameraAuthorityPolicy(StrictModel):
    """Explicit first-stage execution policy.

    ``max_inflight_observations`` controls bounded preparation only; the Mage
    endpoint remains generation-concurrency=1 on the 8 GiB development GPU.
    """

    policy_version: Literal["single-camera-mage-authority-v1"] = (
        SINGLE_CAMERA_AUTHORITY_POLICY_VERSION
    )
    camera_id: CameraId = CameraId.CAM_01
    authority_provider: Literal["MAGE_NATIVE"] = "MAGE_NATIVE"
    shadow_encoder_mode: SmallEncoderShadowMode = SmallEncoderShadowMode.DISABLED
    worker_count: Literal[1] = 1
    generation_concurrency: Literal[1] = 1
    max_inflight_observations: Literal[1, 2] = 1
    raw_refine_provider: Literal["MAGE_NATIVE"] = "MAGE_NATIVE"


class CameraEncoderInput(StrictModel):
    """One bounded camera segment offered to a candidate small encoder."""

    context_manifest_key: str = Field(min_length=1, max_length=512)
    context_manifest_semantic_sha256: Sha256Digest
    camera_id: CameraId
    segment_semantic_sha256: Sha256Digest
    interval: NanosecondInterval
    durable_path: str = Field(min_length=1, max_length=16_384)

    @model_validator(mode="after")
    def validate_context_binding(self) -> CameraEncoderInput:
        if not self.context_manifest_key.endswith(f":{self.context_manifest_semantic_sha256}"):
            raise ValueError("camera encoder input must bind its context semantic digest")
        return self


class SmallEncoderActionCandidate(StrictModel):
    """A bounded, non-authoritative action hint from a candidate encoder."""

    label: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    )
    interval: NanosecondInterval
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SmallEncoderObservation(StrictModel):
    """Candidate camera observation; never a production fact in shadow mode."""

    schema_version: Literal["small-encoder-observation-v1"] = SMALL_ENCODER_OBSERVATION_VERSION
    encoder_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    )
    encoder_revision: str = Field(min_length=1, max_length=256)
    camera_id: CameraId
    context_manifest_key: str = Field(min_length=1, max_length=512)
    context_manifest_semantic_sha256: Sha256Digest
    segment_semantic_sha256: Sha256Digest
    interval: NanosecondInterval
    quality: SmallEncoderQuality
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_actions: tuple[SmallEncoderActionCandidate, ...] = ()
    feature_artifact_exact_sha256: Sha256Digest | None = None
    source_frame_exact_sha256_values: tuple[Sha256Digest, ...] = ()
    shadow_only: Literal[True] = True
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_observation(self) -> SmallEncoderObservation:
        if not self.context_manifest_key.endswith(f":{self.context_manifest_semantic_sha256}"):
            raise ValueError("small encoder context key must bind its semantic digest")
        for candidate in self.candidate_actions:
            if (
                candidate.interval.start_ns < self.interval.start_ns
                or candidate.interval.end_ns > self.interval.end_ns
            ):
                raise ValueError("small encoder action candidate must stay inside its segment")
        if len(set(self.source_frame_exact_sha256_values)) != len(
            self.source_frame_exact_sha256_values
        ):
            raise ValueError("small encoder source frame digests must be unique")
        expected = small_encoder_observation_semantic_sha256(self)
        if self.semantic_sha256 != expected:
            raise ValueError("small encoder observation semantic identity is inconsistent")
        return self


class SmallEncoderShadowComparison(StrictModel):
    """Lineage-only comparison record; it carries no publication decision."""

    schema_version: Literal["small-encoder-shadow-comparison-v1"] = (
        SMALL_ENCODER_SHADOW_COMPARISON_VERSION
    )
    context_manifest_semantic_sha256: Sha256Digest
    camera_id: CameraId
    authority_observation_logical_key: str = Field(min_length=1, max_length=512)
    authority_observation_semantic_sha256: Sha256Digest
    candidate_observation_semantic_sha256: Sha256Digest
    status: Literal["SHADOW_ONLY"] = "SHADOW_ONLY"


class SmallCameraEncoder(Protocol):
    """Provider-neutral seam for a future lightweight per-camera encoder."""

    @property
    def encoder_id(self) -> str: ...

    @property
    def encoder_revision(self) -> str: ...

    def encode(
        self,
        *,
        context: PerceptionContextManifest,
        camera_input: CameraEncoderInput,
    ) -> SmallEncoderObservation: ...


class SmallEncoderShadowSink(Protocol):
    """Optional sink for candidate observations and native-authority lineage."""

    def record(self, comparison: SmallEncoderShadowComparison) -> None: ...


def small_encoder_observation_semantic_sha256(value: SmallEncoderObservation) -> str:
    """Hash candidate content without allowing the hash field to self-reference."""

    return semantic_sha256(value.model_dump(mode="json", exclude={"semantic_sha256"}))


@dataclass(frozen=True, slots=True)
class SingleCameraAuthority:
    """Runtime binding for the current low-concurrency qualification profile."""

    policy: SingleCameraAuthorityPolicy

    def as_projection(self) -> dict[str, object]:
        return {
            "policy_version": self.policy.policy_version,
            "camera_id": self.policy.camera_id.value,
            "authority_provider": self.policy.authority_provider,
            "shadow_encoder_mode": self.policy.shadow_encoder_mode.value,
            "worker_count": self.policy.worker_count,
            "generation_concurrency": self.policy.generation_concurrency,
            "max_inflight_observations": self.policy.max_inflight_observations,
            "raw_refine_provider": self.policy.raw_refine_provider,
        }


__all__ = [
    "SINGLE_CAMERA_AUTHORITY_POLICY_VERSION",
    "SMALL_ENCODER_OBSERVATION_VERSION",
    "SMALL_ENCODER_SHADOW_COMPARISON_VERSION",
    "CameraEncoderInput",
    "SingleCameraAuthority",
    "SingleCameraAuthorityPolicy",
    "SmallCameraEncoder",
    "SmallEncoderActionCandidate",
    "SmallEncoderObservation",
    "SmallEncoderQuality",
    "SmallEncoderShadowComparison",
    "SmallEncoderShadowMode",
    "SmallEncoderShadowSink",
    "small_encoder_observation_semantic_sha256",
]
