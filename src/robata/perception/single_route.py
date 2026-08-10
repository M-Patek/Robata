"""Pure single-camera Mage authority for the local qualification route.

The experimental small-encoder shadow seam was removed after it failed to justify
its additional complexity.  The current policy keeps one native Mage authority,
one preparation worker, and one generation lane.  The serialized v1 projection is
preserved so existing local evidence remains byte-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from robata.contracts.cameras import CameraId
from robata.contracts.common import StrictModel

SINGLE_CAMERA_AUTHORITY_POLICY_VERSION: Final = "single-camera-mage-authority-v1"


class SingleCameraAuthorityPolicy(StrictModel):
    """Explicit pure-Mage execution policy for the first local route."""

    policy_version: Literal["single-camera-mage-authority-v1"] = (
        SINGLE_CAMERA_AUTHORITY_POLICY_VERSION
    )
    camera_id: CameraId = CameraId.CAM_01
    authority_provider: Literal["MAGE_NATIVE"] = "MAGE_NATIVE"
    shadow_encoder_mode: Literal["DISABLED"] = "DISABLED"
    worker_count: Literal[1] = 1
    generation_concurrency: Literal[1] = 1
    max_inflight_observations: Literal[1, 2] = 1
    raw_refine_provider: Literal["MAGE_NATIVE"] = "MAGE_NATIVE"


@dataclass(frozen=True, slots=True)
class SingleCameraAuthority:
    """Runtime binding for the current low-concurrency qualification profile."""

    policy: SingleCameraAuthorityPolicy

    def as_projection(self) -> dict[str, object]:
        return {
            "policy_version": self.policy.policy_version,
            "camera_id": self.policy.camera_id.value,
            "authority_provider": self.policy.authority_provider,
            "shadow_encoder_mode": self.policy.shadow_encoder_mode,
            "worker_count": self.policy.worker_count,
            "generation_concurrency": self.policy.generation_concurrency,
            "max_inflight_observations": self.policy.max_inflight_observations,
            "raw_refine_provider": self.policy.raw_refine_provider,
        }


__all__ = [
    "SINGLE_CAMERA_AUTHORITY_POLICY_VERSION",
    "SingleCameraAuthority",
    "SingleCameraAuthorityPolicy",
]
