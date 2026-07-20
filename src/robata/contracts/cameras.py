"""Native six-camera identifiers and exact-cardinality collections."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Self

from pydantic import BeforeValidator, ConfigDict, RootModel, field_serializer, model_validator


class CameraId(StrEnum):
    CAM_01 = "cam_01"
    CAM_02 = "cam_02"
    CAM_03 = "cam_03"
    CAM_04 = "cam_04"
    CAM_05 = "cam_05"
    CAM_06 = "cam_06"


CAMERA_IDS: tuple[CameraId, ...] = tuple(CameraId)
CAMERA_ID_VALUES: tuple[str, ...] = tuple(camera_id.value for camera_id in CAMERA_IDS)
_CAMERA_ID_SET = frozenset(CAMERA_IDS)


def _normalize_camera_id(value: Any) -> CameraId:
    if isinstance(value, CameraId):
        return value
    if type(value) is str:
        try:
            return CameraId(value)
        except ValueError as exc:
            raise ValueError(f"unknown camera ID: {value!r}") from exc
    raise ValueError("camera IDs must be canonical strings")


_CanonicalCameraId = Annotated[CameraId, BeforeValidator(_normalize_camera_id)]


class SixCameraMap[T](RootModel[Mapping[_CanonicalCameraId, T]]):
    """A mapping containing every canonical camera exactly once.

    Input order is intentionally ignored. The stored and serialized order is always
    ``cam_01`` through ``cam_06`` so downstream projections remain deterministic.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_and_freeze_mapping(self) -> Self:
        actual = frozenset(self.root)
        if actual != _CAMERA_ID_SET:
            missing = [camera_id.value for camera_id in CAMERA_IDS if camera_id not in actual]
            extra = sorted(camera_id.value for camera_id in actual - _CAMERA_ID_SET)
            details = []
            if missing:
                details.append(f"missing={missing!r}")
            if extra:
                details.append(f"extra={extra!r}")
            detail = ", ".join(details)
            raise ValueError(f"SixCameraMap requires exactly the canonical camera IDs: {detail}")

        ordered = {camera_id: self.root[camera_id] for camera_id in CAMERA_IDS}
        object.__setattr__(self, "root", MappingProxyType(ordered))
        return self

    @field_serializer("root")
    def serialize_mapping(self, value: Mapping[CameraId, Any]) -> dict[CameraId, Any]:
        return dict(value)

    def __getitem__(self, camera_id: CameraId) -> T:
        return self.root[camera_id]

    def __len__(self) -> int:
        return len(self.root)

    def keys(self) -> Any:
        return self.root.keys()

    def values(self) -> Any:
        return self.root.values()

    def items(self) -> Any:
        return self.root.items()
