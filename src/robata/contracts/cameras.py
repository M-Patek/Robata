"""Native six-camera identifiers and exact-cardinality collections."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self

from pydantic import ConfigDict, RootModel, field_serializer, model_validator


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


class SixCameraMap[T](RootModel[Mapping[CameraId, T]]):
    """A mapping containing every canonical camera exactly once.

    Input order is intentionally ignored. The stored and serialized order is always
    ``cam_01`` through ``cam_06`` so downstream projections remain deterministic.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def validate_and_order_keys(cls, value: Any) -> dict[CameraId, Any]:
        if isinstance(value, SixCameraMap):
            value = value.root
        if not isinstance(value, Mapping):
            raise ValueError("SixCameraMap must be an object mapping camera IDs to values")

        normalized: dict[CameraId, Any] = {}
        for key, item in value.items():
            if isinstance(key, CameraId):
                camera_id = key
            elif type(key) is str:
                try:
                    camera_id = CameraId(key)
                except ValueError as exc:
                    raise ValueError(f"unknown camera ID: {key!r}") from exc
            else:
                raise ValueError("camera IDs must be canonical strings")
            if camera_id in normalized:
                raise ValueError(f"duplicate camera ID: {camera_id.value}")
            normalized[camera_id] = item

        actual = frozenset(normalized)
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

        return {camera_id: normalized[camera_id] for camera_id in CAMERA_IDS}

    @model_validator(mode="after")
    def freeze_mapping(self) -> Self:
        object.__setattr__(self, "root", MappingProxyType(dict(self.root)))
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
