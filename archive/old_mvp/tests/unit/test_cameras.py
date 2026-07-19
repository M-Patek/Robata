import json

import pytest
from pydantic import ValidationError

from robata.contracts.cameras import (
    CAMERA_ID_VALUES,
    CAMERA_IDS,
    CameraId,
    SixCameraMap,
)


def camera_payload() -> dict[str, int]:
    return {camera_id.value: index for index, camera_id in enumerate(reversed(CAMERA_IDS))}


def test_camera_ids_are_exact_and_canonical() -> None:
    assert CAMERA_ID_VALUES == (
        "cam_01",
        "cam_02",
        "cam_03",
        "cam_04",
        "cam_05",
        "cam_06",
    )
    with pytest.raises(ValueError):
        CameraId("cam_1")


def test_six_camera_map_accepts_all_keys_and_canonicalizes_order() -> None:
    cameras = SixCameraMap[int].model_validate(camera_payload())

    assert tuple(cameras.keys()) == CAMERA_IDS
    assert cameras[CameraId.CAM_06] == 0
    assert list(json.loads(cameras.model_dump_json())) == list(CAMERA_ID_VALUES)


def test_six_camera_map_parses_json_object() -> None:
    payload = json.dumps(camera_payload())

    cameras = SixCameraMap[int].model_validate_json(payload)

    assert tuple(cameras.keys()) == CAMERA_IDS
    assert json.loads(cameras.model_dump_json())["cam_01"] == 5


@pytest.mark.parametrize("removed", list(CameraId))
def test_six_camera_map_rejects_every_missing_slot(removed: CameraId) -> None:
    payload = camera_payload()
    del payload[removed.value]

    with pytest.raises(ValidationError, match="exactly the canonical camera IDs"):
        SixCameraMap[int].model_validate(payload)


def test_six_camera_map_rejects_extra_or_malformed_keys() -> None:
    extra = camera_payload() | {"cam_07": 7}
    malformed = camera_payload()
    malformed["CAM_01"] = malformed.pop("cam_01")

    with pytest.raises(ValidationError, match="unknown camera ID"):
        SixCameraMap[int].model_validate(extra)
    with pytest.raises(ValidationError, match="unknown camera ID"):
        SixCameraMap[int].model_validate(malformed)


def test_six_camera_map_is_strict_about_values_and_frozen_at_model_boundary() -> None:
    payload: dict[str, object] = camera_payload()
    payload["cam_01"] = "5"

    with pytest.raises(ValidationError):
        SixCameraMap[int].model_validate(payload)

    cameras = SixCameraMap[int].model_validate(camera_payload())
    with pytest.raises(ValidationError):
        cameras.root = {}
    with pytest.raises(TypeError):
        cameras.root[CameraId.CAM_01] = 100  # type: ignore[index]
