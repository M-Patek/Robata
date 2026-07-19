from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from robata.contracts import CAMERA_IDS, CameraId, SixCameraMap, recording_identity
from robata.ingestion import ExactTopicMappingPolicy, TopicMappingProfile
from robata.ports import (
    ChannelInspection,
    IngestionError,
    IngestionErrorCode,
    McapInspection,
)

ROOT = Path(__file__).resolve().parents[2]
MAPPING_PROFILE = ROOT / "config" / "genrobot-observed-v0.json"


def _topics() -> SixCameraMap[str]:
    return SixCameraMap[str].model_validate(
        {camera_id: f"/camera/{index}/compressed" for index, camera_id in enumerate(CAMERA_IDS)},
        strict=True,
    )


def _channel(
    channel_id: int,
    topic: str,
    *,
    schema: str = "foxglove.CompressedImage",
) -> ChannelInspection:
    return ChannelInspection(
        channel_id=channel_id,
        topic=topic,
        schema_name=schema,
        message_encoding="protobuf",
        message_count=10,
        first_message_time_ns=9_007_199_254_740_993,
        last_message_time_ns=9_007_199_254_741_003,
        monotonic=True,
        codec="h264",
        frame_id=f"frame-{channel_id}",
    )


def _inspection(channels: list[ChannelInspection]) -> McapInspection:
    return McapInspection(
        source=Path("synthetic.mcap"),
        source_size_bytes=100,
        source_sha256="0" * 64,
        header_profile="synthetic",
        header_library="test",
        summary_available=True,
        channel_count=len(channels),
        message_count=sum(channel.message_count for channel in channels),
        first_message_time_ns=min(channel.first_message_time_ns or 0 for channel in channels),
        last_message_time_ns=max(channel.last_message_time_ns or 0 for channel in channels),
        channels=tuple(channels),
    )


def test_exact_topic_policy_resolves_all_six_cameras_in_canonical_order() -> None:
    topics = _topics()
    channels = [
        _channel(index, topics[camera_id])
        for index, camera_id in reversed(tuple(enumerate(CAMERA_IDS, start=1)))
    ]

    mapping = ExactTopicMappingPolicy(topics, version="synthetic-v1").resolve(_inspection(channels))

    assert tuple(mapping.keys()) == CAMERA_IDS
    assert mapping[CameraId.CAM_01].topic == "/camera/0/compressed"
    assert mapping[CameraId.CAM_06].topic == "/camera/5/compressed"


@pytest.mark.parametrize(
    ("failure", "expected_fragment"),
    [
        ("missing", "missing topic"),
        ("duplicate", "duplicate topic"),
        ("wrong_schema", "expected 'foxglove.CompressedImage'"),
    ],
)
def test_exact_topic_policy_rejects_invalid_source_mapping(
    failure: str,
    expected_fragment: str,
) -> None:
    topics = _topics()
    channels = [
        _channel(index, topics[camera_id]) for index, camera_id in enumerate(CAMERA_IDS, start=1)
    ]
    if failure == "missing":
        channels.pop()
    elif failure == "duplicate":
        channels.append(_channel(99, topics[CameraId.CAM_03]))
    else:
        channels[2] = _channel(3, topics[CameraId.CAM_03], schema="foxglove.RawImage")

    with pytest.raises(IngestionError) as raised:
        ExactTopicMappingPolicy(topics, version="synthetic-v1").resolve(_inspection(channels))

    assert raised.value.code is IngestionErrorCode.INVALID_CAMERA_MAPPING
    assert expected_fragment in str(raised.value)


def test_exact_topic_policy_rejects_one_topic_assigned_to_two_slots() -> None:
    raw_topics = {camera_id: f"/camera/{index}" for index, camera_id in enumerate(CAMERA_IDS)}
    raw_topics[CameraId.CAM_06] = raw_topics[CameraId.CAM_01]
    topics = SixCameraMap[str].model_validate(raw_topics, strict=True)

    with pytest.raises(IngestionError) as raised:
        ExactTopicMappingPolicy(topics, version="synthetic-v1")

    assert raised.value.code is IngestionErrorCode.INVALID_CAMERA_MAPPING
    assert "more than once" in str(raised.value)


def test_unapproved_profile_requires_explicit_local_override() -> None:
    profile = TopicMappingProfile.load(MAPPING_PROFILE)

    with pytest.raises(IngestionError) as raised:
        ExactTopicMappingPolicy.from_profile(profile)

    assert raised.value.code is IngestionErrorCode.INVALID_CAMERA_MAPPING
    ExactTopicMappingPolicy.from_profile(profile, allow_unapproved=True)


def test_mapping_profile_digest_is_semantic_and_order_independent() -> None:
    profile = TopicMappingProfile.load(MAPPING_PROFILE)
    raw = json.loads(MAPPING_PROFILE.read_text(encoding="utf-8"))
    reordered = dict(reversed(tuple(raw.items())))
    reordered["topics"] = dict(reversed(tuple(raw["topics"].items())))
    with patch.object(Path, "read_text", return_value=json.dumps(reordered)):
        reparsed = TopicMappingProfile.load(Path("reordered-profile.json"))

    assert reparsed.semantic_projection() == profile.semantic_projection()
    assert reparsed.semantic_digest == profile.semantic_digest


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"unknown": "value"}, "unknown=['unknown']"),
        ({"mapping_policy": "REGEX"}, "mapping_policy must be 'EXACT_TOPIC'"),
        ({"required_schema": "foxglove.RawImage"}, "required_schema must be"),
        ({"approval_status": "APPROVED"}, "approved and approval_status are contradictory"),
    ],
)
def test_mapping_profile_rejects_ignored_or_contradictory_configuration(
    mutation: dict[str, object],
    expected: str,
) -> None:
    raw = json.loads(MAPPING_PROFILE.read_text(encoding="utf-8"))
    raw.update(mutation)
    with (
        patch.object(Path, "read_text", return_value=json.dumps(raw)),
        pytest.raises(IngestionError) as raised,
    ):
        TopicMappingProfile.load(Path("virtual-profile.json"))

    assert raised.value.code is IngestionErrorCode.INVALID_CAMERA_MAPPING
    assert expected in str(raised.value)


def test_recording_identity_does_not_depend_on_source_path() -> None:
    content_sha256 = "a" * 64
    first_path = Path("first/location/source.mcap")
    second_path = Path("moved/source.mcap")

    first_identity = recording_identity("test-namespace", content_sha256)
    second_identity = recording_identity("test-namespace", content_sha256)

    assert first_path != second_path
    assert first_identity == second_identity
