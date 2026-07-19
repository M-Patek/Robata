"""Exact, versioned topic-to-camera mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robata.contracts import (
    CAMERA_IDS,
    CameraId,
    Sha256Digest,
    SixCameraMap,
    semantic_sha256,
)
from robata.ports import (
    COMPRESSED_IMAGE_SCHEMA,
    ChannelInspection,
    IngestionError,
    IngestionErrorCode,
    McapInspection,
)


@dataclass(frozen=True, slots=True)
class TopicMappingProfile:
    """A checked-in mapping candidate and its approval state."""

    profile_id: str
    version: str
    profile_kind: str
    approval_status: str
    approved: bool
    mapping_policy: str
    required_schema: str
    topics: SixCameraMap[str]

    def __post_init__(self) -> None:
        if self.mapping_policy != "EXACT_TOPIC":
            raise ValueError("mapping_policy must be 'EXACT_TOPIC'")
        if self.required_schema != COMPRESSED_IMAGE_SCHEMA:
            raise ValueError(f"required_schema must be {COMPRESSED_IMAGE_SCHEMA!r}")
        if self.approved != (self.approval_status == "APPROVED"):
            raise ValueError("approved and approval_status are contradictory")

    def semantic_projection(self) -> dict[str, Any]:
        """Return every policy field that materially defines this mapping profile."""

        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "profile_kind": self.profile_kind,
            "approval_status": self.approval_status,
            "approved": self.approved,
            "mapping_policy": self.mapping_policy,
            "required_schema": self.required_schema,
            "topics": self.topics.model_dump(mode="json"),
        }

    @property
    def semantic_digest(self) -> Sha256Digest:
        """Run-independent digest of the parsed mapping semantics."""

        return semantic_sha256(self.semantic_projection())

    @classmethod
    def load(cls, path: Path) -> TopicMappingProfile:
        try:
            raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("mapping profile must be a JSON object")
            expected_fields = {
                "profile_id",
                "version",
                "profile_kind",
                "approval_status",
                "approved",
                "mapping_policy",
                "required_schema",
                "topics",
            }
            actual_fields = set(raw)
            if actual_fields != expected_fields:
                missing = sorted(expected_fields - actual_fields)
                unknown = sorted(actual_fields - expected_fields)
                raise ValueError(
                    f"mapping profile fields differ: missing={missing!r}, unknown={unknown!r}"
                )
            topics = SixCameraMap[str].model_validate(raw["topics"], strict=True)
            return cls(
                profile_id=_required_string(raw, "profile_id"),
                version=_required_string(raw, "version"),
                profile_kind=_required_string(raw, "profile_kind"),
                approval_status=_required_string(raw, "approval_status"),
                approved=_required_bool(raw, "approved"),
                mapping_policy=_required_string(raw, "mapping_policy"),
                required_schema=_required_string(raw, "required_schema"),
                topics=topics,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise IngestionError(
                IngestionErrorCode.INVALID_CAMERA_MAPPING,
                f"invalid topic mapping profile {path}: {exc}",
            ) from exc


class ExactTopicMappingPolicy:
    """Resolve each canonical camera from one configured topic, with no guessing."""

    def __init__(self, topics: SixCameraMap[str], *, version: str) -> None:
        self._topics = topics
        self.version = version
        duplicate_topics = _duplicates(tuple(topics.values()))
        if duplicate_topics:
            raise IngestionError(
                IngestionErrorCode.INVALID_CAMERA_MAPPING,
                f"mapping profile assigns topics more than once: {duplicate_topics!r}",
            )

    @classmethod
    def from_profile(
        cls,
        profile: TopicMappingProfile,
        *,
        allow_unapproved: bool = False,
    ) -> ExactTopicMappingPolicy:
        if not profile.approved and not allow_unapproved:
            raise IngestionError(
                IngestionErrorCode.INVALID_CAMERA_MAPPING,
                f"mapping profile {profile.profile_id!r} is not approved",
            )
        return cls(profile.topics, version=profile.version)

    def resolve(self, inspection: McapInspection) -> SixCameraMap[ChannelInspection]:
        channels_by_topic: dict[str, list[ChannelInspection]] = {}
        for channel in inspection.channels:
            channels_by_topic.setdefault(channel.topic, []).append(channel)

        resolved: dict[CameraId, ChannelInspection] = {}
        errors: list[str] = []
        for camera_id in CAMERA_IDS:
            topic = self._topics[camera_id]
            matches = channels_by_topic.get(topic, [])
            if not matches:
                errors.append(f"{camera_id.value}: missing topic {topic!r}")
                continue
            if len(matches) != 1:
                channel_ids = sorted(channel.channel_id for channel in matches)
                errors.append(
                    f"{camera_id.value}: duplicate topic {topic!r} on channels {channel_ids!r}"
                )
                continue
            channel = matches[0]
            if channel.schema_name != COMPRESSED_IMAGE_SCHEMA:
                errors.append(
                    f"{camera_id.value}: topic {topic!r} has schema {channel.schema_name!r}, "
                    f"expected {COMPRESSED_IMAGE_SCHEMA!r}"
                )
                continue
            resolved[camera_id] = channel

        if errors:
            raise IngestionError(
                IngestionErrorCode.INVALID_CAMERA_MAPPING,
                "; ".join(errors),
            )
        return SixCameraMap[ChannelInspection].model_validate(resolved, strict=True)

    def map(self, inspection: McapInspection) -> SixCameraMap[ChannelInspection]:
        """Alias for callers that use mapping-policy terminology."""

        return self.resolve(inspection)


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _required_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
