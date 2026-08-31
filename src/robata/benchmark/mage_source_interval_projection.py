"""Project P41 Mage bounded clips into label-blind source-interval rows.

This benchmark-only module deliberately has no dependency on the Mage runtime,
cache verifier, media decoder, or post-hoc evaluation code.  It copies only the
small source-location fields needed to decide whether a Mage bounded clip could
be mechanically paired with another model's source interval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

MAGE_SOURCE_INTERVAL_PROJECTION_VERSION: Final = "mage-source-interval-projection-v1"


class MageSourceIntervalProjectionError(ValueError):
    """A supplied label-blind bounded-case pack cannot be projected."""


def build_mage_source_interval_projection(
    input_pack: Mapping[str, object],
) -> dict[str, object]:
    """Return a fresh, source-only projection of P41 bounded-case records.

    The projection reads only ``records[].uid``, optional explicit source-video
    fields, and ``records[].bounded_clip`` location/interval fields.  It never
    copies any other part of an input record.  In particular, missing source
    identity or source-media path is represented as unavailable rather than
    inferred from a UID or reconstructed from media.
    """

    if not isinstance(input_pack, Mapping):
        raise MageSourceIntervalProjectionError("input pack must be an object")
    records_value = input_pack.get("records")
    if not isinstance(records_value, list):
        raise MageSourceIntervalProjectionError("input pack records must be an array")
    records: list[object] = records_value
    if not records:
        raise MageSourceIntervalProjectionError("input pack records must be nonempty")
    _validate_record_count(input_pack.get("record_count"), len(records))

    projected_records: list[dict[str, object]] = []
    seen_uids: set[str] = set()
    for ordinal, value in enumerate(records):
        if not isinstance(value, Mapping):
            raise MageSourceIntervalProjectionError(f"records[{ordinal}] must be an object")
        projected = _project_bounded_case(value, ordinal=ordinal)
        uid = str(projected["uid"])
        if uid in seen_uids:
            raise MageSourceIntervalProjectionError("input pack contains duplicate UIDs")
        seen_uids.add(uid)
        projected_records.append(projected)

    return {
        "projection_version": MAGE_SOURCE_INTERVAL_PROJECTION_VERSION,
        "scope": "P41 Mage bounded-case source interval only",
        "execution": {
            "models_loaded": False,
            "gpu_called": False,
            "media_decoded": False,
            "inference_performed": False,
        },
        "record_count": len(projected_records),
        "records": projected_records,
        "mechanical_overlap_readiness": _mechanical_overlap_readiness(projected_records),
    }


def _project_bounded_case(record: Mapping[str, object], *, ordinal: int) -> dict[str, object]:
    uid = _required_text(record.get("uid"), field_name=f"records[{ordinal}].uid")
    source_video = record.get("source_video")
    source_video_mapping = source_video if isinstance(source_video, Mapping) else None

    source_video_id, source_video_id_reason = _first_text(
        (
            (
                source_video_mapping.get("identity") if source_video_mapping is not None else None,
                "source_video.identity",
            ),
            (
                source_video_mapping.get("id") if source_video_mapping is not None else None,
                "source_video.id",
            ),
            (record.get("source_video_id"), "source_video_id"),
            (record.get("video_id"), "video_id"),
        ),
        missing_reason="not present in the label-blind Mage bounded-case record",
    )
    source_video_path, source_video_path_reason = _first_text(
        (
            (
                source_video_mapping.get("path") if source_video_mapping is not None else None,
                "source_video.path",
            ),
            (record.get("source_video_path"), "source_video_path"),
            (record.get("video_path"), "video_path"),
        ),
        missing_reason="not present in the label-blind Mage bounded-case record",
    )

    bounded_clip = record.get("bounded_clip")
    bounded_clip_mapping = bounded_clip if isinstance(bounded_clip, Mapping) else None
    clip_path, clip_path_reason = _text_from_mapping(
        bounded_clip_mapping,
        key="path",
        missing_reason="bounded_clip.path is not present",
    )
    clip_locator, clip_locator_reason = _text_from_mapping(
        bounded_clip_mapping,
        key="locator",
        missing_reason="bounded_clip.locator is not present",
    )
    if clip_locator is None and clip_path is not None:
        clip_locator = clip_path
        clip_locator_reason = "bounded_clip.path is the available locator"

    start_seconds, start_reason = _number_from_mapping(
        bounded_clip_mapping,
        key="start_seconds",
        missing_reason="bounded_clip.start_seconds is not present or valid",
    )
    end_seconds, end_reason = _number_from_mapping(
        bounded_clip_mapping,
        key="end_seconds",
        missing_reason="bounded_clip.end_seconds is not present or valid",
    )
    if start_seconds is not None and end_seconds is not None and end_seconds <= start_seconds:
        start_seconds = None
        end_seconds = None
        start_reason = "bounded_clip interval is not positive"
        end_reason = "bounded_clip interval is not positive"

    return {
        "uid": uid,
        "source_video_id": source_video_id,
        "source_video_id_unavailable": source_video_id is None,
        "source_video_id_reason": source_video_id_reason,
        "source_video_path": source_video_path,
        "source_video_path_unavailable": source_video_path is None,
        "source_video_path_reason": source_video_path_reason,
        "source_interval_start_seconds": start_seconds,
        "source_interval_start_seconds_unavailable": start_seconds is None,
        "source_interval_start_seconds_reason": start_reason,
        "source_interval_end_seconds": end_seconds,
        "source_interval_end_seconds_unavailable": end_seconds is None,
        "source_interval_end_seconds_reason": end_reason,
        "bounded_clip_locator": clip_locator,
        "bounded_clip_locator_unavailable": clip_locator is None,
        "bounded_clip_locator_reason": clip_locator_reason,
        "bounded_clip_path": clip_path,
        "bounded_clip_path_unavailable": clip_path is None,
        "bounded_clip_path_reason": clip_path_reason,
    }


def _mechanical_overlap_readiness(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    incomplete: list[dict[str, object]] = []
    for record in records:
        missing = [
            field
            for field in (
                "source_video_id",
                "source_video_path",
                "source_interval_start_seconds",
                "source_interval_end_seconds",
                "bounded_clip_locator",
            )
            if record[f"{field}_unavailable"] is True
        ]
        if missing:
            incomplete.append({"uid": record["uid"], "unavailable_fields": missing})

    if incomplete:
        return {
            "status": "MAGE_SOURCE_BINDING_INCOMPLETE",
            "reason": "one or more bounded cases lack source binding fields",
            "incomplete_cases": incomplete,
        }
    return {
        "status": "READY_FOR_EXTERNAL_QWEN_INTERVAL_INTERSECTION",
        "reason": "Mage source identity/path, interval, and bounded clip are explicit",
        "incomplete_cases": [],
    }


def _first_text(
    candidates: Sequence[tuple[object, str]], *, missing_reason: str
) -> tuple[str | None, str | None]:
    invalid_fields: list[str] = []
    for value, field_name in candidates:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip(), f"explicit {field_name}"
        invalid_fields.append(field_name)
    if invalid_fields:
        return None, f"{', '.join(invalid_fields)} is not nonempty text"
    return None, missing_reason


def _text_from_mapping(
    value: Mapping[str, object] | None,
    *,
    key: str,
    missing_reason: str,
) -> tuple[str | None, str | None]:
    if value is None:
        return None, "bounded_clip is not an object"
    return _first_text(((value.get(key), f"bounded_clip.{key}"),), missing_reason=missing_reason)


def _number_from_mapping(
    value: Mapping[str, object] | None,
    *,
    key: str,
    missing_reason: str,
) -> tuple[float | None, str | None]:
    if value is None:
        return None, "bounded_clip is not an object"
    number_value = value.get(key)
    if isinstance(number_value, bool) or not isinstance(number_value, (int, float)):
        return None, missing_reason
    number = float(number_value)
    if not math.isfinite(number) or number < 0:
        return None, missing_reason
    return number, f"explicit bounded_clip.{key}"


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MageSourceIntervalProjectionError(f"{field_name} must be nonempty text")
    return value.strip()


def _validate_record_count(value: object, actual_count: int) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value != actual_count:
        raise MageSourceIntervalProjectionError("input pack record_count does not match records")


__all__ = [
    "MAGE_SOURCE_INTERVAL_PROJECTION_VERSION",
    "MageSourceIntervalProjectionError",
    "build_mage_source_interval_projection",
]
