"""Inspect, exactly map, and decoder-probe one local MCAP source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.adapters import OfficialMcapInspector, PyAvH264DecoderProbe  # noqa: E402
from robata.contracts import CAMERA_IDS, recording_identity  # noqa: E402
from robata.ingestion import ExactTopicMappingPolicy, TopicMappingProfile  # noqa: E402
from robata.ports import (  # noqa: E402
    ChannelInspection,
    DecodeFailure,
    DecoderProbeResult,
    IngestionError,
    IngestionErrorCode,
    McapInspection,
)

DEFAULT_MAPPING_CONFIG = REPOSITORY_ROOT / "config" / "genrobot-observed-v0.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect MCAP structure, apply an exact six-camera map, and probe H.264."
    )
    parser.add_argument("source", type=Path, help="local MCAP file")
    parser.add_argument(
        "--mapping-config",
        type=Path,
        default=DEFAULT_MAPPING_CONFIG,
        help="exact topic mapping profile",
    )
    parser.add_argument(
        "--namespace",
        default="robata",
        help="recording-identity namespace (source URI/path is never in the preimage)",
    )
    parser.add_argument(
        "--allow-unapproved-profile",
        action="store_true",
        help="explicitly allow a local observed profile that is not approved for admission",
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="omit decoder probes while retaining inspection and mapping",
    )
    return parser


def _encoded_ns(value: int | None) -> str | None:
    return str(value) if value is not None else None


def _channel_json(channel: ChannelInspection) -> dict[str, Any]:
    return {
        "channel_id": channel.channel_id,
        "topic": channel.topic,
        "schema": channel.schema_name,
        "message_encoding": channel.message_encoding,
        "count": channel.message_count,
        "first_message_time_ns": _encoded_ns(channel.first_message_time_ns),
        "last_message_time_ns": _encoded_ns(channel.last_message_time_ns),
        "monotonic": channel.monotonic,
        "codec": channel.codec,
        "frame_id": channel.frame_id,
    }


def _inspection_json(inspection: McapInspection) -> dict[str, Any]:
    return {
        "source": str(inspection.source),
        "source_size_bytes": inspection.source_size_bytes,
        "source_sha256": inspection.source_sha256,
        "header": {
            "profile": inspection.header_profile,
            "library": inspection.header_library,
        },
        "summary_available": inspection.summary_available,
        "channel_count": inspection.channel_count,
        "message_count": inspection.message_count,
        "first_message_time_ns": _encoded_ns(inspection.first_message_time_ns),
        "last_message_time_ns": _encoded_ns(inspection.last_message_time_ns),
        "channels": [_channel_json(channel) for channel in inspection.channels],
    }


def _failure_json(failure: DecodeFailure) -> dict[str, Any]:
    return {
        "code": failure.code,
        "timestamp_ns": _encoded_ns(failure.timestamp_ns),
        "message": failure.message,
    }


def _probe_json(result: DecoderProbeResult) -> dict[str, Any]:
    return {
        "topic": result.topic,
        "codec": result.codec,
        "success": result.success,
        "width": result.width,
        "height": result.height,
        "first_decoded_timestamp_ns": _encoded_ns(result.first_decoded_timestamp_ns),
        "messages_examined": result.messages_examined,
        "decoded_frames": result.decoded_frames,
        "failure_count": result.failure_count,
        "failures": [_failure_json(failure) for failure in result.failures],
    }


def _error_json(error: IngestionError) -> dict[str, Any]:
    return {
        "ok": False,
        "provider_requests": 0,
        "error": {"code": error.code.value, "message": str(error)},
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        profile = TopicMappingProfile.load(args.mapping_config)
        mapping_policy = ExactTopicMappingPolicy.from_profile(
            profile,
            allow_unapproved=args.allow_unapproved_profile,
        )
        inspection = OfficialMcapInspector().inspect(args.source)
        mapping = mapping_policy.resolve(inspection)
        probes: dict[str, DecoderProbeResult] = {}
        if not args.no_decode:
            decoder_probe = PyAvH264DecoderProbe()
            probes = {
                camera_id.value: decoder_probe.probe(args.source, mapping[camera_id])
                for camera_id in CAMERA_IDS
            }
        all_decoded = all(result.success for result in probes.values())
        payload: dict[str, Any] = {
            "ok": all_decoded or args.no_decode,
            "provider_requests": 0,
            "namespace": args.namespace,
            "recording_identity": recording_identity(
                args.namespace,
                inspection.source_sha256,
            ),
            "mapping_profile": {
                "profile_id": profile.profile_id,
                "version": profile.version,
                "profile_kind": profile.profile_kind,
                "approval_status": profile.approval_status,
                "approved": profile.approved,
                "mapping_policy": profile.mapping_policy,
                "required_schema": profile.required_schema,
                "unapproved_override": bool(args.allow_unapproved_profile and not profile.approved),
            },
            "inspection": _inspection_json(inspection),
            "camera_mapping": {
                camera_id.value: _channel_json(mapping[camera_id]) for camera_id in CAMERA_IDS
            },
            "decoder_probes": {
                camera_id.value: _probe_json(probes[camera_id.value])
                for camera_id in CAMERA_IDS
                if camera_id.value in probes
            },
        }
        if probes and not all_decoded:
            payload["error"] = {
                "code": IngestionErrorCode.DECODER_PROBE_FAILED.value,
                "message": "one or more mapped camera streams produced no decoded frame",
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["ok"] else 2
    except IngestionError as error:
        print(json.dumps(_error_json(error), indent=2, sort_keys=True))
        return 2
    except ValueError as error:
        wrapped = IngestionError(IngestionErrorCode.SOURCE_IO_ERROR, str(error))
        print(json.dumps(_error_json(wrapped), indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
