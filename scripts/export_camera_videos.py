"""Export one explicitly authorized local MCAP to six immutable MP4 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robata.adapters import (
    EXPORT_CONFIG,
    EXPORT_PROFILE_ID,
    EXPORT_PROFILE_VERSION,
    EXPORTER_NAME,
    EXPORTER_VERSION,
    LocalArtifactRegistry,
    OfficialMcapInspector,
    PyAvH264Mp4Exporter,
)
from robata.application import (
    LocalVideoExportRequest,
    RegisteredSixCameraVideoExportService,
    VideoExporterDescriptor,
    VideoExportRunError,
)
from robata.contracts import SchemaRegistryError, semantic_sha256
from robata.contracts.video_export import VideoExporterMode
from robata.ingestion import ExactTopicMappingPolicy, TopicMappingProfile
from robata.ports import ArtifactRegistryError, IngestionError, VideoExportError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_CONFIG = REPOSITORY_ROOT / "config" / "genrobot-observed-v0.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remux six exactly mapped local H.264 MCAP channels into a complete "
            "provider-neutral derived-artifact directory."
        )
    )
    parser.add_argument("source", type=Path, help="local MCAP source")
    parser.add_argument("output", type=Path, help="new immutable output directory")
    parser.add_argument(
        "--mapping-config",
        type=Path,
        default=DEFAULT_MAPPING_CONFIG,
        help="exact topic mapping profile",
    )
    parser.add_argument(
        "--namespace",
        default="robata",
        help="recording-identity namespace; source paths are excluded from identity",
    )
    parser.add_argument(
        "--allow-unapproved-profile",
        action="store_true",
        help="explicitly permit the local unapproved mapping development override",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help="shared local artifact registry; defaults beside the output directory",
    )
    return parser


def _error_payload(error: Exception) -> dict[str, Any]:
    code = getattr(error, "code", "UNEXPECTED_ERROR")
    code_value = getattr(code, "value", code)
    return {
        "ok": False,
        "error": {
            "code": str(code_value),
            "message": str(error),
        },
        "provider_requests": 0,
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        profile = TopicMappingProfile.load(args.mapping_config)
        # This authorization check intentionally precedes all source access.
        mapping_policy = ExactTopicMappingPolicy.from_profile(
            profile,
            allow_unapproved=args.allow_unapproved_profile,
        )
        inspection = OfficialMcapInspector().inspect(args.source)
        channels = mapping_policy.resolve(inspection)
        descriptor = VideoExporterDescriptor(
            name=EXPORTER_NAME,
            version=EXPORTER_VERSION,
            mode=VideoExporterMode.REMUX,
            export_profile_id=EXPORT_PROFILE_ID,
            profile_version=EXPORT_PROFILE_VERSION,
            canonical_config_sha256=semantic_sha256(EXPORT_CONFIG),
        )
        registry_root = (
            args.registry_root
            if args.registry_root is not None
            else args.output.resolve().parent / ".robata-artifacts"
        )
        result = RegisteredSixCameraVideoExportService(
            PyAvH264Mp4Exporter(),
            LocalArtifactRegistry(registry_root),
        ).export_local(
            LocalVideoExportRequest(
                source=args.source,
                output_directory=args.output,
                namespace=args.namespace,
                inspection=inspection,
                channels=channels,
                mapping_profile=profile,
                mapping_profile_digest=profile.semantic_digest,
                exporter=descriptor,
            )
        )
        payload = {
            "ok": True,
            "execution_mode": result.manifest.execution_mode.value,
            "alignment_status": result.manifest.alignment_status.value,
            "mapping_approved": result.manifest.mapping_profile.approved,
            "ready_manifest_id": result.manifest.ready_manifest_id,
            "recording_identity": result.manifest.recording_identity,
            "source_content_sha256": result.manifest.source_content_sha256,
            "output_directory": str(result.output_directory),
            "schema_version": result.manifest.schema_version,
            "manifest_sha256": result.manifest_sha256,
            "manifest_artifact_id": result.manifest_artifact_id,
            "logical_key": result.logical_key,
            "reused": result.reused,
            "derivation_reused": result.derivation_reused,
            "materialized_view_reused": result.materialized_view_reused,
            "registry_root": str(registry_root.resolve()),
            "provider_requests": 0,
            "cameras": [
                {
                    "camera_id": record.camera_id.value,
                    "video_sha256": record.video_artifact.sha256,
                    "video_bytes": record.video_artifact.bytes,
                    "timestamp_sidecar_sha256": (record.timestamp_sidecar_artifact.artifact.sha256),
                    "input_message_count": record.input_message_count,
                    "exported_packet_count": record.exported_packet_count,
                    "exported_frame_count": record.exported_frame_count,
                    "leading_dropped_message_count": record.leading_drops.count,
                    "trailing_dropped_message_count": record.trailing_drops.count,
                    "keyframe_count": record.keyframe_count,
                    "width": record.width,
                    "height": record.height,
                }
                for record in result.manifest.cameras
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        ArtifactRegistryError,
        IngestionError,
        SchemaRegistryError,
        VideoExportError,
        VideoExportRunError,
        ValueError,
    ) as error:
        print(json.dumps(_error_payload(error), indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
