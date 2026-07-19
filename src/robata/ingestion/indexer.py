"""Stream indexer for building frame-level indexes and resolving camera mappings."""

from __future__ import annotations

from typing import Any

from robata.ingestion.models import (
    CameraMapping,
    CameraMappingRun,
    IngestionResult,
    SourceFrameIndex,
    StreamIndex,
)


class StreamIndexer:
    """Build frame indexes and resolve topic-to-camera mappings for MCAP recordings.

    Responsibilities:
    - Build an immutable frame index per raw video stream.
    - Resolve versioned topic-to-camera mappings.
    - Validate stream properties and consistency.
    """

    def index_streams(
        self,
        mcap_id: str,
        source: Any,
    ) -> IngestionResult:
        """Build a frame index for all video streams in an MCAP recording.

        Args:
            mcap_id: The opaque UUIDv7 identifier for the MCAP recording.
            source: The source MCAP artifact to index.

        Returns:
            An ``IngestionResult`` containing stream indexes, camera mapping run,
            and frame-level source indexes.
        """
        raise NotImplementedError("index_streams() is not yet implemented")

    def resolve_camera_mapping(
        self,
        inspection: Any,
        mapping_run: CameraMappingRun,
    ) -> tuple[CameraMapping, ...]:
        """Resolve topic-to-camera mapping from an inspection and mapping run.

        Args:
            inspection: The MCAP inspection result containing observed channels.
            mapping_run: The versioned camera mapping run to apply.

        Returns:
            Exactly six ``CameraMapping`` rows, one per canonical camera slot
            ``cam_01`` through ``cam_06``.
        """
        raise NotImplementedError("resolve_camera_mapping() is not yet implemented")

    def validate_stream_consistency(
        self,
        stream_indexes: tuple[StreamIndex, ...],
    ) -> dict[str, Any]:
        """Validate that stream properties are consistent across all cameras.

        Checks include:
        - Exactly six streams are present.
        - No duplicate stream IDs or topic assignments.
        - All streams have supported codecs.
        - Timestamp ranges are positive and consistent.
        - Frame counts is non-negative and consistent with duration.

        Args:
            stream_indexes: The stream indexes to validate.

        Returns:
            A dictionary with validation results and any detected issues.
        """
        raise NotImplementedError("validate_stream_consistency() is not yet implemented")
