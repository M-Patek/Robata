"""Alignment service skeleton.

Implements the alignment service boundary from Architecture V1.1 Section 6.
All business methods raise ``NotImplementedError``; this module provides the
structural contract and integration points only.
"""

from __future__ import annotations

from typing import Any

from robata.alignment.models import (
    AlignmentRun,
    AlignmentValidationResult,
    FrameAlignmentProjection,
)
from robata.alignment.rational_time import PiecewiseAlignment, RationalTransformSegment
from robata.contracts.alignment import AlignmentSegment as AlignmentSegmentContract
from robata.contracts.alignment import AlignmentStatus, CameraAlignment
from robata.contracts.cameras import CameraId, SixCameraMap
from robata.contracts.common import Nanoseconds, ValidationOutcome


class AlignmentService:
    """Service boundary for multi-camera timestamp alignment.

    Responsibility: select a source clock, fit transforms into one canonical
    timeline, quantify residual error, detect gaps/non-monotonicity, and
    prevent downstream use when alignment quality is unknown or below the
    configured policy.
    """

    def __init__(self, policy_version: str) -> None:
        self.policy_version = policy_version

    def align_recording(
        self,
        *,
        mcap_id: str,
        camera_mapping_run_id: str,
        stream_timestamps: SixCameraMap[list[Nanoseconds]],
        method_hint: str | None = None,
    ) -> AlignmentRun:
        """Produce one immutable alignment run for a six-camera MCAP recording.

        Args:
            mcap_id: The canonical MCAP recording identifier.
            camera_mapping_run_id: The camera mapping run used to resolve streams.
            stream_timestamps: Per-camera ordered source timestamp series.
            method_hint: Optional preferred alignment method override.

        Returns:
            An immutable ``AlignmentRun`` with exactly six camera entries.

        Raises:
            NotImplementedError: Business logic is not yet implemented.
        """
        raise NotImplementedError("align_recording is not yet implemented")

    def fit_transform(
        self,
        *,
        camera_id: CameraId,
        source_timestamps: list[Nanoseconds],
        reference_timestamps: list[Nanoseconds] | None = None,
        anchors: list[tuple[Nanoseconds, Nanoseconds]] | None = None,
        method: str = "mcap_log_time",
    ) -> PiecewiseAlignment:
        """Fit a piecewise rational transform with anchored segments.

        Args:
            camera_id: The logical camera slot being aligned.
            source_timestamps: Ordered source clock timestamps in nanoseconds.
            reference_timestamps: Optional paired reference timestamps. When
                ``None``, the first source timestamp becomes the origin.
            anchors: Optional list of ``(source_anchor_ns, canonical_anchor_ns)``
                pairs that constrain segment boundaries.
            method: Alignment method name for provenance.

        Returns:
            A ``PiecewiseAlignment`` with one or more rational segments.

        Raises:
            NotImplementedError: Business logic is not yet implemented.
        """
        raise NotImplementedError("fit_transform is not yet implemented")

    def validate_alignment(
        self,
        *,
        alignment_run: AlignmentRun,
        source_timestamps: SixCameraMap[list[Nanoseconds]],
        max_gap_ns: Nanoseconds | None = None,
        min_coverage: float = 0.95,
    ) -> AlignmentValidationResult:
        """Validate an alignment run and return aggregate metrics.

        Checks:
        - Monotonicity of aligned timestamps per camera.
        - Gap detection against ``max_gap_ns``.
        - Residual error distribution (p50, p95, max).
        - Coverage fraction against the recording duration.
        - Segment ordering and non-overlap.

        Args:
            alignment_run: The alignment run to validate.
            source_timestamps: Original source timestamps per camera.
            max_gap_ns: Maximum tolerable gap in nanoseconds.
            min_coverage: Minimum required coverage fraction.

        Returns:
            An ``AlignmentValidationResult`` with per-camera metrics and an
            overall policy-derived status.

        Raises:
            NotImplementedError: Business logic is not yet implemented.
        """
        raise NotImplementedError("validate_alignment is not yet implemented")

    def publish_alignment_manifest(
        self,
        *,
        alignment_run: AlignmentRun,
        validation_result: AlignmentValidationResult,
        projections: list[FrameAlignmentProjection],
        artifact_uri: str | None = None,
    ) -> dict[str, Any]:
        """Publish the alignment manifest and return the published envelope.

        Args:
            alignment_run: The validated alignment run.
            validation_result: Validation outcome that authorized publication.
            projections: Frame-level alignment projections (lazy or eager).
            artifact_uri: Optional durable artifact URI for the manifest.

        Returns:
            A dictionary representing the published manifest envelope.

        Raises:
            NotImplementedError: Business logic is not yet implemented.
        """
        raise NotImplementedError("publish_alignment_manifest is not yet implemented")

    @staticmethod
    def _segment_to_contract(
        segment: RationalTransformSegment,
        *,
        segment_id: str,
        source_epoch_id: str = "default",
        rounding: str = "HALF_EVEN",
    ) -> AlignmentSegmentContract:
        """Convert an internal ``RationalTransformSegment`` to the wire contract.

        This is a pure helper; it does not persist or publish anything.
        """
        return AlignmentSegmentContract(
            segment_id=segment_id,
            source_epoch_id=source_epoch_id,
            source_order_start=segment.source_order_start,
            source_order_end=segment.source_order_end,
            source_start_ns=segment.source_start_ns,
            source_end_ns=segment.source_end_ns,
            source_anchor_ns=segment.source_anchor_ns,
            canonical_anchor_ns=segment.canonical_anchor_ns,
            rate_numerator=str(segment.rate_numerator),
            rate_denominator=str(segment.rate_denominator),
            rounding=rounding,  # type: ignore[arg-type]
        )


__all__ = [
    "AlignmentService",
]
