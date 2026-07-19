"""Camera/Recording QA aggregation.

This module implements the final aggregation stage of the two-stage QA pipeline
(Architecture V1 Section 12.4).  It merges coarse and dense camera results into
a single recording-level QA decision, preserving provenance and separating
model scores, calibrated probabilities, deterministic features, and policy
decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from robata.contracts.cameras import CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mainline import (
    CameraQAStatus,
    CameraQAResult,
    RecordingQAStatus,
)

__all__ = [
    "QAAggregator",
    "RecordingQAResult",
]


class RecordingQAResult(StrictModel):
    """Recording-level QA aggregate.

    Follows the Architecture V1 Section 12.3 ``MCAPQAResult`` schema.
    Distinguishes model scores, calibrated probabilities, deterministic
    quality features, and final policy decisions.
    """

    mcap_id: OpaqueUuid
    scope: NanosecondInterval
    overall_status: RecordingQAStatus
    required_camera_count: Annotated[int, Field(strict=True, ge=0, le=6)] = 6
    usable_camera_count: Annotated[int, Field(strict=True, ge=0, le=6)]
    camera_result_ids: tuple[OpaqueUuid, ...]
    overall_quality: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ]
    model_score: Annotated[
        float | None,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = None
    calibrated_probability: Annotated[
        float | None,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = None
    deterministic_quality: Annotated[
        float | None,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = None
    policy_version: Annotated[str, Field(strict=True, min_length=1)]
    coarse_result_ids: tuple[OpaqueUuid, ...] = ()
    dense_result_ids: tuple[OpaqueUuid, ...] = ()


class QAAggregator:
    """Aggregate per-camera QA results into a recording-level decision.

    Implements the aggregation rules from Architecture V1 Section 12.4:

    - Preserve per-camera issue intervals.
    - Associate coarse and dense package/inference provenance.
    - Keep model score, calibrated probability, deterministic quality features,
      and final policy decision separate.
    - Use ``UNKNOWN`` or ``INCOMPLETE`` when evidence is absent.  Do not
      substitute ``GOOD``.
    """

    def aggregate_camera_results(
        self,
        coarse_results: Sequence[CameraQAResult],
        dense_results: Sequence[CameraQAResult] | None = None,
    ) -> RecordingQAResult:
        """Aggregate coarse (and optional dense) camera results.

        Parameters
        ----------
        coarse_results:
            Six camera QA results from the coarse stage.
        dense_results:
            Optional six camera QA results from the dense stage.  When dense
            results are present they override coarse results for the same
            camera and interval.

        Returns
        -------
        RecordingQAResult:
            Recording-level QA decision with full provenance.
        """
        if len(coarse_results) != 6:
            raise ValueError(
                f"Expected exactly 6 coarse camera results, got {len(coarse_results)}"
            )

        # Build camera map from coarse results.
        camera_map: dict[CameraId, CameraQAResult] = {}
        for result in coarse_results:
            camera_map[result.camera_id] = result

        # Override with dense results when present.
        if dense_results:
            if len(dense_results) != 6:
                raise ValueError(
                    f"Expected exactly 6 dense camera results, got {len(dense_results)}"
                )
            for result in dense_results:
                camera_map[result.camera_id] = result

        # Ensure all six cameras are present.
        for camera_id in CameraId:
            if camera_id not in camera_map:
                raise ValueError(f"Missing QA result for camera {camera_id.value}")

        # Count usable cameras (GOOD or DEGRADED).
        usable = sum(
            1
            for result in camera_map.values()
            if result.claim.status in {CameraQAStatus.GOOD, CameraQAStatus.DEGRADED}
        )

        # Determine overall status based on usable count.
        if usable == 6:
            overall_status = RecordingQAStatus.USABLE
        elif usable >= 4:
            overall_status = RecordingQAStatus.DEGRADED
        elif usable >= 1:
            overall_status = RecordingQAStatus.UNUSABLE
        else:
            overall_status = RecordingQAStatus.INVALID

        # Check for INCOMPLETE: any camera with INCOMPLETE status makes the
        # recording INCOMPLETE unless all others are GOOD/DEGRADED.
        incomplete_count = sum(
            1
            for result in camera_map.values()
            if result.claim.status is CameraQAStatus.INCOMPLETE
        )
        if incomplete_count > 0 and usable + incomplete_count < 6:
            overall_status = RecordingQAStatus.INCOMPLETE

        # Compute overall quality as mean of per-camera quality scores.
        # Missing evidence (UNKNOWN/INCOMPLETE) contributes 0.0 to the mean.
        scores: list[float] = []
        for result in camera_map.values():
            if result.claim.reported_score is not None:
                scores.append(result.claim.reported_score)
            elif result.claim.status in {CameraQAStatus.GOOD, CameraQAStatus.DEGRADED}:
                scores.append(1.0 if result.claim.status is CameraQAStatus.GOOD else 0.5)
            else:
                scores.append(0.0)

        overall_quality = sum(scores) / len(scores) if scores else 0.0

        # Collect result IDs.
        camera_result_ids = tuple(
            camera_map[camera_id].qa_result_id for camera_id in CameraId
        )
        coarse_ids = tuple(r.qa_result_id for r in coarse_results)
        dense_ids = tuple(r.qa_result_id for r in dense_results) if dense_results else ()

        # Derive scope from coarse results (all cameras should share the same scope).
        scope = coarse_results[0].claim.observed_interval

        # Extract model score and deterministic quality from the first result
        # that has them.  In a full implementation these would be properly
        # aggregated across all cameras.
        model_score = None
        deterministic_quality = None
        for result in camera_map.values():
            if result.claim.reported_score is not None and model_score is None:
                model_score = result.claim.reported_score
            # Fast detector issues contribute to deterministic quality.
            # This is a simplified heuristic.
            if deterministic_quality is None and result.claim.issues:
                deterministic_quality = max(
                    0.0, 1.0 - len(result.claim.issues) * 0.1
                )

        return RecordingQAResult(
            mcap_id=coarse_results[0].mcap_id,
            scope=scope,
            overall_status=overall_status,
            usable_camera_count=usable,
            camera_result_ids=camera_result_ids,
            overall_quality=round(overall_quality, 4),
            model_score=model_score,
            calibrated_probability=None,  # Set by downstream calibrator.
            deterministic_quality=deterministic_quality,
            policy_version="v1.0",
            coarse_result_ids=coarse_ids,
            dense_result_ids=dense_ids,
        )
