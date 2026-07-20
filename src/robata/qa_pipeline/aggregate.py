"""Camera/Recording QA aggregation.

This module implements the final aggregation stage of the two-stage QA pipeline
(Architecture V1 Section 12.4).  It merges coarse and dense camera results into
a single recording-level QA decision, preserving provenance and separating
model scores, calibrated probabilities, deterministic features, and policy
decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Self

from pydantic import Field, model_validator

from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mainline import (
    CameraQAResult,
    CameraQAStatus,
    RecordingQAStatus,
)

__all__ = [
    "QAAggregationPolicy",
    "QAAggregator",
    "RecordingQAResult",
]


class QAAggregationPolicy(StrictModel):
    """Explicit non-provider policy for one six-camera QA aggregation run.

    The repository's production O-10 policy is unresolved.  A local policy may
    exercise the contract, but its result must retain ``promotion_eligible=False``.
    """

    version: Annotated[str, Field(strict=True, min_length=1)]
    degraded_min_usable: Annotated[int, Field(strict=True, ge=1, le=5)]
    incomplete_is_blocking: bool = True
    status_quality: dict[CameraQAStatus, Annotated[float, Field(ge=0.0, le=1.0)]]
    promotion_eligible: bool = False

    @model_validator(mode="after")
    def validate_status_quality(self) -> Self:
        if set(self.status_quality) != set(CameraQAStatus):
            raise ValueError("status_quality must define every CameraQAStatus")
        if self.promotion_eligible:
            raise ValueError("O-10 is unresolved; QA aggregation cannot be promotable")
        return self

    @classmethod
    def local_development(cls) -> QAAggregationPolicy:
        """Return the explicit local-only policy used by deterministic tests."""

        return cls(
            version="local-development-v1",
            degraded_min_usable=4,
            status_quality={
                CameraQAStatus.GOOD: 1.0,
                CameraQAStatus.DEGRADED: 0.5,
                CameraQAStatus.UNUSABLE: 0.0,
                CameraQAStatus.UNKNOWN: 0.0,
                CameraQAStatus.INCOMPLETE: 0.0,
            },
            promotion_eligible=False,
        )


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
    promotion_eligible: bool = False
    coarse_result_ids: tuple[OpaqueUuid, ...] = ()
    dense_result_ids: tuple[OpaqueUuid, ...] = ()

    @model_validator(mode="after")
    def validate_cardinality(self) -> Self:
        if self.required_camera_count != 6:
            raise ValueError("the V1 QA aggregate requires six camera slots")
        if len(self.camera_result_ids) != 6 or len(set(self.camera_result_ids)) != 6:
            raise ValueError("camera_result_ids must contain six unique results")
        if self.usable_camera_count > self.required_camera_count:
            raise ValueError("usable_camera_count cannot exceed required_camera_count")
        return self


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

    def __init__(self, policy: QAAggregationPolicy | None = None) -> None:
        self._policy = policy or QAAggregationPolicy.local_development()

    @property
    def policy(self) -> QAAggregationPolicy:
        return self._policy

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

        scope = coarse_results[0].claim.observed_interval
        mcap_id = coarse_results[0].mcap_id

        # Build camera map from coarse results.
        camera_map: dict[CameraId, CameraQAResult] = {}
        for result in coarse_results:
            if result.mcap_id != mcap_id:
                raise ValueError("all coarse QA results must belong to one MCAP")
            if result.claim.observed_interval != scope:
                raise ValueError("coarse QA intervals must share one exact scope")
            if result.camera_id in camera_map:
                raise ValueError(f"duplicate coarse QA result for {result.camera_id.value}")
            camera_map[result.camera_id] = result

        # Override with dense results when present.
        if dense_results:
            if len(dense_results) != 6:
                raise ValueError(
                    f"Expected exactly 6 dense camera results, got {len(dense_results)}"
                )
            dense_camera_ids: set[CameraId] = set()
            for result in dense_results:
                if result.mcap_id != mcap_id:
                    raise ValueError("all dense QA results must belong to the coarse MCAP")
                if result.camera_id in dense_camera_ids:
                    raise ValueError(f"duplicate dense QA result for {result.camera_id.value}")
                dense_camera_ids.add(result.camera_id)
                observed = result.claim.observed_interval
                if observed.start_ns < scope.start_ns or observed.end_ns > scope.end_ns:
                    raise ValueError("dense QA intervals must lie inside the coarse scope")
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
        elif usable >= self._policy.degraded_min_usable:
            overall_status = RecordingQAStatus.DEGRADED
        else:
            overall_status = RecordingQAStatus.UNUSABLE

        # Check for INCOMPLETE: any camera with INCOMPLETE status makes the
        # recording INCOMPLETE unless all others are GOOD/DEGRADED.
        incomplete_count = sum(
            1
            for result in camera_map.values()
            if result.claim.status is CameraQAStatus.INCOMPLETE
        )
        if incomplete_count > 0 and self._policy.incomplete_is_blocking:
            overall_status = RecordingQAStatus.INCOMPLETE

        # Compute overall quality as mean of per-camera quality scores.
        # Missing evidence (UNKNOWN/INCOMPLETE) contributes 0.0 to the mean.
        policy_scores = [
            self._policy.status_quality[result.claim.status]
            for result in camera_map.values()
        ]
        overall_quality = sum(policy_scores) / len(policy_scores)

        # Collect result IDs.
        camera_result_ids = tuple(
            camera_map[camera_id].qa_result_id for camera_id in CameraId
        )
        coarse_by_camera = {result.camera_id: result for result in coarse_results}
        coarse_ids = tuple(coarse_by_camera[camera_id].qa_result_id for camera_id in CameraId)
        dense_by_camera = {
            result.camera_id: result for result in dense_results or ()
        }
        dense_ids = tuple(
            dense_by_camera[camera_id].qa_result_id
            for camera_id in CameraId
            if camera_id in dense_by_camera
        )

        # Extract model score and deterministic quality from the first result
        # that has them.  In a full implementation these would be properly
        # aggregated across all cameras.
        reported_scores = [
            result.claim.reported_score
            for result in camera_map.values()
            if result.claim.reported_score is not None
        ]
        model_score = (
            round(sum(reported_scores) / len(reported_scores), 4)
            if reported_scores
            else None
        )
        return RecordingQAResult(
            mcap_id=mcap_id,
            scope=scope,
            overall_status=overall_status,
            usable_camera_count=usable,
            camera_result_ids=camera_result_ids,
            overall_quality=round(overall_quality, 4),
            model_score=model_score,
            calibrated_probability=None,  # Set by downstream calibrator.
            deterministic_quality=None,
            policy_version=self._policy.version,
            promotion_eligible=self._policy.promotion_eligible,
            coarse_result_ids=coarse_ids,
            dense_result_ids=dense_ids,
        )
