"""Offline fixed-point timestamp alignment with explicit evidence gates."""

from __future__ import annotations

import itertools
from bisect import bisect_left
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from fractions import Fraction
from math import ceil
from uuid import NAMESPACE_URL, uuid5

from robata.alignment.models import (
    AlignmentRun,
    AlignmentValidationMetrics,
    AlignmentValidationResult,
    FrameAlignmentProjection,
)
from robata.alignment.rational_time import PiecewiseAlignment, RationalTransformSegment
from robata.contracts.alignment import (
    AlignmentMethod,
    AlignmentStatus,
    CameraAlignment,
    CanonicalOrigin,
)
from robata.contracts.alignment import AlignmentSegment as AlignmentSegmentContract
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import INT64_MAX, INT64_MIN, Nanoseconds
from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import SchemaRegistry, default_schema_registry

_ALIGNMENT_MANIFEST_SCHEMA_ID = "https://schemas.robata.dev/alignment-manifest"
_SCHEMA_VERSION = "1.0.0"


class AlignmentError(ValueError):
    """Supplied timestamp evidence cannot form a safe transform."""


class AlignmentCapabilityError(RuntimeError):
    """An alignment method has not been approved by policy."""


class AlignmentService:
    """Fit, validate, and publish immutable six-camera alignment evidence."""

    def __init__(
        self,
        policy_version: str,
        *,
        algorithm_version: str = "offline-rational-v1",
        verified_methods: Iterable[AlignmentMethod | str] = (),
        max_gap_ns: Nanoseconds | None = None,
        schema_registry: SchemaRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError("policy_version must be a nonempty string")
        if not isinstance(algorithm_version, str) or not algorithm_version:
            raise ValueError("algorithm_version must be a nonempty string")
        if max_gap_ns is not None:
            _require_int64("max_gap_ns", max_gap_ns)
            if max_gap_ns <= 0:
                raise ValueError("max_gap_ns must be positive")
        normalized: set[AlignmentMethod] = set()
        for method in verified_methods:
            try:
                normalized.add(
                    method if isinstance(method, AlignmentMethod) else AlignmentMethod(method)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown alignment method: {method!r}") from exc
        self.policy_version = policy_version
        self.algorithm_version = algorithm_version
        self._verified_methods = frozenset(normalized)
        self._max_gap_ns = max_gap_ns
        self._schema_registry = schema_registry or default_schema_registry()
        self._clock = clock or (lambda: datetime.now(UTC))

    def align_recording(
        self,
        *,
        mcap_id: str,
        camera_mapping_run_id: str,
        source_content_sha256: str | None = None,
        camera_mapping_semantic_sha256: str | None = None,
        stream_timestamps: SixCameraMap[list[Nanoseconds]],
        method_hint: str | None = None,
        recording_start_utc: str | None = None,
    ) -> AlignmentRun:
        """Fit all cameras against cam_01 and derive a policy status.

        MCAP and mapping IDs are association references only. Canonical
        callers provide verified source-content and mapping-semantic digests
        so replay under different publication row IDs reuses the same
        alignment identity. The all-omitted fallback keeps the legacy local
        fixture API usable and is not a governed source-admission proof.
        """

        if not isinstance(mcap_id, str) or not mcap_id:
            raise ValueError("mcap_id must be a nonempty string")
        if not isinstance(camera_mapping_run_id, str) or not camera_mapping_run_id:
            raise ValueError("camera_mapping_run_id must be a nonempty string")
        try:
            method = (
                AlignmentMethod(method_hint)
                if method_hint is not None
                else AlignmentMethod.MCAP_LOG_TIME
            )
        except ValueError as exc:
            raise AlignmentError(f"unsupported alignment method hint: {method_hint!r}") from exc

        timestamps = {
            camera_id: _validate_series(stream_timestamps[camera_id], camera_id.value)
            for camera_id in CAMERA_IDS
        }
        source_digest, mapping_digest, identity_mode = _alignment_identity_inputs(
            source_content_sha256=source_content_sha256,
            camera_mapping_semantic_sha256=camera_mapping_semantic_sha256,
            timestamps=timestamps,
        )
        reference = timestamps[CameraId.CAM_01]
        if any(right <= left for left, right in itertools.pairwise(reference)):
            raise AlignmentError("reference cam_01 timestamps must be strictly increasing")
        origin_ns = reference[0]
        canonical_reference = [timestamp - origin_ns for timestamp in reference]
        transforms = {
            camera_id: self.fit_transform(
                camera_id=camera_id,
                source_timestamps=timestamps[camera_id],
                reference_timestamps=canonical_reference,
                method=method.value,
            )
            for camera_id in CAMERA_IDS
        }
        alignment_id = _deterministic_id(
            "robata-alignment-v2",
            {
                "source_content_sha256": source_digest,
                "camera_mapping_semantic_sha256": mapping_digest,
                "identity_mode": identity_mode,
                "policy_version": self.policy_version,
                "algorithm_version": self.algorithm_version,
                "method": method.value,
                "timestamps": {
                    camera_id.value: [str(value) for value in timestamps[camera_id]]
                    for camera_id in CAMERA_IDS
                },
            },
        )
        verified = method in self._verified_methods
        initial_status = AlignmentStatus.VALID if verified else AlignmentStatus.UNVERIFIED
        provisional = AlignmentRun(
            schema_version="1.0",
            alignment_id=alignment_id,
            mcap_id=mcap_id,
            camera_mapping_run_id=camera_mapping_run_id,
            reference_timebase="recording_relative_ns",
            canonical_origin=CanonicalOrigin(
                source="mcap_recording_start_in_reference_clock",
                reference_timestamp_ns=origin_ns,
                utc=recording_start_utc,
            ),
            method=method,
            algorithm_version=self.algorithm_version,
            status=initial_status,
            cameras={
                camera_id.value: self._camera_contract(
                    camera_id=camera_id,
                    transform=transforms[camera_id],
                    status=initial_status,
                )
                for camera_id in CAMERA_IDS
            },
            policy_version=self.policy_version,
            created_at=_rfc3339(self._clock()),
        )
        validation = self.validate_alignment(
            alignment_run=provisional,
            source_timestamps=stream_timestamps,
            max_gap_ns=self._max_gap_ns,
        )
        metrics = {metric.camera_id: metric for metric in validation.per_camera}
        return provisional.model_copy(
            update={
                "status": validation.overall_status,
                "cameras": {
                    camera_id.value: self._camera_contract(
                        camera_id=camera_id,
                        transform=transforms[camera_id],
                        status=_camera_status(
                            metrics[camera_id],
                            verified=verified,
                            overall_status=validation.overall_status,
                        ),
                        metrics=metrics[camera_id],
                    )
                    for camera_id in CAMERA_IDS
                },
            }
        )

    def fit_transform(
        self,
        *,
        camera_id: CameraId,
        source_timestamps: list[Nanoseconds],
        reference_timestamps: list[Nanoseconds] | None = None,
        anchors: list[tuple[Nanoseconds, Nanoseconds]] | None = None,
        method: str = "mcap_log_time",
    ) -> PiecewiseAlignment:
        """Fit anchored rational segments using integer arithmetic only."""

        if not isinstance(camera_id, CameraId):
            camera_id = CameraId(camera_id)
        source = _validate_series(source_timestamps, f"{camera_id.value} source")
        reference: list[int] | None = None
        if reference_timestamps is not None:
            reference = _validate_series(reference_timestamps, f"{camera_id.value} reference")
            if len(reference) != len(source):
                raise AlignmentError(
                    "source and reference timestamp series must have equal lengths"
                )
            if any(right <= left for left, right in itertools.pairwise(reference)):
                raise AlignmentError("reference timestamps must be strictly increasing")
        if anchors is not None:
            return self._fit_from_anchors(camera_id, source, reference, anchors, method)
        if reference is None and any(right <= left for left, right in itertools.pairwise(source)):
            raise AlignmentError("clock resets require reference timestamps or explicit anchors")

        boundaries = [0]
        boundaries.extend(
            index for index in range(1, len(source)) if source[index] <= source[index - 1]
        )
        boundaries.append(len(source))
        segments: list[RationalTransformSegment] = []
        global_origin = source[0]
        for epoch, (start, end) in enumerate(itertools.pairwise(boundaries)):
            source_start = source[start]
            source_last = source[end - 1]
            canonical_start = (
                reference[start] if reference is not None else source_start - global_origin
            )
            if end - start == 1:
                numerator, denominator = 1, 1
            else:
                canonical_last = (
                    reference[end - 1] if reference is not None else source_last - global_origin
                )
                numerator, denominator = _positive_rate(
                    canonical_last - canonical_start,
                    source_last - source_start,
                )
            source_end = _exclusive_end(source_last)
            projection = {
                "camera_id": camera_id.value,
                "method": method,
                "epoch": epoch,
                "order_start": start,
                "order_end": end,
                "source_start_ns": str(source_start),
                "source_end_ns": str(source_end),
                "canonical_anchor_ns": str(canonical_start),
                "rate_numerator": str(numerator),
                "rate_denominator": str(denominator),
            }
            segments.append(
                RationalTransformSegment(
                    source_order_start=start,
                    source_order_end=end,
                    source_start_ns=source_start,
                    source_end_ns=source_end,
                    source_anchor_ns=source_start,
                    canonical_anchor_ns=canonical_start,
                    rate_numerator=numerator,
                    rate_denominator=denominator,
                    source_epoch_id=f"epoch-{epoch}",
                    segment_id=_deterministic_id("robata-alignment-segment-v1", projection),
                )
            )
        return PiecewiseAlignment(segments)

    def validate_alignment(
        self,
        *,
        alignment_run: AlignmentRun,
        source_timestamps: SixCameraMap[list[Nanoseconds]],
        max_gap_ns: Nanoseconds | None = None,
        min_coverage: float = 0.95,
    ) -> AlignmentValidationResult:
        """Reapply transforms and return deterministic residual/coverage evidence."""

        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be within [0, 1]")
        if max_gap_ns is None:
            max_gap_ns = self._max_gap_ns
        if max_gap_ns is not None:
            _require_int64("max_gap_ns", max_gap_ns)
            if max_gap_ns <= 0:
                raise ValueError("max_gap_ns must be positive")
        try:
            origin_ns = _parse_int64_string(
                alignment_run.canonical_origin.reference_timestamp_ns,
                "canonical origin",
            )
        except (TypeError, ValueError) as exc:
            raise AlignmentError("alignment run has no canonical reference timestamp") from exc
        reference = _validate_series(source_timestamps[CameraId.CAM_01], "cam_01 source")
        canonical_reference = [timestamp - origin_ns for timestamp in reference]

        per_camera: list[AlignmentValidationMetrics] = []
        issues: list[str] = []
        structural_invalid = False
        degraded = False
        expected_cameras = {camera_id.value for camera_id in CAMERA_IDS}
        if set(alignment_run.cameras) != expected_cameras:
            structural_invalid = True
            issues.append("alignment run must contain exactly cam_01 through cam_06")
        if alignment_run.policy_version != self.policy_version:
            structural_invalid = True
            issues.append("alignment policy version does not match this service")
        for camera_id in CAMERA_IDS:
            source = _validate_series(source_timestamps[camera_id], f"{camera_id.value} source")
            camera = alignment_run.cameras.get(camera_id.value)
            if camera is None:
                structural_invalid = True
                issues.append(f"{camera_id.value}: missing camera alignment")
                per_camera.append(_empty_metrics(camera_id))
                continue
            try:
                piecewise = PiecewiseAlignment(
                    tuple(_contract_to_segment(segment) for segment in camera.segments)
                )
            except (TypeError, ValueError) as exc:
                structural_invalid = True
                issues.append(f"{camera_id.value}: invalid transform segments: {exc}")
                per_camera.append(_empty_metrics(camera_id))
                continue

            aligned: list[int] = []
            transform_failures = 0
            for order, timestamp in enumerate(source):
                try:
                    aligned.append(piecewise.apply(order, timestamp))
                except (OverflowError, TypeError, ValueError):
                    transform_failures += 1
            if transform_failures:
                structural_invalid = True
                issues.append(
                    f"{camera_id.value}: {transform_failures} timestamps fall outside "
                    "transform coverage"
                )
            nonmonotonic = sum(right <= left for left, right in itertools.pairwise(aligned))
            if nonmonotonic:
                structural_invalid = True
                issues.append(f"{camera_id.value}: aligned timestamps are not strictly increasing")
            duplicate_count = sum(right == left for left, right in itertools.pairwise(source))
            if duplicate_count:
                degraded = True
                issues.append(f"{camera_id.value}: source timestamp duplicates={duplicate_count}")
            reset_count = sum(right <= left for left, right in itertools.pairwise(source))
            if reset_count:
                degraded = True
                issues.append(
                    f"{camera_id.value}: source clock resets/non-monotonic points={reset_count}"
                )

            paired_count = min(len(aligned), len(canonical_reference))
            residuals = [
                abs(aligned[index] - canonical_reference[index]) for index in range(paired_count)
            ]
            residual_p50 = _percentile(residuals, 0.50)
            residual_p95 = _percentile(residuals, 0.95)
            max_error = max(residuals, default=0)
            if max_error:
                degraded = True
                issues.append(f"{camera_id.value}: residual max={max_error}ns")
            count_coverage = paired_count / max(len(source), len(canonical_reference), 1)
            span_coverage = _span_coverage(aligned, canonical_reference)
            coverage = min(count_coverage, span_coverage)
            if coverage < min_coverage:
                structural_invalid = True
                issues.append(
                    f"{camera_id.value}: coverage {coverage:.6f} below minimum {min_coverage:.6f}"
                )
            elif coverage < 1.0:
                degraded = True
            gap_count = 0
            if max_gap_ns is not None:
                gap_count = sum(
                    right - left > max_gap_ns for left, right in itertools.pairwise(aligned)
                )
                if gap_count:
                    structural_invalid = True
                    issues.append(
                        f"{camera_id.value}: {gap_count} aligned gaps exceed {max_gap_ns}ns"
                    )
            per_camera.append(
                AlignmentValidationMetrics(
                    camera_id=camera_id,
                    residual_p50_ns=residual_p50,
                    residual_p95_ns=residual_p95,
                    max_error_ns=max_error,
                    derived_drift_ppm=_drift_ppm(camera.segments),
                    coverage=max(0.0, min(1.0, coverage)),
                    gap_count=gap_count,
                    duplicate_count=duplicate_count,
                    out_of_range_count=transform_failures
                    + abs(len(source) - len(canonical_reference)),
                )
            )

        if structural_invalid or alignment_run.status is AlignmentStatus.INVALID:
            overall = AlignmentStatus.INVALID
        elif (
            alignment_run.status is AlignmentStatus.UNVERIFIED
            or alignment_run.method not in self._verified_methods
        ):
            overall = AlignmentStatus.UNVERIFIED
        elif degraded or alignment_run.status is AlignmentStatus.DEGRADED:
            overall = AlignmentStatus.DEGRADED
        else:
            overall = AlignmentStatus.VALID
        return AlignmentValidationResult(
            alignment_id=alignment_run.alignment_id,
            per_camera=tuple(per_camera),
            overall_status=overall,
            issues=tuple(issues),
        )

    def publish_alignment_manifest(
        self,
        *,
        alignment_run: AlignmentRun,
        validation_result: AlignmentValidationResult,
        projections: list[FrameAlignmentProjection],
    ) -> AlignmentRun:
        """Publish the registered manifest body after consistency checks."""

        if alignment_run.alignment_id != validation_result.alignment_id:
            raise AlignmentError("validation result references a different alignment ID")
        if validation_result.overall_status in {
            AlignmentStatus.INVALID,
            AlignmentStatus.UNVERIFIED,
        }:
            raise AlignmentError("INVALID or UNVERIFIED alignment cannot be published")
        if alignment_run.status is not validation_result.overall_status:
            raise AlignmentError("alignment run and validation status differ")
        expected_cameras = {camera_id.value for camera_id in CAMERA_IDS}
        if set(alignment_run.cameras) != expected_cameras:
            raise AlignmentError("alignment manifest must contain exactly six camera keys")
        metrics = {metric.camera_id: metric for metric in validation_result.per_camera}
        for camera_id in CAMERA_IDS:
            camera = alignment_run.cameras[camera_id.value]
            metric = metrics[camera_id]
            expected_status = _camera_status(
                metric,
                verified=alignment_run.method in self._verified_methods,
                overall_status=validation_result.overall_status,
            )
            if (
                camera.residual_p95_ns != metric.residual_p95_ns
                or camera.max_error_ns != metric.max_error_ns
                or camera.derived_drift_ppm != metric.derived_drift_ppm
                or camera.coverage != metric.coverage
                or camera.status is not expected_status
            ):
                raise AlignmentError(
                    f"{camera_id.value}: manifest does not match validation evidence"
                )
        seen_projection_keys: set[tuple[str, str]] = set()
        for projection in projections:
            if projection.alignment_id != alignment_run.alignment_id:
                raise AlignmentError("projection references a different alignment ID")
            projection_camera = alignment_run.cameras.get(projection.camera_id.value)
            if projection_camera is None or projection.segment_id not in {
                segment.segment_id for segment in projection_camera.segments
            }:
                raise AlignmentError("projection references an unknown camera segment")
            key = (projection.source_frame_id, projection.alignment_id)
            if key in seen_projection_keys:
                raise AlignmentError("duplicate source-frame/alignment projection")
            seen_projection_keys.add(key)
        registered = self._schema_registry.resolve_version(
            _ALIGNMENT_MANIFEST_SCHEMA_ID,
            _SCHEMA_VERSION,
        )
        self._schema_registry.validate_pinned(
            registered.ref,
            alignment_run.model_dump(mode="json"),
        )
        return alignment_run

    def _camera_contract(
        self,
        *,
        camera_id: CameraId,
        transform: PiecewiseAlignment,
        status: AlignmentStatus,
        metrics: AlignmentValidationMetrics | None = None,
    ) -> CameraAlignment:
        first = transform.segments[0]
        if metrics is None:
            drift_ppm = _drift_ppm_from_rate(first.rate_numerator, first.rate_denominator)
            residual_p95 = 0
            max_error = 0
            coverage = 1.0
        else:
            drift_ppm = metrics.derived_drift_ppm
            residual_p95 = metrics.residual_p95_ns
            max_error = metrics.max_error_ns
            coverage = metrics.coverage
        return CameraAlignment(
            source_clock_id=f"{camera_id.value}:source",
            source_timestamp_unit="ns",
            derived_drift_ppm=float(drift_ppm),
            residual_p95_ns=residual_p95,
            max_error_ns=max_error,
            coverage=coverage,
            segments=tuple(
                self._segment_to_contract(
                    segment,
                    segment_id=segment.segment_id
                    or _deterministic_id(
                        "robata-alignment-segment-v1",
                        {"camera_id": camera_id.value, "order": segment.source_order_start},
                    ),
                    source_epoch_id=segment.source_epoch_id,
                )
                for segment in transform.segments
            ),
            status=status,
        )

    def _fit_from_anchors(
        self,
        camera_id: CameraId,
        source: list[int],
        reference: list[int] | None,
        anchors: list[tuple[int, int]],
        method: str,
    ) -> PiecewiseAlignment:
        if len(anchors) < 2:
            raise AlignmentError("at least two anchors are required")
        if any(right <= left for left, right in itertools.pairwise(source)):
            raise AlignmentError("explicit anchors require strictly increasing source timestamps")
        normalized = [
            (
                _require_int64("source anchor", source_anchor),
                _require_int64("canonical anchor", canonical_anchor),
            )
            for source_anchor, canonical_anchor in anchors
        ]
        if any(
            right[0] <= left[0] or right[1] <= left[1]
            for left, right in itertools.pairwise(normalized)
        ):
            raise AlignmentError("anchors must increase in both source and canonical time")
        if source[0] != normalized[0][0] or source[-1] != normalized[-1][0]:
            raise AlignmentError("anchors must cover the first and last source timestamp")
        positions: list[int] = []
        for source_anchor, _ in normalized:
            position = bisect_left(source, source_anchor)
            if position >= len(source) or source[position] != source_anchor:
                raise AlignmentError("every source anchor must identify an observed timestamp")
            positions.append(position)
        if reference is not None:
            expected = [reference[position] for position in positions]
            if expected != [canonical_anchor for _, canonical_anchor in normalized]:
                raise AlignmentError("anchors disagree with paired reference timestamps")

        segments: list[RationalTransformSegment] = []
        anchor_pairs = itertools.pairwise(normalized)
        for index, ((source_anchor, canonical_anchor), (next_source, next_canonical)) in enumerate(
            anchor_pairs
        ):
            order_start = positions[index]
            order_end = positions[index + 1]
            if index == len(normalized) - 2:
                order_end += 1
            numerator, denominator = _positive_rate(
                next_canonical - canonical_anchor,
                next_source - source_anchor,
            )
            source_end = _exclusive_end(source[order_end - 1])
            segments.append(
                RationalTransformSegment(
                    source_order_start=order_start,
                    source_order_end=order_end,
                    source_start_ns=source_anchor,
                    source_end_ns=source_end,
                    source_anchor_ns=source_anchor,
                    canonical_anchor_ns=canonical_anchor,
                    rate_numerator=numerator,
                    rate_denominator=denominator,
                    source_epoch_id=f"epoch-{index}",
                    segment_id=_deterministic_id(
                        "robata-alignment-segment-v1",
                        {
                            "camera_id": camera_id.value,
                            "method": method,
                            "order_start": order_start,
                            "order_end": order_end,
                            "source_anchor_ns": str(source_anchor),
                            "canonical_anchor_ns": str(canonical_anchor),
                        },
                    ),
                )
            )
        return PiecewiseAlignment(segments)

    @staticmethod
    def _segment_to_contract(
        segment: RationalTransformSegment,
        *,
        segment_id: str,
        source_epoch_id: str = "default",
        rounding: str = "HALF_EVEN",
    ) -> AlignmentSegmentContract:
        """Convert an internal rational segment to the wire contract."""

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


def _validate_series(values: list[int], label: str) -> list[int]:
    if not isinstance(values, list) or len(values) < 2:
        raise AlignmentError(f"{label} timestamp series must contain at least two values")
    return [_require_int64(label, value) for value in values]


def _require_int64(label: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < INT64_MIN or value > INT64_MAX:
        raise ValueError(f"{label} must fit signed int64")
    return value


def _exclusive_end(value: int) -> int:
    if value >= INT64_MAX:
        raise AlignmentError("timestamp at INT64_MAX cannot form a half-open interval")
    return value + 1


def _positive_rate(numerator: int, denominator: int) -> tuple[int, int]:
    if numerator <= 0 or denominator <= 0:
        raise AlignmentError("alignment rate must be positive")
    ratio = Fraction(numerator, denominator)
    return ratio.numerator, ratio.denominator


def _parse_int64_string(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer string")
    if isinstance(value, int):
        return _require_int64(label, value)
    if not isinstance(value, str) or not value or (value[0] == "-" and not value[1:]):
        raise ValueError(f"{label} must be an integer string")
    digits = value[1:] if value[0] == "-" else value
    if not digits.isdigit() or (len(digits) > 1 and digits[0] == "0") or value == "-0":
        raise ValueError(f"{label} must be canonical decimal")
    return _require_int64(label, int(value))


def _contract_to_segment(segment: AlignmentSegmentContract) -> RationalTransformSegment:
    if segment.rounding != "HALF_EVEN":
        raise ValueError("only HALF_EVEN alignment segments are executable")
    return RationalTransformSegment(
        source_order_start=segment.source_order_start,
        source_order_end=segment.source_order_end,
        source_start_ns=segment.source_start_ns,
        source_end_ns=segment.source_end_ns,
        source_anchor_ns=segment.source_anchor_ns,
        canonical_anchor_ns=segment.canonical_anchor_ns,
        rate_numerator=_parse_positive_decimal(segment.rate_numerator, "rate_numerator"),
        rate_denominator=_parse_positive_decimal(segment.rate_denominator, "rate_denominator"),
        source_epoch_id=segment.source_epoch_id,
        segment_id=segment.segment_id,
    )


def _parse_positive_decimal(value: str, label: str) -> int:
    if not isinstance(value, str) or not value.isdigit() or int(value) <= 0:
        raise ValueError(f"{label} must be a positive decimal string")
    if len(value) > 1 and value[0] == "0":
        raise ValueError(f"{label} must be canonical decimal")
    return int(value)


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, ceil(quantile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _span_coverage(aligned: list[int], reference: list[int]) -> float:
    if not aligned or not reference or reference[-1] <= reference[0]:
        return 0.0
    overlap_start = max(aligned[0], reference[0])
    overlap_end = min(aligned[-1], reference[-1])
    if overlap_end <= overlap_start:
        return 0.0
    return (overlap_end - overlap_start) / (reference[-1] - reference[0])


def _drift_ppm(segments: tuple[AlignmentSegmentContract, ...]) -> float:
    if not segments:
        return 0.0
    values = [
        _drift_ppm_from_rate(
            _parse_positive_decimal(segment.rate_numerator, "rate_numerator"),
            _parse_positive_decimal(segment.rate_denominator, "rate_denominator"),
        )
        for segment in segments
    ]
    return sum(values) / len(values)


def _drift_ppm_from_rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator - 1.0) * 1_000_000.0


def _empty_metrics(camera_id: CameraId) -> AlignmentValidationMetrics:
    return AlignmentValidationMetrics(
        camera_id=camera_id,
        residual_p50_ns=0,
        residual_p95_ns=0,
        max_error_ns=0,
        derived_drift_ppm=0.0,
        coverage=0.0,
    )


def _camera_status(
    metrics: AlignmentValidationMetrics,
    *,
    verified: bool,
    overall_status: AlignmentStatus,
) -> AlignmentStatus:
    if overall_status is AlignmentStatus.INVALID:
        return AlignmentStatus.INVALID
    if not verified or overall_status is AlignmentStatus.UNVERIFIED:
        return AlignmentStatus.UNVERIFIED
    if (
        overall_status is AlignmentStatus.DEGRADED
        or metrics.coverage < 1.0
        or metrics.max_error_ns > 0
        or metrics.gap_count
        or metrics.duplicate_count
        or metrics.out_of_range_count
    ):
        return AlignmentStatus.DEGRADED
    return AlignmentStatus.VALID


def _deterministic_id(namespace: str, projection: object) -> str:
    digest = semantic_sha256({"namespace": namespace, "projection": projection})
    return str(uuid5(NAMESPACE_URL, f"{namespace}:{digest}"))


def _alignment_identity_inputs(
    *,
    source_content_sha256: str | None,
    camera_mapping_semantic_sha256: str | None,
    timestamps: dict[CameraId, list[Nanoseconds]],
) -> tuple[str, str, str]:
    """Resolve explicit identity inputs without allowing row IDs into the key."""

    if (source_content_sha256 is None) != (camera_mapping_semantic_sha256 is None):
        raise ValueError(
            "source_content_sha256 and camera_mapping_semantic_sha256 must be supplied together"
        )
    if source_content_sha256 is not None and camera_mapping_semantic_sha256 is not None:
        _require_sha256("source_content_sha256", source_content_sha256)
        _require_sha256("camera_mapping_semantic_sha256", camera_mapping_semantic_sha256)
        return source_content_sha256, camera_mapping_semantic_sha256, "explicit"

    # Legacy callers do not have admission artifacts. Keep their identity
    # deterministic while making the non-governed fallback visible in the key.
    evidence = {
        camera_id.value: [str(value) for value in timestamps[camera_id]] for camera_id in CAMERA_IDS
    }
    return (
        semantic_sha256({"legacy_alignment_timestamp_evidence": evidence}),
        semantic_sha256({"legacy_camera_slot_order": [camera.value for camera in CAMERA_IDS]}),
        "legacy_alignment_inputs",
    )


def _require_sha256(label: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "AlignmentCapabilityError",
    "AlignmentError",
    "AlignmentService",
]
