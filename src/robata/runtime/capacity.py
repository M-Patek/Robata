"""Deterministic synthetic capacity harness with non-promotional evidence semantics."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import StrEnum
from uuid import NAMESPACE_URL, uuid5

from robata.contracts.hashing import semantic_sha256

_NANOSECONDS_PER_HOUR = 3_600_000_000_000
_SECONDS_PER_HOUR = 3_600
_MILLISECONDS_PER_HOUR = 3_600_000


class SyntheticOutcome(StrEnum):
    """Mutually exclusive outcome observed at the synthetic cutoff."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    PENDING = "PENDING"


class BottleneckKind(StrEnum):
    """Locally observed pressure signal, never a production diagnosis."""

    SERVICE_CAPACITY = "SERVICE_CAPACITY"
    QUEUE_WAIT = "QUEUE_WAIT"
    RELIABILITY = "RELIABILITY"
    DEADLINE = "DEADLINE"
    NONE = "NONE"


class CapacityEvidenceClass(StrEnum):
    """Provenance class shown on every measured-capacity artifact.

    These labels deliberately do not promote a local observation.  A caller must
    supply the class that describes the workload and environment it actually ran.
    """

    LOCAL_CONFORMANCE = "LOCAL_CONFORMANCE"
    REPRESENTATIVE_BENCHMARK = "REPRESENTATIVE_BENCHMARK"
    PRODUCTION_QUALIFICATION = "PRODUCTION_QUALIFICATION"


class ProviderMode(StrEnum):
    """Provider execution mode needed to interpret call and token measurements."""

    LOCAL_OFFLINE_FIXTURE = "LOCAL_OFFLINE_FIXTURE"
    NETWORK_PROVIDER = "NETWORK_PROVIDER"
    NO_PROVIDER_CALLS = "NO_PROVIDER_CALLS"
    UNKNOWN = "UNKNOWN"


class MeasuredCapacityStatus(StrEnum):
    """Whether denominator-safe capacity rates can be stated."""

    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class MeasuredCapacityComparisonKind(StrEnum):
    """Topology of a profile-to-profile comparison."""

    FRESH_VS_REPLAY = "FRESH_VS_REPLAY"
    RECORDING_WORKER_SCALING = "RECORDING_WORKER_SCALING"
    LIKE_FOR_LIKE = "LIKE_FOR_LIKE"
    MIXED = "MIXED"


@dataclass(frozen=True, slots=True)
class MeasuredCapacityInput:
    """Primitive, explicitly-unitized facts from one real profile observation.

    Counts are intentionally optional: an absent instrument must remain absent rather
    than be represented as a zero.  ``recording_duration_ns`` and ``provider_mode``
    are required before this input can produce throughput/capacity rates.
    """

    workload_fingerprint: str
    evidence_class: CapacityEvidenceClass
    provider_mode: ProviderMode
    execution_mode: str
    recording_count: int
    recording_worker_count: int
    camera_count: int
    recording_duration_ns: int | None
    wall_time_ns: int
    windows: int | None = None
    unique_images: int | None = None
    coarse_unique_images: int | None = None
    dense_unique_images: int | None = None
    provider_images: int | None = None
    logical_calls: int | None = None
    call_parts: int | None = None
    call_splits: int | None = None
    http_requests: int | None = None
    retries: int | None = None
    batches: int | None = None
    batch_requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    output_token_responses: int | None = None
    dense_logical_calls: int | None = None
    dense_provider_images: int | None = None

    def __post_init__(self) -> None:
        _require_nonempty("workload_fingerprint", self.workload_fingerprint)
        if not isinstance(self.evidence_class, CapacityEvidenceClass):
            raise TypeError("evidence_class must be a CapacityEvidenceClass")
        if not isinstance(self.provider_mode, ProviderMode):
            raise TypeError("provider_mode must be a ProviderMode")
        if self.execution_mode not in {"FRESH", "REPLAY", "UNKNOWN"}:
            raise ValueError("execution_mode must be FRESH, REPLAY, or UNKNOWN")
        _require_positive_int("recording_count", self.recording_count)
        _require_positive_int("recording_worker_count", self.recording_worker_count)
        _require_positive_int("camera_count", self.camera_count)
        if self.recording_duration_ns is not None:
            _require_positive_int("recording_duration_ns", self.recording_duration_ns)
        _require_nonnegative_int("wall_time_ns", self.wall_time_ns)
        for name in (
            "windows",
            "unique_images",
            "coarse_unique_images",
            "dense_unique_images",
            "provider_images",
            "logical_calls",
            "call_parts",
            "call_splits",
            "http_requests",
            "retries",
            "batches",
            "batch_requests",
            "input_tokens",
            "output_tokens",
            "output_token_responses",
            "dense_logical_calls",
            "dense_provider_images",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_nonnegative_int(name, value)
        for name, selected_images in (
            ("coarse_unique_images", self.coarse_unique_images),
            ("dense_unique_images", self.dense_unique_images),
        ):
            if (
                selected_images is not None
                and self.unique_images is not None
                and selected_images > self.unique_images
            ):
                raise ValueError(f"{name} cannot exceed unique_images")
        if (
            self.dense_provider_images is not None
            and self.provider_images is not None
            and self.dense_provider_images > self.provider_images
        ):
            raise ValueError("dense_provider_images cannot exceed provider_images")
        if (
            self.dense_logical_calls is not None
            and self.logical_calls is not None
            and self.dense_logical_calls > self.logical_calls
        ):
            raise ValueError("dense_logical_calls cannot exceed logical_calls")
        if (
            self.call_parts is not None
            and self.logical_calls is not None
            and self.call_splits is not None
            and self.call_parts != self.logical_calls + self.call_splits
        ):
            raise ValueError("call_parts must equal logical_calls plus call_splits")


@dataclass(frozen=True, slots=True)
class MeasuredCapacityReport:
    """A denominator-safe capacity projection from one observed profile.

    ``dense_upgrade_fraction`` is the fraction of globally unique selected frame
    identities that appear in dense scope; ``dense_provider_image_fraction`` is kept
    separate because overlap and splitting amplify provider payload.  ``production_eligible``
    is always false because calculation and comparison do not grant qualification.
    """

    measurement_status: MeasuredCapacityStatus
    unavailable_reasons: tuple[str, ...]
    workload_fingerprint: str
    evidence_class: CapacityEvidenceClass
    provider_mode: ProviderMode
    execution_mode: str
    recording_count: int
    recording_worker_count: int
    camera_count: int
    recording_duration_ns: int | None
    wall_time_ns: int
    recording_hours: float | None
    camera_hours: float | None
    wall_hours: float | None
    recording_hours_per_wall_hour: float | None
    camera_hours_per_wall_hour: float | None
    windows: int | None
    unique_images: int | None
    coarse_unique_images: int | None
    dense_unique_images: int | None
    provider_images: int | None
    logical_calls: int | None
    call_parts: int | None
    call_splits: int | None
    http_requests: int | None
    retries: int | None
    batches: int | None
    batch_requests: int | None
    input_tokens: int | None
    output_tokens: int | None
    output_token_responses: int | None
    dense_logical_calls: int | None
    dense_provider_images: int | None
    dense_logical_call_fraction: float | None
    dense_upgrade_fraction: float | None
    dense_provider_image_fraction: float | None
    windows_per_wall_hour: float | None
    unique_images_per_wall_hour: float | None
    coarse_unique_images_per_wall_hour: float | None
    dense_unique_images_per_wall_hour: float | None
    provider_images_per_wall_hour: float | None
    logical_calls_per_wall_hour: float | None
    call_parts_per_wall_hour: float | None
    call_splits_per_wall_hour: float | None
    http_requests_per_wall_hour: float | None
    retries_per_wall_hour: float | None
    batches_per_wall_hour: float | None
    batch_requests_per_wall_hour: float | None
    input_tokens_per_wall_hour: float | None
    output_tokens_per_wall_hour: float | None
    output_token_responses_per_wall_hour: float | None
    dense_logical_calls_per_wall_hour: float | None
    dense_provider_images_per_wall_hour: float | None
    windows_per_recording_hour: float | None
    logical_calls_per_recording_hour: float | None
    call_parts_per_recording_hour: float | None
    unique_images_per_camera_hour: float | None
    provider_images_per_camera_hour: float | None
    effective_fps_per_camera: float | None
    provider_images_per_unique_image: float | None
    logical_calls_per_provider_image: float | None
    call_splits_per_logical_call: float | None
    call_parts_per_logical_call: float | None
    logical_calls_per_window: float | None
    windows_per_batch: float | None
    logical_calls_per_batch: float | None
    http_requests_per_logical_call: float | None
    retries_per_http_request: float | None
    requests_per_batch: float | None
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.measurement_status, MeasuredCapacityStatus):
            raise TypeError("measurement_status must be a MeasuredCapacityStatus")
        if tuple(sorted(set(self.unavailable_reasons))) != self.unavailable_reasons:
            raise ValueError("unavailable_reasons must be unique and ordered")
        _require_nonempty("workload_fingerprint", self.workload_fingerprint)
        if not isinstance(self.evidence_class, CapacityEvidenceClass):
            raise TypeError("evidence_class must be a CapacityEvidenceClass")
        if not isinstance(self.provider_mode, ProviderMode):
            raise TypeError("provider_mode must be a ProviderMode")
        _require_positive_int("recording_count", self.recording_count)
        _require_positive_int("recording_worker_count", self.recording_worker_count)
        _require_positive_int("camera_count", self.camera_count)
        if self.recording_duration_ns is not None:
            _require_positive_int("recording_duration_ns", self.recording_duration_ns)
        _require_nonnegative_int("wall_time_ns", self.wall_time_ns)
        if self.production_eligible:
            raise ValueError("capacity calculations cannot grant production eligibility")
        if self.measurement_status is MeasuredCapacityStatus.AVAILABLE:
            if self.unavailable_reasons:
                raise ValueError("available capacity must not carry unavailable reasons")
            if (
                self.recording_duration_ns is None
                or self.provider_mode is ProviderMode.UNKNOWN
                or self.wall_time_ns <= 0
            ):
                raise ValueError(
                    "available capacity requires duration, provider mode, and wall time"
                )
        else:
            rate_fields = (
                self.recording_hours,
                self.camera_hours,
                self.wall_hours,
                self.recording_hours_per_wall_hour,
                self.camera_hours_per_wall_hour,
                self.windows_per_wall_hour,
                self.unique_images_per_wall_hour,
                self.coarse_unique_images_per_wall_hour,
                self.dense_unique_images_per_wall_hour,
                self.provider_images_per_wall_hour,
                self.logical_calls_per_wall_hour,
                self.call_parts_per_wall_hour,
                self.call_splits_per_wall_hour,
                self.http_requests_per_wall_hour,
                self.retries_per_wall_hour,
                self.batches_per_wall_hour,
                self.batch_requests_per_wall_hour,
                self.input_tokens_per_wall_hour,
                self.output_tokens_per_wall_hour,
                self.output_token_responses_per_wall_hour,
                self.dense_logical_calls_per_wall_hour,
                self.dense_provider_images_per_wall_hour,
                self.windows_per_recording_hour,
                self.logical_calls_per_recording_hour,
                self.call_parts_per_recording_hour,
                self.unique_images_per_camera_hour,
                self.provider_images_per_camera_hour,
                self.effective_fps_per_camera,
            )
            if any(value is not None for value in rate_fields):
                raise ValueError("unavailable capacity must not report rates")


@dataclass(frozen=True, slots=True)
class MeasuredCapacityRateRatio:
    """One named rate observed on both sides of a capacity comparison."""

    name: str
    baseline_value: float | None
    candidate_value: float | None
    candidate_to_baseline_ratio: float | None

    def __post_init__(self) -> None:
        _require_nonempty("name", self.name)
        for field_name in ("baseline_value", "candidate_value", "candidate_to_baseline_ratio"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field_name} must be finite or None")


@dataclass(frozen=True, slots=True)
class MeasuredCapacityComparison:
    """Non-promotional comparison with explicit compatibility and unit ratios."""

    comparison_kind: MeasuredCapacityComparisonKind
    comparable: bool
    non_comparable_reasons: tuple[str, ...]
    baseline_workload_fingerprint: str
    candidate_workload_fingerprint: str
    baseline_execution_mode: str
    candidate_execution_mode: str
    baseline_provider_mode: ProviderMode
    candidate_provider_mode: ProviderMode
    baseline_recording_worker_count: int
    candidate_recording_worker_count: int
    rate_ratios: tuple[MeasuredCapacityRateRatio, ...]
    recording_hours_per_wall_hour_ratio: float | None
    camera_hours_per_wall_hour_ratio: float | None
    unique_images_per_wall_hour_ratio: float | None
    provider_images_per_wall_hour_ratio: float | None
    logical_calls_per_wall_hour_ratio: float | None
    call_splits_per_wall_hour_ratio: float | None
    http_requests_per_wall_hour_ratio: float | None
    retries_per_wall_hour_ratio: float | None
    batches_per_wall_hour_ratio: float | None
    batch_requests_per_wall_hour_ratio: float | None
    input_tokens_per_wall_hour_ratio: float | None
    output_tokens_per_wall_hour_ratio: float | None
    output_token_responses_per_wall_hour_ratio: float | None
    dense_logical_calls_per_wall_hour_ratio: float | None
    dense_provider_images_per_wall_hour_ratio: float | None
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.comparison_kind, MeasuredCapacityComparisonKind):
            raise TypeError("comparison_kind must be a MeasuredCapacityComparisonKind")
        if not isinstance(self.baseline_provider_mode, ProviderMode):
            raise TypeError("baseline_provider_mode must be a ProviderMode")
        if not isinstance(self.candidate_provider_mode, ProviderMode):
            raise TypeError("candidate_provider_mode must be a ProviderMode")
        if tuple(sorted(set(self.non_comparable_reasons))) != self.non_comparable_reasons:
            raise ValueError("non_comparable_reasons must be unique and ordered")
        if self.comparable and self.non_comparable_reasons:
            raise ValueError("comparable capacity comparison cannot carry reasons")
        rate_ratio_names = tuple(item.name for item in self.rate_ratios)
        if rate_ratio_names != tuple(sorted(rate_ratio_names)) or len(set(rate_ratio_names)) != len(
            rate_ratio_names
        ):
            raise ValueError("rate_ratios must be unique and ordered")
        if not self.comparable:
            ratios = (
                self.recording_hours_per_wall_hour_ratio,
                self.camera_hours_per_wall_hour_ratio,
                self.unique_images_per_wall_hour_ratio,
                self.provider_images_per_wall_hour_ratio,
                self.logical_calls_per_wall_hour_ratio,
                self.call_splits_per_wall_hour_ratio,
                self.http_requests_per_wall_hour_ratio,
                self.retries_per_wall_hour_ratio,
                self.batches_per_wall_hour_ratio,
                self.batch_requests_per_wall_hour_ratio,
                self.input_tokens_per_wall_hour_ratio,
                self.output_tokens_per_wall_hour_ratio,
                self.output_token_responses_per_wall_hour_ratio,
                self.dense_logical_calls_per_wall_hour_ratio,
                self.dense_provider_images_per_wall_hour_ratio,
            )
            if any(value is not None for value in ratios) or any(
                item.candidate_to_baseline_ratio is not None for item in self.rate_ratios
            ):
                raise ValueError("non-comparable capacity comparisons must not report ratios")
        if self.production_eligible:
            raise ValueError("capacity comparisons cannot grant production eligibility")


def build_measured_capacity_report(measurement: MeasuredCapacityInput) -> MeasuredCapacityReport:
    """Convert explicit measurement facts into rates only when their units are known."""

    if not isinstance(measurement, MeasuredCapacityInput):
        raise TypeError("measurement must be a MeasuredCapacityInput")
    reasons: list[str] = []
    if measurement.recording_duration_ns is None:
        reasons.append("MISSING_WORKLOAD_DURATION")
    if measurement.provider_mode is ProviderMode.UNKNOWN:
        reasons.append("MISSING_PROVIDER_MODE")
    if measurement.wall_time_ns <= 0:
        reasons.append("MISSING_WALL_TIME")
    if reasons:
        return _unavailable_measured_capacity(measurement, tuple(sorted(reasons)))

    # The early return above establishes the denominator required by the rate
    # calculations, but mypy cannot infer that from a dynamically built list.
    assert measurement.recording_duration_ns is not None

    recording_hours = (
        measurement.recording_count * measurement.recording_duration_ns / _NANOSECONDS_PER_HOUR
    )
    camera_hours = recording_hours * measurement.camera_count
    wall_hours = measurement.wall_time_ns / _NANOSECONDS_PER_HOUR
    assert wall_hours > 0
    call_parts = measurement.call_parts
    if (
        call_parts is None
        and measurement.logical_calls is not None
        and measurement.call_splits is not None
    ):
        call_parts = measurement.logical_calls + measurement.call_splits
    unique_images_per_camera_hour = _rate_per_hour(measurement.unique_images, camera_hours)
    provider_images_per_camera_hour = _rate_per_hour(measurement.provider_images, camera_hours)
    return MeasuredCapacityReport(
        measurement_status=MeasuredCapacityStatus.AVAILABLE,
        unavailable_reasons=(),
        workload_fingerprint=measurement.workload_fingerprint,
        evidence_class=measurement.evidence_class,
        provider_mode=measurement.provider_mode,
        execution_mode=measurement.execution_mode,
        recording_count=measurement.recording_count,
        recording_worker_count=measurement.recording_worker_count,
        camera_count=measurement.camera_count,
        recording_duration_ns=measurement.recording_duration_ns,
        wall_time_ns=measurement.wall_time_ns,
        recording_hours=recording_hours,
        camera_hours=camera_hours,
        wall_hours=wall_hours,
        recording_hours_per_wall_hour=recording_hours / wall_hours,
        camera_hours_per_wall_hour=camera_hours / wall_hours,
        windows=measurement.windows,
        unique_images=measurement.unique_images,
        coarse_unique_images=measurement.coarse_unique_images,
        dense_unique_images=measurement.dense_unique_images,
        provider_images=measurement.provider_images,
        logical_calls=measurement.logical_calls,
        call_parts=call_parts,
        call_splits=measurement.call_splits,
        http_requests=measurement.http_requests,
        retries=measurement.retries,
        batches=measurement.batches,
        batch_requests=measurement.batch_requests,
        input_tokens=measurement.input_tokens,
        output_tokens=measurement.output_tokens,
        output_token_responses=measurement.output_token_responses,
        dense_logical_calls=measurement.dense_logical_calls,
        dense_provider_images=measurement.dense_provider_images,
        dense_logical_call_fraction=_fraction(
            measurement.dense_logical_calls,
            measurement.logical_calls,
        ),
        dense_upgrade_fraction=_fraction(
            measurement.dense_unique_images,
            measurement.unique_images,
        ),
        dense_provider_image_fraction=_fraction(
            measurement.dense_provider_images,
            measurement.provider_images,
        ),
        windows_per_wall_hour=_rate_per_hour(measurement.windows, wall_hours),
        unique_images_per_wall_hour=_rate_per_hour(measurement.unique_images, wall_hours),
        coarse_unique_images_per_wall_hour=_rate_per_hour(
            measurement.coarse_unique_images,
            wall_hours,
        ),
        dense_unique_images_per_wall_hour=_rate_per_hour(
            measurement.dense_unique_images,
            wall_hours,
        ),
        provider_images_per_wall_hour=_rate_per_hour(measurement.provider_images, wall_hours),
        logical_calls_per_wall_hour=_rate_per_hour(measurement.logical_calls, wall_hours),
        call_parts_per_wall_hour=_rate_per_hour(call_parts, wall_hours),
        call_splits_per_wall_hour=_rate_per_hour(measurement.call_splits, wall_hours),
        http_requests_per_wall_hour=_rate_per_hour(measurement.http_requests, wall_hours),
        retries_per_wall_hour=_rate_per_hour(measurement.retries, wall_hours),
        batches_per_wall_hour=_rate_per_hour(measurement.batches, wall_hours),
        batch_requests_per_wall_hour=_rate_per_hour(measurement.batch_requests, wall_hours),
        input_tokens_per_wall_hour=_rate_per_hour(measurement.input_tokens, wall_hours),
        output_tokens_per_wall_hour=_rate_per_hour(measurement.output_tokens, wall_hours),
        output_token_responses_per_wall_hour=_rate_per_hour(
            measurement.output_token_responses,
            wall_hours,
        ),
        dense_logical_calls_per_wall_hour=_rate_per_hour(
            measurement.dense_logical_calls,
            wall_hours,
        ),
        dense_provider_images_per_wall_hour=_rate_per_hour(
            measurement.dense_provider_images,
            wall_hours,
        ),
        windows_per_recording_hour=_rate_per_hour(measurement.windows, recording_hours),
        logical_calls_per_recording_hour=_rate_per_hour(
            measurement.logical_calls,
            recording_hours,
        ),
        call_parts_per_recording_hour=_rate_per_hour(call_parts, recording_hours),
        unique_images_per_camera_hour=unique_images_per_camera_hour,
        provider_images_per_camera_hour=provider_images_per_camera_hour,
        effective_fps_per_camera=(
            None
            if unique_images_per_camera_hour is None
            else unique_images_per_camera_hour / _SECONDS_PER_HOUR
        ),
        provider_images_per_unique_image=_ratio_optional(
            measurement.provider_images,
            measurement.unique_images,
        ),
        logical_calls_per_provider_image=_ratio_optional(
            measurement.logical_calls,
            measurement.provider_images,
        ),
        call_splits_per_logical_call=_ratio_optional(
            measurement.call_splits,
            measurement.logical_calls,
        ),
        call_parts_per_logical_call=_ratio_optional(call_parts, measurement.logical_calls),
        logical_calls_per_window=_ratio_optional(measurement.logical_calls, measurement.windows),
        windows_per_batch=_ratio_optional(measurement.windows, measurement.batches),
        logical_calls_per_batch=_ratio_optional(measurement.logical_calls, measurement.batches),
        http_requests_per_logical_call=_ratio_optional(
            measurement.http_requests,
            measurement.logical_calls,
        ),
        retries_per_http_request=_ratio_optional(measurement.retries, measurement.http_requests),
        requests_per_batch=_ratio_optional(measurement.batch_requests, measurement.batches),
    )


def compare_measured_capacity_reports(
    baseline: MeasuredCapacityReport,
    candidate: MeasuredCapacityReport,
    *,
    additional_non_comparable_reasons: tuple[str, ...] = (),
) -> MeasuredCapacityComparison:
    """Compare compatible measured profile reports without inferring promotion."""

    if not isinstance(baseline, MeasuredCapacityReport):
        raise TypeError("baseline must be a MeasuredCapacityReport")
    if not isinstance(candidate, MeasuredCapacityReport):
        raise TypeError("candidate must be a MeasuredCapacityReport")
    if not isinstance(additional_non_comparable_reasons, tuple):
        raise TypeError("additional_non_comparable_reasons must be a tuple")
    for reason in additional_non_comparable_reasons:
        _require_nonempty("additional_non_comparable_reasons", reason)
    comparison_kind = _comparison_kind(baseline, candidate)
    reasons: list[str] = list(additional_non_comparable_reasons)
    if baseline.workload_fingerprint != candidate.workload_fingerprint:
        reasons.append("WORKLOAD_FINGERPRINT_CHANGED")
    # A replay intentionally avoids repeating a fresh run's provider work.  Preserve
    # that mode difference in the source reports, but allow its explicit comparison
    # topology to show zero provider work and stage/resource ratios.
    if baseline.provider_mode != candidate.provider_mode and comparison_kind not in {
        MeasuredCapacityComparisonKind.FRESH_VS_REPLAY,
        MeasuredCapacityComparisonKind.MIXED,
    }:
        reasons.append("PROVIDER_MODE_CHANGED")
    if baseline.evidence_class != candidate.evidence_class:
        reasons.append("EVIDENCE_CLASS_CHANGED")
    if baseline.measurement_status is not MeasuredCapacityStatus.AVAILABLE:
        reasons.append("BASELINE_CAPACITY_UNAVAILABLE")
    if candidate.measurement_status is not MeasuredCapacityStatus.AVAILABLE:
        reasons.append("CANDIDATE_CAPACITY_UNAVAILABLE")
    reasons = sorted(set(reasons))
    return MeasuredCapacityComparison(
        comparison_kind=comparison_kind,
        comparable=not reasons,
        non_comparable_reasons=tuple(reasons),
        baseline_workload_fingerprint=baseline.workload_fingerprint,
        candidate_workload_fingerprint=candidate.workload_fingerprint,
        baseline_execution_mode=baseline.execution_mode,
        candidate_execution_mode=candidate.execution_mode,
        baseline_provider_mode=baseline.provider_mode,
        candidate_provider_mode=candidate.provider_mode,
        baseline_recording_worker_count=baseline.recording_worker_count,
        candidate_recording_worker_count=candidate.recording_worker_count,
        rate_ratios=_measured_capacity_rate_ratios(
            baseline,
            candidate,
            comparable=not reasons,
        ),
        recording_hours_per_wall_hour_ratio=(
            _ratio_float(
                candidate.recording_hours_per_wall_hour, baseline.recording_hours_per_wall_hour
            )
            if not reasons
            else None
        ),
        camera_hours_per_wall_hour_ratio=(
            _ratio_float(candidate.camera_hours_per_wall_hour, baseline.camera_hours_per_wall_hour)
            if not reasons
            else None
        ),
        unique_images_per_wall_hour_ratio=(
            _ratio_float(
                candidate.unique_images_per_wall_hour, baseline.unique_images_per_wall_hour
            )
            if not reasons
            else None
        ),
        provider_images_per_wall_hour_ratio=(
            _ratio_float(
                candidate.provider_images_per_wall_hour, baseline.provider_images_per_wall_hour
            )
            if not reasons
            else None
        ),
        logical_calls_per_wall_hour_ratio=(
            _ratio_float(
                candidate.logical_calls_per_wall_hour, baseline.logical_calls_per_wall_hour
            )
            if not reasons
            else None
        ),
        call_splits_per_wall_hour_ratio=(
            _ratio_float(candidate.call_splits_per_wall_hour, baseline.call_splits_per_wall_hour)
            if not reasons
            else None
        ),
        http_requests_per_wall_hour_ratio=(
            _ratio_float(
                candidate.http_requests_per_wall_hour, baseline.http_requests_per_wall_hour
            )
            if not reasons
            else None
        ),
        retries_per_wall_hour_ratio=(
            _ratio_float(candidate.retries_per_wall_hour, baseline.retries_per_wall_hour)
            if not reasons
            else None
        ),
        batches_per_wall_hour_ratio=(
            _ratio_float(candidate.batches_per_wall_hour, baseline.batches_per_wall_hour)
            if not reasons
            else None
        ),
        batch_requests_per_wall_hour_ratio=(
            _ratio_float(
                candidate.batch_requests_per_wall_hour,
                baseline.batch_requests_per_wall_hour,
            )
            if not reasons
            else None
        ),
        input_tokens_per_wall_hour_ratio=(
            _ratio_float(candidate.input_tokens_per_wall_hour, baseline.input_tokens_per_wall_hour)
            if not reasons
            else None
        ),
        output_tokens_per_wall_hour_ratio=(
            _ratio_float(
                candidate.output_tokens_per_wall_hour, baseline.output_tokens_per_wall_hour
            )
            if not reasons
            else None
        ),
        output_token_responses_per_wall_hour_ratio=(
            _ratio_float(
                candidate.output_token_responses_per_wall_hour,
                baseline.output_token_responses_per_wall_hour,
            )
            if not reasons
            else None
        ),
        dense_logical_calls_per_wall_hour_ratio=(
            _ratio_float(
                candidate.dense_logical_calls_per_wall_hour,
                baseline.dense_logical_calls_per_wall_hour,
            )
            if not reasons
            else None
        ),
        dense_provider_images_per_wall_hour_ratio=(
            _ratio_float(
                candidate.dense_provider_images_per_wall_hour,
                baseline.dense_provider_images_per_wall_hour,
            )
            if not reasons
            else None
        ),
    )


def _measured_capacity_rate_ratios(
    baseline: MeasuredCapacityReport,
    candidate: MeasuredCapacityReport,
    *,
    comparable: bool,
) -> tuple[MeasuredCapacityRateRatio, ...]:
    names = (
        "batch_requests_per_wall_hour",
        "batches_per_wall_hour",
        "call_parts_per_recording_hour",
        "call_parts_per_wall_hour",
        "call_splits_per_wall_hour",
        "camera_hours_per_wall_hour",
        "coarse_unique_images_per_wall_hour",
        "dense_logical_calls_per_wall_hour",
        "dense_provider_images_per_wall_hour",
        "dense_unique_images_per_wall_hour",
        "effective_fps_per_camera",
        "http_requests_per_wall_hour",
        "input_tokens_per_wall_hour",
        "logical_calls_per_recording_hour",
        "logical_calls_per_wall_hour",
        "output_token_responses_per_wall_hour",
        "output_tokens_per_wall_hour",
        "provider_images_per_camera_hour",
        "provider_images_per_wall_hour",
        "recording_hours_per_wall_hour",
        "retries_per_wall_hour",
        "unique_images_per_camera_hour",
        "unique_images_per_wall_hour",
        "windows_per_recording_hour",
        "windows_per_wall_hour",
    )
    return tuple(
        MeasuredCapacityRateRatio(
            name=name,
            baseline_value=getattr(baseline, name),
            candidate_value=getattr(candidate, name),
            candidate_to_baseline_ratio=(
                _ratio_float(getattr(candidate, name), getattr(baseline, name))
                if comparable
                else None
            ),
        )
        for name in names
    )


def _unavailable_measured_capacity(
    measurement: MeasuredCapacityInput,
    reasons: tuple[str, ...],
) -> MeasuredCapacityReport:
    call_parts = measurement.call_parts
    if (
        call_parts is None
        and measurement.logical_calls is not None
        and measurement.call_splits is not None
    ):
        call_parts = measurement.logical_calls + measurement.call_splits
    return MeasuredCapacityReport(
        measurement_status=MeasuredCapacityStatus.NOT_AVAILABLE,
        unavailable_reasons=reasons,
        workload_fingerprint=measurement.workload_fingerprint,
        evidence_class=measurement.evidence_class,
        provider_mode=measurement.provider_mode,
        execution_mode=measurement.execution_mode,
        recording_count=measurement.recording_count,
        recording_worker_count=measurement.recording_worker_count,
        camera_count=measurement.camera_count,
        recording_duration_ns=measurement.recording_duration_ns,
        wall_time_ns=measurement.wall_time_ns,
        recording_hours=None,
        camera_hours=None,
        wall_hours=None,
        recording_hours_per_wall_hour=None,
        camera_hours_per_wall_hour=None,
        windows=measurement.windows,
        unique_images=measurement.unique_images,
        coarse_unique_images=measurement.coarse_unique_images,
        dense_unique_images=measurement.dense_unique_images,
        provider_images=measurement.provider_images,
        logical_calls=measurement.logical_calls,
        call_parts=call_parts,
        call_splits=measurement.call_splits,
        http_requests=measurement.http_requests,
        retries=measurement.retries,
        batches=measurement.batches,
        batch_requests=measurement.batch_requests,
        input_tokens=measurement.input_tokens,
        output_tokens=measurement.output_tokens,
        output_token_responses=measurement.output_token_responses,
        dense_logical_calls=measurement.dense_logical_calls,
        dense_provider_images=measurement.dense_provider_images,
        dense_logical_call_fraction=_fraction(
            measurement.dense_logical_calls,
            measurement.logical_calls,
        ),
        dense_upgrade_fraction=_fraction(
            measurement.dense_unique_images,
            measurement.unique_images,
        ),
        dense_provider_image_fraction=_fraction(
            measurement.dense_provider_images,
            measurement.provider_images,
        ),
        windows_per_wall_hour=None,
        unique_images_per_wall_hour=None,
        coarse_unique_images_per_wall_hour=None,
        dense_unique_images_per_wall_hour=None,
        provider_images_per_wall_hour=None,
        logical_calls_per_wall_hour=None,
        call_parts_per_wall_hour=None,
        call_splits_per_wall_hour=None,
        http_requests_per_wall_hour=None,
        retries_per_wall_hour=None,
        batches_per_wall_hour=None,
        batch_requests_per_wall_hour=None,
        input_tokens_per_wall_hour=None,
        output_tokens_per_wall_hour=None,
        output_token_responses_per_wall_hour=None,
        dense_logical_calls_per_wall_hour=None,
        dense_provider_images_per_wall_hour=None,
        windows_per_recording_hour=None,
        logical_calls_per_recording_hour=None,
        call_parts_per_recording_hour=None,
        unique_images_per_camera_hour=None,
        provider_images_per_camera_hour=None,
        effective_fps_per_camera=None,
        provider_images_per_unique_image=_ratio_optional(
            measurement.provider_images,
            measurement.unique_images,
        ),
        logical_calls_per_provider_image=_ratio_optional(
            measurement.logical_calls,
            measurement.provider_images,
        ),
        call_splits_per_logical_call=_ratio_optional(
            measurement.call_splits,
            measurement.logical_calls,
        ),
        call_parts_per_logical_call=_ratio_optional(call_parts, measurement.logical_calls),
        logical_calls_per_window=_ratio_optional(measurement.logical_calls, measurement.windows),
        windows_per_batch=_ratio_optional(measurement.windows, measurement.batches),
        logical_calls_per_batch=_ratio_optional(measurement.logical_calls, measurement.batches),
        http_requests_per_logical_call=_ratio_optional(
            measurement.http_requests,
            measurement.logical_calls,
        ),
        retries_per_http_request=_ratio_optional(measurement.retries, measurement.http_requests),
        requests_per_batch=_ratio_optional(measurement.batch_requests, measurement.batches),
    )


def _rate_per_hour(value: int | None, wall_hours: float) -> float | None:
    return None if value is None else value / wall_hours


def _fraction(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    return numerator / denominator


def _ratio_optional(candidate: int | None, baseline: int | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return candidate / baseline


def _ratio_float(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0.0:
        return None
    return candidate / baseline


def _comparison_kind(
    baseline: MeasuredCapacityReport,
    candidate: MeasuredCapacityReport,
) -> MeasuredCapacityComparisonKind:
    execution_modes = {baseline.execution_mode, candidate.execution_mode}
    fresh_replay = execution_modes == {"FRESH", "REPLAY"}
    scaling = baseline.recording_worker_count != candidate.recording_worker_count
    if fresh_replay and scaling:
        return MeasuredCapacityComparisonKind.MIXED
    if fresh_replay:
        return MeasuredCapacityComparisonKind.FRESH_VS_REPLAY
    if scaling:
        return MeasuredCapacityComparisonKind.RECORDING_WORKER_SCALING
    return MeasuredCapacityComparisonKind.LIKE_FOR_LIKE


@dataclass(frozen=True, slots=True)
class SyntheticLoadProfile:
    """Versioned deterministic arrival and service pattern."""

    version: str
    unit_count: int
    recording_duration_ns: int
    camera_stream_durations_ns: tuple[int, ...]
    arrival_interval_ms: int
    arrival_batch_size: int
    service_time_pattern_ms: tuple[int, ...]
    deadline_budget_ms: int
    observation_window_ms: int | None = None
    failed_ordinals: tuple[int, ...] = ()
    skipped_ordinals: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("version", self.version)
        _require_positive_int("unit_count", self.unit_count)
        _require_positive_int("recording_duration_ns", self.recording_duration_ns)
        if not self.camera_stream_durations_ns:
            raise ValueError("camera_stream_durations_ns must be nonempty")
        for duration in self.camera_stream_durations_ns:
            _require_positive_int("camera stream duration", duration)
        _require_nonnegative_int("arrival_interval_ms", self.arrival_interval_ms)
        _require_positive_int("arrival_batch_size", self.arrival_batch_size)
        if not self.service_time_pattern_ms:
            raise ValueError("service_time_pattern_ms must be nonempty")
        for service_time in self.service_time_pattern_ms:
            _require_positive_int("service time", service_time)
        _require_positive_int("deadline_budget_ms", self.deadline_budget_ms)
        if self.observation_window_ms is not None:
            _require_positive_int("observation_window_ms", self.observation_window_ms)
        failed = _validate_ordinals("failed_ordinals", self.failed_ordinals, self.unit_count)
        skipped = _validate_ordinals("skipped_ordinals", self.skipped_ordinals, self.unit_count)
        if failed.intersection(skipped):
            raise ValueError("failed_ordinals and skipped_ordinals must be disjoint")

    @property
    def profile_digest(self) -> str:
        """Content address the complete synthetic workload policy."""

        return semantic_sha256(
            {
                "version": self.version,
                "unit_count": self.unit_count,
                "recording_duration_ns": str(self.recording_duration_ns),
                "camera_stream_durations_ns": [
                    str(value) for value in self.camera_stream_durations_ns
                ],
                "arrival_interval_ms": self.arrival_interval_ms,
                "arrival_batch_size": self.arrival_batch_size,
                "service_time_pattern_ms": list(self.service_time_pattern_ms),
                "deadline_budget_ms": self.deadline_budget_ms,
                "observation_window_ms": self.observation_window_ms,
                "failed_ordinals": sorted(self.failed_ordinals),
                "skipped_ordinals": sorted(self.skipped_ordinals),
            }
        )


@dataclass(frozen=True, slots=True)
class SyntheticWorkUnit:
    """One generated unit with explicit arrival, service, and deadline."""

    work_id: str
    ordinal: int
    arrival_at_ms: int
    service_time_ms: int
    deadline_at_ms: int
    recording_duration_ns: int
    camera_video_duration_ns: int
    planned_outcome: SyntheticOutcome


@dataclass(frozen=True, slots=True)
class SyntheticObservation:
    """Scheduled and cutoff-observed lifecycle for one synthetic unit."""

    work: SyntheticWorkUnit
    worker_ordinal: int | None
    started_at_ms: int | None
    scheduled_completion_at_ms: int
    observed_terminal_at_ms: int | None
    outcome: SyntheticOutcome
    queue_wait_ms: int
    service_time_ms: int
    wall_time_ms: int | None
    deadline_missed: bool


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Nearest-rank latency statistics over one explicit population."""

    count: int
    mean_ms: float
    p50_ms: int
    p95_ms: int
    p99_ms: int


@dataclass(frozen=True, slots=True)
class SyntheticCapacityReport:
    """Reconciled local simulation report that cannot claim measured capacity."""

    profile_version: str
    profile_digest: str
    worker_count: int
    cutoff_ms: int
    observations: tuple[SyntheticObservation, ...]
    succeeded_count: int
    failed_count: int
    skipped_count: int
    pending_count: int
    deadline_miss_count: int
    backlog_peak: int
    backlog_end: int
    utilization: float
    offered_recording_hours: float
    offered_camera_video_hours: float
    completed_recording_hours: float
    completed_camera_video_hours: float
    recording_hours_per_wall_hour: float
    camera_video_hours_per_wall_hour: float
    offered_units_per_wall_hour: float
    nominal_service_capacity_units_per_hour: float
    queue_wait: LatencySummary | None
    service_time: LatencySummary | None
    wall_time: LatencySummary | None
    bottlenecks: tuple[BottleneckKind, ...]

    @property
    def evidence_class(self) -> str:
        return "SYNTHETIC_LOCAL"

    @property
    def measurement_status(self) -> str:
        return "NOT_MEASURED"

    @property
    def production_eligible(self) -> bool:
        return False

    @property
    def total_count(self) -> int:
        return self.succeeded_count + self.failed_count + self.skipped_count + self.pending_count

    @property
    def terminal_failure_rate(self) -> float:
        terminal_attempts = self.succeeded_count + self.failed_count
        return 0.0 if terminal_attempts == 0 else self.failed_count / terminal_attempts

    @property
    def skipped_rate(self) -> float:
        return 0.0 if self.total_count == 0 else self.skipped_count / self.total_count

    @property
    def deadline_miss_rate(self) -> float:
        eligible = self.succeeded_count + self.failed_count + self.pending_count
        return 0.0 if eligible == 0 else self.deadline_miss_count / eligible


@dataclass(frozen=True, slots=True)
class WorkerScalingPoint:
    '''One local observation in a recording-worker scaling matrix.'''

    worker_count: int
    report: SyntheticCapacityReport
    drain_report: SyntheticCapacityReport
    throughput_ratio: float | None
    queue_capacity: int | None
    queue_bounded: bool | None

    def __post_init__(self) -> None:
        _require_positive_int('worker_count', self.worker_count)
        if not isinstance(self.report, SyntheticCapacityReport):
            raise TypeError('report must be a SyntheticCapacityReport')
        if not isinstance(self.drain_report, SyntheticCapacityReport):
            raise TypeError('drain_report must be a SyntheticCapacityReport')
        if self.report.worker_count != self.worker_count:
            raise ValueError('scaling point worker_count does not match report')
        if self.drain_report.worker_count != self.worker_count:
            raise ValueError('scaling point worker_count does not match drain report')
        if self.report.profile_digest != self.drain_report.profile_digest:
            raise ValueError('scaling point reports must share a workload profile')
        if self.throughput_ratio is not None and (
            isinstance(self.throughput_ratio, bool)
            or not isinstance(self.throughput_ratio, (int, float))
            or math.isnan(self.throughput_ratio)
        ):
            raise ValueError('throughput_ratio must be finite or infinity')
        if self.queue_capacity is not None:
            _require_positive_int('queue_capacity', self.queue_capacity)
            if not isinstance(self.queue_bounded, bool):
                raise TypeError('queue_bounded must be boolean when queue_capacity is set')
        elif self.queue_bounded is not None:
            raise ValueError('queue_bounded requires queue_capacity')

    @property
    def capacity(self) -> SyntheticCapacityReport:
        '''Compatibility alias for callers that call the observation capacity.'''

        return self.report

    @property
    def speedup(self) -> float | None:
        '''Throughput speedup relative to the one-worker baseline.'''

        return self.throughput_ratio

    @property
    def backlog_peak(self) -> int:
        return self.report.backlog_peak

    @property
    def backlog_end(self) -> int:
        return self.report.backlog_end

    @property
    def drain_backlog_end(self) -> int:
        return self.drain_report.backlog_end

    @property
    def backlog_drained(self) -> bool:
        return self.drain_backlog_end == 0

    @property
    def bottlenecks(self) -> tuple[BottleneckKind, ...]:
        return self.report.bottlenecks

    @property
    def saturated(self) -> bool:
        return BottleneckKind.SERVICE_CAPACITY in self.bottlenecks

    def as_dict(self) -> dict[str, object]:
        return {
            'worker_count': self.worker_count,
            'throughput_ratio': self.throughput_ratio,
            'recording_hours_per_wall_hour': self.report.recording_hours_per_wall_hour,
            'camera_video_hours_per_wall_hour': self.report.camera_video_hours_per_wall_hour,
            'backlog_peak': self.backlog_peak,
            'backlog_end': self.backlog_end,
            'drain_backlog_end': self.drain_backlog_end,
            'backlog_drained': self.backlog_drained,
            'queue_capacity': self.queue_capacity,
            'queue_bounded': self.queue_bounded,
            'bottlenecks': [item.value for item in self.bottlenecks],
            'evidence_class': self.report.evidence_class,
            'measurement_status': self.report.measurement_status,
            'production_eligible': self.report.production_eligible,
        }


@dataclass(frozen=True, slots=True)
class WorkerScalingReport:
    '''Local 1/2/4/N worker scaling and saturation evidence.'''

    profile_version: str
    profile_digest: str
    worker_counts: tuple[int, ...]
    points: tuple[WorkerScalingPoint, ...]
    target_recording_rtf: float
    capacity_projection: WorkerCapacityProjection | None
    queue_capacity: int | None
    four_worker_speedup: float | None

    def __post_init__(self) -> None:
        _require_nonempty('profile_version', self.profile_version)
        _require_nonempty('profile_digest', self.profile_digest)
        if not isinstance(self.worker_counts, tuple) or not self.worker_counts:
            raise ValueError('worker_counts must be a nonempty tuple')
        if self.worker_counts != tuple(sorted(set(self.worker_counts))):
            raise ValueError('worker_counts must be sorted and unique')
        for count in self.worker_counts:
            _require_positive_int('worker count', count)
        if self.worker_counts[0] != 1:
            raise ValueError('worker_counts must include one-worker baseline')
        if not isinstance(self.points, tuple) or tuple(
            item.worker_count for item in self.points
        ) != self.worker_counts:
            raise ValueError('scaling points must match worker_counts')
        for point in self.points:
            if point.report.profile_digest != self.profile_digest:
                raise ValueError('scaling point profile digest does not match report')
        _require_finite_positive('target_recording_rtf', self.target_recording_rtf)
        if self.capacity_projection is not None and not isinstance(
            self.capacity_projection, WorkerCapacityProjection
        ):
            raise TypeError('capacity_projection must be a WorkerCapacityProjection or None')
        if self.queue_capacity is not None:
            _require_positive_int('queue_capacity', self.queue_capacity)
            if any(point.queue_capacity != self.queue_capacity for point in self.points):
                raise ValueError('scaling points must use report queue_capacity')
        elif any(point.queue_capacity is not None for point in self.points):
            raise ValueError('scaling points cannot carry an unbound queue capacity')
        if self.four_worker_speedup is not None and (
            isinstance(self.four_worker_speedup, bool)
            or not isinstance(self.four_worker_speedup, (int, float))
            or math.isnan(self.four_worker_speedup)
        ):
            raise ValueError('four_worker_speedup must be finite or infinity')

    @property
    def evidence_class(self) -> str:
        return 'SYNTHETIC_LOCAL'

    @property
    def measurement_status(self) -> str:
        return 'NOT_MEASURED'

    @property
    def qualification_status(self) -> str:
        return 'NOT_PRODUCTION_QUALIFIED'

    @property
    def production_eligible(self) -> bool:
        return False

    @property
    def baseline(self) -> WorkerScalingPoint:
        return self.points[0]

    @property
    def four_worker_point(self) -> WorkerScalingPoint | None:
        return next((point for point in self.points if point.worker_count == 4), None)

    @property
    def four_worker_meets_2_5x(self) -> bool | None:
        if self.four_worker_speedup is None:
            return None
        return self.four_worker_speedup >= 2.5

    @property
    def backlog_drains_after_burst(self) -> bool:
        return all(point.backlog_drained for point in self.points)

    @property
    def queues_bounded(self) -> bool | None:
        if self.queue_capacity is None:
            return None
        return all(point.queue_bounded for point in self.points)

    @property
    def saturation_worker_count(self) -> int | None:
        for point in self.points:
            if point.saturated:
                return point.worker_count
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            'profile_version': self.profile_version,
            'profile_digest': self.profile_digest,
            'worker_counts': list(self.worker_counts),
            'target_recording_rtf': self.target_recording_rtf,
            'capacity_projection': (
                None if self.capacity_projection is None else self.capacity_projection.as_dict()
            ),
            'queue_capacity': self.queue_capacity,
            'queues_bounded': self.queues_bounded,
            'backlog_drains_after_burst': self.backlog_drains_after_burst,
            'four_worker_speedup': self.four_worker_speedup,
            'four_worker_meets_2_5x': self.four_worker_meets_2_5x,
            'saturation_worker_count': self.saturation_worker_count,
            'points': [point.as_dict() for point in self.points],
            'evidence_class': self.evidence_class,
            'measurement_status': self.measurement_status,
            'qualification_status': self.qualification_status,
            'production_eligible': self.production_eligible,
        }


@dataclass(frozen=True, slots=True)
class LocalSloPolicy:
    """Versioned local thresholds; evaluation never grants promotion."""

    version: str
    maximum_terminal_failure_rate: float
    maximum_skipped_rate: float
    maximum_deadline_miss_rate: float
    maximum_p95_wall_time_ms: int
    require_empty_backlog: bool = True

    def __post_init__(self) -> None:
        _require_nonempty("version", self.version)
        _require_unit_interval(
            "maximum_terminal_failure_rate",
            self.maximum_terminal_failure_rate,
        )
        _require_unit_interval("maximum_skipped_rate", self.maximum_skipped_rate)
        _require_unit_interval(
            "maximum_deadline_miss_rate",
            self.maximum_deadline_miss_rate,
        )
        _require_positive_int("maximum_p95_wall_time_ms", self.maximum_p95_wall_time_ms)
        if not isinstance(self.require_empty_backlog, bool):
            raise TypeError("require_empty_backlog must be boolean")

    @property
    def policy_digest(self) -> str:
        return semantic_sha256(
            {
                "version": self.version,
                "maximum_terminal_failure_rate": self.maximum_terminal_failure_rate,
                "maximum_skipped_rate": self.maximum_skipped_rate,
                "maximum_deadline_miss_rate": self.maximum_deadline_miss_rate,
                "maximum_p95_wall_time_ms": self.maximum_p95_wall_time_ms,
                "require_empty_backlog": self.require_empty_backlog,
            }
        )


@dataclass(frozen=True, slots=True)
class LocalSloEvaluation:
    """Non-promotional comparison of one synthetic report to local thresholds."""

    policy_version: str
    policy_digest: str
    profile_digest: str
    within_local_thresholds: bool
    violations: tuple[str, ...]

    @property
    def measurement_status(self) -> str:
        return "NOT_MEASURED"

    @property
    def production_eligible(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class CapacityRegressionPolicy:
    """Versioned bounds for like-for-like local performance comparisons."""

    version: str
    minimum_throughput_ratio: float
    maximum_p95_wall_time_ratio: float
    maximum_failure_rate_increase: float
    maximum_deadline_miss_rate_increase: float
    maximum_backlog_end_increase: int

    def __post_init__(self) -> None:
        _require_nonempty("version", self.version)
        if (
            isinstance(self.minimum_throughput_ratio, bool)
            or not isinstance(self.minimum_throughput_ratio, (int, float))
            or not math.isfinite(self.minimum_throughput_ratio)
            or not 0 < self.minimum_throughput_ratio <= 1
        ):
            raise ValueError("minimum_throughput_ratio must be in (0, 1]")
        if (
            isinstance(self.maximum_p95_wall_time_ratio, bool)
            or not isinstance(self.maximum_p95_wall_time_ratio, (int, float))
            or not math.isfinite(self.maximum_p95_wall_time_ratio)
            or self.maximum_p95_wall_time_ratio < 1
        ):
            raise ValueError("maximum_p95_wall_time_ratio must be finite and at least 1")
        _require_unit_interval(
            "maximum_failure_rate_increase",
            self.maximum_failure_rate_increase,
        )
        _require_unit_interval(
            "maximum_deadline_miss_rate_increase",
            self.maximum_deadline_miss_rate_increase,
        )
        _require_nonnegative_int(
            "maximum_backlog_end_increase",
            self.maximum_backlog_end_increase,
        )

    @property
    def policy_digest(self) -> str:
        return semantic_sha256(
            {
                "version": self.version,
                "minimum_throughput_ratio": self.minimum_throughput_ratio,
                "maximum_p95_wall_time_ratio": self.maximum_p95_wall_time_ratio,
                "maximum_failure_rate_increase": self.maximum_failure_rate_increase,
                "maximum_deadline_miss_rate_increase": (self.maximum_deadline_miss_rate_increase),
                "maximum_backlog_end_increase": self.maximum_backlog_end_increase,
            }
        )


@dataclass(frozen=True, slots=True)
class CapacityRegressionResult:
    """Like-for-like local comparison with explicit non-promotional semantics."""

    policy_version: str
    policy_digest: str
    profile_digest: str
    within_local_thresholds: bool
    regressions: tuple[str, ...]
    throughput_ratio: float | None
    p95_wall_time_ratio: float | None
    failure_rate_increase: float
    deadline_miss_rate_increase: float
    backlog_end_increase: int

    @property
    def measurement_status(self) -> str:
        return "NOT_MEASURED"

    @property
    def production_eligible(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class WorkerCapacityProjection:
    '''Translate one observed worker rate.'''

    recording_rtf: float
    target_recording_rtf: float
    required_worker_count: int

    def __post_init__(self) -> None:
        _require_finite_positive('recording_rtf', self.recording_rtf)
        _require_finite_positive('target_recording_rtf', self.target_recording_rtf)
        _require_positive_int('required_worker_count', self.required_worker_count)

    @property
    def required_cpu_worker_count(self) -> int:
        return self.required_worker_count

    @property
    def required_nvme_worker_count(self) -> int:
        return self.required_worker_count

    @property
    def evidence_class(self) -> str:
        return 'SYNTHETIC_LOCAL'

    @property
    def measurement_status(self) -> str:
        return 'NOT_MEASURED'

    @property
    def production_eligible(self) -> bool:
        return False

    def as_dict(self) -> dict[str, object]:
        return {
            'recording_rtf': self.recording_rtf,
            'target_recording_rtf': self.target_recording_rtf,
            'required_worker_count': self.required_worker_count,
            'required_cpu_worker_count': self.required_cpu_worker_count,
            'required_nvme_worker_count': self.required_nvme_worker_count,
            'evidence_class': self.evidence_class,
            'measurement_status': self.measurement_status,
            'production_eligible': self.production_eligible,
        }


def generate_synthetic_load(
    profile: SyntheticLoadProfile,
) -> tuple[SyntheticWorkUnit, ...]:
    """Generate stable IDs and burst-shaped arrivals without random state."""

    if not isinstance(profile, SyntheticLoadProfile):
        raise TypeError("profile must be SyntheticLoadProfile")
    failed = frozenset(profile.failed_ordinals)
    skipped = frozenset(profile.skipped_ordinals)
    camera_video_duration_ns = sum(profile.camera_stream_durations_ns)
    profile_digest = profile.profile_digest
    units: list[SyntheticWorkUnit] = []
    for ordinal in range(profile.unit_count):
        arrival_at_ms = (ordinal // profile.arrival_batch_size) * profile.arrival_interval_ms
        if ordinal in skipped:
            planned_outcome = SyntheticOutcome.SKIPPED
        elif ordinal in failed:
            planned_outcome = SyntheticOutcome.FAILED
        else:
            planned_outcome = SyntheticOutcome.SUCCEEDED
        units.append(
            SyntheticWorkUnit(
                work_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"robata:synthetic-capacity:{profile_digest}:{ordinal}",
                    )
                ),
                ordinal=ordinal,
                arrival_at_ms=arrival_at_ms,
                service_time_ms=profile.service_time_pattern_ms[
                    ordinal % len(profile.service_time_pattern_ms)
                ],
                deadline_at_ms=arrival_at_ms + profile.deadline_budget_ms,
                recording_duration_ns=profile.recording_duration_ns,
                camera_video_duration_ns=camera_video_duration_ns,
                planned_outcome=planned_outcome,
            )
        )
    return tuple(units)


def simulate_capacity(
    profile: SyntheticLoadProfile,
    *,
    worker_count: int,
    cutoff_ms: int | None = None,
) -> SyntheticCapacityReport:
    """Run a deterministic discrete-event queue simulation to one explicit cutoff."""

    if not isinstance(profile, SyntheticLoadProfile):
        raise TypeError("profile must be SyntheticLoadProfile")
    _require_positive_int("worker_count", worker_count)
    units = generate_synthetic_load(profile)
    if cutoff_ms is not None:
        _require_positive_int('cutoff_ms', cutoff_ms)
        if cutoff_ms < units[-1].arrival_at_ms:
            raise ValueError('cutoff_ms must include the complete arrival schedule')
    if (
        profile.observation_window_ms is not None
        and profile.observation_window_ms < units[-1].arrival_at_ms
    ):
        raise ValueError("observation_window_ms must include the complete arrival schedule")
    workers = [(0, ordinal) for ordinal in range(worker_count)]
    heapq.heapify(workers)
    scheduled: list[tuple[SyntheticWorkUnit, int | None, int | None, int, int, int]] = []
    busy_ms = 0
    for unit in units:
        if unit.planned_outcome is SyntheticOutcome.SKIPPED:
            scheduled.append((unit, None, None, unit.arrival_at_ms, 0, 0))
            continue
        available_at_ms, worker_ordinal = heapq.heappop(workers)
        started_at_ms = max(unit.arrival_at_ms, available_at_ms)
        completion_at_ms = started_at_ms + unit.service_time_ms
        heapq.heappush(workers, (completion_at_ms, worker_ordinal))
        scheduled.append(
            (
                unit,
                worker_ordinal,
                started_at_ms,
                completion_at_ms,
                started_at_ms - unit.arrival_at_ms,
                unit.service_time_ms,
            )
        )

    natural_cutoff = max(item[3] for item in scheduled)
    cutoff_ms = (
        cutoff_ms
        if cutoff_ms is not None
        else (
            profile.observation_window_ms
            if profile.observation_window_ms is not None
            else max(1, natural_cutoff)
        )
    )
    observations: list[SyntheticObservation] = []
    for (
        unit,
        scheduled_worker_ordinal,
        scheduled_start_ms,
        completion_at_ms,
        queue_ms,
        service_ms,
    ) in scheduled:
        if unit.planned_outcome is SyntheticOutcome.SKIPPED:
            outcome = SyntheticOutcome.SKIPPED
            observed_terminal_at_ms: int | None = unit.arrival_at_ms
            wall_time_ms: int | None = 0
        elif completion_at_ms > cutoff_ms:
            outcome = SyntheticOutcome.PENDING
            observed_terminal_at_ms = None
            wall_time_ms = None
        else:
            outcome = unit.planned_outcome
            observed_terminal_at_ms = completion_at_ms
            wall_time_ms = completion_at_ms - unit.arrival_at_ms
        cutoff_started_at_ms = (
            scheduled_start_ms
            if scheduled_start_ms is not None and scheduled_start_ms <= cutoff_ms
            else None
        )
        cutoff_worker_ordinal = (
            scheduled_worker_ordinal if cutoff_started_at_ms is not None else None
        )
        if unit.planned_outcome is SyntheticOutcome.SKIPPED:
            queue_wait_at_cutoff_ms = 0
        elif cutoff_started_at_ms is not None:
            queue_wait_at_cutoff_ms = queue_ms
        else:
            queue_wait_at_cutoff_ms = max(0, cutoff_ms - unit.arrival_at_ms)
        deadline_missed = (
            cutoff_ms >= unit.deadline_at_ms and completion_at_ms > unit.deadline_at_ms
        )
        if scheduled_start_ms is not None:
            busy_ms += max(0, min(completion_at_ms, cutoff_ms) - scheduled_start_ms)
        observations.append(
            SyntheticObservation(
                work=unit,
                worker_ordinal=cutoff_worker_ordinal,
                started_at_ms=cutoff_started_at_ms,
                scheduled_completion_at_ms=completion_at_ms,
                observed_terminal_at_ms=observed_terminal_at_ms,
                outcome=outcome,
                queue_wait_ms=queue_wait_at_cutoff_ms,
                service_time_ms=service_ms,
                wall_time_ms=wall_time_ms,
                deadline_missed=deadline_missed,
            )
        )

    counts = {
        outcome: sum(item.outcome is outcome for item in observations)
        for outcome in SyntheticOutcome
    }
    succeeded = tuple(item for item in observations if item.outcome is SyntheticOutcome.SUCCEEDED)
    terminal_work = tuple(
        item
        for item in observations
        if item.outcome in (SyntheticOutcome.SUCCEEDED, SyntheticOutcome.FAILED)
    )
    elapsed_hours = cutoff_ms / _MILLISECONDS_PER_HOUR
    offered_recording_hours = (
        sum(item.work.recording_duration_ns for item in observations) / _NANOSECONDS_PER_HOUR
    )
    offered_camera_video_hours = (
        sum(item.work.camera_video_duration_ns for item in observations) / _NANOSECONDS_PER_HOUR
    )
    completed_recording_hours = (
        sum(item.work.recording_duration_ns for item in succeeded) / _NANOSECONDS_PER_HOUR
    )
    completed_camera_video_hours = (
        sum(item.work.camera_video_duration_ns for item in succeeded) / _NANOSECONDS_PER_HOUR
    )
    service_demand = tuple(
        item for item in observations if item.work.planned_outcome is not SyntheticOutcome.SKIPPED
    )
    mean_service_ms = (
        sum(item.service_time_ms for item in service_demand) / len(service_demand)
        if service_demand
        else None
    )
    queue_wait = _latency_summary(tuple(item.queue_wait_ms for item in terminal_work))
    service_time = _latency_summary(tuple(item.service_time_ms for item in terminal_work))
    wall_time = _latency_summary(
        tuple(item.wall_time_ms for item in terminal_work if item.wall_time_ms is not None)
    )
    backlog_peak = _backlog_peak(observations, cutoff_ms)
    backlog_end = counts[SyntheticOutcome.PENDING]
    bottlenecks: list[BottleneckKind] = []
    offered_rate = len(observations) / elapsed_hours
    nominal_capacity = (
        0.0 if mean_service_ms is None else worker_count * _MILLISECONDS_PER_HOUR / mean_service_ms
    )
    service_arrival_rate = len(service_demand) / elapsed_hours
    if backlog_end > 0 or (
        service_demand and service_arrival_rate > nominal_capacity * (1 + 1e-12)
    ):
        bottlenecks.append(BottleneckKind.SERVICE_CAPACITY)
    if (
        queue_wait is not None
        and service_time is not None
        and queue_wait.p95_ms > service_time.p95_ms
    ):
        bottlenecks.append(BottleneckKind.QUEUE_WAIT)
    if counts[SyntheticOutcome.FAILED] > 0:
        bottlenecks.append(BottleneckKind.RELIABILITY)
    if any(item.deadline_missed for item in observations):
        bottlenecks.append(BottleneckKind.DEADLINE)
    if not bottlenecks:
        bottlenecks.append(BottleneckKind.NONE)

    report = SyntheticCapacityReport(
        profile_version=profile.version,
        profile_digest=profile.profile_digest,
        worker_count=worker_count,
        cutoff_ms=cutoff_ms,
        observations=tuple(observations),
        succeeded_count=counts[SyntheticOutcome.SUCCEEDED],
        failed_count=counts[SyntheticOutcome.FAILED],
        skipped_count=counts[SyntheticOutcome.SKIPPED],
        pending_count=counts[SyntheticOutcome.PENDING],
        deadline_miss_count=sum(item.deadline_missed for item in observations),
        backlog_peak=backlog_peak,
        backlog_end=backlog_end,
        utilization=min(1.0, busy_ms / (worker_count * cutoff_ms)),
        offered_recording_hours=offered_recording_hours,
        offered_camera_video_hours=offered_camera_video_hours,
        completed_recording_hours=completed_recording_hours,
        completed_camera_video_hours=completed_camera_video_hours,
        recording_hours_per_wall_hour=completed_recording_hours / elapsed_hours,
        camera_video_hours_per_wall_hour=completed_camera_video_hours / elapsed_hours,
        offered_units_per_wall_hour=offered_rate,
        nominal_service_capacity_units_per_hour=nominal_capacity,
        queue_wait=queue_wait,
        service_time=service_time,
        wall_time=wall_time,
        bottlenecks=tuple(bottlenecks),
    )
    if report.total_count != profile.unit_count:
        raise AssertionError("synthetic outcome ledger does not reconcile")
    return report


def required_worker_count_for_rtf(
    recording_rtf: float,
    *,
    target_recording_rtf: float = 25.0,
) -> int:
    '''Return the ceiling worker count for a measured per-worker recording RTF.'''

    _require_finite_positive('recording_rtf', recording_rtf)
    _require_finite_positive('target_recording_rtf', target_recording_rtf)
    return max(1, math.ceil(target_recording_rtf / recording_rtf))


def run_worker_scaling(
    profile: SyntheticLoadProfile,
    *,
    worker_counts: tuple[int, ...] = (1, 2, 4),
    target_recording_rtf: float = 25.0,
    queue_capacity: int | None = None,
) -> WorkerScalingReport:
    '''Run a deterministic 1/2/4/N worker matrix with backlog and drain evidence.'''

    if not isinstance(profile, SyntheticLoadProfile):
        raise TypeError('profile must be SyntheticLoadProfile')
    if not isinstance(worker_counts, tuple):
        raise TypeError('worker_counts must be a tuple')
    if not worker_counts:
        raise ValueError('worker_counts must be nonempty')
    if worker_counts != tuple(sorted(set(worker_counts))):
        raise ValueError('worker_counts must be sorted and unique')
    for count in worker_counts:
        _require_positive_int('worker count', count)
    if worker_counts[0] != 1:
        raise ValueError('worker_counts must include one-worker baseline')
    _require_finite_positive('target_recording_rtf', target_recording_rtf)
    if queue_capacity is not None:
        _require_positive_int('queue_capacity', queue_capacity)

    measured = {
        count: simulate_capacity(profile, worker_count=count)
        for count in worker_counts
    }
    points: list[WorkerScalingPoint] = []
    baseline_rate = measured[1].recording_hours_per_wall_hour
    for count in worker_counts:
        report = measured[count]
        natural_cutoff = max(
            observation.scheduled_completion_at_ms for observation in report.observations
        )
        drain_report = simulate_capacity(
            profile,
            worker_count=count,
            cutoff_ms=max(natural_cutoff, report.cutoff_ms),
        )
        bounded = None if queue_capacity is None else report.backlog_peak <= queue_capacity
        points.append(
            WorkerScalingPoint(
                worker_count=count,
                report=report,
                drain_report=drain_report,
                throughput_ratio=_ratio(
                    report.recording_hours_per_wall_hour,
                    baseline_rate,
                ),
                queue_capacity=queue_capacity,
                queue_bounded=bounded,
            )
        )

    # A projection is available only from an observed point that completed its
    # offered cutoff without a residual backlog and, when configured, respected
    # the declared queue bound.  This prevents a saturated point from silently
    # becoming a sizing claim.
    sustainable_rates = [
        point.report.recording_hours_per_wall_hour / point.worker_count
        for point in points
        if point.report.backlog_end == 0
        and (point.queue_bounded is None or point.queue_bounded)
        and point.report.recording_hours_per_wall_hour > 0
    ]
    per_worker_rate = max(sustainable_rates) if sustainable_rates else None
    projection = None
    if per_worker_rate is not None:
        projection = WorkerCapacityProjection(
            recording_rtf=per_worker_rate,
            target_recording_rtf=target_recording_rtf,
            required_worker_count=required_worker_count_for_rtf(
                per_worker_rate,
                target_recording_rtf=target_recording_rtf,
            ),
        )

    four_worker = next((point for point in points if point.worker_count == 4), None)
    return WorkerScalingReport(
        profile_version=profile.version,
        profile_digest=profile.profile_digest,
        worker_counts=worker_counts,
        points=tuple(points),
        target_recording_rtf=target_recording_rtf,
        capacity_projection=projection,
        queue_capacity=queue_capacity,
        four_worker_speedup=None if four_worker is None else four_worker.throughput_ratio,
    )


def build_worker_scaling_report(
    profile: SyntheticLoadProfile,
    *,
    worker_counts: tuple[int, ...] = (1, 2, 4),
    target_recording_rtf: float = 25.0,
    queue_capacity: int | None = None,
) -> WorkerScalingReport:
    '''Named builder retained for profile/report command integrations.'''

    return run_worker_scaling(
        profile,
        worker_counts=worker_counts,
        target_recording_rtf=target_recording_rtf,
        queue_capacity=queue_capacity,
    )


def evaluate_local_slo(
    report: SyntheticCapacityReport,
    policy: LocalSloPolicy,
) -> LocalSloEvaluation:
    """Compare synthetic observations without changing their evidence class."""

    if not isinstance(report, SyntheticCapacityReport):
        raise TypeError("report must be SyntheticCapacityReport")
    if not isinstance(policy, LocalSloPolicy):
        raise TypeError("policy must be LocalSloPolicy")
    violations: list[str] = []
    if report.terminal_failure_rate > policy.maximum_terminal_failure_rate:
        violations.append("TERMINAL_FAILURE_RATE")
    if report.skipped_rate > policy.maximum_skipped_rate:
        violations.append("SKIPPED_RATE")
    if report.deadline_miss_rate > policy.maximum_deadline_miss_rate:
        violations.append("DEADLINE_MISS_RATE")
    if report.wall_time is None or report.wall_time.p95_ms > policy.maximum_p95_wall_time_ms:
        violations.append("P95_WALL_TIME")
    if policy.require_empty_backlog and report.backlog_end != 0:
        violations.append("BACKLOG_NOT_DRAINED")
    return LocalSloEvaluation(
        policy_version=policy.version,
        policy_digest=policy.policy_digest,
        profile_digest=report.profile_digest,
        within_local_thresholds=not violations,
        violations=tuple(violations),
    )


def compare_capacity_reports(
    baseline: SyntheticCapacityReport,
    candidate: SyntheticCapacityReport,
    policy: CapacityRegressionPolicy,
) -> CapacityRegressionResult:
    """Detect local regressions only for the same content-addressed workload."""

    if not isinstance(baseline, SyntheticCapacityReport):
        raise TypeError("baseline must be SyntheticCapacityReport")
    if not isinstance(candidate, SyntheticCapacityReport):
        raise TypeError("candidate must be SyntheticCapacityReport")
    if not isinstance(policy, CapacityRegressionPolicy):
        raise TypeError("policy must be CapacityRegressionPolicy")
    if baseline.profile_digest != candidate.profile_digest:
        raise ValueError("capacity regression comparison requires the same workload profile")

    throughput_ratio = _ratio(
        candidate.recording_hours_per_wall_hour,
        baseline.recording_hours_per_wall_hour,
    )
    p95_wall_time_ratio = _optional_latency_ratio(
        candidate.wall_time,
        baseline.wall_time,
    )
    failure_rate_increase = candidate.terminal_failure_rate - baseline.terminal_failure_rate
    deadline_miss_rate_increase = candidate.deadline_miss_rate - baseline.deadline_miss_rate
    backlog_end_increase = candidate.backlog_end - baseline.backlog_end
    regressions: list[str] = []
    if throughput_ratio is None or throughput_ratio < policy.minimum_throughput_ratio:
        regressions.append("THROUGHPUT")
    if p95_wall_time_ratio is None or p95_wall_time_ratio > policy.maximum_p95_wall_time_ratio:
        regressions.append("P95_WALL_TIME")
    if failure_rate_increase > policy.maximum_failure_rate_increase:
        regressions.append("TERMINAL_FAILURE_RATE")
    if deadline_miss_rate_increase > policy.maximum_deadline_miss_rate_increase:
        regressions.append("DEADLINE_MISS_RATE")
    if backlog_end_increase > policy.maximum_backlog_end_increase:
        regressions.append("BACKLOG_END")
    return CapacityRegressionResult(
        policy_version=policy.version,
        policy_digest=policy.policy_digest,
        profile_digest=baseline.profile_digest,
        within_local_thresholds=not regressions,
        regressions=tuple(regressions),
        throughput_ratio=throughput_ratio,
        p95_wall_time_ratio=p95_wall_time_ratio,
        failure_rate_increase=failure_rate_increase,
        deadline_miss_rate_increase=deadline_miss_rate_increase,
        backlog_end_increase=backlog_end_increase,
    )


def _latency_summary(values: tuple[int, ...]) -> LatencySummary | None:
    if not values:
        return None
    return LatencySummary(
        count=len(values),
        mean_ms=sum(values) / len(values),
        p50_ms=_nearest_rank(values, 0.50),
        p95_ms=_nearest_rank(values, 0.95),
        p99_ms=_nearest_rank(values, 0.99),
    )


def _nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _backlog_peak(
    observations: list[SyntheticObservation],
    cutoff_ms: int,
) -> int:
    events: list[tuple[int, int]] = []
    for item in observations:
        if item.work.planned_outcome is SyntheticOutcome.SKIPPED:
            continue
        if item.work.arrival_at_ms <= cutoff_ms:
            events.append((item.work.arrival_at_ms, 1))
        if item.scheduled_completion_at_ms <= cutoff_ms:
            events.append((item.scheduled_completion_at_ms, -1))
    current = 0
    peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def _ratio(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return 1.0 if candidate == 0 else math.inf
    return candidate / baseline


def _optional_latency_ratio(
    candidate: LatencySummary | None,
    baseline: LatencySummary | None,
) -> float | None:
    if candidate is None or baseline is None:
        return None
    if baseline.p95_ms == 0:
        return 1.0 if candidate.p95_ms == 0 else None
    return candidate.p95_ms / baseline.p95_ms


def _require_finite_positive(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f'{name} must be finite and positive')


def _require_nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_unit_interval(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _validate_ordinals(
    name: str,
    values: tuple[int, ...],
    unit_count: int,
) -> frozenset[int]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    for value in values:
        _require_nonnegative_int(name, value)
        if value >= unit_count:
            raise ValueError(f"{name} contains an ordinal outside the workload")
    return frozenset(values)


__all__ = [
    "BottleneckKind",
    "CapacityEvidenceClass",
    "CapacityRegressionPolicy",
    "CapacityRegressionResult",
    "LatencySummary",
    "LocalSloEvaluation",
    "LocalSloPolicy",
    "MeasuredCapacityComparison",
    "MeasuredCapacityComparisonKind",
    "MeasuredCapacityInput",
    "MeasuredCapacityRateRatio",
    "MeasuredCapacityReport",
    "MeasuredCapacityStatus",
    "ProviderMode",
    "SyntheticCapacityReport",
    "SyntheticLoadProfile",
    "SyntheticObservation",
    "SyntheticOutcome",
    "SyntheticWorkUnit",
    'WorkerCapacityProjection',
    'WorkerScalingPoint',
    'WorkerScalingReport',
    "build_measured_capacity_report",
    'build_worker_scaling_report',
    "compare_capacity_reports",
    "compare_measured_capacity_reports",
    "evaluate_local_slo",
    "generate_synthetic_load",
    'required_worker_count_for_rtf',
    'run_worker_scaling',
    "simulate_capacity",
]
