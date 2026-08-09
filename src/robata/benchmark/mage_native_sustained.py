"""Pure analysis for serial-versus-prefetch Mage native qualification.

The module consumes already-collected, monotonic-offset telemetry. It never imports a
provider runtime, opens media, talks to a GPU, or performs network I/O. The resulting
report is local qualification evidence only: it can prove that two runs were compatible,
fresh, and met an explicit performance policy, but it cannot make a production release
claim.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

MAGE_NATIVE_SUSTAINED_REPORT_VERSION: Final[Literal["mage-native-sustained-comparison-v1"]] = (
    "mage-native-sustained-comparison-v1"
)


class MageNativeTelemetryDisposition(StrEnum):
    """Whether a sidecar row represents fresh generation or artifact replay."""

    FRESH_GENERATION = "FRESH_GENERATION"
    ARTIFACT_REPLAY = "ARTIFACT_REPLAY"


class MageNativeQualificationStatus(StrEnum):
    """Aggregate local qualification verdict."""

    PASSED = "PASSED"
    FAILED = "FAILED"


class MageNativeTimeInterval(StrictModel):
    """A non-empty monotonic-time interval relative to run start."""

    start_seconds: NonNegativeFiniteFloat
    end_seconds: PositiveFiniteFloat

    @model_validator(mode="after")
    def validate_nonempty(self) -> Self:
        if self.start_seconds >= self.end_seconds:
            raise ValueError("start_seconds must be less than end_seconds")
        return self

    @property
    def duration_seconds(self) -> float:
        """Return the interval duration."""

        return self.end_seconds - self.start_seconds


def merge_intervals(
    intervals: Iterable[MageNativeTimeInterval],
) -> tuple[MageNativeTimeInterval, ...]:
    """Return a sorted, non-overlapping union of the supplied intervals."""

    ordered = sorted(intervals, key=lambda interval: (interval.start_seconds, interval.end_seconds))
    if not ordered:
        return ()

    merged: list[MageNativeTimeInterval] = []
    start = ordered[0].start_seconds
    end = ordered[0].end_seconds
    for interval in ordered[1:]:
        if interval.start_seconds <= end:
            end = max(end, interval.end_seconds)
            continue
        merged.append(MageNativeTimeInterval(start_seconds=start, end_seconds=end))
        start = interval.start_seconds
        end = interval.end_seconds
    merged.append(MageNativeTimeInterval(start_seconds=start, end_seconds=end))
    return tuple(merged)


def interval_union_seconds(intervals: Iterable[MageNativeTimeInterval]) -> float:
    """Return the duration of the interval union without double-counting overlap."""

    return sum(interval.duration_seconds for interval in merge_intervals(intervals))


def interval_intersection_seconds(
    left: Iterable[MageNativeTimeInterval],
    right: Iterable[MageNativeTimeInterval],
) -> float:
    """Return the duration shared by the two interval unions."""

    left_union = merge_intervals(left)
    right_union = merge_intervals(right)
    left_index = 0
    right_index = 0
    intersection_seconds = 0.0
    while left_index < len(left_union) and right_index < len(right_union):
        left_interval = left_union[left_index]
        right_interval = right_union[right_index]
        start = max(left_interval.start_seconds, right_interval.start_seconds)
        end = min(left_interval.end_seconds, right_interval.end_seconds)
        if start < end:
            intersection_seconds += end - start
        if left_interval.end_seconds <= right_interval.end_seconds:
            left_index += 1
        else:
            right_index += 1
    return intersection_seconds


class MageNativeGenerationGapSummary(StrictModel):
    """Nearest-rank generation gap percentiles."""

    gap_count: NonNegativeInt
    total_seconds: NonNegativeFiniteFloat
    p50_seconds: NonNegativeFiniteFloat
    p95_seconds: NonNegativeFiniteFloat
    max_seconds: NonNegativeFiniteFloat


def _nearest_rank(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = tuple(sorted(values))
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def generation_gap_summary(
    intervals: Iterable[MageNativeTimeInterval],
) -> MageNativeGenerationGapSummary:
    """Measure idle gaps between generation intervals in monotonic-time order.

    Overlap is represented as a zero gap here and is reported independently by the run
    summary. This keeps gap percentiles defined even for invalid multi-flight telemetry.
    """

    ordered = sorted(intervals, key=lambda interval: (interval.start_seconds, interval.end_seconds))
    gaps: list[float] = []
    if ordered:
        previous_end = ordered[0].end_seconds
        for interval in ordered[1:]:
            gaps.append(max(0.0, interval.start_seconds - previous_end))
            previous_end = max(previous_end, interval.end_seconds)
    values = tuple(gaps)
    return MageNativeGenerationGapSummary(
        gap_count=len(values),
        total_seconds=sum(values),
        p50_seconds=_nearest_rank(values, 0.50),
        p95_seconds=_nearest_rank(values, 0.95),
        max_seconds=max(values, default=0.0),
    )


class MageNativeLatencySummary(StrictModel):
    """Nearest-rank latency percentiles for a bounded set of samples."""

    sample_count: NonNegativeInt
    p50_seconds: NonNegativeFiniteFloat
    p95_seconds: NonNegativeFiniteFloat
    max_seconds: NonNegativeFiniteFloat


def latency_summary(values: Iterable[float]) -> MageNativeLatencySummary:
    """Summarize nonnegative latency observations with nearest-rank percentiles."""

    samples = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("latency values must be finite and nonnegative")
    return MageNativeLatencySummary(
        sample_count=len(samples),
        p50_seconds=_nearest_rank(samples, 0.50),
        p95_seconds=_nearest_rank(samples, 0.95),
        max_seconds=max(samples, default=0.0),
    )


class MageNativeRunIdentity(StrictModel):
    """Like-for-like identity pins that must match between A/B arms."""

    model_identity_sha256: Sha256Digest
    checkpoint_sha256: Sha256Digest
    source_media_sha256: Sha256Digest
    segment_manifest_sha256: Sha256Digest
    prompt_sha256: Sha256Digest
    codec_policy_sha256: Sha256Digest
    camera_id: NonEmptyString


class MageNativeGenerationTelemetrySample(StrictModel):
    """One endpoint sidecar observation expressed as run-relative time offsets."""

    telemetry_event_version: NonEmptyString
    segment_ordinal: NonNegativeInt
    request_id: NonEmptyString
    inference_identity_sha256: Sha256Digest
    result_artifact_identity_sha256: Sha256Digest
    output_text_sha256: Sha256Digest
    disposition: MageNativeTelemetryDisposition = MageNativeTelemetryDisposition.FRESH_GENERATION
    request_interval: MageNativeTimeInterval | None = None
    processor_interval: MageNativeTimeInterval | None = None
    generation_interval: MageNativeTimeInterval | None = None
    decode_interval: MageNativeTimeInterval | None = None
    prompt_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    max_new_tokens: PositiveInt
    output_valid: bool
    time_to_first_token_seconds: NonNegativeFiniteFloat | None = None

    @model_validator(mode="after")
    def validate_phase_relationships(self) -> Self:
        if (
            self.time_to_first_token_seconds is not None
            and self.generation_interval is not None
            and self.time_to_first_token_seconds > self.generation_interval.duration_seconds
        ):
            raise ValueError("time_to_first_token_seconds cannot exceed generation duration")
        if self.request_interval is not None:
            for name, interval in (
                ("processor_interval", self.processor_interval),
                ("generation_interval", self.generation_interval),
                ("decode_interval", self.decode_interval),
            ):
                if interval is not None and (
                    interval.start_seconds < self.request_interval.start_seconds
                    or interval.end_seconds > self.request_interval.end_seconds
                ):
                    raise ValueError(f"{name} must be contained by request_interval")
        return self

    @property
    def exhausted_output_budget(self) -> bool:
        """Return whether generation reached its configured output-token ceiling."""

        return self.output_tokens >= self.max_new_tokens


class MageNativeRunMeasurement(StrictModel):
    """One serial or bounded-prefetch run and its retained sidecar rows."""

    measurement_version: Literal["mage-native-run-measurement-v1"] = (
        "mage-native-run-measurement-v1"
    )
    run_id: NonEmptyString
    execution_profile: NonEmptyString
    telemetry_event_version: NonEmptyString
    identity: MageNativeRunIdentity
    expected_segment_count: PositiveInt
    media_duration_seconds: PositiveFiniteFloat
    wall_seconds: PositiveFiniteFloat
    model_load_seconds: NonNegativeFiniteFloat = 0.0
    model_load_included_in_wall: bool = False
    telemetry: tuple[MageNativeGenerationTelemetrySample, ...]

    @model_validator(mode="after")
    def validate_intervals_within_wall(self) -> Self:
        for sample in self.telemetry:
            for name, interval in (
                ("request_interval", sample.request_interval),
                ("processor_interval", sample.processor_interval),
                ("generation_interval", sample.generation_interval),
                ("decode_interval", sample.decode_interval),
            ):
                if interval is not None and interval.end_seconds > self.wall_seconds:
                    raise ValueError(
                        f"telemetry segment {sample.segment_ordinal} {name} exceeds wall_seconds"
                    )
        return self


class MageNativeRunCompatibility(StrictModel):
    """Compatibility result for the two qualification arms."""

    compatible: bool
    mismatch_codes: tuple[NonEmptyString, ...]


class MageNativeFreshnessAssessment(StrictModel):
    """Cardinality, uniqueness, and replay-contamination checks for one arm."""

    passed: bool
    expected_segment_count: PositiveInt
    telemetry_count: NonNegativeInt
    fresh_generation_count: NonNegativeInt
    unique_request_id_count: NonNegativeInt
    unique_inference_identity_count: NonNegativeInt
    unique_result_artifact_identity_count: NonNegativeInt
    replay_count: NonNegativeInt
    missing_generation_interval_count: NonNegativeInt
    missing_segment_ordinals: tuple[NonNegativeInt, ...]
    unexpected_segment_ordinals: tuple[NonNegativeInt, ...]
    duplicate_segment_ordinals: tuple[NonNegativeInt, ...]
    issue_codes: tuple[NonEmptyString, ...]


class MageNativeCrossRunIsolationAssessment(StrictModel):
    """Reject copied sidecars or reuse of an arm's request/result artifacts."""

    passed: bool
    overlapping_request_id_count: NonNegativeInt
    overlapping_result_artifact_identity_count: NonNegativeInt
    issue_codes: tuple[NonEmptyString, ...]


class MageNativeRunSummary(StrictModel):
    """Non-overlapping timing and throughput summary for one run."""

    run_id: NonEmptyString
    execution_profile: NonEmptyString
    expected_segment_count: PositiveInt
    telemetry_count: NonNegativeInt
    media_duration_seconds: PositiveFiniteFloat
    wall_seconds: PositiveFiniteFloat
    model_load_seconds: NonNegativeFiniteFloat
    model_load_included_in_wall: bool
    generation_interval_count: NonNegativeInt
    generation_sum_seconds: NonNegativeFiniteFloat
    generation_union_seconds: NonNegativeFiniteFloat
    generation_overlap_seconds: NonNegativeFiniteFloat
    generation_gap: MageNativeGenerationGapSummary
    time_to_first_token: MageNativeLatencySummary
    generation_duty_cycle: UnitInterval
    wall_rtf: NonNegativeFiniteFloat
    output_tokens: NonNegativeInt
    output_tokens_per_generation_second: NonNegativeFiniteFloat
    output_tokens_per_wall_second: NonNegativeFiniteFloat
    processor_union_seconds: NonNegativeFiniteFloat
    processor_generation_overlap_seconds: NonNegativeFiniteFloat
    processor_overlap_fraction: UnitInterval
    invalid_output_count: NonNegativeInt
    output_budget_exhaustion_count: NonNegativeInt


class MageNativeSustainedQualificationPolicy(StrictModel):
    """Explicit local gates for accepting bounded prefetch over serial execution."""

    policy_version: Literal["mage-native-sustained-qualification-policy-v1"] = (
        "mage-native-sustained-qualification-policy-v1"
    )
    minimum_prefetch_speedup: PositiveFiniteFloat = 1.0
    minimum_prefetch_generation_duty_cycle: UnitInterval = 0.90
    minimum_prefetch_wall_rtf: NonNegativeFiniteFloat = 1.0
    maximum_prefetch_generation_gap_p95_seconds: NonNegativeFiniteFloat = 1.0
    generation_overlap_tolerance_seconds: NonNegativeFiniteFloat = 1e-9
    require_valid_outputs: bool = True
    reject_output_budget_exhaustion: bool = True


class MageNativeQualificationGate(StrictModel):
    """One stable machine-readable qualification decision."""

    gate_id: NonEmptyString
    passed: bool
    detail: NonEmptyString


class MageNativeSustainedComparisonReport(StrictModel):
    """Machine-serializable local A/B qualification report."""

    report_version: Literal["mage-native-sustained-comparison-v1"] = (
        MAGE_NATIVE_SUSTAINED_REPORT_VERSION
    )
    evidence_class: Literal["LOCAL_CONFORMANCE"] = "LOCAL_CONFORMANCE"
    serial_summary: MageNativeRunSummary
    prefetch_summary: MageNativeRunSummary
    compatibility: MageNativeRunCompatibility
    serial_freshness: MageNativeFreshnessAssessment
    prefetch_freshness: MageNativeFreshnessAssessment
    cross_run_isolation: MageNativeCrossRunIsolationAssessment
    prefetch_speedup: NonNegativeFiniteFloat
    policy: MageNativeSustainedQualificationPolicy
    gates: tuple[MageNativeQualificationGate, ...]
    qualification_status: MageNativeQualificationStatus
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_verdict(self) -> Self:
        expected = (
            MageNativeQualificationStatus.PASSED
            if all(gate.passed for gate in self.gates)
            else MageNativeQualificationStatus.FAILED
        )
        if self.qualification_status is not expected:
            raise ValueError("qualification_status must equal the aggregate gate result")
        return self

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report payload."""

        return self.model_dump(mode="json")


def _duplicates(values: Iterable[str | int]) -> tuple[str | int, ...]:
    counts = Counter(values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def assess_run_freshness(run: MageNativeRunMeasurement) -> MageNativeFreshnessAssessment:
    """Assess whether one arm contains exactly one fresh row per expected segment."""

    expected_ordinals = set(range(run.expected_segment_count))
    actual_ordinals = [sample.segment_ordinal for sample in run.telemetry]
    actual_ordinal_set = set(actual_ordinals)
    request_ids = [sample.request_id for sample in run.telemetry]
    inference_ids = [sample.inference_identity_sha256 for sample in run.telemetry]
    artifact_ids = [sample.result_artifact_identity_sha256 for sample in run.telemetry]
    replay_count = sum(
        sample.disposition is MageNativeTelemetryDisposition.ARTIFACT_REPLAY
        for sample in run.telemetry
    )
    fresh_generation_count = sum(
        sample.disposition is MageNativeTelemetryDisposition.FRESH_GENERATION
        for sample in run.telemetry
    )
    missing_generation_interval_count = sum(
        sample.generation_interval is None for sample in run.telemetry
    )
    duplicate_ordinals = tuple(int(value) for value in _duplicates(actual_ordinals))

    issues: list[str] = []
    if len(run.telemetry) != run.expected_segment_count:
        issues.append("TELEMETRY_CARDINALITY_MISMATCH")
    if actual_ordinal_set != expected_ordinals or duplicate_ordinals:
        issues.append("SEGMENT_ORDINAL_SET_MISMATCH")
    if len(set(request_ids)) != len(request_ids):
        issues.append("DUPLICATE_REQUEST_ID")
    if len(set(inference_ids)) != len(inference_ids):
        issues.append("DUPLICATE_INFERENCE_IDENTITY")
    if replay_count:
        issues.append("REPLAY_CONTAMINATION")
    if missing_generation_interval_count:
        issues.append("MISSING_GENERATION_INTERVAL")
    if any(
        sample.telemetry_event_version != run.telemetry_event_version for sample in run.telemetry
    ):
        issues.append("TELEMETRY_VERSION_MISMATCH")

    return MageNativeFreshnessAssessment(
        passed=not issues,
        expected_segment_count=run.expected_segment_count,
        telemetry_count=len(run.telemetry),
        fresh_generation_count=fresh_generation_count,
        unique_request_id_count=len(set(request_ids)),
        unique_inference_identity_count=len(set(inference_ids)),
        unique_result_artifact_identity_count=len(set(artifact_ids)),
        replay_count=replay_count,
        missing_generation_interval_count=missing_generation_interval_count,
        missing_segment_ordinals=tuple(sorted(expected_ordinals - actual_ordinal_set)),
        unexpected_segment_ordinals=tuple(sorted(actual_ordinal_set - expected_ordinals)),
        duplicate_segment_ordinals=duplicate_ordinals,
        issue_codes=tuple(issues),
    )


def assess_run_compatibility(
    serial: MageNativeRunMeasurement,
    prefetch: MageNativeRunMeasurement,
) -> MageNativeRunCompatibility:
    """Check that A/B arms differ only in their execution profile."""

    mismatches: list[str] = []
    identity_fields = (
        ("model_identity_sha256", "MODEL_IDENTITY_MISMATCH"),
        ("checkpoint_sha256", "CHECKPOINT_MISMATCH"),
        ("source_media_sha256", "SOURCE_MEDIA_MISMATCH"),
        ("segment_manifest_sha256", "SEGMENT_MANIFEST_MISMATCH"),
        ("prompt_sha256", "PROMPT_MISMATCH"),
        ("codec_policy_sha256", "CODEC_POLICY_MISMATCH"),
        ("camera_id", "CAMERA_MISMATCH"),
    )
    for field_name, mismatch_code in identity_fields:
        if getattr(serial.identity, field_name) != getattr(prefetch.identity, field_name):
            mismatches.append(mismatch_code)
    if serial.expected_segment_count != prefetch.expected_segment_count:
        mismatches.append("SEGMENT_COUNT_MISMATCH")
    if not math.isclose(
        serial.media_duration_seconds,
        prefetch.media_duration_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        mismatches.append("MEDIA_DURATION_MISMATCH")

    serial_inference = tuple(
        (sample.segment_ordinal, sample.inference_identity_sha256)
        for sample in sorted(serial.telemetry, key=lambda item: item.segment_ordinal)
    )
    prefetch_inference = tuple(
        (sample.segment_ordinal, sample.inference_identity_sha256)
        for sample in sorted(prefetch.telemetry, key=lambda item: item.segment_ordinal)
    )
    if serial_inference != prefetch_inference:
        mismatches.append("INFERENCE_IDENTITY_SEQUENCE_MISMATCH")

    serial_budgets = tuple(
        (sample.segment_ordinal, sample.max_new_tokens)
        for sample in sorted(serial.telemetry, key=lambda item: item.segment_ordinal)
    )
    prefetch_budgets = tuple(
        (sample.segment_ordinal, sample.max_new_tokens)
        for sample in sorted(prefetch.telemetry, key=lambda item: item.segment_ordinal)
    )
    if serial_budgets != prefetch_budgets:
        mismatches.append("DECODER_BUDGET_MISMATCH")

    return MageNativeRunCompatibility(
        compatible=not mismatches,
        mismatch_codes=tuple(mismatches),
    )


def assess_cross_run_isolation(
    serial: MageNativeRunMeasurement,
    prefetch: MageNativeRunMeasurement,
) -> MageNativeCrossRunIsolationAssessment:
    """Reject copied artifacts while allowing deterministic logical request IDs to match."""

    serial_request_ids = {sample.request_id for sample in serial.telemetry}
    prefetch_request_ids = {sample.request_id for sample in prefetch.telemetry}
    serial_artifact_ids = {sample.result_artifact_identity_sha256 for sample in serial.telemetry}
    prefetch_artifact_ids = {
        sample.result_artifact_identity_sha256 for sample in prefetch.telemetry
    }
    request_overlap = serial_request_ids & prefetch_request_ids
    artifact_overlap = serial_artifact_ids & prefetch_artifact_ids
    issues: list[str] = []
    if serial.run_id == prefetch.run_id:
        issues.append("RUN_ID_REUSED_BETWEEN_ARMS")
    if artifact_overlap:
        issues.append("RESULT_ARTIFACT_REUSED_BETWEEN_ARMS")
    return MageNativeCrossRunIsolationAssessment(
        passed=not issues,
        overlapping_request_id_count=len(request_overlap),
        overlapping_result_artifact_identity_count=len(artifact_overlap),
        issue_codes=tuple(issues),
    )


def summarize_run(run: MageNativeRunMeasurement) -> MageNativeRunSummary:
    """Build a non-overlapping timing and throughput summary."""

    generation_intervals = tuple(
        sample.generation_interval
        for sample in run.telemetry
        if sample.generation_interval is not None
    )
    processor_intervals = tuple(
        sample.processor_interval
        for sample in run.telemetry
        if sample.processor_interval is not None
    )
    generation_sum = sum(interval.duration_seconds for interval in generation_intervals)
    generation_union = interval_union_seconds(generation_intervals)
    generation_overlap = max(0.0, generation_sum - generation_union)
    processor_union = interval_union_seconds(processor_intervals)
    processor_generation_overlap = interval_intersection_seconds(
        processor_intervals, generation_intervals
    )
    output_tokens = sum(sample.output_tokens for sample in run.telemetry)
    return MageNativeRunSummary(
        run_id=run.run_id,
        execution_profile=run.execution_profile,
        expected_segment_count=run.expected_segment_count,
        telemetry_count=len(run.telemetry),
        media_duration_seconds=run.media_duration_seconds,
        wall_seconds=run.wall_seconds,
        model_load_seconds=run.model_load_seconds,
        model_load_included_in_wall=run.model_load_included_in_wall,
        generation_interval_count=len(generation_intervals),
        generation_sum_seconds=generation_sum,
        generation_union_seconds=generation_union,
        generation_overlap_seconds=generation_overlap,
        generation_gap=generation_gap_summary(generation_intervals),
        time_to_first_token=latency_summary(
            sample.time_to_first_token_seconds
            for sample in run.telemetry
            if sample.time_to_first_token_seconds is not None
        ),
        generation_duty_cycle=generation_union / run.wall_seconds,
        wall_rtf=run.media_duration_seconds / run.wall_seconds,
        output_tokens=output_tokens,
        output_tokens_per_generation_second=(
            output_tokens / generation_union if generation_union else 0.0
        ),
        output_tokens_per_wall_second=output_tokens / run.wall_seconds,
        processor_union_seconds=processor_union,
        processor_generation_overlap_seconds=processor_generation_overlap,
        processor_overlap_fraction=(
            processor_generation_overlap / processor_union if processor_union else 0.0
        ),
        invalid_output_count=sum(not sample.output_valid for sample in run.telemetry),
        output_budget_exhaustion_count=sum(
            sample.exhausted_output_budget for sample in run.telemetry
        ),
    )


def serial_vs_prefetch_speedup(
    serial: MageNativeRunMeasurement | MageNativeRunSummary,
    prefetch: MageNativeRunMeasurement | MageNativeRunSummary,
) -> float:
    """Return wall-clock speedup where values above one favor bounded prefetch."""

    return serial.wall_seconds / prefetch.wall_seconds


def _gate(gate_id: str, passed: bool, detail: str) -> MageNativeQualificationGate:
    return MageNativeQualificationGate(gate_id=gate_id, passed=passed, detail=detail)


def build_mage_native_sustained_comparison_report(
    *,
    serial: MageNativeRunMeasurement,
    prefetch: MageNativeRunMeasurement,
    policy: MageNativeSustainedQualificationPolicy | None = None,
) -> MageNativeSustainedComparisonReport:
    """Compare fresh serial and bounded-prefetch runs under explicit local gates."""

    active_policy = policy or MageNativeSustainedQualificationPolicy()
    serial_summary = summarize_run(serial)
    prefetch_summary = summarize_run(prefetch)
    compatibility = assess_run_compatibility(serial, prefetch)
    serial_freshness = assess_run_freshness(serial)
    prefetch_freshness = assess_run_freshness(prefetch)
    isolation = assess_cross_run_isolation(serial, prefetch)
    speedup = serial_vs_prefetch_speedup(serial_summary, prefetch_summary)

    serial_single_flight = (
        serial_summary.generation_overlap_seconds
        <= active_policy.generation_overlap_tolerance_seconds
    )
    prefetch_single_flight = (
        prefetch_summary.generation_overlap_seconds
        <= active_policy.generation_overlap_tolerance_seconds
    )
    valid_outputs = (
        serial_summary.invalid_output_count == 0 and prefetch_summary.invalid_output_count == 0
    )
    budget_clear = (
        serial_summary.output_budget_exhaustion_count == 0
        and prefetch_summary.output_budget_exhaustion_count == 0
    )
    serial_output_hashes = tuple(
        (sample.segment_ordinal, sample.output_text_sha256)
        for sample in sorted(serial.telemetry, key=lambda item: item.segment_ordinal)
    )
    prefetch_output_hashes = tuple(
        (sample.segment_ordinal, sample.output_text_sha256)
        for sample in sorted(prefetch.telemetry, key=lambda item: item.segment_ordinal)
    )
    output_text_hash_parity = serial_output_hashes == prefetch_output_hashes

    gates = (
        _gate(
            "LIKE_FOR_LIKE_COMPATIBILITY",
            compatibility.compatible,
            "compatible" if compatibility.compatible else ",".join(compatibility.mismatch_codes),
        ),
        _gate(
            "SERIAL_FRESH_TELEMETRY",
            serial_freshness.passed,
            "fresh" if serial_freshness.passed else ",".join(serial_freshness.issue_codes),
        ),
        _gate(
            "PREFETCH_FRESH_TELEMETRY",
            prefetch_freshness.passed,
            "fresh" if prefetch_freshness.passed else ",".join(prefetch_freshness.issue_codes),
        ),
        _gate(
            "CROSS_RUN_ISOLATION",
            isolation.passed,
            "isolated" if isolation.passed else ",".join(isolation.issue_codes),
        ),
        _gate(
            "SERIAL_SINGLE_GENERATION_IN_FLIGHT",
            serial_single_flight,
            (
                f"overlap={serial_summary.generation_overlap_seconds:.6f}s; "
                f"limit={active_policy.generation_overlap_tolerance_seconds:.6f}s"
            ),
        ),
        _gate(
            "PREFETCH_SINGLE_GENERATION_IN_FLIGHT",
            prefetch_single_flight,
            (
                f"overlap={prefetch_summary.generation_overlap_seconds:.6f}s; "
                f"limit={active_policy.generation_overlap_tolerance_seconds:.6f}s"
            ),
        ),
        _gate(
            "VALID_OUTPUTS",
            valid_outputs or not active_policy.require_valid_outputs,
            (
                f"serial_invalid={serial_summary.invalid_output_count}; "
                f"prefetch_invalid={prefetch_summary.invalid_output_count}"
            ),
        ),
        _gate(
            "OUTPUT_BUDGET_NOT_EXHAUSTED",
            budget_clear or not active_policy.reject_output_budget_exhaustion,
            (
                f"serial_exhausted={serial_summary.output_budget_exhaustion_count}; "
                f"prefetch_exhausted={prefetch_summary.output_budget_exhaustion_count}"
            ),
        ),
        _gate(
            "OUTPUT_TEXT_HASH_PARITY",
            output_text_hash_parity,
            "all segment output hashes match"
            if output_text_hash_parity
            else "output hashes differ",
        ),
        _gate(
            "PREFETCH_SPEEDUP",
            speedup >= active_policy.minimum_prefetch_speedup,
            f"actual={speedup:.6f}x; minimum={active_policy.minimum_prefetch_speedup:.6f}x",
        ),
        _gate(
            "PREFETCH_GENERATION_DUTY_CYCLE",
            (
                prefetch_summary.generation_duty_cycle
                >= active_policy.minimum_prefetch_generation_duty_cycle
            ),
            (
                f"actual={prefetch_summary.generation_duty_cycle:.6f}; "
                f"minimum={active_policy.minimum_prefetch_generation_duty_cycle:.6f}"
            ),
        ),
        _gate(
            "PREFETCH_WALL_RTF",
            prefetch_summary.wall_rtf >= active_policy.minimum_prefetch_wall_rtf,
            (
                f"actual={prefetch_summary.wall_rtf:.6f}x; "
                f"minimum={active_policy.minimum_prefetch_wall_rtf:.6f}x"
            ),
        ),
        _gate(
            "PREFETCH_GENERATION_GAP_P95",
            (
                prefetch_summary.generation_gap.p95_seconds
                <= active_policy.maximum_prefetch_generation_gap_p95_seconds
            ),
            (
                f"actual={prefetch_summary.generation_gap.p95_seconds:.6f}s; "
                "maximum="
                f"{active_policy.maximum_prefetch_generation_gap_p95_seconds:.6f}s"
            ),
        ),
    )
    status = (
        MageNativeQualificationStatus.PASSED
        if all(gate.passed for gate in gates)
        else MageNativeQualificationStatus.FAILED
    )
    return MageNativeSustainedComparisonReport(
        serial_summary=serial_summary,
        prefetch_summary=prefetch_summary,
        compatibility=compatibility,
        serial_freshness=serial_freshness,
        prefetch_freshness=prefetch_freshness,
        cross_run_isolation=isolation,
        prefetch_speedup=speedup,
        policy=active_policy,
        gates=gates,
        qualification_status=status,
    )


__all__ = [
    "MAGE_NATIVE_SUSTAINED_REPORT_VERSION",
    "MageNativeCrossRunIsolationAssessment",
    "MageNativeFreshnessAssessment",
    "MageNativeGenerationGapSummary",
    "MageNativeGenerationTelemetrySample",
    "MageNativeLatencySummary",
    "MageNativeQualificationGate",
    "MageNativeQualificationStatus",
    "MageNativeRunCompatibility",
    "MageNativeRunIdentity",
    "MageNativeRunMeasurement",
    "MageNativeRunSummary",
    "MageNativeSustainedComparisonReport",
    "MageNativeSustainedQualificationPolicy",
    "MageNativeTelemetryDisposition",
    "MageNativeTimeInterval",
    "assess_cross_run_isolation",
    "assess_run_compatibility",
    "assess_run_freshness",
    "build_mage_native_sustained_comparison_report",
    "generation_gap_summary",
    "interval_intersection_seconds",
    "interval_union_seconds",
    "latency_summary",
    "merge_intervals",
    "serial_vs_prefetch_speedup",
    "summarize_run",
]
