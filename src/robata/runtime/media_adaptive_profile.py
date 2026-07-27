"""Local media/adaptive policy measurements for the P2 qualification boundary.

The canonical source and sampling paths own the actual work.  This module only
assembles bounded observations from one workload into a comparison artifact.  It
keeps recording-hours and camera-hours distinct, records selected/provider work
and process resources together, and makes the local evidence boundary explicit.
No field in this module is a released media or QA wire contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import suppress
from enum import StrEnum
from math import isfinite
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Nanoseconds, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.runtime.capacity import CapacityEvidenceClass, ProviderMode

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

MEDIA_ADAPTIVE_PROFILE_VERSION: Final[Literal["media-adaptive-profile-v1"]] = (
    "media-adaptive-profile-v1"
)
MEDIA_ADAPTIVE_PROFILE_PROJECTION_VERSION: Final[str] = "media-adaptive-profile-v1"
_HOUR_NS: Final[int] = 3_600_000_000_000


class MediaAdaptivePolicy(StrEnum):
    """The three P2 paths compared by a local profile."""

    BASELINE = "BASELINE"
    SENTINEL_ONLY = "SENTINEL_ONLY"
    SELECTIVE_GEOMETRY = "SELECTIVE_GEOMETRY"


# Kept as a separate alias for callers that use "mode" terminology.
MediaAdaptiveMode = MediaAdaptivePolicy


class MediaAdaptiveMeasurement(StrictModel):
    """One bounded observation for one media/adaptive policy.

    Counts are observations rather than inferred zeros.  The three mandatory
    workload counters make all rate denominators explicit; process I/O/RSS and
    optional quality scores remain nullable when the host did not instrument them.
    """

    policy: MediaAdaptivePolicy
    workload_fingerprint: Sha256Digest
    evidence_class: CapacityEvidenceClass = CapacityEvidenceClass.LOCAL_CONFORMANCE
    provider_mode: ProviderMode = ProviderMode.LOCAL_OFFLINE_FIXTURE
    recording_count: PositiveInt
    camera_count: PositiveInt
    recording_duration_ns: Nanoseconds
    wall_time_ns: Nanoseconds
    decoded_frames: NonNegativeInt
    selected_images: NonNegativeInt
    provider_images: NonNegativeInt
    provider_calls: NonNegativeInt
    geometry_images: NonNegativeInt = 0
    geometry_calls: NonNegativeInt = 0
    process_read_bytes: NonNegativeInt | None = None
    process_write_bytes: NonNegativeInt | None = None
    peak_rss_bytes: NonNegativeInt | None = None
    process_cpu_ns: NonNegativeInt | None = None
    quality_score: UnitInterval | None = None
    quality_class: NonEmptyString = "TRADITIONAL_CV"
    quality_measurement_status: Literal["NOT_MEASURED", "LOCAL_PROXY"] = "NOT_MEASURED"

    @model_validator(mode="before")
    @classmethod
    def normalize_measurement_aliases(cls, value: Any) -> Any:
        """Accept established capacity names while retaining one canonical output shape."""

        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        aliases = {
            "policy_id": "policy",
            "mode": "policy",
            "decoded_frame_count": "decoded_frames",
            "selected_image_count": "selected_images",
            "provider_image_count": "provider_images",
            "provider_call_count": "provider_calls",
            "logical_calls": "provider_calls",
            "logical_call_count": "provider_calls",
            "geometry_image_count": "geometry_images",
            "geometry_view_count": "geometry_images",
            "geometry_call_count": "geometry_calls",
            "read_bytes": "process_read_bytes",
            "write_bytes": "process_write_bytes",
            "rss_bytes": "peak_rss_bytes",
            "peak_rss": "peak_rss_bytes",
            "wall_duration_ns": "wall_time_ns",
            "quality_status": "quality_measurement_status",
        }
        for alias, canonical in aliases.items():
            if alias not in normalized:
                continue
            if canonical in normalized and normalized[canonical] != normalized[alias]:
                raise ValueError(f"{alias} and {canonical} cannot disagree")
            normalized[canonical] = normalized.pop(alias)
        for field_name, enum_type in (
            ("policy", MediaAdaptivePolicy),
            ("provider_mode", ProviderMode),
            ("evidence_class", CapacityEvidenceClass),
        ):
            candidate = normalized.get(field_name)
            if isinstance(candidate, str):
                normalized[field_name] = enum_type(candidate)
            else:
                candidate_value = getattr(candidate, "value", None)
                if candidate_value is not None:
                    with suppress(ValueError):
                        normalized[field_name] = enum_type(candidate_value)
        return normalized

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        if self.recording_duration_ns <= 0:
            raise ValueError("recording_duration_ns must be positive")
        if self.wall_time_ns <= 0:
            raise ValueError("wall_time_ns must be positive")
        if self.selected_images > self.decoded_frames:
            raise ValueError("selected_images cannot exceed decoded_frames")
        if self.quality_measurement_status == "NOT_MEASURED" and self.quality_score is not None:
            raise ValueError("NOT_MEASURED quality cannot carry a quality_score")
        if self.quality_measurement_status == "LOCAL_PROXY" and self.quality_score is None:
            raise ValueError("LOCAL_PROXY quality requires a quality_score")
        if self.provider_mode is ProviderMode.NO_PROVIDER_CALLS and (
            self.provider_images != 0 or self.provider_calls != 0
        ):
            raise ValueError("NO_PROVIDER_CALLS measurements must report zero provider work")
        return self

    @property
    def recording_hours(self) -> float:
        return self.recording_count * self.recording_duration_ns / _HOUR_NS

    @property
    def camera_hours(self) -> float:
        return self.recording_hours * self.camera_count

    @property
    def wall_hours(self) -> float:
        return self.wall_time_ns / _HOUR_NS

    @property
    def recording_hours_per_wall_hour(self) -> float:
        return self.recording_hours / self.wall_hours

    @property
    def camera_hours_per_wall_hour(self) -> float:
        return self.camera_hours / self.wall_hours

    @property
    def selected_image_fraction(self) -> float | None:
        return _ratio(self.selected_images, self.decoded_frames)

    @property
    def provider_image_amplification(self) -> float | None:
        """Provider payload images per globally selected source image."""

        return _ratio(self.provider_images, self.selected_images)

    @property
    def provider_call_amplification(self) -> float | None:
        """Provider logical calls per globally selected source image."""

        return _ratio(self.provider_calls, self.selected_images)

    @property
    def provider_amplification(self) -> float | None:
        """Compatibility alias for provider images per selected source image."""

        return self.provider_image_amplification

    @property
    def provider_images_per_selected_image(self) -> float | None:
        return self.provider_image_amplification

    @property
    def provider_calls_per_selected_image(self) -> float | None:
        return self.provider_call_amplification

    @property
    def geometry_selection_fraction(self) -> float | None:
        return _ratio(self.geometry_images, self.selected_images)

    # Compatibility names mirror MeasuredCapacityInput without duplicating fields.
    @property
    def camera_video_hours(self) -> float:
        return self.camera_hours

    @property
    def recording_rtf(self) -> float:
        return self.recording_hours_per_wall_hour

    @property
    def camera_rtf(self) -> float:
        return self.camera_hours_per_wall_hour

    @property
    def selected_image_count(self) -> int:
        return self.selected_images

    @property
    def provider_image_count(self) -> int:
        return self.provider_images

    @property
    def provider_call_count(self) -> int:
        return self.provider_calls

    @property
    def rss_bytes(self) -> int | None:
        return self.peak_rss_bytes

    @property
    def read_bytes(self) -> int | None:
        return self.process_read_bytes

    @property
    def write_bytes(self) -> int | None:
        return self.process_write_bytes

    @property
    def unique_images(self) -> int:
        return self.selected_images

    @property
    def logical_calls(self) -> int:
        return self.provider_calls


class MediaAdaptivePolicyComparison(StrictModel):
    """Before/after ratios for one candidate against the baseline policy."""

    baseline_policy: Literal["BASELINE"] = "BASELINE"
    candidate_policy: MediaAdaptivePolicy
    recording_hours_per_wall_hour_ratio: NonNegativeFloat | None
    camera_hours_per_wall_hour_ratio: NonNegativeFloat | None
    wall_time_ratio: NonNegativeFloat | None
    decoded_frames_ratio: NonNegativeFloat | None
    selected_images_ratio: NonNegativeFloat | None
    provider_images_ratio: NonNegativeFloat | None
    provider_calls_ratio: NonNegativeFloat | None
    provider_image_amplification_ratio: NonNegativeFloat | None
    provider_call_amplification_ratio: NonNegativeFloat | None
    geometry_images_ratio: NonNegativeFloat | None
    geometry_calls_ratio: NonNegativeFloat | None
    process_read_bytes_ratio: NonNegativeFloat | None
    process_write_bytes_ratio: NonNegativeFloat | None
    peak_rss_bytes_ratio: NonNegativeFloat | None
    process_cpu_ns_ratio: NonNegativeFloat | None
    quality_score_delta: float | None = None

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.candidate_policy is MediaAdaptivePolicy.BASELINE:
            raise ValueError("comparison candidate must not be BASELINE")
        if self.quality_score_delta is not None and not isfinite(self.quality_score_delta):
            raise ValueError("quality_score_delta must be finite")
        return self


class MediaAdaptiveProfileReport(StrictModel):
    """Deterministic local Pareto report for the P2 media/adaptive policies."""

    profile_version: Literal["media-adaptive-profile-v1"] = MEDIA_ADAPTIVE_PROFILE_VERSION
    profile_sha256: Sha256Digest
    workload_fingerprint: Sha256Digest
    evidence_class: CapacityEvidenceClass
    provider_mode: ProviderMode
    recording_count: PositiveInt
    camera_count: PositiveInt
    recording_duration_ns: Nanoseconds
    policies: tuple[MediaAdaptiveMeasurement, ...]
    comparisons: tuple[MediaAdaptivePolicyComparison, ...]
    pareto_policy_ids: tuple[MediaAdaptivePolicy, ...]
    measurement_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    quality_status: Literal["NOT_MEASURED", "LOCAL_PROXY"] = "NOT_MEASURED"
    evidence_note: Literal["LOCAL_ONLY_NOT_PRODUCTION_QUALIFIED"] = (
        "LOCAL_ONLY_NOT_PRODUCTION_QUALIFIED"
    )
    production_eligible: Literal[False] = False
    qualification_status: Literal["NOT_PRODUCTION_QUALIFIED"] = "NOT_PRODUCTION_QUALIFIED"

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.evidence_class is not CapacityEvidenceClass.LOCAL_CONFORMANCE:
            raise ValueError("media adaptive P2 profiles must use LOCAL_CONFORMANCE evidence")
        expected_policies = (
            MediaAdaptivePolicy.BASELINE,
            MediaAdaptivePolicy.SENTINEL_ONLY,
            MediaAdaptivePolicy.SELECTIVE_GEOMETRY,
        )
        policy_ids = tuple(item.policy for item in self.policies)
        if policy_ids != expected_policies:
            raise ValueError(
                "policies must contain BASELINE, SENTINEL_ONLY, SELECTIVE_GEOMETRY in order"
            )
        if len({item.policy for item in self.policies}) != len(self.policies):
            raise ValueError("policy values must be unique")
        if any(
            (
                item.workload_fingerprint != self.workload_fingerprint
                or item.evidence_class is not self.evidence_class
                or item.provider_mode is not self.provider_mode
                or item.recording_count != self.recording_count
                or item.camera_count != self.camera_count
                or item.recording_duration_ns != self.recording_duration_ns
            )
            for item in self.policies
        ):
            raise ValueError("all policy rows must share workload, units, and evidence scope")
        expected_candidates = (
            MediaAdaptivePolicy.SENTINEL_ONLY,
            MediaAdaptivePolicy.SELECTIVE_GEOMETRY,
        )
        if tuple(item.candidate_policy for item in self.comparisons) != expected_candidates:
            raise ValueError("comparisons must contain the two non-baseline policies in order")
        quality_modes = {item.quality_measurement_status for item in self.policies}
        if len(quality_modes) != 1:
            raise ValueError("all policy rows must use the same quality measurement status")
        if self.quality_status == "LOCAL_PROXY" and quality_modes != {"LOCAL_PROXY"}:
            raise ValueError("LOCAL_PROXY report status requires LOCAL_PROXY policy rows")
        if self.quality_status == "NOT_MEASURED" and quality_modes != {"NOT_MEASURED"}:
            raise ValueError("NOT_MEASURED report status requires NOT_MEASURED policy rows")
        if any(item not in expected_policies for item in self.pareto_policy_ids):
            raise ValueError("pareto_policy_ids contains an unknown policy")
        expected_frontier = _pareto_policy_ids(self.policies)
        if self.pareto_policy_ids != expected_frontier:
            raise ValueError("pareto_policy_ids do not match policy observations")
        expected_digest = semantic_sha256(media_adaptive_profile_projection(self))
        if self.profile_sha256 != expected_digest:
            raise ValueError("profile_sha256 does not match the report projection")
        return self

    @property
    def baseline(self) -> MediaAdaptiveMeasurement:
        return self.policies[0]

    @property
    def recording_hours(self) -> float:
        return self.baseline.recording_hours

    @property
    def camera_hours(self) -> float:
        return self.baseline.camera_hours

    @property
    def wall_hours(self) -> float:
        return self.baseline.wall_hours

    @property
    def recording_hours_per_wall_hour(self) -> float:
        return self.baseline.recording_hours_per_wall_hour

    @property
    def camera_hours_per_wall_hour(self) -> float:
        return self.baseline.camera_hours_per_wall_hour

    @property
    def selected_images(self) -> int:
        return self.baseline.selected_images

    @property
    def provider_images(self) -> int:
        return self.baseline.provider_images

    @property
    def provider_calls(self) -> int:
        return self.baseline.provider_calls

    @property
    def provider_amplification(self) -> float | None:
        return self.baseline.provider_amplification

    @property
    def process_read_bytes(self) -> int | None:
        return self.baseline.process_read_bytes

    @property
    def process_write_bytes(self) -> int | None:
        return self.baseline.process_write_bytes

    @property
    def peak_rss_bytes(self) -> int | None:
        return self.baseline.peak_rss_bytes

    @property
    def sentinel_only(self) -> MediaAdaptiveMeasurement:
        return self.policies[1]

    @property
    def selective_geometry(self) -> MediaAdaptiveMeasurement:
        return self.policies[2]

    def policy(self, policy: MediaAdaptivePolicy | str) -> MediaAdaptiveMeasurement:
        policy = MediaAdaptivePolicy(policy)
        for row in self.policies:
            if row.policy is policy:
                return row
        raise ValueError(f"unknown media policy: {policy}")

    def is_pareto_optimal(self, policy: MediaAdaptivePolicy | str) -> bool:
        return MediaAdaptivePolicy(policy) in self.pareto_policy_ids

    @property
    def frontier_policy_ids(self) -> tuple[MediaAdaptivePolicy, ...]:
        return self.pareto_policy_ids

    @property
    def policy_comparisons(self) -> tuple[MediaAdaptivePolicyComparison, ...]:
        return self.comparisons

    @property
    def quality_classes(self) -> tuple[str, ...]:
        return tuple(row.quality_class for row in self.policies)

    def as_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def render_markdown(self) -> str:
        """Render a compact local-only report for qualification notes."""

        frontier = set(self.pareto_policy_ids)
        lines = [
            "# Local media/adaptive profile",
            "",
            f"- Workload: {self.workload_fingerprint}",
            f"- Evidence class: {self.evidence_class.value}",
            f"- Provider mode: {self.provider_mode.value}",
            f"- Recording hours: {self.baseline.recording_hours:.6f}",
            f"- Camera hours: {self.baseline.camera_hours:.6f}",
            "- Measurement status: NOT_MEASURED",
            f"- Quality classes: {', '.join(self.quality_classes)}",
            "- Production eligible: NO",
            "",
            "| Policy | Selected images | Provider images | Provider calls | Image amp. | "
            + "Call amp. | Read bytes | Write bytes | Peak RSS | CPU ns | Pareto |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in self.policies:
            lines.append(
                "| "
                f"{row.policy.value} | {row.selected_images} | {row.provider_images} | "
                f"{row.provider_calls} | {_format_ratio(row.provider_image_amplification)} | "
                f"{_format_ratio(row.provider_call_amplification)} | "
                f"{_format_optional_int(row.process_read_bytes)} | "
                f"{_format_optional_int(row.process_write_bytes)} | "
                f"{_format_optional_int(row.peak_rss_bytes)} | "
                f"{_format_optional_int(row.process_cpu_ns)} | "
                f"{'YES' if row.policy in frontier else 'NO'} |"
            )
        return "\n".join(lines) + "\n"


# Descriptive aliases for call sites that call rows "observations" or the report a profile.
MediaAdaptivePolicyObservation = MediaAdaptiveMeasurement
MediaAdaptiveProfile = MediaAdaptiveProfileReport
MediaAdaptiveComparison = MediaAdaptivePolicyComparison


def media_adaptive_profile_projection(
    report: MediaAdaptiveProfileReport,
) -> dict[str, object]:
    """Return the complete digest preimage for one report."""

    if not isinstance(report, MediaAdaptiveProfileReport):
        raise TypeError("report must be a MediaAdaptiveProfileReport")
    return {
        "projection_version": MEDIA_ADAPTIVE_PROFILE_PROJECTION_VERSION,
        **report.model_dump(mode="json", exclude={"profile_sha256"}),
    }


def build_media_adaptive_profile_report(
    measurements: Iterable[MediaAdaptiveMeasurement | Mapping[str, object]] | None = None,
    *,
    observations: Iterable[MediaAdaptiveMeasurement | Mapping[str, object]] | None = None,
    policies: Iterable[MediaAdaptiveMeasurement | Mapping[str, object]] | None = None,
) -> MediaAdaptiveProfileReport:
    """Build a content-addressed local comparison over the three P2 policies."""

    supplied = tuple(
        candidate for candidate in (measurements, observations, policies) if candidate is not None
    )
    if len(supplied) != 1:
        raise TypeError("supply exactly one of measurements, observations, or policies")
    checked = tuple(
        item
        if isinstance(item, MediaAdaptiveMeasurement)
        else MediaAdaptiveMeasurement.model_validate(item, strict=True)
        for item in supplied[0]
    )
    if not checked:
        raise TypeError("measurements must contain MediaAdaptiveMeasurement values")
    by_policy = {item.policy: item for item in checked}
    expected = (
        MediaAdaptivePolicy.BASELINE,
        MediaAdaptivePolicy.SENTINEL_ONLY,
        MediaAdaptivePolicy.SELECTIVE_GEOMETRY,
    )
    if set(by_policy) != set(expected) or len(by_policy) != len(checked):
        raise ValueError("measurements must contain each P2 policy exactly once")
    ordered = tuple(by_policy[policy] for policy in expected)
    baseline = ordered[0]
    comparisons = tuple(_compare_media_policy(baseline, candidate) for candidate in ordered[1:])
    quality_statuses = {row.quality_measurement_status for row in ordered}
    quality_status: Literal["NOT_MEASURED", "LOCAL_PROXY"] = (
        "LOCAL_PROXY" if quality_statuses == {"LOCAL_PROXY"} else "NOT_MEASURED"
    )
    draft = MediaAdaptiveProfileReport.model_construct(
        profile_version=MEDIA_ADAPTIVE_PROFILE_VERSION,
        profile_sha256="0" * 64,
        workload_fingerprint=baseline.workload_fingerprint,
        evidence_class=baseline.evidence_class,
        provider_mode=baseline.provider_mode,
        recording_count=baseline.recording_count,
        camera_count=baseline.camera_count,
        recording_duration_ns=baseline.recording_duration_ns,
        policies=ordered,
        comparisons=comparisons,
        pareto_policy_ids=_pareto_policy_ids(ordered),
        measurement_status="NOT_MEASURED",
        quality_status=quality_status,
        evidence_note="LOCAL_ONLY_NOT_PRODUCTION_QUALIFIED",
        production_eligible=False,
        qualification_status="NOT_PRODUCTION_QUALIFIED",
    )
    digest = semantic_sha256(media_adaptive_profile_projection(draft))
    return MediaAdaptiveProfileReport.model_validate(
        {**draft.model_dump(mode="python"), "profile_sha256": digest},
        strict=True,
    )


build_media_adaptive_profile = build_media_adaptive_profile_report


def _compare_media_policy(
    baseline: MediaAdaptiveMeasurement,
    candidate: MediaAdaptiveMeasurement,
) -> MediaAdaptivePolicyComparison:
    return MediaAdaptivePolicyComparison(
        candidate_policy=candidate.policy,
        recording_hours_per_wall_hour_ratio=_ratio(
            candidate.recording_hours_per_wall_hour,
            baseline.recording_hours_per_wall_hour,
        ),
        camera_hours_per_wall_hour_ratio=_ratio(
            candidate.camera_hours_per_wall_hour,
            baseline.camera_hours_per_wall_hour,
        ),
        wall_time_ratio=_ratio(candidate.wall_time_ns, baseline.wall_time_ns),
        decoded_frames_ratio=_ratio(candidate.decoded_frames, baseline.decoded_frames),
        selected_images_ratio=_ratio(candidate.selected_images, baseline.selected_images),
        provider_images_ratio=_ratio(candidate.provider_images, baseline.provider_images),
        provider_calls_ratio=_ratio(candidate.provider_calls, baseline.provider_calls),
        provider_image_amplification_ratio=_ratio(
            candidate.provider_image_amplification,
            baseline.provider_image_amplification,
        ),
        provider_call_amplification_ratio=_ratio(
            candidate.provider_call_amplification,
            baseline.provider_call_amplification,
        ),
        geometry_images_ratio=_ratio(candidate.geometry_images, baseline.geometry_images),
        geometry_calls_ratio=_ratio(candidate.geometry_calls, baseline.geometry_calls),
        process_read_bytes_ratio=_ratio(candidate.process_read_bytes, baseline.process_read_bytes),
        process_write_bytes_ratio=_ratio(
            candidate.process_write_bytes,
            baseline.process_write_bytes,
        ),
        peak_rss_bytes_ratio=_ratio(candidate.peak_rss_bytes, baseline.peak_rss_bytes),
        process_cpu_ns_ratio=_ratio(candidate.process_cpu_ns, baseline.process_cpu_ns),
        quality_score_delta=(
            None
            if candidate.quality_score is None or baseline.quality_score is None
            else candidate.quality_score - baseline.quality_score
        ),
    )


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = float(numerator) / float(denominator)
    return value if isfinite(value) else None


def _pareto_policy_ids(
    policies: tuple[MediaAdaptiveMeasurement, ...],
) -> tuple[MediaAdaptivePolicy, ...]:
    return tuple(
        row.policy
        for row in policies
        if not any(_dominates(other, row) for other in policies if other.policy is not row.policy)
    )


def _dominates(left: MediaAdaptiveMeasurement, right: MediaAdaptiveMeasurement) -> bool:
    """Compare always-present local axes; quality is used only when both are proxy scores."""

    left_cost = (
        left.wall_time_ns,
        left.selected_images,
        left.provider_images,
        left.provider_calls,
    )
    right_cost = (
        right.wall_time_ns,
        right.selected_images,
        right.provider_images,
        right.provider_calls,
    )
    quality_not_worse = (
        left.quality_score is None
        or right.quality_score is None
        or left.quality_score >= right.quality_score
    )
    cost_not_worse = all(a <= b for a, b in zip(left_cost, right_cost, strict=True))
    strictly_better = any(a < b for a, b in zip(left_cost, right_cost, strict=True))
    if left.quality_score is not None and right.quality_score is not None:
        strictly_better = strictly_better or left.quality_score > right.quality_score
    return quality_not_worse and cost_not_worse and strictly_better


def _format_ratio(value: float | None) -> str:
    return "NOT_AVAILABLE" if value is None else f"{value:.6f}"


def _format_optional_int(value: int | None) -> str:
    return "NOT_AVAILABLE" if value is None else str(value)


__all__ = [
    "MEDIA_ADAPTIVE_PROFILE_PROJECTION_VERSION",
    "MEDIA_ADAPTIVE_PROFILE_VERSION",
    "MediaAdaptiveComparison",
    "MediaAdaptiveMeasurement",
    "MediaAdaptiveMode",
    "MediaAdaptivePolicy",
    "MediaAdaptivePolicyComparison",
    "MediaAdaptivePolicyObservation",
    "MediaAdaptiveProfile",
    "MediaAdaptiveProfileReport",
    "build_media_adaptive_profile",
    "build_media_adaptive_profile_report",
    "media_adaptive_profile_projection",
]
