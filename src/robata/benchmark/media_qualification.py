"""Target-media qualification artifacts for the P8 boundary.

The media adapters intentionally remain provider ports: a target deployment may
inject an NVDEC/DeepStream implementation, while the local checkout normally has
only the PyAV reference path.  This module records the facts needed to compare the
two paths without making an acceleration claim on behalf of an injected backend.

The report is an internal qualification artifact.  It is content addressed, keeps
the source/media contract visible, and can only describe a safe envelope; it can
never grant production eligibility.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Nanoseconds, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.runtime.capacity import CapacityEvidenceClass

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]

MEDIA_QUALIFICATION_VERSION: Final[Literal["target-media-qualification-v1"]] = (
    "target-media-qualification-v1"
)
MEDIA_QUALIFICATION_PROJECTION_VERSION: Final[str] = MEDIA_QUALIFICATION_VERSION
DEFAULT_AVERAGE_CAMERA_SECONDS_PER_SECOND: Final[float] = 125.0
DEFAULT_MARGIN_CAMERA_SECONDS_PER_SECOND: Final[float] = 150.0
_NANOSECONDS_PER_SECOND = 1_000_000_000


class MediaBackend(StrEnum):
    """Decode/materialization path measured by one observation."""

    CPU = "CPU"
    NVDEC = "NVDEC"


class MediaExecutionMode(StrEnum):
    """Whether the observation traversed source bytes or replayed a fixture."""

    FRESH = "FRESH"
    REPLAY = "REPLAY"


class MediaParityStatus(StrEnum):
    """Parity state for the timestamp and artifact contracts."""

    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NOT_MEASURED = "NOT_MEASURED"


class MediaSourceProfile(StrictModel):
    """Frozen codec/shape/transport facts for one media matrix row."""

    codec: NonEmptyString
    width: PositiveInt
    height: PositiveInt
    fps_num: PositiveInt
    fps_den: PositiveInt = 1
    gop_frames: PositiveInt
    transfer_path: NonEmptyString
    source_manifest_digest: Sha256Digest | None = None
    profile_digest: Sha256Digest

    @classmethod
    def create(cls, **values: object) -> Self:
        """Build a profile and derive its digest from the exact source facts."""

        if "profile_digest" in values:
            raise ValueError("profile_digest is derived")
        draft = cls.model_construct(**{**values, "profile_digest": "0" * 64})
        digest = semantic_sha256(
            media_source_profile_projection(draft, include_digest=False)
        )
        return cls.model_validate(
            {**draft.model_dump(mode="python"), "profile_digest": digest},
            strict=True,
        )

    @model_validator(mode="before")
    @classmethod
    def normalize_profile_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        aliases = {
            "resolution_width": "width",
            "resolution_height": "height",
            "fps_numerator": "fps_num",
            "fps_denominator": "fps_den",
            "gop": "gop_frames",
            "transfer": "transfer_path",
        }
        for alias, canonical in aliases.items():
            if alias not in normalized:
                continue
            if canonical in normalized and normalized[canonical] != normalized[alias]:
                raise ValueError(f"{alias} and {canonical} cannot disagree")
            normalized[canonical] = normalized.pop(alias)
        return normalized

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        codec = self.codec.strip().lower()
        if not codec:
            raise ValueError("codec must be nonempty")
        if codec != self.codec:
            # Frozen models cannot be mutated after validation; reject rather than
            # silently changing the profile digest supplied by a caller.
            raise ValueError("codec must be lowercase and trimmed")
        if self.fps_num <= 0 or self.fps_den <= 0:
            raise ValueError("FPS numerator and denominator must be positive")
        expected = semantic_sha256(media_source_profile_projection(self, include_digest=False))
        if self.profile_digest != expected:
            raise ValueError("profile_digest does not match source media facts")
        return self

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


class MediaQualificationMeasurement(StrictModel):
    """One bounded CPU/NVDEC decode, resize, and materialization observation."""

    workload_manifest_digest: Sha256Digest
    run_namespace: NonEmptyString
    backend: MediaBackend
    execution_mode: MediaExecutionMode
    source_profile: MediaSourceProfile
    camera_seconds: PositiveFloat
    wall_time_ns: Nanoseconds
    decoded_frames: NonNegativeInt
    resized_frames: NonNegativeInt
    materialized_frames: NonNegativeInt
    read_bytes: NonNegativeInt | None = None
    write_bytes: NonNegativeInt | None = None
    peak_rss_bytes: NonNegativeInt | None = None
    process_cpu_ns: NonNegativeInt | None = None
    timestamp_parity: MediaParityStatus = MediaParityStatus.NOT_MEASURED
    artifact_parity: MediaParityStatus = MediaParityStatus.NOT_MEASURED
    timestamp_contract_digest: Sha256Digest | None = None
    artifact_contract_digest: Sha256Digest | None = None
    nvdec_fallback_count: NonNegativeInt = 0
    raw_rgb_persisted: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def normalize_measurement_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        aliases = {
            "camera_duration_seconds": "camera_seconds",
            "decoded_frame_count": "decoded_frames",
            "resized_frame_count": "resized_frames",
            "materialized_frame_count": "materialized_frames",
            "process_read_bytes": "read_bytes",
            "process_write_bytes": "write_bytes",
            "peak_rss": "peak_rss_bytes",
            "fallback_count": "nvdec_fallback_count",
            "timestamp_status": "timestamp_parity",
            "artifact_status": "artifact_parity",
        }
        for alias, canonical in aliases.items():
            if alias not in normalized:
                continue
            if canonical in normalized and normalized[canonical] != normalized[alias]:
                raise ValueError(f"{alias} and {canonical} cannot disagree")
            normalized[canonical] = normalized.pop(alias)
        return normalized

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        if self.wall_time_ns <= 0:
            raise ValueError("wall_time_ns must be positive")
        if self.resized_frames > self.decoded_frames:
            raise ValueError("resized_frames cannot exceed decoded_frames")
        if self.materialized_frames > self.resized_frames:
            raise ValueError("materialized_frames cannot exceed resized_frames")
        if self.backend is MediaBackend.CPU and self.nvdec_fallback_count:
            raise ValueError("CPU measurements cannot report NVDEC fallbacks")
        if self.nvdec_fallback_count and self.backend is MediaBackend.NVDEC:
            # A fallback means the target path did not process every frame.  Keep the
            # row usable for diagnosis but make it ineligible for the safe envelope.
            if self.timestamp_parity is MediaParityStatus.MATCH and self.artifact_parity is MediaParityStatus.MATCH:
                raise ValueError(
                    "NVDEC fallback observations cannot claim complete media parity"
                )
        return self

    @property
    def wall_seconds(self) -> float:
        return self.wall_time_ns / _NANOSECONDS_PER_SECOND

    @property
    def camera_seconds_per_second(self) -> float:
        return self.camera_seconds / self.wall_seconds

    @property
    def camera_seconds_per_wall_hour(self) -> float:
        return self.camera_seconds_per_second * 3_600.0

    @property
    def throughput_camera_seconds_per_second(self) -> float:
        return self.camera_seconds_per_second

    @property
    def parity_matches(self) -> bool:
        return (
            self.timestamp_parity is MediaParityStatus.MATCH
            and self.artifact_parity is MediaParityStatus.MATCH
            and self.nvdec_fallback_count == 0
        )


class MediaQualificationEnvelope(StrictModel):
    """Measured rates and explicit reasons for a target-media safe envelope."""

    average_camera_seconds_per_second: NonNegativeFloat
    minimum_camera_seconds_per_second: NonNegativeFloat
    required_average_camera_seconds_per_second: PositiveFloat = (
        DEFAULT_AVERAGE_CAMERA_SECONDS_PER_SECOND
    )
    required_margin_camera_seconds_per_second: PositiveFloat = (
        DEFAULT_MARGIN_CAMERA_SECONDS_PER_SECOND
    )
    parity_complete: bool
    required_backends_present: bool
    safe_envelope: bool
    unmet_requirements: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if (
            self.required_margin_camera_seconds_per_second
            < self.required_average_camera_seconds_per_second
        ):
            raise ValueError("margin throughput must be at least the average target")
        expected_safe = (
            self.average_camera_seconds_per_second
            >= self.required_average_camera_seconds_per_second
            and self.minimum_camera_seconds_per_second
            >= self.required_margin_camera_seconds_per_second
            and self.parity_complete
            and self.required_backends_present
        )
        if self.safe_envelope != expected_safe:
            raise ValueError("safe_envelope does not match measured rates and parity")
        if self.safe_envelope and self.unmet_requirements:
            raise ValueError("safe envelope cannot carry unmet requirements")
        if not self.safe_envelope and not self.unmet_requirements:
            raise ValueError("an unsafe envelope must explain its unmet requirements")
        if tuple(sorted(set(self.unmet_requirements))) != self.unmet_requirements:
            raise ValueError("unmet_requirements must be unique and sorted")
        return self


class MediaQualificationReport(StrictModel):
    """Content-addressed target-media qualification matrix."""

    report_version: Literal["target-media-qualification-v1"] = MEDIA_QUALIFICATION_VERSION
    report_sha256: Sha256Digest
    workload_manifest_digest: Sha256Digest
    measurements: tuple[MediaQualificationMeasurement, ...] = Field(min_length=1)
    required_backends: tuple[MediaBackend, ...] = (MediaBackend.CPU, MediaBackend.NVDEC)
    envelope: MediaQualificationEnvelope
    evidence_class: CapacityEvidenceClass = CapacityEvidenceClass.LOCAL_CONFORMANCE
    measurement_status: Literal["MEASURED", "NOT_MEASURED"] = "MEASURED"
    target_hardware_status: Literal["NOT_MEASURED", "MEASURED"] = "NOT_MEASURED"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if not self.required_backends:
            raise ValueError("required_backends must not be empty")
        if tuple(sorted(set(self.required_backends), key=lambda value: value.value)) != tuple(
            sorted(self.required_backends, key=lambda value: value.value)
        ):
            raise ValueError("required_backends must be unique and ordered")
        if any(item.workload_manifest_digest != self.workload_manifest_digest for item in self.measurements):
            raise ValueError("all media measurements must use the report workload manifest")
        keys = tuple((item.backend, item.source_profile.profile_digest, item.execution_mode) for item in self.measurements)
        if len(keys) != len(set(keys)):
            raise ValueError("media measurements must not duplicate backend/profile/execution")
        observed_backends = {item.backend for item in self.measurements}
        if self.envelope.required_backends_present != set(self.required_backends).issubset(observed_backends):
            raise ValueError("required_backends_present does not match media observations")
        expected_parity = all(item.parity_matches for item in self.measurements)
        if self.envelope.parity_complete != expected_parity:
            raise ValueError("parity_complete does not match media observations")
        expected_digest = semantic_sha256(media_qualification_projection(self, include_digest=False))
        if self.report_sha256 != expected_digest:
            raise ValueError("report_sha256 does not match media qualification projection")
        return self

    @property
    def safe_envelope(self) -> bool:
        return self.envelope.safe_envelope

    @property
    def average_camera_seconds_per_second(self) -> float:
        return self.envelope.average_camera_seconds_per_second

    @property
    def minimum_camera_seconds_per_second(self) -> float:
        return self.envelope.minimum_camera_seconds_per_second

    @property
    def missing_backends(self) -> tuple[MediaBackend, ...]:
        observed = {item.backend for item in self.measurements}
        return tuple(backend for backend in self.required_backends if backend not in observed)

    def render_markdown(self) -> str:
        """Render a compact report that keeps local/target evidence explicit."""

        lines = [
            "# Target media qualification report",
            "",
            f"- Workload manifest: {self.workload_manifest_digest}",
            f"- Evidence class: {self.evidence_class.value}",
            f"- Measurement status: {self.measurement_status}",
            f"- Target hardware: {self.target_hardware_status}",
            f"- Average camera-sec/s: {self.average_camera_seconds_per_second:.3f}",
            f"- Minimum camera-sec/s: {self.minimum_camera_seconds_per_second:.3f}",
            f"- Required average/margin: {self.envelope.required_average_camera_seconds_per_second:.3f} / {self.envelope.required_margin_camera_seconds_per_second:.3f}",
            f"- Safe envelope: {'YES' if self.safe_envelope else 'NO'}",
            "- Production eligible: NO",
            "",
            "| Backend | Codec | Resolution | FPS | GOP | Transfer | Camera-sec/s | Timestamp parity | Artifact parity |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |",
        ]
        for item in self.measurements:
            profile = item.source_profile
            lines.append(
                f"| {item.backend.value} | {profile.codec} | {profile.width}x{profile.height} | "
                f"{profile.fps:g} | {profile.gop_frames} | {profile.transfer_path} | "
                f"{item.camera_seconds_per_second:.3f} | {item.timestamp_parity.value} | "
                f"{item.artifact_parity.value} |"
            )
        if self.envelope.unmet_requirements:
            lines.extend(("", "- Unmet requirements: " + ", ".join(self.envelope.unmet_requirements)))
        return "\n".join(lines) + "\n"


def media_source_profile_projection(
    profile: MediaSourceProfile,
    *,
    include_digest: bool = True,
) -> dict[str, object]:
    """Return the exact digest preimage for a source profile."""

    if not isinstance(profile, MediaSourceProfile):
        raise TypeError("profile must be a MediaSourceProfile")
    projection = {
        "projection_version": MEDIA_QUALIFICATION_PROJECTION_VERSION,
        **profile.model_dump(mode="json"),
    }
    if not include_digest:
        projection.pop("profile_digest", None)
    return projection


def media_qualification_projection(
    report: MediaQualificationReport,
    *,
    include_digest: bool = True,
) -> dict[str, object]:
    """Return the exact digest preimage for a qualification report."""

    if not isinstance(report, MediaQualificationReport):
        raise TypeError("report must be a MediaQualificationReport")
    projection = {
        "projection_version": MEDIA_QUALIFICATION_PROJECTION_VERSION,
        **report.model_dump(mode="json"),
    }
    if not include_digest:
        projection.pop("report_sha256", None)
    return projection


def build_media_qualification_report(
    measurements: Iterable[MediaQualificationMeasurement | Mapping[str, object]],
    *,
    workload_manifest_digest: Sha256Digest | None = None,
    required_backends: Iterable[MediaBackend] = (MediaBackend.CPU, MediaBackend.NVDEC),
    required_average_camera_seconds_per_second: float = DEFAULT_AVERAGE_CAMERA_SECONDS_PER_SECOND,
    required_margin_camera_seconds_per_second: float = DEFAULT_MARGIN_CAMERA_SECONDS_PER_SECOND,
    evidence_class: CapacityEvidenceClass = CapacityEvidenceClass.LOCAL_CONFORMANCE,
    measurement_status: Literal["MEASURED", "NOT_MEASURED"] = "MEASURED",
    target_hardware_status: Literal["NOT_MEASURED", "MEASURED"] = "NOT_MEASURED",
) -> MediaQualificationReport:
    """Build a deterministic matrix and derive its weighted throughput envelope."""

    checked = tuple(
        item
        if isinstance(item, MediaQualificationMeasurement)
        else MediaQualificationMeasurement.model_validate(item, strict=True)
        for item in measurements
    )
    if not checked:
        raise ValueError("measurements must contain at least one observation")
    workload = workload_manifest_digest or checked[0].workload_manifest_digest
    if any(item.workload_manifest_digest != workload for item in checked):
        raise ValueError("measurements must use one workload manifest")
    required = tuple(MediaBackend(item) for item in required_backends)
    if not required:
        raise ValueError("required_backends must not be empty")
    total_camera_seconds = sum(item.camera_seconds for item in checked)
    total_wall_seconds = sum(item.wall_seconds for item in checked)
    average_rate = total_camera_seconds / total_wall_seconds
    minimum_rate = min(item.camera_seconds_per_second for item in checked)
    observed_backends = {item.backend for item in checked}
    required_present = set(required).issubset(observed_backends)
    parity_complete = all(item.parity_matches for item in checked)
    unmet: list[str] = []
    if average_rate < required_average_camera_seconds_per_second:
        unmet.append("AVERAGE_THROUGHPUT_BELOW_TARGET")
    if minimum_rate < required_margin_camera_seconds_per_second:
        unmet.append("MARGIN_THROUGHPUT_BELOW_TARGET")
    if not required_present:
        unmet.append("REQUIRED_BACKEND_NOT_MEASURED")
    if not parity_complete:
        unmet.append("MEDIA_CONTRACT_PARITY_NOT_PROVEN")
    unmet_tuple = tuple(sorted(set(unmet)))
    envelope = MediaQualificationEnvelope(
        average_camera_seconds_per_second=average_rate,
        minimum_camera_seconds_per_second=minimum_rate,
        required_average_camera_seconds_per_second=required_average_camera_seconds_per_second,
        required_margin_camera_seconds_per_second=required_margin_camera_seconds_per_second,
        parity_complete=parity_complete,
        required_backends_present=required_present,
        safe_envelope=not unmet_tuple,
        unmet_requirements=unmet_tuple,
    )
    draft = MediaQualificationReport.model_construct(
        report_version=MEDIA_QUALIFICATION_VERSION,
        report_sha256="0" * 64,
        workload_manifest_digest=workload,
        measurements=checked,
        required_backends=required,
        envelope=envelope,
        evidence_class=evidence_class,
        measurement_status=measurement_status,
        target_hardware_status=target_hardware_status,
        production_eligible=False,
    )
    digest = semantic_sha256(media_qualification_projection(draft, include_digest=False))
    return MediaQualificationReport.model_validate(
        {**draft.model_dump(mode="python"), "report_sha256": digest},
        strict=True,
    )


def measure_media_callable(
    workload: Callable[[], object],
    *,
    workload_manifest_digest: Sha256Digest,
    run_namespace: NonEmptyString,
    backend: MediaBackend,
    execution_mode: MediaExecutionMode,
    source_profile: MediaSourceProfile,
    camera_seconds: PositiveFloat,
    stats: Mapping[str, object] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> MediaQualificationMeasurement:
    """Measure one injected media workload without assuming target hardware.

    The callable may return a mapping with counter names, or callers may provide
    ``stats`` explicitly.  Uninstrumented counters remain zero/``None`` rather than
    being invented from elapsed time.
    """

    if not callable(workload):
        raise TypeError("workload must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")
    started = clock()
    result = workload()
    elapsed = clock() - started
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("media qualification clock must advance by a positive finite duration")
    observed = dict(stats or {})
    if isinstance(result, Mapping):
        for key, value in result.items():
            observed.setdefault(str(key), value)

    def counter(name: str, *aliases: str) -> int:
        value = next((observed[key] for key in (name, *aliases) if key in observed), 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    def optional_counter(name: str, *aliases: str) -> int | None:
        value = next((observed[key] for key in (name, *aliases) if key in observed), None)
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer or None")
        return value

    return MediaQualificationMeasurement(
        workload_manifest_digest=workload_manifest_digest,
        run_namespace=run_namespace,
        backend=backend,
        execution_mode=execution_mode,
        source_profile=source_profile,
        camera_seconds=camera_seconds,
        wall_time_ns=max(1, round(elapsed * _NANOSECONDS_PER_SECOND)),
        decoded_frames=counter("decoded_frames", "decoded_frame_count"),
        resized_frames=counter("resized_frames", "resized_frame_count"),
        materialized_frames=counter("materialized_frames", "materialized_frame_count"),
        read_bytes=optional_counter("read_bytes", "process_read_bytes"),
        write_bytes=optional_counter("write_bytes", "process_write_bytes"),
        peak_rss_bytes=optional_counter("peak_rss_bytes", "peak_rss"),
        process_cpu_ns=optional_counter("process_cpu_ns"),
        timestamp_parity=MediaParityStatus(observed.get("timestamp_parity", MediaParityStatus.NOT_MEASURED)),
        artifact_parity=MediaParityStatus(observed.get("artifact_parity", MediaParityStatus.NOT_MEASURED)),
        timestamp_contract_digest=observed.get("timestamp_contract_digest"),
        artifact_contract_digest=observed.get("artifact_contract_digest"),
        nvdec_fallback_count=counter("nvdec_fallback_count", "fallback_count"),
    )


# Descriptive aliases for callers that use "target" or "matrix" terminology.
TargetMediaProfile = MediaSourceProfile
TargetMediaMeasurement = MediaQualificationMeasurement
TargetMediaQualificationReport = MediaQualificationReport
TargetMediaQualificationEnvelope = MediaQualificationEnvelope
build_target_media_qualification_report = build_media_qualification_report


__all__ = [
    "DEFAULT_AVERAGE_CAMERA_SECONDS_PER_SECOND",
    "DEFAULT_MARGIN_CAMERA_SECONDS_PER_SECOND",
    "MEDIA_QUALIFICATION_PROJECTION_VERSION",
    "MEDIA_QUALIFICATION_VERSION",
    "MediaBackend",
    "MediaExecutionMode",
    "MediaParityStatus",
    "MediaQualificationEnvelope",
    "MediaQualificationMeasurement",
    "MediaQualificationReport",
    "MediaSourceProfile",
    "TargetMediaMeasurement",
    "TargetMediaProfile",
    "TargetMediaQualificationEnvelope",
    "TargetMediaQualificationReport",
    "build_media_qualification_report",
    "build_target_media_qualification_report",
    "measure_media_callable",
    "media_qualification_projection",
    "media_source_profile_projection",
]
