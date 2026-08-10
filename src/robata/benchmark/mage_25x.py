"""Evidence-bound capacity calculations for the Mage 25x convergence cycle.

The module consumes the retained Provider V2 qualification report rather than
copying timing values into a benchmark.  It deliberately distinguishes measured
whole-stage walls from a lower-bound calculation over measured worker/model job
sums.  Calculations never confer production eligibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Final, Literal

from robata.contracts.hashing import semantic_sha256

MAGE_25X_CAPACITY_REPORT_VERSION: Final = "mage-25x-capacity-report-v1"
MAGE_PROVIDER_V2_QUALIFICATION_VERSION: Final = "mage-dcvc-provider-v2-local-artifact-ab-report-v1"
MAGE_PROVIDER_V2_BOUNDED_VARIANT_ID: Final = "provider-v2-max-side-448"
DEFAULT_DAILY_CAMERA_HOURS: Final = 500.0
DEFAULT_CAPACITY_HEADROOM: Final = 1.20
DEFAULT_REPOSITORY_DAILY_RECORDING_HOURS: Final = 500.0
DEFAULT_REPOSITORY_CAMERA_COUNT: Final = 6


class MageCapacityEvidenceError(ValueError):
    """Retained Mage capacity evidence is missing, changed, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class MageCapacityStage:
    """One measured service boundary expressed in camera-throughput units."""

    stage_id: str
    wall_seconds: float
    media_seconds: float
    realtime_factor: float
    camera_hours_per_day_per_lane: float

    @classmethod
    def measured(
        cls, *, stage_id: str, wall_seconds: float, media_seconds: float
    ) -> MageCapacityStage:
        wall = _positive_finite(wall_seconds, f"{stage_id}.wall_seconds")
        media = _positive_finite(media_seconds, f"{stage_id}.media_seconds")
        realtime_factor = media / wall
        return cls(
            stage_id=stage_id,
            wall_seconds=wall,
            media_seconds=media,
            realtime_factor=realtime_factor,
            camera_hours_per_day_per_lane=24.0 * realtime_factor,
        )

    def as_projection(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "wall_seconds": self.wall_seconds,
            "media_seconds": self.media_seconds,
            "realtime_factor": self.realtime_factor,
            "camera_hours_per_day_per_lane": self.camera_hours_per_day_per_lane,
        }


@dataclass(frozen=True, slots=True)
class MageCapacityScenario:
    """One explicitly labeled composition of measured or hypothetical service rates."""

    scenario_id: str
    evidence_class: Literal["MEASURED_DERIVATION", "UNMEASURED_SCENARIO"]
    measured: bool
    wall_seconds: float
    media_seconds: float
    realtime_factor: float
    camera_hours_per_day_per_lane: float
    required_aggregate_realtime_factor: float
    required_logical_lanes: int
    includes_full_orchestration_wall: bool
    note: str
    codec_multiplier: float | None = None
    decoder_multiplier: float | None = None

    @classmethod
    def build(
        cls,
        *,
        scenario_id: str,
        evidence_class: Literal["MEASURED_DERIVATION", "UNMEASURED_SCENARIO"],
        measured: bool,
        wall_seconds: float,
        media_seconds: float,
        required_aggregate_realtime_factor: float,
        includes_full_orchestration_wall: bool,
        note: str,
        codec_multiplier: float | None = None,
        decoder_multiplier: float | None = None,
    ) -> MageCapacityScenario:
        wall = _positive_finite(wall_seconds, f"{scenario_id}.wall_seconds")
        media = _positive_finite(media_seconds, f"{scenario_id}.media_seconds")
        required = _positive_finite(
            required_aggregate_realtime_factor,
            f"{scenario_id}.required_aggregate_realtime_factor",
        )
        realtime_factor = media / wall
        return cls(
            scenario_id=scenario_id,
            evidence_class=evidence_class,
            measured=measured,
            wall_seconds=wall,
            media_seconds=media,
            realtime_factor=realtime_factor,
            camera_hours_per_day_per_lane=24.0 * realtime_factor,
            required_aggregate_realtime_factor=required,
            required_logical_lanes=math.ceil(required / realtime_factor),
            includes_full_orchestration_wall=includes_full_orchestration_wall,
            note=note,
            codec_multiplier=codec_multiplier,
            decoder_multiplier=decoder_multiplier,
        )

    def as_projection(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "evidence_class": self.evidence_class,
            "measured": self.measured,
            "wall_seconds": self.wall_seconds,
            "media_seconds": self.media_seconds,
            "realtime_factor": self.realtime_factor,
            "camera_hours_per_day_per_lane": self.camera_hours_per_day_per_lane,
            "required_aggregate_realtime_factor": self.required_aggregate_realtime_factor,
            "required_logical_lanes": self.required_logical_lanes,
            "includes_full_orchestration_wall": self.includes_full_orchestration_wall,
            "codec_multiplier": self.codec_multiplier,
            "decoder_multiplier": self.decoder_multiplier,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class MageProviderV2LocalBaseline:
    """Validated timing facts from the bounded Provider V2 retained qualification."""

    source_path: Path
    source_reference: str
    source_exact_sha256: str
    source_semantic_sha256: str
    variant_id: str
    media_seconds: float
    camera_count: int
    segment_count: int
    worker_count: int
    preparation_wall_seconds: float
    preparation_worker_sum_seconds: float
    preparation_orchestration_delta_seconds: float
    preparation_per_segment_seconds: tuple[float, ...]
    generation_stream_wall_seconds: float
    observation_sum_seconds: float
    observation_per_segment_seconds: tuple[float, ...]
    warm_observation_mean_seconds: float
    generation_sum_seconds: float
    generation_per_segment_seconds: tuple[float, ...]
    first_generation_seconds: float
    warm_generation_mean_seconds: float
    warm_time_to_first_token_mean_seconds: float
    processor_sum_seconds: float
    input_materialization_sum_seconds: float
    decode_sum_seconds: float
    endpoint_startup_to_health_seconds: float
    model_load_seconds: float

    def stages(self) -> tuple[MageCapacityStage, ...]:
        return (
            MageCapacityStage.measured(
                stage_id="provider_v2_preparation_full_wall",
                wall_seconds=self.preparation_wall_seconds,
                media_seconds=self.media_seconds,
            ),
            MageCapacityStage.measured(
                stage_id="provider_v2_preparation_worker_job_sum",
                wall_seconds=self.preparation_worker_sum_seconds,
                media_seconds=self.media_seconds,
            ),
            MageCapacityStage.measured(
                stage_id="mage_stream_run_wall",
                wall_seconds=self.generation_stream_wall_seconds,
                media_seconds=self.media_seconds,
            ),
            MageCapacityStage.measured(
                stage_id="mage_generation_sum",
                wall_seconds=self.generation_sum_seconds,
                media_seconds=self.media_seconds,
            ),
        )

    def scenarios(
        self,
        *,
        daily_camera_hours: float = DEFAULT_DAILY_CAMERA_HOURS,
        headroom: float = DEFAULT_CAPACITY_HEADROOM,
        codec_multiplier: float | None = None,
        decoder_multiplier: float | None = None,
    ) -> tuple[MageCapacityScenario, ...]:
        required = required_aggregate_realtime_factor(
            daily_camera_hours=daily_camera_hours,
            headroom=headroom,
        )
        scenarios = [
            MageCapacityScenario.build(
                scenario_id="local_serial_retained_whole_stage_walls",
                evidence_class="MEASURED_DERIVATION",
                measured=True,
                wall_seconds=(self.preparation_wall_seconds + self.generation_stream_wall_seconds),
                media_seconds=self.media_seconds,
                required_aggregate_realtime_factor=required,
                includes_full_orchestration_wall=True,
                note=(
                    "Sequential sum of retained preparation cache-manifest wall and Mage stream "
                    "run wall; endpoint/model lifecycle wall is not added."
                ),
            ),
            MageCapacityScenario.build(
                scenario_id="local_two_stage_overlap_retained_whole_stage_walls",
                evidence_class="MEASURED_DERIVATION",
                measured=True,
                wall_seconds=max(
                    self.preparation_wall_seconds,
                    self.generation_stream_wall_seconds,
                ),
                media_seconds=self.media_seconds,
                required_aggregate_realtime_factor=required,
                includes_full_orchestration_wall=True,
                note=(
                    "Mathematical two-stage overlap using retained whole-stage walls. It is not "
                    "a measured same-device overlap and is bounded by the slower preparation stage."
                ),
            ),
            MageCapacityScenario.build(
                scenario_id="local_two_stage_overlap_measured_job_sums_lower_bound",
                evidence_class="MEASURED_DERIVATION",
                measured=True,
                wall_seconds=max(
                    self.preparation_worker_sum_seconds,
                    self.generation_sum_seconds,
                ),
                media_seconds=self.media_seconds,
                required_aggregate_realtime_factor=required,
                includes_full_orchestration_wall=False,
                note=(
                    "Optimistic lower bound over measured per-job sums. It excludes orchestration, "
                    "durable admission, queueing, and endpoint handoff and is not an achieved run."
                ),
            ),
        ]
        if (codec_multiplier is None) != (decoder_multiplier is None):
            raise MageCapacityEvidenceError(
                "codec_multiplier and decoder_multiplier must be supplied together"
            )
        if codec_multiplier is not None and decoder_multiplier is not None:
            codec = _positive_finite(codec_multiplier, "codec_multiplier")
            decoder = _positive_finite(decoder_multiplier, "decoder_multiplier")
            scenarios.append(
                MageCapacityScenario.build(
                    scenario_id="target_separate_device_multiplier_scenario",
                    evidence_class="UNMEASURED_SCENARIO",
                    measured=False,
                    wall_seconds=max(
                        self.preparation_wall_seconds / codec,
                        self.generation_stream_wall_seconds / decoder,
                    ),
                    media_seconds=self.media_seconds,
                    required_aggregate_realtime_factor=required,
                    includes_full_orchestration_wall=True,
                    codec_multiplier=codec,
                    decoder_multiplier=decoder,
                    note=(
                        "Unmeasured independent codec/decoder multiplier scenario over retained "
                        "whole-stage walls; it is not H100 qualification."
                    ),
                )
            )
        return tuple(scenarios)

    def as_projection(self) -> dict[str, object]:
        return {
            "source_evidence": {
                "path": self.source_reference,
                "exact_sha256": self.source_exact_sha256,
                "semantic_sha256": self.source_semantic_sha256,
            },
            "variant_id": self.variant_id,
            "media_seconds": self.media_seconds,
            "camera_count": self.camera_count,
            "segment_count": self.segment_count,
            "worker_count": self.worker_count,
            "preparation": {
                "wall_seconds": self.preparation_wall_seconds,
                "worker_job_sum_seconds": self.preparation_worker_sum_seconds,
                "orchestration_delta_seconds": self.preparation_orchestration_delta_seconds,
                "per_segment_seconds": list(self.preparation_per_segment_seconds),
            },
            "generation": {
                "stream_run_wall_seconds": self.generation_stream_wall_seconds,
                "observation_sum_seconds": self.observation_sum_seconds,
                "observation_per_segment_seconds": list(self.observation_per_segment_seconds),
                "warm_observation_mean_seconds": self.warm_observation_mean_seconds,
                "generation_sum_seconds": self.generation_sum_seconds,
                "generation_per_segment_seconds": list(self.generation_per_segment_seconds),
                "first_generation_seconds": self.first_generation_seconds,
                "warm_generation_mean_seconds": self.warm_generation_mean_seconds,
                "warm_time_to_first_token_mean_seconds": (
                    self.warm_time_to_first_token_mean_seconds
                ),
                "processor_sum_seconds": self.processor_sum_seconds,
                "input_materialization_sum_seconds": self.input_materialization_sum_seconds,
                "decode_sum_seconds": self.decode_sum_seconds,
                "endpoint_startup_to_health_seconds": self.endpoint_startup_to_health_seconds,
                "model_load_seconds": self.model_load_seconds,
            },
            "stages": [stage.as_projection() for stage in self.stages()],
        }


def required_aggregate_realtime_factor(*, daily_camera_hours: float, headroom: float) -> float:
    """Return aggregate camera-throughput required over a 24-hour day."""

    daily = _positive_finite(daily_camera_hours, "daily_camera_hours")
    margin = _positive_finite(headroom, "headroom")
    if margin < 1.0:
        raise MageCapacityEvidenceError("headroom must be at least 1.0")
    return daily * margin / 24.0


def load_provider_v2_local_baseline(
    *,
    path: Path,
    expected_exact_sha256: str | None = None,
    expected_semantic_sha256: str | None = None,
    source_reference: str | None = None,
) -> MageProviderV2LocalBaseline:
    """Load and verify the exact bounded Provider V2 local qualification report."""

    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise MageCapacityEvidenceError(f"could not read retained evidence: {source}") from error
    exact_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_exact_sha256 is not None and exact_sha256 != _sha256(
        expected_exact_sha256,
        "expected_exact_sha256",
    ):
        raise MageCapacityEvidenceError("retained evidence exact SHA-256 differs")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MageCapacityEvidenceError("retained evidence is not valid JSON") from error
    root = _mapping(document, "report")
    if root.get("format_version") != MAGE_PROVIDER_V2_QUALIFICATION_VERSION:
        raise MageCapacityEvidenceError("retained evidence format_version differs")
    claimed_semantic = _sha256(root.get("semantic_sha256"), "report.semantic_sha256")
    semantic_projection = dict(root)
    semantic_projection.pop("semantic_sha256", None)
    computed_semantic = semantic_sha256(semantic_projection)
    if claimed_semantic != computed_semantic:
        raise MageCapacityEvidenceError("retained evidence semantic SHA-256 differs from content")
    if expected_semantic_sha256 is not None and claimed_semantic != _sha256(
        expected_semantic_sha256,
        "expected_semantic_sha256",
    ):
        raise MageCapacityEvidenceError("retained evidence expected semantic SHA-256 differs")
    if root.get("production_eligible") is not False:
        raise MageCapacityEvidenceError("local retained evidence must be production_eligible=false")

    scope = _mapping(root.get("scope"), "report.scope")
    sample_duration_ns = _positive_int(scope.get("sample_duration_ns"), "scope.sample_duration_ns")
    camera_count = _positive_int(scope.get("camera_count"), "scope.camera_count")
    segment_count = _positive_int(scope.get("segment_count"), "scope.segment_count")
    worker_count = _positive_int(scope.get("worker_count"), "scope.worker_count")
    if camera_count != 1 or worker_count != 1:
        raise MageCapacityEvidenceError("local convergence baseline requires one camera and worker")

    variant = _find_variant(root, MAGE_PROVIDER_V2_BOUNDED_VARIANT_ID)
    preparation = _mapping(variant.get("preparation"), "variant.preparation")
    generation = _mapping(variant.get("generation"), "variant.generation")
    measurement = _mapping(generation.get("measurement"), "variant.generation.measurement")

    preparation_per_segment = _ordered_seconds(
        preparation.get("per_segment"),
        segment_count=segment_count,
        seconds_field="preparation_seconds",
        label="variant.preparation.per_segment",
    )
    generation_rows = _ordered_rows(
        measurement.get("per_segment"),
        segment_count=segment_count,
        label="variant.generation.measurement.per_segment",
    )
    generation_per_segment = tuple(
        _positive_float(
            row.get("generate_seconds"),
            "variant.generation.measurement.per_segment[].generate_seconds",
        )
        for row in generation_rows
    )
    observation_per_segment = tuple(
        _positive_float(
            row.get("observation_seconds"),
            "variant.generation.measurement.per_segment[].observation_seconds",
        )
        for row in generation_rows
    )
    time_to_first_token_per_segment = tuple(
        _positive_float(
            row.get("time_to_first_token_seconds"),
            "variant.generation.measurement.per_segment[].time_to_first_token_seconds",
        )
        for row in generation_rows
    )
    processor_per_segment = tuple(
        _positive_float(
            row.get("processor_seconds"),
            "variant.generation.measurement.per_segment[].processor_seconds",
        )
        for row in generation_rows
    )
    input_materialization_per_segment = tuple(
        _positive_float(
            row.get("input_materialization_seconds"),
            "variant.generation.measurement.per_segment[].input_materialization_seconds",
        )
        for row in generation_rows
    )
    decode_per_segment = tuple(
        _positive_float(
            row.get("decode_seconds"),
            "variant.generation.measurement.per_segment[].decode_seconds",
        )
        for row in generation_rows
    )
    preparation_wall = _positive_float(
        preparation.get("wall_seconds"),
        "variant.preparation.wall_seconds",
    )
    preparation_sum = sum(preparation_per_segment)
    if preparation_wall + 1e-9 < preparation_sum:
        raise MageCapacityEvidenceError("preparation wall is smaller than measured job sum")
    generation_stream_wall = _positive_float(
        measurement.get("stream_run_wall_seconds"),
        "variant.generation.measurement.stream_run_wall_seconds",
    )
    observation_sum = sum(observation_per_segment)
    generation_sum = sum(generation_per_segment)
    if generation_stream_wall + 1e-9 < observation_sum:
        raise MageCapacityEvidenceError("stream run wall is smaller than observation sum")
    if len(generation_per_segment) < 2:
        raise MageCapacityEvidenceError("warm generation mean requires at least two segments")

    return MageProviderV2LocalBaseline(
        source_path=source,
        source_reference=(source_reference if source_reference is not None else str(path)),
        source_exact_sha256=exact_sha256,
        source_semantic_sha256=claimed_semantic,
        variant_id=MAGE_PROVIDER_V2_BOUNDED_VARIANT_ID,
        media_seconds=sample_duration_ns / 1_000_000_000,
        camera_count=camera_count,
        segment_count=segment_count,
        worker_count=worker_count,
        preparation_wall_seconds=preparation_wall,
        preparation_worker_sum_seconds=preparation_sum,
        preparation_orchestration_delta_seconds=preparation_wall - preparation_sum,
        preparation_per_segment_seconds=preparation_per_segment,
        generation_stream_wall_seconds=generation_stream_wall,
        observation_sum_seconds=observation_sum,
        observation_per_segment_seconds=observation_per_segment,
        warm_observation_mean_seconds=mean(observation_per_segment[1:]),
        generation_sum_seconds=generation_sum,
        generation_per_segment_seconds=generation_per_segment,
        first_generation_seconds=generation_per_segment[0],
        warm_generation_mean_seconds=mean(generation_per_segment[1:]),
        warm_time_to_first_token_mean_seconds=mean(time_to_first_token_per_segment[1:]),
        processor_sum_seconds=sum(processor_per_segment),
        input_materialization_sum_seconds=sum(input_materialization_per_segment),
        decode_sum_seconds=sum(decode_per_segment),
        endpoint_startup_to_health_seconds=_positive_float(
            measurement.get("endpoint_startup_to_health_seconds"),
            "variant.generation.measurement.endpoint_startup_to_health_seconds",
        ),
        model_load_seconds=_positive_float(
            measurement.get("model_load_seconds"),
            "variant.generation.measurement.model_load_seconds",
        ),
    )


def build_mage_25x_capacity_report(
    *,
    baseline: MageProviderV2LocalBaseline,
    daily_camera_hours: float = DEFAULT_DAILY_CAMERA_HOURS,
    headroom: float = DEFAULT_CAPACITY_HEADROOM,
    codec_multiplier: float | None = None,
    decoder_multiplier: float | None = None,
    repository_daily_recording_hours: float = DEFAULT_REPOSITORY_DAILY_RECORDING_HOURS,
    repository_camera_count: int = DEFAULT_REPOSITORY_CAMERA_COUNT,
) -> dict[str, object]:
    """Build a deterministic non-production capacity report."""

    required = required_aggregate_realtime_factor(
        daily_camera_hours=daily_camera_hours,
        headroom=headroom,
    )
    repository_recording_hours = _positive_finite(
        repository_daily_recording_hours,
        "repository_daily_recording_hours",
    )
    repository_cameras = _positive_int(repository_camera_count, "repository_camera_count")
    repository_independent_camera_rtf = required_aggregate_realtime_factor(
        daily_camera_hours=repository_recording_hours * repository_cameras,
        headroom=headroom,
    )
    payload: dict[str, object] = {
        "format_version": MAGE_25X_CAPACITY_REPORT_VERSION,
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "target": {
            "cycle_assumption": {
                "unit": "CAMERA_HOURS_PER_DAY",
                "daily_camera_hours": daily_camera_hours,
                "headroom": headroom,
                "required_aggregate_realtime_factor": required,
            },
            "repository_requirement_conflict": {
                "source": "governance/REQUIREMENTS.md",
                "unit": "RECORDING_HOURS_PER_DAY",
                "daily_recording_hours": repository_recording_hours,
                "independent_camera_count": repository_cameras,
                "independent_camera_stream_equivalent_rtf": (repository_independent_camera_rtf),
                "resolved": False,
            },
        },
        "baseline": baseline.as_projection(),
        "scenarios": [
            item.as_projection()
            for item in baseline.scenarios(
                daily_camera_hours=daily_camera_hours,
                headroom=headroom,
                codec_multiplier=codec_multiplier,
                decoder_multiplier=decoder_multiplier,
            )
        ],
        "decision": {
            "state": "HOLD",
            "reason": (
                "Local evidence defines the bottleneck, but the camera-hour versus "
                "recording-hour target conflict and target H100 service rates remain unresolved."
            ),
        },
    }
    payload["semantic_sha256"] = semantic_sha256(payload)
    return payload


def _find_variant(root: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
    variants = _mapping(root.get("variants"), "report.variants")
    candidates: list[Mapping[str, Any]] = []
    for value in variants.values():
        if isinstance(value, Mapping):
            candidates.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            candidates.extend(_mapping(item, "report.variants[]") for item in value)
        else:
            raise MageCapacityEvidenceError("report variant must be an object or object list")
    matches = [item for item in candidates if item.get("variant_id") == variant_id]
    if len(matches) != 1:
        raise MageCapacityEvidenceError(f"expected exactly one retained variant {variant_id}")
    return matches[0]


def _ordered_rows(
    value: object,
    *,
    segment_count: int,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MageCapacityEvidenceError(f"{label} must be a list")
    rows = tuple(_mapping(item, f"{label}[]") for item in value)
    if len(rows) != segment_count:
        raise MageCapacityEvidenceError(f"{label} count differs from segment_count")
    for expected_ordinal, row in enumerate(rows):
        ordinal = _nonnegative_int(row.get("ordinal"), f"{label}[].ordinal")
        if ordinal != expected_ordinal:
            raise MageCapacityEvidenceError(f"{label} ordinals must be contiguous and ordered")
    return rows


def _ordered_seconds(
    value: object,
    *,
    segment_count: int,
    seconds_field: str,
    label: str,
) -> tuple[float, ...]:
    return tuple(
        _positive_float(row.get(seconds_field), f"{label}[].{seconds_field}")
        for row in _ordered_rows(value, segment_count=segment_count, label=label)
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MageCapacityEvidenceError(f"{label} must be an object")
    return value


def _positive_finite(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise MageCapacityEvidenceError(f"{label} must be positive and finite")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MageCapacityEvidenceError(f"{label} must be numeric")
    return _positive_finite(float(value), label)


def _positive_int(value: object, label: str) -> int:
    number = _nonnegative_int(value, label)
    if number == 0:
        raise MageCapacityEvidenceError(f"{label} must be positive")
    return number


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MageCapacityEvidenceError(f"{label} must be a nonnegative integer")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MageCapacityEvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "DEFAULT_CAPACITY_HEADROOM",
    "DEFAULT_DAILY_CAMERA_HOURS",
    "DEFAULT_REPOSITORY_CAMERA_COUNT",
    "DEFAULT_REPOSITORY_DAILY_RECORDING_HOURS",
    "MAGE_25X_CAPACITY_REPORT_VERSION",
    "MAGE_PROVIDER_V2_BOUNDED_VARIANT_ID",
    "MageCapacityEvidenceError",
    "MageCapacityScenario",
    "MageCapacityStage",
    "MageProviderV2LocalBaseline",
    "build_mage_25x_capacity_report",
    "load_provider_v2_local_baseline",
    "required_aggregate_realtime_factor",
]
