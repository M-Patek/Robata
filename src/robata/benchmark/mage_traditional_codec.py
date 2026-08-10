"""Validate and compare the local Mage traditional H.264/HEVC qualification evidence.

The report intentionally separates three measured boundaries: cv-preinfer subprocess
service, the one-container workload envelope, and the host Docker envelope.  It then
combines the retained Mage observation timing only as an explicitly unmeasured
cross-route scenario.  No local result grants production eligibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Final

from robata.benchmark.mage_25x import (
    DEFAULT_CAPACITY_HEADROOM,
    DEFAULT_DAILY_CAMERA_HOURS,
    MageProviderV2LocalBaseline,
    required_aggregate_realtime_factor,
)
from robata.contracts.hashing import semantic_sha256

MAGE_TRADITIONAL_LOCAL_REPORT_VERSION: Final = "mage-traditional-codec-local-qualification-v1"
MAGE_TRADITIONAL_RECEIPT_SCHEMA: Final = "robata-mage-traditional-codec-container-receipt"
EXPECTED_POLICY: Final[dict[str, object]] = {
    "engine": "hevc",
    "target_canvas": 8,
    "group_size": 8,
    "images_per_group": 1,
    "patch": 16,
    "max_pixels": 65_536,
    "min_group_frames": 8,
    "max_group_frames": 128,
    "canvas_format": "jpg",
    "readiness_sum_threshold": 0,
    "avoid_keyframes": True,
}


class MageTraditionalEvidenceError(ValueError):
    """Traditional-codec evidence is missing, changed, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class TraditionalJobEvidence:
    ordinal: int
    job_id: str
    source_basename: str
    source_sha256: str
    source_byte_count: int
    wall_seconds: float
    core_seconds: float
    canvas_count: int
    position_rows: int
    loader_payload_sha256: str
    normalized_loader_meta_sha256: str
    exact_asset_set_sha256: str
    selected_block_count: int

    def as_projection(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "job_id": self.job_id,
            "source_basename": self.source_basename,
            "source_sha256": self.source_sha256,
            "source_byte_count": self.source_byte_count,
            "wall_seconds": self.wall_seconds,
            "core_seconds": self.core_seconds,
            "canvas_count": self.canvas_count,
            "position_rows": self.position_rows,
            "loader_payload_sha256": self.loader_payload_sha256,
            "normalized_loader_meta_sha256": self.normalized_loader_meta_sha256,
            "exact_asset_set_sha256": self.exact_asset_set_sha256,
            "selected_block_count": self.selected_block_count,
        }


@dataclass(frozen=True, slots=True)
class TraditionalReceiptEvidence:
    source_path: Path
    source_reference: str
    exact_sha256: str
    content_sha256: str
    run_id: str
    policy_sha256: str
    job_manifest_sha256: str
    workload_wall_seconds: float
    sum_job_wall_seconds: float
    jobs: tuple[TraditionalJobEvidence, ...]
    toolchain_projection: Mapping[str, Any]

    @property
    def core_sum_seconds(self) -> float:
        return sum(item.core_seconds for item in self.jobs)

    @property
    def mean_job_wall_seconds(self) -> float:
        return mean(item.wall_seconds for item in self.jobs)

    @property
    def max_job_wall_seconds(self) -> float:
        return max(item.wall_seconds for item in self.jobs)

    def as_projection(self) -> dict[str, object]:
        return {
            "source_evidence": {
                "path": self.source_reference,
                "exact_sha256": self.exact_sha256,
                "receipt_content_sha256": self.content_sha256,
            },
            "run_id": self.run_id,
            "policy_sha256": self.policy_sha256,
            "job_manifest_sha256": self.job_manifest_sha256,
            "workload_wall_seconds": self.workload_wall_seconds,
            "sum_job_wall_seconds": self.sum_job_wall_seconds,
            "core_sum_seconds": self.core_sum_seconds,
            "mean_job_wall_seconds": self.mean_job_wall_seconds,
            "max_job_wall_seconds": self.max_job_wall_seconds,
            "jobs": [item.as_projection() for item in self.jobs],
            "toolchain": dict(self.toolchain_projection),
        }


def load_traditional_receipt(
    *,
    path: Path,
    expected_exact_sha256: str | None = None,
    source_reference: str | None = None,
) -> TraditionalReceiptEvidence:
    source = Path(path).expanduser().resolve()
    raw = _read_bytes(source, "traditional receipt")
    exact_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_exact_sha256 is not None and exact_sha256 != _sha256(
        expected_exact_sha256, "expected receipt exact SHA-256"
    ):
        raise MageTraditionalEvidenceError("traditional receipt exact SHA-256 differs")
    document = _json_object(raw, "traditional receipt")
    if document.get("schema_name") != MAGE_TRADITIONAL_RECEIPT_SCHEMA:
        raise MageTraditionalEvidenceError("traditional receipt schema_name differs")
    if document.get("schema_version") != 1:
        raise MageTraditionalEvidenceError("traditional receipt schema_version differs")
    if document.get("production_eligible") is not False:
        raise MageTraditionalEvidenceError(
            "local traditional receipt must not be production eligible"
        )
    content_sha256 = _sha256(document.get("receipt_content_sha256"), "receipt_content_sha256")
    projection = dict(document)
    projection.pop("receipt_content_sha256", None)
    if _canonical_sha256(projection) != content_sha256:
        raise MageTraditionalEvidenceError("traditional receipt content identity differs")
    if _mapping(document.get("policy"), "receipt.policy") != EXPECTED_POLICY:
        raise MageTraditionalEvidenceError("traditional receipt policy differs from control")
    scope = _mapping(document.get("scope"), "receipt.scope")
    if (
        scope.get("model_loaded") is not False
        or scope.get("semantic_generation_executed") is not False
        or scope.get("semantic_quality_evaluated") is not False
    ):
        raise MageTraditionalEvidenceError("preparation receipt overclaims semantic execution")

    rows = _sequence(document.get("jobs"), "receipt.jobs")
    measurement = _mapping(document.get("measurement"), "receipt.measurement")
    if _positive_int(measurement.get("job_count"), "measurement.job_count") != len(rows):
        raise MageTraditionalEvidenceError("traditional receipt job count differs")
    measured_per_job = tuple(
        _positive_float(value, "measurement.per_job_wall_seconds[]")
        for value in _sequence(
            measurement.get("per_job_wall_seconds"),
            "measurement.per_job_wall_seconds",
        )
    )
    if len(measured_per_job) != len(rows):
        raise MageTraditionalEvidenceError("per-job timing count differs")

    jobs: list[TraditionalJobEvidence] = []
    for ordinal, raw_job in enumerate(rows):
        job = _mapping(raw_job, f"receipt.jobs[{ordinal}]")
        wall = _positive_float(job.get("wall_seconds"), f"jobs[{ordinal}].wall_seconds")
        if not math.isclose(wall, measured_per_job[ordinal], rel_tol=0.0, abs_tol=1e-9):
            raise MageTraditionalEvidenceError("job wall differs from measurement vector")
        if job.get("return_code") != 0:
            raise MageTraditionalEvidenceError("traditional provider job did not succeed")
        source_projection = _mapping(job.get("source"), f"jobs[{ordinal}].source")
        output = _mapping(job.get("output"), f"jobs[{ordinal}].output")
        if output.get("loader_compatible") is not True:
            raise MageTraditionalEvidenceError("traditional output is not Mage-loader compatible")
        if output.get("semantic_quality_evaluated") is not False:
            raise MageTraditionalEvidenceError("traditional output overclaims semantic quality")
        positions = _mapping(output.get("src_positions"), f"jobs[{ordinal}].src_positions")
        shape = _sequence(positions.get("shape"), f"jobs[{ordinal}].src_positions.shape")
        if len(shape) != 2 or shape[1] != 3:
            raise MageTraditionalEvidenceError("traditional patch-position shape differs")
        row_count = _positive_int(positions.get("row_count"), "src_positions.row_count")
        if shape[0] != row_count:
            raise MageTraditionalEvidenceError("traditional patch-position row count differs")
        meta = _mapping(output.get("meta"), f"jobs[{ordinal}].output.meta")
        timing = _mapping(meta.get("timing_sec"), f"jobs[{ordinal}].meta.timing_sec")
        groups = _sequence(meta.get("groups"), f"jobs[{ordinal}].meta.groups")
        selected_blocks = 0
        for group in groups:
            selected_blocks += _nonnegative_int(
                _mapping(group, "meta.groups[]").get("selected_blocks"),
                "meta.groups[].selected_blocks",
            )
        source_path = str(source_projection.get("path"))
        jobs.append(
            TraditionalJobEvidence(
                ordinal=ordinal,
                job_id=_nonempty_string(job.get("job_id"), "job_id"),
                source_basename=Path(source_path).name,
                source_sha256=_sha256(source_projection.get("sha256"), "source.sha256"),
                source_byte_count=_positive_int(
                    source_projection.get("byte_count"), "source.byte_count"
                ),
                wall_seconds=wall,
                core_seconds=_positive_float(timing.get("total"), "meta.timing_sec.total"),
                canvas_count=_positive_int(output.get("canvas_count"), "output.canvas_count"),
                position_rows=row_count,
                loader_payload_sha256=_sha256(
                    output.get("loader_payload_sha256"), "loader_payload_sha256"
                ),
                normalized_loader_meta_sha256=_sha256(
                    output.get("normalized_loader_meta_sha256"),
                    "normalized_loader_meta_sha256",
                ),
                exact_asset_set_sha256=_sha256(
                    output.get("exact_asset_set_sha256"), "exact_asset_set_sha256"
                ),
                selected_block_count=selected_blocks,
            )
        )

    sum_job_wall = _positive_float(
        measurement.get("sum_job_wall_seconds"), "measurement.sum_job_wall_seconds"
    )
    if not math.isclose(
        sum_job_wall, sum(item.wall_seconds for item in jobs), rel_tol=0.0, abs_tol=1e-9
    ):
        raise MageTraditionalEvidenceError("sum_job_wall_seconds differs from job timings")
    workload_wall = _positive_float(
        measurement.get("workload_wall_seconds"), "measurement.workload_wall_seconds"
    )
    if workload_wall < sum_job_wall:
        raise MageTraditionalEvidenceError("workload wall cannot be below sequential job sum")

    return TraditionalReceiptEvidence(
        source_path=source,
        source_reference=source_reference or source.as_posix(),
        exact_sha256=exact_sha256,
        content_sha256=content_sha256,
        run_id=_nonempty_string(document.get("run_id"), "run_id"),
        policy_sha256=_sha256(document.get("policy_sha256"), "policy_sha256"),
        job_manifest_sha256=_sha256(document.get("job_manifest_sha256"), "job_manifest_sha256"),
        workload_wall_seconds=workload_wall,
        sum_job_wall_seconds=sum_job_wall,
        jobs=tuple(jobs),
        toolchain_projection=_mapping(document.get("toolchain"), "receipt.toolchain"),
    )


def load_host_measurement(
    *, path: Path, expected_image_digest: str | None = None, source_reference: str | None = None
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    raw = _read_bytes(source, "host measurement")
    document = _json_object(raw, "host measurement")
    if document.get("docker_run_rc") != 0 or document.get("network") != "none":
        raise MageTraditionalEvidenceError(
            "host measurement did not use successful network-none Docker"
        )
    if document.get("platform") != "linux/amd64":
        raise MageTraditionalEvidenceError("host measurement platform differs")
    image_id = _digest_reference(document.get("image_id"), "host.image_id")
    if expected_image_digest is not None and image_id != _sha256(
        expected_image_digest, "expected image digest"
    ):
        raise MageTraditionalEvidenceError("traditional container image digest differs")
    return {
        "source_evidence": {
            "path": source_reference or source.as_posix(),
            "exact_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "docker_run_rc": 0,
        "network": "none",
        "host_wall_seconds": _positive_float(
            document.get("host_wall_seconds"), "host.host_wall_seconds"
        ),
        "image": _nonempty_string(document.get("image"), "host.image"),
        "image_digest": image_id,
        "platform": "linux/amd64",
    }


def verify_receipt_sources(
    *,
    receipt: TraditionalReceiptEvidence,
    baseline_report_path: Path,
    control_source_verifier: Callable[[Path], tuple[str, int]] | None = None,
) -> tuple[dict[str, object], ...]:
    """Cross-check source identities, re-reading live bytes unless explicitly overridden.

    The injectable verifier is an internal artifact-replay/test seam. Production callers
    omit it and therefore retain fail-closed exact-file verification.
    """

    raw = _read_bytes(Path(baseline_report_path).expanduser().resolve(), "baseline report")
    document = _json_object(raw, "baseline report")
    variants = _mapping(document.get("variants"), "baseline.variants")
    bounded = _sequence(variants.get("provider_v2_bounded"), "provider_v2_bounded")
    matches = [
        _mapping(item, "provider_v2_bounded[]")
        for item in bounded
        if _mapping(item, "provider_v2_bounded[]").get("variant_id") == "provider-v2-max-side-448"
    ]
    if len(matches) != 1:
        raise MageTraditionalEvidenceError("bounded Provider V2 baseline is missing")
    preparation = _mapping(matches[0].get("preparation"), "baseline.preparation")
    baseline_jobs = _sequence(preparation.get("per_segment"), "baseline.preparation.per_segment")
    if len(baseline_jobs) != len(receipt.jobs):
        raise MageTraditionalEvidenceError("traditional and DCVC segment counts differ")
    verify_control_source = (
        _exact_file if control_source_verifier is None else control_source_verifier
    )
    verified: list[dict[str, object]] = []
    for ordinal, (traditional, raw_control) in enumerate(
        zip(receipt.jobs, baseline_jobs, strict=True)
    ):
        control = _mapping(raw_control, f"baseline.jobs[{ordinal}]")
        if control.get("ordinal") != ordinal:
            raise MageTraditionalEvidenceError("baseline preparation ordinals differ")
        control_path = Path(_nonempty_string(control.get("source_path"), "baseline.source_path"))
        if control_path.name != traditional.source_basename:
            raise MageTraditionalEvidenceError("traditional source basename differs from control")
        sha256, byte_count = verify_control_source(control_path)
        if sha256 != traditional.source_sha256 or byte_count != traditional.source_byte_count:
            raise MageTraditionalEvidenceError("traditional source bytes differ from control")
        verified.append(
            {
                "ordinal": ordinal,
                "path": str(control_path),
                "byte_count": byte_count,
                "sha256": sha256,
            }
        )
    return tuple(verified)


def build_traditional_local_qualification_report(
    *,
    baseline: MageProviderV2LocalBaseline,
    baseline_report_path: Path,
    receipt: TraditionalReceiptEvidence,
    host_measurement: Mapping[str, object],
    single_receipt: TraditionalReceiptEvidence | None = None,
    control_source_verifier: Callable[[Path], tuple[str, int]] | None = None,
    daily_camera_hours: float = DEFAULT_DAILY_CAMERA_HOURS,
    headroom: float = DEFAULT_CAPACITY_HEADROOM,
) -> dict[str, object]:
    if receipt.jobs.__len__() != baseline.segment_count:
        raise MageTraditionalEvidenceError(
            "traditional receipt segment count differs from baseline"
        )
    source_proof = verify_receipt_sources(
        receipt=receipt,
        baseline_report_path=baseline_report_path,
        control_source_verifier=control_source_verifier,
    )
    required = required_aggregate_realtime_factor(
        daily_camera_hours=daily_camera_hours, headroom=headroom
    )
    media_seconds = baseline.media_seconds
    host_wall = _positive_float(host_measurement.get("host_wall_seconds"), "host wall")
    traditional_stage_rtf = media_seconds / receipt.sum_job_wall_seconds
    workload_rtf = media_seconds / receipt.workload_wall_seconds
    host_rtf = media_seconds / host_wall
    serial_cross_route_wall = receipt.workload_wall_seconds + baseline.observation_sum_seconds
    overlap_cross_route_wall = max(receipt.workload_wall_seconds, baseline.observation_sum_seconds)

    repeatability: dict[str, object] = {
        "single_segment_receipt_present": single_receipt is not None,
        "loader_payload_equal": None,
        "normalized_loader_meta_equal": None,
        "raw_asset_set_equal": None,
        "interpretation": (
            "Raw meta.json contains path/timing observations; loader payload and normalized "
            "metadata are the recomputation comparison boundaries. Artifact replay "
            "remains byte-exact."
        ),
    }
    if single_receipt is not None:
        if len(single_receipt.jobs) != 1:
            raise MageTraditionalEvidenceError("single receipt must contain one job")
        first = receipt.jobs[0]
        probe = single_receipt.jobs[0]
        if first.source_sha256 != probe.source_sha256:
            raise MageTraditionalEvidenceError("single and five-segment source differ")
        repeatability.update(
            {
                "single_receipt_source": single_receipt.source_reference,
                "single_receipt_exact_sha256": single_receipt.exact_sha256,
                "loader_payload_equal": first.loader_payload_sha256 == probe.loader_payload_sha256,
                "normalized_loader_meta_equal": first.normalized_loader_meta_sha256
                == probe.normalized_loader_meta_sha256,
                "raw_asset_set_equal": first.exact_asset_set_sha256 == probe.exact_asset_set_sha256,
            }
        )
        if (
            not repeatability["loader_payload_equal"]
            or not repeatability["normalized_loader_meta_equal"]
        ):
            raise MageTraditionalEvidenceError("traditional loader payload is not repeatable")

    selected_blocks = sum(item.selected_block_count for item in receipt.jobs)
    report: dict[str, object] = {
        "report_version": MAGE_TRADITIONAL_LOCAL_REPORT_VERSION,
        "production_eligible": False,
        "measurement_scope": {
            "media_seconds": media_seconds,
            "camera_count": baseline.camera_count,
            "segment_count": baseline.segment_count,
            "worker_count": 1,
            "generation_lane_count": 1,
            "hardware": (
                "local Docker Desktop CPU preparation; retained RTX 4060 Mage decoder timing"
            ),
        },
        "capacity_target": {
            "daily_camera_hours": float(daily_camera_hours),
            "headroom": float(headroom),
            "required_aggregate_realtime_factor": required,
            "repository_recording_hour_conflict_resolved": False,
        },
        "control": baseline.as_projection(),
        "traditional_preparation": {
            **receipt.as_projection(),
            "host_measurement": dict(host_measurement),
            "verified_sources": list(source_proof),
            "service_rates": {
                "cv_preinfer_core_sum_seconds": receipt.core_sum_seconds,
                "provider_job_sum_seconds": receipt.sum_job_wall_seconds,
                "container_workload_wall_seconds": receipt.workload_wall_seconds,
                "host_docker_wall_seconds": host_wall,
                "provider_job_sum_rtf": traditional_stage_rtf,
                "container_workload_rtf": workload_rtf,
                "host_envelope_rtf": host_rtf,
            },
            "repeatability": repeatability,
        },
        "comparison": {
            "dcvc_worker_job_sum_to_traditional_job_sum_speedup": (
                baseline.preparation_worker_sum_seconds / receipt.sum_job_wall_seconds
            ),
            "dcvc_full_wall_to_traditional_workload_speedup": (
                baseline.preparation_wall_seconds / receipt.workload_wall_seconds
            ),
            "codec_bottleneck_transferred_to_decoder": (
                receipt.mean_job_wall_seconds < baseline.warm_observation_mean_seconds
            ),
            "measured_preparation_only": {
                "rtf": traditional_stage_rtf,
                "required_logical_lanes_for_target": math.ceil(required / traditional_stage_rtf),
            },
            "unmeasured_cross_route_scenarios": [
                {
                    "scenario_id": "traditional_workload_plus_retained_observation_serial",
                    "evidence_class": "UNMEASURED_SCENARIO",
                    "wall_seconds": serial_cross_route_wall,
                    "rtf": media_seconds / serial_cross_route_wall,
                    "required_logical_lanes_for_target": math.ceil(
                        required / (media_seconds / serial_cross_route_wall)
                    ),
                },
                {
                    "scenario_id": "traditional_workload_retained_observation_ideal_overlap",
                    "evidence_class": "UNMEASURED_SCENARIO",
                    "wall_seconds": overlap_cross_route_wall,
                    "rtf": media_seconds / overlap_cross_route_wall,
                    "required_logical_lanes_for_target": math.ceil(
                        required / (media_seconds / overlap_cross_route_wall)
                    ),
                },
            ],
        },
        "quality": {
            "mage_generation_executed_with_traditional_assets": False,
            "business_projection_compared": False,
            "loader_asset_contract": "PASS",
            "selected_block_count": selected_blocks,
            "salience_coverage": (
                "FALLBACK_OR_FULL_FRAME_ONLY" if selected_blocks == 0 else "CODEC_SALIENCE_OBSERVED"
            ),
            "representative_quality_gate": "NOT_RUN",
        },
        "decision": {
            "state": "HOLD_TRADITIONAL",
            "preparation_performance_gate": "PASS",
            "reason": (
                "Traditional H.264 preparation is faster than the retained DCVC control and no "
                "longer dominates the retained decoder timing, but real Mage generation, business "
                "quality, segment-ready integration, and representative salience remain unmeasured."
            ),
        },
    }
    report["semantic_sha256"] = semantic_sha256(report)
    return report


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise MageTraditionalEvidenceError(f"could not read {label}: {path}") from error


def _json_object(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MageTraditionalEvidenceError(f"{label} is not valid JSON") from error
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MageTraditionalEvidenceError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MageTraditionalEvidenceError(f"{label} must be an array")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MageTraditionalEvidenceError(f"{label} must be a non-empty string")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MageTraditionalEvidenceError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise MageTraditionalEvidenceError(f"{label} must be positive and finite")
    return number


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MageTraditionalEvidenceError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    number = _nonnegative_int(value, label)
    if number == 0:
        raise MageTraditionalEvidenceError(f"{label} must be positive")
    return number


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MageTraditionalEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _digest_reference(value: object, label: str) -> str:
    text = _nonempty_string(value, label)
    prefix = "sha256:"
    if not text.startswith(prefix):
        raise MageTraditionalEvidenceError(f"{label} must be sha256:<digest>")
    return _sha256(text[len(prefix) :], label)


def _exact_file(path: Path) -> tuple[str, int]:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise MageTraditionalEvidenceError(f"could not read control source: {resolved}") from error
    if byte_count <= 0:
        raise MageTraditionalEvidenceError(f"control source is empty: {resolved}")
    return digest.hexdigest(), byte_count


__all__ = [
    "EXPECTED_POLICY",
    "MAGE_TRADITIONAL_LOCAL_REPORT_VERSION",
    "MageTraditionalEvidenceError",
    "TraditionalJobEvidence",
    "TraditionalReceiptEvidence",
    "build_traditional_local_qualification_report",
    "load_host_measurement",
    "load_traditional_receipt",
    "verify_receipt_sources",
]
