"""Synthetic serial-vs-parallel benchmark fixtures and certification guardrails."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from robata.runtime.benchmark import BenchmarkSummary, run_repeated


class BenchmarkCertificationError(RuntimeError):
    """Raised when a local smoke result is incorrectly promoted to certification."""


@dataclass(frozen=True, slots=True)
class SyntheticFixture:
    fixture_id: str
    recording_duration_ns: int
    camera_count: int
    payload: bytes

    def __post_init__(self) -> None:
        if not self.fixture_id.strip():
            raise ValueError("fixture_id must be non-empty")
        if self.recording_duration_ns <= 0 or self.camera_count <= 0:
            raise ValueError("fixture duration and camera_count must be positive")
        if not self.payload:
            raise ValueError("payload must be non-empty")


@dataclass(frozen=True, slots=True)
class SyntheticBenchmarkReport:
    fixture_count: int
    serial: BenchmarkSummary
    parallel: BenchmarkSummary
    serial_output_sha256: str
    parallel_output_sha256: str
    output_hash_equal: bool
    speedup: float
    measurement_status: str = "NOT_MEASURED"
    certifying: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "fixture_count": self.fixture_count,
            "serial": self.serial.as_dict(),
            "parallel": self.parallel.as_dict(),
            "serial_output_sha256": self.serial_output_sha256,
            "parallel_output_sha256": self.parallel_output_sha256,
            "output_hash_equal": self.output_hash_equal,
            "speedup": self.speedup,
            "measurement_status": self.measurement_status,
            "certifying": self.certifying,
        }


def build_synthetic_fixtures(
    count: int = 8,
    *,
    recording_duration_ns: int = 3_600_000_000,
    camera_count: int = 6,
) -> tuple[SyntheticFixture, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    return tuple(
        SyntheticFixture(
            fixture_id=f"synthetic-{index:04d}",
            recording_duration_ns=recording_duration_ns,
            camera_count=camera_count,
            payload=hashlib.sha256(f"synthetic-frame-{index}".encode()).digest() * 4,
        )
        for index in range(count)
    )


def _process_fixture(fixture: SyntheticFixture) -> bytes:
    # This intentionally models CPU work without claiming model quality or capacity.
    digest = hashlib.sha256()
    for camera_index in range(fixture.camera_count):
        digest.update(fixture.payload)
        digest.update(camera_index.to_bytes(2, "big"))
    return fixture.fixture_id.encode() + b":" + digest.digest()


def _serial_output(fixtures: Iterable[SyntheticFixture]) -> bytes:
    return b"".join(_process_fixture(fixture) for fixture in fixtures)


def _parallel_output(fixtures: tuple[SyntheticFixture, ...], workers: int) -> bytes:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = tuple(pool.map(_process_fixture, fixtures))
    return b"".join(results)


def run_synthetic_benchmark(
    fixtures: tuple[SyntheticFixture, ...] | None = None,
    *,
    iterations: int = 3,
    warmups: int = 1,
    parallel_workers: int = 2,
) -> SyntheticBenchmarkReport:
    fixtures = fixtures or build_synthetic_fixtures()
    if not fixtures:
        raise ValueError("fixtures must be non-empty")
    if (
        isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, int)
        or parallel_workers <= 0
    ):
        raise ValueError("parallel_workers must be a positive integer")
    serial = run_repeated(
        lambda: _serial_output(fixtures),
        workload_id="synthetic-serial",
        recording_duration_ns=sum(item.recording_duration_ns for item in fixtures),
        iterations=iterations,
        warmups=warmups,
        camera_count=sum(item.camera_count for item in fixtures) // len(fixtures),
    )
    parallel = run_repeated(
        lambda: _parallel_output(fixtures, parallel_workers),
        workload_id="synthetic-parallel",
        recording_duration_ns=sum(item.recording_duration_ns for item in fixtures),
        iterations=iterations,
        warmups=warmups,
        camera_count=sum(item.camera_count for item in fixtures) // len(fixtures),
    )
    serial_hash = hashlib.sha256(_serial_output(fixtures)).hexdigest()
    parallel_hash = hashlib.sha256(_parallel_output(fixtures, parallel_workers)).hexdigest()
    speedup = serial.mean_elapsed_ms / parallel.mean_elapsed_ms
    return SyntheticBenchmarkReport(
        fixture_count=len(fixtures),
        serial=serial,
        parallel=parallel,
        serial_output_sha256=serial_hash,
        parallel_output_sha256=parallel_hash,
        output_hash_equal=serial_hash == parallel_hash,
        speedup=speedup,
    )


def certify_summary(
    summary: BenchmarkSummary,
    *,
    corpus_id: str,
    governed_approval: bool,
    execution_mode: str,
) -> BenchmarkSummary:
    """Promote only an explicitly governed, non-fake measurement to certifying status."""

    if not governed_approval:
        raise BenchmarkCertificationError("governed approval is required")
    if execution_mode == "LOCAL_DEVELOPMENT_FAKE_MODEL":
        raise BenchmarkCertificationError("fake/local smoke runs cannot be certifying")
    if not corpus_id.strip():
        raise BenchmarkCertificationError("corpus_id is required")
    return BenchmarkSummary(
        workload_id=summary.workload_id,
        samples=summary.samples,
        certifying=True,
        corpus_id=corpus_id,
    )


__all__ = [
    "BenchmarkCertificationError",
    "SyntheticBenchmarkReport",
    "SyntheticFixture",
    "build_synthetic_fixtures",
    "certify_summary",
    "run_synthetic_benchmark",
]
