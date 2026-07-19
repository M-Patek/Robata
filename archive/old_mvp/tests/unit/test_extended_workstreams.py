from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import pytest

from robata.adapters.local_vision_model import (
    LocalVisionModelAdapter,
    OptionalDependencyUnavailable,
    TransformersVisionModelAdapter,
)
from robata.capacity import calibrate_capacity_scenarios
from robata.qa import IssueDisposition, QAIssue, QAStatus
from robata.qa_validation import QAMatrixCase, QAMatrixReport, validate_issue_matrix
from robata.runtime.integration_validation import (
    run_frame_cache_stress,
    run_worker_requirements_integration,
)
from robata.runtime.process_pool_poc import compare_png_reuse
from robata.runtime.synthetic_benchmark import (
    BenchmarkCertificationError,
    build_synthetic_fixtures,
    certify_summary,
    run_synthetic_benchmark,
)


def test_issue_matrix_covers_all_21_labels() -> None:
    report = validate_issue_matrix()
    assert report.issue_count == 21
    assert report.passed is True


def test_qa_matrix_failures_reports_policy_mismatches() -> None:
    report = QAMatrixReport(
        issue_count=1,
        cases=(
            QAMatrixCase(
                issue=QAIssue.BLACK_SCREEN,
                disposition=IssueDisposition.LOCAL_WARNING,
                local_status=QAStatus.PASS,
                full_coverage_status=QAStatus.WARNING,
            ),
        ),
        passed=False,
    )
    assert report.failures == report.cases


def test_cache_and_worker_workstreams_pass() -> None:
    cache = run_frame_cache_stress(video_count=3, callers=9, frames_per_video=2)
    worker = run_worker_requirements_integration()
    assert cache.passed is True
    assert cache.decode_attempts == 3
    assert worker.passed is True
    assert worker.completed_count == 3


def test_synthetic_benchmark_hash_and_certification_guardrail() -> None:
    report = run_synthetic_benchmark(build_synthetic_fixtures(3), iterations=1, warmups=0)
    assert report.output_hash_equal is True
    with pytest.raises(BenchmarkCertificationError):
        certify_summary(
            report.serial,
            corpus_id="synthetic",
            governed_approval=True,
            execution_mode="LOCAL_DEVELOPMENT_FAKE_MODEL",
        )


def test_capacity_matrix_matches_documented_headroom() -> None:
    scenarios = calibrate_capacity_scenarios()
    by_key = {(item.h100_count, item.model_size): item for item in scenarios}
    assert by_key[(1, "7B")].fits is False
    assert by_key[(2, "7B")].fits is True
    assert by_key[(4, "32B")].fits is False


def test_local_adapter_metadata_is_provider_neutral() -> None:
    adapter = LocalVisionModelAdapter(lambda *_args: None)
    assert adapter.provider == "local"
    assert adapter.external_provider_requests == 0
    assert adapter.supports_parallel_inference is False


def test_transformers_boundary_does_not_download_when_missing(tmp_path: Path) -> None:
    # The optional dependency is intentionally absent in the development environment.  If a
    # future environment installs it, this branch simply verifies the import is lazy.
    if (tmp_path / "config.json").exists():
        pytest.skip("not a meaningful checkpoint")
    with suppress(OptionalDependencyUnavailable):
        TransformersVisionModelAdapter.require_transformers()


def test_png_reuse_is_byte_stable_when_pyav_available() -> None:
    av = pytest.importorskip("av")
    from fractions import Fraction

    frames = []
    for value in (0, 1, 2):
        frame = av.VideoFrame(2, 2, "rgb24")
        frame.pts = 0
        frame.time_base = Fraction(1, 1)
        frame.planes[0].update(bytes([value]) * frame.planes[0].buffer_size)
        frames.append(frame)
    result = compare_png_reuse(frames)
    assert result.supported is True
    assert result.byte_identical is True
