from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from robata.runtime.e2e_participation import E2EParticipationBoundary


def _load_runner():
    script = Path(__file__).parents[2] / "scripts" / "run_local_qwen_canonical_e2e.py"
    spec = importlib.util.spec_from_file_location("robata_local_qwen_e2e_report", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _fragment(*counts: int) -> SimpleNamespace:
    assert len(counts) == len(E2EParticipationBoundary)
    return SimpleNamespace(
        stages=tuple(SimpleNamespace(observed_span_count=count) for count in counts)
    )


def test_scheduler_success_is_derived_without_sample_specific_work_count() -> None:
    assert RUNNER._scheduler_succeeded(
        {
            "work_item_states": {"SUCCEEDED": 10},
            "attempt_outcomes": {"SUCCEEDED": 10},
            "expected_windows": 2,
            "stream_window_results": 2,
            "stream_window_evidence_commits": 2,
        }
    )
    assert not RUNNER._scheduler_succeeded(
        {
            "work_item_states": {"SUCCEEDED": 9, "FAILED": 1},
            "attempt_outcomes": {"SUCCEEDED": 9, "FAILED": 1},
            "expected_windows": 2,
            "stream_window_results": 2,
            "stream_window_evidence_commits": 2,
        }
    )


def test_endpoint_preflight_failure_is_not_reported_as_quality_gate_failure() -> None:
    error = RuntimeError("endpoint unavailable")
    fragment = _fragment(0, 0, 0, 0, 0, 0, 0)
    boundary = RUNNER._failure_boundary(
        error=error,
        error_phase="ENDPOINT_PREFLIGHT",
        fragment=fragment,
    )

    outcomes = RUNNER._semantic_stage_outcomes(
        error=error,
        failure_boundary=boundary,
        source={"measurement_status": "NOT_MEASURED"},
        inference={},
        pipeline={},
    )

    assert boundary is E2EParticipationBoundary.ORCHESTRATION
    assert outcomes["source"]["status"] == "NOT_ADMITTED_OR_UNPROVEN"
    assert outcomes["reduction"]["status"] == "NOT_REACHED_OR_UNPROVEN"


def test_persisted_six_camera_media_report_is_the_source_admission_basis() -> None:
    source = {
        "media_admission": {
            "camera_ledger_count": 6,
            "semantic_sha256": "a" * 64,
            "path": "media-quality-report.json",
        }
    }
    outcomes = RUNNER._semantic_stage_outcomes(
        error=None,
        failure_boundary=None,
        source=source,
        inference={"lineage_complete": True, "dense_unresolved_coordinates": []},
        pipeline={},
    )

    assert RUNNER._source_admitted(source) is True
    assert outcomes["source"]["status"] == "ADMITTED"
    assert outcomes["reduction"]["status"] == "SUCCEEDED"


def test_incomplete_canonical_run_maps_to_reduction_boundary() -> None:
    error = RuntimeError("canonical run ended as INCOMPLETE: dense QA unresolved")
    boundary = RUNNER._failure_boundary(
        error=error,
        error_phase="CANONICAL_RUN",
        fragment=_fragment(1, 1, 1, 1, 1, 1, 1),
    )

    assert boundary is E2EParticipationBoundary.REDUCTION


def test_endpoint_preflight_requires_the_pinned_checkpoint_manifest_identity() -> None:
    expected_digest = "b" * 64
    payload = {
        "status": "READY",
        "model_identifier": RUNNER.MODEL_NAME,
        "model_version": RUNNER.MODEL_VERSION,
        "loaded": True,
        "concurrency": 1,
        "checkpoint_identity": {
            "manifest_version": "local-hf-checkpoint-manifest-v1",
            "manifest_sha256": expected_digest,
            "included_file_count": 4,
            "hf_revision": "local-test-revision",
        },
    }

    assert (
        RUNNER._validate_endpoint_health(
            payload,
            expected_checkpoint_manifest_sha256=expected_digest,
        )
        == payload
    )

    with pytest.raises(RuntimeError, match=r"checkpoint_identity\.manifest_sha256"):
        RUNNER._validate_endpoint_health(
            payload,
            expected_checkpoint_manifest_sha256="c" * 64,
        )
