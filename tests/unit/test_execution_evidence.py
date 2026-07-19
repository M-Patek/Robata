from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robata.runtime.execution import (
    ExecutionEvidenceError,
    execution_manifest_semantic_sha256,
    verify_execution_evidence,
    write_execution_evidence,
)


def _report(*, duration_ms: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        run_id="11111111-1111-1111-1111-111111111111",
        status="PRIMARY_COMPLETE",
        source_mcap_id="22222222-2222-2222-2222-222222222222",
        source_recording_identity="a" * 64,
        source_content_sha256="b" * 64,
        video_manifest_artifact_id="33333333-3333-3333-3333-333333333333",
        video_manifest_sha256="c" * 64,
        video_manifest_semantic_sha256="d" * 64,
        pipeline_version="local-mainline-v0",
        config_sha256="e" * 64,
        started_at="2026-07-19T10:00:00Z",
        completed_at="2026-07-19T10:00:01Z",
        duration_ms=duration_ms,
        window_count=1,
        package_count=1,
        inference_attempt_count=2,
        inference_success_count=2,
        inference_failure_count=0,
        inference_invalid_output_count=0,
        candidate_count=1,
        event_count=1,
        fake_inference_attempt_count=2,
        stages=(
            SimpleNamespace(
                stage="WINDOWING",
                status="SUCCEEDED",
                planned=1,
                succeeded=1,
                failed=0,
                pending=0,
                skipped=0,
                duration_ms=duration_ms,
            ),
        ),
    )


def test_execution_evidence_is_canonical_and_verifiable(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "analysis").mkdir(parents=True)
    (root / "video").mkdir()
    (root / "analysis" / "run-report.json").write_text("volatile", encoding="utf-8")
    (root / "video" / "cam_01.mp4").write_bytes(b"video")

    evidence = write_execution_evidence(
        root,
        report=_report(),
        model=SimpleNamespace(provider="fake", model_name="deterministic", model_version="v0"),
    )

    manifest_path = root / "execution-manifest.json"
    audit_path = root / "execution-audit.ndjson"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert evidence.manifest_sha256
    assert manifest["semantic_sha256"] == execution_manifest_semantic_sha256(manifest)
    assert verify_execution_evidence(root)["run_id"] == manifest["run_id"]
    assert all(json.loads(line) for line in audit_path.read_bytes().splitlines())
    assert str(root) not in audit_path.read_text(encoding="utf-8")


def test_semantic_hash_ignores_wall_clock_accounting_but_exact_inventory_detects_mutation(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root in (root_a, root_b):
        root.mkdir()
        (root / "artifact.bin").write_bytes(b"same")
    first = write_execution_evidence(root_a, report=_report(duration_ms=10))
    second = write_execution_evidence(root_b, report=_report(duration_ms=999))
    assert first.manifest_semantic_sha256 == second.manifest_semantic_sha256

    (root_a / "artifact.bin").write_bytes(b"changed")
    with pytest.raises(ExecutionEvidenceError, match="hash mismatch"):
        verify_execution_evidence(root_a)


def test_execution_evidence_rejects_provider_requests(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    with pytest.raises(ExecutionEvidenceError, match="zero provider requests"):
        write_execution_evidence(root, report=_report(), provider_requests=1)
