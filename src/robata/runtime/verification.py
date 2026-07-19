"""Offline verification for a published local fake-model mainline root."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.mainline import MainlineBundle, MainlineRunReport
from robata.contracts.video_export_v2 import CameraVideoExportManifestV2
from robata.runtime.execution import ExecutionEvidenceError, verify_execution_evidence


class LocalMainlineVerificationError(RuntimeError):
    """Raised when a published local mainline root fails offline verification."""


def _read_json(path: Path) -> tuple[bytes, Any]:
    try:
        data = path.read_bytes()
        value = json.loads(data)
    except (OSError, json.JSONDecodeError) as error:
        raise LocalMainlineVerificationError(
            f"cannot read JSON artifact {path}: {error}"
        ) from error
    return data, value


def _validate_canonical(path: Path, data: bytes, value: Any) -> None:
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise LocalMainlineVerificationError(f"cannot canonicalize {path}: {error}") from error
    if canonical != data:
        raise LocalMainlineVerificationError(f"{path.name} is not canonical JSON")


def verify_local_mainline_output(output_root: Path) -> dict[str, Any]:
    """Verify execution evidence and all typed primary artifacts in one output root."""

    root = Path(os.path.abspath(output_root))
    if root.is_symlink() or not root.is_dir():
        raise LocalMainlineVerificationError(f"output root is not a regular directory: {root}")
    try:
        execution_manifest = verify_execution_evidence(root)
    except ExecutionEvidenceError as error:
        raise LocalMainlineVerificationError(str(error)) from error

    analysis_root = root / "analysis"
    video_root = root / "video"
    report_bytes, _ = _read_json(analysis_root / "run-report.json")
    bundle_bytes, _ = _read_json(analysis_root / "mainline-bundle.json")
    video_bytes, _ = _read_json(video_root / "camera-video-export-manifest.json")
    try:
        report = MainlineRunReport.model_validate_json(report_bytes)
        bundle = MainlineBundle.model_validate_json(bundle_bytes)
        video_manifest = CameraVideoExportManifestV2.model_validate_json(video_bytes)
    except ValidationError as error:
        raise LocalMainlineVerificationError(
            f"typed artifact validation failed: {error}"
        ) from error
    _validate_canonical(analysis_root / "run-report.json", report_bytes, report)
    _validate_canonical(analysis_root / "mainline-bundle.json", bundle_bytes, bundle)
    _validate_canonical(
        video_root / "camera-video-export-manifest.json", video_bytes, video_manifest
    )

    if bundle.report != report:
        raise LocalMainlineVerificationError("bundle report does not match run-report.json")
    if report.run_id != execution_manifest.get("run_id"):
        raise LocalMainlineVerificationError("run report run_id does not match execution manifest")
    source = execution_manifest.get("source")
    if not isinstance(source, dict):
        raise LocalMainlineVerificationError("execution manifest source lineage is missing")
    if report.source_mcap_id != source.get("mcap_id"):
        raise LocalMainlineVerificationError("run report MCAP identity does not match manifest")
    if report.source_recording_identity != source.get("recording_identity"):
        raise LocalMainlineVerificationError(
            "run report recording identity does not match manifest"
        )
    if report.source_content_sha256 != source.get("content_sha256"):
        raise LocalMainlineVerificationError("run report source digest does not match manifest")

    video_lineage = execution_manifest.get("video")
    if not isinstance(video_lineage, dict):
        raise LocalMainlineVerificationError("execution manifest video lineage is missing")
    exact_video_sha = exact_bytes_sha256(video_bytes)
    if report.video_manifest_sha256 != exact_video_sha:
        raise LocalMainlineVerificationError("run report video manifest hash does not match bytes")
    if report.video_manifest_semantic_sha256 != video_manifest.semantic_content_sha256:
        raise LocalMainlineVerificationError(
            "run report video semantic hash does not match manifest"
        )
    if video_lineage.get("manifest_sha256") != exact_video_sha:
        raise LocalMainlineVerificationError("execution manifest video hash does not match bytes")
    if video_lineage.get("manifest_semantic_sha256") != video_manifest.semantic_content_sha256:
        raise LocalMainlineVerificationError("execution manifest video semantic hash mismatch")

    if report.real_provider_request_count != 0:
        raise LocalMainlineVerificationError("run report records provider requests")
    if any(event.production_eligible for event in bundle.events):
        raise LocalMainlineVerificationError(
            "local fake-model output contains a production-eligible event"
        )

    accounting = execution_manifest.get("accounting")
    if not isinstance(accounting, dict):
        raise LocalMainlineVerificationError("execution manifest accounting is missing")
    if accounting.get("event_count") != report.event_count:
        raise LocalMainlineVerificationError("execution event count does not match run report")
    if accounting.get("provider_request_count") != 0:
        raise LocalMainlineVerificationError("execution manifest records provider requests")

    return {
        "ok": True,
        "run_id": report.run_id,
        "run_status": report.status.value,
        "event_count": report.event_count,
        "provider_requests": 0,
        "production_eligible": False,
        "execution_manifest_semantic_sha256": execution_manifest["semantic_sha256"],
        "execution_manifest_sha256": exact_bytes_sha256(
            (root / "execution-manifest.json").read_bytes()
        ),
        "bundle_sha256": exact_bytes_sha256(bundle_bytes),
        "video_manifest_sha256": exact_video_sha,
        "artifact_count": len(execution_manifest.get("artifacts", [])),
    }


__all__ = ["LocalMainlineVerificationError", "verify_local_mainline_output"]
