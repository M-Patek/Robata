"""Deterministic local execution evidence and artifact verification helpers.

The execution evidence is deliberately scoped to the local development mainline.  It
contains no provider credentials, network metadata, source paths, or raw frame bytes.  The
manifest inventories every published file (except the manifest and audit file themselves)
with exact SHA-256 hashes, while its semantic hash excludes volatile wall-clock accounting so
that the same source/configuration has a stable execution identity.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256

EXECUTION_MANIFEST_FILENAME = "execution-manifest.json"
EXECUTION_AUDIT_FILENAME = "execution-audit.ndjson"
EXECUTION_SCHEMA_VERSION = "1.0"
EXECUTION_MODE = "LOCAL_DEVELOPMENT_FAKE_MODEL"


class ExecutionEvidenceError(RuntimeError):
    """Raised when local execution evidence cannot be created or verified."""


@dataclass(frozen=True, slots=True)
class PublishedExecutionEvidence:
    """Hashes returned after execution evidence is atomically written."""

    manifest_sha256: str
    manifest_semantic_sha256: str
    audit_sha256: str
    artifact_count: int


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        enum_value = value.value
        if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
            return enum_value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return value


def _get(value: object, name: str, default: Any = None) -> Any:
    return getattr(value, name, default)


def _artifact_entries(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise ExecutionEvidenceError(f"execution root is not a regular directory: {root}")
    entries: list[dict[str, Any]] = []
    excluded = {EXECUTION_MANIFEST_FILENAME, EXECUTION_AUDIT_FILENAME}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            if path.is_symlink():
                raise ExecutionEvidenceError(f"symlink is not allowed in execution output: {path}")
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        data = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "size_bytes": len(data),
                "sha256": exact_bytes_sha256(data),
            }
        )
    return entries


def _stage_entries(report: object) -> list[dict[str, Any]]:
    stages = _get(report, "stages", ()) or ()
    result: list[dict[str, Any]] = []
    for stage in stages:
        value = _json_value(stage)
        if isinstance(value, Mapping):
            result.append({str(key): value[key] for key in value})
        else:
            result.append({"value": value})
    return result


def _semantic_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable identity projection for one execution manifest.

    Wall-clock timestamps, measured durations, and exact artifact bytes are observational
    fields.  The exact artifact hashes remain available for verification; semantic identity
    intentionally consists of source/config/provider accounting plus relative artifact paths.
    """

    stages = []
    for raw_stage in manifest.get("stages", []):
        if not isinstance(raw_stage, Mapping):
            continue
        stages.append(
            {
                key: raw_stage[key]
                for key in raw_stage
                if key not in {"duration_ms", "started_at", "completed_at"}
            }
        )
    artifacts = manifest.get("artifacts", [])
    artifact_paths: list[str] = []
    if isinstance(artifacts, Iterable) and not isinstance(artifacts, (str, bytes, Mapping)):
        for artifact in artifacts:
            if isinstance(artifact, Mapping) and isinstance(artifact.get("path"), str):
                artifact_paths.append(artifact["path"])
    return {
        "schema_version": manifest.get("schema_version"),
        "manifest_kind": manifest.get("manifest_kind"),
        "run_id": manifest.get("run_id"),
        "execution_mode": manifest.get("execution_mode"),
        "source": manifest.get("source"),
        "video": manifest.get("video"),
        "pipeline": manifest.get("pipeline"),
        "model": manifest.get("model"),
        "accounting": {
            key: value
            for key, value in (manifest.get("accounting") or {}).items()
            if key not in {"duration_ms", "started_at", "completed_at"}
        },
        "stages": stages,
        "artifact_paths": sorted(artifact_paths),
    }


def execution_manifest_semantic_sha256(manifest: Mapping[str, Any]) -> str:
    """Compute the stable semantic digest for a parsed execution manifest."""

    return semantic_sha256(_semantic_projection(manifest))


def build_execution_manifest(
    output_root: Path,
    *,
    report: object,
    video: object | None = None,
    model: object | None = None,
    provider_requests: int = 0,
) -> dict[str, Any]:
    """Build a canonical manifest from a completed local run.

    ``report`` and the optional component objects are intentionally duck-typed so the helper
    remains useful in CLI composition tests.  A production local run supplies the typed
    Pydantic contracts and therefore receives the complete source/video/model lineage.
    """

    if provider_requests != 0:
        raise ExecutionEvidenceError("local execution evidence requires zero provider requests")
    artifacts = _artifact_entries(output_root)
    report_status = _json_value(_get(report, "status"))
    if isinstance(report_status, Mapping):
        report_status = report_status.get("value")
    stages = _stage_entries(report)
    pipeline_version = _get(report, "pipeline_version")
    if hasattr(pipeline_version, "value"):
        pipeline_version = pipeline_version.value
    manifest: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "manifest_kind": "ROBATA_LOCAL_EXECUTION",
        "run_id": _get(report, "run_id"),
        "execution_mode": EXECUTION_MODE,
        "source": {
            "mcap_id": _get(report, "source_mcap_id"),
            "recording_identity": _get(report, "source_recording_identity"),
            "content_sha256": _get(report, "source_content_sha256"),
        },
        "video": {
            "manifest_artifact_id": _get(report, "video_manifest_artifact_id")
            or _get(video, "manifest_artifact_id"),
            "manifest_sha256": _get(report, "video_manifest_sha256")
            or _get(video, "manifest_sha256"),
            "manifest_semantic_sha256": _get(report, "video_manifest_semantic_sha256"),
        },
        "pipeline": {
            "version": pipeline_version,
            "config_sha256": _get(report, "config_sha256"),
        },
        "model": {
            "provider": _get(model, "provider", "fake"),
            "name": _get(model, "model_name"),
            "version": _get(model, "model_version"),
        },
        "accounting": {
            "status": report_status,
            "started_at": _get(report, "started_at"),
            "completed_at": _get(report, "completed_at"),
            "duration_ms": _get(report, "duration_ms", 0),
            "window_count": _get(report, "window_count", 0),
            "package_count": _get(report, "package_count", 0),
            "inference_attempt_count": _get(report, "inference_attempt_count", 0),
            "inference_success_count": _get(report, "inference_success_count", 0),
            "inference_failure_count": _get(report, "inference_failure_count", 0),
            "inference_invalid_output_count": _get(report, "inference_invalid_output_count", 0),
            "candidate_count": _get(report, "candidate_count", 0),
            "event_count": _get(report, "event_count", 0),
            "fake_inference_attempt_count": _get(report, "fake_inference_attempt_count", 0),
            "provider_request_count": provider_requests,
        },
        "stages": stages,
        "artifacts": artifacts,
    }
    manifest["semantic_sha256"] = execution_manifest_semantic_sha256(manifest)
    return manifest


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise ExecutionEvidenceError(f"cannot write execution evidence {path}: {error}") from error


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _audit_lines(manifest: Mapping[str, Any]) -> bytes:
    run_id = manifest.get("run_id")
    accounting = manifest.get("accounting") or {}
    events: list[dict[str, Any]] = [
        {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "sequence": 0,
            "event": "run_started",
            "run_id": run_id,
            "execution_mode": manifest.get("execution_mode"),
            "provider_requests": 0,
        }
    ]
    for sequence, raw_stage in enumerate(manifest.get("stages", []), start=1):
        if isinstance(raw_stage, Mapping):
            events.append(
                {
                    "schema_version": EXECUTION_SCHEMA_VERSION,
                    "sequence": sequence,
                    "event": "stage_completed",
                    "run_id": run_id,
                    "stage": raw_stage.get("stage"),
                    "status": raw_stage.get("status"),
                    "planned": raw_stage.get("planned", 0),
                    "succeeded": raw_stage.get("succeeded", 0),
                    "failed": raw_stage.get("failed", 0),
                    "pending": raw_stage.get("pending", 0),
                    "skipped": raw_stage.get("skipped", 0),
                }
            )
    events.append(
        {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "sequence": len(events),
            "event": "run_completed",
            "run_id": run_id,
            "status": accounting.get("status"),
            "event_count": accounting.get("event_count", 0),
            "artifact_count": len(manifest.get("artifacts", [])),
            "provider_requests": 0,
            "execution_manifest_semantic_sha256": manifest.get("semantic_sha256"),
        }
    )
    return b"".join(canonical_json_bytes(event) + b"\n" for event in events)


def write_execution_evidence(
    output_root: Path,
    *,
    report: object,
    video: object | None = None,
    model: object | None = None,
    provider_requests: int = 0,
) -> PublishedExecutionEvidence:
    """Write manifest and append-only NDJSON audit evidence into a staged output root."""

    manifest = build_execution_manifest(
        output_root,
        report=report,
        video=video,
        model=model,
        provider_requests=provider_requests,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    audit_bytes = _audit_lines(manifest)
    _write_exclusive(output_root / EXECUTION_MANIFEST_FILENAME, manifest_bytes)
    _write_exclusive(output_root / EXECUTION_AUDIT_FILENAME, audit_bytes)
    _sync_directory(output_root)
    return PublishedExecutionEvidence(
        manifest_sha256=exact_bytes_sha256(manifest_bytes),
        manifest_semantic_sha256=str(manifest["semantic_sha256"]),
        audit_sha256=exact_bytes_sha256(audit_bytes),
        artifact_count=len(manifest["artifacts"]),
    )


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ExecutionEvidenceError("manifest artifact path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ExecutionEvidenceError(f"manifest artifact path is not safely relative: {value!r}")
    return value


def verify_execution_evidence(output_root: Path) -> dict[str, Any]:
    """Verify manifest hashes, artifact inventory, and canonical audit records."""

    manifest_path = output_root / EXECUTION_MANIFEST_FILENAME
    audit_path = output_root / EXECUTION_AUDIT_FILENAME
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        audit_bytes = audit_path.read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionEvidenceError(f"cannot read execution evidence: {error}") from error
    if not isinstance(manifest, dict):
        raise ExecutionEvidenceError("execution manifest must be a JSON object")
    if manifest.get("schema_version") != EXECUTION_SCHEMA_VERSION:
        raise ExecutionEvidenceError("unsupported execution manifest schema version")
    expected_semantic = execution_manifest_semantic_sha256(manifest)
    if manifest.get("semantic_sha256") != expected_semantic:
        raise ExecutionEvidenceError("execution manifest semantic hash does not match contents")

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ExecutionEvidenceError("execution manifest artifacts must be an array")
    listed: dict[str, tuple[int, str]] = {}
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            raise ExecutionEvidenceError("execution artifact entries must be objects")
        relative = _safe_relative_path(raw_artifact.get("path"))
        size = raw_artifact.get("size_bytes")
        digest = raw_artifact.get("sha256")
        if not isinstance(size, int) or size < 0 or not isinstance(digest, str):
            raise ExecutionEvidenceError(f"invalid artifact entry for {relative!r}")
        if relative in listed:
            raise ExecutionEvidenceError(f"duplicate artifact path: {relative!r}")
        listed[relative] = (size, digest)
        path = output_root / Path(relative)
        if not path.is_file() or path.is_symlink():
            raise ExecutionEvidenceError(
                f"manifest artifact is missing or not regular: {relative!r}"
            )
        data = path.read_bytes()
        if len(data) != size or exact_bytes_sha256(data) != digest:
            raise ExecutionEvidenceError(f"artifact hash mismatch: {relative!r}")

    actual = {entry["path"]: entry for entry in _artifact_entries(output_root)}
    if set(actual) != set(listed):
        raise ExecutionEvidenceError("manifest artifact inventory does not match output files")

    lines = audit_bytes.splitlines()
    if not lines:
        raise ExecutionEvidenceError("execution audit is empty")
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExecutionEvidenceError(f"audit line {index} is not JSON") from error
        if canonical_json_bytes(event) != line:
            raise ExecutionEvidenceError(f"audit line {index} is not canonical JSON")
        if not isinstance(event, dict) or event.get("sequence") != index:
            raise ExecutionEvidenceError(f"audit line {index} has an invalid sequence")
    return manifest


__all__ = [
    "EXECUTION_AUDIT_FILENAME",
    "EXECUTION_MANIFEST_FILENAME",
    "EXECUTION_MODE",
    "EXECUTION_SCHEMA_VERSION",
    "ExecutionEvidenceError",
    "PublishedExecutionEvidence",
    "build_execution_manifest",
    "execution_manifest_semantic_sha256",
    "verify_execution_evidence",
    "write_execution_evidence",
]
