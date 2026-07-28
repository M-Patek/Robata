"""P0 scope fingerprints and evidence registers for local profile artifacts.

This module is deliberately an internal qualification boundary.  It wraps the existing
canonical profile without changing its published shape or any run, artifact, queue, or
identity contract.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from robata.contracts.hashing import semantic_sha256
from robata.contracts.measurement_truth import (
    IDENTITY_FORMULA_VERSION,
    EvidenceClass,
    EvidenceRegister,
    MeasurementAxes,
    MeasurementEnvironment,
    MeasurementExecutionMode,
    MeasurementStatus,
    MeasurementWorkload,
    ScopeDigestInputs,
    ScopeEvidenceRegister,
    ScopeFingerprint,
)
from robata.runtime.canonical_profile import (
    CanonicalProfileManifest,
    CanonicalProfileReport,
    StateTreeSnapshot,
    canonical_profile_workload_fingerprint,
    unique_runtime_counter_value,
)
from robata.runtime.capacity import MeasuredCapacityStatus

_CODE_ROOTS: Final[tuple[str, ...]] = (
    "src/",
    "scripts/",
    "schemas/",
    "config/",
    "conformance/",
)
_ROOT_FILES: Final[frozenset[str]] = frozenset({"pyproject.toml", "uv.lock"})
_IGNORED_CODE_DIRS: Final[frozenset[str]] = frozenset(
    {
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
_IGNORED_CODE_SUFFIXES: Final[frozenset[str]] = frozenset({".pyc", ".pyo"})


def _git_paths(repository_root: Path) -> tuple[str, ...]:
    """Return tracked and non-ignored paths without embedding local filesystem paths."""

    try:
        git_root = (
            subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
            )
            .stdout.decode("utf-8")
            .strip()
        )
        if Path(git_root).resolve() != repository_root.resolve():
            return ()
        tracked = subprocess.run(
            ["git", "-C", str(repository_root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8")
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *_CODE_ROOTS,
                *_ROOT_FILES,
            ],
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8")
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        return ()
    # The pathspec keeps local recordings and historical snapshots outside the scan while
    # still including new implementation files under the bounded source roots.
    paths = {item for item in (*tracked.split("\0"), *untracked.split("\0")) if item}
    return tuple(sorted(path for path in paths if _is_code_path(path)))


def _git_worktree_root(repository_root: Path) -> Path:
    """Normalize only the worktree root and its declared source roots."""

    resolved_root = repository_root.resolve()
    try:
        output = (
            subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
            )
            .stdout.decode("utf-8")
            .strip()
        )
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        return resolved_root
    worktree_root = Path(output).resolve()
    try:
        relative_to_worktree = resolved_root.relative_to(worktree_root)
    except ValueError:
        return resolved_root
    code_roots = {root.rstrip("/") for root in _CODE_ROOTS}
    if relative_to_worktree != Path(".") and relative_to_worktree.as_posix() not in code_roots:
        return resolved_root
    return worktree_root


def _is_code_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    return normalized in _ROOT_FILES or normalized.startswith(_CODE_ROOTS)


def _fallback_paths(repository_root: Path) -> tuple[str, ...]:
    candidates: list[str] = []
    for root in (*_CODE_ROOTS,):
        directory = repository_root / root.rstrip("/")
        if not directory.is_dir():
            continue
        candidates.extend(
            path.relative_to(repository_root).as_posix()
            for path in directory.rglob("*")
            if (
                path.is_file()
                and not path.is_symlink()
                and not any(part in _IGNORED_CODE_DIRS for part in path.parts)
                and path.suffix.lower() not in _IGNORED_CODE_SUFFIXES
            )
        )
    candidates.extend(name for name in _ROOT_FILES if (repository_root / name).is_file())
    return tuple(sorted(set(candidates)))


def repository_code_digest(repository_root: Path) -> str:
    """Hash current code/config bytes, including tracked dirty changes and new files."""

    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be pathlib.Path")
    root = _git_worktree_root(repository_root.resolve())
    if not root.is_dir():
        raise ValueError("repository_root must be an existing directory")
    paths = _git_paths(root) or _fallback_paths(root)
    facts: list[dict[str, object]] = []
    for relative in paths:
        path = root / Path(relative)
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        facts.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "byte_count": len(data),
            }
        )
    return semantic_sha256({"domain": "robata.code-tree", "version": "v1", "files": facts})


def _identity_projection_digest() -> str:
    return semantic_sha256(
        {
            "identity_formula_version": IDENTITY_FORMULA_VERSION,
            "recording_identity": "namespace+source_content_sha256",
            "logical_key": "key_namespace+semantic_sha256",
            "idempotency": "transport-independent logical identity",
            "fence": "durable owner fence",
        }
    )


def _policy_digest(manifest: CanonicalProfileManifest) -> str:
    return semantic_sha256(manifest.policies.model_dump(mode="json"))


def build_profile_scope_fingerprint(
    manifest: CanonicalProfileManifest,
    *,
    repository_root: Path | None = None,
) -> ScopeFingerprint:
    """Build a scope digest over code, catalog, workload, policies, and identity."""

    if not isinstance(manifest, CanonicalProfileManifest):
        raise TypeError("manifest must be a CanonicalProfileManifest")
    code_root = (
        repository_root if repository_root is not None else Path(__file__).resolve().parents[3]
    )
    code_digest = repository_code_digest(code_root)
    workload_digest = canonical_profile_workload_fingerprint(manifest)
    return ScopeFingerprint.create(
        inputs=ScopeDigestInputs(
            code_revision=manifest.git.head_commit,
            code_digest=code_digest,
            schema_catalog_digest=manifest.schema_catalog.sha256,
            workload_digest=workload_digest,
            policy_digest=_policy_digest(manifest),
            identity_formula_version=IDENTITY_FORMULA_VERSION,
            identity_projection_digest=_identity_projection_digest(),
            seam_versions=tuple(sorted(manifest.policies.inference_policy_versions)),
        )
    )


def _counter_value(report: CanonicalProfileReport, names: Iterable[str]) -> int | None:
    for name in names:
        value = unique_runtime_counter_value(report.observer, name)
        if value is not None:
            return value
    return None


def _resource_value(snapshot: StateTreeSnapshot, _name: str) -> int | None:
    del snapshot
    # StateTreeSnapshot intentionally has no accelerator/NVMe attribution.  Do not
    # relabel process-wide I/O as device-specific work.
    return None


def _terminal_count(report: CanonicalProfileReport) -> int | None:
    """Return an observed terminal-row count, never infer one from receipt presence."""

    reconciliation = report.reconciliation
    if reconciliation is None:
        return None
    for fact in reconciliation.ledger.facts:
        if fact.name == "inference.calls_vs_terminals":
            return fact.observed
    return None


def build_profile_evidence_register(
    report: CanonicalProfileReport,
    *,
    repository_root: Path | None = None,
    evidence_class: EvidenceClass = EvidenceClass.LOCAL_CONFORMANCE,
    provider: str | None = None,
    hardware: str | None = None,
    observed_at: str | None = None,
) -> ScopeEvidenceRegister:
    """Wrap one canonical profile in a dated, replayable evidence register."""

    if not isinstance(report, CanonicalProfileReport):
        raise TypeError("report must be a CanonicalProfileReport")
    if not isinstance(evidence_class, EvidenceClass):
        raise TypeError("evidence_class must be an EvidenceClass")
    scope = build_profile_scope_fingerprint(report.manifest, repository_root=repository_root)
    capacity = report.capacity
    recording_duration_ns = (
        capacity.recording_duration_ns if capacity is not None else report.recording_duration_ns
    )
    recording_count = (
        capacity.recording_count
        if capacity is not None
        else report.measurements.recording_count
        if report.measurements is not None
        else 1
    )
    recording_hours = None if capacity is None else capacity.recording_hours
    camera_hours = None if capacity is None else capacity.camera_hours
    axes = MeasurementAxes(
        recording_hours=recording_hours,
        camera_hours=camera_hours,
        decoded_frames=_counter_value(
            report,
            ("source.decoded_frames", "source.frame_count", "source.indexed_frames"),
        ),
        selected_images=None if capacity is None else capacity.unique_images,
        provider_images=None if capacity is None else capacity.provider_images,
        provider_calls=None if capacity is None else capacity.logical_calls,
        http_requests=None if capacity is None else capacity.http_requests,
        input_tokens=None if capacity is None else capacity.input_tokens,
        output_tokens=None if capacity is None else capacity.output_tokens,
        process_cpu_ns=report.observer.process_cpu_ns,
        gpu_time_ns=None,
        nvme_read_bytes=_resource_value(report.state_after, "read"),
        nvme_write_bytes=_resource_value(report.state_after, "write"),
        queue_backlog=(
            report.work_queue_after.nonterminal_backlog_count
            if report.work_queue_after.status.value == "AVAILABLE"
            else None
        ),
        terminal_count=_terminal_count(report),
    )
    workload = MeasurementWorkload(
        workload_fingerprint=canonical_profile_workload_fingerprint(report.manifest),
        recording_count=recording_count,
        camera_count=report.manifest.camera_count,
        recording_duration_ns=recording_duration_ns,
        frame_count=axes.decoded_frames,
        source_bytes=report.manifest.source.byte_count,
    )
    provider_mode = capacity.provider_mode.value if capacity is not None else "UNKNOWN"
    environment = MeasurementEnvironment(
        provider=provider if provider is not None else provider_mode,
        provider_mode=provider_mode,
        hardware=(
            hardware
            if hardware is not None
            else f"{report.manifest.runtime.platform}/{report.manifest.runtime.machine}"
        ),
        accelerator=None,
    )
    execution_mode = MeasurementExecutionMode(report.execution_mode)
    if evidence_class is EvidenceClass.PRODUCTION_QUALIFIED:
        raise ValueError("PRODUCTION_QUALIFIED requires the external qualification gate")
    measurement_status = (
        MeasurementStatus.MEASURED
        if (
            evidence_class is not EvidenceClass.LOCAL_CONFORMANCE
            and capacity is not None
            and capacity.measurement_status is MeasuredCapacityStatus.AVAILABLE
        )
        else MeasurementStatus.NOT_MEASURED
    )
    return ScopeEvidenceRegister.create(
        scope=scope,
        evidence_class=evidence_class,
        execution_mode=execution_mode,
        workload=workload,
        environment=environment,
        axes=axes,
        observed_at=observed_at,
        measurement_status=measurement_status,
        profile_manifest_digest=report.manifest.manifest_sha256,
        profile_report_digest=semantic_sha256(report.model_dump(mode="json")),
    )


def load_profile_evidence_register(
    profile_path: Path,
    *,
    repository_root: Path | None = None,
    evidence_class: EvidenceClass = EvidenceClass.LOCAL_CONFORMANCE,
    provider: str | None = None,
    hardware: str | None = None,
    observed_at: str | None = None,
) -> ScopeEvidenceRegister:
    """Load a v3 profile JSON and produce its scope/evidence register."""

    if not isinstance(profile_path, Path):
        raise TypeError("profile_path must be pathlib.Path")
    try:
        profile_bytes = profile_path.read_bytes()
        report = CanonicalProfileReport.model_validate_json(profile_bytes, strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"cannot read profile JSON: {error}") from error
    return build_profile_evidence_register(
        report,
        repository_root=repository_root,
        evidence_class=evidence_class,
        provider=provider,
        hardware=hardware,
        observed_at=observed_at,
    )


__all__ = [
    "EvidenceClass",
    "EvidenceRegister",
    "ScopeEvidenceRegister",
    "ScopeFingerprint",
    "build_profile_evidence_register",
    "build_profile_scope_fingerprint",
    "load_profile_evidence_register",
    "repository_code_digest",
]
