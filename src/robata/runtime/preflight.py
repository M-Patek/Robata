"""Offline readiness checks for the local fake-model mainline.

This module intentionally performs no source decoding, model invocation, provider import,
network access, or output creation.  It only validates the local environment and paths that a
subsequent ``run_local_mainline.py`` invocation will use.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from robata.ingestion import TopicMappingProfile

EXPECTED_EXECUTION_SPEC_SHA256 = "434902fed026726e9e4924042dd1f3f2d2ec26172011efe188aac3f7986e3c0a"
REQUIRED_IMPORTS = ("av", "mcap", "mcap_protobuf", "pydantic", "jsonschema")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def run_preflight(
    source: Path,
    output: Path,
    *,
    mapping_config: Path,
    registry_root: Path | None = None,
    allow_unapproved: bool = False,
    spec_path: Path | None = None,
    verify_spec_hash: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serializable preflight result without side effects."""

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    source = _absolute(source)
    output = _absolute(output)
    mapping_config = _absolute(mapping_config)
    effective_registry = (
        _absolute(registry_root)
        if registry_root is not None
        else output.parent / ".robata-artifacts"
    )

    version = sys.version_info
    version_ok = (
        (version.major, version.minor) >= (3, 12) and version.major == 3 and version.minor < 14
    )
    checks.append(
        _check(
            "python_version",
            version_ok,
            f"Python {version.major}.{version.minor}.{version.micro}; required >=3.12,<3.14",
        )
    )

    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as error:  # pragma: no cover - platform-specific import failures
            checks.append(
                _check(f"import:{module_name}", False, f"unavailable: {type(error).__name__}")
            )
        else:
            checks.append(_check(f"import:{module_name}", True, "available"))

    try:
        profile = TopicMappingProfile.load(mapping_config)
    except Exception as error:
        checks.append(_check("mapping_config", False, f"invalid: {type(error).__name__}"))
        profile = None
    else:
        assert profile is not None
        checks.append(_check("mapping_config", True, "parseable six-camera topic profile"))
        if profile.approved:
            checks.append(_check("mapping_authorization", True, "profile is approved"))
        elif allow_unapproved:
            checks.append(
                _check("mapping_authorization", True, "unapproved profile explicitly allowed")
            )
            warnings.append("mapping profile is unapproved; local development override is active")
        else:
            checks.append(
                _check(
                    "mapping_authorization", False, "profile is unapproved; pass --allow-unapproved"
                )
            )

    source_ok = source.is_file() and not source.is_symlink() and os.access(source, os.R_OK)
    checks.append(
        _check(
            "source",
            source_ok,
            "regular readable file" if source_ok else "missing, symlink, or unreadable",
        )
    )

    output_ok = not output.exists() and not output.is_symlink()
    checks.append(
        _check(
            "output",
            output_ok,
            "absent and non-symlink" if output_ok else "must be absent and non-symlink",
        )
    )
    parent_ok = output.parent.is_dir() and os.access(output.parent, os.W_OK)
    checks.append(
        _check(
            "output_parent",
            parent_ok,
            "writable parent directory" if parent_ok else "parent is missing or not writable",
        )
    )

    registry_inside_output = False
    try:
        registry_inside_output = effective_registry.resolve(strict=False).is_relative_to(
            output.resolve(strict=False)
        )
    except OSError:
        registry_inside_output = True
    if registry_inside_output or effective_registry.is_symlink():
        registry_ok = False
        registry_detail = "must be outside output and not a symlink"
    elif effective_registry.exists():
        registry_ok = effective_registry.is_dir() and os.access(effective_registry, os.W_OK)
        registry_detail = (
            "existing writable directory"
            if registry_ok
            else "existing registry root must be a writable directory"
        )
    else:
        registry_ok = effective_registry.parent.is_dir() and os.access(
            effective_registry.parent, os.W_OK
        )
        registry_detail = (
            "absent; writable parent directory"
            if registry_ok
            else "registry parent is missing or not writable"
        )
    checks.append(_check("registry_root", registry_ok, registry_detail))

    if verify_spec_hash:
        checked_spec = (
            _absolute(spec_path)
            if spec_path is not None
            else Path(__file__).resolve().parents[3]
            / "large_scale_6camera_video_agent_execution_spec.md"
        )
        try:
            digest = hashlib.sha256(checked_spec.read_bytes()).hexdigest()
        except OSError:
            checks.append(
                _check("execution_spec_hash", False, "specification file is missing or unreadable")
            )
        else:
            spec_ok = digest == EXPECTED_EXECUTION_SPEC_SHA256
            checks.append(
                _check(
                    "execution_spec_hash",
                    spec_ok,
                    "pinned specification hash matches"
                    if spec_ok
                    else "pinned specification hash mismatch",
                )
            )
    else:
        warnings.append("execution specification hash check skipped")

    mapping_digest = profile.semantic_digest if profile is not None else None
    result = {
        "ok": all(bool(check["ok"]) for check in checks),
        "checks": checks,
        "warnings": warnings,
        "mapping_profile_digest": mapping_digest,
        "registry_root": str(effective_registry),
        "provider_requests": 0,
    }
    return result


__all__ = [
    "EXPECTED_EXECUTION_SPEC_SHA256",
    "REQUIRED_IMPORTS",
    "run_preflight",
]
