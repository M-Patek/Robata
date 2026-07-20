"""Reject published schema mutation except exact repair to an unchanged catalog pin."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH: Final = "schemas/schema-catalog.json"


class SchemaImmutabilityError(RuntimeError):
    """Raised when a Git tree removes or mutates a published schema fact."""


@dataclass(frozen=True, slots=True)
class SchemaImmutabilityResult:
    baseline_commit: str
    current_commit: str
    preserved_schema_versions: int
    added_schema_versions: int
    preserved_upcasters: int
    added_upcasters: int
    reconciled_schema_versions: tuple[str, ...]


def _run_git(repo: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise SchemaImmutabilityError(f"cannot execute git: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SchemaImmutabilityError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _repository_root(repo: Path) -> Path:
    raw = _run_git(repo, "rev-parse", "--show-toplevel")
    return Path(raw.decode("utf-8").strip()).resolve()


def _resolve_commit(repo: Path, ref: str) -> str:
    raw = _run_git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{ref}^{{commit}}",
    )
    return raw.decode("ascii").strip()


def _read_tree_blob(repo: Path, commit: str, path: str) -> bytes:
    return _run_git(repo, "cat-file", "blob", f"{commit}:{path}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaImmutabilityError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise SchemaImmutabilityError(f"non-JSON numeric constant: {value}")


def _load_catalog(repo: Path, commit: str) -> dict[str, Any]:
    raw = _read_tree_blob(repo, commit, CATALOG_PATH)
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except SchemaImmutabilityError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaImmutabilityError(f"invalid {CATALOG_PATH} at {commit}: {exc}") from exc
    if not isinstance(document, dict):
        raise SchemaImmutabilityError(f"{CATALOG_PATH} at {commit} must be a JSON object")
    return document


def _catalog_array(
    catalog: dict[str, Any],
    field: str,
    *,
    commit: str,
) -> list[Any]:
    value = catalog.get(field)
    if not isinstance(value, list):
        raise SchemaImmutabilityError(
            f"{CATALOG_PATH} field {field!r} at {commit} must be an array"
        )
    return value


def _schema_entries(
    catalog: dict[str, Any],
    *,
    commit: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for position, entry in enumerate(_catalog_array(catalog, "schemas", commit=commit)):
        if not isinstance(entry, dict):
            raise SchemaImmutabilityError(f"schema entry {position} at {commit} must be an object")
        ref = entry.get("ref")
        if not isinstance(ref, dict):
            raise SchemaImmutabilityError(f"schema entry {position} at {commit} has no object ref")
        schema_id = ref.get("schema_id")
        version = ref.get("version")
        artifact_path = entry.get("artifact_path")
        if not isinstance(schema_id, str) or not isinstance(version, str):
            raise SchemaImmutabilityError(
                f"schema entry {position} at {commit} has an invalid logical key"
            )
        if not isinstance(artifact_path, str) or not artifact_path:
            raise SchemaImmutabilityError(
                f"schema {schema_id}@{version} at {commit} has an invalid artifact_path"
            )
        key = (schema_id, version)
        if key in indexed:
            raise SchemaImmutabilityError(
                f"duplicate schema version at {commit}: {schema_id}@{version}"
            )
        indexed[key] = entry
    return indexed


def _upcaster_entries(
    catalog: dict[str, Any],
    *,
    commit: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(_catalog_array(catalog, "upcasters", commit=commit)):
        if not isinstance(entry, dict):
            raise SchemaImmutabilityError(
                f"upcaster entry {position} at {commit} must be an object"
            )
        upcaster_id = entry.get("upcaster_id")
        if not isinstance(upcaster_id, str) or not upcaster_id:
            raise SchemaImmutabilityError(
                f"upcaster entry {position} at {commit} has an invalid upcaster_id"
            )
        if upcaster_id in indexed:
            raise SchemaImmutabilityError(f"duplicate upcaster_id at {commit}: {upcaster_id}")
        indexed[upcaster_id] = entry
    return indexed


def _safe_upcaster_artifact_path(relative_path: object, *, label: str) -> str:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise SchemaImmutabilityError(f"{label} has an invalid artifact path")
    parts = relative_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SchemaImmutabilityError(f"{label} has an unsafe artifact path: {relative_path}")
    if any(re.fullmatch(r"[A-Za-z0-9._-]+", part) is None for part in parts):
        raise SchemaImmutabilityError(f"{label} has an unsafe artifact path: {relative_path}")
    return f"schemas/{relative_path}"


def _upcaster_artifact_pins(
    upcaster_id: str,
    entry: dict[str, Any],
    *,
    commit: str,
) -> tuple[tuple[str, str, str], ...]:
    pins: list[tuple[str, str, str]] = []

    def add(kind: str, path_field: object, digest_field: object) -> None:
        label = f"upcaster {upcaster_id} {kind} at {commit}"
        tree_path = _safe_upcaster_artifact_path(path_field, label=label)
        if not isinstance(digest_field, str) or re.fullmatch(r"[0-9a-f]{64}", digest_field) is None:
            raise SchemaImmutabilityError(f"{label} has an invalid SHA-256 pin")
        pins.append((kind, tree_path, digest_field))

    add("code", entry.get("code_artifact_path"), entry.get("code_sha256"))
    add("runtime", entry.get("runtime_artifact_path"), entry.get("runtime_sha256"))
    golden_vectors = entry.get("golden_vectors")
    if not isinstance(golden_vectors, list) or not golden_vectors:
        raise SchemaImmutabilityError(
            f"upcaster {upcaster_id} at {commit} must have golden vector pairs"
        )
    for position, vector in enumerate(golden_vectors):
        if not isinstance(vector, dict):
            raise SchemaImmutabilityError(
                f"upcaster {upcaster_id} golden vector {position} at {commit} must be an object"
            )
        add(
            f"golden[{position}].input",
            vector.get("input_artifact_path"),
            vector.get("input_sha256"),
        )
        add(
            f"golden[{position}].output",
            vector.get("output_artifact_path"),
            vector.get("output_sha256"),
        )
    return tuple(pins)


def _load_upcaster_artifacts(
    repo: Path,
    commit: str,
    entries: dict[str, dict[str, Any]],
    violations: list[str],
) -> dict[tuple[str, str], bytes]:
    loaded: dict[tuple[str, str], bytes] = {}
    for upcaster_id, entry in sorted(entries.items()):
        for kind, tree_path, expected_sha256 in _upcaster_artifact_pins(
            upcaster_id,
            entry,
            commit=commit,
        ):
            try:
                raw = _read_tree_blob(repo, commit, tree_path)
            except SchemaImmutabilityError as exc:
                violations.append(
                    f"upcaster artifact is missing: {upcaster_id} {kind} at {commit}: {exc}"
                )
                continue
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            if actual_sha256 != expected_sha256:
                violations.append(
                    "upcaster artifact SHA-256 mismatch: "
                    f"{upcaster_id} {kind} at {commit} "
                    f"catalog={expected_sha256} actual={actual_sha256}"
                )
            loaded[(upcaster_id, kind)] = raw
    return loaded


def check_schema_immutability(
    repo: Path,
    *,
    baseline_ref: str,
    current_ref: str = "HEAD",
) -> SchemaImmutabilityResult:
    """Compare immutable catalog facts and exact artifact bytes across two Git trees."""

    root = _repository_root(repo)
    baseline_commit = _resolve_commit(root, baseline_ref)
    current_commit = _resolve_commit(root, current_ref)
    baseline_catalog = _load_catalog(root, baseline_commit)
    current_catalog = _load_catalog(root, current_commit)
    baseline_schemas = _schema_entries(baseline_catalog, commit=baseline_commit)
    current_schemas = _schema_entries(current_catalog, commit=current_commit)
    baseline_upcasters = _upcaster_entries(baseline_catalog, commit=baseline_commit)
    current_upcasters = _upcaster_entries(current_catalog, commit=current_commit)
    violations: list[str] = []
    reconciled_schema_versions: list[str] = []
    baseline_upcaster_artifacts = _load_upcaster_artifacts(
        root,
        baseline_commit,
        baseline_upcasters,
        violations,
    )
    current_upcaster_artifacts = _load_upcaster_artifacts(
        root,
        current_commit,
        current_upcasters,
        violations,
    )

    for key, baseline_entry in sorted(baseline_schemas.items()):
        schema_id, version = key
        current_entry = current_schemas.get(key)
        label = f"{schema_id}@{version}"
        if current_entry is None:
            violations.append(f"published schema version was deleted: {label}")
            continue
        if current_entry != baseline_entry:
            violations.append(f"published schema entry changed: {label}")
            continue
        artifact_path = baseline_entry["artifact_path"]
        baseline_bytes = _read_tree_blob(root, baseline_commit, f"schemas/{artifact_path}")
        current_bytes = _read_tree_blob(root, current_commit, f"schemas/{artifact_path}")
        if current_bytes != baseline_bytes:
            pinned_digest = baseline_entry["ref"].get("sha256")
            if not isinstance(pinned_digest, str):
                violations.append(f"published schema has no exact-byte pin: {label}")
                continue
            baseline_digest = hashlib.sha256(baseline_bytes).hexdigest()
            current_digest = hashlib.sha256(current_bytes).hexdigest()
            if baseline_digest != pinned_digest and current_digest == pinned_digest:
                reconciled_schema_versions.append(label)
            else:
                violations.append(f"published schema exact bytes changed: {label}")

    for upcaster_id, baseline_entry in sorted(baseline_upcasters.items()):
        current_entry = current_upcasters.get(upcaster_id)
        if current_entry is None:
            violations.append(f"published upcaster was deleted: {upcaster_id}")
        elif current_entry != baseline_entry:
            violations.append(f"published upcaster changed: {upcaster_id}")
        else:
            for kind, _path, _digest in _upcaster_artifact_pins(
                upcaster_id,
                baseline_entry,
                commit=baseline_commit,
            ):
                key = (upcaster_id, kind)
                baseline_bytes = baseline_upcaster_artifacts.get(key)
                current_bytes = current_upcaster_artifacts.get(key)
                if (
                    baseline_bytes is not None
                    and current_bytes is not None
                    and current_bytes != baseline_bytes
                ):
                    violations.append(
                        f"published upcaster exact bytes changed: {upcaster_id} {kind}"
                    )

    if violations:
        raise SchemaImmutabilityError("; ".join(violations))

    return SchemaImmutabilityResult(
        baseline_commit=baseline_commit,
        current_commit=current_commit,
        preserved_schema_versions=len(baseline_schemas),
        added_schema_versions=len(current_schemas.keys() - baseline_schemas.keys()),
        preserved_upcasters=len(baseline_upcasters),
        added_upcasters=len(current_upcasters.keys() - baseline_upcasters.keys()),
        reconciled_schema_versions=tuple(reconciled_schema_versions),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--baseline-ref",
        required=True,
        help="protected release commit or tag whose published facts are immutable",
    )
    parser.add_argument(
        "--current-ref",
        default="HEAD",
        help="candidate commit to inspect (default: HEAD)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = check_schema_immutability(
            args.repo,
            baseline_ref=args.baseline_ref,
            current_ref=args.current_ref,
        )
    except SchemaImmutabilityError as exc:
        print(f"schema immutability check failed: {exc}", file=sys.stderr)
        return 1

    print(
        "schema immutability check passed: "
        f"baseline={result.baseline_commit} current={result.current_commit} "
        f"preserved_schemas={result.preserved_schema_versions} "
        f"added_schemas={result.added_schema_versions} "
        f"preserved_upcasters={result.preserved_upcasters} "
        f"added_upcasters={result.added_upcasters} "
        f"reconciled_schemas={len(result.reconciled_schema_versions)} "
        "reconciled_labels="
        f"{json.dumps(result.reconciled_schema_versions, separators=(',', ':'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SchemaImmutabilityError",
    "SchemaImmutabilityResult",
    "check_schema_immutability",
    "main",
]
