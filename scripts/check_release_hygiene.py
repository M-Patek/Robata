"""Validate tracked source files and emit a deterministic release manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

MANIFEST_FORMAT: Final = "robata-source-release-manifest-v1"

_FORBIDDEN_DIRECTORY_NAMES: Final = frozenset(
    {".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)
_FORBIDDEN_FILENAMES: Final = frozenset({".schema-publication-transaction.json"})
_ROOT_ARCHIVE_SUFFIXES: Final = (
    ".7z",
    ".tar",
    ".tar.bz2",
    ".tar.gz",
    ".tar.xz",
    ".tgz",
    ".zip",
)


class ReleaseHygieneError(RuntimeError):
    """Raised when repository state cannot produce a source release."""


@dataclass(frozen=True, slots=True)
class TrackedFile:
    """One exact Git blob included in the source release."""

    path: str
    mode: str
    object_id: str


@dataclass(frozen=True, slots=True)
class ReleaseFile:
    """One content-addressed file in the release manifest."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    """Deterministic description of one source tree."""

    commit_sha: str
    files: tuple[ReleaseFile, ...]
    format_version: str = MANIFEST_FORMAT

    def to_bytes(self) -> bytes:
        payload = {
            "commit_sha": self.commit_sha,
            "files": [
                {"path": item.path, "sha256": item.sha256, "size": item.size} for item in self.files
            ],
            "format_version": self.format_version,
        }
        return (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )


def _run_git(repo: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        input=input_bytes,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseHygieneError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _repository_root(repo: Path) -> Path:
    raw = _run_git(repo, "rev-parse", "--show-toplevel")
    return Path(raw.decode("utf-8").strip()).resolve()


def _require_clean_worktree(repo: Path) -> None:
    status = _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ReleaseHygieneError(
            "release requires a clean worktree (including the index and untracked files)"
        )


def _commit_sha(repo: Path) -> str:
    return _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()


def _resolve_expected_commit(repo: Path, expected: str) -> str:
    if not expected or expected.isspace():
        raise ReleaseHygieneError("expected commit must not be empty")
    return (
        _run_git(repo, "rev-parse", "--verify", "--end-of-options", f"{expected}^{{commit}}")
        .decode("ascii")
        .strip()
    )


def _tracked_files(repo: Path, commit_sha: str) -> tuple[TrackedFile, ...]:
    raw = _run_git(repo, "ls-tree", "-r", "-z", "--full-tree", commit_sha)
    tracked: list[TrackedFile] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, object_type, object_id = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ReleaseHygieneError("HEAD contains an unsupported tracked path") from exc
        if object_type != b"blob":
            raise ReleaseHygieneError(
                f"tracked entry is not a file blob: {path} ({object_type.decode('ascii')})"
            )
        mode = raw_mode.decode("ascii")
        if mode not in {"100644", "100755", "120000"}:
            raise ReleaseHygieneError(f"tracked blob has an unsupported mode: {path} ({mode})")
        tracked.append(TrackedFile(path=path, mode=mode, object_id=object_id.decode("ascii")))
    return tuple(sorted(tracked, key=lambda item: item.path))


def _ignored_tracked_paths(repo: Path, paths: tuple[str, ...]) -> tuple[str, ...]:
    if not paths:
        return ()
    request = b"".join(path.encode("utf-8") + b"\0" for path in paths)
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "check-ignore",
            "--no-index",
            "--stdin",
            "-z",
        ],
        check=False,
        input=request,
        capture_output=True,
    )
    if completed.returncode not in {0, 1}:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseHygieneError(f"git check-ignore failed: {detail}")
    try:
        ignored = tuple(item.decode("utf-8") for item in completed.stdout.split(b"\0") if item)
    except UnicodeDecodeError as exc:
        raise ReleaseHygieneError("git check-ignore returned an unsupported path") from exc
    unexpected = sorted(set(ignored).difference(paths))
    if unexpected:
        rendered = ", ".join(unexpected)
        raise ReleaseHygieneError(f"git check-ignore returned unexpected paths: {rendered}")
    return tuple(sorted(set(ignored)))


def forbidden_tracked_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Return generated or packaged paths that must not enter a source release."""

    forbidden: list[str] = []
    for raw_path in paths:
        path = PurePosixPath(raw_path)
        lowered_parts = tuple(part.casefold() for part in path.parts)
        if path.name.casefold() in _FORBIDDEN_FILENAMES:
            forbidden.append(raw_path)
            continue
        if any(part in _FORBIDDEN_DIRECTORY_NAMES for part in lowered_parts):
            forbidden.append(raw_path)
            continue
        if any(part.startswith(("pytest-temp-", "pytest-cache-files-")) for part in lowered_parts):
            forbidden.append(raw_path)
            continue
        if len(path.parts) == 1 and path.name.casefold().endswith(_ROOT_ARCHIVE_SUFFIXES):
            forbidden.append(raw_path)
    return tuple(sorted(forbidden))


def _forbidden_schema_symlinks(tracked: tuple[TrackedFile, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            item.path
            for item in tracked
            if item.mode == "120000"
            and PurePosixPath(item.path).parts
            and PurePosixPath(item.path).parts[0].casefold() == "schemas"
        )
    )


def _read_blobs(repo: Path, tracked: tuple[TrackedFile, ...]) -> tuple[ReleaseFile, ...]:
    if not tracked:
        return ()
    request = b"".join(f"{item.object_id}\n".encode("ascii") for item in tracked)
    response = _run_git(repo, "cat-file", "--batch", input_bytes=request)
    offset = 0
    files: list[ReleaseFile] = []
    for expected in tracked:
        header_end = response.find(b"\n", offset)
        if header_end < 0:
            raise ReleaseHygieneError("git cat-file returned a truncated header")
        header = response[offset:header_end].split(b" ")
        if len(header) != 3:
            raise ReleaseHygieneError("git cat-file returned an invalid header")
        object_id, object_type, raw_size = header
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise ReleaseHygieneError("git cat-file returned an invalid blob size") from exc
        if object_id.decode("ascii") != expected.object_id or object_type != b"blob":
            raise ReleaseHygieneError(f"git returned the wrong blob for {expected.path}")
        content_start = header_end + 1
        content_end = content_start + size
        if content_end >= len(response) or response[content_end : content_end + 1] != b"\n":
            raise ReleaseHygieneError(
                f"git cat-file returned truncated content for {expected.path}"
            )
        content = response[content_start:content_end]
        files.append(
            ReleaseFile(
                path=expected.path,
                sha256=hashlib.sha256(content).hexdigest(),
                size=size,
            )
        )
        offset = content_end + 1
    if offset != len(response):
        raise ReleaseHygieneError("git cat-file returned unexpected trailing data")
    return tuple(files)


def _open_archive(raw_archive: bytes) -> tarfile.TarFile:
    try:
        return tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:")
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseHygieneError("git archive returned an invalid tar stream") from exc


def _verify_archive(
    repo: Path,
    commit_sha: str,
    tracked: tuple[TrackedFile, ...],
    files: tuple[ReleaseFile, ...],
) -> bytes:
    expected_tracked = {item.path: item for item in tracked}
    expected_files = {item.path: item for item in files}
    if expected_tracked.keys() != expected_files.keys():
        raise ReleaseHygieneError("internal manifest entry set does not match tracked blobs")

    raw_archive = _run_git(repo, "archive", "--format=tar", commit_sha)
    seen: set[str] = set()
    with _open_archive(raw_archive) as archive:
        for member in archive:
            if member.isdir():
                continue
            path = member.name
            if path in seen:
                raise ReleaseHygieneError(f"git archive contains a duplicate entry: {path}")
            expected = expected_files.get(path)
            tracked_file = expected_tracked.get(path)
            if expected is None or tracked_file is None:
                raise ReleaseHygieneError(f"git archive contains an unexpected entry: {path}")
            seen.add(path)

            if tracked_file.mode == "120000":
                if not member.issym():
                    raise ReleaseHygieneError(
                        f"git archive entry type does not match tracked symlink: {path}"
                    )
                try:
                    content = member.linkname.encode(archive.encoding, errors=archive.errors)
                except UnicodeError as exc:
                    raise ReleaseHygieneError(
                        f"git archive returned an unsupported symlink target: {path}"
                    ) from exc
            else:
                if not member.isfile():
                    raise ReleaseHygieneError(
                        f"git archive entry type does not match tracked file: {path}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ReleaseHygieneError(f"git archive file cannot be read: {path}")
                content = extracted.read()

            if (
                len(content) != expected.size
                or hashlib.sha256(content).hexdigest() != expected.sha256
            ):
                raise ReleaseHygieneError(f"git archive content does not match manifest: {path}")

    missing = sorted(expected_files.keys() - seen)
    if missing:
        raise ReleaseHygieneError(f"git archive is missing manifest entries: {', '.join(missing)}")
    return raw_archive


def _require_head_unchanged(repo: Path, commit_sha: str) -> None:
    current_sha = _commit_sha(repo)
    if current_sha != commit_sha:
        raise ReleaseHygieneError(
            f"HEAD changed during release validation: {commit_sha} -> {current_sha}"
        )


def _write_verified_archive(repo: Path, destination: Path, raw_archive: bytes) -> None:
    output = destination.expanduser().resolve()
    if output.is_relative_to(repo):
        raise ReleaseHygieneError("archive output must be outside the repository worktree")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ReleaseHygieneError(f"archive output is not a regular file: {output}")

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw_archive)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, output)
            if os.name != "nt":
                directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            temporary_path.unlink(missing_ok=True)
    except OSError as exc:
        raise ReleaseHygieneError(f"cannot write verified archive '{output}': {exc}") from exc


def build_release_manifest(
    repo: Path,
    *,
    archive_output: Path | None = None,
    expected_commit: str | None = None,
) -> ReleaseManifest:
    """Build a manifest from exact HEAD blobs after enforcing release hygiene."""

    root = _repository_root(repo)
    commit_sha = _commit_sha(root)
    if expected_commit is not None:
        resolved_expected = _resolve_expected_commit(root, expected_commit)
        if resolved_expected != commit_sha:
            raise ReleaseHygieneError(
                f"expected commit does not match HEAD: {resolved_expected} != {commit_sha}"
            )
    _require_clean_worktree(root)
    tracked = _tracked_files(root, commit_sha)
    tracked_paths = tuple(item.path for item in tracked)
    ignored = _ignored_tracked_paths(root, tracked_paths)
    if ignored:
        rendered = ", ".join(ignored)
        raise ReleaseHygieneError(f"tracked files match ignore rules: {rendered}")
    forbidden = forbidden_tracked_paths(tracked_paths)
    if forbidden:
        rendered = ", ".join(forbidden)
        raise ReleaseHygieneError(f"generated files are tracked: {rendered}")
    schema_symlinks = _forbidden_schema_symlinks(tracked)
    if schema_symlinks:
        rendered = ", ".join(schema_symlinks)
        raise ReleaseHygieneError(f"tracked schema paths must not be symlinks: {rendered}")
    files = _read_blobs(root, tracked)
    raw_archive = _verify_archive(root, commit_sha, tracked, files)
    _require_clean_worktree(root)
    _require_head_unchanged(root, commit_sha)
    if archive_output is not None:
        _write_verified_archive(root, archive_output, raw_archive)
    return ReleaseManifest(commit_sha=commit_sha, files=files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--archive-output",
        type=Path,
        help="atomically write the exact tar stream validated for this manifest",
    )
    parser.add_argument(
        "--expected-commit",
        help="require this SHA or ref to resolve to the captured HEAD commit",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate release hygiene without writing the manifest to stdout",
    )
    args = parser.parse_args(argv)

    try:
        manifest = build_release_manifest(
            args.repo,
            archive_output=args.archive_output,
            expected_commit=args.expected_commit,
        )
    except ReleaseHygieneError as exc:
        print(f"release hygiene check failed: {exc}", file=sys.stderr)
        return 1

    if not args.check_only:
        sys.stdout.buffer.write(manifest.to_bytes())
    print(
        f"release hygiene check passed: {len(manifest.files)} tracked file(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANIFEST_FORMAT",
    "ReleaseFile",
    "ReleaseHygieneError",
    "ReleaseManifest",
    "build_release_manifest",
    "forbidden_tracked_paths",
    "main",
]
