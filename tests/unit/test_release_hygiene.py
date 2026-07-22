from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import scripts.check_release_hygiene as release_hygiene
from scripts.check_release_hygiene import (
    MANIFEST_FORMAT,
    ReleaseHygieneError,
    build_release_manifest,
    forbidden_tracked_paths,
)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path, files: dict[str, bytes]) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "user.email", "release-test@example.invalid")
    _git(repo, "config", "user.name", "Release Test")
    for relative_path, content in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo


def test_forbidden_paths_cover_caches_temp_runs_and_root_archives() -> None:
    assert forbidden_tracked_paths(
        (
            "src/robata/__pycache__/module.pyc",
            ".mypy_cache/state.json",
            "nested/.ruff_cache/state",
            "nested/pytest-cache-files-worker/state.json",
            "pytest-temp-final/results.json",
            "schemas/.schema-publication-transaction.json",
            "workspace.zip",
            "fixtures/example.zip",
            "src/robata/module.py",
        )
    ) == (
        ".mypy_cache/state.json",
        "nested/.ruff_cache/state",
        "nested/pytest-cache-files-worker/state.json",
        "pytest-temp-final/results.json",
        "schemas/.schema-publication-transaction.json",
        "src/robata/__pycache__/module.pyc",
        "workspace.zip",
    )


def test_manifest_uses_sorted_exact_head_blobs_and_is_deterministic(tmp_path: Path) -> None:
    files = {
        "z-last.txt": b"last\r\n",
        "a-first.bin": b"\x00\xfffirst\n",
    }
    repo = _repository(tmp_path, files)

    first = build_release_manifest(repo)
    second = build_release_manifest(repo)

    assert first == second
    assert first.format_version == MANIFEST_FORMAT
    assert first.commit_sha == _git(repo, "rev-parse", "HEAD")
    assert tuple(item.path for item in first.files) == ("a-first.bin", "z-last.txt")
    assert tuple(item.sha256 for item in first.files) == tuple(
        hashlib.sha256(files[path]).hexdigest() for path in ("a-first.bin", "z-last.txt")
    )
    assert tuple(item.size for item in first.files) == tuple(
        len(files[path]) for path in ("a-first.bin", "z-last.txt")
    )
    rendered = first.to_bytes()
    assert rendered.endswith(b"\n")
    assert b"\r\n" not in rendered
    assert json.loads(rendered)["files"][0]["path"] == "a-first.bin"


def test_manifest_archive_is_independent_of_host_line_ending_config(
    tmp_path: Path,
) -> None:
    repo = _repository(
        tmp_path,
        {
            ".gitattributes": b"*.txt text eol=lf\n",
            "source.txt": b"source\n",
        },
    )
    _git(repo, "config", "core.autocrlf", "true")
    _git(repo, "config", "core.eol", "crlf")

    manifest = build_release_manifest(repo)

    assert tuple(item.path for item in manifest.files) == (
        ".gitattributes",
        "source.txt",
    )


def test_manifest_rejects_files_force_tracked_despite_ignore_rules(tmp_path: Path) -> None:
    ignore_rules = b".env\n.venv/\nbuild/\ndist/\n*.log\ncoverage/\ndata/\n.claude/\n"
    repo = _repository(
        tmp_path,
        {
            ".gitignore": ignore_rules,
            "source.txt": b"source\n",
        },
    )
    ignored_files = {
        ".env": b"secret\n",
        ".venv/pyvenv.cfg": b"home = fixture\n",
        "build/app.whl": b"build\n",
        "dist/app.zip": b"dist\n",
        "logs/run.log": b"log\n",
        "coverage/report.xml": b"coverage\n",
        "data/source/input.bin": b"data\n",
        ".claude/settings.json": b"{}\n",
    }
    for relative_path, content in ignored_files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git(repo, "add", "--force", "--", *ignored_files)
    _git(repo, "commit", "--quiet", "-m", "force ignored files")

    with pytest.raises(ReleaseHygieneError, match="tracked files match ignore rules") as caught:
        build_release_manifest(repo)

    for relative_path in ignored_files:
        assert relative_path in str(caught.value)


@pytest.mark.parametrize(
    ("attributes", "target_content", "message"),
    [
        ("hidden.txt export-ignore\n", b"hidden\n", "missing manifest entries: hidden.txt"),
        (
            "version.txt export-subst\n",
            b"commit=$Format:%H$\n",
            "content does not match manifest: version.txt",
        ),
    ],
)
def test_manifest_rejects_archive_entry_or_content_drift(
    tmp_path: Path,
    attributes: str,
    target_content: bytes,
    message: str,
) -> None:
    target_name = attributes.split(" ", 1)[0]
    repo = _repository(
        tmp_path,
        {
            ".gitattributes": attributes.encode("ascii"),
            target_name: target_content,
        },
    )

    with pytest.raises(ReleaseHygieneError, match=message):
        build_release_manifest(repo)


def test_manifest_rejects_head_change_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"source.txt": b"source\n"})
    original_verify_archive = release_hygiene._verify_archive

    def verify_archive_and_advance_head(
        checked_repo: Path,
        commit_sha: str,
        tracked: tuple[release_hygiene.TrackedFile, ...],
        files: tuple[release_hygiene.ReleaseFile, ...],
    ) -> bytes:
        raw_archive = original_verify_archive(checked_repo, commit_sha, tracked, files)
        (repo / "next.txt").write_bytes(b"next\n")
        _git(repo, "add", "next.txt")
        _git(repo, "commit", "--quiet", "-m", "advance head")
        return raw_archive

    monkeypatch.setattr(release_hygiene, "_verify_archive", verify_archive_and_advance_head)

    archive_output = tmp_path / "release.tar"
    with pytest.raises(ReleaseHygieneError, match="HEAD changed during release validation"):
        build_release_manifest(repo, archive_output=archive_output)
    assert not archive_output.exists()


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_manifest_rejects_dirty_worktree(tmp_path: Path, dirty_kind: str) -> None:
    repo = _repository(tmp_path, {"source.txt": b"source\n"})
    if dirty_kind == "tracked":
        (repo / "source.txt").write_bytes(b"changed\n")
    else:
        (repo / "new.txt").write_bytes(b"untracked\n")

    with pytest.raises(ReleaseHygieneError, match="clean worktree"):
        build_release_manifest(repo)


def test_manifest_rejects_generated_files_committed_to_head(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {".pytest_cache/state": b"generated\n"})

    with pytest.raises(ReleaseHygieneError, match=r"generated files are tracked: \.pytest_cache"):
        build_release_manifest(repo)


def test_archive_output_is_the_exact_verified_tar_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"source.txt": b"source\n"})
    output = tmp_path / "artifacts" / "source.tar"
    original_verify_archive = release_hygiene._verify_archive
    verified: list[bytes] = []

    def capture_verified_archive(
        checked_repo: Path,
        commit_sha: str,
        tracked: tuple[release_hygiene.TrackedFile, ...],
        files: tuple[release_hygiene.ReleaseFile, ...],
    ) -> bytes:
        raw_archive = original_verify_archive(checked_repo, commit_sha, tracked, files)
        verified.append(raw_archive)
        return raw_archive

    monkeypatch.setattr(release_hygiene, "_verify_archive", capture_verified_archive)
    head = _git(repo, "rev-parse", "HEAD")
    manifest = build_release_manifest(repo, archive_output=output, expected_commit=head)

    assert output.read_bytes() == verified[0]
    assert manifest.commit_sha == _git(repo, "rev-parse", "HEAD")


def test_release_rejects_expected_commit_that_is_not_head(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"source.txt": b"source\n"})
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "next.txt").write_bytes(b"next\n")
    _git(repo, "add", "next.txt")
    _git(repo, "commit", "--quiet", "-m", "next")
    output = tmp_path / "source.tar"

    with pytest.raises(ReleaseHygieneError, match="expected commit does not match HEAD"):
        build_release_manifest(repo, archive_output=output, expected_commit=first)
    assert not output.exists()


def test_release_rejects_tracked_schema_publication_marker_and_orphan(tmp_path: Path) -> None:
    repo = _repository(
        tmp_path,
        {
            "schemas/.schema-publication-transaction.json": b"{}\n",
            "schemas/v2/orphan.schema.json": b"{}\n",
        },
    )

    with pytest.raises(
        ReleaseHygieneError,
        match=r"generated files are tracked: schemas/\.schema-publication-transaction\.json",
    ):
        build_release_manifest(repo)


def test_release_rejects_schema_symlink_blob_with_plain_file_checkout(tmp_path: Path) -> None:
    path = "schemas/v1/alias.schema.json"
    repo = _repository(tmp_path, {path: b"base.schema.json"})
    _git(repo, "config", "core.symlinks", "false")
    blob = _git(repo, "hash-object", path)
    _git(repo, "update-index", "--cacheinfo", f"120000,{blob},{path}")
    _git(repo, "commit", "--quiet", "-m", "store schema symlink blob")
    _git(repo, "reset", "--hard", "--quiet", "HEAD")

    assert not (repo / path).is_symlink()
    with pytest.raises(ReleaseHygieneError, match=r"schema paths must not be symlinks"):
        build_release_manifest(repo)


def test_quality_workflow_uses_dual_schema_baselines_and_bound_archive_output() -> None:
    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "quality.yml"
    ).read_text(encoding="utf-8")

    assert "${{ vars.SCHEMA_BASELINE_REF }}" in workflow
    assert "--baseline-ref $eventBaseline" in workflow
    assert "--baseline-ref $env:SCHEMA_BASELINE_REF" in workflow
    assert "HEAD^" not in workflow
    assert "--archive-output $archive" in workflow
    assert '--expected-commit "${{ github.sha }}"' in workflow
    assert "git archive" not in workflow
    assert workflow.count("if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }") >= 6
