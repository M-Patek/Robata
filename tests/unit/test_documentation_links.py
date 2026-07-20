from __future__ import annotations

from pathlib import Path

from scripts.check_doc_links import check_document_links


def test_document_link_checker_reports_only_invalid_local_targets(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "present.md").write_text("# Present\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[valid](docs/present.md) [anchor](#local) "
        "[external](https://example.invalid/x) [missing](docs/missing.md)\n",
        encoding="utf-8",
    )

    issues = check_document_links(tmp_path)

    assert len(issues) == 1
    assert issues[0].target == "docs/missing.md"
    assert issues[0].reason == "TARGET_NOT_FOUND"


def test_current_authoritative_document_links_resolve() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    assert check_document_links(repository_root) == ()
