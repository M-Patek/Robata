"""Check local Markdown links in current authoritative documentation."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)\n]+)\)")


@dataclass(frozen=True, slots=True)
class DocumentLinkIssue:
    """One invalid local link and its stable failure reason."""

    document: Path
    target: str
    reason: str


def _markdown_files(root: Path) -> tuple[Path, ...]:
    files = list(root.glob("*.md"))
    docs = root / "docs"
    if docs.is_dir():
        files.extend(docs.rglob("*.md"))
    return tuple(sorted({path.resolve() for path in files}))


def _link_destination(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    elif " " in target:
        target = target.split(" ", 1)[0]

    parsed = urlsplit(target)
    if parsed.scheme or target.startswith("#"):
        return None
    return unquote(parsed.path)


def check_document_links(root: Path) -> tuple[DocumentLinkIssue, ...]:
    """Return broken or escaping local links outside the historical archive scan."""

    resolved_root = root.resolve()
    issues: list[DocumentLinkIssue] = []
    for document in _markdown_files(resolved_root):
        text = document.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(text):
            raw_target = match.group("target")
            destination = _link_destination(raw_target)
            if not destination:
                continue
            candidate = (document.parent / destination).resolve()
            try:
                candidate.relative_to(resolved_root)
            except ValueError:
                issues.append(DocumentLinkIssue(document, raw_target, "TARGET_OUTSIDE_REPOSITORY"))
                continue
            if not candidate.exists():
                issues.append(DocumentLinkIssue(document, raw_target, "TARGET_NOT_FOUND"))
    return tuple(issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    issues = check_document_links(args.root)
    for issue in issues:
        document = issue.document.relative_to(args.root.resolve())
        print(f"{document}: {issue.target}: {issue.reason}")
    if issues:
        print(f"documentation link check failed: {len(issues)} issue(s)")
        return 1
    print("documentation link check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DocumentLinkIssue", "check_document_links", "main"]
