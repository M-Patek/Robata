"""Lightweight audit of the local production-video corpus.

The production sample bundle is intentionally treated as *unlabelled source
media* until a reviewed annotation sidecar is supplied.  This module inspects
ZIP central-directory metadata only: it does not extract media, decode frames,
compute digests, or infer labels from filenames.  It is therefore suitable for
the P0 data-readiness gate and keeps the evidence boundary explicit.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProductionCorpusAuditError(ValueError):
    """Raised when an archive cannot be inspected as a bounded corpus."""


_ANNOTATION_NAME_PARTS = (
    "annotation",
    "annotated",
    "label",
    "review",
    "ground_truth",
    "ground-truth",
    "action_segment",
    "action-segment",
    "structured_label",
    "structured-label",
)
_KNOWN_RULE_FILES = {"annotation principal.txt", "qa_issue_list.md", "qa issue list.md"}


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """Non-content metadata for one archive member."""

    name: str
    size_bytes: int
    is_directory: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "is_directory": self.is_directory,
        }


@dataclass(frozen=True, slots=True)
class ProductionCorpusInventory:
    """Bounded, non-content inventory used by the production data gate."""

    archive_path: str
    archive_entry_count: int
    archive_total_uncompressed_bytes: int
    mcap_entries: tuple[CorpusEntry, ...]
    annotation_like_entries: tuple[CorpusEntry, ...]
    structured_action_sidecar_candidates: tuple[CorpusEntry, ...]
    rule_or_qa_entries: tuple[CorpusEntry, ...]

    @property
    def mcap_total_uncompressed_bytes(self) -> int:
        return sum(item.size_bytes for item in self.mcap_entries)

    @property
    def has_reviewed_action_sidecar(self) -> bool:
        """Whether a likely structured action sidecar is present.

        This is deliberately a *readiness hint*, not proof that labels are
        valid.  A human-reviewed manifest still needs schema and provenance
        validation before it can be used as a gold reference.
        """

        return bool(self.structured_action_sidecar_candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "robata-production-corpus-inventory-v1",
            "archive_path": self.archive_path,
            "archive_entry_count": self.archive_entry_count,
            "archive_total_uncompressed_bytes": self.archive_total_uncompressed_bytes,
            "mcap_count": len(self.mcap_entries),
            "mcap_total_uncompressed_bytes": self.mcap_total_uncompressed_bytes,
            "mcap_entries": [item.to_dict() for item in self.mcap_entries],
            "annotation_like_entries": [item.to_dict() for item in self.annotation_like_entries],
            "structured_action_sidecar_candidates": [
                item.to_dict() for item in self.structured_action_sidecar_candidates
            ],
            "rule_or_qa_entries": [item.to_dict() for item in self.rule_or_qa_entries],
            "has_reviewed_action_sidecar": self.has_reviewed_action_sidecar,
            "label_status": (
                "POSSIBLE_SIDECAR_REQUIRES_REVIEW"
                if self.has_reviewed_action_sidecar
                else "UNLABELLED_SOURCE_MEDIA"
            ),
            "content_extracted": False,
            "frames_decoded": False,
            "sha_or_digest_computed": False,
        }


def _entry(info: zipfile.ZipInfo) -> CorpusEntry:
    name = str(info.filename)
    return CorpusEntry(
        name=name,
        size_bytes=int(info.file_size),
        is_directory=bool(info.is_dir() or name.endswith(("/", "\\"))),
    )


def _is_annotation_like(name: str) -> bool:
    lowered = name.replace("\\", "/").casefold()
    basename = lowered.rsplit("/", 1)[-1]
    if basename in _KNOWN_RULE_FILES:
        return False
    suffix = Path(basename).suffix
    if suffix not in {".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".txt"}:
        return False
    return any(part in lowered for part in _ANNOTATION_NAME_PARTS)


def _is_structured_sidecar(entry: CorpusEntry) -> bool:
    lowered = entry.name.replace("\\", "/").casefold()
    basename = lowered.rsplit("/", 1)[-1]
    if basename in _KNOWN_RULE_FILES:
        return False
    suffix = Path(basename).suffix
    if suffix not in {".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml"}:
        return False
    # A filename alone cannot establish a label contract, but these names are
    # useful candidates to surface for a later schema-aware review.
    return any(part in lowered for part in ("action", "segment", "label", "annot"))


def audit_zip_archive(path: str | Path) -> ProductionCorpusInventory:
    """Inspect a ZIP central directory without reading member contents."""

    archive = Path(path).expanduser().resolve()
    if not archive.is_file():
        raise ProductionCorpusAuditError(f"archive is not a file: {archive}")
    try:
        with zipfile.ZipFile(archive) as handle:
            infos = tuple(handle.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProductionCorpusAuditError(f"cannot inspect ZIP archive {archive}: {exc}") from exc

    entries = tuple(_entry(info) for info in infos)
    mcap_entries = tuple(
        item for item in entries if not item.is_directory and item.name.casefold().endswith(".mcap")
    )
    annotation_like = tuple(
        item for item in entries if not item.is_directory and _is_annotation_like(item.name)
    )
    structured = tuple(item for item in annotation_like if _is_structured_sidecar(item))
    rules = tuple(
        item
        for item in entries
        if not item.is_directory
        and item.name.replace("\\", "/").rsplit("/", 1)[-1].casefold() in _KNOWN_RULE_FILES
    )
    return ProductionCorpusInventory(
        archive_path=str(archive),
        archive_entry_count=len(entries),
        archive_total_uncompressed_bytes=sum(item.size_bytes for item in entries),
        mcap_entries=tuple(sorted(mcap_entries, key=lambda item: item.name)),
        annotation_like_entries=tuple(sorted(annotation_like, key=lambda item: item.name)),
        structured_action_sidecar_candidates=tuple(sorted(structured, key=lambda item: item.name)),
        rule_or_qa_entries=tuple(sorted(rules, key=lambda item: item.name)),
    )


__all__ = [
    "CorpusEntry",
    "ProductionCorpusAuditError",
    "ProductionCorpusInventory",
    "audit_zip_archive",
]
