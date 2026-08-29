from __future__ import annotations

import json
import zipfile
from pathlib import Path
from uuid import uuid4

from robata.benchmark.production_corpus_audit import audit_zip_archive


def _archive_path() -> Path:
    root = Path(__file__).resolve().parents[2] / ".agent_tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"production-corpus-audit-test-{uuid4().hex}.zip"


def test_audit_zip_surfaces_media_and_distinguishes_rules_from_sidecars() -> None:
    archive = _archive_path()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("file/sample.mcap", b"mcap")
        handle.writestr("file/annotation principal.txt", b"rules")
        handle.writestr("file/QA_issue_list.md", b"qa")
        handle.writestr("labels/actions.jsonl", b"[]")

    try:
        inventory = audit_zip_archive(archive)
        assert len(inventory.mcap_entries) == 1
        assert inventory.mcap_total_uncompressed_bytes == 4
        assert len(inventory.rule_or_qa_entries) == 2
        assert len(inventory.structured_action_sidecar_candidates) == 1
        assert inventory.has_reviewed_action_sidecar is True
        payload = inventory.to_dict()
        assert payload["content_extracted"] is False
        assert payload["sha_or_digest_computed"] is False
    finally:
        archive.unlink(missing_ok=True)


def test_rule_only_archive_is_explicitly_unlabelled() -> None:
    archive = _archive_path()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("sample.mcap", b"x")
        handle.writestr("annotation principal.txt", b"rules")

    try:
        payload = audit_zip_archive(archive).to_dict()
        assert payload["mcap_count"] == 1
        assert payload["structured_action_sidecar_candidates"] == []
        assert payload["label_status"] == "UNLABELLED_SOURCE_MEDIA"
    finally:
        archive.unlink(missing_ok=True)


def test_script_payload_is_json_serialisable() -> None:
    archive = _archive_path()
    output = archive.with_suffix(".json")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("sample.mcap", b"x")
    try:
        payload = audit_zip_archive(archive).to_dict()
        output.write_text(json.dumps(payload), encoding="utf-8")
        assert json.loads(output.read_text(encoding="utf-8"))["mcap_count"] == 1
    finally:
        archive.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
