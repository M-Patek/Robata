from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INDEX_CATALOG_PATH = "schemas/schema-catalog.json"
LEGACY_EXACT_BYTE_SCHEMAS = {
    "schemas/v1/inference-attempt-selection.schema.json",
    "schemas/v1/inference-intent.schema.json",
    "schemas/v1/model-inference.schema.json",
    "schemas/v1/parsed-provider-claim-artifact.schema.json",
    "schemas/v1/raw-provider-response-artifact.schema.json",
    "schemas/v1/selected-attempt-output.schema.json",
}


def _git(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def _index_blob(path: str) -> bytes:
    return _git("show", f":{path}")


def _index_attributes(paths: tuple[str, ...]) -> dict[tuple[str, str], str]:
    raw = _git(
        "check-attr",
        "--cached",
        "-z",
        "text",
        "eol",
        "whitespace",
        "--",
        *paths,
    )
    fields = raw.split(b"\0")
    assert fields.pop() == b""
    assert len(fields) % 3 == 0
    return {
        (fields[index].decode("utf-8"), fields[index + 1].decode("ascii")): fields[
            index + 2
        ].decode("ascii")
        for index in range(0, len(fields), 3)
    }


def test_release_index_preserves_exact_schema_pins_and_checkout_attributes() -> None:
    policy_probes = (
        "schemas/v-next/release-policy-probe.schema.json",
        "src/release_policy_probe.py",
        "docs/release-policy-probe.md",
    )
    catalog = json.loads(_index_blob(INDEX_CATALOG_PATH).decode("utf-8"))
    entries = catalog["schemas"]
    artifact_paths = (entry["artifact_path"] for entry in entries)
    schema_paths = tuple(sorted(f"schemas/{path}" for path in artifact_paths))
    attributes = _index_attributes((*schema_paths, *policy_probes))
    non_lf_artifacts: set[str] = set()

    for entry in entries:
        artifact_path = entry["artifact_path"]
        path = f"schemas/{artifact_path}"
        document = _index_blob(path)
        assert hashlib.sha256(document).hexdigest() == entry["ref"]["sha256"], path
        if b"\r\n" in document:
            non_lf_artifacts.add(path)

        if path in LEGACY_EXACT_BYTE_SCHEMAS:
            assert attributes[path, "text"] == "unset"
            assert attributes[path, "eol"] == "unset"
            assert attributes[path, "whitespace"] == "cr-at-eol"
        else:
            assert attributes[path, "text"] == "set"
            assert attributes[path, "eol"] == "lf"

    assert non_lf_artifacts == LEGACY_EXACT_BYTE_SCHEMAS
    for path in policy_probes:
        assert attributes[path, "text"] == "set"
        assert attributes[path, "eol"] == "lf"
