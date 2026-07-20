"""Verify that every checked-in wire schema is valid and locally resolvable."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.contracts.schema_registry import (  # noqa: E402
    SCHEMA_PUBLICATION_MARKER_FILENAME,
    SchemaRegistry,
)
from robata.contracts.schema_upcasting import SchemaUpcasterGraph  # noqa: E402


def main() -> int:
    schema_root = REPOSITORY_ROOT / "schemas"
    marker = schema_root / SCHEMA_PUBLICATION_MARKER_FILENAME
    if marker.exists() or marker.is_symlink():
        raise RuntimeError(f"pending schema publication marker must be recovered: {marker}")
    registry = SchemaRegistry(schema_root / "schema-catalog.json")
    registry.validate_schema_documents()
    for registered in registry.entries:
        if registry.resolve_exact(registered.ref) != registered:
            raise RuntimeError(f"exact schema resolution failed for {registered.ref!r}")
    SchemaUpcasterGraph(registry)
    print(
        f"verified schema catalog and {len(registry.entries)} pinned "
        "JSON Schema 2020-12 documents with offline references; "
        f"verified {len(registry.registered_upcasters)} pinned upcaster artifact sets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
