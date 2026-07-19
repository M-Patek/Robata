"""Verify that every checked-in wire schema is valid and locally resolvable."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.contracts.schema_registry import SchemaRegistry  # noqa: E402


def main() -> int:
    registry = SchemaRegistry(REPOSITORY_ROOT / "schemas" / "schema-catalog.json")
    registry.validate_schema_documents()
    for registered in registry.entries:
        if registry.resolve_exact(registered.ref) != registered:
            raise RuntimeError(f"exact schema resolution failed for {registered.ref!r}")
    print(
        f"verified schema catalog and {len(registry.entries)} pinned "
        "JSON Schema 2020-12 documents with offline references"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
