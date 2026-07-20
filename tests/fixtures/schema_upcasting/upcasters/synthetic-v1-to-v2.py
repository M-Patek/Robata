"""Pure transform used only by the registry-backed schema-upcasting fixture."""

from __future__ import annotations

from typing import Any


def upcast(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "schema_version": "2.0"}
