"""Deterministic canonical JSON and SHA-256 identity helpers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

import rfc8785
from pydantic import BaseModel

from robata.contracts.common import Sha256Digest

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented by RFC 8785 canonical JSON."""


def _json_projection(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                projected_key = key
            elif isinstance(key, Enum) and isinstance(key.value, str):
                projected_key = key.value
            else:
                raise TypeError("canonical JSON object keys must be strings")
            if projected_key in projected:
                raise TypeError(f"duplicate canonical JSON object key: {projected_key!r}")
            projected[projected_key] = _json_projection(item)
        return projected
    if isinstance(value, tuple):
        return [_json_projection(item) for item in value]
    if isinstance(value, list):
        return [_json_projection(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible semantic projection according to RFC 8785."""

    try:
        return rfc8785.dumps(_json_projection(value))
    except (rfc8785.CanonicalizationError, TypeError, ValueError) as exc:
        raise CanonicalizationError(str(exc)) from exc


def exact_bytes_sha256(data: bytes | bytearray | memoryview) -> Sha256Digest:
    """Hash exact stored bytes without canonicalization or text conversion."""

    return hashlib.sha256(data).hexdigest()


def semantic_sha256(projection: Any) -> Sha256Digest:
    """Hash an explicit semantic projection using RFC 8785 canonical JSON."""

    return exact_bytes_sha256(canonical_json_bytes(projection))


def recording_identity(namespace: str, source_content_sha256: Sha256Digest) -> Sha256Digest:
    """Derive a URI-independent recording identity from namespace and content.

    Source aliases, object versions, and paths are deliberately absent from the preimage.
    """

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a nonempty string")
    if (
        not isinstance(source_content_sha256, str)
        or _SHA256_PATTERN.fullmatch(source_content_sha256) is None
    ):
        raise ValueError("source_content_sha256 must be 64 lowercase hexadecimal characters")
    return semantic_sha256(
        {
            "namespace": namespace,
            "source_content_sha256": source_content_sha256,
        }
    )


# Explicit aliases make the distinction visible at call sites.
canonical_sha256 = semantic_sha256
sha256_bytes = exact_bytes_sha256
