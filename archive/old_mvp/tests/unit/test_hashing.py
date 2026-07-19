import hashlib

import pytest

from robata.contracts.hashing import (
    CanonicalizationError,
    canonical_json_bytes,
    exact_bytes_sha256,
    recording_identity,
    semantic_sha256,
)


def test_canonical_json_is_rfc8785_ordered_and_compact() -> None:
    assert canonical_json_bytes({"z": 2, "a": "value"}) == b'{"a":"value","z":2}'


def test_semantic_hash_is_independent_of_mapping_insertion_order() -> None:
    left = {"namespace": "fixture", "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "namespace": "fixture"}

    assert semantic_sha256(left) == semantic_sha256(right)
    assert semantic_sha256(left) == hashlib.sha256(canonical_json_bytes(left)).hexdigest()


def test_exact_byte_hash_does_not_canonicalize() -> None:
    compact = b'{"a":1}'
    spaced = b'{ "a": 1 }'

    assert exact_bytes_sha256(compact) == hashlib.sha256(compact).hexdigest()
    assert exact_bytes_sha256(compact) != exact_bytes_sha256(spaced)


def test_recording_identity_uses_only_namespace_and_verified_content() -> None:
    content_digest = exact_bytes_sha256(b"same immutable MCAP bytes")

    first_alias = {"uri": "file:///capture.mcap", "version": "1"}
    moved_alias = {"uri": "object://archive/capture.mcap", "version": "9"}
    assert first_alias != moved_alias
    assert recording_identity("tenant-a", content_digest) == recording_identity(
        "tenant-a", content_digest
    )
    assert recording_identity("tenant-a", content_digest) == semantic_sha256(
        {"namespace": "tenant-a", "source_content_sha256": content_digest}
    )
    assert recording_identity("tenant-a", content_digest) != recording_identity(
        "tenant-b", content_digest
    )


@pytest.mark.parametrize(
    "namespace,digest",
    [
        ("", "0" * 64),
        ("tenant", "0" * 63),
        ("tenant", "A" * 64),
        ("tenant", "g" * 64),
    ],
)
def test_recording_identity_rejects_ambiguous_inputs(namespace: str, digest: str) -> None:
    with pytest.raises(ValueError):
        recording_identity(namespace, digest)


def test_canonical_json_rejects_non_json_values() -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes({"bad": {1, 2, 3}})
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes({1: "integer keys must not be coerced"})
