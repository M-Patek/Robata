"""Opt-in, deterministic test sharding for constrained CI runners."""

from __future__ import annotations

import os
from hashlib import sha256

import pytest

_SHARD_INDEX_ENV = "ROBATA_TEST_SHARD_INDEX"
_SHARD_TOTAL_ENV = "ROBATA_TEST_SHARD_TOTAL"


def _shard_configuration() -> tuple[int, int] | None:
    raw_index = os.environ.get(_SHARD_INDEX_ENV)
    raw_total = os.environ.get(_SHARD_TOTAL_ENV)
    if raw_index is None and raw_total is None:
        return None
    if raw_index is None or raw_total is None:
        raise pytest.UsageError(f"{_SHARD_INDEX_ENV} and {_SHARD_TOTAL_ENV} must be set together")
    try:
        index = int(raw_index)
        total = int(raw_total)
    except ValueError as error:
        raise pytest.UsageError("test shard values must be integers") from error
    if total < 1:
        raise pytest.UsageError("ROBATA_TEST_SHARD_TOTAL must be positive")
    if index < 0 or index >= total:
        raise pytest.UsageError("ROBATA_TEST_SHARD_INDEX must be in [0, ROBATA_TEST_SHARD_TOTAL)")
    return index, total


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Assign each collected node ID to exactly one stable CI shard."""

    shard = _shard_configuration()
    if shard is None:
        return
    index, total = shard
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        digest = sha256(item.nodeid.encode("utf-8")).digest()
        owner = int.from_bytes(digest[:8], byteorder="big") % total
        (selected if owner == index else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
