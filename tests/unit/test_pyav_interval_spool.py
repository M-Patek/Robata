from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("av")

from robata.adapters.pyav_mp4_exporter import _Int64IntervalSpool


def test_many_intervals_retain_only_fixed_size_memory_state(tmp_path: Path) -> None:
    path = tmp_path / "intervals.bin"
    path.touch()
    spool = _Int64IntervalSpool(path)
    try:
        for _ in range(100):
            for value in range(1, 1_001):
                spool.append(value)

        assert spool.count == 100_000
        assert spool.minimum == 1
        assert spool.maximum == 1_000
        assert spool.median_half_even() == 500
        assert path.stat().st_size == spool.count * 8
        assert not hasattr(spool, "__dict__")
        assert all(
            not isinstance(getattr(spool, slot), list) for slot in _Int64IntervalSpool.__slots__
        )
    finally:
        spool.close()
