from __future__ import annotations

from dataclasses import dataclass

import pytest

from robata.benchmark.production_wemm_decode_cache import (
    ProductionWemmDecodeCache,
    ProductionWemmDecodeCacheError,
    close_decoded_chunks,
    measure_decode_cache_pass,
)


@dataclass
class _Frame:
    token: str
    closed: bool = False

    def copy(self) -> _Frame:
        return _Frame(self.token)

    def close(self) -> None:
        self.closed = True


@dataclass
class _Group:
    frames: tuple[_Frame, ...]


def _factory(calls: list[int], originals: list[_Frame]):
    def decode():
        calls.append(1)
        frame = _Frame("a")
        originals.append(frame)
        yield {"cam_01": {"w00": _Group((frame,))}}

    return decode


def test_cache_materializes_once_and_returns_close_safe_copies() -> None:
    calls: list[int] = []
    originals: list[_Frame] = []
    cache = ProductionWemmDecodeCache(max_scopes=2)

    first = tuple(cache.iter_chunks(("cohort", 4, 1), _factory(calls, originals)))
    assert len(first) == 1
    first_frame = first[0]["cam_01"]["w00"].frames[0]
    assert first_frame is not originals[0]
    close_decoded_chunks(first)
    assert originals[0].closed is False

    second = tuple(cache.iter_chunks(("cohort", 4, 1), _factory(calls, originals)))
    second_frame = second[0]["cam_01"]["w00"].frames[0]
    assert len(calls) == 1
    assert second_frame is not first_frame
    close_decoded_chunks(second)

    stats = cache.stats()
    assert stats.hit_count == 1
    assert stats.miss_count == 1
    assert stats.scope_count == 1
    cache.clear()
    assert originals[0].closed is True


def test_cache_scope_eviction_closes_owned_frames() -> None:
    calls: list[int] = []
    originals: list[_Frame] = []
    cache = ProductionWemmDecodeCache(max_scopes=1)
    tuple(cache.iter_chunks("first", _factory(calls, originals)))
    tuple(cache.iter_chunks("second", _factory(calls, originals)))
    assert len(originals) == 2
    assert originals[0].closed is True
    assert cache.stats().eviction_count == 1
    cache.close()
    assert originals[1].closed is True


def test_cache_rejects_invalid_factory_and_uncopyable_closeable_frame() -> None:
    cache = ProductionWemmDecodeCache()
    with pytest.raises(ProductionWemmDecodeCacheError, match="decoder must be callable"):
        tuple(cache.iter_chunks("scope", None))  # type: ignore[arg-type]

    class CloseOnly:
        def close(self) -> None:
            return None

        def __copy__(self):
            raise TypeError("copy disabled")

    with pytest.raises(ProductionWemmDecodeCacheError, match="cannot be copied"):
        tuple(
            cache.iter_chunks(
                "uncopyable",
                lambda: iter({"cam_01": {"w00": _Group((CloseOnly(),))}} for _ in [0]),
            )
        )


def test_measure_decode_cache_pass_distinguishes_miss_and_hit() -> None:
    calls: list[int] = []
    originals: list[_Frame] = []
    cache = ProductionWemmDecodeCache()
    decoder = _factory(calls, originals)

    miss = measure_decode_cache_pass(cache, ("cohort", 4), decoder)
    hit = measure_decode_cache_pass(cache, ("cohort", 4), decoder)
    assert miss["cache_hit"] is False
    assert hit["cache_hit"] is True
    assert miss["chunk_count"] == hit["chunk_count"] == 1
    assert miss["elapsed_seconds"] >= 0.0
    assert hit["elapsed_seconds"] >= 0.0
    assert calls == [1]
    cache.clear()
