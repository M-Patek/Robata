"""Small, process-local decode cache for WeMM benchmark passes.

The production WeMM runner intentionally keeps decode and model scheduling
separate.  That is useful for a single pass, but a frame/grid matrix otherwise
replays MCAP/H.264 decode for every arm.  This module provides an *opt-in*
in-memory seam for those bounded experiments:

* callers provide an explicit ``scope_key`` (the cache never derives an
  identity, reads source files, writes files, or computes a digest);
* a miss materializes the caller's chunk iterator once;
* every consumer receives close-safe copies of the frame groups, so the
  existing runner may release its PIL images without invalidating the cache;
* the cache is bounded by a caller-selected number of scopes and exposes only
  lightweight hit/miss/eviction counters.

This is deliberately not a durable production media cache.  It is a bounded
benchmark helper until the canonical source-media cache contract is wired into
the production ingest path.
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Final

type DecodeChunk = Mapping[str, Mapping[str, Any]]
type DecodeFactory = Callable[[], Iterable[DecodeChunk]]

DECODE_CACHE_VERSION: Final = "robata-production-wemm-decode-cache-v1"


class ProductionWemmDecodeCacheError(ValueError):
    """Raised when a bounded decode-cache operation is invalid."""


@dataclass(frozen=True, slots=True)
class ProductionWemmDecodeCacheStats:
    """Lightweight process-local cache counters."""

    scope_count: int
    hit_count: int
    miss_count: int
    eviction_count: int
    cached_chunk_count: int

    @property
    def request_count(self) -> int:
        return self.hit_count + self.miss_count

    def to_dict(self) -> dict[str, int]:
        return {
            "scope_count": self.scope_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
            "cached_chunk_count": self.cached_chunk_count,
            "request_count": self.request_count,
        }


def _close_frame(frame: Any) -> None:
    close = getattr(frame, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


def close_decoded_chunks(chunks: Iterable[DecodeChunk]) -> None:
    """Close frame objects contained in chunks returned by the cache.

    The normal WeMM runner already owns this lifecycle.  The helper is exposed
    for small benchmark callers and tests that consume a cache directly.
    """

    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        for camera_groups in chunk.values():
            if not isinstance(camera_groups, Mapping):
                continue
            for group in camera_groups.values():
                frames = getattr(group, "frames", None)
                if frames is None and isinstance(group, Mapping):
                    frames = group.get("frames", ())
                if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes, bytearray)):
                    continue
                for frame in frames:
                    _close_frame(frame)


def _clone_frame(frame: Any) -> Any:
    """Return an independently closeable copy when the frame supports it."""

    copier = getattr(frame, "copy", None)
    if callable(copier):
        with suppress(Exception):
            return copier()
    with suppress(Exception):
        return copy.copy(frame)
    # Immutable fixture values (for example strings) are safe to share.  A
    # closeable value that cannot be copied is rejected by _clone_group below
    # rather than silently handing ownership of the cache's frame to a caller.
    return frame


def _clone_group(group: Any) -> Any:
    """Clone a decoded frame-group while retaining its metadata."""

    frames = getattr(group, "frames", None)
    if frames is None and isinstance(group, Mapping):
        frames = group.get("frames")
    if frames is None:
        # Some lightweight fakes have no frame attribute.  They are metadata
        # only and can be safely shared.
        return group
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes, bytearray)):
        raise ProductionWemmDecodeCacheError("decoded group frames must be a sequence")
    cloned_frames = tuple(_clone_frame(frame) for frame in frames)
    for original, cloned in zip(frames, cloned_frames, strict=True):
        if cloned is original and callable(getattr(original, "close", None)):
            raise ProductionWemmDecodeCacheError(
                "decoded frame is closeable but cannot be copied for cache replay"
            )

    if is_dataclass(group):
        with suppress(TypeError, ValueError):
            return replace(group, frames=cloned_frames)  # type: ignore[type-var]

    if isinstance(group, Mapping):
        mapping_copy = dict(group)
        mapping_copy["frames"] = cloned_frames
        return mapping_copy

    with suppress(Exception):
        object_copy: Any = copy.copy(group)
        object_copy.frames = cloned_frames
        return object_copy
    raise ProductionWemmDecodeCacheError("decoded frame group cannot be copied")


def _clone_chunk(chunk: DecodeChunk) -> dict[str, dict[str, Any]]:
    if not isinstance(chunk, Mapping):
        raise ProductionWemmDecodeCacheError("decode factory yielded a non-mapping chunk")
    cloned: dict[str, dict[str, Any]] = {}
    for camera_id, camera_groups in chunk.items():
        if not isinstance(camera_groups, Mapping):
            raise ProductionWemmDecodeCacheError("decode chunk camera groups must be mappings")
        cloned[str(camera_id)] = {
            str(window_id): _clone_group(group) for window_id, group in camera_groups.items()
        }
    return cloned


def _close_stored_chunks(chunks: Iterable[DecodeChunk]) -> None:
    """Close cache-owned groups without touching caller-owned clones."""

    close_decoded_chunks(chunks)


class ProductionWemmDecodeCache:
    """Bounded in-memory cache for decoded WeMM frame-group chunks.

    ``scope_key`` must be supplied by the caller and be hashable.  A useful key
    is normally a tuple containing the recording/window cohort and decode
    parameters, but this class deliberately does not inspect or derive it.
    """

    def __init__(self, *, max_scopes: int = 2) -> None:
        if isinstance(max_scopes, bool) or not isinstance(max_scopes, int) or max_scopes <= 0:
            raise ProductionWemmDecodeCacheError("max_scopes must be a positive integer")
        self.max_scopes = max_scopes
        self._scopes: OrderedDict[Hashable, tuple[DecodeChunk, ...]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def iter_chunks(self, scope_key: Hashable, decoder: DecodeFactory) -> Iterator[DecodeChunk]:
        """Yield close-safe decoded chunks, materializing ``decoder`` on miss."""

        if not isinstance(scope_key, Hashable):
            raise ProductionWemmDecodeCacheError("scope_key must be hashable")
        if not callable(decoder):
            raise ProductionWemmDecodeCacheError("decoder must be callable")

        cached = self._scopes.get(scope_key)
        if cached is not None:
            self._hits += 1
            self._scopes.move_to_end(scope_key)
            for chunk in cached:
                yield _clone_chunk(chunk)
            return

        self._misses += 1
        stored: list[DecodeChunk] = []
        iterator: Iterator[DecodeChunk] | None = None
        try:
            raw = decoder()
            if raw is None:
                raise ProductionWemmDecodeCacheError("decoder returned None")
            iterator = iter(raw)
            for chunk in iterator:
                # Keep a shallow mapping copy so a caller cannot mutate the
                # cache's indexing structure.  The frame groups remain owned
                # by the cache and are never returned directly.
                if not isinstance(chunk, Mapping):
                    raise ProductionWemmDecodeCacheError(
                        "decode factory yielded a non-mapping chunk"
                    )
                stored_chunk: dict[str, dict[str, Any]] = {}
                for camera_id, camera_groups in chunk.items():
                    if not isinstance(camera_groups, Mapping):
                        raise ProductionWemmDecodeCacheError(
                            "decode chunk camera groups must be mappings"
                        )
                    stored_chunk[str(camera_id)] = dict(camera_groups)
                stored.append(stored_chunk)
        except Exception:
            _close_stored_chunks(stored)
            raise
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()

        if not stored:
            raise ProductionWemmDecodeCacheError("decoder yielded no chunks")
        previous = self._scopes.pop(scope_key, None)
        if previous is not None:
            _close_stored_chunks(previous)
        self._scopes[scope_key] = tuple(stored)
        while len(self._scopes) > self.max_scopes:
            _evicted_key, evicted = self._scopes.popitem(last=False)
            del _evicted_key
            _close_stored_chunks(evicted)
            self._evictions += 1
        for chunk in self._scopes[scope_key]:
            yield _clone_chunk(chunk)

    def stats(self) -> ProductionWemmDecodeCacheStats:
        return ProductionWemmDecodeCacheStats(
            scope_count=len(self._scopes),
            hit_count=self._hits,
            miss_count=self._misses,
            eviction_count=self._evictions,
            cached_chunk_count=sum(len(chunks) for chunks in self._scopes.values()),
        )

    def clear(self) -> None:
        """Release all cached frame groups and reset scope storage."""

        for chunks in self._scopes.values():
            _close_stored_chunks(chunks)
        self._scopes.clear()

    close = clear

    def __enter__(self) -> ProductionWemmDecodeCache:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.clear()


def measure_decode_cache_pass(
    cache: ProductionWemmDecodeCache,
    scope_key: Hashable,
    decoder: DecodeFactory,
) -> dict[str, Any]:
    """Measure one cache pass without invoking a model.

    The returned timing is intentionally small and diagnostic: it separates a
    cache miss (decode/materialization) from a replay hit.  Returned frame
    groups are closed before the helper returns; use ``cache.iter_chunks`` when
    a benchmark needs to feed the groups to a model.
    """

    before = cache.stats()
    started = time.perf_counter()
    chunks = tuple(cache.iter_chunks(scope_key, decoder))
    elapsed = max(0.0, time.perf_counter() - started)
    close_decoded_chunks(chunks)
    after = cache.stats()
    hit = after.hit_count > before.hit_count
    return {
        "cache_version": DECODE_CACHE_VERSION,
        "cache_hit": hit,
        "elapsed_seconds": elapsed,
        "chunk_count": len(chunks),
        "stats": after.to_dict(),
    }


__all__ = [
    "DECODE_CACHE_VERSION",
    "DecodeChunk",
    "DecodeFactory",
    "ProductionWemmDecodeCache",
    "ProductionWemmDecodeCacheError",
    "ProductionWemmDecodeCacheStats",
    "close_decoded_chunks",
    "measure_decode_cache_pass",
]
