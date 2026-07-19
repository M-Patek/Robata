"""Windows-spawn and PNG encoder-reuse engineering probes.

These helpers are deliberately non-certifying.  They provide repeatable evidence that can be
attached to a governed benchmark once a production corpus and approval are available.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True, slots=True)
class SpawnProbeReport:
    """Result of a small ``spawn`` process-pool probe."""

    requested: int
    completed: int
    outputs_sha256: str
    elapsed_ms: int
    supported: bool
    error: str | None = None

    @property
    def certifying(self) -> bool:
        return False


def _spawn_identity(value: int) -> tuple[int, str]:
    """Top-level worker function required by Windows spawn pickling."""

    return value, multiprocessing.current_process().name


def run_spawn_probe(iterations: int = 8, *, max_workers: int = 2) -> SpawnProbeReport:
    """Run a bounded Windows-compatible spawn probe.

    The function returns a report rather than raising when a restricted runtime cannot spawn
    workers (for example an interactive interpreter).  This makes the result suitable for CI
    diagnostics while preserving a strict ``supported`` flag.
    """

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    started = time.perf_counter()
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
            values = tuple(pool.map(_spawn_identity, range(iterations)))
        digest = hashlib.sha256(repr(values).encode("utf-8")).hexdigest()
        elapsed_ms = max(1, round((time.perf_counter() - started) * 1_000))
        return SpawnProbeReport(
            requested=iterations,
            completed=len(values),
            outputs_sha256=digest,
            elapsed_ms=elapsed_ms,
            supported=len(values) == iterations,
        )
    except Exception as error:  # pragma: no cover - platform/runtime dependent
        elapsed_ms = max(1, round((time.perf_counter() - started) * 1_000))
        return SpawnProbeReport(
            requested=iterations,
            completed=0,
            outputs_sha256=hashlib.sha256(b"").hexdigest(),
            elapsed_ms=elapsed_ms,
            supported=False,
            error=f"{type(error).__name__}: {error}",
        )


@dataclass(frozen=True, slots=True)
class PngReuseReport:
    """Byte-stability result for isolated versus reusable PyAV PNG encoding."""

    frame_count: int
    isolated_sha256: tuple[str, ...]
    reused_sha256: tuple[str, ...]
    byte_identical: bool
    supported: bool
    error: str | None = None

    @property
    def certifying(self) -> bool:
        return False


class ReusablePngEncoder:
    """Reuse a configured PyAV PNG codec for same-size RGB frames.

    PNG packets are emitted without flushing between frames.  A codec is recreated when the
    dimensions or pixel format change.  The caller should call :meth:`close` after the final
    frame if it needs to drain a stream; materialization uses one independent PNG packet per
    frame and therefore does not flush.
    """

    def __init__(self) -> None:
        self._encoder: Any = None
        self._shape: tuple[int, int, str] | None = None

    def encode(self, frame: Any, *, max_width: int | None = None) -> tuple[bytes, int, int]:
        try:
            output_width = frame.width if max_width is None else min(frame.width, max_width)
            output_height = max(1, (frame.height * output_width + frame.width // 2) // frame.width)
            rgb_frame = frame.reformat(
                width=output_width,
                height=output_height,
                format="rgb24",
            )
            rgb_frame.pts = 0
            rgb_frame.time_base = Fraction(1, 1)
            shape = (rgb_frame.width, rgb_frame.height, "rgb24")
            if self._encoder is None or self._shape != shape:
                import av

                encoder = av.CodecContext.create("png", "w")
                encoder.width = rgb_frame.width
                encoder.height = rgb_frame.height
                encoder.pix_fmt = "rgb24"
                encoder.time_base = Fraction(1, 1)
                self._encoder = encoder
                self._shape = shape
            packets = list(self._encoder.encode(rgb_frame))
            png_bytes = b"".join(bytes(packet) for packet in packets)
            if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("reusable PyAV encoder returned invalid PNG bytes")
            return png_bytes, rgb_frame.width, rgb_frame.height
        except Exception:
            # A codec may become unusable after a PyAV error; do not reuse it silently.
            self._encoder = None
            self._shape = None
            raise

    def close(self) -> None:
        if self._encoder is not None:
            try:
                self._encoder.encode(None)
            except Exception:
                pass
            finally:
                self._encoder = None
                self._shape = None


def compare_png_reuse(frames: list[Any], *, max_width: int | None = None) -> PngReuseReport:
    """Compare deterministic bytes from isolated and reusable encoders."""

    if not frames:
        raise ValueError("frames must be non-empty")
    try:
        from robata.adapters.pyav_frame_materializer import _encode_png

        isolated: list[bytes] = []
        for frame in frames:
            isolated.append(_encode_png(frame, max_width=max_width)[0])
        reusable_encoder = ReusablePngEncoder()
        reused: list[bytes] = []
        try:
            for frame in frames:
                reused.append(reusable_encoder.encode(frame, max_width=max_width)[0])
        finally:
            reusable_encoder.close()
        isolated_hashes = tuple(hashlib.sha256(item).hexdigest() for item in isolated)
        reused_hashes = tuple(hashlib.sha256(item).hexdigest() for item in reused)
        return PngReuseReport(
            frame_count=len(frames),
            isolated_sha256=isolated_hashes,
            reused_sha256=reused_hashes,
            byte_identical=isolated == reused,
            supported=True,
        )
    except Exception as error:  # pragma: no cover - optional PyAV/runtime dependent
        return PngReuseReport(
            frame_count=len(frames),
            isolated_sha256=(),
            reused_sha256=(),
            byte_identical=False,
            supported=False,
            error=f"{type(error).__name__}: {error}",
        )


__all__ = [
    "PngReuseReport",
    "ReusablePngEncoder",
    "SpawnProbeReport",
    "compare_png_reuse",
    "run_spawn_probe",
]
