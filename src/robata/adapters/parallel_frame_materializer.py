"""Opt-in local parallel frame materialization adapter."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from robata.adapters.pyav_frame_materializer import PyAvFrameMaterializer


class ParallelPyAvFrameMaterializer(PyAvFrameMaterializer):
    """Materialize independent camera plans concurrently with deterministic merge order."""

    def __init__(
        self,
        *,
        max_width: int | None = 320,
        max_workers: int = 6,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if clock is None:
            super().__init__(max_width=max_width, max_parallel_cameras=max_workers)
        else:
            super().__init__(
                max_width=max_width,
                max_parallel_cameras=max_workers,
                clock=clock,
            )


__all__ = ["ParallelPyAvFrameMaterializer"]
