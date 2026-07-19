"""Opt-in local parallel camera export adapter.

The adapter deliberately reuses the verified serial publication service and only changes
how the six independent camera exports are scheduled.  Publication, source revalidation,
manifest construction, and registry semantics remain identical to the serial path.
"""

from __future__ import annotations

from robata.application.video_export import SixCameraVideoExportService
from robata.ports.video_export import CameraVideoExporter


class ParallelSixCameraVideoExportService(SixCameraVideoExportService):
    """Run independent camera exports concurrently with deterministic publication order."""

    def __init__(self, exporter: CameraVideoExporter, *, max_workers: int = 6) -> None:
        super().__init__(exporter, max_parallel_exports=max_workers)


__all__ = ["ParallelSixCameraVideoExportService"]
