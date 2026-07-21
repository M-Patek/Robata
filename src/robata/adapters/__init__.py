"""Concrete local adapters for optional ingestion dependencies."""

from robata.adapters.in_memory_task_queue import InMemoryTaskQueue
from robata.adapters.local_artifact_registry import (
    LocalArtifactRegistry,
    allocate_local_artifact_id,
    deterministic_local_artifact_id,
)
from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.adapters.sqlite_barrier import SQLiteBarrierStorage, SQLiteBarrierStorageError
from robata.adapters.sqlite_inference_evidence import (
    SQLiteInferenceEvidenceLedger,
    SQLiteInferenceEvidenceLedgerError,
)
from robata.adapters.sqlite_primary_completion import SQLitePrimaryCompletionRepository

# Optional: PyAV-based adapters (requires av package)
try:
    from robata.adapters.parallel_frame_materializer import ParallelPyAvFrameMaterializer
    from robata.adapters.pyav_frame_materializer import PyAvFrameMaterializer
    from robata.adapters.pyav_mp4_exporter import (
        EXPORT_CONFIG,
        EXPORT_PROFILE_ID,
        EXPORT_PROFILE_VERSION,
        EXPORTER_NAME,
        EXPORTER_VERSION,
        PyAvH264Mp4Exporter,
    )
except ImportError:
    ParallelPyAvFrameMaterializer = None  # type: ignore[misc,assignment]
    PyAvFrameMaterializer = None  # type: ignore[misc,assignment]
    EXPORT_CONFIG = None  # type: ignore[misc,assignment]
    EXPORT_PROFILE_ID = None  # type: ignore[misc,assignment]
    EXPORT_PROFILE_VERSION = None  # type: ignore[misc,assignment]
    EXPORTER_NAME = None  # type: ignore[misc,assignment]
    EXPORTER_VERSION = None  # type: ignore[misc,assignment]
    PyAvH264Mp4Exporter = None  # type: ignore[misc,assignment]

# Optional: video export (requires av package)
try:
    from robata.adapters.parallel_video_export import ParallelSixCameraVideoExportService
except ImportError:
    ParallelSixCameraVideoExportService = None  # type: ignore[misc,assignment]

# Optional: decoder probe (requires av package)
try:
    from robata.adapters.pyav_decoder import PyAvH264DecoderProbe
except ImportError:
    PyAvH264DecoderProbe = None  # type: ignore[misc,assignment]

# Optional: MCAP inspector (requires mcap package)
try:
    from robata.adapters.mcap_inspector import OfficialMcapInspector
except ImportError:
    OfficialMcapInspector = None  # type: ignore[misc,assignment]

__all__ = [
    "EXPORTER_NAME",
    "EXPORTER_VERSION",
    "EXPORT_CONFIG",
    "EXPORT_PROFILE_ID",
    "EXPORT_PROFILE_VERSION",
    "InMemoryTaskQueue",
    "LocalArtifactRegistry",
    "LocalLogicalNodeRegistry",
    "OfficialMcapInspector",
    "ParallelPyAvFrameMaterializer",
    "ParallelSixCameraVideoExportService",
    "PyAvFrameMaterializer",
    "PyAvH264DecoderProbe",
    "PyAvH264Mp4Exporter",
    "SQLiteBarrierStorage",
    "SQLiteBarrierStorageError",
    "SQLiteInferenceEvidenceLedger",
    "SQLiteInferenceEvidenceLedgerError",
    "SQLitePrimaryCompletionRepository",
    "allocate_local_artifact_id",
    "deterministic_local_artifact_id",
]
