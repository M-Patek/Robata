"""Concrete local adapters for optional ingestion dependencies."""

from robata.adapters.fake_vision_model import (
    FAKE_MODEL_NAME,
    FAKE_MODEL_VERSION,
    FAKE_PROVIDER,
    DeterministicFakeVisionModelAdapter,
    FakeVisionModelAdapter,
)
from robata.adapters.in_memory_task_queue import InMemoryTaskQueue
from robata.adapters.local_artifact_registry import (
    LocalArtifactRegistry,
    allocate_local_artifact_id,
    deterministic_local_artifact_id,
)
from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.adapters.mcap_inspector import OfficialMcapInspector
from robata.adapters.parallel_frame_materializer import ParallelPyAvFrameMaterializer
from robata.adapters.parallel_video_export import ParallelSixCameraVideoExportService
from robata.adapters.pyav_decoder import PyAvH264DecoderProbe
from robata.adapters.pyav_frame_materializer import PyAvFrameMaterializer
from robata.adapters.pyav_mp4_exporter import (
    EXPORT_CONFIG,
    EXPORT_PROFILE_ID,
    EXPORT_PROFILE_VERSION,
    EXPORTER_NAME,
    EXPORTER_VERSION,
    PyAvH264Mp4Exporter,
)

__all__ = [
    "EXPORTER_NAME",
    "EXPORTER_VERSION",
    "EXPORT_CONFIG",
    "EXPORT_PROFILE_ID",
    "EXPORT_PROFILE_VERSION",
    "FAKE_MODEL_NAME",
    "FAKE_MODEL_VERSION",
    "FAKE_PROVIDER",
    "DeterministicFakeVisionModelAdapter",
    "FakeVisionModelAdapter",
    "InMemoryTaskQueue",
    "LocalArtifactRegistry",
    "LocalLogicalNodeRegistry",
    "LocalVisionModelAdapter",
    "LocalVisionModelAdapterError",
    "OfficialMcapInspector",
    "OptionalDependencyUnavailable",
    "ParallelPyAvFrameMaterializer",
    "ParallelSixCameraVideoExportService",
    "PyAvFrameMaterializer",
    "PyAvH264DecoderProbe",
    "PyAvH264Mp4Exporter",
    "TransformersVisionModelAdapter",
    "VisionRunner",
    "allocate_local_artifact_id",
    "deterministic_local_artifact_id",
]
from robata.adapters.local_vision_model import (
    LocalVisionModelAdapter,
    LocalVisionModelAdapterError,
    OptionalDependencyUnavailable,
    TransformersVisionModelAdapter,
    VisionRunner,
)
