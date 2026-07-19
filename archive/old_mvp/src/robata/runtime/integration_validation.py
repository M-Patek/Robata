"""Offline stress and worker-chain validation for the local requirements path."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robata.adapters.in_memory_task_queue import InMemoryTaskQueue
from robata.annotation import AnnotationPipeline
from robata.frame_cache import FramePayload, SharedFrameCache
from robata.ports.task_queue import PipelineTask, TaskId
from robata.qa import ClipMark, QAClassifier, QAIssue
from robata.search import ClipSearchIndex
from robata.worker import PipelineWorker, WorkerConfig, WorkerRunStatus


@dataclass(frozen=True, slots=True)
class FrameCacheStressReport:
    video_count: int
    callers: int
    decode_attempts: int
    cache_hits: int
    cache_misses: int
    manifests_equal: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_count": self.video_count,
            "callers": self.callers,
            "decode_attempts": self.decode_attempts,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "manifests_equal": self.manifests_equal,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class WorkerIntegrationReport:
    task_count: int
    completed_count: int
    statuses: tuple[str, ...]
    qa_status: str
    annotation_draft_count: int
    search_hit_count: int
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "completed_count": self.completed_count,
            "statuses": self.statuses,
            "qa_status": self.qa_status,
            "annotation_draft_count": self.annotation_draft_count,
            "search_hit_count": self.search_hit_count,
            "passed": self.passed,
        }


def run_frame_cache_stress(
    *,
    video_count: int = 8,
    callers: int = 16,
    frames_per_video: int = 4,
    root: Path | None = None,
) -> FrameCacheStressReport:
    if video_count <= 0 or callers <= 0 or frames_per_video <= 0:
        raise ValueError("video_count, callers, and frames_per_video must be positive")
    cache_root = (
        Path(root)
        if root is not None
        else Path("tmp") / f"robata-frame-stress-{threading.get_ident()}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = SharedFrameCache(cache_root)
    all_manifests: list[Any] = []
    lock = threading.Lock()

    def call(video_index: int) -> None:
        video_id = f"stress-{video_index:03d}"
        def decoder() -> list[FramePayload]:
            return [
                FramePayload(timestamp_sec=frame / 2.0, data=f"{video_id}-{frame}".encode())
                for frame in range(frames_per_video)
            ]
        result = cache.feed_once(video_id, f"local://{video_id}", decoder)
        with lock:
            all_manifests.append(result.manifest)

    with ThreadPoolExecutor(max_workers=min(callers, video_count * 2)) as pool:
        futures = [pool.submit(call, index % video_count) for index in range(callers)]
        for future in futures:
            future.result()
    stats = cache.stats()
    per_video_manifests: dict[str, set[str]] = {}
    for manifest in all_manifests:
        per_video_manifests.setdefault(manifest.video_id, set()).add(manifest.cache_key)
    manifests_equal = len(per_video_manifests) == video_count and all(
        len(keys) == 1 for keys in per_video_manifests.values()
    )
    report = FrameCacheStressReport(
        video_count=video_count,
        callers=callers,
        decode_attempts=stats.decode_attempts,
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
        manifests_equal=manifests_equal,
        passed=manifests_equal and stats.decode_attempts == video_count,
    )
    return report


def run_worker_requirements_integration() -> WorkerIntegrationReport:
    classifier = QAClassifier()
    qa_pass = classifier.assess("worker-pass", 12.0)
    qa_warning = classifier.assess(
        "worker-warning",
        12.0,
        [ClipMark(start_sec=2.0, end_sec=3.0, issue=QAIssue.HAIR_BLOCKING_VIEW, confidence=0.8)],
    )
    qa_fail = classifier.assess(
        "worker-fail",
        12.0,
        [ClipMark(start_sec=0.0, end_sec=12.0, issue=QAIssue.BLACK_SCREEN, confidence=1.0)],
    )
    annotations = AnnotationPipeline().run((qa_pass, qa_warning, qa_fail))
    index = ClipSearchIndex(annotations.drafts)
    query_hits = index.search("interact object")
    queue = InMemoryTaskQueue()
    task_ids = tuple(
        TaskId(value)
        for value in ("worker-qa", "worker-annotation", "worker-search")
    )
    stages = ("qa", "annotation", "search")
    for task_id, stage in zip(task_ids, stages, strict=True):
        queue.enqueue(
            PipelineTask(
                task_id=task_id,
                recording_id="worker-recording",
                stage=stage,
                payload=json.dumps({"stage": stage}).encode(),
                created_at=datetime.now(UTC),
                max_retries=0,
            )
        )

    def handler(task: PipelineTask) -> bytes:
        if task.stage == "qa":
            return qa_pass.status.value.encode()
        if task.stage == "annotation":
            return str(annotations.draft_count).encode()
        if task.stage == "search":
            return str(len(query_hits)).encode()
        raise ValueError(task.stage)

    worker = PipelineWorker(
        queue,
        handler,
        config=WorkerConfig(
            worker_id="requirements-worker",
            lease_duration_seconds=10,
            heartbeat_interval_seconds=1,
            poll_interval_seconds=0.01,
        ),
        sleep=lambda _seconds: None,
    )
    runs = tuple(worker.run_once() for _ in stages)
    statuses = tuple(run.status.value for run in runs)
    completed = sum(run.status is WorkerRunStatus.COMPLETED for run in runs)
    passed = (
        completed == len(stages)
        and qa_pass.status.value == "pass"
        and annotations.draft_count > 0
        and len(query_hits) > 0
        and "worker-fail" in annotations.skipped_fail_video_ids
    )
    return WorkerIntegrationReport(
        task_count=len(stages),
        completed_count=completed,
        statuses=statuses,
        qa_status=qa_pass.status.value,
        annotation_draft_count=annotations.draft_count,
        search_hit_count=len(query_hits),
        passed=passed,
    )


__all__ = [
    "FrameCacheStressReport",
    "WorkerIntegrationReport",
    "run_frame_cache_stress",
    "run_worker_requirements_integration",
]
