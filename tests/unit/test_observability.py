from __future__ import annotations

import logging

import pytest

from robata.runtime.observability import (
    MetricsRegistry,
    StructuredLogger,
    build_log_event,
    correlation_scope,
    current_correlation_id,
    new_correlation_id,
)


def test_metrics_registry_canonical_snapshot_and_prometheus() -> None:
    registry = MetricsRegistry()
    registry.increment("worker_tasks_completed", labels={"stage": "QA", "worker": "w1"})
    registry.increment("worker_tasks_completed", 2, labels={"worker": "w1", "stage": "QA"})
    registry.set_gauge("queue_depth", 3, labels={"queue": "local"})
    registry.observe("pipeline_latency_ms", 10, labels={"stage": "QA"})
    registry.observe("pipeline_latency_ms", 20, labels={"stage": "QA"})

    points = registry.snapshot()
    assert [point.name for point in points] == [
        "pipeline_latency_ms_count",
        "pipeline_latency_ms_max",
        "pipeline_latency_ms_sum",
        "queue_depth",
        "worker_tasks_completed",
    ]
    assert registry.render_prometheus() == (
        'pipeline_latency_ms_count{stage="QA"} 2\n'
        'pipeline_latency_ms_max{stage="QA"} 20\n'
        'pipeline_latency_ms_sum{stage="QA"} 30\n'
        'queue_depth{queue="local"} 3\n'
        'worker_tasks_completed{stage="QA",worker="w1"} 3\n'
    )


def test_metrics_reject_invalid_names_and_values() -> None:
    registry = MetricsRegistry()
    with pytest.raises(ValueError, match="invalid metric name"):
        registry.increment("bad-name")
    with pytest.raises(ValueError, match="nonnegative"):
        registry.increment("valid_counter", -1)
    with pytest.raises(TypeError, match="numeric"):
        registry.set_gauge("queue_depth", "3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        registry.observe("latency", float("nan"))
    with pytest.raises(ValueError, match="label name"):
        registry.increment("valid_counter", labels={"bad-label": "x"})


def test_correlation_scope_and_structured_logger(caplog: pytest.LogCaptureFixture) -> None:
    correlation_id = new_correlation_id("run-1")
    assert correlation_id == new_correlation_id("run-1")
    assert current_correlation_id() is None
    with correlation_scope(correlation_id):
        assert current_correlation_id() == correlation_id
        assert build_log_event("worker.started", fields={"task_id": "task-1"}) == {
            "correlation_id": correlation_id,
            "event": "worker.started",
            "task_id": "task-1",
        }
        logger = StructuredLogger(logging.getLogger("robata.test"))
        with caplog.at_level(logging.INFO, logger="robata.test"):
            logger.emit(logging.INFO, "worker.started", fields={"task_id": "task-1"})
    assert current_correlation_id() is None
    assert '"correlation_id":"' + correlation_id in caplog.messages[0]


def test_log_event_requires_correlation_id() -> None:
    with pytest.raises(ValueError, match="correlation_id"):
        build_log_event("event")
