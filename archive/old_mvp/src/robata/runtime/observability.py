"""Dependency-free metrics and correlation-ID structured logging.

These primitives are local contracts, not a Prometheus server or a logging transport. They keep
metric names/labels deterministic and make it possible to attach queue/worker telemetry without
introducing network dependencies or provider SDKs.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import re
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_CORRELATION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "robata_correlation_id",
    default=None,
)


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """One immutable metric value with canonical labels."""

    name: str
    kind: str
    value: float
    labels: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "labels": dict(self.labels),
        }


class MetricsRegistry:
    """Thread-safe in-process counter/gauge/histogram registry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(
            float
        )
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: defaultdict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = (
            defaultdict(list)
        )

    def increment(
        self,
        name: str,
        amount: float = 1.0,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> float:
        key = _key(name, labels)
        amount = _finite_number(amount, "counter increment")
        if amount < 0:
            raise ValueError("counter increment must be nonnegative")
        with self._lock:
            self._counters[key] += float(amount)
            return self._counters[key]

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        key = _key(name, labels)
        with self._lock:
            self._gauges[key] = _finite_number(value, "gauge value")

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        key = _key(name, labels)
        with self._lock:
            self._histograms[key].append(_finite_number(value, "histogram observation"))

    def snapshot(self) -> tuple[MetricPoint, ...]:
        with self._lock:
            points: list[MetricPoint] = []
            points.extend(
                MetricPoint(name, "counter", value, labels)
                for (name, labels), value in self._counters.items()
            )
            points.extend(
                MetricPoint(name, "gauge", value, labels)
                for (name, labels), value in self._gauges.items()
            )
            for (name, labels), values in self._histograms.items():
                points.extend(
                    (
                        MetricPoint(f"{name}_count", "histogram", float(len(values)), labels),
                        MetricPoint(f"{name}_sum", "histogram", sum(values), labels),
                        MetricPoint(f"{name}_max", "histogram", max(values), labels),
                    )
                )
            return tuple(sorted(points, key=lambda point: (point.name, point.labels, point.kind)))

    def as_dict(self) -> list[dict[str, object]]:
        return [point.as_dict() for point in self.snapshot()]

    def render_prometheus(self) -> str:
        """Render a deterministic text exposition without opening a network endpoint."""

        lines: list[str] = []
        for point in self.snapshot():
            labels = ""
            if point.labels:
                labels = (
                    "{" + ",".join(f'{key}="{_escape(value)}"' for key, value in point.labels) + "}"
                )
            lines.append(f"{point.name}{labels} {point.value:g}")
        return "\n".join(lines) + ("\n" if lines else "")


def _key(name: str, labels: Mapping[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
    _validate_metric_name(name)
    canonical_items: list[tuple[str, str]] = []
    for key, value in (labels or {}).items():
        if not isinstance(key, str) or not _LABEL_NAME.fullmatch(key):
            raise ValueError(f"invalid metric label name: {key}")
        canonical_items.append((key, str(value)))
    canonical = tuple(sorted(canonical_items))
    return name, canonical


def _finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _validate_metric_name(name: str) -> None:
    if not isinstance(name, str) or not _METRIC_NAME.fullmatch(name):
        raise ValueError(f"invalid metric name: {name!r}")


def _escape(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)[1:-1]


def new_correlation_id(seed: str) -> str:
    """Return a deterministic UUID correlation ID for a stable seed."""

    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("correlation seed must be non-empty")
    return str(uuid5(NAMESPACE_URL, f"robata-correlation/{seed}"))


def current_correlation_id() -> str | None:
    return _CORRELATION_ID.get()


@contextmanager
def correlation_scope(correlation_id: str) -> Iterator[str]:
    """Set a correlation ID for structured logs in the current context."""

    if not isinstance(correlation_id, str) or not correlation_id.strip():
        raise ValueError("correlation_id must be non-empty")
    token = _CORRELATION_ID.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _CORRELATION_ID.reset(token)


def build_log_event(
    event: str,
    *,
    correlation_id: str | None = None,
    fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a JSON-compatible structured event with deterministic field ordering."""

    if not isinstance(event, str) or not event.strip():
        raise ValueError("event must be non-empty")
    resolved = correlation_id if correlation_id is not None else current_correlation_id()
    if resolved is None or not resolved.strip():
        raise ValueError("a correlation_id is required")
    payload: dict[str, object] = {
        "correlation_id": resolved,
        "event": event,
    }
    if fields:
        payload.update(fields)
    return dict(sorted(payload.items(), key=lambda item: item[0]))


class StructuredLogger:
    """Emit structured JSON records through a standard-library logger."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def emit(
        self,
        level: int,
        event: str,
        *,
        correlation_id: str | None = None,
        fields: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        payload = build_log_event(event, correlation_id=correlation_id, fields=fields)
        self._logger.log(level, json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return payload


__all__ = [
    "MetricPoint",
    "MetricsRegistry",
    "StructuredLogger",
    "build_log_event",
    "correlation_scope",
    "current_correlation_id",
    "new_correlation_id",
]
