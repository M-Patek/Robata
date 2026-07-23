"""Fail-open runtime observation primitives for local performance evidence.

The values emitted here are operational observations.  They are deliberately independent of
canonical run, artifact, and completion identities: callers may enable or disable an observer
without changing domain inputs or outputs.
"""

from __future__ import annotations

import asyncio
import ctypes
import math
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar, Token
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Protocol

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import StrictModel

type RuntimeAttributeValue = str | int | float | bool
_NonEmptyName = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class RuntimeSpanStatus(StrEnum):
    """Terminal status for one observed span."""

    OK = "OK"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class RuntimeResourceStatus(StrEnum):
    """Availability of one process-resource observation."""

    AVAILABLE = "AVAILABLE"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class RuntimeAttribute(StrictModel):
    """One stable, low-cardinality span or counter attribute."""

    name: _NonEmptyName
    value: RuntimeAttributeValue

    @model_validator(mode="after")
    def validate_finite_float(self) -> RuntimeAttribute:
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("floating-point runtime attributes must be finite")
        return self


class RuntimeResourceMeasurement(StrictModel):
    """One resource value with absence represented explicitly rather than as zero."""

    status: RuntimeResourceStatus
    value: _NonNegativeInt | None = None
    error_type: _NonEmptyName | None = None

    @model_validator(mode="after")
    def validate_status_shape(self) -> RuntimeResourceMeasurement:
        if self.status is RuntimeResourceStatus.AVAILABLE:
            if self.value is None or self.error_type is not None:
                raise ValueError("AVAILABLE resources require value and forbid error_type")
        elif self.status is RuntimeResourceStatus.UNSUPPORTED:
            if self.value is not None or self.error_type is not None:
                raise ValueError("UNSUPPORTED resources forbid value and error_type")
        elif self.value is not None or self.error_type is None:
            raise ValueError("ERROR resources require error_type and forbid value")
        return self


class ProcessResourceSample(StrictModel):
    """Absolute process resource counters captured at one instant."""

    rss_bytes: RuntimeResourceMeasurement
    read_bytes: RuntimeResourceMeasurement
    write_bytes: RuntimeResourceMeasurement


class RuntimeResourceSnapshot(StrictModel):
    """End RSS and process-I/O deltas over the recorder lifetime."""

    rss_bytes: RuntimeResourceMeasurement
    read_bytes_delta: RuntimeResourceMeasurement
    write_bytes_delta: RuntimeResourceMeasurement


class RuntimeSpanSnapshot(StrictModel):
    """Immutable terminal observation for one span."""

    sequence: _PositiveInt
    parent_sequence: _PositiveInt | None = None
    name: _NonEmptyName
    attributes: tuple[RuntimeAttribute, ...] = ()
    status: RuntimeSpanStatus
    error_type: _NonEmptyName | None = None
    started_offset_ns: _NonNegativeInt
    ended_offset_ns: _NonNegativeInt
    elapsed_ns: _NonNegativeInt

    @model_validator(mode="after")
    def validate_span_shape(self) -> RuntimeSpanSnapshot:
        if self.ended_offset_ns < self.started_offset_ns:
            raise ValueError("ended_offset_ns must not precede started_offset_ns")
        if self.elapsed_ns != self.ended_offset_ns - self.started_offset_ns:
            raise ValueError("elapsed_ns must equal the span offset difference")
        if self.status is RuntimeSpanStatus.OK:
            if self.error_type is not None:
                raise ValueError("OK spans forbid error_type")
        elif self.error_type is None:
            raise ValueError("ERROR and CANCELLED spans require error_type")
        if tuple(sorted(self.attributes, key=lambda item: item.name)) != self.attributes:
            raise ValueError("span attributes must be ordered by name")
        return self


class RuntimeCounterSnapshot(StrictModel):
    """Immutable aggregate for one counter and exact attribute set."""

    name: _NonEmptyName
    attributes: tuple[RuntimeAttribute, ...] = ()
    value: _PositiveInt

    @model_validator(mode="after")
    def validate_attribute_order(self) -> RuntimeCounterSnapshot:
        if tuple(sorted(self.attributes, key=lambda item: item.name)) != self.attributes:
            raise ValueError("counter attributes must be ordered by name")
        return self


class RuntimeProfileSnapshot(StrictModel):
    """Frozen local runtime profile ordered independently of completion timing."""

    version: str = "runtime-profile-v1"
    elapsed_ns: _NonNegativeInt
    process_cpu_ns: _NonNegativeInt
    resources: RuntimeResourceSnapshot
    spans: tuple[RuntimeSpanSnapshot, ...] = ()
    counters: tuple[RuntimeCounterSnapshot, ...] = ()

    @model_validator(mode="after")
    def validate_order(self) -> RuntimeProfileSnapshot:
        if tuple(sorted(self.spans, key=lambda item: item.sequence)) != self.spans:
            raise ValueError("spans must be ordered by sequence")
        counter_order = tuple(
            sorted(
                self.counters,
                key=lambda item: (
                    item.name,
                    tuple(
                        (attribute.name, _attribute_sort_value(attribute.value))
                        for attribute in item.attributes
                    ),
                ),
            )
        )
        if counter_order != self.counters:
            raise ValueError("counters must be ordered by name and attributes")
        return self


class RuntimeObserver(Protocol):
    """Minimal observer port accepted by canonical runtime components."""

    def begin_span(
        self,
        name: str,
        attributes: Mapping[str, RuntimeAttributeValue] | None = None,
    ) -> object:
        """Begin a span and return an opaque token."""

    def end_span(
        self,
        token: object,
        *,
        status: RuntimeSpanStatus = RuntimeSpanStatus.OK,
        error_type: str | None = None,
    ) -> None:
        """Finish a previously returned opaque token."""

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        attributes: Mapping[str, RuntimeAttributeValue] | None = None,
    ) -> None:
        """Add a positive integer to one counter."""


class _ActiveSpan:
    __slots__ = (
        "attributes",
        "context_token",
        "name",
        "parent_sequence",
        "recorder_identity",
        "sequence",
        "started_ns",
    )

    def __init__(
        self,
        *,
        recorder_identity: object,
        sequence: int,
        parent_sequence: int | None,
        name: str,
        attributes: tuple[RuntimeAttribute, ...],
        started_ns: int,
        context_token: Token[tuple[int, ...]],
    ) -> None:
        self.recorder_identity = recorder_identity
        self.sequence = sequence
        self.parent_sequence = parent_sequence
        self.name = name
        self.attributes = attributes
        self.started_ns = started_ns
        self.context_token = context_token


class RuntimeProfileRecorder:
    """Thread-safe span/counter recorder with task-local nesting.

    ``snapshot`` is a terminal operation.  Its first call freezes and caches the result, so
    repeated serialization cannot extend the measured interval or mutate its values.
    """

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        process_cpu_clock_ns: Callable[[], int] = time.process_time_ns,
        resource_sampler: Callable[[], ProcessResourceSample] = lambda: sample_process_resources(),
    ) -> None:
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if not callable(process_cpu_clock_ns):
            raise TypeError("process_cpu_clock_ns must be callable")
        if not callable(resource_sampler):
            raise TypeError("resource_sampler must be callable")

        self._clock_ns = clock_ns
        self._process_cpu_clock_ns = process_cpu_clock_ns
        self._resource_sampler = resource_sampler
        self._lock = threading.RLock()
        self._recorder_identity = object()
        self._span_stack: ContextVar[tuple[int, ...]] = ContextVar(
            f"robata_runtime_span_stack_{id(self)}",
            default=(),
        )
        self._next_sequence = 1
        self._active_spans: dict[int, _ActiveSpan] = {}
        self._finished_spans: list[RuntimeSpanSnapshot] = []
        self._counters: dict[
            tuple[str, tuple[tuple[str, str, RuntimeAttributeValue], ...]],
            tuple[tuple[RuntimeAttribute, ...], int],
        ] = {}
        self._snapshot: RuntimeProfileSnapshot | None = None
        self._started_ns = _safe_clock_value(self._clock_ns)
        self._started_cpu_ns = _safe_clock_value(self._process_cpu_clock_ns)
        self._started_resources = _safe_resource_sample(self._resource_sampler)

    def begin_span(
        self,
        name: str,
        attributes: Mapping[str, RuntimeAttributeValue] | None = None,
    ) -> object:
        normalized_name = _validate_name(name, field="span name")
        normalized_attributes = _normalize_attributes(attributes)
        with self._lock:
            self._require_open()
            sequence = self._next_sequence
            self._next_sequence += 1
            stack = self._span_stack.get()
            context_token = self._span_stack.set((*stack, sequence))
            active = _ActiveSpan(
                recorder_identity=self._recorder_identity,
                sequence=sequence,
                parent_sequence=stack[-1] if stack else None,
                name=normalized_name,
                attributes=normalized_attributes,
                started_ns=_safe_clock_value(self._clock_ns),
                context_token=context_token,
            )
            self._active_spans[sequence] = active
            return active

    def end_span(
        self,
        token: object,
        *,
        status: RuntimeSpanStatus = RuntimeSpanStatus.OK,
        error_type: str | None = None,
    ) -> None:
        if not isinstance(status, RuntimeSpanStatus):
            raise TypeError("status must be a RuntimeSpanStatus")
        normalized_error_type = _validate_terminal_error(status, error_type)
        if (
            not isinstance(token, _ActiveSpan)
            or token.recorder_identity is not self._recorder_identity
        ):
            raise ValueError("span token does not belong to this recorder")

        ended_ns = _safe_clock_value(self._clock_ns)
        with suppress(LookupError, RuntimeError, ValueError):
            self._span_stack.reset(token.context_token)

        with self._lock:
            active = self._active_spans.pop(token.sequence, None)
            if active is None:
                if self._snapshot is not None:
                    return
                raise ValueError("span token has already ended")
            started_offset_ns = max(0, active.started_ns - self._started_ns)
            ended_offset_ns = max(started_offset_ns, ended_ns - self._started_ns)
            self._finished_spans.append(
                RuntimeSpanSnapshot(
                    sequence=active.sequence,
                    parent_sequence=active.parent_sequence,
                    name=active.name,
                    attributes=active.attributes,
                    status=status,
                    error_type=normalized_error_type,
                    started_offset_ns=started_offset_ns,
                    ended_offset_ns=ended_offset_ns,
                    elapsed_ns=ended_offset_ns - started_offset_ns,
                )
            )

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        attributes: Mapping[str, RuntimeAttributeValue] | None = None,
    ) -> None:
        normalized_name = _validate_name(name, field="counter name")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("counter value must be an integer")
        if value <= 0:
            raise ValueError("counter value must be positive")
        normalized_attributes = _normalize_attributes(attributes)
        key_attributes = tuple(
            (attribute.name, type(attribute.value).__name__, attribute.value)
            for attribute in normalized_attributes
        )
        key = (normalized_name, key_attributes)
        with self._lock:
            self._require_open()
            existing = self._counters.get(key)
            total = value if existing is None else existing[1] + value
            self._counters[key] = (normalized_attributes, total)

    def snapshot(self) -> RuntimeProfileSnapshot:
        """Freeze and return an idempotent immutable snapshot."""

        with self._lock:
            if self._snapshot is not None:
                return self._snapshot

            ended_ns = _safe_clock_value(self._clock_ns)
            ended_cpu_ns = _safe_clock_value(self._process_cpu_clock_ns)
            ended_resources = _safe_resource_sample(self._resource_sampler)
            ended_offset_ns = max(0, ended_ns - self._started_ns)

            # Snapshotting with live spans is caller misuse, but observation must remain
            # fail-open.  Preserve those spans explicitly instead of silently dropping them.
            for sequence in sorted(self._active_spans):
                active = self._active_spans[sequence]
                started_offset_ns = max(0, active.started_ns - self._started_ns)
                span_end_offset_ns = max(started_offset_ns, ended_offset_ns)
                self._finished_spans.append(
                    RuntimeSpanSnapshot(
                        sequence=active.sequence,
                        parent_sequence=active.parent_sequence,
                        name=active.name,
                        attributes=active.attributes,
                        status=RuntimeSpanStatus.ERROR,
                        error_type="RuntimeProfileSnapshotWhileSpanActive",
                        started_offset_ns=started_offset_ns,
                        ended_offset_ns=span_end_offset_ns,
                        elapsed_ns=span_end_offset_ns - started_offset_ns,
                    )
                )
            self._active_spans.clear()

            counters = tuple(
                RuntimeCounterSnapshot(name=name, attributes=attributes, value=value)
                for (name, _key_attributes), (attributes, value) in sorted(
                    self._counters.items(),
                    key=lambda item: item[0],
                )
            )
            self._snapshot = RuntimeProfileSnapshot(
                elapsed_ns=ended_offset_ns,
                process_cpu_ns=max(0, ended_cpu_ns - self._started_cpu_ns),
                resources=_resource_delta(self._started_resources, ended_resources),
                spans=tuple(sorted(self._finished_spans, key=lambda item: item.sequence)),
                counters=counters,
            )
            return self._snapshot

    def _require_open(self) -> None:
        if self._snapshot is not None:
            raise RuntimeError("runtime profile is already frozen")


class NoOpRuntimeObserver:
    """Stateless observer for dependency-injection sites that require an object."""

    __slots__ = ()

    def begin_span(
        self,
        name: str,
        attributes: Mapping[str, RuntimeAttributeValue] | None = None,
    ) -> object:
        del name, attributes
        return _NOOP_TOKEN

    def end_span(
        self,
        token: object,
        *,
        status: RuntimeSpanStatus = RuntimeSpanStatus.OK,
        error_type: str | None = None,
    ) -> None:
        del token, status, error_type

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        attributes: Mapping[str, RuntimeAttributeValue] | None = None,
    ) -> None:
        del name, value, attributes


_NOOP_TOKEN = object()
NOOP_RUNTIME_OBSERVER: RuntimeObserver = NoOpRuntimeObserver()


@contextmanager
def runtime_span(
    observer: RuntimeObserver | None,
    name: str,
    attributes: Mapping[str, RuntimeAttributeValue] | None = None,
) -> Iterator[None]:
    """Observe a block without allowing observer failures to affect that block."""

    token: object | None = None
    if observer is not None:
        try:
            token = observer.begin_span(name, attributes)
        except Exception:
            token = None
    try:
        yield
    except asyncio.CancelledError as exc:
        _try_end_span(
            observer,
            token,
            status=RuntimeSpanStatus.CANCELLED,
            error_type=type(exc).__name__,
        )
        raise
    except BaseException as exc:
        _try_end_span(
            observer,
            token,
            status=RuntimeSpanStatus.ERROR,
            error_type=type(exc).__name__,
        )
        raise
    else:
        _try_end_span(observer, token, status=RuntimeSpanStatus.OK, error_type=None)


def runtime_increment(
    observer: RuntimeObserver | None,
    name: str,
    value: int = 1,
    attributes: Mapping[str, RuntimeAttributeValue] | None = None,
) -> None:
    """Increment an observer counter without affecting the caller on observer failure."""

    if observer is None:
        return
    try:
        observer.increment_counter(name, value, attributes)
    except Exception:
        return


def sample_process_resources() -> ProcessResourceSample:
    """Sample process RSS and storage I/O using only platform standard-library facilities."""

    if sys.platform == "win32":
        return _sample_windows_process_resources()
    if sys.platform.startswith("linux"):
        return _sample_linux_process_resources()
    unsupported = _unsupported_resource()
    return ProcessResourceSample(
        rss_bytes=unsupported,
        read_bytes=unsupported,
        write_bytes=unsupported,
    )


def _try_end_span(
    observer: RuntimeObserver | None,
    token: object | None,
    *,
    status: RuntimeSpanStatus,
    error_type: str | None,
) -> None:
    if observer is None or token is None:
        return
    try:
        observer.end_span(token, status=status, error_type=error_type)
    except Exception:
        return


def _validate_name(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or len(value) > 256:
        raise ValueError(f"{field} must contain between 1 and 256 characters")
    return value


def _normalize_attributes(
    attributes: Mapping[str, RuntimeAttributeValue] | None,
) -> tuple[RuntimeAttribute, ...]:
    if attributes is None:
        return ()
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping")
    normalized: list[RuntimeAttribute] = []
    for name, value in attributes.items():
        _validate_name(name, field="attribute name")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("floating-point runtime attributes must be finite")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError("runtime attribute values must be scalar")
        normalized.append(RuntimeAttribute(name=name, value=value))
    return tuple(sorted(normalized, key=lambda item: item.name))


def _validate_terminal_error(
    status: RuntimeSpanStatus,
    error_type: str | None,
) -> str | None:
    if status is RuntimeSpanStatus.OK:
        if error_type is not None:
            raise ValueError("OK spans forbid error_type")
        return None
    if error_type is None:
        raise ValueError("ERROR and CANCELLED spans require error_type")
    return _validate_name(error_type, field="error_type")


def _attribute_sort_value(value: RuntimeAttributeValue) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _safe_clock_value(clock: Callable[[], int]) -> int:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("nanosecond clocks must return integers")
    return value


def _safe_resource_sample(
    sampler: Callable[[], ProcessResourceSample],
) -> ProcessResourceSample:
    try:
        sample = sampler()
        if not isinstance(sample, ProcessResourceSample):
            raise TypeError("resource sampler must return ProcessResourceSample")
        return sample
    except Exception as exc:
        failure = _error_resource(type(exc).__name__)
        return ProcessResourceSample(
            rss_bytes=failure,
            read_bytes=failure,
            write_bytes=failure,
        )


def _resource_delta(
    started: ProcessResourceSample,
    ended: ProcessResourceSample,
) -> RuntimeResourceSnapshot:
    return RuntimeResourceSnapshot(
        rss_bytes=ended.rss_bytes,
        read_bytes_delta=_counter_delta(started.read_bytes, ended.read_bytes),
        write_bytes_delta=_counter_delta(started.write_bytes, ended.write_bytes),
    )


def _counter_delta(
    started: RuntimeResourceMeasurement,
    ended: RuntimeResourceMeasurement,
) -> RuntimeResourceMeasurement:
    if ended.status is RuntimeResourceStatus.ERROR:
        return ended
    if started.status is RuntimeResourceStatus.ERROR:
        return started
    if (
        started.status is RuntimeResourceStatus.UNSUPPORTED
        or ended.status is RuntimeResourceStatus.UNSUPPORTED
    ):
        return _unsupported_resource()
    if started.value is None or ended.value is None or ended.value < started.value:
        return _error_resource("NonMonotonicResourceCounter")
    return _available_resource(ended.value - started.value)


def _available_resource(value: int) -> RuntimeResourceMeasurement:
    return RuntimeResourceMeasurement(
        status=RuntimeResourceStatus.AVAILABLE,
        value=value,
    )


def _unsupported_resource() -> RuntimeResourceMeasurement:
    return RuntimeResourceMeasurement(status=RuntimeResourceStatus.UNSUPPORTED)


def _error_resource(error_type: str) -> RuntimeResourceMeasurement:
    return RuntimeResourceMeasurement(
        status=RuntimeResourceStatus.ERROR,
        error_type=error_type,
    )


def _sample_linux_process_resources() -> ProcessResourceSample:
    rss = _probe_linux_rss()
    read_bytes, write_bytes = _probe_linux_io()
    return ProcessResourceSample(
        rss_bytes=rss,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
    )


def _probe_linux_rss() -> RuntimeResourceMeasurement:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) != 3 or fields[2] != "kB":
                    raise RuntimeError("unexpected VmRSS representation")
                return _available_resource(int(fields[1]) * 1024)
        raise RuntimeError("VmRSS is absent")
    except Exception as exc:
        return _error_resource(type(exc).__name__)


def _probe_linux_io() -> tuple[RuntimeResourceMeasurement, RuntimeResourceMeasurement]:
    try:
        counters: dict[str, int] = {}
        for line in Path("/proc/self/io").read_text(encoding="ascii").splitlines():
            name, separator, value = line.partition(":")
            if separator:
                counters[name] = int(value.strip())
        return (
            _available_resource(counters["read_bytes"]),
            _available_resource(counters["write_bytes"]),
        )
    except Exception as exc:
        failure = _error_resource(type(exc).__name__)
        return failure, failure


class _WindowsProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _WindowsIoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


def _sample_windows_process_resources() -> ProcessResourceSample:
    try:
        kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        process = kernel32.GetCurrentProcess()
    except Exception as exc:
        failure = _error_resource(type(exc).__name__)
        return ProcessResourceSample(
            rss_bytes=failure,
            read_bytes=failure,
            write_bytes=failure,
        )

    rss = _probe_windows_rss(process)
    read_bytes, write_bytes = _probe_windows_io(kernel32, process)
    return ProcessResourceSample(
        rss_bytes=rss,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
    )


def _probe_windows_rss(process: object) -> RuntimeResourceMeasurement:
    try:
        psapi: Any = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = _WindowsProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        succeeded = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not succeeded:
            raise OSError(ctypes.get_last_error())
        return _available_resource(int(counters.WorkingSetSize))
    except Exception as exc:
        return _error_resource(type(exc).__name__)


def _probe_windows_io(
    kernel32: Any,
    process: object,
) -> tuple[RuntimeResourceMeasurement, RuntimeResourceMeasurement]:
    try:
        kernel32.GetProcessIoCounters.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsIoCounters),
        ]
        kernel32.GetProcessIoCounters.restype = ctypes.c_int
        counters = _WindowsIoCounters()
        succeeded = kernel32.GetProcessIoCounters(process, ctypes.byref(counters))
        if not succeeded:
            raise OSError(ctypes.get_last_error())
        return (
            _available_resource(int(counters.ReadTransferCount)),
            _available_resource(int(counters.WriteTransferCount)),
        )
    except Exception as exc:
        failure = _error_resource(type(exc).__name__)
        return failure, failure


__all__ = [
    "NOOP_RUNTIME_OBSERVER",
    "NoOpRuntimeObserver",
    "ProcessResourceSample",
    "RuntimeAttribute",
    "RuntimeAttributeValue",
    "RuntimeCounterSnapshot",
    "RuntimeObserver",
    "RuntimeProfileRecorder",
    "RuntimeProfileSnapshot",
    "RuntimeResourceMeasurement",
    "RuntimeResourceSnapshot",
    "RuntimeResourceStatus",
    "RuntimeSpanSnapshot",
    "RuntimeSpanStatus",
    "runtime_increment",
    "runtime_span",
    "sample_process_resources",
]
