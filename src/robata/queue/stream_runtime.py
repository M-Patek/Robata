"""Bounded local runtime lanes for durable stream work.

The SQLite work scheduler remains the authority for every work-item state.
This module deliberately treats its three in-memory lanes as disposable
notifications: losing a process-local queue entry can delay work, but cannot
lose a planned item because :meth:`recover` repopulates ingress from the
durable ledger.  This is the local reference behavior for a later broker
adapter, not a new wire contract or source of truth.

``ingress`` accepts ready durable work, ``provider`` owns the fenced provider
attempt, and ``publish`` performs an idempotent external/delivery side effect
before recording scheduler success.  Each buffered lane is bounded and a full
downstream lane rejects new ingress so pressure reaches callers instead of
turning into an unbounded in-memory backlog.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
from time import monotonic, sleep
from typing import Protocol

from robata.queue.models import (
    TERMINAL_WORK_STATES,
    WorkItem,
    WorkItemState,
    WorkLease,
    WorkLeaseClaim,
)
from robata.queue.stage import Stage

_POLL_SECONDS = 0.02
DEFAULT_OPTIONAL_WORK_SHEDDING_ACTIONS = (
    "STOP_OPTIONAL_DEEP",
    "REDUCE_GPT_SHADOW",
    "DEFER_EMBEDDING",
    "AUTO_SCALE",
    "THROTTLE_LEDGER",
    "EMERGENCY_SAMPLING",
    "QUARANTINE_CANDIDATE",
)


class DurableStreamWorkScheduler(Protocol):
    """The authoritative scheduler surface needed by runtime queue lanes.

    Keeping this as a protocol avoids making :mod:`robata.queue` import a
    concrete SQLite adapter at package-import time.  The local implementation
    is :class:`robata.adapters.sqlite_work_scheduler.SQLiteWorkScheduler`.
    """

    def get(self, work_item_id: str) -> WorkItem:
        """Return one durable work snapshot."""
        ...

    def items_for_run(self, run_id: str) -> tuple[WorkItem, ...]:
        """Return every durable item belonging to one recording run."""
        ...

    def ready_for_run(self, run_id: str, *, limit: int) -> tuple[WorkItem, ...]:
        """Return a bounded priority-ordered READY slice for one run."""
        ...

    def reconcile(self) -> int:
        """Recover expired leases, due retries, and dependency readiness."""
        ...

    def claim_and_start(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        work_item_id: str | None = None,
    ) -> WorkLeaseClaim | None:
        """Atomically start an exact ready item under a fenced lease."""
        ...

    def fail(
        self,
        lease: WorkLease,
        *,
        error_code: str,
        retryable: bool,
        error_detail: str | None = None,
        retry_delay_seconds: int = 0,
    ) -> WorkItem:
        """Record a retryable or terminal attempt outcome."""
        ...

    def heartbeat(
        self,
        lease: WorkLease,
        lease_duration_seconds: int,
    ) -> WorkLease:
        """Renew one active work lease and return its current capability."""
        ...

    def succeed(self, lease: WorkLease) -> WorkItem:
        """Record authoritative success under a live fence."""
        ...

    def cancel(
        self,
        work_item_id: str,
        *,
        reason_code: str = "CANCELLED_BY_REQUEST",
        reason_detail: str | None = None,
    ) -> WorkItem:
        """Cancel an item and revoke any active lease."""
        ...


class StreamQueueLane(StrEnum):
    """The bounded, non-authoritative local runtime lanes."""

    INGRESS = "INGRESS"
    PROVIDER = "PROVIDER"
    PUBLISH = "PUBLISH"


class StreamQueueAdmissionStatus(StrEnum):
    """Explicit admission result; rejection never changes durable work."""

    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"
    CLOSED = "CLOSED"
    NOT_READY = "NOT_READY"
    WRONG_RUN = "WRONG_RUN"
    INGRESS_FULL = "INGRESS_FULL"
    PROVIDER_FULL = "PROVIDER_FULL"
    PUBLISH_FULL = "PUBLISH_FULL"
    SHED_OPTIONAL = "SHED_OPTIONAL"


@dataclass(frozen=True, slots=True)
class StreamQueueAdmission:
    """One ingress admission outcome."""

    work_item_id: str
    status: StreamQueueAdmissionStatus

    @property
    def admitted(self) -> bool:
        """Whether the durable work ID entered the bounded ingress lane."""

        return self.status is StreamQueueAdmissionStatus.ACCEPTED


@dataclass(frozen=True, slots=True)
class StreamQueueLaneSnapshot:
    """Operational counters for one bounded runtime lane."""

    capacity: int
    queued: int
    active: int
    admitted: int
    forwarded: int
    completed: int
    failed: int
    cancelled: int
    rejected: int
    maximum_queued: int
    shed_optional: int = 0
    backpressure_waits: int = 0

    @property
    def optional_work_shed(self) -> int:
        return self.shed_optional


@dataclass(frozen=True, slots=True)
class BoundedStreamWorkQueuesSnapshot:
    """Stable queue health snapshot for local profiles and tests."""

    closed: bool
    scheduled_work_count: int
    ingress: StreamQueueLaneSnapshot
    provider: StreamQueueLaneSnapshot
    publish: StreamQueueLaneSnapshot
    last_error: str | None
    optional_work_offered: int = 0
    optional_work_admitted: int = 0
    optional_work_shed: int = 0
    backpressure_waits: int = 0
    recovery_count: int = 0
    recovery_admitted: int = 0
    recovery_shed: int = 0
    backlog_count: int = 0
    backlog_peak: int = 0
    backlog_end: int = 0
    lease_renewals: int = 0
    lease_renewal_errors: int = 0
    optional_work_shedding_actions: tuple[str, ...] = DEFAULT_OPTIONAL_WORK_SHEDDING_ACTIONS

    @property
    def optional_offered(self) -> int:
        return self.optional_work_offered

    @property
    def optional_admitted(self) -> int:
        return self.optional_work_admitted

    @property
    def optional_shed(self) -> int:
        return self.optional_work_shed

    @property
    def backpressure_events(self) -> int:
        return self.backpressure_waits

    @property
    def recovery_runs(self) -> int:
        return self.recovery_count

    @property
    def backlog(self) -> int:
        return self.backlog_count


@dataclass(frozen=True, slots=True)
class BoundedStreamWorkQueuesConfig:
    """Runtime-only queue and worker limits for one recording-affine run."""

    run_id: str
    ingress_capacity: int = 64
    provider_capacity: int = 64
    publish_capacity: int = 64
    provider_worker_count: int = 1
    publish_worker_count: int = 1
    lease_duration_seconds: int = 60
    retry_delay_seconds: int = 0
    recovery_poll_seconds: float = 0.1
    optional_shed_watermark: float = 1.0
    optional_stages: tuple[Stage, ...] = ()
    optional_work_predicate: Callable[[WorkItem], bool] | None = None
    optional_work_shedding_actions: tuple[str, ...] = DEFAULT_OPTIONAL_WORK_SHEDDING_ACTIONS

    def __post_init__(self) -> None:
        _require_nonempty(self.run_id, "run_id")
        for field in (
            "ingress_capacity",
            "provider_capacity",
            "publish_capacity",
            "provider_worker_count",
            "publish_worker_count",
            "lease_duration_seconds",
        ):
            _require_positive_int(getattr(self, field), field)
        _require_nonnegative_int(self.retry_delay_seconds, "retry_delay_seconds")
        if (
            isinstance(self.recovery_poll_seconds, bool)
            or not isinstance(self.recovery_poll_seconds, (int, float))
            or self.recovery_poll_seconds <= 0
        ):
            raise ValueError("recovery_poll_seconds must be a positive number")
        if (
            isinstance(self.optional_shed_watermark, bool)
            or not isinstance(self.optional_shed_watermark, (int, float))
            or self.optional_shed_watermark <= 0
            or self.optional_shed_watermark > 1
        ):
            raise ValueError("optional_shed_watermark must be in (0, 1]")
        if not isinstance(self.optional_stages, tuple):
            raise TypeError("optional_stages must be a tuple")
        if any(not isinstance(stage, Stage) for stage in self.optional_stages):
            raise TypeError("optional_stages must contain Stage values")
        if len(set(self.optional_stages)) != len(self.optional_stages):
            raise ValueError("optional_stages must be unique")
        if self.optional_work_predicate is not None and not callable(self.optional_work_predicate):
            raise TypeError("optional_work_predicate must be callable or None")
        if not self.optional_work_shedding_actions:
            raise ValueError("optional_work_shedding_actions must not be empty")
        if self.optional_work_shedding_actions[0] != "STOP_OPTIONAL_DEEP":
            raise ValueError("optional_work_shedding_actions must begin with STOP_OPTIONAL_DEEP")
        if len(set(self.optional_work_shedding_actions)) != len(
            self.optional_work_shedding_actions
        ):
            raise ValueError("optional_work_shedding_actions must be unique")
        if any(
            not isinstance(action, str) or not action.strip()
            for action in self.optional_work_shedding_actions
        ):
            raise ValueError("optional_work_shedding_actions must contain non-empty strings")


class StreamQueueRetryableError(RuntimeError):
    """A callback error that should return its fenced work to retry wait."""

    def __init__(
        self,
        error_code: str,
        *,
        detail: str | None = None,
        retry_delay_seconds: int | None = None,
    ) -> None:
        self.error_code = _require_nonempty(error_code, "error_code")
        self.detail = None if detail is None else _require_nonempty(detail, "detail")
        self.retry_delay_seconds = (
            None
            if retry_delay_seconds is None
            else _require_nonnegative_int(retry_delay_seconds, "retry_delay_seconds")
        )
        super().__init__(self.detail or self.error_code)


class StreamQueuePermanentError(RuntimeError):
    """A callback error that must terminalize the durable work item."""

    def __init__(self, error_code: str, *, detail: str | None = None) -> None:
        self.error_code = _require_nonempty(error_code, "error_code")
        self.detail = None if detail is None else _require_nonempty(detail, "detail")
        super().__init__(self.detail or self.error_code)


ProviderExecutor = Callable[[WorkLeaseClaim], object]
Publisher = Callable[[WorkLeaseClaim, object], None]


@dataclass(slots=True)
class _LaneCounters:
    admitted: int = 0
    forwarded: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    rejected: int = 0
    active: int = 0
    maximum_queued: int = 0
    shed_optional: int = 0
    backpressure_waits: int = 0


@dataclass(slots=True)
class _QueueTelemetryCounters:
    optional_work_offered: int = 0
    optional_work_admitted: int = 0
    optional_work_shed: int = 0
    backpressure_waits: int = 0
    recovery_count: int = 0
    recovery_admitted: int = 0
    recovery_shed: int = 0
    backlog_count: int = 0
    lease_renewals: int = 0
    lease_renewal_errors: int = 0


class _LeaseHeartbeat:
    """Own and renew one fenced lease while local callbacks are active/queued.

    The scheduler keeps the durable lease authority.  This helper merely
    refreshes that capability in-process so an intentionally slow provider or
    publisher cannot be recovered and executed a second time by this service.
    A process crash stops the heartbeat, leaving ordinary lease-expiry replay
    intact.
    """

    def __init__(
        self,
        *,
        scheduler: DurableStreamWorkScheduler,
        lease: WorkLease,
        lease_duration_seconds: int,
        on_renewal: Callable[[], None] | None = None,
        on_error: Callable[[], None] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._lease_duration_seconds = lease_duration_seconds
        self._on_renewal = on_renewal
        self._on_error = on_error
        self._lock = RLock()
        self._lease = lease
        self._error: Exception | None = None
        self._stop = Event()
        self._thread = Thread(
            target=self._run,
            name=f"stream-work-heartbeat-{lease.work_item_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def lease(self) -> WorkLease:
        with self._lock:
            return self._lease

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def stop(self) -> WorkLease:
        """Stop renewal and return the newest lease before a terminal write."""

        self._stop.set()
        self._thread.join()
        return self.lease

    def _run(self) -> None:
        interval = _heartbeat_interval(self._lease_duration_seconds)
        while not self._stop.wait(interval):
            with self._lock:
                lease = self._lease
            try:
                renewed = self._scheduler.heartbeat(
                    lease,
                    self._lease_duration_seconds,
                )
            except Exception as error:
                with self._lock:
                    self._error = error
                if self._on_error is not None:
                    with suppress(Exception):
                        self._on_error()
                self._stop.set()
                return
            with self._lock:
                self._lease = renewed
            if self._on_renewal is not None:
                with suppress(Exception):
                    self._on_renewal()


@dataclass(frozen=True, slots=True)
class _PublishTask:
    claim: WorkLeaseClaim
    provider_result: object
    heartbeat: _LeaseHeartbeat


class BoundedStreamWorkQueues:
    """Three bounded lanes that execute and recover one recording's work.

    ``provider_executor`` is called only after an atomic durable
    ``claim_and_start``.  ``publisher`` must be idempotent for the work item's
    durable identity: a crash after its side effect but before ``succeed`` is
    intentionally replayed.  The queue then calls ``succeed`` only after that
    publisher returns successfully.

    Callback failures are retryable by default so transient local/provider
    errors never silently discard durable work.  A callback can explicitly
    raise :class:`StreamQueuePermanentError` when retrying is not valid.
    """

    def __init__(
        self,
        *,
        scheduler: DurableStreamWorkScheduler,
        config: BoundedStreamWorkQueuesConfig,
        provider_executor: ProviderExecutor,
        publisher: Publisher,
        worker_id_prefix: str = "local-stream-runtime",
    ) -> None:
        if not _is_scheduler(scheduler):
            raise TypeError("scheduler does not implement the durable stream-work surface")
        if not isinstance(config, BoundedStreamWorkQueuesConfig):
            raise TypeError("config must be BoundedStreamWorkQueuesConfig")
        if not callable(provider_executor):
            raise TypeError("provider_executor must be callable")
        if not callable(publisher):
            raise TypeError("publisher must be callable")
        self._scheduler = scheduler
        self._config = config
        self._provider_executor = provider_executor
        self._publisher = publisher
        self._worker_id_prefix = _require_nonempty(worker_id_prefix, "worker_id_prefix")
        self._ingress: Queue[str] = Queue(maxsize=config.ingress_capacity)
        self._provider: Queue[str] = Queue(maxsize=config.provider_capacity)
        self._publish: Queue[_PublishTask] = Queue(maxsize=config.publish_capacity)
        self._lock = RLock()
        self._closed = False
        self._stop = Event()
        self._scheduled_work_ids: set[str] = set()
        self._cancel_requested_ids: set[str] = set()
        # Every claimed lease remains tracked through provider execution,
        # publish backlog, and the external publisher. A non-graceful local
        # stop releases these renewers so another runtime can replay safely.
        self._heartbeats: set[_LeaseHeartbeat] = set()
        self._counters = {lane: _LaneCounters() for lane in StreamQueueLane}
        self._telemetry = _QueueTelemetryCounters()
        self._optional_work_ids: set[str] = set()
        self._shed_optional_ids: set[str] = set()
        self._backlog_peak = 0
        self._last_error: str | None = None

        workers: list[Thread] = [
            Thread(
                target=self._run_ingress,
                name=f"{self._worker_id_prefix}-ingress",
                daemon=True,
            )
        ]
        workers.extend(
            Thread(
                target=self._run_provider,
                args=(ordinal,),
                name=f"{self._worker_id_prefix}-provider-{ordinal}",
                daemon=True,
            )
            for ordinal in range(config.provider_worker_count)
        )
        workers.extend(
            Thread(
                target=self._run_publish,
                args=(ordinal,),
                name=f"{self._worker_id_prefix}-publish-{ordinal}",
                daemon=True,
            )
            for ordinal in range(config.publish_worker_count)
        )
        self._workers = tuple(workers)
        self._recovery_worker = Thread(
            target=self._run_recovery,
            name=f"{self._worker_id_prefix}-recovery",
            daemon=True,
        )
        for worker in self._workers:
            worker.start()
        self._recovery_worker.start()

    @property
    def config(self) -> BoundedStreamWorkQueuesConfig:
        """Return the immutable runtime-only configuration."""

        return self._config

    @property
    def snapshot(self) -> BoundedStreamWorkQueuesSnapshot:
        """Return a small bounded-lane snapshot without scanning durable work."""

        with self._lock:
            backlog_end = len(self._scheduled_work_ids) + len(self._shed_optional_ids)
            self._backlog_peak = max(self._backlog_peak, backlog_end)
            return BoundedStreamWorkQueuesSnapshot(
                closed=self._closed,
                scheduled_work_count=len(self._scheduled_work_ids),
                ingress=self._lane_snapshot(StreamQueueLane.INGRESS, self._ingress),
                provider=self._lane_snapshot(StreamQueueLane.PROVIDER, self._provider),
                publish=self._lane_snapshot(StreamQueueLane.PUBLISH, self._publish),
                last_error=self._last_error,
                optional_work_offered=self._telemetry.optional_work_offered,
                optional_work_admitted=self._telemetry.optional_work_admitted,
                optional_work_shed=self._telemetry.optional_work_shed,
                backpressure_waits=self._telemetry.backpressure_waits,
                recovery_count=self._telemetry.recovery_count,
                recovery_admitted=self._telemetry.recovery_admitted,
                recovery_shed=self._telemetry.recovery_shed,
                backlog_count=backlog_end,
                backlog_peak=self._backlog_peak,
                backlog_end=backlog_end,
                lease_renewals=self._telemetry.lease_renewals,
                lease_renewal_errors=self._telemetry.lease_renewal_errors,
                optional_work_shedding_actions=self._config.optional_work_shedding_actions,
            )

    def admit(
        self,
        work_item_id: str,
        *,
        optional: bool = False,
    ) -> StreamQueueAdmission:
        """Admit one currently READY durable item into bounded ingress.

        The method never plans, mutates, or drops durable state.  A full lane
        is a caller-visible backpressure outcome; ``recover`` can later admit
        the same READY item once capacity returns.
        """

        checked_id = _require_nonempty(work_item_id, "work_item_id")
        if not isinstance(optional, bool):
            raise TypeError("optional must be a boolean")
        with self._lock:
            if optional:
                self._optional_work_ids.add(checked_id)
                if checked_id not in self._shed_optional_ids:
                    self._telemetry.optional_work_offered += 1
            if checked_id in self._scheduled_work_ids:
                return StreamQueueAdmission(checked_id, StreamQueueAdmissionStatus.DUPLICATE)
            if self._closed:
                return StreamQueueAdmission(checked_id, StreamQueueAdmissionStatus.CLOSED)
        item = self._scheduler.get(checked_id)
        # A runtime instance owns one recording-affine state root.  Never let
        # an explicit caller route a different run through its worker lanes.
        if item.run_id != self._config.run_id:
            with self._lock:
                return self._reject_ingress(
                    checked_id,
                    StreamQueueAdmissionStatus.WRONG_RUN,
                )
        return self._admit_ready(item, external=True)

    def admit_optional(self, work_item_id: str) -> StreamQueueAdmission:
        """Admit work marked optional; pressure may leave it durably READY."""

        return self.admit(work_item_id, optional=True)

    def recover(self) -> int:
        """Recover expired/retryable work and fill ingress only to its bound.

        This is a restart and maintenance path, never a source-message hot
        loop.  It deliberately rereads the one recording's durable run so a
        process crash or local queue cancellation leaves exactly replayable
        work rather than a lost in-memory message.
        """

        if self._stop.is_set():
            return 0
        with self._lock:
            self._telemetry.recovery_count += 1
        self._scheduler.reconcile()
        ready = self._scheduler.ready_for_run(
            self._config.run_id,
            limit=self._recovery_ready_limit(),
        )
        admitted = 0
        for item in ready:
            outcome = self._admit_ready(item, external=False)
            if outcome.admitted:
                admitted += 1
                with self._lock:
                    self._telemetry.recovery_admitted += 1
                continue
            if outcome.status is StreamQueueAdmissionStatus.SHED_OPTIONAL:
                with self._lock:
                    self._telemetry.recovery_shed += 1
                continue
            if outcome.status in {
                StreamQueueAdmissionStatus.INGRESS_FULL,
                StreamQueueAdmissionStatus.PROVIDER_FULL,
                StreamQueueAdmissionStatus.PUBLISH_FULL,
            }:
                # Preserve scheduler ordering: do not leapfrog a blocked
                # higher-priority item with a lower-priority ready item.
                break
        return admitted

    def cancel(
        self,
        work_item_id: str,
        *,
        reason_code: str = "CANCELLED_BY_REQUEST",
        reason_detail: str | None = None,
    ) -> WorkItem:
        """Durably cancel an admitted item and revoke its active lease.

        Python cannot safely interrupt arbitrary provider callbacks.  The
        durable cancellation fence prevents their eventual result from being
        published, while the callback itself is allowed to return normally.
        """

        checked_id = _require_nonempty(work_item_id, "work_item_id")
        item = self._scheduler.get(checked_id)
        if item.run_id != self._config.run_id:
            raise ValueError(
                "work item does not belong to this recording-affine runtime "
                f"(work_item_id={checked_id!r}, run_id={item.run_id!r})"
            )
        with self._lock:
            tracked = checked_id in self._scheduled_work_ids
            if tracked:
                self._cancel_requested_ids.add(checked_id)
        checked_reason_detail = (
            None if reason_detail is None else _require_nonempty(reason_detail, "reason_detail")
        )
        try:
            cancelled = self._scheduler.cancel(
                checked_id,
                reason_code=_require_nonempty(reason_code, "reason_code"),
                reason_detail=checked_reason_detail,
            )
        except Exception:
            if tracked:
                with self._lock:
                    self._cancel_requested_ids.discard(checked_id)
            raise
        with self._lock:
            self._scheduled_work_ids.discard(checked_id)
            self._optional_work_ids.discard(checked_id)
            self._shed_optional_ids.discard(checked_id)
            if not tracked:
                self._cancel_requested_ids.discard(checked_id)
        return cancelled

    def drain(self, *, timeout: float | None = None) -> bool:
        """Drive due retries/recovery until this run has no nonterminal work.

        ``False`` means the timeout elapsed; the durable scheduler remains
        replayable and callers may retry ``drain`` or restart the service.
        """

        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0
        ):
            raise ValueError("timeout must be nonnegative or None")
        deadline = None if timeout is None else monotonic() + float(timeout)
        while True:
            self.recover()
            if self._is_drained():
                return True
            if deadline is not None and monotonic() >= deadline:
                return False
            sleep(min(_POLL_SECONDS, self._config.recovery_poll_seconds))

    def close(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        """Stop workers, optionally durably cancelling admitted queued work.

        A non-cancelling early close does not discard authority state: a
        subsequent instance's :meth:`recover` sees READY work immediately or
        recovers an expired running lease.  A graceful waiting close drains
        before stopping workers.
        """

        if not isinstance(wait, bool) or not isinstance(cancel_pending, bool):
            raise TypeError("wait and cancel_pending must be booleans")
        with self._lock:
            already_closed = self._closed
            self._closed = True
        if already_closed:
            if wait:
                self._join_workers()
            return
        if cancel_pending:
            # Mark cancellation before stopping workers. A callback that is
            # already active may still return, but it must observe the fence
            # and skip publish/succeed. Stopping first also prevents recovery
            # or forwarding loops from adding notifications while pending
            # lanes are discarded.
            work_item_ids = self._mark_cancel_pending()
            with self._lock:
                self._stop.set()
            self._cancel_admitted_work(work_item_ids)
            self._discard_pending_notifications(work_item_ids)
        elif wait:
            self.drain()
            with self._lock:
                self._stop.set()
        else:
            with self._lock:
                self._stop.set()
        self._stop_all_heartbeats()
        if wait:
            self._join_workers()

    def __enter__(self) -> BoundedStreamWorkQueues:
        return self

    def __exit__(self, exc_type: object, _exc: object, _tb: object) -> None:
        self.close(cancel_pending=exc_type is not None)

    def _admit_ready(self, item: WorkItem, *, external: bool) -> StreamQueueAdmission:
        work_item_id = item.work_item_id
        with self._lock:
            if work_item_id in self._scheduled_work_ids:
                return StreamQueueAdmission(work_item_id, StreamQueueAdmissionStatus.DUPLICATE)
            if self._closed and external:
                return StreamQueueAdmission(work_item_id, StreamQueueAdmissionStatus.CLOSED)
            if self._stop.is_set():
                return StreamQueueAdmission(work_item_id, StreamQueueAdmissionStatus.CLOSED)
            if item.state is not WorkItemState.READY:
                self._optional_work_ids.discard(work_item_id)
                self._shed_optional_ids.discard(work_item_id)
                return StreamQueueAdmission(work_item_id, StreamQueueAdmissionStatus.NOT_READY)
            optional = self._is_optional(item)
            if optional and work_item_id not in self._optional_work_ids:
                self._optional_work_ids.add(work_item_id)
                self._telemetry.optional_work_offered += 1
            pressure = self._optional_pressure()
            if optional and pressure:
                first_shed = work_item_id not in self._shed_optional_ids
                self._shed_optional_ids.add(work_item_id)
                if first_shed:
                    self._telemetry.optional_work_shed += 1
                    self._backlog_peak = max(
                        self._backlog_peak,
                        len(self._scheduled_work_ids) + len(self._shed_optional_ids),
                    )
                    self._counters[StreamQueueLane.INGRESS].shed_optional += 1
                    self._observe("optional_work_shed")
                return StreamQueueAdmission(
                    work_item_id,
                    StreamQueueAdmissionStatus.SHED_OPTIONAL,
                )

            self._shed_optional_ids.discard(work_item_id)
            # Downstream pressure is intentionally visible at ingress rather
            # than allowing all three buffers to fill before callers slow down.
            if self._publish.full():
                return self._reject_ingress(work_item_id, StreamQueueAdmissionStatus.PUBLISH_FULL)
            if self._provider.full():
                return self._reject_ingress(work_item_id, StreamQueueAdmissionStatus.PROVIDER_FULL)
            try:
                self._ingress.put_nowait(work_item_id)
            except Full:
                return self._reject_ingress(work_item_id, StreamQueueAdmissionStatus.INGRESS_FULL)
            self._scheduled_work_ids.add(work_item_id)
            self._backlog_peak = max(
                self._backlog_peak,
                len(self._scheduled_work_ids) + len(self._shed_optional_ids),
            )
            if optional:
                self._telemetry.optional_work_admitted += 1
            counters = self._counters[StreamQueueLane.INGRESS]
            counters.admitted += 1
            counters.maximum_queued = max(counters.maximum_queued, self._ingress.qsize())
            return StreamQueueAdmission(work_item_id, StreamQueueAdmissionStatus.ACCEPTED)

    def _reject_ingress(
        self,
        work_item_id: str,
        status: StreamQueueAdmissionStatus,
    ) -> StreamQueueAdmission:
        self._counters[StreamQueueLane.INGRESS].rejected += 1
        if status in {
            StreamQueueAdmissionStatus.INGRESS_FULL,
            StreamQueueAdmissionStatus.PROVIDER_FULL,
            StreamQueueAdmissionStatus.PUBLISH_FULL,
        }:
            self._record_backpressure_wait(StreamQueueLane.INGRESS)
        return StreamQueueAdmission(work_item_id, status)

    def _optional_pressure(self) -> bool:
        watermark = self._config.optional_shed_watermark
        occupancy = max(
            self._ingress.qsize() / self._ingress.maxsize,
            self._provider.qsize() / self._provider.maxsize,
            self._publish.qsize() / self._publish.maxsize,
        )
        return occupancy >= watermark

    def _is_optional(self, item: WorkItem) -> bool:
        if item.work_item_id in self._optional_work_ids:
            return True
        if item.stage in self._config.optional_stages:
            return True
        predicate = self._config.optional_work_predicate
        if predicate is None:
            return False
        try:
            return bool(predicate(item))
        except Exception as error:
            self._record_error(error)
            return False

    def _observe(self, name: str) -> None:
        observer = getattr(self._config, "runtime_observer", None)
        if observer is None:
            return
        try:
            observer.increment_counter(
                f"stream_queue_{name}",
                1,
                {"run_id": self._config.run_id},
            )
        except Exception:
            return

    def _record_backpressure_wait(self, lane: StreamQueueLane) -> None:
        with self._lock:
            self._telemetry.backpressure_waits += 1
            self._counters[lane].backpressure_waits += 1
        self._observe("backpressure_wait")

    def _record_lease_renewal(self) -> None:
        with self._lock:
            self._telemetry.lease_renewals += 1

    def _record_lease_renewal_error(self) -> None:
        with self._lock:
            self._telemetry.lease_renewal_errors += 1

    def _run_ingress(self) -> None:
        while not self._stop.is_set():
            try:
                work_item_id = self._ingress.get(timeout=_POLL_SECONDS)
            except Empty:
                continue
            self._begin(StreamQueueLane.INGRESS)
            try:
                if self._cancelled(work_item_id):
                    self._cancelled_finish(work_item_id, StreamQueueLane.INGRESS)
                elif self._put_provider(work_item_id):
                    self._forward(StreamQueueLane.INGRESS)
                else:
                    # Shutdown/cancellation removed only a notification. The
                    # scheduler row remains ready or becomes a durable cancel.
                    self._finish_scheduled(work_item_id)
            finally:
                self._end(StreamQueueLane.INGRESS)
                self._ingress.task_done()

    def _run_provider(self, ordinal: int) -> None:
        worker_id = f"{self._worker_id_prefix}:provider:{ordinal}"
        while not self._stop.is_set():
            try:
                work_item_id = self._provider.get(timeout=_POLL_SECONDS)
            except Empty:
                continue
            self._begin(StreamQueueLane.PROVIDER)
            heartbeat: _LeaseHeartbeat | None = None
            try:
                if self._cancelled(work_item_id):
                    self._cancelled_finish(work_item_id, StreamQueueLane.PROVIDER)
                    continue
                claim = self._scheduler.claim_and_start(
                    worker_id,
                    self._config.lease_duration_seconds,
                    work_item_id=work_item_id,
                )
                if claim is None:
                    # A competing worker/cancellation/retry transition owns
                    # the durable row. Recovery will reconsider it if READY.
                    self._finish_scheduled(work_item_id)
                    continue
                heartbeat = _LeaseHeartbeat(
                    scheduler=self._scheduler,
                    lease=claim.lease,
                    lease_duration_seconds=self._config.lease_duration_seconds,
                    on_renewal=self._record_lease_renewal,
                    on_error=self._record_lease_renewal_error,
                )
                self._track_heartbeat(heartbeat)
                try:
                    result = self._provider_executor(claim)
                except Exception as error:
                    latest_claim = _claim_with_lease(claim, self._stop_heartbeat(heartbeat))
                    self._fail_claim(latest_claim, error, StreamQueueLane.PROVIDER)
                    self._finish_scheduled(work_item_id)
                    continue
                if heartbeat.error is not None:
                    if self._cancelled(work_item_id):
                        self._cancelled_finish(work_item_id, StreamQueueLane.PROVIDER)
                    else:
                        self._record_error(heartbeat.error)
                        # A lost fence/storage health must never publish an
                        # unconfirmed provider result. Recovery owns replay.
                        self._finish_scheduled(work_item_id)
                    continue
                if self._cancelled(work_item_id):
                    self._cancelled_finish(work_item_id, StreamQueueLane.PROVIDER)
                    continue
                publish_claim = _claim_with_lease(claim, heartbeat.lease)
                task = _PublishTask(
                    claim=publish_claim,
                    provider_result=result,
                    heartbeat=heartbeat,
                )
                if self._put_publish(task):
                    # Publish now owns continuous lease renewal through its
                    # own backlog and external callback.
                    heartbeat = None
                    self._forward(StreamQueueLane.PROVIDER)
                    continue
                if self._cancelled(work_item_id):
                    self._cancelled_finish(work_item_id, StreamQueueLane.PROVIDER)
                elif heartbeat.error is not None:
                    self._record_error(heartbeat.error)
                    self._finish_scheduled(work_item_id)
                else:
                    latest_claim = _claim_with_lease(claim, self._stop_heartbeat(heartbeat))
                    self._fail_claim(
                        latest_claim,
                        StreamQueueRetryableError("RUNTIME_SHUTDOWN"),
                        StreamQueueLane.PROVIDER,
                    )
                    self._finish_scheduled(work_item_id)
            except Exception as error:
                # Scheduler/storage exceptions must leave authority state for
                # recovery; do not fabricate a terminal result from a worker.
                self._record_error(error)
                self._finish_scheduled(work_item_id)
            finally:
                if heartbeat is not None:
                    self._stop_heartbeat(heartbeat)
                self._end(StreamQueueLane.PROVIDER)
                self._provider.task_done()

    def _run_publish(self, _ordinal: int) -> None:
        while not self._stop.is_set():
            try:
                task = self._publish.get(timeout=_POLL_SECONDS)
            except Empty:
                continue
            work_item_id = task.claim.work_item.work_item_id
            heartbeat = task.heartbeat
            self._begin(StreamQueueLane.PUBLISH)
            try:
                if self._cancelled(work_item_id):
                    self._cancelled_finish(work_item_id, StreamQueueLane.PUBLISH)
                    continue
                if heartbeat.error is not None:
                    if self._cancelled(work_item_id):
                        self._cancelled_finish(work_item_id, StreamQueueLane.PUBLISH)
                    else:
                        self._record_error(heartbeat.error)
                        self._finish_scheduled(work_item_id)
                    continue
                callback_claim = _claim_with_lease(task.claim, heartbeat.lease)
                try:
                    self._publisher(callback_claim, task.provider_result)
                except Exception as error:
                    latest_claim = _claim_with_lease(task.claim, self._stop_heartbeat(heartbeat))
                    self._fail_claim(latest_claim, error, StreamQueueLane.PUBLISH)
                    self._finish_scheduled(work_item_id)
                    continue
                if heartbeat.error is not None:
                    if self._cancelled(work_item_id):
                        self._cancelled_finish(work_item_id, StreamQueueLane.PUBLISH)
                    else:
                        # The publisher may have completed an idempotent side
                        # effect, but the durable fence is no longer known
                        # live. Do not claim success; replay is intentional.
                        self._record_error(heartbeat.error)
                        self._finish_scheduled(work_item_id)
                    continue
                if self._cancelled(work_item_id):
                    self._cancelled_finish(work_item_id, StreamQueueLane.PUBLISH)
                    continue
                latest_lease = self._stop_heartbeat(heartbeat)
                if heartbeat.error is not None:
                    self._record_error(heartbeat.error)
                    self._finish_scheduled(work_item_id)
                    continue
                try:
                    self._scheduler.succeed(latest_lease)
                except Exception as error:
                    # A publisher may have committed its idempotent side effect
                    # before a process/storage fault. Leave the durable running
                    # lease to expire and replay instead of guessing a result.
                    self._record_error(error)
                    self._finish_scheduled(work_item_id)
                    continue
                self._complete(StreamQueueLane.PUBLISH)
                self._finish_scheduled(work_item_id)
            finally:
                self._stop_heartbeat(heartbeat)
                self._end(StreamQueueLane.PUBLISH)
                self._publish.task_done()

    def _run_recovery(self) -> None:
        while not self._stop.wait(self._config.recovery_poll_seconds):
            try:
                self.recover()
            except Exception as error:
                self._record_error(error)

    def _put_provider(self, work_item_id: str) -> bool:
        while not self._stop.is_set():
            try:
                with self._lock:
                    if self._stop.is_set() or self._cancelled(work_item_id):
                        return False
                    self._provider.put_nowait(work_item_id)
            except Full:
                self._record_backpressure_wait(StreamQueueLane.PROVIDER)
                sleep(_POLL_SECONDS)
                continue
            self._enqueued(StreamQueueLane.PROVIDER, self._provider)
            return True
        return False

    def _put_publish(self, task: _PublishTask) -> bool:
        work_item_id = task.claim.work_item.work_item_id
        while not self._stop.is_set():
            try:
                with self._lock:
                    if (
                        self._stop.is_set()
                        or self._cancelled(work_item_id)
                        or task.heartbeat.error is not None
                    ):
                        return False
                    self._publish.put_nowait(task)
            except Full:
                self._record_backpressure_wait(StreamQueueLane.PUBLISH)
                sleep(_POLL_SECONDS)
                continue
            self._enqueued(StreamQueueLane.PUBLISH, self._publish)
            return True
        return False

    def _fail_claim(
        self,
        claim: WorkLeaseClaim,
        error: Exception,
        lane: StreamQueueLane,
    ) -> None:
        retryable = not isinstance(error, StreamQueuePermanentError)
        if isinstance(error, (StreamQueueRetryableError, StreamQueuePermanentError)):
            error_code = error.error_code
            detail = error.detail
        else:
            error_code = f"{lane.value}_CALLBACK_FAILED"
            detail = _exception_detail(error)
        delay = (
            self._config.retry_delay_seconds
            if not isinstance(error, StreamQueueRetryableError) or error.retry_delay_seconds is None
            else error.retry_delay_seconds
        )
        try:
            self._scheduler.fail(
                claim.lease,
                error_code=error_code,
                retryable=retryable,
                error_detail=detail,
                retry_delay_seconds=delay,
            )
        except Exception as failure:
            # Cancellation and an expired/stale fence are benign from this
            # worker's perspective. Other failures remain observable and the
            # authoritative row is intentionally left replayable.
            if not self._cancelled(claim.work_item.work_item_id) and not _is_fence_error(failure):
                self._record_error(failure)
        finally:
            with self._lock:
                self._counters[lane].failed += 1

    def _mark_cancel_pending(self) -> tuple[str, ...]:
        with self._lock:
            # Optional work that was shed is durable READY work too, even
            # though it never entered ``_scheduled_work_ids``.
            work_item_ids = tuple(self._scheduled_work_ids | self._shed_optional_ids)
            self._cancel_requested_ids.update(work_item_ids)
        return work_item_ids

    def _cancel_admitted_work(self, work_item_ids: tuple[str, ...]) -> None:
        for work_item_id in work_item_ids:
            try:
                self._scheduler.cancel(work_item_id, reason_code="RUNTIME_QUEUE_CANCELLED")
            except Exception as error:
                if not _is_fence_error(error):
                    self._record_error(error)

    def _discard_pending_notifications(self, cancelled_ids: tuple[str, ...]) -> None:
        """Drop local notifications after a cancelling shutdown.

        Queue entries are disposable hints; durable cancellation above is the
        authority. ``Queue.task_done`` is still paired for every removed item,
        and publish entries release their heartbeat so a close cannot strand a
        renewal thread.
        """

        cancelled = set(cancelled_ids)
        self._discard_queue(self._ingress, StreamQueueLane.INGRESS, cancelled)
        self._discard_queue(self._provider, StreamQueueLane.PROVIDER, cancelled)
        self._discard_queue(self._publish, StreamQueueLane.PUBLISH, cancelled)
        with self._lock:
            # A notification can be dequeued concurrently with the drain. A
            # worker that owns such an item keeps its cancellation request;
            # ``_finish_scheduled`` clears it when the callback/claim exits.
            self._scheduled_work_ids.difference_update(cancelled)
            self._optional_work_ids.difference_update(cancelled)
            self._shed_optional_ids.difference_update(cancelled)

    def _discard_queue(
        self,
        queue: Queue[str] | Queue[_PublishTask],
        lane: StreamQueueLane,
        cancelled_ids: set[str],
    ) -> None:
        while True:
            try:
                value = queue.get_nowait()
            except Empty:
                return
            try:
                work_item_id = (
                    value if isinstance(value, str) else value.claim.work_item.work_item_id
                )
                if isinstance(value, _PublishTask):
                    # The task no longer has a worker that can stop this
                    # heartbeat in its normal finally block.
                    self._stop_heartbeat(value.heartbeat)
                with self._lock:
                    if work_item_id in cancelled_ids:
                        self._counters[lane].cancelled += 1
                    else:
                        self._counters[lane].rejected += 1
                    self._scheduled_work_ids.discard(work_item_id)
                    self._optional_work_ids.discard(work_item_id)
                    self._shed_optional_ids.discard(work_item_id)
            finally:
                queue.task_done()

    def _cancelled(self, work_item_id: str) -> bool:
        with self._lock:
            return work_item_id in self._cancel_requested_ids

    def _cancelled_finish(self, work_item_id: str, lane: StreamQueueLane) -> None:
        with self._lock:
            self._counters[lane].cancelled += 1
        self._finish_scheduled(work_item_id)

    def _finish_scheduled(self, work_item_id: str) -> None:
        with self._lock:
            self._scheduled_work_ids.discard(work_item_id)
            self._optional_work_ids.discard(work_item_id)
            self._shed_optional_ids.discard(work_item_id)
            self._cancel_requested_ids.discard(work_item_id)

    def _track_heartbeat(self, heartbeat: _LeaseHeartbeat) -> None:
        with self._lock:
            self._heartbeats.add(heartbeat)
            stopping = self._stop.is_set()
        if stopping:
            # A close can race a provider claim. Do not leave a freshly
            # created renewer alive after the service has relinquished work.
            self._stop_heartbeat(heartbeat)

    def _stop_heartbeat(self, heartbeat: _LeaseHeartbeat) -> WorkLease:
        lease = heartbeat.stop()
        with self._lock:
            self._heartbeats.discard(heartbeat)
        return lease

    def _stop_all_heartbeats(self) -> None:
        with self._lock:
            heartbeats = tuple(self._heartbeats)
        for heartbeat in heartbeats:
            self._stop_heartbeat(heartbeat)

    def _begin(self, lane: StreamQueueLane) -> None:
        with self._lock:
            self._counters[lane].active += 1

    def _end(self, lane: StreamQueueLane) -> None:
        with self._lock:
            self._counters[lane].active -= 1

    def _enqueued(self, lane: StreamQueueLane, queue: Queue[str] | Queue[_PublishTask]) -> None:
        with self._lock:
            counters = self._counters[lane]
            counters.admitted += 1
            counters.maximum_queued = max(counters.maximum_queued, queue.qsize())

    def _forward(self, lane: StreamQueueLane) -> None:
        with self._lock:
            self._counters[lane].forwarded += 1

    def _complete(self, lane: StreamQueueLane) -> None:
        with self._lock:
            self._counters[lane].completed += 1

    def _record_error(self, error: Exception) -> None:
        with self._lock:
            self._last_error = _exception_detail(error)

    def _recovery_ready_limit(self) -> int:
        """Bound one durable ready query by all local notification slots."""

        return (
            self._config.ingress_capacity
            + self._config.provider_capacity
            + self._config.publish_capacity
            + self._config.provider_worker_count
            + self._config.publish_worker_count
        )

    def _is_drained(self) -> bool:
        with self._lock:
            runtime_idle = (
                not self._scheduled_work_ids
                and self._ingress.empty()
                and self._provider.empty()
                and self._publish.empty()
                and all(counters.active == 0 for counters in self._counters.values())
            )
        if not runtime_idle:
            return False
        return all(
            item.state in TERMINAL_WORK_STATES
            for item in self._scheduler.items_for_run(self._config.run_id)
        )

    def _lane_snapshot(
        self,
        lane: StreamQueueLane,
        queue: Queue[str] | Queue[_PublishTask],
    ) -> StreamQueueLaneSnapshot:
        counters = self._counters[lane]
        return StreamQueueLaneSnapshot(
            capacity=queue.maxsize,
            queued=queue.qsize(),
            active=counters.active,
            admitted=counters.admitted,
            forwarded=counters.forwarded,
            completed=counters.completed,
            failed=counters.failed,
            cancelled=counters.cancelled,
            rejected=counters.rejected,
            maximum_queued=counters.maximum_queued,
            shed_optional=counters.shed_optional,
            backpressure_waits=counters.backpressure_waits,
        )

    def _join_workers(self) -> None:
        for worker in (*self._workers, self._recovery_worker):
            worker.join()


def _is_scheduler(value: object) -> bool:
    return all(
        callable(getattr(value, method, None))
        for method in (
            "get",
            "items_for_run",
            "ready_for_run",
            "reconcile",
            "claim_and_start",
            "heartbeat",
            "fail",
            "succeed",
            "cancel",
        )
    )


def _is_fence_error(error: Exception) -> bool:
    """Avoid importing concrete adapter exceptions into the queue package."""

    return error.__class__.__name__ in {"WorkFenceError", "WorkStateError"}


def _exception_detail(error: Exception) -> str:
    detail = str(error).strip()
    return detail if detail else error.__class__.__name__


def _claim_with_lease(claim: WorkLeaseClaim, lease: WorkLease) -> WorkLeaseClaim:
    """Refresh callback context after a heartbeat without changing its identity."""

    if lease == claim.lease:
        return claim
    return WorkLeaseClaim(
        work_item=claim.work_item.model_copy(update={"lease_expires_at": lease.lease_expires_at}),
        lease=lease,
    )


def _heartbeat_interval(lease_duration_seconds: int) -> float:
    """Renew well before expiry without turning normal long leases into a hot loop."""

    return min(5.0, max(_POLL_SECONDS, lease_duration_seconds / 3))


def _require_nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


__all__ = [
    "DEFAULT_OPTIONAL_WORK_SHEDDING_ACTIONS",
    "BoundedStreamWorkQueues",
    "BoundedStreamWorkQueuesConfig",
    "BoundedStreamWorkQueuesSnapshot",
    "DurableStreamWorkScheduler",
    "ProviderExecutor",
    "Publisher",
    "StreamQueueAdmission",
    "StreamQueueAdmissionStatus",
    "StreamQueueLane",
    "StreamQueueLaneSnapshot",
    "StreamQueuePermanentError",
    "StreamQueueRetryableError",
]
