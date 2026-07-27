"""Recording-level local canonical service and bounded provider composition.

The one-recording canonical command remains the authority for source binding,
identity, durable evidence, completion, and replay.  This module owns only
runtime topology: bounded recording admission, state-affine local workers, and
a shared provider dispatcher.  Worker/queue configuration never enters a
canonical identity or persisted result payload.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
from typing import TypeVar, cast

from robata.application.canonical.local_composition import (
    CanonicalLocalCompositionError,
    CanonicalLocalCompositionErrorCode,
    CanonicalLocalRunReceipt,
    run_local_canonical_fixture,
)

_T = TypeVar("_T")


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class CanonicalLocalProviderQueueSnapshot:
    """Stable operational counters for one shared provider dispatcher."""

    max_concurrency: int
    queue_capacity: int
    queue_depth: int
    active: int
    admitted: int
    completed: int
    failed: int
    cancelled: int
    rejected: int
    backpressure_waits: int
    max_queue_depth: int
    closed: bool


@dataclass(slots=True)
class _ProviderWork:
    callback: Callable[[], object]
    future: Future[object]


class CanonicalLocalProviderQueue:
    """A bounded, provider-neutral dispatcher shared across recordings.

    ``dispatch`` is the composition-facing asynchronous API.  It executes
    actual adapter calls, rather than entire recording commands, on a bounded
    set of provider workers.  Calls wait at the runtime boundary when the
    queue is full, propagating backpressure to the recording pipeline without
    adding an unbounded provider backlog.  ``submit`` remains useful for
    deterministic queue lifecycle tests and synchronous provider adapters.
    """

    def __init__(self, *, max_concurrency: int, queue_capacity: int) -> None:
        self._max_concurrency = _positive_int(max_concurrency, "max_concurrency")
        self._queue_capacity = _positive_int(queue_capacity, "queue_capacity")
        self._queue: Queue[_ProviderWork] = Queue(maxsize=self._queue_capacity)
        self._lock = RLock()
        self._closed = False
        self._shutdown_requested = Event()
        self._active = 0
        self._admitted = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._rejected = 0
        self._backpressure_waits = 0
        self._max_queue_depth = 0
        self._workers = tuple(
            Thread(
                target=self._worker,
                name=f"robata-provider-worker-{ordinal}",
                daemon=True,
            )
            for ordinal in range(self._max_concurrency)
        )
        for worker in self._workers:
            worker.start()

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def queue_capacity(self) -> int:
        return self._queue_capacity

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def snapshot(self) -> CanonicalLocalProviderQueueSnapshot:
        with self._lock:
            return CanonicalLocalProviderQueueSnapshot(
                max_concurrency=self._max_concurrency,
                queue_capacity=self._queue_capacity,
                queue_depth=self._queue.qsize(),
                active=self._active,
                admitted=self._admitted,
                completed=self._completed,
                failed=self._failed,
                cancelled=self._cancelled,
                rejected=self._rejected,
                backpressure_waits=self._backpressure_waits,
                max_queue_depth=self._max_queue_depth,
                closed=self._closed,
            )

    def submit(
        self,
        callback: Callable[[], object],
        *,
        block: bool = False,
        timeout: float | None = None,
    ) -> Future[object]:
        """Admit one synchronous callback or raise a backpressure error."""

        return self._submit(
            callback,
            block=block,
            timeout=timeout,
            count_full_rejection=True,
        )

    async def dispatch(
        self,
        operation: Callable[[], Awaitable[_T]],
    ) -> _T:
        """Dispatch one async provider operation through the shared bound.

        The caller's event loop remains free while an operation waits for an
        available provider-queue slot.  The local composition has bounded
        recording workers and bounded call-part concurrency, so waiters are
        themselves bounded by the admitted recording workload.
        """

        if not callable(operation):
            raise TypeError("operation must be callable")

        async def invoke_operation() -> _T:
            return await operation()

        def invoke() -> object:
            return asyncio.run(invoke_operation())

        waited = False
        while True:
            try:
                future = self._submit(
                    invoke,
                    block=False,
                    timeout=None,
                    count_full_rejection=False,
                )
            except CanonicalLocalCompositionError:
                if self.closed:
                    raise
                if not waited:
                    with self._lock:
                        self._backpressure_waits += 1
                    waited = True
                # Yielding here turns a full provider queue into upstream
                # backpressure instead of an unbounded application queue.
                await asyncio.sleep(0.001)
            else:
                break

        try:
            return cast(_T, await asyncio.wrap_future(future))
        except asyncio.CancelledError:
            # A queued work item will be observed as cancelled by its provider
            # worker.  Active provider calls are not forcibly interrupted.
            future.cancel()
            raise

    def run(
        self,
        callback: Callable[[], object],
        *,
        timeout: float | None = None,
    ) -> object:
        """Submit a callback and synchronously await its provider result."""

        future = self.submit(callback, block=True, timeout=timeout)
        try:
            return future.result(timeout=timeout)
        except BaseException:
            future.cancel()
            raise

    def cancel_pending(self) -> int:
        """Cancel queued callbacks without interrupting active provider work."""

        cancelled = 0
        retained: list[_ProviderWork] = []
        while True:
            try:
                work = self._queue.get_nowait()
            except Empty:
                break
            try:
                if work.future.cancel():
                    cancelled += 1
                else:
                    retained.append(work)
            finally:
                self._queue.task_done()
        for work in retained:
            self._queue.put(work)
        if cancelled:
            with self._lock:
                self._cancelled += cancelled
        return cancelled

    def drain(self) -> None:
        """Wait until all admitted provider work has terminally settled."""

        self._queue.join()

    def close(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        """Stop admission and optionally wait for active provider calls to drain.

        Shutdown is signalled outside the bounded work queue. In-band sentinels
        would make ``close(wait=False)`` block behind a full queue while an
        active provider call is still running.
        """

        with self._lock:
            first_close = not self._closed
            self._closed = True
            workers = self._workers
        if first_close:
            if cancel_pending:
                self.cancel_pending()
            self._shutdown_requested.set()
        if wait:
            for worker in workers:
                worker.join()

    def __enter__(self) -> CanonicalLocalProviderQueue:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close(cancel_pending=_exc_type is not None)

    def _submit(
        self,
        callback: Callable[[], object],
        *,
        block: bool,
        timeout: float | None,
        count_full_rejection: bool,
    ) -> Future[object]:
        if not callable(callback):
            raise TypeError("callback must be callable")
        if timeout is not None and (isinstance(timeout, bool) or timeout < 0):
            raise ValueError("timeout must be nonnegative or None")
        future: Future[object] = Future()
        work = _ProviderWork(callback=callback, future=future)
        deadline = None if not block or timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if self._closed:
                    self._rejected += 1
                    raise CanonicalLocalCompositionError(
                        CanonicalLocalCompositionErrorCode.BACKPRESSURE,
                        "provider queue is closed",
                    )
                try:
                    # Never hold the service lock while waiting for capacity.
                    # Provider workers need this lock to finish the item that
                    # frees the queue slot.
                    self._queue.put_nowait(work)
                except Full as error:
                    if not block:
                        if count_full_rejection:
                            self._rejected += 1
                        raise CanonicalLocalCompositionError(
                            CanonicalLocalCompositionErrorCode.BACKPRESSURE,
                            f"provider queue is full (capacity {self._queue_capacity})",
                        ) from error
                else:
                    self._admitted += 1
                    self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
                    return future

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if count_full_rejection:
                        with self._lock:
                            self._rejected += 1
                    raise CanonicalLocalCompositionError(
                        CanonicalLocalCompositionErrorCode.BACKPRESSURE,
                        f"provider queue is full (capacity {self._queue_capacity})",
                    )
                time.sleep(min(0.001, remaining))
            else:
                time.sleep(0.001)

    def _worker(self) -> None:
        while True:
            try:
                work = self._queue.get(timeout=0.05)
            except Empty:
                if self._shutdown_requested.is_set():
                    return
                continue
            try:
                if not work.future.set_running_or_notify_cancel():
                    with self._lock:
                        self._cancelled += 1
                    continue
                with self._lock:
                    self._active += 1
                try:
                    result = work.callback()
                except BaseException as error:
                    # A caller can cancel an asyncio wrapper while this
                    # provider callback is still running. The underlying
                    # concurrent Future may then already be terminal when the
                    # callback returns; publishing again must not escape the
                    # worker thread or strand Queue.join. The callback still
                    # reached a terminal provider outcome, so preserve its
                    # failed/completed accounting; cancelled is reserved for
                    # work cancelled before execution begins.
                    with self._lock:
                        self._failed += 1
                    with contextlib.suppress(InvalidStateError):
                        work.future.set_exception(error)
                else:
                    with self._lock:
                        self._completed += 1
                    with contextlib.suppress(InvalidStateError):
                        work.future.set_result(result)
                finally:
                    with self._lock:
                        self._active -= 1
            finally:
                self._queue.task_done()


@dataclass(frozen=True, slots=True)
class CanonicalLocalFixtureJob:
    """One recording-affine local fixture command."""

    source_path: Path
    state_dir: Path
    # Run key selects a replay/execution invocation; it is not a recording
    # shard.  One immutable source remains affine to one active worker.
    run_key: str = "primary"


@dataclass(frozen=True, slots=True)
class CanonicalLocalRecordingServiceSnapshot:
    """Operational state for bounded recording-level local execution."""

    recording_worker_count: int
    ingress_queue_capacity: int
    ingress_depth: int
    active: int
    admitted: int
    completed: int
    failed: int
    cancelled: int
    rejected: int
    max_ingress_depth: int
    max_active: int
    claimed_state_dir_count: int
    unique_state_dir_count: int
    state_dir_claim_conflicts: int
    claimed_recording_key_count: int
    unique_recording_key_count: int
    recording_key_claim_conflicts: int
    fresh_receipts: int
    replayed_receipts: int
    completion_outbox_inline: bool
    closed: bool
    provider_queue: CanonicalLocalProviderQueueSnapshot


@dataclass(slots=True)
class _RecordingWork:
    job: CanonicalLocalFixtureJob
    state_dir: Path
    recording_key: str
    future: Future[CanonicalLocalRunReceipt]


class CanonicalLocalRecordingService:
    """Bounded recording workers with a shared provider dispatcher.

    A worker owns a job's state directory from admission until its result or
    failure becomes visible.  This prevents two local workers from opening the
    same SQLite/WAL state concurrently.  The default runner keeps CPU/media and
    completion/outbox work in recording workers; only provider invocation passes
    through :class:`CanonicalLocalProviderQueue`.

    This is an in-process LOCAL_CONFORMANCE topology.  Deployment workers should
    use process isolation and host-local NVMe roots, but that deployment choice
    remains outside canonical identity and this deterministic local command.
    """

    def __init__(
        self,
        *,
        recording_worker_count: int = 1,
        ingress_queue_capacity: int | None = None,
        provider_concurrency: int | None = None,
        provider_queue_capacity: int | None = None,
        fixture_runner: Callable[[CanonicalLocalFixtureJob], CanonicalLocalRunReceipt]
        | None = None,
    ) -> None:
        workers = _positive_int(recording_worker_count, "recording_worker_count")
        ingress_capacity = workers if ingress_queue_capacity is None else _positive_int(
            ingress_queue_capacity,
            "ingress_queue_capacity",
        )
        provider_workers = workers if provider_concurrency is None else _positive_int(
            provider_concurrency,
            "provider_concurrency",
        )
        provider_capacity = (
            max(1, workers * 2)
            if provider_queue_capacity is None
            else _positive_int(provider_queue_capacity, "provider_queue_capacity")
        )
        if fixture_runner is not None and not callable(fixture_runner):
            raise TypeError("fixture_runner must be callable")
        self._queue: Queue[_RecordingWork] = Queue(maxsize=ingress_capacity)
        self._provider_queue = CanonicalLocalProviderQueue(
            max_concurrency=provider_workers,
            queue_capacity=provider_capacity,
        )
        self._fixture_runner = fixture_runner or self._default_fixture_runner
        self._workers = tuple(
            Thread(
                target=self._worker,
                name=f"robata-recording-worker-{ordinal}",
                daemon=True,
            )
            for ordinal in range(workers)
        )
        self._lock = RLock()
        self._closed = False
        self._shutdown_requested = Event()
        self._claimed_state_dirs: set[Path] = set()
        self._claimed_recording_keys: set[str] = set()
        self._active = 0
        self._admitted = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._rejected = 0
        self._max_ingress_depth = 0
        self._max_active = 0
        self._state_dirs_seen: set[Path] = set()
        self._state_dir_claim_conflicts = 0
        self._recording_keys_seen: set[str] = set()
        self._recording_key_claim_conflicts = 0
        self._fresh_receipts = 0
        self._replayed_receipts = 0
        for worker in self._workers:
            worker.start()

    def _default_fixture_runner(
        self,
        job: CanonicalLocalFixtureJob,
    ) -> CanonicalLocalRunReceipt:
        return run_local_canonical_fixture(
            source_path=job.source_path,
            state_dir=job.state_dir,
            run_key=job.run_key,
            provider_dispatcher=self._provider_queue,
        )

    @property
    def recording_worker_count(self) -> int:
        return len(self._workers)

    @property
    def ingress_queue_capacity(self) -> int:
        return self._queue.maxsize

    @property
    def ingress_depth(self) -> int:
        return self._queue.qsize()

    @property
    def provider_queue(self) -> CanonicalLocalProviderQueue:
        return self._provider_queue

    @property
    def snapshot(self) -> CanonicalLocalRecordingServiceSnapshot:
        with self._lock:
            return CanonicalLocalRecordingServiceSnapshot(
                recording_worker_count=len(self._workers),
                ingress_queue_capacity=self._queue.maxsize,
                ingress_depth=self._queue.qsize(),
                active=self._active,
                admitted=self._admitted,
                completed=self._completed,
                failed=self._failed,
                cancelled=self._cancelled,
                rejected=self._rejected,
                max_ingress_depth=self._max_ingress_depth,
                max_active=self._max_active,
                claimed_state_dir_count=len(self._claimed_state_dirs),
                unique_state_dir_count=len(self._state_dirs_seen),
                state_dir_claim_conflicts=self._state_dir_claim_conflicts,
                claimed_recording_key_count=len(self._claimed_recording_keys),
                unique_recording_key_count=len(self._recording_keys_seen),
                recording_key_claim_conflicts=self._recording_key_claim_conflicts,
                fresh_receipts=self._fresh_receipts,
                replayed_receipts=self._replayed_receipts,
                # Completion/outbox stays in the recording-affine command; a
                # separate publish queue belongs to stream-control.
                completion_outbox_inline=True,
                closed=self._closed,
                provider_queue=self._provider_queue.snapshot,
            )

    def submit_fixture(self, job: CanonicalLocalFixtureJob) -> Future[CanonicalLocalRunReceipt]:
        """Attempt non-blocking recording admission under the ingress bound."""

        return self._submit(job, block=False)

    def run_fixture(self, job: CanonicalLocalFixtureJob) -> CanonicalLocalRunReceipt:
        return self.submit_fixture(job).result()

    def run_fixtures(
        self,
        jobs: Sequence[CanonicalLocalFixtureJob],
    ) -> tuple[CanonicalLocalRunReceipt, ...]:
        """Submit a finite batch, respecting ingress capacity and receipt order."""

        futures = [self._submit(job, block=True) for job in jobs]
        return tuple(future.result() for future in futures)

    async def arun_fixture(self, job: CanonicalLocalFixtureJob) -> CanonicalLocalRunReceipt:
        """Await one recording worker without blocking the caller event loop."""

        return await asyncio.wrap_future(self.submit_fixture(job))

    async def arun_fixtures(
        self,
        jobs: Sequence[CanonicalLocalFixtureJob],
    ) -> tuple[CanonicalLocalRunReceipt, ...]:
        futures: list[Future[CanonicalLocalRunReceipt]] = []
        for job in jobs:
            while True:
                try:
                    future = self._submit(job, block=False)
                except CanonicalLocalCompositionError as error:
                    if self._closed or "recording ingress queue is full" not in str(error):
                        raise
                    await asyncio.sleep(0.001)
                else:
                    futures.append(future)
                    break
        results = await asyncio.gather(*(asyncio.wrap_future(future) for future in futures))
        return tuple(results)

    def drain(self) -> None:
        """Drain recording work and then all provider work it produced."""

        self._queue.join()
        self._provider_queue.drain()

    def close(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        """Stop new recordings and settle or cancel queued local work.

        Shutdown is an event rather than an in-band queue sentinel. A sentinel
        cannot be inserted without blocking when a bounded ingress queue is
        full behind a running recording, which would make ``wait=False`` lie.
        Workers drain admitted work (or observe cancellation) and exit once the
        queue becomes empty after shutdown has been requested.
        """

        with self._lock:
            first_close = not self._closed
            self._closed = True
        if first_close and cancel_pending:
            self._cancel_pending_recordings()
        if first_close:
            # Cancellation temporarily removes retained work from the queue.
            # Signal shutdown only after it has restored that work.
            self._shutdown_requested.set()
        if wait:
            self._join_and_close_provider(cancel_pending=cancel_pending)
        elif first_close:
            Thread(
                target=self._join_and_close_provider,
                kwargs={"cancel_pending": cancel_pending},
                name="robata-recording-service-close",
                daemon=True,
            ).start()

    def __enter__(self) -> CanonicalLocalRecordingService:
        return self

    def __exit__(self, exc_type: object, _exc: object, _tb: object) -> None:
        self.close(cancel_pending=exc_type is not None)

    def _submit(
        self,
        job: CanonicalLocalFixtureJob,
        *,
        block: bool,
    ) -> Future[CanonicalLocalRunReceipt]:
        if not isinstance(job, CanonicalLocalFixtureJob):
            raise TypeError("job must be CanonicalLocalFixtureJob")
        state_dir = _state_dir(job.state_dir)
        recording_key = _recording_key(job)
        future: Future[CanonicalLocalRunReceipt] = Future()
        work = _RecordingWork(
            job=job,
            state_dir=state_dir,
            recording_key=recording_key,
            future=future,
        )
        claimed = False
        while True:
            with self._lock:
                if self._closed:
                    if claimed:
                        self._claimed_state_dirs.discard(state_dir)
                        self._claimed_recording_keys.discard(recording_key)
                    self._rejected += 1
                    raise CanonicalLocalCompositionError(
                        CanonicalLocalCompositionErrorCode.BACKPRESSURE,
                        "recording service is closed",
                    )
                if not claimed:
                    if state_dir in self._claimed_state_dirs:
                        self._rejected += 1
                        self._state_dir_claim_conflicts += 1
                        raise CanonicalLocalCompositionError(
                            CanonicalLocalCompositionErrorCode.BACKPRESSURE,
                            "state directory is already assigned to an active recording: "
                            f"{state_dir}",
                        )
                    if recording_key in self._claimed_recording_keys:
                        self._rejected += 1
                        self._recording_key_claim_conflicts += 1
                        raise CanonicalLocalCompositionError(
                            CanonicalLocalCompositionErrorCode.BACKPRESSURE,
                            "recording shard is already assigned to an active worker",
                        )
                    self._claimed_state_dirs.add(state_dir)
                    self._claimed_recording_keys.add(recording_key)
                    claimed = True
                try:
                    self._queue.put_nowait(work)
                except Full as error:
                    if not block:
                        self._claimed_state_dirs.discard(state_dir)
                        self._claimed_recording_keys.discard(recording_key)
                        self._rejected += 1
                        raise CanonicalLocalCompositionError(
                            CanonicalLocalCompositionErrorCode.BACKPRESSURE,
                            f"recording ingress queue is full (capacity {self._queue.maxsize})",
                        ) from error
                else:
                    self._admitted += 1
                    self._state_dirs_seen.add(state_dir)
                    self._recording_keys_seen.add(recording_key)
                    self._max_ingress_depth = max(self._max_ingress_depth, self._queue.qsize())
                    return future
            time.sleep(0.001)

    def _cancel_pending_recordings(self) -> int:
        cancelled = 0
        retained: list[_RecordingWork] = []
        while True:
            try:
                work = self._queue.get_nowait()
            except Empty:
                break
            try:
                if work.future.cancel():
                    cancelled += 1
                    with self._lock:
                        self._claimed_state_dirs.discard(work.state_dir)
                        self._claimed_recording_keys.discard(work.recording_key)
                else:
                    retained.append(work)
            finally:
                self._queue.task_done()
        for work in retained:
            self._queue.put(work)
        if cancelled:
            with self._lock:
                self._cancelled += cancelled
        return cancelled

    def _worker(self) -> None:
        while True:
            try:
                # Poll only to observe an out-of-band shutdown request. This
                # keeps a full ingress queue from making non-waiting close block.
                work = self._queue.get(timeout=0.05)
            except Empty:
                if self._shutdown_requested.is_set():
                    return
                continue
            try:
                if not work.future.set_running_or_notify_cancel():
                    with self._lock:
                        self._cancelled += 1
                        self._claimed_state_dirs.discard(work.state_dir)
                        self._claimed_recording_keys.discard(work.recording_key)
                    continue
                with self._lock:
                    self._active += 1
                    self._max_active = max(self._max_active, self._active)
                try:
                    result = self._fixture_runner(work.job)
                except BaseException as error:
                    with self._lock:
                        self._failed += 1
                    with contextlib.suppress(InvalidStateError):
                        work.future.set_exception(error)
                else:
                    with self._lock:
                        self._completed += 1
                        if bool(getattr(result, "replayed", False)):
                            self._replayed_receipts += 1
                        else:
                            self._fresh_receipts += 1
                    with contextlib.suppress(InvalidStateError):
                        work.future.set_result(result)
                finally:
                    with self._lock:
                        self._active -= 1
                        self._claimed_state_dirs.discard(work.state_dir)
                        self._claimed_recording_keys.discard(work.recording_key)
            finally:
                self._queue.task_done()

    def _join_and_close_provider(self, *, cancel_pending: bool) -> None:
        for worker in self._workers:
            worker.join()
        self._provider_queue.close(wait=True, cancel_pending=cancel_pending)


def _recording_key(job: CanonicalLocalFixtureJob) -> str:
    if not isinstance(job.source_path, Path):
        raise TypeError("source_path must be pathlib.Path")
    source = job.source_path.resolve()
    try:
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError:
        # Preserve the canonical command's SOURCE_INVALID handling for a broken
        # local source while still preventing duplicate work for its locator.
        source_sha256 = f"unreadable:{source}"
    return f"fixture-source:{source_sha256}"


def _state_dir(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError("state_dir must be pathlib.Path")
    return value.resolve()


def run_local_canonical_fixtures(
    jobs: Sequence[CanonicalLocalFixtureJob],
    *,
    recording_worker_count: int = 1,
    ingress_queue_capacity: int | None = None,
    provider_concurrency: int | None = None,
    provider_queue_capacity: int | None = None,
) -> tuple[CanonicalLocalRunReceipt, ...]:
    """Run a finite fixture batch using bounded local parallel composition."""

    with CanonicalLocalRecordingService(
        recording_worker_count=recording_worker_count,
        ingress_queue_capacity=ingress_queue_capacity,
        provider_concurrency=provider_concurrency,
        provider_queue_capacity=provider_queue_capacity,
    ) as service:
        return service.run_fixtures(jobs)


__all__ = [
    "CanonicalLocalFixtureJob",
    "CanonicalLocalProviderQueue",
    "CanonicalLocalProviderQueueSnapshot",
    "CanonicalLocalRecordingService",
    "CanonicalLocalRecordingServiceSnapshot",
    "run_local_canonical_fixtures",
]
