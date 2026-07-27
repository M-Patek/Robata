"""Atomic Redis transport for the provider-neutral :mod:`task_queue` port.

The durable scheduler remains authoritative.  This adapter only manages broker
local delivery: admission, priority selection, leases, retry/DLQ, and result
bytes.  The Redis script is the transaction boundary, so a stale lease cannot
acknowledge work that has been reclaimed by another worker.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, NoReturn, Protocol, cast

from robata.ports.task_queue import (
    InspectableTaskQueue,
    LeaseId,
    PipelineTask,
    TaskId,
    TaskQueueError,
    TaskQueueErrorCode,
    TaskSnapshot,
    TaskStatus,
)
from robata.runtime.observability import RuntimeObserver, runtime_increment


class RedisTaskQueueClient(Protocol):
    """The synchronous Redis surface used by :class:`RedisTaskQueue`."""

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any: ...


_DEFAULT_PREFIX = "robata:task-queue:v1"
_UNAVAILABLE = "Redis task queue adapter is unavailable"


# KEYS[1] is a key prefix; the adapter targets a single Redis primary (or a
# Redis Cluster setup that co-locates all keys for the prefix).  Redis TIME is
# deliberately used as the lease clock rather than a worker's wall clock.
_SCRIPT = r"""
local root = KEYS[1]
local op = ARGV[1]
local backoff = tonumber(ARGV[2])
local tombstone_ttl = tonumber(ARGV[3])
local function k(s) return root .. ':' .. s end
local function tk(t) return k('task:' .. t) end
local function pk(p) return k('pending:' .. tostring(p)) end
local function now_us()
  local t = redis.call('TIME')
  return tonumber(t[1]) * 1000000 + tonumber(t[2])
end
local function pad(v)
  local s = tostring(v)
  return string.rep('0', math.max(0, 20 - string.len(s))) .. s
end
local function err(c) return {'error', c} end
local function load(t)
  local raw = redis.call('GET', tk(t))
  if not raw then return nil, 'MISSING' end
  local ok, r = pcall(cjson.decode, raw)
  if not ok or type(r) ~= 'table' or r.token ~= t
    or type(r.status) ~= 'string' or type(r.priority) ~= 'number'
    or type(r.available_at) ~= 'number' or type(r.retry_count) ~= 'number'
    or type(r.max_retries) ~= 'number' then return nil, 'CORRUPT' end
  return r, nil
end
local function save(r) redis.call('SET', tk(r.token), cjson.encode(r)) end
local function remove_pending(r)
  if not r.pending then return end
  local q = pk(r.priority)
  redis.call('ZREM', q, r.pending)
  if redis.call('ZCARD', q) == 0 then
    redis.call('DEL', q)
    redis.call('ZREM', k('priorities'), tostring(r.priority))
  end
  r.pending = nil
end
local function add_pending(r)
  r.pending = r.created_sort .. '|' .. pad(r.sequence) .. '|' .. r.token
  redis.call('ZADD', pk(r.priority), r.available_at, r.pending)
  redis.call('ZADD', k('priorities'), r.priority, tostring(r.priority))
end
local function clear_lease(r)
  local l = r.lease_id
  if l then
    redis.call('HDEL', k('leases'), l)
    redis.call('ZREM', k('lease-expiry'), l)
  end
  r.lease_id = nil
  r.leased_by = nil
  r.lease_expires_at = nil
  r.lease_duration = nil
  return l
end
local function retry(r, reason, now)
  r.retry_count = r.retry_count + 1
  r.failure_reason = reason
  r.result = nil
  if r.retry_count > r.max_retries then
    r.status = 'DEAD_LETTER'
    r.available_at = now
    r.pending = nil
    redis.call('SREM', k('active'), r.token)
    redis.call('ZADD', k('dead-letters'), redis.call('INCR', k('dead-seq')), r.token)
    return nil
  end
  local delay = backoff * (2 ^ (r.retry_count - 1))
  if delay ~= delay or delay == math.huge then return 'CORRUPT' end
  r.status = 'PENDING'
  r.available_at = now + math.floor(delay + 0.5)
  add_pending(r)
  return nil
end
local function expire(r, lease, now)
  clear_lease(r)
  redis.call('SET', k('retired:' .. lease), 'EXPIRED', 'EX', tombstone_ttl)
  local e = retry(r, 'lease expired', now)
  if not e then save(r) end
  return e
end
local function sweep(now)
  local ids = redis.call('ZRANGEBYSCORE', k('lease-expiry'), '-inf', now)
  local loaded = {}
  for _, l in ipairs(ids) do
    local t = redis.call('HGET', k('leases'), l)
    if t then
      local r, e = load(t)
      if e == 'CORRUPT' then return 0, e end
      table.insert(loaded, {lease = l, record = r})
    end
  end
  local count = 0
  for _, item in ipairs(loaded) do
    local r = item.record
    if r and r.status == 'CLAIMED' and r.lease_id == item.lease
      and r.lease_expires_at <= now then
      local e = expire(r, item.lease, now)
      if e then return count, e end
      count = count + 1
    end
    redis.call('HDEL', k('leases'), item.lease)
    redis.call('ZREM', k('lease-expiry'), item.lease)
  end
  for _, l in ipairs(ids) do
    redis.call('HDEL', k('leases'), l)
    redis.call('ZREM', k('lease-expiry'), l)
  end
  return count, nil
end
local function active_lease(lease, now)
  local t = redis.call('HGET', k('leases'), lease)
  if not t then
    if redis.call('GET', k('retired:' .. lease)) then return nil, 'LEASE_EXPIRED' end
    return nil, 'LEASE_NOT_FOUND'
  end
  local r, e = load(t)
  if e == 'CORRUPT' then return nil, e end
  if not r or r.status ~= 'CLAIMED' or r.lease_id ~= lease then
    redis.call('HDEL', k('leases'), lease)
    redis.call('ZREM', k('lease-expiry'), lease)
    return nil, 'LEASE_NOT_FOUND'
  end
  if r.lease_expires_at <= now then
    local x = expire(r, lease, now)
    if x then return nil, x end
    return nil, 'LEASE_EXPIRED'
  end
  return r, nil
end
local now = now_us()
local swept, sweep_error = sweep(now)
if sweep_error then return err(sweep_error) end

if op == 'enqueue' then
  local max_size, token, id = tonumber(ARGV[4]), ARGV[5], ARGV[6]
  local priority, created_us = tonumber(ARGV[10]), tonumber(ARGV[13])
  local retry_count, max_retries = tonumber(ARGV[14]), tonumber(ARGV[15])
  if not max_size or not priority or not created_us or not retry_count or not max_retries then
    return err('CORRUPT')
  end
  if redis.call('EXISTS', tk(token)) == 1 then return err('DUPLICATE_TASK') end
  if max_size >= 0 and redis.call('SCARD', k('active')) >= max_size then
    return err('QUEUE_FULL')
  end
  local r = {
    token = token, task_id = id, recording_id = ARGV[7], stage = ARGV[8],
    payload = ARGV[9], priority = priority, created_at = ARGV[11],
    created_sort = ARGV[12], retry_count = retry_count, max_retries = max_retries,
    status = 'PENDING', available_at = math.max(now, created_us),
    sequence = redis.call('INCR', k('task-seq')),
  }
  add_pending(r)
  save(r)
  redis.call('SADD', k('active'), token)
  return {'ok', id}
end

if op == 'claim' then
  local worker, duration = ARGV[4], tonumber(ARGV[5])
  if not duration or duration <= 0 then return err('CORRUPT') end
  local priorities = redis.call('ZREVRANGE', k('priorities'), 0, -1)
  for _, p in ipairs(priorities) do
    local q = pk(p)
    while true do
      local c = redis.call('ZRANGEBYSCORE', q, '-inf', now, 'LIMIT', 0, 1)
      if #c == 0 then break end
      local member = c[1]
      local token = string.match(member, '.*|([^|]+)$')
      local r, e = token and load(token) or nil, token and nil or 'MISSING'
      if e == 'CORRUPT' then return err(e) end
      if not r or r.status ~= 'PENDING' or r.pending ~= member or r.available_at > now then
        redis.call('ZREM', q, member)
        if r and r.status == 'PENDING' then r.pending = nil; add_pending(r); save(r) end
      else
        remove_pending(r)
        local lease = 'lease-' .. pad(redis.call('INCR', k('lease-seq')))
        r.status = 'CLAIMED'; r.lease_id = lease; r.leased_by = worker
        r.lease_duration = duration; r.lease_expires_at = now + duration
        redis.call('HSET', k('leases'), lease, token)
        redis.call('ZADD', k('lease-expiry'), r.lease_expires_at, lease)
        save(r)
        return {'ok', r.task_id, r.recording_id, r.stage, r.payload,
          tostring(r.priority), r.created_at, tostring(r.retry_count),
          tostring(r.max_retries), lease, worker, tostring(r.lease_expires_at)}
      end
    end
    if redis.call('ZCARD', q) == 0 then
      redis.call('DEL', q)
      redis.call('ZREM', k('priorities'), p)
    end
  end
  return {'ok', 'none'}
end

if op == 'heartbeat' then
  local lease, requested = ARGV[4], ARGV[5]
  local r, e = active_lease(lease, now)
  if e then
    if e == 'CORRUPT' then return err(e) end
    return {'ok', '0'}
  end
  local duration = requested == '' and r.lease_duration or tonumber(requested)
  if not duration or duration <= 0 then return {'ok', '0'} end
  r.lease_duration = duration; r.lease_expires_at = now + duration
  redis.call('ZADD', k('lease-expiry'), r.lease_expires_at, lease)
  save(r)
  return {'ok', '1'}
end

if op == 'complete' then
  local lease, result = ARGV[4], ARGV[5]
  local r, e = active_lease(lease, now)
  if e then return err(e) end
  clear_lease(r)
  r.status = 'COMPLETED'; r.result = result; r.failure_reason = nil
  redis.call('SREM', k('active'), r.token)
  save(r)
  return {'ok'}
end

if op == 'fail' then
  local lease, reason = ARGV[4], ARGV[5]
  local r, e = active_lease(lease, now)
  if e then return err(e) end
  clear_lease(r)
  redis.call('SET', k('retired:' .. lease), 'EXPIRED', 'EX', tombstone_ttl)
  local x = retry(r, reason, now)
  if x then return err(x) end
  save(r)
  return {'ok'}
end

if op == 'status' or op == 'result' then
  local r, e = load(ARGV[4])
  if e == 'MISSING' then return err('TASK_NOT_FOUND') end
  if e then return err(e) end
  if op == 'status' then return {'ok', r.status} end
  if r.result == nil then return {'ok', '0'} end
  return {'ok', '1', r.result}
end
if op == 'inspect' then
  local r, e = load(ARGV[4])
  if e == 'MISSING' then return err('TASK_NOT_FOUND') end
  if e then return err(e) end
  -- Return one JSON value so the adapter can validate the complete snapshot
  -- without making a second, non-atomic read of the task record.
  return {'ok', cjson.encode(r)}
end
if op == 'dead_letters' then
  local limit = tonumber(ARGV[4])
  if not limit or limit <= 0 then return err('CORRUPT') end
  local tokens = redis.call('ZRANGE', k('dead-letters'), 0, limit - 1)
  local records = {}
  for _, token in ipairs(tokens) do
    local r, e = load(token)
    if e == 'CORRUPT' then return err(e) end
    if r and r.status == 'DEAD_LETTER' then table.insert(records, r) end
  end
  return {'ok', cjson.encode(records)}
end
if op == 'depth' then return {'ok', tostring(redis.call('SCARD', k('active')))} end
if op == 'sweep' then return {'ok', tostring(swept)} end
return err('CORRUPT')
"""


def _microseconds(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    delta = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def _from_microseconds(value: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=value)


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    value = value.astimezone(UTC)
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
        f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}.{value.microsecond:06d}Z"
    )


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stored timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _token(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _positive_seconds(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TaskQueueError(TaskQueueErrorCode.INVALID_REQUEST, f"{name} must be positive")
    return value * 1_000_000


def _backoff(value: float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskQueueError(
            TaskQueueErrorCode.INVALID_REQUEST,
            "retry_backoff_seconds must be a non-negative number",
        )
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise TaskQueueError(
            TaskQueueErrorCode.INVALID_REQUEST,
            "retry_backoff_seconds must be finite and non-negative",
        )
    return round(value * 1_000_000)


def _b64decode(value: str, field: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError(f"stored {field} is not valid base64") from error


class RedisTaskQueue(InspectableTaskQueue):
    """Redis implementation of ``TaskQueue`` with a client-injection seam.

    When no client is supplied, the optional ``redis`` dependency is loaded
    lazily.  Construction avoids a network probe; any unavailable client or
    Redis command fails explicitly with ``ADAPTER_UNAVAILABLE``.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        client: RedisTaskQueueClient | None = None,
        key_prefix: str = _DEFAULT_PREFIX,
        retry_backoff_seconds: float = 1.0,
        max_size: int | None = None,
        retired_lease_ttl_seconds: int = 86_400,
        runtime_observer: RuntimeObserver | None = None,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(key_prefix, str) or not key_prefix.strip():
            raise ValueError("key_prefix must be a non-empty string")
        if runtime_observer is not None and not callable(
            getattr(runtime_observer, "increment_counter", None)
        ):
            raise TypeError("runtime_observer must implement increment_counter")
        if failure_injector is not None and not callable(failure_injector):
            raise TypeError("failure_injector must be callable or None")
        if max_size is not None and (
            isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0
        ):
            raise ValueError("max_size must be a positive integer")
        self._prefix = key_prefix.rstrip(":")
        self._backoff = _backoff(retry_backoff_seconds)
        self._max_size = max_size
        self._tombstone_ttl = _positive_seconds(
            retired_lease_ttl_seconds,
            "retired_lease_ttl_seconds",
        ) // 1_000_000
        self._client = client
        self._runtime_observer = runtime_observer
        self._failure_injector = failure_injector
        self._unavailable_reason: str | None = None
        if client is not None:
            return
        if not isinstance(redis_url, str) or not redis_url.strip():
            raise ValueError("redis_url is required when client is not supplied")
        try:
            import redis
        except ImportError:
            self._unavailable_reason = "redis-py is not installed"
            return
        try:
            self._client = cast(
                RedisTaskQueueClient,
                redis.Redis.from_url(redis_url, decode_responses=False),
            )
        except Exception:
            self._unavailable_reason = "Redis client could not be configured"

    @property
    def depth(self) -> int:
        """Number of active (pending or leased) tasks after expiry recovery."""

        response = self._run("depth")
        return self._integer(response, 1, "depth")

    def enqueue(self, task: PipelineTask) -> TaskId:
        if not isinstance(task, PipelineTask):
            raise TaskQueueError(TaskQueueErrorCode.INVALID_REQUEST, "task must be PipelineTask")
        if task.lease_id is not None:
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "a task with lease metadata cannot be enqueued",
            )
        token = _token(task.task_id.value)
        created_at = _iso(task.created_at)
        response = self._run(
            "enqueue",
            str(self._max_size if self._max_size is not None else -1),
            token,
            task.task_id.value,
            task.recording_id,
            task.stage,
            base64.b64encode(task.payload).decode("ascii"),
            str(task.priority),
            created_at,
            created_at,
            str(_microseconds(task.created_at)),
            str(task.retry_count),
            str(task.max_retries),
        )
        if len(response) != 2 or response[1] != task.task_id.value:
            self._unavailable("enqueue returned an invalid acknowledgement")
        return task.task_id

    def claim(self, worker_id: str, lease_duration_seconds: int) -> PipelineTask | None:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "worker_id must be a non-empty string",
            )
        response = self._run(
            "claim",
            worker_id.strip(),
            str(_positive_seconds(lease_duration_seconds, "lease_duration_seconds")),
        )
        if response == ("ok", "none"):
            return None
        if len(response) != 12:
            self._unavailable("claim returned an invalid task")
        try:
            return PipelineTask(
                task_id=TaskId(response[1]),
                recording_id=response[2],
                stage=response[3],
                payload=_b64decode(response[4], "payload"),
                priority=int(response[5]),
                created_at=_parse_iso(response[6]),
                retry_count=int(response[7]),
                max_retries=int(response[8]),
                lease_id=LeaseId(response[9]),
                leased_by=response[10],
                lease_expires_at=_from_microseconds(int(response[11])),
            )
        except (TypeError, ValueError, OverflowError) as error:
            self._unavailable("claim returned malformed task data", error)

    def heartbeat(
        self,
        lease_id: LeaseId,
        lease_duration_seconds: int | None = None,
    ) -> bool:
        if not isinstance(lease_id, LeaseId):
            return False
        if lease_duration_seconds is None:
            duration = ""
        else:
            try:
                duration = str(_positive_seconds(lease_duration_seconds, "lease_duration_seconds"))
            except TaskQueueError:
                return False
        response = self._run("heartbeat", lease_id.value, duration)
        if len(response) != 2 or response[1] not in {"0", "1"}:
            self._unavailable("heartbeat returned an invalid acknowledgement")
        return response[1] == "1"

    def complete(self, lease_id: LeaseId, result: bytes) -> None:
        if not isinstance(lease_id, LeaseId):
            raise TaskQueueError(TaskQueueErrorCode.INVALID_REQUEST, "lease_id must be LeaseId")
        if not isinstance(result, bytes):
            raise TaskQueueError(TaskQueueErrorCode.INVALID_REQUEST, "result must be bytes")
        response = self._run("complete", lease_id.value, base64.b64encode(result).decode("ascii"))
        if response != ("ok",):
            self._unavailable("complete returned an invalid acknowledgement")

    def fail(self, lease_id: LeaseId, reason: str) -> None:
        if not isinstance(lease_id, LeaseId):
            raise TaskQueueError(TaskQueueErrorCode.INVALID_REQUEST, "lease_id must be LeaseId")
        if not isinstance(reason, str) or not reason.strip():
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "reason must be a non-empty string",
            )
        response = self._run("fail", lease_id.value, reason.strip())
        if response != ("ok",):
            self._unavailable("fail returned an invalid acknowledgement")

    def get_status(self, task_id: TaskId) -> TaskStatus:
        response = self._run("status", self._task_token(task_id))
        if len(response) != 2:
            self._unavailable("status returned an invalid response")
        try:
            return TaskStatus(response[1])
        except ValueError as error:
            self._unavailable("status returned an unknown state", error)

    def get_result(self, task_id: TaskId) -> bytes | None:
        """Return completed exact result bytes; useful for broker reconciliation."""

        response = self._run("result", self._task_token(task_id))
        if response == ("ok", "0"):
            return None
        if len(response) == 3 and response[1] == "1":
            try:
                return _b64decode(response[2], "result")
            except ValueError as error:
                self._unavailable("result is malformed", error)
        self._unavailable("result returned an invalid response")

    def sweep_expired(self) -> int:
        """Move all currently expired leases to retry/DLQ state."""

        return self._integer(self._run("sweep"), 1, "sweep count")

    def inspect(self, task_id: TaskId) -> TaskSnapshot:
        """Return an atomic broker snapshot for one task."""

        response = self._run("inspect", self._task_token(task_id))
        if len(response) != 2:
            self._unavailable("inspect returned an invalid response")
        try:
            document = json.loads(response[1])
        except (TypeError, json.JSONDecodeError) as error:
            self._unavailable("inspect returned malformed JSON", error)
        try:
            return self._snapshot_from_document(document)
        except (TypeError, ValueError, OverflowError) as error:
            self._unavailable("inspect returned malformed task state", error)

    def list_dead_letters(self, *, limit: int = 100) -> tuple[TaskSnapshot, ...]:
        """Return dead-letter snapshots in Redis insertion order."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        response = self._run("dead_letters", str(limit))
        if len(response) != 2:
            self._unavailable("dead_letters returned an invalid response")
        try:
            documents = json.loads(response[1])
            if not isinstance(documents, list):
                raise ValueError("dead_letters payload must be a list")
            return tuple(self._snapshot_from_document(document) for document in documents)
        except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
            self._unavailable("dead_letters returned malformed task state", error)

    @staticmethod
    def _snapshot_from_document(document: object) -> TaskSnapshot:
        if not isinstance(document, dict):
            raise ValueError("task snapshot must be an object")

        def required_text(name: str) -> str:
            value = document.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"task snapshot {name} must be nonempty text")
            return value

        def required_int(name: str) -> int:
            value = document.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"task snapshot {name} must be an integer")
            return value

        def optional_text(name: str) -> str | None:
            value = document.get(name)
            if value is None:
                return None
            if not isinstance(value, str) or not value:
                raise ValueError(f"task snapshot {name} must be optional text")
            return value

        status = TaskStatus(required_text("status"))
        retry_count = required_int("retry_count")
        max_retries = required_int("max_retries")
        available_at = (
            _from_microseconds(required_int("available_at"))
            if status is TaskStatus.PENDING
            else None
        )
        lease_value = optional_text("lease_id")
        leased_by = optional_text("leased_by")
        expires_value = document.get("lease_expires_at")
        if expires_value is None:
            lease_expires_at = None
        elif isinstance(expires_value, bool) or not isinstance(expires_value, int):
            raise ValueError("task snapshot lease_expires_at must be an integer")
        else:
            lease_expires_at = _from_microseconds(expires_value)
        if status is TaskStatus.CLAIMED and (
            lease_value is None or leased_by is None or lease_expires_at is None
        ):
            raise ValueError("claimed task snapshot lacks lease metadata")
        result_value = document.get("result")
        result = None if result_value is None else _b64decode(result_value, "result")
        return TaskSnapshot(
            task_id=TaskId(required_text("task_id")),
            status=status,
            retry_count=retry_count,
            max_retries=max_retries,
            available_at=available_at,
            lease_id=None if lease_value is None else LeaseId(lease_value),
            leased_by=leased_by,
            lease_expires_at=lease_expires_at,
            failure_reason=optional_text("failure_reason"),
            result=result,
        )
    def _task_token(self, task_id: TaskId) -> str:
        if not isinstance(task_id, TaskId):
            raise TaskQueueError(TaskQueueErrorCode.INVALID_REQUEST, "task_id must be TaskId")
        return _token(task_id.value)

    def _run(self, operation: str, *arguments: str) -> tuple[str, ...]:
        runtime_increment(
            self._runtime_observer,
            "redis.task_queue.operations",
            attributes={"operation": operation},
        )
        client = self._client
        if client is None:
            self._unavailable(self._unavailable_reason or "no Redis client is configured")
        if not callable(getattr(client, "eval", None)):
            self._unavailable("Redis client does not support EVAL")
        try:
            if self._failure_injector is not None:
                self._failure_injector(operation)
            raw = client.eval(
                _SCRIPT,
                1,
                self._prefix,
                operation,
                str(self._backoff),
                str(self._tombstone_ttl),
                *arguments,
            )
        except Exception as error:
            runtime_increment(
                self._runtime_observer,
                "redis.task_queue.failures",
                attributes={"operation": operation},
            )
            self._unavailable("Redis command failed", error)
        if not isinstance(raw, (list, tuple)):
            self._unavailable("Redis script returned a non-array response")
        try:
            response = tuple(self._text(item) for item in raw)
        except ValueError as error:
            self._unavailable("Redis script response is malformed", error)
        if not response:
            self._unavailable("Redis script returned an empty response")
        if response[0] == "error":
            if len(response) != 2:
                self._unavailable("Redis script returned a malformed error")
            try:
                raise TaskQueueError(
                    TaskQueueErrorCode(response[1]),
                    f"Redis task queue operation failed: {response[1]}",
                )
            except ValueError as error:
                self._unavailable("Redis task queue state is corrupt", error)
        if response[0] != "ok":
            self._unavailable("Redis script returned an unknown response")
        runtime_increment(
            self._runtime_observer,
            "redis.task_queue.successes",
            attributes={"operation": operation},
        )
        return response

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        raise ValueError("unsupported Redis response value")

    def _integer(self, response: tuple[str, ...], index: int, field: str) -> int:
        if len(response) != index + 1:
            self._unavailable(f"{field} returned an invalid response")
        try:
            result = int(response[index])
        except ValueError as error:
            self._unavailable(f"{field} is not an integer", error)
        if result < 0:
            self._unavailable(f"{field} must not be negative")
        return result

    @staticmethod
    def _unavailable(message: str, cause: Exception | None = None) -> NoReturn:
        error = TaskQueueError(TaskQueueErrorCode.ADAPTER_UNAVAILABLE, f"{_UNAVAILABLE}: {message}")
        if cause is not None:
            raise error from cause
        raise error


__all__ = ["RedisTaskQueue", "RedisTaskQueueClient"]
