# Stream Control

## Scope and path anchors
- Queue and barrier APIs: `src/robata/queue/**`
- Local scheduler adapters: `sqlite_barrier.py`, `sqlite_work_scheduler.py`, `sqlite_stream_work_ledger.py`, `sqlite_stream_delivery.py`
- Run bridge helpers: `application/canonical/{durable_work,stream_scheduler,stream_recording_reduction}.py`

## How to dispatch
`stream-control / P<n> - <work-state, scheduler, barrier, retry, lease, or stream-reduction task>`

## Construction phases
1. **Work state** - model pending/running/succeeded/failed work and local persistence.
2. **Scheduling and barriers** - dispatch bounded work, leases, fences, retries, and recovery.
3. **Stream reduction** - turn recording work into canonical input without losing delivery facts.
4. **Broker adaptation** - add a real broker only after local scheduling behavior is proven.

## Relevant tests
- Fast: `python -m pytest tests/unit/test_barrier.py tests/unit/test_sqlite_barrier.py tests/unit/test_sqlite_work_scheduler.py tests/unit/test_stream_recording_reduction.py`
- Broader: `python -m pytest tests/integration/test_canonical_local_command.py`

## Read alongside
Read `source-media` for bounded source work, `identity-delivery` for idempotent completion, and `canonical-integration` before changing durable-work or scheduler composition.
