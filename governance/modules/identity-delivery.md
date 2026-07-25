# Identity and Delivery

## Scope and path anchors
- Admission and review: `src/robata/admission/**`, `src/robata/review/**`
- Identity/completion/outbox adapters: `sqlite_event_identity_registry.py`, `sqlite_primary_completion.py`, `sqlite_outbox.py`, `sqlite_review_queue.py`
- Run helpers: `application/canonical/{logical_nodes,primary_completion,local_outbox_delivery}.py`, `application/canonical_run_membership.py`

## How to dispatch
`identity-delivery / P<n> - <identity, completion, outbox, admission, reconciliation, or review task>`

## Construction phases
1. **Logical identity** - derive stable memberships, nodes, revisions, and idempotency behavior.
2. **Completion and outbox** - persist completion facts, publish local delivery rows, and reconcile crashes.
3. **Admission and review** - make local output decisions and route non-blocking review work.
4. **Production adaptation** - connect durable storage/broker only after local recovery behavior is tested.

## Relevant tests
- Fast: `python -m pytest tests/unit/test_event_identity_registry.py tests/unit/test_admission_ledgers.py tests/unit/test_review_routing.py`
- Broader: `python -m pytest tests/integration/test_sqlite_primary_completion.py tests/integration/test_sqlite_outbox_relay.py tests/integration/test_canonical_local_review_routing.py`

## Read alongside
Read `event-semantics` for event inputs, `stream-control` for work/retry behavior, and `canonical-integration` before changing the completion path. Version any change to hashes, logical keys, idempotency keys, or fences.
