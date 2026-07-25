# Canonical Integration

## Scope and path anchors
- Compatibility entry: `src/robata/application/canonical_offline.py`
- Runner and composition: `application/canonical/{runner,runner_support,local_composition,result_validation}.py`
- Semantic core: `application/canonical/{models,projections,reduction,output_admission,boundary_windows}.py`
- Canonical bridges: `mcap_source.py`, `durable_work.py`, `primary_completion.py`, `logical_nodes.py`, local finalization/review helpers

## How to dispatch
`canonical-integration / P<n> - <runner, reduction, composition, replay, recovery, or cross-module bridge task>`

## Construction phases
1. **Runner and reduction** - advance stages and reduce facts into deterministic results.
2. **Local composition** - wire local source, inference, scheduler, identity, and delivery implementations.
3. **Replay and recovery** - validate closure, result consistency, and restart behavior.
4. **End-to-end runs** - exercise the complete local path and make external gaps explicit.

## Relevant tests
- Fast: `python -m pytest tests/unit/test_canonical_offline_reducer.py tests/unit/test_canonical_run_membership.py tests/unit/test_canonical_offline_call_part_concurrency.py`
- Broader: `python -m pytest tests/integration/test_canonical_offline.py tests/integration/test_canonical_local_command.py`

## Read alongside
Read every module named by the dispatched bridge. In practice, source/media changes start with `source-media`; QA changes start with `sampling-qa`; event semantics start with `event-semantics`; completion changes start with `identity-delivery`.
