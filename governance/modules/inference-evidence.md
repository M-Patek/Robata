# Inference Evidence

## Scope and path anchors
- Core plans and orchestration: `src/robata/inference/**`
- Durable local ledger: `src/robata/adapters/sqlite_inference_evidence.py`
- Provider tests and adapters: `test_inference_*.py`, `test_sqlite_inference_*.py`, `test_runpod_adapter.py`

## How to dispatch
`inference-evidence / P<n> - <input-plan, provider, evidence-ledger, replay, or throughput task>`

## Construction phases
1. **Provider-neutral plans** - build stable requests, sampling purposes, and call plans.
2. **Evidence lineage** - persist intent, raw response, parsed artifact, selection, and replay facts.
3. **Provider adapters** - keep mocks useful locally; add real-provider behavior only behind explicit configuration.
4. **Throughput path** - batch/parallelize safely after timing identifies the actual bottleneck.

## Relevant tests
- Fast: `python -m pytest tests/unit/test_inference_input_plan.py tests/unit/test_sqlite_inference_evidence.py tests/unit/test_runpod_adapter.py`
- Broader: `python -m pytest tests/integration/test_canonical_offline.py`

## Read alongside
Read `sampling-qa` for call purposes and selected media, `contract-governance` for request/response shape changes, and `canonical-integration` for call ordering and replay.
