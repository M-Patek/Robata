# Qualification and Operations

## Scope and path anchors
- Runtime and capacity: `src/robata/runtime/**` - start with `capacity.py`, `canonical_profile.py`
- Benchmarking: `src/robata/benchmark/**` - start with `metrics.py`
- Relevant test families: `test_runtime_*.py`, `test_benchmark*.py`, release/documentation checks

## How to dispatch
`qualification-ops / P<n> - <timing, capacity, benchmark, smoke, hygiene, or release-evidence task>`

## Construction phases
1. **Measure** - add span timing and make units/input mode/hardware explicit.
2. **Local capacity** - benchmark fixtures, mocks, workers, storage, and decode throughput.
3. **Qualification** - add representative data, real providers, and long-run evidence when available.
4. **Release hygiene** - verify clean builds, tracked artifacts, documentation, and reproducible commands.

## Relevant tests
- Fast: `python -m pytest tests/unit/test_runtime_capacity.py tests/unit/test_canonical_profile.py tests/unit/test_benchmark.py`
- Broader: `python -m pytest tests/unit/test_local_streaming_smoke.py tests/unit/test_local_streaming_benchmark.py`

## Read alongside
Read `canonical-integration` for end-to-end timing boundaries, `source-media` for decode/materialization, and `inference-evidence` for provider/ledger costs. Report recording-hours and camera-hours separately.
