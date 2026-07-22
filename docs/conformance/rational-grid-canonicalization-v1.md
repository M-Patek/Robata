# Rational-grid and canonicalization vectors V1

This local conformance suite implements the cross-language evidence requested by
[Architecture Sections 25.3 and 25.11](../../ARCHITECTURE_DESIGN_V1.md#253-exact-rational-sampling-grid).
The language-neutral fixture is
[`conformance/rational-grid-canonicalization-v1.json`](../../conformance/rational-grid-canonicalization-v1.json).

Every exact integer is encoded as canonical base-10 JSON text. This prevents a JavaScript
runtime from first losing precision through binary `Number` parsing. Each vector pins the RFC
8785 canonical bytes of its `input` as lowercase hexadecimal plus their exact SHA-256. Expected
outcomes separately pin rounded results, the reduced period, ordered targets, frame facts,
signed deltas, and selection status.

The five vectors cover:

- positive and negative `HALF_EVEN` ties;
- negative grid indices and lowest-`k` retention for duplicate rounded timestamps;
- clipping without resetting the persisted grid phase;
- inclusive tolerance, decode failure, and nearest-frame ties by aligned time, source time,
  then canonical locator bytes;
- one-use-per-source-frame dedupe, including the `(absolute delta, target time, k)` winner.

The Python verifier calls the authoritative implementations in
[`alignment/rational_time.py`](../../src/robata/alignment/rational_time.py) and
[`sampling/grid.py`](../../src/robata/sampling/grid.py). The Node runner is a dependency-free,
independent BigInt implementation; it does not import or shell out to Python.

```powershell
python scripts/verify_rational_grid_vectors.py
node scripts/verify_rational_grid_vectors.mjs
python -m pytest tests/conformance/test_rational_grid_vectors.py
```

Changing a vector input requires updating both canonical pins and the independently verified
outcome in a reviewed new suite version. This is local deterministic conformance evidence only;
it does not approve sampling rates/tolerances, source data, provider behavior, capacity, or a
production promotion decision.
