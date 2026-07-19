# Track T1 Local Parallel Inference Smoke ? 2026-07-19

## Scope

This is a non-certifying local fake-model experiment for ADR 0006. It exercises only the
opt-in concurrent `QA_DENSE` + `ACTION_EVIDENCE` path. It does not parallelize video export or
frame materialization, does not connect a provider, and does not close any normative Architecture
V1.1 phase gate.

## Command

```powershell
uv run --locked python scripts/run_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/governance-parallel-smoke-medium-20260719 `
  --allow-unapproved `
  --parallel-independent-inference `
  --registry-root tmp/governance-parallel-smoke-medium-registry-20260719
```

## Result

- exit code: `0`
- run status: `PRIMARY_COMPLETE`
- fake inference attempts: `5`
- event count: `1`
- provider requests: `0`
- production eligible: `false`
- bundle SHA-256: `cfb6225364767f2e42a91d6f7e4f6cb264052745d5cb64335f7075a5dd2d220e`
- execution manifest SHA-256: `32208432d7dfd4fa03cc6ee6a230ac08c5ea870dd95cfb5d36779cf628a33755`
- execution semantic SHA-256: `55706e26c9ebba75fcc68da70875db8c8ff657cdabd83c201b20652e10937130`

The published root passed `scripts/verify_local_mainline.py`. The parallel flag is an execution
strategy and is excluded from the mainline semantic projection; the unit suite also proves that
serial and opt-in parallel runs preserve canonical request ordering, outcomes, event semantics,
and bundle bytes.

## Limitations

- No wall-time speedup claim is made; the source is a local development fixture and no benchmark
  corpus or resource instrumentation is registered.
- `BOUNDARY_REFINEMENT` remains serial by dependency.
- Export/materialization and distributed tracks remain unimplemented.
- Normative Phase 0/1B/2 gates, O-01/O-03/O-04/O-10, provider approval, and real-model quality
  evaluation remain open.
