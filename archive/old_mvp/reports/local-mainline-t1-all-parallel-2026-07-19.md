# Local Mainline T1 All-Parallel Smoke Report

- Date: 2026-07-19
- Source: `data/source/sample-medium.mcap`
- Scope: local fake-model path only; no real model/provider was connected
- Flags: `--parallel-video-export --parallel-frame-materialization --parallel-independent-inference`
- Verification: `scripts/verify_local_mainline.py` passed

## Invariants

```text
execution_mode = LOCAL_DEVELOPMENT_FAKE_MODEL
provider_requests = 0
production_eligible = false
run_status = PRIMARY_COMPLETE
fake_inference_attempt_count = 5
event_count = 1
```

## Determinism comparison

A serial replay used the same local registry and source after the all-parallel run. The two runs
produced the same execution semantic and audit hashes, event identity, and video manifest hash;
wall-clock report bytes differ as expected because stage durations are observational.

| Evidence | All-parallel | Serial replay |
|---|---|---|
| `execution_manifest_semantic_sha256` | `1ddd39b8fb9747ba6be6247f34c3f5b065db1ed87cdef839eb5be6ff668e7db4` | same |
| `execution_audit_sha256` | `8419de9916722eee48fe03dc9be6e20904b45f6710d66c18e518fe3080af8057` | same |
| video `manifest_sha256` | `6674e7e29d1e5a8ff72e6a461b630edae8f8da928450878715bb14c08394864b` | same |
| event ID | `6acca923-ce1f-6a91-cdc6-23a50a45a55b` | same |
| bundle SHA-256 | `72126a1fa08712165b85b183dd07b3c2641d90425eb848e249516d59839be962` | `87e2868a3c5c6343f70865424b8694fb68b154a31b5c14afe8a9f2416dc46fb6` |

The differing bundle SHA-256 is expected: the bundle contains observational stage-duration values.
No throughput or production-capacity claim is made by this smoke run.
