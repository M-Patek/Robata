# Robata 25? Route Decision ? 2026-08-09

> **Authority:** local, non-production evidence on one RTX 4060 Laptop GPU. This document does not claim H100 capacity or production readiness.

## Executive decision

- **Mage vNext:** retain as the architectural default and continue to hold/qualify it; do not promote it to a 25? production route from the current local evidence.
- **Qwen Batch4 hybrid:** activate as the versioned local hedge and integration candidate:
  - one-claim groups use one native `generate_batch` call;
  - multi-claim groups remain explicitly serial;
  - no hidden serial fallback after a native-batch failure;
  - rollback is the unchanged serial Qwen binding.
- **Production:** **HOLD**. The local results prove implementation and replay behavior, not production capacity.

The long-running PowerShell incident was an **outer orchestration failure**, not a one-hour model inference. The bounded runner now owns the child process, applies hard deadlines, reaps the process tree, and reports timeout separately from model failure.

## Fixed local comparison

| Item | Value |
|---|---:|
| Source duration | 40.0 s |
| Common comparison | `cam_01`, five non-overlapping segments |
| Hardware | RTX 4060 Laptop, 8,188 MiB |
| Worker / generation lanes | 1 / 1 |
| Target | 25? sustained aggregate real-time factor (500 camera-hours/day + 20% margin) |

## Measured route evidence

| Route | Recurring wall | RTF | Local lanes for 25? | Quality / admission |
|---|---:|---:|---:|---|
| Mage native DCVC v2 | 21.962 s stream wall | 1.821? | 14 | codec variant locally qualified; production hold |
| Mage traditional H.264/HEVC + 8?131K | 24.263 s hot wall | 1.248? | 21 | HOLD: green-book object hallucination |
| Mage temporal memory | 21.112 s | 1.895? | 14 | REJECT: repeated-action semantic collapse |
| Qwen v2 serial common | 31.070 s | 1.287? | 20 | 5/5 strict parse and downstream pass |
| Qwen v2 Batch4 common | 19.068 s | 2.098? | 12 | 5/5 strict parse and downstream pass; 4/5 semantic exact to serial |
| Qwen r12 Batch4 hybrid (QA-only) | 65.588 s median full control | 3.728? conservative | 7 | 51/51 normalized exact to serial; not full-pipeline qualification |

### Mage bottleneck conclusion

The native path is not decoder-only. DCVC preparation remains a material recurring cost (`37.408 s` worker sum and `50.310 s` full preparation wall for the fixed sample). Traditional H.264/HEVC removes most preparation cost, but transfers the bottleneck to autoregressive generation and exposes semantic quality failures. Temporal memory is not an acceptable speed shortcut because its local gain comes with action collapse.

The first generation is a cold/long-output outlier; the Mage local baseline measured `7.671 s` for the first generation and about `3.072 s` warm mean. Model load is recorded separately and is not included in the recurring stream wall.

### Qwen Batch4 evidence

The fresh common v2 comparison passed the structural quality gate in both modes:

- serial: 5/5 strict parse, downstream recomputation enabled;
- Batch4: 5/5 strict parse, downstream recomputation enabled;
- recurring speedup: `1.629?`;
- physical generation speedup: `1.647?`;
- raw and semantic exactness versus the serial run: `4/5`.

The real endpoint smoke used four frozen single-claim QA requests:

- first pass: one physical `generate_batch`, `5.607 s` wall;
- replay: zero additional generation, `0.142 s` wall;
- raw and normalized parity: `4/4` (raw parity is now an admission gate);
- endpoint batch rows: `4`; evidence intents: `4`; raw provider responses: `4`;
- child reaped and GPU returned to approximately `614 MiB` used.

## Capacity interpretation

The local-equivalent lane calculation is:

```text
ceil(25 / measured_realtime_factor)
```

For the original 500 camera-hours/day target with 20% margin, the corresponding daily target is 600 camera-hours/day. The lane counts above are **not** H100 worker counts. They are only a way to expose whether a route is structurally close to the requested throughput.

The current local evidence does **not** justify saying that one H100 or two H100s will meet the target. The next required measurement is a sustained Linux/H100 run with the same identities, output contract, replay semantics, and labeled quality gate.

## Operational route and rollback

### Local default candidate

```text
local-qwen-task-claim-group-hybrid-batch-v1
  single claim group -> NATIVE_BATCH_GENERATE_V1 (max batch 4)
  multi claim group  -> SERIAL_GENERATE_V1
```

### Rollback

Select the unchanged serial Qwen binding. Do not rewrite existing artifacts, cache entries, evidence rows, or idempotency namespaces. Batch and serial namespaces are isolated, so rollback is a route-selection change rather than a data migration.

### Production admission checklist

1. Representative labeled semantic quality passes for Mage and Qwen on the real task distribution.
2. Linux H100 BF16/NF4 sustained capacity and thermal measurements.
3. Full QA ? event ? evidence ? track ? fusion ? publication path, not QA-only.
4. Canary and shadow traffic with bounded backpressure and restart/replay proof.
5. RunPod/Supabase/R2 production adapters participate in the trace and are independently verified.

## Evidence hashes

The machine-readable companion report is `docs/robata-25x-route-decision-2026-08-09.json`. It records the exact SHA-256 of every retained report and local real-run artifact used above. Its semantic SHA-256 is stored inside the report itself.
