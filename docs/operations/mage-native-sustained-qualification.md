# Mage Native Sustained-Utilization Qualification ? August 8, 2026

## Verdict

The bounded-prefetch Mage-native route is accepted for **local steady-state
qualification** on the tested RTX 4060 Laptop GPU:

- warm-cache comparison: **PASSED** all local gates;
- cold-cache comparison: **FAILED** the real-time/duty-cycle/gap gates, while still
  passing compatibility, freshness, single-generation, output validity, output parity,
  token-budget, and speedup gates;
- evidence class: `LOCAL_CONFORMANCE`;
- `production_eligible=false` for both comparisons.

The result demonstrates that the runtime now keeps the single Mage generation lane much
closer to continuously occupied without introducing concurrent generations. It does not
claim that cold neural-codec preparation is real-time on an 8 GB laptop GPU.

## Fixed test identity

| Item | Value |
|---|---|
| Date | August 8, 2026 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB |
| Model | `Mage-VL`, local revision |
| Load profile | `bitsandbytes_4bit_nf4_v1` |
| Checkpoint manifest SHA-256 | `88cbc3b3e60cc9277449dfa5a02943834a78634e120e33a4cf6b5db2b4242416` |
| Source | `cam_01.mp4`, 20,980,510 bytes |
| Source SHA-256 | `4337aefbc597a28fa97c10f17ea24555ad03f694b10d3710ac5f96022a565b47` |
| Plan SHA-256 | `2e9213d9921ca957f4f30e3838d74f9b0261d499f135b40383fd5f4ea30b3f45` |
| Media interval | 0?40.0 seconds |
| Segments | five non-overlapping 8-second contexts |
| Camera / worker | one camera, one worker |
| Generation concurrency | exactly one |
| Decoder budget | 256 new tokens |
| Codec | native neural/DCVC, CUDA preprocessing, canvas 8, group 8, 65,536 max pixels, max side 448 |
| Model load | endpoint-preloaded; excluded from runner wall time |

The two endpoint loads in the valid warm pair were 33.47 s and 33.50 s. They are
reported separately and are not hidden inside the 22.80 s / 20.03 s run walls.

## Warm steady-state comparison

Both arms used fresh endpoint idempotency/result state and the same already-populated
codec cache. The serial arm used queue depth 1; bounded prefetch used queue depth 2.

| Metric | Serial native | Bounded prefetch | Change |
|---|---:|---:|---:|
| Wall time | 22.796 s | 20.034 s | **1.138? speedup** |
| Media / wall RTF | 1.755? | **1.997?** | +13.8% |
| Generation duty cycle | 93.47% | **97.14%** | +3.67 pp |
| Generation gap p95 | 266.82 ms | **5.22 ms** | ?98.0% |
| Mean GPU utilization | 46.20% | **54.89%** | +8.69 pp |
| Peak GPU utilization | 100% | 100% | unchanged |
| Output tokens / generation second | 11.123 | **12.178** | +9.5% |
| Output tokens / wall second | 10.396 | **11.830** | +13.8% |
| TTFT p50 | 396 ms | **390 ms** | essentially equal |
| TTFT p95 / max | **1.623 s** | 1.749 s | first-request variance |
| Processor?generation overlap | 0 | 71.83 ms | bounded overlap proven |
| Peak VRAM | 4,945 MiB | 4,973 MiB | +28 MiB |
| Peak VRAM fraction | 60.39% | 60.74% | safe on 8 GB |
| Mean power | 52.66 W | 55.57 W | expected higher occupancy |
| Peak temperature | 71 ?C | 73 ?C | within observed host range |

Machine report:

- content semantic SHA-256:
  `152c827c3aeb37a5e1734f1b856ec774c0bc20ccc432a68af6aeb1ca8aa1f53e`
- exact report-file SHA-256:
  `de70d4f14bf9cb3bf3b1f1907b67854f80400a5c8246494e32367c2b111d487c`
- local path:
  `D:\Github\Robata\.local\mage-native-sustained-20260808\warm-comparison-v2.json`

All gates passed, including fresh 5/5 generations, no replay rows, no overlapping
generation intervals, exact output-text hash parity, zero invalid outputs, and zero
256-token budget exhaustion.

## Cold codec-cache comparison

The cold prefetch arm used a new empty codec-cache directory. The serial control showed
the same 30?32 s per-segment cache-miss processor behavior. This comparison measures
first-pass neural-codec cost rather than steady-state inference.

| Metric | Serial native | Bounded prefetch | Change |
|---|---:|---:|---:|
| Wall time | 170.049 s | 158.719 s | **1.071? speedup** |
| Media / wall RTF | 0.235? | **0.252?** | still below real-time |
| Generation duty cycle | 11.99% | **15.35%** | codec work dominates wall |
| Generation gap p95 | 29.663 s | **26.565 s** | still far above target |
| Mean GPU utilization | 73.57% | **79.46%** | +5.89 pp |
| Processor?generation overlap | 0 | **19.822 s** | overlap proven under real codec work |
| Peak VRAM | 5,788 MiB | 5,798 MiB | no OOM; ~70.8% of 8,188 MiB |
| Mean power | 70.86 W | 76.16 W | higher sustained occupancy |
| Peak temperature | 81 ?C | 82 ?C | thermal pressure visible |
| TTFT p95 / max | 1.134 s | 1.204 s | decoder not the cold bottleneck |

The strict cold report is `FAILED` only because its generation duty cycle, wall RTF,
and generation-gap p95 miss the steady-state acceptance thresholds. The prefetch speedup,
compatibility, freshness, isolation, single-generation, valid-output, output-parity, and
budget gates all pass.

Machine report:

- content semantic SHA-256:
  `ac79094c33dd6743645773be632c6713a083977e8e85efaec7230aa0063d0dd1`
- exact report-file SHA-256:
  `6a7d9c959adf68e5bff23d463e377fb5447040e31530efe31841a08b210ea9e0`
- local path:
  `D:\Github\Robata\.local\mage-native-sustained-20260808\cold-comparison-v2.json`

## Quality and replay audit

Across serial cold, serial warm, prefetch warm, and prefetch cold:

- all five deterministic request identities recur as expected for the same logical work;
- every arm has five fresh result artifacts with distinct artifact identities;
- all five raw output texts are byte-identical across all four arms;
- output token counts are identical: `80, 41, 37, 39, 40` (237 total);
- no output reaches the 256-token ceiling;
- semantic downstream projections are identical across all arms:
  - event-track projection SHA-256:
    `e7362e4eee172eaeb16e021457337a7de392954ea453b3be77750263df7e8709`;
  - fusion projection SHA-256:
    `c538b7a136a2ff03c98591c64649a2b2fbf5f2ce3619579e103bb4b91d659843`;
  - refine-request projection SHA-256:
    `f07477cbce6cbdeb8ac904c6d4c88f51f04d17a10d768caf0907ca1ed4a1cf3b`.

The deterministic result is six closed event tracks, six fusion decisions, and six
boundary-refine requests. No refine model call was executed. All tracks remain
`production_eligible=false` because this qualification intentionally uses one camera and
the returned boundary confidences require refinement. That is a known evidence-quality
limitation, not a scheduling regression.

## Bottleneck conclusion

1. **Warm path:** autoregressive generation remains the dominant cost. Bounded prefetch
   removes almost all inter-generation scheduler/processor gap, but cannot accelerate the
   NF4 decoder itself.
2. **Cold path:** native neural-codec/DCVC preparation is the dominant bottleneck at
   roughly 30?32 s per segment. It keeps the GPU busy but prevents generation from being
   continuously resident.
3. **First segment:** the first generation is 7?9 s because it combines decoder warm-up
   with the longest 80-token output. Later 37?41-token generations are about 2.9?3.4 s
   warm. Model loading is a separate ~33.5 s endpoint startup cost.
4. **8 GB pressure:** depth 2 is safe in this test. Warm peak VRAM is ~4.97 GiB and cold
   peak is ~5.80 GiB; there is no evidence supporting deeper queues or concurrent
   generations on this host.

## Operational decision

Use `BOUNDED_PREFETCH_NATIVE_V1` as the local sustained profile, with:

- `max_inflight_observations=2`;
- `generation_concurrency=1`;
- `max_new_tokens=256`;
- Mage native codec/video as the only authority;
- no small encoder and no Qwen route in the default composition.

For production qualification, rerun the same evidence on the target GPU and storage
stack. Measure cold cache separately from warm steady state, and do not convert this local
single-camera result into a publication/production claim.
