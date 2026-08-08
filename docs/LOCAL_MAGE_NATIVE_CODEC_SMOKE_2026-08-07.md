# Robata Mage Native Codec Local Qualification — August 7–8, 2026

## Verdict

Robata's local Mage stream-oriented path has now completed two progressively stronger checks
on the NVIDIA GeForce RTX 4060 Laptop GPU:

1. **August 7, 2026:** a two-second native-video smoke proved that the local Mage checkpoint,
   NF4 runtime, Mage codec processor, and one decoder generation can execute on this host.
2. **August 8, 2026:** a persisted 40-second v6 run completed five non-overlapping perception
   segments with **five normal Mage calls**, deterministic QA/event/evidence projection,
   temporal reconciliation, fusion, and typed refinement scheduling.

The August 8 result is the current local architecture baseline. It proves the new physical
path no longer repeats QA, event, evidence, and boundary generations over overlapping windows.
It does **not** prove six-camera production quality, cold codec-preprocessing capacity,
H100/RunPod capacity, R2/PostgreSQL integration, or production eligibility.

## Bound environment

| Item | Value |
| --- | --- |
| Observation dates | August 7–8, 2026 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB |
| NVIDIA driver | 610.62 |
| Python | 3.13.5 (Conda base) |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.13.0 |
| bitsandbytes | 0.49.2 |
| ffmpeg | 8.0.1 Windows essentials build |
| Mage checkpoint | `D:\HuggingFace\Mage-VL` |
| Runtime profile | `bitsandbytes_4bit_nf4_v1` |
| Codec mode | Mage neural codec / `dcvc-rt` |

Qwen was not resident during these checks. Its checkpoint was not deleted.

## August 7: first native-video smoke

The first smoke used a codec-preserving two-second cut from the persisted `cam_01.mp4` view
of the same local sample used by the earlier Qwen run.

| Input | Value |
| --- | --- |
| Parent camera SHA-256 | `4337aefbc597a28fa97c10f17ea24555ad03f694b10d3710ac5f96022a565b47` |
| Smoke segment | `D:\tmp\mage-neural-smoke-cam01-2s.mp4` |
| Segment SHA-256 | `99c6954132ca9197051eae80254b8f403ed72e9aa2c6c61e5837bf046a543271` |
| Container/video | MP4 / H.264 |
| Dimensions | 1600 × 1300 |
| Frames | 60 |
| Duration | 2.000074 seconds |
| Segment bytes | 1,083,864 |

The cut used ffmpeg stream copy (`-c copy`), so Robata did not create a decoded-and-reencoded
2×3 mosaic before Mage. A standalone processor call exercised Mage's
`videos=[...]`, `video_backend="codec"`, `engine="dcvc-rt"` path and produced codec-derived
`pixel_values`, `image_grid_thw`, and `patch_positions` tensors. The model then loaded under
NF4 and generated one response.

This smoke established compatibility, not throughput. The host lacked the optional DCVC CUDA
extension and required a local namespace shim because the Conda environment already contained
an unrelated top-level `src` package. Subsequent source review also showed that some per-call
DCVC sampling/downscale fields were not forwarded all the way into the bundled child process.
Therefore the historical preprocessing timing must not be used as a production codec-capacity
claim.

## August 8: full 40-second v6 stream baseline

### Immutable report identity

| Item | Value |
| --- | --- |
| Report | `D:\Github\Robata\.local\mage-full-v6-20260808-report.json` |
| Report SHA-256 | `ECF00A57CE623B7BE9F61D65FE0D8BD1ADE41B4E46DF68A4D8292BAC1CDF658B` |
| Checkpoint manifest SHA-256 | `a15e49d965e4ad61455ef02bb770b626755959a4b7aa46a140a342f2ed62e290` |
| Recording duration evaluated | 40.000 seconds |
| Non-overlapping segments | 5 |
| Selected cameras | 1 (`cam_01`) |
| Maximum in-flight observations | 2 |

The run reused already materialized native-video segments and an existing neural-codec cache.
It therefore measures the resident stream execution and generation path, **not** a cold rebuild
of every DCVC asset.

### Physical-call and wall-time result

| Metric | Measured value |
| --- | ---: |
| Model load wall time, excluded from run wall | 25.638 s |
| Stream run wall time after load | **32.518 s** |
| Input duration / run wall | **1.23× real time** |
| Normal `PERCEPTION_OBSERVE` calls | **5** |
| Model-backed `PERCEPTION_REFINE` calls | **0** |
| Event tracks | 9 |
| Fusion decisions | 9 |
| Typed refinement requests | 9 |
| Adapter diagnostics | 0 |

Endpoint generation time per segment was:

```text
18.510 s, 3.146 s, 3.746 s, 3.658 s, 3.011 s
```

The first call contains warm-up effects. The four steady-state generations processed each
approximately eight-second segment in about 3.0–3.75 seconds, or roughly 2.1–2.7 recording
seconds per wall second for the one-camera resident-generation portion.

The pipeline's accumulated `PERCEPTION_OBSERVE` elapsed value is 60.799 seconds. That number
is the sum of per-context worker wall intervals and includes overlapped waiting/work; it must
not be added to the 32.518-second end-to-end wall time. Queue depth two allowed preparation
and request work to overlap while ordered projection, tracking, and finalization remained
causal.

### Deterministic-stage cost

| Stage | Invocations | Accumulated elapsed |
| --- | ---: | ---: |
| `MEDIA_SCAN` | 5 | 0.336 s |
| `PERCEPTION_OBSERVE` | 5 | 60.799 s (overlapped worker sum) |
| `OBSERVATION_PROJECT` | 5 | 0.010 s |
| `TEMPORAL_RECONCILE` | 5 | 0.009 s |
| `FUSION` | 9 | 0.004 s |
| `PERCEPTION_REFINE` | 0 | 0 s |
| `FINALIZE` | 1 | <0.001 s |

The expensive physical surface is now one normal Mage observation per non-overlapping segment.
QA, event, and evidence remain separate logical products, but their projectors do not invoke
the model again.

### Persisted evidence

The run persisted:

- 5 exact endpoint result artifacts;
- 5 context manifests;
- 5 observations;
- 5 media-health reports;
- 5 QA projections;
- 5 event projections;
- 5 evidence projections;
- 5 temporal-reconcile artifacts;
- 5 per-context execution reports;
- 5 durable endpoint idempotency rows.

All 40 local perception-CAS filenames matched the SHA-256 digest of their exact bytes. The
endpoint SQLite database returned `integrity_check=ok` and zero foreign-key violations.
Artifact replay is separate from model recomputation: accepted raw bytes can be parsed and
projected again without invoking Mage or resolving the original media.

## Quality and admission boundary

The model produced concrete actions involving picking up, folding, placing, holding, and
wiping with a green shirt or cloth. This is evidence that the v6 prompt no longer merely
copies a fixed example action. It is not a labeled accuracy score.

All nine local fusion decisions were correctly **not production eligible**. Only one camera
participated, while the fusion policy requires sufficient observable/supporting cameras; the
result also emitted bounded refinement requests for boundary/support ambiguity. The local
runner intentionally schedules those requests without silently performing a second model
cascade. A production-safe refinement executor needs a versioned refine-result/provenance
contract before it can apply changes to event tracks.

## What is established

1. The local Mage checkpoint loads under the declared NF4 profile on the RTX 4060 Laptop GPU.
2. The native video/codec path executes against real persisted media.
3. A 40-second input is partitioned into five immutable, non-overlapping segments.
4. Each segment causes one normal model call; QA/event/evidence projections are deterministic.
5. Bounded producer/consumer execution overlaps work without losing ordinal reconciliation.
6. Exact raw artifacts, observation lineage, deterministic products, and idempotency state are
   persisted locally.
7. Qwen is not loaded by the Mage default path and its files remain preserved for explicit
   legacy use.

## Still not established

1. Six-camera codec encoding with one multi-view decoder generation.
2. Production-quality event recall, boundary accuracy, gate recall, or fusion calibration.
3. Cold neural-codec preprocessing throughput after all DCVC policy fields are correctly
   forwarded and profiled.
4. Automatic, provenance-safe application of targeted refinement results.
5. RunPod/H100 saturation, multi-worker capacity, endpoint-failure behavior, or long soak.
6. R2 object storage, PostgreSQL/Supabase canonical composition, RLS, backup/restore, or
   outbox publication under production faults.
7. The 500 recording-hours/day target.

The next qualification boundary is a six-camera/H100 run with production storage enabled,
representative labels, separate cold/warm codec timing, and explicit capacity/backlog metrics.
