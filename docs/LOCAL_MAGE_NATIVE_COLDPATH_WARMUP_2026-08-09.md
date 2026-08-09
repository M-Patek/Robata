# Robata Mage Native DCVC Provider V2 Qualification — August 9, 2026

## Executive decision

Robata should adopt **Provider V2 with `max_side=448` as the local and pre-production
single-camera preparation profile**, while retaining `observed-v1` as an explicit rollback path.
The decision is based on one controlled 40-second, five-segment A/B set:

- cold native-DCVC preparation fell from **178.146 s to 50.310 s**;
- the reduction is **71.76%**, or a **3.541x speedup**;
- every non-metadata codec asset was byte-identical across `observed-v1`, Provider V2 full
  resolution, and Provider V2 `max_side=448` for this sample;
- all five raw Mage outputs were byte-for-byte semantically equal after parsing;
- the same six event tracks, fusion dispositions, and refinement requests were produced;
- the endpoint consumed the exact admitted Provider V2 directories without invoking
  `_run_dcvc_rt` again.

This is an **adoption of a local/pre-production candidate**, not production qualification.
`production_eligible` remains `false`. The evidence is one camera, one RTX 4060 Laptop GPU, one
preparation worker, one generation lane, one recording, and no model-backed refinement calls.
Linux/H100, multi-camera quality, sustained capacity, and fault-injected production recovery still
require target-side evidence.

## What this cycle changed

The old observed path had two independent problems:

1. Robata recorded requested DCVC controls, but upstream Mage did not propagate all of them to the
   child readiness process. In particular, a requested `max_side=448` could coexist with an
   effective child configuration of `max_side=0`.
2. The old preparation path started a child and loaded DCVC state per segment, which made cold
   preparation dominate the local run.

Provider V2 replaces that ambiguity with an internal, versioned operational contract:

```text
immutable segment bytes
        |
        v
mage-dcvc-provider-job-v2
        |
        v
one resident preparation worker
        |
        v
worker-authored receipt + exact assets
        |
        v
cache namespace v2
        |
        v
strict endpoint admission
        |
        v
exact provider directory consumed by Mage processor
```

The effective configuration is installed before the provider modules are imported. The worker
reports what actually ran, not merely what the parent requested. A cache entry binds source bytes,
qualified checkpoint, provider implementation, effective configuration, receipt, and every output
asset. Endpoint admission re-verifies the entry on each request and passes the exact admitted
provider directory to the runtime. The runtime loads that directory through Mage's native codec
loader and has no on-demand DCVC fallback for a Provider V2-bound request.

No published JSON Schema changed. These are internal operational contracts and identities. The
original Mage checkpoint and observed cache were not rewritten.

## Bound experiment

| Dimension | Bound value |
| --- | --- |
| Date | August 9, 2026 |
| Host | Windows 11 local workstation |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB |
| Qualified model | `Mage-VL-Robata-DCVC-V2` |
| Qualified revision | `local-2026-08-09-r4` |
| Decoder profile | `bitsandbytes_4bit_nf4_v1` |
| Recording | persisted `cam_01` native MP4 |
| Recording exact SHA-256 | `4337aefbc597a28fa97c10f17ea24555ad03f694b10d3710ac5f96022a565b47` |
| Recording interval | exactly 0–40 s |
| Stream plan semantic SHA-256 | `2e9213d9921ca957f4f30e3838d74f9b0261d499f135b40383fd5f4ea30b3f45` |
| Segmentation | five non-overlapping, keyframe-aligned segments |
| Reasoning horizon | one focus segment per observation, approximately 8 s |
| Cameras | one, `cam_01` |
| Preparation workers | one |
| Generation workers/lanes | one / one |
| Normal Mage calls | five |
| Executed refine calls | zero |
| Decoder budget | 256 new tokens maximum |
| Qwen residency | none during these measurements |

The exact recording key is `mage-native-sustained-control-20260808`. A different recording key
produces different segment identities even when materialized MP4 bytes happen to match. Strict
source-path and identity admission correctly rejects that substitution; the qualification did not
weaken it.

## Identity pins

| Identity | SHA-256 |
| --- | --- |
| Qualified checkpoint manifest semantic identity | `717566330171aa91a93fb3fc083342c744b842f93ed8190ed54190f48cf33283` |
| Qualified checkpoint manifest exact bytes | `0c1244826f3552de9f13e9e55e1ac126373b7c98bda29f70ef604ab83a58049e` |
| Qualified-provider manifest semantic identity | `ce8a884abf41636fa0c43933dc158bd476468f05752c2f19f404d0b64e7b84eb` |
| Qualified-provider manifest exact bytes | `0a479c5288630c412219c07f94ac29e195d5b9c19fb09c61ff49d690fea184d6` |
| Provider bundle semantic identity | `fb0ce911c498f976f3a2fdab0e1f21c8dbb9a3a1a8f058753da3bc599575d5ad` |
| Provider implementation identity | `8d39f941fb28ade60a56770bf7bf27b987215826849a10d3abfeb29ba0dc6dd2` |
| Provider V2 full-resolution effective config | `5bbeab0271cca60dfde7c9a9f043adf8be35ae715436e556fbf6be50cb4e49ea` |
| Provider V2 `max_side=448` effective config | `b147bc7bc9d0d723ac06f172104b753d7c7bd0a013241c8c5d9c1be274d230a0` |

`sequence_length_frames=0` is deliberate. A sampled-frame count is **not** a cap on recurrent DCVC
work. The provider advances the reference chain through the final sampled frame and reports the
actual maximum encoded frame ID. This report does not describe `seq_len_frames=8`, or any other
sampling parameter, as an eight-step compute limit.

## Cold preparation A/B

The measured wall includes provider process startup and DCVC model loading. The Provider V2 process/load counts are worker receipts; the observed-v1 five-start/five-load count is inferred from its audited one-child-per-segment topology because its retained manifest did not persist lifecycle counters.

| Variant | Effective `max_side` | Cold wall | Media / wall | Wall / media | Speedup vs observed | Wall reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| observed-v1 | 0 | 178.146 s | 0.225x | 4.454x | 1.000x | 0% |
| Provider V2 full | 0 | 144.179 s | 0.277x | 3.604x | 1.236x | 19.07% |
| Provider V2 bounded | 448 | **50.310 s** | **0.795x** | **1.258x** | **3.541x** | **71.76%** |

Provider V2 full resolution proves the effect of the persistent, explicit provider independently
of the resolution bound. Provider V2 `max_side=448` then varies only the effective bound.

The formal qualifier also reports a derived **cold total** as preparation aggregate wall plus the
fresh generation harness overall wall. These are retained sequential measurements, not a claim
that the two phases were observed in one process:

| Variant | Cold total | Speedup vs observed |
| --- | ---: | ---: |
| observed-v1 | 265.483 s | 1.000x |
| Provider V2 full | 235.078 s | 1.129x |
| Provider V2 448 | **145.913 s** | **1.819x** |

### Persistent-worker evidence

Both Provider V2 variants used:

- one worker process for all five jobs;
- one inferred DCVC engine load;
- one sequence reset per segment;
- no cross-segment DPB reuse;
- atomic publication followed by exact verification.

Provider V2 `max_side=448` job walls, retained by the companion worker report, were:

```text
9.318502 s
7.116746 s
7.232306 s
6.528999 s
7.211185 s
```

Their sum is 37.408 s. The cache-manifest wall was 50.310 s. The companion orchestration report recorded 50.512 s
through its slightly wider reporting boundary. The inferred engine-load cost was 0.781 s and occurred once.

Provider V2 full-resolution job walls were approximately 28.087, 25.899, 25.774, 25.762, and
25.775 seconds. Its cache-manifest wall was 144.179 s; the companion orchestration report used a
slightly wider boundary and recorded 144.383 s. The first job contains more one-time startup work; subsequent jobs reuse the same
loaded engine.

### Preparation telemetry

| Variant | Mean GPU utilization over full preparation wall | Peak GPU | Peak VRAM | Peak temperature | Peak power |
| --- | ---: | ---: | ---: | ---: | ---: |
| observed-v1 | 57.70% | 100% | 1,926 MiB | 79 C | 94.10 W |
| Provider V2 full | 78.96% | 100% | 1,926 MiB | 81 C | 103.87 W |
| Provider V2 448 | 19.00% | 91% | 1,454 MiB | 63 C | 56.81 W |

The lower mean utilization for the 448 run is not a regression. The run finishes much sooner and
spends a larger fraction of its retained wall in startup, IPC, and sampling boundaries. Its
absolute work, wall, memory, temperature, and power are all lower on this sample.

## Asset equivalence

A read-only, ACL-elevated hash audit compared, for every ordinal and every variant:

```text
canvas_000.jpg ... canvas_006.jpg
frame_ids.npy
src_patch_position.npy
```

Result:

```text
observed-v1 == Provider V2 full == Provider V2 max_side=448
for all 5 segments and all 9 non-metadata files
```

The `meta.json`, receipt, entry sidecar, logical cache identity, and namespace identity are
expected to differ. They bind different recipes and effective configurations. Exact payload
parity is therefore accompanied by correct identity isolation rather than accidental namespace
reuse.

This sample's final Mage codec payload was already 224x256 with seven canvases per segment. The
byte equality is strong evidence for this recording, but it must not be generalized to other
resolutions, motion patterns, or cameras without a larger labeled set.

## Real Mage generation A/B

Each variant started a fresh endpoint process, loaded the same R4 NF4 model, consumed its exact
prebuilt cache, and ran the same five observation requests. Codec preparation was excluded from
these runs. Model load is included in `overall wall`, but excluded from `Robata run wall` and the
per-request generation figures.

| Metric | observed-v1 | Provider V2 full | Provider V2 448 |
| --- | ---: | ---: | ---: |
| Overall wall including endpoint startup/model load | 87.337 s | 90.899 s | 95.603 s |
| Endpoint startup to health | 58.748 s | 67.017 s | 69.994 s |
| Model load reported by first request | 35.749 s | 37.336 s | 38.368 s |
| Stream wrapper wall | 28.301 s | 23.877 s | 25.604 s |
| Robata run wall | 24.843 s | **20.608 s** | 21.962 s |
| Media / Robata run wall | 1.610x | **1.941x** | 1.821x |
| Generation sum | 22.851 s | **18.959 s** | 19.958 s |
| First generation | 8.250 s | **6.880 s** | 7.671 s |
| Mean of generations 2–5 | 3.650 s | **3.020 s** | 3.072 s |
| Processor sum | 0.215 s | **0.141 s** | 0.312 s |
| Output tokens | 80, 41, 37, 39, 40 | same | same |
| Full-wall mean / peak GPU | 17.71% / 100% | 18.09% / 100% | 17.37% / 100% |
| Peak VRAM | 5,248 MiB | 5,304 MiB | 5,267 MiB |
| Peak temperature | 71 C | 73 C | 72 C |
| Peak power | 93.57 W | 94.62 W | 95.35 W |

The small hot-wall difference between Provider V2 full and 448 is run-to-run decoder/startup
noise, not a codec-quality signal: their loaded payload bytes and output texts are identical. The
cold-preparation result is the adoption signal.

### First request and cold start

The first generation is 2.3–2.5 times slower than the mean of requests 2–5. Two effects are
separate:

1. **Model loading:** 35.7–38.4 seconds. This is part of endpoint startup and overall wall, but it
   is not part of the Robata run wall or the per-request generation time.
2. **First authoritative generation:** 6.9–8.3 seconds. It has longer CUDA/kernel/cache warm-up and
   produces 80 output tokens, roughly twice the 37–41 tokens of later requests.

Once resident, one camera's 40 seconds of media completes in 20.6–25.0 seconds, or approximately
1.6–1.94x real time in these three controlled runs.

### GPU-duty interpretation

Full-wall utilization means are intentionally low because they include model loading, endpoint
health polling, process transitions, and telemetry sampling. They do not mean the decoder runs at
18% while generating. Peak utilization reaches 100%, and generation accounts for about 91–92% of
the Robata run wall in this A/B.

The remaining processor work is already small: 0.14–0.31 seconds over five requests. On the same
8 GiB GPU, adding CPU/CUDA preparation overlap would not materially improve this exact-cache hot
path and could violate the shared-device safety rule. The next hot-path levers are decoder output
length, model/runtime kernels, and target hardware—not another in-request codec pipeline.

## Raw Mage output parity

The five parsed output documents are equal across all three variants:

1. `pick up green shirt`, 2–3 s, followed by `fold green shirt`, 3–6 s;
2. `folding green shirt`, 0–7 s;
3. `holding green shirt`, 0–8 s;
4. `folding green piece of clothing`, 0–8 s;
5. `wiping table with green cloth`, 0–7 s.

The exact output-token counts are also equal: 80, 41, 37, 39, and 40.

## Downstream semantic parity

All three variants produced the same sorted event tracks:

| Action | Absolute interval |
| --- | --- |
| `pick_up_a_green_shirt` | 2–3 s |
| `fold_a_green_shirt` | 3–6 s |
| `a_person_in_a_white_shirt_is_folding_a_green_shirt` | 8.000009–15.000009 s |
| `a_person_is_holding_a_green_shirt` | 15.999996–23.999996 s |
| `a_person_in_a_white_shirt_is_folding_a_green_piece_of_clothing` | 24.000003–32.000003 s |
| `a_person_in_a_white_shirt_is_wiping_a_table_with_a_green_cloth` | 32.000005–39.000005 s |

QA projections, event hypotheses, evidence projections, event tracks, fusion dispositions, and
refinement requests have exact semantic parity under their comparison projections.

Every final fusion decision is still ambiguous because this was intentionally a one-camera run:

- one observable/supporting camera rather than multi-view support;
- zero boundary reliability in the current compact output;
- `START_BOUNDARY_UNCERTAIN` and `END_BOUNDARY_UNCERTAIN`;
- `INSUFFICIENT_SUPPORT`, with some decisions also `LOW_CONFIDENCE`.

Six durable boundary-refinement requests were scheduled. None was executed by a model, by design.
The final state is therefore `PENDING_REFINEMENT`, and no production fact was published. This is a
known single-camera/refinement-boundary limitation, not a Provider V2 regression.

### Provider V2 448 stage cost

| Stage | Invocations | Accumulated elapsed |
| --- | ---: | ---: |
| `MEDIA_SCAN` | 5 | 0.320 s |
| `PERCEPTION_OBSERVE` | 5 | 20.686 s |
| `OBSERVATION_PROJECT` | 5 | 0.008 s |
| `TEMPORAL_RECONCILE` | 5 | 0.006 s |
| `FUSION` | 6 | 0.002 s |
| `PERCEPTION_REFINE` | 0 | 0 s |
| `FINALIZE` | 1 | <0.001 s |

The deterministic stages are not the hotspot. After exact codec pre-admission, the remaining
live-path cost is overwhelmingly Mage autoregressive generation.

## Adoption boundary

### Adopt now

Use the following as the local/pre-production single-camera candidate:

```text
provider_version = robata-mage-dcvc-provider-v2
recipe_version = mage-dcvc-readiness-explicit-v2
max_side = 448
sequence_length_frames = 0
one preparation worker
one generation lane
strict Provider V2 cache admission
exact provider-directory consumption
```

The tracked dual-H100 target profile also uses `max_side=448`, but remains target-only until run on
the actual Linux/H100 environment.

### Do not adopt as production evidence

This cycle does not qualify:

- six-camera or multi-view perception;
- two-H100 runtime behavior;
- Linux containers, RunPod, R2, or PostgreSQL/Supabase participation;
- native BF16 output/performance;
- model-backed refinement application;
- labeled recall, boundary accuracy, or calibration;
- restart, cache loss, and authoritative replay under target failure injection;
- sustained 500 recording-hours/day capacity.

### Rollback

Rollback is configuration-only and non-destructive:

1. stop admitting new Provider V2 cache entries;
2. launch with the explicit observed-v1 model/policy/cache composition;
3. remove `--require-provider-v2-cache` only for that explicit rollback composition;
4. retain `--require-verified-codec-cache`;
5. do not delete, rewrite, or rehash either cache family;
6. start a new run identity rather than reusing Provider V2 durable work.

Strict admission must never silently fall back from a missing Provider V2 entry to dynamic DCVC
recomputation.

## Dual-H100 target composition

The tracked target separates the two expensive roles:

```text
H100 physical GPU 0: serial Provider V2/DCVC preparation
H100 physical GPU 1: resident Mage BF16 decoder
```

It preserves one camera, one endpoint worker, and one generation lane. This avoids placing DCVC
and Mage generation on the same device while keeping the architecture ready for later horizontal
scaling. The configuration is intentionally labeled `TARGET_CONFIGURATION_UNVALIDATED` and
`production_eligible=false`.

See:

- `config/mage-h100-bf16-single-worker-v1.json`;
- `scripts/render_mage_endpoint_profile.py`;
- `docs/operations/mage-h100-bf16-single-worker.md`.

## Evidence inventory

The machine-readable qualification is:

```text
docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json
exact SHA-256: 7298d21fb05f0ecbc4bc1e11481f67abf2c82b4b13380227177edfbbbaa24287
semantic SHA-256: ea659e3e78243e43e4c1f921ff0898c64f18c4e68993c9c219d2425c8a25b0d8
all declared gates: passed
```

The document records observed-v1 per-segment preparation timing as `UNAVAILABLE`; the old retained
manifest contains only the total cold wall. It does not fabricate an even split or reconstruct
per-segment timing from coarse GPU samples.

The report can be reproduced from the retained, already-completed artifacts without starting a GPU
process:

```powershell
.\.venv\Scripts\python.exe scripts\qualify_local_mage_dcvc_provider_v2.py `
  --observed-preparation-dir D:\Github\Robata\.local\e4obs2 `
  --observed-generation-dir D:\Github\Robata\.local\e4genobsc `
  --provider-v2-full-preparation-dir D:\Github\Robata\.local\e4full `
  --provider-v2-full-generation-dir D:\Github\Robata\.local\e4genfullc `
  --provider-v2-bounded-preparation-dir D:\Github\Robata\.local\e4b448 `
  --provider-v2-bounded-generation-dir D:\Github\Robata\.local\e4gen448c `
  --output docs\mage-dcvc-provider-v2-local-qualification-2026-08-09.json
```

### Cold preparation

| Variant | Manifest/report exact SHA-256 |
| --- | --- |
| observed-v1 cache manifest | `29a9e215301048547f84bef125e9c60759c459030ed7a91bd4b93dfce697d0f5` |
| observed-v1 GPU telemetry | `f6bf13ee5366cb05a50b5674e77b0dd22b3010037b7b7b046aa353375a57e610` |
| Provider V2 full cache manifest | `4434213cd6ab87d674ae5a2622f63a5660613db66a406766a450943a94cc8640` |
| Provider V2 full prewarm report | `f8a60781ba47c4da6dbd0e5b263e9839b7f8d4dea4d2110af51d55a357788b9d` |
| Provider V2 full GPU telemetry | `ef10e9a0bb522a6ff1eb6db4d9300524b81677faa42f3efdf2eb624fa4a4c9c3` |
| Provider V2 448 cache manifest | `f68c327279c1a055cae2e6fedeff33c86d272ecada231ccca250a2a9b62efda9` |
| Provider V2 448 prewarm report | `b0e68ab45aa354ce163be8a8584d685e4efb0b5f772bdd4f645b406937bc5ce8` |
| Provider V2 448 GPU telemetry | `31800145afd1fbf26d3d3cb417942ecff4449612f7d86cc1cb8b42d1a319f334` |

### Real Mage runs

| Variant | Harness exact SHA-256 | Stream report exact SHA-256 |
| --- | --- | --- |
| observed-v1 | `723c7830f13de7bfcd6ef5d89321118298ec4425116fd2cddb3348c1673e2fae` | `f3777caace44c83e9146a62d6406f5b6f995df6a34d9664967892351663c8c7f` |
| Provider V2 full | `7236227d62f31cdd8ce6ae98eee8a66410ad233f68f76f49b751140453480bbf` | `cfe550ac1da461f0f83ea5682a167bc79acf42f0d311ce8e5e002878e4c7dec0` |
| Provider V2 448 | `e88c9abd3554df5d8d1faffd3cb8b12a7ccaa0ebf3be6e050edd8bbad6830443` | `f012349e6bd584363347539d6a7bbf81917c59ba65aa4b2343eaf5095144237d` |

Retained local roots:

```text
D:\Github\Robata\.local\e4obs2
D:\Github\Robata\.local\e4full
D:\Github\Robata\.local\e4b448
D:\Github\Robata\.local\e4genobsc
D:\Github\Robata\.local\e4genfullc
D:\Github\Robata\.local\e4gen448c
```

These paths are local evidence locations, not portable production storage contracts.

## Final verdict

```text
LOCAL QUALIFICATION: PASSED
LOCAL/PRE-PRODUCTION CANDIDATE: PROVIDER V2 max_side=448
OBSERVED-V1: RETAIN AS EXPLICIT ROLLBACK
PRODUCTION ELIGIBLE: FALSE
NEXT REQUIRED ENVIRONMENT: LINUX + DUAL H100 TARGET QUALIFICATION
```

The architecture has removed duplicate codec computation from the live request path and made the
cold preparation recipe explicit, durable, and fail-closed. For this sample, the largest measured
engineering gain is not more concurrency; it is one persistent Provider V2 worker plus an exact
prebuilt `max_side=448` cache. The remaining hot-path cost is the Mage decoder itself.
