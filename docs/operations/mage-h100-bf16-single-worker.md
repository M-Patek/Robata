# Mage Provider V2 Dual-H100 BF16 Target

**As of:** August 9, 2026
**Tracked profile:** `config/mage-h100-bf16-single-worker-v1.json`
**Profile identity:** `mage-h100-bf16-single-worker-v1`
**Profile schema:** `mage-endpoint-deployment-profile-v2`
**Evidence class:** `TARGET_CONFIGURATION_UNVALIDATED`
**Production eligible:** **No**

The historical filename says `single-worker` because the serving side intentionally keeps one
Mage generation worker and one generation lane. The current target uses two H100 devices: one for
endpoint-external Provider V2/DCVC preparation and one for the resident Mage BF16 decoder.

This file is a target composition and command-rendering contract. It is not evidence that Linux,
native BF16, the DCVC CUDA extension, or either H100 has been exercised successfully. It does not
bind any local R-series path or digest as production evidence.

## Current local R4 evidence boundary

As of August 9, 2026, retained local R4 evidence compares observed-v1, Provider V2 full resolution,
and Provider V2 `max_side=448` on one 40-second, five-segment, single-camera recording. The bounded
candidate preserved exact non-metadata codec assets, parsed Mage outputs, and downstream semantic
projections for that sample while reducing cold preparation wall from 178.146 seconds to 50.512
seconds. The successful 448 harness still has authority `LOCAL_QUALIFICATION_NON_PRODUCTION` and
`production_eligible=false`.

The local/pre-production candidate therefore uses:

- `max_side=448`;
- `sequence_length_frames=0` and `canvas_token_side=null`;
- recurrent work through the last sampled frame;
- one resident provider worker, one inferred engine load, and five serial segment resets;
- one camera and one Mage generation lane;
- local `exclusive-shared-device-v1` placement on the RTX 4060.

The dual-H100 target keeps the 448 candidate but changes placement to `separate-device-v1`. Because
the concurrency policy is part of the effective configuration and cache namespace, the local R4
cache cannot be copied or admitted. The target must rebuild the 448 cache and prove a separate
all-hit replay on the actual two-H100 environment.

## Fixed topology and CUDA remapping

```text
outer CUDA selector 0                    outer CUDA selector 1
+-----------------------------+          +-----------------------------+
| Provider V2 DCVC worker     |          | Mage-VL BF16 endpoint       |
| process-local cuda:0        |          | process-local cuda:0        |
| one resident worker per     |          | one Uvicorn worker          |
| prewarm invocation          |          | one generation in flight    |
+--------------+--------------+          +--------------+--------------+
               | exact Provider V2 cache                | strict admission
               +----------------------------------------+

prewarm process:
  CUDA_VISIBLE_DEVICES=0,1
  preparation-device=cuda  -> prewarm-local cuda:0 -> outer selector 0
  generation-device=cuda:1 -> prewarm-local cuda:1 -> outer selector 1

endpoint process:
  CUDA_VISIBLE_DEVICES=1
  decoder cuda:0           -> endpoint-local cuda:0 -> outer selector 1
```

The values `0` and `1` are outer CUDA-visible selectors evaluated before each rendered process
applies its own `CUDA_VISIBLE_DEVICES` remapping. They are not physical-device evidence. The
renderer therefore emits:

```text
device_mapping_authority = OUTER_SELECTORS_UNATTESTED
device_uuid_attestation_required = true
```

The container runtime must expose exactly the intended two H100 devices in this order. Before
calling the mapping verified, record GPU UUID, PCI bus ID, model name, and memory size for:

1. each outer selector before remapping;
2. prewarm-local `cuda:0` and `cuda:1`;
3. endpoint-local `cuda:0`.

| Dimension | Fixed target |
|---|---|
| Codec selector | outer CUDA ordinal `0`; UUID attestation required |
| Decoder selector | outer CUDA ordinal `1`; endpoint-local `cuda:0`; UUID attestation required |
| Device policy | `separate-device-v1` |
| Provider processes | 1 |
| Provider lifetime | one resident worker across all videos in one prewarm invocation |
| Endpoint Uvicorn workers | 1 |
| Generation concurrency | 1 |
| Native batch | disabled; maximum size 1 |
| Camera inputs per request | 1 |
| Model profile | `production-native-bf16` |
| Cache admission | exact Provider V2 manifest only |
| Model warm-up output | `NON_AUTHORITATIVE_DISCARDED` |
| Exact artifact replay | authoritative |
| Attention backend | `runtime-default` |

This is not a high-concurrency or multi-camera profile. Faster hardware changes placement, not the
logical contract. Multi-camera fan-in, batching, or additional generation lanes require a new
profile and new semantic and capacity evidence.

## Selected 448 candidate and recurrent-work semantics

The target carries the locally selected 448 candidate into H100 qualification:

```text
resolution_policy = max-side-448-target-candidate
bounded_resolution_selected = true
bounded_resolution_production_qualified = false
max_side = 448
```

The local single-camera output and downstream semantic gates selected 448 for pre-production use.
Production promotion still requires the target-side H100, multi-camera, recovery, and capacity gates;
speed alone is insufficient.

Provider V2 also binds:

```text
sampled_frame_count = 64
sequence_length_frames = 0
sequence_length_frames_is_compute_cap = false
canvas_token_side = null
encoded_frame_extent = through-last-sampled-frame
```

`sequence_length_frames=0` does not mean zero frames, eight frames, or a finite recurrent compute
limit. Sampling and readiness grouping choose materialized observations, while recurrent DCVC work
continues from frame zero through the largest sampled frame. Reports must not describe the sampled
frame count as a temporal compute cap.

## Why the target has no shared-device guard

With `CUDA_VISIBLE_DEVICES=0,1`, prewarm preparation uses local `cuda:0` and the generation identity
uses local `cuda:1`. The prewarm launcher derives:

```text
device_concurrency_policy = separate-device-v1
```

The endpoint exposes only outer selector `1`. Preparation on the UUID attested to selector `0` and
generation on the UUID attested to selector `1` may overlap without a same-GPU exclusion file. The
renderer intentionally omits `--shared-device-guard-file`.

If both roles are ever moved onto one accelerator, this profile is invalid. Use a separately
versioned `exclusive-shared-device-v1` profile and cooperative guard instead.

## Required bindings

All rendered paths must be normalized absolute Linux paths. The renderer treats the provider
state root, codec cache root, and endpoint state directory as three mutable output roots and requires
them to be path-disjoint. Cache-manifest and provider-prewarm report files must be strict descendants
of the provider state root; generation telemetry and the warm-up report must be strict descendants of
the endpoint state directory. Output files must not alias one another. Neither an output root nor an
output file may overlap the qualified model tree, identity manifests, warm-up prompt, durable input
roots, or prewarm videos. These lexical container-path checks fail closed before any command is
rendered; deployment mounts must preserve the same separation and must not reintroduce aliases via
symlinks or bind mounts.

### Qualified identity

- qualified Provider V2 model directory;
- immutable qualified model revision;
- canonical qualified-provider manifest and exact file SHA-256;
- qualified checkpoint manifest and its bound manifest SHA-256.

### Provider V2 preparation

- durable provider state root;
- durable cache base root;
- Provider V2 cache-manifest output path;
- prewarm-report output path;
- every exact video segment, using repeated `--prewarm-video` arguments.

### Endpoint

- durable endpoint state directory;
- one or more approved durable input roots;
- generation telemetry JSONL;
- a warm-up video that is also present in `--prewarm-video`;
- exact warm-up video SHA-256;
- warm-up prompt and durable non-authoritative warm-up report paths.

### Network boundary

The launcher defaults to `127.0.0.1`. A wildcard bind such as `0.0.0.0` or `::` fails before model
loading unless the operator supplies exactly one of these declarations:

- `--network-boundary controlled-private-network`;
- `--network-boundary authenticated-reverse-proxy`;
- `--allow-unauthenticated-public-bind`, an explicit high-risk acknowledgement.

The tracked H100 profile uses `0.0.0.0` only with
`--network-boundary controlled-private-network`. This is an operator assertion about Pod/container
ingress, firewall rules, and routing; it does **not** claim that the Mage endpoint implements
authentication. Keep the endpoint off the public Internet and restrict ingress to the Robata
production caller. If an authenticated reverse proxy is used instead, it must be independently
configured and verified before changing to that declaration. The tracked profile intentionally does
not render the high-risk public-bind acknowledgement.

Secrets, object-store credentials, database credentials, and mutable deployment state stay outside
the tracked profile.

## 1. Build a target-qualified Provider V2 model tree

Never modify the source Mage directory in place.

```bash
python scripts/build_qualified_mage_dcvc_provider_v2.py \
  --source-model-dir /workspace/models/Mage-VL \
  --source-checkpoint-manifest /workspace/identity/source-checkpoint-manifest-v2.json \
  --target-model-dir /workspace/models/Mage-VL-Robata-DCVC-V2 \
  --qualified-model-identifier Mage-VL-Robata-DCVC-V2 \
  --qualified-model-revision <immutable-release-revision> \
  --qualification-manifest /workspace/identity/qualified-provider-v2.json \
  --checkpoint-manifest /workspace/identity/qualified-checkpoint-manifest-v2.json \
  --copy-mode copy
```

Retain the exact qualification-manifest file SHA-256 and the qualified checkpoint manifest SHA-256.
The renderer consumes these pins but does not create or bless the qualified tree.

## 2. Render pre-admission and endpoint commands

This example admits five non-overlapping segments. The first is also the endpoint warm-up source.

```bash
mkdir -p /workspace/state/provider-v2 /workspace/state/mage-endpoint

python scripts/render_mage_endpoint_profile.py \
  --profile config/mage-h100-bf16-single-worker-v1.json \
  --codec-cuda-selector 0 \
  --decoder-cuda-selector 1 \
  --qualified-model-dir /workspace/models/Mage-VL-Robata-DCVC-V2 \
  --qualified-model-revision <immutable-release-revision> \
  --qualified-provider-manifest /workspace/identity/qualified-provider-v2.json \
  --qualification-manifest-sha256 <64-lowercase-hex> \
  --checkpoint-manifest-path /workspace/identity/qualified-checkpoint-manifest-v2.json \
  --checkpoint-manifest-sha256 <64-lowercase-hex> \
  --provider-state-root /workspace/state/provider-v2/worker \
  --cache-base-root /workspace/cache/provider-v2 \
  --codec-cache-manifest /workspace/state/provider-v2/cache-manifest-v2.json \
  --provider-prewarm-report-json /workspace/state/provider-v2/prewarm-report-v2.json \
  --prewarm-video /workspace/data/segments/000000.mp4 \
  --prewarm-video /workspace/data/segments/000001.mp4 \
  --prewarm-video /workspace/data/segments/000002.mp4 \
  --prewarm-video /workspace/data/segments/000003.mp4 \
  --prewarm-video /workspace/data/segments/000004.mp4 \
  --state-dir /workspace/state/mage-endpoint \
  --durable-input-root /workspace/data/segments \
  --generation-telemetry-jsonl /workspace/state/mage-endpoint/generation.jsonl \
  --warmup-video /workspace/data/segments/000000.mp4 \
  --warmup-video-sha256 <64-lowercase-hex> \
  --warmup-prompt-file /run/secrets/mage-warmup-prompt.json \
  --warmup-report-json /workspace/state/mage-endpoint/warmup-report.json \
  --readiness-only \
  --render-phase all \
  --output-format posix-shell \
  > /workspace/state/mage-endpoint/launch-readiness.sh
```

The complete script has this shape:

```bash
set -euo pipefail
env CUDA_VISIBLE_DEVICES=0,1 python scripts/prewarm_local_mage_dcvc_provider_v2.py ...
exec env CUDA_VISIBLE_DEVICES=1 python scripts/run_mage_video_endpoint.py ...
```

The prewarm command pins the target candidate `max-side=448`, canvas/group policy, QP/reset/intra values,
readiness controls, bit-cost policy, manifests, and output paths. The endpoint command includes:

```text
--qualified-provider-manifest
--codec-cache-manifest
--require-verified-codec-cache
--require-provider-v2-cache
--network-boundary controlled-private-network
--neural-max-side 448
--neural-sequence-length-frames 0
```

`--neural-canvas-token-side` is intentionally omitted, leaving the value `null`. No observed-v1
fallback or shared-device guard is rendered.

The renderer shell-quotes every argv item. It rejects control characters, non-normalized Linux
paths, duplicate video paths, selectors other than the fixed `0` and `1`, and a warm-up source that
was not included in pre-admission.

## 3. Run sequential bootstrap or independently supervised phases

For a fixed evaluation set, execute the complete script. `set -e` prevents endpoint launch after a
prewarm or manifest-publication failure.

```bash
bash /workspace/state/mage-endpoint/launch-readiness.sh
```

For continuing arrivals, render the phases separately:

```bash
# Supervised preparation process on outer selector 0.
python scripts/render_mage_endpoint_profile.py ... \
  --render-phase prewarm --output-format posix-shell \
  > /workspace/state/provider-v2/prewarm-batch.sh

# Supervised endpoint process on outer selector 1.
python scripts/render_mage_endpoint_profile.py ... \
  --render-phase endpoint --output-format posix-shell \
  > /workspace/state/mage-endpoint/endpoint.sh
```

A single-phase shell command begins with `exec env` so process supervisors receive correct exit and
signal behavior. In the combined script, prewarm cannot use `exec` because the endpoint follows it;
the endpoint is the final `exec` process.

Each prewarm invocation uses one resident worker for every supplied video and loads the engine once
per invocation. This is bounded invocation persistence, not a permanent network service. Route a
source only after its exact Provider V2 entry and manifest are durably and atomically published and
the endpoint is restarted or launched with that manifest. Missing entries fail closed; the endpoint
has no silent on-demand codec fallback.

## 4. Prove cold build and exact replay separately

Run once against an empty target namespace and retain:

- process start and inferred engine-load counts;
- per-segment and total preparation wall time;
- effective-config, provider, checkpoint, qualification, namespace, and asset identities;
- source, sidecar, receipt, payload, and manifest hashes;
- codec-device UUID, utilization, VRAM, power, temperature, and throttling telemetry.

Run the same prewarm set again with a separate report output. Require every entry to be
`VERIFIED_HIT`. Do not combine hit timing with cold-build timing. Artifact replay can be exact and
authoritative; DCVC or Mage recomputation is a new computation and is not byte-exact replay.

## 5. Endpoint readiness acceptance

Before removing `--readiness-only`, require:

- startup `ok=true` and `status=READY`;
- `load_profile=native_bf16_v1` without quantization substitution;
- expected qualified checkpoint, provider manifest, model identifier, and revision;
- cache family `provider-v2` with `provider_v2_required=true`;
- expected effective-config and namespace identities;
- `max_side=448`, `sequence_length_frames=0`, and `canvas_token_side=null`;
- `device_concurrency_policy=separate-device-v1`;
- bind diagnostics report `host=0.0.0.0`, `network_boundary=controlled-private-network`,
  `endpoint_authentication=NOT_PROVIDED_BY_LAUNCHER`, and no public-bind acknowledgement;
- independently verified firewall/ingress confinement to the controlled private network;
- `shared_device_guard.required=false`;
- admitted exact warm-up source;
- `warmup.performed=true` and `warmup.authority=NON_AUTHORITATIVE_DISCARDED`;
- all model weights on endpoint-local `cuda:0`, with the UUID attested to outer selector `1`;
- no CPU/disk offload, OOM, codec fallback, missing required CUDA extension, or identity mismatch.

Retain the before/after-remap UUID evidence with the startup report.

## 6. Production qualification gates

The profile remains `production_eligible=false` until the actual two-H100 environment supplies:

1. Linux dependency and Provider V2 source/checkpoint verification.
2. Exact outer-selector, process-local ordinal, and GPU-UUID mapping.
3. Native BF16 residency on the UUID assigned to decoder selector `1`.
4. A target-built Provider V2 `max_side=448` cache on codec selector `0`.
5. One-load/five-reset evidence and a separate all-hit replay pass.
6. Cold and warm execution of the same 40-second/five-segment sample.
7. Load, codec, endpoint, generation, TTFT, token rate, utilization, VRAM, power, temperature, and
   idle-gap measurements reported separately.
8. Raw Mage output plus QA/event/evidence/track/fusion semantic comparison.
9. Worker death, endpoint death, partial publication, cache loss, restart, and replay recovery.
10. Sustained capacity, thermal, and cost qualification.

A bounded-resolution candidate requires separate quality gates and a successor profile. Do not
infer H100 speedup or semantic parity from the RTX 4060. Successful process startup and static tests
are not production qualification.

## Static checks available without an H100

```bash
python -m pytest -q -p no:cacheprovider \
  tests/unit/test_render_mage_endpoint_profile.py
python -m ruff check \
  scripts/render_mage_endpoint_profile.py \
  tests/unit/test_render_mage_endpoint_profile.py
```

These checks cover profile validation, CUDA remapping intent, command order, shell quoting, and
compatibility with the current prewarm and endpoint parsers. They do not run a model, invoke CUDA,
prepare video, attest a GPU UUID, or qualify an H100.
