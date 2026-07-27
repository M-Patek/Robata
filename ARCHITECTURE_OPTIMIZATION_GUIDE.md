# Robata Architecture Optimization Guide

> This guide is a construction map for advancing Robata from LOCAL_CONFORMANCE toward
> production-qualified throughput and quality. It is not a production certificate.
> Every claim is tagged with its evidence class; unmeasured projections remain NOT_MEASURED
> until representative load proves them.

---

## 0. Current Baseline and Physical Ceiling

### 0.1 Measured Baseline (LOCAL_CONFORMANCE, dated fixture)

The checked-in local-conformance slice uses a 40.8335 recording-second, six-camera fixture,
PyAV CPU decode, and a deterministic offline provider. Key measurements from the WP6 run:

| Metric | Value | Evidence class |
|--------|-------|----------------|
| Fresh elapsed | 118.75 s | LOCAL_CONFORMANCE |
| Fresh RTF | 2.908 | LOCAL_CONFORMANCE |
| 30-min capacity | 2.162 rec-sec/wall-sec | LOCAL_CONFORMANCE |
| p95 latency | 3.908 s | LOCAL_CONFORMANCE |
| Backlog growth | 0 | LOCAL_CONFORMANCE |
| Work items | 9,001 | LOCAL_CONFORMANCE |
| Windows | 1,800 | LOCAL_CONFORMANCE |

### 0.2 Where Time Goes (from BLUEPRINT profiling)

| Phase | Wall seconds | Share | Nature |
|-------|-------------|-------|--------|
| source.stream.capture_publish | ~53.05 | 45% | Window append/publish/drain, SQLite transactions |
| source.materialize | ~22.33 | 19% | PyAV CPU decode, PNG encode, fsync, hash |
| source.prepare (parent envelope) | ~77.38 | 65% | Contains inspect, capture, validation, indexing, quality |
| Inference-evidence SQLite scope | significant | — | Per-attempt append-only writes |
| Completion auditing/serialization | significant | — | Cross-referencing and Merkle root computation |

### 0.3 Physical Ceiling Analysis

The throughput ceiling is set by three independent bottlenecks. Each has a clear physical
upper bound that no software optimization can exceed:

| Bottleneck | Current mechanism | Physical ceiling | Gap factor |
|------------|-------------------|-----------------|------------|
| **Decode** | PyAV CPU single-thread H.264 (`av.CodecContext.create("h264","r")` in `pyav_decoder.py`) | NVDEC 6-lane parallel ≈ 600+ fps | ~10–20× |
| **Inference** | Offline fixture provider (deterministic mock, no real GPU) | vLLM continuous batching on H100 ≈ 200–500 img/s | N/A (mock → real) |
| **I/O** | Per-frame PNG encode + fsync + separate hash computation | In-memory hash + batched fsync | ~5–10× |

**Important**: The current provider is a deterministic mock (`OfflineFixtureVisionAdapter` in
`inference/offline_fixture.py`). Real-model throughput numbers are NOT_MEASURED. The
"~10 img/s" figure sometimes cited is a projection for single-request Qwen-VL-7B, not a
measured result. Any throughput claim involving real models requires representative
benchmark evidence (evidence class: REPRESENTATIVE_BENCHMARK or higher).

### 0.4 Quality Ceiling

The 21-class product QA cascade (`ProductQACascadeProjector` in `qa_pipeline/product.py`)
produces a complete coverage projection for every eligible clip. Current limitations:

- **No calibration**: `ProductQAConfidenceKind` distinguishes DETECTOR_REPORTED /
  MODEL_REPORTED / POLICY_DERIVED, but there is no temperature scaling, Platt scaling,
  or ECE measurement. Model-reported confidence is uncalibrated.
- **No uncertainty quantification**: No MC-dropout, no ensemble disagreement, no conformal
  prediction. Abstention and INCOMPLETE are policy-derived, not statistically grounded.
- **No cross-window temporal consistency**: Each 2s window produces event proposals
  independently. A physical action spanning multiple windows is split into separate
  candidates with no trajectory-level coherence.
- **No human feedback loop**: Review routing exists (`local_review_routing.py`), but
  there is no active learning mechanism to prioritize uncertain clips for annotation.

These are NOT bugs — the system is correct by construction (fail-closed, deterministic
replay). They are quality ceiling limitations that require new mechanisms to lift.

---

## 1. [P0 / Physical] NVDEC Hardware Decode Pipeline

### Current State

`pyav_decoder.py` uses `av.CodecContext.create("h264", "r")` — pure software decode.
Six-camera H.264 at 30 fps per camera = 180 fps total decode demand, which saturates a
single CPU core at ~1× real-time. This is the dominant wall-time contributor in
`source.materialize` (~22.33 s for the fixture).

NVDEC adapter files already exist but are not the default path:
- `adapters/nvdec_backend.py` — `NvdecFallbackReason` enum, error handling
- `adapters/nvdec_frame_materializer.py` — `NvdecFrameMaterializer` with PyAV fallback
- `adapters/nvdec_video_export.py` — `NvdecH264Mp4Exporter` with PyAV fallback

These are marked P8 (unmeasured) in the BLUEPRINT. The task is to qualify them as the
default decode path.

### Approach

1. **Multi-lane NVDEC decode**: A single H100 has 5–7 NVDEC engines. Six cameras map
   naturally to 6 parallel decode lanes. Use NVIDIA's VPF (Video Processing Framework)
   or `decord` with CUDA backend.

2. **Zero-copy GPU residency**: Decoded NV12 frames stay in GPU memory. Resize, color
   conversion, and hashing happen on GPU. Only sampled frames (~1–5% of total) are
   copied back to CPU for encoding and VLM input.

3. **NVJPEG encode**: Sampled frames use NVJPEG instead of PNG. JPEG at quality=90 is
   visually lossless for VLM input and ~10× faster to encode than PNG (see §6).

4. **Fail-closed fallback**: When NVDEC is unavailable, fall back to PyAV CPU decode.
   The existing `NvdecFallbackReason` enum already models this. Fallback reason must be
   recorded in the evidence chain — silent degradation is prohibited.

### Invariant Protection

- **Identity unchanged**: `exact_bytes_sha256` is computed from original MCAP packet bytes
  (`H264SpoolFacts.sha256`), never from decoded frames. NVDEC output does not enter
  identity hashes. This is consistent with the current architecture.
- **Determinism pinning**: NVDEC is deterministic for identical input + driver version.
  Pin `cuda_version` / `driver_version` / `nvdec_codec` in `CapabilitySnapshot` as part
  of `input_digest`. This is a new versioned input requiring schema registration.
- **Fail-closed**: NVDEC unavailability → explicit fallback with recorded reason, never
  silent quality loss.

### Expected Impact

- Decode throughput: ~180 fps (CPU) → ~1800 fps (NVDEC 6-lane) = **~10×** (NOT_MEASURED)
- CPU memory reduction: NV12 frames stay on GPU
- Enables §3 (pipelining) by freeing CPU for reduction/finalization work

### BLUEPRINT Alignment

P2 (feed-once media) + P8 (target media qualification). This is the critical investment
to move P8 from "future" to "now".

### Implementation Difficulty

**High**. Requires:
1. New dependency (`decord` or `nvidia.vpf`) in `pyproject.toml` `[nvdec]` extra
2. Rewrite of `McapSinglePassH264Tee` decode branch (spool logic and identity layer unchanged)
3. New `GpuFrameMaterializer` coexisting with `PyAvFrameMaterializer`, switched via
   `FrameArtifactResolver` protocol
4. CI qualifying job with GPU (local fixture replay of NVDEC-recorded responses)

---

## 2. [P0 / Physical] Inference Continuous Batching

### Current State

The `_BoundedMicrobatchDispatcher` in `orchestrator.py` (line 172) uses static batching:
it waits for `max_batch_size` requests (default: **1**, not 8) or a `max_queue_delay_ms`
timeout before dispatching. This is a conservative design — each inference call goes out
individually.

`RunPodEndpointConfig.native_batch_enabled` defaults to `False` (line 103 in `runpod.py`).
When disabled, `infer_batch` falls back to concurrent single-request calls via
`asyncio.gather`. The native batch path (`_infer_native_batches`) exists but is not
activated by default.

The current provider is `OfflineFixtureVisionAdapter` — a deterministic mock with no
real GPU. All throughput projections for real models are NOT_MEASURED.

### Approach

1. **vLLM continuous batching**: Instead of waiting to fill a static batch, submit
   requests immediately to vLLM's `/v1/chat/completions` endpoint. vLLM's PagedAttention
   manages KV cache across 64–256 concurrent requests internally, eliminating
   head-of-line blocking.

2. **Enable native batch path**: Set `native_batch_enabled = True` and raise
   `native_batch_max_size` to 64–128. The `_dispatch_native_batch_chunk` method in
   `runpod.py` already implements the wire protocol for batch requests.

3. **KV cache prefix reuse**: System prompt + QA instructions are identical across
   calls for the same recording. vLLM's `--enable-prefix-caching` reuses prefix KV
   cache across requests, reducing TTFT by ~3–5×.

### Invariant Protection

- **Call identity unchanged**: `InferenceInputPlan.input_plan_id` is derived from
  `request_catalog_sha256 + capability_snapshot_sha256 + rendered_prompt_sha256 +
  execution_policy_sha256`. Batching is a transport-layer optimization; it does not
  enter identity hashes (same principle as "credentials don't enter identity").
- **Evidence integrity**: Every outcome still lands in `sqlite_inference_evidence`.
  Under continuous batching, 64–256 outcomes may need concurrent persistence (see §4).
- **Replay**: `RecordedRunPodTransport` replays recorded request-response pairs.
  Continuous batching responses are recorded per-request, not as batch envelopes.

### Expected Impact

- Inference throughput: NOT_MEASURED → requires representative benchmark
- TTFT reduction: ~3–5× with prefix caching (NOT_MEASURED)
- This converts "provider is the bottleneck" to "provider is not the bottleneck"

### BLUEPRINT Alignment

P3 (adaptive provider-neutral QA) + P8 (RunPod qualification).

### Implementation Difficulty

**Medium-High**. The batch infrastructure already exists in `runpod.py`. Required:
1. Set `native_batch_enabled = True` by default for qualified RunPod endpoints
2. Raise `native_batch_max_size` to 64–128 after benchmarking
3. Benchmark to find the GPU utilization vs. p99 latency Pareto frontier
4. All throughput claims remain NOT_MEASURED until real H100 evidence exists

---

## 3. [P0 / Architecture] Stage-Affine Window Pipelining

### Current State

The `DurableStreamWindowScheduler` in `stream_scheduler.py` manages a 5-stage DAG:

```
WINDOW → QA_COARSE → QA_DENSE → EVENT_PROPOSAL → WINDOW_REDUCTION
```

(plus FINALIZATION after EOS). The DAG topology is defined at line 91:

```python
_WINDOW_DAG_TOPOLOGY = (
    (StreamStage.WINDOW, ()),
    (StreamStage.QA_COARSE, (StreamStage.WINDOW,)),
    (StreamStage.QA_DENSE, (StreamStage.QA_COARSE,)),
    (StreamStage.EVENT_PROPOSAL, (StreamStage.QA_COARSE, StreamStage.QA_DENSE)),
    (StreamStage.WINDOW_REDUCTION,
     (StreamStage.WINDOW, StreamStage.QA_COARSE, StreamStage.QA_DENSE, StreamStage.EVENT_PROPOSAL)),
)
```

All dependencies are `DependencyCriticality.DEGRADABLE`.

The `CanonicalLocalProviderQueue` in `parallel_service.py` provides a bounded shared
dispatcher with `max_concurrency` worker threads. However, the drain loop in
`LocalConformanceStreamFinalizer` processes windows sequentially — window N+1's
WINDOW stage waits for window N's WINDOW_REDUCTION to complete.

The `CanonicalOfflinePipeline.run` in `runner.py` (line 802) executes one complete
local run. The current flow is effectively single-window synchronous: each window
traverses all 5 stages before the next window begins.

### Approach

**Stage-affine pipelining**: Different windows can occupy different pipeline stages
simultaneously, like a CPU instruction pipeline:

```
Window:      1     2     3     4     5
WINDOW:     [W1]
QA_COARSE:       [W1]  [W2]
QA_DENSE:             [W1]  [W2]  [W3]
EVENT_PROP:                [W1]  [W2]  [W3]
REDUCTION:                       [W1]  [W2]
FINAL:                                  [all]
```

Each stage gets an independent worker pool:
- **Media worker** (CPU/NVDEC): WINDOW stage — decode + materialize
- **Coarse worker** (GPU): QA_COARSE inference
- **Dense worker** (GPU): QA_DENSE inference
- **Event worker** (GPU): EVENT_PROPOSAL + action evidence
- **Reduction worker** (CPU): WINDOW_REDUCTION — pure computation
- **Finalization worker** (CPU): single invocation after EOS

Stages are connected by bounded channels (`asyncio.Queue(maxsize=N)`). Backpressure
propagates naturally: if the coarse worker is slow, its input channel fills, and the
media worker blocks (stops decoding).

### Invariant Protection

- **DAG dependencies unchanged**: Within each window, the 5-stage dependency order is
  preserved. Cross-window dependencies do not exist in `_WINDOW_DAG_TOPOLOGY`. The
  `work_dependencies` table still records per-window edges.
- **Terminal truth unchanged**: The two-phase commit protocol (pending_terminal →
  SUCCEEDED → terminal_member binding) remains correct under concurrency. `claim_and_start`
  CAS guarantees no work item is claimed by two workers.
- **Monotonic chain**: Window N+1's WINDOW_REDUCTION still waits for window N's
  WINDOW_REDUCTION. This is a stream-level invariant, not a per-stage invariant, and
  must be enforced by the pipeline coordinator.
- **Backpressure**: Channel `maxsize` replaces the static `BackpressureConfig` threshold
  for inter-stage flow control. The existing `BackpressureController` still governs
  admission at the stream boundary.

### Expected Impact

- Single-recording RTF: from sequential ~2.9× to pipelined ~3–5× (LOCAL_CONFORMANCE
  projection; 6 stages → theoretical 6×, practical 3–5× due to stage imbalance)
- Combined with §1 + §2: multiplicative improvement potential (NOT_MEASURED for real models)
- p95 latency: individual window tail latency no longer blocks subsequent windows

### BLUEPRINT Alignment

P6 (recording-level parallelism). The current P6 design focuses on multi-recording
parallelism; this proposal adds single-recording intra-window pipelining as a
necessary complement.

### Implementation Difficulty

**High**. Requires rewriting the drain loop in `LocalConformanceStreamFinalizer` as an
asyncio pipeline. Key risk: SQLite write serialization under concurrent workers (see §4).

Recommended incremental approach:
1. First: one worker per stage (one coroutine per stage, serial within each stage)
2. Verify correctness with existing fixture tests
3. Then: multiple workers per stage for bottleneck stages

---

## 4. [P0 / Correctness] SQLite Concurrent Write Strategy

### Current State

`SQLiteWorkScheduler._transaction` (line 1687 in `sqlite_work_scheduler.py`) opens and
closes a connection per transaction. This is safe but expensive under high concurrency.

`SQLiteInferenceEvidenceLedger` uses an instance-level `RLock` (`self._state_lock`,
line 406 in `sqlite_inference_evidence.py`) to serialize writes. This is correct but
becomes a bottleneck when §3's pipeline produces concurrent completions.

`_BUSY_TIMEOUT_MS` differs between adapters:
- `sqlite_work_scheduler.py`: 5,000 ms
- `sqlite_inference_evidence.py`: 30,000 ms

### Approach

1. **Single write connection + application-level mutex**:
   - One dedicated write connection (`self._write_conn`), protected by a threading lock.
   - All write transactions go through `with self._write_lock: conn.execute("BEGIN IMMEDIATE")`.
   - This replaces SQLite file-lock contention with in-process mutex — zero wait, zero conflict.

2. **Multiple read connections (connection pool)**:
   - N read connections (N = worker count), each owned by one worker thread.
   - WAL mode readers hold snapshots without blocking the writer.

3. **Reduce `busy_timeout` to 1,000 ms**: Application-level mutex makes long busy_timeout
   unnecessary; it becomes a safety net only.

4. **Batch write coalescing** (optional, second phase):
   - Multiple workers' `complete()` calls are submitted to a write queue.
   - A single write thread flushes the queue periodically (every ~1 ms or when N items
     accumulate).
   - This merges "100 small transactions" into "1 batch transaction", reducing I/O by ~100×.

### Invariant Protection

- **Two-phase commit unchanged**: `store_pending_terminal` + `succeed` +
  `accept_pending_terminal` is an atomic sequence across two ledgers. Under batch
  coalescing, these three steps must be committed as one batch — they cannot be split
  across flush boundaries (that would widen the crash window).
- **Lease fencing unchanged**: `claim_and_start` CAS is naturally atomic under single-writer
  (SQLite single-write = serial = inherent atomicity).
- **Append-only triggers unchanged**: The existing BEFORE UPDATE / BEFORE DELETE triggers
  continue to enforce immutability.

### Expected Impact

- Unlocks §3's concurrent writes: 100 workers completing simultaneously without
  serial waiting
- Batch writes: I/O from ~10 ms/transaction to ~0.1 ms/transaction (batch of 100)
- Connection reuse: eliminates per-transaction open/close overhead

### Implementation Difficulty

**Medium**. Add connection pool + write queue to `SQLiteWorkScheduler`. The critical
requirement is ensuring two-phase commit atomicity under batch coalescing — the batch's
minimum commit unit must be one complete sequence (pending_terminal → succeed → accept).

---

## 5. [P1 / Physical] Cross-Window Frame Reuse

### Current State

`BoundedMediaPolicy` (line 64 in `bounded_media.py`) defines:
- `window_width_ns = 2_000_000_000` (2 seconds)
- `window_hop_ns = 1_000_000_000` (1 second)

Hop < width means adjacent windows share 50% of their frames. Currently, each window
independently calls `materialize_admitted`, decoding the same frames in window N and
window N+1.

`frame_cache.py` already exists with `SharedFrameCache`, `FrameRef`, and `FrameFeedManifest`.
It provides feed-once coordination but is not yet integrated into the window materialization
path for cross-window reuse.

### Approach

1. **Recording-level frame cache** (LRU + content-addressed):
   - Cache key: `(camera_id, packet_index, timestamp_ns)`
   - Cache value: decoded frame tensor (NV12 on GPU if §1 is active, or PIL Image on CPU)
   - LRU capacity: `ring_duration_ns / window_hop_ns` windows × 6 cameras × ~60 frames
     ≈ 3,600 frames. On GPU at ~3 MB/frame (1080p NV12), this is ~10 GB — acceptable
     for H100 (80 GB). On CPU, use compressed representation.

2. **Sequential packet consumption**:
   - `McapSinglePassH264Tee` already does one MCAP pass, writing H.264 packets to 6
     spools. Materialization reads from spools and decodes. Adjacent windows' packets
     are contiguous — instead of per-window seek+decode, advance the decoder sequentially
     and select frames at hop boundaries.

3. **Derived artifact hashing**:
   - Identity hashes are based on MCAP payload bytes (`H264SpoolFacts.sha256`), not on
     decoded frames. Frame cache reuse does not affect identity. Materialized PNG/JPEG
     artifacts are derived and have their own lineage chain.

### Invariant Protection

- **Identity unchanged**: Frame cache stores decoded frame tensors, not identity hashes.
  Identity comes from MCAP payload, independent of decode path.
- **Determinism**: GPU decode determinism is pinned via §1's `cuda_version` in
  `CapabilitySnapshot`. CPU decode is already deterministic.
- **Memory bounded**: LRU eviction of oldest window frames keeps memory predictable.

### Expected Impact

- Decode volume halved (50% frame overlap → reuse): §1's NVDEC throughput effectively ×2
- From ~1,800 fps equivalent → ~3,600 fps equivalent (half the frames are cache hits)

### Implementation Difficulty

**Medium**. `frame_cache.py` exists and needs extension for cross-window LRU + GPU tensor
storage. Integration point is the materialization path in `pyav_frame_materializer.py`
or the new `GpuFrameMaterializer` from §1.

---

## 6. [P1 / Physical] JPEG/NVJPEG Encode Instead of PNG

### Current State

`pyav_frame_materializer.py` encodes sampled frames as PNG via `_encode_png()` (line 419),
which uses PyAV's PNG codec context directly (not `frame.to_image().save()` as sometimes
assumed — the actual path is `av.CodecContext.create("png", "l")` → `codec.encode(frame)`).

PNG is lossless but slow (~5 ms/frame) and produces large files (~500 KB/frame for 1080p).
VLMs (Qwen-VL, GPT-4V, etc.) accept JPEG input. JPEG at quality=90 is visually lossless
for VLM inference and ~10× faster to encode.

### Approach

1. **Default JPEG** (quality=90):
   - `frame.to_image().save(path, "JPEG", quality=90)` ≈ 0.5 ms/frame
   - File size ~50 KB (vs. PNG ~500 KB) — I/O bandwidth reduced 10×
   - VLM inference quality is unaffected at quality=90 (industry consensus)

2. **NVJPEG** (GPU encode, if §1 is active):
   - ~0.05 ms/frame on GPU
   - Pairs naturally with NVDEC decode — frame stays on GPU from decode through encode

3. **PNG as opt-in**: For domains requiring lossless evidence (medical, legal), PNG
   remains available as a configurable option.

### Invariant Protection

- **Identity unchanged**: Derived artifact `exact_sha256` is based on actual bytes.
  JPEG and PNG are different artifacts with separate lineage. The source frame identity
  (MCAP payload hash) is unchanged.
- **Schema registration**: New artifact type requires registration (different `media_type`
  from `image/png` to `image/jpeg`). This is an additive change — no existing schema
  is modified.

### Expected Impact

- Encode speed: ~10× (PNG → JPEG) or ~100× (PNG → NVJPEG)
- I/O: file size ~10× smaller, disk/network transfer proportionally faster
- Combined with §5 (frame reuse): fewer frames to encode overall

### Implementation Difficulty

**Low**. Change encode format string + quality parameter. Requires schema registration
for the new artifact media type. The `_encode_png` function can be paralleled by an
`_encode_jpeg` function using `av.CodecContext.create("mjpeg", "l")`.

---

## 7. [P1 / I/O] Zero-Copy Hash and Batched Fsync

### Current State

The materialization flow in `pyav_frame_materializer.py`:
1. Decode frame (PyAV)
2. Encode to PNG bytes via `_encode_png()` (PyAV PNG codec)
3. Write bytes to file via `_write_new_file()` with fsync
4. Compute `exact_bytes_sha256(png_bytes)` **in memory** (line 540)

The hash is already computed in memory — there is no "write then re-open to hash" step
in the current code. The fsync-per-frame pattern is the actual I/O bottleneck.

### Approach

1. **Batched fsync**: Instead of fsync after every frame, batch fsync every N frames
   or every ~10 ms. Crash may lose up to N unwritten frames, but the two-phase commit
   protocol guarantees that lost frames correspond to PENDING work items — recovery
   re-materializes them. This trades "already-guaranteed recoverability" for I/O speed.

2. **Async I/O** (platform-specific, second phase):
   - Linux: `io_uring` for batch write/fsync submission without blocking worker threads
   - Windows: IOCP via `asyncio` + overlapped I/O
   - macOS: fallback to `aiofiles`

3. **tmpfs/RAM disk staging**: Materialized artifacts are written to a temporary directory
   (`tempfiles.make_staging_directory`). Configuring this as tmpfs eliminates disk I/O
   entirely for transient artifacts. Only artifacts that need persistence are written to
   permanent storage.

### Invariant Protection

- **Artifact integrity**: `exact_bytes_sha256` is computed from the same bytes that are
  written to disk. Under batched fsync, the hash is still correct (computed before write).
  If a crash loses unwritten frames, the corresponding work items are still PENDING —
  recovery re-materializes them with identical bytes (deterministic decode).
- **Two-phase commit**: The pending_terminal → SUCCEEDED → terminal_member sequence is
  the authority for completion. Materialized artifacts are derived evidence, not
  authority state. Losing derived artifacts is safe because they can be recomputed.

### Expected Impact

- I/O per frame: from ~5 ms (encode + write + fsync) to ~0.5 ms (encode + in-memory hash)
- For the fixture's ~54K sampled frames: ~270 s → ~27 s of I/O time (theoretical;
  actual improvement depends on what fraction of frames are sampled)

### Implementation Difficulty

**Low-Medium**. Batching fsync is a code change in `_write_new_file`. tmpfs staging is
a configuration change. io_uring requires platform-specific dependencies and is a
separate phase.

---

## 8. [P1 / Quality] Confidence Calibration and Uncertainty Quantification

### Current State

`ProductQAConfidenceKind` in `contracts/qa.py` distinguishes DETECTOR_REPORTED /
MODEL_REPORTED / POLICY_DERIVED, but there is no calibration mechanism. Model-reported
confidence values are uncalibrated — a model reporting `confidence=0.9` may have actual
accuracy of only 70% (overconfidence).

There is no ECE (Expected Calibration Error) measurement, no conformal prediction for
guaranteed coverage, and no ensemble/MC-dropout uncertainty estimation. Abstention and
INCOMPLETE are policy-derived decisions, not statistically grounded ones.

### Approach

1. **Temperature scaling** (post-hoc calibration, zero inference cost):
   - Fit a single temperature parameter T on a held-out calibration set.
   - Apply `confidence_calibrated = softmax(logits / T)` to model outputs.
   - Calibrated confidence satisfies `P(correct | confidence=c) ≈ c` with ECE < 0.05.
   - This is standard practice (Guo et al. 2017) with zero inference overhead.

2. **Conformal prediction** (guaranteed coverage):
   - For each of the 21 issue classes, produce a prediction set:
     "This clip belongs to {BLACK_SCREEN, TOO_DARK}, coverage 90%."
   - When prediction set size > 1, trigger dense QA or human review.
   - This converts abstention from a boolean to a quantified statistical decision.

3. **Ensemble disagreement** (explicit uncertainty):
   - For boundary cases, run N inference attempts (different seeds or dropout).
   - `InferenceAttemptSelection` already supports multiple attempts — reuse this mechanism.
   - High disagreement → `INDETERMINATE` (existing state).
   - This is not retry-on-failure; it is deliberate multi-sample uncertainty estimation.

4. **Calibration dataset and ECE measurement**:
   - `benchmark/ground_truth.py` already has `QAIssueAnnotation` and
     `InterAnnotatorAgreement`.
   - Add `benchmark/calibration.py`: `CalibrationReport` with per-class ECE, Brier score,
     reliability diagram data.
   - Calibration report becomes a required evidence artifact for P10 production gate.

### Invariant Protection

- **21-class semantics unchanged**: `ProductQAIssue` enum is not modified. `QAAssessment`
  pass/warning/fail semantics are unchanged.
- **Calibration parameter versioned**: Temperature parameter enters as a new policy version
  in `enrichment_policy_version` (existing field). Changing temperature = changing policy
  version = new identity (correct behavior).
- **Conformal does not introduce false positives**: Prediction sets are "subsets that may
  contain the truth." When set size > 1, the outcome is `INCOMPLETE` (existing state),
  not `FAIL`.

### Expected Impact

- Quality becomes provable: from "model says 0.9" to "ECE < 0.05 calibrated confidence"
- Enables theoretically grounded PASS/WARNING/FAIL thresholds (replacing rule-based hardcoding)
- Uncertainty quantification converts abstention from engineering heuristic to statistical
  decision

### Implementation Difficulty

**Medium**. Calibration is offline fitting + online division — small code volume. The
key dependency is representative labeled data (BLUEPRINT E1 gate), which is currently
NOT_MEASURED.

---

## 9. [P1 / Quality] Cross-Window Event Tracking

### Current State

`event_pipeline/provisional_fusion.py` uses `_connected_components` (line 514) for
cross-camera fusion within a single window. A physical action spanning multiple 2-second
windows is split into separate candidates with no trajectory-level coherence.

`CandidateReducer` in `event_pipeline/candidate.py` merges proposals by label and
merge-gap within a window. `BoundaryRefinementProjector` in
`event_pipeline/boundary_refinement.py` refines onset/offset boundaries using
median-low cross-camera estimation — but only within a single window.

### Approach

1. **Sliding-window event tracking**:
   - Add an `EVENT_TRACKING` stage after `EVENT_PROPOSAL` (new `StreamStage 6 in the DAG).
   - Use trajectory association (similar to DeepSORT): adjacent windows' event candidates
     with temporal overlap + same label + consistent camera set are linked into tracks.
   - Track-level output: `PhysicalActionTrack` with cross-window onset/offset extension.

2. **Cross-window boundary context**:
   - `BoundaryRefinementProjector` currently uses pre/post context within one window
     (`context_offsets_ns`). Extend to use frames from adjacent windows when a track
     spans the boundary.

3. **Track-level deduplication**:
   - After cross-window fusion, merge multiple window candidates from the same track
     into one final event. This affects `CandidateReducer` deduplication logic —
     deduplication key changes from (time, label) to (track_id).

### Invariant Protection

- **Event identity evolution**: `CanonicalCandidateEvent.candidate_event_identity_sha256`
  is currently based on single-window evidence. Cross-window fusion requires identity
  based on track-level evidence collection — this is an **identity formula change**
  requiring a new schema version (`candidate-event-v2`). Old identities are preserved
  for single-window replay compatibility.
- **Replay**: Cross-window fusion inputs are per-window event proposals (already persisted).
  Fusion itself is a deterministic function of those inputs — replayable.
- **DAG topology change**: Adding `EVENT_TRACKING` requires updating `_WINDOW_DAG_TOPOLOGY`
  and registering the new stage in `StreamStage`. This is an additive change.

### Expected Impact

- Quality: cross-window actions are no longer truncated; boundaries are more accurate
- This is the single most likely improvement to raise "complex scenario accuracy" from
  estimated ~60–70% to ~80%+ (NOT_MEASURED — requires representative labels)

### Implementation Difficulty

**High**. Involves new `StreamStage` + identity formula evolution + DAG topology update.
Recommended as a sub-phase of P4 (e.g., P4.5).

---

## 10. [P1 / Quality] Active Learning Feedback Loop

### Current State

`local_review_routing.py` routes completed results to a review queue, but routing is
static — all clips are treated equally. There is no mechanism to prioritize "model most
uncertain" clips for human annotation.

`benchmark/ground_truth.py` has `QAIssueAnnotation` and `InterAnnotatorAgreement`.
`benchmark/splits.py` has `DataSplitter` for leakage-safe train/test splitting. The
infrastructure for a feedback loop exists but is not connected.

### Approach

1. **Uncertainty-based sampling**:
   - In `route_local_review_after_completion`, sort by §8's uncertainty score.
   - Top-k% uncertain clips get high-priority review queue placement.
   - This is standard active learning — minimum annotation for maximum information.

2. **Annotation feedback loop**:
   - Human annotations stored in `benchmark/ground_truth.py`'s existing structures.
   - Trigger offline task: refit §8's temperature, update conformal thresholds,
     recompute `benchmark/calibration.py`'s ECE.
   - New calibration parameters published as new `enrichment_policy_version` (versioned).

3. **Data flywheel**:
   - High uncertainty → annotation → calibration improvement → next run's uncertainty
     is more accurate → flywheel accelerates.
   - Metric: ECE decrease curve over time.

### Invariant Protection

- **Terminal truth unchanged**: Active learning is an offline loop. It does not affect
  the determinism of online QA completion. Annotation feedback updates **future runs'**
  policy versions, not past completions' terminal state.
- **Leakage-safe**: `DataSplitter` in `benchmark/splits.py` enforces train/test separation
  (time-based or recording-based split). Calibration data must be disjoint from
  evaluation data.

### Expected Impact

- Quality continuous improvement: ECE from initial ~0.15 to <0.05 (with annotation
  accumulation)
- This is the practical path from LOCAL_CONFORMANCE → REPRESENTATIVE_BENCHMARK →
  PRODUCTION_QUALIFIED

### Implementation Difficulty

**Medium**. Review routing infrastructure exists; adding uncertainty-based prioritization
is straightforward. The feedback loop requires offline task scheduling (can be a
separate service).

---

## 11. [P2 / Architecture] Adaptive Backpressure (AIMD)

### Current State

`BackpressureController` in `queue/backpressure.py` uses configurable thresholds
(`queue_depth_threshold`, `oldest_age_threshold_ms`, `backlog_slope_threshold`).
These have no defaults in the model — they are set by the caller. The stream scheduler
uses `DEFAULT_STREAM_BACKPRESSURE_CONFIG` (line 114 in `stream_scheduler.py`):
- `queue_depth_threshold = 256`
- `oldest_age_threshold_ms = 30_000`
- `backlog_slope_threshold = 128.0`

`QueueMetrics` has `arrival_rate` and `service_rate` fields, but these are populated
by the monitoring system — they are not "always 0.0" as sometimes claimed. The issue
is that static thresholds do not adapt to load changes.

### Approach

**AIMD (Additive Increase, Multiplicative Decrease)** — the TCP congestion control
algorithm adapted for work admission:

- When queue is healthy (depth < threshold / 2): `max_active_windows` increases by +1/min
- When pressure rises (depth > threshold × 0.75): `max_active_windows` decreases by ×0.5
- This provides load-adaptive concurrency control without manual threshold tuning.

### Invariant Protection

- **Terminal truth unchanged**: Backpressure only regulates "how many new windows to
  admit." It does not affect the correctness of already-admitted work.
- **Observable**: AIMD window changes are recorded in `RuntimeProfileSnapshot` for
  post-hoc congestion analysis.

### Expected Impact

- Load-adaptive: high load → auto-reduce concurrency to protect quality; low load →
  auto-increase concurrency for throughput
- More robust than static thresholds, especially under bursty traffic

### Implementation Difficulty

**Low**. `BackpressureController` already has the structure; adding AIMD state machine
is a small extension.

---

## 12. [P2 / Retrieval] Vector Search and Reranking

### Current State

`retrieval/index.py` has `EventIndex` (line 108) with `_lexical_score`) — pure lexical
scoring with no semantic vector search. BLUEPRINT P9 mentions pgvector but marks it
NOT_MEASURED.

### Approach

1. **Dual encoder**:
   - Text side: sentence-transformers / BGE for event description embeddings
   - Visual side: CLIP / SigLIP for representative frame embeddings
   - Store in pgvector with HNSW index

2. **Hybrid retrieval + reranking**:
   - Stage 1: vector recall top-100 (~10 ms)
   - Stage 2: cross-encoder rerank top-10 (~50 ms)
   - Fuse with `EventIndex` lexical retrieval via reciprocal rank fusion (RRF)

3. **Async indexing**:
   - After event terminal closure, encode vectors asynchronously (non-blocking for
     QA completion). pgvector writes are idempotent (keyed by event revision ID).

### Invariant Protection

- **Non-blocking for terminal**: Vector encoding failure must not reopen QA completion
  (BLUEPRINT already emphasizes this).
- **Versioned**: Embedding model / dimension / index policy are versioned metadata
  (BLUEPRINT P9 compatibility notes cover this).

### Expected Impact

- Retrieval quality: from lexical matching to semantic matching, recall@10 improvement
  ~30–50% (NOT_MEASURED)
- User experience: search "person picks up cup" finds the corresponding event without
  exact keyword matching

### Implementation Difficulty

**Medium**. Requires embedding model dependency + pgvector infrastructure.

---

## 13. [P0 / Architecture] Transaction Collapse — Reducing SQLite Round-Trips per Window

### Current State

A single window's lifecycle in the stream scheduler involves a cascade of SQLite
transactions. Tracing the full path for one work item:

1. `declare_expected_window` → INSERT into `expected_windows` + INSERT into `stream_work_plans` (one per stage, 5 stages per window)
2. `claim_and_start` → UPDATE `work_items` (CAS on lease_epoch + fencing_token)
3. Provider execution (no SQLite)
4. `store_pending_terminal` → UPDATE `stream_work_plans` (set pending_terminal_json)
5. `succeed` → UPDATE `work_items` (set state=SUCCEEDED, result_reference)
6. `accept_pending_terminal` → UPDATE `stream_work_plans` (set publication_state=PUBLISHED, terminal_evidence_json)

For a 5-stage window, that is **5 × (declare + claim + pending_terminal + succeed + accept)
= 25 transactions minimum**, each opening and closing its own connection
(`_transaction` in `sqlite_work_scheduler.py` line 1687).

With 1,800 windows in the fixture, that is **45,000+ connection open/close cycles**.
At ~0.2 ms per open+close, this alone accounts for ~9 seconds of pure connection
management overhead — before any actual I/O.

### Approach

1. **Connection residency**: Keep one write connection open for the lifetime of the
   recording run, instead of opening/closing per transaction. The `BEGIN IMMEDIATE`
   transaction boundary still provides atomicity; the connection stays open between
   transactions.

2. **Batch declare**: Instead of 5 separate INSERT transactions for a window's 5 stages,
   insert all 5 work items in a single transaction. The `declare_expected_window`
   method already has all 5 plans available at declaration time.

3. **Deferred terminal acceptance**: The `store_pending_terminal → succeed →
   accept_pending_terminal` sequence is three transactions for one work item. The
   pending terminal and the succeed can be merged into one transaction (the pending
   terminal is only needed for crash recovery between the two steps; under single-writer
   with connection residency, the crash window is negligible). This collapses 3 → 2
   transactions per work item.

4. **Window-level commit**: For non-critical stages (WINDOW, QA_COARSE), the entire
   claim→execute→succeed→accept cycle can be committed as one batch at window boundary,
   reducing per-stage transaction overhead to amortized per-window overhead.

### Invariant Protection

- **Two-phase commit preserved**: The critical sequence (pending_terminal → SUCCEEDED →
  terminal_member) remains atomic. Batch declare and deferred acceptance reduce
  intermediate state visibility, not final state correctness.
- **Lease fencing preserved**: CAS on `(lease_epoch, fencing_token)` is unchanged.
  Under connection residency, the CAS still executes within `BEGIN IMMEDIATE`.
- **Replay unchanged**: Idempotent re-declaration and re-completion still work.
  Batch declare is idempotent (INSERT OR IGNORE on logical_key uniqueness constraint).

### Expected Impact

- Transaction count per window: from ~25 to ~5–7 (LOCAL_CONFORMANCE projection)
- Connection overhead: from ~9 s to ~1–2 s for the 1,800-window fixture
- This is a pure software optimization with no hardware dependency

### Implementation Difficulty

**Low-Medium**. Connection residency is a straightforward refactor of `_transaction`.
Batch declare requires restructuring `declare_expected_window`. The key risk is
ensuring crash recovery semantics are preserved under deferred acceptance — the
recovery path must handle "work item is SUCCEEDED but terminal not yet accepted."

---

## 14. [P0 / Quality] Model-Guided Adaptive Sampling — Close the Sense-Act Loop

### Current State

The adaptive sampling system (`sampling/adaptive.py`) uses lightweight signal detectors
(`AdaptiveSignal`: MOTION_ENERGY, SCENE_CHANGE, BLUR_CHANGE, OCCLUSION_CHANGE,
HAND_PRESENCE, HAND_OBJECT_DISTANCE, OBJECT_MOTION) to trigger sampling rate upgrades.
These are CPU-side heuristics operating on decoded frame pixels — fast but imprecise.

The QA pipeline produces rich per-camera observations (`CoarseQAResult`,
`CameraCoarseResult`) that include quality assessments, issue detection, and confidence
scores. But these results **do not feed back into sampling decisions** for subsequent
windows. The sampling plan is fixed at recording start; coarse QA results are consumed
only by the dense QA planner (for suspicious regions within the same window).

This is an open-loop system: sense (signal detectors) → act (sample), but no
sense (QA results) → act (adjust sampling). The dense QA planner partially closes
this loop within one window, but not across windows.

### Approach

**Model-guided adaptive sampling** — use coarse QA results from window N to inform
sampling decisions for window N+1 and beyond:

1. **QA-driven upgrade requests**: When `CoarseQAResult` flags a camera as DEGRADED or
   UNUSABLE, or when `ProductQAIssueEvidence` identifies a quality issue with high
   confidence, generate `AdaptiveUpgradeRequest` with reason `COARSE_UNCERTAINTY`
   (this reason already exists in the enum but is not populated by the QA pipeline).

2. **Cross-camera disagreement triggers**: When `CoarseQAResult` shows significant
   disagreement between cameras covering the same temporal region (e.g., cam_01 says
   GOOD but cam_02 says DEGRADED), generate `CROSS_CAMERA_DISAGREEMENT` upgrade
   requests. This is a strong signal that the region needs more evidence.

3. **Event-candidate-triggered dense sampling**: When `CandidateReductionResult` produces
   a candidate event, the boundary regions need higher sampling density for accurate
   onset/offset estimation. Generate `BOUNDARY_REFINEMENT` upgrade requests (this
   reason also exists in the enum but is not populated).

4. **Feedback loop architecture**:
   ```
   Window N: signal detectors → base sampling → decode → coarse QA
                                                     ↓
   Window N+k: QA results → upgrade requests → adjusted sampling → decode → coarse QA
   ```
   The feedback delay is `k` windows (k = pipeline depth from §3). With 2s windows
   and 1s hop, the feedback latency is 2–6 seconds — acceptable for slowly evolving
   quality issues (lens blur, lighting change) and marginal for fast events (occlusion).

### Invariant Protection

- **Sampling plan versioned**: `AdaptiveCoveragePolicy.version` already exists.
  Model-guided upgrades change the effective policy for subsequent windows, which
  must be recorded as a new policy version or a policy extension.
- **Determinism preserved**: The upgrade request is a deterministic function of
  coarse QA results (which are themselves deterministic). Replay produces the same
  upgrade requests.
- **Budget bounded**: `AdaptiveCoveragePlan` already enforces `max_upgrade_requests`
  and `max_targets_per_camera`. Model-guided upgrades consume the same budget —
  they don't create unbounded sampling.

### Expected Impact

- Sampling efficiency: instead of uniform sampling, concentrate frames where the model
  is uncertain → fewer total frames for equivalent quality, or higher quality for
  the same frame budget
- Dense QA reduction: model-guided sampling may make some dense QA calls unnecessary
  (the region was already sampled densely enough), reducing inference cost
- This is the most architecturally natural quality improvement — it uses existing
  infrastructure (`AdaptiveUpgradeReason` enum already has the right values)

### Implementation Difficulty

**Medium**. The `AdaptiveUpgradeReason` enum already defines the reasons. The
implementation requires: (1) a callback from the QA pipeline to the sampling planner,
(2) cross-window state propagation (the sampling planner for window N+k needs access
to QA results from window N), and (3) version tracking for the adjusted policy.

---

## 15. [P1 / Quality] Quality-Aware Boundary Estimation — Camera-Weighted Refinement

### Current State

`BoundaryRefinementProjector` in `event_pipeline/boundary_refinement.py` uses
`_median_low_int` (lower median) for cross-camera boundary estimation. All cameras
contribute equally — a camera with DEGRADED quality has the same weight as one with
GOOD quality.

`_role_reduction_values` requires `minimum_observed_cameras` (default from policy)
to produce a boundary estimate. If fewer cameras observe, the result is INDETERMINATE.
But among the observing cameras, no quality weighting is applied.

### Approach

**Quality-weighted boundary estimation**: Use per-camera quality signals from
`CoarseQAResult` / `LocalMediaQualityReport` to weight camera contributions:

1. **Quality-derived weights**: For each observing camera, compute a weight based on:
   - `CameraCoarseResult.local_status`: GOOD=1.0, DEGRADED=0.5, UNUSABLE=0.0
   - `LocalQualityFlag` presence: FROZEN_CONTENT or OBSERVED_BLACK_LUMA → weight=0.0
   - Frame count: more frames in the observation → higher weight (better temporal resolution)

2. **Weighted median instead of lower median**: Replace `_median_low_int` with a
   weighted median that accounts for camera quality. A camera with weight 0.5
   contributes half as much to the position of the median as a camera with weight 1.0.

3. **Quality-aware uncertainty**: The boundary `uncertainty_ns` should expand when
   high-quality cameras disagree and contract when they agree. Current: uncertainty =
   max over cameras of (|center - estimate| + half-duration). Proposed: uncertainty =
   weighted max, where low-quality cameras contribute less to the uncertainty bound.

4. **Graceful degradation**: When `minimum_observed_cameras` is not met by GOOD cameras
   but is met by GOOD + DEGRADED, produce a boundary estimate with expanded uncertainty
   instead of INDETERMINATE. The current all-or-nothing threshold loses information
   from partially-observing cameras.

### Invariant Protection

- **Boundary identity unchanged**: `BoundaryRefinementResult` identity is based on
  semantic_sha256 of the result. Quality-weighted estimation produces different
  (better) estimates but the identity derivation is unchanged.
- **Policy versioned**: Weight computation is governed by a new
  `boundary_refinement_quality_weight_policy_version`. Changing the weighting scheme
  = changing policy = new identity.
- **Fail-closed**: If quality information is unavailable (no coarse QA result for a
  camera), fall back to equal weighting (current behavior). Never assume quality.

### Expected Impact

- Boundary accuracy: weighted estimation should reduce boundary error when camera
  quality is heterogeneous (common in egocentric video — body-mounted cameras
  frequently get occluded or shaken)
- Uncertainty calibration: quality-aware uncertainty better reflects actual estimation
  confidence, enabling downstream consumers to make informed decisions
- Reduced INDETERMINATE outcomes: graceful degradation with DEGRADED cameras recovers
  boundary estimates that are currently lost

### Implementation Difficulty

**Low-Medium**. The `_median_low_int` function is small and self-contained. Adding
weighted median is a local change. Quality signal propagation from coarse QA to
boundary refinement requires a new parameter in the projector call, but the data
is already available in the pipeline state.

---

## 16. [P1 / Architecture] Early Window Reduction — Skip Stages When No Events Are Possible

### Current State

The 5-stage DAG always executes all stages for every window:
```
WINDOW → QA_COARSE → QA_DENSE → EVENT_PROPOSAL → WINDOW_REDUCTION
```

But many windows contain no events (the recording is mostly quiescent). For these
windows, EVENT_PROPOSAL returns `NO_EVENTS`, and WINDOW_REDUCTION produces a trivial
result. The QA_DENSE stage is particularly expensive (GPU inference on suspicious
regions) and is wasted when there are no events to refine.

The `_stream_terminal_outcome` function in `pre_eos_execution.py` (line 246) already
maps EVENT_PROPOSAL with empty output to `TerminalOutcome.NO_EVENTS`. But this mapping
happens **after** the stage executes, not before.

### Approach

**Early-exit DAG execution**: Allow stages to be skipped when upstream evidence
indicates they would produce trivial results:

1. **QA_DENSE skip when no suspicious regions**: If `CoarseQAResult.local_status` is
   COMPLETE for all cameras (no DEGRADED or UNUSABLE regions), skip QA_DENSE entirely.
   The `QACompletionProjector` already handles this case (status = QA_COMPLETE without
   dense evidence). Currently, the pipeline still invokes the dense planner, which
   returns an empty manifest — but the planning overhead (sorting, grouping, padding)
   is still incurred.

2. **EVENT_PROPOSAL early exit**: If `QACompletionResult` indicates all cameras are
   GOOD with no issues, the probability of finding an event is low. Allow a
   configurable `event_proposal_skip_threshold`: if the maximum issue severity across
   all cameras is below the threshold, skip EVENT_PROPOSAL and emit NO_EVENTS directly.
   This is a policy decision, not a correctness shortcut — the skip must be recorded
   as `SKIPPED_POLICY` with reason code `NO_SUSPICIOUS_EVIDENCE`.

3. **WINDOW_REDUCTION fast path**: When all upstream stages are SUCCEEDED or
   SKIPPED_POLICY, WINDOW_REDUCTION can use a simplified reduction that skips
   cross-referencing and Merkle root computation for trivial results.

### Invariant Protection

- **Every eligible clip still gets a complete result**: Skipping a stage produces
  `SKIPPED_POLICY` (a terminal state in `StreamWorkItemState`), not a missing result.
  The 21-class product QA cascade still covers every clip — skipped stages contribute
  `INCOMPLETE_INPUT` for the affected classes, which is the correct semantics.
- **DAG dependencies preserved**: Skipped stages still have their dependency edges
  in `work_dependencies`. Downstream stages see `SKIPPED_POLICY` as a successful
  dependency (it is in `SUCCESSFUL_DEPENDENCY_STATES`).
- **Policy versioned**: The skip threshold is a versioned policy parameter.
  Changing the threshold = changing policy = new identity.

### Expected Impact

- For quiescent recordings (estimated 70–80% of windows in typical egocentric video):
  skip QA_DENSE + EVENT_PROPOSAL → save 2 GPU inference calls per window
- For the fixture's 1,800 windows: potentially 1,200–1,400 windows skip 2 calls each
  = 2,400–2,800 fewer inference calls (NOT_MEASURED — depends on content)
- This is the single highest-impact optimization for recordings with low event density

### Implementation Difficulty

**Medium**. The DAG topology and stage execution are in `stream_scheduler.py`.
Early exit requires: (1) a policy predicate per stage that evaluates upstream evidence,
(2) a `SKIPPED_POLICY` terminal outcome when the predicate fires, and (3) downstream
stage adaptation to handle skipped inputs. The `TerminalOutcome.SKIPPED_POLICY` state
already exists and is handled by downstream stages.

---

## 17. [P1 / Quality] Inference-Result Caching Across Windows — Deduplication at the Semantic Level

### Current State

Each window's inference is independently dispatched through the orchestrator, which
checks for existing selections via `_selected_terminal_for_initial_delivery`. This
provides deduplication within a single run (replay safety), but not across windows.

With 50% frame overlap (hop=1s, window=2s), adjacent windows share many of the same
frames. If the shared frames produce identical inference inputs (same images, same
prompt, same config), the inference results are identical — but the orchestrator
dispatches them again because the `logical_invocation_id` differs (it includes the
window identity).

### Approach

**Semantic-level inference caching**: Cache inference results by input semantic digest,
not by logical invocation ID:

1. **Input digest as cache key**: The `InferenceInputPlan.input_plan_id` already
   captures the complete semantic identity of the inference input (request catalog,
   rendered items, prompt, config). If two windows produce the same `input_plan_id`,
   their inference results are identical.

2. **Cross-window cache hit**: Before dispatching to the adapter, check if a
   `ModelInference` with the same `input_plan_id` (or `rendered_input_digest`) already
   exists in the evidence ledger. If so, create a new `InferenceAttemptSelection`
   pointing to the existing terminal, without re-dispatching.

3. **Cache scope**: The cache is recording-scoped (within one recording's stream run).
  Cross-recording caching is possible but requires careful handling of model version
  drift and is not proposed here.

4. **Interaction with §5 (frame reuse)**: Frame reuse reduces decode cost. Inference
   caching reduces inference cost. They are complementary: frame reuse ensures the
   same pixels are available; inference caching ensures the same pixels don't trigger
   redundant inference.

### Invariant Protection

- **Identity unchanged**: Each window still has its own `logical_invocation_id` and
  `inference_id`. The cached result is referenced by a new selection that binds it
  to the current window's logical invocation. The original terminal's identity is
  preserved.
- **Evidence chain complete**: The cached terminal is already in the evidence ledger
  with its full lineage (intent → raw → parsed → selection). The new selection
  creates a second reference to the same terminal, which is correct — the terminal
  is content-addressed and immutable.
- **Replay preserved**: On replay, the cache is populated from the evidence ledger,
  so the same cache hits occur.

### Expected Impact

- Inference call reduction: for 50% frame overlap, estimated 30–50% of QA_COARSE
  calls are cache hits (the shared frames produce the same input digest). QA_DENSE
  and EVENT_PROPOSAL have lower hit rates because their inputs include window-specific
  context.
- This is a pure software optimization that reduces GPU cost without changing any
  hardware

### Implementation Difficulty

**Low**. The orchestrator already has `_selected_terminal_for_initial_delivery` for
deduplication. Extending it to check by `rendered_input_digest` is a small change.
The evidence ledger needs an index on `rendered_input_digest` for efficient lookup.

---

## 18. [P2 / Architecture] Completion Audit Lazy Verification — Defer Cross-Reference Checks Off Critical Path

### Current State

The primary completion path in `primary_completion.py` constructs a
`CanonicalPrimaryCompletionDetail` with 30+ fields, including:
- `processing_run` + `run_memberships` (logical node registry writes)
- `coarse_qa_result` + `qa_completion_result` + `dense_qa_executions`
- `event_proposal_result` + `candidate_reduction_result`
- `action_evidence_executions` + `provisional_fusion_result`
- `boundary_refinement_executions` + `final_fusion_context`
- `input_plan` + `reference_catalog` + `part_results`
- `barrier_reduction` + `fusion_reduction` + `output_decision`
- `hypotheses` + `prepared_identities` + `action_event_publications`

The `PrimaryCompletionRecord` includes 9 Merkle-like count+root pairs
(run_membership, barrier_member, hypothesis, identity_assignment, etc.).
Computing these roots requires reading back all member records and hashing them
in sorted order — this is O(N) in the number of members per category.

The `validate_registered_primary_completion_record` function resolves both schema
refs against the registry and validates the JSON payload. This is done on the
critical path (during completion), adding latency proportional to the detail size.

### Approach

1. **Lazy Merkle root computation**: Compute count+root pairs in a background task
   after the completion record is persisted. The initial persistence stores counts
   only; the roots are backfilled asynchronously. The completion record is still
   valid without roots (they are evidence, not identity).

2. **Deferred schema validation**: `validate_registered_primary_completion_record`
   resolves schema refs and validates the JSON payload. Move this to a post-commit
   verification step. The Pydantic model validation (which is fast) remains on the
   critical path; the JSON Schema validation (which is slower) is deferred.

3. **Incremental membership attachment**: Instead of collecting all
   `ProcessingRunNodeMembership` records and computing the root at completion time,
   attach memberships incrementally as each stage completes. The root is computed
   once at finalization from the already-persisted records.

### Invariant Protection

- **Completion identity unchanged**: `PrimaryCompletionRecord.semantic_sha256` includes
  the count+root pairs. If roots are computed lazily, the initial record has
  placeholder roots and the semantic_sha256 is computed after backfill. This requires
  a two-phase persistence: insert with placeholder, update with real roots.
  The `row_version` column tracks this transition.
- **Fail-closed**: If the background root computation fails, the completion record
  has incomplete roots. Downstream consumers (review, outbox) must check root
  presence before relying on them. Missing roots → evidence class downgrade
  (LOCAL_CONFORMANCE without root proof).

### Expected Impact

- Completion latency: from O(N_members) to O(1) for the critical path
- For recordings with many events (hundreds of hypotheses, thousands of membership
  records), this can save 100–500 ms of serialization + hashing time
- Frees the completion thread for the next recording

### Implementation Difficulty

**Medium**. Requires two-phase persistence for `PrimaryCompletionRecord` and a
background task for root computation. The existing `row_version` column supports
the transition. The key risk is ensuring that crash recovery handles the
"roots not yet computed" state correctly.

---

## 19. [P2 / Quality] Temporal Consistency Scoring — Window-to-Window Quality Signal

### Current State

Each window's QA result is independent. There is no mechanism to detect temporal
inconsistencies across windows, such as:
- Window N reports a camera as GOOD, window N+1 reports it as DEGRADED, with no
  quality flag explaining the transition
- An event candidate in window N has no corresponding candidate in window N+1
  despite temporal overlap (the event "disappears" and "reappears")
- Boundary estimates for the same event differ by >1s between adjacent windows

These inconsistencies are not errors (each window's result is internally consistent),
but they indicate either model uncertainty or input quality transitions that should
be flagged for review.

### Approach

1. **Per-camera quality trajectory**: Maintain a sliding window of
   `CameraCoarseResult.local_status` values per camera. Detect:
   - Sudden status changes (GOOD → UNUSABLE without an intervening DEGRADED)
   - Oscillating status (GOOD → DEGRADED → GOOD → DEGRADED) — indicates model
     uncertainty at the boundary

2. **Event continuity score**: For each event candidate in window N, check if
   window N+1 has a candidate with overlapping interval + same label. Compute a
   continuity score: 1.0 if perfectly continuous, 0.0 if the event disappears.
   Low continuity → `INCOMPLETE` or review trigger.

3. **Boundary consistency check**: If the same event is detected in adjacent windows,
  compare their boundary estimates. If onset/offset differ by more than the
  combined uncertainty, flag as `BOUNDARY_INCONSISTENCY` (new ambiguity code in
  `BoundaryRefinementProjector`).

4. **Recording-level quality signal**: Aggregate per-camera trajectories and event
   continuity scores into a `TemporalConsistencyReport` that becomes part of the
   `CanonicalPrimaryCompletionDetail`. This provides a holistic quality view that
   per-window results cannot.

### Invariant Protection

- **Additive evidence**: Temporal consistency is a new evidence type, not a
  replacement for per-window results. It does not change any existing terminal
  state or identity.
- **Versioned**: The consistency scoring policy is versioned. Changing the scoring
  algorithm = new policy version.
- **Non-blocking**: Consistency scoring happens after WINDOW_REDUCTION for each
  window, using already-persisted upstream results. It does not block any stage.

### Expected Impact

- Detects model uncertainty that per-window analysis cannot see
- Provides a recording-level quality signal for production monitoring
- Enables targeted review: "this recording has temporal inconsistencies in
  cam_03 around t=45s" instead of "review all 1,800 windows"

### Implementation Difficulty

**Medium**. Requires cross-window state accumulation (sliding window of QA results)
and a new scoring module. The data is already available in the pipeline state;
the challenge is propagating it across window boundaries without adding per-window
persistence overhead.

---

## Appendix A: Invariant Protection Summary

Every optimization in this guide must respect the following invariants. Any conflict
requires an explicit schema registration / migration decision before implementation —
never an in-place edit of a released contract.

| Invariant | Scope | Protection mechanism |
|-----------|-------|---------------------|
| Released schema bytes/catalog pins are immutable | All | Schema registration workflow |
| Original MCAP bytes, timestamps, mapping, decode facts, provenance are never rewritten | Source | Append-only triggers, content-addressed storage |
| Derived artifacts bind source frame/time, exact bytes, policy, and calibration | Media | Lineage chain in `ArtifactRegistryEntry` |
| Every eligible clip gets a complete 21-class projection | QA | `ProductQACascadeProjector` coverage enforcement |
| Event candidate/proposal/boundary evidence retains lineage and deterministic replay | Events | Content-addressed identity, append-only storage |
| Inference intent → raw → parsed → selection → accepted lineage closes in the ledger | Inference | `InferenceOrchestrationError` on gap |
| Work, barriers, leases, fences, retry/DLQ/backpressure, and recovery remain durable | Stream | Two-phase commit, lease fencing CAS |
| Logical identity is transport-independent; one terminal truth, no lost/duplicate completion | Identity | `WorkItem` as authority, broker messages as projections |
| Local, representative, external, and production evidence are never conflated | Evidence | Explicit `evidence_class` on every report |
| `production_eligible` is always `False` unless externally qualified | All | Hardcoded in every data structure |

New mechanisms introduced by this guide require additional invariant protections:

| New mechanism | New invariant | Protection |
|---------------|---------------|------------|
| NVDEC decode | `cuda_version` / `driver_version` in `CapabilitySnapshot` | Schema registration + pin |
| Continuous batching | Batch size does not enter identity | Same principle as "credentials don't enter identity" |
| Temperature calibration | `temperature` enters `enrichment_policy_version` | Change T = new policy = new identity |
| Conformal prediction | Prediction set size > 1 → `INCOMPLETE` (not `FAIL`) | Reuse existing state |
| Cross-window tracking | Track identity based on track-level evidence collection | New schema `candidate-event-v2` |
| Batched fsync | Crash-lost frames don't affect terminal truth | Two-phase commit guarantees |
| AIMD backpressure | Only adjusts concurrency, not terminal truth | Existing mechanism |
| Transaction collapse | Two-phase commit preserved; batch declare is idempotent | INSERT OR IGNORE on logical_key |
| Model-guided sampling | Upgrade reasons are versioned; budget bounded by existing caps | `AdaptiveCoveragePolicy.version` |
| Quality-weighted boundary | Weight scheme is versioned policy; fallback to equal weight | New `boundary_quality_weight_policy_version` |
| Early window reduction | Skipped stages emit `SKIPPED_POLICY` (existing terminal state) | Already in `SUCCESSFUL_DEPENDENCY_STATES` |
| Inference result caching | Cached terminal is immutable; new selection references it | Content-addressed evidence |
| Lazy completion audit | Two-phase persistence with `row_version`; missing roots → evidence downgrade | Existing column |
| Temporal consistency scoring | Additive evidence; does not change existing terminal states | New versioned policy |

---

## Appendix B: Evidence Class Requirements

Every performance or quality claim in this guide must be tagged with its evidence class.
No claim may be promoted to a higher class without the corresponding proof.

| Evidence class | Meaning | Production eligible |
|----------------|---------|-------------------|
| LOCAL_CONFORMANCE | Local mechanism verified with fixture | No |
| LOCAL_BENCHMARK | Local benchmark with real hardware | No |
| REPRESENTATIVE_BENCHMARK | Representative workload on production-equivalent hardware | Pending |
| EXTERNAL_QUALIFICATION | Third-party audited | Pending |
| PRODUCTION_QUALIFIED | Measured under production SLOs | Yes |

Current status: **LOCAL_CONFORMANCE**. All throughput projections in this guide that
involve real GPU models are **NOT_MEASURED** until representative benchmark evidence
is collected.

---

## Appendix C: Implementation Roadmap

```
Phase 1 (Month 1–2): Physical bottleneck breakthrough — target RTF 10× (LOCAL_BENCHMARK)
  §1  NVDEC hardware decode pipeline (advance P8)
  §6  JPEG/NVJPEG encode (quick win)
  §7  Zero-copy hash + batched fsync
  §5  Cross-window frame reuse
  §13 Transaction collapse — connection residency + batch declare (quick win)
  → Run 30-min capacity benchmark, verify RTF ≥ 10× on real hardware
  → Evidence class: LOCAL_BENCHMARK (requires real GPU)

Phase 2 (Month 2–3): Inference and concurrency breakthrough — target RTF 30×
  §2  vLLM continuous batching + prefix caching (native_batch_enabled = True)
  §4  SQLite single-write connection + batch writes (unlocks concurrency)
  §3  Stage-affine pipelining (one worker per stage first, then multi-worker)
  §16 Early window reduction — skip QA_DENSE/EVENT_PROPOSAL for quiescent windows
  → Verify RTF ≥ 30×, p95 ≤ 1s on real hardware
  → Evidence class: LOCAL_BENCHMARK

Phase 3 (Month 3–4): Quality breakthrough — target calibrated ≥ 90%
  §8  Temperature calibration + conformal prediction + ECE measurement
  §9  EVENT_TRACKING stage + cross-window trajectory (new schema version)
  §14 Model-guided adaptive sampling — close the sense-act loop
  §15 Quality-aware boundary estimation — camera-weighted refinement
  §17 Inference-result caching across windows — deduplication at semantic level
  → Requires representative labeled data (BLUEPRINT E1 gate)
  → Evidence class: REPRESENTATIVE_BENCHMARK

Phase 4 (Month 4–5): Retrieval, elasticity, and operational quality
  §12 Vector search + reranking (pgvector)
  §11 AIMD adaptive backpressure
  §18 Completion audit lazy verification — defer cross-reference checks
  §19 Temporal consistency scoring — window-to-window quality signal
  Schema cold-start + attribute normalization (maintainability)

Phase 5 (Month 5–6): Production qualification
  Follow BLUEPRINT E0–E6 gates:
  E2 NVDEC/R2 parity, E3 two-H100 topology, E4 24h soak, E5 500 rec-hours/day
  → Only after external review: set production_eligible = True
  → Evidence class: PRODUCTION_QUALIFIED
```

---

## Appendix D: Industry Benchmark Alignment

### Throughput targets (single H100 + 6-lane NVDEC)

| Metric | Current (LOCAL_CONFORMANCE) | Target | Evidence class |
|--------|-----------------------------|--------|----------------|
| RTF (rec-sec/wall-sec) | 2.16 | 20–50 | NOT_MEASURED |
| Decode throughput (fps) | ~180 (CPU PyAV) | ~1,800–3,600 | NOT_MEASURED |
| Inference throughput (img/s) | N/A (mock provider) | ~200–500 | NOT_MEASURED |
| p95 latency | 3.9 s | ≤ 0.5 s | NOT_MEASURED |
| recording-hours/day (single H100) | ~52 (projected) | 500–1,300 | NOT_MEASURED |

Reference points: Google Video Intelligence API (~1× real-time), Meta SAM2 (~3× real-time
on A100), ByteDance video understanding pipeline (batch 100×+ GPU utilization). The
20–50× target is the theoretical upper bound for "offline batch + 6-lane hardware decode
+ vLLM continuous batching" — it must be measured, not assumed.

### Quality targets

| Metric | Current | Target | Evidence class |
|--------|---------|--------|----------------|
| 21-class QA accuracy | NOT_MEASURED | ≥ 90% (calibrated) | NOT_MEASURED |
| ECE (calibration error) | None | < 0.05 | NOT_MEASURED |
| Event boundary IoU | NOT_MEASURED | ≥ 0.85 | NOT_MEASURED |
| Abstention quality | Policy-derived heuristic | Conformal guaranteed 90% coverage | NOT_MEASURED |
| Cross-window event coherence | None | Trajectory-level consistency | NOT_MEASURED |
