# Local Six-Camera Video Export Evidence - 2026-07-18

## Status and Scope

**Evidence class: local development, non-promotional.**

This is the historical frozen-V1 export report. The subsequent V2 exact-schema catalog,
artifact registry, typed lineage, replay, and registry-backed view evidence is recorded
in `reports/local-artifact-registry-v2-2026-07-18.md`. Counts below describe this V1 run
at the time it was executed, not the current V2 repository state.

This report records an isolated `MCAP -> six MP4` derived-media exercise under
`execution_mode=LOCAL_DEVELOPMENT_OVERRIDE`. The mapping remains
`mapping_profile.approved=false`, `ready_manifest_id=null`, and
`alignment_status=UNVERIFIED`. The result is not an `MCAPReadyManifest`, does not update
a source or alignment ledger, and does not satisfy a Phase 1B exit gate.

At the application boundary, this slice configures no network or provider path and made
no Qwen, GPT, model-provider, or upload request. Provider request count, frames sent,
tokens, and provider cost were all zero; host packet telemetry was not measured.

## Inputs and Configuration

| Item | Exact value |
|---|---|
| Execution specification | `large_scale_6camera_video_agent_execution_spec.md` |
| Execution-spec SHA-256 | `434902fed026726e9e4924042dd1f3f2d2ec26172011efe188aac3f7986e3c0a` |
| Architecture SHA-256 | `40cb12e713e138335dbf4bab5412039fa494f2476392f1ace312b332b186ecff` |
| Readable local source bytes | 130,303,923 |
| Readable local source SHA-256 | `9fd5094bf29cd4ee50cd8c7d8c053e89d1c93660a0f4e57daaa726bae2b6156c` |
| Recording identity | `d6ce6673fa1c6a35736cedb16b181b0b4a80a741d84961d416ba50e39e5ad7bc` |
| Corrupt local source bytes | 1,026,677 |
| Corrupt local source SHA-256 | `530a5e9a18ce2143b6143577fe92ea32a23462d00972329453cc4f2621adcf1c` |
| Mapping version | `genrobot-observed-v0` |
| Mapping semantic SHA-256 | `7fffbb2cc313cacd84dc7910b38c366c4499e9dcd1fd65afcd74719b55ee8999` |
| Mapping approval | `false` |
| Exporter | `robata.pyav_h264_mp4_exporter` `0.1.0` |
| Export profile | `direct-h264-remux-no-reordering` `1.0` |
| Export mode | `REMUX` |
| Canonical config SHA-256 | `085338db2bc0cb9b4ce70c1f1fcf31bc2583dcafb09d4d696d5e74bb64e8160c` |
| Python / PyAV | 3.13.5 / 18.0.0 |
| MCAP packages | `mcap` 1.4.0; `mcap-protobuf-support` 0.5.4 |
| Host observation | Windows 10.0.26200, AMD64, 32 logical processors |
| Source-tree revision | `UNAVAILABLE`: the local `.git` directory is not a usable repository |

The exporter requires Foxglove `CompressedImage` H.264, Annex-B access units, an
SPS+PPS+IDR bootstrap, strictly increasing log time, and no B-frame or delayed-frame
reordering. It writes PTS/DTS in integer timebase `1/1,000,000,000`, uses the first
exported source log time as zero, and estimates the final duration using
`MEDIAN_POSITIVE_INTERVAL` with `HALF_EVEN` arithmetic.

## Complete Export Results

Two independent output directories were generated from the same source and configuration.
Each directory contains six MP4 files, six registered-schema timestamp JSONL sidecars,
and one canonical `CameraVideoExportManifest`.

| Camera | Input | Leading drop | Output packets/frames | Keyframes | MP4 bytes | MP4 SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| `cam_01` | 1,226 | 1 | 1,225 / 1,225 | 41 | 20,980,510 | `4337aefbc597a28fa97c10f17ea24555ad03f694b10d3710ac5f96022a565b47` |
| `cam_02` | 1,226 | 1 | 1,225 / 1,225 | 41 | 21,000,336 | `8a314d2129dbc339e1aa0223f06450016fc099175615ecfb811b32c0bf7f54d2` |
| `cam_03` | 1,226 | 1 | 1,225 / 1,225 | 41 | 20,984,009 | `56e94951980f8ebeb15450114f863546daa9dc0d6a2ac1d8080bae32e7036c3f` |
| `cam_04` | 1,226 | 1 | 1,225 / 1,225 | 41 | 20,988,589 | `d46bb16d0a1d1f313fd1da7a91bd78f3d73bd989fdabb99f44f84fe18542de4f` |
| `cam_05` | 1,226 | 1 | 1,225 / 1,225 | 41 | 20,959,558 | `13ec4a292fcc1648a330ad1da691872351669620a9d7671da3eb832f955a5a1a` |
| `cam_06` | 1,226 | 1 | 1,225 / 1,225 | 41 | 20,993,128 | `d6040ff1e1f7e64ad9f9fce9a394e3bab81b92c029d3f38711cb310861c6d935` |

Every leading drop has reason
`BEFORE_FIRST_DECODABLE_KEYFRAME`, an exact source timestamp range, and count 1.
Every trailing drop has reason `NONE`, count 0, and null timestamps. For every camera:

```text
1226 input messages = 1 leading drop + 1225 exported packets + 0 trailing drops
1225 timestamp rows = 1225 exported packets
```

Each timestamp row is canonical JSON, conforms to the registered
`camera-video-timestamp-row` schema, identifies packet ordinal and source sequence, and
records source log/publish/header nanoseconds, PTS, DTS, duration, keyframe state, profile,
and integer timebase. Maximum observed timestamp mapping error was 0 ns.

## Artifact Accounting

One complete output directory contains:

| Class | Files | Bytes |
|---|---:|---:|
| MP4 | 6 | 125,906,130 |
| Timestamp JSONL | 6 | 3,737,508 |
| Canonical manifest | 1 | 9,572 |
| **Total** | **13** | **129,653,210** |

The canonical manifest SHA-256 is
`968d3fdccd34d67d1ca6fe43f1b76143f7d3407ac068714a553db62bbebf323f`.
Output paths are local locators only and are absent from logical identity.

## Determinism and Reuse

| Observation | Result |
|---|---|
| Independent export A wall time | 66.353065 s |
| Independent export B wall time | 66.522104 s |
| Exact file comparison | All 13 file sizes and SHA-256 digests identical |
| Manifest comparison | Exact canonical bytes and digest identical |
| MP4 comparison | All six exact-byte digests identical |
| Timestamp comparison | All six exact-byte digests identical |
| Existing-output verification/reuse | `reused=true`, 6.5 s, no rewrite |

The source was rehashed after camera export and before directory publication. Its size and
SHA-256 still matched the inspected input. Strict reuse revalidated the manifest, current
inspection and mapping facts, artifact layout and bytes, sidecar semantics, and source
hash before returning. Each new run wrote into a private sibling staging directory,
verified every returned fact and artifact, then renamed the complete directory once.
Unit tests injected an exporter failure and an invalid sidecar; neither published a
complete directory and both removed owned staging data.

## Independent Media Validation

After publication, a separate read-only PyAV pass compared both output sets byte-for-byte
and fully decoded all six files from export A:

- PyAV 18.0.0 and FFmpeg `libavcodec` 62.28.102 / `libavformat` 62.12.102.
- Exactly one H.264 High video stream per MP4.
- Integer stream timebase `1/1,000,000,000`.
- 1,225 decoded `1600 x 1300` frames and 41 keyframes per camera.
- 7,350 decoded frames and 246 keyframes in total.
- Full independent comparison/decode wall time: 19.612966 s.

This proves local media readability for these exact bytes. It is not a codec/profile
approval for a representative corpus; OD-MEDIA-001 remains open.

## Failure Closure

The known corrupt local source returned exit code 2 with stable code `CORRUPT_MCAP`:

```text
unknown (opcode 48) record has length 3472328296227680304
that exceeds limit 4294967296
```

Observed failure wall time was 0.376957 s. The requested result directory did not exist
after failure. Provider request count was zero.

The automated suite also covers authorization failure before source access, output
appearance/mismatch, exact-byte mismatch, source mutation before publication,
nonmonotonic source time, unsupported schema/codec, missing bootstrap, frame reordering,
decode-validation failure, invalid timestamp metadata, and non-overwriting pair commit.

## Diagnostic Rate, Not Capacity

The six exported media durations total 245.000801 camera-video seconds, equivalent to an
average 40.833467 recording seconds across six equal-duration slots. For the two local
runs only:

| Run | Recording-h/wall-h | Camera-video-h/wall-h |
|---|---:|---:|
| A | 0.615397 | 3.692381 |
| B | 0.613833 | 3.682998 |

These are single-file, cache-state-uncontrolled diagnostics. The implementation rereads
the source per camera and performs complete source and output decode validation. CPU
utilization, peak RSS, disk throughput, cache state, and power were not instrumented.
Therefore these ratios are not a benchmark, production estimate, or capacity verdict.

OD-SCALE-001 is unresolved, so both incoming-work interpretations remain mandatory:

| Scenario | Incoming per day | Required recording-h/wall-h | Required camera-video-h/wall-h |
|---|---:|---:|---:|
| A: 500 recording hours/day | 500 recording h / 3,000 camera-video h | 20.833333 | 125.000000 |
| B: 500 aggregate camera-video hours/day | 83.333333 recording h / 500 camera-video h | 3.472222 | 20.833333 |

No claim is made that either scenario is met.

## Resource, Quality, and SLO Accounting

| Measure | Result |
|---|---|
| CPU utilization / CPU-hours | `NOT_MEASURED` |
| Peak RSS | `NOT_MEASURED` |
| Disk read/write throughput | `NOT_MEASURED` |
| GPU utilization / GPU-hours | `NOT_MEASURED` |
| Network bytes | No network path exists in this slice; packet telemetry `NOT_MEASURED` |
| Provider requests / frames / tokens / cost | 0 / 0 / 0 / 0 |
| QA accuracy, precision, recall | `NOT_MEASURED`; no ground truth or inference |
| Event/action metrics | `NOT_MEASURED`; no event inference |
| T+1 verdict | `UNRESOLVED`; OD-SLO-001 is open |
| Production capacity verdict | `UNRESOLVED` |

## Acceptance Assessment

Repository verification after implementation:

- Default full suite: `165 passed, 1 skipped` in 31.15 s. The one skip is the deliberately
  opt-in double full-export test.
- Integration module with explicit real six-camera acceptance enabled:
  `ROBATA_RUN_REAL_EXPORT_ACCEPTANCE=1`, `3 passed` in 214.08 s; this includes the
  independent double full-export test.
- Targeted contract/service/mapping/CLI set: `66 passed, 1 skipped` in 4.95 s.
- Ruff formatting: 38 files passed the format check; Ruff lint clean.
- Strict mypy: 22 source files, no issues.
- Offline registry: 6 JSON Schema Draft 2020-12 documents verified.
- `uv lock --check --offline` and `uv sync --dev --locked --offline` both passed for 34
  resolved packages.

The local slice proves complete six-slot output, exact provenance currently represented
by the registered contracts, media validity, timestamp conservation, exact-byte
determinism, corrupt/injected failure closure, precise local boundary labels, and provider
isolation.

It does **not** close these promotion blockers:

- Phase 0 security, privacy, governance, retention, audit, and provider approvals.
- Governed representative corpus admission, O-03 source mapping, and O-04 clock evidence.
- Selected VALID source report, durable source, `MCAPReadyManifest`, or admissible
  alignment.
- OD-SCALE-001, OD-SLO-001, OD-QUALITY-001, or OD-MEDIA-001.
- This frozen V1 result has no artifact-registry `artifact_id`, lifecycle, immutable
  locator version, or parent-ID integration. ADR 0003 and the V2 follow-up report close
  that gap for the local V2 export slice, not for V1 or the whole Phase 1A gate.
- Repeated cold/warm-cache benchmarks with CPU, memory, disk, GPU, and failure-rate
  telemetry.

The outputs under `tmp/` are ignored local evidence, not a durable artifact store.
