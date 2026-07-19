# Local MCAP Development Spike Report

- Date: 2026-07-18
- Status: Non-promotional local development evidence
- Architecture: `ARCHITECTURE_DESIGN_V1.md` V1.1
- Promotion decision: **DO NOT PROMOTE Phase 1B**

This report uses the minimum phase-report fields required by Architecture Section 23. It
records a developer-directed local probe, not governed corpus admission. No source frame,
prompt, or model output was transmitted to a provider. Phase 0 approval, O-03, O-04, and
an approved representative corpus remain missing.

## Implemented

- Official MCAP reader inspection with CRC validation, complete channel inventory, stable
  adapter error codes, streaming source size, and lowercase SHA-256.
- URI-independent recording identity derived from namespace and source-content SHA-256.
- Strict exact-topic mapping for `cam_01` through `cam_06`; observed profiles fail closed
  unless a caller supplies the explicit local-development override.
- A real PyAV/FFmpeg H.264 decoder probe that preserves failed access-unit diagnostics.
- Exact anchored rational timestamp transforms, piecewise source-order clock epochs,
  HALF_EVEN sampling grids, tolerance, canonical tie-breaks, decode-failure outcomes, and
  selected-frame deduplication.
- Unit, property, contract, and local real-sample integration tests.

## Input

The local archive inventories 37 MCAP members, but only two deliberately selected samples
were extracted for this spike. The selection is not random or representative.

| Input | Exact bytes | SHA-256 | Admission state |
|---|---:|---|---|
| `sample-medium.mcap` | 130,303,923 | `9fd5094bf29cd4ee50cd8c7d8c053e89d1c93660a0f4e57daaa726bae2b6156c` | Local diagnostic only |
| `sample-small.mcap` | 1,026,677 | `530a5e9a18ce2143b6143577fe92ea32a23462d00972329453cc4f2621adcf1c` | Local corrupt case only |
| `genrobot-observed-v0.json` | 574 | `42ca37abf7dd15bdc63d1c1064469db91e78ce18efc00182a9edbe261fe83c50` | `OBSERVED` / `UNAPPROVED` |

The archive digest is `NOT_COMPUTED`; its 18,873,393,164-byte size and member inventory do
not identify an admitted corpus. The source-tree revision is `UNAVAILABLE` because this
workspace is not currently a Git repository. The locked Python environment is identified
by `uv.lock` SHA-256
`3db137d7547f9542660942a6ed2eda0a8a0463dff5926bf28c5378b80690888a`.

## Output

For `sample-medium.mcap`, the probe observed:

- Header profile/library `Genrobot` / `libmcap`, 17 channels, and 16,210 messages.
- Exactly six configured `foxglove.CompressedImage` topics, 1,226 messages per camera,
  protobuf message encoding, and declared `h264` format.
- Six successful payload decoder probes at 1600 x 1300. Each stream preserved one initial
  H.264 decode failure before a subsequent access unit produced the first frame.
- Recording identity
  `d6ce6673fa1c6a35736cedb16b181b0b4a80a741d84961d416ba50e39e5ad7bc`
  for namespace `robata` and the source-content digest above.

For `sample-small.mcap`, the reader returned exit code 2 and stable code `CORRUPT_MCAP` for
an unknown opcode whose claimed record length exceeded the configured MCAP record limit.

A separate exploratory timestamp pass observed equality of MCAP log time, publish time,
and embedded compressed-image time for all 7,356 mapped messages. Frame-index-relative
camera offsets had observed p95 absolute values no greater than 50 microseconds and an
observed maximum of 73 microseconds. These facts do not establish a common hardware clock,
synchronization method, uncertainty bound, or admissible alignment; alignment remains
`UNVERIFIED` until O-04 is approved and an alignment report is emitted.

The CLI prints exploratory JSON to stdout. It does **not** publish an
`MCAPValidationReport`, `MCAPReadyManifest`, alignment manifest, or durable ledger row.

## Schema and Schema Version

- Contract registry: checked-in JSON Schema 2020-12 documents under `schemas/v1`.
- Mapping profile: `genrobot-observed-v0`, exact-topic policy,
  `foxglove.CompressedImage`, explicitly unapproved.
- Time/sampling behavior: Architecture V1.1 Section 25.3.
- CLI inspection JSON: `UNREGISTERED_EXPLORATORY_OUTPUT`; it cannot be treated as a
  validation report or READY manifest.

## Architecture Changes and Decision Records

- Architecture V1.1 Section 25 supplies normative trust, identity, time, admission,
  review, security-gate, and conformance clarifications.
- `docs/adr/0001-executable-baseline.md` separates Phase 1A contract work from governed
  Phase 1B real-source admission.
- `docs/architecture/qa-policy-input-v0.md` preserves source QA ambiguities without
  resolving O-10.

No open decision was silently closed. The observed mapping is a candidate for O-03, not an
approved answer. The timestamp observations are evidence input for O-04, not an approved
clock model.

## Test/Benchmark Corpus and Configuration Digest

- Corpus size exercised: 2 local MCAPs out of 37 inventoried members.
- Expected-path case: 1 intentionally selected readable source.
- Invalid-path case: 1 intentionally selected corrupt source.
- Mapping configuration digest: listed in the Input table.
- Decoder: PyAV 18.0.0 using its bundled/linked FFmpeg implementation.
- MCAP packages: `mcap` 1.4.0 and `mcap-protobuf-support` 0.5.4.
- Repetitions/warm-up/random seed: one diagnostic CLI invocation per reported wall time;
  no warm-up removal and no random sampling.
- Minimal runtime verification: `uv sync --no-dev --locked` left MCAP/PyAV absent by
  design, while all core package imports passed.
- Full development verification: 109 tests passed with the MCAP toolchain installed.

This configuration is insufficient for a benchmark or confidence interval.

## Recording Hours and Camera-Video Hours

Official recording hours and camera-video hours are `NOT_MEASURED`: the sources are not an
approved representative corpus, and first/last message timestamps are observations rather
than authoritative half-open recording bounds.

For scale only, the medium file's observed container message span was 40.890455 seconds
(0.01135846 observed-span hours). The sum of the six mapped cameras' first-to-last message
spans was 245.000807 seconds (0.06805578 observed camera-span hours). These values are not
promoted denominators.

## Throughput and Wall Time

- Full medium inspection, hashing, mapping, and six decoder probes: one local observation
  of 0.753137 seconds.
- Corrupt small-source failure path: one local observation of 0.418380 seconds.
- Throughput: `NOT_MEASURED`; a single warm-cache state with no host/resource capture is
  not a throughput result.

## Latency

Average, p50, p95, and p99 queue/service/API latency are `NOT_MEASURED`. There is no queue
or provider API in this slice, and one end-to-end diagnostic observation cannot estimate a
distribution.

## Resource Usage

CPU, GPU, memory, disk, network, and storage rates are `NOT_MEASURED`; no synchronized
resource instrumentation was active. The probe performed no network transfer and created
no retained frame artifacts.

## API Usage and Cost

Qwen/GPT requests, images, and tokens were all 0, and provider cost was USD 0 for this
spike. Local compute and storage cost are `NOT_MEASURED`.

## Failures, Retries, Invalid Outputs, and Quarantine

- Structural outcomes: 1 readable and 1 corrupt source out of 2 deliberately selected
  cases. The resulting 50% test-case ratio is not a corpus failure-rate estimate.
- Decoder terminal outcomes: 6 successful probes out of 6 mapped cameras.
- Preserved payload diagnostics: 6 initial H.264 decode errors; no retry policy was used.
- Retries: 0. Provider invalid outputs: not applicable. Durable quarantine: not
  implemented, so no quarantine-rate claim is possible.

Manual schema exploration also found unreadable descriptor sets on two auxiliary schema
types. The current vertical slice intentionally validates mapped image channels but does
not yet emit complete auxiliary-schema diagnostics; this is an admission-report gap.

## Quality Metrics and Confidence Intervals

`NOT_MEASURED`. There is no registered ground truth, approved QA aggregation policy,
representative split, repeated-run design, or O-10 promotion threshold.

## Known Bottlenecks

- Source hashing and MCAP scanning are separate full reads.
- The decoder probe reopens the source for each camera, producing six additional scans.
- The CLI output lacks a registered schema and durable artifact identity.
- Auxiliary schema diagnostics, validation-report selection, READY publication, source
  and alignment ledgers, and alignment policy execution are not implemented.
- No resource telemetry or repeated cold/warm-cache benchmark exists.

## Open Questions and Decisions

- Phase 0 governance approval and O-15 security/privacy/retention controls.
- O-03 source topics, schemas, camera-role meaning, auxiliary-channel policy, codec/rate,
  and keyframe/decoder expectations.
- O-04 clock provenance, synchronization evidence, reset/drift behavior, residual limits,
  and missing-frame tolerance.
- O-10 six-camera QA aggregation and acceptance semantics.
- Approved representative corpus, workload digest, and source-system owner sign-off.

## Next Step and Promotion/Rollback Decision

Keep Phase 0 and Phase 1A open. Do not promote Phase 1B and do not publish READY from this
spike. Next, complete the remaining Phase 1A identity/revision/artifact primitives and
obtain Phase 0, O-03, and O-04 decisions. After those gates pass, implement the immutable
validation-report/READY split, explicit alignment report, and independently reconcilable
source/alignment ledgers, then rerun against an approved representative corpus.
