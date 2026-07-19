# Local Artifact Registry and Six-Camera V2 Evidence - 2026-07-18

## Status and Scope

**Evidence class: local development, non-promotional.**

This report records the implemented V2 schema-catalog, immutable artifact-registry,
typed-lineage, six-camera export, replay, and registry-backed materialized-view slice
defined by ADR 0003. V1 schema bytes and V1 reader behavior remain frozen; there is no
V1-to-V2 upcast because an upcaster cannot invent artifact identity or lineage.

The real-media exercise used `execution_mode=LOCAL_DEVELOPMENT_OVERRIDE`,
`mapping_profile.approved=false`, `ready_manifest_id=null`, and
`alignment_status=UNVERIFIED`. It published no `MCAPReadyManifest`, changed no source
or alignment ledger, and made no Qwen, GPT, upload, or other provider request.

This evidence closes the artifact-registry blocker for this local V2 derived-media slice.
It does **not** declare Phase 0, all of Phase 1A, or Phase 1B complete.

## Governing Inputs

| Item | Exact value |
|---|---|
| Execution specification | `large_scale_6camera_video_agent_execution_spec.md` |
| Execution-spec SHA-256 | `434902fed026726e9e4924042dd1f3f2d2ec26172011efe188aac3f7986e3c0a` |
| Architecture decision | `docs/adr/0003-artifact-registry-and-schema-evolution.md` |
| Readable local source bytes | 130,303,923 |
| Readable local source SHA-256 | `9fd5094bf29cd4ee50cd8c7d8c053e89d1c93660a0f4e57daaa726bae2b6156c` |
| Recording identity | `d6ce6673fa1c6a35736cedb16b181b0b4a80a741d84961d416ba50e39e5ad7bc` |
| Mapping profile | `genrobot-observed-v0`, unapproved local override |
| Export mode | direct H.264 `REMUX` |
| Provider requests / frames / tokens / cost | 0 / 0 / 0 / 0 |

The source and outputs under `data/source/` and `tmp/` are ignored local evidence, not
governed production data or a durable production artifact store.

## Exact Schema Authority

At the time this artifact-registry evidence was captured, the catalog contained 11
pinned JSON Schema 2020-12 documents: six frozen V1 documents and five V2 documents. ADR
0004 later added two independent logical-node/run-membership contracts, bringing the
catalog to 13. ADR 0005 subsequently added three immutable-revision/selection contracts,
bringing the current catalog to 16 without changing this export snapshot's schema
closure. Each catalog entry fixes the logical schema
ID, semantic version, schema artifact ID, exact-byte SHA-256, document path/ID, owner,
lifecycle, compatibility policy, canonicalization/projection versions, and supported
software range.

The V2 export snapshot closes over four schema artifacts:

- V2 mapping-profile schema;
- V2 export-config schema;
- frozen V1 camera-video timestamp-row schema;
- V2 camera-video export-manifest schema.

Persisted payloads resolve the exact
`(schema_id, version, artifact_id, sha256)` reference. Ambiguous short-name lookup fails
closed. The verifier resolved every catalog entry by its exact reference and validated
all offline schema references.

## Published Derivation Identity

The first real V2 publication produced:

| Identity | Exact value |
|---|---|
| Logical derivation key | `camera-video-export:v2:1bab7c7d49490bc7afa601e54fa76d0f66d215840145f17e20564052be4f1f09` |
| Manifest artifact ID | `170ec194-a6af-d619-aead-d6294b284264` |
| Manifest exact-byte SHA-256 | `f7dfad56e34f2daf067508f376b93ea2d2db9e20ecebf5a07acd04d61f16ec29` |
| Manifest bytes | 22,032 |

The manifest body carries its exact schema reference, semantic-content digest, three
registered input references, and twelve registered camera-output references. The
manifest's own artifact ID, immutable locator/object version, byte size, and exact-byte
digest exist only in its external registry entry. Excluding those values from the body
avoids a self-referential digest.

Opaque artifact IDs, run/work IDs, creation time, hostnames, and locators are excluded
from semantic identity. Consumers do not infer business meaning from the locally
deterministic opaque IDs.

## Registry Snapshot and Lineage

The committed derivation snapshot contains exactly 20 immutable ACTIVE entries:

| Artifact type | Count |
|---|---:|
| Exact JSON Schema artifacts | 4 |
| Raw MCAP | 1 |
| Mapping profile | 1 |
| Export configuration | 1 |
| Camera MP4 | 6 |
| Camera timestamp map | 6 |
| Camera-video export manifest | 1 |
| **Total** | **20** |

The SQLite authority recorded:

| Registry row class | Count |
|---|---:|
| Artifact entries | 20 |
| Logical derivations | 1 |
| Typed parent edges | 51 |
| Immutable locations | 20 |

The 51 edges reconcile exactly: each of the twelve camera outputs has raw MCAP, mapping
profile, and export configuration parents (`12 * 3 = 36`); the manifest has the three
inputs plus all twelve camera outputs as parents (`3 + 12 = 15`). All parents exist in
the committed snapshot, are unique and canonically ordered, and the graph is acyclic.

Blob bytes are published and rehashed before the SQLite transaction exposes the logical
derivation. The database commit is publication authority. User-facing directories are
reconstructible views and are neither identity nor publication authority.

The observed local `registry.sqlite3` file was 135,168 bytes. This is a single
development database observation, not a storage-capacity, compaction, durability, or
production sizing result.

## Registry-Backed Views and Replay

Two different output directories were materialized from the same logical derivation and
shared registry:

| Observation | First publication | Registry replay |
|---|---:|---:|
| Wall time | 103.5 s | 9.4 s |
| Media derivation reused | no | yes |
| Existing view reused | no | no; a new view was materialized |
| Files in view | 13 | 13 |
| Exact-byte comparison | baseline | all 13 files identical |

Each view contains six MP4s, six timestamp NDJSON sidecars, and one canonical V2 manifest:

| Class | Files | Bytes |
|---|---:|---:|
| MP4 | 6 | 125,906,130 |
| Timestamp NDJSON, 622,918 bytes each | 6 | 3,737,508 |
| Canonical V2 manifest | 1 | 22,032 |
| **Total** | **13** | **129,665,670** |

Replay began with logical-key lookup, exact registry/DAG validation, input comparison,
and blob rehashing. It did not rerun the six-camera exporter. The new output directory
was then atomically materialized only from committed registry blobs.

An already existing view is accepted only if its exact 13-name layout, canonical
manifest bytes, and all content bytes match registry authority. Unexpected files,
missing files, symlinks, coherent local manifest/blob replacement, and any digest
mismatch fail closed.

The measured 103.5 s and 9.4 s values are single-source, cache-state-uncontrolled local
observations. CPU utilization, peak RSS, disk throughput, and power were not measured, so
these timings are not a throughput benchmark or production-capacity claim.

## Media and Timestamp Validation

All six MP4 artifacts were nonempty H.264 video outputs. An independent read-only media
probe decoded the first frame of every camera as `1600 x 1300`.

Every camera contained 1,226 input messages, one structured leading drop before the
first decodable keyframe, 1,225 exported packets/frames, and no trailing drop. Every
sidecar contained exactly 1,225 rows:

```text
1226 input messages = 1 leading drop + 1225 exported packets + 0 trailing drops
1225 timestamp rows = 1225 exported packets
```

Each NDJSON row was validated against the exact pinned timestamp-row schema and retained
source timestamps/sequence, packet ordinal, PTS/DTS, duration, keyframe state, export
profile, and integer timebase facts.

## Failure Closure and Recovery

Automated tests cover:

- invalid artifact records, exact-byte mismatches, semantic conflicts, missing parents,
  duplicate identities, cycles, schema-reference mismatches, and blob tampering;
- registry-first reuse and rejection of an unregistered pre-existing output directory;
- strict rejection of a tampered materialized view without changing committed blobs;
- exporter failure before registry publication;
- view-rename failure after registry commit followed by registry-only rematerialization;
- commit-uncertain recovery by querying and verifying the logical-key winner rather than
  publishing a competing derivation;
- rejection of a registry-valid foreign derivation whose camera output producer differs
  from the manifest exporter or whose timestamp artifact names the wrong exact schema;
- two independent registry connections missing concurrently with different publication
  clocks, then converging on one verified 20-artifact/one-derivation winner;
- pure, pinned, deterministic upcasting behavior, including the explicit absence of a
  V1-to-V2 camera-manifest path.

Failed pre-commit work exposes no logical derivation. A post-commit view failure does not
roll back the committed derivation; retry verifies registry authority and rematerializes
the view. Cleanup errors do not replace the primary error.

## Repository Verification

Verification after the V2 implementation:

- Full default suite: `268 passed, 2 skipped` in 32.21 s.
- Ruff lint and formatting checks: all 52 files passed.
- Strict mypy: all 29 source files passed.
- Schema verifier at capture: catalog plus all 11 then-pinned documents and offline
  references passed. The later 13-document catalog is verified separately by the ADR
  0004 evidence. ADR 0005's current 16-document catalog and final verification status are
  finalized in its separate evidence report.
- Real V2 first publication and shared-registry replay: passed.
- Explicit real CLI acceptance module: `3 passed` in 114.99 s.
- Two independently materialized views: all 13 file sizes and SHA-256 digests identical.
- Original execution specification: unchanged at the SHA-256 recorded above.

The skipped tests are environment/opt-in cases and do not convert this local evidence
into a promotion claim.

## Remaining Gates

The following remain open:

- Phase 0 security, privacy, data-governance, retention, audit, provider approval, and
  incident-control evidence.
- Phase 1A concrete producer identity/revision admission, typed payload and lineage
  projections, business eligibility and selection policy, invalidation/work propagation,
  and other applicable gates. The generic logical-node and immutable-revision/current-
  selection primitives that were open when this evidence was captured are implemented
  separately by ADR 0004, ADR 0005, and their local evidence reports.
- Phase 1B governed representative-corpus admission, selected VALID source report,
  durable source, `MCAPReadyManifest`, O-03 mapping, O-04 clock evidence, and admissible
  alignment.
- OD-SCALE-001, OD-SLO-001, OD-QUALITY-001, OD-MEDIA-001, and the other open decisions
  in the normalization overlay.
- Production registry/object-store selection, multi-process and distributed concurrency
  qualification, lifecycle projection, retention/deletion/legal hold, backup, restore,
  disaster recovery, observability, and capacity measurement.

SQLite and the local content-addressed blob layout are adapter choices. The exact V2
contracts, identities, and lineage are useful executable evidence, but this one local
exercise cannot establish production durability, quality, SLA, or 500-hours/day capacity.
