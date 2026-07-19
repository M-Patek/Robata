# Local Full Mainline ? Deterministic Fake Model

## Scope

This report records a local development-only run of the complete MCAP-to-action vertical
slice. It validates real MCAP inspection, six-camera registered video export, strict frame
materialization, provider-neutral package construction, deterministic fake QA/proposal/action/
boundary inference, lineage validation, deterministic fusion, execution evidence, and atomic
publication. It does not validate a real model, provider policy, production capacity, or
promotion eligibility.

## Command

```powershell
uv run --locked python scripts/run_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-mainline-final2-20260719 `
  --allow-unapproved `
  --registry-root tmp/local-mainline-final-registry-20260719
```

Source file: `data/source/sample-medium.mcap` (130,303,923 bytes). The checked-in mapping
profile is observed/unapproved and was enabled explicitly with `--allow-unapproved`.

## Result

The command returned exit code `0` and produced the following machine result:

| Field | Value |
|---|---|
| `run_status` | `PRIMARY_COMPLETE` |
| `execution_mode` | `LOCAL_DEVELOPMENT_FAKE_MODEL` |
| `fake_inference_attempt_count` | `5` |
| `event_count` | `1` |
| `provider_requests` | `0` |
| `event_id` | `6acca923-ce1f-6a91-cdc6-23a50a45a55b` |
| `action_type` | `object_interaction` |
| event interval | `16213599687` to `24619866312` ns (half-open) |
| event status | `FINAL` |
| `production_eligible` | `false` |
| analysis duration | `48,991` ms |

The final output root was atomically published at
`tmp/local-mainline-final2-20260719`:

- `video/`: exactly 13 files ? six MP4s, six canonical timestamp sidecars, and one V2
  `camera-video-export-manifest.json`.
- `analysis/frames/`: 510 selected PNG frames across the two coarse/dense packages.
- `analysis/inferences/`: 10 files ? five canonical request/outcome pairs.
- `analysis/qa-aggregates.json`: both coarse and dense recording-level QA aggregates.
- `analysis/candidates.json`: one candidate event.
- `analysis/action-events.json`: one fused action event.
- `analysis/run-report.json` and `analysis/mainline-bundle.json`: complete accounting and
  connected lineage bundle.
- `execution-manifest.json` and `execution-audit.ndjson`: exact artifact inventory and
  canonical stage audit.

### Source and artifact lineage

| Artifact | Value |
|---|---|
| source recording identity | `d6ce6673fa1c6a35736cedb16b181b0b4a80a741d84961d416ba50e39e5ad7bc` |
| source content SHA-256 | `9fd5094bf29cd4ee50cd8c7d8c053e89d1c93660a0f4e57daaa726bae2b6156c` |
| video manifest artifact ID | `170ec194-a6af-d619-aead-d6294b284264` |
| video manifest exact SHA-256 | `0b6a8f68f38cb3257ef3251b5331ff09633cf6d1c1588c12014c7e44ed7ecf63` |
| video manifest semantic SHA-256 | `8dd872d9dab06cd62048b61443d81119a2f86db339e1df2c89d0e11e218e29ae` |
| execution manifest semantic SHA-256 | `d2107fac168eb4e0106df34fcb2e92a5e3c844e3d4bd394e41b0b9d55a754522` |
| analysis bundle SHA-256 | `46dfba3d1bd0bac54ee92170ac872ee8fff02c366dfe50faf0493212bc1498f7` |

## Verification

The published root passed the offline verifier:

```powershell
uv run --locked python scripts/verify_local_mainline.py `
  tmp/local-mainline-final2-20260719
```

The locked development gates also passed after the runtime/preflight hardening:

- full pytest suite: `472 passed, 3 skipped`;
- Ruff format and lint: passed;
- strict mypy: passed for 44 source files;
- offline schema registry: 16 pinned documents verified.

The real sample used no network or provider call. The fake adapter is deterministic and its
event is intentionally not production eligible. Alignment is unverified, the mapping is
unapproved, and no Phase 0, O-03/O-04, O-10, provider, real-model, quality, capacity, or
Phase 1B promotion gate is claimed.
