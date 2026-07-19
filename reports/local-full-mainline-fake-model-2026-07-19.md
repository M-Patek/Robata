# Local Full Mainline — Deterministic Fake Model

## Scope

This report records a local development-only run of the complete MCAP-to-action vertical
slice. It validates real MCAP inspection, six-camera registered video export, strict frame
materialization, provider-neutral package construction, deterministic fake QA/proposal/action/
boundary inference, lineage validation, deterministic fusion, and atomic publication. It does
not validate a real model, provider policy, production capacity, or promotion eligibility.

## Command

```powershell
uv run --locked python scripts/run_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-full-mainline-real-20260719-a `
  --allow-unapproved `
  --registry-root tmp/local-mainline-real-registry-20260719-a
```

Source file: `data/source/sample-medium.mcap` (130,303,923 bytes). The checked-in mapping
profile is observed/unapproved and was enabled explicitly with `--allow-unapproved`.

## Result

The command returned exit code `0` and the following machine result:

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
| analysis duration | `55,416` ms |

The final output root was atomically published at
`tmp/local-full-mainline-real-20260719-a`:

- `video/`: exactly 13 files — six MP4s, six canonical timestamp sidecars, and one V2
  camera-video manifest.
- `analysis/frames/`: 510 selected PNG frames across the two coarse/dense packages.
- `analysis/inferences/`: 10 files — five canonical request/outcome pairs.
- `analysis/qa-aggregates.json`: both coarse and dense recording-level QA aggregates.
- `analysis/candidates.json`: one candidate event.
- `analysis/action-events.json`: one fused action event.
- `analysis/run-report.json` and `analysis/mainline-bundle.json`: complete accounting and
  connected lineage bundle.

### Source and artifact lineage

| Artifact | Value |
|---|---|
| source recording identity | `d6ce6673fa1c6a35736cedb16b181b0b4a80a741d84961d416ba50e39e5ad7bc` |
| source content SHA-256 | `9fd5094bf29cd4ee50cd8c7d8c053e89d1c93660a0f4e57daaa726bae2b6156c` |
| video manifest artifact ID | `170ec194-a6af-d619-aead-d6294b284264` |
| video manifest exact SHA-256 | `430bef5b3ba53de72450709d7639124f283bbb23d788ac92c7a40cff365af391` |
| video manifest semantic SHA-256 | `8dd872d9dab06cd62048b61443d81119a2f86db339e1df2c89d0e11e218e29ae` |
| analysis bundle SHA-256 | `b82bfb37d69365ca9d6c52b14dfacb4fd9f19f76a5ef8e0a618e74b6a0a0f011` |

## Verification

The implementation was checked with the locked environment:

- full pytest suite: `467 passed, 3 skipped in 39.50s`;
- Ruff lint: passed;
- Ruff format check: passed;
- mypy: passed for 40 source files.

The real sample used no network or provider call. The fake adapter is deterministic and its
event is intentionally not production eligible. Alignment is unverified, the mapping is
unapproved, and no Phase 0, O-03/O-04, O-10, provider, real-model, quality, capacity, or
Phase 1B promotion gate is claimed.
