# Mage traditional H.264/HEVC preprocessing toolchain

This directory pins the Linux `cv-preinfer` preparation boundary used by Mage's
traditional codec backend. It is an alternative to the neural DCVC-RT preparation
backend, not a cold-start-only phase. Both routes produce the canvas images,
`src_patch_position.npy`, and `meta.json` consumed by Mage's `_load_codec_result`.

## Pinned build inputs

- `python:3.12-slim` by immutable digest (Python 3.12.13, Debian 13 base)
- `mwader/static-ffmpeg:7.1.1` by immutable digest, copied as `/usr/local/bin/ffmpeg`
  and `/usr/local/bin/ffprobe`
- `codec-video-prep==0.2.5`, NumPy, OpenCV headless, and Pillow by exact wheel hashes
  in `requirements.lock`

Wheels are fetched as external build inputs and verified with `pip --require-hashes`.
No wheel is stored in the repository.

## Build

```powershell
docker build --pull=false --platform linux/amd64 `
  --tag robata/mage-traditional-codec:0.2.5-static `
  --file docker/mage-traditional-codec/Dockerfile `
  docker/mage-traditional-codec
```

## Benchmark job manifest

The container accepts one or more jobs and runs them serially in one resident Python
process. Source paths must remain under `/input`; outputs, manifest, and receipt must
remain under `/output`. The source mount should be read-only.

```json
{
  "schema_name": "robata-mage-traditional-codec-jobs",
  "schema_version": 1,
  "run_id": "traditional-five-segment-example",
  "policy": {
    "engine": "hevc",
    "target_canvas": 8,
    "group_size": 8,
    "images_per_group": 1,
    "patch": 16,
    "max_pixels": 65536,
    "min_group_frames": 8,
    "max_group_frames": 128,
    "canvas_format": "jpg",
    "readiness_sum_threshold": 0,
    "avoid_keyframes": true
  },
  "jobs": [
    {
      "job_id": "segment-000",
      "video": "/input/segment-000.mp4",
      "out_dir": "/output/assets/segment-000",
      "source_content_sha256": "<64 lowercase hexadecimal characters>"
    }
  ]
}
```

Example execution (network is deliberately disabled at runtime):

```powershell
docker run --rm --network none --platform linux/amd64 `
  --mount "type=bind,source=<segment-directory>,target=/input,readonly" `
  --mount "type=bind,source=<output-directory>,target=/output" `
  robata/mage-traditional-codec:0.2.5-static `
  --job-manifest /output/jobs.json `
  --receipt /output/container-receipt.json
```

## Windows/developer-host bridge

The pinned image can be invoked from a host where `codec-video-prep` is not installed
locally. The bridge is an explicit adapter, not a frame or neural-codec fallback:

```powershell
$env:CV_PREINFER_BIN = (Resolve-Path scripts\mage_cv_preinfer_host.cmd)
$env:MAGE_CV_PREINFER_IMAGE = "registry.example/robata/mage-traditional-codec@sha256:<64-lowercase-hex>"
$env:MAGE_CV_PREINFER_BACKEND = "docker"
```

The adapter requires a locally available, digest-pinned image and runs with
`--pull=never`, `--network none`, and `--platform linux/amd64`. It validates
`meta.json`, `src_patch_position.npy`, and canvas output before returning to Mage.
If Docker, the image, or the selected binary is unavailable, it exits non-zero and
leaves no model-loader output; it never silently switches to frames or another
backend. On Linux/macOS use `scripts/mage_cv_preinfer_host.sh`.

## Evidence boundary

The receipt records exact source/config/toolchain identities, the invoked command,
subprocess and workload walls, output hashes, canvas dimensions, position-array shape,
and an asset-contract validation equivalent to Mage's loader. It does **not** load the
Mage model or establish semantic quality.

`cv-preinfer` embeds output paths and runtime timing fields in `meta.json`; therefore
raw `meta.json` bytes vary between runs. The runner separately hashes the normalized
loader metadata and the actual inference payload (canvases plus
`src_patch_position.npy`). Repeated qualification must compare those fields rather than
claiming byte-identical raw metadata.