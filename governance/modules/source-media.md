# Source Media

## Scope and path anchors
- Ingestion and alignment: `src/robata/ingestion/**`, `src/robata/alignment/**`
- Decode and source adapters: `src/robata/adapters/mcap_*.py`, `src/robata/adapters/pyav_*.py`, `src/robata/adapters/parallel_*.py`
- Media services: `src/robata/application/{artifact_view,video_export,registered_video_export}.py`
- Canonical media helpers: `application/canonical/{bounded_media,media_quality,single_pass_video,source_fixture}.py`

## How to dispatch
`source-media / P<n> - <ingestion, decode, materialization, or throughput task>`

## Construction phases
1. **Source intake** - MCAP discovery, timestamps, camera alignment, and capture records.
2. **Bounded decode** - PyAV access, selected-frame materialization, and video export.
3. **Media efficiency** - parallel decode, artifact reuse, encoding choices, and timing evidence.
4. **Quality observations** - expose source-quality facts for QA without mutating provenance.

## Relevant tests
- Fast: `python -m pytest tests/unit/test_mcap_single_pass.py tests/unit/test_pyav_interval_spool.py tests/unit/test_bounded_media.py tests/unit/test_canonical_media_quality.py`
- Broader: `python -m pytest tests/integration/test_real_mcap_single_pass.py tests/integration/test_canonical_mcap_source.py`

## Read alongside
Read `sampling-qa` for context/ROI and quality-observation consumers. Read `canonical-integration` before changing `application/canonical/mcap_source.py` or the source-to-run bridge.
