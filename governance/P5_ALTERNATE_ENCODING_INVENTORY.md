# P5 Alternate Encoding Inventory

This is an implementation navigation record for the opt-in JPEG experiment. It does
not change a published schema, Product QA meaning, or the canonical PNG default.

| Surface | Current boundary | P5 treatment |
| --- | --- | --- |
| Producer | `adapters/pyav_frame_materializer.py` normalizes RGB24 and has the canonical PNG encoder | Adds the pinned MJPEG encoder; PNG behavior remains the default. |
| Source policy | `application/canonical/mcap_source.py:McapMediaProcessingPolicy` | Pinned implementation/version, qscale, chroma, resize, color, metadata, MIME, and extension are part of the policy projection. |
| Direct materialization | `mcap_source.py:_materialized_frame_artifact` | Defaults to PNG, but accepts an explicit policy and binds its representation into artifact identity. |
| Layered cache | `mcap_source.py:_CachedMcapPngManifest` and cache key helpers | The legacy private name remains for compatibility; manifest/key projections bind the full representation policy and replay validates PNG or JPEG bytes before reuse. |
| Artifact/package identity | `mcap_source.py:_materialized_frame_artifact_from_rendered` and existing materializer contracts | Bytes, MIME, encoding, policy projection, dimensions, and SHA remain distinct; no published manifest shape was changed. |
| Provider preparation | `inference/preparation.py:InputPlanPreparer` | A rendered JPEG requires an explicit accepted media type. |
| Provider dispatch | `inference/orchestrator.py:InferenceOrchestrator` and `inference/runpod.py` | Dispatch rejects any rendered media type outside the adapter capability snapshot; RunPod retains the rendered item facts in its exact request form. |
| Product QA consumer | `qa_pipeline/supplemental.py:DeterministicSupplementalQaDenseConsumer` | Intentionally PNG-only. A JPEG supplemental artifact fails closed until a separate Product QA policy/schema decision exists. |
| Local canonical defaults | `application/canonical/local_composition.py` | Rendering policy, task policy, and local capability snapshot remain `image/png` only. |
| Error paths | `FrameMaterializationErrorCode`, MCAP policy/cache checks, provider capability check, supplemental PNG decoder | JPEG encoding failures are distinct; malformed or unsupported representations fail closed. |
| Fixtures and replay | `test_canonical_mcap_source.py`, `test_pyav_frame_materializer_durability.py`, `test_inference_input_plan.py`, `test_supplemental_temporal_package.py` | Cover deterministic MJPEG bytes, direct and cached JPEG materialization, warm cache replay without re-encoding, provider admission, and PNG-only QA rejection. |

## Qualification State

- Baseline remains PNG and experimental JPEG is created only through
  `McapMediaProcessingPolicy.jpeg_experiment(...)`.
- `benchmark/alternate_encoding_qualification.py` records separate selected-frame,
  per-class QA/event/boundary, speed, size, quality, and end-to-end evidence.
- The report defaults every external-label/provider state to `NOT_MEASURED`, binds
  measured evidence by digest, and cannot authorize production. A default-policy
  candidate requires representative evidence plus a signoff bound to the unsigned
  comparison digest.
- Representative labels, deployed-provider acceptance, deployed-provider replay,
  and hardware measurements remain external P15 inputs; no local fixture is reported
  as their substitute.
