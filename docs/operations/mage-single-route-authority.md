# Mage Single-Route Authority Profile

## Decision

The current qualification profile uses **Mage native codec/video as the only
publication authority**. The lightweight-encoder seam is disabled by default. When an implementation is configured, it is an additive,
`SHADOW_ONLY` candidate and cannot publish facts, suppress Mage inference, or
replace a native Mage artifact until it passes a paired qualification.

## Current execution profile

- one selected camera (`cam_01` by default);
- one resident Mage runtime;
- one generation worker (`generation_concurrency=1`);
- one observation in flight by default;
- small-encoder execution disabled until an implementation is explicitly configured;
- optional second preparation slot only for a controlled prefetch experiment;
- native Mage raw-video remains the normal and refinement provider;
- raw media and exact segment lineage remain durable for replay and targeted refine.

The single-route restriction is deliberate. It makes the trust boundary and the
performance baseline unambiguous before multi-camera or high-concurrency work is
introduced.

## Small-encoder shadow seam

`robata.perception.single_route` defines a provider-neutral `SmallCameraEncoder`
protocol and bounded `SmallEncoderObservation`. These observations include camera
identity, interval, quality, confidence, candidate actions, feature lineage, and
source-frame references. They are explicitly `shadow_only` and are compared to
the native Mage observation by lineage, not promoted automatically.

A small encoder must not be treated as trusted merely because it is faster. Before
promotion, paired runs on the same segments must measure event recall, boundary
error, false silence, confidence calibration, output token reduction, latency,
VRAM, and replay lineage. A text-only summary is not a sufficient replacement for
visual evidence; a final implementation must retain structured semantics and a
compatible compact visual representation or use native Mage refinement.

## Decoder budget and telemetry

`run_local_mage_stream.py --max-new-tokens` exposes the decoder budget for an
identity-bound A/B test. The default remains 512 until a versioned compact prompt
contract is qualified.

The runtime now records a non-wire timing sidecar for local diagnostics:
processor-lock wait, processor preparation, generation-lock wait, input
materialization, `model.generate`, decode, and total request time. The published
endpoint response remains unchanged; endpoint logs carry the timing breakdown.

## Future migration

On a high-memory GPU, the same contracts can be reused with:

```text
camera_01 ... camera_06
        -> bounded encoder micro-batch
        -> SceneObservation
        -> one Mage reasoning request
```

That future expansion changes runtime composition and capacity settings, not the
authority boundary or published evidence lineage. The native Mage path remains a
reference and targeted-refine route until the small encoder demonstrates no
unacceptable semantic regression.
