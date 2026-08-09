# Mage Single-Route Authority Profile

## Decision

The active local qualification route uses **Mage native codec/video as the only
perception authority**. The small-encoder seam is not executed by this profile and
cannot publish facts, suppress Mage inference, or replace a Mage artifact. It remains
an offline/shadow research boundary only; it is not part of the sustained runtime.

## Current execution profile

- one selected camera (`cam_01` by default);
- one resident Mage runtime;
- one worker and exactly one `model.generate` call in flight;
- bounded queue depth `2` by default: one active request plus one preparation slot;
- queue depth `1` remains the explicit serial control profile;
- native Mage codec/video is used for both normal observation and any future targeted
  refinement;
- raw media, segment manifests, inference identities, result artifacts, and downstream
  projections remain durable and replayable;
- no Qwen model or lightweight encoder is selected by the default route.

Queue depth `2` is **not model concurrency**. The runtime keeps generation serialized,
but permits segment `N+1` processor/codec preparation to overlap segment `N` generation.
Generated token IDs are moved to CPU before the generation lane is released, so CPU
decode can overlap the next GPU generation without retaining request-local GPU tensors.
No per-segment CUDA cache flush is performed.

## Compact decoder contract

`run_local_mage_stream.py --max-new-tokens` is identity-bound. The sustained profile
default is `256` tokens. Qualification fails if any output reaches that ceiling; this
prevents a shorter budget from being reported as a performance win when it truncates the
observation.

## Operational telemetry

The authoritative v2 response and result artifact bytes remain unchanged. Optional
non-wire sidecars record:

- endpoint generation event v3: request, processor, input materialization, generation,
  first token, decode, lock waits, TTFT, output tokens/s, result artifact lineage, model
  load time, and token-budget exhaustion;
- full-wall `nvidia-smi` samples: utilization, VRAM, power, and temperature;
- run timing: wall time, media duration, RTF, preparation/generation overlap, bounded
  in-flight count, and execution profile;
- comparison evidence: freshness, artifact replay isolation, single-generation proof,
  exact output-text hash parity, duty cycle, idle gaps, and qualification gates.

These sidecars are local diagnostic evidence. They are not published schemas, do not
participate in inference identity, and cannot make a run production-eligible.

## Cache conditioning

Native neural-codec preparation has a large cold-cache cost on the RTX 4060 test host.
Performance reports must therefore label whether codec assets were absent or already
populated. A cold serial arm must not be compared with a warm prefetch arm. The retained
August 8, 2026 qualification records both a warm steady-state pair and a cold-path pair;
see `mage-native-sustained-qualification.md`.

## Future scaling

The current route intentionally stays single-camera and single-generation. Moving to a
high-memory GPU may add native multi-camera encoding or provider-internal feature reuse,
but that is a separate qualification. It must not silently turn the disabled small
encoder into the authority or change the published evidence lineage.
