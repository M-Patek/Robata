# Local model qualification summary - August 10, 2026

This document closes the **local development** comparison cycle for the Qwen and Mage
routes. It records what is runnable, what has local evidence, and what is still blocked
from production admission. It does not claim Linux/H100 capacity, representative-data
accuracy, multi-camera production quality, or canonical publication readiness.

## Status at a glance

| Route | Local execution | Structural result | Local throughput evidence | Admission state |
|---|---:|---|---:|---|
| Qwen serial control | Yes | Stable 51/51 exact control | 0.988x camera real time | Control baseline only |
| Qwen Hybrid Batch4 | Yes | 51/51 raw and normalized parity under the selected hedge | 3.728x camera real time | Best local throughput candidate; production HOLD |
| Mage neural codec stream | Yes | Native stream pipeline is runnable | 1.821x camera real time in retained local evidence | Long-term route; production HOLD |
| Mage traditional codec, 8 canvases | Yes | Strict output, but retained object-class hallucination | 1.248x hot end-to-end | HOLD |
| Mage traditional codec, 16 canvases | Yes | 256-token exhaustion and truncated non-JSON output | Not decision eligible | STOP |
| Mage fixed frames, 6 frames per 8-second segment | Yes | 5/5 strict compact observations | 1.958x camera real time | Diagnostic baseline only |

All numbers are single-camera, single-worker, local RTX 4060 Laptop evidence unless the
underlying report says otherwise. They must not be multiplied into an H100 production
claim without a sustained qualification run.

## Qwen serial and Hybrid Batch4

The retained Qwen native-batch qualification establishes the serial route as the exact
control:

- serial raw parity: **51/51**;
- serial normalized parity: **51/51**;
- serial quality gate: **pass**;
- serial execution wall: **247.995 s**;
- serial camera real-time factor: **0.988x**.

The selected Hybrid Batch4 hedge retains **51/51 raw and normalized parity** and reduces
the median execution wall to **65.588 s**, for **3.728x camera real time**. This remains
the strongest local short-term throughput candidate.

The common Mage/Qwen comparison is **unlabeled model agreement**, not ground-truth
accuracy. Earlier August 9 report bytes recorded structural parse success as capacity
decision eligibility. The current generator now separates those concepts and fails
closed:

- `semantic_quality_qualified = false`;
- `capacity_decision_eligible = false`;
- hold reason: `HOLD_UNLABELED_MODEL_AGREEMENT_ONLY_V1`.

Therefore no common-projection report may promote a route solely because strict JSON
parsing and deterministic downstream projection succeeded.

## Mage traditional codec

The traditional route is not blocked by normalization or by an overly strict evaluator.
The 8-canvas retained raw output itself claims a **green book** where retained frames show
green fabric. Higher spatial controls remove the book claim, supporting low-resolution
input degradation as a contributing factor, but garment taxonomy remains unstable.

The local decisions are therefore:

- `traditional_target_canvas_8`: **HOLD** with reason
  `UNSUPPORTED_OBJECT_CLASS_CLAIM`;
- `traditional_target_canvas_16`: **STOP** because generation exhausts the 256-token
  budget and produces truncated, non-strict JSON;
- DCVC remains a retained comparison control rather than a silently replaced route.

Authoritative retained report:
`docs/mage-traditional-codec-generation-qualification-2026-08-09.json`.

## Mage fixed-frame control

Robata now has an explicit Mage `video_backend="frames"` runtime path. It consumes an
ordered sequence of caller-verified images and does not invoke codec preparation,
recurrent stream state, cognition-gate admission, or hidden fallback to the codec path.
The fixed-frame identity binds:

- the exact six frame digests, timestamps, dimensions, and byte counts per segment;
- checkpoint-manifest digest and model revision;
- runtime load profile;
- exact prompt bytes and prompt version;
- generation token budget;
- the declared temporal-coordinate policy.

The default runner prompt is the exact native Mage binding prompt. The earlier
cross-model `common_qwen` prompt experiment is retained only as a negative diagnostic:
Mage echoed that foreign prompt and exhausted the output budget, so it is not a valid
fixed-frame quality result.

### Real local result

Tracked report:
`docs/mage-fixed-frame-native-prompt-qualification-2026-08-10.json`

Exact report SHA-256:
`0a7433b59541a4fb01db5934826a646ec0f3bcbe737c2f575203254182c96320`

Configuration and result:

- model: local Mage-VL, NF4 4-bit profile;
- camera: `cam_01` only;
- plan: five non-overlapping 8-second contexts, six exact frames per context;
- prompt: exact native Mage v6 binding prompt;
- effective generation budget: 256 tokens per context;
- model load: **11.747 s**;
- recurring five-context wall: **20.428 s**;
- generation sum: **20.189 s**;
- cold total: **32.175 s**;
- strict compact projection: **5/5**;
- recurring camera real-time factor: **1.958x**;
- local-equivalent lanes for a 25x camera-time target: **13**.

The first generation took **8.181 s** and the four warm generations took
**2.796-3.138 s**. The run produced six action observations. Compared with the frozen
native-Mage observations, unlabeled agreement was:

- mean label-token F1: **0.4883**;
- mean temporal IoU: **0.7103**.

In that comparison block, legacy `mage_*` fields mean the fixed-frame candidate and
legacy `qwen_*` fields mean the frozen native-Mage reference; the report records this
mapping explicitly. Agreement values are diagnostic only. The fixed-frame report keeps
`quality_qualified=false`, `decision_eligible=false`, and `production_eligible=false`.

## What is locally complete

- Qwen serial remains a stable exact control.
- Qwen Hybrid Batch4 is implemented and locally qualified as the current throughput
  hedge.
- Mage neural, traditional, and fixed-frame frontends all have explicit runnable local
  paths.
- Mage fixed-frame execution is separated from native codec execution rather than using
  an implicit fallback.
- Exact prompt/frame/runtime identities and fail-closed qualification states are recorded.
- The Qwen common comparison no longer treats unlabeled agreement as semantic admission.
- Traditional-codec hallucination evidence is classified at the raw-model-output surface.

## What is not production-ready

The following evidence is still required on the production machines:

1. representative labeled quality for every route selected for canary or production;
2. multi-camera input/fusion qualification rather than the current `cam_01` control;
3. Linux/container execution using the exact production image and model manifests;
4. sustained two-H100 capacity, VRAM, thermal, queueing, and backpressure measurements;
5. end-to-end canonical storage, completion, outbox, replay, and publication evidence;
6. canary/shadow comparison under the same real workload and failure policy.

Accordingly, the local development phase is closed as **runnable and locally evidenced**,
not as **production admitted**.
