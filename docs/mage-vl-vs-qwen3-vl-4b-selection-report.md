# Mage-VL 4B vs. Qwen3-VL 4B Selection Report

**Research cut:** 2026-07-30 03:16 CDT (UTC-05:00)

**Scope:** published primary-source evidence and source-code/configuration inspection.

**Not in scope:** downloading weights, running either model, or claiming Robata production capacity.

> **中文摘要：** 目前尚不能宣称任一模型已经赢得 Robata 的生产选型。应先以 `Qwen/Qwen3-VL-4B-Instruct` 完成短窗口、六摄像头帧级推理的 RunPod 资格验证；它与现有请求规划和标准服务运行时更直接兼容。`microsoft/Mage-VL` 是值得重点 A/B 测试的视频时序候选，但其论文优势依赖编解码器原生输入、自定义运行时和单视频示例，尚未在 Robata 的同步多摄像头、证据链和故障处理条件下复现。文中所有数值均标记为作者发布值、由发布值推导，或尚未实测，避免把论文成绩当作生产容量承诺。

## Executive Decision

There is not yet a production winner for Robata. The evidence supports a two-track
decision rather than an immediate model replacement:

1. **Qualify `Qwen/Qwen3-VL-4B-Instruct` first** as the conservative short-window
   inference candidate. It has an official vLLM path, a released FP8 variant, clear
   model identifiers, and its image/video request model maps more directly to
   Robata's current frame-oriented input planner.
2. **Qualify `microsoft/Mage-VL` as the video/temporal challenger**, not as a
   drop-in endpoint. In the Mage authors' direct 4B comparison it wins 14 of 15
   video/temporal/tracking rows and has an event-gating design relevant to streaming.
   Its advertised advantage, however, depends on codec-native video input, a custom
   runtime path, and a single-video streaming example. None has been demonstrated
   against Robata's six synchronized cameras.
3. **Do not select Mage solely from its paper.** Its direct comparison is
   author-reported, very recent, and not a Robata reproduction. The codec-input and
   multi-camera adaptation would change the current provider-facing input
   representation and must be versioned rather than hidden in an endpoint handler.

The operational default for a first representative RunPod qualification is therefore
**Qwen3-VL-4B-Instruct**. Mage becomes the primary candidate only if it passes the
acceptance gates in [Robata Qualification Plan](#robata-qualification-plan) with a
meaningful quality or compute advantage on the same six-camera corpus.

## Evidence Status and Method

This report distinguishes three kinds of number:

| Label | Meaning |
| --- | --- |
| `PUBLISHED` | A value transcribed from an author-published paper, model card, or official repository. It is not independently reproduced here. |
| `DERIVED` | Arithmetic performed from explicitly listed `PUBLISHED` values. It is reproducible from the tables below but is not a new benchmark. |
| `NOT MEASURED` | A value that needs Robata's own model endpoint, representative recordings, and qualification harness. |

The source collection was pinned to the following revisions where a revision is
available:

| Source | Pinned reference inspected | Used for |
| --- | --- | --- |
| [Mage source repository](https://github.com/microsoft/Mage/tree/8c94a0ac905167f40b05b09332b78752b7f9fbef) | `8c94a0ac905167f40b05b09332b78752b7f9fbef` | Mage implementation, model identity, direct comparison tables, dependencies, streaming example |
| [Mage-VL paper](https://arxiv.org/abs/2607.24904) | arXiv `2607.24904` | cited technical report for Mage-VL |
| [Qwen3-VL source repository](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da) | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` | Qwen release, runtime, evaluation and deployment guidance |
| [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631) | arXiv `2511.21631` | cited model architecture and benchmarks |
| [Qwen 4B Instruct config](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/ebb281ec70b05090aa6165b016eac8ec08e71b17/config.json) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | exact Instruct architecture and context limit |
| [Qwen 4B Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking/tree/1de27d8c51f12e819435303b9e84c4e25ba8401e) | `1de27d8c51f12e819435303b9e84c4e25ba8401e` | exact Thinking checkpoint identity |

No benchmark score has been sourced from a blog summary, a leaderboard repost, or
an unsourced model comparison. The direct Mage-versus-Qwen numbers below are all
from the Mage authors' official table. Qwen's own values are retained separately
for selecting **Instruct versus Thinking**; they are not merged into the direct
comparison.

## Exact Models Under Review

The phrase "Qwen3-VL 4B" is ambiguous. There is no canonical bare
`Qwen/Qwen3-VL-4B` release. A deployment manifest must pin one of these variants
and a revision:

| Role | Model ID | Revision | Published format | License signal |
| --- | --- | --- | --- | --- |
| Qwen fast/default candidate | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | BF16 | Apache-2.0 |
| Qwen reasoning candidate | `Qwen/Qwen3-VL-4B-Thinking` | `1de27d8c51f12e819435303b9e84c4e25ba8401e` | BF16 | Apache-2.0 |
| Qwen Instruct FP8 candidate | `Qwen/Qwen3-VL-4B-Instruct-FP8` | `fefbb44cbcce8d1bb7e20b920b94f77432b3446d` | mixed FP8/BF16 | Apache-2.0 |
| Qwen Thinking FP8 candidate | `Qwen/Qwen3-VL-4B-Thinking-FP8` | `219b8e195ea30e383c55c954278767990974bba9` | mixed FP8/BF16 | Apache-2.0 |
| Mage challenger | `microsoft/Mage-VL` | record revision at download time | BF16 inference checkpoint | see license caveat below |

Qwen released both 4B dense variants on 2025-10-15. Mage's repository says its
single unified checkpoint was released on 2026-07-26. Mage is consequently a much
newer dependency with materially less time for independent integration evidence.

### Architecture and Capability Comparison

| Dimension | Mage-VL 4B | Qwen3-VL 4B | Robata consequence |
| --- | --- | --- | --- |
| Core design | From-scratch Mage-ViT codec-native vision stack, two-layer projector, and Qwen3-4B-Instruct-2507 causal decoder | Dense VLM with Interleaved-MRoPE, DeepStack, and text-timestamp alignment | Mage's claimed gain is tied to video representation; Qwen maps more naturally to ordinary image/video messages. |
| Published scale | Fixed 4B-parameter family budget | 4,437,815,808 parameters in the 4B config | Do not infer equal memory or throughput from the family label alone. |
| Visual interface | Traditional H.264/H.265 motion/residual data or DCVC-RT rate maps; frame mode exists | Images, multi-image, and video request paths | Current Robata planning is source-frame-to-provider-item, so neither is an unqualified full-video integration today; Mage needs more new representation work. |
| Streaming behavior | Built-in cognition gate can suppress routine segments before full generation | No equivalent published event-gate in the reviewed 4B materials | Mage's gate is promising for high-volume streaming, but must be calibrated against false silence and missed events. |
| Native context | Not stated as a directly comparable single number in the inspected Mage materials | 262,144 tokens by default; 1M only through YaRN configuration change | A large context limit is not a small-GPU deployment recommendation. |
| Official serving path | Offline Transformers; online instructions use an SGLang `feat/mage-vl` branch | Transformers >= 4.57.0, `qwen-vl-utils==0.0.14`, vLLM >= 0.11.0, with SGLang also documented | Qwen has the lower-risk first RunPod handler. Mage requires a custom image/runtime validation. |

Qwen's 4B configuration has a 36-layer text decoder (hidden 2560, 32 query heads,
8 KV heads, head dimension 128, intermediate size 9728) and a 24-layer vision
encoder (hidden 1024, patch 16, temporal patch 2, spatial merge 2, DeepStack
indices `[5, 11, 17]`). These are configuration facts, not Robata runtime results.

### License and Release Gate

Qwen's repository and the inspected 4B model cards consistently state Apache-2.0.
Mage's metadata is not internally consistent in the inspected source tree:

| Mage artifact | License/release statement |
| --- | --- |
| Root and Mage-VL README | Mage-VL is stated to be Apache-2.0. |
| Root `LICENSE` and `mage_vl/pyproject.toml` | MIT. |
| Mage root README, Responsible AI section | States the models are released for research purposes and are not intended for product or service deployment. |

MIT and Apache-2.0 are both permissive, but the conflicting metadata and
research-use statement mean **Mage must be marked license/release-unresolved** in a
production deployment record. Before downloading weights into a deployment image,
record the exact Hugging Face revision, the model-card license, the weight files'
license notice, and counsel's decision. This is a release-control issue, not a
benchmark-quality issue.

## Published Direct Comparison: Mage Authors' Matched 4B Table

The following values are transcribed from the Mage-VL README's direct comparison.
That source states that Mage-VL-4B and Qwen3-VL-4B use the same 4B Qwen3 LLM
backbone. `Delta` is `Mage - Qwen` in the source benchmark's native units. It is
not a universal percentage-point measure.

### Static Image and Spatial Tasks

| Domain | Benchmark | Mage-VL-4B | Qwen3-VL-4B | Delta |
| --- | --- | ---: | ---: | ---: |
| Document | DocVQA-val | 95.14 | 94.69 | +0.45 |
| Document | InfoVQA-val | 80.33 | 79.50 | +0.83 |
| Document | AI2D w/ Mask | 83.16 | 81.54 | +1.62 |
| Document | ChartQA | 84.88 | 83.96 | +0.92 |
| Document | OCRBench | 81.80 | 81.60 | +0.20 |
| Document | MultiDocVQA-val | 87.46 | 87.21 | +0.25 |
| Document | ChartQAPro | 32.57 | 26.79 | +5.78 |
| Document | TextVQA-val | 77.28 | 80.55 | -3.27 |
| Document | CC-OCR Doc | 32.25 | 39.69 | -7.44 |
| General VQA | MMBench-EN-dev | 84.02 | 83.25 | +0.77 |
| General VQA | MMBench-CN-dev | 82.04 | 80.58 | +1.46 |
| General VQA | MMStar | 67.32 | 62.04 | +5.28 |
| General VQA | MME-Perception | 1709.54 | 1703.50 | +6.04 |
| General VQA | SeedBench (All) | 76.78 | 75.65 | +1.13 |
| General VQA | CV-Bench | 87.79 | 85.37 | +2.42 |
| General VQA | MME-RealWorld | 66.52 | 63.20 | +3.32 |
| Spatial | CV-Bench-2D | 82.13 | 81.00 | +1.13 |
| Spatial | CV-Bench-3D | 94.75 | 92.30 | +2.45 |
| Spatial | BLINK | 65.11 | 65.10 | +0.01 |
| Spatial | EmbSpatial | 82.67 | 77.50 | +5.17 |
| Spatial | CrossPoint | 80.00 | 26.90 | +53.10 |
| Spatial | CRPE-Relation | 76.12 | 77.70 | -1.58 |
| Spatial | SAT | 67.33 | 69.30 | -1.97 |

### Video, Temporal Grounding, and Tracking

| Domain | Benchmark | Mage-VL-4B | Qwen3-VL-4B | Delta |
| --- | --- | ---: | ---: | ---: |
| Video QA | MV-Bench | 65.10 | 66.70 | -1.60 |
| Video QA | NextQA | 83.10 | 79.80 | +3.30 |
| Video QA | VideoMME | 64.00 | 59.70 | +4.30 |
| Video QA | LongVideoBench | 61.30 | 57.70 | +3.60 |
| Video QA | LVBench | 41.80 | 39.20 | +2.60 |
| Video QA | MLVU-dev | 68.70 | 61.50 | +7.20 |
| Video QA | VideoEval-Pro | 45.20 | 20.70 | +24.50 |
| Temporal grounding | Timelens-Charades | 50.70 | 43.10 | +7.60 |
| Temporal grounding | Timelens-ActivityNet | 45.40 | 28.40 | +17.00 |
| Temporal grounding | Timelens-QVHighlight | 57.40 | 34.90 | +22.50 |
| Spatial video | VSI-Bench | 64.30 | 53.30 | +11.00 |
| Tracking, J&F | Ref-DAVIS17 | 25.83 | 7.48 | +18.35 |
| Tracking, J&F | MeViS-ValidU | 22.55 | 3.16 | +19.39 |
| Tracking, J&F | ReasonVOS | 17.76 | 9.66 | +8.10 |
| Tracking, J&F | Ref-YT-VOS | 25.57 | 5.28 | +20.29 |

### Derived Directional Summary

The following is arithmetic from the preceding raw table, not an author metric.
Mixed benchmark scales make it invalid to treat these means as a scientific
meta-score. They are useful only as an auditable direction check.

| Slice | Rows | Mage wins | Qwen wins | Derived result |
| --- | ---: | ---: | ---: | --- |
| Static table, all rows | 23 | 19 | 4 | `DERIVED`: Mage leads on 19 of 23 rows. |
| Static percentage-like rows, excluding MME-Perception's 0-2000 scale | 22 | 18 | 4 | `DERIVED`: simple mean 75.79 vs. 72.52, delta +3.27. |
| Document rows | 9 | 7 | 2 | `DERIVED`: simple mean delta -0.07; broadly tied on this heterogeneous slice. |
| General VQA rows excluding MME-Perception | 6 | 6 | 0 | `DERIVED`: simple mean delta +2.40. |
| Spatial rows | 7 | 5 | 2 | `DERIVED`: simple mean delta +8.33, dominated by CrossPoint's +53.10. |
| Video/temporal/tracking table | 15 | 14 | 1 | `DERIVED`: mean delta +11.21; median delta +8.10. |
| Video QA subset | 7 | 6 | 1 | `DERIVED`: simple mean 61.31 vs. 55.04, delta +6.27. |
| Temporal-grounding subset | 3 | 3 | 0 | `DERIVED`: simple mean 51.17 vs. 35.47, delta +15.70. |
| Tracking subset | 4 | 4 | 0 | `DERIVED`: simple mean 22.93 vs. 6.40, delta +16.53. |

The strongest conclusion justified by these rows is narrow: **the Mage authors'
controlled comparison reports a pronounced video/temporal advantage at this model
scale.** It is not evidence that Mage will improve Robata's clip-boundary accuracy,
six-view fusion, schema validity, or cost without a matched Robata experiment.

## Streaming Results: Relevant but Not a Fair Pairwise Test

Mage also reports a proactive-streaming evaluation, but it must not be folded into
the direct head-to-head totals:

| Dataset/protocol | Mage-VL-4B | Qwen3-VL-4B | Why it is not a strict comparison |
| --- | ---: | ---: | --- |
| SoccerNet response timing: TriggerAcc | 79.21 | not reported | Qwen has no published row in this table. |
| SoccerNet response timing: TimVal / F1 / ROC-AUC / PR-AUC | 55.54 / 16.35 / 83.14 / 9.30 | not reported | Mage's gate is evaluated as a dedicated streaming mechanism. |
| OVO-Bench: RT-Avg | 79.84 | 72.80 | Mage uses 1 fps; Qwen row uses 64 frames. |
| OVO-Bench: BT-Avg | 48.15 | 53.10 | Input budget and protocol differ. |
| OVO-Bench: Overall | 64.00 | 63.00 | The +1.00 is interesting but not a fair efficiency claim. |

Mage claims more than 75% visual-token reduction and up to 3.5x wall-clock
inference speedup over uniform frame sampling under its reported codec-native
conditions. It is a `PUBLISHED` claim, not a multiplier that can be applied to
Robata's current capacity target.

## Qwen 4B Variant Data: Instruct vs. Thinking

The following values are transcribed from Qwen's official 4B model-card figures.
They choose a Qwen route; they do not overwrite the direct comparison above.
`Thinking - Instruct` uses each benchmark's native scale.

| Robata-relevant benchmark | Instruct | Thinking | Difference |
| --- | ---: | ---: | ---: |
| MMMU Val | 67.4 | 70.8 | +3.4 |
| MathVision | 51.6 | 60.0 | +8.4 |
| HallusionBench | 57.6 | 64.1 | +6.5 |
| ERQA | 41.3 | 47.3 | +6.0 |
| VSI-Bench | 58.4 | 55.2 | -3.2 |
| EmbSpatialBench | 79.6 | 80.7 | +1.1 |
| RefSpatialBench | 46.6 | 45.3 | -1.3 |
| RoboSpatialHome | 61.7 | 63.2 | +1.5 |
| MVBench | 68.9 | 69.3 | +0.4 |
| VideoMME, no subtitles | 69.3 | 68.9 | -0.4 |
| MLVU MCQ | 75.3 | 75.7 | +0.4 |
| LVBench | 56.2 | 53.5 | -2.7 |
| CharadesSTA | 55.5 | 59.0 | +3.5 |
| VideoMMMU | 56.2 | 69.4 | +13.2 |
| ScreenSpot | 94.0 | 92.9 | -1.1 |
| ScreenSpot Pro | 59.5 | 49.2 | -10.3 |
| OSWorldG | 58.2 | 53.9 | -4.3 |
| AndroidWorld | 45.3 | 52.0 | +6.7 |
| OSWorld | 26.2 | 31.4 | +5.2 |
| DocVQA Test | 95.3 | 94.2 | -1.1 |
| OCRBench | 881 | 808 | -73 |

Thinking improves several reasoning and long-horizon rows, but not every spatial,
screen-grounding, long-video, or document row. It should therefore be a
**low-frequency adjudication/shadow route** for conflicts or hard cases, not an
automatic replacement for Instruct in a high-throughput QA stage.

### Qwen Memory Numbers

Qwen publishes no 4B tokens/s, first-token latency, minimum VRAM, concurrent-request,
or RunPod-GPU measurement. The following are therefore capacity accounting aids,
not vendor performance claims:

| Item | Value | Status |
| --- | ---: | --- |
| BF16 checkpoint safetensor bytes | 8,875,719,344 bytes = 8.266 GiB | `PUBLISHED` artifact size |
| BF16 KV cache per token per sequence | 147,456 bytes = 144 KiB | `DERIVED`: `36 layers x 8 KV heads x 128 head dim x 2 (K/V) x 2 bytes` |
| KV cache at 2K / 8K / 16K / 32K | 0.281 / 1.125 / 2.250 / 4.500 GiB | `DERIVED`, one sequence, BF16 |
| KV cache at native 256K context | 36.000 GiB | `DERIVED`, one sequence, BF16 |
| KV cache at 1M YaRN extension | 144.000 GiB | `DERIVED`, one sequence, BF16 |
| 4B Instruct FP8 checkpoint disk footprint | 5.608 GiB | `PUBLISHED` artifact size; it remains mixed FP8/BF16 |

Actual device memory also includes the vision stack, activations, CUDA allocator,
runtime workspace, prompt/image/video tokens, and batching headroom. The table must
not be used as a GPU sizing guarantee.

## Why Published Tables Cannot Be Combined Blindly

Several similarly named benchmarks disagree across official documents. This does
not automatically imply an error; it means the configuration is not comparable
without a fully pinned protocol:

| Example | Mage direct table's Qwen row | Qwen 4B official figure | Likely non-equivalence to preserve |
| --- | ---: | ---: | --- |
| MVBench | 66.70 | 68.90 Instruct / 69.30 Thinking | model variant, decoding, frame/context budget, and evaluation harness are not proven identical |
| VideoMME | 59.70 | 69.30 Instruct / 68.90 Thinking, no subtitles | stated subtitle setting and protocol differ |
| MLVU | 61.50 | 75.30 Instruct / 75.70 Thinking, MCQ | reported subset/task format differs |
| LVBench | 39.20 | 56.20 Instruct / 53.50 Thinking | variant and protocol are not shown as identical |
| VSI-Bench | 53.30 | 58.40 Instruct / 55.20 Thinking | model/evaluation configuration is not pinned across reports |

Qwen's official evaluation notes also say that some benchmark prompts were modified
and some benchmarks are internally constructed with assets to be released later.
Its family README and individual model-card figures use different generation
settings in places. The minimum reproducibility record must therefore pin:

- model identifier and immutable revision;
- processor revision and video/frame preprocessing policy;
- prompt artifact and JSON schema hash;
- sampling parameters, output-token cap, seed, and runtime version;
- exact benchmark split and scoring harness revision;
- GPU type, driver/CUDA, batch shape, concurrency, and warm-up policy.

## Serving and Dependency Comparison

| Area | Mage-VL | Qwen3-VL 4B | Deployment implication |
| --- | --- | --- | --- |
| Standard dependencies | `torch>=2.9`, `transformers>=5.7`, `accelerate`, `codec-video-prep>=0.2.5`, `mamba-ssm>=2.2`, `flash-attn>=2.7` | `transformers>=4.57.0`, `qwen-vl-utils==0.0.14`, `vllm>=0.11.0`; FlashAttention-2 recommended for BF16/FP16 | Use separate, locked images. Do not assume a Qwen vLLM image can serve Mage correctly. |
| Offline input | image, frame-sampled video, traditional H.264/HEVC codec video, or DCVC-RT codec video | image and video messages | Mage can fall back to frames, but its claimed efficiency path needs codec information. |
| Online path | Mage instructions build a custom SGLang branch, then use an OpenAI-compatible server | vLLM and SGLang are documented | Neither is directly Robata's current RunPod wire contract. Each needs a handler translating the immutable Robata request into the model runtime request and returning the normalized response envelope. |
| Streaming sample | Non-overlapping segments, default 8.0 seconds, default codec backend, gate threshold 0.5 | no published equivalent gate | Mage's default segmentation does not match Robata's 2-second windows with 1-second hop. |
| Multi-camera evidence | Sample program accepts one video path | official examples cover multi-image and video, but no six-synchronized-camera qualification is published | Validate multi-view packaging and camera order rather than assuming either model fuses six feeds correctly. |

## Robata Compatibility Assessment

The repository's inference boundary already provides a disciplined integration
surface: `VisionModelAdapter`, immutable `ModelCapabilities`, a model/version-pinned
`VisionInferenceRequest`, an exact `InferenceInputPlan`, idempotency, and a RunPod
adapter that requires a provider-specific response contract. It does **not** mean
either model has already been integrated or qualified.

| Existing boundary | Qwen fit | Mage fit | Required action |
| --- | --- | --- | --- |
| `VisionModelAdapter` / model capability snapshot | Good conceptual fit | Good conceptual fit | Pin the exact model and handler revision; truthfully advertise only verified tasks, input modes, limits, JSON behavior, and concurrency. |
| RunPod request/response wrapper | Custom handler required | Custom handler required | The current adapter speaks `robata-runpod-vision-request-v1`, not generic vLLM/OpenAI JSON. Preserve the wrapper and translate inside the endpoint. |
| Current frame-first input planner | Direct first qualification route | Does not exercise codec-native value proposition | Qwen: render ordered six-camera frame items. Mage: either accept a temporary frame-mode baseline or introduce a versioned codec-segment input representation. |
| Six-camera canonical order and provenance | Must be tested | Must be tested | Preserve camera ID, frame timestamp, transform, SHA-256, and ordinal in every provider item. Never reduce six views to an unlabeled collage. |
| 2-second window / 1-second hop | Natural high-frequency workload | Mismatched with published 8-second non-overlapping sample | Mage requires a timing-policy experiment and new evidence about overlap, gate state, and event boundaries. |
| Structured, schema-valid output | Must be proven at endpoint | Must be proven at endpoint | Do not set `supports_json_schema=true` from a model marketing claim. Validate the exact response schema, retry path, and raw-output evidence. |

### Codec-Native Mage Is a Contract Decision

Mage's reported speed/token benefit relies on codec motion vectors and residual
energy. Robata's current provider input plan explicitly represents rendered source
frames and records a one-frame-to-one-provider-item provenance path. Shipping raw
codec segments, their decode policy, frame/time mapping, and codec-derived features
would alter the provider-facing semantic representation.

That work must be treated as a schema/version/migration decision under this
repository's contract rules. It is not an implementation detail to hide in a
RunPod handler. A frame-mode Mage test is still useful, but it cannot validate the
claimed codec-native deployment advantage.

### Workload Accounting for the Qualification Harness

Robata's documented target workload is six cameras, 2-second windows, and a
1-second hop. That implies approximately:

```text
3600 temporal windows / recording-hour
6 camera-window units / temporal window
= 21,600 camera-window units / recording-hour

At 500 recording-hours/day:
500 x 21,600 = 10,800,000 camera-window units/day
```

This is a workload-shape calculation, not an inference-call count. Calls may batch
multiple cameras or split a window by provider limits. If each camera-window uses
`F` rendered frames, the upper-level input count is `10,800,000 x F` frames/day.
The endpoint qualification must measure the actual call plan rather than assume
that model context capacity or a paper speedup closes this budget.

## Robata Qualification Plan

The following gates decide whether either candidate can become a production
selection. All should run against representative, permissioned six-camera data and
persist the resulting evidence through Robata's existing qualification path.

| Gate | Required measurement | Pass condition to set before test | Why it matters |
| --- | --- | --- | --- |
| Identity and supply chain | model revision, processor, handler image digest, runtime/CUDA, model-card license snapshot | all immutable identifiers recorded; Mage license/release ambiguity resolved | avoids a result that cannot be replayed or legally released |
| Input fidelity | six camera IDs/order, timestamps, frame/segment SHA-256, transform record | zero unexplained reorder, duplication, or unlabeled view loss | multi-view correctness is a product requirement, not a visual benchmark |
| Quality | QA issue F1, false-positive rate, clip-boundary IoU or boundary MAE, action/event recall, evidence citation accuracy | pre-register thresholds and a human-labeled holdout before seeing results | published VQA does not measure Robata's outcomes |
| Contract behavior | JSON/schema validity, invalid-output rate, idempotent retry, raw-output persistence | 100% schema-valid accepted output on test corpus; all failures traceable | model fluency cannot substitute for evidence-chain integrity |
| Streaming safety | missed critical event rate, false-silence rate, gate calibration, late-event rate | zero tolerance or explicit threshold for critical misses | especially necessary before adopting Mage's suppression gate |
| Performance | P50/P95 end-to-end latency, GPU seconds per recording-hour, input/output tokens, peak VRAM, queue backlog | target thresholds tied to the production workload, not to paper claims | determines actual cost and T+1/T+3 feasibility |
| Failure handling | timeout, 429/5xx, malformed payload, partial camera input, pod restart | no duplicate publication; terminal state and evidence remain correct | validates the actual RunPod adapter boundary |
| Shadow comparison | matched prompts and inputs for both candidates; blinded human adjudication of disagreements | pre-agreed winner rule and confidence interval | prevents a model choice based on an incomparable vendor chart |

Recommended rollout order:

1. Build a Qwen Instruct handler that honors the existing RunPod wrapper and
   qualifies frame-mode short-window QA first.
2. Run Qwen Thinking only as a capped shadow/adjudication route, measuring whether
   its extra reasoning improves disputed or temporally ambiguous cases enough to
   justify cost and latency.
3. Build a Mage frame-mode handler to test whether the paper's video advantage
   transfers at all before changing input contracts.
4. Only after that result, design and review a versioned codec-segment contract for
   Mage. Re-run the same test set in codec mode and compare quality, GPU seconds,
   schema validity, and critical-event recall.
5. Promote a candidate only after representative endpoint qualification; leave
   `production_eligible` unchanged until the broader production evidence exists.

## Source Notes

- Mage's direct benchmarks, training-data counts, token-reduction claim,
  dependencies, and streaming script behavior come from the
  [official Mage-VL README at the pinned revision](https://github.com/microsoft/Mage/blob/8c94a0ac905167f40b05b09332b78752b7f9fbef/mage_vl/README.md), its
  [requirements file](https://github.com/microsoft/Mage/blob/8c94a0ac905167f40b05b09332b78752b7f9fbef/mage_vl/requirements.txt), and
  [streaming script](https://github.com/microsoft/Mage/blob/8c94a0ac905167f40b05b09332b78752b7f9fbef/mage_vl/inference_streaming.py).
- Qwen's release date, runtime guidance, context-expansion instructions, and
  evaluation-reproduction caveats come from the
  [official Qwen3-VL README at the pinned revision](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md).
- Qwen's 4B Instruct/Thinking benchmark figures are the official images linked by
  the repository: [Instruct](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_4b_8b_vl_instruct.jpg) and
  [Thinking](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_4b_8b_vl_thinking.jpg).
- Local integration assertions in this report are grounded in
  [`src/robata/inference/adapter.py`](../src/robata/inference/adapter.py),
  [`src/robata/inference/input_plan.py`](../src/robata/inference/input_plan.py),
  [`src/robata/inference/runpod.py`](../src/robata/inference/runpod.py), the
  repository's deployment-status caveats in [`README.md`](../README.md), and the
  target-workload context in [`governance/REQUIREMENTS.md`](../governance/REQUIREMENTS.md).

## Bottom Line

**Use Qwen3-VL-4B-Instruct as the first integration and qualification target.** It
minimizes initial serving and input-representation risk. **Keep Mage-VL as the
high-upside video-streaming contender**, because the published matched 4B table
shows a large temporal advantage, but do not promote it until a six-camera,
codec-aware, schema-valid, failure-tested Robata A/B experiment proves that
advantage under the actual workload.
