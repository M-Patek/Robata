# Mage-Compatible Small Encoder Shadow Qualification (2026-08-08)

## Decision

The first Mage-compatible small-encoder experiment is **not qualified** for a
canary or production authority role.

- **Publication authority remains the complete Mage native codec/video path.**
- The candidate remains `shadow_only=True`; it cannot publish facts, suppress a
  native Mage invocation, or replace a native inference artifact.
- The 24-layer token-selection candidate reduced prompt tokens but was slower and
  lost too much action content.
- The 16-layer early-exit candidate was materially worse: it produced semantic
  drift, two invalid/truncated outputs, and decoder runaway that made it about
  2.84 times as slow as its paired native control.
- A genuinely separate small encoder is not yet a credible decoder input. It needs
  a trained projector or distilled adapter, not an untrained tensor shortcut.

Under the current deterministic gates, all retained real-run reports evaluate to:

```text
REJECTED_SHADOW_KEEP_MAGE_NATIVE_AUTHORITY
```

The archived reports were produced before the aggregate gates were added and store
`SHADOW_ONLY_NOT_AUTHORITY` as their original verdict. Re-evaluating their retained
native and candidate output text with the current evaluator produces the rejection
above; the historical report files are not rewritten.

## Scope of the experiment

This qualification deliberately tested the smallest useful execution shape before
multi-camera or high-concurrency work:

- one 40-second real sample;
- one camera, `cam_01`;
- five non-overlapping segments of approximately eight seconds each;
- one resident Mage-VL model;
- 4-bit NF4 inference on one NVIDIA GeForce RTX 4060 Laptop GPU with 8 GiB VRAM;
- one worker and one generation in flight;
- native and candidate generations executed sequentially for the same segment;
- the same compact observation prompt and maximum output budget for both paths.

This is a paired parity and performance test. It is **not** a six-camera fusion
qualification, a concurrency benchmark, or a human-labeled accuracy benchmark.
Mage native is the control and current authority, but its output is not claimed to
be ground truth.

## Implemented shadow architecture

The implementation does not accept arbitrary embeddings from an unrelated small
encoder. Mage's decoder expects visual features in its own 2560-dimensional merger
space, with the count and ordering of visual features matching the image placeholder
tokens in the prompt. No separately trained projector checkpoint is available in
this repository or in the local Mage checkpoint.

The safe experimental path therefore reuses the resident Mage components:

```text
same codec video segment
        |
        +-----------------------------+
        |                             |
        v                             v
Mage native authority          shadow candidate
full Mage visual tower         Mage visual tower
        |                      (24 layers or early exit)
Mage visual merger                    |
        |                       Mage visual merger
all temporal visual runs              |
        |                       keep complete runs
        |                       [0, 2, 4, 7] of 8
        |                             |
        +------------+----------------+
                     v
              same Qwen decoder
            one path at a time
```

The candidate:

1. runs Mage's own visual encoder and merger;
2. optionally limits the visual tower to an early layer budget;
3. identifies complete temporal placeholder runs;
4. keeps evenly distributed complete runs, always including the first and last;
5. rewrites the placeholder sequence to match the retained feature rows;
6. injects only Mage merger-space features through `inputs_embeds` into the same
   decoder;
7. records policy identity, token counts, retained run indices, preparation timing,
   and a feature-content digest.

It does **not** perform cross-time mean pooling. Earlier probes showed that such
pooling erased useful boundary and action evidence. It also rejects arbitrary
external feature dimensions rather than pretending that shape-compatible tensors
are semantically decoder-compatible.

The experimental report is an internal diagnostic artifact, not a published wire
schema. The implementation does not change a registered schema, publication
identity, canonical fact contract, or the Mage native authority boundary.

## What “small encoder” means in these two runs

The two candidates answer different questions and must not be conflated:

| Run | Visual layers | Temporal operation | Correct interpretation |
| --- | ---: | --- | --- |
| r2 | 24 / 24 | Keep 4 of 8 complete runs | Token-compression probe after the full Mage visual tower; mechanically compatible, but not a truly smaller encoder. |
| r3-lite16 | 16 / 24 | Keep 4 of 8 complete runs | First untrained encoder-lite probe using early exit inside Mage's own tower. |

Both runs reduced each segment from 384 to 192 visual tokens. Because non-visual
prompt tokens remained, total prompt length fell from 768 to 576 tokens per segment,
or 25%, rather than 50%.

## Reproducibility and lineage

### Fixed local inputs

| Item | Value |
| --- | --- |
| Mage checkpoint | `D:\HuggingFace\Mage-VL` |
| Checkpoint manifest SHA-256 | `a15e49d965e4ad61455ef02bb770b626755959a4b7aa46a140a342f2ed62e290` |
| Source qualification root | `D:\Github\Robata\.local\mage-vnext-qualification-20260808-r9` |
| Source report file SHA-256 | `3609a70ab638b77f6c5088f715d5af3c7c0567db2e010e7ec19f06dd27a9180a` |
| Codec cache | `D:\Github\Robata\.tmp_mage_cache_final` |
| Materialized segment root | `D:\Github\Robata\.tmp_mage_stream_materialization\segments` |
| Runtime profile | `bitsandbytes_4bit_nf4_v1` |
| Model revision label | `local-2026-08-07` |
| Camera | `cam_01` |
| Recording duration | 40.0 seconds |
| Segment count | 5 |

The source qualification report binds each focus segment to a context-manifest
semantic digest and durable media path. Each new report additionally retains the
segment content digest, candidate policy digest, retained temporal runs, and
feature-content digest.

### Retained real-run reports

| Run | Report path | Embedded report identity | Byte-exact file SHA-256 | Policy SHA-256 |
| --- | --- | --- | --- | --- |
| r2 | `D:\Github\Robata\.local\mage-small-encoder-shadow-r2\report.json` | `7b9f4c31177a70668c00d8e6ee7a005b43c9cf060b8602cf65cf7daa7316cf0a` | `42e4c8b5171c03d1e39c55c548eae4d75989d7e0d6c28809b9041d2c8582541d` | `feba9e52a6313600dc2df56f266cc19bb65ef009299c9aa1fe7d1884ec997599` |
| r3-lite16 | `D:\Github\Robata\.local\mage-small-encoder-shadow-r3-lite16\report.json` | `5e3269a6b5abea9ad26482dbb7d5ec83840672c0665c1a6a04f4f26d3d2eebfe` | `ea6c5a98020dbdd9de3a83925bd906664725cfa77959be4e919d1fae9ae51874` | `e5c0f3bc7df4559a885b4d9eb44a8c2cd259ab738875a1b347912e0505cfc90c` |

“Embedded report identity” is the canonical digest stored inside the report. It is
not the SHA-256 of the final serialized file after the digest field is inserted;
the byte-exact file digest is listed separately.


The retained reports were also re-evaluated without modifying their historical
bytes. The deterministic analysis artifacts are local evidence, not repository
contracts:

| Run | Analysis path | Analysis SHA-256 | Source embedded hash verified |
| --- | --- | --- | --- |
| r2 | `D:\Github\Robata\.local\mage-small-encoder-shadow-r2\analysis-v3.json` | `055f9bc8bfc76be6f6dc95a20257c56cbc236eb7e802659568441898d447985a` | yes |
| r3-lite16 | `D:\Github\Robata\.local\mage-small-encoder-shadow-r3-lite16\analysis-v3.json` | `16ff2efd9a69de469786c7809974e835e51e38ee7b40e29c36fc0065a7f304ea` | yes |
| r4-v2 | `D:\Github\Robata\.local\mage-small-encoder-shadow-r4-v2-20260808\analysis-v3.json` | `2fa48fe72db077d3359be8873c2682bac2ec19a30ff14cef1c0ae083923def54` | yes |

Each analysis binds the byte-exact source report SHA-256, verifies the report's
embedded canonical digest, records evaluator version and exact evaluator-source
SHA-256, recomputes every segment comparison from retained raw output text, and
applies the current aggregate gates. An invalid embedded report identity produces
`INVALID_SOURCE_REPORT_IDENTITY` and can never qualify.

### Equivalent PowerShell reproduction

Run from this branch's worktree. The codec overlay is required by the local Mage
codec processor.

```powershell
$env:PYTHONPATH = "D:\tmp\mage-dcvc-python-overlay;$PWD\src"
$env:HF_HOME = "C:\Users\asus\.cache\huggingface"

python scripts\run_local_mage_small_encoder_shadow.py `
  --qualification-root D:\Github\Robata\.local\mage-vnext-qualification-20260808-r9 `
  --model-dir D:\HuggingFace\Mage-VL `
  --codec-cache-dir D:\Github\Robata\.tmp_mage_cache_final `
  --output-dir D:\Github\Robata\.local\mage-small-encoder-shadow-r2-repro `
  --visual-layer-count 24 `
  --max-temporal-runs 4 `
  --checkpoint-manifest-path D:\Github\Robata\.local\mage-qual-v7-endpoint\checkpoint-manifest-v2.json `
  --repetitions 3 `
  --warmup-max-new-tokens 8 `
  --max-new-tokens 512
```

For the early-exit candidate:

```powershell
python scripts\run_local_mage_small_encoder_shadow.py `
  --qualification-root D:\Github\Robata\.local\mage-vnext-qualification-20260808-r9 `
  --model-dir D:\HuggingFace\Mage-VL `
  --codec-cache-dir D:\Github\Robata\.tmp_mage_cache_final `
  --output-dir D:\Github\Robata\.local\mage-small-encoder-shadow-r3-lite16-repro `
  --visual-layer-count 16 `
  --max-temporal-runs 4 `
  --checkpoint-manifest-path D:\Github\Robata\.local\mage-qual-v7-endpoint\checkpoint-manifest-v2.json `
  --repetitions 3 `
  --warmup-max-new-tokens 8 `
  --max-new-tokens 512
```

The current command verifies the pinned Mage checkpoint manifest before loading one
model, performs an unscored 8-token warmup for both paths, alternates native-first and
candidate-first order, and records three repetitions per segment by default. Model-load
time is recorded separately and is not included in scored generation sums or warm RTF
calculations. The retained r2/r3 reports below were produced by the earlier v1 runner;
the v2 real rerun remains the current raw qualification evidence. The checked-in
runner now emits report v3 for future runs: it fail-closes on durable-media hash
mismatch and records native and candidate CUDA allocation peaks separately.

For a report already on disk, no model load is needed to reproduce the current gates:

```powershell
$env:PYTHONPATH = "D:\Github\Robata\.worktrees\small-encoder-shadow-20260808\src"
python scripts\re_evaluate_mage_small_encoder_shadow.py `
  D:\Github\Robata\.local\mage-small-encoder-shadow-r4-v2-20260808\report.json `
  --output D:\Github\Robata\.local\mage-small-encoder-shadow-r4-v2-20260808\analysis-v3.json
```

The same command can re-evaluate the historical r2 and r3 reports. It verifies the
embedded report identity, binds the byte-exact source report hash, and never rewrites
the historical report.

## Quantitative results

### Current v2 selected-only rerun (three repetitions)

The v2 policy is an explicit identity bump from v1:

```text
mage-small-encoder-shadow-v2
UNIFORM_TEMPORAL_RUN_KEEP_NO_EMPTY_SPANS_V2
```

For dropped temporal runs it removes the matching `vision_start`/`vision_end`
wrapper instead of leaving an empty visual span. It retains the timestamp marker,
validates the decoder hidden-size alignment, synchronizes CUDA phase timing, and
uses an alternating order with one warmup. This avoids the most important v1
confound without changing the native authority path.

| Item | Value |
| --- | --- |
| Report | `D:\Github\Robata\.local\mage-small-encoder-shadow-r4-v2-20260808\report.json` |
| Embedded report identity | `d3e78983d12d29556e3b42d331ce84e33e829b2ad6e167a84a48beb0074d66c8` |
| Byte-exact report SHA-256 | `ed50304ad81194a0adb7bce2da46d20d11e1fe678ec90164b3b9bd8e73dadc79` |
| Policy SHA-256 | `e743009ea93c37d992c1b310bff8caf684af8cb6e09621ce658bd7c123b0d145` |
| Checkpoint manifest SHA-256 | `a15e49d965e4ad61455ef02bb770b626755959a4b7aa46a140a342f2ed62e290` |
| Manifest file SHA-256 | `d8748fcde67eea30019a59c3725de8230ac304e3d7eca7068c6094569bda25d6` |
| Model load (not scored) | 14.362 s |
| Segments / repetitions / paired calls | 5 / 3 / 15 |
| Native generation total / p50 / p95 | 106.176 s / 4.074 s / 20.803 s |
| Candidate generation total / p50 / p95 | 143.655 s / 4.207 s / 26.064 s |
| Candidate preparation total / p50 / p95 | 2.755 s / 0.183 s / 0.207 s |
| Candidate total (prep + generation) | 146.410 s |
| Native / candidate speedup | 0.725x |
| Effective-media warm RTF (native / candidate) | 1.130x / 0.820x |
| Native / candidate prompt tokens | 11,520 / 8,520 (26.0% fewer candidate tokens) |
| Native / candidate output tokens | 1,110 / 1,761 |
| Native / candidate parsed actions | 27 / 39 |
| Exact-label matches / recall / precision | 6 / 22.2% / 15.4% |
| False-silence repetitions | 0 |
| Candidate compact-contract invalid repetitions | 3 / 15 |
| Repeated-label excess / rate (diagnostic) | 24 / 61.5% |
| Aggregate verdict | **REJECTED_SHADOW_KEEP_MAGE_NATIVE_AUTHORITY** |

All three invalid compact-contract cases are segment 0's misspelled interval key;
JSON syntax remained valid. The candidate still repeated one wrong `reaches for`
action six times in that segment. Segments 1 and 2 matched exactly; segment 3
repeated `fold a green cloth` four times rather than the native action; segment 4
consistently hallucinated a mouse-at-desk action instead of wiping the table.

The v2 result is stronger than the original single-pass result: the order is
alternated, warmup is excluded, p50/p95 are reported, and the candidate remains
slower and far below the exact parity gates. The first-segment long generation is
not a cold model-load artifact: model load is separate, and the candidate's r4
segment-0 generation remains 22.176 to 26.064 seconds after warmup.

### End-to-end paired summary (historical v1 probes)

Candidate time is `candidate preparation + candidate generation`. RTF here means
`40 seconds of media / measured warm path time`; values above 1.0 are faster than
real time. These RTF values exclude one-time model loading.

| Metric | r2: 24 layers + run keep | r3-lite16: 16 layers + run keep |
| --- | ---: | ---: |
| Model load, separate from warm path | 14.010 s | 12.560 s |
| Native generation sum | 36.225 s | 35.988 s |
| Candidate preparation sum | 0.911 s | 0.660 s |
| Candidate generation sum | 38.370 s | 101.645 s |
| Candidate total | 39.280 s | 102.305 s |
| Native / candidate speedup | 0.922x | 0.352x |
| Candidate change versus native time | 8.4% slower | 184.3% slower |
| Native warm RTF | 1.104x | 1.111x |
| Candidate warm RTF | 1.018x | 0.391x |
| Native prompt tokens | 3,840 | 3,840 |
| Candidate prompt tokens | 2,880 | 2,880 |
| Prompt-token reduction | 25.0% | 25.0% |
| Native output tokens | 370 | 370 |
| Candidate output tokens | 463 | 1,182 |
| Native parsed actions | 9 | 9 |
| Candidate parsed actions | 10 | 3 |
| Exact-label matches | 2 | 0 |
| Exact-label recall | 22.2% | 0.0% |
| Exact-label precision | 20.0% | 0.0% |
| False-silence segments | 0 | 2 |
| Candidate duplicate actions | 5 | 0 parsed duplicates |
| Candidate duplicate rate | 50.0% | 0.0% of parsed actions |
| Matched start/end boundary MAE | 0.0 s / 0.0 s | Not measured: no exact matches |

The r3 duplicate value is not a success signal. Its last two candidate generations
reached the 512-token limit and ended as invalid JSON, so their repeated partial
content was intentionally excluded by the strict parser.

### GPU activity during paired execution

These samples cover the combined sequential native/candidate loop after model load.
They do not isolate one path, and they are not directly comparable to telemetry from
a different endpoint harness or process boundary.

| Metric | r2 | r3-lite16 |
| --- | ---: | ---: |
| Samples | 71 | 129 |
| Mean GPU utilization | 87.24% | 86.65% |
| Maximum GPU utilization | 100% | 100% |
| Mean GPU memory used | 5,353 MiB | 5,359 MiB |
| Maximum GPU memory used | 5,378 MiB | 5,396 MiB |
| Reported total memory | 8,188 MiB | 8,188 MiB |
| Mean power | 28.59 W | 28.60 W |
| Maximum power | 32.64 W | 31.85 W |
| Maximum temperature | 54 C | 54 C |

The paired loop demonstrates that sequential work can keep this GPU busy. High GPU
utilization did not translate into better candidate throughput or quality. The
important bottleneck was decoder behavior after visual conditioning changed, not a
lack of runnable GPU work.

## Current deterministic qualification gates

The v3 evaluator applies all gates together:

1. every native and candidate output must be valid JSON syntax;
2. every native and candidate output must satisfy the compact interval/confidence
   contract (syntax alone is not enough);
3. candidate false silence must be zero;
4. exact-label recall and precision against the paired native output must each be at
   least 0.90;
5. every exact-label match must have measurable start and end offsets, and aggregate
   start/end boundary MAE must each be at most 0.50 seconds;
6. `native seconds / candidate total seconds` must be at least 1.0.

Repeated-label excess remains a diagnostic rather than a standalone gate until a
versioned event ontology can distinguish legitimate repeated actions from decoder
collapse.

| Gate | r2 re-evaluation | r3-lite16 re-evaluation | r4-v2 |
| --- | --- | --- | --- |
| JSON syntax valid | PASS | **FAIL**: segments 3 and 4 truncated | PASS |
| Compact contract valid | **FAIL**: misspelled interval key | **FAIL**: truncated segments | **FAIL**: 3 segment-0 repetitions |
| Zero false silence | PASS | **FAIL**: 2 segments | PASS |
| Exact-label recall >= 0.90 | **FAIL**: 0.222 | **FAIL**: 0.000 | **FAIL**: 0.222 |
| Exact-label precision >= 0.90 | **FAIL**: 0.200 | **FAIL**: 0.000 | **FAIL**: 0.154 |
| Boundary measurement complete | PASS | **FAIL**: no exact matches | PASS |
| Start boundary MAE <= 0.50 s | PASS: 0.0 s | **FAIL**: no exact matches | PASS: 0.0 s |
| End boundary MAE <= 0.50 s | PASS: 0.0 s | **FAIL**: no exact matches | PASS: 0.0 s |
| Candidate no slower than native | **FAIL**: 0.922x | **FAIL**: 0.352x | **FAIL**: 0.725x |
| Aggregate qualification | **REJECT** | **REJECT** | **REJECT** |

A pass on an individual gate never authorizes promotion. All gates must pass, and a
future canary would still require reviewed human-labeled quality evidence.

## Behavioral findings

### r2: token reduction without visual early exit

The mechanically safest candidate still failed:

- Segment 0 replaced five control observations with six repetitions of one
  “reaches for” action. Five of its six parsed actions were duplicates.
- Segments 1 and 2 matched the control action text exactly and also matched the
  reported start/end offsets exactly.
- Segment 3 used `fold a green cloth` where the control used
  `a person folds a green piece of clothing`. This may be semantically similar,
  but the exact evaluator intentionally does not award a match.
- Segment 4 diverged from the control's table-wiping action and described using a
  mouse at a desk.
- The candidate emitted 463 output tokens versus 370 for the control. The extra
  autoregressive work more than consumed the prompt-token saving.

This run proves mechanical decoder compatibility, not semantic compatibility.
Keeping half the temporal runs after the complete visual tower was insufficient.

### r3-lite16: untrained early exit

The 16-layer experiment failed more severely:

- Its first three valid segments all diverged from the control action, describing a
  bag of food, a gray dress, and walking to a couch.
- Segments 3 and 4 entered long repetitive generations, reached the 512-token
  budget, and produced invalid/truncated JSON.
- Those last two generations took approximately 44.54 and 43.59 seconds.
- Total candidate output grew to 1,182 tokens and candidate time grew to 102.305
  seconds despite saving about 0.25 seconds of preparation versus r2.

The result is consistent with an untrained representation shift: decoder
conditioning deteriorated, output length expanded, and the autoregressive decoder
became the dominant cost. Simply stopping Mage's visual tower at layer 16 is not a
credible small-encoder design.

## Strict compact + exact lexical evaluator: interpretation and limits

`robata.benchmark.mage_small_encoder` separates syntax validity from compact-contract
validity, then performs a narrow deterministic parity comparison:

- syntax validity requires a JSON object;
- compact validity requires an `observations` list, a non-empty action string, and
  exactly one complete finite increasing interval coordinate pair;
- optional confidence and visibility values must be finite numbers in `[0, 1]`;
- each action is lowercased and every non-alphanumeric run becomes `_`;
- actions are grouped by the resulting exact normalized string;
- repeated-label excess is retained as a diagnostic, not treated as a governed
  duplicate gate until an event ontology exists;
- false silence means the control has at least one parsed action and the candidate
  has none;
- boundary deltas are calculated only for paired occurrences with the same exact
  normalized label and finite corresponding offsets.

It does **not** perform embedding similarity, ontology mapping, synonym matching,
LLM judging, confidence calibration, or comparison to human annotations. Therefore:

- `fold a green cloth` and `a person folds a green piece of clothing` do not match;
- the reported 0.0-second boundary MAE in r2 covers only two exact-label matches and
  says nothing about the seven unmatched control actions;
- invalid/truncated JSON contributes no parsed action; syntax-valid output with a
  misspelled interval key remains syntax-valid but fails compact-contract validity;
- exact recall measures candidate parity with Mage native, not objective event
  recall.

This conservatism is intentional for a first shadow gate. A future semantic metric
must use a versioned action ontology and a reviewed labeled calibration set rather
than silently treating a language model judge as ground truth.

## Why prompt compression did not create a speedup

The experiment separates two costs:

```text
candidate total = visual preparation + autoregressive generation
```

Visual preparation was already small: 0.911 seconds across all five r2 segments and
0.660 seconds for r3. Reducing visual tokens or layers only helps if decoder output
quality and length remain controlled. They did not:

- r2 used 25% fewer prompt tokens but generated 25% more output tokens;
- r3 used the same smaller prompt but generated more than three times the control's
  output tokens;
- two r3 generations ran to the 512-token ceiling.

The first optimization target is therefore not an even more aggressive untrained
encoder cut. It is preserving a decoder-compatible semantic representation and
maintaining a bounded compact output contract.

## Required next step for a true separate small encoder

A real architecture of:

```text
camera stream -> separate small encoder -> compressed observation -> Mage
```

requires training and qualification work that this first experiment intentionally
did not fabricate:

1. select and version the separate visual encoder checkpoint;
2. define temporal and, later, camera-position semantics for its output;
3. train a projector or adapter from that representation into Mage's expected
   decoder conditioning space, or distill a structured observation model against
   the complete Mage path;
4. train on representative robot actions and hard negative/no-action segments;
5. retain native media and Mage native inference as replay and targeted-refinement
   authority;
6. evaluate with human-labeled events, boundaries, false silence, duplicates,
   calibration, latency, output length, GPU memory, and RTF;
7. repeat shadow qualification before any canary decision.

Until that work exists, a separate lightweight encoder may be used for a cheap
non-authoritative signal such as camera ranking or admission/gate research, but its
features must not be presented to Mage as if they were native visual tokens.

## Operational conclusion

The small-encoder seam was necessary to test and is useful as an experimental
boundary, but neither tested candidate is production-ready. The result supports a
simple operational decision:

```text
keep one camera + one worker + Mage native codec/video as authority
retain small encoder only as shadow research
train compatibility before attempting promotion
```

No production route, event publication, evidence record, or replay authority should
change as a result of these runs.
