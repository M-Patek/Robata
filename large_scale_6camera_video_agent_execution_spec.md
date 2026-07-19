# Large-Scale 6-Camera Video QA & Annotation System
## Agent Execution and Research Specification

Version: V1  
Status: Consolidated execution baseline  
Purpose: This document packages the current project requirements, fixed architectural decisions, throughput constraints, research directions, and execution steps into one agent-readable specification.

---

# 1. Mission

Design and implement a large-scale multimodal video processing system for robotic / egocentric manipulation data.

The system processes `.mcap` recordings that are decoded into MP4 video streams.

Each `.mcap` contains:

- 6 camera streams
- approximately 2–5 minutes of recording
- synchronized or logically aligned multi-camera views

Current workload target:

- approximately 500 hours/day of incoming video data
- QA target: data generated on day T should ideally have QA completed by end of T+1
- the exact definition of "500 hours/day" must be verified:
  - whether it means 500 physical recording hours/day
  - or 500 aggregated camera-video hours/day

If 500 hours/day means physical recording duration, then 6-camera aggregate volume is:

```text
500 recording hours/day
× 6 cameras
= 3000 camera-video-hours/day
```

This distinction must remain explicit in all throughput and capacity calculations.

The project priority is:

> Speed first, while preserving quality.

The system must maximize throughput without unacceptable degradation in QA, temporal annotation, action understanding, or downstream data value.

---

# 2. Fixed Design Decisions

The following decisions are considered current baseline requirements and should not be silently changed.

## 2.1 Native 6-Camera Architecture

The system must be designed as a native 6-camera system from the beginning.

Do not treat the architecture as a 2-camera prototype that may later be extended.

The system must support:

- QA across all 6 cameras
- action understanding using evidence from all 6 cameras
- multi-view reasoning
- camera-level and event-level outputs
- synchronized retrieval of clips from all relevant views

However:

> 6-camera coverage does not necessarily mean every expensive model call must consume all 6 cameras.

The system may dynamically select the most useful camera evidence for expensive reasoning, as long as full 6-camera coverage remains available.

---

## 2.2 MCAP Is the Source Container, MP4 Is the Main Video Analytics Input

The current data path is:

```text
MCAP
↓
Decode / extraction
↓
6 × MP4
↓
Video Analytics Pipeline
```

The main large-throughput optimization target is therefore:

```text
MP4
→ useful QA / annotation / ActionEvent
```

Do not over-optimize the MCAP container layer unless profiling shows it is a real bottleneck.

---

## 2.3 Qwen Is the Primary Production VLM

The default production visual-language inference path is:

```text
Qwen
```

Qwen is responsible for the primary production workload.

The architecture must not hard-code Qwen inside business logic.

Use a model abstraction layer.

---

## 2.4 GPT Is a Shadow / Research / Canary Path

GPT should initially be used as:

- Shadow model
- Judge
- Teacher
- Hard-case escalation model
- Research comparison model
- Replacement candidate

GPT must not initially block the Qwen production path.

The architecture must support gradual migration:

```text
Stage 1
Qwen Production
+
GPT Shadow

Stage 2
Qwen Production
+
GPT Hard-Case Escalation

Stage 3
Qwen Majority
+
GPT Canary

Stage 4
Dynamic Routing

Stage 5
Potential GPT Primary
if quality + effective cost + capacity + reliability are proven
```

---

## 2.5 Video Must Be Compressed Before Expensive VLM Reasoning

Do not directly send complete MP4 videos or all frames to a large VLM.

The system should transform raw video into compact temporal evidence.

Possible mechanisms include:

- sparse sampling
- adaptive sampling
- event-driven sampling
- dense sampling around candidate boundaries
- ROI extraction
- camera selection
- semantic compression
- temporal token compression

---

# 3. Core System Entities

The system should define the following logical entities.

## 3.1 MCAP

Fields should include:

- mcap_id
- source_path
- recording_start_time
- recording_end_time
- duration
- camera_count
- ingestion_time
- processing_status

Expected:

```text
camera_count = 6
```

Unexpected camera counts must be logged explicitly.

---

## 3.2 CameraStream

Each MCAP should map to 6 camera streams.

Fields:

- camera_id
- camera_role
- source_mcap_id
- mp4_path
- resolution
- fps
- codec
- start_timestamp
- end_timestamp
- frame_count

Traceability requirement:

```text
CameraStream
→ source MCAP
```

must always be available.

---

## 3.3 TemporalWindow

A TemporalWindow represents a time interval for analysis.

Example:

```json
{
  "window_id": "...",
  "mcap_id": "...",
  "start_time": 12.0,
  "end_time": 16.0
}
```

---

## 3.4 TemporalVisualPackage

This is the unified model input abstraction.

Example:

```json
{
  "package_id": "...",
  "mcap_id": "...",
  "start_time": 12.0,
  "end_time": 16.0,
  "cameras": {
    "cam_01": [],
    "cam_02": [],
    "cam_03": [],
    "cam_04": [],
    "cam_05": [],
    "cam_06": []
  },
  "sampling_metadata": {
    "strategy": "...",
    "sampling_rate": "...",
    "frame_count": "...",
    "resolution": "...",
    "total_pixels": "..."
  }
}
```

This should be model-agnostic.

It may be consumed by:

- Qwen
- GPT
- future VLMs

---

## 3.5 CameraQAResult

Example:

```json
{
  "mcap_id": "...",
  "camera_id": "...",
  "start_time": 0.0,
  "end_time": 120.0,
  "status": "usable",
  "issues": [],
  "severity": "low",
  "confidence": 0.98
}
```

QA issues may include:

- camera blur
- motion blur
- hand occlusion
- object occlusion
- viewpoint quality
- framing problems
- exposure problems
- corrupted stream
- unusable view

---

## 3.6 MCAPQAResult

Example:

```json
{
  "mcap_id": "...",
  "overall_status": "usable",
  "usable_camera_count": 5,
  "camera_results": [],
  "confidence": 0.94
}
```

Do not reject an entire MCAP automatically because one camera is degraded.

Distinguish:

```text
Camera Quality
vs
Overall Event/Data Quality
```

---

## 3.7 CandidateEventWindow

Represents a possible action interval found during event proposal.

Example:

```json
{
  "mcap_id": "...",
  "start_time": 14.0,
  "end_time": 18.0,
  "proposal_type": "interaction_candidate",
  "confidence": 0.82
}
```

---

## 3.8 ActionEvent

The central downstream semantic entity.

Example:

```json
{
  "event_id": "...",
  "mcap_id": "...",
  "start_time": 13.12,
  "end_time": 15.48,
  "action_type": "grasp",
  "hand": "right",
  "object": "cup",
  "confidence": 0.94,
  "camera_evidence": {
    "cam_01": {},
    "cam_02": {},
    "cam_03": {},
    "cam_04": {},
    "cam_05": {},
    "cam_06": {}
  }
}
```

One physical action should ideally correspond to one ActionEvent.

Cameras are evidence sources, not six separate copies of the same action.

---

## 3.9 ModelInference

Every model invocation should be logged.

Fields:

- inference_id
- package_id
- provider
- model_name
- model_version
- prompt_version
- input_config
- sampling_config
- visual_token_estimate
- latency
- output
- confidence
- retry_count
- failure_status
- created_at

---

## 3.10 ModelDisagreementSample

Store cases where Qwen and GPT disagree.

Examples:

```text
Qwen: grasp
GPT: reach
```

or:

```text
Qwen start = 13.1
GPT start = 14.0
```

These cases are valuable for:

- model evaluation
- prompt optimization
- human review
- teacher labeling
- student training
- distillation
- routing design

---

# 4. Main Processing Flow

The high-level production path is:

```text
6-Camera MCAP
        ↓
Decode to MP4
        ↓
Timestamp Alignment
        ↓
Video Analytics / Sampling Layer
        ↓
TemporalWindow / TemporalVisualPackage
        ↓
Qwen Primary Path
        ↓
QA / Event Proposal / Action Understanding
        ↓
Multi-View Fusion
        ↓
ActionEvent
        ↓
Data Value Scoring
        ↓
Optional Deep Processing
        ↓
Structured Storage
        ↓
Multimodal Temporal Index
        ↓
Action-Level Retrieval
```

Shadow path:

```text
TemporalVisualPackage
        ↓
GPT Shadow / Research
        ↓
GPT Result
        ↓
Qwen vs GPT Evaluation
        ↓
Disagreement Dataset
```

---

# 5. Time Alignment

All six cameras must use a common logical time base.

Core indexing must use timestamps rather than frame index alone.

Every result should be traceable through:

```text
ActionEvent
→ Timestamp
→ Camera
→ MP4
→ MCAP
```

For the same physical event, the system must be able to retrieve synchronized or corresponding intervals from all camera views.

---

# 6. Sampling and Compute Elimination

The system should not assume that every frame deserves expensive inference.

The main optimization objective is broader than caching.

Think in three categories:

```text
1. Do not compute
   → filtering / sampling / cascade

2. Compute more cheaply
   → lower resolution / ROI / smaller model / quantization

3. Do not recompute
   → cache / feature reuse / encoder reuse
```

The first category may provide the largest gains.

---

# 7. Sampling Modes

## 7.1 Uniform Sampling

Examples:

- 0.2 FPS
- 0.5 FPS
- 1 FPS
- 2 FPS

Use for:

- coarse QA
- scene understanding
- initial event proposal

Do not hard-code one FPS.

---

## 7.2 Adaptive Sampling

Sampling density should increase when visual or semantic activity increases.

Candidate signals:

- frame difference
- optical flow
- motion vectors
- scene change
- hand motion
- object motion
- hand-object distance changes
- occlusion changes
- contact candidate
- blur spike

Concept:

```text
Stable Segment
→ very sparse sampling

Motion Segment
→ medium sampling

Potential Interaction
→ high sampling

Boundary Candidate
→ dense sampling
```

---

## 7.3 Dense Temporal Sampling

Used only for high-value candidate windows.

Example:

```text
coarse scan detects possible action at 12–16s
↓
expand to 11–17s
↓
sample at higher temporal density
↓
refine action boundary
```

---

# 8. Visual Budget Optimization

Do not optimize only FPS.

Visual compute depends on:

```text
Frames
×
Resolution
×
Total Pixels
≈
Visual Token Budget
```

The system should support:

- sampling rate
- resolution
- frame count
- total pixels
- ROI size

as independent control variables.

Recommended conceptual profiles:

```text
Coarse QA
Low FPS
Low resolution
Low total pixels

Fine QA
Medium FPS
Medium resolution

Action Analysis
Medium FPS
Selective high-detail ROI

Boundary Refinement
High FPS
Focused spatial region
Controlled total pixels
```

---

# 9. ROI / Region Intelligence

Consider a two-level spatial representation:

```text
Low-resolution full frame
+
High-resolution hand/object/interaction ROI
```

Potential ROI targets:

- hands
- manipulated objects
- hand-object interaction zone
- relevant workspace region

This is intended to preserve:

- global temporal context

while spending high-resolution compute only where detail matters.

---

# 10. Camera Compute Strategy

The architecture must preserve 6-camera coverage.

However, expensive reasoning may use dynamic evidence selection.

Possible process:

```text
6 Camera Coarse Analysis
        ↓
Camera Utility / Quality Scoring
        ↓
Select Best Evidence Cameras
        ↓
Joint Reasoning
```

Possible future Camera Utility Model inputs:

- action type
- hand position
- object position
- occlusion
- viewpoint quality
- motion visibility
- image quality

Output:

```text
cam_2 = high relevance
cam_5 = high relevance
cam_3 = medium relevance
...
```

---

# 11. Progressive Multi-Camera Evidence

Instead of always loading all six cameras into every expensive request:

```text
Step 1
Best primary view

If sufficient:
→ output

If uncertain:
→ add second view

If still uncertain:
→ add more views

Maximum:
→ all 6 cameras
```

This is called here:

```text
Progressive Evidence Acquisition
```

The production architecture remains 6-camera native.

---

# 12. Multi-View Fusion

Do not create six independent ActionEvents for one physical action.

Desired logic:

```text
Camera 1
Camera 2
Camera 3
Camera 4
Camera 5
Camera 6
        ↓
Camera Evidence
        ↓
Multi-View Fusion
        ↓
One Physical ActionEvent
```

Example:

```text
Cam 2:
contact evidence

Cam 3:
object movement evidence

Cam 5:
object identity evidence

Cam 6:
overall action evidence
```

Final output:

```text
ActionEvent = grasp
```

---

# 13. QA Design

QA must cover all 6 cameras.

QA should contain:

## Camera-Level QA

Each camera independently evaluates:

- blur
- motion blur
- occlusion
- camera obstruction
- framing
- exposure
- corruption
- view usability

## MCAP-Level QA

Aggregate all six camera results.

Do not reject the full MCAP solely because one camera fails.

---

# 14. QA Quality vs Throughput Tradeoff

The system must experimentally measure:

```text
Sampling Rate
vs
QA Recall
vs
QA Precision
vs
Inference Cost
vs
Latency
```

Recommended benchmark grid:

```text
0.2 FPS
0.5 FPS
1 FPS
2 FPS
5 FPS
```

Also benchmark:

```text
FPS
×
Resolution
×
Total Pixels
```

The goal is not minimum FPS.

The goal is:

> minimum compute while maintaining acceptable QA error detection quality.

---

# 15. Two-Stage QA

Preferred research direction:

```text
Coarse QA
↓
Suspicious Window Detection
↓
Dense QA
```

Example:

```text
1 FPS scan
↓
Camera 4, 22–27s suspicious
↓
2–10 FPS re-analysis only around 21–28s
```

---

# 16. Event Proposal

Do not perform one huge action reasoning request over the entire 2–5 minute video.

First detect candidate intervals.

Example:

```text
0–10s idle
10–14s reach
14–17s grasp candidate
17–25s move
25–30s place
```

Then create candidate windows.

---

# 17. Temporal Action Reasoning

Action understanding must be temporal.

Core action classes may include:

- reach
- grasp
- move
- place
- release

Single frames may look similar.

Reasoning should consider:

- state before action
- state after action
- motion direction
- hand motion
- object motion
- contact establishment
- contact release
- whether object begins moving with the hand

The system is not a simple image classification pipeline.

---

# 18. Boundary Refinement

Separate:

```text
Action Detection
```

from:

```text
Temporal Boundary Refinement
```

Example:

```text
coarse grasp estimate:
12–15s

dense re-analysis:
11.5–15.5s

final:
13.12–13.84s
```

---

# 19. Model Abstraction Layer

Define:

```text
VisionModelAdapter
```

Input:

```text
TemporalVisualPackage
```

Output:

```text
Structured Result
```

Adapters:

- QwenAdapter
- GPTAdapter
- future VLM adapters

Business logic must not depend directly on provider-specific APIs.

---

# 20. Qwen Primary Path

Default:

```text
provider = qwen
```

Qwen handles production:

- QA
- action understanding
- temporal interpretation
- structured annotation
- event proposal
- multi-view reasoning

Record:

- model
- version
- latency
- visual input size
- frame count
- retry count
- failure status
- output confidence

---

# 21. GPT Shadow Path

Config:

```text
gpt_shadow_enabled = true / false
```

When enabled:

```text
TemporalVisualPackage
├── Qwen
└── GPT
```

Production output initially remains Qwen.

GPT result is saved separately.

GPT failure must not block Qwen production.

---

# 22. GPT Shadow Sampling

Do not automatically send 100% of production traffic to GPT.

Support:

```text
shadow_sample_ratio
```

Examples:

- 0.01
- 0.05
- 0.10
- 1.00

Selection modes:

- random sample
- Qwen low confidence
- high camera disagreement
- QA ambiguity
- temporal boundary uncertainty
- hard cases
- rare action classes
- research cohorts

---

# 23. GPT Replacement Evaluation

GPT may eventually replace part or most of Qwen if it satisfies all four:

```text
Quality
Effective Cost
Capacity
Operational Reliability
```

The key metric is not account price or API token price alone.

Use:

```text
Cost per successfully processed raw video hour
```

and:

```text
Cost per accepted / useful output
```

Possible migration:

```text
Qwen 95 / GPT 5
↓
Qwen 70 / GPT 30
↓
Qwen 20 / GPT 80
↓
GPT Primary / Qwen Backup
```

Only move after benchmark evidence.

---

# 24. Model Cascade

Do not assume one general VLM should process every case.

Research:

```text
Cheap CV
↓
Specialist Model
↓
Small Qwen
↓
Large Qwen
↓
GPT
```

Every stage should maximize:

```text
Exit Rate
```

while preserving required quality.

Conceptual example:

```text
100% raw windows
↓
cheap CV removes obvious irrelevant cases
↓
small model handles easy cases
↓
large Qwen handles hard cases
↓
GPT handles selected difficult / research cases
```

---

# 25. Domain-Specialized Student Models

The domain is relatively constrained:

- robotic manipulation
- fixed or semi-fixed camera layouts
- limited action vocabulary
- limited object vocabulary
- repeated hand-object interactions

Long-term research should evaluate:

```text
GPT / Large Qwen
↓
Teacher Labels
↓
Domain Dataset
↓
Specialized Student Model
```

Potential specialists:

- QA Specialist
- Event Proposal Specialist
- Hand-Object Specialist
- Boundary Specialist

Goal:

cheap specialists process most traffic;
Qwen/GPT handle the tail.

---

# 26. Video Analytics Front-End

After MCAP is decoded to MP4, benchmark a standard video processing stack.

Avoid assuming ordinary Python frame loops are sufficient.

Research:

```text
MP4 streams
↓
hardware decode
↓
stream muxing / batching
↓
cheap analysis
↓
sampling / ROI
↓
VLM
```

Possible technologies to benchmark:

- FFmpeg
- GStreamer
- NVIDIA DeepStream
- NVDEC
- DALI

Do not adopt any without profiling.

---

# 27. Hardware Video Decode

Potential preferred path:

```text
Compressed MP4
↓
NVDEC
↓
GPU Frame
↓
Resize / Crop / Sample
↓
Vision Encoder
```

Avoid repeated:

```text
MP4
→ CPU decode
→ NumPy
→ JPEG
→ disk
→ base64
→ GPU
```

when large-scale profiling shows these copies are expensive.

---

# 28. Dynamic / Continuous Batching

The inference server should support dynamic or continuous batching.

Do not process one request at a time.

Concept:

```text
Request A
Request B
Request C
Request D
↓
Dynamic Batch
↓
GPU
```

For VLM workloads, fixed request count is insufficient.

---

# 29. Visual Budget-Aware Batch Scheduling

Do not define batches only as:

```text
batch_size = N requests
```

Different requests have different costs.

Estimate workload using:

```text
total frames
+
total pixels
+
visual tokens
+
text tokens
```

Then perform approximate bin packing.

Goal:

```text
Σ estimated visual compute
<= batch budget
```

Avoid:

- GPU underutilization from tiny batches
- OOM from many large multimodal requests

---

# 30. Data Parallel First

If the selected Qwen model fits efficiently on one GPU or one small GPU group, prefer scaling throughput with data parallel replicas.

Example:

```text
GPU 1 → Batch A
GPU 2 → Batch B
GPU 3 → Batch C
GPU 4 → Batch D
```

rather than automatically using large Tensor Parallel groups.

The project is throughput-oriented, not ultra-low-latency single-request serving.

---

# 31. Vision Encoder / LLM Disaggregation

Research whether VLM serving can separate:

```text
Vision Encoder
```

from:

```text
LLM Reasoner
```

Possible architecture:

```text
Video Frames
↓
Vision Encoder Fleet
↓
Vision Embeddings
↓
LLM Fleet
↓
Short Structured JSON
```

This may be especially valuable because the workload likely has:

```text
heavy visual input
+
short textual output
```

Do not assume this is automatically better; benchmark.

---

# 32. Encoder / Feature Reuse

This is the caching-related part.

If the same TemporalVisualPackage is used for:

- QA
- action reasoning
- boundary refinement
- value scoring

avoid recomputing identical visual features where technically feasible.

Concept:

```text
TemporalVisualPackage
↓
Vision Encoder
↓
Visual Embedding
├── QA
├── Action
├── Boundary
└── Value
```

Research:

- preprocessing cache
- multimodal encoder cache
- feature reuse
- shared embeddings

Caching is one optimization dimension, not the whole architecture.

---

# 33. FP8 / Quantization

Benchmark:

```text
BF16
vs
FP8
```

and other supported precisions.

Measure:

- throughput
- VRAM
- QA recall
- action accuracy
- temporal boundary error
- cost per processed video hour

Do not assume quantization is acceptable without task-specific quality testing.

---

# 34. Prefix Caching

Fixed prompt components such as:

- system prompt
- output schema
- QA instructions

may benefit from prefix caching.

This is secondary.

Visual computation is expected to dominate.

---

# 35. Speculative Decoding

Low priority for this workload.

Outputs are expected to be short structured responses.

The dominant cost is likely:

- vision encoding
- multimodal prefill

rather than long autoregressive decoding.

Do not prioritize speculative decoding until profiling shows decode is material.

---

# 36. Semantic Video Compression Research

Do not assume frame sampling is the final architecture.

Long-term research should also investigate:

```text
Long Video
↓
Lightweight Video Encoder
↓
Temporal / Spatial Compression
↓
Compressed Visual Tokens
↓
VLM Reasoner
```

Research directions:

- keyframe selection
- temporal token compression
- multi-frame fusion
- query-aware video compression
- visual token pruning

Goal:

retain temporal semantics with fewer tokens than naive frame sampling.

---

# 37. Quality-Aware and Deadline-Aware Scheduling

This is an offline throughput system.

Target:

```text
T-day data
→ QA by T+1 EOD
```

Therefore the scheduler should optimize:

```text
Throughput
+
Deadline Compliance
+
Quality Floor
```

not minimum per-request latency.

Each job may include:

- deadline
- priority
- quality target
- estimated cost
- current backlog impact

---

# 38. SLA-Aware Load Shedding

When backlog grows:

```text
1. pause GPT Shadow
2. pause noncritical Deep Processing
3. prioritize QA
4. reduce research jobs
5. preserve minimum QA quality floor
```

Do not allow research traffic to threaten the production SLA.

---

# 39. Queue Architecture

Use queue-based stages.

Suggested structure:

```text
MP4 Queue
↓
Decode Queue
↓
Sampling / Activity Analysis Queue
↓
TemporalWindow Queue
↓
QA Queue
↓
Event Proposal Queue
↓
Qwen Inference Queue
↓
Fusion Queue
↓
Storage Queue
```

Shadow:

```text
TemporalVisualPackage
↓
Async GPT Shadow Queue
↓
Evaluation Storage
```

Each stage should have independent concurrency and backpressure.

---

# 40. Failure Recovery

Every processing unit should support:

- retry
- checkpoint
- resume
- idempotency

Track:

- processing stage
- completed windows
- failed windows
- pending windows

A partially processed MCAP should not restart from zero unless necessary.

---

# 41. Structured Storage

The system must preserve explicit relationships:

```text
MCAP
→ Camera
→ MP4
→ TemporalWindow
→ CameraQAResult
→ MCAPQAResult
→ CandidateEvent
→ ActionEvent
→ ModelInference
→ DataValueScore
→ DeepVisualResult
→ Embedding
```

Do not create large amounts of disconnected JSON files without stable IDs and traceability.

---

# 42. Data Value Scoring

After ActionEvents are generated, calculate a DataValueScore.

Potential factors:

- QA quality
- action clarity
- hand visibility
- object visibility
- multi-camera coverage
- annotation confidence
- rare actions
- complete action sequences
- scene diversity
- model disagreement

Only selected high-value data should enter expensive deep processing.

---

# 43. Deep Visual Processing

For selected high-value data:

- hand keypoints
- hand skeleton
- hand pose
- hand trajectory
- object trajectory
- hand-object geometric relationships

Do not assume all raw video should receive these expensive operations.

---

# 44. Retrieval

Final retrieval target:

```text
Find all clips where the right hand grasps a cup.
```

Preferred process:

```text
Structured Metadata Filter
+
Temporal Metadata
+
Optional Multimodal Embedding
```

Return:

- ActionEvent
- source MCAP
- camera IDs
- start time
- end time
- action metadata
- synchronized clips if requested

Do not rely entirely on vector search.

---

# 45. Action Clip Extraction

For any ActionEvent:

```text
event_id
↓
mcap_id
↓
start_time / end_time
↓
camera streams
↓
clip extraction
```

Support:

- single-camera clips
- multi-camera synchronized clip packages

---

# 46. Throughput Metrics

Every major stage must measure:

- input video hours
- recording hours
- camera-video hours
- wall-clock processing time
- video-hours/hour
- camera-video-hours/GPU-hour
- API requests
- average latency
- P95 latency
- failure rate
- retry rate
- frames sent
- total pixels
- visual token estimate
- Qwen cost
- GPT shadow cost
- GPU utilization
- CPU utilization
- decode utilization
- queue backlog

---

# 47. Core Capacity Model

Always calculate:

```text
Incoming Rate
vs
Processing Rate
```

If:

```text
incoming_rate > processing_rate
```

report:

- backlog growth rate
- required capacity
- required GPU count
- required optimization factor

Do not claim production readiness without sustained benchmark evidence.

---

# 48. Core Benchmark Matrix

## 48.1 QA Sampling

Test:

- 0.2 FPS
- 0.5 FPS
- 1 FPS
- 2 FPS
- 5 FPS

Measure:

- QA recall
- QA precision
- throughput
- cost

---

## 48.2 Visual Budget

Test combinations of:

- FPS
- resolution
- total pixels
- ROI size

Measure:

- QA recall
- action accuracy
- boundary error
- cost

---

## 48.3 Event Sampling

Test different temporal densities.

Measure:

- event recall
- false negatives
- throughput

---

## 48.4 Boundary Refinement

Test different dense sampling rates.

Measure:

```text
Temporal Boundary Error
```

---

## 48.5 Camera Ablation

Test:

- 1 camera
- 2 cameras
- 3 cameras
- 6 cameras

Measure:

- QA
- action recognition
- boundary accuracy
- cost

Production remains native 6-camera.

---

## 48.6 Progressive Camera Evidence

Compare:

```text
always all 6 cameras
```

vs:

```text
progressive evidence loading
```

Measure quality and compute reduction.

---

## 48.7 Model Comparison

Compare:

- small Qwen
- large Qwen
- GPT shadow

Metrics:

- QA accuracy
- action accuracy
- temporal boundary accuracy
- object identification
- hand identification
- multi-view reasoning
- structured output stability
- latency
- effective cost

---

## 48.8 GPT Rescue Rate

Measure:

> Among Qwen errors, what percentage does GPT correctly fix?

---

## 48.9 Disagreement Precision

When:

```text
Qwen != GPT
```

measure:

- Qwen correct rate
- GPT correct rate

This may become a routing signal.

---

## 48.10 GPU / Precision Matrix

Benchmark:

```text
Qwen model
×
GPU type
×
precision
×
sampling strategy
×
batch strategy
```

Examples:

- Qwen small + L40S
- Qwen small + A100
- Qwen small + H100
- Qwen larger + H100
- BF16
- FP8

Final metric:

```text
$/processed-video-hour
```

and:

```text
$/accepted-video-hour
```

---

# 49. Research Tracks

Do not lock the project into only one optimization path.

Maintain at least these parallel research tracks.

## Track A — Adaptive Temporal Compute

Goal:

video changes more
→ compute more

video changes less
→ compute less

---

## Track B — Hierarchical Model Cascade

Goal:

```text
Cheap CV
→ Specialist
→ Small Qwen
→ Large Qwen
→ GPT
```

---

## Track C — Multi-Camera Compute Reduction

Goal:

preserve 6-camera coverage while dynamically selecting useful evidence.

---

## Track D — Quality / Deadline-Aware Scheduling

Goal:

use T+1 SLA as an optimization advantage.

---

## Track E — Semantic Video Compression

Goal:

move beyond discrete frame sampling if long-video token compression proves superior.

---

## Track F — Encoder / Feature Reuse

Goal:

avoid recomputing the same visual representation.

---

## Track G — Serving Optimization

Goal:

maximize utilization using:

- dynamic batching
- visual-budget bin packing
- data parallel
- encoder disaggregation
- FP8
- autoscaling

---

# 50. Priority Ranking

Recommended current priority:

```text
1. Adaptive / Sparse-to-Dense Sampling
2. Visual Pixel / Token Budget
3. ROI / Resolution Adaptation
4. Dynamic / Continuous Batching
5. Visual Budget-Aware Batch Scheduling
6. Data Parallel Replicas
7. FP8 Benchmark
8. Hardware Video Decode
9. Progressive Camera Evidence
10. Model Cascade
11. Vision Encoder Disaggregation
12. Multimodal Encoder Cache / Feature Reuse
13. Chunked Prefill
14. Prompt Prefix Cache
15. GPUDirect Storage
16. Prefill/Decode Disaggregation
17. Speculative Decoding
```

This ranking is provisional and must be updated based on profiling.

---

# 51. Agent Execution Order

## Phase 1 — Architecture Baseline

Produce Architecture Design V1 containing:

1. MCAP → MP4 ingestion
2. 6-camera data model
3. timestamp alignment
4. TemporalWindow
5. TemporalVisualPackage
6. VisionModelAdapter
7. Qwen primary path
8. GPT shadow path
9. QA pipeline
10. event proposal
11. ActionEvent
12. multi-view fusion
13. structured storage
14. queue architecture
15. failure recovery
16. throughput measurement

Do not perform large-scale implementation before interfaces are clear.

---

## Phase 2 — Baseline Production Pipeline

Implement:

```text
MCAP
→ 6 MP4
→ Timestamp Alignment
→ Uniform Sampling
→ Qwen
→ 6-Camera QA
→ Structured Output
```

Benchmark baseline throughput.

---

## Phase 3 — Event and Action Pipeline

Implement:

```text
Event Proposal
→ CandidateEventWindow
→ Dense Sampling
→ Action Understanding
→ Boundary Refinement
→ Multi-View Fusion
→ ActionEvent
```

---

## Phase 4 — GPT Shadow

Implement:

- GPTAdapter
- async shadow queue
- shadow_sample_ratio
- hard-case routing
- disagreement storage

---

## Phase 5 — Adaptive Compute

Implement and benchmark:

- motion-driven sampling
- activity maps
- adaptive FPS
- ROI
- resolution scaling
- total pixel budget

---

## Phase 6 — Serving Throughput

Benchmark:

- dynamic batching
- visual-budget batching
- data parallel
- FP8
- hardware decode

---

## Phase 7 — Progressive Camera Evidence

Benchmark:

```text
all-6-always
vs
progressive camera loading
```

---

## Phase 8 — Model Cascade

Evaluate:

```text
Cheap CV
→ Specialist
→ Small Qwen
→ Large Qwen
→ GPT
```

---

## Phase 9 — Advanced Research

Evaluate only after baseline profiling:

- vision encoder disaggregation
- encoder cache
- feature reuse
- semantic video compression
- long-video token compression

---

# 52. Required Phase Report

Every phase must report:

1. implemented components
2. input
3. output
4. schema
5. architecture changes
6. throughput
7. latency
8. GPU / CPU / memory usage
9. API usage
10. failure rate
11. current bottleneck
12. quality metrics
13. cost per processed video hour
14. capacity against 500h/day target
15. open questions
16. next experiment

Do not report only:

```text
Done
```

---

# 53. Critical Non-Assumptions

Do not assume:

1. higher FPS is always better
2. all six cameras are equally useful for every event
3. every expensive model request needs all six cameras
4. Qwen is always better than GPT
5. GPT is always better than Qwen
6. all data deserves deep processing
7. embedding alone solves retrieval
8. single-frame vision solves temporal action understanding
9. model confidence equals true accuracy
10. H100 is automatically the cheapest production GPU
11. fixed batch size is sufficient for multimodal workloads
12. caching is the main optimization
13. one model should process all workload classes
14. production can meet SLA before sustained benchmark proof

---

# 54. Final Optimization Framework

Think about large-scale throughput across three compute principles:

```text
A. Compute Elimination
   - filtering
   - adaptive sampling
   - event-driven analysis
   - camera selection
   - model cascade

B. Cheap Compute
   - small models
   - lower resolution
   - ROI
   - quantization
   - visual budget control

C. Compute Reuse
   - preprocessing cache
   - encoder cache
   - feature reuse
```

Then improve the unavoidable compute using:

```text
Batching
+
Parallelism
+
Scheduling
+
Autoscaling
```

The optimization dimensions are multiplicative, not merely additive.

The strategic objective is not:

> process one Qwen request slightly faster

The strategic objective is:

> minimize the total amount of expensive model compute required per successfully processed raw video hour.

---

# 55. Final Target Architecture

```text
                         MCAP
                          │
                          ▼
                    Decode to MP4
                          │
                          ▼
                  6-Camera Alignment
                          │
                          ▼
              Video Analytics Front-End
                 /        |        \
        Motion/Change   Cheap QA   ROI/Event Probe
                 \        |        /
                          ▼
                    Visual Budgeter
              FPS + Resolution + ROI
                          │
                          ▼
                    Visual Job Queue
                          │
                          ▼
              Visual-Budget Batch Scheduler
                          │
                          ▼
                  Qwen Primary Fleet
                          │
                 Confidence / Difficulty
                   /                \
                Easy                Hard
                 │                   │
                 ▼                   ▼
               Done             Larger Qwen
                                     │
                                     ▼
                            GPT Shadow / Judge
                                     │
                                     ▼
                         Disagreement / Research
                          │
                          ▼
                    Multi-View Fusion
                          │
                          ▼
                      ActionEvent
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
            QA       Data Value     Retrieval
                          │
                          ▼
                   High-Value Selection
                          │
                          ▼
                   Deep Visual Processing
                          │
                          ▼
                  Structured Storage / Index
```

---

# 56. Current Highest-Value Experiments

The agent should prioritize these experiments before making major architecture commitments:

## Experiment 1

Measure:

```text
FPS × Resolution × Total Pixels
```

against:

```text
QA Recall
Action Accuracy
Boundary Error
```

---

## Experiment 2

Compare:

```text
Fixed Sampling
vs
Adaptive / Motion-Driven Sampling
```

---

## Experiment 3

Compare:

```text
Always 6 Cameras
vs
Progressive Evidence Loading
```

---

## Experiment 4

Compare:

```text
Qwen Small
vs
Qwen Large
vs
GPT Shadow
```

on hard cases.

---

## Experiment 5

Measure:

```text
Qwen BF16
vs
Qwen FP8
```

for quality and throughput.

---

## Experiment 6

Compare:

```text
Fixed Request Count Batching
vs
Visual Budget-Aware Batching
```

---

## Experiment 7

Measure:

```text
camera-video-hours / GPU-hour
```

on candidate GPU types.

---

## Experiment 8

Measure whether vision encoder disaggregation and encoder reuse improve throughput enough to justify added complexity.

---

# 57. Success Criteria

A successful system must be able to demonstrate:

1. Full traceability:
   ```text
   MCAP → MP4 → Camera → Timestamp → ActionEvent
   ```

2. Native 6-camera support.

3. Qwen primary inference with non-blocking GPT shadow.

4. Measured quality-throughput tradeoffs.

5. Sustained throughput compatible with the real 500h/day definition.

6. Controlled backlog under T+1 QA SLA.

7. Action-level retrieval.

8. Evidence-based decisions about:
   - sampling
   - camera usage
   - model routing
   - GPU type
   - precision
   - batching
   - GPT replacement potential

9. No claim of production readiness without real sustained benchmarks.

---

# 58. Agent Instruction Summary

The agent should treat this document as:

- a requirements baseline
- an architecture research specification
- a benchmark plan
- an execution roadmap

The agent should NOT treat every research direction as mandatory production architecture.

Maintain a strict distinction between:

```text
Fixed Requirements
```

and:

```text
Experimental Hypotheses
```

Fixed requirements include:

- 6-camera source data
- MCAP decoded to MP4
- native 6-camera support
- Qwen primary path
- GPT shadow / research path
- timestamp traceability
- QA across six cameras
- ActionEvent as a core semantic entity
- T+1 QA target
- approximately 500h/day workload
- structured storage
- action-level retrieval

Experimental hypotheses include:

- adaptive sampling
- progressive camera loading
- model cascades
- encoder disaggregation
- encoder caching
- semantic video compression
- specialized student models
- FP8
- specific GPU choices
- specific serving frameworks

All experimental hypotheses must be validated using benchmark data before becoming fixed architectural decisions.
