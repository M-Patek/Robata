# Vision-Language Models and Computer Vision References

## 1. Vision-Language Models (VLM) — Survey

### Foundational models

| Model | Organization | Reference |
|---|---|---|
| CLIP | OpenAI | Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision." *ICML*, 2021. |
| Flamingo | DeepMind | Alayrac, J.-B. et al. "Flamingo: a Visual Language Model for Few-Shot Learning." *NeurIPS*, 2022. |
| GPT-4V | OpenAI | OpenAI. "GPT-4 Technical Report." *arXiv:2303.08774*, 2023. |
| Qwen-VL | Alibaba | Bai, J. et al. "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond." *arXiv:2308.12966*, 2023. |
| LLaVA | UW / Microsoft | Liu, H. et al. "Visual Instruction Tuning." *NeurIPS*, 2023. |
| InternVL | Shanghai AI Lab | Chen, Z. et al. "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks." *CVPR*, 2024. |

### Robata Application

- `inference/adapter.py` — `VisionModelAdapter` is a provider-neutral Protocol;
  any of the models above can be plugged in by implementing a single async method.
- `inference/runpod.py` — `RunPodVisionAdapter` wraps a RunPod serverless
  endpoint that may host Qwen-VL, InternVL, or any compatible open-weight VLM.
- `inference/offline_fixture.py` — deterministic fixture exercises the full
  trust boundary without calling a real model.

### Structured output

Robata's `VisionInferenceRequest` includes a pinned `output_schema` (JSON
Schema reference). This aligns with the *constrained generation* literature:

- Willard, B. T. and Louf, R. "Efficient Guided Generation for Large Language
  Models." *arXiv:2307.09702*, 2023.
- Guidance AI. "Guidance: A guidance language for controlling large language
  models." *GitHub*, 2023.

---

## 2. Multi-View Geometry

**References**:
- Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*.
  2nd ed. Cambridge University Press, 2003.
- Longuet-Higgins, H. C. "A computer algorithm for reconstructing a scene from
  two projections." *Nature*, 293, 1981, pp. 133–135.
- Seitz, S. M. et al. "A Comparison and Evaluation of Multi-View Stereo
  Reconstruction Algorithms." *CVPR*, 2006.

### Principle

Multiple cameras observing the same scene from different viewpoints provide
redundant evidence that can compensate for occlusion, poor lighting in one
view, or sensor failure. Cross-view consistency is a strong signal of correctness.

### Robata Application

- Six cameras (`cam_01`–`cam_06`) provide independent observations of each
  workspace event.
- `event_pipeline/evidence.py` — `ActionEvidenceProjector` requires all six
  camera slots to be filled before emitting a result; a single-camera
  occlusion does not propagate as silence.
- `event_pipeline/boundary_refinement.py` — `BoundaryRefinementProjector`
  applies a *median-low-max envelope* reducer across six views to produce a
  single robust boundary estimate.
- `inference/input_plan.py` — `InferenceInputPlan` groups frames by camera
  and enforces provider limits across the six-stream budget.

---

## 3. Temporal Action Detection and Localization

**References**:
- Zhao, Y. et al. "Temporal Action Detection with Structured Segment Networks."
  *ICCV*, 2017.
- Lin, T. et al. "BSN: Boundary Sensitive Network for Temporal Action Proposal
  Generation." *ECCV*, 2018.
- Liu, X. et al. "End-to-End Temporal Action Detection with Transformer."
  *IEEE Transactions on Image Processing*, 2022.

### Principle

Temporal action detection first proposes candidate intervals ("something is
happening here") and then refines the start and end boundaries independently.
The two-stage approach separates coarse detection from fine localization.

### Robata Application

The pipeline mirrors the two-stage paradigm precisely:

1. **Coarse stage**: `EVENT_PROPOSAL` → `CandidateReducer` → `ProvisionalPhysicalActionFuser`
   — produces 0/1/N coarse action intervals.
2. **Fine stage**: `BoundaryRefinementProjector` with independent `ONSET` and
   `OFFSET` windows — refines each boundary separately, consistent with BSN.

The independent ONSET/OFFSET window design prevents a biased boundary estimate
from one role from contaminating the other.

---

## 4. Quality Assessment in Video Understanding

**References**:
- Mittal, A., Moorthy, A. K. and Bovik, A. C. "No-Reference Image Quality
  Assessment in the Spatial Domain." *IEEE Transactions on Image Processing*,
  21(12), 2012.
- Tu, Z. et al. "UGFRN: Unified Generative Framework for Reference-Free
  Video Quality Assessment." *ECCV*, 2022.

### Robata Application

- `qa_pipeline/coarse.py` — `CoarseQAProjector` performs a model-driven
  quality sweep before event detection; low-quality frames are flagged early.
- `qa_pipeline/completion.py` — three-state gate: `QA_COMPLETE` (proceed),
  `DENSE_REQUIRED` (re-examine degraded coordinates), `QA_INCOMPLETE` (abort).
- The QA stage prevents low-quality inputs from producing spurious events, a
  design consistent with the "garbage-in, garbage-out" principle in no-reference
  quality assessment literature.

---

## 5. Adaptive and Active Sampling

**References**:
- Settles, B. "Active Learning Literature Survey." *Computer Sciences Technical
  Report 1648*, University of Wisconsin–Madison, 2009.
- Chapelle, O. and Li, L. "An Empirical Evaluation of Thompson Sampling."
  *NeurIPS*, 2011.
- Li, L. et al. "A Contextual-Bandit Approach to Personalized News Article
  Recommendation." *WWW*, 2010.

### Principle

Rather than processing all frames at a uniform rate, an adaptive sampler
raises the rate when signals of interest are detected and lowers it during
quiescent periods. This is related to active learning (query by uncertainty)
and the multi-armed bandit literature (exploration vs. exploitation trade-off).

### Robata Application

- `sampling/adaptive.py` — `SignalDetector` protocol; implementations will
  detect motion, boundary uncertainty, and QA degradation as triggers.
- `sampling/grid.py` — base rational grid provides the uniform-rate fallback.
- `AdaptiveSampler.sample()` currently raises `NotImplementedError` pending
  O-13 policy; the architecture is in place for plug-in detectors.

---

## 6. Shadow Testing and Online Evaluation

**References**:
- Kohavi, R., Longbotham, R. and Walker, T. "Online Experiments: Lessons
  Learned." *Computer*, 40(9), 2007, pp. 103–105.
- Tang, D. et al. "Overlapping Experiment Infrastructure: More, Better, Faster
  Experimentation." *KDD*, 2010. (Google)
- Sculley, D. et al. "Hidden Technical Debt in Machine Learning Systems."
  *NeurIPS*, 2015.

### Principle

Shadow traffic routes a copy of production requests to a candidate model in
isolation. The candidate's output is compared with the primary model's output
but never served to end users. This avoids the cold-start problem and provides
realistic traffic distribution for evaluation.

### Robata Application

- `inference/shadow.py` — `ShadowRoute` and deterministic routing; the shadow
  model receives the same input but its output is never merged into the primary
  event stream.
- `inference/evaluation.py` — `EvaluationResult` with field-level `FieldDelta`;
  disagreements are append-only evidence, not mutable state.
- `review/models.py` — `ReviewTrigger.DISAGREEMENT` routes high-disagreement
  pairs to the human review queue automatically.

The combination of shadow evaluation + disagreement-triggered review implements
the "human-in-the-loop" safeguard described in Sculley et al.'s ML debt
framework.
