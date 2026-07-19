# QA Policy Input v0

- Status: Source summary; not an approved production policy
- Date: 2026-07-18
- Architecture authority: `ARCHITECTURE_DESIGN_V1.md`, Section 25
- Related open decision: O-10 (QA taxonomy, severity, and camera/recording acceptance)

## Purpose and authority

This document converts three local source inputs into reviewable implementation input. It does not promote a QA policy, resolve O-10, or override Architecture V1.1. The 21-item issue list supplies vocabulary, the PDF supplies decision guidance for a subset of that vocabulary, and the annotation principles supply action-labeling guidance. When a source is silent or ambiguous, an implementation must preserve that uncertainty rather than invent a threshold.

The raw inputs are worker-local, ignored by Git, and must not be committed. Only this summary and the verified digests are repository artifacts.

| Source | Local path | SHA-256 | Role |
|---|---|---|---|
| QA specification | `data/source/qa-specification.pdf` | `756d8cd8fa327eb021f84c76dbafbbeb91a3e1ffd86a9fb1fb1dba2baec7e4f0` | Review workflow, severity guidance, and duration thresholds |
| QA issue list | `data/source/QA_issue_list.md` | `d34f27b813c79a2964e5d8efe5816671f53ef9fedfe5f43ed69654655d6ab824` | Canonical source vocabulary of 21 issues |
| Annotation principles | `data/source/annotation-principal.txt` | `2d150a6fc5895bf691f2410af4d8bfa51157d125820a5599550e65c619af1271` | Observable action-label style |

A future registered policy must reference these source digests, its own immutable policy artifact and version, and any adjudication that resolves ambiguities described below.

## Severity semantics

The source uses three severities. Normalized values are:

| Source label | Normalized value | Meaning |
|---|---|---|
| Hint | `INFO` | A limited condition worth recording. It does not reject a camera or recording by itself. |
| Warning | `WARNING` | A material degradation or task-quality concern. Disposition requires the versioned aggregation policy. |
| Error | `ERROR` | The reviewed subject fails the applicable source rule. At camera scope this does not, by itself, reject a six-camera recording. |

Severity is distinct from camera usability and recording acceptance. A model score is also distinct from severity and, under Section 25.1, remains untrusted and uncalibrated unless a registered calibrator or deterministic policy derives a different confidence kind.

## Review workflow

1. Resolve the task metadata and expected task type. Without it, task relevance, completeness, authenticity in context, and SST conformance are `NOT_EVALUATED` or `UNKNOWN`.
2. Scan each planned camera view for recording-wide visual failures. The PDF says a Step 2 `ERROR` ends further Step 3/4 review. Architecture V1.1 narrows that source instruction to the affected view: record durable evidence, short-circuit that view, continue evidence collection for the other planned camera slots, and do not automatically reject the MCAP. The per-view scope, evidence-durability requirement, and six-camera aggregation behavior are architecture constraints, not semantics stated by the PDF.
3. Inspect hand visibility, unrelated/static intervals, task completeness, authenticity, and applicable cross-recording diversity evidence.
4. Compare observable content with the task's SST theme and description. A mismatch is `INFO` plus a note describing the observed action, not an automatic `ERROR`.

### Step 2 fatal conditions

The PDF defines these Step 2 decisions:

| Condition | Decision |
|---|---|
| Fully black view | `ERROR` |
| Glitched/color-corrupted view, including colored lines, ripples, snow/noise, or inter-view color fringing | `ERROR` |
| Hands/gripper unrecognizable for the complete view because it is too dark or overexposed | `ERROR`; a partial interval or region is `WARNING` |
| Revealing outfit as defined by the source privacy rule | `ERROR` |
| Backwards, displaced, inverted, or flipped ego-device view | `ERROR` |
| Hands mostly or completely unrecognizable because of blur | `ERROR`; large-area barely visible blur is `WARNING`; localized blur with overall visibility is `INFO` |
| Large obstruction that completely prevents recognition | `ERROR`; hair/object obstruction affecting recognition is `WARNING`; slight hair with negligible effect is `INFO` |

Any Step 2 `ERROR` short-circuits Step 3/4 for that affected view only after durable issue evidence has been emitted. The PDF supplies the short-circuit instruction; Architecture V1.1 supplies the per-view and evidence-preservation constraints.

## Threshold and task rules

### Hand visibility

- A single hand-obstruction or out-of-frame interval of at most 5 seconds is listed as `INFO`.
- A single interval of at least 5 seconds is listed as `WARNING`.
- Recording-wide or high-frequency obstruction/out-of-frame behavior is `ERROR` when not required by the task.
- Brief scratching or wiping sweat is exempt. Task-natural movement, such as hands moving out of view while mopping, can remain acceptable when the evidence and task metadata support that interpretation.

The source overlaps at exactly 5 seconds (`<= 5` and `>= 5`). A policy must resolve this boundary explicitly before automation. Until then, an exact 5-second observation has no deterministic severity, must produce an explicit policy-ambiguity/`UNKNOWN` outcome, and must not emit a normalized issue with an invented severity.

### Unrelated action and static intervals

- An unrelated action is outside the first and last 5 seconds, has no relation to the task, and has no logical connection to surrounding actions.
- Brief scratching or wiping sweat is exempt.
- The PDF's workflow table lists duration `> 3s` and `< 5s` as `INFO`.
- Its later issue-lookup table instead lists duration `> 3s` and `<= 5s` as `INFO`.
- Both tables list duration `>= 5s` as `WARNING`.
- Recording-wide unrelated behavior is `ERROR`.
- The vocabulary separately defines `CAMERA_STATIONARY_OVER_5S`; its severity and acceptance effect are not specified by the source and remain part of O-10.

The two unrelated-action tables conflict at exactly 5 seconds: one excludes 5 seconds from `INFO`, while the other includes it, and both include it in `WARNING`. This is a source conflict, not a text-extraction artifact. Until a registered policy adjudicates the boundary, an exact 5-second observation has no deterministic severity, must produce an explicit policy-ambiguity/`UNKNOWN` outcome, and must not emit a normalized issue with an invented severity.

### Completeness, authenticity, and SST

- Completing a substantial but incomplete subset of expected steps is `WARNING`; performing only a minimal initial portion is `ERROR`.
- Unnatural or perfunctory execution is `WARNING`; deliberately staged, rigid demonstration is `ERROR`.
- SST theme or task-description mismatch is `INFO` with a concise description of what was actually observed.
- These judgments require immutable task metadata: task/SST identity and version, task name and description, expected steps or key scenes, allowed context, and ontology/policy references. Missing metadata must not be replaced by model assumptions.

### Diversity is cross-recording QA

The source is not fully quantitative: one table associates `almost identical` with greater than 95% similarity, while another says `almost completely identical or greater than 95% similarity`. A policy must quantify or adjudicate the first branch, not silently drop it.

The PDF marks `LACK_OF_DIVERSITY` as `ERROR` for a consecutive sequence of at least three recordings from the same collector when the recordings are described as almost completely identical or have greater than 95% similarity. Evidence considers at least prop style, prop count, capture location, and object placement, with at least one dimension expected to change.

This rule cannot be evaluated inside one MCAP, one camera result, or one recording-only QA job. It requires collector identity, an authoritative recording order, the complete cohort of recording identities, similarity features/scores, and similarity-policy/model versions. The result scope is `CROSS_RECORDING_SEQUENCE`; derived per-recording links may reference the cohort decision but must not pretend it was single-recording evidence.

## Normalized 21-issue vocabulary

The following codes preserve the complete issue list. The listed scope is the narrowest expected evidence scope, not a recording-acceptance decision. A registered policy may refine scope but must not reuse a code for different semantics.

| # | Issue code | Source label | Expected evidence scope |
|---:|---|---|---|
| 1 | `BLACK_SCREEN` | Black Screen | `CAMERA_INTERVAL` or `CAMERA_RECORDING` |
| 2 | `GLITCHED_SCREEN` | Glitched Screen | `CAMERA_INTERVAL` or `CAMERA_RECORDING` |
| 3 | `BLURRY_LENS` | Blurry Lens | `CAMERA_INTERVAL` or `CAMERA_RECORDING` |
| 4 | `EXCESSIVE_SPEED` | Excessive Speed | `CAMERA_INTERVAL` |
| 5 | `EGO_DEVICE_WORN_BACKWARDS` | Ego - Device worn backwards | `CAMERA_RECORDING` |
| 6 | `EGO_HAND_NOT_CENTERED` | Ego - Hand not centered in frame | `CAMERA_INTERVAL` |
| 7 | `CAMERA_STATIONARY_OVER_5S` | Camera stationary for more than 5s | `CAMERA_INTERVAL` |
| 8 | `HAIR_BLOCKING_VIEW` | Hair blocking view | `CAMERA_INTERVAL` |
| 9 | `IRRELEVANT_ACTION_PARTIAL_SEGMENT` | Irrelevant actions in partial segments | `TASK_INTERVAL` |
| 10 | `TASK_IRRELEVANT_ACTION` | Task irrelevant actions | `TASK_INTERVAL` or `TASK_RECORDING` |
| 11 | `ARM_HAND_OBSTRUCTED` | Arm/Hand obstructed | `CAMERA_INTERVAL` |
| 12 | `HAND_OVERLAP_CONTACT_CROSSING` | Hand overlap / contact / crossing | `CAMERA_INTERVAL` |
| 13 | `INCOMPLETE_TASK` | Incomplete task | `TASK_RECORDING` |
| 14 | `LACK_OF_DIVERSITY` | Lack of diversity | `CROSS_RECORDING_SEQUENCE` |
| 15 | `LACK_OF_AUTHENTICITY` | Lack of authenticity | `TASK_INTERVAL` or `TASK_RECORDING` |
| 16 | `VIDEO_ABNORMAL_ENDING` | Video Abnormally Ending | `TASK_RECORDING` |
| 17 | `TOO_DARK_OR_OVEREXPOSED` | Too Dark / Overexposed | `CAMERA_INTERVAL` or `CAMERA_RECORDING` |
| 18 | `UNAUTHORIZED_PERSON_OR_ANIMAL` | Unauthorized Person/Animal Entering Frame | `CAMERA_INTERVAL` |
| 19 | `REVEALING_OUTFIT` | Revealing outfit | `CAMERA_INTERVAL` or `CAMERA_RECORDING` |
| 20 | `PERFORMED_OTHER_EXISTING_TASK` | Performed other existing Tasks | `TASK_RECORDING` |
| 21 | `OTHER` | Other (please specify) | Explicitly declared scope; nonempty note required |

The issue list alone does not define severity thresholds. Codes not covered by the PDF must remain observational or policy-unresolved until O-10 supplies a reviewed mapping.

## Normalized issue record

Every emitted issue must contain, at minimum:

```json
{
  "issue_code": "ARM_HAND_OBSTRUCTED",
  "severity": "INFO|WARNING|ERROR",
  "scope": {
    "kind": "CAMERA_INTERVAL|CAMERA_RECORDING|TASK_INTERVAL|TASK_RECORDING|CROSS_RECORDING_SEQUENCE",
    "subject_refs": ["authoritative immutable references"]
  },
  "evidence_refs": ["immutable package/frame/inference/artifact references"],
  "interval": {"start_ns": "12000000000", "end_ns": "15000000000"},
  "policy_version": "immutable registered policy version"
}
```

- `issue_code` is one of the 21 stable codes above.
- `severity` is required and must be derived by the named policy, not copied from a model score.
- `scope` names both the scope kind and authoritative subjects. Camera evidence includes camera, recording, selected mapping, and alignment references; task evidence also includes task metadata identity/version.
- `evidence_refs` is nonempty. It preserves the observed source material and, where applicable, deterministic or provider-claim lineage.
- `interval` is a canonical half-open, integer-nanosecond interval for temporal issues. It may be null only for a whole-subject or cross-recording decision whose subject references completely define coverage; the null reason must be explicit.
- `policy_version` resolves the taxonomy version, severity rules, thresholds, exemptions, aggregation behavior, and source-input digest bundle. Unknown or ambiguous policy versions fail closed.

## Six-camera aggregation constraint

Production QA remains native six-camera. Every planned camera slot must reach a terminal result, and missing evidence is `UNKNOWN` or `INCOMPLETE`, never clean. A single-camera `ERROR`, including a Step 2 fatal result, must not automatically reject an MCAP. The recording-level policy must aggregate all six camera results, distinguish camera quality from recording/task utility, retain per-camera intervals and evidence, and declare any task-role requirements using task metadata.

O-10 is still required to define camera-to-recording severity mapping, required-role rules, usable/degraded/unusable thresholds, and task-specific acceptance. This source summary is not sufficient to promote an automatic six-camera rejection policy.

## Annotation principles

Action labels should:

1. Use concise action-attributes-object-location-hands form.
2. Use a consistent verb-noun vocabulary for repeated action/object pairs.
3. Use present tense.
4. Describe only observable actions.
5. Avoid assumptions about intent.
6. Identify object interactions explicitly.
7. Prefer one action per segment.
8. Split a segment when the visible action changes.
9. Use short instruction-style labels.

These principles constrain annotation text; they do not allocate persisted identity, establish lineage, or authorize inference about task intent.

## Promotion gaps

Before automated QA promotion, the policy owner must resolve and register:

- O-10 aggregation and acceptance thresholds across all six cameras.
- The exact 5-second hand-visibility boundary and the conflicting exact 5-second unrelated-action boundary.
- Severity rules for issue-list entries not covered by the PDF.
- Required task metadata and behavior when it is absent or contradictory.
- The cross-recording ordering authority, greater-than-95% similarity method, and disposition of the unquantified `almost completely identical` diversity branch.
- Golden examples and adjudicated tests for Step 2 fatal conditions, completeness, authenticity, and SST mismatch.
