# Robata Construction Blueprint Template

Use this template when an architecture or main agent needs to turn a product goal into
an executable, repository-wide development plan. This file is the permanent reference:
do not overwrite it. When a concrete roadmap is requested, the architecture agent creates
or updates `governance/BLUEPRINT.md` from this template.

This is a **local planning aid**, not a task tracker, approval system, or remote
service. Its job is to let a new window quickly answer: *which module and phase should
I implement, which files matter, and how do I prove it locally?*

## How to Build the Blueprint

1. State the product outcome and the measurable constraint it serves.
2. Split the work into outcome-oriented phases, not into every individual file edit.
3. Assign each phase to one or more of the Robata modules below.
4. For every phase, name the important paths, the local proof, and the next dependency.
5. Mark genuine external dependencies separately so mockable internal work can proceed.

Keep a phase small enough that one main agent can finish it in one focused development
session. A phase may involve two or more modules when the result only makes sense
end-to-end; say so directly rather than inventing a separate process.

## Robata Module Map

Use these stable module IDs in the roadmap and dispatch prompts.

| Module | Use it for |
| --- | --- |
| `contract-governance` | Wire contracts, schemas, release policy, and storage boundaries |
| `source-media` | MCAP/video ingestion, alignment, decoding, and media artifacts |
| `sampling-qa` | Adaptive sampling, QA evidence, and 21-class product projection |
| `event-semantics` | Event candidates, evidence, proposals, and boundary semantics |
| `inference-evidence` | Provider-neutral calls, evidence ledger, replay, and adapters |
| `stream-control` | Durable work, queues, retries, barriers, leases, and scheduling |
| `identity-delivery` | Logical identity, completion, outbox delivery, admission, and review |
| `canonical-integration` | Canonical state progression, composition, and end-to-end execution |
| `qualification-ops` | Profiles, capacity evidence, benchmarks, and operational tooling |
| `web-product` | React/Vite product UI and client-side workflow |

Read the relevant file under `governance/modules/` before starting a phase. It is a
shortcut to the module's entry points and focused tests; it is not a permission gate.

---

# <Program or Delivery Cycle Name>

## Product Outcome

**Outcome:** <What will a user or operator be able to do when this plan is complete?>

**Why now:** <Quality, throughput, reliability, product, or delivery reason.>

**Success measures:**

- <Measured quality or correctness result>
- <Measured throughput / latency / resource result>
- <Concrete local evidence expected before external validation>

**Non-goals for this cycle:**

- <What intentionally remains outside this plan>

## Overall Roadmap

Use one row per meaningful engineering phase. Leave a module out until there is real
work for it; not every roadmap needs all ten modules.

| Phase | Outcome | Main module(s) | Depends on | Local proof | External follow-up |
| --- | --- | --- | --- | --- | --- |
| P1 - <name> | <user-visible or architectural result> | `<module-id>` | <earlier phase / none> | <test, replay, smoke, or profile> | <none or dependency> |
| P2 - <name> |  |  |  |  |  |
| P3 - <name> |  |  |  |  |  |

## Module Phases

Create one section for each module that has work in this cycle. Prefer a direct,
implementable description over a large backlog.

### `<module-id>` - P<id>: <phase name>

**Result**

<One or two sentences describing the finished capability.>

**Primary paths and entry points**

- `<path/to/entry.py>` - <why it matters>
- `<path/to/test.py>` - <what it proves>

**Implementation outline**

1. <First coherent change>
2. <Second coherent change>
3. <Any compatibility or migration action>

**Keep intact**

- <Existing invariant, schema, replay property, or performance characteristic>

**Done when**

- [ ] <Observable behavior is implemented>
- [ ] <Focused test or fixture proves it>
- [ ] <Any relevant profile/benchmark/replay is recorded>

**Run locally**

```powershell
<smallest useful test command>
<affected lint/type command, if useful>
```

**Next boundary**

<Which module, phase, or end-to-end scenario consumes this result. Write `None` when
the phase is self-contained.>

## Cross-Module Phases

Use this only when a capability genuinely spans modules. It is still ordinary
development work: one main agent may read and modify the named areas, then prove the
combined behavior locally.

### P<id> - <cross-module capability>

**Participating modules:** `<module-a>`, `<module-b>`[, `<module-c>`]

**End-to-end result:** <What becomes possible only after the pieces work together?>

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `<module-a>` | <change> | `<path>` | <test> |
| `<module-b>` | <change> | `<path>` | <test> |

**Compatibility notes:** <Schema, identity, replay, timing, or artifact constraints.
Say `None` when there are none.>

**Combined proof**

```powershell
<small integration / smoke / replay / profile command>
```

## Blockers and External Dependencies

Only list conditions that cannot be settled from the repository and local fixtures.
Do not label ordinary implementation work as a blocker.

| Condition | What can still be completed locally | Temporary substitute | Later external proof |
| --- | --- | --- | --- |
| Real model/GPU endpoint | <adapter, request mapping, replay, timeout behavior> | Deterministic mock or local fixture | Provider integration and quality evaluation |
| Production storage or broker | <interface, SQLite/local recovery behavior> | Local durable adapter | Load, failure, and reconciliation test |
| Governed labels or representative video | <pipeline correctness and synthetic/fixture regression> | Versioned local fixture | Benchmark and quality review |
| Capacity / long-run hardware | <profile harness and bottleneck attribution> | Small local profile | Representative soak and capacity run |

## Acceptance and Verification

The architecture agent should name the smallest proof that makes each claim credible.
Use the following checklist only when relevant; do not turn it into a ceremony.

- [ ] Focused unit tests cover changed decision logic.
- [ ] A focused integration, replay, or smoke test covers changed module boundaries.
- [ ] Schema, wire shape, logical identity, or exact-byte rules are checked when changed.
- [ ] A timing/profile result is included when the phase claims throughput improvement.
- [ ] External limits are stated plainly instead of being presented as completed work.

## Suggested Dispatch Prompt

Use this compact prompt when opening a development window:

```text
Work on <module-id> / P<id> - <phase name>.

Read AGENTS.md, governance/BLUEPRINT.md when it exists, and governance/modules/<module-id>.md.
Goal: <one-sentence outcome>.
Primary paths: <paths>.
Do: <short implementation outline>.
Preserve: <important invariant or compatibility rule>.
Run locally: <test command(s)>.
Report: changed files, command results, measured result if applicable, and any real
external blocker. Do not expand the task beyond this phase without reporting it.
```

For a cross-module phase, list each module and its paths in the same prompt. There is
no need to create a separate workflow: the phase itself is the coordination note.
