# Robata Agent Guide

`governance/` is a local navigation aid for developing this repository. It is not a
task service, approval workflow, or upload requirement.

## Design a Blueprint

When an architecture or main agent is asked to design a repository-wide construction
blueprint:

1. Read `governance/BLUEPRINT_TEMPLATE.md` and the relevant module cards.
2. Inspect the actual code and tests needed to make the roadmap concrete.
3. Create or update `governance/BLUEPRINT.md` from the template.

`BLUEPRINT.md` is an on-demand planning output, not a pre-created requirement. The
template remains unchanged and reusable for every future planning cycle.

## Start a Dispatched Phase

A normal dispatch uses this form:

```text
<module-id> / P<n> - <phase name>
```

For example: `source-media / P2 - parallel camera decoding`.

1. If `governance/BLUEPRINT.md` exists and covers the phase, read its relevant section.
2. Read `governance/modules/<module-id>.md` and the phase's named paths.
3. If the phase names another module, read that module card too.
4. Implement the requested change without expanding into unrelated work.
5. Run the phase-local tests or checks named in the module card.
6. Report changed files, commands run and their results, plus any blocker or external
   dependency that could not be exercised locally.

All of this can be performed locally. rchive/ and any local historical materials are background only; they cannot define or
override a product contract. Detailed requirements, reports, and historical
materials may be useful context when present, but they are not required for a clean
checkout.

## Product Contract Note

Use the following locations for contract decisions: published schemas and `schemas/schema-catalog.json` for wire contracts; tracked source, tests, and conformance fixtures for executable behavior. `governance/` is navigation only.

Published schemas are immutable and must use the registered schema workflow rather
than in-place edits. A change to an identity or hash, logical key, idempotency key,
fence, semantic projection, or wire shape needs an explicit version or migration
decision before implementation.
