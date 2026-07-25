# Robata Construction Map

`governance/` is a small, local Markdown aid for planning or opening an engineering
window in Robata. It helps an agent find relevant modules, code paths, and focused
tests without turning documentation into a process.

## Start here

1. To design a roadmap, read the [blueprint template](BLUEPRINT_TEMPLATE.md).
2. The architecture agent creates `BLUEPRINT.md` only when a concrete roadmap is
   requested; it is not a pre-created file and never replaces the template.
3. To implement a phase, read the applicable file in [modules/](modules/), plus the
   relevant `BLUEPRINT.md` section when that output exists.

These documents are local navigation aids. The implementation, tests, and published
contracts remain where their code lives.
