# Technical References and Governance

This directory documents the academic foundations, industry standards, enterprise
practices, and architectural patterns that inform the Robata system design.

## Contents

| File | Topic |
|---|---|
| [architecture-patterns.md](architecture-patterns.md) | Hexagonal architecture, Event Sourcing, CQRS, Saga, Outbox |
| [distributed-systems.md](distributed-systems.md) | Consensus, leases, fencing tokens, CAS |
| [vision-language-models.md](vision-language-models.md) | VLM survey, multi-view geometry, adaptive sampling |
| [standards.md](standards.md) | RFC 8785, IEEE 754, ISO/IETF normative references |
| [enterprise-governance.md](enterprise-governance.md) | Netflix, Google, Uber, Stripe engineering practices |

## Reading Guide

The normative authority for this project is **Architecture V1.1 Section 25**.
The documents in this directory are explanatory and trace how each design
decision relates to published literature or industry evidence.

No citation here overrides a Section 25 rule or an accepted ADR. When a
citation and the architecture disagree, the architecture governs and the
discrepancy should be noted in a new ADR.
