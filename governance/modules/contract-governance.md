# Contract Governance

## Scope and path anchors
- `schemas/**`, `src/robata/contracts/**`, `src/robata/ports/**`
- Schema catalog and registration: `schemas/schema-catalog.json`, `scripts/register_schema.py`
- Experimental storage boundary: `src/robata/storage/**`

## How to dispatch
`contract-governance / P<n> - <schema, contract, port, or storage-boundary task>`

## Construction phases
1. **Schema release** - register immutable wire schemas and verify catalog pins.
2. **Contract models and ports** - add or evolve typed producer/consumer contracts.
3. **Evolution paths** - implement and test compatible upcasters or migrations.
4. **Storage experiments** - keep unfinished adapters fail-closed until they have a real contract.

## Relevant tests
- Fast: `python -m pytest tests/unit/test_register_schema.py tests/unit/test_schema_immutability.py`
- Broader: `python -m pytest tests/contract/test_schema_catalog.py tests/contract/test_schema_release_policy.py tests/contract/test_schema_upcasting.py`

## Read alongside
Read `inference-evidence`, `identity-delivery`, or `web-product` when a changed payload has consumers there. Published schema bytes are product contracts: register a new version rather than edit a released version in place.
