# Web Product

## Scope and path anchors
- UI source: `web/src/**`
- Project tooling: `web/package.json`, `web/vite.config.ts`
- The web package is independently buildable and consumes product-facing payloads.

## How to dispatch
`web-product / P<n> - <view, workflow, visualization, fixture, or client-contract task>`

## Construction phases
1. **Product shell** - routes, views, local fixtures, and visualization primitives.
2. **Contract consumption** - render stable output, review, and identity-bearing fields.
3. **Workflow polish** - connect user actions to a real service when one is available.
4. **Build evidence** - keep lint/build healthy and record fixture versus live-service mode.

## Relevant tests
- Fast: `npm --prefix web run lint`
- Broader: `npm --prefix web run build`

## Read alongside
Read `contract-governance` before changing a consumed payload and `identity-delivery` for output/review meaning. Do not import Python implementation details into the client.
