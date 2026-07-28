# Robata Committed Run Workbench

A read-only operational workbench for committed canonical-run projections. It does
not start pipelines, replay data, or manufacture client-side run state.

## Local Development

Create or reuse local canonical state, then start the read-only API from the
repository root:

```bash
uv sync --locked --extra web
python scripts/run_canonical_fixture.py tests/fixtures/canonical/source-recording.json --state-dir tmp/canonical-state --run-key primary
python scripts/run_web_api.py --state-dir tmp/canonical-state
```

In a second terminal, install the frontend dependencies and run Vite from this
directory:

```bash
npm install
npm run dev
```

Vite serves the UI at `http://localhost:5173` and proxies `/api` and `/ws` to the
local API service at `http://127.0.0.1:8000` during development. The production
build is verified with:

```bash
npm run build
```

## API Contract

The viewer consumes only the versioned read projection exposed by the backend:

- `GET /api/v1/runs`
- `GET /api/v1/runs/{run_id}/snapshot`
- `WS /ws/v1/runs/{run_id}`

The WebSocket sends a `snapshot` envelope when a committed cursor is available
and whenever that cursor changes. The UI first fetches the REST snapshot and
then subscribes to the selected run. A transport failure leaves the committed
snapshot visible and reports the update connection as unavailable.

Intervals and durations stay decimal strings in transport. The UI formats them
with `BigInt`, so nanosecond values retain their exact precision.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_ROBATA_API_BASE` | `/api/v1` | Base URL for the REST projection API. |
| `VITE_ROBATA_WS_BASE` | `/ws/v1` | Base URL for committed-snapshot WebSocket updates. |

Leave both unset for same-origin deployment or Vite proxy development. Absolute
HTTP(S) and WS(S) overrides are supported for a separately hosted API.

For Vite development only, `ROBATA_API_DEV_TARGET` and
`ROBATA_WS_DEV_TARGET` select the proxy upstream. They default to the local
API listener and are not included in the browser bundle.

## Workbench Behavior

- Uses a compact committed-run picker; one run selects automatically, otherwise
  selection remains explicit.
- Organizes true committed intervals in a shared timeline and Source & QA /
  Decision & delivery planes. An interval, stage, object, integrity field, or
  evidence reference opens an on-demand detail drawer.
- Uses only backend responses. There is no mock data, simulation timer, replay,
  control endpoint, in-flight progress, watermark, or backpressure display.
