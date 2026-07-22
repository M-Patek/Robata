# Robata Web UI

ComfyUI-inspired visual workflow interface for the Robata canonical pipeline.

## Features

- **Real-time node graph** — Live status updates via WebSocket (or auto-demo mode)
- **Schema-typed edges** — Color-coded connections matching Architecture V1.1 contracts
- **Six-camera 3D view** — Interactive Three.js visualization of multi-view geometry
- **Node inspector** — Detailed metrics, schema versions, and architecture references
- **Run summary** — Stage counts, review queue, evidence class tracking

## Tech Stack

- React + TypeScript + Vite
- **React Flow** — Node graph editor
- **Tailwind CSS** — Styling
- **Three.js + React Three Fiber** — 3D six-camera layout
- **Zustand** — State management

## Quick Start

```bash
# Install dependencies
npm install

# Development server (auto-reloads on changes)
npm run dev

# Production build
npm run build
npm run preview
```

The dev server runs at **http://localhost:5173** by default.

## Architecture Mapping

| UI Component | Python Backend | Section |
|---|---|---|
| `RobataNode` | `CanonicalRunner` stages | 25.10 |
| `SchemaEdge` | Wire contracts (44 registered schemas) | 25.7 |
| `SixCameraPanel` | Multi-view geometry | 25.1 |
| `NodeInspector` | Metrics from `StageMetrics` | — |
| `RunSummaryPanel` | `RobataRun` projection | — |
| WebSocket | Real-time updates (planned: FastAPI backend) | — |

## Demo Mode

If no WebSocket backend is available at `ws://localhost:8000/ws/pipeline`, the UI automatically falls back to a simulated run with realistic stage progression.

To integrate with the Python backend, implement a WebSocket endpoint that emits:

```typescript
// Run initialization
{ type: 'run_update', run: RobataRun }

// Stage status changes
{ type: 'node_status', node_id: string, status: NodeStatus }

// Review queue updates
{ type: 'review_tasks', tasks: ReviewTask[] }
```

## Next Steps

1. **Backend integration** — Add FastAPI WebSocket endpoint to `src/robata/` 
2. **Review UI** — Embed six-video player in review nodes
3. **Video streaming** — Serve MP4 exports with signed URLs
4. **Authentication** — Add reviewer identity and permissions
5. **Metrics dashboard** — Real-time capacity and SLO charts

## Project Structure

```
web/
├── src/
│   ├── nodes/          # Node types (RobataNode)
│   ├── edges/          # Edge types (SchemaEdge)
│   ├── panels/         # Side panels (Inspector, 6-Cam, Summary)
│   ├── hooks/          # WebSocket and other hooks
│   ├── data/           # Mock pipeline graph
│   ├── types.ts        # Domain types
│   ├── store.ts        # Zustand state
│   └── App.tsx
├── public/
├── index.html
└── package.json
```

## Evidence Class

All displayed runs are marked `LOCAL_CONFORMANCE` and `production_eligible: false` per Architecture V1.1 Section 25.11.
