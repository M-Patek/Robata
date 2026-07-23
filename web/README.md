# Robata Web UI

Streaming-architecture visual interface for the Robata canonical pipeline.

## Features

- **Two-plane layout** — Plane A (Media + Inference) and Plane B (Durable Window DAG)
- **Event-time timeline** — Visual bands for segments, windows, and watermark progression
- **Live simulation** — Mock stream events replay with configurable speed
- **Subject inspection** — Click any subject to see its complete identity (SHA-256, keys, digests)
- **Backpressure monitoring** — Queue depth, oldest age, and pressure class visualization

## Architecture

The frontend reflects the streaming-throughput rearchitecture (WP0–WP7):

| UI Component | Backend Counterpart | Contract |
|---|---|---|
| `CaptureScopeCard` | `PreEosCaptureSubject` | `stream_source.py` |
| `SegmentTimeline` | `StreamSegment` | `stream_source.py` |
| `WindowCard` | `IncrementalWindow` | `stream_window.py` |
| `InferenceCard` | `StreamInference` | `stream_window.py` |
| `ExpectedWindowPlanCard` | `ExpectedWindowPlan` | `stream_planning.py` |
| `TerminalClosureCard` | `WindowTerminalClosure` | `stream_finalization.py` |
| `FinalizationCard` | `RecordingFinalizationMap` | `stream_finalization.py` |
| `TimelineBand` | Event-time / watermark | `stream_common.py` |
| `WatermarkBar` | Backpressure state | `stream_common.py` |

## Tech Stack

- React + TypeScript + Vite
- Zustand — State management
- Tailwind CSS — Styling

## Quick Start

```bash
# Install dependencies
npm install

# Development server
npm run dev

# Production build
npm run build
npm run preview
```

The dev server runs at **http://localhost:5173** by default.

## Mock Data

The demo uses a deterministic mock event stream for a 40.89-second six-camera recording:

- **40 segments** per camera (1-second logical chunks)
- **39 windows** (2s width, 1s hop)
- **Multiple inferences** per window (QA_COARSE, QA_DENSE, EVENT_PROPOSAL, etc.)
- **Expected window plan** with sealed manifest
- **Terminal closure** with reconciled outcomes
- **Recording finalization** mapping

Mock data is generated in `src/data/mock_stream_events.ts` and mirrors the Python contract shapes.

## Two-Plane Model

### Plane A: Media + Inference (Replaceable)

Shows the live execution of one window. This plane is replaceable because the same window identity can be produced by PyAV (local) or DeepStream/Triton (accelerated).

```
CaptureScope
  ├── Segment[0..5] (one per camera)
  ├── Window (purpose, interval, semantic SHA-256)
  ├── Inference (attempt, terminal outcome)
  └── WindowResult (evidence ref)
```

### Plane B: Durable Window DAG (Persistent)

Shows the append-only expected-window plan and its terminal closure. This plane survives restarts and is the authority for recording finalization.

```
ExpectedWindowPlan
  ├── Declarations (appended before child publication)
  ├── Sealed manifest (at EOS)
  └── Terminal closure (reconciled after execution)

RecordingFinalizationMap
  ├── Capture scope → Final source identity
  └── Incremental windows → Recording-scoped identities
```

## Evidence Class

All displayed runs are marked `LOCAL_CONFORMANCE` and `production_eligible: false` per Architecture V1.1 Section 25.11.

## Next Steps

1. **Backend integration** — Connect to FastAPI WebSocket endpoint when WP3–WP6 complete
2. **Real-time updates** — Replace mock simulation with live stream events
3. **Video player** — Embed six-video player in segment detail
4. **Metrics dashboard** — Real-time capacity and SLO charts

## Project Structure

```
web/
├── src/
│   ├── data/
│   │   └── mock_stream_events.ts   # Mock event stream data
│   ├── hooks/
│   │   └── useWebSocket.ts         # WebSocket + simulation hook
│   ├── panels/
│   │   ├── PlaneAView.tsx          # Media + Inference plane
│   │   ├── PlaneBView.tsx          # Durable Window DAG plane
│   │   ├── TimelineBand.tsx        # Event-time timeline
│   │   ├── WatermarkBar.tsx        # Watermark + backpressure
│   │   └── SubjectDetailDrawer.tsx # Subject identity inspector
│   ├── types.ts                    # Domain types (mirrors Python contracts)
│   ├── store.ts                    # Zustand store (StreamViewState)
│   ├── App.tsx                     # Two-plane layout
│   └── main.tsx                    # Entry point
├── dist/                           # Production build
├── index.html
├── package.json
└── README.md
```
