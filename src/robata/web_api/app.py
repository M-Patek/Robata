"""ASGI application exposing committed local runs without worker authority."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status

from robata.web_api.models import (
    HealthResponse,
    RunListResponse,
    RunSnapshotResponse,
    WebSocketSnapshot,
)
from robata.web_api.read_model import (
    LocalStateIntegrityError,
    LocalStateUnavailable,
    ReadOnlyLocalRunProjection,
    RunNotFound,
)


def create_app(
    *,
    state_dir: Path,
    poll_interval_seconds: float = 1.0,
) -> FastAPI:
    """Create a local explorer API backed only by committed completion bytes."""

    if not isinstance(state_dir, Path):
        raise TypeError("state_dir must be pathlib.Path")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")

    projection = ReadOnlyLocalRunProjection(state_dir)
    app = FastAPI(
        title="Robata local run explorer",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        try:
            await asyncio.to_thread(projection.health_check)
        except LocalStateUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return HealthResponse()

    @app.get("/api/v1/runs", response_model=RunListResponse)
    async def list_runs() -> RunListResponse:
        try:
            return await asyncio.to_thread(projection.list_runs)
        except (LocalStateUnavailable, LocalStateIntegrityError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/v1/runs/{run_id}/snapshot", response_model=RunSnapshotResponse)
    async def run_snapshot(run_id: str) -> RunSnapshotResponse:
        try:
            return await asyncio.to_thread(projection.snapshot, run_id)
        except RunNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (LocalStateUnavailable, LocalStateIntegrityError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.websocket("/ws/v1/runs/{run_id}")
    async def run_updates(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        cursor: str | None = None
        try:
            while True:
                try:
                    snapshot = await asyncio.to_thread(projection.snapshot, run_id)
                except RunNotFound:
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return
                except (LocalStateUnavailable, LocalStateIntegrityError):
                    await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                    return
                if snapshot.cursor != cursor:
                    message = WebSocketSnapshot(snapshot=snapshot)
                    await websocket.send_json(message.model_dump(mode="json"))
                    cursor = snapshot.cursor
                await asyncio.sleep(poll_interval_seconds)
        except WebSocketDisconnect:
            return

    return app
