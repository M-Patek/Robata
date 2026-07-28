"""Integration coverage for the read-only committed-run explorer API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from robata.application.canonical.local_composition import run_local_canonical_fixture
from robata.web_api.app import create_app

SOURCE_FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical" / "source-recording.json"


@pytest.fixture(scope="module")
def committed_state(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    state_dir = tmp_path_factory.mktemp("web-api-state")
    receipt = run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=state_dir,
        run_key="web-api-integration",
    )
    assert receipt.ok is True
    return state_dir, receipt.run_id


@pytest.fixture
def client(committed_state: tuple[Path, str]) -> TestClient:
    state_dir, _ = committed_state
    return TestClient(create_app(state_dir=state_dir, poll_interval_seconds=0.01))


def test_health_and_run_list_are_backed_by_committed_state(
    client: TestClient,
    committed_state: tuple[Path, str],
) -> None:
    _, run_id = committed_state

    health = client.get("/healthz")
    listed = client.get("/api/v1/runs")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert listed.status_code == 200
    assert listed.json() == {
        "api_version": "v1",
        "runs": [
            {
                "run_id": run_id,
                "recording_identity": listed.json()["runs"][0]["recording_identity"],
                "status": "SUCCEEDED",
                "started_at": "2026-07-20T00:00:00Z",
                "completed_at": "2026-07-20T00:00:00Z",
                "pipeline_version": "canonical-offline-v5",
                "output_decision": "ADMITTED",
                "event_count": 1,
            }
        ],
    }


def test_snapshot_and_websocket_are_json_safe_committed_views(
    client: TestClient,
    committed_state: tuple[Path, str],
) -> None:
    state_dir, run_id = committed_state
    database = state_dir / "primary-completion.sqlite3"
    database_bytes = database.read_bytes()
    sidecars = tuple(database.with_name(f"{database.name}-{suffix}") for suffix in ("wal", "shm"))
    sidecar_bytes = {sidecar.name: sidecar.read_bytes() for sidecar in sidecars if sidecar.exists()}

    response = client.get(f"/api/v1/runs/{run_id}/snapshot")

    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["api_version"] == "v1"
    assert len(snapshot["cursor"]) == 64
    assert snapshot["run"]["run_id"] == run_id
    assert snapshot["run"]["window"]["effective_interval"] == {
        "start_ns": "0",
        "end_ns": "1000005001",
    }
    assert snapshot["run"]["camera_quality"][0]["interval"]["start_ns"] == "0"
    assert snapshot["run"]["decision"] == {
        "decision": "ADMITTED",
        "reason_code": "FUSION_REDUCTION_VALIDATED",
        "admitted_claim_count": 1,
    }
    assert len(snapshot["run"]["hypotheses"]) == 1
    assert len(snapshot["run"]["publications"]) == 1
    assert database.read_bytes() == database_bytes
    assert {
        sidecar.name: sidecar.read_bytes() for sidecar in sidecars if sidecar.exists()
    } == sidecar_bytes

    with client.websocket_connect(f"/ws/v1/runs/{run_id}") as websocket:
        message = websocket.receive_json()

    assert message == {"type": "snapshot", "snapshot": snapshot}


def test_unknown_run_and_missing_database_are_not_silently_simulated(
    client: TestClient,
    tmp_path: Path,
) -> None:
    unknown = client.get("/api/v1/runs/not-a-run/snapshot")
    unavailable = TestClient(create_app(state_dir=tmp_path / "missing")).get("/healthz")

    assert unknown.status_code == 404
    assert unavailable.status_code == 503
