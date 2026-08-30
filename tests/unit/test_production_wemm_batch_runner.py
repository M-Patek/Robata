from __future__ import annotations

import json
import runpy
import zipfile
from pathlib import Path

import pytest

import robata.benchmark.production_wemm_batch_runner as runner
from robata.benchmark.production_wemm_batch_runner import (
    ProductionWemmBatchRunnerError,
    build_recording_manifest,
    load_source_preflight,
    run_production_wemm_batch,
    select_preflight_items,
    stage_zip_member,
)


def _preflight(archive: Path) -> dict[str, object]:
    return {
        "format": "robata-production-source-preflight-v1",
        "status": "SOURCE_PREFLIGHT_COMPLETE",
        "source": {"archive_path": str(archive)},
        "items": [
            {
                "ordinal": 0,
                "name": "file/a.mcap",
                "size_bytes": 5,
                "source_preflight_status": "PASS",
                "batch": "B1_stratified_pilot",
                "qa_status": "PENDING",
            },
            {
                "ordinal": 1,
                "name": "file/b.mcap",
                "size_bytes": 6,
                "source_preflight_status": "FAIL",
                "qa_status": "PENDING",
            },
            {
                "ordinal": 2,
                "name": "file/c.mcap",
                "size_bytes": 7,
                "source_preflight_status": "PASS",
                "batch": "B2_remaining",
                "qa_status": "PENDING",
            },
        ],
    }


def _catalog() -> list[str]:
    return ["move cloth"]


def test_preflight_selects_only_pass_and_supports_batch_filter(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    payload = _preflight(archive)
    loaded = load_source_preflight(payload)
    assert len(loaded["items"]) == 3
    selected = select_preflight_items(payload, batch="B1_stratified_pilot")
    assert [item["archive_member"] for item in selected] == ["file/a.mcap"]
    assert select_preflight_items(payload) == (
        {
            "ordinal": 0,
            "archive_member": "file/a.mcap",
            "size_bytes": 5,
            "source_preflight_status": "PASS",
            "batch": "B1_stratified_pilot",
            "source_preflight_reason": None,
            "qa_status": "PENDING",
            "duration_seconds": None,
            "camera_count": None,
            "camera_frames_total": None,
        },
        {
            "ordinal": 2,
            "archive_member": "file/c.mcap",
            "size_bytes": 7,
            "source_preflight_status": "PASS",
            "batch": "B2_remaining",
            "source_preflight_reason": None,
            "qa_status": "PENDING",
            "duration_seconds": None,
            "camera_count": None,
            "camera_frames_total": None,
        },
    )


def test_clip_level_qa_fail_is_excluded_even_when_source_preflight_passes(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    payload = _preflight(archive)
    payload["items"] = [
        {
            "ordinal": 0,
            "name": "file/a.mcap",
            "source_preflight_status": "PASS",
            "qa_status": "FAIL",
        }
    ]
    assert select_preflight_items(payload) == ()


def test_safe_staging_streams_one_member_and_cleans_up(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("file/a.mcap", b"mcap-bytes")
    staging = tmp_path / "staging"
    with stage_zip_member(archive, "file/a.mcap", staging, ordinal=4) as path:
        assert path.read_bytes() == b"mcap-bytes"
        assert path.is_file()
    assert not (staging / "recording-0004").exists()
    with (
        pytest.raises(ProductionWemmBatchRunnerError, match="unsafe"),
        stage_zip_member(archive, "../escape.mcap", staging),
    ):
        pass


def test_recording_manifest_marks_processing_windows_not_boundaries(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_manifest(source: Path, *, window_seconds: float, include_tail: bool):
        assert source == tmp_path / "a.mcap"
        assert window_seconds == 8.0
        assert include_tail is True
        return {
            "format": "robata-production-shaped-cohort-v1",
            "source": {"path": str(source), "cameras": [], "camera_count": 0},
            "window_policy": {"window_seconds": 8.0},
            "windows": [
                {"window_id": "sample-medium-w00", "start_seconds": 0.0, "end_seconds": 8.0}
            ],
            "controls": {},
        }

    monkeypatch.setattr(runner, "build_manifest", fake_manifest)
    manifest = build_recording_manifest(
        tmp_path / "a.mcap",
        recording_id="production-000-a",
        archive_member="file/a.mcap",
        source_preflight_status="PASS",
    )
    assert manifest["windows"][0]["window_id"] == "production-000-a-w0000"  # type: ignore[index]
    assert manifest["windows"][0]["action_boundary"] is False  # type: ignore[index]
    assert manifest["window_policy"]["window_semantics"] == "PROCESSING_WINDOW_NOT_ACTION_BOUNDARY"  # type: ignore[index]
    assert manifest["controls"]["hash_or_sha_used"] is False  # type: ignore[index]


def test_batch_dry_run_writes_checkpoint_without_staging_or_model(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("file/a.mcap", b"mcap")
    preflight = _preflight(archive)
    preflight["items"] = [preflight["items"][0]]  # type: ignore[index]
    report = run_production_wemm_batch(
        preflight,
        phrase_catalog=_catalog(),
        model_directory=None,
        output_directory=tmp_path / "run",
        dry_run=True,
    )
    assert report["status"] == "DRY_RUN"
    assert report["summary"]["selected_count"] == 1  # type: ignore[index]
    assert report["summary"]["planned_count"] == 1  # type: ignore[index]
    assert report["config"]["window_chunk_size"] == 1  # type: ignore[index]
    assert report["config"]["inference_batch_size"] == 1  # type: ignore[index]
    assert "include_pipeline" not in report["config"]  # type: ignore[operator]
    assert "queue_capacity" not in report["config"]  # type: ignore[operator]
    assert report["controls"]["epic_ontology_used"] is False  # type: ignore[index]
    assert Path(report["checkpoint_path"]).is_file()  # type: ignore[arg-type]


def test_batch_execution_is_serial_and_resumable(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("file/a.mcap", b"mcap-a")
        zf.writestr("file/c.mcap", b"mcap-c")
    preflight = _preflight(archive)
    calls: list[str] = []
    chunk_sizes: list[int] = []
    inference_batch_sizes: list[int] = []
    scheduling_kwargs: list[dict[str, object]] = []

    def fake_manifest(source: Path, **kwargs):
        return {
            "format": "robata-production-shaped-cohort-v1",
            "source": {"path": str(source), "cameras": [], "camera_count": 0},
            "window_policy": {"window_seconds": kwargs["window_seconds"]},
            "windows": [{"window_id": "w00", "start_seconds": 0.0, "end_seconds": 8.0}],
            "controls": {},
        }

    def fake_open(manifest, **kwargs):
        calls.append(str(manifest["source"]["archive_member"]))
        scheduling_kwargs.append(dict(kwargs))
        chunk_sizes.append(int(kwargs["window_chunk_size"]))
        inference_batch_sizes.append(int(kwargs["inference_batch_size"]))
        return {
            "format": "robata-production-wemm-preannotation-v1",
            "production_eligible": False,
            "windows": [],
            "label_space": {"kind": "OPEN_PROVISIONAL_PHRASES"},
        }

    monkeypatch.setattr(runner, "build_manifest", fake_manifest)
    monkeypatch.setattr(runner, "run_production_wemm_open", fake_open)
    monkeypatch.setattr(
        runner,
        "build_review_pack",
        lambda envelope: {"envelope": envelope["format"]},
    )

    # Avoid loading a real checkpoint while still exercising the serial loop.
    class FakeBackend:
        def __init__(self) -> None:
            self.observations: list[object] = []

        def close(self) -> None:
            return None

    monkeypatch.setattr(runner, "WemmEmbeddingBackend", lambda **kwargs: FakeBackend())

    report = run_production_wemm_batch(
        preflight,
        phrase_catalog=_catalog(),
        model_directory=tmp_path / "model",
        output_directory=tmp_path / "run",
        max_windows=1,
        window_chunk_size=2,
        inference_batch_size=4,
    )
    assert report["status"] == "COMPLETE"
    assert report["summary"]["complete_count"] == 2  # type: ignore[index]
    assert calls == ["file/a.mcap", "file/c.mcap"]
    assert chunk_sizes == [2, 2]
    assert inference_batch_sizes == [4, 4]
    assert all("include_pipeline" not in kwargs for kwargs in scheduling_kwargs)
    assert all("queue_capacity" not in kwargs for kwargs in scheduling_kwargs)
    # A resumed run must skip both completed recordings and not call the model.
    calls.clear()
    resumed = run_production_wemm_batch(
        preflight,
        phrase_catalog=_catalog(),
        model_directory=tmp_path / "model",
        output_directory=tmp_path / "run",
        max_windows=1,
    )
    assert resumed["summary"]["skipped_count"] == 2  # type: ignore[index]
    assert resumed["summary"]["complete_count"] == 2  # type: ignore[index]
    assert calls == []
    json.dumps(resumed)


def test_batch_forwards_opt_in_pipeline_options(monkeypatch, tmp_path: Path) -> None:
    """The resumable wrapper forwards scheduling without changing defaults."""

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("file/a.mcap", b"mcap")
    preflight = _preflight(archive)
    preflight["items"] = [preflight["items"][0]]  # type: ignore[index]
    calls: list[dict[str, object]] = []

    def fake_open(manifest, **kwargs):
        del manifest
        calls.append(dict(kwargs))
        return {
            "format": "robata-production-wemm-preannotation-v1",
            "production_eligible": False,
            "windows": [],
            "label_space": {"kind": "OPEN_PROVISIONAL_PHRASES"},
        }

    def fake_manifest(source: Path, **kwargs):
        return {
            "format": "robata-production-shaped-cohort-v1",
            "source": {"path": str(source), "cameras": [], "camera_count": 0},
            "window_policy": {"window_seconds": kwargs["window_seconds"]},
            "windows": [{"window_id": "w00", "start_seconds": 0.0, "end_seconds": 8.0}],
            "controls": {},
        }

    monkeypatch.setattr(runner, "build_manifest", fake_manifest)
    monkeypatch.setattr(runner, "run_production_wemm_open", fake_open)
    monkeypatch.setattr(
        runner,
        "build_review_pack",
        lambda envelope: {"envelope": envelope["format"]},
    )

    class FakeBackend:
        def __init__(self) -> None:
            self.observations: list[object] = []

        def close(self) -> None:
            return None

    monkeypatch.setattr(runner, "WemmEmbeddingBackend", lambda **kwargs: FakeBackend())
    report = run_production_wemm_batch(
        preflight,
        phrase_catalog=_catalog(),
        model_directory=tmp_path / "model",
        output_directory=tmp_path / "run",
        include_pipeline=True,
        queue_capacity=3,
    )
    assert report["status"] == "COMPLETE"
    assert calls and calls[0]["include_pipeline"] is True
    assert calls[0]["queue_capacity"] == 3
    assert report["config"]["include_pipeline"] is True  # type: ignore[index]
    assert report["config"]["queue_capacity"] == 3  # type: ignore[index]


def test_batch_cli_exposes_and_forwards_pipeline_options(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "scripts" / "run_production_wemm_batches.py"
    namespace = runpy.run_path(str(script), run_name="robata_batch_cli")
    parser = namespace["_parser"]()
    defaults = parser.parse_args(
        [
            "--source-preflight",
            "preflight.json",
            "--phrase-catalog",
            "phrases.json",
            "--output-dir",
            "output",
        ]
    )
    assert defaults.pipeline is False
    assert defaults.queue_capacity == 1
    parsed = parser.parse_args(
        [
            "--source-preflight",
            "preflight.json",
            "--phrase-catalog",
            "phrases.json",
            "--output-dir",
            "output",
            "--pipeline",
            "--queue-capacity",
            "4",
        ]
    )
    assert parsed.pipeline is True
    assert parsed.queue_capacity == 4

    calls: list[dict[str, object]] = []

    def fake_batch(preflight, **kwargs):
        calls.append({"preflight": preflight, **kwargs})
        return {
            "status": "DRY_RUN",
            "production_eligible": False,
            "summary": {
                "selected_count": 0,
                "complete_count": 0,
                "failed_count": 0,
                "skipped_count": 0,
                "window_count": 0,
                "estimated_window_count": 0,
            },
            "checkpoint_path": str(tmp_path / "checkpoint.json"),
        }

    namespace["main"].__globals__["run_production_wemm_batch"] = fake_batch
    assert (
        namespace["main"](
            [
                "--source-preflight",
                "preflight.json",
                "--phrase-catalog",
                "phrases.json",
                "--output-dir",
                "output",
                "--pipeline",
                "--queue-capacity",
                "4",
            ]
        )
        == 0
    )
    assert calls and calls[0]["include_pipeline"] is True
    assert calls[0]["queue_capacity"] == 4


@pytest.mark.parametrize("bad", [0, True, 1.5])
def test_batch_rejects_invalid_queue_capacity_before_loading_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: object
) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("file/a.mcap", b"mcap")
    preflight = _preflight(archive)
    preflight["items"] = [preflight["items"][0]]  # type: ignore[index]
    backend_calls = 0

    def fail_backend(**_kwargs: object):
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("backend must not be loaded for invalid configuration")

    monkeypatch.setattr(runner, "WemmEmbeddingBackend", fail_backend)
    with pytest.raises(ProductionWemmBatchRunnerError, match="queue_capacity"):
        run_production_wemm_batch(
            preflight,
            phrase_catalog=_catalog(),
            model_directory=tmp_path / "model",
            output_directory=tmp_path / "run",
            queue_capacity=bad,  # type: ignore[arg-type]
        )
    assert backend_calls == 0


@pytest.mark.parametrize("bad", [0, 65, True, 1.5])
def test_batch_rejects_invalid_inference_batch_size_before_loading_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: object
) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("file/a.mcap", b"mcap")
    preflight = _preflight(archive)
    preflight["items"] = [preflight["items"][0]]  # type: ignore[index]
    backend_calls = 0

    def fail_backend(**_kwargs: object):
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("backend must not be loaded for invalid configuration")

    monkeypatch.setattr(runner, "WemmEmbeddingBackend", fail_backend)
    with pytest.raises(ProductionWemmBatchRunnerError, match="inference_batch_size"):
        run_production_wemm_batch(
            preflight,
            phrase_catalog=_catalog(),
            model_directory=tmp_path / "model",
            output_directory=tmp_path / "run",
            inference_batch_size=bad,  # type: ignore[arg-type]
        )
    assert backend_calls == 0
