from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import robata.benchmark.production_qwen_ambiguity_batch as batch


def _selection(archive: Path) -> dict[str, Any]:
    rows = []
    for recording_id in ("recording-a", "recording-b"):
        rows.append(
            {
                "recording_id": recording_id,
                "window_id": f"{recording_id}-w0000",
                "ordinal": 0,
                "source_interval": {
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                    "status": "WINDOW_CONTEXT_ONLY",
                    "is_action_boundary": False,
                },
                "source_context_is_action_boundary": False,
                "declared_camera_ids": ["cam_01", "cam_02"],
                "source_ref": {
                    "archive_path": str(archive),
                    "archive_member": f"file/{recording_id}.mcap",
                    "source": {
                        "archive_path": str(archive),
                        "archive_member": f"file/{recording_id}.mcap",
                        "source_preflight_status": "PASS",
                        "qa_status": "PENDING",
                    },
                },
                "proposal_diagnostics": [{"top_k": [{"rank": 1, "label_text": "do thing"}]}],
            }
        )
    return {"format": batch.AMBIGUITY_SELECTION_FORMAT, "windows": rows}


def test_dry_run_groups_by_recording_and_does_not_invoke_verifier(
    tmp_path: Path, monkeypatch: Any
) -> None:
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"fixture")
    selection = _selection(archive)

    @contextmanager
    def fake_stage(*args: Any, **kwargs: Any):
        staged = tmp_path / "staged.mcap"
        staged.write_bytes(b"mcap")
        yield staged

    monkeypatch.setattr(batch, "stage_zip_member", fake_stage)
    monkeypatch.setattr(
        batch,
        "build_recording_manifest",
        lambda source, **kwargs: {
            "format": "manifest",
            "source": {"cameras": []},
            "window_policy": {},
            "controls": {},
            "windows": [
                {
                    "window_id": f"{kwargs['recording_id']}-w0000",
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        batch,
        "build_qwen_native_video_plan",
        lambda source, output, **kwargs: {
            "status": "DRY_RUN",
            "output": {"directory": str(output)},
        },
    )

    def should_not_run(_: Any) -> dict[str, Any]:  # pragma: no cover - assertion path
        raise AssertionError("dry-run invoked verifier")

    report = batch.run_production_qwen_ambiguity_batch(
        selection,
        output_directory=tmp_path / "out",
        dry_run=True,
        limit=1,
        verifier_runner=should_not_run,
        allow_unapproved_profile=True,
    )

    assert report["status"] == "DRY_RUN"
    assert report["summary"]["selected_recording_count"] == 1
    assert report["summary"]["planned_recording_count"] == 1
    assert report["summary"]["failed_row_count"] == 0
    assert len(report["windows"]) == 2
    assert all(row["status"] == "PLANNED" for row in report["windows"])
    assert all(
        row["provenance"]["source_context_is_action_boundary"] is False for row in report["windows"]
    )
    assert (tmp_path / "out" / "batch-report.json").is_file()


def test_dry_run_accepts_direct_mcap_without_zip_staging(tmp_path: Path, monkeypatch: Any) -> None:
    """A local diagnostic MCAP uses a no-op staging context only."""

    direct = tmp_path / "sample-medium.mcap"
    direct.write_bytes(b"mcap")
    selection = _selection(tmp_path / "unused.zip")
    row = selection["windows"][0]
    row["source_ref"] = {
        "source_path": str(direct),
        "source": {
            "path": str(direct),
            "source_path": str(direct),
            "source_preflight_status": "PASS",
            "qa_status": "PENDING",
        },
    }

    def should_not_stage_zip(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise AssertionError("direct MCAP unexpectedly entered ZIP staging")

    monkeypatch.setattr(batch, "stage_zip_member", should_not_stage_zip)
    monkeypatch.setattr(
        batch,
        "build_recording_manifest",
        lambda source, **kwargs: {
            "format": "manifest",
            "source": {"cameras": []},
            "window_policy": {},
            "controls": {},
            "windows": [
                {
                    "window_id": f"{kwargs['recording_id']}-w0000",
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        batch,
        "build_qwen_native_video_plan",
        lambda source, output, **kwargs: {
            "status": "DRY_RUN",
            "output": {"directory": str(output)},
        },
    )

    report = batch.run_production_qwen_ambiguity_batch(
        selection,
        output_directory=tmp_path / "out-direct",
        dry_run=True,
        limit=1,
        allow_unapproved_profile=True,
    )

    assert report["status"] == "DRY_RUN"
    assert report["controls"]["source_media_direct"] is True
    assert report["controls"]["source_media_staged"] is False
    recording = report["recordings"][0]
    assert recording["source_mode"] == "DIRECT_MCAP"
    assert recording["source_path"] == str(direct.resolve())
    assert recording["archive_member"] is None
    manifest_path = Path(recording["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["archive_member"] is None
    assert manifest["source"]["source_path"] == str(direct.resolve())


def test_actual_route_preserves_verifier_rows_and_fills_missing_camera(
    tmp_path: Path, monkeypatch: Any
) -> None:
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"fixture")
    selection = _selection(archive)

    @contextmanager
    def fake_stage(*args: Any, **kwargs: Any):
        staged = tmp_path / "staged.mcap"
        staged.write_bytes(b"mcap")
        yield staged

    monkeypatch.setattr(batch, "stage_zip_member", fake_stage)
    monkeypatch.setattr(
        batch,
        "build_recording_manifest",
        lambda source, **kwargs: {
            "format": "manifest",
            "source": {"cameras": []},
            "window_policy": {},
            "controls": {},
            "windows": [
                {
                    "window_id": f"{kwargs['recording_id']}-w0000",
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                }
            ],
        },
    )

    class Materialized:
        def __init__(self) -> None:
            self.manifest = {"status": "MATERIALIZED"}

    monkeypatch.setattr(
        batch,
        "materialize_qwen_native_video_inputs",
        lambda source, output, **kwargs: Materialized(),
    )
    seen: list[Any] = []

    def fake_verifier(args: Any) -> dict[str, Any]:
        seen.append(args)
        return {
            "windows": [
                {
                    "window_id": args.window_id[0],
                    "camera_id": "cam_01",
                    "status": "SUCCEEDED",
                }
            ]
        }

    report = batch.run_production_qwen_ambiguity_batch(
        selection,
        output_directory=tmp_path / "out",
        model_directory=tmp_path / "model",
        dry_run=False,
        limit=1,
        verifier_runner=fake_verifier,
        allow_unapproved_profile=True,
    )

    # The verifier returned only one of the two camera rows.  The batch fills
    # the missing row with a structured failure and must keep the recording
    # retryable rather than claiming a top-level COMPLETE result.
    assert report["status"] == "PARTIAL"
    assert len(seen) == 1
    assert seen[0].window_id == ["recording-a-w0000"]
    assert {row["camera_id"] for row in report["windows"]} == {"cam_01", "cam_02"}
    assert any(row["status"] == "FAILED" for row in report["windows"])
    assert report["summary"]["failed_recording_count"] == 1
    assert report["summary"]["succeeded_recording_count"] == 0
    assert report["controls"]["model_invoked"] is True
    assert (tmp_path / "out" / "verifier" / "recording-a.json").is_file()
    json.loads((tmp_path / "out" / "batch-report.json").read_text(encoding="utf-8"))


def test_native_route_reuses_one_resident_runtime_across_recordings(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The default native path loads Qwen once while staging per recording."""

    archive = tmp_path / "source.zip"
    archive.write_bytes(b"fixture")
    selection = _selection(archive)

    @contextmanager
    def fake_stage(*args: Any, **kwargs: Any):
        staged = tmp_path / f"staged-{kwargs.get('ordinal', 0)}.mcap"
        staged.write_bytes(b"mcap")
        yield staged

    monkeypatch.setattr(batch, "stage_zip_member", fake_stage)
    monkeypatch.setattr(
        batch,
        "build_recording_manifest",
        lambda source, **kwargs: {
            "format": "manifest",
            "source": {"cameras": []},
            "window_policy": {},
            "controls": {},
            "windows": [],
        },
    )

    class Materialized:
        def __init__(self) -> None:
            self.manifest = {"status": "MATERIALIZED"}

    monkeypatch.setattr(
        batch,
        "materialize_qwen_native_video_inputs",
        lambda source, output, **kwargs: Materialized(),
    )

    resident_calls: list[tuple[list[str], Any]] = []

    def legacy_runner(_: Any) -> dict[str, Any]:  # pragma: no cover - should not run
        raise AssertionError("legacy per-recording runner was invoked")

    def resident_runner(args: Any, runtime: Any, *, load_observation: Any = None) -> dict[str, Any]:
        resident_calls.append((list(args.window_id), load_observation))
        return {
            "windows": [
                {
                    "window_id": args.window_id[0],
                    "camera_id": "cam_01",
                    "status": "SUCCEEDED",
                },
                {
                    "window_id": args.window_id[0],
                    "camera_id": "cam_02",
                    "status": "SUCCEEDED",
                },
            ]
        }

    legacy_runner.run_with_runtime = resident_runner
    monkeypatch.setattr(batch, "_native_verifier_run", lambda: legacy_runner)

    runtimes: list[Any] = []

    class FakeResidentRuntime:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.load_calls = 0
            self.close_calls = 0
            runtimes.append(self)

        def load(self) -> Any:
            self.load_calls += 1
            return type("LoadObservation", (), {"load_seconds": 0.1, "gpu_name": "fake"})()

        def close(self) -> None:
            self.close_calls += 1

    import robata.inference.local_hf_runtime as local_runtime

    monkeypatch.setattr(local_runtime, "LocalHuggingFaceVisionRuntime", FakeResidentRuntime)
    report = batch.run_production_qwen_ambiguity_batch(
        selection,
        output_directory=tmp_path / "out",
        model_directory=tmp_path / "model",
        dry_run=False,
        allow_unapproved_profile=True,
    )

    assert report["status"] == "COMPLETE"
    assert report["runtime_scope"] == "resident_batch"
    assert report["controls"]["model_invocation_scope"] == "resident_batch"
    assert report["runtime"]["resident"] is True
    assert report["runtime"]["model_load_count"] == 1
    assert len(resident_calls) == 2
    assert len(runtimes) == 1
    assert runtimes[0].load_calls == 1
    assert runtimes[0].close_calls == 1


def test_explicit_resume_skips_completed_recordings_and_retains_rows(
    tmp_path: Path, monkeypatch: Any
) -> None:
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"fixture")
    selection = _selection(archive)

    @contextmanager
    def fake_stage(*args: Any, **kwargs: Any):
        staged = tmp_path / f"staged-{kwargs.get('ordinal', 0)}.mcap"
        staged.write_bytes(b"mcap")
        yield staged

    monkeypatch.setattr(batch, "stage_zip_member", fake_stage)
    monkeypatch.setattr(
        batch,
        "build_recording_manifest",
        lambda source, **kwargs: {
            "format": "manifest",
            "source": {"cameras": []},
            "window_policy": {},
            "controls": {},
            "windows": [],
        },
    )

    class Materialized:
        def __init__(self) -> None:
            self.manifest = {"status": "MATERIALIZED"}

    monkeypatch.setattr(
        batch,
        "materialize_qwen_native_video_inputs",
        lambda source, output, **kwargs: Materialized(),
    )
    calls: list[str] = []

    def fake_verifier(args: Any) -> dict[str, Any]:
        calls.extend(args.window_id)
        return {
            "windows": [
                {
                    "recording_id": args.window_id[0].split("-w")[0],
                    "window_id": args.window_id[0],
                    "camera_id": camera,
                    "status": "SUCCEEDED",
                }
                for camera in ("cam_01", "cam_02")
            ]
        }

    first = batch.run_production_qwen_ambiguity_batch(
        selection,
        output_directory=tmp_path / "out",
        model_directory=tmp_path / "model",
        verifier_runner=fake_verifier,
        allow_unapproved_profile=True,
    )
    assert first["status"] == "COMPLETE"
    assert len(calls) == 2
    calls.clear()

    resumed = batch.run_production_qwen_ambiguity_batch(
        selection,
        output_directory=tmp_path / "out",
        model_directory=tmp_path / "model",
        verifier_runner=lambda _: (_ for _ in ()).throw(
            AssertionError("completed recordings must not invoke verifier")
        ),
        resume=True,
        allow_unapproved_profile=True,
    )
    assert resumed["status"] == "COMPLETE"
    assert resumed["resume"]["resumed_from_partial"] is True
    assert resumed["summary"]["skipped_recording_count"] == 2
    assert len(resumed["recordings"]) == 2
    assert len(resumed["windows"]) == 4
    assert calls == []


def test_resume_rejects_different_selection_before_invoking_verifier(
    tmp_path: Path, monkeypatch: Any
) -> None:
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"fixture")
    selection = _selection(archive)

    @contextmanager
    def fake_stage(*args: Any, **kwargs: Any):
        staged = tmp_path / "staged.mcap"
        staged.write_bytes(b"mcap")
        yield staged

    monkeypatch.setattr(batch, "stage_zip_member", fake_stage)
    monkeypatch.setattr(
        batch,
        "build_recording_manifest",
        lambda source, **kwargs: {
            "format": "manifest",
            "source": {"cameras": []},
            "window_policy": {},
            "controls": {},
            "windows": [],
        },
    )
    monkeypatch.setattr(
        batch,
        "build_qwen_native_video_plan",
        lambda source, output, **kwargs: {"status": "DRY_RUN"},
    )
    batch.run_production_qwen_ambiguity_batch(
        selection,
        output_directory=tmp_path / "out",
        dry_run=True,
        allow_unapproved_profile=True,
    )
    changed = json.loads(json.dumps(selection))
    changed["windows"][0]["window_id"] = "recording-a-different-window"
    try:
        batch.run_production_qwen_ambiguity_batch(
            changed,
            output_directory=tmp_path / "out",
            dry_run=True,
            resume=True,
            allow_unapproved_profile=True,
        )
    except batch.ProductionQwenAmbiguityBatchError as exc:
        assert "selection differs" in str(exc)
    else:  # pragma: no cover - assertion path
        raise AssertionError("selection mismatch must fail explicitly")


def test_resume_requires_a_partial_checkpoint(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"fixture")
    with pytest.raises(
        batch.ProductionQwenAmbiguityBatchError, match="partial batch report is missing"
    ):
        batch.run_production_qwen_ambiguity_batch(
            _selection(archive),
            output_directory=tmp_path / "out",
            dry_run=True,
            resume=True,
            allow_unapproved_profile=True,
        )
