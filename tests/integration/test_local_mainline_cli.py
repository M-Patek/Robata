from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("av")
pytest.importorskip("mcap")
pytest.importorskip("mcap_protobuf")

import scripts.run_local_mainline as cli
from robata.application.mainline import MainlineRunError, MainlineRunErrorCode
from robata.contracts.mainline import RunStatus

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_local_mainline.py"
MEDIUM_SAMPLE = ROOT / "data" / "source" / "sample-medium.mcap"
RUN_REAL_MAINLINE = os.environ.get("ROBATA_RUN_REAL_MAINLINE_ACCEPTANCE") == "1"

pytestmark = pytest.mark.acceptance


def _subprocess_run(*arguments: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_rejects_unapproved_mapping_before_source_access(tmp_path: Path) -> None:
    missing_source = tmp_path / "source-is-never-opened.mcap"
    output = tmp_path / "must-not-exist"

    completed = _subprocess_run(str(missing_source), str(output))

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload == {
        "ok": False,
        "stage": "mapping_authorization",
        "error": {
            "code": "INVALID_CAMERA_MAPPING",
            "message": "mapping profile 'genrobot-observed-v0' is not approved",
        },
        "provider_requests": 0,
    }
    assert not output.exists()
    assert tuple(tmp_path.glob(".must-not-exist.partial-*")) == ()


@pytest.mark.parametrize("rate", ["0", "1/0", "abc", "1/2/3"])
def test_cli_returns_json_for_invalid_rate(rate: str, tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"

    completed = _subprocess_run(
        str(tmp_path / "never-opened.mcap"),
        str(output),
        "--coarse-rate",
        rate,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["stage"] == "arguments"
    assert payload["error"]["code"] == "INVALID_ARGUMENT"
    assert payload["provider_requests"] == 0
    assert not output.exists()
    assert tuple(tmp_path.glob(".must-not-exist.partial-*")) == ()


def test_cli_source_failure_has_stage_and_zero_provider_requests(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"

    completed = _subprocess_run(
        str(tmp_path / "missing.mcap"),
        str(output),
        "--allow-unapproved",
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["stage"] == "source_inspection"
    assert payload["error"]["code"] == "SOURCE_NOT_FOUND"
    assert payload["provider_requests"] == 0
    assert not output.exists()
    assert tuple(tmp_path.glob(".must-not-exist.partial-*")) == ()


def test_cli_rejects_an_existing_output_root_before_mapping_access(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    assert cli.main([str(tmp_path / "never-opened.mcap"), str(output)]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "arguments"
    assert payload["error"] == {
        "code": "INVALID_ARGUMENT",
        "message": "output root must be absent and must not be a symlink",
    }


def test_cli_rejects_an_existing_registry_file(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    registry = tmp_path / "registry-file"
    registry.write_text("not a directory", encoding="utf-8")

    assert (
        cli.main(
            [
                str(tmp_path / "never-opened.mcap"),
                str(output),
                "--allow-unapproved",
                "--registry-root",
                str(registry),
            ]
        )
        == 2
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "arguments"
    assert payload["error"] == {
        "code": "INVALID_ARGUMENT",
        "message": "registry root must be a directory when it already exists",
    }
    assert not output.exists()


def test_cli_rejects_an_explicit_registry_nested_in_the_output(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"

    assert (
        cli.main(
            [
                str(tmp_path / "never-opened.mcap"),
                str(output),
                "--registry-root",
                str(output / "registry"),
            ]
        )
        == 2
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "arguments"
    assert payload["error"] == {
        "code": "INVALID_ARGUMENT",
        "message": "registry root must not be inside the output root",
    }
    assert not output.exists()


@pytest.mark.parametrize(
    ("no_event", "status", "event_count", "attempts", "explicit_registry"),
    [
        (False, RunStatus.PRIMARY_COMPLETE, 1, 5, True),
        (True, RunStatus.PRIMARY_COMPLETE_NO_EVENTS, 0, 2, False),
    ],
)
def test_cli_composes_video_and_analysis_with_fake_model_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    *,
    no_event: bool,
    status: RunStatus,
    event_count: int,
    attempts: int,
    explicit_registry: bool,
) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "run"
    registry = tmp_path / "shared-registry"
    captured: dict[str, Any] = {}
    profile = SimpleNamespace(semantic_digest="a" * 64)
    inspection = object()
    channels = object()
    video = SimpleNamespace(
        output_directory=None,
        manifest=SimpleNamespace(schema_version="2.0"),
        manifest_sha256="b" * 64,
        manifest_artifact_id="artifact-video-manifest",
        derivation_reused=False,
        materialized_view_reused=False,
    )
    report = SimpleNamespace(
        status=status,
        event_count=event_count,
        fake_inference_attempt_count=attempts,
        real_provider_request_count=0,
        run_id="11111111-1111-1111-1111-111111111111",
        pipeline_version="local-mainline-v0",
    )
    events = (
        ()
        if no_event
        else (
            SimpleNamespace(
                event_id="22222222-2222-2222-2222-222222222222",
                action_type="object_interaction",
                interval=SimpleNamespace(start_ns=10, end_ns=20),
                status=SimpleNamespace(value="FINAL"),
                production_eligible=False,
            ),
        )
    )
    analysis = SimpleNamespace(
        output_directory=None,
        bundle=SimpleNamespace(report=report, events=events),
        bundle_sha256="c" * 64,
    )

    class _Policy:
        def resolve(self, actual_inspection: object) -> object:
            assert actual_inspection is inspection
            return channels

    class _Inspector:
        def inspect(self, actual_source: Path) -> object:
            assert actual_source == source
            captured["inspected"] = True
            return inspection

    class _VideoService:
        def __init__(self, exporter: object, artifact_registry: object) -> None:
            captured["video_exporter"] = exporter
            captured["artifact_registry"] = artifact_registry

        def export_local(self, request: Any) -> object:
            captured["video_request"] = request
            request.output_directory.mkdir()
            (request.output_directory / "video-marker").write_text("video")
            captured["staging_root"] = request.output_directory.parent
            video.output_directory = request.output_directory
            return video

    class _Model:
        external_provider_requests = 0

        def __init__(self, *, no_event: bool) -> None:
            captured["no_event"] = no_event

    class _Pipeline:
        def __init__(
            self,
            frames: object,
            model: object,
            *,
            config: object,
        ) -> None:
            captured["frames"] = frames
            captured["model"] = model
            captured["config"] = config

        def run(self, actual_video: object, analysis_output: Path) -> object:
            assert actual_video is video
            assert analysis_output == captured["staging_root"] / "analysis"
            analysis_output.mkdir()
            (analysis_output / "analysis-marker").write_text("analysis")
            analysis.output_directory = analysis_output
            return analysis

    monkeypatch.setattr(cli.TopicMappingProfile, "load", lambda _path: profile)
    monkeypatch.setattr(
        cli.ExactTopicMappingPolicy,
        "from_profile",
        lambda actual_profile, *, allow_unapproved: (
            _Policy()
            if actual_profile is profile and allow_unapproved
            else pytest.fail("mapping authorization was composed incorrectly")
        ),
    )
    monkeypatch.setattr(cli, "OfficialMcapInspector", _Inspector)
    monkeypatch.setattr(cli, "PyAvH264Mp4Exporter", lambda: object())
    monkeypatch.setattr(
        cli,
        "LocalArtifactRegistry",
        lambda actual_root: captured.setdefault("registry_root", actual_root) or object(),
    )
    monkeypatch.setattr(cli, "RegisteredSixCameraVideoExportService", _VideoService)
    monkeypatch.setattr(cli, "PyAvFrameMaterializer", lambda: object())
    monkeypatch.setattr(cli, "DeterministicFakeVisionModelAdapter", _Model)
    monkeypatch.setattr(cli, "LocalMainlinePipeline", _Pipeline)

    arguments = [
        str(source),
        str(output),
        "--allow-unapproved",
        "--namespace",
        "test-namespace",
        "--coarse-rate",
        "3/2",
        "--dense-rate",
        "7/3",
    ]
    if explicit_registry:
        arguments.extend(("--registry-root", str(registry)))
    if no_event:
        arguments.append("--no-event")

    assert cli.main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)

    assert captured["inspected"] is True
    assert captured["no_event"] is no_event
    expected_registry = (
        registry.resolve() if explicit_registry else output.parent / ".robata-artifacts"
    )
    assert captured["registry_root"] == expected_registry
    staging_root = captured["staging_root"]
    assert staging_root.parent == output.parent
    assert staging_root.name.startswith(f".{output.name}.partial-")
    assert captured["video_request"].output_directory == staging_root / "video"
    assert captured["video_request"].namespace == "test-namespace"
    assert captured["config"].coarse_rate_num == 3
    assert captured["config"].coarse_rate_den == 2
    assert captured["config"].dense_rate_num == 7
    assert captured["config"].dense_rate_den == 3
    assert payload["run_status"] == status.value
    assert payload["fake_inference_attempt_count"] == attempts
    assert payload["event_count"] == event_count
    assert payload["provider_requests"] == 0
    assert Path(payload["registry_root"]) == expected_registry
    assert payload["video"]["output_directory"] == str(output / "video")
    assert payload["analysis"]["output_directory"] == str(output / "analysis")
    assert (output / "video" / "video-marker").read_text() == "video"
    assert (output / "analysis" / "analysis-marker").read_text() == "analysis"
    assert tuple(output.parent.glob(f".{output.name}.partial-*")) == ()
    assert len(payload["analysis"]["events"]) == event_count
    if event_count:
        assert payload["analysis"]["events"][0]["start_ns"] == "10"
        assert payload["analysis"]["events"][0]["end_ns"] == "20"
        assert payload["analysis"]["events"][0]["production_eligible"] is False


def test_cli_analysis_failure_removes_the_complete_top_level_staging_tree(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "run"
    profile = SimpleNamespace(semantic_digest="a" * 64)
    inspection = object()

    class _Policy:
        def resolve(self, actual_inspection: object) -> object:
            assert actual_inspection is inspection
            return object()

    class _Inspector:
        def inspect(self, actual_source: Path) -> object:
            assert actual_source == source
            return inspection

    class _VideoService:
        def __init__(self, _exporter: object, _registry: object) -> None:
            pass

        def export_local(self, request: Any) -> object:
            request.output_directory.mkdir()
            (request.output_directory / "complete-video").write_text("video")
            return SimpleNamespace(output_directory=request.output_directory)

    class _Model:
        external_provider_requests = 0

        def __init__(self, *, no_event: bool) -> None:
            assert no_event is False

    class _FailingPipeline:
        def __init__(
            self,
            _frames: object,
            _model: object,
            *,
            config: object,
        ) -> None:
            assert config is not None

        def run(self, _video: object, analysis_output: Path) -> object:
            analysis_output.mkdir()
            (analysis_output / "partial-analysis").write_text("analysis")
            raise MainlineRunError(
                MainlineRunErrorCode.MODEL_INFERENCE_FAILED,
                "deliberate analysis failure",
            )

    monkeypatch.setattr(cli.TopicMappingProfile, "load", lambda _path: profile)
    monkeypatch.setattr(
        cli.ExactTopicMappingPolicy,
        "from_profile",
        lambda actual_profile, *, allow_unapproved: (
            _Policy()
            if actual_profile is profile and allow_unapproved
            else pytest.fail("mapping authorization was composed incorrectly")
        ),
    )
    monkeypatch.setattr(cli, "OfficialMcapInspector", _Inspector)
    monkeypatch.setattr(cli, "PyAvH264Mp4Exporter", lambda: object())
    monkeypatch.setattr(cli, "LocalArtifactRegistry", lambda _root: object())
    monkeypatch.setattr(cli, "RegisteredSixCameraVideoExportService", _VideoService)
    monkeypatch.setattr(cli, "PyAvFrameMaterializer", lambda: object())
    monkeypatch.setattr(cli, "DeterministicFakeVisionModelAdapter", _Model)
    monkeypatch.setattr(cli, "LocalMainlinePipeline", _FailingPipeline)

    assert cli.main([str(source), str(output), "--allow-unapproved"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "analysis"
    assert payload["error"] == {
        "code": "MODEL_INFERENCE_FAILED",
        "message": "deliberate analysis failure",
    }
    assert payload["provider_requests"] == 0
    assert not output.exists()
    assert tuple(tmp_path.glob(".run.partial-*")) == ()


@pytest.mark.skipif(
    not RUN_REAL_MAINLINE or not MEDIUM_SAMPLE.exists(),
    reason="set ROBATA_RUN_REAL_MAINLINE_ACCEPTANCE=1 with the local medium sample",
)
def test_real_mcap_reaches_one_complete_action_event(tmp_path: Path) -> None:
    output = tmp_path / "full-mainline"

    completed = _subprocess_run(
        str(MEDIUM_SAMPLE),
        str(output),
        "--allow-unapproved",
        timeout=1_200,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["run_status"] == "PRIMARY_COMPLETE"
    assert payload["fake_inference_attempt_count"] == 5
    assert payload["event_count"] == 1
    assert payload["provider_requests"] == 0
    assert Path(payload["video"]["output_directory"]) == output / "video"
    assert Path(payload["analysis"]["output_directory"]) == output / "analysis"
    assert len(tuple((output / "video").iterdir())) == 13
    assert (output / "analysis" / "run-report.json").is_file()
    assert (output / "analysis" / "mainline-bundle.json").is_file()
    assert len(tuple((output / "analysis" / "inferences").iterdir())) == 10
    assert json.loads((output / "analysis" / "action-events.json").read_text())
