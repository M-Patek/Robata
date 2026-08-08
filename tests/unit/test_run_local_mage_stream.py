from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from robata.perception.durable_scheduler import (
    DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_local_mage_stream.py"


def _script_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "run_local_mage_stream_test", SCRIPT_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_dry_run_defaults_to_mage_and_emits_absolute_nonoverlap_plan(
    tmp_path: Path,
    capsys: object,
) -> None:
    module = _script_module()
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"native-video")

    exit_code = module.main(
        [
            str(source),
            "--start-ns",
            "1000000000",
            "--end-ns",
            "17000000000",
            "--segment-seconds",
            "8",
            "--reasoning-horizon-seconds",
            "12",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["profile"] == module.MAGE_STREAM_VNEXT_PROFILE
    assert payload["composition"] == {
        "mode": "MAGE_STREAM",
        "qwen_autoload": False,
        "scheduler_policy_version": DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
        "version": "mage-stream-composition-v1",
    }
    assert payload["qwen_weights_preserved"] is True
    assert "exactly one selected native-video camera" in payload["v1_limitation"]
    segments = payload["plan"]["storage_segments"]
    assert [segment["interval"] for segment in segments] == [
        {"end_ns": 9000000000, "start_ns": 1000000000},
        {"end_ns": 17000000000, "start_ns": 9000000000},
    ]


def test_legacy_execution_is_explicit_and_never_autoloads_qwen(
    tmp_path: Path,
    capsys: object,
) -> None:
    module = _script_module()
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"native-video")

    exit_code = module.main(
        [
            str(source),
            "--start-ns",
            "0",
            "--end-ns",
            "8000000000",
            "--profile",
            module.LEGACY_QWEN_WINDOW_PROFILE,
            "--execute",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert exit_code == 2
    assert payload["profile"] == module.LEGACY_QWEN_WINDOW_PROFILE
    assert payload["qwen_weights_preserved"] is True
    assert "not performed" in payload["detail"]


def test_execute_uses_in_repository_composition_without_pipeline_runner_hook(
    tmp_path: Path,
    capsys: object,
) -> None:
    module = _script_module()
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"native-video")
    calls: list[tuple[object, object, object]] = []

    def fake_execute(*, arguments, plan, source):  # type: ignore[no-untyped-def]
        calls.append((arguments, plan, source))
        return {"performed": True, "queue_depth": 1, "normal_model_call_count": 1}

    module._execute = fake_execute
    module.probe_video_keyframe_offsets_ns = lambda *args, **kwargs: (0,)
    exit_code = module.main(
        [
            str(source),
            "--start-ns",
            "0",
            "--end-ns",
            "8000000000",
            "--execute",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls
    assert payload["execution"]["performed"] is True
    assert not hasattr(
        module._parser().parse_args([str(source), "--start-ns", "0", "--end-ns", "8000000000"]),
        "pipeline_runner",
    )
