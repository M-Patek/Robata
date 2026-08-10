from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

from robata.perception.durable_scheduler import (
    DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
    SQLitePerceptionWorkScheduler,
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
    parsed = module._parser().parse_args([str(source), "--start-ns", "0", "--end-ns", "8000000000"])
    assert parsed.max_inflight_observations == 2
    assert parsed.max_new_tokens == 256


def test_execute_constructs_and_passes_the_default_vnext_scheduler(
    tmp_path: Path,
) -> None:
    module = _script_module()
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"native-video")
    arguments = module._parser().parse_args(
        [
            str(source),
            "--start-ns",
            "0",
            "--end-ns",
            "8000000000",
            "--execute",
            "--artifact-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    plan = module.plan_mage_stream(
        recording=module.MageStreamRecording(
            recording_key="runner-default-scheduler",
            recording_exact_sha256=module.exact_file_sha256(source)[0],
            interval=module.AbsoluteNanosecondInterval(0, 8_000_000_000),
        ),
        policy=module.MageStreamPolicy(
            scan_segment_duration_ns=8_000_000_000,
            reasoning_horizon_duration_ns=8_000_000_000,
        ),
    )
    captured: dict[str, object] = {}

    class _FakeModelIdentity:
        def model_dump(self, *, mode: str) -> dict[str, str]:
            assert mode == "json"
            return {"model_identifier": "fake-mage"}

    def fake_execute_local_mage_stream(**kwargs: object) -> object:
        scheduler = kwargs["durable_scheduler"]
        assert isinstance(scheduler, SQLitePerceptionWorkScheduler)
        current_plan = kwargs["plan"]
        codec_policy_version = kwargs["codec_policy_version"]
        assert isinstance(current_plan, type(plan))
        assert isinstance(codec_policy_version, str)
        run = scheduler.register_plan(current_plan, codec_policy_version=codec_policy_version)
        captured["scheduler"] = scheduler
        durable_execution = SimpleNamespace(
            run=run,
            snapshot=scheduler.snapshot(run.run_key),
            finalization_state="PLANNED",
            fusion_work_item_ids=(),
            pending_refinement_work_item_ids=(),
        )
        return SimpleNamespace(
            queue_depth=2,
            execution_profile=SimpleNamespace(value="BOUNDED_PREFETCH_NATIVE_V1"),
            timing=SimpleNamespace(
                as_projection=lambda: {
                    "profile": "BOUNDED_PREFETCH_NATIVE_V1",
                    "run_wall_seconds": 1.0,
                }
            ),
            durable_execution=durable_execution,
            run_manifest=None,
            pipeline_result=SimpleNamespace(
                normal_model_call_count=0,
                refinement_model_call_count=0,
                total_model_call_count=0,
                stage_measurements=(),
                event_tracks=(),
                fusion_decisions=(),
                refine_requests=(),
            ),
            contexts=(),
        )

    patch_names = (
        "fetch_mage_video_endpoint_health",
        "MageVideoObservationAdapter",
        "MageVideoHttpTransport",
        "FileMageVideoResultArtifactReader",
        "StreamPerceptionPipeline",
        "execute_local_mage_stream",
    )
    originals = {name: getattr(module, name) for name in patch_names}
    replacements = {
        "fetch_mage_video_endpoint_health": lambda **_kwargs: SimpleNamespace(
            model_identity=_FakeModelIdentity()
        ),
        "MageVideoObservationAdapter": lambda **kwargs: (
            captured.__setitem__("adapter_config", kwargs.get("config")) or object()
        ),
        "MageVideoHttpTransport": lambda **_kwargs: object(),
        "FileMageVideoResultArtifactReader": lambda: object(),
        "StreamPerceptionPipeline": lambda **_kwargs: object(),
        "execute_local_mage_stream": fake_execute_local_mage_stream,
    }
    for name, replacement in replacements.items():
        setattr(module, name, replacement)
    try:
        report = module._execute(arguments=arguments, plan=plan, source=source)
    finally:
        for name, original in originals.items():
            setattr(module, name, original)

    scheduler = captured["scheduler"]
    assert isinstance(scheduler, SQLitePerceptionWorkScheduler)
    assert (
        scheduler.database_path == (tmp_path / "artifacts" / "perception-vnext.sqlite3").resolve()
    )
    assert scheduler.scheduler_policy_version == DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION
    assert report["durable_execution"]["run"]["run_key"]
    assert report["durable_execution"]["stage_counts"]
    assert report["single_route"] == {
        "policy_version": "single-camera-mage-authority-v1",
        "camera_id": "cam_01",
        "authority_provider": "MAGE_NATIVE",
        "shadow_encoder_mode": "DISABLED",
        "worker_count": 1,
        "generation_concurrency": 1,
        "max_inflight_observations": 2,
        "raw_refine_provider": "MAGE_NATIVE",
    }
    assert report["decoder"] == {"max_new_tokens": 256}
    assert report["execution_profile"] == "BOUNDED_PREFETCH_NATIVE_V1"
    assert report["execution_timing"]["run_wall_seconds"] == 1.0
    adapter_config = captured["adapter_config"]
    assert adapter_config.max_new_tokens == 256


def test_compact_decoder_output_profile_is_explicit_and_identity_bound() -> None:
    module = _script_module()
    arguments = module._parser().parse_args(
        [
            "cam_01.mp4",
            "--start-ns",
            "0",
            "--end-ns",
            "8000000000",
            "--execute",
            "--decoder-output-profile",
            "compact-v1",
            "--max-new-tokens",
            "160",
        ]
    )

    assert arguments.decoder_output_profile == "compact-v1"
    assert arguments.max_new_tokens == 160
    config = module._observation_adapter_config(arguments)
    assert config.output_profile == "COMPACT_V1"
    assert config.prompt_version == "mage-unified-observation-prompt-v7-compact"
    assert config.decoder_id == "mage-observation-decoder-v3-compact"
    assert config.max_new_tokens == 160
