from __future__ import annotations

import importlib.util
import json
import shlex
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPOSITORY_ROOT / "config" / "mage-h100-bf16-single-worker-v1.json"
RENDERER_PATH = REPOSITORY_ROOT / "scripts" / "render_mage_endpoint_profile.py"
ENDPOINT_LAUNCHER_PATH = REPOSITORY_ROOT / "scripts" / "run_mage_video_endpoint.py"
PREWARM_LAUNCHER_PATH = REPOSITORY_ROOT / "scripts" / "prewarm_local_mage_dcvc_provider_v2.py"


def _module(path: Path, *, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _renderer() -> ModuleType:
    return _module(RENDERER_PATH, name="render_mage_endpoint_profile_test")


def _endpoint_launcher() -> ModuleType:
    return _module(ENDPOINT_LAUNCHER_PATH, name="run_mage_video_endpoint_profile_test")


def _prewarm_launcher() -> ModuleType:
    return _module(PREWARM_LAUNCHER_PATH, name="prewarm_mage_provider_v2_profile_test")


def _arguments(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "profile": PROFILE_PATH,
        "codec_cuda_selector": "0",
        "decoder_cuda_selector": "1",
        "qualified_model_dir": "/models/Mage-VL-Qualified-V2",
        "qualified_model_revision": "provider-v2-production-candidate",
        "qualified_provider_manifest": "/identity/qualified-provider-v2.json",
        "qualification_manifest_sha256": "c" * 64,
        "checkpoint_manifest_path": "/identity/checkpoint-manifest-v2.json",
        "checkpoint_manifest_sha256": "a" * 64,
        "provider_state_root": "/state/provider-v2",
        "cache_base_root": "/cache/provider-v2",
        "codec_cache_manifest": "/state/provider-v2/cache-manifest-v2.json",
        "provider_prewarm_report_json": "/state/provider-v2/prewarm-report-v2.json",
        "prewarm_video": [
            "/data/segments/000000.mp4",
            "/data/segments/000001.mp4",
        ],
        "state_dir": "/state/endpoint",
        "durable_input_root": ["/data/segments"],
        "generation_telemetry_jsonl": "/state/endpoint/generation.jsonl",
        "warmup_video": "/data/segments/000000.mp4",
        "warmup_video_sha256": "b" * 64,
        "warmup_prompt_file": "/run/secrets/warmup-prompt.json",
        "warmup_report_json": "/state/endpoint/warmup-report.json",
        "host": None,
        "port": None,
        "readiness_only": False,
        "render_phase": "all",
        "output_format": "json",
    }
    values.update(overrides)
    return Namespace(**values)


def test_tracked_dual_h100_profile_renders_provider_v2_and_endpoint_commands() -> None:
    renderer = _renderer()
    endpoint = _endpoint_launcher()
    prewarm = _prewarm_launcher()
    profile, profile_sha256 = renderer.load_and_validate_profile(PROFILE_PATH)

    rendered = renderer.render_command(profile, _arguments())

    assert rendered["profile_semantic_sha256"] == profile_sha256
    assert rendered["evidence_class"] == "TARGET_CONFIGURATION_UNVALIDATED"
    assert rendered["production_eligible"] is False
    assert rendered["device_concurrency_policy"] == "separate-device-v1"
    assert rendered["device_mapping_authority"] == "OUTER_SELECTORS_UNATTESTED"
    assert rendered["device_uuid_attestation_required"] is True
    assert rendered["bind_security"] == {
        "host": "0.0.0.0",
        "network_boundary": "controlled-private-network",
        "endpoint_authentication": "NOT_PROVIDED_BY_LAUNCHER",
        "unauthenticated_public_bind_acknowledged": False,
    }
    assert rendered["resolution_policy"] == "max-side-448-target-candidate"
    assert rendered["bounded_resolution_selected"] is True
    assert rendered["bounded_resolution_production_qualified"] is False
    assert rendered["sequence_length_frames"] == 0
    assert rendered["sequence_length_frames_is_compute_cap"] is False
    assert rendered["canvas_token_side"] is None
    assert rendered["encoded_frame_extent"] == "through-last-sampled-frame"

    prewarm_command = rendered["commands"]["provider_v2_prewarm"]
    assert prewarm_command["environment"] == {"CUDA_VISIBLE_DEVICES": "0,1"}
    assert prewarm_command["codec_outer_cuda_selector"] == "0"
    assert prewarm_command["codec_process_logical_device"] == "cuda:0"
    assert prewarm_command["decoder_prewarm_logical_device"] == "cuda:1"
    parsed_prewarm = prewarm._parser().parse_args(prewarm_command["argv"][2:])
    assert parsed_prewarm.preparation_device == "cuda"
    assert parsed_prewarm.generation_device == "cuda:1"
    assert parsed_prewarm.max_side == 448
    assert parsed_prewarm.target_canvas == 8
    assert parsed_prewarm.group_size == 8
    assert parsed_prewarm.images_per_group == 1
    assert parsed_prewarm.max_pixels == 65_536
    assert parsed_prewarm.min_group_frames == 8
    assert parsed_prewarm.max_group_frames == 128
    assert parsed_prewarm.shared_device_guard_file is None
    assert len(parsed_prewarm.video) == 2

    endpoint_command = rendered["commands"]["endpoint"]
    assert endpoint_command["environment"] == {"CUDA_VISIBLE_DEVICES": "1"}
    assert endpoint_command["decoder_outer_cuda_selector"] == "1"
    assert endpoint_command["decoder_process_logical_device"] == "cuda:0"
    parsed_endpoint = endpoint._parser().parse_args(endpoint_command["argv"][2:])
    assert parsed_endpoint.load_profile == endpoint.PRODUCTION_NATIVE_PROFILE
    assert parsed_endpoint.model_identifier == "Mage-VL-Robata-DCVC-V2"
    assert parsed_endpoint.codec_mode == "neural"
    assert parsed_endpoint.preprocess_device == "cuda"
    assert parsed_endpoint.codec_target_canvas == 8
    assert parsed_endpoint.codec_group_size == 8
    assert parsed_endpoint.codec_images_per_group == 1
    assert parsed_endpoint.codec_max_pixels == 65_536
    assert parsed_endpoint.codec_max_group_frames == 128
    assert parsed_endpoint.neural_max_side == 448
    assert parsed_endpoint.neural_sequence_length_frames == 0
    assert parsed_endpoint.neural_canvas_token_side is None
    assert "--neural-canvas-token-side" not in endpoint_command["argv"]
    assert parsed_endpoint.require_verified_codec_cache is True
    assert parsed_endpoint.require_provider_v2_cache is True
    assert parsed_endpoint.qualified_provider_manifest is not None
    assert parsed_endpoint.shared_device_guard_file is None
    assert parsed_endpoint.warmup_max_new_tokens == 32
    assert parsed_endpoint.host == "0.0.0.0"
    assert parsed_endpoint.network_boundary == "controlled-private-network"
    assert parsed_endpoint.allow_unauthenticated_public_bind is False
    assert endpoint._validate_bind_security(parsed_endpoint)["wildcard"] is True
    assert parsed_endpoint.port == 8_102
    assert len(parsed_endpoint.durable_input_root) == 1


def test_profile_rejects_concurrency_sequence_or_qualification_claim_drift(
    tmp_path: Path,
) -> None:
    renderer = _renderer()
    original = json.loads(PROFILE_PATH.read_text(encoding="utf-8-sig"))

    policy_drift = json.loads(json.dumps(original))
    policy_drift["provider_v2_prewarm"]["device_concurrency_policy"] = "exclusive-shared-device-v1"
    modified = tmp_path / "unsafe-policy.json"
    modified.write_text(json.dumps(policy_drift), encoding="utf-8")
    with pytest.raises(renderer.MageEndpointProfileError, match="device_concurrency_policy"):
        renderer.load_and_validate_profile(modified)

    cap_drift = json.loads(json.dumps(original))
    cap_drift["provider_v2_prewarm"]["sequence_length_frames_is_compute_cap"] = True
    modified.write_text(json.dumps(cap_drift), encoding="utf-8")
    with pytest.raises(
        renderer.MageEndpointProfileError, match="sequence_length_frames_is_compute_cap"
    ):
        renderer.load_and_validate_profile(modified)

    bounded_drift = json.loads(json.dumps(original))
    bounded_drift["provider_v2_prewarm"]["bounded_resolution_production_qualified"] = True
    bounded_drift["provider_v2_prewarm"]["max_side"] = 0
    bounded_drift["endpoint_launcher"]["neural_max_side"] = 0
    modified.write_text(json.dumps(bounded_drift), encoding="utf-8")
    with pytest.raises(
        renderer.MageEndpointProfileError,
        match="bounded_resolution_production_qualified",
    ):
        renderer.load_and_validate_profile(modified)

    canvas_drift = json.loads(json.dumps(original))
    canvas_drift["provider_v2_prewarm"]["canvas_token_side"] = 8
    canvas_drift["endpoint_launcher"]["neural_canvas_token_side"] = 8
    modified.write_text(json.dumps(canvas_drift), encoding="utf-8")
    with pytest.raises(renderer.MageEndpointProfileError, match="canvas_token_side"):
        renderer.load_and_validate_profile(modified)

    network_boundary_drift = json.loads(json.dumps(original))
    network_boundary_drift["endpoint_launcher"]["network_boundary"] = "authenticated-reverse-proxy"
    modified.write_text(json.dumps(network_boundary_drift), encoding="utf-8")
    with pytest.raises(renderer.MageEndpointProfileError, match="network_boundary"):
        renderer.load_and_validate_profile(modified)

    qualification_drift = json.loads(json.dumps(original))
    qualification_drift["qualification"]["production_eligible"] = True
    modified.write_text(json.dumps(qualification_drift), encoding="utf-8")
    with pytest.raises(renderer.MageEndpointProfileError, match="production_eligible"):
        renderer.load_and_validate_profile(modified)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"decoder_cuda_selector": "0"}, "must be distinct"),
        ({"codec_cuda_selector": "2"}, "must be '0'"),
        ({"qualified_model_dir": "models/Mage-VL"}, "absolute Linux path"),
        ({"qualified_model_dir": "/models/../tmp/Mage-VL"}, "normalized absolute"),
        ({"qualification_manifest_sha256": "C" * 64}, "lowercase SHA-256"),
        ({"qualified_model_revision": "bad\nrevision"}, "non-empty string"),
        ({"port": 0}, "port must be"),
        ({"warmup_video": "/data/segments/not-prewarmed.mp4"}, "listed as a prewarm_video"),
        (
            {"prewarm_video": ["/unadmitted/segment.mp4", "/data/segments/000000.mp4"]},
            "beneath a durable_input_root",
        ),
    ],
)
def test_renderer_rejects_ambiguous_or_unbound_topology(
    overrides: dict[str, object], match: str
) -> None:
    renderer = _renderer()
    profile, _digest = renderer.load_and_validate_profile(PROFILE_PATH)

    with pytest.raises(renderer.MageEndpointProfileError, match=match):
        renderer.render_command(profile, _arguments(**overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"codec_cache_manifest": "/cache/provider-v2/cache-manifest-v2.json"},
            "codec_cache_manifest.*strictly beneath provider_state_root",
        ),
        (
            {"provider_prewarm_report_json": "/state/prewarm-report-v2.json"},
            "provider_prewarm_report_json.*strictly beneath provider_state_root",
        ),
        (
            {"generation_telemetry_jsonl": "/state/generation.jsonl"},
            "generation_telemetry_jsonl.*strictly beneath state_dir",
        ),
        (
            {"warmup_report_json": "/state/warmup-report.json"},
            "warmup_report_json.*strictly beneath state_dir",
        ),
    ],
)
def test_renderer_requires_outputs_beneath_their_explicit_state_roots(
    overrides: dict[str, object], match: str
) -> None:
    renderer = _renderer()
    profile, _digest = renderer.load_and_validate_profile(PROFILE_PATH)

    with pytest.raises(renderer.MageEndpointProfileError, match=match):
        renderer.render_command(profile, _arguments(**overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"cache_base_root": "/state/provider-v2/cache"},
            "output roots provider_state_root.*cache_base_root.*path-disjoint",
        ),
        (
            {
                "state_dir": "/state/provider-v2/endpoint",
                "generation_telemetry_jsonl": ("/state/provider-v2/endpoint/generation.jsonl"),
                "warmup_report_json": "/state/provider-v2/endpoint/warmup-report.json",
            },
            "output roots provider_state_root.*state_dir.*path-disjoint",
        ),
        (
            {"provider_prewarm_report_json": "/state/provider-v2/cache-manifest-v2.json"},
            "output files codec_cache_manifest.*provider_prewarm_report_json.*path-disjoint",
        ),
        (
            {"warmup_report_json": "/state/endpoint/generation.jsonl"},
            "output files generation_telemetry_jsonl.*warmup_report_json.*path-disjoint",
        ),
    ],
)
def test_renderer_rejects_colliding_output_roots_and_files(
    overrides: dict[str, object], match: str
) -> None:
    renderer = _renderer()
    profile, _digest = renderer.load_and_validate_profile(PROFILE_PATH)

    with pytest.raises(renderer.MageEndpointProfileError, match=match):
        renderer.render_command(profile, _arguments(**overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {
                "provider_state_root": "/models/Mage-VL-Qualified-V2/state",
                "codec_cache_manifest": (
                    "/models/Mage-VL-Qualified-V2/state/cache-manifest-v2.json"
                ),
                "provider_prewarm_report_json": (
                    "/models/Mage-VL-Qualified-V2/state/prewarm-report-v2.json"
                ),
            },
            "output root provider_state_root.*protected input qualified_model_dir",
        ),
        (
            {"cache_base_root": "/data/segments/provider-v2-cache"},
            r"output root cache_base_root.*protected input durable_input_root\[0\]",
        ),
        (
            {
                "state_dir": "/run/secrets",
                "generation_telemetry_jsonl": "/run/secrets/generation.jsonl",
                "warmup_report_json": "/run/secrets/warmup-prompt.json",
            },
            "output file warmup_report_json.*protected input warmup_prompt_file",
        ),
        (
            {
                "provider_state_root": "/identity",
                "codec_cache_manifest": "/identity/qualified-provider-v2.json",
                "provider_prewarm_report_json": "/identity/prewarm-report-v2.json",
            },
            "output file codec_cache_manifest.*protected input qualified_provider_manifest",
        ),
        (
            {
                "provider_state_root": "/identity",
                "codec_cache_manifest": "/identity/checkpoint-manifest-v2.json",
                "provider_prewarm_report_json": "/identity/prewarm-report-v2.json",
            },
            "output file codec_cache_manifest.*protected input checkpoint_manifest_path",
        ),
        (
            {
                "state_dir": "/data/segments",
                "generation_telemetry_jsonl": "/data/segments/000000.mp4",
                "warmup_report_json": "/data/segments/warmup-report.json",
            },
            r"output file generation_telemetry_jsonl.*protected input prewarm_video\[0\]",
        ),
        (
            {
                "state_dir": "/data/segments/endpoint-state",
                "generation_telemetry_jsonl": ("/data/segments/endpoint-state/generation.jsonl"),
                "warmup_report_json": "/data/segments/endpoint-state/warmup-report.json",
            },
            r"output root state_dir.*protected input durable_input_root\[0\]",
        ),
    ],
)
def test_renderer_keeps_mutable_outputs_disjoint_from_protected_inputs(
    overrides: dict[str, object], match: str
) -> None:
    renderer = _renderer()
    profile, _digest = renderer.load_and_validate_profile(PROFILE_PATH)

    with pytest.raises(renderer.MageEndpointProfileError, match=match):
        renderer.render_command(profile, _arguments(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"generation_telemetry_jsonl": "/state/endpoint/./generation.jsonl"},
        {"warmup_report_json": "/state//endpoint/warmup-report.json"},
        {"provider_state_root": "/state/provider-v2/"},
        {"cache_base_root": "//cache/provider-v2"},
    ],
)
def test_renderer_rejects_noncanonical_paths_that_can_bypass_alias_checks(
    overrides: dict[str, object],
) -> None:
    renderer = _renderer()
    profile, _digest = renderer.load_and_validate_profile(PROFILE_PATH)

    with pytest.raises(renderer.MageEndpointProfileError, match="normalized absolute Linux path"):
        renderer.render_command(profile, _arguments(**overrides))


def test_posix_shell_all_phase_orders_prewarm_before_single_endpoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = _renderer()
    argv = [
        "--profile",
        str(PROFILE_PATH),
        "--codec-cuda-selector",
        "0",
        "--decoder-cuda-selector",
        "1",
        "--qualified-model-dir",
        "/models/Mage's VL Qualified",
        "--qualified-model-revision",
        "checkpoint-1",
        "--qualified-provider-manifest",
        "/identity/qualified provider.json",
        "--qualification-manifest-sha256",
        "c" * 64,
        "--checkpoint-manifest-path",
        "/identity/checkpoint.json",
        "--checkpoint-manifest-sha256",
        "a" * 64,
        "--provider-state-root",
        "/state/provider",
        "--cache-base-root",
        "/cache/provider",
        "--codec-cache-manifest",
        "/state/provider/cache.json",
        "--provider-prewarm-report-json",
        "/state/provider/prewarm.json",
        "--prewarm-video",
        "/data/segments/warmup.mp4",
        "--state-dir",
        "/state/endpoint",
        "--durable-input-root",
        "/data/segments",
        "--generation-telemetry-jsonl",
        "/state/endpoint/generation.jsonl",
        "--warmup-video",
        "/data/segments/warmup.mp4",
        "--warmup-video-sha256",
        "b" * 64,
        "--warmup-prompt-file",
        "/run/secrets/warmup's prompt.json",
        "--warmup-report-json",
        "/state/endpoint/warmup.json",
        "--readiness-only",
        "--render-phase",
        "all",
        "--output-format",
        "posix-shell",
    ]

    assert renderer.main(argv) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "set -euo pipefail"
    assert lines[1].startswith(
        "env CUDA_VISIBLE_DEVICES=0,1 python scripts/prewarm_local_mage_dcvc_provider_v2.py"
    )
    prewarm_tokens = shlex.split(lines[1])
    assert prewarm_tokens[:3] == ["env", "CUDA_VISIBLE_DEVICES=0,1", "python"]
    assert "/models/Mage's VL Qualified" in prewarm_tokens
    assert "/identity/qualified provider.json" in prewarm_tokens

    assert lines[2].startswith(
        "exec env CUDA_VISIBLE_DEVICES=1 python scripts/run_mage_video_endpoint.py"
    )
    endpoint_tokens = shlex.split(lines[2])
    assert endpoint_tokens[:4] == ["exec", "env", "CUDA_VISIBLE_DEVICES=1", "python"]
    assert "/run/secrets/warmup's prompt.json" in endpoint_tokens
    assert endpoint_tokens[-1] == "--readiness-only"


def test_endpoint_only_shell_does_not_render_or_run_prewarm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = _renderer()
    arguments = _arguments(render_phase="endpoint", output_format="posix-shell")
    argv: list[str] = []
    for key, value in vars(arguments).items():
        if key == "profile":
            argv.extend(("--profile", str(value)))
        elif (key in {"host", "port"} and value is None) or key == "readiness_only":
            continue
        elif key == "output_format":
            argv.extend(("--output-format", str(value)))
        elif key == "render_phase":
            argv.extend(("--render-phase", str(value)))
        elif isinstance(value, list):
            flag = "--" + key.replace("_", "-")
            for item in value:
                argv.extend((flag, str(item)))
        else:
            argv.extend(("--" + key.replace("_", "-"), str(value)))

    assert renderer.main(argv) == 0
    output = capsys.readouterr().out.strip()
    assert output.startswith(
        "exec env CUDA_VISIBLE_DEVICES=1 python scripts/run_mage_video_endpoint.py"
    )
    assert "prewarm_local_mage_dcvc_provider_v2.py" not in output


def test_prewarm_only_shell_replaces_supervisor_process() -> None:
    renderer = _renderer()
    profile, _digest = renderer.load_and_validate_profile(PROFILE_PATH)
    rendered = renderer.render_command(profile, _arguments())

    output = renderer._posix_shell(rendered, phase="prewarm")

    assert output.startswith(
        "exec env CUDA_VISIBLE_DEVICES=0,1 python scripts/prewarm_local_mage_dcvc_provider_v2.py"
    )
    assert "run_mage_video_endpoint.py" not in output
