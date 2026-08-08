from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_mage_video_endpoint.py"


def _script_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "run_mage_video_endpoint_test", SCRIPT_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_launcher_defaults_to_explicit_local_nf4_profile() -> None:
    module = _script_module()
    arguments = module._parser().parse_args(["--model-dir", "D:/models/mage"])

    assert arguments.load_profile == module.LOCAL_4BIT_PROFILE
    assert (
        module.PRODUCTION_NATIVE_PROFILE
        in module._parser()._option_string_actions["--load-profile"].choices
    )


def test_launcher_builds_and_verifies_checkpoint_manifest(tmp_path: Path) -> None:
    module = _script_module()
    model_directory = tmp_path / "mage"
    model_directory.mkdir()
    (model_directory / "config.json").write_text("{}", encoding="utf-8")
    weights = model_directory / "weights.bin"
    weights.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "checkpoint-manifest.json"
    state_root = tmp_path / "state"

    arguments = module._parser().parse_args(
        [
            "--model-dir",
            str(model_directory),
            "--model-identifier",
            "Mage-VL",
            "--model-revision",
            "revision-1",
            "--checkpoint-manifest-path",
            str(manifest_path),
        ]
    )
    first, first_path = module._checkpoint_manifest(arguments, state_root=state_root)
    second, second_path = module._checkpoint_manifest(arguments, state_root=state_root)

    assert first_path == manifest_path.resolve()
    assert second_path == first_path
    assert first == second
    assert first.manifest_version == "mage-checkpoint-manifest-v2"
    assert first.included_file_count == 2

    weights.write_bytes(b"changed")
    with pytest.raises(module.MageVideoEndpointLaunchError, match="changed"):
        module._checkpoint_manifest(arguments, state_root=state_root)


def test_launcher_expected_checkpoint_digest_is_only_a_verified_pin(tmp_path: Path) -> None:
    module = _script_module()
    model_directory = tmp_path / "mage"
    model_directory.mkdir()
    (model_directory / "config.json").write_text("{}", encoding="utf-8")
    arguments = module._parser().parse_args(
        [
            "--model-dir",
            str(model_directory),
            "--checkpoint-manifest-sha256",
            "0" * 64,
        ]
    )

    with pytest.raises(module.MageVideoEndpointLaunchError, match="does not match"):
        module._checkpoint_manifest(arguments, state_root=tmp_path / "state")


def test_launcher_uses_runtime_identity_for_each_declared_profile(tmp_path: Path) -> None:
    module = _script_module()
    from robata.inference import mage_video_runtime

    model_directory = tmp_path / "mage"
    model_directory.mkdir()
    runtime, local_profile = module._create_runtime(
        runtime_module=mage_video_runtime,
        model_directory=model_directory,
        offload_directory=tmp_path / "offload",
        requested_profile=module.LOCAL_4BIT_PROFILE,
    )
    native_runtime, native_profile = module._create_runtime(
        runtime_module=mage_video_runtime,
        model_directory=model_directory,
        offload_directory=tmp_path / "native-offload",
        requested_profile=module.PRODUCTION_NATIVE_PROFILE,
    )

    assert local_profile == "bitsandbytes_4bit_nf4_v1"
    assert runtime.runtime_identity.load_profile.value == local_profile
    assert native_profile == "native_bf16_v1"
    assert native_runtime.runtime_identity.load_profile.value == native_profile
