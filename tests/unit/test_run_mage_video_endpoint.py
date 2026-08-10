from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference import mage_traditional_codec_cache, mage_video_endpoint
from robata.inference.mage_traditional_codec_cache import (
    MageTraditionalCodecCacheAdmission,
    build_mage_traditional_codec_cache_manifest,
    build_mage_traditional_codec_effective_config,
    build_mage_traditional_codec_toolchain_identity,
    mage_traditional_codec_provider_identity,
    write_mage_traditional_codec_cache_manifest,
)
from robata.inference.mage_video_endpoint import MageVideoCodecPolicy
from robata.inference.mage_video_runtime import (
    MageVideoCodecCacheBinding,
    MageVideoExactCodecCacheAsset,
    MageVideoTraditionalCodecCacheBinding,
)

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
    assert arguments.host == "127.0.0.1"
    assert arguments.network_boundary is None
    assert arguments.allow_unauthenticated_public_bind is False
    assert module._validate_bind_security(arguments) == {
        "host": "127.0.0.1",
        "wildcard": False,
        "endpoint_authentication": "NOT_PROVIDED_BY_LAUNCHER",
        "network_boundary": None,
        "unauthenticated_public_bind_acknowledged": False,
    }


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "[::]", "0:0:0:0:0:0:0:0", "*"])
def test_launcher_rejects_wildcard_bind_without_network_boundary(host: str) -> None:
    module = _script_module()
    arguments = module._parser().parse_args(["--model-dir", "D:/models/mage", "--host", host])

    with pytest.raises(module.MageVideoEndpointLaunchError, match="wildcard bind requires"):
        module._validate_bind_security(arguments)


def test_launcher_main_rejects_wildcard_bind_before_dependency_import(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _script_module()

    def unexpected_import(_name: str) -> None:
        raise AssertionError("dependency import must not run before bind validation")

    monkeypatch.setattr(module.importlib, "import_module", unexpected_import)

    assert module.main(["--model-dir", "D:/models/mage", "--host", "0.0.0.0"]) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["code"] == "MAGE_VIDEO_ENDPOINT_FAILED"
    assert "wildcard bind requires" in failure["detail"]


@pytest.mark.parametrize(
    "network_boundary",
    ["controlled-private-network", "authenticated-reverse-proxy"],
)
def test_launcher_accepts_wildcard_bind_with_declared_network_boundary(
    network_boundary: str,
) -> None:
    module = _script_module()
    arguments = module._parser().parse_args(
        [
            "--model-dir",
            "D:/models/mage",
            "--host",
            "0.0.0.0",
            "--network-boundary",
            network_boundary,
        ]
    )

    report = module._validate_bind_security(arguments)

    assert report["wildcard"] is True
    assert report["network_boundary"] == network_boundary
    assert report["endpoint_authentication"] == "NOT_PROVIDED_BY_LAUNCHER"
    assert report["unauthenticated_public_bind_acknowledged"] is False


def test_launcher_accepts_explicit_high_risk_public_bind_acknowledgement() -> None:
    module = _script_module()
    arguments = module._parser().parse_args(
        [
            "--model-dir",
            "D:/models/mage",
            "--host",
            "::",
            "--allow-unauthenticated-public-bind",
        ]
    )

    report = module._validate_bind_security(arguments)

    assert report["wildcard"] is True
    assert report["network_boundary"] is None
    assert report["unauthenticated_public_bind_acknowledged"] is True


def test_launcher_allows_explicit_private_interface_without_public_acknowledgement() -> None:
    module = _script_module()
    arguments = module._parser().parse_args(
        ["--model-dir", "D:/models/mage", "--host", "10.42.0.7"]
    )

    report = module._validate_bind_security(arguments)

    assert report["wildcard"] is False
    assert report["network_boundary"] is None
    assert report["unauthenticated_public_bind_acknowledged"] is False


def test_launcher_generation_telemetry_jsonl_is_opt_in_and_resolved(
    tmp_path: Path,
) -> None:
    module = _script_module()
    from robata.inference import mage_video_endpoint

    default_arguments = module._parser().parse_args(["--model-dir", "D:/models/mage"])
    assert default_arguments.generation_telemetry_jsonl is None
    assert module._generation_telemetry_sink(mage_video_endpoint, default_arguments) == (None, None)

    requested = tmp_path / "telemetry" / "mage-generation.jsonl"
    arguments = module._parser().parse_args(
        [
            "--model-dir",
            "D:/models/mage",
            "--generation-telemetry-jsonl",
            str(requested),
        ]
    )
    sink, resolved = module._generation_telemetry_sink(mage_video_endpoint, arguments)

    assert resolved == requested.resolve()
    assert sink is not None
    assert sink.path == requested.resolve()
    assert requested.parent.is_dir()


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
    guard_path = tmp_path / "device-guards" / "cuda-0.lock"
    native_runtime, native_profile = module._create_runtime(
        runtime_module=mage_video_runtime,
        model_directory=model_directory,
        offload_directory=tmp_path / "native-offload",
        requested_profile=module.PRODUCTION_NATIVE_PROFILE,
        shared_device_guard_file=guard_path,
    )

    assert local_profile == "bitsandbytes_4bit_nf4_v1"
    assert runtime.runtime_identity.load_profile.value == local_profile
    assert runtime.shared_device_guard_file is None
    assert native_profile == "native_bf16_v1"
    assert native_runtime.runtime_identity.load_profile.value == native_profile
    assert native_runtime.shared_device_guard_file == guard_path.resolve()
    assert native_runtime.runtime_identity == mage_video_runtime.MageVideoRuntimeIdentity(
        load_profile=mage_video_runtime.MageVideoLoadProfile.NATIVE_BF16
    )


class _WarmupRuntime:
    def __init__(self, *, output_text: str = "private generated observation") -> None:
        self.runtime_identity = SimpleNamespace(load_profile="nf4-v1")
        self.load_observation = SimpleNamespace(load_seconds=1.25)
        self.output_text = output_text
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> SimpleNamespace:
        self.generate_calls.append(dict(kwargs))
        return SimpleNamespace(
            output_text=self.output_text,
            output_tokens=7,
            telemetry=None,
        )


class _WarmupCodecPolicy:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"policy_version": "mage-video-codec-policy-v2", "mode": "neural"}

    def native_codec_config(self) -> dict[str, object]:
        return {"codec_engine": "dcvc-rt", "max_side": 448}


class _WarmupModelIdentity:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "model_identifier": "Mage-VL",
            "model_revision": "test",
            "checkpoint_manifest_sha256": "a" * 64,
        }


def _warmup_arguments(
    module: ModuleType,
    *,
    video: Path,
    prompt: Path,
    report: Path,
    digest: str,
    omit: str | None = None,
) -> object:
    options = {
        "video": ("--warmup-video", str(video)),
        "digest": ("--warmup-video-sha256", digest),
        "prompt": ("--warmup-prompt-file", str(prompt)),
        "report": ("--warmup-report-json", str(report)),
    }
    argv = ["--model-dir", "D:/models/mage"]
    for name, pair in options.items():
        if name != omit:
            argv.extend(pair)
    return module._parser().parse_args(argv)


def test_launcher_warmup_is_disabled_by_default() -> None:
    module = _script_module()
    arguments = module._parser().parse_args(["--model-dir", "D:/models/mage"])

    class FailIfGenerated:
        def generate(self, **_kwargs: object) -> None:
            pytest.fail("warm-up must not generate unless all warm-up flags are explicit")

    assert (
        module._run_non_authoritative_warmup(
            runtime=FailIfGenerated(),
            model_identity=object(),
            codec_policy=object(),
            arguments=arguments,
            durable_input_roots=(),
            codec_cache_manifest=None,
        )
        is None
    )


@pytest.mark.parametrize("omitted", ["video", "digest", "prompt", "report"])
def test_launcher_warmup_rejects_incomplete_arguments(
    tmp_path: Path,
    omitted: str,
) -> None:
    module = _script_module()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("observe", encoding="utf-8")
    arguments = _warmup_arguments(
        module,
        video=video,
        prompt=prompt,
        report=tmp_path / "warmup.json",
        digest=hashlib.sha256(video.read_bytes()).hexdigest(),
        omit=omitted,
    )

    with pytest.raises(module.MageVideoEndpointLaunchError, match="requires"):
        module._run_non_authoritative_warmup(
            runtime=_WarmupRuntime(),
            model_identity=_WarmupModelIdentity(),
            codec_policy=_WarmupCodecPolicy(),
            arguments=arguments,
            durable_input_roots=(tmp_path,),
            codec_cache_manifest=None,
        )


def test_launcher_warmup_rejects_video_outside_durable_roots_or_wrong_digest(
    tmp_path: Path,
) -> None:
    module = _script_module()
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    video = tmp_path / "outside.mp4"
    video.write_bytes(b"video")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("observe", encoding="utf-8")
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    arguments = _warmup_arguments(
        module,
        video=video,
        prompt=prompt,
        report=tmp_path / "warmup.json",
        digest=digest,
    )
    runtime = _WarmupRuntime()

    with pytest.raises(module.MageVideoEndpointLaunchError, match="outside durable"):
        module._run_non_authoritative_warmup(
            runtime=runtime,
            model_identity=_WarmupModelIdentity(),
            codec_policy=_WarmupCodecPolicy(),
            arguments=arguments,
            durable_input_roots=(durable_root.resolve(),),
            codec_cache_manifest=None,
        )
    assert runtime.generate_calls == []

    admitted_video = durable_root / "input.mp4"
    admitted_video.write_bytes(b"different-video")
    wrong_digest_arguments = _warmup_arguments(
        module,
        video=admitted_video,
        prompt=prompt,
        report=tmp_path / "warmup.json",
        digest="0" * 64,
    )
    with pytest.raises(module.MageVideoEndpointLaunchError, match="SHA-256 does not match"):
        module._run_non_authoritative_warmup(
            runtime=runtime,
            model_identity=_WarmupModelIdentity(),
            codec_policy=_WarmupCodecPolicy(),
            arguments=wrong_digest_arguments,
            durable_input_roots=(durable_root.resolve(),),
            codec_cache_manifest=None,
        )
    assert runtime.generate_calls == []


def test_launcher_warmup_calls_runtime_directly_and_writes_redacted_canonical_report(
    tmp_path: Path,
) -> None:
    module = _script_module()
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    video = durable_root / "input.mp4"
    video.write_bytes(b"exact-private-video-bytes")
    prompt_text = "PRIVATE PROMPT: identify the action"
    output_text = "PRIVATE OUTPUT: object grasped"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(prompt_text, encoding="utf-8")
    report_path = tmp_path / "reports" / "warmup.json"
    arguments = _warmup_arguments(
        module,
        video=video,
        prompt=prompt,
        report=report_path,
        digest=hashlib.sha256(video.read_bytes()).hexdigest(),
    )
    runtime = _WarmupRuntime(output_text=output_text)
    runtime_identity_before = runtime.runtime_identity
    codec_policy = _WarmupCodecPolicy()

    summary = module._run_non_authoritative_warmup(
        runtime=runtime,
        model_identity=_WarmupModelIdentity(),
        codec_policy=codec_policy,
        arguments=arguments,
        durable_input_roots=(durable_root.resolve(),),
        codec_cache_manifest=None,
    )

    assert summary is not None
    assert summary["performed"] is True
    assert summary["authority"] == "NON_AUTHORITATIVE_DISCARDED"
    assert runtime.runtime_identity is runtime_identity_before
    assert runtime.generate_calls == [
        {
            "video_paths": [video.resolve()],
            "prompt": prompt_text,
            "max_new_tokens": 32,
            "codec_config": codec_policy.native_codec_config(),
        }
    ]
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    assert report_bytes == canonical_json_bytes(report)
    assert prompt_text.encode("utf-8") not in report_bytes
    assert output_text.encode("utf-8") not in report_bytes
    assert report["prompt_sha256"] == exact_bytes_sha256(prompt.read_bytes())
    assert report["output_text_sha256"] == exact_bytes_sha256(output_text.encode("utf-8"))
    assert report["authority"] == "NON_AUTHORITATIVE_DISCARDED"
    assert report["model_load_included_in_warmup_wall"] is False
    assert summary["report_exact_sha256"] == exact_bytes_sha256(report_bytes)


def test_launcher_provider_v2_warmup_uses_exact_verified_cache_binding(
    tmp_path: Path,
) -> None:
    module = _script_module()
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    video = durable_root / "input.mp4"
    video.write_bytes(b"exact-private-video-bytes")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("observe", encoding="utf-8")
    provider_cache = tmp_path / "qualified-cache" / "provider-entry"
    provider_cache.mkdir(parents=True)
    (provider_cache / "meta.json").write_text("{}", encoding="utf-8")
    (provider_cache / "src_patch_position.npy").write_bytes(b"positions")
    arguments = _warmup_arguments(
        module,
        video=video,
        prompt=prompt,
        report=tmp_path / "warmup.json",
        digest=hashlib.sha256(video.read_bytes()).hexdigest(),
    )
    runtime = _WarmupRuntime()
    manifest = SimpleNamespace(
        manifest_version=module.PROVIDER_V2_CACHE_MANIFEST_VERSION,
        manifest_semantic_sha256="a" * 64,
        entries=(
            SimpleNamespace(
                source_path=str(video.resolve()),
                provider_cache_directory=str(provider_cache.resolve()),
            ),
        ),
    )

    module._run_non_authoritative_warmup(
        runtime=runtime,
        model_identity=_WarmupModelIdentity(),
        codec_policy=_WarmupCodecPolicy(),
        arguments=arguments,
        durable_input_roots=(durable_root.resolve(),),
        codec_cache_manifest=manifest,
        codec_cache_binding_type=MageVideoCodecCacheBinding,
    )

    assert len(runtime.generate_calls) == 1
    binding = runtime.generate_calls[0]["codec_cache_binding"]
    assert isinstance(binding, MageVideoCodecCacheBinding)
    assert binding.source_path == video.resolve()
    assert binding.provider_cache_directory == provider_cache.resolve()


def test_launcher_warmup_fails_if_runtime_identity_changes(tmp_path: Path) -> None:
    module = _script_module()
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    video = durable_root / "input.mp4"
    video.write_bytes(b"video")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("observe", encoding="utf-8")
    arguments = _warmup_arguments(
        module,
        video=video,
        prompt=prompt,
        report=tmp_path / "warmup.json",
        digest=hashlib.sha256(video.read_bytes()).hexdigest(),
    )

    class IdentityMutatingRuntime(_WarmupRuntime):
        def generate(self, **kwargs: object) -> SimpleNamespace:
            generated = super().generate(**kwargs)
            self.runtime_identity = SimpleNamespace(load_profile="changed")
            return generated

    with pytest.raises(module.MageVideoEndpointLaunchError, match="identity changed"):
        module._run_non_authoritative_warmup(
            runtime=IdentityMutatingRuntime(),
            model_identity=_WarmupModelIdentity(),
            codec_policy=_WarmupCodecPolicy(),
            arguments=arguments,
            durable_input_roots=(durable_root.resolve(),),
            codec_cache_manifest=None,
        )
    assert not (tmp_path / "warmup.json").exists()


def _verified_cache_configuration(
    module: ModuleType,
    tmp_path: Path,
    *,
    manifest_checkpoint: str = "a" * 64,
    manifest_policy: str = "b" * 64,
) -> tuple[object, object, Path, list[dict[str, object]]]:
    source = (tmp_path / "segment.mp4").resolve()
    source.write_bytes(b"segment")
    provider_cache = (tmp_path / "provider-cache").resolve()
    provider_cache.mkdir()
    verified_entry = SimpleNamespace(
        source_path=str(source),
        source_content_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_byte_count=source.stat().st_size,
    )
    manifest_entry = SimpleNamespace(
        source_path=str(source),
        provider_cache_directory=str(provider_cache),
    )
    manifest = SimpleNamespace(
        manifest_version="mage-codec-cache-manifest-v1",
        checkpoint_manifest_sha256=manifest_checkpoint,
        codec_policy_sha256=manifest_policy,
        entries=(manifest_entry,),
        qualified_cache_root=str((tmp_path / "qualified-cache").resolve()),
        recipe=SimpleNamespace(semantic_sha256="c" * 64),
        namespace_identity="d" * 64,
    )
    verification_calls: list[dict[str, object]] = []
    cache_module = SimpleNamespace(
        load_mage_codec_cache_manifest=lambda *, path: manifest,
        verify_mage_codec_cache_manifest=lambda *, manifest: (verified_entry,),
        verify_mage_codec_cache_entry=lambda **kwargs: verification_calls.append(kwargs),
    )
    endpoint_module = SimpleNamespace(
        MageVideoCodecCacheBinding=MageVideoCodecCacheBinding,
        build_mage_video_codec_policy_identity=lambda policy: SimpleNamespace(
            policy_sha256=policy.policy_sha256
        ),
    )
    arguments = module._parser().parse_args(
        [
            "--model-dir",
            "D:/models/mage",
            "--codec-cache-manifest",
            str(tmp_path / "manifest.json"),
            "--require-verified-codec-cache",
        ]
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"manifest_version": "mage-codec-cache-manifest-v1"}),
        encoding="utf-8",
    )
    codec_policy = SimpleNamespace(policy_sha256="b" * 64)
    configuration = module._codec_cache_configuration(
        cache_module=cache_module,
        endpoint_module=endpoint_module,
        arguments=arguments,
        checkpoint_sha256="a" * 64,
        codec_policy=codec_policy,
    )
    assert configuration.cache_root == Path(manifest.qualified_cache_root)
    assert configuration.manifest is manifest
    assert configuration.cache_family == module.OBSERVED_V1_CACHE_FAMILY
    assert configuration.admission is not None
    return configuration.admission, verified_entry, source, verification_calls


def test_launcher_verified_cache_requirement_without_manifest_fails_closed() -> None:
    module = _script_module()
    arguments = module._parser().parse_args(
        [
            "--model-dir",
            "D:/models/mage",
            "--require-verified-codec-cache",
        ]
    )

    with pytest.raises(module.MageVideoEndpointLaunchError, match="requires"):
        module._codec_cache_configuration(
            cache_module=object(),
            endpoint_module=object(),
            arguments=arguments,
            checkpoint_sha256="a" * 64,
            codec_policy=object(),
        )


@pytest.mark.parametrize(
    ("manifest_checkpoint", "manifest_policy", "match"),
    [
        ("f" * 64, "b" * 64, "checkpoint identity"),
        ("a" * 64, "f" * 64, "policy identity"),
    ],
)
def test_launcher_verified_cache_configuration_rejects_identity_mismatch(
    tmp_path: Path,
    manifest_checkpoint: str,
    manifest_policy: str,
    match: str,
) -> None:
    module = _script_module()

    with pytest.raises(module.MageVideoEndpointLaunchError, match=match):
        _verified_cache_configuration(
            module,
            tmp_path,
            manifest_checkpoint=manifest_checkpoint,
            manifest_policy=manifest_policy,
        )


@pytest.mark.parametrize("mismatch", ["source-path", "source-digest", "checkpoint", "policy"])
def test_launcher_verified_cache_admission_fails_closed(
    tmp_path: Path,
    mismatch: str,
) -> None:
    module = _script_module()
    admission, verified_entry, source, verification_calls = _verified_cache_configuration(
        module,
        tmp_path,
    )
    request = SimpleNamespace(
        camera_encodings=(
            SimpleNamespace(
                segment_manifest=SimpleNamespace(
                    content_sha256=verified_entry.source_content_sha256,
                    byte_count=verified_entry.source_byte_count,
                )
            ),
        ),
        model_identity=SimpleNamespace(checkpoint_manifest_sha256="a" * 64),
        codec_policy=SimpleNamespace(policy_sha256="b" * 64),
    )
    paths = [source]
    expected = ""
    if mismatch == "source-path":
        paths = [tmp_path / "unlisted.mp4"]
        expected = "source is absent"
    elif mismatch == "source-digest":
        request.camera_encodings[0].segment_manifest.content_sha256 = "f" * 64
        expected = "source digest"
    elif mismatch == "checkpoint":
        request.model_identity.checkpoint_manifest_sha256 = "f" * 64
        expected = "checkpoint"
    else:
        request.codec_policy.policy_sha256 = "f" * 64
        expected = "codec policy"

    with pytest.raises(RuntimeError, match=expected):
        admission(request, paths)
    assert verification_calls == []


def test_launcher_verified_cache_admission_reverifies_exact_entry(tmp_path: Path) -> None:
    module = _script_module()
    admission, verified_entry, source, verification_calls = _verified_cache_configuration(
        module,
        tmp_path,
    )
    request = SimpleNamespace(
        camera_encodings=(
            SimpleNamespace(
                segment_manifest=SimpleNamespace(
                    content_sha256=verified_entry.source_content_sha256,
                    byte_count=verified_entry.source_byte_count,
                )
            ),
        ),
        model_identity=SimpleNamespace(checkpoint_manifest_sha256="a" * 64),
        codec_policy=SimpleNamespace(policy_sha256="b" * 64),
    )

    admission(request, [source])

    assert verification_calls == [
        {
            "cache_directory": (tmp_path / "provider-cache").resolve(),
            "expected_source_path": source,
            "expected_checkpoint_manifest_sha256": "a" * 64,
            "expected_codec_policy_sha256": "b" * 64,
            "expected_recipe_sha256": "c" * 64,
            "expected_namespace_identity": "d" * 64,
        }
    ]


def test_launcher_provider_v2_gate_rejects_observed_v1_rollback(tmp_path: Path) -> None:
    module = _script_module()
    manifest_path = tmp_path / "observed-v1.json"
    manifest_path.write_text(
        json.dumps({"manifest_version": "mage-codec-cache-manifest-v1"}),
        encoding="utf-8",
    )
    arguments = module._parser().parse_args(
        [
            "--model-dir",
            "D:/models/mage",
            "--codec-cache-manifest",
            str(manifest_path),
            "--require-provider-v2-cache",
        ]
    )

    with pytest.raises(module.MageVideoEndpointLaunchError, match="rejects the observed-v1"):
        module._codec_cache_configuration(
            cache_module=object(),
            endpoint_module=object(),
            arguments=arguments,
            checkpoint_sha256="a" * 64,
            codec_policy=object(),
        )


def test_launcher_provider_v2_requires_qualified_manifest(tmp_path: Path) -> None:
    module = _script_module()
    manifest_path = tmp_path / "provider-v2.json"
    manifest_path.write_text(
        json.dumps({"manifest_version": "mage-codec-cache-manifest-v2"}),
        encoding="utf-8",
    )
    arguments = module._parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "qualified-model"),
            "--codec-cache-manifest",
            str(manifest_path),
        ]
    )

    with pytest.raises(module.MageVideoEndpointLaunchError, match="qualified-provider-manifest"):
        module._codec_cache_configuration(
            cache_module=object(),
            cache_v2_module=object(),
            qualified_provider_module=object(),
            preparation_worker_module=object(),
            endpoint_module=object(),
            arguments=arguments,
            checkpoint_sha256="a" * 64,
            checkpoint_manifest=object(),
            codec_policy=object(),
        )


def _provider_v2_cache_configuration(
    module: ModuleType,
    tmp_path: Path,
    *,
    observed_implementation: str | None = None,
    qualified_model_override: Path | None = None,
    intra_checkpoint_sha256: str | None = None,
    runtime_source_tamper: str | None = None,
    bundle_source_names: tuple[str, ...] | None = None,
    device_concurrency_policy: str = "exclusive-shared-device-v1",
    include_shared_device_guard: bool = True,
) -> tuple[object, Path, object, dict[str, list[object]], object]:
    implementation = "e" * 64
    model_identifier = "Mage-VL-Robata-DCVC-V2"
    model_revision = "mage-test+robata-dcvc-provider-v2-20260809"
    model_root = (tmp_path / "qualified-model").resolve()
    neural_root = model_root / "neural_codec"
    neural_root.mkdir(parents=True)
    intra = neural_root / "dcvc_rt_intra.tar"
    inter = neural_root / "dcvc_rt_inter.tar"
    intra.write_bytes(b"intra-checkpoint")
    inter.write_bytes(b"inter-checkpoint")
    actual_intra_sha = hashlib.sha256(intra.read_bytes()).hexdigest()
    actual_inter_sha = hashlib.sha256(inter.read_bytes()).hexdigest()

    runtime_source_root = tmp_path / "runtime-provider-sources"
    runtime_source_root.mkdir()
    guard_source = runtime_source_root / "device_execution_guard.py"
    protocol_source = runtime_source_root / "mage_dcvc_preparation_protocol.py"
    worker_source = runtime_source_root / "mage_dcvc_preparation_worker.py"
    guard_source.write_bytes(b"provider shared device guard v1\n")
    protocol_source.write_bytes(b"provider protocol v2\n")
    worker_source.write_bytes(b"provider worker v2\n")
    source_by_name = {
        guard_source.name: guard_source,
        protocol_source.name: protocol_source,
        worker_source.name: worker_source,
    }

    def provider_file(source_name: str) -> object:
        provider_source = source_by_name[source_name]
        payload = {
            "relative_path": f"neural_codec/robata_provider_v2/{source_name}",
            "byte_count": provider_source.stat().st_size,
            "sha256": hashlib.sha256(provider_source.read_bytes()).hexdigest(),
        }
        return SimpleNamespace(
            **payload,
            model_dump=lambda *, mode, _payload=payload: dict(_payload),
        )

    bound_source_names = bundle_source_names or tuple(source_by_name)
    provider_files = tuple(provider_file(name) for name in bound_source_names)
    if runtime_source_tamper is not None:
        source_by_name[runtime_source_tamper].write_bytes(b"tampered runtime source\n")

    source = (tmp_path / "segment.mp4").resolve()
    source.write_bytes(b"segment-v2")
    provider_cache = (tmp_path / "cache" / "namespace-v2" / "provider-entry").resolve()
    provider_cache.mkdir(parents=True)
    (provider_cache / "meta.json").write_text("{}", encoding="utf-8")
    (provider_cache / "src_patch_position.npy").write_bytes(b"positions")
    effective_config = SimpleNamespace(
        effective_config_version="mage-dcvc-effective-config-v2",
        effective_config_sha256="c" * 64,
        provider_implementation_sha256=implementation,
        intra_checkpoint_sha256=intra_checkpoint_sha256 or actual_intra_sha,
        inter_checkpoint_sha256=actual_inter_sha,
        preparation_device=(
            "cuda:0" if device_concurrency_policy == "exclusive-shared-device-v1" else "cpu"
        ),
        device_concurrency_policy=device_concurrency_policy,
    )
    verified_entry = SimpleNamespace(
        source_path=str(source),
        source_content_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_byte_count=source.stat().st_size,
        checkpoint_manifest_sha256="a" * 64,
        codec_policy_sha256="b" * 64,
        namespace_identity="d" * 64,
        provider_implementation_sha256=implementation,
        effective_config_sha256=effective_config.effective_config_sha256,
        recipe_version="mage-dcvc-readiness-explicit-v2",
    )
    manifest_entry = SimpleNamespace(
        source_path=str(source),
        provider_cache_directory=str(provider_cache),
    )
    manifest = SimpleNamespace(
        manifest_version="mage-codec-cache-manifest-v2",
        manifest_semantic_sha256="1" * 64,
        checkpoint_manifest_sha256="a" * 64,
        codec_policy_sha256="b" * 64,
        entries=(manifest_entry,),
        entry_count=1,
        qualified_cache_root=str(provider_cache.parent),
        namespace_identity="d" * 64,
        provider_version="robata-mage-dcvc-provider-v2",
        provider_implementation_sha256=implementation,
        effective_config=effective_config,
        recipe_version="mage-dcvc-readiness-explicit-v2",
    )
    checkpoint_manifest = SimpleNamespace(
        manifest_sha256="a" * 64,
        model_identifier=model_identifier,
        model_revision=model_revision,
    )
    qualified_manifest = SimpleNamespace(
        manifest_version="mage-dcvc-qualified-provider-manifest-v2",
        manifest_semantic_sha256="2" * 64,
        provider_version="robata-mage-dcvc-provider-v2",
        qualified_model_directory=str((qualified_model_override or model_root).resolve()),
        qualified_checkpoint_manifest=checkpoint_manifest,
        bundle=SimpleNamespace(
            bundle_semantic_sha256="3" * 64,
            qualified_model_identifier=model_identifier,
            qualified_model_revision=model_revision,
            provider_files=provider_files,
        ),
    )
    calls: dict[str, list[object]] = {
        "manifest": [],
        "entry": [],
        "qualified": [],
        "policy": [],
        "implementation": [],
    }

    def verify_entry(**kwargs: object) -> object:
        calls["entry"].append(kwargs)
        return kwargs["expected_entry"]

    cache_v2_module = SimpleNamespace(
        load_mage_codec_cache_manifest_v2=lambda *, path: manifest,
        verify_mage_codec_cache_manifest_v2=lambda *, manifest: (verified_entry,),
        verify_mage_codec_cache_entry_v2=verify_entry,
        validate_mage_dcvc_effective_config_for_policy=(
            lambda **kwargs: calls["policy"].append(kwargs)
        ),
    )
    qualified_provider_module = SimpleNamespace(
        load_mage_dcvc_qualified_provider_manifest=lambda *, manifest_path: qualified_manifest,
        verify_mage_dcvc_qualified_provider=(
            lambda *, manifest: calls["qualified"].append(manifest)
        ),
    )

    def implementation_sha(model_directory: Path) -> str:
        calls["implementation"].append(model_directory)
        return observed_implementation or implementation

    preparation_worker_module = SimpleNamespace(
        __file__=str(worker_source),
        _device_guard=SimpleNamespace(__file__=str(guard_source)),
        _protocol=SimpleNamespace(__file__=str(protocol_source)),
        build_mage_dcvc_provider_implementation_sha256=implementation_sha,
    )
    endpoint_module = SimpleNamespace(
        MageVideoCodecCacheBinding=MageVideoCodecCacheBinding,
        build_mage_video_codec_policy_identity=lambda policy: SimpleNamespace(
            policy_sha256=policy.policy_sha256
        ),
    )
    manifest_path = tmp_path / "provider-v2-cache.json"
    manifest_path.write_text(
        json.dumps({"manifest_version": "mage-codec-cache-manifest-v2"}),
        encoding="utf-8",
    )
    qualified_path = tmp_path / "qualified-provider.json"
    qualified_path.write_text("{}", encoding="utf-8")
    argument_values = [
        "--model-dir",
        str(model_root),
        "--model-identifier",
        model_identifier,
        "--model-revision",
        model_revision,
        "--codec-cache-manifest",
        str(manifest_path),
        "--qualified-provider-manifest",
        str(qualified_path),
        "--require-provider-v2-cache",
    ]
    if include_shared_device_guard:
        argument_values.extend(
            ["--shared-device-guard-file", str(tmp_path / "device-guards" / "cuda-0.lock")]
        )
    arguments = module._parser().parse_args(argument_values)
    codec_policy = SimpleNamespace(policy_sha256="b" * 64)
    configuration = module._codec_cache_configuration(
        cache_module=object(),
        cache_v2_module=cache_v2_module,
        qualified_provider_module=qualified_provider_module,
        preparation_worker_module=preparation_worker_module,
        endpoint_module=endpoint_module,
        arguments=arguments,
        checkpoint_sha256="a" * 64,
        checkpoint_manifest=checkpoint_manifest,
        codec_policy=codec_policy,
    )
    calls["manifest"].append(manifest)
    return configuration, source, verified_entry, calls, arguments


def test_launcher_provider_v2_shared_gpu_requires_operational_guard(tmp_path: Path) -> None:
    module = _script_module()

    with pytest.raises(
        module.MageVideoEndpointLaunchError,
        match="shared local GPU requires --shared-device-guard-file",
    ):
        _provider_v2_cache_configuration(
            module,
            tmp_path,
            include_shared_device_guard=False,
        )


def test_launcher_provider_v2_separate_devices_preserve_optional_guard_compatibility(
    tmp_path: Path,
) -> None:
    module = _script_module()
    configuration, _source, _entry, _calls, arguments = _provider_v2_cache_configuration(
        module,
        tmp_path,
        device_concurrency_policy="separate-device-v1",
        include_shared_device_guard=False,
    )

    assert configuration.cache_family == module.PROVIDER_V2_CACHE_FAMILY
    assert arguments.shared_device_guard_file is None
    report = module._codec_cache_startup_report(configuration, arguments=arguments)
    assert report is not None
    assert report["shared_device_guard"] == {
        "required": False,
        "path": None,
        "identity_authoritative": False,
    }


def test_launcher_provider_v2_verifies_qualification_and_recomputes_current_provider(
    tmp_path: Path,
) -> None:
    module = _script_module()
    configuration, source, verified_entry, calls, arguments = _provider_v2_cache_configuration(
        module, tmp_path
    )

    assert configuration.cache_family == module.PROVIDER_V2_CACHE_FAMILY
    assert configuration.manifest_path == (tmp_path / "provider-v2-cache.json").resolve()
    assert calls["qualified"] == [configuration.qualified_provider_manifest]
    assert calls["implementation"] == [(tmp_path / "qualified-model").resolve()]
    assert len(calls["policy"]) == 1

    request = SimpleNamespace(
        camera_encodings=(
            SimpleNamespace(
                segment_manifest=SimpleNamespace(
                    content_sha256=verified_entry.source_content_sha256,
                    byte_count=verified_entry.source_byte_count,
                )
            ),
        ),
        model_identity=SimpleNamespace(checkpoint_manifest_sha256="a" * 64),
        codec_policy=SimpleNamespace(policy_sha256="b" * 64),
    )
    assert configuration.admission is not None
    binding = configuration.admission(request, [source])
    assert isinstance(binding, MageVideoCodecCacheBinding)
    assert binding.source_path == source
    assert (
        binding.provider_cache_directory
        == (tmp_path / "cache" / "namespace-v2" / "provider-entry").resolve()
    )
    assert calls["entry"] == [
        {
            "cache_directory": (tmp_path / "cache" / "namespace-v2" / "provider-entry").resolve(),
            "expected_entry": verified_entry,
            "effective_config": configuration.manifest.effective_config,
        }
    ]

    report = module._codec_cache_startup_report(configuration, arguments=arguments)
    assert report is not None
    assert report["cache_family"] == module.PROVIDER_V2_CACHE_FAMILY
    assert report["shared_device_guard"] == {
        "required": True,
        "path": str((tmp_path / "device-guards" / "cuda-0.lock").resolve()),
        "identity_authoritative": False,
    }
    assert report["provider_identity"] == {
        "provider_version": "robata-mage-dcvc-provider-v2",
        "provider_implementation_sha256": "e" * 64,
    }
    assert report["effective_config_identity"] == {
        "effective_config_version": "mage-dcvc-effective-config-v2",
        "effective_config_sha256": "c" * 64,
    }
    assert report["recipe_identity"] == {"recipe_version": "mage-dcvc-readiness-explicit-v2"}
    assert report["qualified_provider_identity"] == {
        "manifest_path": str((tmp_path / "qualified-provider.json").resolve()),
        "manifest_version": "mage-dcvc-qualified-provider-manifest-v2",
        "manifest_semantic_sha256": "2" * 64,
        "bundle_semantic_sha256": "3" * 64,
        "qualified_checkpoint_manifest_sha256": "a" * 64,
        "qualified_model_identifier": "Mage-VL-Robata-DCVC-V2",
        "qualified_model_revision": "mage-test+robata-dcvc-provider-v2-20260809",
        "provider_source_files": [
            item.model_dump(mode="json")
            for item in configuration.qualified_provider_manifest.bundle.provider_files
        ],
    }


def test_launcher_provider_v2_rejects_runtime_source_tamper(tmp_path: Path) -> None:
    module = _script_module()

    with pytest.raises(module.MageVideoEndpointLaunchError, match="source bytes differ"):
        _provider_v2_cache_configuration(
            module,
            tmp_path,
            runtime_source_tamper="mage_dcvc_preparation_worker.py",
        )


def test_launcher_provider_v2_rejects_incomplete_qualified_source_set(tmp_path: Path) -> None:
    module = _script_module()

    with pytest.raises(module.MageVideoEndpointLaunchError, match="exact executing source set"):
        _provider_v2_cache_configuration(
            module,
            tmp_path,
            bundle_source_names=("mage_dcvc_preparation_protocol.py",),
        )


@pytest.mark.parametrize("mismatch", ["implementation", "model-directory", "checkpoint"])
def test_launcher_provider_v2_rejects_current_identity_drift(
    tmp_path: Path,
    mismatch: str,
) -> None:
    module = _script_module()
    kwargs: dict[str, object] = {}
    expected = ""
    if mismatch == "implementation":
        kwargs["observed_implementation"] = "f" * 64
        expected = "current Provider V2 implementation"
    elif mismatch == "model-directory":
        kwargs["qualified_model_override"] = tmp_path / "other-model"
        expected = "model directory"
    else:
        kwargs["intra_checkpoint_sha256"] = "f" * 64
        expected = "dcvc_rt_intra.tar"

    with pytest.raises(module.MageVideoEndpointLaunchError, match=expected):
        _provider_v2_cache_configuration(module, tmp_path, **kwargs)


_TRADITIONAL_CHECKPOINT_SHA256 = "a" * 64
_TRADITIONAL_PROVIDER_IMPLEMENTATION_SHA256 = "b" * 64
_TRADITIONAL_PACKAGE_SHA256 = "c" * 64
_TRADITIONAL_EXECUTABLE_SHA256 = "d" * 64
_TRADITIONAL_COMMAND_SHA256 = "e" * 64
_TRADITIONAL_IMAGE_DIGEST = "f" * 64


def _traditional_launcher_configuration(
    module: ModuleType,
    tmp_path: Path,
    *,
    extra_arguments: tuple[str, ...] = (),
    omitted_pin: str | None = None,
) -> tuple[object, object, Path, Path, MageVideoCodecPolicy, object]:
    policy = MageVideoCodecPolicy(
        codec_mode="traditional",
        preprocess_device="cpu",
        target_canvas=8,
        group_size=8,
        images_per_group=4,
        patch_size=16,
        max_pixels=65_536,
        min_group_frames=8,
        max_group_frames=128,
    )
    toolchain = build_mage_traditional_codec_toolchain_identity(
        package_version="0.2.5",
        package_artifact_sha256=_TRADITIONAL_PACKAGE_SHA256,
        executable_sha256=_TRADITIONAL_EXECUTABLE_SHA256,
        provider_command_contract_sha256=_TRADITIONAL_COMMAND_SHA256,
        container_image_reference=(
            "robata/mage-traditional-codec@sha256:" + _TRADITIONAL_IMAGE_DIGEST
        ),
        container_image_digest=_TRADITIONAL_IMAGE_DIGEST,
        container_platform="linux/amd64",
    )
    effective_config = build_mage_traditional_codec_effective_config(
        codec_policy=policy,
        provider_options={
            "decode_backend": "cv_reader_pixels",
            "grouping_mode": "readiness",
            "parallel_decode_cv_reader": False,
        },
    )
    provider_identity = mage_traditional_codec_provider_identity(
        provider_implementation_sha256=_TRADITIONAL_PROVIDER_IMPLEMENTATION_SHA256,
        toolchain_identity_sha256=toolchain.toolchain_identity_sha256,
        effective_config_sha256=effective_config.effective_config_sha256,
    )
    source = (tmp_path / "durable" / "segment-000000.mp4").resolve()
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"exact-h264-segment")
    cache_base = (tmp_path / "traditional-cache").resolve()
    # The manifest builder requires the directory under its derived namespace;
    # derive that locator with a provisional empty directory name after the
    # source/policy/provider identities are known.
    policy_sha = mage_video_endpoint.build_mage_video_codec_policy_identity(policy).policy_sha256
    codec_config_sha = mage_traditional_codec_cache.mage_video_codec_config_sha256(
        policy.native_codec_config()
    )
    namespace = mage_traditional_codec_cache.mage_traditional_codec_namespace_identity(
        checkpoint_manifest_sha256=_TRADITIONAL_CHECKPOINT_SHA256,
        codec_policy_sha256=policy_sha,
        codec_config_sha256=codec_config_sha,
        provider_identity_sha256=provider_identity,
    )
    provider_directory = cache_base / namespace / "segment-000000"
    provider_directory.mkdir(parents=True)
    (provider_directory / "canvas_000.jpg").write_bytes(b"jpeg")
    (provider_directory / "meta.json").write_bytes(b'{"canvas_files":["canvas_000.jpg"],"fps":30}')
    (provider_directory / "src_patch_position.npy").write_bytes(b"positions")
    manifest = build_mage_traditional_codec_cache_manifest(
        checkpoint_manifest_sha256=_TRADITIONAL_CHECKPOINT_SHA256,
        codec_policy=policy,
        provider_implementation_sha256=_TRADITIONAL_PROVIDER_IMPLEMENTATION_SHA256,
        toolchain=toolchain,
        effective_config=effective_config,
        cache_base_root=cache_base,
        observations=[(source, provider_directory)],
    )
    manifest_path = (tmp_path / "traditional-cache-manifest.json").resolve()
    write_mage_traditional_codec_cache_manifest(manifest=manifest, path=manifest_path)
    pin_arguments = {
        "provider": (
            "--traditional-provider-identity-sha256",
            provider_identity,
        ),
        "toolchain": (
            "--traditional-toolchain-identity-sha256",
            toolchain.toolchain_identity_sha256,
        ),
        "image": (
            "--traditional-container-image-digest",
            _TRADITIONAL_IMAGE_DIGEST,
        ),
    }
    argv = [
        "--model-dir",
        str(tmp_path / "model"),
        "--codec-mode",
        "traditional",
        "--preprocess-device",
        "cpu",
        "--codec-target-canvas",
        "8",
        "--codec-group-size",
        "8",
        "--codec-images-per-group",
        "4",
        "--codec-patch-size",
        "16",
        "--codec-max-pixels",
        "65536",
        "--codec-min-group-frames",
        "8",
        "--codec-max-group-frames",
        "128",
        "--codec-cache-manifest",
        str(manifest_path),
        "--require-traditional-codec-cache",
    ]
    for name, pair in pin_arguments.items():
        if name != omitted_pin:
            argv.extend(pair)
    argv.extend(extra_arguments)
    arguments = module._parser().parse_args(argv)
    configuration = module._codec_cache_configuration(
        cache_module=object(),
        endpoint_module=mage_video_endpoint,
        traditional_cache_module=mage_traditional_codec_cache,
        arguments=arguments,
        checkpoint_sha256=_TRADITIONAL_CHECKPOINT_SHA256,
        codec_policy=policy,
    )
    return configuration, arguments, source, provider_directory, policy, manifest


def test_launcher_cache_family_gates_are_mutually_exclusive() -> None:
    module = _script_module()

    with pytest.raises(SystemExit):
        module._parser().parse_args(
            [
                "--model-dir",
                "D:/models/mage",
                "--require-provider-v2-cache",
                "--require-traditional-codec-cache",
            ]
        )


def test_launcher_traditional_cache_admission_and_startup_identity(tmp_path: Path) -> None:
    module = _script_module()
    configuration, arguments, source, provider_directory, policy, manifest = (
        _traditional_launcher_configuration(module, tmp_path)
    )

    assert configuration.cache_family == module.TRADITIONAL_V1_CACHE_FAMILY
    assert configuration.cache_root == Path(manifest.qualified_cache_root).resolve()
    assert isinstance(configuration.admission, MageTraditionalCodecCacheAdmission)
    request = SimpleNamespace(
        codec_policy=policy,
        model_identity=SimpleNamespace(checkpoint_manifest_sha256=_TRADITIONAL_CHECKPOINT_SHA256),
        camera_encodings=(
            SimpleNamespace(
                segment_manifest=SimpleNamespace(
                    content_sha256=exact_bytes_sha256(source.read_bytes()),
                    byte_count=source.stat().st_size,
                    durable_path=str(source),
                )
            ),
        ),
    )
    binding = configuration.admission(request, [source])
    assert isinstance(binding, MageVideoTraditionalCodecCacheBinding)
    assert binding.provider_cache_directory == provider_directory.resolve()
    assert binding.provider_identity_sha256 == manifest.provider_identity_sha256
    assert binding.toolchain_identity_sha256 == manifest.toolchain.toolchain_identity_sha256

    report = module._codec_cache_startup_report(configuration, arguments=arguments)
    assert report is not None
    assert report["cache_family"] == module.TRADITIONAL_V1_CACHE_FAMILY
    assert report["traditional_required"] is True
    assert report["provider_v2_required"] is False
    assert report["manifest_exact_sha256"] == exact_bytes_sha256(
        configuration.manifest_path.read_bytes()
    )
    assert report["provider_identity"] == {
        "provider_version": manifest.provider_version,
        "provider_implementation_sha256": manifest.provider_implementation_sha256,
        "provider_identity_sha256": manifest.provider_identity_sha256,
    }
    assert report["toolchain_identity"]["toolchain_identity_sha256"] == (
        manifest.toolchain.toolchain_identity_sha256
    )
    assert report["container_image_identity"] == {
        "reference": manifest.toolchain.container_image_reference,
        "digest": _TRADITIONAL_IMAGE_DIGEST,
    }
    assert report["shared_device_guard"] == {
        "required": False,
        "path": None,
        "identity_authoritative": False,
    }


@pytest.mark.parametrize("omitted_pin", ["provider", "toolchain", "image"])
def test_launcher_traditional_cache_requires_all_deployment_pins(
    tmp_path: Path,
    omitted_pin: str,
) -> None:
    module = _script_module()

    with pytest.raises(module.MageVideoEndpointLaunchError, match="identity pins"):
        _traditional_launcher_configuration(module, tmp_path, omitted_pin=omitted_pin)


@pytest.mark.parametrize(
    ("extra_arguments", "match"),
    [
        (("--qualified-provider-manifest", "qualified.json"), "Provider V2"),
        (("--shared-device-guard-file", "guard.lock"), "DCVC Provider V2 control"),
    ],
)
def test_launcher_traditional_cache_rejects_dcvc_only_controls(
    tmp_path: Path,
    extra_arguments: tuple[str, ...],
    match: str,
) -> None:
    module = _script_module()

    with pytest.raises(module.MageVideoEndpointLaunchError, match=match):
        _traditional_launcher_configuration(
            module,
            tmp_path,
            extra_arguments=extra_arguments,
        )


def test_launcher_traditional_cache_rejects_non_cpu_or_neural_policy(tmp_path: Path) -> None:
    module = _script_module()
    configuration, arguments, _source, _directory, policy, _manifest = (
        _traditional_launcher_configuration(module, tmp_path)
    )
    assert configuration.cache_family == module.TRADITIONAL_V1_CACHE_FAMILY

    arguments.preprocess_device = "cuda"
    with pytest.raises(module.MageVideoEndpointLaunchError, match="preprocess-device cpu"):
        module._codec_cache_configuration(
            cache_module=object(),
            endpoint_module=mage_video_endpoint,
            traditional_cache_module=mage_traditional_codec_cache,
            arguments=arguments,
            checkpoint_sha256=_TRADITIONAL_CHECKPOINT_SHA256,
            codec_policy=policy,
        )
    arguments.preprocess_device = "cpu"
    arguments.codec_mode = "neural"
    with pytest.raises(module.MageVideoEndpointLaunchError, match="codec-mode traditional"):
        module._codec_cache_configuration(
            cache_module=object(),
            endpoint_module=mage_video_endpoint,
            traditional_cache_module=mage_traditional_codec_cache,
            arguments=arguments,
            checkpoint_sha256=_TRADITIONAL_CHECKPOINT_SHA256,
            codec_policy=policy,
        )


def test_launcher_traditional_cache_revalidates_assets(tmp_path: Path) -> None:
    module = _script_module()
    configuration, arguments, _source, provider_directory, policy, _manifest = (
        _traditional_launcher_configuration(module, tmp_path)
    )
    assert configuration.cache_family == module.TRADITIONAL_V1_CACHE_FAMILY
    (provider_directory / "meta.json").write_bytes(b"tampered")

    with pytest.raises(module.MageVideoEndpointLaunchError, match="provider assets changed"):
        module._codec_cache_configuration(
            cache_module=object(),
            endpoint_module=mage_video_endpoint,
            traditional_cache_module=mage_traditional_codec_cache,
            arguments=arguments,
            checkpoint_sha256=_TRADITIONAL_CHECKPOINT_SHA256,
            codec_policy=policy,
        )


@pytest.mark.parametrize(
    ("attribute", "match"),
    [
        ("traditional_provider_identity_sha256", "provider identity mismatch"),
        ("traditional_toolchain_identity_sha256", "toolchain identity mismatch"),
        ("traditional_container_image_digest", "container image identity mismatch"),
    ],
)
def test_launcher_traditional_cache_revalidates_every_deployment_pin(
    tmp_path: Path,
    attribute: str,
    match: str,
) -> None:
    module = _script_module()
    _configuration, arguments, _source, _directory, policy, _manifest = (
        _traditional_launcher_configuration(module, tmp_path)
    )
    setattr(arguments, attribute, "0" * 64)

    with pytest.raises(module.MageVideoEndpointLaunchError, match=match):
        module._codec_cache_configuration(
            cache_module=object(),
            endpoint_module=mage_video_endpoint,
            traditional_cache_module=mage_traditional_codec_cache,
            arguments=arguments,
            checkpoint_sha256=_TRADITIONAL_CHECKPOINT_SHA256,
            codec_policy=policy,
        )


def test_launcher_exact_traditional_replay_does_not_require_cv_preinfer() -> None:
    module = _script_module()
    calls: list[tuple[object, object]] = []
    runtime_module = SimpleNamespace(
        require_mage_video_codec_dependencies=lambda config, model: calls.append((config, model))
    )
    policy = SimpleNamespace(native_codec_config=lambda: {"engine": "hevc"})

    module._require_launch_codec_dependencies(
        runtime_module=runtime_module,
        codec_policy=policy,
        model_directory=Path("D:/models/mage"),
        cache_family=module.TRADITIONAL_V1_CACHE_FAMILY,
    )
    assert calls == []
    module._require_launch_codec_dependencies(
        runtime_module=runtime_module,
        codec_policy=policy,
        model_directory=Path("D:/models/mage"),
        cache_family=None,
    )
    assert len(calls) == 1


def test_launcher_traditional_warmup_uses_full_exact_binding(tmp_path: Path) -> None:
    module = _script_module()
    _configuration, arguments, source, provider_directory, policy, manifest = (
        _traditional_launcher_configuration(module, tmp_path)
    )
    prompt = tmp_path / "warmup-prompt.txt"
    prompt.write_text("observe", encoding="utf-8")
    report_path = tmp_path / "traditional-warmup.json"
    arguments.warmup_video = source
    arguments.warmup_video_sha256 = exact_bytes_sha256(source.read_bytes())
    arguments.warmup_prompt_file = prompt
    arguments.warmup_report_json = report_path
    runtime = _WarmupRuntime()

    module._run_non_authoritative_warmup(
        runtime=runtime,
        model_identity=_WarmupModelIdentity(),
        codec_policy=policy,
        arguments=arguments,
        durable_input_roots=(source.parent,),
        codec_cache_manifest=manifest,
        traditional_codec_cache_binding_type=MageVideoTraditionalCodecCacheBinding,
        exact_codec_cache_asset_type=MageVideoExactCodecCacheAsset,
    )

    assert len(runtime.generate_calls) == 1
    binding = runtime.generate_calls[0]["codec_cache_binding"]
    assert isinstance(binding, MageVideoTraditionalCodecCacheBinding)
    assert binding.provider_cache_directory == provider_directory.resolve()
    assert binding.asset_set_sha256 == manifest.entries[0].entry.asset_set_sha256
    report = json.loads(report_path.read_bytes())
    assert report["codec_cache_family"] == module.TRADITIONAL_V1_CACHE_FAMILY


def test_launcher_provider_v2_manifest_rejects_traditional_identity_flags(
    tmp_path: Path,
) -> None:
    module = _script_module()
    manifest_path = tmp_path / "provider-v2.json"
    manifest_path.write_text(
        json.dumps({"manifest_version": module.PROVIDER_V2_CACHE_MANIFEST_VERSION}),
        encoding="utf-8",
    )
    arguments = module._parser().parse_args(
        [
            "--model-dir",
            "D:/models/mage",
            "--codec-cache-manifest",
            str(manifest_path),
            "--traditional-provider-identity-sha256",
            "0" * 64,
        ]
    )

    with pytest.raises(module.MageVideoEndpointLaunchError, match="invalid for Provider V2"):
        module._codec_cache_configuration(
            cache_module=object(),
            cache_v2_module=object(),
            qualified_provider_module=object(),
            preparation_worker_module=object(),
            endpoint_module=object(),
            arguments=arguments,
            checkpoint_sha256="a" * 64,
            checkpoint_manifest=object(),
            codec_policy=object(),
        )
