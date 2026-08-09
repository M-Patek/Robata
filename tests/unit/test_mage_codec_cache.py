from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from robata.contracts.hashing import canonical_json_bytes
from robata.inference.mage_checkpoint_identity import (
    MageCheckpointManifest,
    build_mage_checkpoint_manifest,
)
from robata.inference.mage_codec_cache import (
    MageCodecCacheError,
    MageCodecCacheManifest,
    build_mage_codec_effective_recipe,
    load_mage_codec_cache_manifest,
    mage_codec_cache_sidecar_path,
    mage_codec_logical_cache_identity,
    mage_codec_namespace_identity,
    prewarm_mage_codec_cache,
    verify_mage_codec_cache_manifest,
    write_mage_codec_cache_manifest,
)
from robata.inference.mage_video_endpoint import (
    MageVideoCodecPolicy,
    MageVideoNeuralCodecParameters,
    build_mage_video_codec_policy_identity,
)

_ENVIRONMENT = {"runtime": "unit-test", "cuda": "ABSENT"}


def _digest(value: int) -> str:
    return f"{value:064x}"


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_model_tree(tmp_path: Path) -> tuple[Path, MageCheckpointManifest]:
    root = tmp_path / "Mage-VL"
    files: dict[str, bytes] = {
        "codec_video_processing_mage_vl.py": b"def process_codec_video(): pass\n",
        "processing_mage_vl.py": b"class MageProcessor: pass\n",
        "preprocessor_config.json": canonical_json_bytes(
            {
                "codec": {
                    "dcvc": {
                        "intra_period": -1,
                        "max_group_frames": 128,
                        "max_side": 448,
                        "qp": 42,
                        "reset_interval": 64,
                    }
                }
            }
        ),
        "neural_codec/codec_dcvc_config.py": b"QP = 42\n",
        "neural_codec/dcvc_readiness_gen.py": b"def readiness(): pass\n",
        "neural_codec/dcvc_rt_engine.py": b"class Engine: pass\n",
        "neural_codec/codec_tools/pipeline/process_video_bitcost_readiness.py": (
            b"def process_video(): pass\n"
        ),
        "neural_codec/DCVC/src/__init__.py": b"# dcvc package\n",
        "config.json": b'{"model_type":"mage_vl"}',
        "model.safetensors": b"checkpoint-bytes",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = build_mage_checkpoint_manifest(
        model_directory=root,
        model_identifier="Mage-VL",
        model_revision="unit-test",
    )
    return root, manifest


def _codec_policy() -> MageVideoCodecPolicy:
    return MageVideoCodecPolicy(
        codec_mode="neural",
        preprocess_device="cpu",
        target_canvas=8,
        group_size=8,
        images_per_group=1,
        patch_size=16,
        max_pixels=65_536,
        min_group_frames=8,
        max_group_frames=8,
        neural_parameters=MageVideoNeuralCodecParameters(
            quantization_parameter=42,
            reset_interval=64,
            intra_period=-1,
            max_side=448,
            sequence_length_frames=8,
        ),
    )


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "<video>Observe this video segment."

    def __call__(self, **kwargs: Any) -> dict[str, list[int]]:
        self.calls.append(kwargs)
        video = Path(kwargs["videos"][0]).resolve()
        cache_root = Path(kwargs["codec_config"]["cache_root"])
        provider_name = f"{video.stem}_{hashlib.sha256(str(video).encode()).hexdigest()[:16]}"
        provider = cache_root / provider_name
        provider.mkdir(parents=True)
        canvas_name = "canvas_000.jpg"
        (provider / canvas_name).write_bytes(b"fake-canvas-bytes")
        (provider / "src_patch_position.npy").write_bytes(b"fake-position-bytes")
        (provider / "meta.json").write_bytes(
            canonical_json_bytes({"canvas_files": [canvas_name], "source": video.name})
        )
        return {"input_ids": [1]}


def _build_once(
    tmp_path: Path,
) -> tuple[MageCodecCacheManifest, _FakeProcessor, Path, Path, MageCheckpointManifest]:
    model_root, checkpoint = _write_model_tree(tmp_path)
    source = tmp_path / "segments" / "segment-000.mp4"
    source.parent.mkdir()
    source.write_bytes(b"immutable-video-segment")
    processor = _FakeProcessor()
    manifest = prewarm_mage_codec_cache(
        model_directory=model_root,
        checkpoint_manifest=checkpoint,
        codec_policy=_codec_policy(),
        cache_base_root=tmp_path / "cache",
        video_paths=[source],
        processor_factory=lambda _root: processor,
        environment_projection=_ENVIRONMENT,
    )
    return manifest, processor, source, model_root, checkpoint


def test_logical_identity_is_path_independent_and_bound_to_all_exact_inputs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "segment.mp4"
    second = tmp_path / "second" / "renamed.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    payload = b"same immutable media bytes"
    first.write_bytes(payload)
    second.write_bytes(payload)
    assert _file_digest(first) == _file_digest(second)

    inputs = {
        "source_content_sha256": _file_digest(first),
        "source_byte_count": len(payload),
        "checkpoint_manifest_sha256": _digest(1),
        "codec_policy_sha256": _digest(2),
        "recipe_sha256": _digest(3),
    }
    first_identity = mage_codec_logical_cache_identity(**inputs)
    second_identity = mage_codec_logical_cache_identity(
        **{**inputs, "source_content_sha256": _file_digest(second)}
    )
    assert first_identity == second_identity

    mutations = (
        {"source_content_sha256": _digest(4)},
        {"source_byte_count": len(payload) + 1},
        {"checkpoint_manifest_sha256": _digest(5)},
        {"codec_policy_sha256": _digest(6)},
        {"recipe_sha256": _digest(7)},
    )
    for mutation in mutations:
        assert mage_codec_logical_cache_identity(**{**inputs, **mutation}) != first_identity


def test_namespace_is_qualified_by_checkpoint_policy_and_effective_recipe() -> None:
    inputs = {
        "checkpoint_manifest_sha256": _digest(1),
        "codec_policy_sha256": _digest(2),
        "recipe_sha256": _digest(3),
    }
    identity = mage_codec_namespace_identity(**inputs)

    assert mage_codec_namespace_identity(**inputs) == identity
    assert (
        mage_codec_namespace_identity(**{**inputs, "checkpoint_manifest_sha256": _digest(4)})
        != identity
    )
    assert (
        mage_codec_namespace_identity(**{**inputs, "codec_policy_sha256": _digest(5)}) != identity
    )
    assert mage_codec_namespace_identity(**{**inputs, "recipe_sha256": _digest(6)}) != identity


def test_first_prewarm_builds_and_second_is_verified_hit_without_processor(
    tmp_path: Path,
) -> None:
    first, processor, source, model_root, checkpoint = _build_once(tmp_path)

    assert first.built_count == 1
    assert first.verified_hit_count == 0
    assert first.entries[0].admission == "BUILT"
    assert len(processor.calls) == 1
    assert Path(first.qualified_cache_root).name == first.namespace_identity
    assert Path(first.qualified_cache_root).parent == (tmp_path / "cache").resolve()

    def fail_if_loaded(_root: Path) -> Any:
        raise AssertionError("processor must not load for a verified cache hit")

    second = prewarm_mage_codec_cache(
        model_directory=model_root,
        checkpoint_manifest=checkpoint,
        codec_policy=_codec_policy(),
        cache_base_root=tmp_path / "cache",
        video_paths=[source],
        processor_factory=fail_if_loaded,
        environment_projection=_ENVIRONMENT,
    )

    assert second.built_count == 0
    assert second.verified_hit_count == 1
    assert second.entries[0].admission == "VERIFIED_HIT"
    assert second.entries[0].logical_cache_identity == first.entries[0].logical_cache_identity
    assert second.entries[0].entry_semantic_sha256 == first.entries[0].entry_semantic_sha256
    assert len(processor.calls) == 1


@pytest.mark.parametrize("corruption", ["sidecar", "asset", "source"])
def test_verification_fails_closed_for_sidecar_asset_or_source_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    manifest, _processor, source, _model_root, _checkpoint = _build_once(tmp_path)
    provider = Path(manifest.entries[0].provider_cache_directory)

    if corruption == "sidecar":
        sidecar = mage_codec_cache_sidecar_path(provider)
        sidecar.write_bytes(sidecar.read_bytes() + b"\n")
        match = "canonical JSON"
    elif corruption == "asset":
        canvas = provider / "canvas_000.jpg"
        canvas.write_bytes(canvas.read_bytes() + b"tampered")
        match = "asset bytes"
    else:
        source.write_bytes(source.read_bytes() + b"tampered")
        match = "source content"

    with pytest.raises(MageCodecCacheError, match=match):
        verify_mage_codec_cache_manifest(manifest=manifest)


def test_prewarm_rejects_unbound_provider_directory_in_qualified_namespace(
    tmp_path: Path,
) -> None:
    model_root, checkpoint = _write_model_tree(tmp_path)
    policy = _codec_policy()
    recipe = build_mage_codec_effective_recipe(
        model_directory=model_root,
        codec_policy=policy,
        environment_projection=_ENVIRONMENT,
    )
    policy_identity = build_mage_video_codec_policy_identity(policy)
    namespace = mage_codec_namespace_identity(
        checkpoint_manifest_sha256=checkpoint.manifest_sha256,
        codec_policy_sha256=policy_identity.policy_sha256,
        recipe_sha256=recipe.semantic_sha256,
    )
    cache_root = tmp_path / "cache"
    (cache_root / namespace / "provider-without-sidecar").mkdir(parents=True)
    source = tmp_path / "segment.mp4"
    source.write_bytes(b"video")

    with pytest.raises(MageCodecCacheError, match="unbound provider directory"):
        prewarm_mage_codec_cache(
            model_directory=model_root,
            checkpoint_manifest=checkpoint,
            codec_policy=policy,
            cache_base_root=cache_root,
            video_paths=[source],
            processor_factory=lambda _root: pytest.fail("processor must not load"),
            environment_projection=_ENVIRONMENT,
        )


def test_manifest_canonical_write_load_and_exact_verification(tmp_path: Path) -> None:
    manifest, _processor, _source, _model_root, _checkpoint = _build_once(tmp_path)
    path = tmp_path / "evidence" / "codec-cache-manifest.json"

    write_mage_codec_cache_manifest(manifest=manifest, path=path)

    assert path.read_bytes() == canonical_json_bytes(manifest.model_dump(mode="json"))
    loaded = load_mage_codec_cache_manifest(path=path)
    assert loaded == manifest
    verified = verify_mage_codec_cache_manifest(manifest=loaded)
    assert len(verified) == 1
    assert verified[0].entry_semantic_sha256 == manifest.entries[0].entry_semantic_sha256

    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    with pytest.raises(MageCodecCacheError, match="canonical JSON"):
        load_mage_codec_cache_manifest(path=path)


@pytest.mark.parametrize("escape", ["qualified-root", "provider-directory"])
def test_manifest_verification_rejects_cache_path_escape(
    tmp_path: Path,
    escape: str,
) -> None:
    manifest, _processor, _source, _model_root, _checkpoint = _build_once(tmp_path)
    if escape == "qualified-root":
        escaped_root = tmp_path / "outside" / manifest.namespace_identity
        tampered = manifest.model_copy(update={"qualified_cache_root": str(escaped_root.resolve())})
        match = "outside cache_base_root"
    else:
        escaped_provider = tmp_path / "outside-provider" / "provider"
        tampered_entry = manifest.entries[0].model_copy(
            update={"provider_cache_directory": str(escaped_provider.resolve())}
        )
        tampered = manifest.model_copy(update={"entries": (tampered_entry,)})
        match = "outside qualified cache root"

    with pytest.raises(MageCodecCacheError, match=match):
        verify_mage_codec_cache_manifest(manifest=tampered)
