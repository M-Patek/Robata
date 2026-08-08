from __future__ import annotations

import json
from pathlib import Path

import pytest

from robata.contracts.hashing import canonical_json_bytes
from robata.inference.mage_checkpoint_identity import (
    MAGE_CHECKPOINT_MANIFEST_VERSION,
    MageCheckpointIdentityError,
    MageCheckpointManifest,
    build_mage_checkpoint_manifest,
    load_mage_checkpoint_manifest,
    verify_mage_checkpoint_manifest,
    write_mage_checkpoint_manifest,
)


def _write_execution_tree(tmp_path: Path) -> Path:
    root = tmp_path / "Mage-VL"
    root.mkdir()
    included_payloads = {
        "config.json": b'{"model_type":"mage_vl"}',
        "tokenizer.json": b'{"version":"1"}',
        "chat_template.jinja": b"{{ messages }}",
        "processing_mage_vl.py": b"class MageProcessor: pass\n",
        "codec_video_processing_mage_vl.py": b"def process(): pass\n",
        "model.safetensors.index.json": b'{"weight_map":{}}',
        "model-00001-of-00001.safetensors": b"model-weights",
        "streammind_gate.py": b"def gate(): pass\n",
        "streammind_gate.safetensors": b"gate-weights",
        "neural_codec/codec_loader.py": b"def load_codec(): pass\n",
        "neural_codec/DCVC/src/layers/extensions/inference/kernel.cu": b"// native codec\n",
        "neural_codec/dcvc_rt_intra.tar": b"codec-checkpoint",
    }
    for relative_path, payload in included_payloads.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    for relative_path, payload in {
        "README.md": b"documentation",
        "notes.txt": b"incidental",
        "metadata.json": b"incidental",
        ".cache/ignored.safetensors": b"cache-weights",
        "assets/cover.png": b"asset",
        "examples/quickstart.py": b"example",
        "neural_codec/__pycache__/codec_loader.pyc": b"compiled",
        "neural_codec/README.md": b"codec documentation",
    }.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return root


def _build_manifest(root: Path) -> MageCheckpointManifest:
    return build_mage_checkpoint_manifest(
        model_directory=root,
        model_identifier="Mage-VL",
        model_revision="test-revision",
    )


def test_manifest_hashes_all_mage_execution_inputs_and_excludes_incidental_files(
    tmp_path: Path,
) -> None:
    root = _write_execution_tree(tmp_path)

    manifest = _build_manifest(root)

    expected_paths = {
        "chat_template.jinja",
        "codec_video_processing_mage_vl.py",
        "config.json",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "neural_codec/DCVC/src/layers/extensions/inference/kernel.cu",
        "neural_codec/codec_loader.py",
        "neural_codec/dcvc_rt_intra.tar",
        "processing_mage_vl.py",
        "streammind_gate.py",
        "streammind_gate.safetensors",
        "tokenizer.json",
    }
    assert manifest.manifest_version == MAGE_CHECKPOINT_MANIFEST_VERSION
    assert manifest.model_identifier == "Mage-VL"
    assert manifest.model_revision == "test-revision"
    assert {entry.path for entry in manifest.files} == expected_paths
    assert manifest.included_file_count == len(expected_paths)
    assert manifest.total_byte_count == sum(entry.byte_count for entry in manifest.files)
    assert all("\\" not in entry.path for entry in manifest.files)

    verify_mage_checkpoint_manifest(manifest=manifest, model_directory=root)


def test_manifest_round_trip_is_canonical_and_self_validating(tmp_path: Path) -> None:
    root = _write_execution_tree(tmp_path)
    manifest = _build_manifest(root)
    manifest_path = tmp_path / "mage-checkpoint-manifest.json"

    write_mage_checkpoint_manifest(manifest=manifest, manifest_path=manifest_path)

    assert manifest_path.read_bytes() == canonical_json_bytes(manifest.model_dump(mode="json"))
    assert load_mage_checkpoint_manifest(manifest_path=manifest_path) == manifest

    tampered = manifest.model_dump(mode="json")
    tampered["total_byte_count"] = manifest.total_byte_count + 1
    manifest_path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(MageCheckpointIdentityError, match="pinned contract"):
        load_mage_checkpoint_manifest(manifest_path=manifest_path)

    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    with pytest.raises(MageCheckpointIdentityError, match="canonical JSON"):
        load_mage_checkpoint_manifest(manifest_path=manifest_path)


def test_verifier_detects_changed_missing_and_unexpected_included_files(tmp_path: Path) -> None:
    root = _write_execution_tree(tmp_path)
    manifest = _build_manifest(root)

    weights = root / "model-00001-of-00001.safetensors"
    weights.write_bytes(b"changed-model-weights")
    with pytest.raises(MageCheckpointIdentityError, match="changed included files"):
        verify_mage_checkpoint_manifest(manifest=manifest, model_directory=root)

    weights.write_bytes(b"model-weights")
    config = root / "config.json"
    config.unlink()
    with pytest.raises(MageCheckpointIdentityError, match="missing included files"):
        verify_mage_checkpoint_manifest(manifest=manifest, model_directory=root)

    config.write_bytes(b'{"model_type":"mage_vl"}')
    unexpected = root / "neural_codec" / "new_runtime.py"
    unexpected.write_bytes(b"def changed_execution(): pass\n")
    with pytest.raises(MageCheckpointIdentityError, match="unexpected included files"):
        verify_mage_checkpoint_manifest(manifest=manifest, model_directory=root)


def test_loader_rejects_path_traversal_in_a_canonical_manifest(tmp_path: Path) -> None:
    root = _write_execution_tree(tmp_path)
    manifest = _build_manifest(root)
    manifest_path = tmp_path / "mage-checkpoint-manifest.json"
    traversal = manifest.model_dump(mode="json")
    traversal["files"][0]["path"] = "../outside.py"
    manifest_path.write_bytes(canonical_json_bytes(traversal))

    with pytest.raises(MageCheckpointIdentityError, match="pinned contract"):
        load_mage_checkpoint_manifest(manifest_path=manifest_path)


def test_builder_rejects_symlink_escaping_model_root(tmp_path: Path) -> None:
    root = _write_execution_tree(tmp_path)
    external = tmp_path / "outside.py"
    external.write_bytes(b"outside")
    symlink = root / "neural_codec" / "escape.py"
    try:
        symlink.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is unavailable in this test environment")

    with pytest.raises(MageCheckpointIdentityError, match="symlink escapes model root"):
        _build_manifest(root)
