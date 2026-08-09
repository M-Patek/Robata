from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from robata.contracts.hashing import canonical_json_bytes
from robata.inference.mage_checkpoint_identity import (
    build_mage_checkpoint_manifest,
    write_mage_checkpoint_manifest,
)
from scripts import build_qualified_mage_dcvc_provider_v2 as cli


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "Mage-VL"
    for relative, payload in {
        "config.json": b'{"model_type":"mage_vl"}',
        "model.safetensors": b"weights",
        "modeling_mage_vl.py": b"class Model: pass\n",
        "preprocessor_config.json": canonical_json_bytes({"codec": {"dcvc": {"max_side": 0}}}),
        "neural_codec/dcvc_rt_intra.tar": b"intra",
        "neural_codec/dcvc_rt_inter.tar": b"inter",
        "neural_codec/dcvc_rt_engine.py": b"class Engine: pass\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = build_mage_checkpoint_manifest(
        model_directory=root,
        model_identifier="Mage-VL",
        model_revision="source",
    )
    manifest_path = tmp_path / "source-checkpoint.json"
    write_mage_checkpoint_manifest(manifest=manifest, manifest_path=manifest_path)
    return root, manifest_path


def test_run_builds_qualified_tree_and_both_manifests(tmp_path: Path) -> None:
    source, source_manifest = _tree(tmp_path)
    protocol = tmp_path / "protocol.py"
    worker = tmp_path / "worker.py"
    protocol.write_text("VERSION='v2'\n", encoding="utf-8")
    worker.write_text("def main(): return 0\n", encoding="utf-8")
    target = tmp_path / "qualified"
    qualification = tmp_path / "evidence" / "qualified.json"
    checkpoint = tmp_path / "evidence" / "checkpoint.json"

    payload = cli.run(
        Namespace(
            source_model_dir=source,
            source_checkpoint_manifest=source_manifest,
            target_model_dir=target,
            qualified_model_identifier="Mage-VL-Robata-DCVC-V2",
            qualified_model_revision="source+robata-dcvc-provider-v2-20260809",
            provider_source_file=[protocol, worker],
            qualification_manifest=qualification,
            checkpoint_manifest=checkpoint,
            copy_mode="copy",
        )
    )

    assert payload["ok"] is True
    assert payload["source_model_unchanged"] is True
    assert payload["production_eligible"] is False
    assert payload["qualified_model_revision"].endswith("20260809")
    assert target.is_dir()
    assert qualification.is_file()
    assert checkpoint.is_file()
