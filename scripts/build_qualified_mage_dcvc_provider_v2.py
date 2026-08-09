#!/usr/bin/env python3
"""Build a separate, checkpoint-bound Mage tree for Robata DCVC Provider V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.inference.mage_checkpoint_identity import (  # noqa: E402
    load_mage_checkpoint_manifest,
)
from robata.inference.mage_dcvc_qualified_provider import (  # noqa: E402
    qualify_mage_dcvc_provider_v2,
    verify_mage_dcvc_qualified_provider,
    write_qualified_checkpoint_manifest,
)

_DEFAULT_PROVIDER_SOURCES = (
    ROOT / "src" / "robata" / "inference" / "device_execution_guard.py",
    ROOT / "src" / "robata" / "inference" / "mage_dcvc_preparation_protocol.py",
    ROOT / "src" / "robata" / "inference" / "mage_dcvc_preparation_worker.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--target-model-dir", type=Path, required=True)
    parser.add_argument("--qualified-model-identifier", default="Mage-VL-Robata-DCVC-V2")
    parser.add_argument("--qualified-model-revision", required=True)
    parser.add_argument(
        "--provider-source-file",
        type=Path,
        action="append",
        default=None,
        help="repeat to override the default protocol+worker source bundle",
    )
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help=(
            "copy is the durable default; hardlink is an explicit local space-saving mode whose "
            "source tree must remain immutable"
        ),
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    source_checkpoint = load_mage_checkpoint_manifest(
        manifest_path=arguments.source_checkpoint_manifest
    )
    provider_sources = tuple(arguments.provider_source_file or _DEFAULT_PROVIDER_SOURCES)
    manifest = qualify_mage_dcvc_provider_v2(
        source_model_directory=arguments.source_model_dir,
        source_checkpoint_manifest=source_checkpoint,
        target_model_directory=arguments.target_model_dir,
        qualified_model_identifier=arguments.qualified_model_identifier,
        qualified_model_revision=arguments.qualified_model_revision,
        provider_source_files=provider_sources,
        manifest_path=arguments.qualification_manifest,
        copy_mode=arguments.copy_mode,
    )
    verify_mage_dcvc_qualified_provider(manifest=manifest)
    write_qualified_checkpoint_manifest(
        manifest=manifest,
        path=arguments.checkpoint_manifest,
    )
    return {
        "ok": True,
        "provider_version": manifest.provider_version,
        "copy_mode": manifest.copy_mode,
        "source_checkpoint_manifest_sha256": (manifest.bundle.source_checkpoint_manifest_sha256),
        "qualified_model_directory": manifest.qualified_model_directory,
        "qualified_model_identifier": manifest.bundle.qualified_model_identifier,
        "qualified_model_revision": manifest.bundle.qualified_model_revision,
        "qualified_checkpoint_manifest_sha256": (
            manifest.qualified_checkpoint_manifest.manifest_sha256
        ),
        "qualified_checkpoint_manifest_path": str(
            arguments.checkpoint_manifest.expanduser().resolve()
        ),
        "qualification_manifest_path": str(arguments.qualification_manifest.expanduser().resolve()),
        "qualification_manifest_semantic_sha256": manifest.manifest_semantic_sha256,
        "provider_bundle_semantic_sha256": manifest.bundle.bundle_semantic_sha256,
        "provider_files": [item.model_dump(mode="json") for item in manifest.bundle.provider_files],
        "source_model_unchanged": True,
        "production_eligible": False,
    }


def main() -> int:
    try:
        payload = run(_parser().parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
