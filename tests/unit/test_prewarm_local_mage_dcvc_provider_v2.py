from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts import prewarm_local_mage_dcvc_provider_v2 as cli


def test_cli_policy_preserves_explicit_provider_v2_semantics() -> None:
    policy = cli._codec_policy(
        Namespace(
            preparation_device="cuda",
            target_canvas=8,
            group_size=8,
            images_per_group=1,
            max_pixels=65_536,
            min_group_frames=8,
            max_group_frames=128,
            timeout_seconds=7200,
            max_side=448,
            neural_qp=42,
            neural_reset_interval=64,
            neural_intra_period=-1,
            readiness_coverage_bins=3,
            readiness_delta_ratio=0.05,
            bitcost_percentile=99,
            decode_backsearch_max=16,
        )
    )

    assert policy.target_canvas == 8
    assert policy.group_size == 8
    assert policy.images_per_group == 1
    assert policy.max_pixels == 65_536
    assert policy.max_group_frames == 128
    assert policy.neural_parameters is not None
    assert policy.neural_parameters.max_side == 448
    assert policy.neural_parameters.sequence_length_frames == 0
    assert policy.neural_parameters.canvas_token_side is None


def test_cli_defaults_to_locally_qualified_448_with_explicit_full_resolution_rollback() -> None:
    parser = cli._parser()

    assert parser.get_default("max_side") == 448

    action = next(item for item in parser._actions if item.dest == "max_side")
    assert "pass 0 explicitly" in (action.help or "")
    assert action.type is not None
    assert action.type("0") == 0


def test_cli_rejects_qualification_exact_sha_pin_before_prewarm(tmp_path: Path) -> None:
    qualification = tmp_path / "qualified-provider.json"
    qualification.write_bytes(b"not-the-pinned-bytes")

    with pytest.raises(cli.MageDcvcPrewarmCliError, match="qualification manifest"):
        cli.run(
            Namespace(
                model_dir=tmp_path / "model",
                qualified_provider_manifest=qualification,
                qualification_manifest_sha256="0" * 64,
            )
        )
