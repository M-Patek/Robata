from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_local_mage_dcvc_provider_v2_generation.py"
)


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "provider_v2_generation_harness_test", SCRIPT
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _sha(seed: int) -> str:
    return f"{seed:064x}"


def _fixture(tmp_path: Path, *, sequence_length_frames: int = 0) -> list[str]:
    model = tmp_path / "model"
    model.mkdir()
    materialization = tmp_path / "materialization"
    segments = materialization / "segments"
    segments.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    guard = tmp_path / "guard" / "cuda-0.lock"
    guard.parent.mkdir()

    checkpoint_sha = _sha(1)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "manifest_version": "mage-checkpoint-manifest-v2",
                "model_identifier": "Mage-VL-Robata-DCVC-V2",
                "model_revision": "local-test-r1",
                "manifest_sha256": checkpoint_sha,
            }
        ),
        encoding="utf-8",
    )
    qualified_path = tmp_path / "qualified.json"
    qualified_path.write_text(
        json.dumps(
            {
                "manifest_version": "mage-dcvc-qualified-provider-manifest-v2",
                "qualified_model_directory": str(model.resolve()),
                "bundle": {
                    "qualified_model_identifier": "Mage-VL-Robata-DCVC-V2",
                    "qualified_model_revision": "local-test-r1",
                },
                "qualified_checkpoint_manifest": {"manifest_sha256": checkpoint_sha},
            }
        ),
        encoding="utf-8",
    )
    entries: list[dict[str, object]] = []
    for ordinal in range(5):
        segment = segments / f"{ordinal:06d}.mp4"
        segment.write_bytes(f"segment-{ordinal}".encode())
        entries.append({"source_path": str(segment.resolve())})
    effective = {
        "provider_version": "robata-mage-dcvc-provider-v2",
        "recipe_version": "mage-dcvc-readiness-explicit-v2",
        "device_concurrency_policy": "exclusive-shared-device-v1",
        "sequence_length_frames": sequence_length_frames,
        "canvas_token_side": None,
        "encoded_frame_extent": "through-last-sampled-frame",
        "preparation_device": "cuda",
        "provider_implementation_sha256": _sha(2),
        "effective_config_sha256": _sha(3),
        "target_canvas": 8,
        "group_size": 8,
        "images_per_group": 1,
        "patch": 16,
        "max_pixels": 65_536,
        "min_group_frames": 8,
        "max_group_frames": 128,
        "qp": 42,
        "reset_interval": 64,
        "intra_period": -1,
        "max_side": 448,
        "readiness_coverage_bins": 3,
        "readiness_delta_ratio": 0.05,
        "bitcost_percentile": 99,
        "decode_backsearch_max": 16,
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "manifest_version": "mage-codec-cache-manifest-v2",
                "recipe_version": "mage-dcvc-readiness-explicit-v2",
                "checkpoint_manifest_sha256": checkpoint_sha,
                "provider_implementation_sha256": _sha(2),
                "manifest_semantic_sha256": _sha(4),
                "namespace_identity": _sha(5),
                "effective_config": effective,
                "entry_count": 5,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return [
        "--model-dir",
        str(model),
        "--qualified-provider-manifest",
        str(qualified_path),
        "--checkpoint-manifest-path",
        str(checkpoint_path),
        "--codec-cache-manifest",
        str(cache_path),
        "--shared-device-guard-file",
        str(guard),
        "--source",
        str(source),
        "--materialization-dir",
        str(materialization),
        "--output-root",
        str(tmp_path / "output"),
        "--recording-key",
        "mage-native-sustained-control-20260808",
        "--python",
        str(Path(sys.executable)),
    ]


def test_plan_binds_provider_v2_and_single_generation_lane(tmp_path: Path) -> None:
    module = _module()
    arguments = module._parser().parse_args(_fixture(tmp_path))
    inputs = module.load_provider_v2_generation_inputs(arguments)
    paths = module.generation_paths(arguments.output_root)

    plan = module.build_plan_document(arguments=arguments, inputs=inputs, paths=paths)
    endpoint = plan["commands"]["endpoint"]
    stream = plan["commands"]["stream"]

    assert plan["authority"] == "LOCAL_QUALIFICATION_NON_PRODUCTION"
    assert plan["production_eligible"] is False
    assert plan["execution"]["provider_preparation_started_by_harness"] is False
    assert plan["execution"]["codec_generation_overlap_allowed"] is False
    assert plan["execution"]["generation_inflight_limit"] == 1
    assert plan["recurrent_work"]["sequence_length_is_compute_cap"] is False
    assert "--require-provider-v2-cache" in endpoint
    assert "--require-verified-codec-cache" in endpoint
    assert endpoint[endpoint.index("--shared-device-guard-file") + 1] == str(
        inputs.shared_device_guard_file
    )
    assert endpoint[endpoint.index("--neural-max-side") + 1] == "448"
    assert endpoint[endpoint.index("--neural-sequence-length-frames") + 1] == "0"
    assert "--neural-canvas-token-side" not in endpoint
    assert stream[stream.index("--max-inflight-observations") + 1] == "1"
    assert stream[stream.index("--segment-boundary-mode") + 1] == "keyframe_aligned"
    assert stream[stream.index("--ffmpeg-binary") + 1] == "ffmpeg"
    assert stream[stream.index("--ffprobe-binary") + 1] == "ffprobe"


def test_plan_only_writes_canonical_result_without_starting_endpoint(tmp_path: Path) -> None:
    module = _module()
    argv = [*_fixture(tmp_path), "--plan-only"]

    assert module.main(argv) == 0

    output = tmp_path / "output" / "harness-result.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PLANNED"
    assert payload["execution"]["camera_count"] == 1
    assert payload["sample"]["segment_count"] == 5
    assert not (tmp_path / "output" / "endpoint-state").exists()
    assert output.read_bytes() == module._lossless_json_bytes(payload)


def test_rejects_sequence_length_as_a_bounded_work_claim(tmp_path: Path) -> None:
    module = _module()
    arguments = module._parser().parse_args(_fixture(tmp_path, sequence_length_frames=8))

    with pytest.raises(
        module.ProviderV2GenerationHarnessError,
        match="sequence_length_frames=0",
    ):
        module.load_provider_v2_generation_inputs(arguments)


def test_rejects_cache_source_outside_materialization_root(tmp_path: Path) -> None:
    module = _module()
    argv = _fixture(tmp_path)
    cache_path = Path(argv[argv.index("--codec-cache-manifest") + 1])
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    cache["entries"][2]["source_path"] = str(outside.resolve())
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    arguments = module._parser().parse_args(argv)

    with pytest.raises(
        module.ProviderV2GenerationHarnessError,
        match="outside materialization directory",
    ):
        module.load_provider_v2_generation_inputs(arguments)


def test_actual_run_refuses_nonempty_output_root(tmp_path: Path) -> None:
    module = _module()
    paths = module.generation_paths(tmp_path / "output")
    paths.root.mkdir()
    (paths.root / "prior.json").write_text("{}", encoding="utf-8")

    with pytest.raises(
        module.ProviderV2GenerationHarnessError,
        match="replay cannot contaminate timing",
    ):
        module._prepare_actual_root(paths)


def test_health_ready_accepts_endpoint_uppercase_status() -> None:
    module = _module()

    module._require_ready_health({"status": "READY"})


def test_health_ready_rejects_non_ready_status() -> None:
    module = _module()

    with pytest.raises(module.ProviderV2GenerationHarnessError, match="not READY"):
        module._require_ready_health({"status": "LOADING"})
