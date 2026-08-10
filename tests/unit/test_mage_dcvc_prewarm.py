from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from robata.contracts.hashing import canonical_json_bytes
from robata.inference import mage_dcvc_preparation_protocol, mage_dcvc_preparation_worker
from robata.inference.mage_checkpoint_identity import build_mage_checkpoint_manifest
from robata.inference.mage_dcvc_preparation_protocol import (
    MAGE_DCVC_PREPARATION_SIDECAR_NAME,
    MageDcvcPreparationArtifact,
    MageDcvcPreparationRequest,
    MageDcvcPreparationResponse,
    MageDcvcPreparedAsset,
    mage_dcvc_artifact_semantic_sha256,
    mage_dcvc_preparation_identity,
)
from robata.inference.mage_dcvc_preparation_worker import build_mage_dcvc_effective_config
from robata.inference.mage_dcvc_prewarm import (
    MageDcvcPreparationProcessSession,
    MageDcvcPrewarmError,
    MageDcvcWorkerProcessTelemetryV2,
    _bounded_worker_error_detail,
    _validate_windows_cache_output_paths,
    load_mage_dcvc_prewarm_report_v2,
    prewarm_mage_dcvc_provider_v2,
    run_mage_dcvc_preparation_process,
)
from robata.inference.mage_dcvc_qualified_provider import qualify_mage_dcvc_provider_v2
from robata.inference.mage_video_endpoint import (
    MageVideoCodecPolicy,
    MageVideoNeuralCodecParameters,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _session_request(tmp_path: Path, index: int = 0) -> MageDcvcPreparationRequest:
    source_sha = str(index + 1) * 64
    identity = mage_dcvc_preparation_identity(
        source_content_sha256=source_sha,
        source_byte_count=1,
        effective_config_sha256="a" * 64,
    )
    return MageDcvcPreparationRequest(
        request_id=f"r{index}",
        source_path=str(tmp_path / f"s{index}.mp4"),
        source_content_sha256=source_sha,
        source_byte_count=1,
        output_relative_path=f"cache/{index}",
        effective_config_sha256="a" * 64,
        preparation_identity=identity,
    )


def _session_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    return environment


def _source_model(tmp_path: Path) -> tuple[Path, Any]:
    root = tmp_path / "Mage-VL"
    files = {
        "config.json": b'{"model_type":"mage_vl"}',
        "preprocessor_config.json": b'{"codec":{"dcvc":{"provider":"test"}}}',
        "modeling_mage_vl.py": b"class Mage: pass\n",
        "model.safetensors": b"test-weight",
        "neural_codec/codec_dcvc_config.py": b"DCVC_DEVICE = 'cpu'\n",
        "neural_codec/dcvc_readiness_gen.py": b"def main(): pass\n",
        "neural_codec/dcvc_rt_engine.py": b"class Engine: pass\n",
        "neural_codec/codec_tools/pipeline/process_video_bitcost_readiness.py": b"# r\n",
        "neural_codec/codec_tools/pipeline/process_video_bitcost_mv_mask_collage.py": b"# m\n",
        "neural_codec/codec_tools/pipeline/generate_codec_patch_smart_resize.py": b"# s\n",
        "neural_codec/DCVC/src/__init__.py": b"# dcvc\n",
        "neural_codec/codec_tools/codec_patch_gop/__init__.py": b"# gop\n",
        "neural_codec/dcvc_rt_intra.tar": b"intra-checkpoint",
        "neural_codec/dcvc_rt_inter.tar": b"inter-checkpoint",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    checkpoint = build_mage_checkpoint_manifest(
        model_directory=root,
        model_identifier="Mage-VL",
        model_revision="test-upstream",
    )
    return root, checkpoint


def _qualification(tmp_path: Path) -> tuple[Any, Any, tuple[Path, ...]]:
    source, checkpoint = _source_model(tmp_path)
    provider_sources = (
        Path(mage_dcvc_preparation_protocol.__file__).resolve(),
        Path(mage_dcvc_preparation_worker.__file__).resolve(),
    )
    qualification = qualify_mage_dcvc_provider_v2(
        source_model_directory=source,
        source_checkpoint_manifest=checkpoint,
        target_model_directory=tmp_path / "Mage-VL-Robata-DCVC-V2",
        qualified_model_identifier="Mage-VL-Robata-DCVC-V2",
        qualified_model_revision="test-upstream+robata-dcvc-v2",
        provider_source_files=provider_sources,
        manifest_path=tmp_path / "qualified-provider.json",
    )
    return qualification, qualification.qualified_checkpoint_manifest, provider_sources


def _policy() -> MageVideoCodecPolicy:
    return MageVideoCodecPolicy(
        codec_mode="neural",
        preprocess_device="cpu",
        target_canvas=32,
        group_size=32,
        images_per_group=4,
        patch_size=16,
        max_pixels=150_000,
        min_group_frames=8,
        max_group_frames=128,
        timeout_seconds=60,
        neural_parameters=MageVideoNeuralCodecParameters(
            max_side=448,
            sequence_length_frames=0,
            canvas_token_side=None,
        ),
    )


def _artifact(
    *, directory: Path, request: Any, config: Any, completed_job_count: int
) -> MageDcvcPreparationArtifact:
    directory.mkdir(parents=True, exist_ok=False)
    provider_metadata = {
        "provider_version": config.provider_version,
        "recipe_version": config.recipe_version,
        "effective_config_sha256": config.effective_config_sha256,
        "provider_implementation_sha256": config.provider_implementation_sha256,
        "engine": config.engine,
        "preparation_device": config.preparation_device,
        "device_concurrency_policy": config.device_concurrency_policy,
        "max_side": config.max_side,
        "configured_sampled_frame_count": config.sampled_frame_count,
        "effective_sampled_frame_count": 40,
        "max_encoded_frame_id": 79,
        "engine_load_count": 1,
        "engine_load_seconds": 0.75,
        "worker_completed_job_count": completed_job_count,
        "sequence_reset_count_for_job": 1,
        "sequence_length_frames": config.sequence_length_frames,
        "canvas_token_side": config.canvas_token_side,
        "encoded_frame_extent": config.encoded_frame_extent,
        "segment_state_policy": "reset-per-job",
    }
    payloads = {
        "canvas-000.jpg": b"jpeg",
        "meta.json": canonical_json_bytes(
            {
                "canvas_files": ["canvas-000.jpg"],
                "robata_dcvc_provider": provider_metadata,
            }
        ),
        "src_patch_position.npy": b"npy",
    }
    assets = []
    for relative, payload in sorted(payloads.items()):
        (directory / relative).write_bytes(payload)
        assets.append(
            MageDcvcPreparedAsset(
                relative_path=relative,
                byte_count=len(payload),
                sha256=_sha(payload),
            )
        )
    values = {
        "preparation_identity": request.preparation_identity,
        "effective_config_sha256": config.effective_config_sha256,
        "provider_implementation_sha256": config.provider_implementation_sha256,
        "source_content_sha256": request.source_content_sha256,
        "source_byte_count": request.source_byte_count,
        "assets": tuple(assets),
        "provider_metadata": provider_metadata,
    }
    provisional = MageDcvcPreparationArtifact.model_construct(
        **values,
        artifact_semantic_sha256="0" * 64,
    )
    artifact = MageDcvcPreparationArtifact(
        **values,
        artifact_semantic_sha256=mage_dcvc_artifact_semantic_sha256(provisional),
    )
    (directory / MAGE_DCVC_PREPARATION_SIDECAR_NAME).write_bytes(
        canonical_json_bytes(artifact.model_dump(mode="json"))
    )
    return artifact


class _ArtifactRunner:
    def __init__(self, *, config: Any, status: str = "BUILT") -> None:
        self.config = config
        self.status = status
        self.call_count = 0
        self.request_count = 0

    def __call__(
        self,
        *,
        command: Any,
        requests: Any,
        environment: Any,
        working_directory: Path,
        response_timeout_seconds: float,
    ) -> tuple[tuple[MageDcvcPreparationResponse, ...], MageDcvcWorkerProcessTelemetryV2]:
        del environment, working_directory, response_timeout_seconds
        self.call_count += 1
        self.request_count += len(requests)
        assert command.count("-m") == 1
        assert command[command.index("-m") + 1].endswith("mage_dcvc_preparation_worker")
        output_root = Path(command[command.index("--output-root") + 1]).resolve()
        responses = []
        built_count = 0
        for request in requests:
            output = output_root.joinpath(*PurePosixPath(request.output_relative_path).parts)
            if self.status == "BUILT":
                built_count += 1
                artifact = _artifact(
                    directory=output,
                    request=request,
                    config=self.config,
                    completed_job_count=built_count,
                )
                status = "BUILT"
            elif self.status == "VERIFIED_HIT":
                raw = (output / MAGE_DCVC_PREPARATION_SIDECAR_NAME).read_bytes()
                artifact = MageDcvcPreparationArtifact.model_validate_json(raw, strict=True)
                status = "VERIFIED_HIT"
            else:
                responses.append(
                    MageDcvcPreparationResponse(
                        request_id=request.request_id,
                        status="FAILED",
                        wall_seconds=0.1,
                        error_code="FAKE_REJECTION",
                        error_message="fake failure",
                    )
                )
                continue
            responses.append(
                MageDcvcPreparationResponse(
                    request_id=request.request_id,
                    status=status,
                    preparation_identity=request.preparation_identity,
                    artifact_semantic_sha256=artifact.artifact_semantic_sha256,
                    output_directory=str(output),
                    wall_seconds=1.0,
                )
            )
        return tuple(responses), MageDcvcWorkerProcessTelemetryV2(
            response_count=len(responses),
            wall_seconds=2.0,
            stderr_byte_count=0,
            stderr_sha256=_sha(b""),
        )


def _setup(tmp_path: Path) -> dict[str, Any]:
    qualification, checkpoint, provider_sources = _qualification(tmp_path)
    model = Path(qualification.qualified_model_directory)
    config = build_mage_dcvc_effective_config(
        model_directory=model,
        preparation_device="cpu",
        device_concurrency_policy="separate-device-v1",
        max_side=448,
    )
    segments = tmp_path / "segments"
    segments.mkdir()
    videos = []
    for index in range(2):
        video = segments / f"segment-{index:03d}.mp4"
        video.write_bytes(f"video-{index}".encode())
        videos.append(video)
    return {
        "qualification": qualification,
        "checkpoint": checkpoint,
        "provider_sources": provider_sources,
        "model": model,
        "config": config,
        "videos": tuple(videos),
        "manifest": tmp_path / "evidence" / "cache-v2.json",
        "report": tmp_path / "evidence" / "prewarm-v2.json",
        "state": tmp_path / "provider-state",
        "cache": tmp_path / "cache",
    }


def _run(setup: dict[str, Any], runner: _ArtifactRunner) -> Any:
    return prewarm_mage_dcvc_provider_v2(
        qualified_provider_manifest=setup["qualification"],
        checkpoint_manifest=setup["checkpoint"],
        codec_policy=_policy(),
        effective_config=setup["config"],
        model_directory=setup["model"],
        provider_source_files=setup["provider_sources"],
        provider_state_root=setup["state"],
        cache_base_root=setup["cache"],
        source_paths=setup["videos"],
        cache_manifest_output=setup["manifest"],
        report_output=setup["report"],
        generation_device="cuda:0",
        shared_device_guard_file=None,
        worker_python=Path(sys.executable),
        response_timeout_seconds=60.0,
        worker_process_runner=runner,
    )


def test_prewarm_uses_one_resident_process_for_all_segments(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    runner = _ArtifactRunner(config=setup["config"])

    report = _run(setup, runner)

    assert runner.call_count == 1
    assert runner.request_count == 2
    assert report.worker_process.process_start_count == 1
    assert report.inferred_process_model_load_count == 1
    assert report.inferred_process_model_load_seconds == 0.75
    assert report.built_count == 2
    assert [job.worker_completed_job_count_in_artifact for job in report.jobs] == [1, 2]
    assert report.replay_mode == "fresh-build"
    assert load_mage_dcvc_prewarm_report_v2(path=setup["report"]) == report
    assert setup["manifest"].is_file()


def test_existing_manifest_replays_exact_assets_without_loading_model(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    _run(setup, _ArtifactRunner(config=setup["config"]))
    exact_manifest = setup["manifest"].read_bytes()
    runner = _ArtifactRunner(config=setup["config"], status="VERIFIED_HIT")

    replay = _run(setup, runner)

    assert runner.call_count == 1
    assert replay.replay_mode == "exact-verified-replay"
    assert replay.exact_verified_replay is True
    assert replay.inferred_process_model_load_count == 0
    assert replay.verified_hit_count == 2
    assert setup["manifest"].read_bytes() == exact_manifest


def test_source_bundle_mismatch_fails_before_worker_start(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    changed = tmp_path / "mage_dcvc_preparation_worker.py"
    changed.write_bytes(b"changed-provider")
    runner = _ArtifactRunner(config=setup["config"])
    setup["provider_sources"] = (setup["provider_sources"][0], changed)

    with pytest.raises(MageDcvcPrewarmError, match="qualification"):
        _run(setup, runner)

    assert runner.call_count == 0
    assert not setup["manifest"].exists()


def test_worker_rejection_never_publishes_manifest_or_report(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    runner = _ArtifactRunner(config=setup["config"], status="FAILED")

    with pytest.raises(MageDcvcPrewarmError, match="FAKE_REJECTION: fake failure"):
        _run(setup, runner)

    assert runner.call_count == 1
    assert not setup["manifest"].exists()
    assert not setup["report"].exists()


def test_binary_jsonl_runner_services_multiple_requests_in_one_process(tmp_path: Path) -> None:
    request_values = []
    for index in range(2):
        source_sha = str(index + 1) * 64
        identity = mage_dcvc_preparation_identity(
            source_content_sha256=source_sha,
            source_byte_count=1,
            effective_config_sha256="a" * 64,
        )
        request_values.append(
            MageDcvcPreparationRequest(
                request_id=f"r{index}",
                source_path=str(tmp_path / f"s{index}.mp4"),
                source_content_sha256=source_sha,
                source_byte_count=1,
                output_relative_path=f"cache/{index}",
                effective_config_sha256="a" * 64,
                preparation_identity=identity,
            )
        )
    requests = tuple(request_values)
    program = (
        "import sys;"
        "from robata.contracts.hashing import canonical_json_bytes;"
        "from robata.inference.mage_dcvc_preparation_protocol import "
        "MageDcvcPreparationRequest as Q,MageDcvcPreparationResponse as R;"
        "\nfor line in sys.stdin.buffer:\n"
        " q=Q.model_validate_json(line,strict=True);"
        " r=R(request_id=q.request_id,status='BUILT',preparation_identity=q.preparation_identity,"
        "artifact_semantic_sha256='b'*64,output_directory='fake-output',wall_seconds=0.0);"
        " sys.stdout.buffer.write(canonical_json_bytes(r.model_dump(mode='json'))+b'\\n');"
        " sys.stdout.buffer.flush()\n"
    )
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = source_root

    responses, telemetry = run_mage_dcvc_preparation_process(
        command=(sys.executable, "-u", "-c", program),
        requests=requests,
        environment=environment,
        working_directory=tmp_path,
        response_timeout_seconds=10.0,
    )

    assert [response.request_id for response in responses] == ["r0", "r1"]
    assert telemetry.process_start_count == 1
    assert telemetry.response_count == 2
    assert telemetry.exit_code == 0


def test_process_session_returns_first_response_before_second_request(
    tmp_path: Path,
) -> None:
    seen_path = tmp_path / "seen.txt"
    requests = (_session_request(tmp_path, 0), _session_request(tmp_path, 1))
    program = (
        "import os,sys;"
        "from robata.contracts.hashing import canonical_json_bytes;"
        "from robata.inference.mage_dcvc_preparation_protocol import "
        "MageDcvcPreparationRequest as Q,MageDcvcPreparationResponse as R;"
        f"seen=open({str(seen_path)!r},'a',encoding='utf-8',buffering=1);"
        "\nfor line in sys.stdin.buffer:\n"
        " q=Q.model_validate_json(line,strict=True);"
        " seen.write(str(os.getpid())+':'+q.request_id+'\\n');seen.flush();"
        " r=R(request_id=q.request_id,status='BUILT',preparation_identity=q.preparation_identity,"
        "artifact_semantic_sha256='b'*64,output_directory='fake-output',wall_seconds=0.0);"
        " sys.stdout.buffer.write(canonical_json_bytes(r.model_dump(mode='json'))+b'\\n');"
        " sys.stdout.buffer.flush()\n"
    )
    session = MageDcvcPreparationProcessSession(
        command=(sys.executable, "-u", "-c", program),
        environment=_session_environment(),
        working_directory=tmp_path,
        response_timeout_seconds=10.0,
    )

    first = session.prepare_one(requests[0])
    first_seen = seen_path.read_text(encoding="utf-8").splitlines()
    assert first.request_id == "r0"
    assert [line.split(":", maxsplit=1)[1] for line in first_seen] == ["r0"]

    second = session.prepare_one(requests[1])
    all_seen = seen_path.read_text(encoding="utf-8").splitlines()
    telemetry = session.close()

    assert second.request_id == "r1"
    assert [line.split(":", maxsplit=1)[1] for line in all_seen] == ["r0", "r1"]
    assert len({line.split(":", maxsplit=1)[0] for line in all_seen}) == 1
    assert telemetry.process_start_count == 1
    assert telemetry.response_count == 2
    assert session.close() == telemetry
    with pytest.raises(MageDcvcPrewarmError, match="session is closed"):
        session.prepare_one(requests[0])


def test_process_session_rejects_early_eof(tmp_path: Path) -> None:
    session = MageDcvcPreparationProcessSession(
        command=(sys.executable, "-u", "-c", "import sys;sys.stdin.buffer.readline()"),
        environment=_session_environment(),
        working_directory=tmp_path,
        response_timeout_seconds=10.0,
    )

    with pytest.raises(MageDcvcPrewarmError, match="exited before responding"):
        session.prepare_one(_session_request(tmp_path))
    with pytest.raises(MageDcvcPrewarmError, match="session is closed"):
        session.prepare_one(_session_request(tmp_path))


def test_process_session_rejects_response_timeout(tmp_path: Path) -> None:
    session = MageDcvcPreparationProcessSession(
        command=(
            sys.executable,
            "-u",
            "-c",
            "import sys,time;sys.stdin.buffer.readline();time.sleep(10)",
        ),
        environment=_session_environment(),
        working_directory=tmp_path,
        response_timeout_seconds=0.25,
    )

    with pytest.raises(MageDcvcPrewarmError, match="response timed out"):
        session.prepare_one(_session_request(tmp_path))


def test_process_session_rejects_mismatched_request_id(tmp_path: Path) -> None:
    program = (
        "import sys;"
        "from robata.contracts.hashing import canonical_json_bytes;"
        "from robata.inference.mage_dcvc_preparation_protocol import "
        "MageDcvcPreparationRequest as Q,MageDcvcPreparationResponse as R;"
        "q=Q.model_validate_json(sys.stdin.buffer.readline(),strict=True);"
        "r=R(request_id='wrong',status='BUILT',preparation_identity=q.preparation_identity,"
        "artifact_semantic_sha256='b'*64,output_directory='fake-output',wall_seconds=0.0);"
        "sys.stdout.buffer.write(canonical_json_bytes(r.model_dump(mode='json'))+b'\\n');"
        "sys.stdout.buffer.flush()"
    )
    session = MageDcvcPreparationProcessSession(
        command=(sys.executable, "-u", "-c", program),
        environment=_session_environment(),
        working_directory=tmp_path,
        response_timeout_seconds=10.0,
    )

    with pytest.raises(MageDcvcPrewarmError, match="request_id differs"):
        session.prepare_one(_session_request(tmp_path))


def test_process_session_rejects_extra_output_on_close(tmp_path: Path) -> None:
    program = (
        "import sys;"
        "from robata.contracts.hashing import canonical_json_bytes;"
        "from robata.inference.mage_dcvc_preparation_protocol import "
        "MageDcvcPreparationRequest as Q,MageDcvcPreparationResponse as R;"
        "q=Q.model_validate_json(sys.stdin.buffer.readline(),strict=True);"
        "r=R(request_id=q.request_id,status='BUILT',preparation_identity=q.preparation_identity,"
        "artifact_semantic_sha256='b'*64,output_directory='fake-output',wall_seconds=0.0);"
        "sys.stdout.buffer.write(canonical_json_bytes(r.model_dump(mode='json'))+b'\\n');"
        "sys.stdout.buffer.flush();sys.stdin.buffer.read();"
        "sys.stdout.buffer.write(b'extra\\n');sys.stdout.buffer.flush()"
    )
    session = MageDcvcPreparationProcessSession(
        command=(sys.executable, "-u", "-c", program),
        environment=_session_environment(),
        working_directory=tmp_path,
        response_timeout_seconds=10.0,
    )

    response = session.prepare_one(_session_request(tmp_path))
    assert response.request_id == "r0"
    with pytest.raises(MageDcvcPrewarmError, match="unexpected extra output"):
        session.close()


def test_same_device_run_requires_guard_before_worker_start(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    setup["config"] = build_mage_dcvc_effective_config(
        model_directory=setup["model"],
        preparation_device="cpu",
        device_concurrency_policy="exclusive-shared-device-v1",
        max_side=448,
    )
    runner = _ArtifactRunner(config=setup["config"])

    with pytest.raises(MageDcvcPrewarmError, match="cooperative device guard"):
        prewarm_mage_dcvc_provider_v2(
            qualified_provider_manifest=setup["qualification"],
            checkpoint_manifest=setup["checkpoint"],
            codec_policy=_policy(),
            effective_config=setup["config"],
            model_directory=setup["model"],
            provider_source_files=setup["provider_sources"],
            provider_state_root=setup["state"],
            cache_base_root=setup["cache"],
            source_paths=setup["videos"],
            cache_manifest_output=setup["manifest"],
            report_output=setup["report"],
            generation_device="cpu",
            shared_device_guard_file=None,
            worker_python=Path(sys.executable),
            response_timeout_seconds=60.0,
            worker_process_runner=runner,
        )

    assert runner.call_count == 0


def test_windows_cache_path_budget_fails_before_expensive_worker_work(tmp_path: Path) -> None:
    output = tmp_path / ("x" * 220)

    with pytest.raises(MageDcvcPrewarmError, match="shorter --cache-base-root"):
        _validate_windows_cache_output_paths((output,), platform_name="nt")

    _validate_windows_cache_output_paths((output,), platform_name="posix")


def test_worker_error_detail_is_bounded_and_single_line() -> None:
    assert _bounded_worker_error_detail("first\n second") == "first second"
    bounded = _bounded_worker_error_detail("x" * 1000)
    assert len(bounded) == 512
    assert bounded.endswith("...")
