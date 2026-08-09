from __future__ import annotations

import io
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from types import ModuleType

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import canonical_json_bytes
from robata.inference import mage_dcvc_preparation_worker as worker_module
from robata.inference.mage_dcvc_preparation_protocol import (
    MAGE_DCVC_PREPARATION_SIDECAR_NAME,
    MAGE_DCVC_TEMP_MARKER_NAME,
    MageDcvcEffectiveConfig,
    MageDcvcPreparationArtifact,
    MageDcvcPreparationRequest,
    MageDcvcTempMarker,
    mage_dcvc_effective_config_sha256,
)
from robata.inference.mage_dcvc_preparation_worker import (
    MageDcvcBackendError,
    MageDcvcPreparationRejected,
    MageDcvcPreparationWorker,
    PersistentMageDcvcPreparationBackend,
    build_mage_dcvc_effective_config,
    build_mage_dcvc_preparation_request,
    build_mage_dcvc_provider_implementation_sha256,
    serve_mage_dcvc_preparation_jsonl,
)


def _write_model_tree(tmp_path: Path) -> Path:
    model = tmp_path / "mage"
    for relative in worker_module._PROVIDER_REQUIRED_RELATIVE_FILES:
        path = model / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# test provider source: {relative}\n", encoding="utf-8")
    for relative_root in worker_module._PROVIDER_RECURSIVE_ROOTS:
        root = model / Path(relative_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_source.py").write_text("VALUE = 1\n", encoding="utf-8")
    neural = model / "neural_codec"
    (neural / "dcvc_rt_intra.tar").write_bytes(b"test-intra-checkpoint")
    (neural / "dcvc_rt_inter.tar").write_bytes(b"test-inter-checkpoint")
    return model


def _config(
    tmp_path: Path,
    *,
    preparation_device: str = "cpu",
    policy: str = "separate-device-v1",
    max_side: int = 0,
) -> tuple[Path, MageDcvcEffectiveConfig]:
    model = _write_model_tree(tmp_path)
    return model, build_mage_dcvc_effective_config(
        model_directory=model,
        preparation_device=preparation_device,
        device_concurrency_policy=policy,
        max_side=max_side,
    )


def _provider_meta(config: MageDcvcEffectiveConfig) -> dict[str, object]:
    return {
        "provider_version": config.provider_version,
        "recipe_version": config.recipe_version,
        "effective_config_sha256": config.effective_config_sha256,
        "provider_implementation_sha256": config.provider_implementation_sha256,
        "engine": "dcvc-rt",
        "preparation_device": config.preparation_device,
        "max_side": config.max_side,
        "configured_sampled_frame_count": config.sampled_frame_count,
        "effective_sampled_frame_count": 12,
        "sequence_length_frames": 0,
        "canvas_token_side": None,
        "encoded_frame_extent": "through-last-sampled-frame",
        "segment_state_policy": "reset-per-job",
    }


class _FakeBackend:
    def __init__(self, config: MageDcvcEffectiveConfig, *, fail: bool = False) -> None:
        self.config = config
        self.fail = fail
        self.calls: list[tuple[Path, Path]] = []
        self.closed = False

    @property
    def effective_config_sha256(self) -> str:
        return self.config.effective_config_sha256

    def prepare(self, *, source_path: Path, output_directory: Path) -> dict[str, object]:
        self.calls.append((source_path, output_directory))
        (output_directory / "partial.bin").write_bytes(b"partial")
        if self.fail:
            raise MageDcvcBackendError("synthetic provider failure")
        (output_directory / "canvas_000.jpg").write_bytes(b"canvas")
        (output_directory / "src_patch_position.npy").write_bytes(b"positions")
        metadata = _provider_meta(self.config)
        (output_directory / "meta.json").write_text(
            json.dumps(
                {
                    "canvas_files": ["canvas_000.jpg"],
                    "robata_dcvc_provider": metadata,
                }
            ),
            encoding="utf-8",
        )
        return metadata

    def close(self) -> None:
        self.closed = True


class _FakeGuard:
    def __init__(self) -> None:
        self.entries = 0
        self.active = False

    @contextmanager
    def hold(self) -> Iterator[None]:
        self.entries += 1
        self.active = True
        try:
            yield
        finally:
            self.active = False


def _worker(
    tmp_path: Path,
    config: MageDcvcEffectiveConfig,
    backend: _FakeBackend,
    *,
    generation_device: str = "cuda:0",
    guard: _FakeGuard | None = None,
) -> MageDcvcPreparationWorker:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    return MageDcvcPreparationWorker(
        effective_config=config,
        backend=backend,
        input_roots=[inputs],
        output_root=tmp_path / "outputs",
        generation_device=generation_device,
        device_guard=guard,
    )


def _source(tmp_path: Path, name: str = "segment.mp4") -> Path:
    source = tmp_path / "inputs" / name
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(f"immutable-{name}".encode())
    return source


def test_effective_config_uses_canvas_formula_and_has_no_false_sequence_cap(
    tmp_path: Path,
) -> None:
    _, config = _config(tmp_path)

    assert config.target_canvas == 32
    assert config.images_per_group == 4
    assert config.group_size == 32
    assert config.sampled_frame_count == 256
    assert config.sequence_length_frames == 0
    assert config.canvas_token_side is None
    assert config.encoded_frame_extent == "through-last-sampled-frame"
    assert "seq_len_frames" not in config.model_dump(mode="json")

    values = config.model_dump(mode="json")
    values["sampled_frame_count"] = 8
    provisional = MageDcvcEffectiveConfig.model_construct(**values)
    values["effective_config_sha256"] = mage_dcvc_effective_config_sha256(provisional)
    with pytest.raises(ValidationError, match="canvas formula"):
        MageDcvcEffectiveConfig.model_validate(values, strict=True)


def test_effective_config_and_implementation_identity_change_on_real_inputs(
    tmp_path: Path,
) -> None:
    model, full = _config(tmp_path / "full", max_side=0)
    _, reduced = _config(tmp_path / "reduced", max_side=448)

    assert full.effective_config_sha256 != reduced.effective_config_sha256
    first_implementation = build_mage_dcvc_provider_implementation_sha256(model)
    engine = model / "neural_codec" / "dcvc_rt_engine.py"
    engine.write_text(engine.read_text(encoding="utf-8") + "CHANGED = True\n", encoding="utf-8")
    assert build_mage_dcvc_provider_implementation_sha256(model) != first_implementation


def test_request_rejects_unsafe_output_path(tmp_path: Path) -> None:
    _, config = _config(tmp_path)
    source = _source(tmp_path)
    request = build_mage_dcvc_preparation_request(
        request_id="request-1",
        source_path=source,
        output_relative_path="segment-1",
        effective_config=config,
    )
    values = request.model_dump(mode="json")
    values["output_relative_path"] = "../escape"

    with pytest.raises(ValidationError, match="safe relative POSIX"):
        MageDcvcPreparationRequest.model_validate(values, strict=True)


def test_worker_commits_atomically_and_repeats_as_verified_hit(tmp_path: Path) -> None:
    _, config = _config(tmp_path)
    backend = _FakeBackend(config)
    worker = _worker(tmp_path, config, backend)
    source = _source(tmp_path)
    request = build_mage_dcvc_preparation_request(
        request_id="request-atomic",
        source_path=source,
        output_relative_path="namespace/segment-000",
        effective_config=config,
    )

    first = worker.prepare(request)
    second = worker.prepare(request)

    assert first.status == "BUILT"
    assert second.status == "VERIFIED_HIT"
    assert first.artifact_semantic_sha256 == second.artifact_semantic_sha256
    assert len(backend.calls) == 1
    output = Path(first.output_directory or "")
    assert output.is_dir()
    assert not (output / MAGE_DCVC_TEMP_MARKER_NAME).exists()
    artifact = MageDcvcPreparationArtifact.model_validate_json(
        (output / MAGE_DCVC_PREPARATION_SIDECAR_NAME).read_bytes(),
        strict=True,
    )
    assert artifact.preparation_identity == request.preparation_identity
    assert artifact.effective_config_sha256 == config.effective_config_sha256
    assert not tuple(output.parent.glob(f".robata-dcvc-{request.preparation_identity[:16]}-*"))

    (output / "canvas_000.jpg").write_bytes(b"tampered")
    tampered = worker.prepare(request)
    assert tampered.status == "REJECTED"
    assert len(backend.calls) == 1


def test_existing_sidecarless_output_fails_closed_without_rebuild_or_deletion(
    tmp_path: Path,
) -> None:
    _, config = _config(tmp_path)
    backend = _FakeBackend(config)
    worker = _worker(tmp_path, config, backend)
    source = _source(tmp_path, "sidecarless.mp4")
    request = build_mage_dcvc_preparation_request(
        request_id="request-sidecarless",
        source_path=source,
        output_relative_path="namespace/sidecarless",
        effective_config=config,
    )
    output = tmp_path / "outputs" / "namespace" / "sidecarless"
    output.mkdir(parents=True)
    sentinel = output / "unadmitted-provider-output.bin"
    sentinel.write_bytes(b"preserve-for-audit")

    response = worker.prepare(request)

    assert response.status == "REJECTED"
    assert response.error_code == "DCVC_PREPARATION_REJECTED"
    assert response.error_message == "committed output sidecar is missing or inaccessible"
    assert backend.calls == []
    assert sentinel.read_bytes() == b"preserve-for-audit"


def test_windows_path_budget_rejects_before_provider_or_staging_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config = _config(tmp_path)
    backend = _FakeBackend(config)
    worker = _worker(tmp_path, config, backend)
    source = _source(tmp_path, "long-path.mp4")
    output_root = tmp_path / "outputs"
    namespace = output_root / "namespace"
    required_leaf_length = (
        worker_module._WINDOWS_LEGACY_MAX_PATH
        - len(str(namespace))
        - len(MAGE_DCVC_PREPARATION_SIDECAR_NAME)
    )
    leaf = "x" * max(1, required_leaf_length)
    output = namespace / leaf
    assert (
        len(str(output / MAGE_DCVC_PREPARATION_SIDECAR_NAME))
        >= worker_module._WINDOWS_LEGACY_MAX_PATH
    )
    request = build_mage_dcvc_preparation_request(
        request_id="request-long-path",
        source_path=source,
        output_relative_path=f"namespace/{leaf}",
        effective_config=config,
    )
    monkeypatch.setattr(worker_module, "_windows_legacy_max_path_applies", lambda: True)

    response = worker.prepare(request)

    assert response.status == "REJECTED"
    assert response.error_code == "DCVC_PREPARATION_REJECTED"
    assert response.error_message is not None
    assert "Windows MAX_PATH" in response.error_message
    assert "choose a shorter output root or output_relative_path" in response.error_message
    assert backend.calls == []
    assert not namespace.exists()


def test_failure_cleans_only_owned_staging_directory(tmp_path: Path) -> None:
    _, config = _config(tmp_path)
    backend = _FakeBackend(config, fail=True)
    worker = _worker(tmp_path, config, backend)
    source = _source(tmp_path)
    request = build_mage_dcvc_preparation_request(
        request_id="request-fail",
        source_path=source,
        output_relative_path="namespace/segment-fail",
        effective_config=config,
    )

    response = worker.prepare(request)

    assert response.status == "FAILED"
    parent = tmp_path / "outputs" / "namespace"
    assert not (parent / "segment-fail").exists()
    assert not tuple(parent.glob(f".robata-dcvc-{request.preparation_identity[:16]}-*"))


def test_restart_recovers_owned_staging_but_rejects_unknown_staging(tmp_path: Path) -> None:
    _, config = _config(tmp_path)
    backend = _FakeBackend(config)
    worker = _worker(tmp_path, config, backend)
    source = _source(tmp_path, "recover.mp4")
    request = build_mage_dcvc_preparation_request(
        request_id="request-recover",
        source_path=source,
        output_relative_path="namespace/recovered",
        effective_config=config,
    )
    parent = tmp_path / "outputs" / "namespace"
    parent.mkdir(parents=True)
    stale = parent / f".robata-dcvc-{request.preparation_identity[:16]}-stale"
    stale.mkdir()
    marker = MageDcvcTempMarker(
        request_id=request.request_id,
        preparation_identity=request.preparation_identity,
        effective_config_sha256=request.effective_config_sha256,
    )
    (stale / MAGE_DCVC_TEMP_MARKER_NAME).write_bytes(
        canonical_json_bytes(marker.model_dump(mode="json"))
    )
    (stale / "partial.bin").write_bytes(b"crash-leftover")

    recovered = worker.prepare(request)

    assert recovered.status == "BUILT"
    assert not stale.exists()

    second_source = _source(tmp_path, "unknown.mp4")
    second = build_mage_dcvc_preparation_request(
        request_id="request-unknown",
        source_path=second_source,
        output_relative_path="namespace/unknown",
        effective_config=config,
    )
    unknown = parent / f".robata-dcvc-{second.preparation_identity[:16]}-unknown"
    unknown.mkdir()
    (unknown / "unowned.bin").write_bytes(b"do-not-delete")

    rejected = worker.prepare(second)

    assert rejected.status == "REJECTED"
    assert unknown.is_dir()
    assert (unknown / "unowned.bin").read_bytes() == b"do-not-delete"


class _BlockingBackend(_FakeBackend):
    def __init__(self, config: MageDcvcEffectiveConfig) -> None:
        super().__init__(config)
        self.entered = Event()
        self.release = Event()

    def prepare(self, *, source_path: Path, output_directory: Path) -> dict[str, object]:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise MageDcvcBackendError("test release timed out")
        return super().prepare(source_path=source_path, output_directory=output_directory)


def test_worker_rejects_a_second_concurrent_job(tmp_path: Path) -> None:
    _, config = _config(tmp_path)
    backend = _BlockingBackend(config)
    worker = _worker(tmp_path, config, backend)
    first_request = build_mage_dcvc_preparation_request(
        request_id="request-one",
        source_path=_source(tmp_path, "one.mp4"),
        output_relative_path="one",
        effective_config=config,
    )
    second_request = build_mage_dcvc_preparation_request(
        request_id="request-two",
        source_path=_source(tmp_path, "two.mp4"),
        output_relative_path="two",
        effective_config=config,
    )
    responses: list[object] = []
    thread = Thread(target=lambda: responses.append(worker.prepare(first_request)))
    thread.start()
    assert backend.entered.wait(timeout=2)

    busy = worker.prepare(second_request)
    backend.release.set()
    thread.join(timeout=5)

    assert busy.status == "BUSY"
    assert not thread.is_alive()
    assert len(responses) == 1
    assert responses[0].status == "BUILT"


def test_shared_gpu_requires_and_uses_cooperative_generation_guard(tmp_path: Path) -> None:
    _, config = _config(
        tmp_path,
        preparation_device="cuda:0",
        policy="exclusive-shared-device-v1",
    )
    backend = _FakeBackend(config)
    inputs = tmp_path / "inputs"
    inputs.mkdir()

    with pytest.raises(MageDcvcPreparationRejected, match="cooperative guard"):
        MageDcvcPreparationWorker(
            effective_config=config,
            backend=backend,
            input_roots=[inputs],
            output_root=tmp_path / "outputs",
            generation_device="cuda",
        )

    guard = _FakeGuard()
    worker = MageDcvcPreparationWorker(
        effective_config=config,
        backend=backend,
        input_roots=[inputs],
        output_root=tmp_path / "outputs",
        generation_device="cuda",
        device_guard=guard,
    )
    request = build_mage_dcvc_preparation_request(
        request_id="request-shared-gpu",
        source_path=_source(tmp_path),
        output_relative_path="shared-gpu",
        effective_config=config,
    )

    assert worker.prepare(request).status == "BUILT"
    assert guard.entries == 1


def _write_importable_provider_tree(tmp_path: Path) -> Path:
    model = _write_model_tree(tmp_path)
    neural = model / "neural_codec"
    (neural / "codec_dcvc_config.py").write_text(
        "raise RuntimeError('hidden disk config must not import')\n",
        encoding="utf-8",
    )
    dcvc_source = neural / "DCVC" / "src"
    for relative in (
        "utils/common.py",
        "utils/transforms.py",
        "models/image_model.py",
        "models/video_model.py",
        "layers/cuda_inference.py",
    ):
        target = dcvc_source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VALUE = 1\n", encoding="utf-8")
    (neural / "dcvc_rt_engine.py").write_text(
        "import os, sys\n"
        "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DCVC'))\n"
        "import src.utils.common, src.utils.transforms\n"
        "import src.models.image_model, src.models.video_model\n"
        "import src.layers.cuda_inference\n"
        "class DCVCRTEngine:\n    pass\n",
        encoding="utf-8",
    )
    pipeline_root = neural / "codec_tools" / "pipeline"
    (neural / "codec_tools" / "__init__.py").write_text("", encoding="utf-8")
    (pipeline_root / "__init__.py").write_text("", encoding="utf-8")
    (pipeline_root / "process_video_bitcost_mv_mask_collage.py").write_text(
        "import codec_dcvc_config as _dc\ndef process_group():\n    return _dc.get('max_side')\n",
        encoding="utf-8",
    )
    (pipeline_root / "process_video_bitcost_readiness.py").write_text(
        "import codec_dcvc_config as _dc\n"
        "from pipeline.process_video_bitcost_mv_mask_collage import process_group\n",
        encoding="utf-8",
    )
    (neural / "dcvc_readiness_gen.py").write_text(
        "import importlib\n"
        "import codec_dcvc_config as _dc\n"
        "from dcvc_rt_engine import DCVCRTEngine\n"
        "P = importlib.import_module('codec_tools.pipeline.process_video_bitcost_readiness')\n"
        "_ENGINE = None\n"
        "def _get_engine():\n"
        "    global _ENGINE\n"
        "    if _ENGINE is None:\n"
        "        _ENGINE = DCVCRTEngine()\n"
        "    return _ENGINE\n"
        "def main():\n    return None\n",
        encoding="utf-8",
    )
    return model


def test_production_loader_installs_config_overlay_before_readiness_import(
    tmp_path: Path,
) -> None:
    model = _write_importable_provider_tree(tmp_path)
    config = build_mage_dcvc_effective_config(
        model_directory=model,
        preparation_device="cpu",
        device_concurrency_policy="separate-device-v1",
        max_side=448,
    )
    overlay = worker_module._install_dcvc_config_overlay(
        state_root=tmp_path / "provider-state",
        config=config,
    )
    module_prefixes = (
        "codec_dcvc_config",
        "dcvc_readiness_gen",
        "dcvc_rt_engine",
        "codec_tools",
        "pipeline",
        "src",
    )
    original_path = list(sys.path)
    for name in tuple(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in module_prefixes):
            sys.modules.pop(name, None)
    try:
        readiness = worker_module._load_external_provider(
            neural_root=model / "neural_codec",
            overlay_root=overlay,
        )
        config_module = readiness._dc
        pipeline = readiness.P
        companion = worker_module._companion_module_for(pipeline)

        src_namespace = sys.modules["src"]
        assert list(src_namespace.__path__) == [
            str((model / "neural_codec" / "DCVC" / "src").resolve())
        ]
        assert Path(config_module.__file__).resolve() == (overlay / "codec_dcvc_config.py")
        assert config_module.get("max_side") == 448
        assert config_module.get("num_sampled_frames") == 256
        assert pipeline._dc is config_module
        assert companion._dc is config_module
        with pytest.raises(MageDcvcBackendError, match="imported before"):
            worker_module._load_external_provider(
                neural_root=model / "neural_codec",
                overlay_root=overlay,
            )
    finally:
        sys.path[:] = original_path
        for name in tuple(sys.modules):
            if name == module_prefixes or name.startswith(module_prefixes):
                sys.modules.pop(name, None)


class _FakeCapture:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, _property: object) -> int:
        return self.frame_count

    def release(self) -> None:
        self.released = True


class _FakeCv2:
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.captures: list[_FakeCapture] = []

    def VideoCapture(self, _path: str) -> _FakeCapture:
        capture = _FakeCapture(self.frame_count)
        self.captures.append(capture)
        return capture


class _FakeResidentEngine:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_sequence(self) -> None:
        self.reset_calls += 1


def _fake_readiness_module(
    config: MageDcvcEffectiveConfig,
    *,
    required_guard: _FakeGuard | None = None,
    emit_stdout: bool = False,
) -> tuple[ModuleType, list[int]]:
    module = ModuleType("fake_dcvc_readiness")
    loads: list[int] = []
    module._ENGINE = None

    def get_engine() -> _FakeResidentEngine:
        if emit_stdout:
            print("provider engine load message")
        if required_guard is not None and not required_guard.active:
            raise RuntimeError("engine load escaped the shared-device guard")
        if module._ENGINE is None:
            loads.append(1)
            module._ENGINE = _FakeResidentEngine()
        return module._ENGINE

    def main() -> None:
        if emit_stdout:
            print("provider readiness progress")
        engine = get_engine()
        engine.reset_sequence()
        argv = list(sys.argv)
        output = Path(argv[argv.index("--out_dir") + 1])
        sampled = int(argv[argv.index("--num_sampled_frames") + 1])
        (output / "canvas_000.jpg").write_bytes(b"resident-canvas")
        (output / "src_patch_position.npy").write_bytes(b"resident-positions")
        (output / "meta.json").write_text(
            json.dumps({"canvas_files": ["canvas_000.jpg"], "seq_len": sampled}),
            encoding="utf-8",
        )

    module._get_engine = get_engine
    module.main = main
    return module, loads


def test_persistent_backend_loads_engine_once_and_resets_each_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config = _config(tmp_path)
    readiness, loads = _fake_readiness_module(config)
    cv2 = _FakeCv2(frame_count=12)
    monkeypatch.setattr(
        worker_module,
        "_load_external_provider",
        lambda **_kwargs: readiness,
    )
    monkeypatch.setattr(worker_module, "_load_cv2", lambda: cv2)
    backend = PersistentMageDcvcPreparationBackend(
        effective_config=config,
        model_directory=model,
        provider_state_root=tmp_path / "provider-state",
    )
    source_one = tmp_path / "one.mp4"
    source_two = tmp_path / "two.mp4"
    source_one.write_bytes(b"one")
    source_two.write_bytes(b"two")
    output_one = tmp_path / "out-one"
    output_two = tmp_path / "out-two"
    output_one.mkdir()
    output_two.mkdir()
    original_argv = sys.argv

    first_meta = backend.prepare(source_path=source_one, output_directory=output_one)
    second_meta = backend.prepare(source_path=source_two, output_directory=output_two)

    assert loads == [1]
    assert readiness._ENGINE.reset_calls == 2
    assert first_meta["effective_sampled_frame_count"] == 12
    assert second_meta["effective_sampled_frame_count"] == 12
    assert first_meta["configured_sampled_frame_count"] == 256
    assert first_meta["sequence_length_frames"] == 0
    assert first_meta["canvas_token_side"] is None
    assert sys.argv is original_argv
    assert all(capture.released for capture in cv2.captures)
    overlay_module = (
        tmp_path / "provider-state" / config.effective_config_sha256 / "codec_dcvc_config.py"
    )
    assert overlay_module.is_file()


def test_persistent_provider_stdout_is_redirected_away_from_jsonl_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config = _config(tmp_path)
    readiness, _loads = _fake_readiness_module(config, emit_stdout=True)
    monkeypatch.setattr(worker_module, "_load_external_provider", lambda **_kwargs: readiness)
    monkeypatch.setattr(worker_module, "_load_cv2", lambda: _FakeCv2(frame_count=12))
    backend = PersistentMageDcvcPreparationBackend(
        effective_config=config,
        model_directory=model,
        provider_state_root=tmp_path / "provider-state",
    )
    source = tmp_path / "provider-output.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "provider-output"
    output.mkdir()

    backend.prepare(source_path=source, output_directory=output)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "provider engine load message" in captured.err
    assert "provider readiness progress" in captured.err


def test_first_resident_engine_load_occurs_inside_shared_gpu_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config = _config(
        tmp_path,
        preparation_device="cuda:0",
        policy="exclusive-shared-device-v1",
    )
    guard = _FakeGuard()
    readiness, loads = _fake_readiness_module(config, required_guard=guard)
    monkeypatch.setattr(
        worker_module,
        "_load_external_provider",
        lambda **_kwargs: readiness,
    )
    monkeypatch.setattr(worker_module, "_load_cv2", lambda: _FakeCv2(frame_count=12))
    backend = PersistentMageDcvcPreparationBackend(
        effective_config=config,
        model_directory=model,
        provider_state_root=tmp_path / "provider-state",
    )
    source = _source(tmp_path, "guarded-load.mp4")
    worker = MageDcvcPreparationWorker(
        effective_config=config,
        backend=backend,
        input_roots=[tmp_path / "inputs"],
        output_root=tmp_path / "outputs",
        generation_device="cuda",
        device_guard=guard,
    )
    request = build_mage_dcvc_preparation_request(
        request_id="guarded-load",
        source_path=source,
        output_relative_path="guarded-load",
        effective_config=config,
    )

    assert loads == []
    assert worker.prepare(request).status == "BUILT"
    assert loads == [1]
    assert guard.entries == 1
    assert not guard.active


def test_jsonl_protocol_keeps_invalid_request_isolated(tmp_path: Path) -> None:
    _, config = _config(tmp_path)
    backend = _FakeBackend(config)
    worker = _worker(tmp_path, config, backend)
    request = build_mage_dcvc_preparation_request(
        request_id="jsonl-good",
        source_path=_source(tmp_path),
        output_relative_path="jsonl-good",
        effective_config=config,
    )
    input_stream = io.StringIO(
        request.model_dump_json() + "\n" + '{"request_id":"jsonl-bad"}' + "\n"
    )
    output_stream = io.StringIO()

    serve_mage_dcvc_preparation_jsonl(
        worker=worker,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert [response["status"] for response in responses] == ["BUILT", "REJECTED"]
    assert responses[1]["request_id"] == "jsonl-bad"
    assert responses[1]["error_code"] == "DCVC_PROTOCOL_INVALID"
