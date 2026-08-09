"""Persistent pre-admission orchestration for qualified Mage DCVC Provider V2 caches.

One resident JSONL worker services every segment in a prewarm run. Exact worker
artifacts are admitted through the v2 cache contract, and only fully verified evidence
is published. This is an internal operational contract, not a published schema.
"""

from __future__ import annotations

import hashlib
import os
import queue
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from threading import Thread
from typing import Annotated, Any, Final, Literal, Protocol, Self, cast

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.inference.mage_checkpoint_identity import MageCheckpointManifest
from robata.inference.mage_codec_cache_v2 import (
    MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME,
    MageCodecCacheEntryV2,
    MageCodecCacheManifestV2,
    build_mage_codec_cache_manifest_v2,
    load_mage_codec_cache_manifest_v2,
    mage_codec_v2_namespace_identity,
    upstream_mage_codec_cache_directory_name,
    validate_mage_dcvc_effective_config_for_policy,
    verify_mage_codec_cache_manifest_v2,
    write_mage_codec_cache_manifest_v2,
)
from robata.inference.mage_dcvc_preparation_protocol import (
    MAGE_DCVC_PREPARATION_SIDECAR_NAME,
    MageDcvcEffectiveConfig,
    MageDcvcPreparationRequest,
    MageDcvcPreparationResponse,
)
from robata.inference.mage_dcvc_preparation_worker import (
    build_mage_dcvc_preparation_request,
    build_mage_dcvc_provider_implementation_sha256,
)
from robata.inference.mage_dcvc_qualified_provider import (
    MageDcvcQualifiedProviderManifest,
    verify_mage_dcvc_qualified_provider_sources,
)
from robata.inference.mage_video_endpoint import (
    MageVideoCodecPolicy,
    build_mage_video_codec_policy_identity,
)

MAGE_DCVC_PREWARM_REPORT_V2_VERSION: Final = "mage-dcvc-prewarm-report-v2"
MAGE_DCVC_PREWARM_PROCESS_V2_VERSION: Final = "mage-dcvc-prewarm-process-v2"
MAGE_DCVC_PREWARM_JOB_V2_VERSION: Final = "mage-dcvc-prewarm-job-v2"
MAGE_DCVC_WORKER_MODULE: Final = "robata.inference.mage_dcvc_preparation_worker"
_WINDOWS_CONSERVATIVE_MAX_PATH_CHARS: Final = 260
_WORKER_ERROR_DETAIL_MAX_CHARS: Final = 512

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16_384)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]


class MageDcvcPrewarmError(RuntimeError):
    """A qualified prewarm run failed before durable admission completed."""


class MageDcvcWorkerProcessTelemetryV2(StrictModel):
    telemetry_version: Literal["mage-dcvc-prewarm-process-v2"] = (
        MAGE_DCVC_PREWARM_PROCESS_V2_VERSION
    )
    process_start_count: Literal[1] = 1
    exit_code: Literal[0] = 0
    response_count: NonNegativeInt
    wall_seconds: NonNegativeFloat
    stderr_byte_count: NonNegativeInt
    stderr_sha256: Sha256Digest


class MageDcvcPrewarmJobTelemetryV2(StrictModel):
    telemetry_version: Literal["mage-dcvc-prewarm-job-v2"] = MAGE_DCVC_PREWARM_JOB_V2_VERSION
    request_id: NonEmptyString
    source_path: NonEmptyString
    output_directory: NonEmptyString
    admission: Literal["BUILT", "VERIFIED_HIT"]
    response_wall_seconds: NonNegativeFloat
    preparation_identity: Sha256Digest
    artifact_semantic_sha256: Sha256Digest
    engine_load_count_in_artifact: PositiveInt
    engine_load_seconds_in_artifact: NonNegativeFloat
    worker_completed_job_count_in_artifact: PositiveInt
    sequence_reset_count_for_job: Literal[1]
    effective_sampled_frame_count: PositiveInt
    max_encoded_frame_id: NonNegativeInt


class MageDcvcPrewarmReportV2(StrictModel):
    report_version: Literal["mage-dcvc-prewarm-report-v2"] = MAGE_DCVC_PREWARM_REPORT_V2_VERSION
    qualification_manifest_semantic_sha256: Sha256Digest
    qualified_checkpoint_manifest_sha256: Sha256Digest
    codec_policy_sha256: Sha256Digest
    provider_implementation_sha256: Sha256Digest
    effective_config_sha256: Sha256Digest
    namespace_identity: Sha256Digest
    replay_mode: Literal[
        "fresh-build",
        "mixed-recovery",
        "verified-hit-reconstruction",
        "exact-verified-replay",
    ]
    exact_verified_replay: bool
    inferred_process_model_load_count: Annotated[int, Field(strict=True, ge=0, le=1)]
    inferred_process_model_load_seconds: NonNegativeFloat
    worker_process: MageDcvcWorkerProcessTelemetryV2
    job_count: PositiveInt
    built_count: NonNegativeInt
    verified_hit_count: NonNegativeInt
    prewarm_wall_seconds: NonNegativeFloat
    effective_config_path: NonEmptyString
    cache_manifest_path: NonEmptyString
    cache_manifest_exact_sha256: Sha256Digest
    cache_manifest_semantic_sha256: Sha256Digest
    jobs: tuple[MageDcvcPrewarmJobTelemetryV2, ...]
    report_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.job_count != len(self.jobs):
            raise ValueError("job_count does not match jobs")
        if self.built_count + self.verified_hit_count != self.job_count:
            raise ValueError("prewarm admission counts do not sum to job_count")
        if self.worker_process.response_count != self.job_count:
            raise ValueError("worker response count does not match job_count")
        if self.exact_verified_replay != (self.replay_mode == "exact-verified-replay"):
            raise ValueError("exact_verified_replay does not match replay_mode")
        if self.inferred_process_model_load_count == 0:
            if self.built_count != 0 or self.inferred_process_model_load_seconds != 0.0:
                raise ValueError("zero model loads require a pure verified-hit run")
        elif self.built_count == 0:
            raise ValueError("a model load requires at least one built artifact")
        if self.report_semantic_sha256 != _prewarm_report_semantic_sha256(self):
            raise ValueError("report_semantic_sha256 does not match report")
        return self


class MageDcvcWorkerProcessRunner(Protocol):
    def __call__(
        self,
        *,
        command: Sequence[str],
        requests: Sequence[MageDcvcPreparationRequest],
        environment: Mapping[str, str],
        working_directory: Path,
        response_timeout_seconds: float,
    ) -> tuple[tuple[MageDcvcPreparationResponse, ...], MageDcvcWorkerProcessTelemetryV2]: ...


def run_mage_dcvc_preparation_process(
    *,
    command: Sequence[str],
    requests: Sequence[MageDcvcPreparationRequest],
    environment: Mapping[str, str],
    working_directory: Path,
    response_timeout_seconds: float,
) -> tuple[tuple[MageDcvcPreparationResponse, ...], MageDcvcWorkerProcessTelemetryV2]:
    """Run one binary-safe JSONL subprocess for all requests and require a clean exit."""

    if not command or not requests:
        raise MageDcvcPrewarmError("worker command and requests must be nonempty")
    if response_timeout_seconds <= 0.0:
        raise MageDcvcPrewarmError("worker response timeout must be positive")
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            tuple(str(part) for part in command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(working_directory).expanduser().resolve()),
            env=dict(environment),
            shell=False,
        )
    except OSError as error:
        raise MageDcvcPrewarmError("could not start the resident DCVC worker") from error
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _terminate_process(process)
        raise MageDcvcPrewarmError("resident DCVC worker pipes were not created")
    process_stdin = process.stdin
    process_stdout = process.stdout
    process_stderr = process.stderr

    stdout_queue: queue.Queue[bytes | None] = queue.Queue()
    stderr_digest = hashlib.sha256()
    stderr_byte_count = [0]

    def read_stdout() -> None:
        try:
            while line := process_stdout.readline():
                stdout_queue.put(line)
        finally:
            stdout_queue.put(None)

    def read_stderr() -> None:
        while chunk := process_stderr.read(64 * 1024):
            stderr_digest.update(chunk)
            stderr_byte_count[0] += len(chunk)

    stdout_thread = Thread(target=read_stdout, name="mage-dcvc-v2-stdout", daemon=True)
    stderr_thread = Thread(target=read_stderr, name="mage-dcvc-v2-stderr", daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    responses: list[MageDcvcPreparationResponse] = []
    try:
        for request in requests:
            process_stdin.write(canonical_json_bytes(request.model_dump(mode="json")) + b"\n")
            process_stdin.flush()
            try:
                line = stdout_queue.get(timeout=response_timeout_seconds)
            except queue.Empty as error:
                raise MageDcvcPrewarmError("resident DCVC worker response timed out") from error
            if line is None:
                raise MageDcvcPrewarmError("resident DCVC worker exited before responding")
            payload = line.rstrip(b"\r\n")
            if not payload or b"\n" in payload or b"\r" in payload:
                raise MageDcvcPrewarmError("resident DCVC worker emitted an invalid JSONL record")
            try:
                response = MageDcvcPreparationResponse.model_validate_json(payload, strict=True)
            except ValidationError as error:
                raise MageDcvcPrewarmError("resident DCVC worker response is invalid") from error
            if canonical_json_bytes(response.model_dump(mode="json")) != payload:
                raise MageDcvcPrewarmError("resident DCVC worker response is not canonical JSON")
            if response.request_id != request.request_id:
                raise MageDcvcPrewarmError("resident DCVC worker response request_id differs")
            responses.append(response)
        process_stdin.close()
        try:
            exit_code = process.wait(timeout=response_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise MageDcvcPrewarmError("resident DCVC worker did not exit after EOF") from error
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise MageDcvcPrewarmError("resident DCVC worker output drains did not finish")
        trailing = []
        while not stdout_queue.empty():
            item = stdout_queue.get_nowait()
            if item not in {None, b"", b"\n", b"\r\n"}:
                trailing.append(item)
        if trailing:
            raise MageDcvcPrewarmError("resident DCVC worker emitted unexpected extra output")
        if exit_code != 0:
            raise MageDcvcPrewarmError(f"resident DCVC worker exited with code {exit_code}")
    except (BrokenPipeError, OSError) as error:
        _terminate_process(process)
        raise MageDcvcPrewarmError("resident DCVC worker transport failed") from error
    except Exception:
        _terminate_process(process)
        raise
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process_stdin.close()
        if process.poll() is None:
            _terminate_process(process)
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

    telemetry = MageDcvcWorkerProcessTelemetryV2(
        response_count=len(responses),
        wall_seconds=max(0.0, time.perf_counter() - started),
        stderr_byte_count=stderr_byte_count[0],
        stderr_sha256=stderr_digest.hexdigest(),
    )
    return tuple(responses), telemetry


def prewarm_mage_dcvc_provider_v2(
    *,
    qualified_provider_manifest: MageDcvcQualifiedProviderManifest,
    checkpoint_manifest: MageCheckpointManifest,
    codec_policy: MageVideoCodecPolicy,
    effective_config: MageDcvcEffectiveConfig,
    model_directory: Path,
    provider_source_files: Sequence[Path],
    provider_state_root: Path,
    cache_base_root: Path,
    source_paths: Sequence[Path],
    cache_manifest_output: Path,
    report_output: Path,
    generation_device: str,
    shared_device_guard_file: Path | None,
    worker_python: Path = Path(sys.executable),
    intra_checkpoint_path: Path | None = None,
    inter_checkpoint_path: Path | None = None,
    response_timeout_seconds: float = 7_200.0,
    worker_process_runner: MageDcvcWorkerProcessRunner = run_mage_dcvc_preparation_process,
    clock: Callable[[], float] = time.perf_counter,
) -> MageDcvcPrewarmReportV2:
    """Pre-admit exact Provider V2 assets and atomically publish verified evidence."""

    started = clock()
    model_root = Path(model_directory).expanduser().resolve()
    qualified_root_path = (
        Path(qualified_provider_manifest.qualified_model_directory).expanduser().resolve()
    )
    if model_root != qualified_root_path:
        raise MageDcvcPrewarmError("model directory differs from qualified provider manifest")
    if checkpoint_manifest != qualified_provider_manifest.qualified_checkpoint_manifest:
        raise MageDcvcPrewarmError("checkpoint manifest differs from qualified provider checkpoint")
    try:
        verify_mage_dcvc_qualified_provider_sources(
            manifest=qualified_provider_manifest,
            provider_source_files=provider_source_files,
        )
        validate_mage_dcvc_effective_config_for_policy(
            effective_config=effective_config,
            codec_policy=codec_policy,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise MageDcvcPrewarmError(
            "Provider V2 qualification or policy validation failed"
        ) from error
    observed_implementation = build_mage_dcvc_provider_implementation_sha256(model_root)
    if observed_implementation != effective_config.provider_implementation_sha256:
        raise MageDcvcPrewarmError("effective config implementation digest is not current")

    shared_device = _normalise_device(effective_config.preparation_device) == _normalise_device(
        generation_device
    )
    if shared_device:
        if effective_config.device_concurrency_policy != "exclusive-shared-device-v1":
            raise MageDcvcPrewarmError("same-device prewarm requires exclusive concurrency policy")
        if shared_device_guard_file is None:
            raise MageDcvcPrewarmError("same-device prewarm requires a cooperative device guard")
    elif effective_config.device_concurrency_policy != "separate-device-v1":
        raise MageDcvcPrewarmError("separate-device prewarm requires separate concurrency policy")

    sources = _normalise_sources(source_paths)
    base_root = Path(cache_base_root).expanduser().resolve()
    base_root.mkdir(parents=True, exist_ok=True)
    policy_sha = build_mage_video_codec_policy_identity(codec_policy).policy_sha256
    namespace = mage_codec_v2_namespace_identity(
        checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
        codec_policy_sha256=policy_sha,
        provider_implementation_sha256=effective_config.provider_implementation_sha256,
        effective_config_sha256=effective_config.effective_config_sha256,
    )
    qualified_root = base_root / namespace
    config_path = _publish_effective_config(
        state_root=provider_state_root,
        effective_config=effective_config,
    )
    manifest_path = Path(cache_manifest_output).expanduser().resolve()
    report_path = Path(report_output).expanduser().resolve()
    if manifest_path == report_path or config_path in {manifest_path, report_path}:
        raise MageDcvcPrewarmError("effective config, cache manifest, and report paths must differ")

    existing_manifest: MageCodecCacheManifestV2 | None = None
    existing_manifest_bytes: bytes | None = None
    if manifest_path.exists():
        existing_manifest_bytes = manifest_path.read_bytes()
        existing_manifest = load_mage_codec_cache_manifest_v2(path=manifest_path)
        existing_entries = verify_mage_codec_cache_manifest_v2(manifest=existing_manifest)
        _validate_existing_manifest(
            manifest=existing_manifest,
            entries=existing_entries,
            checkpoint_manifest=checkpoint_manifest,
            codec_policy_sha256=policy_sha,
            effective_config=effective_config,
            cache_base_root=base_root,
            source_paths=sources,
        )

    requests: list[MageDcvcPreparationRequest] = []
    expected_outputs: list[Path] = []
    for index, source in enumerate(sources):
        directory_name = upstream_mage_codec_cache_directory_name(
            video_path=source,
            codec_policy=codec_policy,
            model_directory=model_root,
        )
        output = qualified_root / directory_name
        output_relative = PurePosixPath(namespace, directory_name).as_posix()
        request = build_mage_dcvc_preparation_request(
            request_id=f"prewarm-v2-{index:06d}",
            source_path=source,
            output_relative_path=output_relative,
            effective_config=effective_config,
        )
        requests.append(request)
        expected_outputs.append(output)
    if len(set(expected_outputs)) != len(expected_outputs):
        raise MageDcvcPrewarmError("multiple sources resolved to one Provider V2 cache directory")
    _validate_windows_cache_output_paths(expected_outputs)

    command = _worker_command(
        worker_python=worker_python,
        model_directory=model_root,
        effective_config_path=config_path,
        provider_state_root=provider_state_root,
        input_roots=tuple(sorted({source.parent for source in sources}, key=str)),
        output_root=base_root,
        generation_device=generation_device,
        shared_device_guard_file=shared_device_guard_file,
        intra_checkpoint_path=intra_checkpoint_path,
        inter_checkpoint_path=inter_checkpoint_path,
    )
    responses, process_telemetry = worker_process_runner(
        command=command,
        requests=tuple(requests),
        environment=_worker_environment(),
        working_directory=Path(__file__).resolve().parents[3],
        response_timeout_seconds=response_timeout_seconds,
    )
    if process_telemetry.process_start_count != 1:
        raise MageDcvcPrewarmError("prewarm requires exactly one resident worker process")
    if len(responses) != len(requests) or process_telemetry.response_count != len(requests):
        raise MageDcvcPrewarmError("resident worker response count differs from requests")

    observations: list[tuple[Path, Path, Literal["BUILT", "VERIFIED_HIT"], float]] = []
    for request, response, source, output in zip(
        requests, responses, sources, expected_outputs, strict=True
    ):
        if response.status not in {"BUILT", "VERIFIED_HIT"}:
            detail = _bounded_worker_error_detail(response.error_message)
            suffix = f": {detail}" if detail else ""
            raise MageDcvcPrewarmError(
                f"Provider V2 worker did not admit {request.request_id}: "
                f"{response.error_code or response.status}{suffix}"
            )
        if (
            response.preparation_identity != request.preparation_identity
            or response.output_directory is None
            or Path(response.output_directory).expanduser().resolve() != output
        ):
            raise MageDcvcPrewarmError("resident worker response binding differs from request")
        admission = cast(Literal["BUILT", "VERIFIED_HIT"], response.status)
        observations.append((source, output, admission, response.wall_seconds))

    all_hits = all(item[2] == "VERIFIED_HIT" for item in observations)
    if existing_manifest is not None:
        if not all_hits:
            raise MageDcvcPrewarmError(
                "an existing verified manifest may only replay exact VERIFIED_HIT artifacts"
            )
        if manifest_path.read_bytes() != existing_manifest_bytes:
            raise MageDcvcPrewarmError("existing cache manifest changed during verified replay")
        cache_manifest = existing_manifest
        verified_entries = verify_mage_codec_cache_manifest_v2(manifest=cache_manifest)
        replay_mode: Literal[
            "fresh-build", "mixed-recovery", "verified-hit-reconstruction", "exact-verified-replay"
        ] = "exact-verified-replay"
    else:
        cache_manifest = build_mage_codec_cache_manifest_v2(
            checkpoint_manifest=checkpoint_manifest,
            codec_policy=codec_policy,
            effective_config=effective_config,
            cache_base_root=base_root,
            model_directory=model_root,
            observations=observations,
            prewarm_wall_seconds=max(0.0, clock() - started),
        )
        write_mage_codec_cache_manifest_v2(manifest=cache_manifest, path=manifest_path)
        loaded = load_mage_codec_cache_manifest_v2(path=manifest_path)
        if loaded != cache_manifest:
            raise MageDcvcPrewarmError("published cache manifest differs from verified candidate")
        cache_manifest = loaded
        verified_entries = verify_mage_codec_cache_manifest_v2(manifest=cache_manifest)
        built = sum(item[2] == "BUILT" for item in observations)
        replay_mode = (
            "fresh-build"
            if built == len(observations)
            else "verified-hit-reconstruction"
            if built == 0
            else "mixed-recovery"
        )

    job_telemetry, model_load_count, model_load_seconds = _build_job_telemetry(
        requests=tuple(requests),
        responses=responses,
        source_paths=sources,
        outputs=tuple(expected_outputs),
        verified_entries=verified_entries,
    )
    manifest_raw = manifest_path.read_bytes()
    values: dict[str, Any] = {
        "qualification_manifest_semantic_sha256": (
            qualified_provider_manifest.manifest_semantic_sha256
        ),
        "qualified_checkpoint_manifest_sha256": checkpoint_manifest.manifest_sha256,
        "codec_policy_sha256": policy_sha,
        "provider_implementation_sha256": effective_config.provider_implementation_sha256,
        "effective_config_sha256": effective_config.effective_config_sha256,
        "namespace_identity": namespace,
        "replay_mode": replay_mode,
        "exact_verified_replay": replay_mode == "exact-verified-replay",
        "inferred_process_model_load_count": model_load_count,
        "inferred_process_model_load_seconds": model_load_seconds,
        "worker_process": process_telemetry,
        "job_count": len(job_telemetry),
        "built_count": sum(job.admission == "BUILT" for job in job_telemetry),
        "verified_hit_count": sum(job.admission == "VERIFIED_HIT" for job in job_telemetry),
        "prewarm_wall_seconds": max(0.0, clock() - started),
        "effective_config_path": str(config_path),
        "cache_manifest_path": str(manifest_path),
        "cache_manifest_exact_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "cache_manifest_semantic_sha256": cache_manifest.manifest_semantic_sha256,
        "jobs": job_telemetry,
    }
    provisional = MageDcvcPrewarmReportV2.model_construct(
        **values,
        report_semantic_sha256="0" * 64,
    )
    report = MageDcvcPrewarmReportV2(
        **values,
        report_semantic_sha256=_prewarm_report_semantic_sha256(provisional),
    )
    _write_canonical_file(path=report_path, payload=report.model_dump(mode="json"))
    loaded_report = load_mage_dcvc_prewarm_report_v2(path=report_path)
    if loaded_report != report:
        raise MageDcvcPrewarmError("published prewarm report differs from verified candidate")
    return report


def load_mage_dcvc_prewarm_report_v2(*, path: Path) -> MageDcvcPrewarmReportV2:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        report = MageDcvcPrewarmReportV2.model_validate_json(raw, strict=True)
    except (OSError, ValidationError) as error:
        raise MageDcvcPrewarmError("Provider V2 prewarm report is invalid") from error
    if canonical_json_bytes(report.model_dump(mode="json")) != raw:
        raise MageDcvcPrewarmError("Provider V2 prewarm report must use canonical JSON")
    return report


def _normalise_sources(source_paths: Sequence[Path]) -> tuple[Path, ...]:
    sources = tuple(sorted({Path(path).expanduser().resolve() for path in source_paths}, key=str))
    if not sources:
        raise MageDcvcPrewarmError("at least one source segment is required")
    if len(sources) != len(source_paths):
        raise MageDcvcPrewarmError("source segment paths must be unique")
    for source in sources:
        if not source.is_file() or source.is_symlink():
            raise MageDcvcPrewarmError(f"source segment is missing or unsafe: {source}")
    return sources


def _publish_effective_config(
    *, state_root: Path, effective_config: MageDcvcEffectiveConfig
) -> Path:
    root = Path(state_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"effective-config-{effective_config.effective_config_sha256}.json"
    payload = canonical_json_bytes(effective_config.model_dump(mode="json"))
    if path.exists():
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise MageDcvcPrewarmError("could not read existing effective config") from error
        if observed != payload:
            raise MageDcvcPrewarmError("existing effective config bytes differ at immutable path")
        return path
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise MageDcvcPrewarmError("could not publish effective config") from error
    return path


def _validate_windows_cache_output_paths(
    output_directories: Sequence[Path],
    *,
    platform_name: str = os.name,
) -> None:
    """Reject known-unreadable Provider V2 paths before starting expensive work.

    CPython on a Windows host without the system long-path policy can create the
    provider directory and its short assets, then fail to reopen Robata's longer
    sidecars after the atomic rename.  A short cache root is the portable and
    dependency-compatible remedy; extended-length Windows paths are intentionally not
    written into durable manifests.
    """

    if platform_name != "nt":
        return
    required_names = (
        MAGE_DCVC_PREPARATION_SIDECAR_NAME,
        MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME,
    )
    for output in output_directories:
        for name in required_names:
            path = output / name
            character_count = len(str(path))
            if character_count >= _WINDOWS_CONSERVATIVE_MAX_PATH_CHARS:
                raise MageDcvcPrewarmError(
                    "Windows Provider V2 cache path exceeds the conservative MAX_PATH "
                    f"budget before worker start ({character_count} >= "
                    f"{_WINDOWS_CONSERVATIVE_MAX_PATH_CHARS}): {path}. "
                    "Use a shorter --cache-base-root; no DCVC work was admitted."
                )


def _bounded_worker_error_detail(value: str | None) -> str:
    if value is None:
        return ""
    compact = " ".join(value.split())
    if len(compact) <= _WORKER_ERROR_DETAIL_MAX_CHARS:
        return compact
    return compact[: _WORKER_ERROR_DETAIL_MAX_CHARS - 3] + "..."


def _worker_command(
    *,
    worker_python: Path,
    model_directory: Path,
    effective_config_path: Path,
    provider_state_root: Path,
    input_roots: Sequence[Path],
    output_root: Path,
    generation_device: str,
    shared_device_guard_file: Path | None,
    intra_checkpoint_path: Path | None,
    inter_checkpoint_path: Path | None,
) -> tuple[str, ...]:
    executable = Path(worker_python).expanduser().resolve()
    if not executable.is_file():
        raise MageDcvcPrewarmError("worker Python executable is missing")
    command = [
        str(executable),
        "-u",
        "-m",
        MAGE_DCVC_WORKER_MODULE,
        "--model-dir",
        str(model_directory),
        "--effective-config",
        str(effective_config_path),
        "--provider-state-root",
        str(Path(provider_state_root).expanduser().resolve()),
    ]
    for root in input_roots:
        command.extend(("--input-root", str(root)))
    command.extend(("--output-root", str(output_root), "--generation-device", generation_device))
    if shared_device_guard_file is not None:
        command.extend(
            (
                "--shared-device-guard-file",
                str(Path(shared_device_guard_file).expanduser().resolve()),
            )
        )
    if intra_checkpoint_path is not None:
        intra = Path(intra_checkpoint_path).expanduser().resolve()
        command.extend(("--intra-checkpoint", str(intra)))
    if inter_checkpoint_path is not None:
        inter = Path(inter_checkpoint_path).expanduser().resolve()
        command.extend(("--inter-checkpoint", str(inter)))
    return tuple(command)


def _normalise_device(value: str) -> str:
    normalised = value.strip().lower()
    if normalised == "cuda":
        return "cuda:0"
    if normalised == "cpu":
        return normalised
    if normalised.startswith("cuda:") and normalised[5:].isdigit():
        return f"cuda:{int(normalised[5:])}"
    raise MageDcvcPrewarmError("generation device must be cpu, cuda, or cuda:<index>")


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2])
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not prior else os.pathsep.join((source_root, prior))
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _validate_existing_manifest(
    *,
    manifest: MageCodecCacheManifestV2,
    entries: Sequence[MageCodecCacheEntryV2],
    checkpoint_manifest: MageCheckpointManifest,
    codec_policy_sha256: Sha256Digest,
    effective_config: MageDcvcEffectiveConfig,
    cache_base_root: Path,
    source_paths: Sequence[Path],
) -> None:
    expected_sources = tuple(str(path) for path in source_paths)
    observed_sources = tuple(entry.source_path for entry in entries)
    if (
        manifest.checkpoint_manifest_sha256 != checkpoint_manifest.manifest_sha256
        or manifest.codec_policy_sha256 != codec_policy_sha256
        or manifest.provider_implementation_sha256
        != effective_config.provider_implementation_sha256
        or manifest.effective_config != effective_config
        or Path(manifest.cache_base_root).expanduser().resolve() != cache_base_root
        or observed_sources != expected_sources
    ):
        raise MageDcvcPrewarmError("existing cache manifest does not bind this exact prewarm run")


def _build_job_telemetry(
    *,
    requests: Sequence[MageDcvcPreparationRequest],
    responses: Sequence[MageDcvcPreparationResponse],
    source_paths: Sequence[Path],
    outputs: Sequence[Path],
    verified_entries: Sequence[MageCodecCacheEntryV2],
) -> tuple[tuple[MageDcvcPrewarmJobTelemetryV2, ...], int, float]:
    by_source = {entry.source_path: entry for entry in verified_entries}
    jobs: list[MageDcvcPrewarmJobTelemetryV2] = []
    built_job_counts: list[int] = []
    built_load_seconds: list[float] = []
    for request, response, source, output in zip(
        requests, responses, source_paths, outputs, strict=True
    ):
        entry = by_source.get(str(source))
        if entry is None:
            raise MageDcvcPrewarmError("verified cache entries do not cover every source")
        if response.artifact_semantic_sha256 != entry.preparation_artifact_semantic_sha256:
            raise MageDcvcPrewarmError(
                "resident worker artifact digest differs from verified cache entry"
            )
        metadata = entry.provider_metadata
        engine_load_count = _strict_int(metadata, "engine_load_count", minimum=1)
        engine_load_seconds = _strict_float(metadata, "engine_load_seconds")
        completed_jobs = _strict_int(metadata, "worker_completed_job_count", minimum=1)
        reset_count = _strict_int(metadata, "sequence_reset_count_for_job", minimum=1)
        sampled_frames = _strict_int(metadata, "effective_sampled_frame_count", minimum=1)
        max_frame = _strict_int(metadata, "max_encoded_frame_id", minimum=0)
        if engine_load_count != 1 or reset_count != 1:
            raise MageDcvcPrewarmError("artifact does not prove one resident load and one reset")
        if response.status == "BUILT":
            built_job_counts.append(completed_jobs)
            built_load_seconds.append(engine_load_seconds)
        jobs.append(
            MageDcvcPrewarmJobTelemetryV2(
                request_id=request.request_id,
                source_path=str(source),
                output_directory=str(output),
                admission=cast(Literal["BUILT", "VERIFIED_HIT"], response.status),
                response_wall_seconds=response.wall_seconds,
                preparation_identity=request.preparation_identity,
                artifact_semantic_sha256=entry.preparation_artifact_semantic_sha256,
                engine_load_count_in_artifact=engine_load_count,
                engine_load_seconds_in_artifact=engine_load_seconds,
                worker_completed_job_count_in_artifact=completed_jobs,
                sequence_reset_count_for_job=1,
                effective_sampled_frame_count=sampled_frames,
                max_encoded_frame_id=max_frame,
            )
        )
    if built_job_counts:
        if built_job_counts != list(range(1, len(built_job_counts) + 1)):
            raise MageDcvcPrewarmError(
                "built artifacts do not prove one persistent worker sequence"
            )
        first_load = built_load_seconds[0]
        if any(value != first_load for value in built_load_seconds):
            raise MageDcvcPrewarmError("resident engine load telemetry changed between built jobs")
        return tuple(jobs), 1, first_load
    return tuple(jobs), 0, 0.0


def _strict_int(metadata: Mapping[str, object], key: str, *, minimum: int) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MageDcvcPrewarmError(f"provider metadata {key} is missing or invalid")
    return value


def _strict_float(metadata: Mapping[str, object], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MageDcvcPrewarmError(f"provider metadata {key} is missing or invalid")
    parsed = float(value)
    if parsed < 0.0 or not (parsed < float("inf")):
        raise MageDcvcPrewarmError(f"provider metadata {key} is missing or invalid")
    return parsed


def _prewarm_report_semantic_sha256(report: MageDcvcPrewarmReportV2) -> Sha256Digest:
    return semantic_sha256(
        report.model_dump(
            mode="json",
            exclude={"report_semantic_sha256", "effective_config_path", "cache_manifest_path"},
        )
    )


def _write_canonical_file(*, path: Path, payload: object) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        temporary.replace(resolved)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise MageDcvcPrewarmError("could not publish Provider V2 prewarm report") from error


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


__all__ = [
    "MAGE_DCVC_PREWARM_REPORT_V2_VERSION",
    "MageDcvcPrewarmError",
    "MageDcvcPrewarmJobTelemetryV2",
    "MageDcvcPrewarmReportV2",
    "MageDcvcWorkerProcessRunner",
    "MageDcvcWorkerProcessTelemetryV2",
    "load_mage_dcvc_prewarm_report_v2",
    "prewarm_mage_dcvc_provider_v2",
    "run_mage_dcvc_preparation_process",
]
