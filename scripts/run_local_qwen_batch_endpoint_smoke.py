"""Qualify the versioned local Qwen Batch4 endpoint and adapter in-process.

The real path is intentionally finite: one resident model, one four-member native
batch, and one exact replay through an httpx ASGI loopback. Every model-bearing
invocation must be launched by run_bounded_qwen_batch_benchmark.py; this child
never starts a TCP server or a background process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sqlite3
import sys
import time
import traceback
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robata.adapters.sqlite_inference_evidence import SQLiteInferenceEvidenceLedger  # noqa: E402
from robata.application.canonical.local_real_model import (  # noqa: E402
    LOCAL_QWEN_MODEL_NAME,
    LOCAL_QWEN_MODEL_VERSION,
    LOCAL_QWEN_MULTI_CLAIM_SERIAL_GUARD_POLICY_VERSION,
    LOCAL_QWEN_NATIVE_BATCH_CAPACITY_PROJECTION_VERSION,
    LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE,
    LOCAL_QWEN_NATIVE_BATCH_POLICY_VERSION,
    LOCAL_QWEN_PROVIDER,
    build_local_qwen_batch_model_binding,
)
from robata.benchmark.qwen_r12_request_corpus import (  # noqa: E402
    QWEN_R12_20260806_EXPECTED,
    QwenRequestCase,
    QwenRequestCorpus,
    load_qwen_request_corpus,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256  # noqa: E402
from robata.contracts.schema_registry import SchemaRegistry  # noqa: E402
from robata.inference.adapter import VisionInferenceFailure, VisionInferenceSuccess  # noqa: E402
from robata.inference.local_hf_adapter import (  # noqa: E402
    LOCAL_HF_HYBRID_BATCH_INPUT_MAX_SIZE,
    LOCAL_HF_HYBRID_BATCH_MAX_SIZE,
    LOCAL_HF_HYBRID_BATCH_POLICY_VERSION,
    LocalHfHttpRequest,
    LocalHfHttpResponse,
    LocalHfLoopbackVisionAdapter,
    LocalHfTransportError,
)
from robata.inference.local_hf_endpoint import (  # noqa: E402
    LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION,
    LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION,
    LOCAL_HF_BATCH_IDEMPOTENCY_POLICY_VERSION,
    LOCAL_HF_BATCH_INFER_PATH,
    LOCAL_HF_BATCH_MAX_SIZE,
    LOCAL_HF_BATCH_POLICY_VERSION,
    LocalHfBatchEndpointResponse,
    LocalHfCheckpointIdentity,
    LocalHfEndpointService,
    create_local_hf_endpoint_app,
)
from robata.inference.local_hf_runtime import LocalHuggingFaceVisionRuntime  # noqa: E402
from robata.inference.models import InferenceStatus, VisionTask  # noqa: E402
from robata.inference.offline_fixture import (  # noqa: E402
    InMemoryRawProviderBytesStore,
    StrictProviderClaimParser,
)

REPORT_VERSION = "qwen-batch4-inprocess-endpoint-smoke-v1"
AUTHORITY = "LOCAL_NONPRODUCTION_ONLY"
DEFAULT_CHECKPOINT_MANIFEST_SHA256 = (
    "1f7293b2629473f0240c8675025e1402da4306f05cc9026adf4c801f20f99f10"
)
DEFAULT_CORPUS_DB = Path(
    r"D:\\tmp\\robata-qwen-run-20260806\\canonical-qwen-full-r12-20260806"
    r"\\inference-evidence.sqlite3"
)
EXIT_FAILURE = 2
_ENDPOINT_SERIAL_TABLE = "local_hf_endpoint_idempotency_v1"
_ENDPOINT_BATCH_TABLE = "local_hf_endpoint_batch_idempotency_v1"
_EVIDENCE_INTENT_TABLE = "inference_intents"
_EVIDENCE_RAW_TABLE = "raw_provider_responses"


class QwenBatchEndpointSmokeError(RuntimeError):
    """The finite endpoint/adapter qualification did not satisfy its invariants."""


class _RuntimeDelegate(Protocol):
    @property
    def loaded(self) -> bool: ...

    @property
    def load_observation(self) -> Any: ...

    def load(self) -> Any: ...

    def close(self) -> None: ...

    def generate(self, **kwargs: Any) -> Any: ...

    def generate_batch(self, **kwargs: Any) -> Any: ...


class CountingLocalVisionRuntime:
    """Count logical runtime entrypoints without changing their results."""

    def __init__(self, delegate: _RuntimeDelegate) -> None:
        self._delegate = delegate
        self.load_calls = 0
        self.close_calls = 0
        self.generate_calls = 0
        self.generate_batch_calls = 0

    @property
    def loaded(self) -> bool:
        return bool(self._delegate.loaded)

    @property
    def load_observation(self) -> Any:
        return self._delegate.load_observation

    def load(self) -> Any:
        self.load_calls += 1
        return self._delegate.load()

    def close(self) -> None:
        self.close_calls += 1
        self._delegate.close()

    def generate(self, **kwargs: Any) -> Any:
        self.generate_calls += 1
        return self._delegate.generate(**kwargs)

    def generate_batch(self, **kwargs: Any) -> Any:
        self.generate_batch_calls += 1
        return self._delegate.generate_batch(**kwargs)


@dataclass(frozen=True, slots=True)
class AsgiExchange:
    request: LocalHfHttpRequest
    response: LocalHfHttpResponse


class HttpxAsgiLocalHfTransport:
    """Implement the adapter transport with exact in-process ASGI HTTP bytes."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._exchanges: list[AsgiExchange] = []

    @property
    def exchanges(self) -> tuple[AsgiExchange, ...]:
        return tuple(self._exchanges)

    async def post(self, request: LocalHfHttpRequest) -> LocalHfHttpResponse:
        if not isinstance(request, LocalHfHttpRequest):
            raise TypeError("request must be LocalHfHttpRequest")
        try:
            import httpx

            transport = httpx.ASGITransport(app=self._app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8101",
                timeout=request.timeout_seconds,
            ) as client:
                response = await client.post(
                    request.url,
                    content=request.body,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "User-Agent": "robata-qwen-batch4-endpoint-smoke-v1",
                        "Idempotency-Key": request.idempotency_key,
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise LocalHfTransportError("in-process local HF ASGI request failed") from error
        body = bytes(response.content)
        if len(body) > request.max_response_bytes:
            raise LocalHfTransportError("in-process local HF ASGI response exceeded byte limit")
        result = LocalHfHttpResponse(status_code=int(response.status_code), body=body)
        self._exchanges.append(AsgiExchange(request=request, response=result))
        return result


class _FailClosedTransport:
    async def post(self, request: LocalHfHttpRequest) -> LocalHfHttpResponse:
        del request
        raise AssertionError("verify-only mode must not dispatch inference")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus-db", type=Path, default=DEFAULT_CORPUS_DB)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-image-side", type=_positive_int, default=448)
    parser.add_argument("--gpu-weight-memory-gib", type=_positive_int, default=7)
    parser.add_argument("--cpu-weight-memory-gib", type=_positive_int, default=1)
    parser.add_argument(
        "--expected-checkpoint-manifest-sha256",
        default=DEFAULT_CHECKPOINT_MANIFEST_SHA256,
    )
    return parser


def _write_json(path: Path, payload: object) -> str:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    body = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = resolved.with_name(f".{resolved.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(body)
        os.replace(temporary, resolved)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return exact_bytes_sha256(body)


def _sqlite_row_counts(path: Path, tables: tuple[str, ...]) -> dict[str, int]:
    resolved = path.expanduser().resolve()
    connection = sqlite3.connect(resolved, isolation_level=None, timeout=30.0)
    try:
        existing = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        missing = tuple(table for table in tables if table not in existing)
        if missing:
            raise QwenBatchEndpointSmokeError(
                f"SQLite evidence is missing required tables: {missing!r}"
            )
        return {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in tables
        }
    finally:
        connection.close()


def _checkpoint_selected_file_count(model_directory: Path) -> int:
    """Count relevant local artifacts without rehashing multi-GB weights."""

    exact_names = {
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "feature_extractor_config.json",
        "generation_config.json",
        "image_processor_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
    weight_suffixes = (".bin", ".ckpt", ".pt", ".pth", ".safetensors")
    count = 0
    resolved = model_directory.expanduser().resolve()
    for path in resolved.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(resolved).parts
        if any(part.lower() in {".cache", ".git", "__pycache__"} for part in relative_parts):
            continue
        name = path.name.lower()
        included = (
            name in exact_names
            or name.endswith(weight_suffixes)
            or (
                name.endswith(".index.json")
                and any(name.endswith(f"{suffix}.index.json") for suffix in weight_suffixes)
            )
        )
        count += int(included)
    if count < 1:
        raise QwenBatchEndpointSmokeError("model directory contains no checkpoint identity files")
    return count


def _build_selection_adapter(*, checkpoint_manifest_sha256: str) -> LocalHfLoopbackVisionAdapter:
    binding = build_local_qwen_batch_model_binding(
        transport=_FailClosedTransport(),
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
    )
    if binding.adapter_factory is None:
        raise QwenBatchEndpointSmokeError("Qwen Batch4 binding has no adapter factory")
    adapter = binding.adapter_factory(
        cast(Any, InMemoryRawProviderBytesStore()),
        StrictProviderClaimParser(
            SchemaRegistry(),
            parser_version="qwen-batch4-endpoint-smoke-selection-v1",
        ),
    )
    if not isinstance(adapter, LocalHfLoopbackVisionAdapter):
        raise QwenBatchEndpointSmokeError("Qwen Batch4 binding built an unexpected adapter")
    return adapter


def _batch_compatibility_key(
    adapter: LocalHfLoopbackVisionAdapter,
    case: QwenRequestCase,
) -> tuple[object, ...] | None:
    _selected_items, endpoint_request, _body = adapter._prepare_endpoint_request(case.request)
    try:
        prompt = json.loads(endpoint_request.prompt)
        claim_group_count = int(prompt["compact_output_contract"]["claim_group_count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise QwenBatchEndpointSmokeError(
            f"invalid compact claim-group contract for corpus ordinal {case.ordinal}"
        ) from error
    if claim_group_count != 1:
        return None
    request = case.request
    schema = request.output_schema
    return (
        request.task.value,
        claim_group_count,
        endpoint_request.max_new_tokens,
        request.model_name,
        request.model_version,
        request.prompt_version,
        request.prompt_sha256,
        schema.schema_id,
        schema.version,
        schema.sha256,
    )


def _select_first_compatible_single_claim_cases(
    *,
    corpus: QwenRequestCorpus,
    adapter: LocalHfLoopbackVisionAdapter,
) -> tuple[QwenRequestCase, ...]:
    selected: list[QwenRequestCase] = []
    selected_key: tuple[object, ...] | None = None
    for case in corpus.cases:
        if case.request.task is not VisionTask.QA_COARSE:
            continue
        compatibility = _batch_compatibility_key(adapter, case)
        if compatibility is None:
            continue
        if selected_key is None:
            selected_key = compatibility
        if compatibility != selected_key:
            continue
        selected.append(case)
        if len(selected) == LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE:
            break
    if len(selected) != LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE:
        raise QwenBatchEndpointSmokeError(
            "frozen corpus does not contain four compatible single-claim QA_COARSE requests"
        )
    return tuple(selected)


def _load_baseline(
    database_path: Path,
    cases: tuple[QwenRequestCase, ...],
) -> dict[str, dict[str, Any]]:
    requested_ids = {case.intent.inference_id for case in cases}
    connection = sqlite3.connect(
        f"{database_path.expanduser().resolve().as_uri()}?mode=ro",
        uri=True,
        isolation_level=None,
    )
    try:
        terminals = {
            str(row[0]): (bytes(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT inference_id, payload_json, payload_sha256 FROM model_inference_terminals"
            )
            if row[0] in requested_ids
        }
        responses = {
            str(row[0]): (str(row[1]), int(row[2]))
            for row in connection.execute(
                "SELECT inference_id, exact_bytes_sha256, byte_count FROM raw_provider_responses"
            )
            if row[0] in requested_ids
        }
    finally:
        connection.close()
    if set(terminals) != requested_ids or set(responses) != requested_ids:
        raise QwenBatchEndpointSmokeError("frozen r12 baseline evidence is incomplete")
    baseline: dict[str, dict[str, Any]] = {}
    state_directory = database_path.expanduser().resolve().parent
    for case in cases:
        inference_id = case.intent.inference_id
        terminal_bytes, terminal_sha256 = terminals[inference_id]
        if exact_bytes_sha256(terminal_bytes) != terminal_sha256:
            raise QwenBatchEndpointSmokeError("frozen terminal payload digest mismatch")
        terminal = json.loads(terminal_bytes)
        if canonical_json_bytes(terminal) != terminal_bytes:
            raise QwenBatchEndpointSmokeError("frozen terminal payload is not canonical JSON")
        raw_sha256, raw_byte_count = responses[inference_id]
        raw_path = state_directory / "raw-provider-cas" / raw_sha256[:2] / raw_sha256
        raw_bytes = raw_path.read_bytes()
        if len(raw_bytes) != raw_byte_count or exact_bytes_sha256(raw_bytes) != raw_sha256:
            raise QwenBatchEndpointSmokeError("frozen raw provider evidence digest mismatch")
        baseline[inference_id] = {
            "terminal_payload_sha256": terminal_sha256,
            "normalized_output": terminal["normalized_output"],
            "raw_output": raw_bytes.decode("utf-8"),
            "raw_output_exact_sha256": raw_sha256,
        }
    return baseline


def _route_projection(binding: Any, adapter: LocalHfLoopbackVisionAdapter) -> dict[str, Any]:
    capability = adapter.native_batch_capability
    admission = binding.native_batch_admission
    if admission is None:
        raise QwenBatchEndpointSmokeError("canonical Batch4 binding has no admission evidence")
    expected = {
        "adapter_policy_version": LOCAL_QWEN_NATIVE_BATCH_POLICY_VERSION,
        "adapter_max_batch_size": LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE,
        "endpoint_policy_version": LOCAL_HF_BATCH_POLICY_VERSION,
        "endpoint_max_batch_size": LOCAL_HF_HYBRID_BATCH_MAX_SIZE,
    }
    observed = {
        "adapter_policy_version": adapter.native_batch_policy_version,
        "adapter_max_batch_size": adapter.native_batch_max_size,
        "endpoint_policy_version": capability.endpoint_policy_version,
        "endpoint_max_batch_size": capability.max_batch_size,
    }
    if observed != expected:
        raise QwenBatchEndpointSmokeError(
            f"canonical Batch4 route policy drift: observed={observed!r}, expected={expected!r}"
        )
    if LOCAL_HF_HYBRID_BATCH_INPUT_MAX_SIZE != 8 or LOCAL_HF_HYBRID_BATCH_MAX_SIZE != 4:
        raise QwenBatchEndpointSmokeError("adapter hybrid dispatch bounds changed")
    return {
        "model_provider": LOCAL_QWEN_PROVIDER,
        "model_name": LOCAL_QWEN_MODEL_NAME,
        "model_version": LOCAL_QWEN_MODEL_VERSION,
        "adapter_policy_version": LOCAL_HF_HYBRID_BATCH_POLICY_VERSION,
        "adapter_max_batch_size": LOCAL_HF_HYBRID_BATCH_MAX_SIZE,
        "adapter_max_dispatch_size": LOCAL_HF_HYBRID_BATCH_INPUT_MAX_SIZE,
        "adapter_capability": capability.model_dump(mode="json"),
        "canonical_admission": {
            "policy_version": admission.policy_version,
            "max_batch_size": admission.max_batch_size,
            "capacity_projection_version": admission.capacity_projection_version,
            "serial_guard_policy_version": admission.serial_guard_policy_version,
        },
        "canonical_runtime_capacity_projection": binding.runtime_capacity_projection,
        "expected_capacity_projection_version": LOCAL_QWEN_NATIVE_BATCH_CAPACITY_PROJECTION_VERSION,
        "expected_multi_claim_serial_guard_policy_version": (
            LOCAL_QWEN_MULTI_CLAIM_SERIAL_GUARD_POLICY_VERSION
        ),
        "endpoint_request_version": LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION,
        "endpoint_response_version": LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION,
        "endpoint_policy_version": LOCAL_HF_BATCH_POLICY_VERSION,
        "endpoint_idempotency_policy_version": LOCAL_HF_BATCH_IDEMPOTENCY_POLICY_VERSION,
        "endpoint_path": LOCAL_HF_BATCH_INFER_PATH,
        "endpoint_max_batch_size": LOCAL_HF_BATCH_MAX_SIZE,
        "rollback": "build_local_qwen_model_binding (unchanged serial control)",
    }


def _parse_batch_exchange(exchange: AsgiExchange) -> LocalHfBatchEndpointResponse:
    if exchange.response.status_code != 200:
        raise QwenBatchEndpointSmokeError(
            f"batch endpoint returned HTTP {exchange.response.status_code}: "
            f"{exchange.response.body[:1000]!r}"
        )
    try:
        return LocalHfBatchEndpointResponse.model_validate_json(exchange.response.body, strict=True)
    except (TypeError, ValueError) as error:
        raise QwenBatchEndpointSmokeError("batch endpoint response is invalid") from error


def _successes(
    outcomes: tuple[VisionInferenceSuccess | VisionInferenceFailure, ...],
    *,
    label: str,
) -> tuple[VisionInferenceSuccess, ...]:
    if len(outcomes) != LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE:
        raise QwenBatchEndpointSmokeError(f"{label} returned the wrong outcome count")
    failures = [outcome for outcome in outcomes if not isinstance(outcome, VisionInferenceSuccess)]
    if failures:
        raise QwenBatchEndpointSmokeError(
            f"{label} did not return four successes: "
            + ", ".join(f"{outcome.status.value}:{outcome.failure.code}" for outcome in failures)
        )
    successes = cast(tuple[VisionInferenceSuccess, ...], outcomes)
    if any(outcome.status is not InferenceStatus.SUCCEEDED for outcome in successes):
        raise QwenBatchEndpointSmokeError(f"{label} contains a non-succeeded outcome")
    return successes


async def _execute_two_passes(
    *,
    adapter: LocalHfLoopbackVisionAdapter,
    cases: tuple[QwenRequestCase, ...],
    transport: HttpxAsgiLocalHfTransport,
    runtime: CountingLocalVisionRuntime,
    evidence_ledger: SQLiteInferenceEvidenceLedger,
    endpoint_state_path: Path,
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    requests = tuple(case.request for case in cases)

    first_started = time.perf_counter()
    first_outcomes = _successes(await adapter.infer_batch(requests), label="first pass")
    first_wall = time.perf_counter() - first_started
    if len(transport.exchanges) != 1:
        raise QwenBatchEndpointSmokeError("first pass did not issue exactly one ASGI request")
    first_endpoint = _parse_batch_exchange(transport.exchanges[0])
    first_generate_batch_calls = runtime.generate_batch_calls
    first_endpoint_counts = _sqlite_row_counts(
        endpoint_state_path, (_ENDPOINT_SERIAL_TABLE, _ENDPOINT_BATCH_TABLE)
    )
    first_evidence_counts = _sqlite_row_counts(
        evidence_ledger.database_path, (_EVIDENCE_INTENT_TABLE, _EVIDENCE_RAW_TABLE)
    )

    replay_started = time.perf_counter()
    replay_outcomes = _successes(await adapter.infer_batch(requests), label="replay pass")
    replay_wall = time.perf_counter() - replay_started
    if len(transport.exchanges) != 2:
        raise QwenBatchEndpointSmokeError("replay pass did not issue exactly one ASGI request")
    replay_endpoint = _parse_batch_exchange(transport.exchanges[1])
    replay_generate_batch_calls = runtime.generate_batch_calls
    replay_endpoint_counts = _sqlite_row_counts(
        endpoint_state_path, (_ENDPOINT_SERIAL_TABLE, _ENDPOINT_BATCH_TABLE)
    )
    replay_evidence_counts = _sqlite_row_counts(
        evidence_ledger.database_path, (_EVIDENCE_INTENT_TABLE, _EVIDENCE_RAW_TABLE)
    )

    if runtime.generate_calls != 0:
        raise QwenBatchEndpointSmokeError("native Batch4 smoke unexpectedly used serial generate")
    if first_generate_batch_calls != 1 or replay_generate_batch_calls != 1:
        raise QwenBatchEndpointSmokeError(
            "first pass must perform one physical generate_batch and replay must perform none"
        )
    if first_endpoint.generated_member_count != 4 or first_endpoint.replay_member_count != 0:
        raise QwenBatchEndpointSmokeError("first endpoint pass did not generate all four members")
    if replay_endpoint.generated_member_count != 0 or replay_endpoint.replay_member_count != 4:
        raise QwenBatchEndpointSmokeError("second endpoint pass was not a four-member replay")
    if any(member.disposition != "GENERATED" for member in first_endpoint.members):
        raise QwenBatchEndpointSmokeError("first endpoint member disposition drift")
    if any(member.disposition != "REPLAY" for member in replay_endpoint.members):
        raise QwenBatchEndpointSmokeError("replay endpoint member disposition drift")
    expected_endpoint_counts = {_ENDPOINT_SERIAL_TABLE: 0, _ENDPOINT_BATCH_TABLE: 4}
    if first_endpoint_counts != expected_endpoint_counts:
        raise QwenBatchEndpointSmokeError("first endpoint SQLite row counts are incorrect")
    if replay_endpoint_counts != expected_endpoint_counts:
        raise QwenBatchEndpointSmokeError("replay changed endpoint SQLite row counts")
    expected_evidence_counts = {_EVIDENCE_INTENT_TABLE: 4, _EVIDENCE_RAW_TABLE: 4}
    if first_evidence_counts != expected_evidence_counts:
        raise QwenBatchEndpointSmokeError("first pass evidence row counts are incorrect")
    if replay_evidence_counts != expected_evidence_counts:
        raise QwenBatchEndpointSmokeError("replay changed evidence row counts")

    case_reports: list[dict[str, Any]] = []
    for index, (case, first, replay, first_member, replay_member) in enumerate(
        zip(
            cases,
            first_outcomes,
            replay_outcomes,
            first_endpoint.members,
            replay_endpoint.members,
            strict=True,
        )
    ):
        historical = baseline[case.intent.inference_id]
        normalized_match = first.normalized_output.payload == historical["normalized_output"]
        replay_normalized_match = (
            replay.normalized_output.payload == historical["normalized_output"]
        )
        raw_exact_match = first_member.output_text == historical["raw_output"]
        replay_raw_exact_match = replay_member.output_text == historical["raw_output"]
        replay_same_artifact = first.raw_output_artifact_id == replay.raw_output_artifact_id
        if (
            not normalized_match
            or not replay_normalized_match
            or not raw_exact_match
            or not replay_raw_exact_match
            or not replay_same_artifact
        ):
            raise QwenBatchEndpointSmokeError(
                f"case ordinal {case.ordinal} failed strict raw/normalized/replay parity"
            )
        case_reports.append(
            {
                "batch_member_ordinal": index,
                "corpus_case_ordinal": case.ordinal,
                "inference_id": case.intent.inference_id,
                "request_id": case.request.request_id,
                "task": case.request.task.value,
                "selected_image_sha256": [image.sha256 for image in case.selected_images],
                "baseline_terminal_payload_sha256": historical["terminal_payload_sha256"],
                "baseline_raw_output_exact_sha256": historical["raw_output_exact_sha256"],
                "first_raw_output_exact_sha256": exact_bytes_sha256(
                    first_member.output_text.encode("utf-8")
                ),
                "replay_raw_output_exact_sha256": exact_bytes_sha256(
                    replay_member.output_text.encode("utf-8")
                ),
                "normalized_exact_match": normalized_match,
                "replay_normalized_exact_match": replay_normalized_match,
                "raw_exact_match": raw_exact_match,
                "replay_raw_exact_match": replay_raw_exact_match,
                "raw_evidence_artifact_id": first.raw_output_artifact_id,
                "replay_same_raw_evidence_artifact": replay_same_artifact,
                "first_usage": first.usage.model_dump(mode="json"),
                "replay_usage": replay.usage.model_dump(mode="json"),
            }
        )

    return {
        "first_pass_wall_seconds": first_wall,
        "replay_pass_wall_seconds": replay_wall,
        "runtime_calls_after_first": {
            "load": runtime.load_calls,
            "generate": 0,
            "generate_batch": first_generate_batch_calls,
        },
        "runtime_calls_after_replay": {
            "load": runtime.load_calls,
            "generate": runtime.generate_calls,
            "generate_batch": replay_generate_batch_calls,
        },
        "first_endpoint_response": {
            "batch_request_sha256": first_endpoint.batch_request_sha256,
            "physical_generation_seconds": first_endpoint.physical_generation_seconds,
            "physical_gpu_peak_allocated_bytes": first_endpoint.physical_gpu_peak_allocated_bytes,
            "generated_member_count": first_endpoint.generated_member_count,
            "replay_member_count": first_endpoint.replay_member_count,
            "member_dispositions": [member.disposition for member in first_endpoint.members],
        },
        "replay_endpoint_response": {
            "batch_request_sha256": replay_endpoint.batch_request_sha256,
            "physical_generation_seconds": replay_endpoint.physical_generation_seconds,
            "physical_gpu_peak_allocated_bytes": replay_endpoint.physical_gpu_peak_allocated_bytes,
            "generated_member_count": replay_endpoint.generated_member_count,
            "replay_member_count": replay_endpoint.replay_member_count,
            "member_dispositions": [member.disposition for member in replay_endpoint.members],
        },
        "endpoint_sqlite_rows_after_first": first_endpoint_counts,
        "endpoint_sqlite_rows_after_replay": replay_endpoint_counts,
        "evidence_sqlite_rows_after_first": first_evidence_counts,
        "evidence_sqlite_rows_after_replay": replay_evidence_counts,
        "cases": case_reports,
        "quality": {
            "success_count_first": len(first_outcomes),
            "success_count_replay": len(replay_outcomes),
            "normalized_exact_match_count": sum(
                bool(case["normalized_exact_match"]) for case in case_reports
            ),
            "replay_normalized_exact_match_count": sum(
                bool(case["replay_normalized_exact_match"]) for case in case_reports
            ),
            "raw_exact_match_count": sum(bool(case["raw_exact_match"]) for case in case_reports),
            "replay_raw_exact_match_count": sum(
                bool(case["replay_raw_exact_match"]) for case in case_reports
            ),
            "quality_gate_pass": all(
                bool(case["normalized_exact_match"])
                and bool(case["replay_normalized_exact_match"])
                and bool(case["raw_exact_match"])
                and bool(case["replay_raw_exact_match"])
                and bool(case["replay_same_raw_evidence_artifact"])
                for case in case_reports
            ),
        },
    }


def _common_report(
    *,
    started_at: str,
    arguments: argparse.Namespace,
    corpus: QwenRequestCorpus,
    cases: tuple[QwenRequestCase, ...],
    baseline: dict[str, dict[str, Any]],
    route: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_version": REPORT_VERSION,
        "authority": AUTHORITY,
        "production_eligible": False,
        "started_at": started_at,
        "configuration": {
            "model_directory": str(arguments.model_dir.expanduser().resolve()),
            "corpus_database": str(arguments.corpus_db.expanduser().resolve()),
            "max_image_side": arguments.max_image_side,
            "gpu_weight_memory_gib": arguments.gpu_weight_memory_gib,
            "cpu_weight_memory_gib": arguments.cpu_weight_memory_gib,
            "verify_only": arguments.verify_only,
            "execution_boundary": "ONE_PROCESS_INPROCESS_ASGI_NO_TCP",
        },
        "route": route,
        "corpus": {
            "database_sha256": corpus.database_sha256,
            "manifest_semantic_sha256": corpus.semantic_sha256,
            "selected_case_ordinals": [case.ordinal for case in cases],
            "selected_case_count": len(cases),
            "selection_policy": "FIRST_FOUR_COMPATIBLE_SINGLE_CLAIM_QA_COARSE_V1",
            "baseline": [
                {
                    "case_ordinal": case.ordinal,
                    "inference_id": case.intent.inference_id,
                    "terminal_payload_sha256": baseline[case.intent.inference_id][
                        "terminal_payload_sha256"
                    ],
                    "raw_output_exact_sha256": baseline[case.intent.inference_id][
                        "raw_output_exact_sha256"
                    ],
                }
                for case in cases
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    started_at = _utc_now()
    started = time.perf_counter()
    output_dir = arguments.output_dir.expanduser().resolve()
    report_path = (
        arguments.report_json.expanduser().resolve()
        if arguments.report_json is not None
        else output_dir / "qwen-batch4-endpoint-smoke.json"
    )
    report: dict[str, Any]
    service: LocalHfEndpointService | None = None
    evidence_ledger: SQLiteInferenceEvidenceLedger | None = None
    runtime: CountingLocalVisionRuntime | None = None
    service_started = False
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        corpus = load_qwen_request_corpus(
            database_path=arguments.corpus_db,
            expected=QWEN_R12_20260806_EXPECTED,
        )
        selection_adapter = _build_selection_adapter(
            checkpoint_manifest_sha256=arguments.expected_checkpoint_manifest_sha256,
        )
        cases = _select_first_compatible_single_claim_cases(
            corpus=corpus,
            adapter=selection_adapter,
        )
        baseline = _load_baseline(arguments.corpus_db, cases)

        if arguments.verify_only:
            binding = build_local_qwen_batch_model_binding(
                transport=_FailClosedTransport(),
                checkpoint_manifest_sha256=arguments.expected_checkpoint_manifest_sha256,
            )
            route = _route_projection(binding, selection_adapter)
            report = {
                **_common_report(
                    started_at=started_at,
                    arguments=arguments,
                    corpus=corpus,
                    cases=cases,
                    baseline=baseline,
                    route=route,
                ),
                "status": "CORPUS_VERIFIED",
                "finished_at": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "model": None,
                "execution": None,
                "durability": None,
                "error": None,
            }
            exit_code = 0
        else:
            model_directory = arguments.model_dir.expanduser().resolve()
            if not model_directory.is_dir():
                raise QwenBatchEndpointSmokeError(
                    f"model directory does not exist: {model_directory}"
                )
            checkpoint_identity = LocalHfCheckpointIdentity(
                manifest_sha256=arguments.expected_checkpoint_manifest_sha256,
                included_file_count=_checkpoint_selected_file_count(model_directory),
            )
            endpoint_state_path = output_dir / "endpoint-idempotency.sqlite3"
            evidence_state_path = output_dir / "inference-evidence.sqlite3"
            raw_cas_root = output_dir / "raw-provider-cas"
            for path in (endpoint_state_path, evidence_state_path):
                if path.exists():
                    raise QwenBatchEndpointSmokeError(
                        f"refusing to reuse prior smoke state: {path}"
                    )

            delegate = LocalHuggingFaceVisionRuntime(
                model_directory=model_directory,
                offload_directory=output_dir / "offload",
                max_image_side=arguments.max_image_side,
                gpu_weight_memory_gib=arguments.gpu_weight_memory_gib,
                cpu_weight_memory_gib=arguments.cpu_weight_memory_gib,
            )
            runtime = CountingLocalVisionRuntime(cast(_RuntimeDelegate, delegate))
            service = LocalHfEndpointService(
                runtime=runtime,
                model_identifier=LOCAL_QWEN_MODEL_NAME,
                model_version=LOCAL_QWEN_MODEL_VERSION,
                checkpoint_identity=checkpoint_identity,
                idempotency_state_path=endpoint_state_path,
            )
            app = create_local_hf_endpoint_app(service)
            transport = HttpxAsgiLocalHfTransport(app)
            binding = build_local_qwen_batch_model_binding(
                transport=transport,
                checkpoint_manifest_sha256=arguments.expected_checkpoint_manifest_sha256,
            )
            registry = SchemaRegistry()
            evidence_ledger = SQLiteInferenceEvidenceLedger(
                evidence_state_path,
                registry,
                raw_bytes_cas_root=raw_cas_root,
            )
            for case in cases:
                evidence_ledger.append_intent(case.intent)
            if binding.adapter_factory is None:
                raise QwenBatchEndpointSmokeError("Qwen Batch4 binding has no adapter factory")
            adapter = binding.adapter_factory(
                evidence_ledger,
                StrictProviderClaimParser(
                    registry,
                    parser_version="qwen-batch4-endpoint-smoke-parser-v1",
                ),
            )
            if not isinstance(adapter, LocalHfLoopbackVisionAdapter):
                raise QwenBatchEndpointSmokeError("Qwen Batch4 binding built wrong adapter type")
            route = _route_projection(binding, adapter)

            load_started = time.perf_counter()
            service_started = True
            service.start()
            load_wall_seconds = time.perf_counter() - load_started
            load_observation = runtime.load_observation
            execution = asyncio.run(
                _execute_two_passes(
                    adapter=adapter,
                    cases=cases,
                    transport=transport,
                    runtime=runtime,
                    evidence_ledger=evidence_ledger,
                    endpoint_state_path=endpoint_state_path,
                    baseline=baseline,
                )
            )
            service.stop()
            service_started = False
            evidence_ledger.close()
            evidence_ledger = None

            report = {
                **_common_report(
                    started_at=started_at,
                    arguments=arguments,
                    corpus=corpus,
                    cases=cases,
                    baseline=baseline,
                    route=route,
                ),
                "status": "SUCCEEDED",
                "finished_at": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "model": {
                    "checkpoint_identity": checkpoint_identity.model_dump(mode="json"),
                    "checkpoint_identity_source": "PINNED_R12_EXPECTED_SHA256",
                    "load_wall_seconds": load_wall_seconds,
                    "runtime_load_seconds": load_observation.load_seconds,
                    "gpu_name": load_observation.gpu_name,
                    "gpu_total_bytes": load_observation.gpu_total_bytes,
                    "gpu_free_before_bytes": load_observation.gpu_free_before_bytes,
                    "gpu_allocated_after_load_bytes": (
                        load_observation.gpu_allocated_after_load_bytes
                    ),
                    "load_calls": runtime.load_calls,
                    "close_calls": runtime.close_calls,
                },
                "execution": execution,
                "durability": {
                    "endpoint_sqlite_path": str(endpoint_state_path),
                    "endpoint_sqlite_exact_sha256": exact_bytes_sha256(
                        endpoint_state_path.read_bytes()
                    ),
                    "evidence_sqlite_path": str(evidence_state_path),
                    "evidence_sqlite_exact_sha256": exact_bytes_sha256(
                        evidence_state_path.read_bytes()
                    ),
                    "raw_provider_cas_root": str(raw_cas_root),
                    "raw_provider_cas_file_count": sum(
                        1 for path in raw_cas_root.rglob("*") if path.is_file()
                    ),
                },
                "error": None,
            }
            exit_code = 0
    except Exception as error:
        report = {
            "report_version": REPORT_VERSION,
            "authority": AUTHORITY,
            "production_eligible": False,
            "status": "FAILED",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "configuration": {
                "model_directory": str(arguments.model_dir.expanduser().resolve()),
                "corpus_database": str(arguments.corpus_db.expanduser().resolve()),
                "output_directory": str(output_dir),
                "verify_only": arguments.verify_only,
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        exit_code = EXIT_FAILURE
    finally:
        if service_started and service is not None:
            with suppress(Exception):
                service.stop()
        if evidence_ledger is not None:
            with suppress(Exception):
                evidence_ledger.close()

    report_sha256 = _write_json(report_path, report)
    print(
        json.dumps(
            {
                "ok": exit_code == 0,
                "status": report["status"],
                "report": str(report_path),
                "report_exact_sha256": report_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
