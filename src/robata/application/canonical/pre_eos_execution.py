"""Provider-neutral execution of eligible stream stages before recording EOS.

The stream scheduler owns leases and terminal acceptance.  This module only
bridges an already-ready stream work plan to the canonical inference pipeline,
then writes the resulting immutable ``ModelInference`` bytes for terminal
evidence.  It intentionally does not define a second inference ledger or a
stream-specific provider adapter.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from threading import Thread
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import TypeAdapter, ValidationError

from robata.application.canonical.runner import CanonicalPreEosInferenceInvocation
from robata.contracts.common import SchemaVersion
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    StreamStage,
    TerminalOutcome,
)
from robata.contracts.stream_planning import StreamWorkItemPlan
from robata.inference.models import InferenceStatus, ModelInference, VisionTask
from robata.inference.orchestrator import OrchestratedAttemptResult
from robata.queue.stream_models import StreamTerminalEvidence

MODEL_INFERENCE_SCHEMA_ID = "https://schemas.robata.dev/model-inference"
_MODEL_INFERENCE_ARTIFACT_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "robata:provider-neutral-pre-eos-model-inference-artifact-v1",
)
_SCHEMA_VERSION_ADAPTER = TypeAdapter(SchemaVersion)


class ProviderNeutralStreamStageExecutionError(RuntimeError):
    """A stream-stage execution cannot produce exact, typed terminal evidence."""


class _PreEosInferencePipeline(Protocol):
    async def execute_pre_eos_inference(
        self,
        invocation: CanonicalPreEosInferenceInvocation,
    ) -> OrchestratedAttemptResult:
        """Execute an invocation through the canonical inference orchestrator."""


StreamInvocationFactory = Callable[
    [StreamWorkItemPlan],
    CanonicalPreEosInferenceInvocation | None,
]


class _ContentAddressedModelInferenceStore:
    """Small exact-byte store shared by the local stream artifact root."""

    def __init__(self, root: Path) -> None:
        try:
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise ProviderNeutralStreamStageExecutionError(
                    "artifact_root must identify a directory"
                )
            self._root = root.resolve(strict=True)
        except ProviderNeutralStreamStageExecutionError:
            raise
        except OSError as error:
            raise ProviderNeutralStreamStageExecutionError(
                f"cannot initialize pre-EOS artifact root: {error}"
            ) from error

    def put(self, payload: bytes, schema_ref: SchemaRef) -> ArtifactEvidenceRef:
        digest = hashlib.sha256(payload).hexdigest()
        path = self.path_for_digest(digest)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as output:
                    output.write(payload)
                    output.flush()
            except FileExistsError as error:
                try:
                    existing = path.read_bytes()
                except OSError as read_error:
                    raise ProviderNeutralStreamStageExecutionError(
                        "cannot verify existing pre-EOS model-inference artifact"
                    ) from read_error
                if existing != payload:
                    raise ProviderNeutralStreamStageExecutionError(
                        "content-addressed pre-EOS model-inference artifact conflicts "
                        "with existing bytes"
                    ) from error
        except ProviderNeutralStreamStageExecutionError:
            raise
        except OSError as error:
            raise ProviderNeutralStreamStageExecutionError(
                "cannot persist pre-EOS model-inference artifact"
            ) from error
        return ArtifactEvidenceRef(
            artifact_id=str(uuid5(_MODEL_INFERENCE_ARTIFACT_NAMESPACE, digest)),
            exact_sha256=digest,
            byte_count=len(payload),
            media_type="application/json",
            schema_ref=schema_ref,
        )

    def path_for_digest(self, digest: str) -> Path:
        return self._root / digest[:2] / f"{digest}.json"


class ProviderNeutralStreamStageExecutor:
    """Synchronously execute provider-neutral QA/event work before EOS.

    The invocation factory must derive the same canonical inference inputs for a
    repeated work plan.  Re-delivery is safe because every execution is routed
    through :meth:`CanonicalOfflinePipeline.execute_pre_eos_inference`, which
    delegates to the canonical orchestrator and its durable evidence ledger.
    """

    def __init__(
        self,
        *,
        pipeline: _PreEosInferencePipeline,
        invocation_factory: StreamInvocationFactory,
        artifact_root: str | Path,
        model_inference_schema_ref: SchemaRef,
        terminal_policy_version: str,
    ) -> None:
        if not callable(getattr(pipeline, "execute_pre_eos_inference", None)):
            raise TypeError("pipeline must implement execute_pre_eos_inference")
        if not callable(invocation_factory):
            raise TypeError("invocation_factory must be callable")
        if not isinstance(artifact_root, (str, Path)):
            raise TypeError("artifact_root must be a str or pathlib.Path")
        if not isinstance(model_inference_schema_ref, SchemaRef):
            raise TypeError("model_inference_schema_ref must be a SchemaRef")
        if model_inference_schema_ref.schema_id != MODEL_INFERENCE_SCHEMA_ID:
            raise ValueError(
                "model_inference_schema_ref must identify the registered model-inference schema"
            )
        try:
            checked_policy = _SCHEMA_VERSION_ADAPTER.validate_python(
                terminal_policy_version,
                strict=True,
            )
        except ValidationError as error:
            raise ValueError("terminal_policy_version must be a schema version") from error

        self._pipeline = pipeline
        self._invocation_factory = invocation_factory
        self._artifact_store = _ContentAddressedModelInferenceStore(Path(artifact_root))
        self._model_inference_schema_ref = SchemaRef.model_validate(
            model_inference_schema_ref.model_dump(mode="python"),
            strict=True,
        )
        self._terminal_policy_version = checked_policy

    def __call__(self, plan: StreamWorkItemPlan) -> StreamTerminalEvidence | None:
        """Allow direct use as the local finalizer's synchronous stage hook."""

        return self.execute(plan)

    def execute(self, plan: StreamWorkItemPlan) -> StreamTerminalEvidence | None:
        """Execute one eligible plan, or decline non-QA/event stream stages."""

        if not isinstance(plan, StreamWorkItemPlan):
            raise TypeError("plan must be a StreamWorkItemPlan")
        expected_task = _TASK_BY_STAGE.get(plan.stage)
        if expected_task is None:
            return None

        invocation = self._invocation_factory(plan)
        if invocation is None:
            return None
        if not isinstance(invocation, CanonicalPreEosInferenceInvocation):
            raise TypeError(
                "invocation_factory must return CanonicalPreEosInferenceInvocation or None"
            )
        if invocation.task is not expected_task:
            raise ProviderNeutralStreamStageExecutionError(
                "pre-EOS invocation task does not match the stream stage"
            )

        orchestrated = _run_synchronously(self._pipeline.execute_pre_eos_inference(invocation))
        if not isinstance(orchestrated, OrchestratedAttemptResult):
            raise TypeError(
                "pipeline.execute_pre_eos_inference must return OrchestratedAttemptResult"
            )
        terminal = orchestrated.terminal
        if not isinstance(terminal, ModelInference):
            raise TypeError("orchestrated result terminal must be a ModelInference")
        if terminal.stage is not expected_task:
            raise ProviderNeutralStreamStageExecutionError(
                "canonical pre-EOS terminal task does not match the stream stage"
            )
        if terminal.status is InferenceStatus.SUCCEEDED:
            selection = orchestrated.selection
            if (
                selection is None
                or terminal.shadow
                or not terminal.output_valid
                or selection.inference_id != terminal.inference_id
                or selection.logical_invocation_id != terminal.logical_invocation_id
            ):
                raise ProviderNeutralStreamStageExecutionError(
                    "successful pre-EOS terminal lacks its canonical selected attempt"
                )

        payload = canonical_json_bytes(terminal)
        evidence_ref = self._artifact_store.put(payload, self._model_inference_schema_ref)
        outcome, reason_code, reason_detail = _stream_terminal_outcome(terminal)
        return StreamTerminalEvidence(
            outcome=outcome,
            evidence_ref=evidence_ref,
            terminal_policy_version=self._terminal_policy_version,
            completed_at=terminal.completed_at,
            reason_code=reason_code,
            reason_detail=reason_detail,
        )

    def artifact_path_for(self, evidence_ref: ArtifactEvidenceRef) -> Path:
        """Return the local exact-byte path for a terminal emitted by this executor."""

        if not isinstance(evidence_ref, ArtifactEvidenceRef):
            raise TypeError("evidence_ref must be an ArtifactEvidenceRef")
        return self._artifact_store.path_for_digest(evidence_ref.exact_sha256)


_TASK_BY_STAGE: dict[StreamStage, VisionTask] = {
    StreamStage.QA_COARSE: VisionTask.QA_COARSE,
    StreamStage.QA_DENSE: VisionTask.QA_DENSE,
    StreamStage.EVENT_PROPOSAL: VisionTask.EVENT_PROPOSAL,
}


def _stream_terminal_outcome(
    terminal: ModelInference,
) -> tuple[TerminalOutcome, str | None, str | None]:
    """Map one canonical inference terminal onto the stream closure vocabulary."""

    if terminal.status is InferenceStatus.SUCCEEDED:
        output = terminal.normalized_output
        if output is not None and output.get("abstained") is True:
            return TerminalOutcome.ABSTAINED, "PROVIDER_ABSTAINED", None
        if terminal.stage is VisionTask.EVENT_PROPOSAL and _is_no_event_output(output):
            return TerminalOutcome.NO_EVENTS, "PROVIDER_NO_EVENTS", None
        return TerminalOutcome.SUCCEEDED, None, None

    failure = terminal.failure
    reason_code = failure.code if failure is not None else f"INFERENCE_{terminal.status.value}"
    reason_detail = None if failure is None else failure.detail
    outcome = (
        TerminalOutcome.CANCELLED
        if terminal.status is InferenceStatus.CANCELLED
        else TerminalOutcome.FAILED
    )
    return outcome, reason_code, reason_detail


def _is_no_event_output(output: dict[str, object] | None) -> bool:
    """Recognize the canonical empty proposal envelope without guessing other output shapes."""

    if output is None:
        return False
    for field in ("claims", "events", "proposals"):
        value = output.get(field)
        if isinstance(value, list) and not value:
            return True
    return False


async def _await_result[Result](awaitable: Awaitable[Result]) -> Result:
    return await awaitable


def _run_synchronously[Result](awaitable: Awaitable[Result]) -> Result:
    """Await a pipeline operation from the scheduler's synchronous hook.

    The normal path owns no running event loop.  A short-lived thread fallback
    keeps focused unit callers and embedding applications from attempting a
    forbidden nested ``asyncio.run`` while retaining a synchronous executor API.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_result(awaitable))

    results: list[Result] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            results.append(asyncio.run(_await_result(awaitable)))
        except BaseException as error:  # Preserve cancellation and provider errors.
            errors.append(error)

    thread = Thread(target=runner, name="robata-pre-eos-stage", daemon=False)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    if not results:
        raise ProviderNeutralStreamStageExecutionError(
            "synchronous pre-EOS execution stopped without a result"
        )
    return results[0]


__all__ = [
    "MODEL_INFERENCE_SCHEMA_ID",
    "ProviderNeutralStreamStageExecutionError",
    "ProviderNeutralStreamStageExecutor",
    "StreamInvocationFactory",
]
