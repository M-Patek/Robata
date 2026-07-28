"""Durable canonical bridge for sealed adaptive-sampling decisions.

The adaptive decision store owns *which* additional coordinates are justified.
This module owns the intentionally narrow execution boundary around that decision:
seal the decision, durably publish one execution intent, then run extra work only
for a scheduled-target decision.  It never turns a sampling decision into an
event-presence claim.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, StringConstraints, model_validator

from robata.application.canonical.models import CanonicalOfflineRunStatus
from robata.application.canonical.result_validation import CanonicalOfflineRunResult
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.sampling.adaptive_decision import (
    AdaptiveIncrementalTarget,
    AdaptiveSamplingDecision,
    AdaptiveSamplingDecisionOutcome,
)
from robata.sampling.adaptive_decision_store import (
    SQLiteAdaptiveDecisionStore,
)
from robata.tempfiles import make_temp_file

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
_EXECUTION_INTENT_SCHEMA_VERSION: Literal["1.0"] = "1.0"
_TERMINAL_RECEIPT_SCHEMA_VERSION: Literal["1.0"] = "1.0"
_EXECUTION_INTENT_PROJECTION_VERSION: Literal[
    "canonical-adaptive-sampling-execution-intent-semantic-v1"
] = "canonical-adaptive-sampling-execution-intent-semantic-v1"
_TERMINAL_RECEIPT_PROJECTION_VERSION: Literal[
    "canonical-adaptive-sampling-terminal-receipt-semantic-v1"
] = "canonical-adaptive-sampling-terminal-receipt-semantic-v1"


class AdaptiveSamplingExecutionError(RuntimeError):
    """Base error for the internal adaptive-sampling execution bridge."""


class AdaptiveSamplingExecutionConflict(AdaptiveSamplingExecutionError):
    """An execution path already contains different immutable canonical bytes."""


class AdaptiveSamplingExecutionStorageError(AdaptiveSamplingExecutionError):
    """An execution artifact is malformed, tampered with, or cannot be stored."""


class CanonicalAdaptiveSamplingAuthorityError(AdaptiveSamplingExecutionError):
    """A decision is not bound to the completed canonical result supplied."""


class AdaptiveSamplingExecutionTerminalKind(StrEnum):
    """Terminal states local to extra sampling, deliberately unrelated to events."""

    ADDITIONAL_TARGETS_COMPLETED = "ADDITIONAL_TARGETS_COMPLETED"
    NO_ADDITIONAL_WORK = "NO_ADDITIONAL_WORK"


class AdaptiveSamplingExecutionStatus(StrEnum):
    """The bridge result for this invocation, not a recording terminal status."""

    EXECUTED = "EXECUTED"
    REPLAYED = "REPLAYED"
    NO_ADDITIONAL_WORK = "NO_ADDITIONAL_WORK"


class AdaptiveSamplingExecutionIntent(StrictModel):
    """Immutable work intent published before any additional target is materialized.

    ``execution_id`` is also the callback/provider idempotency key.  It derives
    from the frozen decision, so failure recovery can safely invoke a provider
    again without allocating another logical extra-sampling execution.
    """

    schema_version: Literal["1.0"] = _EXECUTION_INTENT_SCHEMA_VERSION
    execution_id: NonEmptyString
    semantic_sha256: Sha256Digest
    decision_id: NonEmptyString
    decision_scope_sha256: Sha256Digest
    decision_semantic_sha256: Sha256Digest
    decision_outcome: AdaptiveSamplingDecisionOutcome
    incremental_targets_sha256: Sha256Digest
    target_count: Annotated[int, Field(strict=True, ge=0)]
    projection_version: Literal["canonical-adaptive-sampling-execution-intent-semantic-v1"] = (
        _EXECUTION_INTENT_PROJECTION_VERSION
    )

    @property
    def idempotency_key(self) -> str:
        """Return the stable key passed to the materialization/provider callback."""

        return self.execution_id

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = semantic_sha256(adaptive_sampling_execution_intent_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("execution intent semantic digest does not match its projection")
        if self.execution_id != f"canonical-adaptive-sampling-execution:{expected}":
            raise ValueError("execution intent ID does not match its semantic digest")
        expected_target_count = 0
        if self.decision_outcome is AdaptiveSamplingDecisionOutcome.ADDITIONAL_TARGETS_SCHEDULED:
            if self.target_count < 1:
                raise ValueError("scheduled execution intent requires at least one target")
            expected_target_count = self.target_count
        elif self.target_count != 0:
            raise ValueError("no-work execution intent cannot retain additional targets")
        if expected_target_count != self.target_count:
            raise ValueError("execution intent target count is inconsistent")
        return self


class AdaptiveSamplingWorkReceipt(StrictModel):
    """A callback-owned immutable reference to completed additional-target work."""

    work_product_sha256: Sha256Digest
    work_product_locator: NonEmptyString


class AdaptiveSamplingExecutionTerminalReceipt(StrictModel):
    """Append-only outcome of one extra-sampling execution intent.

    No field contains an event outcome.  ``NO_ADDITIONAL_WORK`` says only that
    this decision scheduled no *extra sampling*; it cannot be translated into
    an event-pipeline ``NO_EVENTS`` result.
    """

    schema_version: Literal["1.0"] = _TERMINAL_RECEIPT_SCHEMA_VERSION
    receipt_id: NonEmptyString
    semantic_sha256: Sha256Digest
    execution_id: NonEmptyString
    intent_semantic_sha256: Sha256Digest
    decision_id: NonEmptyString
    decision_semantic_sha256: Sha256Digest
    terminal_kind: AdaptiveSamplingExecutionTerminalKind
    no_additional_work_outcome: AdaptiveSamplingDecisionOutcome | None = None
    work_receipt: AdaptiveSamplingWorkReceipt | None = None
    projection_version: Literal["canonical-adaptive-sampling-terminal-receipt-semantic-v1"] = (
        _TERMINAL_RECEIPT_PROJECTION_VERSION
    )

    @model_validator(mode="after")
    def validate_identity_and_shape(self) -> Self:
        if self.terminal_kind is AdaptiveSamplingExecutionTerminalKind.ADDITIONAL_TARGETS_COMPLETED:
            if self.work_receipt is None or self.no_additional_work_outcome is not None:
                raise ValueError("completed additional-target receipt requires only a work receipt")
        else:
            if self.work_receipt is not None or self.no_additional_work_outcome is None:
                raise ValueError("no-additional-work receipt requires only a no-work outcome")
            if (
                self.no_additional_work_outcome
                is AdaptiveSamplingDecisionOutcome.ADDITIONAL_TARGETS_SCHEDULED
            ):
                raise ValueError("scheduled targets cannot be finalized as no additional work")
        expected = semantic_sha256(adaptive_sampling_execution_terminal_receipt_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("terminal receipt semantic digest does not match its projection")
        if self.receipt_id != f"canonical-adaptive-sampling-terminal-receipt:{expected}":
            raise ValueError("terminal receipt ID does not match its semantic digest")
        return self


class AdaptiveSamplingExecutionResult(StrictModel):
    """Invocation result carrying the frozen execution authority and terminal proof."""

    status: AdaptiveSamplingExecutionStatus
    decision: AdaptiveSamplingDecision
    intent: AdaptiveSamplingExecutionIntent
    terminal_receipt: AdaptiveSamplingExecutionTerminalReceipt
    decision_replayed: bool
    terminal_replayed: bool


AdaptiveSamplingExecutionCallback = Callable[
    [AdaptiveSamplingExecutionIntent, tuple[AdaptiveIncrementalTarget, ...]],
    AdaptiveSamplingWorkReceipt,
]


def adaptive_sampling_execution_intent_projection(
    intent: AdaptiveSamplingExecutionIntent,
) -> dict[str, object]:
    """Return the versioned projection that identifies one extra-work execution."""

    if not isinstance(intent, AdaptiveSamplingExecutionIntent):
        raise TypeError("intent must be an AdaptiveSamplingExecutionIntent")
    return {
        "projection_version": intent.projection_version,
        "schema_version": intent.schema_version,
        "decision_id": intent.decision_id,
        "decision_scope_sha256": intent.decision_scope_sha256,
        "decision_semantic_sha256": intent.decision_semantic_sha256,
        "decision_outcome": intent.decision_outcome.value,
        "incremental_targets_sha256": intent.incremental_targets_sha256,
        "target_count": intent.target_count,
    }


def adaptive_sampling_execution_terminal_receipt_projection(
    receipt: AdaptiveSamplingExecutionTerminalReceipt,
) -> dict[str, object]:
    """Return the complete versioned terminal receipt projection."""

    if not isinstance(receipt, AdaptiveSamplingExecutionTerminalReceipt):
        raise TypeError("receipt must be an AdaptiveSamplingExecutionTerminalReceipt")
    return {
        "projection_version": receipt.projection_version,
        "schema_version": receipt.schema_version,
        "execution_id": receipt.execution_id,
        "intent_semantic_sha256": receipt.intent_semantic_sha256,
        "decision_id": receipt.decision_id,
        "decision_semantic_sha256": receipt.decision_semantic_sha256,
        "terminal_kind": receipt.terminal_kind.value,
        "no_additional_work_outcome": (
            receipt.no_additional_work_outcome.value
            if receipt.no_additional_work_outcome is not None
            else None
        ),
        "work_receipt": receipt.work_receipt,
    }


def build_adaptive_sampling_execution_intent(
    decision: AdaptiveSamplingDecision,
) -> AdaptiveSamplingExecutionIntent:
    """Derive the sole stable execution identity for a frozen decision."""

    checked = _require_model(decision, AdaptiveSamplingDecision, "decision")
    targets_digest = semantic_sha256(checked.incremental_targets)
    values: dict[str, Any] = {
        "schema_version": _EXECUTION_INTENT_SCHEMA_VERSION,
        "decision_id": checked.decision_id,
        "decision_scope_sha256": checked.decision_scope_sha256,
        "decision_semantic_sha256": checked.semantic_sha256,
        "decision_outcome": checked.outcome,
        "incremental_targets_sha256": targets_digest,
        "target_count": len(checked.incremental_targets),
        "projection_version": _EXECUTION_INTENT_PROJECTION_VERSION,
    }
    draft = AdaptiveSamplingExecutionIntent.model_construct(
        execution_id="pending",
        semantic_sha256="0" * 64,
        **values,
    )
    digest = semantic_sha256(adaptive_sampling_execution_intent_projection(draft))
    return AdaptiveSamplingExecutionIntent.model_validate(
        {
            **values,
            "execution_id": f"canonical-adaptive-sampling-execution:{digest}",
            "semantic_sha256": digest,
        },
        strict=True,
    )


def build_adaptive_sampling_execution_terminal_receipt(
    intent: AdaptiveSamplingExecutionIntent,
    *,
    work_receipt: AdaptiveSamplingWorkReceipt | None = None,
) -> AdaptiveSamplingExecutionTerminalReceipt:
    """Build a terminal receipt for the one action permitted by an intent."""

    checked = _require_model(intent, AdaptiveSamplingExecutionIntent, "intent")
    if checked.decision_outcome is AdaptiveSamplingDecisionOutcome.ADDITIONAL_TARGETS_SCHEDULED:
        if not isinstance(work_receipt, AdaptiveSamplingWorkReceipt):
            raise TypeError("scheduled additional targets require an AdaptiveSamplingWorkReceipt")
        terminal_kind = AdaptiveSamplingExecutionTerminalKind.ADDITIONAL_TARGETS_COMPLETED
        no_work_outcome: AdaptiveSamplingDecisionOutcome | None = None
        checked_work = _require_model(work_receipt, AdaptiveSamplingWorkReceipt, "work_receipt")
    else:
        if work_receipt is not None:
            raise ValueError("no-additional-work decision must not receive a work receipt")
        terminal_kind = AdaptiveSamplingExecutionTerminalKind.NO_ADDITIONAL_WORK
        no_work_outcome = checked.decision_outcome
        checked_work = None

    values: dict[str, Any] = {
        "schema_version": _TERMINAL_RECEIPT_SCHEMA_VERSION,
        "execution_id": checked.execution_id,
        "intent_semantic_sha256": checked.semantic_sha256,
        "decision_id": checked.decision_id,
        "decision_semantic_sha256": checked.decision_semantic_sha256,
        "terminal_kind": terminal_kind,
        "no_additional_work_outcome": no_work_outcome,
        "work_receipt": checked_work,
        "projection_version": _TERMINAL_RECEIPT_PROJECTION_VERSION,
    }
    draft = AdaptiveSamplingExecutionTerminalReceipt.model_construct(
        receipt_id="pending",
        semantic_sha256="0" * 64,
        **values,
    )
    digest = semantic_sha256(adaptive_sampling_execution_terminal_receipt_projection(draft))
    return AdaptiveSamplingExecutionTerminalReceipt.model_validate(
        {
            **values,
            "receipt_id": f"canonical-adaptive-sampling-terminal-receipt:{digest}",
            "semantic_sha256": digest,
        },
        strict=True,
    )


class AdaptiveSamplingExecutionStore:
    """Exact-canonical file authority for immutable intents and terminal receipts.

    This intentionally remains separate from ``SQLiteAdaptiveDecisionStore``:
    the decision ledger is the causal authority, while these files form the
    execution/recovery journal for a canonical process.  Each file name is
    derived from a content-addressed identity and an existing different file
    is always a conflict, never an overwrite.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).absolute()
        _ensure_regular_directory(self._root, "execution store root")
        _ensure_regular_directory(self._intents_directory, "execution intent directory")
        _ensure_regular_directory(self._terminals_directory, "execution terminal directory")

    @property
    def root(self) -> Path:
        """Return the non-resolved root that owns execution journal files."""

        return self._root

    @property
    def _intents_directory(self) -> Path:
        return self._root / "intents"

    @property
    def _terminals_directory(self) -> Path:
        return self._root / "terminals"

    def intent_path(self, execution_id: str) -> Path:
        """Return the deterministic path for an execution ID after validating it."""

        digest = _execution_digest(execution_id)
        return self._intents_directory / f"{digest}.json"

    def terminal_path(self, execution_id: str) -> Path:
        """Return the deterministic path for an execution terminal receipt."""

        digest = _execution_digest(execution_id)
        return self._terminals_directory / f"{digest}.json"

    def put_or_get_intent(
        self,
        intent: AdaptiveSamplingExecutionIntent,
    ) -> tuple[AdaptiveSamplingExecutionIntent, bool]:
        """Publish an intent before callback execution, or replay its exact bytes."""

        checked = _require_model(intent, AdaptiveSamplingExecutionIntent, "intent")
        path = self.intent_path(checked.execution_id)
        stored, replayed = self._publish_or_read(
            path,
            canonical_json_bytes(checked),
            AdaptiveSamplingExecutionIntent,
            "execution intent",
        )
        if stored.execution_id != checked.execution_id or stored != checked:
            raise AdaptiveSamplingExecutionConflict(
                "execution intent path contains different immutable canonical bytes"
            )
        return stored, replayed

    def get_intent(
        self,
        execution_id: str,
    ) -> AdaptiveSamplingExecutionIntent | None:
        """Load and validate an intent by stable execution identity."""

        path = self.intent_path(execution_id)
        if not path.exists() and not path.is_symlink():
            return None
        loaded = self._load(path, AdaptiveSamplingExecutionIntent, "execution intent")
        if loaded.execution_id != execution_id:
            raise AdaptiveSamplingExecutionStorageError(
                "execution intent path does not match its execution identity"
            )
        return loaded

    def put_or_get_terminal(
        self,
        receipt: AdaptiveSamplingExecutionTerminalReceipt,
        *,
        intent: AdaptiveSamplingExecutionIntent,
    ) -> tuple[AdaptiveSamplingExecutionTerminalReceipt, bool]:
        """Append a terminal proof, checking that it belongs to the exact intent."""

        checked_receipt = _require_model(
            receipt,
            AdaptiveSamplingExecutionTerminalReceipt,
            "receipt",
        )
        checked_intent = _require_model(intent, AdaptiveSamplingExecutionIntent, "intent")
        _validate_receipt_binding(checked_receipt, checked_intent)
        path = self.terminal_path(checked_intent.execution_id)
        stored, replayed = self._publish_or_read(
            path,
            canonical_json_bytes(checked_receipt),
            AdaptiveSamplingExecutionTerminalReceipt,
            "execution terminal receipt",
        )
        _validate_receipt_binding(stored, checked_intent)
        if stored != checked_receipt:
            raise AdaptiveSamplingExecutionConflict(
                "execution terminal path contains different immutable canonical bytes"
            )
        return stored, replayed

    def get_terminal(
        self,
        intent: AdaptiveSamplingExecutionIntent,
    ) -> AdaptiveSamplingExecutionTerminalReceipt | None:
        """Load a terminal only when it proves the supplied frozen intent."""

        checked_intent = _require_model(intent, AdaptiveSamplingExecutionIntent, "intent")
        path = self.terminal_path(checked_intent.execution_id)
        if not path.exists() and not path.is_symlink():
            return None
        receipt = self._load(
            path,
            AdaptiveSamplingExecutionTerminalReceipt,
            "execution terminal receipt",
        )
        _validate_receipt_binding(receipt, checked_intent)
        return receipt

    def _publish_or_read[TExecutionArtifact: BaseModel](
        self,
        path: Path,
        expected: bytes,
        model_type: type[TExecutionArtifact],
        label: str,
    ) -> tuple[TExecutionArtifact, bool]:
        _ensure_regular_directory(path.parent, f"{label} directory")
        if path.exists() or path.is_symlink():
            actual = self._read_exact(path, label)
            if actual != expected:
                raise AdaptiveSamplingExecutionConflict(
                    f"existing {label} contains different immutable canonical bytes"
                )
            return _parse_exact_model(actual, model_type, label), True

        descriptor, temporary = make_temp_file(
            path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(expected)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                actual = self._read_exact(path, label)
                if actual != expected:
                    raise AdaptiveSamplingExecutionConflict(
                        f"concurrent {label} contains different immutable canonical bytes"
                    ) from None
                return _parse_exact_model(actual, model_type, label), True
            return _parse_exact_model(expected, model_type, label), False
        finally:
            temporary.unlink(missing_ok=True)

    def _load[TExecutionArtifact: BaseModel](
        self,
        path: Path,
        model_type: type[TExecutionArtifact],
        label: str,
    ) -> TExecutionArtifact:
        return _parse_exact_model(self._read_exact(path, label), model_type, label)

    @staticmethod
    def _read_exact(path: Path, label: str) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise AdaptiveSamplingExecutionStorageError(f"{label} is not a regular file")
        try:
            return path.read_bytes()
        except OSError as error:
            raise AdaptiveSamplingExecutionStorageError(f"cannot read {label}: {error}") from error


class CanonicalAdaptiveSamplingBridge:
    """Execute one frozen adaptive decision without changing canonical run semantics."""

    def __init__(
        self,
        decision_store: SQLiteAdaptiveDecisionStore,
        execution_store: AdaptiveSamplingExecutionStore,
    ) -> None:
        if not isinstance(decision_store, SQLiteAdaptiveDecisionStore):
            raise TypeError("decision_store must be a SQLiteAdaptiveDecisionStore")
        if not isinstance(execution_store, AdaptiveSamplingExecutionStore):
            raise TypeError("execution_store must be an AdaptiveSamplingExecutionStore")
        self._decision_store = decision_store
        self._execution_store = execution_store

    def execute_for_canonical_result(
        self,
        decision: AdaptiveSamplingDecision,
        *,
        result: CanonicalOfflineRunResult,
        callback: AdaptiveSamplingExecutionCallback | None = None,
    ) -> AdaptiveSamplingExecutionResult:
        """Execute only after a decision is bound to accepted canonical evidence."""

        validate_adaptive_sampling_decision_for_result(decision, result)
        return self.execute(decision, callback=callback)

    def execute(
        self,
        decision: AdaptiveSamplingDecision,
        *,
        callback: AdaptiveSamplingExecutionCallback | None = None,
    ) -> AdaptiveSamplingExecutionResult:
        """Seal, journal, and execute the one allowed extra-sampling action.

        The ordered durability contract is deliberately visible here:
        ``put_or_get(decision)`` happens before intent publication, and intent
        publication happens before the callback.  A terminal receipt turns all
        later invocations into replay reads.
        """

        stored = self._decision_store.put_or_get(
            _require_model(decision, AdaptiveSamplingDecision, "decision")
        )
        frozen = stored.decision
        intent = build_adaptive_sampling_execution_intent(frozen)
        persisted_intent, _intent_replayed = self._execution_store.put_or_get_intent(intent)
        terminal = self._execution_store.get_terminal(persisted_intent)
        if terminal is not None:
            return AdaptiveSamplingExecutionResult(
                status=AdaptiveSamplingExecutionStatus.REPLAYED,
                decision=frozen,
                intent=persisted_intent,
                terminal_receipt=terminal,
                decision_replayed=stored.replayed,
                terminal_replayed=True,
            )

        if frozen.outcome is not AdaptiveSamplingDecisionOutcome.ADDITIONAL_TARGETS_SCHEDULED:
            terminal = build_adaptive_sampling_execution_terminal_receipt(persisted_intent)
            persisted_terminal, terminal_replayed = self._execution_store.put_or_get_terminal(
                terminal,
                intent=persisted_intent,
            )
            return AdaptiveSamplingExecutionResult(
                status=AdaptiveSamplingExecutionStatus.NO_ADDITIONAL_WORK,
                decision=frozen,
                intent=persisted_intent,
                terminal_receipt=persisted_terminal,
                decision_replayed=stored.replayed,
                terminal_replayed=terminal_replayed,
            )

        if callback is None:
            raise AdaptiveSamplingExecutionError(
                "scheduled additional targets require an execution callback"
            )
        work_receipt = callback(persisted_intent, tuple(frozen.incremental_targets))
        if not isinstance(work_receipt, AdaptiveSamplingWorkReceipt):
            raise TypeError("execution callback must return an AdaptiveSamplingWorkReceipt")
        terminal = build_adaptive_sampling_execution_terminal_receipt(
            persisted_intent,
            work_receipt=work_receipt,
        )
        persisted_terminal, terminal_replayed = self._execution_store.put_or_get_terminal(
            terminal,
            intent=persisted_intent,
        )
        return AdaptiveSamplingExecutionResult(
            status=(
                AdaptiveSamplingExecutionStatus.REPLAYED
                if terminal_replayed
                else AdaptiveSamplingExecutionStatus.EXECUTED
            ),
            decision=frozen,
            intent=persisted_intent,
            terminal_receipt=persisted_terminal,
            decision_replayed=stored.replayed,
            terminal_replayed=terminal_replayed,
        )


def validate_adaptive_sampling_decision_for_result(
    decision: AdaptiveSamplingDecision,
    result: CanonicalOfflineRunResult,
) -> None:
    """Fail closed unless a decision cites one exact accepted result part.

    A durable adaptive decision remains separate from published completion
    payloads. This authority boundary ties its frozen base plan, package set,
    source lineage, effective interval, and selected inference lineage to an
    already-completed canonical result. It never infers authorization from a
    media-quality report or an event-free output.
    """

    if not isinstance(decision, AdaptiveSamplingDecision):
        raise TypeError("decision must be an AdaptiveSamplingDecision")
    if not isinstance(result, CanonicalOfflineRunResult):
        raise TypeError("result must be a CanonicalOfflineRunResult")
    if result.status not in {
        CanonicalOfflineRunStatus.SUCCEEDED,
        CanonicalOfflineRunStatus.NO_EVENTS,
    }:
        raise CanonicalAdaptiveSamplingAuthorityError(
            "adaptive sampling requires a completed SUCCEEDED or NO_EVENTS canonical result"
        )

    window = result.window
    package_set = result.package_set
    if window is None or package_set is None:
        raise CanonicalAdaptiveSamplingAuthorityError(
            "adaptive sampling requires retained root-window and package-set lineage"
        )
    lineage = package_set.lineage
    if (
        lineage.source_content_sha256 != window.source_content_sha256
        or lineage.camera_mapping_semantic_sha256 != window.camera_mapping_semantic_sha256
        or lineage.alignment_semantic_sha256 != window.alignment_semantic_sha256
    ):
        raise CanonicalAdaptiveSamplingAuthorityError(
            "canonical result package lineage does not match its retained root window"
        )

    base = decision.base
    if (
        base.sampling_plan_sha256 != lineage.sampling_plan_sha256
        or base.package_set_id != package_set.package_set_id
        or base.package_set_member_manifest_sha256 != package_set.member_manifest_sha256
        or base.package_set_split_plan_sha256 != package_set.split_plan_digest
    ):
        raise CanonicalAdaptiveSamplingAuthorityError(
            "adaptive decision base does not match the retained canonical package set"
        )

    source = decision.source
    if (
        source.source_content_sha256 != window.source_content_sha256
        or source.camera_mapping_semantic_sha256 != window.camera_mapping_semantic_sha256
        or source.alignment_semantic_sha256 != window.alignment_semantic_sha256
    ):
        raise CanonicalAdaptiveSamplingAuthorityError(
            "adaptive decision source does not match the retained canonical root window"
        )
    if decision.effective_interval != window.interval:
        raise CanonicalAdaptiveSamplingAuthorityError(
            "adaptive decision effective interval does not match the canonical root window"
        )

    accepted = decision.accepted_evidence
    matches = tuple(
        part
        for part in result.part_results
        if part.selection is not None
        and part.selected_output is not None
        and part.enriched_output is not None
        and part.selection.selection_id == accepted.selection_id
        and part.selection.selection_decision_logical_key == accepted.selection_decision_logical_key
        and part.selected_output.selection_decision_logical_key
        == part.selection.selection_decision_logical_key
        and part.selected_output.output_sha256 == accepted.selected_output_sha256
        and part.enriched_output.selected_attempt == part.selected_output
        and part.enriched_output.artifact_id == accepted.enriched_output_artifact_id
        and part.enriched_output.semantic_sha256 == accepted.enriched_output_semantic_sha256
    )
    if len(matches) != 1:
        raise CanonicalAdaptiveSamplingAuthorityError(
            "adaptive decision must cite exactly one accepted canonical inference result part"
        )


def _validate_receipt_binding(
    receipt: AdaptiveSamplingExecutionTerminalReceipt,
    intent: AdaptiveSamplingExecutionIntent,
) -> None:
    if (
        receipt.execution_id != intent.execution_id
        or receipt.intent_semantic_sha256 != intent.semantic_sha256
        or receipt.decision_id != intent.decision_id
        or receipt.decision_semantic_sha256 != intent.decision_semantic_sha256
    ):
        raise AdaptiveSamplingExecutionStorageError(
            "execution terminal receipt is not bound to the frozen execution intent"
        )
    if intent.decision_outcome is AdaptiveSamplingDecisionOutcome.ADDITIONAL_TARGETS_SCHEDULED:
        if (
            receipt.terminal_kind
            is not AdaptiveSamplingExecutionTerminalKind.ADDITIONAL_TARGETS_COMPLETED
        ):
            raise AdaptiveSamplingExecutionStorageError(
                "scheduled execution intent has an invalid no-work terminal receipt"
            )
    elif receipt.terminal_kind is not AdaptiveSamplingExecutionTerminalKind.NO_ADDITIONAL_WORK:
        raise AdaptiveSamplingExecutionStorageError(
            "no-work execution intent has an invalid completed-work terminal receipt"
        )
    elif receipt.no_additional_work_outcome is not intent.decision_outcome:
        raise AdaptiveSamplingExecutionStorageError(
            "no-work terminal receipt does not retain the frozen decision outcome"
        )


def _execution_digest(execution_id: str) -> str:
    prefix = "canonical-adaptive-sampling-execution:"
    if not isinstance(execution_id, str) or not execution_id.startswith(prefix):
        raise ValueError("execution_id must use the canonical adaptive-sampling namespace")
    digest = execution_id.removeprefix(prefix)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("execution_id must end with a lowercase SHA-256 digest")
    return digest


def _ensure_regular_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AdaptiveSamplingExecutionStorageError(f"cannot create {label}: {error}") from error
    if path.is_symlink() or not path.is_dir():
        raise AdaptiveSamplingExecutionStorageError(f"{label} must be a regular directory")


def _parse_exact_model[TExecutionArtifact: BaseModel](
    raw: bytes,
    model_type: type[TExecutionArtifact],
    label: str,
) -> TExecutionArtifact:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise AdaptiveSamplingExecutionStorageError(f"invalid {label} JSON: {error}") from error
    if not isinstance(document, dict):
        raise AdaptiveSamplingExecutionStorageError(f"{label} root must be an object")
    if canonical_json_bytes(document) != raw:
        raise AdaptiveSamplingExecutionStorageError(f"{label} bytes are not exact canonical JSON")
    try:
        # JSON enum strings are the canonical on-disk representation. Using
        # the JSON validation entry point retains strict model rules while
        # correctly reconstructing ``StrEnum`` fields.
        model = model_type.model_validate_json(raw)
    except ValueError as error:
        raise AdaptiveSamplingExecutionStorageError(f"invalid {label}: {error}") from error
    if exact_bytes_sha256(raw) != exact_bytes_sha256(canonical_json_bytes(model)):
        raise AdaptiveSamplingExecutionStorageError(f"{label} exact bytes are inconsistent")
    return model


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _require_model[TModel: BaseModel](
    value: object, model_type: type[TModel], label: str
) -> TModel:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be a {model_type.__name__}")
    try:
        # ``model_copy(update=...)`` intentionally skips Pydantic validation.
        # Rehydrating the exact canonical representation prevents such an
        # in-memory mutation from becoming a durable execution artifact.
        return model_type.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise ValueError(f"invalid {label}: {error}") from error


__all__ = [
    "AdaptiveSamplingExecutionCallback",
    "AdaptiveSamplingExecutionConflict",
    "AdaptiveSamplingExecutionError",
    "AdaptiveSamplingExecutionIntent",
    "AdaptiveSamplingExecutionResult",
    "AdaptiveSamplingExecutionStatus",
    "AdaptiveSamplingExecutionStorageError",
    "AdaptiveSamplingExecutionStore",
    "AdaptiveSamplingExecutionTerminalKind",
    "AdaptiveSamplingExecutionTerminalReceipt",
    "AdaptiveSamplingWorkReceipt",
    "CanonicalAdaptiveSamplingAuthorityError",
    "CanonicalAdaptiveSamplingBridge",
    "adaptive_sampling_execution_intent_projection",
    "adaptive_sampling_execution_terminal_receipt_projection",
    "build_adaptive_sampling_execution_intent",
    "build_adaptive_sampling_execution_terminal_receipt",
    "validate_adaptive_sampling_decision_for_result",
]
