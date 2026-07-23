"""Durable local SQLite ledger for restartable inference evidence."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime
from functools import cache
from pathlib import Path
from time import sleep
from typing import Final, TypeVar
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ValidationError

from robata.contracts.hashing import (
    CanonicalizationError,
    canonical_json_bytes,
    exact_bytes_sha256,
)
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry, SchemaRegistryError
from robata.inference.adapter import JsonSchemaRef
from robata.inference.enrichment import (
    ENRICHED_OUTPUT_SCHEMA_ID,
    ENRICHED_OUTPUT_SCHEMA_VERSION,
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    RawProviderResponseArtifact,
    SelectedAttemptOutput,
)
from robata.inference.models import (
    InferenceAttemptSelection,
    InferenceStatus,
    ModelInference,
    inference_attempt_selection_digest,
)
from robata.inference.offline_fixture import (
    RawProviderBytesNotFoundError,
    RawProviderBytesStoreError,
    StoredRawProviderBytes,
)
from robata.inference.orchestrator import InferenceIntent, InferenceLedgerError
from robata.runtime.observability import (
    RuntimeAttributeValue,
    RuntimeObserver,
    runtime_increment,
    runtime_span,
)

INFERENCE_INTENT_SCHEMA_ID: Final = "https://schemas.robata.dev/inference-intent"
MODEL_INFERENCE_SCHEMA_ID: Final = "https://schemas.robata.dev/model-inference"
INFERENCE_ATTEMPT_SELECTION_SCHEMA_ID: Final = (
    "https://schemas.robata.dev/inference-attempt-selection"
)
RAW_PROVIDER_RESPONSE_SCHEMA_ID: Final = "https://schemas.robata.dev/raw-provider-response-artifact"
PARSED_PROVIDER_CLAIM_SCHEMA_ID: Final = "https://schemas.robata.dev/parsed-provider-claim-artifact"
SELECTED_ATTEMPT_OUTPUT_SCHEMA_ID: Final = "https://schemas.robata.dev/selected-attempt-output"

_CONTRACT_VERSION: Final = "1.0.0"
_SCHEMA_VERSION: Final = 1
_APPLICATION_ID: Final = 0x5249454C  # "RIEL": Robata inference evidence ledger.
_BUSY_TIMEOUT_MS: Final = 30_000
_JOURNAL_MODE_RETRY_ATTEMPTS: Final = 100
_JOURNAL_MODE_RETRY_DELAY_SECONDS: Final = 0.01

_SCHEMA_SQL: Final = """
CREATE TABLE inference_intents (
    inference_id TEXT PRIMARY KEY,
    logical_invocation_id TEXT NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    contract_schema_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    contract_artifact_id TEXT NOT NULL,
    contract_sha256 TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    UNIQUE (inference_id, request_id)
);

CREATE TABLE raw_provider_responses (
    artifact_id TEXT PRIMARY KEY,
    inference_id TEXT NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    provider_request_id TEXT NOT NULL,
    exact_bytes_sha256 TEXT NOT NULL,
    media_type TEXT NOT NULL,
    byte_count INTEGER NOT NULL CHECK (byte_count > 0),
    raw_bytes BLOB NOT NULL,
    UNIQUE (artifact_id, inference_id),
    FOREIGN KEY (inference_id, request_id)
        REFERENCES inference_intents (inference_id, request_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE model_inference_terminals (
    inference_id TEXT PRIMARY KEY,
    logical_invocation_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('SUCCEEDED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'INVALID_OUTPUT')
    ),
    shadow INTEGER NOT NULL CHECK (shadow IN (0, 1)),
    output_valid INTEGER NOT NULL CHECK (output_valid IN (0, 1)),
    raw_artifact_id TEXT,
    contract_schema_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    contract_artifact_id TEXT NOT NULL,
    contract_sha256 TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    FOREIGN KEY (inference_id, request_id)
        REFERENCES inference_intents (inference_id, request_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (raw_artifact_id, inference_id)
        REFERENCES raw_provider_responses (artifact_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE raw_provider_artifacts (
    artifact_id TEXT PRIMARY KEY,
    inference_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    exact_bytes_sha256 TEXT NOT NULL,
    byte_count INTEGER NOT NULL CHECK (byte_count > 0),
    media_type TEXT NOT NULL,
    provider_request_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    contract_schema_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    contract_artifact_id TEXT NOT NULL,
    contract_sha256 TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    UNIQUE (artifact_id, inference_id),
    FOREIGN KEY (artifact_id, inference_id)
        REFERENCES raw_provider_responses (artifact_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (inference_id)
        REFERENCES model_inference_terminals (inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE inference_attempt_selections (
    logical_invocation_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    selection_reason TEXT NOT NULL,
    selection_id TEXT NOT NULL UNIQUE,
    inference_id TEXT NOT NULL,
    contract_schema_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    contract_artifact_id TEXT NOT NULL,
    contract_sha256 TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    PRIMARY KEY (logical_invocation_id, policy_version),
    UNIQUE (selection_id, inference_id),
    FOREIGN KEY (inference_id)
        REFERENCES model_inference_terminals (inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) WITHOUT ROWID;

CREATE TABLE parsed_provider_claims (
    artifact_id TEXT PRIMARY KEY,
    inference_id TEXT NOT NULL,
    raw_artifact_id TEXT NOT NULL,
    semantic_sha256 TEXT NOT NULL,
    provider_claim_schema_sha256 TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    contract_schema_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    contract_artifact_id TEXT NOT NULL,
    contract_sha256 TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    UNIQUE (raw_artifact_id, provider_claim_schema_sha256, parser_version),
    UNIQUE (artifact_id, inference_id, raw_artifact_id),
    FOREIGN KEY (inference_id)
        REFERENCES model_inference_terminals (inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (raw_artifact_id, inference_id)
        REFERENCES raw_provider_artifacts (artifact_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE selected_attempt_outputs (
    selection_id TEXT PRIMARY KEY,
    inference_id TEXT NOT NULL,
    parsed_artifact_id TEXT NOT NULL,
    raw_artifact_id TEXT NOT NULL,
    output_sha256 TEXT NOT NULL UNIQUE,
    contract_schema_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    contract_artifact_id TEXT NOT NULL,
    contract_sha256 TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    UNIQUE (selection_id, inference_id, output_sha256),
    FOREIGN KEY (selection_id, inference_id)
        REFERENCES inference_attempt_selections (selection_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (parsed_artifact_id, inference_id, raw_artifact_id)
        REFERENCES parsed_provider_claims (artifact_id, inference_id, raw_artifact_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE enriched_provider_outputs (
    artifact_id TEXT PRIMARY KEY,
    enrichment_logical_key TEXT NOT NULL UNIQUE,
    semantic_sha256 TEXT NOT NULL UNIQUE,
    selection_id TEXT NOT NULL,
    inference_id TEXT NOT NULL,
    selected_output_sha256 TEXT NOT NULL,
    contract_schema_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    contract_artifact_id TEXT NOT NULL,
    contract_sha256 TEXT NOT NULL,
    payload_json BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    FOREIGN KEY (selection_id, inference_id, selected_output_sha256)
        REFERENCES selected_attempt_outputs (selection_id, inference_id, output_sha256)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX inference_intents_logical_idx
    ON inference_intents (logical_invocation_id, inference_id);
CREATE INDEX raw_provider_responses_inference_idx
    ON raw_provider_responses (inference_id, artifact_id);
CREATE INDEX model_inference_terminals_logical_idx
    ON model_inference_terminals (logical_invocation_id, inference_id);
CREATE INDEX raw_provider_artifacts_inference_idx
    ON raw_provider_artifacts (inference_id, artifact_id);
CREATE INDEX parsed_provider_claims_inference_idx
    ON parsed_provider_claims (inference_id, artifact_id);
CREATE INDEX enriched_provider_outputs_selection_idx
    ON enriched_provider_outputs (selection_id, artifact_id);
"""

_APPEND_ONLY_TABLES: Final = (
    "inference_intents",
    "raw_provider_responses",
    "model_inference_terminals",
    "raw_provider_artifacts",
    "inference_attempt_selections",
    "parsed_provider_claims",
    "selected_attempt_outputs",
    "enriched_provider_outputs",
)


def _split_sql_script(script: str) -> tuple[str, ...]:
    statements: list[str] = []
    pending: list[str] = []
    for line in script.splitlines(keepends=True):
        pending.append(line)
        candidate = "".join(pending)
        if sqlite3.complete_statement(candidate):
            statement = candidate.strip()
            if statement:
                statements.append(statement)
            pending.clear()
    if any(item.strip() for item in pending):
        raise AssertionError("inference evidence SQLite schema is incomplete")
    return tuple(statements)


def _append_only_triggers(table: str) -> tuple[str, str]:
    return (
        f"""
        CREATE TRIGGER {table}_no_update
        BEFORE UPDATE ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{table} is append-only');
        END
        """,
        f"""
        CREATE TRIGGER {table}_no_delete
        BEFORE DELETE ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{table} is append-only');
        END
        """,
    )


_SCHEMA_STATEMENTS: Final = (
    *_split_sql_script(_SCHEMA_SQL),
    *(statement for table in _APPEND_ONLY_TABLES for statement in _append_only_triggers(table)),
)


class SQLiteInferenceEvidenceLedgerError(InferenceLedgerError, RawProviderBytesStoreError):
    """SQLite schema, storage, or persisted evidence failed closed."""


@dataclass(frozen=True, slots=True)
class _LedgerState:
    intents: dict[str, InferenceIntent]
    raw: dict[str, StoredRawProviderBytes]
    terminals: dict[str, ModelInference]
    raw_artifacts: dict[str, RawProviderResponseArtifact]
    selections: dict[tuple[str, str], InferenceAttemptSelection]
    parsed: dict[str, ParsedProviderClaimArtifact]
    selected: dict[str, SelectedAttemptOutput]
    enriched: dict[str, OrchestratorEnrichedOutput]


_ResultT = TypeVar("_ResultT")
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _exact_existing[EvidenceT](
    values: tuple[EvidenceT, ...],
    candidate: EvidenceT,
    *,
    conflict: str,
) -> EvidenceT | None:
    if not values:
        return None
    if any(value != candidate for value in values):
        raise SQLiteInferenceEvidenceLedgerError(conflict)
    return values[0]


class SQLiteInferenceEvidenceLedger:
    """Append-only local conformance ledger for one inference evidence graph.

    CRUD validates the addressed row and its dependency closure. Call
    `verify_integrity()` at every authoritative commit boundary to audit unrelated
    rows; construction also performs this complete audit before the ledger is used.
    """

    def __init__(
        self,
        database_path: Path,
        schema_registry: SchemaRegistry,
        *,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a pathlib.Path")
        if not isinstance(schema_registry, SchemaRegistry):
            raise TypeError("schema_registry must be a SchemaRegistry")
        self._runtime_observer = runtime_observer
        self._schema_registry = schema_registry
        self._pins = {
            "intent": self._resolve_pin(INFERENCE_INTENT_SCHEMA_ID, _CONTRACT_VERSION),
            "terminal": self._resolve_pin(MODEL_INFERENCE_SCHEMA_ID, _CONTRACT_VERSION),
            "selection": self._resolve_pin(
                INFERENCE_ATTEMPT_SELECTION_SCHEMA_ID, _CONTRACT_VERSION
            ),
            "raw_artifact": self._resolve_pin(RAW_PROVIDER_RESPONSE_SCHEMA_ID, _CONTRACT_VERSION),
            "parsed": self._resolve_pin(PARSED_PROVIDER_CLAIM_SCHEMA_ID, _CONTRACT_VERSION),
            "selected": self._resolve_pin(SELECTED_ATTEMPT_OUTPUT_SCHEMA_ID, _CONTRACT_VERSION),
            "enriched": self._resolve_pin(
                ENRICHED_OUTPUT_SCHEMA_ID, ENRICHED_OUTPUT_SCHEMA_VERSION
            ),
        }
        self._prepare_path(database_path)
        self._database_path = database_path.resolve()
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        return self._database_path

    @property
    def schema_registry(self) -> SchemaRegistry:
        return self._schema_registry

    def verify_integrity(self) -> None:
        """Audit the complete immutable evidence graph and all registered contracts."""

        def audit(connection: sqlite3.Connection) -> None:
            self._load_state(connection)

        self._transaction(
            write=False,
            operation_name="verify_integrity",
            operation=audit,
        )

    def append_intent(self, intent: InferenceIntent) -> InferenceIntent:
        checked, payload = self._prepare_model(intent, InferenceIntent, "intent")

        def append(connection: sqlite3.Connection) -> InferenceIntent:
            rows = connection.execute(
                """
                SELECT * FROM inference_intents
                WHERE inference_id = ? OR request_id = ?
                ORDER BY inference_id
                """,
                (checked.inference_id, checked.request_id),
            ).fetchall()
            existing = _exact_existing(
                tuple(self._intent_from_row(row) for row in rows),
                checked,
                conflict=f"conflicting intent: {checked.inference_id}",
            )
            if existing is not None:
                return existing
            pin = self._pins["intent"]
            connection.execute(
                """
                INSERT INTO inference_intents (
                    inference_id, logical_invocation_id, request_id,
                    contract_schema_id, contract_version, contract_artifact_id,
                    contract_sha256, payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.inference_id,
                    checked.logical_invocation_id,
                    checked.request_id,
                    pin.schema_id,
                    pin.version,
                    pin.artifact_id,
                    pin.sha256,
                    sqlite3.Binary(payload),
                    exact_bytes_sha256(payload),
                ),
            )
            stored = self._intent_by_inference_id(connection, checked.inference_id)
            if stored != checked:
                raise SQLiteInferenceEvidenceLedgerError(
                    "persisted inference intent differs after append"
                )
            return stored

        return self._transaction(
            write=True,
            operation_name="append_intent",
            operation=append,
        )

    def get_intent(self, inference_id: str) -> InferenceIntent | None:
        return self._transaction(
            write=False,
            operation_name="get_intent",
            operation=lambda connection: self._intent_by_inference_id(connection, inference_id),
        )

    def append(
        self,
        *,
        request_id: str,
        provider_request_id: str,
        data: bytes,
        media_type: str = "application/json",
    ) -> StoredRawProviderBytes:
        if not isinstance(data, bytes) or not data:
            raise ValueError("raw provider response must be nonempty bytes")
        digest = exact_bytes_sha256(data)
        artifact_id = _stable_uuid("raw-provider-response", request_id, digest)

        def append_raw(connection: sqlite3.Connection) -> StoredRawProviderBytes:
            intent = self._intent_by_request_id(connection, request_id)
            if intent is None:
                raise SQLiteInferenceEvidenceLedgerError(
                    "raw provider bytes require a persisted inference intent"
                )
            candidate = StoredRawProviderBytes(
                artifact_id=artifact_id,
                request_id=request_id,
                provider_request_id=provider_request_id,
                exact_bytes_sha256=digest,
                media_type=media_type,
                data=data,
            )
            terminal = self._terminal_by_inference_id(
                connection,
                intent.inference_id,
            )
            if terminal is not None:
                terminal_raw_artifact_id = _terminal_raw_artifact_id(terminal)
                terminal_raw = (
                    candidate if terminal_raw_artifact_id == candidate.artifact_id else None
                )
                self._validate_terminal(
                    terminal,
                    intent,
                    terminal_raw,
                    candidate,
                )
            rows = connection.execute(
                """
                SELECT * FROM raw_provider_responses
                WHERE artifact_id = ? OR request_id = ?
                ORDER BY artifact_id
                """,
                (artifact_id, request_id),
            ).fetchall()
            existing = _exact_existing(
                tuple(self._raw_from_database_row(connection, row) for row in rows),
                candidate,
                conflict="one inference request cannot append different raw response bytes",
            )
            if existing is not None:
                return existing
            connection.execute(
                """
                INSERT INTO raw_provider_responses (
                    artifact_id, inference_id, request_id, provider_request_id,
                    exact_bytes_sha256, media_type, byte_count, raw_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.artifact_id,
                    intent.inference_id,
                    candidate.request_id,
                    candidate.provider_request_id,
                    candidate.exact_bytes_sha256,
                    candidate.media_type,
                    candidate.byte_count,
                    sqlite3.Binary(candidate.data),
                ),
            )
            stored = self._raw_by_artifact_id(connection, candidate.artifact_id)
            if stored != candidate:
                raise SQLiteInferenceEvidenceLedgerError(
                    "persisted raw provider bytes differ after append"
                )
            return stored

        return self._transaction(
            write=True,
            operation_name="append_raw_provider_response",
            operation=append_raw,
        )

    def get(self, artifact_id: str) -> StoredRawProviderBytes:
        record = self._transaction(
            write=False,
            operation_name="get_raw_provider_response",
            operation=lambda connection: self._raw_by_artifact_id(connection, artifact_id),
        )
        if record is None:
            raise RawProviderBytesNotFoundError(artifact_id)
        return record

    def list_records(self) -> tuple[StoredRawProviderBytes, ...]:
        def load(connection: sqlite3.Connection) -> tuple[StoredRawProviderBytes, ...]:
            return tuple(
                self._raw_from_database_row(connection, row)
                for row in connection.execute(
                    "SELECT * FROM raw_provider_responses ORDER BY artifact_id"
                ).fetchall()
            )

        return self._transaction(
            write=False,
            operation_name="list_raw_provider_responses",
            operation=load,
        )

    def append_terminal(self, inference: ModelInference) -> ModelInference:
        checked, payload = self._prepare_model(inference, ModelInference, "terminal")

        def append(connection: sqlite3.Connection) -> ModelInference:
            existing = self._terminal_by_inference_id(connection, checked.inference_id)
            if existing is not None:
                if existing != checked:
                    raise SQLiteInferenceEvidenceLedgerError(
                        f"conflicting terminal attempt: {checked.inference_id}"
                    )
                return existing
            intent = self._intent_by_inference_id(connection, checked.inference_id)
            if intent is None:
                raise SQLiteInferenceEvidenceLedgerError(
                    "terminal attempt requires a persisted intent"
                )
            raw_artifact_id = _terminal_raw_artifact_id(checked)
            stored_raw = (
                self._raw_by_artifact_id(connection, raw_artifact_id)
                if raw_artifact_id is not None
                else None
            )
            request_raw = self._raw_by_request_id(connection, checked.request_id)
            self._validate_terminal(checked, intent, stored_raw, request_raw)
            pin = self._pins["terminal"]
            connection.execute(
                """
                INSERT INTO model_inference_terminals (
                    inference_id, logical_invocation_id, request_id, status, shadow,
                    output_valid, raw_artifact_id, contract_schema_id, contract_version,
                    contract_artifact_id, contract_sha256, payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.inference_id,
                    checked.logical_invocation_id,
                    checked.request_id,
                    checked.status.value,
                    int(checked.shadow),
                    int(checked.output_valid),
                    raw_artifact_id,
                    pin.schema_id,
                    pin.version,
                    pin.artifact_id,
                    pin.sha256,
                    sqlite3.Binary(payload),
                    exact_bytes_sha256(payload),
                ),
            )
            if stored_raw is not None:
                raw_artifact = RawProviderResponseArtifact.from_bytes(
                    data=stored_raw.data,
                    artifact_id=stored_raw.artifact_id,
                    media_type=stored_raw.media_type,
                    provider_request_id=stored_raw.provider_request_id,
                    inference_id=checked.inference_id,
                    provider=checked.provider,
                    model_name=checked.model_name,
                    model_version=checked.model_version,
                    created_at=checked.completed_at,
                )
                self._append_raw_artifact_row(
                    connection,
                    raw_artifact,
                    allow_terminal_materialization=True,
                )
            stored = self._terminal_by_inference_id(connection, checked.inference_id)
            if stored != checked:
                raise SQLiteInferenceEvidenceLedgerError(
                    "persisted terminal attempt differs after append"
                )
            return stored

        return self._transaction(
            write=True,
            operation_name="append_terminal",
            operation=append,
        )

    def get_terminal(self, inference_id: str) -> ModelInference | None:
        return self._transaction(
            write=False,
            operation_name="get_terminal",
            operation=lambda connection: self._terminal_by_inference_id(connection, inference_id),
        )

    def append_selection(self, selection: InferenceAttemptSelection) -> InferenceAttemptSelection:
        checked, payload = self._prepare_model(selection, InferenceAttemptSelection, "selection")

        def append(connection: sqlite3.Connection) -> InferenceAttemptSelection:
            rows = connection.execute(
                """
                SELECT * FROM inference_attempt_selections
                WHERE (logical_invocation_id = ? AND policy_version = ?)
                   OR selection_id = ?
                ORDER BY logical_invocation_id, policy_version
                """,
                (
                    checked.logical_invocation_id,
                    checked.policy_version,
                    checked.selection_id,
                ),
            ).fetchall()
            existing = _exact_existing(
                tuple(self._selection_from_database_row(connection, row) for row in rows),
                checked,
                conflict="logical invocation already has a different selected attempt",
            )
            if existing is not None:
                return existing
            terminal = self._terminal_by_inference_id(connection, checked.inference_id)
            self._validate_selection(checked, terminal)
            pin = self._pins["selection"]
            connection.execute(
                """
                INSERT INTO inference_attempt_selections (
                    logical_invocation_id, policy_version, selection_reason,
                    selection_id, inference_id,
                    contract_schema_id, contract_version, contract_artifact_id,
                    contract_sha256, payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.logical_invocation_id,
                    checked.policy_version,
                    checked.selection_reason,
                    checked.selection_id,
                    checked.inference_id,
                    pin.schema_id,
                    pin.version,
                    pin.artifact_id,
                    pin.sha256,
                    sqlite3.Binary(payload),
                    exact_bytes_sha256(payload),
                ),
            )
            stored = self._selection_by_logical_key(
                connection,
                checked.logical_invocation_id,
                checked.policy_version,
            )
            if stored != checked:
                raise SQLiteInferenceEvidenceLedgerError(
                    "persisted attempt selection differs after append"
                )
            return stored

        return self._transaction(
            write=True,
            operation_name="append_selection",
            operation=append,
        )

    def get_selection(
        self, logical_invocation_id: str, policy_version: str
    ) -> InferenceAttemptSelection | None:
        return self._transaction(
            write=False,
            operation_name="get_selection",
            operation=lambda connection: self._selection_by_logical_key(
                connection, logical_invocation_id, policy_version
            ),
        )

    def append_raw_artifact(
        self, artifact: RawProviderResponseArtifact
    ) -> RawProviderResponseArtifact:
        checked, payload = self._prepare_model(
            artifact, RawProviderResponseArtifact, "raw_artifact"
        )
        return self._transaction(
            write=True,
            operation_name="append_raw_artifact",
            operation=lambda connection: self._append_raw_artifact_row(
                connection, checked, payload=payload
            ),
        )

    def get_raw_artifact(self, artifact_id: str) -> RawProviderResponseArtifact | None:
        return self._transaction(
            write=False,
            operation_name="get_raw_artifact",
            operation=lambda connection: self._raw_artifact_by_id(connection, artifact_id),
        )

    def append_parsed_claim(
        self, artifact: ParsedProviderClaimArtifact
    ) -> ParsedProviderClaimArtifact:
        checked, payload = self._prepare_model(artifact, ParsedProviderClaimArtifact, "parsed")

        def append(connection: sqlite3.Connection) -> ParsedProviderClaimArtifact:
            rows = connection.execute(
                """
                SELECT * FROM parsed_provider_claims
                WHERE artifact_id = ?
                   OR (
                       raw_artifact_id = ?
                       AND provider_claim_schema_sha256 = ?
                       AND parser_version = ?
                   )
                ORDER BY artifact_id
                """,
                (
                    checked.artifact_id,
                    checked.raw_response.artifact_id,
                    checked.provider_claim_schema.sha256,
                    checked.parser_version,
                ),
            ).fetchall()
            existing = _exact_existing(
                tuple(self._parsed_from_database_row(connection, row) for row in rows),
                checked,
                conflict="parsed provider claim identity has conflicting content",
            )
            if existing is not None:
                return existing
            self._append_raw_artifact_row(connection, checked.raw_response)
            state = self._state_for_raw_artifact(connection, checked.raw_response.artifact_id)
            self._validate_parsed(checked, state)
            pin = self._pins["parsed"]
            connection.execute(
                """
                INSERT INTO parsed_provider_claims (
                    artifact_id, inference_id, raw_artifact_id, semantic_sha256,
                    provider_claim_schema_sha256, parser_version, contract_schema_id,
                    contract_version, contract_artifact_id, contract_sha256,
                    payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.artifact_id,
                    checked.raw_response.inference_id,
                    checked.raw_response.artifact_id,
                    checked.semantic_sha256,
                    checked.provider_claim_schema.sha256,
                    checked.parser_version,
                    pin.schema_id,
                    pin.version,
                    pin.artifact_id,
                    pin.sha256,
                    sqlite3.Binary(payload),
                    exact_bytes_sha256(payload),
                ),
            )
            stored = self._parsed_by_artifact_id(connection, checked.artifact_id)
            if stored != checked:
                raise SQLiteInferenceEvidenceLedgerError(
                    "persisted parsed claim differs after append"
                )
            return stored

        return self._transaction(
            write=True,
            operation_name="append_parsed_claim",
            operation=append,
        )

    def get_parsed_claim(self, artifact_id: str) -> ParsedProviderClaimArtifact | None:
        return self._transaction(
            write=False,
            operation_name="get_parsed_claim",
            operation=lambda connection: self._parsed_by_artifact_id(connection, artifact_id),
        )

    def append_selected_output(self, output: SelectedAttemptOutput) -> SelectedAttemptOutput:
        checked, payload = self._prepare_model(output, SelectedAttemptOutput, "selected")

        def append(connection: sqlite3.Connection) -> SelectedAttemptOutput:
            rows = connection.execute(
                """
                SELECT * FROM selected_attempt_outputs
                WHERE selection_id = ? OR output_sha256 = ?
                ORDER BY selection_id
                """,
                (checked.selection_id, checked.output_sha256),
            ).fetchall()
            existing = _exact_existing(
                tuple(self._selected_from_database_row(connection, row) for row in rows),
                checked,
                conflict="selected attempt output identity has conflicting content",
            )
            if existing is not None:
                return existing
            state = self._state_for_selected_output(
                connection,
                checked.selection_id,
                checked.parsed_claim_artifact_id,
            )
            self._validate_selected(checked, state)
            pin = self._pins["selected"]
            connection.execute(
                """
                INSERT INTO selected_attempt_outputs (
                    selection_id, inference_id, parsed_artifact_id, raw_artifact_id,
                    output_sha256, contract_schema_id, contract_version,
                    contract_artifact_id, contract_sha256, payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.selection_id,
                    checked.inference_id,
                    checked.parsed_claim_artifact_id,
                    checked.raw_response_artifact_id,
                    checked.output_sha256,
                    pin.schema_id,
                    pin.version,
                    pin.artifact_id,
                    pin.sha256,
                    sqlite3.Binary(payload),
                    exact_bytes_sha256(payload),
                ),
            )
            stored = self._selected_by_selection_id(connection, checked.selection_id)
            if stored != checked:
                raise SQLiteInferenceEvidenceLedgerError(
                    "persisted selected output differs after append"
                )
            return stored

        return self._transaction(
            write=True,
            operation_name="append_selected_output",
            operation=append,
        )

    def get_selected_output(self, selection_id: str) -> SelectedAttemptOutput | None:
        return self._transaction(
            write=False,
            operation_name="get_selected_output",
            operation=lambda connection: self._selected_by_selection_id(connection, selection_id),
        )

    def append_enriched_output(
        self, output: OrchestratorEnrichedOutput
    ) -> OrchestratorEnrichedOutput:
        checked, payload = self._prepare_model(output, OrchestratorEnrichedOutput, "enriched")

        def append(connection: sqlite3.Connection) -> OrchestratorEnrichedOutput:
            rows = connection.execute(
                """
                SELECT * FROM enriched_provider_outputs
                WHERE artifact_id = ?
                   OR enrichment_logical_key = ?
                   OR semantic_sha256 = ?
                ORDER BY artifact_id
                """,
                (
                    checked.artifact_id,
                    checked.enrichment_logical_key,
                    checked.semantic_sha256,
                ),
            ).fetchall()
            existing = _exact_existing(
                tuple(self._enriched_from_database_row(connection, row) for row in rows),
                checked,
                conflict="enriched output identity has conflicting content",
            )
            if existing is not None:
                return existing
            state = self._state_for_enriched_output(
                connection, checked.selected_attempt.selection_id
            )
            self._validate_enriched(checked, state)
            pin = self._pins["enriched"]
            selected = checked.selected_attempt
            connection.execute(
                """
                INSERT INTO enriched_provider_outputs (
                    artifact_id, enrichment_logical_key, semantic_sha256, selection_id,
                    inference_id, selected_output_sha256, contract_schema_id,
                    contract_version, contract_artifact_id, contract_sha256,
                    payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.artifact_id,
                    checked.enrichment_logical_key,
                    checked.semantic_sha256,
                    selected.selection_id,
                    selected.inference_id,
                    selected.output_sha256,
                    pin.schema_id,
                    pin.version,
                    pin.artifact_id,
                    pin.sha256,
                    sqlite3.Binary(payload),
                    exact_bytes_sha256(payload),
                ),
            )
            stored = self._enriched_by_artifact_id(connection, checked.artifact_id)
            if stored != checked:
                raise SQLiteInferenceEvidenceLedgerError(
                    "persisted enriched output differs after append"
                )
            return stored

        return self._transaction(
            write=True,
            operation_name="append_enriched_output",
            operation=append,
        )

    def get_enriched_output(self, artifact_id: str) -> OrchestratorEnrichedOutput | None:
        return self._transaction(
            write=False,
            operation_name="get_enriched_output",
            operation=lambda connection: self._enriched_by_artifact_id(connection, artifact_id),
        )

    def _resolve_pin(self, schema_id: str, version: str) -> SchemaRef:
        try:
            return self._schema_registry.resolve_version(schema_id, version).ref
        except SchemaRegistryError as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                f"required inference evidence schema is unavailable: {schema_id}@{version}"
            ) from exc

    def _prepare_model(
        self,
        value: _ModelT,
        model_type: type[_ModelT],
        contract: str,
    ) -> tuple[_ModelT, bytes]:
        with runtime_span(
            self._runtime_observer,
            "sqlite.inference_evidence.serialize_validate",
            {"contract": contract},
        ):
            return self._prepare_model_unobserved(value, model_type, contract)

    def _prepare_model_unobserved(
        self,
        value: _ModelT,
        model_type: type[_ModelT],
        contract: str,
    ) -> tuple[_ModelT, bytes]:
        if not isinstance(value, model_type):
            raise TypeError(f"value must be a {model_type.__name__}")
        try:
            checked = model_type.model_validate(value.model_dump(mode="python"), strict=True)
            self._validate_contract_payload(contract, checked)
            payload = canonical_json_bytes(checked)
        except SQLiteInferenceEvidenceLedgerError:
            raise
        except (CanonicalizationError, ValidationError, TypeError, ValueError) as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                f"{contract} evidence failed strict validation"
            ) from exc
        if isinstance(checked, InferenceIntent):
            _validate_intent(checked)
            self._validate_intent_output_schema(checked)
        return checked, payload

    def _validate_contract_payload(self, contract: str, value: BaseModel) -> None:
        try:
            self._schema_registry.validate_pinned(
                self._pins[contract], value.model_dump(mode="json")
            )
        except (SchemaRegistryError, ValueError) as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                f"{contract} evidence does not satisfy its pinned schema"
            ) from exc

    def _validate_intent_output_schema(self, intent: InferenceIntent) -> None:
        try:
            self._schema_registry.resolve_exact(_schema_ref(intent.request.output_schema))
        except SchemaRegistryError as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                "inference intent output schema is not an exact registered artifact"
            ) from exc

    def _prepare_path(self, original: Path) -> None:
        try:
            if original.exists() and original.is_symlink():
                raise SQLiteInferenceEvidenceLedgerError(
                    f"inference evidence database must not be a symlink: {original}"
                )
            parent = original.parent
            if parent.exists() and parent.is_symlink():
                raise SQLiteInferenceEvidenceLedgerError(
                    f"inference evidence database parent must not be a symlink: {parent}"
                )
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                f"cannot prepare inference evidence database path {original}"
            ) from exc

    def _initialize_database(self) -> None:
        with runtime_span(
            self._runtime_observer,
            "sqlite.inference_evidence.initialize",
        ):
            self._initialize_database_unobserved()

    def _initialize_database_unobserved(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            with self._observed_transaction_scope(
                connection,
                write=False,
                operation_name="initialize_preflight",
            ):
                preflight_version = _pragma_int(connection, "user_version")
                preflight_application_id = _pragma_int(connection, "application_id")
                preflight_has_schema = _has_user_schema(connection)
                if preflight_version == 0:
                    if preflight_application_id != 0 or preflight_has_schema:
                        raise SQLiteInferenceEvidenceLedgerError(
                            "refusing to adopt a nonempty or claimed unversioned SQLite database"
                        )
                elif preflight_version != _SCHEMA_VERSION:
                    raise SQLiteInferenceEvidenceLedgerError(
                        f"unsupported inference evidence schema version: {preflight_version}"
                    )
                elif preflight_application_id != _APPLICATION_ID:
                    raise SQLiteInferenceEvidenceLedgerError(
                        "inference evidence database has an unexpected application identity"
                    )
            _enable_wal_mode(connection)

            with self._observed_transaction_scope(
                connection,
                write=True,
                operation_name="initialize_schema",
            ):
                # Another constructor can initialize the same empty file while this
                # connection waits for the write lock. Re-read only after the lock.
                user_version = _pragma_int(connection, "user_version")
                application_id = _pragma_int(connection, "application_id")
                has_schema = _has_user_schema(connection)
                if user_version == 0:
                    if application_id != 0 or has_schema:
                        raise SQLiteInferenceEvidenceLedgerError(
                            "refusing to adopt a nonempty or claimed unversioned SQLite database"
                        )
                    for statement in _SCHEMA_STATEMENTS:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                elif user_version != _SCHEMA_VERSION:
                    raise SQLiteInferenceEvidenceLedgerError(
                        f"unsupported inference evidence schema version: {user_version}"
                    )
                elif application_id != _APPLICATION_ID:
                    raise SQLiteInferenceEvidenceLedgerError(
                        "inference evidence database has an unexpected application identity"
                    )
                self._verify_database(connection)
                self._load_state(connection)
        except SQLiteInferenceEvidenceLedgerError:
            if connection is not None:
                _rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                _rollback_quietly(connection)
            raise SQLiteInferenceEvidenceLedgerError(
                f"cannot initialize SQLite inference evidence ledger: {exc}"
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        with runtime_span(
            self._runtime_observer,
            "sqlite.inference_evidence.connect",
        ):
            connection = self._connect_unobserved()
        runtime_increment(
            self._runtime_observer,
            "sqlite.inference_evidence.connections",
        )
        return connection

    def _connect_unobserved(self) -> sqlite3.Connection:
        if self._database_path.is_symlink():
            raise SQLiteInferenceEvidenceLedgerError(
                f"inference evidence database became a symlink: {self._database_path}"
            )
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA recursive_triggers = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA trusted_schema = OFF")
            if _pragma_int(connection, "foreign_keys") != 1:
                raise sqlite3.OperationalError("foreign keys are disabled")
            if _pragma_int(connection, "recursive_triggers") != 1:
                raise sqlite3.OperationalError("recursive triggers are disabled")
            if _pragma_int(connection, "synchronous") != 2:
                raise sqlite3.OperationalError("FULL synchronous mode is disabled")
            if _pragma_int(connection, "trusted_schema") != 0:
                raise sqlite3.OperationalError("trusted schema mode is enabled")
            return connection
        except sqlite3.Error as exc:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise SQLiteInferenceEvidenceLedgerError(
                f"cannot open SQLite inference evidence ledger: {exc}"
            ) from exc

    def _transaction(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[sqlite3.Connection], _ResultT],
    ) -> _ResultT:
        return self._transaction_unobserved(
            write=write,
            operation_name=operation_name,
            operation=operation,
        )

    def _transaction_unobserved(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[sqlite3.Connection], _ResultT],
    ) -> _ResultT:
        connection = self._connect()
        attributes: dict[str, RuntimeAttributeValue] = {
            "operation": operation_name,
            "write": write,
        }
        try:
            with self._observed_transaction_scope(
                connection,
                write=write,
                operation_name=operation_name,
            ):
                self._verify_database(connection)
                with runtime_span(
                    self._runtime_observer,
                    "sqlite.inference_evidence.operation",
                    attributes,
                ):
                    result = operation(connection)
            return result
        except (InferenceLedgerError, RawProviderBytesStoreError):
            raise
        except sqlite3.IntegrityError as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                f"SQLite rejected append-only inference evidence: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                f"inference evidence transaction failed: {exc}"
            ) from exc
        finally:
            connection.close()

    @contextmanager
    def _observed_transaction_scope(
        self,
        connection: sqlite3.Connection,
        *,
        write: bool,
        operation_name: str,
    ) -> Iterator[None]:
        attributes: dict[str, RuntimeAttributeValue] = {
            "operation": operation_name,
            "write": write,
        }
        with runtime_span(
            self._runtime_observer,
            "sqlite.inference_evidence.transaction",
            attributes,
        ):
            with runtime_span(
                self._runtime_observer,
                "sqlite.inference_evidence.begin",
                attributes,
            ):
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            runtime_increment(
                self._runtime_observer,
                "sqlite.inference_evidence.transactions",
                attributes=attributes,
            )
            try:
                yield
            except BaseException:
                if connection.in_transaction:
                    try:
                        with runtime_span(
                            self._runtime_observer,
                            "sqlite.inference_evidence.rollback",
                            attributes,
                        ):
                            connection.rollback()
                    except sqlite3.Error:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.inference_evidence.rollback_failures",
                            attributes=attributes,
                        )
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.inference_evidence.transaction_outcomes_unknown",
                            attributes=attributes,
                        )
                    else:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.inference_evidence.rollbacks",
                            attributes=attributes,
                        )
                else:
                    runtime_increment(
                        self._runtime_observer,
                        "sqlite.inference_evidence.transaction_outcomes_unknown",
                        attributes=attributes,
                    )
                raise
            try:
                with runtime_span(
                    self._runtime_observer,
                    "sqlite.inference_evidence.commit",
                    attributes,
                ):
                    connection.commit()
            except BaseException:
                runtime_increment(
                    self._runtime_observer,
                    "sqlite.inference_evidence.commit_failures",
                    attributes=attributes,
                )
                if connection.in_transaction:
                    try:
                        with runtime_span(
                            self._runtime_observer,
                            "sqlite.inference_evidence.rollback",
                            attributes,
                        ):
                            connection.rollback()
                    except sqlite3.Error:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.inference_evidence.rollback_failures",
                            attributes=attributes,
                        )
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.inference_evidence.transaction_outcomes_unknown",
                            attributes=attributes,
                        )
                    else:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.inference_evidence.rollbacks",
                            attributes=attributes,
                        )
                else:
                    runtime_increment(
                        self._runtime_observer,
                        "sqlite.inference_evidence.transaction_outcomes_unknown",
                        attributes=attributes,
                    )
                raise
            runtime_increment(
                self._runtime_observer,
                "sqlite.inference_evidence.commits",
                attributes=attributes,
            )

    def _verify_database(self, connection: sqlite3.Connection) -> None:
        runtime_increment(
            self._runtime_observer,
            "sqlite.inference_evidence.integrity_checks",
        )
        with runtime_span(
            self._runtime_observer,
            "sqlite.inference_evidence.integrity_check",
        ):
            self._verify_database_unobserved(connection)

    def _verify_database_unobserved(self, connection: sqlite3.Connection) -> None:
        if _pragma_int(connection, "application_id") != _APPLICATION_ID:
            raise SQLiteInferenceEvidenceLedgerError(
                "inference evidence database application identity changed"
            )
        if _pragma_int(connection, "user_version") != _SCHEMA_VERSION:
            raise SQLiteInferenceEvidenceLedgerError(
                "inference evidence database schema version changed"
            )
        journal = connection.execute("PRAGMA journal_mode").fetchone()
        if journal is None or not isinstance(journal[0], str) or journal[0].lower() != "wal":
            raise SQLiteInferenceEvidenceLedgerError(
                "inference evidence database is not in WAL mode"
            )
        if _database_schema_fingerprint(connection) != _expected_schema_fingerprint():
            raise SQLiteInferenceEvidenceLedgerError(
                "SQLite inference evidence DDL does not match the canonical schema"
            )
        quick = connection.execute("PRAGMA quick_check(1)").fetchone()
        if quick is None or quick[0] != "ok":
            raise SQLiteInferenceEvidenceLedgerError(
                f"SQLite quick_check failed for inference evidence: {quick!r}"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise SQLiteInferenceEvidenceLedgerError(
                "SQLite foreign-key check found orphaned inference evidence"
            )

    def _load_state(self, connection: sqlite3.Connection) -> _LedgerState:
        intents = {
            item.inference_id: item
            for item in (
                self._intent_from_row(row)
                for row in connection.execute(
                    "SELECT * FROM inference_intents ORDER BY inference_id"
                ).fetchall()
            )
        }
        raw: dict[str, StoredRawProviderBytes] = {}
        raw_inference: dict[str, str] = {}
        for row in connection.execute(
            "SELECT * FROM raw_provider_responses ORDER BY artifact_id"
        ).fetchall():
            item, inference_id = self._raw_from_row(row, intents)
            raw[item.artifact_id] = item
            raw_inference[item.artifact_id] = inference_id
        terminals = {
            item.inference_id: item
            for item in (
                self._terminal_from_row(row, intents, raw, raw_inference)
                for row in connection.execute(
                    "SELECT * FROM model_inference_terminals ORDER BY inference_id"
                ).fetchall()
            )
        }
        selections = {
            (item.logical_invocation_id, item.policy_version): item
            for item in (
                self._selection_from_row(row, terminals)
                for row in connection.execute(
                    """
                    SELECT * FROM inference_attempt_selections
                    ORDER BY logical_invocation_id, policy_version
                    """
                ).fetchall()
            )
        }
        state = _LedgerState(
            intents=intents,
            raw=raw,
            terminals=terminals,
            raw_artifacts={},
            selections=selections,
            parsed={},
            selected={},
            enriched={},
        )
        raw_artifacts = {
            item.artifact_id: item
            for item in (
                self._raw_artifact_from_row(row, state)
                for row in connection.execute(
                    "SELECT * FROM raw_provider_artifacts ORDER BY artifact_id"
                ).fetchall()
            )
        }
        for terminal in terminals.values():
            raw_artifact_id = _terminal_raw_artifact_id(terminal)
            if raw_artifact_id is not None and raw_artifact_id not in raw_artifacts:
                raise SQLiteInferenceEvidenceLedgerError(
                    "terminal raw bytes are missing their typed raw provider artifact"
                )
        state = replace(state, raw_artifacts=raw_artifacts)
        parsed = {
            item.artifact_id: item
            for item in (
                self._parsed_from_row(row, state)
                for row in connection.execute(
                    "SELECT * FROM parsed_provider_claims ORDER BY artifact_id"
                ).fetchall()
            )
        }
        state = replace(state, parsed=parsed)
        selected = {
            item.selection_id: item
            for item in (
                self._selected_from_row(row, state)
                for row in connection.execute(
                    "SELECT * FROM selected_attempt_outputs ORDER BY selection_id"
                ).fetchall()
            )
        }
        state = replace(state, selected=selected)
        enriched = {
            item.artifact_id: item
            for item in (
                self._enriched_from_row(row, state)
                for row in connection.execute(
                    "SELECT * FROM enriched_provider_outputs ORDER BY artifact_id"
                ).fetchall()
            )
        }
        return replace(state, enriched=enriched)

    def _intent_by_inference_id(
        self,
        connection: sqlite3.Connection,
        inference_id: str,
    ) -> InferenceIntent | None:
        row = connection.execute(
            "SELECT * FROM inference_intents WHERE inference_id = ?",
            (inference_id,),
        ).fetchone()
        return None if row is None else self._intent_from_row(row)

    def _intent_by_request_id(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> InferenceIntent | None:
        row = connection.execute(
            "SELECT * FROM inference_intents WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return None if row is None else self._intent_from_row(row)

    def _raw_from_database_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> StoredRawProviderBytes:
        inference_id = _row_text(row, "inference_id")
        intent = self._intent_by_inference_id(connection, inference_id)
        intents = {} if intent is None else {intent.inference_id: intent}
        record, _inference_id = self._raw_from_row(row, intents)
        return record

    def _raw_by_artifact_id(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
    ) -> StoredRawProviderBytes | None:
        row = connection.execute(
            "SELECT * FROM raw_provider_responses WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return None if row is None else self._raw_from_database_row(connection, row)

    def _raw_by_request_id(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> StoredRawProviderBytes | None:
        row = connection.execute(
            "SELECT * FROM raw_provider_responses WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return None if row is None else self._raw_from_database_row(connection, row)

    def _terminal_from_database_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        require_typed_artifact: bool,
    ) -> ModelInference:
        inference_id = _row_text(row, "inference_id")
        request_id = _row_text(row, "request_id")
        raw_artifact_value: object = row["raw_artifact_id"]
        if raw_artifact_value is not None and not isinstance(raw_artifact_value, str):
            raise SQLiteInferenceEvidenceLedgerError(
                "SQLite column 'raw_artifact_id' is neither text nor null"
            )
        raw_artifact_id = raw_artifact_value
        intent = self._intent_by_inference_id(connection, inference_id)
        intents = {} if intent is None else {intent.inference_id: intent}
        stored_raw = (
            self._raw_by_artifact_id(connection, raw_artifact_id)
            if raw_artifact_id is not None
            else None
        )
        request_raw = self._raw_by_request_id(connection, request_id)
        raw_values = tuple(
            {
                item.artifact_id: item for item in (stored_raw, request_raw) if item is not None
            }.values()
        )
        raw = {item.artifact_id: item for item in raw_values}
        raw_inference = {item.artifact_id: inference_id for item in raw_values}
        terminal = self._terminal_from_row(
            row,
            intents,
            raw,
            raw_inference,
        )
        if require_typed_artifact and raw_artifact_id is not None:
            typed = self._raw_artifact_by_id(
                connection,
                raw_artifact_id,
                terminal=terminal,
            )
            if typed is None:
                raise SQLiteInferenceEvidenceLedgerError(
                    "terminal raw bytes are missing their typed raw provider artifact"
                )
        return terminal

    def _terminal_by_inference_id(
        self,
        connection: sqlite3.Connection,
        inference_id: str,
        *,
        require_typed_artifact: bool = True,
    ) -> ModelInference | None:
        row = connection.execute(
            "SELECT * FROM model_inference_terminals WHERE inference_id = ?",
            (inference_id,),
        ).fetchone()
        if row is None:
            return None
        return self._terminal_from_database_row(
            connection,
            row,
            require_typed_artifact=require_typed_artifact,
        )

    def _selection_from_database_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> InferenceAttemptSelection:
        inference_id = _row_text(row, "inference_id")
        terminal = self._terminal_by_inference_id(connection, inference_id)
        terminals = {} if terminal is None else {terminal.inference_id: terminal}
        return self._selection_from_row(row, terminals)

    def _selection_by_logical_key(
        self,
        connection: sqlite3.Connection,
        logical_invocation_id: str,
        policy_version: str,
    ) -> InferenceAttemptSelection | None:
        row = connection.execute(
            """
            SELECT * FROM inference_attempt_selections
            WHERE logical_invocation_id = ? AND policy_version = ?
            """,
            (logical_invocation_id, policy_version),
        ).fetchone()
        return None if row is None else self._selection_from_database_row(connection, row)

    def _selection_by_id(
        self,
        connection: sqlite3.Connection,
        selection_id: str,
    ) -> InferenceAttemptSelection | None:
        row = connection.execute(
            "SELECT * FROM inference_attempt_selections WHERE selection_id = ?",
            (selection_id,),
        ).fetchone()
        return None if row is None else self._selection_from_database_row(connection, row)

    def _raw_artifact_dependencies_state(
        self,
        connection: sqlite3.Connection,
        *,
        artifact_id: str,
        inference_id: str,
    ) -> _LedgerState:
        intent = self._intent_by_inference_id(connection, inference_id)
        raw = self._raw_by_artifact_id(connection, artifact_id)
        terminal = self._terminal_by_inference_id(
            connection,
            inference_id,
            require_typed_artifact=False,
        )
        return _LedgerState(
            intents={} if intent is None else {intent.inference_id: intent},
            raw={} if raw is None else {raw.artifact_id: raw},
            terminals={} if terminal is None else {terminal.inference_id: terminal},
            raw_artifacts={},
            selections={},
            parsed={},
            selected={},
            enriched={},
        )

    def _raw_artifact_from_database_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        terminal: ModelInference | None = None,
    ) -> RawProviderResponseArtifact:
        artifact_id = _row_text(row, "artifact_id")
        inference_id = _row_text(row, "inference_id")
        state = self._raw_artifact_dependencies_state(
            connection,
            artifact_id=artifact_id,
            inference_id=inference_id,
        )
        if terminal is not None:
            state = replace(
                state,
                terminals={terminal.inference_id: terminal},
            )
        return self._raw_artifact_from_row(row, state)

    def _raw_artifact_by_id(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
        *,
        terminal: ModelInference | None = None,
    ) -> RawProviderResponseArtifact | None:
        row = connection.execute(
            "SELECT * FROM raw_provider_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return self._raw_artifact_from_database_row(
            connection,
            row,
            terminal=terminal,
        )

    def _append_raw_artifact_row(
        self,
        connection: sqlite3.Connection,
        artifact: RawProviderResponseArtifact,
        *,
        payload: bytes | None = None,
        allow_terminal_materialization: bool = False,
    ) -> RawProviderResponseArtifact:
        state = self._raw_artifact_dependencies_state(
            connection,
            artifact_id=artifact.artifact_id,
            inference_id=artifact.inference_id,
        )
        self._validate_raw_artifact(artifact, state)
        existing = self._raw_artifact_by_id(connection, artifact.artifact_id)
        if existing is not None:
            if existing != artifact:
                raise SQLiteInferenceEvidenceLedgerError(
                    "raw provider artifact identity has conflicting metadata"
                )
            return existing
        if state.terminals and not allow_terminal_materialization:
            raise SQLiteInferenceEvidenceLedgerError(
                "terminal rows are missing their typed raw provider artifact"
            )
        canonical_payload = canonical_json_bytes(artifact) if payload is None else payload
        pin = self._pins["raw_artifact"]
        intent = state.intents[artifact.inference_id]
        connection.execute(
            """
            INSERT INTO raw_provider_artifacts (
                artifact_id, inference_id, request_id, exact_bytes_sha256,
                byte_count, media_type, provider_request_id, provider, model_name,
                model_version, created_at,
                contract_schema_id, contract_version, contract_artifact_id,
                contract_sha256, payload_json, payload_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.artifact_id,
                artifact.inference_id,
                intent.request_id,
                artifact.exact_bytes_sha256,
                artifact.byte_count,
                artifact.media_type,
                artifact.provider_request_id,
                artifact.provider,
                artifact.model_name,
                artifact.model_version,
                artifact.created_at,
                pin.schema_id,
                pin.version,
                pin.artifact_id,
                pin.sha256,
                sqlite3.Binary(canonical_payload),
                exact_bytes_sha256(canonical_payload),
            ),
        )
        terminal = state.terminals.get(artifact.inference_id)
        stored = self._raw_artifact_by_id(
            connection,
            artifact.artifact_id,
            terminal=terminal,
        )
        if stored != artifact:
            raise SQLiteInferenceEvidenceLedgerError(
                "persisted raw provider artifact differs after append"
            )
        return stored

    def _state_for_raw_artifact(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
    ) -> _LedgerState:
        artifact = self._raw_artifact_by_id(connection, artifact_id)
        if artifact is None:
            return _LedgerState({}, {}, {}, {}, {}, {}, {}, {})
        state = self._raw_artifact_dependencies_state(
            connection,
            artifact_id=artifact.artifact_id,
            inference_id=artifact.inference_id,
        )
        return replace(
            state,
            raw_artifacts={artifact.artifact_id: artifact},
        )

    def _parsed_from_database_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> ParsedProviderClaimArtifact:
        state = self._state_for_raw_artifact(
            connection,
            _row_text(row, "raw_artifact_id"),
        )
        return self._parsed_from_row(row, state)

    def _parsed_by_artifact_id(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
    ) -> ParsedProviderClaimArtifact | None:
        row = connection.execute(
            "SELECT * FROM parsed_provider_claims WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return None if row is None else self._parsed_from_database_row(connection, row)

    def _state_for_selected_output(
        self,
        connection: sqlite3.Connection,
        selection_id: str,
        parsed_artifact_id: str,
    ) -> _LedgerState:
        selection = self._selection_by_id(connection, selection_id)
        parsed = self._parsed_by_artifact_id(connection, parsed_artifact_id)
        selections = (
            {}
            if selection is None
            else {(selection.logical_invocation_id, selection.policy_version): selection}
        )
        return _LedgerState(
            intents={},
            raw={},
            terminals={},
            raw_artifacts={},
            selections=selections,
            parsed={} if parsed is None else {parsed.artifact_id: parsed},
            selected={},
            enriched={},
        )

    def _selected_from_database_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> SelectedAttemptOutput:
        state = self._state_for_selected_output(
            connection,
            _row_text(row, "selection_id"),
            _row_text(row, "parsed_artifact_id"),
        )
        return self._selected_from_row(row, state)

    def _selected_by_selection_id(
        self,
        connection: sqlite3.Connection,
        selection_id: str,
    ) -> SelectedAttemptOutput | None:
        row = connection.execute(
            "SELECT * FROM selected_attempt_outputs WHERE selection_id = ?",
            (selection_id,),
        ).fetchone()
        return None if row is None else self._selected_from_database_row(connection, row)

    def _state_for_enriched_output(
        self,
        connection: sqlite3.Connection,
        selection_id: str,
    ) -> _LedgerState:
        selected = self._selected_by_selection_id(connection, selection_id)
        if selected is None:
            return _LedgerState({}, {}, {}, {}, {}, {}, {}, {})
        parsed = self._parsed_by_artifact_id(connection, selected.parsed_claim_artifact_id)
        intent = self._intent_by_inference_id(connection, selected.inference_id)
        selection = self._selection_by_id(connection, selected.selection_id)
        selections = (
            {}
            if selection is None
            else {(selection.logical_invocation_id, selection.policy_version): selection}
        )
        return _LedgerState(
            intents={} if intent is None else {intent.inference_id: intent},
            raw={},
            terminals={},
            raw_artifacts={},
            selections=selections,
            parsed={} if parsed is None else {parsed.artifact_id: parsed},
            selected={selected.selection_id: selected},
            enriched={},
        )

    def _enriched_from_database_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> OrchestratorEnrichedOutput:
        state = self._state_for_enriched_output(
            connection,
            _row_text(row, "selection_id"),
        )
        return self._enriched_from_row(row, state)

    def _enriched_by_artifact_id(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
    ) -> OrchestratorEnrichedOutput | None:
        row = connection.execute(
            "SELECT * FROM enriched_provider_outputs WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return None if row is None else self._enriched_from_database_row(connection, row)

    def _intent_from_row(self, row: sqlite3.Row) -> InferenceIntent:
        intent = self._model_from_row(row, InferenceIntent, "intent")
        _require_columns(
            row,
            (
                ("inference_id", intent.inference_id),
                ("logical_invocation_id", intent.logical_invocation_id),
                ("request_id", intent.request_id),
            ),
            "intent",
        )
        _validate_intent(intent)
        self._validate_intent_output_schema(intent)
        return intent

    def _raw_from_row(
        self, row: sqlite3.Row, intents: dict[str, InferenceIntent]
    ) -> tuple[StoredRawProviderBytes, str]:
        inference_id = _row_text(row, "inference_id")
        try:
            record = StoredRawProviderBytes(
                artifact_id=_row_text(row, "artifact_id"),
                request_id=_row_text(row, "request_id"),
                provider_request_id=_row_text(row, "provider_request_id"),
                exact_bytes_sha256=_row_text(row, "exact_bytes_sha256"),
                media_type=_row_text(row, "media_type"),
                data=_row_bytes(row, "raw_bytes"),
            )
        except ValueError as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                "persisted raw provider response failed strict validation"
            ) from exc
        if _row_int(row, "byte_count") != record.byte_count:
            raise SQLiteInferenceEvidenceLedgerError(
                "raw provider response byte count does not match exact bytes"
            )
        intent = intents.get(inference_id)
        if intent is None or intent.request_id != record.request_id:
            raise SQLiteInferenceEvidenceLedgerError(
                "raw provider response does not match its indexed intent"
            )
        expected_artifact_id = _stable_uuid(
            "raw-provider-response", record.request_id, record.exact_bytes_sha256
        )
        if record.artifact_id != expected_artifact_id:
            raise SQLiteInferenceEvidenceLedgerError(
                "raw provider response artifact identity is inconsistent"
            )
        return record, inference_id

    def _terminal_from_row(
        self,
        row: sqlite3.Row,
        intents: dict[str, InferenceIntent],
        raw: dict[str, StoredRawProviderBytes],
        raw_inference: dict[str, str],
    ) -> ModelInference:
        terminal = self._model_from_row(row, ModelInference, "terminal")
        raw_artifact_id = _terminal_raw_artifact_id(terminal)
        _require_columns(
            row,
            (
                ("inference_id", terminal.inference_id),
                ("logical_invocation_id", terminal.logical_invocation_id),
                ("request_id", terminal.request_id),
                ("status", terminal.status.value),
                ("raw_artifact_id", raw_artifact_id),
            ),
            "terminal",
        )
        if (
            _row_bool(row, "shadow") != terminal.shadow
            or _row_bool(row, "output_valid") != terminal.output_valid
        ):
            raise SQLiteInferenceEvidenceLedgerError(
                "terminal payload does not match indexed boolean columns"
            )
        intent = intents.get(terminal.inference_id)
        if intent is None:
            raise SQLiteInferenceEvidenceLedgerError("terminal has no persisted intent")
        stored_raw = raw.get(raw_artifact_id) if raw_artifact_id is not None else None
        request_raw = next(
            (item for item in raw.values() if item.request_id == terminal.request_id),
            None,
        )
        if (
            raw_artifact_id is not None
            and raw_inference.get(raw_artifact_id) != terminal.inference_id
        ):
            raise SQLiteInferenceEvidenceLedgerError(
                "terminal raw artifact belongs to a different inference"
            )
        self._validate_terminal(terminal, intent, stored_raw, request_raw)
        return terminal

    def _selection_from_row(
        self, row: sqlite3.Row, terminals: dict[str, ModelInference]
    ) -> InferenceAttemptSelection:
        selection = self._model_from_row(row, InferenceAttemptSelection, "selection")
        _require_columns(
            row,
            (
                ("logical_invocation_id", selection.logical_invocation_id),
                ("policy_version", selection.policy_version),
                ("selection_reason", selection.selection_reason),
                ("selection_id", selection.selection_id),
                ("inference_id", selection.inference_id),
            ),
            "selection",
        )
        self._validate_selection(selection, terminals.get(selection.inference_id))
        return selection

    def _raw_artifact_from_row(
        self, row: sqlite3.Row, state: _LedgerState
    ) -> RawProviderResponseArtifact:
        artifact = self._model_from_row(row, RawProviderResponseArtifact, "raw_artifact")
        intent = state.intents.get(artifact.inference_id)
        _require_columns(
            row,
            (
                ("artifact_id", artifact.artifact_id),
                ("inference_id", artifact.inference_id),
                ("request_id", intent.request_id if intent is not None else None),
                ("exact_bytes_sha256", artifact.exact_bytes_sha256),
                ("byte_count", artifact.byte_count),
                ("media_type", artifact.media_type),
                ("provider_request_id", artifact.provider_request_id),
                ("provider", artifact.provider),
                ("model_name", artifact.model_name),
                ("model_version", artifact.model_version),
                ("created_at", artifact.created_at),
            ),
            "raw provider artifact",
        )
        self._validate_raw_artifact(artifact, state)
        return artifact

    def _parsed_from_row(
        self, row: sqlite3.Row, state: _LedgerState
    ) -> ParsedProviderClaimArtifact:
        parsed = self._model_from_row(row, ParsedProviderClaimArtifact, "parsed")
        raw = parsed.raw_response
        _require_columns(
            row,
            (
                ("artifact_id", parsed.artifact_id),
                ("inference_id", raw.inference_id),
                ("raw_artifact_id", raw.artifact_id),
                ("semantic_sha256", parsed.semantic_sha256),
                ("provider_claim_schema_sha256", parsed.provider_claim_schema.sha256),
                ("parser_version", parsed.parser_version),
            ),
            "parsed claim",
        )
        self._validate_parsed(parsed, state)
        return parsed

    def _selected_from_row(self, row: sqlite3.Row, state: _LedgerState) -> SelectedAttemptOutput:
        selected = self._model_from_row(row, SelectedAttemptOutput, "selected")
        _require_columns(
            row,
            (
                ("selection_id", selected.selection_id),
                ("inference_id", selected.inference_id),
                ("parsed_artifact_id", selected.parsed_claim_artifact_id),
                ("raw_artifact_id", selected.raw_response_artifact_id),
                ("output_sha256", selected.output_sha256),
            ),
            "selected output",
        )
        self._validate_selected(selected, state)
        return selected

    def _enriched_from_row(
        self, row: sqlite3.Row, state: _LedgerState
    ) -> OrchestratorEnrichedOutput:
        enriched = self._model_from_row(row, OrchestratorEnrichedOutput, "enriched")
        selected = enriched.selected_attempt
        _require_columns(
            row,
            (
                ("artifact_id", enriched.artifact_id),
                ("enrichment_logical_key", enriched.enrichment_logical_key),
                ("semantic_sha256", enriched.semantic_sha256),
                ("selection_id", selected.selection_id),
                ("inference_id", selected.inference_id),
                ("selected_output_sha256", selected.output_sha256),
            ),
            "enriched output",
        )
        self._validate_enriched(enriched, state)
        return enriched

    def _model_from_row(
        self, row: sqlite3.Row, model_type: type[_ModelT], contract: str
    ) -> _ModelT:
        pin = self._pins[contract]
        _require_columns(
            row,
            (
                ("contract_schema_id", pin.schema_id),
                ("contract_version", pin.version),
                ("contract_artifact_id", pin.artifact_id),
                ("contract_sha256", pin.sha256),
            ),
            f"{contract} schema pin",
        )
        raw = _row_bytes(row, "payload_json")
        if _row_text(row, "payload_sha256") != exact_bytes_sha256(raw):
            raise SQLiteInferenceEvidenceLedgerError(
                f"persisted {contract} payload digest is inconsistent"
            )
        try:
            value = model_type.model_validate_json(raw, strict=True)
            if canonical_json_bytes(value) != raw:
                raise SQLiteInferenceEvidenceLedgerError(
                    f"persisted {contract} is not canonical JSON"
                )
            self._validate_contract_payload(contract, value)
            return value
        except SQLiteInferenceEvidenceLedgerError:
            raise
        except (CanonicalizationError, ValidationError, TypeError, ValueError) as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                f"persisted {contract} failed strict validation"
            ) from exc

    def _validate_terminal(
        self,
        terminal: ModelInference,
        intent: InferenceIntent,
        raw: StoredRawProviderBytes | None,
        request_raw: StoredRawProviderBytes | None,
    ) -> None:
        _validate_terminal_binding(terminal, intent)
        raw_artifact_id = _terminal_raw_artifact_id(terminal)
        if request_raw is not None and raw_artifact_id != request_raw.artifact_id:
            raise SQLiteInferenceEvidenceLedgerError(
                "terminal must retain already-persisted raw provider bytes"
            )
        if raw_artifact_id is None:
            if terminal.status is InferenceStatus.SUCCEEDED:
                raise SQLiteInferenceEvidenceLedgerError(
                    "successful terminal requires exact raw provider bytes"
                )
        else:
            if raw is None:
                raise SQLiteInferenceEvidenceLedgerError(
                    "terminal references absent raw provider bytes"
                )
            if (
                raw.artifact_id != raw_artifact_id
                or raw.request_id != terminal.request_id
                or raw.provider_request_id != terminal.provider_request_id
            ):
                raise SQLiteInferenceEvidenceLedgerError(
                    "terminal does not match exact raw provider bytes"
                )
        if terminal.status is InferenceStatus.SUCCEEDED:
            if (
                not terminal.output_valid
                or terminal.normalized_output is None
                or terminal.failure is not None
            ):
                raise SQLiteInferenceEvidenceLedgerError(
                    "successful terminal has inconsistent selection semantics"
                )
            try:
                ref = _schema_ref(intent.request.output_schema)
                self._schema_registry.validate_pinned(ref, terminal.normalized_output)
            except (SchemaRegistryError, ValueError) as exc:
                raise SQLiteInferenceEvidenceLedgerError(
                    "successful terminal normalized output is not schema-valid"
                ) from exc
        elif terminal.output_valid or terminal.failure is None:
            raise SQLiteInferenceEvidenceLedgerError(
                "failed terminal must be invalid and retain failure evidence"
            )

    @staticmethod
    def _validate_selection(
        selection: InferenceAttemptSelection, terminal: ModelInference | None
    ) -> None:
        if terminal is None or terminal.status is not InferenceStatus.SUCCEEDED:
            raise SQLiteInferenceEvidenceLedgerError(
                "selection requires a successful terminal attempt"
            )
        expected_selection_id = _stable_uuid(
            "inference-selection",
            inference_attempt_selection_digest(
                logical_invocation_id=selection.logical_invocation_id,
                policy_version=selection.policy_version,
            ),
        )
        if (
            terminal.shadow
            or not terminal.output_valid
            or terminal.failure is not None
            or terminal.logical_invocation_id != selection.logical_invocation_id
            or selection.selection_id != expected_selection_id
            or _rfc3339(selection.selected_at) < _rfc3339(terminal.completed_at)
        ):
            raise SQLiteInferenceEvidenceLedgerError(
                "selected terminal attempt is semantically inconsistent"
            )

    def _validate_raw_artifact(
        self, artifact: RawProviderResponseArtifact, state: _LedgerState
    ) -> None:
        self._validate_contract_payload("raw_artifact", artifact)
        stored = state.raw.get(artifact.artifact_id)
        terminal = state.terminals.get(artifact.inference_id)
        intent = state.intents.get(artifact.inference_id)
        if stored is None or terminal is None or intent is None:
            raise SQLiteInferenceEvidenceLedgerError(
                "raw provider artifact requires its intent, terminal, and exact bytes"
            )
        expected_artifact_id = _stable_uuid(
            "raw-provider-response", intent.request_id, artifact.exact_bytes_sha256
        )
        if (
            artifact.artifact_id != expected_artifact_id
            or _terminal_raw_artifact_id(terminal) != artifact.artifact_id
            or stored.artifact_id != artifact.artifact_id
            or stored.request_id != intent.request_id
            or stored.exact_bytes_sha256 != artifact.exact_bytes_sha256
            or stored.byte_count != artifact.byte_count
            or stored.media_type != artifact.media_type
            or stored.provider_request_id != artifact.provider_request_id
            or artifact.provider_request_id != terminal.provider_request_id
            or artifact.provider != terminal.provider
            or artifact.model_name != terminal.model_name
            or artifact.model_version != terminal.model_version
            or _rfc3339(artifact.created_at) < _rfc3339(terminal.completed_at)
        ):
            raise SQLiteInferenceEvidenceLedgerError(
                "raw provider artifact lineage is inconsistent"
            )

    def _validate_parsed(self, parsed: ParsedProviderClaimArtifact, state: _LedgerState) -> None:
        raw_artifact = parsed.raw_response
        persisted_raw = state.raw_artifacts.get(raw_artifact.artifact_id)
        stored = state.raw.get(raw_artifact.artifact_id)
        terminal = state.terminals.get(raw_artifact.inference_id)
        intent = state.intents.get(raw_artifact.inference_id)
        if (
            persisted_raw is None
            or persisted_raw != raw_artifact
            or stored is None
            or terminal is None
            or intent is None
        ):
            raise SQLiteInferenceEvidenceLedgerError(
                "parsed claim requires its intent, terminal, and exact raw artifact"
            )
        if (
            parsed.task is not terminal.stage
            or parsed.provider_claim_schema != intent.request.output_schema
            or parsed.payload.model_dump(mode="json") != terminal.normalized_output
            or _rfc3339(parsed.created_at) < _rfc3339(raw_artifact.created_at)
        ):
            raise SQLiteInferenceEvidenceLedgerError(
                "parsed provider claim lineage is inconsistent"
            )

    @staticmethod
    def _validate_selected(output: SelectedAttemptOutput, state: _LedgerState) -> None:
        selection = next(
            (
                item
                for item in state.selections.values()
                if item.selection_id == output.selection_id
            ),
            None,
        )
        parsed = state.parsed.get(output.parsed_claim_artifact_id)
        if selection is None or parsed is None:
            raise SQLiteInferenceEvidenceLedgerError(
                "selected output requires its selection and parsed claim"
            )
        try:
            expected = SelectedAttemptOutput.create(parsed, selection)
        except ValueError as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                "selected output lineage cannot be reconstructed"
            ) from exc
        if output != expected:
            raise SQLiteInferenceEvidenceLedgerError(
                "selected output does not exactly bind selection, raw bytes, and parsed claim"
            )

    def _validate_enriched(self, output: OrchestratorEnrichedOutput, state: _LedgerState) -> None:
        selected = state.selected.get(output.selected_attempt.selection_id)
        if selected is None or selected != output.selected_attempt:
            raise SQLiteInferenceEvidenceLedgerError(
                "enriched output does not reference the exact selected attempt output"
            )
        parsed = state.parsed.get(selected.parsed_claim_artifact_id)
        intent = state.intents.get(selected.inference_id)
        selection = next(
            (
                item
                for item in state.selections.values()
                if item.selection_id == selected.selection_id
            ),
            None,
        )
        if (
            parsed is None
            or intent is None
            or selection is None
            or output.provider_claim_schema != parsed.provider_claim_schema
            or output.provider_claim_schema != intent.request.output_schema
            or _schema_ref(output.enriched_output_schema) != self._pins["enriched"]
            or _rfc3339(output.created_at) < _rfc3339(parsed.created_at)
            or _rfc3339(output.created_at) < _rfc3339(selection.selected_at)
        ):
            raise SQLiteInferenceEvidenceLedgerError(
                "enriched output schema lineage is inconsistent"
            )
        try:
            self._schema_registry.resolve_exact(_schema_ref(output.provider_claim_schema))
            self._schema_registry.resolve_exact(_schema_ref(output.enriched_output_schema))
        except SchemaRegistryError as exc:
            raise SQLiteInferenceEvidenceLedgerError(
                "enriched output references an unregistered schema artifact"
            ) from exc


def _validate_intent(intent: InferenceIntent) -> None:
    request = intent.request
    expected = (
        intent.logical_invocation_id,
        intent.request_id,
        intent.idempotency_key,
        intent.task,
        intent.provider,
        intent.model_name,
        intent.model_version,
        intent.input_plan_id,
        intent.input_plan_semantic_sha256,
        intent.input_plan_part_ordinal,
        intent.input_plan_part_count,
        intent.input_plan_part_semantic_sha256,
    )
    actual = (
        request.logical_invocation_id,
        request.request_id,
        request.idempotency_key,
        request.task,
        request.provider,
        request.model_name,
        request.model_version,
        request.input_plan_id,
        request.input_plan_semantic_sha256,
        request.input_plan_part_ordinal,
        request.input_plan_part_count,
        request.input_plan_part_semantic_sha256,
    )
    if expected != actual or intent.retry_count != intent.attempt - 1:
        raise SQLiteInferenceEvidenceLedgerError(
            "inference intent does not match its immutable request"
        )
    if _rfc3339(intent.created_at) > _rfc3339(intent.queued_at):
        raise SQLiteInferenceEvidenceLedgerError(
            "inference intent timestamps are not monotonically ordered"
        )
    shadow_lineage = (
        intent.experiment_id,
        intent.shadow_route_id,
        intent.primary_inference_id,
    )
    if not intent.shadow and any(value is not None for value in shadow_lineage):
        raise SQLiteInferenceEvidenceLedgerError("non-shadow intent cannot retain shadow lineage")
    if intent.shadow and intent.shadow_route_id is None:
        raise SQLiteInferenceEvidenceLedgerError("shadow intent requires a shadow route identity")


def _validate_terminal_binding(terminal: ModelInference, intent: InferenceIntent) -> None:
    request = intent.request
    package_id = request.package_inputs[0].package_id if len(request.package_inputs) == 1 else None
    expected = (
        intent.inference_id,
        intent.logical_invocation_id,
        intent.request_id,
        intent.idempotency_key,
        intent.mcap_id,
        request.package_set_id,
        package_id,
        tuple(item.package_id for item in request.package_inputs),
        intent.camera_mapping_run_id,
        intent.alignment_id,
        intent.start_ns,
        intent.end_ns,
        intent.task,
        intent.provider,
        intent.model_name,
        intent.model_version,
        intent.adapter_version,
        request.prompt_version,
        request.prompt_artifact_id,
        request.prompt_sha256,
        request.rendered_input_digest,
        intent.input_plan_id,
        intent.input_plan_semantic_sha256,
        intent.input_plan_part_ordinal,
        intent.input_plan_part_count,
        intent.input_plan_part_semantic_sha256,
        request.output_schema.schema_id,
        request.output_schema.version,
        request.output_schema.artifact_id,
        request.output_schema.sha256,
        request.capability_snapshot_id,
        request.capability_snapshot_digest,
        request.package_input_set_sha256,
        intent.input_config,
        intent.sampling_config,
        request.generation_config,
        request.provider_idempotency_key,
        intent.experiment_id,
        intent.shadow_route_id,
        intent.primary_inference_id,
        intent.shadow,
        intent.attempt,
        intent.retry_count,
        intent.queued_at,
        intent.created_at,
    )
    actual = (
        terminal.inference_id,
        terminal.logical_invocation_id,
        terminal.request_id,
        terminal.idempotency_key,
        terminal.mcap_id,
        terminal.package_set_id,
        terminal.package_id,
        terminal.package_ids,
        terminal.camera_mapping_run_id,
        terminal.alignment_id,
        terminal.start_ns,
        terminal.end_ns,
        terminal.stage,
        terminal.provider,
        terminal.model_name,
        terminal.model_version,
        terminal.adapter_version,
        terminal.prompt_version,
        terminal.prompt_artifact_id,
        terminal.prompt_sha256,
        terminal.rendered_input_digest,
        terminal.input_plan_id,
        terminal.input_plan_semantic_sha256,
        terminal.input_plan_part_ordinal,
        terminal.input_plan_part_count,
        terminal.input_plan_part_semantic_sha256,
        terminal.output_schema_id,
        terminal.output_schema_version,
        terminal.output_schema_artifact_id,
        terminal.output_schema_sha256,
        terminal.capability_snapshot_id,
        terminal.capability_snapshot_digest,
        terminal.input_manifest_set_sha256,
        terminal.input_config,
        terminal.sampling_config,
        terminal.generation_config,
        terminal.provider_idempotency_key,
        terminal.experiment_id,
        terminal.shadow_route_id,
        terminal.primary_inference_id,
        terminal.shadow,
        terminal.attempt,
        terminal.retry_count,
        terminal.queued_at,
        terminal.created_at,
    )
    if expected != actual:
        raise SQLiteInferenceEvidenceLedgerError(
            "terminal attempt does not match its persisted intent"
        )
    if not (
        _rfc3339(intent.created_at)
        <= _rfc3339(intent.queued_at)
        <= _rfc3339(terminal.started_at)
        <= _rfc3339(terminal.completed_at)
    ):
        raise SQLiteInferenceEvidenceLedgerError(
            "terminal timestamps are not monotonically ordered"
        )


def _terminal_raw_artifact_id(terminal: ModelInference) -> str | None:
    if terminal.raw_output is None:
        return None
    if set(terminal.raw_output) != {"artifact_id"}:
        raise SQLiteInferenceEvidenceLedgerError(
            "terminal raw output must contain exactly one artifact_id"
        )
    artifact_id = terminal.raw_output["artifact_id"]
    if not isinstance(artifact_id, str) or not artifact_id:
        raise SQLiteInferenceEvidenceLedgerError(
            "terminal raw output artifact_id must be nonempty text"
        )
    return artifact_id


def _schema_ref(reference: JsonSchemaRef) -> SchemaRef:
    return SchemaRef(
        schema_id=reference.schema_id,
        version=reference.version,
        artifact_id=reference.artifact_id,
        sha256=reference.sha256,
    )


def _stable_uuid(namespace: str, *parts: object) -> str:
    material = ":".join(str(item) for item in parts)
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{material}"))


def _rfc3339(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SQLiteInferenceEvidenceLedgerError("persisted timestamp is not RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SQLiteInferenceEvidenceLedgerError("persisted timestamp is not timezone-aware")
    return parsed


def _require_columns(
    row: sqlite3.Row,
    expected: tuple[tuple[str, object], ...],
    description: str,
) -> None:
    for column, value in expected:
        stored: object = row[column]
        if stored != value:
            raise SQLiteInferenceEvidenceLedgerError(
                f"persisted {description} does not match indexed column {column}"
            )


def _row_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise SQLiteInferenceEvidenceLedgerError(f"SQLite column {column!r} is not text")
    return value


def _row_int(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SQLiteInferenceEvidenceLedgerError(f"SQLite column {column!r} is not integer")
    return value


def _row_bool(row: sqlite3.Row, column: str) -> bool:
    value = _row_int(row, column)
    if value not in (0, 1):
        raise SQLiteInferenceEvidenceLedgerError(f"SQLite column {column!r} is not boolean")
    return bool(value)


def _row_bytes(row: sqlite3.Row, column: str) -> bytes:
    value: object = row[column]
    if not isinstance(value, bytes):
        raise SQLiteInferenceEvidenceLedgerError(f"SQLite column {column!r} is not a blob")
    return value


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    value: object = None if row is None else row[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SQLiteInferenceEvidenceLedgerError(f"SQLite PRAGMA {name} did not return an integer")
    return value


def _enable_wal_mode(connection: sqlite3.Connection) -> None:
    for attempt in range(_JOURNAL_MODE_RETRY_ATTEMPTS):
        try:
            journal = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        except sqlite3.OperationalError as exc:
            error_code = getattr(exc, "sqlite_errorcode", None)
            if error_code is None or error_code & 0xFF not in {
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            }:
                raise
            if attempt == _JOURNAL_MODE_RETRY_ATTEMPTS - 1:
                raise
        else:
            if journal is not None and isinstance(journal[0], str) and journal[0].lower() == "wal":
                return
            if attempt == _JOURNAL_MODE_RETRY_ATTEMPTS - 1:
                break
        sleep(_JOURNAL_MODE_RETRY_DELAY_SECONDS)
    raise SQLiteInferenceEvidenceLedgerError("SQLite WAL mode could not be enabled")


def _has_user_schema(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT EXISTS (SELECT 1 FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%')"
    ).fetchone()
    if row is None or row[0] not in (0, 1):
        raise SQLiteInferenceEvidenceLedgerError("SQLite schema inventory is invalid")
    return bool(row[0])


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _database_schema_fingerprint(connection: sqlite3.Connection) -> tuple[object, ...]:
    object_rows = connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    objects = tuple(tuple(row) for row in object_rows)
    table_names = tuple(
        _row_text(row, "name") for row in object_rows if _row_text(row, "type") == "table"
    )
    table_facts: list[tuple[object, ...]] = []
    for table in table_names:
        quoted = _quote_identifier(table)
        columns = tuple(
            tuple(row) for row in connection.execute(f"PRAGMA table_xinfo({quoted})").fetchall()
        )
        index_rows = connection.execute(f"PRAGMA index_list({quoted})").fetchall()
        indexes = tuple(
            (
                *tuple(index),
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        f"PRAGMA index_xinfo({_quote_identifier(_row_text(index, 'name'))})"
                    ).fetchall()
                ),
            )
            for index in index_rows
        )
        foreign_keys = tuple(
            tuple(row)
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
        )
        table_facts.append((table, columns, indexes, foreign_keys))
    return objects, tuple(table_facts)


@cache
def _expected_schema_fingerprint() -> tuple[object, ...]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        return _database_schema_fingerprint(connection)
    finally:
        connection.close()


def _rollback_quietly(connection: sqlite3.Connection) -> None:
    with suppress(sqlite3.Error):
        connection.rollback()


__all__ = [
    "INFERENCE_ATTEMPT_SELECTION_SCHEMA_ID",
    "INFERENCE_INTENT_SCHEMA_ID",
    "MODEL_INFERENCE_SCHEMA_ID",
    "PARSED_PROVIDER_CLAIM_SCHEMA_ID",
    "RAW_PROVIDER_RESPONSE_SCHEMA_ID",
    "SELECTED_ATTEMPT_OUTPUT_SCHEMA_ID",
    "SQLiteInferenceEvidenceLedger",
    "SQLiteInferenceEvidenceLedgerError",
]
