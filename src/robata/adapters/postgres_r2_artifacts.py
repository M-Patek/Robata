"""PostgreSQL-backed immutable R2 mirror lifecycle for raw provider evidence.

PostgreSQL retains the exact raw response bytes as the canonical recovery
authority. This adapter makes a tenant-scoped R2 mirror mandatory for every new
raw response without performing provider I/O inside an authority transaction.
A durable STAGED receipt preserves the bytes and deterministic locator before
R2 is contacted; a COMMITTED receipt is required before the existing raw
provider row may be inserted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeVar, cast
from uuid import uuid4

from robata.adapters.postgres_authority import (
    PostgresAuthorityError,
    PostgresCanonicalAuthority,
    PostgresConnection,
)
from robata.adapters.r2_object_store import R2ObjectStore
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.object_storage import ObjectLocator, ObjectPutReceipt, ObjectPutRequest
from robata.inference.offline_fixture import RawProviderBytesStoreError, StoredRawProviderBytes
from robata.ports.object_storage import ObjectStoreError, ObjectStoreErrorCode

_RECEIPT_TABLE: Final = "raw_provider_r2_artifact_receipts"
_OBSERVATION_TABLE: Final = "raw_provider_r2_artifact_observations"
_T = TypeVar("_T")


class RawProviderR2ArtifactState(StrEnum):
    """Durable lifecycle state for one required raw-evidence R2 mirror."""

    STAGED = "STAGED"
    COMMITTED = "COMMITTED"


class RawProviderR2ObservationKind(StrEnum):
    """Append-only observations produced while mirroring or verifying an object."""

    PUT_VERIFIED = "PUT_VERIFIED"
    MISSING = "MISSING"
    PARTIAL = "PARTIAL"
    CONFLICT = "CONFLICT"
    CORRUPT = "CORRUPT"
    PROVIDER_ERROR = "PROVIDER_ERROR"


class PostgresR2ArtifactError(PostgresAuthorityError, RawProviderBytesStoreError):
    """A required immutable R2 raw-evidence mirror could not be proved exact."""


@dataclass(frozen=True, slots=True)
class RawProviderR2ArtifactReceipt:
    """One tenant-bound immutable mirror plan plus its monotonic completion state."""

    artifact_id: str
    inference_id: str
    request_id: str
    provider_request_id: str
    exact_bytes_sha256: str
    byte_count: int
    media_type: str
    payload: bytes
    logical_key: str
    object_uri: str
    object_version: str
    object_etag: str | None
    r2_config_sha256: str
    state: RawProviderR2ArtifactState

    @property
    def locator(self) -> ObjectLocator:
        """Return transport metadata without treating it as content identity."""

        if self.object_etag is None:
            return ObjectLocator(uri=self.object_uri, object_version=self.object_version)
        return ObjectLocator(
            uri=self.object_uri,
            object_version=self.object_version,
            etag=self.object_etag,
        )

    def object_put_request(self) -> ObjectPutRequest:
        """Build the exact immutable R2 write for this persisted receipt."""

        return ObjectPutRequest(
            key=self.logical_key,
            payload=self.payload,
            sha256=self.exact_bytes_sha256,
            byte_count=self.byte_count,
            media_type=self.media_type,
            object_version=self.object_version,
        )


class PostgresR2ArtifactAuthority:
    """Coordinate durable R2 raw-evidence mirroring outside PostgreSQL transactions."""

    def __init__(
        self,
        authority: PostgresCanonicalAuthority,
        r2_object_store: R2ObjectStore,
        *,
        tenant_id: str,
    ) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        if not isinstance(r2_object_store, R2ObjectStore):
            raise TypeError("r2_object_store must be R2ObjectStore")
        if (
            not isinstance(tenant_id, str)
            or not tenant_id.strip()
            or any(character.isspace() for character in tenant_id)
        ):
            raise ValueError("tenant_id must be nonempty and contain no whitespace")
        if r2_object_store.config.allow_delete:
            raise ValueError("production R2 artifact authority requires allow_delete=False")
        self._authority = authority
        self._r2_object_store = r2_object_store
        self._tenant_id = tenant_id
        self._r2_config_sha256 = exact_bytes_sha256(
            canonical_json_bytes(r2_object_store.config.model_dump(mode="json"))
        )
        self._tenant_scope = exact_bytes_sha256(tenant_id.encode("utf-8"))

    @property
    def r2_config_sha256(self) -> str:
        """Return the non-secret R2 configuration digest bound into every receipt."""

        return self._r2_config_sha256

    def verify_startup(self) -> None:
        """Require the receipt and observation relations to be RLS-protected."""

        def operation(connection: PostgresConnection) -> None:
            rows = connection.execute(
                """
                SELECT c.relname AS table_name,
                       c.relrowsecurity AS row_security,
                       c.relforcerowsecurity AS force_row_security
                FROM pg_catalog.pg_class AS c
                JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                WHERE n.nspname = %s
                  AND c.relkind = 'r'
                  AND c.relname = ANY(%s)
                """,
                (self._authority.schema, [_RECEIPT_TABLE, _OBSERVATION_TABLE]),
            ).fetchall()
            by_table = {str(row["table_name"]): row for row in rows}
            missing = sorted({_RECEIPT_TABLE, _OBSERVATION_TABLE}.difference(by_table))
            if missing:
                raise PostgresR2ArtifactError(
                    "R2 artifact migrations are incomplete: " + ", ".join(missing)
                )
            insecure = sorted(
                table
                for table, row in by_table.items()
                if row.get("row_security") is not True or row.get("force_row_security") is not True
            )
            if insecure:
                raise PostgresR2ArtifactError(
                    "R2 artifact tables do not enforce FORCE RLS: " + ", ".join(insecure)
                )

        self._run_transaction(
            write=False,
            operation_name="r2_artifacts.verify_startup",
            operation=operation,
        )

    def stage_and_verify(self, record: StoredRawProviderBytes) -> RawProviderR2ArtifactReceipt:
        """Durably stage, exactly write, and bind one raw response mirror.

        The first and last database operations are short serializable authority
        transactions. The sole R2 PUT happens between them, after the full
        canonical bytes and deterministic key have been committed as STAGED.
        """

        self._require_record(record)
        receipt = self._stage(record)
        self._require_current_r2_configuration(receipt)
        if receipt.state is RawProviderR2ArtifactState.COMMITTED:
            self.verify_record(record)
            return receipt
        try:
            provider_receipt = self._r2_object_store.put(receipt.object_put_request())
        except ObjectStoreError as error:
            self._note_provider_failure(receipt, error)
            raise PostgresR2ArtifactError(
                "R2 could not persist the staged raw provider response"
            ) from error
        return self._commit(receipt, provider_receipt)

    def verify_record(self, record: StoredRawProviderBytes) -> RawProviderR2ArtifactReceipt:
        """Prove an existing raw PostgreSQL record has the same exact R2 bytes."""

        self._require_record(record)
        receipt = self._load_receipt(record.artifact_id)
        if receipt is None:
            raise PostgresR2ArtifactError(
                f"raw provider response has no required R2 receipt: {record.artifact_id}"
            )
        self._require_record_matches_receipt(record, receipt)
        self._require_current_r2_configuration(receipt)
        if receipt.state is not RawProviderR2ArtifactState.COMMITTED:
            raise PostgresR2ArtifactError(
                f"raw provider response R2 receipt is not committed: {record.artifact_id}"
            )
        try:
            head = self._r2_object_store.head(receipt.locator)
            if (
                head.verified is not True
                or head.sha256 != receipt.exact_bytes_sha256
                or head.byte_count != receipt.byte_count
                or head.media_type != receipt.media_type
            ):
                raise ObjectStoreError(
                    ObjectStoreErrorCode.INTEGRITY_ERROR,
                    "R2 object metadata differs from the durable raw-evidence receipt",
                )
            payload = self._r2_object_store.get(receipt.locator)
            if payload != record.data:
                raise ObjectStoreError(
                    ObjectStoreErrorCode.INTEGRITY_ERROR,
                    "R2 object bytes differ from PostgreSQL canonical raw evidence",
                )
        except ObjectStoreError as error:
            self._note_provider_failure(receipt, error)
            raise PostgresR2ArtifactError(
                f"R2 raw provider response cannot be verified: {record.artifact_id}"
            ) from error
        return receipt

    def verify_records(self, records: Iterable[StoredRawProviderBytes]) -> None:
        """Verify every supplied canonical raw record against its immutable mirror."""

        for record in records:
            self.verify_record(record)

    def reconcile_staged(
        self,
        *,
        limit: int = 100,
    ) -> tuple[RawProviderR2ArtifactReceipt, ...]:
        """Resume durable STAGED R2 writes without changing canonical evidence rows.

        A process crash after staging preserves exact bytes in PostgreSQL. This
        method reuses each deterministic immutable key/version and returns the
        receipts it successfully advanced. It deliberately does not synthesize
        a raw-provider evidence row, because typed ledger validation remains the
        owner of that append.
        """

        checked_limit = _positive_limit(limit)
        committed: list[RawProviderR2ArtifactReceipt] = []
        failures: list[str] = []
        for receipt in self._list_receipts(
            state=RawProviderR2ArtifactState.STAGED,
            limit=checked_limit,
        ):
            self._require_current_r2_configuration(receipt)
            try:
                provider_receipt = self._r2_object_store.put(receipt.object_put_request())
            except ObjectStoreError as error:
                self._note_provider_failure(receipt, error)
                failures.append(receipt.artifact_id)
                continue
            finalized = self._commit(receipt, provider_receipt)
            committed.append(finalized)
        if failures:
            raise PostgresR2ArtifactError(
                "R2 staged raw-provider receipts remain unresolved: " + ", ".join(failures)
            )
        return tuple(committed)

    def backfill_unmirrored_raw_provider_responses(
        self,
        *,
        limit: int = 100,
    ) -> tuple[RawProviderR2ArtifactReceipt, ...]:
        """Create exact mirrors for pre-0005 raw evidence without rewriting it."""

        records = self._load_unmirrored_raw_provider_records(limit=_positive_limit(limit))
        return tuple(self.stage_and_verify(record) for record in records)

    def _stage(self, record: StoredRawProviderBytes) -> RawProviderR2ArtifactReceipt:
        logical_key = f"raw-provider-responses/{self._tenant_scope}/{record.artifact_id}"
        object_version = f"raw-v1-{record.exact_bytes_sha256}"
        locator = self._r2_object_store.locator_for(logical_key, object_version)

        def operation(connection: PostgresConnection) -> RawProviderR2ArtifactReceipt:
            intent = connection.execute(
                """
                SELECT inference_id
                FROM inference_intents
                WHERE tenant_id = %s AND request_id = %s
                """,
                (self._tenant_id, record.request_id),
            ).fetchone()
            if intent is None or not isinstance(intent.get("inference_id"), str):
                raise PostgresR2ArtifactError(
                    "raw provider response requires a persisted inference intent before R2 staging"
                )
            inference_id = cast(str, intent["inference_id"])
            rows = connection.execute(
                f"""
                SELECT artifact_id, inference_id, request_id, provider_request_id,
                       exact_bytes_sha256, byte_count, media_type, payload_bytes,
                       logical_key, object_uri, object_version, object_etag,
                       r2_config_sha256, state
                FROM {_RECEIPT_TABLE}
                WHERE tenant_id = %s AND (artifact_id = %s OR request_id = %s)
                FOR UPDATE
                """,
                (self._tenant_id, record.artifact_id, record.request_id),
            ).fetchall()
            if len(rows) > 1:
                raise PostgresR2ArtifactError(
                    "R2 artifact receipt identity is inconsistent for the raw provider response"
                )
            if rows:
                existing = self._receipt_from_row(rows[0])
                self._require_stage_matches(
                    existing,
                    record=record,
                    inference_id=inference_id,
                    logical_key=logical_key,
                    locator=locator,
                )
                return existing
            inserted = connection.execute(
                f"""
                INSERT INTO {_RECEIPT_TABLE} (
                    tenant_id, artifact_id, inference_id, request_id, provider_request_id,
                    exact_bytes_sha256, byte_count, media_type, payload_bytes,
                    logical_key, object_uri, object_version, r2_config_sha256, state
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'STAGED')
                ON CONFLICT DO NOTHING
                RETURNING artifact_id
                """,
                (
                    self._tenant_id,
                    record.artifact_id,
                    inference_id,
                    record.request_id,
                    record.provider_request_id,
                    record.exact_bytes_sha256,
                    record.byte_count,
                    record.media_type,
                    record.data,
                    logical_key,
                    locator.uri,
                    locator.object_version,
                    self._r2_config_sha256,
                ),
            ).fetchone()
            planned = RawProviderR2ArtifactReceipt(
                artifact_id=record.artifact_id,
                inference_id=inference_id,
                request_id=record.request_id,
                provider_request_id=record.provider_request_id,
                exact_bytes_sha256=record.exact_bytes_sha256,
                byte_count=record.byte_count,
                media_type=record.media_type,
                payload=record.data,
                logical_key=logical_key,
                object_uri=locator.uri,
                object_version=locator.object_version,
                object_etag=None,
                r2_config_sha256=self._r2_config_sha256,
                state=RawProviderR2ArtifactState.STAGED,
            )
            if inserted is not None:
                return planned

            conflicted_rows = connection.execute(
                f"""
                SELECT artifact_id, inference_id, request_id, provider_request_id,
                       exact_bytes_sha256, byte_count, media_type, payload_bytes,
                       logical_key, object_uri, object_version, object_etag,
                       r2_config_sha256, state
                FROM {_RECEIPT_TABLE}
                WHERE tenant_id = %s AND (artifact_id = %s OR request_id = %s)
                FOR UPDATE
                """,
                (self._tenant_id, record.artifact_id, record.request_id),
            ).fetchall()
            if len(conflicted_rows) != 1:
                raise PostgresR2ArtifactError(
                    "concurrent R2 artifact receipt creation did not resolve to one receipt"
                )
            existing = self._receipt_from_row(conflicted_rows[0])
            self._require_stage_matches(
                existing,
                record=record,
                inference_id=inference_id,
                logical_key=logical_key,
                locator=locator,
            )
            return existing

        return self._run_transaction(
            write=True,
            operation_name="r2_artifacts.stage_raw_provider_response",
            operation=operation,
        )

    def _commit(
        self,
        staged: RawProviderR2ArtifactReceipt,
        provider_receipt: ObjectPutReceipt,
    ) -> RawProviderR2ArtifactReceipt:
        self._require_current_r2_configuration(staged)
        if (
            provider_receipt.locator.uri != staged.object_uri
            or provider_receipt.locator.object_version != staged.object_version
            or provider_receipt.sha256 != staged.exact_bytes_sha256
            or provider_receipt.byte_count != staged.byte_count
        ):
            raise PostgresR2ArtifactError(
                "R2 provider receipt does not match the durable raw-evidence plan"
            )

        def operation(connection: PostgresConnection) -> RawProviderR2ArtifactReceipt:
            row = connection.execute(
                f"""
                SELECT artifact_id, inference_id, request_id, provider_request_id,
                       exact_bytes_sha256, byte_count, media_type, payload_bytes,
                       logical_key, object_uri, object_version, object_etag,
                       r2_config_sha256, state
                FROM {_RECEIPT_TABLE}
                WHERE tenant_id = %s AND artifact_id = %s
                FOR UPDATE
                """,
                (self._tenant_id, staged.artifact_id),
            ).fetchone()
            if row is None:
                raise PostgresR2ArtifactError(
                    "staged R2 receipt disappeared before it could commit"
                )
            current = self._receipt_from_row(row)
            self._require_receipts_match(staged, current)
            if current.state is RawProviderR2ArtifactState.COMMITTED:
                return current
            updated = connection.execute(
                f"""
                UPDATE {_RECEIPT_TABLE}
                SET state = 'COMMITTED',
                    object_etag = %s,
                    committed_at = CURRENT_TIMESTAMP
                WHERE tenant_id = %s AND artifact_id = %s AND state = 'STAGED'
                """,
                (provider_receipt.locator.etag, self._tenant_id, staged.artifact_id),
            )
            if updated.rowcount != 1:
                raise PostgresR2ArtifactError("staged R2 receipt lost its monotonic commit race")
            committed = RawProviderR2ArtifactReceipt(
                artifact_id=current.artifact_id,
                inference_id=current.inference_id,
                request_id=current.request_id,
                provider_request_id=current.provider_request_id,
                exact_bytes_sha256=current.exact_bytes_sha256,
                byte_count=current.byte_count,
                media_type=current.media_type,
                payload=current.payload,
                logical_key=current.logical_key,
                object_uri=current.object_uri,
                object_version=current.object_version,
                object_etag=provider_receipt.locator.etag,
                r2_config_sha256=current.r2_config_sha256,
                state=RawProviderR2ArtifactState.COMMITTED,
            )
            self._insert_observation(
                connection,
                receipt=committed,
                kind=RawProviderR2ObservationKind.PUT_VERIFIED,
                observation_id=str(uuid4()),
            )
            return committed

        return self._run_transaction(
            write=True,
            operation_name="r2_artifacts.commit_raw_provider_response",
            operation=operation,
        )

    def _load_receipt(self, artifact_id: str) -> RawProviderR2ArtifactReceipt | None:
        def operation(connection: PostgresConnection) -> RawProviderR2ArtifactReceipt | None:
            row = connection.execute(
                f"""
                SELECT artifact_id, inference_id, request_id, provider_request_id,
                       exact_bytes_sha256, byte_count, media_type, payload_bytes,
                       logical_key, object_uri, object_version, object_etag,
                       r2_config_sha256, state
                FROM {_RECEIPT_TABLE}
                WHERE tenant_id = %s AND artifact_id = %s
                """,
                (self._tenant_id, artifact_id),
            ).fetchone()
            return None if row is None else self._receipt_from_row(row)

        return self._run_transaction(
            write=False,
            operation_name="r2_artifacts.load_raw_provider_receipt",
            operation=operation,
        )

    def _list_receipts(
        self,
        *,
        state: RawProviderR2ArtifactState,
        limit: int,
    ) -> tuple[RawProviderR2ArtifactReceipt, ...]:
        def operation(connection: PostgresConnection) -> tuple[RawProviderR2ArtifactReceipt, ...]:
            rows = connection.execute(
                f"""
                SELECT artifact_id, inference_id, request_id, provider_request_id,
                       exact_bytes_sha256, byte_count, media_type, payload_bytes,
                       logical_key, object_uri, object_version, object_etag,
                       r2_config_sha256, state
                FROM {_RECEIPT_TABLE}
                WHERE tenant_id = %s AND state = %s
                ORDER BY artifact_id
                LIMIT %s
                """,
                (self._tenant_id, state.value, limit),
            ).fetchall()
            return tuple(self._receipt_from_row(row) for row in rows)

        return self._run_transaction(
            write=False,
            operation_name="r2_artifacts.list_staged_raw_provider_receipts",
            operation=operation,
        )

    def _load_unmirrored_raw_provider_records(
        self,
        *,
        limit: int,
    ) -> tuple[StoredRawProviderBytes, ...]:
        def operation(connection: PostgresConnection) -> tuple[StoredRawProviderBytes, ...]:
            rows = connection.execute(
                f"""
                SELECT raw.artifact_id, raw.request_id, raw.provider_request_id,
                       raw.exact_bytes_sha256, raw.media_type, raw.raw_bytes
                FROM raw_provider_responses AS raw
                LEFT JOIN {_RECEIPT_TABLE} AS receipt
                  ON receipt.tenant_id = raw.tenant_id
                 AND receipt.artifact_id = raw.artifact_id
                WHERE raw.tenant_id = %s AND receipt.artifact_id IS NULL
                ORDER BY raw.artifact_id
                LIMIT %s
                """,
                (self._tenant_id, limit),
            ).fetchall()
            try:
                return tuple(
                    StoredRawProviderBytes(
                        artifact_id=_row_text(row, "artifact_id"),
                        request_id=_row_text(row, "request_id"),
                        provider_request_id=_row_text(row, "provider_request_id"),
                        exact_bytes_sha256=_row_text(row, "exact_bytes_sha256"),
                        media_type=_row_text(row, "media_type"),
                        data=_row_bytes(row, "raw_bytes"),
                    )
                    for row in rows
                )
            except (TypeError, ValueError) as error:
                raise PostgresR2ArtifactError(
                    "existing raw provider evidence cannot be mirrored safely"
                ) from error

        return self._run_transaction(
            write=False,
            operation_name="r2_artifacts.load_unmirrored_raw_provider_records",
            operation=operation,
        )

    def _append_observation(
        self,
        receipt: RawProviderR2ArtifactReceipt,
        kind: RawProviderR2ObservationKind,
    ) -> None:
        observation_id = str(uuid4())

        def operation(connection: PostgresConnection) -> None:
            self._insert_observation(
                connection,
                receipt=receipt,
                kind=kind,
                observation_id=observation_id,
            )

        self._run_transaction(
            write=True,
            operation_name="r2_artifacts.append_observation",
            operation=operation,
        )

    def _insert_observation(
        self,
        connection: PostgresConnection,
        *,
        receipt: RawProviderR2ArtifactReceipt,
        kind: RawProviderR2ObservationKind,
        observation_id: str,
    ) -> None:
        connection.execute(
            f"""
            INSERT INTO {_OBSERVATION_TABLE} (
                tenant_id, observation_id, artifact_id, observation_kind,
                exact_bytes_sha256, byte_count, media_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                self._tenant_id,
                observation_id,
                receipt.artifact_id,
                kind.value,
                receipt.exact_bytes_sha256,
                receipt.byte_count,
                receipt.media_type,
            ),
        )

    def _note_provider_failure(
        self,
        receipt: RawProviderR2ArtifactReceipt,
        error: ObjectStoreError,
    ) -> None:
        kind = _observation_kind(error.code)
        try:
            self._append_observation(receipt, kind)
        except Exception:
            # The original R2 integrity failure remains decisive. A failed audit
            # append must not disguise it or prompt an unsafe retry decision.
            return

    def _run_transaction(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[PostgresConnection], _T],
    ) -> _T:
        try:
            return self._authority.run_authority_transaction(
                write=write,
                operation_name=operation_name,
                operation=operation,
            )
        except PostgresR2ArtifactError:
            raise
        except Exception as error:
            raise PostgresR2ArtifactError(
                f"PostgreSQL R2 artifact authority failed: {operation_name}"
            ) from error

    def _require_stage_matches(
        self,
        receipt: RawProviderR2ArtifactReceipt,
        *,
        record: StoredRawProviderBytes,
        inference_id: str,
        logical_key: str,
        locator: ObjectLocator,
    ) -> None:
        self._require_record_matches_receipt(record, receipt)
        self._require_current_r2_configuration(receipt)
        if (
            receipt.inference_id != inference_id
            or receipt.logical_key != logical_key
            or receipt.object_uri != locator.uri
            or receipt.object_version != locator.object_version
        ):
            raise PostgresR2ArtifactError(
                "existing R2 artifact receipt conflicts with the immutable raw-evidence plan"
            )

    def _require_current_r2_configuration(self, receipt: RawProviderR2ArtifactReceipt) -> None:
        if receipt.r2_config_sha256 != self._r2_config_sha256:
            raise PostgresR2ArtifactError(
                "R2 configuration changed since the raw-evidence receipt was staged"
            )

    @staticmethod
    def _require_record(record: StoredRawProviderBytes) -> None:
        if not isinstance(record, StoredRawProviderBytes):
            raise TypeError("record must be StoredRawProviderBytes")

    @staticmethod
    def _require_record_matches_receipt(
        record: StoredRawProviderBytes,
        receipt: RawProviderR2ArtifactReceipt,
    ) -> None:
        if (
            receipt.artifact_id != record.artifact_id
            or receipt.request_id != record.request_id
            or receipt.provider_request_id != record.provider_request_id
            or receipt.exact_bytes_sha256 != record.exact_bytes_sha256
            or receipt.byte_count != record.byte_count
            or receipt.media_type != record.media_type
            or receipt.payload != record.data
        ):
            raise PostgresR2ArtifactError(
                "R2 artifact receipt conflicts with immutable raw provider evidence"
            )

    @staticmethod
    def _require_receipts_match(
        expected: RawProviderR2ArtifactReceipt,
        actual: RawProviderR2ArtifactReceipt,
    ) -> None:
        if (
            expected.artifact_id != actual.artifact_id
            or expected.inference_id != actual.inference_id
            or expected.request_id != actual.request_id
            or expected.provider_request_id != actual.provider_request_id
            or expected.exact_bytes_sha256 != actual.exact_bytes_sha256
            or expected.byte_count != actual.byte_count
            or expected.media_type != actual.media_type
            or expected.payload != actual.payload
            or expected.logical_key != actual.logical_key
            or expected.object_uri != actual.object_uri
            or expected.object_version != actual.object_version
            or expected.r2_config_sha256 != actual.r2_config_sha256
        ):
            raise PostgresR2ArtifactError("R2 artifact receipt changed after durable staging")

    @staticmethod
    def _receipt_from_row(row: Mapping[str, object]) -> RawProviderR2ArtifactReceipt:
        values = row
        artifact_id = _row_text(values, "artifact_id")
        inference_id = _row_text(values, "inference_id")
        request_id = _row_text(values, "request_id")
        provider_request_id = _row_text(values, "provider_request_id")
        digest = _row_text(values, "exact_bytes_sha256")
        byte_count = _row_int(values, "byte_count")
        media_type = _row_text(values, "media_type")
        payload = _row_bytes(values, "payload_bytes")
        logical_key = _row_text(values, "logical_key")
        object_uri = _row_text(values, "object_uri")
        object_version = _row_text(values, "object_version")
        object_etag = values.get("object_etag")
        if object_etag is not None and not isinstance(object_etag, str):
            raise PostgresR2ArtifactError("R2 artifact receipt object_etag is malformed")
        r2_config_sha256 = _row_text(values, "r2_config_sha256")
        state_text = _row_text(values, "state")
        try:
            state = RawProviderR2ArtifactState(state_text)
            receipt = RawProviderR2ArtifactReceipt(
                artifact_id=artifact_id,
                inference_id=inference_id,
                request_id=request_id,
                provider_request_id=provider_request_id,
                exact_bytes_sha256=digest,
                byte_count=byte_count,
                media_type=media_type,
                payload=payload,
                logical_key=logical_key,
                object_uri=object_uri,
                object_version=object_version,
                object_etag=object_etag,
                r2_config_sha256=r2_config_sha256,
                state=state,
            )
            if len(payload) != byte_count or exact_bytes_sha256(payload) != digest:
                raise ValueError("payload bytes do not match receipt digest")
            _ = receipt.locator
            _ = receipt.object_put_request()
            return receipt
        except (TypeError, ValueError) as error:
            raise PostgresR2ArtifactError("R2 artifact receipt row is invalid") from error


def _positive_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("limit must be a positive integer")
    return value


def _observation_kind(code: ObjectStoreErrorCode) -> RawProviderR2ObservationKind:
    if code is ObjectStoreErrorCode.NOT_FOUND:
        return RawProviderR2ObservationKind.MISSING
    if code is ObjectStoreErrorCode.CONFLICT:
        return RawProviderR2ObservationKind.CONFLICT
    if code in {ObjectStoreErrorCode.VISIBILITY_UNKNOWN, ObjectStoreErrorCode.INTEGRITY_ERROR}:
        return RawProviderR2ObservationKind.CORRUPT
    return RawProviderR2ObservationKind.PROVIDER_ERROR


def _row_text(row: Mapping[str, object], column: str) -> str:
    value = row.get(column)
    if not isinstance(value, str) or not value:
        raise PostgresR2ArtifactError(f"R2 artifact receipt {column} is malformed")
    return value


def _row_int(row: Mapping[str, object], column: str) -> int:
    value = row.get(column)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PostgresR2ArtifactError(f"R2 artifact receipt {column} is malformed")
    return value


def _row_bytes(row: Mapping[str, object], column: str) -> bytes:
    value = row.get(column)
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, bytes) or not value:
        raise PostgresR2ArtifactError(f"R2 artifact receipt {column} is malformed")
    return value


__all__ = [
    "PostgresR2ArtifactAuthority",
    "PostgresR2ArtifactError",
    "RawProviderR2ArtifactReceipt",
    "RawProviderR2ArtifactState",
    "RawProviderR2ObservationKind",
]
