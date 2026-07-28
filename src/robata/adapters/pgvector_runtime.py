"""Explicit psycopg runtime wiring for the optional pgvector projection store.

The existing ``PgVectorProjectionStore`` owns all SQL, RLS checks, transaction
semantics, and vector contracts. This module only turns an explicit non-secret
connection configuration plus redacted credentials into injected DB-API
factories. Nothing reads ``os.environ`` or opens a database connection at import
time, so local composition remains fail-closed until a caller opts in.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Annotated, Literal, Self, cast

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import StrictModel
from robata.contracts.retrieval import VectorBackend
from robata.ports.vector_projection import (
    PgVectorProjectionStore,
    PgVectorSqlConnection,
    VectorProjectionError,
    VectorProjectionErrorCode,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
PortNumber = Annotated[int, Field(strict=True, ge=1, le=65_535)]
PositiveInt = Annotated[int, Field(strict=True, ge=1, le=120)]
VectorDimension = Annotated[int, Field(strict=True, ge=1, le=65_535)]
SecureSslMode = Literal["verify-full"]

_PG_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WORKER_ENVIRONMENT_KEYS = (
    "PGVECTOR_WORKER_HOST",
    "PGVECTOR_WORKER_DATABASE",
    "PGVECTOR_WORKER_USER",
    "PGVECTOR_WORKER_PORT",
    "PGVECTOR_WORKER_SSLMODE",
    "PGVECTOR_WORKER_SSLROOTCERT",
    "PGVECTOR_WORKER_CONNECT_TIMEOUT_SECONDS",
    "PGVECTOR_WORKER_APPLICATION_NAME",
)


class PgVectorConnectionConfig(StrictModel):
    """Non-secret TLS connection settings for one PostgreSQL role."""

    host: NonEmptyString
    database: NonEmptyString
    user: NonEmptyString
    port: PortNumber = 5432
    sslmode: SecureSslMode = "verify-full"
    sslrootcert: NonEmptyString
    connect_timeout_seconds: PositiveInt = 10
    application_name: NonEmptyString = "robata-pgvector"

    @model_validator(mode="after")
    def validate_connection_values(self) -> Self:
        for label, value in (
            ("host", self.host),
            ("database", self.database),
            ("user", self.user),
            ("application_name", self.application_name),
        ):
            if value != value.strip() or any(character.isspace() for character in value):
                raise ValueError(f"{label} must not contain whitespace")
        if self.sslrootcert != self.sslrootcert.strip():
            raise ValueError("sslrootcert must not start or end with whitespace")
        return self

    @classmethod
    def from_environment(cls, environment: Mapping[str, str], *, prefix: str) -> Self:
        """Build one non-secret connection configuration from an explicit mapping."""

        return cls(
            host=_environment_required(environment, f"{prefix}HOST"),
            database=_environment_required(environment, f"{prefix}DATABASE"),
            user=_environment_required(environment, f"{prefix}USER"),
            port=_environment_positive_int(environment, f"{prefix}PORT", 5432),
            sslmode=_environment_sslmode(environment, f"{prefix}SSLMODE"),
            sslrootcert=_environment_required(environment, f"{prefix}SSLROOTCERT"),
            connect_timeout_seconds=_environment_positive_int(
                environment, f"{prefix}CONNECT_TIMEOUT_SECONDS", 10
            ),
            application_name=environment.get(f"{prefix}APPLICATION_NAME", "robata-pgvector"),
        )


class PgVectorCredentials:
    """Opaque database password; repr/str never expose it."""

    __slots__ = ("_password",)

    def __init__(self, password: str) -> None:
        if not isinstance(password, str) or not password:
            raise ValueError("password must be non-empty")
        self._password = password

    @property
    def password(self) -> str:
        return self._password

    @classmethod
    def from_environment(cls, environment: Mapping[str, str], *, variable: str) -> Self:
        password = environment.get(variable)
        if not password:
            raise ValueError(f"{variable} must be configured")
        return cls(password)

    def __repr__(self) -> str:
        return "PgVectorCredentials(password=REDACTED)"

    __str__ = __repr__


class PgVectorRuntimeConfig(StrictModel):
    """Explicit production-shaped configuration for one optional pgvector store."""

    primary: PgVectorConnectionConfig
    worker: PgVectorConnectionConfig | None = None
    dimension: VectorDimension
    relation: NonEmptyString = "public.robata_vector_projection"
    vector_column: NonEmptyString = "embedding"
    backend: VectorBackend = VectorBackend.POSTGRES
    index_name: NonEmptyString | None = None
    require_rls: Literal[True] = True
    worker_role: NonEmptyString | None = None
    rls_policy_name: NonEmptyString | None = None
    tenant_context_setting: NonEmptyString = "robata.tenant_id"

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        if self.backend not in {VectorBackend.POSTGRES, VectorBackend.SUPABASE}:
            raise ValueError("pgvector runtime backend must be POSTGRES or SUPABASE")
        if (self.worker is None) != (self.worker_role is None):
            raise ValueError("worker and worker_role must be configured together")
        if self.worker is None:
            raise ValueError("RLS-enabled pgvector runtime requires an explicit worker connection")
        for label, value in (
            ("relation", self.relation),
            ("vector_column", self.vector_column),
            ("tenant_context_setting", self.tenant_context_setting),
        ):
            if value != value.strip() or any(character.isspace() for character in value):
                raise ValueError(f"{label} must not contain whitespace")
        optional_identifiers: tuple[tuple[str, str | None], ...] = (
            ("worker_role", self.worker_role),
            ("rls_policy_name", self.rls_policy_name),
        )
        for label, identifier in optional_identifiers:
            if identifier is not None and (
                len(identifier.encode("utf-8")) > 63 or _PG_IDENTIFIER.fullmatch(identifier) is None
            ):
                raise ValueError(f"{label} must be a PostgreSQL identifier of at most 63 bytes")
        return self

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> Self:
        """Build config from an explicit environment map, never process globals."""

        if environment.get("PGVECTOR_REQUIRE_RLS") not in {None, "true"}:
            raise ValueError("PGVECTOR_REQUIRE_RLS cannot disable RLS for the production runtime")
        worker_present = any(key in environment for key in _WORKER_ENVIRONMENT_KEYS)
        backend_value = environment.get("PGVECTOR_BACKEND", VectorBackend.POSTGRES.value)
        try:
            backend = VectorBackend(backend_value)
        except ValueError as error:
            raise ValueError("PGVECTOR_BACKEND must be POSTGRES or SUPABASE") from error
        return cls(
            primary=PgVectorConnectionConfig.from_environment(environment, prefix="PGVECTOR_"),
            worker=(
                PgVectorConnectionConfig.from_environment(environment, prefix="PGVECTOR_WORKER_")
                if worker_present
                else None
            ),
            dimension=_environment_positive_int(environment, "PGVECTOR_DIMENSION", None),
            relation=environment.get("PGVECTOR_RELATION", "public.robata_vector_projection"),
            vector_column=environment.get("PGVECTOR_VECTOR_COLUMN", "embedding"),
            backend=backend,
            index_name=_environment_optional_text(environment, "PGVECTOR_INDEX_NAME"),
            worker_role=_environment_optional_text(environment, "PGVECTOR_WORKER_ROLE"),
            rls_policy_name=_environment_optional_text(environment, "PGVECTOR_RLS_POLICY_NAME"),
            tenant_context_setting=environment.get(
                "PGVECTOR_TENANT_CONTEXT_SETTING", "robata.tenant_id"
            ),
        )


def create_psycopg_connection_factory(
    connection: PgVectorConnectionConfig,
    credentials: PgVectorCredentials,
) -> Callable[[], PgVectorSqlConnection]:
    """Return a lazy DB-API factory without opening a connection yet."""

    if not isinstance(connection, PgVectorConnectionConfig):
        raise TypeError("connection must be PgVectorConnectionConfig")
    if not isinstance(credentials, PgVectorCredentials):
        raise TypeError("credentials must be PgVectorCredentials")
    connect = _load_psycopg_connect()

    def factory() -> PgVectorSqlConnection:
        return cast(
            PgVectorSqlConnection,
            connect(
                host=connection.host,
                port=connection.port,
                dbname=connection.database,
                user=connection.user,
                password=credentials.password,
                sslmode=connection.sslmode,
                sslrootcert=connection.sslrootcert,
                connect_timeout=connection.connect_timeout_seconds,
                application_name=connection.application_name,
                autocommit=False,
            ),
        )

    return factory


def create_pgvector_projection_store(
    config: PgVectorRuntimeConfig,
    primary_credentials: PgVectorCredentials,
    *,
    worker_credentials: PgVectorCredentials | None = None,
) -> PgVectorProjectionStore:
    """Construct the physical adapter without provisioning or verifying a target DB."""

    if not isinstance(config, PgVectorRuntimeConfig):
        raise TypeError("config must be PgVectorRuntimeConfig")
    if not isinstance(primary_credentials, PgVectorCredentials):
        raise TypeError("primary_credentials must be PgVectorCredentials")
    if config.worker is None:
        if worker_credentials is not None:
            raise ValueError("worker_credentials requires a configured worker connection")
        worker_factory = None
    else:
        if not isinstance(worker_credentials, PgVectorCredentials):
            raise ValueError("configured worker connection requires worker_credentials")
        worker_factory = create_psycopg_connection_factory(config.worker, worker_credentials)
    return PgVectorProjectionStore(
        create_psycopg_connection_factory(config.primary, primary_credentials),
        dimension=config.dimension,
        relation=config.relation,
        vector_column=config.vector_column,
        backend=config.backend,
        index_name=config.index_name,
        require_rls=True,
        worker_connection_factory=worker_factory,
        worker_role=config.worker_role,
        rls_policy_name=config.rls_policy_name,
        tenant_context_setting=config.tenant_context_setting,
    )


def create_pgvector_projection_store_from_environment(
    environment: Mapping[str, str],
) -> PgVectorProjectionStore:
    """Build an optional physical store from one caller-owned environment mapping."""

    config = PgVectorRuntimeConfig.from_environment(environment)
    return create_pgvector_projection_store(
        config,
        PgVectorCredentials.from_environment(environment, variable="PGVECTOR_PASSWORD"),
        worker_credentials=(
            PgVectorCredentials.from_environment(environment, variable="PGVECTOR_WORKER_PASSWORD")
            if config.worker is not None
            else None
        ),
    )


def create_verified_pgvector_projection_store(
    config: PgVectorRuntimeConfig,
    primary_credentials: PgVectorCredentials,
    *,
    worker_credentials: PgVectorCredentials | None = None,
) -> PgVectorProjectionStore:
    """Construct and prove the target backend plus worker role before use.

    This intentionally performs real target I/O. It must be called by an
    explicit deployment startup path after the reviewed DDL/RLS migration has
    been applied, never by local default composition.
    """

    store = create_pgvector_projection_store(
        config,
        primary_credentials,
        worker_credentials=worker_credentials,
    )
    store.verify_backend()
    # LIMIT 0 opens the independent worker transaction, checks its database
    # role, and cannot claim or transition a pending row.
    store.drain(limit=0)
    return store


def create_verified_pgvector_projection_store_from_environment(
    environment: Mapping[str, str],
) -> PgVectorProjectionStore:
    """Build and verify the optional production adapter from an explicit map."""

    config = PgVectorRuntimeConfig.from_environment(environment)
    return create_verified_pgvector_projection_store(
        config,
        PgVectorCredentials.from_environment(environment, variable="PGVECTOR_PASSWORD"),
        worker_credentials=PgVectorCredentials.from_environment(
            environment, variable="PGVECTOR_WORKER_PASSWORD"
        ),
    )


def _load_psycopg_connect() -> Callable[..., object]:
    try:
        module = import_module("psycopg")
    except ImportError as error:
        raise VectorProjectionError(
            VectorProjectionErrorCode.ADAPTER_UNAVAILABLE,
            "pgvector runtime requires the optional robata[pgvector] dependency group",
        ) from error
    connect = getattr(module, "connect", None)
    if not callable(connect):
        raise VectorProjectionError(
            VectorProjectionErrorCode.ADAPTER_UNAVAILABLE,
            "psycopg does not expose a callable connect function",
        )
    return cast(Callable[..., object], connect)


def _environment_required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be configured")
    return value


def _environment_optional_text(environment: Mapping[str, str], name: str) -> str | None:
    value = environment.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty when configured")
    return value


def _environment_sslmode(environment: Mapping[str, str], name: str) -> SecureSslMode:
    value = environment.get(name, "verify-full")
    if value == "verify-full":
        return cast(SecureSslMode, value)
    raise ValueError(f"{name} must be verify-full")


def _environment_positive_int(
    environment: Mapping[str, str], name: str, default: int | None
) -> int:
    value = environment.get(name)
    if value is None:
        if default is None:
            raise ValueError(f"{name} must be configured")
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


__all__ = [
    "PgVectorConnectionConfig",
    "PgVectorCredentials",
    "PgVectorRuntimeConfig",
    "create_pgvector_projection_store",
    "create_pgvector_projection_store_from_environment",
    "create_psycopg_connection_factory",
    "create_verified_pgvector_projection_store",
    "create_verified_pgvector_projection_store_from_environment",
]
