"""Fail-closed contract for a future PostgreSQL canonical production composition.

This module deliberately does not construct a canonical worker.  The current
canonical implementation is an explicit SQLite/OfflineFixture conformance path,
and a configuration switch must never make it production-authoritative.  The
contract collects non-secret target bindings, validates the intended PostgreSQL
and provider boundary, and refuses authoritative startup until the required
PostgreSQL canonical adapters and release verifier exist.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, NoReturn, Protocol

from pydantic import Field, StringConstraints, model_validator

from robata.adapters.pgvector_runtime import PgVectorRuntimeConfig
from robata.adapters.r2_object_store import R2ObjectStoreConfig
from robata.application.canonical.production_routing import (
    ModelDeployment,
    ProductionRoute,
    ProductionRouteAuthorization,
    endpoint_config_digest,
)
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.inference.runpod import RunPodEndpointConfig

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
PortNumber = Annotated[int, Field(strict=True, ge=1, le=65_535)]
PositiveInt = Annotated[int, Field(strict=True, ge=1, le=120)]
PostgresIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=63,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    ),
]

_POSTGRES_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOCAL_AUTHORITY_MODULE_PREFIXES = (
    "robata.adapters.sqlite",
    "robata.application.canonical.local_",
    "robata.inference.offline_fixture",
)
_REQUIRED_CANONICAL_ADAPTERS: tuple[str, ...] = (
    "work_scheduler_and_stream_authority",
    "primary_completion_and_outbox",
    "inference_evidence_and_barriers",
    "logical_nodes_and_review",
    "production_read_model",
)


class CanonicalPostgresConnectionConfig(StrictModel):
    """Non-secret TLS configuration for one canonical PostgreSQL runtime role."""

    host: NonEmptyString
    database: NonEmptyString
    user: NonEmptyString
    port: PortNumber = 5432
    sslmode: Literal["verify-full"] = "verify-full"
    sslrootcert: NonEmptyString
    connect_timeout_seconds: PositiveInt = 10
    application_name: NonEmptyString

    @model_validator(mode="after")
    def validate_connection_values(self) -> CanonicalPostgresConnectionConfig:
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


class CanonicalPostgresRuntimeConfig(StrictModel):
    """Declared PostgreSQL/Supabase authority target, without credentials.

    This is intentionally separate from ``PgVectorRuntimeConfig``.  pgvector is
    a derived projection; these fields identify the future canonical authority.
    """

    application: CanonicalPostgresConnectionConfig
    worker: CanonicalPostgresConnectionConfig
    schema_name: PostgresIdentifier = "robata_canonical"
    require_rls: Literal[True] = True
    tenant_context_setting: NonEmptyString = "robata.tenant_id"

    @model_validator(mode="after")
    def validate_runtime(self) -> CanonicalPostgresRuntimeConfig:
        if self.application.user == self.worker.user:
            raise ValueError("canonical application and worker users must differ")
        if (
            self.schema_name != self.schema_name.strip()
            or _POSTGRES_IDENTIFIER.fullmatch(self.schema_name) is None
        ):
            raise ValueError("schema_name must be a PostgreSQL identifier")
        if self.tenant_context_setting != self.tenant_context_setting.strip() or any(
            character.isspace() for character in self.tenant_context_setting
        ):
            raise ValueError("tenant_context_setting must not contain whitespace")
        return self

    @property
    def configuration_sha256(self) -> Sha256Digest:
        """Return a credential-free, content-addressed target configuration."""

        return semantic_sha256(self.model_dump(mode="json"))

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> CanonicalPostgresRuntimeConfig:
        """Read an explicit mapping and validate credentials without retaining them."""

        if not isinstance(environment, Mapping):
            raise TypeError("environment must be a mapping")
        if environment.get("CANONICAL_POSTGRES_REQUIRE_RLS") not in {None, "true"}:
            raise ValueError("CANONICAL_POSTGRES_REQUIRE_RLS cannot disable RLS")
        _environment_required(environment, "CANONICAL_POSTGRES_APP_PASSWORD")
        _environment_required(environment, "CANONICAL_POSTGRES_WORKER_PASSWORD")
        return cls(
            application=_connection_from_environment(
                environment,
                role="APP",
                default_application_name="robata-canonical-app",
            ),
            worker=_connection_from_environment(
                environment,
                role="WORKER",
                default_application_name="robata-canonical-worker",
            ),
            schema_name=environment.get("CANONICAL_POSTGRES_SCHEMA", "robata_canonical"),
            tenant_context_setting=environment.get(
                "CANONICAL_POSTGRES_TENANT_CONTEXT_SETTING", "robata.tenant_id"
            ),
        )


class ProductionPrimaryRunPodBinding(StrictModel):
    """Non-secret pin for the only planned authoritative RunPod deployment."""

    endpoint: RunPodEndpointConfig
    handler_image_sha256: Sha256Digest
    capability_snapshot_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_primary_binding(self) -> ProductionPrimaryRunPodBinding:
        deployment = self.endpoint.deployment_configuration
        if self.endpoint.provider != "runpod":
            raise ValueError("production primary endpoint provider must be runpod")
        if deployment is None:
            raise ValueError("production primary endpoint requires a deployment pin")
        return self

    @property
    def configuration_sha256(self) -> Sha256Digest:
        return semantic_sha256(
            {
                "endpoint": self.endpoint.model_dump(mode="json"),
                "handler_image_sha256": self.handler_image_sha256,
                "capability_snapshot_sha256": self.capability_snapshot_sha256,
            }
        )


class ProductionCompositionState(StrEnum):
    """Truthful state for the currently declarative production target."""

    DECLARED_NOT_RUNNABLE = "DECLARED_NOT_RUNNABLE"


class ProductionCompositionReadiness(StrictModel):
    """Credential-free report that cannot be mistaken for release evidence."""

    format_version: Literal["robata-production-composition-readiness-v1"] = (
        "robata-production-composition-readiness-v1"
    )
    state: Literal[ProductionCompositionState.DECLARED_NOT_RUNNABLE] = (
        ProductionCompositionState.DECLARED_NOT_RUNNABLE
    )
    canonical_authority_backend: Literal["POSTGRESQL"] = "POSTGRESQL"
    canonical_postgres_configuration_sha256: Sha256Digest
    r2_configuration_sha256: Sha256Digest
    pgvector_configuration_sha256: Sha256Digest
    primary_runpod_configuration_sha256: Sha256Digest
    primary_route_release_verifier: Literal["REQUIRED"] = "REQUIRED"
    required_canonical_adapters: tuple[str, ...] = _REQUIRED_CANONICAL_ADAPTERS
    production_eligible: Literal[False] = False


class ProductionCompositionErrorCode(StrEnum):
    """Stable failure reasons for an attempted authority startup."""

    MISSING_CANONICAL_POSTGRES_AUTHORITY = "MISSING_CANONICAL_POSTGRES_AUTHORITY"
    LOCAL_CANONICAL_AUTHORITY_FORBIDDEN = "LOCAL_CANONICAL_AUTHORITY_FORBIDDEN"
    INVALID_CANONICAL_AUTHORITY = "INVALID_CANONICAL_AUTHORITY"
    MISSING_PRIMARY_ROUTE = "MISSING_PRIMARY_ROUTE"
    PRIMARY_ROUTE_MISMATCH = "PRIMARY_ROUTE_MISMATCH"
    MISSING_PRIMARY_ROUTE_RELEASE_VERIFIER = "MISSING_PRIMARY_ROUTE_RELEASE_VERIFIER"
    PRIMARY_ROUTE_RELEASE_NOT_VERIFIED = "PRIMARY_ROUTE_RELEASE_NOT_VERIFIED"
    CANONICAL_ADAPTERS_UNIMPLEMENTED = "CANONICAL_ADAPTERS_UNIMPLEMENTED"


class ProductionCompositionError(RuntimeError):
    """An authoritative production startup was rejected before any work began."""

    def __init__(self, code: ProductionCompositionErrorCode, detail: str) -> None:
        self.code = code
        super().__init__(detail)


class ProductionCanonicalAuthority(Protocol):
    """Future PostgreSQL authority boundary required by production composition."""

    @property
    def backend_kind(self) -> str:
        """Return the durable backend identity, which must be ``POSTGRESQL``."""

    def verify_startup(self) -> None:
        """Verify migrations, worker identity, and durable authority invariants."""


class PrimaryRouteReleaseVerifier(Protocol):
    """Independent release gate for a primary authoritative model route."""

    def verify_primary(
        self,
        *,
        authorization: ProductionRouteAuthorization,
        deployment: ModelDeployment,
    ) -> bool:
        """Return true only when exact release evidence authorizes this deployment."""


class ProductionCompositionContract(StrictModel):
    """Typed target declaration which refuses to construct a worker today."""

    canonical_postgres: CanonicalPostgresRuntimeConfig
    r2: R2ObjectStoreConfig
    pgvector: PgVectorRuntimeConfig
    primary_runpod: ProductionPrimaryRunPodBinding

    @property
    def readiness(self) -> ProductionCompositionReadiness:
        """Return the declared target and its deliberate non-runnable status."""

        return ProductionCompositionReadiness(
            canonical_postgres_configuration_sha256=(self.canonical_postgres.configuration_sha256),
            r2_configuration_sha256=semantic_sha256(self.r2.model_dump(mode="json")),
            pgvector_configuration_sha256=semantic_sha256(self.pgvector.model_dump(mode="json")),
            primary_runpod_configuration_sha256=self.primary_runpod.configuration_sha256,
        )

    def require_authoritative_startup(
        self,
        *,
        canonical_authority: object | None = None,
        primary_route: ProductionRoute | None = None,
        release_verifier: object | None = None,
    ) -> NoReturn:
        """Reject attempted authority startup until all production adapters exist.

        This is intentionally a terminal guard.  A future real composition can
        reuse its binding checks, but cannot bypass the missing-adapter refusal
        by setting an environment variable or passing a local SQLite object.
        """

        authority = _require_postgres_authority(canonical_authority)
        _verify_authority_startup(authority)
        checked_route = _require_matching_primary_route(self.primary_runpod, primary_route)
        _verify_primary_route_release(checked_route, release_verifier)
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.CANONICAL_ADAPTERS_UNIMPLEMENTED,
            "PostgreSQL canonical scheduler, stream authority, completion, evidence, "
            "barrier, outbox, logical-node, review, and read-model adapters are not "
            "implemented; production composition cannot start",
        )


def _connection_from_environment(
    environment: Mapping[str, str],
    *,
    role: Literal["APP", "WORKER"],
    default_application_name: str,
) -> CanonicalPostgresConnectionConfig:
    return CanonicalPostgresConnectionConfig(
        host=_environment_required(environment, "CANONICAL_POSTGRES_HOST"),
        database=_environment_required(environment, "CANONICAL_POSTGRES_DATABASE"),
        user=_environment_required(environment, f"CANONICAL_POSTGRES_{role}_USER"),
        port=_environment_positive_int(environment, "CANONICAL_POSTGRES_PORT", 5432),
        sslmode=_environment_verify_full(environment, "CANONICAL_POSTGRES_SSLMODE"),
        sslrootcert=_environment_required(environment, "CANONICAL_POSTGRES_SSLROOTCERT"),
        connect_timeout_seconds=_environment_positive_int(
            environment, "CANONICAL_POSTGRES_CONNECT_TIMEOUT_SECONDS", 10
        ),
        application_name=environment.get(
            f"CANONICAL_POSTGRES_{role}_APPLICATION_NAME", default_application_name
        ),
    )


def _environment_required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be configured")
    return value


def _environment_positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    value = environment.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _environment_verify_full(environment: Mapping[str, str], name: str) -> Literal["verify-full"]:
    value = environment.get(name, "verify-full")
    if value != "verify-full":
        raise ValueError(f"{name} must be verify-full")
    return "verify-full"


def _require_postgres_authority(value: object | None) -> ProductionCanonicalAuthority:
    if value is None:
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.MISSING_CANONICAL_POSTGRES_AUTHORITY,
            "authoritative startup requires PostgreSQL canonical authority adapters",
        )
    module = type(value).__module__
    if module.startswith(_LOCAL_AUTHORITY_MODULE_PREFIXES):
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.LOCAL_CANONICAL_AUTHORITY_FORBIDDEN,
            "SQLite and OfflineFixture authorities cannot be used for production startup",
        )
    backend_kind = getattr(value, "backend_kind", None)
    verify_startup = getattr(value, "verify_startup", None)
    if backend_kind != "POSTGRESQL" or not callable(verify_startup):
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.INVALID_CANONICAL_AUTHORITY,
            "canonical authority must declare POSTGRESQL and implement verify_startup()",
        )
    return value  # type: ignore[return-value]


def _verify_authority_startup(authority: ProductionCanonicalAuthority) -> None:
    try:
        authority.verify_startup()
    except ProductionCompositionError:
        raise
    except Exception as error:
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.INVALID_CANONICAL_AUTHORITY,
            "PostgreSQL canonical authority startup verification failed",
        ) from error


def _require_matching_primary_route(
    binding: ProductionPrimaryRunPodBinding,
    route: ProductionRoute | None,
) -> ProductionRoute:
    if route is None:
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.MISSING_PRIMARY_ROUTE,
            "authoritative startup requires a pinned ProductionRoute",
        )
    if not isinstance(route, ProductionRoute):
        raise TypeError("primary_route must be ProductionRoute or None")
    deployment = binding.endpoint.deployment_configuration
    assert deployment is not None
    candidate = route.deployment
    if (
        candidate.provider != binding.endpoint.provider
        or candidate.model_name != deployment.model_identifier
        or candidate.model_version != deployment.model_version
        or candidate.adapter_version != binding.endpoint.adapter_version
        or candidate.capability_snapshot_digest != binding.capability_snapshot_sha256
        or candidate.endpoint_config_digest != endpoint_config_digest(binding.endpoint)
        or candidate.max_concurrent_requests != binding.endpoint.max_concurrent_requests
    ):
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.PRIMARY_ROUTE_MISMATCH,
            "ProductionRoute deployment does not match the pinned primary RunPod binding",
        )
    return route


def _verify_primary_route_release(route: ProductionRoute, verifier: object | None) -> None:
    if verifier is None:
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.MISSING_PRIMARY_ROUTE_RELEASE_VERIFIER,
            "authoritative startup requires an independent primary-route release verifier",
        )
    verify_primary = getattr(verifier, "verify_primary", None)
    if not callable(verify_primary):
        raise TypeError("release_verifier must implement verify_primary()")
    try:
        verified = verify_primary(
            authorization=route.authorization,
            deployment=route.deployment,
        )
    except Exception as error:
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.PRIMARY_ROUTE_RELEASE_NOT_VERIFIED,
            "primary-route release verification failed",
        ) from error
    if verified is not True:
        raise ProductionCompositionError(
            ProductionCompositionErrorCode.PRIMARY_ROUTE_RELEASE_NOT_VERIFIED,
            "primary-route release evidence was not verified",
        )


__all__ = [
    "CanonicalPostgresConnectionConfig",
    "CanonicalPostgresRuntimeConfig",
    "PrimaryRouteReleaseVerifier",
    "ProductionCanonicalAuthority",
    "ProductionCompositionContract",
    "ProductionCompositionError",
    "ProductionCompositionErrorCode",
    "ProductionCompositionReadiness",
    "ProductionCompositionState",
    "ProductionPrimaryRunPodBinding",
]
