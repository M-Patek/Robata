"""Executable PostgreSQL/Supabase production composition for Robata.

The local conformance composition remains intentionally SQLite and fixture based.
This module is the separate production boundary: it has no fallback to local
state, requires explicit concrete cloud adapters, and verifies migration/RLS
state before exposing any canonical authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

from pydantic import StringConstraints, model_validator

if TYPE_CHECKING:
    from robata.application.canonical.perception_composition import (
        PerceptionCompositionDecision,
    )


from robata.adapters.postgres_authority import ConnectionFactory, PostgresCanonicalAuthority
from robata.adapters.postgres_capture_authority import PostgresCaptureAuthority
from robata.adapters.postgres_completion_evidence import (
    PostgresBarrierStorage,
    PostgresInferenceEvidenceLedger,
    PostgresPrimaryCompletionRepository,
    PostgresPrimaryOutboxDeliveryStore,
    verify_completion_evidence_schema,
)
from robata.adapters.postgres_logical_review import (
    PostgresLogicalNodeRegistry,
    PostgresReviewQueue,
)
from robata.adapters.postgres_migrations import PostgresMigrationRunner
from robata.adapters.postgres_r2_artifacts import PostgresR2ArtifactAuthority
from robata.adapters.postgres_read_model import PostgresCanonicalReadModel
from robata.adapters.postgres_stream_work_ledger import PostgresStreamWorkLedger
from robata.adapters.postgres_work_scheduler import PostgresWorkScheduler
from robata.adapters.r2_object_store import R2ObjectStore
from robata.application.canonical.production_composition import (
    CanonicalPostgresConnectionConfig,
    ProductionCompositionContract,
    ProductionPrimaryRunPodBinding,
)
from robata.application.canonical.production_routing import ProductionRoute, endpoint_config_digest
from robata.contracts.common import StrictModel
from robata.contracts.schema_registry import SchemaRegistry, default_schema_registry
from robata.inference.runpod import RunPodVisionAdapter
from robata.queue.outbox import OutboxRetryPolicy
from robata.runtime.observability import RuntimeObserver

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]

_CANONICAL_SCHEMA = "robata_canonical"
_TENANT_SETTING = "robata.tenant_id"


def _resolve_production_perception_composition(
    profile: str | None,
) -> PerceptionCompositionDecision:
    """Choose Mage stream for new production runs unless legacy is explicit.

    The historical window scheduler remains available for replay only through a
    caller-supplied legacy profile; production bootstraps without a profile do
    not silently reconstruct ``stream-window-dag-v4``.
    """

    from robata.application.canonical.perception_composition import (
        resolve_perception_composition,
    )

    return resolve_perception_composition(profile, allow_explicit_legacy_qwen=profile is not None)


_REQUIRED_CANONICAL_TABLES = (
    "work_items",
    "work_dependencies",
    "work_attempts",
    "stream_plans",
    "expected_windows",
    "stream_work_plans",
    "stream_backpressure_controllers",
    "capture_authority_metadata",
    "capture_authority_receipts",
    "primary_runs",
    "event_registry_partitions",
    "stable_event_identities",
    "event_identity_assignments",
    "event_identity_relations",
    "action_event_publications",
    "detailed_results",
    "primary_completions",
    "primary_outbox",
    "primary_outbox_deliveries",
    "inference_intents",
    "raw_provider_responses",
    "raw_provider_r2_artifact_receipts",
    "raw_provider_r2_artifact_observations",
    "model_inference_terminals",
    "raw_provider_artifacts",
    "inference_attempt_selections",
    "parsed_provider_claims",
    "selected_attempt_outputs",
    "enriched_provider_outputs",
    "calibration_artifacts",
    "inference_calibration_associations",
    "barrier_definitions",
    "barrier_states",
    "barrier_members",
    "inference_call_barrier_definitions",
    "inference_call_part_completions",
    "inference_call_reductions",
    "logical_nodes",
    "processing_run_nodes",
    "immutable_node_revisions",
    "selection_decisions",
    "current_selections",
    "review_tasks",
    "review_annotations",
    "review_reopen_commands",
)


class ProductionRuntimeErrorCode(StrEnum):
    """Stable reasons a production canonical runtime cannot start."""

    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    INVALID_DEPENDENCY = "INVALID_DEPENDENCY"
    MIGRATIONS_UNVERIFIED = "MIGRATIONS_UNVERIFIED"
    RLS_UNVERIFIED = "RLS_UNVERIFIED"
    PRIMARY_ROUTE_MISMATCH = "PRIMARY_ROUTE_MISMATCH"
    PRIMARY_ROUTE_RELEASE_NOT_VERIFIED = "PRIMARY_ROUTE_RELEASE_NOT_VERIFIED"
    PRIMARY_ADAPTER_MISMATCH = "PRIMARY_ADAPTER_MISMATCH"
    DERIVED_PROJECTION_UNVERIFIED = "DERIVED_PROJECTION_UNVERIFIED"


class ProductionRuntimeError(RuntimeError):
    """The production-only composition rejected an unsafe startup."""

    def __init__(self, code: ProductionRuntimeErrorCode, detail: str) -> None:
        self.code = code
        super().__init__(detail)


class CanonicalPostgresCredentials:
    """Opaque password for a single canonical PostgreSQL runtime role."""

    __slots__ = ("_password",)

    def __init__(self, password: str) -> None:
        if not isinstance(password, str) or not password:
            raise ValueError("password must be non-empty")
        self._password = password

    @property
    def password(self) -> str:
        return self._password

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str], *, variable: str
    ) -> CanonicalPostgresCredentials:
        password = environment.get(variable)
        if not password:
            raise ValueError(f"{variable} must be configured")
        return cls(password)

    def __repr__(self) -> str:
        return "CanonicalPostgresCredentials(password=REDACTED)"

    __str__ = __repr__


@dataclass(frozen=True, slots=True)
class CanonicalPostgresRuntimeCredentials:
    """Opaque application and worker credentials for one runtime."""

    application: CanonicalPostgresCredentials
    worker: CanonicalPostgresCredentials

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
    ) -> CanonicalPostgresRuntimeCredentials:
        if not isinstance(environment, Mapping):
            raise TypeError("environment must be a mapping")
        return cls(
            application=CanonicalPostgresCredentials.from_environment(
                environment,
                variable="CANONICAL_POSTGRES_APP_PASSWORD",
            ),
            worker=CanonicalPostgresCredentials.from_environment(
                environment,
                variable="CANONICAL_POSTGRES_WORKER_PASSWORD",
            ),
        )


class ProductionTenantContext(StrictModel):
    """The one tenant bound transaction-locally to this runtime process."""

    tenant_id: NonEmptyString

    @model_validator(mode="after")
    def validate_tenant_id(self) -> ProductionTenantContext:
        if self.tenant_id != self.tenant_id.strip() or any(
            character.isspace() for character in self.tenant_id
        ):
            raise ValueError("tenant_id must not contain whitespace")
        return self


class ProductionCaptureAuthorityBinding(StrictModel):
    """Pinned authority facts used to issue immutable capture subjects."""

    capture_authority_id: NonEmptyString
    capture_authority_epoch: int
    capture_assignment_policy_version: NonEmptyString

    @model_validator(mode="after")
    def validate_capture_authority(self) -> ProductionCaptureAuthorityBinding:
        if self.capture_authority_epoch < 1:
            raise ValueError("capture_authority_epoch must be positive")
        return self


class PrimaryRouteReleaseVerifier:
    """Structural protocol for an independent primary route release gate."""

    def verify_primary(self, *, authorization: object, deployment: object) -> bool:
        raise NotImplementedError


PrimaryAdapterFactory = Callable[[PostgresInferenceEvidenceLedger], RunPodVisionAdapter]


@dataclass(frozen=True, slots=True)
class ProductionCanonicalRuntime:
    """Concrete, production-only canonical adapters sharing PostgreSQL authority."""

    contract: ProductionCompositionContract
    tenant: ProductionTenantContext
    application_authority: PostgresCanonicalAuthority
    worker_authority: PostgresCanonicalAuthority
    work_scheduler: PostgresWorkScheduler
    stream_work_ledger: PostgresStreamWorkLedger
    capture_authority: PostgresCaptureAuthority
    primary_completion: PostgresPrimaryCompletionRepository
    inference_evidence: PostgresInferenceEvidenceLedger
    r2_artifact_authority: PostgresR2ArtifactAuthority
    barrier_storage: PostgresBarrierStorage
    outbox_delivery: PostgresPrimaryOutboxDeliveryStore
    logical_node_registry: PostgresLogicalNodeRegistry
    review_queue: PostgresReviewQueue
    read_model: PostgresCanonicalReadModel
    r2_object_store: R2ObjectStore
    primary_adapter: RunPodVisionAdapter
    primary_route: ProductionRoute
    pgvector_projection: object


def create_canonical_postgres_connection_factory(
    connection: CanonicalPostgresConnectionConfig,
    credentials: CanonicalPostgresCredentials,
) -> ConnectionFactory:
    """Create a lazy Psycopg factory from typed TLS settings and opaque secrets."""

    if not isinstance(connection, CanonicalPostgresConnectionConfig):
        raise TypeError("connection must be CanonicalPostgresConnectionConfig")
    if not isinstance(credentials, CanonicalPostgresCredentials):
        raise TypeError("credentials must be CanonicalPostgresCredentials")

    def factory() -> object:
        try:
            psycopg = import_module("psycopg")
            rows = import_module("psycopg.rows")
        except ImportError as error:
            raise ProductionRuntimeError(
                ProductionRuntimeErrorCode.INVALID_DEPENDENCY,
                "canonical PostgreSQL runtime requires the robata[pgvector] Psycopg extra",
            ) from error
        connect = getattr(psycopg, "connect", None)
        dict_row = getattr(rows, "dict_row", None)
        if not callable(connect) or dict_row is None:
            raise ProductionRuntimeError(
                ProductionRuntimeErrorCode.INVALID_DEPENDENCY,
                "Psycopg does not expose the required connect/dict_row API",
            )
        return connect(
            host=connection.host,
            port=connection.port,
            dbname=connection.database,
            user=connection.user,
            password=credentials.password,
            sslmode=connection.sslmode,
            sslrootcert=connection.sslrootcert,
            connect_timeout=connection.connect_timeout_seconds,
            application_name=connection.application_name,
            autocommit=True,
            row_factory=dict_row,
        )

    return cast(ConnectionFactory, factory)


def build_production_canonical_runtime(
    *,
    contract: ProductionCompositionContract,
    credentials: CanonicalPostgresRuntimeCredentials,
    tenant: ProductionTenantContext,
    capture_authority: ProductionCaptureAuthorityBinding,
    r2_object_store: R2ObjectStore,
    pgvector_projection: object,
    primary_adapter_factory: PrimaryAdapterFactory,
    primary_route: ProductionRoute,
    release_verifier: object,
    outbox_retry_policy: OutboxRetryPolicy,
    schema_registry: SchemaRegistry | None = None,
    migrations_directory: Path | None = None,
    runtime_observer: RuntimeObserver | None = None,
) -> ProductionCanonicalRuntime:
    """Verify and construct the production-only canonical authority graph.

    This is intentionally explicit about every external boundary.  It performs
    database verification but never dispatches model or object-store traffic;
    the returned adapter is the caller's explicit route for later work.
    """

    if not isinstance(contract, ProductionCompositionContract):
        raise TypeError("contract must be ProductionCompositionContract")
    if not isinstance(credentials, CanonicalPostgresRuntimeCredentials):
        raise TypeError("credentials must be CanonicalPostgresRuntimeCredentials")
    if not isinstance(tenant, ProductionTenantContext):
        raise TypeError("tenant must be ProductionTenantContext")
    if not isinstance(capture_authority, ProductionCaptureAuthorityBinding):
        raise TypeError("capture_authority must be ProductionCaptureAuthorityBinding")
    if not isinstance(r2_object_store, R2ObjectStore):
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.INVALID_DEPENDENCY,
            "production composition requires an explicit R2ObjectStore, never a local object store",
        )
    if not isinstance(outbox_retry_policy, OutboxRetryPolicy):
        raise TypeError("outbox_retry_policy must be OutboxRetryPolicy")
    if not callable(primary_adapter_factory):
        raise TypeError("primary_adapter_factory must be callable")
    if schema_registry is not None and not isinstance(schema_registry, SchemaRegistry):
        raise TypeError("schema_registry must be SchemaRegistry or None")

    _verify_static_contract(contract, r2_object_store)
    _verify_primary_route(contract.primary_runpod, primary_route, release_verifier)
    _verify_projection(pgvector_projection)

    application_factory = create_canonical_postgres_connection_factory(
        contract.canonical_postgres.application,
        credentials.application,
    )
    worker_factory = create_canonical_postgres_connection_factory(
        contract.canonical_postgres.worker,
        credentials.worker,
    )
    migration_root = migrations_directory or _repository_migrations_directory()
    if not isinstance(migration_root, Path):
        raise TypeError("migrations_directory must be pathlib.Path or None")
    try:
        PostgresMigrationRunner(worker_factory, migration_root).verify()
    except Exception as error:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.MIGRATIONS_UNVERIFIED,
            "canonical PostgreSQL migrations are not verified for the worker role",
        ) from error

    application_authority = PostgresCanonicalAuthority(
        application_factory,
        schema=contract.canonical_postgres.schema_name,
        tenant_setting=contract.canonical_postgres.tenant_context_setting,
        tenant_id=tenant.tenant_id,
        runtime_observer=runtime_observer,
    )
    worker_authority = PostgresCanonicalAuthority(
        worker_factory,
        schema=contract.canonical_postgres.schema_name,
        tenant_setting=contract.canonical_postgres.tenant_context_setting,
        tenant_id=tenant.tenant_id,
        runtime_observer=runtime_observer,
    )
    _verify_canonical_authority(application_authority, runtime_role="application")
    _verify_canonical_authority(worker_authority, runtime_role="worker")

    registry = schema_registry or default_schema_registry()
    work_scheduler = PostgresWorkScheduler(worker_authority)
    stream_work_ledger = PostgresStreamWorkLedger(work_scheduler)
    capture = PostgresCaptureAuthority(
        worker_authority,
        capture_authority_id=capture_authority.capture_authority_id,
        capture_authority_epoch=capture_authority.capture_authority_epoch,
        capture_assignment_policy_version=capture_authority.capture_assignment_policy_version,
    )
    primary_completion = PostgresPrimaryCompletionRepository(
        worker_authority,
        registry=registry,
        runtime_observer=runtime_observer,
    )
    r2_artifacts = PostgresR2ArtifactAuthority(
        worker_authority,
        r2_object_store,
        tenant_id=tenant.tenant_id,
    )
    evidence = PostgresInferenceEvidenceLedger(
        worker_authority,
        registry,
        artifact_authority=r2_artifacts,
        runtime_observer=runtime_observer,
    )
    barriers = PostgresBarrierStorage(
        worker_authority,
        runtime_observer=runtime_observer,
    )
    outbox_delivery = PostgresPrimaryOutboxDeliveryStore(
        worker_authority,
        registry=registry,
        retry_policy=outbox_retry_policy,
        runtime_observer=runtime_observer,
    )
    logical_nodes = PostgresLogicalNodeRegistry(
        worker_authority,
        runtime_observer=runtime_observer,
    )
    review_queue = PostgresReviewQueue(
        worker_authority,
        registry=registry,
        runtime_observer=runtime_observer,
    )
    read_model = PostgresCanonicalReadModel(application_authority)
    try:
        verify_completion_evidence_schema(worker_authority)
        r2_artifacts.verify_startup()
        logical_nodes.verify_startup()
        review_queue.verify_startup()
        read_model.verify_startup()
        evidence.verify_integrity()
    except Exception as error:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.MIGRATIONS_UNVERIFIED,
            "PostgreSQL canonical adapters could not verify their migrated state",
        ) from error

    try:
        primary_adapter = primary_adapter_factory(evidence)
    except Exception as error:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.PRIMARY_ADAPTER_MISMATCH,
            "could not construct the pinned primary RunPod adapter",
        ) from error
    _verify_primary_adapter(contract.primary_runpod, primary_adapter, evidence)

    return ProductionCanonicalRuntime(
        contract=contract,
        tenant=tenant,
        application_authority=application_authority,
        worker_authority=worker_authority,
        work_scheduler=work_scheduler,
        stream_work_ledger=stream_work_ledger,
        capture_authority=capture,
        primary_completion=primary_completion,
        inference_evidence=evidence,
        r2_artifact_authority=r2_artifacts,
        barrier_storage=barriers,
        outbox_delivery=outbox_delivery,
        logical_node_registry=logical_nodes,
        review_queue=review_queue,
        read_model=read_model,
        r2_object_store=r2_object_store,
        primary_adapter=primary_adapter,
        primary_route=primary_route,
        pgvector_projection=pgvector_projection,
    )


def _verify_static_contract(
    contract: ProductionCompositionContract,
    r2_object_store: R2ObjectStore,
) -> None:
    postgres = contract.canonical_postgres
    if postgres.schema_name != _CANONICAL_SCHEMA:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.INVALID_CONFIGURATION,
            f"canonical schema must be {_CANONICAL_SCHEMA!r} for the reviewed migration set",
        )
    if postgres.tenant_context_setting != _TENANT_SETTING:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.INVALID_CONFIGURATION,
            f"tenant_context_setting must be {_TENANT_SETTING!r} for the reviewed RLS policies",
        )
    if r2_object_store.config != contract.r2:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.INVALID_DEPENDENCY,
            "R2 object store configuration does not match the production contract",
        )
    if r2_object_store.config.allow_delete:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.INVALID_CONFIGURATION,
            "production R2 configuration must disable application-level deletion",
        )


def _verify_primary_route(
    binding: ProductionPrimaryRunPodBinding,
    route: ProductionRoute,
    release_verifier: object,
) -> None:
    if not isinstance(route, ProductionRoute):
        raise TypeError("primary_route must be ProductionRoute")
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
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.PRIMARY_ROUTE_MISMATCH,
            "ProductionRoute deployment does not match the pinned primary RunPod binding",
        )
    verify_primary = getattr(release_verifier, "verify_primary", None)
    if not callable(verify_primary):
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.PRIMARY_ROUTE_RELEASE_NOT_VERIFIED,
            "production composition requires an independent primary-route release verifier",
        )
    try:
        verified = verify_primary(
            authorization=route.authorization,
            deployment=route.deployment,
        )
    except Exception as error:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.PRIMARY_ROUTE_RELEASE_NOT_VERIFIED,
            "primary-route release verification raised an error",
        ) from error
    if verified is not True:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.PRIMARY_ROUTE_RELEASE_NOT_VERIFIED,
            "primary-route release evidence was not verified",
        )


def _verify_projection(projection: object) -> None:
    verify_backend = getattr(projection, "verify_backend", None)
    drain = getattr(projection, "drain", None)
    if not callable(verify_backend) or not callable(drain):
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.INVALID_DEPENDENCY,
            "production composition requires a PgVectorProjectionStore with verification methods",
        )
    try:
        verify_backend()
        drain(limit=0)
    except Exception as error:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.DERIVED_PROJECTION_UNVERIFIED,
            "pgvector derived projection or its worker role is not verified",
        ) from error


def _verify_canonical_authority(
    authority: PostgresCanonicalAuthority,
    *,
    runtime_role: Literal["application", "worker"],
) -> None:
    """Prove a non-owner runtime role sees canonical state through active RLS."""

    required_privileges = (
        ("can_select",)
        if runtime_role == "application"
        else ("can_select", "can_insert", "can_update")
    )
    forbidden_privileges = (
        ("can_insert", "can_update", "can_delete", "can_truncate")
        if runtime_role == "application"
        else ("can_delete", "can_truncate")
    )

    def operation(connection: object) -> None:
        execute = getattr(connection, "execute", None)
        if not callable(execute):
            raise TypeError("PostgreSQL authority transaction did not supply a connection")
        role = execute(
            """
            SELECT
                rolname AS role_name,
                rolsuper AS is_superuser,
                rolbypassrls AS bypasses_rls
            FROM pg_roles
            WHERE rolname = current_user
            """
        ).fetchone()
        if (
            role is None
            or not isinstance(role.get("role_name"), str)
            or not role["role_name"]
            or role.get("is_superuser") is not False
            or role.get("bypasses_rls") is not False
        ):
            raise RuntimeError("canonical runtime role must be a non-superuser with NOBYPASSRLS")
        role_name = role["role_name"]
        operations = execute(
            """
            SELECT
                has_table_privilege(current_user, 'robata_ops.schema_migrations', 'SELECT')
                    AS can_read_migrations,
                has_table_privilege(
                    current_user,
                    'robata_ops.canonical_authority_state',
                    'SELECT'
                ) AS can_read_authority_state
            """
        ).fetchone()
        if (
            operations is None
            or operations.get("can_read_migrations") is not True
            or operations.get("can_read_authority_state") is not True
        ):
            raise RuntimeError("canonical runtime role lacks required robata_ops read grants")
        state = execute(
            """
            SELECT required_tenant_setting
            FROM robata_ops.canonical_authority_state
            WHERE singleton = true
            """
        ).fetchone()
        if state is None or state.get("required_tenant_setting") != _TENANT_SETTING:
            raise RuntimeError(
                "canonical authority state is absent or has an unexpected tenant setting"
            )
        tenant = execute(
            "SELECT current_setting(%s, true) AS tenant_id",
            (_TENANT_SETTING,),
        ).fetchone()
        if (
            tenant is None
            or not isinstance(tenant.get("tenant_id"), str)
            or not tenant["tenant_id"]
        ):
            raise RuntimeError("canonical PostgreSQL transaction has no tenant context")
        rows = execute(
            """
            SELECT
                c.relname AS table_name,
                c.relrowsecurity AS rls_enabled,
                c.relforcerowsecurity AS rls_forced,
                row_security_active(c.oid::regclass) AS rls_active_for_role,
                pg_get_userbyid(c.relowner) AS table_owner,
                has_table_privilege(current_user, c.oid, 'SELECT') AS can_select,
                has_table_privilege(current_user, c.oid, 'INSERT') AS can_insert,
                has_table_privilege(current_user, c.oid, 'UPDATE') AS can_update,
                has_table_privilege(current_user, c.oid, 'DELETE') AS can_delete,
                has_table_privilege(current_user, c.oid, 'TRUNCATE') AS can_truncate
            FROM pg_class AS c
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
              AND c.relkind = 'r'
              AND c.relname = ANY(%s)
            """,
            (_CANONICAL_SCHEMA, list(_REQUIRED_CANONICAL_TABLES)),
        ).fetchall()
        by_table = {str(row["table_name"]): row for row in rows}
        missing = sorted(set(_REQUIRED_CANONICAL_TABLES).difference(by_table))
        if missing:
            raise RuntimeError("canonical tables are missing: " + ", ".join(missing))
        insecure = sorted(
            table
            for table, row in by_table.items()
            if (
                row.get("rls_enabled") is not True
                or row.get("rls_forced") is not True
                or row.get("rls_active_for_role") is not True
            )
        )
        if insecure:
            raise RuntimeError(
                "canonical tables do not enforce active FORCE RLS for this role: "
                + ", ".join(insecure)
            )
        owned = sorted(
            table for table, row in by_table.items() if row.get("table_owner") == role_name
        )
        if owned:
            raise RuntimeError(
                "canonical runtime role must not own canonical tables: " + ", ".join(owned)
            )
        missing_privileges = sorted(
            f"{table}:{privilege.removeprefix('can_').upper()}"
            for table, row in by_table.items()
            for privilege in required_privileges
            if row.get(privilege) is not True
        )
        if missing_privileges:
            raise RuntimeError(
                "canonical runtime role lacks required table privileges: "
                + ", ".join(missing_privileges)
            )
        excessive_privileges = sorted(
            f"{table}:{privilege.removeprefix('can_').upper()}"
            for table, row in by_table.items()
            for privilege in forbidden_privileges
            if row.get(privilege) is True
        )
        if excessive_privileges:
            raise RuntimeError(
                "canonical runtime role has forbidden table privileges: "
                + ", ".join(excessive_privileges)
            )
        policies = execute(
            """
            SELECT tablename, qual, with_check
            FROM pg_policies
            WHERE schemaname = %s AND policyname = ANY(%s)
            """,
            (
                _CANONICAL_SCHEMA,
                [f"{table}_tenant_isolation" for table in _REQUIRED_CANONICAL_TABLES],
            ),
        ).fetchall()
        protected = {str(row["tablename"]) for row in policies}
        missing_policies = sorted(set(_REQUIRED_CANONICAL_TABLES).difference(protected))
        if missing_policies:
            raise RuntimeError(
                "canonical tenant policies are missing: " + ", ".join(missing_policies)
            )
        malformed_policies = sorted(
            str(row["tablename"])
            for row in policies
            if not all(
                isinstance(row.get(column), str)
                and "tenant_id" in row[column]
                and "current_tenant_id" in row[column]
                for column in ("qual", "with_check")
            )
        )
        if malformed_policies:
            raise RuntimeError(
                "canonical tenant policies do not bind tenant context: "
                + ", ".join(malformed_policies)
            )
        visible = execute(
            "SELECT COUNT(*) AS visible_row_count FROM robata_canonical.work_items"
        ).fetchone()
        if visible is None or not isinstance(visible.get("visible_row_count"), int):
            raise RuntimeError("canonical RLS query did not return a valid work-item count")

    try:
        authority.run_authority_transaction(
            write=False,
            operation_name=(f"production_runtime.verify_{runtime_role}_canonical_authority"),
            operation=cast(Callable[[object], None], operation),
        )
    except Exception as error:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.RLS_UNVERIFIED,
            f"canonical PostgreSQL {runtime_role} authority/RLS verification failed",
        ) from error


def _verify_primary_adapter(
    binding: ProductionPrimaryRunPodBinding,
    adapter: object,
    evidence: PostgresInferenceEvidenceLedger,
) -> None:
    if not isinstance(adapter, RunPodVisionAdapter):
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.PRIMARY_ADAPTER_MISMATCH,
            "primary adapter must be the real RunPodVisionAdapter, never OfflineFixture",
        )
    if adapter.config != binding.endpoint:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.PRIMARY_ADAPTER_MISMATCH,
            "primary RunPod adapter endpoint differs from the pinned production binding",
        )
    if adapter.capabilities_snapshot.snapshot_digest != binding.capability_snapshot_sha256:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.PRIMARY_ADAPTER_MISMATCH,
            "primary RunPod capability snapshot differs from the pinned production binding",
        )
    if adapter.raw_store is not evidence:
        raise ProductionRuntimeError(
            ProductionRuntimeErrorCode.PRIMARY_ADAPTER_MISMATCH,
            "primary RunPod adapter must write raw bytes into the PostgreSQL evidence ledger",
        )


def _repository_migrations_directory() -> Path:
    return Path(__file__).resolve().parents[4] / "db" / "migrations"


__all__ = [
    "CanonicalPostgresCredentials",
    "CanonicalPostgresRuntimeCredentials",
    "PrimaryAdapterFactory",
    "PrimaryRouteReleaseVerifier",
    "ProductionCanonicalRuntime",
    "ProductionCaptureAuthorityBinding",
    "ProductionRuntimeError",
    "ProductionRuntimeErrorCode",
    "ProductionTenantContext",
    "build_production_canonical_runtime",
    "create_canonical_postgres_connection_factory",
]
