"""Optional real-PostgreSQL proof for the production app/worker role boundary.

Set ``ROBATA_TEST_POSTGRES_DSN`` only to an isolated disposable database.  The
test creates short-lived non-login roles and exercises the same verifier used
by the production deployment command.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from robata.adapters.postgres_authority import (
    ConnectionFactory,
    PostgresCanonicalAuthority,
    PostgresConnection,
    psycopg_connection_factory,
)
from robata.adapters.postgres_migrations import PostgresMigrationRunner
from robata.application.canonical.production_runtime import (
    ProductionRuntimeError,
    ProductionRuntimeErrorCode,
    _verify_canonical_authority,
)

_DSN = os.environ.get("ROBATA_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not _DSN,
    reason="requires isolated ROBATA_TEST_POSTGRES_DSN",
)


@pytest.fixture(scope="session")
def postgres_factory() -> ConnectionFactory:
    assert _DSN is not None
    return psycopg_connection_factory(_DSN, application_name="robata-production-role-test")


@pytest.fixture(scope="session", autouse=True)
def migrated_postgres(postgres_factory: ConnectionFactory) -> None:
    migration_root = Path(__file__).resolve().parents[2] / "db" / "migrations"
    PostgresMigrationRunner(postgres_factory, migration_root).apply()


@pytest.fixture
def runtime_roles(
    postgres_factory: ConnectionFactory,
) -> Iterator[tuple[ConnectionFactory, ConnectionFactory]]:
    suffix = uuid4().hex
    app_role = f"robata_production_app_{suffix}"
    worker_role = f"robata_production_worker_{suffix}"
    _create_runtime_role(postgres_factory, app_role, runtime_role="application")
    _create_runtime_role(postgres_factory, worker_role, runtime_role="worker")
    try:
        yield (
            _role_factory(postgres_factory, app_role),
            _role_factory(postgres_factory, worker_role),
        )
    finally:
        _drop_role(postgres_factory, worker_role)
        _drop_role(postgres_factory, app_role)


def test_non_owner_nobypassrls_roles_pass_production_verification(
    runtime_roles: tuple[ConnectionFactory, ConnectionFactory],
) -> None:
    app_factory, worker_factory = runtime_roles

    _verify_canonical_authority(
        _authority(app_factory, tenant_id=f"app-{uuid4()}"),
        runtime_role="application",
    )
    _verify_canonical_authority(
        _authority(worker_factory, tenant_id=f"worker-{uuid4()}"),
        runtime_role="worker",
    )


def test_bypassrls_and_app_write_grants_are_rejected(
    postgres_factory: ConnectionFactory,
) -> None:
    bypass_role = f"robata_production_bypass_{uuid4().hex}"
    app_write_role = f"robata_production_app_write_{uuid4().hex}"
    _create_runtime_role(postgres_factory, bypass_role, runtime_role="application")
    _create_runtime_role(postgres_factory, app_write_role, runtime_role="application")
    administrator = postgres_factory()
    try:
        administrator.execute(f"ALTER ROLE {bypass_role} BYPASSRLS")
        administrator.execute(f"GRANT INSERT ON robata_canonical.work_items TO {app_write_role}")
    finally:
        administrator.close()
    try:
        with pytest.raises(ProductionRuntimeError) as bypass_error:
            _verify_canonical_authority(
                _authority(
                    _role_factory(postgres_factory, bypass_role), tenant_id=f"bypass-{uuid4()}"
                ),
                runtime_role="application",
            )
        assert bypass_error.value.code is ProductionRuntimeErrorCode.RLS_UNVERIFIED

        with pytest.raises(ProductionRuntimeError) as app_write_error:
            _verify_canonical_authority(
                _authority(
                    _role_factory(postgres_factory, app_write_role),
                    tenant_id=f"app-write-{uuid4()}",
                ),
                runtime_role="application",
            )
        assert app_write_error.value.code is ProductionRuntimeErrorCode.RLS_UNVERIFIED
    finally:
        _drop_role(postgres_factory, app_write_role)
        _drop_role(postgres_factory, bypass_role)


def _authority(factory: ConnectionFactory, *, tenant_id: str) -> PostgresCanonicalAuthority:
    return PostgresCanonicalAuthority(
        factory,
        tenant_setting="robata.tenant_id",
        tenant_id=tenant_id,
    )


def _create_runtime_role(
    postgres_factory: ConnectionFactory,
    role_name: str,
    *,
    runtime_role: str,
) -> None:
    administrator = postgres_factory()
    try:
        administrator.execute(f"CREATE ROLE {role_name} NOLOGIN NOSUPERUSER NOBYPASSRLS")
        administrator.execute(f"GRANT USAGE ON SCHEMA robata_canonical TO {role_name}")
        administrator.execute(f"GRANT USAGE ON SCHEMA robata_ops TO {role_name}")
        administrator.execute(
            "GRANT SELECT ON robata_ops.schema_migrations, "
            f"robata_ops.canonical_authority_state TO {role_name}"
        )
        if runtime_role == "application":
            administrator.execute(
                f"GRANT SELECT ON ALL TABLES IN SCHEMA robata_canonical TO {role_name}"
            )
        elif runtime_role == "worker":
            administrator.execute(
                "GRANT SELECT, INSERT, UPDATE ON ALL TABLES "
                f"IN SCHEMA robata_canonical TO {role_name}"
            )
        else:
            raise ValueError("runtime_role must be application or worker")
    finally:
        administrator.close()


def _role_factory(postgres_factory: ConnectionFactory, role_name: str) -> ConnectionFactory:
    def factory() -> PostgresConnection:
        connection = postgres_factory()
        connection.execute(f"SET ROLE {role_name}")
        return connection

    return factory


def _drop_role(postgres_factory: ConnectionFactory, role_name: str) -> None:
    administrator = postgres_factory()
    try:
        administrator.execute(f"DROP OWNED BY {role_name}")
        administrator.execute(f"DROP ROLE {role_name}")
    finally:
        administrator.close()
