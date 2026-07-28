"""Focused tests for explicit psycopg wiring around the pgvector port."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import pytest

from robata.adapters.pgvector_runtime import (
    PgVectorConnectionConfig,
    PgVectorCredentials,
    PgVectorRuntimeConfig,
    create_pgvector_projection_store,
    create_pgvector_projection_store_from_environment,
    create_psycopg_connection_factory,
    create_verified_pgvector_projection_store,
)
from robata.contracts.retrieval import VectorBackend
from robata.ports.vector_projection import (
    PgVectorProjectionStore,
    VectorProjectionError,
    VectorProjectionErrorCode,
)


class _Cursor:
    def __init__(
        self,
        *,
        one_results: Sequence[object | None] = (),
        all_results: Sequence[Sequence[object]] = (),
    ) -> None:
        self._one_results = list(one_results)
        self._all_results = [tuple(items) for items in all_results]
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    def execute(self, query: str, params: Sequence[object] = ()) -> None:
        self.statements.append((query, tuple(params)))

    def fetchone(self) -> object | None:
        return self._one_results.pop(0) if self._one_results else None

    def fetchall(self) -> tuple[object, ...]:
        return self._all_results.pop(0) if self._all_results else ()

    def close(self) -> None:
        self.closed = True


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> _Cursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


class _PsycopgDouble:
    def __init__(self, connections: Sequence[_Connection]) -> None:
        self._connections = list(connections)
        self.calls: list[dict[str, object]] = []

    def connect(self, **kwargs: object) -> _Connection:
        self.calls.append(dict(kwargs))
        return self._connections.pop(0)


def _connection(*, user: str, application_name: str) -> PgVectorConnectionConfig:
    return PgVectorConnectionConfig(
        host="db.example.test",
        database="robata",
        user=user,
        port=5433,
        sslmode="verify-full",
        sslrootcert="/etc/ssl/certs/robata-ca.pem",
        connect_timeout_seconds=11,
        application_name=application_name,
    )


def _runtime_config() -> PgVectorRuntimeConfig:
    return PgVectorRuntimeConfig(
        primary=_connection(user="robata_app", application_name="robata-api"),
        worker=_connection(user="robata_worker", application_name="robata-worker"),
        worker_role="robata_vector_worker",
        dimension=3,
        relation="public.event_vector_projection",
        index_name="event_vector_projection_hnsw",
        backend=VectorBackend.SUPABASE,
    )


def _environment() -> dict[str, str]:
    return {
        "PGVECTOR_HOST": "db.example.test",
        "PGVECTOR_DATABASE": "robata",
        "PGVECTOR_USER": "robata_app",
        "PGVECTOR_PORT": "5433",
        "PGVECTOR_SSLMODE": "verify-full",
        "PGVECTOR_SSLROOTCERT": "/etc/ssl/certs/robata-ca.pem",
        "PGVECTOR_CONNECT_TIMEOUT_SECONDS": "11",
        "PGVECTOR_APPLICATION_NAME": "robata-api",
        "PGVECTOR_PASSWORD": "primary-secret",
        "PGVECTOR_WORKER_HOST": "db.example.test",
        "PGVECTOR_WORKER_DATABASE": "robata",
        "PGVECTOR_WORKER_USER": "robata_worker",
        "PGVECTOR_WORKER_PORT": "5433",
        "PGVECTOR_WORKER_SSLMODE": "verify-full",
        "PGVECTOR_WORKER_SSLROOTCERT": "/etc/ssl/certs/robata-ca.pem",
        "PGVECTOR_WORKER_CONNECT_TIMEOUT_SECONDS": "12",
        "PGVECTOR_WORKER_APPLICATION_NAME": "robata-worker",
        "PGVECTOR_WORKER_PASSWORD": "worker-secret",
        "PGVECTOR_WORKER_ROLE": "robata_vector_worker",
        "PGVECTOR_DIMENSION": "3",
        "PGVECTOR_RELATION": "public.event_vector_projection",
        "PGVECTOR_INDEX_NAME": "event_vector_projection_hnsw",
        "PGVECTOR_BACKEND": "SUPABASE",
    }


def _verified_cursor(*, worker: bool) -> _Cursor:
    results: list[object | None] = [
        (True,),
        (True, True, True, 3),
        (1,),
    ]
    if worker:
        results.append((True,))
    return _Cursor(one_results=results)


def test_runtime_config_requires_tls_rls_and_an_explicit_worker_role() -> None:
    primary = _connection(user="robata_app", application_name="robata-api")
    worker = _connection(user="robata_worker", application_name="robata-worker")

    config = _runtime_config()

    assert config.require_rls is True
    assert config.backend is VectorBackend.SUPABASE
    with pytest.raises(ValueError, match="RLS-enabled"):
        PgVectorRuntimeConfig(primary=primary, dimension=3)
    with pytest.raises(ValueError, match="worker and worker_role"):
        PgVectorRuntimeConfig(primary=primary, worker=worker, dimension=3)
    with pytest.raises(ValueError):
        PgVectorRuntimeConfig(
            primary=primary,
            worker=worker,
            worker_role="robata_vector_worker",
            dimension=3,
            require_rls=False,
        )
    with pytest.raises(ValueError):
        PgVectorConnectionConfig(
            host="db.example.test",
            database="robata",
            user="robata_app",
            sslmode="require",
            sslrootcert="/etc/ssl/certs/robata-ca.pem",
        )

    certificate_path = PgVectorConnectionConfig(
        host="db.example.test",
        database="robata",
        user="robata_app",
        sslrootcert="/etc/ssl/Robata CA/root.pem",
    )
    assert certificate_path.sslrootcert == "/etc/ssl/Robata CA/root.pem"


def test_environment_config_and_credentials_are_explicit_and_redacted() -> None:
    environment = _environment()

    config = PgVectorRuntimeConfig.from_environment(environment)
    primary = PgVectorCredentials.from_environment(environment, variable="PGVECTOR_PASSWORD")
    worker = PgVectorCredentials.from_environment(environment, variable="PGVECTOR_WORKER_PASSWORD")

    assert config.primary.port == 5433
    assert config.worker is not None
    assert config.worker.connect_timeout_seconds == 12
    assert config.require_rls is True
    assert primary.password == "primary-secret"
    assert "primary-secret" not in repr(primary)
    assert "worker-secret" not in repr(worker)
    with pytest.raises(ValueError, match="cannot disable RLS"):
        PgVectorRuntimeConfig.from_environment({**environment, "PGVECTOR_REQUIRE_RLS": "false"})
    with pytest.raises(ValueError, match="PGVECTOR_DIMENSION"):
        PgVectorRuntimeConfig.from_environment(
            {key: value for key, value in environment.items() if key != "PGVECTOR_DIMENSION"}
        )
    with pytest.raises(ValueError, match="PGVECTOR_SSLROOTCERT"):
        PgVectorRuntimeConfig.from_environment(
            {key: value for key, value in environment.items() if key != "PGVECTOR_SSLROOTCERT"}
        )


def test_psycopg_factory_is_lazy_and_creates_a_fresh_tls_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Connection(_Cursor())
    second = _Connection(_Cursor())
    driver = _PsycopgDouble((first, second))
    monkeypatch.setitem(sys.modules, "psycopg", driver)
    config = _connection(user="robata_app", application_name="robata-api")

    factory = create_psycopg_connection_factory(config, PgVectorCredentials("primary-secret"))

    assert driver.calls == []
    assert factory() is first
    assert factory() is second
    assert driver.calls == [
        {
            "host": "db.example.test",
            "port": 5433,
            "dbname": "robata",
            "user": "robata_app",
            "password": "primary-secret",
            "sslmode": "verify-full",
            "sslrootcert": "/etc/ssl/certs/robata-ca.pem",
            "connect_timeout": 11,
            "application_name": "robata-api",
            "autocommit": False,
        },
        {
            "host": "db.example.test",
            "port": 5433,
            "dbname": "robata",
            "user": "robata_app",
            "password": "primary-secret",
            "sslmode": "verify-full",
            "sslrootcert": "/etc/ssl/certs/robata-ca.pem",
            "connect_timeout": 11,
            "application_name": "robata-api",
            "autocommit": False,
        },
    ]


def test_missing_optional_psycopg_dependency_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psycopg", None)

    with pytest.raises(VectorProjectionError) as unavailable:
        create_psycopg_connection_factory(
            _connection(user="robata_app", application_name="robata-api"),
            PgVectorCredentials("primary-secret"),
        )

    assert unavailable.value.code is VectorProjectionErrorCode.ADAPTER_UNAVAILABLE


def test_store_construction_stays_lazy_and_requires_separate_worker_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _PsycopgDouble(())
    monkeypatch.setitem(sys.modules, "psycopg", driver)
    config = _runtime_config()
    primary = PgVectorCredentials("primary-secret")
    worker = PgVectorCredentials("worker-secret")

    with pytest.raises(ValueError, match="worker_credentials"):
        create_pgvector_projection_store(config, primary)
    store = create_pgvector_projection_store(config, primary, worker_credentials=worker)

    assert isinstance(store, PgVectorProjectionStore)
    assert store.dimension == 3
    assert store.relation == "public.event_vector_projection"
    assert "pg_has_role" in store.rls_policy_statement()
    assert driver.calls == []


def test_verified_factory_checks_primary_rls_and_worker_role_without_claiming_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_cursor = _verified_cursor(worker=False)
    worker_cursor = _verified_cursor(worker=True)
    primary_connection = _Connection(primary_cursor)
    worker_connection = _Connection(worker_cursor)
    driver = _PsycopgDouble((primary_connection, worker_connection))
    monkeypatch.setitem(sys.modules, "psycopg", driver)

    store = create_verified_pgvector_projection_store(
        _runtime_config(),
        PgVectorCredentials("primary-secret"),
        worker_credentials=PgVectorCredentials("worker-secret"),
    )

    assert isinstance(store, PgVectorProjectionStore)
    assert primary_connection.commits == 1
    assert worker_connection.commits == 1
    assert primary_connection.closed is True
    assert worker_connection.closed is True
    assert primary_cursor.closed is True
    assert worker_cursor.closed is True
    assert any(query.startswith("SELECT pg_has_role") for query, _ in worker_cursor.statements)
    update = next(
        query for query, _ in worker_cursor.statements if query.startswith("WITH claimed")
    )
    assert "LIMIT %s" in update
    assert driver.calls[0]["user"] == "robata_app"
    assert driver.calls[1]["user"] == "robata_worker"


def test_environment_store_factory_does_not_connect_until_the_port_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _PsycopgDouble(())
    monkeypatch.setitem(sys.modules, "psycopg", driver)

    store = create_pgvector_projection_store_from_environment(_environment())

    assert isinstance(store, PgVectorProjectionStore)
    assert driver.calls == []
