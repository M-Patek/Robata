"""Tests for the opt-in local configuration preflight command."""

from __future__ import annotations

import json
from collections.abc import Mapping

from scripts import preflight_optional_adapters as subject


def _environment() -> dict[str, str]:
    return {
        "R2_ENDPOINT_URL": "https://account-id.r2.cloudflarestorage.com",
        "R2_BUCKET": "robata-production",
        "R2_PREFIX": "artifacts",
        "R2_ACCESS_KEY_ID": "access-secret",
        "R2_SECRET_ACCESS_KEY": "r2-super-secret",
        "PGVECTOR_HOST": "db.example.test",
        "PGVECTOR_DATABASE": "robata",
        "PGVECTOR_USER": "robata_app",
        "PGVECTOR_PASSWORD": "primary-super-secret",
        "PGVECTOR_SSLROOTCERT": "/etc/ssl/certs/robata-ca.pem",
        "PGVECTOR_WORKER_HOST": "db.example.test",
        "PGVECTOR_WORKER_DATABASE": "robata",
        "PGVECTOR_WORKER_USER": "robata_worker",
        "PGVECTOR_WORKER_PASSWORD": "worker-super-secret",
        "PGVECTOR_WORKER_SSLROOTCERT": "/etc/ssl/certs/robata-ca.pem",
        "PGVECTOR_WORKER_ROLE": "robata_vector_worker",
        "PGVECTOR_DIMENSION": "3",
        "PGVECTOR_BACKEND": "POSTGRES",
    }


def test_configured_preflight_is_lazy_and_never_prints_credentials(monkeypatch, capsys) -> None:
    r2_calls: list[object] = []
    pgvector_calls: list[tuple[object, object, object]] = []
    monkeypatch.setattr(
        subject,
        "create_boto3_r2_client",
        lambda config, credentials: r2_calls.append((config, credentials)),
    )
    monkeypatch.setattr(
        subject,
        "create_pgvector_projection_store",
        lambda config, primary, *, worker_credentials: pgvector_calls.append(
            (config, primary, worker_credentials)
        ),
    )

    status = subject.main(["--r2", "--pgvector"], environment=_environment())

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert payload == {
        "checks": [
            {
                "adapter": "r2",
                "bucket": "robata-production",
                "endpoint_url": "https://account-id.r2.cloudflarestorage.com",
                "prefix": "artifacts/",
                "state": "CONFIGURED",
            },
            {
                "adapter": "pgvector",
                "backend": "POSTGRES",
                "dimension": 3,
                "relation": "public.robata_vector_projection",
                "require_rls": True,
                "state": "CONFIGURED",
            },
        ],
        "ok": True,
        "production_eligible": False,
        "qualification_status": "NOT_MEASURED",
    }
    assert len(r2_calls) == 1
    assert len(pgvector_calls) == 1
    for secret in (
        "access-secret",
        "r2-super-secret",
        "primary-super-secret",
        "worker-super-secret",
    ):
        assert secret not in captured.out


def test_preflight_reads_only_an_explicit_local_environment_file(
    monkeypatch, capsys, tmp_path
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        subject,
        "create_boto3_r2_client",
        lambda config, credentials: calls.append((config, credentials)),
    )
    environment_file = tmp_path / "robata-local.env"
    environment_file.write_text(
        "\n".join(
            (
                "R2_ENDPOINT_URL=https://account-id.r2.cloudflarestorage.com",
                "R2_BUCKET=robata-production",
                "R2_PREFIX=artifacts",
                'R2_ACCESS_KEY_ID="access-secret"',
                'R2_SECRET_ACCESS_KEY="r2-super-secret"',
            )
        ),
        encoding="utf-8",
    )
    base_values = {"UNRELATED": "unchanged"}

    status = subject.main(["--r2", "--env-file", str(environment_file)], environment=base_values)

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["checks"][0]["adapter"] == "r2"
    assert len(calls) == 1
    assert base_values == {"UNRELATED": "unchanged"}
    assert "access-secret" not in json.dumps(payload)
    assert "r2-super-secret" not in json.dumps(payload)


def test_verified_pgvector_preflight_requires_an_explicit_target_check(monkeypatch, capsys) -> None:
    calls: list[tuple[object, object, object]] = []
    monkeypatch.setattr(
        subject,
        "create_verified_pgvector_projection_store",
        lambda config, primary, *, worker_credentials: calls.append(
            (config, primary, worker_credentials)
        ),
    )

    status = subject.main(["--pgvector", "--verify-pgvector"], environment=_environment())

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["checks"] == [
        {
            "adapter": "pgvector",
            "backend": "POSTGRES",
            "dimension": 3,
            "relation": "public.robata_vector_projection",
            "require_rls": True,
            "state": "VERIFIED",
        }
    ]
    assert len(calls) == 1


def test_preflight_rejects_invalid_selection_and_redacts_failure_details(capsys) -> None:
    values: Mapping[str, str] = {"R2_SECRET_ACCESS_KEY": "r2-super-secret"}

    status = subject.main(["--verify-pgvector"], environment=values)

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert status == 2
    assert captured.out == ""
    assert payload == {
        "code": "INVALID_REQUEST",
        "detail": "--verify-pgvector requires --pgvector",
        "ok": False,
    }
    assert "r2-super-secret" not in captured.err


def test_preflight_redacts_secret_appearing_in_a_configuration_error(capsys) -> None:
    values = {"R2_SECRET_ACCESS_KEY": "r2-super-secret"}

    status = subject.main(["--r2"], environment=values)

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert status == 2
    assert payload["code"] == "INVALID_CONFIGURATION"
    assert "r2-super-secret" not in payload["detail"]
