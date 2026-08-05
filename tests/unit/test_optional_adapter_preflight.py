"""Tests for the opt-in local configuration preflight command."""

from __future__ import annotations

import json
from collections.abc import Mapping

import robata.inference.runpod as runpod_module
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
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


def _runpod_environment() -> dict[str, str]:
    return {
        "RUNPOD_API_KEY": "shared-runpod-api-key-secret-12345",
        "RUNPOD_CONTROL_ENDPOINT_URL": "https://api.runpod.test/v2/mage-4b/runsync",
        "RUNPOD_CONTROL_MODEL_IDENTIFIER": "mage-vl-4b",
        "RUNPOD_CONTROL_MODEL_VERSION": "1.0",
        "RUNPOD_CONTROL_HANDLER_IMAGE": (
            "registry.example.test/robata/mage-handler@sha256:" + "a" * 64
        ),
        "RUNPOD_CONTROL_HANDLER_IMAGE_SHA256": "a" * 64,
        "RUNPOD_CONTROL_CAPABILITY_SNAPSHOT_SHA256": "b" * 64,
        "RUNPOD_CONTROL_INFERENCE_ENGINE": "vllm",
        "RUNPOD_CONTROL_PRECISION_OR_QUANTIZATION": "bf16",
        "RUNPOD_CONTROL_TOPOLOGY": "TWO_SINGLE_CARD_REPLICAS",
        "RUNPOD_CONTROL_MAX_OUTPUT_TOKENS": "1024",
        "RUNPOD_CONTROL_ADAPTER_VERSION": "runpod-adapter-v1",
        "RUNPOD_CONTROL_NATIVE_BATCH_ENABLED": "false",
        "RUNPOD_CONTROL_NATIVE_BATCH_MAX_SIZE": "1",
        "RUNPOD_CONTROL_MAX_CONCURRENT_REQUESTS": "2",
        "RUNPOD_CONTROL_REQUEST_TIMEOUT_CAP_MS": "120000",
        "RUNPOD_CONTROL_MAX_RESPONSE_BYTES": "4194304",
        "RUNPOD_CONTROL_SUPPORTED_TOPOLOGIES": "TWO_SINGLE_CARD_REPLICAS",
        "RUNPOD_CANDIDATE_API_KEY": "candidate-runpod-api-key-secret-67890",
        "RUNPOD_CANDIDATE_ENDPOINT_URL": "https://api.runpod.test/v2/qwen-4b/runsync",
        "RUNPOD_CANDIDATE_MODEL_IDENTIFIER": "qwen3-vl-4b",
        "RUNPOD_CANDIDATE_MODEL_VERSION": "1.0",
        "RUNPOD_CANDIDATE_HANDLER_IMAGE": (
            "registry.example.test/robata/qwen-handler@sha256:" + "c" * 64
        ),
        "RUNPOD_CANDIDATE_HANDLER_IMAGE_SHA256": "c" * 64,
        "RUNPOD_CANDIDATE_CAPABILITY_SNAPSHOT_SHA256": "d" * 64,
        "RUNPOD_CANDIDATE_INFERENCE_ENGINE": "vllm",
        "RUNPOD_CANDIDATE_PRECISION_OR_QUANTIZATION": "int4",
        "RUNPOD_CANDIDATE_TOPOLOGY": "TWO_CARD_TENSOR_PARALLEL",
        "RUNPOD_CANDIDATE_MAX_OUTPUT_TOKENS": "2048",
        "RUNPOD_CANDIDATE_ADAPTER_VERSION": "runpod-adapter-v1",
        "RUNPOD_CANDIDATE_NATIVE_BATCH_ENABLED": "true",
        "RUNPOD_CANDIDATE_NATIVE_BATCH_MAX_SIZE": "4",
        "RUNPOD_CANDIDATE_MAX_CONCURRENT_REQUESTS": "4",
        "RUNPOD_CANDIDATE_REQUEST_TIMEOUT_CAP_MS": "90000",
        "RUNPOD_CANDIDATE_MAX_RESPONSE_BYTES": "8388608",
        "RUNPOD_CANDIDATE_SUPPORTED_TOPOLOGIES": (
            "TWO_SINGLE_CARD_REPLICAS,TWO_CARD_TENSOR_PARALLEL"
        ),
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


def test_runpod_preflight_pins_two_configurations_without_network_io(monkeypatch, capsys) -> None:
    def unexpected_network(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("offline RunPod preflight must not open HTTP")

    monkeypatch.setattr(runpod_module.urllib_request, "urlopen", unexpected_network)
    values = _runpod_environment()

    status = subject.main(["--runpod"], environment=values)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["qualification_status"] == "NOT_MEASURED"
    assert payload["production_eligible"] is False
    assert len(payload["checks"]) == 1
    check = payload["checks"][0]
    assert check["adapter"] == "runpod"
    assert check["state"] == "CONFIGURED"
    assert check["offline"] is True
    endpoints = check["endpoints"]
    assert [endpoint["role"] for endpoint in endpoints] == ["control", "candidate"]
    assert [endpoint["endpoint_configuration"]["endpoint_url"] for endpoint in endpoints] == [
        "https://api.runpod.test/v2/mage-4b/runsync",
        "https://api.runpod.test/v2/qwen-4b/runsync",
    ]
    assert endpoints[0]["endpoint_configuration"]["deployment_configuration"] == {
        "inference_engine": "vllm",
        "max_output_tokens": 1024,
        "model_identifier": "mage-vl-4b",
        "model_version": "1.0",
        "precision_or_quantization": "bf16",
        "supported_topologies": ["TWO_SINGLE_CARD_REPLICAS"],
        "topology": "TWO_SINGLE_CARD_REPLICAS",
    }
    assert endpoints[1]["deployment_facts"] == {
        "handler_image": "registry.example.test/robata/qwen-handler@sha256:" + "c" * 64,
        "handler_image_sha256": "c" * 64,
        "capability_snapshot_sha256": "d" * 64,
    }
    for endpoint in endpoints:
        configuration = endpoint["endpoint_configuration"]
        deployment_facts = endpoint["deployment_facts"]
        assert endpoint["endpoint_configuration_sha256"] == exact_bytes_sha256(
            canonical_json_bytes(configuration)
        )
        assert endpoint["configuration_sha256"] == exact_bytes_sha256(
            canonical_json_bytes(
                {
                    "endpoint_configuration": configuration,
                    "deployment_facts": deployment_facts,
                }
            )
        )
    captured_text = captured.out + captured.err
    assert values["RUNPOD_API_KEY"] not in captured_text
    assert values["RUNPOD_CANDIDATE_API_KEY"] not in captured_text


def test_runpod_preflight_rejects_duplicate_endpoint_and_redacts_credentials(capsys) -> None:
    values = _runpod_environment()
    values["RUNPOD_CANDIDATE_ENDPOINT_URL"] = values["RUNPOD_CONTROL_ENDPOINT_URL"]

    status = subject.main(["--runpod"], environment=values)

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert status == 2
    assert captured.out == ""
    assert payload == {
        "code": "INVALID_CONFIGURATION",
        "detail": "RunPod control and candidate endpoint URLs must differ",
        "ok": False,
    }
    assert values["RUNPOD_API_KEY"] not in captured.err
    assert values["RUNPOD_CANDIDATE_API_KEY"] not in captured.err
