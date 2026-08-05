"""Validate explicitly configured R2, pgvector, and RunPod adapters before composition."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.adapters.pgvector_runtime import (  # noqa: E402
    PgVectorCredentials,
    PgVectorRuntimeConfig,
    create_pgvector_projection_store,
    create_verified_pgvector_projection_store,
)
from robata.adapters.r2_object_store import (  # noqa: E402
    R2Credentials,
    R2ObjectStoreConfig,
    create_boto3_r2_client,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256  # noqa: E402
from robata.inference.runpod import (  # noqa: E402
    RunPodApiKey,
    RunPodDeploymentConfiguration,
    RunPodEndpointConfig,
)
from robata.ports.object_storage import ObjectStoreError  # noqa: E402
from robata.ports.vector_projection import VectorProjectionError  # noqa: E402

_SECRET_ENVIRONMENT_KEYS = (
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "PGVECTOR_PASSWORD",
    "PGVECTOR_WORKER_PASSWORD",
    "RUNPOD_API_KEY",
    "RUNPOD_CONTROL_API_KEY",
    "RUNPOD_CANDIDATE_API_KEY",
)
_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUNPOD_API_KEY_ENVIRONMENT_KEY = re.compile(r"^RUNPOD(?:_[A-Z0-9]+)*_API_KEY$")
_RUNPOD_ROLES: tuple[tuple[str, str], ...] = (
    ("control", "RUNPOD_CONTROL_"),
    ("candidate", "RUNPOD_CANDIDATE_"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2", action="store_true", help="validate and construct the R2 S3 client")
    parser.add_argument(
        "--pgvector", action="store_true", help="validate and construct the pgvector store"
    )
    parser.add_argument(
        "--runpod",
        action="store_true",
        help=(
            "validate two pinned RunPod model endpoint configurations without sending HTTP requests"
        ),
    )
    parser.add_argument(
        "--verify-pgvector",
        action="store_true",
        help="connect to pgvector and verify the reviewed backend/RLS deployment",
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        metavar="PATH",
        help="read explicit local KEY=VALUE configuration without loading it into the process",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run selected checks against one explicit environment mapping.

    R2 construction checks configuration and the optional SDK only; it does not
    write, read, or reconcile an object. pgvector remains lazy unless
    --verify-pgvector is selected, which performs target database I/O. RunPod
    configuration preflight is always offline and does not construct a transport.
    """

    args = _parser().parse_args(argv)
    base_values = os.environ if environment is None else environment
    values: Mapping[str, str] = base_values
    try:
        values = _load_environment_file(base_values, args.env_file)
        if args.verify_pgvector and not args.pgvector:
            return _write_failure(
                "INVALID_REQUEST", "--verify-pgvector requires --pgvector", values
            )
        if not args.r2 and not args.pgvector and not args.runpod:
            return _write_failure(
                "INVALID_REQUEST", "select at least one of --r2, --pgvector, or --runpod", values
            )
        checks = _run_checks(
            values,
            include_r2=args.r2,
            include_pgvector=args.pgvector,
            include_runpod=args.runpod,
            verify_pgvector=args.verify_pgvector,
        )
    except (ObjectStoreError, VectorProjectionError) as error:
        return _write_failure(error.code.value, str(error), values)
    except (OSError, TypeError, UnicodeError, ValueError) as error:
        return _write_failure("INVALID_CONFIGURATION", str(error), values)
    except Exception:
        return _write_failure(
            "PREFLIGHT_FAILED", "optional adapter preflight raised an unexpected error", values
        )
    _write_json(
        {
            "ok": True,
            "checks": checks,
            "qualification_status": "NOT_MEASURED",
            "production_eligible": False,
        }
    )
    return 0


def _load_environment_file(
    base_values: Mapping[str, str], environment_file: Path | None
) -> dict[str, str]:
    """Merge a deliberately selected local env file without mutating process state."""

    values = dict(base_values)
    if environment_file is None:
        return values
    for line_number, raw_line in enumerate(
        environment_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, raw_value = raw_line.partition("=")
        key = key.strip()
        if not separator or _ENVIRONMENT_KEY.fullmatch(key) is None:
            raise ValueError(
                f"environment file has an invalid KEY=VALUE assignment at line {line_number}"
            )
        values[key] = _parse_environment_value(raw_value.strip(), line_number)
    return values


def _parse_environment_value(value: str, line_number: int) -> str:
    if not value:
        return value
    if value[0] not in {"'", '"'}:
        return value
    if len(value) < 2 or value[-1] != value[0]:
        raise ValueError(f"environment file has an unclosed quoted value at line {line_number}")
    return value[1:-1]


def _run_checks(
    environment: Mapping[str, str],
    *,
    include_r2: bool,
    include_pgvector: bool,
    include_runpod: bool,
    verify_pgvector: bool,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if include_r2:
        r2_config = R2ObjectStoreConfig.from_environment(environment)
        create_boto3_r2_client(r2_config, R2Credentials.from_environment(environment))
        checks.append(
            {
                "adapter": "r2",
                "state": "CONFIGURED",
                "endpoint_url": r2_config.endpoint_url,
                "bucket": r2_config.bucket,
                "prefix": r2_config.normalized_prefix,
            }
        )
    if include_pgvector:
        pgvector_config = PgVectorRuntimeConfig.from_environment(environment)
        primary_credentials = PgVectorCredentials.from_environment(
            environment, variable="PGVECTOR_PASSWORD"
        )
        worker_credentials = PgVectorCredentials.from_environment(
            environment, variable="PGVECTOR_WORKER_PASSWORD"
        )
        if verify_pgvector:
            create_verified_pgvector_projection_store(
                pgvector_config,
                primary_credentials,
                worker_credentials=worker_credentials,
            )
            state = "VERIFIED"
        else:
            create_pgvector_projection_store(
                pgvector_config,
                primary_credentials,
                worker_credentials=worker_credentials,
            )
            state = "CONFIGURED"
        checks.append(
            {
                "adapter": "pgvector",
                "state": state,
                "backend": pgvector_config.backend.value,
                "relation": pgvector_config.relation,
                "dimension": pgvector_config.dimension,
                "require_rls": True,
            }
        )
    if include_runpod:
        checks.append(_runpod_check(environment))
    return checks


def _runpod_check(environment: Mapping[str, str]) -> dict[str, object]:
    """Validate two named RunPod deployments without instantiating a transport."""

    endpoints = tuple(
        _runpod_endpoint_configuration(environment, role=role, prefix=prefix)
        for role, prefix in _RUNPOD_ROLES
    )
    endpoint_urls = tuple(config.endpoint_url for _, config, _ in endpoints)
    if len(set(endpoint_urls)) != len(endpoint_urls):
        raise ValueError("RunPod control and candidate endpoint URLs must differ")
    model_pins: list[tuple[str, str]] = []
    for _, config, _ in endpoints:
        deployment = config.deployment_configuration
        if deployment is None:
            raise RuntimeError("RunPod endpoint configuration lacks a deployment pin")
        model_pins.append((deployment.model_identifier, deployment.model_version))
    if len(set(model_pins)) != len(model_pins):
        raise ValueError("RunPod control and candidate deployment model pins must differ")
    rendered_endpoints: list[dict[str, object]] = []
    for role, config, deployment_facts in endpoints:
        endpoint_configuration = config.model_dump(mode="json")
        pinned_configuration = {
            "endpoint_configuration": endpoint_configuration,
            "deployment_facts": deployment_facts,
        }
        rendered_endpoints.append(
            {
                "role": role,
                "endpoint_configuration": endpoint_configuration,
                "endpoint_configuration_sha256": exact_bytes_sha256(
                    canonical_json_bytes(endpoint_configuration)
                ),
                "deployment_facts": deployment_facts,
                "configuration_sha256": exact_bytes_sha256(
                    canonical_json_bytes(pinned_configuration)
                ),
            }
        )
    return {
        "adapter": "runpod",
        "state": "CONFIGURED",
        "offline": True,
        "endpoints": rendered_endpoints,
    }


def _runpod_endpoint_configuration(
    environment: Mapping[str, str], *, role: str, prefix: str
) -> tuple[str, RunPodEndpointConfig, dict[str, str]]:
    """Build one non-secret RunPod endpoint configuration from named environment values."""

    _ = RunPodApiKey(_runpod_api_key(environment, prefix=prefix))
    deployment_facts = {
        "handler_image": _environment_required(environment, f"{prefix}HANDLER_IMAGE"),
        "handler_image_sha256": _environment_sha256(environment, f"{prefix}HANDLER_IMAGE_SHA256"),
        "capability_snapshot_sha256": _environment_sha256(
            environment, f"{prefix}CAPABILITY_SNAPSHOT_SHA256"
        ),
    }
    deployment = RunPodDeploymentConfiguration(
        model_identifier=_environment_required(environment, f"{prefix}MODEL_IDENTIFIER"),
        model_version=_environment_required(environment, f"{prefix}MODEL_VERSION"),
        inference_engine=_environment_required(environment, f"{prefix}INFERENCE_ENGINE"),
        precision_or_quantization=_environment_required(
            environment, f"{prefix}PRECISION_OR_QUANTIZATION"
        ),
        topology=_environment_required(environment, f"{prefix}TOPOLOGY"),
        max_output_tokens=_environment_positive_int(
            environment, f"{prefix}MAX_OUTPUT_TOKENS", default=None
        ),
        supported_topologies=_runpod_supported_topologies(environment, prefix=prefix),
    )
    return (
        role,
        RunPodEndpointConfig(
            provider="runpod",
            deployment_configuration=deployment,
            endpoint_url=_environment_required(environment, f"{prefix}ENDPOINT_URL"),
            adapter_version=_environment_required(environment, f"{prefix}ADAPTER_VERSION"),
            native_batch_enabled=_environment_boolean(
                environment, f"{prefix}NATIVE_BATCH_ENABLED", default=False
            ),
            native_batch_max_size=_environment_positive_int(
                environment, f"{prefix}NATIVE_BATCH_MAX_SIZE", default=1
            ),
            max_concurrent_requests=_environment_positive_int(
                environment, f"{prefix}MAX_CONCURRENT_REQUESTS", default=None
            ),
            request_timeout_cap_ms=_environment_positive_int(
                environment, f"{prefix}REQUEST_TIMEOUT_CAP_MS", default=None
            ),
            max_response_bytes=_environment_positive_int(
                environment, f"{prefix}MAX_RESPONSE_BYTES", default=None
            ),
        ),
        deployment_facts,
    )


def _runpod_api_key(environment: Mapping[str, str], *, prefix: str) -> str:
    """Return a role-specific key when present, otherwise the shared RunPod key."""

    role_key = f"{prefix}API_KEY"
    value = environment.get(role_key, environment.get("RUNPOD_API_KEY"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role_key} or RUNPOD_API_KEY must be configured")
    return value


def _runpod_supported_topologies(environment: Mapping[str, str], *, prefix: str) -> tuple[str, ...]:
    value = environment.get(f"{prefix}SUPPORTED_TOPOLOGIES")
    if value is None:
        return ()
    if not isinstance(value, str) or not value:
        raise ValueError(f"{prefix}SUPPORTED_TOPOLOGIES must be non-empty when configured")
    raw_topologies = value.split(",")
    topologies = tuple(item.strip() for item in raw_topologies)
    if not all(topologies) or any(item != item.strip() for item in raw_topologies):
        raise ValueError(f"{prefix}SUPPORTED_TOPOLOGIES must be comma-separated without whitespace")
    return topologies


def _environment_required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be configured")
    return value


def _environment_sha256(environment: Mapping[str, str], name: str) -> str:
    value = _environment_required(environment, name)
    if _SHA256_DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _environment_positive_int(
    environment: Mapping[str, str], name: str, *, default: int | None
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


def _environment_boolean(environment: Mapping[str, str], name: str, *, default: bool) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{name} must be true or false")


def _write_failure(code: str, detail: str, environment: Mapping[str, str]) -> int:
    _write_json(
        {"ok": False, "code": code, "detail": _redact(detail, environment)},
        stream=sys.stderr,
    )
    return 2


def _redact(detail: str, environment: Mapping[str, str]) -> str:
    redacted = detail
    for key, value in environment.items():
        if (
            (key in _SECRET_ENVIRONMENT_KEYS or _RUNPOD_API_KEY_ENVIRONMENT_KEY.fullmatch(key))
            and isinstance(value, str)
            and value
        ):
            redacted = redacted.replace(value, "REDACTED")
    return redacted


def _write_json(payload: object, *, stream: TextIO | None = None) -> None:
    print(
        canonical_json_bytes(payload).decode("utf-8"), file=sys.stdout if stream is None else stream
    )


if __name__ == "__main__":
    raise SystemExit(main())
