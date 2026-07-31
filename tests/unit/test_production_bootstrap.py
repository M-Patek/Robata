"""Tests for the immutable production-runtime bootstrap document."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from robata.application.canonical.production_bootstrap import (
    PrimaryRouteReleaseDecision,
    ProductionBootstrapConfigurationError,
    ProductionRuntimeBootstrapConfiguration,
    load_production_runtime_bootstrap_configuration,
)
from robata.application.canonical.production_composition import ProductionPrimaryRunPodBinding
from robata.application.canonical.production_runtime import ProductionCaptureAuthorityBinding
from robata.contracts.hashing import exact_bytes_sha256
from robata.inference.models import (
    ConcurrencyClass,
    InputMode,
    ModelCapabilities,
    VisionTask,
)
from robata.inference.routing import (
    ModelDeployment,
    ProductionRoute,
    ProductionRouteAuthorization,
    endpoint_config_digest,
)
from robata.inference.runpod import (
    RunPodDeploymentConfiguration,
    RunPodEndpointConfig,
    RunPodRetryPolicy,
)


def _uuid(number: int) -> str:
    return str(UUID(int=number))


def _digest(number: int) -> str:
    return f"{number:064x}"


def _bootstrap_configuration(
    tmp_path: Path,
    *,
    release_handler_image_sha256: str | None = None,
) -> tuple[ProductionRuntimeBootstrapConfiguration, Path, Path]:
    qualification_report = tmp_path / "qualification-report.json"
    qualification_report.write_bytes(b'{"status":"PASSED"}\n')

    endpoint = RunPodEndpointConfig(
        provider="runpod",
        endpoint_url="https://api.runpod.test/v2/mage-vl-4b/runsync",
        adapter_version="runpod-adapter-v1",
        deployment_configuration=RunPodDeploymentConfiguration(
            model_identifier="mage-vl-4b",
            model_version="1.0",
            inference_engine="vllm",
            precision_or_quantization="bf16",
            topology="TWO_SINGLE_CARD_REPLICAS",
            max_output_tokens=1024,
            supported_topologies=("TWO_SINGLE_CARD_REPLICAS",),
        ),
        max_concurrent_requests=2,
    )
    capability_snapshot_digest = _digest(1)
    capabilities = ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(1),
        snapshot_digest=capability_snapshot_digest,
        provider="runpod",
        model_name="mage-vl-4b",
        model_version="1.0",
        supported_tasks=(VisionTask.ACTION_EVIDENCE,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=6,
        max_pixels_per_image=640 * 480,
        max_payload_bytes=1_000_000,
        max_input_tokens=4_096,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.LIMITED,
        data_handling_policy_version="production-data-policy-v1",
        observed_at="2026-07-30T12:00:00Z",
    )
    deployment = ModelDeployment(
        deployment_id="mage-vl-4b-primary",
        provider="runpod",
        model_name="mage-vl-4b",
        model_version="1.0",
        adapter_version=endpoint.adapter_version,
        capability_snapshot_id=capabilities.snapshot_id,
        capability_snapshot_digest=capability_snapshot_digest,
        endpoint_config_digest=endpoint_config_digest(endpoint),
        max_concurrent_requests=endpoint.max_concurrent_requests,
    )
    binding = ProductionPrimaryRunPodBinding(
        endpoint=endpoint,
        handler_image_sha256=_digest(2),
        capability_snapshot_sha256=capability_snapshot_digest,
    )
    release_decision = tmp_path / "release-decision.json"
    release_decision.write_text(
        PrimaryRouteReleaseDecision(
            decision="APPROVED",
            route_id="production-mage-primary",
            policy_version="1.0",
            deployment=deployment,
            qualification_report_ref="evidence/qualification-report.json",
            qualification_report_sha256=exact_bytes_sha256(qualification_report.read_bytes()),
            primary_binding_sha256=binding.configuration_sha256,
            handler_image_sha256=(release_handler_image_sha256 or binding.handler_image_sha256),
        ).model_dump_json(),
        encoding="utf-8",
    )
    route = ProductionRoute(
        route_id="production-mage-primary",
        policy_version="1.0",
        deployment=deployment,
        authorization=ProductionRouteAuthorization(
            qualification_report_ref="evidence/qualification-report.json",
            qualification_report_sha256=exact_bytes_sha256(qualification_report.read_bytes()),
            release_decision_ref="evidence/release-decision.json",
            release_decision_sha256=exact_bytes_sha256(release_decision.read_bytes()),
        ),
    )
    return (
        ProductionRuntimeBootstrapConfiguration(
            primary_binding=binding,
            primary_capabilities=capabilities,
            primary_retry_policy=RunPodRetryPolicy(version="runpod-retry-v1"),
            primary_route=route,
            capture_authority=ProductionCaptureAuthorityBinding(
                capture_authority_id="production-capture-authority",
                capture_authority_epoch=1,
                capture_assignment_policy_version="capture-assignment-v1",
            ),
            outbox_retry_policy_version="primary-outbox-retry-v1",
            outbox_max_attempts=3,
            outbox_base_delay_seconds=1.0,
            outbox_max_delay_seconds=8.0,
            qualification_report_file=str(qualification_report),
            release_decision_file=str(release_decision),
        ),
        qualification_report,
        release_decision,
    )


def _write_bootstrap_configuration(
    configuration: ProductionRuntimeBootstrapConfiguration,
    path: Path,
) -> Path:
    path.write_text(configuration.model_dump_json(), encoding="utf-8")
    return path


def test_loads_strict_bootstrap_configuration_with_valid_artifacts(tmp_path: Path) -> None:
    configuration, _, _ = _bootstrap_configuration(tmp_path)
    configuration_path = _write_bootstrap_configuration(
        configuration,
        tmp_path / "production-runtime.json",
    )

    loaded = load_production_runtime_bootstrap_configuration(configuration_path)

    assert loaded == configuration
    assert loaded.outbox_retry_policy().max_attempts == 3
    assert loaded.primary_route.deployment.model_name == "mage-vl-4b"


def test_release_verifier_rejects_mutated_exact_evidence(tmp_path: Path) -> None:
    configuration, qualification_report, _ = _bootstrap_configuration(tmp_path)
    verifier = configuration.release_verifier()

    assert verifier.verify_primary(
        authorization=configuration.primary_route.authorization,
        deployment=configuration.primary_route.deployment,
    )

    qualification_report.write_bytes(b'{"status":"TAMPERED"}\n')

    assert (
        verifier.verify_primary(
            authorization=configuration.primary_route.authorization,
            deployment=configuration.primary_route.deployment,
        )
        is False
    )


def test_release_verifier_rejects_authorized_decision_with_wrong_handler_digest(
    tmp_path: Path,
) -> None:
    configuration, _, release_decision = _bootstrap_configuration(
        tmp_path,
        release_handler_image_sha256=_digest(99),
    )

    assert (
        exact_bytes_sha256(release_decision.read_bytes())
        == configuration.primary_route.authorization.release_decision_sha256
    )
    assert (
        configuration.release_verifier().verify_primary(
            authorization=configuration.primary_route.authorization,
            deployment=configuration.primary_route.deployment,
        )
        is False
    )


def test_load_rejects_missing_required_evidence_artifact(tmp_path: Path) -> None:
    configuration, _, _ = _bootstrap_configuration(tmp_path)
    payload = configuration.model_dump(mode="json")
    payload["qualification_report_file"] = str(tmp_path / "missing-qualification-report.json")
    configuration_path = tmp_path / "production-runtime.json"
    configuration_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ProductionBootstrapConfigurationError,
        match="does not match the reviewed bootstrap contract",
    ):
        load_production_runtime_bootstrap_configuration(configuration_path)
