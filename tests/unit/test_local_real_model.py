from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from robata.application.canonical import local_real_model
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.local_hf_adapter import LocalHfLoopbackVisionAdapter
from robata.inference.offline_fixture import (
    InMemoryRawProviderBytesStore,
    StrictProviderClaimParser,
)


def test_local_qwen_capabilities_and_policies_are_pinned() -> None:
    capabilities = local_real_model.build_local_qwen_capabilities()
    assert capabilities.provider == local_real_model.LOCAL_QWEN_PROVIDER
    assert capabilities.model_name == local_real_model.LOCAL_QWEN_MODEL_NAME
    assert capabilities.model_version == local_real_model.LOCAL_QWEN_MODEL_VERSION
    assert capabilities.max_images_per_request == local_real_model.LOCAL_QWEN_MAX_IMAGES
    assert capabilities.concurrency_class.value == "SERIAL"
    assert capabilities.supported_tasks

    policies = local_real_model.build_local_qwen_policies()
    assert len(policies) == 6
    assert tuple(policy.task for policy in policies) == tuple(capabilities.supported_tasks)
    assert all(policy.provider == capabilities.provider for policy in policies)
    assert all(policy.model_name == capabilities.model_name for policy in policies)
    assert all(policy.model_version == capabilities.model_version for policy in policies)
    assert all(policy.timeout_ms == local_real_model.LOCAL_QWEN_TIMEOUT_MS for policy in policies)
    assert all(
        policy.generation_config["max_new_tokens"] == local_real_model.LOCAL_QWEN_MAX_NEW_TOKENS
        for policy in policies
    )


def test_binding_uses_composition_owned_ledger_and_serial_limits() -> None:
    binding = local_real_model.build_local_qwen_model_binding()
    assert binding.adapter is None
    assert binding.adapter_factory is not None
    assert binding.max_concurrent_call_parts == 1
    assert binding.max_inference_batch_size == 1
    assert binding.normalized_output_lineage_policy is not None
    assert (
        binding.normalized_output_lineage_policy.normalization_contract_sha256
        == local_real_model.LOCAL_QWEN_NORMALIZATION_CONTRACT_SHA256
    )

    parser = StrictProviderClaimParser(SchemaRegistry(), parser_version="local-qwen-test-v1")
    adapter = binding.adapter_factory(InMemoryRawProviderBytesStore(), parser)
    assert isinstance(adapter, LocalHfLoopbackVisionAdapter)
    assert adapter.production_eligible is False
    assert adapter.canonical_authority is False
    assert adapter.config.default_max_new_tokens == local_real_model.LOCAL_QWEN_MAX_NEW_TOKENS
    assert adapter.config.request_timeout_cap_ms == local_real_model.LOCAL_QWEN_TIMEOUT_MS


def test_wrapper_forwards_full_duration_and_local_observer(monkeypatch, tmp_path: Path) -> None:
    observer = object()
    binding = object()
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(local_real_model, "RuntimeProfileRecorder", lambda: observer)
    monkeypatch.setattr(
        local_real_model,
        "build_local_qwen_model_binding",
        lambda *, transport=None, checkpoint_manifest_sha256=None: binding,
    )

    def fake_run(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(local_real_model, "run_local_canonical_mcap", fake_run)
    result = local_real_model.run_local_qwen_canonical_mcap(
        source_path=tmp_path / "input.mcap",
        mapping_config=tmp_path / "mapping.json",
        state_dir=tmp_path / "state",
    )

    assert result is sentinel
    assert captured["max_duration_ns"] is None
    assert captured["runtime_observer"] is observer
    assert captured["model_binding"] is binding
    assert captured["run_key"] == "qwen-local-2026-08-06"
    assert captured["allow_unapproved_profile"] is False


def test_wrapper_honors_injected_observer_and_transport(monkeypatch, tmp_path: Path) -> None:
    observer = object()
    transport = object()
    binding = object()
    captured: dict[str, object] = {}

    def fake_build_binding(
        *,
        transport: object | None = None,
        checkpoint_manifest_sha256: str | None = None,
    ) -> object:
        captured["transport"] = transport
        captured["checkpoint_manifest_sha256"] = checkpoint_manifest_sha256
        return binding

    monkeypatch.setattr(
        local_real_model,
        "build_local_qwen_model_binding",
        fake_build_binding,
    )
    monkeypatch.setattr(
        local_real_model,
        "run_local_canonical_mcap",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(ok=True),
    )

    local_real_model.run_local_qwen_canonical_mcap(
        source_path=tmp_path / "input.mcap",
        mapping_config=tmp_path / "mapping.json",
        state_dir=tmp_path / "state",
        runtime_observer=observer,
        transport=transport,
        checkpoint_manifest_sha256="a" * 64,
    )
    assert captured["transport"] is transport
    assert captured["checkpoint_manifest_sha256"] == "a" * 64
    assert captured["runtime_observer"] is observer


def test_checkpoint_manifest_changes_capability_snapshot_identity() -> None:
    first = local_real_model.build_local_qwen_capabilities(checkpoint_manifest_sha256="a" * 64)
    second = local_real_model.build_local_qwen_capabilities(checkpoint_manifest_sha256="b" * 64)

    assert first.snapshot_digest != second.snapshot_digest
    assert first.snapshot_id != second.snapshot_id


def test_checkpoint_manifest_digest_is_strictly_validated() -> None:
    import pytest

    with pytest.raises(ValueError, match="checkpoint_manifest_sha256"):
        local_real_model.build_local_qwen_capabilities(checkpoint_manifest_sha256="not-a-digest")
