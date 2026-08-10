from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from robata.application.canonical import local_composition as composition
from robata.application.canonical.local_composition import (
    LocalCanonicalModelBinding,
    LocalCanonicalNativeBatchAdmission,
)
from robata.application.canonical.runner import NormalizedOutputLineagePolicy
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.adapter import VisionInferenceRequest
from robata.inference.models import ModelCapabilities


class _Adapter:
    def __init__(self, capabilities: ModelCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def provider(self) -> str:
        return self._capabilities.provider

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        return self._capabilities

    async def infer(self, request: VisionInferenceRequest) -> object:
        raise AssertionError("the focused composition test must not call the adapter")


class _BatchAdapter(_Adapter):
    def __init__(
        self,
        capabilities: ModelCapabilities,
        *,
        policy_version: str = "test-native-batch-v1",
        max_batch_size: int = 4,
    ) -> None:
        super().__init__(capabilities)
        self.native_batch_policy_version = policy_version
        self.native_batch_max_size = max_batch_size

    async def infer_batch(
        self,
        requests: tuple[VisionInferenceRequest, ...],
    ) -> tuple[object, ...]:
        raise AssertionError("the focused composition test must not call the adapter")


def _binding() -> LocalCanonicalModelBinding:
    capabilities = composition._capabilities(composition.LOCAL_CANONICAL_EXECUTION_TIME).model_copy(
        update={
            "provider": "local-qwen-loopback",
            "model_name": "qwen-test",
            "model_version": "1.0",
        }
    )
    policies = tuple(
        policy.model_copy(
            update={
                "provider": capabilities.provider,
                "model_name": capabilities.model_name,
                "model_version": capabilities.model_version,
                "adapter_version": "local-qwen-adapter-v1",
            }
        )
        for policy in composition._local_inference_policies(SchemaRegistry())
    )
    return LocalCanonicalModelBinding(
        adapter=_Adapter(capabilities),
        capabilities=capabilities,
        coarse_qa_policy=policies[0],
        dense_qa_policy=policies[1],
        event_proposal_policy=policies[2],
        action_evidence_policy=policies[3],
        boundary_refinement_policy=policies[4],
        inference_policy=policies[5],
    )


def _batch_admission() -> LocalCanonicalNativeBatchAdmission:
    return LocalCanonicalNativeBatchAdmission(
        policy_version="test-native-batch-v1",
        max_batch_size=4,
        capacity_projection_version="test-batch4-capacity-v1",
        serial_guard_policy_version="test-multi-claim-serial-guard-v1",
    )


def _batch_binding() -> LocalCanonicalModelBinding:
    serial = _binding()
    return replace(
        serial,
        adapter=_BatchAdapter(serial.capabilities),
        native_batch_admission=_batch_admission(),
        max_concurrent_call_parts=4,
        max_inference_batch_size=4,
    )


def _build_bound_runtime(
    *,
    tmp_path: Path,
    binding: LocalCanonicalModelBinding,
) -> object:
    policies = binding.policies
    state_root = tmp_path / f"runtime-{uuid4()}"
    state_root.mkdir()
    return composition._build_runtime(
        state_root=state_root,
        run_id=str(uuid4()),
        registry=SchemaRegistry(),
        execution_policy=composition._execution_policy(),
        coarse_qa_policy=policies[0],
        dense_qa_policy=policies[1],
        event_proposal_policy=policies[2],
        action_evidence_policy=policies[3],
        boundary_refinement_policy=policies[4],
        inference_policy=policies[5],
        clock_value=composition._LOCAL_CANONICAL_EXECUTION_DATETIME,
        observed_at=composition.LOCAL_CANONICAL_EXECUTION_TIME,
        model_binding=binding,
    )


def test_model_binding_pins_policy_descriptor_and_serial_limits() -> None:
    binding = _binding()

    assert binding.policies == (
        binding.coarse_qa_policy,
        binding.dense_qa_policy,
        binding.event_proposal_policy,
        binding.action_evidence_policy,
        binding.boundary_refinement_policy,
        binding.inference_policy,
    )
    assert binding.max_concurrent_call_parts == 1
    assert binding.max_inference_batch_size == 1
    assert binding.max_batch_size == 1

    default_descriptor = composition.local_canonical_runtime_descriptor()
    bound_descriptor = composition.local_canonical_runtime_descriptor(model_binding=binding)
    assert bound_descriptor.inference_policy_versions == tuple(
        f"{policy.task.value}:{policy.policy_version}" for policy in binding.policies
    )
    assert bound_descriptor.runtime_policy_semantic_sha256 != (
        default_descriptor.runtime_policy_semantic_sha256
    )


def test_build_runtime_uses_binding_adapter_and_serial_pipeline_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    calls: list[dict[str, object]] = []

    class _CapturingPipeline:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(composition, "CanonicalOfflinePipeline", _CapturingPipeline)
    registry = SchemaRegistry()
    policies = binding.policies
    state_root = tmp_path / "bound"
    state_root.mkdir()
    runtime = composition._build_runtime(
        state_root=state_root,
        run_id=str(uuid4()),
        registry=registry,
        execution_policy=composition._execution_policy(),
        coarse_qa_policy=policies[0],
        dense_qa_policy=policies[1],
        event_proposal_policy=policies[2],
        action_evidence_policy=policies[3],
        boundary_refinement_policy=policies[4],
        inference_policy=policies[5],
        clock_value=composition._LOCAL_CANONICAL_EXECUTION_DATETIME,
        observed_at=composition.LOCAL_CANONICAL_EXECUTION_TIME,
        model_binding=binding,
    )
    try:
        adapter = calls[-1]["adapter"]
        assert isinstance(adapter, composition._PinnedCapabilityVisionModelAdapter)
        assert adapter._delegate is binding.adapter
        assert calls[-1]["max_concurrent_call_parts"] == 1
        assert calls[-1]["max_inference_batch_size"] == 1
    finally:
        runtime.inference_evidence.close()

    default_root = tmp_path / "fixture"
    default_root.mkdir()
    fixture_policies = composition._local_inference_policies(SchemaRegistry())
    default_runtime = composition._build_runtime(
        state_root=default_root,
        run_id=str(uuid4()),
        registry=SchemaRegistry(),
        execution_policy=composition._execution_policy(),
        coarse_qa_policy=fixture_policies[0],
        dense_qa_policy=fixture_policies[1],
        event_proposal_policy=fixture_policies[2],
        action_evidence_policy=fixture_policies[3],
        boundary_refinement_policy=fixture_policies[4],
        inference_policy=fixture_policies[5],
        clock_value=composition._LOCAL_CANONICAL_EXECUTION_DATETIME,
        observed_at=composition.LOCAL_CANONICAL_EXECUTION_TIME,
    )
    try:
        assert isinstance(calls[-1]["adapter"], composition.OfflineFixtureVisionAdapter)
        assert calls[-1]["max_concurrent_call_parts"] == (
            composition.LOCAL_CANONICAL_MAX_CONCURRENT_CALL_PARTS
        )
        assert calls[-1]["max_inference_batch_size"] == (
            composition.LOCAL_CANONICAL_MAX_INFERENCE_BATCH_SIZE
        )
    finally:
        default_runtime.inference_evidence.close()


def test_build_runtime_adapter_factory_receives_owned_evidence_and_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    factory_inputs: list[tuple[object, object]] = []
    calls: list[dict[str, object]] = []

    def factory(evidence: object, parser: object) -> object:
        factory_inputs.append((evidence, parser))
        return binding.adapter

    factory_binding = replace(binding, adapter=None, adapter_factory=factory)

    class _CapturingPipeline:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(composition, "CanonicalOfflinePipeline", _CapturingPipeline)
    policies = factory_binding.policies
    state_root = tmp_path / "factory"
    state_root.mkdir()
    runtime = composition._build_runtime(
        state_root=state_root,
        run_id=str(uuid4()),
        registry=SchemaRegistry(),
        execution_policy=composition._execution_policy(),
        coarse_qa_policy=policies[0],
        dense_qa_policy=policies[1],
        event_proposal_policy=policies[2],
        action_evidence_policy=policies[3],
        boundary_refinement_policy=policies[4],
        inference_policy=policies[5],
        clock_value=composition._LOCAL_CANONICAL_EXECUTION_DATETIME,
        observed_at=composition.LOCAL_CANONICAL_EXECUTION_TIME,
        model_binding=factory_binding,
    )
    try:
        assert factory_inputs[0][0] is runtime.inference_evidence
        assert factory_inputs[0][1] is calls[-1]["parser"]
        adapter = calls[-1]["adapter"]
        assert isinstance(adapter, composition._PinnedCapabilityVisionModelAdapter)
        assert adapter._delegate is binding.adapter
        assert calls[-1]["max_concurrent_call_parts"] == 1
        assert calls[-1]["max_inference_batch_size"] == 1
    finally:
        runtime.inference_evidence.close()


def test_mcap_entrypoint_forwards_model_binding_without_source_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    source = tmp_path / "recording.mcap"
    mapping = tmp_path / "mapping.json"
    source.write_bytes(b"not parsed by this forwarding test")
    mapping.write_text("{}", encoding="utf-8")

    from robata.application.canonical import mcap_source

    monkeypatch.setattr(
        mcap_source,
        "authorize_mcap_mapping",
        lambda path, *, allow_unapproved_profile: SimpleNamespace(semantic_sha256="a" * 64),
    )
    forwarded: dict[str, object] = {}
    sentinel = object()

    def fake_run(**kwargs: object) -> object:
        forwarded.update(kwargs)
        return sentinel

    monkeypatch.setattr(composition, "_run_local_canonical", fake_run)

    result = composition.run_local_canonical_mcap(
        source_path=source,
        mapping_config=mapping,
        state_dir=tmp_path / "state",
        model_binding=binding,
    )

    assert result is sentinel
    assert forwarded["model_binding"] is binding


def test_model_binding_requires_exactly_one_adapter_source() -> None:
    binding = _binding()
    with pytest.raises(ValueError, match="exactly one of adapter or adapter_factory"):
        replace(binding, adapter=None, adapter_factory=None)


def test_model_binding_rejects_non_serial_limits() -> None:
    binding = _binding()
    with pytest.raises(
        ValueError,
        match="max_concurrent_call_parts must be 1 without native batch admission",
    ):
        LocalCanonicalModelBinding(
            adapter=binding.adapter,
            capabilities=binding.capabilities,
            coarse_qa_policy=binding.coarse_qa_policy,
            dense_qa_policy=binding.dense_qa_policy,
            event_proposal_policy=binding.event_proposal_policy,
            action_evidence_policy=binding.action_evidence_policy,
            boundary_refinement_policy=binding.boundary_refinement_policy,
            inference_policy=binding.inference_policy,
            max_concurrent_call_parts=2,
        )


@pytest.mark.asyncio
async def test_pinned_adapter_rejects_factory_capability_snapshot_drift() -> None:
    binding = _binding()
    drifted = binding.capabilities.model_copy(update={"observed_at": "2026-08-06T01:00:00Z"})
    adapter = composition._PinnedCapabilityVisionModelAdapter(
        _Adapter(drifted),
        binding.capabilities,
    )

    with pytest.raises(ValueError, match="capability snapshot differs"):
        await adapter.capabilities(
            binding.capabilities.model_name,
            binding.capabilities.model_version,
        )


def test_normalized_lineage_policy_is_explicit_and_changes_runtime_identity() -> None:
    binding = _binding()
    policy = NormalizedOutputLineagePolicy(
        version="local-compact-lineage-v1",
        parser_version="local-compact-parser-v1-aabbccdd",
        provider=binding.capabilities.provider,
        model_name=binding.capabilities.model_name,
        model_version=binding.capabilities.model_version,
        adapter_version=binding.inference_policy.adapter_version,
        normalization_contract_sha256="a" * 64,
        allowed_tasks=tuple(item.task for item in binding.policies),
    )
    normalized_binding = replace(binding, normalized_output_lineage_policy=policy)

    strict_descriptor = composition.local_canonical_runtime_descriptor(model_binding=binding)
    normalized_descriptor = composition.local_canonical_runtime_descriptor(
        model_binding=normalized_binding
    )

    assert policy.semantic_sha256 != policy.normalization_contract_sha256
    assert normalized_descriptor.runtime_policy_semantic_sha256 != (
        strict_descriptor.runtime_policy_semantic_sha256
    )


def test_model_binding_rejects_normalized_lineage_model_mismatch() -> None:
    binding = _binding()
    policy = NormalizedOutputLineagePolicy(
        version="local-compact-lineage-v1",
        parser_version="local-compact-parser-v1-aabbccdd",
        provider="different-provider",
        model_name=binding.capabilities.model_name,
        model_version=binding.capabilities.model_version,
        adapter_version=binding.inference_policy.adapter_version,
        normalization_contract_sha256="a" * 64,
        allowed_tasks=tuple(item.task for item in binding.policies),
    )

    with pytest.raises(ValueError, match="capability model identity"):
        replace(binding, normalized_output_lineage_policy=policy)


def test_native_batch_binding_requires_method_policy_and_size_declaration() -> None:
    serial = _binding()
    admission = _batch_admission()

    with pytest.raises(TypeError, match="must implement callable infer_batch"):
        replace(
            serial,
            native_batch_admission=admission,
            max_concurrent_call_parts=4,
            max_inference_batch_size=4,
        )
    with pytest.raises(ValueError, match="adapter policy does not match"):
        replace(
            serial,
            adapter=_BatchAdapter(serial.capabilities, policy_version="wrong-policy-v1"),
            native_batch_admission=admission,
            max_concurrent_call_parts=4,
            max_inference_batch_size=4,
        )
    with pytest.raises(ValueError, match="adapter max size does not match"):
        replace(
            serial,
            adapter=_BatchAdapter(serial.capabilities, max_batch_size=2),
            native_batch_admission=admission,
            max_concurrent_call_parts=4,
            max_inference_batch_size=4,
        )


def test_native_batch_binding_rejects_invalid_capacity_bounds() -> None:
    batch = _batch_binding()

    with pytest.raises(ValueError, match="must match native batch admission"):
        replace(batch, max_inference_batch_size=3)
    with pytest.raises(ValueError, match="sufficient to fill the native batch"):
        replace(batch, max_concurrent_call_parts=3)
    with pytest.raises(ValueError, match="between 1 and 6"):
        replace(batch, max_concurrent_call_parts=7)


def test_native_batch_capacity_projection_changes_only_runtime_identity() -> None:
    serial = _binding()
    batch = _batch_binding()

    assert batch.capabilities == serial.capabilities
    assert batch.policies == serial.policies
    assert batch.runtime_capacity_projection == {
        "semantic_projection_version": (
            composition.LOCAL_CANONICAL_NATIVE_BATCH_ADMISSION_PROJECTION_VERSION
        ),
        "capacity_projection_version": "test-batch4-capacity-v1",
        "policy_version": "test-native-batch-v1",
        "serial_guard_policy_version": "test-multi-claim-serial-guard-v1",
        "max_batch_size": 4,
        "max_concurrent_call_parts": 4,
        "execution_mode": "NATIVE_BATCH_WITH_EXPLICIT_SERIAL_GUARD",
    }
    assert (
        composition.local_canonical_runtime_descriptor(
            model_binding=batch
        ).runtime_policy_semantic_sha256
        != composition.local_canonical_runtime_descriptor(
            model_binding=serial
        ).runtime_policy_semantic_sha256
    )


def test_factory_batch_adapter_is_validated_fail_closed_during_composition(
    tmp_path: Path,
) -> None:
    batch = _batch_binding()
    bad_factory_binding = replace(
        batch,
        adapter=None,
        adapter_factory=lambda evidence, parser: _Adapter(batch.capabilities),
    )

    with pytest.raises(
        composition.CanonicalLocalCompositionError,
        match="model native batch admission failed: native batch adapter must implement",
    ):
        _build_bound_runtime(tmp_path=tmp_path, binding=bad_factory_binding)


def test_build_runtime_admits_declared_native_batch_and_uses_batch_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch_binding()
    calls: list[dict[str, object]] = []

    class _CapturingPipeline:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(composition, "CanonicalOfflinePipeline", _CapturingPipeline)
    factory_binding = replace(
        batch,
        adapter=None,
        adapter_factory=lambda evidence, parser: batch.adapter,
    )
    runtime = _build_bound_runtime(tmp_path=tmp_path, binding=factory_binding)
    try:
        adapter = calls[-1]["adapter"]
        assert isinstance(adapter, composition._PinnedCapabilityBatchVisionModelAdapter)
        assert adapter._delegate is batch.adapter
        assert calls[-1]["max_concurrent_call_parts"] == 4
        assert calls[-1]["max_inference_batch_size"] == 4
    finally:
        runtime.inference_evidence.close()


def test_serial_binding_hides_additive_adapter_batch_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial = _binding()
    batch_capable_serial = replace(
        serial,
        adapter=_BatchAdapter(serial.capabilities),
    )
    calls: list[dict[str, object]] = []

    class _CapturingPipeline:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(composition, "CanonicalOfflinePipeline", _CapturingPipeline)
    runtime = _build_bound_runtime(tmp_path=tmp_path, binding=batch_capable_serial)
    try:
        adapter = calls[-1]["adapter"]
        assert type(adapter) is composition._PinnedCapabilityVisionModelAdapter
        assert not callable(getattr(adapter, "infer_batch", None))
        assert calls[-1]["max_concurrent_call_parts"] == 1
        assert calls[-1]["max_inference_batch_size"] == 1
    finally:
        runtime.inference_evidence.close()
