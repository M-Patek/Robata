from __future__ import annotations

from pathlib import Path

import pytest

from robata.adapters.sqlite_inference_evidence import MODEL_INFERENCE_SCHEMA_ID
from robata.application.canonical import local_composition as local_composition_module
from robata.application.canonical import mcap_source as mcap_source_module
from robata.application.canonical.local_composition import (
    LOCAL_CANONICAL_STREAM_TERMINAL_POLICY_VERSION,
    LocalPreEosExecutorContext,
    run_local_canonical_mcap,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.stream_common import StreamStage
from robata.contracts.stream_planning import StreamWorkItemPlan
from tests.support.six_camera_mcap import SIX_CAMERA_TOPICS, write_six_camera_mcap


def _write_mapping(path: Path) -> Path:
    path.write_bytes(
        canonical_json_bytes(
            {
                "profile_id": "pre-eos-factory-wiring-v1",
                "version": "pre-eos-factory-wiring-v1",
                "profile_kind": "TEST_FIXTURE",
                "approval_status": "UNAPPROVED",
                "approved": False,
                "mapping_policy": "EXACT_TOPIC",
                "required_schema": "foxglove.CompressedImage",
                "topics": {
                    camera_id.value: topic
                    for camera_id, topic in zip(
                        CAMERA_IDS,
                        SIX_CAMERA_TOPICS,
                        strict=True,
                    )
                },
            }
        )
    )
    return path


def test_pre_eos_factory_builds_runtime_before_source_and_reuses_hook_at_eos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One runtime-owned hook spans incremental and EOS stream execution."""

    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    mapping = _write_mapping(tmp_path / "mapping.json")
    state_dir = tmp_path / "canonical-state"
    sequence: list[str] = []
    contexts: list[LocalPreEosExecutorContext] = []
    pre_eos_stages: list[StreamStage] = []
    eos_hooks: list[object] = []

    def hook(plan: StreamWorkItemPlan) -> None:
        sequence.append("pre-eos")
        pre_eos_stages.append(plan.stage)
        # The explicit hook may decline a work item; the local mock then owns
        # only that item. A real ProviderNeutralStreamStageExecutor returns its
        # typed terminal instead.
        return None

    def factory(context: LocalPreEosExecutorContext):
        sequence.append("factory")
        contexts.append(context)
        return hook

    original_source_loader = mcap_source_module.load_canonical_mcap_source

    def observed_source_loader(*args: object, **kwargs: object):
        sequence.append("source")
        assert contexts
        assert kwargs["stage_terminal_executor"] is hook
        return original_source_loader(*args, **kwargs)

    original_finalizer = local_composition_module._finalize_local_stream_graphs

    def observed_finalizer(*args: object, **kwargs: object):
        sequence.append("eos")
        eos_hooks.append(kwargs["stage_terminal_executor"])
        return original_finalizer(*args, **kwargs)

    monkeypatch.setattr(
        mcap_source_module,
        "load_canonical_mcap_source",
        observed_source_loader,
    )
    monkeypatch.setattr(
        local_composition_module,
        "_finalize_local_stream_graphs",
        observed_finalizer,
    )

    first = run_local_canonical_mcap(
        source,
        mapping,
        state_dir,
        run_key="pre-eos-factory-wiring",
        allow_unapproved_profile=True,
        pre_eos_executor_factory=factory,
    )

    assert first.replayed is False
    assert len(contexts) == 1
    context = contexts[0]
    assert context.pipeline is not None
    assert context.artifact_root == state_dir.resolve() / "stream-artifacts"
    assert context.model_inference_schema_ref.schema_id == MODEL_INFERENCE_SCHEMA_ID
    assert context.model_inference_schema_ref.version == "1.0.0"
    assert context.terminal_policy_version == LOCAL_CANONICAL_STREAM_TERMINAL_POLICY_VERSION
    assert sequence.index("factory") < sequence.index("source")
    assert pre_eos_stages
    assert sequence.index("source") < sequence.index("eos")
    assert eos_hooks == [hook]

    # Completion recovery returns before source preparation, runtime/factory
    # construction, or EOS execution. This is the outer no-duplicate-dispatch
    # guard; durable provider selection replay is exercised separately.
    before_replay = (len(contexts), len(pre_eos_stages), len(eos_hooks))
    replay = run_local_canonical_mcap(
        source,
        mapping,
        state_dir,
        run_key="pre-eos-factory-wiring",
        allow_unapproved_profile=True,
        pre_eos_executor_factory=factory,
    )

    assert replay.replayed is True
    assert (len(contexts), len(pre_eos_stages), len(eos_hooks)) == before_replay
