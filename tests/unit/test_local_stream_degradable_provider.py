from __future__ import annotations

from pathlib import Path

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import ArtifactEvidenceRef, StreamStage, TerminalOutcome
from robata.inference.models import InferenceStatus, VisionTask
from robata.queue.stream_models import StreamTerminalEvidence, StreamWorkItemState
from tests.integration.test_canonical_mcap_source import _pre_eos_fixture_model
from tests.unit.test_local_stream_finalization import (
    _composition,
    _digest,
    _finalizer,
    _uuid,
)


def test_failed_degradable_provider_abstains_window_and_closes_eos(tmp_path: Path) -> None:
    composition = _composition(tmp_path)
    model_schema = SchemaRef(
        schema_id="https://schemas.robata.dev/model-inference",
        version="1.0.0",
        artifact_id=_uuid(301),
        sha256=_digest(302),
    )
    task_by_stage = {
        StreamStage.QA_COARSE: VisionTask.QA_COARSE,
        StreamStage.QA_DENSE: VisionTask.QA_DENSE,
        StreamStage.EVENT_PROPOSAL: VisionTask.EVENT_PROPOSAL,
    }

    def provider_terminal(plan: object) -> StreamTerminalEvidence | None:
        stage = plan.stage
        task = task_by_stage.get(stage)
        if task is None:
            return None
        model = _pre_eos_fixture_model(task)
        outcome = TerminalOutcome.SUCCEEDED
        reason_code = None
        if stage is StreamStage.QA_COARSE:
            model = model.model_copy(
                update={
                    "status": InferenceStatus.FAILED,
                    "raw_output": None,
                    "normalized_output": None,
                    "output_valid": False,
                }
            )
            outcome = TerminalOutcome.FAILED
            reason_code = "PROVIDER_FAILURE"
        payload = canonical_json_bytes(model)
        digest = exact_bytes_sha256(payload)
        reference = ArtifactEvidenceRef(
            artifact_id=_uuid(400 + len(digest)),
            exact_sha256=digest,
            byte_count=len(payload),
            media_type="application/json",
            schema_ref=model_schema,
        )
        path = tmp_path / "artifacts" / digest[:2] / f"{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return StreamTerminalEvidence(
            outcome=outcome,
            evidence_ref=reference,
            terminal_policy_version="stream-terminal-policy-v1",
            completed_at="2026-01-01T00:30:00Z",
            reason_code=reason_code,
        )

    outcome = _finalizer(
        composition,
        tmp_path,
        stage_terminal_executor=provider_terminal,
        model_inference_schema_ref=model_schema,
    ).execute()

    reduction = next(
        item
        for item in composition.work_items(recover_graph=False)
        if item.stage is StreamStage.WINDOW_REDUCTION
    )
    assert reduction.state is StreamWorkItemState.ABSTAINED
    assert outcome.window_results[0].terminal_outcome is TerminalOutcome.ABSTAINED
    assert outcome.terminal_closure.complete
    assert outcome.terminal_closure.members[0].terminal_outcome is TerminalOutcome.ABSTAINED
    assert outcome.recording_result.ordered_window_semantic_statuses == ("ABSTAINED",)
    assert outcome.recording_result.output_decision == "ABSTAINED"
    assert outcome.finalization_work.state is StreamWorkItemState.SUCCEEDED
