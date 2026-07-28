"""Versioned, API-owned DTOs for the local committed-run explorer."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field

from robata.contracts.common import StrictModel

WEB_API_VERSION: Final = "v1"


class NanosecondIntervalView(StrictModel):
    """A JSON-safe interval; nanoseconds remain canonical decimal strings."""

    start_ns: str
    end_ns: str


class RunSummaryView(StrictModel):
    """Small list projection for choosing an explicit committed run."""

    run_id: str
    recording_identity: str
    status: str
    started_at: str | None
    completed_at: str | None
    pipeline_version: str
    output_decision: str | None
    event_count: int = Field(ge=0)


class RunListResponse(StrictModel):
    api_version: Literal["v1"] = "v1"
    runs: tuple[RunSummaryView, ...]


class RunWindowView(StrictModel):
    logical_key: str
    purpose: str
    requested_interval: NanosecondIntervalView
    effective_interval: NanosecondIntervalView
    recording_duration_ns: str


class RunPackageView(StrictModel):
    package_id: str
    ordinal: int = Field(ge=0)
    part_count: int = Field(ge=1)
    interval: NanosecondIntervalView


class CameraQualityView(StrictModel):
    camera_id: str
    status: str
    interval: NanosecondIntervalView


class PipelineStageView(StrictModel):
    name: str
    state: Literal["COMPLETE", "NOT_RUN"]
    semantic_sha256: str | None


class RunDecisionView(StrictModel):
    decision: str
    reason_code: str | None
    admitted_claim_count: int = Field(ge=0)


class RunHypothesisView(StrictModel):
    ordinal: int = Field(ge=0)
    logical_key: str
    semantic_sha256: str
    effective_interval: NanosecondIntervalView


class RunPublicationView(StrictModel):
    event_id: str
    revision_id: str
    effective_interval: NanosecondIntervalView


class EvidenceView(StrictModel):
    role: str
    schema_id: str
    schema_version: str
    semantic_sha256: str
    exact_bytes_sha256: str
    byte_count: int = Field(ge=1)


class RunIntegrityView(StrictModel):
    command_sha256: str
    completion_semantic_sha256: str


class RunSnapshotView(RunSummaryView):
    evidence_class: str
    production_eligible: bool
    window: RunWindowView | None
    packages: tuple[RunPackageView, ...]
    camera_quality: tuple[CameraQualityView, ...]
    stages: tuple[PipelineStageView, ...]
    decision: RunDecisionView | None
    hypotheses: tuple[RunHypothesisView, ...]
    publications: tuple[RunPublicationView, ...]
    integrity: RunIntegrityView
    evidence: tuple[EvidenceView, ...]


class RunSnapshotResponse(StrictModel):
    api_version: Literal["v1"] = "v1"
    cursor: str
    run: RunSnapshotView


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"


class WebSocketSnapshot(StrictModel):
    type: Literal["snapshot"] = "snapshot"
    snapshot: RunSnapshotResponse
