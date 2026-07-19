"""Pipeline stage definitions and their associated status state machine."""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    """Canonical pipeline stage identifiers.

    Stages are ordered roughly by their position in the processing pipeline,
    from ingestion through inference to final publication.
    """

    # Ingestion and alignment
    MCAP_INGEST = "MCAP_INGEST"
    INSPECT_MAP = "INSPECT_MAP"
    ALIGN = "ALIGN"

    # Sampling and coarse QA planning
    QA_COARSE_PLAN = "QA_COARSE_PLAN"
    SAMPLE_MATERIALIZE = "SAMPLE_MATERIALIZE"

    # Coarse QA inference and reduction
    QWEN_QA_COARSE = "QWEN_QA_COARSE"
    QA_SUSPICION_REDUCE = "QA_SUSPICION_REDUCE"
    QA_DENSE_PLAN = "QA_DENSE_PLAN"
    QWEN_QA_DENSE = "QWEN_QA_DENSE"

    # QA aggregation
    QA_AGGREGATE = "QA_AGGREGATE"

    # Event proposal
    EVENT_PROPOSAL_PLAN = "EVENT_PROPOSAL_PLAN"
    QWEN_EVENT_PROPOSAL = "QWEN_EVENT_PROPOSAL"
    EVENT_PROPOSAL_REDUCE = "EVENT_PROPOSAL_REDUCE"

    # Action evidence
    ACTION_DENSE_PLAN = "ACTION_DENSE_PLAN"
    QWEN_ACTION_EVIDENCE = "QWEN_ACTION_EVIDENCE"

    # Fusion and boundary refinement
    FUSION_PROVISIONAL = "FUSION_PROVISIONAL"
    BOUNDARY_PLAN = "BOUNDARY_PLAN"
    QWEN_BOUNDARY = "QWEN_BOUNDARY"
    FUSION_FINAL = "FUSION_FINAL"

    # Publication and indexing
    ACTION_PUBLISH = "ACTION_PUBLISH"
    RETRIEVAL_INDEX = "RETRIEVAL_INDEX"
    VALUE_SCORE = "VALUE_SCORE"


class StageStatus(StrEnum):
    """Terminal and non-terminal states in the work-item lifecycle.

    The state machine transitions are:

    * PENDING -> RUNNING
    * RUNNING -> SUCCEEDED | FAILED | CANCELLED | EXPIRED | QUARANTINED
    * (any) -> SKIPPED_POLICY | SKIPPED_NOT_NEEDED
    * PENDING | RUNNING -> INCOMPLETE (partial failure, may retry)
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED_POLICY = "SKIPPED_POLICY"
    SKIPPED_NOT_NEEDED = "SKIPPED_NOT_NEEDED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    QUARANTINED = "QUARANTINED"
    INCOMPLETE = "INCOMPLETE"


class DependencyCriticality(StrEnum):
    """How strongly a downstream work item depends on an upstream one."""

    REQUIRED = "REQUIRED"
    DEGRADABLE = "DEGRADABLE"
    OPTIONAL = "OPTIONAL"


__all__ = [
    "DependencyCriticality",
    "Stage",
    "StageStatus",
]
