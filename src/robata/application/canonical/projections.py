"""Semantic projections and deterministic identity helpers for the canonical flow."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final
from uuid import NAMESPACE_URL, uuid5

from robata.contracts.cameras import CAMERA_ID_VALUES
from robata.contracts.common import INT64_MAX, INT64_MIN, NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.contracts.pipeline import SamplingPurpose
from robata.event_pipeline.identity_registry import (
    AdmissionEvidenceClass,
    OutputAdmissionProof,
    PlatformEnrichedOutputReference,
    ProductionOutputAdmissionPolicyRef,
    platform_enriched_output_logical_projection,
    validate_evidence_eligibility,
)
from robata.inference.enrichment import EnrichedProviderClaim

if TYPE_CHECKING:
    from robata.application.canonical.models import CanonicalOfflineExecutionPolicy
    from robata.application.canonical.output_admission import CanonicalOutputAdmissionDecision
    from robata.application.canonical.reduction import (
        CanonicalFusionPartSource,
        CanonicalFusionReduction,
        CanonicalReducedFusionClaim,
    )


CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION: Final = (
    "canonical-output-decision-semantic-v2"
)
CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE: Final = "output-admission-decision-v2"
CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE: Final = "canonical-output-admission-v2"
CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION: Final = "canonical-output-decision-node-v2"
CANONICAL_EXECUTION_POLICY_SEMANTIC_PROJECTION_VERSION: Final = (
    "canonical-offline-execution-policy-semantic-v3"
)
# EventIndex is an internal structured-retrieval projection.  It deliberately
# has its own version so changing the shape of the index row does not change
# any ActionEvent/revision identity or require mutating a released schema.
CANONICAL_EVENT_INDEX_PROJECTION_VERSION: Final = "canonical-event-index-projection-v1"


def canonical_fusion_reduction_projection(
    reduction: CanonicalFusionReduction,
) -> dict[str, object]:
    return _canonical_fusion_reduction_projection_values(
        schema_version=reduction.schema_version,
        input_plan_semantic_sha256=reduction.input_plan_semantic_sha256,
        barrier_reduction_semantic_sha256=reduction.barrier_reduction_semantic_sha256,
        reduction_policy=reduction.reduction_policy,
        reduction_policy_version=reduction.reduction_policy_version,
        outcome=reduction.outcome,
        parts=reduction.parts,
        claims=reduction.claims,
    )


def _canonical_fusion_reduction_projection_values(
    *,
    schema_version: str,
    input_plan_semantic_sha256: str,
    barrier_reduction_semantic_sha256: str,
    reduction_policy: str,
    reduction_policy_version: str,
    outcome: str,
    parts: Sequence[CanonicalFusionPartSource],
    claims: Sequence[CanonicalReducedFusionClaim],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "input_plan_semantic_sha256": input_plan_semantic_sha256,
        "barrier_reduction_semantic_sha256": barrier_reduction_semantic_sha256,
        "reduction_policy": reduction_policy,
        "reduction_policy_version": reduction_policy_version,
        "outcome": outcome,
        "parts": [
            {
                "part_ordinal": part.part_ordinal,
                "part_semantic_sha256": part.part_semantic_sha256,
                "selected_attempt_output_sha256": part.selected_attempt_output_sha256,
                "enrichment": platform_enriched_output_logical_projection(part.enrichment),
                "abstained": part.abstained,
            }
            for part in parts
        ],
        "claims": [
            {
                "fusion_output_ordinal": claim.fusion_output_ordinal,
                "claim_semantic_sha256": claim.claim_semantic_sha256,
                "representative": _fusion_claim_reduction_projection(claim.representative),
                "sources": [
                    {
                        "part_ordinal": item.part_ordinal,
                        "source_claim_ordinal": item.source_claim_ordinal,
                        "enrichment_logical_key": item.enrichment_logical_key,
                    }
                    for item in claim.sources
                ],
            }
            for claim in claims
        ],
    }


def canonical_root_window_projection_values(values: Mapping[str, object]) -> dict[str, object]:
    """Project root-window semantics without row IDs, associations, or clocks."""

    requested = values["requested_interval"]
    effective = values["interval"]
    if not isinstance(requested, NanosecondInterval) or not isinstance(
        effective, NanosecondInterval
    ):
        raise TypeError("window intervals must be NanosecondInterval values")
    purpose = values["purpose"]
    if not isinstance(purpose, SamplingPurpose):
        raise TypeError("window purpose must be a SamplingPurpose")
    return {
        "source_content_sha256": values["source_content_sha256"],
        "camera_mapping_semantic_sha256": values["camera_mapping_semantic_sha256"],
        "alignment_semantic_sha256": values["alignment_semantic_sha256"],
        "requested_interval": {
            "start_ns": str(requested.start_ns),
            "end_ns": str(requested.end_ns),
        },
        "interval": {
            "start_ns": str(effective.start_ns),
            "end_ns": str(effective.end_ns),
        },
        "purpose": purpose.value,
        "window_policy_version": values["window_policy_version"],
        "source_subject_type": values["source_subject_type"],
        "source_subject_logical_key": values["source_subject_logical_key"],
        "parent_window_logical_key": values["parent_window_logical_key"],
        "source_lineage_sha256": values["source_lineage_sha256"],
        "refinement_role": values["refinement_role"],
        "generation": values["generation"],
    }


def canonical_execution_policy_projection_values(values: Mapping[str, object]) -> dict[str, object]:
    output_policy = values["output_admission_policy"]
    if not isinstance(output_policy, ProductionOutputAdmissionPolicyRef):
        raise TypeError("output_admission_policy must be a ProductionOutputAdmissionPolicyRef")
    return {
        "semantic_projection_version": CANONICAL_EXECUTION_POLICY_SEMANTIC_PROJECTION_VERSION,
        "policy_version": values["policy_version"],
        "window_policy_version": values["window_policy_version"],
        "token_policy_version": values["token_policy_version"],
        "parser_version": values["parser_version"],
        "enrichment_policy_version": values["enrichment_policy_version"],
        "projector_policy_version": values["projector_policy_version"],
        "reduction_policy": values["reduction_policy"],
        "reduction_policy_version": values["reduction_policy_version"],
        "provisional_fusion_policy_version": values["provisional_fusion_policy_version"],
        "boundary_refinement_policy_version": values["boundary_refinement_policy_version"],
        "max_attempts": values["max_attempts"],
        "output_admission_policy": output_policy.model_dump(mode="json"),
    }


def canonical_execution_policy_projection(
    policy: CanonicalOfflineExecutionPolicy,
) -> dict[str, object]:
    return canonical_execution_policy_projection_values(
        {
            "policy_version": policy.policy_version,
            "window_policy_version": policy.window_policy_version,
            "token_policy_version": policy.token_policy_version,
            "parser_version": policy.parser_version,
            "enrichment_policy_version": policy.enrichment_policy_version,
            "projector_policy_version": policy.projector_policy_version,
            "reduction_policy": policy.reduction_policy,
            "reduction_policy_version": policy.reduction_policy_version,
            "provisional_fusion_policy_version": policy.provisional_fusion_policy_version,
            "boundary_refinement_policy_version": (policy.boundary_refinement_policy_version),
            "max_attempts": policy.max_attempts,
            "output_admission_policy": policy.output_admission_policy,
        }
    )


def canonical_output_decision_projection(
    decision: CanonicalOutputAdmissionDecision,
) -> dict[str, object]:
    return _canonical_output_decision_projection_values(
        decision=decision.decision,
        evidence_class=decision.evidence_class,
        production_eligible=decision.production_eligible,
        recording_identity=decision.recording_identity,
        source_enrichments=decision.source_enrichments,
        fusion_reduction_logical_key=decision.fusion_reduction_logical_key,
        fusion_reduction_semantic_sha256=decision.fusion_reduction_semantic_sha256,
        policy_version=decision.policy_version,
        policy_sha256=decision.policy_sha256,
        admitted_claim_ordinals=decision.admitted_claim_ordinals,
        reason_code=decision.reason_code,
        production_output_admission=decision.production_output_admission,
    )


def _canonical_output_decision_projection_values(
    *,
    decision: str,
    evidence_class: AdmissionEvidenceClass,
    production_eligible: bool,
    recording_identity: str,
    source_enrichments: Sequence[PlatformEnrichedOutputReference],
    fusion_reduction_logical_key: str,
    fusion_reduction_semantic_sha256: str,
    policy_version: str,
    policy_sha256: str,
    admitted_claim_ordinals: Sequence[int],
    reason_code: str,
    production_output_admission: OutputAdmissionProof | None,
) -> dict[str, object]:
    validate_evidence_eligibility(evidence_class, production_eligible)
    return {
        "semantic_projection_version": CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION,
        "decision": decision,
        "evidence_class": evidence_class.value,
        "production_eligible": production_eligible,
        "recording_identity": recording_identity,
        "source_enrichments": [
            platform_enriched_output_logical_projection(item) for item in source_enrichments
        ],
        "fusion_reduction_logical_key": fusion_reduction_logical_key,
        "fusion_reduction_semantic_sha256": fusion_reduction_semantic_sha256,
        "policy_version": policy_version,
        "policy_sha256": policy_sha256,
        "admitted_claim_ordinals": list(admitted_claim_ordinals),
        "reason_code": reason_code,
        "production_output_admission_semantic_sha256": (
            None
            if production_output_admission is None
            else production_output_admission.semantic_sha256
        ),
    }


# ---------------------------------------------------------------------------
# Structured retrieval projection
# ---------------------------------------------------------------------------


def _projection_mapping(value: object, *, field: str = "value") -> Mapping[str, object]:
    """Return a read-only view of a Pydantic model or mapping.

    The canonical ActionEvent preparation objects are intentionally kept out of
    the retrieval package.  Accepting their ``model_dump`` representation here
    keeps this bridge transport independent and also makes persisted replay
    records usable without importing the application models.
    """

    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        # JSON mode is intentional here: terminal publications contain
        # StrEnum values (camera IDs, evidence classes, dispositions) and the
        # EventIndex bridge must emit plain JSON-compatible structured facts,
        # not Python enum instances that happen to hash the same way.
        projected = dump(mode="json")
        if isinstance(projected, Mapping):
            return projected
    raise TypeError(f"{field} must be a mapping or a Pydantic model")


def _projection_text(value: object, *, field: str, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{field} must be a non-empty string")
        return None
    # StrEnum values (camera IDs/observations) are deliberately converted to
    # their wire spelling rather than their Python repr.
    value = getattr(value, "value", value)
    if not isinstance(value, str) or not value:
        if required:
            raise ValueError(f"{field} must be a non-empty string")
        return None
    return value


def _projection_digest(value: object, *, field: str) -> str | None:
    """Normalize an optional immutable SHA-256 identity field."""

    text = _projection_text(value, field=field, required=value is not None)
    if text is not None and re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _projection_consistent_digest(
    primary: object,
    fallback: object,
    *,
    field: str,
    required: bool = False,
) -> str | None:
    first = _projection_digest(primary, field=field)
    second = _projection_digest(fallback, field=field)
    if first is not None and second is not None and first != second:
        raise ValueError(f"{field} differs across the publication envelope")
    resolved = first if first is not None else second
    if required and resolved is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return resolved


def _projection_consistent_text(
    primary: object,
    fallback: object,
    *,
    field: str,
    required: bool = False,
) -> str | None:
    """Resolve two transport spellings while rejecting identity divergence."""

    first = _projection_text(
        primary,
        field=field,
        required=primary is not None,
    )
    second = _projection_text(
        fallback,
        field=field,
        required=fallback is not None,
    )
    if first is not None and second is not None and first != second:
        raise ValueError(f"{field} differs across the publication envelope")
    resolved = first if first is not None else second
    if required and resolved is None:
        raise ValueError(f"{field} must be a non-empty string")
    return resolved


def _projection_alias_text(
    values: Sequence[object],
    *,
    field: str,
    required: bool = False,
) -> str | None:
    """Resolve aliases (for example action_label/action_type) consistently."""

    resolved: str | None = None
    for value in values:
        if value is None:
            continue
        candidate = _projection_text(value, field=field, required=True)
        assert candidate is not None
        if resolved is not None and candidate != resolved:
            raise ValueError(f"{field} differs across the publication envelope")
        resolved = candidate
    if required and resolved is None:
        raise ValueError(f"{field} must be a non-empty string")
    return resolved


def _projection_interval(value: object) -> tuple[int, int]:
    if value is None:
        raise ValueError("event index projection requires an effective interval")
    if isinstance(value, Mapping):
        start, end = value.get("start_ns"), value.get("end_ns")
    else:
        start, end = getattr(value, "start_ns", None), getattr(value, "end_ns", None)

    def parse(raw: object, field: str) -> int:
        if isinstance(raw, bool):
            raise ValueError(f"{field} must be an integer nanosecond")
        if isinstance(raw, int):
            parsed = raw
        elif isinstance(raw, str) and re.fullmatch(r"(?:0|-?[1-9][0-9]*)", raw):
            parsed = int(raw)
        else:
            raise ValueError(f"{field} must be a canonical integer nanosecond")
        if parsed < INT64_MIN or parsed > INT64_MAX:
            raise ValueError(f"{field} must fit signed int64")
        return parsed

    start_ns = parse(start, "start_ns")
    end_ns = parse(end, "end_ns")
    if start_ns >= end_ns:
        raise ValueError("event index interval must be non-empty")
    return start_ns, end_ns


def _projection_camera_sources(payload: Mapping[str, object]) -> tuple[dict[str, str], ...]:
    raw = payload.get("camera_sources")
    if raw is None:
        statuses = payload.get("camera_statuses")
        if isinstance(statuses, Mapping):
            status_sources: list[dict[str, str]] = []
            for camera, status in sorted(statuses.items(), key=lambda item: str(item[0])):
                camera_id = _projection_text(camera, field="camera_id", required=True)
                status_text = _projection_text(status, field="camera_status", required=True)
                assert camera_id is not None and status_text is not None
                if camera_id not in CAMERA_ID_VALUES:
                    raise ValueError(f"unknown camera ID: {camera_id}")
                if any(item["camera_id"] == camera_id for item in status_sources):
                    raise ValueError(f"duplicate camera ID: {camera_id}")
                status_sources.append({"camera_id": camera_id, "status": status_text})
            return tuple(status_sources)
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("camera_sources must be a sequence")
    result: list[dict[str, str]] = []
    for source in raw:
        item = _projection_mapping(source, field="camera_source")
        camera = _projection_text(item.get("camera_id"), field="camera_id", required=True)
        status = _projection_text(
            item.get("citation_status", item.get("status")),
            field="citation_status",
            required=True,
        )
        assert camera is not None and status is not None
        if camera not in CAMERA_ID_VALUES:
            raise ValueError(f"unknown camera ID: {camera}")
        if any(item["camera_id"] == camera for item in result):
            raise ValueError(f"duplicate camera ID: {camera}")
        result.append({"camera_id": camera, "status": status})
    return tuple(result)


def canonical_event_index_revision_projection(
    publication: object,
    *,
    select: bool = True,
) -> dict[str, object]:
    """Project one terminal ActionEvent revision into an EventIndex row.

    ``publication`` may be a ``PreparedInitialActionEventRevision`` or its
    mapping/model-dump equivalent.  The projection contains structured event
    facets first; lineage/identity fields are retained as opaque metadata so a
    Postgres/Supabase adapter can enforce idempotency without changing the
    published ActionEvent bytes.  No vector or storage locator is generated.
    """

    root = _projection_mapping(publication, field="publication")
    payload_value = root.get("payload", root)
    payload = _projection_mapping(payload_value, field="payload")
    revision_value = root.get("revision", {})
    revision = _projection_mapping(revision_value, field="revision")
    lineage_value = root.get("lineage", {})
    lineage = _projection_mapping(lineage_value, field="lineage")
    subject_value = root.get("subject")
    subject = (
        _projection_mapping(subject_value, field="subject") if subject_value is not None else None
    )
    selection_value = root.get("selection", root.get("current", root))
    selection = _projection_mapping(selection_value, field="selection")

    event_id = _projection_consistent_text(
        payload.get("event_id"),
        root.get("event_id"),
        field="event_id",
        required=True,
    )
    revision_id = _projection_consistent_text(
        revision.get("revision_id"),
        root.get("event_revision_id", root.get("revision_id")),
        field="event_revision_id",
        required=True,
    )
    mcap_id = _projection_consistent_text(
        payload.get("mcap_id"),
        root.get("mcap_id"),
        field="mcap_id",
        required=True,
    )

    interval_value = payload.get("effective_interval", payload.get("interval"))
    if interval_value is None and "start_ns" in payload and "end_ns" in payload:
        interval_value = {"start_ns": payload.get("start_ns"), "end_ns": payload.get("end_ns")}
    start_ns, end_ns = _projection_interval(interval_value)
    if "start_ns" in root and "end_ns" in root and interval_value is not None:
        envelope_interval = {"start_ns": root.get("start_ns"), "end_ns": root.get("end_ns")}
        if (start_ns, end_ns) != _projection_interval(envelope_interval):
            raise ValueError("effective interval differs across the publication envelope")

    action_label = _projection_alias_text(
        (
            payload.get("action_label"),
            payload.get("action_type"),
            payload.get("label"),
            root.get("action_label"),
            root.get("action_type"),
        ),
        field="action_label",
        required=True,
    )
    observation = _projection_text(payload.get("observation"), field="observation")
    event_status = _projection_text(payload.get("event_status"), field="event_status")
    identity_disposition = _projection_text(
        payload.get("identity_disposition"),
        field="identity_disposition",
    )
    evidence_class = _projection_text(payload.get("evidence_class"), field="evidence_class")
    camera_sources = _projection_camera_sources(payload)
    camera_statuses: dict[str, str] = {item["camera_id"]: item["status"] for item in camera_sources}
    # Keep explicit camera statuses if a producer supplied them; camera source
    # citations are only a fallback and must not erase a richer QA status.
    supplied_statuses = payload.get("camera_statuses")
    if isinstance(supplied_statuses, Mapping):
        supplied: dict[str, str] = {}
        for camera, status in supplied_statuses.items():
            camera_id = _projection_text(camera, field="camera_id", required=True)
            camera_status = _projection_text(status, field="camera_status", required=True)
            assert camera_id is not None and camera_status is not None
            if camera_id not in CAMERA_ID_VALUES:
                raise ValueError(f"unknown camera ID: {camera_id}")
            if camera_id in supplied:
                raise ValueError(f"duplicate camera ID: {camera_id}")
            supplied[camera_id] = camera_status
        if any(
            camera in camera_statuses and camera_statuses[camera] != status
            for camera, status in supplied.items()
        ):
            raise ValueError("camera statuses differ between citations and structured statuses")
        camera_statuses = supplied
    usable = payload.get("usable_camera_count")
    if usable is not None:
        if isinstance(usable, bool) or not isinstance(usable, int) or not 0 <= usable <= 6:
            raise ValueError("usable_camera_count must be an integer in [0, 6]")
    else:
        usable = sum(
            str(status).upper() in {"CITED", "PASS", "SUPPORTING", "PARTIAL", "USABLE"}
            for status in camera_statuses.values()
        )

    active_hand = _projection_text(payload.get("active_hand"), field="active_hand")
    object_class_id = _projection_text(payload.get("object_class_id"), field="object_class_id")
    object_label = _projection_text(payload.get("object_label"), field="object_label")
    confidence = payload.get("confidence_value", payload.get("confidence"))
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence_value must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence_value must be finite and within [0, 1]")

    text = payload.get("text")
    if text is None:
        text = " ".join(
            item
            for item in (action_label, object_label, active_hand, observation)
            if isinstance(item, str) and item
        )
    if not isinstance(text, str):
        raise ValueError("text must be a string when present")

    # ``revision.semantic_sha256`` is the identity used by the immutable
    # revision registry.  Fall back to the projection digest only for a
    # flattened replay mapping that predates the revision envelope.
    revision_semantic = _projection_consistent_digest(
        revision.get("semantic_sha256"),
        root.get("revision_semantic_sha256"),
        field="revision_semantic_sha256",
    )
    revision_logical = _projection_consistent_text(
        revision.get("revision_logical_key"),
        root.get("revision_logical_key"),
        field="revision_logical_key",
    )
    payload_sha256 = _projection_consistent_digest(
        revision.get("payload_sha256"),
        root.get("payload_sha256"),
        field="payload_sha256",
    )
    lineage_sha256 = _projection_consistent_digest(
        revision.get("lineage_sha256"),
        root.get("lineage_sha256"),
        field="lineage_sha256",
    )
    recording_identity = _projection_consistent_digest(
        payload.get("recording_identity"),
        root.get("recording_identity"),
        field="recording_identity",
    )

    # The stable subject and immutable revision both bind the same logical
    # ActionEvent node. Reconcile every identity envelope before indexing so a
    # stale terminal wrapper cannot create a searchable row under another
    # subject.
    subject_type = _projection_consistent_text(
        None if subject is None else subject.get("node_type"),
        root.get("subject_type"),
        field="subject_type",
    )
    subject_id = _projection_consistent_text(
        None if subject is None else subject.get("node_logical_key"),
        root.get("subject_id"),
        field="subject_id",
    )
    subject_type = _projection_consistent_text(
        subject_type,
        revision.get("subject_type"),
        field="subject_type",
    )
    subject_id = _projection_consistent_text(
        subject_id,
        revision.get("subject_id"),
        field="subject_id",
    )
    if (subject_type is None) != (subject_id is None):
        raise ValueError("subject_type and subject_id must be supplied together")
    _projection_consistent_text(
        selection.get("subject_type"),
        subject_type,
        field="selection.subject_type",
    )
    _projection_consistent_text(
        selection.get("subject_id"),
        subject_id,
        field="selection.subject_id",
    )
    current_subject_value = root.get("current")
    if current_subject_value is not None:
        current_subject = _projection_mapping(current_subject_value, field="current")
        _projection_consistent_text(
            current_subject.get("subject_type"),
            subject_type,
            field="current.subject_type",
        )
        _projection_consistent_text(
            current_subject.get("subject_id"),
            subject_id,
            field="current.subject_id",
        )
    _projection_consistent_digest(
        None if subject is None else subject.get("semantic_sha256"),
        root.get("subject_semantic_sha256"),
        field="subject_semantic_sha256",
    )

    # Terminal preparation carries both the append-only SelectionDecision and
    # its replaceable CurrentSelection projection.  They are separate rows in
    # storage, but their identity facts must agree before either is projected.
    current_value = root.get("current")
    if current_value is not None:
        current = _projection_mapping(current_value, field="current")
        _projection_consistent_text(
            current.get("selected_revision_id"),
            revision_id,
            field="selected_revision_id",
        )
        _projection_consistent_text(
            current.get("selected_revision_id"),
            selection.get("selected_revision_id"),
            field="selected_revision_id",
        )
        _projection_consistent_text(
            current.get("selection_decision_id"),
            selection.get("selection_decision_id"),
            field="selection_decision_id",
        )

    # The identity current-revision reference is another terminal envelope;
    # reconcile it with the payload/revision facts when present instead of
    # silently allowing a stale reference to enter structured search.
    current_revision_value = root.get("current_revision")
    if current_revision_value is not None:
        current_revision = _projection_mapping(current_revision_value, field="current_revision")
        _projection_consistent_text(
            current_revision.get("event_id"),
            event_id,
            field="event_id",
        )
        _projection_consistent_text(
            current_revision.get("revision_id"),
            revision_id,
            field="event_revision_id",
        )
        _projection_consistent_digest(
            current_revision.get("recording_identity"),
            recording_identity,
            field="recording_identity",
        )
        _projection_consistent_text(
            current_revision.get("revision_logical_key"),
            revision_logical,
            field="revision_logical_key",
        )
        _projection_consistent_digest(
            current_revision.get("revision_semantic_sha256"),
            revision_semantic,
            field="revision_semantic_sha256",
        )
        reference_interval = current_revision.get("effective_interval")
        if reference_interval is not None and (start_ns, end_ns) != _projection_interval(
            reference_interval
        ):
            raise ValueError("effective interval differs in current-revision reference")

    assignment_value = root.get("assignment")
    if assignment_value is not None:
        assignment = _projection_mapping(assignment_value, field="assignment")
        _projection_consistent_text(assignment.get("event_id"), event_id, field="event_id")
        _projection_consistent_digest(
            assignment.get("recording_identity"),
            recording_identity,
            field="recording_identity",
        )
        _projection_consistent_text(
            assignment.get("assignment_logical_key"),
            lineage.get("identity_assignment_logical_key"),
            field="identity_assignment_logical_key",
        )
        _projection_consistent_digest(
            assignment.get("assignment_semantic_sha256"),
            lineage.get("identity_assignment_semantic_sha256"),
            field="identity_assignment_semantic_sha256",
        )

    raw_conflict_codes = payload.get("conflict_codes", ())
    if isinstance(raw_conflict_codes, (list, tuple)):
        conflict_codes: list[str] = [str(item) for item in raw_conflict_codes]
    elif raw_conflict_codes is None:
        conflict_codes = []
    else:
        raise ValueError("conflict_codes must be a sequence when present")

    record: dict[str, object] = {
        "event_id": event_id,
        "event_revision_id": revision_id,
        "mcap_id": mcap_id,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "action_type": action_label,
        "action_label": action_label,
        "active_hand": active_hand,
        "object_class_id": object_class_id,
        "object_label": object_label,
        "confidence_value": confidence,
        "camera_statuses": camera_statuses,
        "usable_camera_count": usable,
        "text": text,
        "projection_version": CANONICAL_EVENT_INDEX_PROJECTION_VERSION,
        "semantic_projection_version": CANONICAL_EVENT_INDEX_PROJECTION_VERSION,
        "recording_identity": recording_identity,
        "revision_logical_key": revision_logical,
        "revision_semantic_sha256": revision_semantic,
        "payload_sha256": payload_sha256,
        "lineage_sha256": lineage_sha256,
        "event_status": event_status,
        "observation": observation,
        "identity_disposition": identity_disposition,
        "conflict_codes": conflict_codes,
        "evidence_class": evidence_class,
        "production_eligible": payload.get("production_eligible"),
        # Preserve the complete immutable lineage envelope. The values are
        # logical keys/digests and source references, not storage locators; the
        # canonical digest therefore remains replay-stable across tuple/list
        # JSON decoding while retaining provenance for structured joins.
        "lineage": dict(lineage),
    }
    if subject is not None:
        record["subject"] = dict(subject)
    # Selection is a separate projection.  Embedding it in the revision row
    # would make a replay with a new current selection look like a mutation.
    if select:
        selection_id = _projection_text(
            selection.get("selection_decision_id"),
            field="selection_decision_id",
            required=selection.get("selection_decision_id") is not None,
        )
        selected_revision_id = _projection_consistent_text(
            selection.get("selected_revision_id"),
            revision_id,
            field="selected_revision_id",
            required=selection.get("selected_revision_id") is not None,
        )
        if selection_id is not None:
            record["selection"] = {
                "event_id": event_id,
                "selected_revision_id": selected_revision_id or revision_id,
                "selection_decision_id": selection_id,
                "selection_sequence": selection.get("selection_sequence"),
            }
    return record


# A values/row spelling mirrors the other semantic projection helpers and
# is useful for replay records that are already flattened into an index row.
canonical_event_index_projection_values = canonical_event_index_revision_projection
canonical_event_index_row_projection = canonical_event_index_revision_projection


def canonical_event_index_batch_projection(batch: object) -> dict[str, object]:
    """Project a terminal ActionEvent publication batch for EventIndex.

    The returned shape mirrors ``EventIndex.build_index`` and is intentionally
    plain JSON-compatible data, making it suitable for an async local/DB
    projector.  Empty NO_EVENTS/ABSTAINED batches produce an empty projection.
    """

    root = _projection_mapping(batch, field="batch")
    nested = root.get("action_event_publications")
    if nested is not None:
        return canonical_event_index_batch_projection(nested)
    detail = root.get("detail")
    if detail is not None:
        detail_mapping = _projection_mapping(detail, field="detail")
        nested_detail = detail_mapping.get("action_event_publications")
        if nested_detail is not None:
            return canonical_event_index_batch_projection(nested_detail)
        if any(key in detail_mapping for key in ("publications", "event_revisions", "revisions")):
            return canonical_event_index_batch_projection(detail_mapping)
    publications = root.get("publications", root.get("event_revisions", ()))
    if not isinstance(publications, (list, tuple)):
        raise ValueError("publication batch must contain a sequence")
    outcome = root.get("outcome")
    if outcome is not None:
        outcome_text = _projection_text(outcome, field="outcome", required=True)
        assert outcome_text is not None
        if outcome_text not in {"PREPARED", "NO_EVENTS", "ABSTAINED"}:
            raise ValueError(f"unsupported terminal publication outcome: {outcome_text}")
        if outcome_text == "PREPARED" and not publications:
            raise ValueError("PREPARED publication batch requires publications")
        if outcome_text in {"NO_EVENTS", "ABSTAINED"} and publications:
            raise ValueError("empty terminal outcome cannot carry publications")
        if outcome_text == "PREPARED" and (
            root.get("expected_generation") is None or root.get("expected_fence") is None
        ):
            raise ValueError("PREPARED publication batch requires generation and fence")
    revisions: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    supplied_selections = root.get("current_selections", root.get("selections", ()))
    if not isinstance(supplied_selections, (list, tuple)):
        raise ValueError("current selections must be a sequence")
    for supplied in supplied_selections:
        supplied_mapping = _projection_mapping(supplied, field="current_selection")
        candidate = dict(supplied_mapping)
        if candidate not in selections:
            selections.append(candidate)
    batch_recording_identity = _projection_digest(
        root.get("recording_identity"),
        field="recording_identity",
    )
    for item in publications:
        projected = canonical_event_index_revision_projection(item)
        row_recording_identity = projected.get("recording_identity")
        if row_recording_identity is not None:
            row_recording_identity = _projection_digest(
                row_recording_identity,
                field="recording_identity",
            )
            if (
                batch_recording_identity is not None
                and row_recording_identity != batch_recording_identity
            ):
                raise ValueError("publication batch crosses recording identity")
            if batch_recording_identity is None:
                batch_recording_identity = row_recording_identity
        selection = projected.pop("selection", None)
        revisions.append(projected)
        if isinstance(selection, Mapping):
            candidate = dict(selection)
            if candidate not in selections:
                selections.append(candidate)
    result: dict[str, object] = {
        "schema_version": "1.0",
        "projection_version": CANONICAL_EVENT_INDEX_PROJECTION_VERSION,
        "semantic_projection_version": CANONICAL_EVENT_INDEX_PROJECTION_VERSION,
        "event_revisions": revisions,
        "current_selections": selections,
    }
    if batch_recording_identity is not None:
        result["recording_identity"] = batch_recording_identity
    for key in ("outcome", "expected_generation", "expected_fence"):
        if key in root:
            result[key] = root[key]
    return result


def canonical_event_index_projection(value: object) -> dict[str, object]:
    """Dispatch to a single-revision or terminal-batch EventIndex projection."""

    root = _projection_mapping(value, field="value")
    if (
        "publications" in root
        or "event_revisions" in root
        or "action_event_publications" in root
        or "detail" in root
        or "outcome" in root
    ):
        return canonical_event_index_batch_projection(value)
    return canonical_event_index_revision_projection(value)


# Verbose aliases make the bridge discoverable to adapters without coupling
# callers to whether they project one publication or an entire terminal batch.
canonical_terminal_event_index_projection = canonical_event_index_projection
canonical_event_index_projection_batch = canonical_event_index_batch_projection


def _fusion_claim_reduction_projection(
    claim: EnrichedProviderClaim,
) -> dict[str, object]:
    """Project enriched claim content without row IDs or storage locators."""

    confidence = claim.model_reported_confidence
    return {
        "kind": claim.kind.value,
        "package_ordinal": claim.package_ordinal,
        "camera_id": None if claim.camera_id is None else claim.camera_id.value,
        "interval": None if claim.interval is None else claim.interval.model_dump(mode="json"),
        "label": claim.label,
        "observation": claim.observation.value,
        "evidence": [
            {
                "package_ordinal": item.package_ordinal,
                "package_semantic_content_sha256": item.package_semantic_content_sha256,
                "camera_id": item.camera_id.value,
                "camera_ordinal": item.camera_ordinal,
                "frame_ordinal": item.frame_ordinal,
                "aligned_timestamp_ns": str(item.aligned_timestamp_ns),
                "source_timestamp_ns": str(item.source_timestamp_ns),
                "source_artifact_sha256": item.source_artifact_sha256,
            }
            for item in claim.evidence
        ],
        "model_reported_confidence": (
            None
            if confidence is None
            else {
                "kind": confidence.kind,
                "semantics": confidence.semantics,
                "producer_type": confidence.producer_type,
                "producer_version": confidence.producer_version,
                "value": confidence.value,
            }
        ),
        "conflict_codes": list(claim.conflict_codes),
    }


def _fusion_claim_reduction_digest(claim: EnrichedProviderClaim) -> str:
    return semantic_sha256(_fusion_claim_reduction_projection(claim))


def _stable_uuid(namespace: str, *parts: object) -> str:
    material = ":".join(str(item) for item in parts)
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{material}"))


__all__ = [
    "CANONICAL_EVENT_INDEX_PROJECTION_VERSION",
    "CANONICAL_EXECUTION_POLICY_SEMANTIC_PROJECTION_VERSION",
    "CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION",
    "CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE",
    "CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION",
    "CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE",
    "canonical_event_index_batch_projection",
    "canonical_event_index_projection",
    "canonical_event_index_projection_batch",
    "canonical_event_index_projection_values",
    "canonical_event_index_revision_projection",
    "canonical_event_index_row_projection",
    "canonical_execution_policy_projection",
    "canonical_execution_policy_projection_values",
    "canonical_fusion_reduction_projection",
    "canonical_output_decision_projection",
    "canonical_root_window_projection_values",
    "canonical_terminal_event_index_projection",
]
