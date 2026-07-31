-- Robata canonical PostgreSQL/Supabase authority: completion, evidence, barriers, and outbox.
--
-- Canonical payloads are exact RFC-8785 bytes in bytea. They must never be
-- recreated from a database projection or used with a different hash basis.
--
-- This migration intentionally does not alter a published Robata wire schema.

CREATE SCHEMA IF NOT EXISTS robata_canonical;


CREATE OR REPLACE FUNCTION robata_canonical.current_tenant_id()
RETURNS text
LANGUAGE sql
STABLE
AS $$
    SELECT NULLIF(current_setting('robata.tenant_id', true), '')
$$;

CREATE OR REPLACE FUNCTION robata_canonical.reject_immutable_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'robata canonical facts are append-only'
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$;

-- Completion aggregate and immutable identity/publication facts.
CREATE TABLE robata_canonical.primary_runs (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    run_id text NOT NULL,
    recording_identity text NOT NULL,
    mcap_id text NOT NULL,
    pipeline_version text NOT NULL,
    config_sha256 text NOT NULL,
    started_at text NOT NULL,
    primary_status text NOT NULL CHECK (primary_status IN ('RUNNING', 'SUCCEEDED', 'NO_EVENTS')),
    completed_at text,
    run_version integer NOT NULL CHECK (run_version IN (0, 1)),
    command_sha256 text,
    run_json bytea NOT NULL,
    run_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, run_id),
    UNIQUE (tenant_id, run_id, recording_identity),
    CHECK (
        (primary_status = 'RUNNING' AND completed_at IS NULL
            AND run_version = 0 AND command_sha256 IS NULL)
        OR
        (primary_status IN ('SUCCEEDED', 'NO_EVENTS') AND completed_at IS NOT NULL
            AND run_version = 1 AND command_sha256 IS NOT NULL)
    )
);

CREATE TABLE robata_canonical.event_registry_partitions (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    recording_identity text NOT NULL,
    generation integer NOT NULL CHECK (generation >= 0),
    fence integer NOT NULL CHECK (fence >= 1),
    PRIMARY KEY (tenant_id, recording_identity)
);

CREATE TABLE robata_canonical.stable_event_identities (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    event_id text NOT NULL,
    recording_identity text NOT NULL,
    payload_json bytea NOT NULL,
    payload_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, event_id),
    UNIQUE (tenant_id, recording_identity, event_id),
    FOREIGN KEY (tenant_id, recording_identity)
        REFERENCES robata_canonical.event_registry_partitions (tenant_id, recording_identity)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.event_identity_assignments (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    assignment_logical_key text NOT NULL,
    recording_identity text NOT NULL,
    event_hypothesis_logical_key text NOT NULL,
    identity_policy_version text NOT NULL,
    identity_policy_sha256 text NOT NULL,
    event_id text NOT NULL,
    payload_json bytea NOT NULL,
    payload_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, assignment_logical_key),
    UNIQUE (tenant_id, recording_identity, assignment_logical_key),
    UNIQUE (
        tenant_id, recording_identity, event_hypothesis_logical_key,
        identity_policy_version, identity_policy_sha256
    ),
    FOREIGN KEY (tenant_id, recording_identity, event_id)
        REFERENCES robata_canonical.stable_event_identities (
            tenant_id, recording_identity, event_id
        ) ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.event_identity_relations (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    relation_logical_key text NOT NULL,
    recording_identity text NOT NULL,
    assignment_logical_key text NOT NULL,
    from_event_id text NOT NULL,
    to_event_id text NOT NULL,
    payload_json bytea NOT NULL,
    payload_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, relation_logical_key),
    FOREIGN KEY (tenant_id, recording_identity, assignment_logical_key)
        REFERENCES robata_canonical.event_identity_assignments (
            tenant_id, recording_identity, assignment_logical_key
        ) ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, recording_identity, from_event_id)
        REFERENCES robata_canonical.stable_event_identities (
            tenant_id, recording_identity, event_id
        ) ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, recording_identity, to_event_id)
        REFERENCES robata_canonical.stable_event_identities (
            tenant_id, recording_identity, event_id
        ) ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (from_event_id <> to_event_id)
);

CREATE TABLE robata_canonical.action_event_publications (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    subject_id text NOT NULL,
    recording_identity text NOT NULL,
    event_id text NOT NULL,
    revision_logical_key text NOT NULL,
    selection_decision_logical_key text NOT NULL,
    publication_json bytea NOT NULL,
    publication_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, subject_id),
    UNIQUE (tenant_id, event_id),
    UNIQUE (tenant_id, revision_logical_key),
    UNIQUE (tenant_id, selection_decision_logical_key),
    FOREIGN KEY (tenant_id, recording_identity, event_id)
        REFERENCES robata_canonical.stable_event_identities (
            tenant_id, recording_identity, event_id
        ) ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.detailed_results (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    artifact_id text NOT NULL,
    exact_bytes_sha256 text NOT NULL,
    byte_count bigint NOT NULL CHECK (byte_count > 0),
    schema_id text NOT NULL,
    schema_version text NOT NULL,
    schema_artifact_id text NOT NULL,
    schema_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    PRIMARY KEY (tenant_id, artifact_id),
    UNIQUE (tenant_id, exact_bytes_sha256)
);

CREATE TABLE robata_canonical.primary_completions (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    run_id text NOT NULL,
    command_sha256 text NOT NULL,
    command_json bytea NOT NULL,
    command_json_sha256 text NOT NULL,
    committed_json bytea NOT NULL,
    committed_json_sha256 text NOT NULL,
    detailed_result_artifact_id text NOT NULL,
    PRIMARY KEY (tenant_id, run_id),
    FOREIGN KEY (tenant_id, run_id)
        REFERENCES robata_canonical.primary_runs (tenant_id, run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, detailed_result_artifact_id)
        REFERENCES robata_canonical.detailed_results (tenant_id, artifact_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.primary_outbox (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    outbox_id text NOT NULL,
    completion_run_id text NOT NULL,
    recording_identity text NOT NULL,
    outbox_ordinal integer NOT NULL CHECK (outbox_ordinal >= 0),
    assignment_logical_key text NOT NULL,
    payload_json bytea NOT NULL,
    payload_json_sha256 text NOT NULL,
    delivered_at timestamptz,
    PRIMARY KEY (tenant_id, outbox_id),
    UNIQUE (tenant_id, completion_run_id, outbox_ordinal),
    FOREIGN KEY (tenant_id, completion_run_id)
        REFERENCES robata_canonical.primary_completions (tenant_id, run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, completion_run_id, recording_identity)
        REFERENCES robata_canonical.primary_runs (tenant_id, run_id, recording_identity)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, assignment_logical_key)
        REFERENCES robata_canonical.event_identity_assignments (tenant_id, assignment_logical_key)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX primary_outbox_pending_idx
    ON robata_canonical.primary_outbox (tenant_id, delivered_at, completion_run_id, outbox_ordinal);

CREATE TABLE robata_canonical.primary_outbox_deliveries (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    outbox_id text NOT NULL,
    status text NOT NULL CHECK (
        status IN ('PENDING', 'LEASED', 'RETRY_WAIT', 'DELIVERED', 'DEAD_LETTER')
    ),
    attempt_count integer NOT NULL CHECK (attempt_count >= 0),
    lease_epoch bigint NOT NULL CHECK (lease_epoch >= 0),
    fencing_token text,
    claimed_by text,
    lease_expires_at timestamptz,
    next_attempt_at timestamptz NOT NULL,
    retry_policy_version text NOT NULL,
    max_attempts integer NOT NULL CHECK (max_attempts > 0),
    base_delay_seconds double precision NOT NULL CHECK (base_delay_seconds >= 0),
    max_delay_seconds double precision NOT NULL CHECK (max_delay_seconds >= base_delay_seconds),
    last_error text,
    delivered_at timestamptz,
    dead_lettered_at timestamptz,
    PRIMARY KEY (tenant_id, outbox_id),
    FOREIGN KEY (tenant_id, outbox_id)
        REFERENCES robata_canonical.primary_outbox (tenant_id, outbox_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (
        status <> 'LEASED' OR (
            fencing_token IS NOT NULL AND claimed_by IS NOT NULL AND lease_expires_at IS NOT NULL
        )
    ),
    CHECK ((status = 'DELIVERED') = (delivered_at IS NOT NULL)),
    CHECK ((status = 'DEAD_LETTER') = (dead_lettered_at IS NOT NULL))
);

CREATE INDEX primary_outbox_delivery_claim_idx
    ON robata_canonical.primary_outbox_deliveries (
        tenant_id, status, next_attempt_at, outbox_id
    );

-- Inference evidence. All model-shaped records are retained in canonical bytes;
-- indexed identity fields are duplicated only to enforce graph edges efficiently.
CREATE TABLE robata_canonical.inference_intents (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    inference_id text NOT NULL,
    logical_invocation_id text NOT NULL,
    request_id text NOT NULL,
    contract_schema_id text NOT NULL,
    contract_version text NOT NULL,
    contract_artifact_id text NOT NULL,
    contract_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, inference_id),
    UNIQUE (tenant_id, request_id),
    UNIQUE (tenant_id, inference_id, request_id)
);

CREATE TABLE robata_canonical.raw_provider_responses (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    artifact_id text NOT NULL,
    inference_id text NOT NULL,
    request_id text NOT NULL,
    provider_request_id text NOT NULL,
    exact_bytes_sha256 text NOT NULL,
    media_type text NOT NULL,
    byte_count bigint NOT NULL CHECK (byte_count > 0),
    raw_bytes bytea NOT NULL,
    PRIMARY KEY (tenant_id, artifact_id),
    UNIQUE (tenant_id, artifact_id, inference_id),
    UNIQUE (tenant_id, request_id),
    FOREIGN KEY (tenant_id, inference_id, request_id)
        REFERENCES robata_canonical.inference_intents (tenant_id, inference_id, request_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);


CREATE TABLE robata_canonical.model_inference_terminals (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    inference_id text NOT NULL,
    logical_invocation_id text NOT NULL,
    request_id text NOT NULL,
    status text NOT NULL CHECK (
        status IN ('SUCCEEDED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'INVALID_OUTPUT')
    ),
    shadow smallint NOT NULL CHECK (shadow IN (0, 1)),
    output_valid smallint NOT NULL CHECK (output_valid IN (0, 1)),
    raw_artifact_id text,
    contract_schema_id text NOT NULL,
    contract_version text NOT NULL,
    contract_artifact_id text NOT NULL,
    contract_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, inference_id),
    FOREIGN KEY (tenant_id, inference_id, request_id)
        REFERENCES robata_canonical.inference_intents (tenant_id, inference_id, request_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, raw_artifact_id, inference_id)
        REFERENCES robata_canonical.raw_provider_responses (tenant_id, artifact_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.raw_provider_artifacts (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    artifact_id text NOT NULL,
    inference_id text NOT NULL,
    request_id text NOT NULL,
    exact_bytes_sha256 text NOT NULL,
    byte_count bigint NOT NULL CHECK (byte_count > 0),
    media_type text NOT NULL,
    provider_request_id text NOT NULL,
    provider text NOT NULL,
    model_name text NOT NULL,
    model_version text NOT NULL,
    created_at text NOT NULL,
    contract_schema_id text NOT NULL,
    contract_version text NOT NULL,
    contract_artifact_id text NOT NULL,
    contract_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, artifact_id),
    UNIQUE (tenant_id, artifact_id, inference_id),
    FOREIGN KEY (tenant_id, artifact_id, inference_id)
        REFERENCES robata_canonical.raw_provider_responses (tenant_id, artifact_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, inference_id)
        REFERENCES robata_canonical.model_inference_terminals (tenant_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.inference_attempt_selections (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    logical_invocation_id text NOT NULL,
    policy_version text NOT NULL,
    selection_reason text NOT NULL,
    selection_id text NOT NULL,
    inference_id text NOT NULL,
    contract_schema_id text NOT NULL,
    contract_version text NOT NULL,
    contract_artifact_id text NOT NULL,
    contract_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, logical_invocation_id, policy_version),
    UNIQUE (tenant_id, selection_id, inference_id),
    FOREIGN KEY (tenant_id, inference_id)
        REFERENCES robata_canonical.model_inference_terminals (tenant_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.parsed_provider_claims (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    artifact_id text NOT NULL,
    inference_id text NOT NULL,
    raw_artifact_id text NOT NULL,
    semantic_sha256 text NOT NULL,
    provider_claim_schema_sha256 text NOT NULL,
    parser_version text NOT NULL,
    contract_schema_id text NOT NULL,
    contract_version text NOT NULL,
    contract_artifact_id text NOT NULL,
    contract_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, artifact_id),
    UNIQUE (tenant_id, raw_artifact_id, provider_claim_schema_sha256, parser_version),
    UNIQUE (tenant_id, artifact_id, inference_id, raw_artifact_id),
    FOREIGN KEY (tenant_id, inference_id)
        REFERENCES robata_canonical.model_inference_terminals (tenant_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, raw_artifact_id, inference_id)
        REFERENCES robata_canonical.raw_provider_artifacts (tenant_id, artifact_id, inference_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.selected_attempt_outputs (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    selection_id text NOT NULL,
    inference_id text NOT NULL,
    parsed_artifact_id text NOT NULL,
    raw_artifact_id text NOT NULL,
    output_sha256 text NOT NULL,
    contract_schema_id text NOT NULL,
    contract_version text NOT NULL,
    contract_artifact_id text NOT NULL,
    contract_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, selection_id),
    UNIQUE (tenant_id, selection_id, inference_id, output_sha256),
    UNIQUE (tenant_id, output_sha256),
    FOREIGN KEY (tenant_id, selection_id, inference_id)
        REFERENCES robata_canonical.inference_attempt_selections (
            tenant_id, selection_id, inference_id
        ) ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, parsed_artifact_id, inference_id, raw_artifact_id)
        REFERENCES robata_canonical.parsed_provider_claims (
            tenant_id, artifact_id, inference_id, raw_artifact_id
        ) ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.enriched_provider_outputs (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    artifact_id text NOT NULL,
    enrichment_logical_key text NOT NULL,
    semantic_sha256 text NOT NULL,
    selection_id text NOT NULL,
    inference_id text NOT NULL,
    selected_output_sha256 text NOT NULL,
    contract_schema_id text NOT NULL,
    contract_version text NOT NULL,
    contract_artifact_id text NOT NULL,
    contract_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, artifact_id),
    UNIQUE (tenant_id, enrichment_logical_key),
    UNIQUE (tenant_id, semantic_sha256),
    FOREIGN KEY (tenant_id, selection_id, inference_id, selected_output_sha256)
        REFERENCES robata_canonical.selected_attempt_outputs (
            tenant_id, selection_id, inference_id, output_sha256
        ) ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.calibration_artifacts (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    artifact_id text NOT NULL,
    artifact_sha256 text NOT NULL,
    score_family text NOT NULL,
    fitting_method text NOT NULL CHECK (
        fitting_method IN ('IDENTITY', 'PLATT_LOGISTIC', 'ISOTONIC_LINEAR')
    ),
    applicability_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, artifact_id),
    UNIQUE (tenant_id, artifact_sha256),
    UNIQUE (tenant_id, artifact_id, artifact_sha256)
);

CREATE TABLE robata_canonical.inference_calibration_associations (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    association_id text NOT NULL,
    selection_id text NOT NULL,
    inference_id text NOT NULL,
    score_family text NOT NULL,
    outcome text NOT NULL CHECK (
        outcome IN (
            'APPLIED', 'RAW_FALLBACK_MISSING_ARTIFACT',
            'RAW_FALLBACK_INAPPLICABLE', 'RAW_FALLBACK_UNAVAILABLE_SCORE'
        )
    ),
    raw_score double precision CHECK (raw_score IS NULL OR (raw_score >= 0.0 AND raw_score <= 1.0)),
    calibrated_probability double precision CHECK (
        calibrated_probability IS NULL
        OR (calibrated_probability >= 0.0 AND calibrated_probability <= 1.0)
    ),
    deterministic_inputs_sha256 text NOT NULL,
    calibration_artifact_id text,
    calibration_artifact_sha256 text,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, association_id),
    UNIQUE (tenant_id, selection_id, score_family),
    CHECK (
        (calibration_artifact_id IS NULL AND calibration_artifact_sha256 IS NULL)
        OR (calibration_artifact_id IS NOT NULL AND calibration_artifact_sha256 IS NOT NULL)
    ),
    FOREIGN KEY (tenant_id, selection_id, inference_id)
        REFERENCES robata_canonical.inference_attempt_selections (
            tenant_id, selection_id, inference_id
        ) ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, calibration_artifact_id, calibration_artifact_sha256)
        REFERENCES robata_canonical.calibration_artifacts (
            tenant_id, artifact_id, artifact_sha256
        ) ON UPDATE RESTRICT ON DELETE RESTRICT
);

-- Generic barriers and inference-call reductions.
CREATE TABLE robata_canonical.barrier_definitions (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    barrier_id text NOT NULL,
    logical_key text NOT NULL,
    expected_member_count integer NOT NULL CHECK (expected_member_count >= 0),
    empty_semantics text NOT NULL,
    reduction_policy text NOT NULL,
    status text NOT NULL CHECK (status IN ('OPEN', 'CLOSED', 'FAILED')),
    required_success_count integer NOT NULL CHECK (
        required_success_count >= 0 AND required_success_count <= expected_member_count
    ),
    max_degraded_failures integer NOT NULL CHECK (
        max_degraded_failures >= 0 AND max_degraded_failures <= expected_member_count
    ),
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, barrier_id),
    UNIQUE (tenant_id, logical_key)
);

CREATE TABLE robata_canonical.barrier_states (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    barrier_id text NOT NULL,
    state_version bigint NOT NULL CHECK (state_version >= 0),
    completed_members integer NOT NULL CHECK (completed_members >= 0),
    pending_members integer NOT NULL CHECK (pending_members >= 0),
    failed_members integer NOT NULL CHECK (failed_members >= 0),
    status text NOT NULL CHECK (status IN ('OPEN', 'CLOSED', 'FAILED')),
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, barrier_id),
    FOREIGN KEY (tenant_id, barrier_id)
        REFERENCES robata_canonical.barrier_definitions (tenant_id, barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.barrier_members (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    barrier_id text NOT NULL,
    work_item_id text NOT NULL,
    criticality text NOT NULL CHECK (criticality IN ('REQUIRED', 'DEGRADABLE', 'OPTIONAL')),
    outcome text NOT NULL CHECK (
        outcome IN (
            'SUCCEEDED', 'SKIPPED_POLICY', 'SKIPPED_NOT_NEEDED',
            'FAILED', 'CANCELLED', 'EXPIRED', 'QUARANTINED', 'INCOMPLETE'
        )
    ),
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, barrier_id, work_item_id),
    FOREIGN KEY (tenant_id, barrier_id)
        REFERENCES robata_canonical.barrier_definitions (tenant_id, barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.inference_call_barrier_definitions (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    barrier_id text NOT NULL,
    barrier_semantic_sha256 text NOT NULL,
    barrier_logical_key text NOT NULL,
    input_plan_semantic_sha256 text NOT NULL,
    call_plan_sha256 text NOT NULL,
    part_count integer NOT NULL CHECK (part_count > 0),
    reduction_policy text NOT NULL,
    reduction_policy_version text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, barrier_id),
    FOREIGN KEY (tenant_id, barrier_id)
        REFERENCES robata_canonical.barrier_definitions (tenant_id, barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.inference_call_part_completions (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    barrier_id text NOT NULL,
    part_ordinal integer NOT NULL CHECK (part_ordinal >= 0),
    part_count integer NOT NULL CHECK (part_count > 0),
    part_semantic_sha256 text NOT NULL,
    part_logical_key text NOT NULL,
    part_idempotency_key text NOT NULL,
    completion_id text NOT NULL,
    completion_semantic_sha256 text NOT NULL,
    inference_id text NOT NULL,
    logical_invocation_id text NOT NULL,
    status text NOT NULL CHECK (
        status IN ('SUCCEEDED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'INVALID_OUTPUT')
    ),
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, barrier_id, part_ordinal),
    UNIQUE (tenant_id, barrier_id, part_semantic_sha256),
    UNIQUE (tenant_id, completion_id),
    FOREIGN KEY (tenant_id, barrier_id)
        REFERENCES robata_canonical.inference_call_barrier_definitions (tenant_id, barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.inference_call_reductions (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    barrier_id text NOT NULL,
    reduction_id text NOT NULL,
    reduction_semantic_sha256 text NOT NULL,
    normalized_output_sha256 text NOT NULL,
    payload_json bytea NOT NULL,
    payload_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, barrier_id),
    UNIQUE (tenant_id, reduction_id),
    FOREIGN KEY (tenant_id, barrier_id)
        REFERENCES robata_canonical.inference_call_barrier_definitions (tenant_id, barrier_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
-- Immutable facts are enforced at the database boundary.  Delivery and barrier
-- state are intentionally the only mutable records; their adapters use fences
-- and conditional updates instead of broad updates.
DO $$
DECLARE
    fact_table text;
BEGIN
    FOREACH fact_table IN ARRAY ARRAY[
        'stable_event_identities', 'event_identity_assignments', 'event_identity_relations',
        'action_event_publications', 'detailed_results', 'primary_completions',
        'inference_intents', 'raw_provider_responses', 'model_inference_terminals',
        'raw_provider_artifacts', 'inference_attempt_selections', 'parsed_provider_claims',
        'selected_attempt_outputs', 'enriched_provider_outputs', 'calibration_artifacts',
        'inference_calibration_associations', 'barrier_definitions',
        'inference_call_barrier_definitions', 'inference_call_part_completions',
        'inference_call_reductions'
    ]
    LOOP
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON robata_canonical.%I '
            'FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation()',
            fact_table || '_append_only', fact_table
        );
    END LOOP;
END;
$$;

CREATE OR REPLACE FUNCTION robata_canonical.primary_outbox_delivery_immutable_guard()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.outbox_id <> NEW.outbox_id THEN
        RAISE EXCEPTION 'outbox delivery identity is immutable'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF OLD.delivered_at IS NOT NULL AND NEW.delivered_at IS DISTINCT FROM OLD.delivered_at THEN
        RAISE EXCEPTION 'outbox delivery acknowledgement is monotonic'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER primary_outbox_delivery_guard
BEFORE UPDATE ON robata_canonical.primary_outbox_deliveries
FOR EACH ROW EXECUTE FUNCTION robata_canonical.primary_outbox_delivery_immutable_guard();
CREATE OR REPLACE FUNCTION robata_canonical.primary_outbox_immutable_guard()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'primary outbox facts cannot be deleted'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF OLD.outbox_id <> NEW.outbox_id
       OR OLD.completion_run_id <> NEW.completion_run_id
       OR OLD.recording_identity <> NEW.recording_identity
       OR OLD.outbox_ordinal <> NEW.outbox_ordinal
       OR OLD.assignment_logical_key <> NEW.assignment_logical_key
       OR OLD.payload_json <> NEW.payload_json
       OR OLD.payload_json_sha256 <> NEW.payload_json_sha256 THEN
        RAISE EXCEPTION 'primary outbox facts are append-only'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF OLD.delivered_at IS NOT NULL AND NEW.delivered_at IS DISTINCT FROM OLD.delivered_at THEN
        RAISE EXCEPTION 'outbox delivery acknowledgement is monotonic'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER primary_outbox_immutable_guard
BEFORE UPDATE OR DELETE ON robata_canonical.primary_outbox
FOR EACH ROW EXECUTE FUNCTION robata_canonical.primary_outbox_immutable_guard();

-- Deny by default.  The deployment migration supplies concrete role grants.
ALTER TABLE robata_canonical.primary_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.event_registry_partitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.stable_event_identities ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.event_identity_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.event_identity_relations ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.action_event_publications ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.detailed_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.primary_completions ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.primary_outbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.primary_outbox_deliveries ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.inference_intents ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.raw_provider_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.model_inference_terminals ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.raw_provider_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.inference_attempt_selections ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.parsed_provider_claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.selected_attempt_outputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.enriched_provider_outputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.calibration_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.inference_calibration_associations ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.barrier_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.barrier_states ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.barrier_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.inference_call_barrier_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.inference_call_part_completions ENABLE ROW LEVEL SECURITY;
ALTER TABLE robata_canonical.inference_call_reductions ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE
    protected_table text;
BEGIN
    FOREACH protected_table IN ARRAY ARRAY[
        'primary_runs', 'event_registry_partitions', 'stable_event_identities',
        'event_identity_assignments', 'event_identity_relations', 'action_event_publications',
        'detailed_results', 'primary_completions', 'primary_outbox', 'primary_outbox_deliveries',
        'inference_intents', 'raw_provider_responses', 'model_inference_terminals',
        'raw_provider_artifacts', 'inference_attempt_selections', 'parsed_provider_claims',
        'selected_attempt_outputs', 'enriched_provider_outputs', 'calibration_artifacts',
        'inference_calibration_associations', 'barrier_definitions', 'barrier_states',
        'barrier_members', 'inference_call_barrier_definitions',
        'inference_call_part_completions', 'inference_call_reductions'
    ]
    LOOP
        EXECUTE format(
            'ALTER TABLE robata_canonical.%I FORCE ROW LEVEL SECURITY',
            protected_table
        );
    END LOOP;
END;
$$;

DO $$
DECLARE
    protected_table text;
BEGIN
    FOREACH protected_table IN ARRAY ARRAY[
        'primary_runs', 'event_registry_partitions', 'stable_event_identities',
        'event_identity_assignments', 'event_identity_relations', 'action_event_publications',
        'detailed_results', 'primary_completions', 'primary_outbox', 'primary_outbox_deliveries',
        'inference_intents', 'raw_provider_responses', 'model_inference_terminals',
        'raw_provider_artifacts', 'inference_attempt_selections', 'parsed_provider_claims',
        'selected_attempt_outputs', 'enriched_provider_outputs', 'calibration_artifacts',
        'inference_calibration_associations', 'barrier_definitions', 'barrier_states',
        'barrier_members', 'inference_call_barrier_definitions',
        'inference_call_part_completions', 'inference_call_reductions'
    ]
    LOOP
        EXECUTE format(
            'CREATE POLICY %I ON robata_canonical.%I '
            'USING (tenant_id = robata_canonical.current_tenant_id()) '
            'WITH CHECK (tenant_id = robata_canonical.current_tenant_id())',
            protected_table || '_tenant_isolation', protected_table
        );
    END LOOP;
END;
$$;
