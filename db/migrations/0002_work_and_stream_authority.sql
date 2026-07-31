-- PostgreSQL canonical work, stream, and capture authority.
--
-- Apply only after 0001_canonical_schema.sql using a migration-only role. Every
-- control-plane fact is tenant-bound through the transaction-local
-- robata.tenant_id setting. Runtime roles receive ordinary table grants only;
-- FORCE ROW LEVEL SECURITY keeps the table owner from accidentally bypassing
-- tenant isolation. Identifiers and canonical RFC3339 timestamps remain text
-- because they participate in immutable local wire and replay contracts. JSON
-- payloads are bytea because canonical bytes, not database re-encoding, are
-- authoritative.

CREATE SCHEMA IF NOT EXISTS robata_canonical;

CREATE OR REPLACE FUNCTION robata_canonical.current_tenant_id()
RETURNS TEXT
LANGUAGE sql
STABLE
AS $$
    SELECT NULLIF(current_setting('robata.tenant_id', true), '')
$$;

CREATE TABLE robata_canonical.work_items (
    tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    schema_version TEXT NOT NULL CHECK (schema_version = '1.0'),
    work_item_id TEXT NOT NULL,
    work_logical_key TEXT NOT NULL,
    run_id TEXT NOT NULL,
    mcap_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    config_digest TEXT NOT NULL,
    priority INTEGER NOT NULL CHECK (priority >= 0),
    sla_deadline_at TEXT,
    execution_expiry_at TEXT,
    max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
    trace_id TEXT,
    created_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'PLANNED', 'READY', 'LEASED', 'RUNNING', 'RETRY_WAIT', 'SUCCEEDED',
        'FAILED_PERMANENT', 'SKIPPED_POLICY', 'SKIPPED_NOT_NEEDED', 'CANCELLED',
        'EXPIRED', 'INVALIDATED'
    )),
    cancel_requested SMALLINT NOT NULL CHECK (cancel_requested IN (0, 1)),
    lease_epoch INTEGER NOT NULL CHECK (lease_epoch >= 0),
    fencing_token TEXT,
    leased_by TEXT,
    lease_expires_at TEXT,
    attempt INTEGER NOT NULL CHECK (attempt >= 0),
    retry_not_before_at TEXT,
    terminal_reason_code TEXT,
    terminal_reason_detail TEXT,
    result_reference TEXT,
    result_sha256 TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    row_version INTEGER NOT NULL CHECK (row_version >= 0),
    PRIMARY KEY (tenant_id, work_item_id),
    UNIQUE (tenant_id, work_logical_key),
    CHECK ((result_reference IS NULL) = (result_sha256 IS NULL)),
    CHECK (
        (state IN ('LEASED', 'RUNNING')) =
        (fencing_token IS NOT NULL AND leased_by IS NOT NULL AND lease_expires_at IS NOT NULL)
    )
);

CREATE TABLE robata_canonical.work_dependencies (
    tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    dependency_id TEXT NOT NULL,
    downstream_work_item_id TEXT NOT NULL,
    upstream_work_item_id TEXT NOT NULL,
    criticality TEXT NOT NULL,
    PRIMARY KEY (tenant_id, dependency_id),
    FOREIGN KEY (tenant_id, downstream_work_item_id)
        REFERENCES robata_canonical.work_items(tenant_id, work_item_id) ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, upstream_work_item_id)
        REFERENCES robata_canonical.work_items(tenant_id, work_item_id) ON DELETE RESTRICT,
    UNIQUE (tenant_id, downstream_work_item_id, upstream_work_item_id),
    CHECK (downstream_work_item_id <> upstream_work_item_id)
);

CREATE TABLE robata_canonical.work_attempts (
    tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    work_item_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    lease_epoch INTEGER NOT NULL CHECK (lease_epoch >= 1),
    fencing_token TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    outcome TEXT NOT NULL,
    error_code TEXT,
    error_detail TEXT,
    PRIMARY KEY (tenant_id, work_item_id, attempt_number),
    FOREIGN KEY (tenant_id, work_item_id)
        REFERENCES robata_canonical.work_items(tenant_id, work_item_id) ON DELETE RESTRICT,
    UNIQUE (tenant_id, work_item_id, lease_epoch),
    UNIQUE (tenant_id, fencing_token)
);

CREATE INDEX work_items_state_retry
    ON robata_canonical.work_items(tenant_id, state, retry_not_before_at, created_at, work_item_id);
CREATE INDEX work_items_due_expiry
    ON robata_canonical.work_items(tenant_id, execution_expiry_at, work_item_id)
    WHERE completed_at IS NULL AND execution_expiry_at IS NOT NULL;
CREATE INDEX work_items_due_lease
    ON robata_canonical.work_items(tenant_id, state, lease_expires_at, work_item_id)
    WHERE lease_expires_at IS NOT NULL;
CREATE INDEX work_items_run_state
    ON robata_canonical.work_items(tenant_id, run_id, state, work_item_id);
CREATE INDEX work_items_run_dispatch_order
    ON robata_canonical.work_items(
        tenant_id, run_id, state, priority DESC,
        (CASE WHEN sla_deadline_at IS NULL THEN 1 ELSE 0 END),
        sla_deadline_at, created_at, work_item_id
    );
CREATE INDEX work_items_dispatch_order
    ON robata_canonical.work_items(
        tenant_id, state, priority DESC,
        (CASE WHEN sla_deadline_at IS NULL THEN 1 ELSE 0 END),
        sla_deadline_at, created_at, work_item_id
    );
CREATE INDEX work_dependencies_upstream
    ON robata_canonical.work_dependencies(
        tenant_id, upstream_work_item_id, downstream_work_item_id
    );
CREATE INDEX work_dependencies_downstream
    ON robata_canonical.work_dependencies(
        tenant_id, downstream_work_item_id, upstream_work_item_id
    );

CREATE TABLE robata_canonical.stream_plans (
    tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    plan_key TEXT NOT NULL,
    plan_json BYTEA NOT NULL,
    source_subject_json BYTEA NOT NULL,
    composition_config_json BYTEA NOT NULL,
    planner_eos_sha256 TEXT,
    seal_json BYTEA,
    terminal_closure_json BYTEA,
    export_manifest_sha256 TEXT,
    export_member_count INTEGER,
    PRIMARY KEY (tenant_id, plan_key)
);

CREATE TABLE robata_canonical.expected_windows (
    tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    plan_key TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    declaration_json BYTEA NOT NULL,
    window_json BYTEA NOT NULL,
    terminal_member_json BYTEA,
    PRIMARY KEY (tenant_id, plan_key, ordinal),
    FOREIGN KEY (tenant_id, plan_key)
        REFERENCES robata_canonical.stream_plans(tenant_id, plan_key) ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.stream_work_plans (
    tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    work_item_id TEXT NOT NULL,
    work_logical_key TEXT NOT NULL,
    plan_key TEXT NOT NULL,
    expected_ordinal INTEGER,
    role_order INTEGER NOT NULL,
    stage TEXT NOT NULL,
    plan_json BYTEA NOT NULL,
    publication_state TEXT NOT NULL CHECK (publication_state IN ('GATED', 'PENDING', 'PUBLISHED')),
    terminal_evidence_json BYTEA,
    pending_terminal_json BYTEA,
    pending_lease_epoch INTEGER,
    pending_fencing_token TEXT,
    PRIMARY KEY (tenant_id, work_item_id),
    UNIQUE (tenant_id, work_logical_key),
    FOREIGN KEY (tenant_id, plan_key)
        REFERENCES robata_canonical.stream_plans(tenant_id, plan_key) ON DELETE RESTRICT,
    CHECK ((pending_terminal_json IS NULL) = (pending_lease_epoch IS NULL)),
    CHECK ((pending_terminal_json IS NULL) = (pending_fencing_token IS NULL))
);

CREATE INDEX stream_work_plan_order
    ON robata_canonical.stream_work_plans(
        tenant_id, plan_key, expected_ordinal, role_order, work_item_id
    );
CREATE INDEX stream_work_pending_publication
    ON robata_canonical.stream_work_plans(
        tenant_id, plan_key, publication_state, expected_ordinal, role_order
    );

CREATE TABLE robata_canonical.stream_backpressure_controllers (
    tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    plan_key TEXT NOT NULL,
    controller_key TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    owner_fence INTEGER NOT NULL CHECK (owner_fence > 0),
    state_json BYTEA NOT NULL,
    PRIMARY KEY (tenant_id, plan_key, controller_key),
    FOREIGN KEY (tenant_id, plan_key)
        REFERENCES robata_canonical.stream_plans(tenant_id, plan_key) ON DELETE RESTRICT
);
CREATE INDEX stream_backpressure_controller_partition
    ON robata_canonical.stream_backpressure_controllers(
        tenant_id, controller_key, policy_version, plan_key
    );

CREATE TABLE robata_canonical.capture_authority_metadata (
    tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    singleton SMALLINT NOT NULL CHECK (singleton = 1),
    capture_authority_id TEXT NOT NULL,
    capture_authority_epoch INTEGER NOT NULL CHECK (capture_authority_epoch >= 1),
    capture_assignment_policy_version TEXT NOT NULL,
    next_acquisition_sequence BIGINT NOT NULL CHECK (next_acquisition_sequence >= 1),
    PRIMARY KEY (tenant_id, singleton)
);

CREATE TABLE robata_canonical.capture_authority_receipts (
    tenant_id TEXT NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    receipt_slot TEXT NOT NULL,
    acquisition_sequence BIGINT NOT NULL CHECK (acquisition_sequence >= 1),
    request_json BYTEA NOT NULL,
    subject_json BYTEA NOT NULL,
    capture_scope_digest TEXT NOT NULL,
    PRIMARY KEY (tenant_id, receipt_slot),
    UNIQUE (tenant_id, acquisition_sequence),
    UNIQUE (tenant_id, capture_scope_digest)
);

DO $$
DECLARE
    protected_table TEXT;
BEGIN
    FOREACH protected_table IN ARRAY ARRAY[
        'work_items', 'work_dependencies', 'work_attempts', 'stream_plans',
        'expected_windows', 'stream_work_plans', 'stream_backpressure_controllers',
        'capture_authority_metadata', 'capture_authority_receipts'
    ]
    LOOP
        EXECUTE format(
            'ALTER TABLE robata_canonical.%I ENABLE ROW LEVEL SECURITY',
            protected_table
        );
        EXECUTE format(
            'ALTER TABLE robata_canonical.%I FORCE ROW LEVEL SECURITY',
            protected_table
        );
        EXECUTE format(
            'CREATE POLICY %I ON robata_canonical.%I '
            'USING (tenant_id = robata_canonical.current_tenant_id()) '
            'WITH CHECK (tenant_id = robata_canonical.current_tenant_id())',
            protected_table || '_tenant_isolation', protected_table
        );
    END LOOP;
END;
$$;