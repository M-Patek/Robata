-- Robata canonical PostgreSQL/Supabase authority: logical nodes, immutable
-- revisions, review work, and the committed-run read side.
--
-- The public node, revision, selection, and review documents remain their
-- registered canonical JSON contracts.  This migration only supplies their
-- tenant-bound durable representation; it does not redefine any wire shape,
-- logical key, semantic digest, idempotency key, or fence.

CREATE SCHEMA IF NOT EXISTS robata_canonical;

-- Run-independent logical-node facts and immutable processing-run memberships.
CREATE TABLE robata_canonical.logical_nodes (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    node_type text NOT NULL,
    node_logical_key text NOT NULL,
    schema_version text NOT NULL CHECK (schema_version = '1.0'),
    key_namespace text NOT NULL,
    semantic_sha256 text NOT NULL,
    identity_policy_version text NOT NULL,
    node_json bytea NOT NULL,
    node_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, node_type, node_logical_key)
);

CREATE TABLE robata_canonical.processing_run_nodes (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    run_id text NOT NULL,
    node_type text NOT NULL,
    node_logical_key text NOT NULL,
    role text NOT NULL,
    schema_version text NOT NULL CHECK (schema_version = '1.0'),
    disposition text NOT NULL CHECK (
        disposition IN ('CREATED', 'REUSED', 'INVALIDATED', 'OBSERVED')
    ),
    first_work_item_id text NOT NULL,
    attached_at text NOT NULL,
    membership_json bytea NOT NULL,
    membership_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, run_id, node_type, node_logical_key, role),
    FOREIGN KEY (tenant_id, node_type, node_logical_key)
        REFERENCES robata_canonical.logical_nodes (tenant_id, node_type, node_logical_key)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX processing_run_nodes_node_idx
    ON robata_canonical.processing_run_nodes (
        tenant_id, node_type, node_logical_key, run_id, role
    );

CREATE UNIQUE INDEX processing_run_nodes_creator_idx
    ON robata_canonical.processing_run_nodes (tenant_id, node_type, node_logical_key)
    WHERE disposition = 'CREATED';

-- Immutable revisions and their deterministic, replaceable current projection.
CREATE TABLE robata_canonical.immutable_node_revisions (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    subject_type text NOT NULL,
    subject_id text NOT NULL,
    revision_id text NOT NULL,
    schema_version text NOT NULL CHECK (schema_version = '1.0'),
    revision_key_namespace text NOT NULL,
    revision_logical_key text NOT NULL,
    semantic_sha256 text NOT NULL,
    payload_sha256 text NOT NULL,
    lineage_sha256 text NOT NULL,
    status_at_publication text NOT NULL,
    eligibility_at_publication text NOT NULL CHECK (
        eligibility_at_publication IN ('ELIGIBLE', 'INELIGIBLE')
    ),
    revision_policy_version text NOT NULL,
    supersedes_revision_id text,
    supersedes_revision_logical_key text,
    published_at text NOT NULL,
    revision_json bytea NOT NULL,
    revision_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, subject_type, subject_id, revision_id),
    UNIQUE (tenant_id, revision_id),
    UNIQUE (tenant_id, subject_type, subject_id, revision_logical_key),
    UNIQUE (
        tenant_id, subject_type, subject_id, revision_id, revision_logical_key
    ),
    FOREIGN KEY (tenant_id, subject_type, subject_id)
        REFERENCES robata_canonical.logical_nodes (tenant_id, node_type, node_logical_key)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (
        tenant_id,
        subject_type,
        subject_id,
        supersedes_revision_id,
        supersedes_revision_logical_key
    ) REFERENCES robata_canonical.immutable_node_revisions (
        tenant_id,
        subject_type,
        subject_id,
        revision_id,
        revision_logical_key
    ) ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (
        (supersedes_revision_id IS NULL AND supersedes_revision_logical_key IS NULL)
        OR (
            supersedes_revision_id IS NOT NULL
            AND supersedes_revision_logical_key IS NOT NULL
        )
    )
);

CREATE TABLE robata_canonical.selection_decisions (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    subject_type text NOT NULL,
    subject_id text NOT NULL,
    selection_decision_id text NOT NULL,
    schema_version text NOT NULL CHECK (schema_version = '1.0'),
    selection_key_namespace text NOT NULL,
    selection_decision_logical_key text NOT NULL,
    semantic_sha256 text NOT NULL,
    selected_revision_id text NOT NULL,
    selected_revision_logical_key text NOT NULL,
    previous_selection_decision_id text,
    previous_selection_decision_logical_key text,
    selection_sequence integer NOT NULL CHECK (selection_sequence >= 1),
    selection_policy_version text NOT NULL,
    projection_version text NOT NULL,
    selected_at text NOT NULL,
    decision_json bytea NOT NULL,
    decision_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, subject_type, subject_id, selection_decision_id),
    UNIQUE (tenant_id, selection_decision_id),
    UNIQUE (tenant_id, subject_type, subject_id, selection_decision_logical_key),
    UNIQUE (tenant_id, subject_type, subject_id, selection_sequence),
    UNIQUE (
        tenant_id,
        subject_type,
        subject_id,
        selection_decision_id,
        selection_decision_logical_key
    ),
    UNIQUE (
        tenant_id,
        subject_type,
        subject_id,
        selection_decision_id,
        selected_revision_id,
        selection_policy_version,
        projection_version,
        selected_at
    ),
    FOREIGN KEY (
        tenant_id,
        subject_type,
        subject_id,
        selected_revision_id,
        selected_revision_logical_key
    ) REFERENCES robata_canonical.immutable_node_revisions (
        tenant_id,
        subject_type,
        subject_id,
        revision_id,
        revision_logical_key
    ) ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (
        tenant_id,
        subject_type,
        subject_id,
        previous_selection_decision_id,
        previous_selection_decision_logical_key
    ) REFERENCES robata_canonical.selection_decisions (
        tenant_id,
        subject_type,
        subject_id,
        selection_decision_id,
        selection_decision_logical_key
    ) ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (
        (
            previous_selection_decision_id IS NULL
            AND previous_selection_decision_logical_key IS NULL
        )
        OR (
            previous_selection_decision_id IS NOT NULL
            AND previous_selection_decision_logical_key IS NOT NULL
        )
    )
);

CREATE UNIQUE INDEX selection_decisions_genesis_idx
    ON robata_canonical.selection_decisions (tenant_id, subject_type, subject_id)
    WHERE previous_selection_decision_id IS NULL;

CREATE UNIQUE INDEX selection_decisions_successor_idx
    ON robata_canonical.selection_decisions (
        tenant_id, subject_type, subject_id, previous_selection_decision_id
    ) WHERE previous_selection_decision_id IS NOT NULL;

CREATE TABLE robata_canonical.current_selections (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    subject_type text NOT NULL,
    subject_id text NOT NULL,
    schema_version text NOT NULL CHECK (schema_version = '1.0'),
    selected_revision_id text NOT NULL,
    selection_decision_id text NOT NULL,
    selection_policy_version text NOT NULL,
    projection_version text NOT NULL,
    selected_at text NOT NULL,
    current_json bytea NOT NULL,
    current_json_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, subject_type, subject_id),
    FOREIGN KEY (tenant_id, subject_type, subject_id, selected_revision_id)
        REFERENCES robata_canonical.immutable_node_revisions (
            tenant_id, subject_type, subject_id, revision_id
        ) ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (
        tenant_id,
        subject_type,
        subject_id,
        selection_decision_id,
        selected_revision_id,
        selection_policy_version,
        projection_version,
        selected_at
    ) REFERENCES robata_canonical.selection_decisions (
        tenant_id,
        subject_type,
        subject_id,
        selection_decision_id,
        selected_revision_id,
        selection_policy_version,
        projection_version,
        selected_at
    ) ON UPDATE RESTRICT ON DELETE RESTRICT
);

-- Fenced review work.  The definition is immutable; only independently
-- visible lease and completion state may transition.
CREATE TABLE robata_canonical.review_tasks (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    review_task_id text NOT NULL,
    request_id text NOT NULL,
    task_semantic_sha256 text NOT NULL,
    priority integer NOT NULL CHECK (priority >= 0),
    requested_at_ns bigint NOT NULL,
    due_at_ns bigint NOT NULL CHECK (due_at_ns > requested_at_ns),
    task_json bytea NOT NULL,
    task_exact_sha256 text NOT NULL,
    status text NOT NULL CHECK (status IN ('PENDING', 'LEASED', 'COMPLETED')),
    lease_fence bigint NOT NULL DEFAULT 0 CHECK (lease_fence >= 0),
    attempt_count bigint NOT NULL DEFAULT 0 CHECK (attempt_count = lease_fence),
    lease_owner text,
    lease_expires_at_ns bigint,
    completed_annotation_id text,
    PRIMARY KEY (tenant_id, review_task_id),
    UNIQUE (tenant_id, request_id),
    UNIQUE (tenant_id, task_semantic_sha256),
    CHECK (
        (status = 'PENDING' AND lease_owner IS NULL AND lease_expires_at_ns IS NULL
            AND completed_annotation_id IS NULL)
        OR (status = 'LEASED' AND lease_owner IS NOT NULL AND lease_expires_at_ns IS NOT NULL
            AND completed_annotation_id IS NULL AND lease_fence > 0)
        OR (status = 'COMPLETED' AND lease_owner IS NULL AND lease_expires_at_ns IS NULL
            AND completed_annotation_id IS NOT NULL)
    )
);

CREATE INDEX review_tasks_schedule_idx
    ON robata_canonical.review_tasks (
        tenant_id, priority, due_at_ns, requested_at_ns, review_task_id
    ) WHERE status <> 'COMPLETED';

CREATE TABLE robata_canonical.review_annotations (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    annotation_id text NOT NULL,
    review_task_id text NOT NULL,
    lease_fence bigint NOT NULL CHECK (lease_fence > 0),
    annotation_semantic_sha256 text NOT NULL,
    annotation_json bytea NOT NULL,
    annotation_exact_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, annotation_id),
    UNIQUE (tenant_id, annotation_semantic_sha256),
    UNIQUE (tenant_id, review_task_id, lease_fence),
    FOREIGN KEY (tenant_id, review_task_id)
        REFERENCES robata_canonical.review_tasks (tenant_id, review_task_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE robata_canonical.review_reopen_commands (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    reopen_id text NOT NULL,
    review_task_id text NOT NULL,
    expected_annotation_id text NOT NULL,
    command_semantic_sha256 text NOT NULL,
    command_json bytea NOT NULL,
    command_exact_sha256 text NOT NULL,
    PRIMARY KEY (tenant_id, reopen_id),
    FOREIGN KEY (tenant_id, review_task_id)
        REFERENCES robata_canonical.review_tasks (tenant_id, review_task_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, expected_annotation_id)
        REFERENCES robata_canonical.review_annotations (tenant_id, annotation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE OR REPLACE FUNCTION robata_canonical.reject_review_task_definition_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
       OR NEW.review_task_id IS DISTINCT FROM OLD.review_task_id
       OR NEW.request_id IS DISTINCT FROM OLD.request_id
       OR NEW.task_semantic_sha256 IS DISTINCT FROM OLD.task_semantic_sha256
       OR NEW.priority IS DISTINCT FROM OLD.priority
       OR NEW.requested_at_ns IS DISTINCT FROM OLD.requested_at_ns
       OR NEW.due_at_ns IS DISTINCT FROM OLD.due_at_ns
       OR NEW.task_json IS DISTINCT FROM OLD.task_json
       OR NEW.task_exact_sha256 IS DISTINCT FROM OLD.task_exact_sha256 THEN
        RAISE EXCEPTION 'review task definition is immutable'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER logical_nodes_no_update
BEFORE UPDATE ON robata_canonical.logical_nodes
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER logical_nodes_no_delete
BEFORE DELETE ON robata_canonical.logical_nodes
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER processing_run_nodes_no_update
BEFORE UPDATE ON robata_canonical.processing_run_nodes
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER processing_run_nodes_no_delete
BEFORE DELETE ON robata_canonical.processing_run_nodes
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER immutable_node_revisions_no_update
BEFORE UPDATE ON robata_canonical.immutable_node_revisions
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER immutable_node_revisions_no_delete
BEFORE DELETE ON robata_canonical.immutable_node_revisions
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER selection_decisions_no_update
BEFORE UPDATE ON robata_canonical.selection_decisions
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER selection_decisions_no_delete
BEFORE DELETE ON robata_canonical.selection_decisions
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER review_tasks_immutable_definition
BEFORE UPDATE ON robata_canonical.review_tasks
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_review_task_definition_mutation();

CREATE TRIGGER review_tasks_no_delete
BEFORE DELETE ON robata_canonical.review_tasks
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER review_annotations_no_update
BEFORE UPDATE ON robata_canonical.review_annotations
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER review_annotations_no_delete
BEFORE DELETE ON robata_canonical.review_annotations
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER review_reopen_commands_no_update
BEFORE UPDATE ON robata_canonical.review_reopen_commands
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE TRIGGER review_reopen_commands_no_delete
BEFORE DELETE ON robata_canonical.review_reopen_commands
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

-- Tenant context is a transaction-local setting installed by
-- PostgresCanonicalAuthority.  FORCE prevents the owner role from quietly
-- bypassing RLS; deployment administrators still retain their native role.
DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'logical_nodes',
        'processing_run_nodes',
        'immutable_node_revisions',
        'selection_decisions',
        'current_selections',
        'review_tasks',
        'review_annotations',
        'review_reopen_commands'
    ] LOOP
        EXECUTE format('ALTER TABLE robata_canonical.%I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE robata_canonical.%I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format(
            'CREATE POLICY %I ON robata_canonical.%I '
            || 'USING (tenant_id = robata_canonical.current_tenant_id()) '
            || 'WITH CHECK (tenant_id = robata_canonical.current_tenant_id())',
            table_name || '_tenant_isolation',
            table_name
        );
    END LOOP;
END;
$$;
