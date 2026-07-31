-- PostgreSQL/R2 immutable raw-provider evidence mirror lifecycle.
--
-- PostgreSQL remains the exact-byte recovery authority.  R2 is a mandatory,
-- tenant-scoped immutable mirror for every raw provider response inserted after
-- this migration.  The receipt is staged before provider I/O and advances only
-- once the deterministic R2 key/version has been exactly verified.

CREATE SCHEMA IF NOT EXISTS robata_canonical;

CREATE TABLE robata_canonical.raw_provider_r2_artifact_receipts (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    artifact_id text NOT NULL,
    inference_id text NOT NULL,
    request_id text NOT NULL,
    provider_request_id text NOT NULL,
    exact_bytes_sha256 text NOT NULL,
    byte_count bigint NOT NULL CHECK (byte_count > 0),
    media_type text NOT NULL,
    payload_bytes bytea NOT NULL,
    logical_key text NOT NULL,
    object_uri text NOT NULL,
    object_version text NOT NULL,
    object_etag text,
    r2_config_sha256 text NOT NULL,
    state text NOT NULL CHECK (state IN ('STAGED', 'COMMITTED')),
    staged_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    committed_at timestamptz,
    PRIMARY KEY (tenant_id, artifact_id),
    UNIQUE (tenant_id, request_id),
    UNIQUE (tenant_id, object_uri, object_version),
    FOREIGN KEY (tenant_id, inference_id, request_id)
        REFERENCES robata_canonical.inference_intents (tenant_id, inference_id, request_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (
        (state = 'STAGED' AND committed_at IS NULL)
        OR (state = 'COMMITTED' AND committed_at IS NOT NULL)
    )
);

CREATE TABLE robata_canonical.raw_provider_r2_artifact_observations (
    tenant_id text NOT NULL DEFAULT robata_canonical.current_tenant_id(),
    observation_id text NOT NULL,
    artifact_id text NOT NULL,
    observation_kind text NOT NULL CHECK (observation_kind IN (
        'PUT_VERIFIED', 'MISSING', 'PARTIAL', 'CONFLICT', 'CORRUPT', 'PROVIDER_ERROR'
    )),
    exact_bytes_sha256 text NOT NULL,
    byte_count bigint NOT NULL CHECK (byte_count > 0),
    media_type text NOT NULL,
    observed_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, observation_id),
    FOREIGN KEY (tenant_id, artifact_id)
        REFERENCES robata_canonical.raw_provider_r2_artifact_receipts (tenant_id, artifact_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX raw_provider_r2_artifact_receipts_state_idx
    ON robata_canonical.raw_provider_r2_artifact_receipts (tenant_id, state, staged_at, artifact_id);

CREATE INDEX raw_provider_r2_artifact_observations_artifact_idx
    ON robata_canonical.raw_provider_r2_artifact_observations (
        tenant_id, artifact_id, observed_at, observation_id
    );

-- Existing historical raw rows are preserved.  NOT VALID avoids falsely
-- asserting they were mirrored before this additive lifecycle existed, while
-- PostgreSQL still enforces the foreign key for every future insert.
ALTER TABLE robata_canonical.raw_provider_responses
    ADD CONSTRAINT raw_provider_responses_r2_receipt_fk
    FOREIGN KEY (tenant_id, artifact_id)
    REFERENCES robata_canonical.raw_provider_r2_artifact_receipts (tenant_id, artifact_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT
    NOT VALID;

CREATE OR REPLACE FUNCTION robata_canonical.raw_provider_r2_receipt_guard()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.state <> 'STAGED'
           OR NEW.committed_at IS NOT NULL
           OR NEW.object_etag IS NOT NULL THEN
            RAISE EXCEPTION 'raw provider R2 artifact receipts must begin staged'
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        RETURN NEW;
    END IF;

    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'raw provider R2 artifact receipts cannot be deleted'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    IF OLD.tenant_id IS DISTINCT FROM NEW.tenant_id
       OR OLD.artifact_id IS DISTINCT FROM NEW.artifact_id
       OR OLD.inference_id IS DISTINCT FROM NEW.inference_id
       OR OLD.request_id IS DISTINCT FROM NEW.request_id
       OR OLD.provider_request_id IS DISTINCT FROM NEW.provider_request_id
       OR OLD.exact_bytes_sha256 IS DISTINCT FROM NEW.exact_bytes_sha256
       OR OLD.byte_count IS DISTINCT FROM NEW.byte_count
       OR OLD.media_type IS DISTINCT FROM NEW.media_type
       OR OLD.payload_bytes IS DISTINCT FROM NEW.payload_bytes
       OR OLD.logical_key IS DISTINCT FROM NEW.logical_key
       OR OLD.object_uri IS DISTINCT FROM NEW.object_uri
       OR OLD.object_version IS DISTINCT FROM NEW.object_version
       OR OLD.r2_config_sha256 IS DISTINCT FROM NEW.r2_config_sha256
       OR OLD.staged_at IS DISTINCT FROM NEW.staged_at THEN
        RAISE EXCEPTION 'raw provider R2 artifact receipt plan is immutable'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    IF OLD.state = 'STAGED'
       AND NEW.state = 'COMMITTED'
       AND OLD.committed_at IS NULL
       AND NEW.committed_at IS NOT NULL THEN
        RETURN NEW;
    END IF;

    IF OLD.state = NEW.state
       AND OLD.object_etag IS NOT DISTINCT FROM NEW.object_etag
       AND OLD.committed_at IS NOT DISTINCT FROM NEW.committed_at THEN
        RETURN NEW;
    END IF;

    RAISE EXCEPTION 'raw provider R2 artifact receipt state transition is invalid'
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$;

CREATE TRIGGER raw_provider_r2_artifact_receipt_guard
BEFORE INSERT OR UPDATE OR DELETE ON robata_canonical.raw_provider_r2_artifact_receipts
FOR EACH ROW EXECUTE FUNCTION robata_canonical.raw_provider_r2_receipt_guard();

CREATE TRIGGER raw_provider_r2_artifact_observation_append_only
BEFORE UPDATE OR DELETE ON robata_canonical.raw_provider_r2_artifact_observations
FOR EACH ROW EXECUTE FUNCTION robata_canonical.reject_immutable_mutation();

CREATE OR REPLACE FUNCTION robata_canonical.require_committed_raw_provider_r2_receipt()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    receipt record;
BEGIN
    SELECT * INTO receipt
    FROM robata_canonical.raw_provider_r2_artifact_receipts
    WHERE tenant_id = NEW.tenant_id AND artifact_id = NEW.artifact_id;

    IF NOT FOUND OR receipt.state <> 'COMMITTED' THEN
        RAISE EXCEPTION 'raw provider response requires a committed immutable R2 receipt'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    IF receipt.inference_id <> NEW.inference_id
       OR receipt.request_id <> NEW.request_id
       OR receipt.provider_request_id <> NEW.provider_request_id
       OR receipt.exact_bytes_sha256 <> NEW.exact_bytes_sha256
       OR receipt.byte_count <> NEW.byte_count
       OR receipt.media_type <> NEW.media_type
       OR receipt.payload_bytes <> NEW.raw_bytes THEN
        RAISE EXCEPTION 'raw provider response differs from its committed immutable R2 receipt'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER raw_provider_response_requires_r2_receipt
BEFORE INSERT ON robata_canonical.raw_provider_responses
FOR EACH ROW EXECUTE FUNCTION robata_canonical.require_committed_raw_provider_r2_receipt();

DO $$
DECLARE
    protected_table text;
BEGIN
    FOREACH protected_table IN ARRAY ARRAY[
        'raw_provider_r2_artifact_receipts',
        'raw_provider_r2_artifact_observations'
    ]
    LOOP
        EXECUTE format('ALTER TABLE robata_canonical.%I ENABLE ROW LEVEL SECURITY', protected_table);
        EXECUTE format('ALTER TABLE robata_canonical.%I FORCE ROW LEVEL SECURITY', protected_table);
        EXECUTE format(
            'CREATE POLICY %I ON robata_canonical.%I '
            || 'USING (tenant_id = robata_canonical.current_tenant_id()) '
            || 'WITH CHECK (tenant_id = robata_canonical.current_tenant_id())',
            protected_table || '_tenant_isolation', protected_table
        );
    END LOOP;
END;
$$;
