-- Robata canonical PostgreSQL/Supabase authority foundation.
--
-- This migration creates namespaces and operational ownership boundaries only.
-- Runtime roles are provisioned outside this file because Supabase projects do
-- not permit an application migration to create arbitrary login roles.

CREATE SCHEMA IF NOT EXISTS robata_canonical;
CREATE SCHEMA IF NOT EXISTS robata_projection;
CREATE SCHEMA IF NOT EXISTS robata_ops;

REVOKE ALL ON SCHEMA robata_canonical FROM PUBLIC;
REVOKE ALL ON SCHEMA robata_projection FROM PUBLIC;
REVOKE ALL ON SCHEMA robata_ops FROM PUBLIC;

-- Every canonical table uses this transaction-local setting for tenant isolation.
-- Application and worker connections set it before issuing any query; a missing
-- setting deliberately yields NULL and therefore fails NOT NULL / RLS checks.
CREATE OR REPLACE FUNCTION robata_canonical.current_tenant_id()
RETURNS text
LANGUAGE sql
STABLE
AS $$
    SELECT NULLIF(current_setting('robata.tenant_id', true), '')
$$;
CREATE TABLE IF NOT EXISTS robata_ops.canonical_authority_state (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    authority_version text NOT NULL,
    required_tenant_setting text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

INSERT INTO robata_ops.canonical_authority_state (
    singleton,
    authority_version,
    required_tenant_setting
)
VALUES (true, 'robata-postgres-canonical-v1', 'robata.tenant_id')
ON CONFLICT (singleton) DO NOTHING;

COMMENT ON SCHEMA robata_canonical IS
    'Authoritative scheduler, completion, evidence, identity, review, and outbox state.';
COMMENT ON SCHEMA robata_projection IS
    'Derived read and vector projections; never canonical authority.';
COMMENT ON SCHEMA robata_ops IS
    'Migration and operational audit metadata; not browser-addressable.';
