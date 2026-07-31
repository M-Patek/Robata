# PostgreSQL/Supabase Production Composition

Robata's canonical authority is PostgreSQL, not SQLite and not pgvector. The
canonical scheduler, stream state, completion, evidence, barriers, logical
nodes, review queue, and outbox live in `robata_canonical`. `pgvector` is a
derived retrieval projection only. R2 stores immutable external artifact bytes
through `R2ObjectStore`; an object location is never a content identity.

## Scope and boundary

`compose.production.yaml` is a production admission sequence, not a substitute
for a source-specific worker implementation. It builds one image and exposes
four explicit one-shot gates:

1. `canonical-migrate` applies immutable canonical migrations with the DDL role.
2. `canonical-postgres-verify` proves the app and worker PostgreSQL roles,
   migration ledger, minimum grants, non-ownership, active `FORCE RLS`, and
   transaction-local tenant isolation.
3. `optional-adapter-preflight` constructs R2 and RunPod clients and verifies
   the real pgvector backend and its worker role.
4. `canonical-runtime-verify` loads the reviewed bootstrap and release
   artifacts, then constructs `build_production_canonical_runtime` using real
   PostgreSQL, R2, pgvector, and RunPod adapters without dispatching inference.

The existing local MCAP composition remains an explicit SQLite plus
`OfflineFixtureVisionAdapter` conformance path. It must not be repointed at
production credentials. A source-specific production task process must receive
the runtime returned by `build_production_canonical_runtime`; this repository
will not claim that a generic local fixture runner is a serving worker.

## Runtime roles

Provision three distinct PostgreSQL logins outside the application migration:

- `canonical_migrator`: applies reviewed DDL and owns canonical tables.
- `canonical_worker`: reads and transitions canonical work, evidence, review,
  and outbox rows.
- `canonical_app`: read-only canonical API/read-model access.

Both runtime roles must be `NOSUPERUSER NOBYPASSRLS NOINHERIT`, must not own
canonical tables, and must be distinct from the migrator. The exact role names
are deployment inputs, but the following grant shape is required after running
the migrations, using a database administrator or the table owner:

```sql
GRANT USAGE ON SCHEMA robata_canonical, robata_ops
    TO canonical_worker, canonical_app;
GRANT EXECUTE ON FUNCTION robata_canonical.current_tenant_id()
    TO canonical_worker, canonical_app;

REVOKE ALL ON ALL TABLES IN SCHEMA robata_canonical
    FROM canonical_worker, canonical_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA robata_canonical
    TO canonical_worker;
GRANT SELECT ON ALL TABLES IN SCHEMA robata_canonical
    TO canonical_app;
GRANT SELECT ON robata_ops.schema_migrations,
    robata_ops.canonical_authority_state
    TO canonical_worker, canonical_app;

ALTER DEFAULT PRIVILEGES FOR ROLE canonical_migrator
    IN SCHEMA robata_canonical
    GRANT SELECT, INSERT, UPDATE ON TABLES TO canonical_worker;
ALTER DEFAULT PRIVILEGES FOR ROLE canonical_migrator
    IN SCHEMA robata_canonical
    GRANT SELECT ON TABLES TO canonical_app;
```

Do not grant `DELETE`, `TRUNCATE`, table ownership, a provider service role, or
`BYPASSRLS` to either runtime login. Every canonical migration enables and
forces RLS. Its policy binds `tenant_id` to the transaction-local
`robata.tenant_id` setting; the runtime sets this via `SET LOCAL`, so a missing
tenant cannot read or write tenant-bound state.

## Configuration and mounted artifacts

Copy [`.env.production.example`](../../.env.production.example) to untracked
`.env.production`, or inject the same variables through the deployment secret
manager. The example is intentionally tracked; `/secrets/` is ignored and must
never be committed. Compose requires four host files:

- `CANONICAL_POSTGRES_CA_FILE`: CA certificate mounted at
  `/run/secrets/postgres-ca.pem`.
- `ROBATA_PRODUCTION_RUNTIME_CONFIG_FILE`: reviewed bootstrap JSON mounted at
  `/run/secrets/production-runtime.json`.
- `ROBATA_PRIMARY_QUALIFICATION_REPORT_FILE`: exact qualification-report bytes
  mounted at `/run/secrets/primary-qualification-report.json`.
- `ROBATA_PRIMARY_RELEASE_DECISION_FILE`: exact release-decision bytes mounted
  at `/run/secrets/primary-release-decision.json`.

`ROBATA_PRODUCTION_RUNTIME_CONFIG` must be
`/run/secrets/production-runtime.json` inside the container. Compose always
passes one dotenv source into the containers. By default it is
`.env.production`; an external secret manager must materialize a short-lived
mode-restricted dotenv file and set `ROBATA_RUNTIME_ENV_FILE` to that path. Pass
that same path to `docker compose --env-file` so Compose can resolve the four
host-file mounts without leaking values into the checked-in YAML. Confirm the
CA file is readable by container UID `10001`; local Compose implementations can
ignore requested secret ownership or mode values.

The runtime bootstrap is a strict JSON document with these top-level fields:

```text
schema_version
primary_binding
primary_capabilities
primary_retry_policy
primary_route
capture_authority
outbox_retry_policy_version
outbox_max_attempts
outbox_base_delay_seconds
outbox_max_delay_seconds
primary_parser_version
qualification_report_file
release_decision_file
```

`primary_binding` is the pinned `ProductionPrimaryRunPodBinding`; it contains
the endpoint configuration, handler image SHA-256, and capability snapshot
SHA-256. `primary_route` contains the authoritative `ProductionRoute` and the
exact byte SHA-256 values for the mounted qualification report and release
decision. Both file paths in the bootstrap must use the container paths above.

The release-decision file has a closed contract. It does not contain its own
SHA-256, avoiding a self-referential digest. It must contain the reviewed facts
below, with `deployment` serialized as the route's `ModelDeployment`:

```json
{
  "schema_version": "robata-primary-route-release-decision-v1",
  "decision": "APPROVED",
  "route_id": "production-mage-primary",
  "policy_version": "1.0",
  "deployment": { "...": "exact ProductionRoute deployment fields" },
  "qualification_report_ref": "r2://robata/.../qualification.json",
  "qualification_report_sha256": "<64 lowercase hex>",
  "primary_binding_sha256": "<ProductionPrimaryRunPodBinding.configuration_sha256>",
  "handler_image_sha256": "<64 lowercase hex>"
}
```

The production gate verifies the exact bytes of both mounted artifacts, then
parses this decision and matches its route, deployment, qualification digest,
full binding digest, and handler image digest. Mutating an artifact or swapping
an image, endpoint, model version, capability snapshot, or release decision
fails closed before work is admitted.

## Deployment sequence

Run these commands from the repository after installing the reviewed image
inputs. They are intentionally separate so a failure is attributable and leaves
no hidden background worker running.

```powershell
docker compose --env-file .env.production -f compose.production.yaml build
docker compose --env-file .env.production -f compose.production.yaml run --rm canonical-migrate
docker compose --env-file .env.production -f compose.production.yaml run --rm canonical-postgres-verify
docker compose --env-file .env.production -f compose.production.yaml run --rm optional-adapter-preflight
docker compose --env-file .env.production -f compose.production.yaml run --rm canonical-runtime-verify
```

The first command's migration role is DDL-only. The second command makes
read-only checks with app and worker credentials. The third command sends no
RunPod inference and performs no R2 write, but it does connect to pgvector and
prove the reviewed backend/RLS worker configuration. The fourth command builds
the concrete production adapter graph and emits only non-secret identifiers.

After all four gates pass, launch the real source-specific task process with the
same mounted bootstrap and environment. That process must construct the runtime
through the shared production root before it accepts a source or claims work;
it must not invoke the local SQLite composition. For tomorrow's real-sample
exercise, retain the four gate outputs, the exact input manifest, R2 receipts,
RunPod response evidence, and the Mage/Qwen qualification artifacts.

## R2 lifecycle

`R2ObjectStore.put` writes one versioned physical key with conditional create,
then performs a verified HEAD. If a provider response is lost, it re-reads the
same immutable key and accepts it only when SHA-256, byte count, media type, and
object version all match. `get` rechecks exact bytes. Use one R2 prefix per
environment and treat deletion as a retention workflow, not normal pipeline
cleanup.

## External release gates

Local verification proves software wiring, not cloud behavior. Before admitting
real traffic, retain evidence for the following:

- PostgreSQL/Supabase TLS, role grants, RLS, backup, and restore rehearsal.
- R2 write/read/recovery against the actual bucket and retention policy.
- RunPod handler image, capability snapshot, endpoint contract, and real
  control/candidate workload qualification.
- Independent primary-route release review tied to the exact artifacts mounted
  into `canonical-runtime-verify`.