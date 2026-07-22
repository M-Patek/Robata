# Normative Standards and Specifications

## 1. RFC 8785 — JSON Canonicalization Scheme (JCS)

**Issuer**: IETF (Internet Engineering Task Force)  
**Reference**: Rundgren, A., Jordan, B. and Erdtman, S. "JSON Canonicalization
Scheme (JCS)." *RFC 8785*, IETF, June 2020.  
**URL**: https://www.rfc-editor.org/rfc/rfc8785

### Specification Summary

RFC 8785 defines a deterministic serialization of JSON values:
- Object keys are sorted by Unicode code point.
- Numbers use the shortest IEEE 754 round-trip representation.
- Strings use `\uXXXX` escapes for control characters.
- No insignificant whitespace.

The output is uniquely determined by the logical value, independent of the
language runtime, library version, or key-insertion order.

### Robata Application

- `contracts/hashing.py` — `canonical_json_bytes()` implements JCS; all
  semantic identity preimages pass through this function before SHA-256.
- `conformance/rational-grid-canonicalization-v1.json` — every test vector
  pins the RFC 8785 canonical bytes as a lowercase hex string plus their
  SHA-256 digest.
- `scripts/verify_rational_grid_vectors.mjs` — the Node.js verifier
  independently implements JCS using BigInt (no Python dependency) and
  verifies the same vectors.

**Why this matters**: Without a canonicalization standard, two systems that
agree on the logical content of a JSON object may produce different digests
because their runtimes serialize keys in different orders. RFC 8785 eliminates
that ambiguity.

---

## 2. IEEE 754-2008 — Floating-Point Arithmetic

**Issuer**: IEEE (Institute of Electrical and Electronics Engineers)  
**Reference**: IEEE. *IEEE Standard for Floating-Point Arithmetic*. IEEE
Std 754-2008. IEEE, 2008.  
**Successor**: IEEE Std 754-2019.

### Relevant Rules

- **Round half to even (HALF_EVEN / Banker's rounding)**: when a value is
  exactly halfway between two representable values, round to the value whose
  last digit is even. This eliminates the cumulative upward bias of
  round-half-up over large datasets.
- **Exact rational arithmetic**: integer-numerator / integer-denominator
  fractions are representable exactly if computed with integer arithmetic;
  binary floating-point introduces representation error for values like 1/3.

### Robata Application

- `alignment/rational_time.py` — `round_half_even()` implements the IEEE 754
  HALF_EVEN rule using pure integer arithmetic (no floating-point involved).
  Timestamps are stored as integer nanoseconds; fractional results arise only
  when dividing by a non-integer frame rate.
- `sampling/grid.py` — `SamplingGrid` uses the same integer-rational arithmetic
  to place grid points; HALF_EVEN rounding is applied when mapping a grid point
  to the nearest representable nanosecond timestamp.
- Cross-language consistency: the Python and Node.js conformance verifiers both
  implement HALF_EVEN using integer (BigInt) arithmetic, ensuring no
  floating-point divergence between runtimes.

---

## 3. ISO 8601 / RFC 3339 — Date and Time Representation

**References**:
- ISO. *Data elements and interchange formats — Information interchange —
  Representation of dates and times*. ISO 8601:2004.
- Klyne, G. and Newman, C. "Date and Time on the Internet: Timestamps."
  *RFC 3339*, IETF, 2002.

### Robata Application

- All wall-clock timestamps on the wire use RFC 3339 format with UTC offset
  (`Z`), microsecond precision, e.g. `2026-07-20T00:00:00.000000Z`.
- Canonical nanosecond timestamps are integer strings, not ISO 8601 strings,
  to avoid precision loss and representation ambiguity.
- `contracts/logical_nodes.py` — `Rfc3339Timestamp` type alias enforces the
  format at the Pydantic model boundary.

---

## 4. SHA-256 — Secure Hash Algorithm

**Issuer**: NIST (National Institute of Standards and Technology)  
**Reference**: NIST. *Secure Hash Standard (SHS)*. FIPS PUB 180-4, 2015.

### Robata Application

SHA-256 is used for two distinct purposes:

| Purpose | Function | Input |
|---|---|---|
| **Exact-byte identity** | `exact_bytes_sha256()` | Raw bytes of an artifact |
| **Semantic identity** | `semantic_sha256()` | RFC 8785 canonical JSON of selected fields |

Exact-byte identity detects any change to a stored artifact. Semantic identity
is stable across serialization changes that do not affect business-meaningful
fields (e.g., adding a new audit field does not change the event's semantic key).

---

## 5. JSON Schema (Draft 7 / 2020-12)

**References**:
- Wright, A., Andrews, H. and Hutton, B. (eds). *JSON Schema: A Media Type for
  Describing JSON Documents*. IETF Internet-Draft, 2019.
- Wright, A. et al. *JSON Schema Validation: A Vocabulary for Structural
  Validation of JSON*. IETF Internet-Draft, 2022.

### Robata Application

- All wire contracts in `schemas/` are checked-in, immutable JSON Schema files.
- `contracts/schema_registry.py` — `SchemaRegistry` stores the exact SHA-256
  of every registered schema artifact; a mismatch fails closed.
- Pydantic models are a second parse layer; they must conform to the registered
  schema but are not a second authority. The JSON Schema file governs.

---

## 6. Semantic Versioning (SemVer)

**Reference**: Preston-Werner, T. "Semantic Versioning 2.0.0."
*semver.org*, 2013.

### Robata Application

Schema versions follow SemVer (`MAJOR.MINOR.PATCH`):
- **MAJOR** increment: breaking wire change; old readers cannot parse new payloads.
- **MINOR** increment: additive change; old readers safely ignore new fields.
- **PATCH** increment: documentation or constraint clarification; wire shape unchanged.

The `upcasters` registry in `schema-catalog.json` maps `(source_version,
target_version)` pairs to registered migration artifacts. Ambiguous or cyclic
paths fail closed.

---

## 7. MCAP File Format

**Reference**: Foxglove. "MCAP File Format Specification." *mcap.dev*, 2022.  
**URL**: https://mcap.dev/spec

### Robata Application

- `adapters/mcap_inspector.py` — `OfficialMcapInspector` reads MCAP summary
  and index structures; falls back to sequential scan when the index is absent.
- `contracts/mcap.py` — typed models for MCAP channel, schema, and message
  metadata.
- MCAP is treated as an immutable source artifact; its SHA-256 and byte length
  are recorded as part of the source provenance before any derived artifact is
  produced.

---

## 8. H.264 / AVC Video Coding Standard

**Reference**: ITU-T. *Advanced video coding for generic audiovisual services*.
Recommendation H.264 (08/2021). ITU-T, 2021. Also published as ISO/IEC 14496-10.

### Robata Application

- `adapters/pyav_decoder.py` — `PyAvH264DecoderProbe` validates that a camera
  stream can be decoded before producing a `VALID` admission report.
- `adapters/pyav_mp4_exporter.py` — uses `REMUX` (bitstream copy) where the
  source codec is already H.264; only transcodes when the source requires it.
- The distinction between `REMUX`, `EXTRACT`, `TRANSCODE`, and `FRAME_DECODE`
  in the export contracts is grounded in H.264 container semantics.
