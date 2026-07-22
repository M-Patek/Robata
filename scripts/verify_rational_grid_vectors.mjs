#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_VECTOR_PATH = resolve(
  REPOSITORY_ROOT,
  "conformance",
  "rational-grid-canonicalization-v1.json",
);
const CANONICAL_DECIMAL = /^(?:0|-[1-9][0-9]*|[1-9][0-9]*)$/;
const LOWER_HEX_OCTETS = /^(?:[0-9a-f]{2})+$/;
const INT64_MIN = -(2n ** 63n);
const INT64_MAX = 2n ** 63n - 1n;

class VectorVerificationError extends Error {}

function requireObject(value, label) {
  if (value === null || Array.isArray(value) || typeof value !== "object") {
    throw new VectorVerificationError(`${label} must be a JSON object`);
  }
  return value;
}

function requireArray(value, label) {
  if (!Array.isArray(value)) {
    throw new VectorVerificationError(`${label} must be a JSON array`);
  }
  return value;
}

function requireText(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    throw new VectorVerificationError(`${label} must be a nonempty string`);
  }
  return value;
}

function integer(value, label) {
  const text = requireText(value, label);
  if (!CANONICAL_DECIMAL.test(text)) {
    throw new VectorVerificationError(`${label} must use canonical base-10 integer text`);
  }
  return BigInt(text);
}

function int64(value, label) {
  const parsed = integer(value, label);
  if (parsed < INT64_MIN || parsed > INT64_MAX) {
    throw new VectorVerificationError(`${label} must fit signed int64`);
  }
  return parsed;
}

function assertUnicodeScalarString(value) {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        throw new VectorVerificationError("RFC8785 input contains an unpaired surrogate");
      }
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new VectorVerificationError("RFC8785 input contains an unpaired surrogate");
    }
  }
}

function canonicalJson(value) {
  if (value === null || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "string") {
    assertUnicodeScalarString(value);
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new VectorVerificationError("RFC8785 input contains a non-finite number");
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  }
  if (typeof value === "object") {
    const fields = Object.keys(value).sort();
    return `{${fields
      .map((key) => {
        assertUnicodeScalarString(key);
        return `${JSON.stringify(key)}:${canonicalJson(value[key])}`;
      })
      .join(",")}}`;
  }
  throw new VectorVerificationError(`value of type ${typeof value} is not JSON`);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function absolute(value) {
  return value < 0n ? -value : value;
}

function gcd(left, right) {
  let a = absolute(left);
  let b = absolute(right);
  while (b !== 0n) {
    const remainder = a % b;
    a = b;
    b = remainder;
  }
  return a;
}

function floorDivmod(numerator, denominator) {
  let quotient = numerator / denominator;
  let remainder = numerator % denominator;
  if (remainder < 0n) {
    quotient -= 1n;
    remainder += denominator;
  }
  return [quotient, remainder];
}

function roundHalfEven(numerator, denominator) {
  if (denominator <= 0n) {
    throw new VectorVerificationError("rounding denominator must be positive");
  }
  const [quotient, remainder] = floorDivmod(numerator, denominator);
  const doubledRemainder = remainder * 2n;
  if (doubledRemainder < denominator) return quotient;
  if (doubledRemainder > denominator) return quotient + 1n;
  return quotient % 2n === 0n ? quotient : quotient + 1n;
}

function createGrid(input, label) {
  const fields = requireObject(input.grid, `${label}.grid`);
  const origin = int64(fields.grid_origin_ns, `${label}.grid.grid_origin_ns`);
  let rateNum = integer(fields.rate_num, `${label}.grid.rate_num`);
  let rateDen = integer(fields.rate_den, `${label}.grid.rate_den`);
  if (rateNum <= 0n || rateDen <= 0n) {
    throw new VectorVerificationError(`${label}.grid rate must be positive`);
  }
  const rateDivisor = gcd(rateNum, rateDen);
  rateNum /= rateDivisor;
  rateDen /= rateDivisor;
  const rawPeriodNumerator = 1_000_000_000n * rateDen;
  const periodDivisor = gcd(rawPeriodNumerator, rateNum);
  return {
    origin,
    periodNum: rawPeriodNumerator / periodDivisor,
    periodDen: rateNum / periodDivisor,
  };
}

function targetNs(grid, k) {
  const target =
    grid.origin + roundHalfEven(k * grid.periodNum, grid.periodDen);
  if (target < INT64_MIN || target > INT64_MAX) {
    throw new VectorVerificationError("sampling target does not fit signed int64");
  }
  return target;
}

function firstKAtOrAfter(grid, timestamp) {
  const relativeTarget = timestamp - grid.origin;
  const boundary = (2n * relativeTarget - 1n) * grid.periodDen;
  const scale = 2n * grid.periodNum;
  const [quotient, remainder] = floorDivmod(boundary, scale);
  if (remainder === 0n && relativeTarget % 2n === 0n) return quotient;
  return quotient + 1n;
}

function enumerateTargets(grid, start, end) {
  if (start >= end) {
    throw new VectorVerificationError("interval start must be less than end");
  }
  const targets = [];
  let k = firstKAtOrAfter(grid, start);
  const stopK = firstKAtOrAfter(grid, end);
  while (k < stopK) {
    const timestamp = targetNs(grid, k);
    targets.push({ k, targetNs: timestamp });
    if (targets.length > 100_000) {
      throw new VectorVerificationError("vector target budget exceeded");
    }
    k = firstKAtOrAfter(grid, timestamp + 1n);
  }
  return targets;
}

function parseFrame(value, label) {
  const fields = requireObject(value, label);
  const locatorHex = requireText(fields.source_locator_hex, `${label}.source_locator_hex`);
  if (!LOWER_HEX_OCTETS.test(locatorHex)) {
    throw new VectorVerificationError(
      `${label}.source_locator_hex must be lowercase hexadecimal octets`,
    );
  }
  if (typeof fields.decodable !== "boolean") {
    throw new VectorVerificationError(`${label}.decodable must be a boolean`);
  }
  return {
    alignedTimestampNs: int64(fields.aligned_timestamp_ns, `${label}.aligned_timestamp_ns`),
    sourceTimestampNs: int64(fields.source_timestamp_ns, `${label}.source_timestamp_ns`),
    locatorHex,
    locatorBytes: Buffer.from(locatorHex, "hex"),
    decodable: fields.decodable,
  };
}

function compareBigInt(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function compareFramesForTarget(left, right, target) {
  return (
    compareBigInt(
      absolute(left.alignedTimestampNs - target.targetNs),
      absolute(right.alignedTimestampNs - target.targetNs),
    ) ||
    compareBigInt(left.alignedTimestampNs, right.alignedTimestampNs) ||
    compareBigInt(left.sourceTimestampNs, right.sourceTimestampNs) ||
    Buffer.compare(left.locatorBytes, right.locatorBytes)
  );
}

function selectFrames(targets, frames, start, end, tolerance) {
  if (tolerance < 0n) {
    throw new VectorVerificationError("selection tolerance must be nonnegative");
  }
  const intervalFrames = frames.filter(
    (frame) => start <= frame.alignedTimestampNs && frame.alignedTimestampNs < end,
  );
  const provisional = targets.map((target) => {
    const withinTolerance = intervalFrames
      .filter(
        (frame) => absolute(frame.alignedTimestampNs - target.targetNs) <= tolerance,
      )
      .sort((left, right) => compareFramesForTarget(left, right, target));
    const selected = withinTolerance.find((frame) => frame.decodable);
    if (selected !== undefined) {
      return {
        ...target,
        status: "SELECTED",
        frame: selected,
        deltaToTargetNs: selected.alignedTimestampNs - target.targetNs,
      };
    }
    if (withinTolerance.length > 0) {
      const failed = withinTolerance[0];
      return {
        ...target,
        status: "DECODE_FAILED",
        frame: failed,
        deltaToTargetNs: failed.alignedTimestampNs - target.targetNs,
      };
    }
    return {
      ...target,
      status: "NO_FRAME_WITHIN_TOLERANCE",
      frame: null,
      deltaToTargetNs: null,
    };
  });

  const assignments = new Map();
  provisional.forEach((selection, index) => {
    if (selection.status === "SELECTED") {
      const indexes = assignments.get(selection.frame.locatorHex) ?? [];
      indexes.push(index);
      assignments.set(selection.frame.locatorHex, indexes);
    }
  });
  for (const indexes of assignments.values()) {
    if (indexes.length < 2) continue;
    indexes.sort((leftIndex, rightIndex) => {
      const left = provisional[leftIndex];
      const right = provisional[rightIndex];
      return (
        compareBigInt(absolute(left.deltaToTargetNs), absolute(right.deltaToTargetNs)) ||
        compareBigInt(left.targetNs, right.targetNs) ||
        compareBigInt(left.k, right.k)
      );
    });
    for (const index of indexes.slice(1)) {
      provisional[index] = { ...provisional[index], status: "DEDUPLICATED_FRAME" };
    }
  }
  return provisional;
}

function frameProjection(frame) {
  if (frame === null) return null;
  return {
    aligned_timestamp_ns: frame.alignedTimestampNs.toString(),
    source_timestamp_ns: frame.sourceTimestampNs.toString(),
    source_locator_hex: frame.locatorHex,
    decodable: frame.decodable,
  };
}

function gridOutcome(input, label) {
  const grid = createGrid(input, label);
  const interval = requireObject(input.interval, `${label}.interval`);
  const start = int64(interval.start_ns, `${label}.interval.start_ns`);
  const end = int64(interval.end_ns, `${label}.interval.end_ns`);
  const tolerance = int64(input.selection_tolerance_ns, `${label}.selection_tolerance_ns`);
  const frames = requireArray(input.frames, `${label}.frames`).map((value, index) =>
    parseFrame(value, `${label}.frames[${index}]`),
  );
  const targets = enumerateTargets(grid, start, end);
  const selections = selectFrames(targets, frames, start, end, tolerance);
  return {
    period: {
      period_num_ns: grid.periodNum.toString(),
      period_den: grid.periodDen.toString(),
    },
    targets: targets.map((target) => ({
      k: target.k.toString(),
      target_ns: target.targetNs.toString(),
    })),
    selections: selections.map((selection) => ({
      k: selection.k.toString(),
      target_ns: selection.targetNs.toString(),
      status: selection.status,
      frame: frameProjection(selection.frame),
      delta_to_target_ns:
        selection.deltaToTargetNs === null ? null : selection.deltaToTargetNs.toString(),
    })),
  };
}

function roundingOutcome(input, label) {
  const fractions = requireArray(input.fractions, `${label}.fractions`);
  return {
    rounding_results: fractions.map((value, index) => {
      const fraction = requireObject(value, `${label}.fractions[${index}]`);
      const numerator = integer(fraction.numerator, `${label}.fractions[${index}].numerator`);
      const denominator = integer(
        fraction.denominator,
        `${label}.fractions[${index}].denominator`,
      );
      return roundHalfEven(numerator, denominator).toString();
    }),
  };
}

function requireEqual(actual, expected, label) {
  if (canonicalJson(actual) !== canonicalJson(expected)) {
    throw new VectorVerificationError(`${label} does not match`);
  }
}

function verifyVectors(path) {
  const root = requireObject(JSON.parse(readFileSync(path, "utf8")), "document");
  if (root.suite_id !== "robata-rational-grid-canonicalization") {
    throw new VectorVerificationError("unexpected suite_id");
  }
  if (root.suite_version !== "1.0.0") {
    throw new VectorVerificationError("unexpected suite_version");
  }
  if (root.canonicalization !== "RFC8785") {
    throw new VectorVerificationError("canonicalization must be RFC8785");
  }
  if (root.integer_encoding !== "canonical-base10-string") {
    throw new VectorVerificationError("integer_encoding must be canonical-base10-string");
  }

  const checked = new Set();
  for (const [index, value] of requireArray(root.vectors, "vectors").entries()) {
    const vector = requireObject(value, `vectors[${index}]`);
    const caseId = requireText(vector.case_id, `vectors[${index}].case_id`);
    if (checked.has(caseId)) {
      throw new VectorVerificationError(`duplicate case_id: ${caseId}`);
    }
    const label = `vector ${JSON.stringify(caseId)}`;
    const input = requireObject(vector.input, `${label}.input`);
    const expected = requireObject(vector.expected, `${label}.expected`);
    const canonicalInput = Buffer.from(canonicalJson(input), "utf8");
    if (expected.canonical_input_bytes_hex !== canonicalInput.toString("hex")) {
      throw new VectorVerificationError(`${label} canonical input bytes do not match`);
    }
    if (expected.canonical_input_sha256 !== sha256(canonicalInput)) {
      throw new VectorVerificationError(`${label} canonical input SHA-256 does not match`);
    }

    let outcome;
    if (vector.operation === "ROUND_HALF_EVEN") {
      outcome = roundingOutcome(input, label);
    } else if (vector.operation === "GRID_SELECTION") {
      outcome = gridOutcome(input, label);
    } else {
      throw new VectorVerificationError(
        `${label} has unsupported operation ${JSON.stringify(vector.operation)}`,
      );
    }
    requireEqual(outcome, expected.outcome, `${label} runtime outcome`);
    checked.add(caseId);
  }
  if (checked.size === 0) {
    throw new VectorVerificationError("vector suite must not be empty");
  }
  return checked.size;
}

const vectorPath = process.argv[2] === undefined ? DEFAULT_VECTOR_PATH : resolve(process.argv[2]);
const verifiedVectors = verifyVectors(vectorPath);
process.stdout.write(
  `${JSON.stringify({
    implementation: "node-bigint-independent-v1",
    suite: "robata-rational-grid-canonicalization@1.0.0",
    verified_vectors: verifiedVectors,
  })}\n`,
);
