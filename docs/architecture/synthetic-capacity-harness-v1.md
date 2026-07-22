# Synthetic Capacity Harness V1

## Purpose

The dependency-free harness in src/robata/runtime/capacity.py exercises the accounting and
state-reconciliation rules in Architecture V1.1 Section 19 before representative infrastructure
or provider traffic is available. It is a deterministic discrete-event simulator, not a claim
about production capacity.

A content-addressed SyntheticLoadProfile defines:

- offered logical-unit count and stable burst-shaped arrival times;
- physical recording duration and every mapped camera-stream duration;
- a repeating service-time pattern;
- deadline budget and optional observation cutoff;
- explicit failed and skipped outcome ordinals.

Worker count is supplied separately so the same workload identity can compare scheduler capacity
changes. Work IDs derive from the complete profile digest and ordinal.

## Report Invariants

Every SyntheticCapacityReport:

- reports recording-hours and camera-video-hours separately;
- reconciles every offered unit into exactly one of SUCCEEDED, FAILED, SKIPPED, or PENDING;
- keeps failure, skip, and deadline denominators explicit: attempted terminal work for failure,
  all offered work for skips, and non-skipped eligible work for deadline misses;
- reports queue, service, and wall latency over terminal attempted work with nearest-rank
  p50/p95/p99;
- retains backlog peak/end, deadline misses, utilization, offered rate, and nominal service rate;
- never records a scheduled start after the cutoff as observed; not-yet-started work keeps a null
  worker/start and reports only queue wait accrued through the cutoff;
- flags local service-capacity, queue-wait, reliability, and deadline pressure;
- counts completed throughput from successful units rather than hiding failed, skipped, or pending
  work;
- carries SYNTHETIC_LOCAL, NOT_MEASURED, and production_eligible=false semantics.

Unequal camera-stream durations are summed as observed inputs. The harness never assumes that
camera-video-hours equal six times recording-hours.

## Local Thresholds And Regression

LocalSloPolicy and CapacityRegressionPolicy are explicit versioned inputs with content digests.
The SLO policy independently bounds attempted-work failure, offered-work skipping, deadline misses,
terminal p95 wall time, and optional backlog drain. Regression policy compares like-for-like
throughput, p95 wall time, failure rate, deadline-miss rate, and ending backlog. Results remain
NOT_MEASURED and cannot feed a promotion decision as certifying evidence.

Regression comparison requires the same workload profile digest. This prevents a faster result
from being manufactured by silently changing burst shape, durations, failures, deadlines, or
service-time inputs.

## Qualification Boundary

The harness closes only the framework and arithmetic layer. A real capacity or SLO statement still
requires the Section 19.9 acceptance sequence:

- representative source, codec, duration, QA, and action distributions;
- approved peak shape and both 500-hour interpretations;
- live provider endpoint, quota, cache, model, region, and price-card pins;
- a registered warm-up, soak, burst, repetition, and block-bootstrap policy;
- real queue, database, object-store, network, CPU/GPU, and failover observations;
- shadow saturation and primary-isolation evidence;
- approved deadline, T+1, headroom, cost, and promotion thresholds.

Cached, fixture, mocked-provider, and synthetic runs remain non-certifying for provider latency,
quota, cost, sustained capacity, or production SLOs.
