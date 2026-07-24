"""Independently verify the pre-EOS stream identity golden chain.

This reference path intentionally uses only the Python standard library. It
does not import the Robata package, Pydantic, or the runtime RFC 8785 helper.
The checked-in vector stays in an integer/string JSON domain where the compact
serializer below is byte-equivalent to RFC 8785.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_PATH = REPOSITORY_ROOT / "conformance" / "stream_identity_chain_v1.json"

_CAMERA_IDS = tuple(f"cam_{index:02d}" for index in range(1, 7))
_EXPECTED_SECTIONS = (
    "capture",
    "segments",
    "window",
    "inference",
    "inference_attempt",
    "work",
    "plan",
    "declaration",
    "seal",
    "closure",
    "finalization",
)


class VectorVerificationError(ValueError):
    """The vector is malformed or disagrees with the independent formula."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VectorVerificationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise VectorVerificationError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise VectorVerificationError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VectorVerificationError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VectorVerificationError(f"{label} must be an integer")
    return value


def _field(source: Mapping[str, object], name: str, label: str) -> object:
    if name not in source:
        raise VectorVerificationError(f"{label}.{name} is required")
    return source[name]


def _projection(
    source: Mapping[str, object],
    names: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    return {name: _field(source, name, label) for name in names}


def _validate_canonical_domain(value: object, label: str = "projection") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        if not -(2**53 - 1) <= value <= 2**53 - 1:
            raise VectorVerificationError(f"{label} integer exceeds the exact JSON range")
        return
    if isinstance(value, float):
        raise VectorVerificationError(f"{label} floats are outside this reference profile")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_canonical_domain(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.isascii():
                raise VectorVerificationError(f"{label} keys must be ASCII strings")
            _validate_canonical_domain(item, f"{label}.{key}")
        return
    raise VectorVerificationError(f"{label} contains unsupported JSON value {type(value)!r}")


def _canonical_json_bytes(value: object) -> bytes:
    _validate_canonical_domain(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _semantic_sha256(projection: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(projection)).hexdigest()


def _derived_uuid(namespace_label: str, name: str) -> str:
    namespace = uuid5(NAMESPACE_URL, f"robata:stream-namespace:{namespace_label}")
    return str(uuid5(namespace, name))


def _interval_projection(source: Mapping[str, object], label: str) -> dict[str, str]:
    return {
        "start_ns": str(_integer(_field(source, "start_ns", label), f"{label}.start_ns")),
        "end_ns": str(_integer(_field(source, "end_ns", label), f"{label}.end_ns")),
    }


def _policy_binding(source: Mapping[str, object], label: str) -> dict[str, object]:
    return {
        "version": _text(_field(source, "version", label), f"{label}.version"),
        "semantic_sha256": _text(
            _field(source, "semantic_sha256", label),
            f"{label}.semantic_sha256",
        ),
    }


def recompute_identity_chain(document: Mapping[str, object]) -> dict[str, object]:
    """Build every normalized preimage and identity from controlled vector input."""

    vector_input = _object(_field(document, "input", "document"), "input")
    schema_ref = _projection(
        _object(_field(vector_input, "schema_ref", "input"), "input.schema_ref"),
        ("schema_id", "version", "artifact_id", "sha256"),
        "input.schema_ref",
    )
    interval = _object(_field(vector_input, "interval", "input"), "input.interval")
    interval_projection = _interval_projection(interval, "input.interval")

    capture_input = _object(_field(vector_input, "capture", "input"), "input.capture")
    channel_inputs = _array(
        _field(capture_input, "channel_bindings", "input.capture"),
        "input.capture.channel_bindings",
    )
    channel_bindings: list[dict[str, object]] = []
    for index, raw_binding in enumerate(channel_inputs):
        label = f"input.capture.channel_bindings[{index}]"
        binding = _object(raw_binding, label)
        channel_bindings.append(
            _projection(
                binding,
                (
                    "camera_id",
                    "source_channel_id",
                    "source_channel_epoch",
                    "channel_binding_semantic_sha256",
                ),
                label,
            )
        )
    if tuple(binding["camera_id"] for binding in channel_bindings) != _CAMERA_IDS:
        raise VectorVerificationError("capture channel bindings must be in six-camera order")

    authority_fields = (
        "authority_id",
        "authority_epoch",
        "policy_version",
        "initial_binding_semantic_sha256",
    )
    capture_projection = {
        "semantic_projection_version": "pre-eos-capture-subject-semantic-v1",
        "identity_policy_version": "pre-eos-capture-identity-v1",
        **_projection(
            capture_input,
            (
                "capture_authority_id",
                "capture_authority_epoch",
                "capture_assignment_policy_version",
                "acquisition_id",
                "acquisition_epoch",
            ),
            "input.capture",
        ),
        "ordered_channel_bindings": channel_bindings,
        "mapping_authority": _projection(
            _object(
                _field(capture_input, "mapping_authority", "input.capture"),
                "input.capture.mapping_authority",
            ),
            authority_fields,
            "input.capture.mapping_authority",
        ),
        "clock_authority": _projection(
            _object(
                _field(capture_input, "clock_authority", "input.capture"),
                "input.capture.clock_authority",
            ),
            authority_fields,
            "input.capture.clock_authority",
        ),
    }
    capture_digest = _semantic_sha256(capture_projection)
    capture_key = f"pre-eos-capture-v1:{capture_digest}"
    capture = {
        "capture_scope_digest": capture_digest,
        "capture_scope_key": capture_key,
        "capture_scope_id": _derived_uuid("pre-eos-capture-v1", capture_key),
    }

    segment_input = _object(_field(vector_input, "segment", "input"), "input.segment")
    packet_closure = [
        _text(value, f"input.segment.ordered_packet_or_sequence_closure[{index}]")
        for index, value in enumerate(
            _array(
                _field(
                    segment_input,
                    "ordered_packet_or_sequence_closure",
                    "input.segment",
                ),
                "input.segment.ordered_packet_or_sequence_closure",
            )
        )
    ]
    segments: list[dict[str, object]] = []
    slot_closure: list[dict[str, object]] = []
    for camera_id in _CAMERA_IDS:
        segment_projection = {
            "semantic_projection_version": "stream-segment-semantic-v1",
            "identity_policy_version": "stream-segment-identity-v1",
            "capture_scope_digest": capture_digest,
            "camera_id": camera_id,
            "requested_interval": interval_projection,
            "effective_interval": interval_projection,
            "ordered_packet_or_sequence_closure": packet_closure,
            **_projection(
                segment_input,
                (
                    "exact_content_sha256",
                    "mapping_semantic_sha256",
                    "clock_or_alignment_semantic_sha256",
                    "segmentation_policy_version",
                ),
                "input.segment",
            ),
        }
        segment_digest = _semantic_sha256(segment_projection)
        segment_key = f"stream-segment-v1:{segment_digest}"
        segments.append(
            {
                "camera_id": camera_id,
                "segment_semantic_sha256": segment_digest,
                "segment_key": segment_key,
                "segment_id": _derived_uuid("stream-segment-v1", segment_key),
            }
        )
        slot_closure.append(
            {
                "kind": "SEGMENT",
                "camera_id": camera_id,
                "capture_scope_digest": capture_digest,
                "segment_key": segment_key,
                "segment_semantic_sha256": segment_digest,
            }
        )

    window_input = _object(_field(vector_input, "window", "input"), "input.window")
    window_projection = {
        "semantic_projection_version": "incremental-window-semantic-v1",
        "identity_policy_version": "incremental-window-identity-v1",
        "capture_scope_digest": capture_digest,
        "purpose": _field(window_input, "purpose", "input.window"),
        "requested_interval": interval_projection,
        "effective_interval": interval_projection,
        "ordered_six_slot_segment_or_explicit_absence_closure": slot_closure,
        **_projection(
            window_input,
            (
                "mapping_semantic_sha256",
                "clock_or_alignment_semantic_sha256",
                "parent_subject_key_or_none",
                "refinement_role_or_none",
                "refinement_generation",
                "window_policy_version",
            ),
            "input.window",
        ),
    }
    window_digest = _semantic_sha256(window_projection)
    window_key = f"incremental-window-v1:{window_digest}"
    window = {
        "window_semantic_sha256": window_digest,
        "window_key": window_key,
        "window_id": _derived_uuid("incremental-window-v1", window_key),
    }

    plan_input = _object(_field(vector_input, "plan", "input"), "input.plan")
    policy_names = (
        "segmentation_policy_binding",
        "window_policy_binding",
        "watermark_policy_binding",
        "lateness_policy_binding",
        "idle_source_policy_binding",
    )
    policies = {
        name: _policy_binding(
            _object(_field(plan_input, name, "input.plan"), f"input.plan.{name}"),
            f"input.plan.{name}",
        )
        for name in policy_names
    }
    plan_projection = {
        "plan_projection_version": "expected-window-plan-semantic-v1",
        "plan_identity_policy_version": "expected-window-plan-identity-v1",
        "capture_scope_digest": capture_digest,
        **policies,
        "planner_version": _field(plan_input, "planner_version", "input.plan"),
    }
    plan_digest = _semantic_sha256(plan_projection)
    plan_key = f"expected-window-plan-v1:{plan_digest}"
    plan = {
        "plan_key": plan_key,
        "plan_digest": plan_digest,
        "plan_id": _derived_uuid("expected-window-plan-v1", plan_key),
    }

    declaration_input = _object(
        _field(vector_input, "declaration", "input"),
        "input.declaration",
    )
    ordinal = _integer(
        _field(declaration_input, "ordinal", "input.declaration"),
        "input.declaration.ordinal",
    )
    declaration_projection = {
        "declaration_projection_version": "expected-window-declaration-semantic-v1",
        "plan_key": plan_key,
        "ordinal": ordinal,
        "window_key": window_key,
        "window_semantic_sha256": window_digest,
        "requested_interval": interval_projection,
        "effective_interval": interval_projection,
        "ordered_six_slot_segment_or_explicit_absence_closure": slot_closure,
        "watermark_source_facts_sha256": _field(
            declaration_input,
            "watermark_source_facts_sha256",
            "input.declaration",
        ),
    }
    declaration_digest = _semantic_sha256(declaration_projection)
    previous_chain = _field(
        declaration_input,
        "previous_append_chain_sha256",
        "input.declaration",
    )
    append_chain_digest = _semantic_sha256(
        {
            "version": "expected-window-plan-append-v1",
            "plan_key": plan_key,
            "ordinal": ordinal,
            "declaration_semantic_sha256": declaration_digest,
            "previous": previous_chain,
        }
    )
    declaration = {
        "declaration_semantic_sha256": declaration_digest,
        "append_chain_sha256": append_chain_digest,
    }
    expected_member_root = _semantic_sha256(
        {
            "version": "expected-window-member-root-v1",
            "ordered_expected_members": [
                {
                    "ordinal": ordinal,
                    "window_key": window_key,
                    "window_semantic_sha256": window_digest,
                    "declaration_semantic_sha256": declaration_digest,
                }
            ],
        }
    )

    seal_input = _object(_field(vector_input, "seal", "input"), "input.seal")
    seal_projection = {
        "seal_projection_version": "expected-window-plan-seal-semantic-v1",
        "plan_key": plan_key,
        "capture_scope_digest": capture_digest,
        "eos_source_receipt_semantic_sha256": _field(
            seal_input,
            "eos_source_receipt_semantic_sha256",
            "input.seal",
        ),
        "final_source_timeline_semantic_sha256": _field(
            seal_input,
            "final_source_timeline_semantic_sha256",
            "input.seal",
        ),
        "final_duration_ns": str(
            _integer(
                _field(seal_input, "final_duration_ns", "input.seal"),
                "input.seal.final_duration_ns",
            )
        ),
        **_projection(
            seal_input,
            (
                "ordered_six_channel_health_closure_sha256",
                "mapping_closure_semantic_sha256",
                "clock_or_alignment_closure_semantic_sha256",
            ),
            "input.seal",
        ),
        **policies,
        "planner_version": _field(plan_input, "planner_version", "input.plan"),
        "expected_member_count": 1,
        "first_ordinal": 0,
        "last_ordinal_or_none": 0,
        "final_append_chain_sha256": append_chain_digest,
        "ordered_expected_member_root_sha256": expected_member_root,
    }
    seal_digest = _semantic_sha256(seal_projection)
    seal = {
        "seal_semantic_sha256": seal_digest,
        "ordered_expected_member_root_sha256": expected_member_root,
    }

    inference_input = _object(
        _field(vector_input, "inference", "input"),
        "input.inference",
    )
    inference_projection = {
        "inference_projection_version": "stream-inference-semantic-v1",
        "inference_identity_policy_version": "stream-inference-identity-v1",
        "window_key": window_key,
        "window_semantic_sha256": window_digest,
        "purpose": _field(inference_input, "purpose", "input.inference"),
        "input_plan_semantic_sha256": _field(
            inference_input,
            "input_plan_semantic_sha256",
            "input.inference",
        ),
    }
    inference_digest = _semantic_sha256(inference_projection)
    inference_key = f"stream-inference-v1:{inference_digest}"
    logical_id = _derived_uuid("stream-inference-v1", inference_key)
    inference = {
        "inference_semantic_sha256": inference_digest,
        "inference_key": inference_key,
        "stream_inference_logical_id": logical_id,
    }
    attempt_number = _integer(
        _field(inference_input, "attempt_number", "input.inference"),
        "input.inference.attempt_number",
    )
    attempt_key = f"stream-inference-attempt-v1:{logical_id}:{attempt_number}"
    inference_attempt = {
        "inference_attempt_key": attempt_key,
        "inference_attempt_id": _derived_uuid("stream-inference-attempt-v1", attempt_key),
    }

    work_input = _object(_field(vector_input, "work", "input"), "input.work")
    dependencies = _array(
        _field(work_input, "ordered_dependency_projections", "input.work"),
        "input.work.ordered_dependency_projections",
    )
    work_projection = {
        "work_projection_version": "stream-work-plan-semantic-v1",
        "work_key_policy_version": "stream-work-key-v1",
        "stream_run_id": _field(work_input, "stream_run_id", "input.work"),
        "capture_scope_digest": capture_digest,
        "stage": _field(work_input, "stage", "input.work"),
        "typed_subject_key": window_key,
        "typed_subject_semantic_sha256": window_digest,
        "ordered_dependency_projections": dependencies,
        "input_semantic_sha256": _field(
            work_input,
            "input_semantic_sha256",
            "input.work",
        ),
        "config_semantic_sha256": _field(
            work_input,
            "config_semantic_sha256",
            "input.work",
        ),
    }
    work_digest = _semantic_sha256(work_projection)
    work_key = f"stream-work-v1:{work_digest}"
    work_id = _derived_uuid("stream-work-v1", work_key)
    work = {
        "work_logical_key": work_key,
        "work_item_id": work_id,
        "created_at": _field(work_input, "created_at", "input.work"),
    }

    terminal_input = _object(
        _field(vector_input, "terminal", "input"),
        "input.terminal",
    )
    evidence_input = _object(
        _field(terminal_input, "evidence", "input.terminal"),
        "input.terminal.evidence",
    )
    evidence_ref = {
        **_projection(
            evidence_input,
            ("artifact_id", "exact_sha256", "byte_count", "media_type"),
            "input.terminal.evidence",
        ),
        "schema_ref": schema_ref,
    }
    terminal_member_projection = {
        "member_projection_version": "window-terminal-member-semantic-v1",
        "plan_key": plan_key,
        "expected_ordinal": ordinal,
        "window_key": window_key,
        "window_semantic_sha256": window_digest,
        "terminal_outcome": _field(
            terminal_input,
            "terminal_outcome",
            "input.terminal",
        ),
        "terminal_work_item_id": work_id,
        "terminal_work_logical_key": work_key,
        "terminal_evidence_ref": evidence_ref,
        "terminal_policy_version": _field(
            terminal_input,
            "terminal_policy_version",
            "input.terminal",
        ),
    }
    terminal_member_digest = _semantic_sha256(terminal_member_projection)
    terminal_member_root = _semantic_sha256(
        {
            "version": "window-terminal-member-root-v1",
            "plan_seal_semantic_sha256": seal_digest,
            "ordered_member_semantic_sha256_values": [terminal_member_digest],
        }
    )
    terminal_closure_digest = _semantic_sha256(
        {
            "projection_version": "window-terminal-closure-semantic-v1",
            "plan_key": plan_key,
            "plan_seal_semantic_sha256": seal_digest,
            "expected_member_count": 1,
            "terminal_member_root": terminal_member_root,
        }
    )
    closure = {
        "terminal_member_semantic_sha256": terminal_member_digest,
        "terminal_member_root": terminal_member_root,
        "terminal_closure_digest": terminal_closure_digest,
    }

    finalization_input = _object(
        _field(vector_input, "finalization", "input"),
        "input.finalization",
    )
    subject_mapping_input = _object(
        _field(finalization_input, "subject_mapping", "input.finalization"),
        "input.finalization.subject_mapping",
    )
    subject_mapping = {
        "incremental_subject_type": _field(
            subject_mapping_input,
            "incremental_subject_type",
            "input.finalization.subject_mapping",
        ),
        "incremental_subject_key": window_key,
        "incremental_subject_semantic_sha256": window_digest,
        **_projection(
            subject_mapping_input,
            (
                "final_subject_type",
                "final_subject_key",
                "final_subject_semantic_sha256",
            ),
            "input.finalization.subject_mapping",
        ),
    }
    finalization_projection = {
        "finalization_projection_version": "recording-finalization-map-semantic-v1",
        "finalization_policy_version": "recording-finalization-policy-v1",
        "capture_scope_key": capture_key,
        "capture_scope_digest": capture_digest,
        **_projection(
            finalization_input,
            (
                "final_source_subject_type",
                "final_source_subject_id",
                "final_source_exact_sha256",
                "final_recording_identity",
            ),
            "input.finalization",
        ),
        "final_duration_ns": str(
            _integer(
                _field(finalization_input, "final_duration_ns", "input.finalization"),
                "input.finalization.final_duration_ns",
            )
        ),
        **_projection(
            finalization_input,
            (
                "final_mapping_semantic_sha256",
                "final_alignment_semantic_sha256",
            ),
            "input.finalization",
        ),
        "expected_plan_seal_semantic_sha256": seal_digest,
        "window_terminal_closure_semantic_sha256": terminal_closure_digest,
        "export_manifest_semantic_sha256": _field(
            finalization_input,
            "export_manifest_semantic_sha256",
            "input.finalization",
        ),
        "ordered_subject_mappings": [subject_mapping],
    }
    finalization_digest = _semantic_sha256(finalization_projection)
    finalization = {
        "finalization_key": f"recording-finalization-map-v1:{finalization_digest}",
        "finalization_semantic_sha256": finalization_digest,
    }

    return {
        "capture": capture,
        "segments": segments,
        "window": window,
        "inference": inference,
        "inference_attempt": inference_attempt,
        "work": work,
        "plan": plan,
        "declaration": declaration,
        "seal": seal,
        "closure": closure,
        "finalization": finalization,
    }


def verify_vector(path: Path = DEFAULT_VECTOR_PATH) -> dict[str, object]:
    """Load a vector and compare every expected identity with an independent result."""

    try:
        loaded: object = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VectorVerificationError(f"cannot read vector document {path}: {exc}") from exc
    document = _object(loaded, "document")
    if document.get("fixture_version") != "stream-identity-chain-v1":
        raise VectorVerificationError("unexpected fixture_version")
    if document.get("canonicalization") != "RFC8785":
        raise VectorVerificationError("canonicalization must be RFC8785")

    actual = recompute_identity_chain(document)
    for section in _EXPECTED_SECTIONS:
        if document.get(section) != actual[section]:
            raise VectorVerificationError(f"{section} identity does not match controlled input")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_VECTOR_PATH)
    args = parser.parse_args(argv)
    verified = verify_vector(args.path)
    print(
        json.dumps(
            {
                "implementation": "python-stdlib-independent-v1",
                "suite": "stream-identity-chain-v1",
                "verified_sections": len(verified),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
