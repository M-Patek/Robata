"""Deterministic evaluation for Mage small-encoder shadow outputs.

The evaluator intentionally measures exact compact-contract behavior. It does not
pretend that lexical equality is a governed semantic label metric; synonymous action
phrases remain unmatched until a labeled ontology/calibration set exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final

SMALL_ENCODER_EVALUATOR_VERSION: Final = "mage-small-encoder-shadow-evaluator-v3"
MAX_MATCHED_BOUNDARY_MAE_SECONDS: Final = 0.5
_ACTION_TOKEN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class CompactAction:
    label: str
    start_offset_seconds: float | None
    end_offset_seconds: float | None


@dataclass(frozen=True, slots=True)
class ParsedCompactOutput:
    json_syntax_valid: bool
    compact_contract_valid: bool
    actions: tuple[CompactAction, ...]

    @property
    def json_valid(self) -> bool:
        """Backward-compatible alias for syntax validity."""

        return self.json_syntax_valid

    @property
    def normalized_labels(self) -> tuple[str, ...]:
        return tuple(action.label for action in self.actions)


@dataclass(frozen=True, slots=True)
class SmallEncoderPairEvaluation:
    native_json_valid: bool
    candidate_json_valid: bool
    native_compact_contract_valid: bool
    candidate_compact_contract_valid: bool
    native_action_count: int
    candidate_action_count: int
    exact_label_match_count: int
    false_silence: bool
    candidate_repeated_label_excess_count: int
    matched_boundary_start_absolute_seconds: tuple[float, ...]
    matched_boundary_end_absolute_seconds: tuple[float, ...]

    @property
    def exact_label_recall(self) -> float | None:
        if self.native_action_count == 0:
            return None
        return self.exact_label_match_count / self.native_action_count

    @property
    def exact_label_precision(self) -> float | None:
        if self.candidate_action_count == 0:
            return None
        return self.exact_label_match_count / self.candidate_action_count

    @property
    def boundary_start_mae_seconds(self) -> float | None:
        if not self.matched_boundary_start_absolute_seconds:
            return None
        return sum(self.matched_boundary_start_absolute_seconds) / len(
            self.matched_boundary_start_absolute_seconds
        )

    @property
    def boundary_end_mae_seconds(self) -> float | None:
        if not self.matched_boundary_end_absolute_seconds:
            return None
        return sum(self.matched_boundary_end_absolute_seconds) / len(
            self.matched_boundary_end_absolute_seconds
        )

    def as_projection(self) -> dict[str, object]:
        return {
            "native_json_valid": self.native_json_valid,
            "candidate_json_valid": self.candidate_json_valid,
            "native_compact_contract_valid": self.native_compact_contract_valid,
            "candidate_compact_contract_valid": self.candidate_compact_contract_valid,
            "native_action_count": self.native_action_count,
            "candidate_action_count": self.candidate_action_count,
            "exact_label_match_count": self.exact_label_match_count,
            "exact_label_recall": self.exact_label_recall,
            "exact_label_precision": self.exact_label_precision,
            "false_silence": self.false_silence,
            "candidate_repeated_label_excess_count": self.candidate_repeated_label_excess_count,
            "boundary_start_mae_seconds": self.boundary_start_mae_seconds,
            "boundary_end_mae_seconds": self.boundary_end_mae_seconds,
            "matched_boundary_start_absolute_seconds": list(
                self.matched_boundary_start_absolute_seconds
            ),
            "matched_boundary_end_absolute_seconds": list(
                self.matched_boundary_end_absolute_seconds
            ),
        }


@dataclass(frozen=True, slots=True)
class SmallEncoderShadowQualification:
    """Aggregate, deterministic promotion gates for one paired shadow run."""

    segment_count: int
    native_generation_seconds: float
    candidate_generation_seconds: float
    candidate_preparation_seconds: float
    native_action_count: int
    candidate_action_count: int
    exact_label_match_count: int
    false_silence_count: int
    candidate_repeated_label_excess_count: int
    all_json_syntax_valid: bool
    all_compact_contract_valid: bool
    matched_boundary_start_absolute_seconds: tuple[float, ...]
    matched_boundary_end_absolute_seconds: tuple[float, ...]

    @property
    def candidate_total_seconds(self) -> float:
        return self.candidate_generation_seconds + self.candidate_preparation_seconds

    @property
    def generation_plus_preparation_speedup(self) -> float | None:
        if self.candidate_total_seconds <= 0:
            return None
        return self.native_generation_seconds / self.candidate_total_seconds

    @property
    def exact_label_recall(self) -> float | None:
        if self.native_action_count == 0:
            return None
        return self.exact_label_match_count / self.native_action_count

    @property
    def exact_label_precision(self) -> float | None:
        if self.candidate_action_count == 0:
            return None
        return self.exact_label_match_count / self.candidate_action_count

    @property
    def candidate_repeated_label_excess_rate(self) -> float:
        if self.candidate_action_count == 0:
            return 0.0
        return self.candidate_repeated_label_excess_count / self.candidate_action_count

    @property
    def matched_boundary_start_mae_seconds(self) -> float | None:
        values = self.matched_boundary_start_absolute_seconds
        return sum(values) / len(values) if values else None

    @property
    def matched_boundary_end_mae_seconds(self) -> float | None:
        values = self.matched_boundary_end_absolute_seconds
        return sum(values) / len(values) if values else None

    @property
    def matched_boundary_measurement_complete(self) -> bool:
        return (
            self.exact_label_match_count > 0
            and len(self.matched_boundary_start_absolute_seconds) == self.exact_label_match_count
            and len(self.matched_boundary_end_absolute_seconds) == self.exact_label_match_count
        )

    @property
    def gates(self) -> dict[str, bool]:
        speedup = self.generation_plus_preparation_speedup
        recall = self.exact_label_recall
        start_mae = self.matched_boundary_start_mae_seconds
        end_mae = self.matched_boundary_end_mae_seconds
        return {
            "all_json_syntax_valid": self.all_json_syntax_valid,
            "all_compact_contract_valid": self.all_compact_contract_valid,
            "false_silence_zero": self.false_silence_count == 0,
            "exact_label_recall_at_least_0_90": recall is not None and recall >= 0.90,
            "exact_label_precision_at_least_0_90": (
                self.exact_label_precision is not None and self.exact_label_precision >= 0.90
            ),
            "matched_boundary_measurement_complete": self.matched_boundary_measurement_complete,
            "matched_boundary_start_mae_at_most_0_50_seconds": (
                start_mae is not None and start_mae <= MAX_MATCHED_BOUNDARY_MAE_SECONDS
            ),
            "matched_boundary_end_mae_at_most_0_50_seconds": (
                end_mae is not None and end_mae <= MAX_MATCHED_BOUNDARY_MAE_SECONDS
            ),
            "candidate_faster_than_native": speedup is not None and speedup >= 1.0,
        }

    @property
    def qualified(self) -> bool:
        return all(self.gates.values())

    def as_projection(self) -> dict[str, object]:
        return {
            "evaluator_version": SMALL_ENCODER_EVALUATOR_VERSION,
            "max_matched_boundary_mae_seconds": MAX_MATCHED_BOUNDARY_MAE_SECONDS,
            "segment_count": self.segment_count,
            "native_generation_seconds": self.native_generation_seconds,
            "candidate_generation_seconds": self.candidate_generation_seconds,
            "candidate_preparation_seconds": self.candidate_preparation_seconds,
            "candidate_total_seconds": self.candidate_total_seconds,
            "generation_plus_preparation_speedup": self.generation_plus_preparation_speedup,
            "native_action_count": self.native_action_count,
            "candidate_action_count": self.candidate_action_count,
            "exact_label_match_count": self.exact_label_match_count,
            "exact_label_recall": self.exact_label_recall,
            "exact_label_precision": self.exact_label_precision,
            "false_silence_count": self.false_silence_count,
            "candidate_repeated_label_excess_count": self.candidate_repeated_label_excess_count,
            "candidate_repeated_label_excess_rate": self.candidate_repeated_label_excess_rate,
            "all_json_syntax_valid": self.all_json_syntax_valid,
            "all_compact_contract_valid": self.all_compact_contract_valid,
            "matched_boundary_start_mae_seconds": self.matched_boundary_start_mae_seconds,
            "matched_boundary_end_mae_seconds": self.matched_boundary_end_mae_seconds,
            "matched_boundary_measurement_complete": self.matched_boundary_measurement_complete,
            "gates": self.gates,
            "qualified": self.qualified,
        }


def normalize_compact_action_label(value: str) -> str:
    return _ACTION_TOKEN.sub("_", value.strip().lower()).strip("_")


def _valid_interval(interval: object) -> bool:
    if not isinstance(interval, dict):
        return False
    relative = ("start_offset_seconds", "end_offset_seconds")
    absolute = ("start_ns", "end_ns")
    has_relative = all(key in interval for key in relative)
    has_absolute = all(key in interval for key in absolute)
    if has_relative == has_absolute:
        return False
    values = relative if has_relative else absolute
    parsed = [_finite_float(interval[key]) for key in values]
    start, end = parsed
    if start is None or end is None or start < 0 or end < 0:
        return False
    return start < end


def _valid_unit_interval(value: object) -> bool:
    parsed = _finite_float(value)
    return parsed is not None and 0.0 <= parsed <= 1.0


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def parse_compact_output(text: str) -> ParsedCompactOutput:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ParsedCompactOutput(
            json_syntax_valid=False, compact_contract_valid=False, actions=()
        )
    if not isinstance(value, dict):
        return ParsedCompactOutput(
            json_syntax_valid=False, compact_contract_valid=False, actions=()
        )
    observations = value.get("observations")
    if not isinstance(observations, list):
        return ParsedCompactOutput(json_syntax_valid=True, compact_contract_valid=False, actions=())
    contract_valid = True
    actions: list[CompactAction] = []
    for observation in observations:
        if not isinstance(observation, dict):
            contract_valid = False
            continue
        raw_label = observation.get("action")
        if not isinstance(raw_label, str) or not raw_label.strip():
            contract_valid = False
            continue
        interval = observation.get("interval")
        if not _valid_interval(interval):
            contract_valid = False
        start = end = None
        if isinstance(interval, dict):
            start = _finite_float(interval.get("start_offset_seconds"))
            end = _finite_float(interval.get("end_offset_seconds"))
        for field in ("confidence", "visibility"):
            if field in observation and not _valid_unit_interval(observation[field]):
                contract_valid = False
        actions.append(
            CompactAction(
                label=normalize_compact_action_label(raw_label),
                start_offset_seconds=start,
                end_offset_seconds=end,
            )
        )
    return ParsedCompactOutput(
        json_syntax_valid=True,
        compact_contract_valid=contract_valid,
        actions=tuple(actions),
    )


def _group_actions(actions: tuple[CompactAction, ...]) -> dict[str, list[CompactAction]]:
    grouped: dict[str, list[CompactAction]] = {}
    for action in actions:
        grouped.setdefault(action.label, []).append(action)
    for values in grouped.values():
        values.sort(
            key=lambda action: (
                action.start_offset_seconds is None,
                action.start_offset_seconds or 0.0,
                action.end_offset_seconds is None,
                action.end_offset_seconds or 0.0,
            )
        )
    return grouped


def aggregate_small_encoder_shadow_run(
    *,
    evaluations: tuple[SmallEncoderPairEvaluation, ...],
    native_generation_seconds: float,
    candidate_generation_seconds: float,
    candidate_preparation_seconds: float,
) -> SmallEncoderShadowQualification:
    """Aggregate exact-contract quality and performance without semantic overclaim."""

    if not evaluations:
        raise ValueError("at least one segment evaluation is required")
    timings = (
        native_generation_seconds,
        candidate_generation_seconds,
        candidate_preparation_seconds,
    )
    if any(value < 0 for value in timings):
        raise ValueError("generation and preparation timings must be non-negative")
    return SmallEncoderShadowQualification(
        segment_count=len(evaluations),
        native_generation_seconds=native_generation_seconds,
        candidate_generation_seconds=candidate_generation_seconds,
        candidate_preparation_seconds=candidate_preparation_seconds,
        native_action_count=sum(item.native_action_count for item in evaluations),
        candidate_action_count=sum(item.candidate_action_count for item in evaluations),
        exact_label_match_count=sum(item.exact_label_match_count for item in evaluations),
        false_silence_count=sum(int(item.false_silence) for item in evaluations),
        candidate_repeated_label_excess_count=sum(
            item.candidate_repeated_label_excess_count for item in evaluations
        ),
        all_json_syntax_valid=all(
            item.native_json_valid and item.candidate_json_valid for item in evaluations
        ),
        all_compact_contract_valid=all(
            item.native_compact_contract_valid and item.candidate_compact_contract_valid
            for item in evaluations
        ),
        matched_boundary_start_absolute_seconds=tuple(
            value for item in evaluations for value in item.matched_boundary_start_absolute_seconds
        ),
        matched_boundary_end_absolute_seconds=tuple(
            value for item in evaluations for value in item.matched_boundary_end_absolute_seconds
        ),
    )


def evaluate_small_encoder_pair(
    *, native_output_text: str, candidate_output_text: str
) -> SmallEncoderPairEvaluation:
    native = parse_compact_output(native_output_text)
    candidate = parse_compact_output(candidate_output_text)
    native_groups = _group_actions(native.actions)
    candidate_groups = _group_actions(candidate.actions)
    matches = 0
    start_deltas: list[float] = []
    end_deltas: list[float] = []
    for label in sorted(native_groups.keys() & candidate_groups.keys()):
        native_values = native_groups[label]
        candidate_values = candidate_groups[label]
        count = min(len(native_values), len(candidate_values))
        matches += count
        for native_action, candidate_action in zip(
            native_values[:count], candidate_values[:count], strict=True
        ):
            if (
                native_action.start_offset_seconds is not None
                and candidate_action.start_offset_seconds is not None
            ):
                start_deltas.append(
                    abs(native_action.start_offset_seconds - candidate_action.start_offset_seconds)
                )
            if (
                native_action.end_offset_seconds is not None
                and candidate_action.end_offset_seconds is not None
            ):
                end_deltas.append(
                    abs(native_action.end_offset_seconds - candidate_action.end_offset_seconds)
                )
    unique_candidate_labels = {action.label for action in candidate.actions}
    return SmallEncoderPairEvaluation(
        native_json_valid=native.json_syntax_valid,
        candidate_json_valid=candidate.json_syntax_valid,
        native_compact_contract_valid=native.compact_contract_valid,
        candidate_compact_contract_valid=candidate.compact_contract_valid,
        native_action_count=len(native.actions),
        candidate_action_count=len(candidate.actions),
        exact_label_match_count=matches,
        false_silence=bool(native.actions and not candidate.actions),
        candidate_repeated_label_excess_count=len(candidate.actions) - len(unique_candidate_labels),
        matched_boundary_start_absolute_seconds=tuple(start_deltas),
        matched_boundary_end_absolute_seconds=tuple(end_deltas),
    )


__all__ = [
    "MAX_MATCHED_BOUNDARY_MAE_SECONDS",
    "SMALL_ENCODER_EVALUATOR_VERSION",
    "CompactAction",
    "ParsedCompactOutput",
    "SmallEncoderPairEvaluation",
    "SmallEncoderShadowQualification",
    "aggregate_small_encoder_shadow_run",
    "evaluate_small_encoder_pair",
    "normalize_compact_action_label",
    "parse_compact_output",
]
