"""Strict shared values used at every application boundary."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    StringConstraints,
    model_validator,
)
from pydantic_core import PydanticCustomError, core_schema

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

_CANONICAL_DECIMAL_PATTERN = re.compile(r"^(?:0|-?[1-9][0-9]*)$")


class StrictModel(BaseModel):
    """Base contract model: strict, immutable, and closed to unknown fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


# Kept as a descriptive alias for call sites that prefer domain terminology.
ContractModel = StrictModel


def _parse_nanosecond_string(value: str) -> int:
    if _CANONICAL_DECIMAL_PATTERN.fullmatch(value) is None:
        raise PydanticCustomError(
            "nanoseconds_format",
            "Nanoseconds must be a canonical base-10 integer string",
        )

    parsed = int(value)
    if parsed < INT64_MIN or parsed > INT64_MAX:
        raise PydanticCustomError(
            "nanoseconds_range",
            "Nanoseconds must fit in a signed 64-bit integer",
        )
    return parsed


class _NanosecondsAnnotation:
    """Use distinct JSON and Python schemas so JSON numbers can never be coerced."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        encoded_string = core_schema.no_info_after_validator_function(
            _parse_nanosecond_string,
            core_schema.str_schema(
                strict=True,
                min_length=1,
                max_length=20,
                pattern=_CANONICAL_DECIMAL_PATTERN.pattern,
            ),
        )
        python_value = core_schema.union_schema(
            [
                core_schema.int_schema(strict=True, ge=INT64_MIN, le=INT64_MAX),
                encoded_string,
            ],
        )
        return core_schema.json_or_python_schema(
            json_schema=encoded_string,
            python_schema=python_value,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str,
                return_schema=core_schema.str_schema(),
                when_used="json",
            ),
        )


Nanoseconds = Annotated[int, _NanosecondsAnnotation]
"""Signed int64 nanoseconds internally and canonical decimal strings on JSON wires."""

Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
"""A lowercase hexadecimal SHA-256 digest."""

SchemaVersion = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._+-]*[A-Za-z0-9])?$",
    ),
]


class NanosecondInterval(StrictModel):
    """A nonempty half-open interval ``[start_ns, end_ns)``."""

    start_ns: Nanoseconds
    end_ns: Nanoseconds

    @model_validator(mode="after")
    def validate_nonempty(self) -> NanosecondInterval:
        if self.start_ns >= self.end_ns:
            raise ValueError("start_ns must be less than end_ns")
        return self

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def contains(self, timestamp_ns: int) -> bool:
        return self.start_ns <= timestamp_ns < self.end_ns


class ValidationSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class ValidationIssue(StrictModel):
    """A stable, machine-readable contract or semantic validation finding."""

    code: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]
    message: Annotated[str, StringConstraints(strict=True, min_length=1)]
    severity: ValidationSeverity = ValidationSeverity.ERROR
    path: tuple[str | int, ...] = ()


class ValidationOutcome(StrictModel):
    """Immutable collection of validation findings."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return all(issue.severity is not ValidationSeverity.ERROR for issue in self.issues)
