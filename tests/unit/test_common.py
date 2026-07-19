import json

import pytest
from pydantic import ValidationError

from robata.contracts.common import (
    INT64_MAX,
    INT64_MIN,
    NanosecondInterval,
    Nanoseconds,
    StrictModel,
)


class TimestampEnvelope(StrictModel):
    timestamp_ns: Nanoseconds


@pytest.mark.parametrize("value", [INT64_MIN, -1, 0, 1, INT64_MAX])
def test_nanoseconds_round_trip_as_json_string(value: int) -> None:
    model = TimestampEnvelope.model_validate_json(f'{{"timestamp_ns":"{value}"}}')

    assert model.timestamp_ns == value
    assert json.loads(model.model_dump_json()) == {"timestamp_ns": str(value)}
    assert model.model_dump() == {"timestamp_ns": value}


@pytest.mark.parametrize("value", [INT64_MIN, 0, INT64_MAX])
def test_nanoseconds_accept_strict_python_ints(value: int) -> None:
    assert TimestampEnvelope(timestamp_ns=value).timestamp_ns == value


@pytest.mark.parametrize(
    "payload",
    [
        '{"timestamp_ns":0}',
        '{"timestamp_ns":1.0}',
        '{"timestamp_ns":true}',
        '{"timestamp_ns":"+1"}',
        '{"timestamp_ns":"01"}',
        '{"timestamp_ns":"-0"}',
        '{"timestamp_ns":" 1"}',
        '{"timestamp_ns":"1.0"}',
        f'{{"timestamp_ns":"{INT64_MIN - 1}"}}',
        f'{{"timestamp_ns":"{INT64_MAX + 1}"}}',
    ],
)
def test_nanoseconds_reject_noncanonical_or_out_of_range_json(payload: str) -> None:
    with pytest.raises(ValidationError):
        TimestampEnvelope.model_validate_json(payload)


@pytest.mark.parametrize("value", [True, 1.0, INT64_MIN - 1, INT64_MAX + 1])
def test_nanoseconds_reject_invalid_python_values(value: object) -> None:
    with pytest.raises(ValidationError):
        TimestampEnvelope(timestamp_ns=value)  # type: ignore[arg-type]


def test_interval_is_nonempty_half_open_and_serializes_wire_values() -> None:
    interval = NanosecondInterval(start_ns=-1, end_ns=2)

    assert interval.duration_ns == 3
    assert interval.contains(-1)
    assert interval.contains(1)
    assert not interval.contains(2)
    assert json.loads(interval.model_dump_json()) == {"start_ns": "-1", "end_ns": "2"}


@pytest.mark.parametrize("start_ns,end_ns", [(0, 0), (1, 0)])
def test_interval_rejects_empty_or_reversed_bounds(start_ns: int, end_ns: int) -> None:
    with pytest.raises(ValidationError):
        NanosecondInterval(start_ns=start_ns, end_ns=end_ns)


def test_contract_models_are_frozen_and_forbid_extra_fields() -> None:
    envelope = TimestampEnvelope(timestamp_ns=1)

    with pytest.raises(ValidationError):
        TimestampEnvelope.model_validate({"timestamp_ns": 1, "surprise": True})
    with pytest.raises(ValidationError):
        envelope.timestamp_ns = 2
