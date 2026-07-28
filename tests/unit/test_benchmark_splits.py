from __future__ import annotations

import pytest

from robata.benchmark.models import StratificationDimension
from robata.benchmark.splits import (
    DataSplitResult,
    DataSplitter,
    SplitConfig,
    SplitMetadataError,
    SplitRecord,
)


def _splitter() -> DataSplitter:
    return DataSplitter(
        SplitConfig(
            version="split-v1",
            development_ratio=0.5,
            validation_ratio=0.3,
            frozen_test_ratio=0.2,
            random_seed=17,
        )
    )


def _records() -> list[SplitRecord]:
    return [
        SplitRecord(
            mcap_id=f"mcap-{index}",
            session_id=session,
            actor=actor,
            scene=f"scene-{index % 2}",
            collection_day=f"2026-07-{index + 1:02d}",
            rig=f"rig-{index % 2}",
            strata={"action": "rare" if index == 0 else "common"},
        )
        for index, (session, actor) in enumerate(
            (
                ("session-0", "actor-0"),
                ("session-0", "actor-1"),
                ("session-1", "actor-0"),
                ("session-2", "actor-2"),
                ("session-3", "actor-3"),
                ("session-4", "actor-4"),
                ("session-5", "actor-5"),
                ("session-6", "actor-6"),
            )
        )
    ]


def _dimension() -> StratificationDimension:
    return StratificationDimension(
        dimension="action",
        values=("common", "rare"),
        proportions=(0.75, 0.25),
    )


def test_legacy_split_is_deterministic_but_non_certifying() -> None:
    splitter = _splitter()
    ids = [f"mcap-{index}" for index in range(7)]
    first = splitter.split(ids)
    second = splitter.split(tuple(reversed(ids)))

    assert first == second
    assert first.grouping_metadata_complete is False
    assert first.stratification_report["development"]["target_count"] == 4
    assert first.stratification_report["validation"]["target_count"] == 2
    assert first.stratification_report["frozen_test"]["target_count"] == 1
    assert not any(
        key.startswith("stratum:") for row in first.stratification_report.values() for key in row
    )
    assert splitter.validate_no_leakage(first)


def test_requested_strata_without_records_fails_closed() -> None:
    with pytest.raises(SplitMetadataError, match="records are required"):
        _splitter().split(["mcap-0"], (_dimension(),))


def test_grouped_split_keeps_session_and_shared_actor_together() -> None:
    records = _records()
    ids = [record.mcap_id for record in records]
    result = _splitter().split(ids, (_dimension(),), records=records)

    membership = {
        mcap_id: split_name
        for split_name, values in (
            ("development", result.development),
            ("validation", result.validation),
            ("frozen_test", result.frozen_test),
        )
        for mcap_id in values
    }
    assert result.grouping_metadata_complete is True
    assert membership["mcap-0"] == membership["mcap-1"] == membership["mcap-2"]
    assert _splitter().validate_no_leakage(result)
    assert _splitter().validate_no_leakage(result, records=records)


def test_report_contains_observed_stratum_counts_not_synthetic_equal_counts() -> None:
    records = _records()
    result = _splitter().split(
        [record.mcap_id for record in records],
        (_dimension(),),
        records=records,
    )
    observed = {
        value: sum(record.strata["action"] == value for record in records)
        for value in ("common", "rare")
    }
    for value, expected in observed.items():
        actual = sum(
            row[f"stratum:action:{value}"] for row in result.stratification_report.values()
        )
        rounded_target = sum(
            row[f"stratum_target:action:{value}"] for row in result.stratification_report.values()
        )
        assert actual == expected
        assert rounded_target == expected


def test_invalid_or_incomplete_record_metadata_is_rejected() -> None:
    with pytest.raises(SplitMetadataError, match="invalid split record"):
        _splitter().split(
            ["mcap-0"],
            records=[{"mcap_id": "mcap-0", "session_id": "s0"}],
        )
    incomplete = _records()
    incomplete[0] = incomplete[0].model_copy(update={"strata": {}})
    with pytest.raises(SplitMetadataError, match="lacks stratum"):
        _splitter().split(
            [record.mcap_id for record in incomplete],
            (_dimension(),),
            records=incomplete,
        )


def test_metadata_mapping_alias_is_supported_and_id_conflicts_fail_closed() -> None:
    records = _records()
    metadata = {record.mcap_id: record.model_dump(exclude={"mcap_id"}) for record in records}
    ids = [record.mcap_id for record in records]

    from_records = _splitter().split(ids, (_dimension(),), records=records)
    from_metadata = _splitter().split(ids, (_dimension(),), metadata=metadata)
    assert from_metadata == from_records

    metadata["mcap-0"]["mcap_id"] = "different-id"
    with pytest.raises(SplitMetadataError, match="conflicts"):
        _splitter().split(ids, (_dimension(),), metadata=metadata)
    with pytest.raises(SplitMetadataError, match="not both"):
        _splitter().split(ids, records=records, metadata=metadata)


def test_calibration_protocol_binds_grouped_roles_and_excludes_frozen_test_from_fit() -> None:
    splitter = _splitter()
    records = [
        SplitRecord(
            mcap_id=f"mcap-{index}",
            session_id=f"session-{index}",
            actor=None,
            scene=None,
            collection_day=None,
            rig=None,
        )
        for index in range(4)
    ]
    result = DataSplitResult(
        development=("mcap-0",),
        validation=("mcap-1",),
        frozen_test=("mcap-2", "mcap-3"),
        stratification_report={},
        grouping_metadata_complete=True,
        leakage_group_ids={
            record.mcap_id: f"group-{index}" for index, record in enumerate(records)
        },
    )

    protocol = splitter.calibration_protocol(result, records=records)

    assert result.calibration == result.validation
    assert protocol.development_mcap_ids == result.development
    assert protocol.calibration_mcap_ids == result.validation
    assert protocol.frozen_test_mcap_ids == result.frozen_test
    assert set(protocol.fitting_mcap_ids) == set(result.development) | set(result.validation)
    assert not set(protocol.fitting_mcap_ids) & set(protocol.frozen_test_mcap_ids)
    assert protocol.protocol_identity == f"calibration-split-protocol:{protocol.protocol_digest}"


def test_calibration_protocol_rejects_legacy_or_leaking_splits() -> None:
    splitter = _splitter()
    records = _records()
    legacy = splitter.split([record.mcap_id for record in records])
    with pytest.raises(SplitMetadataError, match="complete grouping metadata"):
        splitter.calibration_protocol(legacy)

    leaking = DataSplitResult(
        development=("mcap-0",),
        validation=("mcap-1",),
        frozen_test=tuple(record.mcap_id for record in records[2:]),
        stratification_report={},
        grouping_metadata_complete=True,
    )
    with pytest.raises(SplitMetadataError, match="leakage"):
        splitter.calibration_protocol(leaking, records=records)


def test_validate_no_leakage_checks_group_metadata_when_supplied() -> None:
    records = _records()
    bad = DataSplitResult(
        development=("mcap-0",),
        validation=("mcap-1",),
        frozen_test=tuple(
            record.mcap_id for record in records if record.mcap_id not in {"mcap-0", "mcap-1"}
        ),
        stratification_report={},
    )
    assert not _splitter().validate_no_leakage(bad, records=records)
