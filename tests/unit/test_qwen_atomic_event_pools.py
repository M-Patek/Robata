"""Tests for the small frozen raw-first Qwen experiment pools."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from robata.benchmark.qwen_atomic_event_pools import (
    C24,
    D12,
    FROZEN_QWEN_ATOMIC_EVENT_POOLS,
    H8,
    M9,
    AtomicEventStratum,
    FrozenAtomicEventPoolCase,
    FrozenPoolPartitionError,
    FrozenPoolRecordResolutionError,
    QwenAtomicEventPoolName,
    get_frozen_pool,
    resolve_pool_records,
    validate_pool_partition,
)


def _pools(
    *,
    h8: tuple[FrozenAtomicEventPoolCase, ...] = H8,
    d12: tuple[FrozenAtomicEventPoolCase, ...] = D12,
    c24: tuple[FrozenAtomicEventPoolCase, ...] = C24,
    m9: tuple[FrozenAtomicEventPoolCase, ...] = M9,
) -> dict[str, tuple[FrozenAtomicEventPoolCase, ...]]:
    return {"H8": h8, "D12": d12, "C24": c24, "M9": m9}


def test_frozen_pools_have_declared_order_sizes_strata_and_historical_roles() -> None:
    assert [case.uid for case in H8] == [
        "P37_102_15",
        "P35_101_13",
        "P37_102_33",
        "P37_102_35",
        "P37_102_46",
        "P35_101_78",
        "P35_101_81",
        "P35_101_90",
    ]
    assert len(D12) == 12
    assert len(C24) == 24
    assert len(M9) == 9
    assert [case.uid for case in D12] == [
        "P01_13_14",
        "P01_13_19",
        "P06_14_0",
        "P09_08_1",
        "P09_08_3",
        "P20_07_23",
        "P01_13_23",
        "P06_14_6",
        "P20_07_7",
        "P09_08_12",
        "P09_08_13",
        "P20_07_19",
    ]
    assert [case.uid for case in C24] == [
        "P03_25_12",
        "P03_25_14",
        "P11_21_6",
        "P26_30_2",
        "P28_23_5",
        "P35_101_90",
        "P02_15_41",
        "P02_15_49",
        "P03_26_1",
        "P03_26_3",
        "P28_23_1",
        "P28_23_3",
        "P02_15_10",
        "P03_25_9",
        "P11_18_1",
        "P11_18_16",
        "P26_30_1",
        "P28_23_2",
        "P02_15_21",
        "P03_25_1",
        "P04_26_1",
        "P11_18_12",
        "P35_101_81",
        "P37_102_46",
    ]
    assert {case.stratum for case in D12} == set(AtomicEventStratum)
    assert {case.stratum for case in C24} == set(AtomicEventStratum)
    assert sum(case.historical for case in H8) == 8
    assert sum(case.historical for case in D12) == 0
    assert sum(case.historical for case in C24) == 3
    assert {case.uid for case in C24 if case.historical} == {
        "P35_101_90",
        "P35_101_81",
        "P37_102_46",
    }
    assert [case.uid for case in M9] == [
        "P03_25_12",
        "P03_25_14",
        "P28_23_3",
        "P02_15_10",
        "P11_18_1",
        "P26_30_1",
        "P04_26_1",
        "P35_101_81",
        "P37_102_46",
    ]


def test_pool_data_and_mapping_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        H8[0].uid = "P99_99_99"  # type: ignore[misc]
    with pytest.raises(TypeError):
        FROZEN_QWEN_ATOMIC_EVENT_POOLS[QwenAtomicEventPoolName.H8] = ()  # type: ignore[index]


def test_default_partition_is_valid_and_pool_lookup_accepts_name_or_string() -> None:
    assert validate_pool_partition() is None
    assert get_frozen_pool(QwenAtomicEventPoolName.D12) is D12
    assert get_frozen_pool("C24") is C24
    with pytest.raises(FrozenPoolPartitionError, match="unknown frozen pool"):
        get_frozen_pool("unknown")


def test_partition_rejects_duplicate_uid_inside_a_pool() -> None:
    duplicate_d12 = (*D12[:-1], D12[0])

    with pytest.raises(FrozenPoolPartitionError, match=r"D12: duplicate UIDs: P01_13_14"):
        validate_pool_partition(_pools(d12=duplicate_d12))


def test_partition_rejects_d12_c24_uid_overlap() -> None:
    shared_uid = replace(
        C24[0],
        uid=D12[0].uid,
        participant_id=D12[0].participant_id,
        video_id=D12[0].video_id,
    )
    mutated_c24 = (shared_uid, *C24[1:])

    with pytest.raises(FrozenPoolPartitionError, match=r"must not share UIDs: P01_13_14"):
        validate_pool_partition(_pools(c24=mutated_c24))


def test_partition_rejects_d12_c24_video_group_overlap() -> None:
    overlapping_video = replace(
        C24[4],
        uid="P01_13_99",
        participant_id="P01",
        video_id="P01_13",
    )
    mutated_c24 = (*C24[:4], overlapping_video, *C24[5:])

    with pytest.raises(FrozenPoolPartitionError, match=r"must not share video groups: P01_13"):
        validate_pool_partition(_pools(c24=mutated_c24))


def test_partition_rejects_d12_c24_participant_overlap_even_for_new_video() -> None:
    overlapping_participant = replace(
        C24[4],
        uid="P01_99_99",
        participant_id="P01",
        video_id="P01_99",
    )
    mutated_c24 = (*C24[:4], overlapping_participant, *C24[5:])

    with pytest.raises(FrozenPoolPartitionError, match=r"must not share participants: P01"):
        validate_pool_partition(_pools(c24=mutated_c24))


def test_partition_rejects_m9_case_not_in_c24() -> None:
    outside_c24 = replace(
        M9[0],
        uid="P99_99_1",
        participant_id="P99",
        video_id="P99_99",
    )
    mutated_m9 = (outside_c24, *M9[1:])

    with pytest.raises(FrozenPoolPartitionError, match=r"M9 must be a C24 UID subset"):
        validate_pool_partition(_pools(m9=mutated_m9))


def test_partition_rejects_m9_metadata_mismatch_for_a_c24_uid() -> None:
    mismatched_m9 = (
        replace(M9[0], stratum=AtomicEventStratum.OPEN_CLOSE),
        M9[1],
        replace(M9[2], stratum=AtomicEventStratum.SWITCH_DIRECTION),
        *M9[3:],
    )

    with pytest.raises(FrozenPoolPartitionError, match=r"M9 metadata must match its C24 cases"):
        validate_pool_partition(_pools(m9=mismatched_m9))


def test_partition_rejects_invalid_case_identity_and_pool_shape() -> None:
    invalid_identity = replace(C24[0], participant_id="P98")
    with pytest.raises(FrozenPoolPartitionError, match=r"participant_id must be P03"):
        validate_pool_partition(_pools(c24=(invalid_identity, *C24[1:])))

    with pytest.raises(FrozenPoolPartitionError, match=r"D12: expected 12 cases, got 11"):
        validate_pool_partition(_pools(d12=D12[:-1]))


def test_resolve_pool_records_orders_shuffled_iterable_by_uid_without_reading_labels() -> None:
    records = [
        {
            "uid": case.uid,
            "raw_output_text": "this must remain untouched",
            "official_reference": {"verb": "intentionally unrelated"},
        }
        for case in reversed(D12)
    ]

    resolved = resolve_pool_records("D12", records)

    assert [record["uid"] for record in resolved] == [case.uid for case in D12]
    assert all("posthoc_official_action" not in record for record in resolved)
    assert all(
        record["official_reference"]["verb"] == "intentionally unrelated" for record in resolved
    )


def test_resolve_pool_records_accepts_a_mapping_and_custom_uid_getter() -> None:
    mapping = {case.uid: {"opaque": case.uid} for case in M9}
    assert resolve_pool_records(QwenAtomicEventPoolName.M9, mapping) == tuple(
        mapping[case.uid] for case in M9
    )

    class Record:
        def __init__(self, identifier: str) -> None:
            self.identifier = identifier

    records = [Record(case.uid) for case in reversed(M9)]
    resolved = resolve_pool_records(M9, records, uid_getter=lambda record: record.identifier)
    assert [record.identifier for record in resolved] == [case.uid for case in M9]


def test_resolve_pool_records_rejects_missing_duplicate_and_unaddressable_records() -> None:
    with pytest.raises(
        FrozenPoolRecordResolutionError,
        match=r"H8: records missing for UIDs: P35_101_90",
    ):
        resolve_pool_records("H8", [{"uid": case.uid} for case in H8[:-1]])

    with pytest.raises(FrozenPoolRecordResolutionError, match=r"records contain duplicate UID"):
        resolve_pool_records("H8", [{"uid": H8[0].uid}, {"uid": H8[0].uid}])

    with pytest.raises(FrozenPoolRecordResolutionError, match=r"require a uid field"):
        resolve_pool_records("H8", [object()])


def test_resolve_pool_records_rejects_duplicate_uid_in_a_custom_pool() -> None:
    duplicate_pool = (H8[0], H8[0])

    with pytest.raises(FrozenPoolPartitionError, match=r"provided pool: duplicate UIDs"):
        resolve_pool_records(duplicate_pool, {H8[0].uid: {"uid": H8[0].uid}})
