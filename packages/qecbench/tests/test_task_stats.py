"""Tests for ``qecbench.task.stats.TaskStats``."""

import csv

import numpy as np
import pytest

from qecbench.task import TaskMetadata, TaskStats


# -- Helpers ---------------------------------------------------------------


def _make_update_batch():
    """Hand-crafted 4-shot batch with known counts.

    - obser_correct_mask = [T, T, F, F]
    - synd_match_mask    = [T, F, T, F]
    - decoding_iters     = [3, 5, 2, 0]
    => shots=4, obser_correct=2, synd_matches=2, success=1 (index 0 only)
    """
    return (
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
        np.array([3, 5, 2, 0], dtype=np.int64),
    )


# -- __post_init__ ---------------------------------------------------------


def test_iterative_init_creates_zero_histograms(bp_metadata):
    stats = TaskStats(metadata=bp_metadata)
    expected_len = bp_metadata.max_iter + 1
    assert stats.iters_hist_on_converged is not None
    assert stats.iters_hist_on_success is not None
    assert stats.iters_hist_on_converged.shape == (expected_len,)
    assert stats.iters_hist_on_success.shape == (expected_len,)
    assert int(stats.iters_hist_on_converged.sum()) == 0
    assert int(stats.iters_hist_on_success.sum()) == 0


def test_noniterative_keeps_histograms_none(mwpm_metadata):
    stats = TaskStats(metadata=mwpm_metadata)
    assert stats.iters_hist_on_converged is None
    assert stats.iters_hist_on_success is None


# -- update ----------------------------------------------------------------


def test_update_iterative_accumulates_counts_and_hist(bp_metadata):
    stats = TaskStats(metadata=bp_metadata)
    obser, synd, iters = _make_update_batch()
    stats.update(obser, synd, iters)

    assert stats.shots == 4
    assert stats.obser_correct == 2
    assert stats.synd_matches == 2
    assert stats.success == 1
    # iters_hist_on_converged: one shot at iter 3, one at iter 2
    assert stats.iters_hist_on_converged[3] == 1
    assert stats.iters_hist_on_converged[2] == 1
    assert int(stats.iters_hist_on_converged.sum()) == 2
    # iters_hist_on_success: only the shot at iter 3
    assert stats.iters_hist_on_success[3] == 1
    assert int(stats.iters_hist_on_success.sum()) == 1


def test_update_noniterative_ignores_iters(mwpm_metadata):
    stats = TaskStats(metadata=mwpm_metadata)
    obser, synd, _ = _make_update_batch()
    stats.update(obser, synd, decoding_iters=None)
    assert stats.shots == 4
    assert stats.obser_correct == 2
    assert stats.synd_matches == 2
    assert stats.success == 1
    assert stats.iters_hist_on_converged is None
    assert stats.iters_hist_on_success is None


def test_update_multiple_batches_accumulate(bp_metadata):
    stats = TaskStats(metadata=bp_metadata)
    obser, synd, iters = _make_update_batch()
    stats.update(obser, synd, iters)
    stats.update(obser, synd, iters)
    assert stats.shots == 8
    assert stats.obser_correct == 4
    assert stats.synd_matches == 4
    assert stats.success == 2
    assert stats.iters_hist_on_converged[3] == 2
    assert stats.iters_hist_on_converged[2] == 2


# -- Derived properties ----------------------------------------------------


def test_derived_count_properties(bp_metadata):
    stats = TaskStats(metadata=bp_metadata)
    obser, synd, iters = _make_update_batch()
    stats.update(obser, synd, iters)
    assert stats.obser_errors == 2  # 4 - 2
    assert stats.synd_mismatches == 2
    assert stats.failures == 3  # 4 - 1


def test_rates_with_zero_shots_are_nan(bp_metadata):
    stats = TaskStats(metadata=bp_metadata)
    assert np.isnan(stats.logical_error_rate)
    assert np.isnan(stats.syndrome_mismatch_rate)
    assert np.isnan(stats.failure_rate)
    assert np.isnan(stats.avg_iters)
    assert np.isnan(stats.avg_iters_on_converged)
    assert np.isnan(stats.avg_iters_on_success)


def test_rates_match_counts(bp_metadata):
    stats = TaskStats(metadata=bp_metadata)
    obser, synd, iters = _make_update_batch()
    stats.update(obser, synd, iters)
    assert stats.logical_error_rate == pytest.approx(2 / 4)
    assert stats.syndrome_mismatch_rate == pytest.approx(2 / 4)
    assert stats.failure_rate == pytest.approx(3 / 4)


def test_avg_iters_on_converged_value(bp_metadata):
    stats = TaskStats(metadata=bp_metadata)
    obser, synd, iters = _make_update_batch()
    stats.update(obser, synd, iters)
    # Converged iters are [3, 2]: average = 2.5
    assert stats.avg_iters_on_converged == pytest.approx(2.5)
    # Only success iter is [3]: average = 3.0
    assert stats.avg_iters_on_success == pytest.approx(3.0)


def test_avg_iters_on_success_nan_when_no_success(bp_metadata):
    stats = TaskStats(metadata=bp_metadata)
    # synd_match all-True but obser_correct all-False => success=0
    stats.update(
        np.array([False, False]),
        np.array([True, True]),
        np.array([2, 4], dtype=np.int64),
    )
    assert stats.success == 0
    assert np.isnan(stats.avg_iters_on_success)


def test_avg_iters_includes_unconverged_at_max_iter(bp_metadata):
    stats = TaskStats(metadata=bp_metadata)
    obser, synd, iters = _make_update_batch()
    stats.update(obser, synd, iters)
    # Sum of converged iters = 3 + 2 = 5; unconverged = 2 shots * 20 = 40.
    # Total iters = 45 over 4 shots.
    assert stats.avg_iters == pytest.approx(45 / 4)


def test_avg_iters_raises_for_noniterative(mwpm_metadata):
    stats = TaskStats(metadata=mwpm_metadata)
    with pytest.raises(ValueError, match="Unsupported"):
        _ = stats.avg_iters
    with pytest.raises(ValueError, match="Unsupported"):
        _ = stats.avg_iters_on_converged
    with pytest.raises(ValueError, match="Unsupported"):
        _ = stats.avg_iters_on_success


# -- is_complete -----------------------------------------------------------


def test_is_complete_by_shots_cap(bp_metadata):
    stats = TaskStats(metadata=bp_metadata, shots=100, obser_correct=100)
    assert stats.is_complete(shots_cap=100, errors_cap=1_000_000)
    assert not stats.is_complete(shots_cap=101, errors_cap=1_000_000)


def test_is_complete_by_errors_cap(bp_metadata):
    stats = TaskStats(metadata=bp_metadata, shots=50, obser_correct=40)
    # obser_errors == 10
    assert stats.is_complete(shots_cap=10_000, errors_cap=10)
    assert not stats.is_complete(shots_cap=10_000, errors_cap=11)


# -- merge -----------------------------------------------------------------


def test_merge_accumulates(bp_metadata):
    a = TaskStats(metadata=bp_metadata)
    b = TaskStats(metadata=bp_metadata)
    obser, synd, iters = _make_update_batch()
    a.update(obser, synd, iters)
    b.update(obser, synd, iters)
    a.merge(b)
    assert a.shots == 8
    assert a.obser_correct == 4
    assert a.synd_matches == 4
    assert a.success == 2
    assert a.iters_hist_on_converged[3] == 2
    assert a.iters_hist_on_success[3] == 2


def test_merge_different_metadata_raises(bp_metadata, mwpm_metadata):
    a = TaskStats(metadata=bp_metadata)
    b = TaskStats(metadata=mwpm_metadata)
    with pytest.raises(ValueError, match="different metadata"):
        a.merge(b)


# -- CSV I/O ---------------------------------------------------------------


def _populated_iterative_stats(metadata):
    stats = TaskStats(metadata=metadata)
    obser, synd, iters = _make_update_batch()
    stats.update(obser, synd, iters)
    return stats


def _populated_noniterative_stats(metadata):
    stats = TaskStats(metadata=metadata)
    obser, synd, _ = _make_update_batch()
    stats.update(obser, synd, decoding_iters=None)
    return stats


def test_save_load_csv_roundtrip_iterative(tmp_path, bp_metadata):
    path = tmp_path / "stats.csv"
    original = _populated_iterative_stats(bp_metadata)
    TaskStats.save_csv([original], path)
    loaded_list = TaskStats.load_csv(path)
    assert len(loaded_list) == 1
    loaded = loaded_list[0]
    assert loaded.metadata == original.metadata
    assert loaded.shots == original.shots
    assert loaded.obser_correct == original.obser_correct
    assert loaded.synd_matches == original.synd_matches
    assert loaded.success == original.success
    np.testing.assert_array_equal(
        loaded.iters_hist_on_converged, original.iters_hist_on_converged
    )
    np.testing.assert_array_equal(
        loaded.iters_hist_on_success, original.iters_hist_on_success
    )


def test_save_load_csv_roundtrip_noniterative(tmp_path, mwpm_metadata):
    path = tmp_path / "stats.csv"
    original = _populated_noniterative_stats(mwpm_metadata)
    TaskStats.save_csv([original], path)
    loaded_list = TaskStats.load_csv(path)
    assert len(loaded_list) == 1
    loaded = loaded_list[0]
    assert loaded.metadata == original.metadata
    assert loaded.shots == original.shots
    assert loaded.iters_hist_on_converged is None
    assert loaded.iters_hist_on_success is None


def test_save_csv_creates_parent_directories(tmp_path, bp_metadata):
    path = tmp_path / "nested" / "dir" / "stats.csv"
    stats = _populated_iterative_stats(bp_metadata)
    TaskStats.save_csv([stats], path)
    assert path.exists()


def test_load_csv_missing_file_returns_empty_list(tmp_path):
    path = tmp_path / "does_not_exist.csv"
    assert TaskStats.load_csv(path) == []


def test_load_csv_schema_mismatch_raises(tmp_path, bp_metadata):
    path = tmp_path / "stats.csv"
    stats = _populated_iterative_stats(bp_metadata)
    TaskStats.save_csv([stats], path)

    # Read back, mutate SCHEMA_VERSION, write out again.
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        row["SCHEMA_VERSION"] = "999"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="schema_version"):
        TaskStats.load_csv(path)


# -- List ops --------------------------------------------------------------


def test_find_by_metadata_hit(bp_metadata, mwpm_metadata):
    a = TaskStats(metadata=bp_metadata)
    b = TaskStats(metadata=mwpm_metadata)
    found = TaskStats.find_by_metadata([a, b], bp_metadata)
    assert found is a


def test_find_by_metadata_miss_returns_none(mwpm_metadata):
    other = TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params={"d": 3, "rounds": 3},
        error_rate=0.10,  # different error rate
        decoder_name="MWPM",
        decoder_params={},
    )
    a = TaskStats(metadata=mwpm_metadata)
    assert TaskStats.find_by_metadata([a], other) is None


def test_upsert_replaces_existing_with_same_metadata(bp_metadata):
    old = TaskStats(metadata=bp_metadata, shots=5, obser_correct=5)
    new = TaskStats(metadata=bp_metadata, shots=10, obser_correct=10)
    lst = [old]
    TaskStats.upsert(lst, new)
    assert len(lst) == 1
    assert lst[0] is new


def test_upsert_appends_new(bp_metadata, mwpm_metadata):
    a = TaskStats(metadata=bp_metadata)
    b = TaskStats(metadata=mwpm_metadata)
    lst = [a]
    TaskStats.upsert(lst, b)
    assert lst == [a, b]
