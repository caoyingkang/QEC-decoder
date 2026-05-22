"""Tests for ``qecbench.collector.params.CollectorParams``."""

import dataclasses

import pytest

from qecbench import CollectorParams


def test_use_multiprocessing_false_when_workers_zero():
    p = CollectorParams(
        batch_size=64, shots_cap=1000, errors_cap=100, num_parallel_workers=0
    )
    assert p.use_multiprocessing is False


def test_use_multiprocessing_true_when_workers_positive():
    p = CollectorParams(
        batch_size=64, shots_cap=1000, errors_cap=100, num_parallel_workers=2
    )
    assert p.use_multiprocessing is True


def test_frozen_dataclass_immutable():
    p = CollectorParams(
        batch_size=64, shots_cap=1000, errors_cap=100, num_parallel_workers=0
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.batch_size = 128
