"""Shared pytest fixtures for the qecdec test suite."""

import numpy as np
import pytest

from qecdec.experiments import RotatedSurfaceCode_Memory


@pytest.fixture(scope="module")
def rsc_d5():
    """Rotated surface-code memory experiment (d=5, 5 rounds, Z basis)."""
    return RotatedSurfaceCode_Memory(
        d=5,
        rounds=5,
        basis="Z",
        data_qubit_error_rate=0.05,
        meas_error_rate=0.05,
    )


@pytest.fixture(scope="module")
def rsc_d5_data(rsc_d5):
    """64 sampled syndromes plus PCM / prior / shape metadata for the d=5 RSC."""
    sampler = rsc_d5.dem.compile_sampler()
    syndromes, _, _ = sampler.sample(64)
    return {
        "expmt": rsc_d5,
        "syndromes": syndromes.astype(np.uint8),
        "pcm": rsc_d5.chkmat,
        "prior": rsc_d5.prior,
        "num_vars": rsc_d5.num_error_mechanisms,
        "num_chks": rsc_d5.chkmat.shape[0],
    }


@pytest.fixture(scope="module")
def gamma_array(rsc_d5_data):
    """A reproducible per-VN gamma vector for DMemBP / DMemOffsetBP tests."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.25, 0.95, rsc_d5_data["num_vars"])
