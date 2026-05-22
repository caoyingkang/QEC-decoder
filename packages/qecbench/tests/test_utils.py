"""Tests for ``qecbench.utils``."""

import numpy as np

from qecbench.utils import _sample


def test_sample_shapes_and_dtypes(rep_code_circuit):
    sampler = rep_code_circuit.stim_dem.compile_sampler(seed=1)
    batch_size = 16
    syndromes, observables = _sample(sampler, batch_size)
    assert syndromes.shape == (batch_size, rep_code_circuit.num_detectors)
    assert observables.shape == (batch_size, rep_code_circuit.num_observables)
    assert syndromes.dtype == np.uint8
    assert observables.dtype == np.uint8
