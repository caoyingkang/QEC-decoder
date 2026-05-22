"""Shared pytest fixtures for the qecbench test suite."""

import numpy as np
import pytest
from qecdec.circuits import RepetitionCode_Circuit
from qecdec.decoders import BPDecoder, MWPMDecoder

from qecbench.task import TaskMetadata


_CIRCUIT_KWARGS = {"d": 3, "rounds": 3}
_ERROR_RATE = 0.05
_BP_DECODER_PARAMS = {"max_iter": 20, "norm": 0.9}


@pytest.fixture(scope="module")
def rep_code_circuit():
    """Tiny repetition-code memory circuit (d=3, 3 rounds, uniform p=0.05)."""
    return RepetitionCode_Circuit.with_uniform_error_rate(
        _ERROR_RATE, **_CIRCUIT_KWARGS
    )


@pytest.fixture(scope="module")
def rep_code_syndromes(rep_code_circuit):
    """32 sampled (syndromes, observables) pairs as uint8 arrays."""
    sampler = rep_code_circuit.stim_dem.compile_sampler(seed=0)
    syndromes, observables, _ = sampler.sample(32)
    return syndromes.astype(np.uint8), observables.astype(np.uint8)


@pytest.fixture(scope="module")
def bp_metadata():
    return TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params=_CIRCUIT_KWARGS,
        error_rate=_ERROR_RATE,
        decoder_name="BP",
        decoder_params=_BP_DECODER_PARAMS,
    )


@pytest.fixture(scope="module")
def mwpm_metadata():
    return TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params=_CIRCUIT_KWARGS,
        error_rate=_ERROR_RATE,
        decoder_name="MWPM",
        decoder_params={},
    )


@pytest.fixture(scope="module")
def bp_decoder(rep_code_circuit):
    return BPDecoder(
        rep_code_circuit.chkmat, rep_code_circuit.prior, **_BP_DECODER_PARAMS
    )


@pytest.fixture(scope="module")
def mwpm_decoder(rep_code_circuit):
    return MWPMDecoder(rep_code_circuit.chkmat, rep_code_circuit.prior)
