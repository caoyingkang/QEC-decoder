"""Tests for ``qecbench.decoder_wrapper._BenchmarkDecoder``."""

import numpy as np

from qecbench.decoder_wrapper import _BenchmarkDecoder


def test_iterative_decode_shapes_and_dtypes(
    rep_code_circuit, rep_code_syndromes, bp_decoder
):
    syndromes, observables = rep_code_syndromes
    batch_size = syndromes.shape[0]
    wrapper = _BenchmarkDecoder(bp_decoder, rep_code_circuit.obsmat)
    result = wrapper.decode(syndromes, observables)

    assert result.obser_correct_mask.shape == (batch_size,)
    assert result.obser_correct_mask.dtype == np.bool_
    assert result.synd_match_mask.shape == (batch_size,)
    assert result.synd_match_mask.dtype == np.bool_
    assert result.decoding_iters is not None
    assert result.decoding_iters.shape == (batch_size,)
    assert np.issubdtype(result.decoding_iters.dtype, np.integer)


def test_iterative_masks_match_underlying_decoder(
    rep_code_circuit, rep_code_syndromes, bp_decoder
):
    syndromes, observables = rep_code_syndromes
    obsmat = rep_code_circuit.obsmat
    wrapper = _BenchmarkDecoder(bp_decoder, obsmat)
    result = wrapper.decode(syndromes, observables)

    ehat, conv_mask, iters = bp_decoder.decode_batch_detailed(syndromes, parallel=True)
    obser_pred = (ehat @ obsmat.T) % 2
    expected_obser_correct = np.all(obser_pred == observables, axis=1)

    np.testing.assert_array_equal(result.synd_match_mask, conv_mask)
    np.testing.assert_array_equal(result.decoding_iters, iters)
    np.testing.assert_array_equal(result.obser_correct_mask, expected_obser_correct)


def test_noniterative_returns_none_iters(
    rep_code_circuit, rep_code_syndromes, mwpm_decoder
):
    syndromes, observables = rep_code_syndromes
    wrapper = _BenchmarkDecoder(mwpm_decoder, rep_code_circuit.obsmat)
    result = wrapper.decode(syndromes, observables)
    assert result.decoding_iters is None
    assert result.obser_correct_mask.shape == (syndromes.shape[0],)
    assert result.synd_match_mask.shape == (syndromes.shape[0],)


def test_noniterative_mask_logic_matches_independent_recomputation(
    rep_code_circuit, rep_code_syndromes, mwpm_decoder
):
    syndromes, observables = rep_code_syndromes
    obsmat = rep_code_circuit.obsmat
    pcm = rep_code_circuit.chkmat
    wrapper = _BenchmarkDecoder(mwpm_decoder, obsmat)
    result = wrapper.decode(syndromes, observables)

    ehat = mwpm_decoder.decode_batch(syndromes)
    expected_obser = np.all((ehat @ obsmat.T) % 2 == observables, axis=1)
    expected_synd = np.all((ehat @ pcm.T) % 2 == syndromes, axis=1)
    np.testing.assert_array_equal(result.obser_correct_mask, expected_obser)
    np.testing.assert_array_equal(result.synd_match_mask, expected_synd)
