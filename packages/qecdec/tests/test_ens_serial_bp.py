"""Tests for the EnsSerialBPDecoder ensemble decoder."""

import pickle

import numpy as np
import pytest

from qecdec.decoders import (
    ALL_DECODERS,
    ITERATIVE_DECODERS,
    EnsSerialBPDecoder,
    SerialBPDecoder,
    create_decoder,
)


def test_factory_registration(rsc_d5_data):
    assert "EnsSerialBP" in ALL_DECODERS
    assert "EnsSerialBP" in ITERATIVE_DECODERS
    dec = create_decoder(
        "EnsSerialBP",
        pcm=rsc_d5_data["pcm"],
        prior=rsc_d5_data["prior"],
        max_iter=20,
        ensemble_size=4,
        topk=2,
        seed=0,
    )
    assert isinstance(dec, EnsSerialBPDecoder)


def test_constructor_validation(rsc_d5_data):
    pcm, prior = rsc_d5_data["pcm"], rsc_d5_data["prior"]
    with pytest.raises(ValueError):
        EnsSerialBPDecoder(pcm, prior, max_iter=10, ensemble_size=4, topk=0)
    with pytest.raises(ValueError):
        EnsSerialBPDecoder(pcm, prior, max_iter=10, ensemble_size=4, topk=5)
    with pytest.raises(ValueError):
        EnsSerialBPDecoder(pcm, prior, max_iter=10, ensemble_size=0, topk=1)


def test_output_shapes(rsc_d5_data):
    syndromes = rsc_d5_data["syndromes"]
    batch_size = syndromes.shape[0]
    num_vars = rsc_d5_data["num_vars"]
    dec = EnsSerialBPDecoder(
        rsc_d5_data["pcm"],
        rsc_d5_data["prior"],
        max_iter=20,
        ensemble_size=4,
        topk=2,
        seed=0,
    )
    ehat = dec.decode(syndromes[0])
    assert ehat.shape == (num_vars,)
    assert ehat.dtype == np.uint8

    ehat_batch = dec.decode_batch(syndromes)
    assert ehat_batch.shape == (batch_size, num_vars)
    assert ehat_batch.dtype == np.uint8

    ehat2, conv, n_iter = dec.decode_detailed(syndromes[0])
    assert ehat2.shape == (num_vars,)
    assert isinstance(conv, bool)
    assert isinstance(n_iter, int)

    ehat_b, conv_mask, iters = dec.decode_batch_detailed(syndromes)
    assert ehat_b.shape == ehat_batch.shape
    assert conv_mask.shape == (batch_size,)
    assert conv_mask.dtype == np.bool_
    assert iters.shape == (batch_size,)
    assert iters.dtype == np.int64


def test_syndrome_satisfaction(rsc_d5_data):
    """For any returned ehat with converged=True, pcm @ ehat == syndrome (mod 2)."""
    syndromes = rsc_d5_data["syndromes"]
    batch_size = syndromes.shape[0]
    pcm = rsc_d5_data["pcm"]
    dec = EnsSerialBPDecoder(
        pcm,
        rsc_d5_data["prior"],
        max_iter=30,
        ensemble_size=8,
        topk=4,
        seed=0,
    )
    ehat_batch, conv_mask, _ = dec.decode_batch_detailed(syndromes)
    for i in range(batch_size):
        if conv_mask[i]:
            lhs = (pcm.astype(np.int64) @ ehat_batch[i].astype(np.int64)) % 2
            np.testing.assert_array_equal(
                lhs.astype(np.uint8),
                syndromes[i],
                err_msg=f"Converged ehat at index {i} doesn't satisfy syndrome",
            )


def test_topk_monotone_iterations(rsc_d5_data):
    """Average num_iter should be monotone non-decreasing in topk."""
    syndromes = rsc_d5_data["syndromes"]
    kwargs = dict(
        pcm=rsc_d5_data["pcm"],
        prior=rsc_d5_data["prior"],
        max_iter=50,
        ensemble_size=8,
        seed=42,
    )
    dec_top1 = EnsSerialBPDecoder(topk=1, **kwargs)
    dec_top4 = EnsSerialBPDecoder(topk=4, **kwargs)
    dec_top8 = EnsSerialBPDecoder(topk=8, **kwargs)

    _, _, iters_top1 = dec_top1.decode_batch_detailed(syndromes)
    _, _, iters_top4 = dec_top4.decode_batch_detailed(syndromes)
    _, _, iters_top8 = dec_top8.decode_batch_detailed(syndromes)

    assert float(iters_top1.mean()) <= float(iters_top4.mean())
    assert float(iters_top4.mean()) <= float(iters_top8.mean())


def test_determinism(rsc_d5_data):
    """Same seed -> identical vn_orders and identical decode outputs."""
    syndromes = rsc_d5_data["syndromes"]
    kwargs = dict(
        pcm=rsc_d5_data["pcm"],
        prior=rsc_d5_data["prior"],
        max_iter=20,
        ensemble_size=4,
        topk=2,
        seed=123,
    )
    dec_a = EnsSerialBPDecoder(**kwargs)
    dec_b = EnsSerialBPDecoder(**kwargs)
    np.testing.assert_array_equal(dec_a.vn_orders, dec_b.vn_orders)

    ehat_a, conv_a, iters_a = dec_a.decode_batch_detailed(syndromes)
    ehat_b, conv_b, iters_b = dec_b.decode_batch_detailed(syndromes)
    np.testing.assert_array_equal(ehat_a, ehat_b)
    np.testing.assert_array_equal(conv_a, conv_b)
    np.testing.assert_array_equal(iters_a, iters_b)


def test_ensemble_size_one_matches_single(rsc_d5_data):
    """ensemble_size=1 with the natural-order vn_order should match
    SerialBPDecoder's default behavior."""
    syndromes = rsc_d5_data["syndromes"]
    num_vars = rsc_d5_data["num_vars"]
    single = SerialBPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], max_iter=20)
    ens = EnsSerialBPDecoder(
        rsc_d5_data["pcm"],
        rsc_d5_data["prior"],
        max_iter=20,
        ensemble_size=1,
        topk=1,
        seed=0,
    )
    ens.vn_orders = np.arange(num_vars, dtype=np.int64)[np.newaxis, :]
    ens._decoder = ens._build_decoder()

    ehat_s = single.decode_batch(syndromes)
    ehat_e = ens.decode_batch(syndromes)
    np.testing.assert_array_equal(ehat_s, ehat_e)


def test_pickle_round_trip(rsc_d5_data):
    syndromes = rsc_d5_data["syndromes"]
    dec = EnsSerialBPDecoder(
        rsc_d5_data["pcm"],
        rsc_d5_data["prior"],
        max_iter=20,
        ensemble_size=4,
        topk=2,
        seed=0,
    )
    blob = pickle.dumps(dec)
    dec2 = pickle.loads(blob)
    assert dec2.ensemble_size == dec.ensemble_size
    assert dec2.topk == dec.topk
    np.testing.assert_array_equal(dec2.vn_orders, dec.vn_orders)

    ehat_a = dec.decode_batch(syndromes)
    ehat_b = dec2.decode_batch(syndromes)
    np.testing.assert_array_equal(ehat_a, ehat_b)
