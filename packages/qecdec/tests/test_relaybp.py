"""Tests for the RelayBPDecoder."""

import numpy as np
import pytest

from qecdec.decoders import (
    DECODERS_REGISTRY,
    ITERATIVE_DECODERS_REGISTRY,
    DMemBPDecoder,
    RelayBPDecoder,
    create_decoder,
)


def _kwargs(num_vars: int, **overrides):
    defaults = dict(
        gamma0=np.full(num_vars, 0.125, dtype=np.float64),
        gamma_dist_interval=(-0.25, 0.95),
        num_relays=4,
        pre_iter=20,
        max_iter_per_relay=20,
        stop_nconv=2,
    )
    defaults.update(overrides)
    return defaults


def test_factory_registration(rsc_d5_data):
    assert "RelayBP" in DECODERS_REGISTRY
    assert "RelayBP" in ITERATIVE_DECODERS_REGISTRY
    dec = create_decoder(
        "RelayBP",
        rsc_d5_data["pcm"],
        rsc_d5_data["prior"],
        **_kwargs(rsc_d5_data["num_vars"]),
    )
    assert isinstance(dec, RelayBPDecoder)


def test_constructor_validation(rsc_d5_data):
    num_vars = rsc_d5_data["num_vars"]
    # stop_nconv must satisfy 1 <= stop_nconv <= num_relays + 1.
    with pytest.raises(ValueError):
        RelayBPDecoder(
            rsc_d5_data["pcm"],
            rsc_d5_data["prior"],
            **_kwargs(num_vars, num_relays=2, stop_nconv=0),
        )
    with pytest.raises(ValueError):
        RelayBPDecoder(
            rsc_d5_data["pcm"],
            rsc_d5_data["prior"],
            **_kwargs(num_vars, num_relays=2, stop_nconv=4),
        )
    # gamma_dist_interval must have low <= high.
    with pytest.raises(ValueError):
        RelayBPDecoder(
            rsc_d5_data["pcm"],
            rsc_d5_data["prior"],
            **_kwargs(num_vars, gamma_dist_interval=(0.9, -0.1)),
        )


def test_output_shape_and_dtype(rsc_d5_data):
    dec = RelayBPDecoder(
        rsc_d5_data["pcm"], rsc_d5_data["prior"], **_kwargs(rsc_d5_data["num_vars"])
    )
    syndromes = rsc_d5_data["syndromes"]
    batch_size = syndromes.shape[0]
    num_vars = rsc_d5_data["num_vars"]

    ehat = dec.decode(syndromes[0], seed=0)
    assert ehat.shape == (num_vars,)
    assert ehat.dtype == np.uint8

    ehat_batch = dec.decode_batch(syndromes, seed=0)
    assert ehat_batch.shape == (batch_size, num_vars)
    assert ehat_batch.dtype == np.uint8

    ehat2, conv, n_iter = dec.decode_detailed(syndromes[0], seed=0)
    assert ehat2.shape == (num_vars,)
    assert isinstance(conv, bool)
    assert isinstance(n_iter, int)

    ehat_b, conv_mask, iters = dec.decode_batch_detailed(syndromes, seed=0)
    assert ehat_b.shape == ehat_batch.shape
    assert conv_mask.shape == (batch_size,)
    assert conv_mask.dtype == np.bool_
    assert iters.shape == (batch_size,)
    assert iters.dtype == np.int64


def test_all_zero_syndrome_fast_path(rsc_d5_data):
    dec = RelayBPDecoder(
        rsc_d5_data["pcm"], rsc_d5_data["prior"], **_kwargs(rsc_d5_data["num_vars"])
    )
    zero = np.zeros(rsc_d5_data["num_chks"], dtype=np.uint8)
    ehat, conv, n_iter = dec.decode_detailed(zero, seed=0)
    np.testing.assert_array_equal(
        ehat, np.zeros(rsc_d5_data["num_vars"], dtype=np.uint8)
    )
    assert conv is True
    assert n_iter == 0


def test_syndrome_satisfaction(rsc_d5_data):
    dec = RelayBPDecoder(
        rsc_d5_data["pcm"], rsc_d5_data["prior"], **_kwargs(rsc_d5_data["num_vars"])
    )
    syndromes = rsc_d5_data["syndromes"]
    batch_size = syndromes.shape[0]
    pcm = rsc_d5_data["pcm"]
    ehat_batch, conv_mask, _ = dec.decode_batch_detailed(syndromes, seed=0)
    for i in range(batch_size):
        if conv_mask[i]:
            lhs = (pcm.astype(np.int64) @ ehat_batch[i].astype(np.int64)) % 2
            np.testing.assert_array_equal(
                lhs.astype(np.uint8),
                syndromes[i],
                err_msg=f"Converged ehat at index {i} doesn't satisfy syndrome",
            )


def test_repeated_decode_determinism_seeded(rsc_d5_data):
    """Same seed twice on the same decoder → identical results. Also rules out
    scratch-buffer state leaking across calls."""
    dec = RelayBPDecoder(
        rsc_d5_data["pcm"], rsc_d5_data["prior"], **_kwargs(rsc_d5_data["num_vars"])
    )
    syndromes = rsc_d5_data["syndromes"]
    batch_size = syndromes.shape[0]
    for i in range(batch_size):
        s = syndromes[i]
        a = dec.decode_detailed(s, seed=12345)
        b = dec.decode_detailed(s, seed=12345)
        np.testing.assert_array_equal(a[0], b[0], err_msg=f"shot {i} differs")
        assert a[1] == b[1]
        assert a[2] == b[2]


def test_seed_reproducibility_across_instances(rsc_d5_data):
    """Two fresh decoder instances with the same seed produce identical batches."""
    kwargs = _kwargs(rsc_d5_data["num_vars"])
    dec_a = RelayBPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], **kwargs)
    dec_b = RelayBPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], **kwargs)
    syndromes = rsc_d5_data["syndromes"]
    ehat_a, conv_a, iters_a = dec_a.decode_batch_detailed(syndromes, seed=42)
    ehat_b, conv_b, iters_b = dec_b.decode_batch_detailed(syndromes, seed=42)
    np.testing.assert_array_equal(ehat_a, ehat_b)
    np.testing.assert_array_equal(conv_a, conv_b)
    np.testing.assert_array_equal(iters_a, iters_b)


def test_parallel_matches_sequential_with_seed(rsc_d5_data):
    """Seed-driven child-seed derivation is the same in both branches."""
    dec = RelayBPDecoder(
        rsc_d5_data["pcm"], rsc_d5_data["prior"], **_kwargs(rsc_d5_data["num_vars"])
    )
    syndromes = rsc_d5_data["syndromes"]
    ehat_seq, conv_seq, iters_seq = dec.decode_batch_detailed(
        syndromes, parallel=False, seed=7
    )
    ehat_par, conv_par, iters_par = dec.decode_batch_detailed(
        syndromes, parallel=True, seed=7
    )
    np.testing.assert_array_equal(ehat_seq, ehat_par)
    np.testing.assert_array_equal(conv_seq, conv_par)
    np.testing.assert_array_equal(iters_seq, iters_par)


def test_stage0_only_reduces_to_dmembp(rsc_d5_data):
    """With num_relays=0, stop_nconv=1, RelayBP runs only its first DMemBP stage
    (with norm hardcoded to 1.0 in Rust). It should match DMemBPDecoder exactly."""
    num_vars = rsc_d5_data["num_vars"]
    gamma0 = np.full(num_vars, 0.125, dtype=np.float64)
    pre_iter = 30
    relay = RelayBPDecoder(
        rsc_d5_data["pcm"],
        rsc_d5_data["prior"],
        gamma0=gamma0,
        gamma_dist_interval=(-0.1, 0.5),
        num_relays=0,
        pre_iter=pre_iter,
        max_iter_per_relay=1,
        stop_nconv=1,
    )
    dmem = DMemBPDecoder(
        rsc_d5_data["pcm"],
        rsc_d5_data["prior"],
        gamma=gamma0,
        max_iter=pre_iter,
        norm=1.0,
    )
    syndromes = rsc_d5_data["syndromes"]
    ehat_relay, conv_relay, iters_relay = relay.decode_batch_detailed(syndromes, seed=0)
    ehat_dmem, conv_dmem, iters_dmem = dmem.decode_batch_detailed(syndromes)
    np.testing.assert_array_equal(ehat_relay, ehat_dmem)
    np.testing.assert_array_equal(conv_relay, conv_dmem)
    np.testing.assert_array_equal(iters_relay, iters_dmem)
