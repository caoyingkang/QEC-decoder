"""Basic correctness tests for BPDecoder (min-sum BP)."""

import numpy as np

from qecdec.decoders import ALL_DECODERS, ITERATIVE_DECODERS, BPDecoder, create_decoder


def test_factory_registration(rsc_d5_data):
    assert "BP" in ALL_DECODERS
    assert "BP" in ITERATIVE_DECODERS
    dec = create_decoder(
        "BP", pcm=rsc_d5_data["pcm"], prior=rsc_d5_data["prior"], max_iter=20
    )
    assert isinstance(dec, BPDecoder)


def test_output_shape_and_dtype(rsc_d5_data):
    dec = BPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], max_iter=20, norm=0.9)
    syndromes = rsc_d5_data["syndromes"]
    batch_size = syndromes.shape[0]
    num_vars = rsc_d5_data["num_vars"]

    ehat = dec.decode(syndromes[0])
    assert ehat.shape == (num_vars,)
    assert ehat.dtype == np.uint8

    ehat_batch = dec.decode_batch(syndromes)
    assert ehat_batch.shape == (batch_size, num_vars)
    assert ehat_batch.dtype == np.uint8

    ehat2, conv, n_iter, llr_hist = dec.decode_detailed(syndromes[0])
    assert ehat2.shape == (num_vars,)
    assert isinstance(conv, bool)
    assert isinstance(n_iter, int)
    assert llr_hist is None

    ehat_b, conv_mask, iters = dec.decode_batch_detailed(syndromes)
    assert ehat_b.shape == ehat_batch.shape
    assert conv_mask.shape == (batch_size,)
    assert conv_mask.dtype == np.bool_
    assert iters.shape == (batch_size,)
    assert iters.dtype == np.int64


def test_all_zero_syndrome_fast_path(rsc_d5_data):
    dec = BPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], max_iter=20)
    zero = np.zeros(rsc_d5_data["num_chks"], dtype=np.uint8)
    ehat, conv, n_iter, _ = dec.decode_detailed(zero)
    np.testing.assert_array_equal(
        ehat, np.zeros(rsc_d5_data["num_vars"], dtype=np.uint8)
    )
    assert conv is True
    assert n_iter == 0


def test_syndrome_satisfaction(rsc_d5_data):
    dec = BPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], max_iter=50, norm=0.9)
    syndromes = rsc_d5_data["syndromes"]
    batch_size = syndromes.shape[0]
    pcm = rsc_d5_data["pcm"]
    ehat_batch, conv_mask, _ = dec.decode_batch_detailed(syndromes)
    for i in range(batch_size):
        if conv_mask[i]:
            lhs = (pcm.astype(np.int64) @ ehat_batch[i].astype(np.int64)) % 2
            np.testing.assert_array_equal(
                lhs.astype(np.uint8),
                syndromes[i],
                err_msg=f"Converged ehat at index {i} doesn't satisfy syndrome",
            )


def test_repeated_decode_is_deterministic(rsc_d5_data):
    """Calling decode_detailed twice on the same decoder must give identical
    results — verifies internal message buffers don't leak state across calls."""
    dec = BPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], max_iter=20, norm=0.9)
    syndromes = rsc_d5_data["syndromes"]
    batch_size = syndromes.shape[0]
    for i in range(batch_size):
        s = syndromes[i]
        ehat_a, conv_a, n_a, _ = dec.decode_detailed(s)
        ehat_b, conv_b, n_b, _ = dec.decode_detailed(s)
        np.testing.assert_array_equal(
            ehat_a, ehat_b, err_msg=f"Nondeterministic decode at shot {i}"
        )
        assert conv_a == conv_b
        assert n_a == n_b


def test_llr_history_shape(rsc_d5_data):
    dec = BPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], max_iter=20, norm=0.9)
    syndromes = rsc_d5_data["syndromes"]
    # Use a nonzero syndrome so llr_hist is recorded (the all-zeros fast path
    # returns an empty history).
    idx = next(i for i in range(syndromes.shape[0]) if syndromes[i].any())
    s = syndromes[idx]
    ehat, conv, n_iter, llr_hist = dec.decode_detailed(s, record_llr_history=True)
    assert llr_hist is not None
    assert llr_hist.shape == (n_iter, rsc_d5_data["num_vars"])
    assert llr_hist.dtype == np.float64


def test_decode_matches_decode_batch_single_row(rsc_d5_data):
    dec = BPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], max_iter=20, norm=0.9)
    s = rsc_d5_data["syndromes"][0]
    ehat_single = dec.decode(s)
    ehat_batch = dec.decode_batch(s[np.newaxis, :])
    np.testing.assert_array_equal(ehat_single, ehat_batch[0])


def test_parallel_matches_sequential(rsc_d5_data):
    dec = BPDecoder(rsc_d5_data["pcm"], rsc_d5_data["prior"], max_iter=20, norm=0.9)
    syndromes = rsc_d5_data["syndromes"]
    ehat_seq, conv_seq, iters_seq = dec.decode_batch_detailed(syndromes, parallel=False)
    ehat_par, conv_par, iters_par = dec.decode_batch_detailed(syndromes, parallel=True)
    np.testing.assert_array_equal(ehat_seq, ehat_par)
    np.testing.assert_array_equal(conv_seq, conv_par)
    np.testing.assert_array_equal(iters_seq, iters_par)
