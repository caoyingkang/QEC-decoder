"""Test that parallel=True produces identical results to parallel=False for all BP decoders."""

import unittest

import numpy as np
from qecdec.experiments import RotatedSurfaceCode_Memory
from qecdec.decoders import (
    BPDecoder,
    MemBPDecoder,
    DMemBPDecoder,
    DMemOffsetBPDecoder,
    SerialBPDecoder,
)


class TestParallelDecode(unittest.TestCase):
    """Compare sequential vs parallel batch decoding for correctness."""

    @classmethod
    def setUpClass(cls):
        """Set up a QEC experiment and sample syndromes."""
        cls.expmt = RotatedSurfaceCode_Memory(
            d=9,
            rounds=9,
            basis="Z",
            data_qubit_error_rate=0.05,
            meas_error_rate=0.05,
        )
        # Sample a batch of syndromes
        sampler = cls.expmt.dem.compile_sampler()
        syndromes, _, _ = sampler.sample(256)
        cls.syndromes = syndromes.astype(np.uint8)
        cls.pcm = cls.expmt.chkmat
        cls.prior = cls.expmt.prior
        # Randomly generate gamma for DMemBP and DMemOffsetBP
        rng = np.random.default_rng(42)
        cls.gamma_array = rng.uniform(-0.25, 0.95, cls.expmt.num_error_mechanisms)

    def test_bp_decode_batch(self):
        dec = BPDecoder(self.pcm, self.prior, max_iter=50, norm=0.9)
        result_seq = dec.decode_batch(self.syndromes, parallel=False)
        result_par = dec.decode_batch(self.syndromes, parallel=True)
        np.testing.assert_array_equal(result_seq, result_par)

    def test_bp_decode_batch_detailed(self):
        dec = BPDecoder(self.pcm, self.prior, max_iter=20, norm=0.9)
        ehat_seq, conv_seq, iters_seq = dec.decode_batch_detailed(
            self.syndromes, parallel=False
        )
        ehat_par, conv_par, iters_par = dec.decode_batch_detailed(
            self.syndromes, parallel=True
        )
        np.testing.assert_array_equal(ehat_seq, ehat_par)
        np.testing.assert_array_equal(conv_seq, conv_par)
        np.testing.assert_array_equal(iters_seq, iters_par)

    def test_membp_decode_batch(self):
        dec = MemBPDecoder(self.pcm, self.prior, gamma=0.2, max_iter=50, norm=0.9)
        result_seq = dec.decode_batch(self.syndromes, parallel=False)
        result_par = dec.decode_batch(self.syndromes, parallel=True)
        np.testing.assert_array_equal(result_seq, result_par)

    def test_membp_decode_batch_detailed(self):
        dec = MemBPDecoder(self.pcm, self.prior, gamma=0.2, max_iter=50, norm=0.9)
        ehat_seq, conv_seq, iters_seq = dec.decode_batch_detailed(
            self.syndromes, parallel=False
        )
        ehat_par, conv_par, iters_par = dec.decode_batch_detailed(
            self.syndromes, parallel=True
        )
        np.testing.assert_array_equal(ehat_seq, ehat_par)
        np.testing.assert_array_equal(conv_seq, conv_par)
        np.testing.assert_array_equal(iters_seq, iters_par)

    def test_dmembp_decode_batch(self):
        dec = DMemBPDecoder(
            self.pcm, self.prior, gamma=self.gamma_array, max_iter=50, norm=0.9
        )
        result_seq = dec.decode_batch(self.syndromes, parallel=False)
        result_par = dec.decode_batch(self.syndromes, parallel=True)
        np.testing.assert_array_equal(result_seq, result_par)

    def test_dmembp_decode_batch_detailed(self):
        dec = DMemBPDecoder(
            self.pcm, self.prior, gamma=self.gamma_array, max_iter=50, norm=0.9
        )
        ehat_seq, conv_seq, iters_seq = dec.decode_batch_detailed(
            self.syndromes, parallel=False
        )
        ehat_par, conv_par, iters_par = dec.decode_batch_detailed(
            self.syndromes, parallel=True
        )
        np.testing.assert_array_equal(ehat_seq, ehat_par)
        np.testing.assert_array_equal(conv_seq, conv_par)
        np.testing.assert_array_equal(iters_seq, iters_par)

    def test_dmemoffsetbp_decode_batch(self):
        dec = DMemOffsetBPDecoder(
            self.pcm,
            self.prior,
            gamma=self.gamma_array,
            max_iter=50,
            norm=0.9,
            offset=0.1,
        )
        result_seq = dec.decode_batch(self.syndromes, parallel=False)
        result_par = dec.decode_batch(self.syndromes, parallel=True)
        np.testing.assert_array_equal(result_seq, result_par)

    def test_dmemoffsetbp_decode_batch_detailed(self):
        dec = DMemOffsetBPDecoder(
            self.pcm,
            self.prior,
            gamma=self.gamma_array,
            max_iter=50,
            norm=0.9,
            offset=0.1,
        )
        ehat_seq, conv_seq, iters_seq = dec.decode_batch_detailed(
            self.syndromes, parallel=False
        )
        ehat_par, conv_par, iters_par = dec.decode_batch_detailed(
            self.syndromes, parallel=True
        )
        np.testing.assert_array_equal(ehat_seq, ehat_par)
        np.testing.assert_array_equal(conv_seq, conv_par)
        np.testing.assert_array_equal(iters_seq, iters_par)

    def test_serialbp_decode_batch(self):
        dec = SerialBPDecoder(self.pcm, self.prior, max_iter=50)
        result_seq = dec.decode_batch(self.syndromes, parallel=False)
        result_par = dec.decode_batch(self.syndromes, parallel=True)
        np.testing.assert_array_equal(result_seq, result_par)

    def test_serialbp_decode_batch_detailed(self):
        dec = SerialBPDecoder(self.pcm, self.prior, max_iter=20)
        ehat_seq, conv_seq, iters_seq = dec.decode_batch_detailed(
            self.syndromes, parallel=False
        )
        ehat_par, conv_par, iters_par = dec.decode_batch_detailed(
            self.syndromes, parallel=True
        )
        np.testing.assert_array_equal(ehat_seq, ehat_par)
        np.testing.assert_array_equal(conv_seq, conv_par)
        np.testing.assert_array_equal(iters_seq, iters_par)

    def test_serialbp_custom_vn_order(self):
        rng = np.random.default_rng(0)
        order = rng.permutation(self.expmt.num_error_mechanisms).astype(np.int64)
        dec_default = SerialBPDecoder(self.pcm, self.prior, max_iter=50)
        dec_perm = SerialBPDecoder(
            self.pcm, self.prior, max_iter=50, vn_order=order
        )
        # Both decoders run without error and produce valid-shaped outputs.
        ehat_default = dec_default.decode_batch(self.syndromes, parallel=False)
        ehat_perm = dec_perm.decode_batch(self.syndromes, parallel=False)
        assert ehat_default.shape == ehat_perm.shape == self.syndromes.shape[:1] + (
            self.expmt.num_error_mechanisms,
        )

    def test_serialbp_factory(self):
        from qecdec.decoders import create_decoder, ALL_DECODERS

        assert "SerialBP" in ALL_DECODERS
        dec = create_decoder(
            "SerialBP", pcm=self.pcm, prior=self.prior, max_iter=20
        )
        ehat = dec.decode_batch(self.syndromes, parallel=False)
        assert ehat.shape == self.syndromes.shape[:1] + (
            self.expmt.num_error_mechanisms,
        )


if __name__ == "__main__":
    unittest.main()
