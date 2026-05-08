"""Tests for the EnsSerialBPDecoder ensemble decoder."""

import pickle
import unittest

import numpy as np

from qecdec.decoders import (
    ALL_DECODERS,
    ITERATIVE_DECODERS,
    EnsSerialBPDecoder,
    SerialBPDecoder,
    create_decoder,
)
from qecdec.experiments import RotatedSurfaceCode_Memory


class TestEnsSerialBPDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expmt = RotatedSurfaceCode_Memory(
            d=5,
            rounds=5,
            basis="Z",
            data_qubit_error_rate=0.05,
            meas_error_rate=0.05,
        )
        sampler = cls.expmt.dem.compile_sampler()
        syndromes, _, _ = sampler.sample(64)
        cls.syndromes = syndromes.astype(np.uint8)
        cls.pcm = cls.expmt.chkmat
        cls.prior = cls.expmt.prior
        cls.num_vars = cls.expmt.num_error_mechanisms

    def test_factory_registration(self):
        self.assertIn("EnsSerialBP", ALL_DECODERS)
        self.assertIn("EnsSerialBP", ITERATIVE_DECODERS)
        dec = create_decoder(
            "EnsSerialBP",
            pcm=self.pcm,
            prior=self.prior,
            max_iter=20,
            ensemble_size=4,
            topk=2,
            seed=0,
        )
        self.assertIsInstance(dec, EnsSerialBPDecoder)

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            EnsSerialBPDecoder(
                self.pcm, self.prior, max_iter=10, ensemble_size=4, topk=0
            )
        with self.assertRaises(ValueError):
            EnsSerialBPDecoder(
                self.pcm, self.prior, max_iter=10, ensemble_size=4, topk=5
            )
        with self.assertRaises(ValueError):
            EnsSerialBPDecoder(
                self.pcm, self.prior, max_iter=10, ensemble_size=0, topk=1
            )

    def test_output_shapes(self):
        dec = EnsSerialBPDecoder(
            self.pcm, self.prior, max_iter=20, ensemble_size=4, topk=2, seed=0
        )
        ehat = dec.decode(self.syndromes[0])
        self.assertEqual(ehat.shape, (self.num_vars,))
        self.assertEqual(ehat.dtype, np.uint8)

        ehat_batch = dec.decode_batch(self.syndromes)
        self.assertEqual(
            ehat_batch.shape, (self.syndromes.shape[0], self.num_vars)
        )
        self.assertEqual(ehat_batch.dtype, np.uint8)

        ehat2, conv, n_iter, llr_hist = dec.decode_detailed(self.syndromes[0])
        self.assertEqual(ehat2.shape, (self.num_vars,))
        self.assertIsInstance(conv, bool)
        self.assertIsInstance(n_iter, int)
        self.assertIsNone(llr_hist)

        ehat_b, conv_mask, iters = dec.decode_batch_detailed(self.syndromes)
        self.assertEqual(ehat_b.shape, ehat_batch.shape)
        self.assertEqual(conv_mask.shape, (self.syndromes.shape[0],))
        self.assertEqual(conv_mask.dtype, np.bool_)
        self.assertEqual(iters.shape, (self.syndromes.shape[0],))
        self.assertEqual(iters.dtype, np.int64)

    def test_record_llr_history_unsupported(self):
        dec = EnsSerialBPDecoder(
            self.pcm, self.prior, max_iter=10, ensemble_size=4, topk=2, seed=0
        )
        with self.assertRaises(NotImplementedError):
            dec.decode_detailed(self.syndromes[0], record_llr_history=True)

    def test_syndrome_satisfaction(self):
        """For any returned ehat with converged=True, pcm @ ehat == syndrome (mod 2)."""
        dec = EnsSerialBPDecoder(
            self.pcm, self.prior, max_iter=30, ensemble_size=8, topk=4, seed=0
        )
        ehat_batch, conv_mask, _ = dec.decode_batch_detailed(self.syndromes)
        for i in range(self.syndromes.shape[0]):
            if conv_mask[i]:
                lhs = (self.pcm.astype(np.int64) @ ehat_batch[i].astype(np.int64)) % 2
                np.testing.assert_array_equal(
                    lhs.astype(np.uint8),
                    self.syndromes[i],
                    err_msg=f"Converged ehat at index {i} doesn't satisfy syndrome",
                )

    def test_convergence_dominance(self):
        """With topk=ensemble_size, the ensemble's converged count should be >=
        the single SerialBPDecoder's converged count."""
        max_iter = 20
        single = SerialBPDecoder(self.pcm, self.prior, max_iter=max_iter)
        _, conv_single, _ = single.decode_batch_detailed(self.syndromes)

        ens = EnsSerialBPDecoder(
            self.pcm,
            self.prior,
            max_iter=max_iter,
            ensemble_size=8,
            topk=8,
            seed=0,
        )
        _, conv_ens, _ = ens.decode_batch_detailed(self.syndromes)

        self.assertGreaterEqual(int(conv_ens.sum()), int(conv_single.sum()))

    def test_topk_monotone_iterations(self):
        """Average num_iter should be monotone non-decreasing in topk."""
        max_iter = 50
        kwargs = dict(
            pcm=self.pcm,
            prior=self.prior,
            max_iter=max_iter,
            ensemble_size=8,
            seed=42,
        )
        dec_top1 = EnsSerialBPDecoder(topk=1, **kwargs)
        dec_top4 = EnsSerialBPDecoder(topk=4, **kwargs)
        dec_top8 = EnsSerialBPDecoder(topk=8, **kwargs)

        _, _, iters_top1 = dec_top1.decode_batch_detailed(self.syndromes)
        _, _, iters_top4 = dec_top4.decode_batch_detailed(self.syndromes)
        _, _, iters_top8 = dec_top8.decode_batch_detailed(self.syndromes)

        self.assertLessEqual(float(iters_top1.mean()), float(iters_top4.mean()))
        self.assertLessEqual(float(iters_top4.mean()), float(iters_top8.mean()))

    def test_determinism(self):
        """Same seed -> identical vn_orders and identical decode outputs."""
        kwargs = dict(
            pcm=self.pcm,
            prior=self.prior,
            max_iter=20,
            ensemble_size=4,
            topk=2,
            seed=123,
        )
        dec_a = EnsSerialBPDecoder(**kwargs)
        dec_b = EnsSerialBPDecoder(**kwargs)
        np.testing.assert_array_equal(dec_a.vn_orders, dec_b.vn_orders)

        ehat_a, conv_a, iters_a = dec_a.decode_batch_detailed(self.syndromes)
        ehat_b, conv_b, iters_b = dec_b.decode_batch_detailed(self.syndromes)
        np.testing.assert_array_equal(ehat_a, ehat_b)
        np.testing.assert_array_equal(conv_a, conv_b)
        np.testing.assert_array_equal(iters_a, iters_b)

    def test_ensemble_size_one_matches_single(self):
        """ensemble_size=1 with the natural-order vn_order should match
        SerialBPDecoder's default behavior."""
        # SerialBPDecoder defaults to vn_order = np.arange(num_vars).
        # EnsSerialBPDecoder generates random permutations, so we override.
        single = SerialBPDecoder(self.pcm, self.prior, max_iter=20)
        ens = EnsSerialBPDecoder(
            self.pcm, self.prior, max_iter=20, ensemble_size=1, topk=1, seed=0
        )
        ens.vn_orders = np.arange(self.num_vars, dtype=np.int64)[np.newaxis, :]
        ens._decoder = ens._build_decoder()

        ehat_s = single.decode_batch(self.syndromes)
        ehat_e = ens.decode_batch(self.syndromes)
        np.testing.assert_array_equal(ehat_s, ehat_e)

    def test_pickle_round_trip(self):
        dec = EnsSerialBPDecoder(
            self.pcm, self.prior, max_iter=20, ensemble_size=4, topk=2, seed=0
        )
        blob = pickle.dumps(dec)
        dec2 = pickle.loads(blob)
        self.assertEqual(dec2.ensemble_size, dec.ensemble_size)
        self.assertEqual(dec2.topk, dec.topk)
        self.assertEqual(dec2.seed, dec.seed)
        np.testing.assert_array_equal(dec2.vn_orders, dec.vn_orders)

        ehat_a = dec.decode_batch(self.syndromes)
        ehat_b = dec2.decode_batch(self.syndromes)
        np.testing.assert_array_equal(ehat_a, ehat_b)


if __name__ == "__main__":
    unittest.main()
