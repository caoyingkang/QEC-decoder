"""Run from repo root:

uv run python -m unittest discover -s packages/torchdecoder-core/tests -p "test_*.py" -v
"""

import unittest

import numpy as np
import torch

from qecdec.experiments import RepetitionCode_Memory
from torchdecoder_core.models import LearnedDMemBP, MultiDMemBP
from torchdecoder_core.models.base import DecoderModel, InferenceResult
from torchdecoder_core.utils.decoding_utils import diagnose_convergence, gather_ehat


def _pcm_and_prior() -> tuple[np.ndarray, np.ndarray]:
    expmt = RepetitionCode_Memory(
        d=5,
        rounds=5,
        data_qubit_error_rate=0.01,
        meas_error_rate=0.01,
    )
    return expmt.chkmat, expmt.prior


def _decode_ref(
    model: DecoderModel,
    syndromes: torch.Tensor,
    chkmat: torch.Tensor,
) -> InferenceResult:
    llrs = model(syndromes)
    hard_decisions = (llrs < 0).float()
    converged_mask, output_iters = diagnose_convergence(
        hard_decisions, syndromes, chkmat
    )
    ehat = gather_ehat(hard_decisions, output_iters)
    decoding_iters = output_iters + 1

    trivial_mask = torch.all(syndromes == 0, dim=1)  # (B,), bool
    ehat[trivial_mask] = 0
    converged_mask |= trivial_mask
    decoding_iters[trivial_mask] = 0

    return InferenceResult(ehat, converged_mask, decoding_iters)


class TestDecodeInferenceLearnedDMemBP(unittest.TestCase):
    def test_matches_reference(self) -> None:
        pcm, prior = _pcm_and_prior()
        num_iters = 10
        model = LearnedDMemBP(
            pcm,
            prior,
            num_iters,
            min_impl_method="hard",
            sign_impl_method="hard",
        )
        model.eval()
        chkmat = torch.tensor(pcm, dtype=torch.float32)
        torch.manual_seed(42)
        syndromes = torch.randint(0, 2, (1024, pcm.shape[0]), dtype=torch.int32)

        with torch.inference_mode():
            result_ref = _decode_ref(model, syndromes, chkmat)
            result = model.decode_inference(syndromes, chkmat)

        torch.testing.assert_close(result.ehat, result_ref.ehat)
        self.assertTrue(torch.equal(result.converged_mask, result_ref.converged_mask))
        self.assertTrue(torch.equal(result.decoding_iters, result_ref.decoding_iters))


class TestDecodeInferenceMultiDMemBP(unittest.TestCase):
    def test_matches_reference(self) -> None:
        pcm, prior = _pcm_and_prior()
        num_iters = 10
        model = MultiDMemBP(
            pcm,
            prior,
            num_iters,
            msg_features=16,
            mlp_hidden_features=64,
            mlp_hidden_depth=2,
            mlp_activation="Tanh",
            mlp_norm=None,
            mlp_dropout_p=None,
            min_impl_method="hard",
            sign_impl_method="hard",
            gamma_shared=False,
            gamma_init=[-0.1, 0.8],
        )
        model.eval()
        chkmat = torch.tensor(pcm, dtype=torch.float32)
        torch.manual_seed(42)
        syndromes = torch.randint(0, 2, (1024, pcm.shape[0]), dtype=torch.int32)

        with torch.inference_mode():
            result_ref = _decode_ref(model, syndromes, chkmat)
            result = model.decode_inference(syndromes, chkmat)

        torch.testing.assert_close(result.ehat, result_ref.ehat)
        self.assertTrue(torch.equal(result.converged_mask, result_ref.converged_mask))
        self.assertTrue(torch.equal(result.decoding_iters, result_ref.decoding_iters))
