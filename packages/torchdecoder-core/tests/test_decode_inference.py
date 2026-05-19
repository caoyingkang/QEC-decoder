import numpy as np
import pytest
import torch

from qecdec.experiments import RepetitionCode_Memory
from torchdecoder_core.models import (
    DecoderModel,
    InferenceResult,
    LearnedDMemBP,
    MultiDMemBP,
)
from torchdecoder_core.utils.decoding_utils import diagnose_convergence, gather_ehat


NUM_ITERS = 10
BATCH_SIZE = 256
SEED = 42


@pytest.fixture(scope="module")
def pcm_prior() -> tuple[np.ndarray, np.ndarray]:
    expmt = RepetitionCode_Memory(
        d=5,
        rounds=5,
        data_qubit_error_rate=0.01,
        meas_error_rate=0.01,
    )
    return expmt.chkmat, expmt.prior


@pytest.fixture(scope="module")
def chkmat(pcm_prior) -> torch.Tensor:
    pcm, _ = pcm_prior
    return torch.tensor(pcm, dtype=torch.float32)


@pytest.fixture(scope="module")
def syndromes(pcm_prior) -> torch.Tensor:
    pcm, _ = pcm_prior
    torch.manual_seed(SEED)
    return torch.randint(0, 2, (BATCH_SIZE, pcm.shape[0]), dtype=torch.int32)


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


def _assert_inference_matches_reference(
    model: DecoderModel,
    syndromes: torch.Tensor,
    chkmat: torch.Tensor,
) -> None:
    model.eval()
    with torch.inference_mode():
        result_ref = _decode_ref(model, syndromes, chkmat)
        result = model.decode_inference(syndromes, chkmat)

    torch.testing.assert_close(result.ehat, result_ref.ehat)
    assert torch.equal(result.converged_mask, result_ref.converged_mask)
    assert torch.equal(result.decoding_iters, result_ref.decoding_iters)


def test_learned_dmembp_matches_reference(pcm_prior, chkmat, syndromes) -> None:
    pcm, prior = pcm_prior
    model = LearnedDMemBP(
        pcm,
        prior,
        NUM_ITERS,
        min_impl_method="hard",
        sign_impl_method="hard",
    )
    _assert_inference_matches_reference(model, syndromes, chkmat)


def test_multi_dmembp_matches_reference(pcm_prior, chkmat, syndromes) -> None:
    pcm, prior = pcm_prior
    model = MultiDMemBP(
        pcm,
        prior,
        NUM_ITERS,
        msg_features=16,
        mlp_hidden_features=64,
        mlp_hidden_depth=2,
        mlp_activation="Tanh",
        mlp_norm=None,
        mlp_dropout_p=None,
        gamma_shared=False,
        gamma_init=[-0.1, 0.8],
        min_impl_method="hard",
        sign_impl_method="hard",
    )
    _assert_inference_matches_reference(model, syndromes, chkmat)
