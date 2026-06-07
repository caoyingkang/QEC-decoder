from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from qecdec.decoders import (
    create_decoder,
    DECODERS_REGISTRY,
    ITERATIVE_DECODERS_REGISTRY,
)
from qecdec.circuits import RepetitionCode_Circuit
from torchdecoder_core.models import LearnedDMemBP, MultiDMemBP
from torchdecoder_core.qecdec_adapters import (
    LearnedDMemBPDecoder,
    MultiDMemBPDecoder,
    TorchModelDecoder,
    TORCHMODEL_DECODERS_REGISTRY,
)


NUM_ITERS = 10
BATCH = 256
SEED = 42


def _save_fake_lightning_ckpt(model: nn.Module, path: Path) -> None:
    """Save the model's state_dict in the format expected by
    DecoderModel.load_lightning_checkpoint: keys prefixed with "model."."""
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict}, path)


@pytest.fixture(scope="module")
def pcm_prior() -> tuple[np.ndarray, np.ndarray]:
    circuit = RepetitionCode_Circuit(
        d=5,
        rounds=5,
        data_qubit_error_rate=0.01,
        meas_error_rate=0.01,
    )
    return circuit.chkmat, circuit.prior


@pytest.fixture(scope="module")
def syndromes_np(pcm_prior) -> np.ndarray:
    pcm, _ = pcm_prior
    torch.manual_seed(SEED)
    return (
        torch.randint(0, 2, (BATCH, pcm.shape[0]), dtype=torch.int32)
        .numpy()
        .astype(np.uint8)
    )


@pytest.fixture(scope="module")
def learned_setup(tmp_path_factory, pcm_prior) -> dict:
    pcm, prior = pcm_prior
    torch.manual_seed(SEED)
    model = LearnedDMemBP(
        pcm,
        prior,
        NUM_ITERS,
        min_impl_method="hard",
        sign_impl_method="hard",
    )
    ckpt_path = tmp_path_factory.mktemp("ckpt") / "learned.ckpt"
    _save_fake_lightning_ckpt(model, ckpt_path)
    model_cfg = {
        "name": "LearnedDMemBP",
        "num_iters": NUM_ITERS,
        "min_impl_method": "hard",
        "sign_impl_method": "hard",
    }
    return {
        "model": model,
        "ckpt_path": ckpt_path,
        "model_cfg": model_cfg,
        "adapter_cls": LearnedDMemBPDecoder,
        "registry_name": "TorchModel(LearnedDMemBP)",
        "other_name": "MultiDMemBP",
    }


@pytest.fixture(scope="module")
def multi_setup(tmp_path_factory, pcm_prior) -> dict:
    pcm, prior = pcm_prior
    torch.manual_seed(SEED)
    model = MultiDMemBP(
        pcm,
        prior,
        NUM_ITERS,
        msg_features=16,
        mlp_hidden_features=64,
        mlp_hidden_depth=2,
        mlp_activation="Tanh",
        mlp_norm=None,
        mlp_dropout_p=0.0,
        gamma_shared=False,
        gamma_init=[-0.1, 0.8],
        min_impl_method="hard",
        sign_impl_method="hard",
    )
    ckpt_path = tmp_path_factory.mktemp("ckpt") / "multi.ckpt"
    _save_fake_lightning_ckpt(model, ckpt_path)
    model_cfg = {
        "name": "MultiDMemBP",
        "num_iters": NUM_ITERS,
        "msg_features": 16,
        "mlp": {
            "hidden_features": 64,
            "hidden_depth": 2,
            "activation": "Tanh",
            "norm": None,
            "dropout_p": 0.0,
        },
        "min_impl_method": "hard",
        "sign_impl_method": "hard",
        "gamma_shared": False,
        "gamma_init": [-0.1, 0.8],
    }
    return {
        "model": model,
        "ckpt_path": ckpt_path,
        "model_cfg": model_cfg,
        "adapter_cls": MultiDMemBPDecoder,
        "registry_name": "TorchModel(MultiDMemBP)",
        "other_name": "LearnedDMemBP",
    }


@pytest.fixture
def setup(request, learned_setup, multi_setup) -> dict:
    return {"learned": learned_setup, "multi": multi_setup}[request.param]


def _build_adapter(
    setup: dict, pcm_prior: tuple[np.ndarray, np.ndarray]
) -> TorchModelDecoder:
    pcm, prior = pcm_prior
    return setup["adapter_cls"](
        pcm=pcm,
        prior=prior,
        max_iter=NUM_ITERS,
        model_cfg=setup["model_cfg"],
        ckpt_path=setup["ckpt_path"],
        device="cpu",
    )


@pytest.mark.parametrize("setup", ["learned", "multi"], indirect=True)
def test_factory_registration(setup, pcm_prior):
    name = setup["registry_name"]
    assert name in TORCHMODEL_DECODERS_REGISTRY
    assert name in DECODERS_REGISTRY
    assert name in ITERATIVE_DECODERS_REGISTRY

    pcm, prior = pcm_prior
    dec = create_decoder(
        name,
        pcm,
        prior,
        max_iter=NUM_ITERS,
        model_cfg=setup["model_cfg"],
        ckpt_path=setup["ckpt_path"],
        device="cpu",
    )
    assert isinstance(dec, setup["adapter_cls"])


def test_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError, match="cannot be instantiated"):
        TorchModelDecoder()


@pytest.mark.parametrize("setup", ["learned", "multi"], indirect=True)
def test_model_cfg_name_mismatch_raises(setup, pcm_prior):
    pcm, prior = pcm_prior
    bad_cfg = dict(setup["model_cfg"])
    bad_cfg["name"] = setup["other_name"]
    with pytest.raises(ValueError, match="different architecture"):
        setup["adapter_cls"](
            pcm=pcm,
            prior=prior,
            max_iter=NUM_ITERS,
            model_cfg=bad_cfg,
            ckpt_path=setup["ckpt_path"],
            device="cpu",
        )


@pytest.mark.parametrize("setup", ["learned", "multi"], indirect=True)
def test_missing_ckpt_raises(setup, pcm_prior, tmp_path):
    pcm, prior = pcm_prior
    with pytest.raises(FileNotFoundError):
        setup["adapter_cls"](
            pcm=pcm,
            prior=prior,
            max_iter=NUM_ITERS,
            model_cfg=setup["model_cfg"],
            ckpt_path=tmp_path / "does_not_exist.ckpt",
            device="cpu",
        )


@pytest.mark.parametrize("setup", ["learned", "multi"], indirect=True)
def test_output_shape_and_dtype(setup, pcm_prior, syndromes_np):
    pcm, _ = pcm_prior
    dec = _build_adapter(setup, pcm_prior)
    num_vars = pcm.shape[1]

    ehat, conv_mask, iters = dec.decode_batch_detailed(syndromes_np)
    assert ehat.shape == (BATCH, num_vars)
    assert ehat.dtype == np.uint8
    assert conv_mask.shape == (BATCH,)
    assert conv_mask.dtype == np.bool_
    assert iters.shape == (BATCH,)
    assert iters.dtype == np.int64

    ehat_only = dec.decode_batch(syndromes_np)
    np.testing.assert_array_equal(ehat_only, ehat)


@pytest.mark.parametrize("setup", ["learned", "multi"], indirect=True)
def test_decode_and_decode_detailed_not_implemented(setup, pcm_prior, syndromes_np):
    dec = _build_adapter(setup, pcm_prior)
    with pytest.raises(NotImplementedError):
        dec.decode(syndromes_np[0])
    with pytest.raises(NotImplementedError):
        dec.decode_detailed(syndromes_np[0])


@pytest.mark.parametrize("setup", ["learned", "multi"], indirect=True)
def test_matches_underlying_model(setup, pcm_prior, syndromes_np):
    pcm, _ = pcm_prior
    adapter = _build_adapter(setup, pcm_prior)

    model = setup["model"]
    model.eval()
    chkmat_t = torch.tensor(pcm, dtype=torch.float32)
    syndromes_t = torch.tensor(syndromes_np, dtype=torch.int32)
    with torch.inference_mode():
        ref = model.decode_inference(syndromes_t, chkmat_t)

    ehat_a, conv_a, iters_a = adapter.decode_batch_detailed(syndromes_np)
    np.testing.assert_array_equal(ehat_a, ref.ehat.cpu().numpy().astype(np.uint8))
    np.testing.assert_array_equal(conv_a, ref.converged_mask.cpu().numpy())
    np.testing.assert_array_equal(
        iters_a, ref.decoding_iters.cpu().numpy().astype(np.int64)
    )


@pytest.mark.parametrize("setup", ["learned", "multi"], indirect=True)
def test_syndrome_satisfaction(setup, pcm_prior, syndromes_np):
    pcm, _ = pcm_prior
    dec = _build_adapter(setup, pcm_prior)
    ehat_batch, conv_mask, _ = dec.decode_batch_detailed(syndromes_np)
    for i in range(BATCH):
        if conv_mask[i]:
            lhs = (pcm.astype(np.int64) @ ehat_batch[i].astype(np.int64)) % 2
            np.testing.assert_array_equal(
                lhs.astype(np.uint8),
                syndromes_np[i],
                err_msg=f"Converged ehat at index {i} doesn't satisfy syndrome",
            )


@pytest.mark.parametrize("setup", ["learned", "multi"], indirect=True)
def test_all_zero_syndrome_fast_path(setup, pcm_prior):
    pcm, _ = pcm_prior
    dec = _build_adapter(setup, pcm_prior)
    num_chks = pcm.shape[0]
    num_vars = pcm.shape[1]

    zero_batch = np.zeros((4, num_chks), dtype=np.uint8)
    ehat, conv_mask, iters = dec.decode_batch_detailed(zero_batch)
    np.testing.assert_array_equal(ehat, np.zeros((4, num_vars), dtype=np.uint8))
    np.testing.assert_array_equal(conv_mask, np.ones(4, dtype=np.bool_))
    np.testing.assert_array_equal(iters, np.zeros(4, dtype=np.int64))


@pytest.mark.parametrize("setup", ["learned", "multi"], indirect=True)
def test_repeated_decode_is_deterministic(setup, pcm_prior, syndromes_np):
    dec = _build_adapter(setup, pcm_prior)
    ehat_a, conv_a, iters_a = dec.decode_batch_detailed(syndromes_np)
    ehat_b, conv_b, iters_b = dec.decode_batch_detailed(syndromes_np)
    np.testing.assert_array_equal(ehat_a, ehat_b)
    np.testing.assert_array_equal(conv_a, conv_b)
    np.testing.assert_array_equal(iters_a, iters_b)
