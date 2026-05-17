"""Helpers for instantiating PyTorch decoders from Lightning checkpoints.

These functions take explicit ``ckpt_path`` arguments — they do not discover
checkpoints. Filesystem discovery (walking ``runs/`` directories, picking out
the ``best_model.ckpt``, etc.) belongs to callers.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torchdecoder_core.models import DecoderModel, build_decoder_model

from .types import Bit2DArray, Float1DArray


def load_torchdecoder(
    *,
    chkmat: Bit2DArray,
    prior: Float1DArray,
    model_cfg: DictConfig,
    max_iter: int,
    ckpt_path: Path,
    use_prior_in_ckpt: bool,
) -> DecoderModel:
    """Build a :class:`DecoderModel` from a checkpoint.

    The architecture is constructed from ``model_cfg`` (with
    ``num_iters`` overridden by ``max_iter`` for inference), and the weights
    are loaded from ``ckpt_path``. If ``use_prior_in_ckpt`` is False, the
    ``prior_llr`` buffer in the checkpoint is skipped and the ``prior``
    passed in is used instead.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    assert "num_iters" in model_cfg
    new_model_cfg = deepcopy(model_cfg)
    new_model_cfg.num_iters = max_iter
    model = build_decoder_model(chkmat, prior, new_model_cfg)

    if use_prior_in_ckpt:
        model.load_lightning_checkpoint(ckpt_path, skip_keys=[])
    else:
        model.load_lightning_checkpoint(ckpt_path, skip_keys=["prior_llr"])
    return model


def load_gamma_from_checkpoint(ckpt_path: Path) -> np.ndarray:
    """Extract the trained ``gamma`` parameter from a LearnedDMemBP checkpoint.

    The Lightning state_dict stores it under ``"model.gamma"`` (DecodingModule
    wraps the model under attribute ``model``). Returns a 1D float64 array of
    shape ``(num_vars,)`` ready to pass as the ``gamma`` argument to
    ``qecdec.decoders.DMemBPDecoder``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    if "model.gamma" not in state_dict:
        raise KeyError(
            f"'model.gamma' not found in checkpoint state_dict at {ckpt_path}. "
            f"Available keys: {sorted(state_dict.keys())}"
        )
    gamma = state_dict["model.gamma"].detach().cpu().to(torch.float64).numpy()
    if gamma.ndim != 1:
        raise ValueError(
            f"Expected 1D gamma vector, got shape {gamma.shape} from {ckpt_path}"
        )
    return gamma


def load_prior_from_checkpoint(ckpt_path: Path) -> np.ndarray:
    """Extract the prior probability vector from a LearnedDMemBP checkpoint.

    The checkpoint stores ``prior_llr`` (LLRs); convert back to probabilities
    via ``p = 1 / (1 + exp(prior_llr))`` so it can be passed to ``DMemBPDecoder``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    if "model.prior_llr" not in state_dict:
        raise KeyError(
            f"'model.prior_llr' not found in checkpoint state_dict at {ckpt_path}. "
            f"Available keys: {sorted(state_dict.keys())}"
        )
    prior_llr = state_dict["model.prior_llr"].detach().cpu().to(torch.float64).numpy()
    if prior_llr.ndim != 1:
        raise ValueError(
            f"Expected 1D prior_llr vector, got shape {prior_llr.shape} from {ckpt_path}"
        )
    return 1.0 / (1.0 + np.exp(prior_llr))
