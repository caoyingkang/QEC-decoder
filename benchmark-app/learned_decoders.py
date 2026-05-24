from pathlib import Path

import numpy as np

from qecdec.decoders import DMemBPDecoder, MultiRelayBPDecoder, RelayBPDecoder
from qecdec.types import Bit2DArray, Float1DArray
import torch

from constants import REPO_ROOT


def _load_gamma_from_checkpoint(ckpt_rel_path: Path) -> np.ndarray:
    """Extract the trained ``gamma`` parameter from a LearnedDMemBP Lightning checkpoint.

    The Lightning state_dict stores it under ``"model.gamma"`` (DecodingModule
    wraps the model under attribute ``model``). Returns a 1D float64 array of
    shape ``(num_vars,)`` ready to pass as the ``gamma`` argument to
    ``qecdec.decoders.DMemBPDecoder``.
    """
    ckpt_abs_path = REPO_ROOT / ckpt_rel_path
    ckpt = torch.load(ckpt_abs_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    if "model.gamma" not in state_dict:
        raise KeyError(
            f"'model.gamma' not found in checkpoint state_dict at {ckpt_abs_path}. "
            f"Available keys: {sorted(state_dict.keys())}"
        )
    gamma = state_dict["model.gamma"].detach().cpu().to(torch.float64).numpy()
    if gamma.ndim != 1:
        raise ValueError(
            f"Expected 1D gamma vector, got shape {gamma.shape} from {ckpt_abs_path}"
        )
    return gamma


class LearnedDMemBPDecoder(DMemBPDecoder, registry_name="LearnedDMemBP"):
    """DMemBP decoder with learned memory weights."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        ckpt_rel_path: Path | str,
        max_iter: int,
    ):
        """
        ``ckpt_rel_path`` is the relative path from the repo root to the Lightning
        checkpoint file containing the learned parameters. Other parameters are
        passed to the DMemBPDecoder constructor.
        """
        gamma = _load_gamma_from_checkpoint(ckpt_rel_path)
        super().__init__(pcm, prior, gamma=gamma, max_iter=max_iter)


class LearnedRelayBPDecoder(RelayBPDecoder, registry_name="LearnedRelayBP"):
    """RelayBP decoder with learned memory weights in the initial stage."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        ckpt_rel_path: Path | str,
        gamma_dist_interval: tuple[float, float],
        num_relays: int,
        pre_iter: int,
        max_iter_per_relay: int,
        stop_nconv: int,
    ):
        """
        ``ckpt_rel_path`` is the relative path from the repo root to the Lightning
        checkpoint file containing the learned parameters. Other parameters are
        passed to the RelayBP constructor.
        """
        gamma0 = _load_gamma_from_checkpoint(ckpt_rel_path)
        super().__init__(
            pcm,
            prior,
            gamma0=gamma0,
            gamma_dist_interval=gamma_dist_interval,
            num_relays=num_relays,
            pre_iter=pre_iter,
            max_iter_per_relay=max_iter_per_relay,
            stop_nconv=stop_nconv,
        )


class LearnedMultiRelayBPDecoder(
    MultiRelayBPDecoder, registry_name="LearnedMultiRelayBP"
):
    """MultiRelayBP decoder with learned memory weights in the initial stage."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        ckpt_rel_path: Path | str,
        gamma_dist_interval: tuple[float, float],
        num_chains: int,
        num_relays: int,
        pre_iter: int,
        max_iter_per_relay: int,
        stop_nconv: int,
    ):
        """
        ``ckpt_rel_path`` is the relative path from the repo root to the Lightning
        checkpoint file containing the learned parameters. Other parameters are
        passed to the MultiRelayBP constructor.
        """
        gamma0 = _load_gamma_from_checkpoint(ckpt_rel_path)
        super().__init__(
            pcm,
            prior,
            gamma0=gamma0,
            gamma_dist_interval=gamma_dist_interval,
            num_chains=num_chains,
            num_relays=num_relays,
            pre_iter=pre_iter,
            max_iter_per_relay=max_iter_per_relay,
            stop_nconv=stop_nconv,
        )
