from pathlib import Path

import numpy as np

from qecdec.decoders import DMemBPDecoder
from qecdec.types import Bit2DArray, Float1DArray
import torch


def _load_gamma_from_checkpoint(ckpt_path: Path) -> np.ndarray:
    """Extract the trained ``gamma`` parameter from a LearnedDMemBP Lightning checkpoint.

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


class LearnedDMemBPDecoder(DMemBPDecoder, registry_name="LearnedDMemBP"):
    """Disordered-memory min-sum BP decoder."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        ckpt_path: Path | str,
        max_iter: int,
    ):
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), uint8 ∈ {0,1}.
            Each row (check) must have at least two nonzero entries; each column
            (variable) must have at least one nonzero entry.
        prior : ndarray
            Prior error probabilities, shape=(num_vars,), float64 ∈ (0,0.5).
        ckpt_path : Path or str
            Path to the Lightning checkpoint file containing the learned parameters.
        max_iter : int
            Max number of BP iterations.
        """
        gamma = _load_gamma_from_checkpoint(Path(ckpt_path))
        super().__init__(pcm, prior, gamma=gamma, max_iter=max_iter)
