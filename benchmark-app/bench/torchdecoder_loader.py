from pathlib import Path
from copy import deepcopy

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
    """
    Construct a `torchdecoder_core.models.DecoderModel` from `chkmat`, `prior`, `model_cfg`,
    and load its parameters from checkpoint at `ckpt_path`. Use `max_iter` as the number of
    iterations for inference (overriding `model_cfg.num_iters`).

    If `use_prior_in_ckpt` is True, load and use the prior LLRs from the checkpoint.
    Otherwise, use the prior passed as an argument.
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
