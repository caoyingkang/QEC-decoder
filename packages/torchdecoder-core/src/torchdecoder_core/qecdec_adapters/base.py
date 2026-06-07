from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from qecdec.decoders import IterativeDecoder
from qecdec.types import Bit1DArray, Bit2DArray, Bool1DArray, Float1DArray, Int1DArray

from ..models import build_decoder_model


class TorchModelDecoder(IterativeDecoder):
    """Abstract base class for iterative decoders wrapping PyTorch `DecoderModel`.

    Subclasses must define class-attribute `model_name` to a str understood by
    `build_decoder_model`.
    """

    registry: ClassVar[dict[str, type["TorchModelDecoder"]]] = {}

    def __init_subclass__(cls, registry_name: str | None = None) -> None:
        super().__init_subclass__(registry_name)

        # Make sure subclasses define class-attribute 'model_name'
        if not hasattr(cls, "model_name"):
            raise TypeError(
                f"Class {cls.__name__} must define a 'model_name' class-attribute."
            )

        # Only register subclasses that set `registry_name`.
        if registry_name is not None:
            if registry_name in TorchModelDecoder.registry:
                raise ValueError(
                    f"TorchModelDecoder registry_name {registry_name!r} is already assigned."
                )
            TorchModelDecoder.registry[registry_name] = cls

    def __new__(cls, *args, **kwargs):
        if cls is TorchModelDecoder:
            raise TypeError(f"Abstract class {cls.__name__} cannot be instantiated")
        return super().__new__(cls)

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        max_iter: int,
        model_cfg: DictConfig | dict[str, Any],
        ckpt_path: Path | str,
        device: str,
    ):
        """
        Build a fresh ``DecoderModel`` for the given parity-check matrix, prior vector,
        and model configuration, load weights from a Lightning checkpoint, move the
        model to the target device, and switch it to eval mode.

        Parameters
        ----------
        pcm : np.ndarray
            Parity-check matrix, shape=(num_chks, num_vars), uint8 ∈ {0,1}.
        prior : np.ndarray
            Prior error probabilities, shape=(num_vars,), float64 ∈ (0, 0.5).
        max_iter : int
            Max iterations. Overrides any value in ``model_cfg.num_iters``.
        model_cfg : DictConfig | dict[str, Any]
            Architecture hyperparameters (excluding "name" and "num_iters").
        ckpt_path : Path | str
            Lightning checkpoint to load (excluding "prior_llr").
        device : str
            Device for inference (e.g. "cpu", "cuda").
        """
        model_cfg = DictConfig(model_cfg)
        if model_cfg.name != self.model_name:
            raise ValueError("`model_cfg` is for a different architecture")

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        super().__init__(max_iter=max_iter, pcm=pcm, prior=prior)
        self.device = device

        assert "num_iters" in model_cfg
        model_cfg = OmegaConf.merge(model_cfg, {"num_iters": max_iter})
        self.model = build_decoder_model(pcm, prior, model_cfg)
        self.model.load_lightning_checkpoint(ckpt_path, skip_keys=["prior_llr"])
        self.model.to(device)
        self.model.eval()

        # Store helper tensors on device.
        self._chkmat_t = torch.tensor(pcm, dtype=torch.float32, device=device)

    def decode(self, syndrome: Bit1DArray) -> Bit1DArray:
        raise NotImplementedError

    def decode_batch(self, syndrome_batch: Bit2DArray) -> Bit2DArray:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.

        Returns
        -------
        ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.
        """
        ehat_batch, _, _ = self.decode_batch_detailed(syndrome_batch)
        return ehat_batch

    def decode_detailed(self, syndrome: np.ndarray, **kwargs) -> tuple:
        raise NotImplementedError

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndrome vectors with detailed diagnostics.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.

        Returns
        -------
        ehat_batch : ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.
        converged_mask : ndarray
            Whether the decoder converged in each shot, shape=(batch_size,), dtype=bool.
        decoding_iters : ndarray
            Number of BP iterations actually run in each shot, shape=(batch_size,), dtype=int64.
        """
        syndromes_t = torch.as_tensor(
            syndrome_batch, dtype=torch.int32, device=self.device
        )
        with torch.inference_mode():
            ehat_t, converged_mask_t, decoding_iters_t = self.model.decode_inference(
                syndromes_t, self._chkmat_t
            )
        return (
            ehat_t.cpu().numpy().astype(np.uint8),
            converged_mask_t.cpu().numpy(),
            decoding_iters_t.cpu().numpy().astype(np.int64),
        )


# Module-level alias for the class-attribute registry. This points to the
# same underlying object, so updates from `__init_subclass__` flow through to
# `from torchdecoder_core.qecdec_adapters import TORCHMODEL_DECODERS_REGISTRY` callers.
TORCHMODEL_DECODERS_REGISTRY = TorchModelDecoder.registry
