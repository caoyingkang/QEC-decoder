"""Bridge PyTorch decoder models to sinter for Monte Carlo benchmarking."""
import numpy as np
import sinter
import torch

from .models import DecoderModel
from .utils.llr_utils import llrs_to_ehat
from .utils.tensor_utils import INT_DTYPE


class PyTorchSinterDecoder(sinter.Decoder):
    """Wrap a `DecoderModel` as a `sinter.Decoder` for use in `sinter.collect`."""

    def __init__(
        self,
        model: DecoderModel,
        obsmat: np.ndarray,
        *,
        device: str | None = None,
    ):
        """
        Parameters
        ----------
        model : DecoderModel
            PyTorch decoder model.

        obsmat : np.ndarray
            Observable matrix, shape=(num_obsers, num_vars). Used to convert error predictions to 
            observable predictions for sinter.

        device : str | None
            Device for inference ("cuda", "cpu", etc.). Default: "cuda" if available, else "cpu".
        """
        self.model = model
        self.obsmat = obsmat
        self.num_chks = model.num_chks

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._pcm_tensor = torch.tensor(model.pcm, dtype=INT_DTYPE, device=device)

        # Move model to device.
        self.model.to(device)
        # Set model to evaluation mode.
        self.model.eval()

    def compile_decoder_for_dem(self, *, dem):
        """Dummy method to satisfy the interface of `sinter.Decoder`. You should never call this method."""
        return self

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        """
        Decode bit-packed syndromes, and return bit-packed observable predictions.
        This method is meant to be used by `sinter.collect`. You should not call this method directly.
        """
        unpacked = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little")
        unpacked = unpacked[:, :self.num_chks]
        syndromes = torch.tensor(unpacked, dtype=INT_DTYPE, device=self.device)

        with torch.no_grad():
            llrs = self.model(syndromes)

        ehat, _ = llrs_to_ehat(llrs, syndromes, self._pcm_tensor)
        ehat_np = ehat.cpu().numpy().astype(np.uint8)

        observable_predict = (ehat_np @ self.obsmat.T) % 2
        return np.packbits(observable_predict, axis=1, bitorder="little")
