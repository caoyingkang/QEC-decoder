"""Bridge PyTorch decoder models to sinter for Monte Carlo benchmarking."""

import numpy as np
import torch
import sinter
from torchdecoder_core.models import DecoderModel
from torchdecoder_core.utils.decoding_utils import diagnose_convergence, gather_ehat
from torchdecoder_core.utils.tensor_utils import matmul_GF2


class PyTorchSinterDecoder(sinter.Decoder):
    """Wrap a `DecoderModel` as a `sinter.Decoder` for use in `sinter.collect`."""

    def __init__(
        self,
        model: DecoderModel,
        obsmat: np.ndarray,
        *,
        device: str,
    ):
        """
        Parameters
        ----------
        model : DecoderModel
            PyTorch decoder model.

        obsmat : np.ndarray
            Observable matrix, shape=(num_obsers, num_vars). Used to convert error predictions to
            observable predictions for sinter.

        device : str
            Device for inference ("cuda", "cpu", etc.).
        """
        self.model = model
        self.num_chks = model.num_chks
        self.device = device

        # Move model to device.
        self.model.to(device)
        # Set model to evaluation mode.
        self.model.eval()
        # Store helper tensors on device.
        self._chkmat = torch.tensor(model.pcm, dtype=torch.float32, device=device)
        self._obsmat = torch.tensor(obsmat, dtype=torch.float32, device=device)

    def compile_decoder_for_dem(self, *, dem):
        """Dummy method to satisfy the interface of `sinter.Decoder`. You should never call this method."""
        return self

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        """
        Decode bit-packed syndromes, and return bit-packed observable predictions.
        This method is meant to be used by `sinter.collect`. You should not call this method directly.
        """
        unpacked = np.unpackbits(
            bit_packed_detection_event_data, axis=1, bitorder="little"
        )
        syndromes = unpacked[:, : self.num_chks]
        syndromes_t = torch.as_tensor(syndromes, dtype=torch.int32, device=self.device)
        trivial_mask = torch.all(syndromes_t == 0, dim=1)  # (B,), bool

        with torch.inference_mode():
            llrs = self.model(syndromes_t)  # (I, B, V), float
            hard_decisions = (llrs < 0).float()  # (I, B, V), float ∈ {0.0, 1.0}
            _, output_iters = diagnose_convergence(
                hard_decisions,
                syndromes_t,
                self._chkmat,
            )  # (B,), long
            ehat = gather_ehat(
                hard_decisions, output_iters
            )  # (B, V), float ∈ {0.0, 1.0}

            obser_pred = matmul_GF2(ehat, self._obsmat.T)  # (B, O), int ∈ {0,1}
            obser_pred[trivial_mask] = 0

        obser_pred_np = obser_pred.cpu().numpy().astype(np.uint8)
        return np.packbits(obser_pred_np, axis=1, bitorder="little")
