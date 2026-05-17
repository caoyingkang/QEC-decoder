"""Decoder adapters used by the Monte Carlo collector.

Each :class:`BenchmarkDecoder` exposes a single ``decode(syndromes)`` method
that returns predicted observables, a per-shot syndrome-match mask, and (for
iterative decoders) a per-shot iteration count. The two concrete adapters
bridge to:

- :class:`qecdec.decoders.Decoder` (MWPM, BP, MemBP, RelayBP, BPOSD, …) via
  :class:`QecdecBenchmarkDecoder`;
- :class:`torchdecoder_core.models.DecoderModel` (learned BP variants) via
  :class:`PyTorchBenchmarkDecoder`.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, NamedTuple

import numpy as np
import torch
import qecdec
from qecdec.decoders import IterativeDecoder
from torchdecoder_core.models import DecoderModel
from torchdecoder_core.utils.tensor_utils import matmul_GF2

from .types import Bool1DArray, Bit2DArray, Int1DArray


class DecodeResult(NamedTuple):
    """Result of decoding a batch of syndromes.

    Attributes
    ----------
    obser_pred : Bit2DArray
        Predicted observables, shape=(batch_size, num_obsers).
    synd_match_mask : Bool1DArray
        Boolean mask, shape=(batch_size,). True when the predicted error pattern
        satisfies the syndrome for that shot.
    decoding_iters : Int1DArray or None
        Number of iterations the decoder ran for each shot, shape=(batch_size,).
        0 means the syndrome for that shot was all-zero (no decoding at all).
        `None` for non-iterative decoders.
    """

    obser_pred: Bit2DArray
    synd_match_mask: Bool1DArray
    decoding_iters: Optional[Int1DArray]


class BenchmarkDecoder(ABC):
    """Abstract base class for decoders used by the MC benchmark tool."""

    @abstractmethod
    def decode(self, syndromes: Bit2DArray) -> DecodeResult:
        """Decode a batch of syndromes.

        Parameters
        ----------
        syndromes : Bit2DArray
            Syndrome bits, shape=(batch_size, num_chks).

        Returns
        -------
        DecodeResult
        """
        ...


class PyTorchBenchmarkDecoder(BenchmarkDecoder):
    """Wrap a PyTorch `DecoderModel` for use in the MC benchmark tool."""

    def __init__(
        self,
        model: DecoderModel,
        obsmat: Bit2DArray,
        *,
        device: Literal["cuda", "cpu"],
    ):
        """
        Parameters
        ----------
        model : DecoderModel
            PyTorch decoder model.
        obsmat : Bit2DArray
            Observable matrix, shape=(num_obsers, num_vars).
        device : Literal["cuda", "cpu"]
            Device on which to run the model for inference.
        """
        self.model = model
        self.device = device

        self.model.to(device)
        self.model.eval()

        self._chkmat = torch.tensor(model.pcm, dtype=torch.float32, device=device)
        self._obsmat = torch.tensor(obsmat, dtype=torch.float32, device=device)

    def decode(self, syndromes: Bit2DArray) -> DecodeResult:
        syndromes_t = torch.as_tensor(
            syndromes, dtype=torch.int32, device=self.device
        )  # (B, C), int ∈ {0, 1}

        with torch.inference_mode():
            ehat, converged_mask, decoding_iters = self.model.decode_inference(
                syndromes_t, self._chkmat
            )

            obser_pred = matmul_GF2(ehat, self._obsmat.T)  # (B, O), int ∈ {0,1}

        return DecodeResult(
            obser_pred=obser_pred.cpu().numpy().astype(np.uint8),
            synd_match_mask=converged_mask.cpu().numpy(),
            decoding_iters=decoding_iters.cpu().numpy(),
        )


class QecdecBenchmarkDecoder(BenchmarkDecoder):
    """Wrap a `qecdec.decoders.Decoder` (MWPM, BP, etc.) for use in the MC benchmark tool."""

    def __init__(self, decoder: qecdec.decoders.Decoder, obsmat: Bit2DArray):
        """
        Parameters
        ----------
        decoder : qecdec.decoders.Decoder
            Any object with `pcm` attribute and `decode_batch(syndromes) -> ehat` method.
        obsmat : Bit2DArray
            Observable matrix, shape=(num_obsers, num_vars).
        """
        self.decoder = decoder
        self.chkmat = decoder.pcm.astype(np.uint8)
        self.obsmat = obsmat

    def decode(self, syndromes: Bit2DArray) -> DecodeResult:
        if isinstance(self.decoder, IterativeDecoder):
            return self._decode_iterative(syndromes)
        else:
            return self._decode_noniterative(syndromes)

    def _decode_iterative(self, syndromes: Bit2DArray) -> DecodeResult:
        trivial_mask = np.all(syndromes == 0, axis=1)  # (B,)
        ehat, converged_mask, decoding_iters = self.decoder.decode_batch_detailed(
            syndromes
        )  # (B, V), (B,), (B,)
        obser_pred = (ehat @ self.obsmat.T) % 2  # (B, O)
        obser_pred[trivial_mask] = 0
        synd_match_mask = converged_mask | trivial_mask
        decoding_iters[trivial_mask] = 0
        return DecodeResult(
            obser_pred=obser_pred.astype(np.uint8),
            synd_match_mask=synd_match_mask,
            decoding_iters=decoding_iters,
        )

    def _decode_noniterative(self, syndromes: Bit2DArray) -> DecodeResult:
        ehat = self.decoder.decode_batch(syndromes)  # (B, V)
        obser_pred = (ehat @ self.obsmat.T) % 2  # (B, O)
        synd_pred = (ehat @ self.chkmat.T) % 2  # (B, C)
        synd_match_mask = np.all(synd_pred == syndromes, axis=1)  # (B,)
        return DecodeResult(
            obser_pred=obser_pred.astype(np.uint8),
            synd_match_mask=synd_match_mask,
            decoding_iters=None,
        )
