"""Decoder wrapper used by the Monte Carlo collector."""

from typing import Optional, NamedTuple

import numpy as np
import qecdec
from qecdec.types import Bool1DArray, Bit2DArray, Int1DArray

# torchdecoder_core import has the side effect of registering torch decoders
# (LearnedDMemBP, MultiDMemBP) with qecdec.decoders; keep it so that
# `qecdec.create_decoder(...)` works for those names without any extra setup.
import torchdecoder_core  # noqa: F401


class DecodeResult(NamedTuple):
    """Result of batch decoding.

    Attributes
    ----------
    obser_correct_mask : Bool1DArray
        Boolean mask, shape=(batch_size,). True when the logical observables are
        predicted correctly for that shot.
    synd_match_mask : Bool1DArray
        Boolean mask, shape=(batch_size,). True when the predicted error pattern
        satisfies the syndrome for that shot.
    decoding_iters : Int1DArray or None
        Number of iterations the decoder ran for each shot, shape=(batch_size,).
        0 means the syndrome for that shot was all-zero (no decoding at all).
        ``None`` for non-iterative decoders.
    """

    obser_correct_mask: Bool1DArray
    synd_match_mask: Bool1DArray
    decoding_iters: Optional[Int1DArray]


class BenchmarkDecoder:
    """Wrap a ``qecdec.decoders.Decoder`` for use in the MC benchmark tool."""

    def __init__(self, decoder: qecdec.decoders.Decoder, obsmat: Bit2DArray):
        self.decoder = decoder
        self.chkmat = decoder.pcm
        self.obsmat = obsmat

    def decode(self, syndromes: Bit2DArray, observables: Bit2DArray) -> DecodeResult:
        """Decode a batch of syndromes.

        Parameters
        ----------
        syndromes : Bit2DArray
            Syndrome bits, shape=(batch_size, num_chks).
        observables : Bit2DArray
            Observable bits, shape=(batch_size, num_obsers).

        Returns
        -------
        DecodeResult
        """
        if isinstance(self.decoder, qecdec.decoders.IterativeDecoder):
            return self._decode_iterative(syndromes, observables)
        else:
            return self._decode_noniterative(syndromes, observables)

    def _decode_iterative(
        self, syndromes: Bit2DArray, observables: Bit2DArray
    ) -> DecodeResult:
        ehat, converged_mask, decoding_iters = self.decoder.decode_batch_detailed(
            syndromes, parallel=True
        )
        obser_pred = (ehat @ self.obsmat.T) % 2
        obser_correct_mask = np.all(obser_pred == observables, axis=1)
        return DecodeResult(
            obser_correct_mask=obser_correct_mask,
            synd_match_mask=converged_mask,
            decoding_iters=decoding_iters,
        )

    def _decode_noniterative(
        self, syndromes: Bit2DArray, observables: Bit2DArray
    ) -> DecodeResult:
        ehat = self.decoder.decode_batch(syndromes)
        obser_pred = (ehat @ self.obsmat.T) % 2
        obser_correct_mask = np.all(obser_pred == observables, axis=1)
        synd_pred = (ehat @ self.chkmat.T) % 2
        synd_match_mask = np.all(synd_pred == syndromes, axis=1)
        return DecodeResult(
            obser_correct_mask=obser_correct_mask,
            synd_match_mask=synd_match_mask,
            decoding_iters=None,
        )
