from typing import Optional

import numpy as np
import pymatching

from .base import Decoder
from ..types import (
    Bit1DArray,
    Bit2DArray,
    Float1DArray,
)


class MWPMDecoder(Decoder, registry_name="MWPM"):
    """Minimum Weight Perfect Matching decoder. This class is a wrapper for the pymatching library."""

    def __init__(self, pcm: Bit2DArray, prior: Optional[Float1DArray] = None):
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), uint8 ∈ {0,1}.
            Each row (check) must have at least two nonzero entries; each column
            (variable) must have at least one and at most two nonzero entries.

        prior : ndarray or None
            Prior error probabilities, shape=(num_vars,), float64 ∈ (0,0.5).
            If None, a uniform prior is assumed.
        """
        super().__init__(pcm, prior)

        if not np.all(self.pcm.sum(axis=0) <= 2):
            raise ValueError(
                "Each column (variable) must have at most two nonzero entries."
            )

        self.llr = (
            np.log((1.0 - self.prior) / self.prior) if self.prior is not None else None
        )
        self._decoder = self._build_decoder()

    def _build_decoder(self) -> pymatching.Matching:
        return pymatching.Matching.from_check_matrix(self.pcm, weights=self.llr)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._decoder = self._build_decoder()

    def decode(self, syndrome: Bit1DArray) -> Bit1DArray:
        return self._decoder.decode(syndrome)

    def decode_batch(self, syndrome_batch: Bit2DArray) -> Bit2DArray:
        return self._decoder.decode_batch(syndrome_batch)
