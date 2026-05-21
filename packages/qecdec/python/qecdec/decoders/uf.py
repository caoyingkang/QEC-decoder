import numpy as np

from ..qecdec import UnionFindDecoderRust
from ..types import Bit1DArray, Bit2DArray, Float1DArray
from .base import Decoder


class UnionFindDecoder(Decoder, registry_name="UnionFind"):
    """Union-Find decoder."""

    def __init__(self, pcm: Bit2DArray, prior: Float1DArray):
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), uint8 ∈ {0,1}.
            Each row (check) must have at least two nonzero entries; each column
            (variable) must have at least one and at most two nonzero entries.
        prior : ndarray
            Prior error probabilities, shape=(num_vars,), float64 ∈ (0,0.5).

        Notes
        -----
        Current implementation of the UnionFind decoder is unweighted, i.e.,
        ``prior`` is merely a placeholder.
        """
        super().__init__(pcm, prior)

        if not np.all(self.pcm.sum(axis=0) <= 2):
            raise ValueError(
                "Each column (variable) must have at most two nonzero entries."
            )

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> UnionFindDecoderRust:
        return UnionFindDecoderRust(self.pcm)

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
