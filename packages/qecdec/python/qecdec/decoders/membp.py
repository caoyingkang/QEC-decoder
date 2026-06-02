import numpy as np

from ..qecdec import DMemBPDecoderRust
from ..types import (
    Bit1DArray,
    Bit2DArray,
    Bool1DArray,
    Float1DArray,
    Int1DArray,
)
from .base import IterativeDecoder


class MemBPDecoder(IterativeDecoder, registry_name="MemBP"):
    """Min-sum BP decoder with memory coefficient."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        gamma: float,
        max_iter: int,
        norm: float | None = None,
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
        gamma : float
            Memory coefficient. 0.0 means no memory and reduces to regular BP.
        max_iter : int
            Max number of BP iterations.
        norm : float or None
            Message normalization factor; `None` means no normalization.
        """
        super().__init__(pcm, prior, max_iter=max_iter)

        self.gamma = float(gamma)
        self.norm = norm

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> DMemBPDecoderRust:
        return DMemBPDecoderRust(
            self.pcm,
            self.prior,
            gamma=np.full_like(self.prior, self.gamma),
            max_iter=self.max_iter,
            norm=self.norm,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._decoder = self._build_decoder()

    def decode(self, syndrome: Bit1DArray) -> Bit1DArray:
        ehat, _, _ = self._decoder.decode_detailed(syndrome)
        return ehat

    def decode_batch(
        self, syndrome_batch: Bit2DArray, *, parallel: bool = False
    ) -> Bit2DArray:
        ehat_batch, _, _ = self._decoder.decode_batch_detailed(
            syndrome_batch, parallel=parallel
        )
        return ehat_batch

    def decode_detailed(self, syndrome: Bit1DArray) -> tuple[Bit1DArray, bool, int]:
        """Decode a syndrome vector with detailed diagnostics.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector, shape=(num_chks,), dtype=uint8.

        Returns
        -------
        ehat : ndarray
            Estimated error vector, shape=(num_vars,), dtype=uint8.
        converged : bool
            Whether the decoder converged (i.e. the syndrome was satisfied).
        num_iter : int
            The number of BP iterations actually run.
        """
        return self._decoder.decode_detailed(syndrome)

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray, *, parallel: bool = False
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndrome vectors with detailed diagnostics.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.
        parallel : bool
            Whether to use multithreaded decoding.

        Returns
        -------
        ehat_batch : ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.
        converged_mask : ndarray
            Whether the decoder converged in each shot, shape=(batch_size,), dtype=bool.
        decoding_iters : ndarray
            Number of BP iterations actually run in each shot, shape=(batch_size,), dtype=int64.
        """
        return self._decoder.decode_batch_detailed(syndrome_batch, parallel=parallel)
