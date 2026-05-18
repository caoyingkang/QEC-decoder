from typing import Optional

from .base import IterativeDecoder
from ..qecdec import BPDecoderRust
from ..types import (
    Bool1DArray,
    Bit1DArray,
    Bit2DArray,
    Int1DArray,
    Float1DArray,
    Float2DArray,
)


class BPDecoder(IterativeDecoder, registry_name="BP"):
    """Belief Propagation decoder (min-sum variant)."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        max_iter: int,
        norm: Optional[float] = None,
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
        max_iter : int
            Max number of BP iterations.
        norm : float or None
            Message normalization factor; `None` means no normalization.
        """
        super().__init__(max_iter, pcm, prior)
        self.norm = norm

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> BPDecoderRust:
        return BPDecoderRust(
            self.pcm,
            self.prior,
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
        ehat, _, _, _ = self._decoder.decode_detailed(
            syndrome, record_llr_history=False
        )
        return ehat

    def decode_batch(
        self, syndrome_batch: Bit2DArray, *, parallel: bool = False
    ) -> Bit2DArray:
        ehat_batch, _, _ = self._decoder.decode_batch_detailed(
            syndrome_batch, parallel=parallel
        )
        return ehat_batch

    def decode_detailed(
        self,
        syndrome: Bit1DArray,
        *,
        record_llr_history: bool = False,
    ) -> tuple[Bit1DArray, bool, int, Optional[Float2DArray]]:
        """Decode a syndrome vector with detailed diagnostics.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector, shape=(num_chks,), dtype=uint8.
        record_llr_history : bool
            Whether to return the history of posterior LLR values.

        Returns
        -------
        ehat : ndarray
            Estimated error vector, shape=(num_vars,), dtype=uint8.
        converged : bool
            Whether the decoder converged (i.e. the syndrome was satisfied).
        num_iter : int
            The number of BP iterations actually run.
        llr_hist : ndarray or None
            If `record_llr_history` is True: posterior LLR values at each BP iteration,
            shape=(num_iter, num_vars), dtype=float64; otherwise, `None`.
        """
        return self._decoder.decode_detailed(
            syndrome, record_llr_history=record_llr_history
        )

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
