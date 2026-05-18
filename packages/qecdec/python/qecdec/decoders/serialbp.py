from typing import Optional

import numpy as np

from .base import IterativeDecoder
from ..qecdec import SerialBPDecoderRust
from ..types import (
    Bool1DArray,
    Bit1DArray,
    Bit2DArray,
    Int1DArray,
    Float1DArray,
)


class SerialBPDecoder(IterativeDecoder, registry_name="SerialBP"):
    """Belief Propagation decoder with serial message-passing schedule and
    min-sum CN update rule.
    """

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        max_iter: int,
        vn_order: Optional[Int1DArray] = None,
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
            Maximum number of iterations (one iteration = one full pass over
            `vn_order`).
        vn_order : ndarray or None
            Permutation of variable nodes, shape=(num_vars,).
            If None, the natural order ``np.arange(num_vars)`` is used.
        """
        super().__init__(max_iter, pcm, prior)

        if vn_order is None:
            self.vn_order = np.arange(self.num_vars, dtype=np.int64)
        else:
            assert vn_order.shape == (self.num_vars,), (
                f"vn_order must have shape ({self.num_vars},), got {vn_order.shape}"
            )
            assert np.array_equal(np.sort(vn_order), np.arange(self.num_vars)), (
                "vn_order must be a permutation of 0, 1, ..., num_vars-1"
            )
            self.vn_order = np.asarray(vn_order, dtype=np.int64)

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> SerialBPDecoderRust:
        return SerialBPDecoderRust(
            self.pcm,
            self.prior,
            max_iter=self.max_iter,
            vn_order=self.vn_order,
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
            The number of iterations actually run.
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
            Number of iterations actually run in each shot, shape=(batch_size,), dtype=int64.
        """
        return self._decoder.decode_batch_detailed(syndrome_batch, parallel=parallel)
