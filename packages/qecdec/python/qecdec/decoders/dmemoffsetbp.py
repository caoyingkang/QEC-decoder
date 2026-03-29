from typing import Optional

import numpy as np

from .base import IterativeDecoder
from ..qecdec import DMemOffsetBPDecoderRust
from ..types import (
    Bool1DArray,
    Bit1DArray,
    Bit2DArray,
    Int1DArray,
    Float1DArray,
    Float2DArray,
)


class DMemOffsetBPDecoder(IterativeDecoder):
    """Disordered-memory, offset-normalized min-sum BP decoder."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        gamma: Float1DArray,
        max_iter: int,
        norm: list[list[float]] | float = 1.0,
        offset: list[list[float]] | float = 0.0,
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

        gamma : ndarray
            Memory strength for each variable node, shape=(num_vars,), float64.
            Use 0.0 for no memory at a node.

        max_iter : int
            Max number of BP iterations.

        norm : list[list[float]] or float
            `norm[i][k]` is the normalization factor for the edge from check node `i` to its
            `k`-th neighboring variable node. If a float is provided, the same value is used
            for all normalization factors. Default is 1.0, meaning no normalization.

        offset : list[list[float]] or float
            `offset[i][k]` is the offset parameter for the edge from check node `i` to its
            `k`-th neighboring variable node. If a float is provided, the same value is used
            for all offset parameters. Default is 0.0, meaning no offset.
        """
        super().__init__(max_iter, pcm, prior)

        assert isinstance(gamma, np.ndarray) and gamma.shape == (self.num_vars,)
        self.gamma = gamma

        if isinstance(norm, list):
            assert len(norm) == self.num_chks
            assert all(isinstance(x, list) for x in norm)
            assert all(len(norm[i]) == self.chk_degs[i] for i in range(self.num_chks))
        elif isinstance(norm, (float, int)):
            norm = [
                [norm for _ in range(self.chk_degs[i])] for i in range(self.num_chks)
            ]
        else:
            raise ValueError("Invalid data type for `norm`")
        self.norm = norm

        if isinstance(offset, list):
            assert len(offset) == self.num_chks
            assert all(isinstance(x, list) for x in offset)
            assert all(len(offset[i]) == self.chk_degs[i] for i in range(self.num_chks))
        elif isinstance(offset, (float, int)):
            offset = [
                [offset for _ in range(self.chk_degs[i])] for i in range(self.num_chks)
            ]
        else:
            raise ValueError("Invalid data type for `offset`")
        self.offset = offset

        self._decoder = DMemOffsetBPDecoderRust(
            self.pcm,
            self.prior,
            gamma=self.gamma,
            max_iter=self.max_iter,
            norm=self.norm,
            offset=self.offset,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._decoder = DMemOffsetBPDecoderRust(
            self.pcm,
            self.prior,
            gamma=self.gamma,
            max_iter=self.max_iter,
            norm=self.norm,
            offset=self.offset,
        )

    def decode(self, syndrome: Bit1DArray) -> Bit1DArray:
        return self._decoder.decode(syndrome)

    def decode_batch(self, syndrome_batch: Bit2DArray) -> Bit2DArray:
        return self._decoder.decode_batch(syndrome_batch)

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
        return self._decoder.decode_batch_detailed(syndrome_batch)
