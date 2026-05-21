import numpy as np

from ..qecdec import DMemOffsetBPDecoderRust
from ..types import (
    Bit1DArray,
    Bit2DArray,
    Bool1DArray,
    Float1DArray,
    Int1DArray,
)
from .base import IterativeDecoder


class DMemOffsetBPDecoder(IterativeDecoder, registry_name="DMemOffsetBP"):
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
        super().__init__(pcm, prior, max_iter=max_iter)

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

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> DMemOffsetBPDecoderRust:
        return DMemOffsetBPDecoderRust(
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
