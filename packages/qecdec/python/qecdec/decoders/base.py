from abc import ABC, abstractmethod
from typing import Optional
from functools import cached_property

import numpy as np

from ..types import (
    Bit1DArray,
    Bit2DArray,
    Float1DArray,
)


class Decoder(ABC):
    """Abstract base class for decoders."""

    def __init__(self, pcm: Bit2DArray, prior: Optional[Float1DArray] = None):
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), uint8 ∈ {0,1}.
            Each row (check) must have at least two nonzero entries; each column
            (variable) must have at least one nonzero entry.

        prior : ndarray or None
            Prior error probabilities, shape=(num_vars,), float64 ∈ (0,0.5).
            If None, the decoder either assumes a uniform prior or does not
            depend on the prior at all.
        """
        assert isinstance(pcm, np.ndarray) and pcm.ndim == 2
        assert np.all((pcm == 0) | (pcm == 1))
        if not np.all(pcm.sum(axis=1) >= 2):
            raise ValueError("Each row (check) must have at least two nonzero entries.")
        if not np.all(pcm.sum(axis=0) >= 1):
            raise ValueError(
                "Each column (variable) must have at least one nonzero entry."
            )
        self.pcm = pcm

        if prior is None:
            self.prior = None
        else:
            assert isinstance(prior, np.ndarray) and prior.ndim == 1
            assert pcm.shape[1] == prior.shape[0]
            assert np.all(prior > 0) and np.all(prior < 0.5)
            self.prior = prior

    @cached_property
    def num_chks(self) -> int:
        """Number of check nodes."""
        return self.pcm.shape[0]

    @cached_property
    def num_vars(self) -> int:
        """Number of variable nodes."""
        return self.pcm.shape[1]

    @cached_property
    def chk_degs(self) -> list[int]:
        """Degree (i.e., number of neighbors) of each check node."""
        return self.pcm.sum(axis=1).tolist()

    @cached_property
    def var_degs(self) -> list[int]:
        """Degree (i.e., number of neighbors) of each variable node."""
        return self.pcm.sum(axis=0).tolist()

    def _build_tanner_graph(self):
        """
        Set attributes `self.chk_nbrs`, `self.var_nbrs`, `self.chk_nbr_pos`, and `self.var_nbr_pos`,
        all of type `list[list[int]]`, where:
        - `chk_nbrs[i]` = list of all VNs connected to CN `i`, sorted in increasing order.
        - `var_nbrs[j]` = list of all CNs connected to VN `j`, sorted in increasing order.
        - `chk_nbr_pos[i][k]` = position of CN `i` in the list of neighbors of the VN `chk_nbrs[i][k]`.
            I.e., `var_nbrs[chk_nbrs[i][k]][chk_nbr_pos[i][k]] = i`.
        - `var_nbr_pos[j][k]` = position of VN `j` in the list of neighbors of the CN `var_nbrs[j][k]`.
            I.e., `chk_nbrs[var_nbrs[j][k]][var_nbr_pos[j][k]] = j`.
        """
        m, n = self.pcm.shape
        chk_nbrs = [[] for _ in range(m)]
        var_nbrs = [[] for _ in range(n)]
        chk_nbr_pos = [[] for _ in range(m)]
        var_nbr_pos = [[] for _ in range(n)]
        for i in range(m):
            for j in range(n):
                if self.pcm[i, j]:
                    chk_nbr_pos[i].append(len(var_nbrs[j]))
                    var_nbr_pos[j].append(len(chk_nbrs[i]))
                    chk_nbrs[i].append(j)
                    var_nbrs[j].append(i)
        self.chk_nbrs = chk_nbrs
        self.var_nbrs = var_nbrs
        self.chk_nbr_pos = chk_nbr_pos
        self.var_nbr_pos = var_nbr_pos

    @abstractmethod
    def decode(self, syndrome: Bit1DArray) -> Bit1DArray:
        """Decode a syndrome vector.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector, shape=(num_chks,), dtype=uint8.

        Returns
        -------
        ndarray
            Estimated error vector, shape=(num_vars,), dtype=uint8.
        """
        ...

    @abstractmethod
    def decode_batch(self, syndrome_batch: Bit2DArray) -> Bit2DArray:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.

        Returns
        -------
        ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.
        """
        ...

    def decode_detailed(self, syndrome: Bit1DArray, **kwargs) -> Bit1DArray:
        """
        Decode a syndrome vector with detailed diagnostics. Unless overridden,
        this method simply calls `decode` and returns the result.
        """
        return self.decode(syndrome)

    def decode_batch_detailed(self, syndrome_batch: Bit2DArray, **kwargs) -> Bit2DArray:
        """
        Decode a batch of syndrome vectors with detailed diagnostics. Unless overridden,
        this method simply calls `decode_batch` and returns the result.
        """
        return self.decode_batch(syndrome_batch)
