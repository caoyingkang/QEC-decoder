from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cached_property
from typing import Any, ClassVar

import numpy as np

from ..types import Bit1DArray, Bit2DArray, Float1DArray


class Decoder(ABC):
    """Abstract base class for decoders."""

    registry: ClassVar[dict[str, type["Decoder"]]] = {}

    def __init_subclass__(cls, registry_name: str | None = None) -> None:
        # Only register subclasses that set `registry_name`.
        if registry_name is not None:
            if registry_name in Decoder.registry:
                raise ValueError(
                    f"Decoder registry_name {registry_name!r} is already assigned."
                )
            Decoder.registry[registry_name] = cls

    def __init__(self, pcm: Bit2DArray, prior: Float1DArray):
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), uint8 ∈ {0,1}.
            Each row (check) must have at least two nonzero entries; each column
            (variable) must have at least one nonzero entry.
        prior : ndarray
            Prior error probabilities, shape=(num_vars,), float64 ∈ (0,0.5).
        """
        assert isinstance(pcm, np.ndarray) and pcm.ndim == 2
        assert isinstance(prior, np.ndarray) and prior.ndim == 1
        assert pcm.shape[1] == prior.shape[0]
        assert np.all((pcm == 0) | (pcm == 1))
        assert np.all((prior > 0) & (prior < 0.5))
        if not np.all(pcm.sum(axis=1) >= 2):
            raise ValueError("Each row (check) must have at least two nonzero entries.")
        if not np.all(pcm.sum(axis=0) >= 1):
            raise ValueError(
                "Each column (variable) must have at least one nonzero entry."
            )
        self.pcm = pcm
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
    def decode_batch(
        self, syndrome_batch: Bit2DArray, *, parallel: bool = False
    ) -> Bit2DArray:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.
        parallel : bool
            Whether to use multithreaded decoding.

        Returns
        -------
        ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.
        """
        ...

    def decode_detailed(self, syndrome: Bit1DArray, **kwargs) -> Any:
        """
        Decode a syndrome vector with detailed diagnostics. Unless overridden,
        this method simply calls `decode` and returns the result.
        """
        return self.decode(syndrome)

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray, *, parallel: bool = False, **kwargs
    ) -> Any:
        """
        Decode a batch of syndrome vectors with detailed diagnostics. Unless overridden,
        this method simply calls `decode_batch` and returns the result.
        """
        return self.decode_batch(syndrome_batch, parallel=parallel)


class IterativeDecoder(Decoder):
    """Abstract base class for iterative decoders."""

    registry: ClassVar[dict[str, type["IterativeDecoder"]]] = {}

    def __init_subclass__(cls, registry_name: str | None = None) -> None:
        super().__init_subclass__(registry_name)

        # Only register subclasses that set `registry_name`.
        if registry_name is not None:
            if registry_name in IterativeDecoder.registry:
                raise ValueError(
                    f"IterativeDecoder registry_name {registry_name!r} is already assigned."
                )
            IterativeDecoder.registry[registry_name] = cls

    def __init__(self, pcm: Bit2DArray, prior: Float1DArray, *, max_iter: int):
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
            Max number of iterations.
        """
        super().__init__(pcm, prior)

        assert max_iter > 0
        self.max_iter = max_iter

    @classmethod
    def max_iter_from_params(cls, params: Mapping[str, Any]) -> int:
        """Resolve the max iteration count from a decoder-params dict.

        The default looks up ``params["max_iter"]``. Subclasses whose iteration
        budget is composed from multiple params (e.g. relay-style decoders with
        ``pre_iter`` and ``num_relays * max_iter_per_relay``) should override
        this and call it from their ``__init__``.
        """
        return params["max_iter"]


# Module-level aliases for the class-attribute registries. These point to the
# same underlying objects, so updates from `__init_subclass__` flow through to
# `from qecdec.decoders import DECODERS_REGISTRY, ITERATIVE_DECODERS_REGISTRY` callers.
DECODERS_REGISTRY = Decoder.registry
ITERATIVE_DECODERS_REGISTRY = IterativeDecoder.registry
