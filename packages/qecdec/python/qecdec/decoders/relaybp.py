from typing import Optional, Union

import numpy as np

from .base import IterativeDecoder
from ..qecdec import RelayBPDecoderRust
from ..types import (
    Bit1DArray,
    Bit2DArray,
    Bool1DArray,
    Float1DArray,
    Int1DArray,
)


class RelayBPDecoder(IterativeDecoder):
    """RelayBP decoder."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        gamma0: Union[Float1DArray, float],
        gamma_dist_interval: tuple[float, float],
        num_relays: int,
        pre_iter: int,
        max_iter_per_relay: int,
        stop_nconv: int,
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
        gamma0 : ndarray or float
            Per-variable memory strength for the first DMemBP stage,
            shape=(num_vars,), float64.
        gamma_dist_interval : tuple[float, float]
            (low, high) range for sampling per-variable gamma vectors uniformly at each
            relay stage. Must be tuned per decoding graph.
        num_relays : int
            Number of DMemBP relays beyond the first stage.
        pre_iter : int
            Max number of iterations for the first DMemBP stage.
        max_iter_per_relay : int
            Max number of iterations per relay stage.
        stop_nconv : int
            Stop after collecting this many converged candidates. The returned error
            is the candidate with the smallest LLR weight. Must satisfy
            ``1 <= stop_nconv <= num_relays + 1``.
        """
        max_iter = pre_iter + num_relays * max_iter_per_relay
        super().__init__(max_iter, pcm, prior)

        if isinstance(gamma0, (float, int)):
            gamma0 = np.full(self.num_vars, gamma0)
        assert isinstance(gamma0, np.ndarray) and gamma0.shape == (self.num_vars,)
        if stop_nconv < 1 or stop_nconv > num_relays + 1:
            raise ValueError(
                "stop_nconv must satisfy 1 <= stop_nconv <= num_relays + 1"
            )
        if gamma_dist_interval[0] > gamma_dist_interval[1]:
            raise ValueError("gamma_dist_interval must have low <= high")

        self.gamma0 = np.asarray(gamma0, dtype=np.float64)
        self.gamma_dist_interval: tuple[float, float] = tuple(gamma_dist_interval)
        self.num_relays = num_relays
        self.pre_iter = pre_iter
        self.max_iter_per_relay = max_iter_per_relay
        self.stop_nconv = stop_nconv

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> RelayBPDecoderRust:
        return RelayBPDecoderRust(
            self.pcm,
            self.prior,
            gamma0=self.gamma0,
            gamma_dist_interval=self.gamma_dist_interval,
            num_relays=self.num_relays,
            pre_iter=self.pre_iter,
            max_iter_per_relay=self.max_iter_per_relay,
            stop_nconv=self.stop_nconv,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._decoder = self._build_decoder()

    def decode(self, syndrome: Bit1DArray, *, seed: Optional[int] = None) -> Bit1DArray:
        """Decode a syndrome vector.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector, shape=(num_chks,), dtype=uint8.
        seed : int or None
            Optional RNG seed for reproducibility. None → OS entropy.

        Returns
        -------
        ehat : ndarray
            Estimated error vector, shape=(num_vars,), dtype=uint8.
        """
        ehat, _, _ = self._decoder.decode_detailed(syndrome, seed=seed)
        return ehat

    def decode_batch(
        self,
        syndrome_batch: Bit2DArray,
        *,
        parallel: bool = False,
        seed: Optional[int] = None,
    ) -> Bit2DArray:
        """Decode a batch of syndromes.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.
        parallel : bool
            Whether to use multithreaded decoding.
        seed : int or None
            Optional RNG seed for reproducibility.

        Returns
        -------
        ehat_batch : ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.
        """
        ehat_batch, _, _ = self._decoder.decode_batch_detailed(
            syndrome_batch, parallel=parallel, seed=seed
        )
        return ehat_batch

    def decode_detailed(
        self, syndrome: Bit1DArray, *, seed: Optional[int] = None
    ) -> tuple[Bit1DArray, bool, int]:
        """Decode a syndrome vector with detailed diagnostics.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector, shape=(num_chks,), dtype=uint8.
        seed : int or None
            Optional RNG seed for reproducibility. None → OS entropy.

        Returns
        -------
        ehat : ndarray
            Estimated error vector, shape=(num_vars,), dtype=uint8.
        converged : bool
            Whether at least one converged candidate was found.
        num_iter : int
            Total BP iterations summed across all stages run.
        """
        return self._decoder.decode_detailed(syndrome, seed=seed)

    def decode_batch_detailed(
        self,
        syndrome_batch: Bit2DArray,
        *,
        parallel: bool = False,
        seed: Optional[int] = None,
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndromes with detailed diagnostics.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.
        parallel : bool
            Whether to use multithreaded decoding.
        seed : int or None
            Optional master RNG seed for reproducibility. Each shot's RNG stream
            is derived independently from this master seed. None → OS entropy.

        Returns
        -------
        ehat_batch : ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.
        converged_mask : ndarray
            Whether each shot found at least one converged candidate,
            shape=(batch_size,), dtype=bool.
        decoding_iters : ndarray
            Total BP iterations per shot, shape=(batch_size,), dtype=int64.
        """
        return self._decoder.decode_batch_detailed(
            syndrome_batch, parallel=parallel, seed=seed
        )
