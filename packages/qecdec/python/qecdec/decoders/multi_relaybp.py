from typing import Optional

import numpy as np

from .base import IterativeDecoder
from ..qecdec import MultiRelayBPDecoderRust
from ..types import (
    Bit1DArray,
    Bit2DArray,
    Bool1DArray,
    Float1DArray,
    Int1DArray,
)


class MultiRelayBPDecoder(IterativeDecoder):
    """MultiRelayBP decoder — `num_chains` parallel RelayBP chains sharing a first stage.

    All chains share the deterministic first DMemBP stage driven by `gamma0`. After that
    stage, each chain forks into its own random-relay sequence (each chain has an
    independent RNG stream derived from the per-call `seed`). The chain that collects
    its `stop_nconv`-th converged candidate in the fewest total BP iterations wins
    (ties broken by lowest chain index); within that chain, the min-LLR-weight candidate
    is returned. If no chain converges, the last `ehat` of chain 0 is returned.

    Trades memory and core-count for wall-clock — useful when some chains get stuck while
    others converge quickly.
    """

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        gamma0: Float1DArray,
        gamma_dist_interval: tuple[float, float],
        num_chains: int,
        num_relays: int,
        pre_iter: int,
        max_iter_per_relay: int,
        stop_nconv: int,
    ):
        """
        Parameters
        ----------
        pcm, prior, gamma0, gamma_dist_interval, num_relays, pre_iter,
        max_iter_per_relay, stop_nconv :
            Same meaning as in `RelayBPDecoder`.

        num_chains : int
            Number of independent chains (must be ≥ 1). All chains share the first
            DMemBP stage; thereafter they run independent random-relay sequences in
            parallel.
        """
        super().__init__(pre_iter + num_relays * max_iter_per_relay, pcm, prior)

        assert isinstance(gamma0, np.ndarray) and gamma0.shape == (self.num_vars,)
        if stop_nconv < 1 or stop_nconv > num_relays + 1:
            raise ValueError(
                "stop_nconv must satisfy 1 <= stop_nconv <= num_relays + 1"
            )
        if num_chains < 1:
            raise ValueError("num_chains must be at least 1")
        if gamma_dist_interval[0] > gamma_dist_interval[1]:
            raise ValueError("gamma_dist_interval must have low <= high")

        self.gamma0 = gamma0
        self.gamma_dist_interval: tuple[float, float] = tuple(gamma_dist_interval)
        self.num_relays = num_relays
        self.pre_iter = pre_iter
        self.max_iter_per_relay = max_iter_per_relay
        self.stop_nconv = stop_nconv
        self.num_chains = num_chains

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> MultiRelayBPDecoderRust:
        return MultiRelayBPDecoderRust(
            self.pcm,
            self.prior,
            gamma0=self.gamma0,
            gamma_dist_interval=self.gamma_dist_interval,
            num_chains=self.num_chains,
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
        ehat, _, _ = self._decoder.decode_detailed(syndrome, seed=seed)
        return ehat

    def decode_batch(
        self,
        syndrome_batch: Bit2DArray,
        *,
        parallel: bool = False,
        seed: Optional[int] = None,
    ) -> Bit2DArray:
        ehat_batch, _, _ = self._decoder.decode_batch_detailed(
            syndrome_batch, parallel=parallel, seed=seed
        )
        return ehat_batch

    def decode_detailed(
        self, syndrome: Bit1DArray, *, seed: Optional[int] = None
    ) -> tuple[Bit1DArray, bool, int]:
        """Decode a syndrome with detailed diagnostics.

        Returns
        -------
        ehat : ndarray
            Winning chain's chosen error estimate, shape=(num_vars,), dtype=uint8.

        converged : bool
            Whether at least one chain found a converged candidate.

        num_iter : int
            Total BP iterations of the winning chain (sum across its stages,
            including the shared first stage).
        """
        return self._decoder.decode_detailed(syndrome, seed=seed)

    def decode_batch_detailed(
        self,
        syndrome_batch: Bit2DArray,
        *,
        parallel: bool = False,
        seed: Optional[int] = None,
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndromes with detailed diagnostics."""
        return self._decoder.decode_batch_detailed(
            syndrome_batch, parallel=parallel, seed=seed
        )
