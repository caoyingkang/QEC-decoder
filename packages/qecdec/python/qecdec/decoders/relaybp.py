import numpy as np
from relay_bp import RelayDecoderF64

from .base import IterativeDecoder
from ..types import (
    Bit1DArray,
    Bit2DArray,
    Bool1DArray,
    Float1DArray,
    Int1DArray,
)


class RelayBPDecoder(IterativeDecoder):
    """RelayBP decoder — ensemble BP with disordered memory and relaying.

    Wraps `relay_bp.RelayDecoderF64` from the `relay-bp` package.
    See https://github.com/trmue/relay for details.
    """

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        gamma0: float,
        gamma_dist_interval: tuple[float, float],
        num_relays: int,
        pre_iter: int,
        max_iter_per_relay: int,
        stop_nconv: int,
        num_indep_decoders: int = 1,
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

        gamma0 : float
            Memory parameter for the first MemBP instance.

        gamma_dist_interval : tuple[float, float]
            The uniform distribution for random memory weights used in DMemBP relays.
            Must be tuned per decoding graph.

        num_relays : int
            Number of DMemBP relays (beyond the first MemBP instance).

        pre_iter : int
            Number of iterations for the first MemBP instance.

        max_iter_per_relay : int
            Max number of iterations per DMemBP relay.

        stop_nconv : int
            How many solutions to find before terminating. Must be less than or equal to num_relays + 1.

        num_indep_decoders : int
            Number of independent decoding runs per syndrome. The earliest output
            (fewest iterations) across all runs is returned. Defaults to 1.
        """
        super().__init__(pre_iter + num_relays * max_iter_per_relay, pcm, prior)

        if stop_nconv > num_relays + 1:
            raise ValueError("stop_nconv must be less than or equal to num_relays + 1")
        if num_indep_decoders < 1:
            raise ValueError("num_indep_decoders must be at least 1")

        self.gamma0 = gamma0
        self.gamma_dist_interval: tuple[float, float] = tuple(gamma_dist_interval)
        self.num_relays = num_relays
        self.pre_iter = pre_iter
        self.max_iter_per_relay = max_iter_per_relay
        self.stop_nconv = stop_nconv
        self.num_indep_decoders = num_indep_decoders

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> RelayDecoderF64:
        return RelayDecoderF64(
            self.pcm,
            self.prior,
            gamma0=self.gamma0,
            pre_iter=self.pre_iter,
            num_sets=self.num_relays,
            set_max_iter=self.max_iter_per_relay,
            gamma_dist_interval=self.gamma_dist_interval,
            stop_nconv=self.stop_nconv,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._decoder = self._build_decoder()

    def _select_earliest_result(self, results):
        """Return the earliest result: converged with fewest iterations, or first if none converged."""
        earliest = None
        for result in results:
            if not result.success:
                continue
            if earliest is None or result.iterations < earliest.iterations:
                earliest = result
        if earliest is None:
            return results[0]
        return earliest

    def decode(self, syndrome: Bit1DArray) -> Bit1DArray:
        if self.num_indep_decoders == 1:
            return self._decoder.decode(syndrome)
        results = [
            self._decoder.decode_detailed(syndrome)
            for _ in range(self.num_indep_decoders)
        ]
        earliest = self._select_earliest_result(results)
        return earliest.decoding

    def decode_batch(self, syndrome_batch: Bit2DArray) -> Bit2DArray:
        if self.num_indep_decoders == 1:
            return self._decoder.decode_batch(syndrome_batch)
        batch_size = syndrome_batch.shape[0]
        ehat_batch = np.empty((batch_size, self.num_vars), dtype=np.uint8)
        for i in range(batch_size):
            ehat_batch[i] = self.decode(syndrome_batch[i])
        return ehat_batch

    def decode_detailed(self, syndrome: Bit1DArray):
        """Decode a syndrome vector with detailed diagnostics.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector, shape=(num_chks,), dtype=uint8.

        Returns
        -------
        relay_bp.DecodeResult
            See https://github.com/trmue/relay/blob/main/examples/GettingStarted.ipynb for details.
        """
        if self.num_indep_decoders == 1:
            return self._decoder.decode_detailed(syndrome)
        results = [
            self._decoder.decode_detailed(syndrome)
            for _ in range(self.num_indep_decoders)
        ]
        earliest = self._select_earliest_result(results)
        return earliest

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
        batch_size = syndrome_batch.shape[0]
        ehat_batch = np.empty((batch_size, self.num_vars), dtype=np.uint8)
        converged_mask = np.empty(batch_size, dtype=np.bool_)
        decoding_iters = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            result = self.decode_detailed(syndrome_batch[i])
            ehat_batch[i] = result.decoding
            converged_mask[i] = result.success
            decoding_iters[i] = result.iterations
        return ehat_batch, converged_mask, decoding_iters
