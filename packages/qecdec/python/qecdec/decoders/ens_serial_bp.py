from typing import Optional

import numpy as np

from .base import IterativeDecoder
from ..qecdec import EnsSerialBPDecoderRust
from ..types import (
    Bool1DArray,
    Bit1DArray,
    Bit2DArray,
    Int1DArray,
    Int2DArray,
    Float1DArray,
)


class EnsSerialBPDecoder(IterativeDecoder):
    """Ensemble of serial-schedule BP decoders with random `vn_order` permutations.

    For each syndrome, the ensemble runs an iteration-synchronous loop: at each
    global iteration, every member that hasn't yet converged advances one BP
    iteration in parallel (Rayon, GIL released). After each global iteration we
    count converged members; once `topk` have converged, the loop ends and the
    most-likely candidate among the converged members (lowest prior-LLR weight)
    is returned. If `max_iter` is reached before `topk` converge, the best of
    however many converged is returned. If none converged, the first member's
    estimate is returned with `converged=False`.
    """

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        max_iter: int,
        ensemble_size: int,
        topk: int,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), uint8 ∈ {0,1}.

        prior : ndarray
            Prior error probabilities, shape=(num_vars,), float64 ∈ (0, 0.5).

        max_iter : int
            Maximum number of global iterations.

        ensemble_size : int
            Number of SerialBP members in the ensemble.

        topk : int
            Number of converged members required before terminating remaining
            still-active members. Must satisfy 1 <= topk <= ensemble_size.

        seed : int or None
            Seed for the numpy RNG used to generate the per-member `vn_order`
            permutations. None means non-reproducible (fresh entropy each run).
        """
        super().__init__(max_iter, pcm, prior)
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")
        if not (1 <= topk <= ensemble_size):
            raise ValueError(
                f"Require 1 <= topk <= ensemble_size, got {topk=} and {ensemble_size=}"
            )

        self.ensemble_size = ensemble_size
        self.topk = topk

        rng = np.random.default_rng(seed)
        self.vn_orders: Int2DArray = np.stack(
            [
                rng.permutation(self.num_vars).astype(np.int64)
                for _ in range(ensemble_size)
            ],
            axis=0,
        )

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> EnsSerialBPDecoderRust:
        return EnsSerialBPDecoderRust(
            self.pcm,
            self.prior,
            vn_orders=self.vn_orders,
            max_iter=self.max_iter,
            topk=self.topk,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._decoder = self._build_decoder()

    def decode(self, syndrome: Bit1DArray) -> Bit1DArray:
        return self._decoder.decode(syndrome)

    def decode_batch(self, syndrome_batch: Bit2DArray) -> Bit2DArray:
        return self._decoder.decode_batch(syndrome_batch)

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
            Whether at least one ensemble member converged.

        num_iter : int
            Number of global iterations actually run.
        """
        return self._decoder.decode_detailed(syndrome)

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndrome vectors with detailed diagnostics.

        Returns
        -------
        ehat_batch : ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.

        converged_mask : ndarray
            Whether the ensemble produced a converged candidate for each shot,
            shape=(batch_size,), dtype=bool.

        decoding_iters : ndarray
            Number of global iterations actually run in each shot,
            shape=(batch_size,), dtype=int64.
        """
        return self._decoder.decode_batch_detailed(syndrome_batch)
