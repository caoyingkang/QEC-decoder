from typing import Literal

import numpy as np
import ldpc

from .base import Decoder
from ..types import (
    Bit1DArray,
    Bit2DArray,
    Float1DArray,
)


class BPOSDDecoder(Decoder):
    """(Min-sum)BP+OSD decoder. This class is a wrapper for the ldpc library."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        max_bp_iter: int,
        osd_method: Literal["OSD_0", "OSD_E", "OSD_CS"] = "OSD_CS",
        osd_order: int = 0,
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

        max_bp_iter : int
            Max number of BP iterations.

        osd_method : Literal["OSD_0", "OSD_E", "OSD_CS"]
            OSD method: "OSD_0" for zero-order OSD, "OSD_E" for exhaustive OSD,
            "OSD_CS" for combination-sweep OSD.

        osd_order : int
            OSD order.
        """
        super().__init__(pcm, prior)
        self.max_bp_iter = max_bp_iter
        self.osd_method = osd_method
        self.osd_order = osd_order

        self._decoder = self._build_decoder()

    def _build_decoder(self) -> ldpc.BpOsdDecoder:
        return ldpc.BpOsdDecoder(
            pcm=self.pcm,
            error_channel=self.prior.tolist(),
            max_iter=self.max_bp_iter,
            bp_method="minimum_sum",
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._decoder = self._build_decoder()

    def decode(self, syndrome: Bit1DArray) -> Bit1DArray:
        ehat = self._decoder.decode(syndrome)
        return ehat.astype(np.uint8)

    def decode_batch(self, syndrome_batch: Bit2DArray) -> Bit2DArray:
        batch_size = syndrome_batch.shape[0]
        ehat_batch = np.zeros((batch_size, self.num_vars), dtype=np.uint8)
        for i in range(batch_size):
            ehat_batch[i, :] = self._decoder.decode(syndrome_batch[i, :])
        return ehat_batch
