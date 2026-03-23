"""Abstract base class for decoders used by the Monte Carlo benchmark tool."""
from abc import ABC, abstractmethod
from typing import Optional, NamedTuple

from .types import Bool1DArray, Bit2DArray, Int1DArray

class DecodeResult(NamedTuple):
    """Result of decoding a batch of syndromes.

    Attributes
    ----------
    obser_pred : Bit2DArray
        Predicted observables, shape=(batch_size, num_obsers).
    synd_match_mask : Bool1DArray
        Boolean mask, shape=(batch_size,). True when the predicted error pattern
        satisfies the syndrome for that shot.
    decoding_iters : Int1DArray or None
        Number of iterations the decoder ran for each shot, shape=(batch_size,). 
        0 means the syndrome for that shot was all-zero (no decoding at all). 
        `None` for non-iterative decoders.
    """
    obser_pred: Bit2DArray
    synd_match_mask: Bool1DArray
    decoding_iters: Optional[Int1DArray]


class BenchmarkDecoder(ABC):
    """Abstract base class for decoders used by the MC benchmark tool."""

    @abstractmethod
    def decode(self, syndromes: Bit2DArray) -> DecodeResult:
        """Decode a batch of syndromes.

        Parameters
        ----------
        syndromes : Bit2DArray
            Syndrome bits, shape=(batch_size, num_chks).

        Returns
        -------
        DecodeResult
        """
        ...

