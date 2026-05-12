"""Type stubs for the qecdec Rust extension (PyO3)."""

from __future__ import annotations

from typing import Optional, TypeAlias

from .types import (
    Bit1DArray,
    Bit2DArray,
    Bool1DArray,
    Float1DArray,
    Float2DArray,
    Int1DArray,
    Int2DArray,
)

DecodeDetailedResult: TypeAlias = tuple[
    Bit1DArray,  # ehat
    bool,  # converged
    int,  # num_iter
    Optional[Float2DArray],  # llr_hist
]

DecodeBatchDetailedResult: TypeAlias = tuple[
    Bit2DArray,  # ehat_batch
    Bool1DArray,  # converged_mask
    Int1DArray,  # decoding_iters
]

class BPDecoderRust:
    """Min-sum belief propagation decoder (Rust implementation)."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        norm: Optional[float] = None,
        max_iter: int,
    ) -> None:
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
        prior : ndarray
            Prior error probabilities.
        norm : float or None
            Message normalization factor; `None` means no normalization.
        max_iter : int
            Max number of BP iterations.
        """
        ...

    def decode_detailed(
        self, syndrome: Bit1DArray, *, record_llr_history: bool
    ) -> tuple[Bit1DArray, bool, int, Optional[Float2DArray]]:
        """Decode a syndrome vector.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector.
        record_llr_history : bool
            Whether to return the history of posterior LLR values.

        Returns
        -------
        ehat : ndarray
            Estimated error vector.
        converged : bool
            Whether the decoder converged (i.e. the syndrome was satisfied).
        num_iter : int
            The number of BP iterations actually run.
        llr_hist : ndarray or None
            The history of posterior LLR values.
        """
        ...

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray, *, parallel: bool
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Batch of syndrome vectors.
        parallel : bool
            Whether to use multithreaded decoding.

        Returns
        -------
        ehat_batch : ndarray
            Batch of estimated error vectors.
        converged_mask : ndarray
            Whether the decoder converged in each shot.
        decoding_iters : ndarray
            Number of BP iterations actually run in each shot.
        """
        ...

class DMemBPDecoderRust:
    """Disordered-memory min-sum BP decoder (Rust implementation)."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        gamma: Float1DArray,
        norm: Optional[float] = None,
        max_iter: int,
    ) -> None:
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), dtype=uint8.
            Each row (check) must have at least two nonzero entries; each column
            (variable) must have at least one nonzero entry.

        prior : ndarray
            Prior error probabilities, shape=(num_vars,), dtype=float64.

        gamma : ndarray
            Memory strength for each variable node, shape=(num_vars,), dtype=float64.
            Use 0.0 for no memory at a node.

        norm : float or None
            Message normalization factor; `None` means no normalization.

        max_iter : int
            Max number of BP iterations.
        """
        ...

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

    def decode_detailed(
        self,
        syndrome: Bit1DArray,
        *,
        record_llr_history: bool,
    ) -> DecodeDetailedResult:
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
        ...

    def decode_batch(
        self, syndrome_batch: Bit2DArray, *, parallel: bool = False
    ) -> Bit2DArray:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.

        parallel : bool
            Whether to use multithreaded decoding. Default is False.

        Returns
        -------
        ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.
        """
        ...

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray, *, parallel: bool = False
    ) -> DecodeBatchDetailedResult:
        """Decode a batch of syndrome vectors with detailed diagnostics.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.

        parallel : bool
            Whether to use multithreaded decoding. Default is False.

        Returns
        -------
        ehat_batch : ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.

        converged_mask : ndarray
            Whether the decoder converged in each shot, shape=(batch_size,), dtype=bool.

        decoding_iters : ndarray
            Number of BP iterations actually run in each shot, shape=(batch_size,), dtype=int64.
        """
        ...

class DMemOffsetBPDecoderRust:
    """Disordered-memory, offset-normalized min-sum BP decoder (Rust implementation)."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        gamma: Float1DArray,
        offset: list[list[float]],
        norm: list[list[float]],
        max_iter: int,
    ) -> None:
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
        prior : ndarray
            Prior error probabilities.
        gamma : ndarray
            Per-VN memory strength.
        offset : list[list[float]]
            Offset parameter for each CN-to-VN edge.
        norm : list[list[float]]
            Normalization factor for each CN-to-VN edge.
        max_iter : int
            Max number of BP iterations.
        """
        ...

    def decode_detailed(self, syndrome: Bit1DArray) -> tuple[Bit1DArray, bool, int]:
        """Decode a syndrome vector.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector.

        Returns
        -------
        ehat : ndarray
            Estimated error vector.
        converged : bool
            Whether the decoder converged (i.e. the syndrome was satisfied).
        num_iter : int
            The number of BP iterations actually run.
        """
        ...

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray, *, parallel: bool
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Batch of syndrome vectors.
        parallel : bool
            Whether to use multithreaded decoding.

        Returns
        -------
        ehat_batch : ndarray
            Batch of estimated error vectors.
        converged_mask : ndarray
            Whether the decoder converged in each shot.
        decoding_iters : ndarray
            Number of BP iterations actually run in each shot.
        """
        ...

class EnsSerialBPDecoderRust:
    """Ensemble of serial-schedule min-sum BP decoders (Rust implementation).

    Runs `ensemble_size` serial-schedule BP decoders with different `vn_order`
    permutations in lockstep (one global iteration at a time, parallel across
    members via Rayon). Once `topk` members have converged, the remaining
    still-active members are stopped, and the most-likely candidate among the
    converged members (lowest prior-LLR weight) is returned.
    """

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        vn_orders: Int2DArray,
        max_iter: int,
        topk: int,
    ) -> None:
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), dtype=uint8.
            Each row (check) must have at least two nonzero entries; each column
            (variable) must have at least one nonzero entry.

        prior : ndarray
            Prior error probabilities, shape=(num_vars,), dtype=float64.

        vn_orders : ndarray
            Stack of variable-node permutations, shape=(ensemble_size, num_vars),
            dtype=int64. Each row must be a permutation of 0..num_vars-1.

        max_iter : int
            Maximum number of global iterations (one iteration = one full pass
            over `vn_order` for each still-active member).

        topk : int
            Number of converged members required before terminating remaining
            still-active members. Must satisfy 1 <= topk <= ensemble_size.
        """
        ...

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
        ...

    def decode_batch(self, syndrome_batch: Bit2DArray) -> Bit2DArray:
        """Decode a batch of syndrome vectors. Outer loop is sequential; ensemble
        parallelism happens within each syndrome via Rayon.

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
            Whether the ensemble produced a converged candidate for each shot,
            shape=(batch_size,), dtype=bool.

        decoding_iters : ndarray
            Number of global iterations actually run in each shot,
            shape=(batch_size,), dtype=int64.
        """
        ...

class SerialBPDecoderRust:
    """Serial-schedule min-sum belief propagation decoder (Rust implementation)."""

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        vn_order: Int1DArray,
        max_iter: int,
    ) -> None:
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), dtype=uint8.
            Each row (check) must have at least two nonzero entries; each column
            (variable) must have at least one nonzero entry.

        prior : ndarray
            Prior error probabilities, shape=(num_vars,), dtype=float64.

        vn_order : ndarray
            Permutation of variable nodes, shape=(num_vars,), dtype=int64.

        max_iter : int
            Maximum number of iterations (one iteration = one full pass over
            `vn_order`).
        """
        ...

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

    def decode_detailed(
        self,
        syndrome: Bit1DArray,
        *,
        record_llr_history: bool,
    ) -> DecodeDetailedResult:
        """Decode a syndrome vector with detailed diagnostics.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector, shape=(num_chks,), dtype=uint8.

        record_llr_history : bool
            Whether to return the history of posterior LLR values
            (one snapshot per iteration, taken at the end of each iteration).

        Returns
        -------
        ehat : ndarray
            Estimated error vector, shape=(num_vars,), dtype=uint8.

        converged : bool
            Whether the decoder converged (i.e. the syndrome was satisfied).

        num_iter : int
            The number of iterations actually run.

        llr_hist : ndarray or None
            If `record_llr_history` is True: posterior LLR values at the end of
            each iteration, shape=(num_iter, num_vars), dtype=float64;
            otherwise, `None`.
        """
        ...

    def decode_batch(
        self, syndrome_batch: Bit2DArray, *, parallel: bool = False
    ) -> Bit2DArray:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.

        parallel : bool
            Whether to use multithreaded decoding. Default is False.

        Returns
        -------
        ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.
        """
        ...

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray, *, parallel: bool = False
    ) -> DecodeBatchDetailedResult:
        """Decode a batch of syndrome vectors with detailed diagnostics.

        Parameters
        ----------
        syndrome_batch : ndarray
            Syndrome vectors, shape=(batch_size, num_chks), dtype=uint8.

        parallel : bool
            Whether to use multithreaded decoding. Default is False.

        Returns
        -------
        ehat_batch : ndarray
            Estimated error vectors, shape=(batch_size, num_vars), dtype=uint8.

        converged_mask : ndarray
            Whether the decoder converged in each shot, shape=(batch_size,), dtype=bool.

        decoding_iters : ndarray
            Number of iterations actually run in each shot, shape=(batch_size,), dtype=int64.
        """
        ...

class UnionFindDecoderRust:
    """Union-Find decoder (Rust implementation)."""

    def __init__(self, pcm: Bit2DArray) -> None:
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix, shape=(num_chks, num_vars), dtype=uint8.
            Each row (check) must have at least two nonzero entries; each column
            (variable) must have at least one and at most two nonzero entries.
        """
        ...

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

__all__ = [
    "BPDecoderRust",
    "DMemBPDecoderRust",
    "DMemOffsetBPDecoderRust",
    "EnsSerialBPDecoderRust",
    "SerialBPDecoderRust",
    "UnionFindDecoderRust",
]
