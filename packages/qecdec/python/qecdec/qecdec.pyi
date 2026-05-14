"""Type stubs for the qecdec Rust extension (PyO3)."""

from __future__ import annotations

from typing import Optional

from .types import (
    Bit1DArray,
    Bit2DArray,
    Bool1DArray,
    Float1DArray,
    Float2DArray,
    Int1DArray,
    Int2DArray,
)

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
            Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
        prior : ndarray
            Prior error probabilities.
        gamma : ndarray
            Per-VN memory strength.
        norm : float or None
            Normalization factor. Default is 1.0, meaning no normalization.
        max_iter : int
            Maximum number of BP iterations.
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
            Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
        prior : ndarray
            Prior error probabilities.
        vn_order : ndarray
            Permutation of variable nodes.
        max_iter : int
            Maximum number of iterations (one iteration = one full pass over `vn_order`).
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
            Whether the decoder converged.
        num_iter : int
            The number of iterations actually run.
        """
        ...

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray, *, parallel: bool
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndrome vectors with detailed diagnostics.

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
            Number of iterations actually run in each shot.
        """
        ...

class EnsSerialBPDecoderRust:
    """Ensemble of serial-schedule min-sum BP decoders (Rust implementation)."""

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
            Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
        prior : ndarray
            Prior error probabilities.
        vn_orders : ndarray
            Ensemble of variable node permutations, shape `(ensemble_size, num_vars)`.
        max_iter : int
            Maximum number of iterations (one iteration = one full pass over `vn_order`).
        topk : int
            Number of converged members required before terminating the remaining members.
            Must satisfy 1 <= topk <= ensemble_size.
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
            Whether at least one ensemble member converged.
        num_iter : int
            The number of iterations actually run.
        """
        ...

    def decode_batch_detailed(
        self, syndrome_batch: Bit2DArray
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Batch of syndrome vectors.

        Returns
        -------
        ehat_batch : ndarray
            Batch of estimated error vectors.
        converged_mask : ndarray
            Whether the decoder converged in each shot.
        decoding_iters : ndarray
            Number of iterations actually run in each shot.
        """
        ...

class RelayBPDecoderRust:
    """RelayBP decoder (Rust implementation).

    Run an initial DMemBP stage followed by up to `num_relays` DMemBP stages with
    per-variable gamma vectors sampled uniformly from `gamma_dist_interval`. Each
    stage inherits posterior LLRs from the previous stage. The returned error is
    the smallest-LLR-weight candidate among the converged stages.

    This is a re-implementation of https://github.com/trmue/relay/tree/main.
    Note that the original version only accepts MemBP as the initial stage.
    """

    def __init__(
        self,
        pcm: Bit2DArray,
        prior: Float1DArray,
        *,
        gamma0: Float1DArray,
        gamma_dist_interval: tuple[float, float],
        num_relays: int,
        pre_iter: int,
        max_iter_per_relay: int,
        stop_nconv: int,
    ) -> None:
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
        prior : ndarray
            Prior error probabilities.
        gamma0 : ndarray
            Per-variable memory strength for the initial DMemBP stage.
        gamma_dist_interval : tuple[float, float]
            (low, high) range for sampling per-variable gamma vectors at each relay stage.
        num_relays : int
            Number of DMemBP relays beyond the initial stage.
        pre_iter : int
            Max number of iterations for the initial DMemBP stage.
        max_iter_per_relay : int
            Max number of iterations per relay stage.
        stop_nconv : int
            Stop after this many converged candidates. Must satisfy
            1 <= stop_nconv <= num_relays + 1.
        """
        ...

    def decode_detailed(
        self, syndrome: Bit1DArray, *, seed: Optional[int] = None
    ) -> tuple[Bit1DArray, bool, int]:
        """Decode a syndrome vector.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector.
        seed : int or None
            Optional RNG seed for reproducibility. None → OS entropy.

        Returns
        -------
        ehat : ndarray
            Estimated error vector.
        converged : bool
            Whether at least one converged candidate was found.
        num_iter : int
            Total BP iterations summed across all stages run.
        """
        ...

    def decode_batch_detailed(
        self,
        syndrome_batch: Bit2DArray,
        *,
        parallel: bool,
        seed: Optional[int] = None,
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Batch of syndrome vectors.
        parallel : bool
            Whether to use multithreaded decoding.
        seed : int or None
            Optional master RNG seed. Each shot derives a child seed from it.

        Returns
        -------
        ehat_batch : ndarray
            Batch of estimated error vectors.
        converged_mask : ndarray
            Whether each shot found at least one converged candidate.
        decoding_iters : ndarray
            Total BP iterations per shot.
        """
        ...

class MultiRelayBPDecoderRust:
    """Multi-chain RelayBP decoder (Rust implementation)."""

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
    ) -> None:
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
        prior : ndarray
            Prior error probabilities.
        gamma0 : ndarray
            Per-variable memory strength for the shared initial DMemBP stage.
        gamma_dist_interval : tuple[float, float]
            (low, high) range for sampling per-variable gamma vectors at each relay stage.
        num_chains : int
            Number of independent chains (≥ 1).
        num_relays : int
            Number of DMemBP relays beyond the initial stage.
        pre_iter : int
            Max number of iterations for the shared initial DMemBP stage.
        max_iter_per_relay : int
            Max number of iterations per relay stage.
        stop_nconv : int
            Stop a chain after this many converged candidates. Must satisfy
            1 <= stop_nconv <= num_relays + 1.
        """
        ...

    def decode_detailed(
        self, syndrome: Bit1DArray, *, seed: Optional[int] = None
    ) -> tuple[Bit1DArray, bool, int]:
        """Decode a syndrome vector.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector.
        seed : int or None
            Optional RNG seed for reproducibility. None → OS entropy.

        Returns
        -------
        ehat : ndarray
            Winning chain's chosen error estimate.
        converged : bool
            Whether at least one chain found a converged candidate.
        num_iter : int
            Total BP iterations of the winning chain.
        """
        ...

    def decode_batch_detailed(
        self,
        syndrome_batch: Bit2DArray,
        *,
        parallel: bool,
        seed: Optional[int] = None,
    ) -> tuple[Bit2DArray, Bool1DArray, Int1DArray]:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Batch of syndrome vectors.
        parallel : bool
            Whether to parallelize at the batch level. Chain-level parallelism
            is always on.
        seed : int or None
            Optional master RNG seed. Each shot derives its own stream from it.

        Returns
        -------
        ehat_batch : ndarray
            Batch of estimated error vectors.
        converged_mask : ndarray
            Whether each shot found at least one converged candidate.
        decoding_iters : ndarray
            Total BP iterations per shot.
        """
        ...

class UnionFindDecoderRust:
    """Union-Find decoder (Rust implementation)."""

    def __init__(self, pcm: Bit2DArray) -> None:
        """
        Parameters
        ----------
        pcm : ndarray
            Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 and ≤2 nonzeros.
        """
        ...

    def decode(self, syndrome: Bit1DArray) -> Bit1DArray:
        """Decode a syndrome vector.

        Parameters
        ----------
        syndrome : ndarray
            Syndrome vector.

        Returns
        -------
        ehat : ndarray
            Estimated error vector.
        """
        ...

    def decode_batch(self, syndrome_batch: Bit2DArray) -> Bit2DArray:
        """Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndrome_batch : ndarray
            Batch of syndrome vectors.

        Returns
        -------
        ehat_batch : ndarray
            Batch of estimated error vectors.
        """
        ...

__all__ = [
    "BPDecoderRust",
    "DMemBPDecoderRust",
    "DMemOffsetBPDecoderRust",
    "EnsSerialBPDecoderRust",
    "MultiRelayBPDecoderRust",
    "RelayBPDecoderRust",
    "SerialBPDecoderRust",
    "UnionFindDecoderRust",
]
