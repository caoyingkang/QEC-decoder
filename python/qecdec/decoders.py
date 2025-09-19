from .qecdec import BPDecoder as BPDecoder_Rust
import numpy as np
import pymatching


class Decoder:
    """Base class for decoders.
    """

    def __init__(self, pcm: np.ndarray, prior: np.ndarray):
        """
        Parameters
        ----------
            pcm : ndarray
                Parity check matrix ∈ {0,1}, shape=(m,n).

            prior : ndarray
                Prior error probabilities for each bit, shape=(n,).
        """
        assert isinstance(pcm, np.ndarray) and pcm.ndim == 2
        assert isinstance(prior, np.ndarray) and prior.ndim == 1
        assert pcm.shape[1] == prior.shape[0]

        self.m: int = pcm.shape[0]
        self.n: int = pcm.shape[1]
        self.pcm = pcm.astype(np.uint8)
        self.prior = prior.astype(np.float64)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Vector of syndrome bits ∈ {0,1}, shape=(m,).

        Returns
        -------
            ehat : ndarray
                Decoded error vector ∈ {0,1}, shape=(n,).
        """
        raise NotImplementedError

    def decode_batch(self, syndrome_batch: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome_batch : ndarray
                Array of syndrome vectors ∈ {0,1}, shape=(batch_size,m).

        Returns
        -------
            ehat : ndarray
                Array of decoded error vectors ∈ {0,1}, shape=(batch_size,n).
        """
        raise NotImplementedError


class MWPMDecoder(Decoder):
    """Minimum Weight Perfect Matching decoder. This class is a wrapper for the pymatching library.
    """

    def __init__(self, pcm: np.ndarray, prior: np.ndarray):
        super().__init__(pcm, prior)

        self.mwpm = pymatching.Matching.from_check_matrix(
            self.pcm, weights=np.log((1 - self.prior) / self.prior))

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        assert isinstance(syndrome, np.ndarray)
        assert syndrome.ndim == 1
        assert syndrome.shape[0] == self.m

        return self.mwpm.decode(syndrome)

    def decode_batch(self, syndrome_batch: np.ndarray) -> np.ndarray:
        assert isinstance(syndrome_batch, np.ndarray)
        assert syndrome_batch.ndim == 2
        assert syndrome_batch.shape[1] == self.m

        return self.mwpm.decode_batch(syndrome_batch)


class BPDecoder(Decoder):
    """Belief Propagation decoder (min-sum variant). This class is a wrapper for the Rust implementation.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        max_iter: int,
        scaling_factor: float | None = None,
    ):
        """
        Parameters
        ----------
            pcm : ndarray
                Parity check matrix ∈ {0,1}, shape=(m,n).

            prior : ndarray
                Prior error probabilities for each bit, shape=(n,).

            max_iter : int
                Max number of BP iterations.

            scaling_factor : float or None
                Scaling factor (a.k.a. normalization factor) for the BP messages. If None, 
                no scaling is applied.
        """
        super().__init__(pcm, prior)

        self.bp = BPDecoder_Rust(
            self.pcm, self.prior, max_iter=max_iter, scaling_factor=scaling_factor)

    def decode(self, syndrome: np.ndarray, record_llr_history: bool = False) -> np.ndarray:
        assert isinstance(syndrome, np.ndarray)
        assert syndrome.ndim == 1
        assert syndrome.shape[0] == self.m

        return self.bp.decode(syndrome.astype(np.uint8), record_llr_history)

    def decode_batch(self, syndrome_batch: np.ndarray) -> np.ndarray:
        assert isinstance(syndrome_batch, np.ndarray)
        assert syndrome_batch.ndim == 2
        assert syndrome_batch.shape[1] == self.m

        return self.bp.decode_batch(syndrome_batch.astype(np.uint8))

    def get_llr_history(self) -> np.ndarray:
        return self.bp.get_llr_history()
