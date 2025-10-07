from .qecdec import BPDecoder as BPDecoder_Rust
from .qecdec import DMemBPDecoder as DMemBPDecoder_Rust
from .qecdec import DMemOffNormBPDecoder as DMemOffNormBPDecoder_Rust
from .utils import build_tanner_graph
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

        self.pcm = pcm.astype(np.uint8)
        self.prior = prior.astype(np.float64)

        # self.chk_nbrs[i] = list of all VNs connected to CN i, sorted in increasing order.
        # self.var_nbrs[j] = list of all CNs connected to VN j, sorted in increasing order.
        # self.chk_nbr_pos[i][k] = position of CN i in the list of neighbors of the VN self.chk_nbrs[i][k].
        #       i.e., self.var_nbrs[self.chk_nbrs[i][k]][self.chk_nbr_pos[i][k]] = i.
        # self.var_nbr_pos[j][k] = position of VN j in the list of neighbors of the CN self.var_nbrs[j][k].
        #       i.e., self.chk_nbrs[self.var_nbrs[j][k]][self.var_nbr_pos[j][k]] = j.
        self.chk_nbrs, self.var_nbrs, self.chk_nbr_pos, self.var_nbr_pos = \
            build_tanner_graph(pcm)

    @property
    def num_checks(self) -> int:
        """Number of check nodes."""
        return self.pcm.shape[0]

    @property
    def num_variables(self) -> int:
        """Number of variable nodes."""
        return self.pcm.shape[1]

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Vector of syndrome bits ∈ {0,1}, shape=(#checks,).

        Returns
        -------
            ehat : ndarray
                Decoded error vector ∈ {0,1}, shape=(#variables,).
        """
        raise NotImplementedError

    def decode_batch(self, syndrome_batch: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome_batch : ndarray
                Array of syndrome vectors ∈ {0,1}, shape=(batch_size, #checks).

        Returns
        -------
            ehat_batch : ndarray
                Array of decoded error vectors ∈ {0,1}, shape=(batch_size, #variables).
        """
        raise NotImplementedError


class MWPMDecoder(Decoder):
    """Minimum Weight Perfect Matching decoder. This class is a wrapper for the pymatching library.
    """

    def __init__(self, pcm: np.ndarray, prior: np.ndarray):
        super().__init__(pcm, prior)

        self.decoder = pymatching.Matching.from_check_matrix(
            self.pcm, weights=np.log((1 - self.prior) / self.prior))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.decoder = pymatching.Matching.from_check_matrix(
            self.pcm, weights=np.log((1 - self.prior) / self.prior))

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        assert isinstance(syndrome, np.ndarray)
        assert syndrome.ndim == 1
        assert syndrome.shape[0] == self.num_checks

        return self.decoder.decode(syndrome)

    def decode_batch(self, syndrome_batch: np.ndarray) -> np.ndarray:
        assert isinstance(syndrome_batch, np.ndarray)
        assert syndrome_batch.ndim == 2
        assert syndrome_batch.shape[1] == self.num_checks

        return self.decoder.decode_batch(syndrome_batch)


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
                Parity check matrix ∈ {0,1}, shape=(#checks, #variables).

            prior : ndarray
                Prior error probabilities for each bit, shape=(#variables,).

            max_iter : int
                Max number of BP iterations.

            scaling_factor : float or None
                Scaling factor (a.k.a. normalization factor) for the BP messages. If None, 
                no scaling is applied.
        """
        super().__init__(pcm, prior)
        self.max_iter = max_iter
        self.scaling_factor = scaling_factor

        self.decoder = BPDecoder_Rust(
            self.pcm, self.prior, max_iter=max_iter, scaling_factor=scaling_factor)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.decoder = BPDecoder_Rust(
            self.pcm, self.prior, max_iter=self.max_iter, scaling_factor=self.scaling_factor)

    def decode(self, syndrome: np.ndarray, record_llr_history: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Vector of syndrome bits ∈ {0,1}, shape=(#checks,).

            record_llr_history : bool
                If True, record the history of LLRs for each BP iteration.

        Returns
        -------
            ehat : ndarray
                Decoded error vector ∈ {0,1}, shape=(#variables,).
        """
        assert isinstance(syndrome, np.ndarray)
        assert syndrome.ndim == 1
        assert syndrome.shape[0] == self.num_checks

        return self.decoder.decode(syndrome.astype(np.uint8), record_llr_history)

    def decode_batch(self, syndrome_batch: np.ndarray) -> np.ndarray:
        assert isinstance(syndrome_batch, np.ndarray)
        assert syndrome_batch.ndim == 2
        assert syndrome_batch.shape[1] == self.num_checks

        return self.decoder.decode_batch(syndrome_batch.astype(np.uint8))

    def get_llr_history(self) -> np.ndarray:
        return self.decoder.get_llr_history()


class DMemBPDecoder(Decoder):
    """Disordered Memory min-sum Belief Propagation decoder. This class is a wrapper for the Rust implementation.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        gamma: np.ndarray,
        max_iter: int,
        scaling_factor: float | None = None,
    ):
        """
        Parameters
        ----------
            pcm : ndarray
                Parity check matrix ∈ {0,1}, shape=(#checks, #variables).

            prior : ndarray
                Prior error probabilities for each bit, shape=(#variables,).

            gamma : ndarray
                Memory strength for each variable node, shape=(#variables,).

            max_iter : int
                Max number of BP iterations.

            scaling_factor : float or None
                Scaling factor (a.k.a. normalization factor) for the BP messages. If None, 
                no scaling is applied.
        """
        super().__init__(pcm, prior)

        assert gamma.shape == (self.num_variables,)
        self.gamma = gamma
        self.max_iter = max_iter
        self.scaling_factor = scaling_factor

        self.decoder = DMemBPDecoder_Rust(
            self.pcm, self.prior, gamma=gamma, max_iter=max_iter, scaling_factor=scaling_factor)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.decoder = DMemBPDecoder_Rust(
            self.pcm, self.prior, gamma=self.gamma, max_iter=self.max_iter, scaling_factor=self.scaling_factor)

    def decode(self, syndrome: np.ndarray, record_llr_history: bool = False) -> np.ndarray:
        assert isinstance(syndrome, np.ndarray)
        assert syndrome.ndim == 1
        assert syndrome.shape[0] == self.num_checks

        return self.decoder.decode(syndrome.astype(np.uint8), record_llr_history)

    def decode_batch(
        self,
        syndrome_batch: np.ndarray,
        return_num_iters: bool = False
    ) -> np.ndarray:
        assert isinstance(syndrome_batch, np.ndarray)
        assert syndrome_batch.ndim == 2
        assert syndrome_batch.shape[1] == self.num_checks

        ehat_batch = self.decoder.decode_batch(
            syndrome_batch.astype(np.uint8), return_num_iters)
        if return_num_iters:
            return ehat_batch, self.decoder.get_batch_num_iters()
        else:
            return ehat_batch

    def get_llr_history(self) -> np.ndarray:
        return self.decoder.get_llr_history()


class DMemOffNormBPDecoder(Decoder):
    """Disordered Memory Offset Normalized min-sum Belief Propagation decoder. This class is a wrapper for the Rust implementation.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        gamma: np.ndarray,
        offset: list[list[float]] | float = 0.0,
        nf: list[list[float]] | float = 1.0,
        max_iter: int,
    ):
        """
        Parameters
        ----------
            pcm : ndarray
                Parity check matrix ∈ {0,1}, shape=(#checks, #variables).

            prior : ndarray
                Prior error probabilities for each bit, shape=(#variables,).

            gamma : ndarray
                Memory strength for each variable node, shape=(#variables,).

            offset : list[list[float]]
                Offset parameters. `offset[i][k]` is the offset parameter for the edge connecting CN `i` to its `k`-th VN neighbor. 
                If a float is provided, the same value is used for all offset parameters. Default is 0.0, meaning no offset.

            nf : list[list[float]]
                Normalization factors. `nf[i][k]` is the normalization factor for the edge connecting CN `i` to its `k`-th VN neighbor. 
                If a float is provided, the same value is used for all normalization factors. Default is 1.0, meaning no normalization.

            max_iter : int
                Max number of BP iterations.
        """
        super().__init__(pcm, prior)

        assert gamma.shape == (self.num_variables,)
        self.gamma = gamma

        if isinstance(offset, list):
            assert len(offset) == self.num_checks
            assert all(isinstance(x, list) for x in offset)
            assert all(len(offset[i]) == len(self.chk_nbrs[i])
                       for i in range(self.num_checks))
        elif isinstance(offset, (float, int)):
            offset = [[offset for _ in range(len(self.chk_nbrs[i]))]
                      for i in range(self.num_checks)]
        else:
            raise ValueError("Invalid data type for `offset`")
        self.offset = offset

        if isinstance(nf, list):
            assert len(nf) == self.num_checks
            assert all(isinstance(x, list) for x in nf)
            assert all(len(nf[i]) == len(self.chk_nbrs[i])
                       for i in range(self.num_checks))
        elif isinstance(nf, (float, int)):
            nf = [[nf for _ in range(len(self.chk_nbrs[i]))]
                  for i in range(self.num_checks)]
        else:
            raise ValueError("Invalid data type for `nf`")
        self.nf = nf

        self.max_iter = max_iter

        self.decoder = DMemOffNormBPDecoder_Rust(
            self.pcm, self.prior, gamma=gamma, offset=offset, nf=nf, max_iter=max_iter)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["decoder"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.decoder = DMemOffNormBPDecoder_Rust(
            self.pcm, self.prior, gamma=self.gamma, offset=self.offset, nf=self.nf, max_iter=self.max_iter)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        assert isinstance(syndrome, np.ndarray)
        assert syndrome.ndim == 1
        assert syndrome.shape[0] == self.num_checks

        return self.decoder.decode(syndrome.astype(np.uint8))

    def decode_batch(self, syndrome_batch: np.ndarray) -> np.ndarray:
        assert isinstance(syndrome_batch, np.ndarray)
        assert syndrome_batch.ndim == 2
        assert syndrome_batch.shape[1] == self.num_checks

        return self.decoder.decode_batch(syndrome_batch.astype(np.uint8))


__all__ = [
    "MWPMDecoder",
    "BPDecoder",
    "DMemBPDecoder",
    "DMemOffNormBPDecoder",
]
