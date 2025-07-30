import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List


class Decoder:
    """Base class for decoders.
    """

    def __init__(self, H: np.ndarray):
        """
        Parameters
        ----------
            H : ndarray
                Parity check matrix, shape=(m, n), dtype=int, values in {0, 1}. Make sure every column has weight at least 1 
                and every row has weight at least 2.
        """
        assert isinstance(H, np.ndarray)
        m, n = H.shape
        assert H.dtype == int and np.all(np.isin(H, [0, 1]))

        self.H = H
        self.m: int = m
        self.n: int = n

        # get neighboring check nodes of each variable node in the Tanner graph
        # dict: v-node -> list of c-nodes
        self.neighbors_v = {v: np.nonzero(H[:, v] == 1)[0].tolist()
                            for v in range(n)}
        assert all(len(self.neighbors_v[v]) >= 1 for v in range(n)), \
            "Every variable must be involved in at least one check."

        # get neighboring variable nodes of each check node in the Tanner graph
        # dict: c-node -> list of v-nodes
        self.neighbors_c = {c: np.nonzero(H[c] == 1)[0].tolist()
                            for c in range(m)}
        assert all(len(self.neighbors_c[c]) >= 2 for c in range(m)), \
            "Every check must involve at least two variables."

    def decode(self, syndrome: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Syndrome vector, shape=(m,), dtype=int, values in {0, 1}.
            
            verbose : bool
                If True, print decoding information. Default is False.

        Returns
        -------
            ehat : ndarray
                Decoded error vector.
        """
        pass

    def decode_batch(self, syndrome: np.ndarray, progress_bar: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Array of syndrome vectors, shape=(N,m), dtype=int, values in {0, 1}.
            
            progress_bar : bool
                If True, show a progress bar. Default is True.

        Returns
        -------
            ehat : ndarray
                Array of decoded error vectors, shape=(N,n), dtype=int, values in {0, 1}.
        """
        pass

class BPDecoder(Decoder):
    """Belief Propagation decoder for syndrome-based LDPC decoding, min-sum variant.
    """

    def __init__(self, H: np.ndarray, prior: np.ndarray, max_iter: Optional[int] = None, record_history: bool = False):
        """
        Parameters
        ----------
            H : ndarray
                Parity check matrix, shape=(m, n), dtype=int, values in {0, 1}. Make sure every column has weight at least 1 
                and every row has weight at least 2.
            
            prior : ndarray
                Prior error probabilities for each bit, shape=(n,), dtype=float, values in (0, 0.5).
            
            max_iter : int or None
                Max number of BP iterations. If None, defaults to code length n.
            
            record_history : bool
                If True, record marginals and hard decisions at each iteration. Default is False.
        """
        super().__init__(H)

        assert isinstance(prior, np.ndarray)
        assert prior.shape == (self.n,)
        assert prior.dtype == float and 0 < prior.min() and prior.max() < 0.5

        self.prior = prior
        # log-likelihood ratios of prior probabilities
        self.llr_prior = np.log((1 - prior) / prior)

        # set max number of iterations
        if max_iter is None:
            max_iter = self.n
        self.max_iter = max_iter

        # set whether to record history
        self.record_history = record_history

    def decode(self, syndrome: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Syndrome vector, shape=(m,), dtype=int, values in {0, 1}.
            
            verbose : bool
                If True, print convergence information. Default is False.

        Returns
        -------
            ehat : ndarray
                Decoded error vector.
        """
        assert isinstance(syndrome, np.ndarray)
        assert syndrome.shape == (self.m,)
        assert syndrome.dtype == int and np.all(np.isin(syndrome, [0, 1]))

        # convert syndrome from {0,1} to {1,-1} representation
        syndrome_sign = 1 - 2 * syndrome

        # dict: (v, c) -> message from variable node v to check node c
        msg_v_to_c = {}
        # dict: (c, v) -> message from check node c to variable node v
        msg_c_to_v = {}

        # initialization (iteration 0)
        if self.record_history:
            history_marginal = []  # record marginals obtained at each iteration
            history_marginal.append(np.copy(self.llr_prior))
            history_ehat = []  # record hard decisions obtained at each iteration
            history_ehat.append(np.zeros(self.n, dtype=int))
        for v in range(self.n):
            for c in self.neighbors_v[v]:
                msg_v_to_c[(v, c)] = self.llr_prior[v]
        converged = False

        # main BP iterations (iteration 1 to max_iter)
        it = 1
        while it <= self.max_iter:
            # check-to-variable messages
            for c in range(self.m):
                for v_out in self.neighbors_c[c]:
                    in_msgs = [msg_v_to_c[(v, c)]
                               for v in self.neighbors_c[c] if v != v_out]
                    prod_sign = np.prod(np.sign(in_msgs))
                    min_abs = np.min(np.abs(in_msgs))
                    msg_c_to_v[(c, v_out)] = syndrome_sign[c] * \
                        prod_sign * min_abs
            # variable-to-check messages
            for v in range(self.n):
                for c_out in self.neighbors_v[v]:
                    in_msgs = [msg_c_to_v[(c, v)]
                               for c in self.neighbors_v[v] if c != c_out]
                    msg_v_to_c[(v, c_out)] = self.llr_prior[v] + \
                        np.sum(in_msgs)
            # marginals
            marginal = np.copy(self.llr_prior)
            for v in range(self.n):
                marginal[v] += np.sum([msg_c_to_v[(c, v)]
                                      for c in self.neighbors_v[v]])
            # hard decisions
            ehat = np.zeros(self.n, dtype=int)
            ehat[marginal < 0] = 1
            # record history if needed
            if self.record_history:
                history_marginal.append(np.copy(marginal))
                history_ehat.append(np.copy(ehat))
            # check if syndrome is satisfied
            if np.all((self.H @ ehat) % 2 == syndrome):
                converged = True
                break
            it += 1

        if verbose:
            if converged:
                print(f"Decoding converged at iteration {it}.")
            else:
                print("Max iterations reached without convergence.")

        if self.record_history:
            # (i,j) entry is marginal of variable j after iteration i
            self.history_marginal = np.array(history_marginal, dtype=float)
            # (i,j) entry is hard decision of variable j after iteration i
            self.history_ehat = np.array(history_ehat, dtype=int)

        if not converged:
            # defaults to all-zero error vector if decoding fails
            ehat = np.zeros(self.n, dtype=int)

        return ehat

    def decode_batch(self, syndrome: np.ndarray, progress_bar: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Array of syndrome vectors, shape=(N,m), dtype=int, values in {0, 1}.

            progress_bar : bool
                If True, show a progress bar. Default is True.

        Returns
        -------
            ehat : ndarray
                Array of decoded error vectors, shape=(N,n), dtype=int, values in {0, 1}.
        """
        assert isinstance(syndrome, np.ndarray)
        assert syndrome.shape[1] == self.m
        assert syndrome.dtype == int and np.all(np.isin(syndrome, [0, 1]))

        N = syndrome.shape[0]  # batch size

        # naive implementation
        ehat = np.zeros((N, self.n), dtype=int)
        for i in tqdm(range(N), disable=not progress_bar):
            ehat[i] = self.decode(syndrome[i])

        return ehat

class RelayBPDecoder(Decoder):
    """Relay Belief Propagation decoder proposed in the paper
    ["Improved belief propagation is sufficient for real-time decoding of quantum memory." arXiv:2506.01779 (2025)](https://arxiv.org/abs/2506.01779).
    """

    def __init__(self, H: np.ndarray, prior: np.ndarray, num_sol: int, max_leg: int, mem_strength: np.ndarray, max_iter_list: List[int]):
        """
        Parameters
        ----------
            H : ndarray
                Parity check matrix, shape=(m, n), dtype=int, values in {0, 1}. Make sure every column has weight at least 1 
                and every row has weight at least 2.
            
            prior : ndarray
                Prior error probabilities for each bit, shape=(n,), dtype=float, values in (0, 0.5).
            
            num_sol : int
                Number of solutions sought.
            
            max_leg : int
                Max number of relay legs.
            
            mem_strength : ndarray
                Memory strength for each bit and each relay leg, shape=(max_leg, n), dtype=float. As described in the paper, 
                the memory strength can be negative.
            
            max_iter_list : list[int]
                List of max number of iterations for each relay leg, of length max_leg.
        """
        super().__init__(H)

        assert isinstance(prior, np.ndarray)
        assert prior.shape == (self.n,)
        assert prior.dtype == float and 0 < prior.min() and prior.max() < 0.5

        self.prior = prior
        # log-likelihood ratios of prior probabilities
        self.llr_prior = np.log((1 - prior) / prior)

        self.num_sol = num_sol
        self.max_leg = max_leg

        assert isinstance(mem_strength, np.ndarray)
        assert mem_strength.shape == (max_leg, self.n)
        assert mem_strength.dtype == float
        assert isinstance(max_iter_list, list)
        assert len(max_iter_list) == max_leg

        self.mem_strength = mem_strength
        self.max_iter_list = max_iter_list

    def decode(self, syndrome: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Syndrome vector, shape=(m,), dtype=int, values in {0, 1}.
            
            verbose : bool
                If True, print decoding information. Default is False.

        Returns
        -------
            best_ehat : ndarray
                Decoded error vector.
        """
        cnt = 0  # count number of solutions found
        best_ehat = None  # best solution found so far
        # weight of the best solution found so far (the lower the better)
        best_weight = float('inf')

        marginal = self.llr_prior.copy()
        for l in range(self.max_leg):
            if verbose:
                print(f"Decoding relay leg {l}...")
            ehat, marginal = self._DMem_BP_decode(syndrome=syndrome,
                                                  init_marginal=marginal,
                                                  gamma=self.mem_strength[l],
                                                  max_iter=self.max_iter_list[l],
                                                  verbose=verbose)
            if ehat is not None:  # this relay leg has converged
                cnt += 1
                weight = np.sum(ehat * self.llr_prior)
                if verbose:
                    print(
                        f"Found a solution at relay leg {l} with weight {weight}.")
                # check if this solution is better than the best one found so far
                if weight < best_weight:
                    best_ehat = ehat
                    best_weight = weight
                # check if we have found enough solutions
                if cnt >= self.num_sol:
                    if verbose:
                        print(f"Found enough solutions, stopping early.")
                    break

        if cnt == 0:  # no solution found
            if verbose:
                print("No solution found.")
            # defaults to all-zero error vector if no solution is found
            return np.zeros(self.n, dtype=int)
        else:  # at least one solution found
            if verbose:
                print(
                    f"Found {cnt} solutions, returning the best one with weight {best_weight}.")
            return best_ehat

    def decode_batch(self, syndrome: np.ndarray, progress_bar: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Array of syndrome vectors, shape=(N,m), dtype=int, values in {0, 1}.

            progress_bar : bool
                If True, show a progress bar. Default is True.

        Returns
        -------
            ehat : ndarray
                Array of decoded error vectors, shape=(N,n), dtype=int, values in {0, 1}.
        """
        assert isinstance(syndrome, np.ndarray)
        assert syndrome.shape[1] == self.m
        assert syndrome.dtype == int and np.all(np.isin(syndrome, [0, 1]))

        N = syndrome.shape[0]  # batch size

        # naive implementation
        ehat = np.zeros((N, self.n), dtype=int)
        for i in tqdm(range(N), disable=not progress_bar):
            ehat[i] = self.decode(syndrome[i])

        return ehat

    def _DMem_BP_decode(self, syndrome: np.ndarray, init_marginal: np.ndarray, gamma: np.ndarray, max_iter: int, verbose: bool
                        ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Disordered Memory BP decoding for a single relay leg.

        Parameters
        ----------
            syndrome : ndarray
                Syndrome vector, shape=(m,), dtype=int, values in {0, 1}.
            
            init_marginal : ndarray
                Initial marginals, shape=(n,), dtype=float.
            
            gamma : ndarray
                Memory strength for this relay leg, shape=(n,), dtype=float.
            
            max_iter : int
                Max number of BP iterations for this relay leg.
            
            verbose : bool
                If True, print convergence information.

        Returns
        -------
            ehat : ndarray or None
                Decoded error vector if the BP decoding converges, otherwise None.
            
            marginal : ndarray
                Marginal log-likelihood ratios from the last iteration.
        """
        syndrome_sign = 1 - 2 * \
            syndrome  # convert syndrome from {0,1} to {1,-1} representation

        # dict: (v, c) -> message from variable node v to check node c
        msg_v_to_c = {}
        # dict: (c, v) -> message from check node c to variable node v
        msg_c_to_v = {}

        # initialization (iteration 0)
        for v in range(self.n):
            for c in self.neighbors_v[v]:
                msg_v_to_c[(v, c)] = self.llr_prior[v]
        converged = False

        # main BP iterations (iteration 1 to max_iter)
        it = 1
        prev_marginal = init_marginal
        while it <= max_iter:
            # bias terms (used in calculating variable-to-check messages and marginals)
            bias = (1 - gamma) * self.llr_prior + gamma * prev_marginal
            # check-to-variable messages
            for c in range(self.m):
                for v_out in self.neighbors_c[c]:
                    in_msgs = [msg_v_to_c[(v, c)]
                               for v in self.neighbors_c[c] if v != v_out]
                    prod_sign = np.prod(np.sign(in_msgs))
                    min_abs = np.min(np.abs(in_msgs))
                    msg_c_to_v[(c, v_out)] = syndrome_sign[c] * \
                        prod_sign * min_abs
            # variable-to-check messages
            for v in range(self.n):
                for c_out in self.neighbors_v[v]:
                    in_msgs = [msg_c_to_v[(c, v)]
                               for c in self.neighbors_v[v] if c != c_out]
                    msg_v_to_c[(v, c_out)] = bias[v] + np.sum(in_msgs)
            # marginals
            marginal = np.copy(bias)
            for v in range(self.n):
                marginal[v] += np.sum([msg_c_to_v[(c, v)]
                                      for c in self.neighbors_v[v]])
            # hard decisions
            ehat = np.zeros(self.n, dtype=int)
            ehat[marginal < 0] = 1
            # check if syndrome is satisfied
            if np.all((self.H @ ehat) % 2 == syndrome):
                converged = True
                break
            it += 1
            prev_marginal = marginal  # update previous marginal for next iteration

        if verbose:
            if converged:
                print(f"This relay leg converges at iteration {it}.")
            else:
                print("This relay leg has reached max iterations without convergence.")

        if converged:
            return ehat, marginal
        else:
            return None, marginal
