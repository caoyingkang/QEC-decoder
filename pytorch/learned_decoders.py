import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

INT_DTYPE = torch.int32
FLOAT_DTYPE = torch.float32

EPS = 1e-6
BIG = 1e8


def _smooth_sign(x: torch.Tensor, *, alpha: float = 100.0) -> torch.Tensor:
    """
    Smooth version of sign function. Larger `alpha` => better approximation.
    """
    return torch.tanh(alpha * x)


class Sign_STE(torch.autograd.Function):
    """
    Straight-through estimator (STE) implementation of the sign function: calles `torch.sign` in the forward direction; 
    clamps the gradients to [-1, 1] and passes them through directly in the backward direction.
    """
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.clamp(grad_output, min=-1, max=1)


def _smooth_min(x: torch.Tensor, *, dim: int, temp: float = 0.01) -> torch.Tensor:
    """
    Smooth version of min function along a given dimension `dim`. Smaller `temp` => better approximation.
    """
    return torch.sum(x * F.softmin(x / temp, dim=dim), dim=dim)


def _build_tanner_graph(pcm: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """
    Build the Tanner graph of the parity-check matrix.

    Parameters
    ----------
        pcm : ndarray
            Parity-check matrix ∈ {0,1}, shape=(m, n), integer or bool

    Returns
    -------
        chk_nbrs : tuple[tuple[int, ...], ...]
            chk_nbrs[i] = all VNs connected to CN i
        var_nbrs : tuple[tuple[int, ...], ...]
            var_nbrs[j] = all CNs connected to VN j
    """
    m, n = pcm.shape
    chk_nbrs = tuple(tuple(np.nonzero(pcm[i])[0].tolist())
                     for i in range(m))
    var_nbrs = tuple(tuple(np.nonzero(pcm[:, j])[0].tolist())
                     for j in range(n))
    return chk_nbrs, var_nbrs


class _LearnedBPBase(nn.Module):
    """
    Base class for all trainable BP decoders.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        num_iters: int,
        min_impl_method: str = "smooth",
        sign_impl_method: str = "smooth",
    ):
        super().__init__()
        assert isinstance(pcm, np.ndarray) and isinstance(prior, np.ndarray)
        assert np.issubdtype(pcm.dtype, np.integer) or \
            np.issubdtype(pcm.dtype, np.bool_)
        assert np.issubdtype(prior.dtype, np.floating)
        assert pcm.ndim == 2
        m, n = pcm.shape
        assert prior.shape == (n,)
        assert num_iters > 0

        self.m, self.n = m, n
        self.num_iters = num_iters

        if min_impl_method == "smooth":
            self.min_func = _smooth_min
        elif min_impl_method == "hard":
            self.min_func = torch.amin
        else:
            raise ValueError(f"Invalid min_impl_method: {min_impl_method}")

        if sign_impl_method == "smooth":
            self.sign_func = _smooth_sign
        elif sign_impl_method == "hard":
            self.sign_func = torch.sign
        elif sign_impl_method == "ste":
            self.sign_func = Sign_STE.apply
        else:
            raise ValueError(f"Invalid sign_impl_method: {sign_impl_method}")

        self.chk_nbrs, self.var_nbrs = _build_tanner_graph(pcm)

        # Store prior LLRs
        prior = np.clip(prior, min=EPS, max=1-EPS)
        prior_llr = np.log((1 - prior) / prior)
        self.register_buffer(
            "prior_llr", torch.as_tensor(prior_llr, dtype=FLOAT_DTYPE))  # (n,)


class LearnedDMemBP(_LearnedBPBase):
    """
    A PyTorch Module that implements a Disordered Memory BP decoder with trainable memory strength.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        num_iters: int,
        min_impl_method: str = "smooth",
        sign_impl_method: str = "smooth",
    ):
        """
        Parameters
        ----------
            pcm : ndarray
                Parity-check matrix ∈ {0,1}, shape=(m, n), integer or bool

            prior : ndarray
                Prior probabilities of errors, shape=(n,), float

            num_iters : int
                Number of BP iterations

            min_impl_method : str
                Implementation method of the min function. Can be "smooth" (based on softmin) or "hard" (using torch.amin).

            sign_impl_method : str
                Implementation method of the sign function. Can be "smooth" (based on tanh), "hard" (using torch.sign), or "ste" (straight-through estimator).
        """
        super().__init__(pcm, prior, num_iters, min_impl_method, sign_impl_method)

        # Trainable parameter
        self.gamma = nn.Parameter(
            torch.zeros(self.n, dtype=FLOAT_DTYPE))

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, m), int

        Returns
        -------
            all_llrs : torch.Tensor
                LLRs output by the decoder at all BP iterations, shape=(batch_size, num_iters, n), float.
        """
        all_llrs = []

        device = syndromes.device
        batch_size = syndromes.shape[0]
        syndromes = syndromes.to(FLOAT_DTYPE)
        syndromes_sgn = 1.0 - 2.0 * syndromes  # (batch_size, m) ∈ {+1,-1}

        # Initialize messages
        # c2v_msg[:, i, j] = messages from CN i to VN j
        c2v_msg = torch.zeros(batch_size, self.m, self.n,
                              device=device, dtype=FLOAT_DTYPE)
        # v2c_msg[:, j, i] = messages from VN j to CN i
        v2c_msg = torch.zeros(batch_size, self.n, self.m,
                              device=device, dtype=FLOAT_DTYPE)
        for j in range(self.n):
            v2c_msg[:, j, self.var_nbrs[j]] = self.prior_llr[j]

        # print("Syndromes:", syndromes)  # DEBUG

        # Main BP iteration loop
        for it in range(self.num_iters):
            # ------------------ CN update ------------------
            c2v_msg = torch.zeros_like(c2v_msg)
            for i in range(self.m):
                nbrs = self.chk_nbrs[i]
                num_nbrs = len(nbrs)

                # Gather incoming messages at CN i
                msgs = v2c_msg[:, nbrs, i]  # (batch_size, num_nbrs)
                msgs_abs = msgs.abs()  # (batch_size, num_nbrs)
                msgs_sgn: torch.Tensor = self.sign_func(
                    msgs)  # (batch_size, num_nbrs)

                # print(f"Incoming messages at CN {i}:\n", msgs)  # DEBUG
                # print("msgs_sgn:", msgs_sgn)  # DEBUG
                # print("msgs_abs:", msgs_abs)  # DEBUG

                # For each neighboring VN, compute product over msgs_sgn excluding that VN.
                # We achieve leave-one-out by masking the corresponding entry with 1.0.
                msgs_sgn_repeated = msgs_sgn \
                    .unsqueeze(dim=1) \
                    .repeat(1, num_nbrs, 1)  # (batch_size, num_nbrs, num_nbrs)
                mask = torch.eye(num_nbrs, device=device, dtype=torch.bool) \
                    .unsqueeze(dim=0)  # (1, num_nbrs, num_nbrs)
                msgs_sgn_masked = msgs_sgn_repeated \
                    .masked_fill(mask, 1.0)  # (batch_size, num_nbrs, num_nbrs)
                msgs_sgn_prod_excl = msgs_sgn_masked \
                    .prod(dim=2)  # (batch_size, num_nbrs)

                # print("msgs_sgn_prod_excl:", msgs_sgn_prod_excl)  # DEBUG

                # For each neighboring VN, compute min over msgs_abs excluding that VN.
                # We achieve leave-one-out by masking the corresponding entry with a large number.
                msgs_abs_repeated = msgs_abs \
                    .unsqueeze(dim=1) \
                    .repeat(1, num_nbrs, 1)  # (batch_size, num_nbrs, num_nbrs)
                msgs_abs_masked = msgs_abs_repeated \
                    .masked_fill(mask, BIG)  # (batch_size, num_nbrs, num_nbrs)
                msgs_abs_min_excl = self.min_func(
                    msgs_abs_masked, dim=2)  # (batch_size, num_nbrs)

                # Populate c2v_msg
                c2v_msg[:, i, nbrs] = syndromes_sgn[:, i].unsqueeze(dim=1) * \
                    msgs_sgn_prod_excl * msgs_abs_min_excl

            # print("c2v_msg:\n", c2v_msg)  # DEBUG

            # ------------------ VN update ------------------
            incoming_sum = c2v_msg.sum(dim=1)  # (batch_size, n)
            if it == 0:
                llrs = incoming_sum + \
                    self.prior_llr.unsqueeze(dim=0)  # (batch_size, n)
            else:
                llrs = incoming_sum + \
                    (1 - self.gamma.unsqueeze(dim=0)) * self.prior_llr.unsqueeze(dim=0) + \
                    self.gamma.unsqueeze(dim=0) * llrs  # (batch_size, n)

            all_llrs.append(llrs)

            if it < self.num_iters - 1:  # no need to update v2c_msg in the last iteration
                v2c_msg = torch.zeros_like(v2c_msg)
                for j in range(self.n):
                    nbrs = self.var_nbrs[j]
                    v2c_msg[:, j, nbrs] = llrs[:, j].unsqueeze(dim=1) - \
                        c2v_msg[:, nbrs, j]

            # print("prior_llr:\n", self.prior_llr)  # DEBUG
            # print("llrs:\n", llrs)  # DEBUG
            # print("v2c_msg:\n", v2c_msg)  # DEBUG

        all_llrs = torch.stack(all_llrs, dim=1)
        return all_llrs


class LearnedDMemOffBP(_LearnedBPBase):
    """
    A PyTorch Module that implements a Disordered Memory Offset BP decoder with trainable memory strength and offset parameters.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        num_iters: int,
        min_impl_method: str = "smooth",
        sign_impl_method: str = "smooth",
    ):
        """
        Parameters
        ----------
            pcm : ndarray
                Parity-check matrix ∈ {0,1}, shape=(m, n), integer or bool

            prior : ndarray
                Prior probabilities of errors, shape=(n,), float

            num_iters : int
                Number of BP iterations

            min_impl_method : str
                Implementation method of the min function. Can be "smooth" (based on softmin) or "hard" (using torch.amin).

            sign_impl_method : str
                Implementation method of the sign function. Can be "smooth" (based on tanh), "hard" (using torch.sign), or "ste" (straight-through estimator).
        """
        super().__init__(pcm, prior, num_iters, min_impl_method, sign_impl_method)

        # Trainable parameter
        self.gamma = nn.Parameter(
            torch.zeros(self.n, dtype=FLOAT_DTYPE))
        self.offset = nn.ParameterList([
            nn.Parameter(torch.zeros(len(self.chk_nbrs[i]), dtype=FLOAT_DTYPE))
            for i in range(self.m)
        ])

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, m), int

        Returns
        -------
            all_llrs : torch.Tensor
                LLRs output by the decoder at all BP iterations, shape=(batch_size, num_iters, n), float.
        """
        all_llrs = []

        device = syndromes.device
        batch_size = syndromes.shape[0]
        syndromes = syndromes.to(FLOAT_DTYPE)
        syndromes_sgn = 1.0 - 2.0 * syndromes  # (batch_size, m) ∈ {+1,-1}

        # Initialize messages
        # c2v_msg[:, i, j] = messages from CN i to VN j
        c2v_msg = torch.zeros(batch_size, self.m, self.n,
                              device=device, dtype=FLOAT_DTYPE)
        # v2c_msg[:, j, i] = messages from VN j to CN i
        v2c_msg = torch.zeros(batch_size, self.n, self.m,
                              device=device, dtype=FLOAT_DTYPE)
        for j in range(self.n):
            v2c_msg[:, j, self.var_nbrs[j]] = self.prior_llr[j]

        # print("Syndromes:", syndromes)  # DEBUG

        # Main BP iteration loop
        for it in range(self.num_iters):
            # ------------------ CN update ------------------
            c2v_msg = torch.zeros_like(c2v_msg)
            for i in range(self.m):
                nbrs = self.chk_nbrs[i]
                num_nbrs = len(nbrs)

                # Gather incoming messages at CN i
                msgs = v2c_msg[:, nbrs, i]  # (batch_size, num_nbrs)
                msgs_abs = msgs.abs()  # (batch_size, num_nbrs)
                msgs_sgn: torch.Tensor = self.sign_func(
                    msgs)  # (batch_size, num_nbrs)

                # print(f"Incoming messages at CN {i}:\n", msgs)  # DEBUG
                # print("msgs_sgn:", msgs_sgn)  # DEBUG
                # print("msgs_abs:", msgs_abs)  # DEBUG

                # For each neighboring VN, compute product over msgs_sgn excluding that VN.
                # We achieve leave-one-out by masking the corresponding entry with 1.0.
                msgs_sgn_repeated = msgs_sgn \
                    .unsqueeze(dim=1) \
                    .repeat(1, num_nbrs, 1)  # (batch_size, num_nbrs, num_nbrs)
                mask = torch.eye(num_nbrs, device=device, dtype=torch.bool) \
                    .unsqueeze(dim=0)  # (1, num_nbrs, num_nbrs)
                msgs_sgn_masked = msgs_sgn_repeated \
                    .masked_fill(mask, 1.0)  # (batch_size, num_nbrs, num_nbrs)
                msgs_sgn_prod_excl = msgs_sgn_masked \
                    .prod(dim=2)  # (batch_size, num_nbrs)

                # print("msgs_sgn_prod_excl:", msgs_sgn_prod_excl)  # DEBUG

                # For each neighboring VN, compute min over msgs_abs excluding that VN.
                # We achieve leave-one-out by masking the corresponding entry with a large number.
                msgs_abs_repeated = msgs_abs \
                    .unsqueeze(dim=1) \
                    .repeat(1, num_nbrs, 1)  # (batch_size, num_nbrs, num_nbrs)
                msgs_abs_masked = msgs_abs_repeated \
                    .masked_fill(mask, BIG)  # (batch_size, num_nbrs, num_nbrs)
                msgs_abs_min_excl = self.min_func(
                    msgs_abs_masked, dim=2)  # (batch_size, num_nbrs)

                # Populate c2v_msg
                c2v_msg[:, i, nbrs] = syndromes_sgn[:, i].unsqueeze(dim=1) * \
                    msgs_sgn_prod_excl * \
                    F.relu(msgs_abs_min_excl - self.offset[i].unsqueeze(dim=0))

            # print("c2v_msg:\n", c2v_msg)  # DEBUG

            # ------------------ VN update ------------------
            incoming_sum = c2v_msg.sum(dim=1)  # (batch_size, n)
            if it == 0:
                llrs = incoming_sum + \
                    self.prior_llr.unsqueeze(dim=0)  # (batch_size, n)
            else:
                llrs = incoming_sum + \
                    (1 - self.gamma.unsqueeze(dim=0)) * self.prior_llr.unsqueeze(dim=0) + \
                    self.gamma.unsqueeze(dim=0) * llrs  # (batch_size, n)

            all_llrs.append(llrs)

            if it < self.num_iters - 1:  # no need to update v2c_msg in the last iteration
                v2c_msg = torch.zeros_like(v2c_msg)
                for j in range(self.n):
                    nbrs = self.var_nbrs[j]
                    v2c_msg[:, j, nbrs] = llrs[:, j].unsqueeze(dim=1) - \
                        c2v_msg[:, nbrs, j]

            # print("prior_llr:\n", self.prior_llr)  # DEBUG
            # print("llrs:\n", llrs)  # DEBUG
            # print("v2c_msg:\n", v2c_msg)  # DEBUG

        all_llrs = torch.stack(all_llrs, dim=1)
        return all_llrs


class DecodingLoss_ParityBased(nn.Module):
    """
    A PyTorch Module that implements a loss function for training QEC decoders.

    Given a check matrix `chkmat` and an observable matrix `obsmat`, the loss function consists of two parts:
    1. The first part quantifies how the estimated error pattern recovers the syndrome (i.e., are we back to the code space?).
    2. The second part quantifies how the estimated error pattern predicts the observable (i.e., is there a logical error?).

    More specifically, to calculate the loss for a single shot (`llr`, `syndrome`, `observable`), we first compute the estimated probability 
    that each error bit is 1 from the LLRs: `p[j] = sigmoid(-llr[j])`. Then, we compute the loss from part 1 as `loss1 = sum(loss_syn)`, 
    where `loss_syn[i] = parity_proxy(syndrome[i] + (chkmat @ p)[i])`, where `parity_proxy` is a function that is differentiable almost everywhere, 
    reduces to the parity function for integer inputs, and attains global minimum at even integers. Similarly, we compute the loss from part 2 as 
    `loss2 = sum(loss_obs)`, where `loss_obs[i] = parity_proxy(observable[i] + (obsmat @ p)[i])`. Finally, the total loss is 
    `loss = ß * loss1 + (1-ß) * loss2`, where `ß` ∈ [0,1] is a hyperparameter that controls the relative importance of the two parts.

    To calculate the loss for a batch of shots, we average the loss of each shot over the batch.
    """

    @staticmethod
    def parity_proxy(x: torch.Tensor) -> torch.Tensor:
        """
        Here we choose `parity_proxy(x) = abs(sin(πx/2))` as was used in 
        [Liu and Poulin, Phys. Rev. Lett. 122, 200501 (2019)](https://doi.org/10.1103/PhysRevLett.122.200501).
        """
        return torch.abs(torch.sin(torch.pi * x / 2))

    def __init__(
        self,
        chkmat: np.ndarray,
        obsmat: np.ndarray,
        *,
        beta: float = 0.5,
        incl_intmd_llrs: bool = False,
    ):
        """
        Parameters
        ----------
            chkmat : ndarray
                Check matrix ∈ {0,1}, shape=(m, n), integer or bool

            obsmat : ndarray
                Observable matrix ∈ {0,1}, shape=(k, n), integer or bool

            beta : float
                Hyperparameter that balances the contribution of the two parts of the loss function

            incl_intmd_llrs : bool
                Whether to include LLRs from intermediate BP iterations in the calculation of the loss
        """
        super().__init__()
        assert isinstance(chkmat, np.ndarray)
        assert isinstance(obsmat, np.ndarray)
        assert np.issubdtype(chkmat.dtype, np.integer) or \
            np.issubdtype(chkmat.dtype, np.bool_)
        assert np.issubdtype(obsmat.dtype, np.integer) or \
            np.issubdtype(obsmat.dtype, np.bool_)
        assert chkmat.ndim == 2 and obsmat.ndim == 2
        assert chkmat.shape[1] == obsmat.shape[1]
        assert 0 <= beta <= 1

        self.register_buffer(
            "chkmat", torch.as_tensor(chkmat, dtype=FLOAT_DTYPE))
        self.register_buffer(
            "obsmat", torch.as_tensor(obsmat, dtype=FLOAT_DTYPE))

        self.beta = beta
        self.incl_intmd_llrs = incl_intmd_llrs

    def forward(
        self,
        all_llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            all_llrs : torch.Tensor
                LLRs output by the decoder at all BP iterations, shape=(batch_size, num_iters, n), float.

            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, m), int

            observables : torch.Tensor
                Observable bits ∈ {0,1}, shape=(batch_size, k), int

        Returns
        -------
            loss : torch.Tensor
                Loss, shape=(), float
        """
        syndromes = syndromes.to(FLOAT_DTYPE)
        observables = observables.to(FLOAT_DTYPE)

        if not self.incl_intmd_llrs:
            all_llrs = all_llrs[:, [-1], :]  # view on the last BP iteration

        p = torch.sigmoid(-all_llrs)  # (batch_size, num_iters, n)

        # Compute loss from part 1
        loss_syn = self.__class__.parity_proxy(
            syndromes.unsqueeze(dim=1) +
            torch.matmul(p, self.chkmat.T))  # (batch_size, num_iters, m)
        loss1 = loss_syn.sum(dim=2)  # (batch_size, num_iters)

        # Compute loss from part 2
        loss_obs = self.__class__.parity_proxy(
            observables.unsqueeze(dim=1) +
            torch.matmul(p, self.obsmat.T))  # (batch_size, num_iters, k)
        loss2 = loss_obs.sum(dim=2)  # (batch_size, num_iters)

        # Compute total loss
        loss = self.beta * loss1 + \
            (1. - self.beta) * loss2  # (batch_size, num_iters)
        return loss.mean()


class DecodingLoss_BCEBased(nn.Module):
    """
    A PyTorch Module that implements a loss function for training QEC decoders.

    Given a check matrix `chkmat` and an observable matrix `obsmat`, the loss function consists of two parts:
    1. The first part quantifies how the estimated error pattern recovers the syndrome (i.e., are we back to the code space?).
    2. The second part quantifies how the estimated error pattern predicts the observable (i.e., is there a logical error?).

    More specifically, suppose we want to calculate the loss for a single shot (`llr`, `syndrome`, `observable`). The loss from part 1 is 
    `loss1 = sum(loss_syn)`, where `loss_syn[i] = BCEWithLogitsLoss(-syndrome_pred_llr[i], syndrome[i])`, where `syndrome_pred_llr[i]` is the 
    LLR value of the `i`-th syndrome bit calculated from the LLR values of those error bits corresponding to the `i`-th row of `chkmat`. 
    Similarly, the loss from part 2 is `loss2 = sum(loss_obs)`, where `loss_obs[i] = BCEWithLogitsLoss(-observable_pred_llr[i], observable[i])`, 
    where `observable_pred_llr[i]` is the LLR value of the `i`-th observable bit calculated from the LLR values of those error bits corresponding 
    to the `i`-th row of `obsmat`. Finally, the total loss is `loss = ß * loss1 + (1-ß) * loss2`, where `ß` ∈ [0,1] is a hyperparameter that 
    controls the relative importance of the two parts.

    To calculate the loss for a batch of shots, we average the loss of each shot over the batch.
    """

    def __init__(
        self,
        chkmat: np.ndarray,
        obsmat: np.ndarray,
        *,
        beta: float = 0.5,
        incl_intmd_llrs: bool = False,
    ):
        """
        Parameters
        ----------
            chkmat : ndarray
                Check matrix ∈ {0,1}, shape=(m, n), integer or bool

            obsmat : ndarray
                Observable matrix ∈ {0,1}, shape=(k, n), integer or bool

            beta : float
                Hyperparameter that balances the contribution of the two parts of the loss function

            incl_intmd_llrs : bool
                Whether to include LLRs from intermediate BP iterations in the calculation of the loss
        """
        super().__init__()
        assert isinstance(chkmat, np.ndarray)
        assert isinstance(obsmat, np.ndarray)
        assert np.issubdtype(chkmat.dtype, np.integer) or \
            np.issubdtype(chkmat.dtype, np.bool_)
        assert np.issubdtype(obsmat.dtype, np.integer) or \
            np.issubdtype(obsmat.dtype, np.bool_)
        assert chkmat.ndim == 2 and obsmat.ndim == 2
        assert chkmat.shape[1] == obsmat.shape[1]
        assert 0 <= beta <= 1

        m, n = chkmat.shape
        k = obsmat.shape[0]
        self.m, self.n, self.k = m, n, k
        # chk_supp[i] = support of the i-th row of chkmat
        self.chk_supp = tuple(tuple(j for j in range(n) if chkmat[i, j])
                              for i in range(m))
        # obs_supp[i] = support of the i-th row of obsmat
        self.obs_supp = tuple(tuple(j for j in range(n) if obsmat[i, j])
                              for i in range(k))

        self.beta = beta
        self.incl_intmd_llrs = incl_intmd_llrs

    def forward(
        self,
        all_llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            all_llrs : torch.Tensor
                LLRs output by the decoder at all BP iterations, shape=(batch_size, num_iters, n), float.

            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, m), int

            observables : torch.Tensor
                Observable bits ∈ {0,1}, shape=(batch_size, k), int

        Returns
        -------
            loss : torch.Tensor
                Loss, shape=(), float
        """
        syndromes = syndromes.to(FLOAT_DTYPE)
        observables = observables.to(FLOAT_DTYPE)

        if not self.incl_intmd_llrs:
            all_llrs = all_llrs[:, [-1], :]  # view on the last BP iteration

        tanh_llrs_over_2 = torch.tanh(
            all_llrs / 2)  # (batch_size, num_iters, n)

        # Compute loss from part 1
        syndromes_pred_llr = []
        for i in range(self.m):
            supp = self.chk_supp[i]
            syndrome_i_pred_llr = 2 * (
                torch.prod(tanh_llrs_over_2[:, :, supp], dim=2)
                .clamp(min=-1 + EPS, max=1 - EPS)
                .atanh()
            )  # (batch_size, num_iters)
            syndromes_pred_llr.append(syndrome_i_pred_llr)
        syndromes_pred_llr = torch.stack(
            syndromes_pred_llr, dim=2)  # (batch_size, num_iters, m)

        loss_syn = F.binary_cross_entropy_with_logits(
            -syndromes_pred_llr,
            syndromes.unsqueeze(dim=1).expand_as(syndromes_pred_llr),
            reduction="none"
        )  # (batch_size, num_iters, m)
        loss1 = loss_syn.sum(dim=2)  # (batch_size, num_iters)

        # Compute loss from part 2
        observables_pred_llr = []
        for i in range(self.k):
            supp = self.obs_supp[i]
            observable_i_pred_llr = 2 * (
                torch.prod(tanh_llrs_over_2[:, :, supp], dim=2)
                .clamp(min=-1 + EPS, max=1 - EPS)
                .atanh()
            )  # (batch_size, num_iters)
            observables_pred_llr.append(observable_i_pred_llr)
        observables_pred_llr = torch.stack(
            observables_pred_llr, dim=2)  # (batch_size, num_iters, k)

        loss_obs = F.binary_cross_entropy_with_logits(
            -observables_pred_llr,
            observables.unsqueeze(dim=1).expand_as(observables_pred_llr),
            reduction="none"
        )  # (batch_size, num_iters, k)
        loss2 = loss_obs.sum(dim=2)  # (batch_size, num_iters)

        # Compute total loss
        loss = self.beta * loss1 + \
            (1. - self.beta) * loss2  # (batch_size, num_iters)
        return loss.mean()


class DecodingMetric(Metric):
    """
    A PyTorch Metric that calculates the performance metrics of the decoder.
    """

    def __init__(
        self,
        chkmat: np.ndarray,
        obsmat: np.ndarray,
    ):
        """
        Parameters
        ----------
            chkmat : ndarray
                Check matrix ∈ {0,1}, shape=(m, n), integer or bool

            obsmat : ndarray
                Observable matrix ∈ {0,1}, shape=(k, n), integer or bool
        """
        super().__init__()
        assert isinstance(chkmat, np.ndarray)
        assert isinstance(obsmat, np.ndarray)
        assert np.issubdtype(chkmat.dtype, np.integer) or \
            np.issubdtype(chkmat.dtype, np.bool_)
        assert np.issubdtype(obsmat.dtype, np.integer) or \
            np.issubdtype(obsmat.dtype, np.bool_)
        assert chkmat.ndim == 2 and obsmat.ndim == 2
        assert chkmat.shape[1] == obsmat.shape[1]

        self.register_buffer(
            "chkmat", torch.as_tensor(chkmat, dtype=INT_DTYPE))
        self.register_buffer(
            "obsmat", torch.as_tensor(obsmat, dtype=INT_DTYPE))

        self.add_state("wrong_syndrome", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("wrong_observable", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("wrong_either", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(
        self,
        all_llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor
    ):
        """
        Parameters
        ----------
            all_llrs : torch.Tensor
                LLRs output by the decoder at all BP iterations, shape=(batch_size, num_iters, n), float.

            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, m), int

            observables : torch.Tensor
                Observable bits ∈ {0,1}, shape=(batch_size, k), int
        """
        batch_size, num_iters, n = all_llrs.shape

        # For each shot, check if the decoder converges, i.e., whether the syndrome is matched at any iteration
        hard_decisions = (all_llrs < 0).to(
            INT_DTYPE)  # (batch_size, num_iters, n), int, 0/1
        syndromes_pred = torch.matmul(
            hard_decisions, self.chkmat.T) % 2  # (batch_size, num_iters, m), int, 0/1
        syndromes_matched_mask = torch.all(
            syndromes_pred == syndromes.unsqueeze(dim=1), dim=2)  # (batch_size, num_iters), bool
        converged_mask = torch.any(
            syndromes_matched_mask, dim=1)  # (batch_size,), bool

        # For each shot, find which iteration is the overall output of the decoder:
        # If the decoder converges, this is the first iteration where the syndrome is matched;
        # If the decoder does not converge, this is the last iteration.
        output_iters = torch.where(
            converged_mask,
            syndromes_matched_mask.int().argmax(dim=1),
            num_iters - 1
        )  # (batch_size,), int

        # Get the output error pattern for each shot
        index = output_iters.reshape(batch_size, 1, 1).expand(batch_size, 1, n)
        ehat = hard_decisions \
            .gather(dim=1, index=index) \
            .squeeze(1)  # (batch_size, n), int, 0/1

        # For each shot, check if the decoder predicts the observables correctly
        observables_pred = torch.matmul(
            ehat, self.obsmat.T) % 2  # (batch_size, k), int, 0/1
        observables_correct_mask = torch.all(
            observables_pred == observables, dim=1)  # (batch_size,), bool

        # Update states
        self.wrong_syndrome += torch.sum(~converged_mask)
        self.wrong_observable += torch.sum(~observables_correct_mask)
        self.wrong_either += torch.sum(~converged_mask |
                                       ~observables_correct_mask)
        self.total += batch_size

    def compute(self) -> dict[str, float]:
        return {
            "wrong_syndrome_rate": self.wrong_syndrome.float() / self.total.float(),
            "wrong_observable_rate": self.wrong_observable.float() / self.total.float(),
            "failure_rate": self.wrong_either.float() / self.total.float(),
        }


__all__ = [
    "LearnedDMemBP",
    "LearnedDMemOffBP",
    "DecodingLoss_ParityBased",
    "DecodingLoss_BCEBased",
    "DecodingMetric",
]
