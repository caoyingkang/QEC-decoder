from itertools import combinations
import numpy as np
import stim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from qecdec import detector_error_model_to_check_matrices

dtype = torch.float32

EPS = 1e-6
BIG = 1e8


def smooth_sign(x: torch.Tensor, *, alpha: float = 100.0) -> torch.Tensor:
    """
    Smooth version of sign function. Larger `alpha` => better approximation.
    """
    return torch.tanh(alpha * x)


def smooth_min(x: torch.Tensor, *, dim: int, temp: float = 0.01) -> torch.Tensor:
    """
    Smooth version of min function along a given dimension `dim`. Smaller `temp` => better approximation.
    """
    return torch.sum(x * F.softmin(x / temp, dim=dim), dim=dim)


def build_tanner_graph(pcm: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
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


class DecodingDataset(Dataset):
    """
    A PyTorch Dataset. Each item is a (syndrome, observable) pair.
    """

    def __init__(
        self,
        syndromes: np.ndarray,  # (num_shots, m), ∈ {0,1}
        observables: np.ndarray,  # (num_shots, k), ∈ {0,1}
    ):
        """
        Parameters
        ----------
            syndromes : np.ndarray
                Syndrome bits ∈ {0,1}, shape=(num_shots, m), integer or bool

            observables : np.ndarray
                Observable bits ∈ {0,1}, shape=(num_shots, k), integer or bool
        """
        assert isinstance(syndromes, np.ndarray)
        assert isinstance(observables, np.ndarray)
        assert np.issubdtype(syndromes.dtype, np.integer) or \
            np.issubdtype(syndromes.dtype, np.bool_)
        assert np.issubdtype(observables.dtype, np.integer) or \
            np.issubdtype(observables.dtype, np.bool_)
        assert syndromes.ndim == 2
        assert observables.ndim == 2
        assert syndromes.shape[0] == observables.shape[0]

        self.syndromes = torch.as_tensor(syndromes, dtype=dtype)
        self.observables = torch.as_tensor(observables, dtype=dtype)

    def __len__(self):
        return len(self.syndromes)

    def __getitem__(self, idx):
        return self.syndromes[idx], self.observables[idx]


class LearnedDMemBP(nn.Module):
    """
    A PyTorch Module that implements a DMemBP decoder with trainable memory strength.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        num_iters: int
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
        """
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

        self.chk_nbrs, self.var_nbrs = build_tanner_graph(pcm)

        # Store prior LLRs
        prior = np.clip(prior, min=EPS, max=1-EPS)
        prior_llr = np.log((1 - prior) / prior)
        self.register_buffer("prior_llr",
                             torch.as_tensor(prior_llr, dtype=dtype))  # (n,)

        # Trainable parameter
        self.gamma = nn.Parameter(torch.zeros(self.n, dtype=dtype))  # (n,)

    @classmethod
    def from_dem(
        cls,
        dem: stim.DetectorErrorModel,
        *,
        num_iters: int
    ) -> "LearnedDMemBP":
        """
        Alternative constructor that reads `pcm` and `prior` from a `stim.DetectorErrorModel`.

        Parameters
        ----------
            dem : stim.DetectorErrorModel
                Detector error model

            num_iters : int
                Number of BP iterations
        """
        matrices = detector_error_model_to_check_matrices(dem)
        pcm = matrices.check_matrix.toarray().astype(np.uint8)
        prior = matrices.priors.astype(np.float64)

        return cls(pcm, prior, num_iters=num_iters)

    def forward(
        self,
        syndromes: torch.Tensor,
        *,
        return_intmd_llrs: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, m), float

            return_intmd_llrs : bool
                Whether to return intermediate LLRs. Default is False.

        Returns
        -------
            llrs : torch.Tensor | list[torch.Tensor]
                If `return_intmd_llrs` is False, then return a tensor of the final LLRs, shape=(batch_size, n).
                If `return_intmd_llrs` is True, then return a list of tensors for LLRs at each BP iteration.
        """
        if return_intmd_llrs:
            all_llrs = []

        device = syndromes.device
        batch_size = syndromes.shape[0]
        syndromes_sgn = 1.0 - 2.0 * syndromes  # (batch_size, m) ∈ {+1,-1}

        # Initialize messages
        # c2v_msg[:, i, j] = messages from CN i to VN j
        c2v_msg = torch.zeros(batch_size, self.m, self.n,
                              device=device, dtype=dtype)
        # v2c_msg[:, j, i] = messages from VN j to CN i
        v2c_msg = torch.zeros(batch_size, self.n, self.m,
                              device=device, dtype=dtype)
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
                msgs_sgn = smooth_sign(msgs)  # (batch_size, num_nbrs)

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
                msgs_abs_min_excl = smooth_min(
                    msgs_abs_masked, dim=2)  # (batch_size, num_nbrs)

                # print("msgs_abs_min_excl:", msgs_abs_min_excl)  # DEBUG

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

            if return_intmd_llrs:
                all_llrs.append(llrs)

            if it == self.num_iters - 1:  # no need to update v2c_msg in the last iteration
                break

            v2c_msg = torch.zeros_like(v2c_msg)
            for j in range(self.n):
                nbrs = self.var_nbrs[j]
                v2c_msg[:, j, nbrs] = llrs[:, j].unsqueeze(dim=1) - \
                    c2v_msg[:, nbrs, j]

            # print("prior_llr:\n", self.prior_llr)  # DEBUG
            # print("llrs:\n", llrs)  # DEBUG
            # print("v2c_msg:\n", v2c_msg)  # DEBUG

        if return_intmd_llrs:
            return all_llrs
        else:
            return llrs


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
            "chkmat", torch.as_tensor(chkmat, dtype=dtype))
        self.register_buffer(
            "obsmat", torch.as_tensor(obsmat, dtype=dtype))

        self.beta = beta

    @classmethod
    def from_dem(
        cls,
        dem: stim.DetectorErrorModel,
        *,
        beta: float = 0.5,
    ) -> "DecodingLoss_ParityBased":
        """
        Alternative constructor that reads `chkmat` and `obsmat` from a `stim.DetectorErrorModel`.

        Parameters
        ----------
            dem : stim.DetectorErrorModel
                Detector error model

            beta : float
                Hyperparameter that balances the contribution of the two parts of the loss function
        """
        matrices = detector_error_model_to_check_matrices(dem)
        chkmat = matrices.check_matrix.toarray().astype(np.uint8)
        obsmat = matrices.observables_matrix.toarray().astype(np.uint8)
        return cls(chkmat, obsmat, beta=beta)

    def forward(
        self,
        llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            llrs : torch.Tensor
                LLRs output by the decoder, shape=(batch_size, n), float

            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, m), float

            observables : torch.Tensor
                Observable bits ∈ {0,1}, shape=(batch_size, k), float

        Returns
        -------
            loss : torch.Tensor
                Loss, shape=(), float
        """
        p = torch.sigmoid(-llrs)  # (batch_size, n)

        loss_syn = self.__class__.parity_proxy(
            syndromes + (p @ self.chkmat.T))  # (batch_size, m)
        loss1 = loss_syn.sum(dim=1)  # (batch_size,)

        loss_obs = self.__class__.parity_proxy(
            observables + (p @ self.obsmat.T))  # (batch_size, k)
        loss2 = loss_obs.sum(dim=1)  # (batch_size,)

        loss = self.beta * loss1 + (1. - self.beta) * loss2  # (batch_size,)
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

        self.register_buffer(
            "chkmat", torch.as_tensor(chkmat, dtype=dtype))
        self.register_buffer(
            "obsmat", torch.as_tensor(obsmat, dtype=dtype))

        self.beta = beta

    @classmethod
    def from_dem(
        cls,
        dem: stim.DetectorErrorModel,
        *,
        beta: float = 0.5,
    ) -> "DecodingLoss_BCEBased":
        """
        Alternative constructor that reads `chkmat` and `obsmat` from a `stim.DetectorErrorModel`.

        Parameters
        ----------
            dem : stim.DetectorErrorModel
                Detector error model

            beta : float
                Hyperparameter that balances the contribution of the two parts of the loss function
        """
        matrices = detector_error_model_to_check_matrices(dem)
        chkmat = matrices.check_matrix.toarray().astype(np.uint8)
        obsmat = matrices.observables_matrix.toarray().astype(np.uint8)
        return cls(chkmat, obsmat, beta=beta)

    def forward(
        self,
        llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            llrs : torch.Tensor
                LLRs output by the decoder, shape=(batch_size, n), float

            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, m), float

            observables : torch.Tensor
                Observable bits ∈ {0,1}, shape=(batch_size, k), float

        Returns
        -------
            loss : torch.Tensor
                Loss, shape=(), float
        """
        tanh_llrs_over_2 = torch.tanh(llrs / 2)  # (batch_size, n)

        syndromes_pred_llr = torch.zeros_like(syndromes)  # (batch_size, m)
        for i in range(self.m):
            supp = self.chk_supp[i]
            syndromes_pred_llr[:, i] = 2 * (
                torch.prod(tanh_llrs_over_2[:, supp], dim=1)
                .clamp(min=-1 + EPS, max=1 - EPS)
                .atanh()
            )
        loss_syn = F.binary_cross_entropy_with_logits(
            -syndromes_pred_llr, syndromes, reduction="none")  # (batch_size, m)
        loss1 = loss_syn.sum(dim=1)  # (batch_size,)

        observables_pred_llr = torch.zeros_like(observables)  # (batch_size, k)
        for i in range(self.k):
            supp = self.obs_supp[i]
            observables_pred_llr[:, i] = 2 * (
                torch.prod(tanh_llrs_over_2[:, supp], dim=1)
                .clamp(min=-1 + EPS, max=1 - EPS)
                .atanh()
            )
        loss_obs = F.binary_cross_entropy_with_logits(
            -observables_pred_llr, observables, reduction="none")  # (batch_size, k)
        loss2 = loss_obs.sum(dim=1)  # (batch_size,)

        loss = self.beta * loss1 + (1. - self.beta) * loss2  # (batch_size,)
        return loss.mean()


def build_datasets(
    dem: stim.DetectorErrorModel,
    *,
    train_shots: int,
    dev_shots: int,
    seed: int | None = None,
    train_all_wt1_errors: bool = True,
    train_all_wt2_errors: bool = True,
    remove_trivial_syndromes: bool = True,
) -> tuple[DecodingDataset, DecodingDataset]:
    """
    Parameters
    ----------
        dem : stim.DetectorErrorModel
            Detector error model of the sampling circuit

        train_shots : int
            Number of sampling shots for building `train_dataset`

        dev_shots : int
            Number of sampling shots for building `dev_dataset`

        seed : int | None
            Random seed used for sampling

        train_all_wt1_errors : bool
            Whether to include all weight-1 errors in `train_dataset`

        train_all_wt2_errors : bool
            Whether to include all weight-2 errors in `train_dataset`

        remove_trivial_syndromes : bool
            Whether to filter out trivial (i.e., all-zero) syndromes in `train_dataset` and `dev_dataset`

    Returns
    -------
        train_dataset : DecodingDataset
            Training dataset

        dev_dataset : DecodingDataset
            Development dataset
    """
    m, n, k = dem.num_detectors, dem.num_errors, dem.num_observables
    matrices = detector_error_model_to_check_matrices(dem)
    chkmat = matrices.check_matrix.toarray().astype(np.uint8)
    obsmat = matrices.observables_matrix.toarray().astype(np.uint8)
    assert chkmat.shape == (m, n)
    assert obsmat.shape == (k, n)

    sampler = dem.compile_sampler(seed=seed)
    sampled_syndromes, sampled_observables, _ = sampler.sample(
        train_shots + dev_shots)
    sampled_syndromes = sampled_syndromes.astype(np.uint8)
    sampled_observables = sampled_observables.astype(np.uint8)

    train_syndromes_list = [sampled_syndromes[:train_shots]]
    train_observables_list = [sampled_observables[:train_shots]]

    if train_all_wt1_errors:
        errors = np.eye(n, dtype=np.uint8)
        syndromes = (errors @ chkmat.T) % 2
        observables = (errors @ obsmat.T) % 2
        train_syndromes_list.append(syndromes)
        train_observables_list.append(observables)

    if train_all_wt2_errors:
        errors = np.zeros(((n * (n - 1)) // 2, n), dtype=np.uint8)
        for row, cols in enumerate(combinations(range(n), 2)):
            errors[row, cols] = 1
        syndromes = (errors @ chkmat.T) % 2
        observables = (errors @ obsmat.T) % 2
        train_syndromes_list.append(syndromes)
        train_observables_list.append(observables)

    train_syndromes = np.concatenate(train_syndromes_list)
    train_observables = np.concatenate(train_observables_list)
    if remove_trivial_syndromes:
        mask = np.any(train_syndromes != 0, axis=1)
        train_syndromes = train_syndromes[mask]
        train_observables = train_observables[mask]

    train_dataset = DecodingDataset(train_syndromes, train_observables)

    dev_syndromes = sampled_syndromes[train_shots:]
    dev_observables = sampled_observables[train_shots:]
    if remove_trivial_syndromes:
        mask = np.any(dev_syndromes != 0, axis=1)
        dev_syndromes = dev_syndromes[mask]
        dev_observables = dev_observables[mask]

    dev_dataset = DecodingDataset(dev_syndromes, dev_observables)

    return train_dataset, dev_dataset


def train_gamma(
    model: LearnedDMemBP,  # the decoder to be trained
    train_dataset: DecodingDataset,
    dev_dataset: DecodingDataset,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    incl_intmd_llrs: bool = False,  # whether to include intermediate LLRs in the loss
    num_epochs: int = 10,
    batch_size: int = 64,
    device: str = None,
) -> np.ndarray:
    if device is None:
        if torch.accelerator.is_available():
            device = torch.accelerator.current_accelerator().type
        else:
            device = "cpu"
    print(f"Using {device} device")

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Build dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False)

    # Train model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        # Training phase
        model.train()
        train_losses = []
        for batch, (syndromes, observables) in enumerate(train_dataloader):
            syndromes = syndromes.to(device)
            observables = observables.to(device)
            optimizer.zero_grad()

            # Forward pass
            if incl_intmd_llrs:
                all_llrs = model(syndromes, return_intmd_llrs=True)
                all_losses = [loss_fn(llrs, syndromes, observables)
                              for llrs in all_llrs]
                loss = torch.stack(all_losses).mean()
            else:
                llrs = model(syndromes, return_intmd_llrs=False)
                loss = loss_fn(llrs, syndromes, observables)
            train_losses.append(loss.item())

            # Backpropagation
            loss.backward()
            if batch % 1 == 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float('inf'))
                print(f"Gradient norm: {grad_norm:.6f}")
                print("Loss: {:>8f}  [{:>5d}/{:>5d}]".format(
                    loss.item(),
                    batch * batch_size + len(syndromes),
                    len(train_dataset)))
            optimizer.step()

        # Evaluation phase on dev set
        model.eval()
        dev_losses = []
        with torch.no_grad():
            for syndromes, observables in dev_dataloader:
                syndromes = syndromes.to(device)
                observables = observables.to(device)

                # Forward pass
                if incl_intmd_llrs:
                    all_llrs = model(syndromes, return_intmd_llrs=True)
                    all_losses = [loss_fn(llrs, syndromes, observables)
                                  for llrs in all_llrs]
                    loss = torch.stack(all_losses).mean()
                else:
                    llrs = model(syndromes, return_intmd_llrs=False)
                    loss = loss_fn(llrs, syndromes, observables)
                dev_losses.append(loss.item())

        # Print epoch summary
        avg_train_loss = np.mean(train_losses)
        avg_dev_loss = np.mean(dev_losses)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Dev Loss: {avg_dev_loss:.6f}")
        print()

    return model.gamma
