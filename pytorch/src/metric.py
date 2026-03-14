import numpy as np
import torch
from torchmetrics import Metric

from .utils.llr_utils import llrs_to_ehat
from .utils.tensor_utils import INT_DTYPE


class DecodingMetric(Metric):
    """
    A PyTorch Metric that calculates decoding performance metrics on CPU.
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
                Check matrix, shape=(num_chks, num_vars), integer ∈ {0,1} or bool

            obsmat : ndarray
                Observable matrix, shape=(num_obsers, num_vars), integer ∈ {0,1} or bool
        """
        super().__init__()
        self.chkmat = torch.tensor(chkmat, dtype=INT_DTYPE)
        self.obsmat = torch.tensor(obsmat, dtype=INT_DTYPE)

        self.add_state("wrong_syndrome", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("wrong_observable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("wrong_either", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor
    ):
        """
        Parameters
        ----------
            llrs : torch.Tensor
                LLR values at all iterations, shape=(num_iters, batch_size, num_vars), float, device=cpu

            syndromes : torch.Tensor
                Syndrome bits, shape=(batch_size, num_chks), int ∈ {0,1}, device=cpu

            observables : torch.Tensor
                Observable bits, shape=(batch_size, num_obsers), int ∈ {0,1}, device=cpu
        """
        batch_size = syndromes.size(0)
        ehat, converged_mask = llrs_to_ehat(llrs, syndromes, self.chkmat)

        # For each shot, check if the decoder predicts the observables correctly
        obs_pred = torch.matmul(ehat, self.obsmat.T) % 2  # (batch_size, num_obsers), int ∈ {0,1}
        obs_correct_mask = torch.all(obs_pred == observables, dim=1)  # (batch_size,), bool

        # Update states
        self.wrong_syndrome = self.wrong_syndrome + torch.sum(~converged_mask)
        self.wrong_observable = self.wrong_observable + torch.sum(~obs_correct_mask)
        self.wrong_either = self.wrong_either + torch.sum(~converged_mask | ~obs_correct_mask)
        self.total = self.total + batch_size

    def compute(self) -> dict[str, float]:
        return {
            "wrong_syndrome_rate": self.wrong_syndrome.float() / self.total.float(),
            "wrong_observable_rate": self.wrong_observable.float() / self.total.float(),
            "wrong_either_rate": self.wrong_either.float() / self.total.float(),
        }
