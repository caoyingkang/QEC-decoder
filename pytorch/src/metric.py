import numpy as np
import torch
from torchmetrics import Metric

INT_DTYPE = torch.int32


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
        num_iters, batch_size, num_vars = llrs.shape
        hard_decisions = (llrs < 0).to(INT_DTYPE)

        # For each shot, check if the decoder converges, i.e., whether the syndrome is matched at any iteration
        synd_pred = torch.matmul(hard_decisions, self.chkmat.T) % 2  # (num_iters, batch_size, num_chks), int ∈ {0,1}
        synd_matched_mask = torch.all(synd_pred == syndromes.unsqueeze(dim=0), dim=2)  # (num_iters, batch_size), bool
        converged_mask = torch.any(synd_matched_mask, dim=0)  # (batch_size,), bool

        # For each shot, find which iteration is the overall output of the decoder:
        # If the decoder converges, this is the first iteration where the syndrome is matched;
        # If the decoder does not converge, this is the last iteration.
        output_iters = torch.where(
            converged_mask,
            synd_matched_mask.int().argmax(dim=0),
            num_iters - 1
        )  # (batch_size,), int

        # Get the output error pattern for each shot
        index = output_iters.reshape(1, batch_size, 1).expand(1, batch_size, num_vars)
        ehat = torch.gather(hard_decisions, dim=0, index=index).squeeze(0)  # (batch_size, num_vars), int ∈ {0,1}

        # For each shot, check if the decoder predicts the observables correctly
        obs_pred = torch.matmul(ehat, self.obsmat.T) % 2  # (batch_size, num_obsers), int ∈ {0,1}
        obs_correct_mask = torch.all(obs_pred == observables, dim=1)  # (batch_size,), bool

        # Update states
        self.wrong_syndrome += torch.sum(~converged_mask)
        self.wrong_observable += torch.sum(~obs_correct_mask)
        self.wrong_either += torch.sum(~converged_mask | ~obs_correct_mask)
        self.total += batch_size

    def compute(self) -> dict[str, float]:
        return {
            "wrong_syndrome_rate": self.wrong_syndrome.float() / self.total.float(),
            "wrong_observable_rate": self.wrong_observable.float() / self.total.float(),
            "wrong_either_rate": self.wrong_either.float() / self.total.float(),
        }
