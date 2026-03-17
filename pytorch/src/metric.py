import numpy as np
import torch
from torchmetrics import Metric

from .utils.llr_utils import llrs_to_ehat


class IterativeDecodingMetric(Metric):
    """
    A `torchmetrics.Metric` that evaluates iterative decoder's performance.
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
        self.chkmat = torch.as_tensor(chkmat, dtype=torch.int32)
        self.obsmat = torch.as_tensor(obsmat, dtype=torch.int32)

        # Total number of shots
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # Number of shots where the decoder converged (i.e., syndrome matched)
        self.add_state("converged", default=torch.tensor(0), dist_reduce_fx="sum")
        # Number of shots where the decoder predicted observables correctly
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        # Number of shots where the decoder converged and predicted observables correctly
        self.add_state("converged_and_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        # Sum of the number of decoding iterations for all shots
        self.add_state("iters_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        # Sum of the number of decoding iterations for all converged shots
        self.add_state("iters_sum_on_converged", default=torch.tensor(0), dist_reduce_fx="sum")

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
        ehat, converged_mask, output_iters = llrs_to_ehat(llrs, syndromes, self.chkmat)
        decoding_iters = output_iters + 1  # (batch_size,), int

        # For each shot, check if the decoder predicts the observables correctly
        obser_pred = torch.matmul(ehat, self.obsmat.T) % 2  # (batch_size, num_obsers), int ∈ {0,1}
        correct_mask = torch.all(obser_pred == observables, dim=1)  # (batch_size,), bool

        # Update states
        self.total += batch_size
        self.converged += torch.sum(converged_mask)
        self.correct += torch.sum(correct_mask)
        self.converged_and_correct += torch.sum(converged_mask & correct_mask)
        self.iters_sum += torch.sum(decoding_iters)
        self.iters_sum_on_converged += torch.sum(decoding_iters[converged_mask])

    def compute(self) -> dict[str, float]:
        return {
            "convergence_rate": self.converged.float() / self.total,
            "logical_success_rate": self.correct.float() / self.total,
            "strict_success_rate": self.converged_and_correct.float() / self.total,
            "accidental_success_rate": (self.correct - self.converged_and_correct).float() / self.total,
            "success_rate_on_convergence": self.converged_and_correct.float() / self.converged,
            "avg_iters": self.iters_sum.float() / self.total,
            "avg_iters_on_convergence": self.iters_sum_on_converged.float() / self.converged,
        }
