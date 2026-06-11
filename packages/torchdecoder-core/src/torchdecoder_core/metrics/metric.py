import numpy as np
import torch
from torchmetrics import Metric

from ..utils.decoding_utils import diagnose_convergence, gather_ehat
from ..utils.tensor_utils import matmul_GF2


# Shape hints code:
# B = batch_size
# C = num_chks
# O = num_obsers
# V = num_vars
# I = num_iters


class IterativeDecodingMetric(Metric):
    """
    A `torchmetrics.Metric` that evaluates iterative decoder's performance.
    """

    def __init__(self, chkmat: np.ndarray, obsmat: np.ndarray):
        """
        Parameters
        ----------
            chkmat : ndarray
                Check matrix, shape=(num_chks, num_vars), integer ∈ {0,1} or bool

            obsmat : ndarray
                Observable matrix, shape=(num_obsers, num_vars), integer ∈ {0,1} or bool
        """
        super().__init__()

        self.register_buffer(
            "chkmat", torch.tensor(chkmat, dtype=torch.float32), persistent=False
        )  # (C, V)
        self.register_buffer(
            "obsmat", torch.tensor(obsmat, dtype=torch.float32), persistent=False
        )  # (O, V)

        # Total number of shots
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # Number of shots where the decoder converged (i.e., syndrome matched)
        self.add_state("converged", default=torch.tensor(0), dist_reduce_fx="sum")
        # Number of shots where the decoder predicted observables correctly
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        # Number of shots where the decoder converged and predicted observables correctly
        self.add_state(
            "converged_and_correct", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        # Sum of the number of decoding iterations for all shots
        self.add_state("iters_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        # Sum of the number of decoding iterations for all converged shots
        self.add_state(
            "iters_sum_on_converged", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(
        self, llrs: torch.Tensor, syndromes: torch.Tensor, observables: torch.Tensor
    ):
        """
        Parameters
        ----------
            llrs : torch.Tensor
                LLR output from all iterations, shape=(num_iters, batch_size, num_vars), float

            syndromes : torch.Tensor
                Syndrome bits, shape=(batch_size, num_chks), int ∈ {0,1}

            observables : torch.Tensor
                Observable bits, shape=(batch_size, num_obsers), int ∈ {0,1}
        """
        hard_decisions = (llrs < 0).float()  # (I, B, V), float ∈ {0.0, 1.0}
        converged_mask, output_iters = diagnose_convergence(
            hard_decisions, syndromes, self.chkmat
        )  # (B,), bool; (B,), long
        ehat = gather_ehat(hard_decisions, output_iters)  # (B, V), float ∈ {0.0, 1.0}
        decoding_iters = output_iters + 1  # (B,), long

        # For each shot, check if the decoder predicts the observables correctly
        obser_pred = matmul_GF2(ehat, self.obsmat.T)  # (B, O), int ∈ {0,1}
        correct_mask = torch.all(obser_pred == observables, dim=1)  # (B,), bool

        # Update states
        self.total += syndromes.size(0)
        self.converged += torch.sum(converged_mask)
        self.correct += torch.sum(correct_mask)
        self.converged_and_correct += torch.sum(converged_mask & correct_mask)
        self.iters_sum += torch.sum(decoding_iters)
        self.iters_sum_on_converged += torch.sum(decoding_iters[converged_mask])

    def compute(self) -> dict[str, float]:
        return {
            "convergence_rate": self.converged.float() / self.total.float(),
            "logical_success_rate": self.correct.float() / self.total.float(),
            "strict_success_rate": self.converged_and_correct.float()
            / self.total.float(),
            "accidental_success_rate": (
                self.correct - self.converged_and_correct
            ).float()
            / self.total.float(),
            "success_rate_on_convergence": self.converged_and_correct.float()
            / self.converged.float(),
            "avg_iters": self.iters_sum.float() / self.total.float(),
            "avg_iters_on_convergence": self.iters_sum_on_converged.float()
            / self.converged.float(),
        }
