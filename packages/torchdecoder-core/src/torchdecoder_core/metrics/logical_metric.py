import torch
from torchmetrics import Metric


# Shape hints code:
# B = batch_size
# O = num_obsers


class LogicalDecodingMetric(Metric):
    """
    A `torchmetrics.Metric` that evaluates a logical decoder model's performance.

    The model predicts logical-observable logits with the convention
    `sigmoid(logit) = Pr(observable flipped)`; a logit > 0 predicts a flip.
    A shot counts as correct only if all observables are predicted correctly.
    """

    def __init__(self):
        super().__init__()
        # Total number of shots
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # Number of shots where the decoder predicted observables correctly
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, observables: torch.Tensor):
        """
        Parameters
        ----------
            logits : torch.Tensor
                Logical-observable logits, shape=(batch_size, num_obsers), float

            observables : torch.Tensor
                Observable bits, shape=(batch_size, num_obsers), int ∈ {0,1}
        """
        predictions = (logits > 0).int()  # (B, O), int
        correct_mask = torch.all(predictions == observables, dim=1)  # (B,), bool

        self.total += logits.size(0)
        self.correct += torch.sum(correct_mask)

    def compute(self) -> dict[str, float]:
        return {"logical_success_rate": self.correct.float() / self.total.float()}
