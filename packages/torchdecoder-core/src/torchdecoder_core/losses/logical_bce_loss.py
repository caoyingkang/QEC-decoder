"""BCE loss for logical decoder models that predict observable logits directly."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossResult


class LogicalBCELoss(nn.Module):
    """
    Binary cross-entropy loss for logical decoder models.

    The model outputs one logit per logical observable with the convention
    `sigmoid(logit) = Pr(observable flipped)` (note: opposite sign of the LLR
    convention used by the iterative decoding losses). The loss is
    `binary_cross_entropy_with_logits(logits, observables)`, averaged over
    observables and shots.

    Returns a `LossResult` for compatibility with the iterative-decoder training
    plumbing: `loss = obser_loss` and `synd_loss = None` (logical decoders do not
    predict syndromes).
    """

    def forward(self, logits: torch.Tensor, observables: torch.Tensor) -> LossResult:
        """
        Parameters
        ----------
            logits : torch.Tensor
                Logical-observable logits, shape=(batch_size, num_obsers), float

            observables : torch.Tensor
                Observable bits, shape=(batch_size, num_obsers), int ∈ {0,1}

        Returns
        -------
            LossResult
                Named tuple with fields `loss`, `synd_loss`, `obser_loss` (all float scalar tensors).
        """
        obser_loss = F.binary_cross_entropy_with_logits(logits, observables.float())
        return LossResult(loss=obser_loss, synd_loss=None, obser_loss=obser_loss)
