import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from torchdecoder_core.losses import LogicalBCELoss, LossResult, build_loss_fn


def test_zero_logits_give_log2() -> None:
    loss_fn = LogicalBCELoss()
    logits = torch.zeros(4, 3)
    observables = torch.randint(0, 2, (4, 3), dtype=torch.int32)
    result = loss_fn(logits, observables)
    torch.testing.assert_close(result.loss, torch.tensor(math.log(2.0)))


def test_matches_hand_computed_values() -> None:
    loss_fn = LogicalBCELoss()
    logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    observables = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)

    # BCE with sigmoid(logit) = Pr(obs = 1):
    # -log(p) if obs == 1 else -log(1 - p), averaged over all entries.
    p = torch.sigmoid(logits)
    expected = -(
        torch.log(p[0, 0]) + torch.log(1 - p[0, 1])
        + torch.log(1 - p[1, 0]) + torch.log(p[1, 1])
    ) / 4
    torch.testing.assert_close(loss_fn(logits, observables).loss, expected)


def test_result_fields() -> None:
    loss_fn = LogicalBCELoss()
    torch.manual_seed(0)
    logits = torch.randn(8, 12)
    observables = torch.randint(0, 2, (8, 12), dtype=torch.int32)
    result = loss_fn(logits, observables)

    assert isinstance(result, LossResult)
    assert result.loss.shape == ()
    assert result.synd_loss is None
    assert result.obser_loss.shape == ()
    torch.testing.assert_close(result.loss, result.obser_loss)


def test_gradients_flow() -> None:
    loss_fn = LogicalBCELoss()
    torch.manual_seed(0)
    logits = torch.randn(8, 3, requires_grad=True)
    observables = torch.randint(0, 2, (8, 3), dtype=torch.int32)
    loss_fn(logits, observables).loss.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)


def test_perfect_confident_prediction_gives_small_loss() -> None:
    loss_fn = LogicalBCELoss()
    observables = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    logits = (observables.float() * 2 - 1) * 100.0
    assert loss_fn(logits, observables).loss.item() < 1e-6


def test_factory_builds_logical_bce_loss() -> None:
    chkmat = np.eye(4, dtype=np.int64)
    obsmat = np.ones((1, 4), dtype=np.int64)
    loss_fn = build_loss_fn(chkmat, obsmat, OmegaConf.create({"name": "LogicalBCELoss"}))
    assert isinstance(loss_fn, LogicalBCELoss)


def test_factory_rejects_unknown_name() -> None:
    chkmat = np.eye(4, dtype=np.int64)
    obsmat = np.ones((1, 4), dtype=np.int64)
    with pytest.raises(ValueError):
        build_loss_fn(chkmat, obsmat, OmegaConf.create({"name": "NoSuchLoss"}))
