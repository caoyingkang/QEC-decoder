import torch

from torchdecoder_core.metrics import LogicalDecodingMetric


def test_hand_computed_values() -> None:
    metric = LogicalDecodingMetric()
    # Predictions (logit > 0):     [[1, 0], [0, 1], [1, 1], [0, 0]]
    logits = torch.tensor(
        [[2.0, -1.0], [-0.5, 3.0], [1.0, 0.1], [-2.0, -0.1]]
    )
    # Wrong predictions:           [[0, 0], [1, 0], [0, 1], [0, 0]]
    observables = torch.tensor(
        [[1, 0], [1, 1], [1, 0], [0, 0]], dtype=torch.int32
    )
    metric.update(logits, observables)
    result = metric.compute()

    # Shots 1 and 2 each have a wrong observable -> 2/4 shots fully correct.
    torch.testing.assert_close(result["logical_success_rate"], torch.tensor(0.5))


def test_zero_logit_predicts_no_flip() -> None:
    metric = LogicalDecodingMetric()
    logits = torch.tensor([[0.0], [0.0]])
    observables = torch.tensor([[0], [1]], dtype=torch.int32)
    metric.update(logits, observables)
    result = metric.compute()
    torch.testing.assert_close(result["logical_success_rate"], torch.tensor(0.5))


def test_perfect_prediction() -> None:
    metric = LogicalDecodingMetric()
    torch.manual_seed(0)
    observables = torch.randint(0, 2, (32, 3), dtype=torch.int32)
    logits = (observables.float() * 2 - 1) * 5.0
    metric.update(logits, observables)
    result = metric.compute()
    torch.testing.assert_close(result["logical_success_rate"], torch.tensor(1.0))


def test_single_wrong_observable_fails_whole_shot() -> None:
    metric = LogicalDecodingMetric()
    observables = torch.tensor([[1, 1, 1]], dtype=torch.int32)
    logits = torch.tensor([[5.0, 5.0, -5.0]])  # last observable predicted wrong
    metric.update(logits, observables)
    result = metric.compute()
    torch.testing.assert_close(result["logical_success_rate"], torch.tensor(0.0))


def test_accumulation_over_batches_matches_single_batch() -> None:
    torch.manual_seed(1)
    logits = torch.randn(64, 4)
    observables = torch.randint(0, 2, (64, 4), dtype=torch.int32)

    metric_single = LogicalDecodingMetric()
    metric_single.update(logits, observables)
    result_single = metric_single.compute()

    metric_batched = LogicalDecodingMetric()
    for chunk_logits, chunk_obs in zip(logits.split(16), observables.split(16)):
        metric_batched.update(chunk_logits, chunk_obs)
    result_batched = metric_batched.compute()

    torch.testing.assert_close(
        result_batched["logical_success_rate"], result_single["logical_success_rate"]
    )


def test_reset() -> None:
    metric = LogicalDecodingMetric()
    metric.update(torch.tensor([[5.0, 5.0]]), torch.tensor([[0, 0]], dtype=torch.int32))
    assert metric.compute()["logical_success_rate"].item() == 0.0
    metric.reset()
    metric.update(torch.tensor([[5.0, 5.0]]), torch.tensor([[1, 1]], dtype=torch.int32))
    assert metric.compute()["logical_success_rate"].item() == 1.0
