import pytest
import torch
import torch.nn as nn

from torchdecoder_core.models import LogicalDecoderModel


NUM_CHKS = 24
NUM_OBSERS = 3
BATCH_SIZE = 16
SEED = 42


class _DummyLogicalDecoder(LogicalDecoderModel):
    """Minimal concrete subclass: a single linear layer syndrome → logits."""

    def __init__(self, num_chks: int, num_obsers: int):
        super().__init__(num_chks, num_obsers)
        self.linear = nn.Linear(num_chks, num_obsers)

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        return self.linear(syndromes.float())


@pytest.fixture
def model() -> _DummyLogicalDecoder:
    torch.manual_seed(SEED)
    return _DummyLogicalDecoder(NUM_CHKS, NUM_OBSERS)


@pytest.fixture
def syndromes() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randint(0, 2, (BATCH_SIZE, NUM_CHKS), dtype=torch.int32)


def test_cannot_instantiate_abstract_base() -> None:
    with pytest.raises(TypeError):
        LogicalDecoderModel(NUM_CHKS, NUM_OBSERS)


@pytest.mark.parametrize("num_chks, num_obsers", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_invalid_init_args_raise(num_chks: int, num_obsers: int) -> None:
    with pytest.raises(ValueError):
        _DummyLogicalDecoder(num_chks, num_obsers)


def test_forward_shape_contract(model, syndromes) -> None:
    logits = model(syndromes)
    assert logits.shape == (BATCH_SIZE, NUM_OBSERS)
    assert logits.dtype == torch.float32
    assert model.num_chks == NUM_CHKS
    assert model.num_obsers == NUM_OBSERS


def test_load_lightning_checkpoint_roundtrip(model, syndromes, tmp_path) -> None:
    ckpt_path = tmp_path / "model.ckpt"
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict}, ckpt_path)

    torch.manual_seed(SEED + 1)
    other = _DummyLogicalDecoder(NUM_CHKS, NUM_OBSERS)
    assert not torch.equal(other.linear.weight, model.linear.weight)

    other.load_lightning_checkpoint(ckpt_path)
    torch.testing.assert_close(other.linear.weight, model.linear.weight)
    torch.testing.assert_close(other.linear.bias, model.linear.bias)
    torch.testing.assert_close(other(syndromes), model(syndromes))


def test_load_lightning_checkpoint_skip_keys(model, tmp_path) -> None:
    ckpt_path = tmp_path / "model.ckpt"
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict}, ckpt_path)

    torch.manual_seed(SEED + 1)
    other = _DummyLogicalDecoder(NUM_CHKS, NUM_OBSERS)
    original_weight = other.linear.weight.clone()

    other.load_lightning_checkpoint(ckpt_path, skip_keys=["linear.weight"])
    torch.testing.assert_close(other.linear.weight, original_weight)
    torch.testing.assert_close(other.linear.bias, model.linear.bias)


def test_load_lightning_checkpoint_missing_file(model, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        model.load_lightning_checkpoint(tmp_path / "nonexistent.ckpt")
