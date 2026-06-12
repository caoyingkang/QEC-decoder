from functools import cache

import pytest
import torch
from omegaconf import OmegaConf

from qecdec.circuits import CIRCUITS_REGISTRY
from torchdecoder_core.models import (
    Cascade,
    SurfaceCodeGeometry,
    build_logical_decoder_model,
)


BATCH_SIZE = 7
SEED = 42

CIRCUIT_CASES = [
    ("RotatedSurfaceCode_Phenom", 3, "Z"),
    ("RotatedSurfaceCode_Phenom", 5, "Z"),
    ("RotatedSurfaceCode_Circuit", 3, "Z"),
]


@cache
def make_geometry(circuit_name: str, d: int, basis: str) -> SurfaceCodeGeometry:
    circuit = CIRCUITS_REGISTRY[circuit_name].with_uniform_error_rate(
        0.01, d=d, rounds=d, basis=basis
    )
    return SurfaceCodeGeometry(circuit)


def make_model(
    circuit_name: str, d: int, basis: str, hidden_dim: int = 8, num_blocks: int = 2
) -> Cascade:
    torch.manual_seed(SEED)
    return Cascade(
        make_geometry(circuit_name, d, basis),
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    )


@pytest.mark.parametrize("circuit_name, d, basis", CIRCUIT_CASES)
def test_forward_shape_contract(circuit_name: str, d: int, basis: str) -> None:
    geom = make_geometry(circuit_name, d, basis)
    model = make_model(circuit_name, d, basis)
    assert model.num_chks == geom.num_detectors
    assert model.num_obsers == geom.num_observables

    model.eval()
    torch.manual_seed(SEED)
    syndromes = torch.randint(0, 2, (BATCH_SIZE, model.num_chks), dtype=torch.int32)
    logits = model(syndromes)
    assert logits.shape == (BATCH_SIZE, model.num_obsers)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_forward_rejects_bad_shape() -> None:
    model = make_model("RotatedSurfaceCode_Phenom", 3, "Z")
    with pytest.raises(ValueError):
        model(torch.zeros(BATCH_SIZE, model.num_chks + 1, dtype=torch.int32))
    with pytest.raises(ValueError):
        model(torch.zeros(model.num_chks, dtype=torch.int32))


def test_rejects_bad_config() -> None:
    geom = make_geometry("RotatedSurfaceCode_Phenom", 3, "Z")
    with pytest.raises(ValueError):
        Cascade(geom, hidden_dim=8, num_blocks=0)
    with pytest.raises(ValueError):
        Cascade(geom, hidden_dim=6, num_blocks=2, bottleneck=4)


def test_embedding_zero_where_no_detector() -> None:
    model = make_model("RotatedSurfaceCode_Phenom", 3, "Z")
    torch.manual_seed(SEED)
    syndromes = torch.randint(0, 2, (BATCH_SIZE, model.num_chks), dtype=torch.int32)
    x = model._embed(syndromes)  # (B, H, T, R, C)
    off_sites = model.detector_spacetime_mask == 0
    assert off_sites.any()
    assert (x * off_sites).abs().sum() == 0
    # Detector sites carry a nonzero embedding even for bit 0.
    assert (x * ~off_sites).abs().sum() > 0


def test_backbone_translation_equivariance_in_bulk() -> None:
    # The conv stack (blocks + scatter conv) is built from translation-equivariant
    # ops, so shifting the input shifts the output at positions whose receptive
    # field never touches the zero padding. The stack is size-agnostic, so run it
    # on a synthetic grid large enough to have such a deep interior: with
    # num_blocks + 1 = 3 spatial convs of kernel 3, that is every position at
    # least 3 sites from each boundary.
    model = make_model("RotatedSurfaceCode_Phenom", 5, "Z")
    model.eval()
    torch.manual_seed(SEED)
    x = torch.randn(1, model.hidden_dim, 10, 10, 10)
    x_shifted = torch.roll(x, shifts=(1, 1, 1), dims=(2, 3, 4))

    with torch.no_grad():
        y = model._backbone(x)
        y_shifted = model._backbone(x_shifted)
    torch.testing.assert_close(
        y_shifted[:, :, 4:7, 4:7, 4:7], y[:, :, 3:6, 3:6, 3:6]
    )


def test_overfit_tiny_batch() -> None:
    model = make_model("RotatedSurfaceCode_Phenom", 3, "Z", hidden_dim=16)
    torch.manual_seed(SEED)
    syndromes = torch.randint(0, 2, (16, model.num_chks), dtype=torch.int32)
    observables = torch.randint(0, 2, (16, model.num_obsers)).float()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = None
    for _ in range(600):
        optimizer.zero_grad()
        loss = loss_fn(model(syndromes), observables)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.02:
            break
    assert loss is not None and loss.item() < 0.05


def test_state_dict_is_self_contained() -> None:
    model = make_model("RotatedSurfaceCode_Phenom", 3, "Z")
    state_dict = model.state_dict()
    for key in (
        "detector_flat_indices",
        "detector_spacetime_mask",
        "observable_pool_weights",
    ):
        assert key in state_dict

    other = make_model("RotatedSurfaceCode_Phenom", 3, "Z", hidden_dim=8)
    # Perturb, then restore from the first model's state dict.
    with torch.no_grad():
        for p in other.parameters():
            p.add_(1.0)
    other.load_state_dict(state_dict, strict=True)

    model.eval()
    other.eval()
    torch.manual_seed(SEED)
    syndromes = torch.randint(0, 2, (BATCH_SIZE, model.num_chks), dtype=torch.int32)
    with torch.no_grad():
        torch.testing.assert_close(other(syndromes), model(syndromes))


def test_factory_builds_cascade() -> None:
    circuit = CIRCUITS_REGISTRY["RotatedSurfaceCode_Phenom"].with_uniform_error_rate(
        0.01, d=3, rounds=3, basis="Z"
    )
    model_cfg = OmegaConf.create({"name": "Cascade", "H": 8, "L": 2, "bottleneck": 4})
    model = build_logical_decoder_model(circuit, model_cfg)
    assert isinstance(model, Cascade)
    assert model.hidden_dim == 8
    assert model.num_blocks == 2
    assert model.num_chks == circuit.num_detectors
    assert model.num_obsers == circuit.num_observables


def test_factory_rejects_unknown_name() -> None:
    circuit = CIRCUITS_REGISTRY["RotatedSurfaceCode_Phenom"].with_uniform_error_rate(
        0.01, d=3, rounds=3, basis="Z"
    )
    with pytest.raises(ValueError):
        build_logical_decoder_model(circuit, OmegaConf.create({"name": "Nope"}))
