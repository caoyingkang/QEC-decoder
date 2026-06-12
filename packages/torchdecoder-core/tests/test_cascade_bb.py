from functools import cache

import pytest
import torch
from omegaconf import OmegaConf

from qecdec.circuits import BBCode_Circuit
from torchdecoder_core.models import (
    BBCodeGeometry,
    BipartiteTorusConv,
    Cascade,
    build_logical_decoder_model,
)


BATCH_SIZE = 7
SEED = 42

# The [[72,12,6]] code: A = x^3 + y + y^2, B = y^3 + x + x^2 on a 6x6 torus.
SMALL = dict(
    ell=6, m=6, a_poly=((3, 0), (0, 1), (0, 2)), b_poly=((0, 3), (1, 0), (2, 0))
)


@cache
def make_geometry(basis: str = "Z") -> BBCodeGeometry:
    circuit = BBCode_Circuit(**SMALL, basis=basis, rounds=2, error_rate=0.003)
    return BBCodeGeometry(circuit)


def make_conv(channels: int = 4) -> BipartiteTorusConv:
    geom = make_geometry()
    torch.manual_seed(SEED)
    return BipartiteTorusConv(
        channels, geom.check_dataL_offsets, geom.check_dataR_offsets
    )


def make_model(hidden_dim: int = 8, num_blocks: int = 2, basis: str = "Z") -> Cascade:
    torch.manual_seed(SEED)
    return Cascade(make_geometry(basis), hidden_dim=hidden_dim, num_blocks=num_blocks)


def test_bipartite_conv_shape_contract() -> None:
    geom = make_geometry()
    conv = make_conv(channels=5)
    torch.manual_seed(SEED)
    x = torch.randn(BATCH_SIZE, 5, 4, geom.ell, geom.m)
    # The intermediate check->data step lives on the two data planes.
    mid = conv.check_to_data(x)
    assert mid.shape == (BATCH_SIZE, 5, 4, 2, geom.ell, geom.m)
    y = conv(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("shift", [(1, 0), (0, 1), (3, 2)])
def test_bipartite_conv_torus_equivariance(shift: tuple[int, int]) -> None:
    geom = make_geometry()
    conv = make_conv()
    torch.manual_seed(SEED)
    x = torch.randn(2, 4, 5, geom.ell, geom.m)
    with torch.no_grad():
        y = conv(x)
        y_shifted = conv(torch.roll(x, shifts=shift, dims=(3, 4)))
    torch.testing.assert_close(y_shifted, torch.roll(y, shifts=shift, dims=(3, 4)))


def test_bipartite_conv_temporal_receptive_field() -> None:
    # The two bipartite steps' temporal taps ({-1, 0} then {0, +1}) compose to
    # the symmetric {-1, 0, +1} window of the dense check->check convolution.
    geom = make_geometry()
    conv = make_conv()
    with torch.no_grad():
        conv.check_to_data.bias.zero_()
        conv.data_to_check.bias.zero_()
        torch.manual_seed(SEED)
        x = torch.zeros(1, 4, 7, geom.ell, geom.m)
        x[:, :, 3] = torch.randn(1, 4, geom.ell, geom.m)
        y = conv(x)
    nonzero_layers = y.abs().amax(dim=(0, 1, 3, 4)) > 0
    assert nonzero_layers.tolist() == [False, False, True, True, True, False, False]


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_forward_shape_contract(basis: str) -> None:
    geom = make_geometry(basis)
    model = make_model(basis=basis)
    assert model.num_chks == geom.num_detectors
    assert model.num_obsers == geom.num_observables == 12

    model.eval()
    torch.manual_seed(SEED)
    syndromes = torch.randint(0, 2, (BATCH_SIZE, model.num_chks), dtype=torch.int32)
    logits = model(syndromes)
    assert logits.shape == (BATCH_SIZE, model.num_obsers)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_backbone_torus_equivariance() -> None:
    # Embedding and pooling aside, the whole conv stack commutes with spatial
    # torus translations (exactly: there is no spatial padding on the torus).
    model = make_model()
    model.eval()
    torch.manual_seed(SEED)
    x = torch.randn(1, model.hidden_dim, 3, 6, 6)
    with torch.no_grad():
        y = model._backbone(x)  # (B, H, T, P, ell, m)
        y_shifted = model._backbone(torch.roll(x, shifts=(2, 1), dims=(3, 4)))
    torch.testing.assert_close(y_shifted, torch.roll(y, shifts=(2, 1), dims=(4, 5)))


def test_overfit_tiny_batch() -> None:
    model = make_model(hidden_dim=32)
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
    model = make_model()
    state_dict = model.state_dict()
    for key in (
        "detector_flat_indices",
        "detector_spacetime_mask",
        "observable_pool_weights",
        "blocks.0.net.5.check_to_data.offsets",
        "blocks.0.net.5.data_to_check.offsets",
        "scatter_conv.2.offsets",
    ):
        assert key in state_dict

    other = make_model()
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


def test_rejects_bad_geometry_type() -> None:
    with pytest.raises(TypeError):
        Cascade("not a geometry", hidden_dim=8, num_blocks=2)


def test_factory_builds_bb_cascade() -> None:
    circuit = BBCode_Circuit(**SMALL, basis="Z", rounds=2, error_rate=0.003)
    model_cfg = OmegaConf.create({"name": "Cascade", "H": 8, "L": 2, "bottleneck": 4})
    model = build_logical_decoder_model(circuit, model_cfg)
    assert isinstance(model, Cascade)
    assert isinstance(model.blocks[0].net[5], BipartiteTorusConv)
    assert model.num_chks == circuit.num_detectors
    assert model.num_obsers == circuit.num_observables
