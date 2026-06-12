from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from qecdec.circuits import create_circuit_with_uniform_error_rate
from torchdecoder_core.dataset import (
    DecodingDataset,
    StreamingDecodingDataset,
    sample_decoding_dataset,
)


SEED = 42

# functools.partial of a library function: picklable for DataLoader workers.
circuit_factory = partial(
    create_circuit_with_uniform_error_rate,
    "RotatedSurfaceCode_Phenom",
    d=3,
    rounds=3,
    basis="Z",
)


def make_dataset(
    error_rate: float = 0.01, shots_per_epoch: int = 256, base_seed: int = SEED
) -> StreamingDecodingDataset:
    return StreamingDecodingDataset(
        circuit_factory,
        error_rate=error_rate,
        shots_per_epoch=shots_per_epoch,
        base_seed=base_seed,
    )


def collect(iterable) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = list(iterable)
    syndromes = torch.stack([syn for syn, _ in pairs])
    observables = torch.stack([obs for _, obs in pairs])
    return syndromes, observables


def test_item_contract() -> None:
    ds = make_dataset(shots_per_epoch=16)
    syndromes, observables = collect(ds)
    assert syndromes.shape == (16, ds.circuit.num_detectors)
    assert observables.shape == (16, ds.circuit.num_observables)
    assert syndromes.dtype == observables.dtype == torch.int32
    assert ((syndromes == 0) | (syndromes == 1)).all()
    assert ((observables == 0) | (observables == 1)).all()


def test_determinism_under_fixed_seed() -> None:
    syn1, obs1 = collect(make_dataset())
    syn2, obs2 = collect(make_dataset())
    assert torch.equal(syn1, syn2)
    assert torch.equal(obs1, obs2)

    syn3, _ = collect(make_dataset(base_seed=SEED + 1))
    assert not torch.equal(syn1, syn3)


def test_epochs_differ() -> None:
    ds = make_dataset()
    syn1, _ = collect(ds)
    syn2, _ = collect(ds)
    assert not torch.equal(syn1, syn2)


def test_consistency_vs_direct_stim_sampling() -> None:
    shots = 512
    ds = make_dataset(shots_per_epoch=shots)
    syndromes, observables = collect(ds)

    # Epoch 0 of a single-process stream is seeded as below; the same stim
    # sampler must reproduce the exact (syndrome, observable) pairs.
    seed = int(
        np.random.SeedSequence(entropy=SEED, spawn_key=(0,)).generate_state(1)[0]
    )
    sampler = ds.circuit.stim_dem.compile_sampler(seed=seed)
    expected_syn, expected_obs, _ = sampler.sample(shots)
    assert np.array_equal(syndromes.numpy(), expected_syn.astype(np.int32))
    assert np.array_equal(observables.numpy(), expected_obs.astype(np.int32))

    # Natural distribution: trivial all-zero-syndrome shots are kept.
    assert (syndromes.sum(dim=1) == 0).any()


def test_error_rate_setter_rebuilds_circuit() -> None:
    ds = make_dataset(error_rate=0.01)
    circuit = ds.circuit
    ds.error_rate = 0.01  # unchanged: keep the cached circuit
    assert ds.circuit is circuit

    ds.error_rate = 0.05
    assert ds.error_rate == 0.05
    assert ds.circuit is not circuit
    assert ds.circuit.prior.mean() > circuit.prior.mean()

    # The stream now samples from the higher-rate DEM.
    syn_low, _ = collect(make_dataset(error_rate=0.01))
    syn_high, _ = collect(ds)
    assert syn_high.sum() > syn_low.sum()


def test_multiworker_dataloader() -> None:
    shots = 250  # not divisible by num_workers: sharding must still cover all
    ds = make_dataset(shots_per_epoch=shots)

    def run() -> list[bytes]:
        loader = DataLoader(
            ds,
            batch_size=32,
            num_workers=2,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(SEED),
        )
        rows = []
        for syn, obs in loader:
            rows += [bytes(s.numpy()) + bytes(o.numpy()) for s, o in zip(syn, obs)]
        return rows

    rows1 = run()
    assert len(rows1) == shots
    # Reproducible under the same DataLoader generator seed (compare as
    # multisets: batch interleaving across workers may vary).
    rows2 = run()
    assert sorted(rows1) == sorted(rows2)


def test_sample_decoding_dataset_reproducible() -> None:
    circuit = circuit_factory(0.01)
    ds1 = sample_decoding_dataset(circuit, shots=128, seed=SEED)
    ds2 = sample_decoding_dataset(circuit, shots=128, seed=SEED)
    assert isinstance(ds1, DecodingDataset)
    assert len(ds1) == 128
    assert torch.equal(ds1.syndromes, ds2.syndromes)
    assert torch.equal(ds1.observables, ds2.observables)
    assert ds1.syndromes.shape == (128, circuit.num_detectors)
    assert ds1.observables.shape == (128, circuit.num_observables)
