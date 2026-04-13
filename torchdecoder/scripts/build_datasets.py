"""
Python script to build datasets for training and testing.
The datasets will be saved in the torchdecoder/datasets directory.

Usage:
    uv run python build_datasets.py [options]

Options:
    -h, --help : Show help message and exit
    --force : Rebuild even if datasets already exist
"""

from pathlib import Path
import argparse

import numpy as np
from omegaconf import OmegaConf
from stim import CompiledDemSampler
from qecdec.decoders import MemBPDecoder
from qecdec.experiments import Experiment
from torchdecoder_core.dataset import DecodingDataset
from utils import create_experiment, get_circuit_dir

DATASETS_ROOT = Path(__file__).resolve().parent.parent / "datasets"


def find_config_dirs() -> list[Path]:
    """Find all subdirectories of DATASETS_ROOT containing config.yaml."""
    config_dirs: list[Path] = []
    for path in DATASETS_ROOT.rglob("config.yaml"):
        config_dirs.append(path.parent)
    return config_dirs


def sample_shots(
    sampler: CompiledDemSampler, shots: int
) -> tuple[np.ndarray, np.ndarray]:
    syndromes, observables, _ = sampler.sample(shots)
    return syndromes.astype(np.uint8), observables.astype(np.uint8)


def remove_trivial_shots(
    syndromes: np.ndarray, observables: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove shots that have all-zero syndrome.
    """
    mask = np.any(syndromes != 0, axis=1)
    return syndromes[mask], observables[mask]


def classify_hard_easy(
    syndromes: np.ndarray,
    observables: np.ndarray,
    *,
    chkmat: np.ndarray,
    prior: np.ndarray,
    membp_max_iter: int,
    membp_gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify shots as hard (MemBP does not converge) or easy (MemBP converges).
    Return (hard_syndromes, hard_observables, easy_syndromes, easy_observables).
    """
    bp = MemBPDecoder(chkmat, prior, gamma=membp_gamma, max_iter=membp_max_iter)
    ehat = bp.decode_batch(syndromes)
    synd_pred = (ehat @ chkmat.T) % 2
    hard_mask = np.any(synd_pred != syndromes, axis=1)
    return (
        syndromes[hard_mask],
        observables[hard_mask],
        syndromes[~hard_mask],
        observables[~hard_mask],
    )


def shuffle(
    syndromes: np.ndarray, observables: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle the syndromes and observables.
    """
    rng = np.random.default_rng()
    idx = rng.permutation(syndromes.shape[0])
    syndromes = syndromes[idx]
    observables = observables[idx]

    return syndromes, observables


def collect_dataset(
    *,
    expmt: Experiment,
    target_size: int,
    max_easy_sample_ratio: float,
    membp_max_iter: int,
    membp_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect a dataset of `target_size`. The fraction of easy samples (i.e., where
    MemBP converged within the specified number of iterations) must not exceed
    `max_easy_sample_ratio`, after filtering out trivial shots (i.e., shots with
    all-zero syndrome). Any extra easy samples are discarded.
    """
    max_easy_shots = int(target_size * max_easy_sample_ratio)

    syndromes_list: list[np.ndarray] = []
    observables_list: list[np.ndarray] = []
    collected_shots = 0
    easy_shots = 0

    sampler = expmt.dem.compile_sampler()
    while collected_shots < target_size:
        print(f"{collected_shots}/{target_size}")
        syn, obs = sample_shots(sampler, 1_000)
        syn, obs = remove_trivial_shots(syn, obs)

        if syn.shape[0] == 0:
            continue

        h_syn, h_obs, e_syn, e_obs = classify_hard_easy(
            syn,
            obs,
            chkmat=expmt.chkmat,
            prior=expmt.prior,
            membp_max_iter=membp_max_iter,
            membp_gamma=membp_gamma,
        )

        # Add all hard samples to the dataset
        if h_syn.shape[0] > 0:
            syndromes_list.append(h_syn)
            observables_list.append(h_obs)
            collected_shots += h_syn.shape[0]

        # Add easy samples to the dataset until we reach the max number allowed
        if e_syn.shape[0] > 0 and easy_shots < max_easy_shots:
            inc = min(e_syn.shape[0], max_easy_shots - easy_shots)
            syndromes_list.append(e_syn[:inc])
            observables_list.append(e_obs[:inc])
            collected_shots += inc
            easy_shots += inc

    syndromes = np.concatenate(syndromes_list, axis=0)
    observables = np.concatenate(observables_list, axis=0)
    syndromes, observables = shuffle(syndromes, observables)

    assert easy_shots <= max_easy_shots
    assert collected_shots == syndromes.shape[0] == observables.shape[0]
    assert collected_shots >= target_size

    print(f"Fraction of easy shots: {easy_shots / collected_shots:.2f}")

    # If oversampled, randomly select the target number of shots
    if collected_shots > target_size:
        rng = np.random.default_rng()
        idx = rng.choice(collected_shots, size=target_size, replace=False)
        syndromes = syndromes[idx]
        observables = observables[idx]

    assert target_size == syndromes.shape[0] == observables.shape[0]
    return syndromes, observables


def distribute_evenly(total: int, num_parts: int) -> list[int]:
    """
    Distribute `total` into `num_parts` as evenly as possible.
    Example: `distribute_evenly(20, 3)` returns `[7, 7, 6]`.
    """
    base = total // num_parts
    remainder = total % num_parts
    return [base + 1] * remainder + [base] * (num_parts - remainder)


def build_dataset_for_config(config_dir: Path, force: bool) -> None:
    """
    Build train/val/test datasets for a single config directory.

    If `force` is True, rebuild even if datasets already exist.
    """
    config_path = config_dir / "config.yaml"
    train_path = config_dir / "train_dataset.pt"
    val_path = config_dir / "val_dataset.pt"
    test_path = config_dir / "test_dataset.pt"

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    qec_cfg = cfg.qec
    code: str = qec_cfg.code
    noise_model: str = qec_cfg.noise_model
    d: int = qec_cfg.d
    rounds: int = qec_cfg.rounds
    basis: str = qec_cfg.basis
    p_range: list[float] = list(qec_cfg.p_range)
    train_size: int = cfg.data.train_size
    val_size: int = cfg.data.val_size
    test_size: int = cfg.data.test_size
    max_easy_sample_ratio: float = cfg.data.max_easy_sample_ratio
    membp_max_iter: int = cfg.data.membp_max_iter
    membp_gamma: float = cfg.data.membp_gamma

    # If the circuits should be loaded from file, validate that the files exist.
    if qec_cfg.load_circuit_from_file:
        circuit_dir = get_circuit_dir(code, noise_model, d, rounds, basis)
        for p in p_range:
            circuit_file = circuit_dir / f"error_rate={p}.stim"
            if not circuit_file.exists():
                raise FileNotFoundError(f"Missing circuit file: {circuit_file}")

    if not force and train_path.exists() and val_path.exists() and test_path.exists():
        print(f">>>>>> Skipping datasets inside {config_dir}.")
    else:
        print(f">>>>>> Building datasets inside {config_dir}.")

        total_size = train_size + val_size + test_size
        sizes_per_p = distribute_evenly(total_size, len(p_range))
        syndromes_per_p: list[np.ndarray] = []
        observables_per_p: list[np.ndarray] = []
        for p, target_size in zip(p_range, sizes_per_p):
            expmt = create_experiment(
                code, noise_model, d, rounds, basis, p, qec_cfg.load_circuit_from_file
            )
            syn, obs = collect_dataset(
                expmt=expmt,
                target_size=target_size,
                max_easy_sample_ratio=max_easy_sample_ratio,
                membp_max_iter=membp_max_iter,
                membp_gamma=membp_gamma,
            )
            syndromes_per_p.append(syn)
            observables_per_p.append(obs)

        syndromes = np.concatenate(syndromes_per_p, axis=0)
        observables = np.concatenate(observables_per_p, axis=0)
        syndromes, observables = shuffle(syndromes, observables)

        assert total_size == syndromes.shape[0] == observables.shape[0]

        # Split into train, val, and test
        train_syndromes = syndromes[:train_size]
        train_observables = observables[:train_size]
        val_syndromes = syndromes[train_size : train_size + val_size]
        val_observables = observables[train_size : train_size + val_size]
        test_syndromes = syndromes[train_size + val_size :]
        test_observables = observables[train_size + val_size :]

        # Save train, val, and test datasets
        train_dataset = DecodingDataset(train_syndromes, train_observables)
        val_dataset = DecodingDataset(val_syndromes, val_observables)
        test_dataset = DecodingDataset(test_syndromes, test_observables)
        train_dataset.save_to_file(train_path, overwrite_ok=True)
        val_dataset.save_to_file(val_path, overwrite_ok=True)
        test_dataset.save_to_file(test_path, overwrite_ok=True)
        print(f"Train dataset size: {len(train_dataset)}, saved to {train_path}.")
        print(f"Val dataset size: {len(val_dataset)}, saved to {val_path}.")
        print(f"Test dataset size: {len(test_dataset)}, saved to {test_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if datasets exist"
    )
    args = parser.parse_args()
    force = args.force

    config_dirs = find_config_dirs()
    if len(config_dirs) == 0:
        print(f"No config.yaml files found in {DATASETS_ROOT}")
        return

    for config_dir in config_dirs:
        build_dataset_for_config(config_dir, force=force)


if __name__ == "__main__":
    main()
