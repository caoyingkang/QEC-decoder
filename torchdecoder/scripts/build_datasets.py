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
from omegaconf import DictConfig, OmegaConf
from stim import CompiledDemSampler
from qecdec.decoders import BPDecoder
from torchdecoder_core.dataset import DecodingDataset
from utils import create_experiment, get_stim_dir

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
    bp_max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify shots as hard (BP does not converge) or easy (BP converges).
    Return (hard_syndromes, hard_observables, easy_syndromes, easy_observables).
    """
    bp = BPDecoder(chkmat, prior, max_iter=bp_max_iter)
    ehat = bp.decode_batch(syndromes)
    synd_pred = (ehat @ chkmat.T) % 2
    hard_mask = np.any(synd_pred != syndromes, axis=1)
    return (
        syndromes[hard_mask],
        observables[hard_mask],
        syndromes[~hard_mask],
        observables[~hard_mask],
    )


def collect_dataset(
    *,
    qec_cfg: DictConfig,
    p_range: list[float],
    target_size: int,
    hard_sample_ratio: float,
    bp_max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect a dataset of `target_size`, about `hard_sample_ratio` fraction of which
    are hard (i.e., BP does not converge within `bp_max_iter` iterations), after
    filtering out trivial shots (i.e., shots with all-zero syndrome). The returned
    syndromes and observables are shuffled.
    """
    target_hard = int(target_size * hard_sample_ratio)
    target_easy = target_size - target_hard

    hard_syndromes_list: list[np.ndarray] = []
    hard_observables_list: list[np.ndarray] = []
    hard_shots = 0
    easy_syndromes_list: list[np.ndarray] = []
    easy_observables_list: list[np.ndarray] = []
    easy_shots = 0

    # Collect shots until we have the target number of hard and easy shots
    while hard_shots < target_hard or easy_shots < target_easy:
        synd_list: list[np.ndarray] = []
        obs_list: list[np.ndarray] = []
        for p in p_range:
            expmt = create_experiment(qec_cfg, p)
            sampler = expmt.dem.compile_sampler()
            s, o, _ = sampler.sample(10000)
            synd_list.append(s)
            obs_list.append(o)
        syn = np.concatenate(synd_list, axis=0, dtype=np.uint8)
        obs = np.concatenate(obs_list, axis=0, dtype=np.uint8)
        syn, obs = remove_trivial_shots(syn, obs)

        if syn.shape[0] > 0:
            h_syn, h_obs, e_syn, e_obs = classify_hard_easy(
                syn,
                obs,
                chkmat=expmt.chkmat,
                prior=expmt.prior,
                bp_max_iter=bp_max_iter,
            )
            if hard_shots < target_hard and h_syn.shape[0] > 0:
                hard_syndromes_list.append(h_syn)
                hard_observables_list.append(h_obs)
                hard_shots += h_syn.shape[0]
            if easy_shots < target_easy and e_syn.shape[0] > 0:
                easy_syndromes_list.append(e_syn)
                easy_observables_list.append(e_obs)
                easy_shots += e_syn.shape[0]

    hard_syndromes = np.concatenate(hard_syndromes_list, axis=0)
    hard_observables = np.concatenate(hard_observables_list, axis=0)
    easy_syndromes = np.concatenate(easy_syndromes_list, axis=0)
    easy_observables = np.concatenate(easy_observables_list, axis=0)

    assert hard_shots == hard_syndromes.shape[0] == hard_observables.shape[0]
    assert easy_shots == easy_syndromes.shape[0] == easy_observables.shape[0]

    # If oversampled, randomly select the target number of shots
    rng = np.random.default_rng()
    if hard_shots > target_hard:
        idx = rng.choice(hard_shots, size=target_hard, replace=False)
        hard_syndromes = hard_syndromes[idx]
        hard_observables = hard_observables[idx]
    if easy_shots > target_easy:
        idx = rng.choice(easy_shots, size=target_easy, replace=False)
        easy_syndromes = easy_syndromes[idx]
        easy_observables = easy_observables[idx]

    # Concatenate hard and easy shots
    syndromes = np.concatenate([hard_syndromes, easy_syndromes], axis=0)
    observables = np.concatenate([hard_observables, easy_observables], axis=0)
    assert target_size == syndromes.shape[0] == observables.shape[0]

    # Shuffle the shots
    idx = rng.permutation(target_size)
    syndromes = syndromes[idx]
    observables = observables[idx]

    return syndromes, observables


def sample_test_dataset(
    *,
    qec_cfg: DictConfig,
    p_range: list[float],
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect a test dataset by randomly sampling from the circuit. No filtering is applied.
    Sample evenly across `p_range`: each p gets approximately `target_size // len(p_range)` shots.
    """
    shots_per_p = target_size // len(p_range)
    remainder = target_size % len(p_range)
    synd_list: list[np.ndarray] = []
    obs_list: list[np.ndarray] = []
    for i, p in enumerate(p_range):
        shots = shots_per_p + (1 if i < remainder else 0)
        expmt = create_experiment(qec_cfg, p)
        sampler = expmt.dem.compile_sampler()
        synd, obs = sample_shots(sampler, shots)
        synd_list.append(synd)
        obs_list.append(obs)
    syndromes = np.concatenate(synd_list, axis=0)
    observables = np.concatenate(obs_list, axis=0)
    return syndromes, observables


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
    p_range: list[float] = list(qec_cfg.p_range)
    train_size = cfg.data.train_size
    val_size = cfg.data.val_size
    test_size = cfg.data.test_size
    hard_sample_ratio = cfg.data.hard_sample_ratio
    bp_max_iter = cfg.data.bp_max_iter

    # For stim-file experiments, validate that all p_range values have a circuit file.
    if qec_cfg.code != "RotatedSurfaceCode":
        stim_dir = get_stim_dir(qec_cfg)
        for p in p_range:
            circuit_file = stim_dir / f"error_rate={p}.stim"
            if not circuit_file.exists():
                raise FileNotFoundError(
                    f"Missing circuit file for p={p}: {circuit_file}"
                )

    if not force and train_path.exists() and val_path.exists():
        print(f">>>>>> Skipping train and val datasets inside {config_dir}.")
    else:
        print(f">>>>>> Building train and val datasets inside {config_dir}.")

        # Collect train_size + val_size shots, about hard_sample_ratio fraction of which are hard
        syndromes, observables = collect_dataset(
            qec_cfg=qec_cfg,
            p_range=p_range,
            target_size=train_size + val_size,
            hard_sample_ratio=hard_sample_ratio,
            bp_max_iter=bp_max_iter,
        )

        assert train_size + val_size == syndromes.shape[0] == observables.shape[0]

        # Split into train and val
        train_syndromes = syndromes[:train_size]
        train_observables = observables[:train_size]
        val_syndromes = syndromes[train_size:]
        val_observables = observables[train_size:]

        # Save train and val datasets
        train_dataset = DecodingDataset(train_syndromes, train_observables)
        val_dataset = DecodingDataset(val_syndromes, val_observables)
        train_dataset.save_to_file(train_path, overwrite_ok=True)
        val_dataset.save_to_file(val_path, overwrite_ok=True)
        print(f"Train dataset size: {len(train_dataset)}, saved to {train_path}.")
        print(f"Val dataset size: {len(val_dataset)}, saved to {val_path}.")

    if not force and test_path.exists():
        print(f">>>>>> Skipping test dataset inside {config_dir}.")
    else:
        print(f">>>>>> Building test dataset inside {config_dir}.")

        test_syndromes, test_observables = sample_test_dataset(
            qec_cfg=qec_cfg,
            p_range=p_range,
            target_size=test_size,
        )
        test_dataset = DecodingDataset(test_syndromes, test_observables)
        test_dataset.save_to_file(test_path, overwrite_ok=True)
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
