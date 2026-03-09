import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from qecdec import RotatedSurfaceCode_Memory

PYTORCH_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PYTORCH_ROOT / "datasets"

SKIP_IF_EXISTS = True # TODO: use CLI argument

def find_config_dirs() -> list[Path]:
    """Find all subdirectories containing config.yaml."""
    config_dirs: list[Path] = []
    for path in DATASETS_ROOT.rglob("config.yaml"):
        config_dirs.append(path.parent)
    return config_dirs


def sample_shots(expmt: RotatedSurfaceCode_Memory, num_shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = expmt.dem.compile_sampler(seed=seed)
    syndromes, observables, _ = sampler.sample(num_shots)
    return syndromes.astype(np.int32), observables.astype(np.int32)


def remove_trivial_syndrome_shots(syndromes: np.ndarray, observables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove shots that have all-zero syndrome.
    """
    mask = np.any(syndromes != 0, axis=1)
    return syndromes[mask], observables[mask]


def remove_bp_easy_shots(syndromes: np.ndarray, observables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove shots that are correctly decoded by vanilla BP decoder in at most 5 iterations.
    """
    # TODO: implement this
    raise NotImplementedError("Not implemented yet.")


def build_dataset_for_config(config_dir: Path) -> None:
    """Build train/val datasets for a single config directory if they do not exist."""
    config_path = config_dir / "config.yaml"
    train_path = config_dir / "train_dataset.pt"
    val_path = config_dir / "val_dataset.pt"

    if SKIP_IF_EXISTS and train_path.exists() and val_path.exists():
        print(f"Skipping {config_dir} (datasets already exist)")
        return
    print(f"Building datasets inside {config_dir}...")

    cfg = OmegaConf.load(config_path)
    qec_cfg = cfg.qec
    data_cfg = cfg.data
    seed = cfg.seed

    if qec_cfg.code == "RotatedSurfaceCode" and qec_cfg.noise_model == "Phenomenological":
        expmt = RotatedSurfaceCode_Memory(
            d=qec_cfg.d,
            rounds=qec_cfg.rounds,
            basis=qec_cfg.basis,
            data_qubit_error_rate=qec_cfg.p,
            meas_error_rate=qec_cfg.p,
        )
    else:
        raise ValueError(f"Unsupported combination: {qec_cfg.code} + {qec_cfg.noise_model}")

    # –––– sample shots from noisy circuit ––––
    raw_sample_shots = data_cfg.raw_sample_shots
    syndromes, observables = sample_shots(expmt, raw_sample_shots, seed)
    print(f"Sampled {raw_sample_shots} shots from the noisy circuit.")

    # –––– remove trivial syndromes ––––
    if data_cfg.remove_trivial_syndrome_shots:
        syndromes, observables = remove_trivial_syndrome_shots(syndromes, observables)
        print(f"Retained {len(syndromes)} shots after removing trivial syndrome shots.")

    # –––– remove BP easy shots ––––
    if data_cfg.remove_bp_easy_shots:
        syndromes, observables = remove_bp_easy_shots(syndromes, observables)
        print(f"Retained {len(syndromes)} shots after removing BP easy shots.")

    # –––– split dataset into train and val ––––
    train_size = int(len(syndromes) * data_cfg.split_ratio)
    train_dataset = DecodingDataset(syndromes[:train_size], observables[:train_size])
    val_dataset = DecodingDataset(syndromes[train_size:], observables[train_size:])
    print(f"Size of train dataset: {len(train_dataset)}")
    print(f"Size of val dataset: {len(val_dataset)}")

    # –––– save datasets ––––
    train_dataset.save_to_file(train_path, overwrite_ok=True)
    val_dataset.save_to_file(val_path, overwrite_ok=True)
    print(f"Saved datasets.")


def main():
    config_dirs = find_config_dirs()
    if len(config_dirs) == 0:
        print(f"No config.yaml files found in {DATASETS_ROOT}")
        return

    print(f"Found {len(config_dirs)} config directories")
    for config_dir in config_dirs:
        build_dataset_for_config(config_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(PYTORCH_ROOT))
    from src.dataset.dataset import DecodingDataset

    main()
