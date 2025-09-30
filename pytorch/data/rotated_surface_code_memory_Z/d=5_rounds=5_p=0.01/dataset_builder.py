import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
    from qecdec import RotatedSurfaceCode_Memory
    from train_utils import build_datasets_version_1

    expmt = RotatedSurfaceCode_Memory(
        d=5,
        rounds=5,
        basis='Z',
        data_qubit_error_rate=0.01,
        meas_error_rate=0.01,
    )

    train_dataset, val_dataset = build_datasets_version_1(
        expmt,
        train_shots=10_000,
        val_shots=1_000,
        seed=42,
        train_all_wt1_errors=True,
        train_all_wt2_errors=True,
        remove_trivial_syndromes=True,
    )

    train_dataset.save_to_file(
        Path(__file__).resolve().parent / "train_dataset.pt")
    val_dataset.save_to_file(
        Path(__file__).resolve().parent / "val_dataset.pt")


# =============================== Output ===============================
# Sampling shots from the noisy circuit...
# Added 8165 samples to the training dataset.
# Added 815 samples to the validation dataset.
# Generating all weight-1 errors...
# Added 186 samples to the training dataset.
# Generating all weight-2 errors...
# Added 17205 samples to the training dataset.
# Size of train_dataset: 25556
# Size of val_dataset: 815
