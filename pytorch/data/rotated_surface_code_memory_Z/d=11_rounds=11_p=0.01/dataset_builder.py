import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
    from qecdec import RotatedSurfaceCode_Memory
    from train_utils import build_datasets_version_2

    expmt = RotatedSurfaceCode_Memory(
        d=11,
        rounds=11,
        basis='Z',
        data_qubit_error_rate=0.01,
        meas_error_rate=0.01,
    )

    train_dataset, val_dataset = build_datasets_version_2(
        expmt,
        train_shots=100_000,
        val_shots=10_000,
        seed=42,
    )

    train_dataset.save_to_file(
        Path(__file__).resolve().parent / "train_dataset.pt")
    val_dataset.save_to_file(
        Path(__file__).resolve().parent / "val_dataset.pt")


# =============================== Output ===============================
# Sampling shots from the noisy circuit...
# Added 45188 samples to the training dataset.
# Added 4636 samples to the validation dataset.
# Generating all weight-1 errors...
# Added 1992 samples to the training dataset.
# Generating all weight-2 errors...
# Added 10058 samples to the training dataset.
# Size of train_dataset: 57238
# Size of val_dataset: 4636
