from pathlib import Path

from torch.utils.data import DataLoader
import lightning as L

from torchdecoder_core.dataset import DecodingDataset


class DecodingDataModule(L.LightningDataModule):
    def __init__(self, data_dir: Path, *, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = DecodingDataset.load_from_file(
                self.data_dir / "train_dataset.pt"
            )
            self.val_ds = DecodingDataset.load_from_file(
                self.data_dir / "val_dataset.pt"
            )
            print(">>>>>> Summary of train_dataset:")
            self.train_ds.print_summary()
            print(">>>>>> Summary of val_dataset:")
            self.val_ds.print_summary()
        elif stage == "validate":
            self.val_ds = DecodingDataset.load_from_file(
                self.data_dir / "val_dataset.pt"
            )
            print(">>>>>> Summary of val_dataset:")
            self.val_ds.print_summary()
        elif stage == "test":
            self.test_ds = DecodingDataset.load_from_file(
                self.data_dir / "test_dataset.pt"
            )
            print(">>>>>> Summary of test_dataset:")
            self.test_ds.print_summary()
        else:
            raise NotImplementedError(f"Stage {stage} is not supported")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
