from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning as L

from qecdec.circuits import QECCircuit
from torchdecoder_core.dataset import (
    DecodingDataset,
    StreamingDecodingDataset,
    sample_decoding_dataset,
)


class DecodingDataModule(L.LightningDataModule):
    def __init__(self, data_dir: Path, *, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        match stage:
            case "fit":
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
            case "validate":
                self.val_ds = DecodingDataset.load_from_file(
                    self.data_dir / "val_dataset.pt"
                )
                print(">>>>>> Summary of val_dataset:")
                self.val_ds.print_summary()
            case "test":
                self.test_ds = DecodingDataset.load_from_file(
                    self.data_dir / "test_dataset.pt"
                )
                print(">>>>>> Summary of test_dataset:")
                self.test_ds.print_summary()
            case _:
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


class StreamingDecodingDataModule(L.LightningDataModule):
    """
    DataModule for on-the-fly streaming training: the train set is a
    `StreamingDecodingDataset` sampling shots from the circuit's DEM at a
    settable error rate (driven by `NoiseCurriculumCallback`); the val set is a
    fixed-seed finite sample at the target error rate, so val metrics are
    comparable across epochs.
    """

    def __init__(
        self,
        circuit_factory: Callable[[float], QECCircuit],
        *,
        error_rate: float,
        shots_per_epoch: int,
        val_shots: int,
        batch_size: int,
        num_workers: int,
        base_seed: int,
        val_seed: int,
    ):
        """
        Parameters
        ----------
            circuit_factory : Callable[[float], QECCircuit]
                Builds the circuit for a given error rate. Must be picklable
                for multiprocessing DataLoader workers.

            error_rate : float
                Target error rate: the streaming train set starts here (a noise
                curriculum may lower it), and the val set is sampled at it.

            shots_per_epoch : int
                Number of training shots streamed per epoch.

            val_shots : int
                Size of the fixed validation set.

            batch_size : int
                Batch size for both loaders.

            num_workers : int
                Number of DataLoader workers.

            base_seed : int
                Seeds the train DataLoader's generator (and thus the per-worker
                per-epoch stream seeds).

            val_seed : int
                Seed for sampling the fixed validation set.
        """
        super().__init__()
        self.circuit_factory = circuit_factory
        self.error_rate = error_rate
        self.shots_per_epoch = shots_per_epoch
        self.val_shots = val_shots
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_seed = base_seed
        self.val_seed = val_seed

    def setup(self, stage: str):
        match stage:
            case "fit":
                self.train_ds = StreamingDecodingDataset(
                    self.circuit_factory,
                    error_rate=self.error_rate,
                    shots_per_epoch=self.shots_per_epoch,
                    base_seed=self.base_seed,
                )
                self.val_ds = sample_decoding_dataset(
                    self.circuit_factory(self.error_rate),
                    shots=self.val_shots,
                    seed=self.val_seed,
                )
            case "validate":
                self.val_ds = sample_decoding_dataset(
                    self.circuit_factory(self.error_rate),
                    shots=self.val_shots,
                    seed=self.val_seed,
                )
            case _:
                raise NotImplementedError(f"Stage {stage} is not supported")

    def train_dataloader(self):
        # persistent_workers must stay False: workers re-copy the dataset each
        # epoch, picking up curriculum error-rate changes and fresh per-epoch
        # worker seeds.
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(self.base_seed),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
