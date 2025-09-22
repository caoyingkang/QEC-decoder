from itertools import combinations
from tqdm import tqdm
import numpy as np
import stim
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Metric
from typing import Any
from qecdec.experiments import MemoryExperiment

INT_DTYPE = torch.int32
FLOAT_DTYPE = torch.float32

EPS = 1e-6
BIG = 1e8


class DecodingDataset(Dataset):
    """
    A PyTorch Dataset. Each item is a (syndrome, observable) pair with integer dtype.
    """

    def __init__(
        self,
        syndromes: np.ndarray,
        observables: np.ndarray,
    ):
        """
        Parameters
        ----------
            syndromes : np.ndarray
                Syndrome bits ∈ {0,1}, shape=(num_shots, m), integer or bool

            observables : np.ndarray
                Observable bits ∈ {0,1}, shape=(num_shots, k), integer or bool
        """
        assert isinstance(syndromes, np.ndarray)
        assert isinstance(observables, np.ndarray)
        assert np.issubdtype(syndromes.dtype, np.integer) or \
            np.issubdtype(syndromes.dtype, np.bool_)
        assert np.issubdtype(observables.dtype, np.integer) or \
            np.issubdtype(observables.dtype, np.bool_)
        assert syndromes.ndim == 2
        assert observables.ndim == 2
        assert syndromes.shape[0] == observables.shape[0]

        self.syndromes = torch.as_tensor(syndromes, dtype=INT_DTYPE)
        self.observables = torch.as_tensor(observables, dtype=INT_DTYPE)

    def __len__(self):
        return len(self.syndromes)

    def __getitem__(self, idx):
        return self.syndromes[idx], self.observables[idx]


class EarlyStopper:
    """
    A class that implements early stopping.
    """

    def __init__(
        self,
        *,
        patience: int,
        min_delta: float = 0.0,
    ):
        """
        Parameters
        ----------
            patience : int
                Number of epochs with no improvement after which training will be stopped

            min_delta : float
                Minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def update(self, val_loss: float):
        if self.early_stop:
            raise RuntimeError("Early stopping has been triggered")

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True


def build_datasets(
    expmt: MemoryExperiment,
    *,
    train_shots: int,
    val_shots: int,
    seed: int | None = None,
    train_all_wt1_errors: bool = True,
    train_all_wt2_errors: bool = True,
    remove_trivial_syndromes: bool = True,
) -> tuple[DecodingDataset, DecodingDataset]:
    """
    Parameters
    ----------
        expmt : MemoryExperiment
            The MemoryExperiment object from which to sample data

        train_shots : int
            Number of sampling shots for building `train_dataset`

        val_shots : int
            Number of sampling shots for building `val_dataset`

        seed : int | None
            Random seed used for sampling

        train_all_wt1_errors : bool
            Whether to include all weight-1 errors in `train_dataset`

        train_all_wt2_errors : bool
            Whether to include all weight-2 errors in `train_dataset`

        remove_trivial_syndromes : bool
            Whether to filter out trivial (i.e., all-zero) syndromes in `train_dataset` and `val_dataset`

    Returns
    -------
        train_dataset : DecodingDataset
            Training dataset

        val_dataset : DecodingDataset
            Validation dataset
    """
    n = expmt.num_error_mechanisms

    sampler = expmt.dem.compile_sampler(seed=seed)
    sampled_syndromes, sampled_observables, _ = sampler.sample(
        train_shots + val_shots)
    sampled_syndromes = sampled_syndromes.astype(np.int32)
    sampled_observables = sampled_observables.astype(np.int32)

    train_syndromes_list = [sampled_syndromes[:train_shots]]
    train_observables_list = [sampled_observables[:train_shots]]

    if train_all_wt1_errors:
        errors = np.eye(n, dtype=np.int32)
        syndromes = (errors @ expmt.chkmat.T) % 2
        observables = (errors @ expmt.obsmat.T) % 2
        train_syndromes_list.append(syndromes)
        train_observables_list.append(observables)

    if train_all_wt2_errors:
        errors = np.zeros(((n * (n - 1)) // 2, n), dtype=np.int32)
        for row, cols in enumerate(combinations(range(n), 2)):
            errors[row, cols] = 1
        syndromes = (errors @ expmt.chkmat.T) % 2
        observables = (errors @ expmt.obsmat.T) % 2
        train_syndromes_list.append(syndromes)
        train_observables_list.append(observables)

    train_syndromes = np.concatenate(train_syndromes_list)
    train_observables = np.concatenate(train_observables_list)
    if remove_trivial_syndromes:
        mask = np.any(train_syndromes != 0, axis=1)
        train_syndromes = train_syndromes[mask]
        train_observables = train_observables[mask]

    train_dataset = DecodingDataset(train_syndromes, train_observables)

    val_syndromes = sampled_syndromes[train_shots:]
    val_observables = sampled_observables[train_shots:]
    if remove_trivial_syndromes:
        mask = np.any(val_syndromes != 0, axis=1)
        val_syndromes = val_syndromes[mask]
        val_observables = val_observables[mask]

    val_dataset = DecodingDataset(val_syndromes, val_observables)

    return train_dataset, val_dataset


def train_gamma(
    model: nn.Module,
    train_dataset: DecodingDataset,
    val_dataset: DecodingDataset,
    loss_fn: nn.Module,
    metric: Metric,
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int = 10,
    batch_size: int = 64,
    device: str | None = None,
    scheduler_kwargs: dict[str, Any] = dict(),
    early_stopper: EarlyStopper | None = None,
):
    """
    Parameters
    ----------
        model : nn.Module
            The decoder to be trained

        train_dataset : DecodingDataset
            The training dataset

        val_dataset : DecodingDataset
            The validation dataset

        loss_fn : nn.Module
            The loss function

        metric : Metric
            The metric to be evaluated

        optimizer : torch.optim.Optimizer
            The optimizer

        num_epochs : int
            The number of epochs

        batch_size : int
            The batch size

        device : str | None
            The device to train on. If None, let PyTorch determine the device automatically.

        scheduler_kwargs : dict[str, Any]
            The keyword arguments for the learning rate scheduler `torch.optim.lr_scheduler.ReduceLROnPlateau`.
            If not provided, use default arguments.

        early_stopper : EarlyStopper | None
            The early stopper. If None, do not use early stopping.
    """
    if device is None:
        if torch.accelerator.is_available():
            device = torch.accelerator.current_accelerator().type
        else:
            device = "cpu"
    print(f"Using {device} device")

    scheduler = ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    model = model.to(device)
    loss_fn = loss_fn.to(device)
    metric = metric.to(device)

    # Build dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # Train model
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dataloader,
                    desc=f"Epoch {epoch+1}/{num_epochs}",
                    total=len(train_dataloader))
        for syndromes, observables in pbar:
            syndromes = syndromes.to(device)
            observables = observables.to(device)
            optimizer.zero_grad()

            # Forward pass
            all_llrs = model(syndromes)
            loss = loss_fn(all_llrs, syndromes, observables)
            running_loss += loss.item()

            # Backpropagation
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float('inf'))
            pbar.set_postfix({
                "avg_loss": f"{running_loss / (pbar.n + 1):.6f}",
                "grad_norm": f"{grad_norm:.6f}"
            })
            optimizer.step()
        avg_train_loss = running_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        metric.reset()
        running_loss = 0.0
        with torch.no_grad():
            for syndromes, observables in val_dataloader:
                syndromes = syndromes.to(device)
                observables = observables.to(device)

                # Forward pass
                all_llrs = model(syndromes)
                loss = loss_fn(all_llrs, syndromes, observables)
                running_loss += loss.item()
                metric.update(all_llrs, syndromes, observables)
        avg_val_loss = running_loss / len(val_dataloader)
        val_metrics = metric.compute()

        # Learning rate scheduler
        scheduler.step(avg_val_loss)

        # Print epoch summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.6f}")
        print(f"  Avg Val Loss: {avg_val_loss:.6f}")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print()

        # Early stopper
        if early_stopper is not None:
            early_stopper.update(avg_val_loss)
            if early_stopper.early_stop:
                print("Early stopping triggered")
                break


__all__ = [
    "DecodingDataset",
    "EarlyStopper",
    "build_datasets",
    "train_gamma",
]
