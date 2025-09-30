from itertools import combinations
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from pathlib import Path
from qecdec.experiments import MemoryExperiment
from qecdec.decoders import BPDecoder
from learned_decoders import LearnedBPBase, DecodingLoss, DecodingMetric

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
        syndromes: np.ndarray | torch.Tensor,
        observables: np.ndarray | torch.Tensor,
    ):
        """
        Parameters
        ----------
            syndromes : np.ndarray | torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(num_shots, m), integer or bool

            observables : np.ndarray | torch.Tensor
                Observable bits ∈ {0,1}, shape=(num_shots, k), integer or bool
        """
        assert isinstance(syndromes, np.ndarray) or isinstance(
            syndromes, torch.Tensor)
        assert isinstance(observables, np.ndarray) or isinstance(
            observables, torch.Tensor)
        assert syndromes.ndim == 2
        assert observables.ndim == 2
        assert syndromes.shape[0] == observables.shape[0]

        self.syndromes = torch.as_tensor(syndromes, dtype=INT_DTYPE)
        self.observables = torch.as_tensor(observables, dtype=INT_DTYPE)

    @classmethod
    def load_from_file(cls, file: str | Path):
        """
        Load the dataset from a file.
        """
        if isinstance(file, str):
            file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist")

        syndromes, observables = torch.load(file)
        return cls(syndromes, observables)

    def save_to_file(self, file: str | Path, overwrite_ok: bool = False):
        """
        Save the dataset to a file.
        """
        if isinstance(file, str):
            file = Path(file)
        if file.exists() and not overwrite_ok:
            raise FileExistsError(
                f"File {file} already exists, and overwrite_ok is set to False")

        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save((self.syndromes, self.observables), file)

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


def build_datasets_version_1(
    expmt: MemoryExperiment,
    *,
    train_shots: int,
    val_shots: int,
    seed: Optional[int] = None,
    train_all_wt1_errors: bool = True,
    train_all_wt2_errors: bool = True,
    remove_trivial_syndromes: bool = True,
    verbose: bool = True,
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

        verbose : bool
            Whether to print verbose output

    Returns
    -------
        train_dataset : DecodingDataset
            Training dataset

        val_dataset : DecodingDataset
            Validation dataset
    """
    def filter(syndromes, observables):
        if remove_trivial_syndromes:
            mask = np.any(syndromes != 0, axis=1)
            syndromes = syndromes[mask]
            observables = observables[mask]
        return syndromes, observables

    n = expmt.num_error_mechanisms

    if verbose:
        print("Sampling shots from the noisy circuit...")
    sampler = expmt.dem.compile_sampler(seed=seed)
    s, o, _ = sampler.sample(train_shots + val_shots)
    s = s.astype(np.int32)
    o = o.astype(np.int32)
    train_syndromes, train_observables = filter(
        s[:train_shots], o[:train_shots])
    val_syndromes, val_observables = filter(
        s[train_shots:], o[train_shots:])
    if verbose:
        print(
            f"Added {len(train_syndromes)} samples to the training dataset.")
        print(
            f"Added {len(val_syndromes)} samples to the validation dataset.")

    if train_all_wt1_errors:
        if verbose:
            print("Generating all weight-1 errors...")
        errors = np.eye(n, dtype=np.int32)
        train_syndromes_from_wt1_errors, train_observables_from_wt1_errors = filter(
            (errors @ expmt.chkmat.T) % 2, (errors @ expmt.obsmat.T) % 2)
        train_syndromes = np.concatenate(
            [train_syndromes, train_syndromes_from_wt1_errors])
        train_observables = np.concatenate(
            [train_observables, train_observables_from_wt1_errors])
        if verbose:
            print(
                f"Added {len(train_syndromes_from_wt1_errors)} samples to the training dataset.")

    if train_all_wt2_errors:
        if verbose:
            print("Generating all weight-2 errors...")
        errors = np.zeros(((n * (n - 1)) // 2, n), dtype=np.int32)
        for row, cols in enumerate(combinations(range(n), 2)):
            errors[row, cols] = 1
        train_syndromes_from_wt2_errors, train_observables_from_wt2_errors = filter(
            (errors @ expmt.chkmat.T) % 2, (errors @ expmt.obsmat.T) % 2)
        train_syndromes = np.concatenate(
            [train_syndromes, train_syndromes_from_wt2_errors])
        train_observables = np.concatenate(
            [train_observables, train_observables_from_wt2_errors])
        if verbose:
            print(
                f"Added {len(train_syndromes_from_wt2_errors)} samples to the training dataset.")

    train_dataset = DecodingDataset(train_syndromes, train_observables)
    val_dataset = DecodingDataset(val_syndromes, val_observables)

    if verbose:
        print(f"Size of train_dataset: {len(train_dataset)}")
        print(f"Size of val_dataset: {len(val_dataset)}")

    return train_dataset, val_dataset


def build_datasets_version_2(
    expmt: MemoryExperiment,
    *,
    train_shots: int,
    val_shots: int,
    seed: Optional[int] = None,
    train_all_wt1_errors: bool = True,
    train_all_wt2_errors: bool = True,
    verbose: bool = True,
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

        verbose : bool
            Whether to print verbose output

    Returns
    -------
        train_dataset : DecodingDataset
            Training dataset

        val_dataset : DecodingDataset
            Validation dataset
    """
    bp = BPDecoder(expmt.chkmat, expmt.prior, max_iter=10)

    def filter(syn, obs, *, remove_easy_instances):
        # Remove trivial syndromes.
        mask = np.any(syn != 0, axis=1)
        syn = syn[mask]
        obs = obs[mask]

        # Remove instances that can be decoded correctly by vanilla BP in 10 iterations.
        if remove_easy_instances:
            ehat = bp.decode_batch(syn)
            mask = np.any(syn != (ehat @ expmt.chkmat.T) % 2, axis=1) \
                | np.any(obs != (ehat @ expmt.obsmat.T) % 2, axis=1)
            syn = syn[mask]
            obs = obs[mask]

        return syn, obs

    n = expmt.num_error_mechanisms

    # =============================== sample shots from noisy circuit ===============================
    if verbose:
        print("Sampling shots from the noisy circuit...")
    sampler = expmt.dem.compile_sampler(seed=seed)
    syn, obs, _ = sampler.sample(train_shots + val_shots)
    syn, obs = syn.astype(np.int32), obs.astype(np.int32)
    train_syndromes, train_observables = filter(
        syn[:train_shots], obs[:train_shots], remove_easy_instances=True)
    val_syndromes, val_observables = filter(
        syn[train_shots:], obs[train_shots:], remove_easy_instances=True)
    if verbose:
        print(
            f"Added {len(train_syndromes)} samples to the training dataset.")
        print(
            f"Added {len(val_syndromes)} samples to the validation dataset.")

    # =============================== weight-1 errors ===============================
    if train_all_wt1_errors:
        if verbose:
            print("Generating all weight-1 errors...")
        errors = np.eye(n, dtype=np.int32)
        syn, obs = (errors @ expmt.chkmat.T) % 2, (errors @ expmt.obsmat.T) % 2
        train_syndromes_from_wt1_errors, train_observables_from_wt1_errors = filter(
            syn, obs, remove_easy_instances=False)
        train_syndromes = np.concatenate(
            [train_syndromes, train_syndromes_from_wt1_errors])
        train_observables = np.concatenate(
            [train_observables, train_observables_from_wt1_errors])
        if verbose:
            print(
                f"Added {len(train_syndromes_from_wt1_errors)} samples to the training dataset.")

    # =============================== weight-2 errors ===============================
    if train_all_wt2_errors:
        if verbose:
            print("Generating all weight-2 errors...")
        errors = np.zeros(((n * (n - 1)) // 2, n), dtype=np.int32)
        for row, cols in enumerate(combinations(range(n), 2)):
            errors[row, cols] = 1
        syn, obs = (errors @ expmt.chkmat.T) % 2, (errors @ expmt.obsmat.T) % 2
        train_syndromes_from_wt2_errors, train_observables_from_wt2_errors = filter(
            syn, obs, remove_easy_instances=True)
        train_syndromes = np.concatenate(
            [train_syndromes, train_syndromes_from_wt2_errors])
        train_observables = np.concatenate(
            [train_observables, train_observables_from_wt2_errors])
        if verbose:
            print(
                f"Added {len(train_syndromes_from_wt2_errors)} samples to the training dataset.")

    # =============================== collect all instances ===============================
    train_dataset = DecodingDataset(train_syndromes, train_observables)
    val_dataset = DecodingDataset(val_syndromes, val_observables)

    if verbose:
        print(f"Size of train_dataset: {len(train_dataset)}")
        print(f"Size of val_dataset: {len(val_dataset)}")

    return train_dataset, val_dataset


def train_decoder(
    model: LearnedBPBase,
    train_dataset: DecodingDataset,
    val_dataset: DecodingDataset,
    loss_fn: DecodingLoss,
    metric: DecodingMetric,
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int,
    batch_size: int,
    device: Optional[str] = None,
    lr_scheduler: Optional[ReduceLROnPlateau] = None,
    early_stopper: Optional[EarlyStopper] = None,
):
    """
    Parameters
    ----------
        model : LearnedBPBase
            The decoder to be trained

        train_dataset : DecodingDataset
            The training dataset

        val_dataset : DecodingDataset
            The validation dataset

        loss_fn : DecodingLoss
            The loss function

        metric : DecodingMetric
            The metric to be evaluated

        optimizer : torch.optim.Optimizer
            The optimizer

        num_epochs : int
            The number of epochs

        batch_size : int
            The batch size

        device : str | None
            The device to train on. If None, let PyTorch determine the device automatically.

        lr_scheduler : ReduceLROnPlateau | None
            The learning rate scheduler. If None, do not use learning rate scheduler.

        early_stopper : EarlyStopper | None
            The early stopper. If None, do not use early stopping.
    """
    if device is None:
        if torch.accelerator.is_available():
            device = torch.accelerator.current_accelerator().type
        else:
            device = "cpu"
    print(f"Using {device} device")

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
            var2llrs = model(syndromes)
            loss = loss_fn(var2llrs, syndromes, observables)
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
                var2llrs = model(syndromes)
                loss = loss_fn(var2llrs, syndromes, observables)
                running_loss += loss.item()
                metric.update(var2llrs, syndromes, observables)
        avg_val_loss = running_loss / len(val_dataloader)
        val_metrics = metric.compute()

        # Learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(avg_val_loss)

        # Print epoch summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.6f}")
        print(f"  Avg Val Loss: {avg_val_loss:.6f}")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")
        if lr_scheduler is not None:
            print(f"  Learning Rate: {lr_scheduler.get_last_lr()[0]:.6f}")
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
    "build_datasets_version_1",
    "build_datasets_version_2",
    "train_decoder",
]
