"""
Evaluate a trained logical decoder checkpoint on a fixed test set, against the
trivial baselines (always-0 and per-observable majority class) and MWPM on
identical shots.

Usage: (from torchdecoder/scripts directory)
    uv run python eval_logical.py --run-dir <path/to/run_dir> [options]

Options:
    --run-dir : Run directory containing config.yaml and checkpoints/
    --ckpt : Checkpoint filename inside <run-dir>/checkpoints (default: best_model.ckpt)
    --shots : Test-set size (default: 65536)
    --seed : Test-set sampling seed (default: 1234)
    --batch-size : Model inference batch size (default: 1024)
"""

import argparse
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tabulate import tabulate
import torch

from qecdec.decoders import MWPMDecoder
from torchdecoder_core.dataset import sample_decoding_dataset
from torchdecoder_core.models import build_logical_decoder_model
from utils import create_circuit_from_config


def shot_error_rate(predictions: np.ndarray, observables: np.ndarray) -> float:
    """Fraction of shots with at least one observable predicted incorrectly."""
    return (predictions != observables).any(axis=1).mean()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a logical decoder")
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--ckpt", default="best_model.ckpt", type=str)
    parser.add_argument("--shots", default=65536, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--batch-size", default=1024, type=int)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(run_dir / "config.yaml")
    circuit = create_circuit_from_config(cfg.circuit)
    print(f">>>>>> Circuit: {cfg.circuit.circuit_name} "
          f"({OmegaConf.to_container(cfg.circuit.circuit_params)}), "
          f"error_rate={cfg.circuit.error_rate}")

    model = build_logical_decoder_model(circuit, cfg.model)
    model.load_lightning_checkpoint(run_dir / "checkpoints" / args.ckpt)
    model.eval()
    print(f">>>>>> Loaded checkpoint: {run_dir / 'checkpoints' / args.ckpt}")

    dataset = sample_decoding_dataset(circuit, shots=args.shots, seed=args.seed)
    syndromes = dataset.syndromes  # (shots, num_chks), int32
    observables = dataset.observables.numpy()  # (shots, num_obsers), int32
    print(f">>>>>> Test set: {args.shots} shots (seed {args.seed})")

    # Model predictions.
    predictions = []
    with torch.no_grad():
        for i in range(0, args.shots, args.batch_size):
            logits = model(syndromes[i : i + args.batch_size])
            predictions.append((logits > 0).int().numpy())
    predictions = np.concatenate(predictions)
    model_err = shot_error_rate(predictions, observables)

    # Trivial baselines.
    zero_err = shot_error_rate(np.zeros_like(observables), observables)
    majority = (observables.mean(axis=0) > 0.5).astype(observables.dtype)
    majority_err = shot_error_rate(
        np.broadcast_to(majority, observables.shape), observables
    )

    # MWPM reference on identical shots.
    mwpm = MWPMDecoder(circuit.chkmat, circuit.prior)
    ehat = mwpm.decode_batch(syndromes.numpy().astype(np.uint8))
    mwpm_obs = (ehat @ circuit.obsmat.T) % 2
    mwpm_err = shot_error_rate(mwpm_obs, observables)

    def fmt(err: float) -> str:
        stderr = math.sqrt(err * (1 - err) / args.shots)
        return f"{err:.5f} ± {stderr:.5f}"

    print(
        tabulate(
            [
                [f"{cfg.model.name} ({args.ckpt})", fmt(model_err)],
                ["always-0 baseline", fmt(zero_err)],
                ["majority-class baseline", fmt(majority_err)],
                ["MWPM", fmt(mwpm_err)],
            ],
            headers=["decoder", "logical error rate (per shot)"],
            tablefmt="fancy_grid",
        )
    )


if __name__ == "__main__":
    main()
