"""Plotting utilities for custom benchmark results."""

from collections import defaultdict
from typing import Literal

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from ..constants import BASELINE_DECODERS
from ..params import QECParams
from .stats import BenchmarkStats


def _wilson_ci(p_hat: float, n: int, *, z: float = 1.0) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a binomial distribution.

    Parameters
    ----------
    p_hat : float
        Observed fraction of successes.
    n : int
        Number of trials.
    z : float
        Number of standard deviations for the interval (default 1.0 for ~68%).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    if n == 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n**2)) / denom
    return (max(centre - half_width, 0.0), min(centre + half_width, 1.0))


def plot_ler_vs_per(
    stats: list[BenchmarkStats],
    *,
    qec_params: QECParams,
    ler_mode: Literal["per shot", "per round"],
) -> matplotlib.figure.Figure:
    """Plot Logical Error Rate vs Physical Error Rate.

    Returns the matplotlib Figure.
    """
    groups: defaultdict[str, list[BenchmarkStats]] = defaultdict(list)
    for s in stats:
        groups[s.metadata.decoder].append(s)

    fig, ax = plt.subplots(1, 1)

    for decoder, group in sorted(groups.items()):
        group.sort(key=lambda s: s.metadata.p)

        ps = []
        lers = []
        err_lo = []
        err_hi = []

        for s in group:
            ler_per_shot = s.logical_error_rate
            ler_per_shot_lb, ler_per_shot_ub = _wilson_ci(ler_per_shot, s.shots)
            if ler_mode == "per shot":
                ler = ler_per_shot
                ler_lb = ler_per_shot_lb
                ler_ub = ler_per_shot_ub
            else:
                rounds = s.metadata.rounds
                ler = ler_per_shot / rounds
                ler_lb = ler_per_shot_lb / rounds
                ler_ub = ler_per_shot_ub / rounds

            ps.append(s.metadata.p)
            lers.append(ler)
            err_lo.append(ler - ler_lb)
            err_hi.append(ler_ub - ler)

        linestyle = "dashed" if decoder in BASELINE_DECODERS else "solid"
        ax.errorbar(
            ps,
            lers,
            yerr=[err_lo, err_hi],
            label=decoder,
            linestyle=linestyle,
            marker="o",
            capsize=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(axis="y")
    ax.set_title(
        f"{qec_params.code}, {qec_params.noise_model}, "
        f"d={qec_params.d}, rounds={qec_params.rounds}, basis={qec_params.basis}"
    )
    ax.set_xlabel("PER")
    ax.set_ylabel("LER per shot" if ler_mode == "per shot" else "LER per round")
    ax.legend()

    return fig
