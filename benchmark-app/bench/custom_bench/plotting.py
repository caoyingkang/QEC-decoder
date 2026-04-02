"""Plotting utilities for custom benchmark results."""

from collections import defaultdict
from typing import Callable, Literal

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from ..constants import BASELINE_DECODERS
from ..params import QECParams
from .stats import BenchmarkStats


def _wilson_ci(p_hat: float, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a binomial distribution.

    Parameters
    ----------
    p_hat : float
        Observed fraction of successes.
    n : int
        Number of trials.
    z : float
        Number of standard deviations for the interval (default 1.96 for ~95% confidence level).

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


def _qec_title(qec_params: QECParams) -> str:
    return (
        f"{qec_params.code}, {qec_params.noise_model}, "
        f"d={qec_params.d}, rounds={qec_params.rounds}, basis={qec_params.basis}"
    )


def _plot_rate_vs_per(
    stats: list[BenchmarkStats],
    *,
    qec_params: QECParams,
    rate_fn: Callable[[BenchmarkStats], float],
    ylabel: str,
    rate_scale_fn: Callable[[BenchmarkStats], float] | None = None,
) -> matplotlib.figure.Figure:
    """Generic rate-vs-PER log-log plot with Wilson CI error bars.

    Parameters
    ----------
    rate_fn : Callable[[BenchmarkStats], float]
        Extracts the per-shot binomial rate from a ``BenchmarkStats``.
    ylabel : str
        Label for the y-axis.
    rate_scale_fn : Callable[[BenchmarkStats], float] or None
        If provided, both the rate and CI bounds are divided by
        ``rate_scale_fn(s)`` (e.g. for per-round scaling).
    """
    groups: defaultdict[str, list[BenchmarkStats]] = defaultdict(list)
    for s in stats:
        groups[s.metadata.decoder_name].append(s)

    fig, ax = plt.subplots(1, 1)

    for decoder, group in sorted(groups.items()):
        group.sort(key=lambda s: s.metadata.p)

        ps: list[float] = []
        rates: list[float] = []
        err_lo: list[float] = []
        err_hi: list[float] = []

        for s in group:
            rate = rate_fn(s)
            lb, ub = _wilson_ci(rate, s.shots)
            if rate_scale_fn is not None:
                scale = rate_scale_fn(s)
                rate /= scale
                lb /= scale
                ub /= scale

            ps.append(s.metadata.p)
            rates.append(rate)
            err_lo.append(rate - lb)
            err_hi.append(ub - rate)

        linestyle = "dashed" if decoder in BASELINE_DECODERS else "solid"
        ax.errorbar(
            ps,
            rates,
            yerr=[err_lo, err_hi],
            label=decoder,
            linestyle=linestyle,
            marker="o",
            capsize=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(axis="y")
    ax.set_title(_qec_title(qec_params))
    ax.set_xlabel("PER")
    ax.set_ylabel(ylabel)
    ax.legend()

    return fig


# -- Rate-vs-PER plots --------------------------------------------------------


def plot_fr_vs_per(
    stats: list[BenchmarkStats],
    *,
    qec_params: QECParams,
    mode: Literal["per shot", "per round"],
) -> matplotlib.figure.Figure:
    """Plot Failure Rate (FR) vs Physical Error Rate (PER)."""
    return _plot_rate_vs_per(
        stats,
        qec_params=qec_params,
        rate_fn=lambda s: s.failure_rate,
        ylabel=f"FR {mode}",
        rate_scale_fn=((lambda s: s.metadata.rounds) if mode == "per round" else None),
    )


def plot_ler_vs_per(
    stats: list[BenchmarkStats],
    *,
    qec_params: QECParams,
    mode: Literal["per shot", "per round"],
) -> matplotlib.figure.Figure:
    """Plot Logical Error Rate (LER) vs Physical Error Rate (PER)."""
    return _plot_rate_vs_per(
        stats,
        qec_params=qec_params,
        rate_fn=lambda s: s.logical_error_rate,
        ylabel=f"LER {mode}",
        rate_scale_fn=((lambda s: s.metadata.rounds) if mode == "per round" else None),
    )


def plot_smr_vs_per(
    stats: list[BenchmarkStats],
    *,
    qec_params: QECParams,
    mode: Literal["per shot", "per round"],
) -> matplotlib.figure.Figure:
    """Plot Syndrome Mismatch Rate (SMR) vs Physical Error Rate (PER)."""
    return _plot_rate_vs_per(
        stats,
        qec_params=qec_params,
        rate_fn=lambda s: s.syndrome_mismatch_rate,
        ylabel=f"SMR {mode}",
        rate_scale_fn=((lambda s: s.metadata.rounds) if mode == "per round" else None),
    )


# -- Average Iterations plot --------------------------------


def plot_avg_iters_vs_per(
    stats: list[BenchmarkStats],
    *,
    qec_params: QECParams,
    avg_over: Literal["all shots", "converged shots", "successful shots"],
) -> matplotlib.figure.Figure:
    """Plot average iterations vs PER for iterative decoders."""
    groups: defaultdict[str, list[BenchmarkStats]] = defaultdict(list)
    for s in stats:
        groups[s.metadata.decoder_name].append(s)

    fig, ax = plt.subplots(1, 1)

    for decoder, group in sorted(groups.items()):
        group.sort(key=lambda s: s.metadata.p)

        ps: list[float] = []
        avgs: list[float] = []

        for s in group:
            if avg_over == "all shots":
                avg = s.avg_iters
            elif avg_over == "converged shots":
                avg = s.avg_iters_on_converged
            elif avg_over == "successful shots":
                avg = s.avg_iters_on_success
            else:
                raise ValueError(f"Invalid avg_over: {avg_over}")

            ps.append(s.metadata.p)
            avgs.append(avg)

        linestyle = "dashed" if decoder in BASELINE_DECODERS else "solid"
        ax.plot(ps, avgs, label=decoder, linestyle=linestyle, marker="o")

    ax.set_xscale("log")
    ax.grid(axis="y")
    ax.set_title(_qec_title(qec_params))
    ax.set_xlabel("PER")
    ax.set_ylabel(f"Avg. iters (over {avg_over})")
    ax.legend()

    return fig


# -- Iterations Distribution plot --------------------------------


def plot_iters_distribution(
    stats: list[BenchmarkStats],
    *,
    p: float,
    qec_params: QECParams,
    dist_over: Literal["all shots", "converged shots", "successful shots"],
) -> go.Figure:
    """Plotly overlaid bar chart of iteration distributions at a given PER."""
    selected_stats = [s for s in stats if s.metadata.p == p]
    selected_stats.sort(key=lambda s: s.metadata.decoder_name)

    fig = go.Figure()

    for s in selected_stats:
        if dist_over == "all shots":
            hist = s.iters_hist_on_converged.copy()
            hist[-1] += s.synd_mismatches
            total = s.shots
        elif dist_over == "converged shots":
            hist = s.iters_hist_on_converged
            total = s.synd_matches
        elif dist_over == "successful shots":
            hist = s.iters_hist_on_success
            total = s.success
        else:
            raise ValueError(f"Invalid dist_over: {dist_over}")
        fractions = hist / total

        fig.add_trace(
            go.Bar(
                x=np.arange(len(hist)),
                y=fractions,
                name=s.metadata.decoder_name,
                opacity=0.6,
            )
        )

    fig.update_layout(
        title=(f"{_qec_title(qec_params)}, p={p}"),
        xaxis_title="Iterations",
        yaxis_title=f"Fraction (over {dist_over})",
        legend_title="Decoder",
    )

    return fig
