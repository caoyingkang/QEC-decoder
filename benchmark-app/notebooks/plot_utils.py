"""Per-axes plotting utilities for QEC decoder benchmark studies.

Each helper draws on a single ``matplotlib.axes.Axes`` passed in by the
caller. Callers own the ``Figure`` (layout, suptitle, per-subplot titles,
colorbar placement, save, show); these helpers only render data and the
data-bound styling (axis labels, scales, legend, grid, ticks).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Iterable, Optional, Literal
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
import sinter

# Add benchmark-app to the path so we can import bench modules
BENCHMARK_APP_DIR = Path("__file__").resolve().parent.parent
if str(BENCHMARK_APP_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_APP_DIR))

from bench.custom_bench.stats import BenchmarkStats

LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 22
SUPTITLE_FONTSIZE = 25
LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 20
HEATMAP_SUPTITLE_FONTSIZE = 30
HEATMAP_ANNOTATION_FONTSIZE = 12
LEGEND_FONTSIZE = 14

METRIC_LABELS = {
    "iter_budget": "Iteration budget",
    "avg_iter": "Average iterations",
    "fr_per_shot": "Failure rate",
    "fr_per_round": "Failure rate per round",
}


def plot_iter_cdf(
    stats: Iterable[BenchmarkStats],
    ax: Axes,
    *,
    min_iter: Optional[int] = None,
    label_fn: Callable[[BenchmarkStats], str | None],
    title: Optional[str] = None,
    xscale: str = "log",
) -> None:
    """Plot the cumulative distribution of iteration numbers, one curve for
    each `BenchmarkStats` in `stats`.

    If `min_iter` is set, the view will be cropped to data points with iteration
    number at least this value.
    """
    if min_iter is None:
        min_iter = 1 if xscale == "log" else 0

    for s in stats:
        if not s.metadata.is_iterative:
            raise RuntimeError("Expect iterative decoders")
        label = label_fn(s)
        hist = s.iters_hist_on_converged.copy()
        hist[-1] += s.shots - s.synd_matches
        assert int(np.sum(hist)) == s.shots > 0
        x = np.arange(len(hist))
        y = np.cumsum(hist) / s.shots * 100
        ax.plot(x[min_iter:], y[min_iter:], label=label)

    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xscale(xscale)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.set_xlabel("Iterations", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Cumulative Probability (%)", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(left=min_iter)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(True, which="both", alpha=0.5)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower right")


def _budget_curve(
    s: BenchmarkStats,
    *,
    budget_step: int,
    x_metric: Literal["iter_budget", "avg_iter"],
    y_metric: Literal["fr_per_shot", "fr_per_round"],
) -> tuple[np.ndarray, np.ndarray]:
    assert s.iters_hist_on_converged is not None and s.iters_hist_on_success is not None
    assert s.shots > 0
    shots = s.shots
    rounds = s.metadata.rounds
    max_budget = s.metadata.max_iter
    budget_list = list(range(budget_step, max_budget + 1, budget_step))
    cum_conv = np.cumsum(s.iters_hist_on_converged)
    cum_conv_weighted = np.cumsum(np.arange(max_budget + 1) * s.iters_hist_on_converged)
    cum_succ = np.cumsum(s.iters_hist_on_success)

    x = np.zeros(len(budget_list))
    y = np.zeros(len(budget_list))

    for idx, budget in enumerate(budget_list):
        conv_cnt = int(cum_conv[budget])
        if x_metric == "iter_budget":
            x[idx] = float(budget)
        elif x_metric == "avg_iter":
            x[idx] = (cum_conv_weighted[budget] + (shots - conv_cnt) * budget) / shots
        else:
            raise ValueError(f"unknown x_metric: {x_metric!r}")

        succ_cnt = int(cum_succ[budget])
        fr = (shots - succ_cnt) / shots
        if y_metric == "fr_per_shot":
            y[idx] = fr
        elif y_metric == "fr_per_round":
            y[idx] = sinter.shot_error_rate_to_piece_error_rate(fr, pieces=rounds)
        else:
            raise ValueError(f"unknown y_metric: {y_metric!r}")

    return x, y


def plot_fr_vs_iter_budget(
    stats: Iterable[BenchmarkStats],
    ax: Axes,
    *,
    budget_step: int,
    x_metric: Literal["iter_budget", "avg_iter"],
    y_metric: Literal["fr_per_shot", "fr_per_round"],
    label_fn: Callable[[BenchmarkStats], str | None],
    title: Optional[str] = None,
    xscale: str = "log",
    yscale: str = "log",
) -> None:
    """Contour plot of failure rate vs iteration budget.

    For each stats entry, synthesize a curve by truncating the iteration
    histogram at a range of different iteration budgets. Any shot that
    originally converged at an iteration greater than the budget is
    treated as un-converged (and therefore a decoding failure). The list
    of budgets for a stats entry `s` is given by
    `range(budget_step, s.metadata.max_iter + 1, budget_step)`.
    """
    if budget_step < 1:
        raise ValueError(f"budget_step must be >= 1, got {budget_step}")

    for s in stats:
        if not s.metadata.is_iterative:
            raise RuntimeError("Expect iterative decoders")
        label = label_fn(s)
        x, y = _budget_curve(
            s,
            budget_step=budget_step,
            x_metric=x_metric,
            y_metric=y_metric,
        )
        ax.plot(x, y, label=label, marker="+")

    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(METRIC_LABELS[x_metric], fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(METRIC_LABELS[y_metric], fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(True, which="both", alpha=0.5)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
