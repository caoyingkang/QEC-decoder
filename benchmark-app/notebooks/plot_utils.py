"""Per-axes plotting utilities for QEC decoder benchmark studies.

Each helper draws on a single ``matplotlib.axes.Axes`` passed in by the
caller. Callers own the ``Figure`` (layout, suptitle, per-subplot titles,
colorbar placement, save, show); these helpers only render data and the
data-bound styling (axis labels, scales, legend, grid, ticks).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Iterable, Optional, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.image import AxesImage

from qecbench import TaskStats


def shot_error_rate_to_piece_error_rate(
    shot_error_rate: float,
    *,
    pieces: float,
    values: float = 1,
) -> float:
    """Convert shot error rate to per-piece (per-round) error rate.

    Mirrors ``sinter.shot_error_rate_to_piece_error_rate`` for the float case.
    See: https://github.com/quantumlib/Stim/blob/main/glue/sample/src/sinter/_probability_util.py
    """
    if not (0 <= shot_error_rate <= 1):
        raise ValueError(f"need 0 <= shot_error_rate={shot_error_rate} <= 1")
    if pieces <= 0:
        raise ValueError("need pieces > 0")
    if pieces == 1:
        return shot_error_rate
    if values != 1:
        p = 1 - (1 - shot_error_rate) ** (1 / values)
        p = shot_error_rate_to_piece_error_rate(p, pieces=pieces)
        return 1 - (1 - p) ** values
    if shot_error_rate > 0.5:
        return 1 - shot_error_rate_to_piece_error_rate(
            1 - shot_error_rate, pieces=pieces
        )
    randomize_rate = 2 * shot_error_rate
    round_randomize_rate = 1 - (1 - randomize_rate) ** (1 / pieces)
    round_error_rate = round_randomize_rate / 2
    if round_error_rate == 0:
        return shot_error_rate / pieces
    return round_error_rate


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
    stats: Iterable[TaskStats],
    ax: Axes,
    *,
    min_iter: Optional[int] = None,
    label_fn: Callable[[TaskStats], str | None],
    title: Optional[str] = None,
    xscale: str = "log",
) -> None:
    """Plot the cumulative distribution of iteration numbers, one curve for
    each `TaskStats` in `stats`.

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
    s: TaskStats,
    *,
    budget_step: int,
    x_metric: Literal["iter_budget", "avg_iter"],
    y_metric: Literal["fr_per_shot", "fr_per_round"],
) -> tuple[np.ndarray, np.ndarray]:
    assert s.iters_hist_on_converged is not None and s.iters_hist_on_success is not None
    assert s.shots > 0
    shots = s.shots
    rounds = s.metadata.circuit_params["rounds"]
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
            y[idx] = shot_error_rate_to_piece_error_rate(fr, pieces=rounds)
        else:
            raise ValueError(f"unknown y_metric: {y_metric!r}")

    return x, y


def plot_fr_vs_iter_budget(
    stats: Iterable[TaskStats],
    ax: Axes,
    *,
    budget_step: int,
    x_metric: Literal["iter_budget", "avg_iter"],
    y_metric: Literal["fr_per_shot", "fr_per_round"],
    label_fn: Callable[[TaskStats], str | None],
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


def _default_fr_annotation_fmt(v: float) -> str:
    return f"{v:.3f}" if v >= 0.01 else f"{v:.1e}"


def plot_heatmap(
    grid: np.ndarray,
    ax: Axes,
    *,
    x_values: Sequence[int],
    y_values: Sequence[int],
    norm: Normalize,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    cmap_name: str = "viridis",
    title: Optional[str] = None,
    annotations: bool = True,
    annotation_fmt: Optional[Callable[[float], str]] = None,
) -> AxesImage:
    """Render a precomputed 2D ``grid`` as a heatmap on ``ax`` and return the image.

    ``grid`` has shape ``(len(y_values), len(x_values))``; ``np.nan`` entries are
    treated as missing and drawn in white. Pass a caller-owned ``norm`` (shared
    across a grid of subplots) for consistent coloring; the returned ``AxesImage``
    lets the caller build a figure-level colorbar.

    Finite non-positive values are clamped up to ``norm.vmin`` for coloring (so a
    ``LogNorm`` does not mask them), but annotations always show the true value.
    """
    import matplotlib.pyplot as plt

    x_values = list(x_values)
    y_values = list(y_values)
    if grid.shape != (len(y_values), len(x_values)):
        raise ValueError(
            f"grid shape {grid.shape} does not match "
            f"(len(y_values)={len(y_values)}, len(x_values)={len(x_values)})"
        )
    if annotation_fmt is None:
        annotation_fmt = _default_fr_annotation_fmt

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    vmin = norm.vmin
    vmax = norm.vmax
    threshold = (
        float(np.sqrt(vmin * vmax))
        if vmin is not None and vmax is not None and vmin > 0 and vmax > 0
        else None
    )

    # Clamp finite non-positive values up to vmin so LogNorm does not mask them;
    # NaN (missing) stays NaN -> drawn as the "bad" color (white).
    color_grid = grid.copy()
    if vmin is not None:
        finite = np.isfinite(color_grid)
        color_grid[finite] = np.where(color_grid[finite] <= 0, vmin, color_grid[finite])

    im = ax.imshow(color_grid, cmap=cmap, norm=norm, aspect="auto", origin="lower")

    if annotations:
        for yi in range(len(y_values)):
            for xi in range(len(x_values)):
                val = grid[yi, xi]
                if not np.isfinite(val):
                    continue
                color = (
                    "white"
                    if threshold is not None and max(val, vmin) < threshold
                    else "black"
                )
                ax.text(
                    xi,
                    yi,
                    annotation_fmt(val),
                    ha="center",
                    va="center",
                    fontsize=HEATMAP_ANNOTATION_FONTSIZE,
                    color=color,
                )

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels(y_values)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if x_label:
        ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
    if y_label:
        ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")

    return im
