"""Per-axes plotting utilities for QEC decoder benchmark studies.

Each helper draws on a single ``matplotlib.axes.Axes`` passed in by the
caller. Callers own the ``Figure`` (layout, suptitle, per-subplot titles,
colorbar placement, save, show); these helpers only render data and the
data-bound styling (axis labels, scales, legend, grid, ticks).
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import numpy as np
from qecbench import TaskStats

HEATMAP_ANNOTATION_FONTSIZE = 7.5
LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 14
SUPLABEL_FONTSIZE = 20
SUPTITLE_FONTSIZE = 25
TICK_FONTSIZE = 14
TITLE_FONTSIZE = 22


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


def load_stats(
    csv_path: Path,
    *,
    error_rates: list[float],
    decoder_fixed_params: dict[str, Any],
    decoder_flex_params: dict[str, list[Any]] | None = None,
) -> Iterator[TaskStats]:
    if decoder_flex_params is None:
        decoder_flex_params = {}
    for s in TaskStats.load_csv(csv_path):
        if (
            s.metadata.error_rate in error_rates
            and all(
                s.metadata.decoder_params.get(k) == v
                for k, v in decoder_fixed_params.items()
            )
            and all(
                s.metadata.decoder_params.get(k) in v_list
                for k, v_list in decoder_flex_params.items()
            )
        ):
            yield s


def _budget_curve(
    s: TaskStats,
    budget_list: list[int],
    *,
    fr_mode: Literal["per_shot", "per_round"],
    iter_mode: Literal["max_iter", "avg_iter"],
) -> tuple[np.ndarray, np.ndarray]:
    assert s.iters_hist_on_converged is not None and s.iters_hist_on_success is not None
    assert s.shots > 0
    shots = s.shots
    rounds = s.metadata.circuit_params["rounds"]
    max_budget = s.metadata.max_iter
    assert all(b <= max_budget for b in budget_list)
    cum_conv = np.cumsum(s.iters_hist_on_converged)
    cum_conv_weighted = np.cumsum(np.arange(max_budget + 1) * s.iters_hist_on_converged)
    cum_succ = np.cumsum(s.iters_hist_on_success)

    x = np.zeros(len(budget_list))
    y = np.zeros(len(budget_list))

    for idx, budget in enumerate(budget_list):
        conv_cnt = int(cum_conv[budget])
        match iter_mode:
            case "max_iter":
                x[idx] = float(budget)
            case "avg_iter":
                x[idx] = (
                    cum_conv_weighted[budget] + (shots - conv_cnt) * budget
                ) / shots
            case _:
                raise ValueError(f"unknown iter_mode: {iter_mode!r}")

        succ_cnt = int(cum_succ[budget])
        fr = (shots - succ_cnt) / shots
        match fr_mode:
            case "per_shot":
                y[idx] = fr
            case "per_round":
                y[idx] = shot_error_rate_to_piece_error_rate(fr, pieces=rounds)
            case _:
                raise ValueError(f"unknown fr_mode: {fr_mode!r}")

    return x, y


def plot_fr_vs_iter_contour_from_inferred(
    stats: Iterable[TaskStats],
    ax: Axes,
    budget_list_fn: Callable[[TaskStats], list[int]],
    *,
    fr_mode: Literal["per_shot", "per_round"],
    iter_mode: Literal["max_iter", "avg_iter"],
    label_fn: Callable[[TaskStats], str],
    color_fn: Callable[[TaskStats], str],
    linestyle: Callable[[TaskStats], str] | str,
    marker: Callable[[TaskStats], str] | str,
    mfc: Callable[[TaskStats], str | None] | str | None,
    xscale: Literal["log", "linear"],
    yscale: Literal["log", "linear"],
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_legend: bool = True,
    legend_loc: str = "upper right",
    title: str | None = None,
) -> None:
    """Contour plot of failure rate vs number of iterations.

    For each ``stats`` entry, synthesize a curve by truncating the iteration
    histogram at a range of different iteration budgets. Any shot that
    originally converged at an iteration greater than the budget is treated
    as un-converged (and therefore a decoding failure).
    """
    for s in stats:
        if not s.metadata.is_iterative:
            raise ValueError("Expect iterative decoders")
        x, y = _budget_curve(
            s,
            budget_list_fn(s),
            fr_mode=fr_mode,
            iter_mode=iter_mode,
        )
        ax.plot(
            x,
            y,
            label=label_fn(s),
            color=color_fn(s),
            linestyle=linestyle(s) if isinstance(linestyle, Callable) else linestyle,
            marker=marker(s) if isinstance(marker, Callable) else marker,
            mfc=mfc(s) if isinstance(mfc, Callable) else mfc,
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(False, which="minor")
    if show_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, loc=legend_loc)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")


GROUPID = TypeVar("GROUPID")


def plot_fr_vs_iter_from_collected(
    stats: Iterable[TaskStats],
    ax: Axes,
    *,
    x_metric: Literal["iter_budget", "avg_iter"],
    y_metric: Literal["fr_per_shot", "fr_per_round"],
    group_fn: Callable[[TaskStats], GROUPID],
    label_fn: Callable[[GROUPID], str],
    color_fn: Callable[[GROUPID], str],
    linestyle: Callable[[GROUPID], str] | str,
    marker: Callable[[GROUPID], str] | str,
    mfc: Callable[[GROUPID], str | None] | str | None,
    xscale: Literal["log", "linear"],
    yscale: Literal["log", "linear"],
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_legend: bool = True,
    legend_loc: str = "upper right",
    title: str | None = None,
) -> None:
    """Contour plot of failure rate vs number of iterations.

    Every data point corresponds to a single ``stats`` entry, every curve
    corresponds to ``stats`` entries in the same group.
    """

    def _x(s: TaskStats) -> float:
        match x_metric:
            case "iter_budget":
                return float(s.metadata.max_iter)
            case "avg_iter":
                return s.avg_iters
            case _:
                raise ValueError(f"unknown x_metric: {x_metric!r}")

    def _y(s: TaskStats) -> float:
        fr = s.failure_rate
        match y_metric:
            case "fr_per_shot":
                return fr
            case "fr_per_round":
                return shot_error_rate_to_piece_error_rate(
                    fr, pieces=s.metadata.circuit_params["rounds"]
                )
            case _:
                raise ValueError(f"unknown y_metric: {y_metric!r}")

    groups: dict[GROUPID, list[TaskStats]] = {}
    for s in stats:
        if not s.metadata.is_iterative:
            raise ValueError("Expect iterative decoders")
        groups.setdefault(group_fn(s), []).append(s)

    for gid, group in groups.items():
        group.sort(key=lambda s: s.metadata.max_iter)  # sort by iteration budget
        x = np.array([_x(s) for s in group])
        y = np.array([_y(s) for s in group])
        ax.plot(
            x,
            y,
            label=label_fn(gid),
            color=color_fn(gid),
            linestyle=linestyle(gid) if isinstance(linestyle, Callable) else linestyle,
            marker=marker(gid) if isinstance(marker, Callable) else marker,
            mfc=mfc(gid) if isinstance(mfc, Callable) else mfc,
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(False, which="minor")
    if show_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, loc=legend_loc)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")


def plot_heatmap(
    grid: np.ndarray,
    ax: Axes,
    *,
    x_values: Sequence[int],
    y_values: Sequence[int],
    norm: Normalize,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap_name: str = "viridis",
    title: str | None = None,
    annotations: bool = True,
    annotation_fmt: Callable[[float], str] = lambda x: f"{x}",
) -> AxesImage:
    """Render a precomputed 2D ``grid`` as a heatmap on ``ax`` and return the image.

    ``grid`` has shape ``(len(y_values), len(x_values))``; ``np.nan`` entries are
    treated as missing and drawn in white. Pass caller-owned ``norm`` and ``cmap_name``
    for consistent coloring; the returned ``AxesImage`` lets the caller build a
    figure-level colorbar.
    """
    x_values = list(x_values)
    y_values = list(y_values)
    if grid.shape != (len(y_values), len(x_values)):
        raise ValueError(
            f"grid shape {grid.shape} does not match "
            f"(len(y_values)={len(y_values)}, len(x_values)={len(x_values)})"
        )

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    vmin = norm.vmin
    vmax = norm.vmax
    threshold = (
        float(np.sqrt(vmin * vmax))
        if vmin is not None and vmax is not None and vmin > 0 and vmax > 0
        else None
    )

    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="equal", origin="lower")

    if annotations:
        for yi in range(len(y_values)):
            for xi in range(len(x_values)):
                val = grid[yi, xi]
                if not np.isfinite(val):
                    continue
                color = (
                    "white" if threshold is not None and val < threshold else "black"
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
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")

    return im


def plot_iter_cdf(
    stats: Iterable[TaskStats],
    ax: Axes,
    *,
    min_iter: int | None = None,
    label_fn: Callable[[TaskStats], str],
    xscale: Literal["log", "linear"],
    show_legend: bool = True,
    legend_loc: str = "lower right",
    title: str | None = None,
) -> None:
    """Plot the cumulative distribution of iteration numbers, one curve for
    each ``stats`` entry.

    If ``min_iter`` is set, the view will be cropped to data points with iteration
    number at least this value.
    """
    if min_iter is None:
        min_iter = 1 if xscale == "log" else 0

    for s in stats:
        if not s.metadata.is_iterative:
            raise ValueError("Expect iterative decoders")
        hist = s.iters_hist_on_converged.copy()
        hist[-1] += s.shots - s.synd_matches
        assert int(np.sum(hist)) == s.shots > 0
        x = np.arange(len(hist))
        y = np.cumsum(hist) / s.shots * 100
        ax.plot(x[min_iter:], y[min_iter:], label=label_fn(s))

    ax.set_xscale(xscale)
    ax.set_xlabel("Iterations", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("CDF (%)", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(left=min_iter)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.grid(True, which="both", alpha=0.5)
    if show_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, loc=legend_loc)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
