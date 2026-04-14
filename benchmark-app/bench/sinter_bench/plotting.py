"""Plotting utilities for sinter-based benchmark results."""

from typing import Literal

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import sinter

from ..constants import ALL_BASELINE_DECODERS
from ..params import QECParams


def plot_ler_vs_per(
    stats: list[sinter.TaskStats],
    *,
    qec_params: QECParams,
    ler_mode: Literal["per shot", "per round"],
) -> matplotlib.figure.Figure:
    """Plot Logical Error Rate vs Physical Error Rate using sinter's plotting API.

    Returns the matplotlib Figure.
    """

    def group_func(stat: sinter.TaskStats) -> dict:
        decoder = stat.json_metadata["decoder_name"]
        return {
            "label": decoder,
            "linestyle": "dashed" if decoder in ALL_BASELINE_DECODERS else "solid",
        }

    fig, ax = plt.subplots(1, 1)
    common_kwargs = dict(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=group_func,
        plot_args_func=lambda index, group_key: {"linestyle": group_key["linestyle"]},
    )
    if ler_mode == "per shot":
        sinter.plot_error_rate(**common_kwargs)
        ax.set_ylabel("LER per shot")
    else:
        sinter.plot_error_rate(
            **common_kwargs,
            failure_units_per_shot_func=lambda stat: stat.json_metadata["rounds"],
        )
        ax.set_ylabel("LER per round")

    ax.loglog()
    ax.grid(axis="y")
    ax.set_title(
        f"{qec_params.code}, {qec_params.noise_model}, "
        f"d={qec_params.d}, rounds={qec_params.rounds}, basis={qec_params.basis}"
    )
    ax.set_xlabel("PER")
    ax.legend()

    return fig
