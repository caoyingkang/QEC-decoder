import stim
import numpy as np


def ceil_div(a: int, b: int) -> int:
    """
    Compute the ceiling of the division `a/b`.
    """
    if a % b == 0:
        return a // b
    else:
        return a // b + 1


def extract_error_mechanisms_from_dem(dem: stim.DetectorErrorModel) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    """
    Extract error mechanisms from a stim.DetectorErrorModel object. Each error mechanism is identified by its effect, 
    which includes the set of detectors and the set of observables that are flipped. Error mechanisms with identical 
    effect will be combined into one.

    The output is a dict with each item representing an error mechanism. For each item, the key is a pair of tuples: 
    the first tuple contains the flipped detectors (sorted in increasing order), and the second tuple contains the 
    flipped observables (sorted in increasing order); the value is the net probability that the error mechanism occurs.
    """
    eff2prob = {}

    instruction: stim.DemInstruction
    for instruction in dem.flattened():
        if instruction.type == "error":
            p = instruction.args_copy()[0]  # probability of the error

            dets: set[int] = set()  # flipped detectors
            obsers: set[int] = set()  # flipped observables
            t: stim.DemTarget
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    if t.val in dets:
                        dets.remove(t.val)
                    else:
                        dets.add(t.val)
                elif t.is_logical_observable_id():
                    if t.val in obsers:
                        obsers.remove(t.val)
                    else:
                        obsers.add(t.val)
                elif t.is_separator():
                    pass
                else:
                    raise RuntimeError("Not supposed to be here")
            eff = (tuple(sorted(dets)), tuple(sorted(obsers)))
            if eff in eff2prob:  # this error has appeared earlier, let's update its probability
                eff2prob[eff] = (1 - eff2prob[eff]) * p + \
                    eff2prob[eff] * (1 - p)
            else:  # this error is new, let's register it
                eff2prob[eff] = p
        elif instruction.type == "detector" or instruction.type == "logical_observable":
            pass
        else:
            raise ValueError(
                f"Instruction type not expected: {instruction.type}")

    return eff2prob


def extract_detector_coords_from_dem(dem: stim.DetectorErrorModel) -> np.ndarray:
    """
    Extract detector coordinates from a stim.DetectorErrorModel object. The output is a numpy array that has num_detectors 
    rows, with the i-th row being the vector of coordinates of the i-th detector.
    """
    coords: list[list[float]] = [None] * dem.num_detectors

    instruction: stim.DemInstruction
    for instruction in dem.flattened():
        if instruction.type == "detector":
            coo = instruction.args_copy()
            t: stim.DemTarget = instruction.targets_copy()[0]
            coords[t.val] = coo
        elif instruction.type == "error" or instruction.type == "logical_observable":
            pass
        else:
            raise ValueError(
                f"Instruction type not expected: {instruction.type}")

    if None in coords:
        raise ValueError("Some detector coordinates are not found.")

    if not all(len(coords[i]) == len(coords[i+1]) for i in range(dem.num_detectors - 1)):
        raise ValueError("Detector coordinates have different lengths.")

    return np.array(coords)


Stat = dict[str, int | float]


def get_stats(
    chkmat: np.ndarray,
    obsmat: np.ndarray,
    syndromes: np.ndarray,
    observables: np.ndarray,
    ehats: np.ndarray,
) -> Stat:
    """
    Calculate decoding performance statistics.

    Parameters
    ----------
    chkmat : ndarray
        Check matrix, shape=(m, n).

    obsmat : ndarray
        Observable matrix, shape=(k, n).

    syndromes : ndarray
        Array of syndromes, shape=(shots, m).

    observables : ndarray
        Array of observables, shape=(shots, k).

    ehats : ndarray
        Array of decoded errors, shape=(shots, n).

    Returns
    -------
    Stat (alias for dict[str, int | float])
    """
    assert isinstance(chkmat, np.ndarray) and chkmat.ndim == 2
    assert isinstance(obsmat, np.ndarray) and obsmat.ndim == 2
    assert isinstance(syndromes, np.ndarray) and syndromes.ndim == 2
    assert isinstance(observables, np.ndarray) and observables.ndim == 2
    assert isinstance(ehats, np.ndarray) and ehats.ndim == 2
    m, n = chkmat.shape
    k = obsmat.shape[0]
    shots = syndromes.shape[0]
    assert obsmat.shape == (k, n)
    assert syndromes.shape == (shots, m)
    assert observables.shape == (shots, k)
    assert ehats.shape == (shots, n)
    chkmat = chkmat.astype(np.uint8)
    obsmat = obsmat.astype(np.uint8)
    syndromes = syndromes.astype(np.uint8)
    observables = observables.astype(np.uint8)
    ehats = ehats.astype(np.uint8)

    syndrome_pred = (ehats @ chkmat.T) % 2
    unmatched_syndrome_mask = np.any(syndrome_pred != syndromes, axis=1)

    observable_pred = (ehats @ obsmat.T) % 2
    unmatched_observable_mask = np.any(observable_pred != observables, axis=1)

    return {
        "shots": shots,
        "unmatched_syndrome_shots": np.sum(unmatched_syndrome_mask),
        "unmatched_observable_shots": np.sum(unmatched_observable_mask),
        "unmatched_syndrome_or_observable_shots": np.sum(unmatched_syndrome_mask | unmatched_observable_mask),
        "unmatched_syndrome_and_observable_shots": np.sum(unmatched_syndrome_mask & unmatched_observable_mask),
        "matched_syndrome_but_unmatched_observable_shots": np.sum(~unmatched_syndrome_mask & unmatched_observable_mask),
        "unmatched_syndrome_but_matched_observable_shots": np.sum(unmatched_syndrome_mask & ~unmatched_observable_mask),
    }


def bar_plot_stats(
    category2label2stats: dict[str, dict[str, Stat]],
    *,
    colors: list[str] = ["skyblue", "lightgreen", "salmon",
                         "khaki", "plum", "lightslategray"],
):
    """Plot bar charts for (1) number of shots with unmatched syndrome, and (2) number of shots with unmatched observable.
    Bars are grouped by category and displayed side by side. Within each category, bars are distinguished by label and are 
    assigned different colors.

    Parameters
    ----------
    category2label2stats : dict[str, dict[str, Stat]]
        Nested dictionary of statistics for each category and label.

    colors : list[str], optional
        Available colors for the bars. Must have length at least the number of labels.
    """
    import matplotlib.pyplot as plt

    categories = list(category2label2stats.keys())
    assert len(categories) > 0
    labels = list(category2label2stats[categories[0]].keys())
    assert len(labels) > 0
    assert all(
        set(category2label2stats[category].keys()) == set(labels)
        for category in categories
    )
    shots = category2label2stats[categories[0]][labels[0]]["shots"]
    assert all(category2label2stats[category][label]["shots"] == shots
               for category in categories for label in labels)
    if len(colors) < len(labels):
        raise ValueError("Not enough colors")

    # Construct data for plotting: label -> list of bar heights (one for each category)
    data_unmatched_syndrome = {}
    data_unmatched_observable = {}
    for label in labels:
        data_unmatched_syndrome[label] = [category2label2stats[category][label]["unmatched_syndrome_shots"]
                                          for category in categories]
        data_unmatched_observable[label] = [category2label2stats[category][label]["unmatched_observable_shots"]
                                            for category in categories]

    # Create figure and axes.
    ax1: plt.Axes
    ax2: plt.Axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Set up bar positions and width.
    width = 0.25
    stride = (len(labels) + 1) * width
    x = stride * np.arange(len(categories))

    # Plot 1: Unmatched Syndromes
    for i, label in enumerate(labels):
        bars = ax1.bar(x + i * width, data_unmatched_syndrome[label], width,
                       align="edge", label=label, alpha=0.8, color=colors[i])
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_title('Unmatched Syndrome (shots = {})'.format(shots))
    ax1.set_ylabel('Number of shots with unmatched syndrome')
    ax1.set_xticks(x + len(labels) * width / 2)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Unmatched Observables
    for i, label in enumerate(labels):
        bars = ax2.bar(x + i * width, data_unmatched_observable[label], width,
                       align="edge", label=label, alpha=0.8, color=colors[i])
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax2.set_title('Unmatched Observable (shots = {})'.format(shots))
    ax2.set_ylabel('Number of shots with unmatched observable')
    ax2.set_xticks(x + len(labels) * width / 2)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def stacked_bar_plot_stats(
    category2label2stats: dict[str, dict[str, Stat]],
    *,
    colors: list[str] = ["skyblue", "lightgreen", "salmon",
                         "khaki", "plum", "lightslategray"],
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    categories = list(category2label2stats.keys())
    assert len(categories) > 0
    labels = list(category2label2stats[categories[0]].keys())
    assert len(labels) > 0
    assert all(
        set(category2label2stats[category].keys()) == set(labels)
        for category in categories
    )
    shots = category2label2stats[categories[0]][labels[0]]["shots"]
    assert all(category2label2stats[category][label]["shots"] == shots
               for category in categories for label in labels)
    if len(colors) < len(labels):
        raise ValueError("Not enough colors")

    # Construct data for plotting: label -> list of bar heights (one for each category)
    data_unmatched_syndrome = {}
    data_unmatched_observable = {}
    data_unmatched_syndrome_or_observable = {}
    data_matched_syndrome_but_unmatched_observable = {}
    for label in labels:
        data_unmatched_syndrome[label] = [category2label2stats[category][label]["unmatched_syndrome_shots"]
                                          for category in categories]
        data_unmatched_observable[label] = [category2label2stats[category][label]["unmatched_observable_shots"]
                                            for category in categories]
        data_unmatched_syndrome_or_observable[label] = [category2label2stats[category][label]["unmatched_syndrome_or_observable_shots"]
                                                        for category in categories]
        data_matched_syndrome_but_unmatched_observable[label] = [category2label2stats[category][label]["matched_syndrome_but_unmatched_observable_shots"]
                                                                 for category in categories]
    # Create figure and axes.
    fig, ax = plt.subplots(figsize=(15, 6))

    # Set up bar positions and width.
    width = 0.25
    stride = (len(labels) + 1) * width
    x = stride * np.arange(len(categories))

    # Plot.
    for i, label in enumerate(labels):
        ax.bar(x + i * width, data_unmatched_syndrome_or_observable[label], width,
               label=label, align="edge", alpha=0.8, color=colors[i], edgecolor="black", linewidth=1)
    for i, label in enumerate(labels):
        ax.bar(x + i * width, data_unmatched_observable[label], width,
               align="edge", color="none", hatch="///", linewidth=1)
        ax.bar(x + i * width, data_unmatched_syndrome[label], width,
               bottom=data_matched_syndrome_but_unmatched_observable[label],
               align="edge", color="none", hatch="\\\\\\", linewidth=1)
        for j in range(len(categories)):
            y1 = data_matched_syndrome_but_unmatched_observable[label][j]
            y2 = data_unmatched_observable[label][j]
            y3 = data_unmatched_syndrome_or_observable[label][j]
            y_list = list(set([y1, y2]) - set([0, y3]))
            if len(y_list) > 0:
                ax.hlines(y_list, j * stride + i * width, j * stride + i * width + width,
                          colors="black", linewidth=1)

    ax.set_title('Decoding performance (shots = {})'.format(shots))
    ax.set_xticks(x + len(labels) * width / 2)
    ax.set_xticklabels(categories)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="none", edgecolor="black", hatch="///"))
    labels.append("unmatched observable")
    handles.append(Patch(facecolor="none", edgecolor="black", hatch="\\\\\\"))
    labels.append("unmatched syndrome")
    ax.legend(handles=handles, labels=labels)

    fig.tight_layout()
    plt.show()


def plot_tanner_graph_interactively(
    chkmat: np.ndarray,
    chk_coords: np.ndarray,
    var_coords: np.ndarray,
    *,
    zaxis_stretch: float = 4,
    html_filename: str = None,
):
    import plotly.graph_objects as go

    fig = go.Figure()
    marker_size = 5

    # Check nodes
    fig.add_trace(go.Scatter3d(
        x=chk_coords[:, 0],
        y=chk_coords[:, 1],
        z=chk_coords[:, 2],
        text=[str(i) for i in range(chkmat.shape[0])],
        mode="markers",
        marker=dict(size=marker_size, color="white", symbol="square",
                    line=dict(color="black", width=1)),
        name="Check Node",
        hovertemplate="Check node index: %{text}<br>x, y, t: %{x}, %{y}, %{z}<extra></extra>",
    ))

    # Variable nodes
    fig.add_trace(go.Scatter3d(
        x=var_coords[:, 0],
        y=var_coords[:, 1],
        z=var_coords[:, 2],
        text=[str(i) for i in range(chkmat.shape[1])],
        mode="markers",
        marker=dict(size=marker_size, color="orange", symbol="circle",
                    line=dict(color="black", width=1)),
        name="Variable Node",
        hovertemplate="Variable node index: %{text}<br>x, y, t: %{x}, %{y}, %{z}<extra></extra>",
    ))

    # Edges
    edge_x, edge_y, edge_z = [], [], []
    for i in range(chkmat.shape[0]):
        for j in range(chkmat.shape[1]):
            if chkmat[i, j] == 1:
                edge_x += [var_coords[j, 0], chk_coords[i, 0], None]
                edge_y += [var_coords[j, 1], chk_coords[i, 1], None]
                edge_z += [var_coords[j, 2], chk_coords[i, 2], None]

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="gray", width=2),
        name="Edges",
        hoverinfo="skip",
        hovertemplate=None,
    ))

    # Set scene parameters
    fig.update_scenes(
        aspectmode="manual",
        xaxis=dict(dtick=2, tick0=0),
        yaxis=dict(dtick=2, tick0=0, range=[0, var_coords[:, 1].max() + 1]),
        zaxis=dict(title="t", dtick=1, tick0=0),
        aspectratio=dict(x=1, y=1, z=zaxis_stretch),
        camera=dict(
            up=dict(x=0, y=-1, z=0),
            eye=dict(x=-2.5, y=0, z=0),
        ),
    )

    if html_filename is None:
        fig.show()
    else:
        fig.write_html(html_filename)


def visualize_bp_decoding_process(
    chkmat: np.ndarray,
    chk_coords: np.ndarray,
    var_coords: np.ndarray,
    syndrome: np.ndarray,
    llr_history: np.ndarray,
    *,
    zaxis_stretch: float = 4
):
    import plotly.graph_objects as go

    num_iters = llr_history.shape[0]

    fig = go.Figure()
    marker_size = 5

    # Check nodes
    mask0 = syndrome == 0
    fig.add_trace(go.Scatter3d(
        x=chk_coords[mask0, 0],
        y=chk_coords[mask0, 1],
        z=chk_coords[mask0, 2],
        mode="markers",
        marker=dict(size=marker_size, color="white", symbol="square",
                    line=dict(color="black", width=1)),
        name="Check Node (syndrome = 0)",
        hovertemplate="x, y, t: %{x}, %{y}, %{z}<extra></extra>",
        visible=True
    ))
    mask1 = syndrome == 1
    fig.add_trace(go.Scatter3d(
        x=chk_coords[mask1, 0],
        y=chk_coords[mask1, 1],
        z=chk_coords[mask1, 2],
        mode="markers",
        marker=dict(size=marker_size, color="black", symbol="square",
                    line=dict(color="black", width=1)),
        name="Check Node (syndrome = 1)",
        hovertemplate="x, y, t: %{x}, %{y}, %{z}<extra></extra>",
        visible=True
    ))

    # Variable nodes
    for it in range(num_iters):
        fig.add_trace(go.Scatter3d(
            x=var_coords[:, 0],
            y=var_coords[:, 1],
            z=var_coords[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size, color=llr_history[it], cmin=-10, cmax=10,
                colorscale=[[0, 'rgb(255,0,0)'],
                            [0.5, 'rgb(255,255,255)'],
                            [1, 'rgb(0,0,255)']],
                colorbar=dict(title="LLR", x=1.12, y=0.3,
                              orientation="h", len=0.25),
                symbol="circle",
                line=dict(color="black", width=1),
            ),
            name="Variable Node",
            hovertemplate="x, y, t: %{x}, %{y}, %{z}<br>LLR: %{marker.color:.3f}<extra></extra>",
            visible=(it == 0)
        ))

    # Edges
    edge_x, edge_y, edge_z = [], [], []
    for i in range(chkmat.shape[0]):
        for j in range(chkmat.shape[1]):
            if chkmat[i, j] == 1:
                edge_x += [var_coords[j, 0], chk_coords[i, 0], None]
                edge_y += [var_coords[j, 1], chk_coords[i, 1], None]
                edge_z += [var_coords[j, 2], chk_coords[i, 2], None]

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="gray", width=2),
        name="Edges",
        hoverinfo="skip",
        hovertemplate=None,
        visible=True
    ))

    # Helper function
    def get_visibility(it):
        return [True, True] + [i == it for i in range(num_iters)] + [True]

    # Add sliders
    sliders = [dict(
        active=0,
        steps=[dict(method="update",
                    label="Iteration {}".format(it),
                    args=[{"visible": get_visibility(it)}])
               for it in range(num_iters)]
    )]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False)
        )
    )

    # Set scene parameters
    fig.update_scenes(
        aspectmode="manual",
        xaxis=dict(dtick=2, tick0=0),
        yaxis=dict(dtick=2, tick0=0, range=[0, var_coords[:, 1].max() + 1]),
        zaxis=dict(title="t", dtick=1, tick0=0),
        aspectratio=dict(x=1, y=1, z=zaxis_stretch),
        camera=dict(
            up=dict(x=0, y=-1, z=0),
            eye=dict(x=-2.5, y=0, z=0),
        ),
    )

    fig.show()
