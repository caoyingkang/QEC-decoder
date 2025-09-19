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
