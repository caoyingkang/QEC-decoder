import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from qecdec import RotatedSurfaceCode, detector_error_model_to_check_matrices


def get_rsc_chkmat_obsmat_pvec(
    *,
    d: int,
    rounds: int,
    p: float,
    noise_model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    code = RotatedSurfaceCode(d)
    if noise_model == "circuit-level":
        circuit = code.make_circuit_memory_z_experiment(
            rounds=rounds,
            data_qubit_error_rate=p,
            meas_error_rate=p,
            prep_error_rate=p,
            gate1_error_rate=p,
            gate2_error_rate=p,
            keep_z_detectors_only=True)
    elif noise_model == "phenomenological":
        circuit = code.make_circuit_memory_z_experiment(
            rounds=rounds,
            data_qubit_error_rate=p,
            meas_error_rate=p,
            keep_z_detectors_only=True)
    else:
        raise ValueError(f"Invalid noise model: {noise_model}")

    dem = circuit.detector_error_model()
    matrices = detector_error_model_to_check_matrices(dem)
    chkmat, obsmat, pvec = matrices.check_matrix, matrices.observables_matrix, matrices.priors
    chkmat = chkmat.toarray().astype(np.uint8)
    obsmat = obsmat.toarray().astype(np.uint8)
    pvec = pvec.astype(np.float64)

    return chkmat, obsmat, pvec


def sample_rsc(
    *,
    d: int,
    rounds: int,
    p: float,
    noise_model: str,
    num_shots: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    code = RotatedSurfaceCode(d)
    if noise_model == "circuit-level":
        circuit = code.make_circuit_memory_z_experiment(
            rounds=rounds,
            data_qubit_error_rate=p,
            meas_error_rate=p,
            prep_error_rate=p,
            gate1_error_rate=p,
            gate2_error_rate=p,
            keep_z_detectors_only=True)
    elif noise_model == "phenomenological":
        circuit = code.make_circuit_memory_z_experiment(
            rounds=rounds,
            data_qubit_error_rate=p,
            meas_error_rate=p,
            keep_z_detectors_only=True)
    else:
        raise ValueError(f"Invalid noise model: {noise_model}")

    sampler = circuit.compile_detector_sampler(seed=seed)
    syndrome_batch, observable_batch = sampler.sample(
        num_shots, separate_observables=True)
    syndrome_batch = syndrome_batch.astype(np.uint8)
    observable_batch = observable_batch.astype(np.uint8)

    return syndrome_batch, observable_batch


def get_LER(
    *,
    chkmat: np.ndarray,
    obsmat: np.ndarray,
    syndrome_batch: np.ndarray,
    observable_batch: np.ndarray,
    ehat_batch: np.ndarray,
    matched_syndrome_required: bool = True
) -> float:
    assert isinstance(chkmat, np.ndarray)
    assert isinstance(obsmat, np.ndarray)
    assert isinstance(syndrome_batch, np.ndarray)
    assert isinstance(observable_batch, np.ndarray)
    assert isinstance(ehat_batch, np.ndarray)

    assert chkmat.ndim == 2
    assert obsmat.ndim == 2
    assert syndrome_batch.ndim == 2
    assert observable_batch.ndim == 2
    assert ehat_batch.ndim == 2

    m, n = chkmat.shape
    k = obsmat.shape[0]
    num_shots = syndrome_batch.shape[0]

    assert obsmat.shape == (k, n)
    assert syndrome_batch.shape == (num_shots, m)
    assert observable_batch.shape == (num_shots, k)
    assert ehat_batch.shape == (num_shots, n)

    syndrome_predict = (ehat_batch @ chkmat.T) % 2
    syndrome_unmatched_mask = np.any(
        syndrome_predict != syndrome_batch, axis=1)

    observable_predict = (ehat_batch @ obsmat.T) % 2
    observable_unmatched_mask = np.any(
        observable_predict != observable_batch, axis=1)

    if matched_syndrome_required:
        failure_mask = syndrome_unmatched_mask | observable_unmatched_mask
    else:
        failure_mask = observable_unmatched_mask

    return float(np.sum(failure_mask)) / num_shots


def print_decoding_performance(
    *,
    p: float,
    chkmat: np.ndarray,
    obsmat: np.ndarray,
    syndrome_batch: np.ndarray,
    observable_batch: np.ndarray,
    ehat_batch: np.ndarray,
) -> None:
    assert isinstance(chkmat, np.ndarray)
    assert isinstance(obsmat, np.ndarray)
    assert isinstance(syndrome_batch, np.ndarray)
    assert isinstance(observable_batch, np.ndarray)
    assert isinstance(ehat_batch, np.ndarray)

    assert chkmat.ndim == 2
    assert obsmat.ndim == 2
    assert syndrome_batch.ndim == 2
    assert observable_batch.ndim == 2
    assert ehat_batch.ndim == 2

    m, n = chkmat.shape
    k = obsmat.shape[0]
    num_shots = syndrome_batch.shape[0]

    assert obsmat.shape == (k, n)
    assert syndrome_batch.shape == (num_shots, m)
    assert observable_batch.shape == (num_shots, k)
    assert ehat_batch.shape == (num_shots, n)

    syndrome_predict = (ehat_batch @ chkmat.T) % 2
    syndrome_unmatched_mask = np.any(
        syndrome_predict != syndrome_batch, axis=1)

    observable_predict = (ehat_batch @ obsmat.T) % 2
    observable_unmatched_mask = np.any(
        observable_predict != observable_batch, axis=1)

    print("""Physical error rate {:.2e}: Out of {} shots, there are
-> {} shots with unmatched syndromes, among which {} have wrong observable predictions;
-> {} shots with matched syndromes, among which {} have wrong observable predictions.""".format(
        p,
        num_shots,
        np.sum(syndrome_unmatched_mask),
        np.sum(syndrome_unmatched_mask & observable_unmatched_mask),
        np.sum(~syndrome_unmatched_mask),
        np.sum((~syndrome_unmatched_mask) & observable_unmatched_mask),
    ))


def plot_tanner_graph_noninteractively(
    chkmat: np.ndarray,
    chk_coords: np.ndarray,
    var_coords: np.ndarray,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot variable nodes as circles (unfilled, black border)
    ax.scatter(var_coords[:, 0], var_coords[:, 1], var_coords[:, 2],
               c="white", s=80, marker="o", edgecolors="black", label="Variable Nodes")

    # Plot check nodes as squares (unfilled, black border)
    ax.scatter(chk_coords[:, 0], chk_coords[:, 1], chk_coords[:, 2],
               c="white", s=80, marker="s", edgecolors="black", label="Check Nodes")

    # Plot edges
    m, n = chkmat.shape
    for i in range(m):
        for j in range(n):
            if chkmat[i, j] == 1:
                x_vals = [var_coords[j, 0], chk_coords[i, 0]]
                y_vals = [var_coords[j, 1], chk_coords[i, 1]]
                z_vals = [var_coords[j, 2], chk_coords[i, 2]]
                ax.plot(x_vals, y_vals, z_vals, c="gray", linewidth=1)

    ax.legend()
    plt.show()


def plot_tanner_graph_interactively(
    chkmat: np.ndarray,
    chk_coords: np.ndarray,
    var_coords: np.ndarray,
    *,
    zaxis_stretch: float = 4
):
    fig = go.Figure()
    marker_size = 5

    # Check nodes
    fig.add_trace(go.Scatter3d(
        x=chk_coords[:, 0],
        y=chk_coords[:, 1],
        z=chk_coords[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color="white", symbol="square",
                    line=dict(color="black", width=1)),
        name="Check Node"
    ))

    # Variable nodes
    fig.add_trace(go.Scatter3d(
        x=var_coords[:, 0],
        y=var_coords[:, 1],
        z=var_coords[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color="orange", symbol="circle",
                    line=dict(color="black", width=1)),
        name="Variable Node"
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
        name="Edges"
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

    fig.show()


def visualize_bp_decoding_process(
    chkmat: np.ndarray,
    chk_coords: np.ndarray,
    var_coords: np.ndarray,
    syndrome: np.ndarray,
    llr_history: np.ndarray,
    *,
    zaxis_stretch: float = 4
):
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
                colorbar=dict(title="LLR", x=1.14, y=0.3,
                              orientation="h", len=0.25),
                symbol="circle",
                line=dict(color="black", width=1),
            ),
            name="Variable Node",
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
