import numpy as np
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
