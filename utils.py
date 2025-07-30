import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple


def sample_error_and_syndrome(
    cm: sp.csr_matrix,
    error_rate: np.ndarray,
    N: int,
    seed: Optional[int] = None
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Sample error patterns and calculate syndromes.

    Parameters
    ----------
        cm : csr_matrix
            Check matrix, shape=(m, n), values in {0, 1}.

        error_rate : ndarray
            Probabilities for each error mechanism, shape=(n,), values in (0, 0.5).

        N : int
            Number of samples.

        seed : int or None
            Random seed for reproducibility. If None, the random seed is not set.

    Returns
    -------
        error : csr_matrix
            Array of error patterns, shape=(N, n), dtype=np.uint8, values in {0, 1}.

        syndrome : csr_matrix
            Array of syndrome vectors, shape=(N, m), dtype=np.uint8, values in {0, 1}.
    """
    assert isinstance(cm, sp.csr_matrix)
    assert np.all(cm.data == 1)
    assert isinstance(error_rate, np.ndarray)
    assert error_rate.min() > 0 and error_rate.max() < 0.5

    m, n = cm.shape
    assert error_rate.shape == (n,)

    if seed is not None:
        np.random.seed(seed)

    error = (np.random.rand(N, n) < error_rate).astype(np.uint8)
    error = sp.csr_matrix(error)
    syndrome = mod2(error @ cm.T).astype(np.uint8)

    assert isinstance(error, sp.csr_matrix)
    assert isinstance(syndrome, sp.csr_matrix)
    assert error.shape == (N, n) and error.dtype == np.uint8
    assert syndrome.shape == (N, m) and syndrome.dtype == np.uint8
    assert np.all(error.data == 1)
    assert np.all(syndrome.data == 1)

    return error, syndrome


def get_logical_error_rate(
    cm: sp.csr_matrix,
    am: sp.csr_matrix,
    true_error: sp.csr_matrix,
    decoded_error: sp.csr_matrix
) -> float:
    """
    Calculate the logical error rate.

    Parameters
    ----------
        cm : csr_matrix
            Check matrix, shape=(m, n), values in {0, 1}.

        am : csr_matrix
            Action matrix, shape=(k, n), values in {0, 1}.

        true_error : csr_matrix
            Array of true error patterns, shape=(N, n), values in {0, 1}.

        decoded_error : csr_matrix
            Array of decoded error patterns, shape=(N, n), values in {0, 1}.

    Returns
    -------
        ler : float
            Logical error rate.
    """
    assert isinstance(cm, sp.csr_matrix)
    assert np.all(cm.data == 1)
    assert isinstance(am, sp.csr_matrix)
    assert np.all(am.data == 1)
    assert cm.shape[1] == am.shape[1]

    n = cm.shape[1]

    assert isinstance(true_error, sp.csr_matrix)
    assert np.all(true_error.data == 1)
    assert isinstance(decoded_error, sp.csr_matrix)
    assert np.all(decoded_error.data == 1)
    assert true_error.shape == decoded_error.shape
    assert true_error.shape[1] == n

    N = true_error.shape[0]

    residual_error = mod2(true_error + decoded_error)
    residual_syndrome = mod2(residual_error @ cm.T)
    residual_action = mod2(residual_error @ am.T)

    correct_cnt = np.sum(
        (residual_syndrome.sum(axis=1) == 0) &
        (residual_action.sum(axis=1) == 0)
    )

    return (N - correct_cnt) / N


def mod2(a: sp.csr_matrix) -> sp.csr_matrix:
    """
    Modulo 2 operation for a sparse matrix with integer values.
    """
    a.data %= 2
    a.eliminate_zeros()
    return a


def sample_depolarizing1_noise(n: int, p: np.ndarray, N: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample independent single-qubit depolarizing noise on n fault locations.

    Parameters
    ----------
        n : int
            Number of fault locations.

        p : ndarray
            Depolarizing error probabilities for each fault location, shape (n,), dtype=float, values in (0, 0.5).

        N : int
            Number of samples.

        seed : int or None
            Random seed for reproducibility. If None, the random seed is not set.

    Returns
    -------
        ex: ndarray
            X-component of the sampled Pauli errors, shape (N, n), dtype=int, values in {0, 1}.

        ez: ndarray
            Z-component of the sampled Pauli errors, shape (N, n), dtype=int, values in {0, 1}.
    """
    assert isinstance(
        p, np.ndarray) and p.dtype == float and 0 < p.min() and p.max() < 0.5
    assert p.shape == (n,)

    if seed is not None:
        np.random.seed(seed)

    # sample erroneous locations
    err_loc_mask = np.random.rand(N, n) < p
    # sample Pauli error terms
    paulis = np.random.choice(['X', 'Y', 'Z'], size=(N, n))
    paulis[~err_loc_mask] = 'I'
    # extract X and Z components of the errors
    x_mask: np.ndarray = (paulis == 'X')
    y_mask: np.ndarray = (paulis == 'Y')
    z_mask: np.ndarray = (paulis == 'Z')
    ex = (x_mask | y_mask).astype(int)
    ez = (z_mask | y_mask).astype(int)

    assert ex.shape == (N, n) and ex.dtype == int and np.all(
        np.isin(ex, [0, 1]))
    assert ez.shape == (N, n) and ez.dtype == int and np.all(
        np.isin(ez, [0, 1]))

    return ex, ez
