import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple


def sample_error_and_syndrome(
    cm: sp.csr_matrix,
    error_rate: np.ndarray,
    N: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
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
        error : ndarray
            Array of error patterns, shape=(N, n), dtype=np.uint8, values in {0, 1}.

        syndrome : ndarray
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
    syndrome = ((error @ cm.T) % 2).astype(np.uint8)

    assert isinstance(error, np.ndarray)
    assert isinstance(syndrome, np.ndarray)
    assert error.shape == (N, n) and error.dtype == np.uint8
    assert syndrome.shape == (N, m) and syndrome.dtype == np.uint8
    assert np.all(np.isin(error, [0, 1]))
    assert np.all(np.isin(syndrome, [0, 1]))

    return error, syndrome


def get_logical_error_rate(
    cm: sp.csr_matrix,
    am: sp.csr_matrix,
    true_error: np.ndarray,
    decoded_error: np.ndarray
) -> float:
    """
    Calculate the logical error rate.

    Parameters
    ----------
        cm : csr_matrix
            Check matrix, shape=(m, n), values in {0, 1}.

        am : csr_matrix
            Action matrix, shape=(k, n), values in {0, 1}.

        true_error : ndarray
            Array of true error patterns, shape=(N, n), values in {0, 1}.

        decoded_error : ndarray
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

    assert isinstance(true_error, np.ndarray)
    assert np.all(np.isin(true_error, [0, 1]))
    assert isinstance(decoded_error, np.ndarray)
    assert np.all(np.isin(decoded_error, [0, 1]))
    assert true_error.shape == decoded_error.shape
    assert true_error.shape[1] == n

    N = true_error.shape[0]

    residual_error = (true_error + decoded_error) % 2
    residual_syndrome = (residual_error @ cm.T) % 2
    residual_action = (residual_error @ am.T) % 2

    correct_syndrome_mask = np.all(residual_syndrome == 0, axis=1)
    correct_action_mask = np.all(residual_action == 0, axis=1)
    correct_mask = correct_syndrome_mask & correct_action_mask
    correct_cnt = np.sum(correct_mask)

    return (N - correct_cnt) / N


def mod2(a: sp.csr_matrix) -> sp.csr_matrix:
    """
    Modulo 2 operation for a sparse matrix with integer values.
    """
    a.data %= 2
    a.eliminate_zeros()
    return a


def sample_depolarizing1_noise(
    n: int,
    p: np.ndarray,
    N: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample independent single-qubit depolarizing noise on n fault locations.

    Parameters
    ----------
        n : int
            Number of fault locations.

        p : ndarray
            Depolarizing error probabilities for each fault location, shape (n,), values in (0, 0.5).

        N : int
            Number of samples.

        seed : int or None
            Random seed for reproducibility. If None, the random seed is not set.

    Returns
    -------
        ex: ndarray
            X-component of the sampled Pauli errors, shape (N, n), values in {0, 1}.

        ez: ndarray
            Z-component of the sampled Pauli errors, shape (N, n), values in {0, 1}.
    """
    assert isinstance(p, np.ndarray)
    assert p.min() > 0 and p.max() < 0.5
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
    ex = (x_mask | y_mask).astype(np.uint8)
    ez = (z_mask | y_mask).astype(np.uint8)

    assert ex.shape == (N, n)
    assert np.all(np.isin(ex, [0, 1]))
    assert ez.shape == (N, n)
    assert np.all(np.isin(ez, [0, 1]))

    return ex, ez
