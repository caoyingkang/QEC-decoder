import numpy as np
from typing import Optional, Tuple


def sample_error_and_syndrome(cm: np.ndarray, error_rate: np.ndarray, N: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample error patterns and calculate syndromes.

    Parameters
    ----------
        cm : ndarray
            Check matrix, shape=(m, n), dtype=int, values in {0, 1}.

        error_rate : ndarray
            Probabilities for each error mechanism, shape (n,), dtype=float, values in (0, 0.5).

        N : int
            Number of samples.

        seed : int or None
            Random seed for reproducibility. If None, the random seed is not set.
    
    Returns
    -------
        error : ndarray
            Error patterns, shape=(N, n), dtype=int, values in {0, 1}.

        syndrome : ndarray
            Syndrome vectors, shape=(N, m), dtype=int, values in {0, 1}.
    """
    assert isinstance(cm, np.ndarray) and cm.dtype == int and np.all(np.isin(cm, [0, 1]))
    assert isinstance(error_rate, np.ndarray) and error_rate.dtype == float and \
        0 < error_rate.min() and error_rate.max() < 0.5

    m = cm.shape[0]
    n = cm.shape[1]
    assert error_rate.shape == (n,)

    if seed is not None:
        np.random.seed(seed)
    
    error = (np.random.rand(N, n) < error_rate).astype(int)
    syndrome = (error @ cm.T) % 2

    assert error.shape == (N, n) and error.dtype == int and np.all(np.isin(error, [0, 1]))
    assert syndrome.shape == (N, m) and syndrome.dtype == int and np.all(np.isin(syndrome, [0, 1]))

    return error, syndrome


def get_logical_error_rate(cm: np.ndarray, am: np.ndarray, true_error: np.ndarray, decoded_error: np.ndarray) -> float:
    """
    Calculate the logical error rate.

    Parameters
    ----------
        cm : ndarray
            Check matrix, shape=(m, n), dtype=int, values in {0, 1}.

        am : ndarray
            Action matrix, shape=(k, n), dtype=int, values in {0, 1}.

        true_error : ndarray
            Array of true error patterns, shape=(N, n), dtype=int, values in {0, 1}.

        decoded_error : ndarray
            Array of decoded error patterns, shape=(N, n), dtype=int, values in {0, 1}.

    Returns
    -------
        ler : float
            Logical error rate.
    """
    assert isinstance(cm, np.ndarray) and cm.dtype == int and np.all(
        np.isin(cm, [0, 1]))
    assert isinstance(am, np.ndarray) and am.dtype == int and np.all(
        np.isin(am, [0, 1]))
    assert cm.shape[1] == am.shape[1]

    n = cm.shape[1]

    assert isinstance(true_error, np.ndarray) and true_error.dtype == int and np.all(
        np.isin(true_error, [0, 1]))
    assert isinstance(decoded_error, np.ndarray)
    assert np.issubdtype(decoded_error.dtype, np.integer)
    assert np.all(np.isin(decoded_error, [0, 1]))
    assert true_error.shape == decoded_error.shape

    N = true_error.shape[0]

    residual_error = (true_error + decoded_error) % 2
    residual_syndrome = (residual_error @ cm.T) % 2
    residual_action = (residual_error @ am.T) % 2

    correct_mask = np.all(residual_syndrome == 0, axis=1) \
        & np.all(residual_action == 0, axis=1)
    correct_cnt = np.sum(correct_mask)

    return (N - correct_cnt) / N


def print_statistics(cm: np.ndarray, am: np.ndarray, true_error: np.ndarray, decoded_error: np.ndarray, num_round: Optional[int] = None):
    """
    Calculate statistics of the decoding performance.

    Parameters
    ----------
        cm : ndarray
            Check matrix, shape=(m, n), dtype=int, values in {0, 1}.

        am : ndarray
            Action matrix, shape=(k, n), dtype=int, values in {0, 1}.

        true_error : ndarray
            Array of true error patterns, shape=(N, n), dtype=int, values in {0, 1}.

        decoded_error : ndarray
            Array of decoded error patterns, shape=(N, n), dtype=int, values in {0, 1}.

        num_round : int or None
            Number of rounds of stabilizer measurement.
    """
    assert isinstance(cm, np.ndarray) and cm.dtype == int and np.all(np.isin(cm, [0, 1]))
    assert isinstance(am, np.ndarray) and am.dtype == int and np.all(np.isin(am, [0, 1]))
    assert cm.shape[1] == am.shape[1]

    n = cm.shape[1]

    assert isinstance(true_error, np.ndarray) and true_error.dtype == int and np.all(np.isin(true_error, [0, 1]))
    assert isinstance(decoded_error, np.ndarray)
    assert np.issubdtype(decoded_error.dtype, np.integer)
    assert np.all(np.isin(decoded_error, [0, 1]))
    assert true_error.shape == decoded_error.shape

    N = true_error.shape[0]

    residual_error = (true_error + decoded_error) % 2
    residual_syndrome = (residual_error @ cm.T) % 2
    residual_action = (residual_error @ am.T) % 2

    # find the samples with correct syndrome
    correct_syndrome_mask = np.all(residual_syndrome == 0, axis=1)
    correct_syndrome_cnt = np.sum(correct_syndrome_mask)

    # among those samples with correct syndrome, find the ones without logical error
    correct_decoding_mask = correct_syndrome_mask & np.all(residual_action == 0, axis=1)
    correct_decoding_cnt = np.sum(correct_decoding_mask)

    print("Total number of samples: ", N)
    print("Number of samples with incorrect syndrome: {}".format(N - correct_syndrome_cnt))
    print("Number of samples with correct syndrome but with logical error: {}".format(correct_syndrome_cnt - correct_decoding_cnt))

    # logical error rate
    ler = (N - correct_decoding_cnt) / N
    print("Decoding failure rate: {}%".format(ler * 100))

    if num_round is not None:
        # logical error rate per cycle
        ler_per_cycle = 1 - (1 - ler) ** (1.0 / num_round)
        print("Logical error rate per cycle: {}%".format(ler_per_cycle * 100))


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
    assert isinstance(p, np.ndarray) and p.dtype == float and 0 < p.min() and p.max() < 0.5
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

    assert ex.shape == (N, n) and ex.dtype == int and np.all(np.isin(ex, [0, 1]))
    assert ez.shape == (N, n) and ez.dtype == int and np.all(np.isin(ez, [0, 1]))

    return ex, ez
