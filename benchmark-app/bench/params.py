from typing import NamedTuple


class BenchTaskParams(NamedTuple):
    """Parameters for the benchmark task."""

    max_iter: int
    p_list: list[float]
    use_prior_in_ckpt: bool


class QECParams(NamedTuple):
    """Parameters for the QEC experiment."""

    code: str
    noise_model: str
    d: int
    rounds: int
    basis: str
