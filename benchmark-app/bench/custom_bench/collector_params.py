from typing import NamedTuple


class CollectorParams(NamedTuple):
    """Parameters for the Monte Carlo collector."""

    batch_size: int
    shots_cap: int
    errors_cap: int
    device: str
    num_parallel_workers: int
