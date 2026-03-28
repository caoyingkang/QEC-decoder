from typing import NamedTuple


class CollectorParams(NamedTuple):
    """Parameters for the Monte Carlo collector."""

    shots_cap: int
    errors_cap: int
    device: str
    num_workers: int
