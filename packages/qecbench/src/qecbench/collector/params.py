from dataclasses import dataclass


@dataclass(frozen=True)
class CollectorParams:
    """Immutable dataclass specifying Monte Carlo collector parameters.

    Attributes
    ----------
    batch_size : int
        Number of shots in one batch.
    shots_cap, errors_cap : int
        Monte Carlo collection is considered complete when either the total
        number of shots reaches ``shots_cap``, or the number of shots with
        incorrect observable predictions reaches ``errors_cap``.
    num_parallel_workers : int
        Number of parallel worker processes (0 = serial, >0 = multiprocessing).
    """

    batch_size: int
    shots_cap: int
    errors_cap: int
    num_parallel_workers: int

    @property
    def use_multiprocessing(self) -> bool:
        return self.num_parallel_workers > 0
