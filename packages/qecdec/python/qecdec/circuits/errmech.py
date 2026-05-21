from dataclasses import dataclass
from functools import total_ordering


@total_ordering
@dataclass(frozen=True)
class ErrorMechanism:
    """Immutable dataclass representing an error mechanism.

    Attributes
    ----------
    dets : tuple[int, ...]
        Flipped detectors (in increasing order), must be nonempty.
    obsers : tuple[int, ...]
        Flipped observables (in increasing order).
    p : float
        Probability of occurrence.
    """

    dets: tuple[int, ...]
    obsers: tuple[int, ...]
    p: float

    def __post_init__(self) -> None:
        if len(self.dets) == 0:
            raise ValueError("`dets` cannot be empty")

    def __eq__(self, other) -> bool:
        """
        Two error mechanisms are considered equal if they flip the same set of detectors. In most cases, this
        guarantees that they also flip the same set of observables, because otherwise the fault distance of the
        circuit would be at most 2.
        """
        if not isinstance(other, ErrorMechanism):
            return NotImplemented
        return self.dets == other.dets

    def __lt__(self, other) -> bool:
        if not isinstance(other, ErrorMechanism):
            return NotImplemented
        return self.dets < other.dets
