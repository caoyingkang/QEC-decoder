from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
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
