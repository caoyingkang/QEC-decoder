class Curriculum:
    """
    Sample-level curriculum learning: Upweight non-converged shots by a factor of
    `(1 + hard_emphasis)` relative to converged shots, where `hard_emphasis` ramps
    linearly from 0 to `max_emphasis` over `ramp_epochs`, then stays at `max_emphasis`
    for the remainder of training.

    This class must be combined with a training callback that calls `update()`
    at the start of each epoch.
    """

    def __init__(self, max_emphasis: float, ramp_epochs: int):
        if max_emphasis < 0:
            raise ValueError(
                f"max_emphasis must be non-negative, but got {max_emphasis}"
            )
        if ramp_epochs < 1:
            raise ValueError(f"ramp_epochs must be at least 1, but got {ramp_epochs}")
        self._max_emphasis = max_emphasis
        self._ramp_epochs = ramp_epochs

        self.hard_emphasis = 0.0

    def update(self, current_epoch: int):
        self.hard_emphasis = min(
            self._max_emphasis, (current_epoch / self._ramp_epochs) * self._max_emphasis
        )
