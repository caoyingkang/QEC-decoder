from __future__ import annotations

from pathlib import Path

import stim

from .base import Experiment


class StimFileExperiment(Experiment):
    """An experiment loaded from a stim circuit."""

    def __init__(self, circuit: stim.Circuit):
        self._circuit = circuit

    @property
    def circuit(self) -> stim.Circuit:
        """Stim circuit for the experiment."""
        return self._circuit

    @classmethod
    def load_from_file(cls, path: str | Path) -> StimFileExperiment:
        """Load a StimFileExperiment from a .stim file.

        Parameters
        ----------
        path : str | Path
            Path to the .stim file.

        Returns
        -------
        StimFileExperiment
            The experiment loaded from the file.
        """
        return cls(stim.Circuit.from_file(path))
