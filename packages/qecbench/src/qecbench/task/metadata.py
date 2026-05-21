from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from typing import Any, ClassVar

from frozendict import frozendict
from qecdec.decoders import DECODERS_REGISTRY, ITERATIVE_DECODERS_REGISTRY
from qecdec.circuits import CIRCUITS_REGISTRY


_METADATA_COLUMNS = [
    "circuit_name",
    "circuit_params",
    "error_rate",
    "decoder_name",
    "decoder_params",
    "decoder_label",
    "SCHEMA_VERSION",
]


@dataclass(frozen=True, eq=True)
class TaskMetadata:
    """Immutable dataclass identifying a benchmark task.

    Attributes
    ----------
    circuit_name : str
        Circuit name.
    circuit_params : frozendict[str, Any]
        Circuit-specific parameters (JSON-serializable).
    error_rate : float
        Physical error rate.
    decoder_name : str
        Decoder name.
    decoder_params : frozendict[str, Any]
        Decoder-specific parameters (JSON-serializable).
    decoder_label : str or None
        Display string for plots and CSVs. ``None`` falls back to ``decoder_name``.
        A different string from ``decoder_name`` is useful when there can be
        multiple instances associated with a common decoder name (e.g. different
        PyTorch checkpoints for ``"LearnedDMemBP"``).
    """

    # Class attribute (shared across all instances)
    SCHEMA_VERSION: ClassVar[int] = 1  # used to track changes of TaskMetadata fields

    # Instance attributes (fields)
    circuit_name: str
    circuit_params: frozendict[str, Any]
    error_rate: float
    decoder_name: str
    decoder_params: frozendict[str, Any]
    decoder_label: str = None

    def __post_init__(self) -> None:
        if self.circuit_name not in CIRCUITS_REGISTRY:
            raise ValueError(
                f"Invalid circuit name: {self.circuit_name!r}. "
                f"Available options: {list(CIRCUITS_REGISTRY.keys())}"
            )

        if self.decoder_name not in DECODERS_REGISTRY:
            raise ValueError(
                f"Invalid decoder name: {self.decoder_name!r}. "
                f"Available options: {list(DECODERS_REGISTRY.keys())}"
            )

        # If decoder_label is not specified, use decoder_name.
        if self.decoder_label is None:
            object.__setattr__(self, "decoder_label", self.decoder_name)

    @cached_property
    def is_iterative(self) -> bool:
        """True if the decoder is iterative."""
        return self.decoder_name in ITERATIVE_DECODERS_REGISTRY

    @cached_property
    def max_iter(self) -> int:
        """Max number of iterations, resolved from ``decoder_params``.

        Raise ``ValueError`` for non-iterative decoders.
        """
        if not self.is_iterative:
            raise ValueError("Non-iterative decoder has no max_iter.")
        return ITERATIVE_DECODERS_REGISTRY[self.decoder_name].max_iter_from_params(
            self.decoder_params
        )

    def to_csv_rowdict(self) -> dict[str, str | int | float]:
        """Convert to a dict that can be written to a CSV file as a row.

        ``circuit_params`` and ``decoder_params`` are serialized as JSON strings
        with ``sort_keys=True``, so equal params produce equal strings regardless of
        key insertion order.

        The class-attribute ``SCHEMA_VERSION`` is also included in the returned dict.
        """
        return {
            "circuit_name": self.circuit_name,
            "circuit_params": json.dumps(self.circuit_params, sort_keys=True),
            "error_rate": self.error_rate,
            "decoder_name": self.decoder_name,
            "decoder_params": json.dumps(self.decoder_params, sort_keys=True),
            "decoder_label": self.decoder_label,
            "SCHEMA_VERSION": self.SCHEMA_VERSION,
        }
