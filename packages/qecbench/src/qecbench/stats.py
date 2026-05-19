"""BenchmarkStats dataclass for accumulating Monte Carlo benchmark results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional
from functools import cached_property

import numpy as np
from qecdec.decoders import DECODERS_REGISTRY, ITERATIVE_DECODERS_REGISTRY
from qecdec.types import Int1DArray, Bool1DArray


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TaskMetadata:
    """Immutable metadata identifying a benchmark task.

    Attributes
    ----------
    circuit_name : str
        Circuit name.
    circuit_params : dict[str, Any]
        Circuit-specific parameters (JSON-serializable).
    p : float
        Physical error rate.
    decoder_name : str
        Decoder name.
    decoder_params : dict[str, Any]
        Decoder-specific parameters (JSON-serializable). Empty dict for
        parameterless decoders.
    decoder_label : str or None
        Display string for plots and CSVs. ``None`` falls back to ``decoder_name``.
        A different string from ``decoder_name`` is useful when there can be
        multiple instances associated with a common decoder name (e.g. different
        PyTorch checkpoints for ``"LearnedDMemBP"``).
    schema_version : int
        Schema version used to track changes of the dataclass fields.
    """

    circuit_name: str
    circuit_params: dict[str, Any]
    p: float
    decoder_name: str
    decoder_params: dict[str, Any]
    decoder_label: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.decoder_name not in DECODERS_REGISTRY:
            raise ValueError(f"Invalid decoder name: {self.decoder_name!r}")

        # If decoder_label is not specified, use decoder_name.
        if self.decoder_label is None:
            object.__setattr__(self, "decoder_label", self.decoder_name)

    def _canonical_tuple(self) -> tuple:
        return (
            self.circuit_name,
            json.dumps(self.circuit_params, sort_keys=True),
            self.p,
            self.decoder_name,
            json.dumps(self.decoder_params, sort_keys=True),
            self.decoder_label,
            self.schema_version,
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, TaskMetadata):
            return NotImplemented
        return self._canonical_tuple() == other._canonical_tuple()

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


METADATA_COLUMNS = [f.name for f in fields(TaskMetadata)]
COUNTER_COLUMNS = ["shots", "obser_correct", "synd_matches", "success"]
HIST_COLUMNS = ["iters_hist_on_converged", "iters_hist_on_success"]
ALL_COLUMNS = METADATA_COLUMNS + COUNTER_COLUMNS + HIST_COLUMNS


@dataclass(eq=False)
class BenchmarkStats:
    """Monte Carlo benchmark statistics for one task.

    Attributes
    ----------
    metadata : TaskMetadata
        Immutable metadata that uniquely identifies the benchmark task.
    shots : int
        Total number of shots.
    obser_correct : int
        Number of shots where the predicted observables are all correct.
    synd_matches : int
        Number of shots where the predicted error pattern satisfies the syndrome.
    success : int
        Number of successful shots (i.e., both syndrome matched and observables correct).
    iters_hist_on_converged : Int1DArray or None
        Applicable to iterative decoders only. Histogram of the number of iterations for all
        converged (i.e., syndrome matched) shots. The array has length ``metadata.max_iter + 1``
        with the following bins:

        - ``iters_hist[0]``: number of shots with all-zero syndrome (no decoding at all).
        - ``iters_hist[i]`` (``1 <= i <= metadata.max_iter``): number of shots that took exactly
        ``i`` iterations for the decoder to converge.

    iters_hist_on_success : Int1DArray or None
        Applicable to iterative decoders only. Same as ``iters_hist_on_converged``, but counting
        successful shots only.
    """

    metadata: TaskMetadata
    shots: int = 0
    obser_correct: int = 0
    synd_matches: int = 0
    success: int = 0
    iters_hist_on_converged: Optional[Int1DArray] = None
    iters_hist_on_success: Optional[Int1DArray] = None

    def __post_init__(self):
        if self.metadata.is_iterative:
            # Initialize histograms if not provided
            if self.iters_hist_on_converged is None:
                self.iters_hist_on_converged = np.zeros(
                    self.metadata.max_iter + 1, dtype=np.int64
                )
            if self.iters_hist_on_success is None:
                self.iters_hist_on_success = np.zeros(
                    self.metadata.max_iter + 1, dtype=np.int64
                )
        self._validate()

    def _validate(self) -> None:
        """Validate the data. For debug usage."""
        assert 0 <= self.success <= self.obser_correct <= self.shots
        assert 0 <= self.success <= self.synd_matches <= self.shots
        if self.metadata.is_iterative:
            assert isinstance(self.iters_hist_on_converged, np.ndarray) and isinstance(
                self.iters_hist_on_success, np.ndarray
            )
            assert (
                len(self.iters_hist_on_converged)
                == len(self.iters_hist_on_success)
                == self.metadata.max_iter + 1
            )
            assert int(np.sum(self.iters_hist_on_converged)) == self.synd_matches
            assert int(np.sum(self.iters_hist_on_success)) == self.success
            assert np.all(self.iters_hist_on_success <= self.iters_hist_on_converged)
        else:
            assert (
                self.iters_hist_on_converged is None
                and self.iters_hist_on_success is None
            )

    @property
    def obser_errors(self) -> int:
        """Number of shots where the predicted observables are not all correct."""
        return self.shots - self.obser_correct

    @property
    def synd_mismatches(self) -> int:
        """Number of shots where the predicted error pattern does not satisfy the syndrome."""
        return self.shots - self.synd_matches

    @property
    def failures(self) -> int:
        """Number of decoding failures (i.e., either syndrome mismatched or observables incorrect)."""
        return self.shots - self.success

    # -- Completion check ------------------------------------------------------

    def is_complete(self, shots_cap: int, errors_cap: int) -> bool:
        """
        Check if the benchmark task is complete: either the total number of shots
        reaches ``shots_cap``, or the number of shots with incorrect observable
        predictions reaches ``errors_cap``.
        """
        return self.shots >= shots_cap or self.obser_errors >= errors_cap

    # -- Batch update & merge --------------------------------------------------

    def update(
        self,
        obser_correct_mask: Bool1DArray,
        synd_match_mask: Bool1DArray,
        decoding_iters: Int1DArray | None = None,
    ) -> None:
        """Accumulate results from one decoded batch.

        Parameters
        ----------
        obser_correct_mask : Bool1DArray
            Boolean mask, shape=(batch_size,). True when the predicted observables are correct for that shot.

        synd_match_mask : Bool1DArray
            Boolean mask, shape=(batch_size,). True when the predicted error pattern satisfies the syndrome for that shot.

        decoding_iters : Int1DArray or None
            Number of iterations the decoder runs for each shot, shape=(batch_size,). 0 means the syndrome for that shot
            was all-zero (no decoding at all). ``None`` for non-iterative decoders.
        """
        success_mask = synd_match_mask & obser_correct_mask
        self.shots += len(obser_correct_mask)
        self.obser_correct += int(np.sum(obser_correct_mask))
        self.synd_matches += int(np.sum(synd_match_mask))
        self.success += int(np.sum(success_mask))

        if self.metadata.is_iterative:
            self.iters_hist_on_converged += np.bincount(
                decoding_iters[synd_match_mask], minlength=self.metadata.max_iter + 1
            )
            self.iters_hist_on_success += np.bincount(
                decoding_iters[success_mask], minlength=self.metadata.max_iter + 1
            )

    def merge(self, other: BenchmarkStats) -> None:
        """Merge another ``BenchmarkStats`` into this one in-place."""
        if self.metadata != other.metadata:
            raise ValueError("Cannot merge BenchmarkStats with different metadata.")

        self.shots += other.shots
        self.obser_correct += other.obser_correct
        self.synd_matches += other.synd_matches
        self.success += other.success

        if self.metadata.is_iterative:
            self.iters_hist_on_converged += other.iters_hist_on_converged
            self.iters_hist_on_success += other.iters_hist_on_success

    # -- Derived metrics -------------------------------------------------------

    @property
    def logical_error_rate(self) -> float:
        """The fraction of shots where the predicted observables are not all correct."""
        return self.obser_errors / self.shots if self.shots > 0 else float("nan")

    @property
    def syndrome_mismatch_rate(self) -> float:
        """The fraction of shots where the predicted error pattern does not satisfy the syndrome."""
        return self.synd_mismatches / self.shots if self.shots > 0 else float("nan")

    @property
    def failure_rate(self) -> float:
        """The fraction of shots where the decoding failed (i.e., either syndrome mismatched or observables incorrect)."""
        return self.failures / self.shots if self.shots > 0 else float("nan")

    @property
    def avg_iters_on_converged(self) -> float:
        """The average number of iterations over converged shots.

        Raise ``ValueError`` for non-iterative decoders.
        """
        if not self.metadata.is_iterative:
            raise ValueError("Unsupported for non-iterative decoders.")
        return (
            float(
                np.average(
                    np.arange(self.metadata.max_iter + 1),
                    weights=self.iters_hist_on_converged,
                )
            )
            if self.synd_matches > 0
            else float("nan")
        )

    @property
    def avg_iters_on_success(self) -> float:
        """The average number of iterations over successful shots.

        Raise ``ValueError`` for non-iterative decoders.
        """
        if not self.metadata.is_iterative:
            raise ValueError("Unsupported for non-iterative decoders.")
        return (
            float(
                np.average(
                    np.arange(self.metadata.max_iter + 1),
                    weights=self.iters_hist_on_success,
                )
            )
            if self.success > 0
            else float("nan")
        )

    @property
    def avg_iters(self) -> float:
        """
        The average number of iterations the decoder ran. This includes unconverged
        shots as well, for which the number of iterations is `self.metadata.max_iter`.

        Raise ``ValueError`` for non-iterative decoders.
        """
        if not self.metadata.is_iterative:
            raise ValueError("Unsupported for non-iterative decoders.")
        return (
            float(
                np.sum(
                    np.arange(self.metadata.max_iter + 1) * self.iters_hist_on_converged
                )
                + self.metadata.max_iter * self.synd_mismatches
            )
            / self.shots
            if self.shots > 0
            else float("nan")
        )

    # -- CSV I/O ---------------------------------------------------------------

    @staticmethod
    def save_csv(stats_list: list[BenchmarkStats], path: Path | str) -> None:
        """
        Write a list of ``BenchmarkStats`` to a CSV file at ``path``, one row for each
        ``BenchmarkStats`` and one column for each element in ``ALL_COLUMNS``.
        If a file already exists at ``path``, it will be overwritten.

        ``metadata.circuit_params`` and ``metadata.decoder_params`` are serialized as
        JSON with ``sort_keys=True``, so equal dicts produce equal strings regardless
        of key insertion order.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            for s in stats_list:
                m = s.metadata
                row = {
                    "circuit_name": m.circuit_name,
                    "circuit_params": json.dumps(m.circuit_params, sort_keys=True),
                    "p": m.p,
                    "decoder_name": m.decoder_name,
                    "decoder_params": json.dumps(m.decoder_params, sort_keys=True),
                    "decoder_label": m.decoder_label,
                    "schema_version": m.schema_version,
                    "shots": s.shots,
                    "obser_correct": s.obser_correct,
                    "synd_matches": s.synd_matches,
                    "success": s.success,
                    "iters_hist_on_converged": (
                        json.dumps(s.iters_hist_on_converged.tolist())
                        if m.is_iterative
                        else ""
                    ),
                    "iters_hist_on_success": (
                        json.dumps(s.iters_hist_on_success.tolist())
                        if m.is_iterative
                        else ""
                    ),
                }
                writer.writerow(row)

    @staticmethod
    def load_csv(path: Path) -> list[BenchmarkStats]:
        """
        Load a list of ``BenchmarkStats`` from a CSV file at ``path``, one for each row.
        If the file does not exist, return an empty list.

        Raise ``ValueError`` on schema mismatch (CSVs from older schema versions
        must be migrated manually).
        """
        if not path.exists():
            return []

        def _intarray_or_none(v: str) -> np.ndarray | None:
            return np.array(json.loads(v), dtype=np.int64) if v != "" else None

        stats_list: list[BenchmarkStats] = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                schema_version = int(row["schema_version"])
                if schema_version != SCHEMA_VERSION:
                    raise ValueError(
                        f"CSV at {path} has entries with schema_version={schema_version}; "
                        f"this build expects {SCHEMA_VERSION}. Please migrate the file."
                    )
                metadata = TaskMetadata(
                    circuit_name=row["circuit_name"],
                    circuit_params=json.loads(row["circuit_params"]),
                    p=float(row["p"]),
                    decoder_name=row["decoder_name"],
                    decoder_params=json.loads(row["decoder_params"]),
                    decoder_label=row["decoder_label"],
                )
                stats_list.append(
                    BenchmarkStats(
                        metadata=metadata,
                        shots=int(row["shots"]),
                        obser_correct=int(row["obser_correct"]),
                        synd_matches=int(row["synd_matches"]),
                        success=int(row["success"]),
                        iters_hist_on_converged=_intarray_or_none(
                            row.get("iters_hist_on_converged", "")
                        ),
                        iters_hist_on_success=_intarray_or_none(
                            row.get("iters_hist_on_success", "")
                        ),
                    )
                )
        return stats_list

    # -- List operations -------------------------------------------------------

    @staticmethod
    def find_by_metadata(
        stats_list: list[BenchmarkStats],
        metadata: TaskMetadata,
    ) -> BenchmarkStats | None:
        """
        Find the ``BenchmarkStats`` in ``stats_list`` that matches the given ``metadata``,
        assuming that at most one can be found. Return ``None`` if not found.
        """
        for s in stats_list:
            if s.metadata == metadata:
                return s
        return None

    @staticmethod
    def upsert(
        stats_list: list[BenchmarkStats],
        new_stats: BenchmarkStats,
    ) -> None:
        """
        Replace the entry in ``stats_list`` with ``new_stats`` that has the same metadata
        (assuming that at most one can be found), or append ``new_stats`` to the list if no
        such entry exists. This operation is in-place for ``stats_list``.
        """
        for i in range(len(stats_list)):
            if stats_list[i].metadata == new_stats.metadata:
                stats_list[i] = new_stats
                return
        stats_list.append(new_stats)
