"""Parameter bundles used throughout qecbench.

The four ``NamedTuple``s here form the public configuration surface:

- :class:`QECParams` identifies the QEC code / noise model / shape under test.
- :class:`BenchTaskParams` enumerates the physical error rates and per-decoder
  configuration dicts.
- :class:`CollectorParams` controls the Monte Carlo collector (batch size,
  stopping criteria, parallelism).
- :class:`TorchDecoderTask` packages everything a PyTorch decoder run needs to
  be benchmarked: display name, model config, checkpoint path, and the CSV
  path where its results are stored.
"""

from pathlib import Path
from typing import Any, NamedTuple

from omegaconf import DictConfig


class QECParams(NamedTuple):
    """Parameters identifying a QEC experiment."""

    code: str
    noise_model: str
    d: int
    rounds: int
    basis: str


class BenchTaskParams(NamedTuple):
    """Parameters for the benchmark task.

    Attributes
    ----------
    p_list : list[float]
        Physical error rates to benchmark at.
    baseline_decoder_params : dict[str, dict]
        Maps each selected baseline decoder name to its config dict. Example:
        ``{"BP": {"max_iter": 50}, "MWPM": {}, "RelayBP": {"gamma0": 0.0, ...}}``.
    torchdecoder_shared_params : dict[str, Any]
        Shared parameters for all PyTorch decoder runs. Example:
        ``{"use_prior_in_ckpt": True, "max_iter": 50}``.
    """

    p_list: list[float]
    baseline_decoder_params: dict[str, dict]
    torchdecoder_shared_params: dict[str, Any]


class CollectorParams(NamedTuple):
    """Parameters for the Monte Carlo collector."""

    batch_size: int
    shots_cap: int
    errors_cap: int
    device: str
    num_parallel_workers: int


class TorchDecoderTask(NamedTuple):
    """One PyTorch decoder run to benchmark.

    Callers (e.g. the Streamlit app) resolve a Lightning run directory into
    the four fields below before handing the task to
    :func:`qecbench.runner.run_custom_benchmark`. qecbench itself does not
    perform any filesystem discovery.

    Attributes
    ----------
    decoder_name : str
        Display name used as the ``decoder_name`` column of the result CSV
        and as the legend label in plots. Typically something like
        ``"LearnedDMemBP/run_0"``.
    model_cfg : DictConfig
        The ``model`` section of the run's training config (OmegaConf).
    ckpt_path : Path
        Path to the Lightning checkpoint (e.g. ``best_model.ckpt``).
    csv_path : Path
        Path to the CSV file used to persist / resume this run's results.
    """

    decoder_name: str
    model_cfg: DictConfig
    ckpt_path: Path
    csv_path: Path
