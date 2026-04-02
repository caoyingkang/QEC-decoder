from typing import Any, NamedTuple


class BenchTaskParams(NamedTuple):
    """Parameters for the benchmark task.

    Attributes
    ----------
    p_list : list[float]
        List of physical error rates to benchmark at.
    baseline_decoder_params : dict[str, dict]
        Maps each selected baseline decoder name to its config dict. Example:
        ``{"BP": {"max_iter": 50}, "MWPM": {}, "RelayBP": {"gamma0": 0.0, ...}}``
    torchdecoder_shared_params : dict[str, Any]
        Shared parameters for the PyTorch decoder(s). Example:
        ``{"use_prior_in_ckpt": True, "max_iter": 50}``
    """

    p_list: list[float]
    baseline_decoder_params: dict[str, dict]
    torchdecoder_shared_params: dict[str, Any]


class QECParams(NamedTuple):
    """Parameters for the QEC experiment."""

    code: str
    noise_model: str
    d: int
    rounds: int
    basis: str
