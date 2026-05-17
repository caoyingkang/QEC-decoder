"""High-level orchestrators that drive :func:`collect_stats` across many tasks.

Three entry points:

- :func:`run_baseline_benchmark` — one qecdec baseline decoder, all ``p`` values.
- :func:`run_torchdecoder_benchmark` — one PyTorch run, all ``p`` values, with
  optional Rust transplant (cpu + LearnedDMemBP) and optional relaybp_mode.
- :func:`run_custom_benchmark` — top-level fan-out across baselines + torch
  runs, using a caller-supplied ``{p: Experiment}`` map.
"""

import threading
from pathlib import Path
from typing import Iterable, Optional, Union

import stim
from omegaconf import DictConfig
from qecdec.decoders import (
    DMemBPDecoder,
    ITERATIVE_DECODERS,
    RelayBPDecoder,
    create_decoder,
)
from qecdec.experiments import Experiment

from .collector import collect_stats
from .decoders import PyTorchBenchmarkDecoder, QecdecBenchmarkDecoder
from .io import get_baseline_csv_path
from .params import BenchTaskParams, CollectorParams, QECParams, TorchDecoderTask
from .stats import TaskMetadata
from .torch_loader import (
    load_gamma_from_checkpoint,
    load_prior_from_checkpoint,
    load_torchdecoder,
)


def run_baseline_benchmark(
    decoder_name: str,
    *,
    experiments: dict[float, Experiment],
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    csv_path: Path,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Benchmark one qecdec baseline decoder across all ``p`` in ``benchtask_params.p_list``."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_params = benchtask_params.baseline_decoder_params[decoder_name]
    is_iterative = decoder_name in ITERATIVE_DECODERS

    for p in benchtask_params.p_list:
        if stop_event is not None and stop_event.is_set():
            return
        expmt = experiments[p]
        metadata = TaskMetadata(
            code=qec_params.code,
            noise_model=qec_params.noise_model,
            d=qec_params.d,
            rounds=qec_params.rounds,
            basis=qec_params.basis,
            p=p,
            decoder_name=decoder_name,
            decoder_params=decoder_params,
            is_iterative=is_iterative,
        )
        decoder = QecdecBenchmarkDecoder(
            create_decoder(
                decoder_name,
                pcm=expmt.chkmat,
                prior=expmt.prior,
                **decoder_params,
            ),
            expmt.obsmat,
        )

        collect_stats(
            expmt.dem,
            decoder,
            metadata,
            batch_size=collector_params.batch_size,
            shots_cap=collector_params.shots_cap,
            errors_cap=collector_params.errors_cap,
            num_parallel_workers=collector_params.num_parallel_workers,
            csv_path=csv_path,
            th_stop_event=stop_event,
        )


def run_torchdecoder_benchmark(
    task: TorchDecoderTask,
    *,
    experiments: dict[float, Experiment],
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Benchmark one PyTorch decoder run across all ``p`` in ``benchtask_params.p_list``.

    When ``device == "cpu"`` and ``model_cfg.name == "LearnedDMemBP"``, the
    trained ``gamma`` (and optionally ``prior_llr``) are pulled out of the
    checkpoint and used to build an equivalent Rust :class:`DMemBPDecoder` or
    :class:`RelayBPDecoder` (when ``relaybp_mode`` is set) for inference; this
    is much faster than running the PyTorch model on CPU.
    """
    decoder_params = benchtask_params.torchdecoder_shared_params
    use_prior_in_ckpt = decoder_params["use_prior_in_ckpt"]
    relaybp_mode = decoder_params.get("relaybp_mode", False)
    relaybp_params = decoder_params.get("relaybp", {})
    max_iter = decoder_params["max_iter"] if not relaybp_mode else None

    use_rust_transplant = (
        collector_params.device == "cpu" and task.model_cfg.name == "LearnedDMemBP"
    )
    if relaybp_mode and not use_rust_transplant:
        raise ValueError(
            "relaybp_mode=True requires device='cpu' and model 'LearnedDMemBP'. "
            f"Got device={collector_params.device!r}, model={task.model_cfg.name!r}."
        )
    decoder_name = (
        f"{task.decoder_name} (RelayBP)" if relaybp_mode else task.decoder_name
    )

    ckpt_gamma = (
        load_gamma_from_checkpoint(task.ckpt_path) if use_rust_transplant else None
    )
    ckpt_prior = (
        load_prior_from_checkpoint(task.ckpt_path)
        if use_rust_transplant and use_prior_in_ckpt
        else None
    )

    metadata_list: list[TaskMetadata] = []
    decoder_list: list[Union[PyTorchBenchmarkDecoder, QecdecBenchmarkDecoder]] = []
    dem_list: list[stim.DetectorErrorModel] = []
    for p in benchtask_params.p_list:
        expmt = experiments[p]
        metadata_list.append(
            TaskMetadata(
                code=qec_params.code,
                noise_model=qec_params.noise_model,
                d=qec_params.d,
                rounds=qec_params.rounds,
                basis=qec_params.basis,
                p=p,
                decoder_name=decoder_name,
                decoder_params=decoder_params,
                is_iterative=True,
            )
        )
        if use_rust_transplant:
            prior = ckpt_prior if use_prior_in_ckpt else expmt.prior
            if relaybp_mode:
                relaybp = RelayBPDecoder(
                    pcm=expmt.chkmat,
                    prior=prior,
                    gamma0=ckpt_gamma,
                    gamma_dist_interval=relaybp_params["gamma_dist_interval"],
                    num_relays=relaybp_params["num_relays"],
                    pre_iter=relaybp_params["pre_iter"],
                    max_iter_per_relay=relaybp_params["max_iter_per_relay"],
                    stop_nconv=relaybp_params["stop_nconv"],
                )
                decoder_list.append(QecdecBenchmarkDecoder(relaybp, expmt.obsmat))
            else:
                dmembp = DMemBPDecoder(
                    pcm=expmt.chkmat,
                    prior=prior,
                    gamma=ckpt_gamma,
                    max_iter=max_iter,
                )
                decoder_list.append(QecdecBenchmarkDecoder(dmembp, expmt.obsmat))
        else:
            model = load_torchdecoder(
                chkmat=expmt.chkmat,
                prior=expmt.prior,
                model_cfg=task.model_cfg,
                max_iter=max_iter,
                ckpt_path=task.ckpt_path,
                use_prior_in_ckpt=use_prior_in_ckpt,
            )
            decoder_list.append(
                PyTorchBenchmarkDecoder(
                    model,
                    expmt.obsmat,
                    device=collector_params.device,
                )
            )
        dem_list.append(expmt.dem)

    for metadata, decoder, dem in zip(metadata_list, decoder_list, dem_list):
        if stop_event is not None and stop_event.is_set():
            return
        collect_stats(
            dem,
            decoder,
            metadata,
            batch_size=collector_params.batch_size,
            shots_cap=collector_params.shots_cap,
            errors_cap=collector_params.errors_cap,
            num_parallel_workers=collector_params.num_parallel_workers,
            csv_path=task.csv_path,
            th_stop_event=stop_event,
        )


def run_custom_benchmark(
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    experiments: dict[float, Experiment],
    baseline_csv_dir: Path,
    baseline_decoders: Iterable[str] = (),
    torchdecoder_tasks: Iterable[TorchDecoderTask] = (),
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run a multi-decoder Monte Carlo benchmark synchronously.

    Iterates over the requested baseline decoders and PyTorch tasks, invoking
    the appropriate per-decoder runner for each. Results are appended to CSVs
    under ``baseline_csv_dir`` (via :func:`get_baseline_csv_path`) for
    baselines and to ``task.csv_path`` for each :class:`TorchDecoderTask`;
    existing CSVs are resumed in place.

    Parameters
    ----------
    qec_params, benchtask_params, collector_params
        QEC code spec, decoder configs + p_list, and Monte Carlo collector
        knobs (batch size, shot/error caps, parallelism).
    experiments
        Pre-built ``{p: Experiment}`` map covering every ``p`` in
        ``benchtask_params.p_list``. qecbench does not construct experiments
        itself — callers wire this in (see ``benchmark-app/experiment_factory.py``
        for an example).
    baseline_csv_dir
        Root directory for baseline CSV outputs.
    baseline_decoders
        Names of qecdec baseline decoders to run (e.g. ``"BP"``, ``"MWPM"``).
        Each must have a matching entry in
        ``benchtask_params.baseline_decoder_params``.
    torchdecoder_tasks
        PyTorch decoder runs to benchmark. Each :class:`TorchDecoderTask`
        bundles the display name, model config, checkpoint path, and result
        CSV path for one run.
    stop_event
        Optional :class:`threading.Event` for cooperative cancellation.
        Checked between decoders and inside :func:`collect_stats` between
        batches.
    """
    for decoder_name in baseline_decoders:
        if stop_event is not None and stop_event.is_set():
            return
        run_baseline_benchmark(
            decoder_name,
            experiments=experiments,
            qec_params=qec_params,
            benchtask_params=benchtask_params,
            collector_params=collector_params,
            csv_path=get_baseline_csv_path(
                baseline_csv_dir, qec_params, decoder_name
            ),
            stop_event=stop_event,
        )

    for task in torchdecoder_tasks:
        if stop_event is not None and stop_event.is_set():
            return
        run_torchdecoder_benchmark(
            task,
            experiments=experiments,
            qec_params=qec_params,
            benchtask_params=benchtask_params,
            collector_params=collector_params,
            stop_event=stop_event,
        )
