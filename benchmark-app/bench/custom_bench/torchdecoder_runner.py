"""Unified benchmark for PyTorch decoders (LearnedDMemBP, MultiDMemBP, etc.)."""

import threading
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
import stim
from qecdec.experiments import Experiment, RotatedSurfaceCode_Memory

from ..params import QECParams, BenchTaskParams
from ..torchdecoder_loader import load_torchdecoder
from .collector import collect_stats
from .collector_params import CollectorParams
from .decoder import PyTorchBenchmarkDecoder
from .stats import TaskMetadata


def run_torchdecoder_benchmark(
    csv_path: Path,
    decoder_name: str,
    model_cfg: DictConfig,
    ckpt_path: Path,
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    stop_event: Optional[threading.Event] = None,
    experiments: Optional[dict[float, Experiment]] = None,
) -> None:
    decoder_params = benchtask_params.torchdecoder_shared_params
    max_iter = decoder_params["max_iter"]
    use_prior_in_ckpt = decoder_params["use_prior_in_ckpt"]

    metadata_list: list[TaskMetadata] = []
    decoder_list: list[PyTorchBenchmarkDecoder] = []
    dem_list: list[stim.DetectorErrorModel] = []
    for p in benchtask_params.p_list:
        if experiments is not None:
            expmt = experiments[p]
        else:
            expmt = RotatedSurfaceCode_Memory(
                d=qec_params.d,
                rounds=qec_params.rounds,
                basis=qec_params.basis,
                data_qubit_error_rate=p,
                meas_error_rate=p,
            )
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
        model = load_torchdecoder(
            chkmat=expmt.chkmat,
            prior=expmt.prior,
            model_cfg=model_cfg,
            max_iter=max_iter,
            ckpt_path=ckpt_path,
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
            csv_path=csv_path,
            th_stop_event=stop_event,
        )
