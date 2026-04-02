"""Unified benchmark for PyTorch decoders (LearnedDMemBP, MultiDMemBP, etc.)."""

from pathlib import Path

from omegaconf import DictConfig
import sinter
from qecdec.experiments import RotatedSurfaceCode_Memory

from ..params import QECParams, BenchTaskParams
from ..torchdecoder_loader import load_torchdecoder
from .collector_params import CollectorParams
from .decoder import PyTorchSinterDecoder


def run_torchdecoder_benchmark(
    csv_path: Path,
    decoder_name: str,
    model_cfg: DictConfig,
    ckpt_path: Path,
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
):
    decoder_params = benchtask_params.torchdecoder_shared_params
    max_iter = decoder_params["max_iter"]
    use_prior_in_ckpt = decoder_params["use_prior_in_ckpt"]

    tasks: list[sinter.Task] = []
    custom_decoders: dict[str, sinter.Decoder] = {}
    for p in benchtask_params.p_list:
        expmt = RotatedSurfaceCode_Memory(
            d=qec_params.d,
            rounds=qec_params.rounds,
            basis=qec_params.basis,
            data_qubit_error_rate=p,
            meas_error_rate=p,
        )
        model = load_torchdecoder(
            chkmat=expmt.chkmat,
            prior=expmt.prior,
            model_cfg=model_cfg,
            max_iter=max_iter,
            ckpt_path=ckpt_path,
            use_prior_in_ckpt=use_prior_in_ckpt,
        )
        custom_decoder_id = f"custom_decoder_{len(custom_decoders)}"
        custom_decoders[custom_decoder_id] = PyTorchSinterDecoder(
            model,
            expmt.obsmat,
            device=collector_params.device,
        )
        tasks.append(
            sinter.Task(
                circuit=expmt.circuit,
                detector_error_model=expmt.dem,
                decoder=custom_decoder_id,
                json_metadata={
                    "code": qec_params.code,
                    "noise_model": qec_params.noise_model,
                    "d": qec_params.d,
                    "rounds": qec_params.rounds,
                    "basis": qec_params.basis,
                    "p": p,
                    "decoder_name": decoder_name,
                    "decoder_params": decoder_params,
                },
            )
        )

    sinter.collect(
        num_workers=collector_params.num_workers,
        tasks=tasks,
        save_resume_filepath=csv_path,
        max_shots=collector_params.shots_cap,
        max_errors=collector_params.errors_cap,
        print_progress=True,
        custom_decoders=custom_decoders,
    )
