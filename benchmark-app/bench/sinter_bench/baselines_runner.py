"""Functions for benchmarking baseline decoders."""

from pathlib import Path
from typing import Optional

import sinter
from qecdec.experiments import Experiment, RotatedSurfaceCode_Memory
from qecdec.decoders import create_decoder
from qecdec.sinter_utils import QecdecSinterDecoder

from ..params import QECParams, BenchTaskParams
from .collector_params import CollectorParams
from .stats_io import get_baseline_csv_path


def run_baseline_benchmark(
    baseline_csv_dir: Path,
    decoder_name: str,
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    experiments: Optional[dict[float, Experiment]] = None,
):
    csv_path = get_baseline_csv_path(baseline_csv_dir, qec_params, decoder_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    decoder_params = benchtask_params.baseline_decoder_params[decoder_name]

    tasks: list[sinter.Task] = []
    custom_decoders: dict[str, sinter.Decoder] = {}
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
        dec = create_decoder(
            decoder_name,
            pcm=expmt.chkmat,
            prior=expmt.prior,
            **decoder_params,
        )
        custom_decoder_id = f"custom_decoder_{len(custom_decoders)}"
        custom_decoders[custom_decoder_id] = QecdecSinterDecoder(dec, expmt.obsmat)
        json_metadata = {
            "code": qec_params.code,
            "noise_model": qec_params.noise_model,
            "d": qec_params.d,
            "rounds": qec_params.rounds,
            "basis": qec_params.basis,
            "p": p,
            "decoder_name": decoder_name,
            "decoder_params": decoder_params,
        }
        tasks.append(
            sinter.Task(
                circuit=expmt.circuit,
                detector_error_model=expmt.dem,
                decoder=custom_decoder_id,
                json_metadata=json_metadata,
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
