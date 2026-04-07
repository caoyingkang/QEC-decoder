import threading
from pathlib import Path
from typing import Optional

from qecdec.experiments import Experiment, RotatedSurfaceCode_Memory
from qecdec.decoders import ITERATIVE_DECODERS, create_decoder

from ..params import QECParams, BenchTaskParams
from .stats import TaskMetadata
from .collector import collect_stats
from .collector_params import CollectorParams
from .stats_io import get_baseline_csv_path
from .decoder import QecdecBenchmarkDecoder


def run_baseline_benchmark(
    baseline_csv_dir: Path,
    decoder_name: str,
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    stop_event: Optional[threading.Event] = None,
    experiments: Optional[dict[float, Experiment]] = None,
) -> None:
    csv_path = get_baseline_csv_path(baseline_csv_dir, qec_params, decoder_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    decoder_params = benchtask_params.baseline_decoder_params[decoder_name]

    for p in benchtask_params.p_list:
        if stop_event is not None and stop_event.is_set():
            return
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
        metadata = TaskMetadata(
            code=qec_params.code,
            noise_model=qec_params.noise_model,
            d=qec_params.d,
            rounds=qec_params.rounds,
            basis=qec_params.basis,
            p=p,
            decoder_name=decoder_name,
            decoder_params=decoder_params,
            is_iterative=decoder_name in ITERATIVE_DECODERS,
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
