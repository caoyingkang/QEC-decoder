from pathlib import Path

import stim
from qecdec.experiments import RotatedSurfaceCode_Memory
from qecdec.decoders import (
    MWPMDecoder,
    BPDecoder,
)

from ..params import QECParams, BenchTaskParams
from .stats import TaskMetadata
from .collector import collect_stats
from .collector_params import CollectorParams
from .stats_io import get_baseline_csv_path
from .decoder import QecdecBenchmarkDecoder


def run_MWPM_benchmark(
    baseline_csv_dir: Path,
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
):
    csv_path = get_baseline_csv_path(baseline_csv_dir, qec_params, "MWPM")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_list: list[TaskMetadata] = []
    decoder_list: list[QecdecBenchmarkDecoder] = []
    dem_list: list[stim.DetectorErrorModel] = []
    for p in benchtask_params.p_list:
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
                decoder="MWPM",
                p=p,
            )
        )
        decoder_list.append(
            QecdecBenchmarkDecoder(
                MWPMDecoder(expmt.chkmat, expmt.prior),
                expmt.obsmat,
            )
        )
        dem_list.append(expmt.dem)

    for metadata, decoder, dem in zip(metadata_list, decoder_list, dem_list):
        collect_stats(
            dem,
            decoder,
            metadata,
            batch_size=collector_params.batch_size,
            shots_cap=collector_params.shots_cap,
            errors_cap=collector_params.errors_cap,
            num_parallel_workers=collector_params.num_parallel_workers,
            csv_path=csv_path,
        )


def run_BP_benchmark(
    baseline_csv_dir: Path,
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
):
    csv_path = get_baseline_csv_path(baseline_csv_dir, qec_params, "BP")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_list: list[TaskMetadata] = []
    decoder_list: list[QecdecBenchmarkDecoder] = []
    dem_list: list[stim.DetectorErrorModel] = []
    for p in benchtask_params.p_list:
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
                decoder="BP",
                p=p,
                max_iter=benchtask_params.max_iter,
            )
        )
        decoder_list.append(
            QecdecBenchmarkDecoder(
                BPDecoder(
                    expmt.chkmat, expmt.prior, max_iter=benchtask_params.max_iter
                ),
                expmt.obsmat,
            )
        )
        dem_list.append(expmt.dem)

    for metadata, decoder, dem in zip(metadata_list, decoder_list, dem_list):
        collect_stats(
            dem,
            decoder,
            metadata,
            batch_size=collector_params.batch_size,
            shots_cap=collector_params.shots_cap,
            errors_cap=collector_params.errors_cap,
            num_parallel_workers=collector_params.num_parallel_workers,
            csv_path=csv_path,
        )
