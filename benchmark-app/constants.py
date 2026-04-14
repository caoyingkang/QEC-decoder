from pathlib import Path

from bench.params import QECParams

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_APP_ROOT = REPO_ROOT / "benchmark-app"
BASELINES_CSV_DIR = BENCHMARK_APP_ROOT / "baselines-results"
TORCH_RUNS_ROOT = REPO_ROOT / "torchdecoder" / "runs"
CIRCUITS_ROOT = REPO_ROOT / "circuits"

DEFAULT_P_LIST = [0.002, 0.003, 0.004, 0.005]  # [0.004, 0.006, 0.008, 0.01]

DEFAULT_BATCH_SIZE = 32
DEFAULT_SHOTS_CAP = 100_000_000
DEFAULT_ERRORS_CAP = 100

DEFAULT_BP_MAX_ITER = 50
DEFAULT_PYTORCH_MAX_ITER = DEFAULT_BP_MAX_ITER

DEFAULT_MEMBP_MAX_ITER = DEFAULT_BP_MAX_ITER
DEFAULT_MEMBP_GAMMA: dict[QECParams, float] = {
    QECParams(
        code="RotatedSurfaceCode",
        noise_model="Phenomenological",
        d=9,
        rounds=9,
        basis="Z",
    ): 0.1,
    QECParams(
        code="RotatedSurfaceCode",
        noise_model="CircuitLevel",
        d=11,
        rounds=11,
        basis="Z",
    ): 0.55,
    QECParams(
        code="RotatedSurfaceCode",
        noise_model="CircuitLevel",
        d=11,
        rounds=11,
        basis="X",
    ): 0.55,
    QECParams(
        code="BB_144_12_12",
        noise_model="CircuitLevel",
        d=12,
        rounds=12,
        basis="Z",
    ): 0.2,
    QECParams(
        code="BB_144_12_12",
        noise_model="CircuitLevel",
        d=12,
        rounds=12,
        basis="X",
    ): 0.2,
    QECParams(
        code="BB_288_12_18",
        noise_model="CircuitLevel",
        d=18,
        rounds=18,
        basis="Z",
    ): 0.25,
    QECParams(
        code="BB_288_12_18",
        noise_model="CircuitLevel",
        d=18,
        rounds=18,
        basis="X",
    ): 0.25,
}

DEFAULT_RELAYBP_PRE_ITER = DEFAULT_BP_MAX_ITER
DEFAULT_RELAYBP_MAX_ITER_PER_RELAY = DEFAULT_BP_MAX_ITER
DEFAULT_RELAYBP_NUM_RELAYS = 16
DEFAULT_RELAYBP_STOP_NCONV = 1
DEFAULT_RELAYBP_GAMMA0 = DEFAULT_MEMBP_GAMMA
DEFAULT_RELAYBP_GDI: dict[QECParams, tuple[float, float]] = {
    QECParams(
        code="RotatedSurfaceCode",
        noise_model="Phenomenological",
        d=9,
        rounds=9,
        basis="Z",
    ): (-0.12862588829501195, 0.6717325722483108),
    QECParams(
        code="RotatedSurfaceCode",
        noise_model="CircuitLevel",
        d=11,
        rounds=11,
        basis="Z",
    ): (-0.2526669073577039, 0.9575754453501071),
    QECParams(
        code="RotatedSurfaceCode",
        noise_model="CircuitLevel",
        d=11,
        rounds=11,
        basis="X",
    ): (-0.2526669073577039, 0.9575754453501071),
    QECParams(
        code="BB_144_12_12",
        noise_model="CircuitLevel",
        d=12,
        rounds=12,
        basis="Z",
    ): (-0.19806670996164882, 0.6200937872049607),
    QECParams(
        code="BB_144_12_12",
        noise_model="CircuitLevel",
        d=12,
        rounds=12,
        basis="X",
    ): (-0.19806670996164882, 0.6200937872049607),
    QECParams(
        code="BB_288_12_18",
        noise_model="CircuitLevel",
        d=18,
        rounds=18,
        basis="Z",
    ): (-0.15590331822127088, 0.753307573477873),
    QECParams(
        code="BB_288_12_18",
        noise_model="CircuitLevel",
        d=18,
        rounds=18,
        basis="X",
    ): (-0.15590331822127088, 0.753307573477873),
}

DEFAULT_BPOSD_MAX_ITER = DEFAULT_BP_MAX_ITER
DEFAULT_BPOSD_OSD_ORDER = 10
