from pathlib import Path

from qecbench import QECParams

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_APP_ROOT = REPO_ROOT / "benchmark-app"
BASELINES_CSV_DIR = BENCHMARK_APP_ROOT / "baselines-results"
TORCH_RUNS_ROOT = REPO_ROOT / "torchdecoder" / "runs"
CIRCUITS_ROOT = REPO_ROOT / "circuits"

DEFAULT_P_LIST = [0.001, 0.002, 0.003, 0.004, 0.005]  # [0.004, 0.006, 0.008, 0.01]

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
        noise_model="Phenomenological",
        d=11,
        rounds=11,
        basis="Z",
    ): 0.55,
    QECParams(
        code="RotatedSurfaceCode",
        noise_model="Phenomenological",
        d=11,
        rounds=11,
        basis="X",
    ): 0.55,
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
    QECParams(
        code="HexColorCode",
        noise_model="Phenomenological",
        d=11,
        rounds=11,
        basis="Z",
    ): 0.55,
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
        noise_model="Phenomenological",
        d=11,
        rounds=11,
        basis="Z",
    ): (-0.2249502476717961, 0.7537800632074495),
    QECParams(
        code="RotatedSurfaceCode",
        noise_model="Phenomenological",
        d=11,
        rounds=11,
        basis="X",
    ): (-0.2249502476717961, 0.7537800632074495),
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
    QECParams(
        code="HexColorCode",
        noise_model="Phenomenological",
        d=11,
        rounds=11,
        basis="Z",
    ): (-0.21295457674166532, 0.697918233656232),
}

DEFAULT_BPOSD_MAX_ITER = DEFAULT_BP_MAX_ITER
DEFAULT_BPOSD_OSD_ORDER = 10

DEFAULT_ENS_SERIAL_BP_MAX_ITER = 20
DEFAULT_ENS_SERIAL_BP_ENSEMBLE_SIZE = 32
DEFAULT_ENS_SERIAL_BP_TOPK = 5
DEFAULT_ENS_SERIAL_BP_SEED = 42

CODE_NOISE_PAIR_TO_D_ROUNDS_BASIS_TRIPLES: dict[
    tuple[str, str], list[tuple[int, int, str]]
] = {
    ("RotatedSurfaceCode", "Phenomenological"): [
        (5, 5, "Z"),
        (5, 5, "X"),
        (7, 7, "Z"),
        (7, 7, "X"),
        (9, 9, "Z"),
        (9, 9, "X"),
        (11, 11, "Z"),
        (11, 11, "X"),
    ],
    ("RotatedSurfaceCode", "CircuitLevel"): [
        (5, 5, "Z"),
        (5, 5, "X"),
        (7, 7, "Z"),
        (7, 7, "X"),
        (9, 9, "Z"),
        (9, 9, "X"),
        (11, 11, "Z"),
        (11, 11, "X"),
    ],
    ("BB_18_4_3", "CircuitLevel"): [
        (3, 3, "Z"),
        (3, 3, "X"),
    ],
    ("BB_72_12_6", "CircuitLevel"): [
        (6, 6, "Z"),
        (6, 6, "X"),
    ],
    ("BB_144_12_12", "CircuitLevel"): [
        (12, 12, "Z"),
        (12, 12, "X"),
    ],
    ("BB_288_12_18", "CircuitLevel"): [
        (18, 18, "Z"),
        (18, 18, "X"),
    ],
    ("HexColorCode", "Phenomenological"): [
        (11, 11, "Z"),
    ],
    ("HexColorCode", "Superdense"): [
        (5, 5, "Z"),
        (5, 5, "X"),
        (7, 7, "Z"),
        (7, 7, "X"),
        (9, 9, "Z"),
        (9, 9, "X"),
        (11, 11, "Z"),
        (11, 11, "X"),
    ],
}

ALL_CODE_NOISE_PAIRS = list(CODE_NOISE_PAIR_TO_D_ROUNDS_BASIS_TRIPLES.keys())
