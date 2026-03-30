from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_APP_ROOT = REPO_ROOT / "benchmark-app"
BASELINES_CSV_DIR = BENCHMARK_APP_ROOT / "baselines-results"
TORCH_RUNS_ROOT = REPO_ROOT / "torchdecoder" / "runs"

DEFAULT_MAX_ITER = 50
DEFAULT_P_LIST = [0.004, 0.006, 0.008, 0.01, 0.012]

DEFAULT_BATCH_SIZE = 1024
DEFAULT_SHOTS_CAP = 100_000_000
DEFAULT_ERRORS_CAP = 100

DEFAULT_BASELINE_DECODERS = ["MWPM", "UnionFind", "BP"]
